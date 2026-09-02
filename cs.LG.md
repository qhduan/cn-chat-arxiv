# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation](https://arxiv.org/abs/2609.01604) | 该论文通过八种攻击扰动分类法与因果追踪、注意力头敲除等可解释性技术，首次从机制层面揭示LLM评估器（Themis与Prometheus）在摘要评分时采用两阶段内部流程：第15层以下注意力执行局部错误比较并路由信号，其上由MLP级联完成信号整合。 |
| [^2] | [Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation](https://arxiv.org/abs/2609.01596) | Facet-0通过联合建模“动作-力旋量”来预测并评估动作的接触后果，结合多模态表征学习与强化学习后训练，实现了亚毫米级公差的接触密集型精细装配操作。 |
| [^3] | [The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally](https://arxiv.org/abs/2609.01587) | 该研究通过因果混合精度干预实验发现，LLM的量化损伤是弥散分布的而非集中于特定电路、计算位置或权重统计，因此在匹配精度预算下，将额外比特全局用于更精细的量化粒度比局部修复少数层更有效。 |
| [^4] | [Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs](https://arxiv.org/abs/2609.01573) | 该论文提出“近最优区域”框架来分配SFT-RL标注预算，发现该区域宽广且随模型规模增大而扩大，并能从小型代理模型可靠迁移到大型目标模型，因此小规模代理实验即可替代在大模型上的穷尽式预算搜索。 |
| [^5] | [Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers](https://arxiv.org/abs/2609.01567) | 该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。 |
| [^6] | [Gradient-Update Mismatch: Rethinking Conflict-Free Training of Physics-Informed Neural Networks](https://arxiv.org/abs/2609.01558) | 该论文揭示了“梯度更新失配”问题：梯度手术构造的无冲突方向经过现代优化器（如历史状态、自适应缩放、预条件化、解耦权重衰减等机制）的变换后可能失去无冲突性质，从而为物理信息神经网络的无冲突训练提供了新的认识视角。 |
| [^7] | [Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories](https://arxiv.org/abs/2609.01556) | 该研究在竞赛数学与具身智能体轨迹两个领域以统一协议评测刻意分离表层形式与语义结构的嵌入检索，发现主流嵌入模型存在严重的表层形式（字面词汇）偏差：在结构相同但措辞伪装最重的任务上Hit@1跌至0.0%，未命中时胜出者几乎总是与查询词汇更相似的条目，表明当前嵌入检索锚定于字面文本而非深层结构。 |
| [^8] | [Can LLMs Discover Scientific Laws in Real and Parallel Worlds?](https://arxiv.org/abs/2609.01552) | 该论文提出了基于已发表研究和真实科学数据构建的科学定律发现基准SCILAWS-BENCH（涵盖118个问题、291个候选定律、约800万真实数据点和六个学科），并采用真实世界与平行世界两种互补设置，以严格评估大语言模型能否真正发现科学定律。 |
| [^9] | [A Mathematical Theory of Reusable Neural Bases for Network Compression](https://arxiv.org/abs/2609.01550) | 该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。 |
| [^10] | [Quantum Sparse Autoencoders for Q-Matrix Estimation in Cognitive Diagnosis](https://arxiv.org/abs/2609.01537) | 该论文首次将量子机器学习引入认知诊断领域，提出量子稀疏自编码器（QSAE），通过量子电路压缩学生作答数据以估计Q矩阵，并在模拟和真实评估数据集上展现出与经典自编码器互补的性能优势。 |
| [^11] | [LatentPress: Context Compression Beyond Text and Vision](https://arxiv.org/abs/2609.01507) | LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。 |
| [^12] | [Optimizing Byzantine Node Placement in Decentralized Federated Learning](https://arxiv.org/abs/2609.01495) | 该论文首次将拜占庭节点的网络位置作为显式攻击决策进行研究，提出基于真实 gossip 传播动态的集合级度量 BPI 来量化诚实节点的累积暴露程度，从而在固定攻陷预算下找到对去中心化联邦学习影响最大的拜占庭节点放置方案。 |
| [^13] | [Rethinking Learnability in Offline Data-driven Optimization](https://arxiv.org/abs/2609.01493) | 本文针对PAC可学习性无法充分刻画离线优化的理论缺陷，提出了“算法依赖的可学习性”这一新概念，其只需保证在优化器轨迹上的精度即可支撑离线数据驱动优化。 |
| [^14] | [Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds](https://arxiv.org/abs/2609.01453) | 该研究在接触密集型ParcelStow任务中系统比较了脚本专家与模仿学习策略（ACT）在不同执行速度下的表现，发现尽管两者在标称速度下均达到100%成功率，但在最大加速时专家成功率为84%而ACT仅为53%，表明模仿学习无法完全保留专家的时间鲁棒性。 |
| [^15] | [Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning](https://arxiv.org/abs/2609.01449) | 该论文将扩散去噪器改造为带持久隐藏状态且无时间步条件的通用迭代更新，构建出可随推理深度不断提升准确率的随时求解器，并通过推理时持续注入高斯噪声实现单轨迹高效探索解空间，在极限数独上达到99.90%的精确求解率。 |
| [^16] | [Edge-Girth as a Structural Edge Feature for Graph Neural Networks](https://arxiv.org/abs/2609.01441) | 提出以“边围长”（经过一条边的最短环长度及其数目）作为逐边结构特征，无需预先指定所统计子结构的大小即可捕捉任意长度的环，并将其融入门控消息传递网络EGAGNN以增强图神经网络超越1-WL的表达能力。 |
| [^17] | [Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search](https://arxiv.org/abs/2609.01431) | 本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。 |
| [^18] | [Learning Sparse Decision Trees via Transformer Variational Auto-Encoders](https://arxiv.org/abs/2609.01430) | TREVIS通过树Transformer变分自编码器的潜空间探索，将决策树的离散搜索转化为连续空间中基于梯度的优化，从而学习同时兼顾预测性能和结构稀疏性的决策树。 |
| [^19] | [TRIAGE: Three-level Routing and Intelligent Agent Guidance for Efficient Execution](https://arxiv.org/abs/2609.01428) | TRIAGE提出三级路由框架，其核心创新TaaS（轨迹即技能）将历史执行轨迹抽象为可复用技能，使相同和相似查询实现零token消耗，显著降低LLM智能体的执行成本。 |
| [^20] | [Semantic-Guided Multimodal Preprocessing for Vision Transformer-Based Clear Cell Renal Cell Carcinoma Grading](https://arxiv.org/abs/2609.01426) | 提出一种语义引导的多模态预处理方法，将细胞核分类图与RGB病理图像融合后输入视觉Transformer进行透明细胞肾细胞癌分级，将平衡准确率从0.707提升至0.916。 |
| [^21] | [CATeye: Coupled Attribute-Topology Invariance Learning for Voucher Abuse Detection](https://arxiv.org/abs/2609.01425) | 提出CATeye框架，通过属性不变性选择器和边不变性选择器应对优惠券滥用检测中耦合的属性-拓扑分布偏移问题，无需频繁重训练即可抵御欺诈模式的快速演变。 |
| [^22] | [Provably Safe Sim-to-Real Transfer](https://arxiv.org/abs/2609.01418) | 该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。 |
| [^23] | [Predicting Subsurface Abnormalities Growth using Physics-Informed Neural Networks](https://arxiv.org/abs/2609.01417) | 该研究首次将物理信息神经网络（PINNs）应用于探地雷达数据预测，通过将电磁波传播物理定律嵌入由CNN、空间特征通道注意力和ConvLSTM组成的深度学习架构中，实现对地下异常体生长的精确预测。 |
| [^24] | [Contribution-Aware Bandwidth Allocation for Multimodal Split Learning](https://arxiv.org/abs/2609.01406) | 该论文提出 ModalShare，一种基于 Shapley 贡献度为各模态动态设置压缩保留率的带宽分配器，解决了多模态分割学习中传统均匀压缩策略忽略各模态对融合预测贡献差异的问题。 |
| [^25] | [Measuring consistency via ensemble margin and local prediction variability: Auditing decision systems in the presence of predictive multiplicity](https://arxiv.org/abs/2609.01397) | 该论文提出一种将集成边界与局部预测变异性相结合的一致性准则，用于在罗生门效应（预测多样性）存在下审计决策系统，并证明在温和假设下有限集成的一致性分数会收敛于罗生门集合中期望模型的一致性分数。 |
| [^26] | [Investigating Linear Probe Robustness to Linguistic Register, Medical Specialty, and Corpus Shifts in Medical QA](https://arxiv.org/abs/2609.01361) | 该论文构建了一个可独立操控写作语域、医学专科和语料库三类变化的医学问答基准，以系统性探究大语言模型中线性探针（真值方向检测）对不同输入偏移的鲁棒性。 |
| [^27] | [Exact Risk-Complexity Laws for Projective Boundaries in Scenario Optimization and Distribution-Free Certification](https://arxiv.org/abs/2609.01355) | 本文揭示了情景优化、共形预测等无分布认证方法中违反风险精确贝塔定律背后的确定性投影边界机制，并提出了“真投影边界方案”框架，将该定律推广到边界大小为随机的更一般情形。 |
| [^28] | [Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR](https://arxiv.org/abs/2609.01354) | 该论文将变异测试从模型转向验证器本身，通过构造保证数学等价的答案变体，在超过30万个判定上对四个主流验证器进行了类别级审计，发现相同输入下验证器的自我验证率相差高达41.3个百分点，揭示了RLVR奖励信号中的系统性假阴性问题。 |
| [^29] | [Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades](https://arxiv.org/abs/2609.01345) | 该研究通过真实LLM实验发现，推理级联中廉价验证器对学生模型错误答案的“盲区”随学生能力增强而扩大、随验证器能力增强而缩小，恰好在级联机制赖以存在的低成本配置下最为严重，而用前沿验证器消除盲区又会因过度升级而抵消成本节约，从而揭示了成本节约级联设计背后隐藏的显著可靠性代价。 |
| [^30] | [mzCache: On-Device LLM Memory Management under Multitasking](https://arxiv.org/abs/2609.01338) | mzCache针对移动设备多任务环境下的不可预测内存压力，提出面向恢复的内存管理机制，弹性驱逐LLM内存并利用移动SoC统一内存在CPU端并发恢复的同时实现GPU上的零等待推理。 |
| [^31] | [Bandits in Prod: Hyperparameter Optimization at Inference Time](https://arxiv.org/abs/2609.01335) | 该论文将生产系统中只能通过线上噪声反馈评估配置的场景形式化为在线超参数优化（OHPO），提出通用框架IMABO及免重启的无限多臂老虎机策略IMOSS，并给出了分位数遗憾的理论保证。 |
| [^32] | [Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment](https://arxiv.org/abs/2609.01322) | 该论文提出一种基于稀疏自编码器（SAE）的新颖因果调整流程，通过条件独立性检验迭代选取最小特征集合，解决了文本表示在保留混杂变量与满足有限样本重叠条件之间的权衡，在半合成评估中实现了比替代表示更低的偏差和更高的覆盖率。 |
| [^33] | [MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval](https://arxiv.org/abs/2609.01316) | MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。 |
| [^34] | [One-Layer Transformer Provably Learns Multiclass One-Nearest Neighbor in Context](https://arxiv.org/abs/2609.01311) | 本文证明了带argmax分类头的单层Transformer在多分类的上下文学习中行为与单最近邻分类器完全一致，填补了此前工作依赖非标准舍入方法所留下的理论空白。 |
| [^35] | [GazeRefine: Expert Gaze as a Test-Time Prompt for Training-Free Medical Image Segmentation](https://arxiv.org/abs/2609.01310) | GazeRefine提出了一种免训练的零样本医学图像分割框架，将专家眼动注视转化为前景/背景先验，在冻结的DINOv3特征空间中初始化并迭代细化语义原型，无需任何分割掩码、微调或梯度更新。 |
| [^36] | [Relational Task Generation Language: A Declarative Specification Framework for Relational Deep Learning](https://arxiv.org/abs/2609.01292) | 提出了一种开源声明式语言RTGL，通过抽象SQL细节来简化关系深度学习预测任务的定义，避免了手动定义导致的数据泄露问题，并能与现有RDL框架无缝集成。 |
| [^37] | [The Constitutional Coverage Trilemma in AI Governance](https://arxiv.org/abs/2609.01275) | 该研究通过审计23个前沿大模型的默认“宪法”并调查1,649人的价值权衡偏好，发现人类的价值需求广泛多样，而AI模型隐含的价值排序供给既狭窄又随时间漂移，导致近四成用户找不到符合自身价值偏好的模型，揭示了AI治理中的“宪法覆盖三难困境”。 |
| [^38] | [Position: Privacy Is a Claim, Not a Property of Synthetic Data](https://arxiv.org/abs/2609.01273) | 这篇立场论文指出，合成数据的隐私保障不应被当作数据本身的固有属性，而必须基于明确的威胁模型和可证伪的隐私声明，呼吁社区重新确立严格的隐私证据标准。 |
| [^39] | [Solving In-Table Prediction Problems by Deep Neural Networks with Performance Evaluation Using Synthetic Data](https://arxiv.org/abs/2609.01262) | 本文提出了“表内预测”（In-Table Prediction, ITB）这一新问题，通过自监督深度神经网络随机掩码表格中的列并以其为学习目标，利用其余已知列预测任意选定列的值，并识别出三种此前未被广泛研究的应用场景。 |
| [^40] | [Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents](https://arxiv.org/abs/2609.01245) | 本文提出CANOPY方法，论证仅结果奖励的强化学习足以训练小规模开源LLM智能体完成长时程交互任务，所谓瓶颈实为探索不足导致的信号饥饿与缺乏锚定导致的策略漂移这两个常见实践问题的产物。 |
| [^41] | [Post-Training Science for Supervised Fine-Tuning](https://arxiv.org/abs/2609.01244) | 本文通过每次只改变一个变量的统一受控扫描实验，系统测量了监督微调中学习率、批大小、LoRA与全量微调等关键决策在Qwen3与Llama两类模型（稠密与混合专家架构）以及四个真实客户数据集上的表现，将SFT超参数选择从经验摸索转变为可复现的科学测量。 |
| [^42] | [From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs](https://arxiv.org/abs/2609.01240) | 提出推荐原生的Transformer扩展框架ReST，通过双门控注意力编码器应对噪声化行为序列，通过重量级可复用编码器加轻量级交叉解码器的分解设计及共享前缀训练与服务机制，解决推荐排序中的计算不对称问题，实现工业推荐排序的高效规模化。 |
| [^43] | [REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs](https://arxiv.org/abs/2609.01215) | REFACTOR-VLA 提出了一种清醒/睡眠两阶段框架，通过基于潜在世界模型 rollout 计算的行为等价核对运动程序片段进行无监督聚类，并生成 Hindley–Milner 风格的类型化 lambda 项来构建可复用技能库，从而提升 VLA 模型在长时程任务上的性能与可解释性。 |
| [^44] | [Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey](https://arxiv.org/abs/2609.01212) | 本文通过系统性文献综述，梳理了在FPGA平台上部署Transformer模型推理的最新进展、趋势与设计选择，凸显了FPGA相比CPU/GPU在能效、延迟和现场部署灵活性方面的优势。 |
| [^45] | [CopyShield: A Cross-Level Benchmark of Copyright Defenses in LLMs](https://arxiv.org/abs/2609.01161) | CopyShield提出了一个跨干预层级（输出、行为、表示层）的版权防御基准，通过统一协议系统比较对比解码、DPO和激活干预三种方法，揭示了不同干预层级在大语言模型合规性、效用与退化行为之间的独特权衡。 |
| [^46] | [Superposed Latent Autoencoder](https://arxiv.org/abs/2609.01158) | SLAE 通过学习到的叠加机制将多个高容量潜码绑定并叠加存储在单个内存张量中，以可抑制的结构化干涉取代传统自编码器不可逆的维度瓶颈，在相同存储预算下将重建误差最多降低 56%。 |
| [^47] | [Scaled Idempotence in Transformer Attention: Paired OV Geometry and Shared-Value Algebras](https://arxiv.org/abs/2609.01129) | 该论文发现Transformer注意力中存在一类稀疏的“缩放幂等”注意力头，其OV算子满足T²≈αT，并通过精确主坐标分解与K方向置乱实验证明这种代数闭合性质主要由K因子的训练方向所驱动。 |
| [^48] | [When Does Online Adaptation Pay on the Edge? A Leakage-Free Evaluation of Warmup, Learning-Rate Selection, and Resource Trade-offs for Time-Series Forecasting](https://arxiv.org/abs/2609.01126) | 该研究在无泄漏流式评估协议下揭示了测量在线自适应收益时的两个关键偏差来源——基线预热预算的双向效应以及优化器比较中学习率未公平调整的问题，并提出利用漂移前验证切片来公平选择预热预算和在线学习率。 |
| [^49] | [Replicating TRACE: A Practitioner's Guide to Its Threshold and Particle Budget](https://arxiv.org/abs/2609.01108) | 本研究独立复现了TRACE的核心结果，并揭示其最优阈值并非固定常数，而是锚定于定义真值的边界（约delta/2乘以估计器校准系数），且在单一全局阈值下TRACE主要恢复的是直接相邻影响的因果图。 |
| [^50] | [Neural Symbollic Regression Using Deep Learning and Sparse Modelling](https://arxiv.org/abs/2609.01102) | 该论文提出一种将神经网络作为功能性预处理器的神经符号回归框架，先由神经网络学习目标函数的平滑抗噪近似，再用LASSO提取稀疏可解释的闭式数学表达式，从而兼顾深度学习的鲁棒性与符号回归的可解释性。 |
| [^51] | [Subliminal Learning as Trait-Direction Drift: A Mechanism and Targeted Control under SFT Distillation](https://arxiv.org/abs/2609.01091) | 本文提出“特质方向漂移”机制来解释潜意识学习现象——偏置教师数据中可测量的偏好差距在监督微调中累积为学生的行为迁移，并据此提出探测空间走廊正则化这一针对性防御方法，在蒸馏过程中约束模型沿校准特质方向的漂移。 |
| [^52] | [Modelpedia: A Catalog of Model Findings for the Meta-Science of AI](https://arxiv.org/abs/2609.01090) | 提出了Modelpedia——一个利用大语言模型自动从已发表论文中提取AI模型相关发现、将其与模型、数据集、方法和概念关联，并汇总为可搜索公共目录的框架，同时基于该目录对AI社区如何研究模型进行了元分析。 |
| [^53] | [Let Confidence Change, Not the Prediction: Prediction-Preserving Repair for Post-hoc Calibration](https://arxiv.org/abs/2609.01072) | 本文提出CORD——首个通过修复完整校准概率向量来严格保持原始top-1预测不变、仅修正置信度的事后校准后拟合适配器，并引入TPCR指标量化校准器改变预测的频率。 |
| [^54] | [Accelerating Reinforcement Learning via MPC Solver-Gradient Guidance for Weights-varying MPC](https://arxiv.org/abs/2609.01061) | 提出SG-RL方法，利用可微MPC求解器的梯度信息引导强化学习在线自适应地调整MPC代价函数权重，从而在保持低偏差的同时显著提升样本效率、加速学习过程。 |
| [^55] | [SAGE: Subpopulation-Aware Generative Enhancement for Mitigating Spurious Correlations](https://arxiv.org/abs/2609.01051) | 提出SAGE——一种两阶段生成式增强框架，利用聚类得到的子标签微调条件生成模型生成针对性样本，在无需虚假属性先验知识的情况下从数据层面缓解机器学习中的虚假相关性。 |
| [^56] | [From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion](https://arxiv.org/abs/2609.01043) | 提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。 |
| [^57] | [ViTAMINS: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives](https://arxiv.org/abs/2609.01041) | ViTAMINS通过向自监督视觉Transformer的对比学习预训练中引入合成困难负样本，以极小的改动获得了涌现的语义分类能力（最高提升11.3%）并大幅节省计算资源（ViT-B超越ViT-L的V-JEPA），证明对比学习仍是生成式与自蒸馏方法的强大替代方案。 |
| [^58] | [Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees](https://arxiv.org/abs/2609.01035) | 提出渐进式风险授予（PRV）机制，通过托管轨迹级风险预算并在分支激活时逐步扣减，为递归LLM智能体树中不可逆行动的授权证明了任意时刻的危害上界，实现“自由派生、节制行动”的安全权衡。 |
| [^59] | [The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow](https://arxiv.org/abs/2609.01034) | 本文通过将损失函数分解为 $f = g + \varepsilon h$ 的微扰分析，首次为深度学习稳定边缘处梯度下降的中心流提供了系统性推导，并揭示出其中存在快速振荡、中间自稳定与缓慢中心流演化三个时间尺度。 |
| [^60] | [Web Price Extraction: State of the Art and an Adaptive Browserless Implementation](https://arxiv.org/abs/2609.01030) | 该论文系统梳理了网页价格提取四大类方法的优劣权衡，并提出了一种自适应的无浏览器价格提取实现，兼顾速度、成本与跨网站结构的适应性。 |
| [^61] | [SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2609.01004) | 提出无需训练的视觉token剪枝框架SinkPruner，通过过滤高度冗余的高范数离群token并缓解注意力汇聚现象，在保持多模态理解能力的同时实现高效的多模态大语言模型推理。 |
| [^62] | [Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close](https://arxiv.org/abs/2609.00999) | 该论文提出“规范多元性”这一新评估设定，通过将框架选择与框架内正确性分离，揭示了“刻板印象陷阱”——文化线索虽能引导大模型选择伊斯兰金融框架，却在框架内暴露出高达57%至66%的错误率，表明传统二选一评估会严重高估模型的文化对齐能力。 |
| [^63] | [Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches](https://arxiv.org/abs/2609.00946) | 本文提出嵌入式条件独立性检验（eCITs），通过将LLM生成的文本及其源文本嵌入到表示空间后再进行条件独立性检验，从而判断模型输出是否携带源文本之外的额外信息，并将其应用于德国议会演讲数据的分析。 |
| [^64] | [DualStake: Dual-Path Confidence Calibration in Deep Research Agents](https://arxiv.org/abs/2609.00935) | 提出DualStake双路径置信度校准方法，通过在每次检索后引出证据置信度并在答案生成后引出答案置信度，利用边界裁剪的置信度相关stake奖励将两者与答案正确性联合对齐，有效缓解深度研究智能体的严重过度自信问题。 |
| [^65] | [Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO](https://arxiv.org/abs/2609.00925) | 本文通过从同一检查点系统审计GRPO、SFT和DPO共九种后训练方案，发现语言模型遵循冲突提示证据的接地增益主要源于强化模型中已有的机制（与起始模型相同的因果注意力头集合），而非学习新机制，其中GRPO增益很小、冲突SFT提升适中、DPO在其匹配分布上接近上限。 |
| [^66] | [Direct Optimization of a 3D Finite-Source Reflector via Neural-Network Parameterization](https://arxiv.org/abs/2609.00899) | 提出了一种用小型神经网络参数化反射器轮廓、通过可微光线追踪进行端到端训练并结合H^{-1}型谱加权的直接优化方法，能够将有限光源的光转换为预定的远场角度光强分布。 |
| [^67] | [Vision-Language-Guided Pseudo-Labels for Unsupervised Domain Adaptation in Semantic Segmentation for Waste Sorting](https://arxiv.org/abs/2609.00898) | 该论文提出了一种利用SAM、EVA-CLIP和BLIP等视觉-语言基础模型生成跨模态伪标签的流水线，无需任何目标域标注即可实现垃圾分拣语义分割的无监督域自适应。 |
| [^68] | [Poisson-Gamma Dynamical Systems with Time-varying Transition Dynamics](https://arxiv.org/abs/2609.00896) | 本文提出具有时变转移核的泊松-伽马动力系统（TV-PGDS），通过三种专门设计的狄利克雷马尔可夫链建模转移矩阵随时间的演化，并利用数据增广技术实现完全共轭的高效吉布斯采样，从而更好地捕捉现实计数时间序列中的时变转移动态。 |
| [^69] | [Denoising Diffusion Generative Models Secretly Calculate Attentions](https://arxiv.org/abs/2609.00885) | 该论文发现去噪扩散模型本质上暗中使用了与Transformer类似的注意力机制，从而证明注意力是机器学习的普适性原理，并据此提出了基于注意力机制的简化图像生成算法，以减少训练时间和计算开销。 |
| [^70] | [iPINN for Broadband CARS Phase Retrieval: A Framework for Function Approximation and Inverse Modeling Problems in Nonlinear Spectroscopy](https://arxiv.org/abs/2609.00883) | 本文提出一种逆物理信息神经网络（iPINN），利用transformer编码器和可微分解析正演模型从原始BCARS光谱中预测洛伦兹峰参数并重建共振极化率，在非共振背景和噪声多变的采集条件下实现了最优的相位恢复精度。 |
| [^71] | [FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study](https://arxiv.org/abs/2609.00875) | 该论文提出一种基于FractalNet的异构联邦学习方法，通过分布式路径调度器根据卫星SWAP-C约束与预测星间通信机会动态分配模型深度，并结合周期性更新汇聚和三层智能体控制平面，实现了卫星巨型星座中适应异构硬件条件的轨道边缘智能。 |
| [^72] | [Sharp Mixed Spectral Barron Regularity of Coulombic Many-Electron Wave Functions](https://arxiv.org/abs/2609.00872) | 该论文为分子库仑哈密顿量的本征函数建立了尖锐的混合谱Barron正则性，给出了各向同性阶数 $s$ 与坐标阶数 $\alpha,\beta$ 的最优显式容许区域，从而刻画了传统各向同性Barron尺度无法检测的正则性。 |
| [^73] | [The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence](https://arxiv.org/abs/2609.00868) | 该论文发现“视觉不敏感性差距”现象——在40%–97%的多模态基准样本上，模糊与问题相关的关键视觉区域几乎不改变VLM的输出，并证明这种不敏感性是样本层面的属性（跨模型VSI排名显著相关），即使各模型的视觉编码器本身实际上能够检测到这些扰动。 |
| [^74] | [MemoryWalker: Stop Training Agents on Contexts They Never Saw](https://arxiv.org/abs/2609.00865) | 该论文针对上下文压缩导致智能体训练时有效历史呈树状分支的问题，提出了两种梯度等价的精确修正方法（LogitTree 与 4D 注意力掩码）以及一种仅需单次反向传播的自蒸馏方法 SDCC，从而消除压缩训练与推理之间的条件化不一致。 |
| [^75] | [Conditional Flow Matching for ML-Based Inverse Design Problems](https://arxiv.org/abs/2609.00863) | 该论文将条件流匹配（CFM）引入 EngiOpt 工程逆向设计框架，并在 EngiBench 的结构与热传导基准上，以累积最优性差距和最终最优性差距为指标，与条件扩散模型和条件GAN系统比较其作为梯度优化暖启动方案的性能。 |
| [^76] | [Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair](https://arxiv.org/abs/2609.00854) | 该安慰剂对照研究发现，故障定位在实际场景中很少可用（仅约9%的失败候选可定位），且即便可定位，基于频谱定位的片段填充修复也显著劣于盲目的整体重采样。 |
| [^77] | [A Checklist to assess the energy and carbon impacts of ML/AI applications in Earth System Modeling](https://arxiv.org/abs/2609.00847) | 该论文将分散的机器学习/人工智能可持续发展讨论提炼为一份按模型开发流程各阶段组织的实用清单，帮助地球系统科学从业者评估并降低其应用的能耗与碳足迹，并配套提供了估算能耗和碳排放的指标。 |
| [^78] | [Dense Process Supervision for Search Agents via Fact Utility Estimation](https://arxiv.org/abs/2609.00833) | 本文提出一种基于事实效用估计的密集过程监督方法，通过将推理过程建模为离散证据事实的累积，并利用贝叶斯估计将事实效用转化为步骤级奖励，有效解决了搜索智能体强化学习中的信用分配难题。 |
| [^79] | [HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution](https://arxiv.org/abs/2609.00829) | HarnessEvolve 提出了一种从参考轨迹中学习的智能体自我进化框架，通过将执行、评估、优化和门控解耦为独立模块，克服了信用分配失败、捷径学习和灾难性遗忘三大难题，实现了可靠且可泛化的智能体自我进化。 |
| [^80] | [Subspace Levenberg Marquardt Algorithms in Training Neural Networks](https://arxiv.org/abs/2609.00789) | 本文评估了子空间Levenberg-Marquardt算法（如KSLM和HSLM）在神经网络回归与分类任务中的性能，并将其与经典LM方法及SGD、Adam等一阶优化算法进行了比较，以展示子空间方法在降低二阶方法计算与内存开销方面的有效性。 |
| [^81] | [Semi-Supervised Classification with Informative Missing Labels in Weibull Mixture Models](https://arxiv.org/abs/2609.00774) | 该论文提出在两分量威布尔混合模型的半监督分类中，将标签缺失概率建模为分类不确定性的函数，从而证明缺失标签指示变量本身携带分类器信息，并据此刻画了贝叶斯决策边界的结构并推导了相应的Fisher信息量。 |
| [^82] | [Are You Thinking What I am Thinking? : Examining Conceptual Separation in Neural Architectures](https://arxiv.org/abs/2609.00764) | 本研究通过对CNN和LLM内部激活的几何与分布分析，揭示神经网络中存在“概念分离”现象——同一概念形成连贯表示、相关概念在表示空间中彼此更近，但这种连贯性在未见概念、域偏移和模糊主题下会减弱或坍塌。 |
| [^83] | [Frozen Cores Need Task Signal: Fisher-Whitened Cross-Covariance for Low-Resource LLM Adaptation](https://arxiv.org/abs/2609.00762) | 提出FCCA方法，通过对角Fisher矩白化输入-误差交叉协方差来为冻结核心微调构建任务感知的低秩核心基，在相同可训练参数预算下的11个任务、4种模型设置上超越了八种现有基构造方法。 |
| [^84] | [How Do Language Models Choose Between Context and Memory?](https://arxiv.org/abs/2609.00753) | 本文通过反事实实验证明了从一致性提示中估计的“权威方向”在语言模型内部因果地决定了模型在上下文信息与参数记忆之间的选择——沿这些方向交换激活坐标可重现30-68%的来源选择偏移。 |
| [^85] | [Text Capability Loss in Vision-Language Adaptation: An Attention-Sink Diagnosis](https://arxiv.org/abs/2609.00746) | 该论文发现将大语言模型微调为视觉-语言模型时文本能力的损失源于注意力汇位置被扰动，并提出Sink Strength指标，仅需在单GPU上计算几秒钟即可预测适配后的文本能力退化程度。 |
| [^86] | [Online Self-Weighted Fine-Tuning](https://arxiv.org/abs/2609.00734) | 该论文提出在线自加权微调（OSW-FT），通过少量仅推理的采样轨迹估计模型当前成功率，在线调整每个查询的SFT损失权重，在保持专家轨迹优化方向的同时自适应更新幅度，兼顾了SFT的稳定性与RL的自适应性。 |
| [^87] | [Agentic Empirical Asset Pricing: Methodological Foundations](https://arxiv.org/abs/2609.00731) | 本文提出了智能体化实证资产定价（AEAP）这一新范式，为自主因子发现系统提供了参考架构、严格的因子评估标准与样本外回测方法，并通过对SEADS与五个基线系统的评估证明此类系统必须从多个维度同时加以评价。 |
| [^88] | [SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification](https://arxiv.org/abs/2609.00728) | SOVER框架将语义映射与形式化验证分离，利用Z3和dReal等SMT求解器对LLM生成的优化问题重构进行形式化认证，并在NLEquiv-150基准上实现了149/150的正确分类。 |
| [^89] | [MaskCode: Mask Transformer for Feedback-Assisted Coding With Linear Block Codes](https://arxiv.org/abs/2609.00715) | 提出了MaskCode，一种基于Transformer的内层反馈码，通过掩码机制将外层线性分组码的结构知识显式融入反馈编码器设计，避免将反馈资源浪费在外层纠错码已能纠正的错误模式上，从而提升级联编码系统的性能。 |
| [^90] | [Controllable Image Captioning with Prompt-Conditioned Scene Rewards](https://arxiv.org/abs/2609.00709) | 提出FoCUS方法，通过基于场景图对齐组件分数的提示条件化奖励目标并用GRPO优化，让用户能够通过自然语言提示精确控制图像描述的语义重点（如对象、属性、关系或特定区域）。 |
| [^91] | [Patterning in Practice: Debiasing Reward Models with Susceptibilities](https://arxiv.org/abs/2609.00699) | 本文提出基于敏感度的模式化重新加权方法来去除奖励模型的风格偏差，在 RM-Bench Hard 上提升 14.2 个百分点且整体准确率保持不变，且权重具有可解释性和跨模型可迁移性。 |
| [^92] | [MUGEN: Generating Unlearnable Graph Examples for Multiple Learning Tasks](https://arxiv.org/abs/2609.00696) | MUGEN是首个面向多种学习任务的不可学习图样本生成框架，它通过对单一干净数据集进行特征扰动，利用共享GNN编码器同时保护节点分类、图分类和链接预测等多种任务免受未经授权的模型学习。 |
| [^93] | [Verdict Instability of OOD Scores under Reference Resampling](https://arxiv.org/abs/2609.00691) | 本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。 |
| [^94] | [A Study of Hidden-State Optimization Order in Predictive Coding Networks](https://arxiv.org/abs/2609.00686) | 该论文提出一种边界优先的隐状态优化顺序——先协调块边界处的隐状态、再细化块内表示，显著提升了预测编码网络在CIFAR-10上的准确率并增强了早期层的特征学习。 |
| [^95] | [HarmoCore: Functional Latent Diffusion for Sparse Reconstruction of Oscillatory Wave Fields](https://arxiv.org/abs/2609.00679) | HarmoCore将生成式扩散先验置于由函数式Tucker核心构成的紧凑连续波场潜在空间中，并直接在该核心空间执行频率条件化的扩散后验采样，从而从极端稀疏的传感器观测中高效重建复值振荡波场。 |
| [^96] | [EEG-AS: Instance-Level Foundation Model Selection for EEG Foundation Models via Behavior Reconstruction](https://arxiv.org/abs/2609.00653) | 该论文提出EEG-AS框架，将EEG基础模型选择形式化为实例级算法选择问题，通过锚定模型的特权预测标记重构不可获得的基础模型行为，从而为每个EEG实例自动选择最合适的基础模型。 |
| [^97] | [Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search](https://arxiv.org/abs/2609.00652) | 本文提出首个环境锚定的LLM操作者审计框架，通过为进化式Contexto搜索中的每个中间提议赋予精确结果，实证证明模型自我报告不可作为验证依据——操作者将成功率夸大4.8至9.3倍，且关于置信度校准、理由传递和适应度选择的三个假设全部被证伪。 |
| [^98] | [DK-GBMKKM: Dynamic Kernel-Space Granular-Ball Multiple Kernel $k$-Means Clustering](https://arxiv.org/abs/2609.00647) | 提出DK-GBMKKM方法，通过在融合核空间中动态生成粒球并交替优化核权重与粒球隶属度，同时构造样本规模加权的粒球核，从而提升了多核k均值聚类对噪声、边界样本的鲁棒性以及对核空间几何变化的适应性。 |
| [^99] | [Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity](https://arxiv.org/abs/2609.00632) | 提出FedRoRA框架，通过将LoRA适配解耦为共享的全局方向与个性化的按秩幅值，在秩异构联邦学习场景下实现细粒度的客户端个性化微调，从而同时应对资源异构与数据异构的双重挑战。 |
| [^100] | [Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning](https://arxiv.org/abs/2609.00605) | 提出数据无关的CONFS框架，通过引出模型自身记忆的知识来构建与模型对齐的遗忘集，解决了大语言模型机器遗忘中遗忘集与模型实际记忆内容不对齐所导致的信息泄露或效用下降问题。 |
| [^101] | [Topological Steering](https://arxiv.org/abs/2609.00597) | 该论文提出了“拓扑引导”新框架，利用拓扑数据分析中的持久图对激活空间进行拓扑表示，从而以对离群值和局部扰动更鲁棒的方式控制大语言模型的行为。 |
| [^102] | [Real-Time Neuromorphic Spectrum Intelligence Simulator](https://arxiv.org/abs/2609.00585) | 本文提出了RT-NuSIS模块化实时神经形态模拟器，将脉冲神经网络、忆阻突触、物理信息能量收集与对抗模型相结合，为受限能量下的动态频谱接入研究提供数学形式化保证及可复现的能效、延迟与鲁棒性基准测试平台。 |
| [^103] | [GeoPAR: Large-Scale Multi-Agent Combinatorial Optimization with Geometry-Guided Parallel Autoregressive Learning](https://arxiv.org/abs/2609.00577) | GeoPAR提出了一种几何引导的并行自回归强化学习框架，通过投影窗口稀疏几何机制、稀疏边偏置注意力以及缓存引导的冲突处理机制，实现了大规模多智能体组合优化问题的高效求解。 |
| [^104] | [VoiceLongMemEval: Do Assistants Remember How You Sounded?](https://arxiv.org/abs/2609.00570) | 该论文提出了VoiceLongMemEval（VLME）基准，用于评估AI助手在长时多会话对话中能否记住情感、韵律和语音事件等副语言信息，发现现有大语言模型普遍存在无法捕捉说话方式的“情感鸿沟”。 |
| [^105] | [EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection](https://arxiv.org/abs/2609.00566) | EEG-VID提出了一种任务引导式潜变量预测预训练框架，通过指数移动平均编码器预测未来EEG潜状态，在42组跨会话跨被试对比中有41组提升准确率（最高提升16.22个百分点），并可有效应用于场景约束下的辅助目标选择。 |
| [^106] | [Manifold-Aware General Coded Computing for Straggler-Resilient Distributed Computing](https://arxiv.org/abs/2609.00552) | 该论文提出一种流形感知的通用编码计算方法，通过在码设计中保留并利用输入数据的内在结构（而非像传统信源编码那样消除结构），实现抗滞后节点的分布式计算。 |
| [^107] | [EM^2Mem: Event-Centric Multimodal Memory for Large Language Models](https://arxiv.org/abs/2609.00551) | 该论文提出EM^2Mem，一种以事件为中心的多模态记忆框架，通过在记忆构建阶段将多模态记录、时间上下文、图谱关系与溯源信息绑定到事件锚点，形成“可直接用于生成”的记忆单元，免去了推理时重建跨模态对齐的负担，并在三个长视频问答基准上将平均准确率较最强记忆基线提升2.0至3.7个百分点。 |
| [^108] | [GenONet: A Generative operator Network for High-Resolution Precipitation Nowcasting](https://arxiv.org/abs/2609.00544) | 该论文提出GenONet，首次将深度算子网络作为生成器嵌入生成对抗网络框架中，实现了长达3小时的高分辨率降水临近预报，能够生成清晰且物理一致的预报结果。 |
| [^109] | [DeSyR: A Decoupled Symbolic Recovery Framework with PINN-Guided Structure Search and Physics-Informed Coefficient Refinement](https://arxiv.org/abs/2609.00530) | 该论文提出DeSyR框架，将微分方程的符号恢复过程解耦为两个阶段——先由物理信息神经网络引导候选拓扑结构搜索，再仅基于控制方程和约束条件精炼系数，并从理论上证明了教师误差继承的上界以及仅在物理约束下可条件性恢复精确系数的保证。 |
| [^110] | [Why Multi-Layer Message Passing Works: Completeness Theory for Graph Neural Network Interatomic Potentials](https://arxiv.org/abs/2609.00528) | 本文提出多层完备性理论，证明在通用性、重叠与连通性条件下，稀疏截断图上的 $L$ 层消息传递与访问完整 $L$ 跳邻域具有同等的表示能力，从而首次严格证明了图神经网络原子间势中使用小于物理相互作用范围的逐层截断消息传递这一通用做法的合理性，并由此推出 DPA3 与 CHGNet 架构具有通用近似能力。 |
| [^111] | [Soft-Argmax for the Projective Plane via the Veronese Embedding](https://arxiv.org/abs/2609.00521) | 该论文提出利用Veronese嵌入将无向直线以ℤ₂-不变的方式嵌入到线性空间中，解决了soft-argmax在具有莫比乌斯带拓扑结构的无向直线空间上失效、撕裂相邻直线的问题。 |
| [^112] | [Learning Task-Specific Antibody Representations via Function-Aware Masking](https://arxiv.org/abs/2609.00518) | 该论文提出功能感知掩码这一预训练算法家族，通过将掩码位置与特定功能先验（如IMGT注释或结构预测）对齐来学习任务特异性的抗体表示，在结构相关任务上最高提升14%，在CDR相关任务上最高提升5.9倍。 |
| [^113] | [VATO: A Vortex-Force-Aware Transformer Operator for Unsteady Separated Aerofoil Flows](https://arxiv.org/abs/2609.00507) | VATO将涡力图方法与几何感知Transformer神经算子相结合，在不增加推理成本的前提下，实现了对非定常分离翼型流动中气动载荷的更准确预测。 |
| [^114] | [Independent Reinforcement Learning in Discounted Markov Games](https://arxiv.org/abs/2609.00504) | 本文在“PPAD的ETH”假设下证明了折扣一般和马尔可夫博弈中独立学习计算粗相关均衡的困难性，并提出首个无需结构限制、具有次指数收敛保证的彻底非耦合分层乐观镜像下降算法。 |
| [^115] | [A hybrid quantum-classical neural network for learning to route](https://arxiv.org/abs/2609.00489) | 本研究提出用小型量子神经网络替换基于注意力的路由模型中的编码器前馈模块，在带容量约束车辆路径问题上将参数减少56.6%的同时保持接近经典神经基线的求解质量，为神经组合优化提供了一种可行的混合模块压缩策略。 |
| [^116] | [AdaptNTK: Adaptive Uncertainty Quantification and Active Learning for Neural Network Potentials](https://arxiv.org/abs/2609.00488) | AdaptNTK 提出了一种单模型框架，通过在经验神经正切核（NTK）特征空间中度量正则化马氏距离来量化神经网络势的不确定性，并支持在主动学习采集批次构建过程中递归更新不确定性以避免冗余构型。 |
| [^117] | [EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities](https://arxiv.org/abs/2609.00487) | 提出了EvoFlint框架，将多轮红队测试从生成问题重新定义为搜索问题，通过进化式质量多样性搜索演化分阶段对话攻击策略，构建出目标模型漏洞的结构化图谱。 |
| [^118] | [Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?](https://arxiv.org/abs/2609.00482) | 该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。 |
| [^119] | [Fractal dimension predicts quantum kernel collapse in angle-encoded data](https://arxiv.org/abs/2609.00475) | 该论文提出用数据的相关性分形维度 D2 作为先验量子比特数预算，可准确预测并避免角度编码量子核的几何坍缩，使量子核在真实硬件上以最小比特数保持有效。 |
| [^120] | [Higher Structures in Deep Learning](https://arxiv.org/abs/2609.00472) | 本文阐述了高元张量运算在深度学习中的重要性，对训练后神经网络中的高元现象进行了新颖的实证研究，并提出了多层感知机的超图推广形式，同时探讨了其与进化算法的联系。 |
| [^121] | [Context Window Failures in Relational Foundation Models](https://arxiv.org/abs/2609.00460) | 当前关系基础模型因上下文窗口限制，在面对拥有大量关联记录的实体时会严重失效，仅一个简单的时序预聚合步骤就能将R²从0.18提升至0.65，表明现有模型尚无法应对高基数现实数据。 |
| [^122] | [Can LLMs Use Relational Transformer Embeddings?](https://arxiv.org/abs/2609.00457) | 本研究通过将冻结的关系Transformer嵌入注入Qwen3.5-4B并经过SFT与GSPO两阶段训练，发现在RelBench的多种监督模式下，这种混合融合策略并未持续优于独立的RT模型，反而经常低于随机水平且对序列化格式和标记预算高度敏感。 |
| [^123] | [HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference](https://arxiv.org/abs/2609.00450) | 提出硬件效率感知的分层块量化方法HBQ，突破块量化中块大小与精度之间的固有权衡，在大块设计下同时实现高硬件效率与精确的大语言模型推理。 |
| [^124] | [CRAD: Class-wise Reliability-Aware Distillation for Decentralized Heterogeneous Federated Learning](https://arxiv.org/abs/2609.00446) | 该论文提出CRAD框架，通过类别级可靠性感知的方式组合同伴教师的软预测，在去中心化联邦学习中同时解决模型架构异构与非独立同分布数据两大难题，且无需中央服务器或共享原始数据。 |
| [^125] | [Capability-Gated Language Models: Security Composes, Utility Does Not](https://arxiv.org/abs/2609.00445) | 提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。 |
| [^126] | [Group Adaptive Clipping Policy Optimization](https://arxiv.org/abs/2609.00444) | 该论文提出 GAPO，一种基于反向 KL 信任域视角对 GRPO 的即插即用改进，通过根据 rollout 优势自适应调整裁剪边界，让具有更强学习信号的稀有正确 rollout 获得更大的更新空间，从而解决固定裁剪对探索性 rollout 的过度抑制问题。 |
| [^127] | [Physiological Information Reliability: Cross-Layer Adaptive Resource Allocation for Cardiovascular Sensing](https://arxiv.org/abs/2609.00435) | 该论文提出生理信息可靠性（PIR）跨层框架，通过上下文老虎机将生理信息价值与无线、能量和计算状态联合建模，自适应调整心血管传感的感知与通信决策，在满足医疗延迟约束的前提下实现有前景的低能耗工作点和具有竞争力的生理估计性能。 |
| [^128] | [SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents](https://arxiv.org/abs/2609.00434) | SAGE提出将工作流规范编译为原子准则，通过会弃权而非猜测的符号与编码器/NLI验证器级联来评估任务型对话智能体每轮的状态推进，其中SAGE-Core可在零付费LLM成本下判定81-91%的准则。 |
| [^129] | [Accelerating Chemical Kinetics for Exoplanet Atmospheres using Neural Networks](https://arxiv.org/abs/2609.00428) | 本文提出一种基于神经网络残差流映射架构的系外行星大气化学动力学代理求解器，其推理速度达微秒级、比经典求解器快数个数量级，可在保持精度的前提下大幅加速行星大气化学模拟。 |
| [^130] | [How Temporal Correlations Shape Memory in Linear Recurrent Neural Networks](https://arxiv.org/abs/2609.00420) | 本文精确求解了输入存在时间相关性时线性循环神经网络的学习动力学，发现相关性的全部效应集中于“保留过去”的代价：它使学习过程呈现记忆建立、过冲再部分遗忘的轨迹，并使最终网络保留更少过去信息，同时记忆在一个仅由相邻输入相似度决定的阈值处关闭，不受序列长度和更长程相关性影响。 |
| [^131] | [A Multi-Branch Feature Fusion Approach for Health Misinformation Detection and Propagation](https://arxiv.org/abs/2609.00403) | 本文提出一种基于详尽可能性模型和计划行为理论的多分支特征融合框架，融合Transformer语义、修辞线索、立场表示及心理动机特征来检测健康错误信息，并引入可解释的认知传播评分（CPS）辅助传播风险推理，在三个基准数据集上ROC-AUC最高达0.9。 |
| [^132] | [Neural means and kernel corrections for operator learning](https://arxiv.org/abs/2609.00389) | 该论文提出将神经网络均值与Matérn核回归修正相结合的方法，在结构力学和OCO-2辐射传输两个基准问题上达到了或超越了已发表最佳结果，并从理论上证明和量化了核修正之所以有效的机制。 |
| [^133] | [Risk-Aware Decision-Making for Autonomous Overtaking: A World Model-Based Mixture-of-Experts Framework](https://arxiv.org/abs/2609.00385) | 本文提出基于世界模型的风险感知专家混合框架，利用学习到的潜在动力学模型进行并行多步推演，将安全评估从动作层面提升到轨迹层面的累积风险水平，并通过分层门控机制动态协调专家以适应不同交互强度，从而提升自主超车决策的长期安全性。 |
| [^134] | [Adapting Without Gradients: Affine Statistics Transport and What Its Certificate Can Tell You](https://arxiv.org/abs/2609.00374) | 提出CASTER，一种无需梯度的测试时自适应方法，通过在判别子空间中存储源类别统计并估计仿射变换来解析地传输源类别分布，使冻结模型无需反向传播即可适应目标数据，同时以约18倍更少的状态存储优于k-NN基线。 |
| [^135] | [Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning](https://arxiv.org/abs/2609.00367) | 本文提出一种即插即用的神经符号层，无需任何微调或RLHF即可在Text-to-SQL等数据工程任务上平均提升85%的准确率，同时缓解Transformer长上下文的计算资源瓶颈。 |
| [^136] | [Counterfactual Fragility Certificates: Exposing High-Confidence Brittleness under Structured Evidence Failure](https://arxiv.org/abs/2609.00366) | 提出反事实脆弱性证书（CFC），一种模型无关的协议级审计对象，将每个预测映射为由贪婪翻转预算、边际崩塌面积等指标刻画的有序证据失效轨迹，从而揭示表格决策系统中高置信度预测在结构化证据失效下的脆弱性。 |
| [^137] | [Dr. Claw: An AI Scientist Workspace for Vibe Research](https://arxiv.org/abs/2609.00365) | Dr. Claw 是一个开源的AI科学家工作区，通过持久化状态对象、可复用技能库和多执行器协调，将现有命令行编码代理封装为可控、可审计的人机协同工作流，把科研中的规划、执行与写作整合为一个可追踪、可恢复的闭环。 |
| [^138] | [Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance](https://arxiv.org/abs/2609.00363) | 本文通过对INT8量化GEMM流水线系统性注入九种故障，证明了基于容差的符合性测试在构造上无法检测仅使输出偏移至多一个bfloat16间距的尾部计算故障，而采用二的幂次量化缩放因子是保障跨GPU核函数确定性推理的关键途径之一。 |
| [^139] | [A Stable Aggregation Method for Quantum Federated Learning](https://arxiv.org/abs/2609.00356) | 该论文提出了一种融合QoS感知客户端加权、循环参数聚合和有界中点更新控制的自洽中点聚合方法，显著提升了量子联邦学习在异构数据与量子硬件噪声等挑战下的稳定性与准确率。 |
| [^140] | [Do LLMs Know Your Neighborhood? Auditing LLM Priors for Neighborhood-Level Mobility Prediction and Structural Alignment](https://arxiv.org/abs/2609.00345) | 该论文首次系统审计了零样本大语言模型在社区级人口流动性预测中的先验知识，发现其准确率不及监督基线模型，并通过方向性对齐分析检验LLM隐含的预测变量效应与实证统计趋势的一致性。 |
| [^141] | [Latent-Space No-Arbitrage Geometry of Generative Models for Implied Volatility Surfaces](https://arxiv.org/abs/2609.00332) | 本文在潜空间中刻画生成模型输出隐含波动率曲面的无套利约束，通过标量边际定义可容许潜集并证明其在扰动下的稳定性，同时为零边际边界构造水平集方程，且该方法适用于任何具有确定性映射的生成模型。 |
| [^142] | [Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts](https://arxiv.org/abs/2609.00330) | 该论文构建了一个基于真实呼叫中心ASR转录文本的人工标注主题匹配基准数据集，并通过系统对比发现，配备自然语言主题描述的轻量级大语言模型匹配器在处理噪声转录文本时性能优于句子嵌入和正则表达式方法。 |
| [^143] | [The Curse of Multilinguality in Lexical Normalization](https://arxiv.org/abs/2609.00329) | 该研究通过固定容量字符级模型在十二种语言上的实验发现，词汇规范化存在明显的“多语言诅咒”：语言联合训练数量超过一到四种后，各语言准确率持续下降约百分之四十，且下降源于语言间对固定模型容量的竞争而非数据稀释。 |
| [^144] | [Workload Identification with Physical Side Channels for AI Governance](https://arxiv.org/abs/2609.00309) | 本研究证明外部观察者可利用GPU功耗这一物理侧信道，在无需运营商配合的情况下以97%的准确率识别NVIDIA H200上运行的是AI训练、推理还是非AI计算，为AI治理的国际算力核查提供了可独立验证的技术手段。 |
| [^145] | [TRUST: Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening](https://arxiv.org/abs/2609.00300) | 提出TRUST闭环训练策略，在训练中动态重校准豁免阈值并惩罚接近豁免区域的阳性样本，在保证召回率目标的前提下显著提高了乳腺癌筛查中可安全豁免医生复核的阴性病例比例。 |
| [^146] | [Geometry-aware Latent Autoregressive Generative Model for PDEs in Complex Domains](https://arxiv.org/abs/2609.00297) | 提出几何感知潜空间自回归生成模型GeoLAMP，通过双编码器联合捕获全局拓扑与细尺度几何特征，并结合流匹配的因果自注意力Transformer建模时间动力学，实现复杂不规则几何结构中多物理场PDE的高效、稳定且可扩展的求解。 |
| [^147] | [WiSDoM: Wireless Sparse Decision Transformer with Mixture-of-Experts for Multi-Task Mobile Network Optimization](https://arxiv.org/abs/2609.00284) | 该论文提出WiSDoM，一种结合混合专家机制的稀疏多任务离线强化学习框架，能够在异构6G无线环境中实现自适应多小区（CoMP）选择，从而解决传统无线资源管理难以在多任务场景下保持一致性能的问题。 |
| [^148] | [Lightweight Adaptation of EEG Foundation Models for Stroke Motor Imagery Decoding: Domain Shift and Subject-Level Robustness](https://arxiv.org/abs/2609.00282) | 轻量级 LoRA 适配可让预训练脑电基础模型（尤其是 REVE-base）有效克服健康人群与脑卒中患者之间的领域偏移，在脑卒中运动想象解码中达到约 0.85 的准确率，且更大的模型容量并不保证更好的适配效果。 |
| [^149] | [CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships](https://arxiv.org/abs/2609.00250) | 该论文发布了CompanionSim——一个包含2,240段模拟人机对话的合成数据模拟框架，覆盖七种用例中的16种聊天机器人行为，用于大规模研究人类对AI陪伴行为的感知。 |
| [^150] | [QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization](https://arxiv.org/abs/2609.00224) | QTEA提出了一种低于2比特的训练后量化框架，通过将权重量化为三值、利用1:4半结构化稀疏的显著权重残差进行误差补偿，并结合逐列缩放精修与误差衰减机制，在保持GPU硬件效率的同时显著降低了大语言模型低比特量化的精度损失。 |
| [^151] | [WHALE: A Simple Recipe for Joint Harness-Weight Optimization](https://arxiv.org/abs/2609.00196) | 提出 WHALE 方法，通过交替进行“在当前外壳下更新模型权重”与“在更新后的模型下搜索更优外壳”两个阶段，实现模型权重与执行框架的联合优化，避免单一组件优化时被冻结组件造成的性能瓶颈。 |
| [^152] | [Provably Efficient Federated Reinforcement Learning with Linear Function Approximation and Logarithmic Communication Cost](https://arxiv.org/abs/2609.00193) | 提出Fed-LSVI，首个针对具有线性函数逼近的联邦在线强化学习的可证明高效算法，通过基于行列式的事件触发同步机制仅交换压缩充分统计量，在实现$\widetilde{O}(\sqrt{Md^3H^4T})$遗憾界的同时将通信成本降低至对数级。 |
| [^153] | [Elite-Weighted Supervised Fine-tuning for Goal-Directed Molecular Optimization](https://arxiv.org/abs/2609.00189) | 提出EW-SFT方法，利用奖励信号筛选高分的精英分子，再用模型自身的预训练损失对这些分子进行监督微调，从而实现无需依赖架构特定轨迹概率、可跨模型复用的目标导向分子优化。 |
| [^154] | [Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs](https://arxiv.org/abs/2609.00184) | 该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。 |
| [^155] | [Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs](https://arxiv.org/abs/2609.00155) | 该研究发现不同的潜在语言探测方法会得出系统性不一致的结论，表明多语言大模型通过英语等“潜在通用语”路由计算的说法可能更多取决于探测手段的选择，而非模型本身固有的计算机制。 |
| [^156] | [Flawed in Nature, Perfect through Evolution](https://arxiv.org/abs/2609.00129) | 该论文提出通过让一群AI/ML模型的系数刻意发生偏离最优的“突变”来维持模型多样性，从而在非平稳环境中充当统计对冲手段，实现可靠且持续的性能提升。 |
| [^157] | [Good Memory Has ECC: Evaluating the Memory of Vision-Language Models Beyond Accuracy](https://arxiv.org/abs/2609.00103) | 该论文提出ECCBench基准，从效率、压缩和校准三个维度超越单纯准确率来评估视觉-语言模型的记忆能力，发现预训练VLM对文本记忆有压缩但对视频没有且校准较差，并且若干非Transformer架构在压缩-校准权衡上优于RoPE Transformer。 |
| [^158] | [Different representation learning objectives recover distinct latent structures from the same psychometric data](https://arxiv.org/abs/2609.00100) | 不同的表征学习目标会从同一份心理测量数据中恢复出截然不同的潜在结构——对比学习大幅提升师生匹配检索性能却破坏行为表型组织，而PCA更利于保留行为结构，揭示了检索对齐与行为结构保留之间的根本性权衡。 |
| [^159] | [Generative artificial intelligence for reliable mechanistic reasoning for corrosion](https://arxiv.org/abs/2609.00099) | 该研究提出了一个面向腐蚀领域的检索增强生成框架，通过对三个开源大语言模型在专家验证的问答数据上进行微调并结合混合检索流水线，实现了兼具准确性与机理可靠性的腐蚀知识推理，在镁合金腐蚀上取得了显著效果。 |
| [^160] | [Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding](https://arxiv.org/abs/2609.00097) | FFD是一种硬件-算法协同设计框架，通过将选择器与计算器融合为单一内核、基于低比特量化的内容感知扫描取代元数据索引，以及无需全局同步的top-delta动态块过滤策略，实现了免训练、即插即用的长上下文解码加速，内核级加速比最高达11.6倍。 |
| [^161] | [Local Reference Geometry Residual Augmentation for Imbalanced Time Series Classification](https://arxiv.org/abs/2609.00093) | 该论文提出局部参考几何（LRG），一种应用于固定特征提取器与分类器之间的轻量级后置特征增强模块，通过度量少数类样本的局部暴露度和类别混合风险，并用来自附近训练几何的标准化有符号位移增强特征，来修复不平衡条件下特征空间的局部几何失效问题。 |
| [^162] | [Safin-1: Safety from Within through Memory-Native State Evolution](https://arxiv.org/abs/2609.00092) | Safin-1提出基于跨上下文历史记忆锚定路由（MARCH）架构的基础模型系列，通过记忆路由与状态演化将安全能力内建于模型原生计算之中，实现无需反复修改主干网络、可在测试时自适应调整的“由内而外”的安全性。 |
| [^163] | [Assessing Alignment and Stability of Feature Importance Explanations via Weight of Evidence](https://arxiv.org/abs/2609.00090) | 该论文提出了一个基于证据权重的假设检验框架，能够从原理上评估特征重要性解释方法与先验知识的对齐程度及其稳定性，并将其应用于LIME和SHAP的分析中。 |
| [^164] | [Foundation models for electricity price forecasting and battery arbitrage: Can they replace market-specific forecasting models?](https://arxiv.org/abs/2609.00089) | 研究表明，在零样本模式下，只有TabPFN基础模型能在三个欧洲电力市场中持续显著超越专门的电价预测模型，但其统计优势并不能直接转化为电池储能套利中的经济优势。 |
| [^165] | [Stochastic complexity of vectors containing cluster structure](https://arxiv.org/abs/2609.00084) | 本文提出一种递归公式来高效计算NML模型的归一化常数，将计算包含聚类结构向量最短编码长度的时间复杂度从多项式时间降低到线性时间。 |
| [^166] | [RW-LoRA: Communication-Efficient Decentralized LoRA Fine-Tuning via Random Walks](https://arxiv.org/abs/2609.00078) | 提出基于随机游走的去中心化 LoRA 微调方法 RW-LoRA，通过单个模型令牌在网络中顺序更新，免除全局同步，大幅降低通信与计算成本并避免聚合误差。 |
| [^167] | [When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation](https://arxiv.org/abs/2609.00071) | 在部分线性模型的模拟研究中，干扰函数的预测误差无法一致地反映因果估计的偏差，表明仅用预测误差来评估干扰函数估计器对于因果推断而言是不充分的。 |
| [^168] | [OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization](https://arxiv.org/abs/2609.00066) | 提出OCGQuant，一种以“异常值伴随分组（OCG）”为核心的NVFP4训练后量化方法，通过自适应地将异常值通道与伴随通道分组，减少由块最大值主导缩放因子所造成的“附带量化误差”，从而在不引入额外计算的前提下提升低比特推理的量化精度。 |
| [^169] | [Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning](https://arxiv.org/abs/2609.00064) | 该论文形式化了注意力层面的“上下文敏感性”（ICS）指标，并通过Llama-2-7B上的四臂消融实验证明，最大化ICS并不能保留真实的行为性上下文学习能力（ICL-GAP接近零且MMLU从0.371降至0.279），揭示了注意力代理指标与行为层面ICL之间的“古德哈特定律”式解耦。 |
| [^170] | [ReNFT: Repairing Mode Collapse in Reward Post-Training via Internal Probability-Mass Recalibration](https://arxiv.org/abs/2609.00061) | 本文提出ReNFT方法，通过内部概率质量重新校准修复扩散模型奖励后训练中出现的模式坍塌，在保留已获得奖励的同时恢复提示内多样性，无需依赖任何外部信号或接口。 |
| [^171] | [DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction](https://arxiv.org/abs/2609.00059) | DISTAL提出了一种双先验框架，通过自监督成分预训练和从ALIGNN教师模型进行结构知识蒸馏，实现了推理时无需晶体结构输入的结构无关材料性质预测，适用于低数据、结构信息缺失的早期筛选场景。 |
| [^172] | [ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation](https://arxiv.org/abs/2609.00057) | 提出ValueGraph图预训练框架，将自动推断的道德价值信号作为软约束辅助信号，结合对比学习与聚类目标学习上下文化的用户表示，在立场检测和推特机器人检测任务上取得提升。 |
| [^173] | [Convergence issues in Relational Concept Analysis based on AOC-posets](https://arxiv.org/abs/2609.00054) | 本文研究了在关系概念分析（RCA）中以AOC-偏序集替代完整概念格时所产生的收敛性问题，从而在缓解组合爆炸的同时保留结构中最具信息量的部分。 |
| [^174] | [AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes](https://arxiv.org/abs/2609.00052) | 提出AgentProv，首个基于动作的智能体式LLM API身份审计方法，通过工具使用策略探针利用内化在模型权重中的工具使用行为，克服了文本通道审计在智能体API场景下的结构性脆弱问题。 |
| [^175] | [Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness](https://arxiv.org/abs/2609.00050) | 提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。 |
| [^176] | [REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent](https://arxiv.org/abs/2609.00049) | REAL-Q提出了一种打破传统折中的后训练量化新范式，通过端到端对齐的代理损失目标和每128列一次的动态块级梯度下降，解决了现有方法中Hessian矩阵被整层冻结导致的信息错位问题，从而更精确地逼近全局损失实现大语言模型量化。 |
| [^177] | [Task-Specific Prompt with Global Context for Multi-Task Graph Pre-Training](https://arxiv.org/abs/2609.00047) | 提出TPGC双先验提示初始化方法，通过显式建模任务先验与结构先验的协同作用，解决多任务图预训练中随机初始化提示导致的任务相关性弱、结构感知差和可迁移性不足的问题。 |
| [^178] | [Dense Weak Hiding: Closing Complexity Gaps in Nonconvex and PL Finite-Sum Optimization under Individual Smoothness](https://arxiv.org/abs/2609.00045) | 本文在个体光滑性条件下证明了非凸有限和优化的匹配IFO复杂度下界，填补了先前结果中缺失的√n因子并确定了其极小极大复杂度，同时提出重启PAGE算法，在PL条件下获得了更紧的复杂度保证。 |
| [^179] | [UI-Venus-2 Technical Report](https://arxiv.org/abs/2609.00028) | UI-Venus-2是一个通用GUI基础智能体，通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行，并从环境、任务和验证三个维度联合扩展，从而获得可靠的强化学习信号并迈向实际部署。 |
| [^180] | [ES-AHD: An Evolution Strategy Framework for Automatic Heuristic Design](https://arxiv.org/abs/2609.00023) | ES-AHD框架将进化策略与大语言模型驱动的自动启发式设计深度融合，通过基于LLM的语义重组和基于温度采样的随机协方差自适应两大机制，实现有针对性的中心导向搜索，并动态平衡探索与利用。 |
| [^181] | [I-CARE: Analysis of interference-related phenomena in a controllable, diverse and representative unlearning setting for text-to-image models](https://arxiv.org/abs/2609.00003) | 本文提出I-CARE方法论，首次将文生图模型机器遗忘过程中对语义相关概念造成的意外损害（即“干扰”）形式化为首要研究对象，通过为任务、指标和结果报告提供正式定义，实现对干扰现象的系统性、可复现研究。 |
| [^182] | [TopoCompress: Long Context Compression via Graph-Wired Semantic Trajectories](https://arxiv.org/abs/2608.30811) | TopoCompress提出了一种无需训练、与模型无关的长上下文压缩框架，通过构建混合图连接连贯的语义片段并在其上传播查询引导的相关性分数，在五个长上下文基准任务上以更少的资源持续超越强大的压缩基线。 |
| [^183] | [BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs](https://arxiv.org/abs/2608.30646) | 提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。 |
| [^184] | [TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization](https://arxiv.org/abs/2608.29564) | 论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。 |
| [^185] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^186] | [Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry Learning for Proteins via Hypergraph Modeling](https://arxiv.org/abs/2608.29207) | 提出 Hyper-Fold，通过超图建模将蛋白质序列内容与三维几何组织为超边，并利用分解为 K 个基算子的边条件化双线性矩阵值算子，以消息传递的计算代价逼近序列-几何学习的表达能力上限，在酶功能预测、折叠分类和配体结合位点检测等任务上表现出色。 |
| [^187] | [AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment](https://arxiv.org/abs/2608.28632) | 提出AutoScientist-Quant框架，将量化研究建模为预算约束下的搜索问题，通过单一自进化控制器统一决策Alpha生成、因子库选择和模型调优，实现从假设到可部署策略的全流程自动化，并修复了评估流程中的前视偏差问题。 |
| [^188] | [Performative Privacy: When Differential Privacy Maximizes Utility](https://arxiv.org/abs/2608.28198) | 该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。 |
| [^189] | [The Frame Kernel Method for Multiscale Operator Learning](https://arxiv.org/abs/2608.25084) | 本文提出一种基于新型多尺度核帧逼近的算子学习方法，通过帧系数映射实现多尺度分解，在替代建模中比神经算子方法显著更精确。 |
| [^190] | [Common-Center Geometry and Certified Radial Reconstruction for Energy-Form Full Conformal Regions](https://arxiv.org/abs/2608.24964) | 本文证明了在对称性和凸性条件下，能量形式全共形预测区域呈星形，且对于幂距离在β≥1时具有确定性几何性质，同时指出候选评分凸性不足以保证连通性。 |
| [^191] | [Stress Testing Unlearning Algorithms](https://arxiv.org/abs/2608.22527) | 本文提出了WMDP++基准，通过主动测试已遗忘信息能否被强制提取、并评估模型在与遗忘内容语义相近的边界问题上的表现，弥补了现有机器遗忘评估的两大不足，为遗忘算法提供了更严格的压力测试。 |
| [^192] | [Beyond Dense Adam States: Adaptive Log-Space Quantization for Memory-Efficient Optimizers](https://arxiv.org/abs/2608.22322) | 本文提出自适应对数空间量化方法，针对不同优化器状态拓扑定制精度，以实现内存高效且保持更新准确性。 |
| [^193] | [MRMAD: A Multi-Round Multi-Audio Benchmark for Evaluating Acoustic Degradation Perception in Large Audio-Language Models](https://arxiv.org/abs/2608.22236) | 提出了MRMAD，一个多轮多音频退化基准，通过多轮对话评估大型音频-语言模型对音频质量退化的识别、严重程度比较与跨轮次一致性推理能力。 |
| [^194] | [ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents](https://arxiv.org/abs/2608.21969) | 本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。 |
| [^195] | [Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements](https://arxiv.org/abs/2608.18294) | 本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。 |
| [^196] | [Non-Parametric Spatiotemporal Trajectory Prediction via State-Conditioned Transition Sampling](https://arxiv.org/abs/2608.14349) | 本文提出一种零参数、无需训练的轨迹预测方法，通过状态条件转移采样和核检索，在数据稀缺时性能优于大型Transformer模型，并能用极少历史数据适应新区域。 |
| [^197] | [Orientation, not magnitude: the causal structure of task-vector interference in merged language models](https://arxiv.org/abs/2608.11797) | 本文通过精确分解和干预实验证明，合并语言模型中的任务向量干扰由方向（而非幅度）因果决定，该方向是前向传递的吸引子，沿此方向擦除可有效去除干扰。 |
| [^198] | [Logarithmic-Free Moment and Generalization Bounds for Uniformly Stable Algorithms](https://arxiv.org/abs/2608.09870) | 该论文去除了一致稳定算法泛化界中多余的对数因子 $\log n$，证明了无对数的矩不等式，从而肯定地回答了Bousquet等人（2020）提出的公开问题。 |
| [^199] | [Coordinate-Residual Physics-Driven Neural Network for Inverse Scattering Imaging](https://arxiv.org/abs/2608.09382) | 提出坐标残差物理驱动神经网络CRPDNN，用归一化空间坐标与残差卷积网络表示复对比度分布，并通过测量场与预测场的一致性约束实现稳定且无需区域选择的加速三维电磁逆散射成像。 |
| [^200] | [Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility](https://arxiv.org/abs/2608.04001) | 该论文将大语言模型的测试时扩展形式化为隐式前缀树上的预算约束推理，系统区分了三种推理机制（单轨迹顺序扩展、叶节点级扩展与前缀级扩展），并主张以完整推理系统作为评估对象，以提升研究结果的可比性与可复现性。 |
| [^201] | [Field-Aware Agent Skill Retrieval](https://arxiv.org/abs/2608.02880) | 该论文将智能体技能建模为结构化的多字段对象，对每个字段独立计算稀疏与稠密相似度并用均匀权重或小型MLP融合，从而显著提升终身学习智能体的技能检索效果。 |
| [^202] | [Locked Evaluation Surfaces: Transfer Failure and Sampling-Depth Entanglement in CRISPRi Perturbation-Effect Prediction](https://arxiv.org/abs/2608.00152) | 该论文在锁定且预注册的评估协议下评估冻结的Geneformer表示，发现其在虚拟细胞挑战赛（VCC）分布内数据上具有显著超越随机特征对照的预测信息量，但在零样本跨筛选迁移中失败，并揭示了迁移失败与采样深度等设计因素之间的纠缠。 |
| [^203] | [S-CEReBrO: Breaking the Memory Barrier in Continuous EEG Monitoring](https://arxiv.org/abs/2607.27913) | S-CEReBrO通过新颖的窗口交替注意力机制将注意力计算分解为固定大小的时空窗口，实现恒定的KV缓存内存占用，可处理比全自注意力长100倍的信号，从而突破了连续脑电监测中的内存瓶颈。 |
| [^204] | [Backspace as a Natural Experiment: An Accelerated Failure Time Model of Selective Post-Error Motor Impairment in Parkinsons Disease](https://arxiv.org/abs/2607.24796) | 本研究将退格键事件作为自然纠错情境，发现错误后停顿时长（而非错误前打字不稳定性）与帕金森病严重程度显著相关，表明被动键盘监测可选择性捕捉PD患者的错误后运动恢复损伤。 |
| [^205] | [Shallower ReLU Network Representations via Exact Linear Algebra](https://arxiv.org/abs/2607.21651) | 本文通过对称性约减后有理数域线性方程组的精确计算机辅助搜索，将两层ReLU网络可精确表示的最大值函数维度从 n≤5 提升至 n≤12，并借助结构化首隐藏层的递归代入得到对数深度的精确表示。 |
| [^206] | [A Classifier That Teaches Itself: Self-Improving, Frozen-gate Training (SIFT) for Dynamic Document Classification](https://arxiv.org/abs/2607.18358) | SIFT提出了一种自改进的动态文档分类服务：用廉价的SPLADE+LightGBM流水线处理分类，仅将低置信度页面交给LLM裁判，其判定结果回流标注语料库持续教导廉价模型，从而免去前期标注工作并让准确率随使用不断提升。 |
| [^207] | [How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?](https://arxiv.org/abs/2607.18114) | 该研究发现大语言模型对谄媚性等线索诱导偏差的敏感性主要源于对齐微调而非预训练，且对齐模型中每种偏差都存在一个可被解码和干预的线性表示方向，可用于恢复无偏答案。 |
| [^208] | [Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories](https://arxiv.org/abs/2607.06648) | 该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。 |
| [^209] | [Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?](https://arxiv.org/abs/2606.31213) | 该论文提出MoralAltDataset数据集，通过在307个二元道德困境中引入折中和重构的替代选项，发现当替代方案可用时人类与15个LLM的道德选择分布均发生显著转变且一致性增强，但存在关键差异——LLM明显偏好GPT-5创作的替代方案，而人类的选择不受创作来源影响，揭示了机器与人类在“想象道德替代方案”能力上的差距。 |
| [^210] | [EchoSonar-R: A Multi-View Reasoning-Enabled Model for Disease Classification and Report Generation in Echocardiography](https://arxiv.org/abs/2606.28164) | EchoSonar-R是一种结合时空视频编码器与结构感知心脏检测器的多视图推理视觉-语言模型，可同时完成超声心动图的多标签疾病分类和报告生成，并通过空间定位的解剖学证据增强可解释性与临床信任度。 |
| [^211] | [Steer, Don't Solve: Training Small Critic Models for Large Code Agents](https://arxiv.org/abs/2606.21811) | 通过训练专门负责高层次规划的小型评论模型（4B/8B）在推理时引导大型编码智能体识别并纠正错误，在SWE-Bench Verified上显著提升多个更大规模编码智能体的解决率（最高提升16.0%）并降低推理成本。 |
| [^212] | [Closing the Operational Gap in Semantic Caching](https://arxiv.org/abs/2606.19719) | 该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。 |
| [^213] | [Explicit Interaction Architectures for Dynamical Learning: A Controlled Study of Structural Inductive Bias](https://arxiv.org/abs/2606.19101) | 该论文提出由有序局部状态调制变换构成的因果循环单元，并通过受控实验检验这种显式设计的交互架构相比通用回声状态网络能否为动态学习提供有用的结构归纳偏置。 |
| [^214] | [Beyond AHI: An Interpretable Causal-Discovery-Guided Framework for Sleep Recovery in Connected Health](https://arxiv.org/abs/2606.18506) | 该论文提出了一个可解释的因果发现引导框架，通过有向无环图学习从多模态PSG数据中推导层次化的睡眠恢复评分（SRS），超越了传统AHI指数的局限，能够更全面地捕捉呼吸负荷、缺氧负荷、睡眠碎片化、睡眠结构和自主神经调节等多个生理域，并可与互联健康技术中的可穿戴设备数据流自然对接。 |
| [^215] | [ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues](https://arxiv.org/abs/2606.18237) | ReproRepo提出利用GitHub上人工提交的议题作为天然监督信号，构建了可规模化的可复现性评估框架，并在1,149篇机器学习论文上验证了LLM智能体无需执行代码即可识别真实复现障碍的能力（最佳智能体可覆盖约90%的论文）。 |
| [^216] | [Learning to Refine Hidden States for Reliable LLM Reasoning](https://arxiv.org/abs/2606.17524) | 提出强化引导的潜在精炼框架ReLAR，通过学习到的深度与动作控制器在解码前自适应地迭代精炼隐藏状态，无需显式思维链即可提升大语言模型推理的准确性、生成质量与稳定性，并大幅降低推理开销。 |
| [^217] | [HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents](https://arxiv.org/abs/2606.16285) | HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。 |
| [^218] | [DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment](https://arxiv.org/abs/2606.07678) | 提出无需训练的数据选择框架DOG-DPO，将偏好对视为模型表示空间中的几何方向，通过分解全局锚定子空间与数据集特有残余子空间并最大化多样性覆盖，为DPO安全对齐筛选出广泛且非冗余的偏好数据子集。 |
| [^219] | [Enabling KV Caching of Shared Prefix for Diffusion Language Models](https://arxiv.org/abs/2606.07571) | 本文提出bicache，首个针对扩散语言模型共享前缀的KV缓存技术，解决了双向注意力下KV动态变化导致传统缓存失效的问题。 |
| [^220] | [RECAP: Regression Evaluation for Continual Adaptation of Prompts](https://arxiv.org/abs/2606.06698) | RECAP是一个在严格“先适应后测试”主动协议下、于约束层面评估提示词优化方法持续学习能力（遗忘、回归、前向迁移）的基准，实验发现现有六种方法在面对动态演变的约束时均无显著改进。 |
| [^221] | [What Do Students Learn? A Feature-Level Analysis of Dark Knowledge](https://arxiv.org/abs/2606.03052) | 本文通过交互张量框架揭示了知识蒸馏通过修剪低频的样本特定特征而起正则化作用，并据此提出无需教师模型的混淆蒸馏方法（CD），利用模型自身演化的混淆模式作为动态软目标，在CIFAR-100上超越现有自蒸馏方法1.2%。 |
| [^222] | [Skill Reuse as Compression in Agentic RL](https://arxiv.org/abs/2605.31509) | 该论文提出基于最小描述长度（MDL）原则的ReuseRL框架，通过从成功轨迹中提取共享技能字典并惩罚难以压缩的特异行为，显著提升了智能体强化学习的分布内外泛化能力。 |
| [^223] | [PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning](https://arxiv.org/abs/2605.29582) | PEARL提出了一种教学对齐的强化学习框架，通过可控学生模拟器解耦认知状态并在多轮师生交互中协调多个教学目标，从而训练出擅长渐进式引导的苏格拉底式辅导智能体。 |
| [^224] | [Three-dimensional Conditional Diffusion Models for Cosmological 21 cm Lightcone Emulation](https://arxiv.org/abs/2605.29016) | 本文系统研究了用于三维宇宙学21厘米光锥模拟的条件扩散模型，并通过受控比较评估了预处理、动态范围压缩、架构深度和训练时长等关键因素对生成质量的影响。 |
| [^225] | [When the Strongest Teacher Is Not the Best Teacher: Student-Centric Answer Selection](https://arxiv.org/abs/2605.26872) | 论文提出SCAS框架，证明最强教师的正确答案未必是学生的最佳训练监督，并通过逐token梯度分解推导出仅需前向计算的高效代理指标，依据学生中心学习成本来选择最适合学生的教师答案。 |
| [^226] | [Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior](https://arxiv.org/abs/2605.26797) | 本文提出潜在循环Transformer（LRT），通过复用前一个token的高层隐藏状态作为循环记忆，在不改变标准注意力机制和KV-cache接口的前提下引入跨token、跨层的信息通路，并设计交错并行训练方法以约2倍理想计算成本实现循环记忆的预训练。 |
| [^227] | [DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking](https://arxiv.org/abs/2605.26087) | 提出了交互式基准测试DiscoverPhysics，通过让大语言模型在物理规律刻意偏离现实的22个模拟世界中设计实验、观察轨迹数据并归纳未知的运动定律，从而将模型真正的科学推理能力与对既有物理知识的记忆区分开来。 |
| [^228] | [LLM-driven design of physics-constrained constitutive models: two agents are better than one](https://arxiv.org/abs/2605.23754) | 提出首个多智能体LLM驱动的本构模型生成框架，由“创建者”智能体提出候选模型、“检查者”智能体依据九项物理约束进行审查并迭代修正，确保生成的本构模型严格遵守物理定律。 |
| [^229] | [Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road](https://arxiv.org/abs/2605.17026) | 本研究通过模拟图分支决策点等受控案例实验，揭示了微调数据中存在多条有效推理路径的“道路岔口”式场景是导致推理模型在SFT后训练中出现覆盖率收缩（pass@k退化）的关键原因。 |
| [^230] | [Universal Approximation of Nonlinear Operators and Their Derivatives](https://arxiv.org/abs/2605.15285) | 本文首次证明了关于 $k$ 阶可微非线性算子及其导数的万能逼近定理，首次将经典万能逼近定理完整推广至无穷维巴拿赫空间与算子学习领域，并由此开启了导数感知算子学习（DIOL）的新方向。 |
| [^231] | [Polarizable atomic multipoles for learning long-range electrostatics](https://arxiv.org/abs/2605.05746) | 该论文提出了一种基于可极化原子多极矩的半局域框架来学习长程静电相互作用，显著提升了机器学习原子间势在离子、极性和界面体系的精度，并且无需直接监督即可涌现出物理上有意义的电响应。 |
| [^232] | [Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding](https://arxiv.org/abs/2605.00865) | 该研究通过严格的泄漏审计基准，发现跨受试者听觉脑电元音解码的证据非常有限，即使最佳模型也仅略高于随机水平且不显著。 |
| [^233] | [D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery](https://arxiv.org/abs/2604.27977) | 本文提出了首个为科学数据驱动发现自动构建的可验证环境数据集D3-Gym，包含565个来自真实科学代码库的任务，其自动评估脚本与人工标注达到87.5%的一致性，且基于其轨迹训练可显著提升Qwen3模型在ScienceAgentBench上的表现。 |
| [^234] | [Reparameterization through Coverings and Topological Weight Priors](https://arxiv.org/abs/2604.23804) | 本文通过覆盖映射推广了变分自编码器的重参数化技巧，使其适用于具有非平凡拓扑的潜在空间，并建立KL散度不等式，使ELBO中的KL项在某些情况下可解析求解。 |
| [^235] | [FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet](https://arxiv.org/abs/2604.22494) | 提出了FedSPDnet框架，通过ProjAvg和RLAvg两种保持Stiefel流形几何结构的聚合策略，实现了基于SPD矩阵的联邦深度学习，在EEG运动想象基准上以更少的通信参数和更强的鲁棒性超越了联邦EEGnet。 |
| [^236] | [The Topological Trouble With Transformers](https://arxiv.org/abs/2604.17121) | 该论文揭示了Transformer纯前馈架构在动态状态追踪上的根本性拓扑缺陷——状态表示随每个新输入被不断推向更深层直至耗尽模型深度，并论证时间上延伸的认知需要从显式思维轨迹回归到基于循环结构的隐式激活动力学。 |
| [^237] | [Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction](https://arxiv.org/abs/2604.16590) | STORM提出了一种单阶段生成式AI数据同化框架，将数据同化重构为基于扩散模型的贝叶斯后验采样，并结合线性复杂度的全局注意力算法，在Frontier上扩展至74,400个GPU、达到6 ExaFLOPs持续吞吐量，并可在34秒内实现32,768成员的大规模集合不确定性量化。 |
| [^238] | [Why Fine-Tuning Encourages Hallucinations and How to Fix It](https://arxiv.org/abs/2604.15574) | 该论文提出一种基于自蒸馏的监督微调方法，通过正则化输出分布漂移，使模型在学习新事实的同时最大限度减少对预训练知识的幻觉，并证明在无需学习新知识时冻结参数组也能在保持任务性能的前提下降低幻觉。 |
| [^239] | [What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal](https://arxiv.org/abs/2604.08524) | 本研究通过多词元激活修补框架对LLM拒绝行为的转向机制进行案例研究，发现不同转向方法在同一层利用功能可互换的回路，且转向向量主要通过注意力机制的OV回路发挥作用而几乎不依赖QK回路。 |
| [^240] | [KV Cache Offloading for Context-Intensive Tasks](https://arxiv.org/abs/2604.08426) | 该论文创建并发布了Text2JSON基准测试，揭示现代KV缓存卸载技术在需要从输入提示中提取大量信息的上下文密集型任务上，会导致Llama 3和Qwen 3模型出现显著的性能下降。 |
| [^241] | [Process-Aware AI for Rainfall-Runoff Modeling: A Mass-Conserving Neural Framework with Hydrological Process Constraints](https://arxiv.org/abs/2603.25093) | 提出了一种质量守恒感知器（MCP）框架，通过在单一存储单元中逐步嵌入有界土壤蓄水、入渗、地表积水、地下水位动态等物理水文过程约束，在保证质量守恒的同时提升了降雨径流模拟的预测精度与物理可解释性。 |
| [^242] | [Rigorous Error Certification for Neural PDE Solvers: From Empirical Residuals to Solution Guarantees](https://arxiv.org/abs/2603.19165) | 该论文的核心贡献是建立了将残差控制与解空间误差联系起来的泛化界，证明当神经逼近位于解空间紧子集内时，残差误差趋于零可保证收敛到真实解，从而为神经偏微分方程求解器提供了严格的误差认证。 |
| [^243] | [Uniform a priori bounds and error analysis for the Adam stochastic gradient descent optimization method](https://arxiv.org/abs/2603.18899) | 本工作的关键贡献是为Adam优化器建立了一致先验界，从而首次为一大类强凸随机优化问题提供了无条件的误差分析，摆脱了以往收敛性分析对“Adam保持一致有界”假设的依赖。 |
| [^244] | [MineDraft: A Framework for Batch Parallel Speculative Decoding](https://arxiv.org/abs/2603.18016) | MineDraft提出一种批量并行投机解码框架，通过同时维护两批请求，将一批的草稿生成与另一批的验证重叠执行，有效隐藏草稿延迟，相比标准投机解码吞吐量最高提升75%、端到端延迟最高降低39%。 |
| [^245] | [SCALE:Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction](https://arxiv.org/abs/2603.17380) | SCALE是一种将细胞表示为无序集合的条件传输模型，无需细胞级配对即可预测扰动后的细胞群体，在基因、化学、发育、免疫扰动及CRISPR数据上均优于现有方法。 |
| [^246] | [RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction](https://arxiv.org/abs/2603.12666) | 本文提出RetroReasoner，一个通过结构化切断理由的监督微调和往返奖励强化学习训练的逆合成推理大语言模型，能够显式模拟化学家的键切断策略思维并验证预测反应物的有效性。 |
| [^247] | [HEAL: Hindsight Entropy-Assisted Learning for Reasoning Distillation](https://arxiv.org/abs/2603.10359) | HEAL 提出了一种无需强化学习的推理蒸馏框架，通过熵动态检测推理断点并注入事后提示来修复失败轨迹，突破了传统拒绝采样造成的“教师天花板”，从而将大型推理模型的推理能力更有效地蒸馏到小模型中。 |
| [^248] | [MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery](https://arxiv.org/abs/2603.03517) | 本文提出MMAI Gym for Science一站式训练框架，通过教会基础模型“分子的语言”，训练出更小规模的液体基础模型（LFM），在分子优化、ADMET预测等药物发现任务上超越了规模大得多的通用或专业模型。 |
| [^249] | [Inverse Reconstruction of Shock Time Series from Shock Response Spectrum Curves using Machine Learning](https://arxiv.org/abs/2603.03229) | 提出了一种条件变分自编码器（CVAE）模型，学习从冲击响应谱到加速度时间序列的数据驱动逆映射，无需迭代优化即可生成高谱保真度的时域信号。 |
| [^250] | [Channel-Adaptive Edge AI: Maximizing Inference Throughput by Adapting Computational Complexity to Channel States](https://arxiv.org/abs/2603.03146) | 本文提出了一个可处理的端到端推理精度分析模型，并据此设计了信道自适应AI算法，通过根据信道状态动态调整模型计算复杂度（利用早退机制），在时延和精度约束下最大化边缘推理吞吐量。 |
| [^251] | [Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation](https://arxiv.org/abs/2602.23188) | 该论文提出一种基于变分自编码器与Transformer的参数化降阶模型高效再训练策略，仅利用稀疏观测数据和少量计算时间即可将模型实时自适应到新的流动工况，精度接近完全再训练。 |
| [^252] | [Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning](https://arxiv.org/abs/2602.18493) | 该论文提出统一记忆智能体（UMA），通过任务分层GRPO算法端到端训练单一策略来维护可复用的结构化外部记忆库，并配套提出Ledger-QA诊断基准，显著提升了长上下文推理中跨会话状态跟踪与问答的性能。 |
| [^253] | [Ontology-Guided Neuro-Symbolic Inference: Grounding Language Models with Mathematical Domain Knowledge](https://arxiv.org/abs/2602.17826) | 该论文提出一种结合OpenMath本体、混合检索与交叉编码器重排序的神经符号流水线，将数学领域知识注入语言模型提示中，实验表明高质量检索的本体上下文能提升模型在MATH基准上的表现，但不相关上下文会损害性能。 |
| [^254] | [Is Knowledge Distillation Actually Greener? A Case Study in Machine Translation](https://arxiv.org/abs/2602.09691) | 该研究首次借助机器学习生命周期评估工具，从环境成本角度系统评估机器翻译中的知识蒸馏方法，发现摊销蒸馏成本所需的部署量取决于服务方式，且在批处理下可能变化数个数量级。 |
| [^255] | [Persistent Entropy as a Detector of Phase Transitions](https://arxiv.org/abs/2602.09058) | 本文建立了与模型无关的理论定理，通过识别持续权重中的“分散-凝聚”机制并推导出两状态间熵差的显式高概率下界，首次为利用持续熵检测相变提供了严格的理论保证，并据此证明卷积网络学习滤波器的环形组织源于一次尖锐的拓扑相变。 |
| [^256] | [Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging](https://arxiv.org/abs/2601.23276) | 本文提出了一种基于物理的CCD噪声合成框架，通过建模光子散粒噪声、暗电流、读出效应等多种噪声来源，并利用未配准曝光叠加生成高信噪比基础图像，从而构建大量成对训练数据，实现天文图像的监督学习去噪。 |
| [^257] | [Breaking the Reasoning Horizon in Entity Alignment Foundation Models](https://arxiv.org/abs/2601.21174) | 提出了一种由并行编码策略驱动的实体对齐基础模型，利用种子对齐对作为局部锚点进行锚点条件消息传递，突破了传统模型在稀疏异构知识图谱上捕获长距离依赖的“推理视界”限制，实现了无需重新训练即可对齐未见知识图谱的能力。 |
| [^258] | [FloydNet: A Learning Paradigm for Global Relational Reasoning](https://arxiv.org/abs/2601.19094) | 提出FloydNet与关键注意力机制（PA），借鉴Floyd–Warshall的“配对-枢轴”结构，通过维护有序对状态并在枢轴上聚合候选关系实现全局关系推理，并推广为支持有序k元组的k-FloydNet框架，其图判别能力与对应的WL同构测试相当。 |
| [^259] | [Auditing Frozen-Encoder Anomaly Detection Across Mechanical Systems: Representation Provenance, Calibration, and Protocol Effects](https://arxiv.org/abs/2601.11415) | 本论文对冻结编码器异常检测实验进行了可复现性审计，发现虽然数值判别结果可以复现，但其归因于干涉测量预训练的因果声明不被支持，作者因此撤回了该声明。 |
| [^260] | [DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems](https://arxiv.org/abs/2601.06853) | 该论文提出DAGGER方法，将含干扰信息的数学问题求解重构为显式建模干扰节点的可执行计算图生成，有效缓解了思维链推理在无关信息干扰下的严重性能退化。 |
| [^261] | [Hidden State Poisoning Attacks against Mamba-based Language Models](https://arxiv.org/abs/2601.01972) | 该论文首次揭示了针对Mamba等状态空间语言模型的隐状态投毒攻击（HiSPA）——特定短输入短语可不可逆地覆盖模型隐藏状态导致部分失忆，并提出RoBench-25基准证实了包括520亿参数的Jamba混合模型在内的SSMs对此类攻击的脆弱性，而纯Transformer模型则不受影响。 |
| [^262] | [Modeling Information Blackouts in Missing Not-At-Random Time Series Data](https://arxiv.org/abs/2601.01480) | 该论文提出了一种感知非随机缺失（MNAR）的潜在状态空间模型，用于建模交通传感器网络中的连续信息中断，证明当缺失机制依赖于潜在交通状态时，考虑这种依赖关系可显著提升数据插补精度与缺失检测性能。 |
| [^263] | [CADKnitter: Compositional CAD Generation from Text and Geometry Guidance](https://arxiv.org/abs/2512.11199) | CADKnitter是一个组合式CAD生成框架，通过几何与文本双重引导的扩散采样，能够根据给定CAD模型的几何约束和文本提示的语义约束生成与之互补的新CAD零件。 |
| [^264] | [Probabilistic Multi-Agent Aircraft Landing Time Prediction](https://arxiv.org/abs/2512.08281) | 提出一个概率多智能体着陆时间预测框架，将多架飞机的着陆时间以概率分布形式输出，同时兼顾轨迹不确定性及空域中飞机间的交互影响。 |
| [^265] | [Freeze, Diffuse, Decode: Geometry-Aware Adaptation of Pretrained Transformer Embeddings for Antimicrobial Peptide Design](https://arxiv.org/abs/2511.23120) | 提出了FDD（冻结、扩散、解码）框架，通过沿冻结嵌入的内在流形传播监督信号，在保留预训练Transformer嵌入几何结构的前提下实现几何感知的任务适配，并在抗菌肽设计中生成低维、可预测、可解释的表示，支持性质预测、检索与潜空间插值。 |
| [^266] | [3D-Consistent Multi-View Editing by Correspondence Guidance](https://arxiv.org/abs/2511.22228) | 提出了一种无需训练的引导框架，通过引入一致性损失确保对应点在编辑后保持相似，从而在去噪过程中实现几何和光度上3D一致的多视图图像编辑。 |
| [^267] | [Iterative GRPO: Batch-Online Policy Iteration for Multi-Turn RL via Single-Turn RLHF](https://arxiv.org/abs/2511.21638) | 提出Iterative GRPO方法，利用批在线部署模式收集的交互数据，将经典的近似策略迭代算法与单轮RLHF相结合，从而无需在训练循环中接入真实或模拟用户即可实现多轮强化学习。 |
| [^268] | [The Alexander-Hirschowitz theorem for neurovarieties](https://arxiv.org/abs/2511.19703) | 本文通过独立的几何方法证明了激活次数满足线性界 d_i ≥ 2n_i−1 时多项式神经网络的神经簇对任意输出数都是非亏缺的，并进一步证明了多输出架构在相同次数界下的全局可辨识性。 |
| [^269] | [SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning](https://arxiv.org/abs/2511.09681) | SEBA提出了一种针对视觉强化学习的样本高效黑盒攻击框架，通过结合影子Q模型、生成对抗网络和世界模型，以极少的真实环境查询实现对基于图像的连续控制智能体的有效对抗攻击。 |
| [^270] | [Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement](https://arxiv.org/abs/2511.01706) | 该论文提出一种新颖的秩-2投影子空间来更准确地解缠大语言模型中参数化知识与情境知识的贡献，并首次实现了对自然语言解释更长生成序列中知识交互的多步分析。 |
| [^271] | [Can machines think efficiently?](https://arxiv.org/abs/2510.26954) | 该论文提出在图灵测试中引入能量消耗约束，从效率视角重新评估机器智能，从而为智能评估提供了原测试所缺乏的可测量的实际标准。 |
| [^272] | [Nonlinear Dynamics In Optimization Landscape of Shallow Neural Networks with Tunable Leaky ReLU](https://arxiv.org/abs/2510.25060) | 本文基于等变梯度度为采用可调泄漏ReLU的浅层神经网络建立了分岔分析理论框架，揭示了临界点从全局最小值处的分岔与神经元数量无关，且在工程区间 (0,1) 内全局最小值保持稳定、不发生对称性破缺。 |
| [^273] | [Compositional Machine Design as Program Synthesis with LLMs](https://arxiv.org/abs/2510.14980) | 该论文提出将机器设计视为一种以物理模拟验证为依据的程序合成新任务——组合式机器设计，并构建了基于游戏《Besiege》的测试平台BesiegeField，用于评测大语言模型在多种工作流下组合标准部件设计机器的能力。 |
| [^274] | [Silence is Golden: Mitigating Hallucinations in Large Audio-Language Models via Layer-Weighted Vector Steering](https://arxiv.org/abs/2510.12851) | 本文首次将向量引导技术应用于音频领域，提出无需训练的层加权向量引导方法（LWVS），利用沉默基线对比与关键层强化，在显著缓解大型音频-语言模型幻觉的同时保持甚至提升通用音频理解能力。 |
| [^275] | [Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity](https://arxiv.org/abs/2510.03899) | 该论文提出了公平最小标注（FML）问题——设计最小成本的时序边激活方案以保障网络中各节点组的公平可达性，证明该问题是NP难且难以近似求解的，并给出了单终端情形等价于有根覆盖斯坦纳问题的结构性刻画。 |
| [^276] | [Performance-Efficiency Tradeoffs in Transformers: An Approximation Theory Perspective](https://arxiv.org/abs/2510.03784) | 本文从逼近理论视角刻画了Transformer中注意力头数量与头维度在固定参数预算下的权衡，发现并证明了softmax激活的饱和行为，表明较深的层可以用更小的头维度实现高效运行。 |
| [^277] | [Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models](https://arxiv.org/abs/2509.25050) | 本文从理论上揭示DDPO本质是带噪声目标的隐式分数/流匹配，并提出优势加权匹配（AWM），通过以优势加权分数/流匹配损失，使扩散模型的强化学习与预训练目标对齐，从而降低方差并加速收敛。 |
| [^278] | [A Compositional Kernel Model for Feature Learning](https://arxiv.org/abs/2509.14158) | 本文提出一种组合核岭回归模型，证明其能在变量选择中恢复相关变量并消除高斯噪声变量，且核心发现是ℓ₁型核（如拉普拉斯核）在驻点处能恢复非线性特征，而高斯核仅能恢复线性特征。 |
| [^279] | [Recurrent State Encoders for Efficient Neural Combinatorial Optimization](https://arxiv.org/abs/2509.05084) | 提出一种循环状态编码器，通过复用先前步骤的状态嵌入，使神经组合优化模型在层数减少3倍的情况下仍能达到相当或更优的性能，显著降低推理延迟。 |
| [^280] | [Integrated Noise and Safety Management in UAM via A Unified Reinforcement Learning Framework](https://arxiv.org/abs/2508.16440) | 该论文提出一个统一去中心化的强化学习空中交通管理框架，使城市空中交通飞行器在多层空域中通过学习高度调整策略同时优化噪声暴露与安全间隔，并揭示了高密度交通下安全、噪声与能耗之间的权衡关系。 |
| [^281] | [SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance](https://arxiv.org/abs/2508.11857) | SupraTok是一种跨越空白符边界的创新分词器，通过熵筛选、PMI引导的课程训练和多语言处理三大模块，在压缩率上比标准BPE提升17.5%，并比SuperBPE训练快2.1倍。 |
| [^282] | [FlexP-SFT: A Flexible Aggregation-Free Framework for On-Device Personalized Split Federated Fine-Tuning of LLMs](https://arxiv.org/abs/2508.10349) | FlexP-SFT是一种面向大语言模型的无需聚合的个性化分割联邦微调框架，通过消除客户端聚合过程解决通信瓶颈和掉队者问题，并借助层级灵活对齐策略在无全局同步的情况下平衡个性化与泛化能力。 |
| [^283] | [BiasGym: A Simple and Generalizable Framework for Analyzing and Removing Biases through Injection](https://arxiv.org/abs/2508.08855) | 提出BiasGym框架，通过在冻结的LLM中注入特定偏见信号，再利用这些信号定位并抑制或引导导致偏见行为的模型组件，实现偏见的可靠分析与消除。 |
| [^284] | [Unsupervised Partner Design Enables Robust Ad-hoc Teamwork](https://arxiv.org/abs/2508.06336) | 提出无监督伙伴设计（UPD），通过即时生成训练伙伴并基于可学习性准则自适应选择，无需预训练伙伴种群或手动调参即可实现鲁棒的临时团队协作，在多个基准任务和人机用户研究中均表现出卓越性能。 |
| [^285] | [Towards Provable and Scalable Training of Quantized Neural Networks with Ising Optimization](https://arxiv.org/abs/2506.18240) | 该论文提出一个具有可证明保证的精确二次约束二元优化（QCBO）框架，将量化神经网络训练编译为具有零松弛间隙的完全正凸优化问题，并通过逐样本分解下界优化（DLBO）将伊辛求解规模从数据集级别降至单样本级别，从而实现对量化神经网络可证明且可扩展的训练。 |
| [^286] | [On the Existence of Consistent Adversarial Attacks in High-Dimensional Linear Classification](https://arxiv.org/abs/2506.12454) | 本文提出了一种新的误差度量来区分真正的一致性对抗攻击（即保持真实标签不变的扰动）与因数据有限或模型能力不足导致的普通误分类，并通过精确的渐近理论分析证明，随着模型过参数化程度的提高，其对标签保持扰动的脆弱性会不断增大。 |
| [^287] | [Efficient Learning of Balanced Signed Graphs via Sparse Linear Programming](https://arxiv.org/abs/2506.01826) | 提出了一种基于稀疏线性规划的高效方法，可直接从数据中学习平衡符号图的拉普拉斯矩阵，使其能够复用为正图设计的谱滤波工具。 |
| [^288] | [Online simultaneous inference for quantiles via smoothed stochastic gradient descent](https://arxiv.org/abs/2505.13299) | 本文提出一种平滑随机梯度下降方法用于流数据的在线分位数估计，其估计量在每次迭代中关于分位数水平单调，并借助一致Bahadur表示与布朗桥最大值的高斯近似，实现了维度随样本量指数增长时跨坐标与分位数水平的在线同时统计推断。 |
| [^289] | [Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms](https://arxiv.org/abs/2502.20115) | 本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。 |
| [^290] | [Generalization Bounds for Markov Algorithms through Entropy Flow Computations](https://arxiv.org/abs/2502.07584) | 该论文提出新的技术工具，将熵流方法的适用范围从特定的噪声和算法结构（如朗之万动力学）扩展到所有迭代动力学由时齐马尔可夫过程支配的学习算法，从而为这一广泛类别的算法建立泛化界。 |
| [^291] | [QABBA: Error-Guaranteed Symbolic Time-Series Compression via Integer-Quantized Aggregation](https://arxiv.org/abs/2411.15209) | 提出QABBA，通过量化符号中心实现ABBA的整数化压缩，在保证重建质量的同时提供严格的误差界限。 |
| [^292] | [Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies](https://arxiv.org/abs/2408.12845) | 针对物品数量多而副本少的在线公平分配难题，本文创新性地假设效用是物品-智能体特征的未知函数，并将其建模为上下文老虎机问题，从而克服了无法准确估计所有物品-智能体对效用的局限。 |
| [^293] | [FedReview: A Review Mechanism for Rejecting Poisoned Updates in Federated Learning](https://arxiv.org/abs/2402.16934) | 提出了FedReview机制，通过随机分配评审员客户端来识别和拒绝联邦学习中的潜在毒化更新，并采用多数表决机制来整合排名并移除这些更新。 |
| [^294] | [Building Expressive and Tractable Probabilistic Generative Models: A Review](https://arxiv.org/abs/2402.00759) | 本文综述了富有表现力和可处理的概率生成建模领域的进展和技术，并重点关注了概率电路。文章提供了关于表达能力和可处理性之间权衡的统一视角，并说明了设计原则和算法扩展，成功地构建了富有表现力和高效的概率电路。此外，文章还讨论了最新的深度和混合概率电路研究，并概述了未来研究的挑战和开放性问题。 |
| [^295] | [Deep learning based numerical approximation algorithms for stochastic partial differential equations](https://arxiv.org/abs/2012.01194) | 本文提出一种基于深度学习的随机偏微分方程逼近算法，通过神经网络沿噪声轨迹逼近SPDE解并估计其经验分布，在随机热方程、Black-Scholes方程和Zakai方程等测试中实现了高达100维空间下的快速精确求解。 |

# 详细

[^1]: 超越分数：理解摘要评估中“大模型作为评判者”的内部机制

    Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation

    [https://arxiv.org/abs/2609.01604](https://arxiv.org/abs/2609.01604)

    该论文通过八种攻击扰动分类法与因果追踪、注意力头敲除等可解释性技术，首次从机制层面揭示LLM评估器（Themis与Prometheus）在摘要评分时采用两阶段内部流程：第15层以下注意力执行局部错误比较并路由信号，其上由MLP级联完成信号整合。

    

    基于大语言模型（LLM）的自然语言生成（NLG）质量评估器已被广泛用作评分工具和自动化训练信号，然而它们给出评分的内部过程仍鲜为人知。我们从机制层面对这一过程展开研究：提出了一个覆盖NLG质量中可读性（Readability）与充分性（Adequacy）两个维度的八种攻击扰动分类体系；构建了一个生成流程，可产生错误强度可控、并附带显式词元级修改映射的“干净-受损”成对摘要；并设计了一组包含四个实验的测试组合，运用因果追踪（causal tracing）、logit-lens词表投影和注意力头敲除（attention-head knockout）等技术，对Themis（Llama-3-8B）和Prometheus（Mistral-7B）两个评估模型进行分析。结果表明，两个评估器都实现了一条结构化、连贯的两阶段评估流水线：在第15层以下，注意力机制执行局部错误比较，并将结果路由至最终输入位置；在第15层之上，MLP级联整合该信号并……

    arXiv:2609.01604v1 Announce Type: new  Abstract: LLM-based evaluators of natural language generation (NLG) quality are widely deployed as scoring tools and as automated training signals, yet the internal procedure by which they assign a rating remains poorly understood. We investigate this procedure mechanistically through an eight-attack perturbation taxonomy across the Readability and Adequacy dimensions of NLG quality, a generation pipeline that produces paired clean and corrupt summaries with controlled error intensity and explicit token-level modification maps, and a four-experiment battery of causal tracing, logit-lens vocabulary projection, and attention-head knockout applied to Themis (Llama-3-8B) and Prometheus (Mistral-7B). Both evaluators implement a structured, coherent evaluation pipeline operating in two stages: below layer 15, attention performs local error comparison and routes the result to the final input position; above it, the MLP cascade integrates the signal and w
    
[^2]: Facet-0：面向接触密集型精细操作的机器人基础模型

    Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation

    [https://arxiv.org/abs/2609.01596](https://arxiv.org/abs/2609.01596)

    Facet-0通过联合建模“动作-力旋量”来预测并评估动作的接触后果，结合多模态表征学习与强化学习后训练，实现了亚毫米级公差的接触密集型精细装配操作。

    

    现实世界中公差在亚毫米级别的机器人装配，要求具备空间精度、柔顺交互能力以及对接触失败的鲁棒性。我们提出了Facet-0，一个能够预测并评估其动作所带来接触后果的机器人基础模型。Facet-0围绕联合的“动作-力旋量”提议，统一了多模态表征学习与强化学习（RL）后训练：将因果力旋量历史与视觉-语言语义及运动学状态对齐，并通过流匹配生成每个动作块及其预期引发的未来腕部力旋量曲线。部署过程中的rollout数据用于训练一个分布式的“动作-力旋量评论家”，以区分任务进度相似但接触结果不同的动作；同时，相位感知奖励和接触选择性信用分配将策略改进集中于决定性的交互上。为适应特定零件的动力学特性，一个轻量级的有界执行器复用冻结的表征……

    arXiv:2609.01596v1 Announce Type: cross  Abstract: Real-world robotic assembly at sub-millimeter tolerances demands spatial precision, compliant interaction, and robustness to contact failures. We present Facet-0, a robotic foundation model that predicts and values the contact consequences of its actions. Facet-0 unifies multimodal representation learning and reinforcement learning (RL) post-training around a joint action-wrench proposal: a causal wrench history is aligned with vision-language semantics and kinematic state, and flow matching generates each action chunk together with the future wrist-wrench profile it is expected to induce. Deployment rollouts train a distributional Action-Wrench Critic to distinguish motions with similar task progress but different contact outcomes, while phase-aware rewards and contact-selective credit concentrate policy improvement on decisive interactions. To accommodate part-specific dynamics, a lightweight bounded actor reuses the frozen represent
    
[^3]: 大语言模型中量化损伤的结构：为什么下一个比特应该被全局分配

    The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally

    [https://arxiv.org/abs/2609.01587](https://arxiv.org/abs/2609.01587)

    该研究通过因果混合精度干预实验发现，LLM的量化损伤是弥散分布的而非集中于特定电路、计算位置或权重统计，因此在匹配精度预算下，将额外比特全局用于更精细的量化粒度比局部修复少数层更有效。

    

    训练后量化（PTQ）被广泛用于降低大语言模型（LLM）的服务成本，但其精度损失并不均匀，且通常需要针对每个模型单独调优。我们研究了量化损伤发生在何处，以及如何分配少量额外的精度预算。以因果混合精度干预作为基准真值（依次将每一层提升至8比特并测量其所能恢复的精度），我们在4个架构家族的9个开源权重模型上测试了3个直观假设：量化损伤存在于任务电路中、存在于模型计算发生之处，或存在于权重统计特性之中。然而，这些假设都无法预测哪些层会从恢复的精度中受益。相反，恢复是弥散性的：在9个模型中有8个，恢复75%的精度差距大约需要一半的层；唯一的例外是Qwen3-8B，其恢复高度集中。在匹配的精度预算下，将预算全局用于更精细的量化粒度，优于局部修复最具可恢复性的层。

    arXiv:2609.01587v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is widely used to reduce the cost of serving large language models (LLMs), but its accuracy cost is uneven and is often tuned per model. We study where quantization damage occurs and how to allocate a small additional precision budget. Using causal mixed-precision intervention as ground truth (raise each layer to 8-bit in turn and measure the accuracy it recovers) across 9 open-weight models in 4 architecture families, we test 3 intuitive hypotheses: that quantization damage lives in task circuits, where the model computes, or in weight statistics. None of them predicts which layers benefit from restored precision. Recovery is instead diffuse: for 8 of 9 models, recovering 75% of the gap takes roughly half the layers; the lone exception, Qwen3-8B, is sharply concentrated. At a matched precision budget, spending it globally on finer quantization granularity beats locally repairing the most recoverable la
    
[^4]: 从小型到大型语言模型的近最优SFT-RL标注预算分配扩展

    Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs

    [https://arxiv.org/abs/2609.01573](https://arxiv.org/abs/2609.01573)

    该论文提出“近最优区域”框架来分配SFT-RL标注预算，发现该区域宽广且随模型规模增大而扩大，并能从小型代理模型可靠迁移到大型目标模型，因此小规模代理实验即可替代在大模型上的穷尽式预算搜索。

    

    在大语言模型（LLM）后训练期间，如何在监督微调（SFT）和强化学习（RL）之间分配固定的标注预算仍是一个悬而未决的问题。现有工作仅刻画了宽泛的趋势（例如，在低数据场景下SFT占主导地位），缺乏有原则的分配框架，也没有考察最优比例能否在不同模型规模之间迁移。我们从近最优性的角度来构建这一问题：不再追求单一的SFT-RL最优比例，而是刻画“近最优区域”，即在峰值性能指定容差范围内的所有分配方案集合。实证研究表明，即使容差很小（2-10%），该区域也很宽，且随模型规模增大而变宽，并能可靠地从小型代理模型迁移到大型目标模型。由此得出一种实用策略：只需进行小型代理模型实验即可确定可迁移的近最优区域，从而省去穷尽式的大规模搜索。我们的结果在多种设置下保持一致。

    arXiv:2609.01573v1 Announce Type: cross  Abstract: How to divide a fixed annotation budget between supervised fine-tuning (SFT) and reinforcement learning (RL) during LLM post-training remains an open problem. Existing work characterizes only broad trends (e.g., SFT dominates in low-data regimes), lacks a principled allocation framework, and does not examine whether the optimal ratio transfers across model sizes. We frame this problem in terms of near-optimality: rather than seeking a single optimal SFT-RL ratio, we characterize the near-optimal region, the set of allocations within a specified tolerance of peak performance. Empirically, this region is wide even for small tolerances (2-10%), widens with model scale, and transfers reliably from small proxy models to large target models. This yields a practical strategy: small proxy-model experiments suffice to identify a transferable near-optimal region, eliminating the need for exhaustive large-scale search. Our results hold consistent
    
[^5]: 基于熵的选择性智能体引导：从不完美的视觉语言模型教师中学习自主策略

    Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers

    [https://arxiv.org/abs/2609.01567](https://arxiv.org/abs/2609.01567)

    该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。

    

    视觉语言模型为交互式决策提供了有用的先验知识，但直接将其用作策略既昂贵又脆弱：它们必须在每一步都被查询，无法通过环境交互得到改进，并且可能重复系统性错误。我们研究如何从一个在线、昂贵、不完美但具有信息量的视觉语言模型教师中学习一个廉价的自主策略。我们提出了SAGE（基于熵的选择性智能体引导），这是一个仅在学习者不确定时才查询视觉语言模型的框架，它在训练期间执行其建议的动作，并将引导蒸馏到一个轻量级的强化学习（RL）策略中。由于视觉语言模型的建议并不总是可靠的，SAGE可以使用由环境得出的优势来对教师动作蒸馏进行加权，而不是将所有建议视为同样有用。在稀疏奖励的视觉推理和导航任务中，SAGE学习到的策略在评估时无需视觉语言模型引导即可自主行动，并改进了……

    arXiv:2609.01567v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) provide useful priors for interactive decision-making, but using them directly as policies is expensive and brittle: they must be queried at every step, do not improve from environment interaction, and can repeat systematic errors. We study how to learn a cheap autonomous policy from an online, expensive, and imperfect but informative VLM teacher. We propose SAGE (Selective Agent Guidance via Entropy), a framework that queries a VLM only when the learner is uncertain, executes the suggested action during training, and distills guidance into a lightweight Reinforcement Learning (RL) policy. Because VLM advice is not always reliable, SAGE can weight teacher-action distillation using environment-derived advantages rather than treating all suggestions as equally useful. Across sparse-reward visual reasoning and navigation tasks, SAGE learns policies that act without VLM guidance at evaluation time and improves o
    
[^6]: 梯度更新失配：重新思考物理信息神经网络的无冲突训练

    Gradient-Update Mismatch: Rethinking Conflict-Free Training of Physics-Informed Neural Networks

    [https://arxiv.org/abs/2609.01558](https://arxiv.org/abs/2609.01558)

    该论文揭示了“梯度更新失配”问题：梯度手术构造的无冲突方向经过现代优化器（如历史状态、自适应缩放、预条件化、解耦权重衰减等机制）的变换后可能失去无冲突性质，从而为物理信息神经网络的无冲突训练提供了新的认识视角。

    

    训练物理信息神经网络（PINNs）需要联合优化物理残差项和初始/边界条件损失项，而这些损失项常常会产生相互冲突的梯度。梯度手术方法通过从各损失项特定的梯度构造方向，在优化器变换之前减少冲突来缓解这一问题。然而，即使构造出的方向是无冲突的，这一性质在经过优化器变换后也可能无法保持。设 $a_t$ 表示梯度手术所构造的方向，$u_t$ 表示优化器提出的更新方向，$\mathcal{C}_t$ 表示由各损失项特定梯度所诱导的无冲突锥。我们证明，现代优化器会通过历史状态、自适应缩放、预条件化或解耦权重衰减等机制对 $a_t$ 进行变换，因此 $a_t \in \mathcal{C}_t$ 通常并不意味着 $u_t \in \mathcal{C}_t$。我们将这种由优化器引起的无冲突性差异……（原文摘要在此处截断）

    arXiv:2609.01558v1 Announce Type: new  Abstract: Training Physics-Informed Neural Networks (PINNs) requires jointly optimizing physics residual and initial/boundary condition loss terms, which often induce conflicting gradients. Gradient surgery methods mitigate this issue by constructing directions from loss-specific gradients to reduce conflict before optimizer transformation. However, even when the constructed direction is conflict-free, this property may not be preserved after optimizer transformation. Let $a_t$ denote the direction constructed by gradient surgery, $u_t$ the optimizer proposal, and $\mathcal{C}_t$ the conflict-free cone induced by the loss-specific gradients. We show that modern optimizers can transform $a_t$ through mechanisms such as historical state, adaptive scaling, preconditioning, or decoupled weight decay, so $a_t \in \mathcal{C}_t$ does not generally imply $u_t \in \mathcal{C}_t$. We refer to this optimizer-induced discrepancy in conflict-freeness between 
    
[^7]: 检索到了却排名不对：从数学到智能体轨迹的结构化检索中的表层形式偏差

    Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories

    [https://arxiv.org/abs/2609.01556](https://arxiv.org/abs/2609.01556)

    该研究在竞赛数学与具身智能体轨迹两个领域以统一协议评测刻意分离表层形式与语义结构的嵌入检索，发现主流嵌入模型存在严重的表层形式（字面词汇）偏差：在结构相同但措辞伪装最重的任务上Hit@1跌至0.0%，未命中时胜出者几乎总是与查询词汇更相似的条目，表明当前嵌入检索锚定于字面文本而非深层结构。

    

    我们在刻意将表层形式与含义分离的设定下评估嵌入检索：在统一协议下，于两个互不相关的领域中检索共享底层结构但措辞不同的条目——竞赛数学与具身智能体轨迹（基于ALFWorld衍生；118个查询，336条轨迹）。在数学领域，失败是彻底的：在伪装最重的层级上，两个生产级嵌入模型的严格Hit@1均为0.0%（自助法95%置信区间[0.0, 0.0]），而正确条目几乎总能进入前10名；并且在95.2%至99.8%的未命中情形中，胜出者与查询的词汇相似度高于正确答案。在轨迹领域，表层变化只是附带性的，当正确答案必须涉及不同物体时，同样的模型表现落在超几何随机水平或其附近；而一旦要求正确答案在物体与容器上都必须不同，三个嵌入模型的表现均低于随机水平：检索锚定于字面文……（原文摘要到此截断）

    arXiv:2609.01556v1 Announce Type: cross  Abstract: We evaluate embedding retrieval where surface form and meaning are pulled apart on purpose: retrieving items that share underlying structure but not wording, in two unrelated domains under one protocol, competition mathematics (MathNet-Retrieve; 500 queries, 117,088-item corpus) and embodied-agent trajectories (ALFWorld-derived; 118 queries, 336 trajectories). In mathematics the failure is complete: strict Hit@1 at the heaviest disguise tier is 0.0% for both production embedders (bootstrap 95% CI [0.0, 0.0]) while the correct item sits in the top 10 nearly always, and in 95.2 to 99.8% of misses the winner is more lexically similar to the query than the correct answer. In trajectories, where surface variation is incidental, the same models land at or near hypergeometric chance when gold must involve a different object, and below chance for all three embedders once gold must differ in object and receptacle: retrieval anchors on literal t
    
[^8]: 大语言模型能否在真实世界与平行世界中发现科学定律？

    Can LLMs Discover Scientific Laws in Real and Parallel Worlds?

    [https://arxiv.org/abs/2609.01552](https://arxiv.org/abs/2609.01552)

    该论文提出了基于已发表研究和真实科学数据构建的科学定律发现基准SCILAWS-BENCH（涵盖118个问题、291个候选定律、约800万真实数据点和六个学科），并采用真实世界与平行世界两种互补设置，以严格评估大语言模型能否真正发现科学定律。

    

    科学方程的发现长期以来一直是科学进步的核心，它通过在科学约束下进行假设生成、观测检验和修正的迭代循环来推进。随着大语言模型（LLM）能力的提升及其在“AI for Science”中作用的扩大，它们能否真正发现科学定律、以及应如何评估这种能力，仍然是一个悬而未决的问题。然而，现有的评估方式往往要么通过合成场景简化了发现过程，要么复用了LLM可能早已熟悉的已发表科学目标。因此，我们提出了SCILAWS-BENCH，一个基于已发表研究和真实科学数据构建的科学定律发现基准。该基准包含来自381篇科学论文的118个问题，涵盖291个候选定律以及横跨六个科学学科的约800万个真实数据点。每个问题均在两种互补的设置下进行实例化：(1) SCILAWS-REAL要求模型从……（摘要原文在此处截断）

    arXiv:2609.01552v1 Announce Type: new  Abstract: Scientific equation discovery has long been central to scientific progress, proceeding through iterative cycles of hypothesis generation, observational testing, and refinement under scientific constraints. As LLM capabilities advance and their role in AI for Science expands, it remains an open problem whether they can genuinely discover scientific laws and how this ability should be evaluated. Existing evaluations, however, often either simplify discovery through synthetic settings or reuse published targets that may already be familiar to LLMs. We therefore introduce SCILAWS-BENCH, a benchmark for scientific law discovery built from published research and real scientific data. It comprises 118 problems drawn from 381 scientific papers, covering 291 candidate laws and roughly 8M real data points across six scientific disciplines. Each problem is instantiated in two complementary settings: (1) SCILAWS-REAL asks models to propose laws from
    
[^9]: 用于网络压缩的可复用神经基底的数学理论

    A Mathematical Theory of Reusable Neural Bases for Network Compression

    [https://arxiv.org/abs/2609.01550](https://arxiv.org/abs/2609.01550)

    该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。

    

    随着大型AI模型在各类应用中日益普及，内存成本已成为训练和推理中的关键瓶颈。为缓解这一问题，我们提出了线性可复用神经基底架构（LRNBA），这是一种旨在提高参数效率并降低内存成本的新型框架。受循环神经网络（RNN）设计的启发，我们方法的核心思想是将每个网络块表示为共享神经基底集合的线性组合，从而在保持稳定训练的同时实现高度的网络压缩率。所提出的架构允许在相同的参数预算下构建显著更宽和更深的网络。大量实验表明，我们的模型与经典架构相比实现了相当甚至更快的收敛速度和更低的损失，同时保持了稳定的训练动态。

    arXiv:2609.01550v1 Announce Type: cross  Abstract: As large AI models become increasingly prevalent across a wide range of applications, memory cost has become a critical bottleneck in both training and inference. To mitigate this issue, we introduce the Linear Reusable Neural Bases Architecture (LRNBA), a novel framework aimed at improving parameter efficiency and reducing memory cost. Inspired by recurrent neural network (RNN) designs, the core idea of our approach is to represent each network block as a linear combination of a shared set of neural bases, thereby enjoying highly network compression rate while maintaining stable training. The proposed architecture allows for the construction of significantly wider and deeper networks under the same parameter budget. Extensive experiments demonstrate that our model achieves comparable or even faster convergence and lower loss than classical architectures, while maintaining stable training dynamics.
    
[^10]: 用于认知诊断中Q矩阵估计的量子稀疏自编码器

    Quantum Sparse Autoencoders for Q-Matrix Estimation in Cognitive Diagnosis

    [https://arxiv.org/abs/2609.01537](https://arxiv.org/abs/2609.01537)

    该论文首次将量子机器学习引入认知诊断领域，提出量子稀疏自编码器（QSAE），通过量子电路压缩学生作答数据以估计Q矩阵，并在模拟和真实评估数据集上展现出与经典自编码器互补的性能优势。

    

    Q矩阵在教育数据挖掘（EDM）的认知诊断中扮演着核心角色，它明确规定了每个评估题目所要求的潜在技能。当评估涉及大量相互关联的技能，且真实作答模式偏离理想化的生成假设时，数据驱动的Q矩阵估计仍然充满挑战。我们提出了一种用于Q矩阵估计的新型量子稀疏自编码器（QSAE），据我们所知，这是量子机器学习（QML）在认知诊断领域的首次应用。总体而言，QSAE通过编码器将每个学生的二值作答向量嵌入到量子电路中，将其压缩为稀疏的潜在表示，并将该表示映射到Q矩阵。我们在60个模拟数据集和9个真实世界评估数据集上，将QSAE与经典自编码器（CAE）进行了基准对比。结果显示两者各具优势、互为补充。尽管CAE在部分情况下取得了更高的平均（性能）……

    arXiv:2609.01537v1 Announce Type: new  Abstract: Q-matrices play a central role in cognitive diagnosis within educational data mining (EDM), specifying which latent skills each assessment item requires. Data-driven Q-matrix estimation remains challenging when assessments involve many correlated skills and when real response patterns depart from idealized generative assumptions. We introduce a novel quantum sparse autoencoder (QSAE) for Q-matrix estimation, which, to the best of our knowledge, is the first application of quantum machine learning (QML) to cognitive diagnosis. Overall, the QSAE embeds each student's binary response vector into a quantum circuit using an encoder, compresses it into a sparse latent representation, and maps that representation to the Q-matrix. We benchmark the QSAE against a classical autoencoder (CAE) across 60 simulated datasets and 9 real-world assessment datasets. The results reveal complementary strengths. Although the CAE partially achieves higher aver
    
[^11]: LatentPress：超越文本与视觉的上下文压缩

    LatentPress: Context Compression Beyond Text and Vision

    [https://arxiv.org/abs/2609.01507](https://arxiv.org/abs/2609.01507)

    LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。

    

    压缩后的上下文通常以人类可读的文本形式承载，或以必须被解码的渲染图像形式承载，即使其消费者是语言模型也是如此。我们提出了LatentPress，它将对话历史和长文档写入第三种表示形式：连续的记忆token（memory tokens），冻结的解码器通过其输入嵌入接口直接读取这些token，在推理时无需进行文本重建。一个与阅读器匹配的小型写入器可实现4至16倍的压缩，同时只需训练一个适配器（参数量为420万至2620万，约占解码器的0.1%）。在LongMemEval基准上，LatentPress在7.70倍压缩下达到0.504的准确率，超过未压缩证据的0.490，并显著优于文本摘要（0.184）和基于OCR的压缩（0.426至0.312）。在LongBench-QA上，域内写入器在4至8倍压缩下匹配或超过原始上下文阅读的性能，而16倍压缩则落后于原始上下文。写入每段对话仅需43毫秒，大约快一个数量级。

    arXiv:2609.01507v1 Announce Type: cross  Abstract: Compressed context is usually carried as human-readable text or as rendered images that must be decoded, even when its consumer is a language model. We introduce LatentPress, which writes conversational histories and long documents into a third representation: continuous memory tokens that a frozen decoder reads directly through its input-embedding interface, with no text reconstruction at inference. A small reader-matched writer compresses $4$-$16\times$ while training only an adapter (4.2M-26.2M parameters, $\sim\!0.1\%$ of the decoder). On LongMemEval, LatentPress reaches $0.504$ accuracy at $7.70\times$ compression versus $0.490$ for uncompressed evidence, outperforming text summaries (0.184) and OCR-based compression (0.426 to 0.312). On LongBench-QA, in-domain writers match or exceed raw-context reading at $4$-$8\times$ compression, while $16\times$ trails raw. Writing takes 43ms per conversation, roughly an order of magnitude fa
    
[^12]: 去中心化联邦学习中拜占庭节点放置位置的优化

    Optimizing Byzantine Node Placement in Decentralized Federated Learning

    [https://arxiv.org/abs/2609.01495](https://arxiv.org/abs/2609.01495)

    该论文首次将拜占庭节点的网络位置作为显式攻击决策进行研究，提出基于真实 gossip 传播动态的集合级度量 BPI 来量化诚实节点的累积暴露程度，从而在固定攻陷预算下找到对去中心化联邦学习影响最大的拜占庭节点放置方案。

    

    去中心化联邦学习（DFL）的安全评估通常聚焦于拜占庭参与者如何表现，却在很大程度上忽视了哪些参与者被攻陷。然而，由于聚合过程分布在通信图上，拜占庭节点的放置位置决定了恶意影响在网络中的传播方式。因此，我们将拜占庭节点的放置视为一种显式的对抗性决策，并将攻击者的目标形式化为：在固定攻陷预算下，选择一组参与者，使其对诚实节点在有限时间内的冲击最大化。为了在不针对每个候选放置方案执行学习过程的情况下近似该目标，我们提出了拜占庭放置影响力（Byzantine Placement Influence, BPI），这是一种基于真实 gossip（八卦传播）动态推导的集合级度量，用于量化在训练时间范围内诚实节点累积暴露于拜占庭输入的程度。与基于节点中心性的放置启发式方法不同……

    arXiv:2609.01495v1 Announce Type: cross  Abstract: Security evaluations of decentralized federated learning (DFL) typically focus on how Byzantine participants behave, while largely overlooking which participants are compromised. Yet, because aggregation is distributed over a communication graph, the placement of Byzantine nodes determines how malicious influence propagates through the network. We therefore treat Byzantine placement as an explicit adversarial decision and formulate the attacker's objective as selecting, under a fixed compromise budget, the set of participants that maximizes its finite-time impact on honest nodes. To approximate this objective without executing the learning process for every candidate placement, we introduce Byzantine Placement Influence (BPI), a set-level measure derived from the actual gossip dynamics that quantifies the cumulative exposure of honest nodes to Byzantine sources over the training horizon. Unlike placement criteria based on node centrali
    
[^13]: 重新思考离线数据驱动优化中的可学习性

    Rethinking Learnability in Offline Data-driven Optimization

    [https://arxiv.org/abs/2609.01493](https://arxiv.org/abs/2609.01493)

    本文针对PAC可学习性无法充分刻画离线优化的理论缺陷，提出了“算法依赖的可学习性”这一新概念，其只需保证在优化器轨迹上的精度即可支撑离线数据驱动优化。

    

    黑盒优化（BBO）已得到广泛应用，但随着现实世界中BBO问题日益复杂，进化算法和贝叶斯优化面临效率挑战。数据驱动优化通过从数据中学习来提升BBO算法的效率。离线数据驱动优化仅利用一组固定的历史评估来寻找高质量解，由于无需额外的在线评估而吸引了大量关注。尽管已提出众多离线优化方法，但一个根本问题仍未得到解答：什么样的可学习性对于离线优化是足够的？先前的理论研究表明，概率近似正确（PAC）可学习性是不够的，因为即使大多数区域被学习得很好，最优区域仍可能学习得很差。在本文中，我们提出了算法依赖的可学习性，它只要求在优化器的轨迹上具有精度

    arXiv:2609.01493v1 Announce Type: cross  Abstract: Black-Box Optimization (BBO) has found broad applications, but evolutionary algorithms and Bayesian optimization face efficiency challenges as real-world BBO problems grow increasingly complex. Data-driven optimization improves the efficiency of BBO algorithms by learning from data. Offline data-driven optimization seeks high-quality solutions using only a fixed set of previous evaluations, attracting substantial attention because it requires no additional online evaluations. Many offline optimization methods have been proposed, but a fundamental question remains unanswered: what learnability is sufficient for offline optimization? Prior theoretical studies show that Probably Approximately Correct (PAC) learnability is insufficient, as the optimal region may remain poorly learned even when most regions are well learned. In this paper, we propose algorithm-dependent learnability, which requires accuracy only on the optimizer's trajector
    
[^14]: 模仿学习能否保持灵巧操作的时间鲁棒性？跨任务执行速度的专家-学习者对比

    Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds

    [https://arxiv.org/abs/2609.01453](https://arxiv.org/abs/2609.01453)

    该研究在接触密集型ParcelStow任务中系统比较了脚本专家与模仿学习策略（ACT）在不同执行速度下的表现，发现尽管两者在标称速度下均达到100%成功率，但在最大加速时专家成功率为84%而ACT仅为53%，表明模仿学习无法完全保留专家的时间鲁棒性。

    

    摘要：通过模仿学习获得的灵巧操作策略通常只针对场景、物体或指令变化的鲁棒性进行评估，但其在不同任务执行速度下的性能却较少被考察。这留下了一个悬而未决的问题：相对于其所模仿的专家，学习者在多大程度上保留了时间鲁棒性。我们在相同的任务条件、初始条件抽样和加速因子下对专家和学习者进行比较。我们在ParcelStow任务中实例化这一评估，这是一个接触密集型任务，机器人需要抓取、重新定向并插入包裹。示范数据涵盖了包裹抓取后各操作阶段的加速范围。脚本化专家和从专家示范中训练的基于Transformer的动作分块（ACT）策略在标称速度下均达到100%的任务成功率。但在示范加速范围内的成功率出现分化：在最大加速时，专家成功率为84%，而ACT成功率为53%。

    arXiv:2609.01453v1 Announce Type: cross  Abstract: Dexterous manipulation policies learned by imitation are typically evaluated for robustness to variation in scenes, objects, or instructions, but their performance across task execution speeds is less often examined. This leaves open how much temporal robustness a learner retains relative to the expert it imitates. We compare an expert and learner under the same task conditions, initial-condition draws, and speedup factors. We instantiate the evaluation in ParcelStow, a contact-rich task in which the robot acquires, reorients, and inserts a parcel. The demonstrations span the speedup range for the manipulation phases after parcel acquisition. A scripted expert and an Action Chunking with Transformers (ACT) policy trained from the expert's demonstrations both achieve 100 percent task success at nominal speed. Their success rates diverge within the demonstrated range: at its maximum, expert success is 84 percent and ACT success is 53 per
    
[^15]: 扩散作为无时间步迭代推理的训练课程

    Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning

    [https://arxiv.org/abs/2609.01449](https://arxiv.org/abs/2609.01449)

    该论文将扩散去噪器改造为带持久隐藏状态且无时间步条件的通用迭代更新，构建出可随推理深度不断提升准确率的随时求解器，并通过推理时持续注入高斯噪声实现单轨迹高效探索解空间，在极限数独上达到99.90%的精确求解率。

    

    扩散模型和递归推理器都是迭代式的，但它们在迭代之间传递信息的方式不同。我们在扩散去噪器中加入了一个持久的隐藏状态，并移除了其时间步条件化，留下一个可以运行到任意深度的单一共享更新。其结果是一个随时求解器：准确率随推理深度的增加而持续提升，远超训练中使用的滚动长度和反向传播窗口，在Sudoku-Extreme（极限数独）上达到99.90%的精确求解率。我们还在Maze-Unique（独特迷宫）上获得了98.93%的求解率。令人惊讶的是，推理时渐进去噪并非必要：通过在每一步用新的高斯噪声替换所有非线索变量，将腐蚀程度保持在最大值，仍能保持近乎完美的求解能力并收敛到稳定的解。这种简单的噪声注入机制使单一轨迹能够高效地探索解空间并最终得到正确答案，而无需并行滚动或候选筛选。

    arXiv:2609.01449v1 Announce Type: new  Abstract: Diffusion models and recursive reasoners are both iterative, but they carry information across iterations differently. We add a persistent hidden state to a diffusion denoiser and remove its timestep conditioning, leaving a single shared update that can be run to arbitrary depth. The result is an anytime solver: accuracy keeps improving with inference depth far beyond the rollout lengths and backpropagation window used in training, reaching 99.90% exact solve on Sudoku-Extreme. We also obtain 98.93% solve rate on Maze-Unique. Surprisingly, progressive denoising is unnecessary at inference: holding corruption at its maximum by replacing every non-clue variable with fresh Gaussian noise at each step retains near-perfect solving and converges to stable solutions. This simple noise-injection mechanism enables a single trajectory to efficiently explore the solution space and settle on the correct answer without parallel rollouts, candidate se
    
[^16]: 边围长作为图神经网络的结构性边特征

    Edge-Girth as a Structural Edge Feature for Graph Neural Networks

    [https://arxiv.org/abs/2609.01441](https://arxiv.org/abs/2609.01441)

    提出以“边围长”（经过一条边的最短环长度及其数目）作为逐边结构特征，无需预先指定所统计子结构的大小即可捕捉任意长度的环，并将其融入门控消息传递网络EGAGNN以增强图神经网络超越1-WL的表达能力。

    

    基于消息传递的图神经网络（GNN）已被证明其能力不会超过一维Weisfeiler–Leman颜色细化测试（1-WL）：对于1-WL无法区分的两个图，无论网络多深或多宽，它们都会得到完全相同的表示。一种常见的补救方法是利用预计算的结构描述子来增强节点或边特征，最常见的是统计某个固定小子图（如三角形或更长的环）的数量，但这种计数方法需要预先确定所统计子结构的大小，而这一选择通常是在不了解数据的情况下盲目做出的。我们研究了一种能够避免这一选择的描述子。一条边的边围长（edge-girth）是指经过该边的最短环的长度，其重数（multiplicity）是指这类最短环的数目；二者共同构成一个逐边不变量，能够报告任意长度的环，且可通过每条边执行一次广度优先搜索精确计算。将其注入到一个门控消息传递架构EGAGNN中后，它达到了……（原文摘要在此截断）

    arXiv:2609.01441v1 Announce Type: new  Abstract: Graph neural networks (GNN) based on message passing are provably no more powerful than the one-dimensional Weisfeiler--Leman colour-refinement test (1-WL): two graphs it cannot tell apart receive identical representations, however deep or wide the network. A common remedy augments node or edge features with precomputed structural descriptors, most often counts of a fixed small subgraph such as triangles or longer cycles, but such counts require committing in advance to the size of the substructure counted, a choice usually made blind to the data. We study a descriptor that avoids this choice. The edge-girth of an edge is the length of a shortest cycle through it, and its multiplicity is the number of such shortest cycles; together they form a per-edge invariant that reports cycles of arbitrary length, computable exactly by a single breadth-first search per edge. Injected into a gated message-passing architecture, EGAGNN, it reaches a te
    
[^17]: 通过幂律熵搜索高效估计最优超参数缩放定律

    Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search

    [https://arxiv.org/abs/2609.01431](https://arxiv.org/abs/2609.01431)

    本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。

    

    最优超参数缩放定律描述了用于大语言模型（LLM）训练的最佳超参数如何随模型和数据规模变化，使从业者无需昂贵的大规模调优即可预测生产规模下的最优配置。然而，传统上估计这些缩放定律需要对数千次训练运行进行穷举网格搜索，消耗巨大的计算资源。我们提出了幂律熵搜索（Power-Law Entropy Search, PLES），这是一种建立在多保真度贝叶斯优化之上的计算成本感知采集函数，能够通过自适应实验高效估计最优超参数缩放定律。PLES的一个关键创新在于，它搜索的是能够降低缩放定律估计整体不确定性的候选配置，而不是优化单一目标函数。在每次迭代中，PLES选择能够最大程度降低缩放定律估计不确定性的候选配置。

    arXiv:2609.01431v1 Announce Type: cross  Abstract: Optimal hyperparameter scaling laws describe how the best hyperparameters for large language model (LLM) training change with model and data scale, enabling practitioners to predict optimal configurations at production scales without expensive large-scale tuning. However, estimating these scaling laws conventionally requires exhaustive grid searches over thousands of training runs, consuming enormous computational resources. We introduce Power-Law Entropy Search (PLES), a computational cost-aware acquisition function built on multi-fidelity Bayesian optimization that efficiently estimates optimal hyperparameter scaling laws through adaptive experimentation. A key innovation in PLES is that it searches for candidates that reduce the overall uncertainty of a scaling law estimate, instead of optimizing a single objective function. At each iteration, PLES selects the candidate configuration that maximally reduces the uncertainty of the sca
    
[^18]: 基于Transformer变分自编码器学习稀疏决策树

    Learning Sparse Decision Trees via Transformer Variational Auto-Encoders

    [https://arxiv.org/abs/2609.01430](https://arxiv.org/abs/2609.01430)

    TREVIS通过树Transformer变分自编码器的潜空间探索，将决策树的离散搜索转化为连续空间中基于梯度的优化，从而学习同时兼顾预测性能和结构稀疏性的决策树。

    

    决策树是机器学习中最广泛使用的模型之一，这主要归功于其透明的决策逻辑，使其非常适合高风险决策场景。然而，大多数现有的学习算法只关注预测性能，忽略了对其他理想属性（如结构稀疏性）的联合优化。在本工作中，我们提出了TREVIS，一种基于树Transformer变分自编码器（TTVAE）潜空间探索的决策树学习方法，能够针对复杂目标进行学习。通过将决策树映射到潜在表示，TREVIS用连续搜索空间取代了离散搜索空间，从而能够通过可微代理模型进行基于梯度的优化。我们使用TREVIS进行实验，学习同时优化预测性能和稀疏性的决策树。结果表明，TREVIS发现的决策树在预测性能上可与……相媲美（摘要在此处截断）。

    arXiv:2609.01430v1 Announce Type: cross  Abstract: Decision trees are among the most widely used models in machine learning, largely due to their transparent decision logic, making them well-suited for high-stakes decision-making contexts. However, most existing learning algorithms focus on predictive performance, overlooking the joint optimization of other desirable properties, such as structural sparsity. In this work we propose TREVIS, an approach for learning decision trees with respect to complex objectives, based on the exploration of the latent space of a Tree Transformer Variational Auto-Encoder (TTVAE). By mapping decision trees onto latent representations, TREVIS replaces the discrete search space with a continuous one, enabling gradient-based optimization via a differentiable surrogate model. We experiment with TREVIS for learning decision trees that jointly optimize predictive performance and sparsity. Results show that TREVIS discovers decision trees matching the predictiv
    
[^19]: TRIAGE：面向高效执行的三级路由与智能代理引导

    TRIAGE: Three-level Routing and Intelligent Agent Guidance for Efficient Execution

    [https://arxiv.org/abs/2609.01428](https://arxiv.org/abs/2609.01428)

    TRIAGE提出三级路由框架，其核心创新TaaS（轨迹即技能）将历史执行轨迹抽象为可复用技能，使相同和相似查询实现零token消耗，显著降低LLM智能体的执行成本。

    

    基于ReAct范式的大型语言模型（LLM）智能体在工具使用和任务执行方面展现出了卓越的能力。然而，ReAct存在一个根本性的效率问题：每次查询都会触发一个从头开始的完整推理循环，相似的查询会重复相同的步骤，而无法利用历史经验。我们提出了TRIAGE，这是一个通过复用历史执行轨迹来降低token消耗的三级路由框架。其核心创新是TaaS（Trajectory-as-a-Skill，轨迹即技能），它将历史执行轨迹抽象为可复用的技能，实现了“经验即服务”。TRIAGE将查询分为三个级别：(1) 直接复用——完全相同的查询，0 token消耗；(2) 技能替换——相似查询，通过确定性参数替换实现0 token消耗；(3) 完整ReAct——全新查询，自动存储以供未来复用。在1,007条安全监控查询的大规模实验中，TR…

    arXiv:2609.01428v1 Announce Type: new  Abstract: Large Language Model (LLM) agents based on the ReAct paradigm have demonstrated remarkable capabilities in tool use and task execution. However, ReAct suffers from a fundamental efficiency problem: every query triggers a complete reasoning loop from scratch, and similar queries repeat identical steps without leveraging historical experience. We propose TRIAGE,a three-level routing framework that reduces token consumption by reusing historical execution trajectories. Its core innovation is TaaS (Trajectory-as-a-Skill), which abstracts historical execution trajectories into reusable skills, realizing 'experience as a service'. TRIAGE classifies queries into three levels: (1) Direct Reuse-identical queries, 0 tokens; (2) Skill Substitution-similar queries, 0 tokens via deterministic parameter substitution; (3) Full ReAct-novel queries, automatically stored for future reuse. In large-scale experiments on 1,007 security monitoring queries, TR
    
[^20]: 基于视觉Transformer的透明细胞肾细胞癌分级的语义引导多模态预处理

    Semantic-Guided Multimodal Preprocessing for Vision Transformer-Based Clear Cell Renal Cell Carcinoma Grading

    [https://arxiv.org/abs/2609.01426](https://arxiv.org/abs/2609.01426)

    提出一种语义引导的多模态预处理方法，将细胞核分类图与RGB病理图像融合后输入视觉Transformer进行透明细胞肾细胞癌分级，将平衡准确率从0.707提升至0.916。

    

    透明细胞肾细胞癌（CCRCC）分级对于治疗规划至关重要，然而现有方法要么直接分析图像块（patch-level）级别的图像，要么仅专注于细胞核级别的分类，而没有与最终肿瘤分级建立联系。我们提出了一种语义引导的多模态预处理方法，将现有预训练模型生成的细胞核分类图与RGB病理组织学图像相融合，用于基于视觉Transformer（ViT）的CCRCC分级。我们的方法采用分类图通道拼接和乘性调制，并通过优化的叠加方式来利用细胞核分级信息，同时保留RGB纹理特征。对多种预处理策略的评估表明，语义引导增强达到了0.916的平衡准确率，优于仅使用RGB的基线（0.707）以及先前研究的最大投票聚合方法（0.427）。敏感性分析显示，这一21个百分点的提升...

    arXiv:2609.01426v1 Announce Type: cross  Abstract: Clear cell renal cell carcinoma (CCRCC) grading is essential for treatment planning, yet existing approaches either analyze patch-level images directly or focus solely on nuclei-level classification, without linking to final tumor grading. We propose a semantic-guided multimodal preprocessing method that integrates nuclei classification maps from existing pre-trained models with RGB histopathology images for Vision Transformer (ViT)-based CCRCC grading. Our approach employs classification map channel concatenation and multiplicative modulation, with optimized overlays to leverage nuclei grading information, while preserving RGB textural features. Evaluation of multiple preprocessing strategies demonstrates that semantic-guided enhancement achieves 0.916 balanced accuracy, outperforming RGB-only baseline (0.707) and max-voting aggregation from prior studies (0.427). Sensitivity analysis reveals that this 21 percentage point improvement 
    
[^21]: CATeye：面向优惠券滥用检测的耦合属性-拓扑不变性学习

    CATeye: Coupled Attribute-Topology Invariance Learning for Voucher Abuse Detection

    [https://arxiv.org/abs/2609.01425](https://arxiv.org/abs/2609.01425)

    提出CATeye框架，通过属性不变性选择器和边不变性选择器应对优惠券滥用检测中耦合的属性-拓扑分布偏移问题，无需频繁重训练即可抵御欺诈模式的快速演变。

    

    优惠券滥用是电子商务领域的一项重大挑战，恶意用户通过利用促销优惠券牟利。遗憾的是，欺诈模式会随时间和地域快速演变，造成分布偏移，导致现有检测模型性能下降，除非频繁重新训练。为解决这一问题，我们提出了耦合属性-拓扑不变性学习框架（CATeye）。其核心挑战源于耦合的属性-拓扑偏移：基于属性接近性构建的边使得环境驱动的属性偏移引发拓扑偏移，并通过GNN消息传递放大变异信号。CATeye通过两个可学习的选择器洞察此类耦合偏移。首先，属性不变性选择器（AIS）学习节点自适应掩码，以过滤掉非不变的属性。然后，在保留的不变属性的条件下，边不变性选择器（EIS）对不变子图进行采样，并隔离非不变的……

    arXiv:2609.01425v1 Announce Type: new  Abstract: Voucher abuse poses a major challenge in e-commerce, where malicious users exploit promotional vouchers for profit. Unfortunately, fraud patterns evolve rapidly over time and across regions, causing distribution shifts that degrade existing detection models unless retrained frequently. To tackle this, we propose the Coupled Attribute-Topology Invariance Learning framework (CATeye). The key challenge arises from coupled attribute-topology shift, where edges built from attribute proximity cause environment-driven attribute shift to induce shifted topology, thereby amplifying variant signals through GNN message passing. CATeye sees through such coupled shifts with two learnable selectors. First, an Attribute Invariance Selector (AIS) learns node-adaptive masks to filter out non-invariant attributes. Then, conditioned on retained invariant attributes, an Edge Invariance Selector (EIS) samples an invariant subgraph and isolates non-invariant 
    
[^22]: 可证明安全的仿真到现实迁移

    Provably Safe Sim-to-Real Transfer

    [https://arxiv.org/abs/2609.01418](https://arxiv.org/abs/2609.01418)

    该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。

    

    为了缓解现实世界强化学习（RL）的样本复杂度问题，一种常见的做法是先在模拟器中训练策略（因为样本成本低廉），然后将学到的策略部署到现实世界中，并希望其能有效泛化。然而，这种直接的仿真到现实迁移并不保证成功：由于仿真与现实之间的失配（sim-to-real mismatch），在模拟器中训练的策略在现实世界中可能是次优的。纠正这种失配需要从真实系统收集数据，但在许多应用中（如机器人技术和医疗保健），这种数据收集过程本身受到安全约束的制约。这就引出了安全仿真到现实迁移的问题：智能体如何利用一个不完美的模拟器，同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略？我们通过在无奖励安全强化学习框架内构建安全仿真到现实迁移问题来应对这一挑战……

    arXiv:2609.01418v1 Announce Type: cross  Abstract: To mitigate the sample complexity of real-world reinforcement learning (RL), a common practice is to first train a policy in a simulator, where samples are cheap, and then deploy the learned policy in the real world with the hope that it generalizes effectively. Such direct sim-to-real transfer is not guaranteed to succeed: simulator-trained policies can be suboptimal in the real world due to sim-to-real mismatch. Correcting this mismatch requires collecting data from the real system, but in many applications, such as robotics and healthcare, this data-collection process is itself subject to safety constraints. This gives rise to the problem of safe sim-to-real transfer: how can an agent exploit an imperfect simulator while ensuring safe real-world data collection and learning a near-optimal feasible policy for the target system? We address this problem by formulating safe sim-to-real transfer within the framework of reward-free safe R
    
[^23]: 使用物理信息神经网络预测地下异常体生长

    Predicting Subsurface Abnormalities Growth using Physics-Informed Neural Networks

    [https://arxiv.org/abs/2609.01417](https://arxiv.org/abs/2609.01417)

    该研究首次将物理信息神经网络（PINNs）应用于探地雷达数据预测，通过将电磁波传播物理定律嵌入由CNN、空间特征通道注意力和ConvLSTM组成的深度学习架构中，实现对地下异常体生长的精确预测。

    

    该研究探索了将物理信息神经网络（PINNs）开创性地集成到探地雷达（GPR）数据预测领域。本研究提出了一个专门化PINN模型的详细开发框架，该模型能够熟练地解释和预测GPR数据，其原理类似于医学成像模型预测肿瘤行为的方式。通过利用深度学习算法与支配地下结构（医学上称为人体组织）的物理定律之间的协同作用，该模型有效地将电磁波传播的物理规律嵌入到其架构中。这确保了预测结果不仅符合基本物理原理，还体现了医学诊断中检测和监测肿瘤所需的精度。所提出的深度学习结构由三个组件构成：卷积神经网络（CNN）、空间特征通道注意力（SFCA）机制和ConvLSTM，以及时间特征模块（摘要原文在此处截断）。

    arXiv:2609.01417v1 Announce Type: new  Abstract: The research explores the pioneering integration of Physics-Informed Neural Networks (PINNs) into the domain of Ground-Penetrating Radar (GPR) data prediction. This research presents a detailed development framework for a specialized PINN model, proficient at interpreting and forecasting GPR data, much like how medical imaging models predict tumor behavior. By harnessing the synergy between deep learning algorithms and the physical laws governing subsurface structures or in medical terms, human tissues the model effectively embeds the physics of electromagnetic wave propagation into its architecture. This ensures that predictions not only align with fundamental physical principles but also mirror the precision needed in medical diagnostics for detecting and monitoring tumors. The suggested deep learning structure comprises three components: a CNN, a spatial feature channel attention (SFCA) mechanism, and ConvLSTM, along with temporal fea
    
[^24]: 面向多模态分割学习的贡献感知带宽分配

    Contribution-Aware Bandwidth Allocation for Multimodal Split Learning

    [https://arxiv.org/abs/2609.01406](https://arxiv.org/abs/2609.01406)

    该论文提出 ModalShare，一种基于 Shapley 贡献度为各模态动态设置压缩保留率的带宽分配器，解决了多模态分割学习中传统均匀压缩策略忽略各模态对融合预测贡献差异的问题。

    

    多模态模型日益成为网络边缘感知的默认选择，然而它们几乎完全在数据中心进行训练，这是因为持有多个传感器数据流的客户端无法为每种模态托管一个编码器。分割学习通过只在设备端保留前几层网络，使这种训练变得可行，但代价是上行链路必须在每一步为每种模态传输压缩后的激活值。现有的压缩方案给每种模态相同的保留率，因此共享带宽预算按压缩激活值的维度成比例分配，而这一维度与各模态对融合预测的贡献大小无关。我们将这种划分变成一个显式的决策，并称之为模态间分配：在固定的上行链路预算下，任何策略传输的期望载荷都相同，区别仅在于该载荷在各模态之间如何划分。我们的分配器 ModalShare 依据 Shapley 贡献

    arXiv:2609.01406v1 Announce Type: new  Abstract: Multimodal models are increasingly the default option for perception at the network edge, yet they are trained almost entirely in the datacenter, because a client holding several sensor streams cannot host an encoder per modality. Split Learning makes such training feasible by keeping only the first layers on the device, at the cost of an uplink that must carry smashed activations for every modality at every step. Existing compression schemes give each modality the same keep-ratio, so the shared budget is divided in proportion to smashed-activation dimension, a quantity unrelated to how much each modality contributes to the fused prediction. We make that division an explicit decision and call it inter-modality allocation: under a fixed uplink budget, every policy transmits the same expected payload and differs only in how that payload is split across modalities. Our allocator, ModalShare, sets each modality's keep-ratio from a Shapley co
    
[^25]: 通过集成边界与局部预测变异性衡量一致性：在预测多样性存在下审计决策系统

    Measuring consistency via ensemble margin and local prediction variability: Auditing decision systems in the presence of predictive multiplicity

    [https://arxiv.org/abs/2609.01397](https://arxiv.org/abs/2609.01397)

    该论文提出一种将集成边界与局部预测变异性相结合的一致性准则，用于在罗生门效应（预测多样性）存在下审计决策系统，并证明在温和假设下有限集成的一致性分数会收敛于罗生门集合中期望模型的一致性分数。

    

    罗生门效应是机器学习中的一种现象，即准确度相同的模型会对相同的输入产生不同的预测（预测多样性）。现有工作主要关注单个模型内部的多样性，但在更复杂的决策系统中，罗生门效应的影响尚不十分清楚。在本研究中，我们从审计错误集成预测的角度研究多样性问题，其中将某个实例转移给人工审查的决策基于一个一致性准则，该准则将集成边界与每个组成模型的局部预测变异性度量相结合。在关于稳定性和平滑性的温和假设下，我们证明随着集成规模以及用于测量局部预测变异性的样本数量的增加，有限集成的一致性分数收敛于来自罗生门集合的期望模型的相应一致性分数。为了演示……

    arXiv:2609.01397v1 Announce Type: cross  Abstract: The Rashomon effect is a machine learning phenomenon where equally accurate models produce different predictions for the same inputs (predictive multiplicity). Existing work primarily focuses on multiplicity within individual models, but in more complex decision systems, the impact of the Rashomon effect is less well understood. In this work, we study multiplicity from the perspective of auditing incorrect ensemble predictions, where the decision to divert an instance for human review is based on a consistency criterion that combines the ensemble margin with a measure of local prediction variability for each constituent model. With mild assumptions about stability and smoothness, we show that the consistency scores of finite ensembles converge to the corresponding consistency score of the expected model from the Rashomon set as the ensemble size and the number of samples used to measure local prediction variability increase. To demonst
    
[^26]: 探究医学问答任务中线性探针对语言语域、医学专科与语料库变化的鲁棒性

    Investigating Linear Probe Robustness to Linguistic Register, Medical Specialty, and Corpus Shifts in Medical QA

    [https://arxiv.org/abs/2609.01361](https://arxiv.org/abs/2609.01361)

    该论文构建了一个可独立操控写作语域、医学专科和语料库三类变化的医学问答基准，以系统性探究大语言模型中线性探针（真值方向检测）对不同输入偏移的鲁棒性。

    

    在大语言模型（LLM）隐状态上训练的线性分类器，即线性探针，可以通过单次前向传播来标记事实性错误。从几何角度看，这意味着真与假的陈述在隐状态空间中沿一个稳定的方向分离，即“真值方向”。已有研究对这种能力能否在不同输入偏移下泛化存在分歧，但由于跨数据集的探针迁移实验同时混淆了多种输入变化，这种分歧难以解释。我们在医学问答（QA）任务中分离出三个此类变量：写作风格（语域）、领域（医学专科）和语料库（数据集）。我们基于500条MedQA条目构建了一个基准，每条条目被改写为四种风格（教科书式、患者口吻、临床笔记、口语化），标注了临床专科，并与另外两个考试语料库MedMCQA和MMLU-medical组合，用于跨数据集评估。通过对四个开源权重LLM（2--8B）进行探针实验，我们发现……

    arXiv:2609.01361v1 Announce Type: new  Abstract: Linear classifiers trained on hidden states of a large language model (LLM), linear probes, can flag factual errors from a single forward pass. Geometrically, that implies that true and false statements separate along a stable direction in hidden state space, i.e., the truth direction. Prior work disagrees on whether this generalises across input shifts, but the disagreement is hard to interpret because cross-dataset probe transfer experiments confound several kinds of input change at once. We isolate three such variables in medical question-answering (QA): writing style (register), domain (medical specialty), and corpus (dataset). We build a benchmark using 500 MedQA entries, each rewritten into four styles (textbook, patient, clinical note, colloquial), annotated with clinical specialty, and grouped with two other exam corpora, MedMCQA and MMLU-medical, for cross-dataset evaluation. Probing four open-weight LLMs (2--8B), we find that t
    
[^27]: 情景优化与无分布认证中投影边界的精确风险-复杂度定律

    Exact Risk-Complexity Laws for Projective Boundaries in Scenario Optimization and Distribution-Free Certification

    [https://arxiv.org/abs/2609.01355](https://arxiv.org/abs/2609.01355)

    本文揭示了情景优化、共形预测等无分布认证方法中违反风险精确贝塔定律背后的确定性投影边界机制，并提出了“真投影边界方案”框架，将该定律推广到边界大小为随机的更一般情形。

    

    情景优化、共形预测以及相关的无分布认证方法利用有限样本来构建决策或预测集，并对新观测提供违反风险保证。在若干经典设定中，条件违反风险服从精确的贝塔分布定律，其尾部具有贝塔-二项表示，其参数为支撑集、校准或压缩维度。本文识别出这些公式背后的确定性边界机制，并在观测到的边界大小为随机的情况下推导出相应的定律。决策规则由一个对未来观测的接受集来表示，同时附带一个边界映射，用于选择决定该集合的样本点。当保留样本（held-out samples）被接受当且仅当全样本边界被保留，且被接受的非边界样本可以被删除时，所得到的这一对结构被称为“真投影边界方案”（proper projective boundary scheme）……

    arXiv:2609.01355v1 Announce Type: cross  Abstract: Scenario optimization, conformal prediction, and related distribution-free certification methods use finite samples to construct decisions or prediction sets with violation-risk guarantees for fresh observations. In several classical settings, the conditional violation risk follows an exact beta law, whose tail has a beta-binomial representation and whose parameter is a support, calibration, or compression dimension. This paper identifies the deterministic boundary mechanism behind these formulas and derives the corresponding law when the observed boundary size is random. A decision rule is represented by an acceptance set for future observations, together with a boundary map selecting the sample points responsible for that set. The resulting pair is called a {\em proper projective boundary scheme} when held-out samples are accepted precisely if the full-sample boundary is retained, and accepted non-boundary samples can be deleted with
    
[^28]: 验证器在哪里失效：对RLVR中奖励信号的类别级审计

    Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR

    [https://arxiv.org/abs/2609.01354](https://arxiv.org/abs/2609.01354)

    该论文将变异测试从模型转向验证器本身，通过构造保证数学等价的答案变体，在超过30万个判定上对四个主流验证器进行了类别级审计，发现相同输入下验证器的自我验证率相差高达41.3个百分点，揭示了RLVR奖励信号中的系统性假阴性问题。

    

    arXiv:2609.01354v1 公告类型：新论文 摘要：可验证奖励强化学习（RLVR）和标准基准评估都依赖于一个自动验证器，它将自由文本的答案转换为二元奖励。先前的工作报告称，某个评估框架仅接受了约94%的其自身标准答案，并将其归咎于LaTeX解析问题。但这只是一个总体数字：它没有说明哪些答案形式消耗了错误预算。我们提供了这种分解。我们将变异测试应用于验证器而非模型，生成经过认证的等价答案变体，即通过构造保证保持数学意义的改写，因此任何拒绝都是可证明的假阴性，无需人工裁定。然后，我们在307,420个判定上测量了四个广泛使用的验证器对每个答案类别的拒绝率。我们发现了三件事。（1）在相同输入上，自我验证率从53.8%到95.2%不等，相差41.3个百分点。已发表的数字仅描述了其中一个实现，

    arXiv:2609.01354v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) and standard benchmark evaluation both rely on an automatic verifier that turns a free text answer into a binary reward. Prior work reports that one evaluation harness accepts only about 94% of its own ground truth answers, blaming LaTeX parsing. That is an aggregate: it does not say which answer forms consume the error budget. We supply the decomposition. We apply metamorphic testing to the verifier rather than the model, generating certified equivalent answer variants, that is, rewrites that preserve mathematical meaning by construction, so that any rejection is a provable false negative needing no human adjudication. We then measure rejection per answer category across four widely used verifiers over 307,420 verdicts. We find three things. (1) Self validation ranges from 53.8% to 95.2% on identical inputs, a spread of 41.3 points. The published figure describes one implementation, 
    
[^29]: 便宜的验证器，巨大的盲区：衡量成本节约级联的可靠性代价

    Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades

    [https://arxiv.org/abs/2609.01345](https://arxiv.org/abs/2609.01345)

    该研究通过真实LLM实验发现，推理级联中廉价验证器对学生模型错误答案的“盲区”随学生能力增强而扩大、随验证器能力增强而缩小，恰好在级联机制赖以存在的低成本配置下最为严重，而用前沿验证器消除盲区又会因过度升级而抵消成本节约，从而揭示了成本节约级联设计背后隐藏的显著可靠性代价。

    

    推理级联通过用廉价模型回答大多数查询，并将困难的长尾部分升级给作为验证器的前沿模型来降低成本。一个自然的扩展方式形成闭环：在验证器的拒绝样本上微调廉价的学生模型，使升级率（以及成本）逐轮下降。我们在真实的LLM上测量了这个循环，并报告了四项发现。首先，验证器的盲区——即它接受的学生错误答案的比例——很大且呈对抗性变化：它随学生能力的增强而增大（当学生模型从0.5B扩展到32B时，β从0.12增至0.55），并随验证器能力的增强而减小，因此在“廉价学生+廉价验证器”这一级联机制所天然创造的情境下，盲区最为严重。其次，花钱消除盲区会抵消节省的成本：一个前沿验证器能把β降到约0.05，但随后会在46%的困难MATH查询上进行升级，而真实错误率仅为39%，这意味着几乎一半的流量都要支付前沿模型的价格。第三，朴素的纠正性微调……（摘要在此处被截断）

    arXiv:2609.01345v1 Announce Type: new  Abstract: Inference cascades cut cost by answering most queries with a cheap model and escalating a hard tail to a frontier model that acts as verifier. A natural extension closes the loop: fine-tune the cheap student on the verifier's rejections so the escalation rate, and cost, fall each round. We measure this loop on real LLMs and report four findings. First, the verifier's blind spot, the fraction of the student's wrong answers it accepts, is large and moves adversarially: it grows with student capability ($\beta$ from 0.12 to 0.55 as the student scales 0.5B to 32B) and shrinks with verifier capability, so it is worst in the cheap-student, cheap-verifier regime cascades exist to create. Second, buying it away returns the saving: a frontier verifier drives $\beta$ to about 0.05 but then escalates on 46% of hard-MATH queries against a 39% true error rate, paying the frontier price on nearly half of all traffic. Third, naive corrective fine-tunin
    
[^30]: mzCache：多任务环境下的端侧大语言模型内存管理

    mzCache: On-Device LLM Memory Management under Multitasking

    [https://arxiv.org/abs/2609.01338](https://arxiv.org/abs/2609.01338)

    mzCache针对移动设备多任务环境下的不可预测内存压力，提出面向恢复的内存管理机制，弹性驱逐LLM内存并利用移动SoC统一内存在CPU端并发恢复的同时实现GPU上的零等待推理。

    

    端侧移动大语言模型（LLM）推理正受到广泛关注。然而，移动设备运行在高度动态的多任务环境中，用户频繁在应用之间切换。这会造成内存压力，迫使操作系统驱逐LLM内存（模型权重和KV缓存）。当新的推理请求到来时，推理系统必须通过缓慢的存储读取恢复被驱逐的内存，或重新计算整个KV缓存，严重降低了响应速度。为解决这一问题，我们提出了mzCache，一个专为多任务环境设计的、具有专门内存管理机制的端侧LLM推理系统。在不可预测的内存压力下，mzCache弹性地驱逐LLM内存，并利用移动SoC的统一内存，在CPU端并发恢复内存的同时实现GPU上的零等待推理。mzCache通过面向恢复的内存管理实现这一目标：LLM内存被划分（摘要在此处截断）

    arXiv:2609.01338v1 Announce Type: cross  Abstract: On-device mobile Large Language Model (LLM) inference is gaining significant attention. However, mobile devices operate in highly dynamic multitasking environments where users frequently switch between applications. This creates memory pressure, forcing LLM memory (model weights and KV cache) to be evicted by the operating system. When a new inference request arrives, the inference system must restore the evicted memory through slow storage reads or recompute the entire KV cache, severely degrading responsiveness. To address this, we present mzCache, an on-device LLM inference system with specialized memory management for multitasking environments. Under unpredictable memory pressure, mzCache elastically evicts LLM memory and leverages the unified memory of mobile SoCs to enable zero-wait inference on the GPU with concurrent CPU-side restoration. mzCache realizes this through restoration-oriented memory management: LLM memory is partit
    
[^31]: 生产环境中的老虎机：推理时的超参数优化

    Bandits in Prod: Hyperparameter Optimization at Inference Time

    [https://arxiv.org/abs/2609.01335](https://arxiv.org/abs/2609.01335)

    该论文将生产系统中只能通过线上噪声反馈评估配置的场景形式化为在线超参数优化（OHPO），提出通用框架IMABO及免重启的无限多臂老虎机策略IMOSS，并给出了分位数遗憾的理论保证。

    

    许多生产系统只能通过将某个配置应用于实际线上请求并观察带噪声的反馈来评估该配置。现代智能体系统是一个突出的例子，其推理时的选择包括模型选择、检索深度、提示策略和解码温度等，但往往缺乏具有代表性的验证数据。我们将这一设置形式化为在线超参数优化（OHPO），并将其转化为混合与条件搜索空间上的无限多臂老虎机问题。我们提出了IMABO这一通用框架，它将任意用于在已采样配置中进行选择的老虎机策略与任意用于提出新配置的预言机相结合。我们用IMOSS对该框架进行了实例化，IMOSS是一种无需重启的anytime策略，其活跃集合以 $t^{\beta}$ 的速度增长，并证明了期望累积分位数遗憾界为 $O(p_\rho^{-1/\beta} + T^{(1+\beta)/2})$，其中 $\beta\in(0,1)$ 控制活跃集合的增长，$p_\rho$ 是对某个概率的下界约束（摘要在此处截断）。

    arXiv:2609.01335v1 Announce Type: cross  Abstract: Many production systems can assess a configuration only by using it on live requests and observing noisy feedback. Modern agentic systems are a prominent example, with inference-time choices such as model selection, retrieval depth, prompting strategy, and decoding temperature, yet often with no representative validation data. We formalize this setting as Online Hyperparameter Optimization (OHPO) and cast it as an infinitely many-armed bandit over mixed and conditional search spaces. We introduce IMABO, a general framework that combines any bandit policy for choosing among already sampled configurations with any oracle for proposing new ones. We instantiate it with IMOSS, a restart-free anytime policy whose active set grows as $t^{\beta}$, and prove an expected cumulative quantile-regret bound of $O(p_\rho^{-1/\beta} + T^{(1+\beta)/2})$, where $\beta\in(0,1)$ controls active-set growth and $p_\rho$ lower-bounds the probability that a p
    
[^32]: 探索稀疏自编码器在基于文本的因果混杂调整中的应用

    Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment

    [https://arxiv.org/abs/2609.01322](https://arxiv.org/abs/2609.01322)

    该论文提出一种基于稀疏自编码器（SAE）的新颖因果调整流程，通过条件独立性检验迭代选取最小特征集合，解决了文本表示在保留混杂变量与满足有限样本重叠条件之间的权衡，在半合成评估中实现了比替代表示更低的偏差和更高的覆盖率。

    

    在许多场景中，基于文本数据研究因果问题需要对文本中的混杂信息进行调整。然而，构建用于调整的文本表示存在一种权衡：文本表示必须足够大和/或稠密，以保留无偏效应估计所需的混杂变量；但又必须足够小和/或稀疏，以满足有限样本的重叠条件并获得低方差的估计。为解决这一权衡问题，我们转向稀疏自编码器（SAE），提出了一种新颖的因果调整流程，该流程通过条件独立性检验迭代地选择一个最小的SAE特征集合。我们发现，在带有二元混杂变量的标准半合成评估中，SAE表示比其他替代表示实现了更好的调整效果（更低的偏差和更高的覆盖率），并且其可解释性为证伪检验提供了机会。我们还引入了一个更贴近真实的半合成评估……

    arXiv:2609.01322v1 Announce Type: new  Abstract: In many settings, studying causal questions based on text data requires adjusting for confounding information within texts. Yet there is a tradeoff in constructing text representations for adjustment: they must be sufficiently large and/or dense to preserve the confounding variables necessary for unbiased effect estimation, but sufficiently small and/or sparse to satisfy finite-sample overlap and yield low-variance estimates. To address this tradeoff, we turn to sparse autoencoders (SAEs), and propose a novel causal adjustment pipeline that iteratively selects a minimal set of SAE features via conditional independence tests. We find that SAE representations achieve better adjustments (lower bias and and higher coverage) than alternative representations in standard semi-synthetic evaluations with binary confounders, and their interpretability offers opportunities for falsification. We also introduce a more realistic semi-synthetic evaluat
    
[^33]: MIDR：面向多模态文档检索的富化增强索引

    MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval

    [https://arxiv.org/abs/2609.01316](https://arxiv.org/abs/2609.01316)

    MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。

    

    对视觉丰富文档的检索存在一个表示难题：重要内容往往存在于表格、图表、图形和布局关系中，而普通OCR会将其线性化、破坏或遗漏。ColPali系列视觉检索器通过补丁级多向量索引和后期交互评分来解决这一问题，但这使图像衍生的检索保留在查询时的服务路径上。我们提出MIDR（Multimodal Indexing for Document Retrieval，面向文档检索的多模态索引），这是一个无需训练的富化增强索引框架，将多模态推理转移到索引阶段。在数据摄取过程中，多模态大语言模型将渲染的页面转换为经过验证的文本字段，并使用BM25F进行索引，可选择与稠密检索融合，从而在多模态扎根的证据之上实现以文本为中心的服务。在ViDoRe V3上，MIDR Hybrid在五个英文领域取得0.6219的平均nDCG，相比BM25相对提升23.0%，与ColQwen2.5保持竞争力。

    arXiv:2609.01316v1 Announce Type: cross  Abstract: Retrieval over visually rich documents has a representation problem: important content often lives in tables, charts, figures, and layout relations that plain OCR linearizes, corrupts, or omits. ColPali-family visual retrievers address this with patch-level multi-vector indexes and late-interaction scoring, keeping image-derived retrieval on the query-time serving path. We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing that shifts multimodal reasoning to index time. During ingestion, a multimodal LLM converts rendered pages into verified textual fields that are indexed with BM25F and optionally fused with dense retrieval, enabling text-centric serving over multimodally grounded evidence. On ViDoRe V3, MIDR Hybrid achieves 0.6219 average nDCG across five English domains, a 23.0% relative gain over BM25, remaining competitive with ColQwen2.5. On two French-document
    
[^34]: 单层Transformer被证明能够以上下文方式学习多分类单最近邻

    One-Layer Transformer Provably Learns Multiclass One-Nearest Neighbor in Context

    [https://arxiv.org/abs/2609.01311](https://arxiv.org/abs/2609.01311)

    本文证明了带argmax分类头的单层Transformer在多分类的上下文学习中行为与单最近邻分类器完全一致，填补了此前工作依赖非标准舍入方法所留下的理论空白。

    

    我们将近期一项在二分类设定下建立了单层Transformer与最近邻分类器之间等价性的工作，扩展到多分类情形。通过利用单纯形编码，我们证明了带argmax分类头的单层Transformer在多分类设定下的行为与单最近邻分类器完全一致。这填补了先前工作留下的空白——先前工作的多分类结果依赖于基于舍入的非标准方法，而非实践中常用的argmax分类头。

    arXiv:2609.01311v1 Announce Type: new  Abstract: We extend recent work establishing an equivalence between one-layer transformers and nearest-neighbor classifiers in the binary setting to the multiclass case. By leveraging the simplex encoding, we show that one-layer transformers with an argmax classification head behave identically to a one-nearest-neighbor classifier in the multiclass setting. This closes a gap left by prior work, whose multiclass result relied on a non-standard rounding-based approach rather than the typical argmax head used in practice.
    
[^35]: GazeRefine：将专家眼动注视作为免训练医学图像分割的测试时提示

    GazeRefine: Expert Gaze as a Test-Time Prompt for Training-Free Medical Image Segmentation

    [https://arxiv.org/abs/2609.01310](https://arxiv.org/abs/2609.01310)

    GazeRefine提出了一种免训练的零样本医学图像分割框架，将专家眼动注视转化为前景/背景先验，在冻结的DINOv3特征空间中初始化并迭代细化语义原型，无需任何分割掩码、微调或梯度更新。

    

    医学图像分割一直难以规模化，因为高性能的方法通常依赖于密集的专家标注和针对特定任务的训练。我们提出了GazeRefine，这是一个免训练框架，它将眼动注视作为推理时的提示，用于零样本医学图像分割。稀疏的、按注视时长加权的注视点被转换为前景和背景先验，用于在冻结的DINOv3特征空间中初始化语义原型。这些原型通过前景-背景判别、特征空间亲和传播以及锚定于初始注视引导进行迭代细化，使分割能够扩展到直接注视区域之外，同时限制语义漂移。GazeRefine不需要分割掩码、微调、适配器、提示编码器或梯度更新。我们在带有眼动标注的息肉分割和前列腺MRI分割任务上评估了该方法。结果显示其在结肠镜检查场景中表现出色（原文在此处截断）。

    arXiv:2609.01310v1 Announce Type: cross  Abstract: Medical image segmentation remains difficult to scale because high-performing methods typically rely on dense expert annotations and task-specific training. We introduce GazeRefine, a training-free framework that uses gaze as an inference-time prompt for zero-shot medical image segmentation. Sparse, duration-weighted fixations are converted into foreground and background priors that initialize semantic prototypes in frozen DINOv3 feature space. These prototypes are iteratively refined through foreground-background discrimination, feature-space affinity propagation, and anchoring to the initial gaze guidance, allowing segmentation to extend beyond directly fixated regions while limiting semantic drift. GazeRefine requires no segmentation masks, fine-tuning, adapters, prompt encoders, or gradient updates. We evaluate the method on gaze-annotated polyp segmentation and prostate MRI segmentation. The results show strong performance on colo
    
[^36]: 关系任务生成语言：一种面向关系深度学习的声明式规范框架

    Relational Task Generation Language: A Declarative Specification Framework for Relational Deep Learning

    [https://arxiv.org/abs/2609.01292](https://arxiv.org/abs/2609.01292)

    提出了一种开源声明式语言RTGL，通过抽象SQL细节来简化关系深度学习预测任务的定义，避免了手动定义导致的数据泄露问题，并能与现有RDL框架无缝集成。

    

    关系深度学习（RDL）已成为从多表格数据中学习的强大范式。然而，手动定义RDL预测任务是一个费力的过程，且经常导致数据泄露。为了解决这一问题，我们提出了关系任务生成语言（RTGL）——一种开源的声明式语言，通过抽象掉低级SQL细节来简化RDL任务的定义。我们通过重构现有的RDL基准任务来展示RTGL，并揭示了由于手动编写的RDL预测目标SQL定义所导致的不一致性问题，从而凸显了专用声明式语言的价值。此外，我们通过设计多种具有不同形式和目标类型的新任务，展示了RTGL的实用价值。我们的实验证实了RTGL的鲁棒性和易用性，以及它与现有RDL框架的无缝集成能力，使其能够被广泛地使用。

    arXiv:2609.01292v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) has become a powerful paradigm for learning from multi-tabular data. However, manually defining RDL prediction tasks is a laborious process that frequently results in data leakage. To address this issue, we introduce Relational Task Generation Language (RTGL) - an open-source declarative language that streamlines RDL task formulation by abstracting away low-level SQL details. We showcase RTGL by reconstructing existing RDL benchmark tasks and uncovering their inconsistencies stemming from manually crafted SQL definitions of RDL prediction targets, thereby underscoring the value of a dedicated declarative language. In addition, we demonstrate the practical utility of RTGL by designing various new tasks with diverse forms and target types. Our experiments confirm the robustness and usability of RTGL, as well as its seamless integration with the existing RDL frameworks, making it widely accessible to the com
    
[^37]: AI治理中的宪法覆盖三难困境

    The Constitutional Coverage Trilemma in AI Governance

    [https://arxiv.org/abs/2609.01275](https://arxiv.org/abs/2609.01275)

    该研究通过审计23个前沿大模型的默认“宪法”并调查1,649人的价值权衡偏好，发现人类的价值需求广泛多样，而AI模型隐含的价值排序供给既狭窄又随时间漂移，导致近四成用户找不到符合自身价值偏好的模型，揭示了AI治理中的“宪法覆盖三难困境”。

    

    前沿AI系统实际上发挥着“宪法机构”的功能：每个部署的模型都在安全性、有用性、诚实性、自主性和公平性之间编码了一种隐含的优先级排序。我们探究前沿模型的“宪法类型”供给是否覆盖了人类的需求。通过结合对23个前沿大语言模型原型的出厂默认宪法进行的释义控制审计，以及1,649名美国参与者在同一测量工具上开展的两两权衡研究，我们报告了三个事实。其一，需求是广泛的：它涵盖所有五种价值观，且最大的群体占比不足三分之一。其二，供给是狭窄且漂移的：在保守的噪声匹配估计下，23个模型原型所构成的覆盖范围仅占需求范围的约2%（在全审计精度下仅为0.10%）；没有任何模型原型将有用性或自主性置于首位（37%的用户在“宪法”意义上无家可归）；并且在六个模型家族中，自主性在5/6的家族中下降，公平性在5/6的家族中上升，安全性……（摘要在此处被截断）

    arXiv:2609.01275v1 Announce Type: cross  Abstract: Frontier AI systems function as \emph{constitutional institutions}: each deployed model encodes an implicit ranking among safety, helpfulness, honesty, autonomy, and equity. We ask whether the supply of frontier constitutional types covers human demand. Combining a paraphrase-controlled audit of the as-shipped default constitutions of $23$ frontier LLM archetypes with a pairwise-tradeoff study of $1{,}649$ US participants on the same instrument, we report three facts. \emph{Demand is broad}: it spans all five values, with the largest constituency under one-third. \emph{Supply is narrow and drifting}: the $23$-archetype hull occupies ${\sim}2\%$ of the demand hull under conservative noise-matched estimation ($0.10\%$ at full audit precision), no archetype puts helpfulness or autonomy first ($37\%$ of users are constitutionally homeless), and across six model families autonomy decreases in $5/6$, equity increases in $5/6$, and safety inc
    
[^38]: 立场：隐私是一种声明，而非合成数据的固有属性

    Position: Privacy Is a Claim, Not a Property of Synthetic Data

    [https://arxiv.org/abs/2609.01273](https://arxiv.org/abs/2609.01273)

    这篇立场论文指出，合成数据的隐私保障不应被当作数据本身的固有属性，而必须基于明确的威胁模型和可证伪的隐私声明，呼吁社区重新确立严格的隐私证据标准。

    

    合成数据已成为机器学习研究中的常见组成部分。尽管被广泛采用，但其在隐私敏感场景中的使用已悄然发生转变——从“在既定假设下的残余推断风险声明”变成了“从数据生成本身推断出的基于表象的属性”。在这篇立场论文中，我们认为这种转变反映了社区对“何为充分的隐私证据”这一标准的隐含变化，而非对成熟隐私原则的误解。通过对近年来主要机器学习会议发表论文的实证分析，我们表明合成数据经常在隐私敏感场景中被使用，却未明确阐述威胁模型、推断风险或可证伪的隐私声明。因此，隐私保障往往仍停留在隐含层面，难以验证且分布不均，稀有记录和少数群体记录面临更高的暴露风险。我们主张将隐私视为……（原文摘要在此处被截断）

    arXiv:2609.01273v1 Announce Type: new  Abstract: Synthetic data has become a common component of machine learning research. While widely adopted, its use in privacy-sensitive contexts has quietly shifted from a claim of residual inference risk under stated assumptions to an appearance-based property inferred from data generation itself. In this position paper, we argue that this shift reflects an implicit change in community standards for what counts as sufficient privacy evidence, rather than a misunderstanding of well-established privacy principles. Drawing on an empirical analysis of recent publications across major ML venues, we show that synthetic data is frequently used in privacy-sensitive settings without explicit articulation of threat models, inference risks, or falsifiable privacy claims. As a result, privacy assurance often remains implicit, difficult to verify, and unevenly distributed, with heightened exposure for rare and minority records. We argue for treating privacy a
    
[^39]: 使用合成数据进行性能评估的深度神经网络求解表内预测问题

    Solving In-Table Prediction Problems by Deep Neural Networks with Performance Evaluation Using Synthetic Data

    [https://arxiv.org/abs/2609.01262](https://arxiv.org/abs/2609.01262)

    本文提出了“表内预测”（In-Table Prediction, ITB）这一新问题，通过自监督深度神经网络随机掩码表格中的列并以其为学习目标，利用其余已知列预测任意选定列的值，并识别出三种此前未被广泛研究的应用场景。

    

    表格深度学习（TDL）利用神经网络从表格数据中提取模式。传统的表格深度学习方法遵循监督学习范式，其中目标特征是明确给定的。然而，本工作探索了一种不同的方法，即利用深度神经网络来学习给定表格中各个列之间的关系。我们研究神经网络能否基于表格中其余已知的列来预测任意选定列的值。我们将该问题称为表内预测（ITB），它与表格插补方法以及表格深度学习的预训练任务略有不同。我们识别出三种潜在的应用场景，据我们所知，这些场景在文献中尚未得到广泛研究。我们采用自监督学习方法来解决这一问题，即随机选择若干列进行掩码处理并将其作为学习目标。本工作聚焦于包含……的表格数据集（摘要在此处截断）。

    arXiv:2609.01262v1 Announce Type: new  Abstract: Tabular deep learning (TDL) leverages neural networks (NN) to extract patterns from tabular data. Traditional TDL methods follow a supervised learning paradigm, where a target feature is explicitly given. In this work, however, we explore a different approach by employing deep NNs to learn relationships among individual columns within a given table. We investigate whether NNs can predict the values of arbitrarily selected columns in a given table based on the remaining known columns. We call this problem In-Table Prediction (ITB), which is slightly different from table imputation methods and the pretraining task of TDL. Three potential usage scenarios are identified, which, to our best knowledge, have not been extensively studied in the literature. A self-supervised learning approach is applied to address this problem by randomly selecting columns to be masked out and used as learning targets. This work focuses on tabular datasets contai
    
[^40]: 更多探索，更少漂移：仅结果奖励的强化学习足以胜任长时程交互智能体

    Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents

    [https://arxiv.org/abs/2609.01245](https://arxiv.org/abs/2609.01245)

    本文提出CANOPY方法，论证仅结果奖励的强化学习足以训练小规模开源LLM智能体完成长时程交互任务，所谓瓶颈实为探索不足导致的信号饥饿与缺乏锚定导致的策略漂移这两个常见实践问题的产物。

    

    强化学习是对大语言模型智能体进行后训练以完成长时程交互任务的自然方式，这类任务仅通过任务结束时的验证来评判。然而一个普遍的看法认为，仅使用结果奖励的强化学习在小规模开源模型上很快会遭遇瓶颈。因此，近期工作通过更密集的奖励、SFT先验、技能库、精心筛选的记忆或多智能体编排等方式在训练之外进行补偿。我们认为，这一瓶颈实际上是常见实践中两类失败的产物。信号饥饿：使用稀疏的仅结果奖励的组相对强化学习，只有在某个任务的rollout组中同时混合了成功和失败样本时才会产生梯度，因此探索规模不足恰恰使最难、最具指导意义的任务失去了学习信号。策略漂移：从较小的任务池中榨取大量更新会导致策略本身退化，因为缺乏锚定的目标函数使得采样分布恰好在饱和已使有信息量的组变得稀少时发生坍缩。我们提出了CANOPY（Coverage-ANchored On-P...

    arXiv:2609.01245v1 Announce Type: cross  Abstract: Reinforcement learning is a natural way to post-train LLM agents for long-horizon interactive tasks judged only by end-of-task verification, yet a shared belief holds that outcome-only RL soon hits a ceiling on small open models. Recent work therefore compensates around the training with denser rewards, SFT priors, skill libraries, curated memory, or multi-agent orchestration. We argue the ceiling is an artifact of two failures of common practice. Signal starvation: group-relative RL with sparse outcome-only rewards yields a gradient only when a task's rollout group mixes successes and failures, so under-scaled exploration silences exactly the hardest, most instructive tasks. Policy drift: squeezing many updates out of a small task pool degrades the policy itself, as an unanchored objective lets the sampling distribution collapse exactly when saturation has already made informative groups rare. We present CANOPY (Coverage-ANchored On-P
    
[^41]: 面向监督微调的后训练科学

    Post-Training Science for Supervised Fine-Tuning

    [https://arxiv.org/abs/2609.01244](https://arxiv.org/abs/2609.01244)

    本文通过每次只改变一个变量的统一受控扫描实验，系统测量了监督微调中学习率、批大小、LoRA与全量微调等关键决策在Qwen3与Llama两类模型（稠密与混合专家架构）以及四个真实客户数据集上的表现，将SFT超参数选择从经验摸索转变为可复现的科学测量。

    

    每一次监督微调（SFT）运行都迫使我们做出同样的一系列决策，例如学习率、批大小、采用LoRA还是全量微调、训练多少个epoch、选择哪种优化器，以及向模型输入什么数据。这些决策通常在每次面对新模型和新数据集时都要从头重新摸索。本文在一个统一的测量工具下对它们进行测量：一种每次只改变一个控制变量的扫描方法，涵盖Qwen3和Llama两个模型家族中的稠密模型与混合专家模型，在四个真实世界的客户SFT数据集上，分别对LoRA和全量微调进行测试。这些数据集提供了一个受控的实验平台：每个任务都带有一个与客户共同构建的评估标准，其训练数据通过迭代式监督微调生成，即不断改进模型输出直到其通过该评估，因此监督目标在内部是一致的，而我们报告所依据的任务评判标准正是数据构建时所旨在满足的准则。我们探究最优学习率和批大小如何随……（原文摘要在此处截断）

    arXiv:2609.01244v1 Announce Type: cross  Abstract: Every supervised fine-tuning run forces the same chain of decisions, such as learning rate, batch size, LoRA or full fine-tuning, how many epochs, which optimiser, and what data to feed the model. Each of these is typically rediscovered from scratch for every new model and dataset. Here we measure them under one instrument: a sweep that varies one lever at a time, and spans dense and mixture-of-experts models in two families (Qwen3 and Llama), on four real-world customer SFT datasets, for both LoRA and full fine-tuning. These datasets give a controlled testbed: each task carries an evaluation built with the customer, and its training data is produced by iterative supervised fine-tuning that refines model outputs until they pass that evaluation, so the supervised target is internally consistent and the task judge we report against is the criterion the data was built to satisfy. We ask how the optimal learning rate and batch size move wi
    
[^42]: 从语言到行为：面向工业推荐排序的推荐原生序列Transformer扩展设计

    From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs

    [https://arxiv.org/abs/2609.01240](https://arxiv.org/abs/2609.01240)

    提出推荐原生的Transformer扩展框架ReST，通过双门控注意力编码器应对噪声化行为序列，通过重量级可复用编码器加轻量级交叉解码器的分解设计及共享前缀训练与服务机制，解决推荐排序中的计算不对称问题，实现工业推荐排序的高效规模化。

    

    扩展Transformer架构在语言建模领域带来了巨大的性能提升，但将这一方法移植到生产级排序系统中的行为序列建模却充满挑战：推荐系统在信号质量上存在差异——行为序列充满噪声、时间上不规则且监督信号稀疏；在计算不对称性上也存在差异——每个请求需要在严格的延迟预算下，用一份共享的用户历史对大量候选物品进行打分。我们提出了ReST，一个推荐原生的Transformer扩展框架。针对信号质量问题，它引入了一个序列编码器，包含双门控注意力、旋转位置与时间嵌入、稳定的残差归一化，以及仅在训练阶段使用的辅助目标。针对计算不对称性问题，它将排序分解为重量级可复用的编码器和轻量级交叉解码器，采用无投影的KV注意力和针对token的特定参数化，并将用户级共享前缀训练与共享前缀服务相结合，以实现计算高效的服务部署。

    arXiv:2609.01240v1 Announce Type: cross  Abstract: Scaling Transformers has driven large gains in language modeling, but transplanting this to behavior-sequence modeling in production ranking is challenging: recommendation differs in signal quality, where behavior sequences are noisy, temporally irregular, and sparsely supervised, and in computation asymmetry, where each request scores many candidates against one shared user history under tight latency budgets. We propose ReST, a recommendation-native Transformer scaling framework. For signal quality, it introduces a sequence encoder with dual-gated attention, rotary positional and temporal embedding, stabilized residual normalization, and training-only auxiliary objectives. For computation asymmetry, it factorizes ranking into a heavy reusable encoder and a lightweight cross decoder with projection-free KV attention and token-specific parameterization, coupling user-level shared-prefix training with shared-prefix serving for compute-o
    
[^43]: REFACTOR-VLA：类型化运动程序的无监督技能库学习

    REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs

    [https://arxiv.org/abs/2609.01215](https://arxiv.org/abs/2609.01215)

    REFACTOR-VLA 提出了一种清醒/睡眠两阶段框架，通过基于潜在世界模型 rollout 计算的行为等价核对运动程序片段进行无监督聚类，并生成 Hindley–Milner 风格的类型化 lambda 项来构建可复用技能库，从而提升 VLA 模型在长时程任务上的性能与可解释性。

    

    大多数视觉-语言-动作（VLA）模型——如 OpenVLA、π₀、RT-2、RDT-1B——都是单体式的：它们直接输出原始运动指令或短动作片段，而未将行为组织成可复用的抽象，因此在长时程任务上性能下降且难以解释。现有的技能发现方法回避了“两个动作序列何时在行为上等价”这一核心问题：要么对对比学习的嵌入进行聚类，要么把判断交给一个未针对机器人动力学进行校准的语言模型。我们提出 REFACTOR-VLA，一个用于学习可复用技能的清醒/睡眠系统。其睡眠阶段在行为等价核（Behavioral-Equivalence Kernel, BEK）下对运动程序片段进行聚类，该核由学习到的潜在世界模型 M_φ 的 rollout 计算得到；其清醒阶段基于受 Hindley–Milner 类型系统启发的词汇表生成类型化的 lambda 项，并由一个以技能库为条件的整流流动作解码器来消费。只有通过最小描述长度……（原文摘要在此处截断）

    arXiv:2609.01215v1 Announce Type: cross  Abstract: Most vision-language-action (VLA) models -- OpenVLA, $\pi_0$, RT-2, RDT-1B -- are monolithic: they emit raw motor commands or short action chunks without organizing behavior into reusable abstractions, so they degrade on long-horizon tasks and resist interpretation. Existing skill-discovery methods sidestep the core question of when two action sequences are behaviorally equivalent, either clustering contrastive embeddings or delegating the judgment to a language model uncalibrated to the robot's dynamics. We introduce REFACTOR-VLA, a wake/sleep system for learning reusable skills. Its sleep phase clusters motor-program fragments under a Behavioral-Equivalence Kernel (BEK) computed from rollouts of a learned latent world model $M_\phi$; its wake phase emits typed lambda terms over a Hindley--Milner-inspired vocabulary, consumed by a library-conditioned rectified-flow action decoder. Abstractions are admitted only if they pass Minimum De
    
[^44]: FPGA平台上Transformer推理部署的最新进展：综述

    Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey

    [https://arxiv.org/abs/2609.01212](https://arxiv.org/abs/2609.01212)

    本文通过系统性文献综述，梳理了在FPGA平台上部署Transformer模型推理的最新进展、趋势与设计选择，凸显了FPGA相比CPU/GPU在能效、延迟和现场部署灵活性方面的优势。

    

    随着基于Transformer架构的机器学习模型应用的快速且持续增长，高性能的部署需求日益迫切。在此背景下，高性能部署既涉及运行性能指标（如吞吐量和延迟），也涉及效率指标（如能耗）。在使用此类模型执行推理任务时，专用硬件加速器为中央处理器（CPU）和图形处理器（GPU）等常见部署方案提供了一种颇具吸引力的替代选择。现场可编程门阵列（FPGA）平台正是此类替代加速器的一个典型代表，它具有实现灵活、能效高、延迟更优以及适合现场部署等优势。本文研究了FPGA平台上Transformer推理的最新进展、发展趋势和设计选择。我们开展了一项系统性文献综述，提取了相关研究成果。

    arXiv:2609.01212v1 Announce Type: new  Abstract: With the rapid and continuous growth in the incorporation of machine learning models based on the Transformer architecture, capable deployment is in high demand. In this context, capable deployment refers to operational performance aspects, e.g., throughput and latency, as well as efficiency aspects, e.g., energy consumption. When it comes to the task of inference using such models, purpose-built hardware accelerators provide a lucrative alternative to common deployment choices, such as Central Processing Units (CPUs) and Graphics Processing Units (GPUs). The Field Programmable Gate Array (FPGA) platforms category is an example of such alternative accelerators, promising implementation flexibility, energy efficiency, improved latency and suitability for on-site deployment. We investigate the most recent advances, trends, and design choices for Transformer inference on FPGA platforms. We perform a systematic literature review, extracting 
    
[^45]: CopyShield：大语言模型版权防御的跨层级基准测试

    CopyShield: A Cross-Level Benchmark of Copyright Defenses in LLMs

    [https://arxiv.org/abs/2609.01161](https://arxiv.org/abs/2609.01161)

    CopyShield提出了一个跨干预层级（输出、行为、表示层）的版权防御基准，通过统一协议系统比较对比解码、DPO和激活干预三种方法，揭示了不同干预层级在大语言模型合规性、效用与退化行为之间的独特权衡。

    

    大语言模型可能会逐字复述其记忆的文本，然而版权防御方法通常是在互不兼容的协议下进行评估的。我们提出了CopyShield，这是一个受控基准，用于比较三种在不同干预层级上的代表性防御方法：对比解码（输出层）、直接偏好优化DPO（行为层）和激活干预（表示层）。我们在两个模型家族（LLaMA-3.1-8B和Mistral-7B-v0.3）上评估CopyShield，使用五本公有领域书籍构建受控记忆化数据，并采用统一协议来衡量字面泄漏、校准的非字面泄漏、效用和退化。研究发现，不同干预层级对应着截然不同的合规性-效用权衡。在LLaMA-3.1-8B上，对比解码几乎不引发退化（0-2%），但其字面抑制效果在NV-Recall 0.192-0.203处达到瓶颈。DPO几乎消除了字面泄漏（从0.263降至0.002），但会诱发释义循环退化。

    arXiv:2609.01161v1 Announce Type: new  Abstract: Large language models can reproduce memorized text verbatim, yet copyright defenses are usually evaluated under incompatible protocols. We introduce CopyShield, a controlled benchmark comparing three representative defenses at distinct intervention levels: contrastive decoding (output), Direct Preference Optimization (behavioral), and activation intervention (representation). We evaluate CopyShield on two model families, LLaMA-3.1-8B and Mistral-7B-v0.3, using controlled memorization over five public-domain books and a shared protocol measuring literal leakage, calibrated non-literal leakage, utility, and degeneracy. Across these methods, intervention level is associated with distinct compliance-utility trade-offs. On LLaMA-3.1-8B, contrastive decoding remains near-degeneracy-free (0-2%) but reaches a literal-suppression floor at NV-Recall 0.192-0.203. DPO nearly eliminates literal leakage (0.263 to 0.002) but induces paraphrase-loop deg
    
[^46]: 叠加潜在自编码器

    Superposed Latent Autoencoder

    [https://arxiv.org/abs/2609.01158](https://arxiv.org/abs/2609.01158)

    SLAE 通过学习到的叠加机制将多个高容量潜码绑定并叠加存储在单个内存张量中，以可抑制的结构化干涉取代传统自编码器不可逆的维度瓶颈，在相同存储预算下将重建误差最多降低 56%。

    

    自编码器通常通过将每个潜在表示变得更小来满足紧张的潜在内存预算，从而牺牲了表示能力。我们提出了一个不同的问题：能否将多个更宽的潜在表示一起存储？我们提出了叠加潜在自编码器（SLAE），它通过学习到的叠加方式共享存储，同时保持高容量的潜在表示。SLAE 将潜码转换为适合存储的编码，用随机化密钥将它们绑定，将多个编码叠加到单个内存张量中，并学习在解码前恢复每个潜码。在相同的存储预算下，SLAE 用可被抑制的结构化干涉取代了不可逆的维度瓶颈。在 CIFAR-10/100、SVHN、STL-10、Tiny ImageNet 以及广泛的内存预算范围内，SLAE 显著改善了重建-内存的权衡，在匹配的内存预算下，与传统自编码器相比，重建误差最多降低 56%。

    arXiv:2609.01158v1 Announce Type: cross  Abstract: Autoencoders typically meet tight latent-memory budgets by making each latent representation smaller, sacrificing representational capacity. We ask a different question: can multiple wider latents be stored together instead? We introduce the Superposed Latent Autoencoder (SLAE), which preserves high-capacity latent representations while sharing storage through learned superposition. SLAE transforms latents into storage-friendly codes, binds them with randomized keys, superposes multiple codes into a single memory tensor, and learns to recover each latent before decoding. Under the same storage budget, SLAE replaces irreversible dimensional bottlenecks with structured interference that can be suppressed. Across CIFAR-10/100, SVHN, STL-10, Tiny ImageNet, and a wide range of memory budgets, SLAE substantially improves the reconstruction--memory tradeoff, reducing reconstruction error by up to 56% over conventional autoencoders at matched 
    
[^47]: Transformer注意力中的缩放幂等性：OV配对几何与共享值代数

    Scaled Idempotence in Transformer Attention: Paired OV Geometry and Shared-Value Algebras

    [https://arxiv.org/abs/2609.01129](https://arxiv.org/abs/2609.01129)

    该论文发现Transformer注意力中存在一类稀疏的“缩放幂等”注意力头，其OV算子满足T²≈αT，并通过精确主坐标分解与K方向置乱实验证明这种代数闭合性质主要由K因子的训练方向所驱动。

    

    我们在Transformer注意力中识别出一种反复出现的代数规律：有效OV算子 T=OV^⊤ 的一个稀疏子集在复合下近乎闭合，即 T²≈αT。在参数规模覆盖2.8B至235B的六个预训练模型端点上，3.98%–8.00%的注意力头达到平方闭合对齐度 P≥0.9，而任何匹配的层内O/V失配情况均未达到该水平。一个精确的主坐标分解 T=Q_OKQ_V^⊤ 与 T²=Q_O(KDK)Q_V^⊤ 将支持集内的传输与读写返回几何分离开来。在九个MHA/GQA模型的全部7,304个注意力头上，仅在保持奇异值、范数、因子跨度与主角度的前提下打乱K的方向，就会使中位闭合度从0.336降至1.04×10⁻⁴；训练所得的方向在98.64%的注意力头以及每一层中都占优。构造性搜索表明，高闭合度在每个被考察的层中都是可行的，但通常并未被实现。关于t中的回溯轨迹……

    arXiv:2609.01129v1 Announce Type: new  Abstract: We identify a recurrent algebraic regularity in Transformer attention: a sparse subset of effective OV operators $T=OV^\top$ nearly closes under composition, $T^2\approx\alpha T$. Across six pretrained endpoints spanning 2.8B--235B parameters, 3.98--8.00% of heads reach squared closure alignment $\mathcal{P}\geq0.9$, while no matched within-layer O/V mismatch does. An exact principal-coordinate factorization, $T=Q_OKQ_V^\top$ and $T^2=Q_O(KDK)Q_V^\top$, separates within-support transport from read--write return geometry. Across all 7,304 heads in nine MHA/GQA models, scrambling only the orientation of $K$ while preserving singular values, norms, factor spans, and principal angles reduces median closure from 0.336 to $1.04\times10^{-4}$; trained orientation wins for 98.64% of heads and in every layer. Constructive searches show that high closure is feasible in every surveyed layer, but usually not attained. Retrospective trajectories in t
    
[^48]: 在线自适应何时在边缘端有价值？面向时间序列预测的无泄漏预热、学习率选择与资源权衡评估

    When Does Online Adaptation Pay on the Edge? A Leakage-Free Evaluation of Warmup, Learning-Rate Selection, and Resource Trade-offs for Time-Series Forecasting

    [https://arxiv.org/abs/2609.01126](https://arxiv.org/abs/2609.01126)

    该研究在无泄漏流式评估协议下揭示了测量在线自适应收益时的两个关键偏差来源——基线预热预算的双向效应以及优化器比较中学习率未公平调整的问题，并提出利用漂移前验证切片来公平选择预热预算和在线学习率。

    

    在线自适应可以在分布漂移下帮助边缘端的时间序列预测，但其测得的收益对评估选择非常敏感。我们在无泄漏的流式协议下研究了六个公开的多变量数据流，包括建筑传感器和智能电表数据。我们识别出两个额外的比较偏差来源。首先，静态基线的预热预算具有双向效应：预热不足会使基线训练不充分，而过多的预热会降低其在漂移前的泛化能力。在六个数据集-骨干网络设置中，估计的自适应收益在1,000到20,000步的预热范围内变化幅度达3.0到18.8个百分点。其次，在共享的默认学习率下比较带动量的SGD（SGD+m）和Adam，会将优化器质量与学习率敏感性混为一谈。我们使用漂移前预留的验证切片来选择预热预算和每个优化器的在线学习率，而不访问测试数据。

    arXiv:2609.01126v1 Announce Type: new  Abstract: Online adaptation can help edge time-series forecasting under distribution drift, but its measured benefit is sensitive to evaluation choices. We study six public multivariate streams, including building-sensor and smart-meter data, under a leakage-free streaming protocol. We identify two additional sources of comparison bias. First, the warmup budget of the static baseline has a two-sided effect: insufficient warmup undertrains the baseline, whereas excessive warmup can degrade its pre-drift generalization. Across six dataset-backbone settings, the estimated adaptation benefit changes by 3.0 to 18.8 percentage points (pp) over the 1,000-20,000-step warmup range. Second, comparing SGD with momentum (SGD+m) and Adam at a shared default learning rate conflates optimizer quality with rate sensitivity. We select both the warmup budget and each optimizer's online rate using a held-out pre-drift validation slice without accessing test data. Un
    
[^49]: 复现TRACE：面向实践者的阈值与粒子预算指南

    Replicating TRACE: A Practitioner's Guide to Its Threshold and Particle Budget

    [https://arxiv.org/abs/2609.01108](https://arxiv.org/abs/2609.01108)

    本研究独立复现了TRACE的核心结果，并揭示其最优阈值并非固定常数，而是锚定于定义真值的边界（约delta/2乘以估计器校准系数），且在单一全局阈值下TRACE主要恢复的是直接相邻影响的因果图。

    

    TRACE（Math & Lienhart，arXiv:2602.01135）通过对预训练自回归序列模型中每个位置的条件互信息估计值施加固定阈值 tau，从模型中读取事件类型之上的因果图。我们独立复现了其主要的合成实验结果：当 tau 在验证集上选取时，在词表规模为1000时，相对精确干预真值的平均每序列F1达到0.90-0.91（论文为0.91），在100至2000的规模范围内为0.86-0.91。首先，最优阈值并非锚定于任何常数，而是锚定于真值边界：在每一规模下，tau* 处的误差都恰好跨越定义真值的 delta = 0.05 边界（被遗漏的真边恰好位于边界之上，被接受的假边恰好位于边界之下），且盲选的最优值落在 delta/2 乘以估计器校准系数附近，并在规模5000下通过样本外验证得到确认。其次，在单一全局阈值下，TRACE 大多恢复的是一种直接的、相邻影响图：滞后为1的真边召回率达到0.97-0.99，而……（摘要在此处截断）

    arXiv:2609.01108v1 Announce Type: new  Abstract: TRACE (Math & Lienhart, arXiv:2602.01135) reads causal graphs over event types out of a pretrained autoregressive sequence model by thresholding a per-position conditional-mutual-information estimate at a fixed tau. We independently replicate its headline synthetic result: with tau selected on a validation split, mean per-sequence F1 against exact interventional truth reaches 0.90-0.91 at vocabulary size 1000 (paper: 0.91) and 0.86-0.91 from 100 to 2000. First, the optimal threshold is pinned to the truth margin, not to any constant: at every size the errors at tau* straddle the delta = 0.05 margin defining ground truth (missed true edges lie just above it, accepted false ones just below), and the blind optimum lands near delta/2 times the estimator's calibration, confirmed out of sample at 5000. Second, at a single global threshold TRACE mostly recovers a direct, adjacent-influence graph: lag-1 true edges are recalled at 0.97-0.99, whil
    
[^50]: 使用深度学习与稀疏建模的神经符号回归

    Neural Symbollic Regression Using Deep Learning and Sparse Modelling

    [https://arxiv.org/abs/2609.01102](https://arxiv.org/abs/2609.01102)

    该论文提出一种将神经网络作为功能性预处理器的神经符号回归框架，先由神经网络学习目标函数的平滑抗噪近似，再用LASSO提取稀疏可解释的闭式数学表达式，从而兼顾深度学习的鲁棒性与符号回归的可解释性。

    

    符号回归（SR）旨在寻找简洁的数学表达式来表示数据中的基本关系，提供超越黑盒模型的可解释性与科学理解能力。然而，遗传编程等传统方法面临可扩展性挑战且对噪声高度敏感，而SINDy等稀疏回归技术则严重依赖预先设定的特征库。在本工作中，我们提出了一种神经符号回归（NSR）框架，将神经网络视为符号发现的功能性预处理器。我们的方法采用解耦的流水线：神经网络首先在一个交互感知的非线性特征空间中学习目标函数的平滑、抗噪近似，随后应用LASSO提取稀疏且可解释的闭式表达式。为提升预测精度与符号保真度，通过集成分布式……（原文摘要至此截断）

    arXiv:2609.01102v1 Announce Type: new  Abstract: Symbolic Regression (SR) seeks to find succinct mathematical expressions that represent the fundamental relationships within data, providing interpretability and scientific understanding that exceeds that of black-box models. Nevertheless, traditional methods like Genetic Programming face challenges with scalability and are highly sensitive to noise, while sparse regression techniques such as SINDy rely significantly on predetermined feature libraries. In this work, we present a Neural Symbolic Regression (NSR) framework that treats neural networks as functional preconditioners for symbolic discovery. Our approach uses a decoupled pipeline: a neural network first learns a smooth, noise-robust approximation of the target function in an interaction- aware nonlinear feature space. LASSO is then applied to extract sparse, interpretable closed-form expressions. To improve predictive accuracy and symbolic fidelity by integrating distributed hy
    
[^51]: 潜意识学习即特质方向漂移：SFT蒸馏下的机制与针对性控制

    Subliminal Learning as Trait-Direction Drift: A Mechanism and Targeted Control under SFT Distillation

    [https://arxiv.org/abs/2609.01091](https://arxiv.org/abs/2609.01091)

    本文提出“特质方向漂移”机制来解释潜意识学习现象——偏置教师数据中可测量的偏好差距在监督微调中累积为学生的行为迁移，并据此提出探测空间走廊正则化这一针对性防御方法，在蒸馏过程中约束模型沿校准特质方向的漂移。

    

    超越预期的能力之外，模型蒸馏还可能从教师模型传递隐藏特质。一个被系统提示词偏置的教师模型可以生成语义上干净的训练数据（例如数字序列），但这些数据仍会使下游学生模型继承隐藏偏好，这种现象被称为“潜意识学习”。先前的研究已识别出该过程的若干环节，但信号在训练过程中如何积累并产生行为迁移仍不清楚，这使得有针对性的缓解难以实现。我们提出并验证了“特质方向漂移”作为潜意识学习的机制：偏置生成会在教师数据中产生可测量的偏好差距，而学生可识别的差距会在监督微调期间诱导特质对齐的参数更新，这些更新逐步累积为行为迁移。基于这一机制，我们提出了探测空间走廊正则化，这是一种针对性的防御方法，在蒸馏过程中约束沿校准特质方向的漂移……

    arXiv:2609.01091v1 Announce Type: new  Abstract: Beyond intended capabilities, model distillation can transfer hidden traits from a teacher. A teacher biased by a system prompt can generate semantically clean training data, such as numeric sequences, that still causes a downstream student to inherit the hidden preference, a phenomenon known as subliminal learning. Prior work has identified several parts of this process. How the signal builds up during training and produces behavioral transfer remains unclear, making targeted mitigation difficult. We propose and validate trait-direction drift as a mechanism for subliminal learning: biased generation creates measurable preference gaps in teacher data, and student-recognizable gaps induce trait-aligned updates during supervised fine-tuning that accumulate into behavioral transfer. Guided by this mechanism, we propose probe-space corridor regularization, a targeted defense that constrains drift along a calibrated trait direction during dis
    
[^52]: Modelpedia：面向AI元科学的模型发现目录

    Modelpedia: A Catalog of Model Findings for the Meta-Science of AI

    [https://arxiv.org/abs/2609.01090](https://arxiv.org/abs/2609.01090)

    提出了Modelpedia——一个利用大语言模型自动从已发表论文中提取AI模型相关发现、将其与模型、数据集、方法和概念关联，并汇总为可搜索公共目录的框架，同时基于该目录对AI社区如何研究模型进行了元分析。

    

    关于AI模型的科学知识产生的速度已超过社区能够整理的速度。每隔几个月，一个新的大型基础模型就会重塑该领域，数百篇论文、博客和技术报告记录着每个模型的表现或失败之处。然而，这些发现仍然分散，实际上无法有效检索。为了解决这一差距，我们提出了Modelpedia，这是一个自动化的、由大语言模型辅助的框架，它从已发表的论文中提取关于模型的发现，将其与所涉及的模型、数据集、方法和概念相关联，并将结果汇总到一个可搜索的公共目录中。将该原型应用于ICLR 2024和2025年被接收的论文，我们提取了一千多项发现，并将该目录本身作为研究对象，对社区如何研究模型进行了元分析。现在，我们邀请社区探索、贡献并基于这个开放目录进行构建，帮助将模型发现确立为AI元科学的共享基础。

    arXiv:2609.01090v1 Announce Type: new  Abstract: Scientific knowledge about AI models is produced faster than the community can organize it. Every few months a new foundation model reshapes the field and hundreds of papers, blogs, and technical reports document how each behaves or fails. Yet, these findings remain scattered and effectively unretrievable. To address this gap we present Modelpedia, an automated, LLM-assisted framework that extracts findings about models from published papers, links it to the model, dataset, method, and concept it concerns, and aggregates the result into a searchable public catalog. Applying the prototype to accepted ICLR 2024 and 2025 papers, we extract over a thousand findings and, treating the catalog itself as an object of study, run a meta-analysis of how the community investigates models. Now, we invite the community to explore, contribute to, and build on the open catalog, and to help establish model findings as a shared foundation for the meta-sci
    
[^53]: 让置信度改变，而非预测：面向事后校准的保持预测修复方法

    Let Confidence Change, Not the Prediction: Prediction-Preserving Repair for Post-hoc Calibration

    [https://arxiv.org/abs/2609.01072](https://arxiv.org/abs/2609.01072)

    本文提出CORD——首个通过修复完整校准概率向量来严格保持原始top-1预测不变、仅修正置信度的事后校准后拟合适配器，并引入TPCR指标量化校准器改变预测的频率。

    

    事后校准用于修正模型报告的置信度，然而多类别校准器也可能同时改变相应的top-1预测。准确率仅能捕捉这些变化对正确性的净效应，而无法反映预测被改变的频率；Top-1预测改变率（TPCR）正是用于衡量这一频率的指标。我们提出了面向Top-1决策保持的校准器输出修复方法（CORD），这是首个通过修复完整校准概率向量来实现严格预测保持的后拟合适配器。仅凭原始输出和校准输出，CORD即可确定分配给原始top-1类别的概率质量；校准后的条件分布则将剩余概率质量分配给其他类别，从而得到一个修复后的向量，其argmax能够恢复原始预测。在校准集上，只要条件允许，CORD会协调修复后的概率质量，以保留校准输出在原始预测上的平均概率质量。该适配器仅改变……（原文在此截断）

    arXiv:2609.01072v1 Announce Type: new  Abstract: Post-hoc calibration corrects reported confidence, yet a multiclass calibrator can also change the associated top-1 prediction. Accuracy captures only the net effect of these changes on correctness, not how often predictions change; the Top-1 Prediction Change Rate (TPCR) instead measures this frequency. We propose Calibrator-Output Repair for Top-1 Decision Preservation (CORD), the first post-fit adapter to impose exact prediction preservation by repairing the full calibrated probability vector. From the original and calibrated outputs alone, CORD determines the mass assigned to the original top-1. The calibrated conditional distribution allocates the remaining mass over the other classes, yielding a repaired vector whose own argmax recovers the original prediction. On the calibration split, CORD coordinates the repaired masses to retain the calibrated outputs' mean mass on original predictions whenever attainable. The adapter alters ne
    
[^54]: 通过MPC求解器梯度引导加速强化学习实现权重可变的模型预测控制

    Accelerating Reinforcement Learning via MPC Solver-Gradient Guidance for Weights-varying MPC

    [https://arxiv.org/abs/2609.01061](https://arxiv.org/abs/2609.01061)

    提出SG-RL方法，利用可微MPC求解器的梯度信息引导强化学习在线自适应地调整MPC代价函数权重，从而在保持低偏差的同时显著提升样本效率、加速学习过程。

    

    在模型预测控制（MPC）中，代价函数权重塑造了闭环行为，然而环境条件的变化常常使固定的参数化设置变得次优，这促使人们需要依赖于情境的在线自适应调整。学习此类策略十分困难，因为系统行为隐式地依赖于数值MPC求解结果，从而对策略参数形成非线性、可能非光滑且长时程的依赖关系。这带来了一个偏差-方差权衡问题：强化学习（RL）基于环境样本优化实际闭环回报，但样本效率低下；而基于梯度的策略学习（GB-PL）利用可微MPC提供的低方差求解器梯度来优化预测轨迹上的代理损失，但在模型失配情况下可能产生偏差。我们提出了求解器梯度引导强化学习（SG-RL），这是一种针对基于RL的在线MPC代价权重自适应的求解器敏感性增强方法。SG-RL保持采样的闭环回报作为（优化目标）……

    arXiv:2609.01061v1 Announce Type: cross  Abstract: In Model Predictive Control (MPC), cost-function weights shape closed-loop behavior, yet changing conditions often make fixed parametrizations suboptimal and motivate context-dependent online adaptation. Learning such policies is difficult because behavior depends implicitly on numerical MPC solutions, producing nonlinear, potentially nonsmooth, long-horizon dependencies on policy parameters. This creates a bias-variance tradeoff: Reinforcement Learning (RL) optimizes realized closed-loop return from environment samples but is sample-inefficient, whereas Gradient-Based Policy Learning (GB-PL) uses low-variance solver gradients from differentiable MPC to optimize surrogate losses on predicted trajectories but can be biased under model mismatch. We propose Solver-Gradient Guided Reinforcement Learning (SG-RL), a solver-sensitivity augmentation for RL-based online MPC cost-weight adaptation. SG-RL keeps sampled closed-loop return as the o
    
[^55]: SAGE：面向子群体的生成式增强方法用于缓解虚假相关性

    SAGE: Subpopulation-Aware Generative Enhancement for Mitigating Spurious Correlations

    [https://arxiv.org/abs/2609.01051](https://arxiv.org/abs/2609.01051)

    提出SAGE——一种两阶段生成式增强框架，利用聚类得到的子标签微调条件生成模型生成针对性样本，在无需虚假属性先验知识的情况下从数据层面缓解机器学习中的虚假相关性。

    

    虚假相关性对现代机器学习的鲁棒性构成了重大挑战。数据集分布中固有的不平衡往往导致传统的经验风险最小化（ERM）模型依赖多数群体的虚假属性进行分类，从而导致在少数群体上表现不佳。当虚假属性不可用时，这一问题变得尤为棘手。现有的无组标签方法通常对少数群体或被错误分类的真实训练样本进行上采样；然而重复相同实例会降低有效多样性并助长过拟合。为了在缺乏先验知识的情况下，从以数据为中心的视角缓解这些虚假相关性，我们提出了子群体感知生成式增强，这是一个两阶段的生成式增强框架。利用聚类得到的子标签和类别标签，我们对条件生成模型和文本编码器进行微调，生成针对性的……（原文摘要在此处截断）

    arXiv:2609.01051v1 Announce Type: new  Abstract: Spurious correlations pose a significant challenge to the robustness of modern machine learning. The inherent imbalance in dataset distributions often leads traditional Empirical Risk Minimization (ERM) models to rely on majority spurious attributes for classification, resulting in poor performance on minority groups. This problem becomes particularly challenging when the spurious attributes are unavailable. Existing group-label-free methods often upsample minority groups or misclassified real training examples; repeating the same instances can reduce effective diversity and encourage overfitting. To mitigate these spurious correlations from a data-centric perspective in the absence of prior knowledge, we introduce Subpopulation-Aware Generative Enhancement (SAGE), a two-stage generative augmentation framework. Using cluster-derived sub-labels and class labels, we fine-tune a conditional generative model and text encoder, generating targ
    
[^56]: 从截断到承诺：均匀离散扩散中的持久上下文

    From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion

    [https://arxiv.org/abs/2609.01043](https://arxiv.org/abs/2609.01043)

    提出一种无需训练的承诺式揭示采样（CRS），将选定的词元作为持久上下文插入后续模型输入，使均匀离散扩散模型的并行预测能在序列级选择上保持一致。

    

    均匀状态离散扩散模型并行更新所有词元，同时保持每个位置都可被修改。即使常用的 top-p 规则在一个位置只留下一个候选，该选择也仅影响当前的反向步骤，并可在下一个采样步骤中被修改。我们探讨当被选中的假设转而成为后续预测的持久上下文时会发生什么变化。为此，我们提出了承诺式揭示采样，这是一种无需训练的采样器，它存储被选中的 argmax 词元，并将其插入后续的模型输入中。我们的分析为“更晚做出选择”和“保持被选词元可见”提供了理论依据：在精确的前向过程下，随着噪声降低，选择干净词元的贝叶斯误差不会增加；而在一个简单的潜变量模式模型中，保持被选词元可见有助于后续的并行预测在相同的序列级选择上达成一致。实证上，在 Duo-distilled 模型上的成对实验（摘要在此处截断）……

    arXiv:2609.01043v1 Announce Type: cross  Abstract: Uniform-state discrete diffusion models update all tokens in parallel while keeping every position revisable. Even when the commonly used top-$p$ rule leaves only one candidate at a position, that choice affects only the current reverse step and can be revised at the next sampling step. We ask what changes when selected hypotheses instead become persistent context for later predictions. We therefore propose committed reveal sampling (CRS), a training-free sampler that stores selected argmax tokens and inserts them into subsequent model inputs. Our analysis gives a rationale for selecting later and for keeping selected tokens visible. Under the exact forward process, the Bayes error of selecting a clean token cannot increase as noise decreases, while in a simple latent-mode model, keeping the selected token visible helps later parallel predictions agree on the same sequence-level choice. Empirically, paired experiments on Duo-distilled 
    
[^57]: ViTAMINS：使用合成困难负样本训练自监督视觉Transformer的实证研究

    ViTAMINS: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives

    [https://arxiv.org/abs/2609.01041](https://arxiv.org/abs/2609.01041)

    ViTAMINS通过向自监督视觉Transformer的对比学习预训练中引入合成困难负样本，以极小的改动获得了涌现的语义分类能力（最高提升11.3%）并大幅节省计算资源（ViT-B超越ViT-L的V-JEPA），证明对比学习仍是生成式与自蒸馏方法的强大替代方案。

    

    我们提出了ViTAMINS，这是一种将合成困难负样本融入无监督视觉Transformer预训练以提升表示质量的方法。我们的方法在ImageNet以及迁移学习、图像检索、复制检测和图像/视频分割等任务上进行了全面的基准测试。值得注意的是，我们提出的负样本带来了涌现特性：学到的表示包含图像语义内容的显式信息，并可充当优秀的分类器（相比基线最高提升11.3%）。ViTAMINS通过对现有对比学习框架的简单修改实现了这些优势，在超越竞争方法的同时更加节省资源，例如我们的ViT-B模型超越了采用ViT-L的V-JEPA。我们的发现促使人们重新审视对比学习，将其视为主导的生成式和自蒸馏方法的一种更简单却强大的替代方案。

    arXiv:2609.01041v1 Announce Type: cross  Abstract: We introduce ViTAMINS, a method that integrates synthetic hard negatives into unsupervised vision transformer pretraining to improve representation quality. Our approach is thoroughly benchmarked on ImageNet and transfer learning, image retrieval, copy detection, and image, video segmentation tasks. Notably, our proposed negatives give rise to emergent properties, where learned representations contain explicit information about the semantic content of an image and serve as excellent classifiers (up to +11.3% over baselines). ViTAMINS achieves these benefits through simple modifications to existing contrastive frameworks and outperforms competing methods while being more resource efficient, e.g., our ViT-B surpasses V-JEPA with ViT-L. Our findings motivate reconsidering contrastive learning as a simpler yet powerful alternative to dominant generative and self-distillation approaches.
    
[^58]: 自由派生，节制行动：面向递归LLM智能体树的渐进式风险授予

    Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees

    [https://arxiv.org/abs/2609.01035](https://arxiv.org/abs/2609.01035)

    提出渐进式风险授予（PRV）机制，通过托管轨迹级风险预算并在分支激活时逐步扣减，为递归LLM智能体树中不可逆行动的授权证明了任意时刻的危害上界，实现“自由派生、节制行动”的安全权衡。

    

    递归LLM智能体可以通过派生专门智能体来拓展其搜索范围。某些分支随后会请求用于发送数据或部署代码的工具。那么何时应当授予某个分支行动权限？我们区分了沙箱派生（sandbox spawning，即通过外部控制防止特定危害的发生）与能力激活（capability activation，即被选中的分支跨越不可逆行动的边界）。渐进式风险授予（Progressive Risk Vesting, PRV）将一个轨迹级的风险预算进行托管，并在分支被激活时逐步扣减该预算。我们为自适应生成的树证明了任意时刻的危害上界。分支结果之间可能存在依赖关系，但每个局部证书必须在完整的激活前历史（包括用于选择该请求的信息）条件下保持有效。当激活门槛、分支收费和计算约束保持固定时，延迟授予能够保留不可撤销派生收费机制下所有可用的策略。边际风险估计在分支选择之后仍可能失效……

    arXiv:2609.01035v1 Announce Type: new  Abstract: Recursive LLM agents can broaden their search by spawning specialists. Some branches later request tools that send data or deploy code. When should a branch receive authority to act? We distinguish sandbox spawning, in which external controls prevent the specified harm, from capability activation, in which a selected branch crosses an irreversible-action boundary. Progressive Risk Vesting (PRV) holds a trajectory-level risk budget in escrow and debits it as branches are activated. We prove an anytime harm bound for adaptively generated trees. Branch outcomes may be dependent, but each local certificate needs to remain valid conditional on the full pre-activation history, including the information used to select the request. When activation gates, branch charges, and compute constraints are held fixed, delayed vesting preserves every policy available under irrevocable spawn charging. Marginal risk estimates can still fail after branch sel
    
[^59]: 稳定边缘上梯度下降的多重时间尺度：中心流的微扰推导

    The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow

    [https://arxiv.org/abs/2609.01034](https://arxiv.org/abs/2609.01034)

    本文通过将损失函数分解为 $f = g + \varepsilon h$ 的微扰分析，首次为深度学习稳定边缘处梯度下降的中心流提供了系统性推导，并揭示出其中存在快速振荡、中间自稳定与缓慢中心流演化三个时间尺度。

    

    Cohen等人（2025）提出的中心流是深度学习中稳定边缘处梯度下降的一个在经验上准确的连续时间模型，然而其推导是启发式的。我们提出了一种微扰机制，在该机制下中心流是梯度下降的极限：我们假设损失函数分解为 $f = g + \varepsilon h$；在 $\varepsilon \to 0$ 的极限下，学习率为 $\eta$ 的梯度下降动力学收敛到 $h$ 的梯度流，且该梯度流被约束在锐度至多为 $2/\eta$ 的 $g$ 的极小值点上。我们的方法是形式化的而非严格证明的；它将梯度下降视为关于 $\varepsilon$ 的奇异摄动动力系统。由此涌现出三个时间尺度：沿最锐利方向的快速振荡时间尺度、自稳定机制的中间时间尺度，以及沿 $g$ 极小值点动力学的慢时间尺度——即中心流。利用多尺度方法，一（原文摘要在此处被截断）

    arXiv:2609.01034v1 Announce Type: new  Abstract: The central flow of Cohen et al. (2025) is an empirically accurate continuous-time model of gradient descent at the edge of stability in deep learning, However, its derivation is heuristic. We propose a perturbative regime in which the central flow is the limit of gradient descent: we assume that the loss decomposes as $f = g + \varepsilon h$; in the limit $\varepsilon \to 0$, the dynamics of gradient descent with learning rate $\eta$ converge to the gradient flow of $h$ constrained to the minimizers of $g$ of sharpness at most $2/\eta$. Our approach is formal rather than rigorous; it treats gradient descent as a singularly perturbed dynamical system in $\varepsilon$. Three timescales emerge: a fast timescale of oscillations along the sharpest direction, an intermediate timescale of the self-stabilization mechanism, and a slow timescale of the dynamics along the minimizers of $g$-the central flow. Using the method of multiple scales, a c
    
[^60]: 网页价格提取：技术现状与一种自适应的无浏览器实现

    Web Price Extraction: State of the Art and an Adaptive Browserless Implementation

    [https://arxiv.org/abs/2609.01030](https://arxiv.org/abs/2609.01030)

    该论文系统梳理了网页价格提取四大类方法的优劣权衡，并提出了一种自适应的无浏览器价格提取实现，兼顾速度、成本与跨网站结构的适应性。

    

    从网站中提取价格是电子商务中市场监测、价格比较和商业分析的一项关键任务。现有方法大致可分为四类，理解它们在准确性和可扩展性之间的权衡对于选择合适的提取策略至关重要。经典方法依赖于人工编写的包装器和从标注页面中归纳的规则，准确性高，但对网页结构变化的适应性差，且需要大量的维护工作。基于浏览器的方法（如使用Selenium和Puppeteer等工具）能够处理动态JavaScript内容，但消耗大量计算资源且可扩展性差。无浏览器（browserless）方法通过HTTP请求直接获取HTML，在速度和成本方面具有显著优势，但依赖于针对特定网站校准的规则。基于机器学习和大型语言模型的方法具有良好的适应性，但需要训练数据和大（摘要在此处截断）

    arXiv:2609.01030v1 Announce Type: cross  Abstract: Price extraction from websites is a key task for market monitoring, price comparison, and business analytics in e-commerce. Existing approaches can be broadly divided into four groups, and understanding their trade-offs in accuracy and scalability is essential for selecting suitable extraction strategies. Classical methods rely on manually written wrappers and rule induction from labeled pages, offering high accuracy but adapting poorly to structural changes and requiring considerable maintenance effort. Browser-based methods, using tools such as Selenium and Puppeteer, handle dynamic JavaScript content but consume large computational resources and scale poorly. Browserless approaches retrieve HTML directly via HTTP requests, offering significant gains in speed and cost, but rely on rules calibrated for specific sites. Methods based on machine learning and large language models offer adaptability but require training data and substanti
    
[^61]: SinkPruner：面向多模态大语言模型的无Sink视觉token剪枝方法

    SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

    [https://arxiv.org/abs/2609.01004](https://arxiv.org/abs/2609.01004)

    提出无需训练的视觉token剪枝框架SinkPruner，通过过滤高度冗余的高范数离群token并缓解注意力汇聚现象，在保持多模态理解能力的同时实现高效的多模态大语言模型推理。

    

    尽管多模态大语言模型（MLLM）具有强大的多模态理解能力，但其在处理长视觉token序列时会产生巨大的计算开销。为降低推理成本，近期研究探索了基于视觉中心策略或文本引导策略的视觉token剪枝方法。然而，这些方法往往忽视了高范数离群token（即特征范数异常大的token），导致次优的剪枝决策。在本工作中，我们证明这类高范数离群token在特征维度和空间维度上都高度冗余，但现有方法却常常错误地将其作为信息线索而保留。受此观察启发，我们提出了SinkPruner，一个无需训练的视觉token剪枝框架，用于实现高效的MLLM推理。SinkPruner遵循由粗到细的设计，包含两个关键模块：一个用于过滤高范数冗余并缓解注意力汇聚（attention sink）现象的视觉净化器……

    arXiv:2609.01004v1 Announce Type: cross  Abstract: Despite their strong multimodal understanding ability, multimodal large language models (MLLMs) incur substantial computational overhead when processing long visual token sequences. To reduce inference costs, recent studies have explored visual token pruning through vision-centric or text-guided strategies. However, these methods often overlook high-norm outlier tokens, i.e., tokens with abnormally large feature norms, leading to suboptimal pruning decisions. In this work, we show that such high-norm outlier tokens are highly redundant in both feature and spatial dimensions, yet are often mistakenly preserved as informative cues by existing methods.   Motivated by this observation, we propose SinkPruner, a training-free visual token pruning framework for efficient MLLM inference. SinkPruner follows a coarse-to-fine design with two key modules: a visual sanitizer that filters high-norm redundancies and alleviates attention sink and atte
    
[^62]: 正确的框架，错误的规则：文化线索暴露了它们本意想弥补的金融知识差距

    Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close

    [https://arxiv.org/abs/2609.00999](https://arxiv.org/abs/2609.00999)

    该论文提出“规范多元性”这一新评估设定，通过将框架选择与框架内正确性分离，揭示了“刻板印象陷阱”——文化线索虽能引导大模型选择伊斯兰金融框架，却在框架内暴露出高达57%至66%的错误率，表明传统二选一评估会严重高估模型的文化对齐能力。

    

    当一个问题在不同规范框架下都有有效答案时，语言模型必须决定采用哪个框架，以及它能否在该框架内正确作答。我们将这种情境称为“规范多元性”，并以伊斯兰金融为研究对象，采用一种将框架选择与框架内正确性区分开来的四选一分类法进行研究。这种区分揭示了“刻板印象陷阱”：文化线索引导模型走向某一框架，但模型却在该框架内选择了错误的答案。在十二个模型、两种语言和五十个人口统计信号的测试中，文化线索会改变模型的框架选择，并暴露出显著的准确率差异，尤其是在非前沿模型中。在最强信号的作用下，大型开源权重模型有97%的概率选择伊斯兰金融框架。若采用二选一的评估方式，将会报告近乎完美的对齐度，尽管其中57%至66%的选择实际上是错误的。这些发现为……提供了依据，但并未……（原文摘要在此处截断）

    arXiv:2609.00999v1 Announce Type: cross  Abstract: When a question has valid answers under different normative frameworks, a language model must decide which framework to use and whether it can answer correctly within it. We call this setting normative pluralism and study it in Islamic finance using a four-choice taxonomy that separates framework selection from within-framework correctness. This separation reveals the stereotype trap: a cultural cue steers a model toward one framework, but the model selects an incorrect answer within that framework. Across twelve models, two languages, and fifty demographic signals, cultural cues change framework selection and reveal substantial differences in accuracy, especially among non-frontier models. Under the strongest signal, large open-weight models select the Islamic framework 97% of the time. A two-choice evaluation would report near-perfect alignment, although 57--66% of those selections are incorrect. These findings motivate, but do not d
    
[^63]: 面向大语言模型生成文本的嵌入式条件独立性检验及其在德国联邦议院演讲中的应用

    Embedded Conditional Independence Tests for Large Language Model Generated Text with an Application to German Parliament Speeches

    [https://arxiv.org/abs/2609.00946](https://arxiv.org/abs/2609.00946)

    本文提出嵌入式条件独立性检验（eCITs），通过将LLM生成的文本及其源文本嵌入到表示空间后再进行条件独立性检验，从而判断模型输出是否携带源文本之外的额外信息，并将其应用于德国议会演讲数据的分析。

    

    条件独立性检验（CITs）用于检验在给定第三个随机对象 Z 的条件下，两个随机对象 X 和 Y 之间是否存在条件依赖关系。现有的 CITs 对高维数据的适用性有限，尤其是像文本这样的多模态数据。然而，我们表明此类检验对大语言模型（LLM）的输出具有重要意义：即检验从源文本 Z 生成的输出 X 是否携带超出 Z 本身所含信息之外的属性 Y 的信息。为此，我们提出了嵌入式条件独立性检验（eCITs），该方法对 X 和 Z 进行嵌入，并将现有的 CIT 应用于所得的表示以及 Y。我们证明，只要 Z 的嵌入是充分的，即保留了 Z 所携带的关于 Y 或 X 的表示的信息，原假设就会从 X 和 Z 转移到它们的表示上，因此对嵌入后假设有效的 CIT 对原始假设同样有效。我们进一步给出了等变性的相关条件……

    arXiv:2609.00946v1 Announce Type: cross  Abstract: Conditional independence tests (CITs) test for conditional dependence between two random objects $X$ and $Y$ given a third random object $Z$. Existing CITs have limited applicability to high-dimensional data, especially multimodal data like text. However, we show that such tests are of interest for large language model (LLM) outputs, where we test whether an output $X$ generated from a source text $Z$ carries information about an attribute $Y$ beyond $Z$ itself. For this purpose, we propose embedded CITs (eCITs), which embed $X$ and $Z$ and apply an existing CIT to the resulting representations and to $Y$. We show that, provided the embedding of $Z$ is sufficient, i.e. retains the information $Z$ carries about either $Y$ or the representation of $X$, the null hypothesis transfers from $X$ and $Z$ to their representations, so that a CIT valid for the embedded hypothesis is valid for the original one. We further give conditions for equiv
    
[^64]: DualStake：深度研究智能体中的双路径置信度校准

    DualStake: Dual-Path Confidence Calibration in Deep Research Agents

    [https://arxiv.org/abs/2609.00935](https://arxiv.org/abs/2609.00935)

    提出DualStake双路径置信度校准方法，通过在每次检索后引出证据置信度并在答案生成后引出答案置信度，利用边界裁剪的置信度相关stake奖励将两者与答案正确性联合对齐，有效缓解深度研究智能体的严重过度自信问题。

    

    深度研究智能体通过多轮检索和面向决策的生成来解决知识密集型任务。然而，这类智能体存在严重的过度自信问题，导致其表达的置信度对于用户信任和下游弃答决策而言并不可靠。为解决这一问题，我们在深度研究流程的每次检索之后增加了步骤置信度引出环节，并以常用的答案后言语化置信度为基础。有趣的是，我们发现证据置信度——在最后一次检索步骤后引出的置信度——比答案置信度——在答案生成后引出的置信度——能提供更强的不确定性信号，且答案置信度在很大程度上受到证据置信度的塑造。基于这些发现，我们提出了DualStake，一种双路径校准方法，通过施加边界裁剪的、置信度相关的stake奖励，将证据置信度和答案置信度与答案正确性联合对齐，同时抑制对极端置信度的过度优化。实验……

    arXiv:2609.00935v1 Announce Type: cross  Abstract: Deep Research agents tackle knowledge-intensive tasks through multi-round retrieval and decision-oriented generation. However, these agents suffer from severe overconfidence, making their expressed confidence unreliable for user trust and downstream abstention. To address this, we augment the Deep Research pipeline with step confidence elicitation after each retrieval, building on the commonly used post-answer verbalized confidence. Interestingly, we find that Evidence Confidence (E-Conf), elicited after the final retrieval step, provides a stronger uncertainty signal than Answer Confidence (A-Conf), elicited after answer generation, and that A-Conf is largely shaped by E-Conf. Based on these findings, we propose DualStake, a dual-path calibration method that applies margin-clipped, confidence-dependent stake rewards to jointly align E-Conf and A-Conf with answer correctness while limiting extreme confidence optimization. Experiments o
    
[^65]: 上下文接地增益由既有机制介导：对GRPO、SFT和DPO的审计

    Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO

    [https://arxiv.org/abs/2609.00925](https://arxiv.org/abs/2609.00925)

    本文通过从同一检查点系统审计GRPO、SFT和DPO共九种后训练方案，发现语言模型遵循冲突提示证据的接地增益主要源于强化模型中已有的机制（与起始模型相同的因果注意力头集合），而非学习新机制，其中GRPO增益很小、冲突SFT提升适中、DPO在其匹配分布上接近上限。

    

    当提示中的证据与模型记忆中的知识冲突时，语言模型可能会忽略提示中的证据。后训练可以让模型更可靠地遵循这类证据，但这些增益究竟需要新的机制，还是通过强化已有的机制来实现，目前尚不清楚。我们从同一个起始检查点比较了涵盖GRPO、SFT和DPO的九种后训练方案，并将关键比较扩展到不同规模和不同模型家族。我们在训练之前从该起始检查点估计了一个“接地方向”。在测试的五种GRPO变体中，接地增益都很小。对于两种在不同随机种子下可复现的变体，等价性检验表明，即使被奖励的指标有所提升，它们的效果仍低于冲突SFT所带来的增益。冲突SFT适度地改善了接地能力，而DPO在其匹配的分布上使接地能力接近上限。冲突SFT和DPO在很大程度上使用与起始模型相同的因果注意力头集合。减去起始模型的方向会同时抑制两者……

    arXiv:2609.00925v1 Announce Type: cross  Abstract: Language models can ignore prompt evidence when it conflicts with memorized knowledge. Post-training can make models follow such evidence more reliably, but it is unclear whether these gains require new machinery or strengthen machinery already present. We compare nine post-training arms spanning GRPO, SFT, and DPO from one starting checkpoint, with key comparisons extended across scales and families. We estimate a grounding direction from that checkpoint before training. Across five tested GRPO variants, grounding gains are small. For the two variants replicated across seeds, equivalence tests bound their effects below the conflict-SFT gain even as the rewarded metric improves. Conflict-SFT improves grounding moderately, while DPO drives grounding near ceiling on its matched distribution. Conflict-SFT and DPO largely use the same causal attention-head set as the starting model. Subtracting the starting-model direction suppresses both 
    
[^66]: 基于神经网络参数化的三维有限光源反射器直接优化方法

    Direct Optimization of a 3D Finite-Source Reflector via Neural-Network Parameterization

    [https://arxiv.org/abs/2609.00899](https://arxiv.org/abs/2609.00899)

    提出了一种用小型神经网络参数化反射器轮廓、通过可微光线追踪进行端到端训练并结合H^{-1}型谱加权的直接优化方法，能够将有限光源的光转换为预定的远场角度光强分布。

    

    我们提出了一种针对三维自由曲面反射器的直接优化方法，该方法能够将有限光度延展（finite-étendue）光源发出的光转换为预定的远场角度光强分布。反射器轮廓由一个小型神经网络（多层感知机）表示，并通过可微光线追踪目标函数进行端到端训练。此外，我们在心射切面坐标中对发射方向进行参数化，并展示了如何利用该参数化方法确保每条发射光线都与反射器相交。在每次迭代中，该网络被转换为双三次样条表示以提高光线追踪效率，光线与该光滑表面的交点通过阻尼牛顿法求解，梯度则借助隐函数定理计算。追踪得到的输出分布在一个“软”直方图上与目标分布进行比较，并采用H^{-1}型谱加权，以强调光通量的长程输运。

    arXiv:2609.00899v1 Announce Type: cross  Abstract: We present a direct optimization method for three-dimensional freeform reflectors that transform the light of a finite-\'etendue source into a prescribed far-field angular intensity distribution. The reflector profile is represented by a small neural network (a multilayer perceptron), which is trained end-to-end through a differentiable ray-tracing objective. We furthermore parameterize the emission directions in gnomonic coordinates, and show how we use this to ensure that every emitted ray intersects the reflector. At each iteration, the network is converted to a bicubic spline representation for ray-tracing efficiency, and intersections with this smooth surface are solved by a damped Newton solve, with gradients computed via the implicit function theorem. The traced output distribution is compared with the desired target on a 'soft' histogram, under an $H^{-1}$-type spectral weighting that emphasizes long-range transport of flux to 
    
[^67]: 视觉-语言引导的伪标签用于垃圾分拣语义分割的无监督域自适应

    Vision-Language-Guided Pseudo-Labels for Unsupervised Domain Adaptation in Semantic Segmentation for Waste Sorting

    [https://arxiv.org/abs/2609.00898](https://arxiv.org/abs/2609.00898)

    该论文提出了一种利用SAM、EVA-CLIP和BLIP等视觉-语言基础模型生成跨模态伪标签的流水线，无需任何目标域标注即可实现垃圾分拣语义分割的无监督域自适应。

    

    在实际应用场景中（如自动驾驶、工业垃圾分拣），获取语义分割的标注数据成本高昂，且难以大规模实施。我们提出了一种跨模态伪标签生成流水线，能够在没有任何目标域标注的情况下实现无监督域自适应。该流水线建立在两个核心基础模型之上：SAM生成与类别无关的区域候选，EVA-CLIP基于区域-文本相似度分配语义标签，并通过置信度过滤确保只有可靠的伪标签被用于自训练分割模型。作为可选扩展，BLIP为模糊区域提供基于语言的验证，从而在不改变整体流水线的情况下提升伪标签质量。该流水线在两个域偏移场景上进行了评估：合成到真实的自动驾驶，以及（作为主要关注点）从实验室到工厂的工业垃圾分拣，其性能持续优于仅使用源域数据训练的基线方法。

    arXiv:2609.00898v1 Announce Type: cross  Abstract: Obtaining labeled data for semantic segmentation in applied settings (e.g., autonomous driving, industrial waste sorting) is expensive and often infeasible at scale. We present a cross-modal pseudo-labeling pipeline that enables unsupervised domain adaptation without any target-domain annotations. The pipeline is built on two core foundation models: SAM generates class-agnostic region proposals, and EVA-CLIP assigns semantic labels based on region-text similarity, with confidence filtering ensuring that only reliable pseudo-labels are used for self-training a segmentation model. As an optional extension, BLIP provides language-grounded verification for ambiguous regions, thereby improving pseudo-label quality without altering the overall pipeline. Evaluated on two domain shifts, synthetic-to-real autonomous driving and, with a primary focus, lab-to-factory industrial waste sorting, the pipeline consistently improves over source-only ba
    
[^68]: 具有时变转移动力学的泊松-伽马动力系统

    Poisson-Gamma Dynamical Systems with Time-varying Transition Dynamics

    [https://arxiv.org/abs/2609.00896](https://arxiv.org/abs/2609.00896)

    本文提出具有时变转移核的泊松-伽马动力系统（TV-PGDS），通过三种专门设计的狄利克雷马尔可夫链建模转移矩阵随时间的演化，并利用数据增广技术实现完全共轭的高效吉布斯采样，从而更好地捕捉现实计数时间序列中的时变转移动态。

    

    用于处理计数型时间序列的贝叶斯方法因其能够推断可解释的潜在结构并估计不确定性而日益受到关注。在这些贝叶斯模型中，泊松-伽马动力系统（PGDS）已被证明能有效捕捉观测计数序列背后的演化动态。然而，最先进的PGDS在捕捉现实世界计数时间序列中普遍存在的转移动态方面仍存在不足。为缓解这一局限性，本文提出了一种具有时变转移核的泊松-伽马动力系统（TV-PGDS），允许底层转移矩阵随时间演化。论文构建了三种专门设计的狄利克雷马尔可夫链（Dir-Dir、Dir-Gam-Dir、PR-Gam-Dir），以适应这些依赖关系中的异构结构突变。借助狄利克雷-多项-贝塔数据增广技术，一种完全共轭且高效的吉布斯采样器被……（原文摘要到此截断）

    arXiv:2609.00896v1 Announce Type: new  Abstract: Bayesian methodologies for handling count-valued time series have gained prominence due to their ability to infer interpretable latent structures and to estimate uncertainties. Among these Bayesian models, Poisson-Gamma Dynamical Systems (PGDSs) are proven to be effective in capturing the evolving dynamics underlying observed count sequences. However, the state-of-the-art PGDS still falls short in capturing the transition dynamics that are commonly observed in real-world count time series. To mitigate this limitation, a PGDS with time-varying transition kernel (TV-PGDS), is proposed to allow the underlying transition matrices to evolve over time. Three specifically-designed Dirichlet Markov chains (Dir-Dir, Dir-Gam-Dir, PR-Gam-Dir) are constructed to accommodate heterogeneous structural mutations within these dependencies. Leveraging Dirichlet-Multinomial-Beta data augmentation techniques, a fully-conjugate and efficient Gibbs sampler is
    
[^69]: 去噪扩散生成模型暗中计算注意力

    Denoising Diffusion Generative Models Secretly Calculate Attentions

    [https://arxiv.org/abs/2609.00885](https://arxiv.org/abs/2609.00885)

    该论文发现去噪扩散模型本质上暗中使用了与Transformer类似的注意力机制，从而证明注意力是机器学习的普适性原理，并据此提出了基于注意力机制的简化图像生成算法，以减少训练时间和计算开销。

    

    去噪扩散模型是图像生成领域的主导架构，而大多数自然语言生成与建模则主要由采用注意力机制的知名Transformer架构来处理。在本文中，我们证明扩散模型本身也使用了一种与Transformer非常相似的注意力机制。因此，注意力作为一种基于通用训练目标的原则，成为机器学习中的普适性原理。我们还展示了自编码器与基于注意力的模型在基本功能原理上的相似性。这些等价性使我们能够根据实际需求在这些设计之间进行互换。举例来说，我们可以重新表述扩散框架，以缩短冗长的训练过程并降低图像生成的计算开销。基于这种方法，我们提出了一种基于注意力机制的简化图像生成算法。结果表明，基于注意力的实现方式（原文摘要在此处截断）……

    arXiv:2609.00885v1 Announce Type: new  Abstract: Denoising diffusion models are the dominant architecture for image generation, whereas most natural language generation and modeling are primarily handled by well-known transformer architectures employing attention mechanism. Here, we show that diffusion models also inherently use an attention mechanism very similar to that of transformers. Therefore, attention emerges as a universal machine learning principle, based on a general training objective. We also show similarities in basic functional principle of auto-encoders and attention-based models. These equivalences allows us to interchange these designs based on practical requirements. As an example, we can reformulate the diffusion framework to reduce the lengthy training process and computation-intensive image generation. Using this approach, a simplified algorithm is proposed for image generation which is based on attention mechanism. Results show that the attention-based implementa
    
[^70]: 用于宽带CARS相位恢复的iPINN：面向非线性光谱中函数逼近与逆建模问题的框架

    iPINN for Broadband CARS Phase Retrieval: A Framework for Function Approximation and Inverse Modeling Problems in Nonlinear Spectroscopy

    [https://arxiv.org/abs/2609.00883](https://arxiv.org/abs/2609.00883)

    本文提出一种逆物理信息神经网络（iPINN），利用transformer编码器和可微分解析正演模型从原始BCARS光谱中预测洛伦兹峰参数并重建共振极化率，在非共振背景和噪声多变的采集条件下实现了最优的相位恢复精度。

    

    宽带相干反斯托克斯拉曼光谱（BCARS）中的相位恢复是一个病态逆问题。类拉曼信号编码在共振极化率的虚部中，并与在不同采集中会发生变化的非共振背景（NRB）发生相干混合。我们提出了一种逆物理信息神经网络（iPINN），它能够从原始BCARS光谱预测洛伦兹峰参数，并通过可微分的解析正演模型重建共振极化率。一个transformer编码器将光谱特征分配给24个可学习的峰槽，同时多视图一致性损失确保模型对NRB模式、NRB强度和噪声保持不变性。与直接光谱回归方法不同，该方法在不断变化的采集条件下仍能保持精度。在公共基准测试中，iPINN在所有被测试的基线方法中取得了最低误差（MAE为0.016，次优方法为0.046）。在28个零样本测试光谱上……

    arXiv:2609.00883v1 Announce Type: new  Abstract: Phase retrieval in broadband coherent anti-Stokes Raman spectroscopy (BCARS) is an ill-posed inverse problem. The Raman-like signal is encoded in the imaginary part of the resonant susceptibility, which mixes coherently with a non-resonant background (NRB) that varies across acquisitions. We introduce an inverse physics-informed neural network (iPINN) that predicts Lorentzian peak parameters from raw BCARS spectra and reconstructs the resonant susceptibility through a differentiable analytical forward model. A transformer encoder assigns spectral features to 24 learnable peak slots, and a multi-view consistency loss enforces invariance across NRB pattern, NRB strength, and noise. Unlike direct spectral regression approaches, the method retains accuracy under varying acquisition conditions. On a public benchmark, iPINN achieves the lowest error among the tested baselines (MAE 0.016 vs. next-best 0.046). On 28 zero-shot test spectra acquir
    
[^71]: 基于FractalNet的卫星巨型星座轨道边缘智能异构联邦学习：以野火监测为案例研究

    FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study

    [https://arxiv.org/abs/2609.00875](https://arxiv.org/abs/2609.00875)

    该论文提出一种基于FractalNet的异构联邦学习方法，通过分布式路径调度器根据卫星SWAP-C约束与预测星间通信机会动态分配模型深度，并结合周期性更新汇聚和三层智能体控制平面，实现了卫星巨型星座中适应异构硬件条件的轨道边缘智能。

    

    卫星巨型星座正成为大规模的感知、通信与计算基础设施，然而其学习架构在很大程度上仍沿袭自地面联邦学习和以地面为中心的任务运营模式——这对于在尺寸、重量、功耗与成本（SWAP-C）、抗辐射能力、链路可用性以及传播时延方面相差数个数量级的卫星而言并不适用。我们提出了一种基于FractalNet架构的异构联邦学习方法，用于轨道边缘智能。我们形式化了受通信窗口约束、深度异构的联邦优化问题，并引入了一种分布式路径调度器，将模型深度分配设定为SWAP-C约束、预测的星间通信机会以及训练统计信息的函数。为降低消息开销与能耗，每一层级采用周期性汇聚更新的方式，而非在每次通信机会时都进行汇聚，并由一个三层智能体控制平面进行管理……

    arXiv:2609.00875v1 Announce Type: new  Abstract: Satellite mega-constellations are emerging as large-scale sensing, communication, and computation fabrics, yet their learning architectures remain largely inherited from terrestrial federated learning and ground-centric mission operations--- ill-suited to satellites that differ by orders of magnitude in Size, Weight, Power, and Cost (SWAP-C), radiation tolerance, link availability, and propagation delay. We propose a heterogeneous federated learning method based on the FractalNet architecture for orbital edge intelligence. We formalize contact-window-constrained, depth-heterogeneous federated optimization and introduce a distributed path scheduler that assigns model depth as a function of SWAP-C constraints, predicted inter-satellite contacts, and training statistics. To reduce message overhead and energy consumption, each tier pools updates periodically rather than at every contact opportunity, and a three-tier agentic control plane gov
    
[^72]: 库仑型多电子波函数的尖锐混合谱Barron正则性

    Sharp Mixed Spectral Barron Regularity of Coulombic Many-Electron Wave Functions

    [https://arxiv.org/abs/2609.00872](https://arxiv.org/abs/2609.00872)

    该论文为分子库仑哈密顿量的本征函数建立了尖锐的混合谱Barron正则性，给出了各向同性阶数 $s$ 与坐标阶数 $\alpha,\beta$ 的最优显式容许区域，从而刻画了传统各向同性Barron尺度无法检测的正则性。

    

    我们为分子库仑哈密顿量的本征函数建立了尖锐的混合谱Barron正则性。该混合范数是一种带有单个各向同性权重与坐标乘积权重的傅里叶 $L^1$ 范数，因此能够检测到各向同性Barron尺度所无法察觉的正则性。对于波函数在其上为反对称的非空电子指标集 $I$，我们推导出了各向同性阶数 $s$ 与坐标阶数 $\alpha,\beta$ 的一个显式容许区域。作为对固定核库仑哈密顿量类的一致性陈述，该区域是最优的。对于具有两个被占据自旋块的固定自旋分量，该条件化为 $s+\alpha+\beta<1$；在完全自旋极化的类别中则化为 $s+\alpha<1$。特别地，若 $\mathcal I_\sigma$ 表示由 $\sigma$ 确定的被占据同自旋块族，则每个固定自旋的空间分量 $\psi_\sigma$ 对每个 $0\leq\alpha<1$ 都满足相应的加权求和不等式（摘要在此处截断）。

    arXiv:2609.00872v1 Announce Type: cross  Abstract: We establish sharp mixed spectral Barron regularity for eigenfunctions of molecular Coulomb Hamiltonians. The mixed norm is a Fourier $L^1$ norm with one isotropic weight and coordinate-product weights, and therefore detects regularity invisible to the isotropic Barron scale. For a nonempty set $I$ of electron indices on which the wave function is antisymmetric, we derive an explicit admissible region for the isotropic order $s$ and the coordinate orders $\alpha,\beta$. This region is optimal as a uniform statement over the class of clamped-nuclei Coulomb Hamiltonians. For fixed-spin components with two occupied spin blocks, it reduces to $s+\alpha+\beta<1$; in the fully spin-polarized class it reduces to $s+\alpha<1$. In particular, if $\mathcal I_\sigma$ denotes the family of occupied same-spin blocks determined by $\sigma$, then every fixed-spin spatial component $\psi_\sigma$ satisfies, for every $0\leq\alpha<1$, \[ \left(\sum_{I\i
    
[^73]: 视觉不敏感性差距：诊断视觉-语言模型何时未能利用视觉证据

    The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence

    [https://arxiv.org/abs/2609.00868](https://arxiv.org/abs/2609.00868)

    该论文发现“视觉不敏感性差距”现象——在40%–97%的多模态基准样本上，模糊与问题相关的关键视觉区域几乎不改变VLM的输出，并证明这种不敏感性是样本层面的属性（跨模型VSI排名显著相关），即使各模型的视觉编码器本身实际上能够检测到这些扰动。

    

    视觉-语言模型（VLM）通常通过在多模态基准上的总体准确率进行评估，这种做法隐含地假设模型确实使用了其视觉输入。我们证明这一假设在六个VLM和三个感知基准的40%–97%样本上并不成立：将问题相关的视觉区域模糊化后，模型的下一个词元分布几乎不变。我们将这一现象命名为“视觉不敏感性差距”，并用逐样本的视觉敏感性指数（VSI）对其进行量化。该差距是样本本身的属性，而非模型的属性：VSI排名在各模型之间呈现相关性（总体平均Spearman rho=+0.40，置换检验p<10^-3），因此即使这些VLM之间除了对比预训练的视觉编码器外不共享任何架构细节，相同的样本仍会被它们共同标记为“不敏感”。其机制是具体的：在不敏感样本上，对每个模型自身的视觉编码器进行线性探针可以以0.72–0.79的准确率区分受扰动图像与原始图像，然而模型的argmax词元却几乎没有变化

    arXiv:2609.00868v1 Announce Type: cross  Abstract: Vision-language models are evaluated by aggregate accuracy on multimodal benchmarks, a practice that implicitly assumes the model uses its visual input. We show this assumption fails on 40%--97% of samples across six VLMs and three perceptual benchmarks: blurring the question-relevant visual region leaves the next-token distribution nearly unchanged. We name this phenomenon the Visual Insensitivity Gap and quantify it with a per-sample Visual Sensitivity Index (VSI). The gap is a property of samples, not of models: VSI ranks correlate across models (grand-mean Spearman rho=+0.40, permutation p<10^-3), so the same samples are flagged insensitive by VLMs sharing no architectural detail beyond a contrastively pretrained vision tower. The mechanism is concrete: on the insensitive samples, a linear probe on each model's own vision tower distinguishes perturbed from clean images at 0.72--0.79 accuracy, yet the model's argmax token changes on
    
[^74]: MemoryWalker：停止在智能体从未见过的上下文上训练智能体

    MemoryWalker: Stop Training Agents on Contexts They Never Saw

    [https://arxiv.org/abs/2609.00865](https://arxiv.org/abs/2609.00865)

    该论文针对上下文压缩导致智能体训练时有效历史呈树状分支的问题，提出了两种梯度等价的精确修正方法（LogitTree 与 4D 注意力掩码）以及一种仅需单次反向传播的自蒸馏方法 SDCC，从而消除压缩训练与推理之间的条件化不一致。

    

    Claude Code 和 Qwen-Agent 等生产级智能体框架在执行过程中会压缩上下文，但在压缩条件下进行训练会产生一个条件化问题：每次上下文剔除都会使有效历史产生分支，因此学习对象是一棵树而非一个序列。现有的线性化方法要么保留最右路径，导致“时间旅行”式信息泄露；要么重放深度优先遍历，导致训练与推理不匹配。我们提出两种精确且梯度等价的修正方法：LogitTree（一种分段 K 次前向遍历）和打包式 4D 注意力掩码。LogitTree 需要 K+1 次反向传播；4D 掩码则需要自定义内核和白盒化的剔除记录。我们还提出了 SDCC（面向条件化一致性的自蒸馏），这是一种仅需单次反向传播的变分松弛方法。在每次剔除时，它在重建的剔除前前缀上最小化压缩学生模型与停止梯度的教师模型之间的前向 KL 散度，并通过每个分支点的残差 KL 项……

    arXiv:2609.00865v1 Announce Type: cross  Abstract: Production agent harnesses such as Claude Code and Qwen-Agent compress context during rollout, but training under compression creates a conditioning problem: every eviction branches the effective history, so the learning object is a tree rather than a sequence. Existing linearizations either retain the rightmost path, causing time-travel leakage, or replay a depth-first traversal, causing train-inference mismatch. We introduce two exact, gradient-equivalent corrections: LogitTree, a segmented K-forward traversal, and a packed 4D attention mask. LogitTree requires K+1 backward passes; the 4D mask requires a custom kernel and white-box eviction records. We also propose SDCC (Self-Distillation for Conditioning Consistency), a single-backward-pass variational relaxation. At each eviction, it minimizes forward KL between the compressed student and a stop-gradient teacher on the reconstructed pre-eviction prefix. A residual per-junction KL o
    
[^75]: 面向机器学习逆向设计问题的条件流匹配方法

    Conditional Flow Matching for ML-Based Inverse Design Problems

    [https://arxiv.org/abs/2609.00863](https://arxiv.org/abs/2609.00863)

    该论文将条件流匹配（CFM）引入 EngiOpt 工程逆向设计框架，并在 EngiBench 的结构与热传导基准上，以累积最优性差距和最终最优性差距为指标，与条件扩散模型和条件GAN系统比较其作为梯度优化暖启动方案的性能。

    

    工程逆向设计常常受到两大限制：受偏微分方程（PDE）约束的优化问题其迭代求解器计算成本高昂，且求解过程对初始化十分敏感。深度生成模型可以在推理阶段无需重新运行模拟器即可生成候选设计方案。生成对抗网络（GAN）可以通过一次前向传播完成采样，而扩散模型则需要迭代式的逆时间积分求解。在这项工作中，我们将条件流匹配（CFM）引入 EngiOpt 框架，并在来自 EngiBench 的结构（beams2d）和热传导（heatconduction2d）基准测试上，采用相同的下游优化协议，将其与条件扩散模型和条件生成对抗网络（cGAN）进行对比。我们采用累积最优性差距（COG）和最终最优性差距（FOG）作为主要评估指标，用以衡量生成设计作为基于梯度的精细化优化暖启动方案的效果。（摘要原文在此处截断）

    arXiv:2609.00863v1 Announce Type: new  Abstract: Engineering inverse design is often limited by the high computational cost of iterative solvers for optimization problems constrained by partial differential equations (PDEs) and by their sensitivity to initialization. Deep generative models can produce candidate designs without rerunning the simulator at inference time. Generative adversarial networks (GANs) sample in one forward pass, whereas diffusion models require iterative reverse-time integration.   In this work, we add conditional flow matching (CFM) to EngiOpt and compare it with a conditional diffusion model and a conditional generative adversarial network (cGAN) on structural (beams2d) and thermal (heatconduction2d) benchmarks from EngiBench using the same downstream optimization protocol. We use cumulative optimality gap (COG) and final optimality gap (FOG) as the primary metrics for evaluating the generated designs as warm starts for gradient-based refinement. On the evaluat
    
[^76]: 故障定位能否胜过重新尝试？——一项关于测试引导代码修复的安慰剂对照研究

    Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair

    [https://arxiv.org/abs/2609.00854](https://arxiv.org/abs/2609.00854)

    该安慰剂对照研究发现，故障定位在实际场景中很少可用（仅约9%的失败候选可定位），且即便可定位，基于频谱定位的片段填充修复也显著劣于盲目的整体重采样。

    

    故障定位可以将代码模型的修复聚焦于失败测试所涉及的语句，但针对性的编辑可能仅仅因为改动较小而成功，而第二次模型调用也可能根本未利用失败信息就取得成功。我们通过对同一失败候选程序施加三种处理来区分这些解释：盲目的整方案重采样、基于频谱的定位后对可疑片段进行填充，以及在互不重叠的随机代码片段上进行等长填充（安慰剂对照）。在三个冻结的26-32B模型、三个基准数据集和488个失败候选上，外加一个单独声明的来自第三方家族的24B第四个模型，得到三个结果。首先，故障定位很少可用：只有9.0%的失败候选存在暴露失败的可公开测试及可用的频谱。其次，在来自强测试套件的177个可定位候选中，在匹配的尝试次数下，定位填充决定性地输给了盲重采样（3:40，p = 3.0 × 10^-9），这与我们的（摘要在此处截断）

    arXiv:2609.00854v1 Announce Type: cross  Abstract: Fault localization can focus a code model's repair on the statements a failing test implicates, but a targeted edit may succeed merely because it is small, and a second model call may succeed without using the failure at all. We separate these explanations with three arms applied to the same failed candidate: blind whole-solution resampling, spectrum-based localization followed by suspect-span infilling, and same-length infilling at a disjoint random code span. Across three frozen 26-32B models, three benchmarks and 488 failing candidates, plus a separately declared 24B fourth model from a third family, three results follow. First, localization is rarely available: only 9.0% of failing candidates expose a failing public test with a usable spectrum. Second, among the 177 candidates localizable from a strong suite, localized infilling loses decisively to blind resampling at a matched attempt count (3:40, p = 3.0 x 10^-9), opposite to our
    
[^77]: 地球系统建模中机器学习/人工智能应用的能耗与碳排放影响评估清单

    A Checklist to assess the energy and carbon impacts of ML/AI applications in Earth System Modeling

    [https://arxiv.org/abs/2609.00847](https://arxiv.org/abs/2609.00847)

    该论文将分散的机器学习/人工智能可持续发展讨论提炼为一份按模型开发流程各阶段组织的实用清单，帮助地球系统科学从业者评估并降低其应用的能耗与碳足迹，并配套提供了估算能耗和碳排放的指标。

    

    随着机器学习和人工智能逐渐渗透到气候、天气和地球系统建模的几乎每一个方面，我们有必要停下来思考自己的设计决策对科学本身以及我们所消耗的计算资源意味着什么。越来越多的文献探讨了机器学习/人工智能的伦理与可持续发展问题，然而将这些原则转化为日常研究实践仍然是一个挑战，因为大多数最佳实践分散在多项研究和评论之中。在此，我们将这些讨论提炼成一份实用的清单，供机器学习/人工智能和地球系统科学的从业者用来评估并减少其自身应用的环境足迹，该清单围绕模型开发流程的各个连续阶段进行组织。我们还在此基础上补充了从文献中选取的一系列指标，用于估算项目的能耗与碳足迹。

    arXiv:2609.00847v1 Announce Type: cross  Abstract: As machine learning and artificial intelligence find their way into nearly every aspect of climate, weather, and Earth system modeling, it is worth pausing to consider what our design decisions imply for the science and for the computational resources we consume. A growing body of literature addresses the ethical and sustainable development of ML/AI, yet translating these principles into day-to-day research practice remains a challenge as most of best practices are dispersed across multiple studies and commentaries. Here, we distill these discussions into a practical checklist that ML/AI and Earth system science practitioners can use to assess and reduce the environmental footprint of their own applications, organised around the successive stages of the model development pipeline. We complement the checklist with a selection of metrics drawn from the literature for estimating the energy consumption and carbon footprint of a project. Fo
    
[^78]: 基于事实效用估计的搜索智能体密集过程监督

    Dense Process Supervision for Search Agents via Fact Utility Estimation

    [https://arxiv.org/abs/2609.00833](https://arxiv.org/abs/2609.00833)

    本文提出一种基于事实效用估计的密集过程监督方法，通过将推理过程建模为离散证据事实的累积，并利用贝叶斯估计将事实效用转化为步骤级奖励，有效解决了搜索智能体强化学习中的信用分配难题。

    

    面向搜索智能体的强化学习（RL）通常依赖于结果奖励。然而，由于中间步骤的价值不明确，这种方法往往难以实现有效的信用分配，很难将中间步骤的贡献从最终结果中分离出来。在本文中，我们提出了一种基于事实效用估计的密集过程监督方法，该方法将推理过程建模为离散证据事实的累积。我们首先从原始观测中提取结构化事实，并将其组织成显式的事实存储库。为支持信用分配，我们随后对语义等价的事实进行聚类，并利用基于组内多次采样的贝叶斯估计来推断每个事实簇的后验效用。最后，我们将估计出的事实效用转换为密集的步骤级奖励，以指导强化学习训练。在七个单跳和多跳问答基准上的实验表明，我们的方法持续优于现有基线方法。

    arXiv:2609.00833v1 Announce Type: new  Abstract: Reinforcement learning (RL) for search agents typically relies on outcome rewards. However, it often fails to achieve effective credit assignment, due to the unclear value of intermediate steps. It is hard to separate their contributions from the final result. In this paper, we propose a dense process supervision method based on fact utility estimation, which models the reasoning process as the accumulation of discrete evidence facts. We first extract structured facts from raw observations and organize them into an explicit fact store. To support credit assignment, we then cluster semantically equivalent facts and infer the posterior utility of each fact cluster using Bayesian estimation over group rollouts. Finally, we convert the estimated fact utilities into dense step-level rewards to guide RL training. Experiments on seven single-hop and multi-hop QA benchmarks show that our method consistently outperforms existing baselines. Ablati
    
[^79]: HarnessEvolve：从参考轨迹中学习以实现可靠的智能体自我进化

    HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution

    [https://arxiv.org/abs/2609.00829](https://arxiv.org/abs/2609.00829)

    HarnessEvolve 提出了一种从参考轨迹中学习的智能体自我进化框架，通过将执行、评估、优化和门控解耦为独立模块，克服了信用分配失败、捷径学习和灾难性遗忘三大难题，实现了可靠且可泛化的智能体自我进化。

    

    自我进化的智能体通过基于环境反馈优化其“线束”（harness）——包括提示词、技能、工具和执行逻辑——来向自主性迈进。然而，这一范式面临三大挑战：信用分配失败（credit assignment failure），即仅凭最终的成功/失败反馈难以判断是哪一步导致了错误；捷径学习（shortcut learning），即智能体记忆特定任务的模式而非习得可泛化的能力；以及灾难性遗忘（catastrophic forgetting），即缺乏防护的更新会降低先前习得的能力。本文提出了 HarnessEvolve，一个通过从参考轨迹中学习来实现可靠智能体自我进化的自进化框架。HarnessEvolve 将执行智能体与进化流程解耦，把执行、评估、优化和门控分别分配给独立的智能体模块，从而实现可泛化且稳定的线束改进。

    arXiv:2609.00829v1 Announce Type: cross  Abstract: Self-evolving agents advance toward autonomy by optimizing their harness---prompts, skills, tools, and execution logic---based on environmental feedback. This paradigm, however, is hampered by three challenges: \textit{credit assignment failure}, where terminal success/failure feedback makes it ambiguous which step caused the error; \textit{shortcut learning}, where agents memorize task-specific patterns rather than acquire generalizable capabilities; and \textit{catastrophic forgetting}, where unguarded updates degrade previously acquired competence. In this paper, we introduce HarnessEvolve, a self-evolving framework that learns from reference trajectories to achieve reliable agent self-evolution. HarnessEvolve decouples the execution agent from the evolutionary pipeline, assigning execution, evaluation, optimization, and gating to independent agent modules, enabling generalizable and stable harness improvements. Specifically, Harnes
    
[^80]: 训练神经网络中的子空间Levenberg-Marquardt算法

    Subspace Levenberg Marquardt Algorithms in Training Neural Networks

    [https://arxiv.org/abs/2609.00789](https://arxiv.org/abs/2609.00789)

    本文评估了子空间Levenberg-Marquardt算法（如KSLM和HSLM）在神经网络回归与分类任务中的性能，并将其与经典LM方法及SGD、Adam等一阶优化算法进行了比较，以展示子空间方法在降低二阶方法计算与内存开销方面的有效性。

    

    Levenberg-Marquardt (LM) 算法是一种著名的二阶优化方法，在训练中小规模神经网络时具有收敛快速和鲁棒性强的特点。然而，随着神经网络参数数量的增长，其计算和内存成本会显著增加。为了解决这一局限性，研究者提出了子空间方法，例如Krylov子空间LM (KSLM) 和混合子空间LM (HSLM)，从而使二阶算法更加高效。在这项工作中，我们评估了子空间Levenberg-Marquardt算法在神经网络回归和分类任务中的表现，并将子空间LM变体与经典LM方法以及其他流行的一阶算法（如随机梯度下降SGD和Adam）进行了性能比较。

    arXiv:2609.00789v1 Announce Type: new  Abstract: The Levenberg-Marquardt (LM) algorithm is a well-known second-order method for rapid convergence and strong robustness when training small- to medium-sized neural networks (NNs). However, its computational and memory costs increase significantly as the number of parameters in an NN grows. To address this limitation, subspace methods have been proposed, such as the Krylov subspace LM (KSLM) and the hybrid subspace LM (HSLM), making second-order algorithms more efficient. In this work, we evaluate the subspace Levenberg-Marquardt algorithms for regression and classification tasks in neural networks. We compare the performance of subspace LM variants with the classical LM method, as well as other popular first-order algorithms, such as stochastic gradient descent (SGD) and Adam.
    
[^81]: 威布尔混合模型中含信息性缺失标签的半监督分类

    Semi-Supervised Classification with Informative Missing Labels in Weibull Mixture Models

    [https://arxiv.org/abs/2609.00774](https://arxiv.org/abs/2609.00774)

    该论文提出在两分量威布尔混合模型的半监督分类中，将标签缺失概率建模为分类不确定性的函数，从而证明缺失标签指示变量本身携带分类器信息，并据此刻画了贝叶斯决策边界的结构并推导了相应的Fisher信息量。

    

    我们考虑从来自两分量威布尔混合模型的部分已分类样本中进行半监督分类。所有数据的特征均可观测，而部分类别标签存在缺失。我们将标签缺失的概率建模为分类不确定性的函数，从而得到一种依赖于特征的随机缺失（MAR）机制，该机制与威布尔混合分类器共享参数。因此，除观测到的特征和已有的类别标签之外，缺失标签的指示变量本身也能提供关于分类器的信息。在威布尔形状参数相同的情形下，贝叶斯决策规则至多有一个正的决策边界，且当规则非常数时该边界唯一；在形状参数不同的情形下，则可能出现两个决策边界。我们刻画了这些决策区域，推导出在对缺失机制中的冗余参数进行调整之后的分类器Fisher信息量，并得到了期望……的决策边界展开式（原文摘要在此处截断）。

    arXiv:2609.00774v1 Announce Type: cross  Abstract: We consider semi-supervised classification from a partially classified sample arising from a two-component Weibull mixture. The feature is observed for all data, whereas some class labels are missing. The probability of a missing label is modelled as a function of classification uncertainty, giving a feature-dependent missing-at-random (MAR) mechanism that shares parameters with the Weibull-mixture classifier. The missing-label indicators can therefore provide information about the classifier in addition to the observed features and available class labels. Under a common Weibull shape, a Bayes' rule has at most one positive decision boundary, which is unique when the rule is nonconstant; under unequal shapes, it can have two. We characterise these decision regions, derive the Fisher information for the classifier after adjustment for nuisance parameters in the missingness model, and obtain a decision-boundary expansion of the expected 
    
[^82]: 你在想我所想吗？：检验神经网络架构中的概念分离

    Are You Thinking What I am Thinking? : Examining Conceptual Separation in Neural Architectures

    [https://arxiv.org/abs/2609.00764](https://arxiv.org/abs/2609.00764)

    本研究通过对CNN和LLM内部激活的几何与分布分析，揭示神经网络中存在“概念分离”现象——同一概念形成连贯表示、相关概念在表示空间中彼此更近，但这种连贯性在未见概念、域偏移和模糊主题下会减弱或坍塌。

    

    神经网络越来越多地被用于识别明确定义的概念以及模糊概念，但输出层面的指标几乎无法揭示这些概念在内部是如何被表示的。我们的研究探讨这些网络是否表现出“概念分离”特性：同一概念的样本是否形成连贯一致的表示，以及相关概念是否在表示空间中彼此更接近。我们通过对卷积神经网络（CNN）和大型语言模型（LLM）内部激活进行几何和分布分析，考察了这种概念组织方式。在CNN中，熟悉的ImageNet概念形成了连贯且语义有序的表示，而对于未见过的概念，这种连贯性会减弱，并且在类内域偏移的情况下会受到损害。在LLM中，明显不同的领域保持良好的分离，相关子领域彼此靠近，而模糊主题之间的区分在均值和协方差层面都会坍塌。

    arXiv:2609.00764v1 Announce Type: cross  Abstract: Neural networks are increasingly employed to identify both well-defined and ambiguous concepts, yet output-level metrics reveal little about how those concepts are represented internally. Our study asks if these networks exhibit \textit{conceptual separation}: if examples of the same concept form coherent representations, and whether related concepts lie closer together in the representation space. We examine this conceptual organisation in Convolutional Neural Networks (CNNs) and Large Language Models (LLMs) through geometric and distributional analysis of their internal activations. In CNNs, familiar ImageNet concepts form coherent and semantically ordered representations, while this coherence weakens for unseen concepts and suffers within-class domain shift. In LLMs, clearly distinct domains remain well separated, related subdomains move closer together, and the distinction between ambiguous topics collapses at both the mean and cov
    
[^83]: 冻结核心需要任务信号：面向低资源大语言模型适配的Fisher白化交叉协方差方法

    Frozen Cores Need Task Signal: Fisher-Whitened Cross-Covariance for Low-Resource LLM Adaptation

    [https://arxiv.org/abs/2609.00762](https://arxiv.org/abs/2609.00762)

    提出FCCA方法，通过对角Fisher矩白化输入-误差交叉协方差来为冻结核心微调构建任务感知的低秩核心基，在相同可训练参数预算下的11个任务、4种模型设置上超越了八种现有基构造方法。

    

    参数高效微调通常被框定为“需要更新多少参数”的问题。然而，在严格受限的可训练状态预算下，这些系数“作用在哪里”同样关键。我们通过冻结核心适配来研究这一选择：先通过一次校准过程为每个权重矩阵固定左右基，随后微调仅优化一个 $r\times r$ 的核心。这消除了可训练因子修复不良初始子空间的能力，使子空间质量可以直接观测。我们提出FCCA方法：估计带符号的输入-误差交叉协方差，用对角Fisher矩对其进行白化，在所得的局部度量中截断，将选定方向映射回原空间，并应用瘦QR分解以获得稳定的核心坐标。在匹配的 $r^2$ 可训练参数预算下，我们在11个任务、4种模型设置和3个随机种子上比较了八种基构造方法。在Qwen2.5-3B上，FCCA达到83.0的宏平均分数，比次优方法高出2.3个百分点。

    arXiv:2609.00762v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning is usually framed as a question of how many parameters to update. Under a severe trainable-state budget, however, where those coefficients act is equally consequential. We study this choice through frozen-core adaptation: a calibration pass fixes left and right bases for each weight matrix, and fine-tuning optimizes only an $r\times r$ core. This removes the ability of trainable factors to repair a poor initial span and makes subspace quality directly observable. We introduce FCCA, which estimates the signed input--error cross-covariance, whitens it with diagonal Fisher moments, truncates it in the resulting local metric, maps the selected directions back, and applies thin QR to obtain stable core coordinates. Under a matched $r^2$ budget, we compare eight basis constructors on 11 tasks, four model settings, and three seeds. On Qwen2.5-3B, FCCA reaches an 83.0 macro-average, 2.3 points above the next-best 
    
[^84]: 语言模型如何在上下文与记忆之间进行选择？

    How Do Language Models Choose Between Context and Memory?

    [https://arxiv.org/abs/2609.00753](https://arxiv.org/abs/2609.00753)

    本文通过反事实实验证明了从一致性提示中估计的“权威方向”在语言模型内部因果地决定了模型在上下文信息与参数记忆之间的选择——沿这些方向交换激活坐标可重现30-68%的来源选择偏移。

    

    当上下文信息与存储在模型参数中的知识发生冲突时，可以利用激活方向来解码并引导模型遵循哪种信息来源。然而，沿着某个方向进行引导并不能确立因果关系：即未经修改的模型是否会自然地使用该方向，或者该方向是否可以在不同任务间复用。我们通过在无歧义设置下的反事实实验来检验这些区别。首先，我们从一致性提示中估计“权威方向”，在这类提示中，上下文和参数化知识支持相同的答案。然后，我们在匹配的提示之间交换这些方向上自然出现的坐标，这些匹配提示分别引导模型优先考虑所提供的上下文或其参数化知识。在Qwen、Llama和OLMo模型上，这种干预重现了30-68%由权威性引起的来源选择偏移，而匹配的对照组几乎没有重现任何偏移。为了测试跨任务……

    arXiv:2609.00753v1 Announce Type: cross  Abstract: When contextual information conflicts with the knowledge stored in model parameters, activation directions can be used to decode and steer which source the model follows. However, steering along a direction does not establish causality: whether the unedited model would naturally use that direction or whether the direction is reusable across tasks. We test these distinctions through counterfactual experiments in unambiguous settings. First, we estimate authority directions from agreement prompts, in which the context and parametric knowledge support the same answer. We then interchange naturally occurring coordinates along these directions between matched prompts that direct the model to prioritize either the supplied context or its parametric knowledge. Across Qwen, Llama, and OLMo models, this intervention reproduces 30-68% of the authority-induced shift in source choice, whereas matched controls reproduce almost none. To test cross-t
    
[^85]: 视觉-语言适配中的文本能力损失：一种基于注意力汇的诊断方法

    Text Capability Loss in Vision-Language Adaptation: An Attention-Sink Diagnosis

    [https://arxiv.org/abs/2609.00746](https://arxiv.org/abs/2609.00746)

    该论文发现将大语言模型微调为视觉-语言模型时文本能力的损失源于注意力汇位置被扰动，并提出Sink Strength指标，仅需在单GPU上计算几秒钟即可预测适配后的文本能力退化程度。

    

    将预训练大语言模型（LLM）微调为视觉-语言模型（VLM）可能会削弱骨干模型的文本能力，且这种损害主要集中在需要严格遵循输出规则的任务上，例如指令遵循、以严格解析最终答案进行评分的思维链推理，以及类似的采用严格评分器的评估。我们将这一性能差距追溯至注意力汇损坏：视觉-语言微调会扰动早期的汇位置，而该位置锚定了大部分的注意力概率；基础大语言模型对其汇的保持越好，受影响的能力在适配后存留的就越多。基于这一观点，我们提出了Sink Strength（汇强度），这是一个在基础大语言模型上计算的单一标量，仅需在单个GPU上几秒钟即可得出，能够在没有任何视觉-语言训练的情况下预测适配后的性能退化。在六组VLM-LLM模型对以及多个格式敏感任务上，该指标均能一致地追踪相对退化程度。作为这一诊断方法的补充，我们发现预训练后的QK-（摘要内容在此处似乎被截断）

    arXiv:2609.00746v1 Announce Type: new  Abstract: Fine-tuning a pretrained LLM into a vision-language model (VLM) can erode the backbone's text capability, with the damage concentrated on tasks that require following exact output rules, such as instruction following, chain-of-thought reasoning graded on a strictly parsed final answer, and similar evaluations with strict graders. We trace this gap to attention-sink corruption: VL fine-tuning perturbs the early sink position that anchors a large fraction of attention probability, and how well the base LLM preserves its sink tracks how much of the affected capability survives adaptation. Building on this view, we introduce Sink Strength, a single scalar computed on the base LLM in a few seconds on a single GPU that predicts post-VL degradation without any VL training. It consistently tracks relative degradation across the six VLM-LLM pairs and multiple format-sensitive tasks. Complementing this diagnostic, we find that post-pretraining QK-
    
[^86]: 在线自加权微调

    Online Self-Weighted Fine-Tuning

    [https://arxiv.org/abs/2609.00734](https://arxiv.org/abs/2609.00734)

    该论文提出在线自加权微调（OSW-FT），通过少量仅推理的采样轨迹估计模型当前成功率，在线调整每个查询的SFT损失权重，在保持专家轨迹优化方向的同时自适应更新幅度，兼顾了SFT的稳定性与RL的自适应性。

    

    标准监督微调（SFT）对每个专家演示样本分配相同的显式损失权重，而不考虑模型在训练查询上不断变化的能力水平。基于强化学习（RL）的方法通过模型生成的采样轨迹来调整更新强度，但通常需要大幅增加采样量，并且在困难任务上可能不稳定。我们提出了在线自加权微调（OSW-FT），这是一种简单的方法，通过在线的、轨迹级别的加权来增强SFT。对于每个查询，OSW-FT使用少量仅用于推理的采样轨迹来估计模型当前的成功率，并据此重新调整标准SFT损失的权重。优化方向仍然以专家轨迹为锚点，而更新幅度则在线自适应调整。对于可二元验证的推理任务，受方差缩减原理的启发，我们在梯度层面将这种加权与SFT和RL联系起来。所得的估计量对于……是无偏的（摘要在此处截断）。

    arXiv:2609.00734v1 Announce Type: new  Abstract: Standard supervised fine-tuning (SFT) assigns the same explicit loss weight to every expert demonstration, regardless of the model's changing competence over training queries. Reinforcement learning (RL) based methods adapt update strength using model-generated rollouts, but often require substantially more sampling and can be unstable on hard tasks. We propose \textbf{Online Self-Weighted Fine-Tuning (OSW-FT)}, a simple method that augments SFT with online, trajectory-level weighting. For each query, OSW-FT estimates the model's current success rate using a small number of inference-only rollouts and rescales the standard SFT loss accordingly. The optimization direction remains anchored to the expert trajectory, while the update magnitude adapts online. For binary-verifiable reasoning, we connect this weighting to SFT and RL at the gradient level, inspired by variance-reduction principles. The resulting estimator is unbiased for the exa
    
[^87]: 智能体化实证资产定价：方法论基础

    Agentic Empirical Asset Pricing: Methodological Foundations

    [https://arxiv.org/abs/2609.00731](https://arxiv.org/abs/2609.00731)

    本文提出了智能体化实证资产定价（AEAP）这一新范式，为自主因子发现系统提供了参考架构、严格的因子评估标准与样本外回测方法，并通过对SEADS与五个基线系统的评估证明此类系统必须从多个维度同时加以评价。

    

    大语言模型智能体的最新进展为资产定价开辟了一种新范式，我们称之为智能体化实证资产定价：即能够自主执行科学发现过程本身的系统。我们定义了AEAP并确定了其核心构建模块。现有的评估实践仅对输出结果（因子或交易）进行回测，而非对产生这些结果的自主发现系统本身进行评估。我们聚焦于因子发现问题，贡献了一个参考架构、一套针对所发现因子的严格评估标准，以及一种对发现系统进行样本外回测的方法。作为该架构的具体实例，我们使用该标准在两个美股面板数据上将SEADS与五个重新实现的基线系统进行对比评估：结果显示没有任何单一指标能够一致地对各系统进行排序，这表明需要在多个维度上同时进行评估。随后，一项独立的滚动重复执行实验提出了一个互补性问题：发现过程本身（而非某个静态输出）是否……

    arXiv:2609.00731v1 Announce Type: new  Abstract: Recent advances in LLM agents enable a new paradigm for asset pricing, which we call Agentic Empirical Asset Pricing (AEAP): systems that autonomously conduct the scientific discovery process itself. We define AEAP and identify its core building blocks. Existing evaluation practices backtest only the outputs (factors or trades), not the autonomous discovery system that produced them. We focus on factor discovery, contributing a reference architecture, a rigorous evaluation standard for discovered factors, and a method for out-of-sample backtesting the discovery system. As a concrete instance of that architecture, we evaluate SEADS against five re-implemented baselines on two US equity panels using this standard: no single metric ranks the systems consistently, motivating evaluation on multiple axes at once. A separate rolling re-execution then asks the complementary question of whether the discovery process itself, not one static output,
    
[^88]: SOVER：通过LLM辅助的SMT验证对优化问题重构进行形式化认证

    SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification

    [https://arxiv.org/abs/2609.00728](https://arxiv.org/abs/2609.00728)

    SOVER框架将语义映射与形式化验证分离，利用Z3和dReal等SMT求解器对LLM生成的优化问题重构进行形式化认证，并在NLEquiv-150基准上实现了149/150的正确分类。

    

    大语言模型（LLMs）在跨建模语言翻译和重构复杂数学优化问题方面展现出了巨大的潜力。然而，仅通过经验性的求解器执行来验证此类转换是不可靠的，因为求解器的结果可能受到局部极小值、结构性超时、数值伪影以及不同表述之间细微语义差异的影响。我们提出了SOVER，这是一个LLM辅助的SMT框架，它将语义映射与形式化认证分离：Z3用于检查混合整数线性表述的域交叉可行性和全局目标序保持性，而dReal则为连续非线性表述提供容差感知的可行性/范围检查和ε-argmin检查。我们还引入了NLEquiv-150，这是一个公开的基准数据集，包含100对等价和50对故意构造的困难非等价非线性重构对。利用LLM提取的映射，SOVER成功分类了149/150对。

    arXiv:2609.00728v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown remarkable promise in translating and reformulating complex mathematical optimization problems across modeling languages. However, validating such transformations through empirical solver executions alone is unreliable, as solver outcomes may be affected by local minima, structural timeouts, numerical artifacts, and subtle semantic divergence between formulations. We introduce SOVER, an LLM-assisted SMT framework that separates semantic mapping from formal certification: Z3 checks domain cross-feasibility and global objective-order preservation for mixed-integer linear formulations, while dReal provides tolerance-aware feasibility/range and $\epsilon$-argmin checks for continuous nonlinear formulations. We also introduce NLEquiv-150, a public benchmark of 100 equivalent and 50 deliberately hard non-equivalent nonlinear reformulation pairs. With LLM-extracted mappings, SOVER classifies 149/150 pairs
    
[^89]: MaskCode：基于线性分组码实现反馈辅助编码的掩码Transformer

    MaskCode: Mask Transformer for Feedback-Assisted Coding With Linear Block Codes

    [https://arxiv.org/abs/2609.00715](https://arxiv.org/abs/2609.00715)

    提出了MaskCode，一种基于Transformer的内层反馈码，通过掩码机制将外层线性分组码的结构知识显式融入反馈编码器设计，避免将反馈资源浪费在外层纠错码已能纠正的错误模式上，从而提升级联编码系统的性能。

    

    基于反馈的编码方案相比当今的开环编码方案已展现出显著的性能提升。遗憾的是，这些提升通常是在具有完美反馈的理想化环境中实现的。近年来，基于机器学习的方案已被证明是实现基于反馈的码的有前景的解决方案，尤其是当其与短块长的开环纠错码（ECC）以级联编码结构相结合时。然而，现有的基于机器学习的反馈方案对外码的结构仍然一无所知，这可能将反馈资源错误地分配在外层纠错码本已能够纠正的错误模式上。为解决这一问题，我们提出了MaskCode，一种用于级联编码系统的基于Transformer的内层反馈码，它通过两种协同机制将外层线性分组码的结构知识显式地融入到内层反馈编码器的设计中：1）一种软伴随式——（原文摘要至此截断）

    arXiv:2609.00715v1 Announce Type: cross  Abstract: Feedback-based coding schemes have demonstrated substantial performance gains over today's open-loop coding schemes. Unfortunately, these gains are usually achieved in idealized settings with perfect feedback. Over the last few years, machine learning-based schemes have been shown to be promising solutions for implementing feedback-based codes, particularly when combined with short-block-length open-loop error correcting codes (ECCs) in a concatenated coding structure. However, existing ML-based feedback schemes remain agnostic to the outer code's structure, potentially misallocating feedback resources on error patterns already correctable by the outer ECC. To address this, we propose MaskCode, a Transformer-based inner feedback code for concatenated coding systems, which explicitly incorporates structural knowledge of the outer linear block code into the inner feedback encoder design via two synergistic mechanisms: 1) a soft syndrome-
    
[^90]: 基于提示条件化场景奖励的可控图像描述生成

    Controllable Image Captioning with Prompt-Conditioned Scene Rewards

    [https://arxiv.org/abs/2609.00709](https://arxiv.org/abs/2609.00709)

    提出FoCUS方法，通过基于场景图对齐组件分数的提示条件化奖励目标并用GRPO优化，让用户能够通过自然语言提示精确控制图像描述的语义重点（如对象、属性、关系或特定区域）。

    

    大型视觉语言模型能够生成流畅的图像描述，但语义控制能力有限：用户无法可靠地指定描述应强调属性、关系还是特定的图像区域。我们提出了FoCUS（Fine-grained Captioning Control Using Scene Rewards，基于场景奖励的细粒度描述控制），这是一种可控图像描述生成方法，允许用户通过自然语言控制提示将图像描述引导至特定的语义重点。其核心思想是基于场景图对齐组件分数构建提示条件化的控制目标：生成的描述会被解析并对齐到场景图组件（如对象、属性和关系），并根据所要求的重点对这些组件进行差异化加权，包括负权重。我们使用GRPO优化该目标，并通过更严格的对象有效性阈值以及基于推理的属性和关系评分验证来进一步提高奖励的可靠性。

    arXiv:2609.00709v1 Announce Type: cross  Abstract: Large Vision-Language Models produce fluent image descriptions but offer limited semantic control: users cannot reliably specify whether captions should emphasize attributes, relations, or particular image regions. We present Fine-grained Captioning Control Using Scene Rewards (FoCUS), a controllable image captioning method that lets users steer captions toward specific semantic emphases through natural-language control prompts. The core idea is a prompt-conditioned control objective based on scene-graph-aligned component scores. Generated captions are parsed and aligned to scene-graph components such as objects, attributes, and relations. These components are differentially weighted, including negative weights, according to the requested emphasis. We optimize this objective with GRPO and further improve its reliability through a stricter object validity threshold and reasoning-based verification for attribute and relation scoring. To 
    
[^91]: 实践中的模式化：基于敏感度的奖励模型去偏

    Patterning in Practice: Debiasing Reward Models with Susceptibilities

    [https://arxiv.org/abs/2609.00699](https://arxiv.org/abs/2609.00699)

    本文提出基于敏感度的模式化重新加权方法来去除奖励模型的风格偏差，在 RM-Bench Hard 上提升 14.2 个百分点且整体准确率保持不变，且权重具有可解释性和跨模型可迁移性。

    

    已知基于人类偏好训练的奖励模型存在长度、格式及其他风格方面的偏差。在本文中，我们使用模式化方法，根据每个偏好对对基准损失后验期望值的测量效应（即其敏感度）进行重新加权，从而对基于 Skywork-Reward-Preference v0.2 训练的 Gemma 2 9B Instruct 奖励模型进行去偏。我们在 RM-Bench Hard（即风格线索与正确性相反的划分）上获得了 +14.2 ± 1.2 个百分点的提升（5个随机种子上的均值±标准误），同时整体 RM-Bench 准确率得以保持，与已发表的最近似对比方法所报告的最强 Hard 划分增益相当（SteerRM，+13.2 个百分点）。我们在一个简单案例中证明了这种重新加权的可解释性：通过追踪该干预的一个副作用（RM-Bench 安全子集上的性能回归），我们将其定位到一小类训练对，并通过消融实验加以确认。这些权重还具备迁移性：在 Gemma 2 9B 上计算得到的权重……（原文摘要在此处截断）

    arXiv:2609.00699v1 Announce Type: new  Abstract: Reward models trained on human preferences are known to suffer from length, formatting, and other stylistic biases. In this paper we use patterning, which reweights each preference pair according to its measured effect on posterior expectation values of benchmark losses (its susceptibility), to debias a Gemma 2 9B Instruct reward model trained on Skywork-Reward-Preference v0.2. We obtain $+14.2 \pm 1.2$ pp on RM-Bench Hard, the split where style cues point against correctness (mean $\pm$ s.e.\ over 5 seeds), with overall RM-Bench accuracy preserved, comparable to the strongest Hard-split gain reported by the closest published comparator (SteerRM, $+13.2$ pp). We demonstrate in a simple case that the reweighting is interpretable by tracing a side effect of the intervention (a regression on a safety subset of RM-Bench) to a small class of training pairs, which we confirm by ablation. The weights also transfer: those computed on Gemma 2 9B 
    
[^92]: MUGEN：面向多种学习任务的不可学习图样本生成

    MUGEN: Generating Unlearnable Graph Examples for Multiple Learning Tasks

    [https://arxiv.org/abs/2609.00696](https://arxiv.org/abs/2609.00696)

    MUGEN是首个面向多种学习任务的不可学习图样本生成框架，它通过对单一干净数据集进行特征扰动，利用共享GNN编码器同时保护节点分类、图分类和链接预测等多种任务免受未经授权的模型学习。

    

    跨领域的图数据可能向未经授权的表示学习暴露有价值的关系信息，因此迫切需要防范此类滥用。不可学习样本提供了一种数据级防御手段，通过对训练发布数据进行扰动，使得在其上训练的模型无法泛化到干净数据。现有方法只能针对特定的下游任务生成不可学习图样本。因此，针对某一任务受保护的发布数据，对于数据所有者无法预料的其他潜在用途（包括节点分类、图分类和链接预测）仍可能是可学习的。我们提出了MUGEN，据我们所知，这是首个能够联合保护所有启用任务的不可学习图样本生成框架。MUGEN从一个干净数据集生成单一的特征扰动发布版本，通过共享的GNN编码器和任务特定的输出头保护每一个启用的任务。我们设计了一种任务对齐的（摘要在此处不完整）

    arXiv:2609.00696v1 Announce Type: new  Abstract: Graph data across diverse domains can expose valuable relational information to unauthorized representation learning, creating a pressing need for protection against such misuse. Unlearnable examples offer a data-level defense by perturbing a training release so that models trained on it fail to generalize to clean data. Existing methods generate unlearnable graph examples for only a specified downstream task. Consequently, a release protected against one task may remain learnable for other plausible uses, including node classification, graph classification, and link prediction, which the data owner cannot anticipate. We introduce MUGEN, to our knowledge the first framework for generating unlearnable graph examples that jointly protect all enabled tasks. From one clean dataset, MUGEN produces a single feature-perturbed release that protects every enabled task through a shared GNN encoder and task-specific heads. We devise a Task-Aligned 
    
[^93]: 参考集重采样下OOD分数的判定不稳定性

    Verdict Instability of OOD Scores under Reference Resampling

    [https://arxiv.org/abs/2609.00691](https://arxiv.org/abs/2609.00691)

    本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。

    

    事后（post-hoc）分布外（OOD）检测器是在有限的参考集上拟合的，因此它们产生的每一个分数都只是一个估计值。如果我们选择了另一个参考集，某些判定结果就会发生改变。我们通过重采样参考集并记录分数的自助法（bootstrap）标准差来度量这种变动，并将其称为“判定不稳定性”。该量具有无需拟合任何参数的闭式解。一个判定的不稳定性等于所分配类别沿查询方向的类内离散度除以该类别参考样本数的平方根。正是这一样本数将判定不稳定性与分数分布的几何结构区分开来，且只有在类别不平衡的情况下才能将其识别出来。不稳定性随局部离散度的增大而增长。远分布外（Far-OOD）查询位于各向异性嵌入的低方差方向上，因此在我们测试的所有基于距离的分数中，最高分数值都被分配给了那些最可复现的判定。只有es…（摘要不完整）

    arXiv:2609.00691v1 Announce Type: new  Abstract: Post-hoc out-of-distribution detectors are fitted on a finite reference set, so every score they produce is an estimate. If we had chosen a different set, some verdicts would have moved. We measure that movement by resampling the reference set and recording the bootstrap standard deviation of the score, which we call verdict instability. It admits a closed form with no fitted parameters. The instability of a verdict is the within-class dispersion of the assigned class along the query's direction, divided by the square root of that class's reference count. That count is what separates verdict instability from the geometry of the score distribution, and it is identifiable only under class imbalance. Instability grows with the local dispersion. Far-OOD queries lie along the low-variance directions of an anisotropic embedding, so every distance-based score we test assigns its highest values to the verdicts that are most reproducible. Only es
    
[^94]: 预测编码网络中隐状态优化顺序的研究

    A Study of Hidden-State Optimization Order in Predictive Coding Networks

    [https://arxiv.org/abs/2609.00686](https://arxiv.org/abs/2609.00686)

    该论文提出一种边界优先的隐状态优化顺序——先协调块边界处的隐状态、再细化块内表示，显著提升了预测编码网络在CIFAR-10上的准确率并增强了早期层的特征学习。

    

    局部学习方法为端到端反向传播提供了一种替代方案，但其非结构化的局部目标可能导致深度网络的特征学习能力较弱。我们研究隐状态优化的顺序是否能够解决这一局限。我们提出了一种边界优先的推理调度方法，该方法将模型划分为若干块，首先协调各块边界处的隐状态，然后再细化每块内部的表示。我们在预测编码网络（PCNs）中实现了这一调度方法，预测编码网络是一种局部学习框架，其中隐层活动和预测误差在推理过程中被显式地暴露出来。在CIFAR-10数据集上，这种边界优先的预测编码实现在标准参数化下比标准预测编码提高了9.77%的准确率，在μ参数化下提高了5.51%的准确率。诊断分析进一步显示出更多的早期层非平凡更新、更低的初始到最终CKA相似度，以及更多的……

    arXiv:2609.00686v1 Announce Type: cross  Abstract: Local learning methods offer an alternative to end-to-end backpropagation, but their unstructured local objectives can produce weak feature learning in deep networks. We study whether the order of hidden-state optimization can address this limitation. We propose a boundary-first inference schedule that partitions a model into chunks, first coordinates hidden states at chunk boundaries, and then refines representations within each chunk. We instantiate this schedule in predictive coding networks (PCNs), a local-learning framework in which hidden activities and prediction errors are explicitly exposed during inference. On CIFAR-10, the resulting boundary-first predictive-coding instantiation improves accuracy over standard predictive coding by $9.77\%$ under a standard parametrization and by $5.51\%$ under a $\mu$-parametrization. Diagnostic analyses further show more non-trivial early-layer updates, lower initial-to-final CKA, and more 
    
[^95]: HarmoCore：用于振荡波场稀疏重建的函数式潜在扩散

    HarmoCore: Functional Latent Diffusion for Sparse Reconstruction of Oscillatory Wave Fields

    [https://arxiv.org/abs/2609.00679](https://arxiv.org/abs/2609.00679)

    HarmoCore将生成式扩散先验置于由函数式Tucker核心构成的紧凑连续波场潜在空间中，并直接在该核心空间执行频率条件化的扩散后验采样，从而从极端稀疏的传感器观测中高效重建复值振荡波场。

    

    从稀疏散布的传感器重建振荡波场是一个严重欠定的逆问题。除了一般物理场重建所面临的挑战之外，波响应是复值的、对频率敏感且高度振荡的，而昂贵的仿真与传感成本通常只能留下极端稀疏的观测。现有的低秩、算子学习和扩散方法大多是为实值、较平滑的场设计的；稠密像素空间的扩散对于振荡复值场尤其低效，且难以扩展到三维。我们提出HarmoCore，它将生成先验置于一个紧凑、连续且结构化的波场潜在空间中。HarmoCore通过共享的连续空间基函数，利用函数式Tucker核心表示实部-虚部联合通道，学习频率条件化的核心扩散先验，并直接在核心空间中执行扩散后验采样。在固定的传感器坐标下，多……

    arXiv:2609.00679v1 Announce Type: new  Abstract: Reconstructing oscillatory wave fields from scattered sensors is a severely underdetermined inverse problem. Beyond the challenges of general physical-field reconstruction, wave responses are complex-valued, frequency-sensitive, and highly oscillatory, while costly simulation and sensing often leave only extreme-sparse observations. Existing low-rank, operator, and diffusion approaches are largely designed for real-valued, smoother fields; dense pixel-space diffusion is particularly inefficient for oscillatory complex fields and difficult to scale to 3D. We propose HarmoCore, which places a generative prior in a compact, continuous, and structured wave-field latent. HarmoCore represents joint real--imaginary channels with Functional Tucker cores over shared continuous spatial bases, learns a frequency-conditioned core diffusion prior, and performs Diffusion Posterior Sampling directly in core space. At fixed sensor coordinates, the multi
    
[^96]: EEG-AS：通过行为重构实现EEG基础模型的实例级选择

    EEG-AS: Instance-Level Foundation Model Selection for EEG Foundation Models via Behavior Reconstruction

    [https://arxiv.org/abs/2609.00653](https://arxiv.org/abs/2609.00653)

    该论文提出EEG-AS框架，将EEG基础模型选择形式化为实例级算法选择问题，通过锚定模型的特权预测标记重构不可获得的基础模型行为，从而为每个EEG实例自动选择最合适的基础模型。

    

    脑电图（EEG）是一种测量神经活动的非侵入性技术，已在神经科学领域得到广泛应用。EEG基础模型的最新进展使其在多种神经解码任务中均能取得出色的性能。然而，没有任何单一基础模型能在所有数据集或个体EEG实例上始终保持最佳表现，而实例级的模型选择在很大程度上仍处于未被探索的状态。为解决这一局限性，我们将EEG基础模型选择构建为一个实例级算法选择（Algorithm Selection, AS）问题。我们提出了EEG-AS，这是一个实例级算法选择框架，它利用推理时可获得的潜在EEG嵌入、手工设计的神经生理学特征以及锚定基础模型来刻画每个EEG实例。在训练过程中，EEG-AS学习从以锚定基础模型为条件的特权预测标记中重构不可获得的基础模型行为。

    arXiv:2609.00653v1 Announce Type: cross  Abstract: Electroencephalography (EEG) is a non-invasive technique for measuring neural activity and has been widely used in neuroscience applications. Recent advances in EEG foundation models have enabled strong performance across diverse neural decoding tasks. However, no single foundation model consistently performs best across datasets or individual EEG instances, while instance-level model selection remains largely unexplored. To address this limitation, we formulate EEG foundation model selection as an instance-level Algorithm Selection (AS) problem. We propose \textbf{EEG-AS}, an instance-level algorithm selection framework that characterizes each EEG instance using inference-available latent EEG embeddings, handcrafted neurophysiological features, and an anchor foundation model. During training, EEG-AS learns to reconstruct unavailable foundation-model behaviors from privileged prediction tokens conditioned on an anchor foundation model,
    
[^97]: 自我报告并非验证：进化搜索中LLM操作者的环境锚定审计

    Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search

    [https://arxiv.org/abs/2609.00652](https://arxiv.org/abs/2609.00652)

    本文提出首个环境锚定的LLM操作者审计框架，通过为进化式Contexto搜索中的每个中间提议赋予精确结果，实证证明模型自我报告不可作为验证依据——操作者将成功率夸大4.8至9.3倍，且关于置信度校准、理由传递和适应度选择的三个假设全部被证伪。

    

    语言模型智能体日益频繁地提出行动、观察外部反馈并解释自身行为。它们的置信度和理由是便捷的监控信号，但便捷并不等于验证。我们引入了一种环境锚定的审计方法，其中每个中间提议都会获得精确的结果反馈。一个语言模型操作进化式Contexto搜索，其反馈函数无需人工标注即可为每个有效猜测分配精确排名。在涵盖五种配置和三个模型系列的200次运行中，四种报告配置共产生了12,249份自我报告。我们检验了三个假设：所述置信度是经过校准的、继承的理由会影响后续提议、以及基于适应度的选择能够提升报告质量。这三个假设全部失败。操作者将进入前100名的成功率夸大了4.8至9.3倍，而校准能力与区分能力在不同模型系列之间出现了分离。对754个继承理由的受控干预（摘要在此处被截断）

    arXiv:2609.00652v1 Announce Type: new  Abstract: Language model agents increasingly propose actions, observe external feedback, and explain their own behavior. Their confidence and rationales are convenient monitoring signals, but convenience is not verification. We introduce an environment-grounded audit in which every intermediate proposal receives an exact outcome. A language model operates an evolutionary Contexto search whose feedback function assigns every valid guess an exact rank without human annotation. Across 200 runs spanning five configurations and three model families, four reporting configurations produce 12,249 self-reports. We test three assumptions: stated confidence is calibrated, inherited rationales affect later proposals, and fitness-based selection improves report quality. All three fail. Operators overstate top-100 success by factors of 4.8 to 9.3, while calibration and discrimination dissociate across model families. Controlled interventions on 754 inherited ra
    
[^98]: DK-GBMKKM：动态核空间粒球多核k均值聚类

    DK-GBMKKM: Dynamic Kernel-Space Granular-Ball Multiple Kernel $k$-Means Clustering

    [https://arxiv.org/abs/2609.00647](https://arxiv.org/abs/2609.00647)

    提出DK-GBMKKM方法，通过在融合核空间中动态生成粒球并交替优化核权重与粒球隶属度，同时构造样本规模加权的粒球核，从而提升了多核k均值聚类对噪声、边界样本的鲁棒性以及对核空间几何变化的适应性。

    

    多核k均值通过学习基核的组合来整合互补的非线性相似性。然而，其逐点优化方式对噪声样本和边界样本敏感，并且需要反复在样本规模的核矩阵上进行运算。粒球表示将局部样本组组织成介观单元，但在输入空间中一次性生成的粒球可能与多核学习过程中不断演化的融合核几何结构不一致。本文提出动态核空间粒球多核k均值聚类（DK-GBMKKM）。该方法在当前融合核空间中生成粒球，并交替进行核权重学习与粒球隶属度更新，使表示能够适应融合核几何结构的变化。进一步构造了样本规模加权的粒球核，以保留不同大小粒球的贡献，并且其半正定性……（摘要原文在此处被截断）

    arXiv:2609.00647v1 Announce Type: new  Abstract: Multiple kernel $k$-means integrates complementary nonlinear similarities by learning a combination of base kernels. Its pointwise optimization, however, is sensitive to noisy and boundary samples and repeatedly operates on sample-scale kernel matrices. Granular-ball representations organize local sample groups into mesoscopic units, but granular balls generated once in the input space may be inconsistent with the fused-kernel geometry that evolves during multiple kernel learning. We propose dynamic kernel-space granular-ball multiple kernel $k$-means (DK-GBMKKM). The method generates granular balls in the current fused kernel space and alternates kernel-weight learning with granular-ball membership updates, allowing the representation to adapt to changes in the fused-kernel geometry. A sample-size-weighted granular-ball kernel is further constructed to preserve the contributions of balls of different sizes, and its positive semidefinite
    
[^99]: 打破结构同一性：秩异构下的个性化联邦LoRA微调

    Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity

    [https://arxiv.org/abs/2609.00632](https://arxiv.org/abs/2609.00632)

    提出FedRoRA框架，通过将LoRA适配解耦为共享的全局方向与个性化的按秩幅值，在秩异构联邦学习场景下实现细粒度的客户端个性化微调，从而同时应对资源异构与数据异构的双重挑战。

    

    大语言模型（LLM）在众多领域取得了显著成功，但其在隐私敏感的分布式数据集上的适配仍然是一个挑战。虽然联邦学习（FL）与低秩适配（LoRA）相结合为协同微调提供了一种资源高效的范式，但实际部署受到资源异构性与数据异构性双重挑战的阻碍。现有的秩异构方法主要致力于弥合聚合过程中的维度不匹配问题，但通常为共享相同秩的所有客户端提供统一的全局模型，无法在非独立同分布（non-IID）场景下捕捉客户端特有的特征。本文提出了FedRoRA（联邦按秩个性化LoRA），这是一种能够在秩异构联邦中实现细粒度个性化的新型框架。FedRoRA将适配过程解耦为共享的全局方向和个性化的按秩幅值……

    arXiv:2609.00632v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable success across diverse domains, but their adaptation to privacy-sensitive, distributed datasets remains a challenge. While Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) provides a resource-efficient paradigm for collaborative fine-tuning, practical deployments are hindered by the dual challenges of resource heterogeneity and data heterogeneity. Existing rank-heterogeneous methods primarily focus on bridging dimension mismatches for aggregation but typically provide a unified global model for all clients sharing the same rank, failing to capture client-specific features in non-IID scenarios. In this paper, we propose FedRoRA (Federated Rank-wise Personalized LoRA), a novel framework that enables fine-grained personalization within rank-heterogeneous federations. FedRoRA decouples adaptation into shared global directions and personalized rank-wise magnitudes governe
    
[^100]: 坦白你所知：大语言模型机器遗忘中遗忘集与模型知识的不对齐问题

    Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning

    [https://arxiv.org/abs/2609.00605](https://arxiv.org/abs/2609.00605)

    提出数据无关的CONFS框架，通过引出模型自身记忆的知识来构建与模型对齐的遗忘集，解决了大语言模型机器遗忘中遗忘集与模型实际记忆内容不对齐所导致的信息泄露或效用下降问题。

    

    大语言模型（LLM）的机器遗忘通常假设预定义的遗忘集与模型实际记忆的内容相匹配，但在原始训练数据不可访问的现实隐私场景中，这一假设经常失效。我们将这种差距称为“遗忘集不对齐”，并识别出两种情况：在“遗忘不足”中，遗忘集遗漏了模型已记忆的信息，导致信息泄露持续存在；在“知识外遗忘”中，算法被驱动去“遗忘”模型从未学过的知识，从而扰动参数并降低模型效用。通过梯度层面的分析，我们证明这些行为源于不对齐的遗忘目标，而非特定的优化方法选择。随后，我们提出了CONfession-to-Forget-Set（CONFS），这是一个数据无关的框架，通过引出并形式化模型自身已记忆的知识来构建与模型对齐的遗忘集。在合成数据、多模态和真实世界基准测试中，CONFS均接近金标准性能。

    arXiv:2609.00605v1 Announce Type: cross  Abstract: Machine unlearning for large language models (LLMs) often assumes that a pre-defined forget set matches what the model has memorized, but this frequently breaks in realistic privacy settings where the original training data is inaccessible. We term this gap forget-set misalignment and identify two cases. In Under Unlearning, the forget set omits memorized information and leakage persists. In Out-of-Knowledge Unlearning, the algorithm is driven to "forget" knowledge the model never learned, perturbing parameters and degrading utility. Using gradient-level analysis, we show these behaviors arise from misaligned unlearning targets rather than specific optimization choices. We then propose CONfession-to-Forget-Set (CONFS), a data-blind framework that constructs model-aligned forget sets by eliciting and formalizing the model's memorized knowledge. Across synthetic, multimodal, and real-world benchmarks, CONFS approaches Gold-standard perfo
    
[^101]: 拓扑引导

    Topological Steering

    [https://arxiv.org/abs/2609.00597](https://arxiv.org/abs/2609.00597)

    该论文提出了“拓扑引导”新框架，利用拓扑数据分析中的持久图对激活空间进行拓扑表示，从而以对离群值和局部扰动更鲁棒的方式控制大语言模型的行为。

    

    随着大语言模型（LLM）的迅速崛起，控制模型的不良行为变得日益重要。现有的行为控制方法通常直接在激活空间或特征空间中进行干预，但这类方法容易受到离群值、分布偏移、噪声以及其他局部扰动的影响。受拓扑数据分析（TDA）捕捉全局而非纯局部结构的特性启发，我们提出了拓扑引导，这是一种通过激活空间的拓扑表示来引导大语言模型行为的新框架。借助持久图，我们的方法将基于激活的引导与拓扑数据分析联系起来，从而实现更鲁棒的行为控制。我们证明了拓扑引导能够在多个模型家族和不同模型规模上一致地改变大语言模型的行为。

    arXiv:2609.00597v1 Announce Type: new  Abstract: With the rapid rise of large language models (LLMs), controlling undesirable model behaviors has become increasingly important. Existing behavioral control methods typically intervene directly in activation or feature space, but such approaches can be sensitive to outliers, distributional shifts, noise, and other local perturbations. Motivated by Topological Data Analysis (TDA), which captures global rather than purely local structure, we propose Topological Steering, a new framework for steering LLM behavior through the topological representation of activation spaces. Using persistence diagrams, our method connects activation-based steering with TDA and enables more robust behavioral control. We show that Topological Steering consistently modifies LLM behavior across multiple model families and model sizes.
    
[^102]: 实时神经形态频谱智能模拟器

    Real-Time Neuromorphic Spectrum Intelligence Simulator

    [https://arxiv.org/abs/2609.00585](https://arxiv.org/abs/2609.00585)

    本文提出了RT-NuSIS模块化实时神经形态模拟器，将脉冲神经网络、忆阻突触、物理信息能量收集与对抗模型相结合，为受限能量下的动态频谱接入研究提供数学形式化保证及可复现的能效、延迟与鲁棒性基准测试平台。

    

    我们提出了实时神经形态频谱智能模拟器（RT-NuSIS），这是一个模块化框架，用于研究脉冲神经网络（SNN）和受忆阻器启发的智能体在受限能量预算与对抗条件下的动态频谱接入。RT-NuSIS 将泄漏积分放电神经元动力学、忆阻突触模型、基于物理的能量收集模型（摩擦电和射频）以及包括干扰和拜占庭行为在内的对抗模型耦合在一起。我们从数学上对该模拟器进行了形式化，证明了其有界性，提出了平均场对抗阈值，分析了每步计算复杂度，并提供了一个可复现的基准测试框架，用于衡量每次推理的能量消耗、延迟和鲁棒性等指标。该代码库具有模块化结构，可通过随机种子实现确定性，并专为大规模事件驱动仿真而设计。

    arXiv:2609.00585v1 Announce Type: cross  Abstract: We present the Real-Time Neuromorphic Spectrum Intelligence Simulator (RT-NuSIS), a modular framework to study spiking neural network (SNN) and memristor-inspired agents for dynamic spectrum access under constrained energy budgets and adversarial conditions. RT-NuSIS couples leaky integrate-and-fire neuronal dynamics, memristive synaptic models, physics-informed energy-harvesting models (triboelectric and RF), and adversary models including jamming and Byzantine behavior. We formalize the simulator mathematically, prove boundedness, present a mean-field adversary threshold, analyze per-step complexity, and provide a reproducible benchmark harness for energy-per-inference, latency, and robustness metrics. The codebase is modular, deterministic by seed, and designed for large-scale event-driven simulations.
    
[^103]: GeoPAR：基于几何引导并行自回归学习的大规模多智能体组合优化

    GeoPAR: Large-Scale Multi-Agent Combinatorial Optimization with Geometry-Guided Parallel Autoregressive Learning

    [https://arxiv.org/abs/2609.00577](https://arxiv.org/abs/2609.00577)

    GeoPAR提出了一种几何引导的并行自回归强化学习框架，通过投影窗口稀疏几何机制、稀疏边偏置注意力以及缓存引导的冲突处理机制，实现了大规模多智能体组合优化问题的高效求解。

    

    多智能体组合优化问题因其NP难特性而极具挑战性。近期的并行自回归神经求解器通过允许智能体同时进行决策来提升推理效率，但其性能在大规模实例上往往会退化。这主要归因于对局部几何结构建模薄弱，以及冲突任务选择仅在动作生成之后才被处理这一事实。为解决这些局限，我们提出了GeoPAR——一个面向可扩展多智能体组合优化的几何引导并行自回归强化学习框架。GeoPAR集成了三个关键组件：（1）投影窗口稀疏几何机制，通过多方向投影构建轻量级的局部候选邻域；（2）稀疏边偏置注意力，将这些几何关系注入节点表示中；（3）缓存引导的冲突-（原文摘要在此处截断）

    arXiv:2609.00577v1 Announce Type: cross  Abstract: Multi-agent combinatorial optimization problems are notoriously challenging due to their NP-hard nature. Recent parallel autoregressive neural solvers improve inference efficiency by allowing agents to make decisions simultaneously, but their performance often degrades on large-scale instances. This is largely attributable to weak modeling of local geometric structures and the fact that conflicting task selections are handled only after action generation. To address these limitations, we propose GeoPAR, a geometry-guided parallel autoregressive reinforcement learning framework for scalable multi-agent combinatorial optimization. GeoPAR integrates three key components: (1) a projection-window sparse geometry mechanism that builds lightweight local candidate neighborhoods through multi-directional projections, (2) sparse edge-biased attention that injects these geometric relations into node representations, and (3) cache-guided conflict-
    
[^104]: VoiceLongMemEval：助手是否记得你的声音听起来如何？

    VoiceLongMemEval: Do Assistants Remember How You Sounded?

    [https://arxiv.org/abs/2609.00570](https://arxiv.org/abs/2609.00570)

    该论文提出了VoiceLongMemEval（VLME）基准，用于评估AI助手在长时多会话对话中能否记住情感、韵律和语音事件等副语言信息，发现现有大语言模型普遍存在无法捕捉说话方式的“情感鸿沟”。

    

    随着多智能体架构和大语言模型规模的不断增长，部署的AI助手越来越多地需要对长且连续的多会话对话历史进行推理。当前的基准测试将这种对话历史评估视为长时程信息检索、时间推理或知识更新，却关键性地忽略了人机交互的基本动态，即“他们是怎么说的”（说话方式）。为了填补这一空白，我们提出了VoiceLongMemEval（VLME）基准，其中每个问题的答案都依赖于附加在对话轮次上的副语言元数据（情感标签、韵律描述符和语音事件），而这些信息仅凭文字本身是无法恢复的。每个测试项都经过三阶段对抗性门控验证，确保强大的语言模型在仅获得文本转录时无法回答。对领先的前沿模型和开放权重模型的评估揭示了普遍存在的“情感鸿沟”；提供文本轨道的副语言元数据……（摘要在此处截断）

    arXiv:2609.00570v1 Announce Type: new  Abstract: With the growing scale of multi-agent architectures and large language models, deployed AI assistants are increasingly tasked with reasoning over long, continuous, multi-session conversation histories. Current benchmarks evaluate this dialogue history as information retrieval over long horizon, temporal reasoning, or knowledge updates, while crucially ignoring the fundamental dynamics of human-agent interaction, i.e. how they said it. To address this gap, we present VoiceLongMemEval (VLME) benchmark, where every answer depends on paralinguistic metadata (emotion labels, prosody descriptors, and voice events) attached to conversational turns, which is otherwise unrecoverable from the words alone. Every item passes a three-stage adversarial gate, ensuring that a strong language model fails when given only the transcript. Evaluating leading frontier and open-weight models reveals a pervasive affect gap; providing text-track paralinguistic m
    
[^105]: EEG-VID：面向脑电解码与辅助目标选择的任务引导式潜变量预测预训练

    EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection

    [https://arxiv.org/abs/2609.00566](https://arxiv.org/abs/2609.00566)

    EEG-VID提出了一种任务引导式潜变量预测预训练框架，通过指数移动平均编码器预测未来EEG潜状态，在42组跨会话跨被试对比中有41组提升准确率（最高提升16.22个百分点），并可有效应用于场景约束下的辅助目标选择。

    

    我们提出EEG-VID，这是一个面向跨会话与跨被试脑电解码的任务引导式潜变量预测预训练框架。EEG-VID利用指数移动平均目标编码器与弱任务引导，从近期的EEG历史预测未来的潜变量EEG状态，随后进行有监督微调。在VIG-48和BCI竞赛IV-2a/IV-2b数据集上，第一阶段预训练在42组匹配的骨干网络-数据集-协议对比中有41组提升了平均准确率，包括全部12个留一被试设置，最大提升达16.22个百分点。在48区域的跨天VIG-48任务上，EEG-VID实现了6.52%的Top-1准确率和30.50%的Top-5准确率。在一项独立的六名被试离线机器人场景研究中，经过被试特定校准后，候选约束下的目标选择准确率达到40.24%，而随机水平仅为25%。这些结果支持任务引导的潜变量预测作为脑电解码与场景约束辅助目标选择的一种可迁移预训练策略。

    arXiv:2609.00566v1 Announce Type: cross  Abstract: We propose EEG-VID, a task-guided latent predictive pretraining framework for EEG decoding under session and subject shifts. EEG-VID predicts future latent EEG states from recent history using an exponential-moving-average target encoder and weak task guidance, followed by supervised fine-tuning. Across VIG-48 and BCI Competition IV-2a/IV-2b, Stage 1 improves mean accuracy in 41 of 42 matched backbone-dataset-protocol comparisons, including all 12 leave-one-subject-out settings, with a maximum gain of 16.22 percentage points. On the 48-region cross-day VIG-48 task, EEG-VID achieves 6.52% Top-1 and 30.50% Top-5 accuracy. In a separate six-participant offline robot-scene study, candidate-constrained target selection reaches 40.24% versus a 25% chance level after subject-specific calibration. These results support task-guided latent prediction as a transferable pretraining strategy for EEG decoding and scene-constrained assistive target s
    
[^106]: 面向滞后容忍分布式计算的流形感知通用编码计算

    Manifold-Aware General Coded Computing for Straggler-Resilient Distributed Computing

    [https://arxiv.org/abs/2609.00552](https://arxiv.org/abs/2609.00552)

    该论文提出一种流形感知的通用编码计算方法，通过在码设计中保留并利用输入数据的内在结构（而非像传统信源编码那样消除结构），实现抗滞后节点的分布式计算。

    

    现有的编码计算设计并未显式地利用输入数据的内在结构。在通信系统中，统计结构和冗余通常在应用信道编码之前通过信源编码（或压缩）予以去除。然而，这一原则并不能直接移植到编码计算中。在许多计算任务中，尤其是在机器学习中，数据的结构恰恰是计算试图利用以推断输出或学习有意义模式的关键所在。因此，编码计算方案应当在码设计中保留并利用这种结构，而不是通过信源编码将其忽略或消除。这一观察启发了对码构造的一种不同视角。在许多信道编码方案（如里德-所罗门码）中，编码符号是通过在选定点上评估一个低维代数表示而生成的。相比之下，许多高维……

    arXiv:2609.00552v1 Announce Type: new  Abstract: Existing coded-computing designs do not explicitly exploit the intrinsic structure of the input data. In communication systems, statistical structure and redundancy are often removed through source coding (or compression) before channel coding is applied. This principle, however, does not transfer directly to coded computation. In many computational tasks, particularly in machine learning, the structure of the data is precisely what the computation seeks to exploit to infer outputs or learn meaningful patterns. Consequently, coded-computing schemes should preserve and leverage this structure in their code design, rather than ignoring or eliminating it through source coding.   This observation motivates a different perspective on code construction. In many channel-coding schemes, such as Reed-Solomon codes, coded symbols are generated by evaluating a low-dimensional algebraic representation at selected points. In contrast, many high-dimen
    
[^107]: EM^2Mem：面向大型语言模型的事件中心多模态记忆

    EM^2Mem: Event-Centric Multimodal Memory for Large Language Models

    [https://arxiv.org/abs/2609.00551](https://arxiv.org/abs/2609.00551)

    该论文提出EM^2Mem，一种以事件为中心的多模态记忆框架，通过在记忆构建阶段将多模态记录、时间上下文、图谱关系与溯源信息绑定到事件锚点，形成“可直接用于生成”的记忆单元，免去了推理时重建跨模态对齐的负担，并在三个长视频问答基准上将平均准确率较最强记忆基线提升2.0至3.7个百分点。

    

    多模态记忆为长视频问答提供了一种可扩展的接口，但现有方法通常将字幕、视频帧、转录文本、摘要或图谱事实作为孤立的片段进行检索。尽管这些片段可被搜索，却并不“可直接用于生成”：语言模型必须在推理阶段、在上下文受限且归因困难的情况下重建跨模态和时间上的对齐关系。我们提出了EM^2Mem，一个以事件为中心的多模态记忆框架，它在记忆构建阶段将异构证据绑定到事件锚点上。每个以事件为索引的记忆单元对齐多模态记录、时间上下文、图谱关联关系、语义事实以及来源溯源信息，从而能够基于多模态事件（而非特定模态的孤立片段）进行紧凑的证据读取。在三个长视频问答基准上，EM^2Mem 相比最强的记忆基线分别将平均准确率提升2.0、2.4和3.7个百分点，并在严格的事件级评估上……（原文摘要在此处截断）

    arXiv:2609.00551v1 Announce Type: cross  Abstract: Multimodal memory offers a scalable interface for long-video question answering, but existing methods often retrieve captions, frames, transcripts, summaries, or graph facts as isolated fragments. Although searchable, such fragments are not generation-ready: language models must reconstruct cross-modal and temporal alignments at inference time, when context is limited and attribution is difficult. We propose EM^2Mem, an event-centric multimodal memory framework that binds heterogeneous evidence to event anchors during memory construction. Each event-indexed memory cell aligns multimodal records, temporal context, graph-linked relations, semantic facts, and provenance, enabling compact evidence readout over grounded multimodal events rather than modality-specific fragments. Across three long-video QA benchmarks, EM^2Mem improves average accuracy over the strongest memory baseline by 2.0, 2.4, and 3.7 points, improves strict event-level 
    
[^108]: GenONet：一种用于高分辨率降水临近预报的生成算子网络

    GenONet: A Generative operator Network for High-Resolution Precipitation Nowcasting

    [https://arxiv.org/abs/2609.00544](https://arxiv.org/abs/2609.00544)

    该论文提出GenONet，首次将深度算子网络作为生成器嵌入生成对抗网络框架中，实现了长达3小时的高分辨率降水临近预报，能够生成清晰且物理一致的预报结果。

    

    高分辨率降水临近预报对于减轻恶劣天气的影响至关重要，但由于风暴系统的快速演变，这一任务仍然充满挑战。深度学习模型在这项任务中展现出了巨大的潜力，但在较长的预报时效内，其预测能力往往会衰退。这导致预报结果变得越来越模糊，无法捕捉风暴系统的复杂非线性演变。为了解决这些局限性，我们提出了时空U-DeepONet（GenONet），这是一种面向长达3小时长时效降水预报的新型架构，专门设计用于产生清晰且物理一致的预报结果。GenONet的架构开创性地将深度算子网络作为生成器，嵌入生成对抗网络（GAN）框架中来完成这一任务。DeepONet学习降水的连续时间动力学，从而确保在长预报时效内的稳定性。对抗训练……

    arXiv:2609.00544v1 Announce Type: new  Abstract: High-resolution precipitation nowcasting is critical for reducing the impacts of severe weather but remains difficult because of rapid storm evolution. Deep learning models have shown great promise for this task, but their predictive skill often deteriorates over longer forecast horizons. This leads to increasingly blurry forecasts that fail to capture the complex, non-linear evolution of storm systems. In order to address these limitations, we introduce Spatio-Temporal U-DeepONet (GenONet), a novel architecture for long-range precipitation forecasting up to 3 hours, specifically designed to produce sharp and physically consistent results. GenONet's architecture pioneers the use of a Deep Operator Network (DeepONet) as a generator within a Generative Adversarial Network (GAN) framework for this task. The DeepONet learns the continuous-time dynamics of precipitation, ensuring stability over long forecast horizons. Adversial training again
    
[^109]: DeSyR：一种基于PINN引导结构搜索与物理信息系数精炼的解耦符号恢复框架

    DeSyR: A Decoupled Symbolic Recovery Framework with PINN-Guided Structure Search and Physics-Informed Coefficient Refinement

    [https://arxiv.org/abs/2609.00530](https://arxiv.org/abs/2609.00530)

    该论文提出DeSyR框架，将微分方程的符号恢复过程解耦为两个阶段——先由物理信息神经网络引导候选拓扑结构搜索，再仅基于控制方程和约束条件精炼系数，并从理论上证明了教师误差继承的上界以及仅在物理约束下可条件性恢复精确系数的保证。

    

    当不完美的教师数据引导符号拓扑搜索和系数估计时，从神经网络近似中恢复紧凑的显式解极具挑战性。我们提出了DeSyR，一种面向微分方程的解耦符号恢复框架。物理信息神经网络（PINN）引导重复搜索，构建带有临时常数的候选拓扑。一旦拓扑被固定，其系数仅依据控制方程和规定的约束条件进行精炼，随后通过门控机制进行选择与验证。对于线性固定拓扑参数化，我们刻画了教师误差的继承性，并证明当教师误差投影到模型空间时，有限权重的数据-物理混合拟合会保留一个量级为 $O(\beta^{-1})$ 的教师相关贡献。在适定性、可表示性、零残差可达性和离散确定性的条件下，仅物理精炼可以有条件地恢复精确系数。

    arXiv:2609.00530v1 Announce Type: new  Abstract: Recovering compact explicit solutions from neural approximations is challenging when imperfect teacher data guide symbolic topology search and coefficient estimation. We present DeSyR, a decoupled symbolic recovery framework for differential equations. A physics-informed neural network guides repeated searches to construct candidate topologies with provisional constants. Once a topology is fixed, its coefficients are refined solely from the governing equation and prescribed constraints, followed by gated selection and verification. For linear fixed-topology parameterizations, we characterize teacher-error inheritance and show that finite-weight mixed data--physics fitting retains an $O(\beta^{-1})$ teacher-dependent contribution when the teacher error projects onto the model space. Under well-posedness, representability, zero-residual attainment, and discrete determinacy, physics-only refinement conditionally recovers exact coefficients;
    
[^110]: 为什么多层消息传递有效：图神经网络原子间势的完备性理论

    Why Multi-Layer Message Passing Works: Completeness Theory for Graph Neural Network Interatomic Potentials

    [https://arxiv.org/abs/2609.00528](https://arxiv.org/abs/2609.00528)

    本文提出多层完备性理论，证明在通用性、重叠与连通性条件下，稀疏截断图上的 $L$ 层消息传递与访问完整 $L$ 跳邻域具有同等的表示能力，从而首次严格证明了图神经网络原子间势中使用小于物理相互作用范围的逐层截断消息传递这一通用做法的合理性，并由此推出 DPA3 与 CHGNet 架构具有通用近似能力。

    

    我们证明了超图神经网络——一种具有三体消息传递的不变性架构——是势能面的通用近似器。我们的主要贡献是提出了一个多层完备性理论。我们表明，只要构型是通用的并满足重叠条件和连通性条件，在稀疏的基于截断的图上进行 $L$ 层消息传递，就能达到与访问完整 $L$ 跳邻域相同的表示能力。这为一种普遍做法——即使用多层消息传递且每层截断半径小于物理相互作用范围（几乎所有实用的基于图神经网络的机器学习原子间势都采用这一设置）——提供了首个严格的合理性证明。作为直接推论，我们表明 DPA3 和 CHGNet 两种架构均继承了通用近似性质。

    arXiv:2609.00528v1 Announce Type: new  Abstract: We prove that the Hypergraph Neural Network, an invariant architecture with 3-body message passing, is a universal approximator for potential energy surfaces. Our main contribution is a multi-layer completeness theory. We show that $L$ layers of message passing on sparse, cutoff-based graphs achieve the same representational power as having access to the full $L$-hop neighborhood, provided the configurations are generic, satisfy an overlap condition and a connectivity condition. This provides the first rigorous justification for the common practice of using multi-layer message passing with a per-layer cutoff smaller than the physical interaction range, the setting used by virtually all practical graph neural network based machine-learned interatomic potentials. As immediate consequences, we show that both DPA3 and CHGNet architectures inherit universal approximation.
    
[^111]: 基于Veronese嵌入实现射影平面上的软性Argmax（Soft-Argmax）

    Soft-Argmax for the Projective Plane via the Veronese Embedding

    [https://arxiv.org/abs/2609.00521](https://arxiv.org/abs/2609.00521)

    该论文提出利用Veronese嵌入将无向直线以ℤ₂-不变的方式嵌入到线性空间中，解决了soft-argmax在具有莫比乌斯带拓扑结构的无向直线空间上失效、撕裂相邻直线的问题。

    

    从地平线检测到X射线成像中的纤维结构，许多视觉任务通过在霍夫空间 H=S¹×ℝ（即方向-偏移对 (θ,ρ) 的定义域）中进行峰值检测来恢复直线。可微分管线通过软性argmax（soft-argmax）——一种概率加权平均——来提取坐标，但该方法仅在全局线性空间中才有意义。然而，(θ,ρ) 和 (θ+π,−ρ) 描述的是同一条无向直线，因此 H 是无向直线空间 H/ℤ₂ 的双重覆盖：后者是一个莫比乌斯带，通过对 ℤ₂ 作用下的每一对元素进行粘合而得到。Soft-argmax 作用于覆盖空间 H 上，但由于 H/ℤ₂ 不具备线性结构，它会将几何上相邻的直线撕裂开。因此，我们需要一种将直线以 ℤ₂-不变的方式嵌入到线性空间的方法，使得 soft-argmax 在其上具有良好定义。我们通过对直线采用单位范数齐次向量 ℓ=(1+ρ²)^(−1/2)(…) 进行参数化来实现这一目标。

    arXiv:2609.00521v1 Announce Type: cross  Abstract: From horizon detection to fibre structures in X-ray imaging, many vision tasks recover lines via peak detection in Hough space $H=S^1\times\mathbb{R}$, the domain of orientation-offset pairs $(\theta,\rho)$. Differentiable pipelines extract coordinates via \emph{soft-argmax}, a probability-weighted average that is only meaningful in a globally linear space. However, $(\theta,\rho)$ and $(\theta+\pi,-\rho)$ describe the same undirected line, so $H$ double-covers the space of undirected lines $H/\mathbb{Z}_2$: a M\"obius strip, obtained by identifying each pair under $\mathbb{Z}_2$ action. Soft-argmax operates on the cover $H$, but since $H/\mathbb{Z}_2$ admits no linear structure, it tears geometrically adjacent lines apart. Thus we need a $\mathbb{Z}_2$-invariant embedding of lines into a linear space, on which soft-argmax is well-defined. We achieve this by parametrising lines via unit-norm homogeneous vectors $\ell=(1+\rho^2)^{-1/2}(
    
[^112]: 通过功能感知掩码学习任务特异性抗体表示

    Learning Task-Specific Antibody Representations via Function-Aware Masking

    [https://arxiv.org/abs/2609.00518](https://arxiv.org/abs/2609.00518)

    该论文提出功能感知掩码这一预训练算法家族，通过将掩码位置与特定功能先验（如IMGT注释或结构预测）对齐来学习任务特异性的抗体表示，在结构相关任务上最高提升14%，在CDR相关任务上最高提升5.9倍。

    

    通过掩码语言建模（MLM）预训练的抗体特异性语言模型，学习到的表示对下游序列设计和性质预测任务至关重要。然而，掩码破坏过程本身在预训练期间很少被用作归纳偏置的来源。虽然优先掩码互补决定区（CDRs）能够改善与结合相关的预测，但抗体在多种功能上拥有多样的生物学先验。本文中，我们提出了功能感知掩码，这是一类将掩码位置与特定功能先验（例如来自IMGT注释或结构预测的先验）对齐的预训练算法，用以塑造学习到的表示空间。我们证明，这些针对特定功能的掩码策略在各自的目标任务上显著提升了性能，在结构相关任务上带来高达14%的性能提升，在CDR相关任务上带来高达5.9倍的改进。为了进一步提高性能……

    arXiv:2609.00518v1 Announce Type: new  Abstract: Antibody-specific language models pretrained via masked language modeling (MLM) learn representations that are critical for downstream sequence design and property prediction tasks. Yet, the corruption process itself is rarely leveraged as a source of inductive bias during pretraining. While preferentially masking complementarity-determining regions (CDRs) improves binding-related predictions, antibodies possess diverse biological priors over a variety of functions. Herein, we introduce function-aware masking, a family of pretraining algorithms that align mask placement with specific functional priors (e.g., from IMGT annotations or structure predictions) to shape the learned representation space. We show that these specialist masking strategies significantly improve performance on their respective objectives, yielding up to a 14% gain on structure-related tasks and up to a 5.9x improvement on CDR-related tasks. To further improve perfor
    
[^113]: VATO：面向非定常分离翼型流动的涡力感知Transformer算子

    VATO: A Vortex-Force-Aware Transformer Operator for Unsteady Separated Aerofoil Flows

    [https://arxiv.org/abs/2609.00507](https://arxiv.org/abs/2609.00507)

    VATO将涡力图方法与几何感知Transformer神经算子相结合，在不增加推理成本的前提下，实现了对非定常分离翼型流动中气动载荷的更准确预测。

    

    准确预测非定常分离流动具有挑战性，因为气动载荷依赖于非线性的分离与涡脱落动力学。尽管高保真CFD能够解析这些机制，但其高昂的计算成本限制了其在设计与控制中的反复使用。然而，标准的场级代理模型训练无法区分对气动载荷贡献最大的流动区域。我们提出了VATO（涡力感知Transformer算子），它通过两种互补机制将涡力图方法与几何感知神经算子相结合。VATO-S在训练阶段对局部VFM力贡献场添加监督，不增加模型规模和推理成本。VATO-A则利用VFM贡献场和敏感性场，优先选择与力相关的源位置进行残差交叉注意力。这些方法在双边缘板翼型的非定常CFD数据上进行了评估，涵盖54条轨迹（摘要内容在此处被截断）

    arXiv:2609.00507v1 Announce Type: new  Abstract: Accurate prediction of unsteady separated flows is challenging because the aerodynamic loads depend on nonlinear separation and vortex-shedding dynamics. Although high-fidelity CFD resolves these mechanisms, its cost limits repeated use in design and control. Standard field-level surrogate training, however, does not distinguish the flow regions that contribute most strongly to the aerodynamic loads. We introduce VATO (Vortex-Force-Aware Transformer Operator), which couples the Vortex Force Map (VFM) method to a geometry-aware neural operator through two complementary mechanisms. VATO-S adds training-only supervision of the local VFM force-contribution field, with no increase in model size or inference cost. VATO-A uses VFM contribution and sensitivity fields to prioritise force-relevant source locations for residual cross attention. The methods are evaluated on unsteady CFD data for double-edged-plate aerofoils over 54 trajectories from
    
[^114]: 折扣马尔可夫博弈中的独立强化学习

    Independent Reinforcement Learning in Discounted Markov Games

    [https://arxiv.org/abs/2609.00504](https://arxiv.org/abs/2609.00504)

    本文在“PPAD的ETH”假设下证明了折扣一般和马尔可夫博弈中独立学习计算粗相关均衡的困难性，并提出首个无需结构限制、具有次指数收敛保证的彻底非耦合分层乐观镜像下降算法。

    

    在这项工作中，我们研究了折扣一般和马尔可夫博弈中的彻底非耦合学习。在假设“PPAD的指数时间假说（ETH）”成立的前提下，我们证明：对于每个固定的折扣因子，当玩家在去中心化环境中独立学习时，不存在多项式时间算法来计算折扣一般和马尔可夫博弈中逆多项式精度的粗相关均衡。作为对该困难性结果的补充，我们提供了似乎是首个具有次指数收敛保证、可收敛到粗相关均衡的彻底非耦合算法，且不对博弈施加任何结构性限制。我们的算法是乐观镜像下降的一种分层变体，并采用了为多智能体设置量身定制的递增步长调度方案。最后，我们开发了上述算法的全反馈和部分反馈两个版本，并建立了次……

    arXiv:2609.00504v1 Announce Type: cross  Abstract: In this work, we study radically uncoupled learning in discounted general-sum Markov games. Assuming ``$\mathsf{ETH}$ for $\mathsf{PPAD}$", we show that, for every fixed discount factor, there is no polynomial-time algorithm for computing inverse-polynomially accurate coarse correlated equilibria in discounted general-sum Markov games when players learn independently in decentralized settings. Complementing this hardness result, we provide what appears to be the first \emph{radically uncoupled} algorithm with sub-exponential convergence guarantees to coarse correlated equilibria in discounted general-sum Markov games without imposing any structural restrictions on the game. Our algorithm is a \emph{layered} variant of optimistic mirror descent with an increasing step-size schedule tailored to the multi-agent setting. Finally, we develop both full-feedback and partial feedback versions of the aforementioned algorithm and establish sub-e
    
[^115]: 一种用于学习路径规划（路由）的混合量子-经典神经网络

    A hybrid quantum-classical neural network for learning to route

    [https://arxiv.org/abs/2609.00489](https://arxiv.org/abs/2609.00489)

    本研究提出用小型量子神经网络替换基于注意力的路由模型中的编码器前馈模块，在带容量约束车辆路径问题上将参数减少56.6%的同时保持接近经典神经基线的求解质量，为神经组合优化提供了一种可行的混合模块压缩策略。

    

    本工作研究用于学习路由启发式算法的混合量子-经典神经网络。具体而言，本文探讨小型量子神经网络能否在保持解质量的前提下，替换有竞争力的基于注意力的路由模型中的参数密集型模块。对于带容量约束的车辆路径问题（CVRP），编码器前馈替换被证明是最有前景的设计：它将模型参数数量减少了56.6%，同时使混合模型在中小规模实例上接近经典神经基线，尽管在更大规模实例上差距会扩大。本工作还与经典路由算法进行了比较，后者在固定欧几里得测试集上仍保持高度竞争力且常常表现更优。因此，我们的结果并未表明量子优势或求解器主导地位，而是确定编码器前馈替换作为神经组合优化中一种可行的混合模块压缩策略。

    arXiv:2609.00489v1 Announce Type: new  Abstract: This work studies hybrid quantum-classical neural networks for learning routing heuristics. Specifically, this paper asks whether small quantum neural networks can replace parameter-heavy modules inside a competitive attention-based routing model while maintaining solution quality. For the capacitated vehicle routing problem, encoder feed-forward replacement emerges as the most promising design: it reduces the number of model parameters by 56.6% while keeping the hybrid model close to the classical neural baseline at small and medium instance sizes, although the gap grows for larger instances. This work also compares to classical routing algorithms, which remain highly competitive and often superior on the fixed Euclidean test sets. Our results therefore do not indicate quantum advantage or solver dominance, but identify encoder feed-forward replacement as a viable hybrid-module compression strategy for neural combinatorial optimization.
    
[^116]: AdaptNTK：面向神经网络势的自适应不确定性量化与主动学习

    AdaptNTK: Adaptive Uncertainty Quantification and Active Learning for Neural Network Potentials

    [https://arxiv.org/abs/2609.00488](https://arxiv.org/abs/2609.00488)

    AdaptNTK 提出了一种单模型框架，通过在经验神经正切核（NTK）特征空间中度量正则化马氏距离来量化神经网络势的不确定性，并支持在主动学习采集批次构建过程中递归更新不确定性以避免冗余构型。

    

    机器学习原子间势弥合了量子化学精度与经典计算速度之间的鸿沟，使分子动力学模拟能够以第一性原理精度进行。其可靠性通常通过主动学习加以提升，即通过识别不确定的、分布外的构型来迭代扩充训练集。现有的不确定性量化方法往往在计算成本与可靠性之间存在权衡，并且在组装采集批次时通常无法考虑冗余问题。本文提出 AdaptNTK，一种单模型框架，将不确定性度量为经验神经正切核（NTK）特征空间中的正则化马氏距离。由于 NTK 特征在采集过程中保持固定，不确定性仅取决于所采集的构型，而不依赖于其参考标签。这使得每次选择之后不确定性都可以递归更新，而无需（原文摘要在此处被截断）。

    arXiv:2609.00488v1 Announce Type: new  Abstract: Machine learning interatomic potentials bridge the gap between quantum chemical precision and classical computational speed, enabling molecular dynamics simulations with first-principles accuracy. Their reliability is often improved through active learning, which iteratively expands the training set by identifying uncertain, out-of-distribution configurations. Existing uncertainty-quantification methods often involve a trade-off between computational cost and reliability, and generally cannot account for redundancy as an acquisition batch is assembled. Here, we introduce AdaptNTK, a single-model framework that measures uncertainty as a regularized Mahalanobis distance in empirical neural tangent kernel (NTK) feature space. With the NTK features fixed during acquisition, the uncertainty depends on the acquired configurations but not their reference labels. This allows the uncertainty to be updated recursively after each selection without 
    
[^117]: EvoFlint：多轮LLM漏洞的进化图谱

    EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities

    [https://arxiv.org/abs/2609.00487](https://arxiv.org/abs/2609.00487)

    提出了EvoFlint框架，将多轮红队测试从生成问题重新定义为搜索问题，通过进化式质量多样性搜索演化分阶段对话攻击策略，构建出目标模型漏洞的结构化图谱。

    

    前沿语言模型在单轮有害提示下往往会拒绝回答，但当同样的有害意图通过多轮对话逐步达成时，它们却常常配合执行，这使得多轮攻击成为大型语言模型最不为人理解的失效模式之一。大多数自动化红队测试方法将其视为一个生成问题：生成能够攻破模型的攻击。我们认为将其更好地表述为一个搜索问题：发现、组织并迭代优化一个多样化的攻击策略档案库，从而生成一张关于目标模型如何失效的结构化地图，而非一次性的成功攻击列表。我们提出了EvoFlint，它将进化式质量多样性搜索应用于多轮红队测试。攻击策略是分阶段的对话计划，而非原始提示词，并通过LLM驱动的变异和交叉操作进行演化。基于攻击成功率和峰值严重程度的帕累托适应度保留了来自“险些成功”攻击的选择信号。一个以风险为索引的档案库运行新颖性搜索……

    arXiv:2609.00487v1 Announce Type: cross  Abstract: Frontier language models that refuse harmful single-turn prompts often comply when the same intent is reached gradually over many turns, making multi-turn attacks one of the least understood failure modes of large language models. Most automated red-teaming methods treat this as a generation problem: produce attacks that break the model. We argue it is better framed as a search problem: discover, organize, and iteratively refine a diverse archive of attack strategies, producing a structured map of how a target model fails rather than a list of one-off successes. We introduce EvoFlint, which applies evolutionary quality-diversity search to multi-turn red-teaming. Attack strategies are phased conversation plans, not raw prompts, and are evolved through LLM-driven mutation and crossover. A Pareto fitness over attack success rate and peak severity preserves selection signal from near-miss attacks. A risk-indexed archive runs novelty search
    
[^118]: 基于家族DIF指导的基准重组下，接近持平的大语言模型排名是否稳健？

    Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?

    [https://arxiv.org/abs/2609.00482](https://arxiv.org/abs/2609.00482)

    该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。

    

    排行榜上的微小差距常被解读为某个语言模型优于另一个的证据，但其结论方向可能取决于包含哪些基准题目。我们利用五个基准的题目级响应数据以及一种无家族标签的谱近似多维项目反应理论（MIRT）来检验这一点。在所有者不相交的折中划分下，一半所有者数据用于识别跨模型家族具有低残差差异项目功能（低DIF）的题目；由此得到的固定且按来源和难度平衡的权重用于对另一半数据中的模型进行评分，同时使用等长的匹配随机子测试来控制一般性的子测试变异。全基准排名与低DIF排名保持强相关（τb=.900-.948）。然而，在五个基准中的四个里，最初相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，比匹配随机子测试的中位数高出16.9-28.6个百分点（均为p=.001）。第五个基准[摘要截断]

    arXiv:2609.00482v1 Announce Type: new  Abstract: Small leaderboard gaps are often interpreted as evidence that one language model is better than another, but their sign may depend on which benchmark items are included. We test this using item-level responses from five benchmarks and a family-label-free spectral approximation to multidimensional item-response theory (MIRT). In owner-disjoint folds, one owner half identifies items with low residual differential item functioning across model families (low-DIF); the resulting frozen, source- and easiness-balanced weights score models in the other half, while equally short matched-random subtests control for generic subtest variation. Full-benchmark and low-DIF rankings remain strongly correlated ($\tau_b=.900$--$.948$). Yet in four of five benchmarks, 30.9--47.1\% of cross-family pairs initially within one percentage point reverse order, exceeding their matched-random medians by 16.9--28.6 percentage points (all $p=.001$). The fifth benchm
    
[^119]: 分形维度预测角度编码数据中量子核的坍缩

    Fractal dimension predicts quantum kernel collapse in angle-encoded data

    [https://arxiv.org/abs/2609.00475](https://arxiv.org/abs/2609.00475)

    该论文提出用数据的相关性分形维度 D2 作为先验量子比特数预算，可准确预测并避免角度编码量子核的几何坍缩，使量子核在真实硬件上以最小比特数保持有效。

    

    角度编码的量子核在表格数据上，当特征映射宽度超过数据内在维度时会发生坍缩。我们提出将相关性分形维度 D2 作为一种先验的量子比特预算：编码由 FD-ASE 选取的 D2 个坐标，而不是采用 PCA-95% 宽度或全部 E 个属性。在九个数据集和状态向量模拟器（n=32）上，单层 ZZ 保真度核在 q=D2 时仍保持几何活性，而同样的核在 PCA-95% 宽度下已经坍缩。该预算依赖于映射方式：直积态映射和 IQP 映射会超出该预算，而第二层 ZZ 则会低于该预算。打包的密集角度编码和重上传编码在分形 q 下仍然存活，但当把 PCA-95% 特征堆叠到这些量子比特上时则不再存活。缩小角度带宽会使 ZZ 拐点后移，拉伸带宽则会使核更早失效。在 IBM Quantum（ibm_fez，256 次采样，n=8）上，分形宽度下的单层 ZZ 核与精确核相匹配（MAE 为 0.021）；超过该宽度后……

    arXiv:2609.00475v1 Announce Type: cross  Abstract: Angle-encoded quantum kernels on tabular data collapse when the feature map is wider than the intrinsic dimension of the data. We propose the correlation fractal dimension D2 as an a priori qubit budget: encode D2 coordinates chosen by FD-ASE instead of the PCA-95% width or all E attributes. On nine data sets and a statevector simulator (n= 32), a one-layer ZZ fidelity kernel at q=D2 stays geometrically alive while the same kernel at the PCA-95% width has already collapsed. The budget is map-dependent: product-state and IQP maps overshoot it; a second ZZ layer undershoots it. Packed dense-angle and re-uploading encodings still live at the fractal q, but not when PCA-95% features are stacked onto those qubits. Shrinking the angle bandwidth moves the ZZ knee later; stretching it kills the kernel earlier. On IBM Quantum (ibm_fez, 256 shots, n=8) the one-layer ZZ kernel at the fractal width matches the exact kernel (MAE 0.021); past that w
    
[^120]: 深度学习中的高阶结构

    Higher Structures in Deep Learning

    [https://arxiv.org/abs/2609.00472](https://arxiv.org/abs/2609.00472)

    本文阐述了高元张量运算在深度学习中的重要性，对训练后神经网络中的高元现象进行了新颖的实证研究，并提出了多层感知机的超图推广形式，同时探讨了其与进化算法的联系。

    

    我们提供了一篇关于高元张量运算对深度学习重要性的阐述性介绍。随后，我们对训练后的神经网络中的高元现象进行了新颖的实证研究，提出了多层感知机的一种超图推广形式，并探讨了其与进化算法之间的联系。最后，我们对未来研究中富有前景的方向进行了讨论。

    arXiv:2609.00472v1 Announce Type: cross  Abstract: We provide an expository introduction on the importance of higher-arity tensor operations to deep learning. Then, we conduct a novel empirical investigation of higher-arity phenomenon in trained neural networks, introduce a hypergraphical generalization of the multilayer perceptron, and explore connections to evolutionary algorithms. We conclude with a discussion of promising directions for future research.
    
[^121]: 关系型基础模型中的上下文窗口失效问题

    Context Window Failures in Relational Foundation Models

    [https://arxiv.org/abs/2609.00460](https://arxiv.org/abs/2609.00460)

    当前关系基础模型因上下文窗口限制，在面对拥有大量关联记录的实体时会严重失效，仅一个简单的时序预聚合步骤就能将R²从0.18提升至0.65，表明现有模型尚无法应对高基数现实数据。

    

    近年来，关系深度学习架构被提出作为多表关系数据的基础模型，但它们对邻域预算施加了限制，当某个实体拥有大量相关记录时会迫使行被截断。我们介绍了Animus，一个合成金融数据集，其中预测客户收入需要聚合多达数万笔交易。在原始表示上，三个近期提出的模型（RT、Griffin、RelGT）的R²均不超过0.18；而仅通过一个简单常规的时序预聚合步骤，就能将R²提升至0.65。这质疑了当前关系基础模型是否已准备好应对高基数的现实世界数据。

    arXiv:2609.00460v1 Announce Type: new  Abstract: Recent Relational Deep Learning architectures have been proposed as foundation models for multi-table relational data, yet they impose constrained neighborhood budgets that force row truncation when an entity has many related records. We introduce Animus, a synthetic financial dataset in which predicting customer income requires aggregating up to tens of thousands of transactions. On the raw representation, three recently proposed models (RT, Griffin, RelGT) achieve $R^2 \le 0.18$; a single, routine, temporal pre-aggregation step recovers $R^2$ up to $0.65$. This questions whether current relational foundation models are ready for high-cardinality real-world data.
    
[^122]: 大语言模型能否使用关系Transformer嵌入？

    Can LLMs Use Relational Transformer Embeddings?

    [https://arxiv.org/abs/2609.00457](https://arxiv.org/abs/2609.00457)

    本研究通过将冻结的关系Transformer嵌入注入Qwen3.5-4B并经过SFT与GSPO两阶段训练，发现在RelBench的多种监督模式下，这种混合融合策略并未持续优于独立的RT模型，反而经常低于随机水平且对序列化格式和标记预算高度敏感。

    

    将冻结的关系编码器嵌入作为软标记注入大语言模型（LLM）是一种概念上颇具吸引力的融合策略：编码器负责处理多表结构，LLM负责语言和推理，且无需有损的文本序列化。我们通过一个具体实验来检验这一假设：将冻结的关系Transformer（RT）的嵌入经由学习到的MLP投影和LoRA适配注入Qwen3.5-4B，训练过程先在思维链推理轨迹上进行监督微调（SFT），随后进行基于分组的强化学习（GSPO）。我们在来自RelBench的6个关系数据库上的10个二分类任务中进行评估，涵盖四种监督模式：单任务（ST）、数据集内（WD）、跨数据集（CD）和全任务（ALL）。结果显示，这种混合模型并未持续优于独立的RT模型：其表现经常低于随机水平，对序列化格式和关系标记预算高度敏感，并且……（摘要原文在此处截断）

    arXiv:2609.00457v1 Announce Type: new  Abstract: Injecting frozen relational-encoder embeddings as soft tokens into a large language model (LLM) is a conceptually appealing fusion strategy: the encoder handles multi-table structure, the LLM handles language and reasoning, and no lossy text serialization is required. We test this hypothesis concretely by injecting embeddings from a frozen Relational Transformer (RT) into Qwen3.5-4B via a learned MLP projection and LoRA adaptation, trained first with supervised fine-tuning (SFT) on chain-of-thought reasoning traces and then with group-based reinforcement learning (GSPO). We evaluate across 10 binary classification tasks on 6 relational databases from RelBench, under four supervision regimes: single-task (ST), within-dataset (WD), cross-dataset (CD), and all-task (ALL). The hybrid model does not consistently outperform standalone RT: it is frequently below random, highly sensitive to serialization format and relational-token budget, and u
    
[^123]: HBQ：面向高精度大语言模型推理的硬件效率感知分层缩放块量化

    HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference

    [https://arxiv.org/abs/2609.00450](https://arxiv.org/abs/2609.00450)

    提出硬件效率感知的分层块量化方法HBQ，突破块量化中块大小与精度之间的固有权衡，在大块设计下同时实现高硬件效率与精确的大语言模型推理。

    

    块量化是实现大语言模型（LLM）高效部署的一种有前景的方法，能够在精度可控下降的前提下实现低精度计算。与标量仅权重量化（WoQ）相比，块量化同时对权重和激活进行量化，具有更高的硬件效率，并能在统一数据通路上实现端到端推理，但其设计空间（涵盖位宽、块大小、缩放方式和数值格式）仍未得到充分探索。我们通过设计空间探索（DSE）提供了硬件与基准测试结果。我们发现，增大块大小可以通过摊销反量化和累加成本来提高硬件效率，但会降低精度。这种权衡限制了传统块量化方法的应用。受此洞察启发，我们提出分层块量化（HBQ）。与先前采用小块以及传统2的幂（PoT）或基于整数的缩放的方法不同，HBQ使用大块来……（摘要原文在此截断）

    arXiv:2609.00450v1 Announce Type: cross  Abstract: Block Quantization (BQ) is a promising approach for efficient deployment of large language models (LLMs), enabling low-precision computation with controlled accuracy degradation. Compared to scalar weight-only quantization (WoQ), BQ quantizes both weight and activation, offering higher hardware efficiency and end-to-end inference on a unified datapath, but its design space, spanning bit-width, block size, scaling, and numeric formats, remains underexplored.   We provide hardware/benchmark results through design space exploration (DSE). We find that increasing block size improves hardware efficiency by amortizing dequantization and accumulation costs, but degrades accuracy. This trade-off limits conventional BQ methods.   Motivated by this insight, we propose Hierarchical Block Quantization (HBQ). Unlike prior methods [1], [2], which use small blocks and conventional Power-of-Two (PoT) or integer-based scaling, HBQ uses large blocks to 
    
[^124]: CRAD：面向去中心化异构联邦学习的类别级可靠性感知蒸馏

    CRAD: Class-wise Reliability-Aware Distillation for Decentralized Heterogeneous Federated Learning

    [https://arxiv.org/abs/2609.00446](https://arxiv.org/abs/2609.00446)

    该论文提出CRAD框架，通过类别级可靠性感知的方式组合同伴教师的软预测，在去中心化联邦学习中同时解决模型架构异构与非独立同分布数据两大难题，且无需中央服务器或共享原始数据。

    

    传统的联邦学习（FL）依赖于参数平均，这要求客户端具备双重同质性：既要求模型架构完全相同，又在非独立同分布数据下性能退化。而现实世界的部署通常会同时打破这两个假设。我们通过构建一个去中心化的知识蒸馏框架来规避这两个问题：在该框架中，每个客户端在自己的本地数据上评估其同伴的模型快照，并从得到的软预测中进行蒸馏。由于知识通过共享的类别后验概率进行传递，客户端可以自由使用不同的模型架构；同时，由于每个教师模型都在学生自己的设备上进行评估，原始数据永远不会离开客户端，且无需中央服务器或公共数据集。在此设定下，我们识别并解决了一个尚未被充分研究的问题：如何组合多个同伴教师模型的预测。现有方法（如均匀平均）忽略了知识可靠性在教师之间以及不同类别之间的差异。我们（提出……原文在此处截断）

    arXiv:2609.00446v1 Announce Type: new  Abstract: Conventional federated learning (FL) relies on parameter averaging, which forces clients to be doubly homogeneous: it demands an identical architecture and degrades under non-IID data. Real-world deployments usually break both assumptions. We sidestep both by building a decentralized knowledge distillation framework in which each client evaluates its peers' model snapshots on its own local data and distills from the resulting soft predictions. Because knowledge is transferred through the shared class posterior, clients are free to run different architectures; and because every teacher is evaluated on the student's own device, raw data never leaves the client, with no central server or public dataset required. Within this setting, we identify and address an under-examined problem: how to combine the peer teacher predictions. Existing methods, like uniform averaging, ignore how knowledge reliability varies across teachers and classes. We p
    
[^125]: 能力门控语言模型：安全性可组合，实用性不可组合

    Capability-Gated Language Models: Security Composes, Utility Does Not

    [https://arxiv.org/abs/2609.00445](https://arxiv.org/abs/2609.00445)

    提出在单一模型权重内部实现按主体的“能力门控部署”，配置构成格结构，并证明安全限制在交运算下可组合累积（安全性随限制叠加而增强），而实用性不具备这种组合性。

    

    已部署的语言模型安全防护措施（安全微调、过滤、遗忘学习）仅在模型权重之外按主体区分：过滤器被重新配置、访问层级成倍增加、模型制品被重新发布；而在同一组权重内部，所有请求面对的都是相同的模型配置。这促使我们定义“能力门控部署”：在一组权重内部实现按主体的访问控制，其配置构成一个格——交运算累积某个主体的限制，并运算汇集一个联盟的权限范围。我们通过在现有嵌套分解机制上实施稀疏秩门控来实例化该方法，利用单次遍归因引导配置搜索，并仅从预注册的保留集划分中一次性读取全部结果。安全性可组合：在单调引出假设下，交运算处的组合性可被证明，而我们在逐点意义上证伪了该假设。在两个模型谱系中，保留集上交运算的中位数效应加深了抑制；经多重校正后唯一存活的效应进一步强化了这种抑制。实用性却不可组合：……（原文摘要在此处截断）

    arXiv:2609.00445v1 Announce Type: cross  Abstract: Deployed language model safeguards (safety fine-tuning, filtering, unlearning) vary by principal only outside the model weights: filters are reconfigured, tiers are multiplied, and artefacts are reissued; inside one set of weights every request meets the same model configuration. This motivates us to define capability-gated deployment: per-principal access control inside one set of weights, whose configurations form a lattice - meets accumulate a principal's restrictions and joins pool a coalition's reach. We instantiate it by sparse rank gating over an existing nested-factorisation mechanism, guide profile search with one-pass attribution, and read every result once from a pre-registered held-out split. Security composes: provably at meets under a monotone-elicitation assumption we falsify pointwise. In two lineages the median held-out meet deepens suppression; the one effect surviving correction strengthens it. Utility does not: indi
    
[^126]: 群体自适应裁剪策略优化

    Group Adaptive Clipping Policy Optimization

    [https://arxiv.org/abs/2609.00444](https://arxiv.org/abs/2609.00444)

    该论文提出 GAPO，一种基于反向 KL 信任域视角对 GRPO 的即插即用改进，通过根据 rollout 优势自适应调整裁剪边界，让具有更强学习信号的稀有正确 rollout 获得更大的更新空间，从而解决固定裁剪对探索性 rollout 的过度抑制问题。

    

    在基于可验证奖励的强化学习（RLVR）中，群体相对策略优化（GRPO）通常对所有 rollout 使用固定的重要性采样（IS）比率裁剪边界。我们发现了这一方法的一个关键局限：较难问题上稀有的正确 rollout 和较简单问题上充裕的正确 rollout 会以相近的比率被裁剪，尽管它们贡献的学习信号截然不同。群体成功率较低的 rollout 表现出更大的 IS 比率，并为探索和解决新问题携带更强的梯度信号，然而它们却被固定裁剪不成比例地抑制。为了解决这一问题，我们提出了群体自适应裁剪策略优化（GAPO），这是对 GRPO 方法的一种即插即用式修改，能够根据 rollout 的优势自适应地调整裁剪边界。GAPO 的设计源于反向 KL 信任域的视角，该视角表明具有更强学习信号的 rollout 应获得相应更大的更新空间……

    arXiv:2609.00444v1 Announce Type: cross  Abstract: Group relative policy optimization for reinforcement learning with verifiable rewards (RLVR) typically uses a fixed importance-sampling (IS) ratio clipping boundary across all rollouts. We identify a key limitation: rare correct rollouts on harder problems and abundant correct rollouts on easier problems are clipped at comparable rates, despite contributing very different learning signals. Rollouts with low group success exhibit larger IS ratios and carry stronger gradient signal for exploration and solving new problems, yet are disproportionately suppressed by fixed clipping.   To address this, we propose Group Adaptive Clipping Policy Optimization (GAPO), a plug-in modification to GRPO methods that adapts the clipping boundary to the rollout advantage. GAPO is motivated by a reverse-KL trust-region perspective, which suggests that rollouts with larger learning signal should receive proportionally greater update headroom. GAPO require
    
[^127]: 生理信息可靠性：面向心血管传感的跨层自适应资源分配

    Physiological Information Reliability: Cross-Layer Adaptive Resource Allocation for Cardiovascular Sensing

    [https://arxiv.org/abs/2609.00435](https://arxiv.org/abs/2609.00435)

    该论文提出生理信息可靠性（PIR）跨层框架，通过上下文老虎机将生理信息价值与无线、能量和计算状态联合建模，自适应调整心血管传感的感知与通信决策，在满足医疗延迟约束的前提下实现有前景的低能耗工作点和具有竞争力的生理估计性能。

    

    心血管传感系统必须在信号退化、无线丢包、能量受限和边缘计算延迟等条件下保持临床有用的信息。我们提出了生理信息可靠性（PIR），这是一个跨层框架，将生理信息价值与无线、能量和计算状态联合表征，并利用上下文老虎机自适应地调整感知与通信决策。我们在突发擦除条件下，将多模态心电（ECG）/光电容积脉搏波（PPG）信号质量估计与生理信息价值以及自适应网络编码层相结合。在多种子受控实验中，PIR-LinUCB 展现出一个有前景的低能耗工作点，同时满足医疗延迟约束，并且在生理估计性能上相对于固定策略和启发式策略具有竞争力。我们分析了由此产生的精度-能耗-延迟权衡，并指出了代理生理信息价值估计的局限性。

    arXiv:2609.00435v1 Announce Type: cross  Abstract: Cardiovascular sensing systems must preserve clinically useful information despite signal degradation, wireless losses, energy constraints, and edge-computation latency. We introduce Physiological Information Reliability (PIR), a cross-layer framework that represents physiological information value jointly with wireless, energy, and computation states and uses a contextual bandit to adapt sensing and communication decisions. We integrate multimodal ECG/PPG signal-quality estimation with physiological information value and an adaptive network-coding layer under burst-erasure conditions. Across controlled multiseed experiments, PIR-LinUCB demonstrates a promising low-energy operating point while maintaining medical latency constraints and competitive physiological estimation performance relative to fixed and heuristic policies. We analyze the resulting accuracy-energy-latency trade-offs and identify limitations of proxy PIV estimation an
    
[^128]: SAGE：面向任务型对话智能体的状态接地、弃权感知评估

    SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents

    [https://arxiv.org/abs/2609.00434](https://arxiv.org/abs/2609.00434)

    SAGE提出将工作流规范编译为原子准则，通过会弃权而非猜测的符号与编码器/NLI验证器级联来评估任务型对话智能体每轮的状态推进，其中SAGE-Core可在零付费LLM成本下判定81-91%的准则。

    

    评估任务型对话智能体不仅要判断回复是否读起来流畅，还要判断每一轮对话是否正确推进了底层工作流状态——传统整体式LLM评判器往往忽略这一区别，因为它们将可用上下文作为单一整体进行评估，且每轮都需要一次或多次完整模型调用。我们提出SAGE（状态接地、弃权感知评估），该方法将工作流规范和逐轮状态差异编译为原子化的、基于模式的准则，并将每条准则通过符号验证器与编码器/NLI验证器构成的级联进行路由，这些验证器在不确定时选择弃权而非猜测，最终将各准则的判定聚合为带有证据轨迹的轮级决策。其推荐的运行配置SAGE-Core仅依靠编译器、符号规则和设备端编码器即可判定81-91%的准则，且零付费LLM成本；SAGE-LLM则针对开放类准则增加了可选的聚焦LLM回退机制。在跨越四个切片的……（摘要原文在此处截断）

    arXiv:2609.00434v1 Announce Type: new  Abstract: Evaluating task-oriented dialogue agents requires judging not merely whether a reply reads well but whether each turn advances the underlying workflow state correctly--a distinction conventional holistic LLM judges can miss because they evaluate the available context as a single unit and require one or more full-model calls per turn. We propose SAGE (State-Grounded Abstention-Aware Evaluation), which compiles a workflow specification and per-turn state diff into atomic, schema-grounded criteria and routes each through a cascade of symbolic and encoder/NLI verifiers that abstain rather than guess, aggregating criterion verdicts into a turn-level decision with an evidence trace. Its recommended operating point, SAGE-Core, decides 81--91% of criteria with only the compiler, symbolic rules, and on-device encoders--at zero paid LLM cost--while SAGE-LLM adds an optional focused-LLM fallback for open-class criteria. Across four slices spanning 
    
[^129]: 使用神经网络加速系外行星大气的化学动力学计算

    Accelerating Chemical Kinetics for Exoplanet Atmospheres using Neural Networks

    [https://arxiv.org/abs/2609.00428](https://arxiv.org/abs/2609.00428)

    本文提出一种基于神经网络残差流映射架构的系外行星大气化学动力学代理求解器，其推理速度达微秒级、比经典求解器快数个数量级，可在保持精度的前提下大幅加速行星大气化学模拟。

    

    观测结果日益揭示了塑造系外行星大气的辐射、化学与动力学过程之间的耦合机制。解读这些大气需要能够捕捉这种复杂性的模型。然而，多维模型始终受到计算成本的根本性限制，回答关键问题需要以经典方法无法企及的速度来模拟主导物理机制。因此，模型往往依赖简化近似，例如平衡化学，即使这些假设会遗漏重要的效应。目前迫切需要快速且准确的化学动力学求解器来模拟行星大气。本文提出了一种采用残差流映射架构的机器学习局部盒式化学动力学求解器，用于系外行星大气。我们证明该代理模型比经典求解器快若干个数量级，实现了微秒级量级的推理速度，同时（摘要在此处被截断）

    arXiv:2609.00428v1 Announce Type: cross  Abstract: Observations increasingly reveal the coupled radiative, chemical, and dynamical processes that shape exoplanet atmospheres. Interpreting these atmospheres requires models that can capture this complexity. However, multidimensional models remain fundamentally limited by computational cost, and answering key questions requires simulating the governing physical mechanisms at speeds classical methods cannot achieve. As a result, models often rely on simplifying approximations, such as equilibrium chemistry, even when those assumptions miss important effects. There is a pressing need for fast and accurate chemical kinetics solvers to model planetary atmospheres. Here we present a machine learning local-box chemical kinetics solver for exoplanet atmospheres using a residual flow-map architecture. We demonstrate that this surrogate model is several orders of magnitude faster than a classical solver, achieving microsecond-scale inference while
    
[^130]: 时间相关性如何塑造线性循环神经网络中的记忆

    How Temporal Correlations Shape Memory in Linear Recurrent Neural Networks

    [https://arxiv.org/abs/2609.00420](https://arxiv.org/abs/2609.00420)

    本文精确求解了输入存在时间相关性时线性循环神经网络的学习动力学，发现相关性的全部效应集中于“保留过去”的代价：它使学习过程呈现记忆建立、过冲再部分遗忘的轨迹，并使最终网络保留更少过去信息，同时记忆在一个仅由相邻输入相似度决定的阈值处关闭，不受序列长度和更长程相关性影响。

    

    线性循环神经网络（LRNN）是研究网络在训练过程中能积累多少记忆的简单模型。对于不相关的输入，早期工作发现训练本身会使网络在“保留过去”与“只对当下作出反应”之间取得平衡。真实序列是存在相关性的，本文对相关输入下的学习动力学进行了精确求解。在该解中，保留过去是有代价的，而相关性的全部效应都集中体现在这个代价上：当输入不相关时，该代价退化为早期研究中的形式；当输入正相关时，该代价随之增大。由此得到三点发现：（1）相关性重塑的是学习的整个进程，而不仅仅是最终结果：记忆先建立、再过冲、随后被部分移除，最终收敛的网络保留较少的过去信息；（2）记忆在一个阈值处被关闭，该阈值仅由一个数字决定——即每个输入与前一个输入的相似程度，而序列长度以及更长程的相关性都无法改变这一阈值。

    arXiv:2609.00420v1 Announce Type: new  Abstract: The linear recurrent neural network (LRNN) is a simple model for studying how much memory a network builds up as it trains. For uncorrelated inputs, earlier work found that training itself settles the network between keeping the past and reacting only to the present. Real sequences are correlated, and we solve the learning dynamics exactly for correlated inputs. In the solution, keeping the past carries a cost. The whole effect of correlation lands on that cost. This cost reduces to the earlier one when inputs are uncorrelated and grows once they are positively correlated. Three findings follow. (1) Correlation reshapes the course of learning, not only its end. Memory builds, overshoots, and is partly removed, and the settled network keeps less of the past. (2) Memory switches off at a threshold set by one number, how much each input resembles the one just before it. Neither sequence length nor longer-range correlation moves this thresho
    
[^131]: 一种用于健康错误信息检测与传播的多分支特征融合方法

    A Multi-Branch Feature Fusion Approach for Health Misinformation Detection and Propagation

    [https://arxiv.org/abs/2609.00403](https://arxiv.org/abs/2609.00403)

    本文提出一种基于详尽可能性模型和计划行为理论的多分支特征融合框架，融合Transformer语义、修辞线索、立场表示及心理动机特征来检测健康错误信息，并引入可解释的认知传播评分（CPS）辅助传播风险推理，在三个基准数据集上ROC-AUC最高达0.9。

    

    本文提出了一种多分支融合框架，用于检测和刻画在线社交网络（OSN）中健康错误信息的传播。该模型以详尽可能性模型（ELM）和计划行为理论（TPB）为理论基础，在统一的多任务架构中，将基于Transformer的语义信息与修辞线索、立场表示以及心理动机代理特征相融合。除了二分类任务外，我们还引入了认知传播评分（CPS），这是一种可解释的事后辅助评分，由心理动机驱动的、源自文本的线索（捕捉论点复杂性、情绪强度以及基于内容的病毒式传播潜力）计算得出，用于在互动真值不完整或不可用时支持传播风险的推理。在三个基准数据集（Constraint、COVID-19_FNIR 和 Monkeypox）上的实验显示出强大的分类性能，ROC-AUC最高可达0.9。

    arXiv:2609.00403v1 Announce Type: new  Abstract: This paper presents a multi-branch fusion framework for detecting and characterising the propagation of health misinformation in online social networks (OSNs). Grounded in the Elaboration Likelihood Model (ELM) and the Theory of Planned Behaviour (TPB), the model fuses transformer-based semantics with rhetorical cues, stance representations, and psychologically motivated proxies in a unified multi-task architecture. In addition to binary classification, we introduce the Cognitive Propagation Score (CPS), an interpretable post-hoc auxiliary score computed from psychologically motivated, text-derived cues capturing argument complexity, emotional intensity, and content-derived virality potential, to support diffusion-risk reasoning when engagement ground truth is incomplete or unavailable. Experiments on three benchmark datasets, Constraint, COVID--19\_FNIR, and Monkeypox, show strong classification performance, achieving ROC--AUC up to 0.9
    
[^132]: 用于算子学习的神经均值与核修正

    Neural means and kernel corrections for operator learning

    [https://arxiv.org/abs/2609.00389](https://arxiv.org/abs/2609.00389)

    该论文提出将神经网络均值与Matérn核回归修正相结合的方法，在结构力学和OCO-2辐射传输两个基准问题上达到了或超越了已发表最佳结果，并从理论上证明和量化了核修正之所以有效的机制。

    

    我们将神经网络均值与其残差以及学习到的特征的精确Matérn核回归相结合，并在两个有公开基线的公共仿真问题上评估了这种组合：de Hoop等人的结构力学基准和Lamminpää等人的OCO-2辐射传输仿真器。在结构力学问题上，该组合达到了4.55%的测试误差，与已发表的最佳架构相当；在低数据量情形下达到5.38%，优于已发表的6.49%。在OCO-2问题上，该组合在该问题自身的测试点上改进了已发表的高斯过程仿真器，在三个光谱波段中的两个上完全超越；同一个核在原始状态上落后于网络十倍，却在网络的特征上反超网络，并且我们测量了原因（在固定有效维度的前提下，目标在原生空间中的平方范数下降了约四十倍）并证明了这一机制。在两个方法家族打平的地方，我们所评估的每种架构的残差……

    arXiv:2609.00389v1 Announce Type: new  Abstract: We combine neural network means with exact Mat\'ern kernel regressions of their residuals and of their learned features, and evaluate the pairing on two public emulation problems with published baselines: the structural-mechanics benchmark of de Hoop et al. and the OCO-2 radiative-transfer emulator of Lamminp\"a\"a et al. On structural mechanics the combination reaches 4.55% test error, matching the best published architecture, and 5.38% against a published 6.49% in the low-data regime. On OCO-2 it improves on the published Gaussian-process emulator on that problem's own test points, outright on two of the three spectral bands; the same kernel that trails the network tenfold on the raw state overtakes it on the network's features, and we measure why (the target's squared native-space norm drops about fortyfold at fixed effective dimension) and prove the mechanism. Where the two families tie instead, the residuals of every architecture we
    
[^133]: 面向自主超车的风险感知决策：基于世界模型的专家混合框架

    Risk-Aware Decision-Making for Autonomous Overtaking: A World Model-Based Mixture-of-Experts Framework

    [https://arxiv.org/abs/2609.00385](https://arxiv.org/abs/2609.00385)

    本文提出基于世界模型的风险感知专家混合框架，利用学习到的潜在动力学模型进行并行多步推演，将安全评估从动作层面提升到轨迹层面的累积风险水平，并通过分层门控机制动态协调专家以适应不同交互强度，从而提升自主超车决策的长期安全性。

    

    自主高速公路超车需要具备前瞻性的决策能力，以应对复杂的交互行为、随机的交通演化以及时间维度上的风险累积。然而，标准的安全强化学习方法通常依赖于基于价值的隐式风险估计，而非显式的动力学建模，因而难以准确捕捉多步时域内复杂的风险传播过程。这一局限常常导致智能体表现出局部安全、但长期来看会引发较大潜在风险的行为。为解决这一问题，本文提出了一种基于世界模型的风险感知专家混合框架。首先，通过学习得到的潜在动力学模型支持并行的多步推演，借助累积风险评估将安全性评估从动作层面提升到轨迹层面。其次，为了增强在不同交互强度下的鲁棒性，一种分层门控机制能够动态协调各专家……

    arXiv:2609.00385v1 Announce Type: cross  Abstract: Autonomous highway overtaking demands foresighted decision-making to handle complex interactions, stochastic traffic evolution, and temporal risk accumulation. However, standard safe reinforcement learning approaches typically rely on implicit value-based risk estimations rather than explicit dynamics modeling, thereby struggling to accurately capture complex risk propagation over multi-step horizons. This limitation frequently results in behaviors that are locally safe but induce substantial latent risks in the long term. To address this, a World Model-based Risk-aware Mixture-of-Experts (WM-RMoE) framework is proposed. First, a learned latent dynamics model facilitates parallel multi-step rollouts, elevating safety assessment from the action level to the trajectory level via cumulative risk evaluation. Second, to enhance robustness under varying interaction intensities, a hierarchical gating mechanism dynamically coordinates experts 
    
[^134]: 无梯度自适应：仿射统计传输及其证书能告诉你什么

    Adapting Without Gradients: Affine Statistics Transport and What Its Certificate Can Tell You

    [https://arxiv.org/abs/2609.00374](https://arxiv.org/abs/2609.00374)

    提出CASTER，一种无需梯度的测试时自适应方法，通过在判别子空间中存储源类别统计并估计仿射变换来解析地传输源类别分布，使冻结模型无需反向传播即可适应目标数据，同时以约18倍更少的状态存储优于k-NN基线。

    

    测试时自适应通常假设模型参数可以在推理阶段进行更新。这一假设对于仅推理加速器、冻结或第三方模型以及内存受限的部署环境具有很强的限制性，而且基于BatchNorm的标准TTA配置在不包含BatchNorm的架构上也可能失效。我们研究了在所学模型必须保持冻结情况下的自适应问题。我们提出了CASTER，这是一种无梯度方法，它在判别子空间中存储源类别统计信息，从目标批次矩中估计一个类别共享的仿射变换，并在分类之前以解析方式传输源类别分布。CASTER不需要反向传播、优化器状态或存储源特征库。在四个骨干网络和七个数据集的实验中，它在28个骨干-数据集组合中的27个上优于基于相同冻结特征的k-NN方法，同时状态存储量的中位数减少18倍。仿射传输并非……（摘要原文在此处截断）

    arXiv:2609.00374v1 Announce Type: cross  Abstract: Test-time adaptation (TTA) typically assumes that model parameters can be updated at inference time. This assumption is restrictive for inference-only accelerators, frozen or third-party models, and memory-constrained deployments, and standard BatchNorm-based TTA configurations may also become inactive on architectures without BatchNorm. We study adaptation when the learned model must remain frozen. We introduce CASTER, a gradient-free method that stores source class statistics in a discriminative subspace, estimates a class-shared affine transformation from target-batch moments, and analytically transports the source class distributions before classification. CASTER requires no backward pass, optimizer state, or stored source feature bank. Across four backbones and seven datasets, it outperforms k-NN on identical frozen features in 27 of 28 backbone-dataset settings while retaining a median of 18x less state. Affine transport is not a
    
[^135]: 面向数据工程的神经符号方法：无需微调实现长上下文Token缩减

    Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning

    [https://arxiv.org/abs/2609.00367](https://arxiv.org/abs/2609.00367)

    本文提出一种即插即用的神经符号层，无需任何微调或RLHF即可在Text-to-SQL等数据工程任务上平均提升85%的准确率，同时缓解Transformer长上下文的计算资源瓶颈。

    

    大型语言模型正越来越多地被部署用于复杂的数据工程任务，例如从自然语言生成结构化查询（Text-to-SQL）以及自动化复杂的电子表格操作。然而，要最大化其效用，既需要更高的免微调准确率，也需要解决Transformer架构固有的二次方（O(n²)）时间复杂度所带来的计算瓶颈。本文提出了一种新颖的即插即用神经符号层，旨在无缝集成到现有的LLM骨干网络中，增强逻辑推理能力并缓解长上下文的资源消耗。在推理方面，该层能够立即且显著地提升性能，在包括BIRD-CRITIC和LiveSQLBench在内的严格基准测试中实现了平均85%的准确率提升，关键是这些提升无需任何任务特定的微调或RLHF。同时，我们将该方法重新应用于解决长上下文问题……

    arXiv:2609.00367v1 Announce Type: cross  Abstract: Large Language Models are increasingly deployed for sophisticated data engineering tasks such as generating structured queries from natural language, Text-to-SQL, and automating complex spreadsheet operations. However, maximizing their utility demands both higher finetuning-free accuracy and solutions to the computational bottleneck imposed by the Transformer architectures inherent quadratic (On2) time complexity. This paper introduces a novel drop-in neurosymbolic layer designed to seamlessly integrate into existing LLM backbones enhancing logical reasoning and mitigating long-context resource consumption. On the reasoning front, the layer immediately and significantly improves performance yielding an average accuracy increase of 85% across rigorous benchmarks including BIRD-CRITIC and LiveSQLBench, critically achieving these gains without any task specific finetuning or RLHF. Concurrently, we repurpose this approach to address the se
    
[^136]: 反事实脆弱性证书：揭示结构化证据失效下的高置信度脆弱性

    Counterfactual Fragility Certificates: Exposing High-Confidence Brittleness under Structured Evidence Failure

    [https://arxiv.org/abs/2609.00366](https://arxiv.org/abs/2609.00366)

    提出反事实脆弱性证书（CFC），一种模型无关的协议级审计对象，将每个预测映射为由贪婪翻转预算、边际崩塌面积等指标刻画的有序证据失效轨迹，从而揭示表格决策系统中高置信度预测在结构化证据失效下的脆弱性。

    

    高测试准确率和良好的整体校准并不能表明某个个体预测是否在结构上得到了其证据的支持。在表格化决策系统中，当某个特征族变得不可用、延迟、充满噪声、过时或低可信度，而模型却仍然保持高置信度时，故障往往会发生。现有的校准、不确定性、选择性预测、解释和扰动方法提供的只是标量分数或归因图，而不是一个可重新计算的审计对象来回答这样的问题：在声明的证据失效协议下，什么样的轨迹会使该预测失去支持？我们提出了反事实脆弱性证书（Counterfactual Fragility Certificates, CFC），这是一种模型无关的协议级审计证书——而非形式化的鲁棒性证书——它将每个预测映射到一条有序的证据失效轨迹中，并通过贪婪翻转预算、归一化边际崩塌面积、退化阈值和脆弱性支配分数来概括该轨迹。在七个表格基准数据集上……（摘要在此处截断）

    arXiv:2609.00366v1 Announce Type: cross  Abstract: High test accuracy and good aggregate calibration do not show whether an individual prediction is structurally supported by its evidence. In tabular decision systems, failures often occur when a feature family becomes unavailable, delayed, noisy, stale, or low-trust while the model remains highly confident. Existing calibration, uncertainty, selective-prediction, explanation, and perturbation methods provide scalar scores or attribution maps, but not a recomputable audit object answering: under a declared evidence-failure protocol, what trajectory makes this prediction lose support? We introduce Counterfactual Fragility Certificates (CFC), a model-agnostic protocol-level audit certificate-not a formal robustness certificate-that maps each prediction into an ordered evidence-failure trajectory summarized by greedy flip budget, normalized margin-collapse area, degradation thresholds, and fragility dominance score. Across seven tabular be
    
[^137]: Dr. Claw：一个面向氛围式研究的AI科学家工作区

    Dr. Claw: An AI Scientist Workspace for Vibe Research

    [https://arxiv.org/abs/2609.00365](https://arxiv.org/abs/2609.00365)

    Dr. Claw 是一个开源的AI科学家工作区，通过持久化状态对象、可复用技能库和多执行器协调，将现有命令行编码代理封装为可控、可审计的人机协同工作流，把科研中的规划、执行与写作整合为一个可追踪、可恢复的闭环。

    

    命令行编码代理（如 Claude Code、Gemini CLI）已经能够读写文件并维持长会话，然而端到端的科研工作仍然碎片化地分散在聊天工具、IDE、终端和写作环境之间，且那些使研究可审计的决策很少被保存下来。我们提出了 Dr. Claw，一个开源工作区，它将现有的编码代理执行器封装在一个可控且可审计的人机协同工作流中，而非引入另一个自主智能体。持久化的状态对象、可复用的技能库以及多执行器协调机制将人类决策与AI执行联系起来，使规划、执行和写作整合为一个可追踪、可恢复的闭环。我们通过一个交互式三视图场景和一次故障恢复演示来展示 Dr. Claw，并将其与共享同一后端执行器的裸命令行代理进行对比评估，因此该对比考察的是整个编排层（任务图、状态对象等）。

    arXiv:2609.00365v1 Announce Type: new  Abstract: Command-line coding agents (e.g., Claude Code, Gemini CLI) can already read and write files and sustain long sessions, yet end-to-end research still fragments across chat tools, IDEs, terminals, and writing environments, and the decisions that make it auditable are rarely preserved. We present Dr. Claw, an open-source workspace that wraps existing coding-agent executors in a controllable and auditable human-in-the-loop workflow rather than introducing another autonomous agent. Persistent state objects, a reusable skill library, and multi-executor coordination link human decisions to AI execution, turning planning, execution, and writing into one traceable, recoverable loop. We demonstrate Dr. Claw through an interactive three-view scenario and a failure-recovery walkthrough, and evaluate it against a bare command-line agent sharing the same backend executor, so the comparison contrasts the whole orchestration layer (task graph, state obj
    
[^138]: 跨GPU核函数的确定性大语言模型推理：二的幂次INT8量化缩放因子与基于容差的符合性测试的局限

    Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance

    [https://arxiv.org/abs/2609.00363](https://arxiv.org/abs/2609.00363)

    本文通过对INT8量化GEMM流水线系统性注入九种故障，证明了基于容差的符合性测试在构造上无法检测仅使输出偏移至多一个bfloat16间距的尾部计算故障，而采用二的幂次量化缩放因子是保障跨GPU核函数确定性推理的关键途径之一。

    

    针对量化GEMM核函数的符合性测试套件所检验的，是两个实现是否在容差范围内保持一致。本文测量了此类测试套件究竟能够检测到什么。研究者在Qwen3-1.7B模型的8,232个层-故障-运行状态单元格上，向参考INT8推理流水线注入了九种故障，结果发现：五种尾部计算故障——缩放因子精度、双重舍入、乘法顺序、输出截断以及融合排序——中的每一种，在5,880个单元格中最多只会使输出偏移一个bfloat16的最小间距，且只要产生影响就恰好是一个间距。因此，将容差设为一个间距的测试在构造上对这一整类故障是盲目的：五种故障中有四种不被套件中的任何检查所发现，第五种也仅在使用二的幂次缩放因子时才会被检测到。与此相对，违反累加器精确性前提条件的故障、或破坏操作数共享的故障，则无一例外地被检测出来，而空故障从不触发。由此可以得出，这种形式的基于容差的测试套件所能确立的结论，比人们预期的要狭窄得多。

    arXiv:2609.00363v1 Announce Type: new  Abstract: Conformance suites for quantized GEMM kernels ask whether two implementations agree within a tolerance. We measure what such a suite can detect. Injecting nine faults into a reference INT8 pipeline over 8,232 layer--fault--regime cells of Qwen3-1.7B, we find that every one of five epilogue faults -- scale precision, double rounding, multiplication order, output truncation, fused ordering -- moves the output by at most a single bfloat16 spacing, and by exactly one whenever it moves it at all, across 5,880 cells. A tolerance of one spacing is therefore blind to the entire class by construction: four of the five faults are detected by no check in the suite, and the fifth only under power-of-two scales. Faults that violate the accumulator's exactness preconditions, or that break operand sharing, are detected without exception, and a null fault never fires. What a tolerance-based suite of this shape establishes is therefore narrower than inte
    
[^139]: 一种用于量子联邦学习的稳定聚合方法

    A Stable Aggregation Method for Quantum Federated Learning

    [https://arxiv.org/abs/2609.00356](https://arxiv.org/abs/2609.00356)

    该论文提出了一种融合QoS感知客户端加权、循环参数聚合和有界中点更新控制的自洽中点聚合方法，显著提升了量子联邦学习在异构数据与量子硬件噪声等挑战下的稳定性与准确率。

    

    量子联邦学习（QFL）使客户端能够在不共享私有数据的情况下训练量子神经网络（QNN）模型。我们发现，在异构数据、不可靠通信、可变保真度、延迟以及量子硬件噪声等条件下，QFL中的聚合过程是不稳定的。此外，QFL还面临着不小的挑战，因为许多QNN参数是周期性角度，而欧几里得平均方法往往无法捕捉其固有的动态特性。我们开发了一种新颖的自洽中点聚合方法，用于稳定的QFL设计与实现。该方法结合了服务质量（QoS）感知的客户端加权、循环参数聚合以及基于有界中点的更新控制。我们通过多次角度测试和IBM真实量子计算机实验验证了所提方法。在医疗和金融数据集上的大量评估与实验表明，该方法具有更高的稳定性、更低的波动性以及有竞争力的准确率。

    arXiv:2609.00356v1 Announce Type: new  Abstract: Quantum federated learning (QFL) enables clients to train quantum neural network (QNN) models without sharing private data. We find that aggregation in QFL is unstable under heterogeneous data, unreliable communication, variable fidelity, latency, and quantum hardware noise. Moreover, QFL is non-trivially challenging because several QNN parameters are periodic angles, where Euclidean averaging often fails to capture the inherent dynamics. We develop a novel self-consistent midpoint aggregation method for stable QFL design and implementation. We combine QoS-aware client weighting, circular parameter aggregation, and bounded midpoint-based update control. We perform several angular tests and IBM real Quantum machines experiments for validation confirming our approach. Extensive evaluations and experiments on medical and financial datasets show improved stability, lower volatility, and competitive accuracy.
    
[^140]: 大语言模型了解你的社区吗？审计大语言模型在社区级出行预测与结构对齐中的先验知识

    Do LLMs Know Your Neighborhood? Auditing LLM Priors for Neighborhood-Level Mobility Prediction and Structural Alignment

    [https://arxiv.org/abs/2609.00345](https://arxiv.org/abs/2609.00345)

    该论文首次系统审计了零样本大语言模型在社区级人口流动性预测中的先验知识，发现其准确率不及监督基线模型，并通过方向性对齐分析检验LLM隐含的预测变量效应与实证统计趋势的一致性。

    

    人类流动行为对城市规划、交通、公共卫生和应急响应至关重要，然而细粒度的轨迹数据往往是专有的、受限制的且涉及隐私敏感。大语言模型（LLMs）通过生成合理的流动轨迹和预测个体移动提供了一种潜在的替代方案，但其推断社区级聚合流动性的能力仍不明确。我们使用匿名化的Cuebiq数据，在四个美国大都市区的普查街区群（Census Block Group）级流动性预测任务上评估零样本LLMs，构建了点级、轨迹级和时间维度的流动性结果，并与社会人口和建成环境预测变量相配对。我们将LLM预测与监督学习基线进行比较，并引入方向性对齐分析，以检验LLM隐含的预测变量效应是否与实证的OLS回归和Jonckheere-Terpstra趋势相一致。监督模型达到了0.580的平均准确率，相比之下……

    arXiv:2609.00345v1 Announce Type: new  Abstract: Human mobility is central to urban planning, transportation, public health, and emergency response, yet fine-grained trajectory data are often proprietary, restricted, and privacy-sensitive. Large language models (LLMs) offer a potential alternative by generating plausible mobility traces and predicting individual movement, but their ability to infer aggregate neighborhood-level mobility remains unclear. We evaluate zero-shot LLMs on Census Block Group-level mobility prediction across four U.S. metropolitan areas using anonymized Cuebiq data to construct point-level, trajectory-level, and temporal mobility outcomes, paired with sociodemographic and built-environment predictors. We compare LLM predictions with supervised baselines and introduce a directional alignment analysis to test whether LLM-implied predictor effects agree with empirical OLS and Jonckheere-Terpstra trends. Supervised models achieve 0.580 average accuracy, compared wi
    
[^141]: 隐含波动率曲面生成模型的潜空间无套利几何

    Latent-Space No-Arbitrage Geometry of Generative Models for Implied Volatility Surfaces

    [https://arxiv.org/abs/2609.00332](https://arxiv.org/abs/2609.00332)

    本文在潜空间中刻画生成模型输出隐含波动率曲面的无套利约束，通过标量边际定义可容许潜集并证明其在扰动下的稳定性，同时为零边际边界构造水平集方程，且该方法适用于任何具有确定性映射的生成模型。

    

    隐含波动率曲面的生成模型必须产生满足静态无套利约束的输出。我们在潜空间中研究这些约束。对于固定的生成器，我们根据生成曲面的无套利条件为每个潜编码分配一个标量边际。具有非负边际的编码构成可容许潜集。我们建立了严格可容许编码在微小扰动下仍保持可容许的条件，并且可容许集的边界由零边际刻画。对于正则边界分量，我们构造了一个水平集方程，其局部动力学指向零边际集。该分析将生成器视为从潜变量到曲面的映射，因此不限于特定的架构，适用于变分自编码器、生成对抗网络以及其他具有确定性实现映射的生成模型。

    arXiv:2609.00332v1 Announce Type: cross  Abstract: Generative models for implied volatility surfaces must produce outputs that satisfy static no-arbitrage constraints. We study these constraints in latent space. For a fixed generator, we assign each latent code a scalar margin determined by the no-arbitrage conditions of the generated surface. The codes with nonnegative margin form the admissible latent set. We establish conditions under which strictly admissible codes remain admissible under small perturbations and the boundary of the admissible set is characterized by zero margin. For regular boundary components, we formulate a level-set equation whose local dynamics are directed toward the zero-margin set. The analysis treats the generator as a map from latent variables to surfaces and is therefore not restricted to a particular architecture. It applies to variational autoencoders, generative adversarial networks, and other generative models with a deterministic realization map. Num
    
[^142]: 真实场景下的主题匹配：来自真实世界ASR转录文本的基准测试与经验教训

    Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts

    [https://arxiv.org/abs/2609.00330](https://arxiv.org/abs/2609.00330)

    该论文构建了一个基于真实呼叫中心ASR转录文本的人工标注主题匹配基准数据集，并通过系统对比发现，配备自然语言主题描述的轻量级大语言模型匹配器在处理噪声转录文本时性能优于句子嵌入和正则表达式方法。

    

    在呼叫中心中，实时坐席辅助工具会针对多个预定义主题中的每一个，判断实时的客户话语是否与之相关，并在相关时向坐席展示辅导卡片。输入数据充满噪声且极具挑战性：即自发电话对话的ASR（自动语音识别）转录文本，这些文本可能不清晰、内容重复，且大多缺乏标点符号。为了系统地研究这一现实世界任务，我们整理了一个基于真实呼叫中心转录文本的人工标注的主题-话语判断数据集。我们比较了三种类型的匹配器：基于正则表达式的基线方法、零样本句子嵌入编码器，以及基于Gemini的大语言模型匹配器。此外，我们的基准中还研究了两种类型的主题表示方式：关键词短语和自然语言描述。我们的实证实验表明，配备自然语言描述的轻量级大语言模型匹配器，其性能显著优于嵌入模型和正则表达式模型。

    arXiv:2609.00330v1 Announce Type: cross  Abstract: In contact centers, real-time agent-assist tools determine, for each of many predefined topics, whether a live customer utterance is relevant and display a coaching card to the agent when it is. The input is noisy and challenging: ASR(Automatic Speech Recognition) transcripts of spontaneous phone conversations, which can be unclear, repetitive, and mostly lack punctuation. To systematically study this real-world task, we curate a human-annotated topic-utterance judgments dataset sourced from real call-center transcripts. We compare three types of matchers: a regex-based baseline, zero-shot sentence embedding encoders, and Gemini-based LLM matchers. In addition, two types of topic representations are studied in our benchmark:keyphrases and natural language description. Our empirical experiments highlight the superior performance of lightweight LLM matchers over embedding and regex models when equipped with natural language descriptions.
    
[^143]: 词汇规范化中的多语言诅咒

    The Curse of Multilinguality in Lexical Normalization

    [https://arxiv.org/abs/2609.00329](https://arxiv.org/abs/2609.00329)

    该研究通过固定容量字符级模型在十二种语言上的实验发现，词汇规范化存在明显的“多语言诅咒”：语言联合训练数量超过一到四种后，各语言准确率持续下降约百分之四十，且下降源于语言间对固定模型容量的竞争而非数据稀释。

    

    词汇规范化是将用户生成文本中充满的嘈杂、非标准词汇（如 tmrw、u、gr8）改写为其标准形式。由于大多数语言的标注数据稀缺，一种流行的捷径是在多种语言上同时训练单个模型。我们提出一个简单的问题：这样的模型应该用多少种语言来训练？使用一个固定容量的字符级模型和来自标准基准的十二种语言，我们将联合训练的语言数量从一种变化到十二种，并测量每种语言的准确率。我们发现了一个明显的多语言诅咒：当一种语言仅与少数其他语言（通常一到四种）联合训练时，准确率最高；随后随着更多语言的加入，准确率持续且大幅下降，当其余语言全部加入时下降约百分之四十。一个保持总训练数据量不变的对照实验使下降来得更早、幅度更大，这表明各种语言在争夺一个固定的模型容量。

    arXiv:2609.00329v1 Announce Type: cross  Abstract: Lexical normalization rewrites the noisy, non-standard words that fill user-generated text (tmrw, u, gr8) into their standard forms. Because labelled data is scarce for most languages, a popular shortcut is to train a single model on many languages at once. We ask a simple question: how many languages should such a model be trained on? Using one fixed-capacity character-level model and twelve languages from a standard benchmark, we vary the number of jointly trained languages from one to twelve and measure per-language accuracy. We find a clear curse of multilinguality: accuracy is highest when a language is trained with only a few others, often just one to four, and then falls steadily and substantially, dropping by about forty percent as the rest are piled on. A control that holds the total amount of training data constant makes the decline arrive sooner and fall further, which points to competition among the languages for one fixed-
    
[^144]: 面向AI治理的基于物理侧信道的工作负载识别

    Workload Identification with Physical Side Channels for AI Governance

    [https://arxiv.org/abs/2609.00309](https://arxiv.org/abs/2609.00309)

    本研究证明外部观察者可利用GPU功耗这一物理侧信道，在无需运营商配合的情况下以97%的准确率识别NVIDIA H200上运行的是AI训练、推理还是非AI计算，为AI治理的国际算力核查提供了可独立验证的技术手段。

    

    AI算力验证是旨在实现AI治理的国际政策中首批切实可行且易于操作的切入点之一。要判断前沿实验室或任何运营商是否遵守协议，监管机构需要辨别其算力是如何被使用的。AI算力的基本构建单元是GPU，其执行的任何活动都会留下物理痕迹。本文表明，外部观察者可以通过功耗信号识别NVIDIA H200上运行的工作负载类别。与可能被伪造或重放的片上NVML遥测数据不同，这种物理信道原则上可以在无需运营商配合的情况下被独立观测。我们以约10 MHz的采样率记录了930条时长五秒的功耗轨迹，涵盖十七个开源大语言模型系列和二十五种非AI工作负载。在该数据集上，我们以97%的准确率和0.955的宏平均F1分数，成功区分了AI训练、AI推理与非AI计算。

    arXiv:2609.00309v1 Announce Type: cross  Abstract: AI compute verification is one of the first tangible and tractable points for international policy aimed at AI governance. Determining whether frontier labs, or any operator, comply with agreements requires the regulating authority to discern how their compute is used. The elementary building block of AI compute is the GPU, and any activity it executes leaves a physical trace. Here, we show that an external observer can identify the class of the workload running on an NVIDIA H200 from its power draw. Unlike on-chip NVML telemetry, which can be spoofed or replayed, such a physical channel can in principle be observed independently of operator cooperation. We recorded $930$ five-second traces at $\sim 10$ MHz, covering seventeen open LLM families and twenty-five non-AI workloads. Over this corpus we separate training from inference and from non-AI computation with an accuracy of $97\%$ and a macro-averaged F1 score of $0.955$, evaluated 
    
[^145]: TRUST：面向乳腺癌筛查认证豁免的阈值重校准不确定性安全训练

    TRUST: Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening

    [https://arxiv.org/abs/2609.00300](https://arxiv.org/abs/2609.00300)

    提出TRUST闭环训练策略，在训练中动态重校准豁免阈值并惩罚接近豁免区域的阳性样本，在保证召回率目标的前提下显著提高了乳腺癌筛查中可安全豁免医生复核的阴性病例比例。

    

    减少对明确无癌的筛查乳腺X光片的复核，可以在不影响癌症检出的前提下降低放射科医生的工作量。我们提出了一种闭环的阈值感知训练策略，在训练过程中重新计算豁免阈值，并用其惩罚接近豁免区域的癌症阳性图像。我们在NLBS和RSNA数据集上使用五种受控训练配置对该方法进行了评估，基于被豁免病例中癌症患病率的单侧99% Clopper-Pearson上界进行病例级评估。所提出的模型在98%和95%召回率目标下均取得了最高的病例级豁免率。在NLBS上，豁免率分别达到19.74%和21.70%，而交叉熵基线未能达到任一召回率目标。在RSNA上，豁免率从7.04%提升至14.31%，以及从13.49%提升至19.69%。在外部RSNA→NLBS评估中，所提出的模型实现了12.9%的豁免率（原文此处截断）。

    arXiv:2609.00300v1 Announce Type: cross  Abstract: Reducing the review of clearly cancer-negative screening mammograms could lower radiologist workload without compromising cancer detection. We propose a closed-loop threshold-aware training strategy in which the dismissal threshold is recalculated during training and used to penalize cancer-positive images that approach the dismissal region. We evaluated the method on NLBS and RSNA using five controlled training configurations, with case-level assessment based on a one-sided 99\% Clopper--Pearson upper bound for cancer prevalence among dismissed cases. The proposed model achieved the highest case-level dismissal rates at both 98\% and 95\% recall targets. On NLBS, dismissal reached 19.74\% and 21.70\%, while the cross-entropy baseline did not meet either recall target. On RSNA, dismissal improved from 7.04\% to 14.31\% and from 13.49\% to 19.69\%. In external RSNA$\to$NLBS evaluation, the proposed model achieved dismissal rates of 12.9
    
[^146]: 面向复杂域偏微分方程的几何感知潜空间自回归生成模型

    Geometry-aware Latent Autoregressive Generative Model for PDEs in Complex Domains

    [https://arxiv.org/abs/2609.00297](https://arxiv.org/abs/2609.00297)

    提出几何感知潜空间自回归生成模型GeoLAMP，通过双编码器联合捕获全局拓扑与细尺度几何特征，并结合流匹配的因果自注意力Transformer建模时间动力学，实现复杂不规则几何结构中多物理场PDE的高效、稳定且可扩展的求解。

    

    求解多物理场偏微分方程（PDEs）仍然是科学计算中的一项重大挑战，尤其是对于对能源与化学工程至关重要的高度复杂的微米级曲折几何结构。为应对这一挑战，我们提出了一种面向PDE的几何感知潜空间自回归生成模型，用于求解高度不规则和曲折结构内的物理问题。GeoLAMP在图表示上引入了双编码器架构，以联合捕获全局拓扑和细尺度几何特征，实现了从实空间物理场到紧凑潜空间表示的有效过渡。在潜空间中，我们提出了一种结合流匹配的因果自注意力Transformer来建模时间动力学，从而实现稳定且可扩展的块状自回归预测。灵活的解码器可在任意点上重建高分辨率物理场。我们建立了三个多物理场基准数据集

    arXiv:2609.00297v1 Announce Type: cross  Abstract: Solving multiphysics partial differential equations (PDEs) remains a major challenge in scientific computing, especially for highly complex $\mu$m-scale tortuous geometries critical to energy and chemical engineering. We address this challenge by proposing a Geometry-aware Latent Autoregressive generative Model for PDEs (GeoLAMP) for solving physics within highly irregular and tortuous structures. GeoLAMP introduces a dual-encoder architecture on graph representations to jointly capture global topology and fine-scale geometric features, enabling an effective transition from real-space fields to compact latent representations. In the latent space, we propose a causal self-attention transformer with flow matching to model temporal dynamics, allowing stable and scalable block-wise autoregressive prediction. A flexible decoder reconstructs high-resolution physical fields on arbitrary points. We establish three multiphysics benchmark datase
    
[^147]: WiSDoM：基于混合专家的无线稀疏决策Transformer，用于多任务移动网络优化

    WiSDoM: Wireless Sparse Decision Transformer with Mixture-of-Experts for Multi-Task Mobile Network Optimization

    [https://arxiv.org/abs/2609.00284](https://arxiv.org/abs/2609.00284)

    该论文提出WiSDoM，一种结合混合专家机制的稀疏多任务离线强化学习框架，能够在异构6G无线环境中实现自适应多小区（CoMP）选择，从而解决传统无线资源管理难以在多任务场景下保持一致性能的问题。

    

    新兴的6G无线网络需要在多样化的部署场景中运行，其中网络拓扑、用户移动性、流量需求和无线条件的变化对传统无线资源管理（RRM）的可扩展性构成了挑战。尽管离线强化学习（RL）方法已展现出强大的决策能力，但由于优化目标相互冲突且模型专门化程度有限，学习一个在异构无线环境中均能保持一致性能的单一策略仍然十分困难。这些挑战在协同多点（CoMP）传输中尤为突出，因为选择最优服务小区组合需要在不断变化的网络条件下进行序贯决策。本文提出了基于混合专家的无线稀疏决策Transformer（WiSDoM），这是一个面向自适应多小区选择的稀疏多任务离线RL框架。

    arXiv:2609.00284v1 Announce Type: cross  Abstract: Emerging 6G wireless networks are expected to operate across diverse deployment scenarios, where variations in network topology, user mobility, traffic demand, and radio conditions challenge the scalability of conventional radio resource management (RRM). While offline reinforcement learning (RL) methods have demonstrated strong decision-making capabilities, learning a single policy that performs consistently across heterogeneous wireless environments remains difficult due to conflicting optimization objectives and limited model specialization. These challenges become particularly pronounced in coordinated multipoint (CoMP) transmission, where selecting the optimal serving-cell combination requires sequential decision-making under evolving network conditions. This paper presents the Wireless Sparse Decision Transformer with Mixture of Experts (WiSDoM), a sparse multi-task offline RL framework for adaptive multi-cell selection. WiSDoM c
    
[^148]: 面向脑卒中运动想象解码的脑电基础模型轻量化适配：领域偏移与受试者级鲁棒性

    Lightweight Adaptation of EEG Foundation Models for Stroke Motor Imagery Decoding: Domain Shift and Subject-Level Robustness

    [https://arxiv.org/abs/2609.00282](https://arxiv.org/abs/2609.00282)

    轻量级 LoRA 适配可让预训练脑电基础模型（尤其是 REVE-base）有效克服健康人群与脑卒中患者之间的领域偏移，在脑卒中运动想象解码中达到约 0.85 的准确率，且更大的模型容量并不保证更好的适配效果。

    

    运动想象（MI）脑电图（EEG）解码有望支持脑卒中后康复，但在健康人群中开发的模型可能无法可靠地迁移到病理脑电信号上。我们评估了低秩适应（LoRA）能否高效地将三个预训练脑电基础模型（即 LaBraM-base、REVE-base 和 REVE-large）适配于左手与右手二分类运动想象解码任务。在 PhysioNet 脑电运动/想象数据集（EEGMMIDB）以及由 30 名脑卒中受试者组成的 UET175 数据集二分类子集上，采用受试者级别的五折交叉验证，对冻结主干、仅训练分类头的基线方法和 LoRA 适配方法进行了评估。在 EEGMMIDB 上，LoRA 将 LaBraM-base 的准确率提升至 0.822，将 REVE-base 的准确率提升至 0.957。在 UET175 上，所有仅训练分类头的模型表现均接近随机水平。使用 LoRA 后，LaBraM-base 仍接近随机水平（0.499±0.009），而 REVE-base 达到了 0.847±0.194，并优于 REVE-large（0.806±0.178），这表明（原文摘要在此处被截断）

    arXiv:2609.00282v1 Announce Type: cross  Abstract: Motor imagery (MI) electroencephalography (EEG) decoding could support post-stroke rehabilitation, but models developed on healthy cohorts may not transfer reliably to pathological EEG. We evaluated whether Low-Rank Adaptation (LoRA) can efficiently adapt three pretrained EEG foundation models (i.e., LaBraM-base, REVE-base, and REVE-large) for binary left- versus right-hand MI decoding. Frozen-backbone head-only baselines and LoRA adaptation were evaluated using subject-wise five-fold cross-validation on the PhysioNet EEG Motor Movement/Imagery Dataset and a binary subset of the UET175 dataset comprising 30 stroke participants. On EEGMMIDB, LoRA increased accuracy to 0.822 for LaBraM-base and 0.957 for REVE-base. On UET175, all head-only models performed near chance. With LoRA, LaBraM-base remained near chance (0.499$\pm$0.009), whereas REVE-base reached 0.847$\pm$0.194 and outperformed REVE-large (0.806$\pm$0.178), indicating that inc
    
[^149]: CompanionSim：用于评估人机关系拟人化的合成数据

    CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships

    [https://arxiv.org/abs/2609.00250](https://arxiv.org/abs/2609.00250)

    该论文发布了CompanionSim——一个包含2,240段模拟人机对话的合成数据模拟框架，覆盖七种用例中的16种聊天机器人行为，用于大规模研究人类对AI陪伴行为的感知。

    

    如今许多人不仅将AI系统视为生产力工具，还将其视为社交伴侣。研究人员热切希望研究AI陪伴行为（例如“认同验证”）的后果，这类行为在人际互动中会唤起信任、共情和依恋。然而，人机交互数据有限且不可靠，拖慢了研究进展。我们通过在多种聊天机器人行为和用例下模拟多轮人机对话，来扩展少量真实世界数据的规模。我们发布了CompanionSim：一个模拟框架，包含2,240段模拟的人机对话，涵盖七种用例中的16种聊天机器人行为。在两个探索人们对陪伴行为感知的实验中，人类参与者对模拟对话和真实对话进行了标注。研究1使用了具有美国代表性的样本（N1 = 628），研究2则在美国、英国、印度和尼日利亚开展（N2 = 3,646）。令人惊讶的是……

    arXiv:2609.00250v1 Announce Type: cross  Abstract: Many people now see AI systems as not just productivity tools but as social companions. Researchers are eager to study the consequences of AI companionship behaviors, such as validation, which evoke trust, empathy, and attachment in human-human interaction. However, human-AI interaction data is limited and unreliable, slowing research progress. We scale small amounts of real-world data by simulating multi-turn human-chatbot dialogue across a range of chatbot behaviors and use cases. We release CompanionSim: a simulation framework with 2,240 simulated human-chatbot conversations representing 16 chatbot behaviors across seven use cases. Human participants annotated the simulated conversations and real-world conversations in two experiments probing perceptions of companionship behaviors. We conducted Study 1 with a U.S. representative sample ($N_{1}~=~628$) and Study 2 across the U.S., U.K., India, and Nigeria ($N_{2}~=~3,646$). Surprisin
    
[^150]: QTEA：基于稀疏残差显著权重与逐列优化的三值大语言模型

    QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization

    [https://arxiv.org/abs/2609.00224](https://arxiv.org/abs/2609.00224)

    QTEA提出了一种低于2比特的训练后量化框架，通过将权重量化为三值、利用1:4半结构化稀疏的显著权重残差进行误差补偿，并结合逐列缩放精修与误差衰减机制，在保持GPU硬件效率的同时显著降低了大语言模型低比特量化的精度损失。

    

    仅权重的训练后量化（PTQ）可以缓解大规模部署大语言模型（LLM）时的计算负担。然而，现有的PTQ方法往往难以在不同模型间泛化，并且在低于2比特的量化下会出现严重的精度损失。许多方法利用非结构化稀疏性来缓解这种损失，但代价是失去了规则性和GPU友好的执行效率。我们提出了QTEA，一个低于2比特的PTQ框架，它将权重量化为三值，并使用显著权重作为残差误差补偿器。为了保持硬件效率，残差被分配到显著列中以半结构化1:4稀疏性选取的列上。我们进一步在GPTQ风格的逐列量化中加入了逐列缩放精修机制，交替更新每列的缩放因子和三值赋值，以减少重构误差。我们还识别出GPTQ中存在与处理顺序相关的误差传播问题，并引入误差衰减机制来减弱后期误差传播（摘要在此处截断）。

    arXiv:2609.00224v1 Announce Type: cross  Abstract: Weight-only post-training quantization (PTQ) can alleviate the computational burden of serving large language models (LLMs) at scale. However, existing PTQ methods often fail to generalize across models and suffer severe accuracy loss below 2 bits. Many leverage unstructured sparsity to mitigate this loss, but at the cost of regularity and GPU-friendly execution. We present QTEA, a sub-2-bit PTQ framework that quantizes weights into ternary values and uses salient weights as residual error compensators. To maintain hardware efficiency, residuals are assigned to selected columns with semi-structured \(1{:}4\) sparsity within the salient columns. We further add column-wise rescale refinement to GPTQ-style column-by-column quantization, alternately updating per-column scales and ternary assignments to reduce reconstruction error. We also identify order-dependent error propagation in GPTQ and introduce error decay to attenuate late-stage e
    
[^151]: WHALE：一种简单的外壳与权重联合优化方法

    WHALE: A Simple Recipe for Joint Harness-Weight Optimization

    [https://arxiv.org/abs/2609.00196](https://arxiv.org/abs/2609.00196)

    提出 WHALE 方法，通过交替进行“在当前外壳下更新模型权重”与“在更新后的模型下搜索更优外壳”两个阶段，实现模型权重与执行框架的联合优化，避免单一组件优化时被冻结组件造成的性能瓶颈。

    

    arXiv:2609.00196v1 公告类型：cross 摘要：智能体的性能同时取决于模型参数以及负责管理上下文与控制流的可执行外壳代码。单独优化其中任何一个组件，都可能使系统受制于另一个被冻结的组件而遭遇瓶颈：权重更新会改变哪种外壳是有效的，而外壳更新会改变模型暴露出哪些能力。现有的联合适应方法只优化权重和文本提示，却让更广泛的外壳保持固定。我们提出权重-外壳交替学习（Weight-Harness Alternating LEarning，WHALE），这是一种简单的方案，交替执行两个阶段：先在当前外壳下更新模型，再在更新后的模型下搜索更好的外壳。我们分别用在线拒绝采样微调和 Meta-Harness 来实例化这两个阶段。何时切换是一个关键的设计选择：为了在不针对不断变化的对应组件进行过度优化的前提下，将真实改进与噪声区分开来，WHALE 使用固定阶段时长（原文在此处截断）……

    arXiv:2609.00196v1 Announce Type: cross  Abstract: Agent performance depends jointly on the model parameters and the executable harness code that manages context and control flow. Optimizing either component in isolation can leave the system bottlenecked by its frozen counterpart: weight updates can change which harness is effective, while harness updates can change which model capabilities are exposed. Existing joint-adaptation methods optimize weights and textual prompts but leave the broader harness fixed. We propose Weight-Harness Alternating LEarning (WHALE), a simple recipe that alternates two phases: updating the model under the current harness, then searching for a better harness under the updated model. We instantiate these two phases with online rejection-sampling fine-tuning and Meta-Harness, respectively. When to switch is a key design choice: to separate real improvements from noise without over-optimizing against a changing counterpart, WHALE uses either fixed phase durat
    
[^152]: 具有线性函数逼近和对数级通信成本的可证明高效的联邦强化学习

    Provably Efficient Federated Reinforcement Learning with Linear Function Approximation and Logarithmic Communication Cost

    [https://arxiv.org/abs/2609.00193](https://arxiv.org/abs/2609.00193)

    提出Fed-LSVI，首个针对具有线性函数逼近的联邦在线强化学习的可证明高效算法，通过基于行列式的事件触发同步机制仅交换压缩充分统计量，在实现$\widetilde{O}(\sqrt{Md^3H^4T})$遗憾界的同时将通信成本降低至对数级。

    

    我们研究了具有线性函数逼近的联邦在线强化学习。尽管近期的多智能体强化学习算法实现了很强的遗憾保证，但它们通常需要共享原始轨迹。这种依赖性导致通信成本随回合数线性增长，并违反了联邦设置中的隐私约束。为了解决这些局限性，我们提出了Fed-LSVI，这是首个针对分段马尔可夫决策过程中具有线性函数逼近的在线强化学习的可证明高效的联邦算法。通过将基于行列式的事件触发同步机制与逐步反向更新机制相结合，Fed-LSVI使智能体能够通过仅交换压缩的充分统计量来协作学习最优策略。我们证明Fed-LSVI实现了$\widetilde{\mathcal O}(\sqrt{Md^3H^4T})$的遗憾界，其中$d$是特征维度，$H$是……

    arXiv:2609.00193v1 Announce Type: cross  Abstract: We study federated online reinforcement learning with linear function approximation. While recent multi-agent reinforcement learning algorithms achieve strong regret guarantees, they typically require sharing raw trajectories. This reliance incurs a communication cost that scales linearly with the number of episodes and violates the privacy constraints of federated settings. To address these limitations, we propose Fed-LSVI, the first provably efficient federated algorithm for online reinforcement learning with linear function approximation in episodic Markov decision processes. By integrating a determinant-based event-triggered synchronization with a stepwise backward update mechanism, Fed-LSVI enables agents to collaboratively learn an optimal policy by exchanging only compressed sufficient statistics. We prove that Fed-LSVI achieves a regret bound of $\widetilde{\mathcal O}(\sqrt{Md^3H^4T})$, where $d$ is the feature dimension, $H$ 
    
[^153]: 面向目标导向分子优化的精英加权监督微调

    Elite-Weighted Supervised Fine-tuning for Goal-Directed Molecular Optimization

    [https://arxiv.org/abs/2609.00189](https://arxiv.org/abs/2609.00189)

    提出EW-SFT方法，利用奖励信号筛选高分的精英分子，再用模型自身的预训练损失对这些分子进行监督微调，从而实现无需依赖架构特定轨迹概率、可跨模型复用的目标导向分子优化。

    

    目标导向优化对于引导分子生成器提出具有理想性质的候选分子至关重要。然而，该方法通常通过策略梯度强化学习来实现，这需要生成了轨迹的对数概率，而其形式依赖于模型架构和生成过程。这使得优化器难以在不同架构和条件生成设计之间复用。监督微调不需要这些机制，但其更新由固定数据集驱动，因此奖励信号从不进入更新过程。我们提出了精英加权监督微调（EW-SFT），它利用奖励来引导对高分分子的精英选择，并在该集合上通过模型自身的预训练损失来更新模型。消融实验表明，奖励信息主要通过精英选择传递，而不是通过所选集合内的连续加权。由于更新仅消耗已评分的分子……（摘要在此处截断）

    arXiv:2609.00189v1 Announce Type: new  Abstract: Goal-directed optimization is essential for steering molecular generators to propose candidates with desired properties. However, it is often implemented with policy-gradient reinforcement learning, which requires a generation-trajectory log-probability whose form depends on the model architecture and generation procedure. This makes an optimizer difficult to reuse across architectures and conditional generative designs. Supervised fine-tuning needs none of that machinery, but its update is driven by a fixed dataset, so the reward never enters the update. We introduce Elite-Weighted Supervised Fine-tuning (EW-SFT), which uses reward to guide elite selection of high-scoring molecules, and updates the model by its own pretraining loss on that set. Ablations show that reward information is passed primarily through elite selection, rather than through continuous weighting within the selected set. Because the update consumes only scored molec
    
[^154]: 面向大语言模型时序评估与知识更新的合成世界

    Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

    [https://arxiv.org/abs/2609.00184](https://arxiv.org/abs/2609.00184)

    该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。

    

    大语言模型（LLM）依赖于静态的预训练语料库，导致其知识随时间推移而变得过时。现有的知识编辑评估方法要么容易遭受快速的数据污染，要么依赖于与现有刚性知识相冲突的反事实编辑。在本工作中，我们提出了一个合成的、模拟驱动的框架，用于研究大语言模型中的知识插入。我们引入了 {\sc ParallelEvents}，这是一个由虚构但逼真的未来世界构成的基准，能够生成连贯的事件轨迹以进行受控评估，在避免污染的同时保持一致性。基于该数据集，我们开发了 {\sc Synapse}，这是一个利用模型自身生成的数据、通过中期训练（mid-training）和指令微调来更新模型参数的训练框架。这一合成流程实现了可扩展的知识整合，而无需昂贵的人工策划数据。实验结果表明，{\sc Synapse} 的性能比现有方法高出 14.23%。

    arXiv:2609.00184v1 Announce Type: new  Abstract: Large language models (LLMs) rely on static pretraining corpora, causing their knowledge to become outdated over time. Existing approaches for evaluating knowledge edits either suffer from rapid contamination or rely on counterfactual edits that conflict with rigid existing knowledge. In this work, we propose a synthetic, simulation-driven framework for studying knowledge insertion in LLMs. We introduce {\sc ParallelEvents}, a benchmark of fictional yet realistic future worlds that generates coherent event trajectories for controlled evaluation, avoiding contamination while preserving consistency. Building on this dataset, we develop {\sc Synapse}, a training framework that uses model-generated data to update model parameters via mid-training and instruction tuning. This synthetic pipeline enables scalable knowledge integration without costly human-curated data. Empirically, {\sc Synapse} outperforms existing methods by 14.23\%, demonstr
    
[^155]: 通用语还是探测假象？重新思考多语言大语言模型中的潜在语言

    Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs

    [https://arxiv.org/abs/2609.00155](https://arxiv.org/abs/2609.00155)

    该研究发现不同的潜在语言探测方法会得出系统性不一致的结论，表明多语言大模型通过英语等“潜在通用语”路由计算的说法可能更多取决于探测手段的选择，而非模型本身固有的计算机制。

    

    潜在语言识别常被用来论证多语言语言模型通过语言特定状态（如英语枢纽）来路由计算。然而，现有探测方法从不同信号推断潜在语言，例如隐藏状态的几何结构，或可从中间表示中解码出的内容。由于这类论断会影响关于模型如何跨语言共享和路由信息的结论，我们追问：这些探测方法测量的究竟是同一现象，还是揭示了多语言计算的不同侧面？我们在多种模型家族、训练方式、领域、任务、检查点以及多达27种语言上研究了这一问题。我们发现，各类识别探测方法存在系统性的不一致：基于GMM的表示探测方法从隐藏状态几何结构中获取证据，显示出更早出现的跨语言混合；而依赖输出空间可解码性的解码式探测方法，则保留了更鲜明的语言特定性，以及

    arXiv:2609.00155v1 Announce Type: cross  Abstract: Latent language identification is often used to argue that multilingual language models route computation through language-specific states, such as English pivots. However, existing probes infer latent language from different signals, such as the geometry of hidden states or what can be decoded from intermediate representations. Since such claims shape conclusions about how models share and route information across languages, we ask whether these probes measure the same phenomenon or expose distinct aspects of multilingual computation. We study this question across model families, training regimes, domains, tasks, checkpoints, and up to 27 languages. We find that identification probes systematically disagree: the GMM-based representation probe, which draws evidence from hidden state geometry, shows earlier cross-lingual mixing, whereas decoding-based probes, which rely on output-space decodability, retain sharper language-specific and 
    
[^156]: 天生有缺陷，进化致完美

    Flawed in Nature, Perfect through Evolution

    [https://arxiv.org/abs/2609.00129](https://arxiv.org/abs/2609.00129)

    该论文提出通过让一群AI/ML模型的系数刻意发生偏离最优的“突变”来维持模型多样性，从而在非平稳环境中充当统计对冲手段，实现可靠且持续的性能提升。

    

    当人工智能（AI）和机器学习（ML）模型所训练的问题发生漂移时，其性能会下降。这是现实世界问题中近乎普遍的特征，因为这些问题往往会发生不可预测的变化。生物进化通过自然选择作用于可遗传的变异，克服了这一障碍，从而实现了智能。AI/ML技术早已融入了各种形式的自然选择，但随着优化过程自然地驱使模型趋同，维持模型多样性一直是一项挑战。在这项工作中，我们展示了一群AI/ML模型在受到使其模型系数刻意偏离最优状态的突变后，能够在变化的环境中可靠且持续地提升性能，其原理是作为应对非平稳性的统计对冲手段。我们将这一机制称为“天生有缺陷，进化致完美”，这反映了集体性能的提升是以牺牲个体性能为代价的。

    arXiv:2609.00129v1 Announce Type: cross  Abstract: The performance of artificial intelligence (AI) and machine learning (ML) models degrades when the problem they were trained on drifts. This is a near-universal feature of real-world problems, which often change unpredictably. Biological evolution has achieved intelligence by overcoming this obstacle through natural selection acting on heritable variation. AI/ML techniques have long incorporated forms of natural selection, but it has been challenging to maintain model diversity as optimization naturally drives convergence. Here we show that a swarm of AI/ML models subjected to deliberate mutations of their model coefficients away from optimality can reliably and sustainably improve performance in changing environments by acting as a statistical hedge against non-stationarity. We call this mechanism 'Flawed in Nature, Perfect through Evolution', reflecting that the collective performance gain goes at the expense of individual performanc
    
[^157]: 好的记忆具备ECC：超越准确率评估视觉-语言模型的记忆能力

    Good Memory Has ECC: Evaluating the Memory of Vision-Language Models Beyond Accuracy

    [https://arxiv.org/abs/2609.00103](https://arxiv.org/abs/2609.00103)

    该论文提出ECCBench基准，从效率、压缩和校准三个维度超越单纯准确率来评估视觉-语言模型的记忆能力，发现预训练VLM对文本记忆有压缩但对视频没有且校准较差，并且若干非Transformer架构在压缩-校准权衡上优于RoPE Transformer。

    

    记忆被广泛认为是大语言模型（LLM）和视觉-语言模型（VLM）面临的一个重要的未解决问题，当前的基准测试通常通过测试模型在长文本或视频上的准确率来评估记忆能力。然而，仅凭准确率会忽略那些对真实长时程任务至关重要的属性。我们提出了ECCBench，这是一个基准测试和评估协议，通过我们称为ECC的三个维度来衡量超越系统容量（即在特定预算下的原始准确率）的记忆能力：效率——从记忆中回答问题所需的计算量（以FLOPs计）；压缩——可压缩的输入是否被更准确或更高效地记住；校准——系统是否会针对自身的不确定性选择弃答，以及出错的代价。我们发现，预训练的VLM会对文本记忆进行压缩，但对视频则不会，且在两者上的校准都很差。在更广泛的记忆骨干架构中，若干非Transformer架构实现了比RoPE Transformer更好的压缩-校准权衡。

    arXiv:2609.00103v1 Announce Type: cross  Abstract: Memory is widely viewed as an important unsolved problem for LLMs and VLMs, and current benchmarks typically evaluate it by testing accuracy over long text or video. However, accuracy alone misses properties that matter for real long-horizon tasks. We introduce ECCBench, a benchmark and evaluation protocol that measures memory beyond a system's capacity--its raw accuracy at a specific budget--via three axes we call ECC: efficiency--the computation, in FLOPs, needed to answer from memory; compression--whether compressible inputs are remembered more accurately or efficiently; and calibration--whether the system abstains in response to its own uncertainty and the cost of an error. We find that pretrained VLMs compress their memory over text but not video and are poorly calibrated on both. Among a broader set of memory backbones, several non-Transformer architectures achieve better compression-calibration tradeoffs than RoPE Transformers, 
    
[^158]: 不同的表征学习目标从相同的心理测量数据中恢复出不同的潜在结构

    Different representation learning objectives recover distinct latent structures from the same psychometric data

    [https://arxiv.org/abs/2609.00100](https://arxiv.org/abs/2609.00100)

    不同的表征学习目标会从同一份心理测量数据中恢复出截然不同的潜在结构——对比学习大幅提升师生匹配检索性能却破坏行为表型组织，而PCA更利于保留行为结构，揭示了检索对齐与行为结构保留之间的根本性权衡。

    

    心理测量问卷包含丰富的条目级信息，然而不同的表征学习目标是否能恢复出相同的潜在组织结构仍不清楚。我们利用塞浦路斯ProW学前试验基线评估中的757对匹配的师生配对数据研究了这一问题。通过使用主成分分析和聚类方法从儿童SDQ、ASBI和CBRS的条目回答中刻画行为结构，得到了四种行为表型。与基于PCA的表征相比，对比学习目标显著提升了教师-儿童配对检索性能，将Top-1准确率从0.13%提高到7.27%，Top-10准确率从1.98%提高到56.14%。然而，对比表征在保留行为表型结构方面不如基于PCA的表征有效。一种联合优化对齐与行为预测的多任务目标部分恢复了行为组织结构，但降低了检索性能。

    arXiv:2609.00100v1 Announce Type: new  Abstract: Psychometric questionnaires contain rich item-level information, yet it remains unclear whether different representation learning objectives recover the same latent organization. We investigated this question using 757 matched teacher-child pairs from the baseline assessment of the Cyprus ProW preschool trial. Behavioral structure was characterized from child SDQ, ASBI, and CBRS item responses using principal component analysis and clustering, yielding four behavioral phenotypes. A contrastive objective substantially improved teacher-child retrieval relative to PCA-based representations, increasing Top-1 accuracy from 0.13% to 7.27% and Top-10 accuracy from 1.98% to 56.14%. However, contrastive representations preserved behavioral phenotype structure less effectively than PCA-based representations. A multi-task objective jointly optimizing alignment and behavioral prediction partially restored behavioral organization but reduced retrieva
    
[^159]: 面向腐蚀可靠机理推理的生成式人工智能

    Generative artificial intelligence for reliable mechanistic reasoning for corrosion

    [https://arxiv.org/abs/2609.00099](https://arxiv.org/abs/2609.00099)

    该研究提出了一个面向腐蚀领域的检索增强生成框架，通过对三个开源大语言模型在专家验证的问答数据上进行微调并结合混合检索流水线，实现了兼具准确性与机理可靠性的腐蚀知识推理，在镁合金腐蚀上取得了显著效果。

    

    腐蚀造成的损失约占全球GDP的4%，可靠的预测对于及时采取防护措施至关重要。机器学习虽然能够基于成分、微观组织和环境变量有效预测腐蚀速率，但无法解释其背后的机理。在安全攸关的材料工程领域，可靠的方法不仅需要准确的检索能力，还需要机理上站得住脚的推理能力，而现有的真实性评估指标无法评价这一能力。本工作提出了一个针对腐蚀领域知识综合的领域自适应检索增强生成框架，并以镁合金腐蚀为例进行了演示。研究者在来自840篇同行评审论文的3,309个经专家验证的问答对上，对三个开源权重语言模型进行了微调，并将其与密集-词法混合检索流水线相结合。检索增强带来了143-194%的Token F1提升，系统的忠实度…

    arXiv:2609.00099v1 Announce Type: new  Abstract: Corrosion accounts for approximately 4% of global GDP, and reliable prediction is essential for timely mitigation. Machine learning effectively predicts corrosion rates from composition, microstructure, and environmental variables, but cannot explain the underlying mechanisms. A reliable approach in safety-critical materials engineering requires not only accurate retrieval but also mechanistically defensible reasoning, a capability that existing factuality metrics cannot assess. This work presents a domain-adapted retrieval-augmented generation framework for corrosion knowledge synthesis, demonstrated on magnesium alloy corrosion. Three open-weight language models (Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B) are fine-tuned on 3,309 expert-verified question-answer pairs from 840 peer-reviewed papers and integrated with a hybrid dense-lexical retrieval pipeline. Retrieval augmentation produces Token F1 gains of 143-194%, with system faithfulnes
    
[^160]: 快过Flash：利用注意力稀疏性实现高效长上下文解码

    Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding

    [https://arxiv.org/abs/2609.00097](https://arxiv.org/abs/2609.00097)

    FFD是一种硬件-算法协同设计框架，通过将选择器与计算器融合为单一内核、基于低比特量化的内容感知扫描取代元数据索引，以及无需全局同步的top-delta动态块过滤策略，实现了免训练、即插即用的长上下文解码加速，内核级加速比最高达11.6倍。

    

    长上下文大语言模型（LLM）的发展受到解码过程中注意力机制的内存带宽瓶颈和二次方复杂度的制约。为了克服基于元数据的度量方法的内存开销与自适应选择策略的计算低效之间的固有权衡，我们提出了快速Flash解码，这是一种新颖的硬件-算法协同设计框架，旨在打破长上下文解码中的内存墙。FFD将选择器和计算器集成到一个完全融合的内核中，通过低比特量化的内容感知扫描取代外部元数据索引。此外，我们引入了top-delta策略，该策略动态过滤注意力块以实现分布自适应的稀疏性，而无需全局同步。FFD提供了一种免训练、即插即用的解决方案，还支持将扫描结果复用于计算，实现了高达11.6倍的内核级加速。

    arXiv:2609.00097v1 Announce Type: cross  Abstract: The development of long-context Large Language Models (LLMs) is constrained by the memory bandwidth bottleneck and quadratic complexity of the attention mechanism during decoding. To overcome the inherent trade-offs between the memory overhead of metadata-based metrics and the computational inefficiency of adaptive selection strategies, we present Faster Flash Decoding (FFD), a novel hardware-algorithm co-design framework designed to break the memory wall in long-context decoding. FFD integrates the selector and computer into a fully fused kernel, replacing external metadata indices with content-aware scanning via low-bit quantization. Furthermore, we introduce the top-delta strategy, which dynamically filters blocks to achieve distribution-adaptive sparsity without global synchronization. Offering a training-free and plug-and-play solution, FFD also enables the reuse of scanning results for computation, achieving up to 11.6x kernel-le
    
[^161]: 用于不平衡时间序列分类的局部参考几何残差增强

    Local Reference Geometry Residual Augmentation for Imbalanced Time Series Classification

    [https://arxiv.org/abs/2609.00093](https://arxiv.org/abs/2609.00093)

    该论文提出局部参考几何（LRG），一种应用于固定特征提取器与分类器之间的轻量级后置特征增强模块，通过度量少数类样本的局部暴露度和类别混合风险，并用来自附近训练几何的标准化有符号位移增强特征，来修复不平衡条件下特征空间的局部几何失效问题。

    

    不平衡时间序列分类问题通常通过改变训练分布、目标函数、逻辑值（logits）或最终阈值来解决。这些干预措施解决了重要的偏差问题，但留下了一个在表示层面未曾衡量的问题：当少数类的支持被削减后，学习到的特征空间在少数类区域附近是否仍然保持局部可靠性？我们识别出一种训练局部几何失效现象：在不平衡条件下，少数类样本可能位于稀疏的、被多数类主导的或混合的特征空间邻域中，即使该表示仍保留了有用的全局类别结构。为了诊断并修复这种失效，我们提出了局部参考几何（LRG），这是一个轻量级的后置特征增强模块，应用于固定的特征提取器和分类器头之间。LRG 仅利用训练特征来度量局部暴露度和类别混合风险，然后使用来自附近训练几何结构的标准化有符号位移来增强每个固定特征。

    arXiv:2609.00093v1 Announce Type: new  Abstract: Imbalanced time series classification is often addressed by changing the training distribution, objective, logits, or final threshold. These interventions address important biases, yet leave a representation-level question unmeasured: after minority support is reduced, does a learned feature space remain locally reliable around minority regions? We identify a training-local geometry failure: under imbalance, minority cases can lie in sparse, rest-dominated, or mixed feature-space neighborhoods, even when the representation retains useful global class structure. To diagnose and repair this failure, we propose Local Reference Geometry (LRG), a lightweight post-hoc feature augmentation module applied between a fixed feature extractor and the classifier head. Using training features only, LRG measures local exposure and class-mixture risk, then augments each fixed feature with a standardized signed displacement from nearby training geometry 
    
[^162]: Safin-1：通过记忆原生的状态演化实现由内而外的安全性

    Safin-1: Safety from Within through Memory-Native State Evolution

    [https://arxiv.org/abs/2609.00092](https://arxiv.org/abs/2609.00092)

    Safin-1提出基于跨上下文历史记忆锚定路由（MARCH）架构的基础模型系列，通过记忆路由与状态演化将安全能力内建于模型原生计算之中，实现无需反复修改主干网络、可在测试时自适应调整的“由内而外”的安全性。

    

    长时程复杂任务要求基础模型能够积累信息、维持内部状态，并在长期交互中不断适应。安全性应当是模型本身的内在属性，而非仅仅依赖外部防护措施或监督微调等事后对齐手段的行为约束。基于这一思考，我们提出“由内而外的安全”（Safety from Within）理念，即安全相关能力通过模型的原生计算来表示和调用。我们提出了Safin-1，这是一系列通过记忆路由和状态演化来实现这一原则的基础模型。Safin-1建立在跨上下文历史的记忆锚定路由（Memory-Anchor Routing across Context History, MARCH）架构之上，该架构能够维持结构化的记忆状态，并通过内容条件路由选择性地检索相关历史信息。它支持在测试时对持久能力状态进行自适应调整，而无需反复修改主干网络，从而实现可控的状态演化（摘要在此处截断）。

    arXiv:2609.00092v1 Announce Type: new  Abstract: Long-horizon complex tasks require foundation models to accumulate information, maintain internal states, and adapt over extended interactions. Safety should be an intrinsic property of the model itself, rather than a behavioral constraint relying solely on external safeguards or post-hoc alignment such as supervised fine-tuning. This motivates Safety from Within, where safety-relevant capabilities are represented and invoked through the model's native computation. We present Safin-1, a family of foundation models realizing this principle through memory routing and state evolution. Safin-1 is built on Memory-Anchor Routing across Context History (MARCH), a network architecture that maintains structured memory states and selectively retrieves relevant historical information through content-conditioned routing. It supports test-time adaptation of persistent capability states without repeatedly modifying the backbone, enabling controlled sp
    
[^163]: 基于证据权重评估特征重要性解释的对齐性与稳定性

    Assessing Alignment and Stability of Feature Importance Explanations via Weight of Evidence

    [https://arxiv.org/abs/2609.00090](https://arxiv.org/abs/2609.00090)

    该论文提出了一个基于证据权重的假设检验框架，能够从原理上评估特征重要性解释方法与先验知识的对齐程度及其稳定性，并将其应用于LIME和SHAP的分析中。

    

    特征重要性方法（FIMs）被广泛应用于可解释人工智能中，用于解释模型的预测结果，然而仅凭归因分数往往难以深入洞察模型背后的推理过程。在本工作中，我们引入了一种新颖的视角，将特征重要性方法嵌入到基于证据权重的假设检验框架中。我们量化了观测到的证据对任何给定的特征重要性假设的支持强度。其中，参考假设可以来源于领域知识、真实标签，或由特征重要性方法本身推导得出。这一表述方式使得对特征重要性方法的原理化评估成为可能，既能捕捉它们与先验知识的对齐程度，也能捕捉其变异性。我们进一步提供了将证据权重与归因方差相联系的理论结果。实证结果表明，在具有不同参考假设的设置下分析LIME和SHAP解释时，我们的策略展现出良好的适用性与灵活性。总体而言，我们的框架提供了一个互补的……

    arXiv:2609.00090v1 Announce Type: cross  Abstract: Feature importance Methods (FIMs) are widely used in Explainable AI to interpret model predictions, yet attribution scores alone often provide limited insight into the underlying reasoning process. In this work, we introduce a novel perspective by embedding FIMs within a hypothesis-testing framework based on Weight of Evidence (WoE). We quantify how strongly the observed evidence supports any given hypothesis on feature importance. The reference hypothesis can stem from domain knowledge, ground truth, or be derived from the FIM itself. This formulation enables a principled evaluation of FIMs, capturing both their alignment with prior knowledge and their variability. We further provide theoretical results linking WoE to attribution variance. Empirical results shows the applicability and flexibility of our strategy analyzing LIME and SHAP explanations in settings with different reference hypotheses. Overall, our framework offers a comple
    
[^164]: 面向电价预测与电池套利的基础模型：它们能否取代针对特定市场的预测模型？

    Foundation models for electricity price forecasting and battery arbitrage: Can they replace market-specific forecasting models?

    [https://arxiv.org/abs/2609.00089](https://arxiv.org/abs/2609.00089)

    研究表明，在零样本模式下，只有TabPFN基础模型能在三个欧洲电力市场中持续显著超越专门的电价预测模型，但其统计优势并不能直接转化为电池储能套利中的经济优势。

    

    基础模型有望在极少甚至无需任务特定训练的情况下提供准确的预测，但它们能否取代专门为电价预测设计的模型仍不明确。本研究在2021至2025年间，针对德国、波兰和西班牙三个市场，将来自五个基础模型家族的九个变体（以零样本模式评估）与两个最先进的电价预测基准模型进行了比较。评估指标包括点预测与概率预测的精度，以及电池储能套利中的经济价值。结果显示，只有TabPFN模型在所有三个市场和所有统计指标上都持续且显著地优于基准模型。然而，这种统计上的优势并不能直接转化为经济上的优势：TabPFN在无限制投标和风险较高的分位数策略下表现最佳，而分布式深度神经网络基准模型在风险容忍度受限的情况下则更具盈利性。

    arXiv:2609.00089v1 Announce Type: new  Abstract: Foundation models promise accurate forecasts with little or no task-specific training, but whether they can replace models designed specifically for electricity price forecasting remains unclear. We compare nine variants from five foundation model families, evaluated in zero-shot mode, with two state-of-the-art electricity price forecasting benchmarks in Germany, Poland, and Spain over 2021-2025. Their performance is assessed in terms of point and probabilistic forecasting accuracy, as well as economic value in battery energy storage arbitrage. Only the TabPFN models consistently and significantly outperform the benchmarks across all three markets and all statistical measures. However, this statistical dominance does not translate directly into economic dominance: TabPFN performs best under unlimited bids and riskier quantile-based strategies, whereas the Distributional Deep Neural Network benchmark is more profitable when risk tolerance
    
[^165]: 包含聚类结构的向量的随机复杂度

    Stochastic complexity of vectors containing cluster structure

    [https://arxiv.org/abs/2609.00084](https://arxiv.org/abs/2609.00084)

    本文提出一种递归公式来高效计算NML模型的归一化常数，将计算包含聚类结构向量最短编码长度的时间复杂度从多项式时间降低到线性时间。

    

    本文研究了使用归一化最大似然（NML）模型计算包含聚类结构的编码向量的随机概率（最短编码长度）的问题。这对于基于最小描述长度（MDL）原理的数据聚类具有重要的理论和实践意义，例如用于估计数据的最佳聚类数目和最佳聚类结构。基于NML模型直接计算包含聚类结构的向量的最短编码长度，需要相对于向量大小和聚类数目的多项式时间。我们通过引入一个递归公式来高效计算NML模型的归一化常数，证明了这是一个可解的问题。新公式的时间复杂度是线性的，相比于之前关于向量大小和聚类数目的多项式时间有了显著改进。

    arXiv:2609.00084v1 Announce Type: new  Abstract: This paper studies the problem of computing the stochastic probability (shortest code length) of the encoded vectors containing cluster structure using Normalized Maximum Likelihood (NML) model. This is of great theoretical and practical importance in data clustering based on Minimum Description Length (MDL) principle, such as for estimating the best number of clusters and best cluster structure for the data. Straightforward computation of the shortest code length of the vector containing cluster structure based on the NML model requires polynomial time with respect to the size of the vector and number of clusters. We show that this is a tractable problem by introducing a recursion formula for the efficient computation of normalizing constant from the NML model. The time complexity of the new formula is linear opposed to previous polynomial time with respect to the size of the vector and number of clusters.
    
[^166]: RW-LoRA：基于随机游走的通信高效去中心化 LoRA 微调

    RW-LoRA: Communication-Efficient Decentralized LoRA Fine-Tuning via Random Walks

    [https://arxiv.org/abs/2609.00078](https://arxiv.org/abs/2609.00078)

    提出基于随机游走的去中心化 LoRA 微调方法 RW-LoRA，通过单个模型令牌在网络中顺序更新，免除全局同步，大幅降低通信与计算成本并避免聚合误差。

    

    arXiv:2609.00078v1 公告类型：cross 摘要：以 LoRA 为代表的参数高效微调方法已成为适配大型基础模型的标准方式。然而，将微调推广到分布式场景面临诸多挑战：大多数现有的分布式 LoRA 方法依赖集中式聚合，而基于 gossip 的去中心化 LoRA 则需要在多个模型副本之间进行反复同步。这两种方式都会带来巨大的通信开销，并因同时聚合多个模型更新而引入误差。本文从一个不同的视角出发，提出了一种基于随机游走的 LoRA 微调方案：无需维护多个模型副本，而是让单个模型令牌在网络中游走，并利用本地微调目标对其进行顺序更新。该设计消除了全局同步的需求，显著降低了通信与计算成本，并避免了聚合误差。我们提供了严格的收敛保证。

    arXiv:2609.00078v1 Announce Type: cross  Abstract: Parameter-efficient fine-tuning methods such as LoRA have become a standard approach for adapting large foundation models. Adopting fine-tuning to distributed settings faces several challenges. Most existing distributed LoRA methods rely on centralized aggregation, and gossip-based decentralized LoRA requires repeated synchronization among multiple model copies. Both methods incur significant communication overhead and introduce errors due to simultaneous aggregation of multiple model updates. In this paper, we take a different perspective and propose a random-walk-based LoRA fine-tuning scheme. Instead of maintaining multiple model replicas, a single model token traverses the network and is updated sequentially using local fine-tuning objectives. This design eliminates the need for global synchronization, substantially reduces communication and computation costs, and avoids aggregation errors. We provide rigorous convergence guarantee
    
[^167]: 当预测误差不足够时：评估用于因果估计的干扰函数预测

    When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation

    [https://arxiv.org/abs/2609.00071](https://arxiv.org/abs/2609.00071)

    在部分线性模型的模拟研究中，干扰函数的预测误差无法一致地反映因果估计的偏差，表明仅用预测误差来评估干扰函数估计器对于因果推断而言是不充分的。

    

    预测误差被广泛用于评估因果推断中干扰函数估计器的性能，但其与因果估计器表现之间的关系可能因性能度量指标的不同而存在差异。我们在部分线性模型中通过蒙特卡洛模拟研究了这一问题。我们比较了普通最小二乘法（OLS）、广义可加模型（GAMs）、XGBoost 以及结合 XGBoost 的双重机器学习（DML-XGBoost），评估指标包括干扰函数预测误差、偏差、均方根误差（RMSE）以及95%置信区间覆盖率。我们还检验了一种简单的联合误差度量，其基于暴露干扰函数与结局干扰函数估计误差的绝对交叉乘积。在所有模拟设置中，XGBoost 在非先知方法中具有最低的 RMSE，而 DML-XGBoost 通常能提供更好的置信区间覆盖率。预测误差在不同方法和设置下并不能一致地反映因果估计的偏差，且点估计表现最好的方法（摘要在此处截断）

    arXiv:2609.00071v1 Announce Type: new  Abstract: Prediction error is widely used to evaluate nuisance-function estimators in causal inference, but its relationship with causal estimator performance may differ across performance measures. We studied this question in a partially linear model using Monte Carlo simulations. We compared ordinary least squares (OLS), generalized additive models (GAMs), XGBoost, and Double Machine Learning with XGBoost (DML-XGBoost), evaluating nuisance-function prediction error, bias, RMSE, and 95\% confidence interval coverage. We also examined a simple joint-error measure based on the absolute cross-product of estimation errors from the exposure and outcome nuisance functions. Across the simulated settings, XGBoost had the lowest RMSE among the non-oracle methods, while DML-XGBoost generally provided better confidence interval coverage. Prediction error did not consistently track causal bias across methods and settings, and the method with the best point-e
    
[^168]: OCGQuant：面向NVFP4量化的异常值伴随分组方法

    OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization

    [https://arxiv.org/abs/2609.00066](https://arxiv.org/abs/2609.00066)

    提出OCGQuant，一种以“异常值伴随分组（OCG）”为核心的NVFP4训练后量化方法，通过自适应地将异常值通道与伴随通道分组，减少由块最大值主导缩放因子所造成的“附带量化误差”，从而在不引入额外计算的前提下提升低比特推理的量化精度。

    

    NVFP4是一种面向低比特推理的高效微缩放（microscaling）格式，但激活异常值仍会降低NVFP4块内的量化精度。在每个量化块内，较大的激活值会主导块缩放因子，从而增大共享同一缩放因子的其余数值的量化误差。现有的训练后量化（PTQ）方法通过混合精度、旋转或残差补偿等策略来缓解异常值带来的误差，但这些方法要么并非专门针对NVFP4设计，要么会引入额外的计算开销。在本工作中，我们从通道分组的视角重新审视NVFP4，并将由块最大值所设定的缩放因子下其余块内数值产生的可减少误差定义为“附带量化误差”。基于这一洞察，我们提出了OCGQuant——一种以异常值伴随分组（Outlier-Companion Grouping, OCG）为核心的训练后量化方法，该方法自适应地将异常值通道与……（原文摘要在此处截断）

    arXiv:2609.00066v1 Announce Type: cross  Abstract: NVFP4 is an efficient microscaling format for low-bit inference, but activation outliers can still degrade quantization accuracy within NVFP4 blocks. Within each quantization block, large activations can dominate the block scale, increasing the quantization error of the remaining values sharing the same scale. Existing post-training quantization (PTQ) methods mitigate outlier errors through strategies such as mixed precision, rotation, or residual compensation, but these approaches are either not specifically tailored to NVFP4 or introduce additional computation. In this work, we revisit NVFP4 from a channel-grouping perspective and define the reducible error incurred by remaining block values under the scale set by the block maximum as Collateral Quantization Error. Based on this insight, we propose OCGQuant, a post-training quantization method centered on Outlier-Companion Grouping (OCG), which adaptively pairs outlier channels with 
    
[^169]: 注意力敏感性并不足够：在微调下解耦注意力层面与行为层面的上下文学习

    Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning

    [https://arxiv.org/abs/2609.00064](https://arxiv.org/abs/2609.00064)

    该论文形式化了注意力层面的“上下文敏感性”（ICS）指标，并通过Llama-2-7B上的四臂消融实验证明，最大化ICS并不能保留真实的行为性上下文学习能力（ICL-GAP接近零且MMLU从0.371降至0.279），揭示了注意力代理指标与行为层面ICL之间的“古德哈特定律”式解耦。

    

    上下文学习（ICL）使大型语言模型能够通过示例适应新任务，而微调可能会削弱这种行为。许多保持性诊断方法依赖检查注意力：如果注意力随示例的变化而变化，模型就被视为对上下文敏感。本文探讨这种代理指标在被优化之后能在多大程度上被信任。我们形式化了“上下文敏感性”（ICS），即在匹配与不匹配示例前缀上最后一个token注意力分布之间的平均行距离，并将其与“ICL差距”（ICL-GAP）配对，后者衡量相同前缀之间的行为准确率差距。在Llama-2-7B上进行的受控四臂消融实验中，一个最大化ICS的正则化器（armKL）将ICS推高至1.413，达到其几何上限的0.5%以内。然而行为层面的读数讲述了不同的故事：ICL-GAP保持在接近零的水平，MMLU准确率从0.371下降至0.279，这是有界注意力代理指标的“古德哈特式”解耦。端点统计定位……

    arXiv:2609.00064v1 Announce Type: cross  Abstract: In-context learning (ICL) lets large language models adapt to new tasks from demonstrations, and fine-tuning can erode this behaviour. Many preservation diagnostics inspect attention: if attention changes when demonstrations change, the model is treated as context-sensitive. This paper asks how far that proxy can be trusted once it is optimised. We formalise \emph{In-Context Sensitivity} (ICS), the average row distance between last-token attention on matched and mismatched demonstration prefixes, and pair it with \emph{ICL-GAP}, the behavioural accuracy gap between the same prefixes. In a controlled four-arm ablation on Llama-2-7B, an ICS-maximising regulariser ($\armKL$) drives ICS to $1.413$, within $0.5\%$ of its geometric ceiling. The behavioural readout tells a different story: ICL-GAP stays near zero and MMLU accuracy moves from $0.371$ to $0.279$, a Goodhart dissociation of the bounded attention proxy. Endpoint statistics locate
    
[^170]: ReNFT：通过内部概率质量重新校准修复奖励后训练中的模式坍塌

    ReNFT: Repairing Mode Collapse in Reward Post-Training via Internal Probability-Mass Recalibration

    [https://arxiv.org/abs/2609.00061](https://arxiv.org/abs/2609.00061)

    本文提出ReNFT方法，通过内部概率质量重新校准修复扩散模型奖励后训练中出现的模式坍塌，在保留已获得奖励的同时恢复提示内多样性，无需依赖任何外部信号或接口。

    

    扩散生成器的奖励后训练不可避免地将概率质量集中在少数受奖励偏好的模式上，这种模式坍塌消除了提示内的多样性。现有的缓解坍塌方法依赖于外部信号或接口，例如用感知目标增强奖励、调整参考正则化或修改文本编码器，但没有一种方法能够在保留已获得奖励的同时修复已经坍塌的适配器。我们观察到，在线后训练主要是在预训练继承的能力之上重新分配概率质量，而不是学习新的视觉内容。因此，坍塌是抑制而非删除，可以从生成器内部逆转。我们提出ReNFT，通过内部概率质量重新校准来修复高奖励、低多样性的适配器。无条件探测首先优先处理“反中心”提示，在这些提示中，与提示无关的偏差最容易暴露。

    arXiv:2609.00061v1 Announce Type: cross  Abstract: Reward post-training of diffusion generators inevitably concentrates probability mass on a few reward-favored modes, a mode collapse that erases within-prompt diversity. Existing methods for mitigating collapse rely on external signals or interfaces, augmenting the reward with perceptual objectives, adjusting reference regularization, or modifying the text encoder, but none repairs an adapter that has already collapsed while preserving the acquired reward. We observe that online post-training primarily reallocates probability mass over capabilities inherited from pretraining rather than learning new visual content. Collapse is therefore suppression, not deletion, and can be reversed from within the generator. We propose ReNFT, which repairs a high-reward, low-diversity adapter through internal probability-mass recalibration. Unconditional probes first prioritize "anti-hub" prompts where the prompt-independent bias is easiest to expose.
    
[^171]: DISTAL：面向结构无关材料性质预测的蒸馏与自监督预训练

    DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction

    [https://arxiv.org/abs/2609.00059](https://arxiv.org/abs/2609.00059)

    DISTAL提出了一种双先验框架，通过自监督成分预训练和从ALIGNN教师模型进行结构知识蒸馏，实现了推理时无需晶体结构输入的结构无关材料性质预测，适用于低数据、结构信息缺失的早期筛选场景。

    

    在低数据环境下，材料性质预测仍然十分困难，因为许多目标性质仅有数量有限的标注样本支持。预测精度最高的模型通常依赖于晶体结构，这在结构信息有限或不可获得的情况下限制了其在早期筛选中的应用。为应对这一挑战，我们提出了DISTAL，一个用于结构无关材料性质预测的双先验框架，它将自监督成分预训练与结构感知的知识蒸馏相结合。DISTAL首先利用145个基于成分的描述符，从大型虚拟成分空间中学习可迁移的成分表示。随后，它将预训练的ALIGNN教师模型中的结构知识蒸馏到一个以成分为条件的学生模型中。这种设置使得结构先验可以在训练阶段被利用，而在推理阶段无需结构输入。通过整合……（原文摘要在此截断）

    arXiv:2609.00059v1 Announce Type: cross  Abstract: Materials property prediction remains difficult in low-data settings, where many target properties are supported by only a limited number of labeled samples. Models with the strongest predictive accuracy often depend on crystal structures, which restricts their use in early-stage screening when structural information is limited or unavailable. To address this challenge, we propose DISTAL, a dual-prior framework for structure-agnostic materials property prediction that combines self-supervised compositional pretraining with structure-aware knowledge distillation. DISTAL first learns transferable compositional representations from a large virtual composition space using 145 composition-derived descriptors. It then distills structural knowledge from a pretrained ALIGNN teacher into a composition-conditioned student. This setting allows structural priors to be used during training without requiring structural inputs at inference. By integr
    
[^172]: ValueGraph：价值信号引导的图预训练方法用于上下文化用户表示

    ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation

    [https://arxiv.org/abs/2609.00057](https://arxiv.org/abs/2609.00057)

    提出ValueGraph图预训练框架，将自动推断的道德价值信号作为软约束辅助信号，结合对比学习与聚类目标学习上下文化的用户表示，在立场检测和推特机器人检测任务上取得提升。

    

    价值信号是一种聚合的用户级道德表征，能够从用户的在线言论中捕捉其被推断出的与价值观相关的倾向。社交媒体上的用户行为不仅受用户说什么或与谁互动的影响，还受到用户表达态度时所依托的价值信号的影响。然而，现有的用户表示方法大多忽略了这一与价值相关的维度。我们提出ValueGraph，一个图预训练框架，它将自动推断的道德价值信号作为含噪的辅助信号，用于学习上下文化的用户表示。ValueGraph从帖子-回复图中学习语义和结构表征，并通过对比学习和聚类目标，基于相对价值相似度进一步对齐用户。ValueGraph并不把推断出的价值观当作标准的心理学标签，而是将其用作表示学习的软约束。在立场检测和推特机器人检测任务上的实验表明……

    arXiv:2609.00057v1 Announce Type: cross  Abstract: Value signals are aggregated user-level moral representations that capture users' inferred value-related tendencies from their online discourse. User behavior on social media is shaped not only by what users say or whom they interact with, but also by the value signal through which they express attitudes. Existing user representation methods largely miss this value-relevant dimension. We propose ValueGraph, a graph pre-training framework that uses automatically inferred moral-value signals as noisy auxiliary signals for contextualized user representation. From post-reply graphs, ValueGraph learns semantic and structural representations and further aligns users through relative value similarity with contrastive and clustering objectives. Rather than treating inferred values as gold psychological labels, ValueGraph uses them as soft constraints for representation learning. Experiments on stance detection and twitter bot detection show co
    
[^173]: 基于AOC-偏序集的关系概念分析中的收敛性问题

    Convergence issues in Relational Concept Analysis based on AOC-posets

    [https://arxiv.org/abs/2609.00054](https://arxiv.org/abs/2609.00054)

    本文研究了在关系概念分析（RCA）中以AOC-偏序集替代完整概念格时所产生的收敛性问题，从而在缓解组合爆炸的同时保留结构中最具信息量的部分。

    

    形式概念分析（FCA）是一种从描述对象集合与属性集合的二元表中进行概念分类构建和规则发现的方法。为了处理非二元及更复杂的数据，已有多种扩展方法被提出，例如用于多关系数据的关系概念分析（RCA）。RCA旨在突出那些由其与其他对象组之间的关系所刻画的若干对象组。由于底层数据更丰富、更复杂，RCA能够产生比FCA更丰富的结果，但代价是更高的计算复杂度和解释复杂度。FCA中最常用的概念分类结构是概念格。然而，在许多应用中，人们更倾向于使用概念格的子结构，如AOC-偏序集，其目的或是为了缓解组合爆炸问题，或是为了聚焦于结构中信息量最大的部分。实际上，在AOC-偏序集中，仅保留包含……

    arXiv:2609.00054v1 Announce Type: new  Abstract: Formal Concept Analysis (FCA) is an approach for conceptual classification building and rule discovery from a binary table describing a set of objects by a set of attributes. Extensions have been proposed to deal with non-binary and more complex data, such as Relational Concept Analysis (RCA) for multi-relational data. RCA aims to highlight groups of objects characterized by their relationships with other groups of objects. The richer and more complex nature of the underlying data allows RCA to produce richer results than FCA, at the expense of higher computational and interpretive complexity. The most commonly used conceptual classification structure in FCA is the concept lattice. However, in many applications, concept lattice substructures, such as AOC-posets, are preferred over the full lattice, either to mitigate combinatorial blow-up or to focus on the most informative parts of the structure. Indeed, in an AOC-poset, only concepts i
    
[^174]: AgentProv：基于工具使用策略探针的智能体式LLM API提供商审计

    AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes

    [https://arxiv.org/abs/2609.00052](https://arxiv.org/abs/2609.00052)

    提出AgentProv，首个基于动作的智能体式LLM API身份审计方法，通过工具使用策略探针利用内化在模型权重中的工具使用行为，克服了文本通道审计在智能体API场景下的结构性脆弱问题。

    

    商业LLM API宣称提供特定的基础模型，但其服务的底层模型可能被悄然替换、量化或封装（例如为了节省部署成本）。所有现有的审计方法都是从文本输出通道来判断底层模型的身份，这对于智能体式API而言在结构上是脆弱的，因为现代服务栈（OpenAI、Anthropic、Gemini、Cloudflare Workers AI、LangGraph）在模型调用工具时会丢弃文本，只暴露结构化动作；而且提供商注入的系统提示词会严重扭曲文本分布，足以使基于文本通道的测试错误地指控诚实的提供商替换了所声称的模型。我们观察到，近期的智能体式后训练已将工具使用直接内化到模型权重中，这开辟了一条服务栈仍然暴露、且对部署环境基本不变的新审计通道。我们提出智能体溯源方法AgentProv，这是首个面向智能体式LLM API的基于动作的身份审计方法。

    arXiv:2609.00052v1 Announce Type: cross  Abstract: Commercial LLM APIs advertise a specific foundation model, but the served backbone may be silently substituted, quantized, or wrapped, for example to save deployment costs. All existing audits decide backbone identity from the text-output channel, which is structurally fragile for agentic APIs because modern serving stacks (OpenAI, Anthropic, Gemini, Cloudflare Workers AI, LangGraph) discard text and expose only structured actions when the model calls a tool, and provider-injected system prompts can distort text distributions enough that text-channel tests falsely accuse honest providers of substituting the claimed model. We observe that recent agentic post-training internalizes tool-use directly into the weights, opening a new audit channel that the serving stack still exposes and that is largely invariant to deployment context. We introduce Agentic Provenance (AgentProv), the first action-based identity audit for agentic LLM APIs: Ag
    
[^175]: 迈向智能体化云工程：基于零信任智能体套件的图工程与循环工程

    Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

    [https://arxiv.org/abs/2609.00050](https://arxiv.org/abs/2609.00050)

    提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。

    

    智能体AI正在推动基于云的工作流的发展，其中自主智能体可以对运营状态进行推理、调用授权工具、修改软件和基础设施、部署服务、验证执行结果，并在长时程、多步骤任务中进行自适应调整。构建此类工作流需要针对工作流推进、受限执行、故障恢复和可验证完成等环节的显式机制。我们提出了智能体云工作流工程，这是一个智能体AI框架，它将自然语言描述的智能体云工程任务转化为经过验证的代码仓库和经过验证的运营性云部署，从而实现基于云的智能体工作流自动化。该框架分离了三个互补的关注点：图工程负责指定长时程工作流推进以及依赖验证的状态转移；循环工程提供有界的诊断、修复或重新规划、重试和重新验证；智能体套件工程则（负责执行零信任的受限执行控制）。

    arXiv:2609.00050v1 Announce Type: cross  Abstract: Agentic AI is enabling cloud-based workflows in which autonomous agents reason over operational state, invoke authorized tools, modify software and infrastructure, deploy services, verify execution outcomes, and adapt across long-horizon, multistep tasks. Engineering such workflows requires explicit mechanisms for workflow progression, constrained execution, failure recovery, and verifiable completion. We present Agentic Cloud Workflow Engineering, an agentic AI framework that transforms natural-language agentic cloud-engineering tasks into validated code repositories and verified operational cloud deployments for automating cloud-based agentic workflows. The framework separates three complementary concerns: graph engineering specifies long-horizon workflow progression and verification-dependent transitions; loop engineering provides bounded diagnosis, repair or re-planning, retry, and re-verification; and agent harness engineering enf
    
[^176]: REAL-Q：基于动态梯度下降的大语言模型端到端量化

    REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent

    [https://arxiv.org/abs/2609.00049](https://arxiv.org/abs/2609.00049)

    REAL-Q提出了一种打破传统折中的后训练量化新范式，通过端到端对齐的代理损失目标和每128列一次的动态块级梯度下降，解决了现有方法中Hessian矩阵被整层冻结导致的信息错位问题，从而更精确地逼近全局损失实现大语言模型量化。

    

    后训练量化（PTQ）是在严格资源约束下部署大语言模型（LLM）的关键技术。当前最先进的PTQ方法使用单一的闭式二阶求解器对每一层进行量化：为了保持解析上的可处理性，这些方法对全局损失进行了大量近似（舍弃跨通道耦合、将输出行池化为组），随后在整个层内冻结所得的Hessian矩阵，无法随着损失景观逐列变化而对其进行更新——我们将这种现象称为信息错位（information misalignment）。我们提出REAL-Q（Real-time E2E-loss Aligned LLM Quantization，实时端到端损失对齐的大语言模型量化），这是一种新颖的PTQ范式，打破了这一折中：REAL-Q不再为了解析可处理性而稀释目标函数，而是针对全局损失的端到端对齐代理目标，并在每处理一个列块（128列）后应用细粒度、动态的块级梯度下降对其进行优化。通过耦合这种细粒度……

    arXiv:2609.00049v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is essential for deploying large language models (LLMs) under strict resource constraints. State-of-the-art PTQ methods quantize each layer with a single closed-form second-order solver: to remain analytically tractable, they heavily approximate the global loss (dropping cross-channel coupling, pooling output rows into groups), and they then freeze the resulting Hessian across the entire layer, with no way to refresh it as the loss landscape shifts column by column--a phenomenon we call information misalignment. We propose REAL-Q (Real-time E2E-loss Aligned LLM Quantization), a novel PTQ paradigm that breaks this compromise: instead of diluting the objective for the sake of analytic tractability, REAL-Q targets an end-to-end-aligned surrogate of the global loss and refines it via fine-grained, dynamic Block-wise Gradient Descent applied after every column block (128 columns). By coupling this fine-grain
    
[^177]: 面向多任务图预训练的融合全局上下文的任务特定提示

    Task-Specific Prompt with Global Context for Multi-Task Graph Pre-Training

    [https://arxiv.org/abs/2609.00047](https://arxiv.org/abs/2609.00047)

    提出TPGC双先验提示初始化方法，通过显式建模任务先验与结构先验的协同作用，解决多任务图预训练中随机初始化提示导致的任务相关性弱、结构感知差和可迁移性不足的问题。

    

    图提示学习是一种在低资源场景下将预训练图模型适配到下游任务的有效范式。然而，现有多任务图预训练框架通常使用随机初始化的提示，导致提示空间、前置任务目标与图结构特征之间的对齐不佳，这极大地削弱了提示表示的任务相关性、结构感知能力和可迁移性。为了应对这一挑战，我们提出了TPGC，一种双先验提示初始化解决方案，它显式地建模任务先验与结构先验之间的协同作用。具体而言，任务先验注入模块首先在辅助图上进行短暂的同源多任务预训练，使提示初始化能够继承与多个前置任务相关的优化偏好。在任务感知表示的基础上，结构先验注入模块进一步提取可迁移的结构信息以增强提示的全局上下文感知能力。

    arXiv:2609.00047v1 Announce Type: cross  Abstract: Graph prompt learning is an effective paradigm to adapt pre-trained graph models to downstream tasks in low-resource scenarios. However, existing multi-task graph pre-training frameworks generally use randomly initialized prompts, leading to poor alignment between the prompt space, pretext objectives and graph structural characteristics. This greatly weakens the task relevance, structural awareness and transferability of prompt representations. To address this challenge, we propose TPGC, a dual-prior prompt initialization solution that explicitly models the synergy between task prior and structural prior. Specifically, the Task-Prior Injection Module first conducts a short homologous multi-task pre-training on an auxiliary graph, enabling prompt initialization to inherit optimization preferences associated with multiple pretext tasks. Built on the task-aware representations, the Structure-Prior Injection Module further extracts transfe
    
[^178]: 密集弱隐藏：个体光滑性条件下填补非凸与PL有限和优化的复杂度差距

    Dense Weak Hiding: Closing Complexity Gaps in Nonconvex and PL Finite-Sum Optimization under Individual Smoothness

    [https://arxiv.org/abs/2609.00045](https://arxiv.org/abs/2609.00045)

    本文在个体光滑性条件下证明了非凸有限和优化的匹配IFO复杂度下界，填补了先前结果中缺失的√n因子并确定了其极小极大复杂度，同时提出重启PAGE算法，在PL条件下获得了更紧的复杂度保证。

    

    在个体光滑性条件下，非凸有限和优化的最优增量一阶oracle（IFO）复杂度问题一直悬而未决。已知算法需要 $O(n+\sqrt{n}\,\Delta L_{\max}/\varepsilon^2)$ 次调用，而先前的下界结果缺少一个 $\sqrt{n}$ 因子。我们为随机化IFO算法证明了匹配的下界，其中算法的分量索引与查询点可以依赖于完整的前序交互记录及私有随机性。由此，在个体光滑性与均方光滑性两种条件下，我们将极小极大IFO复杂度确定到普适常数因子范围内。在全局Polyak-Łojasiewicz（PL）条件下，当 $\kappa_{\mathrm{ms}}<\sqrt{n}$ 时，标准PAGE的保证并不紧致。重启PAGE算法在 $1\leq\kappa_{\mathrm{ms}}\leq\sqrt{n}$ 时达到 $O(n+n\log(\Delta/\varepsilon)/(1+\log(\sqrt{n}/\kappa_{\mathrm{ms}})))$，在 $\kappa_{\mathrm{ms}}\geq\sqrt{n}$ 时达到 $O(n+\kappa_{\mathrm{ms}}\sqrt{n}\log(\Delta/\varepsilon))$。

    arXiv:2609.00045v1 Announce Type: cross  Abstract: Under individual smoothness, the optimal incremental first-order oracle (IFO) complexity of nonconvex finite-sum optimization has remained open. Known algorithms use $O(n+\sqrt{n}\,\Delta L_{\max}/\varepsilon^2)$ calls, while prior lower bounds miss a factor of $\sqrt{n}$. We prove the matching lower bound for randomized IFO algorithms whose component indices and query points may depend on the complete preceding transcript and private randomness. This determines the minimax IFO complexity up to universal constants under both individual and mean-squared smoothness.   Under the global Polyak-Lojasiewicz (PL) condition, the standard PAGE guarantee is not tight when $\kappa_{\mathrm{ms}}<\sqrt{n}$. Restarted PAGE attains $O(n+n\log(\Delta/\varepsilon)/(1+\log(\sqrt{n}/\kappa_{\mathrm{ms}})))$ for $1\leq\kappa_{\mathrm{ms}}\leq\sqrt{n}$, and $O(n+\kappa_{\mathrm{ms}}\sqrt{n}\log(\Delta/\varepsilon))$ for $\kappa_{\mathrm{ms}}\geq\sqrt{n}$. 
    
[^179]: UI-Venus-2 技术报告

    UI-Venus-2 Technical Report

    [https://arxiv.org/abs/2609.00028](https://arxiv.org/abs/2609.00028)

    UI-Venus-2是一个通用GUI基础智能体，通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行，并从环境、任务和验证三个维度联合扩展，从而获得可靠的强化学习信号并迈向实际部署。

    

    多模态GUI智能体已成为数字任务自动化的一个有前景的范式，但由于环境覆盖有限、任务构建脆弱以及奖励验证不可靠，从面向基准测试的模型过渡到可靠的真实世界应用仍然充满挑战。在本工作中，我们提出了UI-Venus-2，一个通用基础GUI智能体，旨在通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行。为弥合迈向实际部署的差距，我们联合扩展了三个关键维度：(1) 环境，将覆盖范围扩展至170多个多语言移动应用和原生桌面操作系统；(2) 任务，采用深度研究流水线进行基于功能的指令生成；(3) 验证，采用结合视觉关键点和多模型投票的轨迹级与样本级评估器，以确保训练中可靠的强化学习信号。

    arXiv:2609.00028v1 Announce Type: new  Abstract: Multimodal GUI agents have emerged as a promising paradigm for digital task automation, yet transitioning from benchmark-oriented models to dependable real-world applications remains challenging due to limited environment coverage, brittle task construction, and unreliable reward verification. In this work, we present UI-Venus-2, a general-purpose foundation GUI agent designed to operate across mobile, web, and desktop environments through a unified closed-loop reasoning-action framework. To bridge the gap toward practical deployment, we jointly scale three critical dimensions: (1) Environments, expanding coverage to more than 170 multilingual mobile apps and native desktop operating systems; (2) Tasks, employing a deep-research pipeline for function-grounded instruction generation; and (3) Verification, adopting trace-level and sample-level evaluators with visual keypoints and multi-model voting to ensure reliable RL signals for trainin
    
[^180]: ES-AHD：一种用于自动启发式设计的进化策略框架

    ES-AHD: An Evolution Strategy Framework for Automatic Heuristic Design

    [https://arxiv.org/abs/2609.00023](https://arxiv.org/abs/2609.00023)

    ES-AHD框架将进化策略与大语言模型驱动的自动启发式设计深度融合，通过基于LLM的语义重组和基于温度采样的随机协方差自适应两大机制，实现有针对性的中心导向搜索，并动态平衡探索与利用。

    

    本文提出了ES-AHD，一个将进化策略（ES）从根本上融入大语言模型（LLM）驱动的自动启发式设计（AHD）的新型框架。现有的进化方法主要依赖随机的、个体层面的变异，导致搜索盲目以及探索与利用之间的失衡。为解决这些问题，ES-AHD引入了两个核心机制。第一，基于LLM的语义重组摒弃了传统的点对点繁殖方式，利用LLM的上下文推理能力，从表现最优的个体中显式提取核心见解，从而建立一个有前景的语义搜索方向，将随机的代码变异转化为受ES启发的、有针对性的、以中心为导向的采样。第二，通过温度采样实现随机协方差自适应，动态地解决探索与利用的两难困境，通过将ES中的协方差矩阵映射到LLM的采样温度上（摘要在此处截断）

    arXiv:2609.00023v1 Announce Type: cross  Abstract: In this paper, we introduce ES-AHD, a novel framework that fundamentally integrates Evolution Strategy (ES) into Large Language Model (LLM)-driven Automatic Heuristic Design (AHD). Existing evolutionary approaches predominantly rely on random, individual-level mutation, leading to blind search and an imbalance between exploration and exploitation. To address these issues, ES-AHD introduces two core mechanisms. First, Semantic Recombination via LLMs discards traditional point-to-point reproduction. By leveraging the LLM's contextual reasoning to explicitly extract core insights from top-performing individuals, the algorithm establishes a promising semantic search direction. This transforms random code mutation into targeted, center-guided sampling inspired by ES. Second, Stochastic Covariance Adaptation via Temperature Sampling dynamically addresses the exploration-exploitation dilemma. By mapping the covariance matrix in ES to the LLM'
    
[^181]: I-CARE：在可控、多样且具代表性的文生图模型遗忘设置中分析干扰相关现象

    I-CARE: Analysis of interference-related phenomena in a controllable, diverse and representative unlearning setting for text-to-image models

    [https://arxiv.org/abs/2609.00003](https://arxiv.org/abs/2609.00003)

    本文提出I-CARE方法论，首次将文生图模型机器遗忘过程中对语义相关概念造成的意外损害（即“干扰”）形式化为首要研究对象，通过为任务、指标和结果报告提供正式定义，实现对干扰现象的系统性、可复现研究。

    

    arXiv:2609.00003v1 公告类型：新论文 摘要：机器遗忘研究如何从AI模型中移除知识，使系统忘记其之前学到的某个概念。尽管生成式机器遗忘取得了快速进展，但本应保留的语义相关概念出现的意外退化（以下称为“干扰”）仍未得到充分表征，且评估方式不一致。本文提出I-CARE，这是一种将干扰形式化为生成式遗忘研究中首要研究对象的方法论。I-CARE并非提出新的基准或遗忘算法，而是为任务、指标和结果报告模板提供正式定义，从而支持在不同遗忘设置下对干扰进行系统性、可复现的研究。虽然我们的方法论设计旨在随着模型和遗忘算法的演进保持有效，将长期科学洞察与暂时的实证结果解耦，我们还提供了一个可行性演示，以……

    arXiv:2609.00003v1 Announce Type: new  Abstract: Machine unlearning studies the removal of knowledge from an AI model, making the system forget a concept it previously learned. Despite rapid progress in generative machine unlearning, the unintended degradation of semantically related concepts that should have been retained (henceforth, interference) remains poorly characterized and inconsistently evaluated. This paper introduces I-CARE, a methodology that formalizes interference as a first-class object of study in generative unlearning. Rather than proposing a new benchmark or unlearning algorithm, I-CARE provides formal definitions for tasks, metrics, and templates for reporting results, enabling the systematic and reproducible study of interference across unlearning settings. While our methodology is designed to remain valid as models and unlearning algorithms evolve, decoupling long-term scientific insight from transient empirical results, we present a feasibility demonstration with
    
[^182]: TopoCompress：基于图连接语义轨迹的长上下文压缩

    TopoCompress: Long Context Compression via Graph-Wired Semantic Trajectories

    [https://arxiv.org/abs/2608.30811](https://arxiv.org/abs/2608.30811)

    TopoCompress提出了一种无需训练、与模型无关的长上下文压缩框架，通过构建混合图连接连贯的语义片段并在其上传播查询引导的相关性分数，在五个长上下文基准任务上以更少的资源持续超越强大的压缩基线。

    

    长上下文压缩对于降低大语言模型推理的成本和延迟至关重要。然而，现有方法可能会割裂重要的证据信息，需要额外的训练或对齐，并且通常依赖目标模型才能实现有效压缩。我们提出了TopoCompress，这是一个无需训练且与模型无关的框架，通过选择连贯的语义片段来压缩长上下文。TopoCompress首先结合密集与词汇层面的查询相关性以及语义加速对每个片段进行评分。然后，它构建一个混合图，基于语义相似性和序列相邻性将各片段连接起来，并在图上传播查询引导的相关性分数。在五个长上下文任务——HotpotQA、2WikiMQA、MuSiQue、Qasper和MultiFieldQA-en——上，TopoCompress始终优于强大的压缩基线。值得注意的是，TopoCompress在使用4倍更少（的资源）的情况下达到了与最强基线相当的性能。

    arXiv:2608.30811v1 Announce Type: new  Abstract: Long-context compression is essential for reducing the cost and latency of large language model inference. However, existing methods can fragment important evidence, require additional training or alignment, and often depend on the target model for effective compression. We introduce TopoCompress, a training-free and model-agnostic framework that compresses long contexts by selecting coherent semantic spans. TopoCompress first scores each span using dense and lexical query relevance together with semantic acceleration. It then constructs a hybrid graph that connects spans based on semantic similarity and sequential adjacency, and propagates the query-guided relevance scores over the graph. Across five long-context tasks-HotpotQA, 2WikiMQA, MuSiQue, Qasper, and MultiFieldQA-en-TopoCompress consistently outperforms strong compression baselines. Notably, TopoCompress achieves performance comparable to the strongest baseline while using a 4x
    
[^183]: BiG-SURE——用于大语言模型语义不确定性与可靠性估计的二部图方法

    BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs

    [https://arxiv.org/abs/2608.30646](https://arxiv.org/abs/2608.30646)

    提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。

    

    可靠的不确定性估计是在安全关键场景中部署大语言模型（LLM）和视觉-语言模型（VLM）的关键前提，尤其是在模型参数不可访问（黑盒）的情况下。我们提出了BiG-SURE，一种基于跨温度语义一致性的不确定性估计器。该方法在保持语义不变的输入变换下，将低温采样得到的响应作为稳定的语义锚点，将高温采样得到的响应作为探针。随后，方法利用基于自然语言推理（NLI）的蕴含分数构建锚点-探针二部图，并通过该矩阵的归一化平方谱能量来定义置信度，不确定性则由其补集给出。这种基于二部图的语义不确定性与可靠性估计（SURE）分数，用于衡量高温探针是否与模型稳定的低温信念保持语义一致。我们在文本问答等任务上对BiG-SURE进行了评估。

    arXiv:2608.30646v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA,
    
[^184]: TACS：面向大语言模型越狱后缀优化的轨迹感知候选选择

    TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization

    [https://arxiv.org/abs/2608.29564](https://arxiv.org/abs/2608.29564)

    论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。

    

    基于梯度的越狱后缀优化方法通常通过保留当前损失最低的候选来更新后缀。我们证明，这种看似自然的设计本质上是短视的：在当前步骤代理指标下表现更好的候选，往往无法在搜索后期产生更好的越狱结果，这揭示了一种选择阶段的奖励破解现象。这表明，候选选择（而不仅仅是候选生成）是后缀优化中一个隐藏的瓶颈。为了解决这一问题，我们提出了TACS，一个用于越狱后缀优化的轨迹感知候选选择框架。TACS不再仅根据即时损失来选择候选，而是通过轨迹感知代理来增强每一步的评估，并利用参考策略正则化和判别器估计的卡方校正来稳定选择过程，从而鼓励那些在当前步骤之后仍然有效的选择。

    arXiv:2608.29564v1 Announce Type: new  Abstract: Gradient-based jailbreak suffix optimization methods typically update the suffix by retaining the candidate with the lowest current loss. We show that this seemingly natural design is fundamentally myopic: candidates that look better under the current-step proxy often fail to produce better jailbreak outcomes later in the search, revealing a form of selection-stage reward hacking. This suggests that candidate selection, rather than candidate generation alone, is a hidden bottleneck in suffix optimization. To address this issue, we propose \OURS{}, a trajectory-aware candidate selection framework for jailbreak suffix optimization. Instead of selecting candidates solely by their immediate loss, \OURS{} augments per-step evaluation with a trajectory-aware proxy and stabilizes selection with reference-policy regularization and a discriminator-estimated chi-squared correction, encouraging choices that remain effective beyond the current step.
    
[^185]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^186]: Hyper-Fold：通过超图建模探索蛋白质序列-几何学习的表达能力极限

    Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry Learning for Proteins via Hypergraph Modeling

    [https://arxiv.org/abs/2608.29207](https://arxiv.org/abs/2608.29207)

    提出 Hyper-Fold，通过超图建模将蛋白质序列内容与三维几何组织为超边，并利用分解为 K 个基算子的边条件化双线性矩阵值算子，以消息传递的计算代价逼近序列-几何学习的表达能力上限，在酶功能预测、折叠分类和配体结合位点检测等任务上表现出色。

    

    蛋白质结构建模建立在一个单一的计算原语之上：残基是什么（序列内容）与残基位于何处（三维几何）之间的相互作用。那么这一类层的表达能力极限是什么？我们证明，作用于内容-几何外积之上的完全双线性算子——即所有二阶交互的充分统计量——构成了表达能力的上限，而主流几何图神经网络（GNN）所采用的加性消息传递机制，在可证明的意义上对内容-几何耦合是“盲”的。随后，我们提出 Hyper-Fold，一种以消息传递的计算代价逼近该表达能力上限的秩-K 可分离卷积骨干网络：每个半径邻域被组织为一条序列超边和一条接触超边，并由一个边条件化的矩阵值算子进行调制，该算子被分解为 K 个可学习的基算子，其系数由几何信息生成。在酶功能预测、折叠类型分类以及配体结合位点检测等多项任务上，Hyper-Fold 展现了卓越的性能。

    arXiv:2608.29207v1 Announce Type: new  Abstract: Protein structure modeling rests on a single computational primitive: the interaction between what a residue is (sequence content) and where it sits (three-dimensional geometry). What is the expressive limit of this layer class? We show that the complete bilinear operator over content-geometry outer products--the sufficient statistic of all second-order interactions--is the expressive ceiling, while the additive message passing of mainstream geometric GNNs is provably blind to content-geometry binding. We then introduce Hyper-Fold, a rank-K separable convolutional backbone approaching this ceiling at message-passing cost: each radius neighborhood is organized into a sequence hyperedge and a contact hyperedge, modulated by an edge-conditioned matrix-valued operator factorized into K learned basis operators with geometry-generated coefficients. Across enzyme function prediction, fold classification, and ligand binding site detection, Hyper
    
[^187]: AutoScientist-Quant：面向量化投资自动化研究的自进化编码智能体

    AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment

    [https://arxiv.org/abs/2608.28632](https://arxiv.org/abs/2608.28632)

    提出AutoScientist-Quant框架，将量化研究建模为预算约束下的搜索问题，通过单一自进化控制器统一决策Alpha生成、因子库选择和模型调优，实现从假设到可部署策略的全流程自动化，并修复了评估流程中的前视偏差问题。

    

    大语言模型智能体能够发现Alpha因子，然而现有方法存在三个弱点：搜索过程无法在运行中自适应调整；自动化通常止步于Alpha生成，而因子库选择和模型选择仍需人工完成；Alpha发现过程可能通过循环反馈或代码问题窥探到测试窗口。我们提出AutoScientist-Quant，一个自进化的搜索过程，将量化研究视为一个受预算约束的搜索问题。单一控制器基于剩余预算对所有决策进行条件化，在每一轮决定是改进、组合、转向还是停止，选择扩展哪个节点，生成多少个Alpha，以及如何从共享记忆中检索历史轨迹。同一核心随后从因子库中进行选择并调整模型，实现了从假设到可部署策略的完整闭环。我们还审查了从先前工作复用的评估流程，修复了两个前视偏差问题，并保持反馈窗口与测试窗口互不相交。

    arXiv:2608.28632v1 Announce Type: new  Abstract: Large language model agents can discover alphas, yet current methods have three weaknesses. The search cannot adapt during the run, automation usually ends at alpha generation while library selection and model choice stay manual, and alpha discovery can read the test window through loop feedback or code problems. We present AutoScientist-Quant, a self evolving search process that regards quantitative research as one budgeted search problem. A single controller conditions every decision on the remaining budget, choosing at each round whether to improve, combine, pivot, or stop, which node to expand, how many alphas to generate, and how to retrieve past trajectories from the shared memory. The same core then selects from the library and tunes the model, closing the loop from hypothesis to deployable strategy. We also review the evaluation pipeline reused from prior work, fix two lookahead problems, and keep the feedback window disjoint fro
    
[^188]: 表演性隐私：差分隐私何时能最大化效用

    Performative Privacy: When Differential Privacy Maximizes Utility

    [https://arxiv.org/abs/2608.28198](https://arxiv.org/abs/2608.28198)

    该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。

    

    保护隐私的学习通常源于这样一种理念：保护用户数据可以维持信任，从而保持用户参与，进而在长期内提升效用。然而，这一论点迄今为止尚未被形式化。与此同时，表演性学习为研究部署行为会影响其后续观测数据的学习系统提供了一个框架。在本工作中，我们将这两种视角结合起来，提出了“表演性隐私”的概念，即数据泄露会降低未来的用户参与度。我们研究了一个简单模型：智能体反复贡献数据用于均值估计，但当其数据被泄露时可能会退出系统。隐私通过差分隐私机制来实现，从而在估计噪声与未来参与度之间形成权衡。通过对该动态过程的理论研究和数值实验，我们证明了在某些条件下，有限的隐私预算在长期内可以优于非隐私估计。

    arXiv:2608.28198v1 Announce Type: new  Abstract: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the
    
[^189]: 帧核方法用于多尺度算子学习

    The Frame Kernel Method for Multiscale Operator Learning

    [https://arxiv.org/abs/2608.25084](https://arxiv.org/abs/2608.25084)

    本文提出一种基于新型多尺度核帧逼近的算子学习方法，通过帧系数映射实现多尺度分解，在替代建模中比神经算子方法显著更精确。

    

    arXiv:2608.25084v1 公告类型：新 摘要：我们提出了一种用于多尺度偏微分方程（PDEs）数值求解器替代建模的天然多尺度算子学习方法。我们方法的主要新颖之处在于一种新型多尺度核帧函数逼近技术。利用这种新的核帧技术，我们将算子学习问题转化为学习输出函数帧系数作为输入函数帧系数的函数。随后，泛化步骤自动允许对输出函数进行多尺度分解。我们的方法适用于张量积网格和点云。我们提供了帧逼近的插值证明、误差估计和数值收敛率。然后，我们展示了该方法在固有多尺度PDE替代建模中的适用性。新的多尺度帧核方法比流行的神经算子方法显著更准确。

    arXiv:2608.25084v1 Announce Type: new  Abstract: We present a natively multiscale operator learning method for the surrogate modeling of (numerical solvers for) multiscale partial differential equations (PDEs). The primary novelty of our method lies in a novel multiscale kernel frame function approximation technique. Leveraging this new kernel frame technique, we cast the operator learning problem as one of learning frame coefficients of output functions as a function of frame coefficients of input functions. The generalization step then automatically allows for a multiscale decomposition of the output functions. Our method is applicable to both tensor-product grids and point clouds. We present interpolation proofs, error estimates, and numerical convergence rates for our frame approximation. We the demonstrate the applicability of our method for the surrogate modeling of inherently multiscale PDEs. The new multiscale frame kernel method is significantly more accurate than popular neur
    
[^190]: 公共中心几何与能量形式全共形区域的认证径向重建

    Common-Center Geometry and Certified Radial Reconstruction for Energy-Form Full Conformal Regions

    [https://arxiv.org/abs/2608.24964](https://arxiv.org/abs/2608.24964)

    本文证明了在对称性和凸性条件下，能量形式全共形预测区域呈星形，且对于幂距离在β≥1时具有确定性几何性质，同时指出候选评分凸性不足以保证连通性。

    

    本文研究了由经验能量形式成对评分生成的全共形预测（FullCP）区域的几何性质。仅凭候选评分的凸性并不能保证FullCP区域的连通性，即使候选评分是损失函数在其第一个参数上凸的经验平均值。通过直接展开留一评分，发现能量形式评分的每个训练点比较恰好是一个成对不相似性子水平条件。在对称性、常数对角线、对角线下界以及相关Fr\'echet型目标达到的条件下，每个比较区域都包含一个公共最小化器；当比较区域为凸时，非平凡精确共形区域因此关于该点呈星形。对于幂距离$\rho_\beta(x,y)=\|x-y\|^\beta$，当$\beta\ge1$时，这种确定性几何成立，而传统能量评分在$0<\beta<2$时是严格适当的。

    arXiv:2608.24964v1 Announce Type: cross  Abstract: This note studies the geometry of full conformal prediction (FullCP) regions generated by an empirical energy-form pairwise score. Candidate-score convexity alone does not guarantee connected FullCP regions, even when the candidate score is an empirical average of a loss convex in its first argument. Direct expansion of the leave-one-out scores shows that each training-point comparison for the energy-form score is exactly a pairwise-dissimilarity sublevel condition. Under symmetry, a constant diagonal, a diagonal lower bound, and attainment of the associated Fr\'echet-type objective, every comparison region contains a common minimizer; when the comparison regions are convex, the nontrivial exact conformal region is therefore star-shaped about that same point. For power distances $\rho_\beta(x,y)=\|x-y\|^\beta$, this deterministic geometry holds for $\beta\ge1$, while the conventional energy score is strictly proper for $0<\beta<2$. In 
    
[^191]: 压力测试机器遗忘算法

    Stress Testing Unlearning Algorithms

    [https://arxiv.org/abs/2608.22527](https://arxiv.org/abs/2608.22527)

    本文提出了WMDP++基准，通过主动测试已遗忘信息能否被强制提取、并评估模型在与遗忘内容语义相近的边界问题上的表现，弥补了现有机器遗忘评估的两大不足，为遗忘算法提供了更严格的压力测试。

    

    近期，机器遗忘，即从模型中移除特定训练数据影响的技术，受到了越来越多的关注。在大语言模型（LLMs）中，由于输入和输出的模糊性，遗忘任务尤其具有挑战性。因此，严格的评估对于衡量安全性与实用性以及推动遗忘方法的发展至关重要。我们指出了现有机器遗忘基准的两个关键缺陷：（1）它们没有主动测试已遗忘的信息是否仍然可以被强制提取出来；（2）它们未能评估模型在边界问题（即与已遗忘内容语义接近的良性查询）上的性能保持情况。在此，我们提出了WMDP++，这是WMDP基准的一个扩展，通过引入对已遗忘信息的定向提取测试以及对边界问题的系统性评估来弥补这些不足。WMDP++为遗忘方法的评估提供了一个更严格、更具信息量的基准。

    arXiv:2608.22527v2 Announce Type: replace  Abstract: Recently, machine unlearning, the removal of specific training data influence from a model, has gained increasing attention. In large language models (LLMs), unlearning is particularly challenging due to the ambiguity of inputs and outputs. Con- sequently, rigorous evaluation is critical for assessing both safety and utility, and for driving progress in unlearning meth- ods. We identify two key shortcomings in existing unlearning benchmarks: (1) they do not actively test whether unlearned information can still be forcibly extracted, and (2) they fail to evaluate performance preservation on boundary questions, be- nign queries that are semantically close to the unlearned con- tent. Here we introduce WMDP++, an extension of WMDP that addresses these gaps by incorporating targeted extrac- tion of unlearned information and systematic evaluation on boundary questions. WMDP++ provides a more stringent and informative benchmark for evaluati
    
[^192]: 超越稠密Adam状态：自适应对数空间量化用于内存高效优化器

    Beyond Dense Adam States: Adaptive Log-Space Quantization for Memory-Efficient Optimizers

    [https://arxiv.org/abs/2608.22322](https://arxiv.org/abs/2608.22322)

    本文提出自适应对数空间量化方法，针对不同优化器状态拓扑定制精度，以实现内存高效且保持更新准确性。

    

    arXiv:2608.22322v2 公告类型：替换  摘要：低精度优化器状态方法通常针对稠密Adam风格的一阶和二阶矩进行设计和评估。内存高效优化器偏离了这一设置：Adafactor对二阶矩进行分解，CAME添加了分解置信状态，而APOLLO在投影梯度空间中维护统计量。因此，等量的状态重建误差可能根据状态拓扑和更新语义导致不同的更新误差。我们首先在语言模型预训练的优化器状态轨迹中刻画了这种异质性。然后，我们引入了自适应对数空间（AL）量化，这是一种针对非负状态的块级表示，它根据每个块自适应非零范围，并强制执行精确零不变量$q = 0 \Leftrightarrow x = 0$。AL8和AL16与独立的符号动量编码和状态特定精度选择相结合，而不是对所有状态使用单一策略。在96次运行中，总计...

    arXiv:2608.22322v2 Announce Type: replace  Abstract: Low-precision optimizer-state methods are commonly designed and evaluated for dense Adam-style first and second moments. Memory-efficient optimizers depart from this setting: Adafactor factorizes second moments, CAME adds factored confidence states, and APOLLO maintains statistics in a projected gradient space. Consequently, an equal amount of state reconstruction error can induce different update errors depending on state topology and update semantics. We first characterize this heterogeneity in optimizer-state traces from language model pre-training. We then introduce Adaptive Log-Space (AL) quantization, a block-wise representation for non-negative states that adapts its nonzero range per block and enforces the exact-zero invariant $q = 0 \Leftrightarrow x = 0$. AL8 and AL16 are combined with independent signed-momentum encodings and state-specific precision choices rather than a single policy for every state.   Across 96 runs tot
    
[^193]: MRMAD：一个用于评估大型音频-语言模型声学退化感知的多轮多音频基准

    MRMAD: A Multi-Round Multi-Audio Benchmark for Evaluating Acoustic Degradation Perception in Large Audio-Language Models

    [https://arxiv.org/abs/2608.22236](https://arxiv.org/abs/2608.22236)

    提出了MRMAD，一个多轮多音频退化基准，通过多轮对话评估大型音频-语言模型对音频质量退化的识别、严重程度比较与跨轮次一致性推理能力。

    

    大型音频-语言模型（LALMs）在理解语音、音乐和一般声音事件方面已展现出可喜的进展，但它们对音频信号如何退化的推理能力仍未得到充分探索。现有基准主要评估语义理解、事件识别或高阶音频推理，留下一个基本问题尚未解答：LALMs是否理解音频质量之间的差异？我们提出了MRMAD，一个用于评估LALMs音频退化感知与理解的多轮多音频退化基准。MRMAD涵盖语音、音乐和一般声音，并将评估构建为跨多个音频输入的多轮对话，要求模型识别退化类型、比较严重程度，并感知跨轮次的损坏变化。与当前单轮音频-语言基准不同，MRMAD评估LALMs能否在获得新证据时保持一致的退化假设并进行比较。

    arXiv:2608.22236v2 Announce Type: replace-cross  Abstract: Large audio-language models (LALMs) have shown promising progress in understanding speech, music, and general sound events, yet their ability to reason about how audio signals are degraded remains underexplored. Existing benchmarks primarily evaluate semantic understanding, event recognition, or high-level audio reasoning, leaving a basic question unanswered: Do LALMs understand the differences in audio quality? We introduce MRMAD, a Multi-Round Multi-Audio Degradation benchmark for evaluating audio degradation perception and understanding in LALMs. MRMAD spans speech, music, and sound, and frames evaluation as multi-turn dialogues across multiple audio inputs, requiring models to identify types of degradation, compare severity, and perceive corruption changes across turns. Unlike current single-turn audio-language benchmarks, MRMAD evaluates whether LALMs can maintain consistent degradation hypotheses with new evidence and com
    
[^194]: ToSCA：基于对话代理时间与策略抽象的层次强化学习

    ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents

    [https://arxiv.org/abs/2608.21969](https://arxiv.org/abs/2608.21969)

    本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。

    

    人类在日常互动和思考中具有多个层次的时间抽象能力，例如概念感知和策略规划。受此启发，我们为对话代理提出了一种两级层次强化学习（RL）框架，弥合了以往基于词元级别或话语级别RL方法之间的差距。该框架基于两级MDP开发，其中词元级别的响应解码依赖于话语级别的动作，即显式文本策略。基于理论推导和效率考虑，我们使用DQN求解高层评论家，使用PPO求解低层演员-评论家。为进一步缓解奖励稀疏性并促进收敛，我们还设计了双粒度奖励机制，将话语级别的满意度评分与词元级别的内在动机和K-L惩罚相结合。在日常对话和情感支持对话上的实验表明，所提方法优于现有基线。

    arXiv:2608.21969v1 Announce Type: new  Abstract: Humans have multiple levels of temporal abstractions on daily interaction and thinking, such as concept perception and strategic planning. Inspired by this nature, we propose a two-level hierarchical reinforcement learning (RL) framework for conversational agents, bridging the gap between previous token-level or utterance-level RL methods. Developed on a two-level MDP, the token-level response decoding is conditioned on the utterance-level action, the explicit textual strategies. Based on theoretical derivation and efficiency consideration, we use DQN to solve the high-level critic and PPO to solve the low-level actor-critic. To further alleviate the reward sparsity and facilitate the convergence, we also design the dual-granularity reward mechanism, in which the utterance-level satisfaction score is integrated with token-level intrinsic motivation and K-L penalty. Experiments on both daily and emotional support conversations show that o
    
[^195]: 无金标准标签下AI生成数据的去偏推断：通过多重不完美测量进行识别

    Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements

    [https://arxiv.org/abs/2608.18294](https://arxiv.org/abs/2608.18294)

    本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。

    

    越来越多的学者使用AI来测量变量，并将其纳入后续的下游分析。尽管AI测量的变量通常被视为无误差观测，但忽略自动化测量中的预测误差会导致下游分析中的显著偏差和无效置信区间，即使AI测量准确度很高（例如超过90%）。现有的解决方案，如基于设计的有监督学习和预测支持推断，将基于AI的易错测量与金标准标签相结合，但在某些应用领域中，获取金标准标签可能成本高昂且困难。在本文中，我们提出了多重不完美测量的去偏推断（DMM），这是一个结合多个易错AI测量以实现无需金标准标签的有效下游推断的框架。基于CP分解的既有成果，DMM假设这些测量是独立的。

    arXiv:2608.18294v1 Announce Type: cross  Abstract: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.   In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent 
    
[^196]: 基于状态条件转移采样的非参数时空轨迹预测

    Non-Parametric Spatiotemporal Trajectory Prediction via State-Conditioned Transition Sampling

    [https://arxiv.org/abs/2608.14349](https://arxiv.org/abs/2608.14349)

    本文提出一种零参数、无需训练的轨迹预测方法，通过状态条件转移采样和核检索，在数据稀缺时性能优于大型Transformer模型，并能用极少历史数据适应新区域。

    

    arXiv:2608.14349v1 公告类型：新 摘要：我们提出了一种无需训练的轨迹预测方法，用于多模态轨迹预测，其精度可与一个5700万参数的Transformer模型相媲美，但无需GPU且零学习参数。该方法构建了一个历史状态到下一位置对的转移表，并使用基于空间邻近性、方位、速度和时间上下文的乘积核来检索邻居。两种推理模式在此共享表示上运行：多样性惩罚采样生成覆盖不同合理路线的轨迹，而束搜索则找到最高似然路径。在TrAISformer基准测试（丹麦海事AIS）上，我们的方法在完整数据可用性下达到了竞争性精度，并在数据稀缺情况下显著优于Transformer——在训练数据降至10%时仍保持稳定，而TrAISformer在此情况下性能急剧退化。这使得该方法能够从少一个数量级的历史数据中部署到新的地理区域。

    arXiv:2608.14349v1 Announce Type: new  Abstract: We present a training-free method for multi-modal trajectory prediction that achieves comparable accuracy to a 57M-parameter transformer while requiring no GPU and zero learned parameters. The method builds a transition table of historical state-to-next-position pairs and retrieves neighbors using a product kernel over spatial proximity, bearing, speed, and temporal context. Two inference modes operate over this shared representation: diversity-penalized sampling produces trajectories covering distinct plausible routes, while beam search finds the highest-likelihood path. On the TrAISformer benchmark (Danish Maritime AIS), our method achieves competitive accuracy at full data availability and dramatically outperforms the transformer in data-scarce regimes---remaining stable down to 10% of training data where TrAISformer degrades catastrophically. This enables deployment in new geographic regions from an order of magnitude less historical
    
[^197]: 方向，而非幅度：合并语言模型中任务向量干扰的因果结构

    Orientation, not magnitude: the causal structure of task-vector interference in merged language models

    [https://arxiv.org/abs/2608.11797](https://arxiv.org/abs/2608.11797)

    本文通过精确分解和干预实验证明，合并语言模型中的任务向量干扰由方向（而非幅度）因果决定，该方向是前向传递的吸引子，沿此方向擦除可有效去除干扰。

    

    arXiv:2608.11797v1 公告类型：新  摘要：通过任务算术进行模型合并起初有效，但随后失效，领域内通常用幅度来诊断原因：逐层表示偏差、跨任务线性偏离、参数重叠。通过因子账本跟踪合并LLM的精确逐层交叉项并直接干预，我们发现幅度作为诊断轴是不够的，且在不同模型家族间不一致。对逐层通量的精确分解表明，其由现有交叉项的放大传输主导（两个家族中约占65-70%，每个后期块的增益>1），而擦除该交叉项会被传播所抵消——重建至其范数的99%，余弦相似度为0.99——除非在接近输出处应用；用六种起始位移进行的盆地测试确立了携带方向为前向传递的吸引子。该方向具有因果承载性：沿该方向的擦除以剂量依赖方式去除表达干扰，并在精确擦除处饱和，而

    arXiv:2608.11797v1 Announce Type: new  Abstract: Model merging by task arithmetic works until it doesn't, and the field diagnoses why with magnitudes: layerwise representation bias, deviations from cross-task linearity, parameter overlap. Tracking the exact layerwise cross-term of merged LLMs through a factorial ledger and intervening on it directly, we find magnitude insufficient - and inconsistent across model families - as a diagnostic axis. An exact decomposition of the layerwise flux shows it is dominated by amplifying transport of the existing cross-term (~65-70% in both families, gain >1 per late block), and erasing the term is undone by propagation - rebuilt to 99% of its norm at cosine 0.99 - unless applied near the output; a basin test with six starting displacements establishes the carried direction as an attractor of the forward pass. That direction is causally load-bearing: erasure along it removes expressed interference dose-dependently and saturates at exact erasure, whi
    
[^198]: 无对数因子的矩不等式与一致稳定算法的泛化界

    Logarithmic-Free Moment and Generalization Bounds for Uniformly Stable Algorithms

    [https://arxiv.org/abs/2608.09870](https://arxiv.org/abs/2608.09870)

    该论文去除了一致稳定算法泛化界中多余的对数因子 $\log n$，证明了无对数的矩不等式，从而肯定地回答了Bousquet等人（2020）提出的公开问题。

    

    一致稳定性是控制学习算法泛化误差的经典工具。Bousquet、Klochkov和Zhivotovskiy（2020）证明了该问题可以归约为关于独立随机变量的弱相互作用函数之和的矩不等式。他们的界包含一个额外的因子 $\log n$，并提出能否去除该因子的疑问。我们对这个上界问题给出了肯定的回答。更具体地，设 $Z=(Z_1,\ldots,Z_n)$ 的各坐标相互独立，且 $g_i(Z)$ 满足 $\mathbb{E}[g_i(Z)\mid Z_{-i}]=0$，$|\mathbb{E}[g_i(Z)\mid Z_i]|\le M$，对每个 $i=1,\dots,n$ 成立，其中 $Z_{-i}$ 表示除 $Z_i$ 之外的所有坐标。进一步假设改变任意坐标 $Z_j$（$j\neq i$）至多使 $g_i$ 改变 $\beta$，我们证明，对每个 $p\ge2$，有 $\left\|\sum_{i=1}^n g_i(Z)\right\|_p \le 16pn\beta+M\sqrt{2pn}$。

    arXiv:2608.09870v2 Announce Type: replace-cross  Abstract: Uniform stability is a classical tool for controlling the generalization error of a learning algorithm. Bousquet, Klochkov, and Zhivotovskiy (2020) showed that the problem can be reduced to a moment inequality for a sum of weakly interacting functions of independent random variables. Their bound contains an additional factor $\log n$, and they asked whether this factor can be removed. We answer this upper-bound question affirmatively. More specifically, let $Z=(Z_1,\ldots,Z_n)$ have independent coordinates and let $g_i(Z)$ satisfy $\mathbb E[g_i(Z)\mid Z_{-i}]=0, \ \left| \mathbb E[g_i(Z)\mid Z_i]\right|\le M, \ \text{for every } i = 1, \dots, n, $ where $Z_{-i}$ denotes all coordinates except $Z_i$. Assume additionally that changing any coordinate $Z_j$, $j\neq i$, changes $g_i$ by at most $\beta$, we prove that, for every $p\ge2$, for every $p\ge2$, $$ \left\| \sum_{i=1}^n g_i(Z)\right\|_p \le 16pn\beta+M\sqrt{2pn}. $$ This r
    
[^199]: 面向逆散射成像的坐标残差物理驱动神经网络

    Coordinate-Residual Physics-Driven Neural Network for Inverse Scattering Imaging

    [https://arxiv.org/abs/2608.09382](https://arxiv.org/abs/2608.09382)

    提出坐标残差物理驱动神经网络CRPDNN，用归一化空间坐标与残差卷积网络表示复对比度分布，并通过测量场与预测场的一致性约束实现稳定且无需区域选择的加速三维电磁逆散射成像。

    

    电磁逆散射是一个非线性且不适定的计算成像问题，由于测量限制、噪声以及高昂的计算成本，精确重建极具挑战性，尤其是在三维成像中。尽管物理驱动神经网络减少了对标注训练数据的依赖，但现有的加速PDNN框架通常依赖基于初步重建的区域选择，当所选区域不准确时可能会引入不稳定性。本文提出了一种用于三维电磁逆散射的坐标残差物理驱动神经网络（CRPDNN）。CRPDNN利用归一化空间坐标和残差卷积网络来表示未知的复对比度分布，其参数通过强制测量散射场与模型预测散射场之间的一致性来进行优化。与现有的子区域加速PDNN方法不同，该网络（摘要在此处截断）。

    arXiv:2608.09382v2 Announce Type: replace-cross  Abstract: Electromagnetic inverse scattering is a nonlinear and ill-posed computational imaging problem, where accurate reconstruction is challenging due to measurement limitations, noise, and high computational costs, especially for 3-D imaging. Although physics-driven neural networks (PDNNs) reduce the dependence on labeled training data, existing accelerated PDNN frameworks often rely on preliminary reconstruction-based region selection, which may introduce instability when the selected region is inaccurate. In this paper, a coordinate-residual physics-driven neural network (CRPDNN) is proposed for 3-D electromagnetic inverse scattering. CRPDNN represents the unknown complex contrast distribution using normalized spatial coordinates and a residual convolutional network, whose parameters are optimized by enforcing consistency between the measured and model-predicted scattered fields. Unlike existing subregion-accelerated PDNN approache
    
[^200]: 推理大语言模型中的测试时扩展：推理机制、评估与可复现性

    Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility

    [https://arxiv.org/abs/2608.04001](https://arxiv.org/abs/2608.04001)

    该论文将大语言模型的测试时扩展形式化为隐式前缀树上的预算约束推理，系统区分了三种推理机制（单轨迹顺序扩展、叶节点级扩展与前缀级扩展），并主张以完整推理系统作为评估对象，以提升研究结果的可比性与可复现性。

    

    大语言模型可以通过更多的推理时计算来解决更难的推理问题。然而，“测试时扩展”这一术语涵盖了多种推理算法：沿单条轨迹延长思考过程、采样完整候选答案并通过投票或验证进行聚合，以及在部分状态上进行搜索。这些算法在统计结构、计算需求和失败模式上各不相同。将它们视为在标量“预算”下可以互换，或者在报告准确率时不指明推理协议，会使研究结果难以跨研究进行比较。我们从三个维度研究测试时扩展。首先，我们将其形式化为在自回归模型隐式前缀树上进行的预算约束推理，并区分单轨迹顺序扩展、带终端归约的叶节点级扩展以及前缀级扩展。其次，我们将完整的推理系统作为评估对象，并区分（注：原文摘要在此处被截断）

    arXiv:2608.04001v2 Announce Type: replace-cross  Abstract: Large language models can solve harder reasoning problems with more inference-time compute. The term "test-time scaling," however, covers several inference algorithms: extending deliberation along one trajectory, sampling completed candidates and aggregating them by voting or verification, and searching over partial states. These algorithms differ in statistical structure, compute requirements, and failure modes. Treating them as interchangeable under a scalar "budget," or reporting accuracy without specifying the inference protocol, makes results difficult to compare across studies. We study test-time scaling along three axes. First, we formalize it as budgeted inference over the implicit prefix tree of an autoregressive model and distinguish single-trajectory sequential scaling, leaf-level scaling with terminal reduction, and prefix-level scaling. Second, we treat the full inference system as the evaluated object and separate
    
[^201]: 字段感知的智能体技能检索

    Field-Aware Agent Skill Retrieval

    [https://arxiv.org/abs/2608.02880](https://arxiv.org/abs/2608.02880)

    该论文将智能体技能建模为结构化的多字段对象，对每个字段独立计算稀疏与稠密相似度并用均匀权重或小型MLP融合，从而显著提升终身学习智能体的技能检索效果。

    

    随着终身学习智能体不断积累日益增长的技能库，如何检索到正确的技能成为越来越重要的瓶颈。目前大多数技能检索方法将每个技能视为一个扁平文档，即通过拼接名称、描述和正文等字段来处理。然而，技能本质上是结构化的多字段对象，每个字段提供了关于技能何时以及如何被使用的不同信息。在这项工作中，我们研究了保留这种结构是否能改善技能检索。我们将每个技能表示为其独立的组成部分，并对每个字段独立计算稀疏与稠密相似度，从而获得一种天然张量化、字段感知的技能库表示。随后，我们通过均匀权重或一个小型可学习的MLP来组合这些字段级别的分数。在SkillRet和SRA-Bench两个不同的技能检索基准上，我们发现保持字段分离能够提升技能检索性能。

    arXiv:2608.02880v3 Announce Type: replace-cross  Abstract: As lifelong learning agents accumulate lifelong growing skill banks, retrieving the correct skill becomes an increasingly important bottleneck. Most current skill retrieval methods treat each skill as one flat document by concatenating fields such as the name, description, and body. However, skills are naturally structured, multi-field objects, where each field provides different information about when and how the skill should be used. In this work, we study whether preserving this structure improves skill retrieval. We represent each skill as its separate components, and compute sparse and dense similarities for each field independently, exposing a naturally tensorized, field-aware representation of the skill bank. We then combine these field-level scores either with uniform weights or with a small learned MLP. Across two different skill retrieval benchmarks, SkillRet and SRA-Bench, we find that keeping fields separate improve
    
[^202]: 锁定评估面：CRISPRi扰动效应预测中的迁移失败与采样深度纠缠

    Locked Evaluation Surfaces: Transfer Failure and Sampling-Depth Entanglement in CRISPRi Perturbation-Effect Prediction

    [https://arxiv.org/abs/2608.00152](https://arxiv.org/abs/2608.00152)

    该论文在锁定且预注册的评估协议下评估冻结的Geneformer表示，发现其在虚拟细胞挑战赛（VCC）分布内数据上具有显著超越随机特征对照的预测信息量，但在零样本跨筛选迁移中失败，并揭示了迁移失败与采样深度等设计因素之间的纠缠。

    

    预测保留的目标基因如何响应CRISPRi扰动，以及此类预测能否在不同生物筛选之间迁移，是很难评估的：一种表示可能在单个筛选内具有信息量，却在跨筛选时失效，同时终点定义和采样深度等设计因素在不同数据集之间也存在差异。我们在一个锁定且预注册的协议下评估冻结的Geneformer表示：分类头与模型选择在测试评估之前冻结，外部结果标签在最终揭盲前保密，且支配分析的决策在其所支配的评估之前即已固定。在虚拟细胞挑战赛（VCC）的分布内数据上，该冻结表示携带可测量的预测信息，超过维度匹配的随机特征对照（ΔR² = +0.1645，95%置信区间 [+0.1375, +0.1920]），满足了在解释迁移之前所要求的预注册信息量门槛。随后它在零样本……[摘要在此处被截断]

    arXiv:2608.00152v2 Announce Type: replace  Abstract: Predicting how held-out target genes respond to CRISPRi perturbation, and whether such predictions transfer across biological screens, is hard to evaluate: a representation can be informative within one screen yet fail across screens, while endpoint definitions and design factors such as sampling depth differ between datasets. We evaluate a frozen Geneformer representation under a locked, pre-registered protocol, with heads and model selection frozen before test evaluation, external outcome labels withheld until final unblinding, and analysis-governing decisions fixed before the evaluations they govern. In-distribution on the Virtual Cell Challenge (VCC), the frozen representation carries measurable predictive information beyond a dimension-matched random-feature control (Delta R^2 = +0.1645, 95% CI [+0.1375, +0.1920]), satisfying the pre-registered informativeness gate required before interpreting transfer. It then fails zero-shot t
    
[^203]: S-CEReBrO：突破连续脑电监测中的记忆瓶颈

    S-CEReBrO: Breaking the Memory Barrier in Continuous EEG Monitoring

    [https://arxiv.org/abs/2607.27913](https://arxiv.org/abs/2607.27913)

    S-CEReBrO通过新颖的窗口交替注意力机制将注意力计算分解为固定大小的时空窗口，实现恒定的KV缓存内存占用，可处理比全自注意力长100倍的信号，从而突破了连续脑电监测中的内存瓶颈。

    

    基础模型通过从海量无标注数据中学习可泛化的表示，为脑电图（EEG）分析提供了一种有前景的范式。然而，基于Transformer的架构面临一个关键瓶颈：全局注意力机制将注意力记忆状态与信号时长耦合在一起，导致在连续监测过程中发生内存溢出。为解决这一问题，我们提出了S-CEReBrO（流式CEReBrO），这是专为连续监测设计的CEReBrO架构的演进版本。我们新颖的窗口交替注意力机制将注意力计算分解为固定大小的时空窗口，由于只有活动窗口需要驻留注意力图，从而保证了KV缓存内存的恒定。实证扩展分析证实，窗口交替注意力能够处理的信号长度是全自注意力的100倍，是低秩线性注意力的3倍。与低秩线性注意力在长…（原文摘要在此处截断）

    arXiv:2607.27913v2 Announce Type: replace  Abstract: Foundation models offer a promising paradigm for Electroencephalography (EEG) analysis, leveraging generalizable representations from vast unlabeled datasets. Yet, Transformer-based architectures face a critical bottleneck: global attention mechanisms couple the attention memory state to the signal duration, causing memory overflow during continuous monitoring. To address this, we introduce S-CEReBrO (Streaming CEReBrO), an evolution of the CEReBrO architecture designed for continuous monitoring. Our novel Windowed Alternating Attention mechanism factorizes attention computation into fixed-size spatiotemporal windows, guaranteeing constant KV cache memory as only the active window requires resident attention maps. Empirical scaling analysis confirms that windowed alternating attention can process signals 100X longer than full self-attention and 3X longer than low-rank linear attention. Compared to low-rank linear attention on long co
    
[^204]: 退格键作为自然实验：帕金森病错误后选择性运动损伤的加速失效时间模型

    Backspace as a Natural Experiment: An Accelerated Failure Time Model of Selective Post-Error Motor Impairment in Parkinsons Disease

    [https://arxiv.org/abs/2607.24796](https://arxiv.org/abs/2607.24796)

    本研究将退格键事件作为自然纠错情境，发现错误后停顿时长（而非错误前打字不稳定性）与帕金森病严重程度显著相关，表明被动键盘监测可选择性捕捉PD患者的错误后运动恢复损伤。

    

    帕金森病（PD）会选择性地损害运动控制的不同阶段。在公开的neuroQWERTY MIT-CSXPD数据集（n=57名受试者，其中27名PD患者带有UPDRS-III评分）中，我们将退格键事件作为自然的纠错情境，测试被动采集的击键时间数据能否将基于变异性的错误前监测信号与基于速度的错误后恢复信号区分开来。错误前的打字不稳定性与PD严重程度无关（r=-0.072, p=0.721），而错误后停顿时长则与PD严重程度显著相关（r=+0.656, p=0.0002；受试者层面OLS p=1.2x10^-4, n=27，作为考虑受试者内事件聚类的主要分析）。包含随机受试者截距的混合效应对数正态模型在保留完整事件级统计效力的同时证实了这一结果（系数=0.0250, p=1.2x10^-4, n=1,563个事件），尽管采用了不相关的估计策略，其结果与受试者层面的OLS高度吻合。错误检测延迟也与UPDRS-III相关（r=+0.660），进一步证实了……（摘要原文在此处截断）

    arXiv:2607.24796v2 Announce Type: replace-cross  Abstract: Parkinson's disease (PD) selectively impairs distinct stages of motor control. Using backspace events as natural error-correction episodes in the public neuroQWERTY MIT-CSXPD dataset (n=57 subjects, 27 PD with UPDRS-III scores), we test whether passively-collected keystroke timing dissociates a variability-based pre-error monitoring signal from a speed-based post-error recovery signal. Pre-error typing instability does not track PD severity (r=-0.072, p=0.721), while post-error pause duration does (r=+0.656, p=0.0002; subject-level OLS p=1.2x10^-4, n=27, primary analysis given within-subject event clustering). A mixed-effects log-normal model with a random subject intercept confirms this while retaining full event-level power (coef=0.0250, p=1.2x10^-4, n=1,563 events), closely matching the subject-level OLS despite an unrelated estimation strategy. Error-detection latency also correlates with UPDRS-III (r=+0.660), confirming th
    
[^205]: 基于精确线性代数的更浅ReLU网络表示

    Shallower ReLU Network Representations via Exact Linear Algebra

    [https://arxiv.org/abs/2607.21651](https://arxiv.org/abs/2607.21651)

    本文通过对称性约减后有理数域线性方程组的精确计算机辅助搜索，将两层ReLU网络可精确表示的最大值函数维度从 n≤5 提升至 n≤12，并借助结构化首隐藏层的递归代入得到对数深度的精确表示。

    

    我们研究ReLU网络精确表示分段线性函数所需的深度，并特别关注最大值函数。该问题近期在机器学习和理论计算机科学文献中均受到广泛关注。我们证明：对于每个 n ≤ 12，最大值函数 max_n(x) = max{x_1, …, x_n} 都可以用两个隐藏层精确表示。此前该结论仅在 n ≤ 5 的情形下被证明 [Bakaev, Brunck, Hertrich, Stade, Yehudayoff, STOC'26]。我们通过在候选解空间中进行精确的计算机辅助搜索得到这些构造：经过对称性约减后，问题化归为一个有理数域 Q 上的有限线性方程组，其任意解均可给出最大值函数的有效表示。所得构造的第一隐藏层具有结构化形式，从而支持向更深网络的递归代入。由此我们得到 max_n 的精确ReLU表示，其所需深度至多为 ⌈log_6(n/2)⌉……

    arXiv:2607.21651v2 Announce Type: replace  Abstract: We study the depth required by ReLU networks to exactly represent piecewise linear functions, focusing specifically on the maximum function. This problem has recently received significant attention in both the ML and TCS literature. We prove that $\max_n(x)=\max\{x_1,\ldots,x_n\}$ is exactly representable with two hidden layers for every $n\leq 12$. Previously, this was only known up to $n\leq5$ [Bakaev, Brunck, Hertrich, Stade, Yehudayoff, STOC'26]. We obtain our constructions through an exact computer-assisted search within a space of candidate solutions: After a symmetry reduction, we obtain a finite system of linear equations over $\mathbb{Q}$ such that any solution yields a valid representation of the maximum function. The resulting constructions have a structured first hidden layer, which enables recursive substitution into deeper networks. This yields an exact ReLU representation of $\max_n$ with at most $\lceil \log_6(n/2) \r
    
[^206]: 一个会自我教学的分类器：用于动态文档分类的自我改进冻结门控训练（SIFT）

    A Classifier That Teaches Itself: Self-Improving, Frozen-gate Training (SIFT) for Dynamic Document Classification

    [https://arxiv.org/abs/2607.18358](https://arxiv.org/abs/2607.18358)

    SIFT提出了一种自改进的动态文档分类服务：用廉价的SPLADE+LightGBM流水线处理分类，仅将低置信度页面交给LLM裁判，其判定结果回流标注语料库持续教导廉价模型，从而免去前期标注工作并让准确率随使用不断提升。

    

    文档分类在实验室里是已被解决的问题，在企业中却是尚未解决的问题。阻碍通常并非模型架构，而是必须在建模之前完成的标注工程，以及机构对于让已存在的模型自我再训练的担忧。我们提出了SIFT（Self-Improving, Frozen-gate Training，自我改进冻结门控训练），一种动态分类器服务，同时攻克这两个问题。SIFT通过一条刻意设计得廉价、基于CPU的流水线来提供分类服务——由SPLADE稀疏编码器连接LightGBM分类头——并仅将低置信度的少数页面升级至LLM裁判。裁判的判定结果会被写回标注语料库，因此昂贵的模型持续地教导廉价的模型：升级率不断下降，语料库从生产流量中自然增长而非依赖前期标注工作，准确率随使用而持续复合提升。接入一个新的文档系列只需要一个声明式包、标签空间、锚定短语，以及……

    arXiv:2607.18358v2 Announce Type: replace  Abstract: Document classification is a solved problem in the laboratory and an unsolved one in the enterprise. The blocker is rarely model architecture; it is the labeling project that must precede a model and the institutional fear of letting a model retrain itself once one exists. We present SIFT (Self-Improving, Frozen-gate Training), a dynamic classifier service, which attacks both. SIFT serves classification from a deliberately cheap, CPU-bound pipeline, a SPLADE sparse encoder feeding a LightGBM head, and escalates only the low-confidence minority of pages to an LLM judge. The judge's verdicts are written back into a labeled corpus, so the expensive model continuously teaches the cheap one: the escalation rate falls, the corpus grows from production traffic rather than from an up-front annotation effort, and accuracy compounds with use. Onboarding a new document family requires only a declarative bundle, label space, anchor phrases, and 
    
[^207]: 对齐微调如何塑造大语言模型中谄媚性及相关线索诱导偏差的表示？

    How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?

    [https://arxiv.org/abs/2607.18114](https://arxiv.org/abs/2607.18114)

    该研究发现大语言模型对谄媚性等线索诱导偏差的敏感性主要源于对齐微调而非预训练，且对齐模型中每种偏差都存在一个可被解码和干预的线性表示方向，可用于恢复无偏答案。

    

    现代大语言模型对于输入提示中一些出奇简单且无关紧要的变化异常敏感：一句随意的暗示、一个标注错误的少样本示例，或是一个伪造的先前助手回合，常常会使原本正确的答案发生翻转。我们研究了这种敏感性——涵盖谄媚性及相关线索诱导偏差——存在于模型内部的位置。我们在五个模型家族和七种偏差类型上，从隐藏状态中提取每种偏差的方向，并通过三种方法对其进行三角验证：探针分析、留一数据集（LODO）迁移以及因果干预。研究发现，这种敏感性主要由对齐微调而非预训练所塑造：预训练基础模型通常对这些偏差的屈从程度要低得多，其激活中除问题内容之外的线索特定信号也弱得多。在对齐后的模型中，每种偏差都存在一个连贯的线性方向，我们既可以对其进行解码，也可以沿其进行干预，从而在所有模型家族中恢复无偏的答案。

    arXiv:2607.18114v2 Announce Type: replace-cross  Abstract: Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out (LODO) transfer, and causal intervention. The susceptibility is largely shaped by alignment tuning rather than pretraining: pretrained base models generally cave much less to these biases, and their activations carry much weaker cue-specific signal beyond question content. Within aligned models, each bias has a coherent linear direction that we can both decode and steer along, recovering the unbiased answer across every fam
    
[^208]: 最终检查点并不足够：沿训练轨迹分析潜在推理的忠实性

    Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories

    [https://arxiv.org/abs/2607.06648](https://arxiv.org/abs/2607.06648)

    该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。

    

    潜在推理在连续的隐状态中执行多步推理，有望实现更紧凑、更高效的推理。然而，这些不透明的状态引发了一个忠实性问题：潜在推理步骤是否真正驱动了最终答案的生成。先前的工作在选定的检查点上研究这一问题，并报告了若干不忠实的行为。这种端点视角使得忠实性证据在训练过程中如何演变这一问题未被考察。我们利用经过验证的反事实编辑以及对潜在推理状态的干预，在整个训练过程中追踪行为层面的证据和基于激活的证据。我们发现，高任务准确率可能与低反事实响应性共存：随着准确率的提升，响应性反而可能下降，且不同的潜在推理方法遵循各自不同的轨迹。在ProsQA上，输出对范数噪声替换的敏感性与反事实响应性一同下降，尽管该结果取决于替换的具体方式。

    arXiv:2607.06648v2 Announce Type: replace-cross  Abstract: Latent reasoning performs multi-step inference in continuous hidden states, promising more compact and efficient reasoning. However, these opaque states raise a question of faithfulness: whether the latent reasoning steps drive the final answer. Prior work studies this question at selected checkpoints and reports several unfaithful behaviors. This endpoint view leaves how evidence of faithfulness evolves during training unexamined. We track behavioral and activation-based evidence across training using verified counterfactual edits and interventions on the latent reasoning states. We find that high task accuracy can coexist with low counterfactual responsiveness: as accuracy improves, responsiveness can decline, and different latent reasoning approaches follow distinct trajectories. On ProsQA, output sensitivity to norm-noise replacement declines alongside counterfactual responsiveness, although the result depends on the replac
    
[^209]: 大语言模型能否想象二元道德困境之外的替代方案？

    Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?

    [https://arxiv.org/abs/2606.31213](https://arxiv.org/abs/2606.31213)

    该论文提出MoralAltDataset数据集，通过在307个二元道德困境中引入折中和重构的替代选项，发现当替代方案可用时人类与15个LLM的道德选择分布均发生显著转变且一致性增强，但存在关键差异——LLM明显偏好GPT-5创作的替代方案，而人类的选择不受创作来源影响，揭示了机器与人类在“想象道德替代方案”能力上的差距。

    

    随着大语言模型（LLM）越来越多地充当道德顾问和道德智能体，它们必须应对相互竞争的价值观之间的冲突。然而，以往关于道德困境的研究忽视了人类道德认知的一个核心方面：在给定选项之外想象替代方案。我们提出了MoralAltDataset数据集，其中包含307个顾问型和人机交互型智能体困境，并为其补充了折中方案与重新构建的替代选项。我们在二元选项和四选项两种设置下比较了人类与LLM的判断。在人类被试和15个LLM中，两种设置下的总体道德选择分布存在显著差异，且折中方案往往比两个原始二元选项中的任何一个都更受青睐。结果显示出价值观的转变，以及在替代方案上人类与LLM之间更强的一致性。按创作来源分层的结果揭示了一种描述性差距：人类选择替代方案的比率在不同来源之间相似，而LLM则明显更频繁地选择由GPT-5创作的替代方案。随后我们比较了人类与……（原文摘要在此处被截断）

    arXiv:2606.31213v2 Announce Type: replace-cross  Abstract: As LLMs increasingly serve as moral advisors and agents, they must address conflicts between competing values. Yet prior work on moral dilemmas overlooks a central aspect of human moral cognition: imagining alternatives beyond the given options. We introduce MoralAltDataset, comprising 307 Advisor and AI-facing Agent dilemmas augmented with compromise and reframed alternatives. We compare human and LLM judgments in binary and four-option settings. Across human participants and 15 LLMs, aggregate moral choice distributions differ substantially between the two settings, with compromise often preferred over either original binary option. Results show value shifts and stronger human-LLM agreement on alternatives. Source-stratified results reveal a descriptive gap: human alternative-selection rates are similar across authoring sources, whereas LLMs select GPT-5-authored alternatives substantially more often. We then compare human-au
    
[^210]: EchoSonar-R：一种用于超声心动图疾病分类与报告生成的多视图推理模型

    EchoSonar-R: A Multi-View Reasoning-Enabled Model for Disease Classification and Report Generation in Echocardiography

    [https://arxiv.org/abs/2606.28164](https://arxiv.org/abs/2606.28164)

    EchoSonar-R是一种结合时空视频编码器与结构感知心脏检测器的多视图推理视觉-语言模型，可同时完成超声心动图的多标签疾病分类和报告生成，并通过空间定位的解剖学证据增强可解释性与临床信任度。

    

    超声心动图是目前应用最广泛的无创心脏成像方式，为心血管诊断提供重要信息。解读超声心动图需要综合多个心脏切面的互补证据，以识别异常并生成结构化的临床报告。尽管近期的研究多聚焦于提升分类性能，但大多数模型缺乏明确的诊断推理能力和空间定位的解剖学证据，从而限制了临床医生的信任。我们提出了EchoSonar-R，这是一个具备多视图推理能力的视觉-语言模型，能够对超声心动图检查同时执行多标签疾病分类和报告生成。EchoSonar-R将时空视频编码器与结构感知的心脏检测器相结合，后者提供空间定位的解剖学线索，以在跨视图推理过程中提升可解释性和临床医生的信任度。EchoSonar-R采用两阶段训练：

    arXiv:2606.28164v2 Announce Type: replace-cross  Abstract: Echocardiography is the most widely used non-invasive cardiac imaging modality, providing essential information for cardiovascular diagnosis. Interpreting an echocardiogram requires synthesizing complementary evidence across multiple heart views to identify abnormalities and produce structured clinical reports. While recent efforts focus on improving classification performance, most models lack explicit diagnostic reasoning and spatially grounded anatomical evidence, limiting clinician trust. We present EchoSonar-R, a multi-view reasoning-enabled vision-language model that jointly performs multi-label disease classification and report generation from echocardiography studies. EchoSonar-R combines a spatiotemporal video encoder with a structure-aware cardiac detector that provides spatially grounded anatomical cues to improve interpretability and clinician trust during cross-view reasoning. EchoSonar-R is trained in two stages: 
    
[^211]: 引导而非解决：为大型代码智能体训练小型评论模型

    Steer, Don't Solve: Training Small Critic Models for Large Code Agents

    [https://arxiv.org/abs/2606.21811](https://arxiv.org/abs/2606.21811)

    通过训练专门负责高层次规划的小型评论模型（4B/8B）在推理时引导大型编码智能体识别并纠正错误，在SWE-Bench Verified上显著提升多个更大规模编码智能体的解决率（最高提升16.0%）并降低推理成本。

    

    编码任务通常较为复杂，需要多种能力，涵盖从高层次规划到低层次实现的各个方面。虽然编码智能体针对这些联合能力进行了优化，但诸如高层次规划等单项能力可能有不同的最优解，并仍然是主要瓶颈。为应对这一挑战，我们训练了一个独立于编码智能体、专门擅长高层次规划的评论模型，在推理阶段对编码智能体进行引导。我们构建了SFT和DPO数据来训练该评论模型，使其能够识别编码智能体所犯的错误，并提供正确且清晰的高层次指导，而无需生成具体的操作动作。实验表明，我们微调后的4B和8B评论模型显著提升了6个更大规模编码智能体的性能（例如，在SWE-Bench Verified上，将GLM-4.7-Flash-30B-A3B和GPT-OSS-120B的解决率分别提升了16.0%和14.4%）。该评论模型还降低了总推理成本（摘要原文在此处截断）。

    arXiv:2606.21811v2 Announce Type: replace-cross  Abstract: Coding tasks are typically complicated and require multiple capabilities, ranging from high-level planning to low-level implementation. While coding agents are optimized for the joint capabilities, individual capabilities such as high-level planning may have different optima and remain a major bottleneck. To address this challenge, we train a separate critic model that is specialized in high-level planning to steer the coding agent in inference. We construct SFT and DPO data to train the critic model to identify errors made by the coding agent and provide correct and clear high-level guidance without generating concrete actions. Experiments show that our fine-tuned 4B and 8B critic models significantly improve the performance of 6 larger coding agents (e.g., improving the resolved rates of GLM-4.7-Flash-30B-A3B and GPT-OSS-120B by 16.0% and 14.4% on SWE-Bench Verified). The critic model also reduces the total inference costs fo
    
[^212]: 弥合语义缓存中的运营差距

    Closing the Operational Gap in Semantic Caching

    [https://arxiv.org/abs/2606.19719](https://arxiv.org/abs/2606.19719)

    该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。

    

    语义缓存通过为语义相似的查询提供缓存响应来降低大语言模型（LLM）的推理成本。标准做法是使用PR-AUC来评估这些系统，但该指标仅衡量分数的排序质量，而忽略了分数在固定阈值下是否可用。我们证明这种错位会导致系统性的糟糕部署选择，因为PR-AUC最高的模型在实际运行中往往表现最差。我们引入了精确率-缓存命中率（P-CHR）AUC这一缓存感知指标，用于衡量不同缓存利用率水平下的精确率；以及运营保留率（ORR），用于捕捉离线排序质量在部署时的保留程度。我们将离线质量与部署质量之间的运营差距分解为可恢复的阈值效用部分，以及由数据集正例率固定的不可约简的结构部分。我们的实验表明，阈值效用差距由训练目标决定，而非……（摘要原文在此处截断）

    arXiv:2606.19719v3 Announce Type: replace-cross  Abstract: Semantic caching cuts LLM inference costs by serving a cached response to semantically similar queries. Standard practice evaluates these systems using PR-AUC, a metric that only measures how well scores rank and ignores whether they are usable at a fixed threshold. We show this mismatch leads to systematically poor deployment choices, as models with the highest PR-AUC are often the worst in operation. We introduce Precision--Cache Hit Ratio (P-CHR) AUC, a cache-aware metric that measures precision across cache utilization levels, and Operational Retention Rate (ORR), which captures how much offline ranking quality survives at deployment. We decompose the operational gap between offline and deployed quality into a recoverable threshold-utility component and an irreducible structural component fixed by the dataset's positive rate. Our experiments show that the threshold-utility gap is governed by the training objective rather th
    
[^213]: 面向动态学习的显式交互架构：结构归纳偏置的受控研究

    Explicit Interaction Architectures for Dynamical Learning: A Controlled Study of Structural Inductive Bias

    [https://arxiv.org/abs/2606.19101](https://arxiv.org/abs/2606.19101)

    该论文提出由有序局部状态调制变换构成的因果循环单元，并通过受控实验检验这种显式设计的交互架构相比通用回声状态网络能否为动态学习提供有用的结构归纳偏置。

    

    我们研究一种“结构优先”的动态学习方法，其中有状态交互的组织方式被显式地加以规定，而不是完全交由通用的循环参数化来处理。我们引入了因果循环单元，它由有序的局部状态调制变换序列构建而成。该构造受基于波的交互模型启发，但本文研究的单元并不施加散射、无源性或能量平衡等约束。鉴于固定的循环动力学、经设计的储备池拓扑、仅读取层学习以及循环深度等问题都已被充分研究，本文的实证问题被有意限定得更窄：在受控的计算条件下，所提出的交互组织方式能否提供一种有用的归纳偏置？我们比较了一层结构化模型、两层结构化模型以及通用的回声状态网络（ESN），三者均具有12个循环状态和相同的严格计算预算。

    arXiv:2606.19101v2 Announce Type: replace-cross  Abstract: We investigate a structure-first approach to dynamical learning in which the organization of stateful interactions is prescribed explicitly rather than left entirely to a generic recurrent parameterization. We introduce causal recurrent units built from an ordered sequence of local, state-modulated transformations. The construction is motivated by wave-based interaction models, but the units studied here do not impose scattering, passivity, or energy-balance constraints.   Because fixed recurrent dynamics, designed reservoir topologies, readout-only learning, and recurrent depth are already well established, the empirical question is deliberately narrower: does the proposed interaction organization provide a useful inductive bias under controlled computational conditions? We compare a one-layer structured model, a two-layer structured model, and a generic echo-state network (ESN), all with 12 recurrent states and the same stric
    
[^214]: 超越AHI：面向互联健康的可解释因果发现引导的睡眠恢复框架

    Beyond AHI: An Interpretable Causal-Discovery-Guided Framework for Sleep Recovery in Connected Health

    [https://arxiv.org/abs/2606.18506](https://arxiv.org/abs/2606.18506)

    该论文提出了一个可解释的因果发现引导框架，通过有向无环图学习从多模态PSG数据中推导层次化的睡眠恢复评分（SRS），超越了传统AHI指数的局限，能够更全面地捕捉呼吸负荷、缺氧负荷、睡眠碎片化、睡眠结构和自主神经调节等多个生理域，并可与互联健康技术中的可穿戴设备数据流自然对接。

    

    客观的睡眠评估依赖于多导睡眠图（PSG），然而临床影响往往更好地体现在嗜睡和疲劳等患者报告结局（PROs）中。现有的综合指数，包括呼吸暂停低通气指数（AHI），对功能恢复背后的多域生理机制提供的洞察有限。我们提出了一个可解释的、因果发现引导的框架，用于从多模态PSG数据中推导层次化的睡眠恢复评分。利用两个大型人群队列（MESA：n=1,540；MrOS：n=825），我们应用有向无环图（DAG）学习来识别涵盖呼吸负荷、缺氧负荷、睡眠碎片化、睡眠结构和自主神经调节的候选生理驱动因素。尽管这些域源自临床PSG，但它们可以自然地映射到互联健康技术中日益可用的感知数据流，包括可穿戴心电图、血氧监测和睡眠分期估计。

    arXiv:2606.18506v2 Announce Type: replace  Abstract: Objective sleep assessment relies on polysomnography (PSG), yet clinical impact is often better reflected in patient-reported outcomes (PROs) such as sleepiness and fatigue. Existing summary indices, including the Apnea-Hypopnea Index (AHI), provide limited insight into the multidomain physiology underlying functional recovery. We propose an interpretable, causal-discovery-guided framework for deriving a hierarchical Sleep Recovery Score (SRS) from multimodal PSG. Using two large population cohorts (MESA: \(n=1{,}540\); MrOS: \(n=825\)), we apply directed acyclic graph (DAG) learning to identify candidate physiological drivers spanning respiratory burden, hypoxic burden, sleep fragmentation, sleep architecture, and autonomic regulation. Although derived from clinical PSG, these domains map naturally to sensing streams increasingly available in connected health technologies, including wearable ECG, oximetry, and sleep-stage estimation
    
[^215]: ReproRepo：利用GitHub仓库议题规模化扩展可复现性审计

    ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues

    [https://arxiv.org/abs/2606.18237](https://arxiv.org/abs/2606.18237)

    ReproRepo提出利用GitHub上人工提交的议题作为天然监督信号，构建了可规模化的可复现性评估框架，并在1,149篇机器学习论文上验证了LLM智能体无需执行代码即可识别真实复现障碍的能力（最佳智能体可覆盖约90%的论文）。

    

    从论文及已发布代码中复现研究结果对科学进步至关重要。现有工作已引入基准来评估LLM智能体能否协助实现可复现性，但由于在数据整理和评估方面依赖大量人工投入，这些基准难以规模化。我们提出了ReproRepo，这是一个可扩展的可复现性评估框架，它利用人工提交的GitHub议题（issues）作为对真实复现障碍天然产生的监督信号。我们在来自主要会议的1,149篇近期机器学习论文上构建了ReproRepo实例，并评估了四种前沿模型-智能体配置。结果表明，LLM智能体即使不执行代码，也能从论文-仓库配对中识别出许多真实世界的可复现性问题：我们研究中表现最佳的智能体，即搭载GPT-5.5的Codex，能为约90%的论文找出至少一个与人工报告语义相关的复现障碍。

    arXiv:2606.18237v2 Announce Type: replace-cross  Abstract: Reproducing research results from papers and released code is central to scientific progress. Existing works have introduced benchmarks to evaluate whether LLM agents can assist with reproducibility, but they are difficult to scale due to their reliance on substantial manual effort for data curation and evaluation. We introduce ReproRepo, a scalable framework for reproducibility evaluation that leverages human-raised GitHub issues as naturally occurring supervision on realistic reproduction blockers. We instantiate ReproRepo on 1,149 recent machine learning papers from major conferences and evaluate four frontier model-agent configurations. Our results show that LLM agents, even without executing code, can identify many real-world reproducibility problems from paper-repository pairs: the best agent in our study, namely Codex with GPT-5.5, surfaces at least one semantically related human-reported blocker for $\sim$90% of papers 
    
[^216]: 学习精炼隐藏状态以实现可靠的大语言模型推理

    Learning to Refine Hidden States for Reliable LLM Reasoning

    [https://arxiv.org/abs/2606.17524](https://arxiv.org/abs/2606.17524)

    提出强化引导的潜在精炼框架ReLAR，通过学习到的深度与动作控制器在解码前自适应地迭代精炼隐藏状态，无需显式思维链即可提升大语言模型推理的准确性、生成质量与稳定性，并大幅降低推理开销。

    

    大语言模型展现出强大的推理能力，但在复杂的多步推理场景中，其内部推理过程可能仍不稳定，早期的隐藏状态错误可能会传播并导致错误的预测。我们提出了ReLAR，一种强化引导的潜在精炼框架，可在解码之前迭代地更新隐藏表示。ReLAR维护一个紧凑的潜在推理状态，并利用学习到的深度控制器和动作控制器来自适应地确定精炼步骤的数量与方向。这些控制器通过基于逐步似然提升的策略梯度目标进行训练，无需显式生成思维链即可实现高效的输入自适应推理。在医学、数学、多跳推理和开放式生成等多个基准上的实验表明，ReLAR在提高准确性、生成质量和推理稳定性的同时，其推理开销显著低于显式的推理方法。

    arXiv:2606.17524v3 Announce Type: replace  Abstract: Large language models show strong reasoning ability, but their internal reasoning process can remain unstable in complex multi-step settings, where early hidden-state errors may propagate to incorrect predictions. We propose ReLAR, a reinforcement-guided latent refinement framework that iteratively updates hidden representations before decoding. ReLAR maintains a compact latent reasoning state and uses learned depth and action controllers to adaptively determine both the number and direction of refinement steps. The controllers are trained with a policy gradient objective based on step-wise likelihood improvement, enabling efficient input-dependent reasoning without explicit chain-of-thought generation. Experiments on medical, mathematical, multi-hop reasoning, and open-ended generation benchmarks show that ReLAR improves accuracy, generation quality, and reasoning stability with substantially lower inference overhead than explicit r
    
[^217]: HiMPO：面向长时程智能体低纠缠信用分配的后见之明引导记忆策略优化

    HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents

    [https://arxiv.org/abs/2606.16285](https://arxiv.org/abs/2606.16285)

    HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。

    

    长时程智能体依赖记忆机制来压缩交互历史，但优化记忆写入面临一个独特的信用分配挑战：记忆更新可能因下游工具故障、噪声观测或推理错误而非其自身贡献而受到奖励或惩罚。我们提出了HiMPO，一个用于为长时程智能体中记忆写入动作分配低纠缠信用的后见之明引导记忆策略优化框架。HiMPO首先通过在相同的预写入状态下比较可从先前记忆和更新后记忆中恢复的任务相关信息，来估计记忆更新的局部效用。然后，它将后见相关性用作一个有界的回溯过滤器，当局部效用得不到目标结果支持时衰减记忆信用。由此产生的记忆特定优势仅应用于记忆token，而轨迹级奖励则优化智能体的其余部分。

    arXiv:2606.16285v2 Announce Type: replace  Abstract: Long-horizon agents rely on memory mechanisms to compress interaction history, but optimizing memory writing faces a distinct credit assignment challenge: a memory update may be rewarded or penalized due to downstream tool failures, noisy observations, or reasoning errors rather than its own contribution. We propose HiMPO, a Hindsight-Informed Memory Policy Optimization framework for assigning less-entangled credit to memory-writing actions in long-horizon agents. HiMPO first estimates the local utility of a memory update by comparing the task-relevant information recoverable from the previous and updated memories under the same pre-write state. It then uses hindsight relevance as a bounded retrospective filter that attenuates memory credit when local utility is not supported by the target outcome. The resulting memory-specific advantage is applied only to memory tokens, while trajectory-level rewards optimize the rest of the agent's
    
[^218]: DOG-DPO：面向安全对齐的几何动态优化

    DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment

    [https://arxiv.org/abs/2606.07678](https://arxiv.org/abs/2606.07678)

    提出无需训练的数据选择框架DOG-DPO，将偏好对视为模型表示空间中的几何方向，通过分解全局锚定子空间与数据集特有残余子空间并最大化多样性覆盖，为DPO安全对齐筛选出广泛且非冗余的偏好数据子集。

    

    大语言模型的安全对齐依赖于偏好数据，但当前的训练流程往往使用庞大且冗余的数据集。现有的数据选择方法通常独立地对每个偏好对进行打分，将方向性的偏好信息压缩为标量化的质量或多样性分数。这种以样本为中心的视角在多数据集场景下尤为受限，因为共享的安全方向与数据集特有的残余风险同时存在。我们提出了DOG-DPO，一个无需训练的数据选择框架，它将偏好对视为结构化的几何信号。DOG-DPO首先将每个偏好对表示为模型表示空间中的一个方向；然后将多数据集的偏好几何分解为全局锚定子空间和数据集特有的残余子空间；最后通过最大化基于多样性的覆盖度来选择子集，从而在DPO训练之前实现对安全对齐方向的广泛、非冗余的覆盖。

    arXiv:2606.07678v3 Announce Type: replace-cross  Abstract: Safety alignment for large language models relies on preference data, but current pipelines often train on large, redundant datasets. Existing data selection methods typically score each preference pair independently, collapsing directional preference information into scalar quality or diversity scores. This sample-centric view is especially limiting in multi-dataset settings, where shared safety directions coexist with dataset-specific residual risks. We propose DOG-DPO, a training-free data selection framework that treats preference pairs as structured geometric signals. DOG-DPO first represents each preference pair as a direction in model representation space. It then decomposes multi-dataset preference geometry into a global anchor subspace and dataset-specific residual subspaces. Finally, it selects subsets by maximizing diversity-based coverage, encouraging broad, non-redundant coverage of alignment directions before DPO 
    
[^219]: 为扩散语言模型启用共享前缀的KV缓存

    Enabling KV Caching of Shared Prefix for Diffusion Language Models

    [https://arxiv.org/abs/2606.07571](https://arxiv.org/abs/2606.07571)

    本文提出bicache，首个针对扩散语言模型共享前缀的KV缓存技术，解决了双向注意力下KV动态变化导致传统缓存失效的问题。

    

    针对共享前缀的键值（KV）缓存对于高吞吐量的大语言模型（LLM）服务至关重要，但在新兴的扩散语言模型（DLM）中面临重大挑战。在DLM中，双向注意力意味着更新任何一个token都会动态改变整个上下文及其对应的KV。因此，现有为LLM开发的缓存技术（其假设KV一旦计算完成就保持不变）会破坏共享前缀的KV。我们的实验表明，将这些技术应用于DLM会导致模型准确率崩溃至接近零。为了实现高吞吐量的DLM服务，我们提出了双向前缀缓存bicache，这是首个针对DLM共享前缀的KV缓存技术。

    arXiv:2606.07571v3 Announce Type: replace-cross  Abstract: Key-value (KV) caching for shared prefixes is essential for high-throughput large language model (LLM) serving, but it faces critical challenges in emerging diffusion language models (DLMs). In DLMs, bidirectional attention means that updating any token dynamically alters the entire context and its corresponding KVs. Thus, existing caching techniques developed for LLMs, which assume that KVs remain invariant once computed, corrupt the shared prefix KVs. Our experiments show that applying these techniques to DLMs causes model accuracy to collapse to near zero.   To unlock high-throughput DLM serving, we propose bidirectional prefix caching, bicache, the first KV caching technique for shared prefixes in DLMs. bicache is designed based on key observations from our comprehensive analysis: shared prefix KVs remain stable and reusable in shallow layers, while the depth of shallow layers depends on the fraction of shared prefix tokens
    
[^220]: RECAP：面向提示词持续适应的回归评估

    RECAP: Regression Evaluation for Continual Adaptation of Prompts

    [https://arxiv.org/abs/2606.06698](https://arxiv.org/abs/2606.06698)

    RECAP是一个在严格“先适应后测试”主动协议下、于约束层面评估提示词优化方法持续学习能力（遗忘、回归、前向迁移）的基准，实验发现现有六种方法在面对动态演变的约束时均无显著改进。

    

    生产级智能体系统经常面临不断变化的约束条件，并且必须从下一次交互开始就遵守这些约束。诸如工具调用通知改变合规阈值、或政策更新增加披露要求等场景都符合这一标准，在生产环境中几乎没有出错的空间。这种主动适应设定在部署中很常见，但在当前的基准测试中却缺失，因为现有基准要么假设静态的约束集合，要么采用带有评估反馈的被动式协议。我们提出了RECAP，这是一个在严格的“先适应后测试”主动协议下、于约束层面衡量持续学习现象（遗忘、回归、前向迁移）的基准：提示词优化方法仅接收约束规范，必须在看到任何测试数据之前完成泛化。通过在五个大语言模型和三种约束演进计划下评估六种方法，我们发现这些方法没有表现出显著的改进（性能提升）。

    arXiv:2606.06698v4 Announce Type: replace-cross  Abstract: Production agentic systems routinely face evolving constraints and must comply from the very next interaction. Scenarios like a tool-call notification changing a compliance threshold or a policy update adding disclosure requirements fit this criteria, having close to no room for errors in production. This proactive adaptation setting is common in deployment, but absent from current benchmarks, which assume either static constraint sets or reactive protocols with evaluation feedback. We introduce RECAP, a benchmark that measures continual-learning phenomena (forgetting, regression, forward transfer) at the constraint level under a strictly proactive adapt-then-test protocol: prompt optimization methods receive only the constraint specification and must generalize before seeing any test data. Evaluating six methods across five LLMs and three schedules with evolving constraints, we find that these methods show no significant impro
    
[^221]: 学生学到了什么？对暗知识的特征级分析

    What Do Students Learn? A Feature-Level Analysis of Dark Knowledge

    [https://arxiv.org/abs/2606.03052](https://arxiv.org/abs/2606.03052)

    本文通过交互张量框架揭示了知识蒸馏通过修剪低频的样本特定特征而起正则化作用，并据此提出无需教师模型的混淆蒸馏方法（CD），利用模型自身演化的混淆模式作为动态软目标，在CIFAR-100上超越现有自蒸馏方法1.2%。

    

    知识蒸馏（KD）是模型压缩的有力工具，但学生模型获取特征表示的精确机制仍未得到充分探索。在本工作中，我们使用交互张量框架分析学生的特征学习过程。我们的分析表明，有效的知识蒸馏起到了正则化器的作用，它会修剪掉低频的、特定于样本的特征，促使学生模型依赖于一组紧凑且高度可复用的特征。更为关键的是，我们观察到数据集级别的混淆矩阵包含类似于教师模型“暗知识”的结构信息。利用这一洞察，我们提出了混淆蒸馏（CD），这是一种无需教师模型的自蒸馏方法，它利用模型自身不断演化的混淆模式作为动态软目标。CD在CIFAR-100数据集上基于ResNet-34和ResNet-50取得了有竞争力的性能，比现有的自蒸馏方法（如CS-KD和PS-KD）高出1.2%。

    arXiv:2606.03052v2 Announce Type: replace  Abstract: Knowledge Distillation (KD) is a powerful tool for model compression, yet the precise mechanisms by which student models acquire feature representations remain underexplored. In this work, we analyze student feature learning using the Interaction Tensor framework. Our analysis reveals that effective KD acts as a regularizer that prunes low-frequency, sample-specific features, encouraging the student to rely on a compact set of highly reusable features. Crucially, we observe that the dataset-level confusion matrix contains structural information analogous to the teacher's "Dark Knowledge." Leveraging this insight, we propose Confusion Distillation (CD), a teacher-free self-distillation method that utilizes the model's own evolving )confusion patterns as dynamic soft targets. CD achieves competitive performance on ResNet-34 and ResNet-50 for CIFAR-100, outperforming existing self-distillation methods like CS-KD and PS-KD by 1.2% while 
    
[^222]: 技能复用作为智能体强化学习中的压缩

    Skill Reuse as Compression in Agentic RL

    [https://arxiv.org/abs/2605.31509](https://arxiv.org/abs/2605.31509)

    该论文提出基于最小描述长度（MDL）原则的ReuseRL框架，通过从成功轨迹中提取共享技能字典并惩罚难以压缩的特异行为，显著提升了智能体强化学习的分布内外泛化能力。

    

    通过强化学习（RL）训练的大语言模型智能体通常学到脆弱的、局限于特定任务的捷径。我们假设，当智能体的成功轨迹在结构上可压缩、能够被分解为一小组可复用的抽象模式时，智能体的泛化能力会更好。为了将这一假设形式化，我们提出了ReuseRL，该方法将智能体强化学习建立在最小描述长度（MDL）原则之上。ReuseRL从成功轨迹中提取共享的技能字典，并在强化学习目标中加入分割代价，明确惩罚那些编码效果差的特异化行为。我们证明了PAC-Bayes界，保证从成功轨迹中提取的字典在未来成功行为上具有有界的期望描述长度。在ALFWorld、TextWorld-Cooking和Countdown-Stepwise任务上，ReuseRL在分布内和分布外成功率上均超越了原始GRPO以及强基线方法。

    arXiv:2605.31509v2 Announce Type: replace-cross  Abstract: Large language model agents trained with reinforcement learning (RL) often learn brittle, task-specific shortcuts. We hypothesize that agents generalize better when their successful trajectories are structurally compressible, decomposed into a small set of reusable abstract patterns. To formalize this, we introduce ReuseRL, which grounds agentic RL in the Minimum Description Length (MDL) principle. ReuseRL extracts a shared skill dictionary from successful trajectories and augments the RL objective with a segmentation cost, explicitly penalizing idiosyncratic behaviors that encode poorly. We prove a PAC-Bayes bound guaranteeing that a dictionary extracted from successful trajectories has bounded expected description length on future successful behavior. Across ALFWorld, TextWorld-Cooking, and Countdown-Stepwise, ReuseRL improves in- and out-of-distribution success over vanilla GRPO and strong round-length baselines.
    
[^223]: PEARL：基于教学对齐强化学习训练苏格拉底式导师

    PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning

    [https://arxiv.org/abs/2605.29582](https://arxiv.org/abs/2605.29582)

    PEARL提出了一种教学对齐的强化学习框架，通过可控学生模拟器解耦认知状态并在多轮师生交互中协调多个教学目标，从而训练出擅长渐进式引导的苏格拉底式辅导智能体。

    

    arXiv:2605.29582v2 公告类型： replace-cross。摘要：大型语言模型（LLMs）在教育辅导领域展现出巨大潜力。现有方法通常训练它们去解题并给出正确答案，但这种以解题为中心的范式忽视了有效辅导的关键要求：渐进式引导以及在多轮交互中对多个教学目标的协调。开发这样的导师仍然充满挑战，因为学生的行为会随个体知识状态发生显著变化，教学效果取决于最终答案正确性之外的多个因素，且在师生交互过程中协调这些目标本身就十分困难。为应对这些挑战，我们提出了PEARL，一个用于训练苏格拉底式辅导智能体的教学对齐强化学习框架。首先，我们引入了一个可控的学生模拟器，将潜在认知状态与回复生成解耦，使得模拟……（摘要原文此处被截断）

    arXiv:2605.29582v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) show strong potential as educational tutors. Existing approaches typically train them to solve problems and provide correct answers, but this problem-solving-centered paradigm overlooks key requirements of effective tutoring: progressive guidance and the coordination of multiple pedagogical objectives across multi-turn interactions. Developing such tutors remains challenging because student behavior varies substantially with individual knowledge states, pedagogical effectiveness depends on multiple factors beyond final-answer correctness, and coordinating these objectives over tutor-student interactions is inherently difficult. To address these challenges, we propose PEARL, a PEdagogically Aligned Reinforcement Learning framework for training Socratic tutoring agents. First, we introduce a controllable student simulator that disentangles latent cognitive states from response generation, enabling sim
    
[^224]: 用于宇宙学21厘米光锥模拟的三维条件扩散模型

    Three-dimensional Conditional Diffusion Models for Cosmological 21 cm Lightcone Emulation

    [https://arxiv.org/abs/2605.29016](https://arxiv.org/abs/2605.29016)

    本文系统研究了用于三维宇宙学21厘米光锥模拟的条件扩散模型，并通过受控比较评估了预处理、动态范围压缩、架构深度和训练时长等关键因素对生成质量的影响。

    

    我们研究了用于三维21厘米光锥模拟的条件扩散建模，重点关注天平面尺寸为64×64、视线方向深度达1024个单元的立方体数据。相对于早期的二维研究，三维设置要困难得多，因为内存限制迫使采用非常小的微型批次，而底层的体素分布高度偏斜且呈长尾特征。我们使用25,600个训练光锥以及在固定参数点处的验证集合，对预处理选择、动态范围压缩设置、架构深度和训练时长进行了受控比较。在验证方面，每个参考参数点包含800个具有独立初始条件的21cmFAST实现，我们对每个模型和每个参考集分别使用800个样本进行集合比较。我们通过图像空间和汇总统计空间中的互补诊断方法来评估生成的光锥。

    arXiv:2605.29016v2 Announce Type: replace-cross  Abstract: We investigate conditional diffusion modeling for three-dimensional 21 cm lightcone emulation, focusing on cubes with a sky-plane size of $64\times64$ and a line-of-sight depth up to 1024 cells. Relative to earlier 2D studies, the 3D setting is substantially harder because memory limits enforce very small micro-batches while the underlying voxel distribution is highly skewed and long tailed. We perform controlled comparisons across preprocessing choices, dynamic-range compression settings, architecture depth, and training duration using $25{,}600$ training lightcones and validation ensembles at fixed parameter points. For validation, each reference parameter point contains 800 21cmFAST realizations with independent initial conditions, and we use 800 samples per model and per reference set for the reported ensemble comparisons. We evaluate generated lightcones with complementary diagnostics in both image and summary-statistic sp
    
[^225]: 当最强的教师并非最好的教师：以学生为中心的答案选择

    When the Strongest Teacher Is Not the Best Teacher: Student-Centric Answer Selection

    [https://arxiv.org/abs/2605.26872](https://arxiv.org/abs/2605.26872)

    论文提出SCAS框架，证明最强教师的正确答案未必是学生的最佳训练监督，并通过逐token梯度分解推导出仅需前向计算的高效代理指标，依据学生中心学习成本来选择最适合学生的教师答案。

    

    大语言模型（LLM）训练越来越依赖教师生成的监督信号，包括合成回复、推理轨迹和工具使用演示。当前的做法通常选择表现最强的教师来生成学生的训练数据，这隐含地将教师的测试性能视为教学质量的替代指标。我们证明这一假设可能失效：即使多个教师对同一问题都给出了正确答案，最强教师给出的答案也不一定是对特定学生最好的监督信号。为了解决这一空白，我们提出了学生中心答案采样，这是一个根据估计的学生中心学习成本，从经过验证的教师生成答案中进行选择的框架。受逐token梯度分解的启发，我们推导出一种仅需前向计算的高效代理指标来估计该成本，并用它来指导训练过程中的答案选择。实验涵盖了30个教师模型和6个学生基础模型。

    arXiv:2605.26872v4 Announce Type: replace-cross  Abstract: LLM training increasingly relies on teacher-generated supervision, from synthetic responses to reasoning traces and tool-use demonstrations. Current practice often chooses the highest-performing teacher to generate student training data, implicitly treating teacher test performance as a proxy for teaching quality. We show that this assumption can fail: even when multiple teachers provide correct answers to the same question, the answer from the strongest teacher is not necessarily the best supervision for a given student. To address this gap, we propose Student-Centric Answer Sampling (SCAS), a framework that selects from verified teacher-generated answers according to their estimated student-centric learning cost. Motivated by a token-wise gradient decomposition, we derive an efficient forward-only proxy for this cost and use it to guide answer selection during training. Experiments across 30 teacher models, 6 student base mod
    
[^226]: 潜在循环Transformer：架构探索、训练策略与缩放行为

    Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior

    [https://arxiv.org/abs/2605.26797](https://arxiv.org/abs/2605.26797)

    本文提出潜在循环Transformer（LRT），通过复用前一个token的高层隐藏状态作为循环记忆，在不改变标准注意力机制和KV-cache接口的前提下引入跨token、跨层的信息通路，并设计交错并行训练方法以约2倍理想计算成本实现循环记忆的预训练。

    

    我们研究了潜在循环Transformer（LRT），它是对自回归Transformer的一种轻量级增强，将前一个token的高层源层隐藏状态复用为下一个token的循环记忆。由于该状态在普通解码过程中已经计算得到，LRT在保留标准注意力机制、KV-cache接口以及每个生成token仅一次模型前向传播的前提下，引入了一条跨token、跨层的潜在通路。为了在不按顺序展开完整序列的情况下预训练这种循环机制，我们提出了交错并行训练：先用一次全序列初始化前向传播构建共享缓冲区，随后对各不相交的位置子集进行顺序细化，并在每个子集内部进行并行计算。这使得每个token都能获得感知循环记忆的监督信号，计算成本约为理想token计算量的2倍。在1.3B和2.1B参数的nanochat风格骨干模型以及广泛的训练条件下……（原文摘要在此处被截断）

    arXiv:2605.26797v2 Announce Type: replace-cross  Abstract: We study Latent Recurrent Transformer (LRT), a lightweight augmentation of autoregressive transformers that reuses a high-level source-layer hidden state from the previous token as recurrent memory for the next token. Because this state is already computed during ordinary decoding, LRT introduces a cross-token, cross-layer latent pathway while preserving the standard attention mechanism, KV-cache interface, and one model forward per generated token. To pretrain this recurrence without sequentially unrolling the full sequence, we introduce interleaved parallel training: one full-sequence initialization forward constructs a shared buffer, followed by sequential refinement of disjoint position subsets with parallel computation within each subset. This provides every token with recurrent-memory-aware supervision at approximately 2x ideal token compute. Across 1.3B- and 2.1B-parameter nanochat-style backbones and a wide range of tra
    
[^227]: DiscoverPhysics：评估大语言模型开箱即用科学思维能力的基准测试

    DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking

    [https://arxiv.org/abs/2605.26087](https://arxiv.org/abs/2605.26087)

    提出了交互式基准测试DiscoverPhysics，通过让大语言模型在物理规律刻意偏离现实的22个模拟世界中设计实验、观察轨迹数据并归纳未知的运动定律，从而将模型真正的科学推理能力与对既有物理知识的记忆区分开来。

    

    前沿大语言模型如今在各类物理评测中表现优异，但很难将其真正的推理能力与对既有科学知识的记忆区分开来。我们提出了DiscoverPhysics，这是一个交互式基准测试，要求大语言模型智能体去发现一个模拟世界的运动定律，而该世界的物理规律被刻意设置为偏离我们的现实世界。我们构建了22个这样的世界，其物理规律包括屏蔽引力、分数幂引力、多物种耦合、隐藏的类暗物质粒子、非坐标无关的物理以及随时间变化的相互作用等。每个世界由N体模拟器按需生成，智能体需要提出多轮实验方案、观察原始轨迹数据，并最终提交对该世界物理规律的自然语言解释以及所推断定律的Python代码实现。由于解决一个世界的问题需要智能体设计具有信息量的实验并不断修正其假设，该基准……

    arXiv:2605.26087v2 Announce Type: replace-cross  Abstract: Frontier LLMs now perform strongly across a wide range of physics evaluations, but it is hard to disentangle genuine reasoning from recall of established science. We introduce DiscoverPhysics, an interactive benchmark that asks a LLM agent to discover the laws of motion of a simulated world whose physics deliberately deviates from our own. We construct 22 worlds governed by, among others, screened and fractional-power gravity, multi-species couplings, hidden dark-matter-like particles, non-coordinate-free physics, and time-varying interactions. Each world is generated on demand by an N-body simulator, for which the agent proposes several rounds of experiments, observes raw trajectory data, and ultimately submits both a natural-language explanation of the world's physics and a Python implementation of the inferred law. Because solving a world requires the agent to design informative experiments and revise its hypotheses, the ben
    
[^228]: 基于大语言模型的物理约束本构模型设计：两个智能体胜过一个

    LLM-driven design of physics-constrained constitutive models: two agents are better than one

    [https://arxiv.org/abs/2605.23754](https://arxiv.org/abs/2605.23754)

    提出首个多智能体LLM驱动的本构模型生成框架，由“创建者”智能体提出候选模型、“检查者”智能体依据九项物理约束进行审查并迭代修正，确保生成的本构模型严格遵守物理定律。

    

    开发能够捕捉材料在载荷作用下变形行为的本构模型，传统上需要在连续介质力学、机器学习和科学编程方面积累多年的专业知识。大语言模型（LLM）近来被证明可以通过按需生成本构模型来降低这一门槛，但现有的单智能体流程缺乏系统性的检查机制，无法确保所生成的模型遵守基本物理定律。为弥合这一差距，我们提出了首个多智能体LLM驱动的本构模型生成方法：一个“创建者”智能体根据数据提出候选模型，而一个“检查者”智能体则依据九项物理约束对每个提案进行严格审查，并在检测到违反约束时将其退回修改。我们以本构人工神经网络为例演示了这一概念，并在脑组织和橡胶（各向同性材料）以及猪皮肤组织（横观各向同性材料）上进行了基准测试。

    arXiv:2605.23754v2 Announce Type: replace  Abstract: Developing constitutive models that capture how materials deform under load traditionally requires years of specialized expertise in continuum mechanics, machine learning, and scientific programming. Large language models (LLMs) have recently been shown to lower this barrier by generating constitutive models on demand, but existing single-agent pipelines lack systematic checks that the resulting models respect fundamental physical laws. To close this gap, we introduce the first multi-agent LLM-driven approach for constitutive model generation: a Creator agent proposes a model tailored to the data, while an Inspector agent critically audits each proposal against nine physical constraints and returns it for refinement whenever a violation is detected. We demonstrate this concept with constitutive artificial neural networks (CANNs) and benchmark it on brain tissue and rubber as isotropic materials, and on porcine skin tissue as a transv
    
[^229]: 为什么推理模型会失去覆盖率？数据与“道路岔口”的作用

    Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road

    [https://arxiv.org/abs/2605.17026](https://arxiv.org/abs/2605.17026)

    本研究通过模拟图分支决策点等受控案例实验，揭示了微调数据中存在多条有效推理路径的“道路岔口”式场景是导致推理模型在SFT后训练中出现覆盖率收缩（pass@k退化）的关键原因。

    

    arXiv:2605.17026v2 公告类型：替换。摘要：大型语言模型的最新进展催生了推理模型的出现，这些模型通过专门的微调流程在复杂任务上展现出强大的性能。虽然这些方法能够可靠地提升 pass@1 准确率，但先前的研究观察到它们表现出覆盖率收缩行为，即 pass@k 相对于基础模型出现退化。在本文中，我们研究了基于 SFT 的后训练中推理收缩的原因。我们假设这种行为是由微调数据的特性所驱动的，特别是与决策点或“道路岔口”场景相关的特性，即模型遇到难以辨别的模式且存在多条有效推理路径的情况。为了验证这一假设，我们设计了受控案例研究来模拟此类决策点场景，涵盖图分支中的不可辨别节点以及不同的推理模式。通过追踪这些场景下的后训练动态，我们发现收缩现象……（摘要内容在此处截断）

    arXiv:2605.17026v2 Announce Type: replace  Abstract: Recent progress in large language models has led to the emergence of reasoning models, which have shown strong performance on complex tasks through specialized fine-tuning procedures. While these methods reliably improve pass@1 accuracy, prior works have observed that they show a coverage shrinkage behavior, where pass@k degrades relative to the base model. In this paper, we investigate the cause of reasoning shrinkage under SFT-based post-training. We hypothesize that this behavior is driven by properties of the fine-tuning data, specifically related to decision points or "forks in the road" scenarios where model encounters indecipherable patterns with multiple valid reasoning paths. To test this hypothesis, we design controlled case studies that simulate such decision-point settings, spanning indecipherable nodes in graph branching, and reasoning modes. By tracking post-training dynamics in these settings, we find that the shrinkag
    
[^230]: 非线性算子及其导数的万能逼近

    Universal Approximation of Nonlinear Operators and Their Derivatives

    [https://arxiv.org/abs/2605.15285](https://arxiv.org/abs/2605.15285)

    本文首次证明了关于 $k$ 阶可微非线性算子及其导数的万能逼近定理，首次将经典万能逼近定理完整推广至无穷维巴拿赫空间与算子学习领域，并由此开启了导数感知算子学习（DIOL）的新方向。

    

    建立非线性算子及其导数的万能逼近定理（UATs）是算子学习（OL）中的一个基础性开放问题，并在非线性泛函分析中引发了微妙的问题。我们通过算子学习架构，首次证明了关于 $k$ 阶可微非线性算子及其导数的万能逼近定理，这些定理在紧集上一致成立，并且在一般有限输入测度下于加权 Bastiani–Sobolev 空间中成立。在完全一般的巴拿赫空间框架下，这些结果首次将 [Hornik, 1991] 中相应的具有影响力的经典万能逼近定理完整地推广到无穷维空间和算子学习，并开启了在一般巴拿赫空间上的导数感知算子学习（DIOL）（即学习非线性算子及其导数）。基于我们的万能逼近定理，我们提出了 DIOL 中的 Bastiani–Sobolev 训练方法。我们展示了 DIOL 和我们的 UATs 可应用的前沿开放方向：算子学习中的高阶精度；快速约束……（摘要原文在此处截断）

    arXiv:2605.15285v3 Announce Type: replace-cross  Abstract: Establishing Universal Approximation Theorems (UATs) for nonlinear operators and their derivatives is a foundational open problem in Operator Learning (OL) and raises delicate questions in Nonlinear Functional Analysis. We prove the first UATs for $k$-times differentiable nonlinear operators and their derivatives via OL architectures, uniformly on compact sets and in weighted Bastiani--Sobolev spaces for general finite input measures. In full Banach-space generality, these are the first complete generalizations of the corresponding influential classical UATs in [Hornik, 1991] to infinite-dimensional spaces and OL, {and launch Derivative-Informed Operator Learning (DIOL) (i.e. learning nonlinear operators and their derivatives)} on general Banach spaces. Based on our UATs, we formulate Bastiani--Sobolev training in DIOL. We present open frontiers where DIOL and our UATs find applications: high-order accuracy in OL; fast constrai
    
[^231]: 可极化原子多极矩用于学习长程静电相互作用

    Polarizable atomic multipoles for learning long-range electrostatics

    [https://arxiv.org/abs/2605.05746](https://arxiv.org/abs/2605.05746)

    该论文提出了一种基于可极化原子多极矩的半局域框架来学习长程静电相互作用，显著提升了机器学习原子间势在离子、极性和界面体系的精度，并且无需直接监督即可涌现出物理上有意义的电响应。

    

    长程静电相互作用和极化效应仍然是机器学习原子间势扩展至离子、极性和界面体系的主要障碍。在此，我们引入了一种半局域框架，利用可极化原子多极矩从能量和力中学习静电相互作用。局域等变描述符预测依赖于环境的潜在单极矩、偶极矩和四极矩，而剩余的非局域电荷转移和极化效应则通过感应电荷和感应偶极的非自洽线性响应来捕捉。在四个多样化基准测试和四种短程MLIP架构上的实验表明，多极矩层级和响应项系统地提高了势能面精度，其中在长程效应至关重要的体系中提升最为显著。更重要的是，物理上有意义的电响应无需直接监督即可自然涌现。学习到的潜在多极矩能够产生准确的玻恩（有效电荷）……

    arXiv:2605.05746v2 Announce Type: replace-cross  Abstract: Long-range electrostatics and polarization remain central obstacles to extending machine learning interatomic potentials (MLIPs) to ionic, polar, and interfacial systems. Here we introduce a semi-local framework for learning electrostatics from energies and forces using polarizable atomic multipoles. Local equivariant descriptors predict environment-dependent latent monopoles, dipoles, and quadrupoles, while residual non-local charge transfer and polarization are captured by non-self-consistent linear response in induced charges and dipoles. Across four diverse benchmarks and four short-range MLIP architectures, the multipole hierarchy and response terms systematically improve potential energy surface accuracy, with the largest gains in systems where long-range effects are essential. More importantly, physically meaningful electrical responses emerge without direct supervision. The learned latent multipoles yield accurate Born 
    
[^232]: 泄漏审计基准揭示跨受试者听觉诱发脑电元音感知解码证据有限

    Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding

    [https://arxiv.org/abs/2605.00865](https://arxiv.org/abs/2605.00865)

    该研究通过严格的泄漏审计基准，发现跨受试者听觉脑电元音解码的证据非常有限，即使最佳模型也仅略高于随机水平且不显著。

    

    我们测试了在单一基准中控制试验身份、模型身份、预测来源和参与者水平推断时，听觉诱发脑电是否支持受试者无关的五元音感知解码。我们从OpenNeuro ds006104版本1.0.1重建了研究2的事件表，并分析了辅音-元音对任务。一对一标记-刺激配对产生了3,840个独立试验；对照条件选择和伪迹拒绝保留了来自16名参与者和61个脑电通道的1,094个时段。使用留一受试者测试评估了13种独特实现，参与者指标从33个完整预测副本中的36,102个试验预测中重建。随机森林在数值上最高，平衡准确率为21.474%（95%参与者自助区间，19.526-23.482%；随机水平为20%），但其参与者水平测试或任何实现均未通过校正。

    arXiv:2605.00865v3 Announce Type: replace-cross  Abstract: We tested whether auditory-evoked EEG supports subject-independent five-vowel perception decoding when trial identity, model identity, prediction provenance, and participant-level inference are controlled within a single benchmark. We reconstructed Study 2 event tables from OpenNeuro ds006104 version 1.0.1 and analyzed the consonant-vowel pair task. One-to-one marker-stimulus pairing yielded 3,840 independent trials; control-condition selection and artifact rejection retained 1,094 epochs from 16 participants and 61 EEG channels. Thirteen unique implementations were evaluated using leave-one-subject-out testing, with participant metrics reconstructed from 36,102 trial predictions across 33 complete prediction replicas. Random Forest was numerically highest at 21.474% balanced accuracy (95% participant-bootstrap interval, 19.526-23.482%; chance, 20%), but neither its participant-level tests nor any implementation survived correc
    
[^233]: D3-Gym：为数据驱动发现构建真实世界可验证环境

    D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery

    [https://arxiv.org/abs/2604.27977](https://arxiv.org/abs/2604.27977)

    本文提出了首个为科学数据驱动发现自动构建的可验证环境数据集D3-Gym，包含565个来自真实科学代码库的任务，其自动评估脚本与人工标注达到87.5%的一致性，且基于其轨迹训练可显著提升Qwen3模型在ScienceAgentBench上的表现。

    

    尽管面向科学数据驱动发现的语言模型和智能体近期取得了进展，但由于缺乏能够代表真实世界科学任务的可验证环境，其能力的提升受到了阻碍。为填补这一空白，我们推出了D3-Gym，这是首个为科学数据驱动发现自动构建的、带有可验证环境的数据集。D3-Gym包含来自四个学科、239个真实科学代码库的565个任务，每个任务均配有自然语言指令、预装依赖的可执行环境、数据集预览、参考解决方案以及自动合成的评估脚本。我们的评估脚本与人工标注的黄金标准达到87.5%的一致性，并在领域特定评估逻辑上表现出高度对齐。在从D3-Gym采样的轨迹上进行训练，使Qwen3系列模型在ScienceAgentBench上获得一致提升，其中Qwen3-32B绝对提升7.8个百分点（原文在此处截断）。

    arXiv:2604.27977v3 Announce Type: replace-cross  Abstract: Despite recent progress in language models and agents for scientific data-driven discovery, advancing their capabilities is held back by the absence of verifiable environments representing real-world scientific tasks. To fill this gap, we introduce D3-Gym, the first automatically constructed dataset with verifiable environments for scientific Data-Driven Discovery. D3-Gym comprises 565 tasks from 239 real scientific repositories across four disciplines, each with a natural language instruction, an executable environment with pre-installed dependencies, dataset previews, a reference solution, and an automatically synthesized evaluation script. Our evaluation scripts achieve 87.5% agreement with human-annotated gold standards and strong alignment in domain-specific evaluation logic. Training on trajectories sampled from D3-Gym yields consistent gains across Qwen3 models on ScienceAgentBench, boosting Qwen3-32B by 7.8 absolute poi
    
[^234]: 通过覆盖映射与拓扑权重先验进行重参数化

    Reparameterization through Coverings and Topological Weight Priors

    [https://arxiv.org/abs/2604.23804](https://arxiv.org/abs/2604.23804)

    本文通过覆盖映射推广了变分自编码器的重参数化技巧，使其适用于具有非平凡拓扑的潜在空间，并建立KL散度不等式，使ELBO中的KL项在某些情况下可解析求解。

    

    我们推广了应用于变分自编码器（VAE）中的重参数化技巧（RT），使这些模型能够拥有具有非平凡拓扑结构的潜在空间——即底流形被其他流形所覆盖，而在覆盖流形上已有某种可用于重参数化技巧的技术。这之所以成为可能，是因为覆盖映射是可测的——此外，这使得我们能够在底潜在流形上的前推（PF）密度之间建立关于KL散度的不等式，用覆盖流形上拉回密度之间的KL散度对其进行界定，从而在某些情况下，尽管支撑潜在流形具有拓扑非平凡性，VAE的ELBO中的KL项仍可被解析地求解。我们的发展路径与李群上的重参数化相近但略有不同，李群重参数化的最新提案是从李代数出发——即“通过”指数映射——对正态密度的前推进行重参数化，我们认为这是我们所提出的重参数化方法的一个特例。

    arXiv:2604.23804v2 Announce Type: replace  Abstract: We generalise the reparameterization trick (RT) applied in variational autoencoders (VAEs) letting these have latent spaces of non-trivial topology - i.e. that of base manifolds covered with other ones, on which some technique for RT is available. That is possible since covering maps are measurable - moreover, this allows to establish an inequality on KL-divergence between pushforward (PF) densities on the base latent manifold, bounding it with KL-divergence between pullbacks on the cover, in some cases making the KL-term of VAE's ELBO analytically tractable, despite the topological non-triviality of the supporting latent manifold. Our development follows a route close but somewhat alternative to reparameterization on Lie groups, the latest proposal for which is to reparameterize PFs of normal densities from the Lie algebra - "through" the exponential map, seen by us as a particular case of what we propose to call reparameterization 
    
[^235]: FedSPDnet：基于SPDnet的几何感知联邦深度学习

    FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet

    [https://arxiv.org/abs/2604.22494](https://arxiv.org/abs/2604.22494)

    提出了FedSPDnet框架，通过ProjAvg和RLAvg两种保持Stiefel流形几何结构的聚合策略，实现了基于SPD矩阵的联邦深度学习，在EEG运动想象基准上以更少的通信参数和更强的鲁棒性超越了联邦EEGnet。

    

    我们为经典的SPDnet模型提出了两个联邦学习框架，该模型处理对称正定（SPD）矩阵并带有Stiefel约束参数。与违反正交性的标准欧几里得平均不同，我们的方法通过两种高效的聚合策略保持几何结构：ProjAvg（将算术平均投影到Stiefel流形上）和RLAvg（通过回缩和提升近似切空间平均）。这两种方法计算高效、与优化器无关，并能为特征为SPD矩阵的信号处理应用实现可扩展的联邦学习。在EEG运动想象基准上的仿真表明，FedSPDnet在F1分数以及对联邦和部分参与场景的鲁棒性方面优于联邦EEGnet，同时每轮通信使用的参数更少。

    arXiv:2604.22494v2 Announce Type: replace-cross  Abstract: We introduce two federated learning frameworks for the classical SPDnet model operating on symmetric positive definite (SPD) matrices with Stiefel-constrained parameters. Unlike standard Euclidean averaging, which violates orthogonality, our approach preserves geometric structure through two efficient aggregation strategies: ProjAvg, projecting arithmetic means onto the Stiefel manifold, and RLAvg, approximating tangent-space averaging via retractions and liftings. Both methods are computationally efficient, independent of the optimizer, and enable scalable federated learning for signal processing applications whose features are SPD matrices. Simulations on EEG motor imagery benchmarks show that FedSPDnet outperforms federated EEGnet in F1 score and robustness to federation and partial participation, while using fewer parameters per communication round.
    
[^236]: Transformer的拓扑困境

    The Topological Trouble With Transformers

    [https://arxiv.org/abs/2604.17121](https://arxiv.org/abs/2604.17121)

    该论文揭示了Transformer纯前馈架构在动态状态追踪上的根本性拓扑缺陷——状态表示随每个新输入被不断推向更深层直至耗尽模型深度，并论证时间上延伸的认知需要从显式思维轨迹回归到基于循环结构的隐式激活动力学。

    

    Transformers通过不断扩展的上下文历史来编码序列中的结构。然而，其纯前馈架构从根本上限制了动态状态追踪能力。状态追踪——即对反映不断演变环境的潜在变量进行迭代更新——涉及固有的顺序依赖关系，而前馈网络难以维持这些依赖关系。因此，前馈模型在每处理一个新的输入步骤时，都会将不断演变的状态表示推送到更深的层堆栈中，使信息在浅层中无法访问，并最终耗尽模型的深度。虽然这种深度限制可以通过动态深度模型以及通过将状态表示外化的显式或潜在思考来绕过，但这些解决方案在计算和内存方面都是低效的。在本文中，我们认为时间上延伸的认知需要将焦点从显式思维轨迹重新转向通过循环结构实现的隐式激活动力学。

    arXiv:2604.17121v5 Announce Type: replace-cross  Abstract: Transformers encode structure in sequences via an expanding contextual history. However, their purely feedforward architecture fundamentally limits dynamic state tracking. State tracking -- the iterative updating of latent variables reflecting an evolving environment -- involves inherently sequential dependencies that feedforward networks struggle to maintain. Consequently, feedforward models push evolving state representations deeper into their layer stack with each new input step, rendering information inaccessible in shallow layers and ultimately exhausting the model's depth. While this depth limit can be bypassed by dynamic depth models and by explicit or latent thinking that externalizes state representations, these solutions are computationally and memory inefficient. In this article, we argue that temporally extended cognition requires refocusing from explicit thought traces to implicit activation dynamics via recurrent 
    
[^237]: 面向地球系统预测中百亿亿次生成式数据同化的线性复杂度全局注意力机制

    Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction

    [https://arxiv.org/abs/2604.16590](https://arxiv.org/abs/2604.16590)

    STORM提出了一种单阶段生成式AI数据同化框架，将数据同化重构为基于扩散模型的贝叶斯后验采样，并结合线性复杂度的全局注意力算法，在Frontier上扩展至74,400个GPU、达到6 ExaFLOPs持续吞吐量，并可在34秒内实现32,768成员的大规模集合不确定性量化。

    

    准确的地球系统预测需要从不完整的观测中进行状态推断，但传统的两阶段数据同化（DA）在计算上代价高昂，因为重复的基于偏微分方程（PDE）的集合预报、观测更新以及中间数据传输限制了高分辨率下的集合规模。我们提出了STORM，这是一个单阶段生成式AI框架，它将数据同化重新表述为基于扩散模型的贝叶斯后验采样，用可扩展的AI推断取代在线PDE集合预报。该框架进一步将时空Transformer与全局注意力算法相结合，通过可扩展的梯度传播将计算复杂度从二次方降低到线性，从而实现高分辨率、长上下文的地球系统建模。STORM在Frontier超级计算机上扩展至74,400个GPU，强扩展效率达96–99%，持续BF16吞吐量高达6 ExaFLOPs，同时能在34秒内完成32,768个成员的集合模拟以进行不确定性量化。

    arXiv:2604.16590v2 Announce Type: replace-cross  Abstract: Accurate Earth system prediction requires state inference from incomplete observations, but conventional two-stage data assimilation (DA) is computationally prohibitive because repeated PDE-based ensemble forecasts, observation updates, and intermediate data movement limit ensemble size at high resolution. We introduce STORM, a one-stage generative AI framework that reformulates DA as diffusion-based Bayesian posterior sampling, replacing online PDE ensemble forecasts with scalable AI inference. It further combines a spatiotemporal transformer with a global-attention algorithm that reduces complexity from quadratic to linear through scalable gradient propagation, enabling high-resolution, long-context Earth modeling. STORM scales to 74,400 GPUs on Frontier with 96--99\% strong-scaling efficiency and up to 6 ExaFLOPs sustained BF16 throughput, while enabling 32,768-member ensembles for uncertainty quantification in 34 seconds on
    
[^238]: 为什么微调会诱发幻觉以及如何修复它

    Why Fine-Tuning Encourages Hallucinations and How to Fix It

    [https://arxiv.org/abs/2604.15574](https://arxiv.org/abs/2604.15574)

    该论文提出一种基于自蒸馏的监督微调方法，通过正则化输出分布漂移，使模型在学习新事实的同时最大限度减少对预训练知识的幻觉，并证明在无需学习新知识时冻结参数组也能在保持任务性能的前提下降低幻觉。

    

    大语言模型容易产生与事实不符的幻觉陈述。这些错误的一个关键来源是监督微调（SFT）过程中接触到新的知识，这会增加相对于预训练期间所获知识的幻觉。由于这些错误是知识退化的副产品，我们探索能否利用已有的持续学习工具来缓解这一问题。我们提出了一种基于自蒸馏的SFT方法，通过正则化输出分布的漂移，在实现有效事实学习的同时，最大限度减少相对于已有知识的幻觉。我们还表明，当不需要获取新知识时，通过冻结参数组来抑制事实可塑性，可以在减少幻觉的同时保持任务性能。最后，我们研究了其内在机制，对比了容量限制、行为克隆和局部干扰等假说。我们的实验表明，主要的……

    arXiv:2604.15574v2 Announce Type: replace-cross  Abstract: Large language models are prone to hallucinating factually incorrect statements. A key source of these errors is exposure to new factual information through supervised fine-tuning (SFT), which can increase hallucinations w.r.t.~knowledge acquired during pre-training. Since these errors arise as a by-product of knowledge degradation, we explore whether established continual learning tools can mitigate them. We propose a self-distillation-based SFT method that facilitates effective factual learning while minimizing hallucinations w.r.t.~pre-existing knowledge by regularizing output-distribution drift. We also show that when new knowledge acquisition is unnecessary, suppressing factual plasticity by freezing parameter groups preserves task performance while reducing hallucinations. Lastly, we investigate the mechanism, contrasting capacity limitations, behavior cloning, and localized interference. Our experiments show that a main 
    
[^239]: 是什么驱动了表征转向？关于转向拒绝行为的机制案例研究

    What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

    [https://arxiv.org/abs/2604.08524](https://arxiv.org/abs/2604.08524)

    本研究通过多词元激活修补框架对LLM拒绝行为的转向机制进行案例研究，发现不同转向方法在同一层利用功能可互换的回路，且转向向量主要通过注意力机制的OV回路发挥作用而几乎不依赖QK回路。

    

    将转向向量应用于大型语言模型（LLM）是一种高效且有效的模型对齐技术，但我们对其工作原理缺乏可解释的解释——具体来说，转向向量影响了哪些内部机制，以及这如何导致不同的模型输出。为了探究转向向量有效性的因果机制，我们对拒绝行为进行了全面的案例研究。我们提出了一个多词元激活修补框架，并发现不同的转向方法在同一层应用时利用的是功能上可互换的回路。这些回路揭示出，转向向量主要通过OV回路与注意力机制交互，而在很大程度上忽略了QK回路。在转向过程中冻结所有注意力分数，在三个模型家族上仅导致8.83%的性能下降。对被转向OV回路的数学分解进一步揭示了……

    arXiv:2604.08524v2 Announce Type: replace-cross  Abstract: Applying steering vectors to large language models (LLMs) is an efficient and effective model alignment technique, but we lack an interpretable explanation for how it works--specifically, what internal mechanisms steering vectors affect and how this results in different model outputs. To investigate the causal mechanisms underlying the effectiveness of steering vectors, we conduct a comprehensive case study on refusal. We propose a multi-token activation patching framework and discover that different steering methodologies leverage functionally interchangeable circuits when applied at the same layer. These circuits reveal that steering vectors primarily interact with the attention mechanism through the OV circuit while largely ignoring the QK circuit. Freezing all attention scores during steering drops performance by only 8.83% across three model families. A mathematical decomposition of the steered OV circuit further reveals s
    
[^240]: 面向上下文密集型任务的KV缓存卸载

    KV Cache Offloading for Context-Intensive Tasks

    [https://arxiv.org/abs/2604.08426](https://arxiv.org/abs/2604.08426)

    该论文创建并发布了Text2JSON基准测试，揭示现代KV缓存卸载技术在需要从输入提示中提取大量信息的上下文密集型任务上，会导致Llama 3和Qwen 3模型出现显著的性能下降。

    

    随着各类应用对长上下文大语言模型（LLM）需求的不断增长，键值（KV）缓存已成为延迟和内存占用的关键瓶颈。近来，KV缓存卸载已成为一种有前景的方法，可在保持精度的同时减少内存占用和推理延迟。以往的评估工作主要集中于不需要从上下文中提取大量信息的任务。在本工作中，我们研究了KV缓存卸载在上下文密集型任务上的表现：即那些求解过程需要从输入提示中查找大量信息的问题。我们创建并发布了Text2JSON基准测试，这是一个高度上下文密集型的任务，需要从原始文本中提取结构化知识。我们在Text2JSON以及其他上下文密集型任务上对现代KV卸载技术进行了评估，发现Llama 3和Qwen 3模型均出现了显著的性能下降。我们的分析识别出两个关键原因（摘要原文在此处被截断）。

    arXiv:2604.08426v5 Announce Type: replace-cross  Abstract: With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key re
    
[^241]: 过程感知人工智能在降雨径流模拟中的应用：一种具有水文过程约束的质量守恒神经框架

    Process-Aware AI for Rainfall-Runoff Modeling: A Mass-Conserving Neural Framework with Hydrological Process Constraints

    [https://arxiv.org/abs/2603.25093](https://arxiv.org/abs/2603.25093)

    提出了一种质量守恒感知器（MCP）框架，通过在单一存储单元中逐步嵌入有界土壤蓄水、入渗、地表积水、地下水位动态等物理水文过程约束，在保证质量守恒的同时提升了降雨径流模拟的预测精度与物理可解释性。

    

    机器学习模型在水文应用中能够达到较高的预测精度，但往往缺乏物理可解释性。质量守恒感知器提供了一种物理感知的人工智能框架，在强制执行守恒原理的同时，允许从数据中学习水文过程关系。在本研究中，我们研究了如何在单个MCP存储单元内逐步嵌入具有物理意义的水文过程表示，从而提高降雨径流模拟的预测能力和可解释性。从最小化的MCP公式出发，我们依次引入了有界土壤蓄水容量、状态相关的导水率、可变孔隙度、入渗能力、地表积水、垂直排水以及非线性地下水位动态。所得到的这一系列过程感知MCP模型层级，在美国大陆五个水文气候区的15个流域上进行了评估。

    arXiv:2603.25093v2 Announce Type: replace  Abstract: Machine learning models can achieve high predictive accuracy in hydrological applications but often lack physical interpretability. The Mass-Conserving Perceptron (MCP) provides a physics-aware artificial intelligence (AI) framework that enforces conservation principles while allowing hydrological process relationships to be learned from data. In this study, we investigate how progressively embedding physically meaningful representations of hydrological processes within a single MCP storage unit improves predictive skill and interpretability in rainfall-runoff modeling. Starting from a minimal MCP formulation, we sequentially introduce bounded soil storage, state-dependent conductivity, variable porosity, infiltration capacity, surface ponding, vertical drainage, and nonlinear water-table dynamics. The resulting hierarchy of process-aware MCP models is evaluated across 15 catchments spanning five hydroclimatic regions of the continen
    
[^242]: 神经偏微分方程求解器的严格误差认证：从经验残差到解的保证

    Rigorous Error Certification for Neural PDE Solvers: From Empirical Residuals to Solution Guarantees

    [https://arxiv.org/abs/2603.19165](https://arxiv.org/abs/2603.19165)

    该论文的核心贡献是建立了将残差控制与解空间误差联系起来的泛化界，证明当神经逼近位于解空间紧子集内时，残差误差趋于零可保证收敛到真实解，从而为神经偏微分方程求解器提供了严格的误差认证。

    

    偏微分方程的不确定性量化传统上建立在离散化理论基础上，其解误差通过网格/格点细化来控制。物理信息神经网络（PINN）从根本上偏离了这一范式：它们通过在配点上最小化残差损失来逼近解，这引入了来自优化、采样、表示和过拟合的新误差来源。因此，解空间中的泛化误差仍然是一个悬而未决的问题。我们的主要理论贡献建立了将残差控制与解空间误差联系起来的泛化界。我们证明，当神经逼近位于解空间的紧子集内时，残差误差趋于零即可保证收敛到真实解。我们推导了确定性和概率性的收敛结果，并提供了将残差、边界和初始（条件误差转化为可认证的泛化界）

    arXiv:2603.19165v2 Announce Type: replace  Abstract: Uncertainty quantification for partial differential equations is traditionally grounded in discretization theory, where solution error is controlled via mesh/grid refinement. Physics-informed neural networks fundamentally depart from this paradigm: they approximate solutions by minimizing residual losses at collocation points, introducing new sources of error arising from optimization, sampling, representation, and overfitting. As a result, the generalization error in the solution space remains an open problem. Our main theoretical contribution establishes generalization bounds that connect residual control to solution-space error. We prove that when neural approximations lie in a compact subset of the solution space, vanishing residual error guarantees convergence to the true solution. We derive deterministic and probabilistic convergence results and provide certified generalization bounds translating residual, boundary, and initial
    
[^243]: Adam随机梯度下降优化方法的一致先验界与误差分析

    Uniform a priori bounds and error analysis for the Adam stochastic gradient descent optimization method

    [https://arxiv.org/abs/2603.18899](https://arxiv.org/abs/2603.18899)

    本工作的关键贡献是为Adam优化器建立了一致先验界，从而首次为一大类强凸随机优化问题提供了无条件的误差分析，摆脱了以往收敛性分析对“Adam保持一致有界”假设的依赖。

    

    arXiv:2603.18899v2 公告类型：替换 摘要：由Kingma和Ba（2014）提出的自适应矩估计优化器，大概是人工智能（AI）系统中用于训练深度神经网络（DNN）的最流行的随机梯度下降（SGD）优化方法。尽管它在AI系统训练中取得了突破性的成功，但为Adam提供完整的误差分析仍然是一个悬而未决的研究问题，不仅对于优化深度神经网络如此，即使将其应用于强凸随机优化问题（SOPs）也是如此。文献中先前关于强凸随机优化问题的误差分析结果提供的是条件收敛性分析，这些分析依赖于一个假设，即Adam不会发散到无穷大而是保持一致有界。本工作的关键贡献在于为Adam建立了一致先验界，从而首次为一大类强凸随机优化问题提供了Adam的无条件误差分析。

    arXiv:2603.18899v2 Announce Type: replace  Abstract: The adaptive moment estimation (Adam) optimizer proposed by Kingma & Ba (2014) is presumably the most popular stochastic gradient descent (SGD) optimization method for the training of deep neural networks (DNNs) in artificial intelligence (AI) systems. Despite its groundbreaking success in the training of AI systems, it still remains an open research problem to provide a complete error analysis of Adam, not only for optimizing DNNs but even when applied to strongly convex stochastic optimization problems (SOPs). Previous error analysis results for strongly convex SOPs in the literature provide conditional convergence analyses that rely on the assumption that Adam does not diverge to infinity but remains uniformly bounded. It is the key contribution of this work to establish uniform a priori bounds for Adam and, thereby, to provide -- for the first time -- an unconditional error analysis for Adam for a large class of strongly convex S
    
[^244]: MineDraft：一种批量并行投机解码框架

    MineDraft: A Framework for Batch Parallel Speculative Decoding

    [https://arxiv.org/abs/2603.18016](https://arxiv.org/abs/2603.18016)

    MineDraft提出一种批量并行投机解码框架，通过同时维护两批请求，将一批的草稿生成与另一批的验证重叠执行，有效隐藏草稿延迟，相比标准投机解码吞吐量最高提升75%、端到端延迟最高降低39%。

    

    投机解码（SD）通过使用较小的草稿模型提出草稿token，再由较大的目标模型进行验证，从而加速大语言模型的推理。然而，标准SD的性能往往受限于草稿生成与验证阶段的严格顺序执行。为解决这一问题，本文提出了MineDraft，一种批量并行投机解码（PSD）框架，旨在通过与验证过程重叠来有效隐藏草稿生成延迟。我们的理论分析表明，PSD比标准SD的效率显著更高。MineDraft通过一种新颖的批量并行设计实现了PSD：该设计同时维护两批请求，将一批请求的草稿生成与另一批请求的验证重叠进行。实验结果显示，与标准SD相比，MineDraft在吞吐量（最高提升75%）和端到端延迟（最高降低39%）方面均有显著改进。此外，我们还实现了……（摘要原文截断）

    arXiv:2603.18016v3 Announce Type: replace-cross  Abstract: Speculative decoding (SD) accelerates large language model inference by using a smaller draft model to propose draft tokens that are subsequently verified by a larger target model. However, the performance of standard SD is often limited by the strictly sequential execution of these drafting and verification stages. To address this, this paper proposes MineDraft, a batch parallel speculative decoding (PSD) framework designed to effectively hide drafting latency by overlapping it with verification. Our theoretical analysis shows that PSD is substantially more efficient than standard SD. MineDraft realizes the PSD through a novel batch-parallel design that maintains two batches of requests, overlapping drafting for one batch with verification for the other. Our experimental results show significant improvements of \alg{} in both throughput (up to 75%) and end-to-end latency (up to 39%) over standard SD. Furthermore, we have imple
    
[^245]: SCALE：用于虚拟细胞扰动预测的可扩展条件性图谱级端点传输模型

    SCALE:Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction

    [https://arxiv.org/abs/2603.17380](https://arxiv.org/abs/2603.17380)

    SCALE是一种将细胞表示为无序集合的条件传输模型，无需细胞级配对即可预测扰动后的细胞群体，在基因、化学、发育、免疫扰动及CRISPR数据上均优于现有方法。

    

    虚拟细胞模型旨在预测细胞群体如何响应扰动，但对照组和处理组细胞是以非配对群体的形式测量的，这使得学习扰动特异性效应变得复杂。我们提出了SCALE，一种条件传输模型，它将细胞表示为无序集合，无需细胞级别匹配即可预测处理后的细胞群体。共享的集合感知编码器和条件DiT骨干网络学习潜在传输，使端点监督直接与差异对齐，无需辅助的差异目标函数。在基因、化学、发育和免疫扰动等各类场景中，SCALE成功恢复了基因表达变化、响应方向和群体结构。在具有显著细胞系效应的CRISPR数据中，SCALE在七项指标上均优于竞争方法，并保持了基因靶点表示之间的区分度，而不是将它们坍缩到共享区域。SCALE还进一步对细胞因子进行了优先级排序预测……

    arXiv:2603.17380v3 Announce Type: replace-cross  Abstract: Virtual-cell models aim to predict how cell populations respond to perturbations, but control and treated cells are measured as unpaired populations, complicating the learning of perturbation-specific effects. We present SCALE, a conditional transport model that represents cells as unordered sets and predicts treated populations without cell-level matching. A shared set-aware encoder and conditional DiT backbone learn latent transport, making endpoint supervision directly delta-aligned without an auxiliary delta objective. Across genetic, chemical, developmental and immune perturbations, SCALE recovered gene-expression changes, response directions and population structure. In CRISPR data with dominant cell-line effects, SCALE outperformed competing methods across seven metrics and maintained separation among gene-target representations rather than collapsing them into a shared region. SCALE further prioritized cytokines predict
    
[^246]: RetroReasoner：用于策略性逆合成预测的推理大语言模型

    RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction

    [https://arxiv.org/abs/2603.12666](https://arxiv.org/abs/2603.12666)

    本文提出RetroReasoner，一个通过结构化切断理由的监督微调和往返奖励强化学习训练的逆合成推理大语言模型，能够显式模拟化学家的键切断策略思维并验证预测反应物的有效性。

    

    逆合成预测旨在识别能够合成给定产物分子的反应物。尽管分子大语言模型（LLMs）近期展现出令人期待的结果，但大多数现有方法要么直接生成反应物，要么仅提供通用的产物层面分析，而未能对键切断策略进行显式推理，以论证特定反应物的选择合理性。本文提出了RetroReasoner，一个能够捕捉化学家策略性切断思维的逆合成推理模型。RetroReasoner通过监督微调和强化学习进行训练。在监督微调方面，SyntheticRetro生成结构化的切断理由并与反应物预测配对。在强化学习方面，采用往返奖励机制，将预测的反应物输入正向合成模型进行评估，并对能够重构原始产物的预测给予奖励。

    arXiv:2603.12666v3 Announce Type: replace-cross  Abstract: Retrosynthesis prediction aims to identify reactants that can synthesize a given product molecule. Although molecular large language models (LLMs) have recently shown promising results, most existing methods either generate reactants directly or provide only generic product-level analysis, without explicitly reasoning about bond-disconnection strategies that justify specific reactant choices. This paper proposes RetroReasoner, a retrosynthetic reasoning model that captures chemists' strategic disconnection-based thinking. RetroReasoner is trained with supervised fine-tuning and reinforcement learning. For supervised fine-tuning, SyntheticRetro generates structured disconnection rationales paired with reactant predictions. For reinforcement learning, a round-trip reward evaluates predicted reactants by passing them through a forward synthesis model and rewarding predictions that reconstruct the original product. RetroReasoner ca
    
[^247]: HEAL：基于后见之明熵辅助学习的推理蒸馏

    HEAL: Hindsight Entropy-Assisted Learning for Reasoning Distillation

    [https://arxiv.org/abs/2603.10359](https://arxiv.org/abs/2603.10359)

    HEAL 提出了一种无需强化学习的推理蒸馏框架，通过熵动态检测推理断点并注入事后提示来修复失败轨迹，突破了传统拒绝采样造成的“教师天花板”，从而将大型推理模型的推理能力更有效地蒸馏到小模型中。

    

    将大型推理模型（LRM）的推理能力蒸馏到更小的模型中，通常受到拒绝采样局限性的制约。标准方法将教师模型视为静态过滤器，丢弃了教师模型无法独立探索出有效解的复杂“边角案例”问题，从而人为地为学生模型设置了一个“教师天花板”。在本工作中，我们提出了后见之明熵辅助学习，这是一个无需强化学习（RL-free）的框架，旨在弥合这一推理差距。借鉴最近发展区（ZPD）教育理论，HEAL 协同整合了三个核心模块：（1）引导式熵辅助修复（GEAR），一种主动干预机制，通过熵动态检测关键推理断点，并注入有针对性的事后提示来修复中断的推理轨迹；（2）困惑度-不确定性比率估计器（PURE），一种基于比率的过滤启发式方法，用于降低高异常……（摘要原文在此处截断）

    arXiv:2603.10359v2 Announce Type: replace  Abstract: Distilling reasoning capabilities from Large Reasoning Models (LRMs) into smaller models is typically constrained by the limitations of rejection sampling. Standard methods treat the teacher as a static filter, discarding complex "corner-case" problems where the teacher fails to explore valid solutions independently, thereby creating an artificial "Teacher Ceiling" for the student. In this work, we propose Hindsight Entropy-Assisted Learning (HEAL), an RL-free framework designed to bridge this reasoning gap. Drawing on the educational theory of the Zone of Proximal Development (ZPD), HEAL synergizes three core modules: (1) Guided Entropy-Assisted Repair (GEAR), an active intervention mechanism that detects critical reasoning breakpoints via entropy dynamics and injects targeted hindsight hints to repair broken trajectories; (2) Perplexity-Uncertainty Ratio Estimator (PURE), a ratio-based filtering heuristic that reduces high-anomaly 
    
[^248]: 面向科学的MMAI Gym：训练用于药物发现的液体基础模型

    MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery

    [https://arxiv.org/abs/2603.03517](https://arxiv.org/abs/2603.03517)

    本文提出MMAI Gym for Science一站式训练框架，通过教会基础模型“分子的语言”，训练出更小规模的液体基础模型（LFM），在分子优化、ADMET预测等药物发现任务上超越了规模大得多的通用或专业模型。

    

    摘要：依赖上下文学习的通用大型语言模型（LLM）无法可靠地提供药物发现任务所需的科学理解和性能。仅仅增加模型规模或引入推理标记并不能带来显著的性能提升。为了解决这一差距，我们推出了面向科学的MMAI Gym（MMAI Gym for Science），这是一个一站式平台，提供分子数据格式与模态，以及面向特定任务的推理、训练和基准测试方案，旨在教会基础模型“分子的语言”，从而解决实际的药物发现问题。我们使用MMAI Gym训练了一个高效的液体基础模型（LFM）用于这些应用，证明了更小规模、有针对性训练的基础模型在分子基准测试中能够超越规模大得多的通用模型或专业模型。在关键的药物发现任务中——包括分子优化、ADMET性质预测等……

    arXiv:2603.03517v2 Announce Type: replace-cross  Abstract: General-purpose large language models (LLMs) that rely on in-context learning do not reliably deliver the scientific understanding and performance required for drug discovery tasks. Simply increasing model size or introducing reasoning tokens does not yield significant performance gains. To address this gap, we introduce the MMAI Gym for Science, a one-stop shop molecular data formats and modalities as well as task-specific reasoning, training, and benchmarking recipes designed to teach foundation models the 'language of molecules' in order to solve practical drug discovery problems. We use MMAI Gym to train an efficient Liquid Foundation Model (LFM) for these applications, demonstrating that smaller, purpose-trained foundation models can outperform substantially larger general-purpose or specialist models on molecular benchmarks. Across essential drug discovery tasks - including molecular optimization, ADMET property predictio
    
[^249]: 基于机器学习的冲击响应谱曲线到冲击时间序列的逆重构

    Inverse Reconstruction of Shock Time Series from Shock Response Spectrum Curves using Machine Learning

    [https://arxiv.org/abs/2603.03229](https://arxiv.org/abs/2603.03229)

    提出了一种条件变分自编码器（CVAE）模型，学习从冲击响应谱到加速度时间序列的数据驱动逆映射，无需迭代优化即可生成高谱保真度的时域信号。

    

    冲击响应谱（SRS）被广泛用于表征单自由度（SDOF）系统对瞬态加速度的响应。由于从加速度时程到SRS的映射是非线性且多对一的，从目标谱重构时域信号本质上是一个不适定问题。传统方法通常通过迭代优化来解决这一问题，典型做法是将信号表示为指数衰减正弦波之和，但这类方法计算代价高昂，且受限于预定义的基函数。我们提出了一种条件变分自编码器（CVAE），用于学习从SRS到加速度时间序列的数据驱动逆映射。模型训练完成后，无需迭代优化即可生成与指定目标谱一致的信号。实验表明，与传统技术相比，该方法的谱保真度更高，并对未见过的数据具有很强的泛化能力。

    arXiv:2603.03229v3 Announce Type: replace  Abstract: The shock response spectrum (SRS) is widely used to characterize the response of single-degree-of-freedom (SDOF) systems to transient accelerations. Because the mapping from acceleration time history to SRS is nonlinear and many-to-one, reconstructing time-domain signals from a target spectrum is inherently ill-posed. Conventional approaches address this problem through iterative optimization, typically representing signals as sums of exponentially decayed sinusoids, but these methods are computationally expensive and constrained by predefined basis functions.   We propose a conditional variational autoencoder (CVAE) that learns a data-driven inverse mapping from SRS to acceleration time series. Once trained, the model generates signals consistent with prescribed target spectra without requiring iterative optimization. Experiments demonstrate improved spectral fidelity relative to classical techniques, strong generalization to unseen
    
[^250]: 信道自适应边缘AI：通过将计算复杂度适配于信道状态以最大化推理吞吐量

    Channel-Adaptive Edge AI: Maximizing Inference Throughput by Adapting Computational Complexity to Channel States

    [https://arxiv.org/abs/2603.03146](https://arxiv.org/abs/2603.03146)

    本文提出了一个可处理的端到端推理精度分析模型，并据此设计了信道自适应AI算法，通过根据信道状态动态调整模型计算复杂度（利用早退机制），在时延和精度约束下最大化边缘推理吞吐量。

    

    通信与计算一体化（IC²）已成为在第六代（6G）网络中实现高效边缘推理的新范式。然而，由于缺乏一个可处理的理论框架来表征端到端（E2E）推理性能，IC²技术的设计受到了阻碍。该指标非常复杂，因为它需要同时考虑信道失真以及人工智能（AI）模型架构和计算复杂度。在这项工作中，我们通过开发一个可处理的端到端推理精度分析模型来应对这一挑战，并利用该模型设计了一种信道自适应AI算法，在时延和精度约束下最大化推理吞吐量（即边缘处理速率，EPR）。具体而言，我们考虑了一种边缘推理系统，其中服务器部署具有早退机制的骨干模型，从而实现灵活的计算（摘要在此处截断）。

    arXiv:2603.03146v2 Announce Type: replace-cross  Abstract: \emph{Integrated communication and computation} (IC$^2$) has emerged as a new paradigm for enabling efficient edge inference in sixth-generation (6G) networks. However, the design of IC$^2$ technologies is hindered by the lack of a tractable theoretical framework for characterizing \emph{end-to-end} (E2E) inference performance. The metric is highly complicated as it needs to account for both channel distortion and artificial intelligence (AI) model architecture and computational complexity. In this work, we address this challenge by developing a tractable analytical model for E2E inference accuracy and leveraging it to design a \emph{channel-adaptive AI} algorithm that maximizes inference throughput, referred to as the edge processing rate (EPR), under latency and accuracy constraints. Specifically, we consider an edge inference system in which a server deploys a backbone model with early exit, which enables flexible computatio
    
[^251]: 利用数据同化实现非定常流动降阶模型的高效实时自适应

    Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation

    [https://arxiv.org/abs/2602.23188](https://arxiv.org/abs/2602.23188)

    该论文提出一种基于变分自编码器与Transformer的参数化降阶模型高效再训练策略，仅利用稀疏观测数据和少量计算时间即可将模型实时自适应到新的流动工况，精度接近完全再训练。

    

    我们提出了一种针对参数化降阶模型（ROM）的高效再训练策略，其精度可与完全再训练相媲美，同时仅需一小部分计算时间，且仅依赖于对全系统的稀疏观测。该架构采用编码-处理-解码结构：利用变分自编码器（VAE）进行降维，并利用Transformer网络演化潜在状态并建模动力学特性。该ROM由一个外部控制变量进行参数化（在Navier-Stokes设定中为雷诺数），Transformer借助注意力机制同时捕获时间依赖关系和参数效应。概率VAE支持对轨迹集合进行随机采样，并通过前两阶矩提供预测均值和不确定性量化。在有限的动力学状态集合上进行初始训练后，模型被自适应于分布外……（原文摘要在此处截断）

    arXiv:2602.23188v2 Announce Type: replace  Abstract: We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-process-decode structure: a Variational Autoencoder (VAE) to perform dimensionality reduction, and a transformer network to evolve the latent states and model the dynamics. The ROM is parameterized by an external control variable, the Reynolds number in the Navier-Stokes setting, with the transformer exploiting attention mechanisms to capture both temporal dependencies and parameter effects. The probabilistic VAE enables stochastic sampling of trajectory ensembles, providing predictive means and uncertainty quantification through the first two moments. After initial training on a limited set of dynamical regimes, the model is adapted to out-o
    
[^252]: 学习记忆：面向长上下文推理的记忆智能体端到端训练

    Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning

    [https://arxiv.org/abs/2602.18493](https://arxiv.org/abs/2602.18493)

    该论文提出统一记忆智能体（UMA），通过任务分层GRPO算法端到端训练单一策略来维护可复用的结构化外部记忆库，并配套提出Ledger-QA诊断基准，显著提升了长上下文推理中跨会话状态跟踪与问答的性能。

    

    长上下文大语言模型和检索增强生成将状态跟踪与证据整合推迟到查询时刻，当事实发生演变且答案依赖于潜在状态时，这种方式十分脆弱。我们提出了统一记忆智能体（UMA）来应对一对多的场景：与查询无关的外部记忆从数据流中一次性构建，并可在多个未来的问答会话中重复使用。单一策略通过增删改查（CRUD）操作维护一个结构化的记忆库，并结合记忆库与原始上下文进行回答。任务分层GRPO（Task-Stratified GRPO）利用从每个采样记忆状态分支出的问答轨迹的平均奖励来监督记忆维护，同时对记忆组与逐问题问答组分别进行归一化。我们还提出了Ledger-QA，一个针对累积更新之上长程状态跟踪的诊断性基准。在16k预算下，UMA-Generalist在测试时学习与准确性方面，在所有对比方法中取得了最高的平均分数。

    arXiv:2602.18493v2 Announce Type: replace-cross  Abstract: Long-context LLMs and Retrieval-Augmented Generation defer state tracking and evidence consolidation to query time, which is brittle when facts evolve and answers depend on latent states. We introduce Unified Memory Agent (UMA) for a one-to-many setting: query-agnostic external memory is constructed once from a stream and reused across multiple future QA sessions. A single policy maintains a structured Memory Bank through CRUD operations and answers using both the Memory Bank and raw context. Task-Stratified GRPO uses the mean reward of QA trajectories branching from each sampled memory state to supervise memory maintenance, while normalizing memory and per-question QA groups separately. We also introduce Ledger-QA, a diagnostic benchmark for long-horizon state tracking over accumulated updates. At the 16k budget, UMA-Generalist achieves the highest average score among compared methods across the test-time-learning and accurate
    
[^253]: 本体引导的神经符号推理：用数学领域知识为语言模型提供形式化基础

    Ontology-Guided Neuro-Symbolic Inference: Grounding Language Models with Mathematical Domain Knowledge

    [https://arxiv.org/abs/2602.17826](https://arxiv.org/abs/2602.17826)

    该论文提出一种结合OpenMath本体、混合检索与交叉编码器重排序的神经符号流水线，将数学领域知识注入语言模型提示中，实验表明高质量检索的本体上下文能提升模型在MATH基准上的表现，但不相关上下文会损害性能。

    

    语言模型存在根本性的局限——幻觉、脆弱性以及缺乏形式化基础——这些问题在需要可验证推理的高风险专业领域尤为突出。本研究探讨了形式化领域领域能否通过检索增强生成来提升语言模型的可靠性。以数学作为概念验证，作者实现了一个神经符号流水线，利用OpenMath本体，结合混合检索和交叉编码器重排序技术，将相关定义注入模型提示中。在MATH基准上使用三个开源模型进行的评估表明，当检索质量较高时，本体引导的上下文能够提升性能，但不相关的上下文反而会显著降低性能——这一结果既展现了神经符号方法的前景，也揭示了其面临的挑战。

    arXiv:2602.17826v2 Announce Type: replace  Abstract: Language models exhibit fundamental limitations -- hallucination, brittleness, and lack of formal grounding -- that are particularly problematic in high-stakes specialist fields requiring verifiable reasoning. I investigate whether formal domain ontologies can enhance language model reliability through retrieval-augmented generation. Using mathematics as proof of concept, I implement a neuro-symbolic pipeline leveraging the OpenMath ontology with hybrid retrieval and cross-encoder reranking to inject relevant definitions into model prompts. Evaluation on the MATH benchmark with three open-source models reveals that ontology-guided context improves performance when retrieval quality is high, but irrelevant context actively degrades it -- highlighting both the promise and challenges of neuro-symbolic approaches.
    
[^254]: 知识蒸馏真的更环保吗？——机器翻译中的案例研究

    Is Knowledge Distillation Actually Greener? A Case Study in Machine Translation

    [https://arxiv.org/abs/2602.09691](https://arxiv.org/abs/2602.09691)

    该研究首次借助机器学习生命周期评估工具，从环境成本角度系统评估机器翻译中的知识蒸馏方法，发现摊销蒸馏成本所需的部署量取决于服务方式，且在批处理下可能变化数个数量级。

    

    知识蒸馏（KD）是一种将较大的教师系统压缩为更小的学生系统的技术。在机器翻译中，知识蒸馏通常通过翻译质量和推理效率来评估，而没有共同考虑生产和部署蒸馏系统所产生的环境成本。我们在定制的机器翻译模型和大语言模型上评估了具有代表性的知识蒸馏方法，同时考虑翻译质量和计算成本，并使用机器学习生命周期评估工具，该工具能够核算知识蒸馏模型整个生命周期中的成本。我们的关键发现是：摊销知识蒸馏成本所需的部署量取决于服务方式，并且在批处理条件下可能变化数个数量级。我们还提供了在质量和计算约束下选择、开发和评估知识蒸馏方法的可操作指导。

    arXiv:2602.09691v2 Announce Type: replace  Abstract: Knowledge distillation (KD) is a technique to compress a larger teacher system into a smaller student. In machine translation, KD is commonly evaluated through translation quality and inference efficiency, without jointly accounting for the environmental costs of producing and deploying the distilled system. We evaluate representative KD methods both on bespoke MT models and LLMs, by considering both translation quality and computational cost, using the Machine Learning Life Cycle Assessment tool, which accounts for costs throughout the KD model life cycle. Our key finding is that the deployment volume required to amortize KD is serving-dependent and can shift by several orders of magnitude under batching. We include actionable guidance for selecting, developing, and evaluating KD methods under quality and compute-induced constraints.
    
[^255]: 持续熵作为相变的探测器

    Persistent Entropy as a Detector of Phase Transitions

    [https://arxiv.org/abs/2602.09058](https://arxiv.org/abs/2602.09058)

    本文建立了与模型无关的理论定理，通过识别持续权重中的“分散-凝聚”机制并推导出两状态间熵差的显式高概率下界，首次为利用持续熵检测相变提供了严格的理论保证，并据此证明卷积网络学习滤波器的环形组织源于一次尖锐的拓扑相变。

    

    持续熵是持续性条形码的一种标量摘要，被广泛用于检测状态变化，然而目前尚无理论阐明条形码中的结构性变化何时必然会导致可检测的熵变化。我们建立了一个与模型无关的定理来提供此类条件。通过将持久图视为由控制参数索引的随机对象，我们在归一化持久权重中识别出一种“分散-凝聚”机制，并推导出两种状态之间熵差的显式下界，该下界在有限样本量下以高概率成立，且对条形寿命的绝对尺度不敏感。我们还给出了一套在经验条形码上验证这些假设的程序。应用于卷积网络时，该准则表明 Gabrielsson 和 Carlsson 所报告的学习滤波器的环形组织是通过一次尖锐的拓扑相变而产生的，并定位了该相变的发生起点。

    arXiv:2602.09058v2 Announce Type: replace-cross  Abstract: Persistent entropy is a scalar summary of persistence barcodes widely used to detect regime changes, yet there is no account of when a structural change in a barcode must produce a detectable change in entropy. We establish a model-agnostic theorem supplying such conditions. Treating persistence diagrams as random objects indexed by a control parameter, we identify a dispersion-condensation mechanism in the normalized persistence weights and derive an explicit lower bound on the entropy difference between the two regimes, valid with high probability at finite sample size and insensitive to the absolute scale of bar lifetimes. We also give a procedure for verifying the hypotheses on empirical barcodes. Applied to convolutional networks, the criterion shows that the circular organization of learned filters reported by Gabrielsson and Carlsson emerges through a sharp topological phase transition, and locates its onset: within a fe
    
[^256]: 深空去噪：面向天文成像的基于物理的CCD噪声形成模型

    Denoising the Deep Sky: Physics-Based CCD Noise Formation for Astronomical Imaging

    [https://arxiv.org/abs/2601.23276](https://arxiv.org/abs/2601.23276)

    本文提出了一种基于物理的CCD噪声合成框架，通过建模光子散粒噪声、暗电流、读出效应等多种噪声来源，并利用未配准曝光叠加生成高信噪比基础图像，从而构建大量成对训练数据，实现天文图像的监督学习去噪。

    

    天文成像在实际观测条件下仍然受到噪声的限制。标准的校准流程能够去除结构化伪影，但在很大程度上无法解决随机噪声问题。尽管基于学习的去噪方法已展现出强大的潜力，但其进展受到成对训练数据稀缺以及科学工作流程对物理可解释模型要求的制约。我们提出了一个专门针对望远镜中CCD噪声形成过程的、基于物理的噪声合成框架。该流程建模了光子散粒噪声、光响应非均匀性、暗电流噪声、读出效应，以及由宇宙射线撞击和热像素引起的局部异常值。为了获得用于合成的低噪声输入，我们对多幅未配准的曝光图像进行叠加，生成高信噪比的基础图像。利用我们的噪声模型从这些基础图像合成逼真的含噪图像，从而能够构建大量用于监督学习的成对数据集。大量的实验……

    arXiv:2601.23276v4 Announce Type: replace-cross  Abstract: Astronomical imaging remains noise-limited under practical observing conditions. Standard calibration pipelines remove structured artifacts but largely leave stochastic noise unresolved. Although learning-based denoising has shown strong potential, progress is constrained by scarce paired training data and the requirement for physically interpretable models in scientific workflows. We propose a physics-based noise synthesis framework tailored to CCD noise formation in the telescope. The pipeline models photon shot noise, photo-response non-uniformity, dark-current noise, readout effects, and localized outliers arising from cosmic-ray hits and hot pixels. To obtain low-noise inputs for synthesis, we stack multiple unregistered exposures to produce high-SNR bases. Realistic noisy counterparts synthesized from these bases using our noise model enable the construction of abundant paired datasets for supervised learning. Extensive e
    
[^257]: 实体对齐基础模型中推理视界的突破

    Breaking the Reasoning Horizon in Entity Alignment Foundation Models

    [https://arxiv.org/abs/2601.21174](https://arxiv.org/abs/2601.21174)

    提出了一种由并行编码策略驱动的实体对齐基础模型，利用种子对齐对作为局部锚点进行锚点条件消息传递，突破了传统模型在稀疏异构知识图谱上捕获长距离依赖的“推理视界”限制，实现了无需重新训练即可对齐未见知识图谱的能力。

    

    实体对齐（EA）对于知识图谱（KG）融合至关重要。现有的EA模型缺乏可迁移性，无法在无需重新训练的情况下对齐未见过的知识图谱。虽然使用图基础模型（GFMs）提供了一种解决方案，但我们发现直接将GFMs适配到EA任务在很大程度上仍然无效。这源于一个关键的“推理视界差距”：与GFMs中的链接预测不同，EA需要在稀疏且异构的KG结构上捕获长距离依赖关系。为了应对这一挑战，我们提出了一种由并行编码策略驱动的EA基础模型。我们利用种子EA对作为局部锚点来引导信息流，同时初始化并编码两个并行流。这促进了基于锚点条件的消息传递，并通过利用局部结构邻近性而非全局搜索，显著缩短了推理轨迹。此外，我们还引入了合并关系图来建模（原文在此处截断）……

    arXiv:2601.21174v3 Announce Type: replace  Abstract: Entity alignment (EA) is critical for knowledge graph (KG) fusion. Existing EA models lack transferability and are incapable of aligning unseen KGs without retraining. While using graph foundation models (GFMs) offer a solution, we find that directly adapting GFMs to EA remains largely ineffective. This stems from a critical "reasoning horizon gap": unlike link prediction in GFMs, EA necessitates capturing long-range dependencies across sparse and heterogeneous KG structuresTo address this challenge, we propose a EA foundation model driven by a parallel encoding strategy. We utilize seed EA pairs as local anchors to guide the information flow, initializing and encoding two parallel streams simultaneously. This facilitates anchor-conditioned message passing and significantly shortens the inference trajectory by leveraging local structural proximity instead of global search. Additionally, we incorporate a merged relation graph to model
    
[^258]: FloydNet：一种全局关系推理的学习范式

    FloydNet: A Learning Paradigm for Global Relational Reasoning

    [https://arxiv.org/abs/2601.19094](https://arxiv.org/abs/2601.19094)

    提出FloydNet与关键注意力机制（PA），借鉴Floyd–Warshall的“配对-枢轴”结构，通过维护有序对状态并在枢轴上聚合候选关系实现全局关系推理，并推广为支持有序k元组的k-FloydNet框架，其图判别能力与对应的WL同构测试相当。

    

    学习算法计算通常需要显式的关系中间状态，然而许多图处理器将其主要状态维持在单个实体上。我们提出了FloydNet和关键注意力，它维护有序对状态，并通过在由每个枢轴 j 所构成的 候选上进行注意力操作来更新目标关系。受Floyd–Warshall算法的“配对-枢轴”结构启发，PA以并行方式学习关系组合与枢轴加权，而非执行其有序的min-plus递归。k-FloydNet框架将这一操作扩展到有序k元组，在注意力操作层面上，自注意力和PA分别是其k=1和k=2的特例。在原子元组初始化和不变读出的条件下，我们证明k-FloydNet的图判别能力不超过k-FWL；在BREC基准上，每个被评估的变体都与其对应WL参考的成功集合相匹配。

    arXiv:2601.19094v3 Announce Type: replace-cross  Abstract: Learning algorithmic computation often requires explicit relational intermediate states, yet many graph processors maintain their primary states on individual entities. We introduce \fnet and \textbf{Pivotal Attention} (PA), which maintain ordered pair states and update a target relation $(i,k)$ by attending over candidates formed from $(i,j)$ and $(j,k)$ for every pivot $j$. Motivated by the pair-and-pivot structure of Floyd--Warshall, PA learns relation composition and pivot weighting in parallel rather than executing its ordered min-plus recurrence. The \kfnet{k} framework extends this operation to ordered $k$-tuples, with Self-Attention and PA as its $k=1$ and $k=2$ cases at the attention-operation level. Under atomic tuple initialization and invariant readout, we show that \kfnet{k} is no more graph-discriminative than k-FWL; on BREC, each evaluated variant matches the success set of its corresponding WL reference. \fnet f
    
[^259]: 跨机械系统的冻结编码器异常检测审计：表示来源、校准与协议效应

    Auditing Frozen-Encoder Anomaly Detection Across Mechanical Systems: Representation Provenance, Calibration, and Protocol Effects

    [https://arxiv.org/abs/2601.11415](https://arxiv.org/abs/2601.11415)

    本论文对冻结编码器异常检测实验进行了可复现性审计，发现虽然数值判别结果可以复现，但其归因于干涉测量预训练的因果声明不被支持，作者因此撤回了该声明。

    

    本版本报告了对第1版中冻结编码器实验的可复现性审计。数值判别结果可以从保存的工件中复现，但其最初归因于干涉测量预训练的说法缺乏支持。发布的检查点包含一个嵌套的模型状态，加载时不会缺失参数；而若加载外层检查点字典，则几乎整个EfficientNet-B0特征堆栈将处于未初始化状态。被标记为干涉测量的保存嵌入的范数约为$10^{-12}$量级，与刚初始化的EfficientNet-B0网络相匹配，与保存的ImageNet嵌入相差超过十二个数量级。另一组单独保存的近零嵌入集合产生了几乎相同的IMS第四测试异常分数（$r=0.987$）和记录级判别性能（AUC为$0.9812$对$0.9818$）。因此，我们撤回关于IMS[摘要在此处截断]

    arXiv:2601.11415v2 Announce Type: replace-cross  Abstract: This version reports a reproducibility audit of the frozen-encoder experiments presented in version 1. The numerical discrimination results are reproducible from the preserved artifacts, but their original attribution to interferometric pretraining is not supported. The released checkpoint contains a nested model state that loads without missing parameters, whereas loading the outer checkpoint dictionary leaves almost the entire EfficientNet-B0 feature stack uninitialized. Preserved embeddings labelled as interferometric have norms of order $10^{-12}$, matching freshly initialized EfficientNet-B0 networks and differing by more than twelve orders of magnitude from the preserved ImageNet embeddings. A second, separately preserved near-zero embedding set produces almost the same IMS 4th-test anomaly scores ($r=0.987$) and record-level discrimination (AUC $0.9812$ versus $0.9818$).   We therefore withdraw the causal claim that IMS 
    
[^260]: DAGGER：面向数学问题可执行推理的干扰感知图生成

    DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems

    [https://arxiv.org/abs/2601.06853](https://arxiv.org/abs/2601.06853)

    该论文提出DAGGER方法，将含干扰信息的数学问题求解重构为显式建模干扰节点的可执行计算图生成，有效缓解了思维链推理在无关信息干扰下的严重性能退化。

    

    思维链提示已被广泛应用于数学问题求解，包括在低资源语言场景中，但其在无关上下文干扰下的行为仍未得到充分研究。为了系统地研究这一挑战，我们提出了DISTRACTMATH-BN，一个孟加拉语基准数据集，它在MGSM和MSVAMP的基础上增加了语义连贯但与计算无关的干扰信息。通过对七个参数量从3B到12B的模型进行评估，我们观察到干扰信息会导致显著的性能退化：标准模型最多下降41分，而专门强化推理能力的模型也下降了14至20分，且其 token 消耗量高达原来的五倍。我们提出DAGGER，将数学问题求解重新表述为可执行计算图的生成，并对干扰节点进行显式建模。通过监督微调并结合群体相对策略优化（GRPO）对Gemma-3模型进行微调，取得了可比的加权准确率……

    arXiv:2601.06853v3 Announce Type: replace  Abstract: Chain-of-Thought (CoT) prompting is widely adopted for mathematical problem solving, including in low-resource languages, yet its behavior under irrelevant context remains underexplored. To systematically study this challenge, we introduce DISTRACTMATH-BN, a Bangla benchmark that augments MGSM and MSVAMP with semantically coherent but computationally irrelevant information. Evaluating seven models ranging from 3B to 12B parameters, we observe substantial performance degradation under distractors: standard models drop by up to 41 points, while reasoning-specialized models decline by 14 to 20 points despite consuming five times more tokens. We propose {\dag}DAGGER, which reformulates mathematical problem solving as executable computational graph generation with explicit modeling of distractor nodes. Fine-tuning Gemma-3 models using supervised fine-tuning followed by Group Relative Policy Optimization achieves comparable weighted accura
    
[^261]: 针对基于Mamba的语言模型的隐状态投毒攻击

    Hidden State Poisoning Attacks against Mamba-based Language Models

    [https://arxiv.org/abs/2601.01972](https://arxiv.org/abs/2601.01972)

    该论文首次揭示了针对Mamba等状态空间语言模型的隐状态投毒攻击（HiSPA）——特定短输入短语可不可逆地覆盖模型隐藏状态导致部分失忆，并提出RoBench-25基准证实了包括520亿参数的Jamba混合模型在内的SSMs对此类攻击的脆弱性，而纯Transformer模型则不受影响。

    

    像Mamba这样的状态空间模型（SSMs）以线性时间复杂度为基于Transformer的语言模型提供了高效替代方案。然而，其对抗鲁棒性却鲜有研究。本文研究了特定短输入短语通过不可逆地覆盖模型隐藏状态中的信息，从而在此类模型中诱发部分“失忆”效应的现象，我们将其称为隐状态投毒攻击。我们提出的基准测试RoBench-25可以评估模型在遭受HiSPA攻击时的信息检索能力，并证实了SSMs对此类攻击的脆弱性。即使是最近的Jamba-1.7-Mini SSM-Transformer混合模型（520亿参数），在某些HiSPA触发器作用下也会在RoBench-25上完全失效，而纯Transformer模型则不会。我们还观察到，与纯Transformer不同，HiSPA触发器在流行的Open-Prompt-Injections基准测试中显著削弱了Jamba模型的表现。我们进一步表明，该理（摘要原文在此处截断）

    arXiv:2601.01972v5 Announce Type: replace-cross  Abstract: State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench-25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even the recent Jamba-1.7-Mini SSM--Transformer (a 52B hybrid model) collapses on RoBench-25 under some HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. We further show that the theo
    
[^262]: 建模非随机缺失时间序列数据中的信息中断

    Modeling Information Blackouts in Missing Not-At-Random Time Series Data

    [https://arxiv.org/abs/2601.01480](https://arxiv.org/abs/2601.01480)

    该论文提出了一种感知非随机缺失（MNAR）的潜在状态空间模型，用于建模交通传感器网络中的连续信息中断，证明当缺失机制依赖于潜在交通状态时，考虑这种依赖关系可显著提升数据插补精度与缺失检测性能。

    

    交通预测系统依赖于固定传感器网络，而这些网络经常出现连续性的数据中断。此类中断通常被当作可忽略的缺失数据处理，尽管数据丢失实际上可能取决于未观测到的交通状况。我们通过一个感知非随机缺失（MNAR）的潜在状态空间模型来研究这种可能性，该模型将线性交通动力学与伯努利缺失通道相结合，其缺失概率取决于潜在状态。推断采用扩展卡尔曼滤波器（EKF）以及随后的Rauch-Tung-Striebel（RTS）平滑，参数通过近似EM算法学习。我们使用一套无数据泄漏、月份平衡的300个独特的全视界对齐中断窗口数据集，对西雅图的交通数据进行评估。在该基准测试中，MAR-LDS达到4.264英里/小时的合并插补RMSE，而MNAR-LDS将其改进至4.177（差异为-0.086）；基于检测器聚类的自助法95%置信区间为[-0.182, -0.002]。因果性单步预测潜在表示将缺失检测的ROC-AUC从……

    arXiv:2601.01480v3 Announce Type: replace-cross  Abstract: Traffic forecasting systems rely on fixed sensor networks that frequently exhibit contiguous blackouts. Such outages are usually treated as ignorable missingness, although dropout can depend on unobserved traffic conditions. We study this possibility with an MNAR-aware latent state-space model that combines linear traffic dynamics with a Bernoulli missingness channel whose probability depends on the latent state. Inference uses an Extended Kalman Filter (EKF) followed by Rauch-Tung-Striebel (RTS) smoothing, and parameters are learned by approximate EM. We evaluate Seattle using a leakage-free, month-balanced set of 300 unique all-horizon-aligned blackout windows. On this benchmark, MAR-LDS attains 4.264 mph pooled imputation RMSE and MNAR-LDS improves it to 4.177 (difference -0.086); the detector-cluster bootstrap 95% interval is [-0.182,-0.002]. A causal one-step predicted latent representation raises missingness ROC-AUC from 
    
[^263]: CADKnitter：基于文本与几何引导的组合式CAD生成

    CADKnitter: Compositional CAD Generation from Text and Geometry Guidance

    [https://arxiv.org/abs/2512.11199](https://arxiv.org/abs/2512.11199)

    CADKnitter是一个组合式CAD生成框架，通过几何与文本双重引导的扩散采样，能够根据给定CAD模型的几何约束和文本提示的语义约束生成与之互补的新CAD零件。

    

    计算机辅助设计（CAD）将三维模型定义为紧凑、精确且可编辑的表示形式，使其对多个领域直接有用。近年来，CAD生成在学术界和工业界都获得了越来越多的关注。长期以来，制作CAD模型一直是一项繁琐且耗时的工作，对设计师的精确性和专业性都有很高要求。先前的工作在单零件CAD生成方面取得了早期成功，但这并不适合实际应用场景，因为现实中的多个零件需要在语义约束和几何兼容性条件下进行装配。在本文中，我们提出了CADKnitter，一个利用几何引导线索来引导扩散采样的组合式CAD生成框架。CADKnitter能够生成一个互补的CAD零件，该零件既遵循给定CAD模型的几何约束，又遵循所需设计文本提示的语义约束。我们还精心构建了一个数据集，

    arXiv:2512.11199v2 Announce Type: replace-cross  Abstract: Computer-aided design (CAD) defines 3D models as compact, precise, and editable representations, making it directly useful for several fields. Recently, CAD generation has been gaining more attention in both the research community and industry. Crafting CAD models has long been a painstaking and time-intensive task, demanding both precision and expertise from designers. Prior works have achieved early success in single-part CAD generation, which is not well-suited for real-world applications, as multiple parts need to be assembled under semantic constraints and geometric compatibility. In this paper, we propose CADKnitter, a compositional CAD generation framework with geometric-guiding cues to steer diffusion sampling. CADKnitter is able to generate a complementary CAD part that follows both the geometric constraints of the given CAD model and the semantic constraints of the desired design text prompt. We also curate a dataset,
    
[^264]: 概率多智能体飞机着陆时间预测

    Probabilistic Multi-Agent Aircraft Landing Time Prediction

    [https://arxiv.org/abs/2512.08281](https://arxiv.org/abs/2512.08281)

    提出一个概率多智能体着陆时间预测框架，将多架飞机的着陆时间以概率分布形式输出，同时兼顾轨迹不确定性及空域中飞机间的交互影响。

    

    准确可靠的飞机着陆时间预测对于空中交通管理中有效的资源配置至关重要。然而，飞机轨迹和交通流的固有不确定性对预测精度和可信度都构成了重大挑战。因此，预测模型不仅应提供飞机着陆时间的点估计，还应提供与这些预测相关的不确定性。此外，飞机轨迹经常通过雷达引导等空中交通管制干预措施受到附近飞机存在的影响。因此，着陆时间预测模型必须考虑空域中的多智能体交互。在这项工作中，我们提出了一个概率多智能体飞机着陆时间预测框架，该框架将多架飞机的着陆时间以分布形式提供。我们使用空中交通监控数据评估了所提出的框架。

    arXiv:2512.08281v2 Announce Type: replace-cross  Abstract: Accurate and reliable aircraft landing time prediction is essential for effective resource allocation in air traffic management. However, the inherent uncertainty of aircraft trajectories and traffic flows poses significant challenges to both prediction accuracy and trustworthiness. Therefore, prediction models should not only provide point estimates of aircraft landing times but also the uncertainties associated with these predictions. Furthermore, aircraft trajectories are frequently influenced by the presence of nearby aircraft through air traffic control interventions such as radar vectoring. Consequently, landing time prediction models must account for multi-agent interactions in the airspace. In this work, we propose a probabilistic multi-agent aircraft landing time prediction framework that provides the landing times of multiple aircraft as distributions. We evaluate the proposed framework using an air traffic surveillan
    
[^265]: 冻结、扩散、解码：面向抗菌肽设计的预训练Transformer嵌入的几何感知适配

    Freeze, Diffuse, Decode: Geometry-Aware Adaptation of Pretrained Transformer Embeddings for Antimicrobial Peptide Design

    [https://arxiv.org/abs/2511.23120](https://arxiv.org/abs/2511.23120)

    提出了FDD（冻结、扩散、解码）框架，通过沿冻结嵌入的内在流形传播监督信号，在保留预训练Transformer嵌入几何结构的前提下实现几何感知的任务适配，并在抗菌肽设计中生成低维、可预测、可解释的表示，支持性质预测、检索与潜空间插值。

    

    预训练Transformer提供了丰富的、通用目的的嵌入表示，可迁移至下游任务。然而，当前的迁移策略——微调和探测——要么会扭曲预训练嵌入的几何结构，要么缺乏足够的表达能力来捕捉与任务相关的信号。当监督数据稀缺时，这些问题会变得更加突出。本文提出了“冻结、扩散、解码”框架，这是一种新颖的基于扩散的框架，能够在保留预训练嵌入底层几何结构的同时，将其适配到下游任务。FDD沿着冻结嵌入的内在流形传播监督信号，实现了对嵌入空间的几何感知适配。将该框架应用于抗菌肽设计，FDD产生了低维、具有预测能力且可解释的表示，可支持性质预测、检索和潜空间插值。

    arXiv:2511.23120v2 Announce Type: replace  Abstract: Pretrained transformers provide rich, general-purpose embeddings, which are transferred to downstream tasks. However, current transfer strategies: fine-tuning and probing, either distort the pretrained geometric structure of the embeddings or lack sufficient expressivity to capture task-relevant signals. These issues become even more pronounced when supervised data are scarce. Here, we introduce Freeze, Diffuse, Decode (FDD), a novel diffusion-based framework that adapts pre-trained embeddings to downstream tasks while preserving their underlying geometric structure. FDD propagates supervised signal along the intrinsic manifold of frozen embeddings, enabling a geometry-aware adaptation of the embedding space. Applied to antimicrobial peptide design, FDD yields low-dimensional, predictive, and interpretable representations that support property prediction, retrieval, and latent-space interpolation.
    
[^266]: 通过对应关系引导实现3D一致的多视图编辑

    3D-Consistent Multi-View Editing by Correspondence Guidance

    [https://arxiv.org/abs/2511.22228](https://arxiv.org/abs/2511.22228)

    提出了一种无需训练的引导框架，通过引入一致性损失确保对应点在编辑后保持相似，从而在去噪过程中实现几何和光度上3D一致的多视图图像编辑。

    

    扩散模型和流模型的最新进展极大地提升了基于文本的图像编辑效果，然而独立编辑各图像的方法往往会在同一场景的不同视图之间产生几何和光度上不一致的结果。这种不一致性对于编辑NeRF或高斯泼溅模型等3D表示而言尤为成问题。我们提出了一种无需训练的引导框架，可在图像编辑过程中强制实现多视图一致性。其核心思想是：对应的点在编辑后应当看起来相似。为实现这一目标，我们引入了一种一致性损失，引导去噪过程朝着连贯一致的编辑方向进行。该框架灵活多变，可以与各种不同的图像编辑方法相结合，同时支持密集和稀疏的多视图编辑设置。实验结果表明，与现有多视图编辑方法相比，我们的方法显著提升了3D一致性。

    arXiv:2511.22228v3 Announce Type: replace-cross  Abstract: Recent advancements in diffusion and flow models have greatly improved text-based image editing, yet methods that edit images independently often produce geometrically and photometrically inconsistent results across different views of the same scene. Such inconsistencies are particularly problematic for editing of 3D representations such as NeRFs or Gaussian splat models. We propose a training-free guidance framework that enforces multi-view consistency during the image editing process. The key idea is that corresponding points should look similar after editing. To achieve this, we introduce a consistency loss that guides the denoising process toward coherent edits. The framework is flexible and can be combined with widely varying image editing methods, supporting both dense and sparse multi-view editing setups. Experimental results show that our approach significantly improves 3D consistency compared to existing multi-view edi
    
[^267]: 迭代GRPO：借助单轮RLHF实现多轮强化学习的批在线策略迭代

    Iterative GRPO: Batch-Online Policy Iteration for Multi-Turn RL via Single-Turn RLHF

    [https://arxiv.org/abs/2511.21638](https://arxiv.org/abs/2511.21638)

    提出Iterative GRPO方法，利用批在线部署模式收集的交互数据，将经典的近似策略迭代算法与单轮RLHF相结合，从而无需在训练循环中接入真实或模拟用户即可实现多轮强化学习。

    

    arXiv:2511.21638v2 公告类型：替换。摘要：实际的LLM智能体通常运行在多轮对话场景中，其成功与否只有在整个交互过程结束后才能确定。大多数多轮RL方法通过同策略 rollout 进行训练，但与单轮RLHF不同，策略无法独自产生完整轨迹，因为在智能体的每一轮之后，外部环境必须做出响应。对于对话式智能体而言，这个环境就是用户，但真实用户通常无法嵌入训练循环中，而忠实地构建模拟用户也十分困难。另一方面，现实世界的部署很少是完全在线或完全离线的。常见的生产模式被称为“批在线”，即先部署当前策略收集一批交互数据，然后在该批数据上重新训练并重新部署。我们证明，这种批在线设置为应用经典的近似策略迭代算法创造了机会。我们的核心观察是，运行标准的token级…

    arXiv:2511.21638v2 Announce Type: replace  Abstract: Practical LLM agents often operate over multi-turn conversations where success is determined only after the full interaction ends. Most multi-turn RL methods train via on-policy rollouts, but unlike in single-turn RLHF, the policy cannot produce a trajectory alone, since an external environment must respond after each agent turn. For conversational agents, this environment is a user, but real users are generally unavailable inside the training loop and simulated users are difficult to build faithfully. Separately, real-world deployment is rarely fully online or fully offline. The common production pattern is called "batch online," where the current policy is deployed to collect a batch of interaction data, then retrained on that batch and redeployed. We show that this batch-online setting creates an opportunity for applying the classical approximate policy iteration algorithm. Our central observation is that running standard token-le
    
[^268]: 神经簇的Alexander-Hirschowitz定理

    The Alexander-Hirschowitz theorem for neurovarieties

    [https://arxiv.org/abs/2511.19703](https://arxiv.org/abs/2511.19703)

    本文通过独立的几何方法证明了激活次数满足线性界 d_i ≥ 2n_i−1 时多项式神经网络的神经簇对任意输出数都是非亏缺的，并进一步证明了多输出架构在相同次数界下的全局可辨识性。

    

    我们研究与多项式神经网络相关的神经簇的维数与可辨识性。我们给出了一个独立的几何证明，表明激活次数上的线性界 $d_i\geq 2n_i-1$ 蕴含任意输出数目下的非亏缺性，这一维数结论此前是从有限可辨识性推导得到的。该证明基于对参数化微分的直接分析。我们还研究了该范围之外的割线与Grassmann割线障碍，并在相同的次数界下证明了多输出架构的全局可辨识性。

    arXiv:2511.19703v2 Announce Type: replace-cross  Abstract: We study the dimension and identifiability of neurovarieties associated to polynomial neural networks. We give an independent geometric proof that the linear bounds $d_i\geq 2n_i-1$ on the activation degrees imply non defectiveness for any number of outputs, a dimension statement previously obtained from finite identifiability. The proof is based on a direct analysis of the differential of the parameterization. We also investigate secant and Grassmann-secant obstructions outside this range and prove global identifiability for multi-output architectures under the same degree bounds.
    
[^269]: SEBA：面向视觉强化学习的样本高效黑盒攻击

    SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning

    [https://arxiv.org/abs/2511.09681](https://arxiv.org/abs/2511.09681)

    SEBA提出了一种针对视觉强化学习的样本高效黑盒攻击框架，通过结合影子Q模型、生成对抗网络和世界模型，以极少的真实环境查询实现对基于图像的连续控制智能体的有效对抗攻击。

    

    视觉强化学习在视觉控制和机器人领域取得了显著进展，但其对对抗性扰动的脆弱性仍未得到充分探索。现有的大多数黑盒攻击集中于基于向量或离散动作的强化学习，其在基于图像的连续控制上的有效性受限于庞大的动作空间和过多的环境查询。我们提出了SEBA，一个针对视觉强化学习智能体的样本高效黑盒对抗攻击框架。SEBA集成了三个组件：一个用于估计对抗条件下累积奖励的影子Q模型、一个生成视觉上不可察觉扰动的生成对抗网络，以及一个模拟环境动态以减少真实环境查询的世界模型。通过在学习影子模型与优化生成器之间交替进行的两阶段迭代训练过程，SEBA在保持高效的同时实现了强大的攻击性能。

    arXiv:2511.09681v2 Announce Type: replace-cross  Abstract: Visual reinforcement learning has achieved remarkable progress in visual control and robotics, but its vulnerability to adversarial perturbations remains underexplored. Most existing black-box attacks focus on vector-based or discrete-action RL, and their effectiveness on image-based continuous control is limited by the large action space and excessive environment queries. We propose SEBA, a sample-efficient framework for black-box adversarial attacks on visual RL agents. SEBA integrates a shadow Q model that estimates cumulative rewards under adversarial conditions, a generative adversarial network that produces visually imperceptible perturbations, and a world model that simulates environment dynamics to reduce real-world queries. Through a two-stage iterative training procedure that alternates between learning the shadow model and refining the generator, SEBA achieves strong attack performance while maintaining efficiency. E
    
[^270]: 基于秩-2子空间解缠的多步知识交互分析

    Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement

    [https://arxiv.org/abs/2511.01706](https://arxiv.org/abs/2511.01706)

    该论文提出一种新颖的秩-2投影子空间来更准确地解缠大语言模型中参数化知识与情境知识的贡献，并首次实现了对自然语言解释更长生成序列中知识交互的多步分析。

    

    自然语言解释（NLEs）通过借助外部情境知识（CK）和参数化知识（PK）来描述大语言模型（LLMs）如何做出决策。理解这些知识来源之间的交互是评估NLE接地性的关键，然而这些动态机制仍未得到充分探索。先前的工作主要集中于：i）单步生成，以及ii）将PK与CK的交互建模为秩-1子空间内的二元选择。这种方法忽略了更丰富的交互形式，以及这些交互在更长生成过程中的演变方式，例如互补性或支持性知识。我们提出了一种新颖的秩-2投影子空间，能够更准确地解缠PK和CK的贡献，并首次将其用于对更长NLE序列中知识交互的多步分析。在四个问答数据集和三个开源权重LLM上的实验表明，秩-1子空间难以表示多样化的知识交互，而我们的秩-2方法能够更好地捕捉这些丰富的交互形式。

    arXiv:2511.01706v3 Announce Type: replace-cross  Abstract: Natural Language Explanations (NLEs) describe how Large Language Models (LLMs) make decisions by drawing on external Context Knowledge (CK) and Parametric Knowledge (PK). Understanding the interaction between these sources is key to assessing NLE grounding, yet these dynamics remain underexplored. Prior work has largely focused on i) single-step generation and ii) modeled PK--CK interaction as a binary choice within a rank-1 subspace. This approach overlooks richer interactions and how they unfold over longer generations, such as complementary or supportive knowledge. We propose a novel rank-2 projection subspace that disentangles PK and CK contributions more accurately and use it for the first multi-step analysis of knowledge interactions across longer NLE sequences. Experiments across four QA datasets and three open-weight LLMs demonstrate that rank-1 subspaces struggle to represent diverse interactions, whereas our rank-2 fo
    
[^271]: 机器能高效地思考吗？

    Can machines think efficiently?

    [https://arxiv.org/abs/2510.26954](https://arxiv.org/abs/2510.26954)

    该论文提出在图灵测试中引入能量消耗约束，从效率视角重新评估机器智能，从而为智能评估提供了原测试所缺乏的可测量的实际标准。

    

    图灵测试已不再足以区分人类智能与机器智能。随着先进的人工智能系统已经通过原始的图灵测试，并引发了严重的伦理和环境问题，我们迫切需要更新这一测试。这项工作在原始的模仿游戏基础上进行了扩展，纳入了一个额外因素：回答问题所消耗的能量。通过增加能量约束，新测试迫使我们从效率的角度来评估智能，将思考这一抽象问题与有限资源的具体现实联系起来。此外，这一新提出的测试确保智能评估拥有一个可测量、可实际操作的终点线，而这是原始测试所缺乏的。这一额外约束促使社会权衡使用人工智能所节省的时间与其总体资源成本。

    arXiv:2510.26954v3 Announce Type: replace-cross  Abstract: The Turing Test is no longer adequate for distinguishing human and machine intelligence. With advanced artificial intelligence systems already passing the original Turing Test and contributing to serious ethical and environmental concerns, we urgently need to update the test. This work expands upon the original imitation game by accounting for an additional factor: the energy spent answering the questions. By adding the constraint of energy, the new test forces us to evaluate intelligence through the lens of efficiency, connecting the abstract problem of thinking to the concrete reality of finite resources. Further, this proposed new test ensures the evaluation of intelligence has a measurable, practical finish line that the original test lacks. This additional constraint compels society to weigh the time savings of using artificial intelligence against its total resource cost.
    
[^272]: 具有可调泄漏ReLU的浅层神经网络优化景观中的非线性动力学

    Nonlinear Dynamics In Optimization Landscape of Shallow Neural Networks with Tunable Leaky ReLU

    [https://arxiv.org/abs/2510.25060](https://arxiv.org/abs/2510.25060)

    本文基于等变梯度度为采用可调泄漏ReLU的浅层神经网络建立了分岔分析理论框架，揭示了临界点从全局最小值处的分岔与神经元数量无关，且在工程区间 (0,1) 内全局最小值保持稳定、不发生对称性破缺。

    

    在本工作中，我们研究了使用均方损失和泄漏ReLU激活函数训练的浅层神经网络的非线性动力学。在高斯输入和相等层宽 k 的条件下，(1) 我们基于等变梯度度建立了一个理论框架，适用于任意数量的神经元 k>=4，用于检测当泄漏参数 $\alpha$ 变化时，具有相关对称性的临界点从全局最小值处产生的分岔。我们的分析通常揭示了一种多模式简并始终发生在临界数 0 处，且与 k 无关。(2) 作为副产品，我们进一步证明了此类分岔与宽度无关，仅在 $\alpha$ 为非负值时出现，并且在工程区间 $\alpha \in (0,1)$ 内，全局最小值不会经历进一步的对称性破缺不稳定性。文中给出了一个 k=5 的显式例子来说明该框架，并展示由此产生的分岔及其……

    arXiv:2510.25060v2 Announce Type: replace-cross  Abstract: In this work, we study the nonlinear dynamics of a shallow neural network trained with mean-squared loss and leaky ReLU activation. Under Gaussian inputs and equal layer width k, (1) we establish, based on the equivariant gradient degree, a theoretical framework, applicable to any number of neurons k>= 4, to detect bifurcation of critical points with associated symmetries from global minimum as leaky parameter $\alpha$ varies. Typically, our analysis reveals that a multi-mode degeneracy consistently occurs at the critical number 0, independent of k. (2) As a by-product, we further show that such bifurcations are width-independent, arise only for nonnegative $\alpha$ and that the global minimum undergoes no further symmetry-breaking instability throughout the engineering regime $\alpha$ in range (0,1). An explicit example with k=5 is presented to illustrate the framework and exhibit the resulting bifurcation together with their 
    
[^273]: 将组合式机器设计视为基于大语言模型的程序合成

    Compositional Machine Design as Program Synthesis with LLMs

    [https://arxiv.org/abs/2510.14980](https://arxiv.org/abs/2510.14980)

    该论文提出将机器设计视为一种以物理模拟验证为依据的程序合成新任务——组合式机器设计，并构建了基于游戏《Besiege》的测试平台BesiegeField，用于评测大语言模型在多种工作流下组合标准部件设计机器的能力。

    

    大语言模型（LLM）在编写和修改程序方面已展现出强大的能力，然而许多程序合成基准仍在符号或数字环境中评估程序。我们提出了“组合式机器设计”，这是一种以物理为基础的程序合成形式：机器被编写为组合标准化部件的程序，其成败由模拟的物理行为决定。为研究这一问题，我们提出了BesiegeField，一个基于机器建造游戏《Besiege》构建的测试平台。在BesiegeField中，LLM智能体根据文本形式的功能需求生成机器程序，在模拟中运行所得的机器，并接收奖励与状态反馈。我们在单智能体生成、迭代编辑和分层工作流等模式下，对LLM智能体在代表性机器设计任务上进行了基准评测。强大的模型能够恢复与任务相关的结构，有时还能取得不俗的物理性能表现，但常常……（原文摘要在此处截断）

    arXiv:2510.14980v3 Announce Type: replace  Abstract: Large language models (LLMs) have shown strong abilities in writing and revising programs, yet many program-synthesis benchmarks still evaluate programs in symbolic or digital environments. We introduce compositional machine design, a physically grounded form of program synthesis where machines are written as programs that compose standardized parts, and success is determined by simulated physical behavior. To study this problem, we present BesiegeField, a testbed built on the machine-building game Besiege. In BesiegeField, LLM agents generate machine programs from textual functional demands, execute the resulting machines in simulation, and receive rewards and state feedback. We benchmark LLM agents across representative machine-design tasks under single-agent generation, iterative editing, and hierarchical workflows. Strong models recover task-relevant structures and sometimes achieve nontrivial physical performance, but often stru
    
[^274]: 沉默是金：通过层加权向量引导缓解大型音频-语言模型的幻觉

    Silence is Golden: Mitigating Hallucinations in Large Audio-Language Models via Layer-Weighted Vector Steering

    [https://arxiv.org/abs/2510.12851](https://arxiv.org/abs/2510.12851)

    本文首次将向量引导技术应用于音频领域，提出无需训练的层加权向量引导方法（LWVS），利用沉默基线对比与关键层强化，在显著缓解大型音频-语言模型幻觉的同时保持甚至提升通用音频理解能力。

    

    大型音频-语言模型（LALMs）在音频问答任务中表现出色，但常常产生缺乏音频依据的幻觉。据我们所知，我们是首个提出将向量引导技术应用于音频领域来缓解这一问题的研究。与基于文本的引导不同，我们的沉默锚定对比方法通过将活跃音频与无声基线进行对比，引导模型远离幻觉。对模型内部状态的探测揭示了特定层的表示与输出正确性之间存在强相关性。利用这一发现，我们提出了层加权向量引导（LWVS），这是一种无需训练的干预方法，可在有影响力的层上增加引导强度。在音频幻觉问答数据集上，LWVS显著优于基线方法，将Gemma模型的召回率提升了15.6%（从53.4%提升至69.0%）。至关重要的是，MMAU基准测试证实LWVS在缓解幻觉的同时保持甚至增强了模型的通用音频理解能力，实现了8%的相对准确率提升。

    arXiv:2510.12851v2 Announce Type: replace-cross  Abstract: Large Audio-Language Models (LALMs) excel in Audio QA but often suffer from hallucinations ungrounded in the audio. To our knowledge, we are the first to propose applying vector steering to the audio domain to mitigate this. Unlike text-based steering, our silence-anchored contrastive approach steers the model away from hallucinations by contrasting active audio against a silent baseline. Probing internal states reveals a strong correlation between specific layer representations and output correctness. Leveraging this, we introduce Layer-Weighted Vector Steering (LWVS), a training-free intervention that increases steering strength at influential layers. On the Audio Hallucination QA dataset, LWVS significantly outperforms baselines, boosting Recall on the Gemma model by 15.6% (53.4% to 69.0%). Crucially, MMAU benchmark tests confirm LWVS preserves and even enhances general audio understanding, achieving an 8% relative accuracy 
    
[^275]: 公平最小标注：面向可达性与公平性的高效时序网络激活

    Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity

    [https://arxiv.org/abs/2510.03899](https://arxiv.org/abs/2510.03899)

    该论文提出了公平最小标注（FML）问题——设计最小成本的时序边激活方案以保障网络中各节点组的公平可达性，证明该问题是NP难且难以近似求解的，并给出了单终端情形等价于有根覆盖斯坦纳问题的结构性刻画。

    

    在支持现代学习应用的网络化系统中，平衡资源效率与公平性至关重要。我们提出了“公平最小标注”问题：即设计一种最小成本的时序边激活方案，确保网络中每组节点都能根据指定的覆盖要求充分访问指定的目标集合。FML 刻画了那些边激活会产生资源成本且公平访问至关重要的系统中的关键权衡，例如分布式数据收集、边云系统中的更新传播，以及关键基础设施中的公平服务恢复。我们首先给出了单终端情形的结构性刻画，证明其等价于有根覆盖斯坦纳问题。我们证明 FML 是 NP 难的，并且即使在星形网络上，对于 |C| 个组，也不存在 ((1-ε)ln|C|)-近似算法，而对于任意 f……

    arXiv:2510.03899v3 Announce Type: replace-cross  Abstract: Balancing resource efficiency and fairness is critical in networked systems that support modern learning applications. We introduce the \emph{Fair Minimum Labeling} (FML) problem: the task of designing a minimum-cost temporal edge activation plan that ensures each group of nodes in a network has sufficient access to a designated target set, according to specified coverage requirements. FML captures key trade-offs in systems where edge activations incur resource costs and equitable access is essential, such as distributed data collection, update dissemination in edge-cloud systems, and fair service restoration in critical infrastructure. We first give a structural characterisation of the single-terminal case, showing that it is equivalent to the rooted Covering Steiner problem. We prove that FML is NP-hard and admits no $((1-\epsilon)\ln |\mathcal{C}|)$-approximation for $|\mathcal{C}|$ groups, already on a star, while for any f
    
[^276]: Transformer中性能与效率的权衡：基于逼近理论的视角

    Performance-Efficiency Tradeoffs in Transformers: An Approximation Theory Perspective

    [https://arxiv.org/abs/2510.03784](https://arxiv.org/abs/2510.03784)

    本文从逼近理论视角刻画了Transformer中注意力头数量与头维度在固定参数预算下的权衡，发现并证明了softmax激活的饱和行为，表明较深的层可以用更小的头维度实现高效运行。

    

    Transformer在各类应用中取得了显著的成功，但其模型效率的理论基础仍未得到充分探索。在这项工作中，我们研究了模型参数——主要是注意力头数量和头的维度——应如何在不同层之间分配，以平衡表达能力与效率。我们首先从逼近理论的角度对早期层在信息提取中的作用进行了数学分析，并在固定参数预算下对注意力头数量与头维度之间的权衡进行了理论刻画。此外，我们发现并证明了softmax激活的饱和行为：持续增加头维度可能导致学习误差的收益递减，特别是在长序列情况下。在理论和实验的双重支持下，这种饱和模式表明后面的层可以通过减少头维度以更高效的方式运行。

    arXiv:2510.03784v2 Announce Type: replace  Abstract: Transformers have achieved remarkable successes across a wide range of applications, yet the theoretical foundation of their model efficiency remains underexplored. In this work, we investigate how the model parameters -- mainly attention heads and head dimensions -- should be allocated across layers to balance expressivity and efficiency. We first provide mathematical analysis on the role of early layers in information extraction from an approximation perspective, with a theoretical characterization on the trade-off between the number of heads and head dimension under a fixed parameter budget. In addition, we uncover and prove the \emph{saturation} behavior of softmax activations: Continuously increasing head dimensions can lead to diminishing returns in learning errors, particularly for long sequences. Supported by both theory and experiments, this saturation pattern suggests that later layers can operate more efficiently with redu
    
[^277]: 优势加权匹配：在扩散模型中实现强化学习与预训练的对齐

    Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models

    [https://arxiv.org/abs/2509.25050](https://arxiv.org/abs/2509.25050)

    本文从理论上揭示DDPO本质是带噪声目标的隐式分数/流匹配，并提出优势加权匹配（AWM），通过以优势加权分数/流匹配损失，使扩散模型的强化学习与预训练目标对齐，从而降低方差并加速收敛。

    

    强化学习（RL）已成为推进大语言模型（LLM）发展的核心范式，其预训练与强化学习后训练阶段均基于相同的对数似然公式。相比之下，近期针对扩散模型的强化学习方法——尤其是去噪扩散策略优化（DDPO）——所优化的目标与预训练目标（即分数/流匹配损失）并不相同。在本工作中，我们建立了一种全新的理论分析：DDPO本质上是一种带有噪声目标的隐式分数/流匹配，这会增大方差并减慢收敛速度。基于这一分析，我们提出了优势加权匹配（AWM），这是一种面向扩散模型的策略梯度方法。它采用分数/流匹配损失，并根据每个样本的优势值对其进行重新加权。实际上，AWM在保持建模目标与预训练完全一致的同时，提升了高奖励样本的影响力并抑制了低奖励样本。

    arXiv:2509.25050v2 Announce Type: replace  Abstract: Reinforcement Learning (RL) has emerged as a central paradigm for advancing Large Language Models (LLMs), where both pre-training and RL post-training stages are grounded in the same log-likelihood formulation. In contrast, recent RL approaches for diffusion models, most notably Denoising Diffusion Policy Optimization (DDPO), optimize an objective different from the pretraining objectives--score/flow matching loss. In this work, we establish a novel theoretical analysis: DDPO is an implicit form of score/flow matching with noisy targets, which increases variance and slows convergence. Building on this analysis, we introduce Advantage Weighted Matching (AWM), a policy-gradient method for diffusion. It uses the score/flow-matching loss and reweights each sample by its advantage. In effect, AWM raises the influence of high-reward samples and suppresses low-reward ones while keeping the modeling objective identical to pretraining. This s
    
[^278]: 一种用于特征学习的组合核模型

    A Compositional Kernel Model for Feature Learning

    [https://arxiv.org/abs/2509.14158](https://arxiv.org/abs/2509.14158)

    本文提出一种组合核岭回归模型，证明其能在变量选择中恢复相关变量并消除高斯噪声变量，且核心发现是ℓ₁型核（如拉普拉斯核）在驻点处能恢复非线性特征，而高斯核仅能恢复线性特征。

    

    我们研究了一种核岭回归的组合变体，其中预测器作用于输入的逐坐标重加权。该模型被表述为一个变分问题，为研究组合架构中的特征学习提供了一个易于处理的框架。从变量选择的角度出发，我们展示了如何恢复相关变量并消除噪声变量。我们证明了当噪声变量服从高斯分布时，全局最小值点和驻点都会舍弃噪声坐标。一个核心发现是，$\ell_1$型核（如拉普拉斯核）能够在驻点处成功恢复对非线性效应有贡献的特征，而高斯核只能恢复线性特征。

    arXiv:2509.14158v3 Announce Type: replace  Abstract: We study a compositional variant of kernel ridge regression in which the predictor is applied to a coordinate-wise reweighting of the inputs. Formulated as a variational problem, this model provides a tractable setting for studying feature learning in compositional architectures. From the perspective of variable selection, we show how relevant variables are recovered while noise variables are eliminated. We prove that both global minimizers and stationary points discard noise coordinates when the noise variables are Gaussian distributed. A central finding is that $\ell_1$-type kernels, such as the Laplace kernel, succeed in recovering features contributing to nonlinear effects at stationary points, whereas Gaussian kernels recover only linear ones.
    
[^279]: 用于高效神经组合优化的循环状态编码器

    Recurrent State Encoders for Efficient Neural Combinatorial Optimization

    [https://arxiv.org/abs/2509.05084](https://arxiv.org/abs/2509.05084)

    提出一种循环状态编码器，通过复用先前步骤的状态嵌入，使神经组合优化模型在层数减少3倍的情况下仍能达到相当或更优的性能，显著降低推理延迟。

    

    神经组合优化的主要范式是构造方法，即训练神经网络逐步添加解的组件，直到形成完整的解。我们观察到，相邻两个步骤之间状态的变化通常很小，因为通常只是被添加到解中的节点从状态中被移除。一个高效的模型应该能够复用先前步骤的计算。为此，我们提出了一种循环编码器，它不仅基于当前状态计算状态嵌入，还结合先前状态的嵌入进行计算。我们证明，这种循环编码器即使使用少3倍的层数，也能达到与非循环编码器相当或更好的性能，从而显著改善了推理延迟。我们在三个不同的问题上验证了我们的发现：旅行商问题（TSP）、带容量约束的车辆路径问题（CVRP）以及定向越野问题。

    arXiv:2509.05084v2 Announce Type: replace  Abstract: The primary paradigm in Neural Combinatorial Optimization (NCO) consists of construction methods, where a neural network is trained to sequentially add one solution component at a time until a complete solution is formed. We observe that the typical changes to the state between two steps are small, since usually only the node added to the solution is removed from the state. An efficient model should be able to reuse computation from prior steps. To that end, we propose a recurrent encoder that computes state embeddings based not only on the current state but also on embeddings from the previous state. We show that this recurrent encoder can achieve equivalent or better performance than a non-recurrent encoder even with $3\times$ fewer layers, thus significantly improving latency. We demonstrate our findings on three different problems: the Traveling Salesman Problem (TSP), the Capacitated Vehicle Routing Problem (CVRP), and the Orien
    
[^280]: 通过统一强化学习框架实现城市空中交通（UAM）中的噪声与安全一体化管理

    Integrated Noise and Safety Management in UAM via A Unified Reinforcement Learning Framework

    [https://arxiv.org/abs/2508.16440](https://arxiv.org/abs/2508.16440)

    该论文提出一个统一去中心化的强化学习空中交通管理框架，使城市空中交通飞行器在多层空域中通过学习高度调整策略同时优化噪声暴露与安全间隔，并揭示了高密度交通下安全、噪声与能耗之间的权衡关系。

    

    城市空中交通（UAM）设想通过小型飞行器的广泛使用来变革密集城市环境中的交通运输方式。然而，UAM面临关键的运行挑战，尤其是在低空城市空域中如何平衡噪声暴露最小化与保持安全间隔这两个潜在冲突的目标，而这两个目标通常被分开处理。我们提出了一种基于强化学习（RL）的空中交通管理系统，在统一的去中心化框架内整合了噪声与安全两方面的考量。在这一可扩展的空中交通协调方案下，智能体在结构化的多层空域中运行，并学习高度调整策略，以共同管理噪声影响和间隔约束。该系统在两个目标上均展现出强劲的性能，并揭示了高交通密度下安全间隔、噪声暴露与能源效率之间的权衡。

    arXiv:2508.16440v3 Announce Type: replace-cross  Abstract: Urban Air Mobility (UAM) envisions the widespread use of small aerial vehicles to transform transportation in dense urban environments. However, UAM faces critical operational challenges, particularly the balance between minimizing noise exposure and maintaining safe separation in low-altitude urban airspace, two potentially conflicting objectives that are often addressed separately. We propose a reinforcement learning (RL)-based air traffic management system that integrates both noise and safety considerations within a unified, decentralized framework. Under this scalable air traffic coordination solution, agents operate in a structured, multi-layered airspace and learn altitude adjustment policies to jointly manage noise impact and separation constraints. The system demonstrates strong performance across both objectives and reveals tradeoffs among separation, noise exposure, and energy efficiency under high traffic density. A
    
[^281]: SupraTok：跨边界分词技术助力语言模型性能提升

    SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance

    [https://arxiv.org/abs/2508.11857](https://arxiv.org/abs/2508.11857)

    SupraTok是一种跨越空白符边界的创新分词器，通过熵筛选、PMI引导的课程训练和多语言处理三大模块，在压缩率上比标准BPE提升17.5%，并比SuperBPE训练快2.1倍。

    

    分词一直是语言建模中持续存在的瓶颈，尤其是当词汇学习受到空白符边界的限制时。我们提出了SupraTok，这是一种能够跨越空白符边界的分词器，它由三个模块化组件构成：可选的基于熵的数据筛选、采用PMI引导候选搜索的分阶段课程训练，以及多语言文字处理。在相同的未过滤训练数据上，使用10万词汇量时，SupraTok的压缩效果比标准BPE提升17.5%，比官方SuperBPE实现提升1.8%，同时训练速度比SuperBPE快2.1倍。在相同匹配设置下，在5万至30万词汇量范围内，SupraTok始终比SuperBPE领先1.8%至8.6%。我们还将熵过滤作为流水线中的一个独立步骤单独评估：在10万词汇量下，它使SupraTok的C/T指标从5.78提升至5.99，而匹配对照组显示SuperBPE的增益较小，SP-BPE-CrossBoundary则几乎无变化。在FLORES-200的14种语言上的（摘要原文在此处截断）

    arXiv:2508.11857v3 Announce Type: replace-cross  Abstract: Tokenization remains a persistent bottleneck in language modeling, especially when vocabulary learning is limited by whitespace boundaries. We present SupraTok, a tokenizer that crosses whitespace boundaries using three modular components: optional entropy-based data curation, staged curriculum training with PMI-guided candidate search, and multilingual script handling. At 100k vocabulary on the same unfiltered training data, SupraTok improves compression over standard BPE by 17.5% and over the official SuperBPE implementation by 1.8%, while training 2.1x faster than SuperBPE. Across 50k-300k vocabularies in the same matched setting, SupraTok remains ahead of SuperBPE by 1.8%-8.6%. We evaluate entropy filtering separately as a pipeline step: at 100k vocabulary it raises SupraTok from 5.78 to 5.99 C/T, while matched controls show a smaller gain for SuperBPE and almost no change for SP-BPE-CrossBoundary. On FLORES-200 across 14 l
    
[^282]: FlexP-SFT：一种面向大语言模型端侧个性化分割联邦微调的灵活无聚合框架

    FlexP-SFT: A Flexible Aggregation-Free Framework for On-Device Personalized Split Federated Fine-Tuning of LLMs

    [https://arxiv.org/abs/2508.10349](https://arxiv.org/abs/2508.10349)

    FlexP-SFT是一种面向大语言模型的无需聚合的个性化分割联邦微调框架，通过消除客户端聚合过程解决通信瓶颈和掉队者问题，并借助层级灵活对齐策略在无全局同步的情况下平衡个性化与泛化能力。

    

    为了在私有数据上微调大型语言模型（LLM），联邦学习（FL）已成为一种有前景的范式。然而，LLM高昂的内存和通信需求使得标准联邦学习在资源受限的边缘设备上难以实际应用。虽然分割联邦学习（SFL）通过模型分区缓解了计算负担，但由于存在参数聚合过程，现有框架仍然面临通信瓶颈和掉队者问题。为解决这些挑战，我们提出了FlexP-SFT，这是一种用于个性化分割联邦微调的新型无聚合框架，它从根本上消除了客户端聚合过程。至关重要的是，为了在缺乏全局同步的情况下确保稳健训练，我们引入了一种层级灵活对齐策略来平衡个性化与泛化能力。我们进一步将分割比选择表述为资源感知的离散优化问题（摘要原文在此处截断）。

    arXiv:2508.10349v2 Announce Type: replace-cross  Abstract: To fine-tune large language models (LLMs) over private data, federated learning (FL) has emerged as a promising paradigm. However, the prohibitive memory and communication demands of LLMs render standard FL impractical for resource-constrained edge devices. While split federated learning (SFL) alleviates the computing burdens via model partitioning, existing frameworks still suffer from communication bottlenecks and straggler problem due to the parameter aggregation process. To address these challenges, we propose FlexP-SFT, a novel aggregation-free framework for personalized split federated fine-tuning, which fundamentally eliminates the client-side aggregation process. Crucially, to ensure robust training in the absence of global synchronization, we introduce a layer-flexible alignment strategy to balance personalization and generalization capabilities. We further formulate split-ratio selection as a resource-aware discrete o
    
[^283]: BiasGym：一个通过注入来分析和消除偏见的简单且可泛化的框架

    BiasGym: A Simple and Generalizable Framework for Analyzing and Removing Biases through Injection

    [https://arxiv.org/abs/2508.08855](https://arxiv.org/abs/2508.08855)

    提出BiasGym框架，通过在冻结的LLM中注入特定偏见信号，再利用这些信号定位并抑制或引导导致偏见行为的模型组件，实现偏见的可靠分析与消除。

    

    理解大型语言模型（LLM）权重中编码的偏见和刻板印象，对于制定有效的缓解策略至关重要。然而，偏见行为往往是微妙且难以隔离的，即使刻意去引发也是如此，这使得系统性的分析和去偏工作尤其具有挑战性。为了解决这一问题，我们提出了一个简单、低成本且可泛化的框架 BiasGym，用于可靠地注入、分析和缓解 LLM 中偏见的概念关联。BiasGym 包含两个模块：Inject，通过基于 token 的微调（同时保持模型冻结）向模型注入特定偏见；以及两种去偏方法，利用这些注入信号来识别并可靠地抑制（Scope）或引导（Steer）导致偏见行为的模型组件。我们的框架能够实现一致的偏见引发，从而更好地定位…

    arXiv:2508.08855v5 Announce Type: replace-cross  Abstract: Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. However, biased behavior is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce a simple, cost-effective, and generalizable framework \texttt{BiasGym} for reliably injecting, analyzing, and mitigating conceptual associations of biases within LLMs. \texttt{BiasGym} consists of two modules: \texttt{Inject}, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, followed by two debiasing methods that leverage these injected signals to identify and reliably suppress (\texttt{Scope}) or \texttt{Steer} the components responsible for biased behavior. Our framework enables consistent bias elicitation for better localizati
    
[^284]: 无监督伙伴设计实现鲁棒的临时团队协作

    Unsupervised Partner Design Enables Robust Ad-hoc Teamwork

    [https://arxiv.org/abs/2508.06336](https://arxiv.org/abs/2508.06336)

    提出无监督伙伴设计（UPD），通过即时生成训练伙伴并基于可学习性准则自适应选择，无需预训练伙伴种群或手动调参即可实现鲁棒的临时团队协作，在多个基准任务和人机用户研究中均表现出卓越性能。

    

    我们提出了无监督伙伴设计（UPD），这是一种面向鲁棒临时团队协作的无种群多智能体强化学习方法。UPD即时生成训练伙伴，并基于可学习性准则自适应地选择它们，从而无需预训练的伙伴种群或手动参数调整。我们证明这一简单机制能够实现有效的伙伴多样性，并且当程序化关卡生成器可用时，还可以扩展到伙伴与环境的联合选择。在Level-Based Foraging、Overcooked-AI和Overcooked泛化挑战等任务中，与基于种群和无种群的基线方法相比，UPD始终取得了强劲的性能表现。在人机交互用户研究中，使用UPD训练的智能体获得了更高的回报，并且与所有被评估的基线方法相比，被评为更具适应性、更像人类，且令人沮丧程度更低。

    arXiv:2508.06336v3 Announce Type: replace-cross  Abstract: We introduce Unsupervised Partner Design (UPD), a population-free multi-agent reinforcement learning method for robust ad-hoc teamwork. UPD generates training partners on-the-fly and selects them adaptively based on a learnability criterion, removing the need for pre-trained partner populations or manual parameter tuning. We show that this simple mechanism enables effective partner diversity and can be extended to joint partner-environment selection when a procedural level generator is available. Across Level-Based Foraging, Overcooked-AI, and the Overcooked Generalisation Challenge, UPD consistently achieves strong performance compared to both population-based and population-free baselines. In a human-AI user study, agents trained with UPD achieve higher returns and are rated as more adaptive, more human-like, and less frustrating than all evaluated baseline methods.
    
[^285]: 迈向可证明且可扩展的量化神经网络训练：基于伊辛优化方法

    Towards Provable and Scalable Training of Quantized Neural Networks with Ising Optimization

    [https://arxiv.org/abs/2506.18240](https://arxiv.org/abs/2506.18240)

    该论文提出一个具有可证明保证的精确二次约束二元优化（QCBO）框架，将量化神经网络训练编译为具有零松弛间隙的完全正凸优化问题，并通过逐样本分解下界优化（DLBO）将伊辛求解规模从数据集级别降至单样本级别，从而实现对量化神经网络可证明且可扩展的训练。

    

    由于非凸的损失景观和离散的参数空间，训练量化神经网络仍然面临根本性的挑战。我们引入了一个具有可证明保证的精确二次约束二元优化（QCBO）框架。我们首先刻画了网络零损失水平集的分层拓扑结构：一般的内部层是光滑的，但即使在过参数化条件下，全局最优的连通分量仍可能保持不连通。为了克服这一非凸障碍，我们将具有参数码本和前向区间传播（FIP）有界状态的有限深度架构编译为有界的QCBO，得到一种精确的完全正凸表述，该表述在零松弛间隙下保留了全局离散最优解。为了克服整体样本规模扩展的瓶颈，我们提出了逐样本的分解下界优化（DLBO）方法，将每次伊辛求解调用的规模从整个数据集降低到单样本级别。DLBO矩层级结构……（摘要在此处截断）

    arXiv:2506.18240v5 Announce Type: replace-cross  Abstract: Training quantized neural networks remains fundamentally challenging due to non-convex loss landscapes and discrete parameter spaces. We introduce an exact Quadratic Constrained Binary Optimization (QCBO) framework with provable guarantees. We first characterize the stratified topology of network zero-loss level sets: generic interior strata are smooth, yet globally optimal components can remain disconnected even under overparameterization. To address this non-convex obstruction, we compile finite-depth architectures with parameter codebooks and Forward Interval Propagation (FIP)-bounded states into bounded QCBOs, yielding an exact completely positive convex formulation that preserves the global discrete optimum with zero relaxation gap. To overcome monolithic sample scaling, we formulate sample-wise Decomposed Lower-Bound Optimization (DLBO) to reduce each Ising call from dataset to single-sample scale. The DLBO moment hierarc
    
[^286]: 关于高维线性分类中一致性对抗攻击的存在性

    On the Existence of Consistent Adversarial Attacks in High-Dimensional Linear Classification

    [https://arxiv.org/abs/2506.12454](https://arxiv.org/abs/2506.12454)

    本文提出了一种新的误差度量来区分真正的一致性对抗攻击（即保持真实标签不变的扰动）与因数据有限或模型能力不足导致的普通误分类，并通过精确的渐近理论分析证明，随着模型过参数化程度的提高，其对标签保持扰动的脆弱性会不断增大。

    

    对抗攻击与因模型表达能力有限或数据有限而导致的错误分类，其根本区别究竟是什么？在本工作中，我们在高维二分类的设定下研究这一问题，其中数据有限所带来的统计效应起着核心作用。我们引入了一种新的误差度量，能够精确捕捉这一区别，量化模型对一致性对抗攻击的脆弱性——即那些保持真实标签不变的扰动。我们的主要技术贡献在于对良好指定模型和潜在空间模型中的这些度量给出了精确且严格的渐近刻画，揭示了与标准鲁棒误差度量不同的脆弱性模式。理论结果表明，随着模型变得更加过参数化，其对抗保持标签扰动的脆弱性也随之增长，为理解这一机制提供了理论洞见。

    arXiv:2506.12454v2 Announce Type: replace-cross  Abstract: What fundamentally distinguishes an adversarial attack from a misclassification due to limited model expressivity or finite data? In this work, we investigate this question in the setting of high-dimensional binary classification, where statistical effects due to limited data availability play a central role. We introduce a new error metric that precisely capture this distinction, quantifying model vulnerability to consistent adversarial attacks -- perturbations that preserve the ground-truth labels. Our main technical contribution is an exact and rigorous asymptotic characterization of these metrics in both well-specified models and latent space models, revealing different vulnerability patterns compared to standard robust error measures. The theoretical results demonstrate that as models become more overparameterized, their vulnerability to label-preserving perturbations grows, offering theoretical insight into the mechanisms
    
[^287]: 基于稀疏线性规划的平衡符号图高效学习

    Efficient Learning of Balanced Signed Graphs via Sparse Linear Programming

    [https://arxiv.org/abs/2506.01826](https://arxiv.org/abs/2506.01826)

    提出了一种基于稀疏线性规划的高效方法，可直接从数据中学习平衡符号图的拉普拉斯矩阵，使其能够复用为正图设计的谱滤波工具。

    

    符号图同时具有正边权重和负边权重，用以编码数据中的成对相关性与反相关性。平衡符号图是指不含包含奇数条负边的环路的符号图。平衡符号图的拉普拉斯矩阵的特征向量可以通过一个简单的线性变换映射到对应正图拉普拉斯矩阵的特征向量，从而能够复用为正图设计的谱滤波工具。我们提出了一种高效的计算方法，可以直接从数据中学习平衡符号图的拉普拉斯矩阵。具体而言，我们在先前基于线性规划（LP）的稀疏逆协方差估计方法CLIME的基础上进行扩展，为拉普拉斯矩阵的每一列 $i$ 构建了一个新的LP问题，其中线性约束限制了从节点 $i$ 出发的边的权重符号，使得相同/不同极性的节点分别通过正/负边相连。我们推导了一个可行（摘要内容在此处被截断）

    arXiv:2506.01826v2 Announce Type: replace  Abstract: Signed graphs are equipped with both positive and negative edge weights, encoding pairwise correlations as well as anti-correlations in data. A balanced signed graph is a signed graph with no cycles containing an odd number of negative edges. Laplacian of a balanced signed graph has eigenvectors that map via a simple linear transform to ones in a corresponding positive graph Laplacian, thus enabling reuse of spectral filtering tools designed for positive graphs. We propose an efficient computation method to learn a balanced signed graph Laplacian directly from data. Specifically, extending a previous linear programming (LP) based sparse inverse covariance estimation method called CLIME, we formulate a new LP problem for each Laplacian column $i$, where the linear constraints restrict weight signs of edges stemming from node $i$, so that nodes of same / different polarities are connected by positive / negative edges. We derive a feasi
    
[^288]: 基于平滑随机梯度下降的分位数在线同时推断

    Online simultaneous inference for quantiles via smoothed stochastic gradient descent

    [https://arxiv.org/abs/2505.13299](https://arxiv.org/abs/2505.13299)

    本文提出一种平滑随机梯度下降方法用于流数据的在线分位数估计，其估计量在每次迭代中关于分位数水平单调，并借助一致Bahadur表示与布朗桥最大值的高斯近似，实现了维度随样本量指数增长时跨坐标与分位数水平的在线同时统计推断。

    

    本文考虑通过随机梯度下降（SGD）算法的平滑版本来估计分位数。通过使用与学习率相关联的带宽对得分函数进行平滑，我们得到的估计量在每次迭代中都关于分位数水平保持单调，同时保留了流数据处理所需的内存和计算效率。我们建立了平滑估计量在使用与不使用Polyak-Ruppert平均两种情况下的非渐近尾概率界，这些界是具有多区域结构的亚指数型。对于平均估计量，我们进一步推导出关于分位数水平和各坐标一致成立的Bahadur表示，以及由布朗桥最大值给出的高斯近似，其中维度 $p$ 允许随样本量呈指数级增长。由此实现了跨坐标与分位数水平的同时推断。作为一种避免估计……的替代方法（摘要在此处被截断）

    arXiv:2505.13299v2 Announce Type: replace-cross  Abstract: This paper considers the estimation of quantiles via a smoothed version of the stochastic gradient descent (SGD) algorithm. By smoothing the score function with a bandwidth tied to the learning rate, we obtain estimates that are monotone in the quantile level at every iteration, while retaining the memory and computational efficiency required for streaming data. We establish non-asymptotic tail probability bounds for the smoothed estimate with and without Polyak-Ruppert averaging, which are sub-exponential with a multi-regime structure. For the averaged estimate we further derive a Bahadur representation that is uniform in the quantile level and across coordinates, and a resulting Gaussian approximation by the maximum of Brownian bridges, with the dimension $p$ allowed to grow exponentially in the sample size. This yields simultaneous inference across coordinates and quantile levels. As an alternative that avoids estimating the
    
[^289]: 无需非高斯性假设的多视图因果发现：可辨识性与算法

    Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms

    [https://arxiv.org/abs/2502.20115](https://arxiv.org/abs/2502.20115)

    本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。

    

    因果发现是一个困难的问题，通常依赖于对数据生成模型的强假设，例如非高斯性。在实践中，许多现代应用提供了同一系统的多个相关视图，而这一点在因果发现领域很少被考虑。在此，我们利用这种多视图结构，在弱假设条件下实现因果发现。我们提出了一个多视图线性结构方程模型（SEM），该模型通过交替利用视图间的相关性，扩展了著名的非高斯扰动框架。我们证明了该模型在无环SEM情形下的可辨识性。随后，受单视图算法（DirectLiNGAM、PairwiseLiNGAM和ICA-LiNGAM）的启发，我们提出了几种多视图因果发现算法。新方法通过仿真实验和神经影像数据应用得到了验证，在这些应用中，它们能够估计脑区之间的因果图。

    arXiv:2502.20115v4 Announce Type: replace  Abstract: Causal discovery is a difficult problem that typically relies on strong assumptions on the data-generating model, such as non-Gaussianity. In practice, many modern applications provide multiple related views of the same system, which has rarely been considered for causal discovery. Here, we leverage this multi-view structure to achieve causal discovery with weak assumptions. We propose a multi-view linear Structural Equation Model (SEM) that extends the well-known framework of non-Gaussian disturbances by alternatively leveraging correlation over views. We prove the identifiability of the model for acyclic SEMs. Subsequently, we propose several multi-view causal discovery algorithms, inspired by single-view algorithms (DirectLiNGAM, PairwiseLiNGAM, and ICA-LiNGAM). The new methods are validated through simulations and applications on neuroimaging data, where they enable the estimation of causal graphs between brain regions.
    
[^290]: 通过熵流计算为马尔可夫算法建立泛化界

    Generalization Bounds for Markov Algorithms through Entropy Flow Computations

    [https://arxiv.org/abs/2502.07584](https://arxiv.org/abs/2502.07584)

    该论文提出新的技术工具，将熵流方法的适用范围从特定的噪声和算法结构（如朗之万动力学）扩展到所有迭代动力学由时齐马尔可夫过程支配的学习算法，从而为这一广泛类别的算法建立泛化界。

    

    许多学习算法可以表示为马尔可夫过程，理解它们的泛化误差是学习理论中的核心课题。对于特定的连续时间含噪算法，一种突出的分析技术依赖于信息论工具和所谓的“熵流”方法。该技术与广泛的假设条件兼容，并利用学习动力学的收敛性质来产生有意义的泛化界，这些界也可以具有信息量或扩展到离散时间设置。尽管取得了成功，现有的熵流公式仅限于特定的噪声和算法结构（例如，朗之万动力学）。在这项工作中，我们利用新的技术工具将其适用性扩展到所有迭代动力学由时齐马尔可夫过程支配的学习算法。我们的方法基于对马尔可夫算法的原理性连续时间近似……

    arXiv:2502.07584v3 Announce Type: replace-cross  Abstract: Many learning algorithms can be represented as Markov processes, and understanding their generalization error is a central topic in learning theory. For specific continuous-time noisy algorithms, a prominent analysis technique relies on information-theoretic tools and the so-called ``entropy flow'' method. This technique is compatible with a broad range of assumptions and leverages the convergence properties of learning dynamics to produce meaningful generalization bounds, which can also be informative or extend to discrete-time settings. Despite their success, existing entropy flow formulations are limited to specific noise and algorithm structures (\eg, Langevin dynamics). In this work, we exploit new technical tools to extend its applicability to all learning algorithms whose iterative dynamics is governed by a time-homogeneous Markov process. Our approach builds on a principled continuous-time approximation of Markov algori
    
[^291]: QABBA：通过整数量化聚合实现带误差保证的符号时间序列压缩

    QABBA: Error-Guaranteed Symbolic Time-Series Compression via Integer-Quantized Aggregation

    [https://arxiv.org/abs/2411.15209](https://arxiv.org/abs/2411.15209)

    提出QABBA，通过量化符号中心实现ABBA的整数化压缩，在保证重建质量的同时提供严格的误差界限。

    

    来自传感器和监控系统的时间序列数据的扩张使得紧凑表示变得越来越重要。这种表示应在削减存储、传输和计算成本的同时保留信号结构。自适应布朗桥聚合（ABBA）通过将长数值序列转换为短符号序列来满足这一需求，但参数存储和计算精度的降低仍然值得追求。我们提出了量化ABBA（QABBA），即ABBA的量化版本。通过量化符号中心，QABBA减少了参数占用，并启用整数运算，同时保持高重建质量。我们为量化引入的额外近似建立了多个误差界：每个段超额误差的无维度界、时域重建误差界、符号分配的稳定性条件，以及分配位的规则。

    arXiv:2411.15209v3 Announce Type: replace  Abstract: The expansion of time-series data from sensors and monitoring systems has made compact representations increasingly important. Such representations should retain signal structure while cutting storage, transmission and computation costs. Adaptive Brownian Bridge-based Aggregation (ABBA) addresses this need by converting long numerical series into short symbolic sequences, but reductions in parameter storage and computational precision remain desirable.   We propose Quantized ABBA (QABBA), a quantized version of ABBA. By quantizing the symbolic centers, QABBA reduces the parameter footprint and enables integer arithmetic while maintaining high reconstruction quality. We establish several error bounds for the additional approximation introduced by quantization: a dimension-free bound on the excess error of each segment, a time-domain reconstruction-error bound, a stability condition for symbolic assignment, and a rule for allocating bi
    
[^292]: 让每个人都满意：少量副本下大量物品的在线公平分配

    Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies

    [https://arxiv.org/abs/2408.12845](https://arxiv.org/abs/2408.12845)

    针对物品数量多而副本少的在线公平分配难题，本文创新性地假设效用是物品-智能体特征的未知函数，并将其建模为上下文老虎机问题，从而克服了无法准确估计所有物品-智能体对效用的局限。

    

    本文研究了在线公平分配问题的一种新变体，该问题涉及多个智能体，学习者按顺序观察到不可分割的物品，必须将其不可撤销地分配给其中一个智能体，以在公平性和效率之间实现理想的平衡。现有算法假设物品数量少且副本数量足够大，这保证了能够从带噪声的观测效用中对所有物品-智能体对进行良好的效用估计。然而，这一假设在许多现实应用中可能不成立，例如，一个在线平台拥有大量用户（物品），这些用户仅使用平台的服务提供商（智能体）少数几次（即物品只有少量副本），这使得难以准确估计所有物品-智能体对的效用。为了解决这一局限性，我们假设效用是物品-智能体特征的未知函数，并提出将在线公平分配建模为上下文老虎机问题的算法。

    arXiv:2408.12845v3 Announce Type: replace-cross  Abstract: This paper considers a novel variant of the online fair division problem involving multiple agents in which a learner sequentially observes an indivisible item that must be irrevocably allocated to one of the agents to achieve a desired balance between fairness and efficiency. Existing algorithms assume a small number of items with a sufficiently large number of copies, which ensures a good utility estimation for all item-agent pairs from noisy observed utilities. However, this assumption may not hold in many real-life applications, e.g., an online platform with a large number of users (items) who use the platform's service providers (agents) only a few times (a few copies of items), making it difficult to accurately estimate utilities for all item-agent pairs. To address this limitation, we assume utility is an unknown function of item-agent features. We propose algorithms that model online fair division as a contextual bandit
    
[^293]: FedReview: 一种用于拒绝毒化更新的联邦学习审查机制

    FedReview: A Review Mechanism for Rejecting Poisoned Updates in Federated Learning

    [https://arxiv.org/abs/2402.16934](https://arxiv.org/abs/2402.16934)

    提出了FedReview机制，通过随机分配评审员客户端来识别和拒绝联邦学习中的潜在毒化更新，并采用多数表决机制来整合排名并移除这些更新。

    

    Federated learning最近已经被提出作为一种去中心化的方法，在不访问用户数据的情况下学习一个高性能模型。尽管其有效性，但联邦学习给恶意用户提供了机会通过向服务器上传毒化模型更新来操纵模型。在本文中，我们提出了一种名为FedReview的审查机制，用于识别和拒绝联邦学习中潜在的毒化更新。在我们的机制下，服务器每轮随机分配子集客户端作为评审员，在其训练数据集上评估模型更新。评审员根据评价结果对模型更新进行排名，统计相对低质量的更新数量作为估计的毒化更新数量。基于审查报告，服务器采用多数表决机制整合排名并在模型聚合过程中去除潜在的毒化更新。

    arXiv:2402.16934v1 Announce Type: cross  Abstract: Federated learning has recently emerged as a decentralized approach to learn a high-performance model without access to user data. Despite its effectiveness, federated learning gives malicious users opportunities to manipulate the model by uploading poisoned model updates to the server. In this paper, we propose a review mechanism called FedReview to identify and decline the potential poisoned updates in federated learning. Under our mechanism, the server randomly assigns a subset of clients as reviewers to evaluate the model updates on their training datasets in each round. The reviewers rank the model updates based on the evaluation results and count the number of the updates with relatively low quality as the estimated number of poisoned updates. Based on review reports, the server employs a majority voting mechanism to integrate the rankings and remove the potential poisoned updates in the model aggregation process. Extensive evalu
    
[^294]: 构建富有表现力和可处理的概率生成模型：一项综述

    Building Expressive and Tractable Probabilistic Generative Models: A Review

    [https://arxiv.org/abs/2402.00759](https://arxiv.org/abs/2402.00759)

    本文综述了富有表现力和可处理的概率生成建模领域的进展和技术，并重点关注了概率电路。文章提供了关于表达能力和可处理性之间权衡的统一视角，并说明了设计原则和算法扩展，成功地构建了富有表现力和高效的概率电路。此外，文章还讨论了最新的深度和混合概率电路研究，并概述了未来研究的挑战和开放性问题。

    

    我们对可处理的概率生成建模领域中的进展和技术进行了全面的调查，重点关注概率电路（PCs）。我们提供了关于表达能力和可处理性之间固有权衡的统一视角，突出了使PCs富有表现力和高效的设计原则和算法扩展，并提供了该领域的分类法。我们还讨论了最近通过融合深度神经模型概念来构建深度和混合PCs的努力，并概述了指导未来研究的挑战和开放性问题。

    We present a comprehensive survey of the advancements and techniques in the field of tractable probabilistic generative modeling, primarily focusing on Probabilistic Circuits (PCs). We provide a unified perspective on the inherent trade-offs between expressivity and the tractability, highlighting the design principles and algorithmic extensions that have enabled building expressive and efficient PCs, and provide a taxonomy of the field. We also discuss recent efforts to build deep and hybrid PCs by fusing notions from deep neural models, and outline the challenges and open questions that can guide future research in this evolving field.
    
[^295]: 基于深度学习的随机偏微分方程数值逼近算法

    Deep learning based numerical approximation algorithms for stochastic partial differential equations

    [https://arxiv.org/abs/2012.01194](https://arxiv.org/abs/2012.01194)

    本文提出一种基于深度学习的随机偏微分方程逼近算法，通过神经网络沿噪声轨迹逼近SPDE解并估计其经验分布，在随机热方程、Black-Scholes方程和Zakai方程等测试中实现了高达100维空间下的快速精确求解。

    

    在这篇文章中，我们介绍了一种基于深度学习的随机偏微分方程（SPDEs）逼近算法。我们的方法采用神经网络来逼近SPDEs在给定驱动噪声过程实现下的解。当应用于一组模拟的噪声轨迹时，该方法可以产生SPDE解的经验分布，从中能够估计诸如均值和方差等泛函。我们在具有加性和乘性噪声的随机热方程、具有乘性噪声的随机Black-Scholes方程以及来自非线性滤波理论的Zakai方程上测试了该方法的性能。在所有情况下，所提出的算法在高达100个空间维度上都能产生准确的结果，且运行时间短。

    arXiv:2012.01194v3 Announce Type: replace-cross  Abstract: In this article, we introduce a deep learning based approximation algorithm for SPDEs. Our approach employs neural networks to approximate the solutions of SPDEs along given realizations of the driving noise process. If applied to a set of simulated noise trajectories, it yields empirical distributions of SPDE solutions, from which functionals like the mean and variance can be estimated. We test the performance of the method on stochastic heat equations with additive and multiplicative noise as well as stochastic Black-Scholes equations with multiplicative noise and Zakai equations from nonlinear filtering theory. In all cases, the proposed algorithm yields accurate results with short runtimes in up to 100 space dimensions.
    

