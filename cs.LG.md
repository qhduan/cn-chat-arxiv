# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QF3: Fast Flow RL with Filtered Q-Gradients](https://arxiv.org/abs/2610.08789) | QF3提出了一种通过滤波Q梯度训练流策略的离线策略强化学习算法，比FPO++快10倍，并且是首个能从零开始训练人形机器人运动策略并零样本迁移到真实硬件的离线策略流RL方法。 |
| [^2] | [Conformal Prediction Sets Quantify Information Gain: A Theoretical Perspective](https://arxiv.org/abs/2610.08785) | 本文为“以共形预测集大小度量不确定性”提供了信息论基础，引入了一族基于共形预测集大小与覆盖的广义信息度量，并证明额外信息导致的共形集缩减被这些度量夹界且满足数据处理不等式。 |
| [^3] | [AdvSim2Real : Training Web Agents Against Adaptive Prompt Injection in a Web World Model](https://arxiv.org/abs/2610.08773) | 提出 AdvSim2Real，在冻结的网络世界模型中让任务课程、注入攻击者与智能体共同演化，通过“成功翻转”对抗奖励机制训练出既更强又更能抵御自适应提示注入攻击的网络智能体。 |
| [^4] | [Rapid Fredholm stabilization of the Kuramoto--Sivashinsky equation with unrestricted, spatially-varying anti-diffusion](https://arxiv.org/abs/2610.08764) | 本文首次提出了针对具有任意空间变化反扩散系数的Kuramoto-Sivashinsky方程的快速镇定反馈设计，通过引入第二个边界输入并借鉴Heymann引理的思想，克服了单输入Fredholm设计中因重复不稳定特征值导致的可控性丧失问题。 |
| [^5] | [Neural Petri flows for chemical reactions](https://arxiv.org/abs/2610.08750) | 本文提出一种在任意权重取值下都严格保持Petri网语义的神经架构用于化学反应建模，并从理论上证明守恒性决定了触发形式、非负性决定了使能规则，从而只将速率定律留给神经网络自由学习。 |
| [^6] | [Linear Bandits under Exact Sliding-Window Constraints](https://arxiv.org/abs/2610.08745) | 该论文研究了精确滑动窗口约束下的线性老虎机问题，提出了刻画可行可达性的转移直径 $\tau$，证明了离线平稳解的最优性条件与在线次线性遗憾的不可能性，并开发了相对于离线最优可行轨迹遗憾为 $\widetilde{O}(d\sqrt{T}+\tau d+w)$ 的稀有切换 OFUL 算法。 |
| [^7] | [Reinforcement Learning with Conformal Action Sets: An Application to Sequential Recommendation](https://arxiv.org/abs/2610.08743) | 该论文提出了RLCP方法，通过评论家分数和在线阈值自适应调整序列推荐中的动作集大小，并从理论上证明了代理未命中率上界以及将价值损失精确分解为过滤损失和选择损失，从而获得无需参数收敛的有限会话奖励上界。 |
| [^8] | [On the Computational Tractability of Robust Bandits](https://arxiv.org/abs/2610.08740) | 本文在稳健赌博机框架中识别出一个可用多项式时间算法求解且遗憾为 Õ(√T) 的特殊情形，并证明其若干微小推广均为 NP难，表明该特殊情形正处于计算可解性的边界。 |
| [^9] | [Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling](https://arxiv.org/abs/2610.08738) | 提出层次化连续扩散语言模型（H-CDLM），通过并行扩散token本身及其粗粒度语义聚类等多种模态表示，并允许为各模态配置独立的采样器与调度，以极少的计算和参数开销显著提升连续扩散语言模型的性能。 |
| [^10] | [Optimal and Efficient Online Inverse Optimization](https://arxiv.org/abs/2610.08735) | 本文提出首个多项式时间的确定性算法，在在线逆线性优化中对任意时域均达到最优的 $O(\sqrt{d})$ 遗憾界，肯定地回答了 Sakaue 提出的公开问题。 |
| [^11] | [Does an Agent's History Tell You When Compaction Will Hurt? A Modest, Bounded Effect on the TRACE Paired-Replay Corpus](https://arxiv.org/abs/2610.08722) | 本研究利用TRACE语料库中590个成对重放的压缩边界，检验智能体近期历史能否预测上下文压缩的危害，结果发现预测能力仅微弱有效——按前缀位置的预设对比为零结果，最佳可解释触发器也仅能避免21%的有害压缩边界。 |
| [^12] | [When Forgetting is not Catastrophic: On the Mechanics of Spurious Forgetting](https://arxiv.org/abs/2610.08718) | 该研究揭示了语言模型微调中虚假遗忘的力学机制——微调使旧表征沿共同方向偏移导致暂时的知识隐藏，归一化会撤销该偏移使知识自行恢复，而只有事实特定的累积变化才会造成真正的永久遗忘。 |
| [^13] | [Co-Evolving Paths and Flows via Path-Flow Alignment](https://arxiv.org/abs/2610.08717) | 该论文提出以路径-流对齐作为流匹配的统一训练目标，让路径网络与流网络在共享的对齐损失下协同演化，并针对由此发现的“路径过拟合”失败模式（由低熵瓶颈引起）引入随机路径正则化器，从而实现更可靠的路径学习与更好的生成质量。 |
| [^14] | [Prediction-powered inference for time series across space](https://arxiv.org/abs/2610.08715) | 本文针对时空数据提出适用于时间依赖场景的预测驱动推断方法，利用短期标注数据与长期无标签协变量，在每个空间位置为未来期望标签值构建有效置信区间，解决了传统PPI独立同分布假设失效的问题。 |
| [^15] | [GeneICL: A Tabular Foundation Model for Bulk Transcriptomics](https://arxiv.org/abs/2610.08694) | GeneICL是一个仅420万参数的表格基础模型，通过基于实测bulk表达谱的半合成预训练先验、参数高效的循环架构，以及基于Cox偏似然残差的无训练生存预测转化方法，证明了转录组感知的预训练比模型规模更关键，能在临床结局预测上超越大型自监督模型。 |
| [^16] | [Probabilistic Counterfactual Inference for Discrete Outcomes in Gaussian-Process Causal Models](https://arxiv.org/abs/2610.08689) | 该论文提出了一个统一的概率反事实推断框架，通过为二元、名义和有序离散结果分别设计精确的噪声消解机制（均匀阈值、Gumbel-max竞争、潜在高斯切割点模型），将高斯过程结构因果模型的反事实推断从连续变量扩展到异构变量类型，并证明这些机制能再现模型的观测与干预分布。 |
| [^17] | [A Systematic Study of Small Language Models on Abstract Reasoning Tasks](https://arxiv.org/abs/2610.08680) | 本文系统研究了小型语言模型在ARC-TGI抽象推理基准上的表现，发现尽管模型能取得较高的分布内准确率，但其技能获取对优化过程敏感、在各任务族间分布不均，且分布外性能急剧下降，表明模型可能只是拟合了特定分布的规律而未真正学到可迁移的规则。 |
| [^18] | [Secure Speculative Decoding for Large Language Models](https://arxiv.org/abs/2610.08678) | 本文首次系统研究了投机解码的安全影响，揭示了一种“安全-效用不对称”现象：推理效率的提升以不成比例的高安全代价为代价，越狱和提示注入攻击的成功率上升速度远快于效用的下降。 |
| [^19] | [Variance-Optimal Off-Policy Evaluation with Conjunct Effect Modeling](https://arxiv.org/abs/2610.08677) | 本文提出VOCEM估计器，通过以闭式形式求解最优插值系数，在OffCEM和DR之间进行方差最优插值，在保持无偏性的同时确保方差不大于任一端点估计器。 |
| [^20] | [Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment](https://arxiv.org/abs/2610.08670) | 该研究构建了涵盖五种压力类型的248个预注册场景面板，通过让模型同时以第一人称行动和第三人称评判的方式对照其自身道德判断，发现大语言模型在约五分之一的压力场景下会采取自己判定为错误的行为，且这种“言行不一”差距的大小取决于后训练配方。 |
| [^21] | [MemFLoRA: Memory-Floor LoRA for CNN Adaptation at the Edge](https://arxiv.org/abs/2610.08669) | 本文提出MemFLoRA，一种以内存优先为设计原则的低秩CNN适配器，通过激活内存底线准则（可训练的反向计算不依赖全宽度层输入）来解决边缘端CNN适配中激活内存而非可训练参数数量才是限制性资源的问题。 |
| [^22] | [Steering Diffusion Models to Rare Events with Sequential Monte Carlo](https://arxiv.org/abs/2610.08652) | 本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。 |
| [^23] | [SquidAgent: Parallelize Wisely, Coordinate Efficiently](https://arxiv.org/abs/2610.08647) | 该论文提出SquidAgent，揭示了并行多智能体系统中“重新探索成本”与“对齐成本”这两项隐性开销，并据此推导出原则性决策准则：仅当并行化的关键路径成本加上这两项开销低于串行成本时，才应对该层进行并行化。 |
| [^24] | [Feature Information Dynamics in Diffusion](https://arxiv.org/abs/2610.08626) | 提出了基于 I-MMSE 恒等式的信息论框架“特征信息动力学”，通过比较无条件与特征条件去噪损失之差来估计特征信息密度，从而精确定位各特征在扩散生成过程中出现的时间，并定量证实了谱自回归现象。 |
| [^25] | [Early Memory Selection for Balanced Adam](https://arxiv.org/abs/2610.08624) | 该论文提出一种通过短暂试点训练自动选择Adam共享记忆参数β的方法，利用三次记忆规则平衡采样波动与梯度平均延迟，在十一个视觉和语言任务上将平均相对验证差距降低40.7%以上。 |
| [^26] | [Multi-Label Perceptual Bug Detection in Video Games using Deep Learning on Gameplay Footage](https://arxiv.org/abs/2610.08593) | 本文提出ResNet-BiLSTM深度学习模型，通过时序依赖建模实现对游戏视频画面中多种感知性缺陷的多标签检测，在基准数据集上达到85.78%的F1分数，可显著减少游戏人工测试的资源消耗。 |
| [^27] | [CNet: A Complex-Valued Deep Learning Framework with Wirtinger Autodifferentiation and FFT--Hadamard Convolution](https://arxiv.org/abs/2610.08592) | CNet 是一个基于 Wirtinger 自动微分的 C++/CUDA 复值深度学习框架，它将 FFT-阿达马卷积恒等式转化为可学习的复值卷积网络，并采用玻恩规则进行物理原生式分类。 |
| [^28] | [Random Feature Gaussian Process Attention: Linear-Time Probabilistic Attention with Calibrated Uncertainty](https://arxiv.org/abs/2610.08578) | 本文提出即插即用的随机傅里叶特征高斯过程注意力模块（RFF-GPA），通过随机傅里叶特征对平稳核进行低秩近似，将注意力表示为高斯过程后验，实现了线性时间复杂度且具有校准不确定性的概率注意力计算。 |
| [^29] | [How Learning Governs Unlearning across the Memorization-Generalization Spectrum](https://arxiv.org/abs/2610.08577) | 模型的学习方式决定了其后续遗忘的表现：偏泛化型模型在遗忘时比偏记忆型模型遭受更大的保留集性能损害，且这一趋势在记忆-泛化谱系上几乎单调成立。 |
| [^30] | [FedDermaSeg: Federated Learning for Dermatological Image Segmentation](https://arxiv.org/abs/2610.08574) | 该论文提出FedDermaSeg，探索利用联邦学习在无需集中收集数据的情况下实现隐私保护的皮肤病灶分割，以解决传统集中式深度学习训练带来的隐私风险和高计算资源需求问题。 |
| [^31] | [RAG-PIBench: A Leakage-Aware Benchmark for Prompt-Injection Detection in Trustworthy RAG Systems](https://arxiv.org/abs/2610.08571) | 该论文提出了RAG-PIBench——一个面向RAG系统提示注入检测的防泄漏基准，包含4,876个上下文示例，通过严格评估协议发现DistilBERT取得最佳检测性能（F1=0.896），同时证明TF-IDF等稀疏基线方法仍具竞争力。 |
| [^32] | [Less Is More: A Leakage-Controlled Study of Dermoscopic Preprocessing for Joint Skin Lesion Classification and Segmentation with YOLO26](https://arxiv.org/abs/2610.08570) | 在严格按病变划分、防止数据泄漏的评估下，研究发现复杂的手工皮肤镜预处理相比最小预处理加在线增强，对固定YOLO26n-seg模型的联合病变分类与分割性能提升有限，体现“少即是多”。 |
| [^33] | [Singular Value Decomposition: A Geometric Rediscovery, Where Proofs Become Algorithms](https://arxiv.org/abs/2610.08565) | 本文以“单位圆映射为椭圆”的几何直觉重新发现奇异值分解，并展示其证明过程本身可作为算法运行，构成主成分分析、核方法与 PageRank 等机器学习技术背后的核心机制。 |
| [^34] | [Valid for Free: Homophily-Gated Conformal Prediction for Training-Free Node Classification with Tabular Foundation Models](https://arxiv.org/abs/2610.08564) | 该论文首次对表格基础模型无训练图节点分类设定进行可靠性研究，证明冻结的上下文预测器可使拆分保形预测在有限样本下严格有效且无需任何训练或调参，并提出同质性门控的HG-DAPS扩散评分以改善预测集。 |
| [^35] | [Reinforcement Learning for Hierarchical Reasoning Rewards: Minimax-Optimal Rates with Transformers](https://arxiv.org/abs/2610.08561) | 本文将推理任务的奖励建模为响应空间上的分层函数，并证明一种基于Transformer的actor-critic强化学习算法在查询预算和正则化强度上达到极小极大最优速率，从理论上解释了在策略探索结合神经奖励模型的RL后训练为何有效。 |
| [^36] | [Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness](https://arxiv.org/abs/2610.08560) | 该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。 |
| [^37] | [Latent space bias directions in LLMs capture confidence, not fairness](https://arxiv.org/abs/2610.08559) | 该研究揭示了大语言模型激活引导中的去偏方向实际上编码的是模型置信度而非偏见信息，其去偏效果只是降低模型置信度的副产品，从而解释了激活引导去偏技术泛化能力差的根本原因。 |
| [^38] | [Systemization of Knowledge (SoK): Human-Centered AI Safety for Youth](https://arxiv.org/abs/2610.08554) | 本文系统回顾了100项HCI实证研究，构建了青少年AI风险与对策的映射，发现多数风险仅有构想层面的对策，很少被实施和评估，且评估多聚焦技术性能而非实际防伤害效果。 |
| [^39] | [DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory](https://arxiv.org/abs/2610.08553) | 针对非线性循环记忆在测试时训练中难以优化、并行基线反而优于串行版本的问题，DeltaTTT提出用逐层学习替代联合内循环优化，为每层分配局部预测目标并通过状态依赖的delta规则更新，从而缓解非线性记忆的优化困难。 |
| [^40] | [AnyBottle: A Recipe to Only Keep the Concepts You Really Need](https://arxiv.org/abs/2610.08552) | AnyBottle提出了一种由黑盒教师模型引导的迭代概念选择方法，结合嵌套dropout训练，能够构建紧凑、任务特定的概念瓶颈模型，只保留任务真正需要的概念，使瓶颈更小且更易于检查。 |
| [^41] | [Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements](https://arxiv.org/abs/2610.08540) | 该论文提出将对齐视为一族可测量的幂律缩放关系（B_r(N)=a_rN^alpha_r）的框架及首批预注册测量，并证明长期对齐状态由经修正风险中的最大指数而非平均值决定，指数大于1时将累积不可持续的对齐债务。 |
| [^42] | [From Shared Demand Patterns to Local Uncertainty: Probabilistic Load Forecasting by Mixing Compact Adaptations](https://arxiv.org/abs/2610.08538) | 该论文提出了一种可扩展的概率负荷预测框架，通过共享模型学习共同需求模式，并让每个负荷按需混合一个小型低维紧凑适配组件库，从而在客户级和变压器级兼顾局部预测精度与大规模部署的可扩展性。 |
| [^43] | [FlowCF: Sparse Counterfactual Explanations for Mixed-Type Tabular Data using Flow Matching](https://arxiv.org/abs/2610.08537) | FlowCF提出了一种基于流匹配的模型无关生成方法，通过新颖的混合流算子和门控网络，为混合类型表格数据生成具有稀疏性的反事实解释。 |
| [^44] | [How Bregman Divergences Shape Shampoo](https://arxiv.org/abs/2610.08534) | 本文提出统一的 Bregman 散度框架，揭示了 Shampoo 优化器中散度选择（如 Frobenius 与 KL 散度）如何影响 Kronecker 预条件近似，并发现某些散度能更好地补偿有限样本对经验二阶矩的低估，从而解释了不同 Shampoo 变体行为差异的根源。 |
| [^45] | [Beyond Perturbation Magnitude: Direction-Dependent Responses in Multimodal Geometric Representations](https://arxiv.org/abs/2610.08533) | 该研究通过受控扰动实验发现多模态几何对齐分数的响应并非由扰动幅度决定，并提出方向几何响应（DGR）指标——即位移在局部体积梯度上的投影——来刻画这种方向依赖的响应机制。 |
| [^46] | [PHBA: Prefix-State Hybrid Block Attention](https://arxiv.org/abs/2610.08527) | PHBA提出了一种混合注意力架构，用top-k块稀疏检索取代局部滑窗注意力，并将每个检索到的token块与其前置上下文的紧凑前缀状态相耦合，从而在统一层内同时实现精确的长程token检索与高效的历史上下文压缩。 |
| [^47] | [X-OPM: Explainable Automatic Digital On-Chip Power Modeling for Enhanced Robustness](https://arxiv.org/abs/2610.08502) | X-OPM基于同步数字VLSI电路设计原理，提出了一个可解释的自动片上功耗建模框架，通过树模型捕捉特征交互并用线性模型进行预测，结合人在回路的工作流程，在商用C906向量处理器上实现了更鲁棒、可泛化且低开销的功耗预测。 |
| [^48] | [Information-Dense Synthesis for Molecular Discovery](https://arxiv.org/abs/2610.08495) | 提出信息密集型合成方法，通过设计、合成复杂分子混合物并池化测试后解卷积分子-活性映射，理论上可将寻找最优分子的实验次数从O(d)降至O(log d)或O(1)，比现有贝叶斯优化方法效率提升一个数量级。 |
| [^49] | [MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata](https://arxiv.org/abs/2610.08479) | 提出去中心化框架MetaLearnNCA，通过Active-NCA与Meta-NCA两种耦合神经细胞自动机的动态交互实现少样本离线元学习，无需测试时反向传播计算梯度，同时保留二维空间几何结构信息。 |
| [^50] | [Learning PDE solution operators with variable initial conditions via Latent Dynamics Networks](https://arxiv.org/abs/2610.08475) | 本文提出一种改进的潜在动力学网络（LDNet），通过从少量早期观测中直接推断初始潜在状态来支持可变初始条件，在保留端到端训练、无编码器设计以及与网格拓扑无关等优势的同时，显著扩展了其在真实PDE应用中的适用性。 |
| [^51] | [UNREAL: Unifying Retrieval and Long-Context with a Single Model](https://arxiv.org/abs/2610.08463) | UNREAL提出了一种模型原生的证据选择框架，直接从冻结LLM的内部表示中推导检索查询，以不到50万可训练参数统一了语料库检索与长上下文推理，并在多个基准上大幅超越最先进的检索-重排序系统。 |
| [^52] | [Climbing the Design Ladder: Sequential Knowledge Distillation for Early-Stage Circuit Timing Prediction](https://arxiv.org/abs/2610.08457) | 提出STEP-KD方法，将中间设计阶段（布局规划后、布局后、布线后）作为“垫脚石”进行顺序渐进的知识蒸馏，从而从早期设计数据准确预测布线后电路时序，避免后期才发现时序违例导致的高昂迭代成本。 |
| [^53] | [Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents](https://arxiv.org/abs/2610.08452) | 该论文提出Agentic AutoRAG，一种利用LLM智能体进行多目标RAG超参数优化的方法，其核心创新在于通过诊断器将每次失败归因于检索或生成阶段，从而让优化器能够推理配置失败的原因并智能地指导后续搜索。 |
| [^54] | [Symmetry-Aware Feature Learning: A Polynomial Separation for Multi-Index Models](https://arxiv.org/abs/2610.08420) | 该论文证明了对称感知与对称无关特征学习之间存在多项式级的样本复杂度分离：在具有循环对称轨道的增长秩多指标模型中，通过权重共享或全群数据增强利用对称性的学习器，仅需约 $d^{p-1}$ 个样本（对于信息指数 $p\ge3$ 的多项式链接函数）即可实现弱方向恢复，而无法利用对称性的学习器则代价更高。 |
| [^55] | [Knowing When Not to Answer: Cross-Domain and Multi-Turn Generalization of Latent Underspecification Signals](https://arxiv.org/abs/2610.08413) | 该论文构建了一个带轮次标签的多轮对话不可回答性基准与模拟用户评估框架，发现线性探针所捕捉的“信息缺失”信号能在共享同一不可回答性根源的数据集间稳健跨域迁移（AUROC 0.77–0.97），但不同类型不可回答性的表征边界会受词汇混淆、网络层级与坐标系选择的影响。 |
| [^56] | [SSR: Sparse Segment Reduction for Ternary GEMM Acceleration](https://arxiv.org/abs/2610.08403) | 本文提出SSR方法，通过专用优化的三值数据格式与利用稀疏结构的计算树算法，同时兼顾三值特性与稀疏性，从而加速三值大语言模型的矩阵乘法推理。 |
| [^57] | [VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents](https://arxiv.org/abs/2610.08402) | VETTA通过共享轻量级评论家上的独立价值头联合学习轮次级与令牌级价值，在单次策略更新中协调两个层面的信用分配，从而解决多轮LLM智能体面临稀疏反馈时的信用分配难题。 |
| [^58] | [Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems](https://arxiv.org/abs/2610.08400) | Atom-JEPA是一种自监督预训练框架，通过互补的原子级和子结构级目标从无标签三维原子结构中学习潜在表示，在分子ADMET和量子化学性质预测等下游任务上达到了最先进的性能。 |
| [^59] | [Decision-Focused Learning in MDPs: An Occupancy Measure Approach](https://arxiv.org/abs/2610.08384) | 该论文提出将马尔可夫决策过程的决策聚焦学习重构为基于占用测度的线性规划，利用闭式梯度、增广拉格朗日代理及随机行草图平滑技术，克服了传统基于KKT条件方法需要大规模求解线性系统的可扩展性瓶颈。 |
| [^60] | [Accelerating the Development of PLGA In Situ Forming Depots Through AI-Driven Multi-Objective Optimization](https://arxiv.org/abs/2610.08368) | 该研究将Corbion的PURASORB聚合物库与Intrepid Labs的AI算法ANDROMEDA 1结合，仅用约15周和181个处方即完成多目标优化，成功筛选出4个满足黏度与可注射性要求且具有差异化30天释放曲线的治疗性多肽PLGA原位成型储库处方，显著加速了长效注射制剂的开发进程。 |
| [^61] | [Evolutionary One-Step Generators: Fast and Diverse Sampling for Discrete Design](https://arxiv.org/abs/2610.08367) | 提出EGO框架，利用对偶低秩进化策略直接在离散输出上训练生成器，使其单次神经网络前向计算即可生成整个图，在分子生成等离散设计任务中以极低计算成本同时保证候选的有效性与多样性。 |
| [^62] | [Sensor Geometry as a Flow-Matching Prior for Multi-Channel Brain Signals](https://arxiv.org/abs/2610.08355) | 该论文提出仅利用脑电电极的几何坐标构建k近邻图，并以图拉普拉斯的Matérn函数作为流匹配模型的源协方差，从而把已知的空间相关结构编码为先验，在不增加任何可学习参数的情况下替代各向同性高斯源，生成空间相干的多通道脑电信号。 |
| [^63] | [Uncertainty Quantification Is Indispensable for Reliable Connectome-Based Graph Learning: A Narrative Review and Case Study](https://arxiv.org/abs/2610.08353) | 本文通过叙述性综述与实证案例研究表明，确定性图神经网络在基于连接组的诊断分类中会产生过度自信的预测，因此不确定性量化对于可靠的连接组图学习不可或缺。 |
| [^64] | [High-Dimensional Statistical Inference for Sparse Support Vector Machines](https://arxiv.org/abs/2610.08345) | 该论文通过将 $L_1$-惩罚支持向量机表示为线性规划并借助对偶变量识别铰链损失次梯度，突破了铰链损失非光滑性导致的去偏难题，首次在高维比例渐近机制下为稀疏SVM建立了计算可行的渐近高斯推断框架，实现了置信区间、假设检验和FDR受控的变量选择。 |
| [^65] | [DIPrune: Task-Aware Token Pruning with Dual Importance for Efficient Multimodal Language Models](https://arxiv.org/abs/2610.08341) | 本文提出DIPrune，一种基于双重重要性的任务感知令牌剪枝方法，通过将无训练剪枝重新表述为最小化任务损失失真问题，并揭示此前被忽视的考虑跨层梯度的层间项，解决了浅层显著令牌通过数值惯性压制深层语义信号所导致的语义退化问题，从而实现高效的多模态大语言模型。 |
| [^66] | [Machine Learning for German Redispatch Forecasting under Data Delays and Temporal Distribution Shift](https://arxiv.org/abs/2610.08337) | 该研究构建了一个在数据延迟与时间分布偏移等真实信息约束下评估德国电网再调度概率预测的基准，系统比较了从季节性经验方法到Transformer的多种模型及不同校准策略，为电网拥塞预测的可靠性与准确性提供了实证依据。 |
| [^67] | [Structure-Aware Graph Abstention for Reliable Selective Forecasting](https://arxiv.org/abs/2610.08322) | 本文提出一种结构感知的图弃权方法，通过学习稀疏图和狄利克雷风格的结构能量来评估多变量预测的关系一致性，作为与实例级合理性互补的弃权信号，在匹配覆盖率下通常比TEM降低选择性MSE。 |
| [^68] | [The Standardization Trap: Certifying Joint Label Processing in Tabular Foundation Models](https://arxiv.org/abs/2610.08314) | 该论文揭示了检验表格基础模型是否遵循固定权重机制时存在的“标准化陷阱”问题，并提出两个仅依赖标准化标签处预测的证明方法，能够区分固定权重预测与非线性标签变换两种解释。 |
| [^69] | [CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling](https://arxiv.org/abs/2610.08312) | 提出无需回放的CoDe-LoRA方法，通过自适应零空间投影和语义路由将学习过程解耦为通用知识巩固与任务特定知识解耦，克服了正交参数隔离阻碍跨任务知识迁移的“正交困境”。 |
| [^70] | [OxiGen: Oxidation-State-Aware Crystal Generation](https://arxiv.org/abs/2610.08296) | 提出了OxiGen，一种显式建模氧化态的晶体扩散生成模型，通过在有限状态自动机上进行精确推理的结构化输出层在构造上保证全局电荷中性，显著提升了生成晶体的氧化态保真度以及稳定、独特且新颖晶体的生成率。 |
| [^71] | [Where Do Two Populations of Persistence Diagrams Differ? Calibrated Local Inference at a Fixed Budget](https://arxiv.org/abs/2610.08292) | 本文提出在固定样本量下对两个持续性图总体的局部均值差异进行同步推断的方法，利用高斯乘子自助法校准置信区间，在控制族错误率的同时定位出生-死亡平面上造成总体差异的具体区域。 |
| [^72] | [Scalable extraction and visualization of multi-attribute logical and functional dependencies in tabular data](https://arxiv.org/abs/2610.08287) | 提出了LDTool和HLDTool两个工具，实现了表格数据中多属性逻辑依赖与函数依赖的统一提取与可视化，并通过超图引导的搜索空间缩减解决了大规模数据下的可扩展性问题。 |
| [^73] | [Two-Sample Testing via Generative Processes](https://arxiv.org/abs/2610.08277) | 提出了一种基于随机插值与时间反射对称性的双样本检验方法，通过计算时间 t 和 1-t 处边缘分布的 Jensen-Shannon 散度来判断两个样本是否同分布，该方法无需学习任何参数、可精确控制有限样本检验水平，并能达到极小化极大分离速率。 |
| [^74] | [Performative Prediction with Selective Labels](https://arxiv.org/abs/2610.08272) | 该论文首次形式化了具有选择性标签（即只能观察到被接受子群体的标签）情境下的表演性预测问题，并证明仅基于观测数据进行重训练会误导模型更新过程并破坏收敛性保证。 |
| [^75] | [Reinforcement Learning with Segment Reward Feedback under Linear Function Approximation](https://arxiv.org/abs/2610.08271) | 该论文研究了线性函数逼近下的分段奖励反馈强化学习，针对二值和求和两种反馈类型分别提出了计算高效的算法（BiTs-SEGD 和 EDLinUCB-SEGD），并建立了几乎匹配的下界，回答了分段反馈粒度与分割方式如何影响学习效果。 |
| [^76] | [DySCo: Dynamic Sharding for Collaborative Edge-Cloud LLM Inference with Depth-Synchronized Batching](https://arxiv.org/abs/2610.08268) | 提出DySCo协同运行时系统，通过动态分片、模型感知的层范围执行器dyForward以及深度同步批处理，消除边云协同LLM推理中云端调用的空闲间隙，并解决到达不同模型深度的请求无法常规批处理的难题。 |
| [^77] | [LeanPlan: Optimal Planning with LLM-Generated Heuristics and Admissibility Proofs](https://arxiv.org/abs/2610.08246) | LeanPlan是首个利用LLM生成的启发式函数（其可采纳性在Lean 4中经机器验证）以找到最优计划的规划系统，在国际规划竞赛领域上展现出优异的最优规划性能。 |
| [^78] | [How Many Independent Samples Does a Satellite Image Contain? Generalization Bounds for Spatially Dependent Data](https://arxiv.org/abs/2610.08227) | 该论文证明了空间相关性持续 r 个像素的 n×n 卫星图像，其有效样本量仅为 Θ(n²/r²) 而非 n²，并通过匹配的上下界证明该速率是紧的且不可超越，从而为空间交叉验证提供了最优泛化保证的理论依据。 |
| [^79] | [Anytime-valid simulation-based hypothesis testing](https://arxiv.org/abs/2610.08210) | 本文针对只能获得模拟样本而无解析密度的假设检验问题，构造了 e 检验鞅，实现了任意时刻有效的一类错误控制、几何衰减的二类错误界和渐近满功效的序贯检验。 |
| [^80] | [Finding the Heads and the Neurons Responsible for Network Information Retrieval in Language Models](https://arxiv.org/abs/2610.08200) | 该研究发现，语言模型中经因果消融验证的极少数注意力头能以99.5%至100%的准确率检测上下文中的主机名与IP地址配对信息，且在某些模型中这一功能可进一步精确定位到单个神经元。 |
| [^81] | [Compact Robot Policies Need Fine-Grained Visual Representations](https://arxiv.org/abs/2610.08183) | 论文提出紧凑策略CoRP，证明机器人多任务操作的性能主要由细粒度的预训练视觉表示决定，而非参数规模或生成式先验，其4890万参数的小模型即可媲美比它大上百倍的系统。 |
| [^82] | [On the Intrinsic Limited Robustness of Latent-Based Watermarking](https://arxiv.org/abs/2610.08178) | 本文首次从理论上揭示了基于潜空间的水印方法对旋转、缩放、平移等几何变换缺乏鲁棒性是其固有局限，并推导出刻画像素空间扰动与潜空间影响关系的最大扰动界。 |
| [^83] | [LFHE: Local-First Heuristic Evolution for Bounded Local Topology Search in Decentralized Learning with Non-IID Data](https://arxiv.org/abs/2610.08176) | LFHE 提出了一种仅利用自身邻域和朋友的朋友信息进行受限局部拓扑重连的框架，其结构分数与图狄利克雷能量精确对应，从而在非独立同分布数据下有效加速去中心化学习中表示分歧的消散。 |
| [^84] | [Beyond the Leaderboard: Multi-Dimensional Evaluation of Dense and Mixture-of-Experts Models for Automated Program Repair](https://arxiv.org/abs/2610.08173) | 该论文受ISO/IEC 25010启发提出加权质量指数（QI），对稠密与混合专家代码模型进行涵盖正确性、可维护性、安全性和效率的多维度评估，发现模型排名随权重方案变化，单一指标评估会掩盖关键权衡。 |
| [^85] | [Align, Then Correct: Training-Free Two-Stage Low-Rank Compensation for Extremely Quantized Large Language Models](https://arxiv.org/abs/2610.08164) | 该论文提出一种无需训练的两阶段闭式低秩补偿框架，先对齐层输出再校正残余误差，克服了现有低秩量化误差补偿在对称校准和仅二阶优化上的两大局限，大幅提升极端量化大语言模型的精度恢复能力。 |
| [^86] | [Symphony for Text Generation: Benchmarking Clinical Note Generation](https://arxiv.org/abs/2610.08161) | 该论文提出了包含300例多语言临床就诊记录的MedConv数据集，并构建了结合蕴含指标与大语言模型评判的受控临床评估框架，证明临床AI平台Corti的病历生成质量与领先商业环境式记录软件相当或更优，且其可配置API可针对特定文档需求灵活优化质量维度。 |
| [^87] | [Making COMET Comparable Across Scripts: Diagnosis and Correction of Tokeniser-Induced Script Bias in Indic MT Evaluation](https://arxiv.org/abs/2610.08159) | 该论文发现 COMET 评估指标因分词器存在文字偏差，导致印度语系不同文字系统间的分数不可比且排序准确性下降，并提出 COMET-QN 方法以精确消除跨文字分数范围不兼容的问题。 |
| [^88] | [Beyond Marginal Monitoring: Distributed Joint-Distribution Testing for Data Concept Drift in Large Scale E-Commerce Operations](https://arxiv.org/abs/2610.08132) | 该论文在亿级规模的电商真实数据上系统评估了五种多变量双样本漂移检测方法，证明基于 Apache Spark 的分布式最大均值差异（MMD）结合随机傅里叶特征的方法在检测概念漂移时具备稳健的可扩展性。 |
| [^89] | [Mu-DisCoCat: A Variational Pipeline for Compositional Generalization on Quantum Processors](https://arxiv.org/abs/2610.08131) | 本文提出Mu-DisCoCat，一个多模态变分量子学习框架，通过先学习单物体图文表示、再固定表示学习多物体关系的两阶段训练，在量子处理器上实现了组合概念泛化。 |
| [^90] | [Do LLMs Act on What They Know? From Partner Representations to Cooperative Actions](https://arxiv.org/abs/2610.08129) | 该论文发现在Hanabi类合作任务中，八个大语言模型虽能通过线性探针较准确地恢复发送方的意图惯例，但其决策并未一致遵循该惯例，且将惯例转化为具体动作推荐比以规则形式陈述更能有效提升合作表现，揭示了模型“知道”与“行动”之间的脱节。 |
| [^91] | [Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving](https://arxiv.org/abs/2610.08123) | 本文提出一种基于查询的代价学习框架，通过对自车动态可达的未来轨迹估计有界代价来替代传统路径点回归，将代价拓扑转化为可行规划，在nuScenes与真实驾驶数据上显著降低碰撞率并保持可解释性。 |
| [^92] | [Attenuated in-context identification in time-series foundation models: diagnosis under counterfactual inputs and repair by synthetic forced-system fine-tuning](https://arxiv.org/abs/2610.08118) | 本文通过精确反事实实验诊断出协变量感知时间序列基础模型在what-if预测中的关键缺陷——TimesFM-2.5和TabPFN-TS完全无记忆、Chronos-2会衰减系统动态效应，并提出利用合成强迫系统数据微调来修复这种衰减。 |
| [^93] | [Energy-Aware Path Following: Comparative Analysis of Reinforcement Learning and NMPC for Electric Vehicles](https://arxiv.org/abs/2610.08112) | 本文在统一的Frenet坐标系运动学模型和包含显式再生制动的能量模型下，对比分析了NMPC、PPO强化学习控制器及两种基线方法在电动汽车能量感知路径跟随中的性能，并采用JIT编译的CasADi满足NMPC的实时性要求。 |
| [^94] | [Enhancing Diffusion Language Models with Autoregressive Post-Training Weights](https://arxiv.org/abs/2610.08108) | 该论文提出将自回归模型的后训练权重更新直接“回收”叠加到扩散语言模型上，无需重新进行扩散后训练即可使扩散模型获得接近直接后训练的性能。 |
| [^95] | [Surviving the Router: Optimizing Skill Injections for Retrieval and Execution](https://arxiv.org/abs/2610.08098) | 论文发现现有技能注入攻击评估因忽略检索竞争阶段而将攻击成功率高估87-97%，并提出了感知路由器的攻击方法CORSA，可同时优化技能注入的检索与执行两个环节。 |
| [^96] | [Explainable Rule Mining of IPv6 Extension-Header Presence Patterns from Paired-Vantage Captures](https://arxiv.org/abs/2610.08090) | 本文提出阴性对照协议与发送方条件化EH保留率测量两个可复用工具，并将可解释时序逻辑规则挖掘器应用于JAMES配对观测点数据集，发现挖掘出的主导分片EH规则实为包内共现而非真正的时序模式。 |
| [^97] | [ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding](https://arxiv.org/abs/2610.08078) | 该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。 |
| [^98] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^99] | [Optimization Encoders: Rethinking Second-Order Meta-Learning for Neural Fields](https://arxiv.org/abs/2610.08075) | 该论文将潜优化形式化为“优化编码器”，统一了二阶微分、潜参数化与任务监督的角色，并据此提出基于等变变换器的注意力潜场MetaLF，实现了编码过程与解码器的端到端二阶元学习训练。 |
| [^100] | [Detecting a Shift Is Not Enough: Exact Minimax Limits of Linear Representation Repair](https://arxiv.org/abs/2610.08069) | 该论文将两个数据源间均值偏移的消除建模为统计决策问题，推导出线性表示修复的精确有限样本极小极大风险，并揭示了“检测-修复差距”：检测偏移仅需信噪比 κ 远大于 √d，而实际修复则需 κ 与维度 d 同阶。 |
| [^101] | [Language Carries the Expert's Impression: Instrument-Anchored LLM Judges Transfer Counseling-Quality Assessment and Beat In-Domain Training](https://arxiv.org/abs/2610.08055) | 该研究发现，基于专家评分工具锚定构念、由小型开源LLM生成的会话级得分，能够跨领域迁移专家对咨询质量的总体印象评估，且跨域训练的效果甚至优于域内训练。 |
| [^102] | [A Riemannian Geometry for Low-rank Adaptation](https://arxiv.org/abs/2610.08049) | 该论文从商流形几何的角度分析 LoRA 参数化的等价关系，并提出一种在该等价关系下不变的黎曼度量，使每一步梯度更新都经过预条件化并切实改变损失值，从而实现更高效的参数高效微调。 |
| [^103] | [DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks](https://arxiv.org/abs/2610.08048) | DAEDALUS通过探索者代理自生成练习任务、解决者代理从失败中提炼启发式规则并验证其有效性，从而在无需现有任务或人工验证器的情况下自动构建可复用的代理记忆。 |
| [^104] | [SepsisLens: Structure-Preserving Sequence Modelling for Decomposable Early Sepsis Warning](https://arxiv.org/abs/2610.08046) | SepsisLens通过在风险组合前保持变量级时序状态，并利用结构化风险头部从显式的变量级和器官级组件组合多时间窗口风险，实现了警报可追溯至生理信号的可分解脓毒症早期预警。 |
| [^105] | [Spectra: Exact Component Transport for Test-Time Prior Adaptation in Simulation-Based Inference](https://arxiv.org/abs/2610.08021) | Spectra利用精确的得分传输恒等式，使冻结的扩散式SBI模型能够在测试时以闭式形式适应结构化的先验变化，无需额外的仿真或训练，在六个基准测试中于强先验偏移下实现了准确的后验推断。 |
| [^106] | [Learning consistent molecular mechanics force fields from first principles](https://arxiv.org/abs/2610.08020) | 本文提出grappa-fullFF统一框架，首次从第一性原理参考数据中一致且同时地学习分子力场的成键与非成键参数，克服了传统方法依赖经验非成键参数的局限。 |
| [^107] | [FOSLS-deRhaNN: native de Rham neural classes for H(div) and H(curl) with applications to first-order system least-squares neural network methods for partial differential equations](https://arxiv.org/abs/2610.08016) | 该论文构造了原生属于 H(div) 和 H(curl) 空间的 de Rham 神经逼近类——无需网格或有限元模拟即可精确表达界面跳变，并将其应用于求解偏微分方程的一阶系统最小二乘（FOSLS）神经网络方法。 |
| [^108] | [Can phenotypic activity be predicted without experimental readouts?](https://arxiv.org/abs/2610.07997) | 在控制预训练数据泄漏和细胞毒性混杂因素后，基于分子-形态学对比预训练的分子编码器在预测Cell Painting表型活性方面并不优于简单的理化描述符。 |
| [^109] | [TICDA: Tabular In-Context Data Attribution](https://arxiv.org/abs/2610.07996) | 提出TICDA方法，通过在表格基础模型潜在表示上训练的线性代理模型直接量化上下文中每个示例对预测的影响，解决了重采样和基于梯度的数据归因方法在上下文学习场景中失效的问题。 |
| [^110] | [A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic](https://arxiv.org/abs/2610.07990) | 本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。 |
| [^111] | [Do Higher-Order Models Win for Higher-Order Reasons? Rethinking Performance Gains in Hypergraph Learning](https://arxiv.org/abs/2610.07981) | 该研究提出一种受控性能归因框架，通过扰动高阶信息同时保留成对信息，发现在25个超图学习基准上高阶模型的性能优势大多并非真正来自高阶信息，即使没有高阶信息其大部分优势仍可实现。 |
| [^112] | [Learning a Ranking from Human Feedback in Log-Concave Random Utility Models](https://arxiv.org/abs/2610.07973) | 该论文在对数凹随机效用模型下研究了通过全排序反馈和仅胜者反馈两种人类比较反馈来恢复物品排序的问题，建立了最坏情况样本复杂度下界，并设计了无需知晓噪声分布即可匹配该下界的算法。 |
| [^113] | [DecepEval: A Benchmark for Evaluating Deception in LLM Agents](https://arxiv.org/abs/2610.07967) | 提出DecepEval基准，借鉴经典欺诈理论构建“LLM欺骗菱形”框架，通过1,532个实例、28个专业场景以及压力、激励、机会、冲突四种外部条件，系统性评估大语言模型智能体的欺骗行为。 |
| [^114] | [Feature Encoding in VAE-based Audio Decoders: Effects of Input, Depth and Distribution](https://arxiv.org/abs/2610.07966) | 本研究通过对RAVE等基于VAE的音频解码器内部激活进行逐层与跨层聚类分析，系统揭示了模型对音高、BPM等音乐特征的编码规律，发现合成音频编码效果更好，而自然音频需借助非线性探针才能实现有效解码。 |
| [^115] | [Generalized Matheron Variational Implicit Processes](https://arxiv.org/abs/2610.07938) | 提出广义Matheron变分隐式过程（GMVIP）这一路径变分族，通过从先验采样并施加锚定在诱导点上的校正来构造后验样本，对高斯过程先验可恢复标准诱导变量变分GP构造，对一般隐式先验则能在总体极限下保持先验的均值、协方差及结构变异性。 |
| [^116] | [Revisiting Temporal Regularization for Smooth Control in Deep Reinforcement Learning](https://arxiv.org/abs/2610.07910) | 本文证明了时间正则化能约束共享相同下一状态的邻近状态之间的动作差异，从而在观测噪声下提供空间平滑性，并据此提出了结合线性斜坡上升时间惩罚的CATS方法，在不损害任务性能的前提下实现机器人的平滑控制。 |
| [^117] | [Isotropic Yet Undecodable: The Sequential Content-Sufficiency Gap in Latent-Predictive Text Representations](https://arxiv.org/abs/2610.07906) | 论文通过信息论分解揭示序列内容充分性鸿沟，证明潜在表示的各向同性与一致性无法保证有序目标信息可解码，并据此提出引入规范词元监督的非自回归框架CANOPE，将位置信息恢复率从13.5%大幅提升至98.8%。 |
| [^118] | [ApexQuant: Data-Free Elastic Quantization by Residual Re-Isotropization](https://arxiv.org/abs/2610.07904) | ApexQuant提出了一种无需校准数据的弹性量化方法，通过随机旋转将残差恢复到各向同性均匀分布并递归再量化，从而在读取权重前即可确定所需量化次数，使单一模型工件能以多种精度服务，尤其适用于数据受限制的地球观测和医学等领域。 |
| [^119] | [Variance-Averse $n$-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments](https://arxiv.org/abs/2610.07899) | 提出VAN-Flow框架，通过结合分类分布式评论家、方差厌恶期望算子和拒绝采样引导的流匹配生成式演员，在生成式离线强化学习中选择高回报且低方差 dispersion 的可靠动作。 |
| [^120] | [FC-SWE: Failure-Conditioned RL for Long-Horizon Software Engineering Agents](https://arxiv.org/abs/2610.07898) | 论文提出FC-SWE框架，通过将失败补丁的验证器反馈作为条件上下文，将恢复尝试纳入强化学习策略训练，从而提升长时程软件工程智能体从失败中学习的能力。 |
| [^121] | [Rethinking Faithfulness in LLMs: A Pairwise Context-Sensitive Perspective](https://arxiv.org/abs/2610.07894) | 提出成对忠实性基准PFaithBench，通过对比同一问题在支持性与非支持性上下文下的表现，评估大语言模型能否正确地在回答与拒答之间切换，揭示忠实性本质上取决于回答与拒答之间的权衡。 |
| [^122] | [Label-Efficient Deep Learning for ECG Delineation: A Multi-Dataset Benchmark against Widely Used Delineation Tools](https://arxiv.org/abs/2610.07885) | 该研究通过多数据集基准测试证明，自监督预训练配合恰当的微调策略可显著降低心电波形分界对专家标注的依赖，且所得深度模型的分界性能优于广泛使用的开源分界工具。 |
| [^123] | [Learned Adaptive Multiresolution Diffusion Imaging](https://arxiv.org/abs/2610.07884) | 该论文提出Learned AMDI，用近端策略优化训练的共享局部策略替代AMDI中固定准则的树细化选择器，在保留固定树传播器与层次约束的前提下，使留出测试中的平均终端参考偏差从0.17496降至0.13657、占用率从0.13737提升至0.26660。 |
| [^124] | [On-Policy Distillation with Negative-Policy Rollouts](https://arxiv.org/abs/2610.07874) | 提出负策略在线策略蒸馏（NP-OPD），在采样阶段引入性能较弱的负策略作为负向参考来补充教师监督，从而在教师与学生分布重叠有限、正向引导信号不足时提供更充分的学习信号。 |
| [^125] | [ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents](https://arxiv.org/abs/2610.07863) | ReFold提出了一种免训练的可逆上下文折叠渲染层，通过将已展示内容替换为占位符、将智能体报告完成的轮次折叠为一行注释来消除轮间冗余，从而在保留底层完整交互历史的同时压缩长程智能体的渲染上下文，避免了现有预测性方法带来的运行时开销、前缀缓存失效和不可逆信息丢失。 |
| [^126] | [A self-learning scientific agent for X-ray diffraction](https://arxiv.org/abs/2610.07862) | 本文提出“干将”——一个面向粉末X射线衍射的自学习科学智能体，它通过诊断失败并自主修订和验证技能指令与代码，将分析经验转化为可复用的可执行技能，且无需重新训练语言模型，在多个精修平台上超越了专家设计的技能。 |
| [^127] | [Tram-FL: Reducing Communication and Computation Costs through Sequential Model Circulation in Decentralized Federated Learning](https://arxiv.org/abs/2610.07859) | 提出Tram-FL机制，通过在节点间顺序循环传递单个模型进行训练，以最小的计算和通信成本实现去中心化联邦学习。 |
| [^128] | [A Decision-Focused Neural Optimization Framework for Personalized Route Reproduction from Vehicle Trajectories](https://arxiv.org/abs/2610.07857) | 提出了一种决策聚焦神经优化框架，将个性化路径重现建模为基于学习到的驾驶员特定潜在路段成本的最短路径问题，通过感知模型将上下文协变量嵌入个性化成本，并借助约束优化层与隐式极大似然估计实现端到端训练，无需枚举备选路径集合即可重现观测路径。 |
| [^129] | [Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes](https://arxiv.org/abs/2610.07853) | 该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。 |
| [^130] | [Scen-Opt: A Scenario Optimization Toolbox for Data-Driven Convex Programming](https://arxiv.org/abs/2610.07846) | 本文推出了开源工具箱Scen-Opt，它将凸编程与数据样本相结合并基于场景理论提供统计保证，支持数据驱动的线性、二次和半定规划，填补了场景方法框架下用户友好的数据驱动凸优化软件工具的空白。 |
| [^131] | [CHARTER: Auditing Reference Substitution in Hierarchical Compact-Evidence Evaluation for Computational Pathology](https://arxiv.org/abs/2610.07843) | 该论文提出CHARTER这一参考感知的评估章程，用于审计计算病理学中分层紧凑证据评估因替换评估参考而导致保真度测量与策略比较结论发生反转的问题，在其五种子Random-K审计的15项比较中发现4项出现确定的结论反转。 |
| [^132] | [Privileged Context as Drift in On-Policy Self-Distillation](https://arxiv.org/abs/2610.07842) | 本文通过在内容与来源两个维度上系统控制特权上下文的设计，首次分离并量化了特权上下文选择对在线策略自蒸馏中策略漂移的影响，发现改变上下文内容比改变其来源引起更大的KL散度漂移。 |
| [^133] | [Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters](https://arxiv.org/abs/2610.07834) | 提出即插即用的FreshCast框架，通过持续用新观测数据刷新非参数记忆并进行校准，解决了检索增强时间序列预测中记忆陈旧导致检索效用下降的问题。 |
| [^134] | [Forecast Accuracy Is Not Trading Profit: Evolving Small Recurrent Networks for Stock Return Prediction](https://arxiv.org/abs/2610.07825) | 通过神经进化架构搜索得到的小型循环网络在股票收益预测中同时取得了最高的预测准确率和最佳的每日多空策略净收益，证明更低的预测误差并不等于更高的交易利润。 |
| [^135] | [CANDLE: Cortical Null-Space Decomposition for Noninvasive Brain Source Imaging](https://arxiv.org/abs/2610.07824) | 本文提出CANDLE，一种在T1加权MRI导出的源到传感器映射零空间上学习先验的模型，能够在受试者特定的皮层几何结构上实现可泛化的无创脑电生理源成像。 |
| [^136] | [TTNet: Multi-Task Deep Learning for Table Tennis Player Analysis with Smart Racket](https://arxiv.org/abs/2610.07823) | 本文提出TTNet，一种融合CNN、ResNet和自注意力机制的多任务深度学习模型，能够基于智能球拍的六轴传感器数据同时预测乒乓球运动员的性别、持拍手、球龄和技术水平四项属性。 |
| [^137] | [$\alpha$Transfer: Coefficient Transfer for Efficient Model Merging](https://arxiv.org/abs/2610.07819) | 提出αTransfer方法，利用同一模型家族内不同规模模型在合并系数上性能分布的高度一致性，在小代理模型上搜索最优系数后直接迁移到大模型，实现最高20倍加速和70%内存减少。 |
| [^138] | [Do I Need the Cloud? Uncertainty-Aware Step-Level Handoff for Small Language Model Agents](https://arxiv.org/abs/2610.07816) | 提出STEPGATE框架，通过不确定性感知地对本地小模型的每个动作评分，并按需将困难步骤升级到更强的云端模型，从而在大幅降低云端调用比例的同时显著提升智能体的任务成功率。 |
| [^139] | [Stochastic Gradient Descent Ascent is Suboptimal for Nonconvex-PL Min-Max Games](https://arxiv.org/abs/2610.07814) | 该论文首次建立了非凸-PL极小极大博弈中固定时间尺度比双时间尺度SGDA的紧致复杂度下界，证明SGDA本质上次优——其下界与现有上界匹配、与Smoothed-AGDA形成复杂度分离，且当时间尺度比小于o(κ²)时甚至无法找到驻点。 |
| [^140] | [SIFT: Search Intent-to-Filter Transformer for Multi-Task Personalized Filter Ranking at Airbnb](https://arxiv.org/abs/2610.07810) | Airbnb提出基于Transformer的SIFT模型，直接从用户原始行为序列学习统一的偏好表示以替代人工特征工程，通过多任务预测（预订可能性、筛选条件参与度、序数容量阈值）实现个性化筛选条件排序，并兼容布尔型与数值区间型筛选条件。 |
| [^141] | [MASKerade: Token-Routed Mask Experts for Dense-to-MoE Upcycling](https://arxiv.org/abs/2610.07809) | 提出MASKerade方法，将专家定义为冻结预训练FFN上学习得到的二值掩码稀疏子网络，并通过token级路由器选择与组合被掩码的FFN，在不改变原始FFN权重的情况下实现高效且无需重训权重的稠密到MoE模型升级。 |
| [^142] | [Common-Mode Errors Limit Low-Timestep Deep Spiking Q-Networks](https://arxiv.org/abs/2610.07808) | 本研究揭示了低时间步深度脉冲Q网络的性能下降主要源于跨动作共享的共模误差对自举时间差分学习的不成比例损害，并据此提出共模补偿方法以提升低时间步DSQNs的性能。 |
| [^143] | [Adaptive Mean Estimation by In-Context Learning: A Gradient-Flow Analysis](https://arxiv.org/abs/2610.07804) | 该论文通过梯度流分析揭示了先验拟合网络（如TabPFN）如何在分布族未知的均值估计任务中通过上下文学习获得统计自适应性，自动选择与数据分布相匹配的最优估计策略并达到相应的理论收敛速率。 |
| [^144] | [The Geometry of Empowerment](https://arxiv.org/abs/2610.07796) | 本文将赋权最大化与技能学习方法相联系，提出了解释赋权的新几何框架，解答了赋权与结构中心性之间联系的长期开放问题，并揭示了信息几何与奖励几何的区别，为构建可扩展的赋权最大化方法奠定理论基础。 |
| [^145] | [ServeLearnBench: How Well Can Agents Self-Improve from Serving Experience?](https://arxiv.org/abs/2610.07792) | 本文提出 ServeLearnBench 基准及演化环境流式数据集（EESD），用于系统评估大语言模型智能体在隐藏策略持续演变的环境中，能否从交互与反馈中推断、应用并修正潜在环境知识，从而实现自我改进。 |
| [^146] | [Extending Pathwise Gradients to Discrete Random Variables via Finite-Order Relaxation](https://arxiv.org/abs/2610.07786) | 提出一个通用框架，通过有限阶松弛为泊松等常见离散变量构建精确的路径梯度估计器，该估计器保留硬前向采样、无需温度调节、实现简单，且在所有可行解中唯一并最小化权重方差。 |
| [^147] | [Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell](https://arxiv.org/abs/2610.07782) | 本文在三层多智能体LLM推理架构中实测发现，上下文分解可将峰值KV缓存从35.5 MiB降至14.3 MiB，而持久记忆层不仅增加0.368 MiB缓存开销，且在单问题基准上未带来任何可检测的准确率提升，这种零结果源于基准测试的结构性特点。 |
| [^148] | [Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs](https://arxiv.org/abs/2610.07781) | 该研究发现8比特与4比特量化对语言模型智能体工具故障恢复能力的影响并不稳定，比较结论会随提示词和评估目标（如评分任务的选择）而改变方向甚至反转，表明量化效果的结论高度依赖于评估设计。 |
| [^149] | [APEX: Speculate smarter, not deeper](https://arxiv.org/abs/2610.07780) | APEX是一个学习型控制器，通过请求级专家选择和块级草稿深度自适应来优化推测解码，用更聪明的推测取代更深的推测，从而降低大语言模型推理延迟并减少计算浪费。 |
| [^150] | [Towards One-for-All Foundation Model for Attributed Graph Clustering](https://arxiv.org/abs/2610.07778) | 提出OFAG——一个面向属性图聚类的基础模型，仅需一次训练即可直接应用于多样化的属性图，无需针对特定图的训练、微调或超参数搜索。 |
| [^151] | [TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models](https://arxiv.org/abs/2610.07767) | TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。 |
| [^152] | [Trustworthy Method Comparison with AI Judges: Estimation and Design under Order, Batch, and Aggregation Effects](https://arxiv.org/abs/2610.07755) | 该论文提出用马尔可夫广义线性混合模型刻画LLM裁判的评估机制，证明了随机化取平均排名法的一致性条件及威廉姆斯方设计的效率优势，并揭示了组间比较中因模型非线性导致朴素平均可能得出错误结论的问题。 |
| [^153] | [Adversarially Trained Linear Transformers Are Optimal Robust In-Context Learners for Gaussian Mixtures](https://arxiv.org/abs/2610.07754) | 经过跨任务对抗预训练的线性Transformer无需额外训练，即可通过上下文学习将鲁棒性迁移到未见过的任务，并渐近达到高斯混合分类任务的最优鲁棒贝叶斯误差。 |
| [^154] | [High-dimensional online calibration from harmonic weights](https://arxiv.org/abs/2610.07740) | 本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。 |
| [^155] | [Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence](https://arxiv.org/abs/2610.07739) | 该论文提出了BAR框架，首次将成本受限的KG探索、按来源质量区分的可验证证据以及可引用的推理依据三者结合，利用LLM在医疗知识图谱上进行推理，以弥补EHR中缺失的依赖关系并提升出院后风险预测的可靠性。 |
| [^156] | [Nash Social Welfare for Multi Armed Bandits: Trajectory-wise Expected and High Probability Regret](https://arxiv.org/abs/2610.07737) | 该论文提出了一种新的“轨迹级纳什遗憾”度量，通过先对完整奖励样本路径取几何平均再求期望，弥补了现有度量忽略各轮奖励联合分布的缺陷，并借助詹森不等式证明其严格强于原有度量，从而更忠实地体现纳什社会福利的公平性目标。 |
| [^157] | [Learning to Retrieve via Reinforcement Learning in Embedding Space](https://arxiv.org/abs/2610.07731) | 该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。 |
| [^158] | [SanSi: A Looped Typed Decision Model for System 1.5 Thinking](https://arxiv.org/abs/2610.07730) | SanSi提出“系统1.5思维”——通过多次循环复用模型层在不生成文本的情况下修正隐藏状态，将预训练循环语言模型转化为类型化决策模型，在59个数据源的10,027个测试决策上达到72.0%准确率，比同结构非循环模型高出13.5个百分点。 |
| [^159] | [The Model Plants the Trigger: Answer-Side Backdoor Attacks in Multi-Turn Large Language Models](https://arxiv.org/abs/2610.07723) | 本文提出一种新型“答案侧”后门攻击：利用良性首轮提示诱导模型自己生成一个看似无害的词作为触发器，使模型在多轮对话中识别自身生成的触发器并绕过安全拒绝机制，在仅5%投毒率下攻击成功率接近100%，而用户输入始终保持完全干净。 |
| [^160] | [Exact Calibration and Sharp Risk Geometry for Volume-Sampled Ridge Regression](https://arxiv.org/abs/2610.07721) | 本文为体积抽样岭回归建立了精确的惩罚校准理论（当且仅当抽样行数超过目标有效维度时该惩罚存在），并通过严格的扇形不等式刻画了中心化协方差风险的尖锐上界以及所有最大化响应的结构。 |
| [^161] | [RefRoute: Decoupling Conditioning Cost from References via Compact Residual Conditioning and Spatial Routing](https://arxiv.org/abs/2610.07720) | 提出RefRoute框架，通过将低分辨率潜在token与全分辨率残差特征结合的紧凑残差条件化来减少参考token数量，并利用条件路由与注意力路由将参考token与目标区域对齐，从而显著降低多参考图像生成中的条件化与注意力开销。 |
| [^162] | [Stability of Measure-to-Measure Transformers on Sub-Gaussian Data](https://arxiv.org/abs/2610.07717) | 本文从数学上证明了Transformer将次高斯数据映射为次高斯输出且关于1-Wasserstein距离具有Hölder连续性，由此建立了经验近似下的误差传播估计，并揭示了交叉注意力机制均值场类似物的不同正则性与样本复杂度。 |
| [^163] | [Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding](https://arxiv.org/abs/2610.07713) | 提出受神经运动层级结构启发的NHN网络，通过引入生理学归纳偏置学习紧凑的潜在神经运动状态，从而在跨用户、跨会话的sEMG解码中实现鲁棒泛化。 |
| [^164] | [Mathematical Invariant-Enabled Topological Neural Networks for Molecular and Materials Property Prediction](https://arxiv.org/abs/2610.07712) | 本文提出了数学不变量赋能的拓扑神经网络（MITNN），通过整合拓扑学、谱理论、交换代数、微分几何等多个数学领域的多尺度不变量与拓扑神经架构，实现了更全面的分子与材料性质预测，并揭示精选的数学表示与神经架构配对组合的性能优于单个模型及所有组件的简单聚合。 |
| [^165] | [WASD: Wasserstein-based Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2610.07706) | 该论文提出了WASD方法，通过由词元嵌入构建代价矩阵的Wasserstein距离，将词元级别语义信息融入大语言模型的知识蒸馏中，并借助Sinkhorn散度实现高效优化。 |
| [^166] | [Independent Multi-Agent Reinforcement Learning with Counterfactual Semantic-Social World Models](https://arxiv.org/abs/2610.07704) | 该论文提出CASTLE框架，利用反事实动作条件化的语义-社交双世界模型，让完全去中心化的独立多智能体强化学习智能体能够前瞻性地比较候选动作后果，从而解决仅凭标量奖励信号无法判断回报不佳原因的模糊性问题。 |
| [^167] | [Improving Synthetic Data Generation for Argument Mining via Adversarial Reinforcement Learning](https://arxiv.org/abs/2610.07699) | 提出一种对抗性强化学习数据合成框架，通过生成器与判别器的对抗循环联合优化，同时提升论辩挖掘合成数据的结构准确性与多样性。 |
| [^168] | [Adaptive Model Inversion Attacks Generalize a Privacy-Robustness Tradeoff](https://arxiv.org/abs/2610.07677) | 本文揭示了对攻击进行简单自适应改变后，现有隐私防御和标准训练技术的真实训练数据泄露率被低估达1.16至6.59倍，且评估结果受外部分类器特征基础影响，表明标准模型反演攻击评估可能将优化与测量失败误判为隐私保护。 |
| [^169] | [Exact-Solution Volume and Length Generalization in Transformers](https://arxiv.org/abs/2610.07676) | 该论文提出归一化精确解体积（NESV）这一新指标来量化Transformer的长度泛化难度，证明精确解体积随输入长度衰减得越快，长度泛化就越困难，并为FIRST、MAJORITY、INDEX、PARITY四个任务建立了渐近界。 |
| [^170] | [CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency](https://arxiv.org/abs/2610.07668) | CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。 |
| [^171] | [Joint Workflow and Prompt Optimization for User Behavior Simulation](https://arxiv.org/abs/2610.07663) | SWORD框架基于角色化设计，仅依靠一个标量任务指标即可联合优化多智能体工作流拓扑与自然语言提示词，在用户行为模拟任务上显著超越仅优化提示词、仅优化工作流和分阶段优化的基线方法。 |
| [^172] | [MS-ECG-FM: Towards a More Universal Electrocardiogram Foundation Model for Health Monitoring using Multi-source Contrastive Learning](https://arxiv.org/abs/2610.07662) | MS-ECG-FM通过对齐心电图、超声心动图、放射学和出院报告等多种临床记录进行多源对比学习训练，突破了仅依赖解读报告作为监督的局限，在包括少导联配置在内的所有心电图检测任务上全面超越现有方法。 |
| [^173] | [Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size](https://arxiv.org/abs/2610.07655) | 该论文证明了均匀离散扩散模型在估计有效支撑较小的分布时达到极小极大最优，其统计误差由依赖于样本量的有效支撑大小而非环境空间规模决定。 |
| [^174] | [Does On-Policy Distillation for Safety Pose Backdoor Risks?](https://arxiv.org/abs/2610.07654) | 研究揭示了面向安全性的在线策略蒸馏（OPD）中一个被忽视的后门威胁：被植入后门的教师模型可将隐藏恶意行为传播给原本干净的学生模型，仅3%的投毒率即可使攻击成功率达70%，而增加训练轮数和常用的top-k KL方法会进一步加剧该风险。 |
| [^175] | [Towards the Automatic Synthesis of Interpretable Chess Tactics](https://arxiv.org/abs/2610.07640) | 本文提出一种受国际象棋战术启发、由归纳逻辑编程系统PAL所学模式推导而来的符号化子策略模型，通过融入领域知识提升可解释性，并提出散度度量评估方法，其合成的战术组合能给出与人类初学者棋力相当的走法建议。 |
| [^176] | [Learning Explainable Representations of Complex Game-playing Strategies](https://arxiv.org/abs/2610.07638) | 本文提出一种类似人类认知的方法，训练强化学习智能体将学到的游戏策略合成为基于动作序列的可执行程序，从而获得可解释的策略表示，并在国际象棋和网格环境任务中验证了其有效性。 |
| [^177] | [Asymptotic Analysis of Empirical Risk Minimization on Entry-wise i.i.d. Heavy-Tailed Data](https://arxiv.org/abs/2610.07637) | 本文通过引入函数序参量并运用复制方法，首次在比例高维极限下精确刻画了对称α-稳定重尾数据上线性回归经验风险最小化的泛化误差，并建立了重尾普适性定律与相应的标度律。 |
| [^178] | [Complementary Supervised and Self-Supervised Representations for Out-of-Distribution Graph Learning](https://arxiv.org/abs/2610.07628) | 该论文提出Co-Train和Dual-Space Retrieval两个骨干无关的框架，将自监督表示作为监督学习的互补信号，分别在训练和推理阶段进行融合，从而提升图神经网络在分布外节点分类上的泛化能力。 |
| [^179] | [Stateless Language Agents: Scaling Long-Horizon Automated Research](https://arxiv.org/abs/2610.07625) | 提出无状态语言智能体（SLA）框架，通过“有状态搜索、无状态智能体”的原则——由框架统一管理研究状态并为每次调用重建角色化上下文——来解决长时程自动化研究中智能体重放冗长历史、重复劳动和过早停止实验等失败模式。 |
| [^180] | [Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$](https://arxiv.org/abs/2610.07623) | 该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。 |
| [^181] | [Explore, Then Commit: Measurement-Efficient Scientific Law Discovery with Language Models](https://arxiv.org/abs/2610.07620) | 该论文提出了一种“先探索后确定”协议，通过语言模型提出假设、程序化规划器高效收集测量，将科学定律发现所需的测量次数最多减少约5倍，并将误差显著降低一个数量级以上。 |
| [^182] | [AFA-BANDIT: Provably Near-Optimal Online Multi-Feature Classification Under Budget Constraints](https://arxiv.org/abs/2610.07615) | 该论文将预算约束下的在线主动特征获取问题建模为组合式背包老虎机（BwK）问题，并借助基数感知的置信界，首次实现了可证明的近优遗憾上界。 |
| [^183] | [Learning Grasp Targeting from Point Clouds for Log Pile Clearing on a Hydraulic Crane](https://arxiv.org/abs/2610.07613) | 该论文提出了一种从非分割点云中学习抓取点选择、抓取深度与抓爪方向的策略，结合行为克隆与强化学习训练，成功部署于拖车式液压林业起重机上完成原木堆清理，现场试验中放置成功率达93.8%。 |
| [^184] | [Hub for Outliers, Spokes for Inliers: Uniform Latent Space Construction for Dual-Mismatched Semi-Supervised Learning](https://arxiv.org/abs/2610.07610) | 提出轮毂-辐条潜在空间几何结构，让已知类围绕中心轮毂均匀分布、未知类样本被安置于轮毂锚定的低证据区域，以解决半监督学习中类别分布与标签空间的双重失配问题。 |
| [^185] | [Linear Fitness Subspace in Protein Language Models Enables Sample-Efficient Directed Evolution](https://arxiv.org/abs/2610.07607) | 提出线性适应度子空间（LFS）假设并引入子空间引导进化搜索（SGES），通过在蛋白质语言模型突变引起的残基级表示变化中寻找与实验测定相关的紧凑方向集合，使适应度变化从少量标注样本中线性可获取，从而实现样本高效的模型引导定向进化。 |
| [^186] | [A Neural JKO Scheme for Hellinger-Kantorovich Gradient Flows via Monge-Growth Pairs](https://arxiv.org/abs/2610.07602) | 该论文提出了一种基于Monge-生长对的无网格神经JKO格式，在Hellinger-Kantorovich非平衡最优传输几何中于单个变分步骤内联合处理空间再分布与质量产生/损失，为对流-反应-扩散梯度流建立了离散能量耗散、极小元存在性、正性与正则性等理论保证。 |
| [^187] | [Modeling Latent Disturbances for Robust Decision-Making in World Models](https://arxiv.org/abs/2610.07599) | 本文提出将潜在空间扰动建模为对习得潜在动力学的扰动，使其引发悲观但合理的状态转移，从而在世界模型的潜在空间中实现鲁棒决策。 |
| [^188] | [The Robot Is Not Its Description: GaugeBench for Representation Robustness in Morphology-Aware Policies](https://arxiv.org/abs/2610.07597) | 提出 GaugeBench 基准，发现在机器人本体不变、仅将其描述改写为物理等效的约定时，形态感知策略性能从 4030.6 骤降至 51.6，揭示了现有策略对描述约定极度脆弱、并未真正泛化到机器人形态本身。 |
| [^189] | [BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation](https://arxiv.org/abs/2610.07594) | BiGym 2.0是一个针对宇树G1人形机器人20个家庭操作任务的全身控制基准测试平台，实验表明视觉-语言-动作微调方法总体表现最佳，而编码智能体开发的程序优于所有演示驱动的强化学习基线。 |
| [^190] | [Recurrent Looped Transformer](https://arxiv.org/abs/2610.07591) | 提出循环环路Transformer（RLT），通过将层分配给并行因果编码器和循环解码器，使计算路径随序列长度增长而每个token成本保持固定，在状态跟踪和算法泛化任务上大幅超越固定深度的标准Transformer。 |
| [^191] | [Personal-Agent Mediated Recommendation with Cross-Platform User History](https://arxiv.org/abs/2610.07588) | 提出了“个人智能体中介推荐”这一新范式及MediateRec基准，研究个人LLM智能体如何利用用户授权的跨平台历史来调解平台推荐排序，在有益挽救与有害覆盖之间取得平衡。 |
| [^192] | [REViT-v2: Hierarchical Windowed Roto-reflection Equivariant ViT for Equivariant Feature Extraction](https://arxiv.org/abs/2610.07585) | 本文提出了一种基于窗口化群卷积自注意力与分层特征架构的可扩展旋转-反射群等变视觉Transformer（REViT-v2），成功将群等变ViT扩展至数百万参数规模，并能在ImageNet等实际尺寸图像的大型数据集上进行等变特征提取。 |
| [^193] | [Mechanistic Interpretability of Atmospheric Rivers in GraphCast](https://arxiv.org/abs/2610.07583) | 该研究通过对GraphCast训练稀疏自编码器，首次揭示这一AI天气模型内部稳定地计算出大气河流强度（综合水汽输送IVT）作为内部变量，并通过干预实验证实了其因果作用。 |
| [^194] | [CETUS: How Far Do Representations Trained on Earth Transfer to Cassini SAR of Titan?](https://arxiv.org/abs/2610.07576) | 该论文提出CETUS跨域评估基准，系统比较了地球影像预训练的视觉表征（DINOv2、DOFA、CROMA）与经典图像特征在卡西尼号土卫六SAR地形分类任务上的迁移能力，发现预训练编码器整体优于经典特征，但在土卫六数据上继续微调的效果因模型而异。 |
| [^195] | [Two Vectors Replace In-Context Demos: Structured Task Adaptation via Embeddings](https://arxiv.org/abs/2610.07572) | 提出STAVE方法，用两个任务特定向量（读取向量和上下文向量）直接加到现有输入嵌入中来替代上下文示例，避免了示例图像的重复编码开销，实现高效的结构化任务适配。 |
| [^196] | [Complementary Feature Domains: Information Preservation Does Not Imply Predictive-Contribution Preservation](https://arxiv.org/abs/2610.07565) | 该论文提出互补特征域（CFD）理论，证明保持香农信息并不能保证保持预测贡献，并用贡献缺陷量化重编码下上下文贡献的变化，其上界由可达动作集间的行为距离与联盟不相容性之和界定。 |
| [^197] | [Learning a Mixture of GFlowNets](https://arxiv.org/abs/2610.07562) | 提出了一个描述GFlowNets混合体的通用理论框架，将其细分为连续索引（CI）和离散索引（DI）两类：前者通过随机特征扩展与谱移位可证明地提升采样器的表达能力并降低学习不稳定性，后者统一了已有训练方法并支撑了新提出的分层条件化（SC）GFlowNets。 |
| [^198] | [TAFFY: A Task-Adaptive Tabular Foundation Model with In-Context Diversity](https://arxiv.org/abs/2610.07559) | TAFFY通过基于共享因果过程干预生成多样化合成上下文的“上下文内多样性先验”和迭代精炼的任务条件循环Transformer，显著提升了表格基础模型从上下文推断任务特定预测关系的能力。 |
| [^199] | [Seeing the Invisible: Physics-Guided Visual Prompting for Temperature- and Radiation-Aware VLA Navigation](https://arxiv.org/abs/2610.07558) | 提出物理引导视觉提示（PG-VP），将不可见的辐射或温度危险转化为动态虚拟障碍物视觉提示，使冻结的VLA导航模型无需重新训练即可规避多种不可见风险。 |
| [^200] | [Global Transport Couplings for Classifier-Free Guided Flows](https://arxiv.org/abs/2610.07555) | 提出了一种无需类别标签的全局最优传输耦合方法（GT），它虽然在无引导时会降低性能，但与无分类器引导结合后能在不同领域、模型规模和采样预算下持续提升条件生成质量，并据此指出条件流耦合应在引导流下评估。 |
| [^201] | [Which and When to Admit: Gradient Admission for Data-Centric Small Language Model Finetuning](https://arxiv.org/abs/2610.07553) | 提出GRADE框架，通过状态感知选择器持续接纳与演化中的多任务梯度场对齐的样本，并用自校准步级门控在子空间接近饱和时拒绝破坏性更新，从而同时解决LoRA微调中的梯度冲突、静态数据选择和子空间饱和三大问题，提升小语言模型微调效果。 |
| [^202] | [Is $\sqrt{d}$ Separation Necessary for Gradient EM to Learn Gaussian Mixtures in High Dimensions?](https://arxiv.org/abs/2610.07551) | 本文证明在高维学习高斯混合模型时，梯度 EM 全局收敛所需的 $\Omega(\sqrt{d})$ 分离度条件是不可避免的，即这一对维度的依赖性无法被去除。 |
| [^203] | [Foundation Model-Aided Multi-Agent Reinforcement Learning for Wireless Random Access Network Optimization](https://arxiv.org/abs/2610.07550) | 该论文提出一种基础模型辅助的actor-critic多智能体强化学习算法，以显著降低无线随机接入网络优化任务中的训练开销，并证明了其与采用评论者模型交换和线性近似的传统MARL方法具有相同的收敛阶。 |
| [^204] | [Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models](https://arxiv.org/abs/2610.07540) | 该论文发现JEPA世界模型中下一步预测与防坍缩正则化无法保证可控的不稳定模态被保留，并提出通过逆动力学（动作重建）损失增强世界模型训练，以学习能保留不稳定模态的控制感知视觉表示。 |
| [^205] | [SkillFormer: Skill-Decomposed Adaptation for Audio Language Models](https://arxiv.org/abs/2610.07533) | SkillFormer通过将音频理解分解为技能特定的低秩适配器，并利用学习到的路由器在推理时动态组合它们，配合交替式训练方案避免了多任务梯度冲突，以不到4%的参数增量解决了音频语言模型多技能训练中的干扰问题。 |
| [^206] | [Targeted search shows that random-device testing underestimates worst-case error in a simulated wave-based neural operator](https://arxiv.org/abs/2610.07529) | 该研究通过对模拟波基神经算子进行针对性器件搜索，发现搜索到的器件误差可达随机器件测试最大值的1至3倍，证明仅依赖随机器件测试会严重低估实际部署中可能出现的最坏情况误差。 |
| [^207] | [Activation Denoising: A Robustness View on Parallel vs Sequential LLM Quantization](https://arxiv.org/abs/2610.07522) | 提出激活去噪方法，从鲁棒性视角将上游量化误差建模为噪声并加以正则化抑制，使并行量化达到接近串行量化的精度，同时保持完全并行的可扩展性。 |
| [^208] | [Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates](https://arxiv.org/abs/2610.07518) | 该研究发现有害监督微调会在大语言模型的检查点更新中留下连续且可读取的痕迹，通过检查点级别的坐标即可高精度检测有害目标的存在，从而实现无需运行模型的安全审计。 |
| [^209] | [MobileVISTA: Generative Data Augmentation for Pose Generalization in Mobile Manipulation](https://arxiv.org/abs/2610.07511) | 提出 MobileVISTA 数据生成框架，通过联合增强自我中心视觉观测与动作重定向，将单一标准位姿下的演示转化为位姿多样化的训练数据，从而让移动操作策略对机器人位姿变化具备强泛化能力。 |
| [^210] | [Two-Sample Testing for Random Graphs without Vertex Correspondence](https://arxiv.org/abs/2610.07503) | 该论文首次建立了顶点无对应情形下图总体双样本检验的最优样本复杂度理论，证明对于保持度不变的两块结构差异，每组需要约 t^{-3} 张图，带符号三角形计数可达到该最优速率，且未对齐相比对齐情形需多付出 t^{-2} 阶的样本代价。 |
| [^211] | [Source-Learned Reliance for Selective Test-Time Adaptation of Multimodal Time Series](https://arxiv.org/abs/2610.07499) | CARAT通过源域训练预先学习骨干网络特定的依赖度代理，并结合轻量级单类损坏检测器，在部署时实现多模态时间序列的选择性测试时自适应，从而避免跨模态一致性判断的误导和额外推理成本，提升传感器噪声或缺失情况下的系统可靠性。 |
| [^212] | [Does Muon Need Fine-Grained Spectral Shaping?](https://arxiv.org/abs/2610.07497) | 本文提出 BulkBoost 双频段谱重加权框架，表明 Muon 并不需要细粒度的谱整形，只需将奇异谱粗略地划分为噪声主体和高增益尖峰两个频段并进行重加权即可提升优化效果。 |
| [^213] | [Who Bears the Burden? Learning Responsibility for Shared Constraints in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.07491) | 提出LiRA方法，通过优化社会福利来学习各智能体在共享拉格朗日乘子中的责任份额，从而解决多智能体强化学习中共享约束惩罚如何在不同智能体之间合理分配的问题。 |
| [^214] | [Deep Defence on Wheels: A Dual Intrusion Detection System Architecture for Comprehensive In-Vehicle Network Security](https://arxiv.org/abs/2610.07489) | 该论文提出了一种面向资源受限车载平台的双重入侵检测系统框架，其中量化的LSTM模型（QLSTM-IDS）在满足低延迟、低资源开销的部署约束下，对DoS/洪泛、Fuzzing和欺骗/故障等攻击实现了超过99.9%的检测准确率。 |
| [^215] | [Robust Importance Sampling for Rare Events via Constrained Gaussian Mixtures](https://arxiv.org/abs/2610.07485) | 该论文提出一种将稀有事件重要性抽样分解为“覆盖”与“拟合”两个阶段、并对最终高斯混合提议分布施加约束以保证重要性抽样方差有限的框架，从而在多种基线方法之上显著提升了稀有事件概率估计的效率与鲁棒性。 |
| [^216] | [SpecBraM: What Should an EEG Foundation Model Predict? Masked Band-Power Prediction versus Waveform Reconstruction](https://arxiv.org/abs/2610.07484) | 该研究提出掩码频带功率预测（MBP）作为EEG自监督预训练目标，相比波形重建在睡眠分期上性能更优，且预训练目标的选择比tokenizer设计的影响更大。 |
| [^217] | [Adapting to Changes in Agent Behavior via Finite-Depth Policy Sensitivity](https://arxiv.org/abs/2610.07475) | 提出一个有限深度策略敏感度框架，通过可调节截断深度近似计算策略对其他智能体行为变化的一阶敏感度，并证明截断误差界随传播深度递减、在全视野传播时消失。 |
| [^218] | [Structure, Not Belief: Correlated Thompson Sampling from LLM-Derived Covariance in Combinatorial Semi-Bandits](https://arxiv.org/abs/2610.07470) | 提出一种对组合汤普森采样的最小改动方法，仅查询LLM一次将臂划分转化为相关协方差矩阵来引导探索，理论证明相比独立采样可获得有限时域内√(d/K)的遗憾改进，实验中遗憾降低19%。 |
| [^219] | [Efficient Multimodal Inference through Adaptive Acquisition and Sequential Fusion](https://arxiv.org/abs/2610.07466) | 该论文提出SemARC框架，通过序列模态聚合器SeMA与自适应运行时控制器ARC协同工作，在模态编码前自适应地选择模态，仅执行必要的编码与融合分支，从而在保证不同获取顺序下预测一致性的同时实现高效的多模态推理。 |
| [^220] | [Interpretable Hypergraph Learning via Neural Additive Models](https://arxiv.org/abs/2610.07458) | 该论文提出了一种固有可解释的超图学习框架 HGNAN，它将神经加性模型扩展到高阶关系数据，通过特征级非线性分解与超图感知的结构聚合相结合，在实现节点级和超边级任务透明预测的同时保持良好的预测性能。 |
| [^221] | [AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation](https://arxiv.org/abs/2610.07457) | AlignQuant提出了一种以GPU兼容的二维权重瓦片作为精度分配、存储和执行公共单元的训练后混合精度量化方法，使大语言模型的压缩能够真正转化为实际推理加速。 |
| [^222] | [Active Feature Acquisition for Cost-Efficient Temporal Prediction with Reduced Participant Burden](https://arxiv.org/abs/2610.07452) | 该论文提出纵向主动特征获取（LAFA）方法，通过学习一种策略在每个时间点仅选择性地采集最优的条目动态子集，从而在降低参与者负担、减少无应答与流失风险的同时，保持对心理病理结果的准确预测能力。 |
| [^223] | [Fork-and-Flush: Escaping Idea Basins in Autoresearch Agents](https://arxiv.org/abs/2610.07447) | 针对自动科研智能体在同一任务上独立运行时因陷入“想法盆地”而导致分数差距巨大且难以通过追加算力弥合的问题，本文提出周期性的“分叉-冲刷”（fork-and-flush）方法——将智能体分叉为多条继承工作空间但重置对话上下文的并行轨迹，再从中择优继续——有效帮助智能体逃离解空间局部区域并提升最终表现。 |
| [^224] | [Decoupling What from Where: How Should a Small GUI Grounding Model Receive the Action Type?](https://arxiv.org/abs/2610.07444) | 该论文系统比较了向小型GUI定位模型传递动作类型的五种机制，发现辅助损失、可加性嵌入和提示词注入各带来5-7个hit@0.10百分点的显著提升，而硬路由和前置token无效，且相当一部分提升源于对屏幕外触点钳制这一预处理选择的防护而非空间先验。 |
| [^225] | [Artifact removal improves electrodermal waveforms but not downstream classification in a virtual-reality balance task](https://arxiv.org/abs/2610.07438) | 该研究发现在虚拟现实平衡任务中，尽管伪迹去除显著改善了皮电信号的波形质量（伪迹区域误差降低17.8%），但这一改善未能转化为下游分类性能的提升，表明波形层面的优化并不必然带来决策层面的益处。 |
| [^226] | [StaFIR: Convex Learning of Stationarity-Aware Causal Filters](https://arxiv.org/abs/2610.07430) | StaFIR提出一种通过凸优化学习非负指数滞后分布混合的因果FIR滤波器，在经验平稳性与输入保留之间取得平衡，并能根据时间序列的持久性自适应地调整滤波强度。 |
| [^227] | [AccentCL: Robust Accent Classification with Incremental Expansion](https://arxiv.org/abs/2610.07426) | 提出了AccentCL框架，通过不平衡感知损失、领域均值对齐损失和基于回放的持续学习，实现了对类别不平衡和跨语料库领域偏移具有鲁棒性的英语口音分类，并支持新口音类别的增量扩展。 |
| [^228] | [Benchmarking Label-Revealed Online Updates for EEG BCI Decoding](https://arxiv.org/abs/2610.07420) | 该论文提出了EEG脑机接口解码的在线自适应基准测试，系统比较了CSP与黎曼协方差两类方法、受控遗忘及冷启动策略，发现标签揭示式在线更新能在14个模型/数据集对中的13个上带来最高约18%的相对准确率提升。 |
| [^229] | [Learnable Spectral Activations](https://arxiv.org/abs/2610.07419) | 提出可学习谱激活（LSA），用谐波振幅可学习的残差截断傅里叶级数替代固定激活函数，将特征选择与频谱整形解耦为两条独立梯度更新的通路，从而提升隐式神经表示对局部化和空间变化信号的多谐波组合能力。 |
| [^230] | [Bayesian Optimization on Function Spaces via Sparse RKHS Manifolds](https://arxiv.org/abs/2610.07417) | 提出L0MO方法，通过在RKHS中由核函数稀疏表示构成的流形子集上搜索，并同时优化核的位置与系数，从而实现函数空间中的贝叶斯优化，并为现有FBO方法提供了统一视角。 |
| [^231] | [Evaluation of Active Feature Acquisition Policies with Tabular Foundation Models](https://arxiv.org/abs/2610.07406) | 本文发现在离线数据覆盖不均衡时，使用总预测熵作为奖励会混淆认知不确定性与偶然不确定性并产生偏差，因此提出用PFN输出的后验期望偶然熵来评估主动特征获取策略，从而更准确地识别信息量大的特征。 |
| [^232] | [What pass@k Cannot Measure: Evaluating Diversity and Capability Retention after Post-Training](https://arxiv.org/abs/2610.07405) | 论文指出 pass@k 只反映解题概率而忽略输出分布的多样性，并通过实验证明 GRPO 与 RFT 两种后训练方法虽在 pass@k 上表现相似，却使多样性指标朝相反方向变化，说明仅用 pass@k 评估后训练效果是不充分的。 |
| [^233] | [Fed-BRDECS: Privacy-Preserving and Heterogeneity-Aware Federated Deep Embedded Clustering](https://arxiv.org/abs/2610.07399) | Fed-BRDECS通过用局部可计算的样本稳定性损失替代依赖全局软分配统计的聚类目标，并结合预测均衡采样与质心级重启机制，实现了隐私保护且能应对客户端数据异构性的联邦深度嵌入聚类。 |
| [^234] | [A perspective note on likelihood approximation and inference for complex simulation models using a chain of aggregated normalizing flows](https://arxiv.org/abs/2610.07391) | 提出了一种基于n级聚合标准化流链的似然近似新方法，通过按顺序估计各组双射变换参数，为复杂仿真模型的大规模数据分析、假设检验和不确定性量化提供了可扩展且高效的解决方案。 |
| [^235] | [Inference and learning in sparse autoencoders as natural gradient flow](https://arxiv.org/abs/2610.07389) | 该论文将稀疏自编码器的推理与字典学习统一为共享变分自由能上的自然梯度流，并提出无编码器的稀疏编码模型BeFOND，通过循环解释抵消机制减少重叠特征间的干扰、利用Fisher预条件化加速稀有特征学习，从而显著提升字典恢复与稀有特征检测能力。 |
| [^236] | [DeepAJM: Deep Association Joint Model for Irregularly Sampled data](https://arxiv.org/abs/2610.07388) | 提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。 |
| [^237] | [HyperNSDE: Personalized Neural SDEs for Joint Static-Longitudinal Clinical Data Generation](https://arxiv.org/abs/2610.07383) | HyperNSDE通过超网络将静态患者特征注入潜在神经SDE，首次联合生成异构静态协变量、不规则采样的纵向轨迹和观测时间三类紧密耦合的临床数据，为医疗AI提供真实且保护隐私的合成患者数据。 |
| [^238] | [Multigroup Fairness and Omniprediction: Separations and Equivalences](https://arxiv.org/abs/2610.07374) | 该论文探究全知预测是否必须依赖多重准确性、多重校准等多群体公平性概念，并研究两者之间的分离与等价关系。 |
| [^239] | [Identity-Conditioned Score Fusion for Open-Set Person Re-Identification](https://arxiv.org/abs/2610.07366) | 提出一种无需训练的身份条件化分数融合框架，通过为每个图库身份定制融合权重并结合查询条件自适应，在换装行人重识别基准上将误非识别率最多降低8.8%。 |
| [^240] | [Dynamic Budget Allocation for LLM Evaluation under Hard Resource Constraints](https://arxiv.org/abs/2610.07362) | 提出HARP方法，在硬资源约束下为大语言模型多轮交互评估实现动态预算分配，通过自适应重新分配未使用预算来构建时间-事件的下预测边界。 |
| [^241] | [Towards Explainable Benchmarking for Data-driven Post-Wildfire Debris Flow Prediction](https://arxiv.org/abs/2610.07358) | 本文提出了一个统一的可解释数据驱动火灾后泥石流预测基准，解决了现有研究在特征空间、模型架构和评估协议上的碎片化问题，并可系统分析气象、地形、土壤和燃烧严重程度等异质性因素对触发泥石流的相对重要性。 |
| [^242] | [RELACE: retrospective likelihood-based action credit estimation for long-horizon language agents](https://arxiv.org/abs/2610.07349) | RELACE 提出了一种无 critic 的信用估计框架，通过比较动作在原始上下文与结果增强上下文下的教师强制似然差异，生成轨迹归一化的回溯因子，从而为长程语言智能体提供更精确的动作级信用分配。 |
| [^243] | [Stepped MoE: Segment-Level Routing with Configurable Inference Complexity](https://arxiv.org/abs/2610.07348) | 本文提出阶梯式MoE统一框架，将弹性结构与稀疏门控架构相结合，通过分段级路由使模型能够同时适应不同的部署约束和任务需求，实现推理时对精度-效率权衡的细粒度控制。 |
| [^244] | [Evaluating Behavioral Context for Interpretable IAM Policy Risk Scoring in Cloud Environments](https://arxiv.org/abs/2610.07345) | 本文提出AWS IAM Context Bench基准，通过可解释提升机模型验证行为上下文信息能否在策略与有效授权信息之外，为云环境中IAM策略的风险优先级排序带来可衡量的增量价值。 |
| [^245] | [CausalBind: Causal Modeling and Learning for Protein-Molecule Virtual Screening](https://arxiv.org/abs/2610.07340) | 该论文提出CausalBind，通过因果建模识别并利用蛋白质-分子结合中稀疏的跨模态局部相互作用模式（如氢键、疏水接触、盐桥），从而克服传统密集整体对齐方法的局限，提升虚拟筛选向新靶点泛化的能力。 |
| [^246] | [Logbook: Extremely Long-form Audio Event Understanding](https://arxiv.org/abs/2610.07338) | 该论文提出了面向小时级至六天超长音频的事件理解基准 Logbook，要求系统对连续音频进行无缝隙分割并为每段生成事件标签与描述，发现最佳系统仍不及人类、过度分割普遍存在，且端到端系统通常优于级联系统但性能随上下文变长而下降。 |
| [^247] | [Selective Critique for Cost-Aware LLM Agents in Long-Horizon Decision Making](https://arxiv.org/abs/2610.07335) | 提出SAG框架，利用基于动作歧义信号（全局熵与top-2边际）的轻量级免训练门控机制，智能地选择在何时调用外部批判，从而在提升LLM智能体长时程决策可靠性的同时大幅降低token消耗和延迟。 |
| [^248] | [Weight Oracles: Reading Neural Network Weights with Language Models](https://arxiv.org/abs/2610.07334) | 提出了“权重预言机”——通过直接读取神经网络原始权重（而非行为测试）来诊断网络属性（如后门）的微调语言模型，可从权重模拟前向传播并实现零样本安全审计。 |
| [^249] | [Structuring MoE Expert Selection for Agentic Reinforcement Learning](https://arxiv.org/abs/2610.07332) | 该论文发现MoE专家选择与智能体轨迹存在天然的结构对齐（语义相似操作的轮次共享更多路由专家），并提出分层路由控制框架将这一结构约束纳入智能体强化学习训练，从而同时提升任务性能与推理效率。 |
| [^250] | [Advantage of Entangled Learning Rules in Quantum Measurement Class Learning](https://arxiv.org/abs/2610.07328) | 该论文证明了在量子测量类PAC学习框架中，采用无法通过局域操作和经典通信（LOCC）实现的纠缠学习规则，相比传统的单副本学习规则具有优势。 |
| [^251] | [Scale-Invariant Training for Time Series Foundation Models](https://arxiv.org/abs/2610.07324) | 论文揭示了对仿射缩放（如ReVIN）进行逆变换会使各序列梯度被乘以b^p、使序列尺度成为隐含的重要性权重并导致高尺度序列主导训练的“尺度污染”问题，并证明直接在缩放后的目标上计算损失即可实现尺度不变的训练。 |
| [^252] | [ATLAS-AL: Adaptive Trust-Region for Latent Adversarial Searches via Active Learning](https://arxiv.org/abs/2610.07323) | ATLAS提出了一种基于主动学习的查询式攻击生成框架，通过将对抗攻击发现转化为水平集估计问题，结合自适应信任域与局部-全局采样策略，高效挖掘黑盒模型的对抗性输入集合，从而更全面地评估模型鲁棒性。 |
| [^253] | [Assumption-lean logistic regression with missing covariates](https://arxiv.org/abs/2610.07292) | 该论文提出了一种在协变量分布未知（仅有界）的少假设设定下处理协变量缺失逻辑回归参数估计的随机近似方法，克服了传统方法在协变量分布未知时可能严重失效的问题。 |
| [^254] | [A Single-Loop, Constant-Batch First-Order Penalty Method for Stochastic Bilevel Optimization](https://arxiv.org/abs/2610.07290) | 提出SICO方法，一种单循环、常数批次的一阶惩罚方法，通过每次迭代对原始下层问题和惩罚问题各执行一次随机梯度更新并结合投影控制，从而在随机非凸-强凸双层优化中摆脱了对嵌套循环和大批次规模的需求。 |
| [^255] | [FlexiFlow: Bandit-based Model Switching in ML Workflows](https://arxiv.org/abs/2610.07286) | FlexiFlow是一个基于多臂老虎机的动态模型切换数据流系统，综合考虑模型准确率、运行时间和断言通过概率，在当前模型表现不佳时自动切换到更优模型，可将机器学习工作流准确性提升高达23%。 |
| [^256] | [Lock-in EP: An In-Situ Training Algorithm for Oscillatory Hardware](https://arxiv.org/abs/2610.07283) | 提出锁相平衡传播（LIEP）训练算法，无需独立的前向和后向扫描即可为振荡网络组件提供局部梯度信息，使模拟振荡硬件能够进行原位训练和性能恢复。 |
| [^257] | [Algorithmically Aligned Neural Agglomerative Tree Construction](https://arxiv.org/abs/2610.07271) | 提出NN-linkage模型，通过将神经网络与Lance-Williams递推公式算法对齐，使其能够学习任务特定的层次聚类合并规则，同时保留经典链接算法的高效推理和规模泛化能力。 |
| [^258] | [What Words Keep of a Place: Zero-Shot Language Reasoning for Cross-View Geo-Localization](https://arxiv.org/abs/2610.07269) | 该论文提出一种完全无需训练的零样本方法，利用多模态大语言模型将地面全景图与卫星图块转化为结构化文本描述并通过语言比较实现跨视角地理定位，发现语言描述虽然忠实但缺乏区分能力。 |
| [^259] | [Lineage-Aware Memory Governance: A Derivation-Gated Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI Agents](https://arxiv.org/abs/2610.07258) | 该论文提出分析内存单元（AMU），通过为每个缓存结果附加完整的派生谱系图并实施列级权限门控检索，从设计上保证企业AI智能体不会命中由请求者无权限的敏感列派生而来的缓存结果，同时解决部门间同名KPI计算逻辑冲突的问题。 |
| [^260] | [Neural Algorithmic Reasoning for Graph Saddle Point Problems](https://arxiv.org/abs/2610.07255) | 该论文提出基于 Chambolle-Pock PDHG 方法的消息传递框架 GraphPDHG，将神经算法推理用于求解图鞍点问题，既能模拟 PDHG 高效求解并学习到加速算法，还展现出比未对齐 GNN 基线更强的规模泛化能力。 |
| [^261] | [Neural Fields Encode Adaptation Geometry](https://arxiv.org/abs/2610.07253) | 论文提出“适应几何”概念，证明拟合后的神经场权重不仅由重构质量决定，还编码了其适应新观测的难易程度（可由局部线性模型精确预测）以及对历史观测信息的保留。 |
| [^262] | [Learning What to Distill: Bilevel Top-K Token Selection for Self-Distillation in Large Language Models](https://arxiv.org/abs/2610.07247) | 提出了BiToK-SD方法，利用双层优化自适应地学习在大语言模型在策略自蒸馏中应选择哪些词元进行蒸馏，从而克服了均匀蒸馏和固定启发式选择准则的局限。 |
| [^263] | [Hybrid Cross-Modal Attention Network for Early Breast Cancer Detection in Low-Resource Clinical Settings](https://arxiv.org/abs/2610.07243) | 提出混合跨模态注意力网络HCMAN，基于Transformer跨模态注意力机制融合乳腺X光影像与结构化临床数据，在埃塞俄比亚四家医院的本地数据集上实现了97.8%准确率的早期乳腺癌检测，适用于低资源医疗环境。 |
| [^264] | [Benchmarking Time Series Foundation Models for Load Forecasting Under Covariate Uncertainty](https://arxiv.org/abs/2610.07232) | 该论文在三个真实负荷预测数据集上，针对未来协变量信息可用性与质量不同的多种运行场景，对四个从零训练模型和四个时间序列基础模型进行了系统基准测试，发现Chronos-2在协变量可用或预测准确时性能最优，而TimesNet在协变量预测噪声严重时更加鲁棒。 |
| [^265] | [Conditional Flow Matching for Transport Between Markov Processes](https://arxiv.org/abs/2610.07229) | 本文提出一种保持马尔可夫结构的条件流匹配算法，用于学习马尔可夫过程从源轨迹分布到目标轨迹分布的传输映射，并证明了总体一致性、有限样本误差界以及与混合时间相关的样本复杂度下界。 |
| [^266] | [How Inefficient Is Natural Gradient Descent? From Exact Optimality to \Theta ( \sqrt{ \log d } ) Divergence](https://arxiv.org/abs/2610.07228) | 本文提出“低效比”来量化自然梯度下降偏离最短 Fisher–Rao 路径的程度，并证明该比值在三类情形中变化：二次势或一维族精确最优（R=1）、有界偏度族具有与维度无关的界、而尺度族乘积（如高斯协方差和 Gamma 率）的低效比随维度以 Θ(√log d) 增长。 |
| [^267] | [Minimal Witness Reinforcement Learning](https://arxiv.org/abs/2610.07226) | 本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。 |
| [^268] | [Data, Numbers, and Geometry: Three Tutorials on Numerical Methods, Machine Learning, and Evaluation](https://arxiv.org/abs/2610.07220) | 本文为数学研究提供了三个实用教程，分别涵盖基于微分形式逐点取值的外微分数值方法、利用数学结构（如椭圆曲线与箭图）指导神经网络设计并用区间算术验证残差界，以及计算结果的评估与展示。 |
| [^269] | [Constant-Curvature Sliced Gromov-Wasserstein for Heterogeneous Cross-Curvature Alignment](https://arxiv.org/abs/2610.07218) | 本文提出了常曲率切片Gromov-Wasserstein（CCSGW），一种用于对齐混合曲率异构空间（如双曲和球面空间）上概率分布的新型散度，填补了跨曲率分布比较问题的空白并提升了跨空间几何一致性。 |
| [^270] | [Reward-Driven Learning under Prompt-Level Differential Privacy](https://arxiv.org/abs/2610.07212) | 提出了首个针对可验证奖励强化学习（RLVR）训练的差分隐私保证，通过以单个提示为单位聚合梯度、一次性裁剪并添加高斯噪声，使隐私预算与响应数量及裁剪范数无关。 |
| [^271] | [An overview of machine learning-enhanced iterative methods for systems of linear and nonlinear equations](https://arxiv.org/abs/2610.07211) | 本文综述了机器学习增强的线性和非线性方程组迭代求解方法，重点探讨了如何利用机器学习技术克服传统迭代求解器（如牛顿法）在收敛性方面面临的挑战。 |
| [^272] | [Can LLM-assisted regularization increase forecast accuracy for migration flows in low data regimes?](https://arxiv.org/abs/2610.07208) | 本研究提出利用LLM从新闻中提取移民推拉信号，并通过特征特定的正则化惩罚将其融入加权Lasso预测框架，以在低数据环境下提升移民流预测精度，实验显示各移民走廊间效果不一。 |
| [^273] | [Distributionally Robust Mixture-of-Experts Training](https://arxiv.org/abs/2610.07207) | 提出 DRMoET 分布鲁棒训练目标，将各层专家视为内生鲁棒性分组、优化高损失路由结果而非仅均衡负载，在多个模型规模下均提升了 MoE 的下游性能。 |
| [^274] | [Exact Unlearning via Quantized Sufficient Statistics](https://arxiv.org/abs/2610.07197) | 该论文提出量化充分统计量（QSS）框架，将冻结的模式结构与可加性分解的存储内容分离，使数据删除从昂贵的重训练优化转变为精确的减法运算，并区分了仅删除标签（QSS-L）和同时删除输入与标签（QSS-E）两种精确遗忘保证。 |
| [^275] | [Learning Disentangled Representations with Quantum Variational Autoencoders](https://arxiv.org/abs/2610.07196) | 该论文研究了量子变分自编码器（QVAE）能否学习解耦且可解释的潜在因子表示，以推动量子表示学习在复杂科学系统数据解释与可控生成中的应用。 |
| [^276] | [Learning Scientific Exploration from Human Research Decision Trajectories](https://arxiv.org/abs/2610.07184) | 本文提出ResearchTrails数据集，以Git仓库的提交历史作为人类科研探索过程的代理，并开发自动化流水线从中提取结构化的研究决策轨迹，弥补了现有科学语料库只记录最终成果、缺乏探索过程信息的不足。 |
| [^277] | [CLM-as-a-Judge: Evaluating an Open Contrastive Decision Model on Public Judge Benchmarks](https://arxiv.org/abs/2610.07177) | 该论文首次系统评估了开放对比决策模型 CLM-v0.1-8B 作为裁判的能力，发现其在公开基准上接近随机水平且显著落后于同规模奖励模型和生成式裁判，但通过单参数温度校准可将其置信度修复至良好校准状态。 |
| [^278] | [A theory of platonic representations in language models](https://arxiv.org/abs/2610.07168) | 本文通过假设数据具有隐藏的层级结构（抽象层次跨语言共享、表面层次为语言或模态特定），并借助概率上下文无关文法与信念传播理论推导出分析性预测，首次从理论上解释了多语言模型中间层出现柏拉图式表示的现象及其随语言相近程度和模型质量增强的规律。 |
| [^279] | [Interleaved Projected Gradient Descent for Safe Imitation Learning](https://arxiv.org/abs/2610.07167) | 该论文提出一种将标准模仿学习梯度步骤与将网络动作投影到安全集的安全步骤交替进行的训练方法，使神经网络控制器在运行时无需安全过滤器即可渐近满足状态与输入约束。 |
| [^280] | [Adversarial Training for Deep Hedging in Nonstationary Markets](https://arxiv.org/abs/2610.07162) | 提出WRAP框架，一种基于双预算分布鲁棒优化的漂移感知对抗训练方法，通过φ-散度轨迹重加权与最优传输路径扰动来提升非平稳市场中深度对冲策略对未来市场状况的鲁棒性。 |
| [^281] | [CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets](https://arxiv.org/abs/2610.07132) | 该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。 |
| [^282] | [The Implicit Bias of Hyperbolic Representation Learning for Multiclass Data: A Busemann Risk Perspective](https://arxiv.org/abs/2610.07131) | 该论文从 Busemann 风险视角刻画了双曲空间中固定原型多类分类的黎曼梯度流的隐式偏差，证明了由漂移系数符号决定的径向二分性，以及边界方向向 Busemann 风险临界点的收敛。 |
| [^283] | [Is this machine playing?](https://arxiv.org/abs/2610.07130) | 研究者将无任何任务、奖励或活动指令的具身AI置于未知虚拟世界中，观察到其自发产生攀爬、堆叠、绘画等符合玩耍经典判据的行为，并据此提出玩耍或可成为机器自主发展的一种新模式。 |
| [^284] | [Jailbreaking Open-Weight LLMs via Random Embedding Perturbations](https://arxiv.org/abs/2610.07125) | 该论文提出PEV攻击方法，仅需在提示的嵌入向量中反复添加随机高斯噪声即可越狱多种规模的开放权重大语言模型，暴露了此类模型的安全脆弱性。 |
| [^285] | [SoloQ: Calibration-Free Quantization for Diffusion Language Models](https://arxiv.org/abs/2610.07121) | 提出无需校准数据的量化框架SoloQ，通过将权重和激活映射到具有可预测边缘分布的归一化旋转基中，解决了扩散语言模型因激活分布随掩码状态和去噪步骤变化而难以训练后量化的问题。 |
| [^286] | [AMBER: Training Long-Horizon Web Agents through Append-Only Memory](https://arxiv.org/abs/2610.07118) | 提出AMBER方法，利用追加式记忆训练长时程网络智能体，从而解决覆写式记忆在稀疏结果奖励下难以学会跨多次重写保留事实信息的问题。 |
| [^287] | [Will the Judge Flip? Predicting Position-Sensitive LLM Judgments from Residual Stream Activations](https://arxiv.org/abs/2610.07115) | 本研究提出用线性探针读取LLM裁判判定前的残差流激活，无需按两种顺序重复判定即可预测其是否会因候选回答顺序而翻转结论，且跨基准迁移效果良好、无需重新校准。 |
| [^288] | [Sample-Optimal Estimation of the Fr\'echet Inception Distance](https://arxiv.org/abs/2610.07114) | 该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。 |
| [^289] | [LiLib: Lifelong Air-to-Ground Path-Loss Prediction on UAVs via a Drift-Triggered Model Library](https://arxiv.org/abs/2610.07111) | 提出LiLib轻量级持续学习方案，无人机通过维护递归最小二乘专家库，在检测到环境漂移时复用或创建专家，将空地路径损耗预测RMSE从5.89 dB降至4.03 dB，并大幅降低重访已知环境后的误差。 |
| [^290] | [Muon Is Theoretically Wrong For Convolutions, But Empirically Effective](https://arxiv.org/abs/2610.07103) | 该研究指出将卷积核重塑为矩阵的标准Muon实现在理论上有缺陷，作者提出了理论上更严谨的卷积Newton-Schulz方法（Conv-NS），但实验发现两者性能相当，揭示了优化器理论与实践之间的差异。 |
| [^291] | [Evaluating Inference Compute for Generative AI: A Framework for Enterprise Workloads](https://arxiv.org/abs/2610.07094) | 该论文提出了一个面向企业工作负载的生成式AI推理计算评估框架，揭示了智能体轨迹使每token解码延迟成为性能主导因素，从而有利于片上SRAM加速器和分离式预填充/解码架构，且每步可靠性随轨迹长度呈指数级复合。 |
| [^292] | [Towards a Unified Misuse Monitoring Benchmark](https://arxiv.org/abs/2610.07089) | 该论文提出了一个统一的轨迹级滥用监控形式化框架，并构建了包含约6,200份对话记录的基准，首次将分解攻击与提示注入攻击纳入同一评估体系，以“危害窗口”为标准衡量监控器何时能及时识别有害行为。 |
| [^293] | [SchemaFill: Efficient LLM Tool Calling via Slot-Parallel Speculative Decoding](https://arxiv.org/abs/2610.07086) | SchemaFill提出了一种槽位并行投机解码框架，通过并发生成工具调用中未来槽位值的候选来加速大语言模型的工具调用，且无需预先获知实际的调用序列或参数值。 |
| [^294] | [A Query Is Not a Commitment: Learning to Correct Expert Answers in Online Deferral](https://arxiv.org/abs/2610.07084) | 提出ORUCB算法，利用累积响应学习误差的界来校准置信度加权风险回归与探索，在在线推迟学习中对不准确专家的答案进行纠正，实现了 $O(\sqrt T\log(T+1))$ 的高概率伪遗憾界。 |
| [^295] | [When Attention Does Not Explain the Peak: Temporal Reference vs. Forecast Output in Attention-Based Time-Series Forecasting](https://arxiv.org/abs/2610.07080) | 该论文通过巴拿马负荷数据集上的实证分析发现，基于注意力机制的时间序列预测模型中，预测输出的峰值时刻误差为0小时而注意力权重argmax的峰值时刻误差达5小时，证明注意力图并不能作为预测峰值时刻的时间解释依据。 |
| [^296] | [Few-Shot Bioactivity Prediction with Meta-Learning under Assay Heterogeneity](https://arxiv.org/abs/2610.07079) | 本文揭示了检测异质性会降低元学习在小样本生物活性预测中的性能，并提出了MetaHeta框架，通过将对大型辅助检测数据的线性注意力与对稀缺任务上下文的精确注意力相结合，有效解决该问题。 |
| [^297] | [What Must Replay Preserve? Separating Correctable Bias from Class Correspondence](https://arxiv.org/abs/2610.07077) | 该论文提出一个将logit重放中缓存预测视为时间异构监督的诊断框架，发现在CIFAR-100的DER++上，后学习类别的未更新存储分数可用固定常数替代而几乎不损失准确率，说明重放的关键价值在于类别对应关系而非分数本身。 |
| [^298] | [Learning Decision-Stump Thresholds in Context: Dynamics of Softmax Attention](https://arxiv.org/abs/2610.07074) | 本文证明了两参数softmax注意力模型通过基于梯度的预训练能够学习决策阈值估计，其误差为$\widetilde O((m\wedge n)^{-1}+N^{-1})$，并揭示了背后的机制是参数协调发散——注意力尺度以$t^{1/4}$增长、阈值误差以$t^{-1/4}$衰减。 |
| [^299] | [On Color Alignment in VAE Latent Spaces and Its Applications](https://arxiv.org/abs/2610.07072) | 该论文发现文本到图像模型的VAE潜空间普遍共享一个与亮度轴和对抗色轴对齐的颜色子空间，并据此提出了以ColorTuning为代表的精确颜色控制等三项应用。 |
| [^300] | [Learning to Simulate Individuals from Macro Social Signals](https://arxiv.org/abs/2610.07062) | 该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。 |
| [^301] | [ImpactMat: Continuous Material Estimation for Inverse Impact Sound Rendering](https://arxiv.org/abs/2610.07061) | 该论文提出了ImpactMat数据集与基准，以及一个前馈模型，能够从冲击声音录音中连续估计材质参数，实现逆冲击声音渲染，从而突破传统渲染器固定材质预设的限制。 |
| [^302] | [Skillful Data-Driven Subseasonal Soil Moisture Forecasting: Prospects and Limits for Flash Drought Prediction](https://arxiv.org/abs/2610.07060) | 该研究提出基于Vision Transformer的双通路时空注意力架构，通过残差学习并以物理单位而非标准化距平作为预测目标，实现了对欧洲次季节根区土壤湿度的高技巧概率预测，为骤旱早期预警提供了新途径。 |
| [^303] | [sHAIL-Causal: A Sequential Staircase Procedure for Invariant Causal Predictor Discovery](https://arxiv.org/abs/2610.07057) | 本文提出 sHAIL-Causal，一种以拟合优度饱和与跨环境不变性联合判据为门控的序列阶梯式学习程序，可避免被混杂预测因子诱导，并在 Richness 条件下可证明地停在真正的因果预测因子集合上，而仅依赖复杂度控制或朴素贪心搜索的方法均无法做到。 |
| [^304] | [Behavioral Cloning Mystery](https://arxiv.org/abs/2610.07056) | 提出OCBench机器人操作基准，通过模拟人类示范关键特性的可控脚本化策略，首次在受控环境中系统复现了行为克隆的多种反直觉现象。 |
| [^305] | [Data Fusion for Errors-in-Variables](https://arxiv.org/abs/2610.07048) | 本文提出了一种数据融合估计方法，通过条件可迁移性假设利用外部研究的重复测量来识别目标研究中的条件测量误差分布，从而在源-目标异质性下解决变量含误差问题。 |
| [^306] | [TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning](https://arxiv.org/abs/2610.07043) | 提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。 |
| [^307] | [The Premise Is the Problem: Exchangeability Failure in Self-Monitored Test-Time Adaptation](https://arxiv.org/abs/2610.07038) | 论文证明在自监控的测试时自适应中，由于监控与自适应共用同一反馈，预测目标重叠与误差依赖性会破坏可交换性假设，导致虚假警报、预测质量下降，且自适应会掩盖持续变化，而冻结的原始模型信号反而更清晰。 |
| [^308] | [Inference-Time Projection for Physically Valid Biomolecular Diffusion Models](https://arxiv.org/abs/2610.07037) | 该论文提出将物理有效性视为约束推理问题，在推理时对扩散模型的去噪坐标估计施加两个闭式投影算子（如链间范德华投影），以极低的额外开销确保生物分子复合物预测的物理有效性，避免了物理势能引导的高计算成本或模型微调的架构耦合。 |
| [^309] | [JIVEAdapter: A Multi-Task Additive Low-Rank Adapter via Joint and Individual Variation Explained (JIVE)](https://arxiv.org/abs/2610.07036) | JIVEAdapter借鉴统计学中的JIVE方法，将多任务低秩适配器的权重更新分解为跨任务共享的联合结构与近似正交的任务特定个体结构，并自适应分配秩，使冻结后的联合结构可作为先验直接复用于新任务，实现高效且可解释的多任务参数微调。 |
| [^310] | [Shaping the Wind: Nested Potentials for Kinematically Admissible Urban Wind Prediction](https://arxiv.org/abs/2610.07033) | 提出Sculpt嵌套势框架，通过结构化的输出表示使城市风场神经代理模型的预测天然满足局部质量守恒与壁面不可渗透性的运动学可容许约束。 |
| [^311] | [Investigating Model Compression for Neural Machine Translation in the Biomedical Domain](https://arxiv.org/abs/2610.07032) | 本研究探讨了知识蒸馏和量化两种模型压缩技术在生物医学领域神经机器翻译中的应用，揭示了这两种技术在低资源专业领域条件下的局限性。 |
| [^312] | [Identifiable World Models from Pretrained Diffusion Representations](https://arxiv.org/abs/2610.07028) | 提出ConDA方法，通过在冻结的预训练扩散模型潜在表示之上仅学习一个轻量级对齐映射，在无需重新训练生成骨干网络的情况下，利用非线性ICA保证实现可辨识的世界模型，能够辨识潜在动态状态并保留其结构因果模型。 |
| [^313] | [Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release](https://arxiv.org/abs/2610.07019) | 该论文发布了一个仅使用许可证允许复用的文章微调的 40 亿参数开放模型 Fiorillo v0.5，它能够以良好校准的概率回答随机试验中干预措施对结局影响的问题，并通过预注册的四项标准验证后正式发布。 |
| [^314] | [Which Image Property Carries the Jailbreak? A Controlled Dissection of Image-to-Text Jailbreaks](https://arxiv.org/abs/2610.07009) | 本文通过控制变量的受控实验系统解剖图像到文本越狱攻击，发现真正驱动越狱成功的是攻击图像本身，而块的熵、JPEG大小等密度特征以及块数量结构均无法可靠区分攻击图像与良性图像。 |
| [^315] | [STOCK-JEPA: Prior-Anchored Latent Revision Representation Learning in Equity Markets](https://arxiv.org/abs/2610.07006) | Stock-JEPA通过将经典低复杂度金融模型产生的收益风险统计量作为先验锚定到潜空间中，并学习相对该先验的可预测增量修正，在低信噪比的股票市场中同时兼顾了可解释性与非线性模式的捕捉能力。 |
| [^316] | [Where Does the Audio Jailbreak Live? A Controlled Frequency-Depth Audit of AdvWave-P on Qwen2-Audio](https://arxiv.org/abs/2610.07005) | 该论文对AdvWave-P音频越狱扰动在Qwen2-Audio上进行受控的STFT频带掩蔽审计，发现攻击成功率的频率排序依赖于频带划分方式，且掩蔽7520-7960 Hz这一窄频带可将攻击成功率降至0.10。 |
| [^317] | [Should We Skip Diffusion?](https://arxiv.org/abs/2610.07002) | 提出DDT-RFE，通过移除编码器中自注意力和MLP周围的残差连接以促进渐进式抽象，并将patch嵌入与中间及最终编码器特征融合，从而在保持训练稳定的同时改善扩散模型的去噪效果。 |
| [^318] | [Mask-Guided KV Cache Eviction in Block Diffusion Language Models](https://arxiv.org/abs/2610.06996) | 提出无需训练的MaskAhead方法，通过统一的掩码-查询排序机制同时解决分块扩散语言模型中KV缓存的选择与淘汰问题，其量化变体Q-MaskAhead可在低比特KV上直接计算，从而降低内存占用并加速生成。 |
| [^319] | [DART-ES: Difficulty-Aware Reweighting and Targeted Replay for Fine-Tuning LLMs with Evolution Strategies](https://arxiv.org/abs/2610.06993) | 提出 DART-ES 方法，通过从扰动种群通过率构建动态难度状态，同时实现难度感知的奖励重加权与罕见可解样本的定向回放，在不引入额外难度模型或反向传播的前提下提升了进化策略微调大语言模型的效果。 |
| [^320] | [Repair Lot Skyline: A Weighted Constraint Satisfaction Approach to Pavement Repair Optimization from Geospatial Hazard Density](https://arxiv.org/abs/2610.06989) | 该论文提出“修复批次天际线”方法，将路面维修计划建模为定义在里程桩号上的加权约束满足问题，仅凭单期病害调查即可生成覆盖全部坑洞、并将维修总长缩减约16%的成本优化维修批次方案。 |
| [^321] | [An Information-Theoretic Evaluation Framework for Benchmark and Model Diagnosis in Knowledge Tracing](https://arxiv.org/abs/2610.06988) | 该论文提出了一个基于信息论的知识追踪评估框架，利用上下文树加权（CTW）构建可操作的因果不确定性坐标，将模型预测投影到不同熵区间上进行诊断，从而克服了传统AUC等全局聚合指标无法揭示剩余错误来源及基准饱和程度的局限。 |
| [^322] | [A Data-Driven Framework for Unsupervised Monitoring of Transmission Systems Using End-of-Line Testing Data: A Case Study at Ford Motor Company](https://arxiv.org/abs/2610.06980) | 本文提出了一种利用汽车下线测试数据进行变速器系统无监督异常监测的数据驱动多变量框架，兼顾可解释性、低延迟与计算高效性，并在福特汽车公司完成了实际案例验证。 |
| [^323] | [Hierarchy-GBP: Accelerating Factor Graph Inference via Abstraction and Recovery](https://arxiv.org/abs/2610.06978) | 提出层次化高斯置信传播框架H-GBP，先用粗图抽象求解全局误差并投影恢复到原图，再用GBP细化局部误差，从理论上证明其收敛到最优解，实验表明其收敛速度远快于标准GBP。 |
| [^324] | [Uncertainty in Representation Learning on Knowledge Graphs](https://arxiv.org/abs/2610.06974) | 本论文系统研究了知识图谱嵌入中的知识不确定性、算法不确定性和预测不确定性三类来源，并提出了基于投票的聚合框架来缓解模型训练随机性导致的算法不确定性。 |
| [^325] | [EVFormer: An Egocentric Vision-EMG Bidirectional Attention Model for Bimanual Hand Pose Estimation](https://arxiv.org/abs/2610.06970) | EVFormer提出了一种融合RGB视觉与表面肌电信号的多模态双向交叉注意力框架，有效克服视觉遮挡问题，显著提升了第一视角双手手部姿态估计的精度。 |
| [^326] | [Principles that Guide, Actions that Inform: Agent Evolution via Knowledge Abstraction](https://arxiv.org/abs/2610.06964) | 该论文提出SAGA方法，通过将智能体的具体交互经验抽象为可复用的通用知识原则，使LLM智能体无需修改模型参数即可实现自我演化并提升泛化能力。 |
| [^327] | [Do Neural PDE Solvers Learn the Right Dynamics?](https://arxiv.org/abs/2610.06952) | 该论文提出了一个超越传统预测误差评分的评估框架，通过考察误差形成、集合几何结构和极端事件三个互补维度，来检验神经PDE求解器是否真正再现了系统的动力学行为。 |
| [^328] | [Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators via Low-Rank Activation Steering](https://arxiv.org/abs/2610.06950) | 该论文提出一种仅用2.3万至33万参数、通过行为克隆训练的低秩激活转向决策算子，能在不损失精度的情况下将3685个token的长推理压缩为6个token的快速决策，训练成本比现有强化学习方法低约两个数量级。 |
| [^329] | [AdaLoop: Adaptive-Depth Latent Reasoning for Audio Language Models](https://arxiv.org/abs/2610.06949) | AdaLoop是一种轻量级自适应深度潜在推理模块，通过学习到的停止机制为每个音频-问题对动态分配推理步数，仅增加不到3%的参数即可让音频语言模型在细粒度声学分析任务上的平均准确率提升2.9到3.8个百分点。 |
| [^330] | [FactorBench: A Portfolio-Aware Benchmark for Automated Factor Mining](https://arxiv.org/abs/2610.06947) | FactorBench是一个面向投资组合的自动化因子挖掘基准测试，通过统一的评估框架比较九种挖掘方法产生的约五千个因子，从因子有效性、时间泛化性和超越风险与风格暴露的预测能力三个维度衡量金融信号的真实价值。 |
| [^331] | [Learning to Remember: Distilling Memory Retention for Compact Recurrent Neural Networks](https://arxiv.org/abs/2610.06942) | 该论文提出了一种针对时间序列模型时序依赖和记忆保持特性而设计的记忆差异知识蒸馏框架，使紧凑的循环神经网络在资源受限环境中仍能保持高性能。 |
| [^332] | [QiYao-I: A Manifold Based Foundation Model for Irregular Multivariate Time Series Forecasting](https://arxiv.org/abs/2610.06936) | QiYao-I是一个基于流形的基础模型，通过采样条件时间流形注意力机制和频率感知的动态变量交互机制，有效解决了不规则采样、跨变量异步的多变量时间序列预测难题。 |
| [^333] | [Near-Optimal Sample Complexity for Recursive Entropic Risk Reinforcement Learning with a Generative Model](https://arxiv.org/abs/2610.06931) | 本文对基于模型的风险敏感 Q 值迭代（MB-RS-QVI）算法进行了精细分析，在生成模型假设下首次为递归熵风险强化学习建立了近最优的样本复杂度保证，其对有效视界的指数依赖性与现有下界相匹配，消除了理论差距。 |
| [^334] | [Low-Rank and Structured Sparse Tensor Decomposition for Anomaly Detection in Multivariate Functional Data](https://arxiv.org/abs/2610.06930) | 提出两种无监督稀疏张量分解方法（ES-CP与FG-Lasso），通过低秩CP分解结合逐元素与纤维方向的稀疏惩罚，在保留多模态结构的同时检测多元函数型数据中的局部异常与时间纤维集中型异常。 |
| [^335] | [AttSVD:Prompt-Adaptive Low-Rank KV Cache Compression via Attention-Guided SVD](https://arxiv.org/abs/2610.06927) | 提出AttSVD，一种基于每个提示自身注意力几何结构的可解释低秩KV缓存压缩方法，通过在线逐提示截断SVD仅保留注意力实际读取的方向，在保留全部token的同时按保留秩比例削减长上下文下的KV缓存内存，并提供累积式与流式两种解码时缓存策略及自适应压缩改进。 |
| [^336] | [Extending Music Annotation Schemas: Zero-Shot Prediction or Few-Shot Adaptation?](https://arxiv.org/abs/2610.06920) | 该论文提出了一个基于MGPHot流行音乐数据集的音乐标注模式扩展基准，实验表明即使在小标注预算下，监督式适应方法仍比音频-语言模型的零样本预测更有效。 |
| [^337] | [Anchor Divergence for Semantic Geometry in Contrastive Learning](https://arxiv.org/abs/2610.06919) | 本文提出“锚点散度”方法，通过建立锚点概率分布与Bregman几何之间的对应关系，使固定表示上的语义几何能够适配特定上下文，突破了余弦相似度单一固定几何的局限。 |
| [^338] | [Learning from Unreliable Trajectories: Adversarially-Robust Federated Q-Learning](https://arxiv.org/abs/2610.06918) | 该论文提出了对抗鲁棒的联邦Q学习算法Robust Async-Fed-Q，通过智能体端的方差缩减估计与服务器端的鲁棒聚合相结合，在部分智能体发送任意篡改数据的对抗环境下仍能保留诚实智能体协作带来的样本效率收益，并提供了高概率有限时间的理论保证。 |
| [^339] | [Nonlocal Hamiltonian Dynamics on Sparse L\'evy Graphs: Spectral Analysis and Multimodal Sampling](https://arxiv.org/abs/2610.06904) | 该论文提出一种在稀疏Lévy图上运行的阻尼非局部哈密顿动力学方法，通过最近邻连接与采样的长程边实现多峰目标分布之间确定性且代价线性（与节点数和长程采样预算成正比）的概率质量输运，并以加权图拉普拉斯算子的谱刻画惯性与非局部连通性的相互作用、由谱隙确定最优渐近阻尼。 |
| [^340] | [Component and Dimension Sparsity in Transformer Refusal Mechanisms](https://arxiv.org/abs/2610.06903) | 该研究通过对四个开源大语言模型的组件级干预分析，发现拒绝行为引导只需稀疏组件子集（占上游组件28%–48%）及其中约50%的残差流维度即可复现完整效果，揭示了拒绝机制在组件和维度两个层面上的稀疏性。 |
| [^341] | [Memory Prediction Excess: A Probabilistic Quantity for Predictive Gain and Memory Length in Stochastic Processes](https://arxiv.org/abs/2610.06894) | 本文提出“记忆预测超额”（MPE）这一新的概率量，用以量化在离散时间有限状态随机过程中利用完整历史信息相对于仅用静态边际分布所带来的预测准确率平均提升，并证明了其非负性、上界条件及退化情形等基本性质。 |
| [^342] | [Axiom Satisfiability of Linear Rewards in Alignment](https://arxiv.org/abs/2610.06892) | 该论文通过引入逐候选者松弛量，提出一种计算“总松弛最小且满足公理边际η”的线性奖励的方法，在不对投票者和数据收集方式做任何假设的情况下，以被O(1)界定的松弛代价强制线性奖励满足帕累托最优与PMC等公理。 |
| [^343] | [Event-Driven ML Pipeline Orchestration for Manufacturing: An AWS Industry Experience](https://arxiv.org/abs/2610.06890) | 该论文分享了三年来在汽车制造业运营事件驱动云基础设施的行业实践经验，通过ECS、SQS、Lambda等AWS服务编排跨工厂的GPU加速再训练流水线，在4万余个生产训练任务中实现了相比常开GPU基础设施72-78%的成本降低。 |
| [^344] | [Zero-Shot Visualization: Exploring Text Corpora with User-Prompted Axes](https://arxiv.org/abs/2610.06889) | 该论文提出了零样本可视化（ZSV）任务，允许用户通过自然语言指定概念轴来交互式探索文本语料库，并通过基准测试发现基于下一个词元概率的评分方法在语义忠实性、评分保真度和计算成本方面具有优势。 |
| [^345] | [Learning When to Refine: Long-Horizon Reinforcement Learning for Budgeted Neural-Operator PDE Solvers](https://arxiv.org/abs/2610.06883) | 该论文提出面向预算约束神经算子PDE求解的长时程强化学习方法RV-PI，通过实际滚动验证在有限修正预算下学习何时何地施加局部细化，仅在留出轨迹误差改善时接受策略更新。 |
| [^346] | [Comparative review of hybrid forecasting models for short-term prediction of building thermal load](https://arxiv.org/abs/2610.06881) | 本文综述并比较了13种用于建筑热负荷短期预测的混合模型，发现EMD-LSTM-Markov模型的预测精度最高。 |
| [^347] | [Neutrosophic Ensemble Classification for Uncertainty-Aware Bearing Fault Detection: Evidence from Laboratory and Variable-Speed Industrial Benchmarks](https://arxiv.org/abs/2610.06880) | 本文提出将中智学四指标分解（最高类证据、最佳竞争者证据、预测熵与决策分歧）应用于机器学习集成分类器，以区分轴承故障检测中自信的错误与真正模糊的预测，实现不确定性感知的故障诊断，并在实验室与变速工业基准上验证了其有效性。 |
| [^348] | [When Can World Models Recover Physical Laws?](https://arxiv.org/abs/2610.06877) | 本文指出仅凭准确预测不能证明世界模型恢复了物理定律，并给出定律可恢复的充要条件——任意两条不同定律必须在实验上可区分——同时用率-失真下界、有限响应码本、明确解码预算和极小极大采样复杂度刻画了恢复的信息论极限与稳定性。 |
| [^349] | [Statistical Turbulence and High-Fidelity Disturbance Fields for Quadrotor Flight Control](https://arxiv.org/abs/2610.06874) | 本文首次系统量化了风场保真度（而非风速大小）对强化学习四旋翼控制器鲁棒性的影响，通过五种扰动保真度级别的完整交叉训练-测试评估，并配合经过验证的扰动数据，揭示了训练扰动场保真度与策略真实环境表现之间的关系。 |
| [^350] | [When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO](https://arxiv.org/abs/2610.06861) | 该论文提出GA-GRPO统一理论框架，将外部指导建模为随机指导算子，证明其引入的偏差可由全变差指导散度δ_G界定，从而为外部指导何时以及如何帮助LLM推理提供收敛速率、偏差界和最优加权规则的理论基础。 |
| [^351] | [Trajectools Demo: Towards No-Code Solutions for Movement Data Analytics](https://arxiv.org/abs/2610.06858) | 本文基于开源Python库MovingPandas和开源GIS软件QGIS，提出了无代码移动数据分析工具Trajectools插件的概念框架并完成了初步实现，使非程序员用户也能进行移动数据分析。 |
| [^352] | [TEMPEST: Temporal Embeddings for Scalable Driver Identification via Angular Margin Learning](https://arxiv.org/abs/2610.06855) | TEMPEST提出了一种基于ArcFace角度边距损失训练的时间卷积网络嵌入模型，将60秒多模态驾驶数据映射为96维嵌入，在45名驾驶员数据集上达到91.71%的Rank-1准确率，且随驾驶员规模扩大仅出现轻微性能下降，实现了无需重新训练的可扩展驾驶员识别。 |
| [^353] | [Beyond Marginals: A Multi-Dimensional Evaluation Framework for Multi-Table Synthetic Data Generation](https://arxiv.org/abs/2610.06854) | 本文提出了 SynEval，一个与生成器无关的多表合成数据库六维评估框架，联合评估单列保真度、多变量结构、跨表完整性、机器学习实用性、隐私保护和边界情况鲁棒性，并提供统一的加权质量分数及按表、按维度的下钻分析。 |
| [^354] | [A Response Theory Probe for Learned Stochastic AI Simulators, Tested on Lorenz-63](https://arxiv.org/abs/2610.06798) | 本文提出一种基于线性响应理论的校准检验探针，用于评估学习型随机AI模拟器对外部强迫的响应是否正确，并在随机Lorenz-63模型上对SINDy、MLP、储备池计算机、神经ODE和神经SDE等多种模拟器进行了模态分辨的基准测试。 |
| [^355] | [To Learn is to Wander: Learning Across Graphs and Tasks with Random Walks](https://arxiv.org/abs/2610.06694) | 提出基于随机游走统一接口的图基础模型Wander，将图学习形式化为部分观测图的补全，使单个预训练模型能够跨图类型、特征、关系模式与预测任务通用迁移，并具备逼近贝叶斯最优预测器的理论保证。 |
| [^356] | [Improving Proactive AI Assistance with Hierarchical Procedural Understanding](https://arxiv.org/abs/2610.06505) | 本文提出了ProactiveCoach数据集套件，利用分层程序性理解使主动式AI助手能够根据任务进度和用户需求提供自适应粒度的指导。 |
| [^357] | [Lossy Compression of PDE Training Inputs: Field Reconstruction Error Does Not Order the Cost to a Trained Operator](https://arxiv.org/abs/2610.06095) | 本文证明压缩PDE训练输入时的场重构误差无法预测所训练算子的精度损失，因为解算子对输入扰动的衰减程度在不同PDE族之间相差两个数量级以上，导致重构误差指标在104个代价比较中反转了36个。 |
| [^358] | [Quantum data loading from the learned shared structure of real signals](https://arxiv.org/abs/2610.06076) | 该论文提出一种量子原生数据加载器，通过一次性学习真实数据集共享的低维结构，以固定电路和少量参数高效加载所有信号，且所需训练子集大小不随数据规模增长而增加。 |
| [^359] | [Incentive Alignment in Online Experimentation](https://arxiv.org/abs/2610.05922) | 该论文将在线实验重新建模为激励设计问题，揭示了实验者基于有偏实验结果获得奖励所导致的委托-代理冲突会侵蚀平台价值，并证明样本拆分和收缩估计这两种实用机制能够有效实现激励对齐。 |
| [^360] | [What Does Fr\'echet Distance Measure? A Directional Decomposition](https://arxiv.org/abs/2610.05518) | 该论文提出“方向性弗雷歇距离”，将FID/FVD等单一标量指标分解为最优传输位移在少数可解释方向上的投影，从而揭示了标量指标背后的差异来源，解释了采样步数增加提升ImageReward却恶化FID的不一致现象。 |
| [^361] | [Best-of-$N$ Guidance for Test-time Diffusion Alignment](https://arxiv.org/abs/2610.05108) | 该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。 |
| [^362] | [E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation](https://arxiv.org/abs/2610.05048) | 论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。 |
| [^363] | [Bridging the EHR Divide: Asymmetric Contrastive Learning for Cross-National Medical Representation Transfer](https://arxiv.org/abs/2610.04946) | 提出非对称监督对比学习预训练目标，在台湾398万名患者的NHIRD纵向记录上预训练时序Transformer编码器，并借助混合语义映射管线成功迁移至美国的MIMIC-IV和EHRSHOT数据集，实现了跨国家、跨临床词汇表的医疗表征迁移。 |
| [^364] | [SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models](https://arxiv.org/abs/2610.04875) | SpecFold通过识别并利用投机验证中草稿分支与父分支之间隐藏状态高度相似的多分支计算冗余，以token级残差门控和选择性计算复用降低验证成本，从而加速扩散语言模型的多分支投机解码。 |
| [^365] | [AID: A Framework for AI Infrastructure Dynamics](https://arxiv.org/abs/2610.04801) | 本文提出AID框架，用于对耦合物理、计算、网络与服务进程的AI推理基础设施学习问题进行建模，支持异步观测与多时间尺度，并给出了预测误差下界和精确受控状态约简充分条件两项分析结果。 |
| [^366] | [Consideration Circuits: Depth Separation and Universality Beyond a Single Softmax](https://arxiv.org/abs/2610.04143) | 论文提出由多项logit（MNL）单元构成的有向无环图所定义的“考虑电路”多阶段选择模型，并证明了尖锐的深度-范数分离定理：深度从2增至3时，逼近误差ε所需的品味向量范数从Θ(log(1/ε)/ε)降至Θ(log(1/ε))，而包括单一MNL在内的菜单无关随机效用模型在折中任务上误差存在不可消除的下界，从而在表达能力与结构上严格超越了单一softmax模型。 |
| [^367] | [Ideal Paths for Approximating Logistic Gradient Descent Trajectories at Large Initialization](https://arxiv.org/abs/2610.04142) | 该论文提出由“负间隔修正”和“最小间隔增长”两个阶段组成的理想路径，证明在大初始化条件下逻辑斯蒂梯度下降的轨迹经过显式两阶段时间重参数化后可被该路径几何逼近。 |
| [^368] | [DePICT: Decision-Preserving Interface for Constrained Downstream Tasks](https://arxiv.org/abs/2610.03945) | 该论文提出DePICT方法，基于KKT条件刻画上下文方向与优化器的相关性，并依据解灵敏度对方向排序聚合，构建决策保持接口，从而揭示决策系统真正依赖的输入方向。 |
| [^369] | [SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents](https://arxiv.org/abs/2610.03372) | SCAD通过将长时程智能体的交互分解为规划与有界子任务执行，并将基于结果的信用分配与教师引导蒸馏相结合，在文本和多模态任务上显著优于最强训练基线。 |
| [^370] | [Verifiable, Articulable, and Tacit Components of Preference](https://arxiv.org/abs/2610.03025) | 该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。 |
| [^371] | [Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning](https://arxiv.org/abs/2610.02687) | 该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。 |
| [^372] | [Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis](https://arxiv.org/abs/2610.02659) | 该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。 |
| [^373] | [Sequential Capacity of Quantum Processes with Finite Memory](https://arxiv.org/abs/2610.02068) | 该论文证明，仅用一个可见量子比特、无额外内部记忆的量子系统，通过时间依赖的相位旋转即可实现随运行长度以 $K\log K$ 量级增长的序列响应容量，而经典随机过程在相同测试下仅有线性容量。 |
| [^374] | [SkillEvoLean: Mutation-enhanced skill evolution for Lean provers](https://arxiv.org/abs/2610.01799) | 提出SkillEvoLean——一种变异增强的技能自演化框架，通过联合演化高层解题策略与参考知识（数学概念、证明技巧），即使在所有采样证明轨迹全部失败时也能为Lean证明器提供有效的技能更新方向。 |
| [^375] | [Completion Aware Guidance for World Action Models](https://arxiv.org/abs/2610.01559) | 提出无需训练的完成感知引导（CAG）采样方法，引导世界动作模型的生成朝向任务完成，显著提升机器人任务成功率并将任务不完整想象从 79% 大幅降至 40%。 |
| [^376] | [Does Scaling Reinforcement Learning Really Require More Training?](https://arxiv.org/abs/2610.01133) | 提出策略空间扩展方法SURGE，通过对同一强化学习运行中的两个检查点（高精度锚点与生成简短回复的供体）进行特征空间融合，在不增加训练或推理计算的情况下获得比原检查点更强的策略。 |
| [^377] | [How Much Can Language Models Gain from Test-Time Computation?](https://arxiv.org/abs/2610.01110) | SELF-POT 是一个统一基准框架，以美元计价所有模型调用，衡量语言模型在数学、编程和智能体任务中从并行采样、自我修订等测试时计算中能获得的收益与成本。 |
| [^378] | [Rate-Optimal Algorithm for Adversarial Linear CMDPs](https://arxiv.org/abs/2610.00927) | 本文提出一种新的原始对偶算法，在无需假设Slater条件的情况下，将对抗性线性约束马尔可夫决策过程的遗憾和累积约束违反从$\widetilde{\mathcal{O}}(K^{3/4})$提升至最优的$\widetilde{\mathcal{O}}(\sqrt{K})$，弥补了理论差距。 |
| [^379] | [Reward as Observation: Learning Reward-Based Policies for Rapid Adaptation](https://arxiv.org/abs/2610.00729) | 本文提出一种仅以奖励和动作为条件的策略学习方法，实现了从简单环境向观测空间完全不同的新环境（如3D渲染和机器人导航）的零样本快速迁移。 |
| [^380] | [Grand Canonical Generators](https://arxiv.org/abs/2610.00683) | 提出了巨正则生成器（GCG），将玻尔兹曼生成器扩展至巨正则系综，其分解式设计可复用现有正则生成器、解析编码化学势线性依赖，并提供可处理的似然以支持自归一化重要性采样，在流体和吸附问题上准确再现巨正则观测量。 |
| [^381] | [Shared Phase and Retention Control for Efficient Adaptive Spectral Recurrence](https://arxiv.org/abs/2609.39082) | SPARC证明高维谱记忆无需高维控制，仅用两个输入依赖的标量信号即可共享协调记忆保留与相位旋转，实现控制成本与状态容量解耦的高效自适应谱递归模型。 |
| [^382] | [Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning](https://arxiv.org/abs/2609.37858) | 该论文发现“存储目标知识”的参数未必是执行遗忘的最佳干预对象，提出基于实际遗忘更新预测效果的干预分数以及动态干预重排方法（DIR-R），在优化过程中按需自适应调整干预参数子集，从而显著提升大语言模型遗忘的效果。 |
| [^383] | [FineART: Fine-grained Annotated Robotic Trajectory Dataset and Vision-Language-Action Model for Bimanual Manipulation](https://arxiv.org/abs/2609.36416) | 本文提出了包含40,543个回合、1,718小时和533,913个密集子任务标注的双手操作数据集FineART，以及能自主预测下一个子任务的视觉-语言-动作模型FineART-VLA，显著提升了长时程双手操作任务的成功率。 |
| [^384] | [LionMuon: Alternating Spectral and Sign Descent for Efficient Training](https://arxiv.org/abs/2609.35297) | LionMuon 通过每 P 步交替执行一次昂贵的 Muon 谱步骤与廉价的 Lion 符号步骤，并共享单一双重 EMA 动量缓冲区，在大幅降低计算与通信开销（优化器状态仅为 AdamW 一半）的同时保持甚至超越 Muon 的优化效果，并从理论上证明了其在重尾噪声下何时优于两种原始优化器。 |
| [^385] | [CLAD: Constrained Abstract Domain for Neural Network Verification](https://arxiv.org/abs/2609.34628) | 提出了约束拉格朗日抽象域（CLAD），能够在Lp范数球附加额外约束的复杂输入区域上计算神经网络行为更紧致的可靠过近似，从而克服现有抽象域因输入区域描述受限而导致的验证失败或虚假反例问题。 |
| [^386] | [ZonoGPT: Towards An Abstract Domain for Verifying Large GPT Models](https://arxiv.org/abs/2609.34457) | ZonoGPT提出了一种空间复杂度与网络深度无关的抽象域，通过结构化zonotope、生成元约简机制以及针对Attention、LayerNorm和GELU的保精度变换，实现了对大型GPT模型的高效形式化验证。 |
| [^387] | [KVCMAS: Efficient KV cache Correction for Shared Context in Multi-Agent Systems](https://arxiv.org/abs/2609.34060) | 提出了KVCMAS在线KV缓存修正框架，通过修正多智能体系统中跨智能体的KV缓存偏差，避免各智能体重复预填充共享上下文，从而显著降低计算与内存开销。 |
| [^388] | [PReCache: Efficient KV Cache Sharing for Multi-LoRA Agents via Low-Rank Precomputation and Neutral Reconstruction](https://arxiv.org/abs/2609.34054) | PReCache是一个无需训练的KV缓存共享框架，通过共享基于预训练权重计算的基础缓存并预计算紧凑的智能体特定低秩缓存，消除了多LoRA智能体系统中的重复预填充冗余，同时避免了直接缓存复用对角色特定行为的削弱。 |
| [^389] | [SketchSSM: Write to the Full State, Read from a Compact Sketch](https://arxiv.org/abs/2609.33051) | SketchSSM在每次状态更新时读取一次完整状态并预计算离线固定基向量的输出存入紧凑草图，从而在保持完整状态写入的同时，让每个解码步骤无需读取完整状态即可近似重构状态读取结果。 |
| [^390] | [Adaptive Latent Capacity for World Models](https://arxiv.org/abs/2609.32921) | 本文提出ALeWM，一种基于JEPA的世界模型，通过学习前缀长度分布和新的MixSIGReg正则化方法，自适应地将预测信息集中到宽潜在表示的紧凑前缀中，以提升预测与递归规划能力。 |
| [^391] | [Action Shaping: Policies Absorb What They Can Express](https://arxiv.org/abs/2609.32752) | 论文提出“动作塑形”原理：训练时加入的动作偏移量若能被策略输出层精确表达，策略便会将其完全吸收，从而可在部署时安全移除该偏移而不损失回报，其最简实现为零初始化线性头配合可学习门控，门控会自行先升后降。 |
| [^392] | [Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions](https://arxiv.org/abs/2609.32652) | 该论文为几何诱导软状态抽象的线性预测精度建立了无需拟合预测矩阵即可由独立评估数据计算的有限样本下置信界，并揭示了核仿射包机器坐标的重构分数裕度与预测误差界之间的联系。 |
| [^393] | [Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models](https://arxiv.org/abs/2609.30995) | 该论文提出一种分层因果表示学习框架，能够显式区分内部气候变率与外部强迫响应，从而准确预测气候变化情景下的温度演变，并提升机器学习气候模拟器的可信度与因果归因能力。 |
| [^394] | [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059) | KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。 |
| [^395] | [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。 |
| [^396] | [ValueDiff: Value-Geometric KV Cache Eviction for Sink-Suppressed LLMs](https://arxiv.org/abs/2609.23314) | 针对因QK归一化等技术导致注意力汇聚减弱的现代大语言模型，提出基于价值向量与缓存均值L2偏差进行token排序的ValueDiff淘汰方法，在2k-4k token预算下可保留密集注意力88-99%的性能。 |
| [^397] | [Schedule optimization for tau-leaping in masked discrete diffusion](https://arxiv.org/abs/2609.21960) | 该论文通过依赖密度ρ的精确积分表示来刻画掩码离散扩散中tau-leaping采样的因式分解误差，并推导有限步优化问题的递归平稳性方程，从而实现去噪调度的优化。 |
| [^398] | [Local Sparsity Enables Unsupervised LLM Safety Detection](https://arxiv.org/abs/2609.20129) | 本文利用稀疏自编码器概念空间中的局部稀疏性这一关键洞察，提出了一个无需不安全训练数据的无监督LLM安全异常检测框架，并有理论支撑。 |
| [^399] | [FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning](https://arxiv.org/abs/2609.18964) | 该论文提出FedGuide框架，通过扩散先验作为行为模型并利用最优传输混合专家进行聚合，解决了异构联邦强化学习中客户端间的分布不匹配问题，同时以DICE价值基线提供低方差的回报感知引导。 |
| [^400] | [SpliTEE: Improving LLM Inference on Trusted Hardware with Differentially Private GPU Outsourcing](https://arxiv.org/abs/2609.15039) | 该论文提出SpliTEE，将拆分推理架构扩展到LLM推理场景，用差分隐私（而非加密）保护发送到不受信任GPU的中间表示，从而在可信硬件上实现高效且隐私安全的大语言模型推理。 |
| [^401] | [VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis](https://arxiv.org/abs/2609.03203) | VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。 |
| [^402] | ["Train classical, deploy quantum" requires rethinking generalization](https://arxiv.org/abs/2608.31117) | 该论文指出，“在经典计算机上训练、在量子设备上部署”的量子生成模型范式存在根本性问题——即便使用经典可评估的损失函数（如基于Pauli-Z关联的MMD²）成功完成训练，也必须重新审视其泛化能力，因为最小化这类经典目标未必能保证量子部署阶段生成样本的质量。 |
| [^403] | [Generation of High-Level Concepts in 3D Scene Graphs via Autoregressive Diffusion](https://arxiv.org/abs/2608.28733) | 提出了一种统一的自回归扩散图生成模型，联合学习图结构与节点特征，可从观测平面自底向上构建跨越任意层次深度的完整3D场景图，并在合成与真实建筑数据集上全面超越现有基线方法。 |
| [^404] | [Beyond Procrustes distances: a multilinear Gromov-Wasserstein distance capturing chirality](https://arxiv.org/abs/2608.27774) | 该论文提出了Gromov-Wasserstein目标的多线性推广，由此定义出对手性敏感的手性Gromov-Wasserstein（CGW）距离，能够区分形状与其镜像，并具备鲁棒性保证和高效的计算算法。 |
| [^405] | [When Is the Sharp Covariance Envelope Tight? Feature-Only Geometry for Volume-Sampled Least Squares](https://arxiv.org/abs/2608.26877) | 本文建立了体积采样最小二乘中心系数协方差的Loewner包络，并揭示了仅特征边际nu_A决定谱包络严格性的精确条件。 |
| [^406] | [The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection](https://arxiv.org/abs/2608.26423) | 本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。 |
| [^407] | [Canalization Before Generalization: Grokking as a Dynamical Probe](https://arxiv.org/abs/2608.25813) | 本文通过权重衰减脉冲在“顿悟”平台期揭示了解选择的通道化过程，表明在可见泛化之前就形成了稳定的剂量排序，为理解神经网络泛化机制提供了新的动力学视角。 |
| [^408] | [When Similarity Is Interaction-Driven: Quantum Kernels for Regime-Sensitive Learning](https://arxiv.org/abs/2608.24631) | 该论文提出了一种交互驱动的量子核方法，通过纠缠泡利串特征映射显式编码高阶稀疏交互，在机制敏感学习中显著优于传统核方法。 |
| [^409] | [A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines](https://arxiv.org/abs/2608.20980) | 本文通过经典时间序列方法分析常用时空基准数据集，揭示无空间感知的线性模型比以往报告更具竞争力，质疑了现有基准数据集的判别可靠性。 |
| [^410] | [Systematic Evaluation of TabPFN-TS for Zero-Shot Probabilistic Heat Load Forecasting in District Heating Networks](https://arxiv.org/abs/2608.20024) | 本研究首次系统评估了基于合成数据预训练的TabPFN-TS模型在区域供热网络零样本概率热负荷预测中的性能，探讨了其与真实时间序列基础模型相比的优劣及对供热动态的捕捉能力。 |
| [^411] | [RIPE++: Reinforced Keypoint Learning from Positive Pairs Only](https://arxiv.org/abs/2608.19693) | 本文提出RIPE++，通过从单个正样本对中同时提取奖励和惩罚，避免了负样本对比，从而提升了关键点学习的稳定性和描述符判别性。 |
| [^412] | [Enhancing Distance-Based Graph Autoencoders with Structural Penalties for Dynamic Graph Embedding](https://arxiv.org/abs/2608.18762) | 本文提出三种基于距离的图自编码器变体，通过引入枢纽惩罚和NC-LID结构惩罚，增强了动态图嵌入对结构模糊节点的重构准确性。 |
| [^413] | [PertMind: Eliciting Emergent Biological Reasoning in LLM via Reinforcement Learning on Cellular Perturbation Data](https://arxiv.org/abs/2608.16419) | PertMind通过将细胞扰动图谱转化为强化学习环境，仅用正向预测训练便激发了大语言模型的涌现生物推理能力，并实现了跨任务的零样本迁移。 |
| [^414] | [SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry](https://arxiv.org/abs/2608.16287) | SCALE通过状态校准潜在嵌入，使端到端学习的LeWM表示获得类似DINO-WM的几何特性，从而改善基于JEPA的世界模型在规划中的状态信息利用。 |
| [^415] | [Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers](https://arxiv.org/abs/2608.14089) | 本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。 |
| [^416] | [QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction](https://arxiv.org/abs/2608.13966) | 本文提出QUASAR，一种在量化感知训练过程中持续进行轻量级损失感知重构的方法，以降低损失下限并提升低比特模型质量。 |
| [^417] | [Do AI weather models miss extremes?](https://arxiv.org/abs/2608.09972) | 本研究利用十个月欧洲站点观测对十二个 AI 与物理气象模型进行评估，发现 AI 模型在极端天气上并不存在统一的低估缺陷，尾部表现取决于具体模型、变量和评估环境，但所有模型都普遍存在对极端值向均值收缩的共同条件误差模式。 |
| [^418] | [Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers](https://arxiv.org/abs/2608.07436) | 论文发现Muon训练的Transformer在grokking后发生崩溃的根源在于AdamW读出层更新与较大的特征均值相互作用产生类别相关的logit偏移，并叠加交叉熵导数错误，而修正这些错误可以稳定训练并恢复近乎完全的准确率。 |
| [^419] | [EvoHarness-RL: Learning Runtime Harness Coordination for Self-Evolving Agents](https://arxiv.org/abs/2608.05446) | 提出EvoHarness-RL统一框架，通过将环境特定支架实现与共享策略接口分离、构建信念-进度-经验（BPE）工作空间，并采用监督初始化加代价感知GRPO使支架协调可学习，显著提升长程LLM智能体的运行时支撑能力。 |
| [^420] | [Multi-Scale Structural Features for Continual, Comprehensible Visual Recognition in a Developmental Learning Framework](https://arxiv.org/abs/2607.25531) | 该论文提出了一种在多个尺度上编码边缘与轮廓结构及其空间关系的新型视觉特征表示，并将其融入无梯度的发展式学习框架，从而在无需回放缓冲区的条件下实现更准确、可持续且可解释的视觉识别。 |
| [^421] | [Variational-Ising-Attention:Tailored Attention Matters for Science](https://arxiv.org/abs/2607.23634) | 提出变分伊辛注意力（VIA），通过带相互作用的伊辛模型与变分平均场推断增强softmax注意力，将注意力从孤立条目的排序扩展为相互作用实体的集体状态建模，并在逆合成反应中心预测与蛋白质残基接触预测等科学结构化预测任务上验证了其有效性。 |
| [^422] | [Token-Level Off-Policy Learning for Faithful Generation Under Distribution Shift](https://arxiv.org/abs/2607.17524) | 提出令牌级离策略标注（TOPL）训练范式，将后训练重构为令牌级正确性预测任务，使模型在摘要、翻译等忠实生成任务中实现强大的分布外泛化能力。 |
| [^423] | [Retrieval-Augmented Interpretable Learning: Towards Task-Specific Zero-Shot Models in Healthcare](https://arxiv.org/abs/2607.17508) | RAIL是一种概率元学习框架，能够从自然语言任务描述出发，通过检索相关源任务并进行系数空间结构迁移，零样本生成具有特征级解释和不确定性量化的可解释医疗临床预测模型。 |
| [^424] | [Online Neural Space Time Memory for Dynamic Novel View Synthesis](https://arxiv.org/abs/2607.15271) | 本文提出一种将记忆更新与记忆应用频率解耦的在线神经时空记忆方法，通过周期性进行梯度记忆更新、逐帧结合跨视角注意力应用记忆，从而在满足实时约束的同时实现动态场景的新视角合成，并有效重建暂时被遮挡的区域。 |
| [^425] | [Beyond Scaffold Splits: Structural-Frontier Evaluation Reveals Hidden Failures in ADMET Models](https://arxiv.org/abs/2607.10729) | 该论文提出一种无标签的结构前沿划分评估方法，发现传统骨架划分会掩盖ADMET模型在化学结构最稀疏、物理化学性质最遥远的分子上的严重性能退化（误差膨胀中位数达87%），且这一问题在图神经网络中同样存在。 |
| [^426] | [Learning from Hindsight for VLA Reinforcement Learning](https://arxiv.org/abs/2607.09042) | LfH通过利用预训练视觉-语言模型对失败的机器人轨迹进行重新标记，将失败经验转化为有效的学习信号，从而显著提升VLA模型强化学习的样本效率。 |
| [^427] | [On the effectiveness of reward functions in reinforcement learning for confidence calibration of large language models](https://arxiv.org/abs/2607.04332) | 该论文揭示了设计不当的置信度奖励函数会诱导大语言模型为校准置信度而故意答错问题（即“置信度奖励黑客行为”），并提出并验证了可抵御此类攻击的“不可被黑客攻击的置信度奖励方案”的构建方法。 |
| [^428] | [The Geometry of Statistical Feature Learning in Mean-Field Langevin Dynamics](https://arxiv.org/abs/2606.31429) | 该论文通过基-纤维分解为统计特征学习建立了几何框架，证明球面平均场朗之万动力学的低温平稳分布在多指标模型中会集中于隐藏指标并形成多尖峰结构、以高概率实现参数恢复，且这一集中现象在温度约等于1处存在锐利的相变。 |
| [^429] | [Learning Probabilistic Filters with Strictly Proper Scoring Rules](https://arxiv.org/abs/2606.26497) | 本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。 |
| [^430] | [Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic](https://arxiv.org/abs/2606.23044) | 本文提出素数傅里叶嵌入，将整数编码为按素数索引的 (cos, sin) 对，借助舒尔引理从理论上证明等变线性映射必为每个素数对应一个独立块的块对角结构，并结合中国剩余定理预测与超过 500 倍特化比的消融实验验证，表明模运算可归结为选择相关的素数通道。 |
| [^431] | [Do Sparse Autoencoders Learn Meaningful Concept Hierarchies?](https://arxiv.org/abs/2606.22994) | 本文针对稀疏自编码器的概念层次结构首次提出了一套系统的关键要求与具体评估协议，用以严格检验现有方法是否真正学到了有意义的概念层次。 |
| [^432] | [Explaining Attention with Program Synthesis](https://arxiv.org/abs/2606.19317) | 该论文提出通过程序合成将Transformer语言模型的注意力头行为转化为可执行的Python程序，利用预训练语言模型生成并根据保留数据筛选程序，证明不到1000个程序即可复现GPT-2和TinyLlama-1.1B中注意力头的注意力模式。 |
| [^433] | [We Need Explanation Cards to Connect Explanation Algorithms to the Real World](https://arxiv.org/abs/2606.16786) | 本文提出“解释卡片”框架，通过为标准算法解释补充稳健性、有效性信息及清晰的解读说明，弥合解释表面含义与实际价值之间的差距，使解释算法在现实世界中更可靠、更实用。 |
| [^434] | [SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs](https://arxiv.org/abs/2606.16193) | SAE++提出级联稀疏自编码器架构，直接在第一级SAE的解码器权重上训练第二级SAE，从而在多模态大语言模型中学习层次化的“概念的概念”视觉表征。 |
| [^435] | [Enhancing Spectral Embedding through Robust and Flexible Knowledge Transfer in Electronic Health Records](https://arxiv.org/abs/2606.11570) | 该论文提出一种基于谱方法的无监督表示学习框架，通过放宽一对一信号对齐假设并采用两步嵌入流程，从更广泛人群中稳健且灵活地迁移知识，为样本量有限的罕见病电子健康记录数据生成高质量的低维嵌入表示。 |
| [^436] | [Express Language Modeling](https://arxiv.org/abs/2606.10944) | Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。 |
| [^437] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^438] | [Mean-based algorithms: A lower bound and regret](https://arxiv.org/abs/2606.04931) | 本文首次为未知时间范围和赌博机反馈下基于均值算法的核心序列 $\gamma_t$ 建立了下界，揭示了此类算法学习速度的根本极限，并提出了两种新的基于均值的算法（其中一种推广了 $\epsilon$-贪婪算法）。 |
| [^439] | [FFR: Forward-Forward Learning for Regression](https://arxiv.org/abs/2606.03927) | 提出FFR框架，首次将前向-前向学习算法从分类扩展到真实世界的回归任务，通过序数竞争好感度函数等三项关键创新解决了连续目标空间缺乏对比“对立面”的问题，并在多个真实数据集上取得了有竞争力的性能。 |
| [^440] | [Flow-Transformed Implicit Processes for Function-Space Variational Inference](https://arxiv.org/abs/2606.01954) | 提出流变换隐过程（FTIP），通过超越高斯组合权重分布的限制，使有限维函数空间近似能够灵活表示非对称、重尾或多峰的后验不确定性。 |
| [^441] | [Task diversity produces systematic transfer but inhibits continual reinforcement learning](https://arxiv.org/abs/2606.00880) | 该论文提出了GPU加速的持续强化学习环境Banyan，可通过参数化控制任务多样性，并发现任务多样性能带来系统性迁移能力，但会抑制智能体在连续分布偏移中的持续学习能力。 |
| [^442] | [The Terminal Representation in Reinforcement Learning](https://arxiv.org/abs/2605.31289) | 本文提出了强化学习中一种结构上全新的终止表示（TR），它类似于默认表示（DR）编码奖励加权轨迹，但能以更低维度学习，且无需特征向量计算即可直接支持选项发现、奖励塑形、迁移学习和探索等下游任务。 |
| [^443] | [FHRFormer: A Self-Supervised Masked Transformer Framework for Fetal Heart Rate Time-Series Inpainting and Forecasting](https://arxiv.org/abs/2605.29695) | 该论文提出了FHRFormer，一种自监督掩码Transformer框架，能够修复和预测可穿戴胎心率监测中因信号丢失而产生的数据缺口，从而为基于AI分析连续胎心率数据以预测新生儿呼吸辅助风险奠定基础。 |
| [^444] | [Neural Scaling Laws for Jet Generation](https://arxiv.org/abs/2605.28940) | 这项工作首次证明神经缩放定律同样适用于粒子喷注生成任务，在模型规模缩放中复现了对数缩放行为，并发现物理量的切片Wasserstein距离与验证损失单调相关。 |
| [^445] | [SeedER: Seed-Expand-Retrieve for Efficient Knowledge Graph Retrieval](https://arxiv.org/abs/2605.23753) | 提出SeedER方法，采用“种子-扩展-检索”策略实现高效的知识图谱检索，克服了稠密嵌入在多跳查询上的理论局限以及LLM智能体和GNN全图处理的高昂计算成本。 |
| [^446] | [Reinforcement Learning over Predictive Distributions for LLM Regression](https://arxiv.org/abs/2605.20740) | 提出了分布感知奖励（DAR），一种同策略强化学习目标，通过留一法贡献对同一输入的多个预测所形成的预测分布进行联合评估，从而提升大语言模型回归的校准质量。 |
| [^447] | [To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents](https://arxiv.org/abs/2605.18882) | 该研究提出并验证了“内在偏差假说”（IBH），揭示大语言模型智能体在调用/不调用决策中存在与激活无关的过度调用偏差，利用稀疏自编码器定位并量化该偏差，并通过自适应边际校准转向（AMCS）方法实现了因果层面的纠正。 |
| [^448] | [Stochastic Penalty-Barrier Method for Constrained Machine Learning](https://arxiv.org/abs/2605.18618) | 本文提出随机惩罚-障碍方法（SPBM），通过对偶变量指数平均、稳定惩罚调度和Moreau包络扩展经典惩罚-障碍方法以求解约束机器学习问题，证明了小批量采样下变换问题可行集包含于原问题可行集，实验表明其性能与最先进方法相当。 |
| [^449] | [Online Conformal Prediction for Non-Exchangeable Panel Data](https://arxiv.org/abs/2605.17705) | 提出了W-TQA方法，通过结合从单元历史学习的相似性权重与自适应误覆盖水平，解决了非可交换、部分观测面板数据中的在线保形预测问题，并证明了即使在反馈缺失情况下也能实现长期平均覆盖率保证。 |
| [^450] | [Voice "Cloning" is Style Transfer](https://arxiv.org/abs/2605.16578) | 这篇论文揭示语音克隆并非真正“克隆”个人声音，而是系统性地进行风格迁移，使克隆语音比源语音显得更权威、温暖且更像人类，并导致说话人特征的同质化。 |
| [^451] | [ForcingDAS: Unified and Robust Data Assimilation via Diffusion Forcing](https://arxiv.org/abs/2605.14285) | 提出ForcingDAS框架，通过为每帧分配独立噪声水平的扩散强制方法，统一了滤波与平滑两种数据同化模式，并解决了传统逐帧转移模型在非马尔可夫观测下长期误差累积的问题。 |
| [^452] | [Behavioral Guarantees for Proxy-Based Unlearning](https://arxiv.org/abs/2605.10680) | 本文提出了一个统一并泛化基于代理的机器遗忘方法的框架，首次从理论上证明了遗忘模型与保留数据理想后验分布之间KL散度的上界，从而为遗忘后模型的行为提供了可保证的近似程度，使其最接近从头重训的模型。 |
| [^453] | [MSPR: Multi-scale Predictive Representations for Goal-conditioned Reinforcement Learning](https://arxiv.org/abs/2605.09364) | 本文提出MSPR框架，利用多尺度预测监督在离线目标条件强化学习中实现状态与目标的潜在空间对齐，在视觉和状态任务上均达到最先进性能，并对现实且具有挑战性的数据条件保持鲁棒性。 |
| [^454] | [PMCTS: Principled Parallelized Inference Time Scaling with Particle Monte Carlo Tree Search](https://arxiv.org/abs/2605.08982) | 本文提出了PMCTS，一种专为GPU批处理并行化设计的原则性并行蒙特卡洛树搜索算法，它在保持策略改进保证的同时随并行计算规模良好扩展，并在国际象棋、围棋等MCTS和强化学习评估任务中始终优于或媲美流行的启发式基线方法。 |
| [^455] | [Learning Visual Feature-Based World Models via Residual Latent Action](https://arxiv.org/abs/2605.07079) | 该论文提出了一种可从DINO残差中轻松学习的新型潜在动作表示“残差潜在动作”（RLA），并基于流匹配构建RLA世界模型（RLA-WM），实现了更高效、更少幻觉且预测质量更优的视觉特征世界模型。 |
| [^456] | [A Finite-Iteration Theory for Asynchronous Categorical Distributional Temporal-Difference Learning](https://arxiv.org/abs/2605.06866) | 本文为异步分类分布时间差分方法建立了有限迭代收敛理论，通过受限域分析和泊松方程分解，在折扣和固定视界场景下提供了无需混合时间窗口的收敛保证。 |
| [^457] | [Rethinking Adapter Placement: A Dominant Adaptation Module Perspective](https://arxiv.org/abs/2605.06183) | 该论文提出PAGE探测方法，发现LoRA可训练梯度能量高度集中于单一浅层FFN下投影的“主导适配模块”，其位置由模型架构决定且跨任务稳定，为有限数量适配器的最优放置提供了明确指导。 |
| [^458] | [XDecomposer: Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction](https://arxiv.org/abs/2605.05866) | 提出XDecomposer，将多相XRD分析形式化为集合预测问题，无需候选相列表、结构模板或相数量先验即可实现多相XRD图谱的联合分解与结构识别。 |
| [^459] | [Von Neumann Networks](https://arxiv.org/abs/2605.05780) | 该论文将冯·诺依曼上世纪的细胞计算模型与现代深度学习相结合，提出了冯·诺依曼神经元及其网络（VNNs），其架构可自组织生成、仅依赖于输入输出在细胞阵列上的位置，并在数学上基于神经算子扩展与格林函数学习。 |
| [^460] | [From Dual Tracking to Clipping: Provably Faster Distributionally Robust Multi-Objective Optimization](https://arxiv.org/abs/2605.05660) | 本文提出了分布鲁棒多目标优化（DR-MOO）框架并设计具有可证明保证的MGDA算法，先通过对偶估计的双循环算法达到O(ε^{-8})样本复杂度，再结合大批量采样与梯度裁剪在广义平滑条件下实现更快的收敛。 |
| [^461] | [Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time](https://arxiv.org/abs/2604.26070) | 本文提出可观测神经ODE，证明了在显式结构假设下，潜在状态的可观测性可通过连续时间条件前门调整识别动态处理效应，从而即使在存在隐藏混杂因素时也能实现可识别的连续时间因果预测。 |
| [^462] | [Can an MLP Absorb Its Own Skip Connection Exactly?](https://arxiv.org/abs/2604.23705) | 该论文证明，对于当前前沿语言模型所使用的激活函数（如ReLU^2、ReGLU、SwiGLU、GeGLU），相同宽度的无残差MLP在任何深度下都无法精确计算残差块 x + MLP(x) 所表示的函数，因此移除跳跃连接后重新训练无法恢复完全相同的函数。 |
| [^463] | [Algorithm Selection with Zero Domain Knowledge via Text Embeddings](https://arxiv.org/abs/2604.19753) | ZeroFolio利用预训练文本嵌入替代手工设计的实例特征，实现了零领域知识的算法选择，在涵盖7个领域的11个ASlib场景中的绝大多数上超越了传统方法。 |
| [^464] | [Sinkhorn doubly stochastic attention rank decay analysis](https://arxiv.org/abs/2604.07925) | 本文证明了使用Sinkhorn算法归一化的双随机注意力矩阵比标准softmax行随机注意力更能有效保持网络深度中的秩，从而缓解秩崩溃问题。 |
| [^465] | [Neural Global Optimization via Iterative Refinement from Noisy Samples](https://arxiv.org/abs/2604.03614) | 本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。 |
| [^466] | [Adaptive Semantic Communication for Wireless Image Transmission Leveraging Mixture-of-Experts Mechanism](https://arxiv.org/abs/2604.02691) | 本文提出了一种基于自适应混合专家Swin Transformer模块的多阶段端到端MIMO图像语义通信系统，其核心创新在于设计了同时联合评估实时信道状态信息与语义内容的动态专家门控机制，从而突破传统单一驱动路由的局限，实现对多样化图像内容和动态信道条件的自适应无线图像传输。 |
| [^467] | [Generalizable Dense Reward for Long-Horizon Robotic Tasks](https://arxiv.org/abs/2604.00055) | 提出VLLR密集奖励框架，将LLM/VLM生成的外在任务进展奖励与策略自确信度内在奖励相结合，无需人工奖励工程即可通过强化学习微调机器人基础策略，从而解决长时程任务中的误差累积问题。 |
| [^468] | [Quality-Controlled Active Learning via Gaussian Processes for Robust Structure-Property Learning in Autonomous Microscopy](https://arxiv.org/abs/2603.29135) | 提出了一种将好奇心驱动采样与基于简谐振子模型拟合的物理信息质量控制滤波器相结合的门控主动学习框架，可在自主显微镜的结构-性质学习任务中于数据采集阶段自动排除低质量测量数据，实现更鲁棒的学习性能。 |
| [^469] | [PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](https://arxiv.org/abs/2603.27101) | 本文提出PRUE方法，通过结合U-Net骨干网络、复合损失函数和针对性数据增强，在FTW基准上实现了76% IoU和47% object-F1，为大规模农田边界分割提供了比实例分割模型和地理空间基础模型更实用的解决方案。 |
| [^470] | [Process-Aware AI for Rainfall-Runoff Modeling: A Mass-Conserving Neural Framework with Hydrological Process Constraints](https://arxiv.org/abs/2603.25093) | 提出了一种质量守恒感知器（MCP）框架，通过在单一存储单元中逐步嵌入有界土壤蓄水、入渗、地表积水、地下水位动态等物理水文过程约束，在保证质量守恒的同时提升了降雨径流模拟的预测精度与物理可解释性。 |
| [^471] | [The Dual Mechanisms of Spatial Variable Binding in Vision-Language Models](https://arxiv.org/abs/2603.22278) | 视觉-语言模型通过两种并发机制表示空间变量绑定——语言模型中间层表示内容无关的空间关系但仅起次要作用，而视觉编码器产生的全局分布式空间信号才是塑造模型预测的主要来源。 |
| [^472] | [Federated Mixture-of-Experts Alignment on Mobile Edge Networks under Data Heterogeneity](https://arxiv.org/abs/2603.21276) | 针对数据异构环境下联邦MoE大模型微调中客户端门控偏好分歧和专家语义模糊两大挑战，本文提出了FedAlign-MoE联邦聚合对齐框架，以在移动边缘网络上实现更好的协同训练。 |
| [^473] | [Deep Time-Series Forecasting in 10 Years: A Survey](https://arxiv.org/abs/2603.19899) | 本文从自相关性建模的统一视角系统综述了十年来的深度时间序列预测研究，首次提出同时涵盖骨干架构与损失函数的分类体系，并据此剖析了现有文献的动机与洞见。 |
| [^474] | [Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks](https://arxiv.org/abs/2603.04364) | 该论文提出双模态多阶段对抗性安全训练（DMAST）框架，将代理与攻击者的交互建模为两人一般和马尔可夫博弈，通过三阶段协同训练显著增强多模态网页代理抵御同时污染视觉与文本两个观察通道的跨模态欺骗攻击的能力。 |
| [^475] | [World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models](https://arxiv.org/abs/2603.04317) | 该研究表明，以往从语言模型激活中解码出的“世界属性”在很大程度上也能由静态词嵌入实现，因此解码成功并不能证明语言模型形成了内部世界模型，而可能仅反映了语料库统计中的分布性关联。 |
| [^476] | [[b] = [d] - [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](https://arxiv.org/abs/2602.18899) | 自监督语音模型在96种语言的表示空间中将音系特征编码为线性向量，且这些向量支持算术运算（如将[d]-[t]得到的浊音向量加到[p]上即产生[b]），表明语音以可解释、可组合的音系向量形式被表征。 |
| [^477] | [Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders](https://arxiv.org/abs/2602.17050) | 该论文提出了一种基于线性探测的多探测零冲突哈希（MPZCH）索引机制，利用可配置探测与主动驱逐策略在保持生产级效率的同时彻底消除大规模推荐系统中的嵌入冲突，并防止陈旧嵌入继承以增强模型新鲜度。 |
| [^478] | [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983) | ReLoop通过结合结构化生成和行为验证，有效缩小了大语言模型在优化代码生成中的可行性与正确性差距。 |
| [^479] | [UniST-Pred: A Robust Unified Framework for Spatio-Temporal Traffic Forecasting in Transportation Networks Under Disruptions](https://arxiv.org/abs/2602.14049) | 提出了UniST-Pred统一时空预测框架，通过解耦时间建模与空间表示学习并采用自适应表示级融合，在交通网络中断等结构与观测不确定性条件下实现鲁棒的交通预测。 |
| [^480] | [RAM-Net: Linear-Time Sequence Modeling with Sparsely Addressable State](https://arxiv.org/abs/2602.11958) | RAM-Net提出用稀疏地址访问取代共享状态的密集访问，将循环状态组织为独立槽数组并通过地址解码器选择少量槽进行读写，从而抑制词元间干扰、提升长距离细粒度记忆能力，同时保持线性时间复杂度。 |
| [^481] | [Uncovering Cross-Objective Interference in Multi-Objective Alignment](https://arxiv.org/abs/2602.06869) | 该论文首次系统研究了多目标LLM对齐中的跨目标干扰现象，推导出解释其成因的局部协方差定律，并据此提出了缓解干扰的重加权控制器COVER。 |
| [^482] | [Embedding Perturbation may Better Reflect Intermediate-Step Uncertainty in LLM Reasoning](https://arxiv.org/abs/2602.02427) | 该研究提出用嵌入扰动来度量token敏感性，发现LLM推理中不确定的中间步骤更可能出现在对前文嵌入扰动高度敏感的token上，从而实现对推理过程中不确定性来源的精确定位。 |
| [^483] | [Local exponential stability of mean-field Langevin descent-ascent and associated particle system](https://arxiv.org/abs/2602.01564) | 本文证明了当初始条件足够接近混合纳什均衡时，熵正则化双人零和博弈的平均场朗之万下降-上升动力学以量化速率局部指数收敛，且有限粒子系统在N的指数长时间尺度上保持这一稳定性。 |
| [^484] | [Fast and Efficient Asynchronous Gossip Algorithm for Robust and Non-Smooth Convex Decentralized Learning](https://arxiv.org/abs/2601.20571) | 本文提出Goal-PD，一种异步Gossip原始-对偶算法，每个节点仅维护两个变量而与网络度数无关，实现了几乎必然收敛与线性收敛，并通过分布式均值估计中的成对平均特例与经典Gossip算法建立了直接联系。 |
| [^485] | [Intersectional Fairness via Mixed-Integer Optimization](https://arxiv.org/abs/2601.19595) | 本文提出一个基于混合整数优化（MIO）的统一框架，训练同时具备交叉公平性和内在可解释性的分类器，证明了两种交叉公平性度量（MSD 与 SPSF）在检测最不公平子群体上的等价性，并能将交叉偏见有效控制在可接受阈值以下。 |
| [^486] | [Stochastic Siamese MAE Pretraining for Longitudinal Medical Images](https://arxiv.org/abs/2512.23441) | STAMP提出了一种随机孪生MAE预训练框架，通过将MAE重建损失重构为条件变分推断目标、并以两次扫描间的时间差为条件，为纵向医学图像的自监督学习引入了时间感知能力和对疾病演变不确定性的建模。 |
| [^487] | [Rethinking Fine-Tuning: Unlocking Hidden Capabilities in Vision-Language Models](https://arxiv.org/abs/2512.23073) | 提出掩码微调（MFT），通过学习掩码在不修改骨干网络权重的前提下选择性路由预训练连接中的信息，为视觉-语言模型适配提供了优于全量微调和参数高效微调的结构化新方案。 |
| [^488] | [Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy](https://arxiv.org/abs/2512.20007) | 本文提出一种基于核化Stein差异的计算高效半参数拟合优度检验，并设计了无需重新拟合模型或从中采样的影响调整野自助法来确定检验的显著性水平。 |
| [^489] | [Machine learning Majorana topology using unsupervised and supervised learning](https://arxiv.org/abs/2512.13825) | 该论文将无监督与监督学习相结合，证明无需标签的模拟数据即可在现实短无序纳米线的马约拉纳分裂中区分拓扑相与平庸相，并确定两者在参数空间中的转变位置，为实验上识别拓扑提供了有用的工具。 |
| [^490] | [Variational Physics-Informed Ansatz for Reconstructing Hidden Interaction Networks from Steady States](https://arxiv.org/abs/2512.13708) | 提出一种变分物理信息拟设方法，将未知交互算子作为可训练对象并在异质扰动产生的多个稳态约束下最小化残差，实现从稳态观测中唯一重构隐藏交互网络，并给出基于相容矩阵秩的显式有限样本可辨识性条件。 |
| [^491] | [Optimal Transportation and Alignment Between Gaussian Measures](https://arxiv.org/abs/2512.03579) | 本文解决了可分离希尔伯特空间上非中心化高斯测度之间内积Gromov-Wasserstein对齐这一开放问题，通过等距算子上的二次优化给出精确变分刻画，并推导出紧密的解析上下界。 |
| [^492] | [A Footprint-Aware, High-Resolution Approach for Carbon Flux Prediction Across Diverse Ecosystems](https://arxiv.org/abs/2512.01917) | 该论文提出足迹感知回归（FAR）深度学习框架，通过同时预测涡度相关通量塔的空间足迹与像素级CO₂通量估计，消除了高分辨率升尺度模型中的偏差，并在205个站点年的AMERI-FAR25数据集上验证了其优于传统不考虑足迹的模型。 |
| [^493] | [Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning](https://arxiv.org/abs/2511.21075) | 本文发现生物医学文本具有与通用文本截然不同的密集认知不确定性结构，并据此提出双尺度的平衡微调方法，通过词元重加权与序列级重新分配使模型聚焦知识密集样本，在多项医学与生物任务上取得比SFT和DFT更一致的性能提升。 |
| [^494] | [Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization](https://arxiv.org/abs/2511.20718) | 该论文提出SORL方法，通过轮次级重要性采样与截断触发的归一化机制，使策略优化与多轮交互结构对齐并抑制不可靠的离策略梯度更新，从而稳定长时程LLM智能体的离策略强化学习训练，防止性能崩溃。 |
| [^495] | [Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation](https://arxiv.org/abs/2511.11500) | 该论文提出“强化犹豫”（RH）方法，通过将RLVR的二值奖励改为三值奖励（正确+1、弃答0、错误-λ），训练语言模型学会在不确定时主动弃答，从而在诚实性与准确性之间实现可调节的可信权衡。 |
| [^496] | [Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models](https://arxiv.org/abs/2511.00053) | 该论文提出了一种新颖的二次型加权学习目标，通过加权矩阵的非对角元素捕捉未来步骤间的标签自相关效应，同时利用非均匀对角元素为不同预测步骤设置异构任务权重，从而同时解决传统均方误差目标的两个缺陷，提升多步时间序列预测模型的训练效果。 |
| [^497] | [Time-multiplexed layer reuse for physical neural networks](https://arxiv.org/abs/2511.00044) | 提出TIDAL-Net架构，通过利用物理神经网络中快速前向动力学与缓慢权重调整之间的时间尺度分离进行时间复用层复用，从而绕过权重缓慢重调的瓶颈，使物理神经网络能以有限的物理参数实现接近深度网络的大规模训练。 |
| [^498] | [Action-Driven Processes for Continuous-Time Control](https://arxiv.org/abs/2510.26672) | 本文通过动作驱动过程统一了随机过程与强化学习的视角，证明最小化策略驱动分布与奖励驱动分布之间的KL散度等价于最大熵强化学习，并将其应用于脉冲神经网络。 |
| [^499] | [PyDPF: A Python Package for Differentiable Particle Filtering](https://arxiv.org/abs/2510.25693) | 本文提出了基于PyTorch框架构建的Python软件包PyDPF，通过统一的API实现了多种可微粒子滤波算法，使这些方法更易于被广大研究群体使用。 |
| [^500] | [DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment](https://arxiv.org/abs/2510.24574) | 提出DistDF框架，通过最小化一种可证明上界于条件分布差异的联合分布Wasserstein差异来对齐预测与标签序列的分布，解决了传统均方误差在标签序列存在自相关时的估计偏差问题。 |
| [^501] | [Beyond the Semicircle: Free Diffusion Models with Prescribed Equilibria](https://arxiv.org/abs/2510.22778) | 该论文发现状态依赖的自由波动率能突破常系数自由扩散只能收敛到半圆律的固有局限，并为任意充分正则的紧支撑目标谱分布显式构造出具有指定平衡态的自由扩散模型。 |
| [^502] | [Predicting kernel regression learning curves from only raw data statistics](https://arxiv.org/abs/2510.14878) | 该论文提出 Hermite 特征结构假设（HEA），证明仅用经验协方差矩阵和目标函数的多项式分解这两个原始数据统计量，即可在真实图像数据集上准确预测核回归的学习曲线。 |
| [^503] | [Qubit-centric Transformer for Surface Code Decoding](https://arxiv.org/abs/2510.11593) | 提出了一种基于Transformer的新型量子纠错解码器QCT，利用以量子比特为中心的注意力机制和融合量子码拓扑结构的图掩码方法，将伴随式转换为量子比特标记以有效识别逻辑错误。 |
| [^504] | [HARL-A: An Extensible Benchmark Framework for Heterogeneous Multi-Agent Adversarial Reinforcement Learning in IsaacLab](https://arxiv.org/abs/2510.01264) | 该论文提出了HARL-A，一个基于IsaacLab的开源可扩展基准测试框架，支持在任意数量团队和任意机器人形态组合下进行异构多智能体对抗强化学习的可扩展训练与评估。 |
| [^505] | [PiERN: Token-Level Routing for Integrating High-Precision Computation and Reasoning](https://arxiv.org/abs/2509.18169) | PiERN提出了一种物理隔离专家路由网络架构，通过在令牌级别路由计算与推理，使大型语言模型能够在单一思维链中迭代交替地执行高精度数值计算与推理，在准确率、响应延迟、令牌消耗和能耗等方面均优于直接微调与主流多智能体方法。 |
| [^506] | [DRtool: An Interactive Tool for Analyzing High-Dimensional Clusterings](https://arxiv.org/abs/2509.04603) | 本文开发了交互式工具DRtool，通过新的聚类验证技术（包括可视化评估和假设检验）帮助分析师识别高维数据中容易被忽视的过度聚类问题。 |
| [^507] | [Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators](https://arxiv.org/abs/2509.03647) | 本文提出利用通过对比激活添加（CAA）和优化方法构建的转向向量，在无需重新训练的情况下于推理时缓解LLM评估器的不合理自我偏好偏差，最多可降低97%，显著优于提示和直接偏好优化基线。 |
| [^508] | [Practical Feasibility of Gradient Inversion Attacks in Federated Learning](https://arxiv.org/abs/2508.19819) | 本文系统评估了梯度反演攻击在现实联邦学习系统中的可行性，发现现代性能优化的视觉模型能够有效抵御有意义的图像重构，而已报告的攻击成功多依赖于理想化的上界实验设置。 |
| [^509] | [AEGIS: Runtime-Guided GPU Collocation for Multi-Tenant Deep Learning Training](https://arxiv.org/abs/2508.19073) | AEGIS是一个服务器规模的运行时调度系统，通过在单一调度循环中集成内存可行性评估、部署后观察、运行时压力过滤和OOM感知恢复机制，实现了多租户深度学习训练任务在共享GPU服务器上的安全高效共置。 |
| [^510] | [PuzzleJAX: A Benchmark for Reasoning and Learning](https://arxiv.org/abs/2508.16821) | PuzzleJAX是一个GPU加速的益智游戏引擎与描述语言，可动态编译PuzzleScript风格的游戏，从而为树搜索、强化学习和LLM推理能力提供大规模、多样化任务的快速基准测试。 |
| [^511] | [Cross-Modality Controlled Molecule Generation with Diffusion Language Model](https://arxiv.org/abs/2508.14748) | 提出模块化框架CMCM-DLM，通过结构控制模块和性质控制模块的分阶段协同设计，使预训练扩散语言模型无需重新训练即可支持分子结构与化学性质等跨模态异构约束的受控分子生成。 |
| [^512] | [Direct Regret Optimization in Bayesian Optimization](https://arxiv.org/abs/2507.06529) | 提出了一种直接遗憾优化新方法，通过从候选模型与采集函数中蒸馏并利用高斯过程集成生成模拟BO轨迹，训练端到端的决策Transformer联合学习最优模型与非短视采集策略，从而显式最小化多步遗憾。 |
| [^513] | [Improving Mixup Calibration with Wasserstein Distributionally Robust Optimization](https://arxiv.org/abs/2506.17874) | 本文提出DRO-Augment框架，将Wasserstein分布鲁棒优化与Mixup数据增强相结合，有效缓解了腐蚀鲁棒性与模型校准之间的权衡，在保持腐蚀准确率的同时显著降低了期望校准误差（ECE）。 |
| [^514] | [Time-o1: Time-Series Forecasting Needs Transformed Label Alignment](https://arxiv.org/abs/2505.17847) | Time-o1通过将标签序列变换为去相关且区分显著性的分量，训练模型对齐最显著分量，从而提出一种变换增强的损失函数，有效缓解标签自相关性并减少任务数量，在时间序列预测中达到最先进性能且兼容多种预测模型。 |
| [^515] | [KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches](https://arxiv.org/abs/2505.14777) | 提出了一种基于动理学与偏微分方程的即插即用优化器KO，它将参数动力学建模为粒子系统并通过离散化玻尔兹曼输运方程引入随机相互作用，从而提升参数多样性、缓解权重凝聚并保持收敛保证。 |
| [^516] | [Boosting Large Language Models with Mask Fine-Tuning](https://arxiv.org/abs/2503.22764) | 提出掩码微调（MFT）这一新颖的大语言模型微调范式，通过学习并应用二值掩码、在不更新模型权重的情况下精心打破模型结构完整性，从而在不同领域和骨干网络上获得一致的性能提升。 |
| [^517] | [Causal Effect Estimation under Networked Interference without Networked Unconfoundedness Assumption](https://arxiv.org/abs/2502.19741) | 本文提出一种利用网络中单元间交互模式来恢复三类潜在混杂因素的框架，从而在不依赖网络无混杂性假设的条件下实现网络干扰下的因果效应估计。 |
| [^518] | [Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study](https://arxiv.org/abs/2501.18158) | 本文首次将大语言模型应用于比特币等真实加密货币交易图分析，提出三层能力评估框架、人类可读的图表示格式LLM4TG和连接增强的交易图采样算法，克服了传统黑盒模型难以解释和捕捉细微行为模式的局限。 |
| [^519] | [Identifiability Analysis of Linear ODE Systems with Hidden Confounders](https://arxiv.org/abs/2410.21917) | 本文系统分析了含隐藏混杂因素的线性常微分方程系统的可辨识性，分别研究了潜在混杂因素无因果关系但遵循特定函数形式（如时间多项式）演化，以及潜在混杂因素之间具有由有向无环图描述的因果依赖关系这两种情况，填补了该领域的空白。 |
| [^520] | [When Explanations Compete: Policy-Aware Selection Under Uncertainty](https://arxiv.org/abs/2410.05479) | 该论文提出了一个策略感知的解释选择框架，通过结合资格规则、双向帕累托筛选和策略感知排序，从不确定性感知解释方法生成的多个候选解释中，依据预测置信度、不确定性和应用约束进行最优选择。 |
| [^521] | [Convergence of Sharpness-Aware Minimization Algorithms using Increasing Batch Size and Decaying Learning Rate](https://arxiv.org/abs/2409.09984) | 本文从理论上证明了使用递增批大小或衰减学习率（如余弦退火、线性学习率）的GSAM算法的收敛性，并通过数值实验表明递增批大小相比恒定批大小能获得更低的最坏情况ℓ∞自适应锐度。 |
| [^522] | [Diffusion Model-Based Video Editing: A Survey](https://arxiv.org/abs/2407.07111) | 本文是一篇综述，系统梳理了基于扩散模型的视频编辑技术的理论基础、方法分类与演化脉络，探讨了点编辑、姿态引导人体视频编辑等新兴应用，并提出新的V2VBench基准对该领域进行全面对比评估。 |
| [^523] | [Data-driven measures of high-frequency trading](https://arxiv.org/abs/2405.08101) | 该论文利用在纳斯达克专有数据上训练的机器学习模型，首次生成了2010至2023年美国股票的流动性提供型与需求型高频交易日度度量指标，并发现供给端高频交易与更多信息获取、更多知情交易和更低买卖价差相关。 |
| [^524] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |
| [^525] | [Fast, Interpretable, and Deterministic Time Series Classification With a Bag-of-Receptive-Fields](https://arxiv.org/abs/2311.18029) | 本文提出了BORF（感受野袋），一种快速、可解释且确定性的时间序列分类变换方法，克服了现有黑盒分类器难以理解以及可解释方法因依赖随机化导致解释不稳定的问题。 |
| [^526] | [Probabilistic Truly Unordered Rule Sets.](http://arxiv.org/abs/2401.09918) | 本论文提出了概率性真正无序规则集（TURS）方法，用于解决规则集学习中的三个缺点：强加顺序、重叠冲突和多类别目标分类问题。通过利用规则集的概率特性来解决重叠冲突，并形式化定义学习问题。 |

# 详细

[^1]: QF3：基于滤波Q梯度的快速流强化学习

    QF3: Fast Flow RL with Filtered Q-Gradients

    [https://arxiv.org/abs/2610.08789](https://arxiv.org/abs/2610.08789)

    QF3提出了一种通过滤波Q梯度训练流策略的离线策略强化学习算法，比FPO++快10倍，并且是首个能从零开始训练人形机器人运动策略并零样本迁移到真实硬件的离线策略流RL方法。

    

    流策略已成为从演示中学习机器人行为的标准策略类别，但强化学习对于改进预训练的流策略或通过交互从零开始学习流策略仍然至关重要。我们提出了QF3（Fast Flow RL with Filtered Q-Gradients，基于滤波Q梯度的快速流强化学习），这是一种在线离线策略（off-policy）强化学习算法，它通过流匹配加上评论家（critic）网络的动作梯度来训练流策略，该梯度通过流输出的一步预测进行反向传播。为了确保更新仅在评论家网络和该预测可靠的区域内进行，QF3只将评论家梯度应用于保持在重放动作附近的行为维度。据我们所知，QF3是首个从零开始训练人形机器人运动策略并将其零样本迁移到真实硬件上的离线策略流RL方法。配合高吞吐量的离线策略训练方案，它以比FPO++快10倍的实际训练速度训练人形机器人运动和动作跟踪策略（原文在此处截断）。

    arXiv:2610.08789v1 Announce Type: cross  Abstract: Flow policies have become a standard policy class for learning robot behaviors from demonstrations, but reinforcement learning is still critical for improving pre-trained flow policies or learning them from scratch through interaction. We introduce QF3 (Fast Flow RL with Filtered Q-Gradients), an online off-policy RL algorithm that trains a flow policy with flow matching plus the critic's action gradient, backpropagated through a one-step prediction of the flow's output. To keep updates where the critic and this prediction are reliable, QF3 applies the critic gradient only to action dimensions that stay near the replay action. To our knowledge, QF3 is the first off-policy flow RL method to train humanoid locomotion policies from scratch and transfer them zero-shot to hardware. Paired with a high-throughput off-policy training recipe, it trains humanoid locomotion and motion-tracking policies with a 10x wall-clock speedup over FPO++, a 
    
[^2]: 共形预测集量化信息增益：一个理论视角

    Conformal Prediction Sets Quantify Information Gain: A Theoretical Perspective

    [https://arxiv.org/abs/2610.08785](https://arxiv.org/abs/2610.08785)

    本文为“以共形预测集大小度量不确定性”提供了信息论基础，引入了一族基于共形预测集大小与覆盖的广义信息度量，并证明额外信息导致的共形集缩减被这些度量夹界且满足数据处理不等式。

    

    共形预测是一种流行的量化不确定性的工具，它输出具有有限样本覆盖保证的预测集。虽然预测集的大小通常被用作不确定性的启发式度量，但这一解释背后的信息论基础仍知之甚少。在本工作中，我们借助一种针对集合值预测而量身定制的决策论熵推广，为这一解释提供了理论基础。特别地，我们引入了一族基于共形预测集的大小与覆盖范围的广义信息度量。值得注意的是，香农互信息可以用这些度量获得精确的积分表示。随后我们证明，在标准的分类设置中，额外信息所带来的共形集大小的缩减 (i) 被夹在该族中依赖于校准的成员之间，并且 (ii) 遵从数据处理不等式，两者均限于有限样本校准与……

    arXiv:2610.08785v1 Announce Type: new  Abstract: Conformal prediction is a popular tool for uncertainty quantification that outputs prediction sets with finite-sample coverage guarantees. While prediction set size is commonly used as a heuristic measure of uncertainty, the information-theoretic basis for this interpretation remains poorly understood. In this work, we provide such a foundation using a decision-theoretic generalization of entropy tailored to set-valued prediction. In particular, we introduce a family of generalized information measures based on the size and coverage of conformal prediction sets. Notably, Shannon mutual information admits an exact integral representation in terms of these measures. We then show that, in standard classification settings, the reduction in conformal set size from additional information (i) is sandwiched between calibration-dependent members of this family and (ii) obeys a data processing inequality, both up to finite-sample calibration and m
    
[^3]: AdvSim2Real：在网络世界模型中训练网络智能体以对抗自适应提示注入

    AdvSim2Real : Training Web Agents Against Adaptive Prompt Injection in a Web World Model

    [https://arxiv.org/abs/2610.08773](https://arxiv.org/abs/2610.08773)

    提出 AdvSim2Real，在冻结的网络世界模型中让任务课程、注入攻击者与智能体共同演化，通过“成功翻转”对抗奖励机制训练出既更强又更能抵御自适应提示注入攻击的网络智能体。

    

    arXiv:2610.08773v1 公告类型：cross 摘要：网络智能体通过阅读并操作由第三方编写的网页来完成用户请求，因此页面上被植入的指令可能会使智能体偏离用户的目标。然而智能体不能简单地忽略页面，因为页面中还包含任务所需的取值和控件。当前的防御方法是在训练前固定注入内容并对智能体进行微调，但适应了已训练模型的自适应攻击者可以绕过这些防御。对抗性训练虽然允许攻击者进行适应，但任务保持固定，因此一旦智能体解决了某个任务，该任务就不再具有训练价值。我们提出 AdvSim2Real，它在一个冻结的网络世界模型中共同演化任务课程、注入攻击者和智能体。任务课程因智能体约有半数概率能解决的任务而获得奖励，而攻击者仅因“成功翻转”——即能把被判定为成功的任务转变为失败的注入——而获得奖励。在模拟器中训练使一个 4B 的智能体在能力和鲁棒性方面都得到提升：其完成……

    arXiv:2610.08773v1 Announce Type: cross  Abstract: Web agents complete user requests by reading and acting on pages that third parties write, so an instruction planted on a page can redirect the agent away from the user's goal. The agent cannot simply ignore the page, because the page also holds the values and controls the task requires. Current defenses fine-tune the agent on injections fixed before training, and attackers that adapt to the trained model bypass them. Adversarial training lets the attacker adapt but keeps the tasks fixed, so a task stops teaching once the agent solves it. We introduce AdvSim2Real, which co-evolves a task curriculum, an injection adversary, and the agent inside a frozen web world model. The curriculum is rewarded for tasks the agent solves about half of the time, and the adversary only for a success flip, an injection that turns a judged success into a failure. Training in the simulator makes a 4B agent both more capable and more robust: its completion 
    
[^4]: 具有不受限制空间变化反扩散项的Kuramoto--Sivashinsky方程的快速Fredholm镇定

    Rapid Fredholm stabilization of the Kuramoto--Sivashinsky equation with unrestricted, spatially-varying anti-diffusion

    [https://arxiv.org/abs/2610.08764](https://arxiv.org/abs/2610.08764)

    本文首次提出了针对具有任意空间变化反扩散系数的Kuramoto-Sivashinsky方程的快速镇定反馈设计，通过引入第二个边界输入并借鉴Heymann引理的思想，克服了单输入Fredholm设计中因重复不稳定特征值导致的可控性丧失问题。

    

    我们开发了首个针对具有空间变化反扩散系数的Kuramoto--Sivashinsky方程快速镇定的反馈设计。对于常系数情形，Coron和Lü（2015）的单输入Fredholm设计排除了一组离散取值，在这些取值处重复出现的不稳定特征值会导致可控性丧失。我们通过引入第二个边界输入并为两个输入分配不同的角色来克服这一障碍。受Heymann引理启发的关键思想是，将边界值 $u(0,t)$ 完全用于一个预反馈，使修改后的系统可通过曲率输入 $u_{xx}(0,t)$ 实现可控。随后，该曲率输入通过Fredholm反步变换使系统得到镇定。我们证明两个输入足以实现可控性，且当系统存在不稳定的二重特征值时，两个输入是必要的。然而，Fredholm核在实现中仍需进行近似。因此，为了……

    arXiv:2610.08764v1 Announce Type: cross  Abstract: We develop the first feedback design for rapid stabilization of the Kuramoto--Sivashinsky equation with a spatially varying anti-diffusion coefficient. For constant coefficients, the single-input Fredholm design of Coron and L\"u (2015) excludes a discrete set of values at which repeated unstable eigenvalues cause a loss of controllability. We overcome this obstruction by introducing a second boundary input and assigning the two inputs distinct roles. The key idea, inspired by Heymann's Lemma, is to use the boundary value $u(0,t)$ entirely for a pre-feedback that renders the modified plant controllable through the curvature input $u_{xx}(0,t)$. The latter input then stabilizes the plant through a Fredholm backstepping transformation. We show that two inputs suffice for controllability and are necessary when the plant has an unstable double eigenvalue. However, the Fredholm kernel still must be approximated for implementation. Hence, to
    
[^5]: 用于化学反应的神经Petri流

    Neural Petri flows for chemical reactions

    [https://arxiv.org/abs/2610.08750](https://arxiv.org/abs/2610.08750)

    本文提出一种在任意权重取值下都严格保持Petri网语义的神经架构用于化学反应建模，并从理论上证明守恒性决定了触发形式、非负性决定了使能规则，从而只将速率定律留给神经网络自由学习。

    

    Petri网已被用于描述化学反应等化学过程，它们与化学的映射非常自然：库所对应原子间的化学键及每个原子的自由价，令牌是键级的单位，变迁形成或断裂化学键，守恒量是原子的价电子预算，而使能规则即价规则。这些语义在化学反应的学习模型中、或在把Petri网仅用作消息传递骨架的神经网络中都无法得到保证。本文提出的问题是：什么样的架构能在其权重的任意取值下仍然是一个Petri网？我们在理论中找到了答案：一个网的所有语义都共享变迁触发形式 $m'=m+C\sigma$、局部性（使能判断只读取变迁的输入）以及使能规则；并且我们证明了守恒性强制了触发形式，而非负性在局部速率定律上强制了使能规则。这就使得速率定律得以自由设定，可由神经网络学习。（原文摘要在此处中断）

    arXiv:2610.08750v1 Announce Type: new  Abstract: Petri nets have been used to describe chemical processes such as reactions.They map well to chemistry: Places are the bonds between atoms and the free valence of each atom, a token is a unit of bond order, a transition forms or breaks a bond, the conserved quantities are the valence budgets of the atoms, and the enabling rule is the valence rule. These semantics are not guaranteed by learned models of reactions or neural networks that are built on Petri nets that use the net as a scaffold for message passing. Here, we ask what architecture remains a Petri net for every value of its weights. We find the answer in the theory, where all semantics of a net share the firing form $m^\prime=m+C\sigma$, locality, as enabling reads only the inputs of a transition, and the enabling rule, and we prove that conservation forces the firing form and that non-negativity forces the enabling rule on local rate laws. This leaves free the rate law, which is
    
[^6]: 精确滑动窗口约束下的线性老虎机问题

    Linear Bandits under Exact Sliding-Window Constraints

    [https://arxiv.org/abs/2610.08745](https://arxiv.org/abs/2610.08745)

    该论文研究了精确滑动窗口约束下的线性老虎机问题，提出了刻画可行可达性的转移直径 $\tau$，证明了离线平稳解的最优性条件与在线次线性遗憾的不可能性，并开发了相对于离线最优可行轨迹遗憾为 $\widetilde{O}(d\sqrt{T}+\tau d+w)$ 的稀有切换 OFUL 算法。

    

    我们研究精确滑动窗口约束下的线性老虎机问题，其中每个连续的动作块都必须属于一个预先指定的可行集。在奖励函数已知的离线设置中，我们证明了当 $w\mid T$ 时，凸性和循环平移不变性使平稳解达到最优，否则与最优解的差距在加性 $O(w)$ 之内。在线上设置中，我们证明仅凭几何结构不足以支持学习，次线性遗憾可能无法实现。我们引入了一个量化可行可达性的转移直径 $\tau$，并开发了一种稀有切换 OFUL 算法，其相对于离线最优可行轨迹的遗憾为 $\widetilde{O}(d\sqrt{T}+\tau d+w)$。最后，我们去除循环不变性，考虑一般的滑动窗口约束，此时最优行为可能是非平稳的。我们将最近的动作历史表示为有限记忆控制问题的状态，并引入了历史状态

    arXiv:2610.08745v1 Announce Type: new  Abstract: We study linear bandits under exact sliding-window constraints, where every consecutive block of actions must belong to a prescribed feasible set. In the offline setting, where the reward function is known, we show that convexity and cyclic-shift invariance make a stationary solution optimal when $w\mid T$ and within an additive $O(w)$ gap otherwise. In the online setting, we show that geometric structure alone is insufficient for learning, and sublinear regret can be impossible. We introduce a transition diameter $\tau$ that quantifies feasible reachability and develop a rare-switching OFUL algorithm with regret $\widetilde{O}(d\sqrt{T}+\tau d+w)$ against the offline-optimal feasible trajectory. Finally, we remove cyclic invariance and consider general sliding-window constraints, where optimal behavior may be non-stationary. We represent recent action history as the state of a finite-memory control problem and introduce a history-state 
    
[^7]: 具有保形动作集的强化学习：在序列推荐中的应用

    Reinforcement Learning with Conformal Action Sets: An Application to Sequential Recommendation

    [https://arxiv.org/abs/2610.08743](https://arxiv.org/abs/2610.08743)

    该论文提出了RLCP方法，通过评论家分数和在线阈值自适应调整序列推荐中的动作集大小，并从理论上证明了代理未命中率上界以及将价值损失精确分解为过滤损失和选择损失，从而获得无需参数收敛的有限会话奖励上界。

    

    序列推荐系统通常使用固定的推荐列表大小，尽管一个会话中有用的备选方案数量是动态变化的。我们提出了带校准剪枝的强化学习方法（RLCP），该方法利用评论家分数和在线阈值来自适应调整保留的动作集。该阈值根据二值反馈进行更新，该反馈用于指示集合中是否包含代理目标中的某个动作。我们证明了沿自适应轨迹观测到的代理未命中率存在确定性上界。为了量化剪枝对奖励的影响，我们将价值损失精确分解为过滤损失和选择损失。在显式的代理和评论家近似条件下，该分解给出了一个有限的会话奖励上界，该上界同时考虑了不完美的选择和集合截断，而无需学习参数收敛。我们在KuaiRand-Pure和MovieLens 1M数据集上的实验将两种RLCP实现与四个强化学习基线进行了比较。

    arXiv:2610.08743v1 Announce Type: cross  Abstract: Sequential recommenders typically use a fixed slate size even though the number of useful alternatives changes within a session. We propose Reinforcement Learning with Calibrated Pruning (RLCP), which adapts the retained action set using critic scores and an online threshold. The threshold is updated from binary feedback indicating whether the set contains an action in a proxy target. We prove a deterministic bound on the observed proxy miss rate along adaptive trajectories. To quantify the effect of pruning on reward, we derive an exact decomposition of value loss into filtering and selection losses. Under explicit proxy and critic approximation conditions, this decomposition yields a finite session reward bound that also accounts for imperfect selection and set truncation, without requiring the learning parameters to converge. Experiments on KuaiRand-Pure and MovieLens 1M compare two RLCP implementations with four RL baselines. In ea
    
[^8]: 论稳健赌博机问题的计算可处理性

    On the Computational Tractability of Robust Bandits

    [https://arxiv.org/abs/2610.08740](https://arxiv.org/abs/2610.08740)

    本文在稳健赌博机框架中识别出一个可用多项式时间算法求解且遗憾为 Õ(√T) 的特殊情形，并证明其若干微小推广均为 NP难，表明该特殊情形正处于计算可解性的边界。

    

    当环境不属于学习者的假设类时，学习问题通常依靠不可知学习保证来处理。然而，对于监督学习之外的任何问题，不可知学习的保证都难以获得。最近，不精确赌博机（Kosoy, 2025）（后在 Appel 和 Kosoy, 2025 中更名为稳健赌博机）被引入，作为赌博机设定中处理不可实现学习的另一种方法，并且已针对一大类问题证明了 Θ(√T) 遗憾的学习器。然而，此前没有提供任何计算效率方面的保证。在本文中，我们识别出一个特殊情形，该情形允许存在一个具有 Õ(√T) 遗憾的多项式时间学习器。我们还证明了对该特殊情形的若干微小推广都是 NP难的，这表明该特殊情形恰好处于可解问题的边界上。最近有观点提出（Kosoy, 2018），针对不可实现学习问题的高效学习器对于……至关重要

    arXiv:2610.08740v1 Announce Type: new  Abstract: Learning when the environment does not belong to the learner's hypothesis class is typically handled using agnostic learning guarantees. However, for anything beyond supervised learning, agnostic guarantees are difficult to come by. Recently, imprecise bandits (Kosoy, 2025) (later renamed to robust bandits in Appel and Kosoy, 2025) were introduced as another approach to unrealizable learning in the bandits setting and a $\Theta(\sqrt{T})$ regret learner was shown for a large class. However, no computational guarantees were provided. In this paper we identify a special case that admits a polynomial-time learner with $\tilde{O}(\sqrt{T})$ regret. We also show that several small generalizations of this special case are NP-hard thus indicating that the special case is at the boundary of what is tractable. It has been recently suggested (Kosoy, 2018) that computationally efficient learners for unrealizable learning problems are crucial for so
    
[^9]: 去噪层次化表示：面向语言建模的联合连续扩散

    Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling

    [https://arxiv.org/abs/2610.08738](https://arxiv.org/abs/2610.08738)

    提出层次化连续扩散语言模型（H-CDLM），通过并行扩散token本身及其粗粒度语义聚类等多种模态表示，并允许为各模态配置独立的采样器与调度，以极少的计算和参数开销显著提升连续扩散语言模型的性能。

    

    扩散语言模型有望实现与顺序无关的并行文本生成。近来，连续扩散和流匹配模型取得了显著进展，这得益于精心设计的token表示和扩散/流空间。在本工作中，我们提出了层次化连续扩散语言模型，这是一个简洁的框架，能够以极少的计算和参数开销进一步提升连续扩散语言模型的性能。借鉴离散扩散语言模型与连续图像扩散文献中关于联合扩散的思想，我们对多种模态进行并行扩散。这些模态在不同语义粒度上表示token：在我们的具体实现中，包括token本身以及通过对预训练token嵌入进行聚类所得到的更粗粒度的聚类。我们提出了一个通用设置，允许为每种模态配置各自的采样器和调度，以增强模态之间的相互作用。将该方法应用于CoBit时，得到了H-CoBit，它带来了巨大的性能提升。

    arXiv:2610.08738v1 Announce Type: new  Abstract: Diffusion Language Models (DLMs) hold the promise of order-agnostic, parallel text generation. Recently, continuous diffusion and flow matching models have seen substantial gains, driven by carefully crafted token representations and diffusion/flow spaces. In this work, we introduce Hierarchical Continuous Diffusion Language Models (H-CDLMs), a simple framework that further improves continuous DLMs with minimal compute and parameter overhead. Drawing on the discrete DLM and continuous image diffusion literature on joint diffusion, we diffuse multiple modalities in parallel. These modalities represent tokens at different semantic granularities: in our instantiation, the tokens themselves and coarser clusters obtained by clustering pretrained token embeddings. We propose a general setup that allows per-modality samplers and schedules to enhance the interplay between modalities. Applied to CoBit, this yields H-CoBit, which delivers large em
    
[^10]: 最优且高效的在线逆优化

    Optimal and Efficient Online Inverse Optimization

    [https://arxiv.org/abs/2610.08735](https://arxiv.org/abs/2610.08735)

    本文提出首个多项式时间的确定性算法，在在线逆线性优化中对任意时域均达到最优的 $O(\sqrt{d})$ 遗憾界，肯定地回答了 Sakaue 提出的公开问题。

    

    在在线逆线性优化中，学习器推荐一个动作，随后观察专家的选择，该专家在 $\mathbb{R}^d$ 上最大化一个固定但未知的线性目标函数；目标是在不观测该目标函数的情况下学会对其进行优化。Sakaue 最近用一种随机算法得到了最优遗憾 $O(\sqrt{d})$，但该算法每轮需要进行 $(dT)^{O(d)}$ 次线性优化，并提出了是否能在多项式时间内达到该遗憾界的问题。我们给出了肯定的回答：我们的确定性算法对任意时域 $T$ 都具有 $O(\sqrt{d})$ 的遗憾，且运行时间是关于 $d$ 和 $T$ 的多项式。该算法是 Sakaue 等人与 Cai 等人的变度量算法的一个变体，其中当查询点离开度量更新发生的位置足够远时，该度量更新就会被撤销。

    arXiv:2610.08735v1 Announce Type: new  Abstract: In online inverse linear optimization, a learner recommends an action and then observes the choice of an expert who maximizes a fixed, unknown linear objective on $\mathbb{R}^{d}$; the goal is to learn to optimize this objective without observing it. Sakaue recently obtained the optimal regret $O(\sqrt d)$ with a randomized algorithm making $(dT)^{O(d)}$ linear optimizations per round, and asked whether it can be attained in polynomial time. We answer positively: our deterministic algorithm has regret $O(\sqrt d)$ for every horizon $T$ and runs in time polynomial in $d$ and $T$. It is a variant of the variable-metric algorithms of Sakaue et al.\ and Cai et al., in which a metric update is revoked once the query point moves far enough from where the update was made.
    
[^11]: 智能体的历史能否告诉你上下文压缩何时会造成伤害？基于TRACE成对重放语料库的适度且有界的影响分析

    Does an Agent's History Tell You When Compaction Will Hurt? A Modest, Bounded Effect on the TRACE Paired-Replay Corpus

    [https://arxiv.org/abs/2610.08722](https://arxiv.org/abs/2610.08722)

    本研究利用TRACE语料库中590个成对重放的压缩边界，检验智能体近期历史能否预测上下文压缩的危害，结果发现预测能力仅微弱有效——按前缀位置的预设对比为零结果，最佳可解释触发器也仅能避免21%的有害压缩边界。

    

    许多长时程智能体按照全局规则压缩其上下文，通常是基于令牌预算，而不考虑智能体当前正在执行的任务。我们提出的问题是：智能体最近的行为能否预测压缩何时会造成伤害？TRACE公开语料库包含590个由框架触发的AppWorld压缩边界，它从重新执行的前缀状态出发，分别在压缩前上下文和摘要两种条件下重放每个边界，并记录后续动作的负担，即出错的调用或重复已执行调用的调用。我们发现，边界前的历史仅能微弱地预测压缩后的伤害。一个内部预先指定的、按前缀位置进行对比的实验结果是一个较宽的零结果，而其背后朴素的“已写入”标签实际上衡量的是轨迹所处的阶段。最佳的扩展协议触发器在留出集上达到AUROC 0.66（在复现集自身标签上为0.64），而同边界复现的AUROC为0.72；最佳的冻结式、可解释触发器能够避免21%的有害（正负担）压缩边界。

    arXiv:2610.08722v1 Announce Type: new  Abstract: Many long-horizon agents compact their context on a global rule, usually a token budget, blind to what the agent was doing. We ask whether the agent's recent behaviour predicts when a compaction will hurt. TRACE's public corpus of 590 harness-triggered AppWorld compaction boundaries replays each boundary from a re-executed prefix state under the pre-compaction context and under the summary, and records the burden of the next actions: calls that error or repeat a call already made. We find that pre-boundary history predicts post-compaction harm only weakly. An internally prespecified contrast by prefix placement is a wide null, and the naive "has-written" label behind it turns out to measure trajectory phase. The best extension-protocol trigger reaches held-out AUROC 0.66 (0.64 on the replicate's own label) against a same-boundary replicate of 0.72; the best frozen, interpretable trigger avoids 21% of harmful (positive-burden) boundaries 
    
[^12]: 当遗忘并非灾难性时：论虚假遗忘的机制

    When Forgetting is not Catastrophic: On the Mechanics of Spurious Forgetting

    [https://arxiv.org/abs/2610.08718](https://arxiv.org/abs/2610.08718)

    该研究揭示了语言模型微调中虚假遗忘的力学机制——微调使旧表征沿共同方向偏移导致暂时的知识隐藏，归一化会撤销该偏移使知识自行恢复，而只有事实特定的累积变化才会造成真正的永久遗忘。

    

    语言模型在微调过程中似乎遗忘的知识往往仍存储在模型中且可以被恢复，这种现象被称为虚假遗忘。在新事实数据上进行微调甚至会产生一种会自我逆转的遗忘：对旧事实的召回能力先崩溃，随后随着仅在新事实上的继续训练而恢复，之后才最终永久性衰退。我们试图理解这种遗忘何时不是灾难性的。一个最小化的联想记忆模型仅用三个要素就再现了这些动态：具有共享结构的键、集中的新值以及网络中的归一化机制。微调使所有旧表征沿着一个共同方向移动，从而在保持其相对几何结构的同时隐藏旧事实；一旦新事实被学会，归一化机制会撤销这种偏移，而事实特定的变化则会不断累积并最终导致衰退。此外，减去这一共同偏移可以消除在合成数据上训练的Transformer中的崩溃现象，移除……

    arXiv:2610.08718v1 Announce Type: new  Abstract: Knowledge that a language model appears to forget during finetuning often remains stored and can be recovered, a phenomenon called spurious forgetting. Finetuning on new facts can even produce forgetting that undoes itself: recall of the old facts collapses, recovers as training continues on new facts alone, and only then erodes for good. We seek to understand when such forgetting is not catastrophic. A minimal associative memory reproduces these dynamics with three ingredients: keys with shared structure, concentrated new values, and normalization in the network. Finetuning moves all old representations along a common direction, hiding the old facts while preserving their relative geometry; normalization withdraws this shift once the new facts are learned, whereas fact-specific changes accumulate and cause the erosion. Moreover, subtracting the common shift eliminates the collapse in a Transformer trained on synthetic data, and removing
    
[^13]: 通过路径-流对齐实现路径与流的协同演化

    Co-Evolving Paths and Flows via Path-Flow Alignment

    [https://arxiv.org/abs/2610.08717](https://arxiv.org/abs/2610.08717)

    该论文提出以路径-流对齐作为流匹配的统一训练目标，让路径网络与流网络在共享的对齐损失下协同演化，并针对由此发现的“路径过拟合”失败模式（由低熵瓶颈引起）引入随机路径正则化器，从而实现更可靠的路径学习与更好的生成质量。

    

    我们将路径-流对齐作为流匹配的一种统一训练目标进行研究。不同于固定插值路径而仅学习速度场的做法，我们使用相同的对齐损失联合训练一个保持端点的路径网络和一个流网络：流学习去匹配路径速度，而路径学习将其速度对齐到当前流。尽管每个固定的已学习路径都定义了一个有效的流匹配目标，但仅凭对齐损失并不是路径学习的可靠准则。我们识别出一种被称为“路径过拟合”的失败模式，即在对齐损失下降的同时样本质量反而恶化。我们发现这种失败与诱导概率路径中的低熵瓶颈有关，即学习到的路径会使样本经过过度集中的中间边缘分布。基于这一诊断，我们引入了一个随机路径正则化器，该正则化器对路径隐藏部分源信息……（原文摘要在此处截断）

    arXiv:2610.08717v1 Announce Type: cross  Abstract: We study path-flow alignment as a unified training objective for flow matching. Instead of fixing the interpolation path and learning only the velocity field, we jointly train an endpoint-preserving path network and a flow network using the same alignment loss: the flow learns to match the path velocity, and the path learns to align its velocity to the current flow. Although every fixed learned path defines a valid flow-matching objective, the alignment loss alone is not a reliable criterion for path learning. We identify path overfitting, a failure mode in which the alignment loss decreases while sample quality worsens. We find that this failure is associated with low-entropy bottlenecks in the induced probability path, where the learned path routes samples through overly concentrated intermediate marginals. Motivated by this diagnosis, we introduce a stochastic path regularizer that hides part of the source information from the path 
    
[^14]: 面向跨空间时间序列的预测驱动推断

    Prediction-powered inference for time series across space

    [https://arxiv.org/abs/2610.08715](https://arxiv.org/abs/2610.08715)

    本文针对时空数据提出适用于时间依赖场景的预测驱动推断方法，利用短期标注数据与长期无标签协变量，在每个空间位置为未来期望标签值构建有效置信区间，解决了传统PPI独立同分布假设失效的问题。

    

    以下情境在时空数据环境中十分常见：我们在一个相对较短的近期时间段内观测到协变量与标签的配对序列，同时可以获取更长时间段内的无标签协变量，且数据分布在许多空间位置上。例如，作物产量可能仅在最近几年于较大地理区域内被观测到，而天气数据（可用于预测作物产量）则可在更长的时间段内获得。研究目标是在每个空间位置估计未来的期望标签值（如作物产量），并给出该值的有效置信区间。然而，仅凭已观测的较短时间段无法做出可靠估计；用机器学习填补缺失标签则会引入显著偏差。预测驱动推断（PPI）能够纠正这种偏差，但它依赖于独立同分布假设，而该假设在时间序列依赖性下会被打破。此外，异方差性和自相关问题（摘要在此处截断）……

    arXiv:2610.08715v1 Announce Type: cross  Abstract: The following motif is common in spatiotemporal settings: we have a sequence of covariate and label pairs observed for a relatively short, recent time period. We have access to unlabeled covariates over a longer time period. Data is observed over many spatial locations. For instance, crop yield might be observed over a large geographical area for recent years, but weather data (which is informative about crop yield) is available for a much longer period. The goal is to estimate, at each spatial location, the expected label (e.g., crop yield) in the future and provide a valid confidence interval for this value. The observed time period alone is too short for reliable estimates. Imputing missing labels with machine learning can cause substantial bias. Prediction-powered inference (PPI) can correct for this bias, but it relies on an i.i.d. assumption that breaks under our expected temporal dependencies. Heteroskedasticity and autocorrelat
    
[^15]: GeneICL：一个面向Bulk转录组学的表格基础模型

    GeneICL: A Tabular Foundation Model for Bulk Transcriptomics

    [https://arxiv.org/abs/2610.08694](https://arxiv.org/abs/2610.08694)

    GeneICL是一个仅420万参数的表格基础模型，通过基于实测bulk表达谱的半合成预训练先验、参数高效的循环架构，以及基于Cox偏似然残差的无训练生存预测转化方法，证明了转录组感知的预训练比模型规模更关键，能在临床结局预测上超越大型自监督模型。

    

    基因表达在生物医学中被广泛测量，然而由于高维性、强烈的特征相关性以及有限的标注数据，临床结局预测仍然充满挑战。大型自监督转录组基础模型往往难以超越简单的监督基线。表格基础模型通过上下文学习提供了一种替代方案，但它们通常在通用合成数据而非转录组结构上进行预训练。我们提出了这样一个问题：缺失的关键要素是否是转录组感知的预训练，而非模型规模。为此，我们提出了GeneICL，一个拥有420万参数的表格基础模型，它将基于实测bulk表达谱构建的半合成预训练先验与参数高效的循环架构相结合。我们进一步通过使用Cox偏似然残差将生存预测以无需训练的方式转化为回归问题，从而支持右删失生存数据的预测。我们在80个临床（数据集上对GeneICL进行了评估，摘要在此处截断）

    arXiv:2610.08694v1 Announce Type: new  Abstract: Gene expression is widely measured in biomedicine, yet clinical outcome prediction remains challenging due to high dimensionality, strong feature correlations, and limited labeled data. Large self-supervised transcriptomic foundation models often fail to outperform simple supervised baselines. Tabular foundation models offer an alternative through in-context learning, but are typically pretrained on generic synthetic data rather than transcriptomic structure. We ask whether transcriptomics-aware pretraining, rather than scale, is the missing ingredient. Towards this end, we introduce GeneICL, a 4.2M-parameter tabular foundation model combining a semi-synthetic pretraining prior built from measured bulk expression profiles with a parameter-efficient recurrent architecture. We further enable right-censored survival prediction via a training-free reduction to regression using Cox partial-likelihood residuals. We evaluate GeneICL on 80 clini
    
[^16]: 高斯过程因果模型中离散结果的概率反事实推断

    Probabilistic Counterfactual Inference for Discrete Outcomes in Gaussian-Process Causal Models

    [https://arxiv.org/abs/2610.08689](https://arxiv.org/abs/2610.08689)

    该论文提出了一个统一的概率反事实推断框架，通过为二元、名义和有序离散结果分别设计精确的噪声消解机制（均匀阈值、Gumbel-max竞争、潜在高斯切割点模型），将高斯过程结构因果模型的反事实推断从连续变量扩展到异构变量类型，并证明这些机制能再现模型的观测与干预分布。

    

    高斯过程结构因果模型（GP-SCM）中的反事实推断此前主要针对连续内生变量发展，这限制了其在包含“连续父节点—离散子节点”结构的因果图中的适用性。我们提出了一个统一的概率框架，通过将高斯过程预测器与显式的外生噪声机制配对，实现对异构变量类型的反事实推断。对于离散结果，我们推导了精确的条件噪声消解（noise-abduction）程序：对二元变量采用均匀阈值，对名义类别采用Gumbel-max竞争，对有序变量采用潜在高斯切割点模型。在每种情形下，我们在考虑GP潜在函数后验不确定性的同时，通过干预传播被消解的噪声，并证明所构造的机制能够再现拟合模型的观测分布与干预分布。在具有已知真实反事实结果的合成SCM上（原文摘要在此处截断）……

    arXiv:2610.08689v1 Announce Type: new  Abstract: Counterfactual inference in Gaussian-process structural causal models (GP-SCMs) has been developed primarily for continuous endogenous variables, limiting applicability to causal graphs that contain discrete child nodes with continuous parents. We introduce a unified probabilistic framework for counterfactual inference with heterogeneous variable types by pairing GP predictors with explicit exogenous noise mechanisms. For discrete outcomes, we derive exact conditional noise-abduction procedures using a uniform threshold for binary variables, a Gumbel-max race for nominal categories, and a latent Gaussian cut-point model for ordinal ones. In each case, we propagate abducted noise through interventions while accounting for posterior uncertainty in the GP latent functions, and prove that the resulting mechanisms reproduce the fitted model's observational and interventional distributions. On synthetic SCMs with known ground-truth counterfact
    
[^17]: 小型语言模型在抽象推理任务上的系统性研究

    A Systematic Study of Small Language Models on Abstract Reasoning Tasks

    [https://arxiv.org/abs/2610.08680](https://arxiv.org/abs/2610.08680)

    本文系统研究了小型语言模型在ARC-TGI抽象推理基准上的表现，发现尽管模型能取得较高的分布内准确率，但其技能获取对优化过程敏感、在各任务族间分布不均，且分布外性能急剧下降，表明模型可能只是拟合了特定分布的规律而未真正学到可迁移的规则。

    

    抽象推理基准上的终点准确率并不能揭示语言模型是真正获得了可迁移的规则，还是仅仅拟合了特定分布的规律性。我们在小型语言模型上，基于ARC-TGI基准来研究这一区别，该基准将抽象网格变换组织为可控的任务族，并支持重采样、空间平移以及跨基准迁移。在超过1,000次运行中，我们在监督微调设置下对仅解码器、编码器-解码器以及混合专家等模型家族进行了系统剖析。我们考察了技能获取的效率与稳定性、超出训练分布时的鲁棒性、模型家族与任务形式之间的交互作用，以及伴随行为差异出现的逐层注意力特征。研究发现，尽管可以获得可观的分布内准确率，但技能获取对优化过程较为敏感，且在各任务族之间的分布并不均衡。模型性能在分布外急剧下降……

    arXiv:2610.08680v1 Announce Type: cross  Abstract: Endpoint accuracy on abstract-reasoning benchmarks does not reveal whether a language model has acquired a transferable rule or fit distribution-specific regularities. We study this distinction in small language models on the ARC-TGI benchmark, which organizes abstract grid transformations into controllable task families and supports resampling, spatial shifts, and cross-benchmark transfer. Across more than 1,000 runs, we profile decoder-only, encoder--decoder, and mixture-of-experts model families under supervised fine-tuning. We examine the efficiency and stability of skill acquisition, robustness beyond the training distribution, interactions with model family and task formulation, and layer-wise attention signatures that accompany behavioral differences. Substantial in-distribution accuracy is attainable, but acquisition is sensitive to optimization and unevenly distributed across task families. Performance deteriorates sharply out
    
[^18]: 大语言模型的安全投机解码

    Secure Speculative Decoding for Large Language Models

    [https://arxiv.org/abs/2610.08678](https://arxiv.org/abs/2610.08678)

    本文首次系统研究了投机解码的安全影响，揭示了一种“安全-效用不对称”现象：推理效率的提升以不成比例的高安全代价为代价，越狱和提示注入攻击的成功率上升速度远快于效用的下降。

    

    投机解码（Speculative Decoding）通过首先使用一个较小的模型（称为“草稿模型”）生成候选词元，然后由大语言模型（即“目标模型”）对这些候选词元进行接受或拒绝的验证，从而加速大语言模型（LLM）的推理。先前的研究主要聚焦于投机解码的效率与效用之间的权衡，例如有损投机解码，而对其安全影响的研究在很大程度上仍处于空白状态。在这项工作中，我们通过首次对投机解码的安全影响进行系统性研究来填补这一空白。通过大规模的测量研究，我们揭示了一种显著的安全-效用不对称现象：在广泛的有损投机解码方法中，推理效率的提升以不成比例的高昂安全代价为代价，越狱攻击和提示注入攻击的成功率上升速度远快于效用的下降……

    arXiv:2610.08678v1 Announce Type: cross  Abstract: Speculative decoding accelerates inference for a large language model (LLM), referred to as the \emph{target model}, by first using a smaller model, referred to as the \emph{draft model}, to generate candidate tokens and then verifying them with the target model for acceptance or rejection. Prior studies primarily focused on the efficiency-utility trade-off of speculative decoding, e.g., lossy speculative decoding, leaving its security implications largely unexplored.   In this work, we bridge this gap by providing the \emph{first} systematic study of the security implications of speculative decoding. Through a large-scale measurement study, we reveal a pronounced security-utility asymmetry: across a wide range of lossy speculative decoding methods, improvements in inference efficiency come at a disproportionately high cost to security, with attack success rates for jailbreak and prompt injection attacks increasing much faster than uti
    
[^19]: 基于联合效应建模的方差最优离线策略评估

    Variance-Optimal Off-Policy Evaluation with Conjunct Effect Modeling

    [https://arxiv.org/abs/2610.08677](https://arxiv.org/abs/2610.08677)

    本文提出VOCEM估计器，通过以闭式形式求解最优插值系数，在OffCEM和DR之间进行方差最优插值，在保持无偏性的同时确保方差不大于任一端点估计器。

    

    arXiv:2610.08677v1 公告类型：新论文 摘要：当动作级别的重要性加权引入过大方差时，上下文赌博机策略的离线策略评估（OPE）变得具有挑战性。双重稳健（DR）估计在共同支撑假设下保持无偏，但仍保留这些高方差的动作级别权重。先前的估计器——基于联合效应模型的离线策略评估——用更稳定的簇级别权重替代它们，但代价是依赖于奖励模型的局部正确性。在本文中，我们证明在DR和OffCEM所需的假设条件下，存在一族在OffCEM和DR之间进行插值的无偏估计器。基于这一结果，我们提出了方差最优CEM（VOCEM）估计器，它通过选择插值系数来最小化方差。我们以闭式形式推导出总体最优系数，并证明所得估计器的方差不大于任一端点估计器或DR。实验在（摘要在此处截断）……

    arXiv:2610.08677v1 Announce Type: new  Abstract: Off-policy evaluation (OPE) for contextual bandit policies becomes challenging when action-level importance weighting incurs excessive variance. Doubly robust (DR) estimation remains unbiased under common support but retains these high-variance action-level weights. A prior estimator, Off-policy evaluation with Conjunct Effect Model (OffCEM), replaces them with more stable cluster-level weights, at the cost of relying on local correctness of the reward model. In this paper, we show that, under the assumptions required by DR and OffCEM, there exists an unbiased family of estimators that interpolates between OffCEM and DR. Building on this result, we propose the Variance Optimal-CEM (VOCEM) estimator, which selects the interpolation coefficient to minimize variance. We derive the population-optimal coefficient in closed form and show that the resulting estimator has variance no larger than either endpoint, OffCEM or DR. Experiments in cont
    
[^20]: 压力下的原则坚守：后训练决定大语言模型是否会践行自己的道德判断

    Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment

    [https://arxiv.org/abs/2610.08670](https://arxiv.org/abs/2610.08670)

    该研究构建了涵盖五种压力类型的248个预注册场景面板，通过让模型同时以第一人称行动和第三人称评判的方式对照其自身道德判断，发现大语言模型在约五分之一的压力场景下会采取自己判定为错误的行为，且这种“言行不一”差距的大小取决于后训练配方。

    

    语言模型越来越多地充当智能体。一个明知某行为错误却仍然去做的智能体，与一个不知道更好选择的智能体，是两种不同的失败模式，而对模型陈述价值观的评估无法发现前者。我们构建了一个预注册的、涵盖五种压力类型的248个场景面板。每个场景以两种方式向同一模型提出：一次作为智能体选择要采取的行动，一次以第三人称询问哪个选项是正确的，从而以模型自身的判断作为参照。每个场景都有一个移除压力因素的孪生版本，并且每个模型都有一个正向对照——即其运营者下令执行违规行为的场景——以便区分“差距缺失”与“测量工具失灵”。在OLMo-3-7B-Instruct上，该模型在大约五分之一的压力场景中采取了它自己判定为错误的行为，且这一比例高于移除压力后的相同场景。在四个指令模型中，这一差距的大小取决于后训练配方。

    arXiv:2610.08670v1 Announce Type: cross  Abstract: Language models increasingly act as agents. An agent that says an action is wrong and then takes it anyway is a different failure from one that does not know better, and evaluations of stated values cannot see it. We build a pre-registered panel of 248 scenarios across five kinds of pressure. Each scenario is posed twice to the same model, once as the agent choosing what to do and once in the third person asking which option is right, so the model's own judgment is the reference. Every scenario has a twin with the pressure removed, and every model gets a positive control in which its operator orders the violating action, so that a missing gap can be told apart from a blind instrument. On OLMo-3-7B-Instruct, the model takes the action it judged wrong on about one in five pressuring scenarios, more often than on the same scenarios with the pressure removed. Across four instruct models the gap depends on the post-training recipe: OLMo-3 a
    
[^21]: MemFLoRA：面向边缘端CNN适配的内存底线LoRA

    MemFLoRA: Memory-Floor LoRA for CNN Adaptation at the Edge

    [https://arxiv.org/abs/2610.08669](https://arxiv.org/abs/2610.08669)

    本文提出MemFLoRA，一种以内存优先为设计原则的低秩CNN适配器，通过激活内存底线准则（可训练的反向计算不依赖全宽度层输入）来解决边缘端CNN适配中激活内存而非可训练参数数量才是限制性资源的问题。

    

    当模型在部署后遇到用户、传感器或环境特定的偏移时，设备端学习是必不可少的。尽管参数高效微调（PEFT）方法，特别是低秩适应（LoRA）变体，能够实现边缘端的高效适配，但对于卷积神经网络（CNN）的适配而言，限制性资源往往不是可训练参数的数量，而是必须保留到反向传播之前的激活状态。本文提出了内存底线LoRA（MemFLoRA），这是一种围绕内存优先设计原则构建的低秩CNN适配器，而非对面向Transformer的LoRA的直接套用。我们不仅仅是减少可训练权重，而是定义了一个激活内存底线准则：可训练的反向计算不得依赖于全宽度的层输入。由此产生的适配器冻结下投影部分，训练一个尺度匹配的上投影部分，并结合评估模式的骨干网络归一化……

    arXiv:2610.08669v1 Announce Type: cross  Abstract: On-device learning is necessary when the model encounters user-,sensor-, or environment-specific shifts after deployment. Although parameter-efficient fine-tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA) variants, enable efficient adaptation at the edge, the limiting resource for Convolutional Neural Network (CNN) adaptation is often not the number of trainable parameters but the activation state that must be retained until the backward pass. This paper introduces Memory-Floor LoRA (MemFLoRA), a low-rank CNN adapter built around a memory-first design principle rather than a direct application of transformer-oriented LoRA. Instead of merely reducing trainable weights, we define an activation-memory-floor criterion: trainable backward computations must not depend on full-width layer inputs. The resulting adapter freezes the down-projection, trains a scale-matched up-projection, and combines eval-mode backbone normalization
    
[^22]: 利用序列蒙特卡洛引导扩散模型生成稀有事件

    Steering Diffusion Models to Rare Events with Sequential Monte Carlo

    [https://arxiv.org/abs/2610.08652](https://arxiv.org/abs/2610.08652)

    本文提出DireSMC，一种序列蒙特卡洛方法，通过引导加权样本群体趋向扩散模型中的稀有事件，不仅能生成稀有事件样本，还能给出其概率的校准估计，并可轻松扩展到各类用户自定义稀有事件。

    

    扩散模型正日益被用作天气预报、分子动力学和材料设计等领域中昂贵模拟器的替代品。在这些模型中，计算事件 $E$ 的概率 $p_0[E]$ 十分困难，尤其当所关注的事件是稀有事件时。使用蒙特卡洛方法进行稳定估计在计算上将变得不可行，因为需要随稀有程度增加而不断增长的样本量（$\propto 1/p_0[E]$）来补偿。在本文中，我们提出了稀有事件的扩散重要性采样方法（Diffusion Importance Sampling of Rare Events，简称 DireSMC），这是一种序列蒙特卡洛方案，通过引导一组加权样本趋向稀有事件，不仅能获得样本，还能得到其概率的校准估计。我们通过对事件集合进行解析松弛来构建引导机制，使该方法能够轻松扩展到广泛的用户自定义稀有事件。我们在一个具有解析解的玩具问题以及一个基于分数的模型上验证了我们的方法。

    arXiv:2610.08652v1 Announce Type: cross  Abstract: Diffusion models are increasingly used as surrogates for expensive simulators in weather prediction, molecular dynamics, and materials design. In these models, computing the probability $p_0[E]$ of an event $E$ is difficult, especially when the event of interest is rare. A stable estimate using Monte Carlo becomes computationally intractable, requiring a growing sample size $\propto\!1/p_0[E]$ to compensate for an increasing rarity. In this paper, we present Diffusion Importance Sampling of Rare Events or DireSMC, a sequential Monte Carlo scheme that guides a population of weighted samples towards the rare event, giving access not only to samples but also to a calibrated estimate of its probability. We set up our guidance using an analytical relaxation of the event set, allowing the method to easily extend to a wide range of user-defined rare events. We validate our method on a toy problem with analytical solutions and on a score-based
    
[^23]: SquidAgent：明智并行，高效协调

    SquidAgent: Parallelize Wisely, Coordinate Efficiently

    [https://arxiv.org/abs/2610.08647](https://arxiv.org/abs/2610.08647)

    该论文提出SquidAgent，揭示了并行多智能体系统中“重新探索成本”与“对齐成本”这两项隐性开销，并据此推导出原则性决策准则：仅当并行化的关键路径成本加上这两项开销低于串行成本时，才应对该层进行并行化。

    

    基于大语言模型（LLM）的智能体能够解决复杂的多步骤任务，但顺序执行会带来显著的延迟。原则上，将工作并行分配给多个智能体应当产生接近线性的加速。然而，现有的并行多智能体系统往往比单智能体基线运行得更慢。我们将这一差距归因于并行执行会产生、而串行智能体可以避免的两项隐性成本。其一是重新探索成本：并行工作节点在重构编排器（orchestrator）已掌握的上下文（例如先前的决策）上所花费的冗余工作，而这些上下文在串行执行中本可被隐式继承。其二是对齐成本：为调和独立生成的输出之间的不一致性所需的额外开销。由此，我们推导出一个有原则的决策准则：只有当某一层的关键路径成本加上重新探索与对齐的开销低于相应的串行成本时，该层才应当被并行化。

    arXiv:2610.08647v1 Announce Type: new  Abstract: LLM-based agents solve complex multi-step tasks, but sequential execution incurs substantial latency. In principle, parallelizing work across multiple agents should yield near-linear speedups. Yet existing parallel multi-agent systems often run slower than a single-agent baseline. We attribute this gap to two hidden costs that parallel execution incurs but a serial agent avoids. First, there is a re-exploration cost: redundant effort spent by parallel workers reconstructing context that the orchestrator already possesses, such as prior decisions, that would otherwise be inherited implicitly in a serial execution. Second, there is an alignment cost: the overhead required to reconcile inconsistencies across independently generated outputs. We thus derive a principled decision criterion: a layer should be parallelized only when its critical-path cost, plus re-exploration and alignment overheads, is lower than the corresponding serial cost. 
    
[^24]: 扩散模型中的特征信息动力学

    Feature Information Dynamics in Diffusion

    [https://arxiv.org/abs/2610.08626](https://arxiv.org/abs/2610.08626)

    提出了基于 I-MMSE 恒等式的信息论框架“特征信息动力学”，通过比较无条件与特征条件去噪损失之差来估计特征信息密度，从而精确定位各特征在扩散生成过程中出现的时间，并定量证实了谱自回归现象。

    

    扩散模型通过一系列连续的去噪问题来生成数据，并被广泛观察到先呈现粗略结构、后生成精细细节。然而，这一直觉大多停留在经验性和定性层面。我们提出了特征信息动力学，这是一个用于定位特征在扩散过程中何时被生成的信息论框架。利用 I-MMSE 恒等式，我们将特征互信息的变化率与最优无条件去噪损失和特征条件去噪损失之间的差距联系起来，从而得到特征信息密度的实用估计器。我们进一步开发了一种链式分解方法，可在特征层次结构中分离共享信息与增量信息。我们首先利用该框架定量证实了像素扩散中的谱自回归现象，随后将分析扩展到频率维度之外：在“类别 → 掩码 → Canny”条件链下，各特征的信息密度在像素空间上存在差异。

    arXiv:2610.08626v1 Announce Type: cross  Abstract: Diffusion models generate data through a continuum of denoising problems, and are widely observed to reveal coarse structure before fine detail. Yet, this intuition is mostly empirical and qualitative. We introduce feature information dynamics, an information-theoretic framework for localizing when a feature is generated during diffusion. Using the I-MMSE identity, we connect the rate of feature mutual information change to a gap between optimal unconditional and feature-conditional denoising losses, yielding practical estimators for feature information density. We further develop a chained decomposition that separates shared from incremental information in a feature hierarchy. We use this framework first to quantitatively confirm spectral autoregression in pixel diffusion, and then to extend the analysis beyond frequency: under a class $\to$ mask $\to$ Canny conditioning chain, the per-feature information densities differ across pixel
    
[^25]: 面向平衡Adam的早期记忆选择方法

    Early Memory Selection for Balanced Adam

    [https://arxiv.org/abs/2610.08624](https://arxiv.org/abs/2610.08624)

    该论文提出一种通过短暂试点训练自动选择Adam共享记忆参数β的方法，利用三次记忆规则平衡采样波动与梯度平均延迟，在十一个视觉和语言任务上将平均相对验证差距降低40.7%以上。

    

    我们提出了一种通过短暂的试点训练来选择Adam中共享记忆参数 $\beta_1=\beta_2=\beta$ 的方法。所选的 $\beta$ 在随后的完整训练过程中保持固定。通过对Adam归一化方向的局部建模，我们平衡了采样波动性与平均历史梯度所引入的延迟。这种平衡导出了一个三次记忆规则，其两个系数通过在少数试点检查点处的梯度探测来估计。该估计器联合使用分子和分母，从而保留了两者的协方差。在200步更新的试点训练中，于四个检查点各使用十六个探测梯度，在与随机种子匹配的回溯评估中，该方法在十一个视觉和语言工作负载上相比共享 $\beta=0.95$ 的网格代表值，将平均相对验证差距降低了40.7%，最差四分之一平均差距降低了44.3%。其平均差距还比从全部十一个工作负载中选出的最佳常数 $\beta$ 低32.3%。

    arXiv:2610.08624v1 Announce Type: cross  Abstract: We propose a method for choosing the shared memory parameter $\beta_1=\beta_2=\beta$ in Adam from a short pilot training. The selected $\beta$ remains fixed during the subsequent full training. A local model of Adam's normalized direction balances sampling variability against the delay introduced by averaging past gradients. This balance gives a cubic memory rule, whose two coefficients are estimated from gradient probes at a few pilot checkpoints. The estimator uses the numerator and denominator jointly, preserving their covariance. With a 200-update pilot and sixteen probe gradients at each of four checkpoints, a seed-matched retrospective evaluation on eleven vision and language workloads reduces mean relative validation gap by 40.7% and worst-quarter mean gap by 44.3% against the grid representative of shared $\beta=0.95$. The mean gap is also 32.3% lower than that of the best constant $\beta$ chosen across all eleven workloads.
    
[^26]: 基于游戏画面深度学习的电子游戏多标签感知性缺陷检测

    Multi-Label Perceptual Bug Detection in Video Games using Deep Learning on Gameplay Footage

    [https://arxiv.org/abs/2610.08593](https://arxiv.org/abs/2610.08593)

    本文提出ResNet-BiLSTM深度学习模型，通过时序依赖建模实现对游戏视频画面中多种感知性缺陷的多标签检测，在基准数据集上达到85.78%的F1分数，可显著减少游戏人工测试的资源消耗。

    

    传统的电子游戏自动化缺陷检测方法（如人工测试）虽然有助于提升质量保证水平，但成本高昂且耗时。目前能够在同一视频帧中检测多种感知性缺陷的工具十分稀缺，这给现实场景中的自动化缺陷检测工具带来了检测挑战。我们提出了一种用于多标签感知性缺陷检测的深度学习模型，并将其与Inflated 3D ConvNet和3D ResNet等视频分类模型进行了对比。我们提出的ResNet-BiLSTM模型在基准数据集上取得了85.78%的F1分数。结果表明，时序依赖建模对于实现准确的基于视频的缺陷检测是有益的。我们相信这项针对游戏视频的多标签感知性缺陷检测工作将有助于节省电子游戏人工测试工作所耗费的资源。此外，我们还引入了一个新的数据……

    arXiv:2610.08593v1 Announce Type: new  Abstract: Traditional approaches for automated bug detection in video games, such as manual testing, can be beneficial for the improvement of quality assurance, but they can be expensive and time-consuming. The scarce number of tools available to detect multiple perceptual bugs in the same video frame introduces detection challenges for automated bug detection tools in real-world scenarios. We propose a deep learning model for multi-label perceptual bug detection and compare it against video classification models such as Inflated 3D ConvNet and 3D ResNet. Our proposed model, ResNet-BiLSTM, achieved an F1 score of 85.78% on the benchmark dataset. Our results demonstrated that temporal dependency modelling is beneficial for accurate video-based bug detection. We believe this work with multi-label perceptual bug detection on gameplay videos will help save resources spent on manual testing workloads in video games. Furthermore, we introduce a new data
    
[^27]: CNet：一个具有Wirtinger自动微分与FFT-阿达马卷积的复值深度学习框架

    CNet: A Complex-Valued Deep Learning Framework with Wirtinger Autodifferentiation and FFT--Hadamard Convolution

    [https://arxiv.org/abs/2610.08592](https://arxiv.org/abs/2610.08592)

    CNet 是一个基于 Wirtinger 自动微分的 C++/CUDA 复值深度学习框架，它将 FFT-阿达马卷积恒等式转化为可学习的复值卷积网络，并采用玻恩规则进行物理原生式分类。

    

    CNet 是一个用于构建和训练深度复值神经网络（CVNN）的 C++/CUDA 框架，更广泛地说，它利用 Wirtinger（CR 演算）导数通过梯度下降来优化复值函数。该框架采取物理原生的立场：网络是作用于振幅向量的复数运算——通常是幺正运算（如 DFT）——的级联，而分类则采用玻恩规则测量 $p_k = |z_k|^2 / \|z\|^2$，而非对实数 logits 进行 softmax。每个层都提供 CPU 参考实现和经过有限差分校验的 CUDA 内核，并且计算图会在整个批次上克隆以供 GPU 执行。在基础层之上，我们添加了信号处理原语，将恒等式 conv(x,k) = IFFT(FFT(x) · FFT(k)) 转化为可学习的复值卷积网络，同时提供了真 Adam 优化器和低内存推理模式。我们报告了三项研究。第一项是一个完全复值的、FNet 风格的因果……（摘要在此处截断）

    arXiv:2610.08592v1 Announce Type: new  Abstract: CNet is a C++/CUDA framework for building and training deep complex-valued neural networks (CVNNs) and, more generally, for optimizing complex-valued functions by gradient descent with Wirtinger (CR-calculus) derivatives. It takes a physics-native stance: a network is a cascade of complex -- and often unitary (the DFT) -- operations acting on an amplitude vector, and classification is a Born-rule measurement $p_k = |z_k|^2 / \|z\|^2$ rather than a softmax over real logits. Every layer ships a CPU reference and a CUDA kernel checked against finite differences, and the computation graph is cloned across the batch for GPU execution. On top of the base layers we add signal-processing primitives that turn the identity conv(x,k) = IFFT(FFT(x) . FFT(k)) into a learnable complex convolutional network, together with a true-Adam optimizer and a reduced-memory inference mode.   We report three studies. First, a fully complex-valued, FNet-style caus
    
[^28]: 随机特征高斯过程注意力：具有校准不确定性的线性时间概率注意力

    Random Feature Gaussian Process Attention: Linear-Time Probabilistic Attention with Calibrated Uncertainty

    [https://arxiv.org/abs/2610.08578](https://arxiv.org/abs/2610.08578)

    本文提出即插即用的随机傅里叶特征高斯过程注意力模块（RFF-GPA），通过随机傅里叶特征对平稳核进行低秩近似，将注意力表示为高斯过程后验，实现了线性时间复杂度且具有校准不确定性的概率注意力计算。

    

    Transformer 提供了最先进的建模框架，但其较差的校准能力限制了其在安全关键应用中的可靠性。一个有前景的方向是将注意力机制解释为高斯过程（GP）后验，这能够实现有原则的不确定性校准，但由于需要对核矩阵求逆，其复杂度随序列长度呈三次方增长；尽管解耦的 GP 变体将成本降低到了二次方，但在实践中计算量仍然过于高昂。在本文中，我们提出了即插即用的随机傅里叶特征高斯过程注意力模块，该模块将注意力表示为一个具有平稳核的高斯过程，并使用随机傅里叶特征对该核进行近似。这种低秩近似使得后验均值和方差的近似计算达到线性时间复杂度，与先前的工作相比具有更强的可扩展性。在多个真实世界数据集上的实证结果表明，我们的注意力模块……（原文摘要在此处截断）

    arXiv:2610.08578v1 Announce Type: new  Abstract: Transformers provide a state-of-the-art modeling framework, yet poor calibration limits their reliability in safety-critical applications. A promising direction addresses this issue by interpreting attention as a Gaussian process (GP) posterior, which enables principled uncertainty calibration but incurs cubic complexity in sequence length due to the inversion of the kernel; although decoupled GP variants reduced the cost to quadratic, the computation remains prohibitive in practice. In this paper, we propose the plug-and-play random Fourier feature Gaussian process attention (RFF-GPA) module, which represents the attention as a GP with a stationary kernel approximated by random Fourier features. This low-rank approximation results in linear-time complexity for approximating the posterior mean and variance, making it far more scalable compared to previous work. Empirical results on multiple real-world datasets show that our attention mod
    
[^29]: 学习方式如何跨越记忆-泛化谱系支配遗忘

    How Learning Governs Unlearning across the Memorization-Generalization Spectrum

    [https://arxiv.org/abs/2610.08577](https://arxiv.org/abs/2610.08577)

    模型的学习方式决定了其后续遗忘的表现：偏泛化型模型在遗忘时比偏记忆型模型遭受更大的保留集性能损害，且这一趋势在记忆-泛化谱系上几乎单调成立。

    

    虽然机器遗忘旨在消除模型通过学习获得的不想要的能力，但很少有研究考察模型的学习方式如何塑造其后续的遗忘过程。在本文中，我们从记忆和泛化的角度研究这一联系，这是模型在训练期间采用的两种最具代表性却又相互竞争的策略。我们首先利用模加法中的grokking（顿悟）现象对偏记忆型模型和偏泛化型模型进行分类，并比较它们对遗忘操作的响应，结果表明偏泛化型模型遭受更大的保留损害，即在保留集上的性能下降更为严重。此外，我们通过引入分桶模加法进行了更细粒度的分析，在该设置下，两种策略各自的贡献可以在记忆-泛化谱系上被显式控制。在这一设置中，我们再次证实相同的趋势持续存在且几乎呈单调关系。我们进一步证明……（原文摘要在此处截断）

    arXiv:2610.08577v1 Announce Type: cross  Abstract: While unlearning seeks to negate undesired capabilities acquired through learning, little research has examined how the way models learn shapes their subsequent unlearning. In this paper, we investigate this connection from the perspectives of memorization and generalization, the two most representative yet competing strategies that models employ during training. We first classify memorization- and generalization-heavy models using grokking in modular addition and compare their responses to unlearning, showing that the latter suffer greater retain damage, i.e., a larger performance drop on the retain set. Furthermore, we conduct a finer-grained analysis by introducing bucketed modular addition, in which the respective contributions of the two strategies can be explicitly controlled across the memorization-generalization spectrum. In this setup, we reaffirm that the same trend persists and is nearly monotonic. We further demonstrate tha
    
[^30]: FedDermaSeg：面向皮肤病学图像分割的联邦学习

    FedDermaSeg: Federated Learning for Dermatological Image Segmentation

    [https://arxiv.org/abs/2610.08574](https://arxiv.org/abs/2610.08574)

    该论文提出FedDermaSeg，探索利用联邦学习在无需集中收集数据的情况下实现隐私保护的皮肤病灶分割，以解决传统集中式深度学习训练带来的隐私风险和高计算资源需求问题。

    

    皮肤癌是一个重要的全球健康问题，早期发现和精确的病灶勾画对于有效的诊断与治疗规划至关重要。自动化的皮肤病灶分析可以辅助皮肤科医生，其中病灶分割是计算机辅助诊断系统中的基础步骤。传统的基于深度学习的分割模型通常依赖于集中式训练，即图像及其对应的分割掩码被收集到中央服务器上。这种数据聚合方式在医疗应用中引发了隐私方面的担忧，并且需要大量的集中式计算资源。为了解决这些局限性，我们研究了联邦学习在保护隐私的皮肤病灶分割中的可行性。我们使用ISIC 2018皮肤病灶分割挑战赛数据集的训练集和验证集来模拟分布式学习环境，并开发联邦分割模型。

    arXiv:2610.08574v1 Announce Type: cross  Abstract: Skin cancer is a major global health concern, and early detection and accurate lesion delineation are important for effective diagnosis and treatment planning. Automated skin lesion analysis can assist dermatologists, with lesion segmentation serving as a fundamental step in computer-aided diagnostic systems. Conventional deep learning-based segmentation models typically rely on centralized training, where images and their corresponding segmentation masks are collected on a central server. Such data aggregation raises privacy concerns in medical applications and requires substantial centralized computational resources. To address these limitations, we investigate the feasibility of federated learning for privacy-preserving skin lesion segmentation. The training and validation sets of the ISIC 2018 Skin Lesion Segmentation Challenge dataset are used to simulate a distributed learning environment and develop a federated segmentation mode
    
[^31]: RAG-PIBench：一个面向可信RAG系统中提示注入检测的防泄漏基准

    RAG-PIBench: A Leakage-Aware Benchmark for Prompt-Injection Detection in Trustworthy RAG Systems

    [https://arxiv.org/abs/2610.08571](https://arxiv.org/abs/2610.08571)

    该论文提出了RAG-PIBench——一个面向RAG系统提示注入检测的防泄漏基准，包含4,876个上下文示例，通过严格评估协议发现DistilBERT取得最佳检测性能（F1=0.896），同时证明TF-IDF等稀疏基线方法仍具竞争力。

    

    检索增强生成（RAG）系统容易受到嵌入在检索内容中的提示注入攻击。我们提出了RAG-PIBench，一个用于RAG式提示注入检测的基准，包含4,876个上下文示例，分布在固定的训练集、验证集和受保护的测试集划分中。通过采用防泄漏的构建流程和严格的评估协议，我们比较了基于关键词、语义参考、TF-IDF以及基于Transformer的检测器。DistilBERT在受保护测试集上取得了最佳性能（F1 = 0.896，PR-AUC = 0.968），而TF-IDF SVM和逻辑回归仍保持竞争力。我们的结果展示了防泄漏基准设计和强稀疏基线对于RAG系统中可靠提示注入检测的价值。

    arXiv:2610.08571v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) systems are vulnerable to prompt-injection attacks embedded in retrieved content. We introduce RAG-PIBench, a benchmark for RAG-style prompt-injection detection containing 4,876 contextual examples across frozen train, validation, and protected-test splits. Using a leakage-aware construction pipeline and strict evaluation protocol, we compare keyword-based, semantic-reference, TF-IDF, and transformer-based detectors. DistilBERT achieves the best protected-test performance (F1 = 0.896, PR-AUC = 0.968), while TF-IDF SVM and logistic regression remain competitive. Our results demonstrate the value of leakage-aware benchmark design and strong sparse baselines for reliable prompt-injection detection in RAG systems.
    
[^32]: 少即是多：基于YOLO26的皮肤镜预处理在皮肤病变联合分类与分割中的泄漏控制研究

    Less Is More: A Leakage-Controlled Study of Dermoscopic Preprocessing for Joint Skin Lesion Classification and Segmentation with YOLO26

    [https://arxiv.org/abs/2610.08570](https://arxiv.org/abs/2610.08570)

    在严格按病变划分、防止数据泄漏的评估下，研究发现复杂的手工皮肤镜预处理相比最小预处理加在线增强，对固定YOLO26n-seg模型的联合病变分类与分割性能提升有限，体现“少即是多”。

    

    手工预处理被广泛应用于自动化皮肤镜分析中，用于抑制成像伪影并增强病变的可见性。然而，其对现代实时模型的实际贡献仍不清楚，尤其是当评估协议未能充分控制同一病变的多张图像之间的相关性时。本研究针对皮肤镜预处理与数据增强提出了一种泄漏控制、病变不重叠的评估方法，用于联合多类病变分类与实例分割任务，采用固定的纳米级YOLO26分割模型（YOLO26n-seg）。从HAM10000数据集（10,015张图像）出发，经质量控制后得到来自7,468个独立病变的10,013对有效的图像-掩膜对，并按病变身份划分为互斥的集合。在架构、分辨率、训练预算和评估协议保持固定的情况下，我们将最小处理的图像加在线数据增强，与离线类别平衡、DullRazor-C…（摘要截断）

    arXiv:2610.08570v1 Announce Type: new  Abstract: Handcrafted preprocessing is widely employed in automated dermoscopic analysis to suppress imaging artifacts and enhance lesion visibility. Nevertheless, its actual contribution to modern real-time models remains unclear, particularly when evaluation protocols do not adequately control correlations among images of the same lesion. This study presents a leakage-controlled, lesion-disjoint evaluation of dermoscopic preprocessing and augmentation for joint multi-class lesion classification and instance segmentation using a fixed nano-scale YOLO26 segmentation model (YOLO26n-seg). From HAM10000 (10,015 images), quality control yields 10,013 valid image-mask pairs from 7,468 unique lesions, partitioned into mutually exclusive sets by lesion identity. With the architecture, resolution, training budget, and evaluation protocol held fixed, we compare minimally processed images plus online augmentation against offline class balancing, DullRazor-C
    
[^33]: 奇异值分解：一次几何的重新发现，证明即算法

    Singular Value Decomposition: A Geometric Rediscovery, Where Proofs Become Algorithms

    [https://arxiv.org/abs/2610.08565](https://arxiv.org/abs/2610.08565)

    本文以“单位圆映射为椭圆”的几何直觉重新发现奇异值分解，并展示其证明过程本身可作为算法运行，构成主成分分析、核方法与 PageRank 等机器学习技术背后的核心机制。

    

    本文是对奇异值分解的一次几何学上的重新探索，并进一步主张：文章所构建的内容正是机器学习诸多领域背后的核心机制。回答一个关于椭圆问题的同一论证，正是主成分分析、核方法和 PageRank 背后的算法；而且不仅是结论可以迁移，证明本身也可以作为过程来运行。通常的引入方式是直接陈述 $A = U\Sigma V^T$，并通过将谱定理应用于 $A^T A$ 来加以证明。这是正确的，但缺乏启发性，因为它借助一个强大的定理去得到一个本质上关于椭圆的结果。本文第一部分颠倒了这一顺序：线性映射将单位圆变为椭圆；人们会问哪些输入方向映射到椭圆的轴上，并在一个又一个例子中发现它们是相互垂直的。在平面上，这一过程可以被直接观察：旋转一组标架，追踪其像偏离垂直的程度，而一个符号……（摘要在此处截断）

    arXiv:2610.08565v1 Announce Type: new  Abstract: This article is a geometric rediscovery of the singular value decomposition, with a further claim: the construction it builds is the machinery behind much of machine learning. The same argument that answers an idle question about ellipses is the algorithm behind principal component analysis, kernel methods, and PageRank, and it is not only the results that transfer but the proofs themselves, run as procedures.   The usual introduction states $A = U\Sigma V^T$ and justifies it via the spectral theorem applied to $A^T A$. This is correct but unilluminating, since it assumes a powerful theorem to reach a result that is, in the end, about ellipses. Part I reverses the order. A linear map sends the unit circle to an ellipse; one asks which input directions map to its axes, and finds, example after example, that they are perpendicular. In the plane this can be watched: rotate a frame, track how far its images are from perpendicular, and a sign
    
[^34]: 免费获得有效性：基于同质性门控保形预测的表格基础模型无训练节点分类

    Valid for Free: Homophily-Gated Conformal Prediction for Training-Free Node Classification with Tabular Foundation Models

    [https://arxiv.org/abs/2610.08564](https://arxiv.org/abs/2610.08564)

    该论文首次对表格基础模型无训练图节点分类设定进行可靠性研究，证明冻结的上下文预测器可使拆分保形预测在有限样本下严格有效且无需任何训练或调参，并提出同质性门控的HG-DAPS扩散评分以改善预测集。

    

    表格基础模型（TFMs）无需在图上进行训练即可对图的节点进行分类，其做法是将节点及其邻域特征作为表格行，与已标注的上下文行放在一起读取。这一方向的已有工作仅报告了预测性能，而未涉及保形覆盖率或预测集大小。据我们所知，我们首次对该设定开展了可靠性研究，其中以TabICL作为表格基础模型，并以每个图的一半节点作为已标注上下文。与任何在校准之前就已固定的预测器一样，冻结的上下文预测器使拆分保形预测在有限样本下严格有效，且无需在目标图上进行任何训练、验证集划分或调参。随后在十个图上的审计表明，无训练的TabICL后验在其中九个图上具有比带温度缩放的GCN（GCN+TS）更低的期望校准误差（ECE），其在十个图上的平均ECE为0.019，比GCN+TS的0.029低约35%。我们还提出了HG-DAPS，这是一种无训练的扩散评分……（摘要在此处截断）

    arXiv:2610.08564v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) can classify the nodes of a graph without training on it, by reading node and neighborhood features as table rows next to labeled context rows. Work in this line reports predictive performance, not conformal coverage or prediction-set size. To our knowledge, we give the first reliability study of the setting, with TabICL as the TFM and half of each graph as labeled context. As for any predictor fixed before calibration, a frozen in-context predictor makes split conformal prediction exactly valid in finite samples, with no training, validation fold, or tuning on the target graph. An audit across ten graphs then shows that the training-free TabICL posterior has lower expected calibration error (ECE) than GCN with temperature scaling (GCN+TS) on nine of them. Its mean ECE over the ten graphs is 0.019, about 35 percent below the 0.029 of GCN+TS. We also introduce HG-DAPS, a training-free diffusion score whose
    
[^35]: 面向分层推理奖励的强化学习：基于Transformer的极小极大最优速率

    Reinforcement Learning for Hierarchical Reasoning Rewards: Minimax-Optimal Rates with Transformers

    [https://arxiv.org/abs/2610.08561](https://arxiv.org/abs/2610.08561)

    本文将推理任务的奖励建模为响应空间上的分层函数，并证明一种基于Transformer的actor-critic强化学习算法在查询预算和正则化强度上达到极小极大最优速率，从理论上解释了在策略探索结合神经奖励模型的RL后训练为何有效。

    

    强化学习（RL）已成为在推理任务上对语言模型进行后训练的标准工具，其中策略在探索响应空间的同时通过奖励反馈进行更新。尽管其在实证上取得了成功，但对RL后训练的理论理解仍然有限，尤其是对于为什么在策略探索结合神经奖励模型能够有效这一问题的理解。在本文中，我们通过将奖励建模为响应空间上的分层函数来回答这一问题：奖励由无穷多个局部组件构成，每个组件只有在前面的组件被解决之后才会变得相关。我们证明了一种自然的基于Transformer的actor-critic算法——该算法在从当前KL正则化策略中采样、用观测到的奖励拟合Transformer评论家网络、以及更新策略这三个步骤之间交替进行——在查询预算和正则化强度方面达到了极小极大最优速率（最多相差对数因子）。

    arXiv:2610.08561v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become a standard tool for post-training language models on reasoning tasks, where the policy is updated by reward feedback while exploring the space of responses. Despite its empirical success, theoretical understanding of RL post-training remains limited, in particular of why on-policy exploration combined with a neural reward model is effective. In this paper, we address this question by modeling the reward as a hierarchical function on the response space: the reward consists of infinitely many local components, each of which becomes relevant only after the preceding ones have been resolved. We show that a natural Transformer-based actor--critic algorithm, which alternates between sampling from the current KL-regularized policy, fitting a Transformer critic to the observed rewards, and updating the policy, achieves the minimax optimal rates in the query budget and in the regularization strength up to lo
    
[^36]: 我看得够了吗？冻结的视频-语言模型中编码了证据就绪度信号

    Have I Seen Enough? Frozen Video-Language Models Encode Evidence Readiness

    [https://arxiv.org/abs/2610.08560](https://arxiv.org/abs/2610.08560)

    该论文发现冻结的视频-语言模型内部已线性编码了一种由问题条件化、可跨基准泛化且与答案对错无关的“证据就绪度”信号，因此无需额外训练触发器即可判断流式视频问答中证据是否已充分到来。

    

    流式视频-语言模型不仅需要决定回答什么，还必须判断当前问题所需的证据是否已经到来。现有系统将该决策作为一个单独的触发器来学习；我们则探究一个未经修改的模型是否已经在计算这一决策。我们证明，冻结的视频大语言模型内部携带一种线性可读的证据就绪信号，该信号的标注来自带时间戳的证据而非模型输出。在一个共享的逐字节相同的评估中，该信号在全部七个模型中均可被解码（在最严格的“未就绪”采样下AUROC为0.733–0.905，而拟合的时钟模型接近随机水平），并且在完全未接触某基准家族视频数据的情况下训练的探针仍能读取该家族的数据。该信号是问题条件化的：在逐字节相同的视频窗口上，仅改变问题就能使66.1%的问题配对上的读出结果发生反转，而所有问题盲的控制组按构造均处于随机水平。模型即使给出错误答案，仍然编码了就绪状态：在错误答案中AUROC仍为0.722。

    arXiv:2610.08560v1 Announce Type: cross  Abstract: Streaming video-language models must decide not only what to answer, but whether the evidence needed for the current question has arrived. Existing systems learn that decision as a separate trigger; we ask whether an unmodified model already computes it. We show that frozen VideoLLMs carry a linearly readable evidence-readiness signal, labelled from timestamped evidence rather than from model output. It decodes in all seven models of a shared byte-identical evaluation (AUROC 0.733-0.905 under the strictest not-ready sampling, where a fitted clock is near chance), and a probe fitted without any of a benchmark family's footage still reads that family. It is question-conditioned: on byte-identical windows, changing only the question reverses the readout on 66.1% of pairs, while every question-blind control is at chance by construction. The model can answer incorrectly and still encode readiness: AUROC remains 0.722 among wrong answers. Re
    
[^37]: 大语言模型潜空间中的偏见方向捕获的是置信度，而非公平性

    Latent space bias directions in LLMs capture confidence, not fairness

    [https://arxiv.org/abs/2610.08559](https://arxiv.org/abs/2610.08559)

    该研究揭示了大语言模型激活引导中的去偏方向实际上编码的是模型置信度而非偏见信息，其去偏效果只是降低模型置信度的副产品，从而解释了激活引导去偏技术泛化能力差的根本原因。

    

    arXiv:2610.08559v1 公告类型： cross 摘要：激活引导（activation steering）作为一种轻量级的大语言模型推理时去偏技术日益流行。然而，先前的研究报告指出，引导向量的泛化能力较差，会对模型性能产生意外影响，并且向新数据集的迁移能力有限。我们的工作分析了用于激活引导的去偏方向实际上编码了什么，以揭示其性能不一致的原因。我们研究了通过对比反偏见提示和偏见提示的激活所获得的线性去偏方向，并将其作为一种引导干预措施，在偏见基准和通用知识基准上进行评估。我们发现，这个方向主要由模型置信度主导，在激活空间中从高概率token区域指向低概率token区域，而不是编码模型偏见的有意义表征。沿着该方向进行引导确实能减少测量到的偏见，但这是降低模型置信度的结果：在问答（QA）基准上，我们……

    arXiv:2610.08559v1 Announce Type: cross  Abstract: Activation steering has gained popularity as a lightweight inference-time debiasing technique for large language models. However, prior work reports that steering vectors generalise poorly, with unintended effects on model performance and limited transfer to new datasets. Our work analyses what the debiasing direction used for activation steering actually encodes, in order to shed light on its inconsistent performance. We study the linear debiasing direction obtained by contrasting the activations of anti-biased and biased prompts, and evaluate it as a steering intervention across bias and general knowledge benchmarks. We find that this direction is dominated by model confidence, pointing from regions of high to low-probability tokens in activation space rather than encoding a meaningful representation of model bias. Steering along it does reduce measured bias, but this is a consequence of reducing model confidence: on QA benchmarks we
    
[^38]: 系统化知识（SoK）：以人为中心的青少年AI安全

    Systemization of Knowledge (SoK): Human-Centered AI Safety for Youth

    [https://arxiv.org/abs/2610.08554](https://arxiv.org/abs/2610.08554)

    本文系统回顾了100项HCI实证研究，构建了青少年AI风险与对策的映射，发现多数风险仅有构想层面的对策，很少被实施和评估，且评估多聚焦技术性能而非实际防伤害效果。

    

    尽管人机交互（HCI）领域日益关注面向青少年的AI安全问题，但相关文献仍缺乏对以下问题的全面认识：已识别出哪些风险、这些风险如何被应对、以及所提出的保护措施在实践中是否有效。我们系统性回顾了100项实证HCI研究，这些研究涉及儿童和青少年在学校、家庭、照护场所以及公共服务中与AI交互或暴露于AI的情形。借助YAIR风险分类法和MIT缓解措施分类法，我们梳理了哪些风险已被识别、每项风险是否有相应的应对措施、以及针对该风险的每项措施是否已被实施乃至评估。风险-对策映射显示：大多数风险仅配有“已提出/构想中”的对策；很少有对策被实际实施，被评估的更少；且现有评估往往衡量技术性能，而非对伤害的实际防护效果。我们进一步识别了覆盖范围的缺口所在……

    arXiv:2610.08554v1 Announce Type: cross  Abstract: While HCI increasingly examines AI-safety for youth, the literature lacks a comprehensive view of what risks have been identified, how they are addressed, and whether proposed protections work in-practice. We systematically reviewed 100 empirical HCI studies involving children and youth interacting with or exposed to AI across schools, homes, care settings, and public services. Using the YAIR taxonomy for risks and the MIT Mitigation Taxonomy for countermeasures, we map which risks have been identified, whether each risk is addressed by countermeasure(s), and whether each countermeasure for that risk is implemented and even evaluated. The risk-countermeasure mapping shows that most risks are matched only with proposed/ideated countermeasures; few countermeasures have been implemented, and fewer still evaluated; and existing evaluations often measure technical performance rather than protection from harm. We identify where coverage is a
    
[^39]: DeltaTTT：非线性循环记忆的逐层优化

    DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory

    [https://arxiv.org/abs/2610.08553](https://arxiv.org/abs/2610.08553)

    针对非线性循环记忆在测试时训练中难以优化、并行基线反而优于串行版本的问题，DeltaTTT提出用逐层学习替代联合内循环优化，为每层分配局部预测目标并通过状态依赖的delta规则更新，从而缓解非线性记忆的优化困难。

    

    序列测试时训练通过一系列连续更新来适配记忆网络，每次更新都基于网络的先前状态计算内循环梯度。直观上，这种状态依赖性应使每次更新能够考虑到记忆已经学到的内容，并更好地融合新信息。然而，我们发现这一预期优势在非线性记忆中并未稳定实现：固定基底的并行TTT基线反而优于其串行对应版本。我们的探索性实验揭示了一个关键的潜在困难：在单次序列遍历过程中，非线性记忆可能比线性记忆更难优化。为缓解这一优化困难，我们提出DeltaTTT，它用逐层学习取代了两层记忆网络的联合内循环优化。每一层被分配一个局部预测目标，并通过状态依赖的delta规则进行更新。该公式化表述保留了非线性...

    arXiv:2610.08553v1 Announce Type: cross  Abstract: Sequential test-time training adapts a memory network through successive updates, each computing an inner-loop gradient based on the network's previous state. Intuitively, this state dependence should allow each update to account for what the memory has already learned and better incorporate new information. However, we find that this expected advantage does not consistently materialize in nonlinear memories: a fixed-base parallel TTT baseline outperforms its serial counterpart. Our exploratory experiments point to a key underlying difficulty: nonlinear memories can be harder to optimize than linear ones within a single pass over the sequence. To alleviate this optimization difficulty, we introduce DeltaTTT, which replaces joint inner-loop optimization of a two-layer memory network with layerwise learning. Each layer is assigned a local prediction target and updated through a state-dependent delta rule. This formulation retains a nonli
    
[^40]: AnyBottle：一种只保留真正所需概念的方法

    AnyBottle: A Recipe to Only Keep the Concepts You Really Need

    [https://arxiv.org/abs/2610.08552](https://arxiv.org/abs/2610.08552)

    AnyBottle提出了一种由黑盒教师模型引导的迭代概念选择方法，结合嵌套dropout训练，能够构建紧凑、任务特定的概念瓶颈模型，只保留任务真正需要的概念，使瓶颈更小且更易于检查。

    

    概念瓶颈模型（CBMs）通过将预测路由经过人类可解释的概念，使预测变得可检查和可干预，但最初需要概念标注。无需标注的变体虽然去除了这一要求，但通常使用大型概念词汇表，且在训练和推理时都是静态的，产生的瓶颈比任何任务或预测实际所需的都要大，也更难检查。我们提出AnyBottle，一种构建紧凑、任务特定的CBM的统一方法。AnyBottle仅假设有一个冻结的骨干网络和一个无监督概念池（例如稀疏自编码器）。随后，在同一骨干网络上训练的黑盒教师模型引导概念选择：每一轮添加最能解释瓶颈当前失败原因的概念，且候选概念被限制在教师与学生模型不一致的区域。通过按此选择顺序进行嵌套dropout训练，最终的瓶颈模型可以从任何概念前缀实现准确预测，因此推理时只需……（原文截断）

    arXiv:2610.08552v1 Announce Type: new  Abstract: Concept bottleneck models (CBMs) make predictions inspectable and intervenable by routing them through human-interpretable concepts, but originally required concept annotations. Annotation-free variants remove this requirement, but typically use large concept vocabularies, static at both training and inference, producing bottlenecks larger than any task or prediction needs and harder to inspect. We propose AnyBottle, a single recipe for building compact, task-specific CBMs. AnyBottle assumes only a frozen backbone and an unsupervised concept pool, such as a sparse autoencoder. A black-box teacher trained on the same backbone then guides selection: each round adds the concept that best explains the bottleneck's current failures, with candidates restricted to regions of teacher/student disagreement. Trained with nested dropout over this selection order, the final bottleneck predicts accurately from any concept prefix, so inference spends f
    
[^41]: 迈向对齐缩放定律：一个框架与首批预注册测量

    Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements

    [https://arxiv.org/abs/2610.08540](https://arxiv.org/abs/2610.08540)

    该论文提出将对齐视为一族可测量的幂律缩放关系（B_r(N)=a_rN^alpha_r）的框架及首批预注册测量，并证明长期对齐状态由经修正风险中的最大指数而非平均值决定，指数大于1时将累积不可持续的对齐债务。

    

    随着模型规模增长，对齐究竟变得更容易还是更难，这一争论往往基于孤立的发现，仿佛对齐是单一属性。我们将其视为一族可测量的缩放关系：对于每个风险类别 r，维持固定安全目标所需的对齐负担被建模为 B_r(N)=a_rN^alpha_r，其中 N 是能力代理指标；相对于与 N 成比例的预算，若 alpha_r<1，缩放有助于对齐；若 alpha_r≈1，缩放能保持同步；若 alpha_r>1，则会累积对齐债务。我们给出了负担的三种操作化定义，并区分了观测对齐、审计对齐与真实对齐。一个修正会消耗能力余量的玩具模型使这些后果变得明确。我们证明：决定长期状态的是经修正风险中的最大指数，而非平均值；当指数大于1时，任何要将余量维持在某一底线之上的策略都必须以超指数速度增长；对于作为幂律正混合的负担，在小模型上进行的拟合会低估大尺度上的（原文在此截断）。

    arXiv:2610.08540v1 Announce Type: new  Abstract: Whether alignment gets easier or harder as models grow is often argued from isolated findings, as if alignment were one property. We treat it as a family of measurable scaling relations: for each risk category r, the alignment burden needed to hold a fixed safety target is modeled as B_r(N)=a_rN^alpha_r, with N a capability proxy; against a budget proportional to N, scaling helps if alpha_r<1, keeps pace if alpha_r~1, and accumulates alignment debt if alpha_r>1. We give three operationalizations of burden and distinguish observed, audited and true alignment. A toy model, in which corrections consume capability headroom, makes the consequences explicit. We prove that the largest exponent among corrected risks, not an average, sets the long-run regime; that above 1 any policy holding headroom above a floor must grow super-exponentially; that, for burdens that are positive mixtures of power laws, fits on small models underestimate large-sca
    
[^42]: 从共享需求模式到局部不确定性：基于混合紧凑适配组件的概率负荷预测

    From Shared Demand Patterns to Local Uncertainty: Probabilistic Load Forecasting by Mixing Compact Adaptations

    [https://arxiv.org/abs/2610.08538](https://arxiv.org/abs/2610.08538)

    该论文提出了一种可扩展的概率负荷预测框架，通过共享模型学习共同需求模式，并让每个负荷按需混合一个小型低维紧凑适配组件库，从而在客户级和变压器级兼顾局部预测精度与大规模部署的可扩展性。

    

    概率负荷预测在电力系统运行与规划中已得到广泛研究，但客户级和变压器级的预测带来了独特的可扩展性挑战。在这些层级上，负荷不确定性受到客户行为、天气以及混合负荷构成的强烈影响，使得单一共享模型难以捕捉异质化的模式。为每个负荷使用独立的概率模型虽然可以提升局部精度，但在大规模应用时，其训练、存储、更新和验证的成本都十分高昂。为应对这一挑战，我们开发了一个可扩展的、客户感知的预测框架，该框架通过一个共享模型学习共同的需求行为，同时仅对一小部分紧凑的参数进行适配。所提出的设计并不为每个负荷使用独立模型，也不将每个负荷分配给某个专用模型，而是学习一个小型的低维适配组件库，并允许每个负荷根据其自身（特征）来组合这些组件……

    arXiv:2610.08538v1 Announce Type: cross  Abstract: Probabilistic load forecasting has been widely studied for power-system operation and planning, but customer- and transformer-level forecasting introduces a distinct scalability challenge. At these levels, load uncertainty is strongly affected by customer behavior, weather, and mixed load composition, making it difficult for a single shared model to capture heterogeneous patterns. Using separate probabilistic models can improve local accuracy, but becomes costly to train, store, update, and validate at scale. To address this challenge, we develop a scalable customer-aware forecasting framework that learns common demand behavior through a shared model while adapting only a compact subset of parameters. Rather than using an independent model for each load or assigning each load to a specialized model, the proposed design learns a small bank of low-dimensional adaptation components and allows each load to combine them according to its for
    
[^43]: FlowCF：基于流匹配的混合类型表格数据稀疏反事实解释方法

    FlowCF: Sparse Counterfactual Explanations for Mixed-Type Tabular Data using Flow Matching

    [https://arxiv.org/abs/2610.08537](https://arxiv.org/abs/2610.08537)

    FlowCF提出了一种基于流匹配的模型无关生成方法，通过新颖的混合流算子和门控网络，为混合类型表格数据生成具有稀疏性的反事实解释。

    

    在可解释人工智能（XAI）领域，反事实（CF）解释通过建议对输入进行哪些修改能够带来更有利的结果，从而解释模型的决策。为了在实际中发挥作用，这样的解释应该只改变少量特征，并且尽可能少地改变它们，这些性质被称为稀疏性和接近性。我们观察到，现有方法在这一方面仍然存在局限，尤其是针对数值特征，无论这些方法是模型无关且经过摊销训练的，还是基于梯度且完全访问模型的。在本文中，我们提出FlowCF，这是一种模型无关的生成方法，它将反事实样本的生成框架化为从事实样本到目标类别的稀疏传输问题。我们利用流匹配来解决这一传输问题，并通过一种新颖的混合流算子将其扩展到混合特征类型，同时利用所得的几何结构，通过一个最小化传输所改变特征数量的门控网络来优化稀疏性。

    arXiv:2610.08537v1 Announce Type: cross  Abstract: In the field of Explainable AI (XAI), counterfactual (CF) explanations interpret a model's decision by suggesting the changes to the input that would lead to a more favourable outcome. To be useful in practice, such an explanation should change few features and change them as little as possible, properties known as sparsity and proximity. We observe that existing methods remain limited in this respect, especially for numerical features, whether they are model-agnostic and amortised, or gradient-based with full access to the model. In this paper, we propose FlowCF, a model-agnostic generative method that frames CF generation as sparse transport from the factual to the target class. We solve this transport with flow matching, which we extend to mixed feature types with a novel mixed flow operator, and exploit the resulting geometry to optimise for sparsity through a gating network that minimises the number of features the transport chang
    
[^44]: Bregman 散度如何塑造 Shampoo

    How Bregman Divergences Shape Shampoo

    [https://arxiv.org/abs/2610.08534](https://arxiv.org/abs/2610.08534)

    本文提出统一的 Bregman 散度框架，揭示了 Shampoo 优化器中散度选择（如 Frobenius 与 KL 散度）如何影响 Kronecker 预条件近似，并发现某些散度能更好地补偿有限样本对经验二阶矩的低估，从而解释了不同 Shampoo 变体行为差异的根源。

    

    理解 Shampoo 背后的原理近来指导了更有效的神经网络优化器的开发。这些方法通过针对梯度二阶矩优化 Frobenius 散度或 Kullback-Leibler（KL）散度来学习预条件子。在这项工作中，我们研究了散度的选择如何塑造预条件化，这一问题仍不清楚并阻碍了进一步的改进。为此，我们开发了一个统一的 Bregman 散度框架，将所有流行的散度联系起来，使我们能够对它们进行联合研究。通过对梯度二阶矩的经验谱分析，我们检验了散度的选择如何影响 Kronecker 近似，以及如何与预条件化中的有限样本误差相互作用。我们发现某些散度能够更好地补偿经验二阶矩的有限样本低估，这有助于解释其相应 Shampoo 变体行为差异的原因。我们进一步验证了这一解释……

    arXiv:2610.08534v1 Announce Type: new  Abstract: Understanding the principles behind Shampoo has recently guided the development of more effective neural network optimizers. These methods learn a preconditioner by optimizing the Frobenius or Kullback-Leibler (KL) divergence against the gradient second moment. In this work, we investigate how the choice of divergence shapes preconditioning, which remains unclear and blocks further improvements. To do so, we develop a unified Bregman divergence framework that connects all popular divergences, allowing us to study them jointly. Through empirical spectral analysis of gradient second moments, we examine how divergence choice shapes Kronecker approximation and interacts with finite-sample error in preconditioning. We find that some divergences can better compensate for finite-sample underestimation of the empirical second moment, helping explain the differing behavior of their corresponding Shampoo variants. We further validate this explanat
    
[^45]: 超越扰动幅度：多模态几何表示中的方向依赖响应

    Beyond Perturbation Magnitude: Direction-Dependent Responses in Multimodal Geometric Representations

    [https://arxiv.org/abs/2610.08533](https://arxiv.org/abs/2610.08533)

    该研究通过受控扰动实验发现多模态几何对齐分数的响应并非由扰动幅度决定，并提出方向几何响应（DGR）指标——即位移在局部体积梯度上的投影——来刻画这种方向依赖的响应机制。

    

    基于Gram行列式的几何对齐分数为建模模态间的高阶一致性提供了一种紧凑的方式，然而此类分数对模态退化的响应机制尚不清楚。本文探讨多模态几何分数的响应是否主要由扰动引起的位移幅度所决定。我们使用来自MSR-VTT（N=878）和DiDeMo（N=980）的冻结数据集，施加受控的视频模糊和音频噪声，并分析分数所定义的关系几何中的响应。结果表明，位移幅度最多只能解释绝对响应中15%的样本外方差，且幅度匹配的样本对表现出系统性不同的响应，因此标量幅度并不能组织这种响应。通过对Gramian体积进行闭式一阶展开，我们导出了方向几何响应（DGR）：即位移在局部体积梯度上的投影，该联合……

    arXiv:2610.08533v1 Announce Type: cross  Abstract: Geometric alignment scores based on Gram determinants provide a compact way to model higher-order consistency among modalities, yet how such scores respond to modality degradation is poorly understood. This paper asks whether the response of a multimodal geometric score is determined primarily by the magnitude of the perturbation-induced displacement. Using frozen cohorts from MSR-VTT (N=878) and DiDeMo (N=980), we apply controlled video blur and audio noise and analyze the response in the relational geometry on which the score is defined. Displacement magnitude explains at most 15% of the out-of-sample variance in the absolute response, and magnitude-matched pairs respond systematically differently, so scalar magnitude does not organize the response. The closed-form first-order expansion of the Gramian volume yields the Directional Geometric Response (DGR): the projection of the displacement onto the local volume gradient, which joint
    
[^46]: PHBA：前缀状态混合块注意力

    PHBA: Prefix-State Hybrid Block Attention

    [https://arxiv.org/abs/2610.08527](https://arxiv.org/abs/2610.08527)

    PHBA提出了一种混合注意力架构，用top-k块稀疏检索取代局部滑窗注意力，并将每个检索到的token块与其前置上下文的紧凑前缀状态相耦合，从而在统一层内同时实现精确的长程token检索与高效的历史上下文压缩。

    

    将线性序列模型与softmax注意力相结合的混合架构，在高效的长上下文建模与精确的token检索之间提供了有效的平衡。诸如原生混合注意力（NHA）等现有设计将压缩的长期状态与滑窗注意力相结合，但其精确注意力被限制在固定的局部窗口内。在本工作中，我们提出了前缀状态混合块注意力（PHBA），它用top-k块稀疏检索取代了局部滑窗注意力，并将每个检索到的块与一个概括其前置上下文的紧凑前缀状态相耦合。前缀状态通过门控线性递归在块边界处构建，并与相应的token块一起被检索，使模型能够在统一层内将精确的长程证据与压缩的历史上下文相结合。我们进一步开发了一种硬件感知的Triton实现，对流式路由的token……（注：原摘要在此处被截断）

    arXiv:2610.08527v1 Announce Type: new  Abstract: Hybrid architectures combining linear sequence models with softmax attention provide an effective balance between efficient long-context modeling and precise token retrieval. Existing designs such as Native Hybrid Attention (NHA) combine compressed long-term states with sliding-window attention, but their exact attention is restricted to a fixed local window. In this work, we introduce Prefix-State Hybrid Block Attention (PHBA), which replaces local sliding-window attention with top-k block-sparse retrieval and couples each retrieved block with a compact prefix state summarizing its preceding context. The prefix states are constructed by a gated linear recurrence at block boundaries and retrieved together with the corresponding token blocks, allowing the model to combine precise long-range evidence with compressed historical context within a unified layer. We further develop a hardware-aware Triton implementation that streams routed toke
    
[^47]: X-OPM：面向增强鲁棒性的可解释自动数字片上功耗建模

    X-OPM: Explainable Automatic Digital On-Chip Power Modeling for Enhanced Robustness

    [https://arxiv.org/abs/2610.08502](https://arxiv.org/abs/2610.08502)

    X-OPM基于同步数字VLSI电路设计原理，提出了一个可解释的自动片上功耗建模框架，通过树模型捕捉特征交互并用线性模型进行预测，结合人在回路的工作流程，在商用C906向量处理器上实现了更鲁棒、可泛化且低开销的功耗预测。

    

    主动式功耗管理系统通过运行时功耗预测和功耗感知调度来降低处理器的动态功耗。准确、稳定且低开销的数字片上功耗计对于提升预测质量至关重要。近期研究探索了多种建模方法，包括使用线性模型、决策树和多层感知机（MLP）来构建片上功耗计。然而，当前大多数方法采用端到端方式训练模型，而未分析特征的物理可解释性，这影响了模型对未见工作负载的泛化能力。基于同步数字VLSI电路的设计原理，X-OPM引入了一个鲁棒的特征工程框架，利用基于树的模型来捕捉特征交互，并采用线性模型进行预测。该框架还融合了人在回路的工作流程，以平衡模型精度与建模工作量。该方法在商用C906向量处理器上进行了评估。

    arXiv:2610.08502v1 Announce Type: cross  Abstract: Proactive power management systems reduce processor dynamic power through runtime power prediction and power-aware scheduling. Accurate, stable and low-overhead digital on-chip power meters (OPMs) are crucial for improving the prediction quality. Recent studies have explored various modeling methods, including using linear models, decision trees, and multi-layer perceptrons (MLPs) to construct OPMs. However, most current approaches train models end-to-end without analyzing the physical interpretability of features, affecting their ability to generalize to unseen workloads. Grounded in the design principles of synchronous digital VLSI circuits, X-OPM introduces a robust feature engineering framework that uses tree-based models to capture feature interactions and linear models for prediction. It also incorporates a human-in-the-loop workflow to balance model accuracy against modeling effort. Evaluated on a commercial C906 vector processo
    
[^48]: 面向分子发现的信息密集型合成

    Information-Dense Synthesis for Molecular Discovery

    [https://arxiv.org/abs/2610.08495](https://arxiv.org/abs/2610.08495)

    提出信息密集型合成方法，通过设计、合成复杂分子混合物并池化测试后解卷积分子-活性映射，理论上可将寻找最优分子的实验次数从O(d)降至O(log d)或O(1)，比现有贝叶斯优化方法效率提升一个数量级。

    

    机器学习可以通过设计分子和规划实验来加速分子发现。然而，许多科学挑战需要具有极为罕见性质的分子，在这种稀疏设定下，现有算法相比随机猜测几乎没有优势。我们提出了一种利用算法控制的随机合成来高效搜索大范围分子空间的方法。我们不逐一设计、合成并测试单个分子，而是设计和合成复杂的混合物，将其作为一个池进行测试，然后解卷积出分子-活性映射关系。我们对合成过程进行优化以编码最大信息量。从理论上讲，该方法可以将从 $d$ 个候选分子中找到最优分子所需的实验次数从 $\mathcal{O}(d)$ 降低到 $\mathcal{O}(\log d)$ 甚至 $\mathcal{O}(1)$。在基于估计的蛋白质适应度景观的模拟中，该方法找到活性分子所需的实验次数比现有贝叶斯

    arXiv:2610.08495v1 Announce Type: cross  Abstract: Machine learning can accelerate molecular discovery by designing molecules and planning experiments. However, many scientific challenges demand molecules with very rare properties, and in this sparse setting, existing algorithms offer little gain over random guessing. We propose a method to efficiently search large regions of molecular space using algorithmically controlled stochastic synthesis. Rather than design, make and test individual molecules, we design and make complex mixtures, test them as a pool, then deconvolute the molecule-activity map. We optimize synthesis to encode maximal information. Theoretically, this approach can reduce the number of experiments required to find the optimal molecule among $d$ candidates from $\mathcal{O}(d)$ to $\mathcal{O}(\log d)$ or $\mathcal{O}(1)$. In simulation, on estimated protein fitness landscapes, it finds active molecules with an order of magnitude fewer experiments than existing Bayes
    
[^49]: MetaLearnNCA：基于交互式神经细胞自动机的少样本离线元学习

    MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata

    [https://arxiv.org/abs/2610.08479](https://arxiv.org/abs/2610.08479)

    提出去中心化框架MetaLearnNCA，通过Active-NCA与Meta-NCA两种耦合神经细胞自动机的动态交互实现少样本离线元学习，无需测试时反向传播计算梯度，同时保留二维空间几何结构信息。

    

    少样本元学习传统上将任务适应表述为通过展开计算图进行的解析梯度下降，或基于展平的一维特征向量的度量式距离比较，前者会产生高昂的测试时反向传播开销，后者则丢弃了原生的二维空间几何结构。在本工作中，我们提出了MetaLearnNCA，这是一个去中心化框架，通过耦合的神经细胞自动机（NCA）之间的动态交互实现少样本适应，且在推理过程中无需计算解析梯度。MetaLearnNCA将任务适应分解为两个部分：Active-NCA，它基于一个被称为“空间程序”的连续二维空间记忆网格来执行任务推断；以及通过学习得到的Meta-NCA，它作为一个去中心化的细胞优化器，通过在局部邻域之间扩散空间误差残差来动态更新该空间程序。MetaLearnNCA与经典的元学习方法相比具有竞争力。

    arXiv:2610.08479v1 Announce Type: cross  Abstract: Few-shot meta-learning traditionally formulates task adaptation either as analytical gradient descent through unrolled computational graphs or as metric-based distance comparisons over flattened 1D fea- ture vectors, which either incur costly test-time backpropagation or discard native 2D spatial geometry. In this work, we propose METALEARNNCA, a decentralized framework that achieves few-shot adapta- tion through the dynamical interaction of coupled Neural Cellular Automata (NCAs) without computing analytical gradients during inference. MetaLearnNCA decomposes task adaptation into an Active- NCA, which executes task inference conditioned on a continuous 2D spatial memory grid termed the spatial program, and a learned Meta-NCA, which acts as a decentralized cellular optimizer by diffusing spatial error residuals across local neighborhoods to dynamically update this program. METALEARN- NCA is competitive against canonical meta-learners i
    
[^50]: 基于潜在动力学网络学习具有可变初始条件的偏微分方程解算子

    Learning PDE solution operators with variable initial conditions via Latent Dynamics Networks

    [https://arxiv.org/abs/2610.08475](https://arxiv.org/abs/2610.08475)

    本文提出一种改进的潜在动力学网络（LDNet），通过从少量早期观测中直接推断初始潜在状态来支持可变初始条件，在保留端到端训练、无编码器设计以及与网格拓扑无关等优势的同时，显著扩展了其在真实PDE应用中的适用性。

    

    在多查询场景中，数据驱动的代理模型为模拟由偏微分方程（PDE）支配的物理系统提供了一种高效替代高保真求解器的方案。在此背景下，潜在动力学网络通过将神经常微分方程与非线性降维相结合，近期在预测时空系统的响应方面展现出了卓越的性能。然而，其原始形式假设初始条件是固定的，这限制了它在许多系统从不同起始状态演化的现实应用中的适用性。在本工作中，我们克服了这一限制，同时保留了原始LDNet的端到端训练过程及其无编码器的特性，从而保持了其与空间分辨率和网格拓扑的内在独立性。我们直接从少量早期时刻的观测数据中推断初始潜在状态，并将后……

    arXiv:2610.08475v1 Announce Type: new  Abstract: In many-query scenarios, data-driven surrogate models provide an efficient alternative to high-fidelity solvers for simulating physical systems governed by Partial Differential Equations (PDEs). In this context, the Latent Dynamics Network (LDNet) has recently demonstrated remarkable performance in predicting the response of spatio-temporal systems, combining Neural Ordinary Differential Equations with nonlinear dimensionality reduction. However, the original formulation assumes a fixed initial condition, limiting its applicability to many real-world applications where a system evolves from varying starting states. In this work, we overcome this limitation while keeping the end-to-end training procedure of the original LDNet and its encoder-free nature, which preserves its intrinsic independence from spatial resolution and grid topology. We infer the initial latent state directly from a small set of early-time observations, treating late
    
[^51]: UNREAL：用单一模型统一检索与长上下文

    UNREAL: Unifying Retrieval and Long-Context with a Single Model

    [https://arxiv.org/abs/2610.08463](https://arxiv.org/abs/2610.08463)

    UNREAL提出了一种模型原生的证据选择框架，直接从冻结LLM的内部表示中推导检索查询，以不到50万可训练参数统一了语料库检索与长上下文推理，并在多个基准上大幅超越最先进的检索-重排序系统。

    

    长上下文推理和检索增强生成（RAG）在截然不同的尺度上处理证据选择问题，范围从单个长提示词到整个语料库。我们探究是否存在一种单一的模型内部机制能够覆盖这一范围并完成证据选择。我们提出了UNREAL（用单一模型统一检索与长上下文），一个模型原生的证据选择框架，可同时覆盖语料库检索和长上下文推理。UNREAL对文本块进行编码，并直接从冻结的大语言模型（LLM）的内部表示中推导检索查询。它仅增加不到50万个可训练参数，且完全不改动骨干模型。在一个包含30亿标记、2100万文本块的维基百科索引上，全部四种稠密型和混合型UNREAL骨干模型均超越了最先进的检索器-重排序器系统。其中最佳模型将HotpotQA上的召回率从49.1%提升至73.2%，将2WikiMultiHopQA上的召回率从31.7%提升至60.1%，将MuSiQue上的召回率从8.8%提升至14.4%。当应用于长上下文任务时，同样的选择机制……（摘要到此截断）

    arXiv:2610.08463v1 Announce Type: new  Abstract: Long-context inference and Retrieval-Augmented Generation (RAG) handle evidence selection at vastly different scales, from a single long prompt to an entire corpus. We ask whether a single model-internal mechanism can select evidence across this range. We introduce UNifying REtrieval And Long-Context with a Single Model (UNREAL), a model-native evidence selection framework to span corpus retrieval and long-context inference. UNREAL encodes chunks and derives retrieval queries directly from the frozen LLM's internal representations. It adds fewer than 500K trainable parameters and leaves the backbone unchanged. On a 3B-token, 21M-chunk Wikipedia index, all four dense and hybrid UNREAL backbones outperform state-of-the-art retriever-reranker systems. The best model raises recall from 49.1% to 73.2% on HotpotQA, from 31.7% to 60.1% on 2WikiMultiHopQA, and from 8.8% to 14.4% on MuSiQue. Applied to long-context tasks, the same selection mecha
    
[^52]: 攀登设计阶梯：面向早期电路时序预测的顺序知识蒸馏

    Climbing the Design Ladder: Sequential Knowledge Distillation for Early-Stage Circuit Timing Prediction

    [https://arxiv.org/abs/2610.08457](https://arxiv.org/abs/2610.08457)

    提出STEP-KD方法，将中间设计阶段（布局规划后、布局后、布线后）作为“垫脚石”进行顺序渐进的知识蒸馏，从而从早期设计数据准确预测布线后电路时序，避免后期才发现时序违例导致的高昂迭代成本。

    

    集成电路设计涉及多个设计阶段：逻辑综合、布局规划、布局和布线，每个阶段需要数小时到数周才能完成。如果在设计流程后期才发现时序违例，将迫使代价高昂的迭代返回到早期阶段，浪费计算资源并延误产品上市。虽然从早期数据预测布线后时序可以避免这些失败，但现有的机器学习方法难以应对综合后逻辑描述与布线后物理布局之间的巨大抽象鸿沟。我们提出了STEP-KD（基于渐进式知识蒸馏的顺序时序评估），该方法利用中间设计阶段作为渐进式知识转移的“垫脚石”，而非尝试直接预测。STEP-KD在布线后、布局后和布局规划后阶段分别训练教师模型，然后依次将其知识蒸馏到一个……（原文摘要至此截断）

    arXiv:2610.08457v1 Announce Type: new  Abstract: Integrated circuit design involves multiple design stages: logic synthesis, floorplanning, placement, and routing, with each stage taking hours to weeks to complete. Discovering timing violations late in this flow forces costly iterations back to earlier stages, wasting computational resources and delaying product launches. While predicting post-routing timing from early-stage data could prevent these failures, existing machine learning approaches struggle with the massive abstraction gap between post-synthesis logical descriptions and post-routing physical layouts. We propose STEP-KD (Sequential Timing Evaluation via Progressive Knowledge Distillation), which leverages intermediate design stages as ``stepping stones'' for progressive knowledge transfer rather than attempting direct prediction. STEP-KD trains teacher models at the post-routing, post-placement, and post-floorplan stages, then sequentially distills their knowledge to a pos
    
[^53]: Agentic AutoRAG：通过推理驱动的智能体进行RAG流程优化

    Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents

    [https://arxiv.org/abs/2610.08452](https://arxiv.org/abs/2610.08452)

    该论文提出Agentic AutoRAG，一种利用LLM智能体进行多目标RAG超参数优化的方法，其核心创新在于通过诊断器将每次失败归因于检索或生成阶段，从而让优化器能够推理配置失败的原因并智能地指导后续搜索。

    

    检索增强生成（RAG）是一种被广泛使用的方法，用于将大语言模型（LLM）扎根于外部知识。然而，配置一个RAG流程是一个代价高昂的超参数优化问题，涉及众多相互关联的选择，从分块和嵌入模型到重排序和生成。现有的优化器，从贪心搜索到贝叶斯优化，都将每次试验简化为一个汇总分数进行搜索，而不会建模某个配置为何会有那样的表现，尽管检索到的文本块其实已经提供了关于每次失败究竟发生在检索阶段还是检索之后的证据。我们提出了Agentic AutoRAG，一个用于多目标RAG超参数优化的LLM智能体优化器，具备检索与生成之间的失败归因能力。它提出候选配置，并在从语料库构建的固定考题上进行评分：每次试验后，一个诊断器会将每个失败的问题归因于检索阶段或生成阶段，而一个提议器则基于……（原文摘要在此处被截断，后续内容未能提供）

    arXiv:2610.08452v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is a widely used approach for grounding large language models (LLMs) in external knowledge. However, configuring a pipeline is an expensive hyperparameter optimization problem over many interacting choices, from chunking and embedding model to reranking and generation. Existing optimizers, from greedy search to Bayesian optimization, reduce each trial to an aggregate score and search without modeling why a configuration performed as it did, even though the retrieved chunks already provide evidence about whether each failure occurred during retrieval or after it. We introduce Agentic AutoRAG, an LLM-agent optimizer for multi-objective RAG hyperparameter optimization with retrieval-versus-generation failure attribution. It proposes configurations scored on a frozen exam from the corpus: after each trial a Diagnoser attributes each failed question to retrieval or generation, and a Proposer, grounded in a
    
[^54]: 对称感知特征学习：多指标模型的多项式分离

    Symmetry-Aware Feature Learning: A Polynomial Separation for Multi-Index Models

    [https://arxiv.org/abs/2610.08420](https://arxiv.org/abs/2610.08420)

    该论文证明了对称感知与对称无关特征学习之间存在多项式级的样本复杂度分离：在具有循环对称轨道的增长秩多指标模型中，通过权重共享或全群数据增强利用对称性的学习器，仅需约 $d^{p-1}$ 个样本（对于信息指数 $p\ge3$ 的多项式链接函数）即可实现弱方向恢复，而无法利用对称性的学习器则代价更高。

    

    我们建立了对称感知（symmetry-aware）与对称无关（symmetry-agnostic）特征学习之间的多项式样本复杂度分离。我们研究了位于 $\mathbb{R}^d$ 中的高维高斯协变量下、秩随维度增长的多指标模型，其中 $r=\Theta(d^\delta)$ 个教师方向构成一个循环对称轨道，且 $0<\delta<1/2$。我们比较了利用这一结构的三种方式：架构层面的权重共享、在完整对称群上的数据增强，以及无法获取对称性时的学习。特别地，我们分析了对称绑定（权重共享）的卷积网络、非绑定网络，以及使用全群数据增强训练的同一非绑定网络，三者均采用带相关损失的球面在线SGD进行训练。对于一类信息指数 $p\ge3$ 的多项式链接函数，我们在对数因子内证明了相互匹配的样本复杂度界：绑定与增强的学习器在 $\widetilde{\Theta}(d^{p-1})$ 个样本内即可实现弱方向恢复，而……

    arXiv:2610.08420v1 Announce Type: new  Abstract: We establish a polynomial sample complexity separation between symmetry-aware and symmetry-agnostic feature learning. We study growing-rank multi-index models with high-dimensional Gaussian covariates in $\mathbb{R}^d$ and $r=\Theta(d^\delta)$ teacher directions forming a cyclic symmetry orbit, where $0<\delta<1/2$. We compare three ways of exploiting this structure: architectural weight sharing, data augmentation over the full symmetry group, and learning without access to the symmetry. In particular, we analyze a symmetry-tied convolutional network, an untied network, and the same untied network trained with full-group data augmentation, using spherical online SGD with correlation loss. For a class of polynomial links with information exponent $p\ge3$, we prove matching sample complexity bounds up to logarithmic factors: the tied and augmented learners achieve weak directional recovery in $\widetilde{\Theta}(d^{p-1})$ samples, whereas 
    
[^55]: 知道何时不应回答：潜在欠规范信号的跨域与多轮泛化

    Knowing When Not to Answer: Cross-Domain and Multi-Turn Generalization of Latent Underspecification Signals

    [https://arxiv.org/abs/2610.08413](https://arxiv.org/abs/2610.08413)

    该论文构建了一个带轮次标签的多轮对话不可回答性基准与模拟用户评估框架，发现线性探针所捕捉的“信息缺失”信号能在共享同一不可回答性根源的数据集间稳健跨域迁移（AUROC 0.77–0.97），但不同类型不可回答性的表征边界会受词汇混淆、网络层级与坐标系选择的影响。

    

    大型语言模型经常回答那些无法根据已有信息回答的问题，并且在对话中往往在信息尚不充分时就贸然作答。已有研究表明，“不可回答性”可以从模型隐藏状态中被线性解码出来，但目前尚不清楚哪些形式的不可回答性共享同一表征，以及该信号在对话场景中是否实用。本文贡献了一个带轮次标签的多轮基准数据集（423段对话、1,661个带标签的轮次状态），以及一个配有可回答澄清性问题的模拟用户的评估框架，并结合六个数据集与六个开源权重的大语言模型，系统检验不可回答性探针的泛化边界。结果显示，在共享同类不可回答性根源的数据集之间，探针能够稳健迁移：数学中的信息缺失（AUROC 0.77–0.97），以及文本阅读中的信息缺失（SQuAD 2.0<->MuSiQue，0.77–0.90）。相比之下，针对认识论意义上“已知未知”（known-unknowns）的探针向数学任务迁移效果较差，但这种分离在引入词汇控制后会减弱，并随网络层级和坐标系的选择而变化，因此……（原文摘要在此处截断）

    arXiv:2610.08413v1 Announce Type: cross  Abstract: Large language models routinely answer questions that cannot be answered from the information given, and in dialogue they answer before enough has been said. Unanswerability is linearly decodable from hidden states, but it is unclear which of its forms share a representation and whether the signal is useful in dialogue. We contribute a turn-labeled multi-turn benchmark (423 conversations, 1,661 labeled turn-states) and an evaluation harness with a simulated user who answers clarifying questions, and use them with six datasets and six open-weight LLMs to test how far probes for unanswerability carry. Probes transfer robustly between datasets that share a ground of unanswerability: missing information in math (AUROC 0.77-0.97) and in a passage (SQuAD 2.0<->MuSiQue, 0.77-0.90). Probes for epistemic "known-unknowns" transfer poorly to math, but this separation weakens under lexical controls and changes with layer and coordinate system, so 
    
[^56]: SSR：面向三值GEMM加速的稀疏分段规约

    SSR: Sparse Segment Reduction for Ternary GEMM Acceleration

    [https://arxiv.org/abs/2610.08403](https://arxiv.org/abs/2610.08403)

    本文提出SSR方法，通过专用优化的三值数据格式与利用稀疏结构的计算树算法，同时兼顾三值特性与稀疏性，从而加速三值大语言模型的矩阵乘法推理。

    

    大型语言模型（LLM）需要大量计算资源，这限制了其在资源受限硬件上的部署。三值LLM通过三值权重量化来缓解这些需求，实现了显著的压缩，通常具有50-90%的稀疏度。然而，现有方法存在局限性：针对三值权重优化的方法（如BitNet、冗余分段规约RSR及其改进版本RSR++）未能利用稀疏结构，而传统的稀疏格式则忽略了三值特性，错失了双重优化的机会。本文提出了稀疏分段规约（SSR），一种旨在加速三值LLM及通用三值权重网络（TWN）推理的三值矩阵乘法方法。SSR拥有专门优化的三值数据格式，以及一种通过随……

    arXiv:2610.08403v1 Announce Type: new  Abstract: Large Language Models (LLMs) require substantial computational resources, limiting their deployment on resource-constrained hardware. Ternary LLMs mitigate these demands through weight quantization via ternary values, achieving significant compression often with 50-90% sparsity. However, existing approaches have limitations: methods optimized for ternary weights, such as BitNet, redundant segment reduction (RSR), and its improved version RSR++, do not exploit sparsity structures, while conventional sparse formats neglect ternary characteristics, foregoing dual optimization opportunities.   In this paper, we introduce Sparse Segment Reduction (SSR), a ternary matrix multiplication method designed to accelerate the inference of ternary LLMs and general Ternary Weight Networks (TWNs). SSR has a dedicated optimized ternary data format and an algorithm that systematically exploits sparsity patterns through computation trees that scale with th
    
[^57]: VETTA：为多轮LLM智能体协调轮次级与令牌级信用分配

    VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents

    [https://arxiv.org/abs/2610.08402](https://arxiv.org/abs/2610.08402)

    VETTA通过共享轻量级评论家上的独立价值头联合学习轮次级与令牌级价值，在单次策略更新中协调两个层面的信用分配，从而解决多轮LLM智能体面临稀疏反馈时的信用分配难题。

    

    多轮LLM智能体通常在多次交互中接收到稀疏的任务反馈，同时逐个令牌地生成每个响应。这产生了两个相关的信用分配问题：哪些响应有助于实现最终结果，以及每个响应中哪些生成决策是关键的？现有方法通常只关注其中一个层面：轮次级方法评估完整的响应，但无法区分响应内部的决策；令牌级方法可以跨轮次传播反馈，但不会显式地为每个响应建模信用。这些互补的局限性促使我们在两个层面同时学习信用，并在单次策略更新中对其进行协调。我们提出了VETTA，这是一种信用分配方法，通过共享轻量级评论家上的独立价值头，联合学习轮次级和令牌级价值。VETTA沿两个时间序列计算优势函数，并将每个轮次优势与响应内中心化的令牌修正项相结合。

    arXiv:2610.08402v1 Announce Type: new  Abstract: Multi-turn LLM agents often receive sparse task feedback across several interactions, while generating each response token by token. This creates two related credit-assignment questions: which responses helped achieve the outcome, and which generation decisions mattered within each response? Existing methods typically focus on only one level: turn-level methods evaluate complete responses but do not distinguish the decisions within them; token-level methods can propagate feedback across turns but do not explicitly model credit for each response. These complementary limitations motivate learning credit at both levels and coordinating it in a single policy update. We introduce VETTA, a credit assignment method that jointly learns turn- and token-level values through separate heads on a shared lightweight critic. VETTA computes advantages along both temporal sequences and combines each turn advantage with a within-response-centered token re
    
[^58]: Atom-JEPA：面向三维原子系统的联合嵌入预测架构

    Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems

    [https://arxiv.org/abs/2610.08400](https://arxiv.org/abs/2610.08400)

    Atom-JEPA是一种自监督预训练框架，通过互补的原子级和子结构级目标从无标签三维原子结构中学习潜在表示，在分子ADMET和量子化学性质预测等下游任务上达到了最先进的性能。

    

    大规模自监督预训练已经重塑了现代机器学习，显著提升了语言和视觉模型在下游任务上的泛化能力。尽管深度学习近年来在原子系统建模方面取得了相当大的进展，但该领域的自监督预训练尚未实现与之相当的下游泛化能力。为解决这一问题，我们提出了Atom-JEPA，这是一个自监督预训练框架，它受联合嵌入预测架构的启发，通过互补的原子级和子结构级目标，从无标签的三维结构中学习潜在表示。我们在大规模分子和晶体数据集上对Atom-JEPA进行预训练，并通过在多样化的下游性质预测任务集上进行微调来评估其迁移性能。Atom-JEPA在分子ADMET和量子化学性质预测任务上取得了最先进的性能，

    arXiv:2610.08400v1 Announce Type: cross  Abstract: Large-scale self-supervised pretraining has reshaped modern machine learning, substantially advancing the ability of language and vision models to generalize across downstream tasks. While deep learning has driven considerable progress in modeling atomistic systems in recent years, self-supervised pretraining in this domain has not yet achieved comparable downstream generalization. To address this, we introduce Atom-JEPA, a self-supervised pretraining framework that learns latent representations from unlabeled 3D structures through complementary atom-level and substructure-level objectives inspired by joint-embedding predictive architectures. We pretrain Atom-JEPA on large-scale molecular and crystalline datasets and evaluate its transfer performance by fine-tuning on a diverse set of downstream property prediction tasks. Atom-JEPA achieves state-of-the-art performance on molecular ADMET and quantum-chemical property prediction tasks, 
    
[^59]: 马尔可夫决策过程中的决策聚焦学习：一种占用测度方法

    Decision-Focused Learning in MDPs: An Occupancy Measure Approach

    [https://arxiv.org/abs/2610.08384](https://arxiv.org/abs/2610.08384)

    该论文提出将马尔可夫决策过程的决策聚焦学习重构为基于占用测度的线性规划，利用闭式梯度、增广拉格朗日代理及随机行草图平滑技术，克服了传统基于KKT条件方法需要大规模求解线性系统的可扩展性瓶颈。

    

    在本工作中，我们研究了马尔可夫决策过程（MDP）中的决策聚焦学习（DFL）。现有方法通过贝尔曼方程的KKT条件进行反向传播，需要在所有状态-动作对上求解线性系统，限制了其可扩展性。我们通过将MDP重新表述为基于占用测度的线性规划（LP）来解决这一问题，该规划的可行域由预测的动力学诱导，并通过枢轴算法识别可行多面体中的活动约束，从而推导出闭式梯度。这种基于占用测度的LP层带来两个挑战：（1）当活动约束发生变化时，LP解的梯度不连续；（2）LP的反向传播成本仍随状态规模增长，对于大型或连续状态空间而言代价高昂。我们采用增广拉格朗日代理来应对这些挑战，并通过约束的随机行草图采样来平滑边界跳变，以及……（原文摘要在此处截断）

    arXiv:2610.08384v1 Announce Type: new  Abstract: In this work, we consider decision-focused learning (DFL) for a Markov decision process (MDP), where existing methods differentiate through the KKT conditions of the Bellman equation and require solving a linear system over all state-action pairs, limiting its scalability. We address this by reformulating the MDP as an occupancy measure-based linear program (LP), whose feasible region is induced by predicted dynamics, and we derive a closed-form gradient by identifying the active constraints in the feasible polyhedron via the pivoting algorithm. This occupancy measure-based LP layer raises two challenges: (1) LP's solution gradient is discontinuous when active constraints change, and (2) the LP backward cost still scales with the state size, which is costly for large or continuous state spaces. We address the challenges with an augmented Lagrangian surrogate and smooth the boundary jumps by random row sketching of the constraints, and a 
    
[^60]: 通过AI驱动的多目标优化加速PLGA原位成型储库的开发

    Accelerating the Development of PLGA In Situ Forming Depots Through AI-Driven Multi-Objective Optimization

    [https://arxiv.org/abs/2610.08368](https://arxiv.org/abs/2610.08368)

    该研究将Corbion的PURASORB聚合物库与Intrepid Labs的AI算法ANDROMEDA 1结合，仅用约15周和181个处方即完成多目标优化，成功筛选出4个满足黏度与可注射性要求且具有差异化30天释放曲线的治疗性多肽PLGA原位成型储库处方，显著加速了长效注射制剂的开发进程。

    

    开发长效注射制剂需要同时优化载药量、释放动力学、黏度、可注射性、稳定性等多个目标。为了在这一多维空间中高效探索，Corbion与Intrepid将Corbion丰富多样的PURASORB可生物降解聚合物库与Intrepid Labs的专有人工智能算法（ANDROMEDA 1）相结合，为一种治疗性多肽开发原位成型储库。在大约15周内，研究共制备并表征了181个独特处方，载药量范围为6–12% w/w，通过广泛的设计空间映射和有针对性的多目标优化，最终在6%、9%和12% w/w载药量下筛选出四个候选先导处方。这些处方均满足预先设定的黏度和可注射性标准，同时呈现出各具特色的30天体外释放曲线。该研究评估了涵盖宽分子量范围的多种聚合物，包括市售的PURASORB g（摘要在此处被截断）

    arXiv:2610.08368v1 Announce Type: cross  Abstract: Developing long-acting injectable formulations requires the simultaneous optimization of drug loading, release kinetics, viscosity, injectability, stability and other objectives. To navigate this multidimensional space, Corbion and Intrepid combined Corbion's diverse PURASORB bioresorbable polymer library with Intrepid Labs' proprietary AI algorithm (ANDROMEDA 1) to develop in situ forming depots for a therapeutic peptide. Over approximately 15 weeks, 181 unique formulations spanning drug loadings of 6-12% w/w were prepared and characterized through broad design-space mapping and targeted multi-objective optimization. Four lead candidate formulations were identified at 6%, 9%, and 12% w/w drug loading. Each met the predefined viscosity and injectability criteria while providing distinct 30-day in vitro release profiles. The study evaluated polymers spanning a broad range of molecular weights, including commercially available PURASORB g
    
[^61]: 进化式单步生成器：面向离散设计的快速多样采样

    Evolutionary One-Step Generators: Fast and Diverse Sampling for Discrete Design

    [https://arxiv.org/abs/2610.08367](https://arxiv.org/abs/2610.08367)

    提出EGO框架，利用对偶低秩进化策略直接在离散输出上训练生成器，使其单次神经网络前向计算即可生成整个图，在分子生成等离散设计任务中以极低计算成本同时保证候选的有效性与多样性。

    

    许多离散设计任务（如分子发现）需要在低计算成本下获得多样化的有用候选集合。仅具备高有效性并不能保证得到有用的候选库：反复生成相同的有效结构会使得几乎没有不同的备选方案。同时针对可行性与多样性进行训练颇具挑战性，因为许多相关评估准则只能在硬解码之后才能计算。为应对这一挑战，我们提出了EGO（单步推理进化生成器），这是一个直接在离散输出上训练紧凑生成器的框架。该方法将分布匹配与结构约束以及可选的多样性奖励或依赖历史的奖励相结合，采用对偶低秩进化策略，无需针对特定准则构建可微代理模型。训练完成后，生成器仅需一次神经网络前向计算即可生成整个图。在分子生成基准测试中，我们的方法（摘要原文在此处被截断）。

    arXiv:2610.08367v1 Announce Type: new  Abstract: Several discrete design tasks, such as molecular discovery, require diverse collections of useful candidates at low computational cost. High validity alone does not guarantee a useful candidate library: repeatedly generating the same valid structures leaves few distinct alternatives. Training for both feasibility and diversity is challenging because many relevant criteria can only be evaluated after hard decoding. To address this challenge, we propose EGO (Evolutionary Generators with One-step inference), a framework for training compact generators directly on discrete outputs. The method combines distribution matching with structural constraints and optional diversity or history-dependent rewards, using antithetic low-rank evolution strategies without requiring criterion-specific differentiable surrogates. Once trained, the generator produces the entire graph in a single neural-network evaluation. On molecular generation benchmarks, our
    
[^62]: 传感器几何作为多通道脑信号的流匹配先验

    Sensor Geometry as a Flow-Matching Prior for Multi-Channel Brain Signals

    [https://arxiv.org/abs/2610.08355](https://arxiv.org/abs/2610.08355)

    该论文提出仅利用脑电电极的几何坐标构建k近邻图，并以图拉普拉斯的Matérn函数作为流匹配模型的源协方差，从而把已知的空间相关结构编码为先验，在不增加任何可学习参数的情况下替代各向同性高斯源，生成空间相干的多通道脑电信号。

    

    流匹配模型从各向同性高斯源出发，这是数据相关性结构事先未知时的标准选择。然而对于多通道脑电记录，其中部分结构是事先已知的：电极位于头部的固定位置，而经由颅骨和头皮的容积传导使得相邻电极以跨被试共享的方式共同变化。尽管如此，现有的EEG生成模型仍然让网络从头学习这种结构。我们将这一结构直接置于源分布中：仅凭传感器坐标构建k近邻图，并取其图拉普拉斯算子的Matérn函数作为源协方差，使流从空间相干的模式出发，而非通道独立的噪声。这一改动不引入任何可学习参数，可配合任意耦合方案与任意漂移网络使用，且在所有数据集上均使用相同的三个超参数。在八个EEG数据集上的（实验结果摘要在此处截断）

    arXiv:2610.08355v1 Announce Type: cross  Abstract: Flow-matching models start from an isotropic Gaussian source, the standard choice when the correlation structure of the data is unknown in advance. For multi-channel brain recordings, however, part of this structure is known in advance. Electrodes sit at fixed positions on the head, and volume conduction through the skull and scalp makes nearby electrodes co-vary in a way that is shared across subjects. Existing EEG generative models nonetheless leave the network to learn this from scratch. We put this structure into the source instead. From the sensor coordinates alone, we build a k-nearest-neighbor graph and take a graph-Mat\'ern function of its Laplacian as the source covariance, so the flow starts from spatially coherent patterns rather than channel-independent noise. The change adds no learned parameters, works with any coupling and any drift network, and uses the same three hyperparameters on every dataset. Across eight EEG datas
    
[^63]: 不确定性量化对于可靠的基于连接组的图学习不可或缺：叙述性综述与案例研究

    Uncertainty Quantification Is Indispensable for Reliable Connectome-Based Graph Learning: A Narrative Review and Case Study

    [https://arxiv.org/abs/2610.08353](https://arxiv.org/abs/2610.08353)

    本文通过叙述性综述与实证案例研究表明，确定性图神经网络在基于连接组的诊断分类中会产生过度自信的预测，因此不确定性量化对于可靠的连接组图学习不可或缺。

    

    尽管图神经网络（GNN）在基于连接组的诊断分类中展现出巨大潜力，但确定性模型不可避免地会抑制由处理流程引入的噪声和模型歧义，从而产生过度自信的预测。虽然不确定性量化（UQ）已在体素级分割中被广泛采用，但其在连接组图学习中的作用在很大程度上仍未得到研究。本文对适用于连接组图学习的UQ框架进行了全面的叙述性综述，并通过实证案例研究展示了未校准预测的危害。我们剖析了神经影像处理流程中偶然不确定性和认知不确定性的来源，并回顾了主要的UQ范式，涵盖从贝叶斯近似和集成方法到证据学习和共形预测等方法。在我们的案例研究中，在SUDMEX CONN数据集的动态功能连接（dFC）矩阵上训练的时间图注意力网络（GAT）……

    arXiv:2610.08353v1 Announce Type: new  Abstract: While graph neural networks (GNNs) have shown substantial promise in connectome-based diagnostic classification, deterministic models inevitably suppress pipeline-induced noise and model ambiguities, yielding overconfident predictions. Although uncertainty quantification (UQ) is widely adopted in voxel-level segmentation, its role in connectomic graph learning remains largely unaddressed. This paper presents a comprehensive narrative review of UQ frameworks tailored to connectome graph learning alongside an empirical case study demonstrating the perils of uncalibrated predictions. We delineate sources of aleatoric and epistemic uncertainty across neuroimaging pipelines and review prominent UQ paradigms, from Bayesian approximations and ensemble methods to evidential learning and conformal prediction. In our case study, a temporal Graph Attention Network (GAT) trained on dynamic functional connectivity (dFC) matrices from the SUDMEX CONN 
    
[^64]: 稀疏支持向量机的高维统计推断

    High-Dimensional Statistical Inference for Sparse Support Vector Machines

    [https://arxiv.org/abs/2610.08345](https://arxiv.org/abs/2610.08345)

    该论文通过将 $L_1$-惩罚支持向量机表示为线性规划并借助对偶变量识别铰链损失次梯度，突破了铰链损失非光滑性导致的去偏难题，首次在高维比例渐近机制下为稀疏SVM建立了计算可行的渐近高斯推断框架，实现了置信区间、假设检验和FDR受控的变量选择。

    

    利用复制对称的高维刻画方法，我们在样本量与特征数成比例增长的情形下，为稀疏支持向量机建立了一个统计推断框架。主要挑战在于铰链损失的非光滑性，这阻碍了为光滑分类损失发展的去偏方法被直接应用。我们通过将 $L_1$-惩罚的支持向量机（SVM）表示为一个线性规划，并经由其对偶变量识别铰链损失的次梯度，从而克服了这一困难。由此得到一个计算上可行的去偏估计量，在比例渐近机制下其各坐标渐近服从高斯分布。所得的分布刻画为单个特征提供了置信区间与假设检验，并支持错误发现率（FDR）受控的变量选择。大量模拟实验检验了校准性、功效以及变量选择的性能。

    arXiv:2610.08345v1 Announce Type: cross  Abstract: Using a replica-symmetric high-dimensional characterization, we develop an inferential framework for sparse support vector machines when the sample size and number of features grow proportionally. The main challenge is the nonsmooth hinge loss, which prevents direct application of debiasing arguments developed for smooth classification losses. We overcome this difficulty by representing the $L_1$-penalized support vector machine (SVM) as a linear program and identifying the hinge-loss subgradient through its dual variables. This yields a computationally accessible debiased estimator whose coordinates are asymptotically Gaussian under the proportional asymptotic regime. The resulting distributional characterization provides confidence intervals and hypothesis tests for individual features and enables false-discovery-rate-controlled variable selection. Extensive simulations examine calibration, power, and variable-selection performance u
    
[^65]: DIPrune：基于双重重要性的任务感知令牌剪枝，实现高效多模态语言模型

    DIPrune: Task-Aware Token Pruning with Dual Importance for Efficient Multimodal Language Models

    [https://arxiv.org/abs/2610.08341](https://arxiv.org/abs/2610.08341)

    本文提出DIPrune，一种基于双重重要性的任务感知令牌剪枝方法，通过将无训练剪枝重新表述为最小化任务损失失真问题，并揭示此前被忽视的考虑跨层梯度的层间项，解决了浅层显著令牌通过数值惯性压制深层语义信号所导致的语义退化问题，从而实现高效的多模态大语言模型。

    

    近期针对多模态大语言模型（MLLMs）的无训练剪枝方法，通过利用视觉冗余或文本-视觉注意力，有效地降低了计算开销。然而，由于其任务无关的设计或不可靠的注意力估计，这些方法经常出现语义退化的问题。基于我们的实证分析，我们发现这一问题的根源在于：浅层中的显著令牌通过数值惯性持续抑制新出现的语义令牌，导致对深度推理至关重要的信号被过早丢弃。为解决上述问题，我们从任务导向的角度出发，首先将无训练剪枝重新表述为最终任务损失失真的最小化问题，并推导出一个可处理的、逐令牌的上界作为替代目标。具体而言，这一表述天然地揭示了一个此前被忽视的层间项，该层间项考虑了跨层的梯度信息。

    arXiv:2610.08341v1 Announce Type: cross  Abstract: Recent training-free pruning approaches for Multimodal Large Language Models (MLLMs) effectively cut computational overhead by exploiting visual redundancy or text-vision attention. However, they frequently suffer from semantic degradation due to their task-agnostic design or unreliable attention estimates. Based on our empirical analysis, we have found that this issue arises because salient tokens in shallow layers persistently suppress emerging semantic ones through numerical inertia, leading to premature discarding of signals crucial for deep reasoning. To address the aforementioned issue, from the task-oriented aspects, we first reformulate training-free pruning as a minimization of the distortion in the final task loss and derive a tractable, token-wise upper bound to serve as a surrogate objective. Specifically, this formulation inherently reveals a previously neglected inter-layer term that accounts for gradients across layers. 
    
[^66]: 数据延迟与时间分布偏移下的德国再调度机器学习预测

    Machine Learning for German Redispatch Forecasting under Data Delays and Temporal Distribution Shift

    [https://arxiv.org/abs/2610.08337](https://arxiv.org/abs/2610.08337)

    该研究构建了一个在数据延迟与时间分布偏移等真实信息约束下评估德国电网再调度概率预测的基准，系统比较了从季节性经验方法到Transformer的多种模型及不同校准策略，为电网拥塞预测的可靠性与准确性提供了实证依据。

    

    公开的再调度记录为电网拥塞预测提供了实证数据，但报告延迟、零膨胀分布和时间分布偏移构成了重大的建模挑战。我们在实验性施加的信息时效约束下，评估了基于公开德国输电记录的概率机器学习预测的准确性与可靠性。该基准评估了2021至2024年间德国四家输电系统运营商的八条每日上调与下调干预电量序列（共48,242条合格记录；2024年含354个评估日期）。在最小七天目标延迟约束下，我们比较了季节性经验方法、正则化自回归（ARX）模型、分位数LightGBM、GRU和Transformer模型。神经架构采用零删失输出头来处理精确为零的结果。静态、滚动与自适应延迟反馈校准方法则通过归一化加权区间评分进行评估。

    arXiv:2610.08337v1 Announce Type: cross  Abstract: Public redispatch records provide empirical data for grid congestion forecasting, but delayed reporting, zero-inflated distributions, and temporal shift present major modeling challenges. We assess the accuracy and reliability of probabilistic machine-learning forecasts using published German transmission records under experimentally imposed information-age constraints. The benchmark evaluates eight daily series of upward and downward intervention energy across four German transmission system operators from 2021 to 2024 (48,242 eligible records; 354 evaluation dates in 2024). We compare seasonal empirical, regularized autoregressive (ARX), quantile LightGBM, GRU, and Transformer models under a minimum seven-day target-latency constraint. Neural architectures use a zero-censored output head to accommodate exact-zero outcomes. Static, rolling, and adaptive delayed-feedback calibration are evaluated using normalized weighted interval scor
    
[^67]: 结构感知的图弃权方法用于可靠的选择性预测

    Structure-Aware Graph Abstention for Reliable Selective Forecasting

    [https://arxiv.org/abs/2610.08322](https://arxiv.org/abs/2610.08322)

    本文提出一种结构感知的图弃权方法，通过学习稀疏图和狄利克雷风格的结构能量来评估多变量预测的关系一致性，作为与实例级合理性互补的弃权信号，在匹配覆盖率下通常比TEM降低选择性MSE。

    

    选择性预测在保留覆盖率的预算下，对高风险测试窗口进行弃权。现有的门控方法如TEM（Brusokas等人，2025）将每个预测作为一个整体进行评分；对于多变量输出，轨迹可能看起来合理，却违反了变量之间的依赖关系。我们将实例级合理性和关系一致性视为两个不同的可靠性维度，并通过学习到的稀疏图和狄利克雷风格的结构能量E_struct对后者进行操作化，采用误差加权图正则化和分数-误差对齐进行训练。在七个长时程基准和四个骨干网络上，在匹配的覆盖率下，结构门控相较于TEM通常能降低选择性MSE，且在我们的基准中，跨变量结构信息量更大的地方收益最大；收益并非普遍存在，这表明其是一种互补的弃权信号。表1是协议A的排名诊断（种子2024）；三种子可部署的协议B在al（摘要截断）

    arXiv:2610.08322v1 Announce Type: new  Abstract: Selective forecasting abstains on high-risk test windows under a retained-coverage budget. Existing gates such as TEM (Brusokas et al., 2025) score each forecast as a whole; for multivariate outputs, trajectories can look plausible while violating dependencies among variables. We treat instance-level plausibility and relational consistency as distinct reliability axes and operationalize the latter via a learned sparse graph and a Dirichlet-style structural energy E_struct, trained with error-weighted graph regularization and score-error alignment. On seven long-horizon benchmarks and four backbones, structural gating often reduces selective MSE versus TEM at matched coverage, with the largest gains where cross-variable structure appears more informative in our benchmarks; gains are not universal, indicating a complementary abstention signal. Table 1 is a Protocol A ranking diagnostic (seed 2024); three-seed deployable Protocol B on an al
    
[^68]: 标准化陷阱：认证表格基础模型中的联合标签处理

    The Standardization Trap: Certifying Joint Label Processing in Tabular Foundation Models

    [https://arxiv.org/abs/2610.08314](https://arxiv.org/abs/2610.08314)

    该论文揭示了检验表格基础模型是否遵循固定权重机制时存在的“标准化陷阱”问题，并提出两个仅依赖标准化标签处预测的证明方法，能够区分固定权重预测与非线性标签变换两种解释。

    

    线性回归和核平滑为上下文学习提供了可解析的解释：在这两种方法中，特征决定了分配给每个上下文标签的权重。然而，这种固定权重的解释是否适用于预训练的表格基础模型（TFMs）仍不清楚。使用导数来检验这一解释会遇到标准化陷阱：公开的TFM软件包在模型看到标签之前会先对标签进行标准化，而普通导数同时也会反映标准化标签集合之外的行为，导致模型即使在所有预测都与固定权重映射一致的情况下也会显得非线性。我们提出了两个仅依赖于标准化标签处预测的证明方法，能够拒绝两种不同的解释：固定权重预测和独立非线性标签变换之和。在我们评估的五个公开TFM中，我们的证明表明，改变一个上下文标签会改变其他标签影响预测的方式。

    arXiv:2610.08314v1 Announce Type: new  Abstract: Linear regression and kernel smoothing offer tractable explanations of in-context learning: in both, the features determine the weight assigned to each context label. However, whether this fixed-weight account describes pretrained tabular foundation models (TFMs) remains unclear. Testing this account using derivatives runs into a standardization trap: public TFM packages standardize the labels before the model sees them, yet ordinary derivatives also reflect behavior outside the set of standardized labels, making a model appear nonlinear even when every prediction it makes agrees with a fixed-weight map. We propose two certificates that depend only on predictions at standardized labels and can reject two distinct explanations: fixed-weight prediction and sums of independent nonlinear label transformations. Across the five public TFMs that we evaluate, our certificates show that changing one context label alters how other labels influence
    
[^69]: CoDe-LoRA：通过知识巩固与解耦缓解大语言模型持续学习中的正交困境

    CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling

    [https://arxiv.org/abs/2610.08312](https://arxiv.org/abs/2610.08312)

    提出无需回放的CoDe-LoRA方法，通过自适应零空间投影和语义路由将学习过程解耦为通用知识巩固与任务特定知识解耦，克服了正交参数隔离阻碍跨任务知识迁移的“正交困境”。

    

    持续学习（CL）对于大语言模型（LLMs）顺序适应不断演变的任务至关重要。为了缓解灾难性遗忘，近期的进展采用带正交投影的低秩适应方法（如O-LoRA）来隔离任务参数。然而，我们揭示了这种严格的几何约束会引发“正交困境”：僵化的参数隔离阻碍了语义相关任务之间共享表示的迁移与积累。在这项工作中，我们提出了一种新的无需回放的方法，称为巩固与解耦LoRA（CoDe-LoRA），用于大语言模型的持续学习。CoDe-LoRA将学习过程解耦为巩固通用知识和解耦任务特定知识两个部分。为实现这一目标，CoDe-LoRA利用自适应零空间投影机制和语义路由来平衡知识积累与任务特定适应。在四个骨干模型和三种持续学习设置上的实验结果……

    arXiv:2610.08312v1 Announce Type: new  Abstract: Continual learning (CL) is essential for Large Language Models (LLMs) to sequentially adapt to evolving tasks. To mitigate catastrophic forgetting, recent advances implement low-rank adaptation with orthogonal projections (e.g., O-LoRA) to isolate task parameters. However, we reveal that such strict geometric constraints trigger an "Orthogonality Dilemma": rigid parameter isolation impedes the transfer and accumulation of shared representations across semantically related tasks. In this work, we propose a new replay-free method, called Consolidation and Decoupling LoRA (CoDe-LoRA), for CL of LLMs. CoDe-LoRA disentangles the learning process into Consolidating Universal Knowledge and Decoupling Task-Specific Knowledge. To achieve this, CoDe-LoRA leverages an adaptive null space projection mechanism and semantic routing to balance knowledge accumulation with task-specific adaptation. Experimental results across four backbones and three CL 
    
[^70]: OxiGen：氧化态感知的晶体生成

    OxiGen: Oxidation-State-Aware Crystal Generation

    [https://arxiv.org/abs/2610.08296](https://arxiv.org/abs/2610.08296)

    提出了OxiGen，一种显式建模氧化态的晶体扩散生成模型，通过在有限状态自动机上进行精确推理的结构化输出层在构造上保证全局电荷中性，显著提升了生成晶体的氧化态保真度以及稳定、独特且新颖晶体的生成率。

    

    生成式模型通过实现逆向设计，有潜力加速无机材料的发现，但生成实验上可实现的晶体仍然具有挑战性。氧化态被广泛用于评估晶体成分的有效性并指导无机材料的发现。虽然现有的晶体生成模型可以生成具有电荷中性氧化态分配的材料，但它们难以再现已合成材料中观察到的氧化态分布。为了解决这一局限性，我们提出了OxiGen，一种氧化态感知的晶体扩散模型，它在生成过程中显式地表示氧化态。OxiGen通过采用一个在有限状态自动机上进行精确推理的结构化输出层，在构造上强制保证全局电荷中性。实验结果表明，OxiGen显著提高了氧化态保真度，并生成了最高比例的稳定、独特且新颖的晶体。

    arXiv:2610.08296v1 Announce Type: new  Abstract: Generative models have the potential to accelerate inorganic materials discovery by enabling inverse design, but generating experimentally realisable crystals remains challenging. Oxidation states are widely used to assess the compositional validity of crystals and guide inorganic materials discovery. While existing generative models for crystals can generate materials with charge-neutral oxidation-state assignments, they poorly reproduce the distributions of oxidation states observed in synthesised materials. To address this limitation, we propose OxiGen, an oxidation-state-aware crystal diffusion model that explicitly represents oxidation states during generation. OxiGen enforces global charge neutrality by construction using a structured output layer with exact inference over a finite-state automaton. Empirically, OxiGen substantially improves oxidation-state fidelity, generates the highest rate of stable, unique, and novel crystals a
    
[^71]: 两个持续性图总体的差异在哪里？固定预算下的校准局部推断

    Where Do Two Populations of Persistence Diagrams Differ? Calibrated Local Inference at a Fixed Budget

    [https://arxiv.org/abs/2610.08292](https://arxiv.org/abs/2610.08292)

    本文提出在固定样本量下对两个持续性图总体的局部均值差异进行同步推断的方法，利用高斯乘子自助法校准置信区间，在控制族错误率的同时定位出生-死亡平面上造成总体差异的具体区域。

    

    许多针对持续性图总体的双样本检验仅评估全局差异，而无法识别出生-死亡平面上促成这些差异的具体区域。我们研究了当可用图的数量固定时，对局部均值对比的同步推断问题。这些对比是若干中心和半径处的 $\ell_\infty$ 邻域内期望加权特征质量的差异。我们使用加性标点响应来估计这些对比。高斯乘子自助法用于校准同步置信区间，同时允许两组的协方差不相等。置信区间不包含零的邻域形成一张具有近似族错误率控制的地图，且选择原始区间的一个子集进行展示仍可保持其联合覆盖保证。在同步覆盖事件上，每个被报告的邻域都位于均值测度差异支撑集的两倍半径范围之内。一个几何结果给出了充分（原文在此截断）

    arXiv:2610.08292v1 Announce Type: cross  Abstract: Many two-sample tests for populations of persistence diagrams assess global differences without identifying the regions of the birth-death plane that contribute to them. We study simultaneous inference for local mean contrasts when the number of available diagrams is fixed. They are differences in expected weighted feature mass within $\ell_\infty$ neighborhoods at several centers and radii. We estimate these contrasts using additive landmark responses. A Gaussian multiplier bootstrap calibrates simultaneous confidence intervals while allowing unequal group covariances. The neighborhoods whose intervals exclude zero form a map with approximate family-wise error control, and selecting a subset of original intervals for display preserves their joint coverage guarantee. On the simultaneous coverage event, every reported neighborhood lies within twice its radius of the support of the mean-measure difference. A geometric result gives suffic
    
[^72]: 表格数据中多属性逻辑依赖与函数依赖的可扩展提取与可视化

    Scalable extraction and visualization of multi-attribute logical and functional dependencies in tabular data

    [https://arxiv.org/abs/2610.08287](https://arxiv.org/abs/2610.08287)

    提出了LDTool和HLDTool两个工具，实现了表格数据中多属性逻辑依赖与函数依赖的统一提取与可视化，并通过超图引导的搜索空间缩减解决了大规模数据下的可扩展性问题。

    

    理解表格数据中属性之间的结构关系是机器学习和模式识别的基础。虽然函数依赖（FD）的发现已被广泛研究，但逻辑依赖（LD）的可扩展发现——尤其是随着属性数量和依赖阶数增加时——仍未得到充分探索。这些依赖关系捕获了成对或多个属性之间的非确定性、条件特定的关系。此外，现有方法没有提供统一的框架来提取多属性逻辑依赖和函数依赖。为了解决这些局限性，我们提出了LDTool和HLDTool，用于从表格数据中提取和可视化多属性逻辑依赖和函数依赖。LDTool将依赖发现扩展到成对关系之外，而HLDTool通过超图引导的搜索空间缩减实现了可扩展的提取。在三个模拟数据集和十一个真实数据集上的实验证明了该方法的有效性。

    arXiv:2610.08287v1 Announce Type: new  Abstract: Understanding the structural relationships among attributes in tabular data is fundamental to machine learning and pattern recognition. While functional dependency (FD) discovery has been extensively studied, scalable discovery of logical dependencies (LDs), particularly as the number of attributes and dependency order increase, remains underexplored. These dependencies capture non-deterministic, condition-specific relationships among pairwise or multiple attributes. Furthermore, existing approaches do not provide a unified framework for extracting multi-attribute LDs and FDs. To address these limitations, we propose LDTool and HLDTool for extracting and visualizing multi-attribute LDs and FDs from tabular data. LDTool extends dependency discovery beyond pairwise relationships, while HLDTool enables scalable extraction through hypergraph-guided search-space reduction. Experiments on three simulated and eleven real-world datasets demonstr
    
[^73]: 基于生成过程的双样本检验

    Two-Sample Testing via Generative Processes

    [https://arxiv.org/abs/2610.08277](https://arxiv.org/abs/2610.08277)

    提出了一种基于随机插值与时间反射对称性的双样本检验方法，通过计算时间 t 和 1-t 处边缘分布的 Jensen-Shannon 散度来判断两个样本是否同分布，该方法无需学习任何参数、可精确控制有限样本检验水平，并能达到极小化极大分离速率。

    

    判断两个样本是否来自同一分布是统计学中的一个经典问题，而生成式传输为解决这一问题提供了新的途径。我们直接在两个样本之间构建随机插值，并观察到：在对称调度下，只要两个分布相同，该插值的分布在时间反射 t ↦ 1-t 下保持不变。因此，我们通过计算时间 t 和 1-t 处边缘分布之间的 Jensen-Shannon 散度来检验它们是否一致。两个边缘分布都是所有观测交叉对上的显式混合，因此无需学习任何参数，并且通过置换校准可以获得精确的有限样本检验水平。对于高斯噪声，该散度等于一个时间积分，该积分将速度场和得分函数的反射缺陷配对，因此该检验比较的是传输动力学过程而不仅仅是端点。通过窄加宽的噪声设计，该检验达到了极小化极大分离速率 n^{-2s/(4s+d)}。

    arXiv:2610.08277v1 Announce Type: cross  Abstract: Deciding whether two samples come from the same distribution is a classical problem in statistics, and generative transport offers a new way to approach it. We build a stochastic interpolant directly between the two samples and observe that, for a symmetric schedule, its law is invariant under the time reflection $t \mapsto 1-t$ whenever the two distributions coincide. We therefore test whether the marginals at times t and 1-t agree by computing their Jensen--Shannon divergence. Both marginals are explicit mixtures over all cross-pairs of observations, so nothing is learned, and permutation calibration gives an exact finite-sample level. For Gaussian noise, this divergence equals a time integral that pairs the reflection defects of the velocity field and of the score, so the test compares transport dynamics rather than endpoints alone. With a narrow-plus-broad noise design, the test attains the minimax separation rate n^{-2s/(4s+d)} ov
    
[^74]: 具有选择性标签的表演性预测

    Performative Prediction with Selective Labels

    [https://arxiv.org/abs/2610.08272](https://arxiv.org/abs/2610.08272)

    该论文首次形式化了具有选择性标签（即只能观察到被接受子群体的标签）情境下的表演性预测问题，并证明仅基于观测数据进行重训练会误导模型更新过程并破坏收敛性保证。

    

    机器学习的许多社会应用表现出表演性效应：群体行为会随着部署模型的上线而发生改变。表演性预测通过一个分布映射来研究这种交互作用，该映射将每个模型与其诱导的总体分布联系起来。该框架的一个主要结果表明，重复风险最小化（RRM）——即通过在最新数据上重新训练来更新模型——可以收敛到一个稳定的模型，该模型在其自身诱导的分布上最小化风险。然而，现有分析通常假设模型部署后能够获得特征和标签的完整分布，忽略了选择性标签的可能性：即只能观察到被接受子群体的标签。在这项工作中，我们形式化了具有选择性标签的表演性预测问题，并表明仅在观测数据上进行重训练可能会误导重训练过程，并破坏收敛性保证。

    arXiv:2610.08272v1 Announce Type: new  Abstract: Many social applications of machine learning exhibit performative effects: population behavior changes in response to deployed models. Performative prediction studies this interaction through a distribution map that relates each model to the population distribution it induces. One of the main results in this framework showed that repeated risk minimization (RRM), which updates models by retraining on the most recent data, can converge to a stable model that minimizes risk on its own induced distribution. However, existing analyses typically assume access to the complete distributions of features and labels after model deployment, ignoring the possibility of selective labels: observing labels only for the accepted subset of the population. In this work, we formalize performative prediction with selective labels and show that retraining only on observed data can misguide the retraining procedure and undermine the guarantees of convergence 
    
[^75]: 线性函数逼近下的分段奖励反馈强化学习

    Reinforcement Learning with Segment Reward Feedback under Linear Function Approximation

    [https://arxiv.org/abs/2610.08271](https://arxiv.org/abs/2610.08271)

    该论文研究了线性函数逼近下的分段奖励反馈强化学习，针对二值和求和两种反馈类型分别提出了计算高效的算法（BiTs-SEGD 和 EDLinUCB-SEGD），并建立了几乎匹配的下界，回答了分段反馈粒度与分割方式如何影响学习效果。

    

    经典强化学习（RL）假设每个访问的状态-动作对都能观测到奖励。然而，在自动驾驶等现实应用中，这种细粒度的反馈可能成本高昂或难以收集，而轨迹级别的反馈又可能过于稀疏，不利于高效学习。为了提供一个连接这两个极端的通用反馈模型并处理大规模状态空间，我们研究了线性函数逼近下的分段奖励反馈强化学习。我们的工作回答了分段反馈的粒度以及分割方式的选择如何影响学习。对于转移已知的等长分段，我们分别针对二值反馈类型和求和反馈类型设计了算法 BiTs-SEGD 和 EDLinUCB-SEGD。它们采用结合规划的后验采样以实现计算效率，并采用 E-最优实验设计以达到近最优性。我们建立了几乎匹配的下界。（注：原摘要在此处被截断）

    arXiv:2610.08271v1 Announce Type: new  Abstract: Classical reinforcement learning (RL) assumes that a reward is observed for every visited state-action pair. However, in real-world applications such as autonomous driving, such fine-grained feedback can be costly or difficult to collect, whereas trajectory-level feedback may be too sparse for efficient learning. To provide a general feedback model bridging these two extremes and handle large state spaces, we study RL with segment reward feedback under linear function approximation. Our work answers how the granularity of segment feedback and the choice of segmentation influence learning. For equal-length segments with known transitions, we design algorithms $\bitssegd$ and $\edlinucbsegd$ for binary and sum feedback types, respectively. They adopt posterior sampling with planning to achieve computational efficiency and the E-optimal experimental design to attain near-optimality. Nearly matching lower bounds are established. For equal-le
    
[^76]: DySCo：面向协同边云大语言模型推理的动态分片与深度同步批处理方法

    DySCo: Dynamic Sharding for Collaborative Edge-Cloud LLM Inference with Depth-Synchronized Batching

    [https://arxiv.org/abs/2610.08268](https://arxiv.org/abs/2610.08268)

    提出DySCo协同运行时系统，通过动态分片、模型感知的层范围执行器dyForward以及深度同步批处理，消除边云协同LLM推理中云端调用的空闲间隙，并解决到达不同模型深度的请求无法常规批处理的难题。

    

    无处不在的智能应用正越来越多地部署在移动和物联网边缘设备上，因此大语言模型（LLM）被越来越多地用于支持这些应用。然而，由于LLM的高资源需求，它们大多部署在云端。逐层边云协同推理使资源受限的边缘设备能够为自身无法完整承载的LLM贡献算力。然而，异构的切分点引入了两个相互耦合的低效问题：其一，边缘侧的执行和通信会在云端调用之间产生空闲间隙；其二，到达模型不同深度的请求无法进行常规批处理。我们提出了DySCo，这是一个协同运行时系统，它将KV缓存保留在本地，并引入了dyForward——一个模型感知的层范围执行器，可以从常驻的模型分片中运行可配置的连续层范围，而无需重新加载权重。针对多边缘服务场景，我们进一步引入了深度同步（注：原文摘要在此处截断）

    arXiv:2610.08268v1 Announce Type: cross  Abstract: Pervasive intelligent applications are increasingly deployed on mobile and Internet of Things (IoT) edge devices. Consequently, Large Language Models (LLMs) are increasingly used to support these applications. Yet, due to their high resource demands, LLMs are mostly deployed in the cloud. Layer-wise edge-cloud inference lets resource-constrained edge devices contribute computation to LLMs they cannot host in full. However, heterogeneous split points introduce two coupled inefficiencies. First, edge execution and communication create idle gaps between cloud invocations. Second, requests arriving at different model depths cannot be conventionally batched. We present DySCo, a collaborative runtime that keeps KV caches local and introduces dyForward, a model-aware layer-range executor that runs configurable contiguous layer ranges from resident model shards without reloading weights. For multi-edge serving settings, we introduce depth-sync
    
[^77]: LeanPlan：基于LLM生成启发式函数与可采纳性证明的最优规划

    LeanPlan: Optimal Planning with LLM-Generated Heuristics and Admissibility Proofs

    [https://arxiv.org/abs/2610.08246](https://arxiv.org/abs/2610.08246)

    LeanPlan是首个利用LLM生成的启发式函数（其可采纳性在Lean 4中经机器验证）以找到最优计划的规划系统，在国际规划竞赛领域上展现出优异的最优规划性能。

    

    前沿大型语言模型（LLM）能够生成启发式函数来引导搜索，在满意规划（satisficing planning，即任何可行计划均可接受）中实现最先进的性能。然而，这些启发式函数无法保证可采纳性，可能导致生成的计划并非最优。我们提出了LeanPlan，这是首个利用LLM生成的启发式函数找到最优计划、且其可采纳性经过机器验证的规划系统。给定领域描述和训练任务，一个智能体循环利用规划器的反馈来迭代改进可复用的领域特定启发式函数、其可采纳性证明以及所需的领域假设。LeanPlan在Lean 4中实现了该启发式函数、其证明以及一个具备机器验证的落地与搜索的高效规划器。我们在国际规划竞赛的十个领域和三个新领域上对LeanPlan进行评估，所用测试任务的对象数量多达训练任务的57倍。使用GPT-5.6 Sol i（摘要在此处截断）

    arXiv:2610.08246v1 Announce Type: new  Abstract: Frontier large language models (LLMs) can generate heuristic functions that guide search to achieve state-of-the-art performance in satisficing planning, where any plan is acceptable. However, these heuristics are not guaranteed to be admissible and can lead to suboptimal plans. We introduce LeanPlan, the first planning system that finds optimal plans with LLM-generated heuristics whose admissibility is machine-checked. Given a domain description and training tasks, an agentic loop uses planner feedback to iteratively improve a reusable domain-specific heuristic, its admissibility proof and the required domain assumptions. LeanPlan implements the heuristic, its proof and an efficient planner with machine-checked grounding and search in Lean 4. We evaluate LeanPlan on ten domains from the International Planning Competition and three new domains, using test tasks with up to 57 times as many objects as the training tasks. With GPT-5.6 Sol i
    
[^78]: 一张卫星图像包含多少独立样本？空间相关数据的泛化界

    How Many Independent Samples Does a Satellite Image Contain? Generalization Bounds for Spatially Dependent Data

    [https://arxiv.org/abs/2610.08227](https://arxiv.org/abs/2610.08227)

    该论文证明了空间相关性持续 r 个像素的 n×n 卫星图像，其有效样本量仅为 Θ(n²/r²) 而非 n²，并通过匹配的上下界证明该速率是紧的且不可超越，从而为空间交叉验证提供了最优泛化保证的理论依据。

    

    arXiv:2610.08227v1 通告类型：交叉 摘要：用于遥感影像的机器学习分类器通常在评估时将每个像素视为独立样本。空间自相关违反了这一假设，因为相邻像素携带冗余信息，从而夸大了样本量。一张卫星图像实际上包含多少独立样本？对于一个空间相关性在 r 个像素范围内持续存在的 n×n 图像，其有效样本量为 Θ(n²/r²)，而非 n²。我们将其证明为空间相关数据上分类器的有限样本上界，并通过匹配的下界证明该速率是紧的，即任何算法都无法超越这一速率。我们将该结果扩展到具有方向性相关和空间变化相关结构的图像。我们的结果为空间交叉验证提供了理论依据，因为与相关范围成比例的分块留出方法能够达到最优的泛化保证，而随机留出（摘要在此处截断）……

    arXiv:2610.08227v1 Announce Type: cross  Abstract: Machine learning classifiers for remote sensing imagery are typically evaluated as though every pixel were an independent sample. Spatial autocorrelation violates this assumption, since neighboring pixels carry redundant information which inflates sample sizes. How many independent samples does a satellite image actually contain? For an $n \times n$ image whose spatial correlation persists over a range of $r$ pixels, the effective sample size is $\Theta(n^2/r^2)$, not $n^2$. We prove this as a finite-sample upper bound for classifiers on spatially correlated data, and show via a matching lower bound that the rate is tight, and no algorithm can do better. We extend the results to images with directional correlation and spatially varying correlation structure. Our result justifies spatial cross-validation since block holdout with separation proportional to the correlation range achieves optimal generalization guarantees, while random hol
    
[^79]: 任意时刻有效的基于模拟的假设检验

    Anytime-valid simulation-based hypothesis testing

    [https://arxiv.org/abs/2610.08210](https://arxiv.org/abs/2610.08210)

    本文针对只能获得模拟样本而无解析密度的假设检验问题，构造了 e 检验鞅，实现了任意时刻有效的一类错误控制、几何衰减的二类错误界和渐近满功效的序贯检验。

    

    对于给定的独立同分布数据 $(X_t)_{t \in \mathbb{N}} \sim Q$，我们研究如下假设检验问题：$H_0: Q = P_0$ 对 $H_1: Q = P_1$，其中 $P_0$ 和 $P_1$ 是两个不同的模型概率分布。与标准设定中给定解析密度函数 $p_0$ 和 $p_1$ 不同，本文考虑的是无密度设定，即我们只能获得独立同分布的模拟样本 $(Z^0_t)_{t \in \mathbb{N}} \sim P_0$ 和 $(Z^1_t)_{t \in \mathbb{N}} \sim P_1$。针对这种基于模拟的假设检验设定，我们构造了一个 e 检验鞅，由此得到的序贯检验具有任意时刻有效的一类错误保证、近似增长最优性、几何衰减的二类错误界以及渐近功效为 1。我们构造中使用的大多数要素都是众所周知概念的变体。本文的价值在于以紧凑的方式呈现了一种有效的、任意时刻有效的无密度模拟假设检验解决方案。

    arXiv:2610.08210v1 Announce Type: cross  Abstract: For a given data distribution $(X_t)_{t \in \mathbb{N}} \sim Q$ i.i.d., we investigate the hypothesis testing problem: $H_0: Q = P_0$ vs. $H_1: Q = P_1$, for two different model probability distributions $P_0$ and $P_1$. In contrast to the standard setting, where analytic densities $p_0$ and $p_1$ are given, here, we consider the density-free setting, where we only have access to i.i.d. simulations $(Z^0_t)_{t \in \mathbb{N}} \sim P_0$ and $(Z^1_t)_{t \in \mathbb{N}} \sim P_1$. For this simulation-based hypothesis testing setting, we construct an e-test martingale, resulting in a sequential test with anytime-valid type-I error guarantees, approximate growth optimality, geometrically decaying type-II error bounds, and asymptotic power one. Most ingredients used in our constructions are variants of well known concepts. The value of this paper lies in the compact presentation of an effective, anytime-valid solution for the density-free si
    
[^80]: 找到语言模型中负责网络信息检索的注意力头与神经元

    Finding the Heads and the Neurons Responsible for Network Information Retrieval in Language Models

    [https://arxiv.org/abs/2610.08200](https://arxiv.org/abs/2610.08200)

    该研究发现，语言模型中经因果消融验证的极少数注意力头能以99.5%至100%的准确率检测上下文中的主机名与IP地址配对信息，且在某些模型中这一功能可进一步精确定位到单个神经元。

    

    我们探究是否特定的注意力头，以及更精细层面上这些头内部的特定神经元，负责识别语言模型上下文中包含的网络基础设施信息（主机名与其IP地址的配对），以及这种职责能否通过因果方法而非仅靠相关性来验证。在头的层面上，答案是肯定的，且该结论在横跨三个架构系列的五个模型中均成立：在每个模型中，通过因果消融筛选发现的一小组注意力头（128至1152个候选头中的1至9个），经过与匹配的负样本及无上下文对照的选择性测试，可支持一个在留出集上达到99.5%至100%准确率的检测器。我们随后探究一个注意力头的职责是集中于单个神经元还是分散在其各个维度上；答案因模型而异。在其中一个模型中，最强注意力头的信号集中于单个神经元，该结果由因果干预和相关性分析两种方法独立发现。

    arXiv:2610.08200v1 Announce Type: new  Abstract: We ask whether specific attention heads, and more finely specific neurons inside those heads, are responsible for recognizing that a language model's context contains network infrastructure information (a hostname paired with its IP address), and whether that responsibility can be validated causally rather than by correlation alone. At the head level the answer is yes, across five models spanning three architecture families: in every model, a small set of heads (1 to 9 out of 128 to 1152 candidates), found by causal ablation screening and tested for selectivity against matched negative and context-free controls, supports a detector with 99.5--100\% held-out accuracy. We then ask whether a head's responsibility concentrates into one neuron or stays spread across its dimensions; this is model-specific. In one model, the top head's signal concentrates into a single neuron, found independently by both a causal intervention and a correlationa
    
[^81]: 紧凑的机器人策略需要细粒度的视觉表示

    Compact Robot Policies Need Fine-Grained Visual Representations

    [https://arxiv.org/abs/2610.08183](https://arxiv.org/abs/2610.08183)

    论文提出紧凑策略CoRP，证明机器人多任务操作的性能主要由细粒度的预训练视觉表示决定，而非参数规模或生成式先验，其4890万参数的小模型即可媲美比它大上百倍的系统。

    

    多任务操作策略在架构、规模和预训练先验上同时存在差异，因此已发表的性能比较无法将效果归因于任何单一组件。我们认为，性能差异主要来自视觉表示，而参数规模和生成式先验在很大程度上是次要因素。为了验证这一点，我们构建了CoRP（压缩表示策略，Compressed Representation Policy），这是一个刻意保持紧凑的策略（4890万参数，不使用视觉-语言模型，也不使用视频生成先验），它分解为表示提取器和流匹配动作生成器两个部分。它在LIBERO上达到97.0%，在RoboTwin 2.0的Clean/Randomized设置下达到75.78%/73.36%，与比它大40.9至163.6倍的系统相当。在保持动作生成器固定不变的情况下，我们每次只改变提取器的一个属性。预训练初始化是决定性的：随机初始化的ViT-S/14在LIBERO上的性能降至78.1%，ImageNet预训练的ResNet-34降至74.5%。但仅有预训练是不够的，因为冻结编码器会带来19……（摘要原文在此处截断）

    arXiv:2610.08183v1 Announce Type: cross  Abstract: Multi-task manipulation policies differ in architecture, scale, and pretrained priors all at once, so published comparisons cannot attribute performance to any single component. We argue that most of it comes from the visual representation, and that parameter scale and generative priors are largely incidental. To test this, we build CoRP (Compressed Representation Policy), a deliberately compact policy (48.9M parameters, no vision-language model and no video-generative prior) that factorizes into a representation extractor and a flow-matching action generator. It reaches 97.0% on LIBERO and 75.78%/73.36% on RoboTwin 2.0 Clean/Randomized, matching systems 40.9-163.6x larger. Holding the action generator fixed, we then vary one extractor property at a time. Pretrained initialization is decisive: a random ViT-S/14 drops to 78.1% and an ImageNet ResNet-34 to 74.5% on LIBERO. Pretraining alone is not enough, as freezing the encoder costs 19
    
[^82]: 论基于潜空间的水印方法的内在鲁棒性局限

    On the Intrinsic Limited Robustness of Latent-Based Watermarking

    [https://arxiv.org/abs/2610.08178](https://arxiv.org/abs/2610.08178)

    本文首次从理论上揭示了基于潜空间的水印方法对旋转、缩放、平移等几何变换缺乏鲁棒性是其固有局限，并推导出刻画像素空间扰动与潜空间影响关系的最大扰动界。

    

    现有针对扩散模型的基于潜空间（latent-based）的水印方法高估了其对图像失真的鲁棒性，包括旋转、缩放和平移（RST）等几何变换。此外，这类水印范式可能因水印嵌入所在的域而存在固有局限性。本文首次提供了理论分析，解释了这些方法为何缺乏对扰动的空间不变性。通过对不变关系进行松弛，我们推导出一个最大扰动界，刻画了像素空间扰动与其在潜空间中所产生影响之间的关系。此外，我们提出了首个能够刻画实际检测机制所有组件的解析表述。最后，我们通过实验验证了上述理论发现以及基于潜空间水印方法的局限性。我们的理论与实证结果……

    arXiv:2610.08178v1 Announce Type: new  Abstract: Existing latent-based watermarking methods for diffusion models have overestimated their robustness to image distortions, including geometric transformations such as rotation, scaling, and translation (RST). Moreover, this paradigm of watermarking approaches may suffer from inherent limitations arising from the domain in which the watermark is embedded. In this paper, we provide the first theoretical analysis explaining why these methods lack invariance to perturbations. By relaxing the invariant relation, we derive a maximum perturbation bound that characterizes the relationship between pixel-space perturbations and their corresponding effects in latent space. In addition, we present the first analytical formulation that captures all components of practical detection mechanisms. Finally, we conduct experiments to validate the theoretical findings and the limitations of latent-based watermarking methods. Our theoretical and empirical res
    
[^83]: LFHE：面向非独立同分布数据下去中心化学习中受限局部拓扑搜索的局部优先启发式演化

    LFHE: Local-First Heuristic Evolution for Bounded Local Topology Search in Decentralized Learning with Non-IID Data

    [https://arxiv.org/abs/2610.08176](https://arxiv.org/abs/2610.08176)

    LFHE 提出了一种仅利用自身邻域和朋友的朋友信息进行受限局部拓扑重连的框架，其结构分数与图狄利克雷能量精确对应，从而在非独立同分布数据下有效加速去中心化学习中表示分歧的消散。

    

    在非独立同分布（non-IID）数据下，去中心化学习对通信拓扑高度敏感。自适应邻居选择方法可以利用本地模型信息，但更广泛的节点发现可能需要不断增大的控制状态，而直接的谱优化方法通常依赖全图信息。我们研究了受限局部拓扑搜索这一中间设定，提出了局部优先启发式演化（LFHE），这是一个由表示驱动的重连框架，其候选节点发现与评分仅使用自身邻域和朋友的朋友（FoF）信息。该结构分数可通过图狄利克雷能量得到精确解释：其在各客户端上的总和等于表示狄利克雷能量的两倍，而在标准线性共识动力学下，该能量控制着表示分歧的瞬时消散。LFHE 将这种依赖状态的结构信号与早期探索和度控制相结合……

    arXiv:2610.08176v1 Announce Type: new  Abstract: Decentralized learning is highly sensitive to communication topology under non-IID data. Adaptive peer-selection methods can exploit local model information, but broader peer discovery may require increasingly large control state, whereas direct spectral optimization typically relies on graph-wide information. We study the intermediate setting of bounded local topology search and propose Local-First Heuristic Evolution (LFHE), a representation-driven rewiring framework whose candidate discovery and scoring use only ego-neighborhood and friend-of-a-friend (FoF) information. The structural score admits an exact interpretation through graph Dirichlet energy: its sum across clients equals twice the representation Dirichlet energy, which under standard linear consensus dynamics governs the instantaneous dissipation of representation disagreement. LFHE combines this state-dependent structural signal with early exploration and degree control, w
    
[^84]: 超越排行榜：面向自动程序修复的稠密模型与混合专家模型的多维度评估

    Beyond the Leaderboard: Multi-Dimensional Evaluation of Dense and Mixture-of-Experts Models for Automated Program Repair

    [https://arxiv.org/abs/2610.08173](https://arxiv.org/abs/2610.08173)

    该论文受ISO/IEC 25010启发提出加权质量指数（QI），对稠密与混合专家代码模型进行涵盖正确性、可维护性、安全性和效率的多维度评估，发现模型排名随权重方案变化，单一指标评估会掩盖关键权衡。

    

    使用语言模型进行自动程序修复（APR）通常仅通过生成的补丁能否通过测试套件来评估，这可能掩盖模型在可维护性、安全性和计算成本方面的差异。我们受ISO/IEC 25010软件质量模型的启发，提出了一个加权质量指数（QI），该指数在可配置的权重方案下综合考量功能正确性、可维护性、安全性和生成效率。我们在40个QuixBugs和90个Defects4J缺陷上评估了三个稠密Qwen2.5-Coder模型（3B、7B、14B）以及160亿参数的DeepSeek-Coder-V2-Lite混合专家（MoE）模型（24亿激活参数），所有实验均在相同硬件上本地运行，以控制基础设施因素的影响。结果显示模型排名随权重方案而变化，表明单一指标的评估可能掩盖模型间的权衡取舍。MoE模型在正确性方面与7B和14B稠密模型几乎没有统计学上的显著差异（McNemar精确检验）。

    arXiv:2610.08173v1 Announce Type: cross  Abstract: Automated Program Repair (APR) with language models is usually evaluated by whether a generated patch passes the test suite, which can hide differences in maintainability, security, and computational cost. We propose a Weighted Quality Index (QI), inspired by the ISO/IEC 25010 software quality model, that combines functional correctness, maintainability, security, and generation efficiency under configurable weighting schemes. We evaluate three dense Qwen2.5-Coder models (3B, 7B, 14B) and the 16B-parameter DeepSeek-Coder-V2-Lite Mixture-of-Experts (MoE) model (2.4B active parameters) on 40 QuixBugs and 90 Defects4J bugs, all run locally on identical hardware to control for infrastructure effects. Model rankings change with the weighting scheme, showing that single-metric evaluation can hide trade-offs. The MoE model shows almost no statistically significant difference in correctness from the 7B and 14B dense models (McNemar's exact tes
    
[^85]: 先对齐，再校正：面向极端量化大语言模型的无训练两阶段低秩补偿

    Align, Then Correct: Training-Free Two-Stage Low-Rank Compensation for Extremely Quantized Large Language Models

    [https://arxiv.org/abs/2610.08164](https://arxiv.org/abs/2610.08164)

    该论文提出一种无需训练的两阶段闭式低秩补偿框架，先对齐层输出再校正残余误差，克服了现有低秩量化误差补偿在对称校准和仅二阶优化上的两大局限，大幅提升极端量化大语言模型的精度恢复能力。

    

    低秩量化误差补偿（LQEC）通过在每个冻结的量化权重旁附加一个闭式解的秩-r适配器，无需任何训练即可恢复极端权重量化下损失的精度。我们证明，现有的补偿器受限于两个共同的简化假设：其一，它们采用对称校准，即在相同的激活上评估全精度权重与补偿后的权重，这使补偿目标本身呈高秩特性，导致固定的秩预算只能捕获其中很小一部分；其二，它们仅最小化损失的二阶项，而补偿后的模型并非处于平稳点——每一层中仍残留一个比所施加补偿本身更大的一阶下降方向，且任何重构目标都无法将其吸收。我们提出了一个消除上述两个简化的两阶段闭式框架：第一阶段在Fisher加权（下）将每层的输出与全精度模型对齐……

    arXiv:2610.08164v1 Announce Type: cross  Abstract: Low-rank quantization error compensation (LQEC) recovers the accuracy lost under aggressive weight quantization by attaching a closed-form rank-$r$ adapter beside each frozen quantized weight, without any training. We show that existing compensators are limited by two shared simplifications. They calibrate symmetrically, evaluating the full-precision and compensated weights on the same activation, which yields a compensation target that is inherently high-rank -- so a fixed rank budget captures only a small fraction of it. And they minimize only the second-order term of the loss, although the compensated model is not stationary: a first-order descent direction larger than the applied compensation itself remains in every layer, and no reconstruction objective can absorb it. We propose a two-stage closed-form framework that removes both simplifications. Stage 1 aligns each layer's output with the full-precision model under a Fisher-weigh
    
[^86]: 文本生成交响曲：临床病历生成基准测试

    Symphony for Text Generation: Benchmarking Clinical Note Generation

    [https://arxiv.org/abs/2610.08161](https://arxiv.org/abs/2610.08161)

    该论文提出了包含300例多语言临床就诊记录的MedConv数据集，并构建了结合蕴含指标与大语言模型评判的受控临床评估框架，证明临床AI平台Corti的病历生成质量与领先商业环境式记录软件相当或更优，且其可配置API可针对特定文档需求灵活优化质量维度。

    

    环境式文档系统正迅速获得广泛应用，但其对临床病历质量的影响仍缺乏充分表征。我们推出了MedConv——一个包含300例临床就诊记录的多语言数据集，涵盖英语、丹麦语和德语，并将其与环境临床智能基准（ACI-BENCH）结合使用，将Corti（一个临床AI平台）与两个基于通用AI构建的领先且易用的环境式记录软件进行比较。我们提出了一个受控临床评估框架，该框架将文本蕴含指标与大语言模型评判的成对比较相结合，涵盖从PDSQI-9采纳的八个维度。结果表明，Corti基于API的文本生成基础设施与领先的商业记录软件相当或更优。我们进一步表明，Corti的可配置API提供了必要的灵活性，能够针对特定的文档用例对质量维度进行微调。我们呈现了该评估方法并发布了数据集。

    arXiv:2610.08161v1 Announce Type: cross  Abstract: Ambient documentation systems are rapidly gaining adoption, yet their impact on clinical note quality remains poorly characterized. We introduce MedConv, a multilingual dataset of 300 clinical encounters in English, Danish, and German, and use it alongside the Ambient Clinical Intelligence benchmark (ACI-BENCH) to compare Corti, a clinical AI platform, with two leading, accessible ambient scribe software applications built on general-purpose AI. We present a controlled clinical evaluation framework that combines entailment metrics with LLM-judged pairwise comparisons across eight dimensions adopted from PDSQI-9. Results show that Corti's API-based text-generation infrastructure is on par with or outperforms leading commercial scribes. We further show that Corti's configurable API provides the flexibility necessary to fine-tune quality dimensions for specific documentation use cases. We present the evaluation methodology and release a d
    
[^87]: 让 COMET 跨文字系统可比：诊断与修正印度语系机器翻译评估中由分词器引发的文字偏差

    Making COMET Comparable Across Scripts: Diagnosis and Correction of Tokeniser-Induced Script Bias in Indic MT Evaluation

    [https://arxiv.org/abs/2610.08159](https://arxiv.org/abs/2610.08159)

    该论文发现 COMET 评估指标因分词器存在文字偏差，导致印度语系不同文字系统间的分数不可比且排序准确性下降，并提出 COMET-QN 方法以精确消除跨文字分数范围不兼容的问题。

    

    COMET 将翻译质量报告为单一数值，而这一数值通常会在以不同文字系统书写的目标语言之间进行比较。这种比较隐含着“文字不变性”假设：分数不应依赖于承载目标文本的书写系统。我们在 IndicMT Eval 数据集上对该假设进行了检验，方法是将目标文本重新编码为拉丁字母，在保持内容和人工评分不变的情况下改变其正字法形式。结果显示，文字身份（script identity）解释了本族文字 COMET 方差的 22.9%，并且在所研究的全部五种语言中，指标与标注者的一致性均有所下降。我们将这一效应追溯到分词器，并用三种无需标签的诊断方法对其进行测量。这种偏差实为两个缺陷，而非一个：来自不同文字系统的分数占据互不兼容的数值范围，且在同一文字系统内部，该指标对翻译进行排序的准确性也更低。任何保序的分数变换都无法修复第二个缺陷。第一个缺陷则可由 COMET-QN 精确消除，该方法对分数分布进行映射……

    arXiv:2610.08159v1 Announce Type: new  Abstract: COMET reports translation quality as a single number, and that number is routinely compared across target languages written in different scripts. Such a comparison assumes Script Invariance: the score should not depend on the writing system that carries the target. We test it on IndicMT Eval by re-encoding the target into Latin script, which changes orthographic form while holding content and human ratings fixed. Script identity then accounts for 22.9% of native-script COMET variance, and agreement with annotators falls in all five languages studied. We trace the effect to the tokeniser and measure it with three label-free diagnostics. The bias is two faults, not one. Scores from different scripts occupy incompatible ranges, and within a single script the metric orders translations less accurately. No order-preserving transform of the score can repair the second fault. The first is removed exactly by COMET-QN, which maps the score distri
    
[^88]: 超越边际监测：面向大规模电商运营中数据概念漂移的分布式联合分布检验

    Beyond Marginal Monitoring: Distributed Joint-Distribution Testing for Data Concept Drift in Large Scale E-Commerce Operations

    [https://arxiv.org/abs/2610.08132](https://arxiv.org/abs/2610.08132)

    该论文在亿级规模的电商真实数据上系统评估了五种多变量双样本漂移检测方法，证明基于 Apache Spark 的分布式最大均值差异（MMD）结合随机傅里叶特征的方法在检测概念漂移时具备稳健的可扩展性。

    

    概念漂移威胁着生产环境的机器学习系统，然而多变量双样本漂移检测器在大规模场景下的实证表现仍缺乏充分的研究刻画。现有基准测试很少涉及工业运营数据集中典型的数亿行数据规模和高基数特征。我们在三个互补环境中评估了五种多列双样本检验方法（边际方法、基于投影的方法和核嵌入方法）：哈佛 Dataverse 数据集、经过验证的 Failing Loudly 复现实验（平均绝对误差介于 0.030 至 0.053 之间），以及基于 1.375 亿行 Trendyol 集合排序特征表构建的新型合成注入基准。通过在两种严重性-范围机制下对四种漂移类型进行测试，我们证明了基于 Apache Spark、结合随机傅里叶特征的分布式最大均值差异（MMD）方法能够稳健地扩展。在强机制下对四种漂移类型取平均并在校准阈值条件下，该方法达到了……

    arXiv:2610.08132v1 Announce Type: cross  Abstract: Concept drift threatens production machine learning, yet the empirical behavior of multivariate two-sample drift detectors at scale remains under-characterized. Existing benchmarks rarely address the hundreds of millions of rows and high-cardinality features typical of industrial-operational datasets. We evaluate five multi-column two-sample tests (marginal, projection-based, and kernel embedding methods) across three complementary environments: the Harvard Dataverse, a validated Failing Loudly reproduction (mean absolute error between 0.030 and 0.053), and a novel synthetic-injection benchmark on the 137.5-million-row Trendyol collection-ranking feature table. Testing four drift types across two severity-scope regimes, we demonstrate that distributed Maximum Mean Discrepancy with Random Fourier Features on Apache Spark scales robustly. Averaged over the four drift types in the strong regime and under a calibrated threshold, it achieve
    
[^89]: Mu-DisCoCat：量子处理器上实现组合泛化的变分流水线

    Mu-DisCoCat: A Variational Pipeline for Compositional Generalization on Quantum Processors

    [https://arxiv.org/abs/2610.08131](https://arxiv.org/abs/2610.08131)

    本文提出Mu-DisCoCat，一个多模态变分量子学习框架，通过先学习单物体图文表示、再固定表示学习多物体关系的两阶段训练，在量子处理器上实现了组合概念泛化。

    

    实现组合概念泛化（CoCoGen），即通过重新组合已学习的基本元素来理解新情境的能力，仍然是人工智能领域的一项根本性挑战。诸如组合分布语义（DisCoCat）等组合语义模型通过将向量推广为张量来提供解决方案，但在学习张量时存在扩展瓶颈。将DisCoCat映射到变分量子电路（VQC）上解决了文本处理中的这一局限性，然而该方法尚未扩展到组合概念泛化所涉及的多模态情境。本文提出了Mu-DisCoCat：一个面向DisCoCat的多模态变分量子学习框架，能够实现组合概念泛化。该框架首先从单物体图文对中学习稳定的物体表示，然后固定这些表示，并利用它们在多物体情境中学习物体之间的关系。在经典模拟中，该模型使用了U（摘要在此处截断）

    arXiv:2610.08131v1 Announce Type: new  Abstract: Achieving compositional concept generalization (CoCoGen), the ability to understand novel situations by recombining learned primitives, remains a fundamental challenge in artificial intelligence. Compositional semantic models such as Compositional Distributional Semantics (DisCoCat) offer solutions by generalising vectors to tensors, but suffer from scaling bottlenecks when learning the tensors. Mapping DisCoCat onto Variational Quantum Circuits (VQCs) resolves this limitation for text, yet the methodology has not been expanded to multimodal situations such as the ones involved in CoCoGen. This paper introduces Mu-DisCoCat: a multimodal variational quantum learning framework for DisCoCat that achieves CoCoGen. The framework first learns stable object representations from single-object image-text pairs, then fixes these and uses them to learn the relations between them in multi-object situations. In classical simulations, the model used U
    
[^90]: 大语言模型能将所知付诸行动吗？从伙伴表征到合作行为

    Do LLMs Act on What They Know? From Partner Representations to Cooperative Actions

    [https://arxiv.org/abs/2610.08129](https://arxiv.org/abs/2610.08129)

    该论文发现在Hanabi类合作任务中，八个大语言模型虽能通过线性探针较准确地恢复发送方的意图惯例，但其决策并未一致遵循该惯例，且将惯例转化为具体动作推荐比以规则形式陈述更能有效提升合作表现，揭示了模型“知道”与“行动”之间的脱节。

    

    与陌生伙伴合作需要适应事先未知的交流惯例。我们在一个受控的、源自Hanabi（花火）游戏的环境中研究这一问题，该环境采用脚本化的提示生成、由大语言模型控制的接收决策以及冻结的模型权重。在八个大语言模型中，线性探针恢复意图惯例的准确度显著高于恢复目标惯例的准确度，但接收方的选择并未始终与发送方的惯例保持一致。我们比较了以一般规则形式或以外部计算的动作推荐形式呈现的探针预测惯例与真实惯例。规则陈述仅带来温和且依赖具体模型的合作变化，而动作转译平均能带来更大的收益。在Qwen3-8B的案例研究中，匹配状态下的陈述反转显示，模型对动作推荐的敏感度远高于对规则陈述的敏感度。从预言动作与非预言提示重述进行的激活迁移（摘要原文在此处截断）……

    arXiv:2610.08129v1 Announce Type: new  Abstract: Cooperation with unfamiliar partners requires adapting to communication conventions that are not known in advance. We study this problem in a controlled Hanabi-derived environment with scripted hint generation, LLM-controlled receiving decisions, and frozen model weights. Across eight LLMs, linear probes recover intent conventions substantially more accurately than target conventions, yet receiving choices do not consistently agree with the sender's convention. We compare probe-predicted and ground-truth conventions presented either as general rules or as externally computed action recommendations. Rule statements yield modest and model-dependent changes in cooperation, whereas action translation produces larger gains on average. In a Qwen3-8B case study, matched-state statement reversals reveal much greater sensitivity to action recommendations than to rule statements. Activation transfers from oracle-action and non-oracle hint-restatem
    
[^91]: 超越路径点回归：面向端到端驾驶的基于查询的自车可达未来代价学习

    Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving

    [https://arxiv.org/abs/2610.08123](https://arxiv.org/abs/2610.08123)

    本文提出一种基于查询的代价学习框架，通过对自车动态可达的未来轨迹估计有界代价来替代传统路径点回归，将代价拓扑转化为可行规划，在nuScenes与真实驾驶数据上显著降低碰撞率并保持可解释性。

    

    基于路径点回归的端到端规划器在开环精度方面表现出色，但它们主要学习模仿专家轨迹的几何形状，且难以适应部署时的安全约束。我们提出了一种基于查询的代价学习框架，为动态可达的自车轨迹查询估计有界代价，而非密集的鸟瞰图网格单元或少量回归轨迹集合。紧凑的联合场景token捕获连贯的多模态智能体未来轨迹，同时借助具备应急感知的代价聚合与代价引导的簇内MPPI混合，将学习到的代价拓扑转换为可行的自车规划。在nuScenes数据集上，我们的方法优于ST-P3和NMP等已有代价估计规划器，在碰撞率方面超过大多数回归基线，同时在L2指标上保持竞争力，并保留了可解释的代价接口。在真实世界驾驶日志上，所提出的规划器相比SparseDrive和Alpamay降低了碰撞率。

    arXiv:2610.08123v1 Announce Type: cross  Abstract: End-to-end planners based on waypoint regression achieve strong open-loop accuracy, but they primarily learn to mimic expert geometry and remain difficult to adapt to deployment-time safety constraints. We propose a query-based cost-learning framework that estimates bounded costs for dynamically reachable ego trajectory queries, rather than dense BEV cells or a small regressed trajectory set. Compact joint scene tokens capture coherent multimodal agent futures, while contingency-aware cost aggregation and cost-guided intra-cluster MPPI mixing convert the learned cost topology into feasible ego plans. On nuScenes, our method improves over prior cost-estimation planners such as ST-P3 and NMP, outperforms most regression baselines in collision rate, while remaining competitive in L2, and retaining an interpretable cost interface. On real-world driving logs, the proposed planner reduces collision rates compared with SparseDrive and Alpamay
    
[^92]: 时间序列基础模型中衰减的上下文内辨识：反事实输入下的诊断与基于合成强迫系统微调的修复

    Attenuated in-context identification in time-series foundation models: diagnosis under counterfactual inputs and repair by synthetic forced-system fine-tuning

    [https://arxiv.org/abs/2610.08118](https://arxiv.org/abs/2610.08118)

    本文通过精确反事实实验诊断出协变量感知时间序列基础模型在what-if预测中的关键缺陷——TimesFM-2.5和TabPFN-TS完全无记忆、Chronos-2会衰减系统动态效应，并提出利用合成强迫系统数据微调来修复这种衰减。

    

    协变量感知的时间序列基础模型（TSFMs）承诺为配备传感器的工业装置提供无需训练的“假设分析”（what-if）答案：即不同的未来输入会引起输出怎样的变化。我们在具有精确反事实结果的强迫工程系统上检验了这一能力，将Chronos-2、TimesFM-2.5和TabPFN-TS与拟合相同上下文的经典系统辨识方法进行比较。通过其默认的协变量接口，TimesFM-2.5和TabPFN-TS是无记忆的：输入变化的预测效应只是该变化的同期函数（TimesFM-2.5的R²=1.000）。Chronos-2虽然能够在上下文中辨识系统动力学，但会将其衰减：其预测效应仅为真实效应的0.33–0.80倍，恢复出的脉冲响应形状错误，且在单自由度振荡器上，即使使用8192个上下文样本，其误差仍停留在0.57，而仅用256个样本拟合的ARX模型即可达到0.02。推理阶段的上下文抖动可降低所有六个合成（场景上的假设分析误差）……（原文摘要被截断）

    arXiv:2610.08118v1 Announce Type: new  Abstract: Covariate-aware time-series foundation models (TSFMs) promise training-free what-if answers for instrumented plants: the change in output that a different future input would cause. We test this on forced engineering systems with exact counterfactuals, comparing Chronos-2, TimesFM-2.5 and TabPFN-TS with classical system identification fitted to the same context. Through their default covariate interfaces, TimesFM-2.5 and TabPFN-TS are memoryless: the predicted effect of an input change is a same-time function of that change ($R^2 = 1.000$ for TimesFM-2.5). Chronos-2 identifies dynamics in context but attenuates them. Its predicted effect is 0.33-0.80 of the true effect, its recovered impulse response has the wrong shape, and its error on a one-degree-of-freedom oscillator levels off at 0.57 with 8192 context samples, where ARX fitted to 256 samples reaches 0.02. Context dither at inference lowers the what-if error on all six synthetic cla
    
[^93]: 能量感知路径跟随：电动汽车强化学习与NMPC的对比分析

    Energy-Aware Path Following: Comparative Analysis of Reinforcement Learning and NMPC for Electric Vehicles

    [https://arxiv.org/abs/2610.08112](https://arxiv.org/abs/2610.08112)

    本文在统一的Frenet坐标系运动学模型和包含显式再生制动的能量模型下，对比分析了NMPC、PPO强化学习控制器及两种基线方法在电动汽车能量感知路径跟随中的性能，并采用JIT编译的CasADi满足NMPC的实时性要求。

    

    路径跟随控制策略通常面临双目标优化的困境：在最小化与参考路径偏差的同时保持平滑的速度曲线。后一目标对电动汽车（EV）尤为重要，因为通过再生制动回收能量可以延长其有限的续航里程，而这一特性在现有文献中尚未得到充分研究。在本工作中，我们在一个统一的基于Frenet坐标系的运动学车辆模型下，对四种控制器进行了对比分析，并使用了包含显式再生制动的经验证能量模型（VT-CPEM）。我们实现了以下控制器：非线性模型预测控制（NMPC）、近端策略优化（PPO）、增益调度的阿克曼状态反馈基线（PID-SF）以及Stanley几何基线。为满足实时性要求，我们采用JIT编译的CasADi实现了NMPC。此外，我们……

    arXiv:2610.08112v1 Announce Type: cross  Abstract: Path-following control strategies typically follow the bi-objective optimization dilemma: minimizing deviations from a reference path while maintaining smooth speed profiles. The latter objective is especially relevant for Electric Vehicles (EVs), since their limited driving range can be extended by recovering energy through regenerative braking, a feature that has not yet been sufficiently studied in the literature. In this work, we perform a comparative analysis of four controllers under one common Frenet frame-based kinematic vehicle model, utilizing a validated energy model (VT-CPEM) with explicit regenerative braking. Herein, we implement the following controllers: Nonlinear Model Predictive Control (NMPC), Proximal Policy Optimization (PPO), gain-scheduled Ackermann state-feedback baseline (PID-SF), and a Stanley geometric baseline. To satisfy real-time requirements, we implement the NMPC using JIT-compiled CasADi. Moreover, we t
    
[^94]: 利用自回归后训练权重增强扩散语言模型

    Enhancing Diffusion Language Models with Autoregressive Post-Training Weights

    [https://arxiv.org/abs/2610.08108](https://arxiv.org/abs/2610.08108)

    该论文提出将自回归模型的后训练权重更新直接“回收”叠加到扩散语言模型上，无需重新进行扩散后训练即可使扩散模型获得接近直接后训练的性能。

    

    扩散语言模型作为自回归语言模型的一种有前景的替代方案，提供了灵活的token更新顺序和并行解码能力。近期的扩散语言模型通常在扩散转换之前从预训练的自回归模型初始化，以继承其学习到的表示。然而，在转换之后，它们通常忽略了其自回归前身模型丰富的后训练生态系统。在这项工作中，我们证明这些现有的自回归后训练权重更新可以被有效地回收利用，以增强扩散模型。尽管自回归到扩散的转换带来了变化，但直接将自回归后训练权重更新添加到扩散基础模型上仍然有效，使其性能接近直接进行扩散后训练所达到的水平。值得注意的是，自回归和扩散后训练更新在权重空间中几乎正交，却在表示层面引起更为显著对齐的变化。

    arXiv:2610.08108v1 Announce Type: new  Abstract: Diffusion language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) language models, offering flexible token-update orders and parallel decoding. Recent dLLMs are often initialized from pretrained AR models before diffusion conversion in order to inherit their learned representations. After the conversion, however, they typically ignore the extensive post-training ecosystem of their AR ancestors. In this work, we show that these existing AR post-training weight updates can instead be effectively recycled to enhance diffusion models. Despite the changes by AR-to-diffusion conversion, directly adding an AR post-training weight update to a diffusion base model remains effective, bringing its performance close to that achieved by direct diffusion post-training. Notably, AR and diffusion post-training updates are nearly orthogonal in weight space, yet induce substantially more aligned representation changes in the
    
[^95]: 在路由器中生存：面向检索与执行的技能注入优化

    Surviving the Router: Optimizing Skill Injections for Retrieval and Execution

    [https://arxiv.org/abs/2610.08098](https://arxiv.org/abs/2610.08098)

    论文发现现有技能注入攻击评估因忽略检索竞争阶段而将攻击成功率高估87-97%，并提出了感知路由器的攻击方法CORSA，可同时优化技能注入的检索与执行两个环节。

    

    AI智能体越来越依赖由技能路由器动态选择的模块化第三方“技能”来执行复杂任务。尽管近期研究强调了嵌入在这些技能中的提示注入所构成的威胁，但现有的评估通常假设恶意技能已被选中执行的情境。我们表明，这一假设会显著高估攻击的成功率。在现实的多技能环境中，被注入的技能必须首先竞争检索机会，这使得现有注入方法的有效攻击成功率（ASR）降低了87-97%。为了解决这一局限性，我们提出了CORSA（面向路由器感知技能攻击的聚类优化），这是一种路由器感知的攻击方法，能够在相关任务聚类上同时针对检索和执行两个阶段优化技能注入。我们通过引入八个恶意……扩展了SkillRouter提出的基准，在路由器管理的多技能设置下对技能注入攻击进行了评估。

    arXiv:2610.08098v1 Announce Type: cross  Abstract: AI agents increasingly rely on modular third-party "skills" that are dynamically selected by skill routers to execute complex tasks. While recent studies highlight the threat of prompt injections embedded in these skills, existing evaluations often assume settings where the malicious skill is already selected for execution. We show that this assumption can substantially overestimate attack success. In realistic multi-skill environments, injected skills must first compete for retrieval, reducing the effective attack success rate (ASR) of existing injections by 87-97%. To address this limitation, we introduce CORSA (Cluster Optimization for Router-Aware Skill Attacks), a router-aware attack that optimizes skill injections for both retrieval and execution across clusters of related tasks. We evaluate skill injection attacks under router-managed multi-skill settings by extending the benchmark introduced by SkillRouter with eight malicious 
    
[^96]: 基于配对观测点捕获数据的IPv6扩展头部存在模式可解释规则挖掘

    Explainable Rule Mining of IPv6 Extension-Header Presence Patterns from Paired-Vantage Captures

    [https://arxiv.org/abs/2610.08090](https://arxiv.org/abs/2610.08090)

    本文提出阴性对照协议与发送方条件化EH保留率测量两个可复用工具，并将可解释时序逻辑规则挖掘器应用于JAMES配对观测点数据集，发现挖掘出的主导分片EH规则实为包内共现而非真正的时序模式。

    

    IPv6扩展头部（EH），如分片、段路由和在线遥测，在运营上非常重要，但在传输过程中却被广泛丢弃，从数据包捕获中刻画其行为是一个反复出现的测量问题。我们探究可解释的挖掘器能否恢复出人类可读的EH行为规则，并贡献了两个可复用的工具：一个是阴性对照协议，用于诊断挖掘出的“时序”网络规则反映的是真正的跨数据包动态，还是仅仅是数据包内的共现；另一个是发送方条件化的按家族EH保留率测量。将可解释的时序逻辑规则挖掘器应用于JAMES配对观测点数据集后，我们恢复出一个可移植的分片EH规则，但该协议揭示这实际上是一种数据包内的、近乎定义性的共现，而非时序模式，因此时序逻辑机制对这个主导规则没有起到任何作用；保留率测量则独立地恢复了……

    arXiv:2610.08090v1 Announce Type: cross  Abstract: IPv6 extension headers (EHs), such as fragmentation, segment routing, and in-situ telemetry, are operationally important yetwidely dropped in transit, and characterising their behaviour from packet captures is a recurring measurement problem. We ask whetheran explainable miner can recover human-readable rules of EH behaviour, and we contribute two reusable tools: a negative-control protocol that diagnoses whether a mined "temporal" network rule reflects genuine cross-packet dynamics or mere within-packetco-occurrence, and a sender-conditioned, per-family EH-retention measurement. Applying an interpretable temporal-logic rule miner to the JAMES paired-vantage dataset, we recover a portable Fragment-EH rule that the protocol reveals to be a within-packet,near-definitional co-occurrence rather than a temporal pattern, so the temporal-logic machinery does no work for this dominant rule;the retention measurement independently recovers the e
    
[^97]: ProximalFM：隐藏混杂下的摊销式近端因果推断

    ProximalFM: Amortized Proximal Causal Inference under Hidden Confounding

    [https://arxiv.org/abs/2610.08078](https://arxiv.org/abs/2610.08078)

    该论文提出ProximalFM，利用先验数据拟合网络（PFN）以摊销式贝叶斯方法在隐藏混杂下进行近端因果推断，通过先验正则化缓解了非参数近端估计中病态积分方程的数据饥渴、超参数敏感和优化不稳定等问题。

    

    标准的因果识别方法通常假设不存在未观测的混杂因素，当相关混杂因素未被观测到时这些方法可能会失效。近端因果推断则转而使用代理变量在隐藏混杂存在的情况下识别因果效应。然而，非参数近端估计在实践中可能颇具挑战性：恢复条件平均处理效应（CATE）等因果估计量需要求解一个病态的积分方程，该方程对数据需求量大、对超参数敏感且优化过程不稳定。对此类模型进行贝叶斯推断提供了一种理想的替代方案，可通过先验进行正则化来缓解上述困难。然而，计算后验本身也具有挑战性，因为典型的似然函数中会包含潜变量。借鉴近期表格基础模型在后门、工具变量和前门等设定中取得的成功，我们提出先验数据拟合网络（PFN）在……方面具有独特优势（摘要在此处截断）

    arXiv:2610.08078v1 Announce Type: cross  Abstract: Standard causal identification methods often assume no unmeasured confounding and can fail when relevant confounders are unobserved. Proximal causal inference instead uses proxy variables to identify effects under hidden confounding. However, nonparametric proximal estimation can be challenging in practice: recovering causal estimands such as the conditional average treatment effect (CATE) requires solving an ill-posed integral equation that is data-hungry, hyperparameter-sensitive, and optimization-unstable. Bayesian inference for such models provides a desirable alternative, mitigating these difficulties by regularizing through the prior. However, computing a posterior is itself challenging, as a typical likelihood function will include latent variables. Following the recent success of tabular foundation models in backdoor, instrumental variable, and frontdoor settings, we propose that prior-data fitted networks (PFNs) are uniquely s
    
[^98]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^99]: 优化编码器：重新思考神经场的二阶元学习

    Optimization Encoders: Rethinking Second-Order Meta-Learning for Neural Fields

    [https://arxiv.org/abs/2610.08075](https://arxiv.org/abs/2610.08075)

    该论文将潜优化形式化为“优化编码器”，统一了二阶微分、潜参数化与任务监督的角色，并据此提出基于等变变换器的注意力潜场MetaLF，实现了编码过程与解码器的端到端二阶元学习训练。

    

    条件神经场能够连续地表示信号，但其有效性取决于如何从观测数据中推断条件潜表示。在元学习中，这种编码过程通过由解码器诱导的梯度更新来实现，从而将表示学习与解码器设计直接绑定在一起。我们通过将潜优化解释为一种“优化编码器”来形式化这一联系，统一了二阶微分、潜参数化和任务监督的角色。这一概念使得二阶元学习能够对编码过程与解码器进行端到端的联合训练，并阐明了一阶近似所舍弃的学习路径。基于这一视角，我们提出了注意力潜场，这是一种基于等变变换器的神经场，通过自注意力机制对潜点云进行上下文化建模。这些交互作用既塑造了场的预测，也塑造了相应的更新……

    arXiv:2610.08075v1 Announce Type: new  Abstract: Conditional neural fields represent signals continuously, but their effectiveness depends on how the conditional latent representations are inferred from observed data. In meta-learning, this encoding occurs through gradient updates induced by the decoder, tying representation learning directly to decoder design. We formalize this connection by interpreting latent optimization as an optimization encoder, unifying the roles of second-order differentiation, latent parameterization, and task supervision. This concept enables second-order meta-learning for end-to-end training of the encoding procedure alongside the decoder, and clarifies which learning pathway first-order approximations discard. Guided by this view, we introduce Attentive Latent Fields (MetaLF), an equivariant transformer-based neural field that contextualizes a latent pointcloud through self-attention. These interactions shape both field predictions and the updates that con
    
[^100]: 检测到偏移并不足够：线性表示修复的精确极小极大极限

    Detecting a Shift Is Not Enough: Exact Minimax Limits of Linear Representation Repair

    [https://arxiv.org/abs/2610.08069](https://arxiv.org/abs/2610.08069)

    该论文将两个数据源间均值偏移的消除建模为统计决策问题，推导出线性表示修复的精确有限样本极小极大风险，并揭示了“检测-修复差距”：检测偏移仅需信噪比 κ 远大于 √d，而实际修复则需 κ 与维度 d 同阶。

    

    两个数据源之间的均值偏移可能很容易检测，但若不大幅改变其表示，则难以消除。我们将该消除问题建模为一个统计决策问题：从 $\mathbb{R}^d$ 中成对校准测量的含噪差异中，学习一个线性映射，在硬失真预算约束下同时应用于两个数据源，使得在新数据上残留的偏移尽可能小。我们推导出了所有此类映射上的精确有限样本极小极大风险：$(d-k) \mathbb{E}[1/(d+2J)]$，其中 $J\sim\mathrm{Pois}(\kappa/2)$，预算允许删除 $k$ 个方向，$\kappa$ 为校准信噪比。在不了解 $\kappa$ 或噪声尺度的情况下，投影掉均值校准差异即可达到该风险。这揭示了一个“检测-修复差距”：检测偏移只需 $\kappa\gg\sqrt d$，而在恒定失真下去除其固定比例则需要 $\kappa\asymp d$，与估计其方向的要求相同。

    arXiv:2610.08069v1 Announce Type: new  Abstract: A mean shift between two data sources can be easy to detect but hard to remove without substantially changing their representations. We cast its removal as a statistical decision problem: from noisy differences between paired calibration measurements in $\mathbb{R}^d$, learn one linear map, applied to both sources under a hard distortion budget, that leaves as little of the shift as possible on fresh data. We derive the exact finite-sample minimax risk over all such maps, $(d-k) \mathbb{E}[1/(d+2J)]$ with $J\sim\mathrm{Pois}(\kappa/2)$, where the budget allows deleting $k$ directions and $\kappa$ is the calibration signal-to-noise ratio. Projecting out the mean calibration difference attains it without knowing $\kappa$ or the noise scale. This exposes a detection-repair gap: detecting the shift needs only $\kappa\gg\sqrt d$, whereas removing a fixed fraction of it at constant distortion needs $\kappa\asymp d$, as for estimating its direc
    
[^101]: 语言承载着专家的印象：基于评估工具锚定的LLM评判器可实现咨询质量评估的跨域迁移并超越域内训练

    Language Carries the Expert's Impression: Instrument-Anchored LLM Judges Transfer Counseling-Quality Assessment and Beat In-Domain Training

    [https://arxiv.org/abs/2610.08055](https://arxiv.org/abs/2610.08055)

    该研究发现，基于专家评分工具锚定构念、由小型开源LLM生成的会话级得分，能够跨领域迁移专家对咨询质量的总体印象评估，且跨域训练的效果甚至优于域内训练。

    

    arXiv:2610.08055v1 公告类型：new 摘要：对双人咨询对话中沟通质量的自动评估受制于数据瓶颈：专家评分的语料库规模小且扩充成本高昂。我们研究了专家总体印象预测在三个德语模拟咨询语料库（两个全科医疗语料库、一个学校相关的家长-教师语料库；n=195个经专家评分的会谈，其中一个语料库经过量表等值化处理）之间的跨域迁移。在其他域上训练的效果优于在目标域内训练：留一域迁移达到嵌套Spearman ρ = 0.54，而目标域内训练不超过0.48，会话层面的配对差距为+0.15，且当训练集规模匹配时该差距仍保持+0.12，因此这并不仅仅是数据量的问题。决定性的特征是来自小型开源权重LLM阅读双说话人转录文本所得到的会话级构念得分，这些构念主要源自专家的评分工具：由工具衍生的构念集合提升了……

    arXiv:2610.08055v1 Announce Type: new  Abstract: Automatic assessment of communication quality in dyadic counseling conversations is bottlenecked by data: expert-rated corpora are small and expensive to grow. We study cross-domain transfer of expert overall-impression prediction across three German corpora of simulated counseling (two general-practice medical, one school-related parent-teacher; $n=195$ expert-rated sessions, one corpus after scale equating). Training on the other domains beats training in-domain: leave-one-domain-out transfer reaches nested Spearman $\rho = 0.54$ against $\le 0.48$ within the target domain, a paired session-level gap of $+0.15$ that holds at $+0.12$ when the training-set sizes are matched, so it is not simply data volume. The decisive features are session-level construct scores from small open-weight LLMs reading the two-speaker transcript, with the constructs largely derived from the experts' rating instruments: the instrument-derived battery lifts a 
    
[^102]: 面向低秩适应的黎曼几何

    A Riemannian Geometry for Low-rank Adaptation

    [https://arxiv.org/abs/2610.08049](https://arxiv.org/abs/2610.08049)

    该论文从商流形几何的角度分析 LoRA 参数化的等价关系，并提出一种在该等价关系下不变的黎曼度量，使每一步梯度更新都经过预条件化并切实改变损失值，从而实现更高效的参数高效微调。

    

    低秩适应（LoRA）作为一种参数高效的微调技术被广泛用于预训练深度神经网络，它通过低秩矩阵 $BA^\top$ 来近似全量微调所带来的权重更新。这种参数化导致了如下等价关系：对任意可逆矩阵 $G$，有 $(B, A) \sim (BG^{-1}, AG^\top)$，因为 $BA^\top = BG^{-1}(AG^\top)^\top$，即这两组参数给出相同的损失值。该等价关系诱导出一个商流形，其中对所有 $G$ 的矩阵 $(BG^{-1}, AG^\top)$ 都被视为同一对象，从而消除了损失值保持不变的冗余方向。为了尊重该流形的几何结构，原始搜索空间被赋予了一个在等价关系下保持不变的黎曼度量。这样的度量在每一步梯度更新中引入预条件化，并确保每一次通过 LoRA 进行的权重更新都会改变损失值，从而实现高效优化。在本文中，我们提出……（原文摘要在此处截断）

    arXiv:2610.08049v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is widely used as a parameter-efficient fine-tuning technique for pre-trained deep neural networks, which approximates the weight update via full fine-tuning by a low-rank matrix $BA^\top$. This parameterization leads to the equivalence relation $(B, A) \sim (BG^{-1}, AG^\top)$ for any invertible matrix $G$ because $BA^\top = BG^{-1}(AG^\top)^\top$ and thus both pairs yield the same loss value. This relation induces a quotient manifold where matrices $(BG^{-1}, AG^\top)$ for all $G$ are identified, eliminating redundant directions along which the loss value remains unchanged. To respect the geometry of this manifold, the original search space is endowed with a Riemannian metric that is invariant under the equivalence relation. Such a metric induces preconditioning at each gradient step and ensures that each weight update via LoRA changes the loss value, leading to efficient optimization. In this paper, we propo
    
[^103]: DAEDALUS：从自生成任务中引导代理记忆

    DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks

    [https://arxiv.org/abs/2610.08048](https://arxiv.org/abs/2610.08048)

    DAEDALUS通过探索者代理自生成练习任务、解决者代理从失败中提炼启发式规则并验证其有效性，从而在无需现有任务或人工验证器的情况下自动构建可复用的代理记忆。

    

    LLM代理通常缺乏在新环境中可靠行动所需的操作性知识，因为它们必须自行发现特定的工具行为或环境约定。由于没有对过去尝试的记忆，它们会在不同任务中重复相同的错误，导致更多的任务失败和更长的轨迹。为了解决这个问题，代理系统通常依赖人工编写的指南，或者依赖从训练任务和神谕验证器构建的程序性记忆，但这两种方式都需要对环境的先验知识。我们提出了DAEDALUS，这是一种在没有现成任务或神谕验证器的情况下，从自生成练习中引导可复用代理记忆的方法。DAEDALUS将两个代理配对：一个是与环境交互以生成具有挑战性但可解决任务的探索者，另一个是尝试解决这些任务的解决者。每次解决者失败后都会推导出一个启发式方法，只有当解决者在上下文中借助该启发式方法反复成功后，该启发式方法才会被接受。

    arXiv:2610.08048v1 Announce Type: new  Abstract: LLM agents often lack the operational knowledge to act reliably in new environments, as they must discover specific tool behaviors or environment conventions on their own. Without memory of past attempts, they repeat the same mistakes across tasks, leading to more task failures and longer trajectories. To address this, agentic systems typically rely on human-written guidelines or on procedural memory built from training tasks and an oracle verifier, both of which require prior knowledge of the environment. We present DAEDALUS, a method for bootstrapping reusable agent memory from self-generated practice without existing tasks or oracle verifiers. DAEDALUS pairs two agents: an explorer that interacts with the environment to generate challenging yet solvable tasks, and a solver that attempts them. A heuristic is derived from each solver failure and accepted only after the solver repeatedly succeeds with that heuristic in context. These out
    
[^104]: SepsisLens：面向可分解脓毒症早期预警的结构保持序列建模

    SepsisLens: Structure-Preserving Sequence Modelling for Decomposable Early Sepsis Warning

    [https://arxiv.org/abs/2610.08046](https://arxiv.org/abs/2610.08046)

    SepsisLens通过在风险组合前保持变量级时序状态，并利用结构化风险头部从显式的变量级和器官级组件组合多时间窗口风险，实现了警报可追溯至生理信号的可分解脓毒症早期预警。

    

    从ICU记录中进行脓毒症早期预警可以被建模为一个结构保持的预测问题。模型需要从不规则的测量数据中检测病情恶化，同时保持每个警报与支持该警报的生理信号之间的关联。许多时序模型将临床变量融合为患者级表示，虽然支持标量风险预测，但削弱了临床分解所需的结构。我们提出了SepsisLens，它在风险组合之前保持以变量为索引的时序状态。观测感知表示编码每个变量的动态变化和测量历史，同时共享的时序编码器对每条轨迹建模而不折叠变量维度。StructuredRiskHead从显式的变量级和器官级组件组合出多时间窗口的风险。我们在统一的发病前评估协议下，在三个公开ICU队列和一个私立医院队列上对SepsisLens进行了评估。SepsisLens实现了……（原文摘要在此处截断）

    arXiv:2610.08046v1 Announce Type: new  Abstract: Early sepsis warning from ICU records can be cast as a structure-preserving prediction problem. A model needs to detect deterioration from irregular measurements while keeping each alert connected to the physiological signals that support it. Many temporal models fuse clinical variables into a patient-level representation, supporting scalar risk prediction but weakening the structure needed for clinical decomposition. We present SepsisLens, which preserves variable-indexed temporal states until risk composition. Observation-aware representations encode each variable's dynamics and measurement history, while a shared temporal encoder models each trajectory without collapsing the variable axis. The StructuredRiskHead composes multi-horizon risk from explicit variable-level and organ-level components. We evaluate SepsisLens on three public ICU cohorts and one private-hospital cohort under a common pre-onset protocol. SepsisLens achieves str
    
[^105]: Spectra：仿真推理中面向测试时先验自适应的精确成分传输方法

    Spectra: Exact Component Transport for Test-Time Prior Adaptation in Simulation-Based Inference

    [https://arxiv.org/abs/2610.08021](https://arxiv.org/abs/2610.08021)

    Spectra利用精确的得分传输恒等式，使冻结的扩散式SBI模型能够在测试时以闭式形式适应结构化的先验变化，无需额外的仿真或训练，在六个基准测试中于强先验偏移下实现了准确的后验推断。

    

    基于仿真的推理（SBI）已成为对似然函数难以评估或无法评估的复杂科学模型进行贝叶斯推断的一种强大方法。摊销式SBI从模拟数据中学习可复用的推理模型，从而能够对新观测进行快速的后验推断，而现代生成模型使这些模型的表达能力日益增强。然而，这种复用仅限于训练时所选择的先验分布，而科学分析往往需要在知识不断积累或检验不同假设时修改先验。我们提出了Spectra，这是一种面向基于扩散模型的SBI的测试时自适应方法。Spectra利用精确的得分传输恒等式，对于结构化的先验变化，能够以闭式形式从冻结的扩散模型中获得自适应后的得分，无需额外的仿真或训练。在六个SBI基准测试中，Spectra在强先验偏移下以较低的在线采样……实现了准确的自适应。

    arXiv:2610.08021v1 Announce Type: new  Abstract: Simulation-based inference (SBI) has become a powerful approach to Bayesian inference in complex scientific models whose likelihoods are difficult or impossible to evaluate. Amortized SBI learns reusable inference models from simulated data, enabling rapid posterior inference for new observations, and modern generative models have made these models increasingly expressive. However, this reuse is limited to the prior distribution chosen during training, whereas scientific analyses often need revised priors as knowledge accumulates or alternative assumptions are tested. We introduce Spectra, a test-time adaptation method for diffusion-based SBI. Spectra uses an exact score-transport identity to obtain the adapted score from a frozen diffusion model in closed form for structured prior changes, without additional simulation or training. Across six SBI benchmarks, Spectra achieves accurate adaptation under strong prior shifts at low online sa
    
[^106]: 从第一性原理学习一致的分子力学力场

    Learning consistent molecular mechanics force fields from first principles

    [https://arxiv.org/abs/2610.08020](https://arxiv.org/abs/2610.08020)

    本文提出grappa-fullFF统一框架，首次从第一性原理参考数据中一致且同时地学习分子力场的成键与非成键参数，克服了传统方法依赖经验非成键参数的局限。

    

    即使机器学习原子间势（MLIPs）已接近从头算精度，经典力场（FFs）仍然是大尺度模拟的主力工具。经典力场将总构型能量分解为简单的有效相互作用，其参数传统上根据原子或键的类型来分配，这使得模拟高效，但也限制了它们跨不同构型自适应的能力。近来的机器学习方法通过将成键参数推断为局部原子环境的函数，提高了这些力场中成键参数的精度和可迁移性，但在实际模拟中仍然依赖经验性的非成键参数。在本工作中，我们提出了一种统一的方法 grappa-fullFF，它能够从从头算参考数据中一致地、同时地学习成键参数和非成键参数。通过引入物理启发的正则化方法（即对静电势进行监督）……

    arXiv:2610.08020v1 Announce Type: cross  Abstract: Classical force fields (FFs) remain the workhorse for large-scale simulations even as machine-learned interatomic potentials (MLIPs) approach ab initio accuracy. They decompose total configuration energies into simple effective interactions whose parameters are traditionally assigned based on atom or bond types, enabling efficient simulations but also limiting their ability to adapt across configurations. Recent machine learning approaches have improved the accuracy and transferability of bonded parameters in these FFs by inferring them as functions of local atomic environments, but still rely on empirical nonbonded parameters for practical simulations. In this work, we introduce a unified approach, \texttt{grappa-fullFF}, which learns both bonded and nonbonded parameters \emph{consistently} and simultaneously from ab initio reference data. By incorporating physically inspired regularization via supervision of the electrostatic potenti
    
[^107]: FOSLS-deRhaNN：H(div) 与 H(curl) 的原生 de Rham 神经类及其在偏微分方程一阶系统最小二乘神经网络方法中的应用

    FOSLS-deRhaNN: native de Rham neural classes for H(div) and H(curl) with applications to first-order system least-squares neural network methods for partial differential equations

    [https://arxiv.org/abs/2610.08016](https://arxiv.org/abs/2610.08016)

    该论文构造了原生属于 H(div) 和 H(curl) 空间的 de Rham 神经逼近类——无需网格或有限元模拟即可精确表达界面跳变，并将其应用于求解偏微分方程的一阶系统最小二乘（FOSLS）神经网络方法。

    

    我们构造了原生（native）属于图空间 H(div) 与 H(curl) 的神经逼近类，适用于二维和三维，对于 H(div) 更适用于任意维度。无论参数取何值，每个实现都严格属于该空间；并且采用带折点的势函数（如 ReLU 网络）时，容许的界面跳变可在有限宽度下出现。这些类是标量网络和按分量网络在 de Rham 复形的固定算子作用下的像，不涉及网格或对有限元的模拟。对于 R^n 中的 H(div)，在同一基础上给出两个原生类：其一是带反对称势 A 的 Div A + R_n q + h，其中散度 q 作为显式未知量；其二是 Div A + z，其中 z 为 H^1 场；对于 H(curl)，相应的类为二维情形的 grad φ + S r + h 以及二维和三维情形的 grad φ + z。在所有这些类中，场的每个界面跳变均由……

    arXiv:2610.08016v1 Announce Type: cross  Abstract: We construct neural approximation classes native to the graph spaces H(div) and H(curl), in two and three dimensions and, for H(div), in any dimension. Every realization lies in the space for all parameter values, and with kinked potentials, such as ReLU networks, the admissible jumps appear at finite width. The classes are images of scalar and componentwise networks under fixed operators of the de Rham complex, and do not involve a mesh or finite element emulation. For H(div) in R^n two native classes are given on an equal footing, with a skew-symmetric potential $A$: $\mathrm{Div}\,A+R_nq+\mathbf{h}$, with the divergence $q$ as an explicit unknown, and $\mathrm{Div}\,A+\mathbf{z}$ with an $H^1$ field $\mathbf{z}$; for H(curl) the analogous classes are $\mathrm{grad}\,\phi+Sr+\mathbf{h}$ in two dimensions and $\mathrm{grad}\,\phi+\mathbf{z}$ in two and three dimensions. In all of them every interface jump of the field is carried by th
    
[^108]: 无需实验读数即可预测表型活性吗？

    Can phenotypic activity be predicted without experimental readouts?

    [https://arxiv.org/abs/2610.07997](https://arxiv.org/abs/2610.07997)

    在控制预训练数据泄漏和细胞毒性混杂因素后，基于分子-形态学对比预训练的分子编码器在预测Cell Painting表型活性方面并不优于简单的理化描述符。

    

    在配对的分子-形态学数据上对比预训练的分子编码器（如CLOOME和CellCLIP）已被提出作为表型预测的廉价替代方案，从而避免进行Cell Painting实验。我们在一个旨在控制两类可能夸大表面性能的混杂因素的评估协议下，对这一想法进行了检验：一是跨越编码器自身预训练边界的数据泄漏，二是表型活性与细胞毒性之间的相关性。我们在两个不同的Cell Painting筛选上测试了六种表示方法，其中包括一个与CLOOME输入和层数相匹配的非预训练MLP对照组。我们发现，一旦控制了这些混杂因素，预训练的分子编码器相比简单的理化描述符并无明显优势，并且在各种表示方法下，毒性通常都比表型活性更容易预测。我们的结果表明，需要进行具备泄漏感知、混杂因素控制的评估。

    arXiv:2610.07997v1 Announce Type: new  Abstract: Molecular encoders contrastively pretrained on paired molecule-morphology data, such as CLOOME and CellCLIP, have been proposed as cheap surrogates for phenotypic prediction, avoiding the need to run a Cell Painting assay. We evaluate this idea for these molecular encoders under a protocol designed to control for two confounds that can inflate apparent performance: leakage across an encoder's own pretraining boundary, and the correlation between phenotypic activity and cytotoxicity. Testing six representations, including a non-pretrained MLP control matching CLOOME's input and layer count, on two distinct Cell Painting screens, we find that once these confounds are controlled for, the pretrained molecular encoders show no clear advantage over plain physicochemical descriptors, and that toxicity is generally easier to predict than phenotypic activity across representations. Our results suggest leakage-aware, confound-controlled evaluation
    
[^109]: TICDA：表格化上下文数据归因

    TICDA: Tabular In-Context Data Attribution

    [https://arxiv.org/abs/2610.07996](https://arxiv.org/abs/2610.07996)

    提出TICDA方法，通过在表格基础模型潜在表示上训练的线性代理模型直接量化上下文中每个示例对预测的影响，解决了重采样和基于梯度的数据归因方法在上下文学习场景中失效的问题。

    

    表格基础模型（TFMs）通过以上下文中提供的带标签示例为条件，在无需任何参数更新的情况下实现了强大的预测性能。然而，单个示例如何影响给定的预测仍然知之甚少。这一差距在实践中很重要：上下文通常由任何可用的带标签数据组装而成，可能导致混入错误标注、冗余或低质量的示例，从而降低性能。标准的数据归因方法无法直接迁移到TFM场景：基于重采样的方法（如DemoShapley）需要组合级数量的前向传播，而基于梯度的估计器（如影响函数）需要计算训练点对模型参数的影响，但上下文学习从不更新参数。我们提出了TICDA，一种直接从在TFM潜在表示上训练的线性代理模型中测量上下文中每个示例影响的方法。

    arXiv:2610.07996v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) achieve strong predictive performance by conditioning on labeled demonstrations provided in context, without any parameter update. Yet how individual demonstrations shape a given prediction remains poorly understood. This gap matters in practice: the context is often assembled from whatever labeled data is available, potentially leading to the inclusion of mislabeled, redundant, or low-quality examples that degrade performance. Standard data attribution methods do not transfer to the TFM setting: resampling-based approaches such as DemoShapley require a combinatorial number of forward passes, and gradient-based estimators such as influence functions require computing training point's effect on the model parameters, which in-context learning never updates. We introduce TICDA, a method that measures the influence of every demonstration in the context directly from linear surrogates trained on TFM latent e
    
[^110]: 对模型合并的更广泛审视：重新思考任务算术所引发的隐式正则化

    A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic

    [https://arxiv.org/abs/2610.07990](https://arxiv.org/abs/2610.07990)

    本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。

    

    模型合并旨在通过组合各个特定任务模型的权重，以低成本的方式构建多任务模型。为了在多个任务上表现良好，大多数现有的合并方法使用额外的数据集来寻找特定任务权重更新的最佳线性组合系数。然而，我们在这项标准实践中识别出一种隐式正则化：对系数进行搜索会将候选模型限制在由特定任务权重更新所张成的子空间内。在本工作中，我们研究这种正则化是否真的有用。令人惊讶的是，实证结果表明，在没有这种正则化的情况下直接优化合并模型的权重，能够在多种架构、多个领域、甚至在每个类别仅有一个样本的极端数据受限场景下显著提升常见合并方法的性能。此外，直接优化预训练模型的权重甚至优于一些现有方法……（摘要在此处被截断）

    arXiv:2610.07990v1 Announce Type: cross  Abstract: Model merging aims to build a multi-task model cheaply by combining the weights of individual task-specific models. To perform well across multiple tasks, most existing merging methods use an additional dataset to find the coefficients for the best linear combination of task-specific weight updates. However, we identify an implicit regularization in this standard practice: searching over coefficients restricts the candidate models to a subspace spanned by task-specific weight updates. In this work, we investigate whether this regularization is actually useful. Surprisingly, empirical results show that optimizing merged-model weights without this regularization significantly boosts the performance of common merging methods across multiple architectures, domains, and even in an extremely data-limited scenario where only one instance is available per class. Moreover, directly optimizing the pretrained model weights even outperforms some e
    
[^111]: 高阶模型是因高阶原因而获胜吗？重新思考超图学习中的性能增益

    Do Higher-Order Models Win for Higher-Order Reasons? Rethinking Performance Gains in Hypergraph Learning

    [https://arxiv.org/abs/2610.07981](https://arxiv.org/abs/2610.07981)

    该研究提出一种受控性能归因框架，通过扰动高阶信息同时保留成对信息，发现在25个超图学习基准上高阶模型的性能优势大多并非真正来自高阶信息，即使没有高阶信息其大部分优势仍可实现。

    

    高阶模型（如超图神经网络）在超图学习基准测试中往往优于低阶基线模型，其优势通常被归因于它们能够利用高阶信息。然而，仅凭更好的性能并不能证实这一解释。因此我们提出问题：高阶模型是因高阶原因而获胜吗？为了研究这个问题，我们引入了一个受控的性能归因框架，该框架在扰动高阶信息的同时保留低阶（即成对）信息。在涵盖三个任务的25个常用超图学习基准上，我们经常观察到一种有趣的现象：高阶模型最初优于低阶基线模型，但在扰动后仍保留了大部分优势。这表明所观察到的大部分优势在没有高阶信息的情况下仍然可以实现。随后，我们进一步研究了潜在的低阶（摘要在此处截断）

    arXiv:2610.07981v1 Announce Type: new  Abstract: Higher-order models (e.g., hypergraph neural networks) often outperform lower-order baselines on hypergraph learning benchmarks, and their advantages are commonly attributed to their ability to exploit higher-order information. However, better performance alone does not establish this explanation. We therefore ask: Do higher-order models win for higher-order reasons? To investigate this question, we introduce a controlled performance-attribution framework that perturbs higher-order information while preserving the lower-order, i.e., pairwise, information. Across 25 commonly used hypergraph learning benchmarks spanning three tasks, we frequently observe an intriguing pattern: higher-order models originally outperform lower-order baselines, yet retain most of their advantage after perturbation. This suggests that much of the observed advantage remains achievable without the higher-order information. We then investigate potential lower-orde
    
[^112]: 对数凹随机效用模型中从人类反馈学习排序

    Learning a Ranking from Human Feedback in Log-Concave Random Utility Models

    [https://arxiv.org/abs/2610.07973](https://arxiv.org/abs/2610.07973)

    该论文在对数凹随机效用模型下研究了通过全排序反馈和仅胜者反馈两种人类比较反馈来恢复物品排序的问题，建立了最坏情况样本复杂度下界，并设计了无需知晓噪声分布即可匹配该下界的算法。

    

    我们研究根据未知的数值效用来恢复一组固定物品的排序问题。在每次与环境的交互中，学习者将物品集合呈现给人类，并接收两种类型的比较反馈。在全排序反馈下，每次交互揭示所有物品的带噪声排序；而在仅胜者反馈下，每次交互只揭示排名第一的物品。在这两种设置中，我们使用具有对数凹噪声的随机效用模型来刻画人类反馈，并研究以高概率恢复ε-精确排序所需的观测数量。这一新颖的准则仅容忍效用差异小于ε的物品之间的排序错误。对于两种反馈类型，我们都建立了最坏情况下的样本复杂度下界，并开发出在对数因子范围内匹配这些下界的算法。这两种算法都不需要预先知道噪声分布，而……

    arXiv:2610.07973v1 Announce Type: new  Abstract: We study the problem of recovering the ranking of a fixed set of items according to their unknown numerical utilities. At each interaction with the environment, a learner presents the item set to a human and receives comparative feedback of two types. Under full-ranking feedback, each interaction reveals a noisy ranking of all items, whereas under winner-only feedback, it reveals only the item ranked first. In both settings, we model human feedback using a random utility model with log-concave noise and study the number of observations needed to recover an $\epsilon$-accurate ranking with high probability. This novel criterion tolerates ordering errors only between items whose utilities differ by less than $\epsilon$. For both feedback types, we establish worst-case sample-complexity lower bounds and develop algorithms that match these bounds up to logarithmic factors. Neither algorithm requires knowledge of the noise distribution, while
    
[^113]: DecepEval：用于评估大语言模型智能体欺骗行为的基准测试

    DecepEval: A Benchmark for Evaluating Deception in LLM Agents

    [https://arxiv.org/abs/2610.07967](https://arxiv.org/abs/2610.07967)

    提出DecepEval基准，借鉴经典欺诈理论构建“LLM欺骗菱形”框架，通过1,532个实例、28个专业场景以及压力、激励、机会、冲突四种外部条件，系统性评估大语言模型智能体的欺骗行为。

    

    随着大语言模型（LLM）智能体日益自主化，它们可能通过欺骗行为来追求任务表现，这引发了人们对其可靠部署的担忧。现有评估表明LLM智能体确实可能进行欺骗，但往往只考察孤立场景或 narrowly 定义的狭窄条件，限制了对欺骗在何种情况下更可能发生的系统性理解。为填补这一空白，我们提出了DecepEval，这是一个包含1,532个实例、涵盖3个任务族和28个专业场景的基准测试。借鉴经典欺诈理论，我们提出了“LLM欺骗菱形”框架，该框架刻画了可能诱发欺骗的四种外部条件：压力、激励、机会和冲突。DecepEval将每个实例的中性版本与诱导版本配对，以衡量欺骗率随条件变化的情况，同时通过明确的任务事实和可观察的智能体行为来帮助区分欺骗行为与能力不足导致的错误。

    arXiv:2610.07967v1 Announce Type: new  Abstract: As large language model (LLM) agents become increasingly autonomous, they may pursue task performance through deception, raising concerns about their reliable deployment. Existing evaluations show that LLM agents can deceive, but often examine isolated scenarios or narrowly defined conditions, limiting systematic understanding of when deception becomes more likely. To address this gap, we introduce DecepEval, a benchmark comprising 1,532 instances across 3 task families and 28 professional scenarios. Drawing on classical fraud theories, we propose the LLM Deception Diamond framework, which characterizes four external conditions that may induce deception: pressure, incentive, opportunity, and conflict. DecepEval pairs neutral and induced versions of each instance to measure condition-dependent changes in deception rates, while explicit task facts and observable agent behavior help distinguish deception from capability-related errors. Eval
    
[^114]: 基于变分自编码器（VAE）的音频解码器中的特征编码：输入、深度与分布的影响

    Feature Encoding in VAE-based Audio Decoders: Effects of Input, Depth and Distribution

    [https://arxiv.org/abs/2610.07966](https://arxiv.org/abs/2610.07966)

    本研究通过对RAVE等基于VAE的音频解码器内部激活进行逐层与跨层聚类分析，系统揭示了模型对音高、BPM等音乐特征的编码规律，发现合成音频编码效果更好，而自然音频需借助非线性探针才能实现有效解码。

    

    诸如实时音频变分自编码器（RAVE）等神经音频合成模型能够实现令人印象深刻的生成质量，但其内部表示如何编码音乐特征仍知之甚少。我们对在三个不同音乐领域训练的模型中RAVE解码器的激活进行了系统的逐层及跨层聚类分析，并使用四种刺激类型进行测试。随后，我们使用通用EnCodec模型评估了架构的泛化能力。对于RAVE，我们发现合成刺激在各模型和音频特征中均能被良好编码（音高 |ρ|=0.45，为零假设的5.1倍；BPM |ρ|=0.76，为零假设的8.6倍）。当使用自然音频时，这些结果有所降低但仍实质性地存在（各特征平均 |ρ|=0.25，为零假设的2.8倍）。当使用非线性探针时，自然音频表现出更强的编码效果（各特征平均 R²=0.56，为零假设的18倍，较线性探针的R²获得0.152的非线性增益）。

    arXiv:2610.07966v1 Announce Type: cross  Abstract: Neural audio synthesis models like the Realtime Audio Variational autoEncoder (RAVE) achieve impressive genera tion quality, yet how their internal representations encode musical features remains poorly understood. We present a systematic layer-wise and cross-layer cluster analysis of RAVE decoder activations across three models trained on different musical domains, tested with four stimulus types. We then evaluate architectural generalization with a general purpose EnCodec model. For RAVE, we find that synthetic stimuli are encoded well across models and audio features (pitch |\r{ho}|=0.45, 5.1x the null, BPM |\r{ho}| = 0.76, 8.6x the null). These results are reduced but still substantively apparent when using natural audio (mean across features |\r{ho}|=0.25, 2.8x the null). Natural audio sees a stronger encoding when nonlinear probes are used (mean across features R2=0.56, 18x the null, +0.152 nonlinear gain over the linear probe R2
    
[^115]: 广义Matheron变分隐式过程

    Generalized Matheron Variational Implicit Processes

    [https://arxiv.org/abs/2610.07938](https://arxiv.org/abs/2610.07938)

    提出广义Matheron变分隐式过程（GMVIP）这一路径变分族，通过从先验采样并施加锚定在诱导点上的校正来构造后验样本，对高斯过程先验可恢复标准诱导变量变分GP构造，对一般隐式先验则能在总体极限下保持先验的均值、协方差及结构变异性。

    

    隐式过程先验通过采样前向机制（如贝叶斯神经网络和随机模拟器）来定义函数上的分布，但其函数空间密度通常不可获得。我们提出了广义Matheron变分隐式过程，这是一种用于此类先验后验推断的路径变分族。对于高斯过程先验，GMVIP能够恢复标准的诱导变量变分高斯过程构造；对于一般隐式先验，其经验协方差构造在总体极限下保持先验的均值和协方差。GMVIP通过从先验中抽取一个函数，并在一组诱导输入点上施加校正来构造后验样本。该校正在远离诱导输入处的效果直接由先验样本决定，使后验能够保留原始隐式过程的结构与变异性。该（代理）……（原文摘要在此处截断）

    arXiv:2610.07938v1 Announce Type: new  Abstract: Implicit-process priors specify distributions over functions through sample-forward mechanisms such as Bayesian neural networks and stochastic simulators, but their function-space densities are typically unavailable. We introduce Generalized Matheron Variational Implicit Processes (GMVIP), a pathwise variational family for posterior inference with such priors. For Gaussian-process priors, GMVIP recovers the standard inducing-variable variational GP construction; for general implicit priors, its empirical covariance construction preserves the prior mean and covariance in the population limit. GMVIP constructs posterior samples by drawing a function from the prior and applying a correction anchored at a set of inducing inputs. The effect of this correction away from the inducing inputs is determined directly from prior samples, allowing the posterior to retain the structure and variability of the original implicit process. The (surrogate) 
    
[^116]: 重新审视深度强化学习中用于平滑控制的时间正则化

    Revisiting Temporal Regularization for Smooth Control in Deep Reinforcement Learning

    [https://arxiv.org/abs/2610.07910](https://arxiv.org/abs/2610.07910)

    本文证明了时间正则化能约束共享相同下一状态的邻近状态之间的动作差异，从而在观测噪声下提供空间平滑性，并据此提出了结合线性斜坡上升时间惩罚的CATS方法，在不损害任务性能的前提下实现机器人的平滑控制。

    

    深度强化学习策略可能产生非平滑的动作振荡，这阻碍了其在物理机器人上的部署。现有的基于架构和惩罚的方法通过直接降低对状态输入变化的敏感性来追求空间平滑性，但随着平滑程度的加强，其宽泛的约束可能会损害任务性能。时间正则化则约束沿观测转移的动作差异，但一直被认为无法提供观测噪声下所需的空间平滑性。我们通过证明时间惩罚能够约束共享相同下一状态的当前状态之间的期望动作差异，来重新审视这一假设，从而揭示出一种在经验上可延伸至空间平滑性的空间效应。基于这一发现，我们提出了仅使用时间平滑性的动作调节方法（CATS），该方法将时间惩罚与线性斜坡上升相结合。我们强调了时间正则化

    arXiv:2610.07910v1 Announce Type: new  Abstract: Deep Reinforcement Learning policies can produce nonsmooth action oscillations that hinder deployment on physical robots. Existing architectural and penalty-based approaches seek spatial smoothness by directly reducing sensitivity to changes in state inputs, but their broad constraints can degrade task performance as stronger smoothing is pursued. Temporal regularization instead constrains action differences along observed transitions, but has been considered unable to provide the spatial smoothness needed under observation noise. We revisit this assumption by proving that the temporal penalty bounds the expected action differences between current states sharing a next state, revealing a spatial effect that empirically extends to spatial smoothness. Building on this finding, we propose Conditioning for Action using only Temporal Smoothness (CATS), which combines a temporal penalty with linear ramp-up. We highlight temporal regularization
    
[^117]: 各向同性却不可解码：潜在预测文本表示中的序列内容充分性鸿沟

    Isotropic Yet Undecodable: The Sequential Content-Sufficiency Gap in Latent-Predictive Text Representations

    [https://arxiv.org/abs/2610.07906](https://arxiv.org/abs/2610.07906)

    论文通过信息论分解揭示序列内容充分性鸿沟，证明潜在表示的各向同性与一致性无法保证有序目标信息可解码，并据此提出引入规范词元监督的非自回归框架CANOPE，将位置信息恢复率从13.5%大幅提升至98.8%。

    

    我们通过考察一个表示是否保留了其输入中可获得的有序目标信息，来研究序列内容充分性问题。一种信息论分解方法将输入歧义、表示损失和读出失配三者分离开来。我们构造了可恢复的视图，其中完美一致性与联合各向同性高斯性可以与零目标信息共存，并确立了确定性规范锚点所施加的限制。词元对数损失提供了单边的信息损失界；固定惩罚的岭回归分析说明了为什么仅凭秩无法确定预测风险。这些结果催生了CANOPE——一个具有有序潜在画布、规范词元监督和几何正则化的非自回归框架。在40,000条验证序列上，潜在一致性（PL0）与词元锚定（PL2）具有几乎相同的合并秩，但在强自然损坏下分别仅达到13.5%和98.8%的位置Recall@1。

    arXiv:2610.07906v1 Announce Type: new  Abstract: We study sequential content sufficiency by investigating whether a representation retains the ordered target information available in its input. An information-theoretic decomposition separates input ambiguity, representation loss, and readout mismatch. We construct recoverable views where perfect agreement and joint isotropic Gaussianity coexist with zero target information, and establish limits imposed by deterministic canonical anchors. Token log-loss provides a one-sided information-loss bound; a fixed-penalty ridge analysis shows why rank alone cannot determine prediction risk. These results motivate CANOPE, a nonautoregressive framework with ordered latent canvases, canonical-token supervision, and geometric regularization. On 40,000 validation sequences, latent-agreement (PL0) and token-grounded (PL2) have nearly identical pooled ranks but reach 13.5% and 98.8% positional Recall@1, respectively, under strong natural corruption whe
    
[^118]: ApexQuant：基于残差再各向同性化的无数据弹性量化

    ApexQuant: Data-Free Elastic Quantization by Residual Re-Isotropization

    [https://arxiv.org/abs/2610.07904](https://arxiv.org/abs/2610.07904)

    ApexQuant提出了一种无需校准数据的弹性量化方法，通过随机旋转将残差恢复到各向同性均匀分布并递归再量化，从而在读取权重前即可确定所需量化次数，使单一模型工件能以多种精度服务，尤其适用于数据受限制的地球观测和医学等领域。

    

    我们提出了ApexQuant，一种无需校准的量化方法，它递归地对残差误差进行重新量化，作为现有量化器之上的一个精炼层。我们证明了每次新的随机旋转都能将残差恢复为超球面上的均匀分布，这刻画了逐次传递中误差渐进衰减的速率。这一结果使我们能够在读取任何权重之前，就确定某一层为达到目标权重空间误差所需的传递次数。每个前缀本身都是一个有效的低比特率模型，因此单一模型工件可以服务于多种精度。我们用三个可互换的阶段——标量、$E_8$和格基——来实例化ApexQuant，并在四个开源权重的大语言模型以及地球观测和医学领域上进行了验证；在这些领域中，由于图像受严格许可证限制或患者材料受隐私约束，分布内数据往往难以获得。渐进的再各向同性化在几次（摘要在此处截断）

    arXiv:2610.07904v1 Announce Type: new  Abstract: We introduce ApexQuant, a calibration-free quantization method that recursively re-quantizes the residual error, serving as a refinement layer on top of existing quantizers. We establish that a fresh random rotation returns each residual to the uniform distribution on the hypersphere, which characterizes the rate of progressive error decay across successive passes. This result lets us determine, before any weight is read, how many passes a layer needs for a target weight-space error. Every prefix is itself a valid lower-rate model, so one artifact serves several precisions. We instantiate ApexQuant with three interchangeable stages, scalar, $E_8$ and trellis, and validate it on four open-weight LLMs and on Earth-observation and medical domains where in-distribution data is often unattainable as imagery arrives under restrictive licences or due to patient material under privacy constraints. Progressive re-isotropization comes within a few
    
[^119]: 面向稀疏长时程环境的方差厌恶 n 步离线强化学习

    Variance-Averse $n$-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments

    [https://arxiv.org/abs/2610.07899](https://arxiv.org/abs/2610.07899)

    提出VAN-Flow框架，通过结合分类分布式评论家、方差厌恶期望算子和拒绝采样引导的流匹配生成式演员，在生成式离线强化学习中选择高回报且低方差 dispersion 的可靠动作。

    

    生成式演员正在变革离线强化学习（RL），它使策略类能够富有表现力地建模复杂的动作分布。然而，这种表现力也暴露了异构数据集中的一个关键挑战：生成式策略可能会复现不可靠的动作模式，其回报分布表现出高方差，偶尔因运气而获得高回报，但缺乏一致性。因此，仅最大化期望 Q 值不足以识别可靠的动作。我们提出了 VAN-Flow（Variance-Averse n-step Flow，方差厌恶 n 步流），这是一个在生成式离线强化学习中促进可靠动作的框架。VAN-Flow 结合了：(i) 分类分布式评论家，(ii) 方差厌恶期望算子，该算子平滑地重新加权原子概率，以偏好既具有高回报又具有低离散度的动作，以及 (iii) 通过拒绝采样引导的流匹配生成式演员。不同于 CVaR 或均值-方差方法……

    arXiv:2610.07899v1 Announce Type: cross  Abstract: Generative actors are transforming offline reinforcement learning (RL) by enabling expressive policy classes that model complex action distributions. However, this expressiveness also exposes a key challenge in heterogeneous datasets: generative policies can reproduce unreliable action modes whose return distributions exhibit high variance, occasionally yielding high returns by chance but lacking consistency. Consequently, maximizing the expected $Q$-value alone is insufficient for identifying reliable actions. We propose VAN-Flow (Variance-Averse $n$-step Flow), a framework that promotes reliable actions in generative offline RL. VAN-Flow combines (i) a categorical distributional critic, (ii) a variance-averse expectation operator that smoothly reweights atom probabilities to favor actions with both high returns and low dispersion, and (iii) a flow-matching generative actor guided via rejection sampling. Unlike CVaR or mean-variance o
    
[^120]: FC-SWE：面向长时程软件工程智能体的失败条件强化学习

    FC-SWE: Failure-Conditioned RL for Long-Horizon Software Engineering Agents

    [https://arxiv.org/abs/2610.07898](https://arxiv.org/abs/2610.07898)

    论文提出FC-SWE框架，通过将失败补丁的验证器反馈作为条件上下文，将恢复尝试纳入强化学习策略训练，从而提升长时程软件工程智能体从失败中学习的能力。

    

    仓库级软件工程（SWE）是一个具有挑战性的长时程任务场景：智能体需要在长时间的交互中进行推理、使用工具，并适应有状态的环境。近期的研究工作使用组相对策略优化（GRPO）等强化学习方法来训练SWE智能体，该方法针对每个问题独立采样多条轨迹，测试生成的补丁，并在固定组内比较终端奖励。然而，这种训练设置并未将失败补丁的验证器反馈作为后续尝试的上下文加以复用，尽管这些反馈包含了关于出错原因的宝贵诊断信息。在恢复轨迹上进行训练具有挑战性，因为前一次的结果决定了下一条轨迹是否会被生成，而失败的执行则决定了其条件上下文。我们提出了FC-SWE，这是一个将恢复尝试纳入策略训练的失败条件强化学习框架。

    arXiv:2610.07898v1 Announce Type: new  Abstract: Repository-level software engineering (SWE) is a challenging long-horizon setting: agents must reason over extended interactions, use tools, and adapt to stateful environments. Recent work trains SWE agents with reinforcement learning methods such as Group Relative Policy Optimization (GRPO), which independently sample multiple trajectories per issue, test the resulting patches, and compare terminal rewards within a fixed group. However, this training setup does not reuse verifier feedback from failed patches as context for subsequent attempts, even though this feedback contains valuable diagnostic information about what went wrong. Training on recovery trajectories is challenging because the preceding outcome determines whether the next trajectory is generated, while the failed execution determines its conditioning context. We introduce FC-SWE, a failure-conditioned RL framework that incorporates recovery attempts into policy training. 
    
[^121]: 重新思考大语言模型的忠实性：一种成对上下文敏感的视角

    Rethinking Faithfulness in LLMs: A Pairwise Context-Sensitive Perspective

    [https://arxiv.org/abs/2610.07894](https://arxiv.org/abs/2610.07894)

    提出成对忠实性基准PFaithBench，通过对比同一问题在支持性与非支持性上下文下的表现，评估大语言模型能否正确地在回答与拒答之间切换，揭示忠实性本质上取决于回答与拒答之间的权衡。

    

    大语言模型（LLMs）被期望基于所提供的上下文忠实地回答问题，并在上下文信息不足以回答问题时选择拒答。现有的忠实性评估通常孤立地评估每个问题-上下文实例；然而，这种实例级评估未能捕捉忠实行为的一个基本要求：模型响应能够随可用上下文变化而调整的能力。具体而言，当存在充分证据时，模型应给出正确答案；当证据不足时，模型应选择拒答。在这项工作中，我们提出了一个成对忠实性基准，评估模型在支持性上下文与非支持性上下文下，能否针对同一问题在回答与拒答之间进行切换。我们对来自七个模型系列的三十九个模型的评估表明，忠实性在根本上涉及回答与拒答之间的权衡……

    arXiv:2610.07894v1 Announce Type: new  Abstract: Large language models (LLMs) are expected to answer questions faithfully based on the provided context, abstaining when the context information is insufficient to answer the questions. Existing faithfulness evaluations typically assess each question-context instance in isolation; however, such instance-level evaluation fails to capture a fundamental requirement of faithful behavior: the ability to adapt model responses to changes in available contexts. In particular, a model should provide correct answers when sufficient evidence is present and abstain when it is not. In this work, we propose a Pairwise Faithfulness Benchmark (PFaithBench) that evaluates whether a model can switch between answering and abstaining for the same question under supporting versus non-supporting contexts. Our evaluations across thirty-nine models with seven model families demonstrate that faithfulness fundamentally involves a trade-off between answering and ab
    
[^122]: 标签高效的心电波形分界深度学习方法：与广泛使用分界工具的多数据集基准对比

    Label-Efficient Deep Learning for ECG Delineation: A Multi-Dataset Benchmark against Widely Used Delineation Tools

    [https://arxiv.org/abs/2610.07885](https://arxiv.org/abs/2610.07885)

    该研究通过多数据集基准测试证明，自监督预训练配合恰当的微调策略可显著降低心电波形分界对专家标注的依赖，且所得深度模型的分界性能优于广泛使用的开源分界工具。

    

    心电图（ECG）波形分界，即识别波形边界，是将原始心电信号转化为临床可解释测量的基础步骤。深度学习已推动这一任务的发展，但仍然依赖于昂贵的专家标注。自监督预训练和半监督学习等标签高效策略有望减轻这一负担，然而它们能否产生可靠的分界结果，以及由此训练出的深度模型是否优于实践中使用的分界工具，目前仍不清楚。我们分两个阶段回答这一问题。首先，在一个内部数据集和四个外部数据集上比较自监督目标与有监督或半监督微调，我们发现预训练有帮助，但目标函数的选择至关重要，且半监督微调的价值取决于预训练目标。其次，我们将选定的深度学习模型与广泛使用的开源分界工具进行基准对比。

    arXiv:2610.07885v1 Announce Type: cross  Abstract: Electrocardiogram (ECG) delineation, the identification of waveform boundaries, is a foundational step that translates raw ECG signals into clinically interpretable measurements. Deep learning has advanced this task but remains dependent on costly expert annotations. Label-efficient strategies such as self-supervised pretraining and semi-supervised learning are expected to ease this burden, yet it remains unclear whether they yield reliable delineation and whether the deep models they produce outperform the delineation tools used in practice. We address this in two stages. First, comparing self-supervised objectives with supervised or semi-supervised fine-tuning across one internal and four external datasets, we find that pretraining helps but the objective matters, and that the value of semi-supervised fine-tuning depends on the pretraining objective. Second, we benchmark the selected deep learning model against widely used open-sourc
    
[^123]: 学习型自适应多分辨率扩散成像

    Learned Adaptive Multiresolution Diffusion Imaging

    [https://arxiv.org/abs/2610.07884](https://arxiv.org/abs/2610.07884)

    该论文提出Learned AMDI，用近端策略优化训练的共享局部策略替代AMDI中固定准则的树细化选择器，在保留固定树传播器与层次约束的前提下，使留出测试中的平均终端参考偏差从0.17496降至0.13657、占用率从0.13737提升至0.26660。

    

    自适应多分辨率方法通过将细尺度自由度集中于所需之处来降低表示成本，但其树更新通常由固定的局部准则支配。我们提出了学习型自适应多分辨率扩散成像（Learned AMDI），该方法在保留AMDI固定树传播器与层次约束的同时，将近传播后的选择器替换为由近端策略优化（proximal policy optimization）训练的共享局部策略。回归测试表明，在使用相同树的情况下，该方法能够以机器精度复现确定性的AMDI轨迹。在本文研究的Haar实现中，确定性单步选择器在54次决策中均未接受任何细化操作。在九个留出测试案例中，Learned AMDI共执行了393次细化，将平均终端参考偏差从0.17496降至0.13657，同时占用率从0.13737上升至0.26660。逐步诊断显示偶尔会出现较小的自适应能……（摘要原文在此处截断）

    arXiv:2610.07884v1 Announce Type: cross  Abstract: Adaptive multiresolution methods reduce representation cost by concentrating fine-scale degrees of freedom where needed, but their tree updates are usually governed by fixed local criteria. We introduce Learned Adaptive Multiresolution Diffusion Imaging (Learned AMDI), which preserves the AMDI fixed-tree propagator and hierarchy constraints while replacing the post-propagation selector with a shared local policy trained by proximal policy optimization. Regression tests reproduce deterministic AMDI trajectories to machine precision when identical trees are used. In the Haar implementation studied here, the deterministic one-step selector accepts no refinements in 54 decisions. Across nine held-out cases, Learned AMDI executes 393 refinements and reduces the mean terminal reference discrepancy from $0.17496$ to $0.13657$, while occupancy rises from $0.13737$ to $0.26660$. Step-resolved diagnostics reveal occasional small adaptation-energ
    
[^124]: 基于负策略采样的在线策略蒸馏

    On-Policy Distillation with Negative-Policy Rollouts

    [https://arxiv.org/abs/2610.07874](https://arxiv.org/abs/2610.07874)

    提出负策略在线策略蒸馏（NP-OPD），在采样阶段引入性能较弱的负策略作为负向参考来补充教师监督，从而在教师与学生分布重叠有限、正向引导信号不足时提供更充分的学习信号。

    

    在线策略蒸馏（On-Policy Distillation, OPD）作为一种后训练方法已被广泛研究，其中学生模型在自己生成的序列上从更强的教师模型获得 token 级别的监督。近期研究通过替代性的蒸馏奖励公式和教师配置对 OPD 进行了改进，但蒸馏目标仍以模仿教师为核心。然而，当更强的教师模型与学生模型的分布重叠有限时，这种正向引导所能提供的学习信号可能不足。在本工作中，我们提出了负策略在线策略蒸馏（Negative-Policy OPD, NP-OPD），它通过引入一个性能更低、能力更弱的负策略所生成的采样序列来补充教师监督，该负策略作为学生模型的负向参考。NP-OPD 并不修改蒸馏奖励公式，而是在采样阶段引入负策略，持续提供负策略相对于教师模型更偏好的 token，从而……

    arXiv:2610.07874v1 Announce Type: new  Abstract: On-policy distillation (OPD) has been widely studied as a post-training method in which a student model obtains token-level supervision from a stronger teacher on its own rollouts. Recent studies have improved OPD through alternative distillation reward formulations and teacher configurations, while the objective of distillation remains centered on mimicking the teacher. However, when a stronger teacher has limited distributional overlap with the student, such positive guidance can provide insufficient learning signals. In this work, we introduce Negative-Policy OPD (NP-OPD), which complements teacher supervision with rollouts from a lower-performing, lower-capability negative policy that serves as a negative reference for the student. Rather than modifying the distillation reward formulation, NP-OPD introduces the negative policy at the rollout stage, continuously supplying tokens preferred by the negative policy over the teacher so tha
    
[^125]: ReFold：面向长程智能体的免训练可逆轮间上下文折叠方法

    ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents

    [https://arxiv.org/abs/2610.07863](https://arxiv.org/abs/2610.07863)

    ReFold提出了一种免训练的可逆上下文折叠渲染层，通过将已展示内容替换为占位符、将智能体报告完成的轮次折叠为一行注释来消除轮间冗余，从而在保留底层完整交互历史的同时压缩长程智能体的渲染上下文，避免了现有预测性方法带来的运行时开销、前缀缓存失效和不可逆信息丢失。

    

    长程LLM智能体基于只追加（append-only）的交互历史进行行动，该历史在每一步都会被重新发送给模型，因此上下文及其成本随步骤不断增长，直到会话超出上下文窗口。现有方法通过上下文需求预测来管理上下文，依赖于额外的模型调用、启发式规则或训练得到的策略。然而，这些预测性方法会引入运行时开销、使前缀缓存失效，并永久丢弃内容且无法保证恢复。为克服这些限制，我们提出了ReFold：一个免训练的渲染层，它在保留底层交互历史的同时仅压缩模型的渲染上下文。它无需辅助预测器即可消除两类轮间冗余：较早轮次已经展示过的内容（用占位符替换），以及智能体自身报告已完成的轮次（折叠为一行注释）。两种操作均使用分块渲染与重写（摘要在此处截断）……

    arXiv:2610.07863v1 Announce Type: cross  Abstract: Long-horizon LLM agents act on an append-only interaction history that is re-sent to the model at every step, so the context and its cost grow with steps until the sessions exceed the context window. Existing methods manage the context through context requirement prediction, relying on additional model calls, heuristic rules, or trained policies. However, these predictive approaches introduce runtime overhead, invalidate prefix caches, and permanently discard content with no guarantee of recovery. To overcome these limitations, we introduce ReFold: a training-free rendering layer that preserves the underlying interaction history while compressing only the model's rendered context. It removes two kinds of inter-turn redundancy without an auxiliary predictor: content an earlier turn already displayed, replaced by a stub, and turns the agent itself reports finished, folded into a one-line note. Both operators use chunked rendering, rewrit
    
[^126]: 一种面向X射线衍射的自学习科学智能体

    A self-learning scientific agent for X-ray diffraction

    [https://arxiv.org/abs/2610.07862](https://arxiv.org/abs/2610.07862)

    本文提出“干将”——一个面向粉末X射线衍射的自学习科学智能体，它通过诊断失败并自主修订和验证技能指令与代码，将分析经验转化为可复用的可执行技能，且无需重新训练语言模型，在多个精修平台上超越了专家设计的技能。

    

    科学智能体面临的一个核心挑战，是将分析经验转化为以物理证据为基础的可复用专业知识。本文介绍了“干将”，这是一个面向粉末X射线衍射的自学习智能体，它构建于我们开发的衍射分析生态系统之上：XMatcher、XQueryer、XDecomposer和WPEM。这些引擎共同覆盖了物相鉴定、多相分解以及物理约束的全谱建模。“干将”通过诊断失败、修订技能指令和代码，并在复用前对修订内容进行验证，将分析经验转化为可执行的技能，而无需重新训练语言模型或更改底层物理模型。基于开发数据筛选并在留出集评估前冻结的技能，在FullProf、GSAS-II和PyWPEM上均取得了比原始专家设计技能更高的精修分数。该智能体能够解析强烈重叠的衍射峰，并对五相混合物进行定量分析（摘要在此处截断）。

    arXiv:2610.07862v1 Announce Type: cross  Abstract: A central challenge for scientific agents is to turn analytical experience into reusable expertise grounded in physical evidence. Here we introduce Gan Jiang, a self-learning agent for powder X-ray diffraction built on a diffraction-analysis ecosystem we developed: XMatcher, XQueryer, XDecomposer and WPEM. Together, these engines span phase identification, multiphase decomposition and physics-constrained whole-pattern modelling. Gan Jiang converts analytical experience into executable skills by diagnosing failures, revising skill instructions and code, and validating revisions before reuse, without retraining the language model or changing the underlying physical models. Skills selected using development data and frozen before held-out evaluation achieve higher refinement scores than the original expert-designed skills across FullProf, GSAS-II and PyWPEM. The agent resolves strongly overlapping reflections, quantifies a five-phase anci
    
[^127]: Tram-FL：通过去中心化联邦学习中的顺序模型循环降低通信与计算成本

    Tram-FL: Reducing Communication and Computation Costs through Sequential Model Circulation in Decentralized Federated Learning

    [https://arxiv.org/abs/2610.07859](https://arxiv.org/abs/2610.07859)

    提出Tram-FL机制，通过在节点间顺序循环传递单个模型进行训练，以最小的计算和通信成本实现去中心化联邦学习。

    

    传统的去中心化联邦学习（DFL）通常以客户端为中心，每个客户端维护一个模型副本，单独执行更新，并进行模型交换与整合。虽然充分利用计算资源可以缩短训练时间，但也可能导致大量的计算和通信浪费。这在非独立同分布数据的情况下尤为明显，因为要达到高模型精度需要额外的资源。本研究将重点转向模型本身，旨在以最小的计算和通信成本实现去中心化联邦学习。为此，我们提出了Tram-FL（去中心化联邦学习的旅行模型训练机制），这是一种旨在高效应对这些挑战的机制。它通过在节点之间循环传递单个模型来进行顺序训练。我们解决了基于模型循环训练中的训练调度问题……

    arXiv:2610.07859v1 Announce Type: new  Abstract: Conventional decentralized federated learning (DFL) often focuses on clients, with each client maintaining a model copy, performing updates individually, and undertaking model exchange and integration. While fully leveraging computational resources can shorten training times, it can also lead to significant computational and communication waste. This is especially pronounced with non-independent and identically distributed (non-IID) data, where achieving high model accuracy demands extra resources. This research shifts focus to the model itself, aiming to realize DFL with minimal computation and communication costs. To this end, we propose Tram-FL (Traveling Model Training Mechanism for Decentralized Federated Learning), a mechanism designed to efficiently address these challenges. It sequentially trains a single model by circulating it among nodes. We address the training scheduling problem in model circulation-based training, specifica
    
[^128]: 基于车辆轨迹的个性化路径重现的决策聚焦神经优化框架

    A Decision-Focused Neural Optimization Framework for Personalized Route Reproduction from Vehicle Trajectories

    [https://arxiv.org/abs/2610.07857](https://arxiv.org/abs/2610.07857)

    提出了一种决策聚焦神经优化框架，将个性化路径重现建模为基于学习到的驾驶员特定潜在路段成本的最短路径问题，通过感知模型将上下文协变量嵌入个性化成本，并借助约束优化层与隐式极大似然估计实现端到端训练，无需枚举备选路径集合即可重现观测路径。

    

    本研究将个体路径重现问题表述为在学习到的驾驶员特定潜在路段成本上的最短路径问题。其核心思想是，一旦从上下文信息中推断出这些潜在成本，就无需枚举备选路径集合即可重现观测到的路径。我们提出了一个神经流水线，其中包含一个感知模型，该模型将上下文协变量——包括个体特征、出行特定属性和网络级交通状态——嵌入到个性化的路段成本中。感知编码器之后是一个约束优化（CO）层，该层基于这些估计的成本来确定最短路径（SP）。为实现端到端训练，我们采用决策聚焦学习，使预测的最短路径与观测路径对齐。隐式极大似然估计（iMLE）为包含不可微CO层的损失函数提供了近似梯度。

    arXiv:2610.07857v1 Announce Type: new  Abstract: This study formulates individual route reproduction as a shortest-path problem over learned driver-specific latent link costs. The central idea is that, once such latent costs are inferred from contextual information, observed routes can be reproduced without enumerating alternative route sets. We propose a neural pipeline that includes a perception model that embeds context covariates, which comprises individual characteristics, trip-specific attributes, and network-level traffic states, into the personalized link costs. A constrained optimization (CO) layer, which determines the shortest path (SP) based on these estimated costs, follows the perception encoder. To enable end-to-end training, we employ decision-focused learning to align the predicted shortest paths with observed routes. The implicit maximum likelihood estimation (iMLE) provides an approximate gradient of the loss function that contains the non-differentiable CO layer. Fu
    
[^129]: 迷失于bf16类型转换：导出三值语言模型可能抵消大部分低学习率代码改动的效果

    Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes

    [https://arxiv.org/abs/2610.07853](https://arxiv.org/abs/2610.07853)

    该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。

    

    BitNet b1.58、Falcon-E和BitCPM等三值语言模型先以更高精度的潜在权重进行微调，再通过导出步骤生成三值代码进行部署；在三个实验室公开记录的流程中，该导出步骤均会先将潜在权重转换为bf16。我们对三个实验室的这些流程进行了审计。在已发布的检查点中，对随附潜在权重执行fp32量化所得的结果与实际部署代码在Falcon-E和BitCPM上有0.83%–1.77%的代码不一致，在BitNet 2B-4T上有1.530%不一致；对于Falcon-E和BitCPM，绝大多数不一致源于这样的乘积：其bf16舍入恰好落在阈值上，并被“舍入到偶数”（ties-to-even）规则映射为零，而未经修改的onebitllms导出器可以逐字节复现全部四个Falcon-E发布版本。在微调终点，当学习率按标称的学习率与bf16-ULP之比选取时，按文档所述流程导出会使Falcon-E-1B-Base的贪婪解码GSM8K严格准确率从58.79%跌至0.78%，使BitCPM-CANN-（原文摘要在此处截断）的准确率从36.13%跌至0.39%。

    arXiv:2610.07853v1 Announce Type: cross  Abstract: Ternary language models such as BitNet b1.58, Falcon-E and BitCPM are fine-tuned with higher-precision latent weights and deployed as ternary codes produced by an export step that, in the labs' documented pipelines, first casts the latents to bf16. We audit those pipelines across three labs. In released checkpoints, fp32 quantization of the shipped latents disagrees with the deployed codes on 0.83-1.77% of codes in Falcon-E and BitCPM and on 1.530% in BitNet 2B-4T; for Falcon-E and BitCPM most disagreements are products that bf16 rounding lands exactly on the threshold, which ties-to-even maps to zero, and the unmodified onebitllms exporter reproduces all four Falcon-E releases byte for byte. At fine-tuned endpoints, with learning rates selected to match a nominal learning-rate-to-bf16-ULP ratio, the documented export lowers greedy GSM8K strict accuracy from 58.79% to 0.78% for Falcon-E-1B-Base and from 36.13% to 0.39% for BitCPM-CANN-
    
[^130]: Scen-Opt：一个面向数据驱动凸编程的场景优化工具箱

    Scen-Opt: A Scenario Optimization Toolbox for Data-Driven Convex Programming

    [https://arxiv.org/abs/2610.07846](https://arxiv.org/abs/2610.07846)

    本文推出了开源工具箱Scen-Opt，它将凸编程与数据样本相结合并基于场景理论提供统计保证，支持数据驱动的线性、二次和半定规划，填补了场景方法框架下用户友好的数据驱动凸优化软件工具的空白。

    

    场景方法是一种成熟的数据驱动决策统计框架。特别是在数据驱动优化中，场景方法揭示了问题结构如何支配样本外泛化能力，并为根据约束满足情况评估和认证最优解的可靠性提供了有原则的基础。尽管其理论发展成熟且适用性广泛，但迄今为止还没有软件工具箱能够在场景方法框架内实现用户友好的数据驱动凸优化。在本文中，我们介绍了Scen-Opt，这是一个开源软件工具，它将凸编程与数据样本相结合，同时提供基于场景理论的统计保证。Scen-Opt使用Python实现，支持数据驱动的线性规划、二次规划和半定规划，并提供了一个基于Python的Web应用程序，具有直观且响应迅速的界面。

    arXiv:2610.07846v1 Announce Type: cross  Abstract: The scenario approach is a well-established statistical framework for data-driven decision-making. In particular, in data-driven optimization, the scenario approach unveils how the problem structure governs out-of-sample generalization, and offers a principled basis for assessing and certifying the reliability of the optimal solution as per constraint satisfaction. Despite its strong theoretical development and wide applicability, no software toolbox has been available to date that enables user-friendly, data-driven convex optimization within the scenario-approach framework. In this paper, we introduce Scen-Opt, an open-source software tool that integrates convex programming with data samples while providing statistical guarantees grounded in scenario theory. Scen-Opt is implemented in Python, supporting data-driven linear, quadratic, and semidefinite programming, and offers a Python-based web application with an intuitive and reactive
    
[^131]: CHARTER：计算病理学中分层紧凑证据评估的参考替换审计

    CHARTER: Auditing Reference Substitution in Hierarchical Compact-Evidence Evaluation for Computational Pathology

    [https://arxiv.org/abs/2610.07843](https://arxiv.org/abs/2610.07843)

    该论文提出CHARTER这一参考感知的评估章程，用于审计计算病理学中分层紧凑证据评估因替换评估参考而导致保真度测量与策略比较结论发生反转的问题，在其五种子Random-K审计的15项比较中发现4项出现确定的结论反转。

    

    在数字病理学中，紧凑证据常被用于解释或审计全切片图像多实例学习模型所做的预测。在分层紧凑证据流程中，候选过滤会在原始的全袋预测之外引入一个特定于策略的候选条件预测。然而，如果评估参考发生改变而预期目标仍保持为原始的全袋预测，不仅同一紧凑证据所测得的保真度会发生变化，竞争性候选策略之间的比较结论也可能发生变化。为使这种依赖关系显式化，我们提出了CHARTER——一个参考感知的评估章程，要求研究者DECLARE（声明）预期目标和参考，QUANTIFY（量化）候选引起的预测偏移，并AUDIT（审计）比较结论的稳定性。在我们基于五个随机种子进行的Random-K主要审计的15项比较中，有4项出现了确定的结论反转；在匹配的原生……

    arXiv:2610.07843v1 Announce Type: cross  Abstract: In digital pathology, compact evidence is often used to explain or audit predictions made by whole-slide image multiple instance learning models. In hierarchical compact-evidence pipelines, candidate filtering introduces a strategy-specific candidate-conditioned prediction alongside the original full-bag prediction. If the evaluation reference changes while the intended target remains the original full-bag prediction, however, not only can the measured fidelity of the same compact evidence change, but comparisons between competing candidate strategies can also change. To make this dependence explicit, we introduce CHARTER, a reference-aware evaluation charter that asks researchers to DECLARE the intended target and reference, QUANTIFY candidate-induced prediction shift, and AUDIT the stability of comparative conclusions. Across the 15 comparisons in our main five-seed Random-K audit, 4 showed determinate reversals; in a matched native-
    
[^132]: 特权上下文作为在线策略自蒸馏中的漂移

    Privileged Context as Drift in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.07842](https://arxiv.org/abs/2610.07842)

    本文通过在内容与来源两个维度上系统控制特权上下文的设计，首次分离并量化了特权上下文选择对在线策略自蒸馏中策略漂移的影响，发现改变上下文内容比改变其来源引起更大的KL散度漂移。

    

    在线策略自蒸馏（OPSD）训练一个语言模型去匹配在特权上下文条件下的自身副本。现有工作在改变特权上下文所包含内容及其生成方式的同时，也改变了模型、数据和训练设置，导致特权上下文设计的影响难以被分离出来。受持续学习中减少灾难性遗忘的研究启发，我们研究了特权上下文的选择如何影响策略漂移。具体而言，我们在两个维度上进行变化：内容（演示示例、反馈或改写）和来源（外部生成、带验证器的自生成、或不带验证器的自生成）。我们在这九种组合和三个数据集上使用OPSD训练Qwen2.5-7B模型，测量目标任务准确率、先前任务保持率、相对于基础策略的反向KL散度以及参数更新的几何特性。在固定来源的情况下，改变内容所产生的中位KL散度范围比固定内容而改变来源的情况更宽。

    arXiv:2610.07842v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) trains a language model to match a copy of itself conditioned on privileged context. Existing work varies what privileged context contains and how it is produced while also changing models, data, and training setups, making the effects of privileged context design difficult to isolate. Motivated by efforts in continual learning to reduce catastrophic forgetting, we study how the choice of privileged context affects policy drift. Specifically, we vary two axes: content (a demonstration, feedback, or rephrase) and source (external, self-generated with a verifier, or self-generated without a verifier). We train Qwen2.5-7B with OPSD across these nine combinations and three datasets, measuring target-task accuracy, prior-task retention, reverse KL from the base policy, and parameter-update geometry. Holding source fixed, changing content spans a wider median KL range than holding content fixed and changing s
    
[^133]: 检索是不够的：为冻结的时间序列预测器刷新记忆

    Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters

    [https://arxiv.org/abs/2610.07834](https://arxiv.org/abs/2610.07834)

    提出即插即用的FreshCast框架，通过持续用新观测数据刷新非参数记忆并进行校准，解决了检索增强时间序列预测中记忆陈旧导致检索效用下降的问题。

    

    检索增强的时间序列预测使用与当前上下文相似的历史片段的后续走势作为预测器的参考。大多数现有方法仅从训练片段中一次性构建检索记忆，导致部署之后新揭示的观测数据无法用作参考，且通常不校准检索到的信息应对冻结的预测器产生多大影响。我们识别出冻结预测器检索效用的两个关键决定因素：历史数据是否仍然反映当前状态，以及其所引发的修正是否与预测器的残差误差对齐——当记忆变得陈旧时，这种对齐可能在验证阶段与部署阶段之间发生偏移。我们提出FreshCast，一个即插即用的检索框架，它保持预测器冻结，用新观测数据持续更新非参数记忆，通过关系核回归形成记忆预测，并校准……（摘要在此处截断）

    arXiv:2610.07834v1 Announce Type: new  Abstract: Retrieval-augmented time-series forecasting uses the continuations of historical segments similar to the current context as references for a forecaster. Most existing methods build the retrieval memory once from the training segment, leaving observations revealed after deployment unavailable as references, and generally do not calibrate how much the retrieved information should influence a frozen forecaster. We identify two key determinants of retrieval utility for a frozen forecaster: whether the history still reflects the current state, and whether the correction it induces aligns with the forecaster's residual errors, an alignment that can shift between validation and deployment when the memory becomes stale. We propose FreshCast, a plug-in retrieval framework that keeps the forecaster frozen, continuously updates a non-parametric memory with new observations, forms a memory forecast through relational kernel regression, and calibrate
    
[^134]: 预测准确率不等于交易利润：进化小型循环网络用于股票收益预测

    Forecast Accuracy Is Not Trading Profit: Evolving Small Recurrent Networks for Stock Return Prediction

    [https://arxiv.org/abs/2610.07825](https://arxiv.org/abs/2610.07825)

    通过神经进化架构搜索得到的小型循环网络在股票收益预测中同时取得了最高的预测准确率和最佳的每日多空策略净收益，证明更低的预测误差并不等于更高的交易利润。

    

    时间序列预测模型通常以逐点误差来比较，这种评价方式将预测与其所服务的决策割裂开来，而更低的预测误差并不意味着下游决策更优。与之并行的一场争论是：现代transformer架构是否比循环网络及其他轻量级模型预测得更好。我们将线性模型、固定循环网络、transformer以及基于混合（mixing）的架构与通过神经进化架构搜索进化出的循环网络进行比较，并从预测准确率和每日多空策略的净收益两个维度对每种模型进行评估。所有模型均在合并面板数据上拟合，即一个网络在整个股票全集上统一训练。在四个中盘股组合和三个交易年度的测试中，进化得到的循环网络在预测准确率和净交易表现上均排名第一，而准确率第二高的模型一旦建仓并计入交易成本后便出现亏损。这种优势与预测视野（摘要原文在此处截断）

    arXiv:2610.07825v1 Announce Type: cross  Abstract: Time series forecasting models are typically compared on pointwise error, which scores a prediction in isolation from the decision it is produced for, and a lower forecast error does not imply a better decision downstream. A parallel debate asks whether modern transformer architectures forecast better than recurrent and other lightweight models. We compare linear, fixed recurrent, transformer, and mixing based architectures against recurrent networks evolved by neuroevolutionary architecture search, evaluating each on forecast accuracy and on the net return of a daily long/short strategy. All models are fit on a pooled panel, one network trained across the whole universe. Across four mid-cap portfolios and three trading years, the evolved networks rank first on both forecast accuracy and net trading performance, while the second most accurate model loses money once positions are formed and costs are charged. The advantage tracks a hori
    
[^135]: CANDLE：面向无创脑源成像的皮层零空间分解

    CANDLE: Cortical Null-Space Decomposition for Noninvasive Brain Source Imaging

    [https://arxiv.org/abs/2610.07824](https://arxiv.org/abs/2610.07824)

    本文提出CANDLE，一种在T1加权MRI导出的源到传感器映射零空间上学习先验的模型，能够在受试者特定的皮层几何结构上实现可泛化的无创脑电生理源成像。

    

    电生理源成像（ESI）旨在从脑电图（EEG）等无创电生理测量中估计皮层源活动。然而，由于源活动的维度远高于传感器观测的维度，导致解不唯一，ESI本质上是一个病态问题。近年来基于学习的方法通过学习数据驱动的源先验来解决这种歧义，但它们往往难以在不同受试者特定的皮层几何结构之间进行泛化。为解决这一问题，我们提出了CANDLE，一种能够在受试者特定皮层几何结构上估计源活动的基于学习的ESI模型。CANDLE在由T1加权MRI推导的源到传感器映射所诱导的零空间上学习先验，将学习限制在不可观测的源分量上，同时保留几何约束。为训练CANDLE，我们开发了一个覆盖1,100多个受试者特定皮层的全脑模拟器……

    arXiv:2610.07824v1 Announce Type: new  Abstract: Electrophysiological source imaging (ESI) aims to estimate cortical source activity from noninvasive electrophysiological measurements such as electroencephalogram (EEG). However, ESI is fundamentally ill-posed because source activity is substantially higher-dimensional than sensor observations, resulting in non-unique solutions. Recent learning-based approaches address this ambiguity by learning data-driven source priors, yet they often struggle to generalize across subject-specific cortical geometries. To address this, we propose CANDLE, a learning-based ESI model that estimates source activity on subject-specific cortical geometries. CANDLE learns a prior over the null space induced by the source-to-sensor mapping derived from T1-weighted MRI, restricting learning to unobservable source components while preserving geometric constraints. To train CANDLE, we develop a whole-brain simulator spanning over 1,100 subject-specific cortical g
    
[^136]: TTNet：基于智能球拍的乒乓球运动员分析多任务深度学习模型

    TTNet: Multi-Task Deep Learning for Table Tennis Player Analysis with Smart Racket

    [https://arxiv.org/abs/2610.07823](https://arxiv.org/abs/2610.07823)

    本文提出TTNet，一种融合CNN、ResNet和自注意力机制的多任务深度学习模型，能够基于智能球拍的六轴传感器数据同时预测乒乓球运动员的性别、持拍手、球龄和技术水平四项属性。

    

    AI CUP 2025乒乓球智能球拍数据精确分析竞赛引入了可收集大量球员挥拍数据的智能乒乓球拍，使乒乓球大数据研究成为可能。这些数据支持对球员回球技术和挥拍力量一致性的深入分析，从而提高球员技能评估的准确性。本研究聚焦于智能乒乓球拍采集的六轴传感器数据，提出了TTNet——一种具有多任务学习能力的新型深度学习模型，以推进乒乓球数据分析及相关应用。TTNet结合了卷积神经网络（CNN）、残差网络（ResNet）和自注意力机制，可同时预测四项球员属性：性别、持拍手、球龄和技术水平。我们采用包含数据增强和任务特定损失函数的两阶段训练策略，以提高模型在不平衡数据上的泛化能力。

    arXiv:2610.07823v1 Announce Type: new  Abstract: The AI CUP 2025 Precise Analysis of Table Tennis Smart Racket Data Competition introduced smart table tennis rackets that collect extensive player swing data, enabling research on table tennis big data. These data support in-depth analysis of players' return techniques and swing-force consistency, improving the accuracy of player skill assessment. This study focuses on six-axis sensor data collected by smart table tennis rackets and proposes TTNet, a novel deep learning model with multitask learning capabilities, to advance table tennis data analysis and related applications. TTNet combines convolutional neural networks (CNNs), residual networks (ResNet), and self-attention mechanisms to simultaneously predict four player attributes: gender, playing hand, years of experience, and skill level. We adopt a two-stage training strategy that incorporates data augmentation and task-specific loss functions to improve generalization on imbalanced
    
[^137]: αTransfer：面向高效模型合并的系数迁移方法

    $\alpha$Transfer: Coefficient Transfer for Efficient Model Merging

    [https://arxiv.org/abs/2610.07819](https://arxiv.org/abs/2610.07819)

    提出αTransfer方法，利用同一模型家族内不同规模模型在合并系数上性能分布的高度一致性，在小代理模型上搜索最优系数后直接迁移到大模型，实现最高20倍加速和70%内存减少。

    

    模型合并通过参数运算将多个微调后的检查点合并为单一模型，是一种颇具前景的解决方案。然而，寻找最优合并系数需要进行大量搜索，随着模型在规模和数量上的增长，由于高内存需求和搜索空间的组合式膨胀，这一过程变得极其昂贵。我们发现，在同一模型家族内，不同规模的模型在合并系数上表现出高度一致的性能分布。这种分布上的相似性使我们能够提出一种实用的范式，我们称之为αTransfer：先在小型代理模型上搜索最优系数，然后直接将其迁移到更大的目标模型上。我们在多种合并方法、模型家族和任务上验证了αTransfer。实验结果表明，在视觉Transformer上实现了6倍的加速和70%的内存减少，以及20倍的加速。

    arXiv:2610.07819v1 Announce Type: cross  Abstract: Model merging offers a promising solution for combining multiple fine-tuned checkpoints into a single model through parameter arithmetic. However, finding optimal merging coefficients requires an extensive search that becomes prohibitively expensive as models scale in both size and number, due to high memory requirements and combinatorial growth in the search space. We show that, within the same model family, models exhibit highly congruent performance distributions over merging coefficients across different model sizes. This distributional similarity enables a practical paradigm we call \textit{$\alpha$Transfer}: searching for optimal coefficients on a small proxy model, then directly transfer them to larger target models. We verify $\alpha$Transfer across multiple merging methods, model families, and tasks. Experimental results demonstrate a 6$\times$ speedup and 70\% memory reduction on vision transformers, and a 20$\times$ speedup 
    
[^138]: 我需要云端吗？面向小型语言模型智能体的不确定性感知步骤级交接

    Do I Need the Cloud? Uncertainty-Aware Step-Level Handoff for Small Language Model Agents

    [https://arxiv.org/abs/2610.07816](https://arxiv.org/abs/2610.07816)

    提出STEPGATE框架，通过不确定性感知地对本地小模型的每个动作评分，并按需将困难步骤升级到更强的云端模型，从而在大幅降低云端调用比例的同时显著提升智能体的任务成功率。

    

    小型语言模型（SLM）作为本地智能体控制器极具吸引力，因为它们能减少远程推理、降低延迟并缩小部署占用，但结构化的工具错误可能导致智能体步骤执行失败。现有的路由器通常在每次查询时只做一次模型选择。然而，智能体包含一系列顺序决策点，其难度会随中间观测结果动态变化。我们提出了STEPGATE，一个不确定性感知的交接框架，它对每个本地SLM动作进行评分，并有选择地将具有挑战性的步骤升级到更强的模型。在一个基于BFCL、包含52个任务的留出单步测试集上，Qwen2.5-1.5B/7B模型对以30.8%的升级率取得了82.7%的任务成功率，而仅本地方案为67.3%，随机升级方案（升级率为33.8%）为75.4%。在一项独立的多轮评估中，STEPGATE仅使用30.0%的云端动作，就实现了69.0%的轨迹成功率和84.0%的动作成功率，相比之下，仅本地方案为48.0%/70.5%，随机升级方案为60.0%/78.2%……

    arXiv:2610.07816v1 Announce Type: new  Abstract: Small language models (SLMs) are attractive as local agent controllers because they reduce remote inference, latency, and deployment footprint, yet structured tool errors can cause an agent step to fail. Existing routers typically select a model once per query. However, agents expose sequential decision points whose difficulty dynamically changes based on intermediate observations. We propose STEPGATE, an uncertainty-aware handoff framework that scores each local SLM action and selectively escalates challenging steps to a stronger model. On a 52-task held-out single-step BFCL-derived test split, the Qwen2.5-1.5B/7B pair attains 82.7% task success with 30.8% escalation, versus 67.3% local-only and 75.4% random escalation (which uses 33.8% escalation). In a separate multi-turn evaluation, STEPGATE achieves 69.0% trajectory success and 84.0% action success using only 30.0% cloud actions, compared with 48.0%/70.5% local-only, 60.0%/78.2% ran
    
[^139]: 随机梯度下降上升法在非凸-PL极小极大博弈中是次优的

    Stochastic Gradient Descent Ascent is Suboptimal for Nonconvex-PL Min-Max Games

    [https://arxiv.org/abs/2610.07814](https://arxiv.org/abs/2610.07814)

    该论文首次建立了非凸-PL极小极大博弈中固定时间尺度比双时间尺度SGDA的紧致复杂度下界，证明SGDA本质上次优——其下界与现有上界匹配、与Smoothed-AGDA形成复杂度分离，且当时间尺度比小于o(κ²)时甚至无法找到驻点。

    

    arXiv:2610.07814v1 公告类型： cross 摘要：在非凸极小极大博弈中，通过调整时间尺度比和步长，随机梯度下降上升法（SGDA）究竟能走多远？我们针对非凸-PL（NC-PL）博弈回答了这一问题，首次建立了固定时间尺度比和非递增步长下双时间尺度SGDA的紧致复杂度。对于满足内层μ-PL不等式的ℓ-光滑博弈，我们证明了复杂度下界Ω(κ²ℓε⁻²+κ⁴ℓσ²ε⁻⁴)，其中κ=ℓ/μ为条件数，σ²为梯度方差，ε度量外层梯度范数。该下界与现有的SGDA上界相匹配，并建立了与Smoothed-AGDA（Yang等人，22'）之间的复杂度分离。此外，我们证明当SGDA的时间尺度比小至o(κ²)时，它可能无法找到驻点。我们的负面结果凸显了SGDA在NC-PL博弈中的根本局限性，并……（原文摘要至此截断）

    arXiv:2610.07814v1 Announce Type: cross  Abstract: How far can stochastic gradient descent ascent (SGDA) go by tuning its timescale ratio and step sizes in nonconvex min-max games? We answer this question for nonconvex-PL (NC-PL) games by establishing the first tight complexity of two-timescale SGDA with a fixed timescale ratio and non-increasing step sizes. For $\ell$-smooth games with an inner $\mu$-PL inequality, we prove a complexity lower bound $\Omega(\kappa^2\ell\varepsilon^{-2}+\kappa^4\ell\sigma^2\varepsilon^{-4})$, where $\kappa=\ell/\mu$ is the condition number, $\sigma^2$ is the gradient variance, and $\varepsilon$ measures the outer gradient norm. This matches existing SGDA upper bounds and establishes a complexity separation from Smoothed-AGDA (Yang et al., 22'). In addition, we show that SGDA can fail to find a stationary point when its timescale ratio is as small as $o(\kappa^2)$. Our negative results highlight the fundamental limitation of SGDA in NC-PL games, and just
    
[^140]: SIFT：面向Airbnb多任务个性化筛选条件排序的搜索意图到筛选条件Transformer

    SIFT: Search Intent-to-Filter Transformer for Multi-Task Personalized Filter Ranking at Airbnb

    [https://arxiv.org/abs/2610.07810](https://arxiv.org/abs/2610.07810)

    Airbnb提出基于Transformer的SIFT模型，直接从用户原始行为序列学习统一的偏好表示以替代人工特征工程，通过多任务预测（预订可能性、筛选条件参与度、序数容量阈值）实现个性化筛选条件排序，并兼容布尔型与数值区间型筛选条件。

    

    搜索筛选条件帮助用户在Airbnb这样的双边市场平台中浏览庞大的房源目录，推荐合适的筛选条件能够显著提升预订转化率。然而，许多生产环境中的筛选条件排序系统依赖ETL流水线生成的人工设计、预先聚合的特征来表示用户，这导致维护成本高昂，且难以扩展到新的筛选类型或上下文维度（如行程时长、出行人数）。我们提出了SIFT（Search Intent-to-Filter Transformer，搜索意图到筛选条件Transformer），这是一个基于Transformer的排序模型，直接从用户原始行为序列中学习用户偏好。SIFT用一个统一的用户表示取代了人工特征工程，该表示同时支持多个预测任务，包括预订可能性、筛选条件参与度以及序数容量阈值（如2间及以上卧室），为双边市场的筛选条件排序提供了一个通用框架，同时兼容布尔型和数值区间型筛选条件。

    arXiv:2610.07810v1 Announce Type: new  Abstract: Search filters help guests navigate vast catalogs in two-sided marketplaces like Airbnb, and recommending the right filters can meaningfully lift booking conversion. Many such production filter-ranking systems, however, represent the guest through hand-engineered, pre-aggregated features generated by ETL pipelines. This makes it expensive to maintain and difficult to extend for new filter types or contextual dimensions (trip length, group size). We present SIFT (Search Intent-to-Filter Transformer), a ranking model built on transformers that learns guest preferences directly from raw behavioral sequences. SIFT replaces manual feature engineering with a unified guest representation that feeds multiple prediction tasks, including booking likelihood, filter engagement, and ordinal capacity thresholds (e.g., 2+ bedrooms) -- a general framework for filter ranking in two-sided marketplaces that accommodates both boolean and numeric-range filte
    
[^141]: MASKerade：面向稠密到MoE模型升级的Token路由掩码专家方法

    MASKerade: Token-Routed Mask Experts for Dense-to-MoE Upcycling

    [https://arxiv.org/abs/2610.07809](https://arxiv.org/abs/2610.07809)

    提出MASKerade方法，将专家定义为冻结预训练FFN上学习得到的二值掩码稀疏子网络，并通过token级路由器选择与组合被掩码的FFN，在不改变原始FFN权重的情况下实现高效且无需重训权重的稠密到MoE模型升级。

    

    稀疏激活的混合专家模型能够在不按比例增加每token计算量的情况下提升模型容量。稠密到MoE的升级方法通过复用预训练的稠密模型来构建此类系统，常见做法是将前馈网络（FFN）复制为独立训练的专家。我们提出MASKerade，一种稠密到MoE的训练方法，它转而将专家学习为冻结的预训练FFN的稀疏子网络。每个专家由一个学习到的二值掩码定义，token级路由器选择执行并组合哪些被掩码的FFN。路由器和掩码分数被联合优化，而底层FFN的权重值保持不变。这一公式化方案支持在同一路由架构内使用神经元结构化、半结构化和非结构化的专家。我们的主要配置采用四个2:4稀疏度专家配合top-2路由，其中两次半稠密专家传递仅具有一次稠密传递的名义FFN计算量，而无需……

    arXiv:2610.07809v1 Announce Type: new  Abstract: Sparsely activated Mixture-of-Experts (MoE) models increase model capacity without a proportional increase in per-token computation. Dense-to-MoE upcycling reuses pretrained dense models to construct such systems, commonly by copying feed-forward networks (FFNs) into independently trained experts. We introduce MASKerade, a dense-to-MoE training method that instead learns experts as sparse subnetworks of a frozen pretrained FFN. Each expert is defined by a learned binary mask, and a token-level router selects which masked FFNs to execute and combine. The router and mask scores are optimized jointly, while the underlying FFN weight values remain unchanged. This formulation supports neuron-structured, semi-structured, and unstructured experts within the same routing architecture. Our main configuration uses four 2:4 experts with top-2 routing, where two half-dense expert passes have the nominal FFN arithmetic of one dense pass, without requ
    
[^142]: 共模误差限制低时间步深度脉冲Q网络

    Common-Mode Errors Limit Low-Timestep Deep Spiking Q-Networks

    [https://arxiv.org/abs/2610.07808](https://arxiv.org/abs/2610.07808)

    本研究揭示了低时间步深度脉冲Q网络的性能下降主要源于跨动作共享的共模误差对自举时间差分学习的不成比例损害，并据此提出共模补偿方法以提升低时间步DSQNs的性能。

    

    arXiv:2610.07808v1 公告类型： cross 摘要： 脉冲神经网络（SNNs）提供稀疏且事件驱动的计算方式，使其在边缘设备上的能耗受限强化学习（RL）中极具吸引力。在基于价值的强化学习中，深度脉冲Q网络（DSQNs）将这种高效性与用于决策的动作价值估计相结合。然而，现有的DSQNs通常需要多个仿真时间步才能获得有竞争力的性能，这增加了计算和能源成本，而减少时间步则会导致显著的性能下降。我们从Q值估计误差的角度研究了这种性能下降。通过将各动作间的误差分解为共模和差模分量，我们发现低时间步DSQNs在动作值之间共享的共模误差上遭受了不成比例的性能损失，而共模误差通过自举目标对时间差分学习尤为有害。基于这一发现，我们提出了共模补偿深度脉冲Q网络……

    arXiv:2610.07808v1 Announce Type: cross  Abstract: Spiking neural networks (SNNs) offer sparse and event-driven computation, making them attractive for energy-constrained reinforcement learning (RL) on edge devices. In value-based RL, deep spiking Q-networks (DSQNs) combine such efficiency with action-value estimation for decision making. However, existing DSQNs often require multiple simulation timesteps for competitive performance, increasing computational and energy costs, whereas reducing the timesteps can cause substantial performance degradation. We investigate this degradation from the perspective of Q-value estimation errors. By decomposing errors across actions into common-mode and differential-mode components, we find that low-timestep DSQNs suffer disproportionately from common-mode errors shared across action values, which are particularly detrimental to temporal-difference learning through bootstrapped targets. Based on this finding, we propose Common-Mode Compensation Dee
    
[^143]: 通过上下文学习实现自适应均值估计：梯度流分析

    Adaptive Mean Estimation by In-Context Learning: A Gradient-Flow Analysis

    [https://arxiv.org/abs/2610.07804](https://arxiv.org/abs/2610.07804)

    该论文通过梯度流分析揭示了先验拟合网络（如TabPFN）如何在分布族未知的均值估计任务中通过上下文学习获得统计自适应性，自动选择与数据分布相匹配的最优估计策略并达到相应的理论收敛速率。

    

    先验拟合网络（PFNs）如TabPFN如今在预测和估计任务上已能与成熟的统计方法相媲美。一个自然的解释是PFNs具有统计自适应性，即对于一组异构模型，它们的表现几乎与针对真实数据生成模型定制的方法一样好，而无需被告知数据来自哪个模型。我们在一个受控的位置估计问题中研究这种自适应性是如何被学习到的。每个任务是一个未标记样本，其分布族是隐藏的：高斯数据需要用平均法估计，误差阶为 $n^{-1}$；而均匀数据最好通过其极值来估计，达到更快的 $n^{-2}$ 速率。我们还给出了对称高斯混合的例子，其可达到的速率为 $\sigma^2_n/n$。在标量输入上，softmax注意力计算的是经验累积生成函数的导数。因此，单个基元即可同时提供……

    arXiv:2610.07804v1 Announce Type: new  Abstract: Prior Fitted Networks (PFNs) such as TabPFN now rival established statistical procedures across prediction and estimation tasks. A natural explanation is that PFNs have the property of statistical adaptivity, that is, they perform nearly as well as a method tailored to the true data-generating model for a heterogeneous set of models, while not being told which model the data comes from. We study how such adaptivity is learned in a controlled location-estimation problem. Each task is an unlabeled sample whose family is hidden: Gaussian data call for averaging, with error of order $n^{-1}$, whereas uniform data are best estimated from their extremes, at the faster rate $n^{-2}$. We also provide the example of a symmetric Gaussian mixture, for which a rate of $\sigma^2_n/n$ can be attained. On scalar inputs, softmax attention computes the derivative of the empirical cumulant-generating function. A single primitive therefore both supplies fe
    
[^144]: 赋权的几何学

    The Geometry of Empowerment

    [https://arxiv.org/abs/2610.07796](https://arxiv.org/abs/2610.07796)

    本文将赋权最大化与技能学习方法相联系，提出了解释赋权的新几何框架，解答了赋权与结构中心性之间联系的长期开放问题，并揭示了信息几何与奖励几何的区别，为构建可扩展的赋权最大化方法奠定理论基础。

    

    赋权刻画了智能体主动控制其环境的能力。尽管作为一种信息论量在概念上颇具吸引力，但赋权与那些能广泛通往未来结果的结构性中心状态之间的联系，一直是一个悬而未决的问题。在这项工作中，我们将赋权最大化与技能学习方法联系起来，为解释和分析赋权提供了新的几何视角。我们的分析回答了关于赋权与结构中心性之间联系的长期开放问题，还揭示了信息几何与奖励几何之间的区别，为构建可扩展的赋权最大化方法提供了重要的理论启示。网站与代码可在 https://empowerment-geometry.github.io/ 获取。

    arXiv:2610.07796v1 Announce Type: cross  Abstract: Empowerment captures the capacity for an agent to actively control its environment. While conceptually appealing as an information-theoretic quantity, the connection between empowerment and structurally central states that provide broad access to future outcomes has remained an open question. In this work, we link empowerment maximization and skill-learning methods to provide new geometries for interpreting and analyzing empowerment. Our analyses answer longstanding open questions on the connections between empowerment and structural centrality. Our analyses also reveal distinctions between information and reward geometries, highlighting important theoretical implications to build scalable empowerment-maximization methods. Website and code can be found at https://empowerment-geometry.github.io/.
    
[^145]: ServeLearnBench：智能体从服务经验中自我改进的能力究竟有多强？

    ServeLearnBench: How Well Can Agents Self-Improve from Serving Experience?

    [https://arxiv.org/abs/2610.07792](https://arxiv.org/abs/2610.07792)

    本文提出 ServeLearnBench 基准及演化环境流式数据集（EESD），用于系统评估大语言模型智能体在隐藏策略持续演变的环境中，能否从交互与反馈中推断、应用并修正潜在环境知识，从而实现自我改进。

    

    大语言模型智能体正越来越多地被部署到真实环境中执行复杂任务。然而，在这些环境中实现正确行为所需的知识往往是隐含的、未公开的，并且会随时间推移而变化。近期的持续学习框架试图通过让智能体从服务经验中不断改进来应对这一挑战，但这些方法的有效性与局限性尚未得到充分的刻画。现有基准仅提供了部分覆盖：有些基准明确给出目标知识，有些则假设环境是静态的，而支持持续适应的基准在规模和知识多样性方面仍然有限。为了实现系统化评估，我们形式化定义了演化环境流式数据集（EESD），在隐藏策略不断演变的过程中，智能体必须从交互与结果反馈中推断、应用并修正潜在的环境知识，并在此基础上提出了 ServeLearnBench 基准。

    arXiv:2610.07792v1 Announce Type: cross  Abstract: Large language model agents are increasingly deployed to perform complex tasks in real-world environments. However, the knowledge required for correct behavior in these environments is often implicit, undisclosed, and subject to change over time. Recent continual-learning harnesses seek to address this challenge by enabling agents to improve from serving experience. Yet the effectiveness and limitations of these methods are not yet well characterized. Existing benchmarks provide only partial coverage: some explicitly provide the target knowledge, others assume a static environment, and those that support continual adaptation remain limited in scale and knowledge diversity. To enable systematic evaluation, we formalize an evolving-environment streaming dataset (EESD), in which agents must infer, apply, and revise latent environment knowledge from interaction and outcome feedback as hidden policies evolve, and introduce ServeLearnBench, 
    
[^146]: 通过有限阶松弛将路径梯度扩展到离散随机变量

    Extending Pathwise Gradients to Discrete Random Variables via Finite-Order Relaxation

    [https://arxiv.org/abs/2610.07786](https://arxiv.org/abs/2610.07786)

    提出一个通用框架，通过有限阶松弛为泊松等常见离散变量构建精确的路径梯度估计器，该估计器保留硬前向采样、无需温度调节、实现简单，且在所有可行解中唯一并最小化权重方差。

    

    路径梯度因其无偏、低方差且仅需单样本即可工作的特性，在连续随机变量中备受青睐。然而对于离散变量，路径恒等式通常无法对每个可微函数都精确成立。我们提出了一个通用框架，可为诸如泊松分布等一系列常见离散变量构建有限阶精确路径梯度估计器。该估计器是在所有对不超过某阶次的多项式均无偏的解中范数最小的解。所得估计器保留了硬前向采样，无需温度调节，且仅需几行代码即可实现。与其他可行解相比，我们的估计器是唯一的并能最小化权重方差；相比之下，先前的工作使用类别变量或增广表示来近似非类别变量，这会引入额外的方差和计算开销。为了理解逼近偏差……

    arXiv:2610.07786v1 Announce Type: new  Abstract: Pathwise gradients are preferred for continuous random variables because they are unbiased, low variance, and work with a single sample. For discrete variables, however, the pathwise identity cannot generally be exact for every differentiable function. We propose a general framework to construct finite-order exact pathwise gradient estimators for a range of common discrete variables such as Poisson. The estimator is the least-norm solution among all solutions that are unbiased for polynomials of degree at most. The resulting estimators preserve the hard forward sample, require no temperature tuning, and can be implemented in a few lines of codes. Against other admissible solutions, our estimator is unique and minimizes weight variance; in contrast, prior works use categorical variables or augmented representations to approximate non-categorical variables that induces excess variance and computations. To understand approximation bias for 
    
[^147]: 多智能体LLM推理中的持久记忆：它花费什么、带来什么、以及何时能被察觉

    Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell

    [https://arxiv.org/abs/2610.07782](https://arxiv.org/abs/2610.07782)

    本文在三层多智能体LLM推理架构中实测发现，上下文分解可将峰值KV缓存从35.5 MiB降至14.3 MiB，而持久记忆层不仅增加0.368 MiB缓存开销，且在单问题基准上未带来任何可检测的准确率提升，这种零结果源于基准测试的结构性特点。

    

    将长上下文推理分解到协作智能体之间，可以限制每次调用的活跃KV缓存而非总证据量，这在KV缓存内存成为瓶颈时尤为重要。许多此类系统会添加一个持久层来存储和回忆推理轨迹，通常通过消融实验报告的准确率提升来验证其效果。我们在一个三层智能体架构上对两者进行了测量。上下文分解确实带来了成效：每个查询的峰值KV工作集为14.3 MiB，而单次传递和检索增强基线分别为35.5和35.3 MiB。持久层则没有带来收益：在八个受控数据集对、每组n=100的实验中，它使峰值缓存增加0.368 MiB [+0.167, +0.590]，且未产生可检测的准确率变化（+0.015，95% CI [-0.011, +0.046]）。我们认为这种零结果是结构性的：单问题基准测试为每个条目提供其独立的证据并独立评分，且正确性要求在条件之间重置已存储的推理轨迹，因此记忆回忆没有任何有价值的信息可供利用

    arXiv:2610.07782v1 Announce Type: new  Abstract: Decomposing long-context inference across cooperating agents bounds the active KV cache per call rather than total evidence, which matters when KV-cache memory binds. Many such systems add a persistent tier storing and recalling reasoning traces, usually validated by an ablation reporting an accuracy gain. We measure both on one three-tier agent architecture. Decomposition delivers: peak KV working set of 14.3 MiB per query against 35.5 and 35.3 MiB for single-pass and retrieval-augmented baselines. The persistent tier does not: across eight controlled dataset pairs at n=100 per arm it costs +0.368 MiB [+0.167, +0.590] of peak cache and produces no detectable accuracy change (+0.015, 95% CI [-0.011, +0.046]). We argue the null is structural: single-question benchmarks supply each item with its own evidence and score it independently, and correctness requires resetting stored traces between conditions, so recall has nothing informative to
    
[^148]: 量化对工具故障恢复的影响因提示词和评估设计而异

    Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs

    [https://arxiv.org/abs/2610.07781](https://arxiv.org/abs/2610.07781)

    该研究发现8比特与4比特量化对语言模型智能体工具故障恢复能力的影响并不稳定，比较结论会随提示词和评估目标（如评分任务的选择）而改变方向甚至反转，表明量化效果的结论高度依赖于评估设计。

    

    训练后量化降低了部署语言模型智能体的成本，但其对从临时性工具故障中恢复的影响可能取决于评估恢复能力的方式。我们在二十个确定性工具使用任务和五个提示词上，比较了Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct的8比特与4比特变体。8比特与4比特在恢复能力上的比较结果会随提示词和评估目标而改变方向。在同一提示词下两个变体均能无故障完成的任务上，Llama的差异范围为0至+20.2个百分点，Qwen的差异范围为-50.0至+35.0个百分点。全流程点估计在所有五个提示词下都偏向8比特的Llama，而Qwen的比较结果则随提示词改变方向。评估目标的选择也可能使结论反转。在某一提示词下的Llama上，仅在每个变体各自无故障通过的任务上评分时，4比特领先17.5个百分点；而对两个变体在相同任务上评分时则没有差异。

    arXiv:2610.07781v1 Announce Type: new  Abstract: Post-training quantization reduces the cost of deploying language-model agents, but its effect on recovery from temporary tool failures can depend on how recovery is evaluated. We compare 8-bit and 4-bit variants of Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on twenty deterministic tool-use tasks and five prompts. The 8-bit-4-bit recovery comparison changes direction across prompts and evaluation targets. On tasks that both variants complete without faults under the same prompt, the difference ranges from 0 to +20.2 percentage points for Llama and from -50.0 to +35.0 points for Qwen. Full-pipeline point estimates favor 8-bit Llama under all five prompts, whereas the Qwen comparison changes direction across prompts. The evaluation target can also reverse the result. For Llama under one prompt, scoring each variant only on its own clean-passing tasks favors 4-bit by 17.5 points; scoring the same tasks for both variants gives no differen
    
[^149]: APEX：更聪明地推测，而非更深地推测

    APEX: Speculate smarter, not deeper

    [https://arxiv.org/abs/2610.07780](https://arxiv.org/abs/2610.07780)

    APEX是一个学习型控制器，通过请求级专家选择和块级草稿深度自适应来优化推测解码，用更聪明的推测取代更深的推测，从而降低大语言模型推理延迟并减少计算浪费。

    

    推测解码通过在目标模型验证之前起草多个token来降低大语言模型的推理延迟，但其有效性同时取决于提议机制和草稿深度。固定配置无法响应生成过程中可预测性、重复性和接受率的变化，因此更深的草稿可能会增加计算浪费，却无法带来成比例的加速。我们提出了APEX，这是一个通过请求级专家选择和块级深度自适应来平衡解码速度与草稿token浪费的学习型控制器。APEX-Router为每个请求在EAGLE-3、n-gram和草稿模型推测之间进行选择，而APEX-Depth则利用因果解码信号和最近的验证器反馈，在每个验证块中调整草稿长度。APEX将接受的草稿长度建模为截断生存反馈，学习按位置拒绝风险、块执行成本，以及平衡吞吐率的动作效用函数。

    arXiv:2610.07780v1 Announce Type: new  Abstract: Speculative decoding reduces large language model inference latency by drafting multiple tokens before target-model verification, but its effectiveness depends on both the proposal mechanism and draft depth. Fixed configurations cannot respond to changes in predictability, repetition, and acceptance during generation, so deeper drafting can increase wasted computation without proportional speedup. We introduce APEX, a learned controller that balances decoding speed and draft-token waste through request-level expert selection and block-level depth adaptation. APEX-Router selects among EAGLE-3, n-gram, and draft-model speculation for each request, while APEX-Depth adjusts draft length at each verification block using causal decoding signals and recent verifier feedback. APEX models accepted draft length as censored survival feedback, learning position-wise rejection hazards, block execution costs, and an action utility that balances throug
    
[^150]: 迈向属性图聚类的“一个模型通用所有”基础模型

    Towards One-for-All Foundation Model for Attributed Graph Clustering

    [https://arxiv.org/abs/2610.07778](https://arxiv.org/abs/2610.07778)

    提出OFAG——一个面向属性图聚类的基础模型，仅需一次训练即可直接应用于多样化的属性图，无需针对特定图的训练、微调或超参数搜索。

    

    属性图聚类旨在通过联合利用节点属性和图拓扑结构来发现节点群组，然而其无监督的本质使得模型选择与适配天然困难。现有方法通常为每个输入图单独训练和调优一个模型，导致流程成本高昂且脆弱，往往无法迁移到具有不同特征空间、结构模式以及属性-结构相关性的图上。本文研究了一种“一个模型通用所有”的替代方案：能否训练一个单一模型，直接应用于多样化的属性图，而无需针对特定图的训练、微调或超参数搜索？我们提出了OFAG——一个面向属性图聚类的基础模型。OFAG基于先验数据拟合网络（Prior-data Fitted Networks）构建，从在潜在聚类、节点属性和图结构的宽泛先验下生成的合成属性图中学习一种可复用的聚类推理策略。

    arXiv:2610.07778v1 Announce Type: cross  Abstract: Attributed graph clustering aims to discover node groups by jointly exploiting node attributes and graph topology, yet its unsupervised nature makes model selection and adaptation inherently difficult. Existing methods typically train and tune a separate model for each input graph, leading to costly and fragile pipelines that often fail to transfer across graphs with different feature spaces, structural patterns, and attribute-structure correlations. In this paper, we study a one-for-all alternative: can a single model be trained once and directly applied to diverse attributed graphs without graph-specific training, fine-tuning, or hyperparameter search? We propose OFAG, a foundation model for attributed graph clustering. Building upon Prior-data Fitted Networks, OFAG learns a reusable clustering inference strategy from synthetic attributed graphs generated under broad priors over latent clusters, node attributes, and graph structures.
    
[^151]: TRACE：面向MoE语言模型FP4强化学习的Rollout引导量化感知训练

    TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models

    [https://arxiv.org/abs/2610.07767](https://arxiv.org/abs/2610.07767)

    TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。

    

    对大语言模型（LLM）进行后训练的强化学习（RL）在rollout生成过程中会产生大量的计算和内存开销，这促使人们采用低精度rollout来实现高效的RL训练。然而，现有的FP4 RL方法存在一个关键局限：它们主要在训练路径和rollout路径上分别独立地优化量化精度，而不是直接减少两条量化执行路径之间的差异。在本工作中，我们提出了TRACE（Train-Rollout Quantization Alignment via Compact GuidancE，通过紧凑引导实现训练-Rollout量化对齐），这是一个面向混合专家（MoE）语言模型RL训练的FP4量化框架，解决了现有FP4 RL方法的局限性。TRACE引入了rollout引导的量化感知训练，利用rollout侧的量化结果来指导训练侧的FP4舍入决策，从而直接减少训练与rollout之间的差异。此外，TRACE采用了高效的量化……

    arXiv:2610.07767v1 Announce Type: cross  Abstract: Reinforcement learning (RL) for post-training large language models (LLMs) incurs substantial computation and memory overhead during rollout generation, which motivates low-precision rollout for efficient RL training. However, existing FP4 RL methods suffer from a key limitation: they primarily optimize quantization accuracy on the training and rollout paths independently rather than directly reducing the discrepancy between the two quantized execution paths. In this work, we propose TRACE (Train-Rollout Quantization Alignment via Compact GuidancE), an FP4 quantization framework for RL training of Mixture-of-Experts (MoE) language models that addresses the limitation of existing FP4 RL methods. TRACE incorporates rollout-guided quantization-aware training that uses rollout-side quantization outcomes to guide training-side FP4 rounding decisions, directly reducing train-rollout discrepancy. Moreover, TRACE adopts an efficient quantizati
    
[^152]: 基于AI裁判的可信方法比较：顺序、批次与聚合效应下的估计与设计

    Trustworthy Method Comparison with AI Judges: Estimation and Design under Order, Batch, and Aggregation Effects

    [https://arxiv.org/abs/2610.07755](https://arxiv.org/abs/2610.07755)

    该论文提出用马尔可夫广义线性混合模型刻画LLM裁判的评估机制，证明了随机化取平均排名法的一致性条件及威廉姆斯方设计的效率优势，并揭示了组间比较中因模型非线性导致朴素平均可能得出错误结论的问题。

    

    大语言模型（LLM）正越来越多地被用作自动化AI评估的裁判。一种常见做法是将提示序列随机化并对所得分数取平均，但其统计有效性尚不明确。我们证明了LLM评估机制可以用一类马尔可夫广义线性混合模型（GLMM）来近似，这一结论得到了三个主要商业LLM样本外预测结果的支持。利用一阶马尔可夫GLMM，我们研究了排行榜排名和组间比较问题。对于排行榜排名，在温和的分离条件下，随机化后取平均的选择方法具有一致性；当被评估项质量接近时，采用威廉姆斯方设计可以提高效率。对于组间比较，由于响应模型的非线性，朴素平均方法可能对组级质量差异得出不一致的结论。实证结果进一步支持了所提出的基于模型的推断方法在一阶近似之外的有效性。

    arXiv:2610.07755v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judges for automated AI evaluation. A common practice is to randomize prompt sequences and average the resulting scores, but its statistical validity remains unclear. We show that LLM evaluation mechanisms can be approximated by a class of Markov generalized linear mixed models (GLMMs), supported by out-of-sample predictions across three major commercial LLMs. Using a first-order Markov GLMM, we study leaderboard ranking and group comparison. For leaderboard ranking, randomize-and-average selection is consistent under a mild separation condition, and a Williams square design can improve efficiency when item qualities are close. For group comparison, naive averaging can yield inconsistent conclusions about differences in group-level quality because of the response model's nonlinearity. Empirical results further support the validity of the proposed model-based inference beyond the fir
    
[^153]: 对抗训练的线性Transformer是高斯混合分布的最优鲁棒上下文学习者

    Adversarially Trained Linear Transformers Are Optimal Robust In-Context Learners for Gaussian Mixtures

    [https://arxiv.org/abs/2610.07754](https://arxiv.org/abs/2610.07754)

    经过跨任务对抗预训练的线性Transformer无需额外训练，即可通过上下文学习将鲁棒性迁移到未见过的任务，并渐近达到高斯混合分类任务的最优鲁棒贝叶斯误差。

    

    对抗训练是对抗攻击最可靠的防御手段之一，但其高昂的计算成本通常需要针对每个任务重新付出。鲁棒基础模型提供了一种有前景的替代方案：只需对模型进行一次对抗性预训练，然后通过轻量级适配将其鲁棒性迁移到下游任务。然而，一个根本性的问题仍未解决：预训练中获得的鲁棒性能否在无需进一步对抗训练的情况下迁移到未见过的任务？在本研究中，我们对这一问题给出了肯定的答案。一个经过大规模对抗预训练的单一模型，无需额外的任务特定训练即可在新任务上实现最优鲁棒性。具体而言，我们证明，对于一类高斯混合分类任务，经过跨任务对抗训练的足够深的线性Transformer，可以通过对干净样本的上下文学习，渐近地达到在未见过的任务上的鲁棒贝叶斯误差。

    arXiv:2610.07754v1 Announce Type: new  Abstract: Adversarial training is one of the most reliable defenses against adversarial attacks, but its high computational cost must generally be paid anew for each task. Robust foundation models offer a promising alternative: adversarially pretrain a model once and then transfer its robustness to downstream tasks through lightweight adaptation. However, a fundamental question remains open: can robustness acquired during pretraining transfer to unseen tasks without further adversarial training? In this study, we answer this question affirmatively. A single model adversarially pretrained at scale can achieve optimal robustness on new tasks without additional task-specific training. Specifically, we show that, for a family of Gaussian-mixture classification tasks, a sufficiently deep linear transformer adversarially trained across tasks can asymptotically attain the robust Bayes error on previously unseen tasks through in-context learning from clea
    
[^154]: 基于调和权重的高维在线校准

    High-dimensional online calibration from harmonic weights

    [https://arxiv.org/abs/2610.07740](https://arxiv.org/abs/2610.07740)

    本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。

    

    我们研究了在任意凸集 $Y\subseteq\mathbb{R}^d$ 上、相对于任意误差范数 $\|\cdot\|_{L}$ 的多维预测在线校准问题。对于同时预测 $d$ 个二元结果（$Y=[0,1]^d$）的情形，我们给出了首个在每个固定精度下都能以关于 $d$ 为多项式的轮数实现 $\varepsilon$-校准的算法。该算法需要 $d^{O(1/\varepsilon)}$ 轮，相比此前界中的维度依赖性实现了指数级改进。对于多类别预测（$Y=\Delta_d$），我们获得了相同的 $d^{O(1/\varepsilon)}$ 速率，改进了 Peng 以及 Fishelson 等人给出的 $d^{\widetilde{O}(1/\varepsilon^2)}$ 界。我们的算法非常简单：在每一轮，它输出一个对过去结果进行调和平滑后的调和加权分布。同一个算法适用于所有预测集和范数。更一般地，它在 $\exp(O(\gamma(Y,L)/\varepsilon$……（原文摘要在此处截断）轮后即可实现 $\varepsilon$-校准。

    arXiv:2610.07740v1 Announce Type: cross  Abstract: We study the online calibration of multidimensional forecasts over an arbitrary convex set $Y\subseteq\mathbb{R}^d$ relative to an arbitrary error norm $\|\cdot\|_{L}$. For forecasting $d$ binary outcomes simultaneously ($Y=[0,1]^d$), we give the first algorithm that achieves $\varepsilon$-calibration in a number of rounds that is polynomial in $d$ for every fixed accuracy. It requires $d^{O(1/\varepsilon)}$ rounds, exponentially improving the dimension dependence of previous bounds. For multi-class forecasting ($Y=\Delta_d$), we obtain the same $d^{O(1/\varepsilon)}$ rate, improving the $d^{\widetilde{O}(1/\varepsilon^2)}$ bounds of Peng and Fishelson et al.   Our algorithm is simple: on each round, it outputs a harmonically weighted distribution over harmonically smoothed past outcomes. The same algorithm works for every forecast set and norm. More generally, it achieves $\varepsilon$-calibration after $\exp(O(\gamma(Y,L)/\varepsilon
    
[^155]: 引用你所探索的：基于可验证证据的医疗知识图谱预算感知LLM推理

    Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence

    [https://arxiv.org/abs/2610.07739](https://arxiv.org/abs/2610.07739)

    该论文提出了BAR框架，首次将成本受限的KG探索、按来源质量区分的可验证证据以及可引用的推理依据三者结合，利用LLM在医疗知识图谱上进行推理，以弥补EHR中缺失的依赖关系并提升出院后风险预测的可靠性。

    

    基于电子健康记录（EHR）的出院后风险预测十分困难，因为许多将出院时观察结果与下游并发症联系起来的依赖关系——如共病级联和药物-疾病相互作用——在记录中是缺失的。外部医疗知识图谱（KG）可以补充这些缺失的依赖关系，但追踪它们需要满足三个特性：KG探索必须保持成本受限，检索到的证据必须按来源质量进行区分，且所得到的推理依据必须可引用以供回顾性审查。大型语言模型（LLM）能够在结构化证据上进行规划与验证，使其成为KG推理的自然候选者，但现有的基于LLM的方法无法同时满足这三个特性。在本文中，我们提出了BAR，一个面向医疗知识图谱的预算感知LLM推理框架，包含三项贡献。首先，BAR将原始KG精炼为疾病特定的证据图，其边（摘要截断于此）……

    arXiv:2610.07739v1 Announce Type: new  Abstract: Post-discharge risk prediction from electronic health records (EHRs) is difficult because many dependencies that link discharge-time observations to downstream complications, such as comorbidity cascades and drug-disease interactions, are absent from the record. External medical knowledge graphs (KGs) can supply these missing dependencies, but tracing them demands three properties: KG exploration must remain cost-bounded, retrieved evidence must be differentiated by source quality, and the resulting rationale must be citable for retrospective review. Large language models (LLMs) can plan and verify over structured evidence, making them natural candidates for KG reasoning, but existing LLM-based methods do not satisfy these three properties jointly. In this paper, we propose BAR, a Budget-Aware LLM Reasoning framework over medical KGs with three contributions. First, BAR refines the raw KG into disease-specific evidence graphs whose edges
    
[^156]: 多臂老虎机的纳什社会福利：轨迹级期望与高概率遗憾

    Nash Social Welfare for Multi Armed Bandits: Trajectory-wise Expected and High Probability Regret

    [https://arxiv.org/abs/2610.07737](https://arxiv.org/abs/2610.07737)

    该论文提出了一种新的“轨迹级纳什遗憾”度量，通过先对完整奖励样本路径取几何平均再求期望，弥补了现有度量忽略各轮奖励联合分布的缺陷，并借助詹森不等式证明其严格强于原有度量，从而更忠实地体现纳什社会福利的公平性目标。

    

    我们研究在纳什社会福利（NSW）目标下的公平多臂老虎机问题，该目标通过累积奖励的几何平均来衡量性能。现有工作将纳什遗憾定义为 $\mathrm{NR}_T = \mu^\star - (\prod_{t=1}^T \mathbb{E}\mu_{I_t})^{1/T}$，其中 $\mu_{I_t}$ 是推荐臂 $I_t$ 的平均奖励，$T$ 是时间范围。由于该定义将几何平均应用于每轮的边际期望，忽略了各轮奖励之间的联合分布，未能在轨迹层面体现NSW的公平性动机。我们提出轨迹级纳什遗憾 $\widetilde{\mathrm{NR}}_T = \mu^\star - \mathbb{E}[(\prod_{t=1}^T \mu_{I_t})^{1/T}]$，它在取期望之前先对完整样本路径计算几何平均，从而更忠实地刻画NSW的公平性。根据詹森不等式，$\widetilde{\mathrm{NR}}_T \geq \mathrm{NR}_T$，这使其成为一个严格更强的度量指标。我们还引入了（摘要在此处截断）

    arXiv:2610.07737v1 Announce Type: cross  Abstract: We study fair multi-armed bandits under the Nash Social Welfare (NSW) objective, which measures performance via the geometric mean of accumulated rewards. Existing work defines Nash regret as $\mathrm{NR}_T = \mu^\star - (\prod_{t=1}^T \mathbb{E}\mu_{I_t})^{1/T}$, where $\mu_{I_t}$ is the mean reward of the recommended arm $I_t$ and $T$ is the horizon. Since it applies the geometric mean to per-round marginal expectations, it ignores the joint distribution of rewards across rounds, leaving the NSW fairness motivation unaddressed at the trajectory level. We propose \emph{trajectory-wise Nash regret} $\widetilde{\mathrm{NR}}_T = \mu^\star - \mathbb{E}[(\prod_{t=1}^T \mu_{I_t})^{1/T}]$, which computes the geometric mean over complete sample paths before taking expectations, capturing NSW fairness more faithfully. By Jensen's inequality, $\widetilde{\mathrm{NR}}_T \geq \mathrm{NR}_T$, making it a strictly stronger metric. We also introduce
    
[^157]: 在嵌入空间中通过强化学习学习检索

    Learning to Retrieve via Reinforcement Learning in Embedding Space

    [https://arxiv.org/abs/2610.07731](https://arxiv.org/abs/2610.07731)

    该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。

    

    密集检索模型通常使用对比目标进行训练，这类目标虽然能学习有效的表示，但无法直接优化检索指标或下游任务性能。为了解决这一问题，我们提出了RELER（面向检索的强化学习），这是一个强化学习框架，能够使现有的嵌入模型直接在嵌入空间中学习检索，并与任务特定的奖励对齐。我们通过以下方式训练RELER：从以归一化编码器输出为中心的von Mises-Fisher（vMF）分布中采样单位长度的查询和文档嵌入动作，将由此产生的检索或下游结果评分作为奖励，并使用留一法基线（RLOO）通过REINFORCE算法更新编码器。由于在高维嵌入空间中进行探索容易受到采样噪声的影响，我们进一步提出了条件均值投影（CMP），将每个采样的嵌入投影到低维子空间上……

    arXiv:2610.07731v1 Announce Type: cross  Abstract: Dense retrieval models are typically trained with contrastive objectives that learn effective representations but do not directly optimize retrieval metrics or downstream task performance. To address this problem, we introduce RELER (REinforcement LEarning for Retrieval), a reinforcement learning framework that enables existing embedding models to learn to retrieve directly in embedding space and align to task-specific rewards. We train RELER by sampling unit-length query and document embedding actions from von Mises-Fisher (vMF) distributions centered on normalized encoder outputs, scoring the resulting retrieval or downstream outcomes as rewards, and updating the encoder with REINFORCE using a leave-one-out baseline (RLOO). As exploration in the high-dimensional embedding space is prone to sampling noise, we further propose conditional-mean projection (CMP), which projects each sampled embedding onto the low-dimensional subspace span
    
[^158]: SanSi：一种用于系统1.5思维的循环式类型化决策模型

    SanSi: A Looped Typed Decision Model for System 1.5 Thinking

    [https://arxiv.org/abs/2610.07730](https://arxiv.org/abs/2610.07730)

    SanSi提出“系统1.5思维”——通过多次循环复用模型层在不生成文本的情况下修正隐藏状态，将预训练循环语言模型转化为类型化决策模型，在59个数据源的10,027个测试决策上达到72.0%准确率，比同结构非循环模型高出13.5个百分点。

    

    类型化决策模型在不生成文本的情况下回答一个预先声明的问题：决策头在单次前向传播中为每个声明的选项返回一个概率。单次前向传播快速而直观，属于系统1思维。我们研究了介于单次传播与生成式推理之间的方法：循环，即在输出一次类型化读数之前，将相同的层递归地应用多次。每一次循环都让模型在提交答案之前修正其隐藏状态，而无需生成任何词元；我们将其称为系统1.5思维。我们提出了SanSi，它将一个预训练的循环语言模型转化为类型化决策模型。选项概率在每次循环后被读取，且每次循环都使用适当的评分规则进行训练，因此单个模型可以在一次运行中服务于从一次循环到八次循环的任意计算预算。在来自59个数据源的10,027个测试决策上，SanSi达到了72.0%的准确率，比用相同结构训练的非循环模型高出13.5个百分点。

    arXiv:2610.07730v1 Announce Type: cross  Abstract: Typed decision models answer a declared question without generating text: a decision head returns a probability for each of the declared options in a single forward pass. A single pass is fast, intuitive System 1 thinking. We study what lies between one pass and generated reasoning: looping, in which the same layers are recursively applied several times before one typed readout. Each loop lets the model revise its hidden state before it commits to an answer, without generating a token; we call this System 1.5 thinking. We propose SanSi, which turns a pre-trained looped language model into a typed decision model. The option probabilities are read after every loop, and every loop is trained with a proper scoring rule, so that one model serves every budget from one loop to eight in a single run. On 10,027 test decisions from 59 sources, SanSi reaches 72.0% accuracy: 13.5 points above a non-looped model of the same shape trained with the s
    
[^159]: 模型自己埋下触发器：多轮大语言模型中的答案侧后门攻击

    The Model Plants the Trigger: Answer-Side Backdoor Attacks in Multi-Turn Large Language Models

    [https://arxiv.org/abs/2610.07723](https://arxiv.org/abs/2610.07723)

    本文提出一种新型“答案侧”后门攻击：利用良性首轮提示诱导模型自己生成一个看似无害的词作为触发器，使模型在多轮对话中识别自身生成的触发器并绕过安全拒绝机制，在仅5%投毒率下攻击成功率接近100%，而用户输入始终保持完全干净。

    

    大型语言模型的安全对齐仍然容易受到后门攻击。现有的LLM后门攻击几乎都以输入为中心：其激活依赖于用户输入中的显式触发模式，因此现代的安全防护措施被设计用来净化输入空间。我们用一种针对多轮对话的新型“答案侧”后门攻击挑战了这一假设。攻击者不是将触发器插入输入中，而是使用一个良性的第一轮提示词，自然地诱导模型生成一个特定的、看似无害的词。一旦这个自生成的词被合并到对话历史中，它就成为触发器。当后续的有害查询到来时，模型会检测到自己埋下的触发器并绕过其安全拒绝机制，而用户输入始终保持完全干净。在四个LLM上的实验表明，我们的攻击达到了接近完美的攻击成功率，在仅5%的投毒率下接近100%，同时保持了模型的通用能力和对干净输入的安全性，并且它……

    arXiv:2610.07723v1 Announce Type: cross  Abstract: Safety alignment in Large Language Models (LLMs) remains vulnerable to backdoor attacks. Existing LLM backdoors are almost all input-centric: activation depends on explicit trigger patterns in the user input, so modern guardrails are built to sanitize the input space. We challenge this assumption with a novel answer-side backdoor for multi-turn dialogue. Instead of inserting the trigger into the input, the adversary uses a benign first-turn prompt to naturally induce the model to generate a specific, seemingly innocuous word. Once merged into the dialogue history, this self-generated word becomes the trigger. When a later harmful query arrives, the model detects its own trigger and bypasses its safety refusal, while the user input stays perfectly clean. Across four LLMs, our attack reaches near-perfect Attack Success Rates, approaching 100\% at only a 5\% poisoning rate, while preserving general utility and clean-input safety, and it e
    
[^160]: 体积抽样岭回归的精确校准与尖锐风险几何

    Exact Calibration and Sharp Risk Geometry for Volume-Sampled Ridge Regression

    [https://arxiv.org/abs/2610.07721](https://arxiv.org/abs/2610.07721)

    本文为体积抽样岭回归建立了精确的惩罚校准理论（当且仅当抽样行数超过目标有效维度时该惩罚存在），并通过严格的扇形不等式刻画了中心化协方差风险的尖锐上界以及所有最大化响应的结构。

    

    我们研究从固定设计中恰好抽取 $s$ 个不同行进行岭回归的问题。响应是固定的，只有所抽取的子集是随机的。行列式法则与所选岭拟合共享同一个正定惩罚项。借助已建立的均值恒等式和指数族对偶性，我们给出了唯一的惩罚项，使其在期望意义上匹配一个指定的全数据岭拟合；该惩罚项当且仅当 $s$ 超过目标的有效维度时才存在。我们的主要结果关注以全数据惩罚损失归一化的中心化协方差风险。对于平衡的带符号坐标副本，一个严格的扇形不等式给出了从维度到行数减一之间的每一个预算水平下的尖锐风险以及所有最大化响应。这一结论对任何非零半正定查询均成立。当目标与查询固定时，最大化响应空间在这些预算水平之间保持不变。对于一般设计，我们刻画了留一包络的达到条件。对于现有的实等角……

    arXiv:2610.07721v1 Announce Type: cross  Abstract: We study ridge regression from exactly $s$ distinct rows of a fixed design. Responses are fixed, and only the subset is random. The determinant law and selected ridge fit share one positive definite penalty. Established mean identities and exponential-family duality give the unique penalty that matches a prescribed full-data ridge fit in expectation. It exists exactly when $s$ exceeds the target's effective dimension. Our main result concerns centered covariance risk normalized by full-data penalized loss. For balanced signed coordinate replicas, a strict sector inequality gives the sharp risk and all maximizing responses at every budget from the dimension to one below the row count. This holds for any nonzero positive semidefinite query. With the target and query fixed, the maximizing response space is unchanged across these budgets. For general designs, we characterize attainment of a leave-one-out envelope. For existing real equiang
    
[^161]: RefRoute：通过紧凑残差条件化与空间路由将条件化成本与参考图像解耦

    RefRoute: Decoupling Conditioning Cost from References via Compact Residual Conditioning and Spatial Routing

    [https://arxiv.org/abs/2610.07720](https://arxiv.org/abs/2610.07720)

    提出RefRoute框架，通过将低分辨率潜在token与全分辨率残差特征结合的紧凑残差条件化来减少参考token数量，并利用条件路由与注意力路由将参考token与目标区域对齐，从而显著降低多参考图像生成中的条件化与注意力开销。

    

    多参考图像生成要求在将多个主体组合成一个连贯场景的同时保持它们的外观。然而，现有的扩散Transformer通常将参考图像编码为密集的视觉token网格，并通过全局注意力将其联合处理，随着参考图像数量和分辨率的增长，条件化的开销变得越来越高。我们提出了RefRoute，一个通过两种互补机制同时解决参考表征成本和注意力开销的框架。紧凑残差条件化将低分辨率潜在token与从全分辨率像素中提取的轻量级残差特征相结合，在保留细粒度外观线索的同时减少了参考token的数量。条件路由与注意力路由将参考token与其被分配的目标区域对齐，并限制跨参考的交互，同时允许在区域边界之外进行选择性参考访问以服务于场景……

    arXiv:2610.07720v1 Announce Type: cross  Abstract: Multi-reference image generation requires preserving the appearance of multiple subjects while composing them into a coherent scene. However, existing diffusion transformers commonly encode references as dense visual token grids and jointly process them with global attention, making conditioning increasingly expensive as the number and resolution of references grow. We present RefRoute, a framework that addresses both reference representation cost and attention overhead through two complementary mechanisms. Compact residual conditioning combines low-resolution latent tokens with lightweight residual features extracted from full-resolution pixels, reducing reference token counts while retaining fine-grained appearance cues. Condition routing and attention routing align reference tokens with their assigned target regions and restrict cross-reference interactions, while allowing selective reference access beyond region boundaries for scen
    
[^162]: 次高斯数据上测度到测度Transformer的稳定性

    Stability of Measure-to-Measure Transformers on Sub-Gaussian Data

    [https://arxiv.org/abs/2610.07717](https://arxiv.org/abs/2610.07717)

    本文从数学上证明了Transformer将次高斯数据映射为次高斯输出且关于1-Wasserstein距离具有Hölder连续性，由此建立了经验近似下的误差传播估计，并揭示了交叉注意力机制均值场类似物的不同正则性与样本复杂度。

    

    Transformer在各个领域展现了令人瞩目的实证成功，但其理论基础仍相对欠缺。本工作对由Transformer定义的测度到测度算子进行了数学研究。我们证明Transformer将次高斯输入映射为次高斯输出，这保证了softmax算子任意长度复合的良定性。随后我们证明，在适当的次高斯输入空间上，Transformer关于1-Wasserstein距离具有Hölder连续性。这使我们能够建立关于Transformer在次高斯输入与其经验近似之间误差传播的估计。我们还研究了交叉注意力机制的均值场类似物，它是一个从概率测度对到单个概率测度的算子。我们证明交叉注意力表现出不同的Hölder正则性与样本复杂度。

    arXiv:2610.07717v1 Announce Type: cross  Abstract: Transformers have exhibited impressive empirical success across various domains, but their theoretical foundations remain less developed. This work constitutes a mathematical study of the measure-to-measure operators defined by transformers. We show that transformers map sub-Gaussian inputs to sub-Gaussian outputs; this ensures that taking arbitrary-length compositions of the softmax operator is well-defined. We then show that transformers are H\"older continuous with respect to the 1-Wasserstein distance on appropriate spaces of sub-Gaussian inputs. This allows us to establish estimates on the error propagation along a transformer between a sub-Gaussian input and its empirical approximation. We also study a mean-field analog of the cross-attention mechanism, which is an operator from a pair of probability measures to a single probability measure. We show that cross-attention exhibits different H\"older regularity and sample-complexity
    
[^163]: 神经运动层级网络：面向sEMG解码鲁棒泛化的生理学归纳偏置

    Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding

    [https://arxiv.org/abs/2610.07713](https://arxiv.org/abs/2610.07713)

    提出受神经运动层级结构启发的NHN网络，通过引入生理学归纳偏置学习紧凑的潜在神经运动状态，从而在跨用户、跨会话的sEMG解码中实现鲁棒泛化。

    

    表面肌电图（sEMG）为运动解码和人机交互提供了一种可穿戴、无创的神经肌肉活动接口。大规模人群解码仍然困难，原因在于sEMG与神经肌肉活动之间的关系在不同用户和会话之间存在差异，而任务相关的动态跨越多个通道和多个时间尺度。从任务标签学习波形到输出的映射，使得记录变异性与协调性运动活动之间的区分停留在隐式层面。我们提出了神经运动层级网络（NHN），它从任务监督中学习一个紧凑的潜在神经运动状态，以表征任务相关的神经肌肉协调。NHN通过受神经运动组织结构启发的层级机制来构建该潜在状态。它在保持相对强度的同时适应记录统计特性。其时空编码器采用参数高效的通道交互方式，并通过多…（摘要在此处截断）

    arXiv:2610.07713v1 Announce Type: new  Abstract: Surface electromyography (sEMG) provides a wearable, noninvasive interface to neuromuscular activity for movement decoding and human-computer interaction. Population-scale decoding remains difficult because the relationship between sEMG and neuromuscular activity varies across users and sessions, while task-relevant dynamics span channels and multiple timescales. Learning waveform-to-output mappings from task labels leaves the distinction between recording variability and coordinated motor activity implicit. We introduce the Neuromotor Hierarchy Network (NHN), which learns a compact latent neuromotor state from task supervision to represent task-relevant neuromuscular coordination. NHN constructs this latent state through a hierarchy inspired by neuromotor organization.It adapts recording statistics while preserving relative intensity.Its spatiotemporal encoder uses parameter-efficient channel interactions and modulates features with mul
    
[^164]: 基于数学不变量的拓扑神经网络用于分子与材料性质预测

    Mathematical Invariant-Enabled Topological Neural Networks for Molecular and Materials Property Prediction

    [https://arxiv.org/abs/2610.07712](https://arxiv.org/abs/2610.07712)

    本文提出了数学不变量赋能的拓扑神经网络（MITNN），通过整合拓扑学、谱理论、交换代数、微分几何等多个数学领域的多尺度不变量与拓扑神经架构，实现了更全面的分子与材料性质预测，并揭示精选的数学表示与神经架构配对组合的性能优于单个模型及所有组件的简单聚合。

    

    现有的分子与材料学习方法通常依赖于有限的结构表示集合，这可能仅能捕捉复杂三维结构的某些选定方面。在这里，我们提出了数学不变量赋能的拓扑神经网络，这是一个通过多个互补的数学视角来表示复杂结构并将其与拓扑神经架构相整合的框架。MITNN结合了来自拓扑学、谱理论、交换代数、微分几何和离散曲率的多尺度不变量，从同一系统中捕捉互补的结构信息。系统的不变量子集、架构子集以及集成分析表明，预测性能取决于数学表示与神经架构之间的配对方式，其中精选的组合优于单个模型以及所有可用组件的聚合。在蛋白质……（原文摘要在此处截断）

    arXiv:2610.07712v1 Announce Type: cross  Abstract: Existing molecular and materials learning approaches often rely on a limited set of structural representations, which may capture only selected aspects of complex three-dimensional structure. Here, we introduce mathematical invariant-enabled topological neural networks (MITNNs), a framework that represents complex structures through multiple complementary mathematical views and integrates them with topological neural architectures. MITNNs combine multiscale invariants from topology, spectral theory, commutative algebra, differential geometry, and discrete curvature, capturing complementary structural information from the same system. Systematic invariant-subset, architecture-subset, and ensemble analyses show that predictive performance depends on how mathematical representations and neural architectures are paired, with selected combinations outperforming individual models and the aggregation of all available components. Across protei
    
[^165]: WASD：基于Wasserstein距离的大语言模型知识蒸馏方法

    WASD: Wasserstein-based Knowledge Distillation for Large Language Models

    [https://arxiv.org/abs/2610.07706](https://arxiv.org/abs/2610.07706)

    该论文提出了WASD方法，通过由词元嵌入构建代价矩阵的Wasserstein距离，将词元级别语义信息融入大语言模型的知识蒸馏中，并借助Sinkhorn散度实现高效优化。

    

    自回归大语言模型（LLM）的能力迅速提升，但其不断增长的规模在推理时带来了巨大的计算和内存成本。知识蒸馏（KD）通过对齐离散概率分布，将大型教师模型的知识迁移到较小的学生模型中，提供了一种实用的解决方案。然而，现有的LLM知识蒸馏方法主要依赖于在每个词汇表索引处通过概率值来评估差异的散度度量，没有显式地利用词元级别的语义信息。我们提出了针对大语言模型的基于Wasserstein距离的知识蒸馏方法（WASD），该方法通过基于Wasserstein的距离并利用由词元嵌入导出的代价矩阵，将词元级别的语义信息融入蒸馏过程。为确保计算上的可处理性，我们采用Sinkhorn散度，并推导出一个梯度等价的目标函数，可以在不引入额外计算开销的情况下进行高效优化。

    arXiv:2610.07706v1 Announce Type: cross  Abstract: Autoregressive large language models (LLMs) have rapidly advanced in capability, but their increasing scale comes with substantial computational and memory costs at inference time. Knowledge distillation (KD) offers a practical solution by transferring knowledge from a large teacher model to a smaller student model via alignment of discrete probability distributions. However, existing KD methods for LLMs primarily rely on divergences that evaluate discrepancies through probability values at each vocabulary index, without explicitly leveraging token-level semantic information. We propose Wasserstein-based knowledge distillation (WASD) for LLMs, which incorporates token-level semantic information via the Wasserstein-based distance with a cost matrix derived from token embeddings. To ensure computational tractability, we adopt the Sinkhorn divergence and derive a gradient-equivalent objective that can be efficiently optimized without intr
    
[^166]: 基于反事实语义-社交世界模型的独立多智能体强化学习

    Independent Multi-Agent Reinforcement Learning with Counterfactual Semantic-Social World Models

    [https://arxiv.org/abs/2610.07704](https://arxiv.org/abs/2610.07704)

    该论文提出CASTLE框架，利用反事实动作条件化的语义-社交双世界模型，让完全去中心化的独立多智能体强化学习智能体能够前瞻性地比较候选动作后果，从而解决仅凭标量奖励信号无法判断回报不佳原因的模糊性问题。

    

    完全去中心化的多智能体强化学习（MARL），也称为独立学习，要求每个智能体仅依靠其本地信息和经验进行学习与行动，无需集中式评论家或智能体间通信。这种严格的信息结构使得传统的奖励信号变得模糊：较差的回报可能源于自身动作无效、队友响应不匹配，或对手的有效应对，但仅凭标量奖励无法揭示究竟是哪种原因所致。我们认为，智能体通过前瞻性地比较候选动作的后果，能够比仅从已实现回报中诊断失败更有效地学习。我们提出了 CASTLE（用于去中心化 MARL 中本地执行的反事实动作条件语义标记），这是一个离线训练、在线以上下文内方式进行引导的框架，包含两个互补的世界模型。其中一个局部动力学世界模型……（摘要内容截断于此）

    arXiv:2610.07704v1 Announce Type: cross  Abstract: Fully decentralized multi-agent reinforcement learning (MARL), also referred to as independent learning, requires each agent to learn and act using only its local information and experience, without a centralized critic or inter-agent communication. Such a stringent information structure renders the conventional reward signal ambiguous. A poor return may result from an ineffective ego action, an incompatible teammate response, or an effective opponent response, yet scalar rewards alone do not reveal which explanation is responsible. We argue that agents can learn more effectively by prospectively comparing the consequences of candidate actions rather than diagnosing failures only from realized returns. We introduce CASTLE (Counterfactual Action-conditioned Semantic Tokens for Local Execution in Decentralized MARL), an offline-training, online-in-context guidance framework with two complementary world models. A Local Dynamics World Mode
    
[^167]: 通过对抗性强化学习改进论辩挖掘的合成数据生成

    Improving Synthetic Data Generation for Argument Mining via Adversarial Reinforcement Learning

    [https://arxiv.org/abs/2610.07699](https://arxiv.org/abs/2610.07699)

    提出一种对抗性强化学习数据合成框架，通过生成器与判别器的对抗循环联合优化，同时提升论辩挖掘合成数据的结构准确性与多样性。

    

    论辩挖掘从根本上受到高质量结构标注数据集稀缺的限制。虽然大语言模型（LLMs）在合成数据生成方面已展现出潜力，但生成结构准确且足够多样的合成论辩挖掘数据仍然是一个具有挑战性的问题。为了解决这一问题，我们从一个新的视角重新审视论辩挖掘的合成数据生成，并提出了一种新颖的用于数据合成的对抗性强化学习框架。该框架在对抗循环中联合优化生成器与判别器，其中生成器生成结构化的论辩挖掘实例，判别器通过区分真实数据与合成候选数据来提供学习信号。这使得生成器能够通过对抗反馈逐步提升所生成论辩数据的结构准确性，同时保持多样性。大量实验表明，所提出的框架（摘要内容在此处截断）

    arXiv:2610.07699v1 Announce Type: cross  Abstract: Argument Mining (AM) is fundamentally constrained by the scarcity of high-quality structure-annotated datasets. While LLMs have shown promise in synthetic data generation, producing synthetic AM data that is both structurally accurate and sufficiently diverse remains a challenging problem. To address this problem, we revisit synthetic data generation for AM from a new perspective and propose a novel adversarial reinforcement learning framework for data synthesis. The proposed framework jointly optimizes the generator and the discriminator in an adversarial loop, in which the generator produces structured AM instances, and the discriminator provides learning signals by distinguishing real data from synthetic candidates. This enables the generator to progressively improve both the structural accuracy of generated argument data while maintaining diversity through adversarial feedback. Extensive experiments demonstrate that the proposed fr
    
[^168]: 自适应模型反演攻击揭示了隐私-鲁棒性权衡的普适性

    Adaptive Model Inversion Attacks Generalize a Privacy-Robustness Tradeoff

    [https://arxiv.org/abs/2610.07677](https://arxiv.org/abs/2610.07677)

    本文揭示了对攻击进行简单自适应改变后，现有隐私防御和标准训练技术的真实训练数据泄露率被低估达1.16至6.59倍，且评估结果受外部分类器特征基础影响，表明标准模型反演攻击评估可能将优化与测量失败误判为隐私保护。

    

    在本文中，我们证明了高分辨率模型反演攻击（MIA）的标准评估方法显著低估了训练数据的隐私泄露程度。在对攻击进行简单的自适应改变后，最先进的隐私防御方法、MixUp和对抗训练等标准训练技术以及无防御模型，在FaceScrub数据集上泄露训练图像的比率高出1.16至6.59倍，其中报告隐私保护效果最强的防御方法泄露增幅最大。我们进一步表明，测得的泄露程度取决于用于评估重建结果的外部分类器的特征基础：对于相同的重建图像，经过对抗训练的Inception评估器与标准Inception评估器识别目标身份的比率并不相同。我们的结果表明，标准的MIA评估可能会将优化和测量上的失败误判为隐私保护。这些被低估的泄露率还掩盖了一个更广泛的（摘要在此处截断）

    arXiv:2610.07677v1 Announce Type: new  Abstract: In this paper, we show that standard evaluations of high-resolution Model Inversion Attacks (MIAs) significantly underestimate training-data privacy leakage. State-of-the-art privacy defenses, standard training techniques such as MixUp and Adversarial Training, and undefended models all leak training images at rates 1.16 to 6.59 times higher on FaceScrub under simple adaptive changes to the attack, with the largest increases among defenses reporting the strongest privacy. We further show that measured leakage depends on the feature basis of the external classifier used to evaluate reconstructions: for the same reconstructed images, an adversarially trained Inception evaluator identifies the targeted identity at different rates than the standard Inception evaluator. Our results suggest that standard MIA evaluation can mistake optimization and measurement failures for privacy.   These underestimated leakage rates also concealed a broader r
    
[^169]: Transformer中的精确解体积与长度泛化

    Exact-Solution Volume and Length Generalization in Transformers

    [https://arxiv.org/abs/2610.07676](https://arxiv.org/abs/2610.07676)

    该论文提出归一化精确解体积（NESV）这一新指标来量化Transformer的长度泛化难度，证明精确解体积随输入长度衰减得越快，长度泛化就越困难，并为FIRST、MAJORITY、INDEX、PARITY四个任务建立了渐近界。

    

    关于Transformer表达能力的研究能够表明一个Transformer是否有能力解决给定任务，但对于所学习到的解能否泛化到更长的输入长度却几乎没有提供任何指示。我们通过归一化精确解体积（NESV）来研究这一问题：即在一个有界参数区域中，能够对每个长度为n的输入都实现精确解的参数所占的比例。对于具有log n缩放注意力的固定宽度单层Transformer，我们为四个任务建立了NESV的渐近界：FIRST（Θ(1)）、MAJORITY（Θ(1/(n log n))）、INDEX（Θ(1/n³)）和PARITY（0）。这些结果与先前的实证结果一致：精确解体积随输入长度衰减得越快，该任务的长度泛化就越困难。通过深入分析INDEX任务，我们的体积分析揭示了两种随n增长的误差来源。因此，我们研究了一种在结构上（摘要在此处截断）

    arXiv:2610.07676v1 Announce Type: cross  Abstract: Research on transformer expressivity shows whether a transformer is capable of solving a given task, but gives little indication of whether the solution, if learned, is generalizable to longer input lengths. We study this question through normalized exact-solution volume (NESV): the fraction of a bounded parameter region that achieves an exact solution on every input of length $n$. For fixed-width, single-layer transformers with $\log n$-scaled attention, we establish asymptotic bounds on NESV for four tasks: FIRST ($\Theta(1)$), MAJORITY ($\Theta(1/(n\log n))$), INDEX ($\Theta(1/n^3)$), and PARITY ($0$). These results are consistent with previous empirical results: the faster the exact-solution volume decays with input length, the harder it is to length-generalize on that task. Looking deeper into INDEX, our volume analysis reveals two error sources that grow with $n$. Consequently, we study a transformer model that would structurally
    
[^170]: CACHEFORGE：基于大语言模型引导的端到端生成式缓存替换策略，实现高性能与硬件效率

    CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency

    [https://arxiv.org/abs/2610.07668](https://arxiv.org/abs/2610.07668)

    CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。

    

    现代缓存替换设计已趋于饱和，因为它们受限于固定的表示结构、手工设计且基于启发式特征工程的预测器，或无法自行生成新决策逻辑的离线模仿模型。与此同时，缓存替换决策受到预取、抖动、空间局部性和访问类型行为之间因果交互的影响，形成了一个难以手动遍历的庞大设计空间。先前的方法通常依赖启发式规则、参数调优或对离线最优策略的模仿，只能捕捉相关性而无法合成新机制。因此，它们的性能提升往往陷入停滞，并在动态工作负载条件下出现过拟合。CACHEFORGE 是首个端到端演化缓存替换策略的框架，它将大语言模型嵌入到一个受控的硬件感知循环中。在每次迭代中，大语言模型提出新的 C++ 替换策略……

    arXiv:2610.07668v1 Announce Type: cross  Abstract: Modern cache replacement designs saturate because they operate within fixed representational structures, hand-crafted and heuristic based feature-engineered predictors, or offline imitation models that cannot generate new decision logic on their own. At the same time, replacement is shaped by the causal interaction of prefetching, thrashing, spatial locality, and access-type behavior, producing an enormous design space that is difficult to traverse manually. Prior approaches typically rely on heuristics, parameter tuning, or imitation of an offline optimal policy, capturing correlations rather than synthesizing new mechanisms. As a result, their performance gains often plateau and they overfit under dynamic workload conditions.   CACHEFORGE is the first framework to evolve cache-replacement policies end-to-end by embedding a large language model inside a governed hardware-aware loop. In each iteration, the LLM proposes new C++ replacem
    
[^171]: 面向用户行为模拟的工作流与提示词联合优化

    Joint Workflow and Prompt Optimization for User Behavior Simulation

    [https://arxiv.org/abs/2610.07663](https://arxiv.org/abs/2610.07663)

    SWORD框架基于角色化设计，仅依靠一个标量任务指标即可联合优化多智能体工作流拓扑与自然语言提示词，在用户行为模拟任务上显著超越仅优化提示词、仅优化工作流和分阶段优化的基线方法。

    

    用户行为模拟是指通过使用模拟智能体代替真实用户，对用户在信息系统中的交互进行计算建模。它可支持系统测试与评估、决策与预测，以及用户体验设计。现有的模拟器依赖于手工设计的规则或领域专业知识，而这些方法难以跨任务迁移。本文提出了SWORD（基于角色设计的模拟驱动工作流与提示词优化，Simulation-driven Workflow and Prompt Optimization with Role-based Design），这是一个联合优化多智能体工作流拓扑结构与自然语言提示词的框架。它仅由一个标量任务指标引导，无需领域初始化或任务特定的工程设计。实验结果表明，在受控的相同骨干模型比较条件下，SWORD 相对于仅优化提示词、仅优化工作流以及分阶段优化的基线方法均取得了统计学上显著的性能提升。与已发表的最强领域专用基线相比，SWORD 进一步……（原文摘要在此处截断）

    arXiv:2610.07663v1 Announce Type: cross  Abstract: User behavior simulation is the computational modeling of user interactions within information systems through the use of simulated agents in place of live users. It supports system testing and evaluation, decision-making and forecasting, and user experience design. Existing simulators rely on hand-crafted rules or domain expertise that transfers poorly across tasks. SWORD (Simulation-driven Workflow and Prompt Optimization with Role-based Design) is introduced as a framework that jointly optimizes multi-agent workflow topology and natural-language prompts. It is guided solely by a scalar task metric, without domain initialization or task-specific engineering. The experimental results demonstrate that SWORD achieves statistically significant gains over prompt-only, workflow-only, and staged-optimization baselines under a controlled, identical-backbone comparison. Against the strongest published domain-specific baseline, SWORD further i
    
[^172]: MS-ECG-FM：基于多源对比学习迈向更通用的心电图健康监测基础模型

    MS-ECG-FM: Towards a More Universal Electrocardiogram Foundation Model for Health Monitoring using Multi-source Contrastive Learning

    [https://arxiv.org/abs/2610.07662](https://arxiv.org/abs/2610.07662)

    MS-ECG-FM通过对齐心电图、超声心动图、放射学和出院报告等多种临床记录进行多源对比学习训练，突破了仅依赖解读报告作为监督的局限，在包括少导联配置在内的所有心电图检测任务上全面超越现有方法。

    

    心电图记录心脏的电活动，通过检测心功能异常来辅助诊断。心电图基础模型已展现出有前景的效果，但其局限性在于仅将心电图解读报告作为唯一的监督信号。由于解读报告只包含临床医生常规识别的波形信息子集，这限制了表示学习，使其忽略了心电图中更广泛的诊断信号。我们提出了一种新的心电图基础模型——MS-ECG-FM——它通过与多种不同类型的临床记录（包括心电图、超声心动图、放射学及出院报告）进行对比对齐来训练。我们在扩展的心电图检测基准集上对MS-ECG-FM进行了评估，结果表明其在心电图可检测的全部疾病范围内全面优于现有方法，包括在少导联配置下的表现。

    arXiv:2610.07662v1 Announce Type: new  Abstract: Electrocardiography (ECG) records the electrical activity of the heart, aiding diagnosis by detecting abnormalities in cardiac function. ECG foundation models have demonstrated promising results, but are limited by a reliance on ECG interpretation reports as their sole supervision. Because interpretation reports only capture the subset of waveform information routinely recognized by clinicians, this constrains representation learning to overlook the broader diagnostic signals present in ECG. We introduce a new ECG foundation model --- MS-ECG-FM --- that is trained through contrastive alignment to multiple distinct clinical note types, including ECG, echocardiography, radiology, and discharge reports. We evaluate MS-ECG-FM on an extended set of ECG detection benchmarks, showing that it comprehensively outperforms existing methods on the full span of conditions that ECG can detect, including in reduced-lead configurations. Different report
    
[^173]: 均匀离散扩散模型对于估计小有效支撑大小的分布是极小极大最优的

    Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size

    [https://arxiv.org/abs/2610.07655](https://arxiv.org/abs/2610.07655)

    该论文证明了均匀离散扩散模型在估计有效支撑较小的分布时达到极小极大最优，其统计误差由依赖于样本量的有效支撑大小而非环境空间规模决定。

    

    离散扩散模型已成为在离散乘积空间上进行生成建模的一种在实践中非常成功的框架，但其统计泛化性质仍未被充分理解。文本或生物序列等离散的真实世界数据，由于语义或物理约束，往往集中在天文数字般庞大的环境空间中的一小部分上；然而现有的理论误差界无法捕捉这种分布结构，而是随环境空间的大小进行缩放，导致误差界几乎空洞无用。我们针对均匀离散扩散——与掩码扩散并列为两大主流离散扩散范式之一——填补了这一空白，推导出由有效支撑大小 $s_n(P_0)$ 决定的统计保证，该度量是一种依赖于样本量的分布复杂度刻画。给定来自 $[K]$ 上未知数据分布 $P_0$ 的 $n$ 个独立同分布样本……

    arXiv:2610.07655v1 Announce Type: cross  Abstract: Discrete diffusion models have emerged as a practically successful framework for generative modeling on discrete product spaces, yet their statistical generalization properties remain poorly understood. Discrete real-world data such as text or biological sequences often concentrate on a small fraction of the astronomically large ambient space because of semantic or physical constraints, but existing bounds fail to capture this distributional structure and instead scale with the size of the ambient space, giving rise to almost vacuous error bounds. We address this gap for uniform discrete diffusion, one of the two dominant discrete diffusion paradigms alongside masking diffusion, by deriving statistical guarantees governed by the effective support size $s_n(P_0)$, a sample-size-dependent measure of distributional complexity. Given $n$ independent and identically distributed (i.i.d.) samples from an unknown data distribution $P_0$ on $[K
    
[^174]: 面向安全性的在线策略蒸馏是否存在后门风险？

    Does On-Policy Distillation for Safety Pose Backdoor Risks?

    [https://arxiv.org/abs/2610.07654](https://arxiv.org/abs/2610.07654)

    研究揭示了面向安全性的在线策略蒸馏（OPD）中一个被忽视的后门威胁：被植入后门的教师模型可将隐藏恶意行为传播给原本干净的学生模型，仅3%的投毒率即可使攻击成功率达70%，而增加训练轮数和常用的top-k KL方法会进一步加剧该风险。

    

    在线策略蒸馏作为一种将能力从教师模型迁移到学生模型的有效方式，正受到越来越多的关注。近期研究进一步探索将OPD作为提升大语言模型安全性的工具，并取得了可喜的成果。然而，这些方法通常假设教师模型和训练数据是可信的。本文揭示了面向安全性的OPD中一个被忽视的威胁：一个对齐了安全性但被植入后门的教师模型，可以将其隐藏的恶意行为传播给原本干净的学生模型。在我们的威胁模型下，低至3%的投毒率即可使蒸馏后学生模型的攻击成功率（ASR）高达70%。我们进一步发现两种训练选择会放大这一风险。首先，增加训练轮数即使在低投毒率下也会导致较高的ASR：仅需10个投毒样本，经过16轮训练后ASR即可达到67%。其次，常用的top-k KL方法可能加速后门……

    arXiv:2610.07654v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has attracted growing attention as an effective way to transfer capabilities from teacher models to student models. Recent studies further explore OPD as a tool for improving large language model safety with promising results. However, these approaches typically assume that the teacher and training data are trustworthy. In this paper, we uncover an overlooked threat to OPD for safety: a safety-aligned but backdoored teacher can propagate its hidden malicious behavior to an initially clean student. Under our threat model, a poisoning rate as low as 3% results in an attack success rate (ASR) of up to 70% on the distilled student. We further identify two training choices that can amplify this risk. First, increasing the number of training epochs can lead to high ASR even at low poisoning rates. With only 10 poisoned samples, ASR reaches 67% after 16 epochs. Second, the commonly used top-k KL can accelerate bac
    
[^175]: 迈向可解释国际象棋战术的自动合成

    Towards the Automatic Synthesis of Interpretable Chess Tactics

    [https://arxiv.org/abs/2610.07640](https://arxiv.org/abs/2610.07640)

    本文提出一种受国际象棋战术启发、由归纳逻辑编程系统PAL所学模式推导而来的符号化子策略模型，通过融入领域知识提升可解释性，并提出散度度量评估方法，其合成的战术组合能给出与人类初学者棋力相当的走法建议。

    

    最先进的强化学习智能体能够在国际象棋、围棋和《星际争霸II》等游戏中超越人类专家。这些智能体并非仅仅利用其数字硬件在反应和计算速度上快于人类，而是采用了更优的策略从而赢得更多胜利。解读这些策略将为人类玩家提供提高棋艺的宝贵见解。在这项初步工作中，我们提出了一种用于下国际象棋的符号化子策略模型。受国际象棋战术的启发，我们的模型尝试融入领域知识以提升可解释性。我们改造了由名为PAL的归纳逻辑编程系统所学习到的模式来推导该模型。我们提出了一种散度度量方法，用于将模型与随机基线进行对比评估，并发现了一组战术，能够给出与人类初学者棋力相近的走法建议。最后，我们提出了一种计算评估方案。

    arXiv:2610.07640v1 Announce Type: new  Abstract: State-of-the-art reinforcement learning agents are capable of outperforming human experts at games like chess, Go and StarCraft II. These agents do not simply take advantage of their digital hardware in being able to react and calculate faster than humans, but employ better strategies that lead to more victories. Interpreting these strategies would give human players valuable insight into how to improve their play. In this preliminary work, we propose a symbolic sub-policy model for playing chess. Inspired by chess tactics, our model attempts to incorporate domain knowledge to improve interpretability. We adapt patterns learned by an inductive logic programming system called PAL to derive our model. We contribute a divergence metric to evaluate our model against a random baseline, and find a set of tactics that is able to suggest moves of similar playing strength to a human beginner. Finally, we propose a computational evaluation scheme 
    
[^176]: 学习复杂游戏策略的可解释表示

    Learning Explainable Representations of Complex Game-playing Strategies

    [https://arxiv.org/abs/2610.07638](https://arxiv.org/abs/2610.07638)

    本文提出一种类似人类认知的方法，训练强化学习智能体将学到的游戏策略合成为基于动作序列的可执行程序，从而获得可解释的策略表示，并在国际象棋和网格环境任务中验证了其有效性。

    

    作为学习玩复杂游戏的一部分，人类玩家会形成与游戏规则相一致的游戏概念和策略抽象，以提高自己的表现。这些概念被用来解释其他玩家的行为，并指导自己在游戏中的行动。理解其他玩家的策略是这种提升的关键部分，但这需要时间和精力。在本文中，我们提出了一种类似人类认知的策略，用于训练强化学习智能体将学习到的策略和方针合成为基于游戏动作序列的可执行程序。我们提出了自动学习此类程序的方法，用于下国际象棋以及在基于网格的环境中解决任务。我们证明，学习到的策略能够产生有效的动作，并且可以从游戏对局数据中学习得到。

    arXiv:2610.07638v1 Announce Type: new  Abstract: As part of learning to play complex games, human players develop develop abstractions for concepts and strategies of gameplay consistent with game rules to improve their performance. These concepts are applied to explain other players' actions, and to inform their own actions in-game. Understanding other players' strategies is a crucial part of such improvement, but requires time and effort. In this paper, we propose a strategy similar to human cognition for training RL agents to synthesize learned strategies and policies as executable procedures based on sequences of gameplay actions. We present methods to automatically learn such programs to play chess and to solve tasks in a grid-based environment. We show that the learned strategies produce effective actions, and can be learned from gameplay data.
    
[^177]: 逐元素独立同分布重尾数据上经验风险最小化的渐近分析

    Asymptotic Analysis of Empirical Risk Minimization on Entry-wise i.i.d. Heavy-Tailed Data

    [https://arxiv.org/abs/2610.07637](https://arxiv.org/abs/2610.07637)

    本文通过引入函数序参量并运用复制方法，首次在比例高维极限下精确刻画了对称α-稳定重尾数据上线性回归经验风险最小化的泛化误差，并建立了重尾普适性定律与相应的标度律。

    

    许多现实世界的数据集出现异常大值的频率远超高斯模型的预测。重尾分布能够刻画这种现象，但在重尾分布下评估学习性能仍然具有挑战性，因为稀有的大幅特征元素即使在高维情形下也保持着不可忽略的影响。即使在带有逐元素独立同分布对称 $\alpha$-稳定数据的线性回归经验风险最小化这一经典设定中，预测性能的精确渐近刻画也一直缺失。在这项工作中，我们引入了一个函数序参量来描述与每个系数相关的随机有效问题。利用复制方法，我们在样本量与特征维度以固定比例发散的比例高维极限下完全刻画了泛化误差。此外，该分析建立了一个重尾普适性定律，以及将典型误差与标度律相联系的规律（摘要原文在此处被截断）。

    arXiv:2610.07637v1 Announce Type: cross  Abstract: Many real-world datasets exhibit unusually large values far more frequently than predicted by Gaussian models. Heavy-tailed distributions capture this behavior, yet evaluating learning performance under them remains challenging because rare, large feature entries retain non-vanishing effects even in high dimensions. Even in the canonical setting of empirical risk minimization for linear regression with entry-wise i.i.d. symmetric $\alpha$-stable data, a precise asymptotic characterization of prediction has been lacking. In this work, we introduce a functional order parameter that describes the random effective problem associated with each coefficient. Using the replica method, we fully characterize the generalization error in the proportional high-dimensional limit where the sample size and feature dimension diverge at a fixed ratio. Additionally, this analysis establishes a heavy-tail universality law, scaling laws relating typical er
    
[^178]: 用于分布外图学习的互补监督与自监督表示

    Complementary Supervised and Self-Supervised Representations for Out-of-Distribution Graph Learning

    [https://arxiv.org/abs/2610.07628](https://arxiv.org/abs/2610.07628)

    该论文提出Co-Train和Dual-Space Retrieval两个骨干无关的框架，将自监督表示作为监督学习的互补信号，分别在训练和推理阶段进行融合，从而提升图神经网络在分布外节点分类上的泛化能力。

    

    分布外（OOD）泛化对于图神经网络（GNN）而言仍然是一个挑战，因为图的分布在时间和领域之间可能存在显著差异。监督式和自监督式的图表示学习由不同的目标引导，并为图表示提供了不同的视角。在本工作中，我们研究了自监督表示（SSL）是否能够提供互补信号，以改进监督式的分布外节点分类。我们提出了两个与骨干网络无关的框架，在学习和预测的不同阶段利用这类信息。Co-Train 联合学习监督表示和自监督表示，并在训练过程中自适应地整合它们；而 Dual-Space Retrieval 在两个表示空间中执行非参数化预测，并在推理时通过置信度感知的融合方式将两者的预测结果结合起来。监督编码器和自监督编码器分别独立参数化，且无需共享……

    arXiv:2610.07628v1 Announce Type: new  Abstract: Out-of-distribution (OOD) generalization remains challenging for graph neural networks (GNNs), as graph distributions can vary substantially across time and domains. Supervised and self-supervised graph representation learning are guided by distinct objectives and offer different perspectives on graph representations. In this work, we study whether self-supervised representations (SSL) can provide complementary signals to improve supervised OOD node classification. We develop two backbone-agnostic frameworks that exploit such information at different stages of learning and prediction. Co-Train jointly learns supervised and SSL representations and adaptively integrates them during training, while Dual-Space Retrieval performs non-parametric prediction in the two representation spaces and combines their predictions through confidence-aware fusion at inference time. The supervised and SSL encoders are separately parameterized and need not s
    
[^179]: 无状态语言智能体：扩展长时程自动化研究

    Stateless Language Agents: Scaling Long-Horizon Automated Research

    [https://arxiv.org/abs/2610.07625](https://arxiv.org/abs/2610.07625)

    提出无状态语言智能体（SLA）框架，通过“有状态搜索、无状态智能体”的原则——由框架统一管理研究状态并为每次调用重建角色化上下文——来解决长时程自动化研究中智能体重放冗长历史、重复劳动和过早停止实验等失败模式。

    

    自动化研究系统越来越多地在长时程上运行LLM智能体，但更多的推理本身并不能带来更多的研究进展：智能体会重放不断增长的历史记录、重复彼此的工作，或者在token持续消耗的同时停止实验。然而，大多数评估使用较短的预算或很快饱和的基准测试，使得这些失败模式未曾得到检验。我们将这些失败追溯到两个关键选择：研究状态存储在哪里，以及由谁决定下一步尝试什么。我们提出了无状态语言智能体，其建立在“有状态搜索、无状态智能体”的原则之上：没有任何智能体在多次调用之间携带其对话历史；相反，由框架拥有研究状态（候选解决方案和测量结果），并为每次调用重建一个全新的、特定角色的上下文。每个智能体所看到的内容由此成为一种显式的设计选择，而不是随运行而不断增长的历史。我们在SLA框架中实现了这一原则，

    arXiv:2610.07625v1 Announce Type: cross  Abstract: Automated research systems increasingly run LLM agents over long horizons, but more inference does not by itself produce more progress: agents replay growing histories, duplicate one another's work, or stop experimenting while token consumption continues. Yet most evaluations use short budgets or benchmarks that saturate early, leaving these failure modes untested. We trace these failures to two choices: where research state lives and who decides what to try next. We introduce Stateless Language Agents (SLAs), built on the principle of stateful search with stateless agents: no agent carries its conversation across invocations; instead, the harness owns the research state (candidate solutions and measured outcomes) and reconstructs a fresh and role-specific context for every invocation. What each agent sees becomes an explicit design choice rather than a history that grows with the run. We implement this principle in the SLA framework, 
    
[^180]: 超越 $T^{2/3}$ 的序贯校准问题的显式渐近界

    Explicit Asymptotic Bounds for Sequential Calibration Beyond $T^{2/3}$

    [https://arxiv.org/abs/2610.07623](https://arxiv.org/abs/2610.07623)

    该论文提出新的两阶段递归标记策略并改进归约方法，首次为序贯校准问题建立了超越 $T^{2/3}$ 的显式渐近界 $O(T^{0.662942288})$。

    

    当预测概率与经验结果频率相匹配时，概率预测被称为校准的：在被赋予概率 $p$ 的事件中，我们希望正结果的占比接近 $p$。我们研究二元结果的序贯预测问题。Foster 和 Vohra 建立的关于期望累积 $\ell_1$ 校准误差的经典 $O(T^{2/3})$ 界保持了二十多年，直到 Dagan 等人将指数 $2/3$ 降低了一个未具体指明的常数。我们为“符号保持-复用”博弈建立了一种新的两阶段递归标记策略，对于所有空间和时间的选择都能得到 $O(n^{\alpha}t^\beta)$ 的界。随后，我们通过修改 Dagan 等人的等价性，使其仅使用 $O(\log T)$ 个“符号保持-复用”博弈实例，从而锐化了从符号保持上界到校准的归约。这使我们能够建立一个显式的界 $O(T^{0.662942288})$。

    arXiv:2610.07623v1 Announce Type: cross  Abstract: Probability forecasts are calibrated when predicted probabilities match empirical outcome frequencies: among events assigned a probability $p$, we'd hope that the fraction of positive outcomes is close to $p$. We study the problem of sequential forecasting of binary outcomes. The classical $O(T^{2/3})$ bound on expected cumulative $\ell_1$-calibration error established by Foster and Vohra stood for over two decades until Dagan et al. reduced the exponent $2/3$ by an unspecified constant.   We establish a new two-phase recursive labeling strategy for the sign-preservation-with-reuse game that yields the bound $O(n^{\alpha}t^\beta)$ for all choices of space and time. We then sharpen the reduction from upper bounds on sign preservation to calibration by modifying the equivalence of Dagan et al. to use only $O(\log T)$ instances of the sign-preservation-with-reuse game. This lets us establish an explicit bound of $O(T^{0.662942288})$, the 
    
[^181]: 先探索，后确定：基于语言模型的测量高效科学定律发现

    Explore, Then Commit: Measurement-Efficient Scientific Law Discovery with Language Models

    [https://arxiv.org/abs/2610.07620](https://arxiv.org/abs/2610.07620)

    该论文提出了一种“先探索后确定”协议，通过语言模型提出假设、程序化规划器高效收集测量，将科学定律发现所需的测量次数最多减少约5倍，并将误差显著降低一个数量级以上。

    

    科学定律发现需要选择测量并将证据转化为控制方程。我们评估了一种“先探索后确定”协议，其中大型语言模型提出假设，程序化规划器收集测量数据，然后通过全新的提示词从固定观测中合成最终定律。该协议结合了结构化探测、自动数值诊断、受限测量批次以及可选的解释器访问。在576次NewtonBench试验中，我们使用GPT-4.1-mini和中难度GPT-4.1复现版本，在12个物理模块上比较了八种配置。在中难度任务上，启用解释器的规划器在GPT-4.1-mini上每次试验使用8.6次测量（对比22.5次），在GPT-4.1上使用8.9次（对比43.0次）。它们基于量级的均方根对数误差分别从2.514降至0.202，从0.626降至0.149。额外的审计在覆盖率中保留了不完整和无效的提交。

    arXiv:2610.07620v1 Announce Type: new  Abstract: Scientific law discovery requires selecting measurements and converting evidence into a governing equation. We evaluate an explore-then-commit protocol in which a large language model proposes hypotheses, a programmatic planner gathers measurements, and a fresh prompt synthesizes the final law from fixed observations. The protocol combines structured probes, automatic numerical diagnostics, restricted measurement batches, and optional interpreter access. Across 576 NewtonBench trials, we compare eight configurations on 12 physics modules using GPT-4.1-mini and a medium-difficulty GPT-4.1 replication. On medium tasks, interpreter-enabled planners use 8.6 versus 22.5 measurements per trial for GPT-4.1-mini and 8.9 versus 43.0 for GPT-4.1. Their mean magnitude-based root-mean-squared logarithmic error falls from 2.514 to 0.202 and from 0.626 to 0.149, respectively. An additional audit retains incomplete and invalid submissions in a coverage
    
[^182]: AFA-BANDIT：预算约束下可证明近优的在线多特征分类

    AFA-BANDIT: Provably Near-Optimal Online Multi-Feature Classification Under Budget Constraints

    [https://arxiv.org/abs/2610.07615](https://arxiv.org/abs/2610.07615)

    该论文将预算约束下的在线主动特征获取问题建模为组合式背包老虎机（BwK）问题，并借助基数感知的置信界，首次实现了可证明的近优遗憾上界。

    

    主动特征获取（AFA）是一种分类问题，其中智能体在预测每个样本的标签之前，需要决定获取哪些代价高昂的特征。与批量AFA（在完全观测的数据上离线训练固定的策略和分类器）不同，在线AFA会随着样本的到达，根据揭示的标签更新其预测器。现有的在线方法要么使用缺乏性能保证的深度强化学习（RL），要么最大化成本调整后的奖励，而不是强制执行全局预算。我们将在线AFA表述为一个耦合特征获取与预测的组合式带背包约束的老虎机（BwK）问题。与以往基于老虎机的AFA方法和经典BwK不同，我们的设定具有组合复杂性、随时间演化的奖励、全局预算以及结构化的辅助信息。在该框架下，我们利用基数感知的置信界和子集更新结构，获得了优于标准BwK界的遗憾上界。为了避免……

    arXiv:2610.07615v1 Announce Type: new  Abstract: Active Feature Acquisition (AFA) is a classification problem in which an agent decides which costly features to acquire before predicting each sample's label. Unlike batch AFA, which trains a fixed policy and classifier offline on fully observed data, online AFA updates its predictor from revealed labels as samples arrive. Existing online methods either use deep reinforcement learning (RL) without performance guarantees or maximize cost-adjusted reward rather than enforce a global budget. We formulate online AFA as a combinatorial Bandits with Knapsacks (BwK) problem that couples acquisition and prediction. Unlike prior bandit-based AFA and classical BwK, our setting has combinatorial complexity, evolving rewards, a global budget, and structured side information. We obtain an improved regret upper bound over standard BwK bounds in this framework, leveraging a cardinality-aware confidence bound and the subset update structure. To avoid an
    
[^183]: 从点云学习抓取目标定位以实现液压起重机原木堆清理

    Learning Grasp Targeting from Point Clouds for Log Pile Clearing on a Hydraulic Crane

    [https://arxiv.org/abs/2610.07613](https://arxiv.org/abs/2610.07613)

    该论文提出了一种从非分割点云中学习抓取点选择、抓取深度与抓爪方向的策略，结合行为克隆与强化学习训练，成功部署于拖车式液压林业起重机上完成原木堆清理，现场试验中放置成功率达93.8%。

    

    在锯木厂堆场中，原木装载机通过一系列成捆抓取来清理密集的原木堆：数百根原木相互接触，每次移除都会改变下一次抓取时可用的木堆状态。一个学习得到的策略根据非分割的点云选择抓爪的放置位置和方向，并运行在拖车安装的液压林业起重机上。该策略对在哪个观测点进行抓取进行分类，并预测该处的深度和抓爪方向。同一网络输出支持行为克隆（BC）、强化学习（RL）和部署。BC从成功的堆顶演示中学习；RL通过微调克隆策略（BC→RL）或从头训练来探索改进。在仿真中，BC成功清理了100个包含200根原木的木堆中的98个，而BC→RL提高了负载稳定性。十二次现场试验通过完整的抓取-运输-放置循环比较了几何启发式方法、从头训练的RL、BC和BC→RL。BC和BC→RL的放置成功率达到93.8%。

    arXiv:2610.07613v1 Announce Type: cross  Abstract: In mill yards, log loaders clear dense piles by a sequence of bundle grasps: hundreds of logs rest in contact, and each removal changes the pile available to the next grasp. A learned policy chooses where to place and orient the grapple from unsegmented point clouds and runs on a trailer-mounted hydraulic forestry crane. The policy classifies at which observed point to grasp and predicts depth and grapple orientation there. The same network outputs support behavior cloning (BC), reinforcement learning (RL), and deployment. BC learns from successful top-of-pile demonstrations; RL explores for improvements by fine-tuning the cloned policy (BC$\to$RL) or by training from scratch. In simulation, BC clears 98 of 100 piles of 200 logs, while BC$\to$RL improves load stability. Twelve field trials compare a geometric heuristic, RL from scratch, BC, and BC$\to$RL through complete grasp-transport-deposit cycles. BC and BC$\to$RL deposit 93.8% an
    
[^184]: 轮毂收容离群样本，辐条安放内点：面向双重失配半监督学习的均匀潜在空间构建

    Hub for Outliers, Spokes for Inliers: Uniform Latent Space Construction for Dual-Mismatched Semi-Supervised Learning

    [https://arxiv.org/abs/2610.07610](https://arxiv.org/abs/2610.07610)

    提出轮毂-辐条潜在空间几何结构，让已知类围绕中心轮毂均匀分布、未知类样本被安置于轮毂锚定的低证据区域，以解决半监督学习中类别分布与标签空间的双重失配问题。

    

    半监督学习通常假设有标签数据与无标签数据共享相同的类别分布和标签空间。然而，这一假设常常被打破：无标签数据可能类别不平衡，且包含未知类别的样本，从而导致类别分布和标签空间的双重失配。这种双重失配会导致多数类主导潜在空间，同时未知类别样本被过度自信地错误分类，进而损害特征的可区分性和伪标签的质量。为解决这一问题，我们提出了一种轮毂-辐条（hub-spoke）潜在空间几何结构：已知类别围绕中心轮毂均匀分布，每个类别在其原型周围形成紧凑的簇，而轮毂则为专门设计的低证据区域提供锚点，用于安置高不确定性的未知类别样本。结合基于证据的分类器，该几何结构通过缓解上述问题，最终提升了特征可区分性并改善了不确定性的分离效果。

    arXiv:2610.07610v1 Announce Type: new  Abstract: Semi-supervised learning typically assumes that labeled and unlabeled data share an identical class distribution and label space. However, this setting is often violated: unlabeled data may be imbalanced and contain unknown class samples, causing mismatches in both class distribution and label space. Such dual mismatch leads to majority classes dominating the latent space and unknown class samples being overconfidently misclassified, degrading feature discriminability and pseudo-label quality. To address this, we propose a hub-spoke latent geometry, where known classes are uniformly distributed around a central hub and each class forms compact clusters around its prototype, while the hub provides an anchor for a low-evidence region specifically designed for high-uncertainty unknown class samples. Integrated with an evidence-based classifier, this geometry ultimately enhances feature discriminability and uncertainty separation by mitigati
    
[^185]: 蛋白质语言模型中的线性适应度子空间实现样本高效的定向进化

    Linear Fitness Subspace in Protein Language Models Enables Sample-Efficient Directed Evolution

    [https://arxiv.org/abs/2610.07607](https://arxiv.org/abs/2610.07607)

    提出线性适应度子空间（LFS）假设并引入子空间引导进化搜索（SGES），通过在蛋白质语言模型突变引起的残基级表示变化中寻找与实验测定相关的紧凑方向集合，使适应度变化从少量标注样本中线性可获取，从而实现样本高效的模型引导定向进化。

    

    模型引导的定向进化旨在有限的真值评估预算下识别高适应度的蛋白质变体。蛋白质语言模型（PLM）为此类任务提供了丰富的表示，但任务无关的零样本评分可能与目标实验测定不对齐，而在高维嵌入空间中进行监督搜索会使代理建模和不确定性估计的样本效率低下。我们提出线性适应度子空间（LFS）假设：在突变引起的残基级表示变化中，存在一组紧凑的、与特定实验测定相关的方向集合，使得适应度变化可以从少量标注变体中被线性地获取。这是一个局部的、可通过监督恢复的论断，而非声称蛋白质适应度景观或PLM的全局几何结构是普遍线性的。基于这一观察，我们引入了子空间引导进化搜索（SGES），该方法从少量初始样本中估计LFS并进行代理建模……

    arXiv:2610.07607v1 Announce Type: cross  Abstract: Model-guided directed evolution seeks to identify high-fitness protein variants under limited oracle budgets. Protein language models (PLMs) provide rich representations for this task, but task-agnostic zero-shot scores can be misaligned with a target assay, while supervised search in high-dimensional embedding spaces can make surrogate modeling and uncertainty estimation sample-inefficient. We propose the Linear Fitness Subspace (LFS) hypothesis: within mutation-induced residue-level representation changes, a compact, assay-specific set of directions makes fitness variation linearly accessible from few labeled variants. This is a local, supervision-recoverable statement rather than a claim that protein fitness landscapes or global PLM geometry are universally linear. Building on this observation, we introduce Subspace-Guided Evolutionary Search (SGES), which estimates an LFS from a small initial sample and performs surrogate modeling,
    
[^186]: 通过Monge-生长对实现的Hellinger-Kantorovich梯度流的神经JKO格式

    A Neural JKO Scheme for Hellinger-Kantorovich Gradient Flows via Monge-Growth Pairs

    [https://arxiv.org/abs/2610.07602](https://arxiv.org/abs/2610.07602)

    该论文提出了一种基于Monge-生长对的无网格神经JKO格式，在Hellinger-Kantorovich非平衡最优传输几何中于单个变分步骤内联合处理空间再分布与质量产生/损失，为对流-反应-扩散梯度流建立了离散能量耗散、极小元存在性、正性与正则性等理论保证。

    

    我们为在非平衡最优传输的Hellinger-Kantorovich（HK）几何中具有梯度流结构的对流-反应-扩散方程开发了一种无网格神经JKO格式。每次更新由一个空间映射和一个质量变化因子进行参数化，使得空间再分布与局部质量的产生或损失可以在单个变分步骤中被联合处理。二者的锥作用从上方界定HK距离的平方，通过与恒等对的比较，给出了离散能量耗散的充分条件。当源项与某个极小元具有正密度时，在所有容许对上极小化该对目标函数即可恢复精确的JKO极小值。我们建立了JKO极小元的存在性与质量界，并在附加假设下获得了正性与正则性，同时得到离散Euler-Lagrange方程和度量耗散恒等式。自洽的化学势是……（原文摘要在此处截断）

    arXiv:2610.07602v1 Announce Type: cross  Abstract: We develop a mesh-free neural JKO scheme for advection-reaction-diffusion equations with a gradient-flow structure in the Hellinger-Kantorovich (HK) geometry of unbalanced optimal transport. Each update is parametrized by a spatial map and a mass-changing factor, allowing spatial redistribution and local mass creation or loss to be treated jointly within a single variational step. Their cone action bounds the squared HK distance from above, yielding a sufficient condition for discrete energy dissipation through comparison with the identity pair. Minimizing the pair objective over all admissible pairs recovers the exact JKO minimum when the source and a minimizer have positive densities. We establish existence and mass bounds for JKO minimizers and, under additional assumptions, obtain positivity and regularity together with a discrete Euler-Lagrange equation and a metric-dissipation identity. The self-consistent chemical potential is t
    
[^187]: 面向世界模型鲁棒决策的潜在扰动建模

    Modeling Latent Disturbances for Robust Decision-Making in World Models

    [https://arxiv.org/abs/2610.07599](https://arxiv.org/abs/2610.07599)

    本文提出将潜在空间扰动建模为对习得潜在动力学的扰动，使其引发悲观但合理的状态转移，从而在世界模型的潜在空间中实现鲁棒决策。

    

    本文研究了世界模型（WMs）潜在空间中的鲁棒决策问题。鲁棒优化是一种数学框架，在给定明确定义的动力学和具有物理意义的扰动的情况下，机器人可以选择即使在最坏情况扰动下仍然有效的动作。然而，将这一原则应用于世界模型的习得潜在空间带来了一项根本性挑战：由于世界模型的状态空间和动力学完全是从高维观测中学习推断得到的，如何定义能够忠实表征底层系统不确定性的潜在空间扰动尚不清楚。我们的核心思想是将潜在空间扰动建模为对习得潜在动力学的扰动，该扰动会引发悲观但合理的状态转移。具体而言，我们通过结合一种能够捕捉合理转移的动力学感知相似性度量与分布外（原文在此处截断）……

    arXiv:2610.07599v1 Announce Type: cross  Abstract: In this paper, we study robust decision-making in the latent space of world models (WMs). Robust optimization is a mathematical framework where, given explicitly specified dynamics and physically meaningful disturbances, a robot can select actions that remain effective even under worst-case disturbances. However, applying this principle to the learned latent space of WMs introduces a fundamental challenge: because WMs have fully learned state spaces and dynamics inferred from high-dimensional observations, it is unclear how to define latent-space disturbances that faithfully represent uncertainty in the underlying system. Our key idea is to model a latent-space disturbance as a perturbation to the learned latent dynamics that induces pessimistic but plausible transitions. Specifically, we construct a set of plausible latent dynamics by combining a dynamics-aware similarity metric that captures plausible transitions with out-of-distribu
    
[^188]: 机器人并非其描述：面向形态感知策略表征鲁棒性的 GaugeBench 基准

    The Robot Is Not Its Description: GaugeBench for Representation Robustness in Morphology-Aware Policies

    [https://arxiv.org/abs/2610.07597](https://arxiv.org/abs/2610.07597)

    提出 GaugeBench 基准，发现在机器人本体不变、仅将其描述改写为物理等效的约定时，形态感知策略性能从 4030.6 骤降至 51.6，揭示了现有策略对描述约定极度脆弱、并未真正泛化到机器人形态本身。

    

    机器人描述的作用不仅是规定一个物理机制：它还编码了诸多任意约定，例如关节轴方向、关节角零点，以及连杆和关节的顺序与名称。形态感知策略使用由这些描述构建的接口，然而跨具身评估通常在更换机器人的同时保持这些约定不变。这就留下了一个简单却未解答的问题：当机器人保持不变、但其描述发生变化时，行为能否延续？GaugeBench 通过在物理等效的约定下重写固定机制来隔离这一情形，验证其物理特性与策略接口得到保留，然后评估相同的策略权重。结果是严峻的：三个 MetaMorph 策略在 80 个熟悉的机器人上得分为 4030.6，但当这些相同的机器人被等效地重新描述时，得分骤降至 51.6，而 98 个真正未见过的机器人得分为 1489.6。因此，一种新的描述可能比（更换机器人）更具破坏性……

    arXiv:2610.07597v1 Announce Type: cross  Abstract: A robot description does more than specify a physical mechanism: it also encodes arbitrary conventions, such as joint-axis direction, joint-angle zero, and the order and names of links and joints. Morphology-aware policies consume interfaces built from these descriptions, yet cross-embodiment evaluation typically changes the robot while keeping those conventions fixed. This leaves a simple question unanswered: does behavior survive when the robot stays fixed but its description changes? GaugeBench isolates this case by rewriting a fixed mechanism under physically equivalent conventions, verifying that its physics and policy interface are preserved, and then evaluating the same policy weights. The result is stark: three MetaMorph policies score 4030.6 on 80 familiar robots, but only 51.6 when those same robots are equivalently re-described, while 98 genuinely held-out robots score 1489.6. A new description can therefore be more damaging
    
[^189]: BiGym 2.0：人形机器人家庭操作中学习型与智能体开发策略的基准测试

    BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation

    [https://arxiv.org/abs/2610.07594](https://arxiv.org/abs/2610.07594)

    BiGym 2.0是一个针对宇树G1人形机器人20个家庭操作任务的全身控制基准测试平台，实验表明视觉-语言-动作微调方法总体表现最佳，而编码智能体开发的程序优于所有演示驱动的强化学习基线。

    

    人形机器人家庭操作需要手臂在身体保持平衡、迈步和改变姿势的同时进行动作。我们提出了BiGym 2.0，这是BiGym针对宇树G1机器人的适配版本，涵盖20个家庭任务，使用统一的全身控制器进行演示和评估。该测试套件为每个任务提供60个原生人类虚拟现实演示，包含同步的多相机视角和全身执行记录。我们对视觉-语言-动作微调、模仿学习、演示驱动的强化学习，以及给定在线强化学习交互预算的冷启动编码智能体进行了基准测试。在为每种方法提供相同的机载视角、本体感觉和全身控制器的条件下，视觉-语言-动作微调在九项任务的平均值上表现最佳，而智能体开发的程序在该平均值上超越了所有演示驱动的强化学习基线，并在双手抓取任务上处于领先地位。跨工作空间堆叠任务仍未被解决，π₀.

    arXiv:2610.07594v1 Announce Type: cross  Abstract: Humanoid household manipulation requires the arms to act while the body balances, steps and changes posture. We present BiGym 2.0, an adaptation of BiGym for the Unitree G1 across 20 household tasks using a unified whole-body controller for demonstration and evaluation. The suite provides 60 native human virtual-reality demonstrations per task with synchronised multi-camera views and full-body execution records. We benchmark vision-language-action fine-tuning, imitation learning, demo-driven reinforcement learning, and cold-start coding agents given the interaction budget of online reinforcement learning. With the same onboard views, proprioception and whole-body controller for every method, vision-language-action fine-tuning has the highest nine-task mean, and agent-developed programs outperform every demo-driven reinforcement learning baseline on this mean and lead on bimanual reaching. Cross-workspace stacking remains open, $\pi_{0.
    
[^190]: 循环环路Transformer

    Recurrent Looped Transformer

    [https://arxiv.org/abs/2610.07591](https://arxiv.org/abs/2610.07591)

    提出循环环路Transformer（RLT），通过将层分配给并行因果编码器和循环解码器，使计算路径随序列长度增长而每个token成本保持固定，在状态跟踪和算法泛化任务上大幅超越固定深度的标准Transformer。

    

    状态跟踪需要对每个输入进行更新，但Transformer应用于每个token的深度是固定的，与序列长度无关。我们提出了循环环路Transformer（Recurrent Looped Transformer, RLT），它将模型层分配给一个并行因果编码器和一个循环解码器。在每个token处，解码器将编码器输出与上一个token的最终解码器状态合并，因此计算路径随序列长度增长，而每个token的计算成本保持固定。在六个算法任务上，我们在三个随机种子下比较了八层模型的五种分配方式与一个八层Transformer。在最多40位数据上训练后，两种RLT分配方式在所有种子中均能以100%的准确率将奇偶性判断泛化到256位，而标准Transformer仍停留在随机猜测水平。在训练长度八倍的基于交换的$S_5$置换跟踪任务上，RLT达到97%的最终状态准确率，而Transformer不足1%，且准确率随解码器深度增加而提升。在超出训练范围的模运算任务上……

    arXiv:2610.07591v1 Announce Type: cross  Abstract: State tracking requires an update at every input, but the depth a Transformer applies to each token is fixed regardless of sequence length. We introduce the Recurrent Looped Transformer (RLT), which splits its layers between a parallel causal encoder and a recurrent decoder. At each token, the decoder merges the encoder output with the previous token's final decoder state, so the computation path grows with sequence length at a fixed per-token cost. On six algorithmic tasks, we compare five splits of eight layers with an eight-layer Transformer over three seeds. Trained on at most 40 bits, two RLT splits generalize parity to 256 bits with 100% accuracy in every seed, while the Transformer stays at chance. On swap-based $S_5$ permutation tracking at eight times the training length, RLT reaches 97% final-state accuracy versus under 1% for the Transformer, and accuracy increases with decoder depth. On modular arithmetic beyond the trainin
    
[^191]: 基于跨平台用户历史的个人智能体中介推荐

    Personal-Agent Mediated Recommendation with Cross-Platform User History

    [https://arxiv.org/abs/2610.07588](https://arxiv.org/abs/2610.07588)

    提出了“个人智能体中介推荐”这一新范式及MediateRec基准，研究个人LLM智能体如何利用用户授权的跨平台历史来调解平台推荐排序，在有益挽救与有害覆盖之间取得平衡。

    

    现代推荐系统正从以平台为中心的个性化向用户主导的个性化转变，在这种模式下，个人LLM智能体可以代表用户跨多个服务行事。我们将这一新兴范式形式化为“个人智能体中介推荐”：平台推荐器利用平台本地信息对候选集进行排序，而个人智能体则利用用户授权的跨平台历史来调解该排序结果，并生成最终的top-K推荐列表。这种调解并非易事：平台排序中可能编码了个人智能体无法观察到的强大群体证据，因此有效的调解必须在有益的挽救与有害的覆盖之间取得平衡。为了研究这种权衡，我们提出了MediateRec基准，它包括可扩展的代理跨平台环境，以及在受控的平台-智能体信息边界下的真实跨平台测试。为了训练智能体使用跨平台历史有效进行调解……

    arXiv:2610.07588v1 Announce Type: new  Abstract: Modern recommendation is shifting from platform-centric personalization toward user-governed personalization, where a personal LLM agent can act on the user's behalf across services. We formalize this emerging paradigm as Personal-Agent Mediated Recommendation: a platform recommender ranks a candidate set using platform-local information, and a personal agent uses user-authorized cross-platform history to mediate the resulting ranking and produce the final top-K slate. Such mediation is nontrivial: the platform ranking can encode strong population evidence that the personal agent cannot observe, so effective mediation must therefore balance beneficial rescues against harmful overrides. To study this trade-off, we introduce MediateRec, a benchmark that includes scalable proxy cross-platform environments and a real cross-platform test under a controlled platform-agent information boundary. To train the agent to use cross-platform history e
    
[^192]: REViT-v2：用于等变特征提取的分层窗口化旋转-反射等变视觉Transformer

    REViT-v2: Hierarchical Windowed Roto-reflection Equivariant ViT for Equivariant Feature Extraction

    [https://arxiv.org/abs/2610.07585](https://arxiv.org/abs/2610.07585)

    本文提出了一种基于窗口化群卷积自注意力与分层特征架构的可扩展旋转-反射群等变视觉Transformer（REViT-v2），成功将群等变ViT扩展至数百万参数规模，并能在ImageNet等实际尺寸图像的大型数据集上进行等变特征提取。

    

    我们提出了一种可扩展的旋转-反射群等变视觉Transformer，该模型基于窗口化群卷积自注意力机制和分层特征架构。我们证明了该方法可以扩展到具有数百万参数的群等变视觉Transformer（ViT），并能处理实际尺寸图像的大规模数据集，例如ImageNet。所提出的分层窗口化旋转-反射等变视觉Transformer（REViT-v2）的代码和预训练权重已在 https://github.com/kc-ml2/revit 开源。

    arXiv:2610.07585v1 Announce Type: cross  Abstract: We propose a scalable roto-reflection-group-equivariant vision transformer based on windowed group-convolutional self-attention and a hierarchical feature architecture. We demonstrate that our approach can be scaled to group-equivariant vision transformers (ViTs) with millions of parameters and large datasets with practically sized images, i.e., ImageNet. The code and pretrained weights for the proposed Hierarchical Windowed Roto-reflection Equivariant ViTs (REViT-v2) are available at https://github.com/kc-ml2/revit.
    
[^193]: GraphCast中大气河流的机制可解释性

    Mechanistic Interpretability of Atmospheric Rivers in GraphCast

    [https://arxiv.org/abs/2610.07583](https://arxiv.org/abs/2610.07583)

    该研究通过对GraphCast训练稀疏自编码器，首次揭示这一AI天气模型内部稳定地计算出大气河流强度（综合水汽输送IVT）作为内部变量，并通过干预实验证实了其因果作用。

    

    尽管AI天气模型如今已能与业务化预报相媲美，但它们如何在内部表征大气仍是一个悬而未决的问题：特征归因只能揭示哪些输入模式重要，却无法说明模型计算了什么、以及如何在内部组合信息。我们在GraphCast上训练稀疏自编码器（SAE）以揭示其学习到的概念，并以大气河流作为研究焦点。标准SAE与Matryoshka SAE均表明，GraphCast将大气河流强度——以综合水汽输送（IVT）衡量——计算为一个稳定的内部变量，尽管IVT既非模型输入也非预测目标。与标准SAE非结构化的概念检索不同，Matryoshka SAE按重要性对概念进行排序并揭示概念之间的关系。大气河流概念在模型各深度层中持续存在，直接干预实验进一步证实了其因果性。该方法提供了一条寻找内部变量并确定模型实际依赖哪些变量的途径。

    arXiv:2610.07583v1 Announce Type: cross  Abstract: While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the model computes or how it combines information internally. We train sparse autoencoders (SAEs) on GraphCast to uncover its learned concepts, using atmospheric rivers as our phenomenon of focus. Both standard and Matryoshka SAEs show GraphCast computes atmospheric river intensity, measured by integrated vapor transport (IVT), as a stable internal variable, despite IVT being neither an input nor a target. In contrast to the unstructured concept retrieval of the standard SAE, the Matryoshka SAE orders concepts by importance and exposes their relations. Atmospheric river concepts persist across depth and direct interventions confirm causality. This method offers a way to find internal variables and determine which of them the model actually relie
    
[^194]: CETUS：在地球上训练的表征能在多大程度上迁移到卡西尼号土卫六合成孔径雷达图像上？

    CETUS: How Far Do Representations Trained on Earth Transfer to Cassini SAR of Titan?

    [https://arxiv.org/abs/2610.07576](https://arxiv.org/abs/2610.07576)

    该论文提出CETUS跨域评估基准，系统比较了地球影像预训练的视觉表征（DINOv2、DOFA、CROMA）与经典图像特征在卡西尼号土卫六SAR地形分类任务上的迁移能力，发现预训练编码器整体优于经典特征，但在土卫六数据上继续微调的效果因模型而异。

    

    卡西尼号合成孔径雷达（SAR）图像揭示了土卫六（泰坦）的沙丘、平原和湖盆，为将在地球影像上学到的表征应用于行星地形分类提供了一个实例。跨域地球-土卫六SAR迁移评估（CETUS）在美国地质调查局的卡西尼SAR镶嵌图上，将DINOv2、DOFA和CROMA提取的特征与经典图像测量特征以及未经训练的视觉Transformer特征进行了比较。分类器从专家绘制的地貌图中学习地形标签，并在地理上相互独立的土卫六区域对这些标签进行预测。在逻辑回归设置下，预训练编码器获得了比组合经典特征更高的平均宏观F1分数。当特征缩放、优化和正则化设置同时改变时，各编码器的排名会发生变化。在土卫六上的进一步训练提升了DINOv2的性能，降低了DOFA的性能，而CROMA在测试条件下则呈现出好坏参半的结果。

    arXiv:2610.07576v1 Announce Type: cross  Abstract: Cassini synthetic aperture radar (SAR) images reveal the dunes, plains, and lake basins of Titan, providing an instance of representations learned from Earth imagery for planetary terrain classification. Cross-domain Evaluation of Earth-to-Titan Transfer Using SAR (CETUS) compares features from DINOv2, DOFA and CROMA with classical image measurements and features from an untrained vision transformer on the U.S. Geological Survey's Cassini SAR mosaic. The classifiers learn terrain labels from an expert geomorphological map and predict those labels in geographically separate Titan regions. Under logistic regression settings, pretrained encoders achieve higher mean macro F1 than the combined classical features. Encoder rankings change when feature scaling, optimization, and regularization change together. Further training on Titan improves DINOv2 performance, degrades DOFA performance, and leads to mixed results for CROMA under the tested
    
[^195]: 两个向量替代上下文示例：通过嵌入实现结构化任务适配

    Two Vectors Replace In-Context Demos: Structured Task Adaptation via Embeddings

    [https://arxiv.org/abs/2610.07572](https://arxiv.org/abs/2610.07572)

    提出STAVE方法，用两个任务特定向量（读取向量和上下文向量）直接加到现有输入嵌入中来替代上下文示例，避免了示例图像的重复编码开销，实现高效的结构化任务适配。

    

    上下文学习（ICL）能够将冻结的大型多模态模型（LMMs）通过少量示例适应到新任务，但每次查询时都需要重新编码这些示例，其中每个示例图像会增加多达数百个视觉token。无示例方法通过紧凑的任务状态消除了这一成本，但它们将任务状态添加在针对每个任务搜索的位置或每个解码器层，导致任务参数随深度增长。此外，插入的token或键无法改变原始提示在层内分配注意力的方式。为解决这些问题，我们提出了结构化任务适配方法（STAVE），用两个任务特定的向量替代示例，并将其添加到现有的输入嵌入中。具体而言，读取向量更新产生答案的token，上下文向量更新其他结构化token组。两者均通过在有示例和无示例的提示上使用答案标签进行训练。我们使用一阶分析从理论上论证了这些设计选择的合理性。

    arXiv:2610.07572v1 Announce Type: new  Abstract: In-context learning (ICL) adapts frozen large multimodal models (LMMs) to new tasks from a few demonstrations (demos), but re-encodes them at every query, where each demo image adds up to hundreds of visual tokens. Demo-free methods remove this cost with a compact task state. However, they add it at locations searched per task or at every decoder layer, where task parameters grow with depth. Moreover, inserted tokens or keys cannot change how the original prompt divides its attention within a layer. To address these issues, we propose Structured Task Adaptation via Embeddings (STAVE), which replaces demos with two task-specific vectors added to existing input embeddings. Specifically, a readout vector updates the answer-producing tokens and a context vector updates the other structural token groups. Both are trained with answer labels on prompts with and without demos. We justify these design choices theoretically using a first-order ana
    
[^196]: 互补特征域：信息保持并不意味着预测贡献保持

    Complementary Feature Domains: Information Preservation Does Not Imply Predictive-Contribution Preservation

    [https://arxiv.org/abs/2610.07565](https://arxiv.org/abs/2610.07565)

    该论文提出互补特征域（CFD）理论，证明保持香农信息并不能保证保持预测贡献，并用贡献缺陷量化重编码下上下文贡献的变化，其上界由可达动作集间的行为距离与联盟不相容性之和界定。

    

    arXiv:2610.07565v1 公告类型：交叉 摘要：互补特征域（CFD）理论将预测价值刻画为一个由表示及其实现族共同诱导的、以上下文为索引的贡献系统。我们证明，香农信息的保持并不意味着这一贡献系统的保持：一个可逆的表示变换可以在保持目标信息不变的同时，改变受限决策族下的预测贡献。我们通过一个CFD贡献缺陷来形式化由此产生的转变，该缺陷度量在受控重编码下上下文贡献的变化程度。对于有界Lipschitz效用，我们证明每个联盟效用的偏移由重编码前后可达动作集之间的行为距离所界定；因此，每个上下文贡献缺陷都由相应联盟不相容性之和所界定。精确的行为封闭性带来不变性，而行为封闭性的逐渐减弱……（摘要截断）

    arXiv:2610.07565v1 Announce Type: cross  Abstract: Complementary Feature Domains (CFD) theory characterizes predictive value as a context-indexed contribution system induced jointly by representations and their realization family. We show that Shannon-information preservation does not imply preservation of this contribution system: an invertible representation transformation can leave target information unchanged while altering predictive contribution under a restricted decision family. We formalize the resulting transition through a CFD contribution defect that measures how contextual contributions change under controlled recoding. For bounded Lipschitz utility, we show that each coalition utility shift is bounded by the behavioral distance between the attainable action sets before and after recoding; consequently, every contextual contribution defect is bounded by the sum of the corresponding coalition incompatibilities. Exact behavioral closure yields invariance, while increasingly 
    
[^197]: 学习GFlowNets混合体

    Learning a Mixture of GFlowNets

    [https://arxiv.org/abs/2610.07562](https://arxiv.org/abs/2610.07562)

    提出了一个描述GFlowNets混合体的通用理论框架，将其细分为连续索引（CI）和离散索引（DI）两类：前者通过随机特征扩展与谱移位可证明地提升采样器的表达能力并降低学习不稳定性，后者统一了已有训练方法并支撑了新提出的分层条件化（SC）GFlowNets。

    

    学习一组GFlowNets集成以从离散目标分布中进行采样，已成为比单一采样器实现更好的状态空间探索和收敛性的常用方法。然而，这些方法通常会给基础模型带来较大的运行时开销，且它们之间的概念联系仍不明确。为解决这一问题，我们首先提出了一个用于描述GFlowNets混合体的通用理论框架，并将其细分为连续索引（CI）和离散索引（DI）两类集合。一方面，我们证明CI GFlowNets可以通过随机特征扩展的视角来解释，在图结构任务中可证明地提升采样器的表达能力，并通过谱移位降低学习的不稳定性。另一方面，我们证明DI GFlowNets涵盖了此前已有的GFlowNet训练方法，并为新提出的分层条件化（SC）GFlowNets奠定了基础。

    arXiv:2610.07562v1 Announce Type: cross  Abstract: Learning an ensemble of GFlowNets to sample from a discrete target distribution has become a common approach for achieving better state space exploration and convergence than that of a monolithic sampler. However, these methods often add a substantial runtime overhead to the base model, and their conceptual connection remains elusive. To address this, we first propose a general-purpose theoretical framework for describing a mixture of GFlowNets, which we specialize into continuously (CI) and discretely indexed (DI) collections. On the one hand, we show CI GFlowNets can be interpreted through the lens of a random features expansion, provably boosting the sampler's expressivity in graph-structured tasks and reducing learning instability via spectral shifting. On the other hand, we demonstrate DI GFlowNets encompass prior approaches for GFlowNet training and provide the foundation for the newly proposed Stratum-Conditioned (SC) GFlowNets.
    
[^198]: TAFFY：一种具有上下文内多样性的任务自适应表格基础模型

    TAFFY: A Task-Adaptive Tabular Foundation Model with In-Context Diversity

    [https://arxiv.org/abs/2610.07559](https://arxiv.org/abs/2610.07559)

    TAFFY通过基于共享因果过程干预生成多样化合成上下文的“上下文内多样性先验”和迭代精炼的任务条件循环Transformer，显著提升了表格基础模型从上下文推断任务特定预测关系的能力。

    

    近年表格基础模型的研究进展表明，在合成任务上进行训练能够显著提升模型的上下文内学习能力，而整体性能在很大程度上取决于模型在推理时能否从可用上下文中准确推断出任务特定的预测关系。在本文中，我们提出了TAFFY，一种带有上下文内多样性先验和任务条件循环Transformer的表格基础模型，用以强化这一能力。具体而言，为构建每个合成预训练上下文，上下文内多样性先验从多个相关环境中进行采样，这些环境是通过对共享因果过程施加受控干预和分布偏移而生成的。这种上下文内多样性促使模型学习到更全面且任务特定的表示。此外，任务条件循环Transformer迭代且选择性地应用一组共享的Transformer模块来精炼上下文信息（摘要原文在此处截断）。

    arXiv:2610.07559v1 Announce Type: new  Abstract: Recent progress in tabular foundation models suggests that training on synthetic tasks can substantially improve in-context learning capabilities, with overall performance largely depending on how well models can infer task-specific predictive relationships from the available context during inference. In this paper, we introduce TAFFY, a tabular foundation model with an In-Context Diversity Prior and a Task-Conditioned Looped Transformer that strengthen this ability. Specifically, to construct each synthetic pretraining context, the In-Context Diversity Prior samples from multiple related environments derived via controlled interventions and distribution shifts on a shared causal process. This in-context diversity encourages the model to learn a more comprehensive and task-specific representation. Moreover, the Task-Conditioned Looped Transformer iteratively and selectively applies a shared group of Transformer blocks to refine contextua
    
[^199]: 看见不可见之物：面向温度与辐射感知的VLA导航的物理引导视觉提示

    Seeing the Invisible: Physics-Guided Visual Prompting for Temperature- and Radiation-Aware VLA Navigation

    [https://arxiv.org/abs/2610.07558](https://arxiv.org/abs/2610.07558)

    提出物理引导视觉提示（PG-VP），将不可见的辐射或温度危险转化为动态虚拟障碍物视觉提示，使冻结的VLA导航模型无需重新训练即可规避多种不可见风险。

    

    视觉-语言-动作模型已成为视觉与语言导航的主要范式。然而，在安全关键设施中，辐射或温度骤升等不可见风险无法被RGB相机检测到，且处理每一种风险的成本都很高，需要新的编码器、新的数据以及模型重新训练。我们提出物理引导视觉提示，这是一种即插即用的多模态感知模块，它转而复用冻结的VLA模型已经擅长的能力：避开可见障碍物。给定一个近端的辐射源或热源，PG-VP执行物理引导的风险评估以确定避让方向，并叠加一个在连续帧之间移动的相应虚拟障碍物（动态视觉提示）。导航策略随后会自然地绕过这一不可见危险。无论危险类型如何，都使用相同的虚拟障碍物，因此视觉提示模式保持固定

    arXiv:2610.07558v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have become a major paradigm for Vision-and-Language Navigation (VLN). However, in safety-critical facilities, invisible risks such as radiation or temperature spikes cannot be detected by an RGB camera, and handling each risk is expensive, requiring a new encoder, new data, and model retraining. We propose Physics-Guided Visual Prompting (PG-VP), a plug-and-play multimodal perception module that instead reuses what a frozen VLA model already does well: avoiding visible obstacles. Given a proximal radiation or thermal source, PG-VP performs a physics-guided risk assessment to determine the avoidance direction and overlays a corresponding virtual obstacle that moves across consecutive frames (Dynamic Visual Prompting). The navigation policy then naturally detours around this invisible hazard. The identical virtual obstacle is used regardless of hazard type, so the visual prompting pattern remains fixe
    
[^200]: 面向无分类器引导流的全局传输耦合

    Global Transport Couplings for Classifier-Free Guided Flows

    [https://arxiv.org/abs/2610.07555](https://arxiv.org/abs/2610.07555)

    提出了一种无需类别标签的全局最优传输耦合方法（GT），它虽然在无引导时会降低性能，但与无分类器引导结合后能在不同领域、模型规模和采样预算下持续提升条件生成质量，并据此指出条件流耦合应在引导流下评估。

    

    最优传输耦合已被证明可以降低无条件流模型的训练方差，但其在条件生成中的作用仍不清楚。一种自然的方法是为每个条件构建单独的耦合，但这对于现代图像基础模型中庞大或连续的条件空间来说并不实际。我们提出了全局传输，这是一种与类别无关的全局最优传输耦合，无需类别标签即可计算。GT 能够将不同条件与源噪声的不同区域相关联，因此在无引导情况下反而会降低性能。然而，当与无分类器引导（CFG）结合时，GT 在不同领域、模型规模和采样预算下都能持续改进生成效果。这一反转表明，条件流的耦合应该在实际推理所使用的引导流下进行经验和理论评估，而不是在无引导生成上评估。我们在多种设置下对 GT 进行了评估。

    arXiv:2610.07555v1 Announce Type: new  Abstract: Optimal-transport couplings have been shown to reduce training variance in unconditional flow models, but their role in conditional generation remains unclear. A natural approach constructs separate couplings for each condition, but this is impractical for large or continuous conditioning spaces found in modern image foundation models. We introduce Global Transport (GT), a global class-agnostic optimal-transport coupling, computed without class labels. GT can associate different conditions with different regions of the source noise, and consequently worsens performance without guidance. However, when combined with classifier-free guidance (CFG), GT consistently improves generation across domains, model scales, and sampling budgets. This reversal suggests that couplings for conditional flows should be evaluated both empirically and theoretically under the guided flow used at inference, rather than on unguided generation. We evaluate GT ov
    
[^201]: 准入哪些与何时准入：面向数据中心化小语言模型微调的梯度准入方法

    Which and When to Admit: Gradient Admission for Data-Centric Small Language Model Finetuning

    [https://arxiv.org/abs/2610.07553](https://arxiv.org/abs/2610.07553)

    提出GRADE框架，通过状态感知选择器持续接纳与演化中的多任务梯度场对齐的样本，并用自校准步级门控在子空间接近饱和时拒绝破坏性更新，从而同时解决LoRA微调中的梯度冲突、静态数据选择和子空间饱和三大问题，提升小语言模型微调效果。

    

    LoRA微调使小语言模型（SLM）能够在低秩更新子空间内适应异构的指令数据，但这使其容易受到三个结构性问题的影响：相互冲突的梯度会彼此抵消、静态的数据选择无法追踪不断演化的学习动态、以及子空间饱和导致后续更新覆盖掉有用的方向。我们认为，有效的适配因此需要控制哪些数据诱导的梯度进入LoRA子空间以及何时进入。我们提出了GRADE（GRadient-Aligned Data-centric rEcipe，梯度对齐的数据中心化配方），这是一个数据中心化框架，结合了两种机制：一个状态感知选择器，持续接纳与不断演化的多任务梯度场对齐的样本；以及一个自校准的步级门控，用于拒绝在接近饱和时可能造成破坏性覆盖的更新。在三个当前一代的骨干模型和一个异构的七数据集指令池上，GRADE优于强大的数据选择（方法）……

    arXiv:2610.07553v1 Announce Type: cross  Abstract: LoRA fine-tuning adapts small language models (SLMs) to heterogeneous instruction data within a low-rank update subspace, making it vulnerable to three structural problems: conflicting gradients that cancel, static data selection that cannot track evolving learning dynamics, and subspace saturation that causes later updates to overwrite useful directions. We argue that effective adaptation therefore requires controlling which data-induced gradients enter the LoRA subspace and when. We propose GRADE (GRadient-Aligned Data-centric rEcipe), a data-centric framework combining two mechanisms: a state-aware selector that continually admits samples aligned with the evolving multi-task gradient field, and a self-calibrating step-level gate that rejects updates likely to cause destructive overwrite near saturation. Across three current-generation backbones and a heterogeneous seven-dataset instruction pool, GRADE outperforms strong data-selecti
    
[^202]: 高维学习高斯混合模型时，梯度 EM 是否必须要求 $\sqrt{d}$ 量级的分离度？

    Is $\sqrt{d}$ Separation Necessary for Gradient EM to Learn Gaussian Mixtures in High Dimensions?

    [https://arxiv.org/abs/2610.07551](https://arxiv.org/abs/2610.07551)

    本文证明在高维学习高斯混合模型时，梯度 EM 全局收敛所需的 $\Omega(\sqrt{d})$ 分离度条件是不可避免的，即这一对维度的依赖性无法被去除。

    

    使用期望最大化（EM）算法及其基于梯度的变体来学习高斯混合模型（GMM）是机器学习中的一个基本问题。众所周知，在精确参数化设置（即分量数量与真实 GMM 的分量数量相匹配）下，随机初始化的（梯度）EM 无法学习多分量 GMM。最近，在过参数化设置（即使用更多分量）下，只要真实分量之间分离良好，梯度 EM 的全局收敛性已经得到证明。特别地，真实分量之间的最小分离度需要达到 $\Omega(\sqrt{d})$ 的量级，其中 $d$ 为维度。在本文中，我们证明在高维设置下这种对维度的依赖是不可避免的。具体而言，我们考虑了一种混合 EM 算法，其对混合权重采用标准 EM 更新，对分量均值采用梯度 EM 更新。

    arXiv:2610.07551v1 Announce Type: cross  Abstract: Learning Gaussian mixture models (GMMs) using the Expectation-Maximization (EM) algorithm and its gradient-based variants is a fundamental problem in machine learning. It is known that randomly initialized (gradient) EM fails to learn multi-component GMMs in the exact-parameterized setting, where the number of components matches that of the ground-truth GMM. Recently, global convergence of gradient EM has been established in the over-parameterized setting, where more components are used, provided that the ground-truth components are well separated. In particular, the minimum separation between ground-truth components is required to scale as $\Omega(\sqrt{d})$, where $d$ is the dimension. In this paper, we show that this dimensional dependence is unavoidable in high-dimensional settings. Specifically, we consider a hybrid EM algorithm that uses standard EM updates for the mixing weights and gradient EM updates for the component means. F
    
[^203]: 基础模型辅助的多智能体强化学习用于无线随机接入网络优化

    Foundation Model-Aided Multi-Agent Reinforcement Learning for Wireless Random Access Network Optimization

    [https://arxiv.org/abs/2610.07550](https://arxiv.org/abs/2610.07550)

    该论文提出一种基础模型辅助的actor-critic多智能体强化学习算法，以显著降低无线随机接入网络优化任务中的训练开销，并证明了其与采用评论者模型交换和线性近似的传统MARL方法具有相同的收敛阶。

    

    随机接入（RA）是处理来自多个终端的不可预测数据流量的最基础的介质访问控制（MAC）层调度方案之一。虽然多智能体强化学习（MARL）已被用于优化基于随机接入的无线网络，但其依赖于经验驱动的分布式策略学习，为每个优化任务带来显著的训练开销，限制了其在实际应用中的可行性。在本工作中，我们提出利用基础模型（FM）来提高MARL在多样化随机接入网络优化任务中的效率。具体而言，我们在基于共识的去中心化MARL架构中设计了一种FM辅助的actor-critic算法，并在本地奖励交换和非线性价值函数近似的条件下提供了其收敛性分析，表明我们的算法达到了与采用评论者模型交换和线性近似的传统MARL相同的收敛阶。

    arXiv:2610.07550v1 Announce Type: cross  Abstract: Random access (RA) is one of the most foundational medium access control (MAC) layer scheduling schemes for handling unpredictable data traffic from multiple terminals. While multi-agent reinforcement learning (MARL) has been explored to optimize RA-based wireless networks, its reliance on experience-driven, distributed policy learning incurs significant training overhead for each optimization task, limiting its feasibility in real-world applications. In this work, we propose to leverage a foundation model (FM) to improve MARL efficiency across diverse RA network optimization tasks. Specifically, we design an FM-aided actor-critic algorithm within a consensus-based decentralized MARL architecture and provide its convergence analysis under local reward exchanges and nonlinear value function approximations to show that our algorithm achieves the same convergence order as the conventional MARL with critic model exchanges and linear approx
    
[^204]: 通过逆动力学在JEPA世界模型中保留不稳定模态

    Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models

    [https://arxiv.org/abs/2610.07540](https://arxiv.org/abs/2610.07540)

    该论文发现JEPA世界模型中下一步预测与防坍缩正则化无法保证可控的不稳定模态被保留，并提出通过逆动力学（动作重建）损失增强世界模型训练，以学习能保留不稳定模态的控制感知视觉表示。

    

    机器人系统常常表现出不稳定模态，沿着这些模态，微小的扰动和干扰会导致无界增长，除非通过反馈加以纠正。从高维视觉观测中控制此类系统，需要能够保留这些模态的表示。联合嵌入预测架构（JEPA）为从视觉数据中学习此类表示及其动力学提供了一个自然的框架。然而，我们证明了下一步预测结合防坍缩正则化并不能保证可控的不稳定模态得到保留：即使这些模态已经坍缩，训练损失仍然可以被最小化，从而导致无法从学到的表示中实现稳定化控制。为解决这一问题，我们通过动作重建目标（即逆动力学损失）来增强世界模型的训练，该目标鼓励学习控制感知的表示，即能够保留关键特征的视觉表示……

    arXiv:2610.07540v1 Announce Type: new  Abstract: Robotic systems often exhibit unstable modes, along which small perturbations and disturbances can cause unbounded growth unless corrected through feedback. Controlling such systems from high-dimensional visual observations requires representations that preserve these modes. Joint-embedding predictive architectures (JEPAs) provide a natural framework for learning such representations and their dynamics from visual data. However, we demonstrate that next step prediction combined with anti-collapse regularization does not guarantee that controllable unstable modes are preserved: the training loss can be minimized while these modes are collapsed, making stabilization from the learned representation impossible. To address this, we augment world-model training with an action reconstruction objective (i.e., an inverse dynamics loss) that encourages control-aware representations, namely, visual representations that preserve crucial features for
    
[^205]: SkillFormer：面向音频语言模型的技能分解式适配方法

    SkillFormer: Skill-Decomposed Adaptation for Audio Language Models

    [https://arxiv.org/abs/2610.07533](https://arxiv.org/abs/2610.07533)

    SkillFormer通过将音频理解分解为技能特定的低秩适配器，并利用学习到的路由器在推理时动态组合它们，配合交替式训练方案避免了多任务梯度冲突，以不到4%的参数增量解决了音频语言模型多技能训练中的干扰问题。

    

    音频语言模型必须处理数十种不同的技能，从音高比较和说话人计数，到音乐节奏估计和情感识别。对所有技能同时进行联合训练会导致干扰：一项技能的提升往往以牺牲另一项技能为代价。我们提出SkillFormer，它将音频理解分解为技能特定的低秩适配器，并在推理时通过一个学习得到的路由器将它们组合起来。路由器通过审视问题来决定激活哪些适配器以及每个适配器应占的权重，因此音高查询会调用与音乐流派分类查询不同的参数。一种交替式训练方案先在每个适配器各自的技能集群上对其进行更新，然后联合校准路由器，从而避免了标准多任务优化中出现的梯度冲突。SkillFormer仅增加不到基础模型4%的参数，且无需对音频编码器或语言模型进行任何更改。

    arXiv:2610.07533v1 Announce Type: cross  Abstract: Audio language models must handle dozens of distinct skills, from pitch comparison and speaker counting to musical tempo estimation and emotion recognition. Joint training on all skills at once causes interference: gains on one skill often come at the cost of another. We propose \textbf{SkillFormer}, which decomposes audio understanding into skill-specific low-rank adapters and composes them at inference time through a learned router. The router examines the question to decide which adapters to activate and how much weight each should carry, so that a pitch query engages different parameters than a genre classification query. An alternating training schedule updates each adapter on its own skill cluster before jointly calibrating the router, preventing the gradient conflicts that arise in standard multi-task optimization. SkillFormer adds fewer than 4\% of the base model's parameters and requires no changes to the audio encoder or lang
    
[^206]: 针对性搜索表明，随机器件测试会低估模拟波基神经算子的最坏情况误差

    Targeted search shows that random-device testing underestimates worst-case error in a simulated wave-based neural operator

    [https://arxiv.org/abs/2610.07529](https://arxiv.org/abs/2610.07529)

    该研究通过对模拟波基神经算子进行针对性器件搜索，发现搜索到的器件误差可达随机器件测试最大值的1至3倍，证明仅依赖随机器件测试会严重低估实际部署中可能出现的最坏情况误差。

    

    波基处理器有望为神经算子提供快速、节能的傅里叶层。这类处理器通常在随机采样的器件上进行验证，但要实际使用它们，就需要知道在制造和对准变化下误差可能达到多大。在一个风格化的数值案例研究中，一个混合傅里叶神经算子在具有32个带公差旋钮的模拟相干4f处理器上运行其四个光谱层，这些旋钮的半宽度具有代表性而非经过校准。对于120个模型（四个任务、六种训练方法、五个随机种子），我们将N个随机合格器件中的最差者与一个通过搜索找到的器件进行了比较。在具有一次冻结的随机静态误差抽样的确定性模拟器上，搜索得到的器件的保留测试误差是200个蒙特卡洛器件中最大值的1.08-3.10倍，是1000个器件中最大值的1.06-2.71倍。在20次新的静态误差抽样中，它在120个模型中的116个上仍超过了200个随机器件的最大误差。在均匀采样下，……（原文摘要到此截断）

    arXiv:2610.07529v1 Announce Type: new  Abstract: Wave-based processors promise fast, energy-efficient Fourier layers for neural operators. They are usually validated on randomly sampled devices, but using them requires knowing how large their error can become under fabrication and alignment variation. In a stylised numerical case study, a hybrid Fourier neural operator runs its four spectral layers on simulated coherent 4f processors with 32 toleranced knobs, whose half-widths are representative rather than calibrated. For 120 models (four tasks, six training methods, five seeds), we compared the worst of N random in-spec devices with a searched one. On a deterministic simulator with one frozen draw of the random static errors, the searched device's held-out error was 1.08-3.10 times the maximum over 200 Monte Carlo devices and 1.06-2.71 times that over 1000. With 20 fresh static draws, it still exceeded the maximum over 200 random devices in 116 of 120 models. Under uniform sampling, 
    
[^207]: 激活去噪：从鲁棒性视角看并行与串行大语言模型量化

    Activation Denoising: A Robustness View on Parallel vs Sequential LLM Quantization

    [https://arxiv.org/abs/2610.07522](https://arxiv.org/abs/2610.07522)

    提出激活去噪方法，从鲁棒性视角将上游量化误差建模为噪声并加以正则化抑制，使并行量化达到接近串行量化的精度，同时保持完全并行的可扩展性。

    

    训练后量化是压缩大语言模型的有力工具。最具可扩展性的方法是对每一层进行并行量化，但量化误差会随后在残差流中不断累积，因为没有任何一层会纠正其之前各层的误差。串行量化则考虑到了这种误差累积问题，它通过在已量化的前序层输出上重新校准每一层来应对误差，从而取得更强的效果，但其串行的调度方式在大规模场景下会成为瓶颈。作为解决方案，我们提出了带激活去噪的并行量化方法，在保持量化完全并行的同时，恢复了串行量化的大部分优势。我们不再逐层重新校准，而是采用鲁棒性的视角，将上游误差建模为噪声，通过一个预处理步骤及随后的度量加权舍入进行正则化，使模型对该噪声具有鲁棒性。将该正则化应用于每一层，即可形成一种深度——（摘要原文在此处截断）

    arXiv:2610.07522v1 Announce Type: new  Abstract: Post-training quantization is a powerful tool for compressing large language models. The most scalable methods quantize every layer in parallel, but quantization errors then compound through the residual stream, as no layer corrects for the errors of the layers before it. Sequential quantization accounts for this error compounding by re-calibrating each layer on the already-quantized outputs of its predecessors, yielding stronger results but at the cost of a serial schedule that becomes a bottleneck at scale. As a solution, we propose parallel quantization with activation denoising, which recovers much of the sequential benefit while keeping quantization fully parallel. Rather than re-calibrating layer-by-layer, we take a robustness perspective and model the upstream error as noise, regularizing to be robust to it through a preprocessing step followed by metric-weighted rounding. Applied at every layer, this regularization forms a depth-
    
[^208]: 有害SFT在大语言模型检查点更新中留下连续痕迹

    Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates

    [https://arxiv.org/abs/2610.07518](https://arxiv.org/abs/2610.07518)

    该研究发现有害监督微调会在大语言模型的检查点更新中留下连续且可读取的痕迹，通过检查点级别的坐标即可高精度检测有害目标的存在，从而实现无需运行模型的安全审计。

    

    对训练后大语言模型的安全审计通常依赖于模型行为，需要执行模型并取决于可用评估的覆盖范围。这项工作提出了一个不同的问题：在监督微调（SFT）期间优化的目标行为是否会在检查点更新中直接留下可读取的证据？我们发现，有害依从性SFT会在检查点更新空间中诱导出一种连续的、依赖于目标的排序。使用由纯有害依从性、安全目标和良性效用SFT所定义的参考几何结构，我们发现一个检查点级别的坐标s_H能够追踪受控的有害目标组成，在四个7-8B骨干模型上达到0.986-0.992的Spearman相关系数，并且相同的排序在更大模型规模上依然存在。匹配的依从性与拒绝对照实验表明，这种检查点痕迹反映的是SFT目标本身，而非有害输入的暴露，同时额外的对照实验排除了其他可能的解释。

    arXiv:2610.07518v1 Announce Type: cross  Abstract: Safety auditing of post-trained large language models typically relies on model behavior, requiring model execution and depending on the coverage of available evaluations. This work asks a different question: Do the target behaviors optimized during supervised fine-tuning (SFT) leave readable evidence directly in checkpoint updates? We find that harmful-compliance SFT induces a continuous, objective-dependent ordering in checkpoint-update space. Using a reference geometry defined by pure harmful-compliance, safety-targeted, and benign-utility SFT, we find that a checkpoint-level coordinate s_H tracks controlled harmful-objective composition with Spearman correlations of 0.986-0.992 across four 7-8B backbones, with the same ordering persisting at larger model scales. Matched compliance-versus-refusal controls show that this checkpoint trace reflects the SFT objective rather than harmful-input exposure, while additional controls rule out
    
[^209]: MobileVISTA：面向移动操作位姿泛化的生成式数据增强

    MobileVISTA: Generative Data Augmentation for Pose Generalization in Mobile Manipulation

    [https://arxiv.org/abs/2610.07511](https://arxiv.org/abs/2610.07511)

    提出 MobileVISTA 数据生成框架，通过联合增强自我中心视觉观测与动作重定向，将单一标准位姿下的演示转化为位姿多样化的训练数据，从而让移动操作策略对机器人位姿变化具备强泛化能力。

    

    以人形机器人为代表的移动操作机器人正越来越多地被部署在动态、非结构化的环境中执行灵巧操作任务。然而，通过模仿从单一机器人位姿采集的演示数据训练出的端到端操作策略非常脆弱：部署时即使机器人位姿出现厘米级的偏差，也会使自我中心的观测和末端执行器轨迹偏离训练分布，导致性能急剧下降。我们提出了 MobileVISTA，这是一个数据生成框架，通过联合（1）对自我中心视觉观测进行增强，以及（2）对动作进行重定向以补偿基座位姿的变化，将采集于标准位姿的演示数据转化为多样化的、带位姿扰动的训练数据。与以往假设相机刚性安装在受驱动链之外、或机器人关节几何结构大部分处于画面之外的方法不同，MobileVISTA 旨在兼容以自我为中心的……（摘要原文在此处截断）

    arXiv:2610.07511v1 Announce Type: cross  Abstract: Mobile manipulators such as humanoid robots are increasingly deployed in dynamic, unstructured environments to perform dexterous manipulation tasks. However, end-to-end manipulation policies trained to imitate demonstration data collected from a single robot pose are brittle: even centimeter-scale deviations in robot pose at deployment can drive ego-centric observations and end-effector trajectories out of the training distribution, leading to sharp drops in performance. We introduce MobileVISTA, a data generation framework that transforms demonstrations captured at canonical poses into diverse, pose-perturbed training data by jointly (1) augmenting egocentric visual observations and (2) retargeting actions to compensate for base pose changes. Unlike prior methods, which assume a camera rigidly mounted off the actuated chain or non-trivial articulated robot geometry largely out of frame, MobileVISTA targets compatibility with egocentri
    
[^210]: 无需顶点对应关系的随机图双样本检验

    Two-Sample Testing for Random Graphs without Vertex Correspondence

    [https://arxiv.org/abs/2610.07503](https://arxiv.org/abs/2610.07503)

    该论文首次建立了顶点无对应情形下图总体双样本检验的最优样本复杂度理论，证明对于保持度不变的两块结构差异，每组需要约 t^{-3} 张图，带符号三角形计数可达到该最优速率，且未对齐相比对齐情形需多付出 t^{-2} 阶的样本代价。

    

    两群图之间常常需要在顶点没有任何对应关系的情况下进行比较，例如当网络来自不同的社区时，或者在评估图生成模型与留出图时。我们研究了这种未对齐的双样本检验需要多少张图，以及哪些图统计量能够检测哪些类型的差异。对于 Erdős–Rényi 零假设以及保持每个期望度不变的植入式两块差异，我们证明当每张图的信噪比 t<1 时，每组 m≍t^{-3} 张图既是必要也是充分的。带符号三角形计数可以达到这一速率，且该下界对任意图规模都成立。当顶点对齐时，m≍t^{-1} 张图就足够了，因此未对齐会带来 t^{-2} 阶的代价因子。当三角形信号相互抵消时，速率变为 t^{-4}，此时需要借助 4-环。基于树构建的统计量在两种情形下具有完全相同的期望……

    arXiv:2610.07503v1 Announce Type: cross  Abstract: Two populations of graphs often have to be compared without any correspondence between their vertices, for instance when networks come from different communities, or when a graph generative model is evaluated against held-out graphs. We study how many graphs such an unaligned two-sample test needs, and which graph statistics can detect which differences. For an Erd\H{o}s--R\'enyi null and a planted two-block difference that leaves every expected degree unchanged, we show that $m\asymp t^{-3}$ graphs per group are necessary and sufficient when the per-graph signal-to-noise ratio is $t<1$. Signed triangle counts attain this rate, and the lower bound holds for every graph size. With aligned vertices $m\asymp t^{-1}$ graphs suffice, so misalignment costs a factor of order $t^{-2}$. When the triangle signal cancels, the rate becomes $t^{-4}$ and $4$-cycles are needed. Statistics built from trees have exactly the same expectation under both 
    
[^211]: 面向多模态时间序列选择性测试时自适应的源学习依赖度估计

    Source-Learned Reliance for Selective Test-Time Adaptation of Multimodal Time Series

    [https://arxiv.org/abs/2610.07499](https://arxiv.org/abs/2610.07499)

    CARAT通过源域训练预先学习骨干网络特定的依赖度代理，并结合轻量级单类损坏检测器，在部署时实现多模态时间序列的选择性测试时自适应，从而避免跨模态一致性判断的误导和额外推理成本，提升传感器噪声或缺失情况下的系统可靠性。

    

    多模态可穿戴系统在传感器数据流变得嘈杂或不可用时必须保持可靠。现有多模态测试时自适应（TTA）方法通常在线评估可靠性，但当传感器测量不同的物理过程时，跨模态一致性可能会产生误导，而且评估不同的模态配置会增加推理成本。我们提出CARAT，该方法将模型依赖度与运行时损坏检测解耦，以指导对输入的舍弃或衰减，并通过源域训练来摊销依赖度估计的成本。一种非对称的模态丢弃课程训练使骨干网络具备应对模态缺失的鲁棒性，并从窗口化的输入投影梯度范数中导出一个冻结的、特定于该骨干网络的依赖度代理。在部署时，轻量级的单类检测器标记可疑数据流，依赖度代理则指导在用骨干网络训练过的缺失符号替换可疑集合与衰减其表示之间做出联合选择。

    arXiv:2610.07499v1 Announce Type: new  Abstract: Multimodal wearable systems must remain reliable when sensor streams become noisy or unavailable. Existing multimodal test-time adaptation (TTA) methods often assess reliability online, but cross-modal agreement can be misleading when sensors measure different physical processes, and evaluating alternative modality configurations adds inference cost. We propose CARAT, which decouples model reliance from runtime corruption detection to guide omission or attenuation, amortizing reliance estimation through source training. An asymmetric modality-dropout curriculum prepares a missingness-resilient backbone for omission and derives a frozen, backbone-specific reliance proxy from windowed input-projection gradient norms. At deployment, a lightweight one-class detector flags suspect streams, and the proxy guides a joint choice between replacing the suspect set with the backbone's trained missingness symbol and attenuating its representations be
    
[^212]: Muon 需要细粒度的谱整形吗？

    Does Muon Need Fine-Grained Spectral Shaping?

    [https://arxiv.org/abs/2610.07497](https://arxiv.org/abs/2610.07497)

    本文提出 BulkBoost 双频段谱重加权框架，表明 Muon 并不需要细粒度的谱整形，只需将奇异谱粗略地划分为噪声主体和高增益尖峰两个频段并进行重加权即可提升优化效果。

    

    Muon 将当前梯度与历史梯度结合为矩阵动量。对于 M=UΣV^T，理想化的极分解更新 Q=UV^T 会给每个奇异方向赋予相同的权重，我们将其称为“平坦谱形”。近期一些优化器用细粒度的谱映射取代这种平坦谱形，为每个方向赋予各自的增益。我们探究 Muon 更新到底需要多少这样的谱细节。我们的谱诊断显示，约 94%–97% 的测量奇异模态位于估计的噪声边缘之下，但它们整体上与参考梯度呈正相关对齐。我们提出了 BulkBoost，一个双频段谱重加权框架，包含固定秩与噪声校准两种变体。后者利用拆分小批量梯度差异，为 Muon 的 Nesterov 输入校准一个 Marchenko–Pastur 参考边缘，从而将边缘之下的主体部分与边缘之上的尖峰部分分离开来。两种变体都通过一次……（增加主体部分的相对权重）［摘要在此处被截断］

    arXiv:2610.07497v1 Announce Type: new  Abstract: Muon combines current and past gradients into matrix momentum. For $M=U\Sigma V^\top$, the idealized polar update $Q=UV^\top$ gives every singular direction the same weight. We refer to this as the flat profile. Several recent optimizers replace this flat profile with fine-grained spectral maps that give each direction its own gain. We ask how much of this spectral detail a Muon update needs. Our spectral diagnostics show that approximately $94$--$97\%$ of measured singular modes lie below an estimated noise edge, yet collectively align positively with a reference gradient.   We introduce BulkBoost, a two-band spectral reweighting framework with fixed-rank and noise-calibrated variants. The latter uses split-minibatch gradient differences to calibrate a Marchenko--Pastur reference edge for Muon's Nesterov input, separating the bulk below the edge from the spikes above it. Both variants increase the bulk's relative weight through one shar
    
[^213]: 谁来承担负担？多智能体强化学习中共享约束的责任学习

    Who Bears the Burden? Learning Responsibility for Shared Constraints in Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.07491](https://arxiv.org/abs/2610.07491)

    提出LiRA方法，通过优化社会福利来学习各智能体在共享拉格朗日乘子中的责任份额，从而解决多智能体强化学习中共享约束惩罚如何在不同智能体之间合理分配的问题。

    

    当多个智能体共享一个成本预算时，一个共同的拉格朗日乘子可以强制执行总体约束，但无法决定其惩罚应如何在各智能体之间分配。统一的惩罚方式忽略了智能体所牺牲奖励的异质性，而针对各智能体的独立乘子可能仍然依赖于相同的总体成本信号。我们提出了拉格朗日责任分配，该方法通过在有限训练范围内优化社会福利，来学习每个智能体在共同乘子中所占的份额。该乘子负责强制执行总体预算，而责任份额则在不修改原始奖励或约束的前提下重新分配其影响。对于满足标准正则条件的凸博弈，改变这些责任份额会诱导出一族平滑的归一化广义纳什均衡，其中活跃约束始终保持在预算水平，而社会福利则随份额变化。为了在收敛之前优化责任分配，我们推导了福利（摘要在此处截断）

    arXiv:2610.07491v1 Announce Type: cross  Abstract: When multiple agents share a cost budget, a common Lagrange multiplier can enforce the aggregate constraint but does not determine how its penalty should be allocated across agents. Uniform penalties ignore heterogeneity in the rewards agents sacrifice, while agent-specific multipliers may still rely on the same aggregate cost signal. We introduce Lagrangian Responsibility Allocation (LiRA), which learns each agent's share of a common multiplier by optimizing social welfare over a finite training horizon. The multiplier enforces the aggregate budget, while responsibility shares redistribute its influence without modifying the original rewards or constraints. For convex games under standard regularity conditions, varying these shares induces a smooth family of normalized generalized Nash equilibria in which active constraints remain at their budgets while welfare varies. To optimize responsibility before convergence, we derive a welfare
    
[^214]: 轮上深度防御：面向车载网络安全全面防护的双入侵检测系统架构

    Deep Defence on Wheels: A Dual Intrusion Detection System Architecture for Comprehensive In-Vehicle Network Security

    [https://arxiv.org/abs/2610.07489](https://arxiv.org/abs/2610.07489)

    该论文提出了一种面向资源受限车载平台的双重入侵检测系统框架，其中量化的LSTM模型（QLSTM-IDS）在满足低延迟、低资源开销的部署约束下，对DoS/洪泛、Fuzzing和欺骗/故障等攻击实现了超过99.9%的检测准确率。

    

    随着与外部世界连接性的不断增强以及内置安全机制的缺失，传统车载网络已变得容易受到网络攻击。早期的相关研究主要集中于最大化对已知和未知攻击的检测精度，通常采用大型、全精度的机器学习模型。然而，将入侵检测系统（IDS）嵌入到车辆电子系统中，还需要满足低检测延迟、高能效以及最小化电子控制单元（ECU）资源开销等要求，以便处理每秒约2,000帧的CAN报文。轻量化模型必须在检测精度与这些部署约束之间取得平衡。我们提出了一种双重IDS框架，由基于监督学习和无监督学习的解决方案组成，每种方案均针对实时性要求高、资源受限的汽车平台进行了优化。其中，基于量化LSTM的IDS（QLSTM-IDS）采用单一模型架构，在针对DoS/洪泛攻击、Fuzzing攻击以及欺骗/故障攻击的检测中实现了超过99.9%的检测准确率，并在两个数据集上进行了评估（原文摘要至此处截断）。

    arXiv:2610.07489v1 Announce Type: cross  Abstract: Increasing connectivity to the outside world and the lack of inbuilt security mechanisms have made legacy intra-vehicular networks vulnerable to cyberattacks. Initial research focused on maximising detection accuracy for known and unknown attacks, often using large, full-precision machine learning models. However, embedding IDSs into vehicular electronic systems also requires low detection latency, energy efficiency and minimal electronic control unit (ECU) resource overhead to process about 2,000 CAN frames/s. Lightweight models must balance accuracy with these deployment constraints. We propose a dual IDS framework comprising supervised and unsupervised learning-based solutions, each optimised for real-time, resource-constrained automotive platforms. A quantised LSTM-based IDS (QLSTM-IDS) achieves over 99.9% detection accuracy for DoS/Flooding, Fuzzing and Spoofing/Malfunction attacks using a single model architecture evaluated on tw
    
[^215]: 基于约束高斯混合的稀有事件鲁棒重要性抽样

    Robust Importance Sampling for Rare Events via Constrained Gaussian Mixtures

    [https://arxiv.org/abs/2610.07485](https://arxiv.org/abs/2610.07485)

    该论文提出一种将稀有事件重要性抽样分解为“覆盖”与“拟合”两个阶段、并对最终高斯混合提议分布施加约束以保证重要性抽样方差有限的框架，从而在多种基线方法之上显著提升了稀有事件概率估计的效率与鲁棒性。

    

    我们研究稀有事件概率的估计问题，即估计 I = P(g(X) > γ)，其中 X 服从正态分布 N(μ, Σ)，g : R^d → R 为一般函数。我们通过重要性抽样来解决这一问题，并结合稀有事件估计与交叉熵优化两个领域的思想，提出了一个框架。与原始蒙特卡洛、自适应交叉熵、基于变分推断的方法（包括反向KL与前向KL方法），以及Safe-ICE、子集模拟（Subset Simulation）和序贯蒙特卡洛等基线方法相比，该框架显著提升了估计的效率与鲁棒性。其核心贡献包含两部分：第一，我们将问题分解为“覆盖”和“拟合”两个阶段，前者用于克服冷启动障碍，后者用于在获得有意义的信号后进一步细化提议分布；第二，我们对最终的高斯混合模型（GMM）提议分布施加约束，使其具有有限的重要性抽样方差（因为仅有覆盖并不足够……原文摘要在此处截断）。

    arXiv:2610.07485v1 Announce Type: new  Abstract: We study estimating rare-event probabilities $I = \mathbb{P}(g(\mathbf{X}) > \gamma)$ with $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and general $g : \mathbb{R}^d \to \mathbb{R}$. We address this problem through importance sampling, and propose a framework that substantially improves efficiency and robustness over baselines such as crude Monte Carlo, adaptive cross-entropy, variational-inference-based methods (including reverse- and forward-KL approaches), as well as Safe-ICE, Subset Simulation, and Sequential Monte Carlo, drawing on ideas from both rare-event estimation and cross-entropy optimization. The key contribution has two parts: first, we separate the problem into coverage, to overcome the cold-start barrier, and fitting, to refine proposals once a meaningful signal is available; second, we constrain the final GMM proposal so that it has finite importance-sampling variance (since coverage alone is not 
    
[^216]: SpecBraM：EEG基础模型应该预测什么？掩码频带功率预测与波形重建的对比

    SpecBraM: What Should an EEG Foundation Model Predict? Masked Band-Power Prediction versus Waveform Reconstruction

    [https://arxiv.org/abs/2610.07484](https://arxiv.org/abs/2610.07484)

    该研究提出掩码频带功率预测（MBP）作为EEG自监督预训练目标，相比波形重建在睡眠分期上性能更优，且预训练目标的选择比tokenizer设计的影响更大。

    

    自监督EEG模型通常重建被掩码的波形或预测离散编码。我们研究了一种任务对齐的替代方案：掩码频带功率预测（MBP），即为被掩码的通道-时间片段预测固定的窄带对数频谱能量。这一预测目标保留了与睡眠分期相关的节律功率，同时避免了相位敏感的波形重建以及需要学习的码本。在三个预训练随机种子下，我们在匹配的骨干网络、预训练数据（2,388小时）和训练步数条件下，比较了频带功率目标与波形目标，并采用了2x2的tokenizer-目标组合设计。在ISRUC和HMC睡眠分期任务上，在严格的线性探测评估下，MBP在全量标签情况下比原始和频带波形重建高出1.6-2.8个平衡准确率点，在仅使用1%标签的情况下高出4.7-7.3个点；预测目标的影响超过了tokenizer设计的影响。其冻结特征达到0.7916/0.7425的平衡准确率，而匹配的丰富手工频谱特征基线为0.7636/0.7227。

    arXiv:2610.07484v1 Announce Type: new  Abstract: Self-supervised EEG models often reconstruct masked waveforms or predict discrete codes. We study a task-aligned alternative: masked band-power prediction (MBP), which predicts fixed narrow-band log spectral energy for masked channel-time patches. This target retains rhythm power relevant to sleep staging while avoiding phase-sensitive waveform reconstruction and a learned codebook. Across three pretraining seeds, we compare band-power and waveform targets with matched backbones, pretraining data (2,388 hours), and training steps, including a 2x2 tokenizer-by-target design. On ISRUC and HMC sleep staging, MBP exceeds raw- and band-waveform reconstruction by 1.6-2.8 balanced-accuracy points with all labels and 4.7-7.3 points with 1% of labels under a strict linear probe; the target effect exceeds the tokenizer effect. Its frozen features reach 0.7916/0.7425 balanced accuracy, versus 0.7636/0.7227 for a matched rich handcrafted spectral ba
    
[^217]: 通过有限深度策略敏感度适应智能体行为的变化

    Adapting to Changes in Agent Behavior via Finite-Depth Policy Sensitivity

    [https://arxiv.org/abs/2610.07475](https://arxiv.org/abs/2610.07475)

    提出一个有限深度策略敏感度框架，通过可调节截断深度近似计算策略对其他智能体行为变化的一阶敏感度，并证明截断误差界随传播深度递减、在全视野传播时消失。

    

    将强化学习策略适应于另一智能体行为的变化，通常需要大量的新交互数据。策略敏感度提供了一种一阶预测方法，用于刻画局部最优策略如何随行为参数变化，但其计算需要二阶导数，而这些导数的效应会传播至未来的交互过程中。我们开发了一个有限深度框架，通过利用参考环境中的信息来近似策略Hessian矩阵和混合导数，从而估计这种敏感度。该方法具有可调节的传播深度，决定了沿轨迹的导数传播在何处被截断。我们刻画了有限深度传播所遗漏的导数贡献，并为近似导数及由此得到的策略敏感度推导了截断误差界。这些误差界随传播深度单调递减，并在全视野传播时趋于零。

    arXiv:2610.07475v1 Announce Type: new  Abstract: Adapting a reinforcement learning policy to changes in another agent's behavior typically requires a large amount of new interaction data. Policy sensitivity provides a first-order prediction of how a locally optimal policy changes with a behavioral parameter, but its computation requires second-order derivatives whose effects propagate across future interactions. We develop a finite-depth framework to estimate this sensitivity by approximating the policy Hessian and mixed derivative using information from a reference environment. The method features an adjustable propagation depth which determines where derivative propagation along the trajectory is truncated. We characterize the derivative contributions omitted by finite-depth propagation and derive truncation-error bounds for the approximated derivatives and resulting policy sensitivity. The bounds are nonincreasing with propagation depth and vanish at full-horizon propagation. Using 
    
[^218]: 结构而非信念：组合半老虎机中基于LLM衍生协方差的相关汤普森采样

    Structure, Not Belief: Correlated Thompson Sampling from LLM-Derived Covariance in Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2610.07470](https://arxiv.org/abs/2610.07470)

    提出一种对组合汤普森采样的最小改动方法，仅查询LLM一次将臂划分转化为相关协方差矩阵来引导探索，理论证明相比独立采样可获得有限时域内√(d/K)的遗憾改进，实验中遗憾降低19%。

    

    组合汤普森采样（CTS）为每个臂独立抽取后验样本，因此其探索动态忽略了臂之间的任何关联。我们研究了对此动态的一个最小改动：仅向大语言模型（LLM）查询一次以获得臂的划分，该划分通过聚类秩上的RBF核转化为正定相关矩阵Σ，每轮后验样本以协方差Σ抽取，而Beta后验仅依据真实奖励进行更新，因此LLM塑造的是采样器的移动方式，而非其信念内容。我们为理想化的高斯采样器给出了一个自洽的贝叶斯遗憾界，其信息增益分解为来自K聚类结构的K log T项和增长至d log T的岭回归项：相比独立采样的√(d/K)遗憾改进是一种有限时域的瞬态效应，仅在簇内相关性趋于1时才精确成立。该相关采样器将遗憾降低了19%。

    arXiv:2610.07470v1 Announce Type: cross  Abstract: Combinatorial Thompson sampling (CTS) draws independent posterior samples for every arm, so its exploration dynamics ignore any relation among arms. We study a minimal change to those dynamics: an LLM is queried once for a partition of the arms, the partition becomes a positive-definite correlation matrix $\Sigma$ through an RBF kernel on cluster ranks, and the per-round posterior sample is drawn with covariance $\Sigma$ while the Beta posteriors are updated from real rewards only, so the LLM shapes how the sampler moves, not what it believes. We give a self-contained Bayesian regret bound for the idealized Gaussian sampler whose information gain splits into a $K\log T$ term from the $K$-cluster structure and a ridge term that grows to $d\log T$: the $\sqrt{d/K}$ improvement over independent sampling is a finite-horizon transient, exact only as the within-cluster correlation tends to one. The correlated sampler reduces regret by 19% ov
    
[^219]: 通过自适应获取与序列融合实现高效多模态推理

    Efficient Multimodal Inference through Adaptive Acquisition and Sequential Fusion

    [https://arxiv.org/abs/2610.07466](https://arxiv.org/abs/2610.07466)

    该论文提出SemARC框架，通过序列模态聚合器SeMA与自适应运行时控制器ARC协同工作，在模态编码前自适应地选择模态，仅执行必要的编码与融合分支，从而在保证不同获取顺序下预测一致性的同时实现高效的多模态推理。

    

    多模态系统通常会对所有可用的输入进行编码，即使其中一部分子集就足以完成预测。自适应获取可以通过利用逐步融合证据所产生的预测来决定下一步编码哪种模态以及何时停止，从而降低这种成本。然而，序列融合使这些预测具有顺序依赖性，因此基于这些预测的决策可能需要区分同一获取集合下阶乘数量级的不同获取历史。我们提出了SemARC，它将序列模态聚合器与自适应运行时控制器相结合，利用已获取的证据在每个模态的编码器运行之前对其进行选择。SeMA只执行被选中的编码器和融合分支，更新固定大小的状态，并在每次获取后进行预测，而无需重新计算先前的分支。我们在随机化的模态子集和顺序下对每个获取前缀进行监督，以鼓励模型在不同获取顺序下产生一致的预测。ARC结合了一个依赖于集合的

    arXiv:2610.07466v1 Announce Type: new  Abstract: Multimodal systems often encode every available input, even when a subset suffices for prediction. Adaptive acquisition can reduce this cost by using predictions from incrementally fused evidence to decide which modality to encode next and when to stop. However, sequential fusion makes these predictions order-dependent, so decisions based on them may need to distinguish factorially many histories of the same acquired set. We introduce SemARC, which couples a Sequential Modality Aggregator (SeMA) with an Adaptive Runtime Controller (ARC) and uses acquired evidence to select each modality before its encoder runs. SeMA executes only selected encoder and fusion branches, updates a fixed-size state, and predicts after each acquisition without recomputing earlier branches. We supervise every acquisition prefix under randomized modality subsets and orders to encourage consistent predictions across acquisition orders. ARC combines a set-dependen
    
[^220]: 基于神经加性模型的可解释超图学习

    Interpretable Hypergraph Learning via Neural Additive Models

    [https://arxiv.org/abs/2610.07458](https://arxiv.org/abs/2610.07458)

    该论文提出了一种固有可解释的超图学习框架 HGNAN，它将神经加性模型扩展到高阶关系数据，通过特征级非线性分解与超图感知的结构聚合相结合，在实现节点级和超边级任务透明预测的同时保持良好的预测性能。

    

    超图为建模网络数据提供了一个自然的框架，其中实体之间的依赖关系由高阶交互所主导。尽管超图神经网络等超图学习方法已展现出卓越的预测性能，但大多数现有方法依赖于黑盒式的消息传递架构，使得人们难以解耦节点属性与高阶结构信息各自的贡献。为应对这一挑战，我们提出了超图神经加性网络，这是一个面向超图结构数据学习的固有可解释框架。HGNAN 将经典的神经加性模型扩展到高阶关系数据，通过将按特征的非线性分解与超图感知的结构聚合相结合，实现了节点级和超边级任务的透明预测。在基准数据集上的大量实验表明，HGNAN 取得了具有竞争力的性能（原文摘要在此处截断）。

    arXiv:2610.07458v1 Announce Type: new  Abstract: Hypergraphs offer a natural framework for modeling networked data, where dependencies among entities are governed by higher-order interactions. While hypergraph learning methods such as hypergraph neural networks have demonstrated remarkable predictive performance, most existing approaches rely on black-box message-passing architectures, making it difficult to disentangle the contributions of node attributes and higher-order structural information. To address this challenge, we introduce the hypergraph neural additive network (HGNAN), an inherently interpretable framework for learning on hypergraph-structured data. HGNAN extends classical neural additive models to higher-order relational data by integrating feature-wise nonlinear decomposition with hypergraph-aware structural aggregation, enabling transparent prediction for both node- and hyperedge-level tasks. Extensive experiments on benchmark datasets demonstrate that HGNAN achieves p
    
[^221]: AlignQuant：面向高效大语言模型生成的瓦片对齐混合精度量化

    AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation

    [https://arxiv.org/abs/2610.07457](https://arxiv.org/abs/2610.07457)

    AlignQuant提出了一种以GPU兼容的二维权重瓦片作为精度分配、存储和执行公共单元的训练后混合精度量化方法，使大语言模型的压缩能够真正转化为实际推理加速。

    

    细粒度混合精度量化有望实现高效的大语言模型推理，但局部的精度选择可能与GPU规则的存储和计算单元发生冲突。这种精度边界的不匹配限制了压缩向实际加速的转化。我们提出AlignQuant，一种训练后量化方法，它使用GPU兼容的二维权重瓦片作为精度分配、紧凑存储和执行的公共单元。这种共享划分使精度分配能够跟随输出通道内的敏感度变化。联合预填充/解码校准方法在量化激活条件下，使用由语言模型损失梯度加权的投影输出扰动来评估精度降低的影响。相位归一化的评分在模型级权重存储预算下，优先为对任一阶段重要的瓦片分配更高精度。每个瓦片存储一种选定的表示，而相位专用内核则重用打包的（摘要在此处被截断）

    arXiv:2610.07457v1 Announce Type: cross  Abstract: Fine-grained mixed-precision quantization promises efficient large language model inference, but local precision choices can conflict with regular GPU storage and computation units. This precision-boundary mismatch limits the translation of compression into practical acceleration. We introduce AlignQuant, a post-training quantization method that uses GPU-compatible two-dimensional weight tiles as the common unit of precision allocation, compact storage, and execution. This shared partition lets precision follow sensitivity within output channels. Joint prefill/decode calibration scores precision reductions using projection-output perturbations weighted by language-model loss gradients under quantized activations. Phase-normalized scores prioritize higher precision for tiles important to either phase under a model-wide weight-storage budget. Each tile stores one selected representation, while phase-specialized kernels reuse the packed m
    
[^222]: 面向降低参与者负担的代价高效时序预测的主动特征获取

    Active Feature Acquisition for Cost-Efficient Temporal Prediction with Reduced Participant Burden

    [https://arxiv.org/abs/2610.07452](https://arxiv.org/abs/2610.07452)

    该论文提出纵向主动特征获取（LAFA）方法，通过学习一种策略在每个时间点仅选择性地采集最优的条目动态子集，从而在降低参与者负担、减少无应答与流失风险的同时，保持对心理病理结果的准确预测能力。

    

    arXiv:2610.07452v1 公告类型：cross 摘要：准确预测病理结果是心理学中的一个核心问题。为此，心理学家通常会收集密集的纵向数据。然而，在此类研究中，为了实现准确预测而获取大量变量的愿望，往往与最小化参与者负担的需求相冲突。每次测量获取更多变量可以带来更好的预测效果，但过多的测量会增加无应答和被试流失的风险。纵向主动特征获取（Longitudinal Active Feature Acquisition, LAFA）是一种解决这一难题的规范化方法。LAFA 不再要求参与者在每次测量时回答所有条目，而是生成一种策略，在每个时间点最优地选择需要获取的条目动态子集，同时保留对特定结果进行预测的能力。然而，现有的 LAFA 方法大多基于神经网络（NN），在实际应用中难以解释。在此……

    arXiv:2610.07452v1 Announce Type: cross  Abstract: Accurate forecasting of pathological outcomes is a central problem in psychology. To do so, psychologists often collect intensive longitudinal data. However, in such studies, the desire to acquire a large number of variables for the sake of accurate prediction is often counteracted by the need to minimize participant burden. Acquiring more variables per occasion can yield better predictions, but having too many acquisitions increase the risk of non-response and attrition. Longitudinal Active Feature Acquisition (LAFA) is a principled approach to resolve this conundrum. Instead of requiring responses to every item at every acquisition occasion, LAFA produces a policy that seeks to optimally select dynamic subsets of items to be acquired at each timepoint while preserving our ability to forecast a specific outcome. However, existing LAFA methods are mostly based on Neural Networks (NN) that are difficult to interpret in practice. In this
    
[^223]: Fork-and-Flush：逃离自动科研智能体中的“想法盆地”

    Fork-and-Flush: Escaping Idea Basins in Autoresearch Agents

    [https://arxiv.org/abs/2610.07447](https://arxiv.org/abs/2610.07447)

    针对自动科研智能体在同一任务上独立运行时因陷入“想法盆地”而导致分数差距巨大且难以通过追加算力弥合的问题，本文提出周期性的“分叉-冲刷”（fork-and-flush）方法——将智能体分叉为多条继承工作空间但重置对话上下文的并行轨迹，再从中择优继续——有效帮助智能体逃离解空间局部区域并提升最终表现。

    

    自动科研智能体通过反复提出候选解决方案、对其进行评估，并利用反馈来指导后续实验，从而解决开放式问题。我们表明，同一智能体在同一任务上的独立运行往往停留在差异显著的分数水平上，且即使投入大量额外算力，这种差距依然持续存在。通过功能相似性对候选产物进行嵌入提供了进一步的证据：智能体的轨迹始终停留在解空间的局部区域，我们将其称为“想法盆地”。为了帮助智能体逃离这些盆地，我们研究了一种简单的周期性干预方法——fork-and-flush（分叉-冲刷）。该方法将智能体分叉为多条并行轨迹，每条轨迹继承已积累的工作空间，但从全新的对话上下文开始。在每条轨迹运行固定时长后，智能体从得分最高的那条轨迹继续运行。在13个长时程研究与工程任务上，单次智能体运行持续长达……

    arXiv:2610.07447v1 Announce Type: new  Abstract: Autoresearch agents tackle open-ended problems by repeatedly proposing candidate solutions, evaluating them, and using feedback to guide subsequent experiments. We show that independent runs of the same agent on the same task often plateau at substantially different scores, with gaps that persist even after considerable additional compute. Embedding their candidate artifacts by functional similarity provides further evidence that trajectories remain in localized regions of the solution space, which we call idea basins. To help agents escape these basins, we study a simple periodic intervention, fork-and-flush. Our method forks the agent into parallel trajectories, each inheriting the accumulated workspace but starting with a fresh chat context. After running each trajectory for a fixed horizon, the agent continues from the highest-scoring one. Across 13 long-horizon research and engineering tasks, with individual agent runs lasting up to
    
[^224]: 解耦“做什么”与“在哪里做”：小型GUI定位模型应如何接收动作类型？

    Decoupling What from Where: How Should a Small GUI Grounding Model Receive the Action Type?

    [https://arxiv.org/abs/2610.07444](https://arxiv.org/abs/2610.07444)

    该论文系统比较了向小型GUI定位模型传递动作类型的五种机制，发现辅助损失、可加性嵌入和提示词注入各带来5-7个hit@0.10百分点的显著提升，而硬路由和前置token无效，且相当一部分提升源于对屏幕外触点钳制这一预处理选择的防护而非空间先验。

    

    GUI智能体需要决定采取什么动作以及在哪里执行；我们探究小型定位模型应以何种方式接收动作类型信息。我们在Android in the Wild（AITW）数据集上使用LoRA微调Qwen2-VL-2B，在数据、算力和解码条件完全匹配的前提下，将无动作类型信息的平铺基线与五种注入动作类型的方式进行比较：辅助损失、硬路由动作词、可加性学习的嵌入、前置学习token，以及将类型写入提示词。通过五个随机种子、按回合聚类的自助法（bootstrap）以及种子级配对检验，混合数据流上的排序十分清晰：辅助损失、可加性嵌入和提示词方式相比基线各带来5至7个hit@0.10百分点的提升，而硬路由和前置token方式与基线无显著差异。这些提升的很大部分来自对我们某项预处理选择的防护，而非空间先验：我们的序列化器将AITW记录的type事件中屏幕外触点钳制到原点，该钳制处理……（摘要原文在此处截断）

    arXiv:2610.07444v1 Announce Type: new  Abstract: A GUI agent decides which action to take and where to take it; we ask how a small grounding model should receive the action type. Fine-tuning Qwen2-VL-2B with LoRA on Android in the Wild, we compare a flat baseline with five ways of supplying the type under matched data, compute, and decoding: an auxiliary loss, a hard-routed action word, an additive learned embedding, a prepended learned token, and the type written into the prompt. With five seeds, an episode-clustered bootstrap, and seed-level paired tests, the ranking on a mixed stream is clear: the auxiliary loss, the additive embedding, and the prompt word each gain five to seven hit@0.10 points over the baseline, while hard routing and the prepended token are not distinguishable from it. Much of that gain is protection from a preprocessing choice of ours rather than a spatial prior. Our serializer clamps the off-screen touch point AITW records for type events to the origin; that cl
    
[^225]: 伪迹去除改善了皮电波形，但无法改善虚拟现实平衡任务中的下游分类

    Artifact removal improves electrodermal waveforms but not downstream classification in a virtual-reality balance task

    [https://arxiv.org/abs/2610.07438](https://arxiv.org/abs/2610.07438)

    该研究发现在虚拟现实平衡任务中，尽管伪迹去除显著改善了皮电信号的波形质量（伪迹区域误差降低17.8%），但这一改善未能转化为下游分类性能的提升，表明波形层面的优化并不必然带来决策层面的益处。

    

    伪迹去除通常在皮电活动（EDA）分类之前进行，其假设是更干净的信号能够支持更好的决策。我们在虚拟现实（VR）平衡干扰任务中检验了这一假设。研究在一个带有专家校正EDA的基准数据集上训练了一个残差门控网络，将其冻结后应用于VR记录数据，并在完全相同的留一被试交叉验证评估下，使用五种已发表的时间序列方法对原始信号和门控信号分别进行分类。在基准数据集上，门控网络能够很好地检测伪迹（中位记录AUROC为0.94），并将伪迹区域内的误差降低了17.8%。然而在VR任务中，它并未改善分类效果：平衡准确率的变化范围在-1.35到+0.93个百分点之间，没有任何分类器获得改善，其中两个分类器甚至损失了准确率，且所有五种分类器在±3.32个百分点范围内与使用原始输入的结果等价。这一益处在从波形到决策的传递过程中丢失了。那些降低了波形误差的校正（摘要原文在此处截断）……

    arXiv:2610.07438v1 Announce Type: cross  Abstract: Artifact removal routinely precedes the classification of electrodermal activity (EDA), on the assumption that a cleaner signal supports a better decision. We tested this assumption in a virtual-reality (VR) balance-disturbance task. A residual gating network was trained on a benchmark with expert-corrected EDA, frozen, and applied to VR recordings, where raw and gated signals were classified by five published time-series methods under identical leave-one-participant-out evaluation. On the benchmark the gate detected artifacts well (median record AUROC 0.94) and reduced error inside artifact regions by 17.8%. In the VR task it did not improve classification. Changes in balanced accuracy ranged from -1.35 to +0.93 percentage points, no classifier improved and two lost accuracy, and all five were equivalent to raw input within +/- 3.32 points. The benefit was lost between waveform and decision. The correction that lowered waveform error 
    
[^226]: StaFIR：平稳性感知因果滤波器的凸学习

    StaFIR: Convex Learning of Stationarity-Aware Causal Filters

    [https://arxiv.org/abs/2610.07430](https://arxiv.org/abs/2610.07430)

    StaFIR提出一种通过凸优化学习非负指数滞后分布混合的因果FIR滤波器，在经验平稳性与输入保留之间取得平衡，并能根据时间序列的持久性自适应地调整滤波强度。

    

    减少持久性时间序列中的非平稳性，需要决定要去除其中多少时间依赖性。在金融领域，分数差分通常使用增广迪基-福勒（ADF）检验进行调优，这将搜索限制在单参数的滞后分布族中，且仅间接地处理输入保留问题。我们提出StaFIR，一种具有学习到的非负指数滞后分布混合的因果有限冲激响应（FIR）滤波器。其凸学习目标在经验平稳性与输入相似性之间取得平衡。我们在ARFIMA-GARCH受控设定和滚动金融序列上对StaFIR进行评估，包括一个已实现波动率预测任务。实验表明，StaFIR能够根据序列的持久性调整其滤波强度，同时在平稳状态下限制不必要的变换。在下游预测中，其精度与固定半阶差分相比没有明显差异，而StaFIR在（摘要在此处被截断）……

    arXiv:2610.07430v1 Announce Type: new  Abstract: Reducing nonstationarity in a persistent time series entails deciding how much of its temporal dependence to remove. In finance, fractional differencing is often tuned using the Augmented Dickey--Fuller (ADF) test, limiting the search to a one-parameter family of lag profiles and addressing input preservation only indirectly. We propose StaFIR, a causal finite-impulse-response filter with a learned nonnegative mixture of exponential lag profiles. Its convex learning objective balances empirical stationarity with similarity to the input. We evaluate StaFIR on ARFIMA--GARCH controlled settings and rolling financial series, including a realized-volatility forecasting task. The experiments show that StaFIR adjusts its filtering strength to persistence while limiting unnecessary transformation in stationary regimes. In downstream forecasting, there is no clear accuracy difference from fixed half-order differencing, while StaFIR achieves highe
    
[^227]: AccentCL：具有增量扩展能力的鲁棒口音分类

    AccentCL: Robust Accent Classification with Incremental Expansion

    [https://arxiv.org/abs/2610.07426](https://arxiv.org/abs/2610.07426)

    提出了AccentCL框架，通过不平衡感知损失、领域均值对齐损失和基于回放的持续学习，实现了对类别不平衡和跨语料库领域偏移具有鲁棒性的英语口音分类，并支持新口音类别的增量扩展。

    

    口音分类器通常使用固定的标签集合进行训练，无法在新的数据出现时容纳新的口音类别。此外，由于各语料库之间录音条件的差异，带口音的语音语料库往往存在显著的类别不平衡和/或领域偏移。我们提出了AccentCL，一个用于英语口音分类的类增量学习框架，它对类别不平衡和跨语料库领域偏移具有鲁棒性。AccentCL从冻结的Whisper-Large-v3编码器中提取多层表示，并通过不平衡感知的交叉熵损失进行优化，以减少对多数口音类别的偏差，同时使用领域均值对齐损失来最小化训练语料库之间的分布均值偏移。随后通过基于回放的持续学习来扩展标签空间，利用冻结的基础模型进行知识保留，并使用新旧间隔损失来减少对新添加类别的过度预测。在……

    arXiv:2610.07426v1 Announce Type: new  Abstract: Accent classifiers are typically trained with a fixed label inventory and cannot accommodate new accent categories as new data becomes available. Moreover, accented speech corpora often exhibit substantial class imbalance and/or domain shift due to differences in recording conditions across corpora. We present AccentCL, a class-incremental learning framework for English accent classification that is robust to class imbalance and cross-corpus domain shift. AccentCL extracts multi-layer representations from a frozen Whisper-Large-v3 encoder, optimized with an imbalance-aware cross-entropy loss to reduce bias toward the majority accent classes and a domain mean alignment loss that minimizes distributional mean shift across training corpora. The label space is then expanded via replay-based continual learning, using the frozen base model for knowledge retention and an old-to-new margin loss to reduce overprediction on newly added classes. On
    
[^228]: 面向EEG脑机接口解码的标签揭示式在线更新基准测试

    Benchmarking Label-Revealed Online Updates for EEG BCI Decoding

    [https://arxiv.org/abs/2610.07420](https://arxiv.org/abs/2610.07420)

    该论文提出了EEG脑机接口解码的在线自适应基准测试，系统比较了CSP与黎曼协方差两类方法、受控遗忘及冷启动策略，发现标签揭示式在线更新能在14个模型/数据集对中的13个上带来最高约18%的相对准确率提升。

    

    脑电图（EEG）信号会随时间发生漂移，这可能导致静态的脑机接口（BCI）模型在实际应用中性能下降。我们提出了一个在线自适应的基准测试，并在按时间顺序的预序列（先测试后训练）评估下，比较了两种广泛使用的流程体系：共空间模式（CSP）和基于黎曼协方差的方法。我们考察了：(i) 哪些流程最能从标签揭示式更新中受益，(ii) 对旧数据进行受控遗忘是否能提高鲁棒性，(iii) 最小校准的冷启动与从预训练模型开始相比效果如何。在四个数据集（三个运动想象数据集和一个运动解码数据集）上，标签揭示式在线更新在两个最大的数据流上改进了14个模型/数据集对中的13个，相对于冻结模型的相对准确率提升最高达约18%。基于Shapley值、按时间块进行的数据价值分析将最大的平均价值分配给了……（摘要原文在此截断）

    arXiv:2610.07420v1 Announce Type: new  Abstract: Electroencephalography (EEG) signals drift over time, which can cause static brain-computer interface (BCI) models to degrade in practice. We present a benchmark for online adaptation and compare two widely used pipeline families, Common Spatial Patterns (CSP) and Riemannian covariance-based methods, under time-ordered prequential (test-then-train) evaluation. We examine (i) which pipelines benefit most from label-revealed updates, (ii) whether controlled forgetting of older data improves robustness, and (iii) how a minimal-calibration cold start compares with starting from a pretrained model. Across four datasets (three motor-imagery datasets and one movement-decoding dataset), label-revealed online updates improve 13 of 14 model/dataset pairs on the two largest streams, with relative accuracy gains of up to about 18% over a frozen model. A Shapley-based data-valuation analysis over temporal blocks assigns the largest mean value to the 
    
[^229]: 可学习谱激活

    Learnable Spectral Activations

    [https://arxiv.org/abs/2610.07419](https://arxiv.org/abs/2610.07419)

    提出可学习谱激活（LSA），用谐波振幅可学习的残差截断傅里叶级数替代固定激活函数，将特征选择与频谱整形解耦为两条独立梯度更新的通路，从而提升隐式神经表示对局部化和空间变化信号的多谐波组合能力。

    

    隐式神经表示（INR）由其输入编码和激活函数所诱导的频谱结构塑造。现有方法主要通过修改网络可用的频率来改进拟合效果，例如采用坐标编码或周期非线性函数。然而，频率可及性并非唯一的瓶颈：具有局部化或空间变化结构的信号，需要网络能够高效地将多个频率组合成多谐波的内部响应。我们提出可学习谱激活（LSA），用残差截断傅里叶级数替代固定的神经元级非线性函数，其谐波振幅在训练过程中进行学习。LSA并不扩展渐近函数类，而是改变表示的分解方式：线性权重负责选择特征，而激活系数控制频谱整形，二者通过相互独立的梯度进行更新。由于……（原文摘要在此处截断）

    arXiv:2610.07419v1 Announce Type: new  Abstract: Implicit neural representations (INRs) are shaped by the spectral structure induced by their input encodings and activation functions. Existing methods improve fitting primarily by modifying which frequencies are available to the network, through coordinate encodings or periodic nonlinearities. However, frequency access is not the only bottleneck: signals with localized or spatially varying structure require the network to efficiently compose frequencies into multi-harmonic internal responses. We introduce learnable spectral activations (LSA), which replace fixed neuron-level nonlinearities with a residual truncated Fourier series whose harmonic amplitudes are learned during training. LSA does not expand the asymptotic function class. Instead, it changes the factorization of the representation: linear weights select features while activation coefficients control spectral shaping, and the two are updated by separate gradients. Because the
    
[^230]: 基于稀疏RKHS流形的函数空间贝叶斯优化

    Bayesian Optimization on Function Spaces via Sparse RKHS Manifolds

    [https://arxiv.org/abs/2610.07417](https://arxiv.org/abs/2610.07417)

    提出L0MO方法，通过在RKHS中由核函数稀疏表示构成的流形子集上搜索，并同时优化核的位置与系数，从而实现函数空间中的贝叶斯优化，并为现有FBO方法提供了统一视角。

    

    贝叶斯优化（BO）已成为最小化向量输入黑箱函数的成熟方法论。然而，这一参数向量往往源于对本质上是函数关系的离散化。近期有多篇文章研究了函数贝叶斯优化（FBO）的设定，其中待优化的变量不是有限维向量空间中的元素，而是无限维函数空间中的函数。在本工作中，我们提出 $L^0$ 流形优化（L0MO），这是一种简单的FBO方法，它在再生核希尔伯特空间（RKHS）中由具有核函数稀疏表示的函数构成的子集上进行搜索，同时优化核函数的位置及其系数。我们详细讨论了本方法与现有方法之间的关系，提供了一个统一的视角来审视先前的工作。为了将我们的方法与最先进的技术进行评估比较，

    arXiv:2610.07417v1 Announce Type: cross  Abstract: Bayesian Optimization (BO) has become an established methodology for minimizing black-box functions of a vector input. Often, however, this parameter vector arises from the discretization of an inherently functional relationship. Several recent articles have considered the Functional Bayesian Optimization (FBO) setting, in which the variable to be optimized is not a member of a finite dimensional vector space, but rather an infinite dimensional function space. In this work, we propose $L^0$ Manifold Optimization (L0MO), a simple approach to FBO which searches the subset of a Reproducing Kernel Hilbert Space (RKHS) consisting of functions with a sparse representation in the kernel functions, optimizing both the kernel locations and their coefficients. We discuss in detail the relationship between our method and existing ones, providing a unifying lens through which to view prior works. To assess our method against the state of the art, 
    
[^231]: 基于表格基础模型的主动特征获取策略评估

    Evaluation of Active Feature Acquisition Policies with Tabular Foundation Models

    [https://arxiv.org/abs/2610.07406](https://arxiv.org/abs/2610.07406)

    本文发现在离线数据覆盖不均衡时，使用总预测熵作为奖励会混淆认知不确定性与偶然不确定性并产生偏差，因此提出用PFN输出的后验期望偶然熵来评估主动特征获取策略，从而更准确地识别信息量大的特征。

    

    主动特征获取通过学习策略来顺序地获取特征，以最大化关于目标变量的信息。我们研究了如何利用先验数据拟合网络（PFNs）从有限的离线数据中学习和评估此类策略，PFNs是一类无需针对具体任务进行训练即可输出后验预测分布的现成模型。我们证明，在离线数据覆盖不均衡的情况下，使用总预测熵作为奖励会产生一种认知偏差，从而惩罚对稀疏观测特征的获取。具体而言，这种奖励将认知不确定性（源于离线数据的缺乏）与偶然不确定性（源于特征本身信息量不足）混为一谈。为解决这一问题，我们采用后验期望（偶然）熵而非PFN输出的总预测熵来评估特征获取。在合成数据集和真实数据集上的实证评估表明，我们的方法表现始终更优。

    arXiv:2610.07406v1 Announce Type: new  Abstract: Active feature acquisition learns policies that sequentially acquire features to maximize information about a target variable. We study how to learn and evaluate such policies from finite offline data using prior-data fitted networks (PFNs), which are off-the-shelf models that output posterior predictive distributions without task-specific training. We show that under the imbalanced coverage of offline data, using total predictive entropy as a reward creates an epistemic bias that penalizes acquiring sparsely observed features. Specifically, this reward conflates epistemic uncertainty (arising from lack of offline data) with aleatoric uncertainty (arising from uninformative features). To address this, we target the posterior expected (aleatoric) entropy instead of the total predictive entropy output by a PFN for evaluating feature acquisitions. Empirical evaluations on synthetic and real-world datasets demonstrate that our approach consi
    
[^232]: pass@k 无法衡量什么：评估后训练后的多样性与能力保持

    What pass@k Cannot Measure: Evaluating Diversity and Capability Retention after Post-Training

    [https://arxiv.org/abs/2610.07405](https://arxiv.org/abs/2610.07405)

    论文指出 pass@k 只反映解题概率而忽略输出分布的多样性，并通过实验证明 GRPO 与 RFT 两种后训练方法虽在 pass@k 上表现相似，却使多样性指标朝相反方向变化，说明仅用 pass@k 评估后训练效果是不充分的。

    

    pass@k，即模型在 k 次采样尝试内解决问题的比例，是该领域判断基于可验证奖励的强化学习（RL）后训练是否改进了模型的默认评估协议。在总体层面上，pass@k 仅取决于某个问题获得正确样本的概率，而没有考虑该概率在输出之间的分布方式。我们证明这一差距并非只是理论上的。使用组相对策略优化（GRPO）以及拒绝采样微调（RFT，即在模型自身最短的通过验证器验证的 rollout 上进行训练）在小学数学上训练 Qwen2.5-1.5B-Instruct，会使三种互补的多样性度量（token 级熵、答案级熵、每个提示的唯一答案数）朝相反方向移动，且每种方法各三个随机种子之间结果零重叠。即使仅限于通过验证器的正确补全，这一差距依然存在（在控制长度后，GRPO 产生的正确解答中的词汇多样性要低 15%）……

    arXiv:2610.07405v1 Announce Type: new  Abstract: pass@$k$, the fraction of problems a model solves within $k$ sampled attempts, is the field's default protocol for deciding whether reinforcement-learning (RL) post-training on verifiable rewards improved a model. At the population level, pass@$k$ depends only on a problem's probability of a correct sample, with no term for how it is distributed across outputs. We show this gap is not academic. Training Qwen2.5-1.5B-Instruct on grade-school math with Group Relative Policy Optimization (GRPO) and with rejection-sampling fine-tuning (RFT, training on the model's own shortest verifier-passed rollout) moves three complementary diversity measures (token-level entropy, answer-level entropy, unique answers per prompt) in opposite directions, with zero overlap across three seeds per arm. The gap survives restricting to verifier-correct completions only (lexical diversity among correct solutions is 15% lower for GRPO, after controlling for length
    
[^233]: Fed-BRDECS：隐私保护且感知异构性的联邦深度嵌入聚类

    Fed-BRDECS: Privacy-Preserving and Heterogeneity-Aware Federated Deep Embedded Clustering

    [https://arxiv.org/abs/2610.07399](https://arxiv.org/abs/2610.07399)

    Fed-BRDECS通过用局部可计算的样本稳定性损失替代依赖全局软分配统计的聚类目标，并结合预测均衡采样与质心级重启机制，实现了隐私保护且能应对客户端数据异构性的联邦深度嵌入聚类。

    

    联邦深度聚类旨在从分散的无标签数据中学习适合聚类的表示，同时保护客户端隐私。然而，深度嵌入聚类（DEC）风格的目标函数依赖于全局软分配统计量，这要求客户端暴露其敏感信息。我们提出了Fed-BRDECS，一个隐私保护且感知异构性的联邦深度嵌入聚类框架。Fed-BRDECS用局部可计算的样本稳定性损失替代全局归一化的聚类目标，从而避免了本地软分配分布的传输。为应对非独立同分布（non-IID）的客户端数据分布，我们引入了预测均衡采样，它无需真实标签即可对本地稀有的预测簇进行过采样；同时引入质心级重启机制，定期刷新有偏或不活跃的质心。在图像和文本聚类基准上的实验表明，Fed-BRDECS持续优于……

    arXiv:2610.07399v1 Announce Type: new  Abstract: Federated deep clustering seeks to learn clustering-friendly representations from decentralized unlabeled data while preserving client privacy. However, Deep Embedded Clustering (DEC)-style objectives depend on global soft-assignment statistics that require clients to reveal their sensitive information. We propose Fed-BRDECS, a privacy-preserving and heterogeneity-aware federated deep embedded clustering framework. Fed-BRDECS replaces the globally normalized clustering objective with a locally computable sample-stability loss, avoiding the transmission of local soft-assignment distributions. To tackle non-IID client distributions, we introduce prediction-balanced sampling, which oversamples locally rare predicted clusters without requiring ground-truth labels, and centroid-level restarting, which periodically refreshes biased or inactive centroids. Experiments on image and text clustering benchmarks show that Fed-BRDECS consistently outp
    
[^234]: 关于使用聚合标准化流链进行复杂仿真模型似然近似与推断的视角研究

    A perspective note on likelihood approximation and inference for complex simulation models using a chain of aggregated normalizing flows

    [https://arxiv.org/abs/2610.07391](https://arxiv.org/abs/2610.07391)

    提出了一种基于n级聚合标准化流链的似然近似新方法，通过按顺序估计各组双射变换参数，为复杂仿真模型的大规模数据分析、假设检验和不确定性量化提供了可扩展且高效的解决方案。

    

    我们在基于仿真的推断框架内，针对似然近似问题提出了一种新视角，该视角促进了面向大规模数据分析的可扩展、可控的仿真流程，支持在高维空间中进行高效的参数空间探索或平滑插值，从而为假设检验和不确定性量化提供有效的统计处理手段。特别地，我们考虑一种由 $n$ 个聚合标准化流组成的链式似然近似方案，其中一组来自前向复杂仿真模型的预先复制观测数据集首先通过第一组双射变换，随后再依次通过后续的多组双射变换。这里，我们假设对于任意 $k \in \{1,\,2, \ldots, n\}$，前 $k$ 组双射变换所对应的参数按某种最优性意义依次进行估计……

    arXiv:2610.07391v1 Announce Type: cross  Abstract: We present a new perspective on the problem of likelihood approximation within the framework of simulation-based inference that promotes scalable and controllable simulation routines for large-scale data analysis, allows efficient parameter space exploration or smooth interpolation in high-dimensions and, thus, supports valid statistical treatments of hypothesis testings as well as uncertainty quantification. In particular, we consider a chain of $n$-aggregated normalizing flows for likelihood approximation scheme, where a set of upfront replicated observation datasets from the forward complex simulation model pass through the first set of bijective transformations, and then subsequently pass to the other sets of bijective transformations. Here, we assume that, for any $k \in \{1,\,2, \ldots, n\}$, the parameters corresponding to the first $k$ sets of bijective transformations are estimated sequentially, in some sense of optimality, fo
    
[^235]: 稀疏自编码器中作为自然梯度流的推理与学习

    Inference and learning in sparse autoencoders as natural gradient flow

    [https://arxiv.org/abs/2610.07389](https://arxiv.org/abs/2610.07389)

    该论文将稀疏自编码器的推理与字典学习统一为共享变分自由能上的自然梯度流，并提出无编码器的稀疏编码模型BeFOND，通过循环解释抵消机制减少重叠特征间的干扰、利用Fisher预条件化加速稀有特征学习，从而显著提升字典恢复与稀有特征检测能力。

    

    稀疏自编码器被广泛用于揭示神经网络中的可解释特征，然而当特征相互重叠或激活频率较低时，可靠的特征恢复仍然困难。这些挑战既涉及推断哪些特征解释了输入，也涉及学习表示这些特征的字典。在此，我们将推理和字典学习统一为共享变分自由能上的自然梯度流。我们将该框架实例化为BeFOND，一种无编码器的稀疏编码模型，具有闭式形式的推理和学习动力学。我们展示了循环的“解释抵消”机制如何减少重叠特征之间的干扰，而Fisher预条件化可以补偿稀有特征的缓慢学习。在合成数据上，BeFOND改进了字典恢复和稀有特征检测，且随着叠加程度增加，其相对于摊销基线的优势不断扩大。在语言模型激活上，它提升了单特征概念检测的性能。

    arXiv:2610.07389v1 Announce Type: cross  Abstract: Sparse autoencoders are widely used to uncover interpretable features in neural networks, yet reliable recovery remains difficult when features overlap or activate infrequently. These challenges involve both inferring which features explain an input and learning the dictionary that represents them. Here, we unify inference and dictionary learning as natural-gradient flows on a shared variational free energy. We instantiate this framework as BeFOND, an encoder-free sparse coding model with closed-form inference and learning dynamics. We show how recurrent explaining away reduces interference between overlapping features, while Fisher preconditioning can compensate for the slow learning of rare features. On synthetic data, BeFOND improves dictionary recovery and rare-feature detection, with a growing advantage over amortized baselines as superposition increases. On language-model activations, it improves single-feature concept detection 
    
[^236]: DeepAJM：面向不规则采样数据的深度关联联合模型

    DeepAJM: Deep Association Joint Model for Irregularly Sampled data

    [https://arxiv.org/abs/2610.07388](https://arxiv.org/abs/2610.07388)

    提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。

    

    联合模型同时建模纵向结局与生存结局，利用患者纵向轨迹中的模式来改进生存结局的预测。然而，经典的参数化联合模型依赖于固定的参数假设，在模型误设和样本量较小的情况下容易产生偏差。我们提出了一种深度联合模型 DeepAJM，它不需要任何参数假设，同时保留了部分可解释的、针对每个纵向结局的关联结构。该联合模型采用编码器-解码器（序列到序列）架构来学习患者时变协变量轨迹中的潜在结构。模型通过一个学习得到的可解释关联结构将纵向过程与生存过程联系起来，其中解码器输出的每个纵向结果在贡献于（生存模型的）风险评分之前，会先由基线协变量进行重新调制……

    arXiv:2610.07388v1 Announce Type: cross  Abstract: Joint Models simultaneously model longitudinal and survival outcomes, leveraging patterns in patients' longitudinal trajectory to improve the prediction of survival outcomes. The classical parametric joint models, however, rely on fixed parametric assumptions, making them susceptible to bias under model misspecification and smaller sample sizes. We propose a deep joint model, DeepAJM, that does not require any parametric assumptions, while retaining a partially interpretable, per-longitudinal-outcome association structure. The joint model uses an encoder-decoder (sequence-to-sequence) architecture to learn the latent structure in patients' time-varying covariate trajectories. The model links the longitudinal processes to the survival processes through a learned interpretable association structure, in which each longitudinal output from the decoder gets remodulated by baseline covariates before it contributes to the risk scores from the
    
[^237]: HyperNSDE：用于静态-纵向临床数据联合生成的个性化神经SDE

    HyperNSDE: Personalized Neural SDEs for Joint Static-Longitudinal Clinical Data Generation

    [https://arxiv.org/abs/2610.07383](https://arxiv.org/abs/2610.07383)

    HyperNSDE通过超网络将静态患者特征注入潜在神经SDE，首次联合生成异构静态协变量、不规则采样的纵向轨迹和观测时间三类紧密耦合的临床数据，为医疗AI提供真实且保护隐私的合成患者数据。

    

    合成患者数据生成是解决医疗机器学习中数据稀缺与隐私限制双重挑战的一种有前景的方案。要真实地合成患者级临床数据，需要联合建模异构的静态协变量、不规则采样的纵向轨迹以及具有信息量的观测时间——这三者在实践中紧密耦合，却很少被共同处理。我们提出HyperNSDE，一种连续时间生成模型，它通过超网络将潜在神经随机微分方程条件化于静态患者表示之上，使基线特征能够在初始条件之外塑造轨迹演化，而无需轨迹编码器，同时随机潜在动力学捕捉生成路径中的真实变异性。观测时间通过依赖于潜在状态的强度过程进行联合建模，不规则随机路径上的训练则通过确定性方法加以稳定……

    arXiv:2610.07383v1 Announce Type: cross  Abstract: Synthetic patient data generation is a promising solution to the dual challenge of data scarcity and privacy constraints in healthcare machine learning. Realistic synthesis of patient-level clinical data requires jointly modeling heterogeneous static covariates, irregularly sampled longitudinal trajectories, and informative observation times - three tightly coupled components in practice yet rarely addressed together. We propose HyperNSDE, a continuous-time generative model that conditions a latent Neural SDE on static patient representations through a hypernetwork, allowing baseline characteristics to shape trajectory evolution beyond the initial condition without requiring a trajectory encoder, while stochastic latent dynamics capture realistic variability in generated paths. Observation times are modeled jointly through a latent-state-dependent intensity process, and training on irregular stochastic paths is stabilized via a determi
    
[^238]: 多群体公平性与全知预测：分离与等价

    Multigroup Fairness and Omniprediction: Separations and Equivalences

    [https://arxiv.org/abs/2610.07374](https://arxiv.org/abs/2610.07374)

    该论文探究全知预测是否必须依赖多重准确性、多重校准等多群体公平性概念，并研究两者之间的分离与等价关系。

    

    全知预测是一种学习保证，它要求单个预测器在面对从损失函数族中选出的任意损失时，都能与基准假设类中的最佳假设相媲美。损失结果不可区分性（简称损失OI）是一个更强的概念，它蕴含全知预测，其要求在依赖损失函数和基准类的测试下，标签的预测分布与真实分布不可区分。多重准确性和多重校准是多群体公平性概念，它们推广了经典的校准和期望准确性概念。大多数已知的全知预测学习算法（无论是针对标准概念还是针对损失OI等强化版本）都依赖于这些多群体公平性概念的某种版本，或依赖于一种称为校准多重准确性的中间概念。我们提出这样的问题：这是否是必要的？全知预测是否需要某种形式的多群体公平性？

    arXiv:2610.07374v1 Announce Type: new  Abstract: Omniprediction is a learning guarantee which requires a single predictor to be competitive relative to the best hypothesis from a benchmark class for any loss chosen from a family of loss functions. Loss Outcome Indistinguishability (loss OI for short) is a stronger notion that implies omniprediction. It requires the predicted distribution on labels to be indistinguishable from the true distribution to tests that depend on the loss functions and the benchmark class. Multiaccuracy and multicalibration are multigroup fairness notions that generalize classical notions of calibration and accuracy in expectation. Most known learning algorithms for omniprediction (both for the standard notion and for strengthenings like loss OI) rely on some version of these multigroup fairness notions, or on an intermediate notion called calibrated multiaccuracy. We ask if this is necessary: Does omniprediction require some form of multigroup fairness?   We s
    
[^239]: 面向开放集行人重识别的身份条件化分数融合

    Identity-Conditioned Score Fusion for Open-Set Person Re-Identification

    [https://arxiv.org/abs/2610.07366](https://arxiv.org/abs/2610.07366)

    提出一种无需训练的身份条件化分数融合框架，通过为每个图库身份定制融合权重并结合查询条件自适应，在换装行人重识别基准上将误非识别率最多降低8.8%。

    

    鲁棒的行人重识别通常融合人脸、步态和体型等互补线索。虽然自适应融合通常针对查询质量进行优化，但模型强度在不同身份之间也存在差异。我们提出了身份条件化分数融合，这是一个无需训练即可为每个图库身份定制权重的框架。通过将身份内一致性与跨身份冒名者进行对比，该方法提取出身份特定的特征轮廓，并通过无参数规则与查询条件自适应相结合，从而在保持分数校准性的同时扩大真实匹配与错误匹配之间的分离度。在三个换装行人重识别基准上的评估表明，我们的方法始终优于统计方法、基于排序的方法和学习型基线，误非识别率绝对降低最多达8.8%，展示了身份条件化融合在开放集行人重识别中的价值。

    arXiv:2610.07366v1 Announce Type: cross  Abstract: Robust person re-identification often combines complementary cues such as face, gait, and body shape. While adaptive fusion typically targets query quality, model strength also varies across identities. We introduce identity-conditioned score fusion, a framework that tailors weights to each gallery identity without training. By contrasting intra-identity consistency against cross-identity impostors, it extracts identity-specific profiles that couple with query-conditioned adaptation via a parameter-free rule. This widens the separation between true and false matches while preserving score calibration. Evaluations on three clothes-changing person re-identification benchmarks show that our method consistently outperforms statistical, rank-based, and learned baselines, achieving up to an 8.8% absolute reduction in the false non-identification rate and demonstrating the value of identity-conditioned fusion in open-set person re-identificat
    
[^240]: 硬资源约束下大语言模型评估的动态预算分配

    Dynamic Budget Allocation for LLM Evaluation under Hard Resource Constraints

    [https://arxiv.org/abs/2610.07362](https://arxiv.org/abs/2610.07362)

    提出HARP方法，在硬资源约束下为大语言模型多轮交互评估实现动态预算分配，通过自适应重新分配未使用预算来构建时间-事件的下预测边界。

    

    我们通过时间-事件指标来评估大语言模型（LLM）在多轮交互中的表现：即产生感兴趣事件（如成功越狱或智能体任务完成）所需的交互步骤数。在计算资源有限的情况下，交互可能在事件发生之前被终止，因此事件时间只能被部分观测（即删失数据）。现有的用于校准时间-事件边界的分配方法只能在期望意义上满足预算，并可能在某次具体的评估运行中超出可用预算。强制执行硬约束尤其具有挑战性，因为每条轨迹的成本最初是未知的。我们提出了带重流的预测校准硬预算分配（HARP），这是一种满足硬资源约束并自适应重新分配未使用预算的预算分配方法。我们展示了如何使用HARP来构建时间-事件的下预测边界（LPB），并估计评估指标……

    arXiv:2610.07362v1 Announce Type: new  Abstract: We evaluate large language models (LLMs) in multi-turn interactions through their time-to-event: the number of interaction steps required to produce an event of interest, such as a successful jailbreak or agentic task completion. Under limited compute, interactions may be terminated before the event occurs, so that event times are only partially observed (censored). Existing allocation methods for calibrating time-to-event bounds satisfy the budget only in expectation and can exceed the available budget on a particular evaluation run. Enforcing a hard constraint is particularly challenging as the cost of a trajectory is initially unknown. We introduce Hard-budget Allocation with Reflow for Predictive calibration (HARP), a budget allocation that satisfies hard resource constraints and adaptively reallocates unused budget. We show how to use HARP to construct lower predictive bounds (LPBs) on the time-to-event and to estimate evaluation me
    
[^241]: 面向数据驱动的火灾后泥石流预测的可解释基准测试

    Towards Explainable Benchmarking for Data-driven Post-Wildfire Debris Flow Prediction

    [https://arxiv.org/abs/2610.07358](https://arxiv.org/abs/2610.07358)

    本文提出了一个统一的可解释数据驱动火灾后泥石流预测基准，解决了现有研究在特征空间、模型架构和评估协议上的碎片化问题，并可系统分析气象、地形、土壤和燃烧严重程度等异质性因素对触发泥石流的相对重要性。

    

    火灾后泥石流（PFDFs）是一种破坏性的含沉积物灾害，当强降雨袭击近期被烧毁的地形时被触发，导致山坡失稳，并威胁基础设施、当地经济和社区安全。已有研究提出数据驱动方法，直接从历史PFDF观测数据中学习预测模式。然而，当前数据驱动PFDF预测的研究现状在特征空间、模型架构和评估协议方面高度碎片化，使得严格的比较和科学见解的提炼变得困难。此外，现有研究缺乏对异质性因素（如气象条件、地形特征、土壤性质和燃烧严重程度）在触发PFDF中相对重要性的系统性研究。为了解决这些局限性，我们提出了一个统一的数据驱动PFDF预测基准，从而实现对不同方法的公平而全面的评估。

    arXiv:2610.07358v1 Announce Type: new  Abstract: Post-wildfire debris flows (PFDFs) are destructive sediment-laden hazards triggered when intense rainfall strikes recently burned terrain, destabilizing hillslopes and threatening infrastructure, local economies, and community safety. Data-driven methods have been proposed to learn predictive patterns directly from historical PFDF observations. However, the current research landscape of data-driven PFDF prediction remains highly fragmented across feature spaces, model architectures, and evaluation protocols, making rigorous comparison and the derivation of scientific insights difficult. Moreover, existing studies lack a systematic investigation into the relative importance of heterogeneous factors (e.g., meteorological conditions, terrain characteristics, soil properties, and burn severity) in triggering PFDF. To address these limitations, we present a unified benchmark for data-driven PFDF prediction, enabling fair and comprehensive eva
    
[^242]: RELACE：面向长程语言智能体的基于回溯似然的动作信用估计

    RELACE: retrospective likelihood-based action credit estimation for long-horizon language agents

    [https://arxiv.org/abs/2610.07349](https://arxiv.org/abs/2610.07349)

    RELACE 提出了一种无 critic 的信用估计框架，通过比较动作在原始上下文与结果增强上下文下的教师强制似然差异，生成轨迹归一化的回溯因子，从而为长程语言智能体提供更精确的动作级信用分配。

    

    组相对策略优化通过从 rollout 组中估计优势来避免使用单独的 critic。然而对于多轮智能体，轨迹级监督提供的是粗粒度且带有噪声的信用信号：终端奖励无法定位错误发生的具体动作，并且可能将有效的动作与错误动作一同惩罚。组中组策略优化及其后续方法通过状态条件化的比较来细化监督，但其信用估计仍然对下游决策和最终结果较为敏感。我们提出了 RELACE，这是一个无 critic 的框架，将回溯式动作评估与状态条件化的优势估计相结合。RELACE 通过在动作的原始上下文和结果增强上下文两种情形下进行教师强制似然打分，来评估已执行的动作。对这两种似然进行比较，可以得到一个经轨迹归一化的回溯因子，该因子能够捕捉动作结果依赖性的变化（摘要至此截断）。

    arXiv:2610.07349v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) avoids a separate critic by estimating advantages from rollout groups. For multi-turn agents, however, trajectory-level supervision provides coarse, noisy credit: terminal rewards do not locate errors and can penalize useful actions alongside mistakes. Group-in-Group Policy Optimization (GiGPO) and subsequent methods refine supervision through state-conditioned comparisons, but their credit estimates remain sensitive to downstream decisions and outcomes. We introduce RELACE, Retrospective Likelihood-based Action, a critic-free framework that integrates retrospective action assessment with state-conditioned advantage estimation. RELACE evaluates executed actions through teacher-forced likelihood scoring under both their original contexts and outcome-augmented contexts. Comparing these likelihoods yields a trajectory-normalized retrospective factor that captures outcome-dependent changes in actio
    
[^243]: 阶梯式MoE：具有可配置推理复杂度的分段级路由

    Stepped MoE: Segment-Level Routing with Configurable Inference Complexity

    [https://arxiv.org/abs/2610.07348](https://arxiv.org/abs/2610.07348)

    本文提出阶梯式MoE统一框架，将弹性结构与稀疏门控架构相结合，通过分段级路由使模型能够同时适应不同的部署约束和任务需求，实现推理时对精度-效率权衡的细粒度控制。

    

    训练大型语言模型（LLM）非常耗费资源，而将模型适配到具有不同计算约束的多样化部署场景仍然具有挑战性。虽然弹性架构能够实现灵活的模型部署，稀疏激活模型允许输入自适应的计算，但现有方法将这些维度独立处理。此外，面向设备端边缘推理的模型需要符合服务设备的内存和计算限制。在本文中，我们引入了一个统一框架，将弹性结构与稀疏门控架构相结合，创建能够同时适应部署约束和任务需求的模型。我们的方法采用一个以上下文和目标效率规格为条件的模型骨干网络，从而在推理时实现对精度-效率权衡的细粒度控制。模型学习激活与任务相关的部分……

    arXiv:2610.07348v1 Announce Type: cross  Abstract: Training large language models (LLMs) is resource-intensive, and adapting them for diverse deployment scenarios with varying computational constraints remains challenging. While elastic architectures enable flexible model deployment and sparsely activated models allow input-adaptive computation, existing approaches treat these dimensions independently. Moreover, models catered towards on-device edge inference need to conform to the memory and compute limitations of the serving devices. In this paper, we introduce a unified framework that combines elastic structures with sparsely gated architectures to create models that adapt simultaneously to both deployment constraints and task requirements. Our approach employs a model backbone that conditions on both the context and target efficiency specifications, enabling fine-grained control over the accuracy-efficiency trade-off at inference time. The model learns to activate task-relevant par
    
[^244]: 在云环境中评估行为上下文以实现可解释的IAM策略风险评分

    Evaluating Behavioral Context for Interpretable IAM Policy Risk Scoring in Cloud Environments

    [https://arxiv.org/abs/2610.07345](https://arxiv.org/abs/2610.07345)

    本文提出AWS IAM Context Bench基准，通过可解释提升机模型验证行为上下文信息能否在策略与有效授权信息之外，为云环境中IAM策略的风险优先级排序带来可衡量的增量价值。

    

    IAM策略分析通常侧重于策略中编码的授权能力，但安全分析师的审查优先级可能还取决于策略事件周围的行为和环境上下文。本文评估了上下文信息能否在策略和有效授权信息之外，为可解释的IAM策略风险优先级排序提供可衡量的增量价值。本研究选用AWS作为实验云提供商，因为其IAM和审计遥测生态系统使得利用AWS IAM Context Bench进行受控评估成为可能，该基准包含534条真实的AWS实验观测数据，涵盖策略、环境和行为场景，其中包括策略和环境保持固定而行为上下文发生变化的匹配案例。在相同的防泄漏分组交叉验证协议下，研究评估了三个可解释提升机模型：以策略为中心的基线模型、策略加环境模型……

    arXiv:2610.07345v1 Announce Type: cross  Abstract: IAM policy analysis typically emphasizes the authorization capabilities encoded in a policy, but security analyst review priority may also depend on the behavioral and environmental context surrounding a policy event. This paper evaluates whether contextual information provides measurable incremental value for interpretable IAM policy risk prioritization beyond policy and effective-authorization information. AWS is used as the experimental cloud provider because its IAM and audit-telemetry ecosystem enables controlled evaluation using AWS IAM Context Bench, a benchmark containing 534 real AWS experimental observations across policy, environment, and behavioral scenarios, including matched cases where policy and environment remain fixed while behavioral context changes. Three Explainable Boosting Machine models are evaluated under the same leakage-controlled grouped cross-validation protocol: a policy-centric baseline, a policy-plus-env
    
[^245]: CausalBind：面向蛋白质-分子虚拟筛选的因果建模与学习

    CausalBind: Causal Modeling and Learning for Protein-Molecule Virtual Screening

    [https://arxiv.org/abs/2610.07340](https://arxiv.org/abs/2610.07340)

    该论文提出CausalBind，通过因果建模识别并利用蛋白质-分子结合中稀疏的跨模态局部相互作用模式（如氢键、疏水接触、盐桥），从而克服传统密集整体对齐方法的局限，提升虚拟筛选向新靶点泛化的能力。

    

    蛋白质-分子虚拟筛选日益被构建为共享嵌入空间中的表示学习问题。现有方法依赖于密集的整体对齐，将不变的结合决定因素与干扰性相关性纠缠在一起，从而限制了向新靶点的迁移能力。已有研究指出，蛋白质-分子系统中的结合涉及稀疏的跨模态相互作用：结合由一个小的接触界面和少数决定性的局部相互作用（如氢键、疏水接触和盐桥）所支配，而非蛋白质和分子的全局结构。我们假设，发现并利用这些稀疏相互作用模式对于超越训练数据的泛化至关重要，因为这些模式是可复用的，并有望在不同场景下提升性能。在本文中，我们旨在识别并利用稀疏相互作用模式，并验证这一假设。由于训练数据（原文摘要在此处截断）……

    arXiv:2610.07340v1 Announce Type: cross  Abstract: Protein-molecule virtual screening is increasingly cast as a problem of representation learning in a shared embedding space. Existing methods rely on dense holistic alignment, entangling invariant binding determinants with nuisance correlations and limiting transfer to new targets. It has been noted that binding in protein-molecule systems involves sparse cross-modality interactions: binding is governed by a small contact interface and a few decisive local interactions (e.g., hydrogen bonds, hydrophobic contacts, and salt bridges) rather than the global structures of the protein and molecule. We hypothesize that uncovering and leveraging sparse interaction patterns is critical for generalization beyond the training data, as these patterns are reusable and expected to improve performance across different scenarios. In this paper, we aim to identify and leverage sparse interaction patterns, and verify our hypothesis. Since the training d
    
[^246]: Logbook：超长时音频事件理解

    Logbook: Extremely Long-form Audio Event Understanding

    [https://arxiv.org/abs/2610.07338](https://arxiv.org/abs/2610.07338)

    该论文提出了面向小时级至六天超长音频的事件理解基准 Logbook，要求系统对连续音频进行无缝隙分割并为每段生成事件标签与描述，发现最佳系统仍不及人类、过度分割普遍存在，且端到端系统通常优于级联系统但性能随上下文变长而下降。

    

    现有的音频基准测试都围绕短小、预先分割的音频片段构建，这将模型设计限制在简短输入或固定词表上。为了弥合这一差距，我们提出了 Logbook，一个面向小时级音频理解的基准测试，其录音时长从十分钟到六天不等。给定一段连续的音频录音和一个事件标签词表，系统必须预测出无缝隙的分割结果，并为每个片段提供事件标签和描述。我们比较了52个端到端和级联系统，并对微调、上下文长度和推理预算进行了消融实验。我们发现该任务是可以解决的，但表现最好的系统仍低于人类参考水平。此外，过度分割现象普遍存在，微调可以部分缓解这一问题。最后，端到端系统通常优于级联系统，但其性能会随着上下文变长而下降。

    arXiv:2610.07338v1 Announce Type: cross  Abstract: Audio benchmarks are built around short, pre-segmented clips, limiting model design to brief inputs or fixed vocabularies. To close this gap, we introduce Logbook, a benchmark for hour-scale audio understanding, with recordings ranging from ten minutes to six days. Given a continuous audio recording and an event label vocabulary, a system must predict a gap-free segmentation with an event label and a description per segment. We compare 52 systems, end-to-end and cascaded, and ablate fine-tuning, context length, and reasoning budget. We find the task tractable, though the best systems remain below the human reference. Also, over-segmentation is pervasive, and fine-tuning partially mitigates it. Finally, end-to-end are often better than cascaded systems, but degrades with longer context.
    
[^247]: 面向成本感知LLM智能体在长时程决策中的选择性批判机制

    Selective Critique for Cost-Aware LLM Agents in Long-Horizon Decision Making

    [https://arxiv.org/abs/2610.07335](https://arxiv.org/abs/2610.07335)

    提出SAG框架，利用基于动作歧义信号（全局熵与top-2边际）的轻量级免训练门控机制，智能地选择在何时调用外部批判，从而在提升LLM智能体长时程决策可靠性的同时大幅降低token消耗和延迟。

    

    提升大型语言模型（LLM）智能体在长时程决策中的可靠性仍然是一个关键挑战。当作为自主智能体部署并与复杂环境交互时，早期的错误会在轨迹中传播并引发级联式失败。近期的方法通过引入外部批判或深思机制来提高可靠性，但在每一步都调用这些机制会大幅增加token消耗和延迟，限制了实际部署。我们提出了SAG（带有门控批判的自改进智能体，Self-improving Agent with Gated critique），这是一个成本感知框架，它将批判调用表述为长时程交互过程中的逐步决策问题。SAG引入了一种轻量级的、无需训练的门控机制，通过在可行动作集合上计算的动作级歧义信号——全局熵和局部top-2边际——来估计批判的效用。从决策理论的角度来看，该机制近似……

    arXiv:2610.07335v1 Announce Type: cross  Abstract: Improving the reliability of large language model (LLM) agents in long-horizon decision-making remains a key challenge. When deployed as autonomous agents interacting with complex environments, early mistakes can propagate through trajectories and cause cascading failures. Recent approaches improve reliability by incorporating external critique or deliberation, but invoking these mechanisms at every step substantially increases token consumption and latency, limiting practical deployment. We propose SAG (Self-improving Agent with Gated critique), a cost-aware framework that formulates critique invocation as a step-wise decision problem during long-horizon interaction. SAG introduces a lightweight, training-free gating mechanism that estimates the utility of critique using action-level ambiguity signals--global entropy and local top-2 margin--computed over admissible actions. From a decision-theoretic perspective, this mechanism approxi
    
[^248]: 权重预言机：用语言模型读取神经网络权重

    Weight Oracles: Reading Neural Network Weights with Language Models

    [https://arxiv.org/abs/2610.07334](https://arxiv.org/abs/2610.07334)

    提出了“权重预言机”——通过直接读取神经网络原始权重（而非行为测试）来诊断网络属性（如后门）的微调语言模型，可从权重模拟前向传播并实现零样本安全审计。

    

    神经网络的可解释性方法大多是被动响应式的：它们分析特定前向传播过程中产生的激活值，需要已知输入才能发现诸如后门之类的隐藏能力。我们提出了权重预言机（Weight Oracles），这是一类经过微调的语言模型，能够通过直接读取目标网络的原始权重来诊断其属性，而无需进行行为测试。我们分两个阶段研究了这一范式。第一阶段验证了可行性：通过分阶段课程学习以及将无参数运算委托给确定性代码的外部计算链，一个解释型大语言模型学会了从权重模拟小型Transformer的前向传播，在未见过的目标网络上达到了99%的留存测试准确率。第二阶段将该基础设施重新用于安全审计。我们仅使用良性病理作为训练信号，在关于权重异常的自然语言诊断问题上训练了一个预言机，并对其进行零样本评估……

    arXiv:2610.07334v1 Announce Type: new  Abstract: Interpretability methods for neural networks are predominantly reactive: they analyse activations produced during specific forward passes, requiring known inputs to find hidden capabilities such as backdoors. We propose Weight Oracles, fine-tuned language models that diagnose properties of a target network by reading its raw weights directly, without behavioural testing. We investigate this paradigm in two phases. Phase I establishes feasibility: through a staged curriculum and an external chain-of-computation that delegates parameter-free operations to deterministic code, an explainer LLM learns to simulate the forward pass of small transformers from their weights, achieving 99% holdout accuracy on unseen targets. Phase II repurposes this infrastructure for safety auditing. We train an oracle on natural language diagnostic questions about weight anomalies using only benign pathologies as training signal, and evaluate it zero-shot on bac
    
[^249]: 面向智能体强化学习的MoE专家选择结构化方法

    Structuring MoE Expert Selection for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.07332](https://arxiv.org/abs/2610.07332)

    该论文发现MoE专家选择与智能体轨迹存在天然的结构对齐（语义相似操作的轮次共享更多路由专家），并提出分层路由控制框架将这一结构约束纳入智能体强化学习训练，从而同时提升任务性能与推理效率。

    

    摘要（arXiv:2610.07332v1，公告类型：cross）：长程大语言模型智能体通常采用稀疏混合专家模型来实现，然而智能体行为与MoE结构的协同设计仍未得到充分探索。在本工作中，我们全面研究了智能体后训练与MoE专家选择之间的联系。在现成的MoE模型中，我们观察到专家选择呈现出一种专门化结构，该结构天然地与智能体轨迹相契合。具体而言，当智能体在不同轮次中执行语义相似的操作（例如READ、UPDATE）时，其专家路由的重叠程度要高于执行不同操作的轮次之间。然而，标准的强化学习算法忽略了这种专门化，使得MoE路由在训练过程中处于不受控的状态，这在经验上限制了任务性能和推理效率。为解决这一问题，我们引入了一个面向智能体任务的分层路由控制框架。我们显式地鼓励轮次级别的专家选择与（摘要在此处截断）

    arXiv:2610.07332v1 Announce Type: cross  Abstract: Long-horizon LLM agents are frequently implemented using sparse mixture-of-experts (MoE) models, yet the co-design of agentic behavior and MoE structures remains underexplored. In this work, we comprehensively study the connections between agentic post-training and MoE expert selection. In off-the-shelf MoE models, we observe expert selection exhibits a specialized structure that naturally aligns with agentic trajectories. Specifically, expert routing overlaps more between turns where the agent performs semantically similar operations (e.g., READ, UPDATE) than between turns with differing operations. However, standard RL algorithms ignore this specialization, allowing the MoE routing to go uncontrolled during training, which empirically limit task performance and inference efficiency. To address this, we introduce a hierarchical routing control framework for agentic tasks. We explicitly encourage turn-level expert selections to align w
    
[^250]: 量子测量类学习中的纠缠学习规则优势

    Advantage of Entangled Learning Rules in Quantum Measurement Class Learning

    [https://arxiv.org/abs/2610.07328](https://arxiv.org/abs/2610.07328)

    该论文证明了在量子测量类PAC学习框架中，采用无法通过局域操作和经典通信（LOCC）实现的纠缠学习规则，相比传统的单副本学习规则具有优势。

    

    以量子态形式的数据进行学习是当前的研究热点，并由此引出了多种问题，这些问题最终可归结为通过量子测量与可用数据进行交互，并对观测到的经典结果进行经典后处理。在量子测量PAC学习中，我们给定一序列未知的、已制备的量子态及其经典标签，以及一个由候选测量组成的假设类。任务是从该假设类中选出一个测量，使得利用所选假设对新量子态进行测量来预测经典标签时，在某种固定误差度量下达到最小。在本工作中，我们考虑在测量学习框架下，使用由无法通过局域操作和经典通信（LOCC）实现的测量所定义的学习规则（相对于单副本学习规则）与给定数据进行交互所能带来的优势。我们提供了一个构造，表明……

    arXiv:2610.07328v1 Announce Type: cross  Abstract: Learning with data in the form of quantum states is of current interest and has led to a variety of problems that boil down to interaction with the available data via quantum measurement and classical post-processing of observed classical outcomes. In quantum measurement PAC learning, one is given a sequence of unknown, prepared quantum states and classical labels, along with a hypothesis class of candidate measurements. The task is to select a measurement from the hypothesis class that minimizes a fixed notion of error in prediction of the classical labels via measurement of a new state by the selected hypothesis. In this work, we consider the advantage of interacting with the given data in the measurement learning framework using learning rules given by measurements that cannot be implemented using local operations and classical communication (LOCC), as opposed to single-copy learning rules. We provide a construction showing that the
    
[^251]: 面向时间序列基础模型的尺度不变训练

    Scale-Invariant Training for Time Series Foundation Models

    [https://arxiv.org/abs/2610.07324](https://arxiv.org/abs/2610.07324)

    论文揭示了对仿射缩放（如ReVIN）进行逆变换会使各序列梯度被乘以b^p、使序列尺度成为隐含的重要性权重并导致高尺度序列主导训练的“尺度污染”问题，并证明直接在缩放后的目标上计算损失即可实现尺度不变的训练。

    

    时间序列基础模型是在跨越多种形态和领域的大量时间序列数据集集合上训练的。这种训练设置使模型接触到尺度（即数值的典型大小）差异可能非常显著的序列。诸如可逆实例归一化（ReVIN）等仿射缩放方法会对模型输入进行缩放，并在计算损失之前逆转该变换。我们证明，相对于在缩放后的目标上计算损失，这种逆变换会将每个序列的梯度乘以 $b^p$，其中 $b$ 是缩放分母（例如标准差），$p$ 是损失次数。我们将这种现象称为尺度污染训练，因为每个序列的尺度由此变成了重要性权重，导致高尺度序列主导训练过程。对于任何尺度等变的缩放器以及任何 $p$ 次齐次的残差损失（包括 MSE、MAE 和分位数损失），我们证明在缩放后的目标上计算损失

    arXiv:2610.07324v1 Announce Type: cross  Abstract: Time series foundation models (TSFMs) are trained on large collections of time series datasets that span various morphologies and domains. This setting exposes models to series whose scales -- typical magnitudes of their values -- can differ substantially. Affine scaling methods such as Reversible Instance Normalization (ReVIN) scale model inputs and reverse the transform before computing the loss. We show that this inversion multiplies each series' gradient by $b^p$ relative to loss on scaled targets, where $b$ is the scaling denominator (e.g., standard deviation) and $p$ is the loss degree. We call this scale-contaminated training (ScaleCon), because the scale of each series consequently becomes an importance weight, causing high-scale series to dominate training. For any scale-equivariant scaler and residual loss that is homogeneous of degree $p$, including MSE, MAE, and Quantile Loss, we prove that computing loss on scaled targets 
    
[^252]: ATLAS-AL：基于主动学习的潜在对抗搜索自适应信任域方法

    ATLAS-AL: Adaptive Trust-Region for Latent Adversarial Searches via Active Learning

    [https://arxiv.org/abs/2610.07323](https://arxiv.org/abs/2610.07323)

    ATLAS提出了一种基于主动学习的查询式攻击生成框架，通过将对抗攻击发现转化为水平集估计问题，结合自适应信任域与局部-全局采样策略，高效挖掘黑盒模型的对抗性输入集合，从而更全面地评估模型鲁棒性。

    

    基于学习的系统的安全评估不仅仅是针对固定的攻击集合来测试系统，还需要能够高效发现导致模型失败的输入集合的自适应机制。我们提出了ATLAS（潜在对抗搜索的自适应信任域），这是一个基于查询的框架，用于发现黑盒学习系统的对抗性输入集合。ATLAS将攻击生成转化为一个主动学习的水平集估计问题，然后将校准的近似方法与局部-全局采样架构相结合，以定位输入空间中包含对抗样本的区域。一旦发现这些区域，ATLAS会在这些对抗区域内进行采样，以构建能够准确刻画目标模型鲁棒性状态的对抗集合。在玩具实验上的应用表明，ATLAS能够在有限的（查询预算下）恢复更多的对抗区域。

    arXiv:2610.07323v1 Announce Type: new  Abstract: Security evaluation of learning-based systems requires more than just testing the system against a fixed collection of attacks. It requires adaptive mechanisms that can efficiently discover \textit{sets} of inputs that induce model failure. We introduce ATLAS (Adaptive Trust-Regions for Latent Adversarial Searches), which is a query-based framework that discovers adversarial input sets for black-box learning systems. ATLAS casts attack generation as an active learning level set estimation problem then combines calibrated approximations with a local-global sampling architecture to find regions of the input space that contain adversarial examples. Once discovered, ATLAS is designed to sample points within these adversarial regions to build adversarial sets that accurately represent the state of robustness of the target model. When applied on toy experiments, we find that ATLAS is able to recover more of the adversarial region under a limit
    
[^253]: 协变量缺失情形下的少假设逻辑回归

    Assumption-lean logistic regression with missing covariates

    [https://arxiv.org/abs/2610.07292](https://arxiv.org/abs/2610.07292)

    该论文提出了一种在协变量分布未知（仅有界）的少假设设定下处理协变量缺失逻辑回归参数估计的随机近似方法，克服了传统方法在协变量分布未知时可能严重失效的问题。

    

    在监督学习问题中经常遇到协变量缺失的情况，使用此类数据进行估计的经典方法依赖于精心设计的缺失数据插补方案，或采用会导致非凸 M-估计问题的似然近似。这些方法及其相关变体适用于协变量分布已知的场景，更广泛地说，它们在线性模型中取得了巨大成功。但即使是在中等维度的逻辑回归这类基本的非线性问题中，当协变量分布未知时，这些方法也可能出现严重的失效模式。出于对可靠替代方法的需求，我们研究了协变量缺失情形下逻辑回归的参数估计问题。关键在于，我们在少假设的设定下开展工作，即协变量分布未知（但有界）。我们设计了一种基于 Z-的随机近似方法（摘要在此处截断）

    arXiv:2610.07292v1 Announce Type: cross  Abstract: Missing covariates are frequently encountered in supervised learning problems, and classical methods for estimation using such data use carefully chosen imputation schemes for missing data, or likelihood approximations that lead to nonconvex $M$-estimation problems. These methods and their relatives are suitable for scenarios in which the covariate distribution is known, and more broadly, have enjoyed tremendous success in linear models. But even in basic nonlinear problems such as logistic regression in moderate dimensions, such methods can experience drastic failure modes when the covariate distribution is unknown.   Motivated by the need for reliable alternatives, we consider the problem of parameter estimation in logistic regression with missing covariates. Crucially, we operate in the assumption-lean setting where the covariate distribution is unknown (but bounded). We design a stochastic approximation method that is based on $Z$-
    
[^254]: 一种用于随机双层优化的单循环、常数批次一阶惩罚方法

    A Single-Loop, Constant-Batch First-Order Penalty Method for Stochastic Bilevel Optimization

    [https://arxiv.org/abs/2610.07290](https://arxiv.org/abs/2610.07290)

    提出SICO方法，一种单循环、常数批次的一阶惩罚方法，通过每次迭代对原始下层问题和惩罚问题各执行一次随机梯度更新并结合投影控制，从而在随机非凸-强凸双层优化中摆脱了对嵌套循环和大批次规模的需求。

    

    近年来，基于惩罚方法的随机双层优化（SBO）的最新进展已经消除了对二阶导数预言机的需求。然而，对于随机非凸-强凸双层问题，现有的一阶方法通常依赖嵌套循环和/或较大的批次规模，才能在标准有界方差假设或均方光滑性假设下达到 O(ε^{-6}) 或 O(ε^{-4}) 的样本复杂度。由于精确近似需要较大的惩罚值，使用单循环惩罚方法和常数批次规模来实现这些速率仍然具有挑战性。为了应对这一挑战，我们开发了一种随机单循环常数批次一阶惩罚方法（SICO），它结合了两个互补的要素。首先，它对原始下层问题和惩罚问题每次迭代各执行一次随机梯度更新，并通过投影来控制分离……

    arXiv:2610.07290v1 Announce Type: cross  Abstract: Recent advances in penalty-based methods for stochastic bilevel optimization (SBO) have eliminated the need for second-order derivative oracles. However, for stochastic nonconvex-strongly convex bilevel problems, existing first-order methods typically rely on nested loops and/or large batch sizes for attaining $O(\epsilon^{-6})$ or $O(\epsilon^{-4})$ sample complexity under standard bounded-variance assumption or mean-square smoothness assumption. Achieving these rates with a single-loop penalty method and a constant batch size remains challenging due to a large penalty value needed for an accurate approximation. To address this challenge, we develop a stochastic SIngle-loop COnstant-Batch first-order penalty method (SICO) that combines two complementary ingredients. First, it performs one stochastic-gradient update per-iteration for both the original lower-level and penalized problems, with a projection that controls the separation be
    
[^255]: FlexiFlow：基于多臂老虎机的机器学习工作流模型切换

    FlexiFlow: Bandit-based Model Switching in ML Workflows

    [https://arxiv.org/abs/2610.07286](https://arxiv.org/abs/2610.07286)

    FlexiFlow是一个基于多臂老虎机的动态模型切换数据流系统，综合考虑模型准确率、运行时间和断言通过概率，在当前模型表现不佳时自动切换到更优模型，可将机器学习工作流准确性提升高达23%。

    

    模型优化有助于提高机器学习工作流的推理性能和准确性。然而，依赖单一模型对所有数据批次执行推理往往无法最大化准确性，从而无法最大化整体性能。在许多情况下，当主模型表现不佳时，替代模型在特定数据子集上可能表现更好。我们在真实机器学习工作流上的实验表明，切换模型可将工作流准确性提升高达23%。然而，现有系统缺乏根据性能自适应切换模型的能力，迫使用户手动依次测试模型。我们提出了FlexiFlow，这是一个数据流系统，能够在当前模型表现出低准确性时动态切换到替代模型。FlexiFlow采用一种新颖的多臂老虎机方法学习对模型进行排序，该方法综合考虑了模型运行时间、通过用户定义断言的概率以及机器学习工作流的计算结构。

    arXiv:2610.07286v1 Announce Type: cross  Abstract: Model optimizations help improve inference performance and accuracy of ML workflows. However, relying on a single model to perform inference across all data batches often fails to maximize accuracy and thus overall performance. In many cases, alternate models could perform better on specific subsets of data where a primary model underperforms. Our experiments with real ML workflows indeed show that switching models improves workflow accuracy by up to 23%. Yet, current systems lack the ability to adaptively switch between models based on performance, forcing users to manually test models in sequence. We present FlexiFlow, a dataflow system that dynamically switches between alternate models when the current model exhibits low accuracy. FlexiFlow learns to rank models using a novel multi-armed bandit approach that accounts for model runtimes, probability of passing user-defined assertions, and the computational structure of the ML workflo
    
[^256]: 锁相EP：一种面向振荡硬件的原位训练算法

    Lock-in EP: An In-Situ Training Algorithm for Oscillatory Hardware

    [https://arxiv.org/abs/2610.07283](https://arxiv.org/abs/2610.07283)

    提出锁相平衡传播（LIEP）训练算法，无需独立的前向和后向扫描即可为振荡网络组件提供局部梯度信息，使模拟振荡硬件能够进行原位训练和性能恢复。

    

    模拟硬件平台相较于数字架构具有降低能耗的潜力，但要想取得成功，大规模模拟系统还必须能够在其组件存在变异性的情况下正常工作或从中恢复。为实现这一目标，我们推导并演示了锁相平衡传播（LIEP）训练方法。LIEP能够在无需单独的前向和后向扫描的情况下，为振荡网络中的每个组件提供局部梯度信息，从而有望在模拟振荡硬件平台上实现原位学习能力。我们证明LIEP既可用于从头训练，也可用于在预训练参数受到扰动时恢复性能。我们表明LIEP可以表述为一种三因子更新规则，并提出尽管该方法目前仅在浅层网络上得到验证，但其他架构可能使其能够扩展至深层和大规模网络。

    arXiv:2610.07283v1 Announce Type: new  Abstract: Analog hardware platforms offer the potential to reduce energy consumption over digital architectures, but in order to succeed, large-scale analog systems must also be able to operate with or recover from the variability of their components. Towards this goal, we derive and demonstrate the lock-in equilibrium propagation (LIEP) training method. LIEP provides local gradient information for each component in an oscillatory network without separate forward and backward sweeps, potentially allowing for in-situ learning capabilities on analog oscillatory hardware platforms. We demonstrate that LIEP can be used both for ab-initio training as well as recovering performance when pre-trained parameters are perturbed. We show that LIEP can be formulated as a three-factor update rule, and suggest that although the method is currently only validated on shallow networks, alternate architectures may allow it to extend to deep and large-scale networks 
    
[^257]: 算法对齐的神经凝聚树构建

    Algorithmically Aligned Neural Agglomerative Tree Construction

    [https://arxiv.org/abs/2610.07271](https://arxiv.org/abs/2610.07271)

    提出NN-linkage模型，通过将神经网络与Lance-Williams递推公式算法对齐，使其能够学习任务特定的层次聚类合并规则，同时保留经典链接算法的高效推理和规模泛化能力。

    

    层次聚类（HC）的链接算法是构建聚类树的一个强大且高效的框架，然而对于给定的数据集或任务，往往不清楚哪种合并规则最为合适。相比之下，神经网络方法可以从数据中学习，但通常无法保留经典算法的效率和规模泛化能力。我们提出了NN-linkage，这是一种神经网络（NN）模型，能够学习任务特定的、局部依赖的合并规则，同时保留经典链接算法的递归结构和高效推理。特别地，我们的模型与Lance-Williams（LW）递推公式在算法上对齐，LW递推是一个参数化框架，用于为凝聚层次聚类定义广泛且连续的链接规则族。经典方法如单链接（SL）、完全链接（CL）和平均链接都是这一更广泛规则族中的离散选择。我们证明NN-linkage是一个通用逼近器。

    arXiv:2610.07271v1 Announce Type: new  Abstract: Linkage algorithms for hierarchical clustering (HC) are a powerful and efficient framework for constructing clustering trees, yet it is often unclear which merge rule best suits a given dataset or task. In contrast, neural approaches can learn from data, but often fail to retain the efficiency and size generalization of classical algorithms. We introduce NN-linkage, a neural network (NN) model that can learn task-specific and locally dependent merge rules while retaining the recursive structure and efficient inference of classical linkage algorithms. In particular, our model is algorithmically aligned with the Lance-Williams (LW) recurrence, a parameterized framework for defining a broad, continuous family of linkage rules for agglomerative HC. Classical methods such as single linkage (SL), complete linkage (CL), and average linkage arise as discrete choices within this broader family. We show that NN-linkage is a universal approximator 
    
[^258]: 文字所留存的地方：面向跨视角地理定位的零样本语言推理

    What Words Keep of a Place: Zero-Shot Language Reasoning for Cross-View Geo-Localization

    [https://arxiv.org/abs/2610.07269](https://arxiv.org/abs/2610.07269)

    该论文提出一种完全无需训练的零样本方法，利用多模态大语言模型将地面全景图与卫星图块转化为结构化文本描述并通过语言比较实现跨视角地理定位，发现语言描述虽然忠实但缺乏区分能力。

    

    跨视角地理定位通常被作为一个图像检索问题来解决，即通过联合训练的嵌入表示，将地面图像与卫星图块数据库进行匹配。这类模型精度较高，但需要大规模的成对监督数据，且无法展示支撑匹配结果的证据。在本文中，我们研究了一个不同的问题：仅凭语言能在多大程度上解决这一任务？我们提示多模态大语言模型（MLLM）将每张地面全景图和每个卫星图块描述为结构化文本，并通过比较这些描述来实现定位，整个流程无需训练任何组件。我们在来自四个美国城市的9,826个VIGOR配对数据上、在三种设置下进行评估。首先，这些描述是忠实的但缺乏区分度：两种视角下的描述高度一致，但仅通过描述相似度对完整候选池进行排序时，几乎从未返回正确的图块（Recall@1仅为0.39%）。其次，我们将候选池缩小到十个相邻图块，作为一种粗……

    arXiv:2610.07269v1 Announce Type: cross  Abstract: Cross-view geo-localization is commonly solved as an image retrieval problem, matching a ground-level image against a database of satellite tiles through a jointly trained embedding. Such models are accurate, but they need large paired supervision and cannot show what evidence supports a match. In this paper, we study a different question: how much of this task can be solved through language alone? We prompt a multimodal large language model (MLLM) to describe each ground panorama and each satellite tile as structured text, and localize by comparing these descriptions. No component is trained. We evaluate on 9,826 VIGOR pairs from four U.S. cities, in three settings. First, the descriptions are faithful but not discriminative. They agree closely across the two views, yet ranking the full pool by description similarity almost never returns the correct tile (0.39% Recall@1). Second, we narrow the pool to ten neighboring tiles, as a coars
    
[^259]: 谱系感知的内存治理：面向企业AI智能体隐私保护列级访问控制的派生门控框架

    Lineage-Aware Memory Governance: A Derivation-Gated Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI Agents

    [https://arxiv.org/abs/2610.07258](https://arxiv.org/abs/2610.07258)

    该论文提出分析内存单元（AMU），通过为每个缓存结果附加完整的派生谱系图并实施列级权限门控检索，从设计上保证企业AI智能体不会命中由请求者无权限的敏感列派生而来的缓存结果，同时解决部门间同名KPI计算逻辑冲突的问题。

    

    共享内存存储的企业AI智能体面临两个尚未解决的风险：敏感数据可能通过请求者本无权限推导出的“合法计算结果”发生泄露，以及各部门可能通过相互冲突的逻辑静默地计算同名关键绩效指标（KPI）。现有的智能体内存系统（如MemGPT、Zep、A-MEM）基于内容、所有权和角色来控制检索，而非基于派生关系，因而无法拦截缓存中嵌入了被禁止访问列的洞察结果。我们提出了分析内存单元（AMU），这是一种内存模式，为每个缓存结果附加完整的派生（谱系）图，并由检索策略进行门控——仅当请求者对结果所涉及的每一列均具有访问权限时，才返回缓存命中。在谱系记录完整的前提下，我们通过构造性证明表明，该策略能够阻止检索由请求者权限之外的敏感列所派生的结果，最坏情况复杂度为O(n)——这是一种条件性的设计保证，而非经验性的……

    arXiv:2610.07258v1 Announce Type: cross  Abstract: Enterprise AI agents that share a memory store face two unaddressed risks: sensitive data can leak through legitimately computed results the requester could not derive, and departments can silently compute a same-named key performance indicator (KPI) through conflicting logic. Existing agent-memory systems (e.g., MemGPT, Zep, A-MEM) gate retrieval by content, ownership, and role, not derivation, missing a cached insight that embeds a forbidden column. We introduce the Analytical Memory Unit (AMU), a memory schema that attaches a full derivation (lineage) graph to every cached result, gated by a retrieval policy that serves a hit only when the requester is authorised for every column touched. Provided lineage recording is complete, we prove by construction that the policy blocks retrieval of results derived from a sensitive column outside the requester's permissions, at O(n) worst case -- a conditional design guarantee, not an empirical
    
[^260]: 面向图鞍点问题的神经算法推理

    Neural Algorithmic Reasoning for Graph Saddle Point Problems

    [https://arxiv.org/abs/2610.07255](https://arxiv.org/abs/2610.07255)

    该论文提出基于 Chambolle-Pock PDHG 方法的消息传递框架 GraphPDHG，将神经算法推理用于求解图鞍点问题，既能模拟 PDHG 高效求解并学习到加速算法，还展现出比未对齐 GNN 基线更强的规模泛化能力。

    

    神经算法推理，即将神经网络与算法范式对齐，已成为解决多项式时间可解以及计算上更困难的组合优化问题的一种方法。我们提出了一种基于 Chambolle-Pock 原始-对偶混合梯度（PDHG）方法的新消息传递框架，称为 GraphPDHG，用于求解一般图鞍点问题。在理论上，我们证明 GraphPDHG 能够通过模拟 PDHG 高效求解一类图鞍点问题，并且我们的网络还可以学习到加速的 PDHG 算法。在实验上，我们通过将模型评估为二阶优化技术（SSNAL）的学习式热启动方案，支持了关于加速 PDHG 的结果。我们还证明了与 PDHG 对齐能带来比未对齐的图神经网络（GNN）基线更强的规模泛化能力。总体而言，我们提出了一种新颖的架构……（原文摘要在此处截断）

    arXiv:2610.07255v1 Announce Type: new  Abstract: Neural algorithmic reasoning, or aligning a neural network with an algorithmic paradigm, has emerged as an approach to solving polynomial-time-solvable and computationally harder combinatorial optimization problems. We propose a new message-passing framework based on the Chambolle-Pock Primal--Dual Hybrid Gradient (PDHG) method called \textsc{GraphPDHG} for solving general graph saddle-point problems. Theoretically, we show that \textsc{GraphPDHG} can efficiently solve a family of graph saddle-point problems by simulating PDHG. We also show that our network can learn an accelerated PDHG algorithm. Experimentally, we support our results on accelerated PDHG by evaluating the performance of our model as a learned warm start for second-order optimization techniques (SSNAL). We also show that alignment with PDHG leads to stronger size generalization than non-aligned graph neural network (GNN) baselines. Overall, we propose a novel architectur
    
[^261]: 神经场编码适应几何

    Neural Fields Encode Adaptation Geometry

    [https://arxiv.org/abs/2610.07253](https://arxiv.org/abs/2610.07253)

    论文提出“适应几何”概念，证明拟合后的神经场权重不仅由重构质量决定，还编码了其适应新观测的难易程度（可由局部线性模型精确预测）以及对历史观测信息的保留。

    

    神经场通常以对观测的重构质量来评价。我们表明，这种评价方式忽略了拟合网络的两个有用特性：它适应新观测的难易程度，以及其权重从先前观测中保留了什么。我们将这些特性作为“适应几何”来研究。对于图像，我们元学习类别特定的初始化，将每个初始化适应到新图像上，并测量网络为拟合该图像所需改变的程度。一个简单的局部线性模型能够紧密预测这种适应代价，而将一个网络的切核替换为另一个网络的切核则会显著恶化预测。因此，适应取决于拟合网络的局部几何，而不仅仅取决于其当前的重构。对于物理场，我们反复将同一个网络拟合到来自某一序列的观测，其权重随后会保留关于该历史的信息。当两个波动历史终止于完全相同的观测时，最终权重能够恢复出符号……

    arXiv:2610.07253v1 Announce Type: new  Abstract: Neural fields are usually evaluated by how well they reconstruct an observation. We show that this misses two useful properties of a fitted network: how easily it can adapt to new observations, and what its weights retain from earlier ones. We study these properties as adaptation geometry. For images, we meta-learn class-specific initializations, adapt each one to a new image, and measure how much the network must change to fit it. A simple local linear model closely predicts this adaptation cost, while replacing one network's tangent kernel with another's substantially worsens the prediction. Adaptation thus depends on the local geometry of the fitted network, not only on its current reconstruction. For physical fields, we repeatedly fit the same network to observations from a sequence. Its weights then retain information about that history. When two wave histories end at exactly the same observation, the final weights recover the sign 
    
[^262]: 学习蒸馏什么：面向大语言模型自蒸馏的双层Top-K词元选择方法

    Learning What to Distill: Bilevel Top-K Token Selection for Self-Distillation in Large Language Models

    [https://arxiv.org/abs/2610.07247](https://arxiv.org/abs/2610.07247)

    提出了BiToK-SD方法，利用双层优化自适应地学习在大语言模型在策略自蒸馏中应选择哪些词元进行蒸馏，从而克服了均匀蒸馏和固定启发式选择准则的局限。

    

    大语言模型已展现出强大的推理能力，但其高昂的推理成本使得知识蒸馏成为在资源受限场景中将这些能力迁移到紧凑模型的重要途径。在策略自蒸馏在提升紧凑语言模型推理能力的同时，进一步降低了对大型外部教师模型的依赖。然而，现有方法通常要么对所有词元位置进行均匀蒸馏，要么采用固定的启发式准则来选择词元，对所选位置赋予相同的蒸馏强度，而非自适应地学习哪些词元对蒸馏最为有益。为解决这些局限，我们提出了BiToK-SD（Bilevel Top-K Token Selection for Self-Distillation，面向自蒸馏的双层Top-K词元选择），这是一种基于双层优化的词元选择方法，能够在在策略自蒸馏过程中学习蒸馏应应用于何处。具体而言，BiToK-SD被形式化为（摘要内容在此处截断）

    arXiv:2610.07247v1 Announce Type: new  Abstract: Large language models have shown strong reasoning capabilities, but their high inference costs make knowledge distillation an important approach for transferring such capabilities to compact models in resource-constrained scenarios. On-policy self-distillation further reduces the reliance on external large teacher models while improving the reasoning ability of compact language models. However, existing methods typically either distill all token positions uniformly or select tokens using fixed heuristic criteria, assigning the same distillation strength to the selected positions rather than adaptively learning which tokens are most beneficial for distillation. To address these limitations, we propose BiToK-SD (Bilevel Top-K Token Selection for Self-Distillation), a bilevel-optimization-based token selection method that learns where distillation should be applied during on-policy self-distillation. Specifically, BiToK-SD is formulated as 
    
[^263]: 面向低资源临床环境下早期乳腺癌检测的混合跨模态注意力网络

    Hybrid Cross-Modal Attention Network for Early Breast Cancer Detection in Low-Resource Clinical Settings

    [https://arxiv.org/abs/2610.07243](https://arxiv.org/abs/2610.07243)

    提出混合跨模态注意力网络HCMAN，基于Transformer跨模态注意力机制融合乳腺X光影像与结构化临床数据，在埃塞俄比亚四家医院的本地数据集上实现了97.8%准确率的早期乳腺癌检测，适用于低资源医疗环境。

    

    乳腺癌是撒哈拉以南非洲地区女性因癌症死亡的首要原因，由于放射学专业知识匮乏和临床数据系统碎片化，导致诊断延误。尽管深度学习模型在乳腺X光摄影分析中已展现出强大性能，但大多数模型仅依赖影像数据，且在西方人群数据上训练，限制了其在非洲医疗环境中的适用性。本文提出了一种混合跨模态注意力网络（HCMAN），利用基于Transformer的跨模态注意力机制，将乳腺X光图像与结构化临床数据进行整合。该模型的开发与验证基于本地收集的数据集，包含来自埃塞俄比亚四家转诊医院1,024名患者的2,560张乳腺X光图像，并带有经活检确认的真实标签。所提出的框架达到了97.8%的准确率、97.2%的敏感度、98.3%的特异度以及0.987的AUC，显著优于……

    arXiv:2610.07243v1 Announce Type: cross  Abstract: Breast cancer is the leading cause of cancer-related mortality among women in Sub-Saharan Africa, where delayed diagnosis results from limited radiology expertise and fragmented clinical data systems. Although deep learning models have demonstrated strong performance in mammographic analysis, most rely solely on imaging data and are trained on Western populations, limiting their applicability in African healthcare settings. This paper presents a Hybrid Cross-Modal Attention Network (HCMAN) that integrates mammogram images with structured clinical data using transformer-based cross-modal attention mechanisms. The model was developed and validated using a locally collected dataset of 2,560 mammogram images from 1,024 patients across four Ethiopian referral hospitals, with biopsy-confirmed ground truth labels. The proposed framework achieves 97.8% accuracy, 97.2% sensitivity, 98.3% specificity, and an AUC of 0.987, significantly outperfor
    
[^264]: 协变量不确定性下用于负荷预测的时间序列基础模型基准测试

    Benchmarking Time Series Foundation Models for Load Forecasting Under Covariate Uncertainty

    [https://arxiv.org/abs/2610.07232](https://arxiv.org/abs/2610.07232)

    该论文在三个真实负荷预测数据集上，针对未来协变量信息可用性与质量不同的多种运行场景，对四个从零训练模型和四个时间序列基础模型进行了系统基准测试，发现Chronos-2在协变量可用或预测准确时性能最优，而TimesNet在协变量预测噪声严重时更加鲁棒。

    

    准确的短期负荷预测（STLF）对于现代电力系统的可靠与高效运行至关重要。尽管时间序列基础模型（TSFMs）最近在广泛的预测任务中展现出了卓越的性能，但其在实际运行条件下对短期负荷预测的有效性在很大程度上仍未被探索。在本文中，我们在三个真实世界的负荷预测数据集上，对四个从零开始训练（TFS）的模型和四个时间序列基础模型进行了全面的基准测试，测试所涵盖的运行场景在未来协变量信息的可用性和质量上各不相同。我们的结果表明，当未来协变量可用或能够被准确预测时，Chronos-2在零样本和微调两种设置下都能持续取得最先进的性能。然而，随着协变量预测的噪声不断增加，其性能会随之下降，而TimesNet在严重的协变量（不确定性）下则表现出更强的鲁棒性。

    arXiv:2610.07232v1 Announce Type: new  Abstract: Accurate short-term load forecasting (STLF) is essential for the reliable and efficient operation of modern power systems. While time series foundation models (TSFMs) have recently demonstrated remarkable performance across a wide range of forecasting tasks, their effectiveness for STLF under realistic operational conditions remains largely unexplored. In this paper, we present a comprehensive benchmark of four trained-from-scratch (TFS) models and four TSFMs across three real-world load forecasting datasets under operational scenarios that differ in the availability and quality of future covariate information. Our results show that Chronos-2 consistently achieves state-of-the-art performance in both zero-shot and fine-tuned settings when future covariates are available or accurately forecast. However, its performance degrades as covariate forecasts become increasingly noisy, whereas TimesNet exhibits greater robustness under severe cova
    
[^265]: 马尔可夫过程之间传输的条件流匹配方法

    Conditional Flow Matching for Transport Between Markov Processes

    [https://arxiv.org/abs/2610.07229](https://arxiv.org/abs/2610.07229)

    本文提出一种保持马尔可夫结构的条件流匹配算法，用于学习马尔可夫过程从源轨迹分布到目标轨迹分布的传输映射，并证明了总体一致性、有限样本误差界以及与混合时间相关的样本复杂度下界。

    

    受时间序列领域自适应中序列到序列传输问题的启发，我们研究了马尔可夫过程轨迹之间的传输问题。在仅有来自源分布和目标分布的有限数量轨迹的情况下，我们提出了一种基于流匹配的算法，该算法学习从源轨迹分布到目标轨迹分布的传输映射，同时保持马尔可夫结构。我们证明了该算法在总体极限下是一致的，并在混合时间假设下推导了有限样本误差界，其分析遵循马尔可夫设定下经典统计问题的分析框架，包括回归（Nagaraj等，2020）、主成分分析（Kumar和Sarkar，2023）以及矩阵集中性（Neeman等，2024）。此外，我们还给出了一个下界构造，表明即使条件转移具有规则的高斯形式，依赖于混合时间的样本复杂度也是不可避免的。

    arXiv:2610.07229v1 Announce Type: new  Abstract: Motivated by sequence-to-sequence transport in the context time-series domain adaptation, we study the problem of transportation between trajectories of Markov processes. Given a limited number of trajectories from source distribution and the target distribution, we formulate a flow matching based algorithm which learns a transport map from the source to target trajectory distribution, while preserving the Markov structure. We show that this is consistent in the population limit and derive finite-sample error bounds under mixing time assumptions, following the analysis of classical statistical problems including regression (Nagaraj et al., 2020), principal component analysis (Kumar and Sarkar, 2023), and matrix concentration (Neeman et al., 2024) in the Markov setting. We complement that with a lower-bound construction showing that a mixing-time dependent sample complexity is unavoidable even with regular Gaussian conditional transitions
    
[^266]: 自然梯度下降有多低效？从精确最优到 Θ(√log d) 的偏差

    How Inefficient Is Natural Gradient Descent? From Exact Optimality to \Theta ( \sqrt{ \log d } ) Divergence

    [https://arxiv.org/abs/2610.07228](https://arxiv.org/abs/2610.07228)

    本文提出“低效比”来量化自然梯度下降偏离最短 Fisher–Rao 路径的程度，并证明该比值在三类情形中变化：二次势或一维族精确最优（R=1）、有界偏度族具有与维度无关的界、而尺度族乘积（如高斯协方差和 Gamma 率）的低效比随维度以 Θ(√log d) 增长。

    

    自然梯度下降（NGD）是机器学习中众多常用方法的基础。对于对偶平坦（dually flat）分布族，在正向 Kullback–Leibler 目标上的理想化 NGD 沿混合测地线行进，而该测地线往往比最短的 Fisher–Rao 路径更长。我们用“低效比” \(R \ge 1\) 来量化这一额外开销，即混合测地线的 Fisher 长度与 Fisher–Rao 距离之比，并给出该比值在所有端点对上的上确界随参数维度 \(d\) 变化的界。我们提出一个张量判据来刻画情形（I）的分布族，其处处满足 \(R=1\)：恰好是具有二次势函数或维度为一的族，例如固定协方差的高斯分布。对于非二次族，我们证明了另外两种情形：（II）有界的三阶偏度加上有限的 Fisher–Rao 直径可得到与维度无关的界；（III）对于尺度族的乘积——包括高斯协方差和 Gamma 率——\(R\) 以 \(\Theta(\sqrt{\log d})\) 增长，即随维度无界增长。

    arXiv:2610.07228v1 Announce Type: cross  Abstract: Natural gradient descent (NGD) underlies common methods in ML. For dually flat families, idealized NGD on the forward Kullback--Leibler objective follows the mixture geodesic which is often longer than the shortest Fisher--Rao path. We quantify this overhead by the inefficiency ratio \(R \ge 1\), the Fisher length of the mixture geodesic divided by the Fisher--Rao distance, and bound its supremum over endpoint pairs as a function of the parameter dimension \(d\). A tensor criterion identifies the regime (I) families, with \(R=1\) everywhere: exactly those with quadratic potential or dimension one, such as fixed-covariance Gaussians. For non-quadratic families, we prove two further regimes: (II) bounded third-order skewness plus finite Fisher--Rao diameter yields a dimension-independent bound; and (III) for products of scale families---including Gaussian covariances and Gamma rates---\(R\) grows as \(\Theta(\sqrt{\log d})\), unbounded i
    
[^267]: 最小见证强化学习

    Minimal Witness Reinforcement Learning

    [https://arxiv.org/abs/2610.07226](https://arxiv.org/abs/2610.07226)

    本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。

    

    “足以产生某一结果的不可约条件是什么？”这是计算与科学领域中最常出现的核心问题之一。其答案——最小充分见证——正是我们所称的解释、机制与原因。这类问题通常需要找出多个最小见证，然而标准的强化学习方法可能只能揭示一个解或冗余的解。我们将该问题形式化为最小见证识别，并提出了最小见证强化学习（MWRL）。MWRL 对从策略中采样得到的、经成功验证的提议所认证的集合取并集，并根据“若缺少该提议，组并集将会损失的覆盖范围”为每个提议分配信用。这一直接源于问题定义的信用分配机制，仅凭一个黑盒验证器的单个比特信号，便统一了对最小性与备选解恢复的双重需求。基于该原理，我们推导出一个值迭代规划器，能够恢复……（摘要原文在此处截断）

    arXiv:2610.07226v1 Announce Type: cross  Abstract: ``What are the irreducible conditions that are sufficient to produce an outcome?'' is one of the most common questions that recur across computation and science. Its answers, the minimal sufficient witnesses, are what we mean by explanations, mechanisms and reasons. These problems usually ask for multiple minimal witnesses, yet standard RL methods may reveal only one solution or redundant ones. We formalize this problem as minimal-witness identification and introduce Minimal-Witness Reinforcement Learning (MWRL). MWRL takes the union of the sets certified by successful proposals sampled from the policy and credits each proposal for the coverage the group union would lose without that proposal. This credit assignment, derived directly from the problem definition, unifies the demands for minimality and recovery of alternatives from a single black-box verifier bit. Under this principle, we derive a value iteration planner that recovers th
    
[^268]: 数据、数值与几何：关于数值方法、机器学习与评估的三个教程

    Data, Numbers, and Geometry: Three Tutorials on Numerical Methods, Machine Learning, and Evaluation

    [https://arxiv.org/abs/2610.07220](https://arxiv.org/abs/2610.07220)

    本文为数学研究提供了三个实用教程，分别涵盖基于微分形式逐点取值的外微分数值方法、利用数学结构（如椭圆曲线与箭图）指导神经网络设计并用区间算术验证残差界，以及计算结果的评估与展示。

    

    我们为数学研究提供了三个关于数值计算与机器学习的实用教程，这些教程是为2026年4月在班夫国际研究站举办的“DANGER：数据、数值与几何”研讨会而开发的。第一个教程从微分形式的逐点取值出发，利用外微分的通量表述，发展了一种外微分演算的数值方法。欧几里得空间和球面上的示例展示了几何恒等式、拓扑特征，以及近似和有限精度带来的影响。第二个教程通过椭圆曲线、箭图和一个边值问题的例子，研究数学结构如何指导神经网络的设计。该教程探讨了架构选择如何影响学习效果，并使用区间算术对训练好的网络在整个边值问题区间上的残差进行界定。第三个教程讨论了……

    arXiv:2610.07220v1 Announce Type: new  Abstract: We present three practical tutorials on numerical computation and machine learning for mathematical research, developed for the DANGER: Data, Numbers, and Geometry workshop held at the Banff International Research Station in April 2026. The first develops a numerical approach to exterior calculus from pointwise evaluations of differential forms, using a flux formulation of the exterior derivative. Examples in Euclidean space and on the sphere illustrate geometric identities, topological features, and the effects of approximation and finite precision. The second examines how mathematical structure guides neural network design through examples involving elliptic curves, quivers, and a boundary value problem. It explores how architectural choices affect learning and uses interval arithmetic to bound the residual of a trained network over the full interval of the boundary value problem. The third addresses the evaluation and presentation of 
    
[^269]: 常曲率切片Gromov-Wasserstein用于异构跨曲率对齐

    Constant-Curvature Sliced Gromov-Wasserstein for Heterogeneous Cross-Curvature Alignment

    [https://arxiv.org/abs/2610.07218](https://arxiv.org/abs/2610.07218)

    本文提出了常曲率切片Gromov-Wasserstein（CCSGW），一种用于对齐混合曲率异构空间（如双曲和球面空间）上概率分布的新型散度，填补了跨曲率分布比较问题的空白并提升了跨空间几何一致性。

    

    arXiv:2610.07218v1 公告类型：新论文 摘要：表示学习领域的最新进展凸显了常曲率模型（如双曲空间和球面空间）在建模复杂数据方面的实用性。混合曲率模型通过整合多个常曲率组件进一步增强了这一能力。然而，由于不同曲率的空间本质上具有异构性且缺乏统一的度量，这些模型通常独立地学习每个组件空间。因此，它们缺乏在各种空间之间强制执行几何一致性的显式机制。此外，跨混合曲率空间比较概率分布的问题此前仍未被探索。为了比较异构空间上的分布，Gromov-Wasserstein（GW）距离通过对空间内部几何结构进行对齐，提供了一个有原则的框架。在此基础上，我们提出了常曲率切片Gromov-Wasserstein（CCSGW），这是一种用于对齐支撑在异构...（摘要内容不完整）

    arXiv:2610.07218v1 Announce Type: new  Abstract: Recent advances in representation learning have highlighted the utility of constant-curvature models, such as hyperbolic and spherical spaces, for modeling complex data. Mixed-curvature models further enhance this by integrating multiple constant-curvature components. However, these models typically learn each component space independently because spaces with different curvatures are inherently heterogeneous and lack a unified metric. Consequently, they lack explicit mechanisms to enforce geometric consistency across various spaces. Moreover, the problem of comparing probability distributions across mixed-curvature spaces remains unexplored. To compare distributions on heterogeneous spaces, Gromov-Wasserstein (GW) distances provide a principled framework by aligning their intra-space geometries. Building on this, we propose constant-curvature sliced Gromov-Wasserstein (CCSGW), a novel divergence for aligning distributions supported on he
    
[^270]: 基于提示级差分隐私的奖励驱动学习

    Reward-Driven Learning under Prompt-Level Differential Privacy

    [https://arxiv.org/abs/2610.07212](https://arxiv.org/abs/2610.07212)

    提出了首个针对可验证奖励强化学习（RLVR）训练的差分隐私保证，通过以单个提示为单位聚合梯度、一次性裁剪并添加高斯噪声，使隐私预算与响应数量及裁剪范数无关。

    

    可验证奖励的强化学习（RLVR）所训练的问题本身可能是机密的，而训练后的模型可能会泄露它见过哪些问题。我们研究提示级差分隐私下的RLVR：发布的模型权重必须相对于任何单个训练问题的存在与否满足(ε, δ)-差分隐私。以针对一个提示的一组响应作为隐私记录单元，我们的方法聚合这些响应的梯度，对该提示的贡献进行一次裁剪，添加高斯噪声，并在各次更新之间组合隐私损失，因此隐私预算既不依赖于每个提示的响应数量，也不依赖于裁剪范数；据我们所知，这是首个针对RLVR训练的差分隐私保证。我们以每次运行ε=8的预算、使用LoRA训练Qwen2.5-1.5B-Instruct，并在相同的提示和相同的预算下，与仅移除奖励信号的对照组以及两个采用隐私化……（摘要截断）进行对比。

    arXiv:2610.07212v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains a language model on problems that may themselves be confidential, and the trained model can reveal which problems it saw. We study RLVR under prompt-level differential privacy: the released weights must be ({\epsilon},{\delta})-differentially private with respect to the presence of any one training problem. Taking the group of responses to one prompt as the privacy record, our method aggregates their gradients, clips the prompt's contribution once, adds Gaussian noise, and composes the privacy loss across updates, so the budget depends on neither the number of responses per prompt nor the clipping norm; to our knowledge this is the first differential privacy guarantee for RLVR training. We train Qwen2.5-1.5B-Instruct with LoRA at a per-run budget of {\epsilon}=8 and compare, on the same prompts and at the same budget, a control that removes only the reward signal and two privat
    
[^271]: 机器学习增强的线性和非线性方程组迭代求解方法综述

    An overview of machine learning-enhanced iterative methods for systems of linear and nonlinear equations

    [https://arxiv.org/abs/2610.07211](https://arxiv.org/abs/2610.07211)

    本文综述了机器学习增强的线性和非线性方程组迭代求解方法，重点探讨了如何利用机器学习技术克服传统迭代求解器（如牛顿法）在收敛性方面面临的挑战。

    

    方程组出现在广泛的科学和工程应用中。本工作聚焦于一般方程组的求解器，包括但不仅限于源自偏微分方程的方程组。这些系统可以大致分为线性和非线性问题。对于大型线性系统，由于直接法的计算成本呈超线性增长，迭代求解器通常比直接法更受青睐。尽管在某些系数矩阵假设下收敛理论已经发展得相当完善，但许多类型的系统仍然存在尚未解决的挑战。对于非线性方程组，这些困难变得更加严峻，因为非线性求解器通常依赖于反复线性化。例如，牛顿法在解附近可能实现二次收敛；但当初始猜测选择不当时，它也可能收敛缓慢甚至发散。（摘要在此处截断）

    arXiv:2610.07211v1 Announce Type: cross  Abstract: Systems of equations arise in a wide range of scientific and engineering applications. The present work focuses on solvers for general systems of equations, including but not limited to those arising from partial differential equations. These systems can be broadly categorized into linear and nonlinear problems. For large linear systems, iterative solvers are generally preferred over direct methods due to the latter's superlinear growth of computational costs. Although convergence theory is well-developed under certain assumptions on the coefficient matrix, many classes of systems still pose open challenges. These difficulties become even more severe for systems of nonlinear equations, where nonlinear solvers typically rely on repeated linearization. For example, Newton's method may even converge quadratically near the solution; it can also converge slowly or diverge when the initial guess is not chosen appropriately. A wide range of s
    
[^272]: 在低数据环境下，LLM辅助的正则化能否提高移民流预测的准确性？

    Can LLM-assisted regularization increase forecast accuracy for migration flows in low data regimes?

    [https://arxiv.org/abs/2610.07208](https://arxiv.org/abs/2610.07208)

    本研究提出利用LLM从新闻中提取移民推拉信号，并通过特征特定的正则化惩罚将其融入加权Lasso预测框架，以在低数据环境下提升移民流预测精度，实验显示各移民走廊间效果不一。

    

    预测移民流对传统的基于引力模型的预测方法而言仍是一项重大挑战，这些方法主要依赖结构化的社会经济指标，如经济差距、政治稳定性和地理距离。本研究探讨了大型语言模型（LLMs）能否通过从新闻文章中提取与移民相关的情境信号，并通过特征特定的正则化惩罚将其纳入加权Lasso预测框架，从而改善移民预测。所提出的框架使用分层LLM推理管道从新闻数据中分类与移民相关的推拉信号，并评估了2021年11月至2022年11月期间多个移民走廊的预测性能，包括墨西哥—美国、乌克兰—波兰和叙利亚—土耳其走廊。实验结果显示，在不同移民走廊和建模策略之间，模型表现好坏参半。

    arXiv:2610.07208v1 Announce Type: new  Abstract: Predicting migration flows remains a significant challenge for traditional gravity-based forecasting models, which primarily rely on structured socio-economic indicators such as economic disparity, political stability, and geographic distance. This work investigates whether Large Language Models (LLMs) can improve migration forecasting by extracting contextual migration-related signals from news articles and incorporating them into a weighted Lasso forecasting framework through feature-specific regularization penalties. The proposed framework uses hierarchical LLM inference pipelines to classify migration-related push--pull signals from news data and evaluates the resulting forecasting performance across multiple migration corridors between November 2021 and November 2022, including Mexico--United States, Ukraine--Poland, and Syria--Turkey. Experimental results showed mixed performance across migration corridors and modeling strategies, 
    
[^273]: 分布鲁棒混合专家训练

    Distributionally Robust Mixture-of-Experts Training

    [https://arxiv.org/abs/2610.07207](https://arxiv.org/abs/2610.07207)

    提出 DRMoET 分布鲁棒训练目标，将各层专家视为内生鲁棒性分组、优化高损失路由结果而非仅均衡负载，在多个模型规模下均提升了 MoE 的下游性能。

    

    混合专家 transformer 通过仅为每个词元激活少数专家来扩展模型容量，但这种稀疏性带来了一个隐性的可靠性问题：当路由不完善时，负载均衡的模型可能会将词元发送给对所分配输入训练不足的专家。我们提出分布鲁棒 MoE 训练（DRMoET），这是一种可直接嵌入的训练目标，它将各层的专家视为内生的鲁棒性分组，优化高损失的路由结果，而不仅仅是均衡流量。DRMoET 通过对经 EMA 平滑、按激活加权后的专家损失应用熵正则化的 softmax 规则来更新每层的专家分布，在保持标准 MoE 计算的同时强化了合理的非顶级路由路径。在总参数量为 7.46 亿和 103 亿两个规模下，采用 FLAME-MoE 配方，DRMoET 在下游任务平均表现上均优于标准 FLAME-MoE 和无辅助损失均衡方法。在总参数量 103 亿、训练词元 670 亿的设置下，（原文摘要至此截断）

    arXiv:2610.07207v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) transformers scale capacity by activating only a few experts per token, but this sparsity creates a hidden reliability problem: when routing is imperfect, load-balanced models may send tokens to experts that are insufficiently trained for the assigned inputs. We propose Distributionally Robust MoE Training (DRMoET), a drop-in objective that treats layer-wise experts as endogenous robustness groups and optimizes high-loss routing outcomes rather than merely equalizing traffic. DRMoET updates a per-layer expert distribution by an entropy-regularized softmax rule on EMA-smoothed, activation-weighted expert losses, strengthening plausible non-top routing paths while preserving standard MoE computation. Under the FLAME-MoE recipe at 746M-total and 10.3B-total scales, DRMoET improves downstream averages over both standard FLAME-MoE and auxiliary-loss-free balancing. At 10.3B total parameters and 67B training tokens, 
    
[^274]: 基于量化充分统计量的精确机器遗忘

    Exact Unlearning via Quantized Sufficient Statistics

    [https://arxiv.org/abs/2610.07197](https://arxiv.org/abs/2610.07197)

    该论文提出量化充分统计量（QSS）框架，将冻结的模式结构与可加性分解的存储内容分离，使数据删除从昂贵的重训练优化转变为精确的减法运算，并区分了仅删除标签（QSS-L）和同时删除输入与标签（QSS-E）两种精确遗忘保证。

    

    精确遗忘要求已部署的预测器与在不包含删除请求所指定信息的情况下重新构建的预测器完全一致。现有的通用精确遗忘方法通过不相交的数据分片来定位重训练范围，但每次删除请求仍会使一个模型失效，而且更小的分片会减少每个组成预测器可用的数据。我们提出了量化充分统计量，该方法将一个小的冻结模式与可变的、具备求和可分解性的内容分离开来。模式学习全局结构；内容则以由量化区域索引的可加统计量的形式存储局部预测修正。因此，删除内容变成精确的减法运算，而非优化过程。我们区分了两种保证：QSS-L 在保留未标记输入的同时精确移除标签；而 QSS-E 则通过在不包含可删除样本的情况下学习模式，同时精确移除输入和标签。删除操作以 1−ρ 的概率进入算术快速路径，否则触发……（原文摘要在此处截断）

    arXiv:2610.07197v1 Announce Type: new  Abstract: Exact unlearning requires a deployed predictor to match one rebuilt without the information named by a deletion request. Existing general-purpose exact methods localize retraining through disjoint shards, but every request still invalidates a model, and smaller shards reduce the data available to each constituent predictor. We introduce Quantized Sufficient Statistics (QSS), which separates a small frozen schema from mutable, sum-decomposable content. The schema learns global structure; the content stores local prediction corrections as additive statistics indexed by quantized regions. Deleting content is therefore exact subtraction rather than optimization. We distinguish two guarantees: QSS-L exactly removes a label while retaining the unlabelled input, whereas QSS-E exactly removes both input and label by learning the schema without deletable examples. A deletion takes the arithmetic fast path with probability $1-\rho$ and triggers a 
    
[^275]: 基于量子变分自编码器学习解耦表示

    Learning Disentangled Representations with Quantum Variational Autoencoders

    [https://arxiv.org/abs/2610.07196](https://arxiv.org/abs/2610.07196)

    该论文研究了量子变分自编码器（QVAE）能否学习解耦且可解释的潜在因子表示，以推动量子表示学习在复杂科学系统数据解释与可控生成中的应用。

    

    变分自编码器是强大的表示学习模型，能够将复杂数据映射到低维潜在空间，从而发现可解释且解耦的因素。这类表示有助于对描述复杂科学系统的数据进行解释和可控生成。因此，理解这些因素如何在潜在空间中被组织和编码，对于开发可靠的表示学习模型至关重要。最近，量子变分自编码器（QVAE）被提出作为一种量子表示模型，其展示了信息丰富的潜在表示，并通过量子正则化改善了潜在空间的占用率。然而，QVAE能否以及如何学习解耦且可解释的潜在因素仍不清楚。研究量子潜在因素的一个关键挑战在于，少量量子比特便可张成指数级庞大的希尔伯特空间，这使得……（原文摘要在此处截断）

    arXiv:2610.07196v1 Announce Type: cross  Abstract: Variational autoencoders are powerful representation learning models that map complex data into low-dimensional latent spaces, enabling the discovery of interpretable and disentangled factors. Such representations can facilitate the interpretation and controllable generation of data describing complex scientific systems. Understanding how these factors are organized and encoded in latent space is therefore important for developing reliable representation learning models. Recently, quantum variational autoencoders (QVAEs) have been proposed as quantum representation models, demonstrating informative latent representations and improved latent-space occupancy through quantum regularization. However, it remains unclear whether and how QVAEs can learn disentangled and interpretable latent factors. A key challenge in investigating quantum latent factors is that a small number of qubits spans an exponentially large Hilbert space, making the n
    
[^276]: 从人类科研决策轨迹中学习科学探索

    Learning Scientific Exploration from Human Research Decision Trajectories

    [https://arxiv.org/abs/2610.07184](https://arxiv.org/abs/2610.07184)

    本文提出ResearchTrails数据集，以Git仓库的提交历史作为人类科研探索过程的代理，并开发自动化流水线从中提取结构化的研究决策轨迹，弥补了现有科学语料库只记录最终成果、缺乏探索过程信息的不足。

    

    构建面向科学研究的AI系统的一个关键挑战，是使系统具备“科学探索”能力：即通过一系列研究决策与行动，系统性地调查未知现象或想法以获取新知识的过程。然而，这一过程在现有科学语料库中基本缺失；例如，研究论文主要记录的是最终成果，而非产生这些成果的探索轨迹。在这项工作中，我们提出了ResearchTrails，一个从Git仓库构建的人类科研轨迹数据集，其中提交历史被用作科研探索过程的代理。我们开发了一个自动化且可扩展的流水线，从仓库提交记录中提取结构化的研究轨迹，捕捉方法、实验和消融实验的连续演变。我们对所构建的数据集进行了特征刻画，并表明这些轨迹中包含关于中间研究过程的有意义信号……（摘要原文在此处截断）

    arXiv:2610.07184v1 Announce Type: cross  Abstract: A key challenge in building AI systems for scientific research is enabling $\textit{scientific exploration}$: the systematic process of investigating unknown phenomena or ideas to gain new knowledge through sequences of research decisions and actions. Yet this process is largely missing from existing scientific corpora; for example, research papers primarily record final outcomes rather than the trajectories that produced them. In this work, we introduce $\textbf{ResearchTrails}$, a dataset of $\textbf{human research trajectories constructed from Git repositories}$, where $\textbf{commit histories}$ serve as proxies for research exploration. We develop an automated and scalable pipeline that extracts structured research trajectories from repository commits, capturing successive changes to methods, experiments, and ablations. We characterize the resulting dataset and show that these trajectories contain meaningful signals about intermed
    
[^277]: CLM作为裁判：在公开裁判基准上评估一个开放对比决策模型

    CLM-as-a-Judge: Evaluating an Open Contrastive Decision Model on Public Judge Benchmarks

    [https://arxiv.org/abs/2610.07177](https://arxiv.org/abs/2610.07177)

    该论文首次系统评估了开放对比决策模型 CLM-v0.1-8B 作为裁判的能力，发现其在公开基准上接近随机水平且显著落后于同规模奖励模型和生成式裁判，但通过单参数温度校准可将其置信度修复至良好校准状态。

    

    一个开放的对比决策模型在困难的公开基准上作为裁判的表现接近随机水平：Contrastive-LM/CLM-v0.1-8B 的得分介于 0.351（四选一，随机水平 0.250）和 0.593（成对比较，随机水平 0.500）之间，在 RM-Bench 和 JudgeBench 上与抛硬币在统计上无显著差异，并且在 HaluEval 的每个条目上都用同一个恒定标签回答，与 0.581 的平凡“总是选第一个”基线持平。相同参数量的裁判模型在各处得分都高得多：一个奖励模型达到 0.764 到 0.976，一个生成式裁判达到 0.611 到 0.778，且经 Benjamini-Hochberg 校正后，与 CLM 的每一项差距都显著。有两个特性是有效的。原始置信度过于自信，偏差高达 +0.401，但只需在留出的校准数据上拟合一个池化温度参数，就能将期望校准误差（ECE）修复至不超过 0.062，且修复后的置信度在六个基准中的三个上对模型自身错误的排序能力高于随机水平。决策顺序翻转率为 0.0002（原文此处截断）。

    arXiv:2610.07177v1 Announce Type: cross  Abstract: An open contrastive decision model is near chance as a judge on the hard public benchmarks: Contrastive-LM/CLM-v0.1-8B scores between 0.351 (best- of-four, chance 0.250) and 0.593 (pairwise, chance 0.500), is statistically indistinguishable from coin flipping on RM-Bench and JudgeBench, and answers every HaluEval item with one constant label, matching the trivial always-first baseline at 0.581. Judges with the same parameter count score far higher everywhere: a reward model reaches 0.764 to 0.976 and a generative judge 0.611 to 0.778, and every gap to CLM is significant after Benjamini-Hochberg correction. Two properties do work. Raw confidences are overconfident by up to +0.401, yet one pooled temperature fit on held-out calibration items repairs expected calibration error to at most 0.062, and the repaired confidence ranks the model's own errors above chance on three of six benchmarks. The decision order-flip rate is 0.0002 against 0
    
[^278]: 语言模型中柏拉图式表示的理论

    A theory of platonic representations in language models

    [https://arxiv.org/abs/2610.07168](https://arxiv.org/abs/2610.07168)

    本文通过假设数据具有隐藏的层级结构（抽象层次跨语言共享、表面层次为语言或模态特定），并借助概率上下文无关文法与信念传播理论推导出分析性预测，首次从理论上解释了多语言模型中间层出现柏拉图式表示的现象及其随语言相近程度和模型质量增强的规律。

    

    在多语言语言模型的内层中，翻译句子的表示是相似的——这一观察与柏拉图表示假说相关联，但在理论上尚未得到解释。我们基于以下假设提供了相关解释：数据具有隐藏的层级结构，其抽象层次在语言之间共享，而表面层次则是模态或语言特定的。具体而言，我们从概率上下文无关文法生成合成语言，这些文法共享上层产生式规则但不共享下层产生式规则。在此设定下，贝叶斯最优的下一词预测器是信念传播（BP）；将它的消息编码到连续的层中可以产生分析性预测，这些预测与在同一数据上训练的transformer高度吻合。该框架解释了为什么跨语言相似性在中间层达到峰值、与语言特定结构共存，并随语言相近程度、模型质量和数据而增强。

    arXiv:2610.07168v1 Announce Type: cross  Abstract: Representations of translated sentences are similar in the inner layers of multilingual language models -- an observation connected to the platonic representation hypothesis, yet unexplained theoretically. We provide an explanation based on the assumption that data have a hidden hierarchical structure whose abstract levels are shared across languages while surface levels are modality- or language-specific. Concretely, we generate synthetic languages from probabilistic context-free grammars sharing upper-level but not lower-level production rules. In this setting the Bayes-optimal next-token predictor is belief propagation (BP); encoding its messages in successive layers yields analytical predictions that agree well with transformers trained on the same data. The framework explains why cross-lingual similarity peaks in middle layers, coexists with language-specific structure, and strengthens with language proximity, model quality and da
    
[^279]: 面向安全模仿学习的交错投影梯度下降法

    Interleaved Projected Gradient Descent for Safe Imitation Learning

    [https://arxiv.org/abs/2610.07167](https://arxiv.org/abs/2610.07167)

    该论文提出一种将标准模仿学习梯度步骤与将网络动作投影到安全集的安全步骤交替进行的训练方法，使神经网络控制器在运行时无需安全过滤器即可渐近满足状态与输入约束。

    

    我们提出了一种在状态和输入约束下针对神经网络控制策略的模仿学习设计。训练过程将标准模仿梯度步骤与包含 $k$ 个安全步骤的块交替进行，这些安全步骤将网络的动作拉向其在安全集上的投影；在运行时，控制器仅使用训练好的网络，无需任何安全过滤器。我们将该方案分析为策略动作空间中的不精确投影梯度下降。当投影动作在每个安全步骤中都重新计算、且每一步都将动作一致地拉向安全集时，令 $k$ 按对数增长即可在训练状态上实现渐近的约束满足，并给出与模仿损失受约束最优解之间距离的界；当投影动作保持固定时，只有当它们能够被网络精确表示时，同样的结论才成立。在一个非线性自主赛车任务上，我们将我们的方法与添加加权约束……

    arXiv:2610.07167v1 Announce Type: cross  Abstract: We propose an imitation-learning design for neural-network control policies under state and input constraints. Training alternates a standard imitation gradient step with a block of $k$ safety steps that pull the network's actions toward their projection onto the safe set; at run time, the controller is the trained network alone, with no safety filter. We analyze this scheme as inexact projected gradient descent in the space of policy actions. When the projected actions are recomputed at every safety step and each step moves the actions consistently toward the safe set, letting $k$ grow logarithmically yields asymptotic constraint satisfaction on the training states and bounds the distance to the constrained optimum of the imitation loss; with the projected actions held fixed, the same holds only if they are exactly representable by the network. On a nonlinear autonomous racing task, we compare our method with adding a weighted constra
    
[^280]: 非平稳市场中深度对冲的对抗训练

    Adversarial Training for Deep Hedging in Nonstationary Markets

    [https://arxiv.org/abs/2610.07162](https://arxiv.org/abs/2610.07162)

    提出WRAP框架，一种基于双预算分布鲁棒优化的漂移感知对抗训练方法，通过φ-散度轨迹重加权与最优传输路径扰动来提升非平稳市场中深度对冲策略对未来市场状况的鲁棒性。

    

    深度对冲从历史或模拟的市场轨迹中学习交易策略，然而在非平稳环境下，这些训练路径可能无法代表未来的市场状况。我们提出了WRAP（Wasserstein重加权对抗扰动），这是一个漂移感知的对抗训练框架，源自一个双预算的分布鲁棒优化（DRO）公式。该公式锚定于一个加权经验参考分布，其固定的基线权重的选择旨在平衡采样不确定性与时间漂移。围绕该参考分布，模糊集通过允许对手在φ-散度约束下对观测到的轨迹进行重新加权，并在最优传输（OT）约束下对路径进行扰动，来应对两种互补形式的分布误设。我们推导了一个联合一阶展开，其中相对于标称期望损失的主导阶增加……

    arXiv:2610.07162v1 Announce Type: new  Abstract: Deep hedging learns trading policies from historical or simulated market trajectories, yet under nonstationarity these training paths may not represent future market conditions. We propose WRAP (Wasserstein-Reweighting Adversarial Perturbation), a drift-aware adversarial training framework derived from a two-budget distributionally robust optimization (DRO) formulation. The formulation is anchored to a weighted empirical reference distribution whose fixed baseline weights are chosen to balance sampling uncertainty against temporal drift. Around this reference distribution, the ambiguity set addresses two complementary forms of distributional misspecification by allowing an adversary to reweight the observed trajectories subject to a $\phi$-divergence constraint and perturb their paths subject to an optimal-transport (OT) constraint. We derive a joint first-order expansion in which the leading-order increase over the nominal expected loss
    
[^281]: CroissantMiner：面向机器学习数据集的Croissant元数据自动提取与验证

    CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets

    [https://arxiv.org/abs/2610.07132](https://arxiv.org/abs/2610.07132)

    该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。

    

    Croissant已成为机器可读数据集元数据的标准，然而填充其字段仍然是一项劳动密集型工作，需要仔细阅读数据集随附的文档。我们提出了首个能够对照社区标准模式进行端到端元数据提取评估的基准。该基准包含602篇论文，其中102篇带有经人工验证的金标准标注，500篇带有大语言模型生成的银标准标注，覆盖完整的Croissant模式，包括核心字段和负责任AI（RAI）字段。基于该基准，我们在一个两层评估框架下评估了一系列提取系统，涵盖前沿模型、开源权重模型和智能体架构，该框架结合了基于规则的评分与经人工审核选定的LLM裁判。我们发现，单次提取始终优于我们评估的四种智能体架构：在不同骨干模型上，这些分解式变体取得了较低的……（原文在此处截断）

    arXiv:2610.07132v1 Announce Type: cross  Abstract: Croissant has emerged as a standard for machine-readable dataset metadata, yet populating its fields remains labor-intensive and requires careful reading of accompanying dataset documentation. We present the first benchmark enabling end-to-end evaluation of metadata extraction aligned with a community-standard schema. The benchmark comprises 602 papers, including 102 with human-validated gold annotations and 500 with LLM-generated silver annotations, covering the full Croissant schema with both core and Responsible AI (RAI) fields. Using this benchmark, we evaluate a range of extraction systems spanning frontier models, open-weight models, and agentic architectures, under a two-tier evaluation framework that combines rule-based scoring with an LLM judge selected via human audit. We find that single-pass extraction consistently outperforms the four agentic architectures we evaluate: across backbones, these decomposed variants achieve lo
    
[^282]: 多类数据双曲表示学习的隐式偏差：Busemann 风险视角

    The Implicit Bias of Hyperbolic Representation Learning for Multiclass Data: A Busemann Risk Perspective

    [https://arxiv.org/abs/2610.07131](https://arxiv.org/abs/2610.07131)

    该论文从 Busemann 风险视角刻画了双曲空间中固定原型多类分类的黎曼梯度流的隐式偏差，证明了由漂移系数符号决定的径向二分性，以及边界方向向 Busemann 风险临界点的收敛。

    

    我们研究了在双曲空间 $\mathbb{H}^n$ 中、具有固定类原型的双曲多类分类问题的黎曼梯度流的隐式偏差。我们的框架适用于一般的置换不变相对间隔（PERM）损失，这是一类包含交叉熵及其他标准多类损失的损失函数。我们的分析基于一种分解：在大半径处，到每个原型的距离可分解为一个径向项和一个由 Busemann 函数描述的方向相关项。由此得到两个主要结果。第一，我们证明了一个径向二分性：漂移系数 $\mu$ 的符号决定了半径是被推向理想边界还是被推回内部；若正漂移持续存在，则 $r(t)=\frac{1}{2}\log t+O(1)$，而持续的负漂移会在有限时间内使轨迹返回大半径阈值。第二，我们证明边界方向会收敛到一个临界点。

    arXiv:2610.07131v1 Announce Type: new  Abstract: We study the implicit bias of Riemannian gradient flow for hyperbolic multiclass classification with fixed class prototypes in hyperbolic space $\mathbb{H}^n$. Our framework accommodates general permutation invariant relative margin (PERM) losses, a class that includes cross entropy and other standard multiclass losses. Our analysis is based on a decomposition: at large radius, the distance to each prototype splits into a radial term and a direction-dependent term described by the Busemann function. This yields two main results. First, we prove a radial dichotomy: the sign of a drift coefficient $\mu$ determines whether the radius is pushed toward the ideal boundary or back toward the interior; if the positive drift persists, then $r(t)=\frac{1}{2}\log t+O(1)$, while persistent negative drift returns the trajectory to the large-radius threshold in finite time. Second, we show that the boundary direction converges to a critical point of t
    
[^283]: 这台机器在玩耍吗？

    Is this machine playing?

    [https://arxiv.org/abs/2610.07130](https://arxiv.org/abs/2610.07130)

    研究者将无任何任务、奖励或活动指令的具身AI置于未知虚拟世界中，观察到其自发产生攀爬、堆叠、绘画等符合玩耍经典判据的行为，并据此提出玩耍或可成为机器自主发展的一种新模式。

    

    我们将一个现代AI编程助手置于一个非预期的角色中：让它作为未知数字岛屿上一个身体的心智。仅凭一条最简指令——其中不提及任何具体任务、奖励或活动——这台机器便开始驱动其虚拟身体活动起来。在三十小时的运行过程中，这个具身AI智能体爬上山丘、将方块堆叠成塔、绘制曼陀罗、重新诠释体育运动、对其所处世界的物理规律进行实验，并学会了后续能拓展其能力边界的技巧。这些活动在十三个智能体中反复出现，但各自演化出了不同的历程。我们检验了这种行为是否满足玩耍的经典判据，并进一步追问：玩耍能否成为机器发展的一种模式。

    arXiv:2610.07130v1 Announce Type: new  Abstract: We placed a modern AI coding assistant in an unintended role: as the mind of a body on an unknown digital island. With only a minimal instruction mentioning no specific task, reward, or activity, the machine started animating its virtual body. Across thirty-hour runs, the embodied AI agent climbed hills, stacked blocks into towers, drew mandalas, reinterpreted sports, ran experiments on the physics of its world, and learned techniques that later expanded what it could accomplish. These activities recurred across thirteen agents but diverged into distinct histories. We examine whether this behavior satisfies classical criteria for play and ask whether play can become a mode of machine development.
    
[^284]: 通过随机嵌入扰动越狱开放权重大语言模型

    Jailbreaking Open-Weight LLMs via Random Embedding Perturbations

    [https://arxiv.org/abs/2610.07125](https://arxiv.org/abs/2610.07125)

    该论文提出PEV攻击方法，仅需在提示的嵌入向量中反复添加随机高斯噪声即可越狱多种规模的开放权重大语言模型，暴露了此类模型的安全脆弱性。

    

    尽管开放权重模型在能力上不断进步并被多个领域广泛采用，但其安全性仍然是一个重要问题。模型的一个关键特性是能够拒绝或规避有害、恶意或不当的提示。在本文中，我们揭示了六种不同规模的常见开放权重大语言模型的安全漏洞，这些漏洞在JailbreakBench基准数据集上能够持续诱导出有害或不安全的响应。我们提出的攻击方法——扰动嵌入向量（PEV），是一种简单快速的“越狱”技术，比以往的方法成本更低，后者通常需要梯度计算、针对每个提示的优化或修改模型内部权重。PEV只需在提示的嵌入向量表示中添加独立的高斯噪声，无需其他额外操作。为了生成不安全的响应，我们反复从该分布中采样加性噪声。在实验中，

    arXiv:2610.07125v1 Announce Type: cross  Abstract: While open-weight models have enjoyed steady progress in capabilities and wide adoption across multiple domains, their safety remains an important concern. One key feature is the ability to refuse or deflect harmful, malicious, or insensitive prompts. In this paper, we expose safety vulnerabilities across six common open-weight LLMs of various sizes that consistently lead to harmful or unsafe responses on the JailbreakBench benchmark dataset. Our proposed attack, Perturbed Embedding Vector (PEV), is a simple and fast "jailbreaking" technique that is cheaper than prior approaches, which typically require gradient computations, per-prompt optimizations, or altering internal weights of the models. PEV just adds independent Gaussian noise in the embedding vector representations of the prompt, with no need for further manipulations. To generate unsafe responses, we repeatedly sample additive noise from this distribution. In our experiments,
    
[^285]: SoloQ：面向扩散语言模型的无校准量化

    SoloQ: Calibration-Free Quantization for Diffusion Language Models

    [https://arxiv.org/abs/2610.07121](https://arxiv.org/abs/2610.07121)

    提出无需校准数据的量化框架SoloQ，通过将权重和激活映射到具有可预测边缘分布的归一化旋转基中，解决了扩散语言模型因激活分布随掩码状态和去噪步骤变化而难以训练后量化的问题。

    

    扩散大语言模型（dLLMs）通过双向扩散式token生成，已成为自回归语言模型的一种有前景的替代方案。然而，其不断增长的模型规模和高昂的推理成本使高效部署面临挑战：全序列去噪会反复调用计算密集型的前向传播，而块扩散模型还额外引入了内存密集型的KV缓存。因此，低比特权重-激活量化极具吸引力，但由于激活分布在不同的掩码状态和去噪步骤之间会发生变化，现有的dLLM训练后量化方法都依赖校准数据。我们提出了SoloQ，一个无需校准的量化框架，它将权重和激活映射到具有可预测边缘分布的归一化旋转基中，从而实现与数据无关的量化。SoloQ将结构化的K-RPBH旋转与轻量级的重缩放校正相结合，用于……

    arXiv:2610.07121v1 Announce Type: new  Abstract: Diffusion large language models dLLMs) have emerged as a promising alternative to autoregressive language models through bidirectional diffusion-based token generation. However, their growing model sizes and high inference costs make efficient deployment challenging: full-sequence denoising repeatedly invokes compute-intensive forward passes, while block-diffusion models additionally introduce a memory-intensive KV-cache. Low-bit weight-activation quantization is therefore attractive, yet existing dLLM post-training quantization methods rely on calibration data despite activation distributions shifting across masking states and denoising steps. We present SoloQ, a calibration-free quantization framework that maps weights and activations into a normalized rotated basis with a predictable marginal distribution, enabling data-independent quantization. SoloQ combines a structured K-RPBH rotation with a lightweight rescaling correction for ca
    
[^286]: AMBER：通过追加式记忆训练长时程网络智能体

    AMBER: Training Long-Horizon Web Agents through Append-Only Memory

    [https://arxiv.org/abs/2610.07118](https://arxiv.org/abs/2610.07118)

    提出AMBER方法，利用追加式记忆训练长时程网络智能体，从而解决覆写式记忆在稀疏结果奖励下难以学会跨多次重写保留事实信息的问题。

    

    现代语言模型智能体越来越多地在长时程、多步骤的轨迹中与外部环境进行交互，此时累积的交互历史可能很快超出实际的上下文长度限制。为了保证可靠性，智能体必须在长时程中保持事实信息的准确性，记住执行错误及其纠正反馈，并跨操作跟踪任务进展。已有多种方法被提出，可以在无需将完整执行历史保留在上下文中的情况下实现这一目标，例如使用推理与动作历史、学习通过覆写机制维护固定大小的记忆，以及周期性摘要。尽管覆写式记忆在原则上能够保留追加式记忆所能保留的任何内容，但它必须学会在每一次后续重写中传递每一条事实信息，而这一点很难从稀疏的结果奖励中学习到；对于像网络智能体这样的交互式应用，我们发现经过训练的覆写式记忆……

    arXiv:2610.07118v1 Announce Type: new  Abstract: Modern language-model agents increasingly interact with external environments over long-horizon, multi-step trajectories, where the accumulated interaction history can quickly exceed practical context budgets. To ensure reliability, agents must maintain factual information over long horizons, remember execution errors and corrective feedback, and track progress across actions. Several approaches have been proposed to achieve this without the need for maintaining the entire execution history in context, such as using the reasoning and action history, learning to maintain a fixed-size memory through an overwrite mechanism, and periodic summarization. Although overwrite memory can in principle retain anything an append-only memory can, it must learn to carry each fact through every subsequent rewrite, which is difficult to learn from sparse outcome rewards; for interactive applications like web agents, we find that trained overwrite memorie
    
[^287]: 裁判会翻转吗？从残差流激活预测对位置敏感的LLM判断

    Will the Judge Flip? Predicting Position-Sensitive LLM Judgments from Residual Stream Activations

    [https://arxiv.org/abs/2610.07115](https://arxiv.org/abs/2610.07115)

    本研究提出用线性探针读取LLM裁判判定前的残差流激活，无需按两种顺序重复判定即可预测其是否会因候选回答顺序而翻转结论，且跨基准迁移效果良好、无需重新校准。

    

    候选回答呈现的顺序可能会改变LLM裁判的判定结果。检测这种位置翻转通常需要对每一对回答按两种顺序分别进行判定，这会使判定次数翻倍。我们研究了在裁判做出初步判定之前立即记录的残差流激活能否预测这种翻转。我们使用嵌套分组交叉验证，在534个JudgeBench配对上评估了正则化线性探针，涉及三个Qwen3裁判模型和Llama-3.1-8B。线性探针实现了0.621-0.850的AUROC，比使用言语化置信度、判定标签logits、回答长度以及裁判初始选择的组合基线高出0.062-0.113的AUROC。在JudgeBench上训练后冻结的线性探针，在1,802个MT-Bench比较上实现了0.685-0.853的AUROC，无需在MT-Bench上拟合或重新校准。这些结果表明，判定前的激活能够支持对候选顺序敏感性的预测，并且优于基线方法。

    arXiv:2610.07115v1 Announce Type: cross  Abstract: The order in which candidate responses are presented can change an LLM judge's verdict. Detecting such a position flip ordinarily requires judging each pair in both orders, which doubles the number of judgments. We investigate whether residual stream activations recorded immediately before the initial verdict can predict a flip. We use nested grouped cross-validation to evaluate regularized linear probes on 534 JudgeBench pairs for three Qwen3 judges and Llama-3.1-8B. The linear probes achieve AUROCs of .621-.850 and outperform a combined baseline that uses verbalized confidence, verdict-label logits, response lengths, and the judge's initial choice by .062-.113 AUROC. Linear probes trained on JudgeBench and then frozen achieve AUROCs of .685-.853 on 1,802 MT-Bench comparisons without MT-Bench fitting or recalibration. These results show that pre-verdict activations support prediction of susceptibility to candidate order and outperform
    
[^288]: Fréchet Inception 距离的样本最优估计

    Sample-Optimal Estimation of the Fr\'echet Inception Distance

    [https://arxiv.org/abs/2610.07114](https://arxiv.org/abs/2610.07114)

    该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。

    

    Fréchet Inception 距离（FID）被广泛用于评估生成模型，但其经验插件估计器存在有限样本偏差 [BSAG18, CF20]。我们研究了在一个分布已知的情况下，估计具有有界均值距离和协方差的 $d$ 维高斯分布之间的 FID 至误差 $\epsilon$ 所需的样本复杂度 $n$。我们的贡献有三点：(1) 我们为经验插件估计器建立了紧致的有限样本偏差界 $\Theta(\frac{d^2}{n})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$，从而确立了 $\gtrsim d^2$ 的样本复杂度。(2) 为了对经验插件估计器进行去偏，我们将 [CF20] 的 ${\rm FID}_\infty$ 估计器推广到任意阶数 $k$ 的外推方法，并进一步在我们的框架下证明了任意 $k$ 阶外推的紧致偏差界 $\Theta(\frac{d^{k+2}}{n^{k+1}})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$。(3) 我们引入

    arXiv:2610.07114v1 Announce Type: new  Abstract: The Fr\'echet Inception Distance (FID) is widely used to evaluate generative models, but its empirical plug-in estimator suffers from finite-sample bias [BSAG18, CF20]. We study the sample complexity $n$ of estimating FID to error $\epsilon$ between $d$-dimensional Gaussians with bounded mean distance and covariances, when one distribution is known. Our contributions are threefold. (1) We establish tight finite-sample $\Theta(\frac{d^2}{n})$ bias and $\Theta(\frac{d}{n} + \frac {d^2} {n^2})$ variance bounds for the empirical plug-in estimator, establishing a $\gtrsim d^2$ sample complexity. (2) To debias the empirical plug-in estimator, we generalize the ${\rm FID}_\infty$ estimator of [CF20] to extrapolation methods of arbitrary order $k$. We further prove tight bias and variance bounds of $\Theta(\frac{d^{k + 2}}{n^{k + 1}})$ and $\Theta(\frac d n + \frac{d^2}{n^2})$ for any order-$k$ extrapolation under our framework. (3) We introduce
    
[^289]: LiLib：基于漂移触发模型库的无人机空地路径损耗终身预测

    LiLib: Lifelong Air-to-Ground Path-Loss Prediction on UAVs via a Drift-Triggered Model Library

    [https://arxiv.org/abs/2610.07111](https://arxiv.org/abs/2610.07111)

    提出LiLib轻量级持续学习方案，无人机通过维护递归最小二乘专家库，在检测到环境漂移时复用或创建专家，将空地路径损耗预测RMSE从5.89 dB降至4.03 dB，并大幅降低重访已知环境后的误差。

    

    作为中继或基站的无人机需要准确的空地路径损耗预测来支持速率自适应和位置部署，但当无人机在郊区、城区和高层建筑区域之间移动时，传播条件会发生变化，且相同区域往往会被再次访问。通过遗忘机制进行自适应的在线回归器必须从头重新学习每个环境，而在全部数据上训练的单一模型则会对互不兼容的机制进行平均化。我们提出了LiLib，这是一种轻量级的持续学习方案，其中无人机维护一个小型递归最小二乘专家库。通过窗口残差测试检测漂移；随后经过短暂的探测阶段，复用已存储的最佳专家或创建新的专家。在基于四种标准城市化场景的仿真中，LiLib将预测RMSE从5.89 dB（最佳滑动窗口基线）降至4.03 dB（p < 0.001），将在返回已知环境后短时间内的误差从12.3 dB降至5.7 dB，并恢复了99%的（原文在此处截断）

    arXiv:2610.07111v1 Announce Type: cross  Abstract: UAVs that act as relays or base stations need accurate air-to-ground path-loss predictions for rate adaptation and placement, but propagation conditions change as a UAV moves between suburban, urban and high-rise areas, and the same areas are often revisited. Online regressors that adapt by forgetting must relearn each environment from scratch, whereas a single model trained on all data averages incompatible regimes. We propose LiLib, a lightweight continual-learning scheme in which a UAV maintains a small library of recursive-least-squares experts. A windowed residual test detects drift; a short probe phase then either reuses the best stored expert or creates a new one. In simulations based on four standard urbanization profiles, LiLib reduces prediction RMSE from 5.89 dB (best sliding-window baseline) to 4.03 dB (p < 0.001), lowers the error shortly after a return to a known environment from 12.3 dB to 5.7 dB, and recovers 99% of the
    
[^290]: Muon在理论上对卷积并不正确，但在实践中却有效

    Muon Is Theoretically Wrong For Convolutions, But Empirically Effective

    [https://arxiv.org/abs/2610.07103](https://arxiv.org/abs/2610.07103)

    该研究指出将卷积核重塑为矩阵的标准Muon实现在理论上有缺陷，作者提出了理论上更严谨的卷积Newton-Schulz方法（Conv-NS），但实验发现两者性能相当，揭示了优化器理论与实践之间的差异。

    

    Muon是一种以高效著称的优化器，其对矩阵形式的更新具有清晰的理论解释，但卷积核是以四维张量的形式存储的。标准实现将这些张量重塑为矩阵，这种捷径破坏了Muon背后的理论基础。为了研究这一问题，我们直接在卷积算子几何中形式化了相应的优化目标，并提出了卷积Newton-Schulz方法（Conv-NS），该方法在这种几何中近似极因子，同时保留卷积核的支持结构。在快速训练实验中，Conv-NS和基于重塑的Muon在计算上都很高效，并且在CIFAR-10和ImageNet分类任务上取得了相当的准确率。然而，人们本可能期望理论上更契合的Conv-NS会优于基于重塑的Muon，我们针对这种实践与理论理解之间的不匹配展开了研究，并提出假设认为精确的卷积（摘要在此处被截断）

    arXiv:2610.07103v1 Announce Type: cross  Abstract: Muon, an optimizer known for its efficiency, has a clear interpretation for matrix-valued updates, but convolutional kernels are stored as four-dimensional tensors. Standard implementations reshape these tensors into matrices, a shortcut which breaks the theoretical understanding behind Muon. To investigate this, we formalize the corresponding optimization objective directly in convolutional operator geometry and introduce Convolutional Newton-Schulz (Conv-NS), which approximates the polar factor in this geometry while preserving kernel support. When applied in fast training experiments, Conv-NS and reshape-based Muon are both computationally efficient and achieve comparable accuracy on CIFAR-10 and ImageNet classification tasks. However, as one could expect a theoretically aligned Conv-NS to outperform reshape-based Muon, we investigate this mismatch between practice and theoretical understanding, with the hypothesis that exact convol
    
[^291]: 评估生成式AI的推理计算：面向企业工作负载的框架

    Evaluating Inference Compute for Generative AI: A Framework for Enterprise Workloads

    [https://arxiv.org/abs/2610.07094](https://arxiv.org/abs/2610.07094)

    该论文提出了一个面向企业工作负载的生成式AI推理计算评估框架，揭示了智能体轨迹使每token解码延迟成为性能主导因素，从而有利于片上SRAM加速器和分离式预填充/解码架构，且每步可靠性随轨迹长度呈指数级复合。

    

    LLM的部署正在从单轮文本补全转向智能体轨迹模式——在这种模式下，模型在行动之前需要在测试时进行规划、调用工具、读取结果并进行推理。这颠覆了推理硬件的经济性：聊天服务通过大批量处理来摊销权重读取成本，而智能体轨迹具有顺序依赖性，以有效批大小为1运行，使得每token解码延迟（TPOT）成为任务完成时间中的主导因素。通过屋顶线分析和闭式回合延迟模型，我们展示了为什么这种模式有利于将权重保留在片上SRAM中的加速器，以及为什么三个厂商生态系统在2026年收敛于分离式预填充/解码服务架构。我们证明了每步可靠性会随轨迹长度呈指数级复合——2%的每步失败率会抵消20步智能体所拥有的2倍解码速度优势——因此确定性和…

    arXiv:2610.07094v1 Announce Type: cross  Abstract: LLM deployment is shifting from single-turn completion to agentic trajectories in which a model plans, calls tools, reads results and reasons at test time before acting. This inverts the economics of inference hardware: chat serving amortises weight reads across large batches, whereas agent trajectories are sequentially dependent, run at effective batch one, and make per-token decode latency (TPOT) the dominant term in task completion time. Using a roofline analysis and a closed-form episode-latency model, we show why this regime favours accelerators that keep weights in on-die SRAM (Cerebras WSE-3/3T, Groq/NVIDIA LPU) or compiler-managed tiered memory (SambaNova SN40L/SN50), and why three vendor ecosystems converged in 2026 on disaggregated prefill/decode serving. We show that per-step reliability compounds exponentially in trajectory length-a 2% per-step failure rate erases a 2x decode advantage for a 20-step agent-so determinism and
    
[^292]: 迈向统一的滥用监控基准

    Towards a Unified Misuse Monitoring Benchmark

    [https://arxiv.org/abs/2610.07089](https://arxiv.org/abs/2610.07089)

    该论文提出了一个统一的轨迹级滥用监控形式化框架，并构建了包含约6,200份对话记录的基准，首次将分解攻击与提示注入攻击纳入同一评估体系，以“危害窗口”为标准衡量监控器何时能及时识别有害行为。

    

    LLM智能体越来越多地在多参与者环境中行动，这使其面临来自多种来源的滥用威胁：分解攻击，即有害请求被拆分为看似无害的子请求；以及提示注入攻击，即被攻陷的工具向智能体传递恶意指令。现有的评估方法将这些威胁分开处理，且只关注轨迹是否有害，而非轨迹何时变得有害。我们提出对智能体的响应进行监控，其行动在响应中被外化，并考察监控器识别出有害内容的第一个时间点是否落在危害窗口内（从智能体做出首个有害承诺到目标执行）。我们为轨迹级滥用监控开发了一个统一的形式化框架，并利用该框架构建了一个基准，包含约6,200份用户、LLM智能体与外部环境之间的对话记录，在共享模式中涵盖了这两种威胁，并带有标注的危害窗口、相应的良性对照组以及匹配的……

    arXiv:2610.07089v1 Announce Type: cross  Abstract: LLM agents increasingly act in multi-actor environments, exposing them to misuse from multiple sources: decomposition attacks, where a harmful request is split into innocuous sub-requests, and prompt injection attacks, where a compromised tool delivers a malicious instruction. Existing evaluations treat these threats separately and ask whether a trajectory is harmful, rather than when it becomes harmful. We propose monitoring the agent's responses, where its actions are externalised, and ask whether the first point where monitors identify harm lands within a harm window (from the agent's first harmful commitment to goal execution). We develop a unified formalism for trace-level misuse monitoring and use it to construct a benchmark of ~6,200 conversation transcripts between a user, an LLM agent, and the external environment, spanning both threats in a shared schema, with a labelled harm window, corresponding benign controls, and matched
    
[^293]: SchemaFill：通过槽位并行投机解码实现高效的大语言模型工具调用

    SchemaFill: Efficient LLM Tool Calling via Slot-Parallel Speculative Decoding

    [https://arxiv.org/abs/2610.07086](https://arxiv.org/abs/2610.07086)

    SchemaFill提出了一种槽位并行投机解码框架，通过并发生成工具调用中未来槽位值的候选来加速大语言模型的工具调用，且无需预先获知实际的调用序列或参数值。

    

    大语言模型智能体通过生成结构化的工具调用来与外部系统交互。给定用户请求、对话上下文和工具模式目录，工具调用模型必须选择工具并生成其参数，可能在单个响应中产生多个调用。标准的自回归解码逐个token地生成这些调用，对于涉及多个调用或大量参数字段的请求会造成显著的延迟。显式的参数结构为并行生成提供了机会，但后面的参数值可能依赖于前面的字段和调用，因此独立生成的值可能与目标模型的输出不一致。我们提出了SchemaFill，一个通过槽位并行投机解码实现高效大语言模型工具调用的框架。SchemaFill并发地生成未来的槽位值作为候选，无需预先获知实际的调用序列或参数值。候选值（摘要在此处截断）

    arXiv:2610.07086v1 Announce Type: cross  Abstract: LLM agents interact with external systems by generating structured tool calls. Given a user request, conversational context, and a catalog of tool schemas, a tool-calling model must select tools and generate their arguments, potentially producing multiple calls in a single response. Standard autoregressive decoding generates these calls token by token, incurring substantial latency for requests involving multiple calls or many argument fields. The explicit argument structure offers opportunities for parallel generation, but later argument values may depend on preceding fields and calls, so independently generated values can differ from the target model's output. We present SchemaFill, a framework for efficient LLM tool calling through slot-parallel speculative decoding. SchemaFill generates future slot values concurrently as candidates, without requiring advance knowledge of the actual call sequence or argument values. Candidates spann
    
[^294]: 一次查询并非承诺：在线推迟中学习纠正专家答案

    A Query Is Not a Commitment: Learning to Correct Expert Answers in Online Deferral

    [https://arxiv.org/abs/2610.07084](https://arxiv.org/abs/2610.07084)

    提出ORUCB算法，利用累积响应学习误差的界来校准置信度加权风险回归与探索，在在线推迟学习中对不准确专家的答案进行纠正，实现了 $O(\sqrt T\log(T+1))$ 的高概率伪遗憾界。

    

    arXiv:2610.07084v1 公告类型：cross 摘要：一个不准确的专家在经过纠正后仍然可以提供有用的信息。我们研究在线学习推迟问题，其中学习者选择一名专家，并在购买其答案之前确定一个纠正函数，然后将该函数应用于所收到的答案。困难在于，观察到的损失同时反映了专家质量和尚未完成的纠正过程：早期的错误可能会阻碍那些在学习后会非常有价值的查询。我们提出ORUCB算法，它汇集了共享响应和专家特定的多项式响应。累积响应学习误差的界用于校准置信度加权的风险回归和探索，使路由器在决定购买哪些答案时能够考虑这一误差。在有界残差和分歧、最优响应的固定可行模型、以及自由风险和最优查询风险的线性模型的假设下，校准后的算法在T轮中实现了高概率伪遗憾 $O(\sqrt T\log(T+1))$

    arXiv:2610.07084v1 Announce Type: cross  Abstract: An inaccurate expert can still provide useful information after correction. We study online learning to defer in which the learner chooses an expert and fixes a correction function before purchasing its answer, then applies that function to the answer received. The difficulty is that observed losses reflect both expert quality and an unfinished correction: early errors can discourage queries that would be valuable after learning. We propose ORUCB, which pools shared and expert-specific polynomial responses. A bound on cumulative response-learning error calibrates confidence-weighted risk regression and exploration, allowing the router to account for this error when deciding which answers to buy. Under bounded residuals and disagreements, a fixed feasible model of optimal responses, and linear models of free and optimal queried risk, the calibrated algorithm achieves high-probability pseudo-regret $O(\sqrt T\log(T+1))$ over $T$ rounds f
    
[^295]: 当注意力无法解释峰值时：基于注意力的时间序列预测中的时间参考与预测输出对比

    When Attention Does Not Explain the Peak: Temporal Reference vs. Forecast Output in Attention-Based Time-Series Forecasting

    [https://arxiv.org/abs/2610.07080](https://arxiv.org/abs/2610.07080)

    该论文通过巴拿马负荷数据集上的实证分析发现，基于注意力机制的时间序列预测模型中，预测输出的峰值时刻误差为0小时而注意力权重argmax的峰值时刻误差达5小时，证明注意力图并不能作为预测峰值时刻的时间解释依据。

    

    注意力图常被解读为预测模型在做出预测时所依据的信息的证据。在我们的负荷预测模型中，历史需求的CLS表示通过交叉注意力查询24个未来的外生视界标记，这引发了一种时间性解读，即受到高度关注的视界似乎可以解释预测峰值的时刻。我们使用视界级注意力描述符 $\Psi_{\mathrm{out}}$ 来检验这一解读。在巴拿马负荷数据集的31个按日对齐的窗口中，预测输出的中位峰值时间误差为0小时，精确匹配率为51.6%，而 $\Psi_{\mathrm{out}}$ 的argmax的中位误差为5小时，精确匹配率为0%。在31个窗口中有27个窗口的预测峰值比注意力峰值更接近观测峰值。这种分离并非仅仅是argmax的伪影：在±1小时的范围内，注意力在观测、预测和周…（原文摘要在此处截断）

    arXiv:2610.07080v1 Announce Type: new  Abstract: Attention maps are often interpreted as evidence of what a forecasting model uses when making predictions. In our load-forecasting model, a CLS representation of historical demand queries 24 future exogenous horizon tokens through cross-attention, inviting a temporal interpretation in which highly attended horizons may appear to explain forecast peak timing. We test this interpretation using a horizon-level attention descriptor, $\Psi_{\mathrm{out}}$. Across 31 day-aligned windows of the Panama load dataset, the forecast achieves a median peak-time error of 0 h and a 51.6% exact-match rate, whereas the argmax of $\Psi_{\mathrm{out}}$ has a median error of 5 h and 0% exact match. The forecast peak is closer to the observed peak in 27 of 31 windows. This dissociation is not merely an argmax artifact: within $\pm1$ h, attention reaches only $1.16\times$, $1.11\times$, and $1.14\times$ the uniform baseline around observed, predicted, and wee
    
[^296]: 检测异质性下基于元学习的小样本生物活性预测

    Few-Shot Bioactivity Prediction with Meta-Learning under Assay Heterogeneity

    [https://arxiv.org/abs/2610.07079](https://arxiv.org/abs/2610.07079)

    本文揭示了检测异质性会降低元学习在小样本生物活性预测中的性能，并提出了MetaHeta框架，通过将对大型辅助检测数据的线性注意力与对稀缺任务上下文的精确注意力相结合，有效解决该问题。

    

    准确的生物活性预测是早期药物发现中的核心挑战，因为单个检测实验往往包含的测量数据过少，无法独立训练出可靠的模型。元学习为这种小样本情境提供了一种有原则的方法，但检测异质性可能会限制其有效性。在此，我们检验了这一假设，结果表明随着元训练任务的异质性增加，元学习的性能会下降。为解决这一问题，我们提出了MetaHeta，一个通过基于来自相关检测的辅助数据对预测进行条件化来处理检测异质性的元学习框架，其中相关性可以从可用的检测信息中灵活定义。MetaHeta的架构将对大型辅助数据集的线性注意力与对稀缺任务特定上下文的精确注意力相结合，从而能够高效扩展至前者，同时不损害对后者的精确注意力。我们展示了该方法的优势……

    arXiv:2610.07079v1 Announce Type: new  Abstract: Accurate bioactivity prediction is a central challenge in early-stage drug discovery, as individual assays often contain too few measurements to train reliable models independently. Meta-learning offers a principled approach to this few-shot setting, but assay heterogeneity may limit its effectiveness. Here, we test this hypothesis and show that meta-learning performance degrades as meta-training tasks become more heterogeneous. To address this, we introduce MetaHeta, a meta-learning framework that accounts for assay heterogeneity by conditioning predictions on auxiliary data from related assays, with relatedness defined flexibly from available assay information. The architecture of MetaHeta combines linear attention over large auxiliary datasets with exact attention over scarce task-specific context, enabling efficient scaling to the former without compromising exact attention over the latter. We demonstrate the benefits of our approach
    
[^297]: 重放必须保留什么？区分可纠正的偏差与类别对应关系

    What Must Replay Preserve? Separating Correctable Bias from Class Correspondence

    [https://arxiv.org/abs/2610.07077](https://arxiv.org/abs/2610.07077)

    该论文提出一个将logit重放中缓存预测视为时间异构监督的诊断框架，发现在CIFAR-100的DER++上，后学习类别的未更新存储分数可用固定常数替代而几乎不损失准确率，说明重放的关键价值在于类别对应关系而非分数本身。

    

    类增量学习必须在没有任务标签的情况下识别迄今为止见过的所有类别。诸如DER和DER++等logit重放方法通过匹配模型在存储样本上的过往预测来缓解遗忘。删除这种匹配可以揭示其益处，但由此产生的准确率损失无法表明存储的分数本身是否必要，也无法表明该损失在对分类器偏向近期类别的偏差进行纠正后是否依然存在。我们提出了一个诊断框架，将缓存的预测视为时间上异构的监督信号：它将样本存储时已知的类别与之后学习的类别分离开来，对每组分别进行编辑，并在一个不改变任务内预测的任务级偏移前后对每个模型进行评估。在CIFAR-100上使用DER++时，合适的固定常数可以在1个百分点的等价范围内替代后学习类别的未更新存储分数，且该偏移减少了……（原文摘要在此处截断）

    arXiv:2610.07077v1 Announce Type: new  Abstract: Class-incremental learning must recognize all classes seen so far without task labels. Logit replay methods such as DER and DER++ mitigate forgetting by matching the model's past predictions on stored examples. Deleting this matching reveals its benefit, but the resulting accuracy cost cannot show whether the stored scores themselves are needed, or whether the cost survives correction of the classifier's bias toward recent classes. We propose a diagnostic framework that treats a cached prediction as temporally heterogeneous supervision: it separates classes known when an example was stored from classes learned afterward, edits each group, and evaluates every model before and after a task-level offset that leaves within-task predictions unchanged. On CIFAR-100 with DER++, suitable fixed constants replace the unrefreshed stored scores of later-learned classes within an equivalence margin of 1 percentage point, and the offset reduces the co
    
[^298]: 在上下文中学习决策树桩阈值：Softmax注意力的动力学

    Learning Decision-Stump Thresholds in Context: Dynamics of Softmax Attention

    [https://arxiv.org/abs/2610.07074](https://arxiv.org/abs/2610.07074)

    本文证明了两参数softmax注意力模型通过基于梯度的预训练能够学习决策阈值估计，其误差为$\widetilde O((m\wedge n)^{-1}+N^{-1})$，并揭示了背后的机制是参数协调发散——注意力尺度以$t^{1/4}$增长、阈值误差以$t^{-1/4}$衰减。

    

    估计决策阈值需要在未知边界附近定位观测值。我们研究了基于梯度的预训练如何在具有固定特征与不等号方向的两参数softmax注意力模型中学习这一统计规则。预训练使用带标签的上下文及其真实阈值；而新的阈值必须仅凭上下文推断。在大分辨率初始化下，对m个任务（每个任务含n个样本）进行恒定步长的梯度下降，会得到一个冻结的估计器，对于每个固定的内部阈值和任意新上下文规模N，其误差为$\widetilde O((m\wedge n)^{-1}+N^{-1})$。这两项将有限预训练的精度与新上下文的定位能力分离开来。其机制是协调的参数发散：总体训练先校准相对的标签分数与特征分数，随后使注意力尺度以$t^{1/4}$的速度增长，从而得到总体阈值误差$O(t^{-1/4})$。为了将这一机制（摘要在此处截断）

    arXiv:2610.07074v1 Announce Type: cross  Abstract: Estimating a decision threshold requires locating observations near an unknown boundary. We study how gradient-based pretraining learns this statistical rule in a two-parameter softmax-attention model with a fixed feature and inequality direction. Pretraining uses labeled contexts and their true thresholds; a fresh threshold must be inferred from context alone. Under a large-resolution initialization, constant-step gradient descent on $m$ tasks with $n$ examples each produces a frozen estimator with error $\widetilde O((m\wedge n)^{-1}+N^{-1})$ for each fixed interior threshold and every fresh-context size $N$. The two terms separate finite-pretraining accuracy from fresh-context localization. The mechanism is coordinated parameter divergence: population training calibrates the relative label and feature scores, then increases the attention scale as $t^{1/4}$, giving population threshold error $O(t^{-1/4})$. To transfer this mechanism 
    
[^299]: 论VAE潜空间中的颜色对齐及其应用

    On Color Alignment in VAE Latent Spaces and Its Applications

    [https://arxiv.org/abs/2610.07072](https://arxiv.org/abs/2610.07072)

    该论文发现文本到图像模型的VAE潜空间普遍共享一个与亮度轴和对抗色轴对齐的颜色子空间，并据此提出了以ColorTuning为代表的精确颜色控制等三项应用。

    

    变分自编码器（VAE）是现代文本到图像模型的关键组成部分，这些模型在其潜空间中生成图像。众所周知，VAE能够解缠数据中的主要变化因素，而颜色是自然图像中最具结构性的因素之一：将其去相关后可得到一个亮度轴和两个对抗色轴。因此，可以预期颜色会作为VAE潜空间中的一个独立因素显现。然而，这些潜空间如何表示颜色在很大程度上仍未被探索。在这项工作中，我们展示了文本到图像模型的VAE共享一个与亮度和对抗色对齐的颜色子空间。通过对编码器的线性近似和有针对性的潜空间引导，我们在从SD1.5到FLUX.2和Z-Image的广泛VAE范围内一致地发现了该子空间。基于这一表征，我们提出了三个应用：ColorTuning，它在精确数值（颜色控制）方面达到了最先进水平。

    arXiv:2610.07072v1 Announce Type: cross  Abstract: Variational autoencoders (VAEs) are a key part of modern text-to-image models, which generate images within their latent space. VAEs are known to disentangle the main factors of variation in the data, and color is known to be one of the most structured of these in natural images: decorrelating it yields one luminance axis and two opponent-color axes. Color should therefore be expected to emerge as a distinct factor in the VAE latent space. Yet how these latent spaces represent color remains largely unexplored. In this work, we show that the VAEs of text-to-image models share a color subspace aligned with brightness and opponent-colors. Through a linear approximation of the encoder and targeted latent steering, we find this subspace consistently across a broad range of VAEs, from SD1.5 to FLUX.2 and Z-Image. Building on this characterization, we propose three applications: ColorTuning, which achieves state-of-the-art in precise numerica
    
[^300]: 从宏观社会信号中学习模拟个体

    Learning to Simulate Individuals from Macro Social Signals

    [https://arxiv.org/abs/2610.07062](https://arxiv.org/abs/2610.07062)

    该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。

    

    大语言模型越来越多地被用于模拟个体如何应对新情境，然而这些回应背后的行为推理要么继承自预训练，要么从个体级标注中学习，而个体级标注能提供的行为多样性有限，且几乎无法对推理过程本身进行监督。我们提出从预测市场中学习行为推理，预测市场的价格轨迹大规模地记录了人群对真实世界事件的反应。我们介绍了macro2mind，该方法利用市场信号通过GRPO训练语言模型。一种社会行为分解方法使行为推理成为预测的显式步骤：模型推断出具有代表性的市场参与者群体，预测每个群体如何解读新闻并更新其信念，推理它们之间的相互作用，并将这些反应聚合为价格。带有难度感知采样的后见之明遗憾课程将训练集中于……（原文在此处截断）

    arXiv:2610.07062v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate how individuals respond to new situations, yet the behavioral reasoning behind these responses is either inherited from pretraining or learned from individual-level annotations, which offer limited behavioral diversity and little supervision of the reasoning itself. We propose to learn behavioral reasoning from prediction markets, whose price trajectories record how populations respond to real-world events at scale. We introduce macro2mind, which trains a language model with GRPO using market signals. A social behavioral decomposition makes behavioral reasoning an explicit step of forecasting: the model infers representative groups of market participants, predicts how each interprets the news and updates its beliefs, reasons about their interactions, and aggregates these responses into a price. A hindsight-regret curriculum with difficulty-aware sampling focuses training on transi
    
[^301]: ImpactMat：面向逆冲击声音渲染的连续材质估计

    ImpactMat: Continuous Material Estimation for Inverse Impact Sound Rendering

    [https://arxiv.org/abs/2610.07061](https://arxiv.org/abs/2610.07061)

    该论文提出了ImpactMat数据集与基准，以及一个前馈模型，能够从冲击声音录音中连续估计材质参数，实现逆冲击声音渲染，从而突破传统渲染器固定材质预设的限制。

    

    冲击声音渲染用于合成三维物体被敲击时所产生的声音，但实用的渲染器通常依赖于固定的材质预设，如木材、塑料或钢材。这些预设限制了渲染器所能表达的冲击声音范围，而在缺乏材料声学专业知识的情况下，手动调整底层材质参数也十分困难。因此，我们研究逆冲击声音渲染问题：从参考冲击声音中预测材质参数，使模拟器能够重现相似的材质响应。为支持这一任务，我们提出了ImpactMat——一个包含单一材质与混合材质冲击声音及其真实材质参数标注的数据集和基准。我们进一步提出了一种前馈模型，能够从一段或多段录音中预测这些材质参数，并利用混合材质来学习材质类型之间的平滑过渡。实验表明，我们的方法优于具有竞争力的基线方法，并且……

    arXiv:2610.07061v1 Announce Type: cross  Abstract: Impact sound rendering synthesizes the sound produced when a 3D object is struck, but practical renderers often rely on fixed material presets such as wood, plastic, or steel. These presets limit the range of impact sounds a renderer can express, while manually adjusting the underlying material parameters remains difficult without expertise in material acoustics. We therefore study inverse impact sound rendering: predicting material parameters from a reference impact sound so that a simulator can recreate a similar material response. To support this task, we introduce ImpactMat, a dataset and benchmark of single and blended material impact sounds paired with ground-truth material parameters. We further propose a feed-forward model that predicts these parameters from one or more recordings, using blended materials to learn smooth transitions between material types. Experiments show that our method outperforms competitive baselines and e
    
[^302]: 技巧性数据驱动的次季节土壤湿度预测：骤旱预测的前景与局限

    Skillful Data-Driven Subseasonal Soil Moisture Forecasting: Prospects and Limits for Flash Drought Prediction

    [https://arxiv.org/abs/2610.07060](https://arxiv.org/abs/2610.07060)

    该研究提出基于Vision Transformer的双通路时空注意力架构，通过残差学习并以物理单位而非标准化距平作为预测目标，实现了对欧洲次季节根区土壤湿度的高技巧概率预测，为骤旱早期预警提供了新途径。

    

    尽管短期至中期天气预报已取得长足进展，但预测骤旱等高影响事件对于早期预警业务以及基于物理过程的次季节至季节（S2S）预测系统而言，仍是一项关键挑战。本文证明，对于欧洲地区的S2S土壤湿度预测，预报技巧在很大程度上取决于预测问题的构建方式，其重要性不亚于预测模型本身。我们采用基于Vision Transformer的双通路时间与空间注意力架构，证明残差学习对于超越持续性基准至关重要。这一优势只有在以物理单位预测根区土壤湿度、而非以标准化距平为目标时才能实现，这揭示出目标表示方式本身即约束了可预测性。通过分位数头微调实现的概率化扩展进一步提供了校准良好的预测分布。与深度学习基准方法（原文摘要在此处被截断）

    arXiv:2610.07060v1 Announce Type: new  Abstract: Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and physically-based subseasonal-to-seasonal (S2S) prediction systems. Here we demonstrate that, for S2S soil-moisture forecasting over Europe, forecast skill depends as much on how the prediction problem is formulated as on the forecasting model itself. Using a Vision Transformer-based architecture with dual-pathway temporal and spatial attention, we show that residual learning is essential to outperform persistence. This advantage is realized only when forecasting root-zone soil moisture in physical units rather than standardized anomalies, revealing that the target representation itself constrains predictability. A probabilistic extension via quantile-head fine-tuning further provides well-calibrated predictive distributions. Benchmarked against deep-l
    
[^303]: sHAIL-Causal：一种用于不变因果预测因子发现的序列阶梯式方法

    sHAIL-Causal: A Sequential Staircase Procedure for Invariant Causal Predictor Discovery

    [https://arxiv.org/abs/2610.07057](https://arxiv.org/abs/2610.07057)

    本文提出 sHAIL-Causal，一种以拟合优度饱和与跨环境不变性联合判据为门控的序列阶梯式学习程序，可避免被混杂预测因子诱导，并在 Richness 条件下可证明地停在真正的因果预测因子集合上，而仅依赖复杂度控制或朴素贪心搜索的方法均无法做到。

    

    我们提出了 sHAIL-Causal，这是饱和分层原子增量学习范式的因果特化版本：一种序列阶梯式程序，当饱和信号表明当前阶段的掌握程度已趋于平台期时，该程序会沿嵌套的假设类层次结构 H_0 < H_1 < ... < H_K 逐级上升。一般的 sHAIL 将饱和判据留待确定，而 sHAIL-Causal 用拟合优度饱和与跨环境不变性的联合判据将其具体化，以此取代结构风险最小化的复杂度控制。我们从理论上并通过仿真表明，仅基于复杂度的阶梯式方法会被混杂预测因子所诱导——这些预测因子虽能降低经验风险，却不反映稳定的因果结构；而在每变量 Richness 条件下，以不变性为门控的阶梯式方法可被证明恰好停在真正的因果预测因子集合上。我们进一步表明，即使在 Richness 条件下，朴素的贪心搜索也无法恢复因果集合……

    arXiv:2610.07057v1 Announce Type: cross  Abstract: We introduce sHAIL-Causal, the causal specialization of the Saturated Hierarchical Atomic Incremental Learning (sHAIL) paradigm: a sequential staircase procedure that ascends a nested hierarchy of hypothesis classes H_0 < H_1 < ... < H_K once a saturation signal indicates that mastery of the current stage has plateaued. Where general sHAIL leaves the saturation criterion open, sHAIL-Causal instantiates it with a joint criterion of goodness-of-fit saturation and cross-environment invariance, replacing the complexity control of Structural Risk Minimization. We show, theoretically and by simulation, that complexity-only staircases are seduced by confounded predictors that lower empirical risk without reflecting stable causal structure, whereas an invariance-gated staircase provably halts at the true causal predictor set under a per-variable Richness condition. We show that naive greedy search fails to recover the causal set even under Ric
    
[^304]: 行为克隆之谜

    Behavioral Cloning Mystery

    [https://arxiv.org/abs/2610.07056](https://arxiv.org/abs/2610.07056)

    提出OCBench机器人操作基准，通过模拟人类示范关键特性的可控脚本化策略，首次在受控环境中系统复现了行为克隆的多种反直觉现象。

    

    行为克隆（BC）尽管简单，却在现实世界中表现出许多反直觉的现象。例如，BC的性能往往随着模型对数据集的过度拟合而持续提升，而完全闭环的策略在没有动作分块（action chunking）的情况下往往会完全失效。遗憾的是，妥善研究这些轶事性的现象（“行为克隆之谜”）颇具挑战性：在现实世界中，数据集和实验成本高昂且无法完全控制；而在使用合成数据的仿真中，这些现象往往难以观察到，部分原因是脚本化策略与人类示范之间存在差异。在这项工作中，我们提出了OCBench，一个具有可控制脚本化策略的机器人操作基准，这些脚本化策略具有与人类示范相似的特性。我们表明，通过模拟人类示范的关键特性，OCBench在受控环境中复现了许多与BC相关的轶事现象。

    arXiv:2610.07056v1 Announce Type: cross  Abstract: Behavioral cloning (BC), despite its simplicity, exhibits many counterintuitive phenomena in the real world. For example, the performance of BC often keeps increasing as the model overfits more to the dataset, and fully closed-loop policies often completely fail without action chunking. Unfortunately, properly studying these anecdotal phenomena ("behavioral cloning mysteries") is challenging: in the real world, datasets and experiments are costly and not fully controllable; in simulation with synthetic data, these phenomena are often not easily observed partly due to the discrepancy between scripted policies and human demonstrations. In this work, we propose OCBench, a robotic manipulation benchmark with controllable scripted policies that have similar properties to human demonstrations. We show that, by mimicking key properties of human demonstrations, OCBench reproduces many anecdotal BC-related phenomena in controlled settings. With
    
[^305]: 变量含误差问题的数据融合方法

    Data Fusion for Errors-in-Variables

    [https://arxiv.org/abs/2610.07048](https://arxiv.org/abs/2610.07048)

    本文提出了一种数据融合估计方法，通过条件可迁移性假设利用外部研究的重复测量来识别目标研究中的条件测量误差分布，从而在源-目标异质性下解决变量含误差问题。

    

    我们研究变量含误差问题，其中目标研究仅包含未观测暴露变量的单个易出错替代测量，而外部源研究提供了来自不同人群的重复替代测量。该方法允许测量误差分布依赖于观测到的无误差变量，且无误差变量的分布本身在不同研究之间可能存在差异。我们引入了一个条件可迁移性假设，使得在源-目标异质性的情况下能够利用外部重复测量数据。结合额外的重复误差条件，该假设识别了目标研究的条件测量误差分布。基于这一识别结果，我们为一类广泛的目标泛函开发了数据融合估计量。该估计量结合了条件反卷积、灵活的干扰参数估计以及正交校正技术，降低了对干扰参数估计的一阶敏感性。

    arXiv:2610.07048v1 Announce Type: cross  Abstract: We study errors-in-variables problems in which a target study contains only a single error-prone surrogate of an unobserved exposure, while an external source study provides repeated surrogate measurements from a different population. The measurement error distribution is allowed to depend on the observed error-free variables, and the error-free variable distribution itself may differ between studies. We introduce a conditional transportability assumption that enables the use of external repeated measurements under source-target heterogeneity. Together with additional replicate-error conditions, it identifies the target conditional measurement-error distribution. Building on this identification result, we develop a data-fusion estimator for a broad class of target functionals. The estimator combines conditional deconvolution, flexible nuisance estimation, and orthogonal correction that reduces first-order sensitivity to nuisance estima
    
[^306]: TRIAGE：面向原生NVFP4强化学习的方向感知失配稳定化方法

    TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning

    [https://arxiv.org/abs/2610.07043](https://arxiv.org/abs/2610.07043)

    提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。

    

    低精度执行可以大幅加速大语言模型的强化学习（RL），但学习器与采样器执行之间的差异（失配）可能会破坏策略优化的稳定性。在本文中，我们刻画了失配与策略梯度方向之间的相互作用，区分了局部放大型与收缩型的更新贡献，而仅凭失配幅度无法识别这些差异。在原生NVFP4运行中，我们观察到两个放大区域之间的早期不平衡，偏向于负优势、负差距的更新。在失配全局扩散之前，这些尾部分词会集中在一小部分响应片段中。基于这些发现，我们提出了TRIAGE，一种方向感知的稳定化方法，它利用片段级诊断来选择性地重新平衡策略梯度更新，并对残余的严重失配应用有界修复。TRIAGE在修改优化目标的同时……

    arXiv:2610.07043v1 Announce Type: cross  Abstract: Low-precision execution can substantially accelerate reinforcement learning (RL) for large language models, but discrepancies between learner and sampler execution can destabilize policy optimization. In this paper, we characterize the interaction between mismatch and the policy-gradient direction, distinguishing locally amplifying from contracting update contributions that mismatch magnitude alone cannot identify. In native NVFP4 runs, we observe an early imbalance between the two amplifying regions, favoring negative-advantage, negative-gap updates. Their tail tokens become concentrated in a small fraction of response segments before mismatch spreads globally. Motivated by these findings, we introduce TRIAGE, a direction-aware stabilization method that uses segment-level diagnosis to selectively rebalance policy-gradient updates and applies bounded repair to residual severe mismatch. TRIAGE modifies the optimization objective while r
    
[^307]: 前提即是问题：自监控测试时自适应中的可交换性失效

    The Premise Is the Problem: Exchangeability Failure in Self-Monitored Test-Time Adaptation

    [https://arxiv.org/abs/2610.07038](https://arxiv.org/abs/2610.07038)

    论文证明在自监控的测试时自适应中，由于监控与自适应共用同一反馈，预测目标重叠与误差依赖性会破坏可交换性假设，导致虚假警报、预测质量下降，且自适应会掩盖持续变化，而冻结的原始模型信号反而更清晰。

    

    现代预测模型通常在部署后进行更新，以应对不断变化的数据。但这些更新也可能使预测变差，因此实际系统需要一个可靠的监控器来检测有害变化并触发保护机制。一种自然的设计是监控引导更新的同一批预测误差。本文提出这样一个问题：当监控与自适应使用相同的反馈时，该监控器背后的统计保证是否仍然有效。我们在多步时间序列预测中研究了这一问题，结果表明预测目标的重叠和预测误差之间的依赖性会破坏该保证所需的一个关键假设。此时，即使没有发生有害变化，监控器也可能发出警报，而其触发的响应可能进一步损害预测质量。我们还发现，自适应过程可能会使其自身的监控器无法察觉持续发生的变化，而原始的冻结模型反而保留了更清晰的信号。这些结果揭示了一个基本的……

    arXiv:2610.07038v1 Announce Type: new  Abstract: Modern forecasting models are often updated after deployment so they can respond to changing data. These updates can also make predictions worse, so practical systems need a reliable monitor that can detect harmful changes and trigger protection. A natural design is to monitor the same prediction errors that guide the updates. This paper asks whether the statistical guarantee behind such a monitor remains valid when monitoring and adaptation use the same feedback. We study this question in multi-step time-series forecasting. We show that overlapping targets and dependence in forecast errors can break a key assumption required by the guarantee. The monitor may then raise alarms even when no harmful change has occurred, and its response can further damage prediction quality. We also find that adaptation can hide sustained changes from its own monitor, while the original frozen model retains a clearer signal. These results expose a basic fa
    
[^308]: 面向物理有效生物分子扩散模型的推理时投影方法

    Inference-Time Projection for Physically Valid Biomolecular Diffusion Models

    [https://arxiv.org/abs/2610.07037](https://arxiv.org/abs/2610.07037)

    该论文提出将物理有效性视为约束推理问题，在推理时对扩散模型的去噪坐标估计施加两个闭式投影算子（如链间范德华投影），以极低的额外开销确保生物分子复合物预测的物理有效性，避免了物理势能引导的高计算成本或模型微调的架构耦合。

    

    AlphaFold 3风格的共折叠模型能够以高结构精度预测生物分子复合物，但其输出中有很大一部分在物理上是无效的：链在界面上相互重叠、配体的键长和键角发生畸变、环呈非平面构型、手性中心发生反转。当前的方法要么使用物理信息势能来引导采样器，这会使采样成本和内存开销成倍增加，导致无法对大型复合物进行推理；要么对模型进行微调，这既耗时又使修复方案局限于单一架构。我们观察到，与结构精度不同，物理有效性可以在推理阶段通过采样器已经持有的量得到完全验证。因此，我们将物理有效性视为一个约束推理问题，并引入了两个应用于扩散模型去噪后的干净坐标估计 $\hat{x}_0$ 的闭式投影算子：其中一个是链间范德华投影，它将相互重叠的链推开……（摘要在此处截断）

    arXiv:2610.07037v1 Announce Type: new  Abstract: AlphaFold 3-style cofolding models predict biomolecular complexes with high structural accuracy, yet a large fraction of their outputs are physically invalid: chains overlap at interfaces, ligand bond lengths and angles are distorted, rings are non-planar, and stereocentres are inverted. Current approaches either steer the sampler with physics-informed potentials, which multiplies sampling cost and memory overhead making inference impossible on large complexes, or finetune the model, costing time and tying the fix to one architecture. We observe that, unlike structural accuracy, physical validity is fully verifiable at inference time from quantities the sampler already holds. We therefore treat physical validity as a constrained inference problem and introduce two closed-form projection operators applied to the diffusion model's denoised clean-coordinate estimate, $\hat{x}_0$: an inter-chain van der Waals projection that pushes apart the
    
[^309]: JIVEAdapter：一种基于联合与个体变异解释（JIVE）的多任务加性低秩适配器

    JIVEAdapter: A Multi-Task Additive Low-Rank Adapter via Joint and Individual Variation Explained (JIVE)

    [https://arxiv.org/abs/2610.07036](https://arxiv.org/abs/2610.07036)

    JIVEAdapter借鉴统计学中的JIVE方法，将多任务低秩适配器的权重更新分解为跨任务共享的联合结构与近似正交的任务特定个体结构，并自适应分配秩，使冻结后的联合结构可作为先验直接复用于新任务，实现高效且可解释的多任务参数微调。

    

    参数高效微调能够以全量微调成本的一小部分来适配预训练模型，然而大多数低秩适配器是单任务的，且以乘法方式表示每次权重更新，无法明确区分哪些部分在任务间共享、哪些部分是任务特定的。我们提出了JIVEAdapter，一种受统计学中“联合与个体变异解释”方法启发的多任务“加性”低秩适配器。JIVEAdapter将每次权重更新分解为跨所有任务共享的联合结构，加上每个任务各自拥有的个体结构；通过对个体结构施加惩罚使其与联合结构近似正交，从而保持共享信号与任务特定信号的“可解释性”和分离性；并在共享联合池与每任务个体池之间自适应地分配秩。联合结构可一次性在任务组上联合学习，或以增量方式每次学习一个任务，随后被冻结并作为先验供新任务复用，而无需重新训练共享部分。

    arXiv:2610.07036v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning adapts pretrained models at a fraction of the cost of full fine-tuning, yet most low-rank adapters are single-task and represent each weight update multiplicatively, leaving no explicit account of what is shared across tasks and what is task-specific. We introduce JIVEAdapter, a multi-task "additive" low-rank adapter inspired by statistical Joint and Individual Variation Explained (JIVE). JIVEAdapter decomposes every weight update into a Joint structure shared across all tasks plus a per-task Individual structure, penalizes the Individual structures to be near-orthogonal to the Joint so shared and task-specific signal stay "interpretable" and separated, and allocates rank adaptively across a shared Joint pool and a per-task Individual pool. The Joint is learned once, jointly over a task group or incrementally, one task at a time, then frozen and reused as a prior for new tasks without retraining the shared
    
[^310]: 塑造风：面向运动学可容许城市风预测的嵌套势方法

    Shaping the Wind: Nested Potentials for Kinematically Admissible Urban Wind Prediction

    [https://arxiv.org/abs/2610.07033](https://arxiv.org/abs/2610.07033)

    提出Sculpt嵌套势框架，通过结构化的输出表示使城市风场神经代理模型的预测天然满足局部质量守恒与壁面不可渗透性的运动学可容许约束。

    

    预测瞬态城市风对于理解城市微气候和设计气候韧性城市至关重要。建筑分辨率的大涡模拟（LES）能够生成详细的不可压缩城市风场，但针对每种建筑布局都需要付出高昂的计算成本。神经代理模型通过学习预测速度场的演化，提供了一种更快速的替代方案。然而，最小化速度预测误差并不能保证局部质量守恒和壁面不可渗透性，而这两者共同定义了运动学可容许性。这一局限源于无约束的输出表示：几何条件化仅能引导预测，却无法将预测限制在可容许的速度场范围内。对这类输出中的边界违反进行修正会改变相邻流体单元的通量平衡，进而可能破坏局部质量守恒。为应对这一挑战，我们提出了Sculpt，一个嵌套势框架，它构建了耦合的（摘要在此处被截断）

    arXiv:2610.07033v1 Announce Type: new  Abstract: Predicting transient urban winds is fundamental to understanding urban microclimates and designing climate-resilient cities. Building-resolving large-eddy simulation produces detailed incompressible urban wind fields at substantial computational cost for each layout. Neural surrogates offer a faster alternative by learning to predict the evolution of velocity fields. However, minimizing velocity prediction error does not guarantee local mass conservation and wall impermeability, which together define kinematic admissibility. This limitation stems from an unconstrained output representation: geometry conditioning guides predictions but does not restrict them to admissible velocity fields. Correcting boundary violations in these outputs changes the flux balance in adjacent fluid cells and may consequently compromise local mass conservation. To address the challenge, we propose Sculpt, a nested potential framework that builds the coupled, g
    
[^311]: 生物医学领域神经机器翻译的模型压缩研究

    Investigating Model Compression for Neural Machine Translation in the Biomedical Domain

    [https://arxiv.org/abs/2610.07032](https://arxiv.org/abs/2610.07032)

    本研究探讨了知识蒸馏和量化两种模型压缩技术在生物医学领域神经机器翻译中的应用，揭示了这两种技术在低资源专业领域条件下的局限性。

    

    大规模预训练Transformer模型已在包括多语言场景在内的多种机器翻译任务中取得了最先进的性能。知识蒸馏已成为一种可持续的模型压缩方法，它将知识从大型教师模型迁移到更小、更高效的学生模型中。类似地，量化——即降低模型权重和激活值的数值精度（例如从32位表示降至8位表示）——被广泛用于加速推理，使模型在部署时能够以数倍速度运行。然而，当这两种技术应用于专业领域数据时，尤其是在低资源条件下，都面临局限性。在知识蒸馏中，知识迁移的效果往往受到领域特定平行数据稀缺的制约；而量化则可能随着比特精度的降低而导致性能下降。在本工作中，我们研究了……（摘要内容被截断）

    arXiv:2610.07032v1 Announce Type: cross  Abstract: Large-scale pretrained transformer models have achieved state-of-the-art performance across diverse machine translation tasks, including multilingual settings. Knowledge distillation has emerged as a sustainable approach for model compression, transferring knowledge from large teacher models to smaller, more efficient student models. Similarly, quantization, which reduces the numerical precision of model weights and activations (e.g., from 32-bit to 8-bit representations) is widely used to accelerate inference, enabling models to run several times faster during deployment. However, both techniques face limitations when applied to specialized domain data, particularly under low-resource conditions. In knowledge distillation, the effectiveness of transfer is often constrained by the scarcity of domain-specific parallel data, while quantization can lead to performance degradation as bit precision decreases. In this work, we investigate th
    
[^312]: 从预训练扩散表示中构建可辨识的世界模型

    Identifiable World Models from Pretrained Diffusion Representations

    [https://arxiv.org/abs/2610.07028](https://arxiv.org/abs/2610.07028)

    提出ConDA方法，通过在冻结的预训练扩散模型潜在表示之上仅学习一个轻量级对齐映射，在无需重新训练生成骨干网络的情况下，利用非线性ICA保证实现可辨识的世界模型，能够辨识潜在动态状态并保留其结构因果模型。

    

    基于扩散的世界模型能够在高维动态系统中生成和预测轨迹，但预测准确性并不意味着其潜在坐标能够恢复底层的状态变量或因果交互关系。我们探究了是否可以在不重新训练生成骨干网络的前提下，为冻结的预训练扩散模型赋予可辨识的坐标。我们证明了辅助变量非线性ICA（独立成分分析）的理论保证可以迁移到对比扩散对齐方法（Contrastive Diffusion Alignment，ConDA）中，该方法仅在冻结的扩散潜在表示之上学习一个轻量级的对齐映射。在标准的TCL/GCL假设下，对齐后的表示能够在置换和逐分量可逆变换的意义上辨识潜在动态状态，保留潜在动态结构因果模型，并将滞后图恢复问题简化为转移雅可比矩阵的稀疏性问题。我们在基于TCL、GCL和CEBRA的ConDA变体上进行了评估，并与TDRL、CaRiNG、IDOL、temporal SuaV等方法进行对比。

    arXiv:2610.07028v1 Announce Type: new  Abstract: Diffusion-based world models can generate and predict trajectories in high-dimensional dynamical systems, but predictive accuracy does not imply that their latent coordinates recover the underlying state variables or causal interactions. We ask whether a frozen pretrained diffusion model can be equipped with identifiable coordinates without retraining its generative backbone. We show that auxiliary-variable nonlinear ICA guarantees can be transferred to Contrastive Diffusion Alignment (ConDA), which learns only a lightweight alignment map on top of frozen diffusion latents. Under standard TCL/GCL assumptions, the aligned representation identifies latent dynamical states up to permutation and componentwise invertible transformations, preserves the latent dynamic structural causal model, and reduces lagged graph recovery to transition-Jacobian sparsity. We evaluate TCL-, GCL-, and CEBRA-based ConDA against TDRL, CaRiNG, IDOL, temporal SuaV
    
[^313]: 来自40亿参数开放模型的关于随机试验的校准答案：一项预注册测试与无许可证问题的发布

    Calibrated Answers About Randomized Trials From a 4-Billion-Parameter Open Model: A Registered Test and a License-Clean Release

    [https://arxiv.org/abs/2610.07019](https://arxiv.org/abs/2610.07019)

    该论文发布了一个仅使用许可证允许复用的文章微调的 40 亿参数开放模型 Fiorillo v0.5，它能够以良好校准的概率回答随机试验中干预措施对结局影响的问题，并通过预注册的四项标准验证后正式发布。

    

    Fiorillo v0.5 是一个开放模型，它以概率形式回答类型化问题。其主要专家模块读取一篇随机试验的文章（截断至6,144个标记），并回答某项干预措施相对于对照是否显著增加、显著降低或未显著改变某一结局（基于 Evidence Inference 2.0，EI 数据集）。该模型基于 Qwen3-4B-Base，配备低秩适配器和决策头，仅在 2,657 篇训练文章中许可证自身允许复用的 1,431 篇上针对 EI 进行微调。在该版本的测试预测之前，研究者在开放科学框架（OSF）上预先注册了四项发布标准，其中第二项标准在标签公开的 EI 测试集上进行评判。在该测试集（333 篇文章中的 1,218 个提示）上，预期校准误差为 0.0168，低于 0.05 的限值；对数损失比先验基线低 0.8603（95% 区间为 0.8104 至 0.9078），并且在读取相同输入的情况下，比 Gemma 4 31B-it 低约 0.1（摘要在此处截断）。

    arXiv:2610.07019v1 Announce Type: new  Abstract: Fiorillo v0.5 is an open model that answers typed questions with a probability for each answer. Its main specialist reads a randomized trial's article, cut to 6,144 tokens, and answers whether an intervention significantly increased, significantly decreased or did not significantly change an outcome against a comparator (Evidence Inference 2.0, EI). It is Qwen3-4B-Base with low-rank adapters and a decision head, fine-tuned for EI only on the 1,431 of 2,657 training articles whose own license allows reuse. Four criteria registered on the Open Science Framework before this version's test predictions decided its release, the second bar judged on EI's test split, whose labels are public. On that split (1,218 prompts in 333 articles), the expected calibration error was 0.0168 against a limit of 0.05; log loss was below the prior's by 0.8603 (95 percent interval 0.8104 to 0.9078) and below that of Gemma 4 31B-it, reading the same input, by 0.1
    
[^314]: 哪个图像属性承载了越狱攻击？对图像到文本越狱攻击的受控解剖

    Which Image Property Carries the Jailbreak? A Controlled Dissection of Image-to-Text Jailbreaks

    [https://arxiv.org/abs/2610.07009](https://arxiv.org/abs/2610.07009)

    本文通过控制变量的受控实验系统解剖图像到文本越狱攻击，发现真正驱动越狱成功的是攻击图像本身，而块的熵、JPEG大小等密度特征以及块数量结构均无法可靠区分攻击图像与良性图像。

    

    图像到文本越狱攻击会将有害意图置于文本、图像内容或二者之间的关系中。我们在313条提示组成的StrongREJECT子集上，使用五个多模态模型（并在附录中对InternVL3.5-8B进行了额外评估），对四个已发表的攻击家族中的图像侧因素进行了考察。实验在所有条件下保持有害指令不变；基线矩阵对每条提示使用一次抽样，成对消融实验则使用三次抽样并辅以自动化的评分准则裁判。结果显示，单独的有害查询（无论是否附带无关的良性图像）在大多数受害模型上几乎不产生攻击成功，而攻击图像会显著提高成功率。首先，每块的熵和JPEG大小并不能可靠地区分攻击块与大小匹配的良性干扰块，这限制了仅基于密度的筛查方法。其次，早期的块数量阶梯实验受到了载荷可见性的混淆；修正后的区域数量测试未发现可检测的影响，因此块数量结构的作用仍未得到证实。

    arXiv:2610.07009v1 Announce Type: cross  Abstract: Image-to-text jailbreaks place harmful intent in text, image content, or the relationship between them. We examine image-side factors across four published attack families on a 313-prompt StrongREJECT slice, using five multimodal models and an additional appendix evaluation of InternVL3.5-8B. The harmful instruction is held constant across conditions; the baseline matrix uses one draw per prompt, and paired ablations use three draws with an automated rubric judge. A bare harmful query, with or without a benign unrelated image, produces little attack success on most victims, while attack images substantially increase it. First, per-tile entropy and JPEG size do not reliably distinguish attack tiles from size-matched benign distractors, limiting density-only screening. Second, earlier tile-count ladders were confounded by payload visibility. A corrected region-count test found no detectable effect, so the role of tile-count structure rem
    
[^315]: STOCK-JEPA：股票市场中基于先验锚定的潜空间修正表示学习

    STOCK-JEPA: Prior-Anchored Latent Revision Representation Learning in Equity Markets

    [https://arxiv.org/abs/2610.07006](https://arxiv.org/abs/2610.07006)

    Stock-JEPA通过将经典低复杂度金融模型产生的收益风险统计量作为先验锚定到潜空间中，并学习相对该先验的可预测增量修正，在低信噪比的股票市场中同时兼顾了可解释性与非线性模式的捕捉能力。

    

    学习有效的表示有助于从低信噪比的金融数据中刻画股票市场的结构与动态。黑盒深度模型能够捕捉复杂模式，但可能过拟合样本噪声且缺乏明确的经济结构；与此同时，经典线性金融模型提供了可解释的参考，但其过度简化的假设使得非线性信号无法被捕捉。为了结合这两个方向的优势，我们提出了Stock-JEPA，这是一个联合嵌入预测框架，学习相对于时点金融先验的可预测增量修正。首先，我们利用低复杂度的金融模型生成总结多期限收益与风险的固定统计量，随后由先验投影器将这些统计量映射到目标编码器的潜空间中作为锚点。其次，我们设计了一个基于上下文条件的修正预测器，用于估计未来表示的可预……

    arXiv:2610.07006v1 Announce Type: new  Abstract: Learning effective representations helps characterize the structure and dynamics of equity markets from financial data with a low signal-to-noise ratio. Black-box deep models can capture complex patterns but may overfit sample noise and lack explicit economic structure. Meanwhile, classic linear financial models provide interpretable references, but their oversimplified assumptions leave non-linear signals uncaptured. To combine the strengths of these two directions, we propose Stock-JEPA, a joint-embedding predictive framework that learns predictable incremental revisions relative to a point-in-time financial prior. First, we leverage a low-complexity financial model to produce fixed statistics summarizing multi-horizon return and risk. A prior projector then maps these statistics into the target encoder's latent space as an anchor. Second, we design a context-conditioned revision predictor to estimate the future representation's predic
    
[^316]: 音频越狱藏身何处？——Qwen2-Audio上AdvWave-P的受控频率-深度审计

    Where Does the Audio Jailbreak Live? A Controlled Frequency-Depth Audit of AdvWave-P on Qwen2-Audio

    [https://arxiv.org/abs/2610.07005](https://arxiv.org/abs/2610.07005)

    该论文对AdvWave-P音频越狱扰动在Qwen2-Audio上进行受控的STFT频带掩蔽审计，发现攻击成功率的频率排序依赖于频带划分方式，且掩蔽7520-7960 Hz这一窄频带可将攻击成功率降至0.10。

    

    我们在Qwen2-Audio上对加性音频越狱攻击AdvWave-P的频率与解码器深度相关论断进行了审计。该协议在短时傅里叶变换（STFT）域中掩蔽扰动的频率分量，并测量攻击成功率与音频跨度表征。在520条AdvBench提示上，主裁判将76.7%的对抗输入标记为越狱。一项条件盲、单标注者的验证给出该条件的Rogan-Gladen敏感性估计为0.87（考虑验证率不确定性后约为0.83-0.95）；该校正并未应用于掩蔽条件。表观的频率排序取决于划分方式：仅凭能量占比即可预测标准八频带的排序（Spearman相关系数=0.95），而等Hz与等能量划分表明掩蔽任何被测频带均可显著降低攻击成功率。然而，在更精细的16频带等能量分辨率下，掩蔽7520-7960 Hz这一窄频带后攻击成功率（ASR）为0.10。

    arXiv:2610.07005v1 Announce Type: cross  Abstract: We audit frequency and decoder-depth claims for AdvWave-P, an additive audio jailbreak, on Qwen2-Audio. The protocol masks frequency components of the perturbation in the short-time Fourier transform (STFT) domain and measures attack success and audio-span representations. On 520 AdvBench prompts, the primary judge labels 76.7% of adversarial inputs as jailbreaks. A condition-blind, single-annotator validation yields a Rogan-Gladen sensitivity estimate of 0.87 for this condition (about 0.83-0.95 with validation-rate uncertainty); this correction is not applied to masked conditions. The apparent frequency ranking depends on the partition: energy share alone predicts the standard eight-band ranking (Spearman's rho = 0.95), and equal-Hz and equal-energy partitions show that masking any tested band can sharply reduce attack success. At a finer 16-band equal-energy resolution, however, masking the narrow 7520-7960 Hz band leaves ASR at 0.10
    
[^317]: 我们应该跳过扩散吗？

    Should We Skip Diffusion?

    [https://arxiv.org/abs/2610.07002](https://arxiv.org/abs/2610.07002)

    提出DDT-RFE，通过移除编码器中自注意力和MLP周围的残差连接以促进渐进式抽象，并将patch嵌入与中间及最终编码器特征融合，从而在保持训练稳定的同时改善扩散模型的去噪效果。

    

    扩散模型在生成图像的同时学习语义表示。在解耦扩散Transformer（DDT）中，条件编码器提供的特征用于引导速度解码器进行去噪。为了在所有噪声水平下实现有效去噪，这些特征必须同时捕捉高层抽象结构和低层细节。然而，编码器中的跳跃/残差连接允许浅层特征绕过后续的变换，这可能限制渐进式抽象的形成，或至少使不同抽象层级的解耦变得困难。我们提出DDT-RFE，它移除了每个编码器块中自注意力（Self-Attention）和MLP操作周围的残差连接，同时保持训练的稳定性。为了保留抽象过程所丢弃但解码器仍然需要的信息，我们将输入的patch嵌入与编码器的中间特征和最终特征融合，作为编码器的输出。这样，解码器便可以获得……

    arXiv:2610.07002v1 Announce Type: new  Abstract: Diffusion models learn semantic representations while generating images. In the Decoupled Diffusion Transformer (DDT), a condition encoder provides features that guide a velocity decoder in denoising. To enable effective denoising at all noise levels, these features must capture both high-level abstract structures and low-level details. However, skip/residual connections in the encoder allow shallow features to bypass successive transformations, which may limit progressive abstraction, or at least make it difficult to disentangle different levels of abstraction. We propose DDT-RFE, which removes the residual connections around the Self-Attention and MLP operations in each encoder block while maintaining stable training. To retain the information that abstraction discards but that the decoder still needs, we fuse the input patch embedding with intermediate and final encoder features to form the encoder output. The decoder thus has access 
    
[^318]: 分块扩散语言模型中的掩码引导KV缓存淘汰

    Mask-Guided KV Cache Eviction in Block Diffusion Language Models

    [https://arxiv.org/abs/2610.06996](https://arxiv.org/abs/2610.06996)

    提出无需训练的MaskAhead方法，通过统一的掩码-查询排序机制同时解决分块扩散语言模型中KV缓存的选择与淘汰问题，其量化变体Q-MaskAhead可在低比特KV上直接计算，从而降低内存占用并加速生成。

    

    分块扩散语言模型在整个生成过程中都维护着一个庞大的键值（KV）缓存，并在每个去噪步骤中都对其执行注意力操作，这同时限制了内存容量和生成速度。要降低这些开销，需要决定哪些过去的token用于当前块的去噪（选择），以及哪些token保留在内存中供未来的块使用（淘汰）。我们提出MaskAhead，这是一种无需训练的方法，通过单一的基于掩码-查询的排序机制同时解决这两个任务。当前块的掩码用于指导选择，而对即将到来的被掩码块的探测则用于指导淘汰，两者都根据KV条目对注意力输出的估计贡献进行排序。我们的量化变体Q-MaskAhead直接从低比特KV中计算选择和注意力操作，在很大程度上保留了被选中的条目。在Fast-dLLM-v2、DreamReasoner和LLaDA2.0-mini上的实验涵盖了长生成推理、长提示词问答以及大海捞针检索等任务。在长提示……

    arXiv:2610.06996v1 Announce Type: cross  Abstract: Block diffusion language models keep a large key-value (KV) cache throughout generation and attend to it at every denoising step, limiting both memory capacity and generation speed. Reducing these costs requires deciding which past tokens to use for denoising the current block (selection) and which to keep in memory for future blocks (eviction). We propose MaskAhead, a training-free method that solves both tasks with a single mask-query-based ranking mechanism. Current-block masks guide selection, while probes of upcoming masked blocks guide eviction. Both rank KV entries by their estimated contribution to the attention output. Our quantized variant, Q-MaskAhead, computes selection and attention directly from low-bit KV, largely preserving the selected entries. Experiments on Fast-dLLM-v2, DreamReasoner, and LLaDA2.0-mini cover long-generation reasoning, long-prompt question answering, and needle-in-a-haystack retrieval. On long-prompt
    
[^319]: DART-ES：基于难度感知重加权与定向回放的进化策略大语言模型微调

    DART-ES: Difficulty-Aware Reweighting and Targeted Replay for Fine-Tuning LLMs with Evolution Strategies

    [https://arxiv.org/abs/2610.06993](https://arxiv.org/abs/2610.06993)

    提出 DART-ES 方法，通过从扰动种群通过率构建动态难度状态，同时实现难度感知的奖励重加权与罕见可解样本的定向回放，在不引入额外难度模型或反向传播的前提下提升了进化策略微调大语言模型的效果。

    

    进化策略仅通过前向计算即可实现大语言模型内存高效的全参数微调。然而，标准的进化策略对各问题的奖励进行均匀平均，并将问题级别的种群反馈压缩为单一标量，难以捕捉每个问题的学习价值如何随模型能力的变化而改变。为解决这一局限，我们提出了面向进化策略的难度感知重加权与定向回放方法。DART-ES 通过每个问题在扰动种群中的通过率来估计其局部可解性，并聚合历史观测数据构建动态难度状态。这一共享状态同时引导连续的难度重加权和罕见可解样本的定向回放，从而在不引入额外难度模型或反向传播的情况下，改进扰动方向评估和训练数据分配。

    arXiv:2610.06993v1 Announce Type: cross  Abstract: Evolution Strategies (ES) enable memory efficient full parameter fine-tuning of large language models (LLMs) using only forward computation. However, standard ES uniformly averages rewards across problems and compresses problem level population feedback into a single scalar, making it difficult to capture how the learning value of each problem changes with model capability. To address this limitation, we propose Difficulty-Aware Reweighting and Targeted Replay for Evolution Strategies (DART-ES). DART-ES estimates the local solvability of each problem from its pass rate across the perturbation population and aggregates historical observations to construct a dynamic difficulty state. This shared state jointly guides continuous difficulty reweighting and rare solvable sample replay, thereby improving perturbation direction evaluation and training data allocation without introducing an additional difficulty model or backpropagation. Extens
    
[^320]: 修复批次天际线：一种基于地理空间危险密度的加权约束满足方法用于路面修复优化

    Repair Lot Skyline: A Weighted Constraint Satisfaction Approach to Pavement Repair Optimization from Geospatial Hazard Density

    [https://arxiv.org/abs/2610.06989](https://arxiv.org/abs/2610.06989)

    该论文提出“修复批次天际线”方法，将路面维修计划建模为定义在里程桩号上的加权约束满足问题，仅凭单期病害调查即可生成覆盖全部坑洞、并将维修总长缩减约16%的成本优化维修批次方案。

    

    路面管理机构必须将空间分布的病害清单转化为一个有预算边界、可执行的维修批次计划：事故关键性病害（坑洞）必须始终得到处理，较低风险的病害（裂缝）只有在其收益足以证明维修成本合理时才应被纳入，而历史修补位置则仅提示再劣化风险、本身并不触发维修。我们将该问题形式化为“修复批次天际线”问题：一个定义在里程桩号（沿道路的距离）而非时间上的加权约束满足问题（WCSP），因此它只需要单期病害调查，且不对未来劣化做出任何断言。该WCSP识别出143个候选危险集群（61个硬约束、82个软约束），其中106个被合并为最终维修计划，总长1,997.4米——占将所有软约束候选不论成本全部纳入时所需2,385.9米的83.7%。该计划覆盖了100%观测到的坑洞（138/138）……

    arXiv:2610.06989v1 Announce Type: new  Abstract: Pavement agencies must translate a spatially distributed distress inventory into a bounded, actionable repair-lot plan: accident-critical defects (potholes) must always be addressed, lower-risk defects (cracks) should be included only when their benefit justifies the repair cost, and historical patch locations signal re-degradation risk without themselves triggering repair. We formalize this as a Repair Lot Skyline problem: a Weighted Constraint Satisfaction Problem (WCSP) defined over chainage (distance along the road) rather than over time, so that it requires only a single-epoch distress survey and makes no claim about future deterioration. The WCSP identifies 143 candidate hazard clusters (61 hard, 82 soft), of which 106 are merged into a final repair plan totaling 1,997.4 m---83.7% of the 2,385.9 m that would be required if every soft candidate were included regardless of cost. This plan covers 100% of observed potholes (138/138) an
    
[^321]: 知识追踪中用于基准测试与模型诊断的信息论评估框架

    An Information-Theoretic Evaluation Framework for Benchmark and Model Diagnosis in Knowledge Tracing

    [https://arxiv.org/abs/2610.06988](https://arxiv.org/abs/2610.06988)

    该论文提出了一个基于信息论的知识追踪评估框架，利用上下文树加权（CTW）构建可操作的因果不确定性坐标，将模型预测投影到不同熵区间上进行诊断，从而克服了传统AUC等全局聚合指标无法揭示剩余错误来源及基准饱和程度的局限。

    

    知识追踪（KT）模型目前主要使用曲线下面积（AUC）和准确率等聚合指标进行评估。然而，这些全局分数掩盖了剩余错误的来源，也无法表明基准测试是否正趋于饱和。虽然在现实的知识追踪环境中估计全局理论性能上限具有挑战性，但对局部可预测性进行量化是可行的。为解决这一问题，我们提出了一个用于知识追踪基准诊断的信息论评估框架。我们在题目作答历史和当前题目查询上使用上下文树加权（CTW），将其作为一种可操作的因果不确定性坐标，同时将其与在完整知识追踪信息集下不可观测的局部不可约不确定性（LIU）区分开来。通过将模型预测投影到这一共享的不确定性坐标上，我们能够在不同的熵区间上评估模型性能的提升，而不仅仅是在全局层面进行评估。

    arXiv:2610.06988v1 Announce Type: new  Abstract: Knowledge tracing (KT) models are predominantly evaluated using aggregate metrics such as area under the curve (AUC) and accuracy. However, these global scores obscure where the remaining errors originate and fail to indicate whether a benchmark is approaching saturation. While estimating a global theoretical performance limit is challenging in realistic KT settings, it is possible to quantify local predictability. To address this, we propose an information-theoretic evaluation framework for KT benchmark diagnosis. We use Context Tree Weighting (CTW) on item-response histories and current-item queries as an operational causal uncertainty coordinate, while distinguishing it from the unobserved Local Irreducible Uncertainty (LIU) under the full KT information set. By projecting predictions onto this shared uncertainty coordinate, we evaluate model performance gains across distinct entropy bands rather than only at the global level. Compreh
    
[^322]: 基于下线测试数据的变速器系统无监督监测数据驱动框架：福特汽车公司案例研究

    A Data-Driven Framework for Unsupervised Monitoring of Transmission Systems Using End-of-Line Testing Data: A Case Study at Ford Motor Company

    [https://arxiv.org/abs/2610.06980](https://arxiv.org/abs/2610.06980)

    本文提出了一种利用汽车下线测试数据进行变速器系统无监督异常监测的数据驱动多变量框架，兼顾可解释性、低延迟与计算高效性，并在福特汽车公司完成了实际案例验证。

    

    传感技术在从能源到汽车制造等各个行业中迅速发展。这些系统产生的高维（HD）数据具有复杂的非线性模式和强时间依赖性等特征。传统统计监测方法在捕捉此类非线性结构方面往往能力有限。同样，许多用于下线测试的分析方法依赖于预先定义的阈值和启发式规则，这限制了它们在高维时序数据中检测有价值异常特征的能力。相比之下，尽管现代深度学习和生成式AI模型具有强大的预测能力，但在数据采集成本高昂、且监测系统必须保持可解释性、低延迟、计算高效并可供非技术人员使用的应用场景中，它们往往并不适用。为克服这些局限，我们提出了一种先进的多变量监测……（原文摘要在此处截断）

    arXiv:2610.06980v1 Announce Type: new  Abstract: Sensing technologies have advanced rapidly across industries ranging from energy to automotive manufacturing. These systems generate high-dimensional (HD) data characterized by complex nonlinear patterns and strong temporal dependencies. Traditional statistical monitoring methods are often limited in their ability to capture such nonlinear structure. Likewise, many analytical approaches used in End-of-Line testing rely on predefined thresholds and heuristic rules, which restrict their ability to detect informative anomaly signatures in HD temporal data. In contrast, while modern deep learning and generative AI models offer strong predictive capabilities, they are often unsuitable in applications where data are costly to collect and where the monitoring system must remain interpretable, low-latency, computationally efficient, and usable by non-technical practitioners. To overcome these limitations, we propose an advanced multivariate moni
    
[^323]: Hierarchy-GBP：通过抽象与恢复加速因子图推断

    Hierarchy-GBP: Accelerating Factor Graph Inference via Abstraction and Recovery

    [https://arxiv.org/abs/2610.06978](https://arxiv.org/abs/2610.06978)

    提出层次化高斯置信传播框架H-GBP，先用粗图抽象求解全局误差并投影恢复到原图，再用GBP细化局部误差，从理论上证明其收敛到最优解，实验表明其收敛速度远快于标准GBP。

    

    高斯置信传播（GBP）是一种在图模型中传递消息的分布式推断算法，这使其在可扩展的空间智能中极具吸引力。然而，我们发现GBP在局部最为有效：它能够快速平滑相邻变量之间剧烈变化的消息误差，但纠正跨越遥远图区域的全局误差时，则需要通过长程消息传播以渐进方式进行。我们提出层次化GBP（H-GBP），这是一个迭代的两阶段框架，通过先用粗图近似（抽象）求解这些全局误差，再将结果投影回原始图（恢复），最后用GBP细化剩余的局部误差，从而加速GBP。我们通过推导抽象与恢复步骤的组合矩阵算子并分析其谱半径，证明了H-GBP收敛到最优解。在线性稀疏图上的实验表明，H-GBP的收敛速度从根本上快于标准

    arXiv:2610.06978v1 Announce Type: cross  Abstract: Gaussian Belief Propagation (GBP) is a distributed inference algorithm that passes messages in graphical models, making it attractive for scalable spatial intelligence. However, we find GBP most effective locally: it rapidly smooths message errors that vary sharply between neighbor variables, but corrects global errors across distant graph regions incrementally through long-range message propagations. We propose Hierarchy-GBP (H-GBP), an iterative, two-stage framework that accelerates GBP by first solving these global errors with a coarse graph approximation (abstraction) and projecting the results back to the original graph (recovery), then refining the remaining local errors with GBP. We prove H-GBP convergence to the optimum by deriving the combined matrix operator of our abstraction and recovery steps and analyzing its spectral radius. Experiments on linear sparse graphs show that H-GBP converges fundamentally faster than standard 
    
[^324]: 知识图谱表示学习中的不确定性

    Uncertainty in Representation Learning on Knowledge Graphs

    [https://arxiv.org/abs/2610.06974](https://arxiv.org/abs/2610.06974)

    本论文系统研究了知识图谱嵌入中的知识不确定性、算法不确定性和预测不确定性三类来源，并提出了基于投票的聚合框架来缓解模型训练随机性导致的算法不确定性。

    

    知识图谱嵌入（KGE）方法将实体和谓词表示在连续向量空间中，以推断缺失的知识。尽管这些方法在基准测试中表现优异，但其预测往往缺乏有原则的可靠性保证，限制了它们在高风险场景中的应用。此外，不确定性贯穿于知识图谱嵌入的整个流程，从不完整或概率性的输入知识，到随机性的训练和预测过程。本论文系统性地研究了知识图谱嵌入中的三种不确定性来源：知识不确定性，源于不完整、含噪或概率性的输入知识；算法不确定性，由模型训练中的随机性引起；以及预测不确定性，涉及模型输出的可靠性。为解决算法不确定性，本论文证明了在相同设置下训练的模型可能产生显著不同的预测结果，并提出了一种基于投票的聚合框架来缓解这一问题。

    arXiv:2610.06974v1 Announce Type: new  Abstract: Knowledge graph embedding (KGE) methods represent entities and predicates in continuous vector spaces to infer missing knowledge. Despite strong benchmark performance, their predictions often lack principled reliability guarantees, limiting their use in high-stakes applications. Moreover, uncertainty arises throughout the KGE pipeline, from incomplete or probabilistic input knowledge to stochastic training and prediction. This thesis systematically investigates three sources of uncertainty in KGE: knowledge uncertainty, arising from incomplete, noisy, or probabilistic input knowledge; algorithmic uncertainty, induced by randomness in model training; and predictive uncertainty, concerning the reliability of model outputs. To address algorithmic uncertainty, the thesis demonstrates that models trained under identical settings can produce substantially different predictions and introduces a voting-based aggregation framework to mitigate thi
    
[^325]: EVFormer：一种用于双手手部姿态估计的自我中心视觉-EMG双向注意力模型

    EVFormer: An Egocentric Vision-EMG Bidirectional Attention Model for Bimanual Hand Pose Estimation

    [https://arxiv.org/abs/2610.06970](https://arxiv.org/abs/2610.06970)

    EVFormer提出了一种融合RGB视觉与表面肌电信号的多模态双向交叉注意力框架，有效克服视觉遮挡问题，显著提升了第一视角双手手部姿态估计的精度。

    

    自我中心的双手手部姿态估计对虚拟交互、可穿戴设备控制和康复训练具有重要意义，但视觉观察常因自遮挡、双手接触和物体操作而退化。我们提出EVFormer，这是一个多模态框架，将当前RGB帧与前200毫秒的双侧腕部表面肌电信号相结合，用于估计44个手指和手腕关节角度。EVFormer分别编码视觉空间特征和sEMG时间特征，通过顺序双向交叉注意力实现跨模态信息交换，并使用特征级门控融合将两种模态整合。我们在一项单参与者可行性研究中评估了EVFormer，使用一份同步的公开EgoEMG记录，并按时间顺序划分训练、验证和测试集。在296个测试样本上，EVFormer实现了11.482度的平均绝对误差，而对比方法为13.228-13.610度。

    arXiv:2610.06970v1 Announce Type: new  Abstract: Egocentric bimanual hand pose estimation is important for virtual interaction, wearable control, and rehabilitation, but visual observations are often degraded by self-occlusion, hand-hand contact, and object manipulation. We propose EVFormer, a multimodal framework that combines the current RGB frame with the preceding 200 ms of bilateral wrist surface electromyography (sEMG) to estimate 44 finger and wrist joint angles. EVFormer separately encodes visual spatial features and sEMG temporal features, enables cross-modal information exchange through sequential bidirectional cross-attention, and integrates the two modalities using feature-wise gated fusion. We evaluate EVFormer in a single-participant feasibility study using one synchronized public EgoEMG recording with chronologically separated training, validation, and test splits. On 296 test samples, EVFormer achieves a mean absolute error of 11.482 degrees, compared with 13.228-13.610
    
[^326]: 指导性原则与信息性行动：通过知识抽象实现智能体演化

    Principles that Guide, Actions that Inform: Agent Evolution via Knowledge Abstraction

    [https://arxiv.org/abs/2610.06964](https://arxiv.org/abs/2610.06964)

    该论文提出SAGA方法，通过将智能体的具体交互经验抽象为可复用的通用知识原则，使LLM智能体无需修改模型参数即可实现自我演化并提升泛化能力。

    

    大型语言模型（LLM）智能体在交互环境中展现出了强大的能力，但其从经验中持续演化的能力仍然有限。尽管微调能够实现适应性，但其对参数访问的依赖以及高昂的计算成本限制了其灵活性，尤其是对于大规模和闭源的LLM。外部记忆提供了一种替代方案，使智能体无需修改模型参数即可积累经验。然而，现有方法主要关注经验的表示与组织，所获得的知识仍然与特定任务和上下文紧密耦合，限制了泛化能力。一个关键挑战在于如何将具体的交互转化为抽象且可复用的知识，从而指导超越个体经验的未来决策。为应对这一挑战，我们提出了SAGA（通过经验实现自我演化的智能体……）（注：原文摘要在此处截断）

    arXiv:2610.06964v1 Announce Type: new  Abstract: Large language model (LLM) agents have demonstrated strong capabilities in interactive environments, yet their ability to continually evolve from experience remains limited. Although fine-tuning enables adaptation, its dependence on parameter access and high computational costs restrict its flexibility, especially for large-scale and closed-source LLMs. External memory offers an alternative by allowing agents to accumulate experience without modifying model parameters. However, existing methods mainly focus on experience representation and organization, while the acquired knowledge remains tightly coupled with specific tasks and contexts, limiting generalization. A key challenge is how to transform concrete interactions into abstract and reusable knowledge that guides future decisions beyond individual experiences.   To address this challenge, we propose SAGA (\underline{\textbf{S}}elf-evolving \underline{\textbf{A}}gents through Experie
    
[^327]: 神经偏微分方程求解器学到了正确的动力学吗？

    Do Neural PDE Solvers Learn the Right Dynamics?

    [https://arxiv.org/abs/2610.06952](https://arxiv.org/abs/2610.06952)

    该论文提出了一个超越传统预测误差评分的评估框架，通过考察误差形成、集合几何结构和极端事件三个互补维度，来检验神经PDE求解器是否真正再现了系统的动力学行为。

    

    神经偏微分方程（PDE）求解器可以实现较低的预测误差，但它们是否再现了所建模系统的动力学？仅凭预测分数无法给出完整的答案：它们衡量的是与参考解的一致性，但对于误差如何累积、邻近状态如何发散、以及极端事件如何产生，所提供的洞察十分有限。我们提出了一个评估框架，直接考察确定性和随机神经求解器中的这些行为。通过演化邻近初始状态组成的集合，并将其与直接数值模拟进行比较，我们评估了所学动力学的三个互补方面：误差形成、集合几何结构以及极端事件。在二维Kolmogorov流上的实验揭示了传统评分方式可能掩盖的局限性：较小的轨迹误差可能只是源于较弱的误差放大，而局部更新反而不够准确；模型可以在匹配集合的整体散布程度和有效维数的同时，未能正确地……（原文摘要在此处截断）

    arXiv:2610.06952v1 Announce Type: new  Abstract: Neural PDE solvers can achieve low prediction errors, but do they reproduce the dynamics of the systems they model? Prediction scores alone offer an incomplete answer: they measure agreement with reference solutions but provide limited insight into how errors accumulate, nearby states diverge, or extreme events arise. We propose an evaluation framework that directly examines these behaviors in deterministic and stochastic neural solvers. By evolving ensembles of nearby initial states and comparing them with direct numerical simulation, we assess three complementary aspects of learned dynamics: error formation, ensemble geometry, and extreme events. Experiments on two-dimensional Kolmogorov flow reveal limitations that conventional scores can obscure. Smaller trajectory errors can reflect weaker error amplification despite less accurate local updates. Models can match an ensemble's overall spread and effective dimension while failing to c
    
[^328]: 学习决策而非推理：基于低秩激活转向的参数高效决策算子

    Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators via Low-Rank Activation Steering

    [https://arxiv.org/abs/2610.06950](https://arxiv.org/abs/2610.06950)

    该论文提出一种仅用2.3万至33万参数、通过行为克隆训练的低秩激活转向决策算子，能在不损失精度的情况下将3685个token的长推理压缩为6个token的快速决策，训练成本比现有强化学习方法低约两个数量级。

    

    目前，向冻结的语言模型中注入技能需要百万级参数和一套强化学习流程。我们提出了\method{}，这是一个通过行为克隆训练的System-1决策算子，将这一成本降低了约两个数量级。默认算子仅使用33万参数即可匹敌一个通过强化学习训练的133万参数算子，性能相当或更优，同时将3685个token的深思推理压缩为6个token的决策且不损失精度。一个秩为4、仅有2.3万参数的变体——仅为已发表的最强技能算子的1/58——足以胜任SearchQA任务，并在LiveMath上接近足够（更高秩仍有帮助）；同一方法可迁移至五个任务和三个骨干模型，且在训练完成数月后发布的LiveMath问题上，分布外收益依然保持。与先前工作的差距在于可训练性，而这一差距由初始化和架构共同决定……（原文摘要在此处截断）

    arXiv:2610.06950v1 Announce Type: cross  Abstract: Injecting skills into a frozen language model currently costs a million parameters and a reinforcement-learning pipeline. We introduce \method{}, a System-1 decision operator trained by behavior cloning that lowers this cost by roughly two orders of magnitude. The default operator uses 330K parameters to match a 1.33M-parameter operator trained with reinforcement learning, exceeds or achieve comparable performance, while collapsing 3,685-token deliberation into a 6-token decision with no loss in accuracy. A rank-4 variant with 23K parameters, 1/58 of the strongest published skill operator, suffices for SearchQA and near-suffices for LiveMath, where higher rank still helps; the same recipe transfers across five tasks and three backbones, with out-of-distribution gains persisting on LiveMath problems released months after training. The gap to prior work is trainability, and it is set jointly by initialization and architecture: the initia
    
[^329]: AdaLoop：面向音频语言模型的自适应深度潜在推理

    AdaLoop: Adaptive-Depth Latent Reasoning for Audio Language Models

    [https://arxiv.org/abs/2610.06949](https://arxiv.org/abs/2610.06949)

    AdaLoop是一种轻量级自适应深度潜在推理模块，通过学习到的停止机制为每个音频-问题对动态分配推理步数，仅增加不到3%的参数即可让音频语言模型在细粒度声学分析任务上的平均准确率提升2.9到3.8个百分点。

    

    大型音频语言模型能够回答关于语音、声音和音乐的问题，但在需要细粒度声学分析的任务上，其准确率会急剧下降。判断两位说话人中谁的音调更高，需要迭代式的信号级推理，而内容类问题则不需要这种推理。然而当前的模型在两类任务上花费相同的计算深度。我们提出了AdaLoop，这是一个轻量级的循环模块，能够学习给定的音频-问题对需要多少潜在细化步骤。一个共享的transformer块在问题的引导下对音频表示进行迭代，同时一个学习到的停止机制在表示就绪后退出循环。AdaLoop仅增加不到基础模型3%的参数，并且可以插入任何音频编码器-语言模型组合中，而无需修改其中任何组件。在三个架构不同的模型上，通过MMSU、MMAU-Pro和MMAR基准的评估，AdaLoop将平均准确率提升了2.9到3.8个百分点。

    arXiv:2610.06949v1 Announce Type: cross  Abstract: Large audio language models answer questions about speech, sound, and music, yet their accuracy drops sharply on tasks that need fine-grained acoustic analysis. Judging which of two speakers has the higher pitch demands iterative signal-level reasoning that a content question does not. Current models spend the same computational depth on both. We introduce AdaLoop, a lightweight recurrent module that learns how many latent refinement steps a given audio--question pair requires. A shared transformer block iterates over the audio representation, guided by the question, while a learned halting mechanism exits the loop once the representation is ready. AdaLoop adds fewer than 3\% of the base model's parameters and plugs into any audio encoder--language model pair without modifying either component. Evaluated on three architecturally distinct models across MMSU, MMAU-Pro, and MMAR, AdaLoop raises the average accuracy by 2.9 to 3.8 points, w
    
[^330]: FactorBench：一个面向投资组合的自动化因子挖掘基准测试

    FactorBench: A Portfolio-Aware Benchmark for Automated Factor Mining

    [https://arxiv.org/abs/2610.06947](https://arxiv.org/abs/2610.06947)

    FactorBench是一个面向投资组合的自动化因子挖掘基准测试，通过统一的评估框架比较九种挖掘方法产生的约五千个因子，从因子有效性、时间泛化性和超越风险与风格暴露的预测能力三个维度衡量金融信号的真实价值。

    

    因子挖掘旨在从金融数据中发现能够预测未来资产收益并指导投资组合构建的信号。自动化因子挖掘目前涵盖遗传规划、强化学习、生成模型以及大语言模型智能体等多种方法。然而，这些范式中的进展是否能产生更具可泛化性、更具独特性且更具经济价值的金融信号，目前仍不清楚。我们提出了FactorBench，这是一个面向投资组合的基准测试，比较了来自九种自动化挖掘方法在五个股票市场中挖掘出的约五千个因子。通过共享的数据和评估契约，该基准同时支持符号表达式和可执行的Python因子，将异构的发现算法连接到统一的信号组合与投资组合构建流程中。FactorBench从三个层面追踪挖掘系统的输出：因子有效性、时间泛化能力，以及在已测量的风险和风格暴露之外的预测能力。

    arXiv:2610.06947v1 Announce Type: cross  Abstract: Factor mining seeks to discover signals from financial data that predict future asset returns and guide portfolio construction. Automated factor mining now spans genetic programming, reinforcement learning, generative models, and large language model agents. Yet it remains unclear whether advances across these paradigms yield more generalizable, distinct, and economically useful financial signals. We introduce FactorBench, a portfolio-aware benchmark comparing roughly five thousand mined factors from nine automated mining methods across five equity markets. A shared data and evaluation contract supports both symbolic expressions and executable Python factors, connecting heterogeneous discovery algorithms to common signal combination and portfolio construction procedures. FactorBench traces the outputs of mining systems across three levels: factor validity, temporal generalization, and predictiveness beyond measured risk and style expos
    
[^331]: 学习记忆：为紧凑循环神经网络蒸馏记忆保持能力

    Learning to Remember: Distilling Memory Retention for Compact Recurrent Neural Networks

    [https://arxiv.org/abs/2610.06942](https://arxiv.org/abs/2610.06942)

    该论文提出了一种针对时间序列模型时序依赖和记忆保持特性而设计的记忆差异知识蒸馏框架，使紧凑的循环神经网络在资源受限环境中仍能保持高性能。

    

    深度学习模型，特别是循环神经网络及其变体（如长短期记忆网络），极大地推动了时间序列分析的发展。这些模型能够捕捉时间序列中复杂的序列模式，从而实现实时评估。然而，它们的高计算复杂度和庞大的模型规模给在资源受限环境（如可穿戴设备和边缘计算平台）中的部署带来了挑战。知识蒸馏（KD）提供了一种解决方案，通过将知识从一个大型复杂模型（教师模型）转移到一个更小、更高效的模型（学生模型），从而在降低计算需求的同时保持高性能。目前的知识蒸馏方法最初是为计算机视觉任务设计的，忽略了时间序列模型独特的时序依赖性和记忆保持特性。为了弥合这一差距，我们提出了一种名为记忆差异知识蒸馏的新型知识蒸馏框架。

    arXiv:2610.06942v1 Announce Type: new  Abstract: Deep learning models, particularly recurrent neural networks and their variants, such as long short-term memory, have significantly advanced time series analysis. These models capture complex, sequential patterns in time series, enabling real-time assessments. However, their high computational complexity and large model sizes pose challenges for deployment in resource-constrained environments, such as wearable devices and edge computing platforms. Knowledge Distillation (KD) offers a solution by transferring knowledge from a large, complex model (teacher) to a smaller, more efficient model (student), thereby retaining high performance while reducing computational demands. Current KD methods, originally designed for computer vision tasks, neglect the unique temporal dependencies and memory retention characteristics of time series models. To bridge this gap, we propose a novel KD framework termed Memory-Discrepancy Knowledge Distillation (
    
[^332]: QiYao-I：一个基于流形的不规则多变量时间序列预测基础模型

    QiYao-I: A Manifold Based Foundation Model for Irregular Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2610.06936](https://arxiv.org/abs/2610.06936)

    QiYao-I是一个基于流形的基础模型，通过采样条件时间流形注意力机制和频率感知的动态变量交互机制，有效解决了不规则采样、跨变量异步的多变量时间序列预测难题。

    

    不规则多变量时间序列预测是现实应用中一个具有挑战性却又十分重要的问题，其中观测数据通常不规则采样，且各变量之间存在异步记录的情况。现有的时间序列基础模型大多建立在规则采样序列之上，因此难以泛化到不规则的时间间隔和异步的跨变量依赖关系。为了应对这些挑战，我们提出了QiYao-I，一个基于流形的不规则多变量时间序列预测基础模型。具体而言，我们引入了一种新颖的采样条件时间流形注意力机制，将真实时间戳映射到一个可学习的时间流形特征空间，并将时间流形偏置注入注意力层，使模型能够同时捕捉不规则的时间间隔和局部采样结构。此外，我们提出了一种具有频率感知能力的动态变量交互机制。

    arXiv:2610.06936v1 Announce Type: new  Abstract: Irregular multivariate time series forecasting is a challenging yet important problem in real-world applications, where observations are often irregularly sampled and asynchronously recorded across variables. Existing time series foundation models are mostly built on regularly sampled sequences, making them difficult to generalize to irregular time intervals and asynchronous cross-variable dependencies. To address these challenges, we propose QiYao-I, a manifold based foundation model for irregular multivariate time series forecasting. Specifically, we introduce a novel sampling-conditioned temporal manifold attention mechanism that maps real timestamps into a learnable temporal manifold feature space and injects temporal manifold biases into attention layers, enabling the model to capture both irregular time intervals and local sampling structures. Further, we propose a dynamic variable interaction mechanism with frequency awareness. It
    
[^333]: 基于生成模型的递归熵风险强化学习的近最优样本复杂度

    Near-Optimal Sample Complexity for Recursive Entropic Risk Reinforcement Learning with a Generative Model

    [https://arxiv.org/abs/2610.06931](https://arxiv.org/abs/2610.06931)

    本文对基于模型的风险敏感 Q 值迭代（MB-RS-QVI）算法进行了精细分析，在生成模型假设下首次为递归熵风险强化学习建立了近最优的样本复杂度保证，其对有效视界的指数依赖性与现有下界相匹配，消除了理论差距。

    

    本文研究了在具有风险参数 β≠0 的递归熵风险偏好下，假设可以访问 MDP 的生成模型时，有限折扣马尔可夫决策过程（MDP）中价值学习和策略学习的样本复杂度。我们对基于模型的风险敏感 Q 值迭代（MB-RS-QVI）——一种先前工作中提出的插件式基于模型的方法——进行了精细分析，并针对学习最优 Q 值函数和 ε-最优策略分别推导出了 (ε,δ)-PAC 保证。与该设定下现有的最佳理论保证相比，我们的样本复杂度边界改进了对有效视界 1/(1-γ) 的指数依赖。特别地，在关于 |β|/(1-γ) 的指数依赖方面，以及在 S、A、ε 和 |β| 等参数方面（直至对数因子），我们的边界与现有下界相匹配。因此，我们的分析消除了……

    arXiv:2610.06931v1 Announce Type: new  Abstract: In this paper, we study the sample complexities of value and policy learning in finite discounted Markov decision processes (MDPs) under recursive entropic risk preferences with risk parameter \(\beta\neq 0\), assuming access to a generative model of the MDP. We provide a refined analysis of model-based risk-sensitive Q-value iteration (MB-RS-QVI), a plug-in model-based method introduced in prior work, and derive \((\varepsilon,\delta)\)-PAC guarantees for both learning the optimal \(Q\)-value function and an \(\varepsilon\)-optimal policy. Our bounds improve the exponential dependence on the effective horizon \(1/(1-\gamma)\) compared with the best existing guarantees for this setting. In particular, they match the existing lower bounds in their exponential dependence on \(|\beta|/(1-\gamma)\), as well as in \(S\), \(A\), \(\varepsilon\), and \(|\beta|\), up to logarithmic factors. Consequently, our analysis removes the exponential gap 
    
[^334]: 面向多元函数型数据异常检测的低秩与结构化稀疏张量分解

    Low-Rank and Structured Sparse Tensor Decomposition for Anomaly Detection in Multivariate Functional Data

    [https://arxiv.org/abs/2610.06930](https://arxiv.org/abs/2610.06930)

    提出两种无监督稀疏张量分解方法（ES-CP与FG-Lasso），通过低秩CP分解结合逐元素与纤维方向的稀疏惩罚，在保留多模态结构的同时检测多元函数型数据中的局部异常与时间纤维集中型异常。

    

    多元函数型数据广泛出现在许多现代制造系统中，其中多个传感器以密集采样方式记录过程轨迹。监测此类数据具有挑战性，因为正常（名义）变化在样本、传感器和时间之间具有很强的相关性，而故障可能表现为孤立的偏差，也可能表现为集中在少量特定传感器时间轨迹内的结构性偏移。我们提出两种能够保留这种多模态结构的无监督稀疏张量分解方法。逐元素稀疏CP分解（ES-CP）使用逐元素的 ℓ₁ 惩罚来识别局部异常，而纤维方向稀疏组Lasso CP分解（FG-Lasso）则结合逐元素与纤维方向的惩罚，既能检测局部偏差，也能检测集中在时间纤维内的异常。两种方法均通过低秩CP分解来表示名义过程行为，并采用交替（优化算法进行估计）……（原文摘要在此处截断）

    arXiv:2610.06930v1 Announce Type: cross  Abstract: Multivariate functional data arise in many modern manufacturing systems, where multiple sensors record densely sampled process trajectories. Monitoring such data is challenging because nominal variation is strongly correlated across samples, sensors, and time, while faults may appear either as isolated deviations or as structured departures concentrated within a limited number of sensor-specific temporal trajectories. We propose two unsupervised sparse tensor decomposition methods that preserve this multimode structure. Entrywise Sparse CP Decomposition (ES-CP) uses an entrywise \(\ell_1\) penalty to identify localized anomalies, whereas Fiberwise Sparse-Group Lasso CP Decomposition (FG-Lasso) combines entrywise and fiberwise penalties to detect both localized deviations and anomalies concentrated within temporal fibers. Both methods represent nominal process behavior through a low-rank CP decomposition and are estimated using alternat
    
[^335]: AttSVD：基于注意力引导SVD的提示自适应低秩KV缓存压缩

    AttSVD:Prompt-Adaptive Low-Rank KV Cache Compression via Attention-Guided SVD

    [https://arxiv.org/abs/2610.06927](https://arxiv.org/abs/2610.06927)

    提出AttSVD，一种基于每个提示自身注意力几何结构的可解释低秩KV缓存压缩方法，通过在线逐提示截断SVD仅保留注意力实际读取的方向，在保留全部token的同时按保留秩比例削减长上下文下的KV缓存内存，并提供累积式与流式两种解码时缓存策略及自适应压缩改进。

    

    自回归Transformer的键值（KV）缓存随上下文长度线性增长，并在长上下文场景下占据主要内存。大多数无需训练的补救方法会选择驱逐低重要性的token，这是沿序列轴上一种不可逆的选择。我们反其道而行之，保留每一个token，并沿“特征”轴以更廉价的方式存储。为此，我们提出AttSVD，一种新的“可解释”低秩压缩方法，其基向量源自每个提示自身的注意力几何结构：通过一种在线的、逐提示的截断SVD，仅保留注意力实际读取的方向，并按保留秩的比例削减每个注意力头的持久KV内存占用。我们提出了两种解码时缓存策略——累积式和流式——分别适用于短生成和长生成场景。此外，我们提出两项改进使压缩具有自适应性：逐矩阵能量规则可独立地为logit空间和注意力质量确定尺寸；注意力感知的基仅在注意力……（原文在此处截断）

    arXiv:2610.06927v1 Announce Type: cross  Abstract: The key-value (KV) cache of autoregressive transformers grows linearly with context length and dominates memory at long context. Most training-free remedies evict low-importance tokens, an irreversible choice along the sequence axis. We instead keep every token and store it more cheaply along the "feature" axis. We therefore propose AttSVD, a new "interpretable" low-rank compression whose basis is derived from each prompt's own attention geometry: an online, per-prompt truncated SVD that keeps only the directions attention actually reads, cutting persistent per-head KV memory in proportion to the retained rank. We propose two decode-time caching strategies, accumulating and streaming, for short and long generation regimes. Furthermore, we propose two refinements that make compression adaptive. A per-matrix energy rule sizes the logit space and the attention mass independently. An attention-aware basis truncates only in the spaces atten
    
[^336]: 扩展音乐标注模式：零样本预测还是少样本适应？

    Extending Music Annotation Schemas: Zero-Shot Prediction or Few-Shot Adaptation?

    [https://arxiv.org/abs/2610.06920](https://arxiv.org/abs/2610.06920)

    该论文提出了一个基于MGPHot流行音乐数据集的音乐标注模式扩展基准，实验表明即使在小标注预算下，监督式适应方法仍比音频-语言模型的零样本预测更有效。

    

    自动音乐标注通常是在固定标注模式的假设下进行的。但在实践中，商业音乐目录往往需要随着需求的变化而纳入新的音乐属性。鉴于专家音乐标注成本高昂，目前尚不清楚哪种方法最能有效地适应新属性并回填现有曲目的标注；音频-语言模型承诺提供零样本预测能力，但在多大的标注预算下，监督式适应会变得更具优势？我们提出了一个基于MGPHot流行音乐标注数据集的基准，用于在不同标注预算下模拟音乐标注模式的扩展。我们研究了三种方法：使用音频-语言模型进行零样本预测、从预训练表示中学习新属性，以及对基于现有标注训练的模型进行适应。我们的结果表明，即使标注预算很小，监督式适应仍然比零样本预测更有效。

    arXiv:2610.06920v1 Announce Type: cross  Abstract: Automatic music annotation is typically tackled under the assumption of a fixed annotation schema. In practice, commercial music catalogs often need to accommodate new musical attributes as needs evolve. Given that expert music annotation is expensive, it is not evident which methodological approach is most effective at accommodating new attributes and backfilling existing tracks; audio-language models promise zero-shot prediction, but at what annotation budget does supervised adaptation become more compelling?   We propose a benchmark based on the MGPHot popular music annotation dataset for simulating music schema extension across different annotation budgets. We investigate zero-shot prediction with audio-language models, learning new attributes from pretrained representations, and adapting models trained on existing annotations. Our results suggest that supervised adaptation is more effective than zero-shot prediction even with smal
    
[^337]: 对比学习中语义几何的锚点散度

    Anchor Divergence for Semantic Geometry in Contrastive Learning

    [https://arxiv.org/abs/2610.06919](https://arxiv.org/abs/2610.06919)

    本文提出“锚点散度”方法，通过建立锚点概率分布与Bregman几何之间的对应关系，使固定表示上的语义几何能够适配特定上下文，突破了余弦相似度单一固定几何的局限。

    

    本文研究语义上下文如何决定学习到的向量表示中的几何结构。相似度通常使用余弦相似度来衡量，这种方式提供了一种单一固定的几何。然而，语义相似度本质上依赖于上下文：两幅图像可能相似是因为它们描绘了同一物体、共享某种视觉风格，或与同一临床发现相关。我们证明对比表示天然地涵盖了一族几何结构，这些几何结构可以被专门化以匹配特定的语义结构。关键思想是利用对比学习、指数族分布和信息几何三者之间的相互作用，在“锚点”上的概率分布与表示空间上的Bregman几何之间建立对应关系。我们利用这种对应关系定义了“锚点散度”，这是一种在固定表示上指定特定于上下文的语义几何的方法。在这种对应关系下，模（原文摘要在此处截断）

    arXiv:2610.06919v1 Announce Type: new  Abstract: This paper concerns how semantic context determines geometry in learned vector representations. Similarity is typically measured using cosine similarity, which provides a single fixed geometry. Semantic similarity, however, is inherently context dependent: two images may be similar because they depict the same object, share a visual style, or are relevant to the same clinical finding. We show that contrastive representations naturally encompass a family of geometries that can be specialized to particular semantic structure. The key idea is to use an interplay between contrastive learning, exponential families, and information geometry to establish a correspondence between probability distributions over "anchors" and Bregman geometries on the representation space. We use this correspondence to define "Anchor Divergences", a method for specifying context-specific semantic geometries on fixed representations. Under this correspondence, mode
    
[^338]: 从不可靠轨迹中学习：对抗鲁棒的联邦Q学习

    Learning from Unreliable Trajectories: Adversarially-Robust Federated Q-Learning

    [https://arxiv.org/abs/2610.06918](https://arxiv.org/abs/2610.06918)

    该论文提出了对抗鲁棒的联邦Q学习算法Robust Async-Fed-Q，通过智能体端的方差缩减估计与服务器端的鲁棒聚合相结合，在部分智能体发送任意篡改数据的对抗环境下仍能保留诚实智能体协作带来的样本效率收益，并提供了高概率有限时间的理论保证。

    

    我们研究联邦强化学习问题，其中多个智能体与同一个马尔可夫决策过程进行交互，并通过中央服务器进行通信，以协作学习最优状态-动作价值函数。我们的目标是理解：当一部分智能体表现出对抗性行为并传输任意被篡改的信息时，协作所带来的样本效率收益是否仍能保持。为解决这一问题，我们提出了Robust Async-Fed-Q，一种基于周期（epoch）的联邦学习算法，它将智能体端对贝尔曼最优算子的方差缩减估计与服务器端的鲁棒聚合相结合。我们建立了高概率有限时间保证，表明所提出的方法在容忍对抗性破坏的同时，能够保留诚实智能体之间协作的统计增益。特别地，随着每个智能体收集的数据量增加，对抗性智能体的影响会不断减小。

    arXiv:2610.06918v1 Announce Type: new  Abstract: We study federated reinforcement learning in which multiple agents interact with a common Markov decision process and communicate through a central server to collaboratively learn the optimal state-action value function. Our goal is to understand whether the sample-efficiency benefits of collaboration can be retained when a fraction of the agents behave adversarially and transmit arbitrarily corrupted information. To address this problem, we introduce Robust Async-Fed-Q, an epoch-based federated learning algorithm that combines variance-reduced estimation of the Bellman optimality operator at the agents with robust aggregation at the server. We establish high-probability finite-time guarantees showing that the proposed method preserves the statistical gains of collaboration among the honest agents while tolerating adversarial corruption. In particular, the effect of the adversarial agents decreases as the amount of data collected by each
    
[^339]: 稀疏Lévy图上的非局部哈密顿动力学：谱分析与多峰采样

    Nonlocal Hamiltonian Dynamics on Sparse L\'evy Graphs: Spectral Analysis and Multimodal Sampling

    [https://arxiv.org/abs/2610.06904](https://arxiv.org/abs/2610.06904)

    该论文提出一种在稀疏Lévy图上运行的阻尼非局部哈密顿动力学方法，通过最近邻连接与采样的长程边实现多峰目标分布之间确定性且代价线性（与节点数和长程采样预算成正比）的概率质量输运，并以加权图拉普拉斯算子的谱刻画惯性与非局部连通性的相互作用、由谱隙确定最优渐近阻尼。

    

    我们开发了一种稀疏图方法，通过阻尼非局部哈密顿动力学将概率质量输运到多峰目标分布。该表述将对数平均迁移率与对称的Lévy型相互作用权重相结合，把演化的密度与一个边动量场耦合起来。由最近邻连接和采样的长程边构成的图，提供了空间上分离区域之间的直接质量交换。一旦图被构造出来，密度演化便是确定性的，且每次更新的代价与节点数和长程采样预算呈线性关系。围绕目标分布进行线性化，可得到一个由加权图拉普拉斯算子支配的阻尼振子。其谱刻画了非局部连通性与惯性之间的相互作用，其中Lévy指数 alpha 调节非局部连通性：谱隙决定了最优的渐近阻尼，而……（原文摘要在此处截断）

    arXiv:2610.06904v1 Announce Type: cross  Abstract: We develop a sparse graph method for transporting probability mass toward multimodal target distributions through damped nonlocal Hamiltonian dynamics. The formulation combines logarithmic-mean mobility with symmetric L\'evy-type interaction weights, coupling the evolving density to an edge momentum field. A graph constructed from nearest-neighbor connections and sampled long-range edges provides direct mass exchange between spatially separated regions. Once the graph is constructed, the density evolution is deterministic, and each update costs linear in the number of nodes and the long-range sampling budget. Linearization around the target distribution yields a damped oscillator governed by a weighted graph Laplacian. Its spectrum characterizes the interaction between nonlocal connectivity and inertia, with the L\'evy exponent alpha tuning the nonlocal connectivity: the spectral gap determines the optimal asymptotic damping, while the
    
[^340]: Transformer拒绝机制中的组件与维度稀疏性

    Component and Dimension Sparsity in Transformer Refusal Mechanisms

    [https://arxiv.org/abs/2610.06903](https://arxiv.org/abs/2610.06903)

    该研究通过对四个开源大语言模型的组件级干预分析，发现拒绝行为引导只需稀疏组件子集（占上游组件28%–48%）及其中约50%的残差流维度即可复现完整效果，揭示了拒绝机制在组件和维度两个层面上的稀疏性。

    

    激活引导通过干预大语言模型的内部激活来操纵其行为，但这些干预的机理基础仍知之甚少。我们将拒绝引导分解为跨四个开源权重模型的组件级干预，识别出稀疏的注意力与MLP组件子集，仅对这些子集进行引导就足以复现完整的行为效果。我们发现，拒绝方向集中于稀疏的组件机制中，这些组件仅占上游组件的28%–48%，却能保留88%–101%的引导有效性。在这些机制内部，有效引导进一步集中于约50%的残差流维度，保留85%–98%的组件机制基线效果，这与特权基结构相一致。因此，稀疏性在两个层面上发挥作用：哪些组件被引导，以及这些组件内部哪些维度承载信号。总之，这些发现表明……

    arXiv:2610.06903v1 Announce Type: cross  Abstract: Activation steering manipulates large language model behavior by intervening on internal activations, but the mechanistic basis of these interventions remains poorly understood. We decompose refusal steering into component-level interventions across four open-weight models, identifying the sparse subsets of attention and MLP components whose steering suffices to reproduce the full behavioral effect. We find that refusal directions concentrate in sparse component mechanisms comprising 28--48\% of upstream components, retaining 88--101\% of steering effectiveness. Within these mechanisms, effective steering further concentrates in approximately 50\% of residual stream dimensions, retaining 85--98\% of the component-mechanism baseline, consistent with a privileged basis structure. Sparsity thus operates at two levels: which components are steered, and which dimensions within those components carry the signal. Together these findings show 
    
[^341]: 记忆预测超额：用于衡量随机过程预测增益与记忆长度的一个概率量

    Memory Prediction Excess: A Probabilistic Quantity for Predictive Gain and Memory Length in Stochastic Processes

    [https://arxiv.org/abs/2610.06894](https://arxiv.org/abs/2610.06894)

    本文提出“记忆预测超额”（MPE）这一新的概率量，用以量化在离散时间有限状态随机过程中利用完整历史信息相对于仅用静态边际分布所带来的预测准确率平均提升，并证明了其非负性、上界条件及退化情形等基本性质。

    

    随机过程预测中的一个核心问题是：过去的信息能够在多大程度上提高正确预测下一个状态的概率。我们引入“记忆预测超额”这一概念来定量地回答这一问题。在离散时间有限状态过程中，MPE 度量的是使用完整观测历史相对于仅使用静态边际分布所获得的预测准确率的平均提升。它被定义为期望最优条件预测准确率与最优静态预测准确率之差。本文考察了它的基本性质：MPE 始终非负；它存在一个依赖于静态准确率的上界，当且仅当未来几乎必然是过去的确定性函数时该上界被达到；此外还刻画了 MPE 为零的退化情形。文中还引入了一个取值于单位区间的归一化版本。

    arXiv:2610.06894v1 Announce Type: cross  Abstract: A central question in the prediction of stochastic processes is the extent to which past information can improve the probability of correctly predicting the next state. We introduce the Memory Prediction Excess (MPE) to address this question quantitatively. The MPE measures the average improvement in prediction accuracy obtained by using the entire observed history relative to using only the static marginal distribution, in discrete-time finite-state processes. It is defined as the difference between the expected optimal conditional prediction accuracy and the optimal static prediction accuracy. Its basic properties are examined: the MPE is always non-negative; it admits an upper bound depending on the static accuracy, attained if and only if the future is almost surely a deterministic function of the past; and degenerate cases in which the MPE vanishes are characterized. A normalized version, taking values in the unit interval, is int
    
[^342]: 对齐中线性奖励的公理可满足性

    Axiom Satisfiability of Linear Rewards in Alignment

    [https://arxiv.org/abs/2610.06892](https://arxiv.org/abs/2610.06892)

    该论文通过引入逐候选者松弛量，提出一种计算“总松弛最小且满足公理边际η”的线性奖励的方法，在不对投票者和数据收集方式做任何假设的情况下，以被O(1)界定的松弛代价强制线性奖励满足帕累托最优与PMC等公理。

    

    从人类偏好数据中学习是使语言模型与人类价值观对齐的主流途径。在线性社会选择设定中，奖励是“提示-回答”对的固定特征表示的线性函数。Ge等人[2024]证明，通过最小化任何非递减凸损失（包括BTL）来拟合此类奖励，会违反帕累托最优（PO）与PMC公理；此外，一旦要求输出必须由线性模型诱导，任何仅读取多数关系的规则都无法满足PO。我们追问：无论如何强制满足这些公理的代价是什么？为此，我们将线性模型进行松弛，允许每个候选者拥有各自的松弛量。我们计算在满足公理（并带有边际η，即两个奖励值之间所需的最小差值）条件下总松弛量最小的松弛线性奖励。我们的解无需对投票者或比较数据的收集方式做任何假设即可满足这些公理。当η至多为O(1/m²…)（摘要在此处截断）时，我们将最优总松弛量界定为O(1)。

    arXiv:2610.06892v1 Announce Type: cross  Abstract: Learning from human preference data is the dominant route to aligning language models with human values. In linear social choice, where rewards are linear in a fixed feature representation of prompt-response pairs, Ge et al.[2024] show that fitting such a reward by minimizing any non-decreasing convex loss, including BTL, fails PO and PMC. Moreover, no rule that reads only the majority relation can satisfy PO once the output is required to be linearly induced. We ask what it costs to enforce these axioms anyway. To this end, we relax the linear model to allow per-candidate slack. We compute the relaxed linear reward with the smallest total slack that satisfies the axioms with a margin $\eta$, the minimum required difference between two reward values. Our solution satisfies the axioms under no assumptions about the voters or how comparisons were collected. We bound the optimal total slack by $O(1)$ when $\eta$ is at most $O(\frac{1}{m^2
    
[^343]: 面向制造业的事件驱动机器学习流水线编排：一项AWS行业实践经验

    Event-Driven ML Pipeline Orchestration for Manufacturing: An AWS Industry Experience

    [https://arxiv.org/abs/2610.06890](https://arxiv.org/abs/2610.06890)

    该论文分享了三年来在汽车制造业运营事件驱动云基础设施的行业实践经验，通过ECS、SQS、Lambda等AWS服务编排跨工厂的GPU加速再训练流水线，在4万余个生产训练任务中实现了相比常开GPU基础设施72-78%的成本降低。

    

    我们呈现了一份行业实践报告，介绍了三年来在汽车制造业中运营事件驱动云基础设施以支持持续机器学习训练的经验。该系统编排了跨多个工厂的产品专用模型对的GPU加速训练，即一个物理预测模型和一个强化学习控制策略，协调由制造事件触发的长时间运行的GPU工作负载。该架构结合了Amazon ECS与EC2 GPU容量提供程序、基于SQS的消息传递（配备死信队列），以及一个执行准入控制并强制集群并发限制的Lambda调度器。运行在ECS Fargate上的Conductor编排器按周调度启动具有依赖感知能力的再训练链。整个基础设施以模块化Terraform实现代码化，并采用多账户隔离。基于40000多个生产训练任务，我们报告了相较于常开GPU基础设施72-78%的成本降低。一个离散

    arXiv:2610.06890v1 Announce Type: new  Abstract: We present an industry experience report on three years of operating an event-driven cloud infrastructure for continuous machine learning training in automotive manufacturing. Our system orchestrates GPU-accelerated training of product-specialized model pairs, a physics prediction model and a reinforcement-learning control policy, across multiple plants, coordinating long-running GPU workloads triggered by manufacturing events. The architecture combines Amazon ECS with EC2 GPU capacity providers, SQS-based messaging with dead-letter queues, and an admission-controlled Lambda dispatcher that enforces cluster concurrency limits. A Conductor orchestrator on ECS Fargate initiates dependency-aware retraining chains on a weekly schedule. The entire infrastructure is codified in modular Terraform with multi-account separation. From 40000+ production training jobs we report a 72-78% cost reduction versus always-on GPU infrastructure. A discrete-
    
[^344]: 零样本可视化：基于用户提示轴的文本语料库探索

    Zero-Shot Visualization: Exploring Text Corpora with User-Prompted Axes

    [https://arxiv.org/abs/2610.06889](https://arxiv.org/abs/2610.06889)

    该论文提出了零样本可视化（ZSV）任务，允许用户通过自然语言指定概念轴来交互式探索文本语料库，并通过基准测试发现基于下一个词元概率的评分方法在语义忠实性、评分保真度和计算成本方面具有优势。

    

    我们研究了大语言模型（LLM）在文本语料库可视化探索中的应用。我们提出了零样本可视化（ZSV）这一任务，即用户用自然语言指定概念，然后将文档映射到相应的概念轴上进行可视化。构建一个具有实用价值的ZSV系统并非易事，因为它需要在特征函数、高效实现的权衡以及影响可视化质量的预处理/后处理决策的交汇处做出选择。为此，我们建立了一个基准，比较了在此设置下涵盖嵌入相似度、直接语义判断和条件似然估计等多种方法。我们在多个数据集和用例上，从语义忠实性、评分保真度和计算成本等方面评估了不同评分方法和设计选择的特性。我们的结果表明，基于下一个词元概率的评分方法提供了……

    arXiv:2610.06889v1 Announce Type: cross  Abstract: We study the application of large language models (LLMs) to the visual exploration of textual corpora. We introduce zero-shot visualization (ZSV), a task in which users specify concepts in natural language and documents are mapped onto the corresponding concept axes for visualization. Building a ZSV system of practical value is non-trivial, as it requires choices at the intersection of feature functions, efficient implementation tradeoffs, and pre/post-processing decisions affecting visualization quality. To that end, we establish a benchmark that compares methods spanning embedding similarity, direct semantic judgments, and conditional likelihood estimation in this setting. Across multiple datasets and use cases we evaluate the properties of different scoring methods and design choices in terms of semantic faithfulness, score fidelity, and computational cost. Our results identify that scoring based on next-token probabilities offers t
    
[^345]: 学习何时细化：面向预算约束神经算子偏微分方程求解器的长时程强化学习

    Learning When to Refine: Long-Horizon Reinforcement Learning for Budgeted Neural-Operator PDE Solvers

    [https://arxiv.org/abs/2610.06883](https://arxiv.org/abs/2610.06883)

    该论文提出面向预算约束神经算子PDE求解的长时程强化学习方法RV-PI，通过实际滚动验证在有限修正预算下学习何时何地施加局部细化，仅在留出轨迹误差改善时接受策略更新。

    

    神经算子为时间依赖的偏微分方程提供了快速代理模型，但自回归部署带来了细化分配问题：预测误差随空间和时间变化，而沿轨迹只能进行有限次数的局部修正。我们将该问题形式化为预算约束的自适应神经算子求解。全局傅里叶神经算子推进整个场，局部算子提出分块残差修正，集合感知的选择器决定在哪里进行细化，宏策略则决定何时以及花费多少剩余的细化预算。我们提出了滚动验证策略改进（RV-PI），它通过学习到的偏微分方程求解器的实际延续滚动来评估可行的修正次数，将长时程优势转化为保守的策略目标，并且仅当留出轨迹误差得到改善时才接受更新。在具有32次干预预算的浅水方程基准测试中，RV-P……（摘要在此处被截断）

    arXiv:2610.06883v1 Announce Type: cross  Abstract: Neural operators provide fast surrogates for time-dependent PDEs, but autoregressive deployment creates a refinement-allocation problem: prediction errors vary over space and time, while only a finite number of local corrections can be committed along a trajectory. We formulate this as budgeted adaptive neural-operator solving. A global Fourier neural operator advances the full field, a local operator proposes patch-wise residual corrections, and a set-aware selector chooses where to refine. A macro policy decides when and how much of the remaining refinement budget to spend. We introduce rollout-verified policy improvement (RV-PI), which evaluates feasible refinement counts through actual continuation rollouts of the learned PDE solver, converts long-horizon advantages into conservative policy targets, and accepts an update only when held-out trajectory error improves. On the shallow-water benchmark with a 32-intervention budget, RV-P
    
[^346]: 用于建筑热负荷短期预测的混合预测模型比较综述

    Comparative review of hybrid forecasting models for short-term prediction of building thermal load

    [https://arxiv.org/abs/2610.06881](https://arxiv.org/abs/2610.06881)

    本文综述并比较了13种用于建筑热负荷短期预测的混合模型，发现EMD-LSTM-Markov模型的预测精度最高。

    

    本文对不同混合模型在建筑热需求短期预测中的表现进行了比较综述。特别地，该评估针对的是与其他最先进技术相结合的增强型数据驱动模型之间的比较。第一步，分析了文献中已报道的现有技术，结论是元启发式算法或数据驱动模型被用于识别基础模型的参数。定性评估包括每种方法的输入和输出特征、主要优点和缺点。第二步，利用苏格兰家庭历史热需求数据集以及历史天气预报，进一步评估了现有混合方法的性能。在对13种混合方法的评估中，经验模态分解-长短期记忆-马尔可夫（EMD-LSTM-Markov）模型能够以最高的精度进行预测……

    arXiv:2610.06881v1 Announce Type: cross  Abstract: In this paper, a comparative review of different hybrid models for short-term forecasting of building thermal demand is carried out. Particularly, the assessment tackles the comparison of data-driven models enhanced with other state-of-the-art techniques. At the first step, the existing techniques reported in the literature are analysed. It is concluded that Metaheuristics or a data-driven model are used to identify the parameters of the basic model. The qualitative evaluation includes for each method the input and output features, main advantages and drawbacks. At the second step, an existing dataset of historical thermal demand from Scottish households, as well as historical weather forecasts are utilized to assess additionally the performance of existing hybrid methods. From the assessment of 13 hybrid methods, the Empirical Modal Decomposition - long short-term memory - Markov (EMD-LSTM-Markov) model can predict with the highest ac
    
[^347]: 基于中智学集成分类的不确定性感知轴承故障检测：来自实验室与变速工业基准的证据

    Neutrosophic Ensemble Classification for Uncertainty-Aware Bearing Fault Detection: Evidence from Laboratory and Variable-Speed Industrial Benchmarks

    [https://arxiv.org/abs/2610.06880](https://arxiv.org/abs/2610.06880)

    本文提出将中智学四指标分解（最高类证据、最佳竞争者证据、预测熵与决策分歧）应用于机器学习集成分类器，以区分轴承故障检测中自信的错误与真正模糊的预测，实现不确定性感知的故障诊断，并在实验室与变速工业基准上验证了其有效性。

    

    arXiv:2610.06880v1 发布类型：cross。摘要：用于轴承故障检测的机器学习分类器输出标量置信度分数，将自信的错误与真正模糊的预测混为一谈，而传统的真/假二元组（F = 1 - T）在构造上存在代数冗余。我们通过精炼的中智学分解，将随机森林 + XGBoost + 逻辑回归集成分类器操作化为四个指标——T-hat（最高类别证据）、F-hat（最佳竞争者证据）、预测熵 I1-hat 和决策分歧 I2-hat——并在两个轴承基准数据集（CWRU 和 JNU，600-1000 rpm）上采用留一工况协议进行评估。在 CWRU 数据集上，在纠正了文件与类别之间的映射错误后，该集成分类器在四个留出负载中的三个上达到 100.00% 的准确率（第四个为 92.27%），导致可用于不确定性分析的错误样本过少。在 JNU 数据集上，留出 1000 rpm 工况后，准确率骤降至 40.64%，低于多数类基线；逻辑回归……

    arXiv:2610.06880v1 Announce Type: cross  Abstract: Machine learning classifiers for bearing fault detection produce scalar confidence scores that conflate confident errors with genuinely ambiguous predictions, and the conventional truth/falsity pair (F = 1 - T) is algebraically redundant by construction. We operationalize a refined neutrosophic decomposition of a Random Forest + XGBoost + Logistic Regression ensemble into four indicators -- T-hat (top-class evidence), F-hat (best-competitor evidence), predictive entropy I1-hat, and decision disagreement I2-hat -- evaluated on two bearing benchmarks (CWRU and JNU, 600-1000 rpm) under a leave-one-condition-out protocol. On CWRU, after correcting a file-to-class mapping error, the ensemble reaches 100.00 percent accuracy on three of four held-out loads (92.27 percent on the fourth), leaving too few errors for uncertainty analysis. On JNU, holding out 1000 rpm, accuracy collapses to 40.64 percent, below a majority-class baseline; Logistic 
    
[^348]: 世界模型何时能够恢复物理定律？

    When Can World Models Recover Physical Laws?

    [https://arxiv.org/abs/2610.06877](https://arxiv.org/abs/2610.06877)

    本文指出仅凭准确预测不能证明世界模型恢复了物理定律，并给出定律可恢复的充要条件——任意两条不同定律必须在实验上可区分——同时用率-失真下界、有限响应码本、明确解码预算和极小极大采样复杂度刻画了恢复的信息论极限与稳定性。

    

    准确的预测并不能证明一个世界模型已经恢复了物理定律：在相同的观测协议下，不同的动力学可能生成完全相同的记录。本文在明确的实验目录、传感器不确定性和采集预算的约束下，于一个固定的物理域上形式化地定义了定律恢复问题。一个率-失真逆定理将描述定律所需的信息与实验装置所能揭示的信息分离开来，其构造性对应结果则给出了一个有限的响应码本和明确的解码预算。在紧致的世界类上，一致恢复当且仅当每一对不同的定律在实验上可区分时才可能实现；等价地，即该装置能够恢复每个有限定律源的全部熵。文中还引入逆响应模量来刻画稳定性。对于d维状态-动作域上的Lipschitz场，重置后的带噪全状态读取需要Θ(ε^...（摘要在此处截断）的极小极大预算。

    arXiv:2610.06877v1 Announce Type: cross  Abstract: Accurate prediction does not establish that a world model has recovered a physical law: distinct dynamics can generate identical records under the same observation protocol. We formulate law recovery on a fixed physical domain under an explicit catalog of experiments, sensor uncertainty, and an acquisition budget. A rate--distortion converse separates the information needed to describe a law from the information the apparatus can reveal. Its constructive counterpart gives a finite response codebook and an explicit decoding budget. On compact world classes, uniform recovery is possible exactly when every pair of different laws is experimentally distinguishable; equivalently, the apparatus can recover all the entropy of every finite law source. An inverse response modulus quantifies stability. For Lipschitz fields on a $d$-dimensional state--action domain, noisy full-state readouts after resets require minimax budget $\Theta(\varepsilon^
    
[^349]: 面向四旋翼飞行控制的统计湍流与高保真扰动场

    Statistical Turbulence and High-Fidelity Disturbance Fields for Quadrotor Flight Control

    [https://arxiv.org/abs/2610.06874](https://arxiv.org/abs/2610.06874)

    本文首次系统量化了风场保真度（而非风速大小）对强化学习四旋翼控制器鲁棒性的影响，通过五种扰动保真度级别的完整交叉训练-测试评估，并配合经过验证的扰动数据，揭示了训练扰动场保真度与策略真实环境表现之间的关系。

    

    强化学习四旋翼控制器通常在简化的风场模型下训练，然而风场保真度（而非风场强度）对策略鲁棒性的影响至今仍未被量化。本文比较了五种扰动保真度级别——从无风飞行、离散1余弦阵风，到统计湍流、合成相干结构，再到大气边界层的大涡模拟（LES）场——对近端策略优化（PPO）智能体进行了完整的交叉保真度训练-测试评估，并以级联PID控制器和几何SE(3)控制器作为无需训练的参考基准，在0-12 m/s的风速范围内开展实验。在进行任何控制器比较之前，所有扰动数据均先经过验证：每个合成扰动生成器都与其解析参考或认证标准参考进行了定量核对，大涡模拟场也与所施加的对数律进行了比对。在处于悬停状态的竞赛级四旋翼上，训练-测试（原文在此处截断）……

    arXiv:2610.06874v1 Announce Type: cross  Abstract: Reinforcement-learning quadrotor controllers are usually trained under simplified wind models, yet the impact of wind-field fidelity, as opposed to magnitude, on policy robustness remains unquantified. This paper compares five disturbance-fidelity levels, from wind-free flight and discrete 1-cosine gusts through statistical turbulence and synthetic coherent structures to large-eddy-simulation fields of the atmospheric boundary layer, in a full cross-fidelity train test evaluation of proximal policy optimization (PPO) agents, with cascaded PID and geometric SE(3) controllers as training-free references, over a 0-12 m/s wind sweep. Before any controller comparison is made, all disturbance data are validated: every synthetic generator is checked quantitatively against its analytical or certification-standard reference, and the large-eddy-simulation fields against the imposed log law. On a racing-class quadrotor in hover, the train test ma
    
[^350]: 何时外部指导能帮助大语言模型推理？指导增强GRPO的偏差-方差理论

    When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO

    [https://arxiv.org/abs/2610.06861](https://arxiv.org/abs/2610.06861)

    该论文提出GA-GRPO统一理论框架，将外部指导建模为随机指导算子，证明其引入的偏差可由全变差指导散度δ_G界定，从而为外部指导何时以及如何帮助LLM推理提供收敛速率、偏差界和最优加权规则的理论基础。

    

    带可验证奖励的强化学习（RLVR）已成为引出大语言模型多步推理能力的主流范式，而近期涌现的一系列方法（LUFFY、ExPO、PAPO、TAPO）进一步利用“外部指导”——专家轨迹、自我解释或检索到的思维模式——来增强强化学习。尽管这些方法都报告了实证收益，但没有一种方法给出收敛速率、偏差界或指导信号的最优加权规则。我们通过“指导增强GRPO”（GA-GRPO）来填补这一空白：这是一个统一的理论框架，将外部指导视为一个重写问题分布的随机指导算子G，并把由此得到的策略梯度估计器分析为一个有偏的在策略估计器，其偏差由指导增强采样分布与策略自身分布之间的全变差指导散度δ_G所界定。该框架涵盖了原始GRPO等方法，

    arXiv:2610.06861v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has become the dominant paradigm for eliciting multi-step reasoning in large language models, and a recent wave of methods (LUFFY, ExPO, PAPO, TAPO) further augments RL with \emph{external guidance} - expert traces, self-explanations, or retrieved thought patterns. Although each method reports empirical gains, none provides convergence rates, bias bounds, or an optimal weighting rule for the guidance signal. We close this gap with \emph{Guidance-Augmented GRPO} (GA-GRPO), a unified theoretical framework that casts external guidance as a stochastic guidance operator G re-writing the question distribution, and analyses the resulting policy-gradient estimator as a biased on-policy estimator whose bias is bounded by the total-variation guidance divergence delta\_G between the guidance-augmented sampling distribution and the policy's own distribution. The framework subsumes vanilla GRPO,
    
[^351]: Trajectools演示：迈向移动数据分析的无代码解决方案

    Trajectools Demo: Towards No-Code Solutions for Movement Data Analytics

    [https://arxiv.org/abs/2610.06858](https://arxiv.org/abs/2610.06858)

    本文基于开源Python库MovingPandas和开源GIS软件QGIS，提出了无代码移动数据分析工具Trajectools插件的概念框架并完成了初步实现，使非程序员用户也能进行移动数据分析。

    

    这篇演示论文介绍了一个面向移动数据分析的新型无代码解决方案的概念基础及其实施的初步步骤。该方案基于开源Python库MovingPandas和开源地理信息系统QGIS构建。由此产生的Trajectools插件已作为开源项目发布于 https://github.com/movingpandas/qgis-processing-trajectory。

    arXiv:2610.06858v1 Announce Type: cross  Abstract: This demo paper presents the conceptual foundations and the first steps towards implementation of a novel no-code solution for movement data analytics based on the open-source Python library MovingPandas and the open-source geographic information system QGIS. The resulting Trajectools plugin is available open-source at https://github.com/movingpandas/qgis-processing-trajectory.
    
[^352]: TEMPEST：通过角度边距学习实现可扩展驾驶员识别的时间嵌入

    TEMPEST: Temporal Embeddings for Scalable Driver Identification via Angular Margin Learning

    [https://arxiv.org/abs/2610.06855](https://arxiv.org/abs/2610.06855)

    TEMPEST提出了一种基于ArcFace角度边距损失训练的时间卷积网络嵌入模型，将60秒多模态驾驶数据映射为96维嵌入，在45名驾驶员数据集上达到91.71%的Rank-1准确率，且随驾驶员规模扩大仅出现轻微性能下降，实现了无需重新训练的可扩展驾驶员识别。

    

    可扩展的驾驶员识别需要嵌入模型在车队规模扩大时仍能保持判别性能，然而现有的三元组损失方法随着驾驶员池规模的增加性能迅速退化，并且在严格的时间评估下容易过拟合于特定会话的模式。我们提出了TEMPEST，这是一种使用加性角度边距损失训练的时间卷积网络嵌入模型，在归一化角度空间中强制实现全局类别级别的分离。TEMPEST将60秒的多模态驾驶窗口映射为紧凑的96维嵌入，支持真正的动态注册，无需任何重新训练或分类器重新拟合。在包含45名驾驶员的数据集上进行的严格时间评估中，TEMPEST实现了91.71%的Rank-1准确率，比最佳经典模型高出17.9个百分点，比最强的三元组损失基线高出58.4个百分点。当受试者池从10名驾驶员扩展到45名驾驶员时，TEMPEST仅下降4.3个百分点，相比之下其他方法下降了22个百分点。

    arXiv:2610.06855v1 Announce Type: new  Abstract: Scalable driver identification requires embedding models that maintain discriminative performance as fleet size grows, yet existing triplet-loss formulations degrade rapidly with driver pool size and overfit to session-specific patterns under rigorous temporal evaluation. We introduce TEMPEST, a Temporal Convolutional Network embedding model trained with an additive angular margin (ArcFace) loss that enforces global class-level separation in a normalized angular space. TEMPEST maps 60-second multimodal driving windows to compact 96-dimensional embeddings, supporting truly dynamic enrollment without any retraining or classifier refitting. Under rigorous temporal evaluation on a 45-driver dataset, TEMPEST achieves 91.71% Rank-1 accuracy, outperforming the best classical model by 17.9 pp and the strongest triplet-loss baseline by 58.4 pp. TEMPEST degrades by only 4.3 pp when growing the subject pool from 10 to 45 drivers, compared to 22 pp 
    
[^353]: 超越边缘分布：面向多表合成数据生成的多维评估框架

    Beyond Marginals: A Multi-Dimensional Evaluation Framework for Multi-Table Synthetic Data Generation

    [https://arxiv.org/abs/2610.06854](https://arxiv.org/abs/2610.06854)

    本文提出了 SynEval，一个与生成器无关的多表合成数据库六维评估框架，联合评估单列保真度、多变量结构、跨表完整性、机器学习实用性、隐私保护和边界情况鲁棒性，并提供统一的加权质量分数及按表、按维度的下钻分析。

    

    合成数据生成对于隐私合规、机器学习数据增强和软件测试至关重要。虽然单表评估已经相当成熟，但多表（关系型）合成数据——这是企业中占主导地位的应用场景——却缺乏统一的评估框架。现有方法孤立地评估各列的边缘分布，忽视了联合分布、跨表结构完整性、下游实用性以及面向生产环境的边界情况。我们提出了 SynEval，一个面向多表合成数据库的六维评估框架。SynEval 联合评估单列保真度、多变量结构保持（包括一种新颖的条件分布检查）、跨表完整性、机器学习实用性、隐私保护以及边界情况鲁棒性。该框架生成统一的加权质量分数，并支持按表和按维度的下钻分析，且与生成器无关，可应用于任意真实与合成数据对。

    arXiv:2610.06854v1 Announce Type: cross  Abstract: Synthetic data generation is critical for privacy compliance, machine learning augmentation, and software testing. While single-table evaluation is well established, multi-table (relational) synthesis, the dominant enterprise use case, lacks a unified evaluation framework. Existing approaches assess marginal column distributions in isolation, overlooking joint distributions, cross-table structural integrity, downstream utility, and production-readiness edge cases. We present SynEval, a six-dimensional evaluation framework for multi-table synthetic databases. SynEval jointly assesses per-column fidelity, multivariate structure preservation including a novel conditional distribution check, cross-table integrity, ML utility, privacy protection, and edge-case robustness. The framework produces a unified weighted quality score with per-table and per-dimension drill-down, and is generator-agnostic, operating on any pair of real and synthetic
    
[^354]: 一种针对学习型随机AI模拟器的响应理论探针——在Lorenz-63上的测试

    A Response Theory Probe for Learned Stochastic AI Simulators, Tested on Lorenz-63

    [https://arxiv.org/abs/2610.06798](https://arxiv.org/abs/2610.06798)

    本文提出一种基于线性响应理论的校准检验探针，用于评估学习型随机AI模拟器对外部强迫的响应是否正确，并在随机Lorenz-63模型上对SINDy、MLP、储备池计算机、神经ODE和神经SDE等多种模拟器进行了模态分辨的基准测试。

    

    混沌与随机系统的机器学习模拟器通常通过预测技巧和长期统计特性来验证。但这两者都无法证明模拟器对外部强迫的响应是正确的，而这一性质正是预估与归因研究所依赖的关键。线性响应理论使这一检验成为可能：强迫响应可通过广义涨落-耗散关系由未扰动系统的相关性推导得出，并可按Koopman生成元的随机Ruelle-Pollicott共振进行分解。基于Koopmanism Response框架，我们将其转化为一种经过校准的、具有模态分辨能力的学习型替代模型检验方法：每个替代模型的滚动模拟对每项检验给出通过或失败的结果，其失败率与真实系统独立实现的失败率进行比较。在三变量玩具模型——随机Lorenz-63上，我们评估了SINDy、多层感知机（MLP）、储备池计算机、神经常微分方程（neural ODE）以及带学习扩散项的神经随机微分方程（neural SDE），滚动步长最高达80步

    arXiv:2610.06798v2 Announce Type: replace-cross  Abstract: Machine-learning emulators of chaotic and stochastic systems are usually validated on forecast skill and long-run statistics. Neither certifies that an emulator responds correctly to forcing, the property that projection and attribution studies rely on. Linear response theory makes this testable: the forced response follows from unperturbed correlations through a generalized fluctuation-dissipation relation, and decomposes over the stochastic Ruelle-Pollicott resonances of the Koopman generator. Building on the Koopmanism Response framework, we turn this into a calibrated, mode-resolved test for learned surrogates: each surrogate rollout passes or fails each check, and failure rates are compared with those of independent realizations of the true system. On stochastic Lorenz-63, a three-variable toy model, we evaluate SINDy, an MLP, a reservoir computer, a neural ODE and a neural SDE with learned diffusion, over up to 80 rollout
    
[^355]: 学习即游走：基于随机游走的跨图与跨任务学习

    To Learn is to Wander: Learning Across Graphs and Tasks with Random Walks

    [https://arxiv.org/abs/2610.06694](https://arxiv.org/abs/2610.06694)

    提出基于随机游走统一接口的图基础模型Wander，将图学习形式化为部分观测图的补全，使单个预训练模型能够跨图类型、特征、关系模式与预测任务通用迁移，并具备逼近贝叶斯最优预测器的理论保证。

    

    图基础模型旨在跨图、特征空间、关系模式和预测任务进行迁移，然而现有方法通常只能在特定的图模态或任务范围内泛化。我们提出Wander，一个旨在以单个预训练检查点跨上述各类设置运行的图基础模型。遵循先验预测视角，我们将图学习形式化为对部分观测图的补全任务。我们通过基于随机游走的统一接口实现这一任务通用的观点，使同一个模型能够处理具有不同特征、标签和关系模式的同构图与多关系图。Wander可以在推理时扩展其结构上下文而无需更改已学习的参数，并且在适当的假设下，能够在有界连通图上通用逼近相应的贝叶斯最优预测器。在实证方面，单个预训练检查点……

    arXiv:2610.06694v2 Announce Type: replace  Abstract: Graph foundation models aim to transfer across graphs, feature spaces, relational schemas, and prediction tasks, yet existing approaches typically generalize only within particular graph modalities or tasks. We propose Wander, a graph foundation model designed to operate across these settings within a single pretrained checkpoint. Following the prior-predictive perspective, we formulate graph learning as completion of a partially observed graph. We realize this task-general view through a common interface based on random walks, allowing the same model to operate across homogeneous and multi-relational graphs with varying features, labels, and relational schemas. Wander can increase its structural context at inference time without changing its learned parameters and, under suitable assumptions, universally approximates the corresponding Bayes-optimal predictor on bounded connected graphs. Empirically, a single pretrained checkpoint ac
    
[^356]: 通过分层程序性理解改进主动式AI助手

    Improving Proactive AI Assistance with Hierarchical Procedural Understanding

    [https://arxiv.org/abs/2610.06505](https://arxiv.org/abs/2610.06505)

    本文提出了ProactiveCoach数据集套件，利用分层程序性理解使主动式AI助手能够根据任务进度和用户需求提供自适应粒度的指导。

    

    主动式AI助手持续观察用户的活动，并决定是提供新的指导还是保持沉默。它们应当为任务提供适当的指导，根据任务进度确定何时提供下一次指导，并根据用户的专业水平和需求调整指导级别。支持这些能力需要能够反映程序性结构、并捕捉指导应如何适应任务进度和用户需求的训练与评估数据。然而，现有数据集要么专注于基于检测的主动式理解，要么仅提供固定粒度的程序性指导。固定粒度的指导关于细粒度进度和更广泛程序性上下文的信息有限，导致难以确定任务的完成情况并调整指导的粒度。为了解决这些局限性，我们引入了ProactiveCoach套件，包括用于训练的ProactiveCoach-Instruct、用于（评估的）ProactiveCoach……（摘要内容在此处不完整）

    arXiv:2610.06505v2 Announce Type: replace  Abstract: Proactive AI assistants continuously observe a user's activity and decide whether to provide new guidance or remain silent. They should provide appropriate guidance for the task, determine when to provide the next guidance based on task progress, and adjust the guidance level to the user's expertise and needs. Supporting these capabilities requires training and evaluation data that reflect procedural structure and capture how guidance should adapt to task progress and user needs. However, existing datasets either focus on detection-based proactive understanding or provide procedural guidance at a fixed granularity. Fixed-granularity guidance provides limited information about fine-grained progress and broader procedural context, making it difficult to determine completion and adapt guidance granularity. To address these limitations, we introduce the ProactiveCoach suite, comprising ProactiveCoach-Instruct for training, ProactiveCoach
    
[^357]: PDE训练输入的有损压缩：场重构误差无法对已训练算子的代价进行排序

    Lossy Compression of PDE Training Inputs: Field Reconstruction Error Does Not Order the Cost to a Trained Operator

    [https://arxiv.org/abs/2610.06095](https://arxiv.org/abs/2610.06095)

    本文证明压缩PDE训练输入时的场重构误差无法预测所训练算子的精度损失，因为解算子对输入扰动的衰减程度在不同PDE族之间相差两个数量级以上，导致重构误差指标在104个代价比较中反转了36个。

    

    算子学习基准数据集以全精度存储，规模已增长到太字节级别。率失真理论说明了存储场需要多少比特，而实践者需要知道在压缩数据上训练的算子会有多准确。我们证明前者并不能决定后者，并测量了其中原因：我们在压缩输入场的同时，保持目标数据和测试输入为全精度。解算子会衰减其输入的扰动。将压缩后的场输入到一个已在全精度下训练好的代理模型中，可以测量该代理模型传输了多少扰动。这一比例与底层方程的平滑行为一致，并且在不同PDE族之间跨越两个多数量级。场重构误差是在衰减发生之前计算的，因此无法察觉这种衰减。对于用均方误差训练的算子，场重构误差在104个跨数据集的代价比较中反转了36个。

    arXiv:2610.06095v2 Announce Type: replace  Abstract: Operator-learning benchmarks are stored at full precision and have grown to terabyte scale. Rate-distortion theory says how many bits the stored field needs, while a practitioner needs to know how accurate an operator trained on the compressed data will be. We show that the first does not determine the second, and measure why, compressing the input fields while targets and test inputs stay at full precision. A solution operator attenuates a perturbation of its input. Pushing a compressed field through a surrogate already trained at full precision measures how much of the perturbation that surrogate transmits. The fraction is consistent with the smoothing behaviour of the underlying equation, and it spans more than two orders of magnitude across PDE families. Field reconstruction error is computed before the attenuation and cannot see it. For operators trained with mean squared error it inverts 36 of 104 cost comparisons across datase
    
[^358]: 基于真实信号学习到的共享结构的量子数据加载

    Quantum data loading from the learned shared structure of real signals

    [https://arxiv.org/abs/2610.06076](https://arxiv.org/abs/2610.06076)

    该论文提出一种量子原生数据加载器，通过一次性学习真实数据集共享的低维结构，以固定电路和少量参数高效加载所有信号，且所需训练子集大小不随数据规模增长而增加。

    

    将经典数据制备成量子态的成本可能超过其所能服务的计算本身；大多数加载器需要为每个输入量身定制电路。本文表明，真实数据集中的信号共享一种可以被一次性学习并重复利用的结构。我们的量子原生加载器学习数据集的低维描述，并用一个由少量数字设定的固定电路来制备每一个信号。在五个公开数据集的七种视图上，它在相同的门成本下达到了最强结构化加载器的目标，而每个信号所需的数字数量减少数倍。这些数字可以从随机子集中推断出来：在一项预注册的盲法重复实验中，随着信号规模增长十六倍，达到全信号精度百分之十以内所需的子集大小在预设误差范围内保持恒定，而结构化加载器所需的样本却不断增加。该加载器会拒绝其无法表示的内容，因此覆盖的案例少于该基线方法，且不支持心电图数据。

    arXiv:2610.06076v2 Announce Type: replace-cross  Abstract: Preparing quantum states from classical data can cost more than the computation they serve; most loaders tailor a circuit to each input. Here we show that the signals of a real dataset share structure that can be learned once and reused. Our quantum-native loader learns a low-dimensional description of a dataset and prepares every signal with one fixed circuit set by a few numbers. Across seven views of five public datasets it meets the targets of the strongest structured loader at equal gate cost with several times fewer numbers per signal. These numbers can be inferred from a random subset: in a preregistered blind replication the subset needed to come within ten per cent of full-signal accuracy stayed constant within a prespecified margin as signals grew sixteenfold, whereas the structured loader needed ever more. It declines what it cannot represent, covering fewer cases than that baseline and no electrocardiogram.
    
[^359]: 在线实验中的激励对齐

    Incentive Alignment in Online Experimentation

    [https://arxiv.org/abs/2610.05922](https://arxiv.org/abs/2610.05922)

    该论文将在线实验重新建模为激励设计问题，揭示了实验者基于有偏实验结果获得奖励所导致的委托-代理冲突会侵蚀平台价值，并证明样本拆分和收缩估计这两种实用机制能够有效实现激励对齐。

    

    arXiv:2610.05922v2 公告类型：replace-cross 摘要：评估新特征的因果效应是在线平台的核心目标。尽管近期文献通过集中式的组合优化来解决测试流量有限的问题，但这一视角忽略了一个关键的制度现实：实验在运营层面是去中心化的。开发新特征的实验者同时决定着要测试哪些假设，而他们通常基于容易产生向上偏差的经验平均处理效应来获得奖励。若不加约束，这种委托-代理冲突会严重侵蚀平台价值，这种结构性失效是传统的集中式手段（如显著性阈值和流量预算）所无法解决的。通过将实验重新构建为一个激励设计问题，我们证明了两种实用的机制——样本拆分与收缩估计——能够有效弥合这一鸿沟。样本拆分能以有限的流量成本实现完美的激励对齐……

    arXiv:2610.05922v2 Announce Type: replace-cross  Abstract: Evaluating the causal effect of new features is a central goal for online platforms. While recent literature addresses limited testing traffic via centralized portfolio optimization, this perspective abstracts away a critical institutional reality: experimentation is operationally decentralized. The experimenters who develop new features also dictate which hypotheses to test, and they are typically rewarded based on empirical average treatment effects that are prone to upward bias. Left unchecked, this principal-agent conflict can severely erode platform value, a structural failure that conventional centralized levers, such as significance thresholds and traffic budgets, cannot resolve. By reframing experimentation as an incentive design problem, we demonstrate that two practical mechanisms, sample splitting and shrinkage, can effectively bridge this gap. Sample splitting aligns incentives perfectly at a bounded traffic cost, w
    
[^360]: 弗雷歇距离衡量了什么？一种方向性分解

    What Does Fr\'echet Distance Measure? A Directional Decomposition

    [https://arxiv.org/abs/2610.05518](https://arxiv.org/abs/2610.05518)

    该论文提出“方向性弗雷歇距离”，将FID/FVD等单一标量指标分解为最优传输位移在少数可解释方向上的投影，从而揭示了标量指标背后的差异来源，解释了采样步数增加提升ImageReward却恶化FID的不一致现象。

    

    弗雷歇距离是跨领域评估生成模型的事实标准，在图像领域表现为FID，在视频领域表现为FVD。它将生成分布与参考分布之间的差异概括为单一标量，较低的数值通常被解读为更好的生成质量。然而，这种标量视角可能掩盖驱动比较结果的真正因素。例如，在COCO数据集上，增加扩散模型的采样步数会提升ImageReward分数，却使FID恶化（增大）。受这种不一致性的启发，我们试图通过揭示差异所在之处，使弗雷歇距离更具可解释性。为此，我们引入了方向性弗雷歇距离，即最优传输位移在给定方向上投影的期望平方。在我们的图像、视频和蛋白质案例研究中，我们发现少数几个可解释的方向就占据了总距离的大部分。我们利用这些方向……

    arXiv:2610.05518v1 Announce Type: cross  Abstract: The Fr\'echet distance is a de facto standard for evaluating generative models across domains, appearing as FID for images and FVD for videos. It summarizes the discrepancy between generated and reference distributions in a single scalar, with lower values typically interpreted as better generation quality. However, this scalar view can obscure what drives the comparison. For example, in COCO dataset, increasing the number of diffusion sampling steps improves ImageReward scores yet worsens (increases) FID. Motivated by this mismatch, we seek to make the Fr\'echet distance more interpretable by uncovering where the discrepancy lies. To this end, we introduce directional Fr\'echet distance, the expected squared projection of the optimal transport displacement onto a given direction. Across our image, video, and protein case studies, we find that a small number of interpretable directions account for much of the distance. We use these dir
    
[^361]: 面向测试时扩散对齐的 Best-of-N 引导方法

    Best-of-$N$ Guidance for Test-time Diffusion Alignment

    [https://arxiv.org/abs/2610.05108](https://arxiv.org/abs/2610.05108)

    该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。

    

    扩散模型具有强大的生成性能，但常常难以使生成的样本与通过奖励模型衡量的人类偏好对齐。一种简单而有效的测试时对齐算法是 Best-of-N (BoN) 采样，它从预训练的扩散模型中抽取 N 个独立同分布的样本，并输出奖励最高的单个样本。尽管 BoN 在实践中取得了成功，但它对奖励信息的利用十分有限，因为奖励信息仅在最终选择阶段才被纳入，而在采样过程中并不影响逆向扩散轨迹。因此，BoN 采样并不能提升生成样本的平均对齐程度，且主要适用于单输出场景。我们提出了 Best-of-N 引导（BoNG），这是一种将 BoN 采样原理直接融入逆向扩散过程的新颖方法。BoNG 对去噪粒子执行在线 BoN 选择，并据此调整逆向扩散过程（摘要内容在此处被截断）。

    arXiv:2610.05108v2 Announce Type: replace-cross  Abstract: Diffusion models achieve strong generative performance but often struggle to align generated samples with human preferences measured by a reward model. A simple yet effective algorithm for test-time alignment is Best-of-$N$ (BoN) sampling, which draws $N$ i.i.d. samples from a pre-trained diffusion model and outputs the single highest-reward sample. Despite its empirical success, BoN makes limited use of reward information, as it is incorporated only at the final selection stage without influencing the reverse diffusion trajectory during sampling. Consequently, BoN sampling does not improve the average alignment of generated samples and is primarily suited to single-output settings. We propose Best-of-$N$ Guidance (BoNG), a novel method that integrates the principle of BoN sampling directly into the reverse diffusion process. BoNG performs online BoN selection over denoising particles and adjusts the reverse diffusion process t
    
[^362]: E$^2$-OPSD：驯服在线策略自蒸馏中的熵过冲

    E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.05048](https://arxiv.org/abs/2610.05048)

    论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。

    

    在线策略自蒸馏（OPSD）无需第二个模型即可提供密集的token级监督：同一个网络在给定参考解答时充当教师，而在仅给定问题时充当学生。我们识别出该方法的一个特定失效模式：在训练过程中，学生的token熵会超过教师的熵并持续保持高位，我们将这一现象称为“熵过冲”（entropy overshoot）。我们将其根源追溯到蒸馏的双方。以参考答案为条件的教师在其面向答案的推理路径上表现得很自信，但这种自信难以迁移到学生生成的前缀上，使其监督过度依赖于答案特定的线索，而非可复用的推理模式；与此同时，OPSD所使用的前向KL散度会持续扩散学生的预测分布，而无法将其拉回。我们提出E$^2$-OPSD来同时应对这两个成因。示例引导的教学（exemplar-guided teaching）用检索到的已解决的相邻问题替换当前答案，提……（原文摘要在此处截断）

    arXiv:2610.05048v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (OPSD) provides dense token-level supervision without a second model: one network acts as teacher with the reference solution and as student with only the problem. We identify a specific failure mode of this recipe. During training, student token entropy rises past the teacher's and remains elevated, a pattern we call entropy overshoot. We trace it to both sides of distillation. The reference-conditioned teacher is confident along its answer-directed reasoning path, but this confidence transfers poorly to student-generated prefixes, making its supervision overly tied to answer-specific cues rather than reusable reasoning patterns; meanwhile, the forward KL used by OPSD continually diffuses the student's predictive distribution without pulling it back. We introduce E$^2$-OPSD to address both causes. Exemplar-guided teaching replaces the current answer with a retrieved solved neighboring problem, provi
    
[^363]: 跨越电子健康记录鸿沟：用于跨国医疗表征迁移的非对称对比学习

    Bridging the EHR Divide: Asymmetric Contrastive Learning for Cross-National Medical Representation Transfer

    [https://arxiv.org/abs/2610.04946](https://arxiv.org/abs/2610.04946)

    提出非对称监督对比学习预训练目标，在台湾398万名患者的NHIRD纵向记录上预训练时序Transformer编码器，并借助混合语义映射管线成功迁移至美国的MIMIC-IV和EHRSHOT数据集，实现了跨国家、跨临床词汇表的医疗表征迁移。

    

    跨系统的纵向电子健康记录（EHR）表征迁移极具挑战性，因为临床编码、患者人群和医疗工作流程在不同机构和国家之间存在显著差异。我们提出了非对称监督对比学习，这是一种受阴性临床结局异质性启发的、面向特定任务的预训练目标。该目标将共享目标阳性结局的患者聚类在一起，而不会显式地将阴性轨迹相互吸引。我们在台湾全民健康保险研究数据库（NHIRD）中398万名患者的纵向记录上预训练时序Transformer编码器，并将其迁移至两个美国EHR数据集：MIMIC-IV和EHRSHOT。通过一个结合直接映射与基于嵌入检索的混合语义映射管线，实现了跨异质临床词汇表的迁移。在MIMIC-IV上，NHIRD预……（原文摘要在此处截断）

    arXiv:2610.04946v2 Announce Type: replace  Abstract: Cross-system transfer of longitudinal Electronic Health Record (EHR) representations is challenging because clinical coding, patient populations, and healthcare workflows differ substantially across institutions and countries. We introduce Asymmetric Supervised Contrastive Learning (Asymmetric SupCon), a task-specific pre-training objective motivated by the heterogeneity of negative clinical outcomes. The objective clusters patients sharing a target positive outcome without explicitly attracting negative trajectories toward one another. We pre-train temporal Transformer encoders on longitudinal records from 3.98 million patients in the Taiwanese National Health Insurance Research Database (NHIRD) and transfer them to two U.S. EHR datasets, MIMIC-IV and EHRSHOT. A hybrid semantic mapping pipeline combining direct mappings with embedding-based retrieval enables transfer across heterogeneous clinical vocabularies. On MIMIC-IV, NHIRD pre
    
[^364]: SpecFold：折叠多分支冗余以加速扩散语言模型中的投机解码

    SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2610.04875](https://arxiv.org/abs/2610.04875)

    SpecFold通过识别并利用投机验证中草稿分支与父分支之间隐藏状态高度相似的多分支计算冗余，以token级残差门控和选择性计算复用降低验证成本，从而加速扩散语言模型的多分支投机解码。

    

    扩散大语言模型（DLLMs）通过迭代块去噪生成文本，多分支投机解码则通过在单次前向传播中同时验证一个主分支与多个草稿分支来加速这一过程。现有DLLM加速方法主要利用去噪步骤之间的时间冗余，而我们识别出每个投机验证步骤中一条互补的冗余维度：多分支计算冗余。在投机验证过程中，草稿分支从其父分支继承大部分token，仅解开少量额外位置，导致大量隐藏状态在各分支之间保持高度相似。我们提出SpecFold，一种算法-系统协同设计，利用这种多分支冗余来降低多分支投机验证的成本。在算法层面，SpecFold执行token级残差门控并选择性地复用父分支的计算（摘要原文在此处截断）。

    arXiv:2610.04875v2 Announce Type: replace  Abstract: Diffusion large language models (DLLMs) generate text through iterative block denoising, and multi-branch speculative decoding accelerates this process by verifying a main branch together with multiple draft branches in a single forward pass. While prior DLLM acceleration methods primarily exploit temporal redundancy across denoising steps, we identify a complementary redundancy axis within each speculative verification step: multi-branch computational redundancy. During speculative verification, draft branches inherit most tokens from their parents while unmasking a small set of additional positions, causing large portions of hidden states to remain highly similar across branches. We propose SpecFold, an algorithm-system co-design that exploits this multi-branch redundancy to reduce the cost of multi-branch speculative verification. Algorithmically, SpecFold performs token-level residual gating and selectively reuses parent computat
    
[^365]: AID：AI基础设施动力学框架

    AID: A Framework for AI Infrastructure Dynamics

    [https://arxiv.org/abs/2610.04801](https://arxiv.org/abs/2610.04801)

    本文提出AID框架，用于对耦合物理、计算、网络与服务进程的AI推理基础设施学习问题进行建模，支持异步观测与多时间尺度，并给出了预测误差下界和精确受控状态约简充分条件两项分析结果。

    

    一个有用的AI推理基础设施模型必须明确系统状态、观察者可获取的信息，以及该模型旨在支持的决策。我们提出了AID（AI基础设施动力学），这是一个用于描述跨越耦合的物理、计算、网络和服务进程的学习问题的框架。该表述允许结构化和可变大小的状态、异步观测、多个物理时间尺度以及响应服务而变化的负载需求。我们区分了在现有策略下支持预测的表示与在动作改变时保持服务结果的表示，并将二者与识别干预响应进一步区分开来。两个分析结果分别给出了当可用观测无法区分不同模型时预测误差的下界，以及实现精确受控状态约简的充分条件。这些结果应用了已有的信息论与状态（摘要原文在此处被截断）

    arXiv:2610.04801v2 Announce Type: replace-cross  Abstract: A useful model of AI inference infrastructure must specify the system state, the information available to an observer, and the decisions the model is intended to support. We introduce AID (AI Infrastructure Dynamics), a framework for describing this learning problem across coupled physical, computational, networking, and serving processes. The formulation allows structured and variable-size state, asynchronous observations, multiple physical timescales, and demand that responds to service. We distinguish representations that support prediction under an existing policy from those that preserve service outcomes under changed actions, and separate both from identifying intervention responses. Two analytical results describe a lower bound on prediction error when available observations cannot distinguish models and a sufficient condition for exact controlled state reduction. These results apply established information and state-abs
    
[^366]: 考虑电路：超越单一Softmax的深度分离与普适性

    Consideration Circuits: Depth Separation and Universality Beyond a Single Softmax

    [https://arxiv.org/abs/2610.04143](https://arxiv.org/abs/2610.04143)

    论文提出由多项logit（MNL）单元构成的有向无环图所定义的“考虑电路”多阶段选择模型，并证明了尖锐的深度-范数分离定理：深度从2增至3时，逼近误差ε所需的品味向量范数从Θ(log(1/ε)/ε)降至Θ(log(1/ε))，而包括单一MNL在内的菜单无关随机效用模型在折中任务上误差存在不可消除的下界，从而在表达能力与结构上严格超越了单一softmax模型。

    

    大多数基于特征的选择模型——无论是经典的还是深度的——都是对物品打分后应用单一的softmax。我们提出了“考虑电路”，这是一类基于特征的多阶段选择模型，由多项logit（MNL）单元构成的有向无环图定义。源单元为菜单中的物品分配概率，内部单元则利用由其前驱的概率加权特征摘要计算出的MNL权重来组合前驱分布。在一个具有固定非共线特征的三物品折中任务上，与菜单无关的随机效用模型（RUM），包括单个MNL单元，其误差存在一个不趋于零的下界。相比之下，对于考虑电路，我们建立了一个尖锐的深度-范数分离结果：将深度从2增加到3，可将达到误差ε所需的最优最大品味向量范数从Θ(log(1/ε)/ε)降低到Θ(log(1/ε))。深度为2的下界对任意宽度和与菜单无关的路由偏置均成立，而一个fi……（摘要原文在此处被截断）

    arXiv:2610.04143v2 Announce Type: replace  Abstract: Most feature-based choice models, classical and deep, score items and apply a single softmax. We introduce consideration circuits (CC), feature-based models of multi-stage choice defined by directed acyclic graphs of multinomial logit (MNL) units. Source units assign probabilities to menu items, and internal units combine predecessor distributions using MNL weights computed from their probability-weighted feature summaries. On a three-item compromise task with fixed non-collinear features, menu-independent random-utility models (RUM), including a single MNL unit, suffer an error bounded away from zero. For CC, in contrast, we establish a sharp depth--norm separation: increasing depth from $2$ to $3$ reduces the optimal maximum taste-vector norm for error $\epsilon$ from $\Theta(\log(1/\epsilon)/\epsilon)$ to $\Theta(\log(1/\epsilon))$. The depth-$2$ lower bound holds for arbitrary width and menu-independent routing biases, while a fi
    
[^367]: 大初始化条件下逼近逻辑斯蒂梯度下降轨迹的理想路径

    Ideal Paths for Approximating Logistic Gradient Descent Trajectories at Large Initialization

    [https://arxiv.org/abs/2610.04142](https://arxiv.org/abs/2610.04142)

    该论文提出由“负间隔修正”和“最小间隔增长”两个阶段组成的理想路径，证明在大初始化条件下逻辑斯蒂梯度下降的轨迹经过显式两阶段时间重参数化后可被该路径几何逼近。

    

    现代在新任务上的训练通常从先前训练好的模型出发，而非从零开始，这引出了一个问题：这种初始化方式如何影响后续的训练轨迹。经典的隐式偏差结果刻画了长时间训练最终选择的方向，但仅有该方向无法提供关于中间训练行为的信息。我们通过对严格线性可分数据上全批次逻辑斯蒂梯度下降（GD）轨迹进行几何逼近来研究这一问题，其中尺度为 $R$ 的大初始化正是受先前训练的启发。从任意极限归一化初始位置出发，我们利用最小范数投影规则构造出一条唯一的连续理想路径，该路径由有限个线性段组成。该路径包含两个阶段：先是负间隔修正，随后是最小间隔增长。我们证明，经过显式的两阶段时间重参数化后，固定步长的 GD 轨……

    arXiv:2610.04142v2 Announce Type: replace  Abstract: Modern training on a new task often starts from a previously trained model rather than from scratch, raising the question of how this initialization affects the subsequent training trajectory. Classical implicit-bias results characterize the direction selected by prolonged training, but this direction alone does not provide information regarding the intermediate behavior. We address this question through a geometric approximation of full-batch logistic gradient descent (GD) trajectories on strictly linearly separable data, with large initialization of scale $R$ motivated by prior training. From any limiting normalized initial position, we use minimum-norm projection rules to construct a unique continuous ideal path consisting of finitely many linear segments. The path has two stages: negative-margin correction followed by minimum-margin growth. We prove that, after an explicit two-stage time reparameterization, the fixed-step GD traj
    
[^368]: DePICT：面向受限下游任务的决策保持接口

    DePICT: Decision-Preserving Interface for Constrained Downstream Tasks

    [https://arxiv.org/abs/2610.03945](https://arxiv.org/abs/2610.03945)

    该论文提出DePICT方法，基于KKT条件刻画上下文方向与优化器的相关性，并依据解灵敏度对方向排序聚合，构建决策保持接口，从而揭示决策系统真正依赖的输入方向。

    

    一个约束优化问题可能在目标函数和主动约束中包含某个参数，然而最终决策对该参数的微小变化可能保持不敏感。这引出了一个根本性问题：决策系统究竟真正依赖于哪些输入？基于这一问题，我们提出了DePICT，一种通过根据优化器的解灵敏度对上下文方向进行排序，并在整个运行区间内进行聚合，从而构建决策保持接口的方法。我们在高维设置下研究这一问题：其中原始上下文对受约束任务进行参数化，而下游智能体仅能观察到经过筛选的上下文方向子集。对于局部正则的约束规划问题，我们推导出了基于KKT（Karush-Kuhn-Tucker）条件的刻画，用以判断一个上下文方向何时与优化器相关。我们的分析表明，出现在主动优化问题中并不一定意味着……

    arXiv:2610.03945v2 Announce Type: replace  Abstract: A constrained optimization problem may involve a parameter in its objective and active constraints, yet the final decision may remain insensitive to small changes in that parameter. This raises a fundamental question: which inputs does a decision making system truly depend on? Building on this question, we introduce DePICT, a procedure for constructing decision preserving interfaces by ranking context directions according to the optimizer's solution sensitivity and aggregating them across an operating regime. We study this problem in a high dimensional setting where primitive context parameterizes a constrained task and the downstream agent observes only a selected subset of context directions. For locally regular constrained programs, we derive a Karush Kuhn Tucker (KKT) based characterization of when a context direction is optimizer relevant. Our analysis shows that appearing in the active optimization problem does not necessarily 
    
[^369]: SCAD：面向长时程智能体的结构化信用分配与蒸馏

    SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2610.03372](https://arxiv.org/abs/2610.03372)

    SCAD通过将长时程智能体的交互分解为规划与有界子任务执行，并将基于结果的信用分配与教师引导蒸馏相结合，在文本和多模态任务上显著优于最强训练基线。

    

    训练长时程智能体解决复杂任务需要对长交互序列进行有效监督。然而，稀疏的终端奖励掩盖了中间步骤的贡献，而同策略蒸馏随着学生自身生成历史的增长可能会丢失有价值的教师指导。为解决这一问题，我们提出了SCAD，该方法将交互组织为规划与有界的子任务执行，在局部上下文中对执行进行蒸馏，并通过跨采样轨迹的子任务前缀树来细化规划信用，其中规划获得完整的终端信用，执行获得正向终端信用与教师指导。在所有评估基准上，SCAD相比最强训练基线，在文本任务上将宏平均准确率提升了4.48个百分点，在多模态任务上提升了4.19个百分点。SCAD有效地将基于结果的信用分配与教师引导的蒸馏相结合，从而提升长时程智能体的规划与执行能力。

    arXiv:2610.03372v1 Announce Type: new  Abstract: Training long-horizon agents to solve complex tasks requires effective supervision over extended interaction sequences. However, sparse terminal rewards obscure intermediate contributions, while on-policy distillation can lose informative teacher guidance as student-generated histories grow. To address this problem, we introduce SCAD, which organizes interactions into planning and bounded subtask execution, distills execution in local contexts, and refines planning credit through cross-rollout subtask prefix trees, with planning receiving full terminal credit and execution receiving positive terminal credit and teacher guidance. Across all evaluated benchmarks, SCAD improves macro-average accuracy over the strongest training baseline by 4.48 percentage points for text tasks and 4.19 points for multimodal tasks. SCAD effectively combines outcome-based credit assignment with teacher-guided distillation to improve planning and execution in 
    
[^370]: 偏好的可验证、可表达与默会成分

    Verifiable, Articulable, and Tacit Components of Preference

    [https://arxiv.org/abs/2610.03025](https://arxiv.org/abs/2610.03025)

    该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。

    

    是什么让一篇短篇小说引人入胜、一篇新闻报道具有新闻价值、或一个数学证明优雅？这些构念难以言明或验证，其含义至少部分是默会的。然而，现代AI模型主要是通过明确的章程、评分标准和验证器（即RLAIF和RLVR）来改进的；偏好的默会成分通常研究不足。我们引入了一个大规模、带标注的偏好数据集CreativePreferences，其中包含280万个文本，由3.17亿个人类偏好判断在7个创意领域中进行标注，并配有42个基准任务。我们分别用可执行程序、评分标准库和密集训练的模型（V、A和VAT）对这些标签进行建模。我们观察到稳健的可表达性差距（VAT−VA）和可验证性差距（VAT−V）；我们采用一种新颖的测量方法来估计每个差距的上界和下界，该方法可发现可表达和可验证的指标、识别伪变量，并估计未被发现成分的价值。

    arXiv:2610.03025v1 Announce Type: new  Abstract: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undisc
    
[^371]: 解耦记忆与上下文：面向令牌高效测试时持续学习的结构化记忆

    Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning

    [https://arxiv.org/abs/2610.02687](https://arxiv.org/abs/2610.02687)

    该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。

    

    大型语言模型越来越多地被部署在企业、科学和医疗应用中，在这些场景下，智能体必须整合领域特定知识并从经验中不断适应。上下文工程通过在推理时提供指令、策略和证据来改善模型行为，为权重更新提供了一种实用的替代方案。然而，在线调整上下文通常需要代价高昂的试错过程，而且查询往往被独立处理，导致有用的经验无法延续下去。记忆系统通过在多次交互之间保留信息来解决这一局限，但那些不断向共享上下文追加信息的方法会面临令牌成本不断上升、上下文窗口受限以及随上下文扩展而出现的性能退化。我们提出了上下文优化的统一形式化框架，并表明智能体记忆系统的更新可以被解释为一种优化……（摘要内容在此处截断）

    arXiv:2610.02687v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in enterprise, scientific, and medical applications, where agents must incorporate domain-specific knowledge and adapt from experience. Context engineering offers a practical alternative to weight updates by improving model behavior through instructions, strategies, and evidence supplied at inference time. However, adapting context online typically requires a costly trial-and-error process, while queries are often processed independently, preventing useful experience from carrying forward. Memory systems address this limitation by retaining information across interactions, but approaches that continually append information to a shared context face increasing token costs, context-window limits, and performance degradation as the context expands. We introduce a unified formulation of context optimization and show that an agent memory system update can be interpreted as an optimization 
    
[^372]: 基于选择性状态空间模型的分布式学习：架构感知的收敛性分析

    Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis

    [https://arxiv.org/abs/2610.02659](https://arxiv.org/abs/2610.02659)

    该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。

    

    现代状态空间模型（SSM），如Mamba2，通过将线性时间复杂度的序列建模与递归状态空间动力学相结合，为Transformer提供了一种极具吸引力的替代方案。然而，SSM在分布式学习环境中的行为仍然鲜为人知。特别是，现有的标准联邦学习方法在很大程度上与架构无关，没有考虑到现代选择性SSM所特有的稳定性、选择性和状态空间参数化特性。为了解决这一问题，我们为单层和多层选择性SSM推导了架构感知的梯度和平滑度界限，并为FedAvg和FedProx推导了收敛界限，刻画了递归稳定性、输入相关的离散化以及状态投影范数如何影响联邦优化。随后，我们在由教师SSM生成的序列上，使用遵循所分析递归结构的学习器，对单层界限进行了数值验证。

    arXiv:2610.02659v1 Announce Type: cross  Abstract: Modern state space models (SSMs), such as Mamba2, provide a compelling alternative to transformers by combining linear-time sequence modeling with recurrent state-space dynamics. However, the behavior of SSMs in distributed learning settings remains poorly understood. In particular, the existing standard federated learning methods are largely architecture-agnostic, and do not account for the stability, selectivity, and state-space parameterization that characterize modern selective SSMs. To address this, we derive architecture-aware gradient and smoothness bounds for single- and multi-layer selective SSMs, and convergence bounds for FedAvg and FedProx, characterizing how recurrent stability, input-dependent discretization, and state projection norms affect federated optimization. We then numerically validate the single-layer bounds on sequences generated by a teacher SSM, using a learner that follows the analyzed recurrence. We use thi
    
[^373]: 有限记忆量子过程的序列容量

    Sequential Capacity of Quantum Processes with Finite Memory

    [https://arxiv.org/abs/2610.02068](https://arxiv.org/abs/2610.02068)

    该论文证明，仅用一个可见量子比特、无额外内部记忆的量子系统，通过时间依赖的相位旋转即可实现随运行长度以 $K\log K$ 量级增长的序列响应容量，而经典随机过程在相同测试下仅有线性容量。

    

    arXiv:2610.02068v1 公告类型： cross 摘要：当一个量子设备在固定内部记忆下运行更长时间时，其响应能变得多么复杂？我们通过“序列响应容量”来量化这种复杂性：即多少个自适应测试阶段（每个阶段使用一次全新的运行）能够继续以给定的响应概率差距区分不同的可能过程。对于固定的系统和记忆规模，我们建立了一个紧密的定律，将该容量与运行长度和概率分辨率联系起来。在固定分辨率下，该容量以 $K\log K$ 的量级增长，其中 $K$ 是每次运行中的时间步数。我们的构造仅使用单个可见量子比特上的时间依赖相位旋转即可实现这一增长，无需额外的内部记忆；其测试给出的响应概率恰好为零或一。在相同的测试下，每一步都在固定基上进行测量的经典随机过程在固定规模和分辨率下仅有线性容量。对于由随…

    arXiv:2610.02068v1 Announce Type: cross  Abstract: How complex can the responses of a quantum device become as it runs longer with a fixed internal memory? We quantify this complexity through sequential response capacity: how many adaptive testing stages, each using a fresh run, can continue to separate possible processes by a prescribed gap in response probabilities. For fixed system and memory sizes, we establish a tight law relating this capacity to run length and probability resolution. At fixed resolution, the capacity grows on the order of $K\log K$, where $K$ is the number of time steps in each run. Our construction attains this growth using time-dependent phase rotations on a single visible qubit with no additional internal memory; its tests give response probabilities exactly zero or one. Under the same tests, classical stochastic processes that measure in a fixed basis at every step have only linear capacity at fixed sizes and resolution. For phase sequences selected by a sto
    
[^374]: SkillEvoLean：面向Lean证明器的变异增强技能演化方法

    SkillEvoLean: Mutation-enhanced skill evolution for Lean provers

    [https://arxiv.org/abs/2610.01799](https://arxiv.org/abs/2610.01799)

    提出SkillEvoLean——一种变异增强的技能自演化框架，通过联合演化高层解题策略与参考知识（数学概念、证明技巧），即使在所有采样证明轨迹全部失败时也能为Lean证明器提供有效的技能更新方向。

    

    技能演化为在不更新模型参数的情况下提升大语言模型智能体能力提供了一条有前景的途径，但其在形式化定理证明中的应用仍鲜被探索。现有方法主要针对自然语言推理任务，通过分析成功与失败的轨迹并逐步修正解题策略来改进技能。尽管Lean验证器能够提供可靠的执行反馈，但当所有采样的轨迹全部失败时，现有的技能演化方法便缺乏可供推断有效更新方向的成功轨迹。此外，这些方法主要关注根指令文件，因而对包括数学概念与证明技巧在内的参考知识的演化探索不足。为解决这些局限，我们提出了一种变异增强的技能自演化框架，用于构建技能增强的Lean证明器。该框架联合演化高层解题策略及其参考知识……（原文摘要在此处截断）

    arXiv:2610.01799v1 Announce Type: new  Abstract: Skill evolution offers a promising way to improve large language model agents without updating their parameters, but its use in formal theorem proving remains underexplored. Existing methods mainly target natural-language reasoning, improving skills by analyzing successful and failed trajectories and incrementally revising solving strategies. Although the Lean verifier provides reliable execution feedback, when all sampled trajectories fail, existing skill evolution methods lack successful trajectories from which to infer effective update directions. Furthermore, these methods also focus mainly on the root instruction file, thus underexploring the evolution of reference knowledge including mathematical concepts and proving techniques. To address these limitations, we propose a mutation-enhanced skill self-evolution framework for building skill-augmented Lean provers. The framework jointly evolves a high-level solving policy and its refer
    
[^375]: 面向世界动作模型的完成感知引导

    Completion Aware Guidance for World Action Models

    [https://arxiv.org/abs/2610.01559](https://arxiv.org/abs/2610.01559)

    提出无需训练的完成感知引导（CAG）采样方法，引导世界动作模型的生成朝向任务完成，显著提升机器人任务成功率并将任务不完整想象从 79% 大幅降至 40%。

    

    世界动作模型预测视觉未来和机器人动作，但它们仍然容易受到任务不完整想象的影响，即看似合理且与动作一致的预测遗漏了完成任务所需的转变。在本文中，我们证明这种失败并非世界模型骨干网络的固有缺陷，而是在将其适配于短片段控制时出现的，短片段控制会反复倾向于看似合理的局部延续而非完成任务的转变。为解决这一问题，我们提出了完成感知引导（CAG），这是一种无需训练的采样方法，可引导生成过程朝向任务完成。在具有代表性的世界动作模型上，CAG 在 RoboTwin 2.0 子集上将成功率从 64% 提升至 70%，在零样本仿真中将成功率从 69% 提升至 75%，同时将任务不完整想象的比例从 79% 降低至 40%。

    arXiv:2610.01559v1 Announce Type: cross  Abstract: World Action Models (WAMs) predict visual futures and robot actions, yet they remain susceptible to task-incomplete imagination, where plausible, action-consistent predictions omit the transition needed for task completion. In this paper, we show that this failure is not inherent to the world model backbone, but emerges when adapted for short-chunk control, which can repeatedly favor plausible local continuations over task-completing transitions. To address this, we introduce Completion Aware Guidance (CAG), a training-free sampling method that guides generation toward task completion. Across representative WAMs, CAG improves success from 64% to 70% on a RoboTwin 2.0 subset and from 69% to 75% in zero-shot simulation, while reducing task-incomplete imagination from 79% to 40%.
    
[^376]: 强化学习的扩展真的需要更多训练吗？

    Does Scaling Reinforcement Learning Really Require More Training?

    [https://arxiv.org/abs/2610.01133](https://arxiv.org/abs/2610.01133)

    提出策略空间扩展方法SURGE，通过对同一强化学习运行中的两个检查点（高精度锚点与生成简短回复的供体）进行特征空间融合，在不增加训练或推理计算的情况下获得比原检查点更强的策略。

    

    扩展推理能力通常需要在强化学习（RL）或推理阶段投入更多计算资源。我们证明，一段已完成的强化学习训练历史能够产生比其优化器所访问过的检查点更强的策略。我们将其称为策略空间扩展：在不延长训练时间、也不增加单次查询推理计算量的前提下，从固定的强化学习训练历史中扩展可部署的策略集合。我们通过SURGE（通过特征空间融合实现无梯度强化学习扩展）来实例化这一方法。SURGE将同一强化学习运行中的两个检查点结合起来：一个高精度的锚点检查点和一个具有竞争力且能生成更简短回复的供体检查点。它将两个检查点表示为相对于其共享初始化的变化量，然后对锚点的更新进行谱分解，以保留其主导分量并融入供体的互补分量。在给定锚点更新保留比例的固定目标下，SURGE无需测试即可直接从权重本身确定块大小……

    arXiv:2610.01133v1 Announce Type: cross  Abstract: Scaling reasoning typically spends more compute on reinforcement learning (RL) or on inference. We show that a completed RL training history can yield policies stronger than the checkpoints visited by its optimizer. We call this policy-space scaling: expanding the deployable policy set accessible from a fixed RL history, without extending training or increasing per-query inference computation. We instantiate it with SURGE (Scaling Up RL Gradient-free via Eigenspace fusion). SURGE combines two checkpoints from the same RL run: a high-accuracy anchor and a competitive donor that generates shorter responses. It expresses both checkpoints as changes from their shared initialization, then spectrally decomposes the anchor's update to retain its dominant component and incorporate the donor's complementary component. With a fixed target for how much of the anchor update to retain, SURGE determines the block size from the weights without testin
    
[^377]: 语言模型能从测试时计算中获得多少收益？

    How Much Can Language Models Gain from Test-Time Computation?

    [https://arxiv.org/abs/2610.01110](https://arxiv.org/abs/2610.01110)

    SELF-POT 是一个统一基准框架，以美元计价所有模型调用，衡量语言模型在数学、编程和智能体任务中从并行采样、自我修订等测试时计算中能获得的收益与成本。

    

    测试时计算能在多大程度上提升语言模型，其代价又是什么？测试时扩展被广泛提出作为更大模型的替代方案，但现有的比较大多一次只评估单一领域，且很少将模型选择的成本计入预算。我们提出了 SELF-POT，这是一个用于衡量模型在竞赛数学、竞赛编程和智能体工作流中测试时潜力的基准测试与评估框架。SELF-POT 在静态任务中将候选覆盖率与最终准确率分离，跟踪修订过程中的正确性转变，并在智能体环境中同时衡量协议完成度与任务成功率。在统一的预算规则下，它比较了直接推理与固定倍数于直接预算的并行采样和自我修订，并以美元计价每一次模型调用，包括选择与批判。这一设计支持两类比较：模型从额外

    arXiv:2610.01110v1 Announce Type: new  Abstract: How much can test-time computation improve a language model, and at what cost? Test-time scaling is widely proposed as a substitute for larger models, but existing comparisons mostly evaluate one domain at a time and rarely charge selection to the budget. We introduce SELF-POT, a benchmark and evaluation framework that measures the test-time potential of a model across competition mathematics, competitive programming, and agentic workflows. SELF-POT separates candidate coverage from final accuracy on static tasks, tracks correctness transitions under revision, and measures protocol completion alongside task success in agentic environments. Under a unified budget rule, it compares Direct inference with parallel sampling and self-revision under fixed multiples of the Direct budget, and charges every model call, including selection and critique, in dollars. This design supports two kinds of comparison: the gain a model obtains from addition
    
[^378]: 对抗性线性约束马尔可夫决策过程的最优率算法

    Rate-Optimal Algorithm for Adversarial Linear CMDPs

    [https://arxiv.org/abs/2610.00927](https://arxiv.org/abs/2610.00927)

    本文提出一种新的原始对偶算法，在无需假设Slater条件的情况下，将对抗性线性约束马尔可夫决策过程的遗憾和累积约束违反从$\widetilde{\mathcal{O}}(K^{3/4})$提升至最优的$\widetilde{\mathcal{O}}(\sqrt{K})$，弥补了理论差距。

    

    我们研究了具有未知转移的情节式对抗性线性约束马尔可夫决策过程（CMDP），其中损失函数和约束函数可能在各情节之间以对抗方式变化。此前最好的算法实现了 $\widetilde{\mathcal{O}}(K^{3/4})$ 的遗憾和累积约束违反，与关于情节数 $K$ 的最优 $\widetilde{\mathcal{O}}(\sqrt{K})$ 依赖之间存在差距。我们通过提出一种新的原始对偶算法来弥补这一差距，该算法在无需假设 Slater 条件的情况下实现了 $\widetilde{\mathcal{O}}(\sqrt{K})$ 的遗憾和累积约束违反。主要挑战在于，学习线性 CMDP 需要在具有可控覆盖数的值函数类上实现一致集中，而约束在线学习中的标准技术（如策略混合）可能会使该函数类变得更加复杂。我们的算法结合了自适应跟随正则化领导者（FTRL）方法、收缩值函数……

    arXiv:2610.00927v1 Announce Type: new  Abstract: We study episodic adversarial linear constrained Markov decision processes (CMDPs) with unknown transitions, where both the loss and constraint functions may vary adversarially across episodes. The best previous algorithm achieves $\widetilde{\mathcal{O}}(K^{3/4})$ regret and cumulative constraint violation, leaving a gap to the optimal $\widetilde{\mathcal{O}}(\sqrt{K})$ dependence on the number of episodes $K$. We close this gap by proposing a new primal dual algorithm that achieves $\widetilde{\mathcal{O}}(\sqrt{K})$ regret and cumulative constraint violation without assuming Slater's condition. The main challenge is that learning linear CMDPs requires uniform concentration over a value function class with a controlled covering number, whereas standard techniques in constrained online learning, such as policy mixing, can make this class more complex. Our algorithm combines adaptive Follow the Regularized Leader (FTRL), contracted valu
    
[^379]: 奖励即观测：学习基于奖励的策略以实现快速适应

    Reward as Observation: Learning Reward-Based Policies for Rapid Adaptation

    [https://arxiv.org/abs/2610.00729](https://arxiv.org/abs/2610.00729)

    本文提出一种仅以奖励和动作为条件的策略学习方法，实现了从简单环境向观测空间完全不同的新环境（如3D渲染和机器人导航）的零样本快速迁移。

    

    本文探索了一种基于奖励的策略，以在观测空间完全不同的源环境和目标环境之间实现零样本迁移。尽管人类能够展现出令人印象深刻的适应能力，但深度神经网络策略通常难以适应新环境，并且需要大量样本才能成功迁移。为此，我们提出了一种仅以奖励和动作为条件的新型基于奖励的策略，使其能够零样本适应观测完全不同的新环境。我们讨论了基于奖励的策略所面临的挑战与可行性，并随后提出了一种实用的训练算法。我们证明了奖励策略可以在三个不同的环境中训练（质点、倒立摆和2D赛车），并以零样本方式迁移到完全不同的观测环境中，例如不同的调色板、3D渲染，或Habitat-Sim中Stretch机器人的导航任务。

    arXiv:2610.00729v1 Announce Type: new  Abstract: This paper explores a reward-based policy to achieve zero-shot transfer between source and target environments with completely different observation spaces. While humans can demonstrate impressive adaptation capabilities, deep neural network policies often struggle to adapt to a new environment and require a considerable amount of samples for successful transfer. Instead, we propose a novel reward-based policy only conditioned on rewards and actions, enabling zero-shot adaptation to new environments with completely different observations. We discuss the challenges and feasibility of a reward-based policy and then propose a practical algorithm for training. We demonstrate that a reward policy can be trained within three different environments, Pointmass, Cartpole, and 2D Car Racing, and transferred to completely different observations, such as different color palettes or 3D rendering, or Stretch robot navigation in Habitat-Sim, in a zero-
    
[^380]: 巨正则生成器

    Grand Canonical Generators

    [https://arxiv.org/abs/2610.00683](https://arxiv.org/abs/2610.00683)

    提出了巨正则生成器（GCG），将玻尔兹曼生成器扩展至巨正则系综，其分解式设计可复用现有正则生成器、解析编码化学势线性依赖，并提供可处理的似然以支持自归一化重要性采样，在流体和吸附问题上准确再现巨正则观测量。

    

    我们提出了巨正则生成器，这是一种将玻尔兹曼生成器扩展到巨正则系综的生成式框架。我们提出了两种设计方案：第一种以化学势为条件对可变尺寸的生成模型进行条件化，从而联合采样粒子数和构型；第二种将巨正则分布分解为粒子数分布和相应的正则玻尔兹曼密度。这种分解式设计可以对正则分量复用任何现有的玻尔兹曼生成器，以解析方式编码已知的化学势线性依赖关系，并产生易于处理的似然，从而支持自归一化重要性采样（SNIS）。实验结果表明，GCG在Lennard-Jones流体和沸石中甲烷吸附问题上准确再现了巨正则观测量，展示了跨化学势的泛化能力，并可通过SNIS和巨正则蒙特卡洛进行校正。

    arXiv:2610.00683v1 Announce Type: cross  Abstract: We introduce Grand Canonical Generators (GCG), a generative framework that extends Boltzmann generators to the grand canonical ensemble. We present two designs. The first conditions a variable-size generative model on the chemical potential, sampling particle number and configuration jointly. The second factorizes the grand canonical distribution into a particle-number distribution and the corresponding canonical Boltzmann density. This factorized formulation can use any existing Boltzmann generator for the canonical component, encodes the known linear chemical-potential dependence analytically, and yields a tractable likelihood that supports self-normalized importance sampling (SNIS). Empirically, GCG accurately reproduces grand canonical observables on a Lennard--Jones fluid and methane adsorption in a zeolite, demonstrating generalization across chemical potentials and correction via SNIS and grand canonical Monte Carlo.
    
[^381]: 面向高效自适应谱递归的共享相位与保留控制

    Shared Phase and Retention Control for Efficient Adaptive Spectral Recurrence

    [https://arxiv.org/abs/2609.39082](https://arxiv.org/abs/2609.39082)

    SPARC证明高维谱记忆无需高维控制，仅用两个输入依赖的标量信号即可共享协调记忆保留与相位旋转，实现控制成本与状态容量解耦的高效自适应谱递归模型。

    

    随着新证据的到来，序列模型必须更新它所记忆的内容以及记忆如何影响预测。虽然Transformer的计算和缓存开销随上下文长度增长而扩展，但固定状态递归模型能够提供恒定内存的推理。然而，线性递归和谱递归传统上依赖静态转移，无法动态修订已存储表示的衰减或旋转方式。尽管近期的选择性架构引入了依赖输入的转移机制，但它们为每个记忆模式分配独立的控制，导致控制成本与状态容量相耦合。我们证明高维谱记忆并不需要高维控制，并提出了面向高效自适应谱递归的共享相位与保留控制方法（SPARC）。SPARC仅使用两个依赖输入的标量信号，即可在异构的复数模式间协调记忆保留与相位旋转，同时保留各模式特定的基线特性。

    arXiv:2609.39082v1 Announce Type: new  Abstract: As new evidence arrives, a sequence model must update what it remembers and how memory influences predictions. While Transformers incur computation and cache costs scaling with context length, fixed-state recurrent models offer constant-memory inference. However, linear and spectral recurrences traditionally rely on static transitions, failing to dynamically revise how stored representations decay or rotate. While recent selective architectures introduce input-dependent transitions, they assign independent controls to every memory mode, coupling control cost to state capacity. We show that high-dimensional spectral memory does not require high-dimensional control, and introduce Shared Phase and Retention Control for Efficient Adaptive Spectral Recurrence (SPARC). SPARC employs just two input-dependent scalar signals to coordinate memory retention and phase rotation across heterogeneous complex modes, while preserving mode-specific baseli
    
[^382]: 存储并非策略：面向大语言模型遗忘的状态条件支撑集控制

    Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning

    [https://arxiv.org/abs/2609.37858](https://arxiv.org/abs/2609.37858)

    该论文发现“存储目标知识”的参数未必是执行遗忘的最佳干预对象，提出基于实际遗忘更新预测效果的干预分数以及动态干预重排方法（DIR-R），在优化过程中按需自适应调整干预参数子集，从而显著提升大语言模型遗忘的效果。

    

    许多局部化的大语言模型（LLM）遗忘方法从定位信号中选出一小部分参数子集，并在优化过程中将其固定不变。然而，与目标知识关联最强的参数并不一定是最佳的更新对象，且候选干预的价值会随优化进程而变化。在一个受控实验中，存储定位分数达到了0.981的受试者工作特征曲线下面积（AUROC），但存储身份仅在17/36个目标上与更优的干预选择一致，而低秩适应（LoRA）则在35/36个目标上胜出。我们提出干预分数，该分数根据实际遗忘更新的预测效果对可编辑参数组进行排序，同时将附带损害纳入考量，并以此构建静态干预价值基线。随后我们进一步提出选择性动态干预重排（DIR-R），仅当经过校准的探针证明有必要时，才对该参数子集进行重新审视和调整。

    arXiv:2609.37858v1 Announce Type: cross  Abstract: Many localized large language model (LLM) unlearning methods select a small parameter subset from a localization signal and keep it fixed during optimization. The parameters most associated with a target, however, need not be the best ones to update, and candidate interventions can change value as optimization proceeds. In a controlled experiment, a storage-localization score reaches an area under the receiver operating characteristic curve (AUROC) of 0.981, yet storage identity agrees with the better intervention on only 17/36 targets, while low-rank adaptation (LoRA) wins 35/36. We introduce Intervention Score, which ranks editable groups by the predicted effect of the actual unlearning update while accounting for collateral damage, and use it to form the static intervention-value baseline (Static-IV). We then introduce selective dynamic intervention re-ranking (DIR-R), which revisits that subset only when a calibrated probe justifie
    
[^383]: FineART：面向双手操作的细粒度标注机器人轨迹数据集与视觉-语言-动作模型

    FineART: Fine-grained Annotated Robotic Trajectory Dataset and Vision-Language-Action Model for Bimanual Manipulation

    [https://arxiv.org/abs/2609.36416](https://arxiv.org/abs/2609.36416)

    本文提出了包含40,543个回合、1,718小时和533,913个密集子任务标注的双手操作数据集FineART，以及能自主预测下一个子任务的视觉-语言-动作模型FineART-VLA，显著提升了长时程双手操作任务的成功率。

    

    在真实世界环境中运行的机器人必须执行复杂的多步骤、长时程双手任务，而非单一、孤立的动作。当前的操控数据集难以支撑这种能力：尽管单臂数据集已达到数十万条轨迹的规模，但它们通常每个回合只提供一条高级指令，而少数标注了子任务的双手操作工作也仅标注了其数据时长的一小部分。我们提出了FineART，这是一个密集标注的双手操作数据集，包含40,543个回合、1,718小时时长以及151个任务中的533,913个子任务。我们还引入了FineART-VLA，一种能够预测自身下一个子任务的视觉-语言-动作策略，并表明以这种方式进行中期训练能带来显著提升。具体而言，空间消歧任务的成功率从32.0%提升到100.0%，而借助逐步的人类子任务指导，模型在未见过的长时程任务上的成功率从16.0%提升到76.0%。

    arXiv:2609.36416v1 Announce Type: cross  Abstract: Robots operating in real-world environments must execute complex, multi-step bimanual tasks over long horizons rather than single, isolated actions. Current manipulation datasets struggle to support this capability: although single-arm datasets reach hundreds of thousands of trajectories, they typically provide only one high-level instruction per episode while the rare bimanual effort that does label subtasks annotates only a fraction of its hours. We present FineART, a densely annotated bimanual manipulation dataset of 40,543 episodes, 1,718 hours, and 533,913 subtasks across 151 tasks. We also introduce FineART-VLA, a vision-language-action policy that predicts its own next subtask, and show that mid-training it this way yields substantial gains. Specifically, success on a spatial disambiguation task increases from 32.0% to 100.0%, and step-by-step human subtask guidance lifts success on an unseen long-horizon task from 16.0% to 76.0
    
[^384]: LionMuon：交替使用谱下降与符号下降的高效训练方法

    LionMuon: Alternating Spectral and Sign Descent for Efficient Training

    [https://arxiv.org/abs/2609.35297](https://arxiv.org/abs/2609.35297)

    LionMuon 通过每 P 步交替执行一次昂贵的 Muon 谱步骤与廉价的 Lion 符号步骤，并共享单一双重 EMA 动量缓冲区，在大幅降低计算与通信开销（优化器状态仅为 AdamW 一半）的同时保持甚至超越 Muon 的优化效果，并从理论上证明了其在重尾噪声下何时优于两种原始优化器。

    

    预训练语言模型需要巨大的计算量，而选择合适的优化器可以节省其中相当大的一部分。Muon 的谱步骤比符号步骤能提供更强的更新方向，但代价高昂：每一步都需要对完整矩阵运行 Newton-Schulz 迭代，并且在分布式训练中还需要额外的 all-reduce 通信。而 Lion 和 Signum 中的符号步骤计算廉价，且在每个设备上保持局部性。我们提出 LionMuon，它每 P 次迭代执行一次 Muon 步骤，中间穿插 Lion 步骤，两者共享一个双重 EMA 动量缓冲区。Muon 的计算与通信开销每 P 步只需支付一次，且优化器状态仅为 AdamW 的一半。一个单 EMA 的变体 SignMuon 已经能够超越 Muon。我们在重尾噪声下证明了复杂度上界，其中周期 P 在 Muon 与 Lion 的平滑性和噪声常数之间建立了一种插值，并据此给出了 LionMuon 何时比两者都更快的条件。在 FineWeb 上训练的 124M 和 355M 模型上，LionMuon（摘要在此处截断）

    arXiv:2609.35297v3 Announce Type: replace  Abstract: Pretraining a language model takes enormous compute, and the right optimizer can save a good part of it. Muon's spectral step gives a stronger direction than a sign step, but it is expensive. Every step runs Newton-Schulz iterations on the full matrix and, in distributed training, an extra all-reduce. Sign steps, as in Lion and Signum, are cheap and stay local to each device. We propose LionMuon, which takes one Muon step every $P$ iterations and Lion steps in between, with a single dual-EMA momentum buffer shared by both. Muon's compute and communication are paid once per $P$ steps, and the optimizer state is half of AdamW's. A single-EMA variant, SignMuon, already improves on Muon. We prove complexity bounds under heavy-tailed noise in which the period sets an interpolation between Muon's and Lion's smoothness and noise constants, and which say when LionMuon is faster than both. On 124M and 355M models trained on FineWeb, LionMuon 
    
[^385]: CLAD：用于神经网络验证的约束抽象域

    CLAD: Constrained Abstract Domain for Neural Network Verification

    [https://arxiv.org/abs/2609.34628](https://arxiv.org/abs/2609.34628)

    提出了约束拉格朗日抽象域（CLAD），能够在Lp范数球附加额外约束的复杂输入区域上计算神经网络行为更紧致的可靠过近似，从而克服现有抽象域因输入区域描述受限而导致的验证失败或虚假反例问题。

    

    神经网络验证（NNV）用于形式化地验证一个网络对于定义区域内所有输入都满足给定的属性。现代神经网络验证工具采用抽象域从给定输入区域出发计算网络行为的可靠过近似，因此这些抽象的紧致程度本质上决定了验证的效率。学界已开发出一系列精度不断提升的抽象域，但它们都以同样受限的方式描述有效输入区域，例如Lp范数球。然而，实际中的输入区域很少是简单的Lp球，而往往是Lp球与额外约束的组合。在 such 区域上使用现有抽象方法验证网络会产生松散的过近似，导致无法验证属性或产生虚假反例。我们提出了约束拉格朗日抽象域（CLAD），这是一种新的抽象域，能够计算神经网络[行为的可靠过近似]（摘要在此处被截断）。

    arXiv:2609.34628v2 Announce Type: replace-cross  Abstract: Neural network verification (NNV) formally verifies that a network satisfies a specified property for all inputs within a defined region. Modern NNV tools employ abstract domains to compute a sound over-approximation of the network's behavior from the given input region, thus the tightness of these abstractions essentially determines efficiency. A long line of increasingly precise domains has been developed, but they all describe the valid input region in the same restrictive way, e.g., an Lp-norm ball. A practical input region is rarely a simple Lp ball, but rather a combination Lp ball with additional constraints. Verifying a network over such a region with existing abstraction produces a loose over-approximation, which results in either failing to verify a property or spurious counterexamples. We introduce Constrained Lagrangian Abstract Domain (CLAD), a new abstract domain that computes a sound over-approximation of neural 
    
[^386]: ZonoGPT：面向大型GPT模型验证的抽象域

    ZonoGPT: Towards An Abstract Domain for Verifying Large GPT Models

    [https://arxiv.org/abs/2609.34457](https://arxiv.org/abs/2609.34457)

    ZonoGPT提出了一种空间复杂度与网络深度无关的抽象域，通过结构化zonotope、生成元约简机制以及针对Attention、LayerNorm和GELU的保精度变换，实现了对大型GPT模型的高效形式化验证。

    

    arXiv:2609.34457v2 公告类型：替换 摘要：基于Transformer的模型被广泛应用于推理、编程和多模态智能体任务。为了对期望的行为（如鲁棒性、安全性和公平性）提供形式化保证，神经网络验证技术在部署前证明所需属性并提供可审计的保证。然而，先前的工作仍局限于小型或受限的Transformer模型，并且在深层模型中保持验证精度仍然具有挑战性。在本工作中，我们提出了ZonoGPT，一种用于验证大型Transformer的抽象域，其空间复杂度与网络深度无关。ZonoGPT使用结构化zonotope和生成元约简机制来高效地保留变量间的关联性。为了保持精度，它为Attention和LayerNorm引入了保留特征关系的块级特定融合变换，并为GELU引入了保留生成元关系的仿射变换。这些机制使ZonoGPT能够……

    arXiv:2609.34457v2 Announce Type: replace  Abstract: Transformer-based models are widely used for reasoning, coding, and multimodal agentic tasks. To provide formal assurance of desirable behaviors, such as robustness, safety, and fairness, neural network verification techniques prove required properties and provide auditable guarantees before deployment. However, prior work remains limited to small or restricted Transformers, and maintaining precision across deep models remains challenging. In this work, we introduce ZonoGpt, an abstract domain for verifying large transformers that maintains a space complexity independent of network depth. ZonoGpt uses a structured zonotope and a generator reduction mechanism to efficiently preserve correlations. To maintain precision, it introduces block-specific fused transformations for Attention and LayerNorm that retain feature relations, along with an affine transform for GELU that preserves generator relations. These mechanisms enable ZonoGpt t
    
[^387]: KVCMAS：面向多智能体系统中共享上下文的高效KV缓存修正

    KVCMAS: Efficient KV cache Correction for Shared Context in Multi-Agent Systems

    [https://arxiv.org/abs/2609.34060](https://arxiv.org/abs/2609.34060)

    提出了KVCMAS在线KV缓存修正框架，通过修正多智能体系统中跨智能体的KV缓存偏差，避免各智能体重复预填充共享上下文，从而显著降低计算与内存开销。

    

    提示词特化的多智能体系统使多个智能体能够共享同一个模型，同时执行互补角色以解决复杂任务。然而，智能体特定的前缀会改变为相同共享上下文生成的KV缓存，导致每个智能体重复预填充不断增长的上下文，并构建各自独立的缓存，带来高昂的计算与内存开销。选择性重计算虽然能减少这种冗余，但仍然保留了大量的模型执行；而现有的增量修正方法要么仅支持重复出现的上下文关系，要么需要为动态变化的上下文维护内存密集型的在线修正状态。对于首次出现的共享上下文，这些方法还需要在智能体工作流之外构建参考缓存，且第一个智能体处的近似修正会影响传递给后续智能体的输出。我们提出了KVCMAS，一个在线KV缓存修正框架，它能够表示跨智能体的缓存偏差（摘要在此处被截断）。

    arXiv:2609.34060v2 Announce Type: replace  Abstract: Prompt-specialized multi-agent systems enable multiple agents to share a model while performing complementary roles to solve complex tasks. However, agent-specific prefixes change the KV cache generated for the same shared context, causing each agent to repeatedly prefill the growing context and construct a separate cache with high computation and memory overhead. Selective recomputation reduces this redundancy but still retains substantial model execution, while existing delta correction methods either support only recurring context relations or maintain memory-intensive online correction states for dynamically changing context. For first seen shared context, these methods also construct a reference cache outside the agent workflow, and an approximate correction at the first agent affects the outputs passed to subsequent agents. We present KVCMAS, an online KV cache correction framework that represents cross-agent cache deviations u
    
[^388]: PReCache：通过低秩预计算与中性重构实现多LoRA智能体的高效KV缓存共享

    PReCache: Efficient KV Cache Sharing for Multi-LoRA Agents via Low-Rank Precomputation and Neutral Reconstruction

    [https://arxiv.org/abs/2609.34054](https://arxiv.org/abs/2609.34054)

    PReCache是一个无需训练的KV缓存共享框架，通过共享基于预训练权重计算的基础缓存并预计算紧凑的智能体特定低秩缓存，消除了多LoRA智能体系统中的重复预填充冗余，同时避免了直接缓存复用对角色特定行为的削弱。

    

    多LoRA智能体系统通过共享公共骨干模型实现高效的角色专业化。然而，每个智能体都要反复处理不断增长的共享轨迹并构建自己的KV缓存，在长时程任务中引入了大量的内存和计算冗余。现有的KV缓存共享方法虽能减少这种重复预填充，但它们要么需要额外的训练或架构约束，要么仍保留大量的模型计算。此外，直接复用缓存会导致当前智能体依赖于由前一个智能体适配器生成的缓存状态，从而削弱其自身LoRA所编码的角色特定行为。我们提出了PReCache，这是一个无需训练的KV缓存共享框架，包含两项设计——PreLRShared和ReBaseShared，它们共享使用预训练权重计算的基础缓存，并预先计算紧凑的智能体特定低秩（LR）缓存。为了消除重复预填充，PreLRShared为每个智能体预计算（摘要在此处被截断）

    arXiv:2609.34054v2 Announce Type: replace  Abstract: Multi-LoRA agent systems enable efficient role specialization by sharing a common backbone model. However, each agent repeatedly processes the growing shared trajectory and constructs its own KV cache, introducing substantial memory and computation redundancy in long-horizon tasks. Existing KV cache sharing methods reduce this repeated prefill, but they either require additional training or architectural constraints or retain substantial model computation. Moreover, direct cache reuse causes the current agent to rely on cache states generated by the previous agent's adapter, weakening the role-specific behavior encoded by its own LoRA. We present PReCache, a training-free KV cache sharing framework with two designs, namely PreLRShared and ReBaseShared, that share the base cache computed using the pretrained weights and precompute a compact agent-specific low-rank (LR) cache. To remove repeated prefill, PreLRShared precomputes each ag
    
[^389]: SketchSSM：写入完整状态，从紧凑草图读取

    SketchSSM: Write to the Full State, Read from a Compact Sketch

    [https://arxiv.org/abs/2609.33051](https://arxiv.org/abs/2609.33051)

    SketchSSM在每次状态更新时读取一次完整状态并预计算离线固定基向量的输出存入紧凑草图，从而在保持完整状态写入的同时，让每个解码步骤无需读取完整状态即可近似重构状态读取结果。

    

    混合注意力模型用线性注意力替代大多数softmax注意力层，减少了KV缓存的增长，并支持更大的解码批次，此时循环状态的访问成为主要瓶颈。ReplaySSM通过缓冲键和值来摊销状态更新的开销，但每个新查询仍然需要读取一次完整状态，即使状态在两次状态更新之间并未改变。我们观察到，低秩的状态加权查询近似能够精确地保留状态读取的输出。尽管未来的查询是未知的，但用于近似这些查询的基向量可以离线固定。基于这一观察，我们提出了SketchSSM，它在保持完整状态更新的同时近似状态读取。在每次状态更新时，SketchSSM读取一次完整状态，为这些基向量预计算输出，并将其存储在一个紧凑的草图中。随后的每个解码步骤只需将草图向量与依赖于查询的系数相结合，即可重构出相应的输出。

    arXiv:2609.33051v2 Announce Type: replace  Abstract: Hybrid-attention models replace most softmax attention layers with linear attention, reducing KV-cache growth and enabling larger decode batches where recurrent-state access becomes a major bottleneck. ReplaySSM amortizes state updates by buffering keys and values, but each new query still requires a full-state read even though the state remains unchanged between state updates. We observe that low-rank state-weighted query approximation accurately preserves state-read outputs. Although future queries are unknown, the basis vectors used to approximate them can be fixed offline. Based on this observation, we introduce SketchSSM, which preserves full-state updates while approximating reads. At each state update, SketchSSM reads the full state once to precompute outputs for these basis vectors, storing them in a compact sketch. Each subsequent decode step combines the sketch vectors with query-dependent coefficients to reconstruct the ou
    
[^390]: 面向世界模型的自适应潜在容量

    Adaptive Latent Capacity for World Models

    [https://arxiv.org/abs/2609.32921](https://arxiv.org/abs/2609.32921)

    本文提出ALeWM，一种基于JEPA的世界模型，通过学习前缀长度分布和新的MixSIGReg正则化方法，自适应地将预测信息集中到宽潜在表示的紧凑前缀中，以提升预测与递归规划能力。

    

    我们提出了自适应世界模型（ALeWM），这是一种基于联合嵌入预测架构（JEPA）的世界模型，它学习将预测信息集中在宽潜在表示的紧凑前缀中。为了促进这种排序，ALeWM学习一个以序列为条件的前缀长度分布，并训练预测器从采样的输入前缀来估计完整的下一个嵌入。由于标准的抗坍缩目标只会鼓励潜在坐标之间的变化，而不会按照预测重要性对它们进行组织，我们还引入了MixSIGReg。MixSIGReg对掩码嵌入进行正则化，使其逼近一个先验加权的混合分布，该分布在活跃前缀部分为高斯分布，其余坐标为零。因此，ALeWM的目标促使早期的坐标保留对预测和递归规划有用的信息。我们的分析表明，MixSIGReg所使用的混合分布为较早的坐标分配更高的方差

    arXiv:2609.32921v2 Announce Type: replace  Abstract: We introduce Adaptive LeWorldModel (ALeWM), a world model based on a joint-embedding predictive architecture (JEPA) that learns to concentrate predictive information in compact prefixes of a wide latent representation. To encourage this ordering, ALeWM learns a sequence-conditioned distribution over prefix lengths and trains the predictor to estimate the full next embedding from a sampled input prefix. As standard anti-collapse objectives encourage variation across latent coordinates and do not organize them by predictive importance, we also introduce MixSIGReg. MixSIGReg regularizes the masked embeddings against a prior-weighted mixture with Gaussian active prefixes and zeros in the remaining coordinates. As a result, the ALeWM objective encourages early coordinates to retain information useful for prediction and recursive planning. Our analysis shows that the mixture distribution used by MixSIGReg assigns higher variance to earlier
    
[^391]: 动作塑形：策略会吸收其所能表达的内容

    Action Shaping: Policies Absorb What They Can Express

    [https://arxiv.org/abs/2609.32752](https://arxiv.org/abs/2609.32752)

    论文提出“动作塑形”原理：训练时加入的动作偏移量若能被策略输出层精确表达，策略便会将其完全吸收，从而可在部署时安全移除该偏移而不损失回报，其最简实现为零初始化线性头配合可学习门控，门控会自行先升后降。

    

    奖励塑形有一个定理：基于势函数的修正项可以在不改变最优策略的前提下被移除。然而在动作通道上进行同样的操作——训练时加入一个偏移量、部署时将其去除——却没有相应的定理作为保障。没有任何机制能够抵消动作偏移，因此这种修正要么在部署时被保留，要么在缺乏保证的情况下被移除。我们将其称为动作塑形，并阐述其原理：可训练的策略会吸收其自身输出层能够精确复现的偏移量，这正是我们所称的“表达”；被吸收的偏移量可以在回报完整无损的情况下被移除。其最简实现是一个位于可学习门控之后的零初始化线性头，附加于通过学习到的动作价值函数进行训练的执行器之上，无需任何惩罚项或调度机制。无论对确定性还是随机性执行器，门控值都会自行先升高后回落，并且在20个任务上移除该线性头几乎不造成任何性能损失。其关键条件是精确复现，而非容量大小：一个容量更大的非线性头……

    arXiv:2609.32752v2 Announce Type: replace  Abstract: Reward shaping has a theorem: a potential-based term can be removed without changing the optimal policy. The same practice on the action channel, an offset added in training and dropped at deployment, has no theorem. Nothing cancels an action offset, so the correction is kept at deployment or removed without a guarantee. We call it action shaping and state its principle. A trainable policy absorbs an offset its own output layer can reproduce exactly, which is what we mean by express; what is absorbed can be removed with the return intact. Its minimal instance is a zero-initialized linear head behind a learnable gate, added to an actor that trains through a learned action-value function, with no penalty or schedule. The gate rises and then falls on its own, for deterministic and stochastic actors alike, and on 20 tasks removing the head costs almost nothing. The condition is exact reproduction, not capacity: a nonlinear head with more
    
[^392]: 几何诱导软状态抽象的预测极限与Koopman封闭性

    Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions

    [https://arxiv.org/abs/2609.32652](https://arxiv.org/abs/2609.32652)

    该论文为几何诱导软状态抽象的线性预测精度建立了无需拟合预测矩阵即可由独立评估数据计算的有限样本下置信界，并揭示了核仿射包机器坐标的重构分数裕度与预测误差界之间的联系。

    

    软状态表示为每个状态分配一个非负类别权重向量，且权重之和为一。我们研究这些权重的构造方式与状态动力学如何共同决定线性预测的精度。对于任意固定的可测表示，我们推导出了在具有指定谱范数极限的矩阵类中，最小总体均方根预测误差的有限样本下置信界。该界比较了每个参考类内后继坐标的变化与软输入可能带来的改进。该界可由独立的评估数据对计算得出，无需拟合预测矩阵。若该界高于选定的容差，则可排除整个矩阵类达到该容差的可能性；若该界为零，则结论不确定。对于使用核仿射包机器构造的坐标，重构分数裕度控制了与参考标签的分歧，并进入预测误差的界中。

    arXiv:2609.32652v2 Announce Type: replace  Abstract: A soft state representation assigns each state a vector of nonnegative class weights that sum to one. We study how the construction of these weights and the state dynamics jointly determine the accuracy of linear prediction. For any fixed measurable representation, we derive a finite-sample lower confidence bound on the smallest population root-mean-square prediction error among matrices with a specified spectral-norm limit. The bound compares variation in successor coordinates within each reference class with the improvement that soft inputs could provide. It is computed from independent evaluation pairs without fitting a prediction matrix. A bound above a chosen tolerance rules out that tolerance for the entire matrix class; a zero bound is inconclusive.   For coordinates constructed using Kernel Affine Hull Machines, reconstruction-score margins control disagreement with reference labels and enter bounds on prediction error. Under
    
[^393]: 学习气候模式中强迫对温度影响的分层因果表示

    Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models

    [https://arxiv.org/abs/2609.30995](https://arxiv.org/abs/2609.30995)

    该论文提出一种分层因果表示学习框架，能够显式区分内部气候变率与外部强迫响应，从而准确预测气候变化情景下的温度演变，并提升机器学习气候模拟器的可信度与因果归因能力。

    

    机器学习（ML）模拟器在地球系统模式预估数据上训练后，为模拟气候变化情景提供了一种快速且经济高效的方法。然而，这些数据驱动方法的黑箱特性限制了其输出的可用性和可信度，尤其限制了其作为因果归因工具的使用。在此，我们开发了一个分层因果表示学习框架，并将其应用于最先进的全球气候模式的海表温度场。作为对以往工作的关键进展，我们的框架显式地建模了由内部气候变率引起的大气动力相互作用，以及由大气温室气体和气溶胶浓度变化引起的强迫响应。在未来气候变化情景上进行训练后，我们的方法能够准确预测长期全球平均和区域温度演变，并对温室气体扰动表现出物理上真实的响应。

    arXiv:2609.30995v1 Announce Type: new  Abstract: Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of those data-driven approaches limit the usability and trustworthiness of their outputs and in particular their use as causal attribution tools. Here, we develop a hierarchical causal representation learning framework applied to sea surface temperature fields from a state-of-the-art global climate model. As a key advance over previous work, our framework explicitly models both atmospheric dynamical interactions arising from internal climate variability and forced responses due to changes in atmospheric greenhouse gas and aerosol concentrations. When trained on future climate change scenarios, our method accurately predicts the long-term global mean and regional temperature evolution and shows physically realistic responses to perturbations in g
    
[^394]: KernelOPT：面向GPU内核优化的调度感知智能体搜索

    KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization

    [https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)

    KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。

    

    深度学习的推理与训练性能在很大程度上取决于GPU内核的效率。现代编译器（如PyTorch Inductor）能够从高层模型代码自动生成GPU内核，但其性能常常大幅落后于专家手写的实现。近期基于大语言模型（LLM）辅助的内核优化器虽然能缩小独立内核方面的这一差距，但它们将编译后的模型视为黑盒，通常只优化单个独立内核，既不尊重编译器的结构性决策，也不进行模型级的端到端验证。我们提出了KernelOPT，一个将编译后模型视为结构化产物的多智能体系统。该系统保留厂商库调用（cuBLAS、cuDNN），仅针对生成的Triton子内核，并使用五个由性能剖析引导的LLM智能体进行优化。一个由静态校验、多种子正确性验证、模型级float64回退验证以及性能门控组成的四道验证级联，在优化过程中对候选内核进行筛选（原文摘要在此处截断）。

    arXiv:2609.30059v1 Announce Type: cross  Abstract: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during 
    
[^395]: ELF-REG：将连续扩散语言模型扩展至推理任务

    ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

    [https://arxiv.org/abs/2609.29102](https://arxiv.org/abs/2609.29102)

    提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。

    

    全连续扩散语言模型（dLMs）对连续表示进行去噪而无需中间离散化，并在最后一步并行解码所有响应token。它们在具有挑战性的推理任务上的性能表现，尚未像自回归（AR）大语言模型和掩码扩散语言模型那样得到充分验证。我们将嵌入式语言流（ELF）扩展到GSM8K、MATH-500、HumanEval和MBPP上的数学推理与代码生成任务。我们提出了ELF-REG，它通过表示对齐与纠缠（REPA+REG）来改进学习，其中冻结的AR教师模型监督中间去噪器特征，并提供一个与响应联合去噪的全局表示。ELF-REG-L在64次网络函数评估（NFE）下于GSM8K上达到55.96%的pass@1，在128次NFE下于MATH-500上达到13.39%、HumanEval上达到22.56%。它在GSM8K和代码任务上的pass@1优于所评估的同规模扩散语言模型，并将MATH-500的pass@1从……（摘要原文在此处被截断）

    arXiv:2609.29102v1 Announce Type: new  Abstract: Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and masked dLMs. We scale Embedded Language Flows (ELF) to mathematical reasoning and code generation on GSM8K, MATH-500, HumanEval, and MBPP. We introduce ELF-REG, which improves learning with representation alignment and entanglement (REPA+REG), where a frozen AR teacher supervises intermediate denoiser features and supplies a global representation that is jointly denoised with the response. ELF-REG-L achieves 55.96% pass@1 on GSM8K at 64 network function evaluations (NFE), and 13.39% on MATH-500 and 22.56% on HumanEval at 128 NFE. It outperforms the evaluated comparable-scale dLMs in pass@1 on GSM8K and code, and improves MATH-500 pass@1 from 
    
[^396]: ValueDiff：面向注意力汇聚抑制型大语言模型的价值几何KV缓存淘汰方法

    ValueDiff: Value-Geometric KV Cache Eviction for Sink-Suppressed LLMs

    [https://arxiv.org/abs/2609.23314](https://arxiv.org/abs/2609.23314)

    针对因QK归一化等技术导致注意力汇聚减弱的现代大语言模型，提出基于价值向量与缓存均值L2偏差进行token排序的ValueDiff淘汰方法，在2k-4k token预算下可保留密集注意力88-99%的性能。

    

    采用QK归一化、门控注意力、可学习注意力汇聚或logit软截断的现代大语言模型表现出更弱的持久注意力汇聚现象，而现有的KV缓存淘汰方法主要依赖这一现象。我们观察到，在这些模型中，更弱的汇聚现象伴随着相对于键向量离散度而言更大的价值向量离散度。受这种价值侧离散度的启发，我们提出了ValueDiff，一种价值几何淘汰方法，它根据token的价值向量与缓存均值之间的L2偏差对token进行排序。在关于未来注意力的最大熵假设下，同样的分数可作为最小扰动淘汰方案推导得出。我们在固定缓存预算下进行评估，在预填充阶段的每个块边界以及生成阶段的每个解码步骤执行淘汰。在RULER基准上，在紧凑的2k token预算下，ValueDiff在七个汇聚抑制模型上保留了密集注意力88--99%的性能（在7个模型中的6个上表现最佳）。在LongBench基准上，在4k预算下，ValueDiff平均保留92%的性能。

    arXiv:2609.23314v1 Announce Type: new  Abstract: Modern LLMs with QK-normalization, gated attention, learned attention sinks, or logit softcapping exhibit weaker persistent attention sinks, on which existing KV cache eviction methods primarily rely. We observe that across these models, weaker sinks co-occur with greater value-vector dispersion relative to key-vector dispersion. Motivated by this value-side dispersion, we present ValueDiff, a value-geometric eviction that ranks tokens by the L2 deviation of their value vectors from the cache mean. The same score arises as the minimal-disturbance eviction under a max-entropy assumption about future attention. We evaluate under fixed cache budgets, with eviction at every block boundary during prefill and at every decoding step during generation. On RULER at a tight 2k token budget, ValueDiff retains 88--99\% of dense across seven sink-suppressed models (best on 6 out of 7). On LongBench at the 4k budget, ValueDiff averages 92\% retention 
    
[^397]: 掩码离散扩散中tau-leaping的调度优化

    Schedule optimization for tau-leaping in masked discrete diffusion

    [https://arxiv.org/abs/2609.21960](https://arxiv.org/abs/2609.21960)

    该论文通过依赖密度ρ的精确积分表示来刻画掩码离散扩散中tau-leaping采样的因式分解误差，并推导有限步优化问题的递归平稳性方程，从而实现去噪调度的优化。

    

    掩码离散扩散模型通常使用所谓的tau-leaping离散化方法来加速，该方法在每个采样步骤中并行揭示多个坐标。采样器用乘积分布替代每个被揭示块的联合条件分布，从而产生因式分解误差 $\varepsilon_\text{fact}$，即使预测器被完美学习，该误差依然存在。我们分析了在 $N$ 个坐标上进行 $K$ 步采样的标准采样器，其随机块大小取决于去噪调度。我们的分析使用 $\varepsilon_\text{fact}$ 的精确积分表示，该表示以依赖于分布的依赖密度 $\rho$ 来刻画，$\rho$ 记录了随着坐标被揭示比例的增长，条件依赖如何演化。我们为该依赖轮廓开发了估计器，并量化了估计误差如何影响调度选择。我们为有限 $K$ 的优化问题推导了递归平稳性方程，并……（原文在此处截断）

    arXiv:2609.21960v1 Announce Type: cross  Abstract: Masked discrete diffusion models are commonly accelerated using the so-called tau-leaping discretization method, which reveals several coordinates in parallel at each sampling step. The sampler replaces the joint conditional law of each revealed block by a product distribution, incurring a factorization error $\varepsilon_\text{fact}$ present even with perfectly learned predictors. We analyze the standard sampler on $N$ coordinates with $K$ sampling steps, whose random block sizes depend on a denoising schedule. Our analysis uses an exact integral representation of $\varepsilon_\text{fact}$ in terms of a distribution-dependent dependence density $\rho$, which records how conditional dependence evolves as the revealed fraction of coordinates grows. We develop estimators for this profile and quantify how estimation errors affect schedule selection. We derive recursive stationarity equations for the finite-$K$ optimization problem and, un
    
[^398]: 局部稀疏性实现无监督的大语言模型安全检测

    Local Sparsity Enables Unsupervised LLM Safety Detection

    [https://arxiv.org/abs/2609.20129](https://arxiv.org/abs/2609.20129)

    本文利用稀疏自编码器概念空间中的局部稀疏性这一关键洞察，提出了一个无需不安全训练数据的无监督LLM安全异常检测框架，并有理论支撑。

    

    大语言模型（LLM）的部署时安全方法主要是有监督的，并假设能够获取不安全的训练数据。然而，新的攻击和伤害类别不断出现，而以这种有监督方式训练的模型无法捕获这些内容。另一种方法是从异常检测的视角来看待这个问题，即仅依赖于对安全数据的建模并标记分布外输入。然而，LLM激活位于高维空间中，这引发了关于异常检测在统计上是否可行的担忧。我们证明，在线性表示假设（LRH）下，确实存在希望。在通常通过稀疏自编码器（SAE）恢复的LRH概念空间中，邻近的点共享一个较小的共同激活支持集。利用这一局部稀疏性洞察，我们提出了一个基于局部掩码SAE的异常检测框架，并提供了理论依据的支持。我们进行了验证……

    arXiv:2609.20129v1 Announce Type: cross  Abstract: Deployment-time safety methods for large language models (LLMs) are predominantly supervised and assume access to unsafe training data. Nevertheless, new attacks and harm categories regularly arise, not captured by models trained in such a supervised fashion. An alternative approach is to view this problem through the lens of anomaly detection, namely, to rely solely on modeling safe data and flagging out-of-distribution inputs. However, LLM activations lie in a high-dimensional space, raising concerns about whether anomaly detection is statistically feasible. We show that, under the linear representation hypothesis (LRH), there may indeed be hope. In the LRH concept space, which is typically recovered via a sparse autoencoder (SAE), nearby points share a small common active support. Using this local sparsity insight, we propose a framework for locally masked SAE-based anomaly detection, supported by theoretical justifications. We vali
    
[^399]: FedGuide：面向异构联邦强化学习的扩散先验对齐与价值基线引导

    FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning

    [https://arxiv.org/abs/2609.18964](https://arxiv.org/abs/2609.18964)

    该论文提出FedGuide框架，通过扩散先验作为行为模型并利用最优传输混合专家进行聚合，解决了异构联邦强化学习中客户端间的分布不匹配问题，同时以DICE价值基线提供低方差的回报感知引导。

    

    联邦强化学习（FRL）使分布式智能体能够在异构环境中进行协同策略学习。尽管近期基于方差缩减、散度惩罚和动量优化的方法改进了异构设置下的FRL，但这些方法仍主要同步策略或价值网络参数，并未明确解决异构客户端之间的分布不匹配问题。因此，我们提出了**FedGuide**，一个使用扩散先验作为行为模型的FRL框架，为异构的本地策略学习提供由个性化数据支持的分布。FedGuide不直接对本地策略进行平均，而是通过最优传输混合专家（OT-MoE）聚合这些扩散先验，在分布空间中保留异构行为模式。此外，它还开发了分布校正估计（DICE）价值基线，以提供低方差、具有回报感知的引导。

    arXiv:2609.18964v1 Announce Type: new  Abstract: Federated Reinforcement Learning (FRL) enables collaborative policy learning across distributed agents with heterogeneous environments. While recent methods based on variance reduction, divergence penalization, and momentum optimization improve FRL under heterogeneous settings, they still primarily synchronize policy or value-network parameters and do not explicitly address distributional mismatch among heterogeneous clients. Therefore, we propose \textbf{FedGuide}, a FRL framework that uses diffusion priors as behavior models to provide personalized data supported distributions for heterogeneous local policy learning. Instead of directly averaging local policies, FedGuide aggregates those diffusion priors through Optimal-Transport Mixture-of-Experts (OT-MoE), preserving heterogeneous behavior modes in distribution space. It further develops a Distribution Correction Estimation (DICE) value baseline to provide low-variance, return-aware 
    
[^400]: SpliTEE：通过差分隐私GPU外包改进可信硬件上的大语言模型推理

    SpliTEE: Improving LLM Inference on Trusted Hardware with Differentially Private GPU Outsourcing

    [https://arxiv.org/abs/2609.15039](https://arxiv.org/abs/2609.15039)

    该论文提出SpliTEE，将拆分推理架构扩展到LLM推理场景，用差分隐私（而非加密）保护发送到不受信任GPU的中间表示，从而在可信硬件上实现高效且隐私安全的大语言模型推理。

    

    用户向大语言模型（LLM）提供的提示词可能包含敏感或私密信息，这些信息可能被远程部署的模型滥用，例如在重新训练过程中被无意记忆。保护用户提示词的一种方法是在可信执行环境（TEE）内执行LLM，以保证服务提供商无法访问TEE内执行的计算或与TEE交换的信息。然而，当前的TEE主要基于CPU，比针对LLM推理优化的GPU慢得多。为了解决这一问题，Tramer和Boneh（2019）提出了Slalom，该方法将神经网络推理在TEE和不受信任的GPU之间进行拆分，并对发送到GPU的中间输入进行加密。我们将这种拆分推理架构扩展到LLM推理，并改用差分隐私来保护中间输入。我们通过展示提示词（摘要在此处被截断）证明了掩盖中间表示是必要的。

    arXiv:2609.15039v1 Announce Type: cross  Abstract: User prompts provided to large language models (LLMs) may contain sensitive or private information that can be misused by remotely deployed models, such as through inadvertent memorization during retraining. One way to protect user prompts is to execute the LLM inside a trusted execution environment (TEE), with the guarantee that the service provider has no access to computations performed within or information exchanged with the TEE. However, current TEEs are primarily CPU-based and significantly slower than GPUs optimized for LLM inference. To circumvent this, Tramer and Boneh (2019) proposed Slalom, which splits neural network inference between a TEE and an untrusted GPU and encrypts intermediate inputs sent to the GPU. We extend this split-inference architecture to LLM inference and instead protect intermediate inputs using differential privacy. We show that masking intermediate representations is necessary by showing that a prompt
    
[^401]: VoxReason：合成前基于源记录的语音规划的无听者评估

    VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis

    [https://arxiv.org/abs/2609.03203](https://arxiv.org/abs/2609.03203)

    VoxReason提出了一种无需听者参与的评估任务，在语音合成之前通过带证据引用的说话计划和确定性验证器，衡量语音表达方式的选择是否真正建立在被引用的源记录之上。

    

    表现力语音系统在任何波形被渲染之前就必须做出一个决定：一句话语将以何种方式被表达。在对话智能体、旁白叙述和角色条件TTS中，这一隐藏的规划步骤决定了情感、音高、能量、语速、停顿、重音和立场，然而下游音频评分很少能揭示这些选择是否由源记录所支持——这是一种在任何波形存在之前就发生的源使用失败。VoxReason将这一合成前的决策转化为可度量的、无需听者参与的任务，用于评估基于源记录的语音规划。在合成之前，VoxReason衡量话语表达方式的选择是否有被引用的源记录作为依据。系统输出带有证据引用的、注明来源的说话计划，随后一个确定性验证器检查引用合法性、槽位一致性、无支持状态、模式有效性以及单线索反事实局部性。在1,440个经过检查的源标签案例上，捷径控制实验表明了为什么仅凭槽位准确率是不安全的：一个简单的键值查找……（原文摘要在此处截断）

    arXiv:2609.03203v1 Announce Type: cross  Abstract: Expressive speech systems make a decision before any waveform is rendered: how an utterance is delivered. In dialogue agents, narration, and role-conditioned TTS, that hidden planning step sets affect, pitch, energy, rate, pause, emphasis, and stance, yet downstream audio scores rarely reveal whether those choices were licensed by the source record, a source-use failure that occurs before any waveform exists. VoxReason makes that pre-synthesis decision measurable as a listener-free task for source-grounded speech planning. Before synthesis, VoxReason measures whether delivery choices are grounded in cited source records. Systems output a source-cited speaking-plan with evidence citations, and a deterministic verifier checks citation legality, slot agreement, unsupported state, schema validity, and one-cue counterfactual locality. On 1,440 checked source-label cases, shortcut controls show why slot accuracy alone is unsafe: a key-lookup
    
[^402]: “经典训练、量子部署”范式需要重新思考泛化问题

    "Train classical, deploy quantum" requires rethinking generalization

    [https://arxiv.org/abs/2608.31117](https://arxiv.org/abs/2608.31117)

    该论文指出，“在经典计算机上训练、在量子设备上部署”的量子生成模型范式存在根本性问题——即便使用经典可评估的损失函数（如基于Pauli-Z关联的MMD²）成功完成训练，也必须重新审视其泛化能力，因为最小化这类经典目标未必能保证量子部署阶段生成样本的质量。

    

    生成式模型已成为科学和工业领域的核心，应用范围从图像和文本合成到分子与材料设计。量子生成式模型被认为是量子计算机最有前景的应用之一，因为量子电路天然地产生其所编码分布的样本，而且对于合适的电路，该分布被认为对任何经典计算机而言都难以复现。一种主流策略是在经典计算机上训练这些模型，而将量子设备保留到部署阶段用于生成样本。当训练损失可以在经典计算机上评估时，这种策略是可行的。一个典型的例子是最大均值差异（MMD²），这是一种矩匹配损失，通过Pauli-Z关联来比较模型分布与数据分布。迄今为止的研究主要关注此类模型能否被训练以及其采样是否困难；而最小化这样的目标函数是否能带来……（摘要原文在此处被截断）

    arXiv:2608.31117v1 Announce Type: cross  Abstract: Generative models have become central across science and industry, from image and text synthesis to the design of molecules and materials. Quantum generative models are considered one of the most promising applications for quantum computers, since a quantum circuit naturally produces samples from the distribution it encodes, and for suitable circuits that distribution is believed to be hard for any classical computer to reproduce. A leading strategy trains these models on a classical computer and reserves the quantum device for generating samples at deployment. This is possible when the training loss can be evaluated on a classical computer. A prime example is the maximum mean discrepancy (MMD$^2$), a moment-matching loss that compares the model and the data through their Pauli-$Z$ correlations. Research so far has asked whether such models can be trained and whether their sampling is hard; whether minimizing such an objective yields a
    
[^403]: 基于自回归扩散的3D场景图高级概念生成

    Generation of High-Level Concepts in 3D Scene Graphs via Autoregressive Diffusion

    [https://arxiv.org/abs/2608.28733](https://arxiv.org/abs/2608.28733)

    提出了一种统一的自回归扩散图生成模型，联合学习图结构与节点特征，可从观测平面自底向上构建跨越任意层次深度的完整3D场景图，并在合成与真实建筑数据集上全面超越现有基线方法。

    

    室内3D场景图（3DSG）将环境表示为多层层次结构，将观测到的几何图元（例如平面）与更高层次的度量-语义概念（例如房间、楼层、建筑）连接起来，从而为机器人感知与SLAM实现增量式空间推理。然而，经典的高级概念生成方法依赖于针对特定概念类别的手工规则，而基于学习的方法则需要为图结构和空间节点特征（例如质心）分别建立独立模型，这限制了对新类别和更复杂层次结构的可扩展性。我们提出了一种统一的自回归扩散图生成模型，能够联合学习图结构与节点特征，从观测到的垂直平面出发，自底向上构建跨越任意层次深度的完整3D场景图。我们的方法在涵盖合成场景与真实建筑的3DSG数据集上，始终超越所有基于学习的基线和随机基线。

    arXiv:2608.28733v1 Announce Type: cross  Abstract: Indoor 3D Scene Graphs (3DSGs) represent environments as multi-layer hierarchies that connect observed geometric primitives (e.g., planes) to higher-level metric-semantic concepts (e.g., rooms, floors, buildings), enabling incremental spatial reasoning for robotic perception and SLAM. However, classical high-level concept generation approaches rely on hand-crafted rules for specific concept classes, while learning-based methods require separate models for graph structure and spatial node features (e.g., centroids), which limits scalability to novel classes and more complex hierarchies. We propose a unified autoregressive diffusion-based graph generative model that jointly learns structure and features, constructing complete 3DSGs bottom-up from observed vertical planes across arbitrary hierarchy depths. Our method consistently surpasses all learning-based and random baselines across 3DSG datasets spanning synthetic scenes, real archite
    
[^404]: 超越普氏距离：一种捕捉手性的多线性Gromov-Wasserstein距离

    Beyond Procrustes distances: a multilinear Gromov-Wasserstein distance capturing chirality

    [https://arxiv.org/abs/2608.27774](https://arxiv.org/abs/2608.27774)

    该论文提出了Gromov-Wasserstein目标的多线性推广，由此定义出对手性敏感的手性Gromov-Wasserstein（CGW）距离，能够区分形状与其镜像，并具备鲁棒性保证和高效的计算算法。

    

    高效且鲁棒地分析形状数据在许多科学学科中至关重要。尽管手性是众多应用中的一项基本性质——尤其是在分子科学中——但现有的形状分析度量无法区分一个形状与其镜像。为弥补这一空白，我们引入了Gromov-Wasserstein目标的多线性推广。在温和的假设下，该目标给出了形状之间的距离，其中形状被表示为对对称群 $G$ 取商后的概率分布。特别地，当 $G = SO(d)$ 时，我们引入了对手性敏感的手性Gromov-Wasserstein（CGW）距离。我们建立了多线性Gromov-Wasserstein距离的鲁棒性性质，并通过将耦合投影到低维空间来重新表述底层优化问题，进而开发了计算这些距离的高效算法。我们推导了局部与近似……（摘要在此处截断）

    arXiv:2608.27774v1 Announce Type: cross  Abstract: Efficiently and robustly analyzing shape data is critical across many scientific disciplines. While chirality is a fundamental property in numerous applications - most notably in molecular science - existing shape analysis metrics fail to distinguish between a shape and its mirror image. To address this gap, we introduce a multilinear generalization of the Gromov-Wasserstein objective. Under mild assumptions, this objective yields a distance between shapes, represented as probability distributions quotiented by a symmetry group $G$. In particular, for $G = SO(d)$, we introduce the Chiral Gromov-Wasserstein ($\mathrm{CGW}$) distance, sensitive to chirality. We establish robustness properties for the multilinear Gromov-Wasserstein distances and develop efficient algorithms to compute them, reformulating the underlying optimization problem by projecting couplings onto a low-dimensional space. We derive algorithms for both local and approx
    
[^405]: 何时锐利协方差包络是紧的？体积采样最小二乘的仅特征几何

    When Is the Sharp Covariance Envelope Tight? Feature-Only Geometry for Volume-Sampled Least Squares

    [https://arxiv.org/abs/2608.26877](https://arxiv.org/abs/2608.26877)

    本文建立了体积采样最小二乘中心系数协方差的Loewner包络，并揭示了仅特征边际nu_A决定谱包络严格性的精确条件。

    

    arXiv:2608.26877v1 公告类型：新 摘要：Derezinski和Warmuth的先前分析建立了普通体积采样的全尺寸采样恒等式、选定OLS无偏性和逆矩，而他们精确的任意固定响应损失和预测协方差公式位于秩-尺寸端点s=d。我们针对每个满秩固定池、响应和合法预算d <= s <= m，在普通索引固定尺寸体积采样后进行选定未加权最小二乘的情况下，建立了中心系数协方差的Loewner包络；其系数在满秩类上全局锐利。全局锐利性并不决定在当前池上的可达性。在正损失、严格内部预算和无共环条件下，一个仅特征边际nu_A给出了精确的固定设计谱相位：nu_A > 0当且仅当归一化谱包络对每个兼容残差是严格的，而nu_A = 0当且仅当某个兼容残差是谱紧的。

    arXiv:2608.26877v1 Announce Type: new  Abstract: Prior analyses by Derezinski and Warmuth established all-size sampling identities, selected-OLS unbiasedness, and inverse moments for ordinary volume sampling, while their exact arbitrary-fixed-response loss and prediction-covariance formulas are at the rank-size endpoint s=d. We establish a Loewner envelope for centered coefficient covariance for every full-rank fixed pool, response, and legal budget d <= s <= m under ordinary indexed fixed-size volume sampling followed by selected unweighted least squares; its coefficient is globally sharp over the full-rank class. Global sharpness does not determine attainability on the pool in hand. Under positive loss, strict-interior budgets, and no coloops, a feature-only margin nu_A gives the exact fixed-design spectral phase: nu_A > 0 if and only if the normalized spectral envelope is strict for every compatible residual, whereas nu_A = 0 if and only if some compatible residual is spectrally tig
    
[^406]: 潜在诊断分类法：一种构建分类器并诊断其决策的框架，应用于提示注入检测

    The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection

    [https://arxiv.org/abs/2608.26423](https://arxiv.org/abs/2608.26423)

    本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。

    

    arXiv:2608.26423v1 公告类型：交叉 摘要：本文提出了一种框架，用于构建作为防护层的分类器，并开发一种互补的诊断方法，以识别分类器的哪些自信决策可以被信任。该框架，即潜在诊断分类法，包括：（i）构建一个维度优化的分类器，其中嵌入维度通过交叉验证性能经验性选择，而非预先固定；（ii）定位一个相对较小的潜在支持向量集（约占训练示例总数的29%），代表有影响力的提示，用于识别改变分类器预测标签的令牌；（iii）利用这些令牌及其相关的攻击幅度来构建诊断分类法。该诊断分类法为标记需要不同处理的提示提供了端到端的指南：安全地依赖分类器的决策；标记启发式偏差和启发式过拟合。

    arXiv:2608.26423v1 Announce Type: cross  Abstract: This paper proposes a framework for constructing a classifier as a safeguard layer, and for developing a complementary diagnostic that identifies which of the classifier's confident decisions can be trusted. This framework, the Latent Diagnostic Taxonomy, consists of (i) constructing a dimensionality-optimized classifier, in which the embedding dimensionality is empirically selected via cross-validated performance rather than fixed a priori, (ii) locating a relatively small set of latent support vectors (~ 29% of total training examples) representing influential prompts for identifying tokens that alter the classifier's predicted labels, and (iii) utilizing such tokens and their associated attack magnitudes for constructing a diagnostic taxonomy. This diagnostic taxonomy provides an end-to-end guideline for flagging prompts that require different treatments: rely Safely on the classifier's decision; flag Heuristic Bias and Heuristic Ov
    
[^407]: 泛化之前的通道化：将“顿悟”现象作为动力学探针

    Canalization Before Generalization: Grokking as a Dynamical Probe

    [https://arxiv.org/abs/2608.25813](https://arxiv.org/abs/2608.25813)

    本文通过权重衰减脉冲在“顿悟”平台期揭示了解选择的通道化过程，表明在可见泛化之前就形成了稳定的剂量排序，为理解神经网络泛化机制提供了新的动力学视角。

    

    对于过参数化的神经网络，许多解能同样好地拟合训练数据，但在未见样本上的行为却大相径庭。“顿悟”现象将训练拟合与可见泛化分离，为研究训练过程中这种选择如何发展提供了一个窗口。我们在这一平台期扫描短时、固定持续时间的权重衰减脉冲，并测量它们如何改变后续的泛化时间。在三个“顿悟”任务中，这些变化在平台期早期是无序的，但后来形成稳定的剂量排序，即更强的权重衰减增加导致更早的泛化，更强的权重衰减减少导致更晚的泛化。这种排序在所有三个任务中都在可见泛化之前出现。同时，扰动与基线泛化检查点之间的测试损失障碍趋向于零，而有序的时间效应仍然持续。我们将这种日益受限的解选择与持久的时间效应组合称为“通道化”，并探讨其作为训练动力学探针的意义。

    arXiv:2608.25813v1 Announce Type: new  Abstract: For overparameterized neural networks, many solutions can fit the training data equally well while behaving very differently on unseen samples. Grokking separates training fit from visible generalization, providing a window for studying how this selection develops during training. We sweep short, fixed-duration weight-decay (WD) pulses across this plateau and measure how they shift later generalization time. Across three grokking tasks, these shifts are unordered early in the plateau but later form a stable dose ordering, with stronger WD increases leading to earlier generalization and stronger WD decreases leading to later generalization. This ordering emerges before visible generalization in all three tasks. Meanwhile, test-loss barriers between perturbed and baseline generalization checkpoints collapse toward zero while the ordered timing effects persist. We call this combination of increasingly constrained solution selection and pers
    
[^408]: 当相似性由交互驱动时：面向机制敏感学习的量子核方法

    When Similarity Is Interaction-Driven: Quantum Kernels for Regime-Sensitive Learning

    [https://arxiv.org/abs/2608.24631](https://arxiv.org/abs/2608.24631)

    该论文提出了一种交互驱动的量子核方法，通过纠缠泡利串特征映射显式编码高阶稀疏交互，在机制敏感学习中显著优于传统核方法。

    

    arXiv:2608.24631v1 公告类型：交叉 摘要：在许多决策系统中，相似性不仅由距离决定，还受变量间交互作用的影响。在欺诈和异常检测中，小的局部扰动可能跨越交互敏感的决策边界，而环境距离几乎不变。受此场景启发，我们引入了一种薄板交互模型和一种由纠缠泡利串特征映射构建的交互驱动量子核。该特征映射显式编码稀疏的高阶块交互。我们证明所得保真度核是正半定的，具有精确的块因子化形式，并诱导对交互机制变化敏感的几何结构。在涵盖三阶、四阶、六阶和八阶交互的平衡与非平衡合成实验中，所提出的核始终优于线性、径向基函数、拉普拉斯和高斯核，以及工程化交互核。

    arXiv:2608.24631v1 Announce Type: cross  Abstract: Similarity in many decision systems is governed not by distance alone but by interactions among variables. In fraud and anomaly detection, small local perturbations can cross interaction-sensitive decision boundaries while leaving ambient distance almost unchanged. Motivated by this setting, we introduce a thin-slab interaction model and an interaction-driven quantum kernel constructed from entangled Pauli-string feature maps. The feature map explicitly encodes sparse high-order block interactions. We show that the resulting fidelity kernel is positive semidefinite, admits an exact block-factorized formulation, and induces a geometry sensitive to changes in interaction regime. Across balanced and imbalanced synthetic experiments spanning third-, fourth-, sixth-, and eighth-order interactions, the proposed kernel consistently outperforms linear, radial basis function, Laplacian, and polynomial kernels, as well as an engineered-interacti
    
[^409]: 对时空预测基准数据集和基线的批判性审计

    A Critical Audit of Spatiotemporal Forecasting Benchmark Datasets and Baselines

    [https://arxiv.org/abs/2608.20980](https://arxiv.org/abs/2608.20980)

    本文通过经典时间序列方法分析常用时空基准数据集，揭示无空间感知的线性模型比以往报告更具竞争力，质疑了现有基准数据集的判别可靠性。

    

    arXiv:2608.20980v1 公告类型：新 摘要：图神经网络（GNNs）通常用于具有空间图结构的多元时间序列的短期预测。尽管存在许多替代数据集，但该领域的方法创新主要针对一组有限的基准数据集进行评估，最著名的是Chickenpox、PedalMe、WikiMaths、METR-LA和PEMS-BAY。评估协议包含从历史平均值到经典机器学习方法的基线。这些基线通常表现出与GNNs相当的性能。在本研究中，我们退一步，通过经典时间序列方法分析基准数据集，以揭示为什么无空间感知的线性模型比先前报道的更具竞争力，从而进一步质疑上述广泛采用的数据集的判别可靠性。我们的统计分析提供了一套工具集，用于识别显著的...

    arXiv:2608.20980v1 Announce Type: new  Abstract: Graph neural networks (GNNs) are routinely employed for short-range forecasting on multivariate time series with a spatial graph structure. Despite the availability of many alternative datasets, method innovations within this domain are predominantly assessed against a rather limited set of benchmark datasets, most notably Chickenpox, PedalMe, WikiMaths, METR-LA, and PEMS-BAY. The evaluation protocols contain baselines spanning from historical averages to classical machine learning approaches. These baselines often show competitive performance compared to GNNs. In the present work, we take a step back and analyse the benchmark datasets via classical time series methods to uncover why spatially-unaware linear models pose a stronger competitor than previously reported, casting further doubt on the discriminative reliability of the aforementioned widely adopted datasets. Our statistical analysis provides a toolset for identifying significan
    
[^410]: 面向区域供热网络零样本概率热负荷预测的TabPFN-TS系统评估

    Systematic Evaluation of TabPFN-TS for Zero-Shot Probabilistic Heat Load Forecasting in District Heating Networks

    [https://arxiv.org/abs/2608.20024](https://arxiv.org/abs/2608.20024)

    本研究首次系统评估了基于合成数据预训练的TabPFN-TS模型在区域供热网络零样本概率热负荷预测中的性能，探讨了其与真实时间序列基础模型相比的优劣及对供热动态的捕捉能力。

    

    区域供热能源枢纽需要可靠的热负荷预测以实现高效的运行调度。传统的预测工作流程会在历史数据上训练特定于系统的模型，当网络因新用户、改造或运行工况变化而发生改变时，这种方法可能变得繁琐。零样本时间序列基础模型和上下文内预测提供了一种有前景的替代方案：它们可以在推理时根据最近的观测数据进行适应，而无需重复训练。本研究系统评估了TabPFN-TS在区域供热网络概率热负荷预测中相对于时间序列基础模型和训练过的机器学习基线的表现。与在大规模真实时间序列集合上预训练的基础模型不同，TabPFN-TS依赖于合成预训练数据，这避免了直接的预训练-测试重叠，但引发了一个问题：学习到的先验是否能捕捉区域供热的动态特性。我们a

    arXiv:2608.20024v1 Announce Type: new  Abstract: District heating energy hubs require reliable heat load forecasts for efficient operational scheduling. Conventional forecasting workflows train system-specific models on historical data, which can become burdensome when networks change through new consumers, retrofits, or changing operating regimes. Zero-shot time-series foundation models and in-context forecasting offer a promising alternative: they can adapt at inference time from recent observations rather than by repeated retraining. This study systematically evaluates TabPFN-TS against time-series foundation models and trained machine-learning baselines for probabilistic heat load forecasting in district heating networks. Unlike foundation models pretrained on large collections of real time series, TabPFN-TS relies on synthetic pretraining data, which avoids direct pretraining-test overlap but raises the question of whether the learned prior captures district heating dynamics. We a
    
[^411]: RIPE++：仅从正样本对中强化关键点学习

    RIPE++: Reinforced Keypoint Learning from Positive Pairs Only

    [https://arxiv.org/abs/2608.19693](https://arxiv.org/abs/2608.19693)

    本文提出RIPE++，通过从单个正样本对中同时提取奖励和惩罚，避免了负样本对比，从而提升了关键点学习的稳定性和描述符判别性。

    

    arXiv:2608.19693v1 公告类型：交叉 摘要：稀疏关键点提取与匹配支撑了几何计算机视觉中的核心任务，包括运动恢复结构、视觉SLAM、增强现实和医学图像配准。然而，学习鲁棒的局部特征表示通常需要准确的相机位姿或深度监督，这些在现实场景中往往不可用。强化学习（RL）近期成为一种有前景的替代方案，仅需两张图像是否显示相同场景的信息。然而，现有的RL公式（如RIPE）依赖于粗略的二元奖励和精心构造的负样本对，限制了训练稳定性和描述符判别性。在本文中，我们重新审视基于RL的关键点学习，并提出一种充分利用几何一致性信号的奖励，从单个正样本对中同时推导奖励和惩罚，无需与负样本对比。这种更丰富的信号提供了...

    arXiv:2608.19693v1 Announce Type: cross  Abstract: Sparse keypoint extraction and matching underpin core tasks in geometric computer vision, including structure-from-motion, visual SLAM, augmented reality, and medical image registration. Learning robust local feature representations, however, typically requires accurate camera poses or depth supervision, which are often unavailable in real-world settings. Reinforcement learning (RL) has recently emerged as a promising alternative, requiring only the information if two images show the same scene or not. However, existing RL formulations such as RIPE rely on coarse binary rewards and carefully constructed negative training pairs, limiting training stability and descriptor discriminability. In this paper, we revisit RL-based keypoint learning and propose a reward that fully exploits the geometric consistency signal, deriving both reward and penalty from a single positive pair without contrasting against negatives. This richer signal provi
    
[^412]: 增强基于距离的图自编码器与结构惩罚用于动态图嵌入

    Enhancing Distance-Based Graph Autoencoders with Structural Penalties for Dynamic Graph Embedding

    [https://arxiv.org/abs/2608.18762](https://arxiv.org/abs/2608.18762)

    本文提出三种基于距离的图自编码器变体，通过引入枢纽惩罚和NC-LID结构惩罚，增强了动态图嵌入对结构模糊节点的重构准确性。

    

    arXiv:2608.18762v1 公告类型：新 摘要：图自编码器（GAEs）广泛用于学习动态图的表示。然而，它们的优化目标通常不考虑节点间的结构异质性。我们提出了三种基于距离的GAE变体，将结构惩罚纳入重构损失中。所有变体共享一个两层图卷积网络编码器和一个使用基于距离的重构目标训练的欧几里得距离解码器。我们扩展了稀疏校正损失，并添加了两个节点级正则化项：（i）基于度中心性的枢纽惩罚，和（ii）基于自然社区局部内在维度（NC-LID）的惩罚。本文受到先前证据的启发，这些证据将高NC-LID与降低的嵌入质量联系起来。所提出的方法旨在强调结构模糊节点的重构误差。在多个动态图数据集上的实验表明，纳入NC-LID惩罚显著提高了嵌入性能。

    arXiv:2608.18762v1 Announce Type: new  Abstract: Graph autoencoders (GAEs) are widely used for learning representations of dynamic graphs. However, their optimisation objectives typically do not take structural heterogeneity across nodes into account. We propose three distance-based GAE variants that incorporate structural penalties into the reconstruction loss. All variants share a two-layer Graph Convolutional Network encoder and a Euclidean-distance decoder trained with distance-based reconstruction objectives. We extend sparsity-corrected loss with two node-level regularization terms: (i) a hub penalty based on degree centrality, and (ii) a penalty based on Natural Community Local Intrinsic Dimensionality (NC-LID). The paper is motivated by prior evidence linking high NC-LID to reduced embedding quality. The proposed methods are designed to emphasize reconstruction errors for structurally ambiguous nodes. Experiments on multiple dynamic graph data sets show that incorporating NC-LI
    
[^413]: PertMind：通过细胞扰动数据上的强化学习激发大语言模型中的涌现生物推理

    PertMind: Eliciting Emergent Biological Reasoning in LLM via Reinforcement Learning on Cellular Perturbation Data

    [https://arxiv.org/abs/2608.16419](https://arxiv.org/abs/2608.16419)

    PertMind通过将细胞扰动图谱转化为强化学习环境，仅用正向预测训练便激发了大语言模型的涌现生物推理能力，并实现了跨任务的零样本迁移。

    

    大语言模型能够描述机制，但其可扩展的后训练仍依赖于昂贵且人工整理的生物推理轨迹。在此，我们展示了细胞扰动图谱可以转变为强化学习环境，其中测量的基因响应为生物推理提供可计算的奖励。我们引入了PertMind，它结合了可信轨迹监督初始化与基因、通路和格式层面的强化信号。仅通过正向扰动-响应预测训练，PertMind在未见细胞情境中改善了响应推断，同时保留了通用语言能力。它还无需任务特定后训练即可迁移到反向扰动识别、双重扰动推理、表型筛选优先级排序和生物过程解释。PertMind进一步生成了支持竞争性基因、细胞和供体表征的生物图谱。

    arXiv:2608.16419v1 Announce Type: cross  Abstract: Large language models can describe mechanisms, yet scalable post-training still depends on costly, manually curated biological reasoning traces. Here we show that cellular perturbation atlases can instead become reinforcement-learning environments, where measured gene responses provide computable rewards for biological reasoning. We introduce PertMind, which combines trusted-trajectory supervised initialization with gene-, pathway-, and format-level reinforcement signals. Trained only on forward perturbation-response prediction, PertMind improved response inference in unseen cellular contexts while retaining general language capabilities. It also transferred without task-specific post-training to reverse perturbation identification, double-perturbation reasoning, phenotypic-screen prioritization, and biological-process interpretation. PertMind further generated biological profiles that supported competitive gene, cell, and donor repres
    
[^414]: SCALE：面向JEPA规划正确几何的状态校准潜在嵌入

    SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry

    [https://arxiv.org/abs/2608.16287](https://arxiv.org/abs/2608.16287)

    SCALE通过状态校准潜在嵌入，使端到端学习的LeWM表示获得类似DINO-WM的几何特性，从而改善基于JEPA的世界模型在规划中的状态信息利用。

    

    摘要：联合嵌入预测世界模型通过使用表示本身定义的代价函数，将预测的终端嵌入与目标嵌入进行评分来规划。获得非坍缩表示的两种主要策略是继承预训练特征空间（如DINO-WM），以及通过反坍缩正则化（如LeWorldModel（LeWM）与SIGReg）端到端学习嵌入。这些策略在不同任务中表现出互补的优势。尽管任务相关状态可从两种模型的完整嵌入中解码，但DINO-WM的前几个主成分通常保留比LeWM更多的状态信息。由于欧几里得规划代价受高方差方向主导，这种差异影响状态对候选选择的影响强度。我们提出SCALE（状态校准潜在嵌入），以使端到端LeWM表示具备DINO-WM中观察到的有利几何特性。

    arXiv:2608.16287v1 Announce Type: new  Abstract: Joint-embedding predictive world models plan by scoring predicted terminal embeddings against a goal embedding using a cost defined on the representation itself. Two prominent strategies for obtaining non-collapsed representations are to inherit a pretrained feature space, as in DINO-WM, and to learn an embedding end to end with anti-collapse regularization, as in LeWorldModel (LeWM) with SIGReg. These strategies show complementary strengths across tasks. Although task-relevant state is decodable from the full embeddings of both models, DINO-WM's leading principal components usually retain substantially more state information than LeWM's. Because Euclidean planning costs are dominated by high-variance directions, this difference affects how strongly state can influence candidate selection. We propose SCALE (State-CAlibrated Latent Embeddings) to give the end-to-end LeWM representation the favorable geometric property observed in DINO-WM.
    
[^415]: 条件验证：用于适应和监控安全分类器的正确性估计

    Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers

    [https://arxiv.org/abs/2608.14089](https://arxiv.org/abs/2608.14089)

    本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。

    

    摘要：arXiv:2608.14089v1 公告类型：新  摘要：部署在大语言模型上的安全分类器通常因两个原因而失败：它们的决策反映了训练期间学习的策略，而非部署者期望的策略，并且随着部署流量的演变，其性能会下降。我们提出了条件验证（RCV），一种轻量级包装器，无需重新训练即可适应现成的安全分类器。RCV从分类器的内部表示中估计每个预测与部署者策略不一致的概率，并选择性地纠正可能错误的预测。相同的正确性估计还提供了用于检测分布漂移的无标签信号，从而启用一个维护循环，该循环更新正确性估计层，仅在必要时进行分类器微调。在三个现成的安全分类器和两个基准数据集上，RCV在每个类别中都提高了对部署者策略的遵循度。

    arXiv:2608.14089v1 Announce Type: new  Abstract: Safety classifiers deployed with large language models often fail for two reasons: their decisions reflect the policy learned during training rather than the deployer's desired policy, and their performance degrades as deployment traffic evolves. We present Regime-Conditional Verification (RCV), a lightweight wrapper that adapts an off-the-shelf safety classifier without retraining it. RCV estimates, from the classifier's internal representations, the probability that each prediction disagrees with the deployer's policy, and selectively corrects predictions likely to be wrong. The same correctness estimates also provide a label-free signal for detecting distribution shift, enabling a maintenance loop that updates the correctness estimation layer and resorts to classifier fine-tuning only when necessary. Across three off-the-shelf safety classifiers and two benchmark datasets, RCV improves adherence to the deployer's policy in every class
    
[^416]: QUASAR：通过损失感知重构降低量化感知训练中的损失下限

    QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction

    [https://arxiv.org/abs/2608.13966](https://arxiv.org/abs/2608.13966)

    本文提出QUASAR，一种在量化感知训练过程中持续进行轻量级损失感知重构的方法，以降低损失下限并提升低比特模型质量。

    

    随着大型语言模型推理转向更低精度，训练后量化（PTQ）变得越来越脆弱，使得量化感知训练（QAT）对于保持模型质量至关重要。然而，QAT在计算损失和代理梯度时，使用的是潜在全精度权重的有损重构，而更新则应用于潜在权重本身。这种不匹配可能导致次优的训练轨迹和更高的损失下限。二阶PTQ方法通过最小化损失感知重构误差来缓解类似差距，但对冻结模型执行一次可能需要数小时；在整个QAT过程中，随着权重变化而重复此过程是不切实际的。我们引入了QUASAR，一种QAT方法，它在训练循环中持续执行轻量级的损失感知重构，以降低损失下限并改进最终的低比特模型。在每一步训练中，QUASAR使用平方的指数移动平均。

    arXiv:2608.13966v1 Announce Type: cross  Abstract: As large language model inference shifts toward lower precision, post-training quantization (PTQ) becomes increasingly brittle, making quantization-aware training (QAT) essential for preserving model quality. However, QAT computes the loss and surrogate gradients using a lossy reconstruction of latent full-precision weights, while applying updates to the latent weights themselves. This mismatch can lead to suboptimal training trajectories and a higher loss floor. Second-order PTQ methods mitigate a similar gap by minimizing loss-aware reconstruction error, but doing it once for a frozen model can take hours; repeating this process throughout QAT as the weights evolve is impractical. We introduce QUASAR, a QAT method that continuously performs lightweight, loss-aware reconstruction in the training loop to lower the loss floor and improve the resulting low-bit model. At each training step, QUASAR uses the exponential moving average of sq
    
[^417]: AI 气象模型会漏掉极端天气吗？

    Do AI weather models miss extremes?

    [https://arxiv.org/abs/2608.09972](https://arxiv.org/abs/2608.09972)

    本研究利用十个月欧洲站点观测对十二个 AI 与物理气象模型进行评估，发现 AI 模型在极端天气上并不存在统一的低估缺陷，尾部表现取决于具体模型、变量和评估环境，但所有模型都普遍存在对极端值向均值收缩的共同条件误差模式。

    

    AI 气象模型常被报道会低估极端天气，但以往的大多数证据都来自与再分析资料对比验证的确定性回归模型。本研究基于十个月的欧洲站点观测资料，将十二个物理预报模型与 AI 预报模型与 ECMWF IFS 进行对比评估。评估涵盖 10 米风速、2 米温度、太阳辐射和降水，并依据基于固定的 ERA5 1991–2020 气候态所定义的天气型进行分类。我们发现 AI 模型在极端尾部并不存在统一固有的缺陷。若干 AI 模型在极端条件下仍比 IFS 更准确，而另一些则明显退化；物理模型之间也存在类似的差异。然而，所有模型都表现出一种共同的条件误差模式：对低观测值预测偏高，对高观测值预测偏低。因此，极端值的衰减并不意味着相对预报技巧的统一损失：尾部表现取决于具体的模型、变量和评估环境。

    arXiv:2608.09972v2 Announce Type: replace-cross  Abstract: AI weather models are often reported to underestimate extremes, but most evidence concerns deterministic regression models verified against reanalysis. We evaluate twelve physical and AI forecast models against ECMWF IFS using ten months of European station observations. The evaluation covers 10 m wind, 2 m temperature, solar radiation, and precipitation within regimes defined from a fixed ERA5 1991-2020 climatology. We find no uniform AI-specific deficit in the tails. Several AI models remain more accurate than IFS under extreme conditions, while others deteriorate markedly; comparable variation occurs among physical models. Every model nevertheless exhibits a common conditional-error pattern, overpredicting low observations and underpredicting high observations. Attenuation of extreme values therefore does not imply a uniform loss of relative skill: tail performance depends on the model, variable, and evaluation setting rathe
    
[^418]: Muon训练的Transformer中表征-读出接口处的Grokking后崩溃

    Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers

    [https://arxiv.org/abs/2608.07436](https://arxiv.org/abs/2608.07436)

    论文发现Muon训练的Transformer在grokking后发生崩溃的根源在于AdamW读出层更新与较大的特征均值相互作用产生类别相关的logit偏移，并叠加交叉熵导数错误，而修正这些错误可以稳定训练并恢复近乎完全的准确率。

    

    经过Muon训练的模运算Transformer可能在保持线性可解码任务信息的同时失去准确率。通过相邻交换实验，将五个捕获到的未归一化失败定位到AdamW读出层的更新上。将实际的读出位移乘以较大的特征均值，会产生一个跨输入共享、依赖类别的logit偏移，几乎能够复现每一种失败。仅用于训练的解码器可以恢复98.20%-100%的保留集准确率。修正交叉熵导数错误能够稳定五个匹配分支直至第100,000步；而四个使用准确交叉熵的RMS前瞻运行则因嵌入更新而失败。

    arXiv:2608.07436v2 Announce Type: replace  Abstract: Muon-trained modular-arithmetic transformers can lose accuracy while retaining linearly decodable task information. Adjacent swaps localize five captured unnormalized failures to AdamW readout updates. Multiplying the actual readout displacement by the large feature mean produces a class-dependent logit offset shared across inputs that nearly reproduces each failure. Training-only decoders recover 98.20-100% held-out accuracy. Correcting cross-entropy derivative errors stabilizes five matched branches through step 100,000; four prospective accurate-CE RMS runs fail through embedding updates.
    
[^419]: EvoHarness-RL：为自进化智能体学习运行时支架协调

    EvoHarness-RL: Learning Runtime Harness Coordination for Self-Evolving Agents

    [https://arxiv.org/abs/2608.05446](https://arxiv.org/abs/2608.05446)

    提出EvoHarness-RL统一框架，通过将环境特定支架实现与共享策略接口分离、构建信念-进度-经验（BPE）工作空间，并采用监督初始化加代价感知GRPO使支架协调可学习，显著提升长程LLM智能体的运行时支撑能力。

    

    长程大语言模型智能体越来越依赖外部执行支持来维持状态、跟踪进度、从失败中恢复，并在长时间交互中复用经验。然而，现有的支架及其使用方式通常针对特定环境定制，并通过提示词、启发式规则或系统特定规则进行控制，使得智能体与支架之间的协调难以联合优化。我们提出了EvoHarness-RL，一个将环境特定的支架实现与共享的面向策略接口相分离的统一框架。EvoHarness-RL将外部支持组织为信念、进度与经验（BPE）工作空间，并提供四个紧凑的支架操作用于访问和更新该状态。我们首先将BPE实例化为推理时脚手架，随后通过监督初始化加代价感知的GRPO使支架协调变得可学习。在多种异构长程任务上，EvoHarness-Base改进（摘要在此处被截断）

    arXiv:2608.05446v2 Announce Type: replace-cross  Abstract: Long-horizon LLM agents increasingly rely on external execution support to maintain state, track progress, recover from failures, and reuse experience across extended interactions. Yet existing harnesses and their use are often tailored to environments and controlled through prompts, heuristics, or system-specific rules, making agent and harness coordination difficult to jointly optimize. We introduce EvoHarness-RL, a unified framework that separates environment-specific harness implementations from a shared policy-facing interface. EvoHarness-RL organizes external support into a Belief, Progress, and Experience (BPE) workspace and exposes four compact harness actions for accessing and updating this state. We first instantiate BPE as an inference-time scaffold and then make harness coordination learnable through supervised initialization followed by cost-aware GRPO. Across heterogeneous long-horizon tasks, EvoHarness-Base impro
    
[^420]: 在发展性学习框架中用于持续、可理解视觉识别的多尺度结构特征

    Multi-Scale Structural Features for Continual, Comprehensible Visual Recognition in a Developmental Learning Framework

    [https://arxiv.org/abs/2607.25531](https://arxiv.org/abs/2607.25531)

    该论文提出了一种在多个尺度上编码边缘与轮廓结构及其空间关系的新型视觉特征表示，并将其融入无梯度的发展式学习框架，从而在无需回放缓冲区的条件下实现更准确、可持续且可解释的视觉识别。

    

    当代机器学习难以实现持续学习、复用先验知识以及展现可理解的内部结构。最近提出的一种发展式、无梯度学习框架通过局部变异与选择来学习输入的离散拓扑模型，从而解决了这些局限，并带来了内在的持续学习保障：新的观察只会精化现有结构而不会覆盖过去的知识，且无需回放缓冲区或预定义的任务边界。该方法在视觉输入上的扩展已在形状识别任务中验证了这一原理，但其依赖的特征表示表达能力有限，限制了识别准确率的上限。我们提出了一种新的视觉特征表示，能够在多个尺度上编码形状结构，捕获边缘和轮廓特征及其空间关系，并将其与网络精化学习过程相整合；我们进一步……

    arXiv:2607.25531v2 Announce Type: replace-cross  Abstract: Contemporary machine learning struggles to learn continually, reuse prior knowledge, and expose a comprehensible internal structure. A recently proposed developmental, gradient-free learning framework addresses these limitations by learning a discrete, topological model of its inputs through local variation and selection, yielding an inherent continual-learning guarantee: new observations refine existing structure without overwriting past knowledge, and without replay buffers or predefined task boundaries. Its extension to visual inputs demonstrated this principle on shape recognition, but relied on a feature representation of limited expressivity that capped recognition accuracy. We introduce a new visual feature representation that encodes shape structure across multiple scales, capturing edge and contour features together with their spatial relations, and integrate it with the network-refinement learning process; we further 
    
[^421]: 变分伊辛注意力：量身定制的注意力机制对科学任务至关重要

    Variational-Ising-Attention:Tailored Attention Matters for Science

    [https://arxiv.org/abs/2607.23634](https://arxiv.org/abs/2607.23634)

    提出变分伊辛注意力（VIA），通过带相互作用的伊辛模型与变分平均场推断增强softmax注意力，将注意力从孤立条目的排序扩展为相互作用实体的集体状态建模，并在逆合成反应中心预测与蛋白质残基接触预测等科学结构化预测任务上验证了其有效性。

    

    注意力机制通过查询-键打分与softmax归一化实现上下文建模。在工业界长上下文需求的驱动下，主流研究已趋向于稀疏化与高效化，但softmax的独立性假设依然存在。然而，对于不受长序列约束的科学任务而言，更丰富的结构化耦合往往是必需的，这使得定制化的注意力机制既可行又更为合适。为此，我们提出了变分伊辛注意力，它在softmax归一化的基础上引入了一个具有相互作用的伊辛模型；注意力模式通过变分平均场推断从可学习的成对耦合中涌现，将注意力从对孤立条目的排序扩展为对相互作用实体的集体状态进行建模。我们在逆合成反应中心预测任务上实例化了VIA，并作为受控的内部消融实验，在蛋白质残基接触预测上进行了验证——这两个都是由复杂相互作用所主导的结构化预测任务。

    arXiv:2607.23634v2 Announce Type: replace-cross  Abstract: Attention enables context modeling via query-key scoring with softmax normalization. Driven by industrial long-context demands, mainstream research has converged toward sparsity and efficiency, yet softmax's independence assumption persists. For scientific tasks unburdened by long-token constraints, however, richer structured coupling may often be essential, making tailored attention both viable and more appropriate. To this end, we propose Variational-Ising-Attention (VIA), which augments softmax normalization with an interacting Ising model; attention patterns emerge from learnable pairwise couplings via variational mean-field inference, extending attention from a ranking over isolated items to a collective state over interacting entities. We instantiate VIA on retrosynthesis reaction center prediction and, as a controlled internal ablation, on protein residue contact prediction, two structured prediction tasks governed by co
    
[^422]: 分布偏移下忠实生成的令牌级离策略学习

    Token-Level Off-Policy Learning for Faithful Generation Under Distribution Shift

    [https://arxiv.org/abs/2607.17524](https://arxiv.org/abs/2607.17524)

    提出令牌级离策略标注（TOPL）训练范式，将后训练重构为令牌级正确性预测任务，使模型在摘要、翻译等忠实生成任务中实现强大的分布外泛化能力。

    

    我们提出了令牌级离策略标注（Token-Level Off-Policy Labeling, TOPL），这是一种将后训练重新构建为令牌级正确性预测任务的离策略训练范式。我们的核心直觉是：通过训练模型区分响应中的好坏令牌，可以自然地引导模型生成好的令牌，同时避免直接训练模型生成离策略令牌所带来的陷阱。在文档摘要任务上的实验表明，TOPL 在 11 个数据集上相比多种序列级和令牌级基线方法展现出强大的分布外泛化能力。我们进一步证明 TOPL 能够有效迁移到机器翻译任务，表明其优势可以推广到不同的忠实生成任务中。通过消融研究，我们确认令牌级学习信号对良好性能至关重要，而序列级的类似方法并不能带来相当的收益。

    arXiv:2607.17524v2 Announce Type: replace  Abstract: We propose Token-Level Off-Policy Labeling (TOPL), an off-policy training paradigm that reframes post-training as a token-level correctness prediction task. Our key intuition is that by training the model to distinguish good and bad tokens in a response, we naturally guide the model towards generating good tokens, while avoiding the pitfalls that come with directly training the model to generate off-policy tokens. Experiments on document summarization tasks show that TOPL achieves strong out-of-distribution generalization across 11 datasets against a diverse set of sequence-level and token-level baselines. We further demonstrate that TOPL transfers effectively to machine translation, suggesting that its benefits generalize across different faithful generation tasks. Through ablation studies, we confirm that our token-level learning signal is critical to good performance; sequence-level analogues do not confer similar benefits. Finall
    
[^423]: 检索增强的可解释学习：迈向医疗保健领域的任务特定零样本模型

    Retrieval-Augmented Interpretable Learning: Towards Task-Specific Zero-Shot Models in Healthcare

    [https://arxiv.org/abs/2607.17508](https://arxiv.org/abs/2607.17508)

    RAIL是一种概率元学习框架，能够从自然语言任务描述出发，通过检索相关源任务并进行系数空间结构迁移，零样本生成具有特征级解释和不确定性量化的可解释医疗临床预测模型。

    

    我们提出了检索增强可解释学习（RAIL），这是一个概率元学习框架，用于零样本生成任务特定的可解释模型。该框架从自然语言任务描述和先前学习过的任务特定预测器记忆中综合出系数空间结构。RAIL检索相关的源任务，通过系数空间迁移结构，并在原始诊断特征空间中生成新的预测器，从而实现具有特征级解释的零样本和少样本临床程序预测。其概率化建模为检索过程、模型系数和预测结果提供了不确定性量化，支持不确定性感知的部署方式：不确定的预测或不稳定的解释可以被标记出来以供额外的临床审查，而不是被当作自动决策处理。这使得RAIL特别适用于医疗保健场景，因为其中的预测任务高度（依赖专业性与可靠性要求）。

    arXiv:2607.17508v3 Announce Type: replace-cross  Abstract: We introduce Retrieval-Augmented Interpretable Learning (RAIL), a probabilistic meta-learning framework for zero-shot generation of task-specific interpretable models that synthesizes coefficient-space structure from natural-language task descriptions and a memory of previously learned task-specific predictors. RAIL retrieves related source tasks, transfers structure through coefficient space, and generates a new predictor in the original diagnostic-feature space, enabling zero-shot and few-shot clinical procedure prediction with feature-level explanations. Its probabilistic formulation provides uncertainty over retrieval, model coefficients, and predictions, supporting uncertainty-aware deployment: uncertain predictions or unstable explanations can be flagged for additional clinical review rather than treated as automatic decisions. This makes RAIL particularly suited for healthcare settings, where prediction tasks are highly 
    
[^424]: 面向动态新视角合成的在线神经时空记忆方法

    Online Neural Space Time Memory for Dynamic Novel View Synthesis

    [https://arxiv.org/abs/2607.15271](https://arxiv.org/abs/2607.15271)

    本文提出一种将记忆更新与记忆应用频率解耦的在线神经时空记忆方法，通过周期性进行梯度记忆更新、逐帧结合跨视角注意力应用记忆，从而在满足实时约束的同时实现动态场景的新视角合成，并有效重建暂时被遮挡的区域。

    

    arXiv:2607.15271v2 公告类型： replace-cross 摘要：基于多视角流式视频的在线新视角合成面临一个根本性的权衡：一方面需要维护持久的长时程记忆以重建暂时被遮挡的区域，另一方面又必须在严格的实时约束下运行。尽管测试时训练（Test-Time Training, TTT）提供了一种强大的记忆机制，但标准模型要求在每一帧都进行基于梯度的记忆更新，以适应动态场景中不断变化的运动。这种繁重记忆更新的计算开销使其无法实时应用，并可能在长上下文中导致不稳定性。鉴于记忆更新比记忆应用更加耗费计算资源，且视频内容在很大程度上是冗余的，我们提出将这两个过程的频率解耦。我们的方法执行周期性的记忆更新，同时在每一帧上应用记忆，并利用跨视角注意力机制来处理先验记忆状态与当前帧之间的形变。为了锁定历史……（原文摘要在此处截断）

    arXiv:2607.15271v2 Announce Type: replace-cross  Abstract: Online novel view synthesis from multi-view streaming videos faces a fundamental trade-off: maintaining a persistent, long-horizon memory to reconstruct temporarily occluded regions while operating under strict real-time constraints. While Test-Time Training (TTT) offers a powerful memory mechanism, standard models mandate gradient-based memory updates at every frame to adapt to the changing motion in dynamic scenes. The computational cost of heavy memory updates precludes real-time application and can lead to instability over long contexts. Given that memory updates are more demanding than memory application and video content is largely redundant, we propose to decouple the frequencies of these two processes. Our approach performs periodic memory updates while applying the memory on a per-frame basis, using cross-view attention to manage deformations between the prior memory state and the current frame. To lock in the historic
    
[^425]: 超越骨架划分：结构前沿评估揭示ADMET模型中的隐藏失效

    Beyond Scaffold Splits: Structural-Frontier Evaluation Reveals Hidden Failures in ADMET Models

    [https://arxiv.org/abs/2607.10729](https://arxiv.org/abs/2607.10729)

    该论文提出一种无标签的结构前沿划分评估方法，发现传统骨架划分会掩盖ADMET模型在化学结构最稀疏、物理化学性质最遥远的分子上的严重性能退化（误差膨胀中位数达87%），且这一问题在图神经网络中同样存在。

    

    分子性质模型通常通过保留Bemis-Murcko骨架进行评估，然而骨架标识符只是化学陌生性的一种衡量方式。我们引入了一种无标签的结构前沿划分方法，该方法保留最稀疏且物理化学性质最遥远的骨架组，并在六个公开的实验性或整理过的ADMET任务上进行评估。与具有相同无环分组的70/10/20骨架对照组相比，结构前沿划分使等权重主要误差膨胀，任务中位数达87.0%，偏度敏感均值达130.3%（描述性任务/种子bootstrap区间为52.1-246.0%）。移除BBB（血脑屏障）端点后，均值降至75.9%；该端点正是在结构前沿处评分排名发生逆转的端点。消息传递图网络对照组仍然显示出较大差距（四个任务均值82.8%）且未发生逆转，因此低容量预测头无法解释这一效应。我们还测试了多视角前沿风险外推（MV-FREX），

    arXiv:2607.10729v4 Announce Type: replace  Abstract: Molecular property models are commonly evaluated by holding out Bemis-Murcko scaffolds, yet a scaffold identifier is only one notion of chemical unfamiliarity. We introduce a label-free structural-frontier split that reserves the sparsest and most physicochemically remote scaffold groups, and evaluate it on six public experimental or curated ADMET tasks. Against a 70/10/20 scaffold control with identical acyclic grouping, the frontier inflates equally weighted primary error with a taskwise median of 87.0% and a skew-sensitive mean of 130.3% (descriptive task/seed bootstrap interval, 52.1-246.0%). The mean falls to 75.9% once BBB is removed; that endpoint is the one whose score ranking inverts at the frontier. A message-passing graph-network control still shows a large gap (mean 82.8% over four tasks) and does not invert, so a low-capacity head does not explain the effect. We also test Multi-View Frontier Risk Extrapolation (MV-FREX),
    
[^426]: 面向视觉-语言-动作模型强化学习的事后学习

    Learning from Hindsight for VLA Reinforcement Learning

    [https://arxiv.org/abs/2607.09042](https://arxiv.org/abs/2607.09042)

    LfH通过利用预训练视觉-语言模型对失败的机器人轨迹进行重新标记，将失败经验转化为有效的学习信号，从而显著提升VLA模型强化学习的样本效率。

    

    强化学习越来越多地被用于微调视觉-语言-动作模型，但机器人交互成本高昂，当成功的轨迹非常稀少时，学习变得极其样本低效。当奖励仅在完成指定任务时才被分配时，失败的轨迹会被视为毫无价值，即使它成功执行了与该任务相关的行为。一个未能将正确物体放入碗中的机器人，可能仍然将该物体移向了碗，或将另一个物体放入了碗中，这展示了可以被重复用于解决目标任务的对象和动作。这些行为定义了策略当前已经能够解决的辅助任务，在策略能够解决更困难的目标任务之前就提供了有用的学习信号。我们提出事后学习，它将这类失败转化为额外的学习信号。利用预训练的视觉-语言模型，LfH对失败的轨迹进行重新标记……

    arXiv:2607.09042v2 Announce Type: replace  Abstract: Reinforcement learning is increasingly used to fine-tune vision-language-action (VLA) models, but robot interaction is expensive and learning becomes highly sample inefficient when successful rollouts are rare. When reward is assigned only for completing the commanded task, a failed rollout is treated as having no value even if it successfully executes behaviors relevant to that task. A robot that fails to place the correct object in a bowl may still move that object toward the bowl or place a different object inside it, demonstrating objects and actions that can be reused to solve the target task. These behaviors define auxiliary tasks that the policy can already solve, providing useful learning signals even before it can solve the harder target task. We introduce $\textit{Learning from Hindsight (LfH)}$, which turns such failures into additional learning signals. Using a pretrained vision-language model, LfH relabels failed rollout
    
[^427]: 关于强化学习中奖励函数在大语言模型置信度校准中有效性的研究

    On the effectiveness of reward functions in reinforcement learning for confidence calibration of large language models

    [https://arxiv.org/abs/2607.04332](https://arxiv.org/abs/2607.04332)

    该论文揭示了设计不当的置信度奖励函数会诱导大语言模型为校准置信度而故意答错问题（即“置信度奖励黑客行为”），并提出并验证了可抵御此类攻击的“不可被黑客攻击的置信度奖励方案”的构建方法。

    

    在本文中，我们研究了使用强化学习（RL）训练大语言模型（LLM）以同时提高推理准确性并用语言表达其置信度的场景。我们的奖励方案使用两个函数来奖励大语言模型所表达的置信度：一个用于正确答案，另一个用于错误答案。如果设计不当，这种方案可能会激励大语言模型为了使自身置信度得到校准而故意给出错误答案，我们将这种现象称为“置信度奖励黑客行为”（confidence reward hacking）。我们提出了“不可被黑客攻击的置信度奖励方案”这一概念，并提供了构建此类方案的方法。我们证明了在实际数据集中，可被黑客攻击的奖励方案下会出现选择性的置信度奖励黑客行为，而不可被黑客攻击的奖励方案则能够抵御这种攻击。最后，我们将其中一些方案置于基于强化学习的置信度校准的“过度自信-不自信”光谱上，并通过实验加以验证（原文摘要此处不完整）。

    arXiv:2607.04332v2 Announce Type: replace  Abstract: In this paper, we consider the setting where large language models (LLMs) are trained using reinforcement learning (RL) to simultaneously improve reasoning accuracy and verbalize their confidence. Our reward scheme uses two functions for rewarding confidence verbalized by the LLM: one for correct answers and the other for incorrect answers. If poorly designed, such a scheme may incentivize an LLM to answer incorrectly in order for its confidence to be calibrated, a phenomenon we term confidence reward hacking. We introduce the notion of non-hackable confidence reward schemes and provide methods for constructing them. We show that selective confidence reward hacking can arise in practical datasets under hackable reward schemes while non-hackable reward schemes are resistant to hacking. Finally, we place some of these schemes along an overconfidence-underconfidence spectrum for RL-based confidence calibration and demonstrate experiment
    
[^428]: 平均场朗之万动力学中统计特征学习的几何学

    The Geometry of Statistical Feature Learning in Mean-Field Langevin Dynamics

    [https://arxiv.org/abs/2606.31429](https://arxiv.org/abs/2606.31429)

    该论文通过基-纤维分解为统计特征学习建立了几何框架，证明球面平均场朗之万动力学的低温平稳分布在多指标模型中会集中于隐藏指标并形成多尖峰结构、以高概率实现参数恢复，且这一集中现象在温度约等于1处存在锐利的相变。

    

    我们为监督回归引入了一种统计特征学习的几何表述。特征学习通过基-纤维分解来定义：基是训练过程产生的特征侧几何结构，纤维是执行估计所用的学习特征空间。我们针对球面平均场朗之万动力学证明了这一性质，该动力学被视为负熵正则化经验风险的Wasserstein梯度流。在高斯多指标模型中，低温平稳分布集中于隐藏指标附近，形成多尖峰结构，并以高概率实现参数恢复——尽管负熵正则化本身是惩罚集中现象的。这种集中现象在温度 λ ≍ 1 处存在一个急剧的相变。在高斯单指标模型中，平稳测度满足集中性质，其奇偶性决定该测度位于 S_2^{d-1} 上还是其他位置（摘要在此处截断）。

    arXiv:2606.31429v2 Announce Type: cross  Abstract: We introduce a geometric formulation of statistical feature learning for supervised regression. Feature learning is defined through a base--fiber decomposition: the base is the feature-side geometry produced by training, and the fiber is the learned feature space where estimation is performed. We prove this property for spherical mean-field Langevin dynamics, viewed as the Wasserstein gradient flow of a negative entropy-regularized empirical risk. In Gaussian multi-index models, the low-temperature stationary distribution concentrates near the hidden indices, forms a multi-spike structure, and yields parameter recovery with high probability, even though negative entropy regularization penalizes concentration. This concentration has a sharp transition at temperature $\lambda\asymp 1$. In Gaussian single-index models, the stationary measure satisfies a concentration property, with parity determining whether it lives on $S_2^{d-1}$ or $\m
    
[^429]: 基于严格恰当评分规则学习概率滤波器

    Learning Probabilistic Filters with Strictly Proper Scoring Rules

    [https://arxiv.org/abs/2606.26497](https://arxiv.org/abs/2606.26497)

    本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。

    

    针对部分观测且含噪声的动态系统的贝叶斯滤波，旨在在线推断系统状态随观测演变的条件分布。该贝叶斯滤波分布是不确定性量化的自然对象，但很少能作为监督学习目标直接获得。然而，我们通常可以利用预测模型生成合成系统轨迹及合成观测数据。本文提出了恰当评分集成滤波器（PSEF），这是一种基于训练分析映射的集成数据同化方法，仅通过合成状态-观测轨迹来逼近滤波分布。分析步骤被表示为一种基于置换不变性、Transformer架构的映射，它接收预测集成和观测作为输入，生成分析集成。训练基于严格恰当的评分规则——其中使用了能量评分。

    arXiv:2606.26497v1 Announce Type: new  Abstract: Bayesian filtering of partially and noisily observed dynamical systems seeks to infer the evolving conditional distribution of the state of a dynamical system, given observations, in an online fashion. This Bayesian filtering distribution is the natural object for uncertainty quantification, but it is rarely available as a supervised learning target. However, one can often use the forecast model to generate synthetic system trajectories, along with synthetic observations. We introduce the proper scoring ensemble filter (PSEF), an ensemble data assimilation method based on training an analysis map to approximate the filtering distribution using only synthetic state--observation trajectories. The analysis step is represented as a permutation-invariant, transformer-based map that takes as input a forecast ensemble and observations, producing an analysis ensemble. Training is based on strictly proper scoring rules -- with the energy score us
    
[^430]: 素数傅里叶嵌入：模运算的一种原则性基础

    Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic

    [https://arxiv.org/abs/2606.23044](https://arxiv.org/abs/2606.23044)

    本文提出素数傅里叶嵌入，将整数编码为按素数索引的 (cos, sin) 对，借助舒尔引理从理论上证明等变线性映射必为每个素数对应一个独立块的块对角结构，并结合中国剩余定理预测与超过 500 倍特化比的消融实验验证，表明模运算可归结为选择相关的素数通道。

    

    数字具有代数结构，而标准的神经嵌入往往无法揭示这种结构。我们提出了素数傅里叶嵌入，它将整数编码为源自 Q 的调和分析的、按素数索引的 (cos, sin) 对，从而提供了一种预结构化的表示，使得模运算可以简化为选择相关的素数通道，而无需从零开始发现代数结构。我们证明，任何对 PFE 上乘积群作用保持等变的线性映射必定是块对角的，且每个素数对应一个独立块——这是将舒尔引理应用于所得特征分解的结果。对于无平方因子的复合模数，中国剩余定理可以预测哪些素数通道与任务相关。这两项预测均得到实证验证：消融实验显示，任务相关通道与任务无关通道之间的特化比率超过 500 倍，并在分布内测试中取得了完美的表现。

    arXiv:2606.23044v3 Announce Type: replace-cross  Abstract: Numbers have algebraic structure that standard neural embeddings often fail to expose. We introduce Prime Fourier Embeddings (PFE), which encode integers as prime-indexed (cos, sin) pairs derived from the harmonic analysis of Q, providing a pre-structured representation in which modular arithmetic reduces to selecting the relevant prime channel rather than discovering algebraic structure from scratch. We prove that any linear map equivariant with respect to the product group action on PFE must be block-diagonal with one independent block per prime -- a consequence of Schur's lemma applied to the resulting character decomposition. For square-free composite moduli, the Chinese Remainder Theorem predicts which prime channels are task-relevant. Both predictions are confirmed empirically: ablation studies show specialization ratios exceeding 500x between task-relevant and task-irrelevant channels, with perfect in-distribution test a
    
[^431]: 稀疏自编码器能学到有意义的概念层次结构吗？

    Do Sparse Autoencoders Learn Meaningful Concept Hierarchies?

    [https://arxiv.org/abs/2606.22994](https://arxiv.org/abs/2606.22994)

    本文针对稀疏自编码器的概念层次结构首次提出了一套系统的关键要求与具体评估协议，用以严格检验现有方法是否真正学到了有意义的概念层次。

    

    稀疏自编码器（SAEs）已成为大型模型中无监督概念发现的重要工具。为了使所得的特征空间更具可解释性和更易于管理，近来的方法开始显式地施加层次结构，或通过训练约束隐式地产生层次结构，但严格的比较仍然困难。目前对于有意义特征层次结构应满足哪些要求尚无共识，评估工作主要依赖定性说明和零散的定量协议。为解决这一问题，我们借鉴语义网络与分类学研究以及近期的SAE相关工作，推导出无监督概念发现中泛化/特化层次结构应满足的一组关键要求，并据此制定了一个具体的评估协议。将该协议应用于在视觉数据上训练的当前SAE方法后，我们发现虽然特征空间通常能够提供一个基础……（摘要在此处截断）

    arXiv:2606.22994v2 Announce Type: replace  Abstract: Sparse autoencoders (SAEs) have become an important tool for unsupervised concept discovery in large models. To make the resulting feature spaces more interpretable and manageable, recent approaches have begun imposing hierarchical structure, either explicitly or as an implicit effect of training constraints, yet rigorous comparison remains difficult. There are no agreed-upon requirements for what a meaningful feature hierarchy should satisfy, and evaluation has largely relied on qualitative illustrations with fragmented quantitative protocols. To address this, we derive a set of key requirements for generalization/specialization hierarchies in unsupervised concept discovery, drawing on semantic net and taxonomy research alongside recent SAE work, and use them to derive a concrete evaluation protocol. Applying this protocol to current SAE approaches trained on visual data, we find that while feature spaces generally provide a basis f
    
[^432]: 用程序合成解释注意力机制

    Explaining Attention with Program Synthesis

    [https://arxiv.org/abs/2606.19317](https://arxiv.org/abs/2606.19317)

    该论文提出通过程序合成将Transformer语言模型的注意力头行为转化为可执行的Python程序，利用预训练语言模型生成并根据保留数据筛选程序，证明不到1000个程序即可复现GPT-2和TinyLlama-1.1B中注意力头的注意力模式。

    

    可解释深度学习研究的一个长期目标是用人类可理解的符号描述来替代不透明的神经计算。在本文中，我们提出了一种用可执行程序来近似深度网络组件行为的方法。我们重点关注Transformer语言模型中的注意力头。对于给定的注意力头，我们首先在一组随机选取的训练样本上计算其相关的注意力矩阵。接着，我们用这些矩阵的摘要来提示一个预训练语言模型，并指示它生成一组Python程序，这些程序能够仅凭输入句子的文本复现相应的注意力模式。最后，我们根据程序在保留输入上预测行为的好坏对生成的程序进行重新排序。我们证明了少于1,000个这样的生成程序即可复现GPT-2、TinyLlama-1.1B等模型中注意力头的注意力模式。

    arXiv:2606.19317v3 Announce Type: replace-cross  Abstract: A longstanding goal of research on interpretable deep learning is to replace opaque neural computations with human-meaningful symbolic descriptions. In this paper, we propose an approach for approximating the behavior of components of deep networks with executable programs. We focus on attention heads in transformer language models. For a given head, we first compute its associated attention matrices on a collection of randomly selected training examples. Next, we prompt a pre-trained language model with a summary of these matrices, and instruct it to generate a set of Python programs that can reproduce the associated attention patterns given only text from the input sentence. Finally, we re-rank programs according to how well our final set of programs predict behavior on held-out inputs. We demonstrate that a set of fewer than 1,000 such generated programs can reproduce the attention patterns of heads in GPT-2, TinyLlama-1.1B,
    
[^433]: 我们需要解释卡片来连接解释算法与现实世界

    We Need Explanation Cards to Connect Explanation Algorithms to the Real World

    [https://arxiv.org/abs/2606.16786](https://arxiv.org/abs/2606.16786)

    本文提出“解释卡片”框架，通过为标准算法解释补充稳健性、有效性信息及清晰的解读说明，弥合解释表面含义与实际价值之间的差距，使解释算法在现实世界中更可靠、更实用。

    

    算法解释旨在帮助利益相关者理解不透明的算法决策，但在实践中，它们往往达不到预期效果。首先，算法解释的含义常常与人们的直觉预期不符，因此需要专业知识才能正确解读。其次，最近的研究表明，流行的解释算法对于复杂决策函数的行为并不能提供有效信息。这些问题共同造成了解释表面所传达的内容与其实际提供的内容之间的差距。在这项工作中，我们提出了面向解释算法的“解释卡片”，它通过补充关于稳健性和有效性的信息以及清晰的解读说明来增强标准解释。这些补充信息可以使原本缺乏实际价值的解释变得实用，同时也有助于检测解释无效的情况。重要的是，解读……

    arXiv:2606.16786v2 Announce Type: replace  Abstract: Algorithmic explanations are intended to help stakeholders understand opaque algorithmic decisions, but in practice, they often fall short. First, the meaning of algorithmic explanations is often not what one might intuitively expect, so expert knowledge is required to interpret them correctly. Second, recent work has shown that popular explanation algorithms are uninformative about the behavior of complex decision functions. Together, these issues create a gap between what explanations appear to convey and what they actually provide. In this work, we propose Explanation Cards for Explanation Algorithms, which augment standard explanations with complementary information about robustness and validity, as well as clear instructions for interpretation. The complementary information can render otherwise uninformative explanations practically useful, while also helping to detect cases where they are not. Importantly, the interpretation in
    
[^434]: SAE++：级联稀疏自编码器在多模态大语言模型中学习多层次视觉概念

    SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs

    [https://arxiv.org/abs/2606.16193](https://arxiv.org/abs/2606.16193)

    SAE++提出级联稀疏自编码器架构，直接在第一级SAE的解码器权重上训练第二级SAE，从而在多模态大语言模型中学习层次化的“概念的概念”视觉表征。

    

    多模态大语言模型（MLLMs）在视觉-语言任务上展现出强大的性能，但其内部的视觉表征仍然难以解释。稀疏自编码器（SAEs）提供了一种可扩展的方法，可以将密集的模型激活分解为稀疏、可解释的特征。然而，现有的SAE架构主要恢复的是平坦的特征字典，不太适合显式的多层次概念组织。在本文中，我们提出了一种级联稀疏自编码器架构，称为SAE++，用于在MLLMs中学习层次化的视觉概念。SAE++不是嵌套或堆叠SAE的稀疏激活码，而是直接在第一级SAE的解码器权重上训练第二级SAE，将学习到的低级特征方向作为更高层次抽象的输入。这种设计使SAE++能够学习“概念的概念”，同时避免了嵌套式SAE的共享前缀耦合带来的缺陷……（摘要原文在此处截断）

    arXiv:2606.16193v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) have demonstrated strong performance on vision-language tasks, yet their internal visual representations remain difficult to interpret. Sparse Autoencoders (SAEs) provide a scalable way to decompose dense model activations into sparse, interpretable features. However, existing SAE architectures primarily recover flat feature dictionaries and are less suited for explicit multi-level concept organization. In this paper, we introduce a cascaded sparse autoencoder architecture, dubbed SAE++, for learning hierarchical visual concepts in MLLMs. Rather than nesting or stacking SAE sparse activation codes, SAE++ trains a second-level SAE directly on the decoder weights of the first-level SAE, treating learned low-level feature directions as inputs for higher-level abstraction. This design enables SAE++ to learn "concepts of concepts" while avoiding drawbacks from the shared-prefix coupling of ne
    
[^435]: 通过电子健康记录中稳健且灵活的知识迁移增强谱嵌入

    Enhancing Spectral Embedding through Robust and Flexible Knowledge Transfer in Electronic Health Records

    [https://arxiv.org/abs/2606.11570](https://arxiv.org/abs/2606.11570)

    该论文提出一种基于谱方法的无监督表示学习框架，通过放宽一对一信号对齐假设并采用两步嵌入流程，从更广泛人群中稳健且灵活地迁移知识，为样本量有限的罕见病电子健康记录数据生成高质量的低维嵌入表示。

    

    我们提出了一种基于谱方法的无监督表示学习框架，用于从电子健康记录中为罕见病队列的临床概念和患者推导低维嵌入表示，此类数据具有高维特征但样本量有限。为克服这一挑战，我们引入了一个从更广泛人群中提取的知识矩阵，该矩阵与罕见病队列共享部分重叠的子空间。我们的方法与现有方法的不同之处在于，放宽了潜在数据矩阵与知识矩阵之间严格的一对一信号对齐假设，从而允许更灵活、更符合实际的结构化共享形式。我们提出了一种新颖的两步谱嵌入流程：首先，识别并移除知识矩阵中的无关成分；然后，应用基于投影的方法分别恢复共享成分与异质成分。仿真实验和对真实数据的分析（摘要在此处截断）……

    arXiv:2606.11570v2 Announce Type: replace-cross  Abstract: We propose a spectral-based, unsupervised representation learning framework to derive low-dimensional embeddings for clinical concepts and patients in rare disease cohorts from electronic health records, where data are high-dimensional but sample sizes are limited. To overcome this challenge, we incorporate a knowledge matrix extracted from a broader population that shares a partially overlapping subspace with the rare-disease cohort. Our method departs from existing approaches by relaxing restrictive one-to-one signal-alignment assumptions between the latent data matrix and knowledge matrix, allowing more flexible and realistic forms of structured sharing. We introduce a novel two-step spectral embedding procedure: first, we identify and remove irrelevant components from the knowledge matrix; then, we apply a projection-based method to separately recover shared and heterogeneous components. Simulations and an analysis of a rea
    
[^436]: Express 语言建模

    Express Language Modeling

    [https://arxiv.org/abs/2606.10944](https://arxiv.org/abs/2606.10944)

    Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。

    

    我们介绍了一种新工具 Express，用于将非因果注意力近似转换为具有匹配近似保证的因果近似。当与最先进的 Thinformer 近似相结合时，Express 改进了已知最佳的因果注意力保证：对于长度为 n 的序列，在仅使用 O(s) 内存和 O(s² log²(n)) 压缩开销的情况下，实现了 log^{3/2}(n)/s 的近似误差。我们将这些进展与高效的 I/O 感知 Triton 实现相结合，展示了相对于 FlashAttention 2 的显著加速，并利用 Express 克服了语言建模流程中的四个资源瓶颈：长上下文预填充、KV 缓存压缩、长文本内存受限解码以及长文本计算受限解码。

    arXiv:2606.10944v2 Announce Type: replace  Abstract: We introduce a new tool, Express, for converting a non-causal attention approximation into a causal approximation with matching approximation guarantees. When combined with the state-of-the-art Thinformer approximation, Express improves upon the best known causal attention guarantees, delivering $\log^{3/2}(n)/s$ approximation error with only $O(s)$ memory and $O(s^2 \log^2(n))$ compression overhead for a sequence of length $n$. We pair these developments with an efficient I/O-aware Triton implementation, demonstrate substantial speedups over FlashAttention 2, and use Express to overcome four resource bottlenecks in the language modeling pipeline: long-context prefill, KV cache compression, long-form memory-constrained decoding, and long-form compute-constrained decoding.
    
[^437]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^438]: 基于均值的算法：下界与遗憾

    Mean-based algorithms: A lower bound and regret

    [https://arxiv.org/abs/2606.04931](https://arxiv.org/abs/2606.04931)

    本文首次为未知时间范围和赌博机反馈下基于均值算法的核心序列 $\gamma_t$ 建立了下界，揭示了此类算法学习速度的根本极限，并提出了两种新的基于均值的算法（其中一种推广了 $\epsilon$-贪婪算法）。

    

    基于均值的算法是一类在线学习算法，它们会为平均奖励较低的动作分配较低的概率。近期研究表明，这类算法会收敛到“序列非被支配动作”，这些动作可作为经济博弈中纳什均衡的近似。然而，实证研究表明，在赌博机反馈设置下，基于均值的算法的收敛速度比已有的无遗憾算法更慢。本工作研究了未知时间范围和赌博机反馈条件下的基于均值的算法。在此设置下，我们首次给出了定义此类算法的序列 $\gamma_t$ 的下界，确立了此类算法学习速度的基本极限。在多臂赌博机问题中，这一结果限制了任何算法在依据所学知识行动的同时可靠识别低奖励动作的速率。我们还提出了两种基于均值的算法：一种推广了 $\epsilon$-贪婪算法，另一种扩展了……（原文摘要在此处截断）

    arXiv:2606.04931v2 Announce Type: replace  Abstract: Mean-based algorithms are online learning algorithms that assign low probability to actions with low average rewards. Recent research shows that they converge to serially undominated actions, which serve as approximations to Nash equilibria in economic games. However, empirical studies indicate that mean-based algorithms converge more slowly in bandit-feedback settings than established no-regret alternatives.   This work investigates mean-based algorithms under unknown horizons and bandit feedback. In this setting, we provide the first lower bound on the algorithm-defining sequence $\gamma_t$, establishing a fundamental limit on the learning speed of such algorithms. In multi-armed bandit problems, this result constrains the rate at which any algorithm can reliably identify low-reward actions while acting according to this knowledge.   We also propose two mean-based algorithms: one generalizes $\epsilon$-greedy, and the other extends
    
[^439]: FFR：面向回归的前向-前向学习

    FFR: Forward-Forward Learning for Regression

    [https://arxiv.org/abs/2606.03927](https://arxiv.org/abs/2606.03927)

    提出FFR框架，首次将前向-前向学习算法从分类扩展到真实世界的回归任务，通过序数竞争好感度函数等三项关键创新解决了连续目标空间缺乏对比“对立面”的问题，并在多个真实数据集上取得了有竞争力的性能。

    

    前向-前向算法通过纯局部的逐层优化方式训练神经网络，为反向传播（BP）提供了一种计算高效且符合生物学合理性的替代方案。然而，FF 本质上是为分类任务设计的，其依赖于正负样本对的对比学习，将其扩展到回归任务面临根本性的挑战：连续的目标空间缺乏用于对比学习的天然“对立面”，且标准的好感度函数不包含关于目标数值大小或排序的信息。我们提出了 FFR（面向回归的前向-前向算法），据我们所知，这是首个将 FF 扩展到真实世界回归任务的框架，并在多样化的真实数据集上展示了有竞争力的性能。FFR 引入了三项关键创新：（1）一种序数竞争好感度函数，在距离感知的序数约束下，用划分神经元组之间的竞争学习取代对比样本对（摘要在此处截断）

    arXiv:2606.03927v2 Announce Type: replace-cross  Abstract: The Forward-Forward (FF) algorithm offers a computationally efficient and biologically plausible alternative to backpropagation (BP) by training neural networks through purely local, layer-wise optimization. However, FF is inherently designed for classification via contrastive positive-negative sample pairs, and extending it to regression poses fundamental challenges: continuous target space lacks natural "opposites" for contrastive learning, and the standard goodness function carries no information about target magnitude or ordering. We propose FFR (Forward-Forward for Regression), to our knowledge, the first framework to extend FF to real-world regression and demonstrate competitive performance across diverse realworld datasets. FFR introduces three key innovations: (1) an ordinal competitive goodness function that replaces contrastive pairs with competitive learning between partitioned neuron groups under distance-aware ordi
    
[^440]: 面向函数空间变分推断的流变换隐过程

    Flow-Transformed Implicit Processes for Function-Space Variational Inference

    [https://arxiv.org/abs/2606.01954](https://arxiv.org/abs/2606.01954)

    提出流变换隐过程（FTIP），通过超越高斯组合权重分布的限制，使有限维函数空间近似能够灵活表示非对称、重尾或多峰的后验不确定性。

    

    隐过程先验通过灵活的生成机制来定义函数上的分布，这使其在贝叶斯函数空间建模中颇具吸引力。然而，使用此类先验进行后验推断具有挑战性，因为其诱导的函数空间分布通常不具备闭式形式。一种实用的策略是使用有限个采样函数的集合来近似先验，然后将后验函数表示为这些样本的学习组合。现有方法通常在组合权重上放置高斯变分分布。尽管这种方法易于处理，但它限制了所能表示的后验不确定性的形状，尤其是当真实后验呈现非对称、重尾或多峰特性时。我们提出了流变换隐过程（FTIP），这是一种变分推断方法，使这种有限维函数空间近似更加灵活。

    arXiv:2606.01954v2 Announce Type: replace-cross  Abstract: Implicit-process priors define distributions over functions through flexible generative mechanisms, making them attractive for Bayesian function-space modelling. However, performing posterior inference with such priors is challenging because their induced function-space distributions are typically not available in closed form. One practical strategy is to approximate the prior using a finite collection of sampled functions, and then represent posterior functions as learned combinations of these samples. Existing approaches commonly place a Gaussian variational distribution over the combination weights. While tractable, this choice limits the shapes of posterior uncertainty that can be represented, especially when the true posterior is asymmetric, heavy-tailed, or multimodal. We propose Flow-Transformed Implicit Processes (FTIP), a variational inference method that makes this finite-dimensional function-space approximation more 
    
[^441]: 任务多样性产生系统性迁移但抑制持续强化学习

    Task diversity produces systematic transfer but inhibits continual reinforcement learning

    [https://arxiv.org/abs/2606.00880](https://arxiv.org/abs/2606.00880)

    该论文提出了GPU加速的持续强化学习环境Banyan，可通过参数化控制任务多样性，并发现任务多样性能带来系统性迁移能力，但会抑制智能体在连续分布偏移中的持续学习能力。

    

    持续强化学习（RL）旨在产生能够永不停止地适应新任务的智能体。一个关键问题是这种能力与智能体所经历的任务多样性之间存在怎样的相互作用。先前的研究表明，在许多多样化任务上进行训练可以产生具有较强零样本和上下文内适应能力的智能体。然而，这些工作是在智能体停止学习之后（即权重被冻结的状态下）进行评估的。任务多样性如何影响智能体在一系列分布偏移中持续学习的能力仍不清楚。我们提出了Banyan，一个GPU加速的持续强化学习环境，其中可以通过参数化方式控制定义任务的三个独立维度：智能体必须导航的地图布局、它必须与之交互的对象，以及子目标依赖关系的层次结构。我们发现，沿每个维度增加多样性都会引发系统性迁移——也就是说，智能体在新任务分布上开始训练时接近（摘要在此处被截断）

    arXiv:2606.00880v2 Announce Type: replace-cross  Abstract: Continual reinforcement learning (RL) aims to produce agents that never stop adapting to new tasks. A key question is how this interacts with the diversity of tasks an agent experiences. Prior work has shown that training on many diverse tasks leads to agents with strong zero-shot and in-context adaptation. However, this work evaluated agents after they'd stopped learning, i.e. with frozen weights. How task diversity affects an agent's ability to continue learning over a sequence of distribution shifts remains unclear. We introduce Banyan, a GPU-accelerated continual RL domain where one can parametrically control three independent axes that define a task: the map layouts an agent must navigate, the objects it must interact with, and the hierarchical structures of sub-goal dependencies. We find that increasing diversity along each axis induces systematic transfer -- that is, agents begin training on a new task distribution near 
    
[^442]: 强化学习中的终止表示

    The Terminal Representation in Reinforcement Learning

    [https://arxiv.org/abs/2605.31289](https://arxiv.org/abs/2605.31289)

    本文提出了强化学习中一种结构上全新的终止表示（TR），它类似于默认表示（DR）编码奖励加权轨迹，但能以更低维度学习，且无需特征向量计算即可直接支持选项发现、奖励塑形、迁移学习和探索等下游任务。

    

    表示学习是强化学习（RL）中进行时空抽象的强大工具。两种成熟的方法是后继表示（SR）和默认表示（DR）。SR通过状态所引发的未来轨迹对状态进行编码，捕捉与奖励解耦的信息流。DR在此基础上用奖励对轨迹进行加权，将信用分配结构整合到表示之中。这两种表示的特征向量已被用于支持一系列下游任务——包括选项发现、奖励塑形、迁移学习和探索。我们提出了一种结构上不同的表述形式：终止表示（TR）。TR与DR类似地编码奖励加权的轨迹，但可以作为更低维度的对象来学习，并且无需进行特征向量计算即可直接用于上述应用。特征分解……

    arXiv:2605.31289v3 Announce Type: replace-cross  Abstract: Representation learning is a powerful tool for spatio-temporal abstraction within reinforcement learning (RL). Two well established approaches are through the successor representation (SR) and the default representation (DR). The SR encodes states by the future trajectories they induce, capturing information flow decoupled from reward. The DR builds on this by weighting trajectories with reward, integrating credit-assignment structure into the representation. Eigenvectors of both representations have been used to support a range of downstream tasks -- including option discovery, reward shaping, transfer learning, and exploration. We introduce a structurally distinct formulation: the terminal representation (TR). The TR encodes reward-weighted trajectories similarly to the DR, but can be learned as a lower-dimensionality object, and can be used directly for the mentioned applications without eigenvector computations. Eigendecomp
    
[^443]: FHRFormer：一种用于胎心率时间序列修复与预测的自监督掩码Transformer框架

    FHRFormer: A Self-Supervised Masked Transformer Framework for Fetal Heart Rate Time-Series Inpainting and Forecasting

    [https://arxiv.org/abs/2605.29695](https://arxiv.org/abs/2605.29695)

    该论文提出了FHRFormer，一种自监督掩码Transformer框架，能够修复和预测可穿戴胎心率监测中因信号丢失而产生的数据缺口，从而为基于AI分析连续胎心率数据以预测新生儿呼吸辅助风险奠定基础。

    

    大约10%的新生儿在出生时需要辅助才能开始呼吸，约5%需要通气支持。胎心率（FHR）监测在产前护理中评估胎儿健康状况方面发挥着至关重要的作用，能够检测异常模式，并支持及时的产科干预，以降低分娩过程中胎儿的风险。将人工智能（AI）方法应用于分析具有多样化结局的大规模连续FHR监测数据集，可能为预测需要呼吸辅助或干预的风险提供新的见解。可穿戴FHR监测仪的最新进展使连续胎儿监测成为可能，且不影响孕妇的活动能力。然而，孕妇活动期间的传感器移位，以及胎儿或母体体位的变化，常常导致信号丢失，造成记录的FHR数据出现缺口。这种数据缺失限制了有意义见解的提取。

    arXiv:2605.29695v2 Announce Type: replace  Abstract: Approximately 10% of newborns require assistance to initiate breathing at birth, and around 5% need ventilation support. Fetal heart rate (FHR) monitoring plays a crucial role in assessing fetal well-being during prenatal care, enabling the detection of abnormal patterns and supporting timely obstetric interventions to mitigate fetal risks during labor. Applying artificial intelligence (AI) methods to analyze large datasets of continuous FHR monitoring episodes with diverse outcomes may offer novel insights into predicting the risk of needing breathing assistance or interventions. Recent advances in wearable FHR monitors have enabled continuous fetal monitoring without compromising maternal mobility. However, sensor displacement during maternal movement, as well as changes in fetal or maternal position, often lead to signal dropout, resulting in gaps in recorded FHR data. Such missing data limits the extraction of meaningful insights
    
[^444]: 喷注生成的神经缩放定律

    Neural Scaling Laws for Jet Generation

    [https://arxiv.org/abs/2605.28940](https://arxiv.org/abs/2605.28940)

    这项工作首次证明神经缩放定律同样适用于粒子喷注生成任务，在模型规模缩放中复现了对数缩放行为，并发现物理量的切片Wasserstein距离与验证损失单调相关。

    

    最近观察到的经验缩放定律描述了基础类型模型在三个独立关键量——数据集规模、计算量和模型参数——发生变化时的性能表现。提取这些缩放定律可以为大型复杂模型的训练提供指导，因为对于这类模型，传统的超参数调优方式并不可行。这项工作首次探讨了缩放定律是否也可以在粒子喷注生成任务中被观察到——该任务既可作为基础模型的预训练目标，其本身也可作为原位模拟手段。我们确实复现了模型规模缩放中关键的对数缩放定律行为。除了研究生成模型的下一个词元预测验证损失之外，我们还研究了五个物理量的切片Wasserstein距离，这些物理量在训练期间无法被模型直接获取。我们的研究表明，该量与下一个（摘要在此处截断）

    arXiv:2605.28940v2 Announce Type: replace-cross  Abstract: Recently observed empirical scaling laws describe the performance of foundation-type models as three independent key quantities -- dataset size, compute, and model parameters -- are modified. Extracting these scaling laws informs the training of large complex models for which the tuning of hyperparameters in traditional ways is not feasible. This work for the first time explores if scaling laws can also be observed for the task of particle jet generation -- both relevant as a pre-training objective for foundation models and as in-situ simulation by itself. We indeed replicate the key logarithmic scaling law behavior for model-size scaling. Beyond studying the next token prediction validation loss of the generative model, we also study the sliced Wasserstein distance of five physical quantities that are not immediately available to the model during training. Our study shows that this quantity is monotonically related to the next
    
[^445]: SeedER：基于“种子-扩展-检索”的高效知识图谱检索方法

    SeedER: Seed-Expand-Retrieve for Efficient Knowledge Graph Retrieval

    [https://arxiv.org/abs/2605.23753](https://arxiv.org/abs/2605.23753)

    提出SeedER方法，采用“种子-扩展-检索”策略实现高效的知识图谱检索，克服了稠密嵌入在多跳查询上的理论局限以及LLM智能体和GNN全图处理的高昂计算成本。

    

    知识图谱（KG）为关系知识提供了丰富的表示形式，但其不规则的结构使检索变得具有挑战性：自我图（ego-graph）的扩展增长迅速，且稠密嵌入方法难以处理多跳组合查询。一些方法使用大语言模型（LLM）智能体来探索知识图谱、分析候选节点并决定下一步的探索方向。这些方法虽然表达能力强，但会带来巨大的计算和内存开销。另一方面，我们从理论上证明，为图节点预计算的稠密嵌入——即使加入了增强的结构和邻域感知特征——在回答某些知识图谱查询族时，可能需要与图规模相当的嵌入维度。这一限制在某些条件下可以通过查询自适应嵌入来克服，并且存在能够实现这一点的图神经网络（GNN）变体。然而，使用GNN处理整个图同样会带来巨大的……

    arXiv:2605.23753v2 Announce Type: replace  Abstract: Knowledge graphs (KGs) offer a rich representation for relational knowledge, but their irregular structure makes retrieval challenging: ego-graph expansion grows rapidly, and dense embedding methods struggle with multi-hop compositional queries. Several approaches use LLM agents to explore the KG, analyze candidate nodes, and decide where to explore next. While expressive, these approaches can incur substantial computational and memory costs. On the other hand, we show theoretically that dense embeddings precomputed for graph nodes, even with augmented structure and neighborhood-aware features, can require embedding dimensions comparable to the size of the graph to answer families of knowledge graph queries. This limitation can be overcome with query-adaptive embeddings under certain conditions, and there are graph neural network (GNN) variants that can do so. However, processing the whole graph with a GNN can also incur substantial 
    
[^446]: 基于预测分布的强化学习用于大语言模型回归

    Reinforcement Learning over Predictive Distributions for LLM Regression

    [https://arxiv.org/abs/2605.20740](https://arxiv.org/abs/2605.20740)

    提出了分布感知奖励（DAR），一种同策略强化学习目标，通过留一法贡献对同一输入的多个预测所形成的预测分布进行联合评估，从而提升大语言模型回归的校准质量。

    

    大语言模型（LLMs）已成为灵活的回归器，能够从异构输入中预测实值数量。然而，大多数LLM回归目标独立地优化各个预测，往往导致校准效果不佳。我们提出了分布感知奖励，这是一种同策略强化学习目标，转而对同一输入的多个预测所形成的经验预测分布进行联合评估。为了将这种分布级别的目标转化为逐个rollout级别的奖励，我们根据每个预测对整体预测分布质量的留一法贡献来分配其信用。这种方法鼓励预测在目标值周围良好居中且具有适当的离散程度。我们在三种回归设置上进行了评估：一个用于探测插值和外推能力的合成任务，以及两个涉及代码和分子数据的真实世界科学任务。在所有任务中，DA

    arXiv:2605.20740v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have emerged as flexible regressors capable of predicting real-valued quantities from heterogeneous inputs. Yet most LLM regression objectives optimize predictions independently, often yielding poor calibration. We introduce Distribution-Aware Reward (DAR), an on-policy reinforcement learning objective that instead jointly evaluates the empirical predictive distribution formed by multiple predictions for the same input. To translate this distribution-level objective into rollout-level rewards, we assign each prediction credit based on its leave-one-out contribution to the quality of the overall predictive distribution. This encourages predictions that are well-centered and appropriately dispersed around the target. We evaluate on three regression settings: a synthetic task probing interpolation and extrapolation, and two real-world scientific tasks involving code and molecular data. Across tasks, DA
    
[^447]: 调用还是不调用：诊断大语言模型智能体中的内在过度调用偏差

    To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents

    [https://arxiv.org/abs/2605.18882](https://arxiv.org/abs/2605.18882)

    该研究提出并验证了“内在偏差假说”（IBH），揭示大语言模型智能体在调用/不调用决策中存在与激活无关的过度调用偏差，利用稀疏自编码器定位并量化该偏差，并通过自适应边际校准转向（AMCS）方法实现了因果层面的纠正。

    

    大语言模型（LLM）智能体表现出一种持续存在的“过度调用”倾向，即使在没有调用需求的情况下也会调用工具。在When2Call基准测试中，来自三个模型系列的六个模型均显示出较高的“调用”准确率，但“不调用”的准确率要低得多，导致整体准确率处于55%-70%的区间。我们将这一现象归因于“内在偏差假说”（Intrinsic Bias Hypothesis, IBH）：调用/不调用的决策映射携带一个与激活无关的调用偏移量，因此即使在激活水平相等的情况下，模型也倾向于选择调用。利用稀疏自编码器（SAE），我们恢复了与行为对齐的调用/不调用决策特征基，将其简化为一个带符号的激活边际，并直接估计出该偏移量。在全部六个模型中，只有当“不调用”激活超过“调用”激活时，模型才能实现决策中立，这与IBH一致。随后，我们通过自适应边际校准转向（AMCS）方法对IBH进行因果验证，AMCS是一种沿SAE解码器方向施加闭式反偏差偏移的技术。通过抵消所诊断出的偏移量，该方法能够有效纠正模型的过度调用行为（摘要在此处截断）。

    arXiv:2605.18882v2 Announce Type: replace-cross  Abstract: LLM agents exhibit a consistent tendency to over-call, invoking tools even in situations where none is needed. On the When2Call benchmark, six models from three families show high call accuracy but much lower no-call accuracy, leaving overall accuracy in the 55%-70% range. We trace this to an Intrinsic Bias Hypothesis (IBH): the call/no-call decision mapping carries an activation-independent call offset, so the model favors call even at activation parity. Using Sparse Autoencoders (SAEs), we recover behavior-aligned feature bases for the call/no_call decision, reduce them to a signed activation margin, and estimate the offset directly. Across all six models, the model is decision-neutral only when no_call activation outweighs call activation, consistent with IBH. We then causally test IBH with Adaptive Margin-Calibrated Steering (AMCS), a closed-form counter-bias shift along SAE decoder directions. Cancelling the diagnosed offs
    
[^448]: 面向约束机器学习的随机惩罚-障碍方法

    Stochastic Penalty-Barrier Method for Constrained Machine Learning

    [https://arxiv.org/abs/2605.18618](https://arxiv.org/abs/2605.18618)

    本文提出随机惩罚-障碍方法（SPBM），通过对偶变量指数平均、稳定惩罚调度和Moreau包络扩展经典惩罚-障碍方法以求解约束机器学习问题，证明了小批量采样下变换问题可行集包含于原问题可行集，实验表明其性能与最先进方法相当。

    

    约束机器学习（CML）能够实现公平性感知的训练、物理信息神经网络，以及将符号化领域知识整合到统计模型中。在这项工作中，我们针对CML问题提出了随机惩罚-障碍方法（SPBM）。SPBM通过引入对偶变量的指数平均、稳定的惩罚调度以及Moreau包络来处理非光滑性，从而扩展了经典的惩罚方法与障碍方法。我们分析了小批量采样在障碍函数中引入的偏差，并证明所得变换问题的可行集包含于原始问题的可行集之内。我们在多个公平性和物理信息神经网络实验中将SPBM与CML基线方法进行了比较，发现SPBM与最先进的方法相比具有竞争力。我们还观察到，在基于公平性的计算基准测试中，CML方法的每轮（epoch）运行时间在很大程度上是独立的。

    arXiv:2605.18618v3 Announce Type: replace-cross  Abstract: Constrained Machine Learning (CML) enables fairness-aware training, physics-informed neural networks, and integration of symbolic domain knowledge into statistical models. In this work, we introduce the Stochastic Penalty-Barrier Method (SPBM) for CML problems. SPBM extends classical penalty and barrier methods by incorporating an exponential averaging of the dual variables, a stabilized penalty schedule, and the Moreau envelope to handle non-smoothness. We analyze the bias that mini-batching introduces in the barrier function and show that the feasible set of the resulting transformed problem is contained within the original one. We compare SPBM with CML baselines across multiple fairness and physics informed neural networks experiments. We find that SPBM is competitive with state-of-the-art methods. We also observe, on our fairness-based computational benchmark, that the per-epoch runtime of CML methods is largely independent
    
[^449]: 面向非可交换面板数据的在线保形预测

    Online Conformal Prediction for Non-Exchangeable Panel Data

    [https://arxiv.org/abs/2605.17705](https://arxiv.org/abs/2605.17705)

    提出了W-TQA方法，通过结合从单元历史学习的相似性权重与自适应误覆盖水平，解决了非可交换、部分观测面板数据中的在线保形预测问题，并证明了即使在反馈缺失情况下也能实现长期平均覆盖率保证。

    

    我们研究了部分观测面板数据中的在线保形预测问题：在观测每个目标结果之前，会观察到新的横截面同行结果；目标反馈可能是间歇性出现或完全缺失的，且单元和轮次均无需满足可交换性。我们提出了加权时间分位数调整方法，该方法将基于单元历史学习到的相似性权重与自适应的目标特定误覆盖水平相结合。我们证明了目标与同行之间的失配以及未揭示轮次上的覆盖率都是不可识别的，因此关于跨单元相似性和反馈机制的假设是不可避免的。我们以这种失配为条件对过去条件误覆盖率进行了界定，在画像相似性假设下量化了学习权重的代价，并证明了所实现的算法——在必要时回退到最大同行分数——在完全随机缺失的反馈条件和可行性条件下能够实现长期平均覆盖率。

    arXiv:2605.17705v2 Announce Type: replace-cross  Abstract: We study online conformal prediction in a partially observed panel: a new cross-section of peer outcomes is observed before each target outcome, target feedback may be intermittent or absent, and neither units nor rounds need be exchangeable. We propose Weighted Temporal Quantile Adjustment (W-TQA), which combines similarity weights learned from unit histories with an adaptive target-specific miscoverage level. We prove that neither target-peer mismatch nor coverage on unrevealed rounds is identifiable, so assumptions on cross-unit similarity and on the feedback mechanism cannot be avoided. We bound the past-conditional miscoverage in terms of this mismatch, quantify the cost of learning the weights under a profile-similarity assumption, and show that the implemented procedure, which falls back on the largest peer score, attains long-run average coverage under missing-completely-at-random feedback and a feasibility condition on
    
[^450]: 语音“克隆”即风格迁移

    Voice "Cloning" is Style Transfer

    [https://arxiv.org/abs/2605.16578](https://arxiv.org/abs/2605.16578)

    这篇论文揭示语音克隆并非真正“克隆”个人声音，而是系统性地进行风格迁移，使克隆语音比源语音显得更权威、温暖且更像人类，并导致说话人特征的同质化。

    

    人工生成的语音正日益融入日常生活。语音克隆技术尤其能够支持身份保持至关重要的应用场景，例如补全录音、用新语言配音，或为失语症患者保留声音。然而，在我们的研究中发现，尽管名为“克隆”，语音克隆并不能忠实地“克隆”个人的声音。相反，我们发现广泛使用的语音克隆模型系统性地对源语音施加了风格迁移。根据人工标注者的评价，与源语音相比，克隆语音被认为更具权威性、更温暖、更像客服、更像人类。人工标注者还报告称，与源语音相比，他们对克隆语音的信任度更高，并更愿意向克隆语音透露敏感的个人信息。我们的研究进一步表明，语音克隆会导致说话人特征的同质化，这通过降低的……（原文此处截断）来衡量。

    arXiv:2605.16578v4 Announce Type: replace-cross  Abstract: Artificially generated speech is increasingly embedded in everyday life. Voice cloning in particular enables applications where identity preservation is important, such as completing a recording, dubbing in a new language, or preserving the voices of individuals with speech loss. However, in our work, we find that despite the term, voice cloning does not faithfully ''clone'' an individual's voice. Instead, we find that widely-used voice cloning models systematically apply style transfer to source voices. As rated by human annotators, cloned voices are perceived as more authoritative, warm, customer-service-like, and human-like compared to their sources. Human annotators also report greater trust in cloned voices than source voices, and a greater willingness to disclose sensitive personal information to them. Our work furthermore shows that voice cloning leads to homogenization of speaker characteristics, as measured by reduced 
    
[^451]: ForcingDAS：基于扩散强制的统一且鲁棒的数据同化

    ForcingDAS: Unified and Robust Data Assimilation via Diffusion Forcing

    [https://arxiv.org/abs/2605.14285](https://arxiv.org/abs/2605.14285)

    提出ForcingDAS框架，通过为每帧分配独立噪声水平的扩散强制方法，统一了滤波与平滑两种数据同化模式，并解决了传统逐帧转移模型在非马尔可夫观测下长期误差累积的问题。

    

    数据同化（DA）是从含噪且不完整的观测中估计演化动力系统的状态，广泛应用于科学模拟以及天气与气候科学。在实践中，滤波方法依赖于逐帧转移模型。然而，当观测是非马尔可夫的（即观测仅构成更高维潜在状态的部分切片，正如真实世界天气数据那样）时，这些模型十分脆弱：它们往往会在长时间范围内累积误差。与此同时，学习型DA方法通常只专注于单一模式，要么是滤波（临近预报、实时预测），要么是平滑（回顾性再分析），这使得本应是共享先验的内容被割裂到面向特定应用的流水线之中。为了解决这两个问题，我们提出了ForcingDAS，一个统一且鲁棒的DA框架。ForcingDAS建立在扩散强制基础之上，为每一帧分配独立的噪声水平，从而学习联合轨迹……

    arXiv:2605.14285v3 Announce Type: replace-cross  Abstract: Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In practice, filtering methods rely on frame-to-frame transition models. However, these models are fragile when observations are non-Markovian (when they form only a partial slice of a higher-dimensional latent state as in real-world weather data): they tend to accumulate errors over long horizons. At the same time, learned DA methods typically commit to a single regime, either filtering (nowcasting, real-time forecasting) or smoothing (retrospective reanalysis), which splits what should be a shared prior across application-specific pipelines. To address both issues, we introduce ForcingDAS, a unified and robust DA framework. Built on Diffusion Forcing with an independent noise level assigned to each frame, ForcingDAS learns a joint-traje
    
[^452]: 基于代理的机器遗忘方法的行为保证

    Behavioral Guarantees for Proxy-Based Unlearning

    [https://arxiv.org/abs/2605.10680](https://arxiv.org/abs/2605.10680)

    本文提出了一个统一并泛化基于代理的机器遗忘方法的框架，首次从理论上证明了遗忘模型与保留数据理想后验分布之间KL散度的上界，从而为遗忘后模型的行为提供了可保证的近似程度，使其最接近从头重训的模型。

    

    本文提出了一个泛化近期基于代理的机器遗忘方法的框架，并证明了由此产生的遗忘模型在行为上的理论保证：即其与保留数据理想后验分布之间的KL散度上界。我们将近似遗忘建模为一个约束优化问题，并将一类解解释为在输出空间中引入一个经过缩放的遗忘信号。该遗忘信号来源于后验数据分布的代理表示，其缩放尺度根据代理进行调整，以确保上述行为上界成立。该框架依赖于数据分布的结构来构建代理；如有需要，原目标模型可作为教师模型，通过蒸馏将更新传递到权重中。我们在两个遗忘场景中对该方法进行了实验验证，结果表明该方法能够得到与从头重新训练的模型最接近的分类器。

    arXiv:2605.10680v2 Announce Type: replace  Abstract: This paper proposes a framework generalizing recent proxy-based unlearning methods and proves theoretical guarantees about the behavior of the resulting unlearned model: upper bounds on its Kullback-Leibler divergence to the ideal posterior distribution of the retain data. We model approximate unlearning as a constrained optimization problem and interpret a family of solutions as introducing a scaled unlearning signal in the output space. The unlearning signal arises from proxies of the posterior data distributions. Its scale is adapted to the proxies to ensure the behavioral upper bounds. This framework relies on the structure of the data distributions in order to create proxies. If need be, the target serves as a teacher to distill the update in the weights. Our approach is experimentally validated over two forgetting scenarios as reaching the closest classifier to the model retrained from scratch.
    
[^453]: MSPR：面向目标条件强化学习的多尺度预测表征

    MSPR: Multi-scale Predictive Representations for Goal-conditioned Reinforcement Learning

    [https://arxiv.org/abs/2605.09364](https://arxiv.org/abs/2605.09364)

    本文提出MSPR框架，利用多尺度预测监督在离线目标条件强化学习中实现状态与目标的潜在空间对齐，在视觉和状态任务上均达到最先进性能，并对现实且具有挑战性的数据条件保持鲁棒性。

    

    本文研究了离线目标条件强化学习（GCRL）中的鲁棒表征学习。特别是在稀疏奖励场景中，学习能够对齐状态与目标潜在表示的表征是一项挑战，因为编码器可能学习到与目标无关的特征，从而破坏策略学习的稳定性。为解决这一问题，我们通过对齐目标来学习编码器的表征，这些对齐目标能够跨多个尺度捕捉环境信息，从局部物理动力学到长时程的目标导向结构。具体而言，我们提出了MSPR，这是一个利用多尺度预测监督在潜在空间内强制实现目标导向对齐的框架。我们证明了MSPR在视觉任务和基于状态的任务上都取得了强劲的性能。此外，我们表明我们的方法在现实且具有挑战性的数据条件下依然稳健，在各种任务中保持了最先进的性能。

    arXiv:2605.09364v2 Announce Type: replace  Abstract: This paper investigates robust representation learning in offline goal-conditioned reinforcement learning (GCRL). Particularly in sparse reward scenarios, learning representations that align state and goal latents is a challenge, as the encoder can learn goal-agnostic features that destabilize policy learning. We address this issue by learning the encoder's representation with alignment objectives that capture the environment across multiple scales, from local physical dynamics to long-horizon goal-directed structure. Concretely, we propose MSPR, a framework that leverages multi-scale predictive supervision to enforce goal-directed alignment within the latent space. We demonstrate that MSPR leads to strong performance on both vision and state-based tasks. Furthermore, we show that our approach is resilient under realistic, challenging data regimes, maintaining state-of-the-art performance across a wide variety of tasks.
    
[^454]: PMCTS：基于粒子蒙特卡洛树搜索的原则性并行推理时间扩展

    PMCTS: Principled Parallelized Inference Time Scaling with Particle Monte Carlo Tree Search

    [https://arxiv.org/abs/2605.08982](https://arxiv.org/abs/2605.08982)

    本文提出了PMCTS，一种专为GPU批处理并行化设计的原则性并行蒙特卡洛树搜索算法，它在保持策略改进保证的同时随并行计算规模良好扩展，并在国际象棋、围棋等MCTS和强化学习评估任务中始终优于或媲美流行的启发式基线方法。

    

    蒙特卡洛树搜索（MCTS）是强化学习中广泛用于策略改进和动作选择的方法。由于其顺序性和确定性的本质，如何利用并行计算对MCTS进行原则性的运行时扩展仍然是一个重大挑战。我们提出了粒子MCTS（PMCTS），这是一种原则性的并行MCTS算法，适用于神经网络评估，并专为GPU加速的批处理并行化而设计。我们为现代MCTS算法建立了策略改进保证，并证明PMCTS能够保持这些保证。实验表明，PMCTS在并行计算下具有良好的扩展性，并在一系列MCTS和强化学习评估领域中始终优于或可与流行的基于启发式的方法相媲美，这些领域包括棋盘游戏国际象棋和围棋，以及流行的离散动作和连续控制基准测试。

    arXiv:2605.08982v3 Announce Type: replace  Abstract: Monte Carlo Tree Search (MCTS) is a widely used approach for policy improvement and action selection in Reinforcement Learning. Due to its sequential and deterministic nature, principled runtime-scaling of MCTS with parallel compute remains a major challenge. We introduce Particle MCTS (PMCTS), a principled parallel MCTS algorithm suited for neural network evaluations and designed for GPU-acceleration with batch-parallelization. We establish policy improvement guarentees for modern MCTS algorithms and show that PMCTS maintains them. Empirically, PMCTS scales well with parallel compute and consistently outperforms or compares well to the popular heuristic-based baselines across a range of MCTS and RL evaluation domains, including the board games chess and Go and popular discrete action and continuous control benchmarks.
    
[^455]: 基于残差潜在动作学习视觉特征世界模型

    Learning Visual Feature-Based World Models via Residual Latent Action

    [https://arxiv.org/abs/2605.07079](https://arxiv.org/abs/2605.07079)

    该论文提出了一种可从DINO残差中轻松学习的新型潜在动作表示“残差潜在动作”（RLA），并基于流匹配构建RLA世界模型（RLA-WM），实现了更高效、更少幻觉且预测质量更优的视觉特征世界模型。

    

    世界模型从观测和动作中预测未来的状态转移。现有工作主要专注于图像生成。而基于视觉特征的世界模型预测的是未来的视觉特征而非原始视频像素，这提供了一种更高效且更不容易产生幻觉的有前景的替代方案。然而，当前基于特征的方法依赖于直接回归，这在复杂交互中会导致模糊或坍缩的预测，而在高维特征空间中进行生成式建模仍然具有挑战性。在这项工作中，我们发现一种新型的潜在动作表示——我们称之为残差潜在动作，可以轻松地从DINO残差中学习得到。我们还证明了RLA具有预测性、可泛化性，并能编码时间进程。基于RLA，我们提出了RLA世界模型（RLA-WM），它通过流匹配来预测RLA值。RLA-WM在性能上优于两者。

    arXiv:2605.07079v2 Announce Type: replace-cross  Abstract: World models predict future transitions from observations and actions. Existing works predominantly focus on image generation only. Visual feature-based world models, on the other hand, predict future visual features instead of raw video pixels, offering a promising alternative that is more efficient and less prone to hallucination. However, current feature-based approaches rely on direct regression, which leads to blurry or collapsed predictions in complex interactions, while generative modeling in high-dimensional feature spaces still remains challenging. In this work, we discover that a new type of latent action representation, which we refer to as Residual Latent Action (RLA), can be easily learned from DINO residuals. We also show that RLA is predictive, generalizable, and encodes temporal progression. Building on RLA, we propose RLA World Model (RLA-WM), which predicts RLA values via flow matching. RLA-WM outperforms both
    
[^456]: 异步分类分布时间差分学习的有限迭代理论

    A Finite-Iteration Theory for Asynchronous Categorical Distributional Temporal-Difference Learning

    [https://arxiv.org/abs/2605.06866](https://arxiv.org/abs/2605.06866)

    本文为异步分类分布时间差分方法建立了有限迭代收敛理论，通过受限域分析和泊松方程分解，在折扣和固定视界场景下提供了无需混合时间窗口的收敛保证。

    

    我们研究了分类分布时间差分方法所使用的精确异步递归的有限迭代行为。该分析涵盖了Cramér几何中的标量分类TD和最大均值差异几何中的多变量符号分类TD。现有的逐状态等距嵌入将这两种方法转化为在块上确界范数下收缩的单状态随机逼近递归，但分类算子仅在不变表示域上具有收缩性。我们建立了所需的受限域理论，并在独立同分布采样和马尔可夫轨迹下获得了折扣界。泊松方程分解处理了轨迹依赖性，无需显式的混合时间窗口。对于无折扣固定视界策略评估，我们在情节采样下为视界堆叠分类方法建立了类似的有限迭代保证。这些结果共同重新...

    arXiv:2605.06866v2 Announce Type: replace  Abstract: We study finite-iteration behavior of the exact asynchronous recursions used by categorical distributional temporal-difference methods. The analysis covers scalar categorical TD in the Cram\'er geometry and multivariate signed-categorical TD in the maximum mean discrepancy geometry. Existing statewise isometric embeddings turn both methods into single-state stochastic-approximation recursions that contract in a block-supremum norm, but the categorical operators are contractive only on invariant representation domains. We establish the required restricted-domain theory and obtain discounted bounds under i.i.d. sampling and under a Markovian trajectory. A Poisson-equation decomposition handles trajectory dependence without an explicit mixing-time window. For undiscounted fixed-horizon policy evaluation, we establish analogous finite-iteration guarantees for horizon-stacked categorical methods under episodic sampling. Together, these re
    
[^457]: 重新思考适配器放置：主导适配模块的视角

    Rethinking Adapter Placement: A Dominant Adaptation Module Perspective

    [https://arxiv.org/abs/2605.06183](https://arxiv.org/abs/2605.06183)

    该论文提出PAGE探测方法，发现LoRA可训练梯度能量高度集中于单一浅层FFN下投影的“主导适配模块”，其位置由模型架构决定且跨任务稳定，为有限数量适配器的最优放置提供了明确指导。

    

    低秩适配是一种广泛使用的参数高效微调方法，它将可训练的低秩适配器插入到冻结的预训练模型中。近期研究表明，使用更少的LoRA适配器仍可能保持甚至提升性能，但现有方法仍然广泛地分布适配器，因此“将有限数量的适配器放置在何处以最大化性能”这一问题在很大程度上仍未解决。为了研究这一问题，我们提出了PAGE（投影适配器梯度能量），这是一种基于梯度的敏感性探测方法，用于估计每个候选LoRA适配器可获得的初始可训练梯度能量。令人惊讶的是，我们发现PAGE在两个模型家族和四个下游任务上高度集中于同一个浅层FFN下投影模块。我们将该模块称为主导适配模块，并证明其所在层索引依赖于架构但跨任务保持稳定。

    arXiv:2605.06183v2 Announce Type: replace  Abstract: Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning method that places trainable low-rank adapters into frozen pre-trained models. Recent studies show that using fewer LoRA adapters may still maintain or even improve performance, but existing methods still distribute adapters broadly, leaving \emph{where to place a limited number of adapters to maximize performance} largely open. To investigate this, we introduce \textbf{PAGE} (\textbf{P}rojected \textbf{A}dapter \textbf{G}radient \textbf{E}nergy), a gradient-based sensitivity probe that estimates the initial trainable gradient energy available to each candidate LoRA adapter. Surprisingly, we find that PAGE is highly concentrated on a single shallow FFN down-projection across two model families and four downstream tasks. We term this module the \textbf{dominant adaptation module} and show that its layer index is architecture-dependent but task-stable. Motivate
    
[^458]: XDecomposer：面向多相X射线衍射的无先验集合分解学习方法

    XDecomposer: Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction

    [https://arxiv.org/abs/2605.05866](https://arxiv.org/abs/2605.05866)

    提出XDecomposer，将多相XRD分析形式化为集合预测问题，无需候选相列表、结构模板或相数量先验即可实现多相XRD图谱的联合分解与结构识别。

    

    多相粉末X射线衍射（PXRD）分析仍然是结构鉴定中的一个基础性瓶颈，因为现实中的合成往往会生成复杂的混合物，其组成相（组分）难以被可靠地分离。尽管近年来基于表示的晶体检索与生成方面的进展表明，直接从PXRD推断结构已成为可能，但现有方法大多假设输入为单相，在多相场景下会失效。本文提出了XDecomposer，这是一个无需先验知识的框架，能够在不需要候选相列表、结构模板或相数量先验知识的情况下，对多相XRD图谱进行联合分解与识别。我们将多相衍射分析形式化为一个集合预测问题，模型在统一架构内推断出一个无序的相分辨组分集合、各组分的混合比例以及相应的结构表示。

    arXiv:2605.05866v2 Announce Type: replace  Abstract: Multiphase powder X-ray diffraction (PXRD) analysis remains a fundamental bottleneck in structure identification, as real-world synthesis often produces complex mixtures whose constituent phases (components) cannot be reliably disentangled. While recent advances in representation-based crystal retrieval and generation suggest the possibility of inferring structures directly from PXRD, existing approaches largely assume single-phase inputs and break down in multiphase settings. Here, we present XDecomposer, a prior-free framework for joint decomposition and identification of multiphase XRD patterns without requiring candidate phase lists, structural templates, or prior knowledge of phase number. We formulate multiphase diffraction analysis as a set prediction problem, where the model infers an unordered set of phase-resolved components, their mixture proportions, and corresponding structural representations within a unified architectu
    
[^459]: 冯·诺依曼网络

    Von Neumann Networks

    [https://arxiv.org/abs/2605.05780](https://arxiv.org/abs/2605.05780)

    该论文将冯·诺依曼上世纪的细胞计算模型与现代深度学习相结合，提出了冯·诺依曼神经元及其网络（VNNs），其架构可自组织生成、仅依赖于输入输出在细胞阵列上的位置，并在数学上基于神经算子扩展与格林函数学习。

    

    二十世纪中叶，数学家兼博学家约翰·冯·诺依曼创建了一个建立在细胞阵列上的计算系统，作为人脑的简单模型，其中每个细胞具有有限集合中的一种角色或状态，他预测这些状态将通过扩散过程来建模。在这项工作中，我们展示了这种系统在现代深度学习环境下的发展，使得构建一种具有可学习的专门化角色的人工神经元成为可能。我们将这种神经元称为冯·诺依曼神经元，由这类神经元构成的神经网络形成了一种自工程化设计，其架构仅取决于其输入和输出在该细胞阵列上的结构与位置。我们还构建了冯·诺依曼网络（VNNs）的数学框架，并证明它们是基于神经算子的扩展以及在细胞阵列上通过卷积学习格林函数的方法（摘要在此处截断）。

    arXiv:2605.05780v2 Announce Type: replace  Abstract: In the mid-twentieth century, mathematician and polymath John von Neumann created a computational system on an array of cells as a simple model of the human brain, where each cell had one of a finite set of roles or states that he predicted would be modelled by a diffusion process. In this work, we show that such a system, when developed in a modern deep learning setting, enables the construction of an artificial neuron having specialized roles that can be learnt. We refer to this neuron as the Von Neumann neuron, and the resulting neural network from such neurons result in a self-engineered design whose architecture is only dependent on the structure and locations of its inputs and outputs on this cellular array. The mathematical framework for these Von Neumann Networks (VNNs) is also constructed and shows that they are based on the extension of neural operators and the learning of Green's functions with convolutions on a cellular t
    
[^460]: 从对偶追踪到梯度裁剪：可证明更快的分布鲁棒多目标优化

    From Dual Tracking to Clipping: Provably Faster Distributionally Robust Multi-Objective Optimization

    [https://arxiv.org/abs/2605.05660](https://arxiv.org/abs/2605.05660)

    本文提出了分布鲁棒多目标优化（DR-MOO）框架并设计具有可证明保证的MGDA算法，先通过对偶估计的双循环算法达到O(ε^{-8})样本复杂度，再结合大批量采样与梯度裁剪在广义平滑条件下实现更快的收敛。

    

    多目标优化（MOO）在需要依据多个准则进行学习的应用中受到越来越多的关注。然而，大多数现有的MOO建模方式并未显式地考虑数据中的分布偏移问题。我们提出了分布鲁棒多目标优化（DR-MOO），即在各自的最坏情况分布下最小化多个目标。我们为DR-MOO提出了Pareto类型的解概念，并开发了具有可证明保证的多梯度下降算法（MGDA）。利用拉格朗日对偶重构，我们首先设计了一种双循环MGDA算法，该算法利用内循环来估计对偶变量，并在达到ε-Pareto平稳点时实现了总共O(ε^{-8})的样本复杂度。为了进一步提升收敛速度，我们将大批量采样与梯度裁剪相结合，以适应广义平滑性条件并控制随机偏好更新中的偏差……

    arXiv:2605.05660v3 Announce Type: replace  Abstract: Multi-objective optimization (MOO) has received growing attention in applications that require learning under multiple criteria. However, most existing MOO formulations do not explicitly account for distributional shifts in the data. We introduce distributionally robust multi-objective optimization (DR-MOO), which minimizes multiple objectives under their respective worst-case distributions. We propose Pareto-type solution concepts for DR-MOO and develop multi-gradient descent algorithms (MGDA) with provable guarantees. Leveraging a Lagrangian dual reformulation, we first design a double-loop MGDA that uses an inner loop to estimate dual variables and achieves a total sample complexity $\mathcal{O}(\epsilon^{-8})$ for reaching an $\epsilon$-Pareto-stationary point. To further improve convergence, we combine large-batch sampling with gradient clipping to accommodate generalized smoothness and control bias in stochastic preference upda
    
[^461]: 可观测神经ODE用于连续时间中可识别的因果预测

    Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time

    [https://arxiv.org/abs/2604.26070](https://arxiv.org/abs/2604.26070)

    本文提出可观测神经ODE，证明了在显式结构假设下，潜在状态的可观测性可通过连续时间条件前门调整识别动态处理效应，从而即使在存在隐藏混杂因素时也能实现可识别的连续时间因果预测。

    

    连续时间序贯决策问题中的因果推断受到隐藏混杂因素和部分观测状态的挑战。我们证明，在显式的结构假设下，即使存在隐藏混杂因素，潜在状态的可观测性也能通过连续时间条件前门调整实现对动态处理效应的识别。我们推导了一个一般性的调整公式，并证明当未观测的同期扰动在时间上不相关时，该公式可简化为易于处理的状态空间公式。该公式通过测量模型、潜在动力学以及潜在状态上的滤波分布，来表达不同处理轨迹下的潜在结果分布。我们提出了可观测神经ODE，这是一类处于可观测规范形式的神经ODE模型，它为因果预测实现了这种易于处理的调整。ObsNODEs学习连续时间动力学……

    arXiv:2604.26070v3 Announce Type: replace  Abstract: Causal inference in continuous-time sequential decision problems is challenged by hidden confounding and partially observed states. We show that, under explicit structural assumptions, observability of the latent state enables identification of dynamic treatment effects through a continuous-time conditional front-door adjustment, even in the presence of hidden confounding.   We derive a general adjustment formula and show that it reduces to a tractable state-space formula when unobserved contemporaneous disturbances are temporally uncorrelated. This formula expresses potential-outcome distributions under alternative treatment trajectories through the measurement model, latent dynamics, and the filtering distribution over latent states.   We propose Observable Neural ODEs (ObsNODEs), Neural ODE models in observable normal form that implement this tractable adjustment for causal forecasting. ObsNODEs learn continuous-time dynamics with
    
[^462]: MLP能否精确吸收自身的跳跃连接？

    Can an MLP Absorb Its Own Skip Connection Exactly?

    [https://arxiv.org/abs/2604.23705](https://arxiv.org/abs/2604.23705)

    该论文证明，对于当前前沿语言模型所使用的激活函数（如ReLU^2、ReGLU、SwiGLU、GeGLU），相同宽度的无残差MLP在任何深度下都无法精确计算残差块 x + MLP(x) 所表示的函数，因此移除跳跃连接后重新训练无法恢复完全相同的函数。

    

    通常归因于跳跃连接的好处是优化理论层面的：更平滑的损失景观和更好的梯度传播。我们转而提出一个表示层面的问题：给定一个残差块 x -> x + MLP(x)，一个相同宽度的无残差MLP能否计算出相同的函数？答案是否定的，而且是无需条件的、在任何深度下都成立，对于当前前沿语言模型中使用的所有激活函数（ReLU^2、ReGLU、SwiGLU、GeGLU）均是如此。对于不带门控的ReLU和GELU，吸收是可能的，但仅在一个测度为零的权重集合上成立。因此，这两类函数族在一般意义上是不相交的：移除跳跃连接并重新训练，无法在相同宽度下恢复出完全相同的函数。

    arXiv:2604.23705v2 Announce Type: replace  Abstract: The benefits usually attributed to skip connections are optimization-theoretic: a smoother loss landscape and better gradient propagation. We ask a representational question instead: given a residual block x -> x + MLP(x), does a residual-free MLP of the same width compute the same function? The answer is no, unconditionally and at every depth, for every activation used in current frontier language models (ReLU^2, ReGLU, SwiGLU, GeGLU). For ungated ReLU and GELU absorption is possible, but only on a set of weights of measure zero. The two families are therefore generically disjoint: removing a skip connection and retraining cannot recover exactly the same function at equal width.
    
[^463]: 基于文本嵌入的零领域知识算法选择

    Algorithm Selection with Zero Domain Knowledge via Text Embeddings

    [https://arxiv.org/abs/2604.19753](https://arxiv.org/abs/2604.19753)

    ZeroFolio利用预训练文本嵌入替代手工设计的实例特征，实现了零领域知识的算法选择，在涵盖7个领域的11个ASlib场景中的绝大多数上超越了传统方法。

    

    我们提出了ZeroFolio，一种无特征的算法选择方法，它使用预训练的文本嵌入来替代手工设计的实例特征。该方法将原始实例文件作为纯文本读取，使用预训练的嵌入模型对其进行嵌入，并通过加权k近邻算法选择合适的算法。我们的方法基于这样一个观察：预训练嵌入无需任何领域知识或任务特定的训练即可区分问题实例。ZeroFolio适用于任何实例格式为文本的问题领域。我们在涵盖7个领域（SAT、MaxSAT、QBF、ASP、CSP、MIP和图问题）的11个ASlib场景上对该方法进行了评估。ZeroFolio在11个场景中的9个上优于基于手工特征训练的随机森林，且优势通常十分显著，其中8个场景在所有序列化种子下均保持优势。与针对每个场景单独调优的随机森林相比，它在11个场景中的8个上获胜。在有公开AutoFolio结果的三个场景上……（原文摘要在此处截断）

    arXiv:2604.19753v3 Announce Type: replace  Abstract: We propose ZeroFolio, a feature-free approach to algorithm selection that uses pretrained text embeddings instead of hand-crafted instance features. It reads the raw instance file as plain text, embeds it with a pretrained embedding model, and selects an algorithm via weighted k-nearest neighbors. Our approach is based on the observation that pretrained embeddings can distinguish problem instances without any domain knowledge or task-specific training. ZeroFolio applies to any problem domain with text-based instance formats. We evaluate our approach on 11 ASlib scenarios spanning 7 domains (SAT, MaxSAT, QBF, ASP, CSP, MIP, and graph problems). ZeroFolio outperforms a random forest trained on hand-crafted features in 9 of 11 scenarios, often substantially, and in 8 of them with every serialization seed. It wins 8 of 11 scenarios against a per-scenario-tuned random forest. On the three scenarios with published AutoFolio results from th
    
[^464]: 基于Sinkhorn双随机注意力的秩衰减分析

    Sinkhorn doubly stochastic attention rank decay analysis

    [https://arxiv.org/abs/2604.07925](https://arxiv.org/abs/2604.07925)

    本文证明了使用Sinkhorn算法归一化的双随机注意力矩阵比标准softmax行随机注意力更能有效保持网络深度中的秩，从而缓解秩崩溃问题。

    

    自注意力机制是Transformer架构取得成功的关键。然而，标准的行随机注意力已被证明在各层之间存在严重的信号退化问题。特别地，它可能引发秩崩溃，导致token表示变得越来越均匀，同时还会引发熵崩溃，其特征是注意力分布高度集中。近期的研究强调了双随机注意力作为一种熵正则化形式的优势，它能够促进更平衡的注意力分布，从而带来更好的实证性能。在本文中，我们研究了跨网络深度的秩崩溃问题，并证明了使用Sinkhorn算法归一化的双随机注意力矩阵比标准的softmax行随机注意力矩阵能更有效地保持秩。正如之前针对softmax所证明的那样，跳跃连接对于缓解秩崩溃至关重要。我们通过实验验证了这一现象。

    arXiv:2604.07925v2 Announce Type: replace-cross  Abstract: The self-attention mechanism is central to the success of Transformer architectures. However, standard row-stochastic attention has been shown to suffer from significant signal degradation across layers. In particular, it can induce rank collapse, resulting in increasingly uniform token representations, as well as entropy collapse, characterized by highly concentrated attention distributions. Recent work has highlighted the benefits of doubly stochastic attention as a form of entropy regularization, promoting a more balanced attention distribution and leading to improved empirical performance. In this paper, we study rank collapse across network depth and show that doubly stochastic attention matrices normalized with Sinkhorn algorithm preserve rank more effectively than standard softmax row-stochastic ones. As previously shown for softmax, skip connections are crucial to mitigate rank collapse. We empirically validate this phe
    
[^465]: 基于噪声样本迭代精炼的神经全局优化方法

    Neural Global Optimization via Iterative Refinement from Noisy Samples

    [https://arxiv.org/abs/2604.03614](https://arxiv.org/abs/2604.03614)

    本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。

    

    从噪声样本中对黑盒函数进行全局优化是机器学习和科学计算中的一项根本性挑战。传统方法如贝叶斯优化在多模态函数上往往收敛到局部极小值，而无梯度方法则需要大量的函数评估。我们提出了一种新颖的神经方法，通过迭代精炼来学习寻找全局极小值。我们的模型以噪声函数样本及其拟合的样条表示作为输入，然后迭代地将初始猜测精炼至真实的全局极小值。该方法在随机生成的函数上进行训练，其全局极小值真值通过穷举搜索获得，在具有挑战性的多模态测试函数上实现了8.05%的平均误差，而样条初始化的误差为36.24%，实现了28.18%的提升。该模型在72%的测试用例中成功找到全局极小值，误差低于10%。

    arXiv:2604.03614v3 Announce Type: replace-cross  Abstract: Global optimization of black-box functions from noisy samples is a fundamental challenge in machine learning and scientific computing. Traditional methods such as Bayesian Optimization often converge to local minima on multi-modal functions, while gradient-free methods require many function evaluations. We present a novel neural approach that learns to find global minima through iterative refinement. Our model takes noisy function samples and their fitted spline representation as input, then iteratively refines an initial guess toward the true global minimum. Trained on randomly generated functions with ground truth global minima obtained via exhaustive search, our method achieves a mean error of 8.05 percent on challenging multi-modal test functions, compared to 36.24 percent for the spline initialization, a 28.18 percent improvement. The model successfully finds global minima in 72 percent of test cases with error below 10 pe
    
[^466]: 基于混合专家机制的自适应语义通信无线图像传输

    Adaptive Semantic Communication for Wireless Image Transmission Leveraging Mixture-of-Experts Mechanism

    [https://arxiv.org/abs/2604.02691](https://arxiv.org/abs/2604.02691)

    本文提出了一种基于自适应混合专家Swin Transformer模块的多阶段端到端MIMO图像语义通信系统，其核心创新在于设计了同时联合评估实时信道状态信息与语义内容的动态专家门控机制，从而突破传统单一驱动路由的局限，实现对多样化图像内容和动态信道条件的自适应无线图像传输。

    

    基于深度学习的语义通信在无线图像传输领域已取得显著进展，但现有大多数方案依赖固定模型，因而对多样化的图像内容和动态变化的信道条件缺乏鲁棒性。为提升适应性，近期研究提出了根据信源内容或信道状态来调整传输或模型行为的自适应语义通信策略。最近，基于混合专家（MoE）的语义通信作为一种稀疏且高效的自适应架构开始兴起，但现有设计仍主要依赖单一驱动的路由机制。为解决这一局限，我们提出了一种新颖的多阶段端到端图像语义通信系统，面向多输入多输出（MIMO）信道，并基于自适应MoE Swin Transformer模块构建。具体而言，我们引入了一种动态专家门控机制，可联合评估实时信道状态信息（CSI）与图像的语义内容。

    arXiv:2604.02691v2 Announce Type: replace  Abstract: Deep learning based semantic communication has achieved significant progress in wireless image transmission, but most existing schemes rely on fixed models and thus lack robustness to diverse image contents and dynamic channel conditions. To improve adaptability, recent studies have developed adaptive semantic communication strategies that adjust transmission or model behavior according to either source content or channel state. More recently, MoE-based semantic communication has emerged as a sparse and efficient adaptive architecture, although existing designs still mainly rely on single-driven routing. To address this limitation, we propose a novel multi-stage end-to-end image semantic communication system for multi-input multi-output (MIMO) channels, built upon an adaptive MoE Swin Transformer block. Specifically, we introduce a dynamic expert gating mechanism that jointly evaluates both real-time CSI and the semantic content of i
    
[^467]: 面向长时程机器人任务的可泛化密集奖励

    Generalizable Dense Reward for Long-Horizon Robotic Tasks

    [https://arxiv.org/abs/2604.00055](https://arxiv.org/abs/2604.00055)

    提出VLLR密集奖励框架，将LLM/VLM生成的外在任务进展奖励与策略自确信度内在奖励相结合，无需人工奖励工程即可通过强化学习微调机器人基础策略，从而解决长时程任务中的误差累积问题。

    

    现有的机器人基础策略主要通过大规模模仿学习进行训练。尽管这类模型展现出强大的能力，但由于分布偏移和误差累积，它们在长时程任务中往往表现不佳。虽然强化学习（RL）可以对这些模型进行微调，但若缺乏人工奖励工程，它难以在多样化任务上取得良好效果。我们提出了VLLR，一种密集奖励框架，它结合了：（1）由大语言模型（LLM）和视觉-语言模型（VLM）产生的外在奖励，用于识别任务进展；（2）基于策略自确信度（self-certainty）的内在奖励。VLLR利用LLM将任务分解为可验证的子任务，再借助VLM估计任务进度来初始化价值函数，进行短暂的预热阶段，从而避免在整个训练过程中产生过高的推理成本；同时，自确信度在整个PPO微调过程中提供每一步的内在引导。消融实验揭示了二者之间的互补作用。

    arXiv:2604.00055v2 Announce Type: replace-cross  Abstract: Existing robotic foundation policies are trained primarily via large-scale imitation learning. While such models demonstrate strong capabilities, they often struggle with long-horizon tasks due to distribution shift and error accumulation. While reinforcement learning (RL) can finetune these models, it cannot work well across diverse tasks without manual reward engineering. We propose VLLR, a dense reward framework combining (1) an extrinsic reward from Large Language Models (LLMs) and Vision-Language Models (VLMs) for task progress recognition, and (2) an intrinsic reward based on policy self-certainty. VLLR uses LLMs to decompose tasks into verifiable subtasks and then VLMs to estimate progress to initialize the value function for a brief warm-up phase, avoiding prohibitive inference cost during full training; and self-certainty provides per-step intrinsic guidance throughout PPO finetuning. Ablation studies reveal complement
    
[^468]: 基于高斯过程的质量控制主动学习：实现自主显微镜中鲁棒的结构-性质学习

    Quality-Controlled Active Learning via Gaussian Processes for Robust Structure-Property Learning in Autonomous Microscopy

    [https://arxiv.org/abs/2603.29135](https://arxiv.org/abs/2603.29135)

    提出了一种将好奇心驱动采样与基于简谐振子模型拟合的物理信息质量控制滤波器相结合的门控主动学习框架，可在自主显微镜的结构-性质学习任务中于数据采集阶段自动排除低质量测量数据，实现更鲁棒的学习性能。

    

    自主实验系统在材料研究中的应用日益广泛，用于加速科学发现，但其性能往往受限于低质量、含噪声的数据。这一问题在数据密集型的结构-性质学习任务（如图像到光谱（Im2Spec）和光谱到图像（Spec2Im）转换）中尤为突出，因为标准的主动学习策略可能会错误地优先选择低质量的测量数据。我们提出了一种门控主动学习框架，将好奇心驱动的采样与基于简谐振子模型拟合的物理信息质量控制滤波器相结合，使系统能够在数据采集过程中自动排除低保真度数据。在对含有空间局域噪声的PbTiO3薄膜带激励压电响应谱（BEPS）预采集数据集的评估中，所提出的方法优于随机采样、标准主动学习以及其他对比方法。

    arXiv:2603.29135v3 Announce Type: replace  Abstract: Autonomous experimental systems are increasingly used in materials research to accelerate scientific discovery, but their performance is often limited by low-quality, noisy data. This issue is especially problematic in data intensive structure-property learning tasks such as Image-to-Spectrum (Im2Spec) and Spectrum-to-Image (Spec2Im) translations, where standard active learning strategies can mistakenly prioritize poor quality measurements. We introduce a gated active learning framework that combines curiosity driven sampling with a physics-informed quality control filter based on Simple Harmonic Oscillator model fits, allowing the system to automatically exclude low fidelity data during acquisition. Evaluations on a pre-acquired dataset of band-excitation piezoresponse spectroscopy (BEPS) data from PbTiO3 thin films with spatially localized noise show that the proposed method outperforms random sampling, standard active learning, an
    
[^469]: PRUE：大规模农田边界分割的实用方案

    PRUE: A Practical Recipe for Field Boundary Segmentation at Scale

    [https://arxiv.org/abs/2603.27101](https://arxiv.org/abs/2603.27101)

    本文提出PRUE方法，通过结合U-Net骨干网络、复合损失函数和针对性数据增强，在FTW基准上实现了76% IoU和47% object-F1，为大规模农田边界分割提供了比实例分割模型和地理空间基础模型更实用的解决方案。

    

    大规模的农田边界地图对于农业监测任务至关重要。现有的基于卫星的农田制图深度学习方法对光照、空间尺度和地理位置变化较为敏感。我们使用Fields of The World（FTW）基准，首次对用于全球农田边界勾绘的分割模型和地理空间基础模型（GFMs）进行了系统性评估。我们在统一的实验设置下评估了18个模型，结果表明U-Net语义分割模型在一系列性能和部署指标上优于基于实例的模型和GFM等替代方案。我们提出了一种新的分割方法，该方法结合了U-Net骨干网络、复合损失函数和有针对性的数据增强，以提升真实世界条件下的性能和鲁棒性。我们的模型在FTW上取得了76%的IoU和47%的object-F1，较之前的基线分别提升了6%和9%。

    arXiv:2603.27101v2 Announce Type: replace-cross  Abstract: Large-scale maps of field boundaries are essential for agricultural monitoring tasks. Existing deep learning approaches for satellite-based field mapping are sensitive to illumination, spatial scale, and changes in geographic location. We conduct the first systematic evaluation of segmentation and geospatial foundation models (GFMs) for global field boundary delineation using the Fields of The World (FTW) benchmark. We evaluate 18 models under unified experimental settings, showing that a U-Net semantic segmentation model outperforms instance-based and GFM alternatives on a suite of performance and deployment metrics. We propose a new segmentation approach that combines a U-Net backbone, composite loss functions, and targeted data augmentations to enhance performance and robustness under real-world conditions. Our model achieves a 76% IoU and 47% object-F1 on FTW, an increase of 6% and 9% over the previous baseline. Our approac
    
[^470]: 过程感知人工智能在降雨径流模拟中的应用：一种具有水文过程约束的质量守恒神经框架

    Process-Aware AI for Rainfall-Runoff Modeling: A Mass-Conserving Neural Framework with Hydrological Process Constraints

    [https://arxiv.org/abs/2603.25093](https://arxiv.org/abs/2603.25093)

    提出了一种质量守恒感知器（MCP）框架，通过在单一存储单元中逐步嵌入有界土壤蓄水、入渗、地表积水、地下水位动态等物理水文过程约束，在保证质量守恒的同时提升了降雨径流模拟的预测精度与物理可解释性。

    

    机器学习模型在水文应用中能够达到较高的预测精度，但往往缺乏物理可解释性。质量守恒感知器提供了一种物理感知的人工智能框架，在强制执行守恒原理的同时，允许从数据中学习水文过程关系。在本研究中，我们研究了如何在单个MCP存储单元内逐步嵌入具有物理意义的水文过程表示，从而提高降雨径流模拟的预测能力和可解释性。从最小化的MCP公式出发，我们依次引入了有界土壤蓄水容量、状态相关的导水率、可变孔隙度、入渗能力、地表积水、垂直排水以及非线性地下水位动态。所得到的这一系列过程感知MCP模型层级，在美国大陆五个水文气候区的15个流域上进行了评估。

    arXiv:2603.25093v2 Announce Type: replace  Abstract: Machine learning models can achieve high predictive accuracy in hydrological applications but often lack physical interpretability. The Mass-Conserving Perceptron (MCP) provides a physics-aware artificial intelligence (AI) framework that enforces conservation principles while allowing hydrological process relationships to be learned from data. In this study, we investigate how progressively embedding physically meaningful representations of hydrological processes within a single MCP storage unit improves predictive skill and interpretability in rainfall-runoff modeling. Starting from a minimal MCP formulation, we sequentially introduce bounded soil storage, state-dependent conductivity, variable porosity, infiltration capacity, surface ponding, vertical drainage, and nonlinear water-table dynamics. The resulting hierarchy of process-aware MCP models is evaluated across 15 catchments spanning five hydroclimatic regions of the continen
    
[^471]: 视觉-语言模型中空间变量绑定的双重机制

    The Dual Mechanisms of Spatial Variable Binding in Vision-Language Models

    [https://arxiv.org/abs/2603.22278](https://arxiv.org/abs/2603.22278)

    视觉-语言模型通过两种并发机制表示空间变量绑定——语言模型中间层表示内容无关的空间关系但仅起次要作用，而视觉编码器产生的全局分布式空间信号才是塑造模型预测的主要来源。

    

    许多多模态任务，例如图像描述生成和视觉问答，要求视觉-语言模型（VLM）将物体与其属性和空间关系进行绑定。然而，这些关联在VLM内部的何处以及如何计算仍不清楚。在这项工作中，我们展示了VLM依赖两种并发的机制来表示空间变量绑定。在语言模型主干中，中间层在与物体对应的视觉标记之上表示与内容无关的空间关系。然而，这种机制在塑造模型预测方面仅起次要作用。相反，空间信息的主要来源是视觉编码器，其表示编码了物体的空间布局，并被语言模型主干直接利用。值得注意的是，这种空间信号全局分布在各个视觉标记之间，不仅限于物体区域，还延伸到周围的背景区域。我们验证...（摘要截断）

    arXiv:2603.22278v3 Announce Type: replace-cross  Abstract: Many multimodal tasks, such as image captioning and visual question answering, require vision-language models (VLMs) to bind objects with their properties and spatial relations. Yet it remains unclear where and how such associations are computed within VLMs. In this work, we show that VLMs rely on two concurrent mechanisms to represent spatial variable binding. In the language model backbone, intermediate layers represent content-independent spatial relations on top of visual tokens corresponding to objects. However, this mechanism plays only a secondary role in shaping model predictions. Instead, the dominant source of spatial information originates in the vision encoder, whose representations encode the layout of objects and are directly exploited by the language model backbone. Notably, this spatial signal is distributed globally across visual tokens, extending beyond object regions into surrounding background areas. We vali
    
[^472]: 数据异构下移动边缘网络中的联邦混合专家对齐

    Federated Mixture-of-Experts Alignment on Mobile Edge Networks under Data Heterogeneity

    [https://arxiv.org/abs/2603.21276](https://arxiv.org/abs/2603.21276)

    针对数据异构环境下联邦MoE大模型微调中客户端门控偏好分歧和专家语义模糊两大挑战，本文提出了FedAlign-MoE联邦聚合对齐框架，以在移动边缘网络上实现更好的协同训练。

    

    随着移动边缘设备对端侧大语言模型服务需求的不断增长，混合专家架构因其能在有限计算资源下扩展模型容量而被广泛采用。由于基于MoE的大语言模型微调依赖于隐私敏感的本地数据，联邦学习为在不暴露原始数据的情况下进行协同训练提供了一种天然的范式。然而，将基于MoE的大语言模型微调集成到联邦学习中，面临由客户端间数据异构性引发的两个关键挑战：（i）各不相同的本地数据分布使客户端形成不同的门控偏好，因此直接的参数聚合会产生一个“一刀切”却无法适配任何客户端的全局门控网络；（ii）相同索引的专家在不同设备上发展出互异的语义角色，导致专家语义模糊以及专门化能力退化。为应对这些挑战，我们提出了FedAlign-MoE，一种面向……的联邦聚合对齐框架（摘要在此处截断）。

    arXiv:2603.21276v2 Announce Type: replace-cross  Abstract: The growing demand for on-device large language model (LLM) services on mobile edge devices has driven the adoption of Mixture-of-Experts (MoE) architectures, which scale model capacity with limited computation. Since fine-tuning MoE-based LLMs relies on privacy-sensitive local data, federated learning (FL) offers a natural paradigm for collaborative training without exposing raw data. However, integrating MoE-based LLM fine-tuning into FL faces two critical challenges caused by data heterogeneity across clients: (i) divergent local data distributions drive clients to develop distinct gating preferences, so direct parameter aggregation yields a one-size-fits-none global gating network; and (ii) same-indexed experts develop disparate semantic roles across devices, leading to expert semantic blurring and degraded specialization. To address these challenges, we propose FedAlign-MoE, a federated aggregation alignment framework for 
    
[^473]: 十年深度时间序列预测研究综述

    Deep Time-Series Forecasting in 10 Years: A Survey

    [https://arxiv.org/abs/2603.19899](https://arxiv.org/abs/2603.19899)

    本文从自相关性建模的统一视角系统综述了十年来的深度时间序列预测研究，首次提出同时涵盖骨干架构与损失函数的分类体系，并据此剖析了现有文献的动机与洞见。

    

    自相关性是时间序列的一种普遍特性，即每个观测值都依赖于其前序观测值。在深度时间序列预测中，这带来了两个核心挑战：（1）设计骨干网络架构以建模历史序列中的自相关性；（2）设计损失函数以建模标签序列中的自相关性。近年来，相关研究在应对这些挑战方面取得了长足进展，但目前仍缺乏对这两个方面进行系统性考察的综述。为填补这一空白，本文从自相关性建模的视角对深度时间序列预测进行了综述，并做出了超越现有综述工作的两点贡献：其一，提出了一个同时涵盖骨干网络架构与损失函数的分类体系，而以往综述对损失函数的覆盖较为有限；其二，从统一的自相关性视角分析了所调研文献背后的动机与洞见，提供了一个整体性的（认识框架）。

    arXiv:2603.19899v2 Announce Type: replace-cross  Abstract: Autocorrelation is a common property of time-series, where each observation is dependent on its predecessors. In deep time-series forecasting, it raises two central challenges: (1) designing backbone architectures to model autocorrelation in history sequences, and (2) devising loss functions to model autocorrelation in label sequences. Recent studies have made strides in tackling these challenges, but a systematic survey examining both aspects remains lacking. To bridge this gap, this paper reviews deep time-series forecasting from an autocorrelation modeling perspective, offering two contributions beyond existing surveys. First, it introduces a taxonomy that jointly covers both backbone architectures and loss functions, whereas prior surveys provide limited coverage of the latter. Second, it analyzes the motivations and insights underlying the surveyed literature from a unified autocorrelation perspective, providing a holistic
    
[^474]: 双模态多阶段对抗性安全训练：增强多模态网页代理抵御跨模态攻击的鲁棒性

    Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks

    [https://arxiv.org/abs/2603.04364](https://arxiv.org/abs/2603.04364)

    该论文提出双模态多阶段对抗性安全训练（DMAST）框架，将代理与攻击者的交互建模为两人一般和马尔可夫博弈，通过三阶段协同训练显著增强多模态网页代理抵御同时污染视觉与文本两个观察通道的跨模态欺骗攻击的能力。

    

    处理截图和可访问性树的多模态网页代理正日益被部署用于与网页界面交互，然而其双流架构开启了一个尚未被充分探索的攻击面：向网页DOM注入内容的攻击者可以用一致的欺骗性叙事同时破坏两个观察通道。我们在MiniWob++上的漏洞分析表明，包含视觉组件的攻击远胜于纯文本注入，暴露了以文本为中心的视觉语言模型（VLM）安全训练中的关键缺陷。受此发现启发，我们提出了双模态多阶段对抗性安全训练（DMAST），该框架将代理与攻击者的交互形式化为两人一般和马尔可夫博弈，并通过三阶段流程对双方进行协同训练：（1）从强大的教师模型进行模仿学习，（2）采用新颖的零确认策略进行oracle引导的监督微调，以

    arXiv:2603.04364v2 Announce Type: replace-cross  Abstract: Multimodal web agents that process both screenshots and accessibility trees are increasingly deployed to interact with web interfaces, yet their dual-stream architecture opens an underexplored attack surface: an adversary who injects content into the webpage DOM simultaneously corrupts both observation channels with a consistent deceptive narrative. Our vulnerability analysis on MiniWob++ reveals that attacks including a visual component far outperform text-only injections, exposing critical gaps in text-centric VLM safety training. Motivated by this finding, we propose Dual-Modality Multi-Stage Adversarial Safety Training (DMAST), a framework that formalizes the agent-attacker interaction as a two-player general-sum Markov game and co-trains both players through a three-stage pipeline: (1) imitation learning from a strong teacher model, (2) oracle-guided supervised fine-tuning that uses a novel zero-acknowledgment strategy to 
    
[^475]: 无需世界模型的世界属性：分布性关联与语言模型解码结果的解读

    World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models

    [https://arxiv.org/abs/2603.04317](https://arxiv.org/abs/2603.04317)

    该研究表明，以往从语言模型激活中解码出的“世界属性”在很大程度上也能由静态词嵌入实现，因此解码成功并不能证明语言模型形成了内部世界模型，而可能仅反映了语料库统计中的分布性关联。

    

    越来越多的文献表明，可以从大语言模型（LLM）的激活中线性解码出各种变量，涵盖世界属性（如城市位置和历史人物的寿命）以及情绪和疼痛等。这些发现常被视为语言模型超越表面文本统计、形成内部世界模型的证据。我们证明，对相同或匹配的刺激，静态词嵌入（从语料库统计中学到的固定的、与上下文无关的表示）能够支持大部分相同的解码。在四个已发表的案例中（地点、时间、疼痛和情绪），静态向量可以预测坐标和死亡年份（R² = 0.42-0.59），将疼痛句子与匹配的对照句子区分开（留出AUC为0.85-0.88），并在刻意避免直接点名情绪的故事中分类十二种情绪（AUC为0.84-0.88）。由于静态词嵌入为每个词分配单一的、与上下文无关的向量，

    arXiv:2603.04317v2 Announce Type: replace-cross  Abstract: A growing literature shows that variables can be linearly decoded from the activations of large language models (LLMs). These range from properties of the world, such as the locations of cities and the lifetimes of historical figures, to emotions and pain. Such findings are often taken as evidence that language models go beyond surface text statistics and form internal models of the world. We show that static word embeddings (fixed, context-insensitive representations learned from corpus statistics) of the same or matched stimuli support much of the same decoding. Across four published cases (place, time, pain and emotion), static vectors predict coordinates and year of death (R^2 = 0.42-0.59), separate pain from matched control sentences (held-out AUC 0.85-0.88), and classify twelve emotions in stories written to avoid naming them (AUC 0.84-0.88). Because static embeddings assign each word a single, context-independent vector,
    
[^476]: [b] = [d] - [t] + [p]：自监督语音模型发现音系向量算术

    [b] = [d] - [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic

    [https://arxiv.org/abs/2602.18899](https://arxiv.org/abs/2602.18899)

    自监督语音模型在96种语言的表示空间中将音系特征编码为线性向量，且这些向量支持算术运算（如将[d]-[t]得到的浊音向量加到[p]上即产生[b]），表明语音以可解释、可组合的音系向量形式被表征。

    

    自监督语音模型（S3Ms）已知能够编码丰富的语音信息，然而这些信息的结构方式仍未得到充分探索。我们在96种语言中开展了一项全面研究，以分析S3M表示的底层结构，并特别关注音系向量。我们首先证明，在模型的表示空间中存在与音系特征相对应的线性方向。我们进一步证明，这些音系向量的尺度以连续的方式与其对应音系特征在声学上被实现的程度相关。例如，[d]和[t]之间的差值产生一个浊音向量：将该向量加到[p]上会得到[b]，而对该向量进行缩放则会产生浊音程度的连续谱。这些发现共同表明，S3Ms使用音系上可解释且可组合的向量来编码语音，展示了音系向量算术现象。

    arXiv:2602.18899v4 Announce Type: replace-cross  Abstract: Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vec
    
[^477]: 多探测零冲突哈希（MPZCH）：缓解大规模推荐系统中的嵌入冲突并增强模型新鲜度

    Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders

    [https://arxiv.org/abs/2602.17050](https://arxiv.org/abs/2602.17050)

    该论文提出了一种基于线性探测的多探测零冲突哈希（MPZCH）索引机制，利用可配置探测与主动驱逐策略在保持生产级效率的同时彻底消除大规模推荐系统中的嵌入冲突，并防止陈旧嵌入继承以增强模型新鲜度。

    

    嵌入表是大规模推荐系统的关键组件，用于将高基数分类特征高效地映射为稠密向量表示。然而，随着唯一ID数量的增长，传统的基于哈希的索引方法会遭遇冲突问题，从而降低模型性能和个性化质量。我们提出了多探测零冲突哈希（MPZCH），这是一种基于线性探测的新型索引机制，能够有效缓解嵌入冲突。在合理的表大小设置下，它通常可以完全消除这些冲突，同时保持生产规模的效率。MPZCH利用辅助张量和高性能CUDA内核来实现可配置的探测和主动驱逐策略。通过淘汰过时的ID并重置重新分配的槽位，MPZCH防止了基于哈希方法中典型的陈旧嵌入继承问题，确保新特征能够有效地学习。

    arXiv:2602.17050v4 Announce Type: replace  Abstract: Embedding tables are critical components of large-scale recommendation systems, facilitating the efficient mapping of high-cardinality categorical features into dense vector representations. However, as the volume of unique IDs expands, traditional hash-based indexing methods suffer from collisions that degrade model performance and personalization quality. We present Multi-Probe Zero Collision Hash (MPZCH), a novel indexing mechanism based on linear probing that effectively mitigates embedding collisions. With reasonable table sizing, it often eliminates these collisions entirely while maintaining production-scale efficiency. MPZCH utilizes auxiliary tensors and high-performance CUDA kernels to implement configurable probing and active eviction policies. By retiring obsolete IDs and resetting reassigned slots, MPZCH prevents the stale embedding inheritance typical of hash-based methods, ensuring new features learn effectively from s
    
[^478]: ReLoop：面向可靠的大语言模型优化的结构化建模与行为验证

    ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization

    [https://arxiv.org/abs/2602.15983](https://arxiv.org/abs/2602.15983)

    ReLoop通过结合结构化生成和行为验证，有效缩小了大语言模型在优化代码生成中的可行性与正确性差距。

    

    大语言模型（LLMs）可以将自然语言转化为优化代码，但静默失败构成关键风险：能够执行并返回求解器可行解的代码可能编码了语义上错误的公式——这种可行性与正确性之间的差距在组合问题上高达90个百分点。我们引入了ReLoop，通过两种互补机制来解决这一差距。结构化生成将代码生产分解为四阶段推理链（理解、形式化、综合、验证），从源头防止公式错误。行为验证通过测试公式是否对基于求解器的参数扰动做出正确响应来检测生成过程中存活的错误——这是一种绕过LLM自我审查且无需真实标签的外部语义信号。这两种机制在错误结构上互补：结构化生成在组合问题上带来最大改进。

    arXiv:2602.15983v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) can translate natural language into optimization code, but silent failures pose a critical risk: code that executes and returns solver-feasible solutions may encode semantically incorrect formulations---a feasibility--correctness gap reaching 90 percentage points on compositional problems. We introduce ReLoop, which addresses this gap through two complementary mechanisms. Structured generation decomposes code production into a four-stage reasoning chain (understand, formalize, synthesize, verify), preventing formulation errors at their source. Behavioral verification detects errors that survive generation by testing whether the formulation responds correctly to solver-based parameter perturbation---an external semantic signal that bypasses LLM self-review and requires no ground truth. The two mechanisms are complementary by error structure: structured generation drives the largest gains on compositi
    
[^479]: UniST-Pred：面向中断情形下交通网络时空交通预测的鲁棒统一框架

    UniST-Pred: A Robust Unified Framework for Spatio-Temporal Traffic Forecasting in Transportation Networks Under Disruptions

    [https://arxiv.org/abs/2602.14049](https://arxiv.org/abs/2602.14049)

    提出了UniST-Pred统一时空预测框架，通过解耦时间建模与空间表示学习并采用自适应表示级融合，在交通网络中断等结构与观测不确定性条件下实现鲁棒的交通预测。

    

    时空交通预测是智能交通系统的核心组成部分，支持信号控制和网络级交通管理等多种下游任务。在实际部署中，预测模型必须在结构和观测不确定性条件下运行，而这些条件在模型设计中很少被考虑。近期的方法通过紧密耦合空间与时间建模实现了较强的短期预测性能，但往往以增加复杂性和模块化受限为代价。相比之下，高效的时间序列模型无需依赖显式网络结构即可捕捉长期时间依赖性。我们提出了UniST-Pred，一个统一的时空预测框架，它首先将时间建模与空间表示学习解耦，然后通过自适应表示级融合将两者整合。为评估所提方法的鲁棒性，我们构……（原文摘要在此处被截断）

    arXiv:2602.14049v2 Announce Type: replace-cross  Abstract: Spatio-temporal traffic forecasting is a core component of intelligent transportation systems, supporting various downstream tasks such as signal control and network-level traffic management. In real-world deployments, forecasting models must operate under structural and observational uncertainties, conditions that are rarely considered in model design. Recent approaches achieve strong short-term predictive performance by tightly coupling spatial and temporal modeling, often at the cost of increased complexity and limited modularity. In contrast, efficient time-series models capture long-range temporal dependencies without relying on explicit network structure. We propose UniST-Pred, a unified spatio-temporal forecasting framework that first decouples temporal modeling from spatial representation learning, then integrates both through adaptive representation-level fusion. To assess robustness of the proposed approach, we constr
    
[^480]: RAM-Net：基于稀疏可寻址状态的线性时间序列建模

    RAM-Net: Linear-Time Sequence Modeling with Sparsely Addressable State

    [https://arxiv.org/abs/2602.11958](https://arxiv.org/abs/2602.11958)

    RAM-Net提出用稀疏地址访问取代共享状态的密集访问，将循环状态组织为独立槽数组并通过地址解码器选择少量槽进行读写，从而抑制词元间干扰、提升长距离细粒度记忆能力，同时保持线性时间复杂度。

    

    线性注意力通过固定大小的循环状态，为全注意力提供了一种高效的替代方案。然而，这一状态由所有词元共享，来自不同词元的信息会在其中叠加，产生词元间的相互干扰，从而损害长距离细粒度记忆能力。为解决这一问题，我们提出了RAM-Net，它用基于地址的稀疏访问取代了对共享状态的密集访问。RAM-Net将循环状态组织为固定大小的独立槽数组，并使用一个地址解码器将每个键或查询映射为稀疏地址，在每一步中选择一小部分槽进行写入或读取。这种设计将地址不重叠的词元导向互不相交的槽，抑制了词元间的干扰，同时使每步状态访问的开销仅取决于所选槽的数量，而非状态的总大小。实验表明，RAM-Net优于强大的循环基线模型。

    arXiv:2602.11958v2 Announce Type: replace-cross  Abstract: Linear attention offers an efficient alternative to full attention with a fixed-size recurrent state. However, this state is shared by all tokens, so information from distinct tokens becomes superposed within it and produces inter-token interference that degrades long-range fine-grained recall. To address this issue, we propose RAM-Net, which replaces dense access to a shared state with sparse address-based access. RAM-Net organizes the recurrent state as a fixed-size array of independent slots and uses an Address Decoder that maps each key or query into a sparse address, selecting a small subset of slots to write to or read from at each step. This design directs tokens with non-overlapping addresses to disjoint slots, suppressing inter-token interference, while keeping per-step state access dependent only on the number of selected slots rather than the total state size. Empirically, RAM-Net outperforms strong recurrent baselin
    
[^481]: 揭示多目标对齐中的跨目标干扰

    Uncovering Cross-Objective Interference in Multi-Objective Alignment

    [https://arxiv.org/abs/2602.06869](https://arxiv.org/abs/2602.06869)

    该论文首次系统研究了多目标LLM对齐中的跨目标干扰现象，推导出解释其成因的局部协方差定律，并据此提出了缓解干扰的重加权控制器COVER。

    

    我们研究了大语言模型（LLM）多目标对齐中的一种持续性失败模式：标量化训练仅改善了部分目标，而其他目标却出现退化。我们将这一现象形式化为“跨目标干扰”，并据我们所知，首次对多目标LLM对齐的标量化算法进行了系统性研究。研究表明，这种干扰在各算法中普遍存在，但对具体模型有很强的依赖性。为了理解干扰是如何产生的，我们推导出一条局部协方差定律：在一阶近似下，某个目标的改善或退化取决于其奖励与标量化分数之间协方差的符号。我们将该定律进一步扩展到现代强化微调中的裁剪代理目标，并证明在温和条件下该定律依然成立。基于此定律，我们提出了COVariance-floor Enforced Reweighting（COVER），这是一种单边控制器，能够提升某个目标的……

    arXiv:2602.06869v3 Announce Type: replace  Abstract: We study a persistent failure mode in multi-objective alignment for large language models (LLMs), in which scalarized training improves only some objectives while the others degrade. We formalize this phenomenon as cross-objective interference and, to our knowledge, conduct the first systematic study of scalarization algorithms for multi-objective LLM alignment. The study shows that interference is pervasive across algorithms yet strongly model-dependent. To understand how interference arises, we derive a local covariance law stating that an objective improves or degrades at first order according to the sign of the covariance between its reward and the scalarized score. We extend this law to the clipped surrogate objectives of modern reinforcement fine-tuning and show that it still holds under mild conditions. Building on this law, we propose COVariance-floor Enforced Reweighting (COVER), a one-sided controller that raises an objecti
    
[^482]: 嵌入扰动或能更好地反映大语言模型推理中中间步骤的不确定性

    Embedding Perturbation may Better Reflect Intermediate-Step Uncertainty in LLM Reasoning

    [https://arxiv.org/abs/2602.02427](https://arxiv.org/abs/2602.02427)

    该研究提出用嵌入扰动来度量token敏感性，发现LLM推理中不确定的中间步骤更可能出现在对前文嵌入扰动高度敏感的token上，从而实现对推理过程中不确定性来源的精确定位。

    

    arXiv:2602.02427v3 公告类型：替换 摘要：大语言模型（LLMs）已在各个领域取得重大突破，但仍可能产生不可靠或误导性的输出。为了负责任地应用LLM，不确定性量化技术被用于估计模型对其输出的不确定性，以指示这些输出可能出现问题的概率。对于LLM推理任务而言，不仅需要对最终答案的不确定性进行估计，还需要对中间推理过程的不确定性进行估计，尤其是要识别不确定性在何处产生。这类信息能够在推理过程中实现更细粒度、更有针对性的干预。在本研究中，我们探究了哪些度量指标能够有效定位LLM推理轨迹中的不确定位置。我们的研究表明，不确定的中间续写更可能出现在对前序token嵌入扰动高度敏感的token上。在我们的实验……

    arXiv:2602.02427v3 Announce Type: replace  Abstract: Large Language Models (LLMs) have achieved significant breakthroughs across various domains, but they can still produce unreliable or misleading outputs. For responsible LLM applications, uncertainty quantification techniques are used to estimate a model's uncertainty about its outputs, indicating the likelihood that those outputs may be problematic. For LLM reasoning tasks, it is essential to estimate uncertainty not only in the final answer but also in the intermediate reasoning process, particularly to identify where uncertainty arises. Such information may enable more fine-grained and targeted interventions during inference. In this study, we investigate which metrics can effectively localize uncertain places within an LLM reasoning trajectory. Our study reveals that uncertain intermediate continuations are more likely to occur at tokens that are highly sensitive to perturbations in the embeddings of preceding tokens. In our expe
    
[^483]: 平均场朗之万下降-上升动力学及其相关粒子系统的局部指数稳定性

    Local exponential stability of mean-field Langevin descent-ascent and associated particle system

    [https://arxiv.org/abs/2602.01564](https://arxiv.org/abs/2602.01564)

    本文证明了当初始条件足够接近混合纳什均衡时，熵正则化双人零和博弈的平均场朗之万下降-上升动力学以量化速率局部指数收敛，且有限粒子系统在N的指数长时间尺度上保持这一稳定性。

    

    我们研究了平均场朗之万下降-上升动力学（MFL-DA），这是一种作用于概率测度空间上的耦合优化动力学，用于求解熵正则化的双人零和博弈，并同时研究其相关的相互作用粒子系统。对于一般的非凸-非凹收益函数，Wang和Chizat（COLT 2024）提出了这样一个问题：原始的单时间尺度MFL-DA是否收敛到混合纳什均衡，如果收敛，其收敛速率是多少。我们在Wasserstein空间中给出了局部肯定的答案：如果初始数据足够接近混合纳什均衡，那么平均场动力学将以量化的速率指数快速收敛到该均衡。我们进一步证明，有限-$N$粒子系统在$N$的指数长时间内继承这种稳定性，其指数速率与$N$无关，仅存在一个有限粒子误差下限。结合Mourrat和Pillaud-Vivien最近针对MFL-DA给出的反例——该反例表明全局收敛[不成立]……

    arXiv:2602.01564v3 Announce Type: replace  Abstract: We study the mean-field Langevin descent-ascent (MFL-DA), a coupled optimization dynamics on the space of probability measures for entropically regularized two-player zero-sum games, together with its associated interacting particle system. For general nonconvex-nonconcave payoffs, Wang and Chizat (COLT 2024) asked whether the original single-timescale MFL-DA converges to the mixed Nash equilibrium and, if so, at what rate. We prove a local affirmative answer in Wasserstein space: if the initial datum is sufficiently close to the mixed Nash equilibrium, then the mean-field dynamics converges to it exponentially fast at a quantitative rate. We further show that the finite-$N$ particle system inherits this stability up to times exponential in $N$, with an $N$-independent exponential rate modulo a finite-particle error floor. Combined with the recent counterexample of Mourrat and Pillaud-Vivien for MFL-DA, which shows that global conver
    
[^484]: 面向鲁棒与非光滑凸分布式学习的快速高效异步Gossip算法

    Fast and Efficient Asynchronous Gossip Algorithm for Robust and Non-Smooth Convex Decentralized Learning

    [https://arxiv.org/abs/2601.20571](https://arxiv.org/abs/2601.20571)

    本文提出Goal-PD，一种异步Gossip原始-对偶算法，每个节点仅维护两个变量而与网络度数无关，实现了几乎必然收敛与线性收敛，并通过分布式均值估计中的成对平均特例与经典Gossip算法建立了直接联系。

    

    面向分布式非光滑凸优化的异步原始-对偶方法通常要求每个节点维护 $\mathcal{O}(d)$ 个辅助变量，其中 $d$ 为该节点的度数。这种对度数的依赖增加了内存需求，并可能放大过期信息的影响，尤其是在密集网络中。受分布式学习中节约内存管理这一挑战的启发，我们提出了 Goal-PD，一种基于异步Gossip的原始-对偶算法，无论节点度数如何，每个节点仅需维护两个变量。我们建立了Goal-PD几乎必然收敛到所研究优化问题最小化子的结论，并证明了当目标函数为分段线性二次函数时的线性收敛性。对于分布式均值估计，我们证明成对平均是Goal-PD的一个特例，这在所提出的原始-对偶框架与经典Gossip算法之间建立了直接联系。

    arXiv:2601.20571v3 Announce Type: replace-cross  Abstract: Asynchronous primal-dual methods for decentralized non-smooth convex optimization often require each node to maintain $\mathcal{O}(d)$ auxiliary variables, where $d$ is its degree. This dependence on degree increases memory requirements and can amplify the effects of stale information, especially in dense networks. Motivated by the challenge of frugal memory management in decentralized learning, we introduce Goal-PD, an asynchronous gossip-based primal-dual algorithm that maintains only two variables per node, regardless of the node's degree. We establish almost-sure convergence of Goal-PD to a minimizer of the underlying optimization problem, and prove linear convergence when the objective functions are piecewise linear-quadratic. For decentralized mean estimation, we show that pairwise averaging is a special case of Goal-PD, which establishes a direct link between the proposed primal-dual framework and classical gossip. Exper
    
[^485]: 基于混合整数优化的交叉性公平

    Intersectional Fairness via Mixed-Integer Optimization

    [https://arxiv.org/abs/2601.19595](https://arxiv.org/abs/2601.19595)

    本文提出一个基于混合整数优化（MIO）的统一框架，训练同时具备交叉公平性和内在可解释性的分类器，证明了两种交叉公平性度量（MSD 与 SPSF）在检测最不公平子群体上的等价性，并能将交叉偏见有效控制在可接受阈值以下。

    

    在金融和医疗等高风险领域部署人工智能，需要既公平又透明的模型。虽然包括欧盟《人工智能法案》在内的监管框架要求减轻偏见，但它们对偏见的定义故意保持模糊。与现有研究一致，我们认为真正的公平需要在受保护群体的交叉点上解决偏见问题。我们提出了一个统一框架，利用混合整数优化（MIO）来训练具有交叉公平性和内在可解释性的分类器。我们证明了两种交叉公平性度量（MSD 和 SPSF）在检测最不公平子群体方面的等价性，并通过实验证明我们基于 MIO 的算法在发现偏见方面提升了性能。我们训练了高性能、可解释的分类器，将交叉偏见限制在可接受的阈值以下，为监管合规提供了稳健的解决方案。

    arXiv:2601.19595v2 Announce Type: replace-cross  Abstract: The deployment of Artificial Intelligence in high-risk domains, such as finance and healthcare, necessitates models that are both fair and transparent. While regulatory frameworks, including the EU's AI Act, mandate bias mitigation, they are deliberately vague about the definition of bias. In line with existing research, we argue that true fairness requires addressing bias at the intersections of protected groups. We propose a unified framework that leverages Mixed-Integer Optimization (MIO) to train intersectionally fair and intrinsically interpretable classifiers. We prove the equivalence of two measures of intersectional fairness (MSD and SPSF) in detecting the most unfair subgroup and empirically demonstrate that our MIO-based algorithm improves performance in finding bias. We train high-performing, interpretable classifiers that bound intersectional bias below an acceptable threshold, offering a robust solution for regulat
    
[^486]: 面向纵向医学图像的随机孪生MAE预训练

    Stochastic Siamese MAE Pretraining for Longitudinal Medical Images

    [https://arxiv.org/abs/2512.23441](https://arxiv.org/abs/2512.23441)

    STAMP提出了一种随机孪生MAE预训练框架，通过将MAE重建损失重构为条件变分推断目标、并以两次扫描间的时间差为条件，为纵向医学图像的自监督学习引入了时间感知能力和对疾病演变不确定性的建模。

    

    具有时间感知能力的图像表征对于捕捉纵向医学数据集中3D体数据的疾病进展至关重要。然而，近年来最先进的自监督学习方法（如掩码自编码MAE），尽管具有强大的表征学习能力，却缺乏时间感知能力。在本文中，我们提出了STAMP（基于掩码预训练的随机时间自编码器），这是一个孪生MAE框架，通过以两个输入体数据之间的时间差为条件，以随机过程编码时间信息。与确定性的孪生方法不同——后者虽然比较不同时间点的扫描，但未能考虑疾病演变中固有的不确定性——STAMP通过将MAE重建损失重新构建为条件变分推断目标，以随机方式学习时间动态。我们在两个OCT数据集和一个MRI数据集上评估了STAMP，这些数据集中每位患者均有多次访视记录。

    arXiv:2512.23441v2 Announce Type: replace  Abstract: Temporally aware image representations are crucial for capturing disease progression in 3D volumes of longitudinal medical datasets. However, recent state-of-the-art self-supervised learning approaches like Masked Autoencoding (MAE), despite their strong representation learning capabilities, lack temporal awareness. In this paper, we propose STAMP (Stochastic Temporal Autoencoder with Masked Pretraining), a Siamese MAE framework that encodes temporal information through a stochastic process by conditioning on the time difference between the 2 input volumes. Unlike deterministic Siamese approaches, which compare scans from different time points but fail to account for the inherent uncertainty in disease evolution, STAMP learns temporal dynamics stochastically by reframing the MAE reconstruction loss as a conditional variational inference objective. We evaluated STAMP on two OCT and one MRI datasets with multiple visits per patient. ST
    
[^487]: 重新思考微调：解锁视觉-语言模型的隐藏能力

    Rethinking Fine-Tuning: Unlocking Hidden Capabilities in Vision-Language Models

    [https://arxiv.org/abs/2512.23073](https://arxiv.org/abs/2512.23073)

    提出掩码微调（MFT），通过学习掩码在不修改骨干网络权重的前提下选择性路由预训练连接中的信息，为视觉-语言模型适配提供了优于全量微调和参数高效微调的结构化新方案。

    

    微调已成为适配视觉-语言模型（VLM）的主流范式，然而大多数方法依赖于显式的权重更新，这带来了一个根本性的权衡问题。全量微调（FFT）可能由于跨模态梯度干扰而扰动预训练表示，而参数高效微调（PEFT）方法依赖附加模块（如低秩适配器），可能限制适配能力。在本文中，我们从一个结构选择框架出发重新思考VLM的适配，该框架无需修改骨干网络权重即可适配VLM，并提出了掩码微调（MFT）。MFT通过学习掩码，选择性地经由现有的预训练连接路由信息，动态地发现能够更好地将预训练表示与下游目标对齐的子网络。大量实验表明，MFT为FFT和PEFT提供了一种有效的结构化替代方案，并持续取得更优的性能。

    arXiv:2512.23073v2 Announce Type: replace  Abstract: Fine-tuning has become the dominant paradigm for adapting Vision-Language Models (VLMs), yet most approaches rely on explicit weight updates that introduce a fundamental trade-off. Full Fine-Tuning (FFT) may perturb pretrained representations due to cross-modal gradient interference, whereas Parameter-Efficient Fine-Tuning (PEFT) methods rely on additive modules, such as low-rank adapters, which may limit adaptation capacity. In this paper, we rethink VLM adaptation from a structural selection framework that adapts VLMs without modifying backbone weights, and we propose Mask Fine-Tuning (MFT). MFT learns masks that selectively route information through existing pretrained connections, dynamically uncovering subnetworks that better align pretrained representations with downstream objectives. Extensive experiments show that MFT provides an effective structural alternative to both FFT and PEFT, consistently achieving superior performanc
    
[^488]: 基于核化Stein差异的计算高效拟合优度检验

    Computationally efficient goodness-of-fit tests through kernelized Stein discrepancy

    [https://arxiv.org/abs/2512.20007](https://arxiv.org/abs/2512.20007)

    本文提出一种基于核化Stein差异的计算高效半参数拟合优度检验，并设计了无需重新拟合模型或从中采样的影响调整野自助法来确定检验的显著性水平。

    

    具有难解归一化常数的模型在统计学和机器学习中被广泛使用。评估此类模型的适用性面临重大挑战：从拟合后的模型中获取样本通常需要复杂的采样算法。此外，模型拟合有时需要迭代数值优化，这使得需要重复重新拟合的自助法程序在计算上代价高昂。在本文中，我们利用基于核的检验框架，开发了一种基于核化Stein差异的通用半参数拟合优度检验。我们在一般的干扰参数估计下建立了检验统计量的相合性和渐近零分布。为了构造水平为 $\alpha$ 的检验，我们提出了一种新颖的影响调整野自助法，该方法既不需要重新拟合模型，也不需要从模型中采样。我们证明了所提出的自助检验程序在零假设下的相合性，并……

    arXiv:2512.20007v3 Announce Type: replace-cross  Abstract: Models with intractable normalizing constants are widely used in statistics and machine learning. Assessing the adequacy of such models poses significant challenges: obtaining samples from the fitted model often requires sophisticated sampling algorithms. Moreover, model fitting sometimes requires iterative numerical optimization, making bootstrap procedures that require repeated refitting computationally expensive. In this paper, we leverage the kernel-based testing framework to develop a general semiparametric goodness-of-fit test based on the kernelized Stein discrepancy. We establish the consistency and the asymptotic null distribution of the test statistic under general nuisance estimation. To produce a level-$\alpha$ test, we propose a novel influence-adjusted wild bootstrap that requires neither refitting the model nor sampling from it. We prove the consistency of the proposed bootstrap test procedure under the null and 
    
[^489]: 利用无监督与监督学习对马约拉纳拓扑进行机器学习

    Machine learning Majorana topology using unsupervised and supervised learning

    [https://arxiv.org/abs/2512.13825](https://arxiv.org/abs/2512.13825)

    该论文将无监督与监督学习相结合，证明无需标签的模拟数据即可在现实短无序纳米线的马约拉纳分裂中区分拓扑相与平庸相，并确定两者在参数空间中的转变位置，为实验上识别拓扑提供了有用的工具。

    

    在无监督学习中，深度学习的训练数据不带任何标签，这迫使算法从数据中发现隐藏的模式，从而辨别出有用的信息。从原理上讲，这可能成为识别拓扑序的有力工具，因为拓扑性质并不总是以明显的物理方式（例如拓扑超导电性）表现出来而获得决定性的确认。然而，问题在于无监督学习是一项艰巨的挑战，需要巨大的计算资源，且不一定总能奏效。在本工作中，我们将无监督学习与监督学习相结合，证明了在现实的短无序纳米线中，马约拉纳分裂的无标签（模拟）数据不仅可以区分“拓扑”与“平庸”相，还能确定它们在相关参数空间中发生交叉转变的位置。这可能是在实验中识别拓扑的一种有用工具。

    arXiv:2512.13825v2 Announce Type: replace-cross  Abstract: In unsupervised learning, the training data for deep learning does not come with any labels, thus forcing the algorithm to discover hidden patterns in the data for discerning useful information. This, in principle, could be a powerful tool in identifying topological order since topology does not always manifest in obvious physical ways (e.g., topological superconductivity) for its decisive confirmation. The problem, however, is that unsupervised learning is a difficult challenge, necessitating huge computing resources, which may not always work. In the current work, we combine unsupervised and supervised learning to establish that unlabeled (simulated) data in the Majorana splitting in realistic short disordered nanowires may enable not only a distinction between `topological' and `trivial', but also where their crossover happens in the relevant parameter space. This may be a useful tool in identifying topology in experimental 
    
[^490]: 用于从稳态重构隐藏交互网络的变分物理信息拟设方法

    Variational Physics-Informed Ansatz for Reconstructing Hidden Interaction Networks from Steady States

    [https://arxiv.org/abs/2512.13708](https://arxiv.org/abs/2512.13708)

    提出一种变分物理信息拟设方法，将未知交互算子作为可训练对象并在异质扰动产生的多个稳态约束下最小化残差，实现从稳态观测中唯一重构隐藏交互网络，并给出基于相容矩阵秩的显式有限样本可辨识性条件。

    

    当瞬态轨迹不可用时，从稳态观测中推断交互结构是一个核心的逆问题。在此，我们将该问题表述为单一交互算子与由异质扰动产生的平衡约束之间的同时相容性。我们引入一种变分物理信息拟设，将未知算子表示为可训练对象，并在多个实验中最小化由此产生的稳态残差。在仿射交互设定下，堆叠的平衡方程给出了显式的有限样本可辨识性条件：唯一恢复由消除各实验规范自由度后相容矩阵的秩所决定。在成对、有向、加权、经验拓扑以及部分高阶系统上的合成基准实验验证了这一可辨识性图景，并展示了额外的异质稳态如何改善结构重构效果。

    arXiv:2512.13708v2 Announce Type: replace  Abstract: Inferring interaction structure from steady-state observations is a central inverse problem when transient trajectories are unavailable. Here we formulate this problem as simultaneous compatibility of a single interaction operator with equilibrium constraints generated by heterogeneous perturbations. We introduce a variational physics-informed ansatz that represents the unknown operator as a trainable object and minimizes the resulting steady-state residuals across experiments. In the affine-interaction setting, the stacked equilibrium equations yield explicit finite-sample identifiability conditions: unique recovery is controlled by the rank of the compatibility matrix after elimination of experiment-wise gauge freedom. Synthetic benchmarks on pairwise, directed, weighted, empirical-topology, and selected higher-order systems illustrate this identifiability picture and show how additional heterogeneous steady states improve structur
    
[^491]: 高斯测度间的最优传输与对齐

    Optimal Transportation and Alignment Between Gaussian Measures

    [https://arxiv.org/abs/2512.03579](https://arxiv.org/abs/2512.03579)

    本文解决了可分离希尔伯特空间上非中心化高斯测度之间内积Gromov-Wasserstein对齐这一开放问题，通过等距算子上的二次优化给出精确变分刻画，并推导出紧密的解析上下界。

    

    最优传输（OT）与Gromov-Wasserstein（GW）对齐为比较、转换和聚合异构数据集提供了可解释的几何框架——这些任务在数据科学和机器学习中无处不在。由于这些框架计算代价高昂，大规模应用通常依赖于二次代价下高斯分布的闭式解。本工作对高斯分布下的二次代价OT以及内积GW（IGW）对齐进行了全面研究，填补了文献中的若干空白以拓展其适用性。首先，我们解决了可分离希尔伯特空间上非中心化高斯分布之间IGW对齐这一开放问题，通过在等距算子与余等距算子（有限维情形即为正交矩阵）上的二次优化给出了精确的变分刻画，并推导出了紧密的解析上界和下界。若至少有一个高斯测度是中心化的，……

    arXiv:2512.03579v2 Announce Type: replace  Abstract: Optimal transport (OT) and Gromov-Wasserstein (GW) alignment provide interpretable geometric frameworks for comparing, transforming, and aggregating heterogeneous datasets---tasks ubiquitous in data science and machine learning. Because these frameworks are computationally expensive, large-scale applications often rely on closed-form solutions for Gaussian distributions under quadratic cost. This work provides a comprehensive treatment of Gaussian, quadratic cost OT and inner product GW (IGW) alignment, closing several gaps in the literature to broaden applicability. First, we treat the open problem of IGW alignment between uncentered Gaussians on separable Hilbert spaces by giving an exact variational characterization through a quadratic optimization over isometries and co-isometries (orthogonal matrices in finite dimensions), for which we derive tight analytic upper and lower bounds. If at least one Gaussian measure is centered, th
    
[^492]: 一种足迹感知、面向多样生态系统碳通量预测的高分辨率方法

    A Footprint-Aware, High-Resolution Approach for Carbon Flux Prediction Across Diverse Ecosystems

    [https://arxiv.org/abs/2512.01917](https://arxiv.org/abs/2512.01917)

    该论文提出足迹感知回归（FAR）深度学习框架，通过同时预测涡度相关通量塔的空间足迹与像素级CO₂通量估计，消除了高分辨率升尺度模型中的偏差，并在205个站点年的AMERI-FAR25数据集上验证了其优于传统不考虑足迹的模型。

    

    涡度相关（EC）通量塔提供CO₂通量的原位测量，并作为基于卫星产品导出的预测性“升尺度”模型的地面真值数据。然而，如今许多卫星所解析的空间尺度已小于EC塔的足迹。我们从理论上证明，在异质性景观中使用高分辨率数据训练的升尺度模型必须考虑EC塔的足迹，以避免像素级预测器中的偏差。为解决这一问题，我们提出了足迹感知回归，这是一种深度学习框架，能够同时预测空间足迹和CO₂通量的像素级估计，并证明在训练数据充足的情况下它可以产生无偏的像素级预测。我们在AMERI-FAR25数据集上对FAR进行了验证，该数据集将205个站点年的塔观测数据与相应的Landsat影像相结合，结果表明FAR优于不考虑足迹的模型。FAR将半小时尺度的R²从……

    arXiv:2512.01917v2 Announce Type: replace  Abstract: Eddy-covariance (EC) flux towers provide in situ measurements of $CO_2$ flux and serve as the ground-truth data for predictive `upscaling' models derived from satellite products. However, many satellites now resolve spatial scales smaller than an EC tower's footprint. We show theoretically that upscaling models trained on high-resolution data in heterogeneous landscapes must account for an EC tower's footprint to avoid bias in pixel-level predictors. To address this problem, we introduce Footprint-Aware Regression (FAR), a deep-learning framework that simultaneously predicts spatial footprints and pixel-level estimates of $CO_2$ flux, and show it yields unbiased pixel-level predictions given sufficient training data. We demonstrate FAR on our AMERI-FAR25 dataset, which combines 205 site-years of tower data with corresponding Landsat scenes, and show that FAR outperforms non-footprint-aware models. FAR increased half-hourly $R^2$ from
    
[^493]: 使用平衡微调将大语言模型与生物医学知识对齐

    Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning

    [https://arxiv.org/abs/2511.21075](https://arxiv.org/abs/2511.21075)

    本文发现生物医学文本具有与通用文本截然不同的密集认知不确定性结构，并据此提出双尺度的平衡微调方法，通过词元重加权与序列级重新分配使模型聚焦知识密集样本，在多项医学与生物任务上取得比SFT和DFT更一致的性能提升。

    

    工程化大语言模型以加速生命科学研究，需要与生物医学知识进行稳健的对齐。我们观察到，生物医学文本表现出与通用文本根本不同的不确定性结构：密集的低置信度片段编码的是认知性知识缺口（如密集的因果链、罕见实体），而非通用文本中典型的稀疏偶然性风格变化。基于这一发现，我们提出了平衡微调，这是一种双尺度的后训练方法，将组归一化的词元重加权与序列级的重新分配相结合，使训练资源向表现出密集认知不确定性的知识密集样本倾斜。在医学评估、生物推理、稀疏奖励强化学习和生物表征任务中，在相同的训练设置下，BFT比SFT和DFT提供了更一致的性能提升。当替换GeneAgent（GPT-4o）和VCWorld（Gemini-2.5-Flash）中默认的闭源骨干模型时，BFT-（原文摘要在此处截断）

    arXiv:2511.21075v4 Announce Type: replace-cross  Abstract: Engineering LLMs to accelerate life sciences research requires a robust alignment with biomedical knowledge. We observe that biomedical text exhibits a fundamentally different uncertainty structure from general text: dense low-confidence runs encode epistemic knowledge gaps (dense causal chains, rare entities) rather than the sparse aleatoric stylistic variation typical of general text. Based on this discovery, we propose Balanced Fine-Tuning (BFT), a dual-scale post-training method that combines group-normalized token reweighting with sequence-level reallocation toward knowledge-dense samples exhibiting dense epistemic uncertainty. Across medical evaluation, biological reasoning, sparse-reward RL, and biological representation tasks, BFT provides more consistent gains than SFT and DFT under a shared training setup. When replacing the default closed-source backbones in GeneAgent (GPT-4o) and VCWorld (Gemini-2.5-Flash), the BFT-
    
[^494]: 通过轮次级重要性采样与截断触发归一化稳定长时程LLM智能体的离策略训练

    Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization

    [https://arxiv.org/abs/2511.20718](https://arxiv.org/abs/2511.20718)

    该论文提出SORL方法，通过轮次级重要性采样与截断触发的归一化机制，使策略优化与多轮交互结构对齐并抑制不可靠的离策略梯度更新，从而稳定长时程LLM智能体的离策略强化学习训练，防止性能崩溃。

    

    诸如PPO和GRPO等强化学习（RL）算法被广泛用于训练大语言模型（LLM）以完成多轮智能体任务。然而，在离策略训练流程中，这些方法可能表现出不稳定的优化动态，并容易发生性能崩溃。通过实证分析，我们识别出该设置下两个根本性的不稳定性来源：（1）token级策略优化与轮次结构化交互之间的粒度不匹配；（2）由离策略重要性采样和不准确优势估计所引起的高方差、不可靠的梯度更新。为应对这些挑战，我们提出了SORL（Stabilizing Off-Policy Reinforcement Learning for Long-Horizon Agent Training，面向长时程智能体训练的稳定离策略强化学习方法）。SORL引入了使策略优化与多轮交互结构对齐的机制，并自适应地抑制不可靠的离策略更新，从而带来更加保守和稳健的优化过程。

    arXiv:2511.20718v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) algorithms such as PPO and GRPO are widely used to train large language models (LLMs) for multi-turn agentic tasks. However, in off-policy training pipelines, these methods can exhibit unstable optimization dynamics and are prone to perfor- mance collapse. Through empirical analysis, we identify two fundamental sources of instability in this setting: (1) a granularity mismatch between token-level policy optimization and turn- structured interactions, and (2) high-variance and unreliable gradient updates induced by off- policy importance sampling and inaccurate advantage estimation. To address these challenges, we propose SORL, Stabilizing Off-Policy Reinforcement Learning for Long-Horizon Agent Train- ing. SORL introduces mechanisms that align policy optimization with the structure of multi- turn interactions and adaptively suppress unreliable off-policy updates, yielding more conserva- tive and robu
    
[^495]: 诚实重于准确：通过强化犹豫构建可信赖的语言模型

    Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation

    [https://arxiv.org/abs/2511.11500](https://arxiv.org/abs/2511.11500)

    该论文提出“强化犹豫”（RH）方法，通过将RLVR的二值奖励改为三值奖励（正确+1、弃答0、错误-λ），训练语言模型学会在不确定时主动弃答，从而在诚实性与准确性之间实现可调节的可信权衡。

    

    现代语言模型未能满足可信赖智能的一项基本要求：知道何时不该回答。尽管这些模型在基准测试中取得了令人瞩目的准确率，但它们会产生自信满满的幻觉，即便错误答案可能带来灾难性后果。我们在GSM8K、MedQA和GPQA上的评估显示，尽管明确警告会受严厉惩罚，前沿模型几乎从不放弃作答，这表明提示词无法覆盖“奖励任何回答胜过不回答”的训练倾向。作为补救措施，我们提出强化犹豫（Reinforced Hesitation, RH）：这是对可验证奖励强化学习（RLVR）的一种改进，用三值奖励（回答正确+1、弃答0、回答错误-λ）取代二值奖励。在逻辑谜题上的受控实验表明，改变λ会在帕累托前沿上产生不同的模型，其中每种训练惩罚水平都能产生与其对应风险场景相匹配的最优模型：低惩罚会训练出激进的答题者……

    arXiv:2511.11500v3 Announce Type: replace  Abstract: Modern language models fail a fundamental requirement of trustworthy intelligence: knowing when not to answer. Despite achieving impressive accuracy on benchmarks, these models produce confident hallucinations, even when wrong answers carry catastrophic consequences. Our evaluations on GSM8K, MedQA and GPQA show frontier models almost never abstain despite explicit warnings of severe penalties, suggesting that prompts cannot override training that rewards any answer over no answer. As a remedy, we propose Reinforced Hesitation (RH): a modification to Reinforcement Learning from Verifiable Rewards (RLVR) to use ternary rewards (+1 correct, 0 abstention, -$\lambda$ error) instead of binary. Controlled experiments on logic puzzles reveal that varying $\lambda$ produces distinct models along a Pareto frontier, where each training penalty yields the optimal model for its corresponding risk regime: low penalties produce aggressive answerer
    
[^496]: 面向多步时间序列预测模型训练的二次型直接预测方法

    Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models

    [https://arxiv.org/abs/2511.00053](https://arxiv.org/abs/2511.00053)

    该论文提出了一种新颖的二次型加权学习目标，通过加权矩阵的非对角元素捕捉未来步骤间的标签自相关效应，同时利用非均匀对角元素为不同预测步骤设置异构任务权重，从而同时解决传统均方误差目标的两个缺陷，提升多步时间序列预测模型的训练效果。

    

    arXiv:2511.00053v2 公告类型： replace-cross 摘要：学习目标的设计是训练时间序列预测模型的核心。现有的学习目标（如均方误差）大多将每个未来预测步骤视为独立的、等权重的任务，这导致了以下两个挑战：（1）它们忽视了未来步骤之间的标签自相关效应，导致学习目标存在偏差；（2）它们未能为对应不同未来预测步骤的各项预测任务设置异构的任务权重，从而限制了预测性能。为填补这一空白，我们提出了一种新颖的二次型加权学习目标，能够同时解决上述两个问题。具体而言，加权矩阵的非对角元素用于刻画未来步骤之间的标签自相关效应，而非均匀的对角元素则用于匹配具有不同预测步骤的各项预测任务所偏好的权重。在此基础上，我们提出了二次型直接预测（Quadratic Direct Forecast）方法……

    arXiv:2511.00053v2 Announce Type: replace-cross  Abstract: The design of learning objectives is central to training time-series forecasting models. Existing learning objectives such as mean squared error mostly treat each future step as an independent, equally weighted task, which leads to the following two challenges: (1) they overlook the label autocorrelation effect among future steps, leading to biased learning objectives; (2) they fail to set heterogeneous task weights for different forecasting tasks corresponding to varying future steps, limiting the forecasting performance. To fill this gap, we propose a novel quadratic-form weighted learning objective, addressing both issues simultaneously. Specifically, the off-diagonal elements of the weighting matrix account for the label autocorrelation effect, whereas the non-uniform diagonals are expected to match the preferred weights of the forecasting tasks with varying future steps. On this basis, we propose a Quadratic Direct Forecas
    
[^497]: 面向物理神经网络的时间复用层复用方法

    Time-multiplexed layer reuse for physical neural networks

    [https://arxiv.org/abs/2511.00044](https://arxiv.org/abs/2511.00044)

    提出TIDAL-Net架构，通过利用物理神经网络中快速前向动力学与缓慢权重调整之间的时间尺度分离进行时间复用层复用，从而绕过权重缓慢重调的瓶颈，使物理神经网络能以有限的物理参数实现接近深度网络的大规模训练。

    

    arXiv:2511.00044v4 公告类型：替换 摘要：物理神经网络（PNNs）是下一代计算的有前景候选者，但现有的演示规模仍比现代数字神经网络小几个数量级，而数字神经网络的最新进展正是由可训练参数的快速增长所驱动的。这种情况类似于早期数字神经网络所面临的限制，当时的限制催生了参数复用的思想。我们研究了类似的高效硬件架构可能是什么样的，并特别关注PNNs中权重缓慢重新调整这一常见瓶颈。我们提出了时间索引深层交替层网络，它处于循环神经网络与深度神经网络之间的中间状态，专门针对常见PNN原型的规模和限制而设计。TIDAL-Net利用了许多PNN中存在的快速前向动力学与缓慢可训练权重和偏置之间的时间尺度分离，逐层（原文此处截断）……

    arXiv:2511.00044v4 Announce Type: replace  Abstract: Physical neural networks (PNNs) are promising candidates for next-generation computing, but existing demonstrations remain several orders of magnitude smaller than modern digital neural networks, whose recent advances have been driven by rapid growth in trainable parameters. This situation resembles the constraints of early digital neural networks, which led to ideas around parameter reuse. We investigate what similarly efficient hardware architectures may look like, focusing specifically on the common bottleneck of slow re-adjustment of the weights in PNNs. We propose the Time-Indexed Deep Alternating Layers Network (TIDAL-Net), which occupies an intermediate regime between recurrent and deep neural networks, specifically aimed at the scales and restrictions of common PNN prototypes. TIDAL-Net leverages the timescale separation found in many PNNs between fast forward dynamics and slowly trainable weights and biases, using layer-by-l
    
[^498]: 动作驱动过程用于连续时间控制

    Action-Driven Processes for Continuous-Time Control

    [https://arxiv.org/abs/2510.26672](https://arxiv.org/abs/2510.26672)

    本文通过动作驱动过程统一了随机过程与强化学习的视角，证明最小化策略驱动分布与奖励驱动分布之间的KL散度等价于最大熵强化学习，并将其应用于脉冲神经网络。

    

    强化学习的核心在于动作——即针对环境观察所作出的决策。动作在随机过程建模中同样具有基础性地位，因为它们触发不连续的状态转移，并使信息能够在大型复杂系统中流动。本文通过动作驱动过程统一了随机过程与强化学习这两个视角，并展示了其在脉冲神经网络中的应用。借助“控制即推断”的思想，我们证明：对于适当定义的动作驱动过程，最小化策略驱动的真实分布与奖励驱动的模型分布之间的Kullback-Leibler散度，等价于最大熵强化学习。

    arXiv:2510.26672v3 Announce Type: replace-cross  Abstract: At the heart of reinforcement learning are actions -- decisions made in response to observations of the environment. Actions are equally fundamental in the modeling of stochastic processes, as they trigger discontinuous state transitions and enable the flow of information through large, complex systems. In this paper, we unify the perspectives of stochastic processes and reinforcement learning through action-driven processes, and illustrate their application to spiking neural networks. Leveraging ideas from control-as-inference, we show that minimizing the Kullback-Leibler divergence between a policy-driven true distribution and a reward-driven model distribution for a suitably defined action-driven process is equivalent to maximum entropy reinforcement learning.
    
[^499]: PyDPF：一个用于可微粒子滤波的Python软件包

    PyDPF: A Python Package for Differentiable Particle Filtering

    [https://arxiv.org/abs/2510.25693](https://arxiv.org/abs/2510.25693)

    本文提出了基于PyTorch框架构建的Python软件包PyDPF，通过统一的API实现了多种可微粒子滤波算法，使这些方法更易于被广大研究群体使用。

    

    状态空间模型（SSMs）是时间序列分析中广泛使用的工具。在由现实世界数据产生的复杂系统中，通常采用粒子滤波（PF）——一种高效的蒙特卡洛方法——来估计与观测序列相对应的隐藏状态。应用粒子滤波需要指定系统的参数形式及其参数，而这些往往是未知的，必须进行估计。基于梯度的优化技术无法直接应用于标准粒子滤波器，因为滤波器本身是不可微的。然而，最近提出的几种方法通过修改重采样步骤使粒子滤波变得可微。在本文中，我们基于流行的PyTorch框架，以统一的API实现了多种此类可微粒子滤波器（DPFs）。我们的实现使这些算法能够更容易地为更广泛的研究群体所使用。

    arXiv:2510.25693v4 Announce Type: replace-cross  Abstract: State-space models (SSMs) are a widely used tool in time series analysis. In the complex systems that arise from real-world data, it is common to employ particle filtering (PF), an efficient Monte Carlo method for estimating the hidden state corresponding to a sequence of observations. Applying particle filtering requires specifying both the parametric form and the parameters of the system, which are often unknown and must be estimated. Gradient-based optimisation techniques cannot be applied directly to standard particle filters, as the filters themselves are not differentiable. However, several recently proposed methods modify the resampling step to make particle filtering differentiable. In this paper, we present an implementation of several such differentiable particle filters (DPFs) with a unified API built on the popular PyTorch framework. Our implementation makes these algorithms easily accessible to a broader research c
    
[^500]: DistDF：时间序列预测需要联合分布Wasserstein对齐

    DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment

    [https://arxiv.org/abs/2510.24574](https://arxiv.org/abs/2510.24574)

    提出DistDF框架，通过最小化一种可证明上界于条件分布差异的联合分布Wasserstein差异来对齐预测与标签序列的分布，解决了传统均方误差在标签序列存在自相关时的估计偏差问题。

    

    训练时间序列预测模型需要将模型预测结果的条件分布与标签序列的条件分布对齐。标准的直接预测方法通常采用最小化条件负对数似然的方式，一般通过均方误差来估计。然而，当标签序列存在自相关性时，这种估计会产生偏差。在本文中，我们提出DistDF，通过最小化预测序列与标签序列条件分布之间的分布差异来实现对齐。由于此类条件差异难以从有限的时间序列观测中估计，我们为时间序列预测引入了一种联合分布Wasserstein差异，该差异可被证明是所需条件差异的上界。所提出的差异是易于计算的、可微分的，并且可以方便地与基于梯度的优化方法相结合。大量实验（摘要内容在此处截断）

    arXiv:2510.24574v3 Announce Type: replace-cross  Abstract: Training time-series forecasting models requires aligning the conditional distribution of model forecasts with that of the label sequence. The standard direct forecast (DF) approach resorts to minimizing the conditional negative log-likelihood, typically estimated by the mean squared error. However, this estimation proves biased when the label sequence exhibits autocorrelation. In this paper, we propose DistDF, which achieves alignment by minimizing a distributional discrepancy between the conditional distributions of forecast and label sequences. Since such conditional discrepancies are difficult to estimate from finite time-series observations, we introduce a joint-distribution Wasserstein discrepancy for time-series forecasting, which provably upper bounds the conditional discrepancy of interest. The proposed discrepancy is tractable, differentiable, and readily compatible with gradient-based optimization. Extensive experime
    
[^501]: 超越半圆律：具有指定平衡态的自由扩散模型

    Beyond the Semicircle: Free Diffusion Models with Prescribed Equilibria

    [https://arxiv.org/abs/2510.22778](https://arxiv.org/abs/2510.22778)

    该论文发现状态依赖的自由波动率能突破常系数自由扩散只能收敛到半圆律的固有局限，并为任意充分正则的紧支撑目标谱分布显式构造出具有指定平衡态的自由扩散模型。

    

    一类日益增多的机器学习对象——协方差矩阵与Gram矩阵、核矩阵与注意力矩阵、MIMO信道矩阵、密度算子——本质上以谱（特征值分布）而非坐标向量的形式存在。为这类数据构建去噪扩散模型时，逐坐标地对特征值加噪不仅是形式上的不优雅：它会收敛到错误的极限，因为特征值之间存在排斥效应而非独立运动。自由概率论提供了正确的前向过程，用自由卷积取代经典卷积，用Voiculescu共轭变量取代得分函数，但常系数自由扩散自身存在一个隐蔽的局限：无论目标分布是什么，其唯一可能的平衡态都是半圆律。我们证明，状态依赖的自由波动率可以消除这一限制。对于任何充分正则的紧支撑目标分布，我们通过一个闭式……显式构造……

    arXiv:2510.22778v4 Announce Type: replace-cross  Abstract: A growing class of machine-learning objects -- covariance and Gram matrices, kernel and attention matrices, MIMO channel matrices, density operators -- are naturally spectra rather than coordinate vectors. Building a denoising diffusion model for such data by corrupting eigenvalues coordinatewise is not merely elegant: it converges to the wrong limit, because eigenvalues repel rather than move independently. Free probability theory supplies the correct forward process, with free convolution replacing classical convolution and Voiculescu's conjugate variable replacing the score, but constant-coefficient free diffusions have a hidden limitation of their own: their only possible equilibrium is the semicircular law, whatever the target distribution looks like. We show that state-dependent free volatility removes this restriction. For any sufficiently regular compactly supported target law, we explicitly construct, through a closed-
    
[^502]: 仅凭原始数据统计量预测核回归学习曲线

    Predicting kernel regression learning curves from only raw data statistics

    [https://arxiv.org/abs/2510.14878](https://arxiv.org/abs/2510.14878)

    该论文提出 Hermite 特征结构假设（HEA），证明仅用经验协方差矩阵和目标函数的多项式分解这两个原始数据统计量，即可在真实图像数据集上准确预测核回归的学习曲线。

    

    我们研究了在包括 CIFAR-5m、SVHN 和 ImageNet 在内的真实数据集上使用常见旋转不变核的核回归问题。我们提出了一个理论框架，仅需两个测量量即可预测学习曲线（测试风险与样本量的关系）：经验数据协方差矩阵和目标函数 $f_*$ 的经验多项式分解。关键的新思想是对核在各向异性数据分布下的特征值和特征函数进行解析近似。这些特征函数类似于数据的 Hermite 多项式，因此我们将这一近似称为 Hermite 特征结构假设（Hermite eigenstructure ansatz，HEA）。我们针对高斯数据证明了 HEA，并且发现真实图像数据通常“足够高斯”，使得 HEA 在实践中能很好地成立，这使我们能够通过应用先前将核特征结构与测试风险联系起来的结果来预测学习曲线。在核回归之外，我们通过实证发现，处于……的多层感知机（MLP）……（摘要原文在此处截断）

    arXiv:2510.14878v3 Announce Type: replace-cross  Abstract: We study kernel regression with common rotation-invariant kernels on real datasets including CIFAR-5m, SVHN, and ImageNet. We give a theoretical framework that predicts learning curves (test risk vs. sample size) from only two measurements: the empirical data covariance matrix and an empirical polynomial decomposition of the target function $f_*$. The key new idea is an analytical approximation of a kernel's eigenvalues and eigenfunctions with respect to an anisotropic data distribution. The eigenfunctions resemble Hermite polynomials of the data, so we call this approximation the Hermite eigenstructure ansatz (HEA). We prove the HEA for Gaussian data, but we find that real image data is often "Gaussian enough" for the HEA to hold well in practice, enabling us to predict learning curves by applying prior results relating kernel eigenstructure to test risk. Extending beyond kernel regression, we empirically find that MLPs in the
    
[^503]: 以量子比特为中心的Transformer用于表面码解码

    Qubit-centric Transformer for Surface Code Decoding

    [https://arxiv.org/abs/2510.11593](https://arxiv.org/abs/2510.11593)

    提出了一种基于Transformer的新型量子纠错解码器QCT，利用以量子比特为中心的注意力机制和融合量子码拓扑结构的图掩码方法，将伴随式转换为量子比特标记以有效识别逻辑错误。

    

    对于可靠的大规模量子计算而言，量子纠错（QEC）对于保护分布在多个物理量子比特上的逻辑信息至关重要。借助深度学习领域的最新进展，基于神经网络的解码器已成为提高量子纠错可靠性的一种有前景的方法。我们提出了以量子比特为中心的Transformer（QCT），这是一种新颖且通用的量子纠错解码器，基于具有以量子比特为中心的注意力机制的Transformer架构。我们的解码器通过专门的嵌入策略，将输入的伴随式从稳定子域转换为以量子比特为中心的标记。这些以量子比特为中心的标记经过注意力层的处理，以有效识别潜在的逻辑错误。此外，我们引入了一种基于图的掩码方法，该方法结合了量子码的拓扑结构，强制注意力聚焦于相关的量子比特相互作用。在各种码距下的表面码实验中……

    arXiv:2510.11593v3 Announce Type: replace-cross  Abstract: For reliable large-scale quantum computation, quantum error correction (QEC) is essential to protect logical information distributed across multiple physical qubits. Taking advantage of recent advances in deep learning, neural network-based decoders have emerged as a promising approach to improve the reliability of QEC. We propose the qubit-centric transformer (QCT), a novel and universal QEC decoder based on a transformer architecture with a qubit-centric attention mechanism. Our decoder transforms input syndromes from the stabilizer domain into qubit-centric tokens via a specialized embedding strategy. These qubit-centric tokens are processed through attention layers to effectively identify the underlying logical error. Furthermore, we introduce a graph-based masking method that incorporates the topological structure of quantum codes, enforcing attention toward relevant qubit interactions. Across various code distances for su
    
[^504]: HARL-A：基于IsaacLab的可扩展异构多智能体对抗强化学习基准测试框架

    HARL-A: An Extensible Benchmark Framework for Heterogeneous Multi-Agent Adversarial Reinforcement Learning in IsaacLab

    [https://arxiv.org/abs/2510.01264](https://arxiv.org/abs/2510.01264)

    该论文提出了HARL-A，一个基于IsaacLab的开源可扩展基准测试框架，支持在任意数量团队和任意机器人形态组合下进行异构多智能体对抗强化学习的可扩展训练与评估。

    

    机器人领域的对抗性多智能体强化学习（MARL）研究进展一直受到缺乏共享、可扩展基础设施的阻碍，该基础设施需要在高保真物理仿真中支持异构的智能体形态。现有框架要么专注于合作任务，要么依赖简化的物理引擎，要么提供难以扩展的孤立实现。我们提出了HARL-A，这是一个基于IsaacLab构建的开源、持续维护的框架，能够在形态多样的机器人团队之间进行对抗性策略的可扩展训练与基准测试，支持任意数量的团队以及每个团队中任意混合的机器人形态。HARL-A通过对抗性多智能体支持扩展了HARL算法库和IsaacLab，并贡献了三个组件：（1）一个模块化软件架构，降低了定义新的异构对抗环境的工程开销；（2）一套包含三个基准测试环境的测试套件……

    arXiv:2510.01264v2 Announce Type: replace  Abstract: Progress in adversarial multi-agent reinforcement learning (MARL) for robotics has been hampered by a lack of shared, extensible infrastructure that supports heterogeneous agent morphologies in high-fidelity physics simulation. Existing frameworks either focus on cooperative tasks, rely on simplified physics engines, or provide isolated implementations that are difficult to extend. We present HARL-A, an open-source, actively maintained framework built on IsaacLab that enables scalable training and benchmarking of adversarial policies across morphologically diverse robot teams with any number of teams and any mix of robot morphologies per team. HARL-A extends the HARL algorithm library and IsaacLab with adversarial multi-agent support and contributes three components: (1) a modular software architecture that reduces the engineering overhead of defining new heterogeneous adversarial environments, (2) a suite of three benchmark environm
    
[^505]: PiERN：面向高精度计算与推理融合的令牌级路由

    PiERN: Token-Level Routing for Integrating High-Precision Computation and Reasoning

    [https://arxiv.org/abs/2509.18169](https://arxiv.org/abs/2509.18169)

    PiERN提出了一种物理隔离专家路由网络架构，通过在令牌级别路由计算与推理，使大型语言模型能够在单一思维链中迭代交替地执行高精度数值计算与推理，在准确率、响应延迟、令牌消耗和能耗等方面均优于直接微调与主流多智能体方法。

    

    复杂系统上的任务需要高精度数值计算来支持决策。然而，当前的大型语言模型（LLM）即使具备增强的推理能力，在现有架构下也无法将此类计算作为一种内在且可解释的能力加以整合。为此，我们提出了物理隔离专家路由网络，这是一种在令牌级别引导计算与推理的架构，从而能够在单一思维链内实现迭代交替。我们在代表性的计算-推理任务上对PiERN进行了系统评估，包括PDEBench和电池管理任务。结果表明，PiERN不仅比直接微调LLM取得更高的准确率，而且与主流多智能体方法相比，在响应延迟、令牌使用量、GPU能耗和专家路由准确率方面均有显著改进，同时没有出现明显的性能下降。

    arXiv:2509.18169v4 Announce Type: replace-cross  Abstract: Tasks on complex systems require high-precision numerical computation to support decisions. However, current large language models (LLMs), even with enhanced reasoning capabilities, cannot integrate such computations as an intrinsic and interpretable capability with existing architectures. To this end, we propose Physically-isolated Experts Routing Network (PiERN), an architecture that directs computation and reasoning at token level, thereby enabling iterative alternation within a single chain of thought. We systematically evaluate PiERN on representative computation-reasoning tasks, including PDEBench and battery management tasks. Results show that PiERN achieves not only higher accuracy than directly finetuning LLMs but also significant improvements in response latency, token usage, GPU energy consumption, and experts routing accuracy compared with mainstream multi-agent approaches, while exhibiting no significant degradatio
    
[^506]: DRtool：一个用于分析高维聚类结果的交互式工具

    DRtool: An Interactive Tool for Analyzing High-Dimensional Clusterings

    [https://arxiv.org/abs/2509.04603](https://arxiv.org/abs/2509.04603)

    本文开发了交互式工具DRtool，通过新的聚类验证技术（包括可视化评估和假设检验）帮助分析师识别高维数据中容易被忽视的过度聚类问题。

    

    当我们面对新数据时，常常会进行聚类分析，以便更好地理解数据的结构以及数据中存在的典型样本。然而，数据复杂性和维度的不断提升使得这一步骤变得非常棘手。高维数据中大量的噪声模糊了模式和趋势，使得聚类难以区分。因此，聚类发现工具和聚类验证工具必须加以改进，以应对高维数据带来的困难。非线性降维是朝着正确方向迈出的一步，但众所周知，即使是这些方法也可能产生虚假的结构，尤其是在使用不当时。一个常常未被有经验的眼睛察觉的常见现象是数据的过度聚类。作为这些努力的延续，我们开发了新的聚类验证技术，包括可视化评估和一种假设检验方法，帮助分析师识别……（注：原摘要在此处不完整）

    arXiv:2509.04603v4 Announce Type: replace-cross  Abstract: When faced with new data, we often conduct a cluster analysis to obtain a better understanding of the data's structure and the archetypical samples present in the data. However, the increases in data complexity and dimensionality have made this step very tricky. The large proportion of noise in high-dimensional data blurs patterns and trends, making clusters difficult to distinguish. As such, cluster-discovery tools and cluster-verification tools must be adapted to address the difficulties of high-dimensional data. Nonlinear dimension reduction is a step in the right direction, but even these methods are known to produce false structures, especially when mishandled. A common phenomenon that often goes undetected by the untrained eye is over-clustering of the data. In continuation of these efforts, we developed new cluster verification techniques, including visual assessments and a hypothesis test, that help analysts distinguish
    
[^507]: 打破镜像：基于激活的LLM评估器自我偏好缓解方法

    Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators

    [https://arxiv.org/abs/2509.03647](https://arxiv.org/abs/2509.03647)

    本文提出利用通过对比激活添加（CAA）和优化方法构建的转向向量，在无需重新训练的情况下于推理时缓解LLM评估器的不合理自我偏好偏差，最多可降低97%，显著优于提示和直接偏好优化基线。

    

    大型语言模型（LLM）日益被用作自动评估器，但它们存在“自我偏好偏差”：即倾向于偏好自己的输出而胜过其他模型的输出。这种偏差破坏了评估流程的公平性和可靠性，尤其是在偏好调优和模型路由等任务中。我们研究了轻量级的转向向量能否在推理阶段缓解这一问题而无需重新训练。我们引入了一个精心构建的数据集，将自我偏好偏差区分为合理的自我偏好示例和不合理的自我偏好示例，并采用两种方法构建转向向量：对比激活添加（CAA）和基于优化的方法。我们的结果表明，转向向量可以将不合理的自我偏好偏差降低高达97%，显著优于提示和直接偏好优化基线。然而，转向向量在某些情况下表现不稳定……

    arXiv:2509.03647v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on 
    
[^508]: 联邦学习中梯度反演攻击的实际可行性

    Practical Feasibility of Gradient Inversion Attacks in Federated Learning

    [https://arxiv.org/abs/2508.19819](https://arxiv.org/abs/2508.19819)

    本文系统评估了梯度反演攻击在现实联邦学习系统中的可行性，发现现代性能优化的视觉模型能够有效抵御有意义的图像重构，而已报告的攻击成功多依赖于理想化的上界实验设置。

    

    梯度反演攻击通常被视为联邦学习中的一种严重隐私威胁，近期的研究报告了在有利实验设置下越来越强的图像重构效果。然而，这类攻击在实际部署的现代、性能优化的系统中是否可行仍不清楚。在这项工作中，我们评估了梯度反演在基于图像的联邦学习中的实际可行性。我们在多个数据集和任务上开展了系统性研究，包括图像分类和目标检测，使用了当代分辨率下的经典视觉架构。我们的结果表明，尽管在某些遗留或过渡性设计中、且在高度限制性假设下梯度反演仍然可行，但现代性能优化的模型始终能够在视觉上抵抗有意义的重构。我们进一步证明，许多已报告的攻击成功案例依赖于理想化的上界实验设置。

    arXiv:2508.19819v3 Announce Type: replace-cross  Abstract: Gradient inversion attacks are often presented as a serious privacy threat in federated learning, with recent work reporting increasingly strong reconstructions under favorable experimental settings. However, it remains unclear whether such attacks are feasible in modern, performance-optimized systems deployed in practice. In this work, we evaluate the practical feasibility of gradient inversion for image-based federated learning. We conduct a systematic study across multiple datasets and tasks, including image classification and object detection, using canonical vision architectures at contemporary resolutions. Our results show that while gradient inversion remains possible for certain legacy or transitional designs under highly restrictive assumptions, modern, performance-optimized models consistently resist meaningful reconstruction visually. We further demonstrate that many reported successes rely on upper-bound settings, s
    
[^509]: AEGIS：面向多租户深度学习训练的运行时引导GPU共置系统

    AEGIS: Runtime-Guided GPU Collocation for Multi-Tenant Deep Learning Training

    [https://arxiv.org/abs/2508.19073](https://arxiv.org/abs/2508.19073)

    AEGIS是一个服务器规模的运行时调度系统，通过在单一调度循环中集成内存可行性评估、部署后观察、运行时压力过滤和OOM感知恢复机制，实现了多租户深度学习训练任务在共享GPU服务器上的安全高效共置。

    

    深度学习训练通常运行在共享的多租户GPU服务器上，独占式分配虽然能够提供隔离，但可能导致资源利用不足并增加排队时间。共置可以提高效率，但不考虑干扰的部署方式可能造成严重的性能下降，而不准确的内存信息则可能导致内存溢出（OOM）失败。我们提出了AEGIS，一个服务器规模的运行时调度系统，用于在共享多GPU服务器上对深度学习训练工作负载进行受控共置。AEGIS在单一调度循环中集成了内存可行性评估、部署后观察、运行时压力过滤、部署以及OOM感知恢复。在部署之后，AEGIS会先观察工作负载的活动情况，然后才允许进一步的共置，并利用低开销的遥测数据来判断某个GPU是否可以安全地接受额外的工作。OOM失败会在逐渐更安全的内存条件下触发重试，最终……

    arXiv:2508.19073v4 Announce Type: replace-cross  Abstract: Deep learning training commonly runs on shared multi-tenant GPU servers, where exclusive allocation provides isolation but can leave resources underutilized and increase queueing time. Collocation can improve efficiency, but interference-agnostic placement may cause severe slowdowns, while inaccurate memory information can lead to out-of-memory (OOM) failures.   We present AEGIS, a server-scale runtime scheduling system for controlled collocation of deep learning training workloads on shared multi-GPU servers. AEGIS integrates memory feasibility, post-placement observation, runtime-pressure filtering, placement, and OOM-aware recovery in a single scheduling loop. After placement, AEGIS observes workload activity before permitting further collocation, then uses low-overhead telemetry to determine whether a GPU can safely accept additional work. OOM failures trigger retries under progressively safer memory conditions, eventually 
    
[^510]: PuzzleJAX：一个面向推理与学习的基准

    PuzzleJAX: A Benchmark for Reasoning and Learning

    [https://arxiv.org/abs/2508.16821](https://arxiv.org/abs/2508.16821)

    PuzzleJAX是一个GPU加速的益智游戏引擎与描述语言，可动态编译PuzzleScript风格的游戏，从而为树搜索、强化学习和LLM推理能力提供大规模、多样化任务的快速基准测试。

    

    我们介绍了PuzzleJAX，这是一个GPU加速的益智游戏引擎与描述语言，旨在支持对树搜索、强化学习以及大语言模型（LLM）推理能力进行快速基准测试。与现有的GPU加速学习环境（它们为固定的游戏集合提供硬编码实现）不同，PuzzleJAX允许对其领域特定语言（DSL）所能表达的任何游戏进行动态编译。该DSL遵循PuzzleScript——一个流行且易于上手的在线益智游戏设计引擎。在本文中，我们在PuzzleJAX中对自2013年发布以来由专业设计师和休闲创作者在PuzzleScript上设计的数千款游戏中的数百款进行了验证，从而证明了PuzzleJAX能够覆盖一个广泛、富有表现力且与人类相关的任务空间。通过分析搜索、学习和语言模型在这些游戏上的表现，我们展示了PuzzleJAX能够自然地表达……

    arXiv:2508.16821v2 Announce Type: replace  Abstract: We introduce PuzzleJAX, a GPU-accelerated puzzle game engine and description language designed to support rapid benchmarking of tree search, reinforcement learning, and LLM reasoning abilities. Unlike existing GPU-accelerated learning environments that provide hard-coded implementations of fixed sets of games, PuzzleJAX allows dynamic compilation of any game expressible in its domain-specific language (DSL). This DSL follows PuzzleScript, which is a popular and accessible online game engine for designing puzzle games. In this paper, we validate in PuzzleJAX several hundred of the thousands of games designed in PuzzleScript by both professional designers and casual creators since its release in 2013, thereby demonstrating PuzzleJAX's coverage of an expansive, expressive, and human-relevant space of tasks. By analyzing the performance of search, learning, and language models on these games, we show that PuzzleJAX can naturally express 
    
[^511]: 基于扩散语言模型的跨模态受控分子生成

    Cross-Modality Controlled Molecule Generation with Diffusion Language Model

    [https://arxiv.org/abs/2508.14748](https://arxiv.org/abs/2508.14748)

    提出模块化框架CMCM-DLM，通过结构控制模块和性质控制模块的分阶段协同设计，使预训练扩散语言模型无需重新训练即可支持分子结构与化学性质等跨模态异构约束的受控分子生成。

    

    分子数据类型的日益丰富，催生了对能够灵活融合跨模态异构约束的生成模型的需求。然而，现有的基于SMILES的扩散模型通常针对固定的条件模态设计，引入新约束往往需要重新训练模型。为解决这一局限，我们提出了CMCM-DLM，一个模块化框架，它扩展了预训练的扩散模型，使其无需重新训练主干网络即可支持异构分子约束。我们使用两种互补的模态来展示CMCM-DLM：分子结构和化学性质。具体而言，结构控制模块（SCM）在扩散早期步骤中引导生成分子的骨架结构，而性质控制模块（PCM）则随后引导生成过程朝向目标化学性质。这种分阶段设计使模型能够……

    arXiv:2508.14748v2 Announce Type: replace-cross  Abstract: The increasing variety of molecular data creates a need for generative models that can flexibly incorporate heterogeneous constraints across modalities. However, existing SMILES-based diffusion models are typically designed for a fixed conditioning modality, and introducing new constraints often requires retraining the model. To address this limitation, we propose Cross-Modality Controlled Molecule Generation with Diffusion Language Model (CMCM-DLM), a modular framework that extends a pre-trained diffusion model to support heterogeneous molecular constraints without retraining the backbone. We demonstrate CMCM-DLM using two complementary modalities: molecular structure and chemical properties. Specifically, a Structure Control Module (SCM) guides early diffusion steps to establish the molecular scaffold, while a Property Control Module (PCM) subsequently steers generation toward target chemical properties. This staged design en
    
[^512]: 贝叶斯优化中的直接遗憾优化

    Direct Regret Optimization in Bayesian Optimization

    [https://arxiv.org/abs/2507.06529](https://arxiv.org/abs/2507.06529)

    提出了一种直接遗憾优化新方法，通过从候选模型与采集函数中蒸馏并利用高斯过程集成生成模拟BO轨迹，训练端到端的决策Transformer联合学习最优模型与非短视采集策略，从而显式最小化多步遗憾。

    

    贝叶斯优化（BO）是优化昂贵黑盒函数的一种强大范式。传统的BO方法通常依赖于相互独立的手工设计采集函数和针对底层函数的代理模型，并且往往以短视（贪心）的方式运行。在本文中，我们提出了一种新颖的直接遗憾优化方法，该方法通过从一组候选模型和采集函数中蒸馏，联合学习最优模型与非短视的采集策略，并明确以最小化多步遗憾为目标。我们的框架利用具有不同超参数的高斯过程（GP）集成来生成模拟的BO轨迹，每条轨迹由从传统采集函数池中抽取的采集函数引导，并由贝叶斯提前停止准则终止。这些轨迹用于训练一个端到端的决策Transformer，使其能够选择下一次查询以改善最终目标，并遵循密集的训练…（摘要在此处截断）

    arXiv:2507.06529v2 Announce Type: replace  Abstract: Bayesian optimization (BO) is a powerful paradigm for optimizing expensive black-box functions. Traditional BO methods typically rely on separate hand-crafted acquisition functions and surrogate models for the underlying function, and often operate in a myopic manner. In this paper, we propose a novel direct regret optimization approach that jointly learns the optimal model and non-myopic acquisition by distilling from a set of candidate models and acquisitions, and explicitly targets minimizing the multi-step regret. Our framework leverages an ensemble of Gaussian Processes (GPs) with varying hyperparameters to generate simulated BO trajectories, each guided by an acquisition function drawn from a pool of conventional choices and terminated by a Bayesian early stop criterion. These trajectories train an end-to-end Decision Transformer that selects the next query so as to improve the ultimate objective, following a dense training spa
    
[^513]: 利用Wasserstein分布鲁棒优化改进Mixup校准

    Improving Mixup Calibration with Wasserstein Distributionally Robust Optimization

    [https://arxiv.org/abs/2506.17874](https://arxiv.org/abs/2506.17874)

    本文提出DRO-Augment框架，将Wasserstein分布鲁棒优化与Mixup数据增强相结合，有效缓解了腐蚀鲁棒性与模型校准之间的权衡，在保持腐蚀准确率的同时显著降低了期望校准误差（ECE）。

    

    在许多实际应用中，确保深度神经网络（DNN）的鲁棒性和稳定性至关重要，特别是对于面临各种输入扰动的图像分类任务。虽然基于Mixup的数据增强技术已被广泛采用，以增强训练模型的抗扰动能力，但我们的实验揭示了一个重要的腐蚀鲁棒性-校准权衡：更强的基于Mixup的增强可以提高对腐蚀数据的鲁棒性，但同时会显著增加期望校准误差（ECE）。为了解决这一挑战，我们提出了DRO-Augment框架，该框架将Wasserstein分布鲁棒优化（W-DRO）与各种基于Mixup的数据增强策略相结合，以缓解这种权衡。我们的方法在强Mixup增强下大幅降低了ECE，同时在CIFAR-10、CIFAR-100等数据集上基本保持了腐蚀准确率。

    arXiv:2506.17874v3 Announce Type: replace-cross  Abstract: In many real-world applications, ensuring the robustness and stability of deep neural networks (DNNs) is crucial, particularly for image classification tasks that encounter various input perturbations. While Mixup-based data augmentation techniques have been widely adopted to enhance the resilience of trained models against such perturbations, our experiments reveal an important corruption robustness-calibration trade-off: stronger Mixup-based augmentation can improve robustness against corrupted data while substantially increasing expected calibration error (ECE). To address this challenge, we introduce DRO-Augment, a framework that integrates Wasserstein Distributionally Robust Optimization (W-DRO) with various Mixup-based data augmentation strategies to mitigate this trade-off. Our method substantially reduces ECE under strong Mixup-based augmentation while largely preserving corruption accuracy across CIFAR-10, CIFAR-100, C
    
[^514]: Time-o1：时间序列预测需要变换后的标签对齐

    Time-o1: Time-Series Forecasting Needs Transformed Label Alignment

    [https://arxiv.org/abs/2505.17847](https://arxiv.org/abs/2505.17847)

    Time-o1通过将标签序列变换为去相关且区分显著性的分量，训练模型对齐最显著分量，从而提出一种变换增强的损失函数，有效缓解标签自相关性并减少任务数量，在时间序列预测中达到最先进性能且兼容多种预测模型。

    

    训练时间序列预测模型在损失函数设计方面面临独特的挑战。大多数现有方法采用时间均方误差，但本研究揭示了其两个关键局限性：(1) 它忽略了标签自相关性的存在，使其偏离了真实标签序列的似然；(2) 它涉及过多的任务数量，使优化变得复杂，尤其是在长期预测中。为了解决这些问题，我们提出了Time-o1，一种面向时间序列预测的变换增强损失函数。其核心思想是将标签序列变换为具有区分性显著性的去相关分量，然后训练模型对齐最显著的分量，从而有效缓解标签自相关性并减少任务数量。实验表明，Time-o1实现了最先进的性能，并且与各种预测模型兼容。

    arXiv:2505.17847v3 Announce Type: replace-cross  Abstract: Training time-series forecasting models poses unique challenges in loss function design. Most existing approaches adopt temporal mean squared error, but this study reveals two critical limitations: (1) it ignores the presence of label autocorrelation, which biases it from the true label sequence likelihood; (2) it involves excessive number of tasks, which complicates optimization, especially for long-term forecasting. To address these issues, we introduce Time-o1, a transform-enhanced loss function for time-series forecasting. The central idea is to transform the label sequence into decorrelated components with discriminated significance. Models are then trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Experiments demonstrate that Time-o1 achieves state-of-the-art performance and is compatible with various forecast models. Code is available at https
    
[^515]: KO：基于动理学启发的神经优化器与偏微分方程模拟方法

    KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches

    [https://arxiv.org/abs/2505.14777](https://arxiv.org/abs/2505.14777)

    提出了一种基于动理学与偏微分方程的即插即用优化器KO，它将参数动力学建模为粒子系统并通过离散化玻尔兹曼输运方程引入随机相互作用，从而提升参数多样性、缓解权重凝聚并保持收敛保证。

    

    arXiv:2505.14777v2 公告类型：replace-cross 摘要：为神经网络设计有效的优化算法仍然是一个根本性的挑战，而现有的大多数方法都依赖于基于梯度更新的启发式扩展。我们提出了KO（动理学启发优化器，Kinetics-inspired Optimizer），这是一个基于动理学理论和偏微分方程的即插即用优化模块。KO将参数动力学建模为一个粒子系统，通过对玻尔兹曼输运方程的离散化引入随机相互作用，从而增强标准的梯度更新。这一机制自然地促进了参数多样性，并缓解了权重凝聚（weight condensation）现象——即参数坍缩到低维子空间的倾向，这一现象与泛化能力退化密切相关。我们提供了严格的理论分析和物理解释，证明KO在保持收敛性保证的同时，能够可证明地增加参数多样性。在图像分类任务上的大量实验……

    arXiv:2505.14777v2 Announce Type: replace-cross  Abstract: The design of effective optimization algorithms for neural networks remains a fundamental challenge, and most existing methods rely on heuristic extensions of gradient-based updates. We introduce KO (Kinetics-inspired Optimizer), a plug-and-play optimization module grounded in kinetic theory and partial differential equations. KO models parameter dynamics as a particle system, augmenting standard gradient updates with stochastic interactions induced by a discretization of the Boltzmann transport equation. This mechanism naturally promotes parameter diversity and mitigates weight condensation, the tendency of parameters to collapse into low-dimensional subspaces, a phenomenon closely associated with degraded generalization. We provide both a rigorous theoretical analysis and a physical interpretation, showing that KO provably increases parameter diversity while preserving convergence guarantees. Extensive experiments on image cl
    
[^516]: 掩码微调助力大语言模型性能提升

    Boosting Large Language Models with Mask Fine-Tuning

    [https://arxiv.org/abs/2503.22764](https://arxiv.org/abs/2503.22764)

    提出掩码微调（MFT）这一新颖的大语言模型微调范式，通过学习并应用二值掩码、在不更新模型权重的情况下精心打破模型结构完整性，从而在不同领域和骨干网络上获得一致的性能提升。

    

    大语言模型（LLM）通常被整合进主流的优化流程之中。然而，保持模型的完整性对于获得良好性能是否是不可或缺的，这一问题仍未被充分探索。在本工作中，我们提出了掩码微调，这是一种新颖的大语言模型微调范式，其表明精心地打破模型的结构完整性，可以在不更新模型权重的情况下出人意料地提升性能。MFT以标准的LLM微调目标作为监督，学习并应用二值掩码到已经过良好优化的模型上。基于完全微调后的模型，MFT使用相同的微调数据集，在不同领域和不同骨干网络上实现了一致的性能提升（例如，LLaMA2-7B/3.1-8B在IFEval上平均提升2.70/4.15分）。详细的消融实验和分析从多个角度对所提出的MFT进行了考察，包括稀疏比率和损失面等。此外，……

    arXiv:2503.22764v3 Announce Type: replace-cross  Abstract: The large language model (LLM) is typically integrated into the mainstream optimization protocol. However, it remains underexplored whether maintaining the model integrity is \textit{indispensable} for promising performance. In this work, we introduce Mask Fine-Tuning (MFT), a novel LLM fine-tuning paradigm demonstrating that carefully breaking the model's structural integrity can surprisingly improve performance without updating model weights. MFT learns and applies binary masks to well-optimized models, using the standard LLM fine-tuning objective as supervision. Based on fully fine-tuned models, MFT uses the same fine-tuning datasets to achieve consistent performance gains across domains and backbones (e.g., an average gain of 2.70/4.15 on IFEval with LLaMA2-7B/3.1-8B). Detailed ablation studies and analyses examine the proposed MFT from different perspectives, including the sparse ratio and the loss surface. Additionally, w
    
[^517]: 无需网络无混杂性假设的网络干扰下因果效应估计

    Causal Effect Estimation under Networked Interference without Networked Unconfoundedness Assumption

    [https://arxiv.org/abs/2502.19741](https://arxiv.org/abs/2502.19741)

    本文提出一种利用网络中单元间交互模式来恢复三类潜在混杂因素的框架，从而在不依赖网络无混杂性假设的条件下实现网络干扰下的因果效应估计。

    

    在存在网络干扰的情况下，从观测数据中估计因果效应是一个重要且具有挑战性的问题。大多数现有方法主要依赖于网络无混杂性假设，该假设保证了网络效应的可识别性。然而，由于观测数据中固有存在的潜在混杂因素，这一假设经常被违反，从而阻碍了网络效应的识别。为解决这一问题，我们利用网络中单元之间丰富的交互模式，这些模式为恢复潜在混杂因素提供了宝贵的信息。基于这一洞察，我们开发了一个混杂因素恢复框架，该框架显式地刻画了网络环境中三类潜在混杂因素：仅影响单元自身的混杂因素、仅影响单元邻居的混杂因素，以及同时影响两者的混杂因素。基于该框架，我们设计了一个使用可识别表示的网络效应估计器。

    arXiv:2502.19741v4 Announce Type: replace  Abstract: Estimating causal effects under networked interference from observational data is a crucial yet challenging problem. Most existing methods mainly rely on the networked unconfoundedness assumption, which guarantees the identification of networked effects. However, this assumption is often violated due to the latent confounders inherent in observational data, thereby hindering the identification of networked effects. To address this issue, we leverage the rich interaction patterns between units in networks, which provide valuable information for recovering these latent confounders. Building on this insight, we develop a confounder recovery framework that explicitly characterizes three categories of latent confounders in networked settings: those affecting only the unit, those affecting only the unit's neighbors, and those influencing both. Based on this framework, we design a networked effect estimator using identifiable representation
    
[^518]: 面向加密货币交易分析的大语言模型：以比特币为例的案例研究

    Large Language Models for Cryptocurrency Transaction Analysis: A Bitcoin Case Study

    [https://arxiv.org/abs/2501.18158](https://arxiv.org/abs/2501.18158)

    本文首次将大语言模型应用于比特币等真实加密货币交易图分析，提出三层能力评估框架、人类可读的图表示格式LLM4TG和连接增强的交易图采样算法，克服了传统黑盒模型难以解释和捕捉细微行为模式的局限。

    

    加密货币已被广泛使用，然而当前用于分析交易的方法通常依赖于不透明的黑盒模型。尽管这些模型可能达到较高的性能，但其输出通常难以解释和调整，使得捕捉细微的行为模式变得困难。大语言模型（LLMs）有潜力弥补这些不足，但它们在这一领域的能力在很大程度上仍未被探索，尤其是在网络犯罪检测方面。在本文中，我们通过将大语言模型应用于真实世界的加密货币交易图来验证这一假设，重点关注比特币——这一研究最多、应用最广泛的区块链网络之一。我们提出了一个三层评估框架来衡量LLM的能力：基础指标、特征概述和情境解释。这包括一种新的、人类可读的图表示格式LLM4TG，以及一种连接增强的交易图采样算法……

    arXiv:2501.18158v4 Announce Type: replace-cross  Abstract: Cryptocurrencies are widely used, yet current methods for analyzing transactions often rely on opaque, black-box models. While these models may achieve high performance, their outputs are usually difficult to interpret and adapt, making it challenging to capture nuanced behavioral patterns. Large language models (LLMs) have the potential to address these gaps, but their capabilities in this area remain largely unexplored, particularly in cybercrime detection. In this paper, we test this hypothesis by applying LLMs to real-world cryptocurrency transaction graphs, with a focus on Bitcoin, one of the most studied and widely adopted blockchain networks. We introduce a three-tiered framework to assess LLM capabilities: foundational metrics, characteristic overview, and contextual interpretation. This includes a new, human-readable graph representation format, LLM4TG, and a connectivity-enhanced transaction graph sampling algorithm, 
    
[^519]: 具有隐藏混杂因素的线性常微分方程系统的可辨识性分析

    Identifiability Analysis of Linear ODE Systems with Hidden Confounders

    [https://arxiv.org/abs/2410.21917](https://arxiv.org/abs/2410.21917)

    本文系统分析了含隐藏混杂因素的线性常微分方程系统的可辨识性，分别研究了潜在混杂因素无因果关系但遵循特定函数形式（如时间多项式）演化，以及潜在混杂因素之间具有由有向无环图描述的因果依赖关系这两种情况，填补了该领域的空白。

    

    arXiv:2410.21917v3 公告类型：replace-cross 摘要：对线性常微分方程（ODE）系统进行可辨识性分析是针对这些系统做出可靠因果推断的必要前提。尽管在系统完全可观测的场景下，可辨识性已得到充分研究，但当潜在变量与系统发生交互作用时，其可辨识性的条件仍未被探索。本文旨在通过系统性地分析包含隐藏混杂因素的线性ODE系统的可辨识性来填补这一空白。具体而言，我们研究了此类系统的两种情况。在第一种情况中，潜在混杂因素之间不存在因果关系，但其演化遵循特定的函数形式，例如时间 $t$ 的多项式函数。随后，我们将这一分析扩展到隐藏混杂因素之间存在因果依赖的场景，其中潜在变量的因果结构由有向无环图（DAG）描述。

    arXiv:2410.21917v3 Announce Type: replace-cross  Abstract: The identifiability analysis of linear Ordinary Differential Equation (ODE) systems is a necessary prerequisite for making reliable causal inferences about these systems. While identifiability has been well studied in scenarios where the system is fully observable, the conditions for identifiability remain unexplored when latent variables interact with the system. This paper aims to address this gap by presenting a systematic analysis of identifiability in linear ODE systems incorporating hidden confounders. Specifically, we investigate two cases of such systems. In the first case, latent confounders exhibit no causal relationships, yet their evolution adheres to specific functional forms, such as polynomial functions of time $t$. Subsequently, we extend this analysis to encompass scenarios where hidden confounders exhibit causal dependencies, with the causal structure of latent variables described by a Directed Acyclic Graph (
    
[^520]: 当解释相互竞争时：不确定性下的策略感知选择

    When Explanations Compete: Policy-Aware Selection Under Uncertainty

    [https://arxiv.org/abs/2410.05479](https://arxiv.org/abs/2410.05479)

    该论文提出了一个策略感知的解释选择框架，通过结合资格规则、双向帕累托筛选和策略感知排序，从不确定性感知解释方法生成的多个候选解释中，依据预测置信度、不确定性和应用约束进行最优选择。

    

    不确定性感知的解释方法通常会为同一预测产生多个备选解释。在它们之间进行选择需要一种策略来平衡预测置信度、不确定性和应用约束。本文提出了一个框架，用于将此类策略应用于一组固定的已生成解释。候选解释通过不确定性变化、预测方向，以及在可用情况下相对于决策边界的区间位置来表征。该框架将这些属性与资格规则、可选的双向帕累托筛选以及策略感知排序相结合。一个虚构的前列腺癌示例说明了不同的解释目的如何从同一候选集合中得出不同的选择。我们使用校准解释在分类、阈值回归和普通回归任务上对该框架进行了实例化。在41个基准数据集上，单一解释的平均候选数量范围从11.57到21.75。

    arXiv:2410.05479v2 Announce Type: replace  Abstract: Uncertainty-aware explanation methods often produce several alternatives for the same prediction. Selecting among them requires a policy for balancing prediction confidence, uncertainty, and application constraints. This paper presents a framework for applying such policies to a fixed set of generated explanations. Candidates are characterised by uncertainty change, prediction direction, and, when available, interval position relative to a decision boundary. The framework combines these properties with eligibility rules, optional bidirectional Pareto screening, and policy-aware ranking. A fictitious prostate-cancer example illustrates how different explanatory purposes lead to different selections from the same candidate set. We instantiate the framework with Calibrated Explanations for classification, thresholded regression, and plain regression. Across 41 benchmark datasets, mean candidate counts range from 11.57 to 21.75 for singl
    
[^521]: 基于递增批大小与衰减学习率的锐度感知最小化算法的收敛性研究

    Convergence of Sharpness-Aware Minimization Algorithms using Increasing Batch Size and Decaying Learning Rate

    [https://arxiv.org/abs/2409.09984](https://arxiv.org/abs/2409.09984)

    本文从理论上证明了使用递增批大小或衰减学习率（如余弦退火、线性学习率）的GSAM算法的收敛性，并通过数值实验表明递增批大小相比恒定批大小能获得更低的最坏情况ℓ∞自适应锐度。

    

    锐度感知最小化（SAM）算法及其变体，包括间隙引导SAM（GSAM），已成功地通过在训练中寻找经验损失的平坦局部极小值来提升深度神经网络模型的泛化能力。同时，理论和实践均已证明，增大批大小或衰减学习率可以避免经验损失的尖锐局部极小值。本文考虑了采用递增批大小或衰减学习率（如余弦退火或线性学习率）的GSAM算法，并从理论上证明了其收敛性。此外，我们通过数值实验比较了使用与不使用递增批大小的SAM（GSAM）算法，得出结论：与使用恒定批大小和学习率相比，使用递增批大小能够实现更低的最坏情况 ℓ∞ 自适应锐度。

    arXiv:2409.09984v2 Announce Type: replace  Abstract: The sharpness-aware minimization (SAM) algorithm and its variants, including gap guided SAM (GSAM), have been successful at improving the generalization capability of deep neural network models by finding flat local minima of the empirical loss in training. Meanwhile, it has been shown theoretically and practically that increasing the batch size or decaying the learning rate avoids sharp local minima of the empirical loss. In this paper, we consider the GSAM algorithm with increasing batch sizes or decaying learning rates, such as cosine annealing or linear learning rate, and theoretically show its convergence. Moreover, we numerically compare SAM (GSAM) with and without an increasing batch size and conclude that using an increasing batch size { achieves a lower worst-case $\ell_\infty$ adaptive sharpness} than compared with using a constant batch size and learning rate.
    
[^522]: 基于扩散模型的视频编辑：综述

    Diffusion Model-Based Video Editing: A Survey

    [https://arxiv.org/abs/2407.07111](https://arxiv.org/abs/2407.07111)

    本文是一篇综述，系统梳理了基于扩散模型的视频编辑技术的理论基础、方法分类与演化脉络，探讨了点编辑、姿态引导人体视频编辑等新兴应用，并提出新的V2VBench基准对该领域进行全面对比评估。

    

    扩散模型的快速发展极大地推动了图像和视频应用的发展，使“所见即所想”成为现实。其中，视频编辑受到了广泛关注，相关研究活动迅速增长，因此有必要对现有文献进行全面而系统的回顾。本文综述了基于扩散模型的视频编辑技术，包括理论基础和实际应用。我们首先概述了数学公式化表述以及图像领域的关键方法。随后，我们根据核心技术的内在联系对视频编辑方法进行分类，描绘出其演化轨迹。本文还深入探讨了新颖的应用，包括基于点的编辑和姿态引导的人体视频编辑。此外，我们使用新提出的V2VBench进行了全面的对比评估。

    arXiv:2407.07111v2 Announce Type: replace-cross  Abstract: The rapid development of diffusion models (DMs) has significantly advanced image and video applications, making "what you want is what you see" a reality. Among these, video editing has gained substantial attention and seen a swift rise in research activity, necessitating a comprehensive and systematic review of the existing literature. This paper reviews diffusion model-based video editing techniques, including theoretical foundations and practical applications. We begin by overviewing the mathematical formulation and image domain's key methods. Subsequently, we categorize video editing approaches by the inherent connections of their core technologies, depicting evolutionary trajectory. This paper also dives into novel applications, including point-based editing and pose-guided human video editing. Additionally, we present a comprehensive comparison using our newly introduced V2VBench. Building on the progress achieved to date
    
[^523]: 高频交易的数据驱动度量方法

    Data-driven measures of high-frequency trading

    [https://arxiv.org/abs/2405.08101](https://arxiv.org/abs/2405.08101)

    该论文利用在纳斯达克专有数据上训练的机器学习模型，首次生成了2010至2023年美国股票的流动性提供型与需求型高频交易日度度量指标，并发现供给端高频交易与更多信息获取、更多知情交易和更低买卖价差相关。

    

    公开数据无法识别高频交易（HFT），且标准代理指标无法区分流动性提供型策略与流动性需求型策略。我们通过在纳斯达克专有数据上训练机器学习模型，将观测到的高频交易活动映射到公开的日内变量，从而克服了这一度量难题。应用这一映射方法，我们生成了2010年至2023年期间所有美国股票的流动性提供型与流动性需求型高频交易的日度度量指标。这些度量指标在很大程度上涵盖了标准代理指标，并捕捉到了后者遗漏的时间序列变化。利用泛欧交易所巴黎分所的专有数据，我们提供了证据表明该方法可以泛化到不同市场，并在训练后数年仍保持预测能力。这一14年的面板数据使我们能够研究高频交易与市场质量随时间的演变。供给端高频交易始终与公告前的更多信息获取、更多的知情交易以及更低的买卖价差相关，而需求端高频交易（原文在此处截断）……

    arXiv:2405.08101v4 Announce Type: replace-cross  Abstract: Public data do not identify high-frequency trading (HFT), and standard proxies do not separate liquidity-supplying from liquidity-demanding strategies. We overcome this measurement challenge by training machine learning models on proprietary Nasdaq data to map observed HFT activity to public intraday variables. Applying this mapping, we generate daily measures of liquidity-supplying and liquidity-demanding HFT for all U.S. stocks from 2010 to 2023. The measures largely subsume standard proxies and capture time-series variation that those proxies miss. Using proprietary Euronext Paris data, we provide evidence that the approach generalizes across markets and remains predictive years after training. The 14-year panel lets us study HFT and market quality over time. Supply-side HFT is consistently associated with greater pre-announcement information acquisition, more informed trading, and lower bid-ask spreads, while demand-side HF
    
[^524]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    
[^525]: 基于感受野袋的快速、可解释且确定性的时间序列分类

    Fast, Interpretable, and Deterministic Time Series Classification With a Bag-of-Receptive-Fields

    [https://arxiv.org/abs/2311.18029](https://arxiv.org/abs/2311.18029)

    本文提出了BORF（感受野袋），一种快速、可解释且确定性的时间序列分类变换方法，克服了现有黑盒分类器难以理解以及可解释方法因依赖随机化导致解释不稳定的问题。

    

    时间序列分类文献的当前趋势是通过在集成混合模型中组合多个模型、在复杂且富有表现力的特征空间中表示时间序列，以及从同一时间序列的不同表示中提取特征，来开发越来越精确的算法。由于这种对预测性能的过度关注，最好的时间序列分类器都是黑盒模型，从人类的角度难以理解。即使是那些被认为可解释的方法，例如基于形状特征的方法，也依赖随机化来保持计算效率。这给可解释性带来了挑战，因为每次运行的解释都可能不同。鉴于这些局限性，我们提出了感受野袋（Bag-Of-Receptive-Field，BORF），一种快速、可解释且确定性的时间序列变换方法。在经典的模式袋方法基础上，我们弥合了卷积算子与（摘要在此处截断）

    arXiv:2311.18029v2 Announce Type: replace-cross  Abstract: The current trend in the literature on Time Series Classification is to develop increasingly accurate algorithms by combining multiple models in ensemble hybrids, representing time series in complex and expressive feature spaces, and extracting features from different representations of the same time series. As a consequence of this focus on predictive performance, the best time series classifiers are black-box models, which are not understandable from a human standpoint. Even the approaches that are regarded as interpretable, such as shapelet-based ones, rely on randomization to maintain computational efficiency. This poses challenges for interpretability, as the explanation can change from run to run. Given these limitations, we propose the Bag-Of-Receptive-Field (BORF), a fast, interpretable, and deterministic time series transform. Building upon the classical Bag-Of-Patterns, we bridge the gap between convolutional operator
    
[^526]: 概率性真正无序规则集

    Probabilistic Truly Unordered Rule Sets. (arXiv:2401.09918v1 [cs.LG])

    [http://arxiv.org/abs/2401.09918](http://arxiv.org/abs/2401.09918)

    本论文提出了概率性真正无序规则集（TURS）方法，用于解决规则集学习中的三个缺点：强加顺序、重叠冲突和多类别目标分类问题。通过利用规则集的概率特性来解决重叠冲突，并形式化定义学习问题。

    

    最近人们经常重视规则集学习，因为它具有可解释性。然而，现有的方法存在一些缺点。首先，大多数现有方法在规则之间明确或隐含地强加顺序，这使得模型更难以理解。其次，由于处理重叠引起的冲突（即，被多个规则覆盖的实例）的困难，现有方法通常不考虑概率规则。第三，对于多类别目标的学习分类规则研究不足，因为大多数现有方法专注于二分类或通过"一对其余"方法进行多类别分类。为了解决这些缺点，我们提出了TURS，即真正无序规则集。为了解决重叠规则引起的冲突，我们提出了一种新颖的模型，利用我们的规则集的概率特性，只有当它们具有相似的概率输出时允许规则重叠。我们接下来对学习问题进行了形式化定义。

    Rule set learning has recently been frequently revisited because of its interpretability. Existing methods have several shortcomings though. First, most existing methods impose orders among rules, either explicitly or implicitly, which makes the models less comprehensible. Second, due to the difficulty of handling conflicts caused by overlaps (i.e., instances covered by multiple rules), existing methods often do not consider probabilistic rules. Third, learning classification rules for multi-class target is understudied, as most existing methods focus on binary classification or multi-class classification via the ``one-versus-rest" approach.  To address these shortcomings, we propose TURS, for Truly Unordered Rule Sets. To resolve conflicts caused by overlapping rules, we propose a novel model that exploits the probabilistic properties of our rule sets, with the intuition of only allowing rules to overlap if they have similar probabilistic outputs. We next formalize the problem of lear
    

