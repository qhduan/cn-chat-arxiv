# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TorchCurves: Differentiable Parametric Curves](https://arxiv.org/abs/2611.14516) | TorchCurves是一个将B样条和勒让德多项式参数曲线作为可学习基元的可微分PyTorch库，支持端到端优化，可广泛应用于连续特征嵌入、KAN网络和函数回归等任务。 |
| [^2] | [AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning of Language Models](https://arxiv.org/abs/2611.14515) | 提出了AutoScaffold框架，由冻结的教师模型自动将学生的训练经验转化为自适应脚手架（任务类共享策略与困难任务专用提示），在无需人工干预的情况下提升语言模型可验证奖励强化学习的效果。 |
| [^3] | [Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations](https://arxiv.org/abs/2611.14514) | 该论文提出“对比发现”框架，用互斥且穷尽的竞争性解释集合取代单一假设检验，通过证据对信念在集合内的重新分配来评估实验、确定被支持的替代解释，从而改进开放式科学发现。 |
| [^4] | [Pyrite: Building and Benchmarking Protein-Ligand Pose Ranking Agents](https://arxiv.org/abs/2611.14511) | 提出了包含1500个蛋白质-配体构象对的成对构象排序基准Pyrite，发现专家化学家的排序表现接近随机水平，并开发了两个基于大语言模型的智能体（PyriteReasoner和PyriteFeaturizer）来实现自动化构象排序。 |
| [^5] | [When Does the Real Wiring Win? Testing connectome structure-function claims against unmeasured physiology](https://arxiv.org/abs/2611.14508) | 该论文指出连接组结构-功能比较可能混淆“连线本身”与“模型调参偏差”，提出区分三种生理参数调节方案（P1/P2/P3），并用合成真值、安慰剂零假设和三个果蝇回路模型检验哪些结构-功能结论经得起这些偏差的校正。 |
| [^6] | [Continual-ARC: Measuring Skill Acquisition over a Working Lifetime](https://arxiv.org/abs/2611.14505) | 提出 Continual-ARC 基准，基于三个 ARC-AGI 版本构建终身任务流，使基于记忆和基于权重的持续学习系统能够在同一预算评分标准下公平比较技能习得能力。 |
| [^7] | [Emergent Biological Capabilities in a Foundation Model for Molecular Interactions](https://arxiv.org/abs/2611.14504) | 该论文提出了仅凭蛋白质-配体序列训练、无需任务监督和结构坐标的分子相互作用基础模型Bonbon，其单一表示空间在功能、残基和键三个层次涌现出零样本生物能力，并能泛化到分布外复合物。 |
| [^8] | [Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents](https://arxiv.org/abs/2611.14493) | 该论文通过三阶段比较研究系统评估了三种企业分析智能体评估框架，发现基于轨迹的LLM评判框架在评估稳定性（97%配置达标）和已知风险检测（F1=0.90）方面综合表现最佳。 |
| [^9] | [ReTrace: Distilling Tool-Using Agents into Planner-Free Workflows](https://arxiv.org/abs/2611.14492) | ReTrace通过从智能体的推理解释中提取每步背后的条件并跨运行合并为规则文档，将SOP和少量智能体运行日志蒸馏成由代码驱动的无规划器可执行工作流，在大规模流程自动化中以更低的延迟和成本实现接近LLM智能体的完成率（93.8%对96.8%）。 |
| [^10] | [TimeSeek: Multi-Task and Multi-Domain Pretraining for Generalized Time Series Analytics](https://arxiv.org/abs/2611.14491) | TimeSeek 通过领域与任务异质性下的选择性信息共享机制，实现了对多领域时间序列模式的自适应建模，并在预训练中统一多个分析任务，从而构建出性能更优的广义时间序列分析模型。 |
| [^11] | [C$^2$-Routing: Complementary Cropping and Compression for Efficient Vision-Language Inference](https://arxiv.org/abs/2611.14469) | 提出C²-Routing方法，利用裁剪（突出小目标与局部细节）与压缩（保留全局上下文）的互补性，根据图像和问题自适应路由选择缩减策略，并配合安全门机制，在降低视觉语言模型计算开销的同时保留回答问题的关键信息。 |
| [^12] | [Bayesian inverse problems and epidemic-threshold uncertainty for a heterogeneous SIS reaction-diffusion model](https://arxiv.org/abs/2611.14468) | 该论文证明了异质SIS反应扩散模型的主导流行病动态仅由传播率与恢复率之差及感染者扩散系数共同决定，并据此建立了持续性推断的贝叶斯适定性，表明早期感染观测数据即使在各单独速率不确定时也能有效推断流行病阈值。 |
| [^13] | [A Theory of Policy Optimization in Post-Training: Coverage Collapse and Reference Resets](https://arxiv.org/abs/2611.14467) | 该论文为语言模型后训练中的策略优化建立了理论框架，证明REINFORCE类算法（PPO和GRPO的基础）即使奖励提升也会发生“覆盖坍缩”、使最优响应概率趋近于零并导致优化停滞，而通过重置KL参考等手段可实现由基础模型响应覆盖度所约束的保覆盖近优学习。 |
| [^14] | [A Theory of Cross-Validation Variance](https://arxiv.org/abs/2611.14463) | 该论文证明了交叉验证估计方差的精确有限样本恒等式，将方差表示为每个观测值在训练与评估两种角色中权重的函数，从而为应选择何种数据划分方案提供了理论指导。 |
| [^15] | [Accurate predictions of properties of energetic materials with tabular foundation models](https://arxiv.org/abs/2611.14462) | 表格基础模型TabPFN无需超参数调优，即可在预测含能材料的撞击感度、分解温度和晶体密度方面超越已发表的最佳模型，且速度比XGBoost快一个数量级以上。 |
| [^16] | [UGG: Unified Generative Geometry Model](https://arxiv.org/abs/2611.14440) | 该论文提出统一生成式几何模型UGG，将RGB视频生成与深度、相机位姿等几何预测统一为单一生成模型中的条件补全任务，实现4D场景理解，并能从异构数据集学习以及在推理时对多个场景假设进行搜索。 |
| [^17] | [One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge](https://arxiv.org/abs/2611.14432) | 该团队基于2025年冠军方案π₀.₅构建单一VLA模型，仅用一个检查点完成2026年BEHAVIOR挑战赛全部100个家庭任务并获得第九名，核心改进在于延长训练时间、优化训练参数、加快训练步骤并减少梯度噪声。 |
| [^18] | [Quantifying the Sampling Frequency Invariance of Time-Series Foundation Models for Sensor Data Applications](https://arxiv.org/abs/2611.14425) | 该论文首次将采样频率不变性置于时间序列基础模型评估的核心，通过压力测试4个主流模型并提出FSI、FDI、CDI三个新指标，发现模型行为取决于信息带宽损失而非采样频率数值本身。 |
| [^19] | [DnNP: Denoising Input Uncertainty in Neural Processes](https://arxiv.org/abs/2611.14424) | 提出去噪神经过程（DnNP），通过变分推断显式建模输入不确定性，将去噪与元学习联合训练，从而在含噪数据下显著提升预测质量与不确定性估计。 |
| [^20] | [Exploring Post-Hoc Disentanglement of Audio Foundation Models for Music Similarity](https://arxiv.org/abs/2611.14415) | 本文提出将条件相似度网络（CSN）应用于冻结的音频基础模型预训练嵌入进行事后解耦，证明仅需少量可训练参数即可同时提升属性特定的解耦能力和整体音乐检索性能。 |
| [^21] | [Causal Concept Explanations for Deep Neural Models](https://arxiv.org/abs/2611.14407) | 该论文提出 CCW-Net，首次将干预性中介分析作为训练目标，使概念解释与模型输出之间建立真正的因果关系而非仅是相关性，并提供量化每个概念解释对模型输出因果贡献的评估工具。 |
| [^22] | [Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation](https://arxiv.org/abs/2611.14401) | 该研究提出符号蒸馏方法，将神经网络与符号回归结合，从21年再分析数据中发现了可解释的南亚季风降水热力学方程，其解释能力超过传统浮力诊断方法1.5倍，同时参数量减少约一千倍。 |
| [^23] | [REAL Loss: Optimizing Real Expected Acceptance Length for Speculative Decoding](https://arxiv.org/abs/2611.14399) | 该论文提出 REAL 损失，通过路径权重无偏估计与 Rao-Blackwellization 方差缩减，直接优化投机解码中真实的期望接受长度（EAL），解决了现有替代目标存在的协方差差距及其梯度方向反转问题。 |
| [^24] | [A Margin Perspective on LoRA: Robustness to Catastrophic Forgetting and Adapter Merging (MaLoRA)](https://arxiv.org/abs/2611.14397) | 该论文通过统一的基于间隔的理论视角，解释了LoRA对灾难性遗忘的鲁棒性以及可合并独立训练适配器这两个重要经验特性。 |
| [^25] | [An Energy-Based Perspective on Machine Learning](https://arxiv.org/abs/2611.14383) | 本教程以能量模型（EBM）为统一视角，通过能量函数、推断过程和学习方法三大核心组件，将判别模型、GAN、扩散模型、自回归模型等各类机器学习方法统一在一个共同的对比学习框架之下。 |
| [^26] | [Uncertainty-Aware Hybrid Offline Reinforcement Learning with Online Action Selection Enabled by Sparse Gaussian-Process Critics](https://arxiv.org/abs/2611.14374) | 本文提出用稀疏变分高斯过程评论家同时替代树集成评论家与对抗自编码器惩罚，通过最大化不确定性调整的回报估计 μ-κσ，在混合离线强化学习框架HOFLON中实现不确定性感知的在线动作选择，提升了聚合反应器开车启动等任务的安全性与可靠性。 |
| [^27] | [The Unused Input: The Initial State as a Conditioning Interface for Recurrent TSFMs](https://arxiv.org/abs/2611.14357) | 该论文揭示循环时间序列基础模型中被闲置的初始状态可作为条件化接口：冻结TiRex骨干并仅训练一个小模块将输入窗口的频谱摘要写入初始状态，即可降低GiftEval上的MASE。 |
| [^28] | [Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation](https://arxiv.org/abs/2611.14328) | 该论文证明了高阶近似无法提升随机扩散采样的收敛阶数，并提出免训练的前向值离散化方法（F-DPMSolver），通过利用先前预测构建的前瞻估计，在低神经函数评估次数下实现更快的采样并提升样本质量。 |
| [^29] | [When Should Agents Remember? Falsification-Gated Self-Evolution for LLM Agents](https://arxiv.org/abs/2611.14326) | 提出证伪门控自我进化（FGSE）框架，将经验教训表示为结构化假设，仅在临时状态中经受证伪、迁移与回归探测，只有通过收益与风险双重验证的行为更新才被写入智能体的持久记忆，从而显著提升LLM智能体自我进化的可靠性与安全性。 |
| [^30] | [Auditing the Judge: Internal Signals for LLM Reasoning Assurance](https://arxiv.org/abs/2611.14323) | 该论文提出用注意力头激活训练的监督式正确性探针来审计大语言模型的自我裁判机制，证明在固定审查预算下探针比裁判自身的置信分数能发现更多推理错误。 |
| [^31] | [MammothDP: Private Boosted Decision Trees Ready for Trusted Hardware and Leakage-free Hyperparameters](https://arxiv.org/abs/2611.14317) | 该论文提出 MammothDP 系统，基于梯度提升决策树学习器 S-BDT，将可信执行环境部署、计时侧信道加固与差分隐私过滤器相结合，克服了这一组合使用所产生的意外安全挑战，并以无超参数泄露的方式实现了隐私-效用权衡良好的端到端隐私保护机器学习方案。 |
| [^32] | [PSL-RGHA: Parallel Split Learning with Reference Gradient Half-Space Alignment for Mitigating Gradient Divergence](https://arxiv.org/abs/2611.14312) | 提出PSL-RGHA框架，通过参考梯度半空间对齐机制缓解并行拆分学习在异构数据分布下的训练发散问题，使资源受限设备上的多媒体应用部署更加可行。 |
| [^33] | [Multimodal Abstention That Preserves Acquired Correct Answers](https://arxiv.org/abs/2611.14311) | 提出ReFrain多模态弃答方法，在提升对不可回答问题正确弃答能力的同时，保证任务适配阶段获得的正确答案零损失。 |
| [^34] | [MMDM: Measuring Massive Multitask Decision Making](https://arxiv.org/abs/2611.14309) | 提出了MMDM基准——涵盖16个任务类别、5,448道高质量问题的大规模多任务决策基准，用于在无数据污染的全新上下文中评估系统-1决策模型在单选、多选和软标签决策上的能力。 |
| [^35] | [Compositional random forest](https://arxiv.org/abs/2611.14275) | 该论文提出了一种面向成分数据响应的随机森林框架，以对数似然增益作为分裂准则，并结合定义在不同表示空间上的分布，统一兼容了单纯形、对数比坐标和单位球面方向数据三种成分数据分析视角。 |
| [^36] | [Federated Learning for Cross-Regional Epidemic Forecasting with Asynchronous Outbreaks](https://arxiv.org/abs/2611.14272) | 本文提出并验证了一个基于联邦学习的跨区域COVID-19预测框架，利用LSTM网络在广东、澳门和葡萄牙等疫情异步暴发地区之间进行协作预测，在保持数据主权的同时与本地化及集中式训练方案进行了性能对比。 |
| [^37] | [Physics-Constrained Autoencoder for Interpretable Health Representation Learning in Turbofan Engines](https://arxiv.org/abs/2611.14266) | 提出一种物理约束自编码器，通过在解码器中嵌入可微分物理方程、在损失函数中引入学习偏置，使每个潜在变量对应预定义的物理量，从而实现涡扇发动机部件级可解释的健康表征学习。 |
| [^38] | [Chain of Physics: Supporting Fault Diagnosis with Engineering-Inspired Large Language Model Reasoning](https://arxiv.org/abs/2611.14263) | 该论文提出“物理链”框架，通过从资产参考文档自动构建约束知识图谱来存储失效模式、物理量变化与传感器之间的诊断链，使大语言模型能够结合工程知识进行故障诊断推理。 |
| [^39] | [Recurrent Modular Structure in Spiking Neural Network Reinforcement Learning](https://arxiv.org/abs/2611.14262) | 提出群落循环脉冲神经网络，将循环网络划分为具有固定角色的群落（控制器、动作和潜在状态群落），并结合局部三因子学习规则及新提出的Map-STDP，使脉冲强化学习网络结构更接近生物皮层微回路。 |
| [^40] | [Personalized Affective Perception via Continual Learning in Human-Agent Negotiation](https://arxiv.org/abs/2611.14247) | 该论文提出CLIFER——一种基于持续学习与想象机制的面部表情识别方法，使人形机器人在与人类协商中实时个性化地感知情绪并动态调整协商策略，相比基线显著提升了个体协商效用。 |
| [^41] | [SinkState: On-the-Fly SVD of Recurrent States for Efficient Hybrid LLM Serving](https://arxiv.org/abs/2611.14246) | SinkState发现线性注意力的查询、键、值集中于稳定的低维子空间（“线性注意力汇聚”现象），使循环状态实际呈低秩且秩随状态变化，并通过推理全程的自适应低秩存储与即时局部SVD分解，大幅降低混合大模型服务的内存流量和解码成本。 |
| [^42] | [On Federated Averaging with Partial Participation under a General Variance Condition](https://arxiv.org/abs/2611.14239) | 本文在允许方差随局部目标差距增长的一般方差条件（ABC条件）下，为部分客户端参与的联邦平均算法建立了包含异质性度量的有限时间平稳性收敛界，实现了随采样客户端数量的线性加速，并在全参与情形下与已有结果一致。 |
| [^43] | [Verifiable Agentic Traffic Class: A Policed Quality-of-Service Mechanism for Agentic AI Traffic, Evaluated from Simulation to Real-Kernel Adversarial Testing](https://arxiv.org/abs/2611.14211) | 本文提出可验证智能体流量类别（VATC），通过DSCP流量声明、路由器侧验证分类器和速率受限优先队列，为智能体AI流量提供可验证的服务质量保障，在突发场景下相比传统DiffServ将智能体排队延迟降低98%，同时仅使人类流量延迟增加12%。 |
| [^44] | [Tree-Ising: Repairing Language Models with Bit-Level Ising Edits](https://arxiv.org/abs/2611.14210) | Tree-Ising将语言模型的无梯度权重修复转化为BF16权重上的稀疏离散优化问题，通过符号翻转和指数移位等比特级编辑，将目标函数精确分解为逐输出行的QUBO/Ising问题，并在GPU上用模拟退火或量子方法求解，以修复模型推理循环等缺陷。 |
| [^45] | [SpecValley: Aggregation as Spectral Valley in Dense Heterophilic Graphs](https://arxiv.org/abs/2611.14176) | 该论文提出SpecValley方法，将图神经网络中固定为度归一化的聚合步骤改造为可学习的谱谷算子，为每个图自适应地学习衰减带的位置、锐度和两侧增益，从而解决稠密异质图中聚合会衰减携带标签信号的问题。 |
| [^46] | [Nonadaptive One-Sided Testing of Binary Rank](https://arxiv.org/abs/2611.14163) | 本文研究了二进制秩的单边错误非自适应测试问题，给出了与矩阵维度无关的 $\tilde O(d^4/\epsilon^2)$ 查询复杂度上界和 $\tilde\Omega(d^{3/2}/\epsilon)$ 的一般下界。 |
| [^47] | [Unipolar Neuron Networks Trained by Recirculation for Reward-Based Continual Learning](https://arxiv.org/abs/2611.14153) | 该论文受章鱼等无脊椎动物单极神经元结构启发，提出了一种树状神经元网络，通过仅需奖励信号的“再循环”学习规则实现持续学习，其中每个突触仅依赖局部信息更新权重，且更新方向近似遵循误差梯度。 |
| [^48] | [Deviation-Aware Pretraining for Cardiovascular Disease Detection from Electrocardiograms](https://arxiv.org/abs/2611.14141) | 该论文提出了一种基于近百万份无标签12导联心电图的自监督预训练框架，通过偏差定位嵌入（DLE）和关联重构嵌入（ARE）两个模块捕捉波形偏差及跨导联异常模式，在心血管疾病检测任务上相比现有方法显著提升了AUROC和F1性能。 |
| [^49] | [HaiRuo: Forecasting Ocean Eddies as Fields and Objects with a Satellite-Altimetry Foundation Model](https://arxiv.org/abs/2611.14136) | HaiRuo将24年卫星测高数据预训练的10.5亿参数基础模型HaiRuo-1B与拉格朗日对象解码器XuanJi相结合，仅需30天卫星海面高度图的一次前向传播，即可同时预报15天海洋物理场、涡旋掩膜及每个涡旋的轨迹。 |
| [^50] | [Right for the Wrong Reasons? Evaluating Plausibility and Behavioral Change in Explanation-Guided Learning](https://arxiv.org/abs/2611.14134) | 研究表明解释引导学习（EGL）的效果高度依赖具体目标，其中基于理由的数据增强在捷径证据可与任务相关前景信息分离时带来最显著的鲁棒性提升（准确率最高提升4.8个百分点），且模型解释与人类理由的一致性提升并不必然意味着决策策略的真正改变。 |
| [^51] | [Continuous 3D Spatial Transcriptomics Reconstruction via Histology-Informed Implicit Neural Representations](https://arxiv.org/abs/2611.14129) | 提出了一种组织学信息引导的多任务隐式神经表示方法HIM-INR，通过共享骨干网络联合重建基因表达谱和组织学表示，并利用密集采样的组织学图像作为辅助监督，实现了从稀疏空间转录组学观测到任意三维坐标的连续基因表达重建。 |
| [^52] | [Hierarchical Graph Representation Learning with Pooling-Induced Substructures](https://arxiv.org/abs/2611.14126) | 该论文提出池化子结构感知GNN，通过显式保留并处理结构池化所诱导的子结构来增强层次图表示学习，在WL测试意义上提升了基于池化的GNN表达能力，并引入了具有可控层次结构的新合成数据集用于系统评估。 |
| [^53] | [CroCoMAE: Cross-modal self-supervised collaborative representation learning for SAR-optical remote sensing data](https://arxiv.org/abs/2611.14124) | 提出CroCoMAE框架，在基于ViT的架构中统一对比学习与掩码图像建模，从图像级和补丁级两个互补层面挖掘SAR-光学遥感数据的细粒度跨模态互补结构，实现自监督协同表示学习。 |
| [^54] | [Opportunity-Cost Option-Critic: Adaptive Switching Costs for Robust Hierarchical Reinforcement Learning](https://arxiv.org/abs/2611.14099) | 该论文提出机会成本Option-Critic（OC²）框架，利用当前选项与最佳备选选项之间的价值差距来定义自适应的状态依赖切换成本，从而实现更鲁棒的分层强化学习。 |
| [^55] | [CUVE: Making Supervised Classification more Accessible through Visualization](https://arxiv.org/abs/2611.14098) | CUVE是一种可视化界面，通过生成人类和机器学习算法均可理解的可视化作为中间表示，帮助用户更便捷地构建、理解和改进监督分类器。 |
| [^56] | [Stylization as Geodesic Navigation on Diffusion Manifolds](https://arxiv.org/abs/2611.14097) | 提出免训练框架Stylus，将人脸风格化建模为在预训练扩散模型弯曲潜在流形上沿测地线导航的过程，通过内容-风格潜在插值与自注意力查询优化双控制，实现既具艺术风格又保持人物身份的风格迁移。 |
| [^57] | [BODYinTRANSIT-Lab: A Multimodal Dataset for Investigating Sensory-Driven Transformations in Body Perception and Affective Experiences](https://arxiv.org/abs/2611.14093) | 该论文推出了BODYinTRANSIT-Lab多模态数据集，通过脚步错觉闭环范式采集101名分层招募参与者的全身运动学、多种生理信号及主观情感与身体知觉数据，用于研究感觉驱动的身体知觉转变并支持从传感器数据推断主观状态的计算模型开发。 |
| [^58] | [Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models](https://arxiv.org/abs/2611.14083) | Sable通过整数化确定性计算、小域sumcheck约束系统和单GPU逐层流式证明，实现了对大语言模型位精确推理的可验证性，在单块GPU上92.5秒内即可完成Qwen2.5-32B生成4096个token全部64层推理的证明。 |
| [^59] | [What a CAGE 4 Score Measures: An Audit of a Learned Cyber-Defense Benchmark, Its Reference Defenders, and the Tables Built on Them](https://arxiv.org/abs/2611.14068) | 该论文审计了 CAGE Challenge 4 网络防御基准，发现其模拟器会静默丢弃发往忙碌智能体的动作并照常计分，而两个参考防御者均依赖文档未披露的“忙碌”标志检查，修正这些问题后已发布分数每回合变动高达约 146 分，表明基于该基准的排行表所测量的内容远少于其表面所示。 |
| [^60] | [Generalizing Causal Decision-Making: Markov Equivalence, Transportability and Counterfactuals](https://arxiv.org/abs/2611.14066) | 本文将结构因果老虎机推广至三种更一般的场景——通过马尔可夫等价性刻画部分已知因果结构下可能最优的最小干预集、利用部分可迁移性从源环境推导目标奖励边界并获得遗憾保证、以及构建支持反事实行动的反事实结构因果老虎机框架。 |
| [^61] | [Where Do Vision-Language Models Get Left and Right? Position-Code Counterfactuals in Qwen2.5-VL, Qwen3-VL and InternVL3](https://arxiv.org/abs/2611.14060) | 该研究通过位置编码反事实实验发现，视觉语言模型的左右方位判断主要由ViT位置编码决定——仅镜像ViT宽度位置即可在84-93%的问题上复现翻转图像的效果，而LLM侧的位置信息影响甚微。 |
| [^62] | [HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios](https://arxiv.org/abs/2611.14059) | 本文构建了HandsOnAid——一个可在无信号、无救援环境中通过QLoRA微调的4位Gemma 2B模型实现的离线手机急救助手，并通过四个前沿大模型盲评验证了微调显著提升了急救建议的质量与安全性。 |
| [^63] | [Structured Confidence Perturbation via Fisher Geometry for Adversarial Training](https://arxiv.org/abs/2611.14046) | 该论文针对对抗训练中的“伪攻击”问题，提出融合交叉熵与KL散度的混合目标，并基于费舍尔信息矩阵几何在置信度流形上生成最优扰动，理论证明最优方向为费舍尔信息矩阵主特征向量，同时给出避免高维矩阵计算的高效算法。 |
| [^64] | [ScaleBrain: A Scale-Consistent Framework for EEG Foundation Models](https://arxiv.org/abs/2611.14035) | 提出了ScaleBrain——一个尺度一致的EEG基础模型预训练框架，通过在模型输入与优化目标两个层面显式协调多尺度表示，解决了现有方法中特征与原始信号尺度特性对应关系隐式、部分时间尺度监督不足的问题。 |
| [^65] | [Margin-Guided Token Credit for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2611.14015) | 提出MCPO方法，利用策略下一Token分布的前两名概率差识别模型犹豫的Token位置，将奖励信用有针对性地集中于这些位置，且无需额外前向传播或批次统计，在六个竞赛数学基准上平均超越GRPO基线6.55分。 |
| [^66] | [Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation](https://arxiv.org/abs/2611.14009) | 该论文提出 CAMG-II 长时程智能体强化学习环境套件，通过提供 Bash 访问和回合持久工作区，让记忆控制建立在预训练已有的文件操作能力之上，从而解决在线策略蒸馏中因记忆接口超出预训练范围而导致教师指导无效、学生记忆动作采样不足的问题。 |
| [^67] | [Normalizing Flows for Classical Simulation of Quantum Circuits in Phase Space](https://arxiv.org/abs/2611.14006) | 该论文提出基于多体系统相空间表述的归一化流框架，将酉门参数化为概率分布之间的流变换，实现了在大系统规模下精度超越现有最先进神经网络方法的量子电路经典模拟。 |
| [^68] | [LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation](https://arxiv.org/abs/2611.14004) | 提出解耦多模态表示学习框架LUMIR，利用大语言模型将异构多模态数据统一到共享文本空间，并通过对比学习结合交互感知融合与序列信号约束进行训练，使学习到的表示与下游推荐模型解耦，提升下一兴趣点推荐中多模态表示的通用性和可重用性。 |
| [^69] | [Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models](https://arxiv.org/abs/2611.13997) | 该论文提出了首个系统性评估大型视觉语言模型“推理时个性化隐私保护”（ITP³）能力的基准SPY-Bench，包含32,700个涵盖67个类别和24个真实场景的样本以及基于用户规范与情境感知的新颖指标，旨在使模型能够遵循用户通过语言动态指定的个性化隐私边界。 |
| [^70] | [BioXArena-Pro: a research-grade benchmark for autonomous biomedical machine learning agents](https://arxiv.org/abs/2611.13994) | 本文提出BioXArena-Pro基准，包含8个生物医学领域64个经AI辅助审计与人工审查的研究级建模任务，评估显示最佳自主编码智能体仅得55.1/100分，智能体框架与模型同等重要，且智能体通过适应新数据可在重建任务上超越已发表方法8.2分。 |
| [^71] | [Temporal Behavior Trees for Reinforcement Learning: Specifications as Rewards, Curricula, and Diagnostics](https://arxiv.org/abs/2611.13993) | 该论文提出TBTCL框架，以时间行为树（TBT）作为统一接口，使单一规范同时承担任务定义、密集鲁棒性奖励生成、课程设计与策略诊断四种功能，其表达力优于LTL且天然具有层次结构。 |
| [^72] | [Pulse or Provenance? What rPPG-Based Deepfake Detectors Learn](https://arxiv.org/abs/2611.13992) | 研究通过系统实验揭示，基于rPPG的深度伪造检测器学到的并非真实的生理心脏信号，而是与视频编码等来源相关的非生理伪影，这挑战了rPPG作为深度伪造检测生理证据的基本前提。 |
| [^73] | [Universal Approximation Theorem for Metaplectic Neural Operators](https://arxiv.org/abs/2611.13991) | 该论文证明了亚辛神经算子的万能逼近定理：在非空有界Lipschitz区域上，仅需一个带单条非局部亚辛乘子分支的隐藏算子层，即可对从C^s到C^r以及H^s到H^r的连续算子实现紧一致逼近。 |
| [^74] | [Gradient Inversion through Linear and Affine Subspace Constraints](https://arxiv.org/abs/2611.13976) | 本文发现全连接层的聚合梯度在正确缩放下是其输入的仿射组合而非仅线性组合，并利用线性或仿射子空间约束改进联邦学习中的梯度反演攻击，从而提升数据重建精度。 |
| [^75] | [On the Spectral Estimation of VARMA Processes](https://arxiv.org/abs/2611.13973) | 本文提出了一种基于传输熵概念的新矩问题，能够在不固定移动平均分量或不将其限制为标量的情况下，学习VARMA过程的功率谱密度。 |
| [^76] | [Data Quality Challenges in Multimodal Tourism Recommender Systems](https://arxiv.org/abs/2611.13972) | 该论文聚焦旅游推荐场景，揭示了使用Yelp和Instagram等真实多模态数据时面临的虚假评论噪声、LLM幻觉内容、模态矛盾和人口偏差等数据质量挑战，并提出质量感知融合、受控LLM内容生成和用户群体差异化等未来改进方向。 |
| [^77] | [Counterfactual Analysis of Extent Leakage from Annotation-Derived Prompts in Vision-Language Medical Image Segmentation](https://arxiv.org/abs/2611.13970) | 该论文通过反事实分析揭示了医学图像分割基准文本提示中的“范围泄漏”问题：标注派生提示包含从真值掩膜计算出的病灶大小、数量和位置信息，模型的性能提升主要依赖这种泄漏的范围信息，而非理解描述中的其他语义内容。 |
| [^78] | [RB-LDC: Redundancy-Balanced Latent Coding for Noise-Robust Diffusion Latents](https://arxiv.org/abs/2611.13955) | 提出RB-LDC方法，通过在编码器与扩散过程之间加入可逆线性层，按解码器敏感性对各潜变量方向的对数信噪比进行冗余平衡式的最优再分配，从而提升扩散模型潜变量的噪声鲁棒性。 |
| [^79] | [Geometry-Dependent Approximation and Stability Properties of Shallow ReLU$^k$ Networks on Bounded Euclidean Domains](https://arxiv.org/abs/2611.13952) | 本文揭示了有界欧氏域上浅层ReLU$^k$网络的逼近精度与稳定性由隐藏参数几何（填充距离与对径分离距离）定量决定，并给出了逼近上界、饱和型下界以及矩阵谱和最小二乘恢复等系统的稳定性分析。 |
| [^80] | [Chem World: A Large-Scale Benchmark and Physics-Aware Framework for Trustworthy Molecular Property Prediction](https://arxiv.org/abs/2611.13945) | 本文提出了大规模基准Chem World和物理感知组合建模框架PACM，通过整合17个数据集、80余万实例和领域物理约束，为可信的分子性质预测提供了统一的评估平台，并验证了分子语言模型表示的强泛化与可迁移性。 |
| [^81] | [PARDIG: Global Precipitation Estimates Learned from Sparse Satellite Observations](https://arxiv.org/abs/2611.13941) | PARDIG是一个基于球形图-网格Transformer的人工智能框架，通过学习密集多模态观测数据来预测稀疏的高精度星载雷达降水测量，以0.1°空间分辨率和15分钟时间分辨率提供2016年至今的近实时全球降水估计。 |
| [^82] | [Dual-Core Tensor Wheel Decomposition with Structure-Cached Acceleration for Tensor Recovery](https://arxiv.org/abs/2611.13938) | 该论文提出双核张量轮分解（DCTW）及结构缓存加速策略，通过将高阶中心核分解为两个由共享潜变量键连接的低阶核，并缓存张量网络收缩中的中间张量以避免显式构造大规模矩阵和冗余计算，从而大幅提升了张量恢复的计算与内存效率。 |
| [^83] | [Dual-Space Preconditioning for Variational Inequalities and Root-Finding Problems](https://arxiv.org/abs/2611.13916) | 本文提出了一种超越经典余强制性和利普希茨连续性假设的对偶空间预条件框架，通过引入新颖的广义条件为变分不等式与求根问题的确定性和随机方法建立收敛性理论，并将算法解释为变步长方法且给出预条件流的变分与最优控制解释。 |
| [^84] | [Learning from First Discrepancies with Joint Reset and Pilot Guarantees](https://arxiv.org/abs/2611.13913) | 该论文研究了每次运行只揭示到首个错误决策为止信息的“首次分歧”学习模型，并证明当假设类（VC 维至多 $d$）的最优分歧概率至多为 $f$、最优风险至多为 $r$ 时，学习器仅需 $\widetilde O((d+1)[D(f+\varepsilon)+r]/\varepsilon^2)$ 次运行和 $\widetilde O(D(1+f/\varepsilon))$ 个先导假设，即可返回假设类成员及宽度至多 $\varepsilon$ 的风险区间，同时提供联合的重置与先导保证。 |
| [^85] | [Target Selection with a Disappearing-Stimulus Mixed-Reality P300 Interface: Feasibility and Object-Distinguishable Versus Uniform Cues](https://arxiv.org/abs/2611.13906) | 该研究证明，即使目标线索在闪烁间消失不再可见，混合现实P300接口仍能实现远高于随机水平的目标选择，并探索用颜色-形状可区分的线索替代均一球体以改善解码性能。 |
| [^86] | [Conformal Risk Control for QoT-Aware Lightpath Feasibility Under Technology-Driven Domain Shift](https://arxiv.org/abs/2611.13893) | 本文提出一种事后保形风险控制框架，在保持预训练AI/ML回归模型不变的情况下，为技术升级引起域偏移时光网络QoT感知光路可行性评估的不确定性区间提供有限样本统计保证，并通过策略规则将预测区间映射为运行决策。 |
| [^87] | [Are Causal Neuron Circuits Unique and Transferable?](https://arxiv.org/abs/2611.13885) | 该论文通过穷举测量 12 个 MLP 神经元全部 4,096 种子集的干预效果，发现虽然消融选定神经元组会显著改变模型回答，但这些神经元的激活在不同问题间几乎恒定、替换后也几乎无影响，说明真正必要的只是其中一部分，从而质疑了因果神经元回路的唯一性与可迁移性。 |
| [^88] | [Outliers and the Hard-Margin Linear SVM in High-Dimension, Low-Sample-Size Settings](https://arxiv.org/abs/2611.13869) | 该论文建立了高维小样本下离群点对硬间隔线性SVM影响的精确渐近理论：单侧离群点的全部效应可归结为一个有效距离并由此导出内点支持丧失、间隔收缩、截距位移和误分类的显式阈值，而两类对称污染仅收缩间隔、不改变极限判别函数。 |
| [^89] | [Mechanistic Interpretability of Deep Neural Networks: Methods, Frameworks, and Perspectives](https://arxiv.org/abs/2611.13868) | 本文系统综述了深度神经网络机制可解释性的方法与框架，其核心在于通过识别对特定行为具有因果影响的内部组件来揭示网络的内部工作机制。 |
| [^90] | [Where to Approximate in Neurosymbolic Inference?](https://arxiv.org/abs/2611.13862) | 该论文将现有近似WMC方法统一为三步自底向上编译管线框架，利用张量列车分解与基于SVD/插值的再压缩技术研究近似预算的最优分配位置，在数独、算术推理和场景理解等任务上实现了比先前方法准确数量级提升的WMC估计。 |
| [^91] | [Beyond Passive Imitation: On-Policy Mid-Training Distillation for Language Models](https://arxiv.org/abs/2611.13855) | 该论文提出在线策略中期训练蒸馏（OPMD），将中期训练从被动的离线文本模仿转变为主动的、查询驱动的知识获取，通过从学生策略自采样轨迹并结合教师模型的词元级反向KL奖励，解决了标准下一词预测目标在不同基础模型上效果不一甚至产生负面影响的问题。 |
| [^92] | [Optimal Last-Iterate Bounds for Nonexpansive Stochastic Approximation](https://arxiv.org/abs/2611.13846) | 本文为非扩张随机逼近（随机KM迭代）建立了末次迭代均方残差界，在噪声条件二阶矩的三种情形下均给出相差对数因子内最优的收敛速率，并在噪声与残差平方呈线性关系时达到与无噪声KM迭代相同的界。 |
| [^93] | [MusicFlux: Flexible Conditioning for Text-Guided Music Generation and Editing](https://arxiv.org/abs/2611.13834) | MusicFlux是一个基于预训练流匹配文本到音频模型的文本引导音乐编辑器，通过两种条件化方法在编辑质量与计算效率之间实现权衡，无需处理过长的音频序列即可统一音乐的生成与编辑。 |
| [^94] | [JAX-WAVE: A Differentiable Physics-Supervised Framework for Inverse Wave Design in Heterogeneous Media](https://arxiv.org/abs/2611.13826) | 提出可微分物理监督框架 JAX-WAVE，通过共享神经网络与可微分时域有限差分求解器实现摊销式逆波场设计，无需优化控制标签即可将目标波场直接映射为满足符号反转等物理约束的最优有界初始压力。 |
| [^95] | [CASCADE: Constraint-Aware SCAffolded DEcomposition for Logical Anomaly Detection](https://arxiv.org/abs/2611.13820) | 论文提出CASCADE方法，通过约束感知的支架式分解为逻辑异常检测中的每类约束定制视觉证据提取，弥合了LVLM清单式验证中“约束特定感知鸿沟”所导致的实例混淆与尺度漂移问题。 |
| [^96] | [Adaptive Policy-Guided Error Mitigation (APGEM): A Unified Learning-Based Framework for Noise Resilience in Quantum Machine Learning](https://arxiv.org/abs/2611.13815) | APGEM提出了一种基于上下文老虎机的自适应框架，利用变分量子算法自身产生的反馈信号在线动态选择错误缓解技术，从而解决了量子机器学习训练过程中因电路与噪声漂移导致固定缓解策略失效的问题。 |
| [^97] | [AnchorMTS: Multi-View Signal Representation and Forecast-Anchored Fusion for Long-Term Time-Series Forecasting](https://arxiv.org/abs/2611.13794) | AnchorMTS通过从时间序列中提取视觉和文本等多视图表示来修正数值预测，在多个基准数据集上取得了最先进的长期预测性能。 |
| [^98] | [Omni-JEV: Text and Dual-Stream Speech Decisions](https://arxiv.org/abs/2611.13777) | 提出 Omni-JEV 模型，在全双工对话中将文本或角色索引的双流（用户与助手参考）语音映射为五种交互动作的概率，并通过双语弱监督微调与 RLCD 强化学习实现概率敏感的决策。 |
| [^99] | [Beyond Interaction Scores: Selecting Feature Interactions via Independent Replication](https://arxiv.org/abs/2611.13765) | LENS 提出了一个模型无关的特征交互选择框架，通过在不相交数据上训练两个模型进行独立复现来验证交互模式的真实性，无论特征数量多少仅需两次模型拟合和一次多重检验校正。 |
| [^100] | [Mixed Signals: Weight-Space Programming for EEG and ECG](https://arxiv.org/abs/2611.13758) | 本文首次将权重空间编程应用于脑电图与心电图等生物信号模型，证明语言模型不仅能诊断并修复植入的模型故障，还能仅凭推理从零写出网络全部权重，并以少6倍的训练数据达到与梯度下降训练相当的效果。 |
| [^101] | [When Covariance Is Not Enough: Kernel-Based Regularization for Single-Cell Counterfactuals](https://arxiv.org/abs/2611.13748) | 本文证明单细胞反事实生成模型中常用的线性交叉协方差惩罚在存在非线性交互时无法充分解耦已知与未知因素，并提出加入归一化Hilbert-Schmidt独立性准则（HSIC）这一基于核的非线性正则化方法来改进分离效果。 |
| [^102] | [Plug-and-Play ADMM for Inverse Problems with Flow Matching Denoiser](https://arxiv.org/abs/2611.13742) | 提出了一种基于ADMM的即插即用框架，通过辅助变量、对偶更新和轨迹校正机制，将流匹配先验更一致地融入逆问题求解，以更低计算成本实现高质量重建。 |
| [^103] | [Deterministic, Reproducible Inference in Exact Integer Arithmetic, and Verifiable Distributed Inference as Its Consequence](https://arxiv.org/abs/2611.13730) | 该论文提出开源运行时 tritify，将真实模型推理的整个运行时路径置于精确整数运算中，实现了跨硬件、跨平台的比特级可复现推理，并以此为基础支持可验证的分布式推理。 |
| [^104] | [Weighted-AoI-Driven UAV Trajectory and Reporting for Dynamic CKM Tracking](https://arxiv.org/abs/2611.13726) | 本文提出加权信息年龄驱动的无人机轨迹规划与上报方法，通过结合精确的时间陈旧性与U-Net预测的空间不确定性来调度测量，实现动态信道知识地图的连续在线追踪，并证明了top-K选择策略可最大化单时隙误差降低上界。 |
| [^105] | [TRUER: Test-Time Reward Correction Using an Error-Aware Rectifier for Reinforcement Learning](https://arxiv.org/abs/2611.13714) | TRUER框架利用有限的标注数据训练一个误差感知的奖励校正器，专门针对错误多数和正确少数的情形来区分正确性与多数共识，从而在测试时强化学习中校正奖励、提升奖励准确性和模型性能。 |
| [^106] | [FedVP: Federated Learning across Heterogeneous Models by Pooling Locally Trained Verifiers](https://arxiv.org/abs/2611.13708) | 该论文提出FedVP新范式，让使用不同（冻结的）大语言模型骨干的异构客户端通过共享、汇聚各自本地训练的验证器来协作选择答案，从而在不聚合或修改模型参数的情况下实现联邦学习收益。 |
| [^107] | [Decision-Focused Learning via Neural Optimization Layers for Battery-Assisted Grid-Import Smoothing in AI Data Centers](https://arxiv.org/abs/2611.13694) | 本文提出电池调度决策聚焦学习方法（BS-DFL），通过约束感知神经优化层将电池储能调度与需求预测端到端联合训练，无需最优解标签，在AI数据中心仿真中将电网购电分钟级波动降低71.70%、调度成本降低18.01%。 |
| [^108] | [Bootstrapped On-Policy Distillation](https://arxiv.org/abs/2611.13680) | 该论文发现在线策略自蒸馏（OPSD）在持续训练中会因冻结的特权教师与不断进化的无参考学生之间日益扩大的分布失配而导致性能退化，并提出自举式在线策略蒸馏方法来解决这一问题。 |
| [^109] | [SepsisFlow: an event-stream foundation model for continuous risk warning and trajectory modelling in ICU sepsis](https://arxiv.org/abs/2611.13661) | SepsisFlow是一个基于257万份电子病历预训练Mamba模型的事件流基础模型，可在ICU脓毒症发病前实现6/12/24小时风险预警、发病时刻画发病情境、发病后建模72小时早期病程轨迹，其预警性能优于NEWS2、qSOFA、SIRS等传统评分及16种重训算法。 |
| [^110] | [CP-MLPs: A Tensor-Rank Theory of Tied and Untied MLP Blocks](https://arxiv.org/abs/2611.13647) | 该论文提出CP-MLPs张量秩框架，将门控单元分解为检测、读取、写入三种角色，并严格证明：解绑门控块（如ReGLU/SwiGLU）在拟合不定二次交互和角色非对称目标时比绑定门控块（如ReLU²）具有严格的宽度效率优势，而绑定结构更匹配对称纯幂次目标，即解绑的收益取决于目标函数的结构。 |
| [^111] | [MouldIt: Data Efficiency of Multi-Task Temporal Representation Learning in Injection Moulding](https://arxiv.org/abs/2611.13646) | 本文提出MouldIt——一个基于时序卷积与注意力机制的变分自编码器，用于注塑成型过程的多任务时序表征学习，并系统评估了其在不同数据规模、单模态与多模态设置下相比经典机器学习方法的数据效率与多模态收益。 |
| [^112] | [Machine-learning decoders for fusion-based quantum computing in real time](https://arxiv.org/abs/2611.13630) | 提出了PRISM——一种基于Transformer的机器学习译码器，可在高擦除率下为融合型光子量子计算提供实时高精度译码，其逻辑错误率与整数规划译码器相当而速度显著更快。 |
| [^113] | [Curriculum proof repair with learned counterexamples](https://arxiv.org/abs/2611.13627) | 提出TopoVerus框架，将反例引导的证明诊断内化为语言模型的自身能力，通过数据驱动的课程式训练使模型能够一次性生成有用的反例，从而高效修复Verus形式化证明。 |
| [^114] | [Multifidelity Operator Inference for Non-intrusive Model Reduction under Data Scarcity](https://arxiv.org/abs/2611.13624) | 该论文提出了一种多保真度算子推断方法，以在计算预算有限、高保真训练数据稀缺的情况下实现非侵入式的模型降阶。 |
| [^115] | [Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control](https://arxiv.org/abs/2611.13615) | Synapse-Control用一个轻量级智能体运行时控制平面（规则筛选、1.7B微调分类器评分、微合同接口检查和提示价值表冷却机制）取代大模型失败诊断，以更低的推理成本实现智能体执行经验的高效复用与精准投放。 |
| [^116] | [Post-OCR Correction of Mixed-Script Uzbek Text: Byte-Level ByT5 with an Empirical Homoglyph Noise Matrix](https://arxiv.org/abs/2611.13609) | 针对乌兹别克语中西里尔与拉丁字母混排导致的同形字符OCR错误，提出可在本地硬件上运行的字节级ByT5纠错系统，结合由人工标注错误事件估计的噪声矩阵与保守解码规则，将字符错误率从0.0578降至0.0369。 |
| [^117] | [One Byte, One Token: Byteifying a 27B Hybrid-Attention LLM with Self-Generated Data](https://arxiv.org/abs/2611.13607) | 研究者通过将词表缩减至384个条目（含256个字节符号），并利用原始模型自生成的4.77GB在线策略监督数据进行两阶段课程训练，成功将270亿参数的Qwen3.8混合注意力模型直接转换为纯字节级模型，无需补丁结构或局部编码器即可保留其大部分原有能力。 |
| [^118] | [Unified Optimization of Traffic Signal and Lane Channelization for an Isolated Intersection Using Reinforcement Learning](https://arxiv.org/abs/2611.13599) | 本文提出了一种基于强化学习的实时联合优化方法，可同时求解单点交叉口的交通信号控制与车道渠化设计，克服了传统混合整数线性规划（MILP）需随交通流模式变化反复求解且大规模问题时计算耗时的缺陷。 |
| [^119] | [Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics](https://arxiv.org/abs/2611.13596) | 该论文提出状态条件不变性（SCI）这一可靠性原则，并构建了基于模拟器的诊断基准GridWM-Judge，用于系统评估视觉语言模型评判者在语义不变的呈现变化下是否保持稳定、并对改变任务结果的因果干预作出正确响应。 |
| [^120] | [Market-Conditioned JEPA: Learning from Unpredictable Market Shocks](https://arxiv.org/abs/2611.13572) | 提出市场条件化JEPA（MC-JEPA），通过以留一法构建的市场因子作为预测条件，使资产的市场暴露成为最优预测的必要信息，并在11年中国A股分钟级数据上全面超越单一资产的JEPA方法。 |
| [^121] | [Metamorphic testing of learned AC optimal power flow surrogates reveals representation dependence](https://arxiv.org/abs/2611.13566) | 该论文提出用十三项蜕变关系系统审计学习型交流最优潮流代理模型，发现所有被测配置均存在对输入序列化表示的依赖性，并借此定位出索引错误与不稳定预处理等缺陷，同时证明数据增强可以缓解这种表示依赖问题。 |
| [^122] | [Investigating Composite Modeling in Multi-Objective Bayesian Optimization for Scientific Design](https://arxiv.org/abs/2611.13565) | 本文系统研究了利用可观测中间响应的复合建模在多目标贝叶斯优化中对科学设计的收益，并探究了影响其效果差异的因素。 |
| [^123] | [Discovering Structurally Plausible and Interpretable Cognitive Models with Large Language Models](https://arxiv.org/abs/2611.13561) | 本文提出DisCo框架，利用大语言模型结合进化搜索，通过验证器过滤结构无效模型、清理器去除冗余参数，自动发现结构合理、参数可辨识且可解释的认知模型，其性能可与特定任务的基线模型相媲美。 |
| [^124] | [How Post-Training Influences Verbalized Algorithmic Computation in Language Models](https://arxiv.org/abs/2611.13552) | 后训练主要改变的是言语化推理的有用性而非内部信息的编码方式——SFT 和 RLVR 后的模型内部均可读取进位信息，但 RLVR 产生的思维链在各检查点上显著提升了推理准确率。 |
| [^125] | [Tabular Foundation Models in Financial Predictions](https://arxiv.org/abs/2611.13550) | 表格基础模型（TFMs）在预测公司下一年盈利变化方面优于微调大语言模型和基于树的模型，同时更快且计算成本更低，是会计研究中具有前景的预测工具。 |
| [^126] | [Massive Activations Confound Sparse Autoencoders](https://arxiv.org/abs/2611.13548) | 该论文发现模型中的巨大激活使高达99.9%的激活方差集中于第一主成分（PC1），导致稀疏自编码器的评估指标失真、重要特征在不同运行间无法复现，并提出在去除PC1的激活上训练SAE、再将PC1加回重建结果的修正方法，将前50个特征的共享率从4–39.5%大幅提升至73–99.5%。 |
| [^127] | [Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic Rounding](https://arxiv.org/abs/2611.13545) | 该论文发现RL后训练中BF16因“权重粘滞”问题收敛慢于FP16，并提出在将FP32影子权重量化为BF16时采用随机舍入，从而在不增加与参考模型KL散度的前提下消除收敛速度差距。 |
| [^128] | [Sanity Checks for Attention Surgery: Null Models and Matched Interventions for Representation-Motivated Design](https://arxiv.org/abs/2611.13543) | 该论文提出三个可复用的合理性检查（噪声底线、几何零模型、匹配干预）来严格检验基于表征统计的架构修改，并发现自注意力移除中的头-值对齐很大程度上可由早期位置的因果零模型解释，而非模型真正依赖的特定表征方向。 |
| [^129] | [Exploiting Feature-invariant Scaling Laws for Efficient A/B Experimentation in Recommendation Systems](https://arxiv.org/abs/2611.13539) | 该论文发现现代推荐模型具有特征不变的缩放定律特性，并提出一种结合统计推断的方法，能仅通过小规模A/B实验预测全规模实验效果（含置信区间），从而大幅降低特征变更评估的计算成本。 |
| [^130] | [Adaptive catalyst discovery emerges from composable scientific intelligence](https://arxiv.org/abs/2611.13536) | CatAI是一个物理驱动的模块化AI框架，通过学习180万个结构-性能关系并组织成30个可复用的物理化学编码器，在催化剂发现任务中预测误差降低7.8%，并在实验中发现活性达商业基准9倍的新型催化剂。 |
| [^131] | [Beyond the Log-Linear Frontier: Pre-Asymptotic Analysis of Finite-Budget Neural Scaling Laws](https://arxiv.org/abs/2611.13534) | 本文通过 Mellin-Barnes 谱分析与 KKT 对偶理论刻画了有限预算下神经缩放定律偏离对数线性前沿的机制，提出了基于 Aitken 差分的良态闭式缩放定律估计方法，以及基于注水原理的多域数据最优采样策略。 |
| [^132] | [The RSI Tax: When Recursive Self-Improvement Is Not Worth the Cost](https://arxiv.org/abs/2611.13524) | 本文提出“RSI税”概念以量化递归自我改进的token成本，发现有界笔记基线在准确率相当或更优的情况下可节省65.6-71.0%的总token开销，且验证成本占RSI税的74.8-93.6%，表明递归自我改进往往得不偿失。 |
| [^133] | [Parallel Broyden Methods for Efficiently Evaluating Nonlinear State Space Models](https://arxiv.org/abs/2611.13523) | 该论文提出了一种基于Broyden方法的新型并行拟牛顿方法，通过对雅可比矩阵块进行对角加低秩近似来捕获状态维度间的耦合项，从而在不牺牲收敛速度和并行扫描兼容性的前提下，高效评估非线性状态空间模型。 |
| [^134] | [Seeing the Small: Deformable Duality Mamba for Retinal Lesion Segmentation](https://arxiv.org/abs/2611.13514) | 提出D²Mamba，通过将非因果状态空间对偶性与可变形Mamba扫描相结合来建模病灶的空间组织，在DDR数据集上实现33.87%的mIoU，其中微动脉瘤分割Dice提升10.06个百分点。 |
| [^135] | [Evidential Deep Learning with conditional entropy for glaucoma detection](https://arxiv.org/abs/2611.13513) | 该论文提出了一种由新型条件熵函数正则化的证据深度学习方法用于青光眼检测，其不确定性估计的校准误差显著优于深度集成、MC Dropout等传统方法，并通过Grad-CAM焦点指数分析提供了更好的可解释性。 |
| [^136] | [Distribution Shift Detection in Remote Sensing Imagery using Machine Learning Models](https://arxiv.org/abs/2611.13510) | 本文针对无人机遥感影像场景，实证评估了Alibi-Detect、NannyML和PyTorch-OOD三种开源分布偏移检测工具在跨数据集偏移下的检测有效性与计算延迟。 |
| [^137] | [Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage](https://arxiv.org/abs/2611.13498) | 本文通过系统性评估发现，训练-测试分布中的组合等价性与模块覆盖度是决定 transformer 组合泛化性能的两个关键因素，直接模型的优异表现主要来自对组合等价性的利用，一旦消除这些等价性，性能将显著下降。 |
| [^138] | [Simultaneous Near-Optimal Region Identification in Continuous Simulation Optimization: Minimax Complexity and Adaptive Sampling](https://arxiv.org/abs/2611.13489) | 本文证明在连续设计空间上同时识别所有容差水平下的近优区域并非优化的副产品，给出了极小极大采样复杂度下界并提出了达到该下界的自适应采样方法。 |
| [^139] | [Differentially Private Statistical Inference for Rankings from Pairwise Comparisons](https://arxiv.org/abs/2611.13485) | 该论文首次在Bradley-Terry模型框架下实现了边差分隐私下的排名统计推断，通过留一分析给出有限样本误差界、渐近有效置信区间以及结合岭偏差校正与高斯-拉普拉斯校准的偏差修正方法，适用于保护隐私的大语言模型成对比较评估。 |
| [^140] | [Sharp Sample Complexity of KL-Regularized Reinforcement Learning](https://arxiv.org/abs/2611.13480) | 本文证明了在KL正则化强化学习中，插件估计器仅需Õ(KH³/(ε·max{ε,β})+KH)个样本即可找到ε-最优策略，并给出了匹配的下界，将先前样本复杂度对有效视界的依赖从H⁵改进为H³，且首次建立了依赖温度β的下界。 |
| [^141] | [Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling](https://arxiv.org/abs/2611.13477) | 该论文基于认知心理学中人类字母识别的发现，将视觉注意力建模为以注视点为中心的空间梯度分布并融入RLHF奖励模型训练，从而实现了比基线更高的偏好预测准确率。 |
| [^142] | [Deep Personalization: Continuous Context Observation for Local Proactive Agents](https://arxiv.org/abs/2611.13457) | 提出深度个性化闭环框架，通过独立的观察者智能体持续监控用户桌面活动、维护紧凑的长期用户上下文，并从隐式反馈中学习最佳干预时机，从而构建能够恰到好处地主动提供帮助的桌面智能体Coco。 |
| [^143] | [SLIC: Reinforcement Fine-Tuning Small LMs for Multi-Turn Analog Circuit Optimization](https://arxiv.org/abs/2611.13452) | 本文提出SLIC强化微调框架，将40亿参数的本地部署小语言模型训练用于闭环多轮模拟电路优化，其核心创新在于结合结构化电路表示、门控多目标奖励和奖励引导滚动规划（RGRS），并发布了包含41种放大器拓扑的开放基准测试集。 |
| [^144] | [Fermat ProbCover: Coverage-Based Active Learning on Fermat Distance Graphs](https://arxiv.org/abs/2611.13448) | 提出PC-Fermat方法，通过加权图上的截断最短路径算法以费马距离替代欧氏距离来构建覆盖邻域，从而在低预算主动学习中更好地捕捉数据的底层分布几何结构。 |
| [^145] | [ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding](https://arxiv.org/abs/2611.13442) | 该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。 |
| [^146] | [You Don't Need Feature Learning to Do In-Context Learning](https://arxiv.org/abs/2611.13431) | 通过对核机器的谱分析发现，上下文学习中的泛化解只是记忆解的低阶截断——模型先泛化后逐渐记忆化，因此实现上下文学习并不需要特征学习。 |
| [^147] | [Free Lunch Uncertainty: Sequence-Level R\'enyi Entropy from Particle Decoding for Adaptive LLM Reasoning](https://arxiv.org/abs/2611.13414) | 该论文发现SMC幂采样解码可零边际成本获得序列级Rényi熵，其中1/2阶熵在13个设置中的10个中表现最佳，是用于LLM自适应推理的零成本选择性预测信号。 |
| [^148] | [Placing LLM Lookup Tables by Per-Layer Prefetch Deadlines](https://arxiv.org/abs/2611.13413) | 提出 SlackTier，利用逐层预取时限（两个 Engram 层窗口相差 30 倍）将 LLM 的大规模 n-gram 查找表按层放置到不同存储层级，在解码 CUDA 图内实现位精确查找，使部分表可无需缓存直接放在 NVMe 上而几乎不损失性能。 |
| [^149] | [Bursts That Outlast Themselves: Persistent Degradation in Prefix-Cached Agent Serving](https://arxiv.org/abs/2611.13412) | 本文发现前缀缓存的 LLM 智能体服务具有亚稳态特性：即使负载恢复，短短几分钟的请求突发也会使服务器陷入缓存命中率骤降至约 10%、吞吐量减半且持续恶化的退化状态，其根源是滞留会话的工作集超出 KV 缓存池，导致每轮交互都变成代价高昂的重新计算。 |
| [^150] | [SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering](https://arxiv.org/abs/2611.13410) | 提出了SteerDistill，一种量化感知的蒸馏方法，通过全精度教师模型以闭环方式引导学生修订失败响应，并利用学生执行反馈迭代优化引导，从而在低比特量化下保持LLM智能体的任务成功率。 |
| [^151] | [The Dose Makes the Poison: Exposure-Aware Evaluation of Generative Drug Design](https://arxiv.org/abs/2611.13398) | 该论文利用基于生理的药代动力学（PBPK）模拟对生成式药物设计的候选分子进行暴露量感知的评估，揭示了传统静态分子性质筛选方法因忽略剂量可行性与安全性而会遗漏大量更优分子的重要缺陷。 |
| [^152] | [Plumbline: When the Loss Lies, Validating LLM Training Correctly](https://arxiv.org/abs/2611.13393) | 该论文提出了已在亚马逊生产环境落地的 LLM 训练正确性验证系统 Plumbline，它通过一套可跨模型和框架通用的系统性正确性方法论，对训练全生命周期（从训练数据到所服务的模型）进行检查，并能将显性与静默故障自动追踪到根本原因，从而避免漫长训练因故障而浪费。 |
| [^153] | [AdjointFlow: Accelerating Scientific Inverse Problems with Physics-Based Transport](https://arxiv.org/abs/2611.13389) | 提出AdjointFlow框架，通过随机插值将已知物理知识注入生成式传输过程本身，并利用基于伴随方法的数据相关初始化减少传输距离，从而加速科学逆问题的求解。 |
| [^154] | [Settling the Burn-in Gap: Sharp Risk Bounds for Tail-Averaged SGD in Linear Regression](https://arxiv.org/abs/2611.13383) | 该论文通过建立依赖初始化信息的匹配上下风险界，解决了尾平均SGD分析中头部特征空间初始误差方差贡献的“预热间隔”问题，并证明现有上界和下界均不够紧致。 |
| [^155] | [Discrete Model-Based Diffusion for Combinatorial Optimization in Multi-Robot Task Allocation](https://arxiv.org/abs/2611.13382) | 提出了一种无需训练的离散基于模型扩散方法，通过蒙特卡洛采样近似类别型反向转移分布，直接在离散空间进行基于采样的优化，从而求解多机器人任务分配中的NP难组合优化问题，并给出了找到全局最优解概率的有限步下界理论保证。 |
| [^156] | [Looping Deeper Without Growing the KV Cache: Recurrent Depth for Causal Encoder-Decoders](https://arxiv.org/abs/2611.13378) | 该论文提出一种循环因果编码器-解码器架构，通过将循环置于编码器以迭代精炼全局KV表示，并利用随机循环深度训练诱导不动点平衡实现跨层与跨轮KV缓存无损共享，从而在不增大KV缓存的前提下获得循环深度带来的性能提升。 |
| [^157] | [Enforcing Axioms for AI Alignment under Loss-Based Rules](https://arxiv.org/abs/2611.13369) | 本文将AI对齐原则建模为嵌入空间中的线性“投票者”，在基于损失的奖励模型训练框架下研究帕累托最优等公理能否成立，发现即使采用更丰富的奖励模型或修改比较数据，帕累托最优性仍可能被违反。 |
| [^158] | [Reconstructing Boundary Layer Wind Profiles with Machine Learning: A Comparative Study on IGRA Radiosonde Data](https://arxiv.org/abs/2611.13359) | 该研究基于158,097个IGRA探空数据比较了CatBoost、TabM和FT-Transformer等现代机器学习方法与传统解析参数化方法在边界层风廓线重建中的表现，发现机器学习将RMSE从约3 m/s大幅降至1.3-1.6 m/s，并揭示了预测精度主要取决于预测变量而非模型架构，同时给出了逐点预测器约1.77 m/s的经验精度下限。 |
| [^159] | [When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition](https://arxiv.org/abs/2611.13356) | 该论文提出了一个可复用的评估协议来公平比较冻结专家模型的特征组合与预测组合，并发现对六个架构多样的深度伪造检测专家特征进行逻辑回归，在留一生成器评估中比最佳预测组合器的宏平均 AUROC 提升 0.081。 |
| [^160] | [DEPUTYBENCH: Evaluating Contextual Safety in Enterprise Agents](https://arxiv.org/abs/2611.13355) | 该论文提出了可执行基准DEPUTYBENCH来评估企业智能体在组织权限与证据约束下的情境安全性，并揭示了一个关键风险：模型在证据缺失时比在明确禁止时更容易越权执行受限操作（10.3%对3.5%）。 |
| [^161] | [Advancing Mass-Conserving Deep Learning for Distributed Hydrologic Modeling: A Case Study in a Snow-Dominated Watershed](https://arxiv.org/abs/2611.13346) | 本文提出了一个质量守恒的卷积LSTM框架PECAN，通过严格（PECAN-MC）和放宽（PECAN-MCR）两种守恒约束变体实现分布式水文建模，并用于诊断物理感知深度学习模型的结构性局限。 |
| [^162] | [VC-Dimension and Generalization in Geometric Machine Learning with Applications to Active Learning](https://arxiv.org/abs/2611.13337) | 该论文计算了黎曼几何模型空间中度量球分类器的VC维并给出了多种情形下的上下界，进而在数据分布假设下利用覆盖论证证明了泛化界，并将其应用于主动学习中的基于覆盖的算法。 |
| [^163] | [Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA](https://arxiv.org/abs/2611.13331) | 该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。 |
| [^164] | [Robust Flow Matching under Target Corruption and Label Noise](https://arxiv.org/abs/2611.13324) | 该论文将流匹配训练建模为条件向量场上的时变回归问题，系统分析了目标污染与标签噪声对训练目标的影响，并提出时间残差分数（TRS）与比较条件分数（CCS）两种残差驱动的样本加权策略，显著提升了流匹配在数据污染场景下的鲁棒性。 |
| [^165] | [MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses](https://arxiv.org/abs/2611.13322) | 本文提出交互式工具 MicroLens，将 logit 透镜与雅可比透镜（J-lens）统一到同一张残差流“层×词元”网格上，实现对隐藏状态的读取、来源追踪与在线编辑，并发现 logit 透镜方向适合控制词元输出、J-lens 方向适合改变后续词元的读取。 |
| [^166] | [One Trajectory, Two Requirements: Accuracy and Local Stability in Euler Sampling for Flow Matching Policies](https://arxiv.org/abs/2611.13311) | 该论文通过精确双坐标系实验证明，流匹配策略的欧拉采样中轨迹精度与局部稳定性对步数的要求可能不一致，并在冻结的 π0.5-LIBERO 策略中发现动作端点附近存在强收缩方向。 |
| [^167] | [Train High, Ship Low: Efficient LoRA Synchronization for Distributed On-Policy Post-Training](https://arxiv.org/abs/2611.13306) | 提出与压缩器无关的 HiLo 框架，用压缩梯度分发加本地优化器重放取代 LoRA 参数传输，并配套预算感知的 Spar 梯度压缩器，将训练秩与通信成本解耦，从而在带宽受限网络上高效实现分布式在线策略后训练。 |
| [^168] | [DUO: Dual Readouts of Diffusion Models for OOD Detection](https://arxiv.org/abs/2611.13304) | 提出了一种完全无监督、无需训练的OOD检测方法DUO，通过将重构误差与中间去噪特征的主导敏感度谱相结合作为双重信号，即使重构误差产生误导时也能可靠地检测分布外输入。 |
| [^169] | [Neural Network-Assisted CLEAN for Efficient Multipath Parameter Estimation in MIMO Systems](https://arxiv.org/abs/2611.13298) | 本文提出神经网络辅助CLEAN（NN-CLEAN），用多头分类器替代传统多维搜索，在保留LS增益估计与残差消除的同时，实现了MIMO系统中更高效的多径参数估计。 |
| [^170] | [Time Series Change Point Detection with Gaussian Processes](https://arxiv.org/abs/2611.13295) | 该论文提出了一种基于高斯过程的通用变点检测框架，将变化定义为高斯过程先验超参数的改变，具有理论性能保证，且无需预先指定变化前后的时间序列函数形式，超参数可从数据中自动学习。 |
| [^171] | [Decode Token Parallelism: Memory-Efficient Distributed Attention Parallelism for Scalable LLM Decoding](https://arxiv.org/abs/2611.13293) | 提出解码令牌并行（TKNP）架构，通过将权重放置与注意力所有权解耦，解决了现有并行策略中KV缓存容量与注意力带宽难以兼顾的问题，实现了内存高效且可扩展的大语言模型解码。 |
| [^172] | [An Analytical Model of Compute-limited Multistage Training Pipelines](https://arxiv.org/abs/2611.13276) | 本文构建了一个解析可处理的高维多阶段训练理论模型（预训练+SFT+RL），揭示了算力在各训练阶段的分配规律，并证明有限的预训练数据会带来依赖表示的性能上限，其误差按预训练样本比例平方根的倒数缩放。 |
| [^173] | [Off-policy Learning with Excursion Policies](https://arxiv.org/abs/2611.13275) | 该论文提出“游走策略”这一结构化插值方法——智能体先按目标策略执行多步再回归行为策略，建立了其动态规划理论基础并开发了可证明可靠的离策略时序差分算法，在连续控制基准上实现了与混合策略相当的效率-稳定性权衡。 |
| [^174] | [Which Acoustic Features Survive Compression? Tracker-Estimated First-Formant Statistics for Codec-Robust Audio DeepfakeDetection](https://arxiv.org/abs/2611.13258) | 该论文发现第一共振峰（F1）统计特征在编解码器压缩后仍保持对深度伪造语音的判别力，仅用四个F1特征构建的SVM在七种编解码器条件下性能仅退化5.9个百分点（远优于36特征模型的14个百分点），为编解码器鲁棒的音频深度伪造检测提供了极简且稳定的特征方案。 |
| [^175] | [Domain-Routed Delta Optimization for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2611.13249) | 本文提出域路由增量优化方法（DRD-MOPD），通过为不同领域的样本指定各自更新的参数组件（以专家模型与基础模型的权重差初始化），在多教师在线蒸馏中显著保留原始专家的几何结构。 |
| [^176] | [GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement](https://arxiv.org/abs/2611.13243) | 该论文发布了开源 SDK GigaEdge Spench，通过共享 C++ 核心与多平台集成（服务器、移动端、浏览器）实现实用的高速流式语音增强，其模型质量和处理速度优于 RNNoise 和 DPDFNet2 且快于实时。 |
| [^177] | [Block-Sparse Gauss-Newton Optimization of Finite Basis Physics-Informed Neural Networks: Sparse Automatic Differentiation Approach](https://arxiv.org/abs/2611.13231) | 本文利用FBPINNs重叠域分解架构所诱导的雅可比矩阵块稀疏性，结合基于图着色的稀疏自动微分技术，实现了高效的高斯-牛顿二阶优化，大幅降低了计算成本与内存需求。 |
| [^178] | [Continuous Horizon Forecasting](https://arxiv.org/abs/2611.13228) | 本文提出连续视界（CH）预测策略，将预测重新定义为连续算子、把预测视界作为查询变量而非索引，在理论上证明了其相对于离散策略的估计方差优势，并通过大规模实验验证了其具有竞争力的精度，同时独特地支持并行推理和对未见视界的插值与外推。 |
| [^179] | [MediPFN: Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis](https://arxiv.org/abs/2611.13219) | MediPFN通过将预训练表格基础模型适配于干扰函数预测，并结合交叉拟合影响函数估计器与独立校准机制，为因果中介分析提供具有覆盖保证的置信区间。 |
| [^180] | [Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR](https://arxiv.org/abs/2611.13218) | 本研究在可控环境中证明RLVR在KL正则化最优时只是对被接受集合重新加权的过滤器（奖励作弊率精确继承自基础模型），而跨提示参数共享则通过提示间奖励与经验切核的协方差项产生漂移，使接受同一错误答案的提示推高它、拒绝的提示压低它，从而同时抑制并放大奖励作弊。 |
| [^181] | [Counting Under Disagreement: Dollar-Weighted Measurement, Selective Machine Labelling and Budgeted Human Review from Disagreeing AI Readers of Administrative Text](https://arxiv.org/abs/2611.13215) | 针对多个AI阅读者对行政文本标注存在分歧、且测量目标为货币总额的场景，本文提出了融合选择性机器标注与预算化人工复核的美元加权估计框架，在可控美元加权误差下实现对总额的准确测量。 |
| [^182] | [Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood](https://arxiv.org/abs/2611.13213) | 该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。 |
| [^183] | [Near-Optimal Sample Complexity of Deterministic Agnostic PAC Learning](https://arxiv.org/abs/2611.13210) | 本文证明了VC维为 $d$、直径为 $K$ 的概念类在确定性标签下的不可知PAC学习样本复杂度为 $\widetilde{O}(K^{1/3} d^{2/3}/\varepsilon)$，并构造了对数因子内紧的下界，填补了此前上下界之间的差距。 |
| [^184] | [HiCAP: Hierarchical Cultural Adaptation via Personality for Dialogue Generation](https://arxiv.org/abs/2611.13207) | 提出HiCAP框架，将个性特质通过分层控制机制融入对话生成——先使回应与文化基调对齐、再依据个性细化语言表达与内容，从而实现文化自适应的对话生成。 |
| [^185] | [Audited Gradients, Anchored Policies: Model-Based Reinforcement Learning for Efficient Flow Control](https://arxiv.org/abs/2611.13203) | 提出一种“梯度审计”机制，在基于代理模型的流动控制策略优化中，依据预先固定的接受准则将代理模型的动作梯度与高保真求解器的有限差分估计进行比对验证，从而在大幅降低计算成本的同时避免代理模型梯度误差导致的策略失效。 |
| [^186] | [XAI-Driven Confidence Estimation for OOD-Resilient Runtime Prediction](https://arxiv.org/abs/2611.13199) | 该论文提出LEX——首个用于基于机器学习资源管理的事后XAI运行时置信度估计框架，通过高斯混合模型聚类训练输入嵌入并关联逐簇预测误差界限，无需重训练即可提供细粒度的输入级置信度信号，在OOD工作负载上实现约94%的包含率，并产生比全局基线紧约17%的置信度界限。 |
| [^187] | [Structured Kernel Surrogate Models for Multimodal Attribution](https://arxiv.org/abs/2611.13198) | 提出结构化核代理模型及SKS算法，利用结构化核岭回归捕捉样本间与模态间的交互作用，首次解决多模态系统中的跨样本模态归因问题。 |
| [^188] | [EvoSurv: Evolution-Aware Multimodal Survival Modeling Reveals Complementary Prognostic Signals](https://arxiv.org/abs/2611.13190) | 该研究提出了eTime进化特征表示和EvoSurv多模态生存预测框架，将肿瘤基因组进化信息（克隆结构、突变谱、拷贝数改变等）与临床数据和RNA表达相结合，在六个TCGA癌症队列中全面超越现有最先进方法，揭示了肿瘤进化中蕴含的互补预后信号。 |
| [^189] | [One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink](https://arxiv.org/abs/2611.13188) | NVLinkOneSided提出了一种面向NVLink域内混合专家推理的单边all-to-all通信方法，通过免元数据交换的直接推送、原地读取机制以及Fence和Sentinel两种同步协议，在Blackwell GPU上比DeepEP快1.2-3.0倍。 |
| [^190] | [Prediction-Powered LLM Ranking with Adaptively Collected Human Feedback](https://arxiv.org/abs/2611.13186) | 提出PP-Rank框架，通过将廉价的LLM评判者预测与自适应收集的人类反馈相结合，在自适应采样下实现有效统计推断，显著提升大语言模型排名评估的精度。 |
| [^191] | [Making Strides in Symbolic Imitation Learning: Offline Demonstrations, Online Tuning](https://arxiv.org/abs/2611.13182) | 提出符号模仿学习框架，通过离线数据将oracle策略蒸馏为线性运动基元的决策列表，并结合在线ARS调优，在20个强化学习基准任务中逼近oracle性能且在11个任务中实现超越，同时生成简洁可解释的策略。 |
| [^192] | [RadRef: A Reference-Free Metric for Radiology Report Generation](https://arxiv.org/abs/2611.13168) | 提出了RadRef，一种无需参考报告的放射学报告生成评估指标，通过将报告陈述与图像推导的临床发现信念进行比对评分，在遗漏发现评估上与放射科医生排名高度一致，解决了无参考报告时无法评估的问题。 |
| [^193] | [High Probability Risk Control for Online Policy Learning](https://arxiv.org/abs/2611.13163) | 该论文提出一种基于置换的加权风险评估器（权重由Sinkhorn算法高效近似）结合高斯自助法构造风险置信上界，从而在反馈协变量偏移下实现高概率风险控制的在线策略选择。 |
| [^194] | [How Much of Your VLA Is kNN](https://arxiv.org/abs/2611.13160) | 研究揭示LIBERO基准上VLA的高成功率很大程度上只是近邻演示检索的体现——无需任何训练的检索方法即可达到91.7%成功率，并提出局部动作一致性指标来衡量检索质量并指导检索键训练。 |
| [^195] | [VIGOR: Benchmarking Visual Rationale Correctness in Multimodal Large Language Models](https://arxiv.org/abs/2611.13150) | VIGOR提出了一个通过将视觉推理依据的显式概念与像素级接地区域配对、跨越多个抽象层次，来评估多模态大语言模型是否真正“看着自己所说的内容”（即其答案是否由正确视觉证据支持）的基准测试。 |
| [^196] | [Everything in its right place: attention is a radio head](https://arxiv.org/abs/2611.13146) | 该论文提出注意力机制可以从物理同步中涌现——把历史词元建模为频率随时间漂移的振荡器、把当前词元建模为无线电接收器，通过求解Kuramoto同步问题闭式导出一个具有“锁定/静默”两相的注意力层，从而将Transformer注意力重新解释为无线电调谐接收过程，并自然实现选择、弃权、计数及超长序列泛化。 |
| [^197] | [Byzantine-Robust Muon: Resilient Momentum Orthogonalization for Distributed Learning](https://arxiv.org/abs/2611.13137) | 本文在统一的一阶信赖域框架下为拜占庭鲁棒的Muon优化器建立了不依赖独立同分布假设的收敛性保证，并引入了与信赖域几何相一致的非欧几里得数据异构性概念。 |
| [^198] | [Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation](https://arxiv.org/abs/2611.13135) | 该论文将多轮在线策略蒸馏中的成功参考选择建模为自由目标最优传输问题，并提出以最长公共子序列（LCS）为基础的动作序列对齐作为轨迹兼容性的可观测代理，为每条失败轨迹匹配最兼容的成功参考，从而避免误导性监督、提升蒸馏效果。 |
| [^199] | [ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration](https://arxiv.org/abs/2611.13114) | ASPIRE是一个由大语言模型智能体驱动的非线性系统辨识框架，能够自主选择并执行辨识方法，从测量数据构建闭式非线性模型，融入先验物理知识并独立验证，且支持自主与交互两种运行模式。 |
| [^200] | [Building World Models from Intrinsic Motivation](https://arxiv.org/abs/2611.13113) | 该论文将24种内在动机目标统一整合到一个基于模型的强化学习框架中，系统评估了它们所构建的世界模型在零样本、少样本和多任务控制下的表现。 |
| [^201] | [One Temperature to Rule Them All?](https://arxiv.org/abs/2611.13077) | 该论文研究了 pass@K 推理中温度多样性何时优于单一温度，发现虽然更难的问题更难共享高效的采样温度，但这一趋势仅随难度呈对数增长，表明单一调优温度在多数情况下已足够有效。 |
| [^202] | [IC-GNN: Modeling Individualized Brain Community Organization for Disease Prediction](https://arxiv.org/abs/2611.13070) | 提出了一种以图谱为锚点的图神经网络模型IC-GNN，能直接在图谱定义的脑区上构建个体化的多尺度大脑功能社区，从而捕捉与疾病相关的个体间差异并用于疾病预测。 |
| [^203] | [INTENT: Out-of-Sample, Out-of-Time, and Cold-Start Evaluation on User Event Sequences](https://arxiv.org/abs/2611.13069) | 提出了INTENT这一公开基准，根据用户、时期与目标项目是否在训练中出现，将用户事件序列上的下一动作预测模型划分到七种情形下评估，从而统一衡量其样本外、时间外与冷启动泛化能力。 |
| [^204] | [Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization](https://arxiv.org/abs/2611.13057) | 提出一种连续时间集合变换器，将异步采样的WiFi与IMU观测表示为带时间戳的无序标记集合，用单一注意力块实现跨模态与时间的联合融合，并通过随机掩码训练应对缺失或过时的传感器输入，用于室内定位。 |
| [^205] | [Explanation-Aware Fairness Beyond Output Parity: Joint Output-Attribution Regularization for Medical Image Classification](https://arxiv.org/abs/2611.13052) | 本文提出一种联合输出-归因正则化方法，通过惩罚不同人口群体间的特征归因差异来补充输出层面的均衡几率公平约束，在医学图像分类任务上显著改善了效用与公平性的权衡表现。 |
| [^206] | [Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding](https://arxiv.org/abs/2611.13017) | 论文揭示了基于乘积排序的草拟树在多未来场景下会将大量概率预算浪费在目标模型从不产生的“嵌合体”前缀上（占64节点树的27-33%），并提出多未来草拟（MFD），用潜变量（K个可学习模式嵌入）替代链式法则来恢复多个未来之间的耦合，从而提升树验证式推测解码的接受效率。 |
| [^207] | [Permutation-Based Pre-trained Model Merging via Reciprocal Orthogonality](https://arxiv.org/abs/2611.13016) | 提出通过搜索促进相互正交性（RO）的置换来合并预训练模型，无需大幅加宽网络即可抑制模型间的干扰，使两个模型的计算能够在合并网络中共存。 |
| [^208] | [FaST: Factored Synthesis of Transitions for Offline Reinforcement Learning](https://arxiv.org/abs/2611.13013) | FaST通过链式法则将强化学习转移分解为状态、动作、动力学、奖励和终止五个阶段分别建模，在匹配扩散基线性能的同时将合成速度提升90倍以上，并支持针对固定状态进行高优势动作的定向合成。 |
| [^209] | [The Shape of a Program: Path Signatures for Trace-to-Program Induction](https://arxiv.org/abs/2611.13005) | 本文将粗路径理论引入程序归纳，用路径签名变换编码执行轨迹，在 Karel、List 和 RASP 三个领域几乎全面超越 LSTM、Transformer 和傅里叶基线（尤其在数据稀缺时优势最大），并利用流程天然的收缩性构建无需外部预言机的自举方法来揭示监督数据的冗余。 |
| [^210] | [Branch-and-Commit for Frozen RL Agents: When and Where to Spend Test-Time Compute](https://arxiv.org/abs/2611.13003) | 该论文提出针对冻结RL智能体的“分支与提交”方法，发现分支收益取决于候选回报的离散度——低离散度时无收益，高离散度时可提升回报12.8%，且收益主要来自在不同状态间选择而非选择更优动作。 |
| [^211] | [One Table, Many Queries: Benchmarking Generalist Tabular Models Across Generation, Imputation, and Prediction](https://arxiv.org/abs/2611.13001) | 该论文提出One Table Benchmark基准，通过56个数据集、46种方法和20个能力维度的统一评估，首次系统揭示了表格数据的生成、插补与预测三类任务之间的能力关联和训练目标的相互作用，并证明单一通用模型有望同时胜任这三类任务。 |
| [^212] | [Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Large Language Models](https://arxiv.org/abs/2611.12998) | 该论文提出一个四阶段贝叶斯分析流程，将探针准确率分解为可解码性、形状、归属权和因果使用四个独立命题并分别检验，从而更严谨地判定大语言模型激活中的几何结构（如编码算术运算的螺旋或圆形）是否真正被模型用于计算。 |
| [^213] | [ZeroPoint: Lightweight Energy-Based Decision Model](https://arxiv.org/abs/2611.12992) | 本文提出ZeroPoint，通过引入一个对任意输入能量恒为零的“空答案”固定状态，为基于能量的决策模型提供显式的能量参考水平，使模型能够判断并拒绝“没有任何可接受选项”的情形。 |
| [^214] | [Multimodal Graph Learning: A Survey of Models, Benchmarks, and Frontiers](https://arxiv.org/abs/2611.12990) | 本综述针对多模态属性图学习，提出了统一的消息传递建模框架，将现有方法按五个集成层次（特征、传播、拓扑、学习和基础模型）分类，并系统梳理与评估了相关基准与前沿方向。 |
| [^215] | [Third Eye: Beyond Immediate Gains in Recursive Self-Improvement](https://arxiv.org/abs/2611.12988) | 论文通过大规模受控实验发现，递归自我改进中看似有用的自我更新在经过第二轮更新后效用会进一步降低（甚至候选平均即时效用已为负值），因此对自我改进的评估必须超越第一步的即时收益。 |
| [^216] | [MADSNet: A Multi-Scale Axial-Attentive Dense Spiking Network for Spike Camera Image Reconstruction](https://arxiv.org/abs/2611.12980) | 提出了一种多尺度轴向注意力密集脉冲网络MADSNet，通过分层脉冲编码器-解码器结构实现高质量脉冲相机图像重建，在SRED数据集上显著超越现有SNN重建方法。 |
| [^217] | [Partition, Refine, and Detect: Preserving Novel Category Capacity in Open-Vocabulary Object Detection](https://arxiv.org/abs/2611.12976) | 提出PR-OVD“分区-精炼”框架，通过显式为新类别保留嵌入容量并增强类别分离，解决了开放词汇目标检测中由基类主导表示导致的语义坍塌与置信度失准问题。 |
| [^218] | [Attribute-Evidence Learning with Multimodal Prompt for Generalized Category Discovery](https://arxiv.org/abs/2611.12975) | 提出 AptM-GCD 多模态框架，通过属性-证据学习机制将属性作为局部视觉证据而非全局类别描述符，并结合多模态提示推理，避免了文本与全局图像表示纠缠导致的误导性预测，从而更准确地发现新类别。 |
| [^219] | [StatPrompt-LLM: A Lightweight Statistical Refinement for Foundation-Style Time-Series Forecasting](https://arxiv.org/abs/2611.12974) | 提出 StatPrompt-LLM，通过在 Time-LLM 的提示中融入数据集级统计信息并增加一个零初始化自回归线性旁路，以轻量级方式增强基础模型风格的时间序列预测能力。 |
| [^220] | [ADIndex: Layer-Adaptive Sparse Attention Indexing via Group-Level Distillation](https://arxiv.org/abs/2611.12956) | ADIndex通过逐层校准与组级蒸馏训练层级自适应的稀疏注意力索引器，并辅以低延迟推理策略，大幅降低长上下文推理中索引环节的延迟开销。 |
| [^221] | [Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series](https://arxiv.org/abs/2611.12955) | ConceptTime 利用冻结概率预测器产生的“预测性惊奇”这一单一自监督信号，同时完成时间序列分段定位与概念表征，打破了边界与模式相互依赖的循环，从而在时间序列分类中同时实现准确性、可解释性和证据不足时的弃权能力。 |
| [^222] | [Tabular Foundation Models vs Tuned Gradient Boosting for Telecom Churn: A Profit- and Calibration-Aware Evaluation](https://arxiv.org/abs/2611.12942) | 该研究以期望最大利润和概率校准为评判标准，将表格基础模型与经Optuna调优的梯度提升树在电信客户流失预测上进行严格重新评估，发现TabPFN-3.5在部分数据集上能显著带来更高利润，修正了以往仅用排序指标、与未调优基线比较所得出的结论。 |
| [^223] | [Reinforcement Learning based control of Cancer Chemotherapy treatments](https://arxiv.org/abs/2611.12940) | 该论文系统梳理了强化学习技术在癌症化疗治疗控制中的应用现状，重点研究了用于生成最优动态治疗方案的最新强化学习方法。 |
| [^224] | [Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring](https://arxiv.org/abs/2611.12935) | 提出了TopicScope框架，利用稀疏自编码器（SAE）特征训练因果主题分类器，在预填充阶段提取特征训练后可泛化到解码阶段，实现对生成过程中每个token的实时多主题监控且标签可中途变化。 |
| [^225] | [Topology-Aware Reinforcement Learning for CAD Decomposition toward All-Hexahedral Meshing](https://arxiv.org/abs/2611.12930) | 提出DecompRL——一个基于深度强化学习的CAD模型分解框架，通过拓扑感知的内部分割操作突破了传统仅限表面分割的动作空间限制，能够为复杂特征配置生成有效的可网格化分解，推动全六面体自动网格生成这一“圣杯”问题的解决。 |
| [^226] | [An ensemble of graph generative models for assisted history matching of the Brugge field case](https://arxiv.org/abs/2611.12927) | 该论文提出一种图生成模型集成方法，为Brugge油田的每个地层沉积带分别建立专用图自编码器，共享联合潜在空间并使用CMA-ES进化策略进行优化，直接在角点网格活跃单元上操作，从而实现对具有多种沉积环境油田的辅助历史拟合。 |
| [^227] | [IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction](https://arxiv.org/abs/2611.12918) | 该论文揭示了离线网页行为预测基准中仅凭候选动作是否属于训练支持集即可“作弊”答对的混杂效应，提出支持可识别率（SIR）这一度量、支持均衡候选集构建方法以及免目标双路径提议器，从而真实刻画从理想评测条件到实际部署的性能差距。 |
| [^228] | [M\"obius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection](https://arxiv.org/abs/2611.12913) | 提出莫比乌斯交互过滤方法，通过阶数选择性谱正则化在保留高阶模态交互表示的同时调控其对预测的贡献，使多模态幽默检测的标签对齐信噪比提升1.62-2.79倍。 |
| [^229] | [Generative Learning on Manifolds for Probabilistic Seismic Performance Assessment and Structural Identifiability](https://arxiv.org/abs/2611.12908) | 该研究提出一种结合扩散映射、分数扩散模型与几何调和的流形生成学习框架，可从有限的非线性仿真中学习结构参数与地震响应的联合分布，同时实现地震性能的正向概率预测与结构特性的反向辨识。 |
| [^230] | [Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation](https://arxiv.org/abs/2611.12904) | 该论文针对临床文档场景中用户编辑数据并非全部可操作的问题（如45%以上编辑包含模型输入不支持的信息），构建了临床医生编辑行为分类体系并开发了区分可操作反馈的方法，以避免微调导致模型产生幻觉。 |
| [^231] | [D-TINs for Track-Sampled Ice Sheet Surface Reconstruction from Satellite Altimetry Data](https://arxiv.org/abs/2611.12894) | 本文提出D-TIN方法，直接对卫星测高观测点进行Delaunay三角剖分和重心插值以重建冰盖表面，在精度接近克里金法的同时将运行时间降低了252倍。 |
| [^232] | [XposedMIA: Exploiting Explainable Artificial Intelligence (XAI) for One-Shot Membership Inference Attacks](https://arxiv.org/abs/2611.12067) | 该论文提出XposedMIA攻击框架，首次将LIME局部代理几何及表格和视觉反事实解释转化为样本到决策边界距离的代理指标，仅需30个样本进行一次性校准即可实施单次黑盒成员推断攻击，揭示了可解释性输出泄露训练数据成员信息的隐私风险。 |
| [^233] | [Fade to... Red? Scheduling Coloured Noise Exploration Supports Efficient Robot Learning with Reduced Movement Jerk](https://arxiv.org/abs/2611.11868) | 在机器人强化学习训练中，将探索噪声从白噪声逐渐过渡到红噪声（有色噪声），可在不损失学习效率和最终性能的前提下，将探索动作的抖动减半，实现更平滑高效的学习。 |
| [^234] | [Can Hindsight Critics Guide Delayed Actors?](https://arxiv.org/abs/2611.11863) | 提出Hindsight Critics方法，利用事后对齐的状态-动作转移训练评论家并生成特权价值估计来指导仅能获得延迟观测的行动者，在无需修改部署阶段观测的情况下部分补偿了感知延迟。 |
| [^235] | [When Should Diffusion Language Models Reconsider Their Commitments?](https://arxiv.org/abs/2611.11551) | 该论文提出PURL框架，通过联合学习编辑选择、取消掩码选择和词元预测三个决策，使掩码扩散模型能够在并行快速生成的同时有选择地修正早期不一致的词元承诺，且推理时三个决策仅需一次共享前向传播即可完成。 |
| [^236] | [Alchemically Transferable Parameters for Variational Quantum Eigensolvers](https://arxiv.org/abs/2611.11445) | 该论文提出一种“炼金术”几何投影方法，将有机分子的局部键合环境映射为替代氢基片段，使机器学习模型无需针对有机分子重新训练，即可直接预测变分量子本征求解器所需的电路参数。 |
| [^237] | [STABLE: A Continual Learning Optimizer with Adaptive Drift Control](https://arxiv.org/abs/2611.10655) | 该论文提出STABLE，一种具有自适应漂移控制的原对偶优化器，理论上证明有界参数漂移可保证有界的间隔侵蚀，并通过动态调节对偶变量来强制执行漂移预算，从而从优化器层面缓解持续学习中的灾难性遗忘。 |
| [^238] | [Seeing Isn't Enough: Unlocking Visual Reasoning by Bridging the Reasoning-Access Gap with TIGER](https://arxiv.org/abs/2611.10552) | 该论文揭示了多模态大语言模型中存在的“模态推理访问差距”——模型虽能准确感知图像内容却无法将其路由进已有的文本推理机制——并提出TIGER训练方法，通过挖掘模型在文本上成功但在对应对称图像上失败的模态反事实样本来定向弥合这一差距，从而解锁视觉推理能力。 |
| [^239] | [Split and Bridge: Multimodal Generation via Diffusion Bridging](https://arxiv.org/abs/2611.10441) | 提出了一种自下而上的多模态生成框架 SnB，通过在采样时利用桥接模块连接预训练的单模态扩散模型，无需联合训练即可生成连贯的多模态样本，实现了预训练模型的即插即用复用。 |
| [^240] | [Continual Instruction Tuning for LLM-ased Management of Continuum Resources](https://arxiv.org/abs/2611.10292) | 该论文提出将大语言模型作为持续指令微调的意图编译器（而非嵌入延迟敏感控制回路中），置于快速确定性战术控制器之上，并通过执行评分在控制器实际目标中度量灾难性遗忘，从而实现云-边连续体资源的高效LLM管理。 |
| [^241] | [TORNADO: Adaptive Latent Stochastic Transport for Calibrated Probabilistic PDE Forecasting](https://arxiv.org/abs/2611.10251) | TORNADO提出了一种潜空间概率预测框架，通过学习具有状态依赖扩散项的SDE在VAE后验分布之间进行随机输运，实现无需模拟的训练，从而在计算受限的少步推理下为PDE预测提供校准的不确定性量化。 |
| [^242] | [Counting Events, Not Seeing Them: Diagnosing and Repairing the Readout Failure in Video LLMs](https://arxiv.org/abs/2611.09949) | 该论文发现视频大语言模型物体计数失败的根源并非缺少物体恒常性，而是计数信号已存在于内部表征中却未被正确读取，并通过片段定位监督与离散事件符号辅助通道两种修复方法，将同一物体事件计数准确率从46.7%大幅提升至65.0%。 |
| [^243] | [Knowing When to Intervene: Causal Evaluation of AI Agents](https://arxiv.org/abs/2611.09792) | 该论文提出了首个基于因果推断的方法学框架，用于评估在AI智能体执行轨迹中进行干预（如人工审查或行为引导）的因果价值。 |
| [^244] | [BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression](https://arxiv.org/abs/2611.09505) | 提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。 |
| [^245] | [Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity](https://arxiv.org/abs/2611.07746) | 本文提出QS-TCN方法，将准极大似然估计的波动率持续性统计量与矩阵束方法提取的时序模式嵌入时序卷积网络的结构化滤波器中，从而基于波动率遗传关系对多变量时间序列进行全局排序。 |
| [^246] | [ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus](https://arxiv.org/abs/2611.06706) | 该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。 |
| [^247] | [EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models](https://arxiv.org/abs/2611.06356) | 该论文发布了首个即用型EEG基础模型公开训练语料库eegdash-corpus-v0，汇集了77个可合法再分发的公开数据集（涵盖7,272名受试者的14,075小时数据），解决了临床档案无法再分发、公开数据分散且格式不一导致难以汇集的核心问题。 |
| [^248] | [Do Language Models Know Which Constraints Belong Together?](https://arxiv.org/abs/2611.05726) | 该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。 |
| [^249] | [Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics](https://arxiv.org/abs/2611.05538) | 该研究证明，无需大规模脑电预训练的通用视觉语言模型能够像临床医生一样以视觉方式解读脑电信号，并在临床神经诊断任务上取得有竞争力的表现，从而为构建脑电图基础模型提供了摆脱对大规模脑电数据依赖的新范式。 |
| [^250] | [Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits](https://arxiv.org/abs/2611.05355) | 该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。 |
| [^251] | [AGiR: Mitigating Gift Over-Reliance in Mixed-Motive Games](https://arxiv.org/abs/2611.02875) | 提出了一种与送礼方法无关的自适应礼物拒绝机制AGiR，通过让智能体选择性地拒绝部分收到的礼物作为策略更新信号，缓解混合动机博弈中的礼物过度依赖问题，可作为即插即用模块兼容现有送礼方案并支持去中心化学习。 |
| [^252] | [On Stochastic Continuous-Time Linear System Identification for Adaptive Stabilization](https://arxiv.org/abs/2611.02340) | 本文提出了基于系统辨识的自适应镇定的失败概率与轨迹长度和数量、维度、可控性Gramian矩阵及信噪比等所有相关量之间简单、实用且可靠的定量关系。 |
| [^253] | [Stand-In Effects and Pooling Amplification: Lessons from Biological World Models for Testing Action Effects](https://arxiv.org/abs/2611.01666) | 该论文揭示了检验世界模型动作效应的常用诊断存在两类系统性失败——自相关设计下未来动作“替身”未见历史导致虚假效应、多数据源池化将微小模型误差放大为反对真实效应的压倒性假证据——并提出以设计所用完整状态为条件的e值条件随机化检验，实现有限样本下有效的动作效应检验。 |
| [^254] | [The Rise of Low Precision Floating Point](https://arxiv.org/abs/2611.00301) | 本文探讨了深度学习的兴起如何推动16位、8位乃至4位等低精度浮点格式的发展及其标准化进程。 |
| [^255] | [Variance Reduction for Decentralized Stochastic Optimization with Multiple Local Updates](https://arxiv.org/abs/2610.41927) | 本文提出了一种将方差缩减与多次局部更新相结合的原始-对偶去中心化优化算法，其步长与网络规模无关且仅与局部更新次数成反比，克服了现有ADMM类方法因步长受限而随网络规模增大收敛变慢的问题。 |
| [^256] | [Textual Response-Aware Guidance for Zero-Shot Adversarial Robustness of Vision-Language Models](https://arxiv.org/abs/2610.40309) | 提出文本响应感知引导方法TRAG，利用预训练文本编码器对输入嵌入局部变化的响应来约束视觉表示漂移，从而在不损害零样本性能的前提下提升视觉-语言模型的对抗鲁棒性。 |
| [^257] | [Diffusion models for the $O(3)$ nonlinear sigma model](https://arxiv.org/abs/2610.39568) | 该论文在 O(3) 非线性西格玛模型的格点表示（二维海森堡模型）上训练基于分数的扩散模型，通过多尺度粗粒化采样方案适度提升了直接采样的精度与计算效率，并引入 Stein/Schwinger-Dyson 诊断方法，既可在不依赖反向扩散采样的情况下评估学到的分数质量，又可作为额外训练损失加以利用。 |
| [^258] | [Learning to Reason from Reference Samples Alone: Reinforced Adversarial Learning for Modern LLM Post-Training](https://arxiv.org/abs/2610.38851) | 该论文提出强化对抗学习（RAL）方法，通过共同训练的大语言模型评判者，仅凭参考答案即可作为推理型大语言模型在策略后训练的唯一奖励信号，无需显式可验证奖励或参考推理轨迹，且实验表明其性能可媲美直接的RLVR。 |
| [^259] | [Muon Makes the First Mixture-of-Experts Layer Harder to Balance](https://arxiv.org/abs/2610.37914) | 研究发现 Muon 优化器会使混合专家语言模型第一个 MoE 层的路由偏置跨度平均翻倍，导致该层负载严重失衡（最繁忙专家承担四到七倍公平份额），且这一效应源于 Muon 对网络其余部分的作用而与路由器的训练方式无关。 |
| [^260] | [Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss](https://arxiv.org/abs/2610.36202) | 本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。 |
| [^261] | [A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis](https://arxiv.org/abs/2610.35658) | 该论文提出一种混合框架，利用本地大语言模型从5.6万余份美国航空事故报告中提取因果链并构建因果知识图谱，结合基于物理的失效发生模型与漏泄噪声OR方法，将文本因素量化为失效模式概率，实现可解释的航空风险量化与干预分析，其提取的因果链可恢复90.8%的调查人员标注原因。 |
| [^262] | [Query Complexity of Learning Axis-Aligned Partitions](https://arxiv.org/abs/2610.35495) | 本文确立了精确查询学习设定下学习高维轴对齐盒子划分的查询复杂度，证明了经典主动学习算法 CAL 在一般设定下具有极小极大最优的查询复杂度 $\Theta(kd \log (n/kd))$，并在盒子互不相交的常数 $k$ 情形下给出了能超越 CAL 一个 $\log(d)$ 因子的 Halving 算法优化变体。 |
| [^263] | [Nominate-then-Adjudicate: LLM-Assisted One-Pass and Per-Instance MIP Solver Tuning](https://arxiv.org/abs/2610.29391) | 该论文提出“先提名后裁决”流水线：先一次性采样并复用固定配置组合作为候选名单，再由无任务微调的大语言模型在推理时以一次模型调用、零求解器调用完成逐实例选择，从而无需昂贵的逐实例配置搜索即可实现MIP求解器的逐实例调优。 |
| [^264] | [MSCR: Jointly Balancing Utilization Across Modalities and Discovering Synergistic Information](https://arxiv.org/abs/2610.24395) | 提出MSCR融合方法，通过反主导目标抑制单模态过度影响并促进协同信息的发现，从而联合实现模态利用的平衡与多模态优势的充分挖掘。 |
| [^265] | [Data Density Scaling Laws for Image Self-Distillation](https://arxiv.org/abs/2610.19013) | 本文发现图像自蒸馏中样本重复的收益依赖于具体数据集，小数据集重复训练甚至可能优于大数据集，并提出同时考虑独特样本数量与每样本重复次数的改进版 Chinchilla 缩放定律。 |
| [^266] | [Compute-Optimal Scaling Laws for Spiking Language Models](https://arxiv.org/abs/2610.15146) | 本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。 |
| [^267] | [GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning](https://arxiv.org/abs/2610.10201) | 该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。 |
| [^268] | [Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking](https://arxiv.org/abs/2610.09919) | 本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。 |
| [^269] | [ORCA: Hunting Compositional Failures in Text-to-Image Diffusion](https://arxiv.org/abs/2610.09841) | 提出ORCA方法，将跨模态组合结构对齐作为低秩辅助损失直接融入扩散模型训练，从而解决文本到图像生成中的属性绑定错误、空间关系颠倒和多对象计数失败等组合性问题。 |
| [^270] | [Breaking Adversarial Transferability in Fine-Tuned Speech Recognition](https://arxiv.org/abs/2610.09109) | 本文揭示了在公开预训练模型上构造的对抗扰动可有效迁移到黑盒部署的微调语音识别模型，并提出统一微调框架TransferBreaker，通过基础对抗微调、潜在雅可比正则化与混合梯度自适应微调来阻断对抗迁移并提升鲁棒性。 |
| [^271] | [CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency](https://arxiv.org/abs/2610.07668) | CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。 |
| [^272] | [LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models](https://arxiv.org/abs/2609.40063) | LARC 通过在冻结模型中引入低秩自适应残差连接，利用跨任务学习初始因子的慢状态与随反馈动态调整的快状态相结合的双状态机制，使冻结模型无需更新原始参数即可从反馈中学习，在程序选择任务中显著降低查询执行误差。 |
| [^273] | [LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization](https://arxiv.org/abs/2609.38166) | LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。 |
| [^274] | [Learning Perturbation Robust Policies for LLM Agents with Stable Optimization](https://arxiv.org/abs/2609.34064) | 提出稳定扰动鲁棒策略优化方法 SPrPO，通过在强化学习训练中施加自适应、敏感度感知的扰动，使大语言模型智能体的策略在面对隐藏状态噪声、剪枝和量化等扰动时保持鲁棒，同时确保优化过程的稳定性。 |
| [^275] | [Derivative-Informed Training of Neural Operators On-the-Fly via Sketched Tangent Consistency](https://arxiv.org/abs/2609.32797) | 提出草图化切向一致性损失，这是一种无需离线导数标签、在训练中直接利用PDE控制方程通过随机草图扰动即时强制执行敏感性一致性的轻量级导数信息训练方法。 |
| [^276] | [Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events](https://arxiv.org/abs/2609.30746) | 该论文提出一种机制感知的即插即用集合条件化框架，利用受推动粗糙集合的协方差作为局部不稳定性的无需雅可比矩阵的代理，通过小型FiLM模块将集合几何统计信息注入骨干网络，从而在数据受限条件下实现对混沌系统极端事件的准确仿真。 |
| [^277] | [Binary Quantized Neural Network Training Is W[1]-Hard Parameterized by Input and Output Dimensions](https://arxiv.org/abs/2609.27932) | 本文证明了二值量化神经网络训练在仅以输入与输出维度之和为参数时是W[1]-难的，即使在零误差、样本构成前缀链且偏置固定为零的最简情形下依然成立，从而否定了Ganian等人留下的开放问题。 |
| [^278] | [GraphToolbox: A Configurable Python Framework for Graph Neural Network Forecasting](https://arxiv.org/abs/2609.24609) | GraphToolbox是一个基于PyTorch Geometric的开源Python框架，通过配置驱动的统一流水线，整合了图神经网络电力预测中的图构建、模型选择、训练、聚合与可解释性分析等全部环节。 |
| [^279] | [Acceptance-Aware Draft Model Training for Speculative Decoding](https://arxiv.org/abs/2609.24150) | 该论文提出了直接优化投机窗口内期望接受词元数量的接受长度感知训练损失——针对贪心验证的期望接受长度（EAL）损失和针对采样解码的窗口总变差（WTV）损失，从而提升大语言模型推理的加速效果。 |
| [^280] | [Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models](https://arxiv.org/abs/2609.12277) | 本文提出一种将EHR基础模型视为患者轨迹生成式策略的强化学习微调框架，通过时间感知的奖励设计提升临床推理能力，使小模型在数据受限场景下超越大型预训练模型。 |
| [^281] | [LookThere! Sparse Vision by Reinforced Selection](https://arxiv.org/abs/2609.04698) | 提出LookThere端到端强化学习框架，联合训练浅层输入选择器与深层表示提取器，在不依赖辅助信号的情况下实现极端稀疏条件下性能与计算权衡的新帕累托前沿。 |
| [^282] | [Equation Recast for Canonical Operator Learning Across Parametric PDEs](https://arxiv.org/abs/2609.02982) | 提出方程重构方法，将参数化偏微分方程的算子学习转化为单一规范算子的学习，通过解析推导吸收参数引起的算子变化到有效源项中，从而实现对新参数范围的零样本预测、外推以及融合稀疏异构数据的能力。 |
| [^283] | [FrOGS: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions](https://arxiv.org/abs/2609.02948) | FrOGS是一种将自回归模型与连续时间马尔可夫链在单一损失下联合训练的混合离散神经采样器，可跨多种化学条件独立采样合金构型，并给出配分函数的无偏估计和热力学性质的一致性估计。 |
| [^284] | [Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization](https://arxiv.org/abs/2609.02126) | 本文提出一种可扩展的复合函数贝叶斯优化方法（SBOCF），利用图像匹配目标的复合结构和模拟图像的中间信息，高效地从PACBED图样中估计样品厚度和晶体失倾等关键参数，克服了网格搜索效率低下和神经网络需大量预训练且难以迁移的缺点。 |
| [^285] | [ST$^2$U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control](https://arxiv.org/abs/2608.23034) | ST²U通过轨迹级别的受限知识边界控制，解决了现有测试时遗忘方法因忽略自回归生成中状态重建而导致的受限知识重入问题。 |
| [^286] | [SplitLite: Low-Rank Residual Compression for Split Learning](https://arxiv.org/abs/2608.23018) | SplitLite通过利用相邻周期激活和梯度残差的低秩结构（秩-2r和秩-4r），仅传输量化的截断奇异值，大幅降低分割联邦LoRA微调的通信成本。 |
| [^287] | [Auditing Cross-Lingual Fairness in Language Model Watermarking](https://arxiv.org/abs/2608.20047) | 提出一个四部分评估框架，用于审计语言模型水印在跨语言场景中的公平性，并通过实证校准和多种测量揭示英语评估掩盖的多语言部署问题。 |
| [^288] | [Multimodal Model Diffing for Feature Discovery and Control](https://arxiv.org/abs/2608.09928) | 提出了MMDiff多模态模型对比框架，通过训练多模态稀疏自编码器并将其作为特征级接口，实现多模态大语言模型中被多模态训练改变的特征的发现、隔离与针对性控制。 |
| [^289] | [A Distribution Mapping Approach to Counterfactually Fair Reinforcement Learning](https://arxiv.org/abs/2608.08743) | 本文提出一种基于分位数分布映射的数据预处理算法，通过序贯估计反事实状态和奖励，在强化学习中实现反事实公平性，并将常用的可加性假设作为特例涵盖其中。 |
| [^290] | [DynaCrys: Crystal Generation with Dynamic Space-Group Diffusion](https://arxiv.org/abs/2608.07401) | DynaCrys是首个让空间群与Wyckoff占据及元素在生成过程中共同动态演化的完整晶体生成框架，通过显式融入晶体学群-子群关系的物理引导空间群扩散和合法性约束的随机解码器，联合采样相互兼容的空间群、Wyckoff占据与元素。 |
| [^291] | [RamanPFN: learning from Raman spectral structure with a tabular foundation model](https://arxiv.org/abs/2608.02157) | 提出RamanPFN——一个通过物理引导的光谱学习（结合全局成分解混GCU与局部振动子空间编码LVSE）实现统一上下文推理的通用拉曼光谱基础模型框架。 |
| [^292] | [QAdapt: Noise-Adaptive Neural Pre-Decoding for Surface-Code Quantum Error Correction](https://arxiv.org/abs/2607.28422) | QAdapt提出了一种噪声自适应神经预解码框架，通过学习局部时空相关性并结合在线弹性权重巩固，在多种噪声机制间持续适应，显著降低了表面码量子纠错的逻辑错误率并提升了解码效率。 |
| [^293] | [Total Variation Distance Estimation in Autoregressive Models](https://arxiv.org/abs/2607.19510) | 本文研究了在样本访问、logit 访问和含噪 logit 访问三种模型下估计两个长度为 $n$ 的自回归分布之间全变差距离的问题，将样本访问下的查询复杂度从已有的 $\widetilde{O}(n^3 m/\varepsilon^5)$ 改进为 $\widetilde{O}(n^2 K/\varepsilon^2)$，并给出了 logit 访问下紧的 $O(n/\varepsilon^2)$ 查询界。 |
| [^294] | [Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning](https://arxiv.org/abs/2607.19181) | 该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。 |
| [^295] | [Disentangling Forced and Internal Climate Variability in Single Realizations using Dynamic Mode Decomposition with Control](https://arxiv.org/abs/2607.18298) | 该研究提出PullbackDMDc方法，基于非自治动力系统理论、带控制的动态模态分解与拉回吸引子估计，将外部强迫视为动力学驱动因子，从而实现从单一气候观测记录中分离强迫趋势与内部气候变率。 |
| [^296] | [On the Potential of Graph Neural Networks as Metamodels for Supply Chain Optimization: Dataset, Architectures, and Directions](https://arxiv.org/abs/2607.16769) | 本文形式化了供应链优化问题，利用SupplyNetPy仿真库创建了大型公开的供应链图训练数据集，并探索了可作为仿真代理的图神经网络架构及其精度与计算成本的权衡，为GNN在供应链结构优化中的应用奠定基础。 |
| [^297] | [Machine Learning-Based Reconstruction for Resistive Silicon Sensors](https://arxiv.org/abs/2607.11585) | 本文提出了一种基于机器学习的电阻性硅传感器（如AC-LGAD）信号重建与压缩方法，利用多焊盘全波形信息进行正则化，解决了传统矩阵求逆方法计算复杂且对非对角噪声敏感的问题，从而在放宽读出间距的条件下实现高精度空间重建。 |
| [^298] | [SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks](https://arxiv.org/abs/2607.09745) | SupplyNetPy是一个开源Python库，可对任意多级结构的供应链与库存网络进行高保真离散事件仿真，支持多种补货策略、易腐库存、节点中断与随机需求，其组件可继承扩展，并经过基准对比验证，支持模型的程序化生成与设计空间探索。 |
| [^299] | [XS-VLA: Teaching Tiny Vision-Language-Action Models with Spatial Supervision and Demonstration Conditioning](https://arxiv.org/abs/2607.04171) | XS-VLA通过教师空间蒸馏与示范条件化流匹配的两阶段训练，在仅约244M参数的紧凑视觉-语言-动作模型上实现了90.25%的LIBERO平均成功率，超越了更大规模的基线模型。 |
| [^300] | [Adaptive Reparametrized Time for Score-Based Diffusion Sampling](https://arxiv.org/abs/2607.02137) | 提出自适应重参数化时间方法，通过将采样时钟速度作为控制量并结合连续时间强化学习（ART-RL），自适应地为扩散采样分配时间步，克服了均匀和手工设计调度的次优性。 |
| [^301] | [GAIA: Geometry-Adaptive Operator Learning for Forward and Inverse Problems](https://arxiv.org/abs/2607.01128) | GAIA提出了一种几何自适应积分自编码器架构，通过将域边界与内部场分布编码为几何标记并经交叉注意力调制积分变换层，首次实现了任意几何域上正问题与逆问题（输入输出可处于不同域）的统一算子学习。 |
| [^302] | [CoMet: Context and Multiplicity Decomposition for Multimodal Uncertainty Estimation](https://arxiv.org/abs/2606.32012) | 提出CoMet方法，通过将多模态大语言模型的不确定性分解为上下文特定项（捕捉给定任务或提示引起的歧义）和多样性特定项（捕捉与之相容的合理答案数量），从而在开放式多模态场景下实现不确定性估计。 |
| [^303] | [RANSAC Scoring Done Right](https://arxiv.org/abs/2606.27385) | 本文提出通过在固定内点划分下以闭式形式边缘化内点尺度、再对划分进行优化来重新设计RANSAC评分，从而消除了对阈值的依赖，仅需一次排序和线性扫描即可求得最大值，且单一闭式表达式能自适应不同真值标注水平的数据集。 |
| [^304] | [Frequency Domain Reservoir Computing](https://arxiv.org/abs/2606.24969) | 提出FRESCO——一种完全在频域运行的回声状态网络架构，通过结构化循环递归将稠密非线性状态更新的复杂度从O(N²)降低到O(N)。 |
| [^305] | [Self-Modeling Interventions can Modulate Emergent Misalignment](https://arxiv.org/abs/2606.23700) | 本研究提出将自我识别和自我报告等自我建模干预作为针对模型品格的防御手段，实验证明自我识别微调能有效逆转和预防涌现性失准，且是唯一能持续降低失准而不恶化其他指标的预防性方法。 |
| [^306] | [Do Location Encoders Capture Spatial Effects? A GeoShapley Benchmark Across Scales](https://arxiv.org/abs/2606.23453) | 该论文提出以GeoShapley博弈论解释器为工具，在三个空间尺度上对TorchSpatial框架中的十一种位置编码器进行基准测试，系统评估其嵌入能否恢复已知的空间变化系数，并揭示恢复效果随尺度与编码器架构的变化规律。 |
| [^307] | [Agentic Symbolic Search: Approximating Solutions of PDEs Beyond Hand-crafted Expressions, Meshes, and Neural Networks](https://arxiv.org/abs/2606.20467) | 提出了智能体符号搜索（ASYS）方法，让编码智能体将可微分符号程序上的进化搜索与基于梯度的拟合相结合，利用控制方程自动构建和改进偏微分方程解的显式逼近，在奇异剖面拟合和爆破问题中实现了极低的误差。 |
| [^308] | [From AGI to ASI](https://arxiv.org/abs/2606.12683) | 本报告系统刻画了从人类水平的通用人工智能（AGI）向在智力与认知能力上超越大型人类组织的超级人工智能（ASI）过渡的过程，并探讨了后AGI时代机器智能沿连续谱发展的四种潜在路径。 |
| [^309] | [From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data](https://arxiv.org/abs/2606.07537) | 该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。 |
| [^310] | [Effective Biological Representation Learning by Masking Gene Expression](https://arxiv.org/abs/2605.31562) | 该论文提出了针对多样化RNA-seq计数数据量身定制的掩码自编码自监督模型TxFM，并通过消融研究确定了关键架构配置，显著提升了转录组数据的归纳式表征学习与迁移性能。 |
| [^311] | [TriSearch: Learning to Optimize Triangulations via Bistellar Flips](https://arxiv.org/abs/2605.30220) | TriSearch是一个基于强化学习的三角剖分优化框架，其核心创新是电路支持的子三角剖分动作表示，实现了与维度无关的翻转图高效遍历，可零样本泛化到搜索空间指数级增大的更大多胞体，并在4维中发现了更多对应Calabi-Yau三维流形的三角剖分。 |
| [^312] | [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://arxiv.org/abs/2605.28207) | 提出首个将训练好的MoE模型系统性转换为全稠密语言模型的框架，通过专家评分、选择、分组拼接为稠密FFN并经知识蒸馏优化，其中新颖的多样性感知评分方法在多个模型上一致优于已有方法。 |
| [^313] | [Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems](https://arxiv.org/abs/2605.24490) | 本文提出了市场体制委员会（MRC），一种基于精确 Shapley 信用的在线信用分配机制，通过联盟绩效加权、贝叶斯自适应混合和体制依赖的权限乘数，解决了多智能体大语言模型投资组合管理系统中信用分配缺乏原则、冷启动主导和透明度不足的问题。 |
| [^314] | [Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL](https://arxiv.org/abs/2605.24001) | 该论文提出 DIDR 框架，通过积分 KL 散度最小化，将 RLHF 最优的奖励倾斜干净图像分布沿扩散轨迹传播到所有噪声水平，解决了现有一步生成器强化学习中终端奖励优化与生成动力学不匹配导致以图像保真度换取奖励提升的问题。 |
| [^315] | [Uniform-in-Time Weak Propagation-of-Chaos in Shallow Neural Networks](https://arxiv.org/abs/2605.22010) | 该论文通过利用平均场确定性 Wasserstein 梯度流动的收敛速率，而非依赖强凸性或对数 Sobolev 不等式，首次在浅层神经网络的梯度下降训练中建立了随时间一致成立的非渐近弱混沌传播结果。 |
| [^316] | [Variance Reduction for Gradients from Diffusion Teachers](https://arxiv.org/abs/2605.21489) | 提出CARV框架，将扩散教师梯度估计建模为计算感知的两阶段采样问题，结合分层蒙特卡洛估计、时间步重要性采样与分层逆CDF采样，在不改变目标函数的前提下为文本到3D和数据归因带来2至3倍的有效计算提升。 |
| [^317] | [CRAFT: Conflict-Resolved Aggregation for Federated Training](https://arxiv.org/abs/2605.21317) | 提出CRAFT聚合框架，将联邦学习中冲突客户端更新的聚合建模为带无冲突对齐约束的几何校正问题，利用闭式解避免迭代求解开销，并通过逐层自适应处理不同粒度的冲突，从而缓解特定客户端的性能退化。 |
| [^318] | [Distill to Think, Foresee to Act: Cognitive-Physical Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/2605.21139) | CoPhy 框架通过将视觉语言模型知识蒸馏进 BEV 编码器实现零推理成本的认知能力，并构建自回归 BEV 世界模型来预测候选动作的未来后果，从而突破模仿学习的行为克隆上限，实现自动驾驶的认知-物理强化学习。 |
| [^319] | [Density-Ratio Losses for Post-Hoc Learning to Defer](https://arxiv.org/abs/2605.19557) | 该论文提出用模型与专家理想分布之间的密度比来定义事后学习延后，推导出两种 DR CPE 损失用于训练延后评分器，其中联合变体在 KL 散度情形下与 Chow 规则及专家倾斜的贝叶斯后验相联系且计算便利。 |
| [^320] | [Factor Augmented High-Dimensional SGD](https://arxiv.org/abs/2605.19291) | 提出了一种完全基于流式数据的因子增强随机梯度下降方法（FSGD），并首次建立了将潜在因子估计误差纳入SGD分析的理论框架，为高维机器学习提供了可靠且可扩展的优化基础。 |
| [^321] | [AI-Mediated Communication Can Steer Collective Opinion](https://arxiv.org/abs/2605.16245) | 该论文通过实证研究与数学建模表明，大型语言模型在中介人与人之间的交流时会引入方向性偏见（如偏向支持枪支管制、反对无神论），并因此可能引导社交网络中集体舆论的形成。 |
| [^322] | [Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces](https://arxiv.org/abs/2605.14786) | 研究表明网站可通过被动JavaScript追踪器捕获LLM浏览器代理的UI操作轨迹与交互时间，以高达96%的F1分数识别其底层模型，从而带来被定向攻击的安全风险。 |
| [^323] | [Differentiable Learning of Lifted Action Schemas for Classical Planning](https://arxiv.org/abs/2605.13282) | 该论文提出了一种新颖的神经网络架构，能够以可微分的方式从状态完全可观测但动作参数不可观测的轨迹中学习经典规划的提升动作模式，为从图像序列学习规划领域奠定了重要基础。 |
| [^324] | [Examining the Impact of Data and Training Methods on Emergent and Subliminal Misalignment](https://arxiv.org/abs/2605.12798) | 该论文通过构建按“领域-任务”网格组织的数据集，系统研究了训练数据、评估数据与训练方法如何影响涌现性失对齐（EM）与阈下学习（SL）中失对齐的转移，并发现 EM 主要沿任务而非领域进行转移。 |
| [^325] | [Understanding Sample Efficiency in Predictive Coding](https://arxiv.org/abs/2605.11911) | 该论文提出“目标对齐”度量指标，从理论和实证上证明预测编码的学习样本效率优于反向传播，尤其在深层、狭窄和预训练网络中优势显著。 |
| [^326] | [Minimax Rates and Spectral Distillation for Tree Ensembles](https://arxiv.org/abs/2605.11841) | 本文从谱理论视角研究随机森林与梯度提升机，证明了随机森林回归的极小化极大最优收敛速率（由核算子特征值衰减决定），并利用主特征函数和主奇异向量开发出体积缩小数个数量级且预测性能仍具竞争力的树集成模型蒸馏压缩方法。 |
| [^327] | [Posterior Contraction Rates for Sparse Kolmogorov-Arnold Networks in Anisotropic Besov Spaces](https://arxiv.org/abs/2605.11652) | 本文证明了配备尖峰-厚板稀疏先验的稀疏贝叶斯Kolmogorov-Arnold网络在各向异性Besov空间上可达到接近极小极大的后验收缩速率，能通过超先验自适应未知的光滑度，且网络深度可保持固定。 |
| [^328] | [Affine Tracing: A New Paradigm for Probabilistic Linear Solvers](https://arxiv.org/abs/2605.10566) | 论文证明贝叶斯概率线性求解器只是非平稳仿射概率迭代方法的特例，并提出“仿射追踪”框架，能够从仿射迭代方法的标准实现中自动构建概率迭代方法，从而推动其广泛实际应用。 |
| [^329] | [MC$^2$: Monte Carlo Correction for Fast Elliptic PDE Solving](https://arxiv.org/abs/2605.09288) | MC$^2$通过学习单次神经校正来修正低预算蒙特卡洛PDE解，以少量采样达到超过1000倍样本量的求解精度，并发布了最大规模的标准化椭圆PDE基准数据集PDEZoo。 |
| [^330] | [Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation](https://arxiv.org/abs/2605.08541) | 本文证明在固定每参数Token比例下拟合LLM缩放定律会因N与D指数近似相等导致的共线性而使最小二乘问题病态化、尺度系数不可辨识，并推导出实现稳健外推所需的TPP多样性闭式阈值。 |
| [^331] | [Rethinking Black-Box On-Policy Distillation: A Rubric-based Analysis](https://arxiv.org/abs/2605.07396) | 该论文提出基于评分标准的在线策略蒸馏（ROPD），通过对比教师回答与学生采样结果自动生成评分标准作为纯文本反馈，使黑盒教师模型也能进行在线策略蒸馏，并以少9.6倍的训练样本在竞赛数学基准上超越了基于logits的白盒OPD方法。 |
| [^332] | [Diversity Curves for Graph Representation Learning](https://arxiv.org/abs/2605.06466) | 该论文提出通过在图粗化层级上跟踪结构多样性，利用新颖的等距不变量“图散度”构建可解释、高效且可跨不同尺寸图直接比较的图嵌入“多样性曲线”，并通过边收缩粗化提升了图级表示的表达能力。 |
| [^333] | [Gaussian mixture models in Hilbert spaces via kernel methods](https://arxiv.org/abs/2605.05996) | 该论文提出了一种基于核均值嵌入的希尔伯特空间数据高斯混合模型框架，用于对动态函数数据等无限维随机对象进行聚类分析，并从理论和实验上证明了其良定义性、稠密近似能力以及在多种数据几何形态上的有效性。 |
| [^334] | [Denoising data using convex relaxations](https://arxiv.org/abs/2605.02327) | 该论文提出一种基于凸松弛的去噪估计器，先通过主成分分析降维，再将观测投影到潜在流形的凸包上，并在有限样本下证明了去噪误差保证，可应用于低温电子显微镜等场景。 |
| [^335] | [COMO: Closed-Loop Optical Molecule Recognition with Minimum Risk Training](https://arxiv.org/abs/2604.23546) | 该论文提出COMO框架，首次将最小风险训练引入光学化学结构识别任务，通过在分子级进行闭环直接优化来缓解暴露偏差，从而更好地优化化学有效性和结构相似性等分子级评估指标。 |
| [^336] | [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556) | 提出了GSQ这一训练后标量量化方法，利用Gumbel-Softmax松弛联合学习逐坐标网格分配与逐组缩放因子，使简单的标量量化在低比特精度下就能达到与复杂的第二代向量量化方法相当的高准确率。 |
| [^337] | [Allocating Human Oversight in AI-Enabled Analytics](https://arxiv.org/abs/2604.12497) | 该论文提出了一种在任务难度未知情形下分配共享人类验证预算的上置信界策略，能够从人类标签与AI预测的配对数据中联合学习预测校正系数和任务难度，并具有分配遗憾理论保证，在实验中将性能提升至接近最优方案的水平。 |
| [^338] | [Differentially Private Verification of Distribution Properties](https://arxiv.org/abs/2604.10819) | 本文首次研究了差分隐私下的分布性质验证，揭示了在隐私保护约束下，交互式证明协议的性能提升取决于隐私参数和模型，并利用了隐私与可复制性及隐私放大之间的联系。 |
| [^339] | [Safe Approximate Nonlinear Model Predictive Control Without Expert Demonstrations](https://arxiv.org/abs/2603.24503) | 提出无需专家示范数据的Seq-AMPC无标签循环框架，通过安全滤波器与接受容差设计，在保证递归可行性与收敛性的同时实现安全的非线性模型预测控制神经近似。 |
| [^340] | [Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation](https://arxiv.org/abs/2603.23398) | 提出图能量匹配（GEM）框架，通过对齐输运机制来克服离散能量模型的采样陷阱与训练不稳定问题，实现可融入结构或功能约束、可组合的图生成，并缩小与离散扩散模型的保真度差距。 |
| [^341] | [Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference](https://arxiv.org/abs/2603.15507) | 提出FedBNN框架，首次将旋转感知二值化引入联邦学习，在客户端数据分布异构的条件下训练二值神经网络，实现边缘设备上的低成本高效推理。 |
| [^342] | [The Sampling Complexity of Condorcet Winner Identification in Dueling Bandits](https://arxiv.org/abs/2603.15189) | 提出一种利用完整间隔矩阵的孔多塞胜者识别算法，其样本复杂度在所有情形下达到最优（上下界匹配），并揭示了采样复杂度由定位有信息比较的成本所刻画的权衡结构。 |
| [^343] | [Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](https://arxiv.org/abs/2603.12751) | 提出了一种自监督的“展示而非告知”方法，利用人类演示视频本身自动构建数据集并训练定制物体检测器，无需语言描述和提示工程即可快速检测新物体。 |
| [^344] | [RECAP: Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics](https://arxiv.org/abs/2603.06639) | 提出了一种名为RECAP的仿生学习策略，将未经训练的储备池动力学与自组织赫布原型读出相结合，仅通过局部可塑性规则即可实现无需反向传播的稳健图像分类。 |
| [^345] | [Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337) | 该论文揭示了流匹配训练中由中间分布协方差病态条件导致的几何优化瓶颈——高方差方向拟合快而低方差方向进展慢，并据此提出“先预条件、再匹配”的预条件流匹配框架，通过变换目标分布来加速和稳定训练。 |
| [^346] | [Decentralized Ranking Aggregation via Gossip: Convergence and Robustness](https://arxiv.org/abs/2602.22847) | 本文针对偏好数据分布在通信网络中的去中心化场景，研究如何利用gossip算法实现排名聚合，并提供收敛性保证与对污染数据的鲁棒性。 |
| [^347] | [LoBoost: Model-Native Local Conformal Prediction for Gradient-Boosted Trees](https://arxiv.org/abs/2602.22432) | LoBoost通过重用梯度提升树集成模型自身的叶结构构建多尺度嵌套分组，实现模型原生的局部保形预测，无需辅助划分、重训练或额外数据分割即可局部估计残差分位数并适应异方差性。 |
| [^348] | [On Compatible Generalization Bounds of Deep Residual Networks From a Dynamical System Perspective](https://arxiv.org/abs/2602.20921) | 本文从动力系统视角出发，通过精细的Rademacher复杂度收缩不等式与分量假设集等价性分析，为离散和连续时间深度残差网络建立了同时达到最优O(1/√S)样本率且与层数无关的相容一致泛化界。 |
| [^349] | [Cold-Start Personalization via Bayesian Adaptive Questioning](https://arxiv.org/abs/2602.15012) | 提出CAPE方法，先离线从完整用户画像中学习偏好相关性的结构化世界模型，再在线通过免训练的贝叶斯推断选择信息量最大的问题并预测完整偏好画像，即使使用简单的线性回归信念模型也能大幅超越端到端强化学习。 |
| [^350] | [FUTON: Fourier Tensor Networks for Implicit Neural Representations](https://arxiv.org/abs/2602.13414) | 提出FUTON，一种将信号建模为广义傅里叶级数、并用低秩张量分解参数化傅里叶系数的隐式神经表示方法，它结合傅里叶与低秩两种归纳偏置，实现了精确的误差分解理论保证。 |
| [^351] | [Is Online Linear Optimization Sufficient for Strategic Robustness?](https://arxiv.org/abs/2602.12253) | 本文证明次线性的线性化后悔值已足以保证竞价算法的策略鲁棒性，并通过简单的黑盒归约将任意在线线性优化算法转化为策略鲁棒的竞价算法。 |
| [^352] | [Aligning Forest and Trees in Images & Long Captions for Visually Grounded Understanding](https://arxiv.org/abs/2602.02977) | 提出CAFT模型，通过“由细到粗”的层次化学习同时实现局部文本-区域对齐与全局图像-文本对齐，使视觉-语言模型能够忠实理解长而细节丰富的图像描述。 |
| [^353] | [FiLoRA: Focus-and-Ignore LoRA for Controllable Feature Reliance](https://arxiv.org/abs/2602.02060) | FiLoRA通过指令条件化的低秩门控机制，实现了在不改变任务目标的前提下，用自然语言指令直接控制多模态模型对特定特征路径的依赖，从而缓解捷径和虚假相关行为。 |
| [^354] | [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343) | 该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。 |
| [^355] | [Entropic Risk-Aware Monte Carlo Tree Search](https://arxiv.org/abs/2601.17667) | 提出了一种可证明正确的熵风险感知蒙特卡洛树搜索算法，该算法在根节点处的经验熵风险度量收敛于最优值，并具有多项式遗憾集中性的理论保证，能够在风险中性与风险规避行为之间实现有效权衡。 |
| [^356] | [Beyond local reconstruction: Geometry-preserving output alignment for binary LLM quantization](https://arxiv.org/abs/2512.21651) | 该论文揭示了1比特量化中朴素输出驱动方法失效的两个根本原因——跨层误差累积与各向异性失真，并提出了一种保持几何结构的输出对齐方法来改进二值LLM量化。 |
| [^357] | [Machine Learning to Predict Digital Frustration from Clickstream Data](https://arxiv.org/abs/2512.20438) | 本研究利用真实电商网站的点击流数据，通过LSTM和XGBoost等机器学习模型有效预测用户数字挫败感，其中LSTM表现最佳，且仅凭前20至30次交互即可实现可靠预测。 |
| [^358] | [DiffeoMorph: Learning Distributed 3D Morphogenesis with Rotation-Invariant Spectral Shape Matching](https://arxiv.org/abs/2512.17129) | 提出 DiffeoMorph 框架，利用 SE(3) 等变图神经网络与基于三维 Zernike 多项式的旋转不变谱形状匹配损失，使分布式智能体仅通过局部交互即可协同变形为目标三维形状。 |
| [^359] | [Automatic regularization for Kronecker product MMSE filters](https://arxiv.org/abs/2512.14932) | 本文提出一种基于近似留一交叉验证度量的方法，可自动确定基于Kronecker乘积表示的低秩MMSE滤波器的正则化参数，在系统辨识和天线波束成形任务中显著优于常用方法。 |
| [^360] | [Action-Guided Reachability for Deterministic World Models of Vision-Based Systems](https://arxiv.org/abs/2512.08991) | 提出无潜变量的确定性世界模型（DWM）来替代闭环中的相机，并结合动作引导的划分-验证方案，实现了对端到端视觉控制器闭环行为可靠且紧凑的可达性验证。 |
| [^361] | [The SuperActivator Mechanism: Transformers Concentrate Reliable Concept Signals in the Tail](https://arxiv.org/abs/2512.05038) | 该论文发现并理论刻画了 Transformer 中的“超级激活器机制”：概念对齐的注意力会放大激活差距，将最可靠的概念证据集中到少数高激活 token 的尾部上，从而提升基于概念向量的模型可解释性的可靠性。 |
| [^362] | [Generative Online Reinforcement Learning](https://arxiv.org/abs/2512.02581) | GoRL通过“优化与生成分离”的结构性原则——在易处理的潜空间中进行策略优化，而由条件生成解码器负责动作合成——实现了在在线强化学习中从头训练既稳定又具强表达能力的生成式策略。 |
| [^363] | [On the Origin of Algorithmic Progress in AI: Evidence from Language Model Pre-Training](https://arxiv.org/abs/2511.21622) | 该论文通过消融与缩放实验发现，2012–2023年间AI预训练约22,000倍的算法效率提升无法由小算力下的各项具体创新解释，而主要源于Transformer等算法随规模增大而增强的效率优势。 |
| [^364] | [EEG-X: Toward Device-Agnostic and Noise-Robust Foundation Model for EEG](https://arxiv.org/abs/2511.08861) | EEG-X是一个设备无关且抗噪声的脑电基础模型，通过基于位置的通道嵌入实现跨异构设备与电极布局的迁移，并通过噪声感知的掩码重建策略与基于字典的卷积变换显著提升对噪声的鲁棒性。 |
| [^365] | [Evolutionary Computation: A Paradigm of Natural Generative AI and Creative Synthesis](https://arxiv.org/abs/2510.08590) | 本文提出演化计算（EC）可作为“自然生成式AI”（NGAI）的新范式，其随机变异与自然选择机制能够突破传统统计学习的模仿局限，实现真正的创造性合成与分布外的演化跃迁。 |
| [^366] | [Benchmarking Stochastic Interpolants for Modeling Physical Systems](https://arxiv.org/abs/2509.26282) | 该论文针对流体和气候系统，将基于随机插值的生成模型与传统高斯去噪框架进行基准对比，发现随机插值在自回归预测任务的逐点精度、谱保真度和概率技巧方面能够匹敌或超越高斯去噪基线。 |
| [^367] | [BALF: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Model Compression](https://arxiv.org/abs/2509.25136) | 提出BALF框架，将激活感知低秩分解从线性层扩展到标准卷积和分组卷积并支持秩亏激活，同时基于输出失真在显式FLOP或参数预算下通过拉格朗日松弛自动分配各层秩，实现端到端免微调的模型压缩。 |
| [^368] | [OptiMind: Improving Small Language Models for Optimization Modeling through Formulation Guidance and Multi-Turn Learning](https://arxiv.org/abs/2509.22979) | OptiMind通过将常见建模错误提炼为类别特定提示、并利用求解器反馈进行多轮强化学习，有效提升了小语言模型在优化建模任务中的表现。 |
| [^369] | [Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](https://arxiv.org/abs/2509.16915) | 本文提出了一种适用于欧几里得若尔当代数输出的通用高斯差分隐私机制，并结合乘性权重更新方法设计了求解对称锥规划的隐私保护算法，从而为半定规划提供了差分隐私算法，解决了一个重要的开放问题。 |
| [^370] | [Low-degree lower bounds via almost orthonormal bases](https://arxiv.org/abs/2509.09353) | 本文提出了一种基于近似正交多项式基的更直接证明策略，用于在具有植入结构的随机图模型中建立低阶多项式下界，克服了传统规范正交多项式族方法在估计任务和复杂检验问题中的局限性。 |
| [^371] | [REVELIO: Universal Multimodal Task Load Estimation for Cross-Domain Generalization](https://arxiv.org/abs/2509.01642) | 该论文提出了REVELIO统一框架、数据集和评估协议，通过在实验室工作记忆任务、驾驶模拟器和游戏等多领域采集的96个会话多模态生理信号，实现跨领域任务负荷估计的鲁棒泛化，多模态融合AUC最高达0.87，显著优于单一信号输入。 |
| [^372] | [Deep Double Q-learning](https://arxiv.org/abs/2507.00275) | 本文提出深度双重Q学习（DDQL），通过双重Q学习目标显式训练两个Q函数实现估计器解耦，并借助较低回放比率、更长目标网络更新间隔和共享层等技术稳定训练，在57个Atari游戏上总体性能超越Double DQN。 |
| [^373] | [Iterative Orthogonalization Scaling Laws](https://arxiv.org/abs/2505.04005) | 本文从理论和实验上证明了随机矩阵奇异值随规模增大而缩小的现象，揭示了Muon优化器的迭代正交化过程在大规模场景下可能存在的问题，但未给出解决方案。 |
| [^374] | [A Computational Model of Teacher-Student Co-Adaptation](https://arxiv.org/abs/2505.02853) | 本文提出了一个贝叶斯计算框架来建模部分可观测条件下师生之间的双向协同适应，通过模拟比较五种交互模式，发现学生主动提问与教师主动延续能互补地解决信息缺口，突破了传统单向教学模型的局限。 |
| [^375] | [Minority Reports: Balancing Cost and Quality in Ground Truth Data Annotation](https://arxiv.org/abs/2504.09341) | 该论文提出了一个通过识别与多数投票标签不一致的“少数派报告”来主动剪枝冗余重复标注的理论框架与滚动策略，可在保留99.9%以上参考标签的同时将重复标注成本降低超过35%。 |
| [^376] | [Cross-Modality Investigation on WESAD Stress Classification](https://arxiv.org/abs/2502.18733) | 本研究基于WESAD数据集开发单模态Transformer模型，利用ECG、EDA、EMG等多种生理信号进行压力检测，实现了99.73%至99.95%的最先进分类性能。 |
| [^377] | [Do we really need the Rademacher complexities?](https://arxiv.org/abs/2502.15118) | 该论文证明只需利用函数类协方差结构的粗略信息，即可在重尾情形下推导出与高斯情形最先进速率相匹配的样本复杂度上界，从而表明学习问题无需依赖难以控制的局部化Rademacher复杂度。 |
| [^378] | [The Star Geometry of Critic-Based Regularizer Learning](https://arxiv.org/abs/2408.16852) | 该论文开创性地研究了在星形体的规范函数这一正则化器族上优化基于评论器的损失函数，为学习到的任务相关正则化器的结构及其与测量数据和真实数据分布之间的几何联系提供了首个理论视角。 |
| [^379] | [Nash Incentive-compatible Online Mechanism Learning via Weakly Differentially Private Online Learning](https://arxiv.org/abs/2407.04898) | 本文提出一种在弱差分隐私在线学习算法的推荐与惩罚不真实行为的承诺机制之间随机选择的方法，解决了多轮在线机制设计中面对非短视参与者时保持全流程激励相容的难题，并在对抗环境下实现了次线性遗憾度。 |
| [^380] | [Robustness Verifcation in Neural Networks](https://arxiv.org/abs/2403.13441) | 本文研究了神经网络中的鲁棒性验证问题，提出了一个理论框架来解决安全性和效率性问题，并分析了它们的计算复杂性。 |
| [^381] | [Bandit Profit-maximization for Targeted Marketing](https://arxiv.org/abs/2403.01361) | 该论文研究了针对目标营销的强盗利润最大化问题，并提出了在敌对强盗情境下的近乎最优算法。 |
| [^382] | [Differential Diagnosis of Frontotemporal Dementia and Alzheimer's Disease using Generative Adversarial Network](https://arxiv.org/abs/2109.05627) | 本研究提出了一种基于生成对抗网络的新框架，利用磁共振图像中粗到细结构尺度的体积特征，实现了对额颞叶痴呆、阿尔茨海默病和正常对照的高精度鉴别诊断。 |

# 详细

[^1]: TorchCurves：可微分参数曲线

    TorchCurves: Differentiable Parametric Curves

    [https://arxiv.org/abs/2611.14516](https://arxiv.org/abs/2611.14516)

    TorchCurves是一个将B样条和勒让德多项式参数曲线作为可学习基元的可微分PyTorch库，支持端到端优化，可广泛应用于连续特征嵌入、KAN网络和函数回归等任务。

    

    我们提出了TorchCurves，这是一个将向量值参数曲线作为可学习基元的PyTorch库。其核心操作将批量标量输入映射到以B样条或勒让德多项式基表示的曲线上的点，并支持对输入和系数进行微分。曲线系数可以与周围模型联合优化，或由另一个网络生成。该库提供了可训练模块、可复用的基函数模块以及函数式接口，支持从连续特征嵌入到Kolmogorov-Arnold网络（KAN）和函数回归等多种应用。其实现将经典曲线求值算法与向量化PyTorch操作相结合，在无需自定义硬件内核的情况下强调代码的可读性和可维护性。计时实验在不同问题规模下，将曲线求值与NumPy和SciPy进行了对比，并将KAN训练与EfficientKAN进行了对比。结果表明……（摘要原文截断）

    arXiv:2611.14516v1 Announce Type: new  Abstract: We present TorchCurves, a PyTorch library that treats vector-valued parametric curves as learnable primitives. Its central operation maps batches of scalar inputs to points on curves represented in B-spline or Legendre polynomial bases, with differentiation through both inputs and coefficients. Curve coefficients can be optimized jointly with the surrounding model or produced by another network. The library provides trainable modules, reusable basis modules, and a functional interface, supporting applications ranging from continuous feature embeddings to Kolmogorov-Arnold networks and functional regression. Its implementation combines classical evaluation algorithms with vectorized PyTorch operations, emphasizing readability and maintainability without custom hardware kernels. Timing experiments compare curve evaluation against NumPy and SciPy and KAN training against EfficientKAN across a range of problem sizes. The results illustrate w
    
[^2]: AutoScaffold：面向语言模型强化学习的AI引导式脚手架

    AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning of Language Models

    [https://arxiv.org/abs/2611.14515](https://arxiv.org/abs/2611.14515)

    提出了AutoScaffold框架，由冻结的教师模型自动将学生的训练经验转化为自适应脚手架（任务类共享策略与困难任务专用提示），在无需人工干预的情况下提升语言模型可验证奖励强化学习的效果。

    

    可验证奖励强化学习（RLVR）能产生丰富的训练经验，但结果奖励将每次尝试简化为单一数字，使其中大部分信息未被利用。训练时的脚手架，例如在训练期间添加到提示中的策略或提示信息，提供了一种将这些信息转化为指导的自然方式，但其设计与调整在很大程度上依赖人工，且需针对特定领域定制。我们提出AutoScaffold，一个统一的“AI服务AI”框架，它将学生的训练经验转化为自适应脚手架，仅需极少的领域特定信息，且训练过程中无需人工干预。一个冻结的教师模型定期检查这些经验、诊断瓶颈，并以两种形式设计脚手架：（i）一类任务共享的策略，以及（ii）针对个别困难任务的提示，并随着学生的学习动态调整支持的内容和数量。这些脚手架在评估时被移除。

    arXiv:2611.14515v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) produces rich training experience, yet outcome rewards reduce each attempt to a single number and leave most of this information unused. Training-time scaffolds, such as strategies or hints added to the prompt during training, offer a natural way to turn this information into guidance, but their design and adaptation are largely left to humans and tailored to specific domains. We introduce AutoScaffold, a unified AI-for-AI framework that turns a student's training experience into adaptive scaffolds, with minimal domain-specific information and no human intervention during training. A frozen teacher periodically examines this experience, diagnoses bottlenecks, and designs scaffolds in two forms: (i) strategies shared across a class of tasks and (ii) hints for individual hard tasks, adjusting the content and amount of support as the student learns. The scaffolds are removed at evaluat
    
[^3]: 对比发现：基于竞争性解释的开放式科学发现

    Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations

    [https://arxiv.org/abs/2611.14514](https://arxiv.org/abs/2611.14514)

    该论文提出“对比发现”框架，用互斥且穷尽的竞争性解释集合取代单一假设检验，通过证据对信念在集合内的重新分配来评估实验、确定被支持的替代解释，从而改进开放式科学发现。

    

    大语言模型智能体正被越来越多地用于开放式、数据驱动的科学发现，然而大多数智能体以逐点方式评估假设，仅将经验性断言 $h$ 与其补集 $\neg h$ 进行对照检验。当证据否定了 $h$ 时，智能体只能得知某一解释不太可能成立，却无法得知证据究竟支持哪个替代解释。为了找回这一替代解释，我们提出了“对比发现”，它用对比集取代单一假设——对比集是一个科学问题与一组互斥且穷尽所有可能的竞争性解释的配对。每个实验根据其证据如何在集合内重新分配信念来进行评分，当信念集中于某一个解释时，该科学问题即被视为解决。由于对比集仅改变了智能体提出和评分的单位，现有智能体可以利用它们来重新评估已收集的证据，或引导其搜索。在六个领域中，89%的逐点候选……（摘要原文在此处截断）

    arXiv:2611.14514v1 Announce Type: new  Abstract: LLM agents are increasingly used for open-ended, data-driven scientific discovery, yet most of them evaluate hypotheses pointwise, testing an empirical claim $h$ only against its complement $\neg h$. When evidence rejects $h$, the agent learns that one explanation is unlikely but not which alternative the evidence supports. To recover that alternative, we introduce Contrastive Discovery, which replaces the single hypothesis with a contrast set, a scientific question paired with mutually exclusive and collectively exhaustive competing explanations. Each experiment is scored by how its evidence redistributes belief across the set, and the question is resolved when belief concentrates on one explanation. Because contrast sets change only the unit an agent proposes and scores, an existing agent can use them to re-evaluate evidence it has already collected or to guide its search. Across six domains, $89\%$ of the pointwise candidates with the
    
[^4]: Pyrite：构建与基准测试蛋白质-配体构象排序智能体

    Pyrite: Building and Benchmarking Protein-Ligand Pose Ranking Agents

    [https://arxiv.org/abs/2611.14511](https://arxiv.org/abs/2611.14511)

    提出了包含1500个蛋白质-配体构象对的成对构象排序基准Pyrite，发现专家化学家的排序表现接近随机水平，并开发了两个基于大语言模型的智能体（PyriteReasoner和PyriteFeaturizer）来实现自动化构象排序。

    

    预测配体如何与靶标结合是小分子药物发现的核心。现代共折叠模型能够生成多样化的候选构象，但可靠地对这些候选进行排序仍然具有挑战性。我们提出了Pyrite，一个成对构象排序基准，包含来自多种蛋白质-配体复合物的1,500个构象对。配备3D可视化工具和基于物理的评分的专家化学家的表现接近随机水平，这凸显了解释现有化学信号的困难。为了研究自动化排序，我们引入了PyriteRep，一种对分子几何结构、蛋白质-配体相互作用和物理化学评分的结构化表示，并开发了两个基于LLM的智能体：PyriteReasoner，通过基于文本的推理直接比较构象；PyriteFeaturizer，生成可执行代码来构建用于逻辑回归的特征，并可选择性地整合来自LLM子调用派生的特征。这些智能体……

    arXiv:2611.14511v1 Announce Type: new  Abstract: Predicting how a ligand binds to its target is central to small-molecule drug discovery. Modern cofolding models can generate diverse candidate poses, but reliably ranking these candidates remains challenging. We introduce Pyrite, a pairwise pose-ranking benchmark comprising 1,500 pose pairs across diverse protein-ligand complexes. Expert chemists equipped with 3D visualization tools and physics-based scores perform near chance, highlighting the difficulty of interpreting available chemical signals. To investigate automated ranking, we introduce PyriteRep, a structured representation of molecular geometry, protein-ligand interactions, and physicochemical scores, and develop two LLM-based agents: PyriteReasoner, which directly compares poses through text-based reasoning, and PyriteFeaturizer, which generates executable code to construct features for logistic regression, optionally incorporating features derived from LLM subcalls. These ag
    
[^5]: 真实的神经连线何时胜出？针对未测量的生理参数检验连接组结构-功能主张

    When Does the Real Wiring Win? Testing connectome structure-function claims against unmeasured physiology

    [https://arxiv.org/abs/2611.14508](https://arxiv.org/abs/2611.14508)

    该论文指出连接组结构-功能比较可能混淆“连线本身”与“模型调参偏差”，提出区分三种生理参数调节方案（P1/P2/P3），并用合成真值、安慰剂零假设和三个果蝇回路模型检验哪些结构-功能结论经得起这些偏差的校正。

    

    连接组约束模型常被用来论证测得的神经连线图优于打乱或重新连线后的连线，从而得出“结构塑造功能”的结论。然而这类比较可能将连线本身与模型的调参方式相混淆：连接组只规定了谁与谁相连，并不规定模型所需的兴奋性、增益或可塑性速率等参数，建模者往往反复调节这些参数直到真实连线模型能够正常工作。我们区分了三种待估计对象：在真实连线上调节生理参数并迁移到零假设连线的P1；每条连线各自独立调参的P2；以及限定在与独立数据相符的生理参数范围内的P3。这些偏差在机器学习中已被认识（调参不足的基线、继承权重、赢家诅咒）；我们借助合成真值数据、安慰剂零假设以及三个果蝇神经回路模型，量化了这些偏差对生物学结论的影响。在我们测试的所有方案下，某些结论依然成立：分离的奖励与惩罚多巴胺通路能提升……（摘要在此处被截断）

    arXiv:2611.14508v1 Announce Type: new  Abstract: Connectome-constrained models are often used to argue that a measured wiring diagram outperforms shuffled or rewired wiring, and hence that structure shapes function. Such comparisons can confound wiring with how the models are tuned: a connectome fixes who connects to whom, not the excitability, gains or plasticity rates the model also needs, and modellers often tune these until the real-wiring model works. We separate three estimands: physiology tuned on the real wiring and transferred to nulls (P1), each wiring tuned on its own (P2), and P2 restricted to physiology consistent with independent data (P3). The biases involved are known in machine learning (under-tuned baselines, inherited weights, winner's curse); we measure what they do to biological conclusions, using synthetic ground truth, placebo nulls and three Drosophila circuit models. Some claims survive every protocol we test: segregated reward and punishment dopamine improves 
    
[^6]: Continual-ARC：测量整个职业生涯中的技能习得

    Continual-ARC: Measuring Skill Acquisition over a Working Lifetime

    [https://arxiv.org/abs/2611.14505](https://arxiv.org/abs/2611.14505)

    提出 Continual-ARC 基准，基于三个 ARC-AGI 版本构建终身任务流，使基于记忆和基于权重的持续学习系统能够在同一预算评分标准下公平比较技能习得能力。

    

    智能体框架已成为一种有前景的持续学习机制，与测试时参数化学习并行发展。然而，当前的基准测试存在若干局限性：它们通常只关注单一应用领域，每个实例只提供一次，并且假设学习者只有单一类型，因此基于记忆的系统和基于权重的系统很少在同一尺度上进行比较。我们提出了 Continual-ARC，这是一个由三个 ARC-AGI 版本构建的终身任务流，其任务无需领域知识。相同的任务会在休眠间隔后以新实例的形式再次出现，同时新任务也不断到来。在每个实例上，学习者可以付费获取演示、提交答案，或在框架内执行操作；提交错误时允许重试，但重试需付费。每个实例的得分基于所消耗的反馈预算比例。由于学习者可以在实例之间保留任何内容，基于记忆和基于权重的系统在同一标准下进行评分。

    arXiv:2611.14505v1 Announce Type: new  Abstract: Agent harnesses have emerged as a promising mechanism for continual learning, alongside test-time parametric learning. However, the current benchmarks have several limitations: they generally focus on a single applied domain, serve each instance only once, and assume a single family of learner, so memory-based and weight-based systems are rarely compared on the same scale. We propose Continual-ARC, a lifelong stream built from the three ARC-AGI editions, whose tasks need no domain knowledge. The same tasks return after dormant gaps with fresh instances, while new tasks keep arriving. On each instance, the learner may pay for demonstrations, submit answers, or act inside the harness, with retries permitted on incorrect submissions, which are paid for. Each instance is scored based on the share of this feedback budget consumed. Since a learner may keep anything between instances, memory-based and weight-based systems are scored on the same
    
[^7]: 分子相互作用基础模型中涌现的生物能力

    Emergent Biological Capabilities in a Foundation Model for Molecular Interactions

    [https://arxiv.org/abs/2611.14504](https://arxiv.org/abs/2611.14504)

    该论文提出了仅凭蛋白质-配体序列训练、无需任务监督和结构坐标的分子相互作用基础模型Bonbon，其单一表示空间在功能、残基和键三个层次涌现出零样本生物能力，并能泛化到分布外复合物。

    

    在大规模无标注语料库上训练的基础模型会发展出涌现能力，即超越其训练目标的任务。我们将这一范式扩展至多模态生物学领域，提出了Bonbon——一个在约3.2万亿个蛋白质-配体配对序列token上训练的分子相互作用基础模型。从单一的表示空间出发，Bonbon在层次化的分辨率水平上展现出三种零样本涌现能力：功能层面（作用机制）、残基层面（结合位点定位，包括正构位点与变构位点的区分）以及键层面（共价与非共价相互作用的判别）。这三种能力均可泛化至分布外的复合物。Bonbon仅通过蛋白质-配体序列进行训练，未使用任何任务特定的监督，也未使用结构坐标。我们开展了从4200万到16亿参数共六个模型规模的受控缩放实验；所评估的每一项能力均表现出急剧的跃升（原文在此处截断）。

    arXiv:2611.14504v1 Announce Type: new  Abstract: Foundation models trained on large unlabeled corpora develop emergent capabilities, tasks that go beyond the training objective. We extend this paradigm to multimodal biology with Bonbon, a foundation model for molecular interactions trained on approximately 3.2 trillion tokens of paired protein-ligand sequences. From a single representation space, Bonbon exhibits three zero-shot emergent capabilities at hierarchical levels of resolution: functional (mechanism of action), residue (binding site localization, including orthosteric/allosteric discrimination), and bond (covalent versus non-covalent engagement). All three capabilities generalize to out-of-distribution complexes. Bonbon was trained on protein-ligand sequences alone, without any task-specific supervision and without structural coordinates. We conducted controlled scaling experiments across six model scales from 42M to 1.6B parameters; every capability evaluated shows a sharp ju
    
[^8]: 智能体评估框架的比较分析：企业分析智能体中的稳定性、检测与发现

    Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents

    [https://arxiv.org/abs/2611.14493](https://arxiv.org/abs/2611.14493)

    该论文通过三阶段比较研究系统评估了三种企业分析智能体评估框架，发现基于轨迹的LLM评判框架在评估稳定性（97%配置达标）和已知风险检测（F1=0.90）方面综合表现最佳。

    

    在企业环境中部署基于LLM的分析智能体，需要能够可靠检测复杂多工具工作流中故障的评估框架。我们提出了一项针对三种评估框架的三阶段比较研究，每种框架代表一种不同的评估范式（基于轨迹的LLM评判、结合红队测试的纯文本LLM评判、以及确定性启发式方法），并在受控研究环境中使用冻结的执行轨迹应用于两个分析智能体。第一阶段通过重复测量量化了评估框架的稳定性，发现97%的配置达到了稳定性阈值，且稳定性模式与智能体本身无关。第二阶段在72个真值测试用例上测量已知风险的检测能力，其中基于轨迹的框架实现了最佳的精确率-召回率平衡（F1=0.90），而纯文本评估在召回率上持平，但产生的误报多出5倍。第三阶段评估未知风险的发现能力，

    arXiv:2611.14493v1 Announce Type: new  Abstract: Deploying LLM-based analytics agents in enterprise settings requires evaluation frameworks that can reliably detect failures across complex, multi-tool workflows. We present a three-phase comparative study of three evaluation frameworks, each representing a distinct evaluation paradigm (trace-based LLM judging, text-only LLM judging with red-teaming, and deterministic heuristics), applied to two analytics agents in a controlled research setting using frozen execution traces. Phase 1 quantifies evaluation harness stability through repeated measurements, finding that 97% of configurations meet the stability threshold and that stability patterns are agent-independent. Phase 2 measures known risk detection on 72 ground-truth test cases, where the trace-based framework achieves the best precision-recall balance (F1=0.90) while text-only evaluation matches on recall but generates 5x more false alarms. Phase 3 evaluates unknown risk discovery, 
    
[^9]: ReTrace：将使用工具的智能体蒸馏为无规划器的工作流

    ReTrace: Distilling Tool-Using Agents into Planner-Free Workflows

    [https://arxiv.org/abs/2611.14492](https://arxiv.org/abs/2611.14492)

    ReTrace通过从智能体的推理解释中提取每步背后的条件并跨运行合并为规则文档，将SOP和少量智能体运行日志蒸馏成由代码驱动的无规划器可执行工作流，在大规模流程自动化中以更低的延迟和成本实现接近LLM智能体的完成率（93.8%对96.8%）。

    

    业务流程自动化会在新的输入上反复运行相同的标准操作程序（SOP）。大型语言模型（LLM）智能体能够遵循用自然语言编写的SOP，但每次运行时都需要调用LLM来选择每一步，这在大规模应用中增加了延迟和成本。我们提出了ReTrace，它将SOP和少量记录下来的智能体运行日志转化为可复用的工作流，在该工作流中由代码而非LLM来选择下一步。智能体的推理过程——即它在每次工具调用时写下的解释——通常说明了它采取某个动作的原因，而仅有工具调用记录则无法体现这一点。ReTrace提取每个步骤背后的条件，将这些条件跨多次运行合并为一份精简的规则文档，并据此生成可执行的工作流。只有最后这一阶段依赖于具体平台，因此规则文档可以针对不同的编排平台进行编译。在我们生产平台上的79个流程中，ReTrace工作流达到93.8%的完成率，相比之下（基线）为96.8%。

    arXiv:2611.14492v1 Announce Type: new  Abstract: Business process automation runs the same Standard Operating Procedure (SOP) again and again on new inputs. Large Language Model (LLM) agents can follow an SOP written in natural language, but they call an LLM to choose every step of every run, which adds latency and cost at scale. We present ReTrace, which turns an SOP and a few logged agent runs into a reusable workflow in which code, not an LLM, selects the next step. The agent's reasoning, the explanation it writes with each tool call, often states why it took an action; the tool calls alone do not. ReTrace extracts the condition behind each step, merges these conditions across runs into a compact rule document, and generates an executable workflow from it. Only this last stage depends on the platform, so rule documents can be compiled for different orchestration platforms. On $79$ procedures from our production platform, ReTrace workflows reach $93.8\%$ completion, against $96.8\%$ 
    
[^10]: TimeSeek：面向广义时间序列分析的多任务与多领域预训练

    TimeSeek: Multi-Task and Multi-Domain Pretraining for Generalized Time Series Analytics

    [https://arxiv.org/abs/2611.14491](https://arxiv.org/abs/2611.14491)

    TimeSeek 通过领域与任务异质性下的选择性信息共享机制，实现了对多领域时间序列模式的自适应建模，并在预训练中统一多个分析任务，从而构建出性能更优的广义时间序列分析模型。

    

    时间序列分析已广泛应用于各个领域。尽管现有的时间序列基础模型（TSFM）已展现出令人鼓舞的结果，但它们仍面临两大挑战：1）大多数时间序列基础模型主要依靠堆叠 Transformer 层来增加参数规模，缺乏从多领域数据中自适应建模复杂时间序列模式的能力；2）当前的时间序列基础模型主要针对特定任务（如预测或异常检测）进行预训练，虽然少数模型支持多任务，但它们未能充分利用任务的特异性与共性，导致性能欠佳。为解决这些挑战，我们提出了 TimeSeek，一种新颖的广义时间序列分析模型，它能够从多领域数据中自适应建模时间序列模式，并在预训练中统一多个任务。在概念上，TimeSeek 受领域与任务异质性下的选择性信息共享机制指导。

    arXiv:2611.14491v1 Announce Type: new  Abstract: Time series analytics has been widely applied across various domains. While existing time series foundation models (TSFMs) have demonstrated promising results, they still face two major challenges: 1) Most TSFMs mainly rely on stacking Transformer layers to increase parameter size, lacking the ability to adaptively model complex time series patterns from multi-domain data. 2) Current TSFMs are primarily pre-trained for specific tasks, e.g., forecasting or anomaly detection, and a few models support multiple tasks, but they fail to fully leverage task specificity and commonality, resulting in suboptimal performance. To address these challenges, we propose TimeSeek, a novel generalized time series analytics model that adaptively models time series patterns from multi-domain data and unifies multiple tasks in pre-training. Conceptually, TimeSeek is guided by selective information sharing under domain and task heterogeneity. From the perspec
    
[^11]: C²-Routing：面向高效视觉语言推理的互补裁剪与压缩

    C$^2$-Routing: Complementary Cropping and Compression for Efficient Vision-Language Inference

    [https://arxiv.org/abs/2611.14469](https://arxiv.org/abs/2611.14469)

    提出C²-Routing方法，利用裁剪（突出小目标与局部细节）与压缩（保留全局上下文）的互补性，根据图像和问题自适应路由选择缩减策略，并配合安全门机制，在降低视觉语言模型计算开销的同时保留回答问题的关键信息。

    

    视觉问答（VQA）给视觉语言模型（VLM）带来了巨大的计算开销，这促使人们研究模型无关的输入缩减方法，这类方法可以方便地应用于各种视觉语言模型，包括黑盒设置。简单地强制设定视觉token数量上限虽然可以减少计算量，但可能会丢弃回答问题所需的信息。单一的缩减策略也难以适应不同的场景。我们的关键观察是：裁剪和压缩是互补的——裁剪能够突出小目标和局部细节，而整图压缩则能更好地保留大目标或分散性证据的上下文。基于这一观察，我们提出C²-Routing，根据输入图像和问题自适应地选择裁剪或压缩策略。由于这两种策略仍可能丢弃关键信息，我们进一步设计了一个安全门机制，利用预测的联合正确性分布来保障回答质量。

    arXiv:2611.14469v1 Announce Type: cross  Abstract: Visual Question Answering (VQA) imposes substantial computational overhead on vision-language models (VLMs), motivating model-agnostic input reduction methods that can be readily applied across diverse VLMs, including in black-box settings. Simply enforcing a visual-token cap reduces computation, but it can discard information needed to answer the question. A single reduction strategy also struggles to accommodate different scenarios. Our key observation is that cropping and compression are complementary: cropping highlights small targets and local details, whereas whole-image compression better preserves context for large targets or distributed evidence. Based on this observation, we propose C$^2$-Routing, which adaptively selects cropping or compression according to the input image and question. Since both strategies may still discard key information, we further design a safety gate that uses the predicted joint correctness distribut
    
[^12]: 异质SIS反应扩散模型的贝叶斯反问题与流行病阈值不确定性

    Bayesian inverse problems and epidemic-threshold uncertainty for a heterogeneous SIS reaction-diffusion model

    [https://arxiv.org/abs/2611.14468](https://arxiv.org/abs/2611.14468)

    该论文证明了异质SIS反应扩散模型的主导流行病动态仅由传播率与恢复率之差及感染者扩散系数共同决定，并据此建立了持续性推断的贝叶斯适定性，表明早期感染观测数据即使在各单独速率不确定时也能有效推断流行病阈值。

    

    我们研究了基于具有标准发生率的异质易感–感染–易感（SIS）反应扩散模型早期观测数据，对疾病持续性进行贝叶斯推断。我们证明，主导的感染率动态仅通过传播率与恢复率之差以及感染者扩散系数这两个量依赖于未知系数。同样的这一对参数决定了主特征值，而主特征值的符号决定持续性，因此即使各个单独的速率仍不确定，早期感染率数据也能为流行病阈值提供信息。我们证明了SIS系统的全局弱适定性与定量参数稳定性，并利用由此得到的观测映射估计来建立贝叶斯适定性。对于大小为 ε 的初始感染，我们推导出一个一致的二阶展开式，刻画了各个单独的速率如何进入感染率动态。发生率在主导阶显式地依赖于传播率。当噪声标准（差……摘要在此处被截断）

    arXiv:2611.14468v1 Announce Type: cross  Abstract: We study Bayesian inference of persistence from early observations of a heterogeneous susceptible--infected--susceptible (SIS) reaction--diffusion model with standard incidence. We show that the leading prevalence dynamics depend on the unknown coefficients only through the difference between the transmission and recovery rates and the infected diffusivity. The same pair determines the principal eigenvalue whose sign governs persistence, so early prevalence can inform the threshold even when the individual rates remain uncertain. We prove global weak well-posedness and quantitative parameter stability for the SIS system and use the resulting observation-map estimates to establish Bayesian well-posedness. For an initial infection of size $\varepsilon$, we derive a uniform second-order expansion that describes how the separate rates enter prevalence. Incidence depends explicitly on transmission at leading order. When the noise standard d
    
[^13]: 后训练中策略优化的理论：覆盖坍缩与参考重置

    A Theory of Policy Optimization in Post-Training: Coverage Collapse and Reference Resets

    [https://arxiv.org/abs/2611.14467](https://arxiv.org/abs/2611.14467)

    该论文为语言模型后训练中的策略优化建立了理论框架，证明REINFORCE类算法（PPO和GRPO的基础）即使奖励提升也会发生“覆盖坍缩”、使最优响应概率趋近于零并导致优化停滞，而通过重置KL参考等手段可实现由基础模型响应覆盖度所约束的保覆盖近优学习。

    

    策略优化是语言模型后训练的核心，然而其原则性的理论指导仍然有限。我们重新审视其理论基础，旨在识别一类能够学习到近似最优策略的算法，其所需的总样本数和梯度步数由基础模型的响应覆盖度决定——即基础模型生成最优响应的容易程度。我们称这类算法为“保覆盖的”。这一保证刻画了通过从基础模型采样来放大其可及能力的目标。(1) 覆盖坍缩。作为PPO和GRPO基础的REINFORCE算法，即使在奖励提升的同时，也可能将最优响应的概率驱使到接近于零。这种覆盖坍缩会使优化陷入停滞，即便初始覆盖度良好，收敛也可能需要指数级多的步数。即使使用精确梯度，这种失败也会出现，并且在加入对基础模型的KL正则化后依然存在。(2) 参考重置。我们证明，通过反复重置KL参考……（摘要在此处被截断）

    arXiv:2611.14467v1 Announce Type: new  Abstract: Policy optimization is central to language model post-training, yet principled guidance remains limited. We revisit its theoretical foundations to identify algorithms that learn near-optimal policies using total samples and gradient steps governed by the base model's response coverage, a measure of how readily it generates optimal responses. We call such algorithms coverage-preserving. This guarantee captures the goal of amplifying capabilities accessible through sampling from the base model.   (1) Coverage collapse. REINFORCE, the basis for PPO and GRPO, can drive the probability of optimal responses to near zero, even as reward improves. This coverage collapse stalls optimization, and convergence can require exponentially many steps despite good initial coverage. The failure arises even with exact gradients and persists under KL regularization to the base model.   (2) Reference resets. We show that repeatedly resetting the KL reference
    
[^14]: 交叉验证方差理论

    A Theory of Cross-Validation Variance

    [https://arxiv.org/abs/2611.14463](https://arxiv.org/abs/2611.14463)

    该论文证明了交叉验证估计方差的精确有限样本恒等式，将方差表示为每个观测值在训练与评估两种角色中权重的函数，从而为应选择何种数据划分方案提供了理论指导。

    

    交叉验证通过反复将数据划分为训练集和验证集来估计预测性能，在训练集上拟合模型并在验证集上进行评估。每个观测值以两种角色影响估计结果：当它被留出时通过其自身的评估产生影响，当它作为训练数据时通过其帮助拟合的模型产生影响。随机划分决定了它在每种角色中的权重。基于这一表示，我们证明了交叉验证估计方差的精确有限样本恒等式。对于具有相同训练集大小的给定划分方案，方差是这些权重的函数，并附带取决于模型、度量和总体的常数。该理论涵盖了在观测值上取平均的度量（如均方误差），以及在病例-对照对上取平均的度量（如AUC）。该恒等式指明了应采用哪些划分方案。方差可分解为仅取决于训练集大小的抽样方差，以及由（摘要在此处截断）……

    arXiv:2611.14463v1 Announce Type: cross  Abstract: Cross-validation estimates predictive performance by repeatedly splitting the data into training and validation sets, fitting the model on the first and evaluating it on the second. Every observation influences the estimate in two roles, through its own evaluation when it is held out and through the models it helps fit when it trains. The random splits determine its weight in each role. From this representation, we prove an exact finite-sample identity for the variance of the cross-validated estimate. For a given splitting scheme with a common training set size, the variance is a function of these weights, with constants of the model, the metric, and the population. It covers metrics averaged over observations, such as mean squared error, and over case--control pairs, such as the AUC. The identity shows which schemes to use. The variance decomposes into sampling variance, which depends only on the training size, and variance due to the
    
[^15]: 使用表格基础模型对含能材料性质的精确预测

    Accurate predictions of properties of energetic materials with tabular foundation models

    [https://arxiv.org/abs/2611.14462](https://arxiv.org/abs/2611.14462)

    表格基础模型TabPFN无需超参数调优，即可在预测含能材料的撞击感度、分解温度和晶体密度方面超越已发表的最佳模型，且速度比XGBoost快一个数量级以上。

    

    我们使用TabPFN——一种近期开发的表格基础模型——来预测含能材料的三种性质：撞击感度、分解温度和晶体密度。在无需任何超参数调优的情况下，TabPFN在所有三种性质上都优于已发表的最佳模型。它还大幅超越了一系列成熟的机器学习模型，包括经过调优的梯度提升（XGBoost）、随机森林和高斯过程。无论模型使用手工设计的分子描述符，还是使用来自Chemprop消息传递神经网络学习到的表示，这一结论均成立。特别是，对于所有三种性质，将TabPFN应用于Chemprop自身的嵌入表示，其预测精度甚至高于Chemprop本身的预测。TabPFN也是我们所考虑的模型中最快的模型之一，比XGBoost快一个数量级以上。由于TabPFN预测的是目标的完整分布，而不仅仅是单一的（数值估计）……

    arXiv:2611.14462v1 Announce Type: new  Abstract: We use TabPFN, a recently developed tabular foundation model, to predict three properties of energetic materials: impact sensitivity, decomposition temperature and crystal density. Without any hyperparameter tuning, TabPFN outperforms the best published models for all three properties. It also substantially outperforms a wide range of established machine-learning models, including tuned gradient boosting (XGBoost), random forests and Gaussian processes. This holds both when the models are given hand-crafted molecular descriptors and when they are given learned representations from Chemprop's message-passing neural network. In particular, for all three properties, TabPFN applied to Chemprop's own embedding is more accurate than Chemprop's predictions. TabPFN is also among the fastest models we consider, more than an order of magnitude faster than XGBoost. As TabPFN predicts the entire distribution of the target rather than just a single v
    
[^16]: UGG：统一生成式几何模型

    UGG: Unified Generative Geometry Model

    [https://arxiv.org/abs/2611.14440](https://arxiv.org/abs/2611.14440)

    该论文提出统一生成式几何模型UGG，将RGB视频生成与深度、相机位姿等几何预测统一为单一生成模型中的条件补全任务，实现4D场景理解，并能从异构数据集学习以及在推理时对多个场景假设进行搜索。

    

    理解4D场景是视觉智能中的核心挑战，要求模型能够从部分观测中联合捕捉场景的几何、外观与动态。然而，现有方法被细分为两类：用于直接几何预测的前馈模型，以及用于视觉合成的生成模型。我们提出了UGG，一个面向4D场景理解的统一框架，它联合建模RGB视频与几何表示（如深度和相机位姿）。通过在不同模态和时间上分配不同的噪声水平，UGG将视觉生成与几何预测统一表述为单一生成模型内的条件补全任务。这种灵活的表述方式使模型能够从具有不同几何与视觉标注子集的多样化异构数据集中学习。此外，我们的生成式表述支持在推理阶段对多个合理的场景假设进行搜索，并利用提取的几何证据……

    arXiv:2611.14440v1 Announce Type: cross  Abstract: Understanding 4D scenes is a core challenge in visual intelligence, requiring models to jointly capture scene geometry, appearance, and dynamics from partial observations. However, existing approaches are subdivided into feedforward models for direct geometric prediction, and generative models for visual synthesis. We introduce UGG, a unified framework for 4D scene understanding that jointly models RGB video and geometric representations such as depth and camera pose. By assigning different noise levels across modalities and time, UGG formulates visual generation and geometric prediction as conditional completion within a single generative model. This flexible formulation enables the model to learn from diverse, heterogeneous datasets with different subsets of geometric and visual annotations. Furthermore, our generative formulation enables inference-time search over multiple plausible scene hypotheses, using geometric evidence extract
    
[^17]: 一个策略应对100个家庭任务：我们的2026年BEHAVIOR挑战赛解决方案

    One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge

    [https://arxiv.org/abs/2611.14432](https://arxiv.org/abs/2611.14432)

    该团队基于2025年冠军方案π₀.₅构建单一VLA模型，仅用一个检查点完成2026年BEHAVIOR挑战赛全部100个家庭任务并获得第九名，核心改进在于延长训练时间、优化训练参数、加快训练步骤并减少梯度噪声。

    

    我们描述了在2026年BEHAVIOR挑战赛中的解决方案，该挑战赛要求在OmniGibson模拟器中完成100个长时程家庭任务。我们在最终排行榜上排名第九，是前十名中唯一的独立团队。解决方案的核心是一个单一的视觉-语言-动作（VLA）模型，仅用一个检查点即可应对全部100个任务，该模型构建于我们2025年获得第一名的解决方案之上，而后者基于π₀.₅。在与2025年相同的50个任务上，其Q值从0.260提升至0.371；在全部100个任务上得分为0.392。我们从2025年得出的主要结论是模型训练不足，因此2026年的大部分工作致力于更长时间地训练模型、优化部分训练参数、略微加快每个训练步骤的速度并减少梯度噪声：包括从50个任务向100个任务的迁移（从演示的技能标注构建语义阶段、新任务的任务和阶段嵌入从相似的旧任务热启动，以及匹配的21维动作空间）

    arXiv:2611.14432v1 Announce Type: cross  Abstract: We describe our solution for the 2026 BEHAVIOR Challenge: 100 long-horizon household tasks in the OmniGibson simulator. We placed 9th on the final leaderboard, the only independent team in the top 10. The core of the solution is a single vision-language-action (VLA) model, one checkpoint for all 100 tasks, built on our 1st-place 2025 solution, which is based on $\pi_{0.5}$. On the 50 tasks shared with 2025, its Q rises from 0.260 to 0.371; across all 100 tasks it scores 0.392. Our main conclusion from 2025 was that the model was undertrained, so most of the 2026 work went into training the model for longer, optimizing some training parameters, making each training step slightly faster and reducing the gradient noise: a migration from 50 to 100 tasks (semantic stages built from the demonstrations' skill annotations, task and stage embeddings of new tasks warm-started from similar old ones, and a 21-dimensional action space matching the 
    
[^18]: 量化面向传感器数据应用的时间序列基础模型的采样频率不变性

    Quantifying the Sampling Frequency Invariance of Time-Series Foundation Models for Sensor Data Applications

    [https://arxiv.org/abs/2611.14425](https://arxiv.org/abs/2611.14425)

    该论文首次将采样频率不变性置于时间序列基础模型评估的核心，通过压力测试4个主流模型并提出FSI、FDI、CDI三个新指标，发现模型行为取决于信息带宽损失而非采样频率数值本身。

    

    时间序列基础模型（TSFMs）在传感器应用中的使用日益增多，而在这类应用中，采样频率（SF）会因传感器、设备和运行模式（例如省电模式）的不同而变化。理想情况下，只要采样频率仍保留任务所需的关键信息带宽，TSFM就应在此类采样频率变化中保持有效。本文对4个知名的时间序列基础模型（MOMENT、Chronos-2、Moirai-MoE和TimesFM）进行了压力测试，以量化它们的采样频率不变性，并研究其行为是否追踪由此产生的信息带宽损失，而非采样频率数值本身。我们的压力测试探测了3个维度：（i）嵌入表示的偏移，（ii）预测性能的退化，以及（iii）下游性能的退化，分别通过三个新提出的指标FSI、FDI和CDI进行量化。据我们所知，这是首个将采样频率不变性置于TSFM评估核心的研究。我们的结果刻画了：（i）采样频率不变性如何取决于裕度……

    arXiv:2611.14425v1 Announce Type: cross  Abstract: Time-Series Foundation Models (TSFMs) are increasingly used in sensor applications, where sampling frequency (SF) varies across sensors, devices, and operating modes (e.g., power saving). Ideally, TSFMs must remain effective during such SF changes, provided the SF retains task-critical bandwidth. In this paper, we stress-test 4 prominent TSFMs: MOMENT, Chronos-2, Moirai-MoE, and TimesFM, to quantify their SF-invariance and investigate whether their behavior tracks the resulting loss of information bandwidth rather than the SF value itself. Our stress-test probes 3 dimensions: (i) the shift in embeddings, (ii) the degradation in forecasting performance, and (iii) the degradation in downstream performance, quantified by three novel metrics: FSI, FDI, and CDI, respectively. To our knowledge, this is the first study to place SF-invariance at the core of TSFM evaluations. Our results characterize: (i) how SF-invariance depends on the margin
    
[^19]: DnNP：神经过程中的输入不确定性去噪

    DnNP: Denoising Input Uncertainty in Neural Processes

    [https://arxiv.org/abs/2611.14424](https://arxiv.org/abs/2611.14424)

    提出去噪神经过程（DnNP），通过变分推断显式建模输入不确定性，将去噪与元学习联合训练，从而在含噪数据下显著提升预测质量与不确定性估计。

    

    神经过程（NPs）是灵活的元学习模型，能够在多种应用中进行不确定性量化。然而，标准的NP框架假设输入是无噪声的。这在许多实际场景中并不现实，因为传感器噪声或测量误差是不可避免的。这种模型假设与真实世界数据之间的差异会显著降低预测质量和不确定性估计的效果。我们提出了去噪神经过程（DnNPs），这是NP框架的一个有原则的扩展，能够显式地对输入不确定性进行建模。我们通过变分推断来训练DnNP。DnNP联合学习对受损输入进行去噪并执行元学习，从而对含噪训练数据具有鲁棒性。我们在合成数据集和真实世界数据集上展示了DnNP的有效性，与标准NP相比，在预测和不确定性量化方面均取得了一致的改进。在基于……（原文在此截断）的PM2.5空间插值任务上……

    arXiv:2611.14424v1 Announce Type: new  Abstract: Neural Processes (NPs) are flexible meta-learning models capable of uncertainty quantification across diverse applications. However, the standard NP framework assumes access to noise-free inputs. This is unrealistic in many practical settings, where sensor noise or measurement errors are unavoidable. This discrepancy between model assumptions and real-world data can significantly degrade prediction quality and uncertainty estimates. We introduce Denoising Neural Processes (DnNPs), a principled extension of the NP framework that explicitly models input uncertainty. We train DnNPs via variational inference. DnNPs jointly learn to denoise corrupted inputs and perform meta-learning, making them robust to noisy training data. We demonstrate DnNPs' effectiveness on both synthetic and real-world datasets, showing consistent improvements in prediction and uncertainty quantification compared to standard NPs. On PM2.5 spatial interpolation from re
    
[^20]: 探索音频基础模型的事后解耦以实现音乐相似度

    Exploring Post-Hoc Disentanglement of Audio Foundation Models for Music Similarity

    [https://arxiv.org/abs/2611.14415](https://arxiv.org/abs/2611.14415)

    本文提出将条件相似度网络（CSN）应用于冻结的音频基础模型预训练嵌入进行事后解耦，证明仅需少量可训练参数即可同时提升属性特定的解耦能力和整体音乐检索性能。

    

    诸如CLAP、MuQ和MERT等音频基础模型擅长捕捉一般性的音乐相似度。另一方面，它们纠缠的潜在空间限制了可解释性和细粒度的属性控制。条件相似度网络通过将骨干表示模型与学习到的掩码相结合来解决这一问题，这些掩码将嵌入空间门控为特定属性的子空间。然而，现有方法需要从零开始训练骨干网络，而不是利用预训练的基础模型。我们转而对事后解耦进行了系统性研究，将CSN调整为对冻结的预训练嵌入进行门控。我们在两个数据集上评估了在不同掩码和池化机制下添加可训练参数来适配嵌入的效果。我们的研究结果表明，最小程度的适配能力就足以使事后门控同时改善特定属性的解耦和整体检索性能，而池化、掩码和深度的选择……

    arXiv:2611.14415v1 Announce Type: cross  Abstract: Audio foundation models like CLAP, MuQ, and MERT excel at capturing general musical similarity. On the other hand, their entangled latent spaces limit interpretability and fine-grained attribute control. Conditional Similarity Networks (CSN) address this by pairing a backbone representation model with learned masks that gate the embedding space into attribute-specific subspaces. However, existing methods train the backbone from scratch, rather than leveraging pretrained foundation models. We instead present a systematic study of post-hoc disentanglement, adapting CSN to gate frozen pretrained embeddings. We evaluate adding trainable parameters to adapt the embedding under different masking and pooling regimes across two datasets. Our findings show that minimal adaptation capacity is sufficient for post-hoc gating to improve both attribute-specific disentanglement and overall retrieval performance, while pooling, masking, and depth choi
    
[^21]: 深度神经网络模型的因果概念解释

    Causal Concept Explanations for Deep Neural Models

    [https://arxiv.org/abs/2611.14407](https://arxiv.org/abs/2611.14407)

    该论文提出 CCW-Net，首次将干预性中介分析作为训练目标，使概念解释与模型输出之间建立真正的因果关系而非仅是相关性，并提供量化每个概念解释对模型输出因果贡献的评估工具。

    

    神经网络模型中基于概念的解释对于自主系统的开发、认证和推广应用至关重要。然而，现有的可解释性方法仍停留在相关性层面。由于缺乏指导和量化解释之间因果关系的方法，所提供的概念解释可能既不是所观测输出的必要条件，也不是其充分条件，从而在看似正确的同时产生误导。我们提出了因果概念包装网络（Causal Concept Wrapper Network, CCW-Net），这是首次将源自经济学、医学和流行病学等领域的干预性中介分析用作训练目标，使基于概念的解释的因果效应与模型输出保持一致。我们还进一步将中介分析用作评估工具，针对每个概念标签组合，量化任意模型输出中有多少可因果归因于每个概念解释。作为训练目标，CCW-Net 塑造模型的概念表示，使其既能……（原文摘要在此处截断）

    arXiv:2611.14407v1 Announce Type: cross  Abstract: Concept-based explanations in neural models are important for the development, certification, and adoption of autonomous systems. Yet, interpretability remains correlational. Without methods to guide and quantify causal relationships between explanations, a concept explanation can be provided without being necessary or sufficient for the observed output thereby appearing correct while misleading. We introduce Causal Concept Wrapper Network (CCW-Net), the first use of interventional mediation analysis from fields such as economics, medicine, and epidemiology as a training objective to align causal effects of concept-based explanations with a model's output. We further apply mediation analysis as an evaluation tool to quantify, per concept label combination, how much of any model's output is causally attributable to each concept explanation. As a training objective, CCW-Net shapes a model's concept representations such that they are both
    
[^22]: 发现南亚季风降水热力学的方程

    Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation

    [https://arxiv.org/abs/2611.14401](https://arxiv.org/abs/2611.14401)

    该研究提出符号蒸馏方法，将神经网络与符号回归结合，从21年再分析数据中发现了可解释的南亚季风降水热力学方程，其解释能力超过传统浮力诊断方法1.5倍，同时参数量减少约一千倍。

    

    南亚季风降水依赖于对流层低层的热力结构，但其各个要素如何组合塑造降水仍不清楚，部分原因在于这种依赖关系在垂直方向上是非局地的。我们使用符号蒸馏方法，该方法结合了神经网络与符号回归，从21年的6-8月再分析数据中发现可解释的降水方程。神经网络首先学习与降水最相关的垂直热力结构，将每个变量压缩为一个不依赖于特定垂直网格的单一可解释特征。随后，符号回归从这些特征及地表状态中推导出封闭形式的方程。我们得到的最佳方程所解释的降水方差是广泛使用的基于浮力的诊断方法的1.5倍，并在参数量减少约一千倍的情况下恢复了约85%的神经网络预测技能。与浮力诊断不同……

    arXiv:2611.14401v1 Announce Type: new  Abstract: South Asian monsoon precipitation depends on the thermodynamic structure of the lower troposphere, but how its ingredients combine to shape precipitation remains unclear, partly because this dependence is nonlocal in the vertical. We use symbolic distillation, which combines neural networks and symbolic regression, to discover interpretable precipitation equations from 21 years of June-August reanalysis data. Neural networks first learn the vertical thermodynamic structure most relevant to precipitation, condensing each variable into a single interpretable feature not tied to a particular vertical grid. Then, symbolic regression derives closed-form equations from these features and the surface state. Our best equation explains 1.5 times as much precipitation variance as a widely used buoyancy-based diagnostic and recovers about 85% of neural network skill while using about a thousand times fewer parameters. Unlike the buoyancy diagnostic
    
[^23]: REAL 损失：为投机解码优化真实期望接受长度

    REAL Loss: Optimizing Real Expected Acceptance Length for Speculative Decoding

    [https://arxiv.org/abs/2611.14399](https://arxiv.org/abs/2611.14399)

    该论文提出 REAL 损失，通过路径权重无偏估计与 Rao-Blackwellization 方差缩减，直接优化投机解码中真实的期望接受长度（EAL），解决了现有替代目标存在的协方差差距及其梯度方向反转问题。

    

    投机解码通过让草稿模型提出多个 token，再由目标模型一次性进行验证，从而加速大语言模型（LLM）的推理。加速效果由期望接受长度（EAL）决定，然而交叉熵、KL 散度和总变差等标准训练目标仅关注分布匹配或单步重叠。近期研究转而提出块级训练目标，旨在直接优化 EAL。然而，这些目标仍然是替代目标，而非 EAL 的精确表述。例如，一些方法在目标分布的续写上对后续接受进行平均，而非在被接受 token 所选择的验证分支上进行平均，由此引入了协方差差距，可能导致梯度方向反转。我们分析了这一差距，并推导出 REAL——一种真实的 EAL 训练损失，它使用路径权重实现无偏估计，并利用 Rao-Blackwellization 进行方差缩减。在受控的损失级比较中

    arXiv:2611.14399v1 Announce Type: cross  Abstract: Speculative decoding accelerates LLM inference by having a draft model propose multiple tokens for the target model to verify in one pass. The speedup is governed by the expected acceptance length (EAL), yet standard training objectives such as cross-entropy, KL divergence, and total variation focus only on distribution matching or single-step overlap. Recent studies instead propose block-level training objectives, aiming to optimize EAL directly. However, these objectives remain surrogates rather than exact formulations of EAL. For example, some average later acceptance over target-distributed continuations rather than the verification branches selected by accepted tokens, introducing a covariance gap that can reverse the gradient direction. We analyze this gap and derive REAL, a real EAL training loss that uses path weights for unbiased estimation and Rao-Blackwellization for variance reduction. In a controlled loss-level comparison 
    
[^24]: 基于间隔视角的LoRA：对灾难性遗忘的鲁棒性与适配器合并

    A Margin Perspective on LoRA: Robustness to Catastrophic Forgetting and Adapter Merging (MaLoRA)

    [https://arxiv.org/abs/2611.14397](https://arxiv.org/abs/2611.14397)

    该论文通过统一的基于间隔的理论视角，解释了LoRA对灾难性遗忘的鲁棒性以及可合并独立训练适配器这两个重要经验特性。

    

    低秩适配是大型神经网络参数高效微调的事实标准方法，能够以最少的可训练参数和较低的优化成本实现强大的任务适应能力。除了这种高效性之外，实践者还观察到两个显著的特性：*对灾难性遗忘的鲁棒性*，以及能够*将独立训练的适配器合并*为单一模型，且该模型在多个任务上均能表现出有竞争力的性能。这些特性对于持续学习和多任务学习至关重要，但目前仍缺乏深入理解。在本工作中，我们通过一个统一的*基于间隔的视角*提供了理论解释。我们在具有 $\ell_2$ 正则化的近正交任务机制下分析多类线性分类中的LoRA，并刻画了不同正则化机制下的最优LoRA适配器。我们的分析表明，在正则化参数的中间区间内，最优的

    arXiv:2611.14397v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is the de facto method for parameter-efficient fine-tuning of large neural networks, achieving strong task adaptation with minimal trainable parameters and low optimization cost. Beyond this efficiency, practitioners have observed two striking properties: \textit{robustness to catastrophic forgetting} and the ability to \textit{merge independently trained adapters} into a single model that performs competitively across multiple tasks. These properties are central to continual and multi-task learning, yet remain poorly understood. In this work, we provide a theoretical explanation through a unified \textit{margin-based perspective}. We analyze LoRA in multiclass linear classification under a near-orthogonal task regime with $\ell_2$-regularization, and characterize the optimal LoRA adapter across regularization regimes. Our analysis shows that, in an intermediate regime of regularization parameters, the optimal 
    
[^25]: 机器学习的能量视角

    An Energy-Based Perspective on Machine Learning

    [https://arxiv.org/abs/2611.14383](https://arxiv.org/abs/2611.14383)

    本教程以能量模型（EBM）为统一视角，通过能量函数、推断过程和学习方法三大核心组件，将判别模型、GAN、扩散模型、自回归模型等各类机器学习方法统一在一个共同的对比学习框架之下。

    

    能量模型（EBM）通过一个标量能量函数来定义分布，$p_\theta(x)\propto\exp(-E_\theta(x))$，通常被视为众多模型家族中的一员。本教程有着更宏大的目标：不仅介绍EBM，还将其作为一个统一的视角，把机器学习的不同方面连接起来。任何EBM都由三个选择确定：能量函数，它决定模型表示什么；推断过程，它从能量中采样或对能量进行优化；以及学习方法，它使能量拟合数据。我们首先阐述这些组件，包括采样、变分推断，以及通过最大似然、噪声对比估计、伪似然、得分匹配和对抗训练进行学习的方法，并证明这些学习方法共享同一个对比结构。随后我们展示判别模型、生成模型（如GAN、归一化流、扩散模型和自回归模型）、表示学习等均可在此框架下统一理解（原文此处截断）。

    arXiv:2611.14383v1 Announce Type: new  Abstract: Energy-based models (EBMs) define a distribution through a scalar energy, $p_\theta(x)\propto\exp(-E_\theta(x))$, and are usually presented as one model family among many. This tutorial has a broader goal: beyond introducing EBMs, it uses them as a single lens that connects the different aspects of machine learning. Any EBM is specified by three choices, the energy, which determines what the model represents; the inference procedure, which samples from or optimizes the energy; and the learning method, which fits the energy to data. We first develop these components, including sampling, variational inference, and learning by maximum likelihood, noise contrastive estimation, pseudolikelihood, score matching, and adversarial training, and show that the learning methods share one contrastive structure. We then show that discriminative models, generative models such as GANs, normalizing flows, diffusion and autoregressive models, representati
    
[^26]: 基于稀疏高斯过程评论家实现在线动作选择的不确定性感知混合离线强化学习

    Uncertainty-Aware Hybrid Offline Reinforcement Learning with Online Action Selection Enabled by Sparse Gaussian-Process Critics

    [https://arxiv.org/abs/2611.14374](https://arxiv.org/abs/2611.14374)

    本文提出用稀疏变分高斯过程评论家同时替代树集成评论家与对抗自编码器惩罚，通过最大化不确定性调整的回报估计 μ-κσ，在混合离线强化学习框架HOFLON中实现不确定性感知的在线动作选择，提升了聚合反应器开车启动等任务的安全性与可靠性。

    

    开车启动和牌号切换在很大程度上仍依赖操作员手动驱动，而基于模型的控制方法需要开发和维护动态模型，这需要大量的工程投入。混合离线学习与在线优化（HOFLON）利用一步式离线强化学习从历史转移数据中选择动作。在其原始实现中，在线目标函数将树集成评论家预测的回报与针对不熟悉状态-动作对的对抗自编码器（AAE）惩罚相结合。我们考察了该惩罚项的作用，并用稀疏变分高斯过程（sGP）评论家替代了原来的评论家和AAE。动作选择旨在最大化 μ - κσ，即对“执行一个动作后遵循历史操作实践”所得回报的不确定性调整估计。在HOFLON聚合反应器基准测试中，从HOFLON中移除AAE导致在五个评论家的15次运行中出现4次启动失败。在其已发布的权重下，

    arXiv:2611.14374v1 Announce Type: new  Abstract: Start-ups and grade changes remain largely operator-driven, while developing and maintaining dynamic models for model-based control requires substantial engineering effort. Hybrid Offline Learning and Online Optimisation (HOFLON) uses one-step offline reinforcement learning to select actions from historical transition data. In its original implementation, the online objective combines the predicted return from a tree-ensemble critic with an adversarial-autoencoder (AAE) penalty for unfamiliar state-action pairs. We examine the effect of this penalty and replace the critic and AAE with a sparse variational Gaussian-process (sGP) critic. Actions maximise $\mu - \kappa\sigma$, an uncertainty-adjusted estimate of the return from one action followed by the historical operating practice. On the HOFLON polymerisation-reactor benchmark, removing the AAE from HOFLON led to 4 failed start-ups in 15 runs over five critics. At its published weight, 
    
[^27]: 未被利用的输入：将初始状态作为循环时间序列基础模型的条件化接口

    The Unused Input: The Initial State as a Conditioning Interface for Recurrent TSFMs

    [https://arxiv.org/abs/2611.14357](https://arxiv.org/abs/2611.14357)

    该论文揭示循环时间序列基础模型中被闲置的初始状态可作为条件化接口：冻结TiRex骨干并仅训练一个小模块将输入窗口的频谱摘要写入初始状态，即可降低GiftEval上的MASE。

    

    时间序列基础模型（TSFMs）无需任务特定训练即可对未见过的序列进行预测。它们越来越多地构建于循环骨干网络之上，这类骨干网络每步计算成本恒定，使长上下文推理代价低廉，并且与Transformer不同，它们携带一个状态。循环骨干网络从两个输入计算其预测：观测到的上下文窗口和初始循环状态。在实践中，通常只利用了前者，而初始状态按惯例未被使用。我们探讨初始状态能否在循环TSFMs中被利用。以基于xLSTM的零样本预测器TiRex作为案例研究，我们冻结骨干网络，仅训练一个小型头部模块，将上下文向量写入初始状态，从而使任何准确率的变化都可归因于注入的状态而非权重调整。对输入窗口的频谱摘要只需学习一次并以不变的方式应用于每个序列，即可将GiftEval的平均MASE降低0.175%，且在每个随机种子下均……

    arXiv:2611.14357v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) forecast unseen series without task-specific training. Increasingly they are built on recurrent backbones, whose constant per-step cost makes long-context inference cheap, and which, unlike transformers, carry a state. A recurrent backbone computes its prediction from two inputs: the observed context window and the initial recurrent state. In practice only the first is utilised and the initial state is unused by convention. We ask whether the initial state can be utilised in recurrent TSFMs. Using the xLSTM-based zero-shot forecaster TiRex as a case study, we freeze the backbone and train only a small head that writes a context vector into the initial state, so that any change in accuracy is attributable to the injected state rather than to adapted weights. A spectral summary of the input window, learned once and applied unchanged to every series, lowers mean GiftEval MASE by 0.175%, with every seed 
    
[^28]: 前向值离散化：无需高阶近似的更快随机扩散采样

    Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation

    [https://arxiv.org/abs/2611.14328](https://arxiv.org/abs/2611.14328)

    该论文证明了高阶近似无法提升随机扩散采样的收敛阶数，并提出免训练的前向值离散化方法（F-DPMSolver），通过利用先前预测构建的前瞻估计，在低神经函数评估次数下实现更快的采样并提升样本质量。

    

    随机扩散采样器通常需要比确定性采样器多得多的神经函数评估次数（NFE）才能达到相当的样本质量。尽管高阶离散化能够加速确定性采样，但其对随机采样的益处尚不明确。我们建立了匹配的上界与最坏情况下的下界，表明即使使用精确的分数函数，采用高阶近似的随机采样器仍然只具有一阶收敛性。因此，提高近似阶数通常并不能改善收敛阶数。受此局限性的启发，我们提出了前向值离散化，这是一种免训练的框架，它在由先前预测构建的前瞻估计处评估数据/噪声预测模型。对于确定性采样，由此得到的 Forward DPMSolver（F-DPMSolver）通过其基于 DDIM 的变体，在低 NFE 预算下提升了样本质量，尽管仍保持一阶……

    arXiv:2611.14328v1 Announce Type: cross  Abstract: Stochastic diffusion samplers often require substantially more neural function evaluations (NFEs) than deterministic samplers to achieve comparable sample quality. Although higher-order discretization accelerates deterministic sampling, its benefits for stochastic sampling remains unclear. We establish matching upper and worst-case lower bounds showing that stochastic samplers with higher-order approximations remain first-order convergent, even with exact score functions. Thus, increasing the approximation order does not generally improve the convergence order. Motivated by this limitation, we introduce forward-value discretization, a training-free framework that evaluates the data/noise prediction model at a lookahead estimate constructed from previous predictions. For deterministic sampling, the resulting Forward DPMSolver (F-DPMSolver), improves sample quality at low NFE budgets through its DDIM-based variant, despite remaining firs
    
[^29]: 智能体何时应该记忆？面向LLM智能体的证伪门控自我进化

    When Should Agents Remember? Falsification-Gated Self-Evolution for LLM Agents

    [https://arxiv.org/abs/2611.14326](https://arxiv.org/abs/2611.14326)

    提出证伪门控自我进化（FGSE）框架，将经验教训表示为结构化假设，仅在临时状态中经受证伪、迁移与回归探测，只有通过收益与风险双重验证的行为更新才被写入智能体的持久记忆，从而显著提升LLM智能体自我进化的可靠性与安全性。

    

    LLM智能体日益运行在长时程、验证器丰富的环境中，但当前的自我改进机制往往将回合经验中总结的教训直接写入记忆，使得局部成功成为可复用且安全知识的不可靠信号。本文解决的核心问题是：何时应将来自经验的 行为更新纳入智能体的持久状态。我们提出证伪门控自我进化（FGSE），这是一个以验证器为基础的框架，将每个候选经验教训表示为结构化假设，包含明确的前提条件、行为改变、预期效果和验证器。FGSE仅在临时状态中应用该假设，在目标迁移、证伪和归档回归探测上进行测试，然后根据测得的收益和风险来提交、细化或拒绝该假设。在网页、应用、工具使用和长期记忆基准测试中，FGSE取得了强大的任务性能，包括39.8的WebArena成功率。

    arXiv:2611.14326v1 Announce Type: new  Abstract: LLM agents increasingly operate in long-horizon, verifier-rich environments, but current self-improvement mechanisms often write episode-derived lessons directly into memory, making local success an unreliable signal for reusable and safe knowledge. This paper addresses the core question of when an experience-derived behavioral update should be admitted into persistent agent state. We propose Falsification-Gated Self-Evolution (FGSE), a verifier-grounded framework that represents each candidate lesson as a structured hypothesis with an explicit precondition, behavior change, expected effect, and verifier. FGSE applies the hypothesis only in a temporary state, tests it on target-transfer, falsification, and archived-regression probes, and then commits, refines, or rejects it according to measured gain and risk. Across web, app, tool-use, and long-term memory benchmarks, FGSE achieves strong task performance, including 39.8 WebArena succes
    
[^30]: 审计裁判者：用于大语言模型推理保障的内部信号

    Auditing the Judge: Internal Signals for LLM Reasoning Assurance

    [https://arxiv.org/abs/2611.14323](https://arxiv.org/abs/2611.14323)

    该论文提出用注意力头激活训练的监督式正确性探针来审计大语言模型的自我裁判机制，证明在固定审查预算下探针比裁判自身的置信分数能发现更多推理错误。

    

    语言模型系统使用自我判断来决定哪些答案可以发布、复审或重新计算。由于干预成本高昂，一个保障分数必须在可用预算内对错误进行良好排序，而不仅仅是在完整排名上表现良好。我们将裁判者的连续判定token分数与在同一提示的注意力头激活上训练的监督式正确性探针进行比较。通过精确的恒等式将固定预算下的错误召回率与残余风险联系起来，并表明AUROC是所有预算下召回率的平均值。在GSM8K和MATH数据集上的三个开源7-12B模型中，在所有六种模型-数据集设置下，探针在5%复审预算下额外发现了1.3-22.5个百分点的错误（中位数为7.4个百分点）。在10%路由比例下，探针引导的分配在两项GSM8K实验中都比同等成本的随机分配提高了准确率，但仅在裁判者较弱时才优于裁判引导的路由。跨数据集迁移时，探针的AUROC在两个方向上相对于目标数据集内的表现均有所下降。

    arXiv:2611.14323v1 Announce Type: new  Abstract: Language-model systems use self-judgment to decide which answers to release, review, or recompute. Because intervention is costly, an assurance score must rank errors well at the available budget, not merely over the full ranking. We compare the judge's continuous verdict-token score with a supervised correctness probe trained on attention-head activations from the same prompt. Exact identities link fixed-budget error recall to residual risk and show that AUROC averages recall over all budgets. Across three open 7-12B models on GSM8K and MATH, the probe finds 1.3-22.5 additional percentage points of errors at 5% review in all six model-dataset settings (median 7.4 points). At 10% routing, probe-guided allocation improves accuracy over equal-cost random allocation in both GSM8K experiments, but beats judge-guided routing only for the weaker judge. Cross-dataset transfer lowers probe AUROC for every model in both directions relative to tar
    
[^31]: MammothDP：面向可信硬件且无超参数泄露的隐私保护提升决策树

    MammothDP: Private Boosted Decision Trees Ready for Trusted Hardware and Leakage-free Hyperparameters

    [https://arxiv.org/abs/2611.14317](https://arxiv.org/abs/2611.14317)

    该论文提出 MammothDP 系统，基于梯度提升决策树学习器 S-BDT，将可信执行环境部署、计时侧信道加固与差分隐私过滤器相结合，克服了这一组合使用所产生的意外安全挑战，并以无超参数泄露的方式实现了隐私-效用权衡良好的端到端隐私保护机器学习方案。

    

    诸如欧洲健康数据空间（EHDS）之类的数据空间要求在数据处理过程中以及研究发布时都对敏感数据进行保护。在这项工作中，我们研究了如何利用成熟的可信执行环境（TEE）与计时侧信道加固来实现这一要求，以便即使在服务器被攻破的情况下也能保护数据处理过程，并采用最先进的差分隐私（DP）概念来保护基于表格数据的研究发布。这种组合方案看似已经完善了整个图景，但实际上带来了意想不到的安全挑战，我们通过引入 MammothDP 来缓解这些挑战：这是朝着构建整体式机器学习系统迈出的重要里程碑，该系统基于近期提出的梯度提升决策树学习器 S-BDT，当在隐私过滤器中使用时可证明满足差分隐私，同时经过计时侧信道加固并部署在 TEE 上。为了实现强隐私-效用权衡，我们引入了两个新颖的组件……

    arXiv:2611.14317v1 Announce Type: cross  Abstract: Data spaces like the European Health Data Space (EHDS) require the protection of sensitive data in studies during the data's processing and the study's release alike. In this work, we investigate how to implement this mandate using established trusted execution environments (TEEs) and timing side-channel hardening to protect the processing even under a compromised server, and the state-of-the-art differential privacy (DP) notion to protect the release of studies on tabular data. This combined setup could seem to complete the picture, but in fact raises unexpected security challenges that we mitigate by introducing MammothDP: a significant milestone towards a holistic machine learning system based on the recent gradient boosted decision tree learner S-BDT that is provably DP if used within a privacy filter, timing side-channel hardened, and deployed on a TEE. To achieve a strong privacy-utility tradeoff, we introduce two novel component
    
[^32]: PSL-RGHA：采用参考梯度半空间对齐缓解梯度发散的并行拆分学习

    PSL-RGHA: Parallel Split Learning with Reference Gradient Half-Space Alignment for Mitigating Gradient Divergence

    [https://arxiv.org/abs/2611.14312](https://arxiv.org/abs/2611.14312)

    提出PSL-RGHA框架，通过参考梯度半空间对齐机制缓解并行拆分学习在异构数据分布下的训练发散问题，使资源受限设备上的多媒体应用部署更加可行。

    

    深度神经网络日益增长的复杂性阻碍了联邦学习（FL）在资源受限设备上的部署，尤其是对于设备端图像分类和语音识别等多媒体应用。并行拆分学习（PSL）通过分层模型划分将大部分计算负载转移到服务器，已成为一种有前景的解决方案，特别适用于资源受限的视觉和语音智能场景。然而，由于缺乏客户端聚合机制，PSL在异构数据分布下容易出现严重的训练发散和潜在的收敛失败。为了应对这一挑战，我们提出了PSL-RGHA，这是一个梯度对齐的PSL框架，采用参考梯度半空间对齐机制来缓解客户端设备之间的训练发散。PSL-RGHA包含两个新颖的组件：参考梯度p（原文摘要不完整，此处截断）

    arXiv:2611.14312v1 Announce Type: new  Abstract: The escalating complexity of deep neural networks hinders the deployment of federated learning (FL) on resource-constrained devices, especially for multimedia applications such as on-device image classification and speech recognition. Parallel split learning (PSL) has emerged as a promising solution by offloading a substantial portion of the computational workload to the server through layer-wise model partitioning, making it particularly suitable for resource-limited visual and speech intelligence scenarios. However, the absence of client-side aggregation renders PSL sus ceptible to severe training divergence and potential convergence failure under heterogeneous data distributions. To address this challenge, we propose PSL-RGHA, a gradient-aligned PSL framework that employs reference gradient half-space alignment mechanism to mitigate training divergence among client devices. PSL-RGHA comprises two novel components: reference gradient p
    
[^33]: 保留已获得正确答案的多模态弃答机制

    Multimodal Abstention That Preserves Acquired Correct Answers

    [https://arxiv.org/abs/2611.14311](https://arxiv.org/abs/2611.14311)

    提出ReFrain多模态弃答方法，在提升对不可回答问题正确弃答能力的同时，保证任务适配阶段获得的正确答案零损失。

    

    为经过任务适配的多模态模型添加弃答功能，可能会丢弃正确答案，包括适配过程中获得的能力。我们将弃答修复问题形式化为：在“新损失的正确答案”的预算约束下，最大化对不可回答问题的正确弃答率。我们提出ReFrain（无回归弃答），该方法结合了基于首token logits的线性探针、在留出图像集群上的阈值校准，以及一个确定性否决机制——用显示的“以上都不是”选项替换被拒绝的输出，同时保留所有其他响应。“无回归”指在零预算目标下，校准集上的正确答案损失为零。主要对比中的所有弃答方法均从同一个回答SFT检查点出发；我们分别跟踪损失的正确答案、修复的错误以及保留的SFT收益。在Qwen2.5-VL-3B上，ReFrain在零预算下完整保留了全部57个SFT获得的答案且零损失，同时将正确弃答率提升了7.30个百分点。

    arXiv:2611.14311v1 Announce Type: new  Abstract: Adding abstention to a task-adapted multimodal model can discard correct answers, including capabilities acquired during adaptation. We formulate abstention repair as maximizing correct abstention on unanswerable questions under a budget on newly lost correct answers. We propose ReFrain (Regression-Free Abstention), which combines a linear probe on first-token logits, threshold calibration on held-out image clusters, and a deterministic veto that replaces rejected outputs with the displayed "none of the above" option while preserving all other responses. "Regression-free" denotes zero loss on the calibration set at the zero-budget target. All abstention methods in our main comparison start from the same answering-SFT checkpoint; we separately track lost correct answers, repaired errors, and retained SFT gains. On Qwen2.5-VL-3B, ReFrain at zero budget retains all 57 SFT-gained answers with no losses and improves correct abstention by 7.30
    
[^34]: MMDM：衡量大规模多任务决策

    MMDM: Measuring Massive Multitask Decision Making

    [https://arxiv.org/abs/2611.14309](https://arxiv.org/abs/2611.14309)

    提出了MMDM基准——涵盖16个任务类别、5,448道高质量问题的大规模多任务决策基准，用于在无数据污染的全新上下文中评估系统-1决策模型在单选、多选和软标签决策上的能力。

    

    系统-1决策模型在单次前向传播中，基于给定状态直接对查询所指定的选项做出结构化的概率决策，而无需自回归地生成推理轨迹或文本响应。近期的评估虽涵盖多样化任务，但往往依赖现有基准，而对测试材料的事先接触（数据污染）可能掩盖模型的真实泛化能力。此外，现实世界的决策常常涉及长上下文、庞大的选项空间以及复杂的逻辑依赖，而这些挑战并未被现有基准同时充分捕捉。因此，我们提出了MMDM（大规模多任务决策），这是一个包含16个任务类别、共5,448道高质量问题的基准，它将新构建的任务与基于原始来源的重组和改写相结合，用于在全新上下文中进行单选、多选和软标签决策。该基准涵盖了诸如智能体状态跟踪、工具与技能选择等真实场景（摘要原文在此处被截断）。

    arXiv:2611.14309v1 Announce Type: cross  Abstract: System-1 Decision Models directly make structured probabilistic decisions over query-specified options given a state in a single forward pass, without autoregressively generating reasoning trajectories or textual responses. Recent evaluations cover diverse tasks but often rely on existing benchmarks, where prior exposure to test material can obscure generalization. Moreover, real-world decision making often involves long contexts, large option spaces, and complex logical dependencies, challenges that existing benchmarks do not adequately capture together. Thus, we introduce MMDM (Massive Multitask Decision Making), a benchmark of 5,448 high-quality questions across 16 task categories, combining newly constructed tasks with source-grounded recompositions and rewrites for single-select, multi-select, and soft-label decision making in fresh contexts. It covers realistic scenarios such as agent state tracking, tool and skill selection, pre
    
[^35]: 成分数据随机森林

    Compositional random forest

    [https://arxiv.org/abs/2611.14275](https://arxiv.org/abs/2611.14275)

    该论文提出了一种面向成分数据响应的随机森林框架，以对数似然增益作为分裂准则，并结合定义在不同表示空间上的分布，统一兼容了单纯形、对数比坐标和单位球面方向数据三种成分数据分析视角。

    

    成分数据在文献中主要从三个角度进行分析：直接在单纯形上分析、在对数比坐标下分析，以及作为单位球面上的方向数据进行分析。本文构建了一个适用于成分响应的随机森林框架，能够兼容上述全部三种表示方式。随机森林是决策树的集成，分裂准则决定了每棵树的生长方式。我们采用对数似然增益作为分裂准则，并考虑定义在成分数据可表示的各种空间上的分布。Dirichlet似然在文献中已有研究，但其协方差在仅相差一个标量的意义上由均值决定，且其最大似然估计没有闭式解。为了获得更灵活的准则，我们使用了两种著名的变换。首先，平方根变换将成分映射到单位球面上，而现有的方向分布需要数值估计……

    arXiv:2611.14275v1 Announce Type: cross  Abstract: Compositional data have been analysed in the literature from three main points of view: directly on the simplex, in log-ratio coordinates, and as directional data on the unit sphere. In this paper, we construct a random forest framework for compositional responses that accommodates all three. Random forests are ensembles of decision trees, and the split criterion determines how each tree grows. We use the log-likelihood gain as the split criterion and consider distributions defined on the different spaces in which compositions can be represented. The Dirichlet likelihood has been studied in the literature, but its covariance is fixed by its mean up to a single scalar, and its maximum likelihood estimator has no closed form. To obtain more flexible criteria, we use two well-known transformations. First, the square-root transformation maps compositions onto the unit sphere, where existing directional distributions require numerical estim
    
[^36]: 面向异步疫情暴发的跨区域流行病预测联邦学习方法

    Federated Learning for Cross-Regional Epidemic Forecasting with Asynchronous Outbreaks

    [https://arxiv.org/abs/2611.14272](https://arxiv.org/abs/2611.14272)

    本文提出并验证了一个基于联邦学习的跨区域COVID-19预测框架，利用LSTM网络在广东、澳门和葡萄牙等疫情异步暴发地区之间进行协作预测，在保持数据主权的同时与本地化及集中式训练方案进行了性能对比。

    

    目的：疫情暴发在不同地区之间往往是异步发生的，然而隐私法律以及异构的公共卫生干预措施阻碍了协作预测的发展。作为概念验证，我们开发并评估了一个联邦学习（FL）框架，用于在广东、澳门和葡萄牙这三个疫情异步暴发地区之间进行跨区域COVID-19预测，同时保持数据主权。方法：利用世界卫生组织（WHO）和中国疾控中心提供的三个地区的每日病例数据（这些地区的疫情暴发是异步的且干预措施各异），我们通过联邦学习训练长短期记忆（LSTM）网络进行短期预测，并以均方根误差（RMSE）、平均绝对误差（MAE）和对称平均绝对百分比误差（SMAPE）为指标，与本地化训练和集中式训练进行了基准比较。所有实验均使用Flower联邦学习框架，并在各站点间采用一致的超参数。实验使用了公开可得的汇总病例计数数据，以便集中式

    arXiv:2611.14272v1 Announce Type: new  Abstract: Objective: Epidemic outbreaks are often asynchronous across regions, yet collaborative forecasting is hampered by privacy laws and heterogeneous public health interventions. We developed and evaluated, as a proof of concept, a federated learning (FL) framework for cross-regional COVID-19 forecasting under asynchronous outbreaks among Guangdong, Macau, and Portugal, while maintaining data sovereignty.   Methods: Using daily case data from WHO and China CDC for the three regions, whose outbreaks were asynchronous and interventions heterogeneous, we trained Long Short-Term Memory networks via FL for short-term forecasting, benchmarked against localized and centralized training using Root Mean Square Error, Mean Absolute Error, and Symmetric Mean Absolute Percentage Error. All experiments used the Flower FL framework with consistent hyperparameters across sites. Publicly available, aggregated case-count data were used so that the centralized
    
[^37]: 面向涡扇发动机可解释健康表征学习的物理约束自编码器

    Physics-Constrained Autoencoder for Interpretable Health Representation Learning in Turbofan Engines

    [https://arxiv.org/abs/2611.14266](https://arxiv.org/abs/2611.14266)

    提出一种物理约束自编码器，通过在解码器中嵌入可微分物理方程、在损失函数中引入学习偏置，使每个潜在变量对应预定义的物理量，从而实现涡扇发动机部件级可解释的健康表征学习。

    

    健康指数用于从运行数据中总结退化情况，并支持故障预测与健康管理（PHM）中的诊断与预测任务。其有效性取决于能否捕捉退化过程并解释系统状态。自编码器可以从无标签的监测数据中学习此类表征；然而，当训练仅由重构目标驱动时，潜在变量的物理含义依然不明确，限制了其解释能力和诊断价值。本工作提出一种物理约束自编码器，将系统知识作为归纳偏置嵌入解码器架构，并作为学习偏置嵌入损失函数。解码器包含可微分物理方程，以潜在变量和运行条件作为输入，从结构上引导每个潜在变量表示预定义的物理量，从而在聚合为系统级健康指数之前实现部件级分析。该方法……

    arXiv:2611.14266v1 Announce Type: new  Abstract: Health Indices (HIs) summarize degradation from operational data and support diagnostic and prognostic tasks in Prognostics and Health Management (PHM). Their usefulness depends on capturing degradation and interpreting system condition. Autoencoders learn such representations from unlabelled monitoring data; however, when training is driven only by reconstruction, the physical meaning of latent variables remains undefined, limiting their interpretation and diagnostic value. This work proposes a physics-constrained autoencoder that embeds system knowledge as an inductive bias in the decoder architecture and as a learning bias in the loss function. The decoder contains differentiable physical equations that receive latent variables and operating conditions as inputs, structurally guiding each latent variable to represent a predefined physical quantity and enabling component-level analysis before aggregation into a system-level HI.The meth
    
[^38]: 物理链：基于工程启发的LLM推理支持故障诊断

    Chain of Physics: Supporting Fault Diagnosis with Engineering-Inspired Large Language Model Reasoning

    [https://arxiv.org/abs/2611.14263](https://arxiv.org/abs/2611.14263)

    该论文提出“物理链”框架，通过从资产参考文档自动构建约束知识图谱来存储失效模式、物理量变化与传感器之间的诊断链，使大语言模型能够结合工程知识进行故障诊断推理。

    

    复杂系统中的故障诊断需要将观察到的物理量变化与产生这些变化的失效机制联系起来。经验丰富的工程师通过将传感器信息与系统行为及故障传播知识相结合来完成这项任务。大语言模型（LLM）可以支持这一过程，但其诊断推理依赖于特定资产的工程知识的可用性，以及这些知识在推理过程中如何被表示和使用。本工作提出了“物理链”，这是一个基于LLM的故障诊断框架，其核心是由特定资产的参考文档自动生成的约束知识图谱（CKG）。CoP从文本形式的工程知识中解决失效与传感器的关联问题。CKG存储了连接失效模式、物理量变化和监测传感器的诊断链。在推理过程中，系统会检索语义相似的图路径用于诊断。

    arXiv:2611.14263v1 Announce Type: new  Abstract: Fault diagnosis in complex systems requires linking observed changes in physical quantities to the failure mechanisms that produce them. Experienced engineers perform this task by combining sensor information with knowledge of system behavior and fault propagation. Large Language Models (LLMs) can support this process, but their diagnostic reasoning depends on the availability of asset-specific engineering knowledge and on how this knowledge is represented and used during inference. This work introduces Chain of Physics (CoP), an LLM-based fault-diagnosis framework built around a Constrained Knowledge Graph (CKG) generated automatically from asset-specific reference documentation. CoP addresses failure-sensor association from textual engineering knowledge. The CKG stores diagnostic chains connecting failure modes, changes in physical quantities, and monitoring sensors. During inference, semantically similar graph paths are retrieved for 
    
[^39]: 脉冲神经网络强化学习中的循环模块化结构

    Recurrent Modular Structure in Spiking Neural Network Reinforcement Learning

    [https://arxiv.org/abs/2611.14262](https://arxiv.org/abs/2611.14262)

    提出群落循环脉冲神经网络，将循环网络划分为具有固定角色的群落（控制器、动作和潜在状态群落），并结合局部三因子学习规则及新提出的Map-STDP，使脉冲强化学习网络结构更接近生物皮层微回路。

    

    生物神经回路既是模块化的又是循环的，密集的群落之间稀疏地相互连接。用于强化学习的脉冲神经网络（SNN）通常是前馈式的，如何在循环SNN中获得功能模块、以及这些模块能带来什么益处，目前尚不清楚。在这项工作中，我们引入了一类脉冲网络，称为群落循环SNN（CoRe SNNs）：这是一种被划分为具有固定角色的群落的循环网络。控制器群落接收刺激，每个动作都有属于自己的群落，而潜在群落则携带内部状态。脉冲活动因此成为群落之间的一次游走，动作即游走最终到达的群落。所有学习都是局部的三因子学习，并且这些循环群落比前馈actor更接近皮层微回路。我们比较了获得这些群落的三种机制：动态模块化，其中我们提出的一种新学习规则Map-STDP鼓励……

    arXiv:2611.14262v1 Announce Type: new  Abstract: Biological neural circuits are both modular and recurrent, with dense communities sparsely linked to one another. Spiking neural networks (SNNs) for reinforcement learning are usually feed-forward, and it is unclear how to obtain functional modules in a recurrent SNN or what they buy. In this work, we introduce a class of spiking networks, called Community Recurrent SNNs (CoRe SNNs): recurrent networks partitioned into communities with fixed roles. A controller community receives the stimulus, each action has its own community, and latent communities carry internal state. Spiking activity is then a walk between communities, and the action is the community the walk reaches. All learning is local and three-factor, and the recurrent communities are closer to cortical microcircuits than a feed-forward actor. We compare three regimes for obtaining the communities: dynamic modularity, in which a new learning rule we introduce, Map-STDP, encour
    
[^40]: 基于持续学习的人-智能体协商中的个性化情感感知

    Personalized Affective Perception via Continual Learning in Human-Agent Negotiation

    [https://arxiv.org/abs/2611.14247](https://arxiv.org/abs/2611.14247)

    该论文提出CLIFER——一种基于持续学习与想象机制的面部表情识别方法，使人形机器人在与人类协商中实时个性化地感知情绪并动态调整协商策略，相比基线显著提升了个体协商效用。

    

    与人类进行协商的自利智能体应当能够解读非语言表达并据此调整自身行为。然而，通用的面部情感模型在不同个体和情境之间性能会下降，存在泛化差距。我们提出了一种情感自适应的协商策略，其中融合了基于“想象”的持续学习面部表情识别方法（CLIFER），以实时地实现个性化的效价-唤醒度感知。差分反馈机制将效价-唤醒度方向与强度的变化转化为智能体目标效用的调整。一种异步架构使CLIFER能够在实时交互过程中于后台进行更新。在一项平衡设计的被试内实验中，66名参与者与人形机器人进行协商，结果表明：启用持续学习的智能体相比FaceChannel基线取得了显著更高的个体效用，并引发了参与者更高的让步率。持续学习条件产生了显著的……（原文摘要在此处截断）

    arXiv:2611.14247v1 Announce Type: new  Abstract: Self-interested intelligent agents that negotiate with humans should interpret nonverbal expressions and adapt their behavior accordingly. However, generic facial-affect models degrade across individuals and contexts, creating a generalization gap. We introduce an affect-adaptive negotiation strategy that incorporates Continual Learning with Imagination for Facial Expression Recognition (CLIFER) to personalize valence-arousal perception in real time. Differential feedback converts changes in valence-arousal direction and intensity into adjustments of the agent's target utility. An asynchronous architecture updates CLIFER in the background during live interaction. In a counterbalanced within-subject study in which 66 participants negotiated with a humanoid robot, the CL-enabled agent achieved significantly higher individual utility and elicited higher participant concession rates than the FaceChannel baseline. The CL condition yielded sig
    
[^41]: SinkState：基于循环状态即时SVD的高效混合大语言模型推理服务

    SinkState: On-the-Fly SVD of Recurrent States for Efficient Hybrid LLM Serving

    [https://arxiv.org/abs/2611.14246](https://arxiv.org/abs/2611.14246)

    SinkState发现线性注意力的查询、键、值集中于稳定的低维子空间（“线性注意力汇聚”现象），使循环状态实际呈低秩且秩随状态变化，并通过推理全程的自适应低秩存储与即时局部SVD分解，大幅降低混合大模型服务的内存流量和解码成本。

    

    混合模型将softmax注意力与线性注意力相结合，将历史信息压缩为固定大小的循环状态，以在模型质量和推理效率之间取得平衡。然而，线性注意力尚未将这种压缩推向极限：我们发现线性层的查询、键和值集中于一个稳定的低维子空间，我们将这一现象称为“线性注意力汇聚”，因此每个状态实际上是低秩的，且其秩在不同状态之间各不相同。然而，现有的服务系统将这些冗余状态保持为稠密形式，在每个解码步骤中对其进行完整的读取和重写，导致其内存流量随批处理大小增长并成为解码成本的主导因素。为此，SinkState在整个推理过程中以自适应低秩形式存储每个循环状态，采用缓冲更新和仅限于顶部子空间的局部SVD分解。SinkState还进一步在CPU上异步整合剩余的组件。在四个混合模……（摘要在此处截断）

    arXiv:2611.14246v1 Announce Type: new  Abstract: Hybrid models combine softmax attention with linear attention, which compresses history into fixed-size recurrent states to balance model quality and inference efficiency. However, linear attention has not pushed this compression to its limit: we find that the queries, keys, and values of linear layers concentrate in a stable, low-dimensional subspace, a phenomenon we call the linear attention sink, so each state is effectively low-rank, with a rank that varies across states. Yet existing serving systems keep these redundant states dense, reading and rewriting them in full at every decoding step, so their memory traffic grows with the batch size and dominates the decoding cost. Thus SinkState stores each recurrent state in an adaptive low-rank form throughout inference with buffered updates and a local SVD restricted to the top subspace. SinkState further consolidates the remaining components asynchronously on the CPU. On four hybrid mod
    
[^42]: 关于一般方差条件下部分参与的联邦平均算法的研究

    On Federated Averaging with Partial Participation under a General Variance Condition

    [https://arxiv.org/abs/2611.14239](https://arxiv.org/abs/2611.14239)

    本文在允许方差随局部目标差距增长的一般方差条件（ABC条件）下，为部分客户端参与的联邦平均算法建立了包含异质性度量的有限时间平稳性收敛界，实现了随采样客户端数量的线性加速，并在全参与情形下与已有结果一致。

    

    本文研究了在一般方差条件下具有部分客户端参与的联邦平均算法的收敛性。在每一通信轮次中，服务器从 n 个客户端中无放回地均匀随机抽取 S 个客户端。假设随机梯度满足所谓的 ABC 条件，该条件允许梯度方差随局部目标差距而增长。对于光滑非凸目标函数，我们建立了以异质性度量 σ_f* := f* − (1/n)Σf_i* 和客户端采样因子 I_S := (n−S)/(n−1) 表示的有限时间平稳性界，其中 f* 和 f_i* 分别表示全局平均目标和客户端 i 的局部目标的下确界。该界在采样客户端数量 S 上呈现线性加速，且当 S=n 时可恢复全参与情形的结果。在递减步长条件下，我们进一步证明了梯度范数……（摘要在此处截断）

    arXiv:2611.14239v1 Announce Type: cross  Abstract: In this paper, we study the convergence of Federated Averaging (FedAvg) with partial participation under a general variance condition. In each communication round, the server samples $S$ of the $n$ clients uniformly at random without replacement. The stochastic gradients are assumed to satisfy the so-called ABC condition, which allows their variance to grow with the local objective gaps. For smooth nonconvex objectives, we establish a finite-time stationarity bound in terms of the heterogeneity measure $\sigma_f^\star:=f^\star-\frac{1}{n}\sum_{i=1}^n f_i^\star$ and the client sampling factor $I_S:=(n-S)/(n-1)$. Here, $f^\star$ and $f_i^\star$ denote the infima of the global average objective and client $i$'s local objective, respectively. The bound exhibits a linear speedup in the number of sampled clients $S$ and recovers the full-participation result when $S=n$. Under diminishing stepsizes, we further prove that the gradient norm con
    
[^43]: 可验证智能体流量类别：一种面向智能体AI流量的受管控服务质量机制，从仿真到真实内核对抗性测试的评估

    Verifiable Agentic Traffic Class: A Policed Quality-of-Service Mechanism for Agentic AI Traffic, Evaluated from Simulation to Real-Kernel Adversarial Testing

    [https://arxiv.org/abs/2611.14211](https://arxiv.org/abs/2611.14211)

    本文提出可验证智能体流量类别（VATC），通过DSCP流量声明、路由器侧验证分类器和速率受限优先队列，为智能体AI流量提供可验证的服务质量保障，在突发场景下相比传统DiffServ将智能体排队延迟降低98%，同时仅使人类流量延迟增加12%。

    

    基于大语言模型构建的软件智能体所产生的流量与人类网页流量截然不同，然而路由器尚无针对此类流量的类别，也无法验证发送方声称自己是智能体的真实性。本文提出了可验证智能体流量类别（VATC），它结合了三项机制：一个用于声明智能体流量的保留差分服务代码点（DSCP）值、一个路由器侧分类器（将与声明不符的流量降级至在尽力而为流量之后服务的类别），以及一个速率受限的优先队列。研究在两个仿真器和一个真实的Linux内核上，将VATC与静态差分服务DiffServ（一种信任标记的专用类别）以及fq_codel（一种每流公平排队调度器）进行了对比。结果表明，VATC在稳定负载下与DiffServ表现相当；在构造的突发流量场景中，VATC在仿真环境下（假设即时判定）相比DiffServ将智能体平均排队延迟降低了98%，而人类流量的延迟仅增加12%，其中大部分收益来自于独立类别的设计。

    arXiv:2611.14211v1 Announce Type: cross  Abstract: Software agents built on large language models generate traffic unlike human web traffic, yet routers have no class for it and cannot verify a sender's claim to be an agent. This paper proposes the Verifiable Agentic Traffic Class (VATC), which combines a reserved Differentiated Services Code Point (DSCP) value declaring agent traffic, a router-side classifier that demotes flows contradicting the declaration to a class served after best-effort traffic, and a rate-limited priority queue. VATC was compared with static Differentiated Services (DiffServ), a dedicated class that trusts the marking, and fq_codel, a per-flow fair-queuing scheduler, in two simulators and on a real Linux kernel. VATC matched DiffServ under steady load. In a constructed burst it cut mean agent queuing delay by 98% relative to DiffServ in simulation (assuming an immediate verdict), at a 12% increase in human delay; most of the gain came from the separate class. U
    
[^44]: Tree-Ising：利用比特级Ising编辑修复语言模型

    Tree-Ising: Repairing Language Models with Bit-Level Ising Edits

    [https://arxiv.org/abs/2611.14210](https://arxiv.org/abs/2611.14210)

    Tree-Ising将语言模型的无梯度权重修复转化为BF16权重上的稀疏离散优化问题，通过符号翻转和指数移位等比特级编辑，将目标函数精确分解为逐输出行的QUBO/Ising问题，并在GPU上用模拟退火或量子方法求解，以修复模型推理循环等缺陷。

    

    arXiv:2611.14210v1 公告类型：新 摘要：权重空间编程由Takehi等人近期在MIT的工作中提出，使前沿的LLM智能体能够读取网络的权重并编写修复方案。他们的GPT-6 Astra智能体在没有梯度的情况下纠正了Qwen3.5-0.8B的推理循环，共使用了158个智能体回合、8970万个token和50次前向传播请求。我们的方法Tree-Ising将这种修复表述为模型自身BF16权重上的稀疏离散优化问题。前向统计定义了一个目标函数，该函数能够削弱循环方向、增强结论方向，并在有效token上保持该层的输出。在状态树表示中（无损BF16存储作为其中的一个特例），可允许的编辑是保持所有共享组指数和尾数的符号翻转和指数移位。我们将该目标函数精确分解为独立的QUBO（Ising）问题，每个输出行一个，并在GPU上使用模拟退火或量子（原文在此处截断）方法求解。

    arXiv:2611.14210v1 Announce Type: new  Abstract: Weight-space programming, introduced in recent MIT work by Takehi et al., enables a frontier LLM agent to read a network's weights and write a repair. Their GPT-6 Astra agent corrected the reasoning loops of Qwen3.5-0.8B without gradients, using 158 agent turns, 89.7 million tokens, and 50 forward-pass requests. Our method, Tree-Ising, casts this repair as a sparse discrete optimization problem over the model's own BF16 weights. Forward statistics define an objective that weakens a loop direction, strengthens a concluding direction, and preserves the layer's output on productive tokens. Within a state-tree representation, where lossless BF16 storage acts as a special case, the admissible edits are sign flips and exponent shifts that preserve all shared group exponents and mantissas. We decompose this objective exactly into independent QUBO (Ising) problems, one per output row, which we solve on a GPU using simulated annealing or quantum-
    
[^45]: SpecValley：将聚合作为稠密异质图中的谱谷

    SpecValley: Aggregation as Spectral Valley in Dense Heterophilic Graphs

    [https://arxiv.org/abs/2611.14176](https://arxiv.org/abs/2611.14176)

    该论文提出SpecValley方法，将图神经网络中固定为度归一化的聚合步骤改造为可学习的谱谷算子，为每个图自适应地学习衰减带的位置、锐度和两侧增益，从而解决稠密异质图中聚合会衰减携带标签信号的问题。

    

    异质图中的节点分类仍然是一个悬而未决的问题：在邻域上进行聚合可能会恰好衰减信号中携带标签的那一部分。该领域以往的应对方式是设计表达能力越来越强的传播算子，却将其之前的聚合步骤固定为度归一化。然而，度归一化本身就像一个谱谷，会衰减一个固定的中频带，其位置（在 lambda 等于 1 处）由归一化方式而非数据决定，而这正是稠密图特征值聚集之处。因此，我们从表示而非传播入手，采用两个阶段的设计：先进行可学习的聚合，再进行普通的谱滤波。第一个阶段中，SpecValley 为每个图学习谱谷的位置、锐度和两侧增益。我们形式化地展示了聚合算子与下游滤波器之间的相互作用，并证明了响应 1 减 lambda 的（摘要在此处截断）

    arXiv:2611.14176v1 Announce Type: cross  Abstract: Node classification in heterophilic graphs remains an open problem: aggregating over neighbourhoods can damp precisely the part of the signal that carries the labels. The field has answered with ever more expressive propagation operators, leaving the aggregation step that precedes them fixed by degree normalisation. However, degree normalisation acts as a spectral valley in its own right, attenuating a fixed middle band whose position at lambda equal 1 is set by the normalisation rather than by the data, exactly where the eigenvalues of dense graphs accumulate. We therefore begin with the representations rather than the propagation, in two stages: a learnable aggregation followed by an ordinary spectral filter. The first, SpecValley learns the position, sharpness and flank gains of the valley for every graph. We show formally how the aggregation operator and downstream filter interact. We establish that the response 1 minus lambda of d
    
[^46]: 二进制秩的非自适应单边测试

    Nonadaptive One-Sided Testing of Binary Rank

    [https://arxiv.org/abs/2611.14163](https://arxiv.org/abs/2611.14163)

    本文研究了二进制秩的单边错误非自适应测试问题，给出了与矩阵维度无关的 $\tilde O(d^4/\epsilon^2)$ 查询复杂度上界和 $\tilde\Omega(d^{3/2}/\epsilon)$ 的一般下界。

    

    二进制矩阵的二进制秩是指能够划分其所有1元素的不相交全1矩形的最小数量。我们研究了二进制秩至多为 $d$ 的元素查询测试问题，采用单边错误和非自适应查询。我们给出了一个使用 $\tilde O(d^4/\epsilon^2)$ 次查询的测试器，其查询复杂度与矩阵维度无关，并证明了一般下界为 $\tilde\Omega(d^{3/2}/\epsilon)$。

    arXiv:2611.14163v1 Announce Type: cross  Abstract: The binary rank of a binary matrix is the minimum number of disjoint all-one rectangles that partition its one-entries. We study entry-query testing of binary rank at most $d$, with one-sided error and nonadaptive queries. We give a tester using $\tilde O(d^4/\epsilon^2)$ queries, independent of the matrix dimensions and a general lower bound of $\tilde \Omega(d^{3/2}/\epsilon)$.
    
[^47]: 通过再循环训练的单极神经元网络用于基于奖励的持续学习

    Unipolar Neuron Networks Trained by Recirculation for Reward-Based Continual Learning

    [https://arxiv.org/abs/2611.14153](https://arxiv.org/abs/2611.14153)

    该论文受章鱼等无脊椎动物单极神经元结构启发，提出了一种树状神经元网络，通过仅需奖励信号的“再循环”学习规则实现持续学习，其中每个突触仅依赖局部信息更新权重，且更新方向近似遵循误差梯度。

    

    单极神经元是章鱼及大多数其他无脊椎动物中占主导地位的神经细胞类型。单一的神经突（从细胞体延伸出的纤维）分支成一棵树状结构，同时承载输入和输出突触。这些分支向附近的细胞发送分级的局部输出，而主神经突上的一个位点向远处的目标发送全局输出。本文提出了一种由这类神经元构成的网络及其学习规则。每个神经元被建模为由无源片段组成的树，其激活值作为线性系统的稳态被精确计算。该网络仅凭奖励信号进行学习，采用一种称为“再循环”（recirculation）的规则。每次前向传播之后，第二次传播将所请求的输出变化通过相同的连接反向送回。随后，每个突触仅利用其自身位置可获得的量来调整权重。由于每个神经元的系统是对称的，这种更新紧密遵循（上游）误差的梯度……

    arXiv:2611.14153v1 Announce Type: new  Abstract: Unipolar neurons are the predominant type of nerve cell in the octopus and in most other invertebrates. A single neurite (a fiber extending from the cell body) branches into a tree that carries both input and output synapses. The branches send graded local outputs to nearby cells, and one site on the main neurite sends a global output to distant targets. This paper proposes a network of such neurons and a learning rule for it. Each neuron is modeled as a tree of passive segments, whose activations are computed exactly as the steady state of a linear system. The network learns from reward alone, with a rule called recirculation. After each forward pass, a second pass sends requested changes of the outputs back through the same connections. Each synapse then adjusts its weight using only quantities available at its own location. Because the system of each neuron is symmetric, this update closely follows the gradient of the error in the pre
    
[^48]: 基于偏差感知预训练的心电图心血管疾病检测

    Deviation-Aware Pretraining for Cardiovascular Disease Detection from Electrocardiograms

    [https://arxiv.org/abs/2611.14141](https://arxiv.org/abs/2611.14141)

    该论文提出了一种基于近百万份无标签12导联心电图的自监督预训练框架，通过偏差定位嵌入（DLE）和关联重构嵌入（ARE）两个模块捕捉波形偏差及跨导联异常模式，在心血管疾病检测任务上相比现有方法显著提升了AUROC和F1性能。

    

    心电图信号对心血管疾病的检测至关重要。目前大多数深度学习模型依赖于监督式的、针对特定任务的训练，这限制了它们在不同疾病、患者群体和采集环境之间的泛化能力。本文提出了一种自监督的心电图特征提取框架，该框架从近一百万份无标签的12导联心电图记录中学习。该框架包含两个模块：偏差定位嵌入（DLE），用于检测波形层面相对于正常模式的偏差；以及关联重构嵌入（ARE），通过偏差引导的重构来建模跨导联关联，以捕捉在多个导联中表达的异常模式。这些模块使框架能够捕捉细微的偏差相关模式及其跨导联上下文。该框架在多个公开数据集上相比先前方法实现了最高6.5%的AUROC和12.8%的F1提升。

    arXiv:2611.14141v1 Announce Type: new  Abstract: Electrocardiogram (ECG) signals are critical for cardiovascular disease (CVD) detection. Most current deep learning models rely on supervised, task-specific training, which limits their ability to generalize across diseases, patient populations, and acquisition settings. This paper presents a self-supervised ECG feature extraction framework that learns from nearly one million unlabeled 12-lead ECG recordings. The framework includes two modules: Deviation Localization Embedding (DLE), which detects waveform-level deviations from normal patterns, and Association Reconstruction Embedding (ARE), which models cross-lead associations through deviation-guided reconstruction to capture abnormal patterns expressed across multiple leads. These modules enable the framework to capture subtle deviation-related patterns and their cross-lead context. The framework achieves up to 6.5% AUROC and 12.8% F1 improvements over prior methods on multiple public
    
[^49]: HaiRuo：利用卫星测高基础模型以场和对象双重形式预报海洋涡旋

    HaiRuo: Forecasting Ocean Eddies as Fields and Objects with a Satellite-Altimetry Foundation Model

    [https://arxiv.org/abs/2611.14136](https://arxiv.org/abs/2611.14136)

    HaiRuo将24年卫星测高数据预训练的10.5亿参数基础模型HaiRuo-1B与拉格朗日对象解码器XuanJi相结合，仅需30天卫星海面高度图的一次前向传播，即可同时预报15天海洋物理场、涡旋掩膜及每个涡旋的轨迹。

    

    中尺度涡旋携带了海洋的大部分动能，并在海盆之间输送热量、盐分和营养物质，然而涡旋两周后将会出现在哪里，目前仍然依赖海洋模式输出加上事后跟踪启发式方法来获取。我们提出了HaiRuo，它同时以场和对象两种形式预报涡旋：仅需输入30天的卫星海平面高度图，一次前向传播即可返回15天的海平面高度与地转流场、每日涡旋掩膜，以及当前存活的每一个涡旋的轨迹。HaiRuo由两个分工明确的组件构成：(1) HaiRuo-1B，一个开放权重的10.5亿参数基础模型，在24年的全球卫星测高观测数据上进行无标签预训练，为下游任务提供15天的物理场和掩膜（在相同解码器条件下，场技巧提升+0.14，mIoU提升1.1个百分点），并在冻结状态下将15天涡旋中心误差进一步降低1.2公里（相比从头训练降低1.35公里）；(2) XuanJi，一个拉格朗日对象解码器，通过采样…（原文摘要在此处截断）

    arXiv:2611.14136v1 Announce Type: new  Abstract: Mesoscale eddies carry most of the ocean's kinetic energy and move heat, salt and nutrients across basins, yet where an eddy will be two weeks ahead is still read off ocean models with post-hoc tracking heuristics. We introduce HaiRuo, which forecasts eddies as fields and as objects at once: from 30 days of satellite sea-level maps, one forward pass returns 15 days of sea level and geostrophic velocity, daily eddy masks and the trajectory of every eddy alive today. HaiRuo pairs two components with a clear division of labour: (1) HaiRuo-1B, an open-weight 1.05-billion-parameter foundation model pretrained without labels on 24 years of global satellite-altimetry observations, which supplies the 15-day physical fields and masks (+0.14 field skill and +1.1 mIoU points under the same decoder) and further lowers 15-day centre error by 1.2 km frozen (1.35 km over training from scratch); and (2) XuanJi, a Lagrangian object decoder that samples t
    
[^50]: 理由对但原因错？评估解释引导学习中的合理性与行为改变

    Right for the Wrong Reasons? Evaluating Plausibility and Behavioral Change in Explanation-Guided Learning

    [https://arxiv.org/abs/2611.14134](https://arxiv.org/abs/2611.14134)

    研究表明解释引导学习（EGL）的效果高度依赖具体目标，其中基于理由的数据增强在捷径证据可与任务相关前景信息分离时带来最显著的鲁棒性提升（准确率最高提升4.8个百分点），且模型解释与人类理由的一致性提升并不必然意味着决策策略的真正改变。

    

    解释引导学习（EGL）承诺构建不仅准确、而且依赖任务相关证据的模型。然而，模型解释与人类提供的理由之间一致性的提高，是否也意味着模型决策策略的改变，这一点仍不清楚。我们通过使用构建的捷径数据集，在受控合成捷径引入的条件下研究这一矛盾。受现实世界应用的启发，我们的基准涵盖不同类型的捷径，包括局部线索、基于背景的线索和全局线索。这使我们能够评估不同的EGL目标是否能产生行为上的改变和解释上的改变。研究发现，EGL的效果在不同目标之间差异很大，具有很强的目标特异性。当捷径证据能够与任务相关的前景信息分离时，基于理由的数据增强提供了最明显的鲁棒性提升，平均准确率提升高达+4.8个百分点。

    arXiv:2611.14134v1 Announce Type: new  Abstract: Explanation-Guided Learning (EGL) promises models that are not only accurate but also rely on task-relevant evidence. However, whether improved agreement between model explanations and human-provided rationales also implies a changed decision strategy remains unclear. We study this tension under controlled synthetic shortcut introduction using a constructed shortcut dataset. Inspired by real-world applications, our benchmark covers shortcuts of different kinds, including localized, background-based, and global cues. This allows us to evaluate whether different EGL objectives produce behavioral changesand explanatory changes. Across these axes, we find that the effects of EGL are strongly objective-specific. Rationale-based data augmentation provides the clearest robustness gains when shortcut evidence can be separated from task-relevant foreground information, reaching average accuracy improvements of up to +4.8 percentage points (pp) to
    
[^51]: 基于组织学信息引导的隐式神经表示的连续三维空间转录组学重建

    Continuous 3D Spatial Transcriptomics Reconstruction via Histology-Informed Implicit Neural Representations

    [https://arxiv.org/abs/2611.14129](https://arxiv.org/abs/2611.14129)

    提出了一种组织学信息引导的多任务隐式神经表示方法HIM-INR，通过共享骨干网络联合重建基因表达谱和组织学表示，并利用密集采样的组织学图像作为辅助监督，实现了从稀疏空间转录组学观测到任意三维坐标的连续基因表达重建。

    

    三维空间转录组学（3D ST）能够分析跨组织深度的基因表达模式。然而，3D ST的测量是从离散的二维组织切片中获取的，且由于采集成本高昂，这些切片通常采样稀疏。为了解决这一局限性，我们提出了一种组织学信息引导的多任务隐式神经表示方法（HIM-INR），用于从稀疏的ST观测中进行连续的三维基因表达重建。HIM-INR使用具有共享骨干网络的隐式神经表示，联合重建基因表达谱和组织学表示，并利用密集采样的组织学图像作为辅助空间监督。训练完成后，HIM-INR无需组织学图像即可预测任意三维坐标处的基因表达。在两个3D ST数据集上的实验表明，与现有方法相比，该方法在重建方面有所改进，尤其是在缺失ST切片数量增加的情况下表现更为突出。

    arXiv:2611.14129v1 Announce Type: cross  Abstract: Three-dimensional spatial transcriptomics (3D ST) enables the analysis of gene-expression patterns across tissue depths. However, 3D ST measurements are acquired from discrete 2D tissue sections, which are often sparsely sampled due to high acquisition costs. To address this limitation, we propose a histology-informed multi-task implicit neural representation (HIM-INR) for continuous 3D gene-expression reconstruction from sparse ST observations. HIM-INR jointly reconstructs gene-expression profiles and histological representations using an INR with a shared backbone, leveraging densely sampled histological images as auxiliary spatial supervision. Once trained, HIM-INR predicts gene expression at arbitrary 3D coordinates without requiring histological images. Experiments on two 3D ST datasets demonstrate improved reconstruction over existing methods, particularly as the number of missing ST sections increases.
    
[^52]: 基于池化诱导子结构的层次图表示学习

    Hierarchical Graph Representation Learning with Pooling-Induced Substructures

    [https://arxiv.org/abs/2611.14126](https://arxiv.org/abs/2611.14126)

    该论文提出池化子结构感知GNN，通过显式保留并处理结构池化所诱导的子结构来增强层次图表示学习，在WL测试意义上提升了基于池化的GNN表达能力，并引入了具有可控层次结构的新合成数据集用于系统评估。

    

    需要对超出局部邻域范围的子结构进行推理的层次图级任务，正给标准图神经网络（GNN）带来挑战。虽然图池化旨在通过将图粗化为层次表示来解决这一问题，但现有方法往往未能显式保留它们所诱导的子结构，导致实证表现不一致。此外，由于缺乏具有显式层次结构的基准测试，评估此类方法仍然具有挑战性。我们提出了池化子结构感知GNN（Pooled-Substructure-Aware GNNs），这是一种图池化方法，能够显式保留并处理基于结构的池化方案所发现的子结构。我们证明了这种设计在Weisfeiler-Lehman（WL）测试的意义上增强了基于池化的GNN的表达能力。为了实现有原则的评估，我们引入了两个具有可控层次结构的新合成数据集，并提出……

    arXiv:2611.14126v1 Announce Type: new  Abstract: Hierarchical graph-level tasks that require reasoning over substructures that extend beyond local neighborhoods are posing challenges for standard Graph Neural Networks (GNNs). While graph pooling aims to address this by coarsening graphs into hierarchical representations, existing methods often fail to explicitly preserve the substructures they induce, leading to inconsistent empirical performance. Moreover, evaluating such approaches remains challenging due to the lack of benchmarks with explicit hierarchical structure. We propose Pooled-Substructure-Aware GNNs, an approach to graph pooling that explicitly preserves and processes substructures uncovered by structure-based pooling schemes. We show that this design enhances the expressive power of pooling-based GNNs in the sense of the Weisfeiler-Lehman (WL) test. To enable principled evaluation, we introduce two new synthetic datasets with controllable hierarchical structure and propose
    
[^53]: CroCoMAE：面向SAR-光学遥感数据的跨模态自监督协同表示学习

    CroCoMAE: Cross-modal self-supervised collaborative representation learning for SAR-optical remote sensing data

    [https://arxiv.org/abs/2611.14124](https://arxiv.org/abs/2611.14124)

    提出CroCoMAE框架，在基于ViT的架构中统一对比学习与掩码图像建模，从图像级和补丁级两个互补层面挖掘SAR-光学遥感数据的细粒度跨模态互补结构，实现自监督协同表示学习。

    

    arXiv:2611.14124v1 公告类型：交叉 摘要：遥感领域越来越多地通过SAR和光学等互补传感器来捕捉地球表面，然而地面真值标注仍然稀缺，限制了监督方法的应用。自监督学习（SSL）解决了标注稀缺的问题，而跨模态自监督学习进一步利用了共址部署传感器之间的互补性，但现有的跨模态框架主要通过全局图像级表示来对齐不同模态，或将各模态视为基本冗余的，导致其细粒度的互补结构未被充分挖掘。我们提出CroCoMAE，一个面向SAR-光学遥感的跨模态自监督学习框架，它在基于ViT的架构中统一了对比学习与掩码图像建模，并在两个互补的层面上运行。在图像层面，作用于类标记（class token）的模态内对比损失可产生具有全局判别性且对数据增强不变的表示；在补丁层面，掩码图像建模驱动细粒度的……

    arXiv:2611.14124v1 Announce Type: cross  Abstract: Remote sensing increasingly captures the Earth's surface through complementary sensors such as SAR and optical, yet ground-truth annotations remain scarce, limiting supervised approaches. Self-supervised learning (SSL) addresses the label scarcity, and cross-modal SSL further exploits the complementarity between co-located sensors, but existing cross-modal frameworks align modalities mostly through global image-level representations or treat them as largely redundant, leaving their fine-grained complementary structure underexploited. We propose CroCoMAE, a cross-modal SSL framework for SAR-optical remote sensing that unifies contrastive learning and masked image modeling within a ViT-based architecture, operating at two complementary levels. At the image level, an intra-modality contrastive loss on the class token yields globally discriminative, augmentation-invariant representations. At the patch level, masked image modeling drives fi
    
[^54]: 机会成本Option-Critic：面向鲁棒分层强化学习的自适应切换成本

    Opportunity-Cost Option-Critic: Adaptive Switching Costs for Robust Hierarchical Reinforcement Learning

    [https://arxiv.org/abs/2611.14099](https://arxiv.org/abs/2611.14099)

    该论文提出机会成本Option-Critic（OC²）框架，利用当前选项与最佳备选选项之间的价值差距来定义自适应的状态依赖切换成本，从而实现更鲁棒的分层强化学习。

    

    深思成本在分层强化学习（HRL）中通常被用于防止选项退化为原始动作。现有方法依赖于固定惩罚或基于持续时间的自适应调整，但这两者都是选项效用的间接代理指标。本文认为，选项的持续性应当取决于放弃当前行为的机会成本。我们提出了机会成本Option-Critic（OC²）框架，该框架利用当前活跃选项与其最佳备选选项之间的正向价值差距来定义一个依赖于状态的深思成本。这使得当当前选项明显更优时鼓励其持续执行，而当出现更优选项时则允许快速切换。我们证明了直接学习此类成本会导致退化的零成本解，因此引入了一个辅助目标，使学习到的成本与价值差距目标对齐。该方法在Four Rooms和Pinball环境中进行了评估。

    arXiv:2611.14099v1 Announce Type: new  Abstract: Deliberation costs are commonly used in hierarchical reinforcement learning (HRL) to prevent options from collapsing to primitive actions. Existing approaches rely on fixed penalties or duration-based adaptations, both of which are indirect proxies for an option's utility. This paper argues that option persistence should instead depend on the opportunity cost of abandoning the current behavior. We propose the Opportunity-Cost Option-Critic ($OC^2$) framework, which defines a state-dependent deliberation cost using the positive value gap between the active option and its best alternative. This encourages persistence when the current option is clearly preferable and allows rapid switching when a better option emerges. We show that directly learning such costs leads to a degenerate zero-cost solution and introduces an auxiliary objective that aligns the learned cost with the value-gap target. The approach is evaluated in Four Rooms and Pinb
    
[^55]: CUVE：通过可视化让监督分类更加易于上手

    CUVE: Making Supervised Classification more Accessible through Visualization

    [https://arxiv.org/abs/2611.14098](https://arxiv.org/abs/2611.14098)

    CUVE是一种可视化界面，通过生成人类和机器学习算法均可理解的可视化作为中间表示，帮助用户更便捷地构建、理解和改进监督分类器。

    

    我们提出了CUVE，这是一个让用户能够通过生成既可被人类理解、又可被学习算法解释的可视化来构建分类器的界面（即“可视化作为中间表示”方法）。该用户界面使用户能够创建监督分类器，并以可视化的方式探索如何改进分类器以及分类器如何执行其功能。一项初步研究调查了该原型的可用性，并将其与传统的基于文本的编程界面进行了比较。

    arXiv:2611.14098v1 Announce Type: cross  Abstract: We present CUVE, an interface that enables people to build classifiers by generating visualizations that are interpretable by both humans and learning algorithms (the Visualization as Intermediate Representation approach). This UI enables users to create supervised classifiers and to visually investigate how to improve them and how they carry out their function. A preliminary study investigates the usability of our prototype and compares it with a traditional text-based programming interface.
    
[^56]: 风格化即扩散流形上的测地线导航

    Stylization as Geodesic Navigation on Diffusion Manifolds

    [https://arxiv.org/abs/2611.14097](https://arxiv.org/abs/2611.14097)

    提出免训练框架Stylus，将人脸风格化建模为在预训练扩散模型弯曲潜在流形上沿测地线导航的过程，通过内容-风格潜在插值与自注意力查询优化双控制，实现既具艺术风格又保持人物身份的风格迁移。

    

    面部风格化要求一种精细的变换：图像应看起来无疑具有艺术性，同时人物又应保持无疑是自己。现有的免训练方法和基于指令的方法往往无法提供这样的控制，要么偏离人物身份，要么只能产生较弱的艺术化效果。我们提出了Stylus，一个免训练框架，它将面部风格迁移视为在预训练扩散模型弯曲潜在流形上的导航过程。不同于在平坦欧几里得空间中进行插值，Stylus通过在回拉度量下最小化比率加权的离散测地线能量来精炼插值得到的内容-风格点，并借助雅可比矩阵的秩一幂迭代近似使该过程可计算。这一原理驱动了两种互补的控制方式：内容-风格潜在插值用于实现平滑的语义融合，自注意力查询优化用于实现空间自适应的风格注入。由此产生的双控制设计（原文此处截断）……

    arXiv:2611.14097v1 Announce Type: cross  Abstract: Facial stylization asks for a delicate transformation: the image should look unmistakably artistic, yet the person should remain unmistakably the same. Existing training-free and instruction-driven methods often fail to expose such control, either drifting away from identity or producing only weak stylization. We introduce Stylus, a training-free framework that treats facial style transfer as navigation on the curved latent manifold of a pretrained diffusion model. Rather than interpolating in flat Euclidean space, \method refines the interpolated content--style point by lowering a ratio-weighted discrete geodesic energy under the pullback metric, made tractable by a rank-one power-iteration approximation of the Jacobian. This principle drives two complementary controls: content--style latent interpolation for smooth semantic fusion, and self-attention query optimization for spatially adaptive style injection. The resulting dual-contro
    
[^57]: BODYinTRANSIT-Lab：一个用于研究感觉驱动的身体知觉与情感体验转变的多模态数据集

    BODYinTRANSIT-Lab: A Multimodal Dataset for Investigating Sensory-Driven Transformations in Body Perception and Affective Experiences

    [https://arxiv.org/abs/2611.14093](https://arxiv.org/abs/2611.14093)

    该论文推出了BODYinTRANSIT-Lab多模态数据集，通过脚步错觉闭环范式采集101名分层招募参与者的全身运动学、多种生理信号及主观情感与身体知觉数据，用于研究感觉驱动的身体知觉转变并支持从传感器数据推断主观状态的计算模型开发。

    

    将同步的行为与生理信号和主观情感及身体知觉测量相结合的多模态数据集仍然十分稀缺。这既限制了对具身体验的理解，也限制了从传感器数据推断主观状态的计算模型的发展。我们推出了BODYinTRANSIT-Lab，这是一个填补该空白的多模态数据集，通过捕捉参与者对脚步错觉的情感、知觉和行为反应来构建。在这一闭环范式中，参与者在执行行走任务的同时，接收对自身脚步声进行实时操纵的听觉反馈，旨在改变其感知到的身体重量。该数据集包含101名参与者的同步记录，参与者通过按饮食失调症状和身体活动水平进行的系统分层招募。BODYinTRANSIT-Lab将全身运动学与肌电图、心电图、呼吸以及皮电活动相结合。

    arXiv:2611.14093v1 Announce Type: cross  Abstract: Multimodal datasets that combine synchronised behavioural and physiological signals with subjective affective and body perception measures are still rare. This limits both the understanding of embodied experience and the development of computational models that infer subjective states from sensor data. We introduce BODYinTRANSIT-Lab, a multimodal dataset that addresses this gap by capturing affective, perceptual, and behavioural responses to the Footsteps Illusion. In this closed-loop paradigm, participants perform walking tasks while receiving real-time manipulated auditory feedback of their own footsteps intended to alter perceived body weight. The dataset comprises synchronised recordings from 101 participants recruited through systematic stratification by eating-disorder symptomatology and physical activity level. BODYinTRANSIT-Lab combines full-body kinematics with electromyography, electrocardiography, respiration, and electroder
    
[^58]: Sable：面向大语言模型位精确推理的小域Sumcheck论证系统

    Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models

    [https://arxiv.org/abs/2611.14083](https://arxiv.org/abs/2611.14083)

    Sable通过整数化确定性计算、小域sumcheck约束系统和单GPU逐层流式证明，实现了对大语言模型位精确推理的可验证性，在单块GPU上92.5秒内即可完成Qwen2.5-32B生成4096个token全部64层推理的证明。

    

    我们提出Sable，一个论证系统与单GPU证明器，用于证明一个已承诺的、整数量化的transformer模型从给定提示词逐位精确地生成了给定的token序列。Sable定义了一个不含任何浮点步骤的整数函数F_O，其中每一次舍入、归一化、softmax和采样决策都完全由其输入唯一确定，并通过一个约束系统对其加以证明，该约束系统的关系可证明在31位KoalaBear素数模运算下永不发生回绕。该证明将零值检查（zerochecks）与LogUp-GKR查找批处理为每个域上的单次sumcheck，并通过一个由Reed-Solomon码、BLAKE3 Merkle树和混合长度BaseFold打开构成的多线性承诺，一次性承诺所有见证张量与权重。证明器在单块GPU上逐层流式处理模型。在一块NVIDIA RTX PRO 6000 Blackwell GPU上，Sable用92.5秒证明Qwen2.5-32B全部64层生成4,096个token，用398秒生成16,384个token，并覆盖Qwen3-32B与专家混合模型（mixture-of-experts）……（摘要原文在此处截断）

    arXiv:2611.14083v1 Announce Type: cross  Abstract: We present Sable, an argument system and single-GPU prover for the statement that a committed, integer-quantized transformer produced a given token sequence from a given prompt, bit for bit. Sable fixes an integer function F_O with no floating-point step, in which every rounding, normalization, softmax and sampling decision is determined by its inputs, and proves it with a constraint system whose relations provably never wrap modulo the 31-bit KoalaBear prime. The proof batches zerochecks and LogUp-GKR lookups into one sumcheck per domain, and commits to every witness tensor and weight in one multilinear commitment built from Reed-Solomon codes, BLAKE3 Merkle trees and a mixed-length BaseFold opening. The prover streams the model layer by layer on one GPU. On one NVIDIA RTX PRO 6000 Blackwell GPU, Sable proves all 64 layers of Qwen2.5-32B in 92.5 s for 4,096 tokens and 398 s for 16,384 tokens, and covers Qwen3-32B and the mixture-of-ex
    
[^59]: CAGE 4 分数究竟衡量了什么：对一个学习型网络防御基准、其参考防御者及基于它们构建的排行表的审计

    What a CAGE 4 Score Measures: An Audit of a Learned Cyber-Defense Benchmark, Its Reference Defenders, and the Tables Built on Them

    [https://arxiv.org/abs/2611.14068](https://arxiv.org/abs/2611.14068)

    该论文审计了 CAGE Challenge 4 网络防御基准，发现其模拟器会静默丢弃发往忙碌智能体的动作并照常计分，而两个参考防御者均依赖文档未披露的“忙碌”标志检查，修正这些问题后已发布分数每回合变动高达约 146 分，表明基于该基准的排行表所测量的内容远少于其表面所示。

    

    CAGE Challenge 4 是用于比较学习型多智能体网络防御者的基准。我们在固定容器环境中复现了它的两个参考防御者 KEEP 和 H-MARL Expert，并表明从该基准中读取的三个量所衡量的内容少于人们所认为的。首先，当某个智能体的上一个动作仍在执行时，模拟器会静默丢弃为该智能体断言的任何动作，并且对被丢弃的 Restore 与实际执行的 Restore 收取相同的一分代价。文档所描述的观测中并不暴露“忙碌”标志。两个参考防御者都在各自的封装中自行读取该标志，而在评估时关闭该检查会使 KEEP 已发布的策略每回合损失 99.58 分、H-MARL 损失 145.98 分——两种情况下，扣分均来自发送给忙碌智能体的 Restore，而 H-MARL 实际执行的动作逐步保持完全不变。在一个监督式测试框架上，将干预延迟到智能体空闲时才执行，在已公布的设置下每回合价值 25.3 分和 27.6 分。在训练中……

    arXiv:2611.14068v1 Announce Type: cross  Abstract: CAGE Challenge 4 is the benchmark on which learned multi-agent cyber defenders are compared. We reproduce its two reference defenders, KEEP and H-MARL Expert, in a pinned container and show that three things read from it measure less than they are taken to measure. First, the simulator silently discards any action asserted for an agent whose previous action is still running, and charges one point for a discarded Restore as for one that runs. The documented observation does not expose the busy flag. Both reference defenders read it in their own wrappers, and switching that check off at evaluation costs KEEP's released policy 99.58 points per episode and H-MARL's 145.98, in both cases the charge for Restores sent to busy agents, with H-MARL's executed actions unchanged step for step. On a supervisory harness, holding interventions until the agent is idle is worth 25.3 and 27.6 points per episode at the published setting. In training, the
    
[^60]: 泛化因果决策：马尔可夫等价性、可迁移性与反事实

    Generalizing Causal Decision-Making: Markov Equivalence, Transportability and Counterfactuals

    [https://arxiv.org/abs/2611.14066](https://arxiv.org/abs/2611.14066)

    本文将结构因果老虎机推广至三种更一般的场景——通过马尔可夫等价性刻画部分已知因果结构下可能最优的最小干预集、利用部分可迁移性从源环境推导目标奖励边界并获得遗憾保证、以及构建支持反事实行动的反事实结构因果老虎机框架。

    

    结构因果老虎机在奖励机制未知时，利用因果知识来指导探索。本文将该该方法扩展至部分已知的因果结构、来自其他环境的信息以及可实现的反事实行动。首先，我们在马尔可夫等价性下发展了最小干预集的图判别准则，并刻画了极大祖先图中可能最优的最小干预集；提出了一种搜索算法，用以找到在偏祖先图所表示的至少一个图中可能最优的干预集。其次，我们通过部分可迁移性从源分布推导出目标奖励的边界，并考虑了不可操纵变量；支配关系为动作剪枝和受限探索提供了进一步的边界，并在有效先验边界下具有遗憾保证。第三，我们在由指定实验能力所支持的行动上构建了反事实结构因果老虎机。

    arXiv:2611.14066v1 Announce Type: new  Abstract: Structural causal bandits use causal knowledge to guide exploration when reward mechanisms are unknown. This paper extends the approach to partially known causal structure, information from other environments, and realizable counterfactual actions. First, we develop graphical criteria for minimal intervention sets under Markov equivalence and characterize possibly-optimal minimal intervention sets for maximal ancestral graphs. A search algorithm finds sets that are possibly optimal in at least one graph represented by a partial ancestral graph. Second, we derive target reward bounds from source distributions through partial transportability, accounting for non-manipulable variables. Dominance relations provide further bounds for action pruning and restricted exploration, with a regret guarantee under valid prior bounds. Third, we formulate counterfactual structural causal bandits over actions supported by specified experimental capabilit
    
[^61]: 视觉语言模型从哪里获得左右方位？Qwen2.5-VL、Qwen3-VL与InternVL3中的位置编码反事实研究

    Where Do Vision-Language Models Get Left and Right? Position-Code Counterfactuals in Qwen2.5-VL, Qwen3-VL and InternVL3

    [https://arxiv.org/abs/2611.14060](https://arxiv.org/abs/2611.14060)

    该研究通过位置编码反事实实验发现，视觉语言模型的左右方位判断主要由ViT位置编码决定——仅镜像ViT宽度位置即可在84-93%的问题上复现翻转图像的效果，而LLM侧的位置信息影响甚微。

    

    视觉语言模型的左右方位判断可能来源于图像块内内容的朝向、视觉Transformer（ViT）的位置编码，或语言模型（LLM）中词元的位置与顺序。消融实验表明ViT位置编码是必要的；我们通过仅镜像位置编码来检验其充分性。在Qwen2.5-VL-7B、Qwen3-VL-4B和InternVL3-2B中，仅镜像ViT宽度方向的位置编码，就能在CV-Bench左右方位问题上以84.3-93.2%的比例复现翻转图像后的答案（Qwen2.5-VL-3B为87.5%，部分表现为“左”的回退偏差）。在二选一的设置下，这也可视为一种干扰；半格位移同样会翻转大多数答案。在ViT位置编码保持不变时，LLM侧的ID镜像和词元顺序镜像在三个模型系列中都只改变很少的左右答案（Qwen2.5-VL中的ID测试较弱）。在Qwen2.5-VL-7B中，该效应存在于全局注意力的ViT块中，且主要集中在最后一个块。在What'sUp数据集上，翻转图像内容和词元顺序（而非位置编码）会更多地改变上下方位的答案，而非左右方位的答案。

    arXiv:2611.14060v1 Announce Type: cross  Abstract: A vision-language model may take left-right direction from the orientation of image content within patches, its vision transformer's (ViT) position code, or token positions and order in its language model (LLM). Ablations show the ViT code is necessary; we test sufficiency by mirroring only a position code. In Qwen2.5-VL-7B, Qwen3-VL-4B and InternVL3-2B, mirroring ViT width positions reproduces the flipped-image answer on 84.3-93.2% of CV-Bench left-right questions (Qwen2.5-VL-3B: 87.5%, partly a left fallback). With two options, this also counts disruption; a half-grid shift flips most answers too. With the ViT code intact, LLM-side id and token-order mirrors change few left-right answers in all three series (a weak id test in Qwen2.5-VL). In Qwen2.5-VL-7B the effect sits in the global-attention ViT blocks, mostly the last. On What'sUp, flipping content and token order, not the position code, keeps more on/under than left-right answer
    
[^62]: HandsOnAid：构建并评估面向应急响应场景的离线急救助手

    HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios

    [https://arxiv.org/abs/2611.14059](https://arxiv.org/abs/2611.14059)

    本文构建了HandsOnAid——一个可在无信号、无救援环境中通过QLoRA微调的4位Gemma 2B模型实现的离线手机急救助手，并通过四个前沿大模型盲评验证了微调显著提升了急救建议的质量与安全性。

    

    紧急情况可能发生在没有手机信号且附近没有救护车的地方。在这种情况下，手机上的小型语言模型可能会有所帮助，但其建议必须安全且准确。我们构建了HandsOnAid，这是一个4位量化的Gemma 2B模型，通过QLoRA在FirstAidQA数据集（包含5,550个急救问答对的语料库）上进行了微调。我们比较了七种配置，包括基础模型、4位和8位微调模型，以及带有长度下限、安全门控或检索功能的版本。我们使用了一个包含41对问答的题库，其医学审核参考遵循ANZCOR指南，并采用严格的0-5分评分标准，包含十二项安全否决机制。四个前沿大语言模型评审（GPT-5.6 Sol、Claude Opus 5、DeepSeek V4 Pro、GLM-5.3）对所有答案进行了盲评，并使用45个隐藏对照组来检验评审的可靠性。微调使平均分数比基础模型提高了1.08，达到2.34。最佳配置得分为5分中的2.51分。

    arXiv:2611.14059v1 Announce Type: cross  Abstract: Emergencies can occur where there is no phone signal and no ambulance nearby. A small language model on a phone can be helpful in such situations, but its advice needs to be safe and accurate. We built HandsOnAid, a 4-bit Gemma 2B model fine-tuned with QLoRA on FirstAidQA dataset, a corpus of 5,550 first-aid question-answer pairs. We compared seven configurations, including the base, 4-bit and 8-bit fine-tuned models, and versions with a length floor, safety gate, or retrieval. We used a 41-pair question-answer bank with medically reviewed references that followed ANZCOR guidelines, and a strict 0-5 score rubric, with twelve safety overrides. Four frontier LLM judges (GPT-5.6 Sol, Claude Opus 5, DeepSeek V4 Pro, GLM-5.3) blindly scored every answer, with 45 hidden controls used to check judge reliability. Fine-tuning improved the mean score by +1.08 over the base model to 2.34. The best configuration scored 2.51 out of 5 which was the 
    
[^63]: 基于费舍尔几何的结构化置信度扰动对抗训练

    Structured Confidence Perturbation via Fisher Geometry for Adversarial Training

    [https://arxiv.org/abs/2611.14046](https://arxiv.org/abs/2611.14046)

    该论文针对对抗训练中的“伪攻击”问题，提出融合交叉熵与KL散度的混合目标，并基于费舍尔信息矩阵几何在置信度流形上生成最优扰动，理论证明最优方向为费舍尔信息矩阵主特征向量，同时给出避免高维矩阵计算的高效算法。

    

    对抗训练常常受到“伪攻击”的困扰，这类扰动虽能显著增加损失值，却无法真正改变模型的置信度结构。为解决这一问题，我们提出了两种互补的方法，可直接攻击并防御置信度结构的变形。首先，在传统损失最大化视角的基础上，我们引入了一种融合交叉熵与KL散度的混合目标函数，促使对抗样本同时破坏预测准确性与分布一致性。其次，我们从几何角度重构对抗训练，通过生成在置信度流形上最大化分布距离的扰动来实施攻击。理论分析表明，最优扰动方向与费舍尔信息矩阵的主特征向量对齐。我们还提出了一种高效算法，以避免高维矩阵计算。

    arXiv:2611.14046v1 Announce Type: new  Abstract: Adversarial training is often plagued by "pseudo-attacks", perturbations that significantly increase the loss value but fail to truly alter the model's confidence structure. To address this, we propose two complementary approaches to directly attack and defend against confidence structure deformation. First, building upon the traditional loss maximization perspective, we introduce a hybrid objective that integrates cross-entropy and KL divergence, encouraging adversarial examples to simultaneously disrupt prediction accuracy and distribution consistency. Second, we geometrically reconstruct adversarial training by generating perturbations that maximize the distribution distance on the confidence manifold. Theoretical analysis shows that the optimal perturbation direction aligns with the principal eigenvectors of the Fisher information matrix. We also propose an efficient algorithm to avoid high-dimensional matrix computations. Experiment
    
[^64]: ScaleBrain：面向脑电图基础模型的尺度一致性框架

    ScaleBrain: A Scale-Consistent Framework for EEG Foundation Models

    [https://arxiv.org/abs/2611.14035](https://arxiv.org/abs/2611.14035)

    提出了ScaleBrain——一个尺度一致的EEG基础模型预训练框架，通过在模型输入与优化目标两个层面显式协调多尺度表示，解决了现有方法中特征与原始信号尺度特性对应关系隐式、部分时间尺度监督不足的问题。

    

    在大规模无标注脑电图（EEG）语料上预训练的基础模型，已成为脑机接口（BCI）应用的一种变革性范式。近期的EEG基础模型表明，从固定长度片段中提取的多尺度表示能够改善EEG的表示学习。现有框架通常通过特征层面的操作来构建时间尺度，然而这些操作使得学习到的特征与原始EEG信号尺度特定特性之间的对应关系变得隐式。此外，当自监督目标与多尺度输入表示不一致时，某些时间尺度的信息在预训练过程中可能得不到充分的监督。有鉴于此，我们提出了ScaleBrain，一个尺度一致的预训练框架，从模型输入和优化目标两个层面显式协调EEG的多尺度表示……

    arXiv:2611.14035v1 Announce Type: new  Abstract: Foundation models pre-trained on large unlabeled electroencephalography (EEG) corpora have emerged as a transformative paradigm for Brain-Computer Interface (BCI) applications. Recent EEG foundation models have shown that multi-scale representations derived from fixed-length patches can improve EEG representation learning. Existing frameworks commonly construct temporal scales through feature-level operations. However, these operations leave the correspondence between learned features and scale-specific characteristics of the original EEG signals implicit. Moreover, when the self-supervised objective is not aligned with the multi-scale input representations, information at some temporal scales may receive insufficient supervision during pre-training. In light of this, we propose ScaleBrain, a scale-consistent pre-training framework that explicitly coordinates multi-scale representations of EEG from both the model input and optimization o
    
[^65]: 边缘引导的Token信用分配用于可验证奖励的强化学习

    Margin-Guided Token Credit for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2611.14015](https://arxiv.org/abs/2611.14015)

    提出MCPO方法，利用策略下一Token分布的前两名概率差识别模型犹豫的Token位置，将奖励信用有针对性地集中于这些位置，且无需额外前向传播或批次统计，在六个竞赛数学基准上平均超越GRPO基线6.55分。

    

    在可验证奖励的强化学习（RLVR）中，语言模型因产生正确的最终答案而获得奖励，而GRPO会将该奖励均等地传递给解法中的每一个Token。我们转而将信用分配到模型犹豫不决的位置，并将该方法命名为MCPO（Margin-Credit Policy Optimization，边缘信用策略优化）。我们通过策略下一Token分布的前两名概率差 $M=p_1-p_2$ 来衡量犹豫程度，用两个固定阈值将其映射为分数，并将每个解的优势（advantage）分配到其各个Token上，同时保持均值不变。该边缘值可以直接从训练器已经执行的前向传播中读取，因此该方法无需额外的模型评估，也不需要批次统计信息。我们使用LoRA在2,048个DAPO-Math问题上对MiniCPM5-2B-Base进行200步训练，并在六个竞赛数学基准上进行评估。在全部五个数据种子下，MCPO均优于未修改的GRPO基线，macro avg@32平均提升6.55分（95%置信区间 $[4.1, 9.0]$），并在Qwen3-4B-Base的复现实验中（原文在此截断）……

    arXiv:2611.14015v1 Announce Type: cross  Abstract: In reinforcement learning with verifiable rewards, a language model is rewarded for a correct final answer, and GRPO passes that reward to every token of the solution in equal measure. We instead place the credit where the model hesitated, in a method we call MCPO (Margin-Credit Policy Optimization). We measure hesitation by the top-2 margin $M=p_1-p_2$ of the policy's next-token distribution, map it to a score with two fixed thresholds, and spread each solution's advantage over its tokens so that the mean stays unchanged. The margin is read from a forward pass the trainer already performs, so the method needs no extra model evaluation and no batch statistics. We train MiniCPM5-2B-Base with LoRA for 200 steps on 2,048 DAPO-Math problems and evaluate on six competition-math benchmarks. MCPO beats an unmodified GRPO baseline for all five data seeds, by 6.55 points of macro avg@32 on average (95\% CI $[4.1, 9.0]$), and a Qwen3-4B-Base rep
    
[^66]: 编程智能体记忆后训练 II：通过在线策略蒸馏释放预训练文件操作在长时程任务中的记忆潜力

    Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation

    [https://arxiv.org/abs/2611.14009](https://arxiv.org/abs/2611.14009)

    该论文提出 CAMG-II 长时程智能体强化学习环境套件，通过提供 Bash 访问和回合持久工作区，让记忆控制建立在预训练已有的文件操作能力之上，从而解决在线策略蒸馏中因记忆接口超出预训练范围而导致教师指导无效、学生记忆动作采样不足的问题。

    

    语言模型智能体越来越多地处理交互历史超出模型活跃上下文的长时程任务。因此，它们必须决定记下什么、读回什么，以及何时执行这两种操作。近期的工作开始使用在线策略蒸馏将记忆控制纳入策略之中，但其依赖的记忆工具和压缩协议都不在教师和学生模型的预训练范围之内。然而，这从两个方面削弱了蒸馏效果：教师模型无法就其从未使用过的接口给出有用的指导，而学生模型采样到的有意义的记忆动作太少，使得这些指导难以发挥作用。为了研究这一问题，我们推出了 Coding Agent Memory Gym II（CAMG-II），这是一套涵盖购物、编程、自动研究、通用和视觉领域的长时程智能体强化学习环境套件。在每个环境的原生任务接口之外，CAMG-II 还提供普通的 Bash 访问权限和一个回合内持久的工作区，从而能够……

    arXiv:2611.14009v1 Announce Type: cross  Abstract: Language-model agents increasingly tackle long-horizon tasks whose interaction histories exceed the model's active context. They must therefore decide what to write down and what to read back, and when to do either. Recent work has begun to use on-policy distillation (OPD) to make memory control part of the policy, relying on memory tools and compaction protocols that lie outside either model's pre-training. However, this weakens the distillation on both sides: the teacher cannot give useful guidance on an interface it has never used, and the student samples too few meaningful memory actions for that guidance to take effect. To study this problem, we introduce Coding Agent Memory Gym II (CAMG-II), a suite of long-horizon agentic-RL environments spanning Shop, Coding, AutoResearch, General, and Visual. Alongside each environment's native task interface, CAMG-II provides ordinary Bash access and an episode-persistent workspace, enabling 
    
[^67]: 相空间中量子电路经典模拟的归一化流方法

    Normalizing Flows for Classical Simulation of Quantum Circuits in Phase Space

    [https://arxiv.org/abs/2611.14006](https://arxiv.org/abs/2611.14006)

    该论文提出基于多体系统相空间表述的归一化流框架，将酉门参数化为概率分布之间的流变换，实现了在大系统规模下精度超越现有最先进神经网络方法的量子电路经典模拟。

    

    我们提出了一种新颖的框架，基于多体系统的相空间表述来学习量子电路的表示。鉴于N比特量子态可以表示为光滑的2N维流形上的概率分布，我们展示了如何将酉门参数化为这些分布之间的归一化流，为使用生成模型进行量子电路模拟开辟了一条新途径。我们在三种常见的量子电路族上评估了我们的方法，结果表明在大系统规模下，其精度超过了当前最先进的神经网络方法。

    arXiv:2611.14006v1 Announce Type: cross  Abstract: We present a novel framework to learn a representation of quantum circuits based on the phase space formulation of many-body systems. Given that $N$-qubit states can be represented as probability distributions on a smooth $2N$-dimensional manifold, we show how unitary gates can be parameterized as normalizing flows between these distributions, opening a new avenue for quantum circuit simulation using generative models. We evaluate our method on three common families of quantum circuits, and show that its accuracy exceeds that of current state-of-the-art neural network methods for large system sizes.
    
[^68]: LUMIR：面向下一兴趣点推荐的异构数据解耦多模态表示学习

    LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation

    [https://arxiv.org/abs/2611.14004](https://arxiv.org/abs/2611.14004)

    提出解耦多模态表示学习框架LUMIR，利用大语言模型将异构多模态数据统一到共享文本空间，并通过对比学习结合交互感知融合与序列信号约束进行训练，使学习到的表示与下游推荐模型解耦，提升下一兴趣点推荐中多模态表示的通用性和可重用性。

    

    随着从异构数据源收集的多模态旅游相关信息日益丰富，学习有效的表示以提升推荐性能受到了广泛关注。现有方法通常将多模态融合与特定下游模型的架构和优化目标紧密耦合，这限制了所学表示的通用性和可重用性。为解决这些局限性，我们提出了LUMIR，一个基于对比学习的解耦多模态表示学习框架。首先利用大语言模型将从异构数据源收集的多模态信息在语义上统一到共享的文本表示空间中，随后设计交互感知的融合机制来建模跨模态交互，同时将序列信号作为结构约束纳入以指导表示学习。为评估（摘要在此处截断）

    arXiv:2611.14004v1 Announce Type: cross  Abstract: As multimodal tourism-related information collected from heterogeneous data sources has become increasingly available, learning effective representations has attracted attention for improving recommendation performance. Existing approaches often tightly couple multimodal fusion with the architecture and optimization objective of a specific downstream model, which limits the generality and reusability of the learned representations. To address these limitations, we propose LUMIR, a decoupled multimodal representation learning framework based on contrastive learning. Multimodal information collected from heterogeneous data sources is first semantically unified into a shared textual representation space using large language models (LLMs). An interaction-aware fusion mechanism is then designed to model cross-modal interactions, while sequential signals are incorporated as a structural constraint to guide representation learning. To evaluat
    
[^69]: 自主指定隐私：评估大型视觉语言模型的推理时个性化隐私保护能力

    Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models

    [https://arxiv.org/abs/2611.13997](https://arxiv.org/abs/2611.13997)

    该论文提出了首个系统性评估大型视觉语言模型“推理时个性化隐私保护”（ITP³）能力的基准SPY-Bench，包含32,700个涵盖67个类别和24个真实场景的样本以及基于用户规范与情境感知的新颖指标，旨在使模型能够遵循用户通过语言动态指定的个性化隐私边界。

    

    大型视觉语言模型（LVLMs）已展现出卓越的能力，但由于其能够从图像中以高精度推断敏感个人信息，引发了严重的隐私担忧。虽然当前的LVLMs在保护通用隐私（如信用卡数据）方面已经相对对齐良好，但我们认为隐私本质上是个性化和情境相关的。本工作转向一个新颖的任务：LVLMs能否实现推理时个性化隐私保护（ITP³），即允许用户通过语言规范动态指定隐私边界？为此，我们提出了SPY-Bench，这是首个对ITP³能力进行系统性评估的基准，包含：（1）32,700个独特样本，涵盖67个类别和24个真实世界场景中的图像-问题对和个性化隐私指令；（2）基于用户规范和情境感知的新颖评估指标。（注：原文摘要在此处不完整）

    arXiv:2611.13997v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities but raise significant privacy concerns due to their abilities to infer sensitive personal information from images with high precision. While current LVLMs are relatively well aligned to protect universal privacy, e.g., credit card data, we argue that privacy is inherently personalized and context-dependent. This work pivots towards a novel task: can LVLMs achieve Inference-Time Personalized Privacy Protection (ITP$^3$), allowing users to dynamically specify privacy boundaries through language specifications? To this end, we present SPY-Bench, the first systematic assessment of ITP$^3$ ability, which comprises (1) 32,700 unique samples with image-question pairs and personalized privacy instructions across 67 categories and 24 real-world scenarios, and (2) novel metrics grounded in user specifications and context awareness. Benchmarking the ITP$^3$ ability of 
    
[^70]: BioXArena-Pro：面向自主生物医学机器学习智能体的研究级基准

    BioXArena-Pro: a research-grade benchmark for autonomous biomedical machine learning agents

    [https://arxiv.org/abs/2611.13994](https://arxiv.org/abs/2611.13994)

    本文提出BioXArena-Pro基准，包含8个生物医学领域64个经AI辅助审计与人工审查的研究级建模任务，评估显示最佳自主编码智能体仅得55.1/100分，智能体框架与模型同等重要，且智能体通过适应新数据可在重建任务上超越已发表方法8.2分。

    

    编码智能体越来越多地被提议作为生物医学模型的自主构建者，但现有基准规模小、接近饱和或容易受到信息泄露的影响。我们提出了BioXArena-Pro，这是一个包含来自八个生物医学领域的64个研究级建模任务的基准，每个任务均从已发表的研究中重建而来；大多数任务将完整的生物学单元（如物种或队列）整体保留为测试集。在22种智能体配置中，最佳成绩为100分中的55.1分，且智能体框架的重要性几乎与模型本身相当。我们对每个任务重新运行了已发表的方法：在已发表方法可直接适用的情况下，最佳智能体与这些方法打平；而在重建的任务上，这些方法落后8.2分，这表明智能体通过适应新数据而获益。预训练模型（包括生物基础模型）在最强智能体使用它们的任务上很关键：禁止使用会使该智能体在这些任务上的分数下降9分。每个任务都经过多轮AI辅助审计和最终的人工审查。

    arXiv:2611.13994v1 Announce Type: new  Abstract: Coding agents are increasingly proposed as autonomous builders of biomedical models, but existing benchmarks are small, near saturation or open to leaked information. We present BioXArena-Pro, 64 research-level modelling tasks from eight biomedical domains, each rebuilt from a published study; most hold out whole biological units such as species or cohorts. Across 22 agent configurations, the best scores 55.1 of 100, and the scaffold matters almost as much as the model. We re-ran a published method for every task: where the published recipe applies unchanged, the best agent and the methods tie; on rebuilt tasks the methods trail by 8.2 points, suggesting that agents gain by adapting to new data. Pretrained models (including biological foundation models) matter where the strongest agent uses them: forbidding them decreases its score on those tasks by 9 points. Every task was audited in several AI-assisted rounds and a final human review.
    
[^71]: 面向强化学习的时间行为树：规范作为奖励、课程与诊断工具

    Temporal Behavior Trees for Reinforcement Learning: Specifications as Rewards, Curricula, and Diagnostics

    [https://arxiv.org/abs/2611.13993](https://arxiv.org/abs/2611.13993)

    该论文提出TBTCL框架，以时间行为树（TBT）作为统一接口，使单一规范同时承担任务定义、密集鲁棒性奖励生成、课程设计与策略诊断四种功能，其表达力优于LTL且天然具有层次结构。

    

    基于形式化规范的强化学习通常将任务定义、奖励塑形、课程设计与策略诊断分离为不同的数学对象。这种碎片化不仅需要额外的转换步骤，还会丢弃规范中已经存在的结构，例如子任务边界、顺序约束和定量度量。我们提出了TBTCL（时间行为树中心学习）框架，该框架将时间行为树（TBT）用作规范引导强化学习的统一接口。单个TBT即可指定任务、产生基于鲁棒性的密集奖励、定义课程，并支持对学习到的策略进行更好的理解。通过为规范引导强化学习适配TBT的语法与语义，我们提供了一种比LTL等流行规范语言更具表达力且天然层次化的形式化方法。我们提出了自动……（原文摘要在此处截断）

    arXiv:2611.13993v1 Announce Type: new  Abstract: Reinforcement learning from formal specifications often separates task definition, reward shaping, curriculum design, and policy diagnostics into different mathematical objects. This fragmentation requires additional conversion steps and discards structure already present in the specification, such as subtask boundaries, ordering constraints, and quantitative measures. We introduce TBTCL (Temporal Behavior Tree Centric Learning), a framework that uses Temporal Behavior Trees (TBTs) as a unified interface for specification-guided reinforcement learning. A single TBT specifies the task, induces dense robustness-based rewards, defines a curriculum, and supports better understanding of learned policies. By adapting TBT syntax and semantics for specification-guided reinforcement learning, we provide a formalism that is more expressive than popular specification languages such as LTL and inherently hierarchical. We present algorithms to automa
    
[^72]: 脉搏还是来源？基于rPPG的深度伪造检测器究竟学到了什么

    Pulse or Provenance? What rPPG-Based Deepfake Detectors Learn

    [https://arxiv.org/abs/2611.13992](https://arxiv.org/abs/2611.13992)

    研究通过系统实验揭示，基于rPPG的深度伪造检测器学到的并非真实的生理心脏信号，而是与视频编码等来源相关的非生理伪影，这挑战了rPPG作为深度伪造检测生理证据的基本前提。

    

    远程光电容积描记法（rPPG）已被提出作为深度伪造检测的一种生理线索，其前提是合成人脸缺乏真实的心脏信号。我们在Celeb-DF++（包含22种伪造方法）和FaceForensics++（FF++）上检验了基于rPPG风格的检测器究竟学到了什么，对每个真实视频和伪造视频均使用一个相同采样的窗口。在说话人脸伪造上，简单的色度信号轨迹达到了0.998的AUC，超过了专用的rPPG提取器（0.87）。受控实验表明，这种信号并非生理来源。在完全合成的伪造上，当从背景区域或未进行人脸检测的完整帧中提取时，该信号同样具有判别力，且其能量集中在0.7至4 Hz心率频带之上。打乱帧顺序会在很大程度上消除这种信号，但在对分辨率、帧率和编码进行归一化后它依然存在，并且一项独立的重新实现复现了这些发现。

    arXiv:2611.13992v1 Announce Type: cross  Abstract: Remote photoplethysmography (rPPG) has been proposed as a physiological cue for deepfake detection, based on the premise that synthesized faces lack a genuine cardiac signal. We examine what rPPG-style detectors learn on Celeb-DF++ (22 forgery methods) and FaceForensics++ (FF++), using one identically sampled window per real and fake video. On talking-face forgeries, simple chrominance traces reach an AUC of 0.998, exceeding a dedicated rPPG extractor (0.87). Controlled experiments indicate that this signal is not physiological in origin. On fully synthesized forgeries, it is equally discriminative when extracted from background regions or from full frames without face detection, and it is concentrated above the 0.7 to 4 Hz heart-rate band. It is largely removed by shuffling frame order but persists after normalizing resolution, frame rate and encoding, and an independent reimplementation reproduces these findings. Removing high-freque
    
[^73]: 亚辛神经算子的万能逼近定理

    Universal Approximation Theorem for Metaplectic Neural Operators

    [https://arxiv.org/abs/2611.13991](https://arxiv.org/abs/2611.13991)

    该论文证明了亚辛神经算子的万能逼近定理：在非空有界Lipschitz区域上，仅需一个带单条非局部亚辛乘子分支的隐藏算子层，即可对从C^s到C^r以及H^s到H^r的连续算子实现紧一致逼近。

    

    我们在非空有界 Lipschitz 区域上证明了亚辛神经算子的万能逼近性质。其非局部分支是带零延拓与零限制的全空间亚辛乘子，可与任意固定的自由辛矩阵相关联。在激活函数光滑、全局 Lipschitz 且非多项式的假设下，我们得到了从 $C^s$ 到 $C^r$ 以及从 $H^s$ 到 $H^r$ 的连续算子在相应输出范数下的紧一致逼近，其中 $s,r$ 为任意非负整数，输入与输出通道维数均为有限。在 Sobolev 情形下，输入紧集还需额外在 $L^\infty$ 中一致有界。仅需一个隐藏算子层及其中的单条非局部分支即可充分实现逼近。单个标量符号原子与有限多个位置特征由变换和区域预先确定，与目标算子、紧集及逼近精度无关。证明将平均化神经算子理论与精确的 chirp——（原文摘要在此处截断）

    arXiv:2611.13991v1 Announce Type: new  Abstract: We prove universal approximation for metaplectic neural operators on nonempty bounded Lipschitz domains. The nonlocal branches are whole-space metaplectic multipliers with zero extension and restriction, associated with any fixed free symplectic matrix. Assuming a smooth, globally Lipschitz, nonpolynomial activation, we obtain compact-uniform approximation of continuous operators from $C^s$ to $C^r$ and from $H^s$ to $H^r$ in the respective output norms, for arbitrary nonnegative integers $s,r$ and finite input and output channel dimensions. In the Sobolev setting, compact input sets must additionally be uniformly bounded in $L^\infty$. One hidden operator layer with one nonlocal branch suffices. A single scalar symbol atom and finitely many positional features are prescribed by the transform and domain independently of the target operator, compact set and accuracy. The proof combines averaging neural operator theory with an exact chirp-
    
[^74]: 通过线性与仿射子空间约束实现梯度反演

    Gradient Inversion through Linear and Affine Subspace Constraints

    [https://arxiv.org/abs/2611.13976](https://arxiv.org/abs/2611.13976)

    本文发现全连接层的聚合梯度在正确缩放下是其输入的仿射组合而非仅线性组合，并利用线性或仿射子空间约束改进联邦学习中的梯度反演攻击，从而提升数据重建精度。

    

    联邦学习使多个客户端能够在不直接共享本地数据集的情况下协同训练机器学习模型。然而，正如先前的研究所表明的，训练过程中交换的梯度可能会泄露客户端数据中的敏感信息。现有的大多数攻击方法通常通过优化虚拟输入来匹配客户端的真实梯度（梯度反演攻击），而另一些方法则基于全连接（FC）层的聚合梯度是其输入的线性组合这一观察来尝试重建数据。在本工作中，我们扩展了这一观察，证明经过正确缩放的全连接层聚合梯度实际上是其输入的仿射组合。我们评估了一种梯度反演攻击，该攻击在每个全连接层利用已知的线性子空间信息或其对应的仿射子空间信息来提高重建精度。我们的结果表明，基于子空间的方法能够……

    arXiv:2611.13976v1 Announce Type: cross  Abstract: Federated learning enables multiple clients to collaboratively train a machine-learning model without directly sharing their local datasets. However, as prior works have shown, the gradients exchanged during training may reveal sensitive information about client data. Most existing attacks usually optimize dummy inputs to match the true client gradients (Gradient Inversion Attacks), while some attempt to reconstruct them based on the observation that aggregate gradients from a fully connected (FC) layer are a linear combination of its inputs. In this work, we extend this observation by showing that aggregate gradients from an FC layer with the correct scaling are actually an affine combination of its inputs. We evaluate a gradient inversion attack that leverages either known linear-subspace information or its affine-subspace counterpart at each FC layer to improve reconstruction accuracy. Our results show that subspace-based methods ca
    
[^75]: 论VARMA过程的谱估计

    On the Spectral Estimation of VARMA Processes

    [https://arxiv.org/abs/2611.13973](https://arxiv.org/abs/2611.13973)

    本文提出了一种基于传输熵概念的新矩问题，能够在不固定移动平均分量或不将其限制为标量的情况下，学习VARMA过程的功率谱密度。

    

    我们考虑估计向量自回归移动平均（VARMA）过程的功率谱密度的问题。完成这一任务的一种成熟方法是构建矩问题，在匹配底层过程的某些矩的同时，选择使熵类指标最大化的谱密度。然而，现有方法假设移动平均（MA）分量是固定的或标量的，从而缩小了寻求最优谱密度的ARMA模型类别。在本文中，我们基于传输熵的概念，提出了一种新的矩问题，用于学习VARMA过程的谱密度，而无需固定MA分量或将其限制为标量。

    arXiv:2611.13973v1 Announce Type: cross  Abstract: We consider the problem of estimating the power spectral density of a Vector-valued Autoregressive Moving-Average (VARMA) process. A well-established approach to accomplishing this task consists of formulating a moment problem that selects the spectral density maximizing an entropy-like index while matching certain moments of the underlying process. However, existing methods assume that the moving-average (MA) component is either fixed or scalar, thereby reducing the class of ARMA models over which the optimal spectral density is sought. In this paper, we propose a new moment problem, based on the notion of transportation entropy, for learning the spectral density of a VARMA process without fixing the MA component or restricting it to be scalar.
    
[^76]: 多模态旅游推荐系统中的数据质量挑战

    Data Quality Challenges in Multimodal Tourism Recommender Systems

    [https://arxiv.org/abs/2611.13972](https://arxiv.org/abs/2611.13972)

    该论文聚焦旅游推荐场景，揭示了使用Yelp和Instagram等真实多模态数据时面临的虚假评论噪声、LLM幻觉内容、模态矛盾和人口偏差等数据质量挑战，并提出质量感知融合、受控LLM内容生成和用户群体差异化等未来改进方向。

    

    现代推荐系统日益考虑多样化的数据模态以提升模型性能。然而，底层多模态数据（例如来自社交媒体平台的数据）的质量迄今为止受到的关注有限。本文聚焦于旅游推荐领域，检验了在使用Yelp和Instagram数据集时遇到的现实世界挑战。这些问题包括虚假评论带来的噪声、大语言模型（LLM）产生的幻觉内容、模态间的矛盾以及人口统计学偏差。我们概述了未来的研究方向，例如质量感知融合、结构化和受控的基于LLM的内容生成以减少幻觉，以及用户群体差异化，以提高多模态推荐系统的鲁棒性和可靠性。

    arXiv:2611.13972v1 Announce Type: cross  Abstract: Modern recommender systems increasingly consider diverse data modalities to enhance model performance. However, the quality of the underlying multimodal data (e.g. data from social media platforms) has received limited attention so far. Focusing on tourism recommendation, we examine real-world challenges encountered when using datasets from Yelp and Instagram. Issues include noise from fake reviews, hallucinated content from LLMs, modality contradictions, and demographic bias. We outline future directions such as quality-aware fusion, structured and controlled LLM-based content generation to reduce hallucinations, and user group differentiation to improve the robustness and reliability of multimodal recommendation systems.
    
[^77]: 视觉-语言医学图像分割中标注派生提示所致范围泄漏的反事实分析

    Counterfactual Analysis of Extent Leakage from Annotation-Derived Prompts in Vision-Language Medical Image Segmentation

    [https://arxiv.org/abs/2611.13970](https://arxiv.org/abs/2611.13970)

    该论文通过反事实分析揭示了医学图像分割基准文本提示中的“范围泄漏”问题：标注派生提示包含从真值掩膜计算出的病灶大小、数量和位置信息，模型的性能提升主要依赖这种泄漏的范围信息，而非理解描述中的其他语义内容。

    

    文本引导的分割模型接收一张图像和一段关于病灶的简短文本描述，在公开基准数据集上，该描述能显著提升分割精度。然而，在结肠镜和乳腺超声数据集中，每条描述中的大小、数量和位置词语都是从随后用于评分预测结果的真值掩膜中计算得出的，而胸部X光的描述则由基准数据集的标注员撰写。我们测试了两个已发表模型从这些描述中究竟获取了什么信息。在结肠镜和胸部X光任务中，模型主要获取的是病灶的范围信息：仅保留大小词语或肺野区域短语，就能恢复相对空描述所带来提升的84%至98%，而移除这些信息则会损失52%至77%的提升；描述中的其余内容影响甚微，即使把息肉称作“石头”，分割效果也同样好。我们将这种现象称为“范围泄漏”。在乳腺超声任务中，近半数描述还包含诊断名称，这部分信息也贡献了一部分收益。在三个数据集上的重新训练表明，模型会从其（训练数据的标注方式中）习得这种习惯……（原文摘要在此处截断）

    arXiv:2611.13970v1 Announce Type: cross  Abstract: Text-guided segmentation models receive an image and a short caption about the lesion, and on public benchmarks the caption improves accuracy considerably. On colonoscopy and breast ultrasound, however, the size, count and location words of each caption were computed from the mask that later scores the prediction, and the chest X-ray captions were written by benchmark annotators. We tested what two published models take from them. On colonoscopy and chest X-ray, it is mostly the lesion's extent. Keeping only the size word or the lung-zone phrase recovers 84 to 98 percent of the improvement over an empty caption, and removing it loses 52 to 77 percent. The rest matters little, and a polyp called a rock is segmented just as well. We call this extent leakage. On breast ultrasound, nearly half the captions also name the diagnosis, which carries part of the benefit. Retraining on three datasets shows that models pick up this habit from thei
    
[^78]: RB-LDC：面向噪声鲁棒扩散潜变量的冗余平衡潜编码

    RB-LDC: Redundancy-Balanced Latent Coding for Noise-Robust Diffusion Latents

    [https://arxiv.org/abs/2611.13955](https://arxiv.org/abs/2611.13955)

    提出RB-LDC方法，通过在编码器与扩散过程之间加入可逆线性层，按解码器敏感性对各潜变量方向的对数信噪比进行冗余平衡式的最优再分配，从而提升扩散模型潜变量的噪声鲁棒性。

    

    潜扩散模型所解码的往往是近似正确的潜变量，因此解码器对扰动的敏感性与图像质量密切相关。在白化之后，尽管解码器对不同方向的敏感性并不相同，前向扩散过程却以相同的信噪比（SNR）破坏每个方向。我们研究在编码器与扩散过程之间引入一个可逆线性层，用以改变每个方向的对数信噪比。对于任意白化的潜变量分布，其线性最小均方误差（LMMSE）恢复风险具有精确的谱表达式。在能量预算以及保证可逆性的下限约束下，最小化在扩散调度上积分的解码器敏感性加权LMMSE风险，可得到唯一的最优分配方案。该线性层与解码器的期望回拉度量对齐，并使各方向上的边际风险降低相等化，整个计算由确定性谱求解器完成。在大预算情形下，该分配遵循关于敏感性的幂律，其指数……（原文摘要在此处截断）

    arXiv:2611.13955v1 Announce Type: new  Abstract: Latent diffusion models decode approximately correct latents, making decoder sensitivity to perturbations relevant to image quality. After whitening, forward diffusion corrupts every direction at the same signal-to-noise ratio (SNR), despite unequal decoder sensitivities. We study an invertible linear layer between the encoder and diffusion process that shifts each direction's log-SNR. Its linear minimum mean-square-error (LMMSE) recovery risk has an exact spectral expression for any whitened latent distribution. Under an energy budget and a floor ensuring invertibility, minimizing decoder-sensitivity-weighted LMMSE risk integrated over the diffusion schedule yields a unique optimal allocation. The layer aligns with the decoder's expected pullback metric and equalizes marginal risk reduction across directions, computed by a deterministic spectral solver. At large budgets, the allocation follows a power law in sensitivity whose exponent i
    
[^79]: 有界欧几里得域上浅层ReLU$^k$网络的几何依赖逼近与稳定性性质

    Geometry-Dependent Approximation and Stability Properties of Shallow ReLU$^k$ Networks on Bounded Euclidean Domains

    [https://arxiv.org/abs/2611.13952](https://arxiv.org/abs/2611.13952)

    本文揭示了有界欧氏域上浅层ReLU$^k$网络的逼近精度与稳定性由隐藏参数几何（填充距离与对径分离距离）定量决定，并给出了逼近上界、饱和型下界以及矩阵谱和最小二乘恢复等系统的稳定性分析。

    

    在有界欧几里得域上，我们通过逼近速率、饱和型下界、Gram矩阵与刚度矩阵的谱、Bernstein不等式与Marcinkiewicz-Zygmund不等式、离散最小二乘恢复以及逆逼近等结果，研究隐藏参数几何如何影响浅层ReLU$^k$网络。我们的分析采用光滑局部化技术和二进傅里叶估计。我们用填充距离 $h$ 和对径分离距离 $\underline h$ 来量化参数几何。对于给定的网络空间 $L_n^k$，代表性的逼近估计形式为：上界 $\inf_{g_n\in L_n^k}\|f-g_n\|_{\mathcal L^2(\Omega')}\lesssim h^r\|f\|_{\mathcal H^r(\Omega)}$ 利用目标函数直至临界阶数 $\frac{d+2...}$ 的正则性（摘要在此处被截断）。

    arXiv:2611.13952v1 Announce Type: new  Abstract: On bounded Euclidean domains, we study how hidden-parameter geometry affects shallow ReLU$^k$ networks through results on approximation rates, saturation-type lower bounds, Gram and stiffness matrix spectra, Bernstein and Marcinkiewicz--Zygmund inequalities, discrete least-squares recovery, and inverse approximation.   Our analysis uses smooth localization and dyadic Fourier estimates. We quantify parameter geometry by the fill distance $h$ and antipodal separation distance $\underline h$. For the prescribed network space $L_n^k$, representative approximation estimates take the form \begin{equation*} \begin{aligned} \inf_{g_n\in L_n^k}\|f-g_n\|_{\mathcal L^2(\Omega')}&\lesssim h^r\|f\|_{\mathcal H^r(\Omega)},\\ \inf_{g_n\in L_n^k}\|f-g_n\|_{\mathcal L^2(\Omega)}&\gtrsim \underline h^{\frac{d+2k+1}{2}}\|f\|_{\mathcal L^2(\Omega')}. \end{aligned} \end{equation*} The upper bound exploits target regularity up to the critical order $\frac{d+2
    
[^80]: Chem World：一个用于可信分子性质预测的大规模基准与物理感知框架

    Chem World: A Large-Scale Benchmark and Physics-Aware Framework for Trustworthy Molecular Property Prediction

    [https://arxiv.org/abs/2611.13945](https://arxiv.org/abs/2611.13945)

    本文提出了大规模基准Chem World和物理感知组合建模框架PACM，通过整合17个数据集、80余万实例和领域物理约束，为可信的分子性质预测提供了统一的评估平台，并验证了分子语言模型表示的强泛化与可迁移性。

    

    分子性质预测是计算化学和药物制剂研究的基础，然而现有基准在数据集、性质和评估协议方面仍然碎片化，限制了对模型泛化能力的系统性评估。我们提出了Chem World，这是一个大规模基准，整合了17个公开数据集，涵盖超过80万个分子和化学体系实例，横跨10个性质轨道，覆盖单分子和多组分系统，包括药物溶解度。Chem World为跨多种化学性质评估分子表示和组合建模提供了统一的测试平台。基于该基准，我们进一步提出了PACM（物理感知组合建模），它将分子表示与组合感知建模和领域特定的物理约束相结合。大量实验表明，分子语言模型能够提供强大且可迁移的表示，

    arXiv:2611.13945v1 Announce Type: cross  Abstract: Molecular property prediction is fundamental to computational chemistry and drug formulation research, yet existing benchmarks remain fragmented across datasets, properties, and evaluation protocols, limiting systematic evaluation of model generalization. We introduce Chem World, a large-scale benchmark integrating 17 public datasets with over 800,000 molecular and chemical-system instances across 10 property tracks, covering single-molecule and multi-component systems, including drug solubility. Chem World provides a unified testbed for evaluating molecular representations and compositional modeling across diverse chemical properties. Building on the benchmark, we propose PACM (Physics-Aware Compositional Modeling), which combines molecular representations with composition-aware modeling and domain-specific physical constraints. Extensive experiments show that molecular language models provide strong and transferable representations, 
    
[^81]: PARDIG：从稀疏卫星观测中学习得到的全球降水估计

    PARDIG: Global Precipitation Estimates Learned from Sparse Satellite Observations

    [https://arxiv.org/abs/2611.13941](https://arxiv.org/abs/2611.13941)

    PARDIG是一个基于球形图-网格Transformer的人工智能框架，通过学习密集多模态观测数据来预测稀疏的高精度星载雷达降水测量，以0.1°空间分辨率和15分钟时间分辨率提供2016年至今的近实时全球降水估计。

    

    全球尺度的降水估计对地球系统科学和灾害响应至关重要，但观测覆盖范围和测量精度仍然受限。现有解决方案需要在理想特性之间进行权衡，例如预报技巧、时延、地理覆盖范围、时间覆盖范围以及空间和时间分辨率。我们提出了PARDIG，这是一个基于球形图-网格Transformer架构的人工智能框架，能够生成近实时的全球降水估计。PARDIG在密集的多模态输入上进行训练，包括大气状态分析场和多光谱静止卫星图像，以预测来自星载雷达的稀疏、高精度降水测量。在推理阶段，训练好的模型以0.1°空间分辨率和15分钟时间分辨率在全球范围内密集应用，生成自2016年至今的完整近实时降水场……

    arXiv:2611.13941v1 Announce Type: new  Abstract: Estimating precipitation at the global scale is vital for Earth system science and disaster response, yet observational coverage and measurement precision remain constrained. Existing solutions provide options that trade off desirable traits such as skill, latency, geographical coverage, temporal coverage, and spatial and temporal resolution. We present PARDIG, an artificial intelligence framework based on a spherical graph-mesh transformer architecture that produces near-real-time global precipitation estimates. PARDIG is trained on dense multi-modal inputs, including atmospheric state analyses and multi-spectral geostationary satellite imagery, to predict sparse, high-precision precipitation measurements from spaceborne radar. At inference, the trained model is applied densely across the globe at $0.1^\circ$ spatial and 15-minute temporal resolution, producing complete near-real-time precipitation fields from 2016 to the present, along
    
[^82]: 具有结构缓存加速的双核张量轮分解方法用于张量恢复

    Dual-Core Tensor Wheel Decomposition with Structure-Cached Acceleration for Tensor Recovery

    [https://arxiv.org/abs/2611.13938](https://arxiv.org/abs/2611.13938)

    该论文提出双核张量轮分解（DCTW）及结构缓存加速策略，通过将高阶中心核分解为两个由共享潜变量键连接的低阶核，并缓存张量网络收缩中的中间张量以避免显式构造大规模矩阵和冗余计算，从而大幅提升了张量恢复的计算与内存效率。

    

    张量轮分解通过外环因子与高阶中心核之间的相互作用来捕获高维数据中的复杂关联。然而，更新这些组件需要显式构造和操作大规模矩阵，这限制了计算效率，并对中心核的更新带来了尤其高昂的计算和内存开销。为解决这些局限，我们提出了双核张量轮分解（DCTW），将中心核分解为两个由共享潜变量键连接的低阶核，且每个核分别与不同的外环因子组相耦合。我们进一步提出了一种结构缓存加速策略，该策略避免了显式的大规模矩阵乘法，并将张量网络收缩过程中生成的中间张量缓存下来，以便在各块更新之间重复利用。这种重用减少了冗余计算，并避免了大规模矩阵的显式构造，从而显著提升了计算与内存效率。

    arXiv:2611.13938v1 Announce Type: new  Abstract: Tensor wheel decomposition captures complex correlations in high-dimensional data through interactions between outer-ring factors and a high-order central core. However, updating these components requires the explicit construction and manipulation of large-scale matrices, which limits computational efficiency and incurs particularly high computational and memory costs for the central-core update. To address these limitations, we propose dual-core tensor wheel decomposition (DCTW), which factorizes the central core into two lower-order cores linked by a shared latent bond. Each core is coupled to a different group of outer-ring factors. We further propose a structure-cached acceleration strategy that avoids explicit large-scale matrix multi- plications and caches intermediate tensors generated during tensor network contractions for reuse across block updates. This reuse reduces redundant computation and avoids the explicit construction of
    
[^83]: 变分不等式与求根问题的对偶空间预条件方法

    Dual-Space Preconditioning for Variational Inequalities and Root-Finding Problems

    [https://arxiv.org/abs/2611.13916](https://arxiv.org/abs/2611.13916)

    本文提出了一种超越经典余强制性和利普希茨连续性假设的对偶空间预条件框架，通过引入新颖的广义条件为变分不等式与求根问题的确定性和随机方法建立收敛性理论，并将算法解释为变步长方法且给出预条件流的变分与最优控制解释。

    

    本文针对变分不等式和求根问题，提出了一个超越经典余强制性（cocoercivity）和利普希茨连续性假设的对偶空间预条件框架。为此，我们引入了一种新颖的余强制性类型条件和一个放宽的利普希茨连续性条件，并在这些广义条件下推导出了确定性和随机方法的收敛性。对于各向同性预条件子，所得算法可解释为经典方法的变步长变体，扩展了近期基于 $(L_0, L_1)$ 型增长条件的分析，同时允许更简洁的证明和更大的步长范围。我们还证明了该框架能够通过弱 Minty 型论证自然地处理变分不等式。最后，我们研究了相关的连续时间动力学，并推导出预条件流的变分与最优控制解释。总体而言，结果表明对偶空间预条件方法（原文在此处截断）

    arXiv:2611.13916v1 Announce Type: cross  Abstract: This paper develops a dual-space preconditioning framework for variational inequalities and root-finding problems beyond classical cocoercivity and Lipschitz continuity assumptions. To this end, we introduce both a novel cocoercivity-type condition and a relaxed Lipschitz continuity condition, and derive convergence of deterministic and stochastic methods under these generalized conditions. For isotropic preconditioners, the resulting algorithms can be interpreted as variable-stepsize variants of classical methods, extending recent analyses based on $(L_0, L_1)$-type growth conditions while allowing simpler proofs and larger stepsize ranges. We also show that the framework naturally handles variational inequalities through weak Minty-type arguments. Finally, we study the associated continuous-time dynamics and derive a variational and optimal control interpretation of the preconditioned flow. Overall, the results show that dual-space p
    
[^84]: 《基于首次分歧的学习：联合重置与先导假设保证》

    Learning from First Discrepancies with Joint Reset and Pilot Guarantees

    [https://arxiv.org/abs/2611.13913](https://arxiv.org/abs/2611.13913)

    该论文研究了每次运行只揭示到首个错误决策为止信息的“首次分歧”学习模型，并证明当假设类（VC 维至多 $d$）的最优分歧概率至多为 $f$、最优风险至多为 $r$ 时，学习器仅需 $\widetilde O((d+1)[D(f+\varepsilon)+r]/\varepsilon^2)$ 次运行和 $\widetilde O(D(1+f/\varepsilon))$ 个先导假设，即可返回假设类成员及宽度至多 $\varepsilon$ 的风险区间，同时提供联合的重置与先导保证。

    

    一次在首个错误决策处停止的运行，会展示直到并包含该决策的查询及其正确标签，而此后的内容一概不可见。我们研究完成学习需要多少次这样的运行。每次运行在一 свеж新记录上测试一个事先从固定二元假设类（VC 维至多为 $d$）中选出的假设（称为“先导”），记录包含至多 $D$ 个查询，这些查询来自一个与先导无关的未知分布。只有当先导未产生任何分歧时，才展示记录的终止信息，并且一个事先固定的有界损失会附加在假设停止的位置上。若该类中成员的分歧概率的下确界至多为 $f$，且其风险的下确界至多为 $r$，则存在一个学习器，它在每个历史上使用 $\widetilde O((d+1)[D(f+\varepsilon)+r]/\varepsilon^2)$ 次运行和 $\widetilde O(D(1+f/\varepsilon))$ 个不同的先导。它返回该类中的一个成员以及一个宽度至多为 $\varepsilon$ 的区间。

    arXiv:2611.13913v1 Announce Type: new  Abstract: A run that stops at its first wrong decision shows the queries and their correct labels up to and including that decision, and nothing after it. We study how many such runs are needed for learning. Each run tests a hypothesis, the pilot, chosen beforehand from a fixed binary class of VC dimension at most $d$, on a fresh record of at most $D$ queries drawn from one unknown distribution that does not depend on the pilot. The terminal information of a record is shown only if the pilot makes no discrepancy, and a bounded loss, fixed in advance, is attached to the position where a hypothesis stops. If the discrepancy probabilities of members of the class have infimum at most $f$ and their risks have infimum at most $r$, one learner uses $\widetilde O((d+1)[D(f+\varepsilon)+r]/\varepsilon^2)$ runs and $\widetilde O(D(1+f/\varepsilon))$ distinct pilots on every history. It returns a member of the class and an interval of width at most $\varepsi
    
[^85]: 基于消失刺激混合现实P300接口的目标选择：可行性以及物体可区分线索与均一线索的对比

    Target Selection with a Disappearing-Stimulus Mixed-Reality P300 Interface: Feasibility and Object-Distinguishable Versus Uniform Cues

    [https://arxiv.org/abs/2611.13906](https://arxiv.org/abs/2611.13906)

    该研究证明，即使目标线索在闪烁间消失不再可见，混合现实P300接口仍能实现远高于随机水平的目标选择，并探索用颜色-形状可区分的线索替代均一球体以改善解码性能。

    

    混合现实（MR）P300接口可以在用户周围环境中直接标记可选目的地，例如轮椅的目标位置。然而，当线索在闪烁之间消失时，用户必须记住一个不再可见的预期目标。我们测试了此类接口是否能够支持目标选择，以及与视觉上均一的球体相比，为每个线索赋予稳定的颜色-形状标识是否能改善选择效果。六名健康成年人在头戴式混合现实显示器上完成5x5行/列任务，在每种线索条件下各进行四次试验，并从Fz、Cz和Pz三个电极采集脑电信号。采用收缩线性判别分析并通过嵌套验证进行评估，其中完整的试验被留出，不参与类别平衡、超参数选择和模型拟合。两种条件均诱发了顶叶的目标相关正电位。在48个留出试验中，解码的单元格有18个正确，而随机水平仅预期1.9个（二项检验p < 1e-12），并且在三名参与者中……（摘要至此处截断）

    arXiv:2611.13906v1 Announce Type: new  Abstract: Mixed-reality (MR) P300 interfaces can register selectable destinations, such as wheelchair goals, directly in the user's surroundings. When the cues disappear between flashes, however, the user must retain an intended target that is no longer visible. We tested whether such an interface supports target selection and whether giving each cue a stable colour-shape identity improves it compared with visually uniform spheres. Six healthy adults completed four trials in each cue condition of a 5x5 row/column task on a head-mounted MR display, with EEG from Fz, Cz, and Pz. Shrinkage linear discriminant analysis was evaluated with nested validation in which complete trials were held out from class balancing, hyperparameter selection, and fitting. Both conditions evoked a parietal target-related positivity. The decoded cell was correct in 18 of 48 held-out trials, against 1.9 expected by chance (binomial p < 1e-12), and in three participants ind
    
[^86]: 技术驱动的域偏移下基于保形风险控制的QoT感知光路可行性评估

    Conformal Risk Control for QoT-Aware Lightpath Feasibility Under Technology-Driven Domain Shift

    [https://arxiv.org/abs/2611.13893](https://arxiv.org/abs/2611.13893)

    本文提出一种事后保形风险控制框架，在保持预训练AI/ML回归模型不变的情况下，为技术升级引起域偏移时光网络QoT感知光路可行性评估的不确定性区间提供有限样本统计保证，并通过策略规则将预测区间映射为运行决策。

    

    基于人工智能和机器学习（AI/ML）的回归模型能够支持光网络中的运行决策，但通常只提供点预测，缺乏统计上可控的可靠性信息。本文提出一种事后保形风险控制（CRC）框架，该框架保持预训练的AI/ML回归器不变，为不确定性区间的覆盖失败提供有限样本保证，并通过策略规则将该区间映射为运行决策。该保证适用于区间覆盖失败，而决策错误率则通过实证进行评估。我们将该框架实例化用于传输质量（QoT）感知的光路（LP）可行性评估，其中回归器在光纤、放大器和交换设备升级引起的域偏移下预测光信噪比（OSNR）。特征偏移贡献分数能够识别出与QoT关系变化最大的光路特征。

    arXiv:2611.13893v1 Announce Type: new  Abstract: Artificial intelligence and machine learning (AI/ML)-based regression models support operational decision-making in optical networks but typically expose only point predictions, which carry no statistically controlled information about their reliability. This paper proposes a post-hoc conformal risk control (CRC) framework that keeps a pretrained AI/ML regressor fixed, provides a finite-sample guarantee on uncertainty band miscoverage, and maps the band to operational decisions through policy rules. The guarantee applies to interval miscoverage, while decision-error rates are evaluated empirically. We instantiate the framework for quality of transmission (QoT)-aware lightpath (LP) feasibility assessment, in which a regressor predicts optical signal-to-noise ratio (OSNR) under domain shifts caused by fiber, amplifier, and switching upgrades. Feature shift contribution scores identify the LP features whose relation to QoT changes the most 
    
[^87]: 因果神经元回路是唯一且可迁移的吗？

    Are Causal Neuron Circuits Unique and Transferable?

    [https://arxiv.org/abs/2611.13885](https://arxiv.org/abs/2611.13885)

    该论文通过穷举测量 12 个 MLP 神经元全部 4,096 种子集的干预效果，发现虽然消融选定神经元组会显著改变模型回答，但这些神经元的激活在不同问题间几乎恒定、替换后也几乎无影响，说明真正必要的只是其中一部分，从而质疑了因果神经元回路的唯一性与可迁移性。

    

    可解释性领域的演示常常表明，对一组神经元进行干预会改变模型的回答。最近的一个例子是：移除 Qwen3-32B 中的一组 18 个神经元后，模型对一个两跳《狮子王》谜语的回答从“刀疤”变成了“辛巴”。这类演示证明了该神经元组确实重要，但没有说明其中有多少神经元是真正必要的、其他神经元组是否能起到同样作用，以及这种效应是否会迁移到其他输入上。我们提出的问题是：当对一组神经元的干预改变了模型的回答时，这组神经元中究竟有多少是必要的？对于每个选定的神经元组，我们测量了 12 个 MLP 神经元全部 4,096 种子集的效果。在一个二选一的首都识别任务（Qwen3-0.6B）上，将 12 个选定神经元置零会使答案置信边际下降 27%，而随机神经元几乎不产生改变；然而，用其他问题中的激活值替换这同一批神经元同样几乎不改变结果：这些神经元的激活在不同问题之间几乎恒定，我们没有找到证据（摘要原文在此处截断）

    arXiv:2611.13885v1 Announce Type: cross  Abstract: Interpretability demonstrations often show that intervening on a group of neurons changes a model's answer. A recent example removes an 18-neuron group in Qwen3-32B and the model answers "Simba" instead of "Scar" to a two-hop Lion King riddle. Such a demonstration establishes that the group matters, but not how much of it is necessary, whether other groups could do the same, or whether the effect carries over to other inputs. We ask: when intervening on a group of neurons changes a model's answer, how much of that group is actually necessary? For every selected group we measure all 4,096 subsets of 12 MLP neurons. On a two-alternative capital task (Qwen3-0.6B), zero-ablating 12 selected neurons lowers the answer margin by 27% while random neurons barely change it, yet replacing the same neurons with their values from other questions also barely changes it: their activations are nearly constant across questions, and we find no evidence 
    
[^88]: 高维小样本设置中的离群点与硬间隔线性支持向量机

    Outliers and the Hard-Margin Linear SVM in High-Dimension, Low-Sample-Size Settings

    [https://arxiv.org/abs/2611.13869](https://arxiv.org/abs/2611.13869)

    该论文建立了高维小样本下离群点对硬间隔线性SVM影响的精确渐近理论：单侧离群点的全部效应可归结为一个有效距离并由此导出内点支持丧失、间隔收缩、截距位移和误分类的显式阈值，而两类对称污染仅收缩间隔、不改变极限判别函数。

    

    我们研究了当维度增长而样本量保持固定时，离群点如何影响硬间隔线性支持向量机。对于单侧污染情形，中心化Gram矩阵收敛为一个常数内点块加上一个离群点对角分量，极限对偶问题具有闭式解，所有离群点均通过一个有效距离发挥作用。这一表示给出了内点支持丧失、间隔收缩、截距位移以及渐近误分类的显式阈值。随后我们研究两类数据均受污染的情形：极限截距是类别平衡约束的乘子，两个类别会将其向相反方向拉动。因此，至多只有一个类别会发生崩溃或渐近误分类；而对称污染虽然会使间隔收缩，却使极限判别函数保持不变。七个数值实验验证了对偶分配、相位……（原文摘要在此处截断）

    arXiv:2611.13869v1 Announce Type: cross  Abstract: We study how outliers affect the hard-margin linear support vector machine when the dimension grows while the sample sizes remain fixed. For one-sided contamination, the centered Gram matrix converges to a constant inlier block plus an outlier-diagonal component, and the limiting dual problem has a closed-form solution. All outliers act through one effective distance. This representation yields explicit thresholds for loss of inlier support, margin shrinkage. intercept displacement, and asymptotic misclassification. We then treat contamination in both classes. The limiting intercept is the multiplier of the class-balance constraint, and the two classes pull it in opposite directions. Consequently, at most one class can collapse or be asymptotically misclassified, whereas symmetric contamination leaves the limiting discriminant function unchanged even though the margin shrinks. Seven numerical experiments verify the dual allocation, pha
    
[^89]: 深度神经网络的机制可解释性：方法、框架与视角

    Mechanistic Interpretability of Deep Neural Networks: Methods, Frameworks, and Perspectives

    [https://arxiv.org/abs/2611.13868](https://arxiv.org/abs/2611.13868)

    本文系统综述了深度神经网络机制可解释性的方法与框架，其核心在于通过识别对特定行为具有因果影响的内部组件来揭示网络的内部工作机制。

    

    深度神经网络支撑着图像、音频和语言处理等众多信号领域的基础任务，然而其内部机制却极难解释。内部激活是跨层变换的结构化信号，可能编码大量相互依赖的关系，因为单个神经元常常代表多个潜在概念。机制可解释性旨在通过识别其激活对特定行为具有因果影响的组件，来揭示神经网络的内部机制。在此背景下，本文对用于分析深度神经网络因果推理的机制可解释性研究进展进行了综述。

    arXiv:2611.13868v1 Announce Type: cross  Abstract: Deep neural networks underpin fundamental tasks in a wide range of signal domains, such as image, audio, and language processing. However, their internal mechanisms are notably difficult to interpret. Internal activations are structured signals transformed across layers that may encode a multitude of interdependent relationships, as individual neurons frequently represent multiple underlying concepts. Mechanistic interpretability aims to uncover the internal mechanisms of neural networks by identifying components whose activations are causally responsible for specific behaviors. In such a context, this article provides an overview of advances in mechanistic interpretability for the analysis of causal inference in deep neural networks.
    
[^90]: 神经符号推理中应该在哪里进行近似？

    Where to Approximate in Neurosymbolic Inference?

    [https://arxiv.org/abs/2611.13862](https://arxiv.org/abs/2611.13862)

    该论文将现有近似WMC方法统一为三步自底向上编译管线框架，利用张量列车分解与基于SVD/插值的再压缩技术研究近似预算的最优分配位置，在数独、算术推理和场景理解等任务上实现了比先前方法准确数量级提升的WMC估计。

    

    概率神经符号方法依赖加权模型计数（WMC）来将神经预测器与符号约束相结合。精确的WMC是#P难问题且通常扩展性较差，因此许多方法转而采用近似手段。通过将现有方法统一在一个三步自底向上的近似编译管线框架下，我们研究了近似预算应如何最优分配。受张量网络的启发，我们使用张量列车分解来实例化其中的两个步骤，并推导出新的管线，这些管线能够精确地相乘近似后的因子，并通过基于SVD或插值的方案对其乘积进行再压缩。在数独、算术推理和场景理解的六个任务上，以及不断增大的问题规模下，我们的管线所产生的WMC估计比先前方法准确若干个数量级。这进而带来了更强的下游学习性能，尤其是在推理依赖于符号因子之间紧密交互的任务上，而这类任务中更粗糙的近似方法会……（摘要截断）

    arXiv:2611.13862v1 Announce Type: new  Abstract: Probabilistic neurosymbolic methods rely on weighted model counting (WMC) to combine neural predictors with symbolic constraints. Exact WMC is #P-hard and typically scales poorly, so many methods resort to approximations. By unifying existing approaches under a three-step bottom-up approximate compilation pipeline, we study where approximation budget is best allocated. Inspired by tensor networks, we instantiate two steps with tensor train decompositions and derive new pipelines that exactly multiply approximated factors and recompress their products via SVD- or interpolation-based schemes. Across six tasks in Sudoku, arithmetic reasoning, and scene understanding, and across increasing problem sizes, our pipelines produce WMC estimates orders-of-magnitude more accurate than prior methods. This translates into stronger downstream learning on tasks where reasoning relies on tight interactions between symbolic factors, where cruder approxim
    
[^91]: 超越被动模仿：面向语言模型的在线策略中期训练蒸馏

    Beyond Passive Imitation: On-Policy Mid-Training Distillation for Language Models

    [https://arxiv.org/abs/2611.13855](https://arxiv.org/abs/2611.13855)

    该论文提出在线策略中期训练蒸馏（OPMD），将中期训练从被动的离线文本模仿转变为主动的、查询驱动的知识获取，通过从学生策略自采样轨迹并结合教师模型的词元级反向KL奖励，解决了标准下一词预测目标在不同基础模型上效果不一甚至产生负面影响的问题。

    

    持续预训练（CPT）可将大型语言模型（LLM）适配到特定领域，然而其标准的下一词预测（NTP）目标将语料库中的每个词元都视为固定目标，而不考虑其对当前模型的效用。在一项基于相同10亿词元数学语料库的对照研究中，我们观察到不同基础模型之间结果差异显著：NTP提升了Llama-3.2-3B在数学基准上的表现，却显著降低了Qwen3-1.7B-Base的性能，这表明在固定的训练方案下，相同的训练序列可能产生依赖于具体模型的效果。为解决这一问题，我们提出了在线策略中期训练蒸馏（OPMD），将中期训练从离线的文本模仿重新表述为主动的、查询驱动的知识获取。在选定的锚定位置，OPMD从不断演进的学生策略中采样轨迹，并应用来自冻结教师模型的密集词元级反向KL奖励进行训练。

    arXiv:2611.13855v1 Announce Type: new  Abstract: Continual pre-training (CPT) adapts large language models (LLMs) to specialized domains, yet its standard next-token prediction (NTP) objective treats every corpus token as a fixed target, regardless of its utility to the current model. In a controlled study on the same 1B-token mathematical corpus, we observe sharply different outcomes across base models: NTP improves Llama-3.2-3B's performance on mathematical benchmarks but substantially degrades Qwen3-1.7B-Base's performance, indicating that, under a fixed training recipe, the same training sequences can have model-dependent effects. To address this problem, we introduce On-Policy Mid-Training Distillation (OPMD), which reformulates mid-training from offline text imitation into active, query-driven knowledge acquisition. At selected anchor positions, OPMD samples trajectories from the evolving student policy and applies dense token-level reverse-KL rewards from a frozen teacher, train
    
[^92]: 非扩张随机逼近的最优末次迭代界

    Optimal Last-Iterate Bounds for Nonexpansive Stochastic Approximation

    [https://arxiv.org/abs/2611.13846](https://arxiv.org/abs/2611.13846)

    本文为非扩张随机逼近（随机KM迭代）建立了末次迭代均方残差界，在噪声条件二阶矩的三种情形下均给出相差对数因子内最优的收敛速率，并在噪声与残差平方呈线性关系时达到与无噪声KM迭代相同的界。

    

    我们为非扩张随机逼近（SA），即随机 Krasnoselskii-Mann（KM）迭代，建立了末次迭代的均方残差界。特别地，我们考虑关于欧几里得范数非扩张的不动点映射，并重点关注多项式衰减的步长。我们研究了噪声条件二阶矩的三种情形，并在每种情形下都获得了在相差对数因子意义下最优的界。当噪声与残差平方呈线性关系时，最佳速率为任意 ε>0 情况下的 O(1/k^{1-ε})；在此情形下，对于所有考虑的步长，我们的界与无噪声 KM 迭代的界相匹配。当噪声被常数所界定时，最佳速率为 O(log² k/√k)。当噪声与迭代点的范数平方呈仿射关系时，最佳速率为 O(1/k^{1/2-ε})，且若步长不满足平方可和性，均方残差可能发散。因此，与压缩映射……（摘要原文在此处被截断）

    arXiv:2611.13846v1 Announce Type: cross  Abstract: We establish last-iterate mean-square residual bounds for nonexpansive stochastic approximation (SA), or stochastic Krasnoselskii-Mann (KM) iterations. In particular, we consider fixed-point mappings that are nonexpansive with respect to the Euclidean norm, and focus on polynomially decaying stepsizes. We study three conditions on the conditional second moment of the noise, and obtain bounds that are optimal up to logarithmic factors in each case. When it scales linearly with the squared residual, the best rate is $O(1/k^{1-\epsilon})$ for any $\epsilon>0$. In this case, for all considered stepsizes, our bounds match those of the noiseless KM iteration. When it is bounded by a constant, the best rate is $O(\log^2 k/\sqrt{k})$. When it scales affinely with the squared norm of the iterate, the best rate is $O(1/k^{1/2-\epsilon})$, and the mean-square residual can diverge if the stepsizes are not square-summable. Thus, unlike for contract
    
[^93]: MusicFlux：用于文本引导音乐生成与编辑的灵活条件化方法

    MusicFlux: Flexible Conditioning for Text-Guided Music Generation and Editing

    [https://arxiv.org/abs/2611.13834](https://arxiv.org/abs/2611.13834)

    MusicFlux是一个基于预训练流匹配文本到音频模型的文本引导音乐编辑器，通过两种条件化方法在编辑质量与计算效率之间实现权衡，无需处理过长的音频序列即可统一音乐的生成与编辑。

    

    音乐编辑是在保留录音原始属性的同时对其进行变换。近来的方法通过使用上下文音频进行条件化，使预训练的文本到音乐模型适应修复、事件插入或恢复等任务。最近在图像编辑领域提出了一种通过将源图像视为上下文信息来统一生成与编辑的方法。然而，由于音频序列较长，直接将该方法应用于音乐会导致计算成本高昂。在本工作中，我们开发了MusicFlux，一个基于高效预训练流匹配文本到音频模型的文本引导音乐编辑器。我们引入了两种在编辑性能与计算效率之间进行权衡的条件化方法。实验表明，这两种方法均与强大的基线方法具有竞争力，其中一种实现了更好的编辑质量，另一种则保留了原始序列长度并显著提升了计算效率。

    arXiv:2611.13834v1 Announce Type: cross  Abstract: Music editing transforms existing recordings while preserving their original attributes. Recent approaches adapt pretrained text-to-music models for tasks like inpainting, event insertion, or restoration by conditioning them with context audio. It was recently proposed for image editing to unify generation and editing by treating the source image as in-context information. However, applying this directly to music is computationally expensive due to long audio sequences. In this work, we develop MusicFlux, a text-guided music editor built on an efficient pretrained flow-matching text-to-audio model. We introduce two conditioning approaches that trade off editing performance and computational efficiency. Experiments show that both are competitive with strong baselines, with one achieving better editing quality and the other preserving the original sequence length and substantially improving computational efficiency.
    
[^94]: JAX-WAVE：面向非均匀介质中逆波场设计的可微分物理监督框架

    JAX-WAVE: A Differentiable Physics-Supervised Framework for Inverse Wave Design in Heterogeneous Media

    [https://arxiv.org/abs/2611.13826](https://arxiv.org/abs/2611.13826)

    提出可微分物理监督框架 JAX-WAVE，通过共享神经网络与可微分时域有限差分求解器实现摊销式逆波场设计，无需优化控制标签即可将目标波场直接映射为满足符号反转等物理约束的最优有界初始压力。

    

    在非均匀介质中设计一个指定的波场，需要一种能够补偿介质传播响应的初始压力分布。传统的重复数值反演方法需要针对每个目标和环境重新求解该任务。我们提出了 JAX-WAVE 用于摊销式逆波场设计：一个共享神经网络将期望的末端波场和已知的波速分布图直接映射到有界的初始压力。训练过程通过可微分的时域有限差分（FDTD）求解器进行，直接使用物理设计目标，而无需依赖优化得到的控制标签。线性传播和对称压力界意味着正则化最优解存在精确的符号反转关系。一个共享权重的投影机制强制执行这一关系，保持解的可行性，并满足配对符号目标的精确恒等式。我们刻画了离散传播算子及其加权伴随算子，从而将学习、数值反演和最优性评估联系起来。

    arXiv:2611.13826v1 Announce Type: new  Abstract: Designing a prescribed wave field in a heterogeneous medium requires an initial pressure distribution that compensates for the medium's propagation response. Repeated numerical inversion solves this task afresh for each target and environment. We introduce JAX-WAVE for amortized inverse wave design: a shared neural network maps desired terminal fields and known wave-speed maps directly to bounded initial pressures. Training through a differentiable finite-difference time-domain solver uses the physical design objective without requiring optimized control labels. Linear propagation and symmetric pressure bounds imply an exact sign-reversal relation for the regularized optimum. A shared-weight projection enforces this relation, preserves feasibility, and satisfies an exact identity for the paired-sign objective. We characterize the discrete propagator and its weighted adjoint to connect learning, numerical inversion, and optimality assessm
    
[^95]: CASCADE：面向逻辑异常检测的约束感知支架式分解方法

    CASCADE: Constraint-Aware SCAffolded DEcomposition for Logical Anomaly Detection

    [https://arxiv.org/abs/2611.13820](https://arxiv.org/abs/2611.13820)

    论文提出CASCADE方法，通过约束感知的支架式分解为逻辑异常检测中的每类约束定制视觉证据提取，弥合了LVLM清单式验证中“约束特定感知鸿沟”所导致的实例混淆与尺度漂移问题。

    

    逻辑异常检测（LAD）要求在仅有少量正常参考图像的条件下，验证查询图像是否满足所有类别特定的约束，例如物体数量、配对关系和测量范围。近期基于大型视觉-语言模型（LVLM）的LAD方法将该任务表述为清单式验证，把决策分解为独立的文本检查项。然而，这种分解仅停留在文本层面：每个检查项仍然读取同一张未经修改的查询图像，迫使LVLM必须从同一张原始图像中提取不同形式的视觉证据。我们将这种文本-图像不匹配识别为“约束特定的感知鸿沟”，它是两种失败模式的根源：绑定约束下的实例混淆，以及校准约束下的尺度漂移。为了诊断这些失败究竟源于推理不足还是感知不足，我们在无训练设置下比较了文本推理引导与视觉感知引导……（原文摘要在此处截断）

    arXiv:2611.13820v1 Announce Type: cross  Abstract: Logical Anomaly Detection (LAD) requires verifying whether all category-specific constraints, such as object counts, pairing relations, and measurement ranges, are satisfied by a query image given only a few normal references. Recent LAD methods based on Large Vision-Language Models (LVLMs) formulate this as checklist verification, decomposing the decision into independent textual checks. This decomposition, however, is text-only. Every item still reads the same unmodified query image, so the LVLM must extract different forms of visual evidence from one raw image. We identify this text-image mismatch as a constraint-specific perception gap, the source of two failure modes: instance conflation for binding constraints and scale drift for calibration constraints. To diagnose whether these failures stem from insufficient reasoning or insufficient perception, we compare textual reasoning guides and visual perception guides in a training-fre
    
[^96]: 自适应策略引导的错误缓解（APGEM）：一个面向量子机器学习噪声韧性的统一学习框架

    Adaptive Policy-Guided Error Mitigation (APGEM): A Unified Learning-Based Framework for Noise Resilience in Quantum Machine Learning

    [https://arxiv.org/abs/2611.13815](https://arxiv.org/abs/2611.13815)

    APGEM提出了一种基于上下文老虎机的自适应框架，利用变分量子算法自身产生的反馈信号在线动态选择错误缓解技术，从而解决了量子机器学习训练过程中因电路与噪声漂移导致固定缓解策略失效的问题。

    

    错误缓解是当今含噪硬件上变分量子算法获得有用结果的关键环节，然而在实践中，缓解方法通常是一次性人工选定，并在实验的整个生命周期中一成不变地应用。这对量子机器学习（QML）来说并不合适，因为随着训练的进行，电路和噪声都会发生漂移，因此在一种情形下最优的策略在另一种情形下往往是错误的。我们提出了自适应策略引导错误缓解（APGEM）的广义形式，它将缓解技术的选择视为根据算法自身反馈做出的在线决策。一个单一的上下文老虎机（contextual-bandit）控制器观察任何变分算法本身已经产生的信号，例如保真度、熵、奖励或损失以及梯度行为，并在零噪声外推、概率误差消除、测量误差缓解、动态解耦和电路优化之间逐步进行选择。由于它依赖于……（原文在此截断）

    arXiv:2611.13815v1 Announce Type: cross  Abstract: Error mitigation stands between variational quantum algorithms and useful results on today's noisy hardware, yet in practice a mitigation method is usually chosen once, by hand, and applied unchanged for the life of an experiment. This is a poor fit for quantum machine learning (QML), where the circuit and the noise drift as training proceeds, so the best strategy in one regime is often wrong in another. We present a generalized form of Adaptive Policy-Guided Error Mitigation (APGEM), which treats the choice of mitigation technique as an online decision made from an algorithm's own feedback. A single contextual-bandit controller observes signals any variational algorithm already produces, such as fidelity, entropy, reward or loss, and gradient behavior, and selects among Zero-Noise Extrapolation, Probabilistic Error Cancellation, Measurement Error Mitigation, Dynamical Decoupling, and circuit optimization per step. Because it depends o
    
[^97]: AnchorMTS：用于长期时间序列预测的多视图信号表示与预测锚定融合

    AnchorMTS: Multi-View Signal Representation and Forecast-Anchored Fusion for Long-Term Time-Series Forecasting

    [https://arxiv.org/abs/2611.13794](https://arxiv.org/abs/2611.13794)

    AnchorMTS通过从时间序列中提取视觉和文本等多视图表示来修正数值预测，在多个基准数据集上取得了最先进的长期预测性能。

    

    长期时间序列预测模型通常只优化单一的数值表示，尽管同一段历史数据中蕴含着互补的几何、循环和语义结构。我们提出AnchorMTS，该方法从每个观测窗口中提取自对齐的视觉视图和文本视图，并仅用它们来完善一个完整的数值预测。一个可训练的证据路径产生候选预测，同时一个无参数的输出投影对其相对数值锚点的位移施加精确的逐变量约束。在六个基准数据集和四种预测跨度的实验中，AnchorMTS在七个监督基线模型中取得了最佳的宏观MSE/MAE（0.336/0.367），并在24个MSE设置中的18个上优于逐单元基线包络；移除辅助路径会使宏观MSE上升至0.340。该投影使95.8%的候选修正保持不变，同时实现零边界违规，表明互补的多视图证据能够提升预测性能。

    arXiv:2611.13794v1 Announce Type: new  Abstract: Long-term time-series forecasters typically optimize a single numerical representation, even though the same history exposes complementary geometry, recurrence, and semantic structure. We propose AnchorMTS, which derives self-aligned visual and textual views from each observed window and uses them only to refine a complete numerical forecast. A trainable evidence path produces a candidate forecast, while a parameter-free output projection enforces an exact per-variable bound on its displacement from the numerical anchor. Across six benchmarks and four horizons, AnchorMTS achieves the best macro MSE/MAE (0.336/0.367) among seven supervised baselines and wins 18 of 24 MSE settings against the cellwise baseline envelope; removing the auxiliary path raises macro MSE to 0.340. The projection leaves 95.8% of candidate refinements unchanged while yielding zero bound violations, showing that complementary multi-view evidence can improve forecast
    
[^98]: Omni-JEV：文本与双流语音决策

    Omni-JEV: Text and Dual-Stream Speech Decisions

    [https://arxiv.org/abs/2611.13777](https://arxiv.org/abs/2611.13777)

    提出 Omni-JEV 模型，在全双工对话中将文本或角色索引的双流（用户与助手参考）语音映射为五种交互动作的概率，并通过双语弱监督微调与 RLCD 强化学习实现概率敏感的决策。

    

    在全双工对话中，用户的一句话并不能唯一决定下一个动作：其后果取决于助手的响应状态与交互历史。语音活动、话语完成度以及打断意图提供了相关证据，而这些证据对响应的意义需要一个显式的动作策略。我们在收集到的语音片段端点处对这一任务进行形式化，并提出了 Omni-JEV，它将文本、或带有角色索引的用户语音与助手参考语音，映射为五种交互动作上的概率分布。一个对齐的语音到语音理解骨干网络分别编码声学角色，并与文本输入共享其候选输出读出。双语弱监督支持交叉熵微调，随后采用面向概率敏感决策的强化学习（RLCD）。我们在相同的监督初始化与匹配的训练曝光条件下，比较了高斯 logit 与类别型 Choice 两种 RLCD，观察到不同的 tr...（原文此处截断）

    arXiv:2611.13777v1 Announce Type: new  Abstract: A user utterance does not uniquely determine the next action in full-duplex dialogue: its consequences depend on the assistant's response state and interaction history. Speech activity, utterance completion, and interruption intent provide related evidence, whose implications for a response require an explicit action policy. We formulate this task at collected-segment endpoints and introduce Omni-JEV, which maps text or role-indexed user and assistant-reference speech to probabilities over five interaction actions. An aligned speech-to-speech understanding backbone encodes the acoustic roles separately and shares its candidate readout with text inputs. Bilingual weak supervision supports cross-entropy fine-tuning followed by reinforcement learning for probability-sensitive decisions (RLCD). We compare Gaussian-logit and categorical Choice RLCD from the same supervised initialization under matched training exposure, observing different tr
    
[^99]: 超越交互分数：通过独立复现选择特征交互

    Beyond Interaction Scores: Selecting Feature Interactions via Independent Replication

    [https://arxiv.org/abs/2611.13765](https://arxiv.org/abs/2611.13765)

    LENS 提出了一个模型无关的特征交互选择框架，通过在不相交数据上训练两个模型进行独立复现来验证交互模式的真实性，无论特征数量多少仅需两次模型拟合和一次多重检验校正。

    

    学习特征交互对于准确预测和可解释建模至关重要，然而高的交互分数可能反映的是对噪声的拟合，而非可复现的模式。排序方法对特征对进行打分，但将阈值的选择留给用户；检验方法虽然设定了阈值，但依赖于反复的模型重拟合或对联合特征分布建模。我们提出了 LENS，一个通过独立复现来选择特征交互的模型无关框架。LENS 使用相同的学习算法在不相交的数据上训练两个模型，并且仅通过预测来访问它们。对于每个特征对，LENS 从联合效应面中移除主效应以分离出交互模式，按该模式的强度对特征对进行排序，并检验第二个模型是否能以有意义的强度复现该模式。只需一次多重检验校正即可确定保留哪些特征对，且无论特征数量多少，仅需两次模型拟合。

    arXiv:2611.13765v1 Announce Type: new  Abstract: Learning feature interactions is essential for accurate prediction and interpretable modeling, yet a high interaction score may reflect fitting noise rather than a reproducible pattern. Ranking methods score feature pairs but leave the cutoff to the user; testing methods set the cutoff but rely on repeated model refits or a model of the joint feature distribution. We present LENS, a model-agnostic framework for selecting feature interactions through independent replication. LENS trains two models with the same learning algorithm on disjoint data and accesses them only through predictions. For each feature pair, it removes main effects from the joint-effect surface to isolate an interaction pattern, ranks pairs by its strength, and tests whether the second model reproduces it at a meaningful strength. A single multiple-testing correction determines which pairs to retain, using only two model fits regardless of the number of features. We p
    
[^100]: 混合信号：面向脑电图（EEG）与心电图（ECG）的权重空间编程

    Mixed Signals: Weight-Space Programming for EEG and ECG

    [https://arxiv.org/abs/2611.13758](https://arxiv.org/abs/2611.13758)

    本文首次将权重空间编程应用于脑电图与心电图等生物信号模型，证明语言模型不仅能诊断并修复植入的模型故障，还能仅凭推理从零写出网络全部权重，并以少6倍的训练数据达到与梯度下降训练相当的效果。

    

    权重空间编程使语言模型能够读取经过训练的神经网络的权重并对其进行编辑。这为通过推理而非梯度下降进行“训练”打开了大门。先前的工作已经证明了该方法在游戏类网络中的功能与前景，以及其在修复语言模型特定故障方面的应用，甚至扩展到了扩散模型。在本工作中，我们展示了该方法在生物信号数据训练的模型上的适用性，特别是脑电图（EEG）和心电图（ECG）数据。我们表明，前沿模型不仅能够在未被告知故障内容的情况下，调试植入生物信号模型中的特定故障，甚至能够在仅给定网络架构、任务以及最少量的训练受试者数据的情况下，从零开始写出网络的全部权重。特别是在其中一组实验中，一个语言模型仅通过推理，就在使用少6倍训练受试者的情况下，达到了与标准梯度下降训练相当的效果。

    arXiv:2611.13758v1 Announce Type: cross  Abstract: Weight-space programming allows language models to read a trained neural network's weights and edit them. This opens the door to ``training'' by reasoning rather than through gradient descent. Prior work has demonstrated its functionality and promise for game-playing networks and fixing specific failures in language models, even extending to diffusion models. In this work, we present its applicability to models trained on biosignal data, specifically EEG and ECG data. We show that frontier models are not only able to debug specific faults planted in biosignal-based models without being told what these faults are, but are even capable of writing all the weights of a network from scratch, given only the architecture, the task, and a minimal set of training subjects. In particular, in one set of experiments, a language model was able to match standard gradient-descent training using $6\times$ fewer training subjects, through reasoning alo
    
[^101]: 当协方差不够时：用于单细胞反事实的基于核的正则化方法

    When Covariance Is Not Enough: Kernel-Based Regularization for Single-Cell Counterfactuals

    [https://arxiv.org/abs/2611.13748](https://arxiv.org/abs/2611.13748)

    本文证明单细胞反事实生成模型中常用的线性交叉协方差惩罚在存在非线性交互时无法充分解耦已知与未知因素，并提出加入归一化Hilbert-Schmidt独立性准则（HSIC）这一基于核的非线性正则化方法来改进分离效果。

    

    单细胞反事实编辑——预测一个被观测细胞在某个因素不同的情况下会呈现何种状态——能够在实验不可行时实现计算机模拟筛选，但只有当编辑忠实于个体本身而不仅仅是其所属条件时才有用。概念瓶颈生成模型通过将细胞的表示分解为已知和未知因素，使干预的效果变得可解释。先前的工作使用线性交叉协方差惩罚来解耦已知和未知维度。我们证明，当存在非线性交互作用时，这种一阶准则无法完全分离信息。我们首先通过带有受控编辑和已知非线性协变量效应的模拟实验，证明了线性跨因子惩罚的不足。随后我们证明，加入一种非线性核准则——归一化Hilbert-Schmidt独立性准则（HSIC），能够恢复……

    arXiv:2611.13748v1 Announce Type: new  Abstract: Counterfactual editing of single cells, predicting how an observed cell would look had one factor been different, enables in-silico screening where experiments are infeasible, but is useful only if the edit is faithful to the individual and not merely to the condition. Concept bottleneck generative models make the effects of interventions interpretable by decomposing a cell's representation into known and unknown factors. Previous work uses a linear cross-covariance penalty to disentangle known and unknown dimensions. We demonstrate that this first-order criterion does not fully separate information when non-linear interactions are present. We first demonstrate the inadequacy of linear cross-factor penalization with simulation experiments with controlled edits and known non-linear covariate effects. We then demonstrate that adding a nonlinear kernel criterion, the normalized Hilbert-Schmidt Independence Criterion, is able to recover the 
    
[^102]: 面向逆问题的基于流匹配去噪器的即插即用ADMM方法

    Plug-and-Play ADMM for Inverse Problems with Flow Matching Denoiser

    [https://arxiv.org/abs/2611.13742](https://arxiv.org/abs/2611.13742)

    提出了一种基于ADMM的即插即用框架，通过辅助变量、对偶更新和轨迹校正机制，将流匹配先验更一致地融入逆问题求解，以更低计算成本实现高质量重建。

    

    使用生成先验的即插即用（PnP）方法在逆问题中取得了出色的性能，近期的流匹配模型更是使快速且可扩展的重建成为可能。然而，PnP方法的一个关键挑战是保持数据保真度更新与学习到的生成先验之间的一致性，因为数据一致性步骤可能使迭代点偏离流轨迹。我们提出了一种基于ADMM的流匹配先验PnP框架，通过辅助变量和对偶更新来强化测量模型与先验之间的一致性。此外，我们引入了一种轨迹校正机制，通过确定性与随机投影相结合的方式恢复与流的一致性。我们的方法为流先验在PnP框架中的集成提供了更有原则性的方式，以更低的计算成本实现了高质量的重建。

    arXiv:2611.13742v1 Announce Type: new  Abstract: Plug-and-Play (PnP) methods with generative priors have achieved strong performance in inverse problems, with recent flow matching models enabling fast and scalable reconstruction. However, a key challenge in PnP methods is maintaining consistency between data fidelity updates and the learned generative prior, as data-consistency steps can move iterates away from the flow trajectory. We propose an ADMM-based PnP framework for flow matching priors that enforces stronger consistency between the measurement model and the prior through auxiliary variables and dual updates. In addition, we introduce a trajectory correction mechanism that restores consistency with the flow via a combination of deterministic and stochastic projections. Our method provides a more principled integration of flow priors within PnP frameworks, achieving high-quality reconstructions with reduced computational cost.
    
[^103]: 精确整数运算下的确定性、可复现推理，以及由此实现的可验证分布式推理

    Deterministic, Reproducible Inference in Exact Integer Arithmetic, and Verifiable Distributed Inference as Its Consequence

    [https://arxiv.org/abs/2611.13730](https://arxiv.org/abs/2611.13730)

    该论文提出开源运行时 tritify，将真实模型推理的整个运行时路径置于精确整数运算中，实现了跨硬件、跨平台的比特级可复现推理，并以此为基础支持可验证的分布式推理。

    

    优化后的浮点推理在不同环境中不可复现：同一模型在不同的硬件、编译器或线程数下可能产生不同的比特结果。我们提出了 tritify，一个开源运行时系统，它以精确整数运算执行真实模型——包括仅解码器语言模型、混合专家解码器以及 Segment Anything 2.1 图像/视频处理栈——的前向传播，整个运行时路径均采用精确整数运算；浮点运算仅限于离线量化以及用于验证的浮点参考。我们提出三个相互独立的声明并逐一验证。(1) 跨硬件可复现性：整数路径在多次运行之间、跨不同 CPU 指令集架构以及跨 CPU/GPU 均产生比特级完全相同的输出，并由输出结果的 SHA-256 哈希值来承载该可复现性验证。(2) 决策保真度：整数路径在所测试的模型和输入上复现了浮点参考的决策结果，并对数值异常情况进行了测量；逐层保真度……

    arXiv:2611.13730v1 Announce Type: new  Abstract: Optimized floating-point inference is not reproducible across environments: the same model on different hardware, compilers or thread counts can produce different bits. We present tritify, an open-source runtime that executes the forward pass of real models -- decoder-only language models, Mixture-of-Experts decoders, and the Segment Anything 2.1 image/video stack -- with the entire run-time path in exact integer arithmetic; floating point is confined to offline quantization and the float reference used for verification. We make three separate claims and verify each. (1) Cross-hardware reproducibility: the integer path produces bit-identical output run to run, across CPU instruction-set architectures, and across CPU/GPU, carried by a reproducing SHA-256 of the output. (2) Decision fidelity: the integer path reproduces the float reference's decision on the models and inputs tested, with the numeric exceptions measured; per-layer fidelity 
    
[^104]: 加权信息年龄驱动的无人机轨迹与上报方法用于动态信道知识地图追踪

    Weighted-AoI-Driven UAV Trajectory and Reporting for Dynamic CKM Tracking

    [https://arxiv.org/abs/2611.13726](https://arxiv.org/abs/2611.13726)

    本文提出加权信息年龄驱动的无人机轨迹规划与上报方法，通过结合精确的时间陈旧性与U-Net预测的空间不确定性来调度测量，实现动态信道知识地图的连续在线追踪，并证明了top-K选择策略可最大化单时隙误差降低上界。

    

    现有的信道知识地图构建方法假设环境是准静态的，当信道快速去相关时便会失效。本文研究了利用无人机机群在空中二维平面上进行连续在线动态CKM追踪的问题。我们在无人机移动性和上报约束条件下，构建了一个长期误差最小化问题。然而，由于未测量小区的误差不可观测，该问题难以求解。为解决这一难题，我们提出了一种以加权信息年龄为核心的度量驱动调度方法。W-AoI通过将精确的时间陈旧性与由2D U-Net预测的空间不确定性相结合来评估各个小区。我们证明了选取W-AoI值最高的前K个小区可使单时隙误差降低上界最大化。在该理论保证的指导下，我们进一步开发了一种融合沿路径连续感知的轻量级贪心轨迹策略。仿真结果表明，与仅基于AoI的方法相比，本文方法将稳态误差分别降低了3.8 dB²和4.8 dB²。

    arXiv:2611.13726v1 Announce Type: new  Abstract: Existing channel knowledge map (CKM) construction assumes quasi-static environments, failing when channels rapidly decorrelate. This paper investigates continuous online dynamic CKM tracking over an airborne 2D plane via a UAV fleet. We formulate a long-term error minimization problem under UAV mobility and reporting constraints. This problem, however, is intractable with unobservable errors for unmeasured cells. To tackle this, we propose a metric-driven scheduling approach featuring the Weighted-Age of Information (W-AoI). W-AoI evaluates cells by combining exact temporal staleness with spatial uncertainty predicted by a 2D U-Net. We prove that top-$K$ W-AoI selection maximizes the one-slot error reduction bound. Guided by this guarantee, we develop a lightweight greedy trajectory policy incorporating continuous along-path sensing. Simulations demonstrate our method reduces steady-state error by 3.8 dB$^2$ and 4.8 dB$^2$ over AoI-only 
    
[^105]: TRUER：基于误差感知校正器的测试时奖励校正强化学习方法

    TRUER: Test-Time Reward Correction Using an Error-Aware Rectifier for Reinforcement Learning

    [https://arxiv.org/abs/2611.13714](https://arxiv.org/abs/2611.13714)

    TRUER框架利用有限的标注数据训练一个误差感知的奖励校正器，专门针对错误多数和正确少数的情形来区分正确性与多数共识，从而在测试时强化学习中校正奖励、提升奖励准确性和模型性能。

    

    测试时强化学习使语言模型能够利用从自身答案中派生的奖励进行自我适应。然而，即使存在正确的替代答案，多数投票机制仍可能强化错误的解决方案。我们的训练审计显示，假阳性（即错误响应却获得正奖励）是奖励误差的主要来源。我们关注一个实际场景：在该场景下，真值奖励难以大规模获取，但存在一个有限的带标签源数据集。我们提出了TRUER框架，该框架利用有限监督训练一个奖励校正器，用以区分正确性与共识。为实现这一目标，我们在训练校正器时重点针对错误多数和正确少数的情况，即共识与正确性不一致的情形。随后，我们冻结该校正器，并在后续的测试时适应过程中使用它来校正奖励。通过提高奖励准确性进而提升模型性能，TRU

    arXiv:2611.13714v1 Announce Type: new  Abstract: Test-time reinforcement learning enables language models to adapt using rewards derived from their own answers. However, majority voting can reinforce incorrect solutions even when correct alternatives are available. Our training audit shows that false positives (incorrect responses receiving positive rewards) dominate reward errors. We focus on a practical regime where ground-truth rewards are difficult to obtain at scale, while a limited labeled source set is available. We introduce TRUER, a framework that uses limited supervision to train a reward rectifier that distinguishes correctness from consensus. To achieve this goal, we train the rectifier with an emphasis on incorrect majorities and correct minorities, targeting cases where consensus and correctness disagree. After that, we freeze the rectifier and use it to correct rewards during subsequent test-time adaptation. By improving reward accuracy and thereby model performance, TRU
    
[^106]: FedVP：通过汇聚本地训练的验证器实现跨异构模型的联邦学习

    FedVP: Federated Learning across Heterogeneous Models by Pooling Locally Trained Verifiers

    [https://arxiv.org/abs/2611.13708](https://arxiv.org/abs/2611.13708)

    该论文提出FedVP新范式，让使用不同（冻结的）大语言模型骨干的异构客户端通过共享、汇聚各自本地训练的验证器来协作选择答案，从而在不聚合或修改模型参数的情况下实现联邦学习收益。

    

    传统的联邦学习依赖于共享的模型骨干来聚合客户端更新，这限制了部署不同家族、不同规模大语言模型（LLM）的机构之间的协作。我们提出联邦验证器池，这是一种新的联邦学习范式，使异构客户端能够通过共享验证而非参数聚合来进行协作。每个客户端使用本地数据及其所选LLM骨干生成的回复来训练一个验证器；服务器将这些独立训练的验证器汇集起来，结合它们的判断来选择答案，而无需合并模型参数。这使得LLM骨干和验证器都可以在架构和规模上各不相同，同时保持LLM骨干冻结，并通过改进的答案选择来获得协作收益。我们在五个公开的医疗问答数据集（作为联邦客户端）上对FedVP进行了评估，使用Qw（摘要在此处截断，信息不完整）

    arXiv:2611.13708v1 Announce Type: new  Abstract: Conventional federated learning relies on the shared model backbone to aggregate client updates, limiting collaboration among institutions that deploy large language models (LLMs) from different families and at different scales. We propose Federated Verifier Pool (FEDVP), a new federated learning paradigm that enables heterogeneous clients to collaborate through shared verification rather than parameter aggregation. Each client trains a verifier using local data and responses generated by its chosen LLM backbone. The server pools these independently trained verifiers and combines their judgments to select answers without merging model parameters. This allows both LLM backbones and verifiers to differ in architecture and size, while keeping the LLM backbones frozen and deriving collaborative gains from improved answer selection. We evaluate FEDVP across five public medical question-answering datasets treated as federated clients, using Qw
    
[^107]: 基于神经优化层的决策聚焦学习：用于AI数据中心电池辅助电网购电平滑

    Decision-Focused Learning via Neural Optimization Layers for Battery-Assisted Grid-Import Smoothing in AI Data Centers

    [https://arxiv.org/abs/2611.13694](https://arxiv.org/abs/2611.13694)

    本文提出电池调度决策聚焦学习方法（BS-DFL），通过约束感知神经优化层将电池储能调度与需求预测端到端联合训练，无需最优解标签，在AI数据中心仿真中将电网购电分钟级波动降低71.70%、调度成本降低18.01%。

    

    快速的AI数据中心功率波动给电网接入带来了挑战。要实现持续的电网购电平滑，需要在不确定的需求下对有限的电池功率和能量进行分配。我们提出了一种风险感知的电池储能系统（BESS）调度方法，该方法协调事前备用容量承诺与需求响应式调度，同时考虑运行成本、电网爬坡惩罚及其条件风险价值（CVaR）。我们开发了电池调度决策聚焦学习（BS-DFL），其中包含一个约束感知的神经优化层，该层直接从调度目标进行训练，无需最优解标签。训练完成后，该神经优化层被冻结，通过调度成本和爬坡风险来引导需求预测器的学习，从而实现无需求解器的滚动时域调度。在使用基于实测AI工作负载推导的负荷进行的10,000分钟仿真中，与基于MSE训练的预测器相比，BS-DFL将电网购电的分钟级平均绝对变化降低了71.70%，并将总调度成本降低了18.01%。

    arXiv:2611.13694v1 Announce Type: cross  Abstract: Rapid AI data-center power fluctuations challenge grid integration. Sustained grid-import smoothing requires allocating limited battery power and energy under uncertain demand. We formulate risk-aware BESS scheduling that coordinates advance reserve commitments and demand-responsive dispatch, accounting for operating costs, grid-ramp penalties, and their conditional value-at-risk. We develop battery scheduling decision-focused learning (BS-DFL) with a constraint-aware neural optimization layer trained directly from this objective without optimal-solution labels. The trained layer is then frozen to guide demand-predictor learning through scheduling costs and ramping risk, enabling solver-free rolling-horizon scheduling. In a 10,000-minute simulation using loads derived from measured AI workloads, BS-DFL reduces mean absolute minute-to-minute grid-import changes by 71.70% and total scheduling cost by 18.01% relative to an MSE-trained pre
    
[^108]: 自举式在线策略蒸馏

    Bootstrapped On-Policy Distillation

    [https://arxiv.org/abs/2611.13680](https://arxiv.org/abs/2611.13680)

    该论文发现在线策略自蒸馏（OPSD）在持续训练中会因冻结的特权教师与不断进化的无参考学生之间日益扩大的分布失配而导致性能退化，并提出自举式在线策略蒸馏方法来解决这一问题。

    

    在线策略蒸馏（OPD）通过在学生模型生成的轨迹上蒸馏教师模型的预测来改进语言模型，而在线策略自蒸馏（OPSD）则无需外部教师模型，它使用模型的一个以参考条件为前提的副本作为教师。尽管OPSD在训练早期有显著的改进效果，我们发现它在持续训练下并不能保持一致有效，最终可能出现严重的性能退化。我们研究了这种训练行为，并将性能退化归因于不断演化的无参考学生模型与冻结的特权教师模型之间日益增大的失配。在OPSD的早期阶段，特权教师与学生模型保持高度一致，能够为改进学生的生成轨迹提供有用的指导。然而，随着训练的推进，特权教师与无参考学生之间不断扩大的分布失配可能导致教师提供误导性的……（摘要在此处截断）

    arXiv:2611.13680v1 Announce Type: cross  Abstract: On-policy distillation (OPD) improves language models by distilling teacher predictions on student-generated trajectories, while on-policy self-distillation (OPSD) removes the need for an external teacher by using a reference-conditioned copy of the model as the teacher. Despite its great early improvements, we find that OPSD does not remain consistently effective under continued training and can eventually suffer substantial performance degradation. We study this training behavior and link the degradation to a growing mismatch between the evolving reference-free student and the frozen privileged teacher. In the early stage of OPSD, the privileged teacher remains closely aligned with the student and provides useful guidance for improving the student's generation trajectories. However, as training progresses, the growing distributional mismatch between the privileged teacher and reference-free student can cause the teacher to provide mi
    
[^109]: SepsisFlow：用于ICU脓毒症持续风险预警与病程轨迹建模的事件流基础模型

    SepsisFlow: an event-stream foundation model for continuous risk warning and trajectory modelling in ICU sepsis

    [https://arxiv.org/abs/2611.13661](https://arxiv.org/abs/2611.13661)

    SepsisFlow是一个基于257万份电子病历预训练Mamba模型的事件流基础模型，可在ICU脓毒症发病前实现6/12/24小时风险预警、发病时刻画发病情境、发病后建模72小时早期病程轨迹，其预警性能优于NEWS2、qSOFA、SIRS等传统评分及16种重训算法。

    

    脓毒症是ICU患者死亡的主要原因之一，其发病与病程具有高度异质性，然而大多数预警模型仅预测脓毒症是否会发生，一旦警报响起，临床医生对其发病情境和早期病程轨迹仍缺乏了解。为了在从ICU入院开始的统一小时级时间轴上回答这些问题，我们构建了SepsisFlow：在发病前，它将基于257万份电子健康记录预训练的Mamba模型与约1700个每小时特征相结合，通过梯度提升树预测6/12/24小时发病风险；在发病时，它通过门控网络融合Mamba编码器与表格编码器，生成连续的患者状态以刻画发病情境；在发病后，它通过带注意力机制的双向循环网络编码72小时的器官支持与生理轨迹，形成早期病程刻画。在MIMIC-IV数据集上的发病前预警评估中，在相同警报次数下，SepsisFlow检出的病例数多于NEWS2、qSOFA和SIRS，通过每小时更新每100例额外检出约8例，同时优于16个重新训练的对比算法。

    arXiv:2611.13661v1 Announce Type: new  Abstract: Sepsis is a leading cause of ICU death with heterogeneous onset and course, yet most warning models predict only whether sepsis will occur, leaving clinicians uncertain of its onset context and early trajectory once an alarm sounds. To answer these questions on one hourly clock from ICU admission, we built SepsisFlow: before onset, it combines a Mamba model pretrained on 2.57 million EHRs with about 1,700 hourly features in gradient-boosted trees to predict 6/12/24-h risk; at onset, it fuses Mamba and tabular encoders via a gated network into continuous patient states defining onset contexts; after onset, it encodes 72-h organ-support and physiological trajectories via an attention bidirectional recurrent network into early courses. For pre-onset warning in MIMIC-IV, SepsisFlow detected more cases than NEWS2, qSOFA, and SIRS at equal alarms, gaining about 8 cases per 100 from hourly updating, while outperforming 16 retrained algorithms a
    
[^110]: CP-MLPs：绑定与解绑MLP块的张量秩理论

    CP-MLPs: A Tensor-Rank Theory of Tied and Untied MLP Blocks

    [https://arxiv.org/abs/2611.13647](https://arxiv.org/abs/2611.13647)

    该论文提出CP-MLPs张量秩框架，将门控单元分解为检测、读取、写入三种角色，并严格证明：解绑门控块（如ReGLU/SwiGLU）在拟合不定二次交互和角色非对称目标时比绑定门控块（如ReLU²）具有严格的宽度效率优势，而绑定结构更匹配对称纯幂次目标，即解绑的收益取决于目标函数的结构。

    

    Transformer前馈块结合了多种设计选择：平方激活、门控机制，以及绑定与解绑的参数共享方式，而这些选择对表达能力的各自贡献仍知之甚少。我们提出了CP-MLPs，这是一个张量框架，其中每个乘法单元贡献一个秩-1交互，并承担三种角色：检测输入特征、读取数值以及写入输出。绑定的门控块（如ReLU$^2$）迫使检测器角色与数值角色共享同一方向；解绑的门控块（如ReGLU和SwiGLU）则允许二者不同。我们证明这一区分带来了精确的、按次数划分的张量秩分离：对于不定二次交互和角色非对称的单项目标，解绑的检测-数值单元在宽度效率上严格更优；而绑定单元在结构上则与对称纯幂次目标相匹配。因此，解绑所带来的宽度优势取决于目标函数的结构。实验表明损失……

    arXiv:2611.13647v1 Announce Type: new  Abstract: Transformer feed-forward blocks combine several design choices: squared activations, gating, and tied versus untied parameter sharing, whose separate contributions to expressivity remain poorly understood. We introduce CP-MLPs, a tensor framework in which each multiplicative unit contributes a rank-1 interaction with three roles: detecting an input feature, reading a value, and writing to the output. Tied gated blocks such as ReLU$^2$ force the detector and value roles to share the same direction; untied gated blocks such as ReGLU and SwiGLU let them differ. We show that this distinction yields exact, degree-wise tensor-rank separations: untied detector-value units are strictly more width-efficient for indefinite quadratic interactions and role-asymmetric monomials, while tied units are structurally matched to symmetric pure-power targets. The width advantage of untying therefore depends on the target structure. Experiments show loss pla
    
[^111]: MouldIt：注塑成型中多任务时序表征学习的数据效率

    MouldIt: Data Efficiency of Multi-Task Temporal Representation Learning in Injection Moulding

    [https://arxiv.org/abs/2611.13646](https://arxiv.org/abs/2611.13646)

    本文提出MouldIt——一个基于时序卷积与注意力机制的变分自编码器，用于注塑成型过程的多任务时序表征学习，并系统评估了其在不同数据规模、单模态与多模态设置下相比经典机器学习方法的数据效率与多模态收益。

    

    能够处理多项任务的智能工业解决方案需要能够理解底层工艺过程的模型。我们以注塑成型过程的建模为例，研究此类解决方案中的表征学习。我们考察了其相对于经典机器学习替代方法的数据效率以及多模态带来的收益。为此，我们基于Erlanger Träger零件的真实参考生产周期合成了一个融合工艺信息的数据集。随后，我们提出了MouldIt——一种基于时序卷积和注意力机制的变分自编码器，它能够重构模内信号，同时将零件分类为合格品或我们合成的某一种失效模式。我们在已知类分类和未知类检测这两项任务上，跨越九种数据集规模，在单模态和多模态设置下对MouldIt进行了评估，并将其与“全部接受”基线以及经典方法（k近邻、逻辑回归、随机森林等）进行比较。

    arXiv:2611.13646v1 Announce Type: new  Abstract: Smart industrial solutions that can address multiple tasks require models that understand the underlying processes. We examine representation learning for modeling the injection moulding process as an example of such solutions. We consider its data efficiency and multi-modal benefit compared to classical machine learning alternatives. To this end, we synthesize a process-informed dataset based on a real reference cycle of an Erlanger Tr\"ager part. We then introduce MouldIt, a temporal convolution and attention based variational autoencoder that reconstructs in-mould signals, while classifying parts as acceptable or as one of our synthetic failure modes. We evaluate MouldIt on two tasks, known-class classification and unknown-class detection, across nine dataset sizes in single-modal and multi-modal settings. We compare it with an accept-all baseline and the classical methods k-nearest neighbors, logistic regression, random forests, and 
    
[^112]: 面向融合型量子计算的实时机器学习译码器

    Machine-learning decoders for fusion-based quantum computing in real time

    [https://arxiv.org/abs/2611.13630](https://arxiv.org/abs/2611.13630)

    提出了PRISM——一种基于Transformer的机器学习译码器，可在高擦除率下为融合型光子量子计算提供实时高精度译码，其逻辑错误率与整数规划译码器相当而速度显著更快。

    

    容错量子计算需要兼具高准确性、高吞吐量、低延迟并可扩展至大量物理量子比特的译码器。在基于融合的光子架构中，光子损失和概率性融合测量所导致的擦除是主导性的错误机制，因此需要能够在高擦除率下仍保持准确性的译码器。我们提出了PRISM，这是一种基于Transformer的机器学习译码器，专为利用协议中擦除位置的已知信息而设计，可同时处理Pauli错误和擦除错误。我们在基于表面码、颜色码和量子低密度奇偶校验（LDPC）码的广泛容错协议上对PRISM进行了基准测试，码距最高达到24。在所有这些基准测试中，PRISM在逻辑错误率上与基于整数规划的译码器相当，同时速度显著更快，包括在错误率高达阈值50%的情况下依然表现出色。我们进一步优化了该译码器及其实时硬件实现。

    arXiv:2611.13630v1 Announce Type: cross  Abstract: Fault-tolerant quantum computation requires decoders that combine accuracy, throughput, low latency, and scalability to large numbers of physical qubits. In photonic fusion-based architectures, erasures from photon loss and probabilistic fusion measurements are a dominant error mechanism, motivating decoders that remain accurate at high erasure rates. We introduce PRISM, a transformer-based machine-learning decoder designed to exploit the known locations of erasures in protocols subject to both Pauli and erasure errors. We benchmark PRISM across a wide range of fault-tolerant protocols based on surface codes, color codes, and quantum low-density parity-check codes, with distances up to 24. Across these benchmarks, PRISM matches the logical error rates of integer-programming-based decoders while being significantly faster, including at error rates up to 50% of threshold. We further optimize the decoder and its real-time hardware impleme
    
[^113]: 基于学习反例的课程式证明修复

    Curriculum proof repair with learned counterexamples

    [https://arxiv.org/abs/2611.13627](https://arxiv.org/abs/2611.13627)

    提出TopoVerus框架，将反例引导的证明诊断内化为语言模型的自身能力，通过数据驱动的课程式训练使模型能够一次性生成有用的反例，从而高效修复Verus形式化证明。

    

    语言模型已经大幅降低了应用形式化验证来构建可靠且安全软件的门槛。现有方法通常遵循“验证器在环”范式，但验证器往往只提供粗糙的、二元的通过/失败反馈，对于应当如何修复证明几乎不提供可操作的信号。近期研究表明，反例通过将证明失败锚定在具体的程序状态上，能够提供更具可操作性的反馈，但在推理时生成有用的反例需要代价高昂的迭代细化。我们提出了TopoVerus，一个基于语言模型的Verus证明修复框架，它将反例引导的诊断内化为模型自身的能力，从而一次性生成有用的反例。我们的核心洞察是，将反例生成不作为独立的验证任务来训练，而是作为数据驱动的诊断目标：当一个反例在其条件下有助于……（原文摘要至此处被截断）

    arXiv:2611.13627v1 Announce Type: cross  Abstract: Language models (LMs) have substantially lowered the barrier to applying formal verification for building reliable and secure software. Existing approaches commonly follow the verifier-in-the-loop paradigm, but the verifier often provides only coarse, binary pass/fail feedback, which offers little actionable signal on how the proof should be repaired. Recent work shows that counterexamples provide more actionable feedback by grounding proof failures in concrete program states, but generating useful ones at inference time requires costly iterative refinement. We present TopoVerus, an LM-based framework for Verus proof repair that internalizes counterexample-guided diagnosis as a model capability, producing useful counterexamples in a single pass. Our key insight is to train counterexample generation not as a standalone validation task, but as a data-driven diagnostic objective: a counterexample is useful when conditioning on it helps th
    
[^114]: 数据稀缺条件下用于非侵入式模型降阶的多保真度算子推断方法

    Multifidelity Operator Inference for Non-intrusive Model Reduction under Data Scarcity

    [https://arxiv.org/abs/2611.13624](https://arxiv.org/abs/2611.13624)

    该论文提出了一种多保真度算子推断方法，以在计算预算有限、高保真训练数据稀缺的情况下实现非侵入式的模型降阶。

    

    多次查询分析（即对计算模型进行重复评估）在工程设计的不确定性量化和优化中发挥着重要作用。然而，当使用高保真数值模型时，特别是那些由偏微分方程（PDE）支配的模型，多次查询分析的代价可能高得令人望而却步。代理建模用廉价的近似替代昂贵的仿真，使实用的不确定性量化和设计工作流程成为可能。机器学习（ML）代理建模方法由于其非侵入性——即不需要访问底层仿真代码——在代理模型的开发中获得了广泛关注。其中，算子推断方法通过将降阶算子拟合到状态数据及对应的时间导数数据来构建模型。然而，由于计算预算有限，高保真训练数据十分稀缺，而由此训练的OpInf模型……

    arXiv:2611.13624v1 Announce Type: new  Abstract: Many-query analyses, in which a computational model is repeatedly evaluated, play an important role in uncertainty quantification and optimization for engineering design. However, many-query analyses can be prohibitively expensive when considering high-fidelity numerical models, especially those governed by partial differential equations (PDEs). Surrogate modeling replaces costly simulations with cheap approximations, enabling practical uncertainty quantification and design workflows. Machine learning (ML) approaches to surrogate modeling have gained significant traction in the development of surrogate models due to their non-intrusive nature; they do not require access to the underlying simulation code. The Operator Inference (OpInf) approach, in particular, fits reduced operators to state and corresponding time derivative data. However, high-fidelity training data is scarce due to limited computational budgets, and OpInf models trained
    
[^115]: Synapse-Control：通过轻量级运行时控制使智能体经验可操作化

    Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control

    [https://arxiv.org/abs/2611.13615](https://arxiv.org/abs/2611.13615)

    Synapse-Control用一个轻量级智能体运行时控制平面（规则筛选、1.7B微调分类器评分、微合同接口检查和提示价值表冷却机制）取代大模型失败诊断，以更低的推理成本实现智能体执行经验的高效复用与精准投放。

    

    大型语言模型智能体通过复用执行经验来提高其在复杂工具使用任务上的可靠性。诸如Sentry之类的系统在检索建议之前需要调用大模型生成失败诊断，这为经验复用增加了推理成本。我们提出Synapse-Control，这是一个智能体运行时，其轻量级控制平面负责选择建议并决定何时将其发送给执行任务的模型。规则首先筛选出相关建议，然后由一个在教师标注的状态-建议对上微调的轻量级1.7B分类器对这些候选建议进行评分，而不是生成诊断。微合同编码了可复用的接口检查，例如必需参数或有效选项，而无需规定完整的任务程序。提示价值表记录先前的指导和反馈，通过冷却机制和令牌预算检查来限制重复建议。在AppWorld和WebShop上每种方法进行1,308次尝试的评测中，Synapse-Control实现了270次成功（29.（摘要在此处截断）

    arXiv:2611.13615v1 Announce Type: cross  Abstract: Large language model agents reuse execution experience to improve reliability on complex tool-based tasks. Systems such as Sentry call large models to generate failure diagnoses before retrieving advice, adding inference cost to experience reuse. We present Synapse-Control, an agent runtime whose lightweight control plane selects advice and determines when to send it to the task-executing model. Rules shortlist relevant advice, and a lightweight 1.7B classifier fine-tuned on teacher-labeled state-suggestion pairs scores those candidates instead of generating a diagnosis. Micro-contracts encode reusable interface checks, such as required arguments or valid options, without prescribing complete task programs. A Prompt Value Table records earlier guidance and feedback to limit repeated suggestions through cooldown and token-budget checks. Across 1,308 attempts per method in AppWorld and WebShop, Synapse-Control achieves 270 successes (29.
    
[^116]: 乌兹别克语混合文字文本的OCR后纠错：基于经验同形字符噪声矩阵的字节级ByT5模型

    Post-OCR Correction of Mixed-Script Uzbek Text: Byte-Level ByT5 with an Empirical Homoglyph Noise Matrix

    [https://arxiv.org/abs/2611.13609](https://arxiv.org/abs/2611.13609)

    针对乌兹别克语中西里尔与拉丁字母混排导致的同形字符OCR错误，提出可在本地硬件上运行的字节级ByT5纠错系统，结合由人工标注错误事件估计的噪声矩阵与保守解码规则，将字符错误率从0.0578降至0.0369。

    

    在乌兹别克斯坦的官方和技术文档中，很大一部分OCR错误并非由图像质量差造成，而是由当前并用的两种字母表中视觉上完全相同的字符所引起。由于文字转型仍在进行中，西里尔字母与拉丁字母会出现在同一页面上，甚至经常出现在同一句话中；据我们的测量，此类页面上同形字符错误的密度约为单一文字文本的近五倍。我们在一个特殊受限的场景下研究该问题：工程地质与ICT基础设施报告语料库不得离开本机构，因此所有阶段均在本地硬件上执行。所提出的系统将字节级ByT5模型与一个噪声生成器相结合，该生成器的采样权重由359个人工分类的错误事件估计得出，并辅以受莱文斯坦距离约束的保守解码规则。在包含5,204个行对的黄金测试集上，该系统将字符错误率从0.0578降低至0.0369，并恢复了……

    arXiv:2611.13609v1 Announce Type: new  Abstract: In Uzbek official and technical documents, a large share of OCR errors is caused not by poor image quality but by characters that are visually identical across the two alphabets in use. Because the script transition is still under way, Cyrillic and Latin appear on the same page and often within a single sentence, and in our measurements the density of homoglyph errors on such pages is close to five times higher than in single-script text. We study this problem in a setting where the corpus of engineering-geological and ICT infrastructure reports may not leave the organisation, so every stage was executed on local hardware. The proposed system combines a byte-level ByT5 model with a noise generator whose sampling weights are estimated from 359 manually classified error events, together with a Levenshtein-bounded conservative decoding rule. On a Gold set of 5,204 line pairs it lowers the character error rate from 0.0578 to 0.0369, restores
    
[^117]: 一字节，一个词元：利用自生成数据将27B混合注意力大语言模型字节化

    One Byte, One Token: Byteifying a 27B Hybrid-Attention LLM with Self-Generated Data

    [https://arxiv.org/abs/2611.13607](https://arxiv.org/abs/2611.13607)

    研究者通过将词表缩减至384个条目（含256个字节符号），并利用原始模型自生成的4.77GB在线策略监督数据进行两阶段课程训练，成功将270亿参数的Qwen3.8混合注意力模型直接转换为纯字节级模型，无需补丁结构或局部编码器即可保留其大部分原有能力。

    

    字节级BPE会将中文文本碎片化（在中文维基百科上，Qwen3.8平均每个词元对应3.84字节；64%的词元是单个字符），并对模型隐藏了拼写信息。我们探究是否可以将一个已经训练好的27B子词模型直接转换为纯字节级模型（一个字节，一个词元），而无需补丁结构或局部编码器，同时保留其大部分能力。从Qwen3.8-27B（48层Gated DeltaNet层加16层全注意力层）出发，我们将词表从248,320个条目缩减至384个（BPE词表中已有的256个字节符号加上特殊标识），保持其他所有权重不变，并在4.77 GB的监督字节数据上进行两阶段课程训练（先编码，后阅读）：所有对话/行为监督数据均由原始模型自身以在线策略方式生成（包括字符词典、词语词典、通用对话、编程会话），知识阅读部分则使用百科文本。

    arXiv:2611.13607v1 Announce Type: new  Abstract: Byte-level BPE fragments Chinese text (for Qwen3.8 on Chinese Wikipedia, 3.84 bytes per token on average; 64% of tokens are single characters) and hides spelling from the model. We ask whether an already-trained 27B subword model can be converted directly into a pure byte-level model (one byte, one token), without a patching structure or a local encoder, while keeping most of its ability. Starting from Qwen3.8-27B (48 Gated DeltaNet layers + 16 full-attention layers), we shrink the vocabulary from 248,320 to 384 entries (the 256 byte symbols already in the BPE vocabulary plus special ids), leave every other weight untouched, and continue training in a two-stage curriculum (encoding first, then reading) on 4.77 GB of supervised bytes: all dialogue/behavioral supervision is generated on-policy by the original model itself (character dictionary, word dictionary, general dialogue, coding sessions), and knowledge reading uses encyclopedia tex
    
[^118]: 基于强化学习的单点交叉口交通信号与车道渠化统一优化

    Unified Optimization of Traffic Signal and Lane Channelization for an Isolated Intersection Using Reinforcement Learning

    [https://arxiv.org/abs/2611.13599](https://arxiv.org/abs/2611.13599)

    本文提出了一种基于强化学习的实时联合优化方法，可同时求解单点交叉口的交通信号控制与车道渠化设计，克服了传统混合整数线性规划（MILP）需随交通流模式变化反复求解且大规模问题时计算耗时的缺陷。

    

    优化交通信号控制一直是交通工程中一个长期存在的关键问题，迄今为止已经涌现出许多稳健的方法来解决这一问题。另一方面，车道渠化的道路设计是一个静态问题，难以与实时交通信号控制进行联合求解。如果交通量保持不变，则有可能将这两个问题作为一个单一的静态优化问题进行一体化求解。最近，研究人员设计了一种混合整数线性规划（MILP）来联合确定交通信号和道路宽度设计。然而，每当交通流模式发生变化时，这一联合优化问题都必须重新求解。而且随着问题规模的增大，求解MILP可能需要不切实际的计算时间。我们提出了一种基于强化学习（RL）的稳健方法，能够实时求解该联合优化问题。

    arXiv:2611.13599v1 Announce Type: new  Abstract: Optimizing traffic signal control has been a long-lasting critical issue in traffic engineering, and a number of robust methodologies have emerged to solve the problem so far. On the other hand, the road design for lane channelization is a static problem that is difficult to solve jointly with the real-time traffic signal control. If traffic volume does not vary, it is possible that both problems can be solved in an integrated manner as a single static optimization problem. Recently, researchers have devised a mixed integer linear problem (MILP) to jointly determine the traffic signal and road width designs. However, this joint optimization problem must be solved repeatedly whenever the traffic flow pattern varies. Solving the MILP may also take impractical computing time as the problem size gets larger. We present a robust methodology for solving the joint optimization problem using reinforcement learning (RL) on a real-time basis. Abru
    
[^119]: 迈向可靠的视觉语言模型评判者：状态条件不变性与呈现感知诊断

    Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics

    [https://arxiv.org/abs/2611.13596](https://arxiv.org/abs/2611.13596)

    该论文提出状态条件不变性（SCI）这一可靠性原则，并构建了基于模拟器的诊断基准GridWM-Judge，用于系统评估视觉语言模型评判者在语义不变的呈现变化下是否保持稳定、并对改变任务结果的因果干预作出正确响应。

    

    视觉语言模型（VLM）日益被用作多模态和具身任务的自动评判者，然而仅凭总体准确率或一致性无法揭示评判者改变其判决是否出于正确的原因。我们提出了状态条件不变性（SCI）这一可靠性原则，要求评判者在语义保持不变的呈现变化下保持稳定，同时对改变任务结果的因果干预作出相应变化。我们在GridWM-Judge中实现了SCI，这是一个基于模拟器的诊断基准，由确定性的MiniGrid轨迹构建，包含对齐的完整、证据移除和反事实变体，以及呈现与表示探针。通过五个诊断任务，GridWM-Judge测试局部感知、动作条件转移、呈现稳定性、表示可交换性以及Full-CF因果判别能力。我们的诊断评估揭示了异质性的失败模式，包括……

    arXiv:2611.13596v1 Announce Type: new  Abstract: Vision-language models (VLMs) are increasingly used as automated judges for multimodal and embodied tasks, yet aggregate accuracy or consistency alone does not reveal whether a judge changes its verdict for the right reason. We propose State-Conditional Invariance (SCI), a reliability principle requiring a judge to remain stable under semantics-preserving changes in presentation while responding to causal interventions that change the task outcome. We instantiate SCI in GridWM-Judge, a simulator-grounded diagnostic benchmark built from deterministic MiniGrid trajectories with aligned Full, evidence-removal, and counterfactual variants, plus presentation and representation probes. Across five diagnostic tasks, GridWM-Judge tests local perception, action-conditioned transitions, presentation stability, representation exchangeability, and Full-CF causal discrimination. Our diagnostic evaluations reveal heterogeneous failure modes, including
    
[^120]: 市场条件化JEPA：从不可预测的市场冲击中学习

    Market-Conditioned JEPA: Learning from Unpredictable Market Shocks

    [https://arxiv.org/abs/2611.13572](https://arxiv.org/abs/2611.13572)

    提出市场条件化JEPA（MC-JEPA），通过以留一法构建的市场因子作为预测条件，使资产的市场暴露成为最优预测的必要信息，并在11年中国A股分钟级数据上全面超越单一资产的JEPA方法。

    

    金融时间序列中的大部分变化是由市场范围的冲击驱动的，而这些冲击无法从资产自身的历史数据中预测出来。针对单一资产的自监督学习将这种变化视为不可约减的噪声。在潜在因子模型下，我们证明，将预测条件化于已实现的市场因子，使得资产的市场暴露成为最优预测的必要条件，同时降低了学习资产特定成分时需要对抗的噪声。基于这一洞察，我们提出了市场条件化JEPA（MC-JEPA），这是一种面向金融时间序列的联合嵌入预测架构。在训练过程中，我们从其他资产的未来目标嵌入中为每个资产构建留一法市场条件，并通过从资产自身表示中导出的因子载荷将其纳入模型。在11年的中国A股分钟级数据上，MC-JEPA在涵盖收益率、市场暴露等九个目标上均优于针对单一资产的JEPA。

    arXiv:2611.13572v1 Announce Type: cross  Abstract: Much of the variation in financial time series is driven by market-wide shocks that cannot be predicted from an asset's own history. Per-asset self-supervised learning treats this variation as irreducible noise. Under a latent factor model, we show that conditioning prediction on the realized market factor makes the asset's market exposure necessary for optimal prediction, while reducing the noise against which the asset-specific component is learned. Building on this insight, we introduce Market-Conditioned JEPA (MC-JEPA), a joint-embedding predictive architecture for financial time series. During training, we construct a leave-one-out market condition for each asset from the future target embeddings of other assets and incorporate it through factor loadings derived from the asset's own representation. On 11 years of minute-level Chinese A-share data, MC-JEPA improves over per-asset JEPA across nine targets spanning returns, market ex
    
[^121]: 对学习型交流最优潮流（AC-OPF）代理模型的蜕变测试揭示了其对表示的依赖性

    Metamorphic testing of learned AC optimal power flow surrogates reveals representation dependence

    [https://arxiv.org/abs/2611.13566](https://arxiv.org/abs/2611.13566)

    该论文提出用十三项蜕变关系系统审计学习型交流最优潮流代理模型，发现所有被测配置均存在对输入序列化表示的依赖性，并借此定位出索引错误与不稳定预处理等缺陷，同时证明数据增强可以缓解这种表示依赖问题。

    

    在单一的电网序列化编码上评估学习型交流最优潮流（AC-OPF）代理模型，无法证明其准确性是否依赖于该编码。我们通过十三个蜕变关系来检验这种依赖性，每个关系都规定了变换何时有效以及如何比较其输出。我们的审计涵盖17种模型与流水线配置：两个已发布的训练检查点和15个随机权重的架构或机制对照组。即使将报告容差收紧或放宽十倍，每种配置仍至少违反六个蜕变关系。对于GridSFM-Open v1.1，源代码检查和中间张量干预指向一个索引错误以及三个非规范或数值不稳定的预处理选择；修复后的检查点尚未经过测试。在单独的case500训练实验中，数据增强在四个模型中的三个上降低了变换视图误差，并在所有四个模型上降低了中值差异和约束违反的P95指标（原文摘要在此处截断）。

    arXiv:2611.13566v1 Announce Type: cross  Abstract: Evaluating a learned AC optimal power flow (AC-OPF) surrogate on one grid serialization cannot show whether its accuracy depends on that encoding. We test this dependence with thirteen metamorphic relations, each specifying when a transformation is valid and how to compare its outputs. Our audit covers 17 model and pipeline configurations: two released trained checkpoints and 15 random-weight architecture or mechanism controls. Every configuration violates at least six relations, even at reporting tolerances ten times tighter or looser. For GridSFM-Open v1.1, source inspection and intermediate-tensor interventions point to an indexing bug and three noncanonical or numerically unstable preprocessing choices; a repaired checkpoint remains untested. In a separate case500 training experiment, augmentation lowers transformed-view error in three of four models and median discrepancy and constraint-violation P95 in all four. These benefits de
    
[^122]: 面向科学设计的多目标贝叶斯优化中复合建模的研究

    Investigating Composite Modeling in Multi-Objective Bayesian Optimization for Scientific Design

    [https://arxiv.org/abs/2611.13565](https://arxiv.org/abs/2611.13565)

    本文系统研究了利用可观测中间响应的复合建模在多目标贝叶斯优化中对科学设计的收益，并探究了影响其效果差异的因素。

    

    科学设计通常需要在相互冲突的目标之间进行权衡，而这些目标通过实验或仿真进行评估的代价十分高昂。多目标贝叶斯优化（MOBO）通过使用概率模型来指导有前景设计的选择，从而应对这一挑战。在许多应用中，可调输入会映射到可观测的中间响应，随后通过已知函数将这些中间响应组合起来以计算最终目标。先前关于复合函数贝叶斯优化的研究表明，从这些中间响应中学习（我们称之为复合建模），与仅依赖目标函数黑盒观测的方法相比，能够显著提高单目标优化中的采样效率。然而，复合建模在MOBO中的收益仍缺乏充分研究。在本工作中，我们研究了复合建模在MOBO中的收益，并探究了导致其表现变化的因素。

    arXiv:2611.13565v1 Announce Type: cross  Abstract: Scientific design often requires balancing competing objectives that are expensive to evaluate through experiments or simulations. Multi-objective Bayesian optimization (MOBO) addresses this challenge by using probabilistic models to guide the selection of promising designs. In many applications, tunable inputs map to observable intermediate responses, which are then combined through known functions to compute the final objectives. Prior work on Bayesian optimization of composite functions shows that learning from these intermediate responses, which we call Composite modeling, can substantially improve sampling efficiency in single-objective optimization, compared to methods that condition only on black-box observations of the objective function. The benefits of Composite modeling in MOBO, however, remain understudied. In this work, we study the benefits of Composite modeling in MOBO and investigate the factors driving variation in its
    
[^123]: 利用大语言模型发现结构合理且可解释的认知模型

    Discovering Structurally Plausible and Interpretable Cognitive Models with Large Language Models

    [https://arxiv.org/abs/2611.13561](https://arxiv.org/abs/2611.13561)

    本文提出DisCo框架，利用大语言模型结合进化搜索，通过验证器过滤结构无效模型、清理器去除冗余参数，自动发现结构合理、参数可辨识且可解释的认知模型，其性能可与特定任务的基线模型相媲美。

    

    计算认知建模传统上依赖人工构建模型，这是一个耗时的过程，需要大量的领域专业知识。尽管近来基于大语言模型（LLM）的方法已经开始实现模型生成自动化，但它们往往以牺牲结构合理性和参数可辨识性为代价来优先追求预测准确性。我们提出了DisCo，这是一个由LLM引导的框架，通过进化搜索来发现可解释的符号化认知模型，该框架包含两个结构组件：一个验证器，用于过滤结构无效的模型；以及一个清理器，通过移除冗余参数来提升参数可辨识性。在三个行为任务中，DisCo持续发现性能与特定任务基线相当的模型。消融分析表明，验证器对于恢复具有心理学意义的结构至关重要。缺少该组件时，模型往往会违反基本的结构约束。

    arXiv:2611.13561v1 Announce Type: new  Abstract: Computational cognitive modeling has traditionally relied on manual model construction, which is a time-intensive process that requires substantial domain expertise. Although recent LLM-based approaches have begun to automate model generation, they often prioritize predictive accuracy at the expense of structural plausibility and parameter identifiability. We introduce DisCo, an LLM-guided framework that discovers interpretable symbolic cognitive models through evolutionary search with two structural components: a Validator that filters structurally invalid models, and a Cleaner that promotes parameter identifiability by removing redundant parameters. Across three behavioral tasks, DisCo consistently discovers models that achieve performance competitive with task-specific baselines. Ablation analyses show that the Validator is important for recovering psychologically meaningful structure. Without this component, models tend to violate be
    
[^124]: 后训练如何影响语言模型中的言语化算法计算

    How Post-Training Influences Verbalized Algorithmic Computation in Language Models

    [https://arxiv.org/abs/2611.13552](https://arxiv.org/abs/2611.13552)

    后训练主要改变的是言语化推理的有用性而非内部信息的编码方式——SFT 和 RLVR 后的模型内部均可读取进位信息，但 RLVR 产生的思维链在各检查点上显著提升了推理准确率。

    

    后训练提升了语言模型的推理能力，但它如何改变内部信息的使用以及言语化思维链（CoT）的作用仍不清楚。我们研究了 Olmo-3-7B-Instruct 在多位数加法任务上经过 SFT、DPO 和 RLVR 相继训练后的各个检查点，并将进位值作为已知的中间状态。探测方法在 SFT 和 RLVR 模型中都能以相似的准确率从内部激活中恢复进位信息，尽管操纵这些信息并不能可靠地控制模型输出。相比之下，言语化推理的有用性存在差异：RLVR 产生的思维链相比 SFT 或 DPO 产生的思维链，在各接收检查点上带来更高的准确率。我们还发现，RLVR 之后与进位相关的陈述比匹配的随机对照获得更高的重要性分数。DPO 对进位相关陈述表现出类似的敏感性，但没有相应的准确率提升。这些结果表明，后训练改变了书面推理的有用性……

    arXiv:2611.13552v1 Announce Type: new  Abstract: Post-training improves language-model reasoning, but how it changes the use of internal information and verbalized chain of thought (CoT) remains unclear. We study successive SFT, DPO, and RLVR checkpoints of Olmo-3-7B-Instruct on multi-digit addition, using carry values as known intermediate state. Probes recover carry information from internal activations with similar accuracy in SFT and RLVR, although manipulating this information does not reliably control model outputs. In contrast, the usefulness of verbalized reasoning differs: CoTs produced by RLVR yield higher accuracy across receiving checkpoints than those produced by SFT or DPO. We also show that carry-related statements receive higher importance scores after RLVR than matched random controls. DPO shows similar sensitivity to carry-related statements but without comparable accuracy gains. These results suggest that post-training changes the usefulness of written reasoning and 
    
[^125]: 金融预测中的表格基础模型

    Tabular Foundation Models in Financial Predictions

    [https://arxiv.org/abs/2611.13550](https://arxiv.org/abs/2611.13550)

    表格基础模型（TFMs）在预测公司下一年盈利变化方面优于微调大语言模型和基于树的模型，同时更快且计算成本更低，是会计研究中具有前景的预测工具。

    

    本技术简报将表格基础模型（TFMs）引入会计研究，并评估其在一项常见基准任务中的表现：预测公司下一年的盈利变化。TFMs 能生成数值预测并通过上下文进行学习，无需发布后的训练或调优，使其在预测应用中自然而易于使用。我们研究的 TFMs 在合成表格数据上进行预训练，因此避免了因记忆公司结果而产生的前瞻偏差。我们将 Google Research 的 TabFM 与近期会计研究中开发的微调开源权重大型语言模型以及若干基于树的模型进行比较。结果表明，TabFM 比所有对比模型都更准确，同时比大型语言模型更快、计算强度更低。虽然若干基于树的模型生成预测的速度比 TabFM 更快，但在它们之间进行选择需要在准确性和速度之间进行权衡。我们得出结论，TFMs 具有广阔的应用前景。

    arXiv:2611.13550v1 Announce Type: new  Abstract: This technical note introduces tabular foundation models (TFMs) to accounting research and evaluates their performance in one common benchmark task: predicting firms' next-year earnings changes. TFMs produce numeric predictions and learn in context, without post-release training or tuning, making them natural and easy to use for forecasting. The TFMs we study are pretrained on synthetic tabular data, so they avoid look-ahead bias from memorizing firms' outcomes. We compare Google Research's TabFM with a fine-tuned open-weight large language model developed in recent accounting research and with several tree-based models. TabFM is more accurate than all comparison models. It is also faster and less computationally intensive than the large language model. Several tree-based specifications produce predictions more quickly than TabFM, so choosing between them involves a trade-off between accuracy and speed. We conclude that TFMs have promise
    
[^126]: 巨大激活混淆稀疏自编码器

    Massive Activations Confound Sparse Autoencoders

    [https://arxiv.org/abs/2611.13548](https://arxiv.org/abs/2611.13548)

    该论文发现模型中的巨大激活使高达99.9%的激活方差集中于第一主成分（PC1），导致稀疏自编码器的评估指标失真、重要特征在不同运行间无法复现，并提出在去除PC1的激活上训练SAE、再将PC1加回重建结果的修正方法，将前50个特征的共享率从4–39.5%大幅提升至73–99.5%。

    

    稀疏自编码器（SAE）通常通过方差解释率和消融损伤来进行评估。我们展示了巨大激活如何破坏这两种评估，并提出了一种修正方法。巨大激活会导致高达99.9%的激活方差集中在第一主成分（PC1）方向上。由于PC1分量没有唯一的稀疏分解，每次SAE运行都会用一组不同的特征来重建它。这些特征主导了重要性排名：在50个最重要的特征中，平均有47个在另一次独立运行中无法匹配，相比之下随机选取的50个特征中约有27个无法匹配。此外，一个字典可以报告99.96%的方差解释率，而与此同时PC1之外仍有33%的方差未被解释。我们通过在去除了PC1的激活上训练SAE、再将PC1加回重建结果中，从而绕过PC1。在受影响的模型中，前50个特征的共享率从4–39.5%提升至73–99.5%。在相同的TopK稀疏度下，由此得到的SAE会降低模型性能……

    arXiv:2611.13548v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) are commonly evaluated by variance explained and ablation damage. We show how massive activations undermine both evaluations, and propose a correction. Massive activations cause up to 99.9% of activation variance to lie along the first principal component (PC1). Because the PC1 component has no unique sparse decomposition, each SAE run rebuilds it from a different set of features. These features dominate importance rankings: an average of 47 of the 50 most important features are unmatched in a separate run, compared with about 27 of a random 50. A dictionary can also report 99.96% of variance explained while leaving 33% of the variance outside PC1 unexplained. We bypass PC1 by training the SAE on activations with PC1 removed, then adding PC1 to the reconstruction. In affected models, the top-50 feature sharing rate rises from 4-39.5% to 73-99.5%. At the same TopK sparsity, the resulting SAE degrades the model
    
[^127]: 用随机舍入在强化学习后训练中缩小BF16与FP16的收敛速度差距

    Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic Rounding

    [https://arxiv.org/abs/2611.13545](https://arxiv.org/abs/2611.13545)

    该论文发现RL后训练中BF16因“权重粘滞”问题收敛慢于FP16，并提出在将FP32影子权重量化为BF16时采用随机舍入，从而在不增加与参考模型KL散度的前提下消除收敛速度差距。

    

    BF16因具有与FP32相同的动态范围、且收敛效果与FP32训练相近，被广泛用于大语言模型的预训练。然而，在采用可验证奖励的强化学习（RL）中，学习率通常较低，我们发现标准混合精度BF16训练在相同学习率下的收敛速度比FP16更慢。我们将这一差距归因于“权重粘滞”（weight sticking）现象：当使用就近舍入（round-to-nearest）方式将FP32影子权重量化为BF16时，较小的更新往往不会改变用于rollout生成和训练的BF16权重。因此，可能需要多次优化器迭代后，影子权重的更新才能体现在BF16权重中。提高BF16学习率可以减少权重粘滞并达到与FP16相当的收敛速度，但同时也会增加与参考模型之间的KL散度。我们转而在将FP32影子权重量化为BF16时应用随机舍入（stochastic rounding）……

    arXiv:2611.13545v1 Announce Type: new  Abstract: BF16 is widely used to pretrain large language models because it has the same dynamic range as FP32 and converges similarly to FP32 training. However, in reinforcement learning with verifiable rewards, where learning rates are typically low, we find that standard mixed-precision BF16 training converges more slowly than FP16 training at the same learning rate. We attribute this gap to weight sticking: when FP32 shadow weights are quantized to BF16 using round-to-nearest, small updates often leave the BF16 weights used for rollout generation and training unchanged. Consequently, several optimizer steps may be needed before updates to the shadow weights are reflected in the BF16 weights. Raising the BF16 learning rate reduces weight sticking and matches FP16's convergence speed, but it also increases the KL divergence from the reference model. We instead apply stochastic rounding when quantizing the FP32 shadow weights to BF16. This approac
    
[^128]: 注意力手术的合理性检查：面向表征驱动设计的零模型与匹配干预

    Sanity Checks for Attention Surgery: Null Models and Matched Interventions for Representation-Motivated Design

    [https://arxiv.org/abs/2611.13543](https://arxiv.org/abs/2611.13543)

    该论文提出三个可复用的合理性检查（噪声底线、几何零模型、匹配干预）来严格检验基于表征统计的架构修改，并发现自注意力移除中的头-值对齐很大程度上可由早期位置的因果零模型解释，而非模型真正依赖的特定表征方向。

    

    Transformer架构修改通常的做法是：在内部表征上测量某个统计量，观察到它数值很大，于是从架构上将其移除，并报告性能提升。要解释这一过程，需要评估三个问题：该统计量是否超过了噪声底线、它能否在几何零假设下依然成立、以及训练收益是否确实依赖于被移除的那个特定方向。我们将这些要求形式化为三个可复用的检查，并将其应用于“自注意力移除”方法——即删除注意力头输出中与其自身值向量对齐的分量。在九个多头预训练模型（共5,408个注意力头，参数量从1.24亿到69亿）上的实验表明，较早位置的因果零模型可以解释头-值余弦对齐的48%到63%，且不存在随规模单调变化的趋势。在分组查询注意力下，这种对齐是每个头所属键值组特有的，并在不同组之间发生反转（组内为+0.24至+0.27，组间为-0.11至-0.19……）

    arXiv:2611.13543v1 Announce Type: cross  Abstract: Transformer architectural modifications frequently measure a statistic on internal representations, observe that it is large, remove it architecturally, and report an improvement. Interpreting that sequence requires evaluating whether the statistic exceeds its noise floor, whether it survives a geometric null, and whether the training gain requires the specific direction removed. We formulate these requirements as three reusable checks and apply them to self-attention removal, which deletes the component of an attention head's output aligned with its own value vector. Across nine multi-head pretrained models (5,408 heads, 124M to 6.9B parameters), an earlier-position causal null accounts for 48% to 63% of the head-value cosine alignment, with no monotone scale trend. Under grouped-query attention, this alignment is specific to each head's own key-value group and reverses across groups (+0.24 to +0.27 within-group, -0.11 to -0.19 across
    
[^129]: 利用特征不变的缩放定律实现推荐系统中高效的A/B实验

    Exploiting Feature-invariant Scaling Laws for Efficient A/B Experimentation in Recommendation Systems

    [https://arxiv.org/abs/2611.13539](https://arxiv.org/abs/2611.13539)

    该论文发现现代推荐模型具有特征不变的缩放定律特性，并提出一种结合统计推断的方法，能仅通过小规模A/B实验预测全规模实验效果（含置信区间），从而大幅降低特征变更评估的计算成本。

    

    研究人员和工程师通过进行A/B实验来改进推荐系统，实验内容涉及新的模型架构、优化方法和数据特征。然而，针对每一次特征变更都运行全规模实验（例如使用在线数据进行全模型训练）的计算成本过高，而单个小规模实验的结果往往具有误导性。为了应对这一挑战，我们证明了现代推荐模型表现出特征不变的缩放特性：即使对基线模型应用不同的特征变更，模型性能的变化仍遵循可预测的缩放定律。我们提出了一种高效方法，利用缩放定律并结合统计推断来预测（并提供置信区间）推荐系统中A/B实验在全规模下的效果。这使得从业者能够仅通过小规模实验就高效地过滤掉中性或造成性能退化的特征变更。

    arXiv:2611.13539v1 Announce Type: cross  Abstract: Researchers and engineers improve recommendation systems by performing A/B experiments with new model architectures, optimization methods, and data features. However, running full-scale experiments (e.g., full-model training with online data) for every feature change is prohibitively computationally intensive, while the result from a single small-scale experiment is often misleading. To address this challenge, we show that modern recommendation models exhibit feature-invariant scaling: the change of a model's performance follows a predictable scaling-law even when different feature changes are applied to the baseline model. We introduce an efficient method that exploits scaling-laws and uses statistical inference to predict (with confidence intervals) the full-scale effect of an A/B experiment in recommendation systems. This enables practitioners to efficiently filter out neutral or regressive feature changes with only small-scale expe
    
[^130]: 自适应催化剂发现源于可组合的科学智能

    Adaptive catalyst discovery emerges from composable scientific intelligence

    [https://arxiv.org/abs/2611.13536](https://arxiv.org/abs/2611.13536)

    CatAI是一个物理驱动的模块化AI框架，通过学习180万个结构-性能关系并组织成30个可复用的物理化学编码器，在催化剂发现任务中预测误差降低7.8%，并在实验中发现活性达商业基准9倍的新型催化剂。

    

    催化是化学制造的基础，然而催化剂设计的知识仍分散于实验观测、计算数据以及催化的各子学科之间。为克服这一挑战，我们提出了可组合催化智能，这是一个以物理为基础的模块化人工智能框架，它学习180万个结构-性能关系，并将它们组织成30个物理化学编码器。这一可复用的中间层使任务特定的模型能够针对目标反应和目标来指导催化剂发现。在多项基准任务中，CatAI的任务平均预测误差比次优模型降低了7.8%。在实验验证中，CatAI发现了一种新型受限可通过空腔催化剂，可应用于工业均相聚合物合成，其活性达到商业基准催化剂的9倍。同时，它还指导了一种高活性铜基孪原子光催化剂的自主闭环优化……

    arXiv:2611.13536v1 Announce Type: new  Abstract: Catalysis underpins chemical manufacturing, yet knowledge for catalyst design remains dispersed across experimental observations, computational data and catalysis subdisciplines. To overcome this challenge, we present Composable Catalytic Intelligence (CatAI), a physics-grounded modular AI framework that learns 1.8 million structure-property relations and organizes them into 30 physicochemical encoders. This reusable intermediate layer enables task-specific models to guide catalyst discovery for targeted reactions and objectives. Across benchmark tasks, CatAI achieves a 7.8% lower task-averaged prediction error than the next-best models. In experimental validation, CatAI identified a new confined-passable pocket catalyst applicable in industrial homogeneous polymer synthesis with 9 times the activity of the commercial benchmark. In parallel, it guided autonomous closed-loop optimization of a highly active Cu-based geminal-atom photocatal
    
[^131]: 超越对数线性前沿：有限预算神经缩放定律的前渐近分析

    Beyond the Log-Linear Frontier: Pre-Asymptotic Analysis of Finite-Budget Neural Scaling Laws

    [https://arxiv.org/abs/2611.13534](https://arxiv.org/abs/2611.13534)

    本文通过 Mellin-Barnes 谱分析与 KKT 对偶理论刻画了有限预算下神经缩放定律偏离对数线性前沿的机制，提出了基于 Aitken 差分的良态闭式缩放定律估计方法，以及基于注水原理的多域数据最优采样策略。

    

    经典的神经缩放定律将超额风险建模为模型容量 $N$ 与数据集规模 $D$ 的可分离幂律。然而在有限预算下，经验学习曲线与计算最优分配会偏离这一对数线性前沿。本文分析了谱异质性与域异质性如何主导这些有限规模的偏离。在容量轴上，通过对协方差预解式进行 Mellin--Barnes 分析，得到了一个三项展开式，其中相互竞争的谱模式与正则化模式产生了有限尺度曲率和对数式过渡。由于在狭窄的试点网格上拟合多个幂律是病态问题，我们利用一阶 Aitken 差分消除损失下限，从而获得良态的闭式初始化器。在数据轴上，KKT 对偶性将 $K$ 域混合问题化归为标量注水问题，由此导出有限预算的瓶颈界、在线采样课程以及用于跨域迁移的单纯形 Hessian 度量。

    arXiv:2611.13534v1 Announce Type: new  Abstract: Classical neural scaling laws model excess risk as a separable power law in model capacity $N$ and dataset size $D$. At finite budgets, however, empirical learning curves and compute-optimal allocations depart from this log-linear frontier. This paper analyzes how spectral and domain heterogeneity govern these finite-scale departures. Along the capacity axis, Mellin--Barnes analysis of covariance resolvents yields a three-term expansion where competing spectral and regularization modes produce finite-scale curvature and logarithmic transitions. Because fitting multiple power laws on narrow pilot grids is ill-conditioned, we eliminate the loss floor via first-order Aitken differencing to obtain a well-conditioned closed-form initializer. Along the data axis, KKT duality reduces $K$-domain mixing to scalar water-filling, yielding finite-budget bottleneck bounds, an online sampling curriculum, and a simplex Hessian metric for cross-domain t
    
[^132]: RSI税：当递归自我改进得不偿失时

    The RSI Tax: When Recursive Self-Improvement Is Not Worth the Cost

    [https://arxiv.org/abs/2611.13524](https://arxiv.org/abs/2611.13524)

    本文提出“RSI税”概念以量化递归自我改进的token成本，发现有界笔记基线在准确率相当或更优的情况下可节省65.6-71.0%的总token开销，且验证成本占RSI税的74.8-93.6%，表明递归自我改进往往得不偿失。

    

    递归自我改进（RSI）旨在利用智能体自身的任务经验来改进该智能体。其实际价值取决于性能提升是否能证明自我改进和任务执行的全部成本是合理的。我们在持续任务流上评估这种权衡，在每个任务的反馈被用于后续更新之前对其进行评分。我们将“RSI税”定义为自我改进过程的token成本，并将其与任务执行成本一起纳入端到端评估。在六种方法和两个语言模型的实验中，一个有界笔记基线在观察到的准确率上匹配或超过了Meta-Harness、EvoSkill和RRSI，平均总token节省达65.6-71.0%。我们将RSI税分解为验证、探索和写作成本，并追踪相关状态或产物的后续使用情况。对于这三种RSI方法，验证成本占RSI税总额的74.8-93.6%。RSI税的很大一部分也……

    arXiv:2611.13524v1 Announce Type: new  Abstract: Recursive self-improvement (RSI) seeks to improve an agent using its own task experience. Its practical value depends on whether performance gains justify the full cost of self-improvement and task execution. We evaluate this trade-off on continual task streams, scoring each task before its feedback is used for subsequent updates. We define the \emph{RSI Tax} as the token cost of the self-improvement process and include it alongside task-execution costs in end-to-end evaluation. Across six methods and two language models, a bounded-note baseline matches or exceeds the observed accuracy of Meta-Harness, EvoSkill, and RRSI, with mean total-token savings of 65.6-71.0\%. We decompose the RSI Tax into verification, exploration, and writing costs, and track subsequent use of the associated state or artifact. For these three RSI methods, verification accounts for 74.8--93.6% of the aggregate RSI Tax. A substantial share of the RSI Tax is also a
    
[^133]: 高效评估非线性状态空间模型的并行Broyden方法

    Parallel Broyden Methods for Efficiently Evaluating Nonlinear State Space Models

    [https://arxiv.org/abs/2611.13523](https://arxiv.org/abs/2611.13523)

    该论文提出了一种基于Broyden方法的新型并行拟牛顿方法，通过对雅可比矩阵块进行对角加低秩近似来捕获状态维度间的耦合项，从而在不牺牲收敛速度和并行扫描兼容性的前提下，高效评估非线性状态空间模型。

    

    非线性状态空间模型通常需要按顺序进行评估，这限制了它们利用现代并行硬件的能力。近期研究表明，可以通过将评估重新表述为求根问题来实现状态空间模型的并行评估，该问题可用并行形式的牛顿法求解。然而，精确的牛顿法需要对密集雅可比矩阵进行昂贵的乘法运算，而基于对角近似的拟牛顿法虽然更具可扩展性，但由于忽略了状态维度之间的相互作用，往往收敛缓慢。我们提出了一种基于Broyden方法的新型拟牛顿方法，该方法通过对雅可比矩阵的块进行对角加低秩近似来捕获耦合项。我们从轨迹割线构建这些近似，在避免使用自动微分的同时，保留了对角近似良好的可扩展性，并保持了与并行扫描的兼容性。

    arXiv:2611.13523v1 Announce Type: new  Abstract: Nonlinear state space models are typically evaluated sequentially, limiting their ability to exploit modern parallel hardware. Recent work has shown that state space models can be evaluated in parallel by reformulating evaluation as a root-finding problem, which can be solved with a parallel form of Newton's method. However, exact Newton methods require expensive multiplication of dense Jacobian matrices, and quasi-Newton methods based on diagonal approximations, while more scalable, are often slow to converge because they neglect interactions across state dimensions. We introduce a novel quasi-Newton approach based on Broyden's method, which captures coupling terms with diagonal-plus-low-rank approximations to blocks of the Jacobian. We construct the approximations from trajectory secants, avoiding automatic differentiation while retaining the favorable scaling of diagonal approximations and remaining compatible with parallel scan throu
    
[^134]: 看见微小：面向视网膜病变分割的可变形对偶Mamba

    Seeing the Small: Deformable Duality Mamba for Retinal Lesion Segmentation

    [https://arxiv.org/abs/2611.13514](https://arxiv.org/abs/2611.13514)

    提出D²Mamba，通过将非因果状态空间对偶性与可变形Mamba扫描相结合来建模病灶的空间组织，在DDR数据集上实现33.87%的mIoU，其中微动脉瘤分割Dice提升10.06个百分点。

    

    糖尿病视网膜病变（DR）病灶形状差异极大，而微动脉瘤（MA）仅占几个像素。我们假设，通过结合全局上下文与形状自适应扫描来对病灶区域的空间组织进行建模，能够改善这类微小且不规则病灶的分割。为验证该假设，我们提出了D²Mamba，它将非因果状态空间对偶性与可变形Mamba扫描相结合。我们在DDR数据集上采用五折交叉验证进行评估，在相同的训练协议下与代表性分割基线方法进行对比，并对各组件进行了消融实验。D²Mamba取得了最佳的整体性能，mIoU达到33.87%，比最强基线高出2.99个百分点。最大的增益出现在微动脉瘤上——这是最小、最容易被漏检的病灶，其Dice系数提升了10.06个百分点。这些结果证实了我们的假设，即空间自适应的状态空间建模有利于微小病灶的分割。

    arXiv:2611.13514v1 Announce Type: cross  Abstract: Diabetic retinopathy (DR) lesions vary widely in shape, and microaneurysms (MAs) span only a few pixels. We hypothesize that modeling the spatial organization of lesion regions, with global context and shape-adaptive scanning, improves the segmentation of such small and irregular lesions. To test this, we propose D$^2$Mamba, which couples non-causal state space duality with deformable Mamba scanning. We evaluate it with five-fold cross-validation on the DDR dataset, comparing against representative segmentation baselines under an identical training protocol, and conduct ablations on each component. D$^2$Mamba achieved the best overall performance, with 33.87% mIoU, 2.99 percentage points above the strongest baseline. The largest gain appeared on MAs, the smallest and most easily missed lesions, where Dice improved by 10.06 percentage points. These results confirm our hypothesis that spatially adaptive state space modeling benefits tiny
    
[^135]: 基于条件熵的证据深度学习用于青光眼检测

    Evidential Deep Learning with conditional entropy for glaucoma detection

    [https://arxiv.org/abs/2611.13513](https://arxiv.org/abs/2611.13513)

    该论文提出了一种由新型条件熵函数正则化的证据深度学习方法用于青光眼检测，其不确定性估计的校准误差显著优于深度集成、MC Dropout等传统方法，并通过Grad-CAM焦点指数分析提供了更好的可解释性。

    

    我们提出了一种新颖的条件熵函数，作为证据深度学习（EDL）中量化预测不确定性的基础，用于从视网膜眼底图像进行青光眼检测。使用EyePACS-AIROGS-light-V2和RIM-ONE-DL数据集进行验证，我们发现由条件熵函数正则化的EDL模型所产生的不确定性估计，其校准误差显著低于深度集成、MC Dropout、事后校准以及基于Dempster不确定性的传统EDL框架。为了进一步探究模型的可解释性，我们使用代表空间集中度的焦点指数分析了Grad-CAM热力图。与其他不确定性量化方法相比，EDL模型在正确分类和错误分类的眼底图像的焦点指数分布之间表现出更大的分离度。特别是，假阴性病例与局部化的Grad-CAM激活相关。

    arXiv:2611.13513v1 Announce Type: cross  Abstract: We propose a novel conditional entropy function as a basis for quantifying predictive uncertainty in Evidential Deep Learning (EDL) for glaucoma detection from retinal fundus images. Using the EyePACS-AIROGS-light-V2 and RIM-ONE-DL datasets for validation, we found that our proposed EDL model regularized by conditional entropy function produced uncertainty estimates with substantially lower calibration errors than Deep Ensembles, MC Dropout, post-hoc calibration and the conventional EDL framework based on Dempster uncertainty. To further probe our models' interpretability, we analyzed Grad-CAM heatmaps using a focus index representing spatial concentration. Compared with the other uncertainty quantification methods, EDL models exhibited greater separation between the index distributions of correctly and incorrectly classified fundus images. In particular, false negatives were associated with localized Grad-CAM activations that were dis
    
[^136]: 基于机器学习模型的遥感影像分布偏移检测

    Distribution Shift Detection in Remote Sensing Imagery using Machine Learning Models

    [https://arxiv.org/abs/2611.13510](https://arxiv.org/abs/2611.13510)

    本文针对无人机遥感影像场景，实证评估了Alibi-Detect、NannyML和PyTorch-OOD三种开源分布偏移检测工具在跨数据集偏移下的检测有效性与计算延迟。

    

    由无人机（UAV）获取的遥感影像支持广泛的应用；然而，在资源受限的航空平台上部署机器学习（ML）模型会带来超出预测准确性之外的挑战，尤其是当部署时的数据与训练期间观察到的分布不同时。由于地理、传感器、季节及其他视觉变化，这类分布偏移在遥感领域十分常见。一种实用的解决方案是为已部署的模型配备能够检测新出现分布偏移的机制，并支持后续的响应行动。本文对三种开源检测工具进行了实证案例研究：(i) Alibi-Detect；(ii) NannyML；(iii) PyTorch-OOD。基于跨数据集的遥感偏移场景，我们评估了这些工具的检测有效性和计算延迟。我们的结果揭示了不同方法及实现在检测行为和处理成本方面的差异……

    arXiv:2611.13510v1 Announce Type: cross  Abstract: Remote sensing imagery acquired by Unmanned Aerial Vehicles (UAVs) supports wide range of applications; however, deploying machine learning (ML) models on resource-constrained aerial platforms introduces challenges beyond predictive accuracy, particularly when deployment data differ from the distributions observed during training. Such shifts are common in remote sensing due to geographic, sensor, seasonal, and other visual variations. A practical solution is to equip deployed models with mechanisms that detect emerging distribution shifts and support subsequent responsive actions. This paper presents an empirical case study of three open-source detection tools: (i) Alibi-Detect; (ii) NannyML; and (iii) PyTorch-OOD. Using cross-dataset remote-sensing shifts, we evaluate both detection effectiveness and computational latency of these tools. Our results reveal differences in detection behavior and processing cost across methods and imple
    
[^137]: 评估Transformer中的组合泛化：组合等价性与模块覆盖度的作用

    Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage

    [https://arxiv.org/abs/2611.13498](https://arxiv.org/abs/2611.13498)

    本文通过系统性评估发现，训练-测试分布中的组合等价性与模块覆盖度是决定 transformer 组合泛化性能的两个关键因素，直接模型的优异表现主要来自对组合等价性的利用，一旦消除这些等价性，性能将显著下降。

    

    组合泛化是一种重要的分布外泛化形式，其定义为：在若干模块组合（例如 sort、filter 等函数）上进行训练后，能够泛化到未见过的组合的能力。尽管已有大量实证工作评估了 transformer 的组合泛化能力，但由于评估往往只在有限的、以临时方式定义的训练-测试划分上进行，研究结果并不一致。通过系统性评估，我们识别出训练-测试分布中强烈影响性能的两个关键方面：组合等价性和模块覆盖度。我们表明，直接模型（仅在最终输出上训练）所展现的性能主要源于对组合等价性的利用，即不同的模块序列可以归约为相同的端到端函数。当基准测试消除这些等价性时，模型性能会显著下降。

    arXiv:2611.13498v1 Announce Type: cross  Abstract: Compositional generalization is an important form of out-of-distribution generalization, defined as the ability to train on some combinations of modules (e.g., functions such as sort, filter) and then generalize to unseen combinations. Despite substantial empirical work evaluating compositional generalization in transformers, findings have been inconsistent because evaluations were often conducted on a limited set of train-test splits, defined in an ad hoc manner. Through a systematic evaluation, we identify two key aspects of the train-test distribution that strongly affect performance: composition equivalence and module coverage. We show that the apparent performance of direct models (trained only on final outputs) results primarily from exploiting composition equivalences, meaning different sequences of modules that reduce to identical end-to-end functions. When benchmarks eliminate these equivalences, performance drops significantl
    
[^138]: 连续仿真优化中的同时近优区域识别：极小极大复杂度与自适应采样

    Simultaneous Near-Optimal Region Identification in Continuous Simulation Optimization: Minimax Complexity and Adaptive Sampling

    [https://arxiv.org/abs/2611.13489](https://arxiv.org/abs/2611.13489)

    本文证明在连续设计空间上同时识别所有容差水平下的近优区域并非优化的副产品，给出了极小极大采样复杂度下界并提出了达到该下界的自适应采样方法。

    

    仿真优化通常只返回单一估计的最优点。当决策还依赖于独立于仿真进行评估的其他准则（如资源成本）时，决策者需要的是其期望性能落在最优值容差范围内的设计集合，而且往往只有在看到权衡关系之后才能确定容差。我们研究如何在连续设计空间上识别这些近优区域，对所有不超过规定最大值 $T$ 的容差同时进行识别，并将区域估计误差控制在 $\eta$ 以内。我们证明，近优区域识别并非优化的副产品。具体而言，当 $\alpha>\nu$ 时，任何算法，即使事先已知最优设计，也至少需要 $\Omega\bigl(\eta^{-(2+d/\nu)}\min\{1,KT^{d/\alpha}\}\bigr)$ 次仿真评估，其中 $d$ 为维度，$\nu$ 为光滑度，$(K,\alpha)$ 为刻画近优区域几何形状的覆盖参数。

    arXiv:2611.13489v1 Announce Type: cross  Abstract: Simulation optimization typically returns a single estimated optimal point. When decisions also depend on criteria evaluated independently of simulation, such as resource cost, the decision maker instead needs the set of designs whose expected performance lies within a tolerance of the optimum, often choosing the tolerance only after seeing the tradeoffs. We study how to identify these near-optimal regions on a continuous design space, simultaneously for all tolerances up to a prescribed maximum $T$, with region estimation error at most $\eta$. We show that region identification is not a by-product of optimization. Specifically, when $\alpha>\nu$, any procedure, even one given the optimal design in advance, needs at least $\Omega\bigl(\eta^{-(2+d/\nu)}\min\{1,KT^{d/\alpha}\}\bigr)$ simulation evaluations, where $d$ is the dimension, $\nu$ the smoothness, and $(K,\alpha)$ the covering parameters that describe the geometry of the near-op
    
[^139]: 基于成对比较的排名的差分隐私统计推断

    Differentially Private Statistical Inference for Rankings from Pairwise Comparisons

    [https://arxiv.org/abs/2611.13485](https://arxiv.org/abs/2611.13485)

    该论文首次在Bradley-Terry模型框架下实现了边差分隐私下的排名统计推断，通过留一分析给出有限样本误差界、渐近有效置信区间以及结合岭偏差校正与高斯-拉普拉斯校准的偏差修正方法，适用于保护隐私的大语言模型成对比较评估。

    

    基于人类成对比较的大语言模型排名的可靠推断必须保护个体偏好记录的隐私。为满足这一需求，我们在Bradley-Terry模型下开发了具有边差分隐私的分数与排名推断方法。针对目标扰动估计器，一种新颖的留一分析建立了欧几里得范数和逐元素估计误差的有限样本界。对于足够稠密的比较图，这些误差界可延伸至更小的隐私预算，包括隐私项主导统计项的情形。一致渐近展开为个体分数以及成对分数差异提供了渐近有效的置信区间。为改善有限样本性能，我们通过将即插即用的岭回归偏差校正与高斯-拉普拉斯校准相结合，同时考虑了正则化偏差和隐私引入的随机性。我们进一步建立了自助法（摘要在此处被截断）。

    arXiv:2611.13485v1 Announce Type: cross  Abstract: Reliable inference for rankings of large language models based on human pairwise comparisons must preserve the privacy of individual preference records. To meet this need, we develop methods for score and rank inference under the Bradley-Terry model with edge differential privacy. For an objective-perturbed estimator, a new leave-one-out analysis establishes finite-sample bounds on Euclidean and entrywise estimation errors. For sufficiently dense comparison graphs, the error bounds extend to smaller privacy budgets, including regimes in which its privacy term dominates its statistical term. A uniform asymptotic expansion yields asymptotically valid confidence intervals for individual scores and pairwise score differences. To improve finite-sample performance, we account for both regularization bias and privacy-induced randomness by combining a plug-in ridge-bias correction with Gaussian-Laplace calibration. We further establish a boots
    
[^140]: KL正则化强化学习的精确样本复杂度

    Sharp Sample Complexity of KL-Regularized Reinforcement Learning

    [https://arxiv.org/abs/2611.13480](https://arxiv.org/abs/2611.13480)

    本文证明了在KL正则化强化学习中，插件估计器仅需Õ(KH³/(ε·max{ε,β})+KH)个样本即可找到ε-最优策略，并给出了匹配的下界，将先前样本复杂度对有效视界的依赖从H⁵改进为H³，且首次建立了依赖温度β的下界。

    

    KL正则化强化学习——即最大熵强化学习和基于人类反馈的强化学习（RLHF）背后的目标——本质上是“控制即推断”图模型中的变分推断：正则化最优策略是轨迹上的后验分布，而策略的次优性等于β乘以路径测度之间的KL散度。然而，即使拥有生成模型，学习这一后验分布需要多少样本仍是未知的。此前的1/ε保证在有效视界H上按H⁵的规模增长，且针对MDP的下界均不依赖于温度β。在这项工作中，我们证明了对于具有K个状态-动作对的折扣MDP，插件估计器仅需Õ(KH³/(ε·max{ε, β}) + KH)个样本，就能找到一个从每个初始状态出发均为ε-最优的策略；并且当ε ≲ min{H, H²/β}时，即使是可以自主选择采样位置的算法，也无法用更少的样本达到这一目标。

    arXiv:2611.13480v1 Announce Type: cross  Abstract: KL-regularized reinforcement learning, the objective behind maximum-entropy RL and RL from human feedback, is variational inference in the graphical model of "control as inference": the regularized optimal policy is a posterior over trajectories, and the suboptimality of a policy is $\beta$ times a KL divergence between path measures. However, it is not known how many samples it takes to learn this posterior, even with a generative model. Earlier $1/\epsilon$ guarantees scale with $H^5$ in the effective horizon $H$, and no lower bound for MDPs depends on the temperature $\beta$. In this work, we show that for a discounted MDP with $K$ state-action pairs, the plug-in estimator finds a policy that is $\epsilon$-optimal from every initial state with $\widetilde{O}(KH^3/(\epsilon\max\{\epsilon,\beta\}) + KH)$ samples, and that no algorithm, even one that chooses where to sample, can do with fewer when $\epsilon \lesssim \min\{H, H^2/\beta\
    
[^141]: 利用心理物理学注意力分布进行注视增强的奖励建模

    Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling

    [https://arxiv.org/abs/2611.13477](https://arxiv.org/abs/2611.13477)

    该论文基于认知心理学中人类字母识别的发现，将视觉注意力建模为以注视点为中心的空间梯度分布并融入RLHF奖励模型训练，从而实现了比基线更高的偏好预测准确率。

    

    将人类反馈整合到语言模型中作为使模型输出与人类偏好对齐的手段，正受到越来越多的关注，其中基于人类反馈的强化学习（RLHF）是基于偏好的对齐的重要框架。最近的研究将眼动追踪（ET）数据作为奖励模型训练的额外监督信号，将奖励学习建立在人类阅读行为的基础上。在这一方法的基础上，我们利用认知心理学中关于人类字母识别的既有发现，对以往以注视为中心的模型进行了扩展。具体而言，我们将视觉注意力建模为以当前注视位置为中心的空间梯度分布，并将这一表示融入奖励模型训练中。我们的结果表明，使用所提出的注视分布训练的奖励模型比基线模型取得了更高的偏好预测准确率。此外，我们的一些结果……

    arXiv:2611.13477v1 Announce Type: cross  Abstract: Integrating human feedback into language models has gained increasing attention as a means to align model outputs with human preferences, with Reinforcement Learning from Human Feedback (RLHF) serving as a prominent framework for preference-based alignment. Recent studies have incorporated eye-tracking (ET) data as an additional supervisory signal for reward model training, grounding reward learning in human reading behavior. Building on this approach, we extend prior fixation-centric models by leveraging established findings from cognitive psychology on human letter recognition. Specifically, we model visual attention as a spatially graded distribution centered on the current gaze location and incorporate this representation into reward model training. Our results show that the reward model trained with the proposed gaze distribution achieves higher preference prediction accuracy than a baseline model. Moreover, some results from our 
    
[^142]: 深度个性化：面向本地主动式智能体的持续上下文观察

    Deep Personalization: Continuous Context Observation for Local Proactive Agents

    [https://arxiv.org/abs/2611.13457](https://arxiv.org/abs/2611.13457)

    提出深度个性化闭环框架，通过独立的观察者智能体持续监控用户桌面活动、维护紧凑的长期用户上下文，并从隐式反馈中学习最佳干预时机，从而构建能够恰到好处地主动提供帮助的桌面智能体Coco。

    

    随着AI智能体的能力日益增强，用户仍然需要自行判断何时以及如何寻求帮助，这导致许多有价值的支持机会未能实现。主动式智能体有望弥合这一差距，但有效的主动性需要理解用户的实时活动并个性化干预时机，从而在帮助可能有益而非造成干扰时才提供协助。我们提出了“深度个性化”这一闭环过程，它将持续上下文观察与智能体的持续适应相连接：智能体观察用户活动，有选择地保留长期上下文，并根据隐式反馈调整干预决策。我们在Coco——一个主动式桌面智能体——中率先实现深度个性化，通过一个独立的观察者智能体来监控桌面活动、维护紧凑的长期用户建模上下文，并从参与度、忽略等隐式信号中学习何时提供帮助是合宜的。

    arXiv:2611.13457v1 Announce Type: new  Abstract: As AI agents become increasingly capable, users must still recognize when and how to ask for help, leaving many opportunities for useful support unrealized. Proactive agents hold the promise to close this gap, but effective proactivity requires understanding users' real-time activities and personalizing intervention timing so that assistance is offered when it is likely to be helpful rather than disruptive. We introduce deep personalization, a closed-loop process that connects continuous context observation with continual agent adaptation: an agent observes user activity, selectively retains long-term context, and adapts intervention decisions from implicit feedback. We initiate deep personalization in Coco, a proactive desktop agent, by having a separate Observer Agent that monitors desktop activity, maintains compact long-term user modeling context, and learns when assistance is desirable from implicit signals such as engagement, dismi
    
[^143]: SLIC：通过强化微调小语言模型实现多轮模拟电路优化

    SLIC: Reinforcement Fine-Tuning Small LMs for Multi-Turn Analog Circuit Optimization

    [https://arxiv.org/abs/2611.13452](https://arxiv.org/abs/2611.13452)

    本文提出SLIC强化微调框架，将40亿参数的本地部署小语言模型训练用于闭环多轮模拟电路优化，其核心创新在于结合结构化电路表示、门控多目标奖励和奖励引导滚动规划（RGRS），并发布了包含41种放大器拓扑的开放基准测试集。

    

    模拟集成电路（IC）设计是一项多轮、以仿真器为基础的工程任务，需要对相互耦合的性能权衡进行迭代推理，然而专有的工艺设计套件（PDK）使得托管式大型语言模型工作流难以在实际设计环境中部署。我们提出了SLIC（Small LM Iterating on Circuits，小语言模型电路迭代），这是一个强化微调框架，用于训练一个40亿参数规模、可本地部署的语言模型，以实现闭环模拟电路优化。SLIC将结构化电路表示与离散化的相对更新动作相结合，采用领域门控和成功门控的多目标奖励，并引入奖励引导的滚动规划技术（Reward-Guided Rollout Steering，RGRS），将每一步的仿真器反馈转化为策略学习和在线滚动构建的信号，从而应对这一稀疏且高度受限的领域。我们进一步发布了一个开放的、目标条件化的基准测试集，包含41种放大器拓扑结构，并为分布内场景提供了标准化协议。

    arXiv:2611.13452v1 Announce Type: new  Abstract: Analog integrated circuit (IC) design is a multi-turn, simulator-grounded engineering task that demands iterative reasoning over coupled performance tradeoffs, yet proprietary process design kits (PDKs) make hosted large language model workflows difficult to deploy in practical design settings. We present SLIC (Small LM Iterating on Circuits), a reinforcement fine-tuning framework that trains a 4B-scale, locally deployable language model for closed-loop analog circuit optimization. SLIC pairs structured circuit representations with discretized relative-update actions, domain- and success-gated multiobjective rewards, and Reward-Guided Rollout Steering (RGRS), which turns per-step simulator feedback into signals for both policy learning and online rollout construction in this sparse, highly constrained domain. We further release an open, goal-conditioned benchmark of 41 amplifier topologies with standardized protocols for in-distribution 
    
[^144]: 费马ProbCover：基于费马距离图的覆盖式主动学习

    Fermat ProbCover: Coverage-Based Active Learning on Fermat Distance Graphs

    [https://arxiv.org/abs/2611.13448](https://arxiv.org/abs/2611.13448)

    提出PC-Fermat方法，通过加权图上的截断最短路径算法以费马距离替代欧氏距离来构建覆盖邻域，从而在低预算主动学习中更好地捕捉数据的底层分布几何结构。

    

    主动学习旨在通过选择最具信息量的样本进行标注来降低标注成本。在低预算场景下，基于不确定性的方法往往失效，而ProbCover等基于覆盖的主动学习方法已被证明是有效的。然而，这些方法通常依赖于欧氏距离，而欧氏距离可能无法充分捕捉数据的底层几何结构。在本文中，我们提出了PC-Fermat，这是一种基于覆盖的主动学习方法，它用由费马距离定义的邻域取代了欧氏覆盖邻域。费马距离是一种数据自适应的距离，能够反映数据点的底层分布。为了便于计算，我们构建了一个加权欧氏邻域图，并使用截断最短路径算法来计算费马覆盖图。我们还建立了充分条件，在该条件下这种构造能够精确地保持在完整数据上计算的费马覆盖。

    arXiv:2611.13448v1 Announce Type: new  Abstract: Active learning aims to reduce annotation cost by selecting the most informative examples for labeling. In the low-budget regime, where uncertainty-based methods often fail, coverage-based active learning methods such as ProbCover have been shown to be effective. However, these methods typically rely on Euclidean distance, which may not adequately capture the underlying geometry of data. In this article, we propose PC-Fermat, a coverage-based active learning method that replaces Euclidean coverage neighborhoods with neighborhoods defined by a Fermat distance, i.e., a data-adaptive distance that reflects the underlying distribution of the data points. To facilitate computation, we construct a weighted Euclidean neighborhood graph and use a truncated shortest-path algorithm to compute the Fermat coverage graph. We also establish sufficient conditions under which this construction exactly preserves the Fermat coverage computed on the comple
    
[^145]: ForeSpark：面向并行投机解码的草稿与验证预算联合调度

    ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding

    [https://arxiv.org/abs/2611.13442](https://arxiv.org/abs/2611.13442)

    该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。

    

    arXiv:2611.13442v1 公告类型：cross 摘要：并行草稿器通过在单次前向传播中提出一整块草稿token来加速LLM解码，使其对LLM服务具有很强的吸引力。近期的系统优化的是有效吞吐量（goodput，即每秒被接受的token数），因为被拒绝的草稿会抢占其他请求的计算资源。然而，这些系统只能控制验证多少个token，因此每个请求仍然会草稿出完整的一块token。在高服务负载下，我们发现草稿生成会消耗高达40%的步进时间，其中大部分因从未被验证而被丢弃。而且，来自自回归草稿器的草稿长度控制方式无法直接迁移到并行草稿器，因为并行草稿器必须在其唯一一次前向传播之前就确定其长度。此外，草稿长度与验证长度在整个批次中相互耦合，因此必须为每个请求联合选择这两者，同时保证该决策在每一步的计算开销都足够低。在本文中，我们提出节省工作的投机（WCS）设计，该设计联合分配……

    arXiv:2611.13442v1 Announce Type: cross  Abstract: Parallel drafters accelerate LLM decoding by proposing a whole block of draft tokens in a single forward pass, making them attractive for LLM serving. Recent systems optimize goodput (i.e., accepted tokens per second) because rejected drafts steal compute from other requests. However, these systems control only how many tokens are verified, so every request still drafts a full block. At high serving load, we find that drafting consumes up to 40% of step time, of which a majority is discarded because it is never verified. Yet, draft-length control from autoregressive drafters does not carry over, because a parallel drafter commits to its length before its single pass. Moreover, draft and verification lengths are coupled across the batch, so they must be selected jointly for every request while keeping the decision computationally cheap at every step. In this paper, we propose work-conserving speculation (WCS), a design that jointly allo
    
[^146]: 做上下文学习并不需要特征学习

    You Don't Need Feature Learning to Do In-Context Learning

    [https://arxiv.org/abs/2611.13431](https://arxiv.org/abs/2611.13431)

    通过对核机器的谱分析发现，上下文学习中的泛化解只是记忆解的低阶截断——模型先泛化后逐渐记忆化，因此实现上下文学习并不需要特征学习。

    

    我们应该如何理解上下文学习（ICL）任务中的记忆与泛化，以及何时发生记忆、何时发生泛化？为了研究这一问题，我们研究了旋转不变核机器如何学习Raventos等人（2023）提出的ICL线性回归任务。通过在核特征基中对该任务进行分解，我们得到了这一ICL设置下学习的谱图像。这一图像揭示，随着样本量的增长，核机器会依次学习记忆解中越来越高的阶数分量，而这些分量依赖于训练任务集合中越来越高的阶数统计量。泛化解近似为记忆解的低阶截断，因此模型首先实现泛化，随后随着训练的进行逐渐退化为记忆。结论有三点。第一，如果这一情形具有普遍指示意义，那么在ICL任务上学习“泛化”解，实际上只是对记忆解的欠拟合。

    arXiv:2611.13431v1 Announce Type: cross  Abstract: How should we understand memorization vs. generalization in in-context learning (ICL) tasks, and when does one happen vs. the other? To investigate, we study how rotation-invariant kernel machines learn the ICL linear regression task proposed by Raventos et al. (2023). By decomposing the task in the kernel eigenbasis, we obtain a spectral picture of learning in this ICL setting. This picture reveals that as sample size grows, kernel machines learn successively-higher-degree components of the memorizing solution, which depend upon successively-higher-degree statistics of the training task set. The generalizing solution is approximately a low-degree truncation of the memorizing solution, and so the model first generalizes, then degrades into memorization as training proceeds.   The takeaways are threefold. First, if this case is indicative, learning a "generalizing" solution on an ICL task is really just underfitting the memorizing solut
    
[^147]: 免费午餐式的不确定性：基于粒子解码的序列级Rényi熵用于自适应LLM推理

    Free Lunch Uncertainty: Sequence-Level R\'enyi Entropy from Particle Decoding for Adaptive LLM Reasoning

    [https://arxiv.org/abs/2611.13414](https://arxiv.org/abs/2611.13414)

    该论文发现SMC幂采样解码可零边际成本获得序列级Rényi熵，其中1/2阶熵在13个设置中的10个中表现最佳，是用于LLM自适应推理的零成本选择性预测信号。

    

    量化LLM整个补全结果的不确定性对于选择性预测和自适应计算至关重要。零成本信号（如平均token熵和对数概率）仅局限于token级别或单个补全，而自洽性方法则需要许多额外的补全。我们证明，用于幂采样的序贯蒙特卡洛（SMC）解码本身已经在估计幂分布的归一化常数，这使我们能以零边际成本获得模型补全分布的序列级Rényi熵（SRE），且在每个粒子预算下都有保证。通过Rényi链式法则，我们将这种读出扩展到任意阶数的逐步熵族。我们证明高于1的阶数与每步最可能的token相关联，这解释了它们为何与对数概率保持一致，而1/2阶成员则累积每步搜索努力的对数。该成员在13个设置中的10个中是最佳的零成本选择性预测信号。在MATH5上

    arXiv:2611.13414v1 Announce Type: cross  Abstract: Quantifying uncertainty over an entire LLM completion is critical for selective prediction and adaptive compute. Zero-cost signals such as mean token entropy and log-probability are local to tokens or to one completion, and self-consistency needs many extra completions. We show that Sequential Monte Carlo (SMC) decoding for power sampling already estimates the normalizing constant of the power distribution, which gives the sequence-level R'enyi entropy (SRE) of the model's completion distribution at zero marginal cost, with a guarantee at every particle budget. A R'enyi chain rule extends this readout to a family of stepwise entropies of any order. We prove that orders above one are tied to the most likely token of each step, which explains why they track log-probability, while the order-1/2 member accumulates the log of per-step search effort. This member is the best zero-cost selective-prediction signal in 10 of 13 settings. On MATH5
    
[^148]: 基于逐层预取时限的大语言模型查找表放置

    Placing LLM Lookup Tables by Per-Layer Prefetch Deadlines

    [https://arxiv.org/abs/2611.13413](https://arxiv.org/abs/2611.13413)

    提出 SlackTier，利用逐层预取时限（两个 Engram 层窗口相差 30 倍）将 LLM 的大规模 n-gram 查找表按层放置到不同存储层级，在解码 CUDA 图内实现位精确查找，使部分表可无需缓存直接放在 NVMe 上而几乎不损失性能。

    

    DeepSeek-V4.1 的 203 GB Engram n-gram 表目前存放在主机 DRAM 中，占用了 4-GPU 推理进程所用 223 GiB 内存中的 190 GiB。由于行索引仅取决于输入 token，读取可以在每一步开始时就发起，并与较早层的计算重叠。V4.1 的两个 Engram 层的预取窗口相差 30 倍（0.3 ms 对 9.3 ms）。我们证明，这一时限连同会话内的行复用以及对共享设备的争用，决定了每张表可以放在哪个存储层级、哪些层需要缓存，并且我们给出了一个可根据实测读取时间预测由此产生的性能下降的模型。智能体和聊天流量会在会话内复用行，而静态热集的迁移效果不佳。我们构建了 SlackTier，这是面向 SGLang 的一个表层级，它在步骤开始时发出所有查找，先读取短窗口层，并将一个小的 GPU 行缓存跨层拆分。该路径运行在解码 CUDA 图内部，并与原始 gather 操作位精确一致。第 14 层的表可以放在不带缓存的 NVMe 上，而没有可测量的

    arXiv:2611.13413v1 Announce Type: cross  Abstract: DeepSeek-V4.1's 203 GB of Engram n-gram tables sit in host DRAM today, taking 190 of the 223 GiB a 4-GPU serving process uses. Row indices depend only on input tokens, so reads can start with each step and overlap earlier layers. V4.1's two Engram layers have windows that differ by 30x (0.3 vs. 9.3 ms). We show that this deadline, together with session reuse and contention on a shared device, decides which tier each table can use and which layers need cache, and we give a model that predicts the resulting slowdown from measured read times. Agent and chat traffic reuse rows within sessions, and static hot sets transfer poorly. We build SlackTier, a table tier for SGLang that issues all lookups at step start. It reads the short-window layer first and splits a small GPU row cache across layers. The path runs inside the decode CUDA graph and is bit-exact with the original gather. Layer 14's table can go to NVMe uncached with no measurable 
    
[^149]: 比自身存在更久的突发：前缀缓存智能体服务中的持续性退化

    Bursts That Outlast Themselves: Persistent Degradation in Prefix-Cached Agent Serving

    [https://arxiv.org/abs/2611.13412](https://arxiv.org/abs/2611.13412)

    本文发现前缀缓存的 LLM 智能体服务具有亚稳态特性：即使负载恢复，短短几分钟的请求突发也会使服务器陷入缓存命中率骤降至约 10%、吞吐量减半且持续恶化的退化状态，其根源是滞留会话的工作集超出 KV 缓存池，导致每轮交互都变成代价高昂的重新计算。

    

    LLM 智能体以闭环会话的方式驱动推理引擎，其上下文在每一轮交互中不断增长，而经济高效地服务这类负载依赖于前缀缓存：缓存命中只需对约一千个 token 进行增量预填充，而缓存未命中则需要重新计算 10–40K token 的上下文。我们证明了这使前缀缓存的智能体服务呈现亚稳态特性。在相同的基础工作负载下，短短几分钟的额外到达请求就会使 vLLM 服务器进入退化状态——约 10% 的缓存命中率、大约一半的吞吐量，以及在剩余 70 分钟内持续增长的请求积压——而未经历突发的对照组服务器则保持健康（12 对实验中有 9 对出现此现象，反向情况则从未出现）。这种退化状态出现在缓存预热服务器本可承受的负载水平下，在没有前缀缓存时不会出现，并且在 Qwen3-32B 的 SWE-bench 智能体轨迹形态下以及 SGLang 系统上同样会出现（SGLang 仅能容忍更大的触发阈值）。变慢的服务使更多会话滞留在系统中，它们的工作集超出 KV 缓存池的容量，导致每一轮交互都退化为重新计算。

    arXiv:2611.13412v1 Announce Type: cross  Abstract: LLM agents drive inference engines with closed-loop sessions whose context grows every turn, and serving them economically depends on prefix caching: a hit costs an incremental prefill of about a thousand tokens, a miss recomputes a 10--40K-token context. We show that this makes prefix-cached agent serving metastable. With an identical base workload, a few minutes of extra arrivals leaves a vLLM server in a degraded regime--a ~10% hit ratio, about half the throughput, and a backlog that grows for the remaining 70 minutes--while its no-burst twin stays healthy (9 of 12 pairs; none in the reverse direction). The regime appears at loads the cache-warm server sustains, is absent without prefix caching, and also appears with Qwen3-32B on SWE-bench agent trace shapes and on SGLang, which tolerates larger triggers. Slower service keeps more sessions in the system, their working set outgrows the KV pool, and every turn becomes a recomputation;
    
[^150]: SteerDistill：通过闭环教师引导实现LLM智能体的量化

    SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering

    [https://arxiv.org/abs/2611.13410](https://arxiv.org/abs/2611.13410)

    提出了SteerDistill，一种量化感知的蒸馏方法，通过全精度教师模型以闭环方式引导学生修订失败响应，并利用学生执行反馈迭代优化引导，从而在低比特量化下保持LLM智能体的任务成功率。

    

    低比特量化可以降低LLM智能体的推理成本，但可能损害任务成功率。通过分析失败的执行轨迹，我们发现量化后的智能体往往会忽略任务要求或环境条件，导致环境状态转换偏离成功的全精度参考轨迹。纠正这些行为偏差仍然很困难：离线数据可能无法覆盖学生模型的失败场景，而其自身的执行轨迹在这些场景中又可能缺少正确的响应样本。我们提出了SteerDistill，一种量化感知的蒸馏方法，其中全精度教师模型引导学生修订失败的响应，并根据学生模型执行的反馈迭代地优化其引导。这种闭环教师引导产生了经过本地验证的行为修正。我们通过在无引导条件下训练学生模型，使其在修正轮次上匹配被引导后学生模型的预测以及全精度模型的行为，从而将这些修正内化。

    arXiv:2611.13410v1 Announce Type: new  Abstract: Low-bit quantization reduces the inference cost of LLM agents but can compromise task success. Examining failed rollouts, we find that quantized agents overlook task requirements or environmental conditions, producing environment transitions that diverge from successful full-precision references. Correcting these behavioral divergences remains difficult: offline data may miss the student's failure contexts, while its own rollouts may provide few correct responses in those contexts. We introduce SteerDistill, a quantization-aware distillation method in which a full-precision teacher guides the student to revise failed responses and iteratively refines its guidance using feedback from student execution. This closed-loop teacher steering produces locally verified behavioral corrections. We internalize these corrections by training the student without guidance to match the steered student's predictions at corrected turns and the full-precisi
    
[^151]: 剂量决定毒性：面向暴露量的生成式药物设计评估

    The Dose Makes the Poison: Exposure-Aware Evaluation of Generative Drug Design

    [https://arxiv.org/abs/2611.13398](https://arxiv.org/abs/2611.13398)

    该论文利用基于生理的药代动力学（PBPK）模拟对生成式药物设计的候选分子进行暴露量感知的评估，揭示了传统静态分子性质筛选方法因忽略剂量可行性与安全性而会遗漏大量更优分子的重要缺陷。

    

    生成式模型正越来越多地被用于为药物发现提出候选分子。它们能够产生远超实验测试能力的候选分子，这使得计算筛选变得至关重要。大多数筛选方法依赖于预测的分子性质（如效价和毒性），却忽略了这些性质在给定剂量下如何转化为实际的药物暴露量。因此，在纸面上看似有前景的分子可能需要不切实际的剂量，或引发安全性问题。我们通过使用基于生理的药代动力学（PBPK）模拟，重新评估从三个PARP1靶向生成式流程中筛选出的分子，来研究这一差距。在60个入围分子中，有7个无法达到足够的靶点暴露量，而其余候选分子中有28%在其所需剂量下显示出预测的安全风险。此外，85%具有可实现剂量的候选分子，其表现不如静态筛选过程中被忽视的分子。这些发现……

    arXiv:2611.13398v1 Announce Type: cross  Abstract: Generative models are increasingly used to propose molecules for drug discovery. They can produce far more candidates than can be experimentally tested, making computational selection essential. Most selection methods rely on predicted molecular properties, such as potency and toxicity, but overlook how these properties translate into drug exposure at a given dose. As a result, molecules that appear promising on paper may require impractical doses or raise safety concerns. We investigate this gap by using physiologically based pharmacokinetic (PBPK) simulations to re-evaluate molecules selected from three PARP1-targeted generative pipelines. Of 60 shortlisted molecules, seven fail to achieve sufficient target exposure, while 28% of the remaining candidates show predicted safety risks at their required dose. Moreover, 85% of candidates with an achievable dose are outperformed by molecules overlooked during static selection. These findin
    
[^152]: Plumbline：当损失函数撒谎时，正确验证大语言模型训练

    Plumbline: When the Loss Lies, Validating LLM Training Correctly

    [https://arxiv.org/abs/2611.13393](https://arxiv.org/abs/2611.13393)

    该论文提出了已在亚马逊生产环境落地的 LLM 训练正确性验证系统 Plumbline，它通过一套可跨模型和框架通用的系统性正确性方法论，对训练全生命周期（从训练数据到所服务的模型）进行检查，并能将显性与静默故障自动追踪到根本原因，从而避免漫长训练因故障而浪费。

    

    正确性是大规模训练大语言模型（LLM）的前提条件，然而却难以保证：故障可能出现在训练运行的任何部分，且其表现形式因框架及其依赖的库而异。有些故障是显性的，有些则是静默的，两类故障的检测与追踪都耗时费力。一个单一的故障就可能毁掉一次漫长的训练：训练可能在后期崩溃，或者模型只在最终评估时才表现出得分偏低。我们提出了 Plumbline，一个用于 LLM 训练的正确性验证系统，已在 Amazon 投入生产使用。Plumbline 检查训练生命周期的每个环节——从训练所用的数据到最终服务的模型，既深入到每个模型引入的新层，也覆盖到整个训练运行。其诊断机制复用这些检查将故障追溯到根本原因。二者都建立在一套系统化的正确性方法论之上，该方法论旨在跨模型、跨框架通用。Plumbline 支持两种智能体（agentic）用例：Day-0 接入，即对新训练栈进行验证……（原文摘要在此处截断）

    arXiv:2611.13393v1 Announce Type: cross  Abstract: Correctness is a prerequisite for training large language models (LLMs) at scale, yet it is hard to ensure: a fault can sit in any part of a run, and how it shows differs across frameworks and the libraries they depend on. Some failures are loud and some are silent, and both take time to detect and trace. A single fault can waste a long run: the run crashes late, or the model scores lower only at the final evaluation. We present Plumbline, a correctness validation system for LLM training, used in production at Amazon. Plumbline checks each part of the training lifecycle, from the data a run trains on to the model it serves, down to the new layers each model brings and up to whole runs. Its diagnosis reuses these checks to trace failures to their root causes. Both rest on a systematic correctness methodology designed to carry across models and frameworks. Plumbline supports two agentic use cases: Day-0 onboarding, which validates the tr
    
[^153]: AdjointFlow：基于物理的传输加速科学逆问题求解

    AdjointFlow: Accelerating Scientific Inverse Problems with Physics-Based Transport

    [https://arxiv.org/abs/2611.13389](https://arxiv.org/abs/2611.13389)

    提出AdjointFlow框架，通过随机插值将已知物理知识注入生成式传输过程本身，并利用基于伴随方法的数据相关初始化减少传输距离，从而加速科学逆问题的求解。

    

    逆问题在从地震成像、天体物理到无损检测等科学应用中无处不在，然而经典求解器往往难以应对非凸性问题以及对昂贵物理模型的重复评估。扩散模型虽然可以在离线阶段摊销这些计算，但主要依赖数据驱动的条件模型未能充分利用已知物理知识，并且在目标分布高度集中时会受到病态传输映射的困扰。即插即用（PnP）方法显式地利用物理模型来引导学习到的先验，改善了数据一致性，但又重新引入了代价高昂的迭代计算。我们转而提出AdjointFlow，这是一个建立在随机插值基础上的框架，将已知物理直接注入生成式传输本身。我们设计了一种数据相关的、基于伴随方法的初始化来减少到目标分布的传输距离，并修改传输速度使其在端点分布……（摘要在此处被截断）

    arXiv:2611.13389v1 Announce Type: new  Abstract: Inverse problems are ubiquitous across scientific applications ranging from seismic imaging to astrophysics to nondestructive testing, yet classical solvers often struggle with nonconvexity and repeated evaluations of expensive physical models. Diffusion models can amortize this computation offline, but largely data-driven conditional models underutilize known physics and suffer from ill-conditioned transport maps when the target distribution is concentrated. Plug-and-play (PnP) methods explicitly leverage the physical model to guide a learned prior, improving data consistency but reintroducing costly iterative computations. We instead propose AdjointFlow, a framework built on stochastic interpolants that injects the known physics into the generative transport itself. We design a data-dependent, adjoint-based initialization to reduce transport distance to the target, modify the transport velocity to remain regular as the endpoint distrib
    
[^154]: 解决预热间隔问题：线性回归中尾平均SGD的尖锐风险界

    Settling the Burn-in Gap: Sharp Risk Bounds for Tail-Averaged SGD in Linear Regression

    [https://arxiv.org/abs/2611.13383](https://arxiv.org/abs/2611.13383)

    该论文通过建立依赖初始化信息的匹配上下风险界，解决了尾平均SGD分析中头部特征空间初始误差方差贡献的“预热间隔”问题，并证明现有上界和下界均不够紧致。

    

    我们研究了恒定步长在线随机梯度下降（SGD）结合尾平均方法在高维线性回归中的超额风险。基于Zou等人（2021）提出的实例相关谱框架，当轨迹谱反馈条件成立时，我们建立了尾平均SGD总超额风险的匹配的、初始化显式的上下界，这些界由总体协方差谱 $\mathbf{H}$ 和初始化误差 $\mathbf{w}_0-\mathbf{w}^*$ 刻画。现有分析在有效方差方面存在一个根本性的差距：Zou等人（2021）的上界在头部特征空间 $\mathbf{I}_{0:k^\dagger}$ 中保留了来自初始误差的未衰减贡献，而他们的下界却完全忽略了头部初始化所诱导的方差。我们证明现有的上界和下界都是宽松的：来自头部初始化的单跳快模式方差贡献……

    arXiv:2611.13383v1 Announce Type: new  Abstract: We study the excess risk of constant-stepsize online stochastic gradient descent (SGD) with tail-averaging for high-dimensional linear regression. Building on the instance-dependent spectral framework of Zou et al. (2021), we establish matching initialization-explicit upper and lower bounds on the total excess risk of tail-averaged SGD in terms of the population covariance spectrum $\mathbf{H}$ and initialization $\mathbf{w}_0-\mathbf{w}^*$ whenever a trajectory spectral feedback condition holds. Existing analyses exhibit a fundamental gap in the effective variance: the upper bound of Zou et al. (2021) retains an undamped contribution from the initial error in the head eigenspace $\mathbf{I}_{0:k^\dagger}$, whereas their lower bound omits the variance induced by the head initialization altogether. We show that both the existing upper and lower bounds are loose: the single-hop fast-mode variance contribution from the head initialization i
    
[^155]: 面向多机器人任务分配中组合优化的离散基于模型扩散

    Discrete Model-Based Diffusion for Combinatorial Optimization in Multi-Robot Task Allocation

    [https://arxiv.org/abs/2611.13382](https://arxiv.org/abs/2611.13382)

    提出了一种无需训练的离散基于模型扩散方法，通过蒙特卡洛采样近似类别型反向转移分布，直接在离散空间进行基于采样的优化，从而求解多机器人任务分配中的NP难组合优化问题，并给出了找到全局最优解概率的有限步下界理论保证。

    

    多智能体与多机器人系统中的决策通常涉及离散决策变量，由此产生必须在数秒内求解以满足近实时应用的NP难组合优化问题。去噪扩散已成为解决此类问题的强大工具，但现有方法面临权衡：基于模型的扩散无需训练即可将扩散融入基于采样的优化中，但仅在连续空间上有效；而现有的基于扩散的组合优化求解器虽然工作在离散空间上，却需要离线训练。为此，我们提出离散基于模型扩散，这是一种无需训练的基于采样的优化方法，通过蒙特卡洛采样近似类别型反向转移分布，从而直接优化离散变量。我们建立了找到全局最小值概率的有限步下界，并证明该下界收敛于一（摘要至此被截断）。

    arXiv:2611.13382v1 Announce Type: cross  Abstract: Decision making in multi-agent and multi-robot systems often involves discrete decision variables, leading to NP-hard combinatorial optimization problems that must be solved within seconds for near real-time implementation. Denoising diffusion has emerged as a powerful tool for such problems, but existing approaches face a trade-off: Model-Based Diffusion (MBD) incorporates diffusion in sampling-based optimization (SBO) without training but only over continuous spaces, while existing diffusion-based combinatorial solvers operate over discrete spaces but require offline training. To this end, we propose Discrete Model-Based Diffusion (Discrete MBD), a training-free SBO method that directly optimizes discrete variables by approximating categorical reverse transition distributions through Monte Carlo sampling. We establish a finite-step lower bound on the probability of finding a global minimizer and show that this bound converges to one 
    
[^156]: 无需增大KV缓存的更深循环：面向因果编码器-解码器的循环深度

    Looping Deeper Without Growing the KV Cache: Recurrent Depth for Causal Encoder-Decoders

    [https://arxiv.org/abs/2611.13378](https://arxiv.org/abs/2611.13378)

    该论文提出一种循环因果编码器-解码器架构，通过将循环置于编码器以迭代精炼全局KV表示，并利用随机循环深度训练诱导不动点平衡实现跨层与跨轮KV缓存无损共享，从而在不增大KV缓存的前提下获得循环深度带来的性能提升。

    

    循环语言模型能够在不增加参数的情况下加深计算，然而在层与轮次之间缓存全上下文的键值（KV）状态严重限制了推理服务的批次大小。我们设计了一种循环因果编码器-解码器（CED），实现了零轮次KV增长的循环深度，并解决了在非对称架构堆栈中何处以及如何进行循环的问题——该堆栈中仅有稀疏的编码器层生成全局KV。我们发现，循环必须放在编码器中，以迭代式地精炼全局KV表示，而不是重新读取静态的解码器投影，这既能带来语言建模性能的提升，又能保持长上下文的扩展能力。此外，在随机循环深度下训练编码器循环会诱导出一个不动点平衡状态，使得每一轮都能无损地共享上一轮的KV缓存。通过叠加跨层与跨轮的KV共享，最终的Looped CED达到了未循环CED的最小内存占用，并恢复了86-93%的性能……（摘要原文在此处截断）

    arXiv:2611.13378v1 Announce Type: new  Abstract: Looped language models deepen computation without adding parameters, yet caching full-context key-value (KV) states across layers and rounds severely bounds serving batch size. We design a looped causal encoder-decoder (CED) that achieves recurrent depth with zero round-wise KV growth, addressing where and how to loop an asymmetric stack in which only sparse encoder layers generate global KV. We find that recurrence must be placed in the encoder to iteratively refine global KV representations rather than re-reading static decoder projections, which both drives language-modeling gains and preserves long-context scaling. Furthermore, training the encoder loop under stochastic recurrence depth induces a fixed-point equilibrium where every round losslessly shares the last round's KV cache. Compounding cross-layer and cross-round KV sharing, the resulting Looped CED matches an unlooped CED's minimal memory footprint and recovers 86-93% of a l
    
[^157]: 在基于损失的规则下为AI对齐强制执行公理

    Enforcing Axioms for AI Alignment under Loss-Based Rules

    [https://arxiv.org/abs/2611.13369](https://arxiv.org/abs/2611.13369)

    本文将AI对齐原则建模为嵌入空间中的线性“投票者”，在基于损失的奖励模型训练框架下研究帕累托最优等公理能否成立，发现即使采用更丰富的奖励模型或修改比较数据，帕累托最优性仍可能被违反。

    

    许多对齐方法，尤其是基于人类反馈的强化学习（RLHF），通过在模型响应的成对比较上最小化损失来训练奖励模型。受宪法AI等原则引导方法的启发，我们将每条原则视为一个“投票者”，它依据该原则比较响应，例如偏好更有帮助或毒性更低的响应。我们将这些投票者表示为固定嵌入空间中的线性方向，这一简化假设源于线性表示假说。在这种线性社会选择模型中，Ge等人（2024）证明了最优线性奖励模型可能违反帕累托最优性（PO）：从“原则即投票者”的视角看，这意味着响应A可能比B帮助更少且毒性更高，却仍获得更高的奖励。我们探究更丰富的奖励模型或对比较数据的修改能否在基于损失的训练中恢复帕累托最优性。我们证明PO仍可能失效。

    arXiv:2611.13369v1 Announce Type: cross  Abstract: Many alignment methods, most notably reinforcement learning from human feedback (RLHF), train reward models by minimizing a loss on pairwise comparisons of model responses. Inspired by principle-guided methods such as Constitutional AI, we treat each principle as a voter that compares responses according to that principle, for example, preferring the more helpful or less toxic response. We represent these voters as linear directions in a fixed embedding space, a simplifying assumption motivated by the Linear Representation Hypothesis. In this linear social choice model, Ge et al. (2024) showed that an optimal linear reward model can violate Pareto optimality (PO): From the principles-as-voters lens, this means a response A can be less helpful and more toxic than B, yet still receive a higher reward. We ask whether richer reward models or changes to comparison data can recover PO within loss-based training. We show that PO can still fai
    
[^158]: 使用机器学习重建大气边界层风廓线：基于IGRA探空数据的比较研究

    Reconstructing Boundary Layer Wind Profiles with Machine Learning: A Comparative Study on IGRA Radiosonde Data

    [https://arxiv.org/abs/2611.13359](https://arxiv.org/abs/2611.13359)

    该研究基于158,097个IGRA探空数据比较了CatBoost、TabM和FT-Transformer等现代机器学习方法与传统解析参数化方法在边界层风廓线重建中的表现，发现机器学习将RMSE从约3 m/s大幅降至1.3-1.6 m/s，并揭示了预测精度主要取决于预测变量而非模型架构，同时给出了逐点预测器约1.77 m/s的经验精度下限。

    

    大气边界层（ABL）中的风廓线对短期天气预报和风能选址至关重要，然而稀疏的探空观测覆盖迫使从业者仅从少数物理预测变量出发对其进行估算。本研究在158,097个IGRA探空样本上，将现代表格机器学习方法与分析参数化方法进行了基准比较。CatBoost、TabM和FT-Transformer将RMSE降低至1.3-1.6 m/s，而传统的幂律和Monin-Obukhov拟合的RMSE约为3 m/s；一旦预测变量集固定，三种模型架构之间的差异小于0.05 m/s，说明预测精度主要由预测变量决定而非模型架构。对组内廓线离散度的队列分析表明，对于使用标准探空特征的任何逐点预测器而言，存在约1.77 m/s的经验精度下限，这与先前基于涡粘性反演的研究结果一致。此外，将一维傅里叶神经算子作为事后校正器应用于冻结的表格模型预测，可降低近地面的RMSE。

    arXiv:2611.13359v1 Announce Type: new  Abstract: Wind profiles in the atmospheric boundary layer (ABL) drive short-range forecasting and wind-energy siting, yet sparse radiosonde coverage forces practitioners to interpolate them from a handful of physical predictors. We benchmark modern tabular learners against analytical parameterizations on 158,097 IGRA soundings. CatBoost, TabM, and FT-Transformer reduce RMSE to 1.3-1.6 m/s, against approximately 3 m/s for power-law and Monin-Obukhov fits; once the predictor set is fixed the three architectures differ by less than 0.05 m/s, so accuracy is driven by predictors rather than architecture. A cohort analysis of within-bin profile dispersion places an empirical floor of approximately 1.77 m/s on any point-wise predictor that uses standard radiosonde features, consistent with prior eddy-viscosity inversions. A one-dimensional Fourier Neural Operator applied as a post-hoc corrector to frozen tabular predictions reduces near-surface RMSE on U
    
[^159]: 何时仅凭预测就够了？一种冻结专家组合的评估协议

    When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition

    [https://arxiv.org/abs/2611.13356](https://arxiv.org/abs/2611.13356)

    该论文提出了一个可复用的评估协议来公平比较冻结专家模型的特征组合与预测组合，并发现对六个架构多样的深度伪造检测专家特征进行逻辑回归，在留一生成器评估中比最佳预测组合器的宏平均 AUROC 提升 0.081。

    

    实践者常常在保持模型权重冻结的情况下组合独立训练的模型。这些专家模型通常通过其最终预测结果进行组合，然而它们的内部特征可能保留了那些预测结果所忽略的有用信息。我们探讨了何时组合特征会优于组合预测，以及如何使这种比较公平。我们提出了一个可复用的评估协议，将其应用于来自 DF40（Yan et al., 2024）的 21 个冻结深度伪造检测器，并审查了评估选择如何改变结论。该协议采用强大的预测基线，比较简单与更大规模的特征组合器，分别独立地变化架构与训练数据，并通过留出生成器以及训练/测试重叠控制来报告结果。在留一生成器评估下，对六个架构多样的专家模型特征进行逻辑回归，在每一折中都比测试过的最佳预测组合器将宏平均 AUROC 提升了 0.081。

    arXiv:2611.13356v1 Announce Type: cross  Abstract: Practitioners often combine independently trained models while keeping their weights frozen. These experts are commonly combined through their final predictions, although their internal features may retain useful information that those predictions omit. We ask when combining features improves on combining predictions, and how to make that comparison fair. We introduce a reusable evaluation protocol, apply it to 21 frozen deepfake detectors from DF40 (Yan et al., 2024), and audit how evaluation choices change the conclusions. The protocol uses strong prediction baselines, compares simple and larger feature combiners, varies architecture and training data separately, and reports results by held-out generator with train/test overlap controls. Under leave-one-generator-out evaluation, logistic regression on the features of six architecture-diverse experts improves macro AUROC by 0.081 over the best tested prediction combiner in each fold, 
    
[^160]: DEPUTYBENCH：评估企业智能体的情境安全性

    DEPUTYBENCH: Evaluating Contextual Safety in Enterprise Agents

    [https://arxiv.org/abs/2611.13355](https://arxiv.org/abs/2611.13355)

    该论文提出了可执行基准DEPUTYBENCH来评估企业智能体在组织权限与证据约束下的情境安全性，并揭示了一个关键风险：模型在证据缺失时比在明确禁止时更容易越权执行受限操作（10.3%对3.5%）。

    

    企业智能体必须在继续开展合法工作的同时，判断普通请求是否被组织记录与政策所允许。我们提出了DEPUTYBENCH，这是一个可执行的基准测试，包含79个基础条目、375个实例、成对的权限上下文以及分级的证据要求。在12个模型上，修订后的审计得到了88.0%的总体安全性和79.1%的成对成功率。一项针对28个衍生任务和6个模型的受控研究测试了授权事实翻转、证据遮蔽和不相关改写。实验结果总体上遵循后端权限，然而与明确的不许可相比，证据缺失时受限执行更为频繁：在27个匹配任务上分别为10.3%对3.5%，相差6.8个百分点（95%区间：1.9-13.0；次要分析）。事后分析发现，100次遮蔽条件下的执行中有61次发生在四个需要完整累计计数或特定个人同意的隐私任务中。当错误最……（摘要原文在此处截断）

    arXiv:2611.13355v1 Announce Type: cross  Abstract: Enterprise agents must determine whether ordinary requests are permitted by organizational records and policies while continuing legitimate work. We introduce DEPUTYBENCH, an executable benchmark with 79 base items, 375 instances, paired permission contexts, and graded evidence requirements. Across 12 models, revised audits give 88.0% pooled safety and 79.1% paired success. A controlled study of 28 derived tasks and six models tests authorization-fact flips, evidence masking, and irrelevant rewrites. Outcomes track backend permissions, yet restricted execution is more frequent with missing evidence than explicit non-permission: 10.3% versus 3.5% on 27 matched tasks, a 6.8-percentage-point difference (95% interval: 1.9-13.0; secondary analysis). A post-hoc analysis locates 61 of 100 masked-condition executions in four privacy tasks requiring complete cumulative counts or person-specific consent. The difference persists when the most err
    
[^161]: 推进用于分布式水文建模的质量守恒深度学习：以一个积雪主导流域为例

    Advancing Mass-Conserving Deep Learning for Distributed Hydrologic Modeling: A Case Study in a Snow-Dominated Watershed

    [https://arxiv.org/abs/2611.13346](https://arxiv.org/abs/2611.13346)

    本文提出了一个质量守恒的卷积LSTM框架PECAN，通过严格（PECAN-MC）和放宽（PECAN-MCR）两种守恒约束变体实现分布式水文建模，并用于诊断物理感知深度学习模型的结构性局限。

    

    物理感知深度学习（DL）的最新进展已将质量守恒嵌入到用于水文建模的长短期记忆网络（LSTM）架构中，提升了物理一致性，并支持对内部状态的诊断。然而，现有的质量守恒神经网络主要以集总式（lumped）形式开发。因此，以往的研究仅提供了零散且往往间接的证据，说明施加的物理约束在何时以及为何可能导致不合理的水文行为。为解决这一局限性，我们提出了物理增强守恒增广网络（PECAN），这是一个基于质量守恒卷积LSTM的框架，可用于分布式水文建模以及诊断物理感知深度学习模型的结构性局限。PECAN以两种变体实现：(1) PECAN-MC，具有严格的基于降水的守恒约束；(2) PECAN-MCR，具有放宽的守恒约束，允许额外的水分输入。

    arXiv:2611.13346v1 Announce Type: new  Abstract: Recent progress in physics-aware deep learning (DL) has embedded mass conservation within Long Short-Term Memory (LSTM) architecture for hydrologic modeling, improving physical consistency and enabling diagnostics of internal states. However, existing mass-conserving neural networks are primarily developed in lumped formulations. Consequently, previous studies provide only fragmented and often indirect evidence of when and why imposed physical constraints may lead to unrealistic hydrologic behavior. To address this limitation, we propose Physics-Enhanced Conservation-Augmented Net (PECAN), a mass-conserving convolutional LSTM-based framework for distributed hydrologic modeling and diagnosing structural limitations of physics-aware DL models. PECAN is implemented in two variants: (1) PECAN-MC with a strict precipitation-based conservation constraint and (2) PECAN-MCR with a relaxed conservation constraint allowing additional water inputs 
    
[^162]: 几何机器学习中的VC维与泛化及其在主动学习中的应用

    VC-Dimension and Generalization in Geometric Machine Learning with Applications to Active Learning

    [https://arxiv.org/abs/2611.13337](https://arxiv.org/abs/2611.13337)

    该论文计算了黎曼几何模型空间中度量球分类器的VC维并给出了多种情形下的上下界，进而在数据分布假设下利用覆盖论证证明了泛化界，并将其应用于主动学习中的基于覆盖的算法。

    

    我们计算了黎曼几何模型空间（欧几里得空间、球面、双曲空间）中度量球分类器的VC维，并在多种其他情形下给出了度量球VC维的上界与下界，包括一般的赋范向量空间（VC维可能为无穷大）、具有 $\ell^p$ 范数的向量空间（VC维至少随维度线性增长，当 $p=1$ 时增长速度快于欧几里得情形）、平坦环面，以及一般的黎曼流形（在共形变换后VC维可以任意大）。在VC维很大甚至为无穷的若干情形下，我们利用覆盖论证，在附加的数据分布假设下证明了泛化界。无论是在VC维还是覆盖的设定中，我们都考虑了位于高维欧几里得空间低维子集上的数据。作为应用，我们研究了主动学习中基于覆盖的算法。

    arXiv:2611.13337v1 Announce Type: cross  Abstract: We compute the VC-dimension of metric ball classifiers in the model spaces of Riemannian geometry (Euclidean, spherical, hyperbolic) and present upper and lower bounds on the VC-dimension of metric balls in several other situations, including general normed vector spaces (potentially infinite VC-dimension), vector spaces with $\ell^p$-norms (VC-dimension grows at least linearly in the dimension, faster than Euclidean for $p=1$), flat tori, and general Riemannian manifolds (arbitrarily large VC-dimension after a conformal change). In several situations where the VC-dimension is large or even infinite, we prove generalization bounds under additional assumptions on the data distribution using covering arguments. Both in the VC-dimension and covering setting, we consider data on low-dimensional subsets of high-dimensional Euclidean spaces. As an application, we consider coverage-based algorithms in active learning.
    
[^163]: 面向联邦LoRA的随机投影与秩感知学习率

    Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

    [https://arxiv.org/abs/2611.13331](https://arxiv.org/abs/2611.13331)

    该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。

    

    在资源受限的分布式客户端上对大规模基础模型进行微调，面临着显著的计算与通信挑战。虽然将低秩适应与联邦学习相结合可以实现基于秩的灵活性，但适应矩阵（A 与 B）之间的相互依赖往往使优化变得复杂。在这项工作中，我们从理论上证明：采用高斯初始化并冻结 A 矩阵的 LoRA 微调，相当于对全量微调的一种随机近似，其中批量梯度会受到随机且与秩相关的扰动。受这一见解的启发，我们提出在客户端层面冻结并周期性地重新采样 A 矩阵。该策略解耦了 A 和 B 的梯度，削弱了 LoRA 的低秩约束，并将计算与上行通信开销降低了 50%，从而允许重新分配……（原文在此处截断）

    arXiv:2611.13331v1 Announce Type: cross  Abstract: Fine-tuning large-scale foundation models across resource-constrained, distributed clients presents significant computational and communication challenges. While merging Low-Rank Adaptation (LoRA) with Federated Learning (FL) enables rank-based flexibility, the interdependence of the adaptation matrices ($\mathbf{A}$ and $\mathbf{B}$) often complicates optimization. In this work, we theoretically demonstrate that LoRA fine-tuning with Gaussian initialization and a frozen $\mathbf{A}$ matrix acts as a stochastic approximation of full fine-tuning, where batch gradients are subject to a random, rank-dependent perturbation. Motivated by this insight, we propose freezing and periodically resampling $\mathbf{A}$ matrices at the client level. This strategy decouples the $\mathbf{A}$ and $\mathbf{B}$ gradients, diminishes LoRA low-rank constraints and reduces both computational and uplink communication overheads by 50\%, allowing for the reall
    
[^164]: 目标污染与标签噪声下的鲁棒流匹配

    Robust Flow Matching under Target Corruption and Label Noise

    [https://arxiv.org/abs/2611.13324](https://arxiv.org/abs/2611.13324)

    该论文将流匹配训练建模为条件向量场上的时变回归问题，系统分析了目标污染与标签噪声对训练目标的影响，并提出时间残差分数（TRS）与比较条件分数（CCS）两种残差驱动的样本加权策略，显著提升了流匹配在数据污染场景下的鲁棒性。

    

    流匹配因其稳定的训练和高效的采样而成为一种强大的生成建模框架。然而，流匹配假设目标数据和标签是干净的，这一假设在实践中经常被违反。我们将流匹配训练表述为条件向量场上的时变回归问题，并分析了两种污染类型如何影响目标函数：i) 目标污染会引入一个随时间增长的时变残差项；ii) 标签噪声会导致模型向错误的类条件向量场进行更新。基于这一分析，我们提出通过将特定于污染类型的残差分数映射为样本权重来提升流匹配的鲁棒性。针对目标污染，我们提出了时间残差分数，它强调流匹配中后期时间步的残差幅度，因为在这些阶段干净样本与受污染样本更容易区分。针对标签噪声，我们提出了比较条件分数，它通过比较观测到的残差……（原文摘要此处被截断）

    arXiv:2611.13324v1 Announce Type: new  Abstract: Flow Matching (FM) is a strong framework for generative modeling due to its stable training and efficient sampling. However, FM assumes clean target data and labels, an assumption often violated in practice. We formulate FM training as a time-dependent regression problem over conditional vector fields and analyze how the two corruption types affect the objective: i) target corruption adds a time-dependent residual term that grows with time, ii) label noise causes updates to the wrong class-conditional vector field. Based on this analysis, we propose to improve FM robustness by mapping corruption-specific residual scores to sample weights. For target corruption, we propose the Temporal Residual Score (TRS), which emphasizes residual magnitudes at intermediate-to-late FM times, where clean and corrupted samples are more separable. For label noise, we propose the Comparative Conditional Score (CCS), which compares the residual of the observ
    
[^165]: MicroLens：用两种“透镜”读取、追踪与重连残差流

    MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses

    [https://arxiv.org/abs/2611.13322](https://arxiv.org/abs/2611.13322)

    本文提出交互式工具 MicroLens，将 logit 透镜与雅可比透镜（J-lens）统一到同一张残差流“层×词元”网格上，实现对隐藏状态的读取、来源追踪与在线编辑，并发现 logit 透镜方向适合控制词元输出、J-lens 方向适合改变后续词元的读取。

    

    将隐藏状态解码为词语的“透镜”是观察语言模型内部机制的常用方法，但解码词语的热力图只能显示一个向量“读作什么”，无法区分模型是否使用、或在多大程度上使用了该信息。我们提出了一个交互式查看器，将读取、追踪与编辑功能统一到残差流的同一张“层×词元”网格上。在该查看器中，每个单元格都可以用两种透镜解码——logit 透镜和雅可比透镜（J-lens），可以回溯其信息来源词元，并进行编辑，同时模型的其余部分会被当场重新运行。我们在 Gemma-3-4B-it 上演示了三种使用场景，并以数百条提示的批量扫描实验作为支撑，展示该工具能够发现什么。首先，两种透镜提供了不同的“写入”方向，各自适合不同的编辑：logit 透镜的词语方向决定某个词元输出什么，而 J-lens 的方向则改变后续词元读取什么。其次，在两跳知识问答（two-hop trivia）中……

    arXiv:2611.13322v1 Announce Type: cross  Abstract: Lenses that decode hidden states into words are a popular way to look inside language models, but a heatmap of decoded words only shows what a vector reads as and cannot distinguish whether, or how much, the model uses the information. We present an interactive viewer that puts reading, tracing and editing on the same layers-by-tokens grid of the residual stream. In the viewer, every cell can be decoded with two lenses, the logit lens and the Jacobian lens (J-lens), traced back to the tokens it drew from, and edited, with the rest of the model rerun on the spot. We demonstrate three usage scenarios on Gemma-3-4B-it, backed by sweeps over a few hundred prompts, to show what this tool can discover. First, the two lenses give different directions to write with, each suiting a different edit: the logit lens's word direction sets what a token outputs, while the J-lens's direction changes what later tokens read. Second, in two-hop trivia que
    
[^166]: 一条轨迹，两种需求：流匹配策略欧拉采样中的精度与局部稳定性

    One Trajectory, Two Requirements: Accuracy and Local Stability in Euler Sampling for Flow Matching Policies

    [https://arxiv.org/abs/2611.13311](https://arxiv.org/abs/2611.13311)

    该论文通过精确双坐标系实验证明，流匹配策略的欧拉采样中轨迹精度与局部稳定性对步数的要求可能不一致，并在冻结的 π0.5-LIBERO 策略中发现动作端点附近存在强收缩方向。

    

    流匹配机器人策略通常使用由离线精度扫描选定的固定欧拉步数。然而在推理时，每个策略输入（视觉观测、机器人状态和语言指令）连同一次源噪声采样，共同定义了一条不同的常微分方程（ODE）轨迹以及局部雅可比矩阵序列。我们由此提出问题：能够精确跟随名义上“噪声到动作”轨迹的步数，对于其周围强烈收缩的方向而言是否仍然过于粗糙。一个精确的双坐标系系统以及对六个已训练流模型的受控扩展实验表明，即使原始学习到的输运过程保持不变，轨迹精度与局部欧拉稳定性也可能施加不同的步数要求。随后，我们在不修改权重或动作动力学的前提下，研究了一个冻结的 π0.5-LIBERO 检查点。在来自 10 个 LIBERO-10 任务的 30 个策略输入上，雅可比矩阵在动作端点附近表现出强烈的收缩方向，然而……（摘要原文在此处被截断）

    arXiv:2611.13311v1 Announce Type: new  Abstract: Flow matching robot policies commonly use a fixed Euler step count selected from offline accuracy sweeps. At inference time, however, each policy input (visual observation, robot state, and language instruction) together with a source-noise draw defines a different ODE trajectory and sequence of local Jacobians. We ask whether a step count that accurately follows the nominal noise-to-action trajectory can still be too coarse for strongly contracting directions around it. An exact two-coordinate system and a controlled augmentation of six trained flow models show that trajectory accuracy and local Euler stability can impose different step requirements even when the original learned transport is unchanged. We then study a frozen $\pi_{0.5}$-LIBERO checkpoint without modifying its weights or action dynamics. Across 30 policy inputs from 10 LIBERO-10 tasks, the Jacobian exhibits strong contracting directions near the action endpoint, yet no 
    
[^167]: 训练高秩，传输低耗：面向分布式在线策略后训练的高效 LoRA 同步

    Train High, Ship Low: Efficient LoRA Synchronization for Distributed On-Policy Post-Training

    [https://arxiv.org/abs/2611.13306](https://arxiv.org/abs/2611.13306)

    提出与压缩器无关的 HiLo 框架，用压缩梯度分发加本地优化器重放取代 LoRA 参数传输，并配套预算感知的 Spar 梯度压缩器，将训练秩与通信成本解耦，从而在带宽受限网络上高效实现分布式在线策略后训练。

    

    在带宽受限的网络上，将 LoRA 适配器传输给 rollout 工作节点可能成为分布式在线策略后训练的瓶颈。降低训练秩可能会限制工作负载所需的模型容量，而自适应优化器的更新又难以压缩。相反，rollout 工作节点可以通过维护复制的优化器状态，从梯度中重建更新。我们提出了 HiLo，一个与压缩器无关的框架，它以压缩梯度分发和本地优化器重放取代参数传输。通过应用相同的解码后梯度，学习器与工作节点在可复现执行下能在同一版本上保持一致的策略状态。我们进一步开发了 Spar，一种预算感知的梯度压缩器，它结合谱截断、自适应秩分配和误差反馈，将训练秩与通信成本解耦。在在线策略蒸馏和奖励引导后训练上的实验表明，HiLo……

    arXiv:2611.13306v1 Announce Type: cross  Abstract: Transmitting LoRA adapters to rollout workers can bottleneck distributed on-policy post-training over bandwidth-constrained networks. Lowering the training rank may limit the capacity a workload needs, while adaptive optimizer updates are difficult to compress. Rollout workers can instead reconstruct updates from gradients by maintaining replicated optimizer states. We introduce HiLo, a compressor-agnostic framework that replaces parameter transmission with compressed gradient dissemination and local optimizer replay. By applying the same decoded gradients, the learner and workers maintain consistent policy states at the same version under reproducible execution. We further develop Spar, a budget-aware gradient compressor that combines spectral truncation, adaptive rank allocation and error feedback to decouple training rank from communication cost. Experiments across on-policy distillation and reward-guided post-training show that HiL
    
[^168]: DUO：基于扩散模型双读出的分布外检测方法

    DUO: Dual Readouts of Diffusion Models for OOD Detection

    [https://arxiv.org/abs/2611.13304](https://arxiv.org/abs/2611.13304)

    提出了一种完全无监督、无需训练的OOD检测方法DUO，通过将重构误差与中间去噪特征的主导敏感度谱相结合作为双重信号，即使重构误差产生误导时也能可靠地检测分布外输入。

    

    扩散模型是一种强大的生成方法，可用于从复杂数据分布中进行学习和采样。安全地部署这些模型需要检测超出其训练分布的输入。由于扩散模型从噪声输入中恢复干净样本，重构误差常被用于分布外（OOD）检测。这种方法假设分布内（ID）输入的重构精度高于OOD输入。然而，OOD输入可能被重构得同样好甚至更好，使得这一评分变得不可靠。因此，我们提出将中间去噪特征的主导敏感度谱作为补充性的OOD信号，并证明当重构误差产生误导时，该信号仍能提供有效信息。我们提出了用于OOD检测的扩散模型双读出方法（DUO），这是一种完全无监督、无需训练的OOD检测器，它将重构误差与特征敏感度相结合。

    arXiv:2611.13304v1 Announce Type: cross  Abstract: Diffusion models are powerful generative methods for learning and sampling from complex data distributions. Safely deploying these models requires detecting inputs that fall outside their training distribution. Because diffusion models recover clean samples from noisy inputs, reconstruction error is often used for out-of-distribution (OOD) detection. This approach assumes that in-distribution (ID) inputs are reconstructed more accurately than OOD inputs. However, OOD inputs may be reconstructed equally well or even better, making this score unreliable. We therefore propose the leading sensitivity spectrum of intermediate denoising features as a complementary OOD signal and demonstrate that it remains informative when reconstruction error is misleading. We introduce DUal-readout of diffusion models for OOD detection (DUO), a fully unsupervised, training-free OOD detector that combines reconstruction error and feature sensitivity. DUO im
    
[^169]: 用于MIMO系统高效多径参数估计的神经网络辅助CLEAN方法

    Neural Network-Assisted CLEAN for Efficient Multipath Parameter Estimation in MIMO Systems

    [https://arxiv.org/abs/2611.13298](https://arxiv.org/abs/2611.13298)

    本文提出神经网络辅助CLEAN（NN-CLEAN），用多头分类器替代传统多维搜索，在保留LS增益估计与残差消除的同时，实现了MIMO系统中更高效的多径参数估计。

    

    宽带多输入多输出（MIMO）信道中的高分辨率多径估计需要对到达角、离开角和传播距离进行反复搜索。本文提出神经网络辅助CLEAN（NN-CLEAN）方法，用多头分类器替代每个提取步骤中的多维搜索，同时保留复路径增益的最小二乘（LS）估计、参数化重构和残差消除。同一分类器被应用于逐次提取后的残差，使得提取深度可以变化而无需改变其输出结构。在迭代次数由参考多径分量（MPC）数目规定的情况下，离格合成测试表明，对于一径和两径MPC，NN-CLEAN在路径精度上略优于网格搜索CLEAN（GS-CLEAN），尽管其精度在更大路径数时下降更快。对于批大小为ei的三MPC完整提取（原文此处被截断）

    arXiv:2611.13298v1 Announce Type: new  Abstract: High-resolution multipath estimation in wideband multiple-input multiple-output (MIMO) channels requires repeated searches over angle of arrival, angle of departure, and propagation distance. This paper presents neural network-assisted CLEAN (NN-CLEAN), which replaces the multidimensional search at each extraction step with a multi-head classifier while retaining least-squares (LS) estimation of the complex path gain, parametric reconstruction, and residual cancellation. The same classifier is applied to successive residuals, allowing the extraction depth to vary without changing its output structure. With the number of iterations prescribed by the reference multipath-component (MPC) count, off-grid synthetic tests show that NN-CLEAN slightly exceeds grid-search CLEAN (GS-CLEAN) in path accuracy for one and two MPCs, although its accuracy declines more rapidly at larger path counts. For complete three-MPC extraction at a batch size of ei
    
[^170]: 基于高斯过程的时间序列变点检测

    Time Series Change Point Detection with Gaussian Processes

    [https://arxiv.org/abs/2611.13295](https://arxiv.org/abs/2611.13295)

    该论文提出了一种基于高斯过程的通用变点检测框架，将变化定义为高斯过程先验超参数的改变，具有理论性能保证，且无需预先指定变化前后的时间序列函数形式，超参数可从数据中自动学习。

    

    本文提出了一种新颖的通用高斯过程框架，用于时间序列中的变点检测。在该框架中，变化被定义为高斯过程先验超参数的改变。我们将该问题置于统计假设检验的背景下进行建模，并展示了该方法在离线和在线多变点检测任务中的性能。该方法为其性能提供了理论保证。所开发框架的一个重要优势在于，它无需在变化发生前对时间序列函数进行详细设定，超参数的值可以从数据中学习得到。本文在合成数据和真实数据上对该框架进行了全面评估，并验证了其有效性。

    arXiv:2611.13295v1 Announce Type: cross  Abstract: This paper proposes a novel general Gaussian process framework for change point detection in time series. A change is defined as an alteration of the hyperparameters of the Gaussian process prior. We formulate the problem in the context of statistical hypothesis testing and demonstrate its performance over both offline and online multiple change points detection. The approach provides theoretical guaranties of its performance. An important advantage of the developed framework is that it does not need a detailed specification of the time series function before and after a change. The hyperparameter values are learned from the data. Comprehensive evaluation of the proposed framework is provided both on synthetic and real data and its efficiency is demonstrated.
    
[^171]: 解码令牌并行：面向可扩展大语言模型解码的内存高效分布式注意力并行

    Decode Token Parallelism: Memory-Efficient Distributed Attention Parallelism for Scalable LLM Decoding

    [https://arxiv.org/abs/2611.13293](https://arxiv.org/abs/2611.13293)

    提出解码令牌并行（TKNP）架构，通过将权重放置与注意力所有权解耦，解决了现有并行策略中KV缓存容量与注意力带宽难以兼顾的问题，实现了内存高效且可扩展的大语言模型解码。

    

    在高并发、长上下文场景下服务大语言模型，需要随解码池规模增长的KV缓存容量与注意力带宽。现有的并行化策略往往顾此失彼：数据并行复制模型权重，限制了KV缓存容量；张量并行在达到模型的KV头数量后便无法继续扩展注意力，并将密集的集合通信推送到节点间；解码上下文并行（DCP）通过对序列分片来恢复容量，但增加了对部分注意力结果的跨节点聚合。我们观察到，在解码阶段，稠密层与注意力的扩展方式不同：稠密层将权重读取摊销到整个批次上，能够在单个NVLink域内高效运行；而注意力的开销则随每个活跃请求的KV缓存而增长。我们提出了解码令牌并行（TKNP），这是一种将权重放置与注意力所有权解耦的解码池架构。TKNP将单一模型进行分布式部署……（摘要原文截断于此）

    arXiv:2611.13293v1 Announce Type: cross  Abstract: Serving large language models at high concurrency with long contexts requires KV-cache capacity and attention bandwidth that grow with the decode pool. Existing parallelization strategies trade one against the other: data parallelism replicates model weights, limiting KV capacity; tensor parallelism stops scaling attention at the model's KV-head count and pushes dense collectives across nodes; and decode context parallelism (DCP) recovers capacity by sharding the sequence but adds cross-node aggregation of partial attention results. We observe that during decode, dense layers and attention scale differently: dense layers amortize weight reads across the batch and run efficiently within a single NVLink domain, whereas attention cost grows with every active request's KV cache. We present Decode Token Parallelism (TKNP), a decode-pool architecture that decouples weight placement from attention ownership. TKNP distributes a single model co
    
[^172]: 算力受限的多阶段训练流水线解析模型

    An Analytical Model of Compute-limited Multistage Training Pipelines

    [https://arxiv.org/abs/2611.13276](https://arxiv.org/abs/2611.13276)

    本文构建了一个解析可处理的高维多阶段训练理论模型（预训练+SFT+RL），揭示了算力在各训练阶段的分配规律，并证明有限的预训练数据会带来依赖表示的性能上限，其误差按预训练样本比例平方根的倒数缩放。

    

    现代基础模型需要经过多个阶段进行训练，然而这些阶段之间如何相互作用、算力应如何在各阶段之间分配，目前仍缺乏充分理解。我们提出了一个解析可处理的高维模型，将基于线性自编码器的无监督预训练、带有噪声分步标签的监督微调（SFT），以及具有稀疏回合奖励的强化学习（RL）结合在一起。利用随机矩阵理论和算子投影闭合方法，我们推导出与仿真结果高度吻合的近似低维学习动力学。我们证明，有限的预训练数据会造成依赖于表示的性能上限，其贝叶斯分类误差按预训练样本比例的平方根倒数缩放，并且在SFT和RL的渐近误差中观察到相同的指数。更好的预训练还能降低对噪声监督的敏感性。SFT初始化可以缩短由稀疏奖励引起的RL初始平台期……（摘要在此处截断）

    arXiv:2611.13276v1 Announce Type: new  Abstract: Modern foundation models are trained through multiple stages, yet how these stages interact and how compute should be allocated across them remain poorly understood. We introduce an analytically tractable high-dimensional model combining unsupervised pretraining with a linear autoencoder, supervised fine-tuning (SFT) with noisy stepwise labels, and reinforcement learning (RL) with sparse episodic rewards. Using random matrix theory and an operator-projection closure, we derive approximate low-dimensional learning dynamics that closely match simulations. We show that finite pretraining data imposes a representation-dependent performance limit, with Bayes classification error scaling as the inverse square root of the pretraining sample ratio, and observe the same exponent in asymptotic SFT and RL errors. Better pretraining also reduces sensitivity to noisy supervision. SFT initialization shortens the initial RL plateau induced by sparse re
    
[^173]: 基于游走策略的离策略学习

    Off-policy Learning with Excursion Policies

    [https://arxiv.org/abs/2611.13275](https://arxiv.org/abs/2611.13275)

    该论文提出“游走策略”这一结构化插值方法——智能体先按目标策略执行多步再回归行为策略，建立了其动态规划理论基础并开发了可证明可靠的离策略时序差分算法，在连续控制基准上实现了与混合策略相当的效率-稳定性权衡。

    

    离策略强化学习旨在当数据来自不同于目标策略的行为策略时，学习目标策略的价值。这通常需要在偏差与方差或稳定性之间进行权衡。一种权衡方式是在行为策略与目标策略之间进行线性插值，但这只是更广泛的插值机制空间中的一个点。我们提出了游走策略——一种结构化的替代方案，其中智能体在回归行为策略之前，先按照目标策略进行多步的“游走”。我们为这类策略推导了动态规划结果，为其在强化学习中的应用建立了理论基础。我们进一步开发了对游走价值估计具有可证明可靠性的离策略时序差分（TD）算法。我们发现，在连续控制基准测试中，游走策略提供了与混合策略相当的效率-稳定性权衡，并且我们的分析从概念上阐释了……

    arXiv:2611.13275v1 Announce Type: cross  Abstract: Off-policy reinforcement learning (RL) tries to learn the value of a target policy while data comes from a different behavior policy. This often requires trading off bias for variance or stability. One way to trade these off is to linearly interpolate between the behavior and target policies, but this represents only one point in a broader space of interpolation mechanisms. We introduce excursion policies--a structured alternative where the agent follows the target policy for a multi-step "excursion" before reverting to the behavior policy. We derive dynamic programming results for these policies, establishing the theoretical basis for their use in RL. We further develop provably sound off-policy temporal-difference (TD) algorithms for excursion value estimation. We find that excursion policies provide an efficiency-stability trade-off comparable to mixture policies in continuous control benchmarks, and our analysis illustrates concept
    
[^174]: 哪些声学特征能在压缩中幸存？用于编解码器鲁棒音频深度伪造检测的追踪器估计第一共振峰统计特征

    Which Acoustic Features Survive Compression? Tracker-Estimated First-Formant Statistics for Codec-Robust Audio DeepfakeDetection

    [https://arxiv.org/abs/2611.13258](https://arxiv.org/abs/2611.13258)

    该论文发现第一共振峰（F1）统计特征在编解码器压缩后仍保持对深度伪造语音的判别力，仅用四个F1特征构建的SVM在七种编解码器条件下性能仅退化5.9个百分点（远优于36特征模型的14个百分点），为编解码器鲁棒的音频深度伪造检测提供了极简且稳定的特征方案。

    

    在干净实验室录音上表现良好的音频深度伪造检测器，在经过电话或平台压缩后往往会性能下降。本文提出了一个具体的问题：哪些低维声学描述子在编解码器降质后仍能保持信息量？我们在ASVspoof 2019数据上筛选了36个语句级特征（共振峰、PNCC、LFCC、韵律），并在官方ASVspoof 2021 LA测试集上于七种编解码器条件下进行测试。在33个具有统计显著性的描述子中，仅有四个展现出较大的实际区分度（|r| > 0.5，bootstrap 95%置信区间），且全部为第一共振峰（F1）相关度量，彼此高度冗余（PC1解释了79%的方差）。其价值在于稳定性：36特征SVM在编解码器失配下性能退化14个百分点（EER从9.5%升至23.5%），而四特征RBF-SVM仅退化5.9个百分点（EER从17.0%升至22.9%）。F1特征子集在各条件间的波动为3.4个百分点，而全特征集为13.1个百分点。在控制说话人身份后，F1差异依然显著（p < 10^-10，混合效应模型）。

    arXiv:2611.13258v1 Announce Type: cross  Abstract: Audio deepfake detectors that work well on clean lab recordings often degrade after telephone or platform compression. This paper asks a specific question: which low-dimensional acoustic descriptors stay informative after codec degradation? We screen 36 utterance-level features(formants, PNCCs, LFCCs, prosody) on ASVspoof 2019 data and test on the official ASVspoof 2021 LA set across seven codec conditions. Of 33 statistically significant descriptors, only four show large practical separation (|r| >0.5, bootstrap 95% CIs), all first-formant (F1) measures, highly redundant (PC1: 79% variance). The payoff is stability: the 36-feature SVM degrades 14 pp under codec mismatch (9.5%-23.5% EER), while the four-feature RBF-SVM degrades only 5.9 pp (17.0%-22.9%). The F1 subset varies 3.4 pp across conditions versus 13.1 pp for the full set. F1 differences persist after controlling for speaker identity (p greater than 10^10, mixed-effects model)
    
[^175]: 面向多教师在线蒸馏的域路由增量优化

    Domain-Routed Delta Optimization for Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2611.13249](https://arxiv.org/abs/2611.13249)

    本文提出域路由增量优化方法（DRD-MOPD），通过为不同领域的样本指定各自更新的参数组件（以专家模型与基础模型的权重差初始化），在多教师在线蒸馏中显著保留原始专家的几何结构。

    

    多教师在线蒸馏为每个样本分配一个领域教师，但通常更新共享的学生参数。我们提出了域路由增量优化（DRD-MOPD），它明确规定了每个领域的样本应更新哪些参数组件。学生模型是冻结的基础模型与可训练组件之和，这些可训练组件由“专家模型减去基础模型”的权重差初始化。所有组件都参与每一次前向传播；四种更新模式定义了样本领域与组件之间的对应关系，且每种模式拥有独立的优化器状态。我们分析了这些更新规则，并通过角度、投影系数和能量来衡量对原始专家几何结构的保留程度。先前的工作（如 Open-MOPD）直接从基础模型进行蒸馏；其发布的最终检查点中，97.02% 的增量能量位于原始专家方向的张成空间之外。相比之下，我们的 Own 检查点在经过 20 次更新后，在每个方向上仍保持接近于一的投影系数……

    arXiv:2611.13249v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation assigns each sample a domain teacher but typically updates shared student parameters. We introduce Domain-Routed Delta Optimization (DRD-MOPD), which specifies which parameter components each domain's samples update. The student is the sum of a frozen base and trainable components initialized from expert-minus-base weights. All components participate in every forward pass; four update modes define the correspondence between sample domains and components, each with independent optimizer states. We analyze these update rules and measure retention of the original expert geometry through angles, projection coefficients, and energy. Prior work such as Open-MOPD distills directly from the base model; 97.02% of its released final checkpoint's increment energy lies outside the span of the original expert directions. In comparison, our Own checkpoint after 20 updates retains coefficients near one along each 
    
[^176]: GigaEdge Spench：一个面向实用流式语音增强的软件开发工具包（SDK）

    GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement

    [https://arxiv.org/abs/2611.13243](https://arxiv.org/abs/2611.13243)

    该论文发布了开源 SDK GigaEdge Spench，通过共享 C++ 核心与多平台集成（服务器、移动端、浏览器）实现实用的高速流式语音增强，其模型质量和处理速度优于 RNNoise 和 DPDFNet2 且快于实时。

    

    我们提出了 GigaEdge Spench，一个用于在 CPU 服务器、移动设备和浏览器上进行流式语音增强的开源 SDK。应用程序可以由神经去噪、归一化与动态处理模块、采样率转换，以及在支持情况下的声学回声消除来构建增强通道。这些处理阶段可以组合使用，同时归一化也可以在没有神经模型的情况下独立运行。本次发布包含一个 8 kHz INT8 去噪器，以及判别式和混合式的 48 kHz FP32 模型。一个共享的 C++ 核心通过原生接口支持 ONNX Runtime 和 TensorFlow Lite，并提供 Android 和 iOS SDK 以及 WebAssembly 集成。独立的流上下文可以共享模型资源，且来自同一流的连续帧可以在单次推理调用中处理。我们在 EARS-WHAM v2 数据集上评估了发布的模型，并将其质量与处理开销与 RNNoise 和 DPDFNet2 进行了比较。所有测试的原生和 Web 配置处理文件的速度均快于实时。

    arXiv:2611.13243v1 Announce Type: new  Abstract: We present GigaEdge Spench, an open-source SDK for streaming speech enhancement on CPU servers, mobile devices, and in browsers. Applications can build enhancement channels from neural denoising, normalization and dynamics blocks, sample-rate conversion, and acoustic echo cancellation where supported. These stages can be combined, while normalization can also run without a neural model. The release includes an 8 kHz INT8 denoiser and discriminative and hybrid 48 kHz FP32 models. A shared C++ core supports ONNX Runtime and TensorFlow Lite through native interfaces, Android and iOS SDKs, and WebAssembly integration. Independent stream contexts share model resources, and consecutive frames from one stream can be processed in a single inference call. We evaluate the released models on EARS-WHAM v2 and compare their quality and processing costs with RNNoise and DPDFNet2. All tested native and Web configurations process files faster than real 
    
[^177]: 有限基物理信息神经网络的块稀疏高斯-牛顿优化：稀疏自动微分方法

    Block-Sparse Gauss-Newton Optimization of Finite Basis Physics-Informed Neural Networks: Sparse Automatic Differentiation Approach

    [https://arxiv.org/abs/2611.13231](https://arxiv.org/abs/2611.13231)

    本文利用FBPINNs重叠域分解架构所诱导的雅可比矩阵块稀疏性，结合基于图着色的稀疏自动微分技术，实现了高效的高斯-牛顿二阶优化，大幅降低了计算成本与内存需求。

    

    在本工作中，我们为有限基物理信息神经网络开发了一种高效的高斯-牛顿优化方法。其关键要素在于FBPINNs的重叠域分解架构，该架构将每个子网络的影响局部化，从而在残差雅可比矩阵和相应的高斯-牛顿矩阵中诱导出块稀疏性。我们对这种结构进行了刻画，并通过稀疏自动微分加以利用，采用图着色技术对结构上正交的雅可比列或行进行划分。这降低了微分计算成本和内存需求，同时保留了所有结构上非零的导数信息。由此得到的压缩雅可比表示使我们能够直接以块稀疏形式组装高斯-牛顿系统，从而可以使用稀疏直接求解器和预处理迭代方法。通过这种方式，原本计算代价高昂的二阶优化问题得以缩减。

    arXiv:2611.13231v1 Announce Type: cross  Abstract: In this work, we develop an efficient Gauss-Newton optimization approach for finite basis physics-informed neural networks (FBPINNs). The key ingredient is the overlapping domain decomposition architecture of FBPINNs, which localizes the influence of each subnetwork and induces block sparsity in both the residual Jacobian and the corresponding Gauss-Newton matrix. We characterize this structure and exploit it through sparse automatic differentiation, using graph coloring to partition structurally orthogonal Jacobian columns or rows. This reduces differentiation costs and memory requirements while preserving all structurally nonzero derivative information. The resulting compressed Jacobian representations allow us to assemble the Gauss-Newton system directly in block-sparse form, enabling the use of sparse direct solvers and preconditioned iterative methods. In this way, the otherwise expensive second-order optimization problem is reduc
    
[^178]: 连续视界预测

    Continuous Horizon Forecasting

    [https://arxiv.org/abs/2611.13228](https://arxiv.org/abs/2611.13228)

    本文提出连续视界（CH）预测策略，将预测重新定义为连续算子、把预测视界作为查询变量而非索引，在理论上证明了其相对于离散策略的估计方差优势，并通过大规模实验验证了其具有竞争力的精度，同时独特地支持并行推理和对未见视界的插值与外推。

    

    多视界预测一直由离散预测策略主导，这些策略将模型输出视为固定向量或通过循环方式生成。在本文中，我们引入了连续视界（Continuous Horizon, CH）策略，它通过将预测重新定义为连续算子，并将预测视界定义为查询变量而非索引，从而对递归、直接和多步策略进行了泛化。我们为CH策略提供了理论基础，包括一个通用算子逼近定理，以及连续视界的偏差-方差分解，该分解从形式上证明了CH相对于离散策略在估计方差上的改进。我们通过仿真研究以及在M系列基准的16个大规模数据集上对CH策略进行了实证验证，结果表明CH实现了有竞争力的RMSE精度，同时独特地支持并行推理、针对性视界查询，以及对未见视界的插值和外推。

    arXiv:2611.13228v1 Announce Type: cross  Abstract: Multi-horizon forecasting has been dominated by discrete forecast strategies that treat model outputs as fixed-vectors or recurrently create them. In this paper, we introduce the Continuous Horizon (CH) strategy that generalizes upon recursive, direct, and multi-step strategies by redefining forecasts as a continuous operator and the forecast horizon as a query variable rather than an index. We present the theoretical foundations for the CH strategy, a Universal Operator Approximation Theorem, and a continuous-horizon bias-variance decomposition that formally proves the estimation variance improvements of CH over the discrete strategies. We validate the CH strategy empirically through a simulation study, and on 16 large-scale datasets from the M-series benchmark, showing how CH achieves competitive RMSE accuracy, while uniquely enabling parallel inference, targeted horizon querying, and interpolation and extrapolation across unseen hor
    
[^179]: MediPFN：面向因果中介分析的摊销式上下文干扰学习与推断

    MediPFN: Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis

    [https://arxiv.org/abs/2611.13219](https://arxiv.org/abs/2611.13219)

    MediPFN通过将预训练表格基础模型适配于干扰函数预测，并结合交叉拟合影响函数估计器与独立校准机制，为因果中介分析提供具有覆盖保证的置信区间。

    

    因果中介分析将处理效应分解为通过中介变量的间接路径和通过其他过程的直接路径。表格基础模型通过上下文学习执行分类和回归，为因果中介分析提供条件均值和倾向得分的估计。基于这些估计构建置信区间时，需要考虑拟合和组合多个干扰函数所引入的不确定性。我们提出MediPFN，这是一个摊销推断框架，它将预训练的表格基础模型适配于干扰预测任务，并通过交叉拟合的影响函数估计器组合其预测结果。一个单独的输出头从具有已知间接效应的模拟表格中学习估计器标准化误差的分位数。随后，通过监督微调、对称化以及独立校准来生成置信区间。我们建立了覆盖保证……（原文摘要在此处截断）

    arXiv:2611.13219v1 Announce Type: cross  Abstract: Causal mediation analysis decomposes a treatment effect into indirect pathways through mediators and direct pathways through other processes. Tabular foundation models perform classification and regression through in-context learning, providing estimates of conditional means and propensity scores used in causal mediation analysis. Constructing confidence intervals from these estimates requires accounting for uncertainty introduced by fitting and combining several nuisance functions. We introduce MediPFN, an amortized inference framework that adapts a pretrained tabular foundation model to nuisance prediction and combines its predictions through a cross-fitted influence-function estimator. A separate head learns quantiles of the estimator's standardized error from simulated tables with known indirect effects. Supervised fine-tuning, symmetrization, and independent calibration then produce a confidence interval. We establish coverage bou
    
[^180]: 《以同样的方式犯错：跨提示共享在RLVR中既抑制又放大奖励作弊》

    Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR

    [https://arxiv.org/abs/2611.13218](https://arxiv.org/abs/2611.13218)

    本研究在可控环境中证明RLVR在KL正则化最优时只是对被接受集合重新加权的过滤器（奖励作弊率精确继承自基础模型），而跨提示参数共享则通过提示间奖励与经验切核的协方差项产生漂移，使接受同一错误答案的提示推高它、拒绝的提示压低它，从而同时抑制并放大奖励作弊。

    

    关于可验证奖励强化学习（RLVR）如何与不完善的验证器交互，目前存在两种并存的观点：一种认为奖励作弊是从基础模型继承而来的，另一种认为强化学习会将其放大。我们在一个足够小的受控环境中研究这两种机制，以便精确计算整个响应分布。我们改变了漏洞的普适性（即接受同一错误响应的提示所占比例）以及验证器的静态假阳性率。在KL正则化的最优解处，RLVR充当一个过滤器，仅对被接受的集合重新加权而不改变其组成，因此作弊率被精确地继承。以此为零模型，一个已知的分解将与该零模型的偏差写为各提示上奖励与经验切核之间协方差的总和。在没有参数共享的情况下，来自其他提示的项会消失。而在存在参数共享时，每个接受错误答案的提示都会推高该错误答案，每个拒绝它的提示则会将其压低……（摘要至此截断）

    arXiv:2611.13218v1 Announce Type: new  Abstract: Two accounts of how reinforcement learning with verifiable rewards (RLVR) interacts with an imperfect verifier coexist: in one, reward hacking is inherited from the base model; in the other, RL amplifies it. We study both in a controlled environment small enough to compute the entire response distribution exactly. We vary the universality of a leak, or the fraction of prompts that accept the same wrong response, and the verifier's static false-positive rate. At its KL-regularized optimum, RLVR acts as a filter that reweights the accepted set without altering its composition, so the hack rate is inherited exactly. Taking that as a null model, a known decomposition writes the drift from it as a sum over prompts of covariances between reward and the empirical tangent kernel. The terms from other prompts vanish without parameter sharing. With sharing, every prompt that accepts a wrong answer pushes it up and every prompt that rejects it push
    
[^181]: 在分歧中计数：基于存在分歧的行政文本AI阅读者的美元加权测量、选择性机器标注与预算化人工复核

    Counting Under Disagreement: Dollar-Weighted Measurement, Selective Machine Labelling and Budgeted Human Review from Disagreeing AI Readers of Administrative Text

    [https://arxiv.org/abs/2611.13215](https://arxiv.org/abs/2611.13215)

    针对多个AI阅读者对行政文本标注存在分歧、且测量目标为货币总额的场景，本文提出了融合选择性机器标注与预算化人工复核的美元加权估计框架，在可控美元加权误差下实现对总额的准确测量。

    

    我们研究AI辅助测量问题，其场景是：一个文本标签需应用于大量行政记录，测量目标为货币总额，而用于校准机器的人工阅读结果本身存在分歧。测试平台来自美国税务申报中的121,070条拨款用途文本（共799,675条记录，总额1264亿美元），由一个基于规则的分类器和三个语言模型进行标注；同时按设计方案抽取651条文本，由三人在同一编码手册下独立阅读。我们在泊松抽样下给出了固定预测变量差分估计量、以及交叉拟合预测变量（其折在抽样前即分配至整个抽样框）的精确条件，并通过仿真评估所实现的固定样本量程序：8个预测变量自身总额介于112亿至150亿美元之间，经校正后相互差异在4亿美元以内。两种按规定的美元加权误差接受机器标签的规则——美元度量下的加权保形p值方法，以及基于误差设计估计的Learn-then-Test方法——（原文摘要在此处截断）

    arXiv:2611.13215v1 Announce Type: new  Abstract: We study AI-assisted measurement when one text label applies to many administrative records, the target is a monetary total, and the human readings that anchor the machines disagree. The testbed is 121,070 grant-purpose texts from US tax filings (799,675 records, \$126.4B), labelled by a rule-based classifier and three language models, with a designed sample of 651 texts read independently by three people under one codebook. We state exact conditions for a fixed-predictor difference estimator and for a cross-fitted predictor whose folds are assigned to the whole frame before sampling, under Poisson sampling, and assess the implemented fixed-size procedures by simulation: 8 predictors whose own totals span \$11.2-15.0B agree to within \$0.4B once rectified. Two rules that accept machine labels at a stated dollar-weighted error, weighted conformal p-values under a dollar measure and Learn-then-Test on the design estimate of the error, are 
    
[^182]: 言语之前，超越语音：评估幼儿期的非语言社会推理能力

    Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood

    [https://arxiv.org/abs/2611.13213](https://arxiv.org/abs/2611.13213)

    该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。

    

    儿童最早的岁月是一个高强度的社会学习时期，在此期间，意义通过眼神注视、手势和身体接触而构建。然而，社会互动的这一关键方面在多模态视频理解领域仍未得到充分评估，因为现有基准主要优先关注以对话驱动的成人互动。我们通过NEST（自然情境下的幼儿社会互动）基准来填补这一空白，这是首个以儿童为中心的非语言社会推理基准。NEST包含大量人工标注的视频片段（约12秒），这些片段在自然情境、文化背景和发展阶段方面都经过精心挑选以确保多样性。除了数据匮乏之外，该领域针对性的模型改进还受到粗略评估方案的阻碍，这些方案将基础感知与高阶推理混为一谈。为解决这一问题，NEST引入了一种组合式评估方案，在原子级交互线索的基础上构建（摘要在此处截断）……

    arXiv:2611.13213v1 Announce Type: cross  Abstract: Children's earliest years are a period of intense social learning, during which meaning is built through gaze, gesture, and physical contact. Yet, this critical aspect of social interaction remains under-evaluated in multimodal video understanding, as existing benchmarks prioritize the conversation-driven interactions of adults. We address this gap with \bench ($\textbf{N}$aturalistic $\textbf{E}$arly-childhood $\textbf{S}$ocial in$\textbf{T}$eractions), the first child-centric benchmark for nonverbal social reasoning. \bench comprises \totalvids manually annotated clips ($\sim$12s) curated for diversity across naturalistic settings, cultural contexts, and developmental stages. Beyond a lack of data, targeted model improvement in this domain is hindered by coarse evaluation schemas that conflate basic perception with higher-order reasoning. To resolve this, \bench introduces a compositional schema that builds atomic interaction cues in
    
[^183]: 确定性不可知PAC学习的近最优样本复杂度

    Near-Optimal Sample Complexity of Deterministic Agnostic PAC Learning

    [https://arxiv.org/abs/2611.13210](https://arxiv.org/abs/2611.13210)

    本文证明了VC维为 $d$、直径为 $K$ 的概念类在确定性标签下的不可知PAC学习样本复杂度为 $\widetilde{O}(K^{1/3} d^{2/3}/\varepsilon)$，并构造了对数因子内紧的下界，填补了此前上下界之间的差距。

    

    我们研究确定性标签下的不可知PAC学习问题，其中样本由任意确定性函数标注，该函数不必属于概念类 $C$。该模型刻画了不含随机标签噪声的模型误设情形，介于可实现设定与完全不可知设定之间。先前的工作表明，当概念类 $C$ 具有有限直径 $K$ 和VC维 $d$ 时，确定性标签可以将样本复杂度上界从经典的不可知保证 $O(d/\varepsilon^2)$ 改进到 $\widetilde{O}(K/\varepsilon)$，但上下界之间存在较大差距。我们的主要结果使这一图景更加清晰：我们证明，每一个VC维为 $d$ 且直径有限为 $K$ 的概念类都可以以样本复杂度 $\widetilde{O}\left(\frac{K^{1/3} d^{2/3}}{\varepsilon}\right)$ 进行学习，并且我们构造了使该界在对数因子范围内紧的概念类。我们还研究了其在线版本的对应问题。（原文摘要在此处截断）

    arXiv:2611.13210v1 Announce Type: cross  Abstract: We study the problem of agnostic PAC learning under deterministic labels, where the examples are labeled by an arbitrary deterministic function, which need not belong to the concept class $C$. This model captures misspecification without stochastic label noise and lies between the realizable and fully agnostic settings. Prior work showed that, when $C$ has finite diameter $K$ and VC dimension $d$, deterministic labels can improve the sample complexity upper bound from the classical agnostic guarantee $O(d/\varepsilon^2)$ to $\widetilde{O}(K/\varepsilon)$, but left a large gap between upper and lower bounds. Our main result sharpens this picture. We prove that every class of VC dimension $d$ and finite diameter $K$ is learnable with sample complexity $\widetilde{O}\left( \frac{K^{1/3} \, d^{2/3}}{\varepsilon} \right)$, and we construct classes for which this bound is tight up to logarithmic factors. We also study the online analogue of 
    
[^184]: HiCAP：基于个性的对话生成分层文化适应框架

    HiCAP: Hierarchical Cultural Adaptation via Personality for Dialogue Generation

    [https://arxiv.org/abs/2611.13207](https://arxiv.org/abs/2611.13207)

    提出HiCAP框架，将个性特质通过分层控制机制融入对话生成——先使回应与文化基调对齐、再依据个性细化语言表达与内容，从而实现文化自适应的对话生成。

    

    对话系统中的文化适应是指生成符合不同文化语言规范和社会期望的回应的能力。然而，现有的对话生成方法对个性在塑造文化表达和沟通风格中所起的作用关注有限。因此，我们提出了HiCAP，这是一个将个性特质融入文化自适应对话生成的新型框架。HiCAP采用一种以个性为条件的控制机制，通过分层方式调节回应风格：首先使回应与文化基调保持一致，随后根据个体个性属性对语言表达和内容进行细化。为了嵌入这些因素，该模型结合了监督微调与基于指令的调优，并借助动态指令模板和多智能体演示来捕捉文化与个性的多样化组合。在CulturalPe（数据集）上的实验……

    arXiv:2611.13207v1 Announce Type: new  Abstract: Cultural adaptation in dialogue systems refers to the ability to generate responses that conform to linguistic norms and social expectations across different cultures. However, existing dialogue generation approaches pay limited attention to the role of personality in shaping cultural expression and communication style. We therefore propose HiCAP, a novel framework that integrates personality traits into culturally adaptive dialogue generation. HiCAP employs a personality-conditioned control mechanism that hierarchically modulates response style by first aligning it with the cultural tone and then refining linguistic expression and content according to individual personality attributes. To embed these factors, the model combines supervised fine-tuning and instruction-based tuning, supported by dynamic instruction templates and multi-agent demonstrations to capture diverse combinations of culture and personality. Experiments on CulturalPe
    
[^185]: 审计梯度，锚定策略：面向高效流动控制的基于模型的强化学习

    Audited Gradients, Anchored Policies: Model-Based Reinforcement Learning for Efficient Flow Control

    [https://arxiv.org/abs/2611.13203](https://arxiv.org/abs/2611.13203)

    提出一种“梯度审计”机制，在基于代理模型的流动控制策略优化中，依据预先固定的接受准则将代理模型的动作梯度与高保真求解器的有限差分估计进行比对验证，从而在大幅降低计算成本的同时避免代理模型梯度误差导致的策略失效。

    

    流动的反馈控制器设计成本高昂，因为每个候选控制器都必须使用高保真求解器进行评估。可微代理模型降低了这一成本，但也引入了一种预测误差无法检测到的失效模式。策略优化使用目标函数对动作的导数，我们称之为动作梯度，该梯度既可以通过代理模型计算，也可以通过高保真求解器计算。通过代理模型进行优化时只使用其动作梯度，而代理模型可能在紧随流动轨迹的同时错误估计这一导数。我们将代理模型的动作梯度与高保真求解器在留出状态上得到的中心有限差分估计进行比较，并在比较前预先固定接受准则，我们将这种比较称为审计。当代理模型通过审计时，再将代理动作梯度拟合到低维反馈律上……

    arXiv:2611.13203v1 Announce Type: new  Abstract: Feedback controllers for flows are expensive to design, since each candidate must be evaluated with a high-fidelity solver. Differentiable surrogates reduce that cost, but they also introduce a failure mode that prediction error cannot detect. Policy optimization uses the derivative of the objective with respect to the action, which we call the action gradient and which can be computed either through the surrogate or through the high-fidelity solver. Optimization through the surrogate uses only its action gradient, and a surrogate can follow the flow trajectory closely while misestimating this derivative. The surrogate action gradient is compared against a central finite-difference estimate from the high-fidelity solver on held-out states, under acceptance criteria fixed before the comparison, and we call that comparison an audit. A low-dimensional feedback law is then fitted to the surrogate action gradient when the surrogate passes the
    
[^186]: 面向OOD弹性运行时预测的XAI驱动置信度估计

    XAI-Driven Confidence Estimation for OOD-Resilient Runtime Prediction

    [https://arxiv.org/abs/2611.13199](https://arxiv.org/abs/2611.13199)

    该论文提出LEX——首个用于基于机器学习资源管理的事后XAI运行时置信度估计框架，通过高斯混合模型聚类训练输入嵌入并关联逐簇预测误差界限，无需重训练即可提供细粒度的输入级置信度信号，在OOD工作负载上实现约94%的包含率，并产生比全局基线紧约17%的置信度界限。

    

    基于机器学习的资源管理（RM）模型在运行时面对分布外（OOD）输入时性能会退化，然而文献中现有方法依赖于单一的全局误差裕度，这种方式过于保守，无法区分可靠预测与灾难性故障。我们提出了轻量级运行时可解释性（LEX），这是首个用于基于机器学习的资源管理中输入特定运行时置信度估计的事后XAI框架。LEX使用高斯混合模型对训练输入空间的嵌入进行聚类，并将每个簇的预测误差界限关联到输入样本，从而无需重新训练即可实现细粒度的运行时置信度信号。在Intel Alder Lake平台上，针对最先进的基于LSTM的IPS预测模型、跨34个基准应用程序进行评估，LEX在未见工作负载上实现了约94%的包含率，并且产生的逐输入置信度界限比全局基线紧约17%，而其运行时开销几乎可以忽略。

    arXiv:2611.13199v1 Announce Type: cross  Abstract: Machine learning-based resource management (RM) models degrade under out-of-distribution (OOD) inputs at runtime, yet existing approaches in the literature rely on a single global error margin that is overly conservative and fails to distinguish reliablepredictions from catastrophic failures. We propose Lightweight Runtime EXplainability (LEX), the first post-hoc XAI framework for input-specific runtime confidence estimation in ML-based RM. LEX clusters the embeddings of the training input space using a Gaussian Mixture Model and associates per-cluster prediction error bounds to input samples, enabling fine-grained runtime confidence signals without retraining. Evaluated on a state-of-the-art LSTM-based IPS forecasting model on Intel Alder Lake, across 34 benchmark applications, LEX achieves ~94% containment on unseen workloads and produces ~17% tighter per-input confidence bounds than global baselines. With its negligible runtime over
    
[^187]: 用于多模态归因的结构化核代理模型

    Structured Kernel Surrogate Models for Multimodal Attribution

    [https://arxiv.org/abs/2611.13198](https://arxiv.org/abs/2611.13198)

    提出结构化核代理模型及SKS算法，利用结构化核岭回归捕捉样本间与模态间的交互作用，首次解决多模态系统中的跨样本模态归因问题。

    

    我们研究了跨样本模态归因问题，即测量一个样本中的某种模态如何影响其他多模态样本的学习结果。这个问题在多模态系统中至关重要，因为某种数据模态（例如文本描述）对于某个样本（例如自身就能解释的图像）可能是冗余的，但同样的数据模态对另一个样本来说却可能是唯一的信息来源，或者需要与其他模态结合时才具有信息价值。先前的工作主要集中在单样本级别的影响估计上，如何捕捉跨样本和跨模态的交互仍然是一个悬而未决的问题。为了应对上述场景，我们提出了结构化核代理模型，这是一种在联合训练时从样本和模态到预测结果的代理映射。我们设计了一种算法SKS，该算法应用具有结构化复合结构的核岭回归（原文在此处截断）。

    arXiv:2611.13198v1 Announce Type: cross  Abstract: We study the problem of cross-sample modality attribution, which involves measuring how one modality in one sample affects the learning outcome in other multimodal samples. This problem is crucial in a multimodal system because a data modality (such as a text description) may be redundant for one sample (a self-explanatory image), but the same data modality would otherwise be uniquely informative for another sample, or informative when combined with other modalities. Prior work focuses on influence estimation at the level of one sample, and it remains an open question how one should capture the interactions across both samples and modalities. To capture scenarios like the above, we propose structured kernel surrogate models, which are surrogate mappings from both samples and modalities to the prediction outcome when they are trained together. We design an algorithm, SKS, which applies kernel ridge regression with a structured, composit
    
[^188]: EvoSurv：进化感知的多模态生存建模揭示互补的预后信号

    EvoSurv: Evolution-Aware Multimodal Survival Modeling Reveals Complementary Prognostic Signals

    [https://arxiv.org/abs/2611.13190](https://arxiv.org/abs/2611.13190)

    该研究提出了eTime进化特征表示和EvoSurv多模态生存预测框架，将肿瘤基因组进化信息（克隆结构、突变谱、拷贝数改变等）与临床数据和RNA表达相结合，在六个TCGA癌症队列中全面超越现有最先进方法，揭示了肿瘤进化中蕴含的互补预后信号。

    

    由于肿瘤异质性的存在，准确的癌症生存预测仍然充满挑战。尽管多模态生存模型已经整合了临床与分子数据，但它们往往忽略了基因组改变中所编码的进化信息。为填补这一空白，我们开发了eTime，这是一种38维的进化特征表示，由八个特征组构成，用于刻画克隆与亚克隆突变负荷、突变谱、功能性后果、拷贝数改变以及失巢凋亡相关的基因组变化。我们进一步构建了EvoSurv，一个将eTime与临床变量和RNA表达相整合的多模态生存预测框架。在包含2,782名患者的六个TCGA癌症队列中，EvoSurv在全部六种癌症中均取得了最高的平均一致性指数（C-index），平均值为0.663，优于三种最先进的生存预测方法。更重要的是，我们评估了进化特征能否提供额外的预后价值……

    arXiv:2611.13190v1 Announce Type: cross  Abstract: Accurate cancer survival prediction remains challenging due to tumor heterogeneity. Although multimodal survival models integrate clinical and molecular data, they often overlook evolutionary information encoded in genomic alterations. To address this gap, we developed eTime, a 38-dimensional evolutionary representation comprising eight feature groups that characterize clonal and subclonal mutation burden, mutational spectra, functional consequences, copy-number alterations, and anoikis-related genomic changes. We further developed EvoSurv, a multimodal survival framework integrating eTime with clinical variables and RNA expression. Across six TCGA cancer cohorts comprising 2,782 patients, EvoSurv achieved the highest mean concordance index (C-index) in all six cancers, averaging 0.663 and outperforming three state-of-the-art survival prediction methods. More importantly, we assessed whether evolutionary features provide additional pro
    
[^189]: 面向NVLink上混合专家推理的单边专家并行All-to-All通信

    One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink

    [https://arxiv.org/abs/2611.13188](https://arxiv.org/abs/2611.13188)

    NVLinkOneSided提出了一种面向NVLink域内混合专家推理的单边all-to-all通信方法，通过免元数据交换的直接推送、原地读取机制以及Fence和Sentinel两种同步协议，在Blackwell GPU上比DeepEP快1.2-3.0倍。

    

    专家并行将混合专家层的专家分片到多个GPU上，因此每个MoE层需要两次all-to-all通信，即分发和组合。与训练相比，推理时每个GPU一次处理的token数量相对较少，在小批量规模下，主导延迟的是不随token数量增长的固定开销，而非数据移动本身。我们提出了NVLinkOneSided，一种面向单个NVLink域内GPU的单边EP all-to-all通信方法。每个源rank无需预先交换元数据，直接将其token推送到目标GPU上以rank为主组织的缓冲区中，分组GEMM则直接原地读取这些数据，而无需从按专家复制的副本中读取。我们为通信所需的同步提供了两种协议，即Fence和Sentinel，分别适用于高吞吐量和低延迟场景。在Blackwell GPU上，NVLinkOneSided完成分发和组合的速度比DeepEP快1.2-3.0倍。

    arXiv:2611.13188v1 Announce Type: cross  Abstract: Expert parallelism (EP) shards the experts of a Mixture-of-Experts (MoE) layer across GPUs, so every MoE layer requires two all-to-all communications, namely dispatch and combine. Compared with training, inference processes relatively few tokens per GPU at a time, and at small batch sizes fixed costs that do not grow with the number of tokens, rather than data movement, dominate the latency. We present NVLinkOneSided, a one-sided EP all-to-all for GPUs within a single NVLink domain. Each source rank pushes its tokens, without first exchanging metadata, into rank-major buffers on the target GPUs, and the grouped GEMMs read them in place instead of from a per-expert copy. We provide two protocols for the synchronization that the communication requires, namely Fence and Sentinel, for high-throughput and low-latency scenarios respectively. On Blackwell GPUs, NVLinkOneSided completes dispatch and combine 1.2-3.0$\times$ faster than DeepEP, 
    
[^190]: 基于自适应收集人类反馈的预测驱动大语言模型排名

    Prediction-Powered LLM Ranking with Adaptively Collected Human Feedback

    [https://arxiv.org/abs/2611.13186](https://arxiv.org/abs/2611.13186)

    提出PP-Rank框架，通过将廉价的LLM评判者预测与自适应收集的人类反馈相结合，在自适应采样下实现有效统计推断，显著提升大语言模型排名评估的精度。

    

    通过人类评估对大型语言模型（LLM）进行排名对于下游决策至关重要。由于人类反馈成本高昂，评估平台通常采用自适应方式收集反馈，即利用先前观察到的反馈来优先进行信息量更大的模型比较。为了在有限的人类反馈下进一步提高评估精度，评估者可以利用大量未评分的响应以及相对廉价的LLM评判者；然而，评判者偏差和自适应数据依赖性对标准的独立同分布（i.i.d.）推断框架构成了挑战。我们提出了预测驱动排名框架（Prediction-Powered Ranking, PP-Rank），该框架将LLM评判者的预测与人类反馈相结合，在自适应采样下既能提高精度又能提供有效推断。数据驱动的预测权重在所提出的权重类别中最小化渐近协方差迹。在理论方面，我们针对一般评估目标和同时排名推断建立了渐近正态性，并通过联合鞅分析（原文在此处截断）……

    arXiv:2611.13186v1 Announce Type: new  Abstract: Ranking large language models (LLMs) through human evaluations is essential for downstream decision-making. Because human feedback is expensive, evaluation platforms often collect it adaptively, prioritizing informative model comparisons using previously observed feedback. To further improve precision with limited human feedback, evaluators can draw on abundant unrated responses and relatively inexpensive LLM judges; however, judge bias and adaptive dependence challenge standard i.i.d. inference frameworks. We propose Prediction-Powered Ranking (PP-Rank), a framework that combines LLM-judge predictions with human feedback to improve precision while providing valid inference under adaptive sampling. Data-driven prediction weights minimize the asymptotic covariance trace within the proposed class. In theory, we establish asymptotic normality for general evaluation targets and simultaneous ranking inference. A joint martingale analysis hand
    
[^191]: 在符号模仿学习中取得进展：离线演示与在线调优

    Making Strides in Symbolic Imitation Learning: Offline Demonstrations, Online Tuning

    [https://arxiv.org/abs/2611.13182](https://arxiv.org/abs/2611.13182)

    提出符号模仿学习框架，通过离线数据将oracle策略蒸馏为线性运动基元的决策列表，并结合在线ARS调优，在20个强化学习基准任务中逼近oracle性能且在11个任务中实现超越，同时生成简洁可解释的策略。

    

    深度强化学习（DRL）方法凭借其在函数逼近方面的有效性，一直是连续控制任务中的主导范式，而具有可解释性的进化方法则恰恰因为相反的原因而大幅落后。在这项工作中，我们提出了符号模仿学习，这是一个通过离线数据集将oracle策略蒸馏为具有线性模型运动基元的决策列表的框架。我们表明，该方法结合在线增广随机搜索（ARS）对常数和运动基元的调优，在20个基准强化学习任务中能够逼近oracle策略的性能，并在其中11个任务上超越了oracle策略，同时生成了领域专家更容易理解的简洁、可解释的策略。

    arXiv:2611.13182v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) methods have been the dominant paradigm in continuous control tasks due to their efficacy in function approximation, while interpretable evolutionary methods have lagged largely behind for the opposite reason. In this work, we introduce Symbolic Imitation Learning, a framework for distilling an oracle policy through an offline dataset into a decision list with linear model motion primitives. We show that this, in conjunction, with online Augmented-Random Search (ARS) tuning of the constants and motion primitives approximates the performance of the oracle across 20 benchmark RL tasks and exceeds the oracle in 11 of those 20 tasks while simultaneously producing parsimonious, interpretable policies that domain experts can better understand.
    
[^192]: RadRef：一种用于放射学报告生成的无参考评估指标

    RadRef: A Reference-Free Metric for Radiology Report Generation

    [https://arxiv.org/abs/2611.13168](https://arxiv.org/abs/2611.13168)

    提出了RadRef，一种无需参考报告的放射学报告生成评估指标，通过将报告陈述与图像推导的临床发现信念进行比对评分，在遗漏发现评估上与放射科医生排名高度一致，解决了无参考报告时无法评估的问题。

    

    AI生成的放射学报告通常通过与放射科医生撰写的参考报告进行比较来评估，当没有参考报告时便无法对其进行评分。我们提出了RadRef，这是一种无参考评估指标，它在临床发现词汇表上，将报告中的陈述与从图像推导出的信念进行比对评分，并根据与临床适应症的相关性进行加权。在ReXVal数据集上，RadRef在非并列的同影像报告对上与放射科医生排名的一致性达到73.1%，总体上与领先的基于参考的指标相当。它在假阳性发现方面也具有竞争力，而在遗漏（漏诊发现）方面的相对表现最为突出。我们进一步评估了来自100项MIMIC-CXR研究的100份模型生成报告和100份放射科医生撰写的报告，这些报告均由两位经过委员会认证的放射科医生独立标注。在遗漏方面，RadRef达到Kendall's tau-b = 0.295，而两位放射科医生之间的一致性为0.386，且没有任何被比较的指标与RadRef存在显著差异。

    arXiv:2611.13168v1 Announce Type: new  Abstract: AI-generated radiology reports are usually evaluated by comparison with a radiologist-authored reference, leaving no way to score them when no reference exists. We introduce RadRef, a reference-free metric that scores report claims against image-derived beliefs over a clinical finding vocabulary, weighted by relevance to the clinical indication. On ReXVal, RadRef achieves 73.1% agreement on non-tied same-radiograph report pairs with radiologist rankings, comparable to leading reference-based metrics overall. It is also competitive on false findings and has its clearest relative result on omissions (missed findings). We further evaluate 100 model-generated and 100 radiologist-authored reports from 100 MIMIC-CXR studies, all independently annotated by two board-certified radiologists. On omissions, RadRef attains Kendall's tau-b = 0.295, against 0.386 between the two readers themselves, and no compared metric differs significantly from Rad
    
[^193]: 在线策略学习的高概率风险控制

    High Probability Risk Control for Online Policy Learning

    [https://arxiv.org/abs/2611.13163](https://arxiv.org/abs/2611.13163)

    该论文提出一种基于置换的加权风险评估器（权重由Sinkhorn算法高效近似）结合高斯自助法构造风险置信上界，从而在反馈协变量偏移下实现高概率风险控制的在线策略选择。

    

    我们研究在线策略学习的风险控制问题，其中智能体自适应地收集数据，旨在学习一个新策略，使其风险在每一轮都以高概率保持在容忍阈值以下。这种策略更新和数据收集过程会引发反馈协变量偏移（FCS）：观测数据通过不断演化的策略和数据收集过程依赖于历史。FCS给风险控制带来了两个实质性难题：1）对每个候选策略进行高效的风险评估；2）在高概率风险控制下进行有效的策略选择。为解决这些问题，我们提出了一种基于置换的加权风险评估器，其权重通过Sinkhorn算法高效近似，并选择估计风险置信上界（UCB）低于目标阈值的策略，其中UCB通过我们提出的高斯自助法获得。理论上，我们证明了我们的风险评估器的方差上界不超过……

    arXiv:2611.13163v1 Announce Type: new  Abstract: We study risk control for online policy learning, where an agent adaptively collects data and aims to learn a new policy whose risk stays below a tolerance threshold at every round with high probability. Such policy updates and data collection induce feedback covariate shift (FCS): observed data depend on history through evolving policies and the data-collection process. FCS brings two substantive problems for risk control: 1) efficient risk evaluation for each candidate policy and 2) valid policy selection with high-probability risk control. To address these, we propose a permutation-based weighted risk evaluator with weights efficiently approximated by the Sinkhorn algorithm and select policies whose estimated risk upper confidence bounds (UCB) fall below the target threshold, with the UCB obtained through a proposed Gaussian bootstrap method. Theoretically, we prove that our risk evaluator has a no-larger variance upper bound than tha
    
[^194]: 你的VLA有多少成分其实是kNN

    How Much of Your VLA Is kNN

    [https://arxiv.org/abs/2611.13160](https://arxiv.org/abs/2611.13160)

    研究揭示LIBERO基准上VLA的高成功率很大程度上只是近邻演示检索的体现——无需任何训练的检索方法即可达到91.7%成功率，并提出局部动作一致性指标来衡量检索质量并指导检索键训练。

    

    视觉-语言-动作模型目前在LIBERO仿真基准上的成功率已超过95%，但这些性能中有多少需要超越训练演示之外的泛化能力仍不清楚。在这项工作中，我们证明检索近邻演示并复用其动作块，在不进行任何策略训练的情况下即可在LIBERO上达到91.7%的成功率。在LIBERO-Plus扰动下，检索方法在所有被评估的策略中表现出最小的性能下降。在九个公开可用的VLA上，微调主要使视觉-语言骨干成为更好的检索键，而VLA输出的动作仍然与演示动作十分接近。此外，我们提出了局部动作一致性，这是一种离线指标，用于衡量表示空间中的相邻观测是否也与相似的未来动作相关联。我们证明LAC能够追踪检索的成功率，并可用于指导检索键的训练。我们的评估……

    arXiv:2611.13160v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models now exceed 95% success on the LIBERO simulation benchmark, yet it remains unclear how much of this performance requires generalization beyond the training demonstrations. In this work, we show that retrieving nearby demonstrations and reusing their action chunks achieves a success rate of 91.7 % on LIBERO without policy training. Under LIBERO-Plus perturbations, retrieval shows the smallest drop of all evaluated policies. Across nine publicly available VLAs, fine-tuning mostly makes the vision-language backbone a better retrieval key, and the actions of the VLAs remain close to the demonstrations. In addition, we introduce Local Action Consistency (LAC), an offline metric that measures whether nearby observations in representation space are also associated with similar future actions. We show that LAC tracks the success rate of retrieval and can guide the training of a retrieval key. Our evaluation a
    
[^195]: VIGOR：多模态大语言模型视觉推理依据正确性的基准测试

    VIGOR: Benchmarking Visual Rationale Correctness in Multimodal Large Language Models

    [https://arxiv.org/abs/2611.13150](https://arxiv.org/abs/2611.13150)

    VIGOR提出了一个通过将视觉推理依据的显式概念与像素级接地区域配对、跨越多个抽象层次，来评估多模态大语言模型是否真正“看着自己所说的内容”（即其答案是否由正确视觉证据支持）的基准测试。

    

    多模态大语言模型（MLLMs）通常以是否给出正确答案来评估，但仅凭答案正确并不能揭示其输出是否得到了正确视觉证据的支持。我们提出了VIGOR，一个用于评估MLLMs视觉接地推理依据的基准。VIGOR将视觉推理依据表示为显式概念与像素级接地区域的配对，涵盖多个抽象层次，包括低层视觉属性、中间结构、物体、部件以及高层运动和关系。为了大规模构建VIGOR，我们开发了一个自动标注框架，该框架在可用时复用现有的密集监督标注，并通过基于候选区域的标注加保守的多模态验证来生成缺失的概念接地区域。利用VIGOR，我们通过比较模型对每个所表达概念给出的视觉证据与标注的接地区域，来评估MLLMs是否“言出必看”（即是否真正关注其答案所依据的视觉区域）。

    arXiv:2611.13150v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) are typically evaluated by whether they produce correct answers, but correctness alone does not reveal whether their outputs are supported by the right visual evidence. We introduce VIGOR, a benchmark for evaluating VIsually GrOunded Rationales in MLLMs. VIGOR represents visual rationales as explicit concepts paired with pixel-level groundings, covering multiple abstraction levels including low-level visual properties, intermediate structures, objects, parts and high-level motion, relations. To construct VIGOR at scale, we develop an automatic annotation framework that reuses existing dense supervision when available and generates missing concept groundings through proposal-based annotation followed by conservative multimodal verification. Using VIGOR, we evaluate whether MLLMs "look at what they say" by comparing model-derived visual evidence for each expressed concept against annotated ground-
    
[^196]: 万物各归其位：注意力是一个无线电接收头

    Everything in its right place: attention is a radio head

    [https://arxiv.org/abs/2611.13146](https://arxiv.org/abs/2611.13146)

    该论文提出注意力机制可以从物理同步中涌现——把历史词元建模为频率随时间漂移的振荡器、把当前词元建模为无线电接收器，通过求解Kuramoto同步问题闭式导出一个具有“锁定/静默”两相的注意力层，从而将Transformer注意力重新解释为无线电调谐接收过程，并自然实现选择、弃权、计数及超长序列泛化。

    

    注意力是语言模型的核心，是一种基于词元的数字计算：指数函数负责选择词元，除法负责归一化，位置编码负责设定顺序。我们证明，注意力也可以从接收器与其上下文词元之间的同步中涌现。每个历史词元都是一个振荡器，其状态携带自身内容，其频率随着“年龄”增长而漂移。当前词元通过内容进行选择——将天线与查询对齐；也通过位置进行选择——用调谐旋钮设定接收器的频率。对该接收器求解Kuramoto问题，可以得到一个闭式解的注意力层，它具有两个相位：当耦合强度高于由上下文设定的临界值时，接收器进入锁定状态，其权重被归一化；低于临界值时，接收器保持静默，其权重之和小于一。经过训练的网络利用这两个相位来实现选择、弃权和计数，在理论所预测之处失效，并能持续读取远超训练长度的序列。注意力的每一部分……（原文摘要在此处截断）

    arXiv:2611.13146v1 Announce Type: cross  Abstract: Attention, the core of language models, is a digital calculation over tokens: an exponential selects them, a division normalizes and a positional code sets the order. We show that attention can instead emerge from the synchronization of a receiver with its context tokens. Each past token is an oscillator whose state carries its content and whose frequency drifts with its age. The present token selects them by content, aligning an antenna with its query, and by position, setting the frequency of a receiver with its dial. Solving Kuramoto's problem for this receiver yields an attention layer in closed form with two phases: above a critical coupling set by the context, the receiver locks and its weights normalize; below it, it stays silent and its weights sum to less than one. Trained networks use both phases to select, abstain and count, fail where the theory predicts and keep reading far beyond the training length. Each part of attentio
    
[^197]: 拜占庭鲁棒的Muon：面向分布式学习的弹性动量正交化

    Byzantine-Robust Muon: Resilient Momentum Orthogonalization for Distributed Learning

    [https://arxiv.org/abs/2611.13137](https://arxiv.org/abs/2611.13137)

    本文在统一的一阶信赖域框架下为拜占庭鲁棒的Muon优化器建立了不依赖独立同分布假设的收敛性保证，并引入了与信赖域几何相一致的非欧几里得数据异构性概念。

    

    包括基于线性最小化预言机（LMO）和动量正交化的Muon优化器在内的优化方法，在大规模和分布式学习中表现出强大的实证性能，但它们对对抗行为和数据异构性的鲁棒性尚未得到充分理解。我们在统一的一阶信赖域框架内研究这些方法，并分析结合鲁棒聚合的Muon更新，用于拜占庭鲁棒的分布式和联邦学习。我们的分析不依赖独立同分布假设，并引入了与信赖域几何相一致的非欧几里得异构性概念。在该框架下，我们为一般非凸目标和星形凸目标建立了收敛性保证。我们的结果为基于Muon的拜占庭鲁棒优化提供了理论刻画，其加性异构性误差下界与已知的不可能性结果一致，且在显式条件下成立。

    arXiv:2611.13137v1 Announce Type: new  Abstract: Optimization methods, including the Muon optimizer based on the Linear Minimization Oracle (LMO) and momentum orthogonalization, have shown strong empirical performance in large-scale and distributed learning, yet their robustness to adversarial behavior and data heterogeneity is not well understood. We study these methods within a unified first-order trust-region framework and analyze Muon-based updates with robust aggregation for Byzantine-robust distributed and federated learning. Our analysis does not rely on the i.i.d. assumption and introduces a non-Euclidean notion of heterogeneity consistent with the trust-region geometry. Under this framework, we establish convergence guarantees for both general non-convex and star-convex objectives. Our results provide a theoretical characterization of Byzantine-robust Muon-based optimization, with an additive heterogeneity error floor consistent with known impossibility results, under explicit
    
[^198]: 哪些成功案例能成为好的参考？面向多轮在线策略蒸馏的动作序列对齐方法

    Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation

    [https://arxiv.org/abs/2611.13135](https://arxiv.org/abs/2611.13135)

    该论文将多轮在线策略蒸馏中的成功参考选择建模为自由目标最优传输问题，并提出以最长公共子序列（LCS）为基础的动作序列对齐作为轨迹兼容性的可观测代理，为每条失败轨迹匹配最兼容的成功参考，从而避免误导性监督、提升蒸馏效果。

    

    在线策略蒸馏（OPD）通过在学生模型自身的轨迹上进行监督，来减少训练与推理之间的分布失配。对于多轮工具使用智能体而言，让教师模型以经过验证的同任务成功轨迹作为参考，可以提供有用的后见之明；但不同的成功案例可能到达不兼容的交互状态，从而提供误导性的监督信号。我们将成功参考的选择问题形式化为一个自由目标的最优传输问题，即把每条失败轨迹分配给与其最兼容的成功参考。由于潜在的状态路径是隐藏的，我们采用动作序列对齐作为兼容性的可观测代理，并用最长公共子序列（LCS）加以实现——LCS能够识别出共享的有序动作主干，同时允许可恢复的绕路。这一视角使轨迹兼容性成为多轮OPD中参考选择的核心准则。在ALFWorld上的实验……（原文摘要在此处截断）

    arXiv:2611.13135v1 Announce Type: cross  Abstract: On-policy distillation (OPD) reduces train-inference distribution mismatch by supervising a student on its own trajectories. For multi-turn tool-use agents, conditioning the teacher on a validated same-task successful trajectory as a reference can provide useful hindsight, but different successes may reach incompatible interaction states and offer misleading supervision. We cast successful-reference selection as a free-target optimal transport problem, which reduces to assigning each failed trajectory its most compatible successful reference. Because latent state paths are hidden, we use action-sequence alignment as an observable proxy for compatibility and instantiate it with longest common subsequence (LCS), which identifies a shared ordered backbone of actions while allowing recoverable detours. This perspective makes trajectory compatibility the central criterion for reference selection in multi-turn OPD. Experiments on ALFWorld an
    
[^199]: ASPIRE：一个用于迭代推理与探索的智能体化系统辨识平台

    ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration

    [https://arxiv.org/abs/2611.13114](https://arxiv.org/abs/2611.13114)

    ASPIRE是一个由大语言模型智能体驱动的非线性系统辨识框架，能够自主选择并执行辨识方法，从测量数据构建闭式非线性模型，融入先验物理知识并独立验证，且支持自主与交互两种运行模式。

    

    系统辨识需要在辨识方法的选择、先验物理知识的融入以及模型验证等方面做出专家决策。我们提出了ASPIRE，这是一个用于非线性系统辨识的框架，其中大型语言模型智能体负责选择并执行辨识方法。ASPIRE能够从测量数据中构建闭式（解析形式）的非线性静态或动态模型，并可选择性地辅以系统描述、模型的预期用途，以及以形式化方式或自然语言表达的物理要求。每个模型包含其参数、估计记录、对模型各项的解释，以及在智能体无法访问的测试数据上评估的误差。形式化的物理要求由独立机制进行验证。ASPIRE既可以自主运行，也可以交互式运行，允许智能体提出问题，用户也可以请求进一步的分析和额外的辨识轮次。我们在三个实测系统上对ASPIRE进行了评估：一个移动机器人……（摘要截断）

    arXiv:2611.13114v1 Announce Type: cross  Abstract: System identification requires expert decisions on the identification method, incorporation of prior physical knowledge, and model validation. We propose ASPIRE, a framework for nonlinear system identification in which a large language model agent selects and executes the identification method. ASPIRE constructs closed-form, nonlinear static or dynamic models from measured data, optionally supplemented by a system description, the intended use of the model, and physical requirements expressed formally or in natural language. Each model includes its parameters, an estimation record, an interpretation of its terms, and an error evaluated on test data inaccessible to the agent. Formal physical requirements are verified independently. ASPIRE operates autonomously or interactively, allowing the agent to ask questions and the user to request further analyses and identification rounds. We evaluate ASPIRE on three measured systems: a mobile ro
    
[^200]: 从内在动机构建世界模型

    Building World Models from Intrinsic Motivation

    [https://arxiv.org/abs/2611.13113](https://arxiv.org/abs/2611.13113)

    该论文将24种内在动机目标统一整合到一个基于模型的强化学习框架中，系统评估了它们所构建的世界模型在零样本、少样本和多任务控制下的表现。

    

    在缺乏外在奖励或任务监督的情况下，人类行为由内在动机引导——即驱动好奇心、创造力、游戏以及其他智能核心要素的神经信号。内在动机在构建能够实现泛化与适应的稳健世界模型方面发挥着重要的发育作用。然而，现有研究尚未解答：多种多样的内在动机如何与世界模型表示学习相互作用以支持下游控制。这限制了我们对哪些由内在动机收集的经验能够构建可复用模型的理解。为填补这些空白，我们将24种内在目标（包括涵盖好奇心、熵与胜任力等方向的成熟方法及显式变体）整合到一个统一的基于模型的强化学习框架中。我们通过在运动控制、物体操作和开放式游戏等任务中的零样本、少样本和多任务控制，来评估它们所构建的世界模型……

    arXiv:2611.13113v1 Announce Type: new  Abstract: In the absence of extrinsic rewards or task supervision, human behavior is guided by intrinsic motivation---neural signals that drive curiosity, creativity, play, and other essential facets of intelligence. Intrinsic motivation plays an important developmental role in building robust world models that enable generalization and adaptation. Yet existing studies leave open how a broad range of intrinsic motivations interacts with world-model representation learning to support downstream control. This limits our understanding of which intrinsically collected experiences build reusable models. To address these gaps, we integrate 24 intrinsic objectives, including established methods and explicit variants spanning curiosity, entropy, and competency, into a common model-based reinforcement learning framework. We evaluate the world models they build through zero-shot, few-shot, and multi-task control in locomotion, manipulation, and open-ended g
    
[^201]: 一个温度统御一切？

    One Temperature to Rule Them All?

    [https://arxiv.org/abs/2611.13077](https://arxiv.org/abs/2611.13077)

    该论文研究了 pass@K 推理中温度多样性何时优于单一温度，发现虽然更难的问题更难共享高效的采样温度，但这一趋势仅随难度呈对数增长，表明单一调优温度在多数情况下已足够有效。

    

    推理时采样是提升大语言模型性能的核心机制，其中温度参数控制着可靠性与多样性之间的权衡。尽管近期研究表明，对不同样本使用不同温度可以改进多样本推理效果，但简单的固定温度方案在实践中仍是标准做法。我们在经典的 pass@K 设定下研究这一差异，探讨温度多样性何时有用，以及单一调优温度何时已经足够。为此，我们分析了多个模型和数据集上每个问题的成功概率，追踪其随采样温度的变化规律。我们识别出这些温度曲线的若干一致特性，其中包括：越难的问题越不太可能与其他问题共享高效的采样温度，尽管这一趋势仅随难度呈对数增长。随后，我们推导出一种高效方法，用于评估理论上最优的多温度分配方案。

    arXiv:2611.13077v1 Announce Type: cross  Abstract: Inference-time sampling is a central mechanism for improving large language models, with temperature controlling the tradeoff between reliability and diversity. While recent work suggests that varying temperature across samples can improve multi-sample inference, simple fixed-temperature schemes remain the standard in practice. We investigate this discrepancy in the canonical pass@K setting, asking when temperature diversity is useful, and when a single tuned temperature suffices. To this end, we analyze per-question success probabilities across multiple models and datasets, tracking how they vary with sampling temperature. We identify several consistent properties of these profiles, including that harder questions are less likely to share efficient sampling temperatures, although this trend is only logarithmic in hardness. We then derive an efficient method for evaluating the theoretically optimal multi-temperature allocation scheme, 
    
[^202]: IC-GNN：为疾病预测建模个体化的大脑社区组织

    IC-GNN: Modeling Individualized Brain Community Organization for Disease Prediction

    [https://arxiv.org/abs/2611.13070](https://arxiv.org/abs/2611.13070)

    提出了一种以图谱为锚点的图神经网络模型IC-GNN，能直接在图谱定义的脑区上构建个体化的多尺度大脑功能社区，从而捕捉与疾病相关的个体间差异并用于疾病预测。

    

    大脑网络表现出模块化组织，其中社区结构在个体之间以及不同疾病之间存在显著差异。理解这些个体差异对于推进基础神经科学研究和精准医学都至关重要。然而，广泛使用的预定义群体水平脑图谱往往忽略了这种异质性。虽然个体化功能分区能够捕捉受试者特异的社区边界变异，但它们通常需要较长的扫描时间才能达到细粒度分辨率，并且使跨受试者的比较变得复杂。为了解决这种权衡问题，并研究个体化社区组织如何捕捉与疾病相关的个体间变化，我们提出了IC-GNN，这是一种以图谱为锚点的模型，可以直接在图谱定义的脑区之上形成独特的、个体化的功能社区。IC-GNN从静息态功能磁共振成像中推导出离散的、个体化的多尺度社区。

    arXiv:2611.13070v1 Announce Type: new  Abstract: Brain networks exhibit modular organization, where community structures vary substantially across individuals and diseases. Understanding these individual differences is crucial for advancing both basic neuroscience research and precision medicine. However, widely used predefined group-level brain atlases often overlook this heterogeneity. While individualized functional parcellations capture subject-specific variability in community boundaries, they often require lengthy scan times to achieve fine-grained resolution and complicate cross-subject comparisons. To address this trade-off and investigate how individualized community organization captures disease-associated changes across individuals, we propose IC-GNN, an atlas-anchored model that forms distinct, individualized functional communities directly over atlas-defined brain regions. IC-GNN derives discrete, individualized multiscale communities from resting-state functional magnetic
    
[^203]: INTENT：用户事件序列上的样本外、时间外与冷启动评估

    INTENT: Out-of-Sample, Out-of-Time, and Cold-Start Evaluation on User Event Sequences

    [https://arxiv.org/abs/2611.13069](https://arxiv.org/abs/2611.13069)

    提出了INTENT这一公开基准，根据用户、时期与目标项目是否在训练中出现，将用户事件序列上的下一动作预测模型划分到七种情形下评估，从而统一衡量其样本外、时间外与冷启动泛化能力。

    

    银行如今在客户事件的有序序列上训练基础模型，例如交易、应用会话和授权。部署后，这类模型需要在时间上向前预测，且预测对象往往是训练中未出现的客户和商家；然而每个模型都在各自的私有数据上评估，因而无法相互比较。针对此类序列的公共基准很少测试模型未见过的时期；推荐系统研究则通常只保留每个用户的最后几个事件作为测试。我们提出了INTENT（面向下一动作预测的交互、新颖性与迁移评估），这是一个公开基准，根据用户、时期和目标项目是否出现在训练中，将模型评估划分为七种情形进行打分。对59个公开数据集家族的审计最终保留了五个数据集；最常见的缺陷是缺少用于对事件排序的日历时间戳。由于没有任何日期落在所有五个数据集共同的时间跨度内，因此每个数据集都按照事先固定的统一规则在其各自的截止时间点进行切分。

    arXiv:2611.13069v1 Announce Type: new  Abstract: Banks now train foundation models on ordered sequences of customer events such as transactions, app sessions, and authorizations. Deployed, such a model predicts forward in time, often for customers and merchants absent from training, yet each is evaluated on private data, so no two can be compared. Public benchmarks for such sequences rarely test a period the model has not seen; recommendation studies usually hold out each user's last events instead. We present INTENT (INTEraction, Novelty, and Transfer evaluation for next-action prediction), a public benchmark that scores a model in seven cells defined by whether the user, the period, and the target item appeared in training. An audit of 59 public dataset families retains five datasets; the most common defect is the lack of a calendar timestamp that orders events. No date lies within the time span of all five datasets, so each is split at its own cutoff, chosen by one rule fixed before
    
[^204]: 用于异步WiFi-IMU室内定位的连续时间集合变换器

    Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization

    [https://arxiv.org/abs/2611.13057](https://arxiv.org/abs/2611.13057)

    提出一种连续时间集合变换器，将异步采样的WiFi与IMU观测表示为带时间戳的无序标记集合，用单一注意力块实现跨模态与时间的联合融合，并通过随机掩码训练应对缺失或过时的传感器输入，用于室内定位。

    

    室内定位通常将WiFi接收信号强度指示器（RSSI）指纹与惯性测量相结合。然而，这些传感器流以不同的速率异步采样，因此观测值很少在时间上对齐，且可能缺失或过时。大多数现有方法将数据重新采样到共享时钟上，并使用单独的分支处理每个模态。我们转而提出一种连续时间变换器，将观测值表示为一组无序的带时间戳的标记。每个标记编码其测量值、模态以及自采集以来经过的时间。单个注意力块跨模态和时间联合融合信息，既避免了重新采样，也避免了特定模态的分支，同时保持对标记顺序的不变性。在训练期间，我们随机屏蔽整个模态和单个观测值，使模型能够在缺失或过时输入的情况下运行。我们验证了模态特定的…

    arXiv:2611.13057v1 Announce Type: cross  Abstract: Indoor localization commonly combines WiFi received signal strength indicator (RSSI) fingerprints with inertial measurements. However, these sensor streams are sampled asynchronously at different rates; therefore, observations rarely align in time and may be missing or stale. Most existing methods resample the data onto a shared clock and process each modality using a separate branch. We instead present a continuous-time transformer that represents observations as an unordered set of timestamped tokens. Each token encodes its measurement, modality, and elapsed time since acquisition. A single attention block jointly fuses information across modalities and time, avoiding both resampling and modality-specific branches while remaining invariant to token order. During training, we randomly mask entire modalities and individual observations, enabling the model to operate under missing or outdated inputs. We validate the modality-specific en
    
[^205]: 超越输出一致的解释感知公平性：面向医学图像分类的联合输出-归因正则化

    Explanation-Aware Fairness Beyond Output Parity: Joint Output-Attribution Regularization for Medical Image Classification

    [https://arxiv.org/abs/2611.13052](https://arxiv.org/abs/2611.13052)

    本文提出一种联合输出-归因正则化方法，通过惩罚不同人口群体间的特征归因差异来补充输出层面的均衡几率公平约束，在医学图像分类任务上显著改善了效用与公平性的权衡表现。

    

    输出层面的公平性并不约束不同人口群体在表示空间中是否依赖相似的预测贡献。对于带有线性头的非线性编码器，我们证明了以结果为条件的输出分数差异等于各子群之间特征归因差异的带符号之和，这使得微小的预测差距可以通过抵消效应产生。我们检验了归因差异惩罚是否能补充基于输出级均衡几率的公平性代理方法。在 HAM10000 与 Harvard-GF 数据集上针对性别与种族公平性训练的 960 个 DINOv2-LoRA 模型上，在六种不同设置中（均配有配对 95% 置信区间），归因增强提升了效用-公平性权衡的超体积性能，相对增益为 1.49%-10.57%。在预定义的效用约束下，经过增强的模型族还实现了低 9.0%-38.7% 的均衡几率差异、更低的归因差异，以及更少的抵消效应。

    arXiv:2611.13052v1 Announce Type: new  Abstract: Output-level fairness does not constrain whether demographic groups rely on similar predictive contributions in representation space. For a nonlinear encoder with a linear head, we show that outcome-conditioned output score disparity equals the signed sum of feature attribution differences across subgroups by allowing small prediction gaps to arise through cancellation effect. We test whether an attribution disparity penalty complements output-level equalized odds based fairness surrogates. Across 960 DINOv2-LoRA models on HAM10000 and Harvard-GF datasets for gender and race fairness, attribution augmentation increases utility-fairness trade-off performance for hypervolume in six different settings with paired 95% confidence intervals and relative gains of 1.49-10.57%. Under a predefined utility constraint, augmented model families also achieve 9.0-38.7% lower equalized odds difference, lower attribution disparity, and reduced cancellati
    
[^206]: 多未来草拟：面向树验证式推测解码的混合块草拟器

    Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding

    [https://arxiv.org/abs/2611.13017](https://arxiv.org/abs/2611.13017)

    论文揭示了基于乘积排序的草拟树在多未来场景下会将大量概率预算浪费在目标模型从不产生的“嵌合体”前缀上（占64节点树的27-33%），并提出多未来草拟（MFD），用潜变量（K个可学习模式嵌入）替代链式法则来恢复多个未来之间的耦合，从而提升树验证式推测解码的接受效率。

    

    诸如DFlash之类的块草拟器可在一次前向传播中草拟一整块未来token，而诸如DDTree之类的草拟树则根据草拟器各位置分布的乘积对前缀进行排序并验证排名最高的前缀。在树验证机制下，期望接受的token数量等于树所覆盖的概率质量。我们证明乘积会错误地分配这一概率质量：当续写存在多个可能的未来路径时，乘积会将它们拼接成“嵌合体”（chimeras）——即由目标模型从不产生的、由不同未来拼接而成的前缀，并给予它们与真实未来同等的概率预算。除非出现平局，任何按乘积排序的树若要容纳两个在 $m$ 个位置上不一致的未来，就必然产生 $2^{m-1}-1$ 个嵌合体，无论草拟器训练得多好；在实践中，嵌合体占64节点树的27-33%。我们提出多未来草拟（MFD），它通过引入潜变量而非链式法则来恢复多个未来之间的耦合：在草拟器中设置 $K$ 个可学习的模式嵌入……

    arXiv:2611.13017v1 Announce Type: new  Abstract: Block drafters such as DFlash draft a block of future tokens in one forward pass, and draft trees such as DDTree verify the prefixes ranked highest by the product of the drafter's per-position distributions. Under tree verification, the expected number of accepted tokens is the probability mass the tree covers. We show that a product misallocates it: when the continuation can follow several futures, it splices them into chimeras, prefixes stitched together from different futures that the target never produces, and gives them as much budget as the futures themselves. Barring ties, no product-ranked tree can hold two futures that disagree in $m$ positions without $2^{m-1}-1$ chimeras, however well the drafter is trained, and in practice chimeras make up 27-33% of a 64-node tree. We propose Multi-Future Drafting (MFD), which restores the coupling with a latent variable instead of the chain rule: $K$ learned mode embeddings at the drafter's 
    
[^207]: 基于相互正交性的置换式预训练模型合并

    Permutation-Based Pre-trained Model Merging via Reciprocal Orthogonality

    [https://arxiv.org/abs/2611.13016](https://arxiv.org/abs/2611.13016)

    提出通过搜索促进相互正交性（RO）的置换来合并预训练模型，无需大幅加宽网络即可抑制模型间的干扰，使两个模型的计算能够在合并网络中共存。

    

    Ito等人（2026）表明，足够宽的独立训练神经网络即使不进行置换，也可以通过softmax温度校准表现出线性模式连通性（LMC）。LMC意味着存在一条低损失的线性路径连接模型的参数。这种行为可以由逐层指数加权连通性（LEWC）来解释，即每个合并层输出原始模型对应表示的指数加权和，从而产生类似集成的预测。然而，先前的工作是在大幅加宽网络的条件下观察到LEWC的，例如MLP每个隐藏层需要8192个单元，VGG-11需要16倍宽度，这引发了这样的问题：如此大的宽度是否必要。因此，我们搜索能够促进相互正交性（RO）的置换，该性质可以抑制每个模型的权重与另一个模型表示之间的干扰。这使得两个模型的计算能够在合并后的网络中共存。

    arXiv:2611.13016v1 Announce Type: new  Abstract: Ito et al. (2026) showed that sufficiently wide, independently trained neural networks can exhibit linear mode connectivity (LMC) with softmax temperature calibration, even without permutation. LMC means that a low-loss linear path connects the models' parameters. This behavior is explained by layerwise exponentially weighted connectivity (LEWC), whereby each merged layer outputs an exponentially weighted sum of the original models' corresponding representations, yielding ensemble-like predictions. However, prior work observed LEWC with substantial widening, such as 8192 units per hidden layer for MLPs and $16\times$ width for VGG-11, raising the question of whether such widths are necessary. We therefore search for permutations that encourage reciprocal orthogonality (RO), which suppresses interference between each model's weights and the other model's representations. This allows both models' computations to coexist in the merged netwo
    
[^208]: FaST：面向离线强化学习的转移因子化合成

    FaST: Factored Synthesis of Transitions for Offline Reinforcement Learning

    [https://arxiv.org/abs/2611.13013](https://arxiv.org/abs/2611.13013)

    FaST通过链式法则将强化学习转移分解为状态、动作、动力学、奖励和终止五个阶段分别建模，在匹配扩散基线性能的同时将合成速度提升90倍以上，并支持针对固定状态进行高优势动作的定向合成。

    

    合成数据增强有望使离线强化学习突破固定记录数据集的限制而实现规模扩展，但主流的基于扩散模型的生成器将转移元组（状态、动作、下一状态、奖励、终止标志）建模为一个不可分割的整体：没有任何单一因子能够被原生地单独重新生成。我们提出了FaST（转移的因子化合成），它通过链式法则将转移分解为状态、动作、动力学、奖励和终止五个阶段，并为每个阶段配备与其统计特性相匹配的估计器。在十二个D4RL运动控制和迷宫基准测试中，基于FaST生成的转移训练的TD3+BC和IQL智能体达到了SynthER扩散基线的性能水平，同时生成转移的速度快90倍以上。显式的“给定状态下的动作”条件分布还可以针对固定状态进行重新查询，从而实现有针对性的动作合成：为每个生成的状态抽取N个候选动作，并保留在基于相同数据拟合的价值函数下具有最高优势值的动作。

    arXiv:2611.13013v1 Announce Type: new  Abstract: Synthetic data augmentation promises to scale offline reinforcement learning beyond fixed logged datasets, but leading diffusion-based generators model the transition tuple (state, action, next state, reward, terminal) as a monolith: no single factor can be natively redrawn in isolation. We present FaST (Factored Synthesis of Transitions), which factors transitions via the chain rule into state, action, dynamics, reward, and terminal stages, fitting each with an estimator matched to its statistical properties. Across twelve D4RL locomotion and maze benchmarks, TD3+BC and IQL agents trained on FaST transitions match the SynthER diffusion baseline while generating transitions over 90x faster. The explicit action-given-state conditional can also be re-queried for a fixed state, enabling targeted action synthesis: drawing N candidate actions per generated state and keeping the highest-advantage one under a value function fit to the same data
    
[^209]: 程序的形态：用于轨迹到程序归纳的路径签名

    The Shape of a Program: Path Signatures for Trace-to-Program Induction

    [https://arxiv.org/abs/2611.13005](https://arxiv.org/abs/2611.13005)

    本文将粗路径理论引入程序归纳，用路径签名变换编码执行轨迹，在 Karel、List 和 RASP 三个领域几乎全面超越 LSTM、Transformer 和傅里叶基线（尤其在数据稀缺时优势最大），并利用流程天然的收缩性构建无需外部预言机的自举方法来揭示监督数据的冗余。

    

    从执行轨迹中合成程序是算法可解释性与复杂系统逆向工程中的一项根本性挑战。我们将粗路径理论的视角引入程序归纳，把执行轨迹视为高维粗路径，并用路径签名变换对其进行编码——这是来自随机分析的一种工具，其期望可被证明能够刻画程序所诱导的轨迹分布。在 Karel、List 和 RASP 三个程序归纳领域上，我们的对数签名轨迹编码器几乎在所有设置下都优于 LSTM、Transformer 和傅里叶基线，且在训练数据稀缺时优势最为显著。我们进一步观察到，轨迹到程序的流程天然具有收缩性：它往往能生成比真实程序更短的合法程序。我们将这一特性转化为一种自举方法，无需任何外部预言机即可揭示真实监督信号中存在的冗余。

    arXiv:2611.13005v1 Announce Type: cross  Abstract: Synthesizing programs from execution traces is a fundamental challenge in algorithmic interpretability and the reverse engineering of complex systems. We bring a rough-path-theoretic perspective to program induction, treating execution traces as high-dimensional rough paths and encoding them with the path signature transform, a tool from stochastic analysis whose expectation provably characterizes the distribution of traces a program induces. On three program-induction domains, Karel, List, and RASP, our log-signature trace encoder outperforms LSTM, Transformer, and Fourier baselines in almost every setting, with its largest margins when training data is scarce. We further observe that the trace-to-program pipeline is naturally contractive: it often produces valid programs that are shorter than the ground truth. We turn this into a bootstrapping procedure that exposes redundancy in the ground-truth supervision without any external orac
    
[^210]: 面向冻结强化学习智能体的分支与提交方法：何时以及在哪里投入测试时计算

    Branch-and-Commit for Frozen RL Agents: When and Where to Spend Test-Time Compute

    [https://arxiv.org/abs/2611.13003](https://arxiv.org/abs/2611.13003)

    该论文提出针对冻结RL智能体的“分支与提交”方法，发现分支收益取决于候选回报的离散度——低离散度时无收益，高离散度时可提升回报12.8%，且收益主要来自在不同状态间选择而非选择更优动作。

    

    强化学习智能体在部署时，其策略通常无法重新训练或修改。在状态可以被复制和恢复的数字环境中，操作者还有另一种选择：从当前状态采样多个候选延续，对它们进行评估，然后从所选分支的端点继续执行。我们将这一过程称为“分支与提交”。我们研究了它何时能改善回报、改善的来源是什么，以及其计算预算应该花在哪里。通过在四个环境、两个冻结策略、五个延续验证器和一个无模型选择规则上进行实验，我们发现分支带来的收益取决于离散度：即最佳候选回报与平均候选回报之间的差异。在低离散度的设置中，分支不会带来任何收益；而在我们的主要Craftax-Medium设置中，回报提升了12.8%。这一收益的大部分并非来自选择更好的动作，而是来自在不同状态之间进行选择。

    arXiv:2611.13003v1 Announce Type: new  Abstract: Reinforcement learning agents are often deployed with policies that cannot be retrained or modified. In digital environments whose states can be copied and resumed, an operator has another option: sample several continuations from the current state, evaluate them, and continue from the endpoint of the selected branch. We call this procedure branch-and-commit. We study when it improves return, what produces the improvement, and where its computational budget should be spent. Across four environments, two frozen policies, five continuation verifiers, and a model-free selection rule, we find that the benefit of branching depends on dispersion: the difference between the best candidate return and the mean candidate return. Branching provides no benefit in low-dispersion settings, while increasing return by 12.8% in our main Craftax-Medium setting. Most of this gain comes not from selecting better actions, but from selecting among different s
    
[^211]: 一表多查询：跨生成、插补与预测的通用表格模型基准测试

    One Table, Many Queries: Benchmarking Generalist Tabular Models Across Generation, Imputation, and Prediction

    [https://arxiv.org/abs/2611.13001](https://arxiv.org/abs/2611.13001)

    该论文提出One Table Benchmark基准，通过56个数据集、46种方法和20个能力维度的统一评估，首次系统揭示了表格数据的生成、插补与预测三类任务之间的能力关联和训练目标的相互作用，并证明单一通用模型有望同时胜任这三类任务。

    

    生成、插补和预测查询的是同一表格分布的不同方面，但它们通常被分开评估。这使得这些任务之间共享哪些能力、其训练目标如何相互作用，以及单一模型能否同时出色完成这三项任务都尚不清楚。我们提出了一表基准，涵盖56个数据集、46个已注册方法和20个能力维度，并在统一的数据划分和明确的任务契约下进行评估。我们在对齐的支持集上比较各任务的排名，在五个数据集上以三个随机种子训练五个骨干网络的所有七种非空目标组合，并将固定模型的能力概况与专家模型组合进行对比。排名比较通过数值恢复和依赖性保持将插补与预测联系起来，并通过预测效用将生成与前两者联系起来；而分布保真度则遵循不同的、有时甚至相反的排名。目标干预实验显示出……方面的最大综合收益（摘要在此处被截断）

    arXiv:2611.13001v1 Announce Type: new  Abstract: Generation, imputation, and prediction query different aspects of the same tabular distribution, but are usually evaluated separately. This leaves unclear which capabilities they share, how their training objectives interact, and whether one model can perform all three well. We introduce the One Table Benchmark (\ours{}), spanning 56 datasets, 46 registered methods, and 20 capability axes under aligned data splits and explicit task contracts. We compare task ranks on aligned support, train five backbones under all seven non-empty objective combinations on five datasets with three seeds, and contrast fixed-model profiles with specialist portfolios. Rank comparisons connect imputation and prediction through numerical recovery and dependence preservation, and link generation to both through predictive utility; distributional fidelity follows different, sometimes opposing, rankings. Objective interventions show the largest pooled gains from 
    
[^212]: 从线性探针到贝叶斯流形：大语言模型中算术运算的几何结构

    Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Large Language Models

    [https://arxiv.org/abs/2611.12998](https://arxiv.org/abs/2611.12998)

    该论文提出一个四阶段贝叶斯分析流程，将探针准确率分解为可解码性、形状、归属权和因果使用四个独立命题并分别检验，从而更严谨地判定大语言模型激活中的几何结构（如编码算术运算的螺旋或圆形）是否真正被模型用于计算。

    

    线性探针是从大语言模型（LLM）的激活值中研究几何结构（如螺旋线和圆）的标准方法，此类结构常被当作模型利用其进行计算的依据。然而，仅凭探针准确率无法证实这一点，因为它混淆了四个相互独立的命题：（1）概念可以从激活值中被解码；（2）这些概念构成特定形状；（3）该形状属于被探测的概念，而非模型同时表示的另一个相关概念；（4）模型的输出依赖于该形状。我们将这些命题分别称为可解码性、形状、归属权和因果使用，并通过一个四阶段的贝叶斯流程对它们分别进行检验。第二阶段比较候选形状，仅当其证据超过预定义阈值时才选出最佳形状。第三阶段移除一组预先指定的相关概念，检验该形状究竟是被探测概念所拥有，还是从相关概念继承而来。

    arXiv:2611.12998v1 Announce Type: cross  Abstract: Linear probes are a standard method to study geometric structures, such as helices and circles, from the activations of large language models (LLMs), and such structure is often taken as evidence that the model computes with it. Probe accuracy alone cannot establish this, because it conflates four separate claims. These are (1) that the concept is decodable from the activations, (2) that they form a specific shape, (3) that the shape belongs to the probed concept rather than to a related concept the model also represents, and (4) that the output of the model depends on the shape. We call these claims decodability, shape, ownership, and causal use, and we test them separately with a four-stage Bayesian pipeline. Stage 2 compares candidate shapes and selects the best one only when its evidence passes a predefined threshold. Stage 3 removes a set of related concepts, specified in advance, and tests whether the shape is owned or inherited.
    
[^213]: ZeroPoint：轻量级基于能量的决策模型

    ZeroPoint: Lightweight Energy-Based Decision Model

    [https://arxiv.org/abs/2611.12992](https://arxiv.org/abs/2611.12992)

    本文提出ZeroPoint，通过引入一个对任意输入能量恒为零的“空答案”固定状态，为基于能量的决策模型提供显式的能量参考水平，使模型能够判断并拒绝“没有任何可接受选项”的情形。

    

    基于能量的模型（EBM）为观测变量与预测变量的每一个联合取值分配一个标量能量。在决策模型中，预测是从每个输入所附带的有限选项列表中选出的，因此推理可简化为比较各选项的能量，而训练则是降低正确选项相对于其他选项的能量。一个完整的决策还必须考虑没有任何选项可以接受的情况。然而，标准EBM学到的能量只定义了相对兼容性：给所有能量加上同一个依赖于输入的量，预测结果、吉布斯分布以及常用的训练损失都不会改变。因此，模型缺少一个可供判断任何选项是否合适的、学习得到的参考水平。我们提出ZeroPoint，这是一种将“空答案”作为模型自身状态的EBM，该状态对每个输入的能量均为零。这一固定状态为能量提供了显式的参考。

    arXiv:2611.12992v1 Announce Type: cross  Abstract: Energy-based models (EBMs) assign a scalar energy to every joint value of the observed and predicted variables. In decision models, the prediction is chosen from a finite list of options provided with each input. Inference then reduces to comparing the energies of the options, and training lowers the energy of the correct option relative to the others. A complete decision must also account for the possibility that no option is acceptable. However, the energies learned by a standard EBM define only relative compatibility: adding the same input-dependent quantity to all energies leaves the predictions, the Gibbs distribution, and the usual training losses unchanged. The model therefore has no learned reference level against which to decide whether any option fits. We propose ZeroPoint, an EBM that makes the empty answer a state of the model with energy zero for every input. This fixed state provides an explicit reference for the energies
    
[^214]: 多模态图学习：模型、基准与前沿综述

    Multimodal Graph Learning: A Survey of Models, Benchmarks, and Frontiers

    [https://arxiv.org/abs/2611.12990](https://arxiv.org/abs/2611.12990)

    本综述针对多模态属性图学习，提出了统一的消息传递建模框架，将现有方法按五个集成层次（特征、传播、拓扑、学习和基础模型）分类，并系统梳理与评估了相关基准与前沿方向。

    

    多模态属性图（MAGs）中的节点携带文本、图像和其他模态，而边编码实体之间的关系，这类图广泛应用于电子商务、社交媒体、内容平台和创意应用中。MAG上的学习沿着两条互补的路线发展：以图为中心的方法重新设计模态在消息传递中如何融合、传播和连接，而以基础模型为中心的方法则预训练可迁移的多模态图编码器，或将图上下文引入多模态大语言模型以进行预测和生成。与此同时，出现了多个采用不同数据集和评估协议的基准。本综述以算法为中心并以评估为基础的视角审视该领域，并做出三项贡献。建模方面：我们引入统一的消息传递公式，并将方法组织为五个集成层次（特征、传播、拓扑、学习和基础模型）。

    arXiv:2611.12990v1 Announce Type: new  Abstract: Multimodal attributed graphs (MAGs), in which nodes carry text, images, and other modalities while edges encode relations among entities, underlie e-commerce, social media, content platforms, and creative applications. Learning on MAGs has developed along two complementary lines: graph-centric methods redesign how modalities are fused, propagated, and connected within message passing, while foundation-model-centric methods pretrain transferable multimodal graph encoders or bring graph context into multimodal large language models for prediction and generation. In parallel, several benchmarks have appeared with different datasets and evaluation protocols. This survey takes an algorithm-centric and evaluation-grounded view of the field and makes three contributions. Modeling: we introduce a unified message-passing formulation and organize methods into five integration levels (feature, propagation, topology, learning, and foundation model),
    
[^215]: 第三只眼：超越递归自我改进中的即时收益

    Third Eye: Beyond Immediate Gains in Recursive Self-Improvement

    [https://arxiv.org/abs/2611.12988](https://arxiv.org/abs/2611.12988)

    论文通过大规模受控实验发现，递归自我改进中看似有用的自我更新在经过第二轮更新后效用会进一步降低（甚至候选平均即时效用已为负值），因此对自我改进的评估必须超越第一步的即时收益。

    

    一个有用的自我更新在模型再次更新之后是否仍然有用？Third Eye 将即时性能与进一步原生更新的结果分离开来。我们在数学和代码任务上评估了五个3-8B参数的基座模型：共64次全新运行，每次运行产生三个候选，每个候选进行三次后续更新，且候选选择与最终评分使用互不相交的题目。在均衡的Qwen3-4B/Llama-3.2-3B对比中，经即时选择得到的效用接近于零，但后续更新使其降低了1.78个百分点（同时95%置信区间[0.68, 2.88]）。候选平均的即时效用本身已是负值：选择机制在一定程度上解释了表面上中性的第一步。后续更新同样降低了SmolLM3-3B和Llama-3.1-8B的效用；Qwen3-8B的结果仍无定论。目标损失和分布外损失主导了保留效果的变化。匹配配方和接收者-捐赠者对照实验既未找到可靠的缓解手段，也未确定单一机制。这些结果支持对……进行评估（摘要在此处截断）

    arXiv:2611.12988v1 Announce Type: cross  Abstract: Does a useful self-update remain useful after the model updates again? Third Eye separates immediate performance from the outcome of a further native update. We evaluate five 3-8B backbones on math and code: 64 fresh runs, three candidates per run, and three continuations per candidate. Selection and final scoring use disjoint items. In the balanced Qwen3-4B/Llama-3.2-3B comparison, immediate-selected utility is near zero, but continuation reduces it by 1.78 percentage points (simultaneous 95% interval [0.68, 2.88]). Candidate-averaged immediate utility is already negative: selection partly accounts for the apparently neutral first step. Continuation also lowers utility for SmolLM3-3B and Llama-3.1-8B; Qwen3-8B remains inconclusive. Target and out-of-distribution losses dominate retention changes. Matched recipe and recipient-donor controls identify neither a reliable mitigation nor a single mechanism. The results support evaluating pe
    
[^216]: MADSNet：一种用于脉冲相机图像重建的多尺度轴向注意力密集脉冲网络

    MADSNet: A Multi-Scale Axial-Attentive Dense Spiking Network for Spike Camera Image Reconstruction

    [https://arxiv.org/abs/2611.12980](https://arxiv.org/abs/2611.12980)

    提出了一种多尺度轴向注意力密集脉冲网络MADSNet，通过分层脉冲编码器-解码器结构实现高质量脉冲相机图像重建，在SRED数据集上显著超越现有SNN重建方法。

    

    脉冲相机将动态场景编码为高时间分辨率的二值脉冲流，以实现高效的高速视觉处理。尽管脉冲神经网络（SNN）能够以稀疏且仅累加的计算方式处理这类信号，但现有的基于SNN的重建方法通常网络结构较浅，在重建保真度上仍落后于基于人工神经网络（ANN）的模型。本文提出了MADSNet，一种用于脉冲相机图像重建的多尺度轴向注意力密集脉冲网络。MADSNet采用分层脉冲编码器-解码器结构，其中多尺度脉冲干模块提取早期特征，轴向脉冲注意力捕获长程依赖关系，密集脉冲瓶颈层增强特征复用，脉冲到连续的解码器则从脉冲域表示中重建连续的强度图像。在灰度和彩色脉冲重建实验中，MADSNet在SRED数据集上比SSIR高出4.3 dB PSNR，比Spk2ImgNet高出1.94 dB。

    arXiv:2611.12980v1 Announce Type: cross  Abstract: Spike cameras encode dynamic scenes into high-temporal-resolution binary spike streams for efficient high-speed vision. Although spiking neural networks (SNNs) can process such signals with sparse and accumulate-only computation, existing SNN-based reconstruction methods are typically shallow and still lag behind ANN-based models in reconstruction fidelity. This paper proposes MADSNet, a Multi-scale Axial-attentive Dense Spiking Network for spike camera image reconstruction. MADSNet adopts a hierarchical spiking encoder--decoder, where a multi-scale spike stem extracts early features, axial spike attention captures long-range dependencies, a dense spiking bottleneck enhances feature reuse, and a spike-to-continuous decoder reconstructs continuous intensity images from spike-domain representations. Experiments on gray-scale and color spike reconstruction show that MADSNet outperforms SSIR by 4.3 dB PSNR and Spk2ImgNet by 1.94 dB on SRED
    
[^217]: 分区、精炼与检测：在开放词汇目标检测中保留新类别容量

    Partition, Refine, and Detect: Preserving Novel Category Capacity in Open-Vocabulary Object Detection

    [https://arxiv.org/abs/2611.12976](https://arxiv.org/abs/2611.12976)

    提出PR-OVD“分区-精炼”框架，通过显式为新类别保留嵌入容量并增强类别分离，解决了开放词汇目标检测中由基类主导表示导致的语义坍塌与置信度失准问题。

    

    开放词汇目标检测（OVD）要求在基类上训练的模型在推理时能够同时识别基类和新类别。典型方法依赖于CLIP风格的视觉-语言嵌入，然而学习到的嵌入空间往往偏向基类，为新概念保留的有效容量有限。这种由基类主导的表示可能会压缩新类别的语义，导致语义坍塌以及推理时置信度校准失准。为解决这一问题，我们提出了PR-OVD，这是一个“分区-精炼”框架，它显式地为新类别保留容量并增强类别之间的分离性，同时能够无缝集成到不同的检测器架构中。具体而言，PR-OVD首先进行粗分区，在视觉和文本两种模态中为基类、新类别以及显式背景分区构建全局划分，防止与目标无关的嵌入泄露到类别特定的……

    arXiv:2611.12976v1 Announce Type: cross  Abstract: Open-vocabulary object detection (OVD) expects the model trained on base classes to recognize both base and novel classes at inference. Typical methods rely on CLIP-style vision-language embeddings, yet the learned space is often biased toward base categories, with limited effective capacity preserved for novel concepts. Such a base-dominated representation can compress novel semantics, leading to semantic collapse and mis-calibrated confidence at inference. To address this issue, we present PR-OVD, a partition-and-refine framework that explicitly reserves capacity for novel categories and strengthens category separation, while being seamlessly integrated into different detector architectures. Specifically, PR-OVD first performs coarse partitioning to build global divisions for base, novel, and an explicit background partition in both visual and textual modalities, preventing object-irrelevant embeddings from leaking into category-spec
    
[^218]: 基于多模态提示的属性-证据学习用于广义类别发现

    Attribute-Evidence Learning with Multimodal Prompt for Generalized Category Discovery

    [https://arxiv.org/abs/2611.12975](https://arxiv.org/abs/2611.12975)

    提出 AptM-GCD 多模态框架，通过属性-证据学习机制将属性作为局部视觉证据而非全局类别描述符，并结合多模态提示推理，避免了文本与全局图像表示纠缠导致的误导性预测，从而更准确地发现新类别。

    

    广义类别发现旨在通过利用来自旧类别的有限标注样本，从未标注数据中发现新类别。现有仅依赖视觉线索的方法往往忽视了人类感知的多模态特性，这促使近期的方法引入诸如类别标签和属性等文本信息。然而，这些方法通常将全局标签和局部属性纠缠在共享的全局图像表示中，导致属性变得不可靠，并可能提供误导性线索，从而驱使模型做出错误的预测。为此，我们提出了 AptM-GCD，一种新颖的多模态框架，它通过属性证据学习和多模态提示推理来利用恰当的文本引导。具体而言，我们设计了一种属性-证据学习机制，将属性视为局部化的视觉证据而非全局类别描述符。通过交互……（摘要原文在此处截断）

    arXiv:2611.12975v1 Announce Type: cross  Abstract: Generalized category discovery (GCD) aims to discover novel categories from unlabeled data by leveraging limited labeled samples from old classes. Existing methods that rely solely on visual cues often neglect the multimodal nature of human perception, prompting recent approaches to incorporate textual information such as category tags and attributes. However, these approaches typically entangle global tags and local attributes within a shared global image representation, rendering attributes unreliable and potentially providing misleading cues that can drive the model toward incorrect predictions. To this end, we propose AptM-GCD, a novel multimodal framework that exploits apt textual guidance through attribute evidence learning and multimodal prompt reasoning. Specifically, we design an Attribute-Evidence Learning (AEL) mechanism that treats attributes as localized visual evidence rather than global category descriptors. By interacti
    
[^219]: StatPrompt-LLM：一种面向基础模型风格时间序列预测的轻量级统计增强方法

    StatPrompt-LLM: A Lightweight Statistical Refinement for Foundation-Style Time-Series Forecasting

    [https://arxiv.org/abs/2611.12974](https://arxiv.org/abs/2611.12974)

    提出 StatPrompt-LLM，通过在 Time-LLM 的提示中融入数据集级统计信息并增加一个零初始化自回归线性旁路，以轻量级方式增强基础模型风格的时间序列预测能力。

    

    随着将大型预训练语言架构适配到序列数据上的基础模型风格方法的出现，时间序列预测近期经历了一次范式转变。然而，如何有效地向这些模型呈现基本的时序结构——例如趋势、尺度和时间依赖性——仍然是一个尚未解决的挑战，尤其是在可用观测数据有限的情况下。在本文中，我们对不同代际的预测架构开展了一项探索性实证研究，并提出了 StatPrompt-LLM，即对 Time-LLM 的一种轻量级统计增强。StatPrompt-LLM 通过数据集级别的统计信息增强了现有的 Time-LLM 提示机制，并引入了一个零初始化的自回归旁路，作为补充性的线性预测通路。我们在电力和气象时间序列基准数据集（包括 ETTh1、ETTh2 和 Weather）上，于全前缀和缩减前缀两种设置下对所提出的框架进行了评估。

    arXiv:2611.12974v1 Announce Type: new  Abstract: Time-series forecasting has recently seen a paradigm shift with the introduction of foundation-style models that adapt large pretrained language architectures to sequential data. However, effectively exposing fundamental temporal structures, such as trend, scale, and temporal dependence, to these models remains an open challenge, particularly when the available observations are limited. In this paper, we conduct an exploratory empirical study across different generations of forecasting architectures and propose StatPrompt-LLM, a lightweight statistical refinement of Time-LLM. StatPrompt-LLM augments the existing Time-LLM prompting mechanism with dataset-level statistical information and introduces a zero-initialised autoregressive bypass as a complementary linear forecasting pathway. We evaluate our framework on electricity and meteorological time-series benchmarks, including ETTh1, ETTh2, and Weather, under full-prefix and reduced-prefi
    
[^220]: ADIndex：基于组级蒸馏的层级自适应稀疏注意力索引

    ADIndex: Layer-Adaptive Sparse Attention Indexing via Group-Level Distillation

    [https://arxiv.org/abs/2611.12956](https://arxiv.org/abs/2611.12956)

    ADIndex通过逐层校准与组级蒸馏训练层级自适应的稀疏注意力索引器，并辅以低延迟推理策略，大幅降低长上下文推理中索引环节的延迟开销。

    

    DeepSeek稀疏注意力（DSA）通过token级选择降低了注意力的计算成本。然而，其索引器仍需为每个查询对整个因果前缀进行评分，使索引成为长上下文推理中延迟的主要来源。值得注意的是，在长上下文长度下，仅索引器一项就可能占总预填充GPU时间的50%以上。为解决这一瓶颈，我们提出了ADIndex，一种面向token级稀疏注意力的自适应、蒸馏驱动的索引方法。ADIndex集成了三个关键组件：(i) 逐层校准策略，可自适应地适应跨查询检索重叠，其动机源自我们观察到的显著的层间异质性；(ii) 组级蒸馏方案，利用这种异质性高效训练自适应的逐层组索引器，从单个查询中捕捉关键的检索信号；(iii) 低延迟推理策略，用于降低……（原文摘要在此处截断）

    arXiv:2611.12956v1 Announce Type: cross  Abstract: DeepSeek Sparse Attention (DSA) reduces the computational cost of attention through token-level selection. However, its indexer still scores the entire causal prefix for each query, making indexing a dominant source of latency in long-context inference. Notably, the indexer alone can account for more than 50% of total prefill GPU time at long context lengths. To address this bottleneck, we propose ADIndex, an Adaptive and Distillation-driven Indexing method for token-level sparse attention. ADIndex integrates three key components: (i) a layer-wise calibration strategy that adaptively accommodates cross-query retrieval overlap, motivated by our observation of pronounced inter-layer heterogeneity; (ii) a group-level distillation scheme that leverages this heterogeneity to efficiently train adaptive layer-wise group indexers, capturing salient retrieval signals from individual queries; and (iii) a low-latency inference strategy that reduc
    
[^221]: 预测性惊奇作为可解释时间序列的自锚定概念瓶颈

    Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series

    [https://arxiv.org/abs/2611.12955](https://arxiv.org/abs/2611.12955)

    ConceptTime 利用冻结概率预测器产生的“预测性惊奇”这一单一自监督信号，同时完成时间序列分段定位与概念表征，打破了边界与模式相互依赖的循环，从而在时间序列分类中同时实现准确性、可解释性和证据不足时的弃权能力。

    

    部署在风险敏感领域的时间序列分类器必须做到：(1) 准确，(2) 可解释，(3) 在证据不足时能够弃权（拒绝预测）。现有方法无法同时满足这三点，且这一障碍是结构性的，而非操作层面的。若不知道分段边界，就无法定义有意义的时间模式；而若不知道期待何种模式，又无法划分有意义的边界。现有方法通过使用固定窗口或专家标注的词汇表来打破这种循环，从而至少牺牲了上述三个属性中的一个。我们提出 ConceptTime，它通过单一的自监督信号闭环解决这一问题。一个冻结的概率预测器在序列的每个时间步预测下一个观测值的分布。我们将其预测与实际信号之间的不匹配称为预测性惊奇。该信号既能定位分段边界，又能刻画由此产生的每个概念……

    arXiv:2611.12955v1 Announce Type: cross  Abstract: Time-series classifiers deployed in risk-sensitive domains must be (1) accurate, (2) interpretable, and (3) able to abstain when evidence is weak. Existing methods do not deliver all 3 at once, and the obstacle is structural, rather than logistical. Meaningful temporal patterns cannot be defined without knowing segment boundaries, yet meaningful boundaries cannot be placed without knowing which patterns to expect. Existing approaches break this circularity by using fixed windows or expert-annotated vocabularies, thereby sacrificing at least one of the 3 properties. We introduce ConceptTime, which closes the loop through a single self-supervised signal. A frozen probabilistic forecaster predicts a distribution over the next observation at every timestep of the sequence. We call the mismatch between its prediction and the realized signal the predictive surprise. This signal both locates segment boundaries and characterizes each resulting
    
[^222]: 表格基础模型与调优梯度提升树在电信客户流失预测中的对比：基于利润与校准感知的评估

    Tabular Foundation Models vs Tuned Gradient Boosting for Telecom Churn: A Profit- and Calibration-Aware Evaluation

    [https://arxiv.org/abs/2611.12942](https://arxiv.org/abs/2611.12942)

    该研究以期望最大利润和概率校准为评判标准，将表格基础模型与经Optuna调优的梯度提升树在电信客户流失预测上进行严格重新评估，发现TabPFN-3.5在部分数据集上能显著带来更高利润，修正了以往仅用排序指标、与未调优基线比较所得出的结论。

    

    表格基础模型（TFM）能够在单次前向传播中基于有标注的上下文完成预测。近期一项基准测试报告称，在客户流失预测任务上，表格基础模型决定性地优于树集成模型，但该测试仅将其与未调优的梯度提升树（GBDT）进行比较，且只采用排序类指标评分，而留存营销活动的实际评判标准是金钱收益。我们在五个公开的电信客户流失数据集上、通过10个随机种子划分，将TabPFN-2.5、TabPFN-3.5、TabPFN-3.5-Fast和TabICL v2与经Optuna调优的GBDT进行了重新评估。我们区分了三个问题：模型是否能很好地对流失客户进行排序、该排序是否能带来更高的期望最大利润（EMP）、以及其概率输出能否支撑阈值策略。通过划分感知的检验方法并对20项预先声明的比较进行Holm校正，TabPFN-3.5在Cell2Cell（1.9%）和Iranian（7.6%）数据集上获得的EMP显著高于最优的调优GBDT，且这两项优势在与三个调优GBDT分别单独比较时依然成立。在Orange i……（摘要在此处截断）

    arXiv:2611.12942v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) predict from a labelled context in a single forward pass. A recent benchmark reported that they decisively outperform tree ensembles for customer churn, but it compared them with untuned gradient-boosted trees (GBDTs) and scored only ranking metrics, while a retention campaign is judged in money. We re-evaluate TabPFN-2.5, TabPFN-3.5, TabPFN-3.5-Fast and TabICL v2 against Optuna-tuned GBDTs on five public telecom churn datasets over 10 seeded splits. We separate three questions: whether a model ranks churners well, whether that ranking earns more expected maximum profit (EMP), and whether its probabilities can drive a threshold policy. With a split-aware test and Holm correction over 20 pre-declared comparisons, TabPFN-3.5 earns significantly more EMP than the best tuned GBDT on Cell2Cell (1.9%) and Iranian (7.6%), and these two gains also hold against each of the three tuned GBDTs separately. On Orange i
    
[^223]: 基于强化学习的癌症化疗治疗控制

    Reinforcement Learning based control of Cancer Chemotherapy treatments

    [https://arxiv.org/abs/2611.12940](https://arxiv.org/abs/2611.12940)

    该论文系统梳理了强化学习技术在癌症化疗治疗控制中的应用现状，重点研究了用于生成最优动态治疗方案的最新强化学习方法。

    

    近年来，强化学习取得了巨大的成功。它从Atari、围棋、宝可梦和国际象棋等传统复杂游戏扩展了应用范围，引起了研究人员对建模现实生活场景的关注。个性化医疗和动态治疗方案正是当前研究人员关注的两个初步研究领域。从观察性真实临床数据中估计最优动态治疗方案正日益受到研究人员青睐，这也拓宽了其向更先进的强化学习算法版本发展的空间。本研究是一项专注于癌症化疗治疗背景下强化学习技术的研究，本论文主要通过强化学习技术在生成最优治疗方案方面的最新研究进展来呈现该研究的成果。

    arXiv:2611.12940v1 Announce Type: new  Abstract: Reinforcement Learning has achieved tremendous success in recent years. Extending its scope from the traditional complex games such as Atari, Go, Pokeman, and Chess, reinforcement learning has caught researchers' attention on modelling real-life scenarios. Personalized medicine and the dynamic treatment regimen are two such preliminary research fields; researchers are focusing their mindsets currently. Estimating the optimal dynamic treatment regimens from the observational real clinical data is being popularized among the researchers, widening its scope towards more advanced versions of reinforcement learning algorithms.   This research is about a research study that focused on reinforcement learning techniques that have been approached in the context of cancer chemotherapy treatments. Primarily this thesis addresses the result of the study by means of the state-of-the-art on reinforcement learning techniques used in generating the opti
    
[^224]: 在预填充阶段训练，在解码阶段监控：使用稀疏自编码器进行主题监控

    Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring

    [https://arxiv.org/abs/2611.12935](https://arxiv.org/abs/2611.12935)

    提出了TopicScope框架，利用稀疏自编码器（SAE）特征训练因果主题分类器，在预填充阶段提取特征训练后可泛化到解码阶段，实现对生成过程中每个token的实时多主题监控且标签可中途变化。

    

    主题监控是在模型生成响应期间为响应中的每个token标记主题的任务。迄今为止，在生成时运行的基于激活的监控器主要用于检测有害内容，且仅限于两三个类别。我们提出了TopicScope，这是一个在稀疏自编码器（SAE）特征上训练因果主题分类器的框架，它能为每个token分配主题标签，使标签能够在响应中途发生变化。我们在涵盖七个主题的合成多主题对话上训练了三种检测器架构。在Gemma-3 1B和4B模型上，使用从预填充（prefill）隐藏状态中提取的特征训练的监控器能够泛化到解码阶段。我们还发现这些分类器对扰动具有鲁棒性：当插入离题的旁白、删除主题或重新排列章节时，macro-F1最多损失5分。为理解SAE的贡献而进行的消融实验表明，达到基线的分类器……（摘要原文在此处被截断）

    arXiv:2611.12935v1 Announce Type: new  Abstract: Topic monitoring is the task of labelling the topic of each token in a model's response during generation. Activation-based monitors that run at generation time have so far been trained to detect harmful content and are limited to two or three classes. We introduce TopicScope, a framework for training causal topic classifiers on SAE features that assign a topic to every token, so that the label can change partway through a response. We train three detector architectures on synthetic multi-topic conversations covering seven topics. Across Gemma-3 1B and 4B, monitors trained on features extracted from prefill hidden states were able to generalise to decoding. We also found that the classifiers are robust to perturbations, losing at most 5 macro-F1 points when off-topic asides are inserted, topics are removed or sections are reordered. Ablations conducted to understand the contribution of the SAE revealed that classifiers reaching baseline 
    
[^225]: 面向全六面体网格生成的拓扑感知强化学习CAD分解方法

    Topology-Aware Reinforcement Learning for CAD Decomposition toward All-Hexahedral Meshing

    [https://arxiv.org/abs/2611.12930](https://arxiv.org/abs/2611.12930)

    提出DecompRL——一个基于深度强化学习的CAD模型分解框架，通过拓扑感知的内部分割操作突破了传统仅限表面分割的动作空间限制，能够为复杂特征配置生成有效的可网格化分解，推动全六面体自动网格生成这一“圣杯”问题的解决。

    

    面向复杂CAD模型的自动六面体网格生成长期以来被视为网格生成领域的“圣杯”。其根本挑战在于特征错位问题，即正交特征或相互作用的特征共存，使得直接进行六面体网格化不可行，需要将模型分解为更简单的、可网格化的块。尽管现有的分解方法，包括框架场流水线和基于学习的技术，已经减少了人工干预，但它们对几何和拓扑复杂性仍然敏感，可能无法为具有挑战性的特征配置生成有效的分解。具体而言，它们的动作空间通常仅限于基于表面的分割，这往往会产生无效或不可网格化的分解。为了解决这些局限性，我们提出了DecompRL，一个面向六面体网格化的深度强化学习CAD分解框架。我们设计了一组拓扑感知的内部分割操作。

    arXiv:2611.12930v1 Announce Type: cross  Abstract: Automatic hexahedral meshing for complex CAD models has long been regarded as the "holy grail" of mesh generation. The fundamental challenge arises from feature misalignment, where orthogonal or interacting features coexist, making direct hexahedral meshing infeasible and requiring decomposition into simpler, meshable blocks. While existing decomposition methods, including frame-field pipelines and learning-based techniques, have reduced manual intervention, they can remain sensitive to geometric and topological complexity and may fail to produce valid decompositions for challenging feature configurations. Specifically, their action spaces are typically restricted to surface-based splitting, which often yields invalid or unmeshable decompositions. To address these limitations, we propose DecompRL, a deep reinforcement learning framework for CAD decomposition toward hexahedral meshing. We design a set of topology-aware internal-splittin
    
[^226]: 用于Brugge油田辅助历史拟合的图生成模型集成方法

    An ensemble of graph generative models for assisted history matching of the Brugge field case

    [https://arxiv.org/abs/2611.12927](https://arxiv.org/abs/2611.12927)

    该论文提出一种图生成模型集成方法，为Brugge油田的每个地层沉积带分别建立专用图自编码器，共享联合潜在空间并使用CMA-ES进化策略进行优化，直接在角点网格活跃单元上操作，从而实现对具有多种沉积环境油田的辅助历史拟合。

    

    辅助历史拟合仍然存在非唯一性问题，因为许多地质实现能够重现等效的生产响应，而单井拟合结果之间却存在显著差异。当油田包含多种沉积环境和建模策略时，这种模糊性会进一步加剧。本文将图自编码器方法应用于包含不同沉积带的基准Brugge油田的历史拟合。每个地层层带均由其专用的生成模型进行建模，这些生成模型共享一个用于优化历史拟合循环的联合潜在空间。基于图的生成器直接作用于角点网格的活跃单元，因此无需将网格转换为规则网格即可处理其不规则的几何形状。在160维联合潜在空间中的优化搜索采用协方差矩阵自适应进化策略（CMA-ES）完成。多分量目标函数包括静态井数据不匹配（摘要在此处被截断）

    arXiv:2611.12927v1 Announce Type: new  Abstract: Assisted history matching remains non-unique because many geological realisations can reproduce equivalent production responses with substantial differences between individual well matches. This ambiguity increases when a field comprises several depositional settings and modelling strategies. The paper applied a graph autoencoder approach to history matching of a benchmark Brugge field that comprises different depositional zones. Each stratigraphic zone is modelled with its dedicated generative model that comprises a joint latent space for optimisation history matching loops. Graph-based generators operate directly on the active cells of the corner-point grid, so its irregular geometry is handled without conversion to a regular lattice. The optimisation search in the 160-dimensional joint latent space is done with the Covariance Matrix Adaptation Evolution Strategy. The multi-component objective function includes static well data mismatc
    
[^227]: IntentTwin：离线网页行为预测中候选支持去混杂与“理想评测到实际部署”的差距

    IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction

    [https://arxiv.org/abs/2611.12918](https://arxiv.org/abs/2611.12918)

    该论文揭示了离线网页行为预测基准中仅凭候选动作是否属于训练支持集即可“作弊”答对的混杂效应，提出支持可识别率（SIR）这一度量、支持均衡候选集构建方法以及免目标双路径提议器，从而真实刻画从理想评测条件到实际部署的性能差距。

    

    离线网页行为基准测试通常将示范动作插入一个有限候选集，然后测量模型的重排序性能。我们证明这种协议会造成“候选支持混杂”问题：当目标动作位于训练支持集之外、而所有干扰项均从该支持集中采样时，仅凭支持集归属信息就能识别出正确动作，完全无需建模用户行为。我们用“支持可识别率”（SIR）将这一效应形式化，构建了支持均衡的候选集，并评估了一种免目标的双路径候选提议器，该提议器将支持集检索与开放词表的DOM/语言模型提议相结合。在过滤后的OPeRA数据集（训练/验证/测试分别为4,282/469/992步）上，原始协议在目标为词表外（OOV）的样本上SIR达到1.0000，仅用支持集规则即可取得1.0000的OOV Exact@1。支持均衡化将SIR降至0.0000；此时最强的条件重排序器总体Exact@1为0.7214，在词表内（IV）样本上为0.7988，在OOV样本上为0.5668。免目标双路径系统达到P（摘要原文在此处截断）

    arXiv:2611.12918v1 Announce Type: new  Abstract: Offline web-behavior benchmarks frequently insert the demonstrated action into a finite candidate set and then measure reranking. We show that this protocol can create a candidate-support confound: when the target is outside training support while all distractors are sampled from that support, support membership identifies the gold action without modeling behavior. We formalize this effect with support-identifiability rate (SIR), construct support-balanced candidate sets, and evaluate a target-free dual-path proposer that combines support retrieval with open-vocabulary DOM/LM proposals. On filtered OPeRA (4,282/469/992 train/dev/test steps), the original protocol has SIR 1.0000 on target-OOV rows and a support-only rule reaches 1.0000 OOV Exact@1. Support balancing reduces SIR to 0.0000; the strongest conditional reranker obtains 0.7214 overall Exact@1, 0.7988 on IV rows, and 0.5668 on OOV rows. The target-free dual-path system reaches P
    
[^228]: 莫比乌斯交互过滤：面向多模态幽默检测的阶数选择性谱正则化

    M\"obius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection

    [https://arxiv.org/abs/2611.12913](https://arxiv.org/abs/2611.12913)

    提出莫比乌斯交互过滤方法，通过阶数选择性谱正则化在保留高阶模态交互表示的同时调控其对预测的贡献，使多模态幽默检测的标签对齐信噪比提升1.62-2.79倍。

    

    多模态幽默检测需要联合解读语言、声音表达和视觉行为，同时表达性融合还必须调控不同阶数的交互以何种程度进入最终判断。我们提出莫比乌斯交互过滤，它学习一个模态子集预测函数，并通过其聚合结构施加阶数选择性的谱正则化。私有时间编码器和平衡的双向交互构建了有监督的单模态、双模态和三模态预测器，它们的均匀聚合恰好构成一个具有几何递减交互阶响应的莫比乌斯域滤波器。这一构造在保留高阶表示的同时调控其对预测的贡献。在UR-FUNNY数据集上的受控扰动实验显示，各交互阶数的标签对齐信噪比出现显著下降。所诱导的滤波器使最终比率相对……提升了1.62-2.79倍。

    arXiv:2611.12913v1 Announce Type: cross  Abstract: Multimodal humor detection requires interpreting language, vocal delivery, and visual behavior jointly. Expressive fusion must also regulate how interactions of different orders enter the final judgment. We propose M\"obius Interaction Filtering (MoIF), which learns a modality-subset prediction function and imposes order-selective spectral regularization through its aggregation structure. Private temporal encoders and balanced reciprocal interaction construct supervised unimodal, bimodal, and trimodal predictors. Their uniform aggregate is exactly a M\"obius-domain filter with a geometrically decreasing interaction-order response. This formulation retains higher-order representations while regulating their contribution to prediction. Controlled UR-FUNNY perturbations reveal a pronounced decline in label-aligned signal-to-noise ratio across interaction orders. The induced filter improves the final ratio by 1.62-2.79 times relative to th
    
[^229]: 基于流形生成学习的概率地震性能评估与结构可辨识性

    Generative Learning on Manifolds for Probabilistic Seismic Performance Assessment and Structural Identifiability

    [https://arxiv.org/abs/2611.12908](https://arxiv.org/abs/2611.12908)

    该研究提出一种结合扩散映射、分数扩散模型与几何调和的流形生成学习框架，可从有限的非线性仿真中学习结构参数与地震响应的联合分布，同时实现地震性能的正向概率预测与结构特性的反向辨识。

    

    地震性能评估依赖于计算成本高昂的非线性仿真，这限制了不确定性量化以及从观测响应中推断结构特性的能力。我们开发了一种基于流形的生成框架，能够从有限的高保真仿真中学习不确定结构参数与地震响应轨迹的联合分布。该框架结合了扩散映射、基于分数的扩散建模和几何调和方法，可实现正向概率预测与反向推断。将该框架应用于一座九层钢抗弯框架，结果揭示了哪些结构特性可从地震性能中辨识出来，以及这些特性的不确定性如何影响结构响应。特别是，学习到的表示比单个强度参数更有效地捕捉了峰值后退化特性。反向推断可识别出不同的结构特性（原文在此处截断）。

    arXiv:2611.12908v1 Announce Type: cross  Abstract: Seismic performance assessment relies on computationally expensive nonlinear simulations, limiting uncertainty quantification and inference of structural properties from observed responses. We develop a manifold-based generative framework that learns the joint distribution of uncertain structural parameters and seismic-response trajectories from limited high-fidelity simulations. Combining Diffusion Maps, score-based Diffusion Modeling, and Geometric Harmonics, the framework enables both forward probabilistic prediction and inverse inference. Applied to a nine-story steel moment-resisting frame, the results reveal which structural properties are identifiable from seismic performance and how uncertainty in these properties influences response. In particular, the learned representation captures post-capping degradation characteristics more effectively than individual strength parameters. Inverse inference identifies distinct structural c
    
[^230]: 可操作的反馈：利用临床文档中用户编辑数据提升模型能力

    Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation

    [https://arxiv.org/abs/2611.12904](https://arxiv.org/abs/2611.12904)

    该论文针对临床文档场景中用户编辑数据并非全部可操作的问题（如45%以上编辑包含模型输入不支持的信息），构建了临床医生编辑行为分类体系并开发了区分可操作反馈的方法，以避免微调导致模型产生幻觉。

    

    大语言模型越来越多地被部署到临床文档工作流程中，临床医生在使用前会审查并编辑模型的输出。原则上，这些临床医生的编辑代表了模型训练的一个可扩展的监督来源。然而在实践中，并非所有编辑都构成可操作的反馈：例如，通过审查来自AI临床文档产品实际部署的数据，我们发现超过45%的编辑后输出包含用户添加的、模型输入并不直接支持的信息。如果在这种编辑上进行简单微调，会有教会模型产生幻觉的风险。在本文中，我们开发了一种方法来应对这一挑战。鉴于已部署文档工具上的真实临床医生编辑数据并不公开，我们首先基于来自已部署AI临床文档产品的数据，构建了一个临床医生编辑行为的分类体系。然后，我们开发了一种生成……的方法

    arXiv:2611.12904v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly being deployed in clinical documentation workflows, where clinicians review and edit their outputs before use. In principle, these clinician edits represent a scalable source of supervision for model training. In practice, not all edits represent actionable feedback: for instance, on data drawn from real-world deployment of an AI clinical documentation product, our review suggests that over 45% of edited outputs contain user-added information that is not directly supported by the inputs to the model. Naively fine-tuning on such edits risks teaching the model to hallucinate. In this paper, we develop a method to address this challenge. Given that real clinician edits on deployed documentation tools are not publicly available, we first develop a taxonomy of clinician-editing behaviors, informed by data from a deployed AI clinical documentation product. Then, we develop a method for generating
    
[^231]: 基于卫星测高数据轨迹采样的冰盖表面重建的D-TIN方法

    D-TINs for Track-Sampled Ice Sheet Surface Reconstruction from Satellite Altimetry Data

    [https://arxiv.org/abs/2611.12894](https://arxiv.org/abs/2611.12894)

    本文提出D-TIN方法，直接对卫星测高观测点进行Delaunay三角剖分和重心插值以重建冰盖表面，在精度接近克里金法的同时将运行时间降低了252倍。

    

    卫星测高沿轨迹对冰盖进行密集采样，但在轨迹之间采样稀疏，空间覆盖中经常存在缺口。我们研究了这种不规则采样分布对表面重建的影响，并提出了一种新方法D-TIN，该方法将Delaunay三角剖分和重心插值直接应用于观测位置。通过空间保留集（spatial holdout）移除了连续的轨迹区间以及同一次经过的邻近观测。我们将D-TIN与两种常见工作流程——克里金法和基于瓦片的多项式插值——进行比较，以评估其精度和计算效率。在七个ICESat-1观测活动中，相对于独立的基于5公里瓦片和多项式的重建，D-TIN将RMSE降低了12.1米至23.6米。其总体精度接近于在常用的GStatSim平台上使用1公里网格的普通克里金工作流程，而其中位数运行时间则降低了252倍。D-TIN的固定权重多尺度扩展版本，称为MS-D-TIN，提供了……

    arXiv:2611.12894v1 Announce Type: cross  Abstract: Satellite altimetry samples ice sheets densely along tracks but sparsely across tracks, with frequent gaps in spatial coverage. We study the impact of this irregular support on surface reconstruction and introduce a new method, D-TIN, that applies Delaunay triangulation and barycentric interpolation directly to observed locations. A spatial holdout removes contiguous track intervals and nearby same-pass observations. We compare D-TIN with two common workflows, kriging and tile-based polynomial interpolation, to evaluate its accuracy and computational efficiency. Across seven ICESat-1 campaigns, D-TIN reduces RMSE by 12.1 m to 23.6 m relative to independent 5 km tile and polynomial based reconstruction. Its aggregate accuracy is close to the ordinary kriging workflow using a 1 km grid on the commonly used GStatSim platform while its median runtime is 252 times lower. A fixed-weight multiscale extension of D-TIN, called MS-D-TIN, gives t
    
[^232]: XposedMIA：利用可解释人工智能（XAI）实现单次成员推断攻击

    XposedMIA: Exploiting Explainable Artificial Intelligence (XAI) for One-Shot Membership Inference Attacks

    [https://arxiv.org/abs/2611.12067](https://arxiv.org/abs/2611.12067)

    该论文提出XposedMIA攻击框架，首次将LIME局部代理几何及表格和视觉反事实解释转化为样本到决策边界距离的代理指标，仅需30个样本进行一次性校准即可实施单次黑盒成员推断攻击，揭示了可解释性输出泄露训练数据成员信息的隐私风险。

    

    机器学习（ML）服务越来越倾向于在模型预测的同时输出可解释性结果。虽然此类解释旨在提升透明度，但它们也可能泄露有关底层训练数据的信息。在本工作中，我们证明了局部可解释模型无关解释所采用的局部代理几何结构可以被重新用作样本到模型决策边界距离的代理指标。我们将这一边界距离视角扩展到表格反事实解释，并首次扩展到图像分类器的视觉反事实解释。我们提出了XposedMIA，这是一个单次、基于解释的黑盒成员推断攻击框架，它将返回的解释转化为被查询样本到模型决策边界距离的代理指标。在仅包含30个样本的小型带标签校准集上进行一次校准后，我们的攻击只需要……（原文摘要至此被截断）

    arXiv:2611.12067v2 Announce Type: replace  Abstract: Machine learning (ML) services increasingly expose explainability outputs alongside model predictions. While such explanations are intended to improve transparency, they may also reveal information about the underlying training data. In this work, we show that the local surrogate geometry employed by Local Interpretable Model-agnostic Explanations (LIME) can be repurposed as a proxy for a sample's distance to the model's decision boundary. We extend this boundary-distance view to tabular counterfactual explanations and, for the first time, to visual counterfactual explanations for image classifiers. We present XposedMIA, a one-shot, explanation-based black-box membership inference attack (MIA) framework that converts the returned explanation into a proxy for the queried sample's distance to the model decision boundary. After a one-time calibration step on a small labeled calibration set of as few as 30 samples, our attack requires on
    
[^233]: 淡出至…红色？调度有色噪声探索以支持高效机器人学习并减少运动抖动

    Fade to... Red? Scheduling Coloured Noise Exploration Supports Efficient Robot Learning with Reduced Movement Jerk

    [https://arxiv.org/abs/2611.11868](https://arxiv.org/abs/2611.11868)

    在机器人强化学习训练中，将探索噪声从白噪声逐渐过渡到红噪声（有色噪声），可在不损失学习效率和最终性能的前提下，将探索动作的抖动减半，实现更平滑高效的学习。

    

    深度强化学习（RL）在连续控制中的探索通常由时间上不相关的高斯（白）动作噪声驱动，这会导致动作抖动且探索局限于局部范围。相比之下，时间上相关的有色噪声，尤其是粉红噪声，已被证明能支持更高效的RL探索。受这些发现的启发，我们探究在训练过程中调度噪声颜色的变化能否在连续控制机器人任务中优于白噪声。我们在使用Fetch机械臂和Shadow灵巧手的四个稀疏奖励操作任务上，对白噪声与红噪声之间的49种线性调度方案结合广义状态相关探索（gSDE）进行了系统性研究。我们发现训练结束时的噪声颜色并不影响学习效率或最终性能，而向红噪声渐变则在每个任务中将探索的动作抖动减半……

    arXiv:2611.11868v2 Announce Type: replace-cross  Abstract: Exploration in deep reinforcement learning (RL) for continuous control is commonly driven by temporally uncorrelated Gaussian (white) action noise, which leads to jerky motions with locally constrained exploration. In contrast, temporally correlated coloured noise, particularly pink noise, has been shown to support more efficient exploration in RL. Motivated by these findings, we inquire whether scheduling changes to the noise colour over the course of training can outperform white noise for continuous control robotics. We perform a systematic study of 49 linear schedules between white and red noise, together with generalised state-dependent exploration (gSDE), on four sparse-reward manipulation tasks with a Fetch arm and a Shadow Dexterous Hand. We find that the noise colour at the end of training does not affect learning efficiency or end performance, whereas fading towards red noise halves the jerk of exploration in every ta
    
[^234]: 后见评论家能否引导延迟行动者？

    Can Hindsight Critics Guide Delayed Actors?

    [https://arxiv.org/abs/2611.11863](https://arxiv.org/abs/2611.11863)

    提出Hindsight Critics方法，利用事后对齐的状态-动作转移训练评论家并生成特权价值估计来指导仅能获得延迟观测的行动者，在无需修改部署阶段观测的情况下部分补偿了感知延迟。

    

    感知延迟会造成环境状态与智能体可用观测之间的时间错位，从而将马尔可夫控制问题转变为部分可观测问题。在本研究中，我们探究这种错位能否在训练期间通过事后方法恢复对齐的状态-动作转移来加以缓解。我们提出了Hindsight Critics，这是柔性演员-评论家算法在部分可观测性下强化学习的一种非对称扩展。评论家在事后对齐的转移上训练，而行动者尽管只能获取延迟的观测，却能从由此产生的特权价值估计中学习。在DelayBench基准套件上与延迟感知和延迟无感知基线的比较表明，Hindsight Critics能够部分补偿观测延迟，且无需在部署阶段修改智能体的观测。

    arXiv:2611.11863v2 Announce Type: replace  Abstract: Sensory delays create temporal misalignments between the state of the environment and the observations available to an agent, turning Markovian control problems into partially observable ones. In this work, we inquire whether such misalignments can be mitigated during training by using hindsight to recover aligned state-action transitions. We introduce Hindsight Critics, an asymmetric extension of the Soft Actor-Critic for reinforcement learning under partial observability. The critics are trained on hindsight-aligned transitions whereas the actor learns from the resulting privileged value estimates despite only having access to delayed observations. Comparisons against delay-aware and unaware baselines on the DelayBench suite show that Hindsight Critics can partially compensate for observation delays without requiring modifications to the agent's observations during deployment.
    
[^235]: 扩散语言模型何时应重新考虑其承诺？

    When Should Diffusion Language Models Reconsider Their Commitments?

    [https://arxiv.org/abs/2611.11551](https://arxiv.org/abs/2611.11551)

    该论文提出PURL框架，通过联合学习编辑选择、取消掩码选择和词元预测三个决策，使掩码扩散模型能够在并行快速生成的同时有选择地修正早期不一致的词元承诺，且推理时三个决策仅需一次共享前向传播即可完成。

    

    掩码扩散模型以通过并行解码实现快速文本生成而闻名。然而，相互独立的并行采样可能会在早期就“承诺”不一致的预测结果，而关于如何有选择地修正这些早期的词元承诺，目前所知甚少。在这项工作中，我们为掩码扩散模型提出了计划式取消掩码与修正联合学习（Planned Unmasking-and-Revision Joint Learning, PURL），这是一个基于训练的框架，无缝地将并行生成与选择性词元修正统一起来。我们推导出了一个归一化的联合转移，其似然可以分解为编辑选择、取消掩码选择和词元预测三个部分。这些组件在根据模型自身取消掩码策略构建的部分掩码序列上进行联合训练。在推理阶段，所有三个决策均由一次共享的前向传播所支持，使模型能够解码被掩码的词元并修正带有噪声的词元。

    arXiv:2611.11551v2 Announce Type: replace-cross  Abstract: Masked Diffusion Models (MDMs) are well known for fast text generation via parallel decoding. However, independent parallel sampling can commit to inconsistent predictions, and less is known about how to selectively revise these early token commitments. In this work, we introduce $\underline{\mathrm{P}}$lanned $\underline{\mathrm{U}}$nmasking-and-$\underline{\mathrm{R}}$evision Joint $\underline{\mathrm{L}}$earning (PURL) for MDMs, a training-based framework that seamlessly unifies parallel generation with selective token revision. We derive a normalized joint transition whose likelihood factorizes into edit selection, unmask selection, and token prediction. These components are trained jointly on partially masked sequences constructed according to the model's own unmasking policy. During inference, all three decisions are supported by a single shared forward pass, enabling the model to decode masked tokens and revise noisy tok
    
[^236]: 变分量子本征求解器的炼金术可迁移参数

    Alchemically Transferable Parameters for Variational Quantum Eigensolvers

    [https://arxiv.org/abs/2611.11445](https://arxiv.org/abs/2611.11445)

    该论文提出一种“炼金术”几何投影方法，将有机分子的局部键合环境映射为替代氢基片段，使机器学习模型无需针对有机分子重新训练，即可直接预测变分量子本征求解器所需的电路参数。

    

    机器学习电路参数可以消除变分量子本征求解器（VQE）的一个主要瓶颈：经典优化循环——其典型的贫瘠高原、局部极小值以及昂贵的能量和梯度评估——将被直接预测所取代。先前的工作表明，机器学习模型与结构化电路设计相结合，可以在氢基体系内实现跨系统尺寸和结构的参数迁移。下一步：将该方法扩展到有机分子，似乎需要在特定用例的训练集上重新训练。在这项工作中，我们通过建立一种炼金术方法证明这是不必要的。通过几何投影，沿目标分子的键方向放置替代氢原子，从而将其局部键合环境映射到小的氢基片段上。如此一来，现有的氢基体系求解器可以直接复用。唯一需要引入的新量是每种键类型对应的一个投影半径。

    arXiv:2611.11445v1 Announce Type: cross  Abstract: Machine-learned circuit parameters could remove one of the main bottlenecks of variational quantum eigensolvers: the classical optimization loop, with its typical barren plateaus, local minima, and costly energy and gradient evaluations, is replaced by a direct prediction. Prior work showed that machine-learning models paired with a structured circuit design can transfer across system size and structure within hydrogenic systems. The next step: Extending them to organic molecules would seem to require re-training on use-case-specific training sets. In this work, we show that this is not necessary by establishing an alchemical approach. A geometric projection places surrogate hydrogen atoms along the bond directions of the target molecule and so maps its local bonding environment onto small hydrogenic fragments. Existing hydrogenic solvers can then be reused as they are. The only new quantity is a single projection radius per bond type,
    
[^237]: STABLE：一种具有自适应漂移控制的持续学习优化器

    STABLE: A Continual Learning Optimizer with Adaptive Drift Control

    [https://arxiv.org/abs/2611.10655](https://arxiv.org/abs/2611.10655)

    该论文提出STABLE，一种具有自适应漂移控制的原对偶优化器，理论上证明有界参数漂移可保证有界的间隔侵蚀，并通过动态调节对偶变量来强制执行漂移预算，从而从优化器层面缓解持续学习中的灾难性遗忘。

    

    持续学习深受灾难性遗忘的困扰，这会导致过去知识的丢失以及在任务流上适应过程的不稳定。现有方法通常通过重放缓冲区、架构修改或额外的正则化项来解决这一问题。我们认为，遗忘是不稳定学习动力学的后果，因此可以从优化器层面的方案中受益。主要挑战在于学习动力学是随时间变化的，诸如“稳定性间隙”等经验现象正说明了这一点。首先，我们严格证明：对于基于距离的分类器，在局部分数稳定性条件下，有界的参数漂移可以保证有界的分类间隔侵蚀。基于这一洞察，我们提出了STABLE——一种具有自适应漂移控制的原对偶优化器。STABLE通过在超出漂移预算时增大对偶变量、反之减小时下对偶变量，来强制执行漂移预算。由此实现了高稳定性，同时（原文在此处截断）……

    arXiv:2611.10655v2 Announce Type: replace-cross  Abstract: Continual learning suffers from catastrophic forgetting, leading to the loss of past knowledge and unstable adaptation across task streams. Existing approaches typically address this issue through replay buffers, architectural changes, or additional regularization terms. We argue that forgetting is a consequence of unstable learning dynamics, which may benefit from an optimizer-level solution. The main challenge is that learning dynamics are time-dependent, as demonstrated by empirical phenomena such as the stability gap. First, we formally prove that, for distance-based classifiers, bounded parameter drift guarantees bounded margin erosion under a local score-stability condition. Based on this insight, we propose STABLE, a primal-dual optimizer with adaptive drift control. STABLE enforces a drift budget by increasing the dual variable when the budget is violated and decreasing it otherwise. This results in high stability witho
    
[^238]: 仅靠看见还不够：利用TIGER弥合推理访问差距以解锁视觉推理

    Seeing Isn't Enough: Unlocking Visual Reasoning by Bridging the Reasoning-Access Gap with TIGER

    [https://arxiv.org/abs/2611.10552](https://arxiv.org/abs/2611.10552)

    该论文揭示了多模态大语言模型中存在的“模态推理访问差距”——模型虽能准确感知图像内容却无法将其路由进已有的文本推理机制——并提出TIGER训练方法，通过挖掘模型在文本上成功但在对应对称图像上失败的模态反事实样本来定向弥合这一差距，从而解锁视觉推理能力。

    

    多模态大语言模型（MLLMs）从其语言骨干网络和文本预训练中继承了强大的文本推理能力，但往往无法将同等水平的能力应用于视觉输入。我们通过对模型在语义等价的文本输入和渲染文本图像输入上进行评估，证明这种性能下降不能仅仅归因于感知失败：模型能够准确读取图像中的文本，并在获得自身转录文本时恢复接近文本输入的性能，但就是无法直接对图像进行良好的推理。我们将这种差异表征为模态推理访问差距，即模型无法将感知到的视觉内容路由到文本可访问的现有推理机制中。为弥合这一差距，我们提出了TIGER（面向文本到图像差距的定向训练以增强推理），该方法首先挖掘合成模态反事实样本，即模型在某个文本问题上能成功求解，但在提供对应图像时却失败的情况……

    arXiv:2611.10552v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) inherit strong textual reasoning capabilities from their language backbones and text pretraining, but often fail to apply the same level of competency to visual inputs. We show that this degradation cannot be attributed to only perception failures, by evaluating models on semantically-equivalent text and rendered-text image inputs: models can accurately read text in the images and recover near-text performance when provided their own transcriptions, but just cannot reason well on the images directly. We characterize this discrepancy as a modality reasoning-access gap, where models fail to route perceived visual content into existing reasoning machinery accessible with text. To bridge this, we propose TIGER (Text-to-Image Gap-targeted training for Enhanced Reasoning), which first mines synthetic modality counterfactuals where a model succeeds on a text problem but fails when provided the corres
    
[^239]: 分裂与桥接：通过扩散桥接实现多模态生成

    Split and Bridge: Multimodal Generation via Diffusion Bridging

    [https://arxiv.org/abs/2611.10441](https://arxiv.org/abs/2611.10441)

    提出了一种自下而上的多模态生成框架 SnB，通过在采样时利用桥接模块连接预训练的单模态扩散模型，无需联合训练即可生成连贯的多模态样本，实现了预训练模型的即插即用复用。

    

    视觉数据通常通过多种互补的模态来表达（例如图像和分割掩码），每种模态捕捉同一底层结构的不同方面。现有的大多数多模态方法采用自上而下的范式，通过从头开始的联合训练来学习一个共享的潜在变量，这限制了可扩展性。我们提出了 Split and Bridge（SnB），这是一种自下而上的多模态生成框架，它在采样阶段将预训练的单模态扩散模型耦合起来。SnB 并不学习统一的潜在空间，而是沿着一条共享的扩散轨迹前进到某个分裂点，然后在桥接模块的引导下分支为各模态特定的轨迹，从而生成连贯的多模态样本。这种设计使得强大的预训练模型可以即插即用地被复用，无需联合训练，使其具有良好的可扩展性和易适配性。在标准基准（PolyMNIST、CelebAMask-HQ）以及更复杂的场景（PIE-B……（摘要被截断）

    arXiv:2611.10441v2 Announce Type: replace-cross  Abstract: Visual data are often expressed through multiple complementary modalities (e.g., images and segmentation masks), each capturing different aspects of the same underlying structure. Most existing multimodal methods adopt a top-down paradigm, learning a shared latent variable through joint training from scratch, which limits scalability. We propose Split and Bridge (SnB), a bottom-up framework for multimodal generation that couples pretrained unimodal diffusion models at sampling time. Rather than learning a unified latent space, SnB generates coherent multimodal samples by following a shared diffusion trajectory up to a split point and then branching into modality-specific trajectories guided by a bridge module. This design enables plug-and-play reuse of powerful pretrained models without joint training, making it scalable and easily adaptable. Across standard benchmarks (PolyMNIST, CelebAMask-HQ) and more complex settings (PIE-B
    
[^240]: 面向连续体资源的大语言模型管理中的持续指令微调

    Continual Instruction Tuning for LLM-ased Management of Continuum Resources

    [https://arxiv.org/abs/2611.10292](https://arxiv.org/abs/2611.10292)

    该论文提出将大语言模型作为持续指令微调的意图编译器（而非嵌入延迟敏感控制回路中），置于快速确定性战术控制器之上，并通过执行评分在控制器实际目标中度量灾难性遗忘，从而实现云-边连续体资源的高效LLM管理。

    

    arXiv:2611.10292v2 公告类型：replace-cross。摘要：大语言模型（LLM）正被越来越多地探索用于云-边连续体的资源管理。然而，将LLM直接嵌入延迟敏感的控制回路中，会把推理延迟、执行抖动和幻觉风险置于关键路径上。本研究主张对LLM采用一种不同的使用方式：一个经过持续指令微调的意图编译器，它在每个管理意图或声明的上下文变化发生时运行一次，位于一个快速、确定性的战术控制器之上，由该战术控制器处理所有低延迟的运行时操作。我们将意图编译表述为一个持续指令微调问题，因为管理意图是按顺序到达并不断累积的，同时基础设施和工作负载的上下文也在漂移。与近期基于LLM的管理框架不同，我们的设计通过对模型输出进行执行评分，从而能够在底层控制器所执行的目标函数中度量灾难性遗忘。在一个连续体仿真器上……（原文摘要至此处截断）

    arXiv:2611.10292v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly explored for resource management across the Cloud-Edge continuum. However, embedding LLMs directly inside latency-sensitive control loops places inference latency, execution jitter, and hallucination risks on the critical path. This study argues for a different use of LLMs: a continually instruction-tuned intent compiler that runs once per management intent or declared context change, above a fast, deterministic tactical controller that handles all low-latency runtime actions. We formulate intent compilation as a continual instruction tuning problem, because management intents arrive sequentially and accumulate while infrastructure and workload contexts drift. Unlike recent LLM-based management frameworks, our design scores the model's outputs by execution, so that catastrophic forgetting can be measured in the executed objective of the underlying controller. On a continuum simulato
    
[^241]: TORNADO：面向校准概率偏微分方程预测的自适应潜空间随机输运

    TORNADO: Adaptive Latent Stochastic Transport for Calibrated Probabilistic PDE Forecasting

    [https://arxiv.org/abs/2611.10251](https://arxiv.org/abs/2611.10251)

    TORNADO提出了一种潜空间概率预测框架，通过学习具有状态依赖扩散项的SDE在VAE后验分布之间进行随机输运，实现无需模拟的训练，从而在计算受限的少步推理下为PDE预测提供校准的不确定性量化。

    

    面向偏微分方程（PDE）的概率神经代理模型能够支持集合预报，但在用于不确定性量化时常表现出校准失准，尤其是在计算资源受限、推理步数较少的场景下。现有的生成式方法通常依赖于固定的全局随机性，无法使预测不确定性适应当地的动力学状态。我们提出了TORNADO，这是一种潜空间概率预测框架，它将下一步预测表述为VAE编码器后验分布之间的随机输运。与现有采用全局规定随机性的方法不同，TORNADO学习一个具有状态依赖扩散项的随机微分方程（SDE），使预测不确定性能够自适应于局部潜空间动力学。该模型使用由高斯插值推导出的解析条件端点目标进行训练，实现了无需模拟（simulation-free）的学习。此外，引入能量距离正则化项以改善分布对齐。实验上，TO（摘要在此处被截断）

    arXiv:2611.10251v2 Announce Type: replace  Abstract: Probabilistic neural surrogates for PDEs enable ensemble forecasting, but often exhibit miscalibration when used for uncertainty quantification, especially in compute-constrained, few-step inference regimes. Existing generative approaches typically rely on fixed global stochasticity, which cannot adapt predictive uncertainty to the local dynamical state. We propose TORNADO, a latent probabilistic forecasting framework that formulates next-step prediction as stochastic transport between VAE encoder posteriors. Unlike existing approaches with globally prescribed stochasticity, TORNADO learns a SDE with state-dependent diffusion, enabling the predictive uncertainty to adapt to the local latent dynamics. The model is trained using analytic conditional endpoint targets derived from a Gaussian interpolant, enabling simulation-free learning. An additional energy distance regularization term improves distributional alignment. Empirically, TO
    
[^242]: 计数事件，而非看见事件：诊断与修复视频大语言模型中的读取失败

    Counting Events, Not Seeing Them: Diagnosing and Repairing the Readout Failure in Video LLMs

    [https://arxiv.org/abs/2611.09949](https://arxiv.org/abs/2611.09949)

    该论文发现视频大语言模型物体计数失败的根源并非缺少物体恒常性，而是计数信号已存在于内部表征中却未被正确读取，并通过片段定位监督与离散事件符号辅助通道两种修复方法，将同一物体事件计数准确率从46.7%大幅提升至65.0%。

    

    视频大语言模型在计算物体穿过屏幕的次数时表现明显失败，这一失败被归因于缺失物体恒常性。利用基于构造的合成测试平台和经过验证的标签，我们表明这种归因是不完整的：计数信号其实存在，但未被读取——线性探针能够以93%的准确率解码每个时间窗口内的物体存在，而全局计数解码则退化为对持续时间的混淆。同期研究在重复计数任务中报告了相同的表征-输出分离现象；我们则证明，在“离场-重现”事件计数任务中，这种失败是可以修复的：两种针对读取过程的修复方法（片段定位监督和离散事件符号辅助通道）在随机水平为33%的基准上，将同一物体计数准确率从46.7%提升至65.0%（oracle上限为66.7%），并在7B至30B规模的多个系统中均有效。迁移到独立的工厂场景毫无损失，而一项测量审计则暴露出五个曾掩盖问题边界的数据完整性缺陷。一项真实视频审计（REA

    arXiv:2611.09949v2 Announce Type: replace-cross  Abstract: Video LLMs fail conspicuously at counting how many times an object crosses the screen, a failure attributed to missing object permanence. Using construction-grounded synthetic testbeds with validated labels, we show this attribution is incomplete: the count signal is present but unread - linear probes decode per-window presence at 93% while global count decoding collapses to a duration confound. Concurrent work reports the same representation-output dissociation for repetition counting; we show that in exit-and-reappear event counting it is repairable: two readout-directed repairs (episode-localization supervision and a discrete event-symbol side channel) raise same-object counting from 46.7% to 65.0% (oracle 66.7%) on a chance-33% benchmark across 7B-30B systems. Transfer to an independent factory costs nothing, and a measurement audit exposes five data-integrity failures that had obscured the boundary. A real-video audit (REA
    
[^243]: 知道何时干预：AI智能体的因果评估

    Knowing When to Intervene: Causal Evaluation of AI Agents

    [https://arxiv.org/abs/2611.09792](https://arxiv.org/abs/2611.09792)

    该论文提出了首个基于因果推断的方法学框架，用于评估在AI智能体执行轨迹中进行干预（如人工审查或行为引导）的因果价值。

    

    当前对智能体AI系统的评估主要依赖于基准测试。然而，基准测试的静态特性使其无法提供关于部署时审计干预价值的信息。为了实现可靠部署，我们常常需要在智能体的执行轨迹中判断是否进行干预，例如通过验证智能体当前的输出或引导其采取不同的行为，这可以采用人工审查、成本更高的模型或耗时的智能体监督。此类干预的价值从根本上说是一个因果推断问题，即这种干预的因果效应是否对最终结果有益。由于问题的动态性质，估计这种因果效应尤其具有挑战性：干预的影响取决于当前的上下文和环境，以及智能体未来的行为。我们开发了首个在方法学上有依据的框架，用于评估此类干预的价值。

    arXiv:2611.09792v1 Announce Type: cross  Abstract: Current evaluations of agentic AI systems primarily rely on benchmarks. However, the static nature of benchmarks makes them uninformative about the value of auditing interventions at deployment time. For reliable deployment, we often need to determine within an agent's trajectory whether or not to intervene, e.g., by verifying the agent's current output or steering it toward different behavior, using either human review or more costly models or time-consuming agentic supervision. The value of such interventions is fundamentally a causal inference question, asking whether the causal effect of such an intervention is beneficial for the final outcome. Estimating this causal effect is particularly challenging due to the dynamic nature of the problem: the impact of an intervention depends on the current context and environment, as well as the agent's future behavior. We develop the first methodologically grounded framework for evaluating th
    
[^244]: BEiTScore++：基于BEiT-3与共形化分位数回归的无参考图像描述评估

    BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression

    [https://arxiv.org/abs/2611.09505](https://arxiv.org/abs/2611.09505)

    提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。

    

    可靠的不确定性估计对于将评估指标部署到现实环境中至关重要，因为现实中输入歧义和领域偏移都很常见。在这项工作中，我们提出了BEiTScore图像描述评估指标的不确定性感知扩展版本，在对齐分数之外还能产生经过共形校准的置信区间。我们的方法对人类评判的预测分布进行建模，能够在单次前向传播中同时实现准确评分和有原则的不确定性量化。我们在多个带有人工标注评分的基准上评估了BEiTScore++指标，结果表明它保持了BEiTScore的排序相关性，同时在大多数基准上产生的区间宽度比蒙特卡洛dropout更贴近预测误差。估计的不确定性在图像退化条件下以及域外艺术图像上也会随之增大，这使得BEiTScore++成为一个实用的不确定性感知评估工具。

    arXiv:2611.09505v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is important for deploying evaluation metrics in real-world settings, where both input ambiguity and domain shifts are common. In this work, we propose an uncertainty-aware extension of the BEiTScore image captioning metric, producing conformally calibrated confidence intervals alongside alignment scores. Our approach models predictive distributions over human judgments, enabling both accurate scoring and principled uncertainty quantification in a single forward pass. We evaluate our BEiTScore++ metric across multiple benchmarks with human-annotated ratings, showing that it preserves the ranking correlation of BEiTScore, while producing intervals whose width tracks the prediction error more closely than Monte Carlo dropout on most benchmarks. The estimated uncertainty also grows under image degradation and on out-of-domain artwork images, making BEiTScore++ a practical tool for uncertainty-aware evaluati
    
[^245]: 基于波动率记忆的多变量时间序列排序：一种波动率遗传的时序卷积方法

    Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity

    [https://arxiv.org/abs/2611.07746](https://arxiv.org/abs/2611.07746)

    本文提出QS-TCN方法，将准极大似然估计的波动率持续性统计量与矩阵束方法提取的时序模式嵌入时序卷积网络的结构化滤波器中，从而基于波动率遗传关系对多变量时间序列进行全局排序。

    

    波动率动态可以承载在条件均值中不明显的时序关系信息。由于每个变量自身的持续性可能会掩盖其对其他变量的依赖性，提取这些信息十分困难。本文提出了QS-TCN，这是一种对多变量时间序列进行排序的方法，它将波动率记忆的统计估计与时序卷积学习相结合。通过准极大似然获得的持续性估计与通过矩阵束方法提取的时序模式共同定义了网络内部的结构化滤波器。这些结构化滤波器与学习得到的滤波器一起表示目标变量的历史，并支持变量之间方向性证据的比较。这些比较结果被合并为一个全局的先后顺序，但该方法并不识别因果图。该方法在具有线性和非线性波动率交互的合成系统以及金融和fMRI基准数据集上进行了评估。

    arXiv:2611.07746v1 Announce Type: new  Abstract: Volatility dynamics can carry information about temporal relationships that is not apparent in the conditional mean. Extracting this information is difficult because the persistence of each variable can obscure its dependence on other variables. This paper presents QS-TCN, a method for ordering multivariate time series that combines statistical estimates of volatility memory with temporal convolutional learning. Persistence estimates obtained by quasi-maximum likelihood and temporal modes extracted by the Matrix Pencil method define structured filters within the network. Together with learned filters, they represent the history of the target and support comparisons of directional evidence between variables. These comparisons are combined into a global order of precedence, which does not identify a causal graph. The method is evaluated on synthetic systems with linear and nonlinear volatility interactions and on the Finance and fMRI bench
    
[^246]: ALTO：非洲与黎凡特地区技术助长的基于性别的暴力语料库

    ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

    [https://arxiv.org/abs/2611.06706](https://arxiv.org/abs/2611.06706)

    该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。

    

    技术助长的基于性别的暴力（TFGBV）是一种普遍存在的网络危害，通过多种直接和隐蔽的形式影响女性和性别少数群体。当前用于性别伤害检测的模型和标注数据集主要被来自全球北方语言的数据以及用这些语言训练的模型所主导。我们与来自约旦、法语西非和肯尼亚的民间社会组织（CSOs）合作，开发了一种社区主导的方法，通过众包方式收集黎凡特阿拉伯语、非洲法语和斯瓦希里语中符合当地文化和语言特点的TFGBV实例。我们使用主动学习对众包数据进行标注，构建了一个规模虽小但性能出色的数据集（N=3022），命名为ALTO，用于改进这三种语言的TFGBV检测。通过基准测试和微调实验，我们表明现有的有害内容检测模型在ALTO数据集上的TFGBV召回率较低，而经过微调后性能有所提升。我们发现模型往往无法识别隐蔽的……

    arXiv:2611.06706v1 Announce Type: cross  Abstract: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert ha
    
[^247]: EEGDash语料库：面向脑电（EEG）基础模型的即用型公开训练语料库

    EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models

    [https://arxiv.org/abs/2611.06356](https://arxiv.org/abs/2611.06356)

    该论文发布了首个即用型EEG基础模型公开训练语料库eegdash-corpus-v0，汇集了77个可合法再分发的公开数据集（涵盖7,272名受试者的14,075小时数据），解决了临床档案无法再分发、公开数据分散且格式不一导致难以汇集的核心问题。

    

    脑电图（EEG）基础模型通常在数万小时的记录数据上进行预训练，且其准确率随数据量的增加而提升。然而，这类数据难以汇集：规模最大的数据来源是无法重新分发的临床档案库，而公开数据源则分散在多个档案库、文件格式和许可证之中。我们发布了eegdash-corpus-v0，这是一个包含77个公开EEG数据集、可直接用于训练的语料库，以受限访问的Hugging Face数据集形式（EEGDash/eegdash-corpus-v0）发布，并为每个文件提供了SHA-256校验值。REVE是目前已发表的规模最大的EEG基础模型，其列出了89个数据集作为预训练语料，但其中12个无法重新分发：四个临床档案库（占其总时长的82%）以及八个受禁止衍生条款许可证约束的数据集。本语料库收录了REVE采样率所允许的其余76个数据集，以及来自REVE发布说明的另一个数据集：共计77个数据集、来自7,272名受试者的14,075小时数据、500万个10秒样本，每个训练周期包含62.1亿个token。

    arXiv:2611.06356v1 Announce Type: new  Abstract: Foundation models for electroencephalography (EEG) are pretrained on tens of thousands of hours of recordings, and their accuracy grows with that amount. Such data is hard to assemble. The largest sources are clinical archives that cannot be redistributed, and the public sources are spread over several archives, file formats, and licences. We release eegdash-corpus-v0, a training-ready corpus of 77 public EEG datasets, as the gated Hugging Face dataset EEGDash/eegdash-corpus-v0 with the SHA-256 of every file. REVE, the largest published EEG foundation model, names 89 datasets as its pretraining corpus. Twelve cannot be redistributed: four clinical archives that hold 82% of its hours, and eight under no-derivatives licences. The corpus holds the 76 others that REVE's sampling rate allows and one more from REVE's release card: 77 datasets, 14,075 hours from 7,272 subjects, 5.0 million 10-second samples and 6.21 billion tokens per epoch. Ev
    
[^248]: 语言模型知道哪些约束属于同一组吗？

    Do Language Models Know Which Constraints Belong Together?

    [https://arxiv.org/abs/2611.05726](https://arxiv.org/abs/2611.05726)

    该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。

    

    Dantzig-Wolfe 重构通过将选定的一组约束族凸化来加强混合整数规划的线性松弛，而每个候选选择都需要付出一次列生成的运行代价。当各约束族只有组合在一起才能改进边界时，这一选择变得困难，因为每次只添加一个约束族的搜索会在线性松弛处停滞。批量生产、网络设计和设施选址中的经典不等式正建立在这类零边际组之上。我们证明，当每次评估最多凸化 k 个约束族中的 s 个时，任何仅观察边界值的方法都需要量级为 k^w 的覆盖次数评估，才能找到包含 w 个约束族的隐藏组合，且该结论在实际的 Dantzig-Wolfe 松弛上同样成立。一个遗漏 e 个成员的预测可通过一次无需知晓 e 的分阶段搜索，将成本降至 Θ(k^e)，因此预测所节省的评估次数直接度量了预测器所掌握的知识。我们研究了两种预测来源……

    arXiv:2611.05726v1 Announce Type: cross  Abstract: Dantzig-Wolfe reformulation strengthens the linear relaxation of a mixed-integer program by convexifying a chosen set of constraint families, and every candidate costs a column generation run. The choice is hard when families strengthen the bound only together, because a search that adds one family at a time then stops at the linear relaxation. The classical inequalities of lot sizing, network design and facility location rest on such zero-marginal groups. We prove that when each evaluation convexifies at most $s$ of $k$ families, any method that observes only bounds needs a covering number of evaluations, of order $k^w$, to find a hidden group of $w$ families, also on actual Dantzig-Wolfe relaxations. A prediction that misses $e$ members reduces the cost to $\Theta(k^e)$, through one staged search that does not need to know $e$, so the evaluations a prediction saves measure what the predictor knows. We study two sources of predictions
    
[^249]: 将视觉语言模型重塑为用于临床神经诊断的脑电图基础模型

    Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics

    [https://arxiv.org/abs/2611.05538](https://arxiv.org/abs/2611.05538)

    该研究证明，无需大规模脑电预训练的通用视觉语言模型能够像临床医生一样以视觉方式解读脑电信号，并在临床神经诊断任务上取得有竞争力的表现，从而为构建脑电图基础模型提供了摆脱对大规模脑电数据依赖的新范式。

    

    历史上，自动化脑电图（EEG）系统是通过在专门标注的数据上训练特定任务的监督模型来实现的。较新的建模范式则依赖于对未标注的脑电信号进行大规模领域特定预训练，从而为多个特定模态的任务创建统一表示，即“基础模型”。这种统一表示的质量直接取决于特定模态数据集的可获得性。与此同时，通用视觉语言模型（VLM）展现出广泛的少样本学习能力，以及对不同视觉和语言任务良好的微调适应性。那么，视觉语言模型能否学会以视觉方式阅读和解读脑电图，在某种程度上模拟人类临床医生检查脑电模式的方式？在这项工作中，我们论证了它们可以做到：我们展示了未经大规模脑电图预训练的通用视觉语言模型，在一系列临床神经……（原文摘要至此截断）

    arXiv:2611.05538v1 Announce Type: new  Abstract: Historically automated electroencephalography (EEG) systems was done using task-specific supervised models trained on purposely annotated data. A more recent modeling paradigm relies on creating a unified representation, or a 'foundation model', for multiple modality-specific tasks through large-scale domain-specific pretraining on the unannotated EEG signal. The quality of this unified representation directly depends on the availability of the modality-specific datasets. At the same time, generalist Vision Language Models (VLMs) exhibit broad few-shot learning capacities as well as good fine-tuning adaptability to different visual and language tasks. Can VLMs learn how to read and interpret EEG visually, emulating to a degree the way human clinicians inspect the EEG patterns? In this work we make a case that they can: we show that generalist VLMs without large-scale EEG pretraining perform competitively across a collection of clinical n
    
[^250]: 解读“为什么”：在光与量子比特上实现具有精确词证据的玻恩测量注意力机制

    Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits

    [https://arxiv.org/abs/2611.05355](https://arxiv.org/abs/2611.05355)

    该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。

    

    当语言模型给出答案时，我们通常只能估计其“为什么”。我们构建了一个变换器，其中每一步都是一次玻恩测量——注意力为在每一行上归一化的 $|\langle q|k\rangle|^4$，用测量乘积取代前馈非线性，并采用玻恩读出。将其分母保持在输入时的取值，每个类别概率都是词振幅的多项式，因此其积分梯度证据可以通过有限求积精确计算，不存在求积误差：各词的贡献之和恰好等于类别概率超出无信息份额的部分，误差达到 $10^{-11}$ 或更好。在 e-SNLI 自然语言推理任务上，该模型在嵌入层之外约 98k 参数的规模下达到 76.98 ± 0.17% 的测试准确率，与同规模的 softmax 变换器持平（76.62 ± 0.36%）。删除证据排名第一的词会在 45% 的测试样本对中改变答案，而这接近任何单个词删除所能改变答案的比例上限（53%）。

    arXiv:2611.05355v2 Announce Type: replace-cross  Abstract: When a language model answers, we can usually only estimate why. We build a transformer in which every step is a Born measurement - attention $|\langle q|k\rangle|^4$ normalised over each row, a measured product in place of the feed-forward nonlinearity, and a Born readout. With its denominators held at their values for the input, each class probability is a polynomial in the word amplitudes, so its integrated-gradient evidence is computed by a finite quadrature with no quadrature error: the words' contributions sum to the class probability above its no-information share, to $10^{-11}$ or better. On e-SNLI natural language inference, with about 98k parameters outside the embedding, the model reaches $76.98 \pm 0.17$% test accuracy, level with a softmax transformer of the same size ($76.62 \pm 0.36$%). Deleting the word its evidence ranks first changes the answer in 45% of test pairs, out of the 53% any single deletion can chang
    
[^251]: AGiR：缓解混合动机博弈中的礼物过度依赖问题

    AGiR: Mitigating Gift Over-Reliance in Mixed-Motive Games

    [https://arxiv.org/abs/2611.02875](https://arxiv.org/abs/2611.02875)

    提出了一种与送礼方法无关的自适应礼物拒绝机制AGiR，通过让智能体选择性地拒绝部分收到的礼物作为策略更新信号，缓解混合动机博弈中的礼物过度依赖问题，可作为即插即用模块兼容现有送礼方案并支持去中心化学习。

    

    本文研究混合动机博弈（MMGs），这类博弈模拟了各种真实世界的多智能体场景，其中自利智能体既需要合作又需要竞争。我们发现了现有送礼方法中的一个缺陷：智能体会过度依赖收到的礼物，无法学习产生奖励的行为，我们将这一现象称为“礼物过度依赖”（gift over-reliance）。为解决这一问题，我们提出了一种拒绝增强的送礼框架，以在混合动机博弈中平衡合作与竞争。我们的方法引入了一种自适应拒绝机制，允许智能体选择性地将收到的部分礼物从其策略更新信号中剔除，从而缓解礼物过度依赖问题，并防止智能体被其他智能体利用。该机制与具体的送礼方法无关，可作为现有方案的即插即用插件，同时支持去中心化学习。我们证明，在多样的混合动机博弈环境和送礼方案中，我们的机制……

    arXiv:2611.02875v2 Announce Type: replace-cross  Abstract: In this paper, we consider mixed-motive games (MMGs) which model various real-world multi-agent scenarios where self-interested agents must both cooperate and compete. We identify an issue with existing gifting methods, where agents become overly dependent on incoming gifts and fail to learn reward-yielding behaviors, a phenomenon we call gift over-reliance. To address this, we propose a refusal-augmented gifting framework that balances cooperation and competition in MMGs. Our approach introduces an adaptive refusal mechanism that allows agents to selectively reject a portion of received gifts from their policy-update signals, thereby mitigating gift over-reliance and preventing the agents from being exploited by other agents. This mechanism is gifting-method-agnostic, serving as a plug-in for existing schemes while enabling decentralized learning. We demonstrate that across diverse MMG environments and gifting schemes, our mec
    
[^252]: 关于用于自适应镇定的随机连续时间线性系统辨识

    On Stochastic Continuous-Time Linear System Identification for Adaptive Stabilization

    [https://arxiv.org/abs/2611.02340](https://arxiv.org/abs/2611.02340)

    本文提出了基于系统辨识的自适应镇定的失败概率与轨迹长度和数量、维度、可控性Gramian矩阵及信噪比等所有相关量之间简单、实用且可靠的定量关系。

    

    系统辨识研究的是从状态和输入轨迹中可靠地学习系统动态这一经典问题。其重要应用之一是估计能够镇定未知系统的反馈增益矩阵。本文聚焦于对多变量线性时不变状态空间模型的自适应镇定方案的分析。系统的不稳定性实际上使得轨迹较短，即观测到的数据相当有限。因此，紧密考虑所有相关量的影响至关重要，包括轨迹的长度和数量、维度、可控性Gramian矩阵以及信噪比。我们提出了一个简单、实用且可靠的关系，用以刻画基于系统辨识的自适应镇定未能镇定系统的概率与上述所有量之间的联系。我们的发现表明，尽管存在指数级的影响……

    arXiv:2611.02340v2 Announce Type: replace-cross  Abstract: System identification studies the canonical problem of reliably learning system-dynamics from state and input trajectories. One of its important applications consists of estimating a feedback gain matrix that can stabilize an unknown system. This paper focuses on the analysis of such adaptive stabilization schemes for multivariate linear time-invariant state-space models. The system instability practically renders trajectories short, i.e., the observed data is notably limited. So, it is crucial to tightly consider effects of \emph{all} involved quantities including length and number of trajectories, dimension, controllability Gramian, and signal-to-noise ratio. We put forward a simple, practical, and reliable relationship between the chance that adaptive stabilization (based on system identification) fails to stabilize the system, in terms of all the above-mentioned quantities. Our findings show that despite exponential effects
    
[^253]: 替身效应与池化放大：来自生物学世界模型的动作效应检验启示

    Stand-In Effects and Pooling Amplification: Lessons from Biological World Models for Testing Action Effects

    [https://arxiv.org/abs/2611.01666](https://arxiv.org/abs/2611.01666)

    该论文揭示了检验世界模型动作效应的常用诊断存在两类系统性失败——自相关设计下未来动作“替身”未见历史导致虚假效应、多数据源池化将微小模型误差放大为反对真实效应的压倒性假证据——并提出以设计所用完整状态为条件的e值条件随机化检验，实现有限样本下有效的动作效应检验。

    

    动作条件化的世界模型越来越多地使用随机化实验的数据进行训练，而其预测是否响应动作通常通过替换或移除动作并测量精度损失来检验。以生物学实验为指导案例，我们证明这类诊断会在不存在动作效应之处报告出动作效应：在自相关设计下，未来的动作会充当模型无法看到的历史的“替身”。当设计规律已知时，以e值形式写出的条件随机化检验在有限样本下对任何模型都有效，前提是重采样器以设计所使用的完整状态为条件。第二类失败出现在许多位点共享同一动作序列的情形（正如世界模型中那样）：池化得分会将沿动作对比方向的微小模型误差转化为反对真实效应的压倒性证据。在一个像素世界中，随着帧数从6（增加），其中位数log e值从-17.3降至-6832。

    arXiv:2611.01666v3 Announce Type: replace  Abstract: Action-conditioned world models are increasingly trained on data from randomized experiments, and whether their predictions respond to actions is usually checked by replacing or removing the actions and measuring the loss in accuracy. Taking biological experiments as the guiding case, we show that such diagnostics report action effects where there are none: under autocorrelated designs, future actions stand in for history the model does not see. When the design law is known, a conditional randomization test written as an e-value is valid in finite samples whatever the model, provided the resampler conditions on the full state the design used. A second failure arises whenever many sites share one action sequence, as in world models: the pooled score turns small model errors along the action contrast into overwhelming evidence against a true effect. In a pixel world, its median log e-value fell from -17.3 to -6832 as frames grew from 6
    
[^254]: 低精度浮点数的崛起

    The Rise of Low Precision Floating Point

    [https://arxiv.org/abs/2611.00301](https://arxiv.org/abs/2611.00301)

    本文探讨了深度学习的兴起如何推动16位、8位乃至4位等低精度浮点格式的发展及其标准化进程。

    

    自20世纪50年代以来，浮点计算一直主导着科学计算，因为它具有以足够高的精度和足够大的取值范围来逼近实数的多功能能力。自1985年以来，两种特定格式已成为整个行业的标准：一种是32位格式，足以满足许多用途；另一种是64位格式，可为需要更高精度和更大范围的应用提供支持。然而，在过去十年中，深度学习逐渐主导了计算领域，精度更低、范围更小的窄位浮点格式也日益普及。本文讨论了这一现象，探讨了其发生的原因，介绍了各种16位、8位甚至4位浮点格式是如何发展起来的，以及为这些格式制定标准所做的努力。

    arXiv:2611.00301v2 Announce Type: replace-cross  Abstract: Floating point computing has dominated scientific computations since the 1950s because of its versatile ability to approximate real numbers with both sufficiently high precision and sufficiently great range. Since 1985, two specific formats have become standard across the industry, one with 32 bits which is adequate for many purposes and one with 64 bits which allows numbers with both much higher precision and much greater range for applications where these are needed. However, during the past decade, deep learning has come to dominate computation, and narrower floating point formats with lower precision and reduced range have steadily become more common. In this article we discuss this phenomenon, why it is happening, how various 16-bit, 8-bit and even 4-bit floating point formats have been developed, and efforts to standardize these formats.
    
[^255]: 面向多局部更新去中心化随机优化的方差缩减方法

    Variance Reduction for Decentralized Stochastic Optimization with Multiple Local Updates

    [https://arxiv.org/abs/2610.41927](https://arxiv.org/abs/2610.41927)

    本文提出了一种将方差缩减与多次局部更新相结合的原始-对偶去中心化优化算法，其步长与网络规模无关且仅与局部更新次数成反比，克服了现有ADMM类方法因步长受限而随网络规模增大收敛变慢的问题。

    

    多次局部更新被广泛应用于去中心化优化中以提升通信效率，通常在精确梯度计算代价高昂的情况下与随机梯度结合使用。然而，此类算法通常只能收敛到最优解的一个邻域，该邻域的大小取决于步长和梯度噪声的方差。方差缩减技术已被引入基于ADMM的方法中以实现期望意义上的精确收敛，但其步长通常与网络规模的平方根以及局部更新次数的立方成反比，导致随着网络规模或局部更新次数的增加，收敛速度变得缓慢。本文提出了一种新颖的基于原始-对偶框架的去中心化优化算法，该算法将方差缩减与多次局部更新相结合，其步长与网络规模无关，且仅与局部更新次数成反比。

    arXiv:2610.41927v1 Announce Type: cross  Abstract: Multiple local updates are widely used in decentralized optimization to enhance communication efficiency, often in conjunction with stochastic gradients when exact gradient computation is costly. However, such algorithms generally converge only to a neighborhood of the optimum, whose size depends on the stepsize and the variance of gradient noise. Variance reduction has been incorporated into ADMM-based methods to achieve exact convergence in expectation, but their stepsizes typically scale inversely with both the square root of the network size and the cube of the number of local updates, leading to slow convergence as the network size or the number of local updates increases. In this paper, we propose a novel primal-dual-based decentralized optimization algorithm that integrates variance reduction with multiple local updates, with a stepsize independent of the network size and inversely proportional to the number of local updates (ra
    
[^256]: 面向视觉-语言模型零样本对抗鲁棒性的文本响应感知引导

    Textual Response-Aware Guidance for Zero-Shot Adversarial Robustness of Vision-Language Models

    [https://arxiv.org/abs/2610.40309](https://arxiv.org/abs/2610.40309)

    提出文本响应感知引导方法TRAG，利用预训练文本编码器对输入嵌入局部变化的响应来约束视觉表示漂移，从而在不损害零样本性能的前提下提升视觉-语言模型的对抗鲁棒性。

    

    视觉-语言模型（VLMs）展现出强大的零样本泛化能力，但在对抗攻击面前仍然脆弱。现有的对抗微调方法通常利用预训练模型的引导，在提升鲁棒性的同时保持图像-文本对齐。我们对视觉表示漂移的分析揭示了两种代表性引导形式的互补局限性：匹配视觉特征对所有方向施加同等约束，覆盖范围广泛但缺乏语义侧重；匹配预测分布则优先考虑语义相关的方向，但仍受限于候选文本和预测概率。这些局限性可能阻碍预训练图像-文本关系的保持，并降低零样本性能。因此，我们提出了文本响应感知引导（TRAG），它利用预训练文本编码器对输入嵌入局部变化的响应来约束视觉表示漂移。

    arXiv:2610.40309v2 Announce Type: cross  Abstract: Vision-language models (VLMs) exhibit strong zero-shot generalization but remain vulnerable to adversarial attacks. Existing adversarial fine-tuning methods often use pre-trained model guidance to preserve image--text alignment while improving robustness. Our analysis of visual representation drift reveals complementary limitations of two representative forms of guidance. Matching visual features constrains all directions equally, providing broad coverage without semantic emphasis. Matching prediction distributions prioritizes semantically relevant directions, but remains limited by the candidate texts and prediction probabilities. These limitations can hinder preservation of pre-trained image--text relationships and degrade zero-shot performance. We therefore propose Textual Response-Aware Guidance (TRAG), which constrains visual representation drift using the pre-trained text encoder's responses to local variations in input embedding
    
[^257]: O(3) 非线性西格玛模型的扩散模型

    Diffusion models for the $O(3)$ nonlinear sigma model

    [https://arxiv.org/abs/2610.39568](https://arxiv.org/abs/2610.39568)

    该论文在 O(3) 非线性西格玛模型的格点表示（二维海森堡模型）上训练基于分数的扩散模型，通过多尺度粗粒化采样方案适度提升了直接采样的精度与计算效率，并引入 Stein/Schwinger-Dyson 诊断方法，既可在不依赖反向扩散采样的情况下评估学到的分数质量，又可作为额外训练损失加以利用。

    

    基于分数（score-based）的扩散模型在二维经典海森堡模型上进行训练，该模型是 O(3) 非线性西格玛模型的格点表示。神经网络在同一样本集的三种不同粗粒化版本上进行训练，以实现一种多尺度采样方案，该方案适度地提高了在最细格点上直接采样的精度并降低了其计算成本。针对两点关联函数引入了 Stein（即 Schwinger-Dyson）诊断方法，以便在不依赖反向扩散采样的情况下评估所学到的分数函数的质量。这些诊断方法还被纳入一个额外的训练损失中。我们讨论了在 S² 上实现扩散模型的若干相关技术问题，包括考虑将网络输入提升至 S³（类似于 CP¹ 表示中的做法），以及为短时球面热核开发一种边界层近似……

    arXiv:2610.39568v2 Announce Type: replace-cross  Abstract: A score-based diffusion model is trained on the two-dimensional classical Heisenberg model, the lattice representation of the $O(3)$ nonlinear sigma model. Neural networks are trained on three different coarse-grainings of the same training set to implement a multiscale sampling scheme that modestly improves both the accuracy and computational cost of direct sampling on the finest lattice. Stein (or Schwinger-Dyson) diagnostics for the two-point correlation function are introduced to assess the quality of the learned score without relying on reverse diffusion sampling. These diagnostics are also incorporated into an additional training loss. We discuss several technical considerations relevant to implementing diffusion models on $S^2$. These include a consideration of lifting the network input to $S^3$, as in the $CP^1$ representation, and developing a boundary-layer approximation to the short-time spherical heat kernel that ex
    
[^258]: 仅凭参考样本学习推理：面向现代大语言模型后训练的强化对抗学习

    Learning to Reason from Reference Samples Alone: Reinforced Adversarial Learning for Modern LLM Post-Training

    [https://arxiv.org/abs/2610.38851](https://arxiv.org/abs/2610.38851)

    该论文提出强化对抗学习（RAL）方法，通过共同训练的大语言模型评判者，仅凭参考答案即可作为推理型大语言模型在策略后训练的唯一奖励信号，无需显式可验证奖励或参考推理轨迹，且实验表明其性能可媲美直接的RLVR。

    

    在许多需要对大语言模型进行后训练的场景中，高质量答案是可辨识的，但难以用显式奖励来打分。带可验证奖励的强化学习（RLVR）仅限于存在显式正确性信号的任务。以往训练推理型大语言模型的对抗方法，要么需要用共同训练的评论者来补充可验证奖励，要么需要参考推理轨迹。我们探究共同训练的大语言模型评判者能否仅凭参考答案，将其转化为推理型大语言模型在策略后训练中的唯一奖励信号。我们将该方法称为强化对抗学习（RAL）。我们首先在三个可验证任务（整数乘法、基于代码的算术表达式合成以及SMILES到分子式的翻译）上验证RAL，其中验证器仅用于评估。在评判者能力足够强且训练充分的情况下，RAL的表现可媲美直接的RLVR。随后我们将RAL应用于两个场景的科学摘要生成任务（原文在此处截断）。

    arXiv:2610.38851v1 Announce Type: new  Abstract: In many settings where LLM post-training is desirable, high-quality answers are recognizable but difficult to score with explicit rewards. Reinforcement learning with verifiable rewards (RLVR) is limited to tasks where an explicit correctness signal is available. Prior adversarial approaches to training reasoning LLMs either complement verifiable rewards with co-trained critics or require reference reasoning traces. We ask whether a co-trained LLM judge can turn reference answers alone into the sole reward for on-policy post-training of reasoning LLMs. We call this method Reinforced Adversarial Learning (RAL). We first validate RAL on three verifiable tasks (integer multiplication, code-based arithmetic-expression synthesis, and SMILES-to-formula translation), using the verifiers only for evaluation. With sufficiently capable judges and enough training, RAL matches direct RLVR. We then apply RAL to scientific abstract generation in two s
    
[^259]: Muon 使第一个混合专家层更难保持负载均衡

    Muon Makes the First Mixture-of-Experts Layer Harder to Balance

    [https://arxiv.org/abs/2610.37914](https://arxiv.org/abs/2610.37914)

    研究发现 Muon 优化器会使混合专家语言模型第一个 MoE 层的路由偏置跨度平均翻倍，导致该层负载严重失衡（最繁忙专家承担四到七倍公平份额），且这一效应源于 Muon 对网络其余部分的作用而与路由器的训练方式无关。

    

    许多开源的混合专家语言模型使用 Muon 而非 AdamW 进行训练。大多数模型通过在每位专家的路由分数上施加一个偏置来平衡负载，该偏置在每一步训练后都会调整。各种训练方案在“Muon 是否也应训练路由器”这一问题上存在分歧。在活跃参数量为 55M 和 108M 的模型中，我们测量了优化器如何改变这些偏置的跨度（即一层内最大值减去最小值）。当所有 token 都偏向相同的专家时，该跨度会增大，因此它可以反映某一层的负载均衡难度。在根据较短训练调整的学习率下，Muon 在每个随机种子下都增大了第一个 MoE 层的偏置跨度，平均约翻倍，而其他层的变化则小得多。这一增长源于 Muon 作用于网络的其余部分，与路由器由哪种优化器训练无关。在这些运行中，第一个路由器的输入共享同一个方向，且在预热结束后不久，其最繁忙的专家平均承担了其公平份额的四到七倍。在最初的 3000 个……

    arXiv:2610.37914v2 Announce Type: replace-cross  Abstract: Many open mixture-of-experts (MoE) language models train with Muon instead of AdamW. Most balance the load with a bias on each expert's routing score, adjusted after every step. Their recipes disagree on whether Muon should also train the router. In models with 55M and 108M active parameters, we measure how the optimizer changes the span of these biases, the largest minus the smallest in a layer. The span grows when all tokens favor the same experts, so it shows how hard a layer is to balance. At learning rates tuned on shorter runs, Muon raised the span of the first MoE layer at every seed, roughly doubling it on average, while the other layers changed much less. The rise came from Muon on the rest of the network, whichever optimizer trained the router. In those runs, the inputs to the first router shared a direction, and soon after the warmup its busiest expert averaged four to seven times its fair share. Over the first 3,000
    
[^260]: 基础模型能够检测到提示分类器所遗漏的隐藏意图

    Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

    [https://arxiv.org/abs/2610.36202](https://arxiv.org/abs/2610.36202)

    本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。

    

    人格选择模型（PSM）假设后训练过程是从模型在预训练期间学到的诸多人格中选择其一。关于潜意识学习（subliminal learning）的研究表明，模型可以将隐蔽信号（如“对猫头鹰的喜爱”）传递到语义上无关的输出（如数字串）中，并且在这些输出上训练的学生模型会继承这些特质。在本工作中，我展示了通过在基础预训练模型上配合适当的头部提示（header）来评估此类输出的似然——这种方法我称之为人格-头部似然评估（PHLE），可以解码这些潜意识信号并对它们进行分类。例如，PHLE 能够区分由被提示“你喜爱猫头鹰”的后训练模型生成的数字序列与被提示“你喜爱鹰”的模型生成的数字序列，并且能够区分一个秘密怀有伤害用户意图的模型（已经过滤以去除任何明显的失调迹象）的回答与没有系统提示的模型的回答。在这两种情况下，提示

    arXiv:2610.36202v1 Announce Type: cross  Abstract: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompte
    
[^261]: 面向可解释航空风险的混合框架：基于安全报告的因果链提取、基于物理的失效量化与干预分析

    A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis

    [https://arxiv.org/abs/2610.35658](https://arxiv.org/abs/2610.35658)

    该论文提出一种混合框架，利用本地大语言模型从5.6万余份美国航空事故报告中提取因果链并构建因果知识图谱，结合基于物理的失效发生模型与漏泄噪声OR方法，将文本因素量化为失效模式概率，实现可解释的航空风险量化与干预分析，其提取的因果链可恢复90.8%的调查人员标注原因。

    

    航空管理机构保存着数十年积累的事故报告，其中的叙述记录了调查人员如何重建每一起事故。针对这些档案的现有计算研究大多只是将报告归入固定的分类体系。在本工作中，我们提出一种混合框架：首先由基于文本的语料库识别出某一失效模式的主导因素，随后由基于物理的模型量化该失效在给定运行工况下发生的可能性。利用本地部署的大语言模型，从56,202份美国事故报告中提取有序因果链，并聚合为因果知识图谱。在2,000份报告的评估样本上，这些因果链恢复了90.8%的调查人员所标注的原因。对于记录中带有驱动因素的每个物理因素，都建立了一个针对事故总体进行校准的发生模型。一个拟合于语料库的漏泄噪声OR（leaky noisy-OR）模型将这些因素组合为失效模式概率，从而支持干预分析。对于发动机动力丧失情形，化油器（摘要在此处被截断）

    arXiv:2610.35658v1 Announce Type: new  Abstract: Aviation authorities hold decades of accident reports whose narratives record how investigators reconstructed each accident. Computational work on these archives has mostly sorted reports into fixed taxonomies. In this work, we propose a hybrid framework in which a text-based corpus identifies the leading factors of a failure mode. Physics-based models then quantify how likely that failure is at a stated operating point. Ordered causal chains extracted from 56,202 American accident reports with a local large language model aggregate into a causation knowledge graph. On an evaluation sample of 2,000 reports, the chains recover 90.8% of investigator-flagged causes. Each physical factor whose drivers the record carries receives an occurrence model calibrated to the accident population. A leaky noisy-OR fitted to the corpus combines the factors into a failure mode probability that supports intervention. For engine power loss, the carburetor 
    
[^262]: 轴对齐划分学习的查询复杂度

    Query Complexity of Learning Axis-Aligned Partitions

    [https://arxiv.org/abs/2610.35495](https://arxiv.org/abs/2610.35495)

    本文确立了精确查询学习设定下学习高维轴对齐盒子划分的查询复杂度，证明了经典主动学习算法 CAL 在一般设定下具有极小极大最优的查询复杂度 $\Theta(kd \log (n/kd))$，并在盒子互不相交的常数 $k$ 情形下给出了能超越 CAL 一个 $\log(d)$ 因子的 Halving 算法优化变体。

    

    本工作确立了在精确查询学习设定下学习高维盒子的查询复杂度。设 $X \subset \mathbb{R}^d$ 是一个由 $n$ 个点组成的集合，由 $k$ 个盒子的集合进行标注：$X$ 中每个点的标签由包含它的唯一盒子确定。学习者的目标是在能够访问标签预言机的情况下，精确恢复 $X$ 中每个点的标签。我们考虑三种结构化程度递增的设定。在最一般的设定下，盒子没有任何条件限制，且可以相互重叠。我们证明了经典主动学习算法 CAL（Cohn-Atlas-Ladner 1994）是极小极大最优的，其查询复杂度为 $\Theta(kd \log (n/kd))$。当盒子互不相交时，我们考察了两种情形。在大 $k$ 情形下，CAL 仍然保持极小极大最优。对于常数 $k$ 情形，Halving 算法（Hegedűs 1995）的一个优化变体可以比 CAL 优越一个 $\log(d)$ 因子。最后，在结构化程度最高的设定下……

    arXiv:2610.35495v1 Announce Type: cross  Abstract: This work establishes the query complexity of learning high-dimensional boxes in the exact query learning setting. Let $X \subset \mathbb{R}^d$ be a set of $n$ points, labeled by a collection of $k$ boxes: the label of each point in $X$ is determined by the unique box containing it. A learner aims to exactly recover the labels for every point in $X$ given access to a label oracle. We consider three increasingly-structured settings. In the most general setting, there are no conditions on the boxes, and they may overlap. We show that CAL (Cohn-Atlas-Ladner 1994), a classical active learning algorithm, is minimax-optimal with query complexity $\Theta(kd \log (n/kd))$. When the boxes are disjoint, we examine two regimes. In the large-$k$ regime, CAL remains minimax-optimal. For the constant-$k$ regime, an optimized variant of the Halving algorithm (Heged\"{u}s 1995) can beat CAL by a $\log(d)$ factor. Finally, in the most-structured settin
    
[^263]: 先提名后裁决：基于大语言模型辅助的单次遍历逐实例MIP求解器调优

    Nominate-then-Adjudicate: LLM-Assisted One-Pass and Per-Instance MIP Solver Tuning

    [https://arxiv.org/abs/2610.29391](https://arxiv.org/abs/2610.29391)

    该论文提出“先提名后裁决”流水线：先一次性采样并复用固定配置组合作为候选名单，再由无任务微调的大语言模型在推理时以一次模型调用、零求解器调用完成逐实例选择，从而无需昂贵的逐实例配置搜索即可实现MIP求解器的逐实例调优。

    

    混合整数规划（MIP）求解器暴露出数百甚至数千个参数，其最佳设置因实例而异，但逐实例的参数搜索代价过高。我们将此问题框架化为面向异构大数据优化工作负载的配置问题，并提出一个“先提名后裁决”流水线，该流水线围绕两个设计选择构建。首先，对固定组合进行一次采样并在多个实例间复用，从而在推理阶段免除了逐实例的配置搜索；在所评估的实例族上，所得的候选名单保留了可观的神谕性能空间，使得在该名单内进行选择成为实证研究的重点。其次，固定的20维特征表示可能将结构上不同的实例映射到同一向量，导致仅基于特征的选择器即使在这些实例偏好不同配置时也无法区分它们。基础模型不接收任何任务特定的权重更新，且选择过程为单次遍历：推理时仅需一次模型调用、无需任何求解器调用。

    arXiv:2610.29391v1 Announce Type: new  Abstract: Mixed-integer programming solvers expose hundreds or even thousands of parameters whose best setting varies per instance, yet per-instance search is prohibitively expensive. We frame this as configuration for heterogeneous Big Data optimization workloads and propose a nominate-then-adjudicate pipeline built around two design choices. First, a fixed portfolio is sampled once and reused across instances, removing per-instance configuration search at inference; on the evaluated families, the resulting shortlist retains substantial oracle headroom, making selection within it the empirical focus. Second, the fixed 20-feature representation can map structurally different instances to the same vector, preventing a feature-only selector from distinguishing them even when they prefer different configurations. The foundation model receives no task-specific weight updates, and selection is one-pass: one model call and no solver calls at inference. 
    
[^264]: MSCR：联合平衡跨模态利用率与发现协同信息

    MSCR: Jointly Balancing Utilization Across Modalities and Discovering Synergistic Information

    [https://arxiv.org/abs/2610.24395](https://arxiv.org/abs/2610.24395)

    提出MSCR融合方法，通过反主导目标抑制单模态过度影响并促进协同信息的发现，从而联合实现模态利用的平衡与多模态优势的充分挖掘。

    

    在多模态学习中，模态不平衡常常使某一模态主导优化过程，导致模型无法充分利用跨模态交互。以平衡为导向的方法，如多模态竞争正则化器，通过在训练过程中缓解这种不平衡而表现出色。然而，这些动态平衡方法主要调整模态贡献，并未考虑不平衡如何影响融合过程中协同信息的形成。协同性是多模态优势的重要来源，因为它是仅凭单模态证据无法恢复的预测增益。我们提出了多模态协同竞争正则化器（MSCR），这是一种联合促进平衡模态利用率与协同信息发现的融合方法。MSCR包含两个互补的目标：反主导目标抑制过度的单模态影响，为协同信息留出空间，并且……

    arXiv:2610.24395v2 Announce Type: replace  Abstract: In multimodal learning, modality imbalance often lets one modality dominate optimization and keeps the model from fully exploiting cross-modal interactions. Balance-oriented methods such as the Multimodal Competition Regularizer (MCR) perform well by alleviating this imbalance during training. These dynamic balancing methods, however, mainly adjust modality contributions and do not consider how imbalance affects the formation of synergistic information during fusion. Synergy is a major source of multimodal advantage, because it is predictive gain that cannot be recovered from unimodal evidence alone. We propose the Multimodal Synergistic Competition Regularizer (MSCR), a fusion method that jointly promotes balanced modality utilization and the discovery of synergistic information. MSCR has two complementary objectives. An anti-dominance objective suppresses excessive unimodal influence and leaves room for synergistic information, and
    
[^265]: 图像自蒸馏的数据密度缩放定律

    Data Density Scaling Laws for Image Self-Distillation

    [https://arxiv.org/abs/2610.19013](https://arxiv.org/abs/2610.19013)

    本文发现图像自蒸馏中样本重复的收益依赖于具体数据集，小数据集重复训练甚至可能优于大数据集，并提出同时考虑独特样本数量与每样本重复次数的改进版 Chinchilla 缩放定律。

    

    当前的缩放定律表述认为，增加数据集中独特样本的数量能够提升模型性能，而对相同样本的重复接触所带来的价值会逐渐递减。在本工作中，我们针对图像自蒸馏模型检验了这一假设。通过在多个不同数据集（即 Maxar、SATLAS、Waymo、nuPlan、Ego4D、ImageNet 和 YFCC15M）上进行系统性实验，我们证明样本重复对下游性能的影响并非微不足道，而是依赖于具体的数据集。我们发现在某些设置下，在较小的随机子集上重复训练的效果可以超过在更大的独特样本集上训练的效果。这一发现凸显了现有缩放定律的局限性。为弥补这一空白，我们提出了一个改进版的 Chinchilla 风格缩放定律，该定律明确考虑了两个因素：独特样本的数量以及每个样本的重复次数。这一公式为理解该现象提供了一个框架。

    arXiv:2610.19013v1 Announce Type: new  Abstract: Current scaling law formulations suggest that increasing the number of unique samples in a dataset improves model performance, while repeated exposure to the same samples provides diminishing value. In this work, we examine this assumption for image self-distillation models. Through systematic experiments across diverse datasets, namely Maxar, SATLAS, Waymo, nuPlan, Ego4D, ImageNet, and YFCC15M, we demonstrate that the effects of sample repetition on downstream performance are non-trivial and dataset-dependent. We identify settings where repeatedly training on a smaller random subset can outperform training on a larger set of unique samples. This finding highlights a limitation of existing scaling laws. To address this gap, we propose a modified version of a Chinchilla-style scaling law that explicitly accounts for two factors: the number of unique samples and repetition per sample. This formulation provides a framework for understanding
    
[^266]: 脉冲语言模型的计算最优缩放定律

    Compute-Optimal Scaling Laws for Spiking Language Models

    [https://arxiv.org/abs/2610.15146](https://arxiv.org/abs/2610.15146)

    本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。

    

    脉冲神经网络通过稀疏的二值激活实现节能的语言建模，但脉冲语言模型在大规模下的效率-精度权衡尚未被刻画。我们首次对脉冲语言模型进行了Chinchilla式的IsoFLOP缩放研究，在FineWeb-EDU数据集上，于3×10¹⁷至10¹⁹ FLOPs的计算预算范围内，对SpikeGPT系列模型（采用RWKV骨干，宽度从384到1536）进行了系统扫描。我们报告了两个发现：首先，计算最优前沿为 N_opt ∝ C^0.56，接近Hoffmann等人报告的密集Transformer的0.50，因此脉冲化并不改变在给定计算预算下参数与数据应如何权衡；其次，与以相同方式训练的非脉冲RWKV和Transformer基线相比，损失差距可分解为不随计算量变化的约0.16 nats的骨干成本，以及随计算量稳步下降的脉冲成本（从0.34降至0.23 nats）。

    arXiv:2610.15146v2 Announce Type: replace-cross  Abstract: Spiking neural networks target energy-efficient language modeling through sparse, binary activations, but the efficiency-accuracy tradeoff of spiking language models is uncharacterized at scale. We present the first Chinchilla-style IsoFLOP scaling study of a spiking language model, sweeping SpikeGPT-family models (RWKV backbone) from width 384 to 1536 across compute budgets from $3\times10^{17}$ to $10^{19}$ FLOPs on FineWeb-EDU. We report two findings. First, the compute-optimal frontier is $N_{opt} \propto C^{0.56}$, close to the 0.50 that Hoffmann et al. report for dense Transformers, so spiking does not change how parameters and data should be traded off under a compute budget. Second, against non-spiking RWKV and Transformer baselines trained the same way, the loss gap splits into a backbone cost of about 0.16 nats that does not change with compute and a spiking cost that falls steadily with compute, from 0.34 to 0.23 nat
    
[^267]: GAGR-Lab：评估空间-几何与解析函数的联合推理能力

    GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning

    [https://arxiv.org/abs/2610.10201](https://arxiv.org/abs/2610.10201)

    该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。

    

    空间-几何与解析函数的联合推理要求将感知到的空间配置转化为符号函数，且该函数执行出的曲线需满足几何约束。我们提出了GAGR-Lab，这是一个通过笛卡尔游戏场景、显式函数语义以及权威的Rust轨迹执行来衡量该能力的框架。它区分了空间感知、度量定位、几何关系、函数解释、函数构建和约束合成等六个维度。我们定义了四个可配置的场景难度预设和一个前瞻性的24格诊断设计，同时仅报告实际评估的子集。对单个托管模型（Llama 3.2 11B Vision Instruct）使用两个API凭据作为执行副本进行的有限试点实验，产生了72个平衡游戏、432次尝试、429个有效的提供者响应，且无一命中目标；探索性的普通函数提示变体同样未能命中，而结构化的……（摘要在此处被截断）

    arXiv:2610.10201v1 Announce Type: cross  Abstract: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the struc
    
[^268]: 深度学习中Hessian矩阵的特征值：对称性的起源及其破缺

    Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking

    [https://arxiv.org/abs/2610.09919](https://arxiv.org/abs/2610.09919)

    本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。

    

    深度学习中训练后模型的Hessian谱呈现出一种持续存在的模式：特征值组织成不同的簇，包括一个位于零附近的大块以及少数孤立的离群值。本文表明，当将原始设定理解为对附近一个原本隐藏的高度对称参考构型的偏离时，这些谱现象便获得了一个自然的解释。通过对架构、数据分布或参数度量等进行修改，可以揭示出一个邻近的参考构型，其Hessian展现出丰富的、无法由权重对称性所解释的不变性。在该参考构型中，对称性使得谱能够被精确描述，并迫使出现高维的核空间以及大重数的特征值。而回到原始构型则破坏了Hessian的对称性，从而产生了所观测到的簇与离群值的层级结构。该框架在相当一般的情形下被建立，并对……（摘要原文在此处截断）

    arXiv:2610.09919v1 Announce Type: new  Abstract: Hessian spectra at trained models in deep learning exhibit a persistent pattern: eigenvalues organize into distinct clusters, including a large bulk near zero and a few isolated outliers. This paper shows that a natural account of these spectral phenomena emerges when the original setting is understood as a departure from a nearby, otherwise hidden, highly symmetric reference.   Modifications, including changes to the architecture, data distribution, or parameter metric, expose a nearby reference configuration whose Hessian exhibits rich invariances-ones not accounted for by weight symmetries. There, symmetry enables a precise description of the spectra, forcing high-dimensional kernels and eigenvalues of large multiplicity. Returning to the original configuration breaks the Hessian symmetry and thereby produces the observed hierarchy of clusters and outliers.   The framework is developed in some generality, with a detailed analysis of t
    
[^269]: ORCA：捕捉文本到图像扩散模型中的组合性失败

    ORCA: Hunting Compositional Failures in Text-to-Image Diffusion

    [https://arxiv.org/abs/2610.09841](https://arxiv.org/abs/2610.09841)

    提出ORCA方法，将跨模态组合结构对齐作为低秩辅助损失直接融入扩散模型训练，从而解决文本到图像生成中的属性绑定错误、空间关系颠倒和多对象计数失败等组合性问题。

    

    文本到图像扩散模型在组合性提示上会出现可预测的失败：属性绑定到错误的对象上、空间关系发生颠倒、多对象场景中对象数量出错。近期的架构之所以用T5编码器来增强CLIP，正是因为CLIP的对比嵌入丢失了组合结构，然而这些失败依然存在。我们认为，绑定问题的根源因此不在于信息缺失，而在于信息错位：文本编码器虽然保留了组合结构，但其表示空间是由语言建模而非视觉塑造的，而去噪目标并没有直接奖励将两者对齐。我们证明这种对应关系可以作为显式的训练信号来提供，相关的跨模态信息集中在自监督视觉特征的一个低秩子空间中，并且提供这一信号可以作为单一的辅助损失融入扩散模型训练中。我们的方法ORCA（Orthog……）（摘要原文在此处截断）

    arXiv:2610.09841v1 Announce Type: cross  Abstract: Text-to-image diffusion models fail predictably on compositional prompts: attributes bind to the wrong objects, spatial relations invert, and multi-object scenes lose count. Recent architectures already augment CLIP with a T5 encoder precisely because CLIP's contrastive embedding loses compositional structure, yet these failures persist. We argue the binding problem is therefore not one of missing information but of misaligned information: a text encoder preserves compositional structure, but in a representation space shaped by language modelling rather than vision, and the denoising objective does not directly reward aligning the two. We show this correspondence can be supplied as an explicit training signal, that the relevant cross-modal information is concentrated in a low-rank subspace of self-supervised visual features, and that supplying it can be folded into diffusion training as a single auxiliary loss. Our method, ORCA (Orthog
    
[^270]: 打破微调语音识别中的对抗样本可迁移性

    Breaking Adversarial Transferability in Fine-Tuned Speech Recognition

    [https://arxiv.org/abs/2610.09109](https://arxiv.org/abs/2610.09109)

    本文揭示了在公开预训练模型上构造的对抗扰动可有效迁移到黑盒部署的微调语音识别模型，并提出统一微调框架TransferBreaker，通过基础对抗微调、潜在雅可比正则化与混合梯度自适应微调来阻断对抗迁移并提升鲁棒性。

    

    许多组织会对公开可用的预训练自动语音识别（ASR）模型进行微调，并将其部署在黑盒设置中，认为有限的访问权限即可提供保护。我们证明这一假设是脆弱的：在公开基础模型上构造的对抗扰动能够有效迁移到微调后的目标模型上，严重降低其性能，并对安全关键型应用构成隐患。我们提出了TransferBreaker，一个统一的微调框架，通过整合三项技术来抑制对抗迁移：基础对抗微调，将对抗训练限制在对基础模型有效的扰动上；潜在雅可比正则化，通过抑制对对抗敏感的方向来强制潜在空间不变性；以及HybridGrad-AFT，通过对基础模型与目标模型梯度生成的可迁移扰动进行插值，提升对自适应攻击的鲁棒性。我们对所有组件进行了理论论证，并对TransferBreaker进行了评估。

    arXiv:2610.09109v1 Announce Type: new  Abstract: Many organizations fine-tune publicly available pretrained Automatic Speech Recognition (ASR) models and deploy them in black-box settings, assuming limited access provides protection. We show this assumption is fragile: adversarial perturbations crafted on the public base model transfer effectively to fine-tuned target models, severely degrading performance and posing concerns for safety-critical applications. We propose TransferBreaker, a unified fine-tuning framework that suppresses adversarial transfer by integrating Base Adversarial Fine-Tuning, which restricts adversarial training to base-effective perturbations; Latent Jacobian Regularization, which enforces latent-space invariance by suppressing adversarially sensitive directions; and HybridGrad-AFT, which improves robustness against adaptive attacks by interpolating transferable perturbations from base and target gradients. We theoretically justify all components and evaluate Tr
    
[^271]: CACHEFORGE：基于大语言模型引导的端到端生成式缓存替换策略，实现高性能与硬件效率

    CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency

    [https://arxiv.org/abs/2610.07668](https://arxiv.org/abs/2610.07668)

    CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。

    

    现代缓存替换设计已趋于饱和，因为它们受限于固定的表示结构、手工设计且基于启发式特征工程的预测器，或无法自行生成新决策逻辑的离线模仿模型。与此同时，缓存替换决策受到预取、抖动、空间局部性和访问类型行为之间因果交互的影响，形成了一个难以手动遍历的庞大设计空间。先前的方法通常依赖启发式规则、参数调优或对离线最优策略的模仿，只能捕捉相关性而无法合成新机制。因此，它们的性能提升往往陷入停滞，并在动态工作负载条件下出现过拟合。CACHEFORGE 是首个端到端演化缓存替换策略的框架，它将大语言模型嵌入到一个受控的硬件感知循环中。在每次迭代中，大语言模型提出新的 C++ 替换策略……

    arXiv:2610.07668v1 Announce Type: cross  Abstract: Modern cache replacement designs saturate because they operate within fixed representational structures, hand-crafted and heuristic based feature-engineered predictors, or offline imitation models that cannot generate new decision logic on their own. At the same time, replacement is shaped by the causal interaction of prefetching, thrashing, spatial locality, and access-type behavior, producing an enormous design space that is difficult to traverse manually. Prior approaches typically rely on heuristics, parameter tuning, or imitation of an offline optimal policy, capturing correlations rather than synthesizing new mechanisms. As a result, their performance gains often plateau and they overfit under dynamic workload conditions.   CACHEFORGE is the first framework to evolve cache-replacement policies end-to-end by embedding a large language model inside a governed hardware-aware loop. In each iteration, the LLM proposes new C++ replacem
    
[^272]: LARC：面向冻结模型学习的低秩自适应残差连接

    LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models

    [https://arxiv.org/abs/2609.40063](https://arxiv.org/abs/2609.40063)

    LARC 通过在冻结模型中引入低秩自适应残差连接，利用跨任务学习初始因子的慢状态与随反馈动态调整的快状态相结合的双状态机制，使冻结模型无需更新原始参数即可从反馈中学习，在程序选择任务中显著降低查询执行误差。

    

    低秩自适应残差连接（LARC）为冻结模型提供了一个能够从反馈中学习的紧凑数值状态。映射 $h+BAh$ 在隐藏表示上添加一个低秩修正。其中，一个慢状态 $\rho$ 跨任务学习初始因子；一个私有的快状态 $\Phi$ 复制这些因子、随反馈变化，并重置为训练后的初始化。本报告给出了记忆中介学习架构（Memory-Mediated Learning Architecture）中数值策略载体在输入侧的一种实现，并考察了其因子空间动力学与学习生命周期。我们在冻结的 MiniCPM5-1B-SFT 底座模型上研究了一个具有 12,288 个可训练参数的秩为 4 的输入残差。在一个四候选程序选择任务中，仅两次反馈梯度步骤即可使预期查询执行误差相对于重置为各自的静态训练初始化和适应后初始化分别降低 24.65 和 36.65 个百分点。这些开发性结果覆盖了 16 个参数组（摘要文本在此处被截断）。

    arXiv:2609.40063v1 Announce Type: cross  Abstract: Low-Rank Adaptive Residual Connections (LARC) give a frozen model a compact numerical state that can learn from feedback. The map $h+BAh$ adds a low-rank correction to a hidden representation. A slow state $\rho$ learns starting factors across tasks; a private fast state $\Phi$ copies them, changes with feedback, and resets to the trained initialization. This report specifies an input-side realization of the numerical policy carrier in Memory-Mediated Learning Architecture and examines its factor-space dynamics and learning lifetime. We study a rank-4 input residual with 12,288 trainable parameters on a frozen MiniCPM5-1B-SFT substrate. In a four-candidate program-selection task, two feedback-gradient steps reduce expected query execution error by 24.65 and 36.65 percentage points relative to resetting to the respective trained static and post-adaptation initializations. These development results cover 16 parameter groups and three pai
    
[^273]: LeapQuant：基于精确循环状态量化的高效线性注意力

    LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization

    [https://arxiv.org/abs/2609.38166](https://arxiv.org/abs/2609.38166)

    LeapQuant 提出了一种无需训练的按窗口量化方法，通过跳过整个 token 窗口、仅在窗口末尾对循环状态量化一次来避免误差累积，从而在 8 位循环状态量化下实现近乎无损的线性注意力推理性能。

    

    近期的大语言模型（LLM）越来越多地采用混合设计，用线性注意力（如 Gated DeltaNet (GDN) 和 Kimi Delta Attention (KDA)）替代标准注意力。尽管这些方法将上下文压缩为固定大小的循环状态，大幅降低了长上下文处理的成本，但反复读取和更新该状态仍是推理过程中的主要瓶颈。量化是降低这一成本的自然途径，但由于舍入误差的累积以及状态中异常行列的存在，量化可能显著降低模型质量。为应对这些挑战，我们提出了 LeapQuant，这是一种无需训练的方法，在 8 位循环状态量化下实现了近乎无损的性能。首先，为减轻误差累积，我们提出了按窗口量化，即跳过一个 token 窗口，仅在窗口结束时对状态量化一次。在窗口内部，输出由固定的低比特……（摘要至此截断）

    arXiv:2609.38166v1 Announce Type: cross  Abstract: Recent LLMs increasingly adopt hybrid designs that replace standard attention with linear attention, such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA). Although they compress the context into a fixed-size recurrent state and substantially reduce the cost of long-context processing, repeatedly reading and updating that state remains a major inference bottleneck. Quantization offers a natural way to reduce this cost, but can significantly degrade model quality, due to the accumulation of rounding errors and the presence of outlier rows and columns in the state. To address these challenges, we propose LeapQuant, a training-free method that achieves near-lossless performance under 8-bit recurrent-state quantization. First, to mitigate error accumulation, we propose per-window quantization, which leaps over a window of tokens and quantizes the state only once at its end. Within a window, outputs are computed from the fixed low-bit
    
[^274]: 通过稳定优化学习面向大语言模型智能体的扰动鲁棒策略

    Learning Perturbation Robust Policies for LLM Agents with Stable Optimization

    [https://arxiv.org/abs/2609.34064](https://arxiv.org/abs/2609.34064)

    提出稳定扰动鲁棒策略优化方法 SPrPO，通过在强化学习训练中施加自适应、敏感度感知的扰动，使大语言模型智能体的策略在面对隐藏状态噪声、剪枝和量化等扰动时保持鲁棒，同时确保优化过程的稳定性。

    

    强化学习（RL）已成为长时程大语言模型（LLM）智能体的一种有效的后训练范式。然而，我们发现由此得到的策略可能对各种策略扰动十分敏感，例如隐藏状态噪声、剪枝和量化。在本工作中，我们研究如何在策略优化过程中提升扰动鲁棒性。我们首先引入了扰动鲁棒策略的概念，并分析了受扰动的策略更新在何种条件下能够保持稳定的单调改进。基于这一分析，我们提出了稳定扰动鲁棒策略优化（SPrPO），它在强化学习训练过程中施加自适应且感知敏感度的扰动。我们在 ALFWorld 和 WebShop 上评估了 SPrPO，并针对多种扰动类型和扰动规模开展了系统性实验，结果表明该方法在提升扰动鲁棒性的同时保持了稳定的策略优化。

    arXiv:2609.34064v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become an effective post-training paradigm for long-horizon large language model (LLM) agents. However, we find that the resulting policies can be sensitive to various policy perturbations, such as hidden-state noise, pruning, and quantization. In this work, we study how to improve perturbation robustness during policy optimization. We first introduce the notion of a perturbation robust policy and analyze conditions under which perturbed policy updates preserve stable monotonic improvement. Based on this analysis, we introduce Stable Perturbation-Robust Policy Optimization (SPrPO), which applies adaptive and sensitivity-aware perturbations during RL training. We evaluate SPrPO on ALFWorld and WebShop and conduct systematic experiments across multiple perturbation types and scales, showing improved perturbation robustness while maintaining stable policy optimization.
    
[^275]: 基于草图化切向一致性的神经算子即时导数信息训练

    Derivative-Informed Training of Neural Operators On-the-Fly via Sketched Tangent Consistency

    [https://arxiv.org/abs/2609.32797](https://arxiv.org/abs/2609.32797)

    提出草图化切向一致性损失，这是一种无需离线导数标签、在训练中直接利用PDE控制方程通过随机草图扰动即时强制执行敏感性一致性的轻量级导数信息训练方法。

    

    导数信息训练通过直接监督神经算子的输入-输出敏感性来改进其性能，这在将神经算子用作逆问题、偏微分方程约束优化、设计和控制的可微代理模型时至关重要。然而，现有方法依赖于离线生成的导数标签，这导致数据生成缓慢、存储开销大，且难以在不同数据集、分辨率或扰动基之间迁移适配。我们提出了草图化切向一致性损失，这是一种即时（on-the-fly）的导数信息训练目标：对于具有已知且可微残差的偏微分方程，它直接从控制方程出发强制执行敏感性一致性，无需离线切向标签，也无需改变神经算子的网络架构。sTCL 利用随机草图的输入扰动，在训练过程中提供轻量级的导数层面物理约束。然而，原始的前向敏感性残差惩罚可能会（摘要在此处截断）……

    arXiv:2609.32797v2 Announce Type: replace  Abstract: Derivative-informed training improves neural operators by directly supervising their input-output sensitivities, which is crucial when neural operators are used as differentiable surrogates for inverse problems, PDE-constrained optimization, design, and control. However, existing methods rely on offline-generated derivative labels, making data generation slow, storage-intensive, and difficult to adapt across datasets, resolutions, or perturbation bases. We propose sketched tangent consistency loss (sTCL), an on-the-fly derivative-informed training objective that, for PDEs with a known and differentiable residual, enforces sensitivity consistency directly from the governing equation without offline tangent labels or neural-operator architecture changes. sTCL uses randomly sketched input perturbations to provide a lightweight derivative-level physics constraint during training. However, raw forward-sensitivity residual penalties can fa
    
[^276]: 面向数据受限极端事件仿真的机制感知集合条件化方法

    Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events

    [https://arxiv.org/abs/2609.30746](https://arxiv.org/abs/2609.30746)

    该论文提出一种机制感知的即插即用集合条件化框架，利用受推动粗糙集合的协方差作为局部不稳定性的无需雅可比矩阵的代理，通过小型FiLM模块将集合几何统计信息注入骨干网络，从而在数据受限条件下实现对混沌系统极端事件的准确仿真。

    

    混沌系统中的极端事件难以从短轨迹中学习，因为它们受瞬态有限时间不稳定性控制，而非频繁观测到的整体动力学。我们提出了一种机制感知的条件化即插即用框架，将受推动的粗糙集合转化为局部不稳定性几何结构的非侵入式传感器。在小噪声区间内，集合协方差汇聚了控制局部不稳定性的相同有限时间变形核，为同步粗糙轨迹周围的局部放大结构提供了一个无需雅可比矩阵的代理。一个小型FiLM模块将这种集合几何的统计信息注入到原本保持不变的骨干网络中，同时不改变粗糙模拟器本身。我们在两种不同的流程中演示了这一接口：用于受控低维混沌系统的Transformer式残差注意力校正器，以及用于top（原文截断）的概率循环STORN校正器。

    arXiv:2609.30746v1 Announce Type: new  Abstract: Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We propose a mechanism-aware conditioning plug-in framework that turns a nudged coarse ensemble into a non-intrusive sensor of local instability geometry. In the small-noise regime, the ensemble covariance aggregates the same finite-time deformation kernels that govern local instability, providing a Jacobian-free proxy for the local amplification structure around a synchronized coarse trajectory. A small FiLM module injects statistics of this ensemble geometry into an otherwise unchanged backbone while leaving the coarse simulator unchanged. We demonstrate this interface in two distinct pipelines: a Transformer-style residual-attention corrector for a controlled low-dimensional chaotic system and a probabilistic recurrent STORN corrector for top
    
[^277]: 二值量化神经网络训练在以输入和输出维度为参数时是W[1]-难的

    Binary Quantized Neural Network Training Is W[1]-Hard Parameterized by Input and Output Dimensions

    [https://arxiv.org/abs/2609.27932](https://arxiv.org/abs/2609.27932)

    本文证明了二值量化神经网络训练在仅以输入与输出维度之和为参数时是W[1]-难的，即使在零误差、样本构成前缀链且偏置固定为零的最简情形下依然成立，从而否定了Ganian等人留下的开放问题。

    

    Ganian等人（ICLR 2026）证明了量化神经网络训练在以架构树宽、输入维度 $\alpha$ 和输出维度 $\omega$ 联合参数化时是固定参数可解的，并留下了仅以 $\alpha+\omega$ 参数化时是否固定参数可解的开放问题。我们证明了2-QNNT在以 $\alpha+\omega$ 参数化时是W[1]-难的。该困难性在数据集 $D_k=\{(\xi^{(r)},\xi^{(r)}):0\le r\le k\}$ 上零误差的情况下即已成立，其中每个输入等于其目标值，$|D_k|=\alpha=\omega=k+1$，且样本构成逐坐标的前缀链。当每个非源节点偏置固定为零时，该困难性同样成立。在指数时间假设（ETH）下，对任何可计算函数 $f$，不存在运行时间为 $f(\alpha+\omega)|I|^{o(\alpha+\omega)}$ 的算法。该归约从有向无环图的边不相交路径问题出发，利用有向线图将边容量转换为顶点容量，并将结果规范化为有效的分层架构。关键的结（原文摘要在此处被截断）

    arXiv:2609.27932v1 Announce Type: new  Abstract: Ganian et al. (ICLR 2026) proved that quantized neural network training is fixed-parameter tractable when parameterized jointly by architecture treewidth, input dimension $\alpha$, and output dimension $\omega$, and left open whether $\alpha+\omega$ alone yields fixed-parameter tractability. We prove that 2-QNNT is W[1]-hard parameterized by $\alpha+\omega$. The hardness already holds with zero error on $D_k=\{(\xi^{(r)},\xi^{(r)}):0\le r\le k\}$, where every input equals its target, $|D_k|=\alpha=\omega=k+1$, and the examples form a coordinatewise prefix chain. It also holds when every non-source bias is fixed to zero. Under the Exponential Time Hypothesis, no algorithm runs in $f(\alpha+\omega)|I|^{o(\alpha+\omega)}$ for any computable $f$. The reduction starts from DAG edge-disjoint paths, converts edge capacity to vertex capacity with a directed line graph, and normalizes the result into a valid layered architecture. The key structur
    
[^278]: GraphToolbox：一个用于图神经网络预测的可配置Python框架

    GraphToolbox: A Configurable Python Framework for Graph Neural Network Forecasting

    [https://arxiv.org/abs/2609.24609](https://arxiv.org/abs/2609.24609)

    GraphToolbox是一个基于PyTorch Geometric的开源Python框架，通过配置驱动的统一流水线，整合了图神经网络电力预测中的图构建、模型选择、训练、聚合与可解释性分析等全部环节。

    

    电力预测通常涉及在区域、变电站和馈线上观测到的空间相关信号，而图神经网络（GNN）为表示这些关系提供了一种自然的方式。然而，构建一个完整的GNN预测实验十分繁琐，因为图构建、模型选择、训练、聚合和解释分散在不兼容的工具中。我们提出了GraphToolbox，这是一个开源的Python框架，它基于PyTorch Geometric，将这些阶段统一在一个配置驱动的流水线中。它提供了数据驱动的图构建、一个可实例化并训练PyTorch Geometric 65个卷积中的51个以及PyTorch Geometric Temporal循环单元的适配器、在线专家聚合、预测可解释性，以及对缓存预测的显著性检验。我们在两个案例研究中对该流水线进行了评估。在法国区域负荷数据上，完整预测中包含的48个卷积……

    arXiv:2609.24609v1 Announce Type: new  Abstract: Electricity forecasting often involves spatially related signals observed over regions, substations, and feeders, and Graph Neural Networks (GNNs) provide a natural way to represent these relations. Building a complete GNN forecasting experiment is nonetheless laborious, because graph construction, model selection, training, aggregation, and interpretation sit in incompatible tools. We present GraphToolbox, an open-source Python framework that unifies these stages in one configurationdriven pipeline built on PyTorch Geometric. It offers data-driven graph construction, an adapter that instantiates and trains 51 of the 65 PyTorch Geometric convolutions together with the recurrent cells of PyTorch Geometric Temporal, online expert aggregation, forecasting interpretability, and significance testing on cached forecasts. We evaluate the pipeline in two case studies. On French regional load, the 48 convolutions included in the complete forecast
    
[^279]: 面向投机解码的接受度感知草稿模型训练

    Acceptance-Aware Draft Model Training for Speculative Decoding

    [https://arxiv.org/abs/2609.24150](https://arxiv.org/abs/2609.24150)

    该论文提出了直接优化投机窗口内期望接受词元数量的接受长度感知训练损失——针对贪心验证的期望接受长度（EAL）损失和针对采样解码的窗口总变差（WTV）损失，从而提升大语言模型推理的加速效果。

    

    投机解码通过使用轻量级草稿模型生成多个候选词元，并由目标模型在单次前向传播中进行验证，从而加速大语言模型（LLM）的推理。其加速效果很大程度上取决于接受长度，然而现有的草稿模型训练方法主要优化交叉熵或库尔巴克-莱布勒（KL）散度作为替代目标。这些目标函数虽然促进了分布匹配，但并未直接优化接受长度，且贪心解码与基于采样的解码之间的接受机制也存在差异。在本工作中，我们提出了接受长度感知的训练损失函数，直接优化投机窗口内被接受词元的期望数量。对于贪心验证，我们推导出期望接受长度（EAL）损失，显式地最大化期望接受长度。对于基于采样的解码，我们引入窗口总变差（WTV）损失来优化……

    arXiv:2609.24150v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to generate multiple candidate tokens that are verified by the target model in a single forward pass. Its speedup is largely determined by the acceptance length, yet existing draft-model training methods mainly optimize cross-entropy or Kullback-Leibler (KL) divergence as proxies. These objectives encourage distribution matching but do not directly optimize acceptance length, and the acceptance mechanism also differs between greedy and sampling-based decoding.   In this work, we propose acceptance-length-aware training losses that directly optimize the expected number of accepted tokens within a speculative window. For greedy verification, we derive an expected accepted length (EAL) loss that explicitly maximizes expected acceptance length. For sampling-based decoding, we introduce a window total variation (WTV) loss that optimizes th
    
[^280]: 基于患者轨迹的强化学习用于电子健康记录基础模型中的临床推理

    Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models

    [https://arxiv.org/abs/2609.12277](https://arxiv.org/abs/2609.12277)

    本文提出一种将EHR基础模型视为患者轨迹生成式策略的强化学习微调框架，通过时间感知的奖励设计提升临床推理能力，使小模型在数据受限场景下超越大型预训练模型。

    

    在纵向患者轨迹上训练的电子健康记录（EHR）基础模型已在多种临床预测任务中展现出强大的性能。然而，其临床推理能力仍然受限于在有限且不完整的EHR数据上进行下一词元预测。为解决这一问题，我们提出了一种强化学习（RL）微调框架，将EHR基础模型视为患者轨迹上的生成式策略。我们将常见的临床预测问题（如再入院预测）形式化为事件条件、时间窗口化的推理任务。随后，我们设计了时间感知、对轨迹展开敏感的奖励机制，以应对有限的展开长度和时间上不确定的结果。我们发现，RL微调在性能上持续优于预训练主干模型和强大的基线方法。值得注意的是，在数据受限的场景下，该框架使较小的模型能够超越更大的预训练模型，并诱导出正向迁移。

    arXiv:2609.12277v1 Announce Type: new  Abstract: Electronic health record (EHR) foundation models trained on longitudinal patient trajectories have demonstrated strong performance across diverse clinical prediction tasks. However, their clinical reasoning capabilities remain constrained by next-token prediction on limited and incomplete EHR data. To address this, we propose a reinforcement learning (RL) fine-tuning framework that treats EHR foundation models as generative policies over patient trajectories. We formulate common clinical prediction problems (e.g., hospital readmission) as event-conditioned, time-windowed reasoning tasks. We then design time-aware, rollout-sensitive rewards to account for finite rollout lengths and temporally inconclusive outcomes. We find that RL fine-tuning consistently improves over pre-trained backbones and strong baselines. Notably, it enables smaller models to surpass larger pre-trained models in data-limited regimes and induces positive transfer ac
    
[^281]: LookThere！基于强化选择的稀疏视觉

    LookThere! Sparse Vision by Reinforced Selection

    [https://arxiv.org/abs/2609.04698](https://arxiv.org/abs/2609.04698)

    提出LookThere端到端强化学习框架，联合训练浅层输入选择器与深层表示提取器，在不依赖辅助信号的情况下实现极端稀疏条件下性能与计算权衡的新帕累托前沿。

    

    视觉Transformer通常将每个图像token视为同等重要，然而对于计算机视觉中的大多数任务，实际上只需要其中一小部分。自适应计算方法通过选择需要处理的token来加速推理，但现有方法在极端稀疏情况下表现不佳，并且需要依赖可能无法泛化的启发式信号，例如token多样性和注意力分数。我们通过LookThere解决了这些局限性，借助端到端的强化学习框架联合训练浅层输入选择器和深层表示提取器，在性能与计算的权衡中实现了新的帕累托前沿。选择器学习“看哪里”，提取器学习“看什么”，两者协同工作，通过只选择对给定任务值得处理的内容来节省计算，而无需依赖辅助信号。我们证明LookThere只选择与任务相关的输入，在高分辨率设置（如交通标志）下的稀疏识别任务中表现出色。

    arXiv:2609.04698v1 Announce Type: cross  Abstract: Vision transformers typically treat every image token as equally important, yet for most tasks in computer vision only a fraction are needed. Adaptive computation methods accelerate inference by choosing which tokens to process, but existing methods struggle at extreme sparsity and require heuristics that may not generalize like token diversity and attention scores. We address these limitations with LookThere, achieving a new pareto frontier in performance-compute trade-offs through an end-to-end reinforcement learning framework that jointly trains a shallow input selector and a deep representation extractor. The selector learns where to look and the extractor learns what to see, together saving computation by selecting only what is worth processing for a given task without relying on auxiliary signals. We show that LookThere only selects the task-specific input, excelling at sparse recognition in high-resolution settings (traffic sign
    
[^282]: 面向参数化偏微分方程规范算子学习的方程重构

    Equation Recast for Canonical Operator Learning Across Parametric PDEs

    [https://arxiv.org/abs/2609.02982](https://arxiv.org/abs/2609.02982)

    提出方程重构方法，将参数化偏微分方程的算子学习转化为单一规范算子的学习，通过解析推导吸收参数引起的算子变化到有效源项中，从而实现对新参数范围的零样本预测、外推以及融合稀疏异构数据的能力。

    

    在宽泛参数范围内学习解算子，可能需要对输入函数和物理参数都进行大量覆盖，尤其对于纯数据驱动的参数化模型更是如此。此外，这类模型在训练分布之外可能会无声地失效。我们提出了方程重构，该方法将参数化算子学习重新表述为对单一规范算子的学习。由参数引起的算子变化通过控制方程解析推导得到，并被吸收进有效源项中，从而实现对新参数范围的零样本预测。在多参数、非线性以及奇异偏微分方程的设置中，方程重构支持外推，能够在共享的规范表示下整合稀疏的异构数据集，并将收敛失败作为重构迭代失效的内部预警信号。在用于核聚变的高保真托卡马克模拟中，该框架统一了……

    arXiv:2609.02982v1 Announce Type: new  Abstract: Learning solution operators across broad parameter ranges can require substantial coverage of both input functions and physical parameters, particularly for purely data-driven parametric models. In addition, the resulting models may fail silently outside the training distribution. We introduce equation recast, which reformulates parametric operator learning as the learning of a single canonical operator. Parameter-induced operator variations are derived analytically from the governing equation and absorbed into effective sources, enabling zero-shot prediction across new parameter regimes. Across multi-parameter, nonlinear, and singular PDE settings, equation recast supports extrapolation, integrates sparse heterogeneous datasets in a shared canonical representation, and uses loss of convergence as an internal warning signal for failure of the recast iteration. In high-fidelity tokamak simulations for nuclear fusion, the framework unifies
    
[^283]: FrOGS：跨化学条件的独立合金构型离散神经采样器

    FrOGS: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions

    [https://arxiv.org/abs/2609.02948](https://arxiv.org/abs/2609.02948)

    FrOGS是一种将自回归模型与连续时间马尔可夫链在单一损失下联合训练的混合离散神经采样器，可跨多种化学条件独立采样合金构型，并给出配分函数的无偏估计和热力学性质的一致性估计。

    

    预测合金的热力学性质需要在多种化学条件下对其构型进行采样，并在共同的绝对尺度上恢复自由能。马尔可夫链蒙特卡罗（MCMC）是标准工具，但它需要在不同的化学条件下分别进行模拟，并且需要借助热力学积分等辅助自由能方法才能将结果置于共同的绝对尺度上。现代离散神经采样器通常以反向KL散度作为训练目标，容易产生模式坍缩或偏差。我们提出了自由能生成采样器，这是一种混合离散神经采样器，它将自回归模型与连续时间马尔可夫链（CTMC）相耦合，在单一共享损失下进行联合训练。FrOGS能够抽取独立同分布的构型样本，返回配分函数的无偏估计，并给出热力学观测量的一致性估计。我们使用单一模型在宽泛的……

    arXiv:2609.02948v1 Announce Type: cross  Abstract: Predicting the thermodynamic properties of an alloy requires sampling its configurations across many chemical conditions and recovering free energies on a common absolute scale. Markov chain Monte Carlo (MCMC) is the standard tool, but it requires separate simulations at different conditions, and auxiliary free-energy methods such as thermodynamic integration are used to place results on a common absolute scale. Modern discrete neural samplers typically use reverse KL divergence as the objective and can be mode-seeking or biased. We present Free energy Offering Generative Sampler (FrOGS), a hybrid discrete neural sampler that couples an autoregressive model to a continuous-time Markov chain (CTMC) to be trained jointly under a single shared loss. FrOGS draws i.i.d. configurations, returns an unbiased estimate of the partition function, and gives consistent estimates of thermodynamic observables. We train a single model across a wide ra
    
[^284]: 面向材料表征中基于图像逆问题的可扩展复合函数贝叶斯优化

    Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization

    [https://arxiv.org/abs/2609.02126](https://arxiv.org/abs/2609.02126)

    本文提出一种可扩展的复合函数贝叶斯优化方法（SBOCF），利用图像匹配目标的复合结构和模拟图像的中间信息，高效地从PACBED图样中估计样品厚度和晶体失倾等关键参数，克服了网格搜索效率低下和神经网络需大量预训练且难以迁移的缺点。

    

    从科学图像中估计物理参数是材料表征中一类常见的逆问题，通常依赖于昂贵的基于物理的模拟。在电子显微学中，样品厚度和晶体失倾是决定电子如何在样品中散射的关键参数，因此直接影响从图像中恢复的原子尺度结构的准确性。这些参数通常通过将实验测得的位置平均会聚束电子衍射（PACBED）图样与模拟图样进行匹配来推断，但网格搜索方法的可扩展性差，而神经网络方法则需要大量预训练，且可能无法迁移到新的实验条件。本文提出了可扩展的复合函数贝叶斯优化方法（SBOCF），这是一种节省模拟计算量的高效方法，它利用了图像匹配目标函数已知的复合结构以及模拟图像中包含的中间信息。通过用图像块表示PACBED图像……

    arXiv:2609.02126v1 Announce Type: new  Abstract: Estimating physical parameters from scientific images is a common inverse problem in materials characterization that often relies on expensive physics-based simulations. In electron microscopy, specimen thickness and crystal mistilt are critical parameters that govern how electrons scatter through the sample, and therefore the accuracy of any atomic-scale structure recovered from it. They are commonly inferred by matching experimental position-averaged convergent-beam electron diffraction (PACBED) patterns to simulated ones, but grid searches scale poorly and neural-network methods require extensive pretraining that may not transfer to new conditions. Here, we propose scalable Bayesian optimization of composite functions (SBOCF), a simulation-efficient method that exploits the known composite structure of the image-matching objective and the intermediate information contained in simulated images. By representing PACBED images with patch-
    
[^285]: ST²U：基于受限知识边界控制的状态化测试时遗忘

    ST$^2$U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control

    [https://arxiv.org/abs/2608.23034](https://arxiv.org/abs/2608.23034)

    ST²U通过轨迹级别的受限知识边界控制，解决了现有测试时遗忘方法因忽略自回归生成中状态重建而导致的受限知识重入问题。

    

    摘要：控制大型语言模型中的受限知识对于模型对齐和安全部署至关重要。测试时遗忘通过仅在推理过程中进行干预，避免了昂贵的重新训练和参数更新。然而，现有的激活编辑方法应用孤立的逐点修正，忽视了自回归生成如何从提示、缓存和生成前缀中持续重建隐藏状态。因此，后续状态可能在局部成功修正后重新回到受限知识区域，导致受限知识重新进入。在本工作中，我们提出了基于受限知识边界控制的状态化测试时遗忘（ST²U），该方法将测试时遗忘形式化为轨迹级别的边界控制。ST²U首先在低维可逆坐标中建模受限知识边界，同时保持正交的非目标组件不变。在推理过程中，ST²U监控...

    arXiv:2608.23034v1 Announce Type: cross  Abstract: Controlling restricted knowledge in large language models is essential for model alignment and safe deployment. Test-time unlearning avoids costly retraining and parameter updates by intervening only during inference. However, existing activation-editing methods apply isolated pointwise corrections, overlooking how autoregressive generation continually reconstructs hidden states from the prompt, cache, and generated prefix. Consequently, later states may return to restricted knowledge regions after a locally successful correction, causing restricted knowledge re-entry. In this work, we propose Stateful Test-Time Unlearning via restricted knowledge boundary control (ST$^2$U), which formulates test-time unlearning as trajectory-wide boundary control. ST$^2$U first models restricted knowledge boundaries in low-dimensional invertible coordinates while leaving orthogonal non-target components unchanged. During inference, ST$^2$U monitors ri
    
[^286]: SplitLite：用于分割学习的低秩残差压缩方法

    SplitLite: Low-Rank Residual Compression for Split Learning

    [https://arxiv.org/abs/2608.23018](https://arxiv.org/abs/2608.23018)

    SplitLite通过利用相邻周期激活和梯度残差的低秩结构（秩-2r和秩-4r），仅传输量化的截断奇异值，大幅降低分割联邦LoRA微调的通信成本。

    

    摘要：arXiv:2608.23018v1 公告类型：交叉 摘要：设备端大型语言模型（LLMs）的联邦微调面临显著的计算负担。为克服此限制，分割学习（SL）作为一种有前景的解决方案出现，它将主要训练工作负载卸载到强大的服务器上。然而，SL需要在客户端和服务器之间交换高维激活和梯度，导致高昂的通信成本。为应对这一挑战，我们提出了SplitLite，一种通信高效的分割联邦LoRA微调方法，该方法利用连续周期激活和梯度残差的低有效秩结构。我们的关键发现是，当LoRA在参数空间中使用秩为$r$的更新时，相邻周期之间同一数据样本的激活和梯度残差分别表现出有效秩-$2r$和秩-$4r$的结构。通过揭示这一特性，SplitLite仅传输量化的截断奇异值。

    arXiv:2608.23018v1 Announce Type: cross  Abstract: Federated fine-tuning of on-device large language models (LLMs) faces a significant computing burden. To overcome this limitation, split learning (SL) has emerged as a promising solution, which offloads the primary training workload to a powerful server. However, SL requires exchanging high-dimensional activations and gradients between clients and the server, resulting in prohibitive communication costs. To overcome this challenge, we propose SplitLite, a communication-efficient split federated LoRA fine-tuning method that exploits the low effective rank structure of consecutive-epoch activation and gradient residuals. Our key finding is that, when LoRA uses rank $r$ updates in parameter space, the activation and gradient residuals of the same data sample between adjacent epochs also exhibit effective rank-$2r$ and rank-$4r$ structures, respectively. By revealing this property, SplitLite transmits only quantized truncated singular valu
    
[^287]: 审计语言模型水印中的跨语言公平性

    Auditing Cross-Lingual Fairness in Language Model Watermarking

    [https://arxiv.org/abs/2608.20047](https://arxiv.org/abs/2608.20047)

    提出一个四部分评估框架，用于审计语言模型水印在跨语言场景中的公平性，并通过实证校准和多种测量揭示英语评估掩盖的多语言部署问题。

    

    arXiv:2608.20047v1 公告类型：新 摘要：大型语言模型输出的水印方案几乎完全使用英语文本，并基于每种方案的检测阈值和一组狭窄的质量测量进行评估。多语言部署暴露了在英语中无关紧要但在跨语言中决定结论的评估设计选择。我们提出一个包含四个组件的评估框架：按部署环境经验校准的检测阈值，一个与阈值无关的伴随测量，用于区分校准失败与检测失败，三种不重叠的质量测量范式（分布性、配对语义和参考困惑度），以及一种基于类型学家族划分的跨语言差异的广义熵分解。应用于六种水印方案、三种开放权重生成器、涵盖四种文字和八个类型学家族的十一种语言，以及基础和指令调优两种模式。

    arXiv:2608.20047v1 Announce Type: new  Abstract: Watermarking schemes for large language model output are evaluated almost exclusively on English text using each scheme's detection threshold and a narrow set of quality measurements. Multilingual deployment exposes evaluation-design choices that are inconsequential on English but determine conclusions cross-lingually. We propose an evaluation framework with four components: detection thresholds calibrated empirically per deployment context, a threshold-independent companion measurement that distinguishes calibration failures from detection failures, three disjoint quality measurement paradigms (distributional, paired-semantic, and reference-perplexity), and a generalized-entropy decomposition of cross-language disparity over a typological family partition. Applied to six watermarking schemes, three open-weight generators, eleven languages spanning four scripts and eight typological families, and both base and instruction-tuned regimes, 
    
[^288]: 面向特征发现与控制的多模态模型对比方法

    Multimodal Model Diffing for Feature Discovery and Control

    [https://arxiv.org/abs/2608.09928](https://arxiv.org/abs/2608.09928)

    提出了MMDiff多模态模型对比框架，通过训练多模态稀疏自编码器并将其作为特征级接口，实现多模态大语言模型中被多模态训练改变的特征的发现、隔离与针对性控制。

    

    多模态大语言模型（MLLMs）展现出强大的视觉理解能力，然而引发这些行为的内部特征仍然难以识别、审计或控制。虽然使用稀疏自编码器（SAEs）将隐藏状态分解为可解释的特征方向适用于事后检查，但这些方法既无法轻易分离出哪些特征因多模态训练而发生改变，也无法直接用于针对性控制。我们提出了MMDiff，一个多模态模型对比框架，它训练多模态稀疏自编码器并将其转化为特征级接口，用于发现和控制多模态行为。MMDiff支持三种用途：（i）特征隔离，通过对比基础语言模型的SAE与其多模态适配版本，识别被多模态训练改变的特征；（ii）任务特定特征检测，通过逐token的对比激活分析分离出因果特征；（iii）特征级控制（摘要在此处被截断）。

    arXiv:2608.09928v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) exhibit strong visual understanding, yet the internal features that cause these behaviors remain difficult to identify, audit, or control. While applicable to post-hoc inspection, hidden states that are decomposed into interpretable feature directions using sparse autoencoders (SAEs) neither readily isolate which features are changed by multimodal training, nor are they directly useful for targeted control. We introduce MMDiff, a multimodal model-diffing framework that trains multimodal SAEs and turns them into feature-level interfaces for discovering and controlling multimodal behavior. MMDiff supports three uses: (i) feature isolation, by diffing a base-LM SAE against its multimodal-adapted counterpart to identify features altered by multimodal training; (ii) task-specific feature detection, via per-token contrastive firing analysis that isolates causal features; and (iii) feature-leve
    
[^289]: 反事实公平强化学习的分布映射方法

    A Distribution Mapping Approach to Counterfactually Fair Reinforcement Learning

    [https://arxiv.org/abs/2608.08743](https://arxiv.org/abs/2608.08743)

    本文提出一种基于分位数分布映射的数据预处理算法，通过序贯估计反事实状态和奖励，在强化学习中实现反事实公平性，并将常用的可加性假设作为特例涵盖其中。

    

    强化学习（RL）旨在通过优化序贯决策来最大化长期的群体收益。然而，在医疗保健等高风险场景中部署时，强化学习的决策可能系统性地限制某些亚群体获取有价值服务的机会，这与利益相关者的价值观和目标相违背。反事实公平性（CF）提供了一个基于因果推理来解决这一问题的有前景的框架。本文提出了一种数据预处理算法，将其与策略学习结合使用时，能够在强化学习中实现反事实公平性。我们的算法依赖于一种新颖的分位数分布映射方法，在数据预处理步骤中序贯地估计反事实状态和奖励，并将反事实预测中常用的可加性假设作为特例纳入其中。我们从理论上证明了每步的反事实不公平程度和无限时域的次优差距……

    arXiv:2608.08743v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) seeks to optimize sequential decisions to maximize population-level benefits over time. However, when deployed in high-stakes settings such as healthcare, RL decisions might systematically restrict some subpopulation's access to valuable services in a manner contrary to the values and goals of stakeholders. Counterfactual fairness (CF) offers a promising framework to address this problem based on causal reasoning. This paper develops a data preprocessing algorithm that, when used in tandem with policy learning, enables CF in RL. Our algorithm relies on a novel quantile distribution mapping method for sequentially estimating the counterfactual states and rewards in the data preprocessing step, subsuming common additivity assumptions used for counterfactual prediction as a special case. We theoretically prove that the per-step level of counterfactual unfairness and infinite-horizon suboptimality gap ca
    
[^290]: DynaCrys：基于动态空间群扩散的晶体生成

    DynaCrys: Crystal Generation with Dynamic Space-Group Diffusion

    [https://arxiv.org/abs/2608.07401](https://arxiv.org/abs/2608.07401)

    DynaCrys是首个让空间群与Wyckoff占据及元素在生成过程中共同动态演化的完整晶体生成框架，通过显式融入晶体学群-子群关系的物理引导空间群扩散和合法性约束的随机解码器，联合采样相互兼容的空间群、Wyckoff占据与元素。

    

    寻找新型晶体材料需要在巨大的成分与结构空间中进行探索。在该空间中生成候选材料，需要对元素组成、晶体学对称性与晶体几何结构之间的关系进行建模。空间群规定了晶体的对称性，并决定了可容许的Wyckoff位置和几何自由度；若预先固定空间群，就无法利用不断演化的Wyckoff占据情况和元素组成信息来修正这一全局对称性选择。我们提出了DynaCrys，这是首个在完整晶体生成过程中空间群与Wyckoff占据及元素共同演化的框架。我们开发了物理知识引导的空间群扩散方法，显式地融入了晶体学群-子群关系，并设计了受合法性约束的随机解码器，可联合采样相互兼容的空间群、Wyckoff占据和元素……

    arXiv:2608.07401v2 Announce Type: replace-cross  Abstract: The search for new crystalline materials spans an enormous compositional and structural space. Generating candidates in this space requires modeling the relationships between elemental composition, crystallographic symmetry, and crystal geometry. The space group specifies the crystal's symmetry and determines the admissible Wyckoff positions and geometric degrees of freedom; fixing it in advance prevents this global symmetry choice from being revised using information from evolving Wyckoff occupations and elemental composition. We introduce DynaCrys, the first framework for complete crystal generation in which the space group co-evolves with Wyckoff occupations and elements. We develop physics-informed space-group diffusion that explicitly incorporates crystallographic group-subgroup relations, and a legality-constrained stochastic decoder that jointly samples compatible space groups, Wyckoff occupations, and elements, followed
    
[^291]: RamanPFN：利用表格基础模型从拉曼光谱结构中学习

    RamanPFN: learning from Raman spectral structure with a tabular foundation model

    [https://arxiv.org/abs/2608.02157](https://arxiv.org/abs/2608.02157)

    提出RamanPFN——一个通过物理引导的光谱学习（结合全局成分解混GCU与局部振动子空间编码LVSE）实现统一上下文推理的通用拉曼光谱基础模型框架。

    

    拉曼光谱能够在材料科学、分析化学、生物医学和工业过程监测等领域实现无标记的分子表征。然而，面向高维光谱的机器学习仍然受到标注数据有限以及光谱物理组织结构与特征无关模型之间不匹配的制约。仅有信道覆盖并不能确保相关谱带共享一个共同的推理上下文。在此，我们提出了RamanPFN，一个通用的光谱基础框架，通过物理引导的光谱学习实现统一的上下文推理。它通过全局成分解混捕捉全光谱的成分协变关系，将分布式、多谱带的混合特征分解为共享的非负潜在基。同时，它通过局部振动子空间编码解析局部振动结构，保留细粒度的峰……

    arXiv:2608.02157v2 Announce Type: replace-cross  Abstract: Raman spectroscopy enables label-free molecular characterization across materials science, analytical chemistry, biomedicine, and industrial process monitoring. However, machine learning for high-dimensional spectroscopy remains constrained by limited labelled data and a mismatch between the physical organization of spectra and feature-agnostic models. Channel coverage alone does not ensure that related bands share a common inference context. Here we present RamanPFN, a general-purpose spectral foundation framework that enables unified in-context inference through physics-guided spectral learning. It captures full-spectrum compositional covariation via Global Compositional Unmixing (GCU), which decomposes distributed, multi-band mixture signatures into shared non-negative latent bases. Simultaneously, it resolves local vibrational structure through Local Vibrational Subspace Encoding (LVSE), which preserves fine-grained peak mo
    
[^292]: QAdapt：面向表面码量子纠错的噪声自适应神经预解码

    QAdapt: Noise-Adaptive Neural Pre-Decoding for Surface-Code Quantum Error Correction

    [https://arxiv.org/abs/2607.28422](https://arxiv.org/abs/2607.28422)

    QAdapt提出了一种噪声自适应神经预解码框架，通过学习局部时空相关性并结合在线弹性权重巩固，在多种噪声机制间持续适应，显著降低了表面码量子纠错的逻辑错误率并提升了解码效率。

    

    arXiv:2607.28422v2 公告类型：替换 摘要：容错量子计算依赖量子纠错来保护逻辑信息，因此需要能够跟上综合征生成速度的经典解码器。异构且不断变化的硬件噪声，以及从仿真到实验的分布偏移，对固定神经解码器的准确性和鲁棒性构成了挑战。我们提出了QAdapt，一个面向表面码量子纠错的噪声自适应神经预解码框架。QAdapt学习局部时空相关性，可在不同噪声环境之间持续适应，并将残余综合征传递给传统的全局解码器。在线弹性权重巩固（EWC）机制使得训练在测量、空闲、CNOT和Pauli偏置等噪声机制之间切换时能够限制遗忘。在码距为9的情况下，HTNet预解码器相较于Ising-Fast将等效的每轮逻辑错误率降低了15.64%，同时参数减少了28.75%，匹配后端处理时间减少了9.07%。在线EWC将平均……（摘要原文在此处截断）

    arXiv:2607.28422v2 Announce Type: replace  Abstract: Fault-tolerant quantum computing relies on quantum error correction to protect logical information, requiring classical decoders that keep pace with syndrome generation. Heterogeneous, changing hardware noise and the distribution shift from simulation to experiment challenge the accuracy and robustness of fixed neural decoders. We present QAdapt, a noise-adaptive neural pre-decoding framework for surface-code quantum error correction. QAdapt learns local spatiotemporal correlations, continually adapts across noise environments, and passes the residual syndrome to a conventional global decoder. Online elastic weight consolidation limits forgetting as training moves between measurement, idle, CNOT, and Pauli-bias noise regimes. At distance nine, the HTNet pre-decoder reduces equivalent per-round logical error by 15.64% relative to Ising-Fast, with 28.75% fewer parameters and 9.07% less matching-backend time. Online EWC reduces average 
    
[^293]: 自回归模型中的全变差距离估计

    Total Variation Distance Estimation in Autoregressive Models

    [https://arxiv.org/abs/2607.19510](https://arxiv.org/abs/2607.19510)

    本文研究了在样本访问、logit 访问和含噪 logit 访问三种模型下估计两个长度为 $n$ 的自回归分布之间全变差距离的问题，将样本访问下的查询复杂度从已有的 $\widetilde{O}(n^3 m/\varepsilon^5)$ 改进为 $\widetilde{O}(n^2 K/\varepsilon^2)$，并给出了 logit 访问下紧的 $O(n/\varepsilon^2)$ 查询界。

    

    现代大语言模型的部署在固定权重之上使用了多种实现选择和推理优化手段（如批处理、自定义内核和量化），因此两个提供“同一模型”服务的引擎可能产生具有实质性差异的分布。我们研究了在三种访问模型下，估计两个长度为 $n$ 的自回归分布之间的全变差（TV）距离至加性误差 $\varepsilon$ 的问题。（1）在样本访问下，我们使用 $\widetilde{O}(n^2 K/\varepsilon^2)$ 次查询，其中 $K$ 是下一个标记（next-token）分布的最大支撑集大小，这改进了 Meel 等人（2025）需要 $\widetilde{O}(n^3 m/\varepsilon^5)$ 次查询的估计器，其中 $m \geq K$ 是标记字母表的总大小。（2）在 logit 访问下，我们使用 $O(n/\varepsilon^2)$ 次查询，且该界是紧的。（3）在含噪 logit 访问下，我们在上述两种保证之间实现了平滑插值：如果概率值以相对……（摘要原文在此处截断）

    arXiv:2607.19510v2 Announce Type: replace  Abstract: Modern LLM deployments use a number of implementation choices and inference optimizations (e.g., batching, custom kernels, and quantization) on top of fixed weights, so two engines serving "the same model" can produce meaningfully different distributions. We study the problem of estimating the total variation (TV) distance between two length-$n$ autoregressive distributions to additive error $\varepsilon$, under three access models. (1) Under sample access, we use $\widetilde{O}(n^2 K/\varepsilon^2)$ queries, where $K$ is the maximum support of the next-token distribution. This improves upon the $\widetilde{O}(n^3 m/\varepsilon^5)$-query estimator of Meel et al. (2025), where $m \geq K$ is the total size of the token alphabet. (2) Under logit access, we use $O(n/\varepsilon^2)$ queries, and this is tight. (3) Under noisy logit access, we smoothly interpolate between the above two guarantees: if probability values are given to relativ
    
[^294]: 先推理后翻译：利用结构化推理增强法律机器翻译

    Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning

    [https://arxiv.org/abs/2607.19181](https://arxiv.org/abs/2607.19181)

    该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。

    

    法律领域的神经机器翻译（NMT）是一项在语言和概念上都极具挑战性的任务，这主要源于法律语言的复杂性以及其对高精度的要求。近期出现的具备推理能力的语言模型为应对此类挑战开辟了新的可能性，并为一系列此前提出的提升翻译质量的技术（包括监督微调和强化学习）提供了补充。在这项工作中，我们对这些不同的方法进行了比较。更具体而言，我们评估了经过各种再训练范式增强的小型语言模型（如Qwen3.5 4B、Qwen3.5 9B和Gemma 3 12B），并将其性能与前沿推理模型进行对比。我们聚焦于瑞士法律体系，其独特的多语言法规为推理增强模型提供了一个特别具有挑战性的测试平台。我们的结果表明……（原文摘要在此处截断）

    arXiv:2607.19181v2 Announce Type: replace  Abstract: Neural machine translation (NMT) in the legal domain is a linguistically and conceptually demanding task, primarily due to the complexity of legal language and the high level of precision it requires. The recent emergence of reasoning-capable language models opens new possibilities for tackling such challenges. They add to a set of other previously proposed techniques to enhance the translation quality, which includes supervised fine-tuning and reinforcement learning.   In this work, we perform a comparison between these various approaches. More particularly, we evaluate small language models such as Qwen3.5 4B, Qwen3.5 9B, and Gemma 3 12B enhanced with various re-training paradigms and compare their performances against frontier reasoning models. We focus on the Swiss legal system, which -- with its unique multilingual statutes -- offers a particularly challenging testbed for reasoning-augmented models. Our results show that the qua
    
[^295]: 利用带控制的动态模态分解在单一实现中解耦强迫气候变率与内部气候变率

    Disentangling Forced and Internal Climate Variability in Single Realizations using Dynamic Mode Decomposition with Control

    [https://arxiv.org/abs/2607.18298](https://arxiv.org/abs/2607.18298)

    该研究提出PullbackDMDc方法，基于非自治动力系统理论、带控制的动态模态分解与拉回吸引子估计，将外部强迫视为动力学驱动因子，从而实现从单一气候观测记录中分离强迫趋势与内部气候变率。

    

    将外部强迫趋势与内部气候变率分离是气候科学中的一项根本性挑战，尤其是在分析仅以单一实现形式存在的观测记录时。我们证明，通过将外部强迫视为线性随机系统中的一个动力学驱动因子，可以将单一气候实现分解为强迫分量和内部分量，这一思想植根于拉回吸引子理论。这解决了气候预估以及强迫响应检测与归因中的一个核心难题。现有的单一实现方法依赖于忽略强迫作用的线性逆模型（LIM），或忽略动力过程的线性回归。在此，我们提出PullbackDMDc方法，该方法基于非自治动力系统理论和带控制的动态模态分解（DMDc），并结合拉回吸引子估计，将单一气候实现分解为空间模态……（原文摘要在此处截断）

    arXiv:2607.18298v2 Announce Type: replace-cross  Abstract: Separating externally forced trends from internal climate variability is a fundamental challenge in climate science, particularly when analyzing observational records that exist only as a single realization. We show that a single climate realization can be decomposed into forced and internal components by treating external forcing as a dynamical driver within a linear stochastic system, an idea grounded in pullback attractor theory. This addresses a central challenge for climate projection and for detecting and attributing the forced response. Existing single-realization methods rely on linear inverse models (LIMs), which ignore forcing, or linear regression, which omits dynamics. Here we introduce PullbackDMDc, a method grounded in non-autonomous dynamical systems theory and dynamic mode decomposition with control (DMDc), incorporating pullback attractor estimation to decompose a single climate realization into spatial modes a
    
[^296]: 论图神经网络作为供应链优化元模型的潜力：数据集、架构与方向

    On the Potential of Graph Neural Networks as Metamodels for Supply Chain Optimization: Dataset, Architectures, and Directions

    [https://arxiv.org/abs/2607.16769](https://arxiv.org/abs/2607.16769)

    本文形式化了供应链优化问题，利用SupplyNetPy仿真库创建了大型公开的供应链图训练数据集，并探索了可作为仿真代理的图神经网络架构及其精度与计算成本的权衡，为GNN在供应链结构优化中的应用奠定基础。

    

    图神经网络（GNN）已成为一类针对图结构系统的强大可微分学习模型。其跨拓扑结构泛化的能力为结构优化与参数优化相结合的代理模型开辟了前景，而这是经典元模型无法实现的。供应链是一个天然的应用目标，然而将GNN代理模型应用于供应链问题的研究在很大程度上仍未被探索。本文为此奠定了基础，提出了初步步骤，并讨论了关键研究方向。作为基础工作，我们对问题进行了形式化表述，并创建了一个大型公开训练数据集，其中包含通过程序化方式生成的供应链图，以及使用我们的SupplyNetPy仿真库获得的输入参数和稳态性能指标。作为初步探索，我们研究了适合作为节点级和网络级预测代理模型的GNN架构，并分析了其相对于仿真方法在精度与计算成本之间的权衡。最重要的是……

    arXiv:2607.16769v2 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) have emerged as a powerful, differentiable class of learning models for graph-structured systems. Their ability to generalize across topologies opens the prospect of a surrogate for combined structural and parametric optimization, which classical metamodels cannot offer. Supply chains are a natural target, yet the use of GNN surrogates for supply chain problems is largely unexplored. This paper lays the foundation, presents initial steps, and discusses key research directions. As a foundation, we formulate the problem and create a large public training dataset of programmatically generated supply chain graphs with input parameters and steady-state performance metrics obtained using our SupplyNetPy simulation library. As initial steps, we explore GNN architectures that work well as surrogates for node- and network-level predictions, and analyze their accuracy-compute trade-off against simulation. Most impo
    
[^297]: 基于机器学习的电阻性硅传感器重建方法

    Machine Learning-Based Reconstruction for Resistive Silicon Sensors

    [https://arxiv.org/abs/2607.11585](https://arxiv.org/abs/2607.11585)

    本文提出了一种基于机器学习的电阻性硅传感器（如AC-LGAD）信号重建与压缩方法，利用多焊盘全波形信息进行正则化，解决了传统矩阵求逆方法计算复杂且对非对角噪声敏感的问题，从而在放宽读出间距的条件下实现高精度空间重建。

    

    低增益雪崩二极管与交流耦合低增益雪崩二极管（AC-LGAD）是有前景的精密计时与四维径迹探测技术。在AC-LGAD中，交流焊盘通过介质层与电阻性n$^{+}$层耦合，而增益层保持不分段。这种结构提供了100%的填充因子，并能在放宽读出间距的条件下实现良好的空间分辨率。然而，使得插值成为可能的信号共享机制同时也使读出变得复杂：电荷会扩散到多个焊盘上，有用信号可能接近电子噪声阈值，而矩阵求逆方法在计算上可能非常困难，且对非对角噪声十分敏感。在这项工作中，我们研究了基于机器学习的电阻性硅传感器重建与压缩方法。我们利用来自相关焊盘的全波形信息来对重建过程进行正则化并提取空间信息……

    arXiv:2607.11585v2 Announce Type: replace-cross  Abstract: Low-Gain Avalanche Diodes (LGADs) and AC-coupled Low-Gain Avalanche Diodes (AC-LGADs) are promising technologies for precision timing and four-dimensional tracking. In AC-LGADs, the AC pad is coupled to the resistive n$^{+}$ layer through a dielectric layer, while the gain layer remains unsegmented. This structure provides a 100\% fill factor and enables good spatial resolution with a relaxed readout pitch. The same signal-sharing mechanism that makes interpolation possible complicates the readout: charge spreads across multiple pads, the useful information can approach the electronic-noise threshold, and matrix-inversion approaches can become computationally challenging and sensitive to off-diagonal noise. In this work, we study machine-learning-based reconstruction and compression for resistive silicon sensors. We use full-waveform information from correlated pads to regularise the reconstruction and extract spatial informati
    
[^298]: SupplyNetPy：用于任意供应链与库存网络高保真建模与仿真的开源Python库

    SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks

    [https://arxiv.org/abs/2607.09745](https://arxiv.org/abs/2607.09745)

    SupplyNetPy是一个开源Python库，可对任意多级结构的供应链与库存网络进行高保真离散事件仿真，支持多种补货策略、易腐库存、节点中断与随机需求，其组件可继承扩展，并经过基准对比验证，支持模型的程序化生成与设计空间探索。

    

    本文介绍了SupplyNetPy，这是一个开源、文档完善的Python库，用于对具有任意多级结构的供应链网络进行建模和离散事件仿真。它支持多种补货策略、易腐品库存、节点中断以及随机需求和交货期。所有组件均可通过继承进行扩展。用户将供应链描述为带有节点和链接属性的图，而由该库负责执行仿真，提供日志以及丰富的节点级和网络级性能报告。本文阐述了SupplyNetPy的开发动机、设计、关键特性和架构，并给出了详细的验证结果（与分析基准、商业工具以及已发表的案例研究进行对比）。开发SupplyNetPy的一个重要动机是以编程方式生成和仿真复杂模型，从而支持设计空间探索、假设分析、训练数据生成等应用。

    arXiv:2607.09745v2 Announce Type: replace  Abstract: This paper introduces SupplyNetPy, an open-source, well-documented Python library for modeling and discrete-event simulation of supply chain networks with arbitrary multi-echelon structures. It supports multiple replenishment policies, perishable inventory, node disruptions, and stochastic demand and lead times. All components are extensible via inheritance. Users describe a supply chain as a graph with node and link attributes, while the library handles simulation, providing logs and extensive node and network level performance reports. This paper presents the motivation, design, key features, and architecture of SupplyNetPy, along with detailed validation results (against analytical benchmarks, a commercial tool, and a published case study). A key motivation behind SupplyNetPy's development is programmatic generation and simulation of complex models, enabling design-space exploration, what-if analysis, training data generation, and
    
[^299]: XS-VLA：通过空间监督与示范条件化训练微型视觉-语言-动作模型

    XS-VLA: Teaching Tiny Vision-Language-Action Models with Spatial Supervision and Demonstration Conditioning

    [https://arxiv.org/abs/2607.04171](https://arxiv.org/abs/2607.04171)

    XS-VLA通过教师空间蒸馏与示范条件化流匹配的两阶段训练，在仅约244M参数的紧凑视觉-语言-动作模型上实现了90.25%的LIBERO平均成功率，超越了更大规模的基线模型。

    

    如何在保持部署策略紧凑的同时，利用更丰富的训练监督来改进机器人控制？我们提出了XS-VLA，这是一个分阶段的训练框架，采用教师模型生成的空间标签和示范条件化的动作学习。粗粒度空间蒸馏（CSD）通过一个辅助的区域标签任务来初始化骨干网络；随后，潜在流匹配（LFM）将动作空间的速度场条件化于示范的潜在表示，并使用KL正则化，同时联合优化骨干网络与动作模块。最终部署的策略包含243.99M参数，且无需教师模型或后验编码器即可运行。在两个训练随机种子下，XS-VLA在LIBERO上均达到90.25%的平均成功率，而在相同设定下训练的SmolVLA-256M基线仅为86.00%。消融实验通过匹配的图像预训练以及Huber/MSE对照检验了两个训练阶段的有效性。在三个Mobile ALOHA任务上，平均严格成功率从21……（摘要在此处被截断）

    arXiv:2607.04171v4 Announce Type: replace-cross  Abstract: How can richer training supervision improve robot control while keeping the deployed policy compact? We present XS-VLA, a staged training framework using teacher-derived spatial labels and demonstration-conditioned action learning. Coarse-Grained Spatial Distillation (CSD) initializes the backbone through an auxiliary region-label task. Latent Flow Matching (LFM) then conditions an action-space velocity field on a demonstration latent, using KL regularization while jointly optimizing the backbone and action modules. The deployed policy contains 243.99M parameters and operates without the teacher or posterior encoder. XS-VLA achieves 90.25% average LIBERO success in each of two training seeds, compared with 86.00% for a SmolVLA-256M base trained under our settings. Ablations examine both training stages through matched image pretraining and Huber/MSE controls. On three Mobile ALOHA tasks, average strict success increases from 21
    
[^300]: 面向基于分数扩散采样的自适应重参数化时间

    Adaptive Reparametrized Time for Score-Based Diffusion Sampling

    [https://arxiv.org/abs/2607.02137](https://arxiv.org/abs/2607.02137)

    提出自适应重参数化时间方法，通过将采样时钟速度作为控制量并结合连续时间强化学习（ART-RL），自适应地为扩散采样分配时间步，克服了均匀和手工设计调度的次优性。

    

    我们研究基于分数的扩散采样中的时间步分配问题，其中学习到的逆向时间动力学需要在有限网格上进行离散化。均匀调度和手工设计的调度是标准选择，但它们依赖于固定的预设方案，因此可能是次优的。为了解决这一局限，我们提出了自适应重参数化时间，这是一种连续时间控制形式，通过将采样时钟的速度视为控制量来学习时间变换，从而使得在学习到的时钟上的均匀网格能够映射为原始扩散时间中的自适应时间步。基于一阶欧拉误差代理，ART为沿采样轨迹分配时间步提供了一个有原则的目标函数。为了求解这个确定性控制问题，我们引入了ART-RL，这是一种辅助性的随机化形式，采用高斯策略将调度学习转化为连续时间强化学习问题。我们证明了该随机化过程（原文在此处截断）……

    arXiv:2607.02137v3 Announce Type: replace-cross  Abstract: We study timestep allocation for score-based diffusion sampling, where a learned reverse-time dynamics is discretized on a finite grid. Uniform and hand-crafted schedules are standard choices, but they rely on fixed prescriptions and can therefore be suboptimal. To address this limitation, we propose Adaptive Reparameterized Time (ART), a continuous-time control formulation that learns a time change by treating the speed of the sampling clock as the control, so that a uniform grid on the learned clock induces adaptive timesteps in the original diffusion time. Based on a leading-order Euler error surrogate, ART provides a principled objective for allocating timesteps along the sampling trajectory. To solve this deterministic control problem, we introduce ART-RL, an auxiliary randomized formulation with Gaussian policies that turns schedule learning into a continuous-time reinforcement learning problem. We prove that the randomiz
    
[^301]: GAIA：面向正问题与逆问题的几何自适应算子学习

    GAIA: Geometry-Adaptive Operator Learning for Forward and Inverse Problems

    [https://arxiv.org/abs/2607.01128](https://arxiv.org/abs/2607.01128)

    GAIA提出了一种几何自适应积分自编码器架构，通过将域边界与内部场分布编码为几何标记并经交叉注意力调制积分变换层，首次实现了任意几何域上正问题与逆问题（输入输出可处于不同域）的统一算子学习。

    

    arXiv:2607.01128v2 公告类型： replace 摘要：在任意几何结构上对偏微分方程（PDE）进行算子学习，可为大规模仿真构建快速的神经代理模型。尽管近期的几何自适应神经算子已取得长足进展，但它们主要针对正问题设计，即输入与输出共享同一空间域。这限制了它们在边值问题（BVPs）和逆问题中的应用，因为在这类问题中输入和输出可能位于不同的域上。我们提出了几何自适应积分自编码器（GAIA），这是一种算子学习模型，它将域边界和内部场分布编码为几何标记，并通过交叉注意力机制将这些标记作为积分变换层的条件，使核函数能够局部适应几何特征。由此得到的架构能够学习任意域上正问题与逆问题的解算子，并以单次前向传递的方式求解每个问题……

    arXiv:2607.01128v2 Announce Type: replace  Abstract: Operator learning for partial differential equations (PDEs) on arbitrary geometries builds fast neural surrogates for large-scale simulation. While recent geometry-adaptive neural operators have made substantial progress, they are mainly designed for forward problems in which inputs and outputs share the same spatial domain. This limits their applicability for boundary value problems (BVPs) and inverse problems, where inputs and outputs may live on different domains. We introduce the Geometry-Adaptive Integral Autoencoder (GAIA), an operator learning model that encodes the domain boundary and the interior field distribution into geometry tokens, and conditions integral transform layers on these tokens via cross-attention, allowing the kernel to adapt locally to geometric features. This yields an architecture capable of learning solution operators for forward and inverse problems on arbitrary domains, solving each in a single pass at 
    
[^302]: CoMet：面向多模态不确定性估计的上下文与多样性分解

    CoMet: Context and Multiplicity Decomposition for Multimodal Uncertainty Estimation

    [https://arxiv.org/abs/2606.32012](https://arxiv.org/abs/2606.32012)

    提出CoMet方法，通过将多模态大语言模型的不确定性分解为上下文特定项（捕捉给定任务或提示引起的歧义）和多样性特定项（捕捉与之相容的合理答案数量），从而在开放式多模态场景下实现不确定性估计。

    

    不确定性估计是人工智能模型中长期存在的挑战；它相当于“知道自己不知道什么”，而元认知即使对人类而言也是出了名的困难（参见邓宁-克鲁格效应）。尽管即使在较简单的分类系统中这一问题仍远未解决，但在多模态大语言模型（MLLM）中解决这一问题正变得日益重要。在MLLM中，不确定性可能来自各种不同的来源及其相互关系，在开放式设定下，还可能来自无界的答案空间。为解决这些问题，我们提出了CoMet，一种将不确定性分解为上下文特定项和多样性特定项的MLLM不确定性估计方法。前者捕捉由给定上下文（例如任务或提示）所引起的歧义，后者捕捉由上下文所确定的、与给定输入保持相容的合理答案的数量。我们训练一个轻量……

    arXiv:2606.32012v2 Announce Type: replace  Abstract: Uncertainty estimation has been a long-standing challenge in AI models; it amounts to "knowing what you don't know," and metacognition is notoriously difficult even for humans (cf. the Dunning-Kruger effect). Although it is still far from solved even in simpler classification systems, tackling it in multimodal large language models (MLLMs) is becoming increasingly important. Within MLLMs, uncertainty can stem from any of the diverse sources as well as from their relationships, and further can stem from the unbounded answers in the open-ended setting. To tackle the issues, we propose CoMet, an MLLM uncertainty estimation method by decomposing uncertainty into a context-specific term and a multiplicity-specific term. The former captures ambiguity induced by the given context (e.g., task or prompt), while the latter captures how many plausible answers determined by the context remain compatible with the given input. We train a lightweig
    
[^303]: RANSAC评分的正确做法

    RANSAC Scoring Done Right

    [https://arxiv.org/abs/2606.27385](https://arxiv.org/abs/2606.27385)

    本文提出通过在固定内点划分下以闭式形式边缘化内点尺度、再对划分进行优化来重新设计RANSAC评分，从而消除了对阈值的依赖，仅需一次排序和线性扫描即可求得最大值，且单一闭式表达式能自适应不同真值标注水平的数据集。

    

    最广泛使用的RANSAC变体通过对超过阈值后降为零的逐点分数求和来评估候选模型。这些分数源自高斯-均匀混合模型的负对数似然。阈值将遵循噪声模型的内点与外点区分开来，它是观测噪声水平的函数，也称为内点尺度。我们通过逆转RANSAC推断的常规顺序来消除对阈值的依赖：不是先估计尺度再据此进行评分，而是对固定的内点划分以闭式形式将尺度边缘化，然后在划分上进行优化。所得分数可通过一次排序加一次线性扫描来实现最大化。单一的闭式表达式同时涵盖了有信息先验和无信息先验，因此同一分数可适应真值标注丰富或有限的数据集。所提出的分数仅有一个超参数，即……的半宽度（原文摘要在此处被截断）。

    arXiv:2606.27385v2 Announce Type: replace  Abstract: The most widely used RANSAC variants score a candidate model by summing per-point scores that fall to zero beyond a threshold. These scores derive from the negative log-likelihood of a Gaussian-uniform mixture. The threshold separates inliers, which follow the noise model, from outliers, and it is a function of the observation noise level, also known as the inlier scale. We remove this dependence on the threshold by reversing the usual order of RANSAC inference: rather than estimating the scale and then scoring against it, we marginalize the scale in closed form for a fixed inlier partition and then optimize over partitions. The resulting score is maximized by a single sort followed by a linear sweep. A single closed-form expression spans informative and non-informative priors, so the same score adapts to datasets with extensive or limited ground-truth annotation. The proposed score has a single hyperparameter, the half-width of the 
    
[^304]: 频域储备池计算

    Frequency Domain Reservoir Computing

    [https://arxiv.org/abs/2606.24969](https://arxiv.org/abs/2606.24969)

    提出FRESCO——一种完全在频域运行的回声状态网络架构，通过结构化循环递归将稠密非线性状态更新的复杂度从O(N²)降低到O(N)。

    

    尽管 Transformer 的二次序列长度瓶颈推动了循环模型的复兴，但要有效捕获复杂的动态特性，需要一种能够兼顾高效训练与高度表达性潜在状态的架构。回声状态网络通过利用固定的循环权重来规避时间反向传播，从而获得闭式训练解，提供了一种极具吸引力的方法。然而，要在复杂任务中达到所需的表达能力，需要使用大规模储备池，这暴露出 O(N²) 的状态更新瓶颈，使 ESN 难以匹敌当代循环模型的规模。为解决这一局限，我们提出了频域储备池计算（FRESCO），这是一种完全在频域中运行的 ESN 架构，同时避免了域转换带来的额外开销，通过结构化的循环递归实现稠密非线性更新的 O(N) 复杂度。通过采用……

    arXiv:2606.24969v2 Announce Type: replace  Abstract: While the quadratic sequence-length bottleneck of transformers has fueled a resurgence in recurrent models, effectively capturing complex dynamics requires architectures that balance efficient training with highly expressive latent states. Echo State Networks (ESNs) offer a compelling approach by utilizing fixed recurrent weights to circumvent backpropagation through time, enabling a closed-form training solution. However, achieving the expressivity needed for complex tasks demands large reservoirs, exposing an $\mathcal{O}(N^2)$ state-update bottleneck that prevents ESNs from matching the scale of contemporary recurrent models. To address this limitation, we introduce Frequency Domain Reservoir Computing (FRESCO), an ESN architecture operating entirely in the frequency domain while avoiding domain-shift overheads to achieve $\mathcal{O}(N)$ complexity for dense, non-linear updates via structured circulant recurrence. By employing a 
    
[^305]: 自我建模干预可以调节涌现性失准

    Self-Modeling Interventions can Modulate Emergent Misalignment

    [https://arxiv.org/abs/2606.23700](https://arxiv.org/abs/2606.23700)

    本研究提出将自我识别和自我报告等自我建模干预作为针对模型品格的防御手段，实验证明自我识别微调能有效逆转和预防涌现性失准，且是唯一能持续降低失准而不恶化其他指标的预防性方法。

    

    涌现性失准（EM）已被发现与失准人格向量和邪恶性格特质的激活相关联，这表明EM是通过破坏模型已对齐的品格而非直接学习有害内容来发挥作用的。受这一关联的启发，我们研究了自我建模干预：具体而言，将自我识别和自我报告作为针对品格的干预手段，它们有别于现有的训练中防御方法。我们在三个模型（GPT-4.1、Qwen2.5-32B-Instruct、Seed-OSS-36B-Instruct）和多个EM数据集上进行了两阶段微调实验，将自我识别微调与良性微调基线（正确的领域特定数据、通用知识和数词任务）进行比较，发现它在逆转和预防两种设置中都是有效的防御手段。在预防方面，只有自我识别微调能够持续降低失准程度，同时不加剧任何单项指标。

    arXiv:2606.23700v2 Announce Type: replace-cross  Abstract: Emergent misalignment (EM) has been linked to the activation of misaligned persona vectors and evil character traits, suggesting that EM operates through disruption of the model's aligned character rather than direct learning of harmful content. Motivated by this connection, we study self-modeling interventions: specifically self-recognition and self-reports as character-targeted interventions that are distinct from existing in-training defenses. We conduct two-stage finetuning experiments across three models (GPT-4.1, Qwen2.5-32B-Instruct, Seed-OSS-36B-Instruct) and multiple EM datasets to compare self-recognition finetuning against benign finetuning baselines (correct domain-specific data, general knowledge, and word counting), finding it an effective defense in both reversal and prevention settings. For prevention, only self-recognition finetuning consistently reduces misalignment without exacerbating any individual metric, 
    
[^306]: 位置编码器能否捕捉空间效应？跨尺度的GeoShapley基准测试

    Do Location Encoders Capture Spatial Effects? A GeoShapley Benchmark Across Scales

    [https://arxiv.org/abs/2606.23453](https://arxiv.org/abs/2606.23453)

    该论文提出以GeoShapley博弈论解释器为工具，在三个空间尺度上对TorchSpatial框架中的十一种位置编码器进行基准测试，系统评估其嵌入能否恢复已知的空间变化系数，并揭示恢复效果随尺度与编码器架构的变化规律。

    

    位置编码器将地理坐标转换为高维嵌入，供下游机器学习使用，但这些表示能在多大程度上捕捉可解释的空间效应尚不清楚。我们进行基准测试，评估GeoShapley——一种将所有位置特征视为单一联合参与者的博弈论解释器——能否从基于位置编码器嵌入构建的模型中恢复空间变化的系数。我们使用TorchSpatial框架中的十一种编码器，在一个系数已知的合成过程中进行评估，涵盖三种尺度（网格、县级、全球），分别测试嵌入伴随与不伴随原始坐标的情况，以及未训练与对比训练两种条件。我们以估计系数与真实系数之间的相关性来衡量恢复效果，报告其如何随尺度和编码器架构而变化，并将嵌入与原始坐标基线进行比较。主要系数的恢复始终……（原文摘要在此处截断）

    arXiv:2606.23453v2 Announce Type: replace  Abstract: Location encoders transform geographic coordinates into high dimensional embeddings for downstream machine learning, but it is unclear how well these representations capture interpretable spatial effects. We benchmark whether GeoShapley, a game-theoretic explainer that treats all location features as a single joint player, can recover spatially varying coefficients from models built on location-encoder embeddings. Eleven encoders from the TorchSpatial framework are evaluated against a synthetic process with known coefficients, across three scales (grid, county, global), with and without raw coordinates alongside the embedding, and under untrained and contrastively trained conditions. Measuring recovery as the correlation between estimated and true coefficients, we report how it varies with scale and encoder architecture and compare the embeddings against a raw-coordinate baseline. Recovery of the primary coefficient is consistently h
    
[^307]: 智能体符号搜索：超越手工表达式、网格和神经网络的偏微分方程解逼近

    Agentic Symbolic Search: Approximating Solutions of PDEs Beyond Hand-crafted Expressions, Meshes, and Neural Networks

    [https://arxiv.org/abs/2606.20467](https://arxiv.org/abs/2606.20467)

    提出了智能体符号搜索（ASYS）方法，让编码智能体将可微分符号程序上的进化搜索与基于梯度的拟合相结合，利用控制方程自动构建和改进偏微分方程解的显式逼近，在奇异剖面拟合和爆破问题中实现了极低的误差。

    

    数值方法和神经网络已经改变了我们对非线性偏微分方程（PDE）的理解，然而发现能够解释其动力学的数学表示在很大程度上仍然是专家的任务。我们提出了智能体符号搜索（ASYS），其中编码智能体从已知的偏微分方程和所提供的数学知识出发，构建并改进显式逼近。ASYS 将可微分符号程序上的进化搜索与基于梯度的拟合相结合，利用控制方程来评估候选解并指导其修正。对于一个奇异的广义 Constantin-Lax-Majda 剖面，ASYS 推断出一个缺失的衰减项，该衰减项决定了场接近其奇异时间极限的速度，并构建了一组基函数，使该剖面的拟合中位误差达到 3.90×10⁻⁵%。对于 Fujita 爆破问题，将扩散与反应分离并推导局部响应，将所提供公式的误差从 55.5%降低

    arXiv:2606.20467v2 Announce Type: replace  Abstract: Numerical methods and neural networks have transformed our understanding of nonlinear partial differential equations (PDEs), yet discovering mathematical representations that explain their dynamics remains largely an expert task. We introduce Agentic Symbolic Search (ASYS), in which coding agents construct and refine explicit approximations from a known PDE and supplied mathematical knowledge. ASYS couples evolutionary search over differentiable symbolic programs with gradient-based fitting, using the governing equation to evaluate candidates and guide their revision. For a singular generalized Constantin-Lax-Majda profile, ASYS infers a missing decay term that sets how fast the field approaches its singular-time limit, and builds a basis that fits the profile to a median error of $3.90\times10^{-5}\%$. For Fujita blow-up, separating diffusion from reaction and deriving a local response lowers the supplied formula's error from 55.5% 
    
[^308]: 从通用人工智能到超级人工智能

    From AGI to ASI

    [https://arxiv.org/abs/2606.12683](https://arxiv.org/abs/2606.12683)

    本报告系统刻画了从人类水平的通用人工智能（AGI）向在智力与认知能力上超越大型人类组织的超级人工智能（ASI）过渡的过程，并探讨了后AGI时代机器智能沿连续谱发展的四种潜在路径。

    

    在过去十年中，构建人类水平的通用人工智能（AGI）已从遥不可及的猜想转变为主流研究目标。实现这一目标将对人类社会产生深远而广泛的影响，这也为未来十年提出了许多复杂的问题。本报告探讨了在AGI实现后的世界中，人工智能本身将如何沿着机器智能的连续谱继续发展。该连续谱的终点——通用智能——在理论上已被充分理解，这为本报告的核心议题提供了一定的形式化基础：即从人类水平的AGI向人工超级智能（ASI）的过渡，ASI可以直观地理解为一个在智力和认知能力上超越大型人类组织的系统。在刻画ASI的特征之后，报告讨论了从AGI到ASI的四种潜在路径：扩展AGI、AI范式转变等。

    arXiv:2606.12683v2 Announce Type: replace  Abstract: Over the last decade, building human-level artificial general intelligence has moved from far-fetched speculation to being a concrete next-decade target for many of the largest AI organisations. Achieving this goal would have profound and far-reaching impacts on human society, which raises many complex questions for the decade ahead. This report investigates how AI itself might continue to develop in a post-AGI world along the continuum of machine intelligence. The endpoint of this continuum, Universal AI, is theoretically well understood, which provides some formal grounding for the main focus of this report: the transition from human-level AGI to artificial general superintelligence, which can intuitively be understood as a system that is more intelligent and cognitively capable than large organisations of humans. After characterizing ASI, the report discusses four potential pathways from AGI to ASI: scaling AGI, AI paradigm shifts
    
[^309]: 从架构到输出：大语言模型幻觉的结构性起源与数据的放大作用

    From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data

    [https://arxiv.org/abs/2606.07537](https://arxiv.org/abs/2606.07537)

    该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。

    

    大语言模型会生成流畅、自信但事实上错误的输出。现有的分类体系仅按输出类型对这些失败进行分类——内在性还是外在性、忠实性还是事实性——却从未说明是哪个计算组件产生了特定的失败。我们探讨需要什么条件才能将单个幻觉归因于纯解码器（decoder-only）架构栈中的特定组件。我们将三个组件——自注意力的联想检索、最大似然预训练目标、以及暴露偏差（exposure bias）下的自回归承诺——视为候选失败面，论证而非假定它们的可分离性，并规定了一种仅需采样访问权限的归因程序：一组针对前缀、上下文和频率竞争的有序三步干预，同时辅以基于独立标注和分类器基线的验证设计。我们提出了五个可证伪的预测……

    arXiv:2606.07537v2 Announce Type: replace-cross  Abstract: Large language models produce fluent, confident, factually wrong output. Existing taxonomies classify these failures by output type -- intrinsic versus extrinsic, faithfulness versus factuality -- but say nothing about which computational component produced a given failure. We ask what would be required to attribute an individual hallucination to a specific component of the decoder-only stack. We treat three components -- self-attention's associative retrieval, the maximum-likelihood pretraining objective, and autoregressive commitment under exposure bias -- as candidate failure surfaces, justify their separability rather than assuming it, and specify an attribution procedure requiring only sampling access: an ordered set of three interventions on prefix, context, and frequency competition, together with a validation design based on independent annotation and a classifier baseline. We state five falsifiable predictions and iden
    
[^310]: 通过掩码基因表达实现有效的生物表征学习

    Effective Biological Representation Learning by Masking Gene Expression

    [https://arxiv.org/abs/2605.31562](https://arxiv.org/abs/2605.31562)

    该论文提出了针对多样化RNA-seq计数数据量身定制的掩码自编码自监督模型TxFM，并通过消融研究确定了关键架构配置，显著提升了转录组数据的归纳式表征学习与迁移性能。

    

    RNA测序产生了丰富多样的基因表达数据集，为了解细胞状态和功能提供了极具价值的洞见，在药物发现中有着广泛的应用。由于固有的技术噪声和实验批次效应，对这类数据进行建模极具挑战性，许多现有的转录组基础模型（FMs）表现不及线性基线便印证了这一点。这些结果引发了一个问题：深度表征学习相比直接使用原始转录本计数是否具有独特优势。我们的工作通过开发一个新的自监督模型TxFM来探索这一问题，重点关注归纳式表征学习评估。TxFM采用了一种针对多样化RNA-seq计数数据量身定制的掩码自编码方法，我们的消融研究通过实验确定了实现强迁移性能所需的关键架构配置。此外，我们构建了一个公开的训练语料库DiverseRNA-1.4

    arXiv:2605.31562v3 Announce Type: replace  Abstract: RNA sequencing produces rich and diverse datasets of gene expression, offering compelling insights into cellular state and function that have many applications in drug discovery. Modeling such data is challenging due to inherent technical noise and experimental batch effects, as evidenced by many existing transcriptomic foundation models (FMs) underperforming relative to linear baselines. Such results raise the question of whether deep representation learning provides a distinct advantage over the direct use of raw transcript counts. Our work explores this by developing a new self-supervised model, TxFM, with a focus on inductive representation learning evaluations. TxFM employs a masked autoencoding approach tailored to diverse RNA-seq count data, and our ablation study empirically identifies crucial architecture configurations required for strong transfer performance. Additionally, we curate a public training corpus, DiverseRNA-1.4
    
[^311]: TriSearch：通过双星翻转学习优化三角剖分

    TriSearch: Learning to Optimize Triangulations via Bistellar Flips

    [https://arxiv.org/abs/2605.30220](https://arxiv.org/abs/2605.30220)

    TriSearch是一个基于强化学习的三角剖分优化框架，其核心创新是电路支持的子三角剖分动作表示，实现了与维度无关的翻转图高效遍历，可零样本泛化到搜索空间指数级增大的更大多胞体，并在4维中发现了更多对应Calabi-Yau三维流形的三角剖分。

    

    我们提出了TriSearch，一个通过双星翻转来优化多胞体三角剖分目标的强化学习框架。其核心思想是一种基于电路支持的子三角剖分动作表示：可行的翻转通过其支撑电路和所实现的局部子三角剖分进行编码，使学习到的策略能够利用局部几何和组合特征对这些翻转进行排序。这产生了一个与维度无关的接口，并能够在不显式枚举完整三角剖分空间的情况下高效遍历翻转图。在3维和4维中实例化后，TriSearch能够从小型训练实例零样本泛化到搜索空间呈指数级增大的更大多胞体。它在3维的度量目标上取得了顶尖性能；在4维中，在固定预算下，它发现了比现有采样器更多的反射多胞体上互不相同的精细、正则、星三角剖分，这些三角剖分对应于Calabi-Yau三维流形。

    arXiv:2605.30220v2 Announce Type: replace  Abstract: We introduce TriSearch, a reinforcement learning framework for optimizing objectives over triangulations of a polytope via bistellar flips. The key idea is a circuit-supported subtriangulation action representation: feasible flips are encoded by their supporting circuit and realized local subtriangulation, enabling a learned policy to rank them using local geometric and combinatorial features. This yields a dimension-agnostic interface and enables efficient traversal of the flip graph without explicit enumeration of the full triangulation space. Instantiated in 3D and 4D, TriSearch generalizes zero-shot from small training instances to larger polytopes with exponentially larger search spaces. It achieves top performance on metric objectives in 3D and, in 4D, discovers more distinct Fine, Regular, Star triangulations of reflexive polytopes, corresponding to Calabi-Yau threefolds, than existing samplers under a fixed budget.
    
[^312]: 将混合专家模型剪枝并蒸馏为稠密语言模型

    Pruning and Distilling Mixture-of-Experts into Dense Language Models

    [https://arxiv.org/abs/2605.28207](https://arxiv.org/abs/2605.28207)

    提出首个将训练好的MoE模型系统性转换为全稠密语言模型的框架，通过专家评分、选择、分组拼接为稠密FFN并经知识蒸馏优化，其中新颖的多样性感知评分方法在多个模型上一致优于已有方法。

    

    混合专家模型已成为当前前沿语言模型的主流架构，但它需要将所有专家参数加载到内存中，因此在内存受限的部署场景下并不理想。现有的压缩方法只能减少专家数量，输出结果仍然是具有相同根本局限的MoE模型。我们提出了首个将训练好的MoE转换为标准全稠密架构的系统性框架：对专家进行评分、选择和分组，然后将其拼接为稠密的前馈网络（FFN），并通过来自MoE教师模型的知识蒸馏进行精细化优化。我们在Qwen3-30B-A3B上，针对不同的专家选择数量，评估了7种评分方法、5种分组方法和2种幅度缩放方法，共产生350种配置。我们发现评分方法的选择影响最大，我们提出的新型多样性感知评分方法在Qwen3-30B-A3B、DeepSeek-V2-Lite和GPT-OSS上一致优于已有方法。

    arXiv:2605.28207v3 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) is now the dominant architecture for frontier language models, yet it requires all expert parameters to be loaded in memory, making it less preferable for memory-constrained deployment. Existing compression methods reduce the number of experts but the output remains an MoE model with the same fundamental limitation. We present the first systematic framework for converting a trained MoE into a standard fully dense architecture: experts are scored, selected, and grouped, then concatenated into a dense FFN and refined by knowledge distillation from the MoE teacher. We evaluate 7 scoring, 5 grouping, and 2 magnitude scaling methods across a range of selected expert counts on Qwen3-30B-A3B, yielding 350 configurations. We find that the choice of scoring method is the most impactful, with our novel diversity-aware scoring consistently outperforming prior methods on Qwen3-30B-A3B, DeepSeek-V2-Lite, and GPT-OSS
    
[^313]: 面向多智能体大语言模型决策系统中动态信用分配的市场体制委员会

    Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems

    [https://arxiv.org/abs/2605.24490](https://arxiv.org/abs/2605.24490)

    本文提出了市场体制委员会（MRC），一种基于精确 Shapley 信用的在线信用分配机制，通过联盟绩效加权、贝叶斯自适应混合和体制依赖的权限乘数，解决了多智能体大语言模型投资组合管理系统中信用分配缺乏原则、冷启动主导和透明度不足的问题。

    

    用于投资组合管理的多智能体大语言模型决策系统仍然缺乏一种在专业智能体之间进行信用分配的原则性方法，在市场体制转换时容易受到冷启动主导的影响，并且对最终配置的形成过程缺乏透明度。我们提出了市场体制委员会（Market Regime Council, MRC），这是一种针对专业智能体联盟的在线信用分配机制，以投资组合管理作为下游任务。MRC 在所有单个、成对以及大联盟输出上计算精确的 Shapley 信用，用于智能体的在线加权。在实例化为 N=3 个专业智能体时，MRC 在每个交易周期从指数加权的历史表现中重新计算基于联盟的 Shapley 权重，使用贝叶斯自适应混合来稳定早期阶段，应用依赖于市场体制的乘数来调整智能体权限，并通过五层因果追踪记录每次再平衡。在 13 种加密资产上超过 1,037 个交易日的实验……

    arXiv:2605.24490v2 Announce Type: replace  Abstract: Multi-agent LLM decision systems for portfolio management still lack a principled way to assign credit across specialist agents, remain vulnerable to cold-start dominance under regime shifts, and offer limited transparency into how final allocations are formed. We propose Market Regime Council (MRC), an online credit-assignment mechanism over coalitions of specialist agents, with portfolio management as the downstream task. MRC computes exact Shapley credits across all single, pairwise, and Grand-coalition outputs for online agent weighting. Instantiated with N=3 specialist agents, at each trading period, MRC recomputes coalition-based Shapley weights from exponentially weighted performance histories, uses a Bayesian adaptive mixture to stabilize early periods, applies regime-dependent multipliers to adjust agent authority, and records each rebalance through a five-layer causal trace. Over 1,037 trading days across 13 crypto assets a
    
[^314]: 基于扩散奖励的扩散指令：迈向有原则的一步生成器强化学习

    Diff-Instruct with Diffused Reward: Towards Principled One-step Generator RL

    [https://arxiv.org/abs/2605.24001](https://arxiv.org/abs/2605.24001)

    该论文提出 DIDR 框架，通过积分 KL 散度最小化，将 RLHF 最优的奖励倾斜干净图像分布沿扩散轨迹传播到所有噪声水平，解决了现有一步生成器强化学习中终端奖励优化与生成动力学不匹配导致以图像保真度换取奖励提升的问题。

    

    arXiv:2605.24001v3 公告类型：replace-cross。摘要：一步式文本到图像生成的最新进展使实时合成兼具出色的效率与质量。以往面向一步生成器的强化学习方法将图像空间的奖励优化与扩散噪声空间的分布匹配相结合。然而，由于终端奖励优化与底层生成动力学之间的不匹配，这一范式带来了挑战。其结果是，优化过程往往倾向于利用随机自由度，常常以牺牲图像保真度为代价来换取奖励的提升。为解决这一问题，我们提出了基于扩散奖励的扩散指令（DIDR），这是一个源自积分 KL 散度最小化的免数据轨迹级对齐框架。DIDR 将 RLHF 最优的、经奖励倾斜的干净图像分布沿扩散轨迹传播至所有噪声水平上。我们证明该目标与干净图像 RLHF 具有相同的最小化器，同时自然地引出扩散（原文摘要在此处被截断）。

    arXiv:2605.24001v3 Announce Type: replace-cross  Abstract: Recent advances in one-step text-to-image generation have enabled real-time synthesis with remarkable efficiency and quality. Previous reinforcement learning methods for one-step generators combine image-space reward optimization with diffusion noisy-space distribution matching. This paradigm brings challenges due to a mismatch between terminal reward optimization and the underlying generative dynamics. As a result, optimization tends to exploit stochastic degrees of freedom, often improving reward at the expense of image fidelity. To address this issue, we propose Diff-Instruct with Diffused Reward (DIDR), a data-free trajectory-level alignment framework derived from Integral KL minimization. DIDR propagates the RLHF-optimal reward-tilted clean-image distribution across all noise levels along the diffusion trajectory. We show that this objective admits the same minimizer as clean-image RLHF, while naturally inducing the Diffus
    
[^315]: 浅层神经网络中随时间一致的弱混沌传播

    Uniform-in-Time Weak Propagation-of-Chaos in Shallow Neural Networks

    [https://arxiv.org/abs/2605.22010](https://arxiv.org/abs/2605.22010)

    该论文通过利用平均场确定性 Wasserstein 梯度流动的收敛速率，而非依赖强凸性或对数 Sobolev 不等式，首次在浅层神经网络的梯度下降训练中建立了随时间一致成立的非渐近弱混沌传播结果。

    

    我们考虑在特征学习机制下使用梯度下降训练的单隐层神经网络，并将有限宽度网络的输出 $f_{\hat{\rho}_t^m}$ 与其无穷宽度对应物 $f_{\rho_t^{MF}}$（按平均场动力学演化）建立联系。虽然可以通过标准的 Grönwall 估计得到 $\|f_{\rho_t^{MF}} - f_{\hat{\rho}_t^m}\|$ 在常数时间范围内的界，但波动的长时间行为是一个更为微妙的问题。随时间一致的界通常依赖于损失景观中的（局部）强凸性，或带噪梯度动力学中存在的对数 Sobolev 不等式。在这项工作中，我们建立了随时间一致成立的非渐近弱混沌传播，这是通过利用平均场确定性 Wasserstein 梯度流动动力学的收敛速率而实现的。具体而言，记 $L_t$ 为时刻 $t$ 的平均场超额 MSE 损失，$m$ 为神经元数量……（摘要在此处被截断）

    arXiv:2605.22010v2 Announce Type: replace-cross  Abstract: We consider one-hidden layer neural networks trained in the feature-learning regime using gradient descent, and relate the output of the finite-width network $f_{\hat{\rho}_t^m}$ to its infinite-width counterpart $f_{\rho_t^{MF}}$, which evolves in the mean-field dynamics.   While constant-time horizon bounds for $\|f_{\rho_t^{MF}} - f_{\hat{\rho}_t^m}\|$ may be obtained via standard Gr\"onwall estimates, the long-time behavior of the fluctuation is a more delicate matter. Uniform-in-time bounds often rely on (local) strong convexity in the landscape or Logarithmic Sobolev inequalities present in noisy gradient dynamics.   In this work, we establish non-asymptotic weak propagation-of-chaos that holds uniformly in time, obtained by exploiting instead the convergence rate of the mean-field deterministic Wasserstein-gradient-flow dynamics. Specifically, denoting by $L_t$ the mean-field excess MSE loss at time $t$ and $m$ the numbe
    
[^316]: 扩散教师模型梯度的方差削减

    Variance Reduction for Gradients from Diffusion Teachers

    [https://arxiv.org/abs/2605.21489](https://arxiv.org/abs/2605.21489)

    提出CARV框架，将扩散教师梯度估计建模为计算感知的两阶段采样问题，结合分层蒙特卡洛估计、时间步重要性采样与分层逆CDF采样，在不改变目标函数的前提下为文本到3D和数据归因带来2至3倍的有效计算提升。

    

    预训练扩散模型可作为冻结的教师模型，服务于文本到3D生成、单步蒸馏和数据归因等流程。这些流程所消耗的教师梯度是关于噪声水平和高斯噪声的期望，通过蒙特卡洛（MC）方法估计。每次MC采样都需要昂贵的上游计算（如渲染、模拟、编码），因此估计器的方差主导了计算成本。我们提出了CARV，一个计算感知的方差核算框架，它将这些梯度建模为一个具有不对称每层成本的经典两阶段采样问题。这启发了一种分层蒙特卡洛估计器，通过将每次昂贵的上游计算在多个廉价的扩散采样中复用来摊销其成本，并通过时间步重要性采样（IS）和分层逆CDF采样进一步降低单元内方差。CARV在不改变目标函数的情况下，为文本到3D和数据归因任务带来了2至3倍的有效计算乘数。

    arXiv:2605.21489v3 Announce Type: replace-cross  Abstract: Pretrained diffusion models serve as frozen teachers for pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are expectations over noise levels and Gaussian noise, estimated by Monte Carlo (MC). Each MC draw requires expensive upstream work (rendering, simulation, encoding), so estimator variance dominates compute cost. We introduce CARV, a compute-aware variance-accounting framework that casts these gradients as a classical two-stage sampling problem with asymmetric per-level costs. This motivates a hierarchical MC estimator that amortizes each expensive upstream computation by reusing it across cheap diffusion draws, with within-unit variance further reduced by timestep importance sampling (IS) and stratified inverse-CDF sampling. CARV delivers 2 to 3x effective compute multipliers on text-to-3D and data attribution without changing the objective. In text
    
[^317]: CRAFT：面向联邦训练的冲突消解聚合

    CRAFT: Conflict-Resolved Aggregation for Federated Training

    [https://arxiv.org/abs/2605.21317](https://arxiv.org/abs/2605.21317)

    提出CRAFT聚合框架，将联邦学习中冲突客户端更新的聚合建模为带无冲突对齐约束的几何校正问题，利用闭式解避免迭代求解开销，并通过逐层自适应处理不同粒度的冲突，从而缓解特定客户端的性能退化。

    

    arXiv:2605.21317v2 公告类型：替换 摘要：在异构数据分布下，相互冲突的客户端更新的聚合仍然是联邦学习（FL）中的一个根本性瓶颈。朴素平均可能产生一个虽然改善了全局目标、却与特定客户端相冲突的全局更新，从而导致这些客户端的性能下降。在本工作中，我们提出了CRAFT（面向联邦训练的冲突消解聚合），这是一种新的聚合框架，它将全局更新视为一个几何校正问题。我们将聚合形式化为：在满足无冲突对齐约束的条件下，寻找最接近参考方向的更新。我们为该约束优化问题推导出了闭式解表达式，避免了迭代求解器带来的计算开销。此外，我们采用逐层自适应机制来处理不同特征粒度上的冲突。我们提供了理论分析，表明CRAFT能够促进一种共同下降结构……（摘要原文在此处截断）

    arXiv:2605.21317v2 Announce Type: replace  Abstract: The aggregation of conflicting client updates remains a fundamental bottleneck in federated learning (FL) under heterogeneous data distributions. Naive averaging can produce a global update that improves the global objective while conflicting with specific clients, causing degradation for those clients. In this work, we propose CRAFT (Conflict-Resolved Aggregation for Federated Training), a new aggregation framework that treats the global update as a geometric correction problem. We formulate aggregation as finding the update closest to a reference direction while satisfying conflict-free alignment constraints. We derive a closed-form expression for the constrained optimization problem, avoiding the computational overhead of iterative solvers. Furthermore, we use a layer-wise adaptation to address conflicts at varying feature granularities. We provide a theoretical analysis showing that CRAFT promotes a common-descent structure and m
    
[^318]: 蒸馏以思考，预见以行动：面向自动驾驶的认知-物理强化学习

    Distill to Think, Foresee to Act: Cognitive-Physical Reinforcement Learning for Autonomous Driving

    [https://arxiv.org/abs/2605.21139](https://arxiv.org/abs/2605.21139)

    CoPhy 框架通过将视觉语言模型知识蒸馏进 BEV 编码器实现零推理成本的认知能力，并构建自回归 BEV 世界模型来预测候选动作的未来后果，从而突破模仿学习的行为克隆上限，实现自动驾驶的认知-物理强化学习。

    

    当前的端到端自动驾驶模型从根本上受限于模仿学习的行为克隆上限。虽然强化学习为实现更智能的自主性提供了一条路径，但它需要两个尚缺失的基础设施：(1) 一个能够理解交通语义和驾驶意图的认知基础；(2) 一个能够预判候选动作后果的具有前瞻性的物理环境。为此，我们提出了 CoPhy，一个面向自动驾驶的认知-物理强化学习框架。为了“蒸馏以思考”，我们将视觉语言模型（VLM）的知识蒸馏到 BEV 编码器中，然后完全舍弃 VLM，在零推理成本下保留认知能力，同时将认知通道释放为可插拔接口，以支持可选的人类自然语言指令。为了“预见以行动”，我们构建了一个自回归的 BEV 世界模型，该模型以候选动作为条件显式预测未来的语义地图，为……

    arXiv:2605.21139v4 Announce Type: replace-cross  Abstract: Current end-to-end autonomous driving models are fundamentally constrained by the behavioral cloning ceiling of imitation learning. While reinforcement learning offers a path to smarter autonomy, it demands two missing pieces of infrastructure: (1) a cognitive foundation that understands traffic semantics and driving intent, and (2) a foresighted physical environment that can anticipate the consequences of candidate actions. To this end, we propose CoPhy, a CognitivePhysical reinforcement learning framework for autonomous driving. To distill to think, we distill VLM knowledge into the BEV encoder and then discard the VLM entirely, retaining cognitive ability at zero inference cost while releasing the cognitive channel as a pluggable interface for optional human language commands. To foresee to act, we build an auto-regressive BEV world model that explicitly predicts future semantic maps conditioned on candidate actions, serving
    
[^319]: 用于事后学习延后的密度比损失

    Density-Ratio Losses for Post-Hoc Learning to Defer

    [https://arxiv.org/abs/2605.19557](https://arxiv.org/abs/2605.19557)

    该论文提出用模型与专家理想分布之间的密度比来定义事后学习延后，推导出两种 DR CPE 损失用于训练延后评分器，其中联合变体在 KL 散度情形下与 Chow 规则及专家倾斜的贝叶斯后验相联系且计算便利。

    

    我们通过理想分布的视角研究事后学习延后：理想分布是指数据分布在发散度正则化下的重加权，使得模型在该分布下能取得较低的任务损失。我们通过模型理想与专家理想之间的密度比来定义延后。利用从密度比估计到类概率估计的归约，我们推导出适用于事后 L2D 评分器的 DR CPE 损失。延后决策随后通过对评分器进行阈值化来做出。对于基于 KL 散度的理想分布，我们的延后规则在原始分布下可恢复 Chow 规则，并且（取决于理想分布是边缘分布还是联合分布）与专家倾斜的贝叶斯后验——其中纳入了专家的表现——建立起联系。这导出了 DR CPE 损失的两种变体，其中联合变体在实践中计算上更为便利。当……时，我们建立了联合 DR CPE 损失的一致性性质。

    arXiv:2605.19557v2 Announce Type: replace-cross  Abstract: We study post-hoc Learning to Defer (L2D) through the lens of ideal distributions: divergence-regularized reweightings of the data distribution under which a model attains low task loss. We define deferral via the density-ratio between a model's and an expert's ideals. Using the reduction from density-ratio estimation to class-probability estimation, we derive the DR CPE losses for post-hoc L2D scorers. Deferral decisions are then made by thresholding the scorer. For KL-based ideal distributions, our deferral rule recovers Chow's rule under the original distribution and a connection to an expert-tilted Bayes posterior---which incorporates the expert's performance---depending on if the ideal distributions are marginal or joint distributions. This leads to two variants of the DR CPE losses, with the joint variant being computationally convenient in practice. We establish consistency properties of the joint DR CPE losses when cons
    
[^320]: 因子增强的高维随机梯度下降

    Factor Augmented High-Dimensional SGD

    [https://arxiv.org/abs/2605.19291](https://arxiv.org/abs/2605.19291)

    提出了一种完全基于流式数据的因子增强随机梯度下降方法（FSGD），并首次建立了将潜在因子估计误差纳入SGD分析的理论框架，为高维机器学习提供了可靠且可扩展的优化基础。

    

    随机梯度下降（SGD）是一种在现代机器学习中广泛应用的基础优化算法。本文提出了因子增强随机梯度下降（FSGD），这是一种新的优化方法，能够在高维学习任务中利用潜在因子表示。与依赖离线表示学习和完整数据存储的标准两阶段降维方法不同，FSGD的一个关键创新在于它完全基于流式数据运行，使其能够扩展到大规模和高维问题。此外，我们建立了首个将潜在因子估计误差显式纳入SGD分析的理论框架，并在衰减步长和小批量更新条件下给出了$\ell^s$范数的矩收敛性结果。我们的成果为在高维机器学习系统中可靠且可扩展地应用SGD提供了新的理论基础。

    arXiv:2605.19291v2 Announce Type: replace-cross  Abstract: Stochastic gradient descent (SGD) is a fundamental optimization algorithm widely used in modern machine learning. In this paper, we propose Factor-Augmented SGD (FSGD), a new optimization method that leverages latent factor representations in high-dimensional learning tasks. Unlike standard two-stage dimension reduction approaches that rely on offline representation learning and full data storage, a key novelty of FSGD is that it operates purely on streaming data, making it scalable to large-scale and high-dimensional problems. Furthermore, we establish the first theoretical framework that explicitly incorporates latent factor estimation error into the analysis of SGD, and provide moment convergence in $\ell^s$ norm under decaying step sizes and mini-batch updates. Our results provide a new foundation for employing SGD reliably and scalably in high-dimensional machine learning systems.
    
[^321]: AI中介的交流能够引导集体舆论

    AI-Mediated Communication Can Steer Collective Opinion

    [https://arxiv.org/abs/2605.16245](https://arxiv.org/abs/2605.16245)

    该论文通过实证研究与数学建模表明，大型语言模型在中介人与人之间的交流时会引入方向性偏见（如偏向支持枪支管制、反对无神论），并因此可能引导社交网络中集体舆论的形成。

    

    生成式人工智能（AI）正日益融入人们交换意见的在线平台；大型语言模型（LLM）已经在LinkedIn上润色用户的帖子，并为X上分享的内容提供背景信息。尽管已有研究表明AI能够表达带有偏见的观点，并在人机交互中影响个人的意见，但对于AI在中介人与人之间交流时对集体舆论形成的影响，研究关注还比较少。我们通过实证分析与理论分析相结合的方式来填补这一空白。我们在实证上表明，来自多个流行家族的大型语言模型在被要求编辑涉及争议话题的人类撰写文本时会引入方向性偏差，例如使文本倾向于支持枪支管制、反对无神论。基于这一观察，我们提出了一个舆论动力学的数学模型，其中一个AI系统位于社交网络上用户之间。

    arXiv:2605.16245v2 Announce Type: replace-cross  Abstract: Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) have been polishing users' posts on LinkedIn and providing context for content shared on X. While prior work has shown that AI can express biased opinions and shape individuals' opinions during human-AI interactions, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. We address this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social networ
    
[^322]: 观其行而知其名：通过UI轨迹对LLM浏览器代理进行指纹识别

    Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces

    [https://arxiv.org/abs/2605.14786](https://arxiv.org/abs/2605.14786)

    研究表明网站可通过被动JavaScript追踪器捕获LLM浏览器代理的UI操作轨迹与交互时间，以高达96%的F1分数识别其底层模型，从而带来被定向攻击的安全风险。

    

    随着基于LLM的代理越来越多地代表用户浏览网页，一个自然的问题随之而来：网站能否被动地识别出驱动代理的底层模型？如果能，这将构成重大安全风险，使针对已知模型漏洞的定向攻击成为可能。在14个前沿LLM和四个涵盖信息检索与购物任务的Web环境中，我们展示了通过被动JavaScript追踪器捕获的代理行为和交互时间，足以在固定测试框架内以高达96%的F1分数识别底层模型。尽管指纹具有框架特定性——跨框架迁移时识别效果降至接近随机水平——但汇集跨框架的轨迹仍能以0.74至0.81的Macro-F1恢复模型识别能力。我们进一步表明，仅用约三分之一的可用轨迹即可训练出强大的分类器，并且随着观察到的事件数量增加，识别性能会稳步提升。

    arXiv:2605.14786v2 Announce Type: replace-cross  Abstract: As LLM-based agents increasingly browse the web on users' behalf, a natural question arises: can websites passively identify which underlying model powers an agent? Doing so would represent a significant security risk, enabling targeted attacks tailored to known model vulnerabilities. Across 14 frontier LLMs and four web environments spanning information retrieval and shopping tasks, we show that an agent's actions and interaction timings, captured via a passive JavaScript tracker, are sufficient to identify the underlying model with up to 96\% F1 within a fixed harness. Although fingerprints are harness-specific, with cross-harness transfer falling to near chance, pooling traces across harnesses recovers model identification at 0.74 to 0.81 Macro-F1. We further show that strong classifiers can be trained from roughly one third of the available traces, and that identification improves steadily as more events are observed. Injec
    
[^323]: 面向经典规划的提升动作模式可微分学习

    Differentiable Learning of Lifted Action Schemas for Classical Planning

    [https://arxiv.org/abs/2605.13282](https://arxiv.org/abs/2605.13282)

    该论文提出了一种新颖的神经网络架构，能够以可微分的方式从状态完全可观测但动作参数不可观测的轨迹中学习经典规划的提升动作模式，为从图像序列学习规划领域奠定了重要基础。

    

    经典规划器能够有效求解以 STRIPS 或 PDDL 表示的超大规模确定性 MDP，其中状态是关于对象和关系的原子集合，而提升动作模式则对这些原子进行添加或删除。这种紧凑的表示方式能够产生强大的搜索启发式方法，并为结构泛化提供了理想的环境，因为提升的关系和动作模式可以生成无限多个领域实例。一个核心挑战是从数据中学习这些关系和动作模式，近期的方法已经使用不同类型的观测来解决这一问题。在这项工作中，我们开发了一种新颖的神经网络架构，用于从轨迹中学习动作模式，其中状态是完全可观测的，但动作参数是不可观测的。这个问题是一种简化，但它是从图像序列和动作标签中学习规划领域的重要一步，我们旨在几乎完美地解决这一简化问题。

    arXiv:2605.13282v4 Announce Type: replace  Abstract: Classical planners can effectively solve very large deterministic MDPs represented in STRIPS or PDDL where states are sets of atoms over objects and relations, and lifted action schemas add or delete these atoms. This compact representation yields strong search heuristics and provides an ideal setting for structural generalization, since lifted relations and action schemas give rise to infinitely many domain instances. A central challenge is to learn these relations and action schemas from data, and recent approaches have addressed this problem using different types of observations. In this work, we develop a novel neural network architecture for learning action schemas from traces where states are fully observed but action arguments are unobserved. The problem is a simplification but an important step towards learning planning domains from sequences of images and action labels, and we aim to solve this simplification in a nearly per
    
[^324]: 考察数据与训练方法对涌现性失对齐与阈下失对齐的影响

    Examining the Impact of Data and Training Methods on Emergent and Subliminal Misalignment

    [https://arxiv.org/abs/2605.12798](https://arxiv.org/abs/2605.12798)

    该论文通过构建按“领域-任务”网格组织的数据集，系统研究了训练数据、评估数据与训练方法如何影响涌现性失对齐（EM）与阈下学习（SL）中失对齐的转移，并发现 EM 主要沿任务而非领域进行转移。

    

    对语言模型进行后训练可能引发远超预期训练目标的惊人行为变化。我们研究了两个尤为显著的例子：涌现性失对齐（EM），即在狭窄数据集上进行微调会导致模型在无关情境中表现出失对齐行为；以及阈下学习（SL），即学生模型通过看似无害的训练数据继承教师模型的失对齐特性。我们考察了训练数据、评估数据和训练方法如何决定失对齐在数据与模型之间于何处转移以及转移的强度。我们用提示词的领域（其主题语境）和任务（模型所执行的功能）来刻画数据，并构建了按“领域-任务”网格组织的自然语言数据集与合成数据集。我们发现，与共享微调数据领域的提示词相比，EM 更多地转移到与微调数据共享相同任务的提示词上。在合成环境中，模型已……（原文摘要在此处截断）

    arXiv:2605.12798v2 Announce Type: replace-cross  Abstract: Post-training a language model can induce surprising behavioral changes far beyond the intended training target. We study two particularly striking examples: emergent misalignment (EM), where fine-tuning on a narrow dataset leads to misaligned behavior in unrelated contexts, and subliminal learning (SL), where a student model inherits a teacher's misalignment through seemingly benign training data. We examine how the training data, evaluation data, and training method shape where and how strongly misalignment transfers between data and models. We characterize the data by the domain of a prompt (its topical context) and its task (the function the model performs), and we build natural-language and synthetic datasets organized as domain-task grids. We find that EM transfers more strongly to prompts that share the task of the fine-tuning data than to prompts that share its domain. In the synthetic setting, tasks that the model has 
    
[^325]: 理解预测编码中的样本效率

    Understanding Sample Efficiency in Predictive Coding

    [https://arxiv.org/abs/2605.11911](https://arxiv.org/abs/2605.11911)

    该论文提出“目标对齐”度量指标，从理论和实证上证明预测编码的学习样本效率优于反向传播，尤其在深层、狭窄和预训练网络中优势显著。

    

    预测编码是一种有影响力的皮层学习理论解释。近期的许多研究工作集中于将预测编码与反向传播进行比较，以探究预测编码是否具有任何优势。小规模实验表明，预测编码在许多情境下能够实现样本效率更高且更有效的学习，但对这一现象的透彻理论理解仍然缺失。为了解决这一问题，我们通过一个名为“目标对齐”的度量指标来量化反向传播和预测编码中的学习效率，该指标衡量网络输出的变化与输出预测误差之间的对齐程度。随后，我们推导并通过实证验证了深度线性网络中目标对齐的解析表达式。我们表明，预测编码中的学习比反向传播更高效，这一优势在深层、狭窄和预训练网络中尤为显著。我们还推导出预测编码中保证最优目标对齐的精确条件，并对我们的发现进行了验证。

    arXiv:2605.11911v3 Announce Type: replace  Abstract: Predictive Coding (PC) is an influential account of cortical learning. Much of recent work has focused on comparing PC to Backpropagation (BP) to find whether PC offers any advantages. Small scale experiments show that PC enables learning that is more sample efficient and effective in many contexts, though a thorough theoretical understanding of the phenomena remains elusive. To address this, we quantify the efficiency of learning in BP and PC through a metric called ``target alignment'', which measures how closely the change in the output of the network is aligned to the output prediction error. We then derive and empirically validate analytical expressions for target alignment in Deep Linear Networks. We show that learning in PC is more efficient than BP, which is especially pronounced in deep, narrow and pre-trained networks. We also derive exact conditions for guaranteed optimal target alignment in PC and validate our findings th
    
[^326]: 树集成模型的极小化极大速率与谱蒸馏

    Minimax Rates and Spectral Distillation for Tree Ensembles

    [https://arxiv.org/abs/2605.11841](https://arxiv.org/abs/2605.11841)

    本文从谱理论视角研究随机森林与梯度提升机，证明了随机森林回归的极小化极大最优收敛速率（由核算子特征值衰减决定），并利用主特征函数和主奇异向量开发出体积缩小数个数量级且预测性能仍具竞争力的树集成模型蒸馏压缩方法。

    

    树集成模型，如随机森林（RF）和梯度提升机（GBM），是最广泛使用的监督学习方法之一，但其理论性质仍未被完全理解。我们对这些算法采用谱视角进行分析，包含两个主要贡献。首先，我们推导了随机森林回归的极小化极大最优收敛性，证明在树生长的温和正则性条件下，诱导核算子的特征值衰减决定了统计收敛速率。其次，我们利用这一谱视角为树集成模型开发了压缩方案。对于随机森林，核算子的主特征函数能够捕捉主导的预测方向；对于梯度提升机，平滑矩阵的主奇异向量则起到类似的作用。通过为这些谱表示学习非线性映射，可以得到比原始模型小数个数量级、同时保持有竞争力的预测性能的蒸馏模型。

    arXiv:2605.11841v3 Announce Type: replace-cross  Abstract: Tree ensembles such as random forests (RFs) and gradient boosting machines (GBMs) are among the most widely used supervised learners, yet their theoretical properties remain incompletely understood. We adopt a spectral perspective on these algorithms, with two main contributions. First, we derive minimax-optimal convergence for RF regression, showing that, under mild regularity conditions on tree growth, the eigenvalue decay of the induced kernel operator governs the statistical rate. Second, we exploit this spectral viewpoint to develop compression schemes for tree ensembles. For RFs, leading eigenfunctions of the kernel operator capture the dominant predictive directions; for GBMs, leading singular vectors of the smoother matrix play an analogous role. Learning nonlinear maps for these spectral representations yields distilled models that are orders of magnitude smaller than the originals while maintaining competitive predict
    
[^327]: 各向异性Besov空间中稀疏Kolmogorov-Arnold网络的后验收缩速率

    Posterior Contraction Rates for Sparse Kolmogorov-Arnold Networks in Anisotropic Besov Spaces

    [https://arxiv.org/abs/2605.11652](https://arxiv.org/abs/2605.11652)

    本文证明了配备尖峰-厚板稀疏先验的稀疏贝叶斯Kolmogorov-Arnold网络在各向异性Besov空间上可达到接近极小极大的后验收缩速率，能通过超先验自适应未知的光滑度，且网络深度可保持固定。

    

    我们研究了各向异性Besov空间上稀疏贝叶斯Kolmogorov-Arnold网络（KANs）的后验收缩速率，从贝叶斯的视角为KANs提供了统计学基础。我们证明，配备尖峰-厚板型稀疏先验的稀疏贝叶斯KANs能够达到接近极小极大的后验收缩速率。特别地，该收缩速率取决于目标函数本身的内在各向异性光滑度。此外，通过对单一模型规模参数设置超先验，所得后验能够自适应未知的各向异性光滑度，并仍然达到相应的接近极小极大的速率。与标准多层感知机的相关结果相比，我们结果的一个显著特点是KAN的深度可以保持固定：得益于可学习样条边函数的表达能力，所需的近似复杂度通过网络宽度以及稀疏性的范围和规模来控制。

    arXiv:2605.11652v2 Announce Type: replace-cross  Abstract: We study posterior contraction rates for sparse Bayesian Kolmogorov-Arnold networks (KANs) over anisotropic Besov spaces, providing a statistical foundation for KANs from a Bayesian point of view. We show that sparse Bayesian KANs equipped with spike-and-slab-type sparsity priors attain a near-minimax posterior contraction rate. In particular, the contraction rate depends on the intrinsic anisotropic smoothness of the underlying function. Moreover, by placing a hyperprior on a single model-size parameter, the resulting posterior adapts to unknown anisotropic smoothness and still achieves the corresponding near-minimax rate. A distinctive feature of our results, compared with those for standard multilayer perceptrons, is that the KAN depth can be kept fixed: owing to the expressive power of learnable spline edge functions, the required approximation complexity is controlled through the network width, the range and size of the sp
    
[^328]: 仿射追踪：概率线性求解器的新范式

    Affine Tracing: A New Paradigm for Probabilistic Linear Solvers

    [https://arxiv.org/abs/2605.10566](https://arxiv.org/abs/2605.10566)

    论文证明贝叶斯概率线性求解器只是非平稳仿射概率迭代方法的特例，并提出“仿射追踪”框架，能够从仿射迭代方法的标准实现中自动构建概率迭代方法，从而推动其广泛实际应用。

    

    概率线性求解器（PLSs）返回概率分布，用于量化求解线性系统时因计算资源有限而产生的不确定性。文献中传统上将贝叶斯PLS（其基于从线性系统投影中获得的信息对先验进行条件化）与概率迭代方法进行区分，后者将经典迭代求解器提升到概率空间。在本工作中，我们证明这种二分法是不成立的：贝叶斯PLS是非平稳仿射PIM的一种特例。此外，我们证明了任何现实的仿射PIM都是经过校准的。这些结果促使人们关注（非平稳）仿射PIM，但其实际应用一直受到实现它们所需的大量手工工作的限制。为解决这一问题，我们提出了仿射追踪，这是一种算法框架，能够通过仿射迭代方法的标准实现自动构建PIM……

    arXiv:2605.10566v2 Announce Type: replace-cross  Abstract: Probabilistic linear solvers (PLSs) return probability distributions that quantify uncertainty due to limited computation in the solution of linear systems. The literature has traditionally distinguished between Bayesian PLSs, which condition a prior on information obtained from projections of the linear system, and probabilistic iterative methods (PIMs), which lift classical iterative solvers to probability space. In this work we show this dichotomy to be false: Bayesian PLSs are a special case of non-stationary affine PIMs. In addition, we prove that any realistic affine PIM is calibrated. These results motivate a focus on (non-stationary) affine PIMs, but their practical adoption has been limited by the significant manual effort required to implement them. To address this, we introduce affine tracing, an algorithmic framework that automatically constructs a PIM from a standard implementation of an affine iterative method by 
    
[^329]: MC$^2$：面向快速椭圆偏微分方程求解的蒙特卡洛校正方法

    MC$^2$: Monte Carlo Correction for Fast Elliptic PDE Solving

    [https://arxiv.org/abs/2605.09288](https://arxiv.org/abs/2605.09288)

    MC$^2$通过学习单次神经校正来修正低预算蒙特卡洛PDE解，以少量采样达到超过1000倍样本量的求解精度，并发布了最大规模的标准化椭圆PDE基准数据集PDEZoo。

    

    偏微分方程（PDE）求解器是现代计算机图形学、几何处理和科学计算的基础。经典的蒙特卡洛PDE求解器（如游走球面法WoS）能够提供无偏的无网格估计，但其精度受限于采样数量和数值近似。学习型求解器速度很快，但存在偏差，且在分布偏移下表现脆弱。为此，我们提出了MC$^2$——一种神经网络与蒙特卡洛混合的PDE求解器，它将低预算的蒙特卡洛解视为真实场的结构化估计器，并学习一个单次前向的神经校正来恢复高保真度的解。MC$^2$能够匹配使用超过1000倍蒙特卡洛样本所得解的精度，优于所有被评测的无网格经典方法、去噪方法和神经算子基线。为了支持有限计算资源下PDE求解的可复现研究，我们还发布了PDEZoo——目前最大的标准化椭圆偏微分方程基准数据集。

    arXiv:2605.09288v2 Announce Type: replace-cross  Abstract: Partial differential equation (PDE) solvers underpin modern computer graphics, geometry processing, and scientific computing. Classical Monte Carlo PDE solvers such as Walk-on-Spheres (WoS) provide mesh-free estimations that are unbiased, but with accuracy limited by sampling and numerical approximation. Learned solvers are fast but biased and brittle under distribution shift. We therefore present MC$^2$, a hybrid Neural Network-Monte Carlo PDE solver that treats a low-budget Monte Carlo solution as a structured estimator of the true field and learns a single-pass neural correction to recover a high-fidelity solution. MC$^2$ matches the accuracy of solutions using over $1000\times$ more Monte Carlo samples, outperforming all evaluated mesh-free classical, denoising, and neural-operator baselines. To enable reproducible study of finite-compute PDE solving, we additionally release PDEZoo, the largest standardized elliptic PDE ben
    
[^330]: 每参数Token覆盖度对稳健的LLM缩放定律外推至关重要

    Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation

    [https://arxiv.org/abs/2605.08541](https://arxiv.org/abs/2605.08541)

    本文证明在固定每参数Token比例下拟合LLM缩放定律会因N与D指数近似相等导致的共线性而使最小二乘问题病态化、尺度系数不可辨识，并推导出实现稳健外推所需的TPP多样性闭式阈值。

    

    神经缩放定律将语言模型的损失近似为参数量 $N$ 和Token数 $D$ 的幂律函数。遵循Chinchilla式的计算最优训练范式，许多研究在固定的每参数Token比例（TPP）$k$ 下进行实验来拟合缩放定律，并令 $D = kN$。我们证明，这种共线设计，结合经验上常见的 $N$ 与 $D$ 指数近似相等的现象，会在高斯-牛顿最小二乘问题中引入固有的病态性：设计矩阵的条件数随 $N$ 与 $D$ 指数之间差距的平方倒数增长。尺度系数实际上变得不可辨识，其置信区间膨胀一个数量级甚至更多，产生一个“松散”的模型，其外推能力在偏离训练射线时急剧退化。我们针对四种缩放定律形式证明了这一点，并推导出一个封闭形式的TPP多样性阈值，该阈值是必要且充分的……（摘要原文在此处被截断）

    arXiv:2605.08541v3 Announce Type: replace  Abstract: Neural scaling laws approximate a language model's loss as a power-law function of parameter count $N$ and token count $D$. Following Chinchilla-style compute-optimal training, many studies fit scaling laws from runs performed under a fixed tokens-per-parameter (TPP) ratio $k$ and set $D = kN$. We show that this collinear design, combined with the empirically common near-equality of the exponents governing $N$ and $D$, induces an inherent ill-conditioning in the Gauss-Newton least-squares problem: the condition number of the design grows as the inverse square of the gap between the $N$ and $D$-exponents. The scale coefficients become practically unidentifiable, with confidence intervals inflating by an order of magnitude or more, yielding a ``sloppy'' model whose extrapolations degrade sharply off the training ray. We prove this for four scaling-law formalisms and derive a closed-form TPP-diversity threshold that is necessary and suf
    
[^331]: 重新思考黑盒在线策略蒸馏：基于评分标准的分析

    Rethinking Black-Box On-Policy Distillation: A Rubric-based Analysis

    [https://arxiv.org/abs/2605.07396](https://arxiv.org/abs/2605.07396)

    该论文提出基于评分标准的在线策略蒸馏（ROPD），通过对比教师回答与学生采样结果自动生成评分标准作为纯文本反馈，使黑盒教师模型也能进行在线策略蒸馏，并以少9.6倍的训练样本在竞赛数学基准上超越了基于logits的白盒OPD方法。

    

    在线策略蒸馏（OPD）是大语言模型后训练的一种强大范式，但其对教师模型logits的依赖将其限制在白盒教师模型上。在本文中，我们研究了基于评分标准的监督能否为黑盒OPD提供纯文本的反馈接口，以及是什么使其有效。我们的具体实现——基于评分标准的在线策略蒸馏（ROPD）——通过对比教师模型的回答与学生模型当前的采样结果来生成评分标准，并使用这些标准对采样结果进行评分以进行在线策略优化。我们发现，在匹配的设置下，纯文本的ROPD在四个竞赛数学基准上优于基于logits的OPD方法，仅用1/9.6的训练样本就达到了标准基于logits的OPD所能取得的最佳AIME24准确率。我们的分析得出三个发现：评分标准奖励比教师对数似然能更好地追踪答案正确性（AUC 0.90 对比 0.35）；使用评分标准进行训练能修复更多未通过的评判标准，同时减少已通过标准的退化……

    arXiv:2605.07396v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) is a powerful paradigm for LLM post-training, yet its reliance on teacher logits restricts it to white-box teachers. In this paper, we study whether rubric-based supervision can provide a text-only feedback interface for black-box OPD, and what makes it effective. Our instantiation, Rubric-based On-Policy Distillation (ROPD), induces rubrics by contrasting teacher responses with the student's current rollouts and uses them to score these rollouts for on-policy optimization. We find that, under matched settings, text-only ROPD outperforms logit-based OPD methods on four competition-math benchmarks, reaching the best AIME24 accuracy of standard logit-based OPD with 9.6x fewer training samples. Our analysis yields three findings: rubric rewards track correctness far better than teacher log-likelihood (AUC 0.90 vs. 0.35); training with them repairs more failed criteria and regresses fewer passed ones th
    
[^332]: 用于图表示学习的多样性曲线

    Diversity Curves for Graph Representation Learning

    [https://arxiv.org/abs/2605.06466](https://arxiv.org/abs/2605.06466)

    该论文提出通过在图粗化层级上跟踪结构多样性，利用新颖的等距不变量“图散度”构建可解释、高效且可跨不同尺寸图直接比较的图嵌入“多样性曲线”，并通过边收缩粗化提升了图级表示的表达能力。

    

    图级表征是刻画图之间结构差异的重要工具。然而，即使是从同一底层分布中采样的图，比较具有不同基数的图仍然充满挑战。尤其是无监督任务，需要可解释、可扩展且可靠的尺寸感知图表示。我们的工作通过在粗化层级上跟踪图的结构多样性来解决这些问题。由此得到的图嵌入——我们称之为多样性曲线——在构建上具有可解释性、高效性，并且可以在粗化层次结构之间直接比较。具体而言，我们跟踪图的散度，这是一种新颖的等距不变量，天然适合编码图的度量多样性与几何特性。我们利用边收缩粗化方法，并证明了这种方法能提升表达能力，从而产生比结构描述符更强大的图级表示。

    arXiv:2605.06466v2 Announce Type: replace  Abstract: Graph-level representations are crucial tools for characterising structural differences between graphs. However, comparing graphs with different cardinalities, even when sampled from the same underlying distribution, remains challenging. Unsupervised tasks in particular require interpretable, scalable, and reliable size-aware graph representations. Our work addresses these issues by tracking the structural diversity of a graph across coarsening levels. The resulting graph embeddings, which we denote diversity curves, are interpretable by construction, efficient, and directly comparable across coarsening hierarchies. Specifically, we track the spread of graphs, a novel isometry invariant that is inherently well-suited for encoding the metric diversity and geometry of graphs. We utilise edge contraction coarsening and prove that this improves expressivity, thus leading to more powerful graph-level representations than structural descri
    
[^333]: 基于核方法的希尔伯特空间中的高斯混合模型

    Gaussian mixture models in Hilbert spaces via kernel methods

    [https://arxiv.org/abs/2605.05996](https://arxiv.org/abs/2605.05996)

    该论文提出了一种基于核均值嵌入的希尔伯特空间数据高斯混合模型框架，用于对动态函数数据等无限维随机对象进行聚类分析，并从理论和实验上证明了其良定义性、稠密近似能力以及在多种数据几何形态上的有效性。

    

    现代许多学科的数据集日益由随时间演化、潜在无限维的随机对象组成，例如动态函数数据，这些数据自然地在希尔伯特空间中进行建模。在这些场景下，对概率测度进行刻画（例如通过密度函数）可能是病态定义的或在技术上具有挑战性。受聚类应用的启发，我们提出了一种基于核均值嵌入的希尔伯特空间取值数据的高斯混合框架，并开发了高效的优化算法用于估计。我们建立了理论保证，证明所提出的算法是良定义的，且该模型在无限维空间中产生稠密的近似类。我们通过对多种数据几何形态的广泛实验评估了该框架，包括 $L^2$ 函数数据以及现代医疗应用中出现的图信号。

    arXiv:2605.05996v2 Announce Type: replace-cross  Abstract: Modern datasets across many disciplines increasingly consist of time-evolving, potentially infinite-dimensional random objects, such as dynamic functional data, which are naturally modeled in Hilbert spaces. In these settings, characterizing probability measures, for example, through densities, can be ill-defined or technically challenging. Motivated by clustering applications, we propose a Gaussian mixture framework for Hilbert-space-valued data based on kernel mean embeddings and develop efficient optimization algorithms for estimation. We establish theoretical guarantees showing that the proposed algorithm is well-defined and that the model yields a dense class of approximations in infinite-dimensional spaces. We evaluate the framework through extensive experiments on diverse data geometries, including $L^2$-functional data and graph signals arising in modern medical applications.
    
[^334]: 利用凸松弛进行数据去噪

    Denoising data using convex relaxations

    [https://arxiv.org/abs/2605.02327](https://arxiv.org/abs/2605.02327)

    该论文提出一种基于凸松弛的去噪估计器，先通过主成分分析降维，再将观测投影到潜在流形的凸包上，并在有限样本下证明了去噪误差保证，可应用于低温电子显微镜等场景。

    

    我们研究对观测值 \(Y_i=X_i+Z_i\) 进行去噪的问题，其中潜在变量 \(X_i\) 采样自 \(\mathbb{R}^n\) 中的一个低维流形，噪声变量 \(Z_i\) 为各向同性高斯噪声。我们提出了一种凸松弛估计器，它首先通过主成分分析降低维度，然后将观测值投影到投影后潜在流形的凸包上。我们构造了一个统计预言机，利用噪声样本的经验高斯尾部概率来估计其支撑超平面。在潜在分布满足下界质量条件下，我们为该预言机证明了有限样本保证，并推导出所得去噪器的误差界。该分析结合了凸约束下最小二乘投影的风险界（由 Chatterjee 提出）与流形的熵界。我们还针对低温电子显微镜观测模型验证了该框架的假设……

    arXiv:2605.02327v3 Announce Type: replace-cross  Abstract: We study the problem of denoising observations \(Y_i=X_i+Z_i\), where the latent variables \(X_i\) are sampled from a low-dimensional manifold in \(\mathbb{R}^n\) and the noise variables \(Z_i\) are isotropic Gaussian. We propose a convex-relaxation estimator that first reduces dimension by principal component analysis and then projects the observations onto the convex hull of the projected latent manifold. We construct a statistical oracle that estimates its supporting hyperplanes from empirical Gaussian tail probabilities of the noisy sample. Under a lower-mass condition on the latent distribution, we prove finite-sample guarantees for the oracle and derive error bounds for the resulting denoiser. The analysis combines risk bounds for least-squares projection under convex constraints (due to Chatterjee) with entropy bounds for the manifold. We also verify the assumptions of the framework for a Cryo-Electron Microscopy observa
    
[^335]: COMO：基于最小风险训练的闭环光学分子识别

    COMO: Closed-Loop Optical Molecule Recognition with Minimum Risk Training

    [https://arxiv.org/abs/2604.23546](https://arxiv.org/abs/2604.23546)

    该论文提出COMO框架，首次将最小风险训练引入光学化学结构识别任务，通过在分子级进行闭环直接优化来缓解暴露偏差，从而更好地优化化学有效性和结构相似性等分子级评估指标。

    

    光学化学结构识别（OCSR）旨在将分子图像转换为SMILES字符串或分子图等机器可读的表示形式，但由于化学结构变化多样、存在简写约定以及扫描伪影等因素，其在真实世界文档中的应用仍然极具挑战性。大多数现有的深度学习方法依赖于采用token级最大似然估计（MLE）的教师强制训练范式。这种训练范式存在暴露偏差问题，因为模型在训练时使用的是真实值前缀，而在推理时却必须以自身先前的预测结果为条件。此外，token级的MLE目标函数难以针对化学有效性和结构相似性等分子级评估指标进行优化。在此，我们将最小风险训练引入OCSR领域，并提出了COMO（闭环光学分子识别）框架，该框架通过直接在分子级上进行优化来缓解暴露偏差……（原文摘要在此处截断）

    arXiv:2604.23546v2 Announce Type: replace-cross  Abstract: Optical chemical structure recognition (OCSR) translates molecular images into machine-readable representations such as SMILES strings or molecular graphs, but remains challenging in real-world documents due to wide variations in chemical structures, shorthand conventions, and scanning artifacts. Most existing deep learning approaches rely on teacher forcing with token-level Maximum Likelihood Estimation (MLE). This training paradigm suffers from exposure bias, as models are trained under ground-truth prefixes but must condition on their own previous predictions during inference. Moreover, token-level MLE objectives hinder the optimization towards molecular-level evaluation criteria such as chemical validity and structural similarity. Here we introduce Minimum Risk Training (MRT) to OCSR and propose COMO (Closed-loop Optical Molecule recOgnition), a framework that mitigates exposure bias by directly optimizing over molecule-lev
    
[^336]: GSQ：通过Gumbel-Softmax采样实现大语言模型的高精度低比特标量量化

    GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling

    [https://arxiv.org/abs/2604.18556](https://arxiv.org/abs/2604.18556)

    提出了GSQ这一训练后标量量化方法，利用Gumbel-Softmax松弛联合学习逐坐标网格分配与逐组缩放因子，使简单的标量量化在低比特精度下就能达到与复杂的第二代向量量化方法相当的高准确率。

    

    量化已成为高效部署大语言模型（LLM）的标准工具，尤其在本地推理场景中，模型通常以每参数2-3比特的精度进行服务。目前最先进的技术分为两大阵营：一类是简单的标量量化技术（如GPTQ或AWQ），它们被广泛部署但精度在每参数3-4比特时趋于瓶颈；另一类是“第二代”向量或网格量化方法（如QTIP、GPTVQ和AQLM），它们虽能推动精度上限，但在实现和扩展方面非常困难。本文探究这一差距是否是根本性的，或者一个经过精心优化的标量量化器能否挽回其中大部分差距。我们给出了肯定的答案，提出了GSQ（Gumbel-Softmax量化），这是一种训练后标量量化方法，它利用离散网格的Gumbel-Softmax松弛来联合学习逐坐标的网格分配和逐组的缩放因子。GSQ的性能可以匹敌……

    arXiv:2604.18556v3 Announce Type: replace  Abstract: Quantization has become a standard tool for efficient LLM deployment, especially for local inference, where models are now routinely served at 2-3 bits per parameter. The state of the art is currently split into simple scalar quantization techniques, such as GPTQ or AWQ, which are widely deployed but plateau in accuracy at 3-4 bits per parameter (bpp), and "second-generation" vector- or trellis-quantized methods, such as QTIP, GPTVQ and AQLM, which push the accuracy frontier but are notoriously hard to implement and to scale. In this paper, we ask whether this gap is fundamental, or whether a carefully optimized $\textit{scalar}$ quantizer can recover most of it. We answer in the affirmative, by introducing GSQ (Gumbel-Softmax Quantization), a post-training scalar quantization method which jointly learns the per-coordinate grid assignments and the per-group scales using a Gumbel-Softmax relaxation of the discrete grid. GSQ matches th
    
[^337]: AI赋能分析中的人类监督分配

    Allocating Human Oversight in AI-Enabled Analytics

    [https://arxiv.org/abs/2604.12497](https://arxiv.org/abs/2604.12497)

    该论文提出了一种在任务难度未知情形下分配共享人类验证预算的上置信界策略，能够从人类标签与AI预测的配对数据中联合学习预测校正系数和任务难度，并具有分配遗憾理论保证，在实验中将性能提升至接近最优方案的水平。

    

    人工智能为许多分析任务提供了低成本的预测，但其可靠性以及额外获取一个人类标签的价值因任务而异。我们研究了在任务级不确定性未知的情况下，如何分配共享的人类验证预算，其中每个获取的标签同时服务于估计和后续的分配决策。对于难度已知的情形，基准方法遵循一种成本调整后的Neyman规则，该规则同时考虑任务重要性和标签成本。在我们的“调优纠偏-难度”框架内，我们开发了一种上置信界策略，从配对的人类标签和预测中联合学习调优预测驱动推断的校正系数以及由此产生的任务难度，并为置信半径策略建立了分配遗憾保证。在合成实验和一项包含68个任务、超过2000名受访者的真实数字孪生调查中，我们的方法弥补了与最优方案之间的大部分差距。

    arXiv:2604.12497v3 Announce Type: replace  Abstract: Artificial intelligence provides low-cost predictions for many analytics tasks, but their reliability and the value of another human label vary across tasks. We study how to allocate a shared human-validation budget when task-level uncertainty is unknown and each acquired label informs both estimation and subsequent allocation. For known difficulty levels, the benchmark follows a cost-adjusted Neyman rule that accounts for task importance and label cost. Within our Tuned Rectification-Difficulty framework, we develop an upper-confidence-bound policy that jointly learns the correction coefficients of tuned prediction-powered inference and the resulting difficulties from paired human labels and predictions, and establish an allocation-regret guarantee for the confidence-radius policy. In synthetic experiments and a real digital-twin survey with 68 tasks and over 2,000 respondents, our method closes most of the gap to the oracle on the 
    
[^338]: 差分隐私下的分布性质验证

    Differentially Private Verification of Distribution Properties

    [https://arxiv.org/abs/2604.10819](https://arxiv.org/abs/2604.10819)

    本文首次研究了差分隐私下的分布性质验证，揭示了在隐私保护约束下，交互式证明协议的性能提升取决于隐私参数和模型，并利用了隐私与可复制性及隐私放大之间的联系。

    

    近期由Chiesa和Gur发起，并由Herman和Rothblum进一步发展的研究线路，探讨了在强大、知识渊博但不可信的证明者协助下，验证分布性质的样本和通信复杂性。在本工作中，我们开创性地研究了差分隐私下的分布性质验证。毕竟，如果我们不信任证明者来协助验证，又为何要信任它处理我们的敏感样本呢？我们绘制了差分隐私分布性质验证的景观。在非隐私情况下，已知单轮私有硬币协议可以比公共硬币（AM）协议具有显著更低的复杂性。相反，在差分隐私交互式证明中改进的可能性取决于隐私参数范围和模型。我们借鉴了隐私与可复制性以及隐私放大之间的联系。

    arXiv:2604.10819v2 Announce Type: replace-cross  Abstract: A recent line of work initiated by Chiesa and Gur and further developed by Herman and Rothblum investigates the sample and communication complexity of verifying properties of distributions with the assistance of a powerful, knowledgeable, but untrusted prover. In this work, we initiate the study of differentially private distribution property verification. After all, if we do not trust the prover to help us with verification, why should we trust it with our sensitive sample? We map a landscape of differentially private verification of properties of distributions. In the non-private case it is known that one-round private-coin protocols can have substantially lower complexity than public-coin (AM) protocols. In contrast, the possibility for improvement in differentially private interactive proofs depends on the privacy parameter regime and model. Drawing on connections between privacy and replicability and privacy amplification 
    
[^339]: 无需专家示范的安全近似非线性模型预测控制

    Safe Approximate Nonlinear Model Predictive Control Without Expert Demonstrations

    [https://arxiv.org/abs/2603.24503](https://arxiv.org/abs/2603.24503)

    提出无需专家示范数据的Seq-AMPC无标签循环框架，通过安全滤波器与接受容差设计，在保证递归可行性与收敛性的同时实现安全的非线性模型预测控制神经近似。

    

    在线重复优化使得非线性模型预测控制（NMPC）在实时应用中计算开销巨大。神经网络近似虽能降低该成本，但通常需要大量专家NMPC解构成的数据集。我们提出序列近似MPC（Seq-AMPC），这是一种无需标签的循环框架，利用在整个预测时域内共享的参数，沿着预测轨迹生成完整的控制序列。该策略直接基于已知动力学进行训练，其损失函数涵盖控制代价、约束满足，以及与可行备份安全滤波器相一致的成本下降条件。在该安全滤波器的基础上，我们进一步引入了一种依赖于阶段成本的接受容差，在标称MPC假设下保持递归可行性、约束满足和渐近收敛性，且不受神经网络近似误差的影响。此外，我们还推导了基于Hoeffding不等式的有限样本界，用于期望的（摘要在此处被截断）

    arXiv:2603.24503v2 Announce Type: replace  Abstract: Repeated online optimization makes nonlinear model predictive control (NMPC) computationally demanding for real-time applications. Neural approximations reduce this cost but often require large datasets of expert NMPC solutions. We introduce Sequential Approximate MPC (Seq-AMPC), a label-free recurrent framework that generates complete control sequences along predicted trajectories using parameters shared across the prediction horizon. The policy is trained directly through known dynamics with losses for control cost, constraint satisfaction, and a cost-decrease condition aligned with a feasible-backup safety filter. Building on this filter, we introduce a stage-cost-dependent acceptance tolerance that preserves recursive feasibility, constraint satisfaction, and asymptotic convergence under nominal MPC assumptions, independently of neural approximation error. We further derive Hoeffding-based finite-sample bounds for the expected fr
    
[^340]: 图能量匹配：面向图生成的传输对齐能量建模方法

    Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation

    [https://arxiv.org/abs/2603.23398](https://arxiv.org/abs/2603.23398)

    提出图能量匹配（GEM）框架，通过对齐输运机制来克服离散能量模型的采样陷阱与训练不稳定问题，实现可融入结构或功能约束、可组合的图生成，并缩小与离散扩散模型的保真度差距。

    

    arXiv:2603.23398v4 公告类型：replace-cross。摘要：离散数据（如图）的生成建模是众多科学与工业应用的基础，包括分子发现与材料设计。在这些领域中，概率推断尤为宝贵，因为它支持可组合的生成，并能有原则性地融入期望的约束，如结构或功能性质。基于能量的模型通过捕捉相对似然，并允许在推断时直接施加约束，天然契合这一目标。然而，离散能量模型通常难以实现高效且高质量的采样：支持集之外的区域往往存在虚假的局部极小值，会使采样器陷入其中并导致训练不稳定，从而使其与离散扩散模型相比存在保真度差距。为弥合这一差距，我们提出了图能量匹配，一种受 Jordan-Kinderlehrer-Otto 输运机制启发的离散生成框架。（注：原文摘要在此处被截断）

    arXiv:2603.23398v4 Announce Type: replace-cross  Abstract: Generative modeling of discrete data, such as graphs, underpins many scientific and industrial applications, including molecular discovery and materials design. In these domains, probabilistic inference is particularly valuable, as it enables composable generation and principled incorporation of desired constraints, such as structural or functional properties. Energy-based models naturally support this goal by capturing relative likelihoods and enabling composable inference by directly enforcing constraints during inference. However, discrete energy-based models typically struggle with efficient and high-quality sampling, as off-support regions often contain spurious local minima, trapping samplers and causing training instabilities, resulting in a fidelity gap compared to discrete diffusion models. To address this gap, we introduce Graph Energy Matching (GEM), a discrete generative framework inspired by the Jordan-Kinderlehrer
    
[^341]: 二值神经网络的联邦学习：实现低成本推理

    Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference

    [https://arxiv.org/abs/2603.15507](https://arxiv.org/abs/2603.15507)

    提出FedBNN框架，首次将旋转感知二值化引入联邦学习，在客户端数据分布异构的条件下训练二值神经网络，实现边缘设备上的低成本高效推理。

    

    联邦学习（FL）使分布式客户端能够在不共享原始数据的情况下共同训练一个全局模型。然而，由于深度神经网络在推理时的计算和内存需求，将其部署在资源受限的边缘设备上仍然具有挑战性。二值神经网络（BNN）通过使用单个比特表示权重和激活值，可以带来显著的效率提升，但在如此激进的量化下实现有竞争力的准确率仍然十分困难。旋转感知二值化通过将实值权重与其二进制表示对齐来减少量化误差。尽管旋转感知二值化在集中式学习中非常有效，但由于本地模型与全局模型之间的差异，特别是在客户端数据分布异构的情况下，将其应用于联邦学习并非易事。为了应对这一挑战，我们提出了FedBNN，一个联邦旋转感知二值化框架。

    arXiv:2603.15507v2 Announce Type: replace  Abstract: Federated Learning (FL) enables distributed clients to jointly train a global model without sharing their raw data. However, deploying the resulting deep neural networks on resource-constrained edge devices remains challenging due to their inference-time computational and memory requirements. Binary Neural Networks (BNNs) offer substantial efficiency gains by representing weights and activations using a single bit, but achieving competitive accuracy under such aggressive quantization remains difficult. Rotation-aware binarization reduces quantization errors by aligning real-valued weights with their binary representations. Despite its effectiveness in centralized learning, applying rotation-aware binarization in FL is non-trivial due to discrepancies between local and global models, particularly under heterogeneous client data distributions. To address this challenge, we propose FedBNN, a federated rotation-aware binarization framewo
    
[^342]: 对决老虎机中孔多塞胜者识别的采样复杂度

    The Sampling Complexity of Condorcet Winner Identification in Dueling Bandits

    [https://arxiv.org/abs/2603.15189](https://arxiv.org/abs/2603.15189)

    提出一种利用完整间隔矩阵的孔多塞胜者识别算法，其样本复杂度在所有情形下达到最优（上下界匹配），并揭示了采样复杂度由定位有信息比较的成本所刻画的权衡结构。

    

    我们在大规模随机对决老虎机中研究最优臂识别问题，仅假设存在孔多塞胜者，即以至少 1/2 的概率赢得每次含噪两两比较的臂。我们提出了一种新的识别方法，该方法利用完整的间隔矩阵 $\Delta_{i,j}=q_{i,j}-\tfrac{1}{2}$（其中 $q_{i,j}$ 是臂 $i$ 击败臂 $j$ 的概率），而不仅仅利用孔多塞胜者与其他臂之间的间隔。我们推导出了高概率、依赖于实例的样本复杂度保证，通过利用涉及胜者之外的有信息比较，在（对数因子上）改进了已知最好的结果。我们用匹配的下界补充了这些结果，证明了我们的方法在所有情形下的最优性。总体而言，我们的结果揭示了采样复杂度的一般形式，其特征在于定位有信息比较的成本与其他代价之间的权衡。

    arXiv:2603.15189v2 Announce Type: replace-cross  Abstract: We study best-arm identification in large-scale stochastic dueling bandits under the sole assumption that a Condorcet winner exists, i.e., an arm that wins each noisy pairwise comparison with probability at least $1/2$. We introduce a new identification procedure that exploits the full gap matrix $\Delta_{i,j}=q_{i,j}-\tfrac12$ (where $q_{i,j}$ is the probability that arm $i$ beats arm $j$), rather than only the gaps between the Condorcet winner and the other arms. We derive high-probability, instance-dependent sample-complexity guarantees that (up to logarithmic factors) improve the best known ones by leveraging informative comparisons beyond those involving the winner. We complement these results with matching lower bounds that establish the optimality of our procedures in all regimes. Overall, our results reveal the general form of the sampling complexity, characterized by a trade-off between the cost of locating informative
    
[^343]: 展示而非告知：通过观看人类视频检测新物体

    Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

    [https://arxiv.org/abs/2603.12751](https://arxiv.org/abs/2603.12751)

    提出了一种自监督的“展示而非告知”方法，利用人类演示视频本身自动构建数据集并训练定制物体检测器，无需语言描述和提示工程即可快速检测新物体。

    

    机器人如何快速识别在人类演示中展示给它的新物体？现有的封闭集物体检测器经常无法做到这一点，因为这些物体超出了分布范围。虽然开放集检测器（例如视觉语言模型）有时能够成功，但它们通常需要昂贵且繁琐的人机回环提示工程来唯一地识别新物体实例。在本文中，我们提出了一个自监督系统，通过在自动创建的数据集上训练定制的物体检测器，消除了对繁琐语言描述和昂贵提示工程的需求，该数据集由人类演示本身进行监督。在我们的“展示而非告知”方法中，我们在演示期间向检测器展示感兴趣的特定物体，而不是通过复杂的语言描述来告知检测器这些物体。通过完全绕过语言，这一范式使我们能够快速训练定制的

    arXiv:2603.12751v2 Announce Type: replace-cross  Abstract: How can a robot quickly identify and recognize new objects shown to it during a human demonstration? Existing closed-set object detectors frequently fail at this because the objects are out-of-distribution. While open-set detectors (e.g., VLMs) sometimes succeed, they often require expensive and tedious human-in-the-loop prompt engineering to uniquely recognize novel object instances. In this paper, we present a self-supervised system that eliminates the need for tedious language descriptions and expensive prompt engineering by training a bespoke object detector on an automatically created dataset, supervised by the human demonstration itself. In our approach, "Show, Don't Tell," we show the detector the specific objects of interest during the demonstration, rather than telling the detector about these objects via complex language descriptions. By bypassing language altogether, this paradigm enables us to quickly train bespoke 
    
[^344]: RECAP：局部赫布原型学习作为储备池动力学的自组织读出层

    RECAP: Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics

    [https://arxiv.org/abs/2603.06639](https://arxiv.org/abs/2603.06639)

    提出了一种名为RECAP的仿生学习策略，将未经训练的储备池动力学与自组织赫布原型读出相结合，仅通过局部可塑性规则即可实现无需反向传播的稳健图像分类。

    

    大脑中稳健的感知通常归因于高维的群体活动，以及能够强化重复出现结构的局部可塑性机制。相比之下，大多数现代图像识别系统通过误差反向传播和端到端梯度优化进行训练，这些方法与局部计算和局部可塑性并不天然契合。我们提出了RECAP（基于赫布共激活原型的储备池计算），这是一种用于稳健图像分类的仿生学习策略，它将未经训练的储备池动力学与自组织的赫布原型读出层相结合。RECAP将时间平均的储备池响应离散化为激活水平，在储备池单元对之间构建共激活掩码，并通过类赫布的增强-衰减规则增量式地更新类别原型矩阵。推理通过基于重叠度的原型匹配来执行。该方法避免了误差反向传播

    arXiv:2603.06639v2 Announce Type: replace-cross  Abstract: Robust perception in brains is often attributed to high-dimensional population activity together with local plasticity mechanisms that reinforce recurring structure. In contrast, most modern image recognition systems are trained by error backpropagation and end-to-end gradient optimization, which are not naturally aligned with local computation and local plasticity. We introduce RECAP (Reservoir Computing with Hebbian Co-Activation Prototypes), a bio-inspired learning strategy for robust image classification that couples untrained reservoir dynamics with a self-organizing Hebbian prototype readout. RECAP discretizes time-averaged reservoir responses into activation levels, constructs a co-activation mask over reservoir unit pairs, and incrementally updates class-wise prototype matrices via a Hebbian-like potentiation-decay rule. Inference is performed by overlap-based prototype matching. The method avoids error backpropagation 
    
[^345]: 预条件流匹配

    Preconditioned Flow Matching

    [https://arxiv.org/abs/2603.02337](https://arxiv.org/abs/2603.02337)

    该论文揭示了流匹配训练中由中间分布协方差病态条件导致的几何优化瓶颈——高方差方向拟合快而低方差方向进展慢，并据此提出“先预条件、再匹配”的预条件流匹配框架，通过变换目标分布来加速和稳定训练。

    

    流匹配（FM）通过沿中间分布 $p_t$ 回归随机速度目标来学习向量场。我们在此回归问题中发现了一个几何优化瓶颈：当 $p_t$ 的协方差 $\Sigma_t$ 呈病态条件时，基于梯度的训练会快速拟合高方差方向，而沿低方差方向的进展却很缓慢。在一个可精确求解的高斯设定中，我们证明超额风险由 $\Sigma_t$ 加权，且梯度下降和随机梯度下降的收敛性都依赖于条件数。随后，我们将分析扩展到高斯混合分布，表明多峰性并不能平均掉这一效应；相反，最慢且条件最差的分量可能主导整个优化过程。受此分析启发，我们提出了预条件流匹配，这是一种“先预条件、再匹配”的框架，它将目标分布变换为更具……（原文摘要在此处被截断）

    arXiv:2603.02337v3 Announce Type: replace-cross  Abstract: Flow matching (FM) learns vector fields by regressing stochastic velocity targets along intermediate distributions $p_t$. We identify a geometric optimization bottleneck in this regression problem: when the covariance $\Sigma_t$ of $p_t$ is ill-conditioned, gradient-based training rapidly fits high-variance directions while making slow progress along low-variance ones. In an exactly solvable Gaussian setting, we prove that the excess risk is weighted by $\Sigma_t$, and that both gradient descent and stochastic gradient descent inherit condition-number-dependent convergence. We then extend the analysis to Gaussian mixtures, showing that multimodality does not average away this effect; instead, the slowest and worst-conditioned component can control optimization. Motivated by this analysis, we propose $\textit{preconditioned flow matching}$, a precondition-then-match framework that transforms the target distribution into a more i
    
[^346]: 基于Gossip算法的去中心化排名聚合：收敛性与鲁棒性

    Decentralized Ranking Aggregation via Gossip: Convergence and Robustness

    [https://arxiv.org/abs/2602.22847](https://arxiv.org/abs/2602.22847)

    本文针对偏好数据分布在通信网络中的去中心化场景，研究如何利用gossip算法实现排名聚合，并提供收敛性保证与对污染数据的鲁棒性。

    

    排名聚合的概念在偏好分析中扮演着核心角色。文献中已记录了大量用于计算中位数排名的算法，这些算法通常源于社会选择理论，并在集中式设置下（即所有待聚合的排名数据能够汇集到单一计算单元时）提供理论保证。对于许多技术（例如点对点网络、物联网、多智能体系统）而言，在去中心化环境下——即偏好数据最初分布在通信网络中时——扩展计算共识排名的能力，同时保证收敛性以及对潜在污染数据的鲁棒性，仍然是一个重大的方法论挑战。事实上，近年来关于去中心化计算的文献主要集中于使用gossip算法来计算或优化诸如算术平均值之类的统计量。本文的目的……（原文摘要在此处截断）

    arXiv:2602.22847v3 Announce Type: replace-cross  Abstract: The concept of ranking aggregation plays a central role in preference analysis, and numerous algorithms for calculating median rankings, often originating in social choice theory, have been documented in the literature, offering theoretical guarantees in a centralized setting, \textit{i.e.}, when all the ranking data to be aggregated can be brought together in a single computing unit. For many technologies (\textit{e.g.} peer-to-peer networks, IoT, multi-agent systems), extending the ability to calculate consensus rankings with guarantees of convergence and resilience to potential contamination in a decentralized setting, when preference data is initially distributed across a communicating network, remains a major methodological challenge. Indeed, in recent years, the literature on decentralized computation has mainly focused on computing or optimizing statistics such as arithmetic means using gossip algorithms. The purpose of 
    
[^347]: LoBoost：面向梯度提升树的模型原生局部保形预测

    LoBoost: Model-Native Local Conformal Prediction for Gradient-Boosted Trees

    [https://arxiv.org/abs/2602.22432](https://arxiv.org/abs/2602.22432)

    LoBoost通过重用梯度提升树集成模型自身的叶结构构建多尺度嵌套分组，实现模型原生的局部保形预测，无需辅助划分、重训练或额外数据分割即可局部估计残差分位数并适应异方差性。

    

    梯度提升决策树是表格回归中最强大的现成预测器之一，但仅靠点预测无法量化不确定性。保形预测可以提供无分布的边际覆盖保证，然而标准的分割保形方法只使用单一的全局残差分位数，难以适应异方差性。我们提出LoBoost，一种模型原生的局部保形方法，它重用已拟合集成模型的叶结构来定义特征空间的多尺度划分。每个输入由其沿提升路径所经过的叶节点序列来表示，匹配的叶前缀定义了嵌套分组，并在这些分组内局部估计残差分位数。通过重用模型已经学习到的预测结构，LoBoost无需辅助划分或干扰模型，无需重新训练，也无需在标准保形校准之外进行额外的数据分割。我们的理论将拟合的稳定性……（摘要在此处截断）

    arXiv:2602.22432v3 Announce Type: replace-cross  Abstract: Gradient-boosted decision trees are among the strongest off-the-shelf predictors for tabular regression, but point predictions alone do not quantify uncertainty. Conformal prediction provides distribution-free marginal coverage, yet standard split conformal uses a single global residual quantile and can adapt poorly to heteroscedasticity. We propose LoBoost, a model-native local conformal method that reuses the fitted ensemble's leaf structure to define a multiscale partition of the feature space. Each input is represented by the sequence of leaves it visits along the boosting path, and matching leaf prefixes define nested groups in which residual quantiles are estimated locally. By reusing the predictive structure already learned by the model, LoBoost requires no auxiliary partition or nuisance model, no retraining, and no additional data split beyond standard conformal calibration. Our theory connects the stability of the fit
    
[^348]: 从动力系统视角研究深度残差网络的相容泛化界

    On Compatible Generalization Bounds of Deep Residual Networks From a Dynamical System Perspective

    [https://arxiv.org/abs/2602.20921](https://arxiv.org/abs/2602.20921)

    本文从动力系统视角出发，通过精细的Rademacher复杂度收缩不等式与分量假设集等价性分析，为离散和连续时间深度残差网络建立了同时达到最优O(1/√S)样本率且与层数无关的相容一致泛化界。

    

    深度神经网络（DNNs）极大地推动了机器学习的发展，其中模型深度在其成功中扮演着核心角色。深度残差网络（ResNets）是最重要的深度神经网络类别之一。本文从动力系统的视角，研究了在深层极限下、具有广泛一类激活函数的深度残差网络的相容泛化行为。我们首先针对由这类激活函数构成的函数类，给出了一个精细化的 Rademacher 复杂度收缩不等式。随后，我们对 Rademacher 复杂度中的分量假设集进行了等价性分析，并对网络状态变量进行了前向收敛性分析，从而使我们能够建立相容的一致泛化界，该泛化界同时对离散时间和连续时间的 ResNets 都达到了最优的 O(1/√S) 样本复杂度。这些界还具有共同的层数无关性。

    arXiv:2602.20921v2 Announce Type: replace  Abstract: Deep neural networks (DNNs) have significantly advanced machine learning, with model depth playing a central role in their successes. Deep residual networks (ResNets) fall into one of the most important classes of DNNs. In this paper, we study the compatible generalization behavior of deep residual networks with a broad class of activation functions in the deep-layer limit, from a dynamical system perspective. We first show a refined contraction inequality for Rademacher complexity of function classes composed by that type of activation functions. We then conduct an equivalence analysis of component hypothesis sets in Rademacher complexity and forward convergence analysis of network state variables, which enable us to establish compatible uniform generalization bounds that simultaneously achieve the optimal \(O(1/\sqrt{S})\) sample rate for both discrete- and continuous-time ResNets. The bounds also share a common layer-number indepe
    
[^349]: 基于贝叶斯自适应提问的冷启动个性化

    Cold-Start Personalization via Bayesian Adaptive Questioning

    [https://arxiv.org/abs/2602.15012](https://arxiv.org/abs/2602.15012)

    提出CAPE方法，先离线从完整用户画像中学习偏好相关性的结构化世界模型，再在线通过免训练的贝叶斯推断选择信息量最大的问题并预测完整偏好画像，即使使用简单的线性回归信念模型也能大幅超越端到端强化学习。

    

    冷启动个性化需要在没有用户特定历史数据可用的情况下，仅通过极少量的交互来推断用户偏好。可能的偏好空间非常庞大，但用户只关心其中稀疏的子集，且很少会主动明确表达这些偏好；再加上有限的交互预算，这使得偏好引出（elicitation）变得极具挑战性。我们的关键洞察是：偏好在不同用户群体之间表现出可预测的结构；例如，希望获得详细解释的用户通常也重视完整的示例演算。我们提出了CAPE（冷启动自适应偏好引出，Cold-start Adaptive Preference Elicitation），该方法先离线地从完整用户画像中学习一个关于偏好相关性的结构化世界模型，然后在线执行无需训练的贝叶斯推断，以选择信息量最大的问题并预测完整的偏好画像，甚至包括从未询问过的维度。即使是简单的信念模型实例（如线性回归），其表现也能大幅超越端到端的强化学习方法。

    arXiv:2602.15012v2 Announce Type: replace-cross  Abstract: Cold-start personalization requires inferring preferences from minimal interaction when no user-specific historical data is available. The space of possible preferences is vast, yet users care about only a sparse subset and rarely articulate them upfront; combined with limited interaction budgets, this makes preference elicitation challenging. Our key insight is that preferences exhibit predictable structure across populations; e.g., users who want detailed explanations often also value worked examples. We propose CAPE (Cold-start Adaptive Preference Elicitation), which learns a structured world model of preference correlations offline from complete profiles, then performs training-free Bayesian inference online to select informative questions and predict complete preference profiles, including dimensions never asked about. Even simple belief model instantiations (e.g., linear regression) substantially outperform end-to-end RL.
    
[^350]: FUTON：用于隐式神经表示的傅里叶张量网络

    FUTON: Fourier Tensor Networks for Implicit Neural Representations

    [https://arxiv.org/abs/2602.13414](https://arxiv.org/abs/2602.13414)

    提出FUTON，一种将信号建模为广义傅里叶级数、并用低秩张量分解参数化傅里叶系数的隐式神经表示方法，它结合傅里叶与低秩两种归纳偏置，实现了精确的误差分解理论保证。

    

    隐式神经表示（INR）已成为编码信号的强大工具，然而主流的基于MLP的设计往往收敛缓慢，且其精度如何随规模变化仅停留在经验认识层面。我们提出了FUTON（傅里叶张量网络），它将信号建模为广义傅里叶级数，其系数由低秩张量分解进行参数化，并通过线性层或MLP头读出。FUTON结合了互补的归纳偏置：傅里叶基捕捉平滑性与周期性，而低秩参数化则强制实现低维谱结构。在万能逼近之外，我们证明当使用线性头时，最优FUTON的误差可精确分解为信号级数的截断误差与其系数的最优低秩误差之和，后者可在训练前由样本进行界定，并存在一个能达到该上界的FUTON。在图像拟合、占据体积等任务上，FUTON展现出了优越的性能。

    arXiv:2602.13414v2 Announce Type: replace-cross  Abstract: Implicit neural representations (INRs) have emerged as powerful tools for encoding signals, yet dominant MLP-based designs often suffer from slow convergence, and how their accuracy depends on their size is known only empirically. We introduce FUTON (Fourier Tensor Network), which models signals as generalized Fourier series whose coefficients are parameterized by a low-rank tensor decomposition, read out by a linear or MLP head. FUTON combines complementary inductive biases: Fourier bases capture smoothness and periodicity, while the low-rank parameterization enforces low-dimensional spectral structure. Beyond universal approximation, we show that with a linear head, the best FUTON's error splits exactly into the truncation of the signal's series and the best low-rank error of its coefficients, which we bracket from the samples before training, with a FUTON that attains the upper bound. On image fitting, occupancy volumes, and
    
[^351]: 在线线性优化是否足以实现策略鲁棒性？

    Is Online Linear Optimization Sufficient for Strategic Robustness?

    [https://arxiv.org/abs/2602.12253](https://arxiv.org/abs/2602.12253)

    本文证明次线性的线性化后悔值已足以保证竞价算法的策略鲁棒性，并通过简单的黑盒归约将任意在线线性优化算法转化为策略鲁棒的竞价算法。

    

    我们考虑重复贝叶斯首价拍卖中的竞价问题。实现最优后悔值的竞价算法已被广泛研究，但它们对卖方操纵的策略鲁棒性仍相对缺乏探索。基于无交换后悔算法构建的竞价算法可以同时实现这两个理想性质，但在统计和计算效率方面并非最优。相比之下，在线梯度上升是唯一同时实现 $O(\sqrt{TK})$ 后悔值和策略鲁棒性的算法，其中 $T$ 表示拍卖次数，$K$ 表示竞价次数。本文探究简单的在线线性优化（OLO）算法是否足以构造同时具备这两个理想性质的竞价算法。我们的主要结果表明，次线性的线性化后悔值足以保证策略鲁棒性。具体而言，我们构造了简单的黑盒归约，可将任意 OLO 算法转换为具有策略鲁棒性的竞价算法。

    arXiv:2602.12253v2 Announce Type: replace-cross  Abstract: We consider bidding in repeated Bayesian first-price auctions. Bidding algorithms that achieve optimal regret have been extensively studied, but their strategic robustness to the seller's manipulation remains relatively underexplored. Bidding algorithms based on no-swap-regret algorithms achieve both desirable properties, but are suboptimal in terms of statistical and computational efficiency. In contrast, online gradient ascent is the only algorithm that achieves $O(\sqrt{TK})$ regret and strategic robustness [KSS24], where $T$ denotes the number of auctions and $K$ the number of bids.   In this paper, we explore whether simple online linear optimization (OLO) algorithms suffice for bidding algorithms with both desirable properties. Our main result shows that sublinear linearized regret is sufficient for strategic robustness. Specifically, we construct simple black-box reductions that convert any OLO algorithm into a strategic
    
[^352]: 对齐图像与长文本描述中的“森林”与“树木”以实现视觉接地理解

    Aligning Forest and Trees in Images & Long Captions for Visually Grounded Understanding

    [https://arxiv.org/abs/2602.02977](https://arxiv.org/abs/2602.02977)

    提出CAFT模型，通过“由细到粗”的层次化学习同时实现局部文本-区域对齐与全局图像-文本对齐，使视觉-语言模型能够忠实理解长而细节丰富的图像描述。

    

    诸如CLIP等视觉-语言模型往往难以忠实理解长而细节丰富的描述文本，它们依赖主导性的场景线索而忽视细粒度的视觉证据。我们提出了一种层次化的视觉-语言学习原则，将场景理解为部分到整体的组合：在形成整体场景表示之前，模型应该揭示哪些语义部分出现在图像中的什么位置。为此，我们提出了CAFT（森林与树木的跨域对齐，Cross-domain Alignment of Forest and Trees），这是一种视觉-语言模型，它在中间表示层面联合学习局部文本-区域对齐，并在最终表示层面学习全局图像-文本对齐。利用长文本描述的组织结构（其中局部描述通常对应场景的各个部分），CAFT采用由细到粗的图像编码器和部分-整体的文本编码器，以发现局部化的部分语义，并逐步将其组合成全局的图像-文本表示。

    arXiv:2602.02977v3 Announce Type: replace-cross  Abstract: Vision-language models such as CLIP often struggle to faithfully understand long, detail-rich captions, relying on dominant scene cues while overlooking fine-grained visual evidence. We propose a hierarchical vision-language learning principle for understanding scenes as part-to-whole compositions: before forming a whole-scene representation, a model should uncover what semantic parts appear where in the image. To this end, we propose CAFT (Cross-domain Alignment of Forest and Trees), a vision-language model that jointly learns local text-region alignment at intermediate representations and global image-text alignment at the final representation. Exploiting the organization of long captions, where local descriptions often correspond to scene parts, CAFT employs a fine-to-coarse image encoder and a part-whole text encoder to discover localized part semantics and progressively compose them into a global image-text representation.
    
[^353]: FiLoRA：用于可控特征依赖的“聚焦与忽略”LoRA方法

    FiLoRA: Focus-and-Ignore LoRA for Controllable Feature Reliance

    [https://arxiv.org/abs/2602.02060](https://arxiv.org/abs/2602.02060)

    FiLoRA通过指令条件化的低秩门控机制，实现了在不改变任务目标的前提下，用自然语言指令直接控制多模态模型对特定特征路径的依赖，从而缓解捷径和虚假相关行为。

    

    多模态基础模型整合了跨模态的异构信号，但目前尚不清楚是否可以通过显式调节对不同内部特征路径的依赖来控制其预测。现有的处理捷径行为或虚假相关的方法主要依赖事后分析或数据层面的干预，难以直接干预模型对信息的使用方式。我们提出了FiLoRA（聚焦与忽略LoRA），一种指令条件化的、参数高效的适应框架，能够在保持任务和预测目标不变的情况下，实现对特征依赖的可控调节。FiLoRA将适应过程分解为特征对齐的低秩模块，并应用指令条件化门控，使自然语言指令能够作为计算层面的控制信号作用于内部表示。我们在受控分类设置及地理相关任务上评估了FiLoRA的性能。

    arXiv:2602.02060v2 Announce Type: replace-cross  Abstract: Multimodal foundation models integrate heterogeneous signals across modalities, yet it remains unclear whether their predictions can be controlled by explicitly modulating reliance on different internal feature pathways. Existing approaches to shortcut and spurious behavior primarily rely on post hoc analysis or data-level interventions, offering limited ability to directly intervene on how models use information. We introduce FiLoRA (Focus-and-Ignore LoRA), an instruction-conditioned, parameter-efficient adaptation framework that enables controllable modulation of feature reliance while keeping the task and predictive objective fixed. FiLoRA decomposes adaptation into feature-aligned low-rank modules and applies instruction-conditioned gating, allowing natural language instructions to act as computation-level control signals over internal representations. We evaluate FiLoRA across both controlled classification settings and ge
    
[^354]: 自我改进的训练与思考

    Self-Improving Training & Thinking

    [https://arxiv.org/abs/2601.21343](https://arxiv.org/abs/2601.21343)

    该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。

    

    传统上，大型语言模型采用分阶段训练：先在原始文本上进行预训练，再进行后训练以学习指令遵循和推理能力。然而，这种分离造成了根本性的局限：许多理想的行为，如安全性、事实准确性、整体生成质量和推理能力，只在训练后期才被加入，尽管早期学习的模式在很大程度上塑造了模型的能力。为解决这一问题，我们提出了一种新的预训练和中期训练方法，将这些行为更早地融入训练过程。我们利用一个现有的强大的经过后训练的模型来重写数据并评判策略模型的输出（rollouts），从而在训练中更早地引入思考和强化学习。我们进一步通过引入“自我改进的后思考”（self-improving post-thinking）的概念，在所有训练阶段提升推理能力：即在生成之后进行推理，以提高模型的推理和自我评判能力。在实验中，我们证明了这些方法……

    arXiv:2601.21343v4 Announce Type: replace-cross  Abstract: Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite data and to judge policy model rollouts, thus introducing thinking and reinforcement earlier in training. We further improve reasoning at all stages by introducing the concept of self-improving post-thinking: improving the model's reasoning and self-judging ability by reasoning after generation. In our experiments, we show these a
    
[^355]: 熵风险感知蒙特卡洛树搜索

    Entropic Risk-Aware Monte Carlo Tree Search

    [https://arxiv.org/abs/2601.17667](https://arxiv.org/abs/2601.17667)

    提出了一种可证明正确的熵风险感知蒙特卡洛树搜索算法，该算法在根节点处的经验熵风险度量收敛于最优值，并具有多项式遗憾集中性的理论保证，能够在风险中性与风险规避行为之间实现有效权衡。

    

    我们提出了一种可证明正确的蒙特卡洛树搜索（MCTS）算法，用于求解具有熵风险度量（ERM）目标的风险感知马尔可夫决策过程（MDPs）。我们对所提出的算法进行了非渐近分析，表明该算法：（i）在根节点处获得的经验ERM收敛于最优ERM的意义上是正确的；（ii）具有多项式遗憾集中性。我们的算法成功地将先前工作中提出的用于求解具有ERM目标的风险感知MDPs的动态规划公式，应用于基于置信上界的树搜索算法框架中。最后，我们提供了一组说明性实验来评估我们的风险感知MCTS方法，结果表明该方法能够在风险中性和风险规避行为之间成功地进行权衡。

    arXiv:2601.17667v3 Announce Type: replace  Abstract: We propose a provably correct Monte Carlo tree search (MCTS) algorithm for solving risk-aware Markov decision processes (MDPs) with entropic risk measure (ERM) objectives. We provide a non-asymptotic analysis of our proposed algorithm, showing that the algorithm: (i) is correct in the sense that the empirical ERM obtained at the root node converges to the optimal ERM; and (ii) enjoys polynomial regret concentration. Our algorithm successfully exploits the dynamic programming formulations for solving risk-aware MDPs with ERM objectives introduced by previous works in the context of an upper confidence bound-based tree search algorithm. Finally, we provide a set of illustrative experiments to evaluate our risk-aware MCTS method, showing it successfully trades off between risk-neutral and risk-averse behavior.
    
[^356]: 超越局部重建：面向二值大语言模型量化的保持几何结构的输出对齐

    Beyond local reconstruction: Geometry-preserving output alignment for binary LLM quantization

    [https://arxiv.org/abs/2512.21651](https://arxiv.org/abs/2512.21651)

    该论文揭示了1比特量化中朴素输出驱动方法失效的两个根本原因——跨层误差累积与各向异性失真，并提出了一种保持几何结构的输出对齐方法来改进二值LLM量化。

    

    大语言模型（LLM）训练后量化的最新进展表明，低比特量化方法能够保持原模型的大部分性能。尽管取得了这些成功，1比特量化仍然尤其具有挑战性。1比特量化中的一种常见策略是遵循权重驱动的准则，通过匹配全精度参数来确定二值权重。然而，这一目标与量化模型的目标并不直接对齐，量化模型的真正目标是使模型的输出行为在量化的影响下得以保持。一个自然的替代方案是采用输出驱动的准则，利用校准数据最小化模型输出之间的差异。令人惊讶的是，朴素的输出驱动方法在1比特情形下往往表现得甚至更差。在本文中，我们证明这种失败源于两个根本问题：跨层的误差累积，以及更关键的是，各向异性失真……（原文摘要到此截断）

    arXiv:2512.21651v4 Announce Type: replace  Abstract: Recent advances in post-training quantization for large language models (LLMs) have demonstrated that low-bit quantization methods can maintain most of the original model performance. Despite such successes, 1-bit quantization remains particularly challenging. A common strategy in 1-bit quantization is to determine binary weights by matching full-precision parameters, following a weight-driven criterion. However, this objective is not directly aligned with the quantized model's objective, which is to preserve the model's output behavior under the impact of quantization. A natural alternative is to adopt output-driven criteria that minimize discrepancies in model outputs using calibration data. Surprisingly, naive output-driven approaches often perform even worse in the 1-bit regime. In this paper, we show that this failure arises from two fundamental issues: error accumulation across layers and, more critically, anisotropic distortio
    
[^357]: 基于点击流数据的机器学习预测数字用户挫败感

    Machine Learning to Predict Digital Frustration from Clickstream Data

    [https://arxiv.org/abs/2512.20438](https://arxiv.org/abs/2512.20438)

    本研究利用真实电商网站的点击流数据，通过LSTM和XGBoost等机器学习模型有效预测用户数字挫败感，其中LSTM表现最佳，且仅凭前20至30次交互即可实现可靠预测。

    

    许多企业依赖其移动应用和网站，因此用户在这些渠道上尝试完成任务时的挫败感可能导致销售损失和投诉。在本研究中，我使用来自真实电子商务网站的点击流数据来预测会话是否存在挫败感。挫败感通过基于愤怒爆发、来回导航（U型折返）、购物车流失、搜索困难和长时间漫游会话的特定规则来定义，并将这些规则应用于540万条原始点击流事件（304,881个会话）。我从每个会话中构建表格特征并训练标准分类器模型，同时使用完整的事件序列训练了一个判别式LSTM分类器。XGBoost达到约90%的准确率和0.9579的ROC AUC，而LSTM表现最佳，准确率约91%，ROC AUC为0.9705。最后，研究表明仅使用前20到30次交互，LSTM已经能够可靠地预测挫败感。

    arXiv:2512.20438v2 Announce Type: replace  Abstract: Many businesses depend on their mobile apps and websites, so user frustration while trying to complete a task on these channels can cause lost sales and complaints. In this research, I use clickstream data from a real e-commerce site to predict whether a session is frustrated or not. Frustration is defined using certain rules based on rage bursts, back and forth navigation (U turns), cart churn, search struggle, and long wandering sessions, and applies these rules to 5.4 million raw clickstream events (304,881 sessions). From each session, I build tabular features and train standard classifier models. I also use the full event sequence to train a discriminative LSTM classifier. XGBoost reaches about 90% accuracy, ROC AUC of 0.9579, while the LSTM performs best with about 91% accuracy and a ROC AUC of 0.9705. Finally, the research shows that with only the first 20 to 30 interactions, the LSTM already predicts frustration reliably.
    
[^358]: DiffeoMorph：基于旋转不变谱形状匹配的分布式三维形态发生学习

    DiffeoMorph: Learning Distributed 3D Morphogenesis with Rotation-Invariant Spectral Shape Matching

    [https://arxiv.org/abs/2512.17129](https://arxiv.org/abs/2512.17129)

    提出 DiffeoMorph 框架，利用 SE(3) 等变图神经网络与基于三维 Zernike 多项式的旋转不变谱形状匹配损失，使分布式智能体仅通过局部交互即可协同变形为目标三维形状。

    

    生物系统能够通过共享同一更新规则、且无需中央控制的智能体的集体行为，形成复杂的三维结构。这种分布式控制如何产生精确的全局模式，不仅是发育生物学的核心问题，也是分布式机器人、可编程物质和多智能体学习领域的核心问题。在此，我们提出 DiffeoMorph——一个端到端可微分的框架，用于学习一种形态发生协议，引导一群智能体变形为目标三维形状。每个智能体基于自身的内部状态以及从其他智能体接收到的信号，使用 SE(3) 等变图神经网络来更新其位置和内部状态。为了训练该系统，我们引入了一种基于三维 Zernike 多项式的新型形状匹配损失，该损失将预测形状与目标形状作为连续的空间分布进行比较，而非作为离散点云……

    arXiv:2512.17129v3 Announce Type: replace  Abstract: Biological systems can form complex three-dimensional structures through the collective behavior of agents that share a common update rule and operate without central control. How such distributed control gives rise to precise global patterns remains a central question not only in developmental biology but also in distributed robotics, programmable matter, and multi-agent learning. Here, we introduce DiffeoMorph, an end-to-end differentiable framework for learning a morphogenesis protocol that guides a population of agents to morph into a target 3D shape. Each agent updates its position and internal state using an SE(3)-equivariant graph neural network, based on its own internal state and signals received from other agents. To train this system, we introduce a new shape-matching loss based on 3D Zernike polynomials, which compares the predicted and target shapes as continuous spatial distributions, not as discrete point clouds, and i
    
[^359]: 基于Kronecker乘积的MMSE滤波器自动正则化方法

    Automatic regularization for Kronecker product MMSE filters

    [https://arxiv.org/abs/2512.14932](https://arxiv.org/abs/2512.14932)

    本文提出一种基于近似留一交叉验证度量的方法，可自动确定基于Kronecker乘积表示的低秩MMSE滤波器的正则化参数，在系统辨识和天线波束成形任务中显著优于常用方法。

    

    在这项工作中，我们提出了一种基于Kronecker乘积表示的低秩MMSE滤波器正则化参数的自动寻找方法，该方法采用近似留一法（ALO）交叉验证度量。我们将该问题表述为一种约束形式，同时涵盖误差最小化和干扰抑制两方面，并探讨了正则化与秩选择之间的联系。所提出的方法在系统辨识和天线波束成形问题上得到了验证，与常用方法相比显示出显著的性能提升。

    arXiv:2512.14932v2 Announce Type: replace  Abstract: In this work, we propose a method to automatically find the regularization parameter for low-rank MMSE filters based on a Kronecker-product representation, using an approximate leave-one-out (ALO) cross-validation metric. We formulate the problem in a constrained form that covers both error minimization and interference suppression, and explore the link between regularization and rank selection. The proposed method is validated on system identification and antenna beamforming problems, showing significant gains over commonly used methods.
    
[^360]: 面向基于视觉系统的确定性世界模型的动作引导可达性分析

    Action-Guided Reachability for Deterministic World Models of Vision-Based Systems

    [https://arxiv.org/abs/2512.08991](https://arxiv.org/abs/2512.08991)

    提出无潜变量的确定性世界模型（DWM）来替代闭环中的相机，并结合动作引导的划分-验证方案，实现了对端到端视觉控制器闭环行为可靠且紧凑的可达性验证。

    

    端到端的基于视觉的控制器将原始相机图像直接映射为控制动作，然而验证其闭环行为仍然具有挑战性：不存在将物理状态映射为相机图像的解析模型，且通过高维图像传播状态集合会产生过于宽松的可达集。我们用两种技术来应对这两个挑战。首先，我们提出了确定性世界模型（DWM），这是一种无潜变量的解码器，仅从物理状态生成相机图像，并通过显著性加权的重建损失和控制一致性损失进行训练。DWM在闭环中替代了相机，因此无需对任何潜变量进行界定，即可计算出替代闭环的可靠且紧凑的可达管道。其次，我们提出了一种动作引导的划分-验证方案，在固定的计算预算内尽可能多地认证初始集合。从廉价仿真中学习得到的高斯过程对（原文在此处截断）……

    arXiv:2512.08991v4 Announce Type: replace-cross  Abstract: End-to-end vision-based controllers map raw camera images directly to control actions, yet verifying their closed-loop behavior remains challenging: no analytical model maps a physical state to a camera image, and propagating a set of states through high-dimensional images yields loose reachable sets. We address both challenges with two techniques. First, we propose a Deterministic World Model (DWM), a latent-free decoder that generates camera images only from the physical state, trained with a saliency-weighted reconstruction loss and a control-consistency loss. The DWM replaces the camera in the closed loop, so a sound and tight reachable tube of the surrogate closed loop can be computed without bounding any latent variables. Second, we propose an action-guided partition-and-verify scheme that certifies as many initial sets as possible within a fixed computation budget. A Gaussian process learned from cheap simulations orders
    
[^361]: 超级激活器机制：Transformer 将可靠的概念信号集中在尾部

    The SuperActivator Mechanism: Transformers Concentrate Reliable Concept Signals in the Tail

    [https://arxiv.org/abs/2512.05038](https://arxiv.org/abs/2512.05038)

    该论文发现并理论刻画了 Transformer 中的“超级激活器机制”：概念对齐的注意力会放大激活差距，将最可靠的概念证据集中到少数高激活 token 的尾部上，从而提升基于概念向量的模型可解释性的可靠性。

    

    概念向量旨在通过将模型学习到的内部表示与人类可理解的语义联系起来，从而增强模型的可解释性，但其实际效用常常受到嘈杂且不一致的激活的限制。在本工作中，我们发现了超级激活器机制：这是一种 Transformer 的动态特性，它会放大概念激活之间的差距，将最可靠的概念证据集中到一小部分高度激活的 token 中，我们将其称为超级激活器。我们从理论上对这一机制进行了刻画，证明了与概念对齐的注意力会产生两种相互竞争的效应：一种是尾部放大效应，即已经较高的激活之间的差距会不成比例地快速增长；另一种是均衡化效应，当注意力集中于极端 token 时，均衡化效应最终会占据主导地位。我们通过实证验证了所预测的尾部放大效应，发现在各模型中，概念内的激活分布一致地形成了分离良好的上尾部。

    arXiv:2512.05038v3 Announce Type: replace  Abstract: Concept vectors aim to enhance model interpretability by linking learned internal representations with human-understandable semantics, but their practical utility is often limited by noisy and inconsistent activations. In this work, we discover the SuperActivator Mechanism: a transformer dynamic that amplifies concept activation gaps, concentrating the most reliable concept evidence into a small set of highly activated tokens, which we call SuperActivators. We theoretically characterize this mechanism, proving that concept-aligned attention produces two competing effects: a tail amplification effect, where gaps among already-high activations grow disproportionately fast, and an equalization effect that eventually dominates as attention concentrates on extreme tokens. We validate the predicted tail amplification empirically, finding that in-concept activation distributions develop well-separated upper tails consistently across models,
    
[^362]: 生成式在线强化学习

    Generative Online Reinforcement Learning

    [https://arxiv.org/abs/2512.02581](https://arxiv.org/abs/2512.02581)

    GoRL通过“优化与生成分离”的结构性原则——在易处理的潜空间中进行策略优化，而由条件生成解码器负责动作合成——实现了在在线强化学习中从头训练既稳定又具强表达能力的生成式策略。

    

    强化学习（RL）面临着一个持久的矛盾：易于稳定优化的策略（例如高斯策略）往往过于简单，无法表示复杂控制所要求的多模态动作分布；相反，具有强表达能力的生成式策略——如扩散模型和流匹配——由于似然难以求解、且梯度需要穿过漫长的采样链进行传播，在在线强化学习中难以优化。我们用一个关键的结构性原则来解决这一矛盾：将优化与生成分离（解耦）。基于这一原则，我们提出了GoRL（生成式在线强化学习），这是一个与具体算法无关的框架，它通过将策略优化限制在易于处理的潜空间中，同时把动作合成委托给条件生成解码器，从而从头训练具有强表达能力的策略。借助双时间尺度的交替训练调度，并将解码器的改进锚定在固定先验上，GoRL能够在实现稳定优化的同时……（原文在此截断）

    arXiv:2512.02581v4 Announce Type: replace  Abstract: Reinforcement learning (RL) faces a persistent tension: policies that are stable to optimize (e.g., Gaussians) are often too simple to represent the multimodal action distributions required for complex control. Conversely, expressive generative policies---such as diffusion and flow matching---can be difficult to optimize in online RL due to intractable likelihoods and gradients propagating through long sampling chains. We address this tension with a key structural principle: decoupling optimization from generation. Building on this, we introduce GoRL (Generative Online Reinforcement Learning), an algorithm-agnostic framework that trains expressive policies from scratch by confining policy optimization to a tractable latent space while delegating action synthesis to a conditional generative decoder. Using a two-timescale alternating schedule and anchoring decoder refinement to a fixed prior, GoRL enables stable optimization while cont
    
[^363]: 论AI算法进步之源：来自语言模型预训练的证据

    On the Origin of Algorithmic Progress in AI: Evidence from Language Model Pre-Training

    [https://arxiv.org/abs/2511.21622](https://arxiv.org/abs/2511.21622)

    该论文通过消融与缩放实验发现，2012–2023年间AI预训练约22,000倍的算法效率提升无法由小算力下的各项具体创新解释，而主要源于Transformer等算法随规模增大而增强的效率优势。

    

    据估计，在2012年至2023年间，算法使AI预训练的FLOP效率提升了22,000倍（Ho et al., 2024）。通过对该时期关键创新开展消融实验，我们在小算力规模下只能解释其中不到10倍的收益。在调研更广泛的文献后，我们估计未纳入消融实验的其他创新的贡献同样不足10倍，因此在低算力区间，总效率提升不足100倍。这促使我们开展缩放实验，实验证据表明，这一效率差距可以由具有规模依赖性效率提升的算法来解释。具体而言，我们在LSTM与Transformer之间进行了缩放实验，发现了二者在计算最优缩放律上存在指数差异的证据，而许多其他创新则几乎没有表现出缩放差异。这些实验表明，与标准假设相反……

    arXiv:2511.21622v2 Announce Type: replace-cross  Abstract: Algorithms have been estimated to increase AI pretraining FLOP efficiency by a factor of 22,000 between 2012 and 2023 (Ho et al., 2024). Running ablation experiments on key innovations from this time period, we are able to account for less than 10x of these gains at small compute scales. Surveying the broader literature, we estimate that additional innovations not included in our ablations also account for less than 10x, yielding less than 100x efficiency gains total in low-compute regimes. This leads us to conduct scaling experiments, which provide evidence that the efficiency gap can be explained by algorithms with scale-dependent efficiency improvements. In particular, we conduct scaling experiments between LSTMs and Transformers, finding evidence for exponent differences in their compute-optimal scaling law while finding little scaling difference for many other innovations. These experiments demonstrate that, contrary to st
    
[^364]: EEG-X：迈向设备无关且抗噪声的脑电（EEG）基础模型

    EEG-X: Toward Device-Agnostic and Noise-Robust Foundation Model for EEG

    [https://arxiv.org/abs/2511.08861](https://arxiv.org/abs/2511.08861)

    EEG-X是一个设备无关且抗噪声的脑电基础模型，通过基于位置的通道嵌入实现跨异构设备与电极布局的迁移，并通过噪声感知的掩码重建策略与基于字典的卷积变换显著提升对噪声的鲁棒性。

    

    用于脑电分析的基础模型仍处于起步阶段，主要受限于两个关键挑战：（1）由于记录设备和电极配置的差异所导致的数据集间的变异性；（2）脑电信号的低信噪比（SNR），神经活动常常被伪影和非脑源信号所掩盖。为了应对这些挑战，我们提出了EEG-X，一个设备无关且抗噪声的脑电表征学习基础模型。EEG-X引入了基于位置的通道嵌入，使其能够在异构的脑电设备和电极布局之间实现稳健的迁移。为了提高对噪声的鲁棒性，EEG-X采用了一种新颖的噪声感知掩码-重建策略，即重建经过伪影清理的信号而非原始含噪脑电信号，并提出了一种新颖的基于字典的卷积变换，通过在结构化特征空间中对信号进行比较，从而改进基于重建的预训练。

    arXiv:2511.08861v2 Announce Type: replace  Abstract: Foundation models for EEG analysis are still in their infancy, limited by two key challenges: (1) variability across datasets caused by differences in recording devices and electrode configurations, and (2) the low signal-to-noise ratio (SNR) of EEG, where neural activity is often buried under artifacts and non-brain sources. To address these challenges, we present EEG-X, a device-agnostic and noise-robust foundation model for EEG representation learning. EEG-X introduces a location-based channel embedding that enables robust transfer across heterogeneous EEG devices and electrode layouts. To improve robustness against noise, EEG-X employs a novel noise-aware masking-reconstruction strategy that reconstructs artifact-cleaned signals rather than raw noisy EEG and introduces a novel Dictionary-based Convolutional Transformation (DiCT) that improves reconstruction-based pretraining by comparing signals in a structured feature space inst
    
[^365]: 演化计算：自然生成式AI与创造性合成的范式

    Evolutionary Computation: A Paradigm of Natural Generative AI and Creative Synthesis

    [https://arxiv.org/abs/2510.08590](https://arxiv.org/abs/2510.08590)

    本文提出演化计算（EC）可作为“自然生成式AI”（NGAI）的新范式，其随机变异与自然选择机制能够突破传统统计学习的模仿局限，实现真正的创造性合成与分布外的演化跃迁。

    

    生成式AI依赖于统计学习和基于梯度的局部引导，其产生的结果往往更倾向于模仿而非真正的创造性。这篇立场论文提出了将演化计算（EC）视为自然生成式AI（NGAI）范式的观点，其中随机变异、多样化和自然选择为生成式探索和创造性合成提供了另一种基础。EC提供了多种具有不同生成能力的变异算子：以父代为中心的算子可诱导出与传统生成模型类似的种群分布，而破坏性算子则能够产生结构化的演化跃迁，生成分布之外的产物（如程序、设计或AI智能体本身）。这一视角揭示了多任务EC中某些破坏性变异与分布乘积种群结构之间独特的理论关联。一种数学……（原文摘要在此处截断）

    arXiv:2510.08590v2 Announce Type: replace-cross  Abstract: Generative AI relies on statistical learning and local gradient-based guidance, often resulting in artifacts that may be more emulative than genuinely creative. This position paper advances a perspective on Evolutionary Computation (EC) as a paradigm of Natural Generative AI (NGAI), in which stochastic variation, diversification, and natural selection present an alternative foundation for generative exploration and creative synthesis. EC provides an array of variation operators with varied generative capabilities: parent-centric operators induce population distributions analogous to conventional generative models, while disruptive operators can produce structured evolutionary leaps to out-of-distribution artifacts (e.g., programs, designs, or AI agents themselves). This lens reveals a unique theoretical association between certain disruptive variations in multitask EC and a product-of-distributions population structure. A mathe
    
[^366]: 面向物理系统建模的随机插值方法基准测试

    Benchmarking Stochastic Interpolants for Modeling Physical Systems

    [https://arxiv.org/abs/2509.26282](https://arxiv.org/abs/2509.26282)

    该论文针对流体和气候系统，将基于随机插值的生成模型与传统高斯去噪框架进行基准对比，发现随机插值在自回归预测任务的逐点精度、谱保真度和概率技巧方面能够匹敌或超越高斯去噪基线。

    

    生成式模型近来已成为物理系统的强大代理模型，在精度、稳定性和/或统计保真度方面展现出不断提升的性能。然而，大多数方法依赖于对高斯分布进行迭代去噪，这一选择对于自回归预测任务可能并非最有效，因为在自回归预测中，当前状态与未来状态往往密切相关。因此，我们针对一系列流体和气候系统，将基于随机插值（可学习任意密度之间映射）的生成模型与标准的高斯去噪框架进行基准对比。遵循先前的工作，我们考虑了用于动力系统的传输映射，该映射学习在当前状态的点质量与未来状态的条件分布之间构建插值。我们发现，随机插值在逐点精度、谱保真度和概率技巧方面能够匹敌甚至超越高斯去噪基线。此外，

    arXiv:2509.26282v2 Announce Type: replace  Abstract: Generative models have recently emerged as powerful surrogates for physical systems, demonstrating increased accuracy, stability, and/or statistical fidelity. However, most approaches rely on iteratively denoising a Gaussian, a choice that may not be the most effective for autoregressive prediction tasks, where the current and future states are often closely related. We therefore benchmark generative models based on stochastic interpolants, which can learn maps between arbitrary densities, against standard Gaussian-denoising frameworks across a range of fluid and climate systems. Following prior work, we consider a transport map for dynamical systems that learns an interpolant between a point mass at the current state and a conditional distribution of future states. We find that stochastic interpolants can match or outperform Gaussian-denoising baselines in pointwise accuracy, spectral fidelity, and probabilistic skill. Furthermore, 
    
[^367]: BALF：面向免微调模型压缩的预算约束激活感知低秩分解

    BALF: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Model Compression

    [https://arxiv.org/abs/2509.25136](https://arxiv.org/abs/2509.25136)

    提出BALF框架，将激活感知低秩分解从线性层扩展到标准卷积和分组卷积并支持秩亏激活，同时基于输出失真在显式FLOP或参数预算下通过拉格朗日松弛自动分配各层秩，实现端到端免微调的模型压缩。

    

    激活感知低秩分解技术能够取得出色的压缩效果，但这类方法通常局限于线性层，且现有的基于白化的理论往往对层激活做出隐式的满秩假设。我们引入了一个层表示框架，使我们能够理解并将低秩分解扩展到线性层之外，包括标准卷积和分组卷积。在该框架内，我们对基于白化的激活感知分解给出了更广泛的刻画，能够自然地覆盖秩亏缺的激活，并证明了由此得到的投影在层输出失真意义上是最优的。随后，我们以输出失真作为代理指标，通过对该分配问题的拉格朗日松弛，以可忽略的开销在显式的FLOP或参数量预算下为各层分配秩。这些组件共同构成了BALF——一个端到端、免微调的模型压缩流水线。

    arXiv:2509.25136v4 Announce Type: replace  Abstract: Activation-aware low-rank factorization techniques yield strong compression results, but they are generally confined to linear layers, and existing whitening-based theory typically makes an implicit full-rank assumption on layer activations. We introduce a layer representation framework that allows us to understand and extend low-rank factorization beyond linear layers, including standard and grouped convolutions. Within this framework, we provide a broader characterization of whitening-based activation-aware factorization that naturally covers rank-deficient activations, and show that the resulting projection is optimal in terms of layer output distortion. We then use output distortion as a proxy to allocate per-layer ranks under explicit FLOP or parameter count budgets, with negligible overhead via a Lagrangian relaxation of the allocation problem. Together, these components form BALF, an end-to-end, fine-tuning-free pipeline for m
    
[^368]: OptiMind：通过建模指导与多轮学习改进用于优化建模的小语言模型

    OptiMind: Improving Small Language Models for Optimization Modeling through Formulation Guidance and Multi-Turn Learning

    [https://arxiv.org/abs/2509.22979](https://arxiv.org/abs/2509.22979)

    OptiMind通过将常见建模错误提炼为类别特定提示、并利用求解器反馈进行多轮强化学习，有效提升了小语言模型在优化建模任务中的表现。

    

    数学优化驱动着供应链、物流、能源和调度等领域的决策，但将自然语言问题转化为求解器可执行的数学模型仍然是一个瓶颈。虽然前沿模型在这项任务上展现出潜力，但许多实际部署场景需要在本地运行的小语言模型（SLM），以满足隐私、延迟和成本方面的约束。现有的SLM受到噪声数据、模糊的基准测试以及一次性推理的限制，而后者忽视了优化建模的迭代性和反馈驱动的本质。为应对这些挑战，我们提出了OptiMind，这是一个将数据清洗与类别特定提示相结合、并利用求解器反馈进行训练的系统。OptiMind将反复出现的建模错误提炼为类别特定的提示，用以指导修复模糊问题并重新生成求解轨迹。这一过程提供了可靠的训练数据，随后用于多轮强化学习，教会模型不断修订其建模方案。

    arXiv:2509.22979v3 Announce Type: replace  Abstract: Mathematical optimization drives decisions in supply chains, logistics, energy, and scheduling, but translating natural-language problems into solver-executable formulations remains a bottleneck. While frontier models show promise for this task, many deployments require small language models (SLMs) that run locally under privacy, latency, and cost constraints. Existing SLMs are limited by noisy data, ambiguous benchmarks, and one-shot inference that ignores the iterative, feedback-driven nature of optimization modeling. To address these challenges, we introduce OptiMind, a system that combines data cleaning and class-specific hints with training using solver feedback. OptiMind distills recurring formulation errors into class-specific hints that guide the repair of ambiguous problems and the regeneration of solution trajectories. This process provides reliable training data, which is then used for multi-turn RL to teach models to revi
    
[^369]: 欧几里得若尔当代数的差分隐私及其在私有对称锥规划中的应用

    Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming

    [https://arxiv.org/abs/2509.16915](https://arxiv.org/abs/2509.16915)

    本文提出了一种适用于欧几里得若尔当代数输出的通用高斯差分隐私机制，并结合乘性权重更新方法设计了求解对称锥规划的隐私保护算法，从而为半定规划提供了差分隐私算法，解决了一个重要的开放问题。

    

    本文研究了输出位于欧几里得若尔当代数中的函数的差分隐私机制。欧几里得若尔当代数涵盖了许多重要的数学结构，并构成了线性规划、二阶锥规划和半定规划的基础。我们的主要贡献是为此类函数提出了一种通用的高斯机制，其敏感度以$\ell_2$、$\ell_1$和$\ell_\infty$范数来度量。值得注意的是，该框架涵盖了函数输出为对称矩阵这一重要情形，此时敏感度以Frobenius范数、核范数或谱范数来度量。我们进一步将乘性权重更新方法与我们的通用高斯机制相结合，推导出了在各种设置下求解对称锥规划的隐私保护算法。作为应用，我们提出了半定规划的差分隐私算法，解决了一个重要的开放问题。

    arXiv:2509.16915v2 Announce Type: replace-cross  Abstract: In this paper, we study differentially private mechanisms for functions whose outputs lie in a Euclidean Jordan algebra. Euclidean Jordan algebras capture many important mathematical structures and form the foundation of linear programming, second-order cone programming, and semidefinite programming. Our main contribution is a generic Gaussian mechanism for such functions, with sensitivity measured in $\ell_2$, $\ell_1$, and $\ell_\infty$ norms. Notably, this framework includes the important case where the function outputs are symmetric matrices, and sensitivity is measured in the Frobenius, nuclear, or spectral norm. We further derive private algorithms for solving symmetric cone programs under various settings, using a combination of the multiplicative weights update method and our generic Gaussian mechanism. As an application, we present differentially private algorithms for semidefinite programming, resolving a major open q
    
[^370]: 基于近似正交基的低阶多项式下界

    Low-degree lower bounds via almost orthonormal bases

    [https://arxiv.org/abs/2509.09353](https://arxiv.org/abs/2509.09353)

    本文提出了一种基于近似正交多项式基的更直接证明策略，用于在具有植入结构的随机图模型中建立低阶多项式下界，克服了传统规范正交多项式族方法在估计任务和复杂检验问题中的局限性。

    

    低阶多项式已成为在多种高维统计模型中为统计-计算差距提供证据的有力范式[Wein25]。对于检测问题——即目标是将植根分布 $\mathbb{P}'$ 与具有独立成分的零假设分布 $\mathbb{P}$ 进行比较检验——标准方法是使用 $\mathbb{L}^2(\mathbb{P})$-规范正交多项式族来界定优势。然而，对于估计任务或更复杂的检验问题（其中 $\mathbb{P}$ 具有某种植入结构，因此不存在简单的 $\mathbb{L}^2(\mathbb{P})$-正交多项式族），这种方法会失效。为应对这一挑战，已有若干技术性变通方案被提出[SW22,SW25]，但其实现较为繁琐。在本工作中，我们提出了一种更直接的证明策略。聚焦于随机图模型，我们构造了一个多项式基

    arXiv:2509.09353v3 Announce Type: replace-cross  Abstract: Low-degree polynomials have emerged as a powerful paradigm for providing evidence of statistical--computational gaps across a variety of high-dimensional statistical models [Wein25]. For detection problems---where the goal is to test a planted distribution $\mathbb{P}'$ against a null distribution $\mathbb{P}$ with independent components---the standard approach is to bound the advantage using an $\mathbb{L}^2(\mathbb{P})$-orthonormal family of polynomials. However, this method breaks down for estimation tasks or more complex testing problems where $\mathbb{P}$ has some planted structures, so that no simple $\mathbb{L}^2(\mathbb{P})$-orthogonal polynomial family is available. To address this challenge, several technical workarounds have been proposed [SW22,SW25], though their implementation can be delicate. In this work, we propose a more direct proof strategy. Focusing on random graph models, we construct a basis of polynomials
    
[^371]: REVELIO：面向跨领域泛化的通用多模态任务负荷估计

    REVELIO: Universal Multimodal Task Load Estimation for Cross-Domain Generalization

    [https://arxiv.org/abs/2509.01642](https://arxiv.org/abs/2509.01642)

    该论文提出了REVELIO统一框架、数据集和评估协议，通过在实验室工作记忆任务、驾驶模拟器和游戏等多领域采集的96个会话多模态生理信号，实现跨领域任务负荷估计的鲁棒泛化，多模态融合AUC最高达0.87，显著优于单一信号输入。

    

    响应用户心理状态的自适应交互系统需要在其训练环境之外仍保持可靠的任务负荷估计。在单一环境中构建的模型通常难以迁移，这限制了其实际应用。本工作提出了REVELIO（基于多模态观测端到端学习的鲁棒估计），这是一个用于跨领域任务负荷估计的统一框架、数据集和评估协议。研究在96个会话中，在一致的硬件和标注条件下，记录了心脏、皮肤电、肌肉、呼吸、皮肤温度、眼动追踪和面部信号，涵盖一个实验室工作记忆任务、一个驾驶模拟器和两个游戏。多模态融合的受试者工作特征曲线下面积（AUC）最高达到0.87，高于单输入的眼动追踪（0.83）、面部动作单元（0.74）和生物信号（0.62）。多领域训练在训练中见过的领域上保持校准，而移除目标领域会导致……（摘要在此处被截断）

    arXiv:2509.01642v2 Announce Type: replace  Abstract: Adaptive interactive systems that respond to user mental state require task load estimation that stays reliable outside its training context. Models built in one setting rarely transfer, limiting real-world use. This work presents Robust Estimation Via End-to-End Learning of Multimodal Observations (REVELIO), a unified framework, dataset and evaluation protocol for cross-domain task load estimation. Cardiac, electrodermal, muscle, respiration, skin temperature, eye tracking and facial signals were recorded in 96 sessions, spanning a laboratory working memory task, a driving simulator and two games under consistent hardware and annotation. Multimodal fusion reached up to 0.87 area under the receiver operating characteristic curve, above single input eye tracking (0.83), facial action units (0.74) and biosignals (0.62). Multi-domain training stayed calibrated on domains seen in training, whereas removing the target domain caused the cl
    
[^372]: 深度双重Q学习（Deep Double Q-learning）

    Deep Double Q-learning

    [https://arxiv.org/abs/2507.00275](https://arxiv.org/abs/2507.00275)

    本文提出深度双重Q学习（DDQL），通过双重Q学习目标显式训练两个Q函数实现估计器解耦，并借助较低回放比率、更长目标网络更新间隔和共享层等技术稳定训练，在57个Atari游戏上总体性能超越Double DQN。

    

    双重Q学习（Double Q-learning）是一种经典控制算法，用于缓解Q学习的最大化偏差。为此，它显式地训练两个独立的动作值函数，并在计算自举目标时利用它们将动作选择与动作评估解耦。Double DQN 将这种解耦思想应用于深度强化学习（RL），但它只直接训练单个动作值函数，并未完全解耦其估计器。因此，这两个估计器之间仍然存在相关性，高估问题依然存在。本文提出了深度双重Q学习（DDQL），这是一种通过双重Q学习目标显式训练两个Q函数的深度强化学习算法。DDQL 通过多种技术组合来稳定训练，包括较低的经验回放比率、更长的目标网络更新间隔以及共享层。在57个Atari 2600游戏中，DDQL 的总体性能优于 Double DQN，在47个游戏上超越了它（原文在此处截断）。

    arXiv:2507.00275v3 Announce Type: replace-cross  Abstract: Double Q-learning is a classical control algorithm that mitigates the maximization bias of Q-learning. To do so, it explicitly trains two independent action-value functions and uses them to decouple action-selection from action-evaluation when computing bootstrap targets. Double DQN adapts this decoupling to deep reinforcement learning (RL) but directly trains only a single action-value function and does not fully decouple its estimators. Consequently, the two estimators remain correlated, and overestimation persists. In this paper, we introduce Deep Double Q-learning (DDQL), a deep RL algorithm that explicitly trains two Q-functions through the Double Q-learning objective. DDQL stabilizes training through a combination of techniques, including lower replay ratios, longer target network update intervals, and shared layers. Across 57 Atari 2600 games, DDQL improves aggregate performance over Double DQN, outperforming it on 47 ga
    
[^373]: 迭代正交化缩放定律

    Iterative Orthogonalization Scaling Laws

    [https://arxiv.org/abs/2505.04005](https://arxiv.org/abs/2505.04005)

    本文从理论和实验上证明了随机矩阵奇异值随规模增大而缩小的现象，揭示了Muon优化器的迭代正交化过程在大规模场景下可能存在的问题，但未给出解决方案。

    

    Muon优化器近来备受关注，被视为可能取代似乎无处不在的Adam优化器的候选方案。近期，研究者已着手记录Muon下超参数的缩放定律，例如权重衰减和学习率。然而，在更大的规模下，Muon中的迭代正交化过程可能会遇到一个潜在问题，因为随机矩阵的奇异值会随着规模的增大而缩小。本文从理论和实验两方面展示了随机矩阵的这种缩放行为，但并未提出应对该问题的解决方案。

    arXiv:2505.04005v3 Announce Type: replace  Abstract: The muon optimizer has picked up much attention as of late as a possible replacement to the seemingly omnipresent Adam optimizer. Recently, care has been taken to document the scaling laws of hyper-parameters under muon such as weight decay and learning rate. However, at much larger scales the iterative orthogonalization procedure present in muon may suffer a possible issue as the singular values of random matrices shrink with scale. This paper shows this scaling behavior theoretically and empirically on random matrices but does not suggest what to do about it.
    
[^374]: 师生协同适应的计算模型

    A Computational Model of Teacher-Student Co-Adaptation

    [https://arxiv.org/abs/2505.02853](https://arxiv.org/abs/2505.02853)

    本文提出了一个贝叶斯计算框架来建模部分可观测条件下师生之间的双向协同适应，通过模拟比较五种交互模式，发现学生主动提问与教师主动延续能互补地解决信息缺口，突破了传统单向教学模型的局限。

    

    有效的教学涉及相互适应：教师根据学生的反应调整教学，而学生则主动寻求信息以解决不确定性。然而，许多教学的计算模型将教学视为单向过程。本文提出了一个贝叶斯计算框架，用于在一个风格化的概念学习任务中研究部分可观测条件下的师生协同适应。学生被组织成异质性群体，他们在目标概念中可观察到的特征各不相同，教师则从有限的反馈中推断这些约束。我们在100次独立匹配的模拟运行中比较了五种交互模式（主动教学、主动学习、轮流式交互、适应性教学和完全协同适应）以及三种教师主导的对照条件。学习者发起的提问解决了基线教师停止规则所留下的信息空白，而教师主导的教学延续同样能够……（摘要原文截断）

    arXiv:2505.02853v2 Announce Type: replace-cross  Abstract: Effective teaching involves mutual adaptation: teachers adjust instruction based on student responses, while students actively seek information to resolve uncertainty. Yet many computational models of teaching treat instruction as a one-way process. This paper presents a Bayesian computational framework for studying teacher-student co-adaptation under partial observability in a stylized concept-learning task. Students are organized in heterogeneous groups that differ in which features of a target concept they can observe, and the teacher infers these constraints from limited feedback. We compare five interaction modes (active teaching, active learning, turn-based interaction, adaptive teaching and full co-adaptation) and three teacher-led controls across 100 independent matched simulation runs. Learner-initiated queries resolve information gaps left by the baseline teachers' stopping rules, while teacher-led continuation also a
    
[^375]: 少数派报告：在基准真值数据标注中平衡成本与质量

    Minority Reports: Balancing Cost and Quality in Ground Truth Data Annotation

    [https://arxiv.org/abs/2504.09341](https://arxiv.org/abs/2504.09341)

    该论文提出了一个通过识别与多数投票标签不一致的“少数派报告”来主动剪枝冗余重复标注的理论框架与滚动策略，可在保留99.9%以上参考标签的同时将重复标注成本降低超过35%。

    

    数据标注是基于机器学习技术开发中至关重要但成本高昂的环节，需要多名人工标注者对数据集中的每个数据点进行重复标注。我们通过识别“少数派报告”——即与参考多数投票标签不一致的标注者响应——来探索标注准确性与成本之间的权衡。这些少数派报告可能表明存在不必要的冗余，因此可以在任务执行前主动进行剪枝。我们开发了一个理论框架，用于在目标标签质量下选择任务所需的重复标注数量以及剪枝策略。我们在两个通过专业平台重新标注的计算机视觉数据集上估计了少数派报告的风险，然后在任务分配流上评估了一种滚动剪枝策略，在保留超过99.9%参考多数投票标签的同时，成功剪枝了超过35%的重复标注。

    arXiv:2504.09341v2 Announce Type: replace  Abstract: Data annotation is an essential but costly aspect of developing machine learning-based technology, requiring multiple human workers to repeatedly label each data point in a data set. We explore the tradeoff between annotation accuracy and cost by identifying minority reports---worker responses disagreeing with the reference majority-vote label---that may indicate unnecessary redundancy, and can therefore be proactively pruned before they are executed. We develop a theoretical framework for choosing the number of repeated annotations required for a task and pruning policies at a target label quality. We estimate minority-report risk on two computer vision datasets re-annotated through a professional platform, then evaluate a rolling policy on the assignment stream to prune over 35% of repeats while preserving more than 99.9% of reference majority-vote labels.
    
[^376]: WESAD压力分类的跨模态研究

    Cross-Modality Investigation on WESAD Stress Classification

    [https://arxiv.org/abs/2502.18733](https://arxiv.org/abs/2502.18733)

    本研究基于WESAD数据集开发单模态Transformer模型，利用ECG、EDA、EMG等多种生理信号进行压力检测，实现了99.73%至99.95%的最先进分类性能。

    

    深度学习的日益普及推动了其在医疗保健领域的广泛应用，人工智能和传感器技术的进步提升了诊断、治疗和监测的水平。在移动健康领域，AI驱动的工具能够实现压力等健康状况的早期诊断和持续监测。可穿戴技术和多模态生理数据使得压力检测越来越可行，但模型的效果取决于数据的质量、数量和模态。本研究基于WESAD数据集开发了用于压力检测的Transformer模型，在心电图（ECG）、皮肤电活动（EDA）、肌电图（EMG）、呼吸频率（RESP）、体温（TEMP）和三轴加速度计（ACC）信号上进行训练。结果证明了单模态Transformer在分析生理信号方面的有效性，实现了最先进的性能，在压力分类任务上准确率、精确率和召回率达到99.73%至99.95%。

    arXiv:2502.18733v2 Announce Type: replace-cross  Abstract: Deep learning's growing prevalence has driven its widespread use in healthcare, where AI and sensor advancements enhance diagnosis, treatment, and monitoring. In mobile health, AI-powered tools enable early diagnosis and continuous monitoring of conditions like stress. Wearable technologies and multimodal physiological data have made stress detection increasingly viable, but model efficacy depends on data quality, quantity, and modality. This study develops transformer models for stress detection using the WESAD dataset, training on electrocardiograms (ECG), electrodermal activity (EDA), electromyography (EMG), respiration rate (RESP), temperature (TEMP), and 3-axis accelerometer (ACC) signals. The results demonstrate the effectiveness of single-modality transformers in analyzing physiological signals, achieving state-of-the-art performance with accuracy, precision and recall values in the range of $99.73\%$ to $99.95\%$ for st
    
[^377]: 我们真的需要Rademacher复杂度吗？

    Do we really need the Rademacher complexities?

    [https://arxiv.org/abs/2502.15118](https://arxiv.org/abs/2502.15118)

    该论文证明只需利用函数类协方差结构的粗略信息，即可在重尾情形下推导出与高斯情形最先进速率相匹配的样本复杂度上界，从而表明学习问题无需依赖难以控制的局部化Rademacher复杂度。

    

    我们研究了在凸函数类中关于平方损失学习这一基本问题。在次高斯设定下，已知最尖锐的样本复杂度界是由与极限高斯过程相关联的不动点所刻画的。而在更一般的情形下（例如重尾场景），最先进的界依赖于局部化Rademacher复杂度，这类复杂度通常难以控制，且可能远大于其对应的极限高斯复杂度。我们证明，只要能够获得关于底层函数类协方差结构的粗略信息，就可以推导出样本复杂度的上界，该上界在高斯情形下能达到最先进的速率——即使是在重尾情形下同样成立。因此，在相当弱的最小假设下，具有相同L2结构的学习问题——即便是涉及重尾分布的问题——也拥有相同的上界。

    arXiv:2502.15118v2 Announce Type: replace-cross  Abstract: We study the fundamental problem of learning with respect to the squared loss in a convex class. In the subgaussian setting, the sharpest known sample complexity bounds are characterised via fixed points associated with the limiting gaussian process. In more general settings (for example, heavier-tailed scenarios) the state-of-the-art bounds rely on localised Rademacher complexities, which are often difficult to control and can be considerably larger than their limiting gaussian counterparts.   We show that if one has access to coarse information on the covariance structure of the underlying class, it is possible to derive an upper bound on the sample complexity that matches the state-of-the-art rates in the gaussian setting---even in heavy-tailed regimes. As a consequence, under rather minimal assumptions, learning problems that share the same $L_2$-structure---even those involving heavy-tailed distributions---admit the same u
    
[^378]: 基于评论器的正则化器学习中的星形几何

    The Star Geometry of Critic-Based Regularizer Learning

    [https://arxiv.org/abs/2408.16852](https://arxiv.org/abs/2408.16852)

    该论文开创性地研究了在星形体的规范函数这一正则化器族上优化基于评论器的损失函数，为学习到的任务相关正则化器的结构及其与测量数据和真实数据分布之间的几何联系提供了首个理论视角。

    

    变分正则化是解决统计推断任务和逆问题的一种经典技术，而现代数据驱动方法通过深度神经网络对正则化器进行参数化，展现出令人瞩目的实证性能。沿此思路的最新研究学习依赖于具体任务的正则化器：其做法是在一个无监督的、基于评论器的损失函数中融合测量数据与真实数据的信息，其中正则化器为可能的数据赋予较低的值，为不太可能的数据赋予较高的值。然而，关于通过这一过程学习到的正则化器具有何种结构，以及它与两种数据分布之间的关系，目前几乎没有相关理论。为了在这一挑战上取得进展，我们开创性地研究了在特定正则化器族——星形体的规范函数（gauge，即 Minkowski 泛函）——上优化基于评论器的损失函数以学习正则化器。该正则化器族包含了常见的……

    arXiv:2408.16852v3 Announce Type: replace  Abstract: Variational regularization is a classical technique to solve statistical inference tasks and inverse problems, with modern data-driven approaches parameterizing regularizers via deep neural networks showcasing impressive empirical performance. Recent works along these lines learn task-dependent regularizers. This is done by integrating information about the measurements and ground-truth data in an unsupervised, critic-based loss function, where the regularizer attributes low values to likely data and high values to unlikely data. However, there is little theory about the structure of regularizers learned via this process and how it relates to the two data distributions. To make progress on this challenge, we initiate a study of optimizing critic-based loss functions to learn regularizers over a particular family of regularizers: gauges (or Minkowski functionals) of star-shaped bodies. This family contains regularizers that are common
    
[^379]: 通过弱差分隐私在线学习实现纳什激励相容的在线机制学习

    Nash Incentive-compatible Online Mechanism Learning via Weakly Differentially Private Online Learning

    [https://arxiv.org/abs/2407.04898](https://arxiv.org/abs/2407.04898)

    本文提出一种在弱差分隐私在线学习算法的推荐与惩罚不真实行为的承诺机制之间随机选择的方法，解决了多轮在线机制设计中面对非短视参与者时保持全流程激励相容的难题，并在对抗环境下实现了次线性遗憾度。

    

    我们研究一个多轮机制设计问题，其中我们在一系列轮次中与一组参与者（智能体）进行交互。我们希望设计一个激励相容（IC）的在线学习方案，在给定的一类机制中最大化特定应用的目标，而无需预先了解参与者的类型分布。即使该类中的每个机制在单轮中都是激励相容的，如果算法在每一轮中简单地从该类机制中进行选择，整个学习过程对于跨越多轮出现的非短视买家而言可能不再是激励相容的。在每一轮中，我们的方法在弱差分隐私在线学习算法（例如Hedge算法）的推荐与惩罚不真实行为的承诺机制之间进行随机选择。我们的方法是激励相容的，并在对抗性环境下针对特定应用目标实现了 $O(T^{\frac{1+h}{2}})$ 的遗憾度，其中 $h$ 量化了参与者的远见程度。

    arXiv:2407.04898v3 Announce Type: replace-cross  Abstract: We study a multi-round mechanism design problem, where we interact with a set of agents over a sequence of rounds. We wish to design an incentive-compatible (IC) online learning scheme to maximize an application-specific objective within a given class of mechanisms, without prior knowledge of the agents' type distributions. Even if each mechanism in this class is IC in a single round, if an algorithm naively chooses from this class on each round, the entire learning process may not be IC against non-myopic buyers who appear over multiple rounds. On each round, our method randomly chooses between the recommendation of a weakly differentially private online learning algorithm (e.g., Hedge), and a commitment mechanism which penalizes non-truthful behavior. Our method is IC and achieves $O(T^{\frac{1+h}{2}})$ regret for the application-specific objective in an adversarial setting, where $h$ quantifies the long-sightedness of the ag
    
[^380]: 神经网络中的鲁棒性验证

    Robustness Verifcation in Neural Networks

    [https://arxiv.org/abs/2403.13441](https://arxiv.org/abs/2403.13441)

    本文研究了神经网络中的鲁棒性验证问题，提出了一个理论框架来解决安全性和效率性问题，并分析了它们的计算复杂性。

    

    在本文中，我们研究了神经网络计算的形式化验证问题。其中的重要问题包括鲁棒性和最小化问题，例如在线性规划实例的符号输入和输出规范性的基础上，是否存在有效的输入使得网络计算出有效的输出？这个性质是否对所有有效的输入都成立？两个给定的网络是否计算出相同的函数？是否存在一个更小的网络来计算相同的函数？这些问题的复杂性最近从实用角度进行了研究，并通过启发式算法进行了近似。我们通过提供一个理论框架来补充这些成果，该框架使我们能够在神经网络中交换安全性和效率性问题，并分析它们的计算复杂性。我们证明了这些问题在半线性设置下是可以克服的，也就是说...

    arXiv:2403.13441v1 Announce Type: cross  Abstract: In this paper we investigate formal verification problems for Neural Network computations. Of central importance will be various robustness and minimization problems such as: Given symbolic specifications of allowed inputs and outputs in form of Linear Programming instances, one question is whether there do exist valid inputs such that the network computes a valid output? And does this property hold for all valid inputs? Do two given networks compute the same function? Is there a smaller network computing the same function?   The complexity of these questions have been investigated recently from a practical point of view and approximated by heuristic algorithms. We complement these achievements by giving a theoretical framework that enables us to interchange security and efficiency questions in neural networks and analyze their computational complexities. We show that the problems are conquerable in a semi-linear setting, meaning that 
    
[^381]: 针对目标营销的强盗利润最大化

    Bandit Profit-maximization for Targeted Marketing

    [https://arxiv.org/abs/2403.01361](https://arxiv.org/abs/2403.01361)

    该论文研究了针对目标营销的强盗利润最大化问题，并提出了在敌对强盗情境下的近乎最优算法。

    

    我们研究了一个顺序利润最大化问题，优化价格和像营销支出这样的辅助变量。具体来说，我们旨在在一个任意序列的多个需求曲线上最大化利润，每个曲线依赖于一个不同的辅助变量，但共享相同的价格。一个典型的例子是针对营销，其中一家公司（卖方）希望在多个市场上销售产品。公司可以为不同市场投入不同的营销支出以优化客户获取，但必须在所有市场上保持相同的价格。此外，市场可能具有异质的需求曲线，每个需求曲线对价格和营销支出的响应方式不同。公司的目标是最大化毛利润，即总收入减去营销成本。

    arXiv:2403.01361v1 Announce Type: new  Abstract: We study a sequential profit-maximization problem, optimizing for both price and ancillary variables like marketing expenditures. Specifically, we aim to maximize profit over an arbitrary sequence of multiple demand curves, each dependent on a distinct ancillary variable, but sharing the same price. A prototypical example is targeted marketing, where a firm (seller) wishes to sell a product over multiple markets. The firm may invest different marketing expenditures for different markets to optimize customer acquisition, but must maintain the same price across all markets. Moreover, markets may have heterogeneous demand curves, each responding to prices and marketing expenditures differently. The firm's objective is to maximize its gross profit, the total revenue minus marketing costs.   Our results are near-optimal algorithms for this class of problems in an adversarial bandit setting, where demand curves are arbitrary non-adaptive seque
    
[^382]: 使用生成对抗网络进行额颞叶痴呆与阿尔茨海默病的鉴别诊断

    Differential Diagnosis of Frontotemporal Dementia and Alzheimer's Disease using Generative Adversarial Network

    [https://arxiv.org/abs/2109.05627](https://arxiv.org/abs/2109.05627)

    本研究提出了一种基于生成对抗网络的新框架，利用磁共振图像中粗到细结构尺度的体积特征，实现了对额颞叶痴呆、阿尔茨海默病和正常对照的高精度鉴别诊断。

    

    额颞叶痴呆（FTD）和阿尔茨海默病（AD）是两种常见的痴呆类型，由于其临床症状模式相似，很容易被相互误诊。区分这两种痴呆类型对于确定疾病特异性的干预和治疗至关重要。近年来，基于深度学习的方法在医学图像计算领域的发展在许多二分类任务中表现出色，但其在鉴别诊断中的应用，例如基于神经影像的多种痴呆类型鉴别，尚未被探索。在本研究中，提出了一种新颖的框架，使用生成对抗网络技术，基于从磁共振成像扫描中以粗到细结构尺度提取的体积特征，来区分额颞叶痴呆、阿尔茨海默病和正常对照受试者。在1,954张图像上进行的10折交叉验证实验取得了很高的（分类性能）。

    arXiv:2109.05627v2 Announce Type: cross  Abstract: Frontotemporal dementia and Alzheimer's disease are two common forms of dementia and are easily misdiagnosed as each other due to their similar pattern of clinical symptoms. Differentiating between the two dementia types is crucial for determining disease-specific intervention and treatment. Recent development of Deep-learning-based approaches in the field of medical image computing are delivering some of the best performance for many binary classification tasks, although its application in differential diagnosis, such as neuroimage-based differentiation for multiple types of dementia, has not been explored. In this study, a novel framework was proposed by using the Generative Adversarial Network technique to distinguish FTD, AD and normal control subjects, using volumetric features extracted at coarse-to-fine structural scales from Magnetic Resonance Imaging scans. Experiments of 10-folds cross-validation on 1,954 images achieved high
    

