# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tokenisation is not in APX](https://arxiv.org/abs/2611.34938) | 本文证明除非P=NP，分词化问题无法在多项式时间内近似到最优解的任意常数因子（即不属于APX类），且直接分词即使在一元字符表上也是APX难的。 |
| [^2] | [Can image generation help visual reasoning? An empirical study](https://arxiv.org/abs/2611.34935) | 这项实证研究发现，通过生成辅助图像来帮助视觉推理在大多数情况下反而会降低性能——在13组视觉语言模型与生成器的配对实验中平均准确率下降3.7个百分点，十个任务中九个出现下降——挑战了“多模态模型可以用图像思考”的假设。 |
| [^3] | [Attack Policy Selection can Mislead Monitor Evaluators](https://arxiv.org/abs/2611.34934) | 攻击策略的生成方法和优化程度会极大影响AI监测器假阴性率的测量结果（从0.5%到23.0%不等），这表明仅使用基础攻击方法的监测器规避评估很容易产生误导性结论。 |
| [^4] | [A gene informed benchmark for BioScoring of pathology foundation models](https://arxiv.org/abs/2611.34930) | 本文提出BioScore基准，通过空间转录组学与完全无监督的稀疏自编码器量化病理学基础模型对52种细胞身份的生物学表征能力，发现所有病理学模型的得分均高于自然图像模型。 |
| [^5] | [Diverse Alternative Reasoning Traces for Hard Math Problems](https://arxiv.org/abs/2611.34928) | 提出了DART轨迹挖掘方法，通过分解被拒绝的推理轨迹并基于位置加权不确定性与参考答案引导生成刻意不同的替代推理步骤，在相同生成成本下比蒙特卡洛采样多恢复37%的复杂数学问题，且每条合格轨迹成本降低30%。 |
| [^6] | [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923) | 提出混合循环Transformer（MoLT），以KV缓存作为跨深度和跨循环的选择性记忆重用机制，在表达能力与效率之间取得新平衡，并支持超越训练循环次数的高效解码。 |
| [^7] | [Enhancing Maritime Domain Awareness using Latent Space Representations](https://arxiv.org/abs/2611.34899) | 该论文利用地理空间基础模型TerraMind生成的压缩潜在空间嵌入，直接从Sentinel-1 SAR图像中实现船舶检测（ROC-AUC 0.95）、定位（中位误差约300米）和类型分类（62%准确率），有效缓解了海事领域感知中地球观测数据处理的数据引力瓶颈。 |
| [^8] | [What a Drop Reveals: Sim-to-Real Material Estimation for Deformable Objects from Video](https://arxiv.org/abs/2611.34894) | 本论文发现仅用合成数据验证会误导输入表示的选择——仿真中最准确的三维轨迹输入在真实坠落视频上反而最差、且无法靠更好的深度修复，而简单的二维光流短轨迹在真实场景中最有效，从而实现了仅凭仿真训练、从单段真实坠落视频估计可变形物体刚度与屈服应力的sim-to-real方法。 |
| [^9] | [Interpretable and Accurate Hybrid Decision Trees with Selective Case-Based Splits](https://arxiv.org/abs/2611.34887) | 这篇论文提出了一种集成基于特征分割和基于案例分割的混合决策树架构，并引入了一种新颖的选择性基于案例的分割方法，通过搜索最具信息量的特征子集来构建既可解释又准确的决策树。 |
| [^10] | [Stranger Things: When Objects Appear Without Their Typical Neighbors](https://arxiv.org/abs/2611.34881) | 该论文提出了通过归一化逐点互信息（NPMI）度量典型性偏差的诊断框架，并构建了照片级真实感、布局受控的STRANGE-Bench基准（涵盖17个从典型到从不共现的物体对子层级），首次系统性地诊断了现代视觉模型中因依赖上下文邻居而忽视物体自身特征的典型性偏差问题。 |
| [^11] | [Focus Only on What's Important: Training-Free Focus Maps for Frozen Vision Transformers in Long-Video Object Segmentation](https://arxiv.org/abs/2611.34863) | 该论文提出一种无需训练的焦点图策略，通过融合速度偏移掩码与块匹配掩码、相机补偿的意外性检查以及不确定性区域委托机制，使冻结的DINOv3 ViT仅在分割目标预期区域保持全分辨率，在长视频目标分割任务中以更少计算量取得显著性能提升。 |
| [^12] | [Model Weights, Airlocks and Sandboxes: A Call for Technical Validation of Re-identification Tests under the EDPB Anonymisation Guidelines](https://arxiv.org/abs/2611.34859) | 针对EDPB匿名化指南将AI模型纳入监管却未规定重识别测试如何设计与验证的问题，本文提出了基于气闸式安全处理环境的评估框架，区分形式化、经验和间接三类证据，并在模型权重发布、气闸审查和监管沙盒三种场景下提出并探讨了七个开放性技术问题。 |
| [^13] | [From Prediction to Decision: Decision-Focused Fine-Tuning of Autoregressive Tabular Transformers for Cost-Sensitive Financial Events](https://arxiv.org/abs/2611.34858) | 本文提出决策导向微调方法，通过直接最大化预训练自回归表格Transformer在按金额缩放成本矩阵下的期望效用，使模型输出从概率预测转变为面向业务动作的决策优化，从而在成本敏感的金融欺诈检测中优于传统的事后阈值划分方法。 |
| [^14] | [Put the World Back: Timestamps Matter in Time Series Reasoning](https://arxiv.org/abs/2611.34848) | 该论文提出从“世界到序列投影”的视角重新定义时间序列推理，将时间戳视为连接数值观测与外部事件、指向共享世界状态的显式索引，并据此开发了首个能区分真正的基于时间戳的推理与相关性捷径的诊断协议，从而暴露了现有基于准确率的评估基准的局限性。 |
| [^15] | [Knowledge-Base Gradient Descent: Self-Structuring Learning through Question Answering](https://arxiv.org/abs/2611.34841) | 该论文提出“知识库梯度下降”方法，通过诊断问答响应与预期答案之间的差异，并以类似梯度下降的方式迭代地对知识库结构提出编辑、验证并保留能降低损失的编辑，从而让大语言模型应用能够以问题和答案为监督信号自动学习构建最优的知识库结构。 |
| [^16] | [Provably Efficient Offline Discounted Reinforcement Learning under Recursive Entropic Risk](https://arxiv.org/abs/2611.34840) | 本文提出了首个在递归熵风险度量下离线强化学习的可证明高效算法 ERM-VI-LCB，并给出匹配的样本复杂度上下界，同时覆盖风险厌恶与风险追求两种情形。 |
| [^17] | [Not all world models can doubt: how training objective and architecture shape epistemic uncertainty in JEPA and RSSM world models](https://arxiv.org/abs/2611.34830) | 本文通过在十个模拟环境中对JEPA架构的LeWorldModel与RSSM架构的DreamerV3进行受控对比，实证揭示了训练目标与架构从根本上决定了世界模型能否有效表达认知不确定性（即“怀疑”的能力）。 |
| [^18] | [Gradient TD($\lambda$) for Non-Linear Function Approximation](https://arxiv.org/abs/2611.34826) | 本文将梯度TD(λ)算法从线性函数逼近扩展到非线性函数逼近，并继续采用投影贝尔曼误差目标，为现代复杂非线性函数逼近器提供了具有收敛保证的离策略学习算法。 |
| [^19] | [Routeability Before Routing: Routeability Audit Protocol (RAP) and RouteabilityBench for Audited LLM Model Selection](https://arxiv.org/abs/2611.34825) | 该论文提出在进行LLM路由方法评估之前先审计基准测试的“可路由性”，并发布了可运行的审计协议RAP和基准套件RouteabilityBench，实验发现跨四个基准和66个LLM的通用路由增强手段并不能带来一致的改进。 |
| [^20] | [LoopTitan: What Does Parameter Revisit Buy a Mixture-of-Experts?](https://arxiv.org/abs/2611.34821) | 研究发现循环MoE架构中参数重访本身几乎没有带来任何收益，其报告的性能提升实际来自网络形状和残差缩放等设计选择，而MoE本身已具备在固定参数量下用专家换深度的自由。 |
| [^21] | [Deep Reinforcement Learning for Multi-UAV Data Harvesting Under Variable Team Sizes and Agent Loss](https://arxiv.org/abs/2611.34818) | 本文提出F-MADDPG，一种灵活的MADDPG变体，能够在传感器与无人机数量可变的场景下进行数据采集，并对任务中智能体的灾难性失效保持鲁棒性。 |
| [^22] | [Transferring Architecture as an Optimization Prior](https://arxiv.org/abs/2611.34816) | 该论文提出将网络的初始几何视为一种优化先验，并证明在训练前让目标网络与一个不传递特征、不接收梯度、不使用标签的随机初始化引导网络的表示几何对齐，可以通过跨架构迁移优化“命运”来使目标网络受益。 |
| [^23] | [Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models](https://arxiv.org/abs/2611.34813) | 该论文在完全相同的输入、输出规范和评估器条件下，让编码智能体通过编写程序渲染答案视频来与视频生成模型直接对比，证明了编码智能体是比视频模型更强大的视觉推理器。 |
| [^24] | [Barron Smoothness for Layers in Deep Neural Networks](https://arxiv.org/abs/2611.34805) | 该论文提出用Barron范数的逐层推广来等价刻画深度神经网络逐层最优离散化问题，并通过优化景观分析证明局部极小值满足类似自适应样条与有限元方法中的等分布原理。 |
| [^25] | [Guidance as Variance Matching](https://arxiv.org/abs/2611.34801) | 论文揭示扩散模型中的引导并非改善后验均值预测，而是恢复了被低估的方差，并提出无需调参的方差匹配引导闭式规则，可准确复现十一个主流模型的手工引导尺度。 |
| [^26] | [Split-RL: Local Conflict Resolution in Reinforcement Learning](https://arxiv.org/abs/2611.34799) | Split-RL是一种基于梯度提升树的强化学习算法，通过将引导标签集成到树构建中，把矛盾反馈路由到互不相交的状态空间区域，实现冲突信号的局部隔离并防止信号干扰。 |
| [^27] | [PersonaFactory: Stylistic User Prompt Augmentation with Synthetic Persona Bank](https://arxiv.org/abs/2611.34785) | 该论文提出PersonaFactory端到端流程，构建了超74万个刻画背景与交流行为的结构化合成人格库，并以85种语言生成约124万条人格风格化对话，解决了合成用户提示风格单一、无法模拟真实用户交流机制的问题。 |
| [^28] | [Comparative Study of Deep Learning Architectures for Brain Tumour Classification](https://arxiv.org/abs/2611.34781) | 本研究比较了六种深度学习架构在多源MRI数据集上脑肿瘤四分类任务中的表现，并提出了利用MD5哈希和感知哈希去除重复图像的防数据泄漏流程，以更真实地评估模型的泛化能力。 |
| [^29] | [Adaptive Reliability Reserves for Residential Battery Fleets](https://arxiv.org/abs/2611.34779) | 该论文提出用县级风险阈值取代固定20%电池备用容量的自适应策略，在休斯顿2024年重放模拟中将五小时备用覆盖率从75.0%提升至86.5%并带来每户约10.2美元的年化模型收益，同时发布了结合真实电价、县级停电数据与模拟住宅的HomeFleet-ERCOT数据集，并对比了独立优化与PPO强化学习训练的表现。 |
| [^30] | [When Every Rollout Fails: Splicing Learner-Conditioned Options into Long-Horizon Agent RL](https://arxiv.org/abs/2611.34777) | 提出SPLICE方法，在线地从全失败的当前策略轨迹中拼接由学习者条件化的局部选项作为训练目标，解决了长程智能体强化学习中全失败组导致无奖励策略梯度的问题。 |
| [^31] | [When to Trust a PFN: Detecting Harmful Shift in Tabular Foundation Models](https://arxiv.org/abs/2611.34767) | 该论文提出了CALD，一种轻量级无监督监控方法，通过微调解码器最大化与测试预测的不一致性来检测表格基础模型（如TabPFN和TabICL）的有害分布偏移，在14个数据集上表现达到或超越基线，且对良性偏移的误报明显更具鲁棒性。 |
| [^32] | [One Model, Many Modes: Evolving System Prompt Ensembles for Creative Generation](https://arxiv.org/abs/2611.34760) | 该论文提出免训练框架 SEPO，通过演化系统提示词集成并结合质量约束下的贪心最大-最小选择来缓解偏好对齐大语言模型的模式坍缩，从而生成兼具高质量、新颖性与多样性的创造性内容。 |
| [^33] | [Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor Detection](https://arxiv.org/abs/2611.34758) | 该论文提出了一个基于token读出的统一评分系统来评估各类模型差异后门检测方法，并引入J透镜与J++透镜改进权重和激活分析方法，其中最强的ΔW J++方法在12个设置中的11个成功将后门相关token排入前十。 |
| [^34] | [Collect for Coverage, Replay with Evidence: Adaptive and Efficient On-Policy Distillation for Long-Horizon Agents](https://arxiv.org/abs/2611.34756) | 该论文提出CARE-OPD框架，通过学生种子教师补全和响应级置信度门控扩大历史覆盖并过滤不可靠教师信号，同时以证据自适应地回放轨迹中任意深度的关键决策，从而实现面向长程智能体的自适应且高效的前缀回放式在线策略蒸馏。 |
| [^35] | [BRIAR: Martingale Belief Smoothing on Bifurcation-Guided Reasoning Trees](https://arxiv.org/abs/2611.34746) | 该论文提出BRIAR框架，利用推理轨迹上前缀通过概率构成有界Doob鞅、其信用方差在脱轨与吸收边界处消失这一鞅几何特性，通过分叉引导的树状采样与推理，解决长链推理中困难问题早期失误脱轨和可解问题冗余验证的两大病态。 |
| [^36] | [Was the Model Given the Right Action? Auditing and Repairing Action Interfaces in Navigation World Models](https://arxiv.org/abs/2611.34739) | 该论文审计了导航世界模型（NWM）训练与推演代码之间相互作用的动作输入差异，发现无需重新训练、仅修正输入准备方式即可将视觉预测误差降低42%，并改变了模型间的性能排名，证明相互作用的输入失配必须联合检查。 |
| [^37] | [From NP-Hardness to Exact Tractability Through Sample Abundance](https://arxiv.org/abs/2611.34734) | 该论文证明了对于最坏情况下NP难的带噪声奇偶校验拟合问题，只需 $O(n/(\log n)^{1/(k-2)})$ 次来自同一来源的额外采样，就能以高概率在 $O(n\log^2 n)$ 的工作量内实现精确优化，表明样本的丰富性可以在不改变问题本身的情况下将NP难问题变为精确可解。 |
| [^38] | [Hierarchical Graph Metanetworks](https://arxiv.org/abs/2611.34726) | 该论文提出层次图元网络（Hierarchical Graph Metanetworks），利用神经网络固有的层次结构（神经元—层—块/模块）作为归纳偏置，突破了现有图元网络只能在最细粒度处理每个神经元和权重、计算成本随参数量扩展的瓶颈，从而实现高效的大规模神经网络权重空间学习。 |
| [^39] | [Bigger Model or Bigger Table? How Much Backbone a Chorus N-gram Module Replaces](https://arxiv.org/abs/2611.34725) | 该论文通过在 13.4 亿波兰语词元上训练 16M 至 349M 参数的模型，量化了 Chorus N-gram 哈希表模块对 transformer 主干网络的替代作用：当查找表与主干网络规模相当时，每个新增参数最多可替代 0.31 个主干参数，使模型能够以较小规模匹敌 1.16 至 1.34 倍大小的普通模型。 |
| [^40] | [LiDAR Intensity Reconstruction with Limited Real Data: An Architecture Benchmark under Fixed-Epoch Training](https://arxiv.org/abs/2611.34724) | 该论文在固定20轮训练协议下，对九种神经架构在真实数据稀缺、以物理仿真数据补充的激光雷达强度重建任务上进行了系统基准测试，发现UNet-V2在纯真实数据训练下表现最佳，并量化了不同真实数据比例对重建性能的影响。 |
| [^41] | [Robust Recourse under Causal Graph Ambiguity](https://arxiv.org/abs/2611.34723) | 针对因果图存在歧义的问题，提出了最坏情况补救和图加权补救两种方法，通过梯度搜索找到在多个候选因果图上均有效的低成本补救行动。 |
| [^42] | [Privileged On-Policy Distillation for Calibrated Alzheimer's Diagnosis With Missing Modalities](https://arxiv.org/abs/2611.34721) | 该论文提出一种在线策略蒸馏方法，让模型自行逐模态插补缺失数据，并由掌握特权信息（真实诊断结果和完整模态）的教师对每步插补和诊断输出进行评分，从而在模态缺失情况下实现校准良好的阿尔茨海默病诊断。 |
| [^43] | [Spec2Weights: Certifying Networks that a Language Model Writes from a Specification](https://arxiv.org/abs/2611.34711) | 该论文提出Spec2Weights方法，首次实现了对语言模型根据自然语言规范直接生成（无数据、无梯度）的神经网络在其真实运行的float32算术上的穷举式精确性证明，在15个网络中成功认证12个，且能指出网络失效的精确边界。 |
| [^44] | [From Shared Patterns to Individual Evidence: Two-Stage Reasoning for Emerging Entities in Temporal Knowledge Graphs](https://arxiv.org/abs/2611.34708) | 该论文提出INSPIRE两阶段推理框架，通过融合全目录事件记忆的共享模式预测与轻量级个体证据建模，有效解决时序知识图谱中新兴实体从零样本到少样本演化场景下的链接预测难题。 |
| [^45] | [When the Proxy Is Wrong: Mismatch-Corrected Bandits for Adaptive Data Sampling](https://arxiv.org/abs/2611.34707) | 该论文提出失配校正老虎机方法，将代理模型与目标模型的排序失建为多保真度老虎机问题，利用稀少审计中的成对残差校正代理奖励并智能分配审计预算，从而在无需假设共享偏移或排序保持的情况下改进大模型的自适应数据采样。 |
| [^46] | [Maco-RL: Scaling Agent Capabilities with Mixed-Task Reinforcement Learning](https://arxiv.org/abs/2611.34703) | Maco-RL通过基于配额的异步调度器和三缓冲区采样机制有效扩展混合任务强化学习，实现了智能体能力的稳定训练、高效计算分配与持续的领域扩展。 |
| [^47] | [Effect of class proportion shifts on classifier curvature](https://arxiv.org/abs/2611.34702) | 该论文通过测量输出层参数的损失Hessian迹来研究类别比例变化对分类器曲率的影响，发现曲率变化极小（-0.152%至+0.071%），并提出了一种基于初始类别梯度的预测方法且通过了前瞻性验证。 |
| [^48] | [Evacuation Traffic Prediction for Dynamic Transportation Networks over Multiple Hurricanes: A Generalized Graph Learning Approach](https://arxiv.org/abs/2611.34687) | 本研究利用12场历史飓风的检测器和疏散数据，提出一种广义动态图学习框架（图卷积LSTM），实现跨多场飓风的疏散交通预测并解决模型泛化难题。 |
| [^49] | [Provably Efficient Concave Utility Reinforcement Learning with Linear Function Approximation](https://arxiv.org/abs/2611.34678) | 本文提出基于模型的乐观凹效用策略优化（OCUPO）框架，利用双线性MDP的低维结构高效估计状态-动作占用度量，首次在线性函数逼近下实现了可证明高效的凹效用强化学习。 |
| [^50] | [Learning to Be Truncated: Preparing LLM Checkpoints for SVD Compression](https://arxiv.org/abs/2611.34667) | 提出LLift方法，仅需在单一压缩比率下用原模型作为教师进行一次低秩微调，即可将预训练大语言模型检查点准备好，使其在多种SVD压缩比率下直接获得准确率提升，无需针对每个比率重复微调。 |
| [^51] | [Do We Still Need Reinforcement Learning When We Can Read the World Model?](https://arxiv.org/abs/2611.34656) | 该论文提出了一种新范式：让基于大语言模型的编程智能体直接阅读可读的世界模型（环境源代码）来推理并合成控制策略，无需任务特定的强化学习训练，即可在大幅减少交互经验的情况下达到甚至超越传统强化学习方法的性能。 |
| [^52] | [A Hard Background Pairing Transfers across Classifiers and Images](https://arxiv.org/abs/2611.34645) | 该论文证明，对某些分类器而言最困难的物体-背景配对，对其他分类器和在新的图像上同样困难，表明这种困难性具有跨分类器和跨图像的迁移性。 |
| [^53] | [Distractor-Robust Reinforcement Learning via Variational Bisimulation](https://arxiv.org/abs/2611.34641) | 提出VIBES新目标，用对抗项取代像素重建并结合固定容量潜在瓶颈，使模型让潜在状态只需预测奖励与动态，从而丢弃无关干扰信息，显著提升强化学习对干扰物的鲁棒性。 |
| [^54] | [Carrying Feature Ablations Across Sparse Autoencoder Seeds](https://arxiv.org/abs/2611.34637) | 该论文提出一种方法：通过分别在两个SAE中移除匹配特征并比较编码器输出的变化，来判断特征消融能否跨随机种子移植，发现在其容差下12%至70%的匹配特征对能够成功跨种子移植。 |
| [^55] | [LatentRL-QAS: Transfer via Latent Reinforcement Learning for Quantum Architecture Search](https://arxiv.org/abs/2611.34634) | 提出LatentRL-QAS框架，利用电路变分自编码器将任意量子比特数的电路映射到统一的潜在空间中，使强化学习智能体在固定大小的状态、动作和网络下实现跨量子比特数的迁移学习，并通过对潜在空间的周期性刷新来适应更大系统。 |
| [^56] | [Predicting acute food crises and discovering actionable drivers with causal machine learning](https://arxiv.org/abs/2611.34629) | 该研究构建了包含驱动因素变量的粮食不安全数据集，开发了缺失数据鲁棒的机器学习流程以在3至9个月提前期预测急性粮食危机，并通过因果机器学习方法对比预测重要性与因果估计，识别出可干预的危机驱动因素。 |
| [^57] | [DecisionReplay: A Corpus and Offline Simulator for Ensembling Decision Models](https://arxiv.org/abs/2611.34627) | DecisionReplay 发布了 17 个商业决策模型在 35 个基准测试上的完整概率向量及成本延迟数据，使集成、路由和级联策略可在离线确定性模拟器中快速评估；实验表明精心挑选的单一模型很难被超越，而最佳路由器能以其 64% 的成本达到同等性能。 |
| [^58] | [HoMe: Toward Practical Harness Optimization of Memory Systems for AI Assistants](https://arxiv.org/abs/2611.34624) | HoMe提出了一个评估驱动的框架，通过迭代评估与编辑候选控制框架，并结合会话级记忆版本控制和执行轨迹分析来定位共享失败机制，从而自动化优化AI助手记忆系统的代码、提示词与配置设计。 |
| [^59] | [Automated Batch-Variable Selection for ComBat-Based Radiomic Harmonization](https://arxiv.org/abs/2611.34618) | 提出ParsiComBat方法，为ComBat影像组学协调自动选择最优批变量（采集参数），在去除批效应与保留特征结构之间取得平衡，并能识别真实批变量（模拟实验准确率92.9%）、在技术变异可忽略时主动放弃校正。 |
| [^60] | [Conditional Independence Graph Learning from Dependent Data Via Time-Delay Embedding](https://arxiv.org/abs/2611.34616) | 提出一种基于时滞嵌入的时域方法，将相依时间序列的条件独立图估计转化为多属性随机向量图估计问题，突破了以往频域方法仅适用于短程相依数据的限制，可同时处理短程相依和长程相依的高维平稳多元高斯时间序列。 |
| [^61] | [Learning from Crowds with Unsupervised Item Response Models](https://arxiv.org/abs/2611.34602) | 本文将众包标注建模为无监督项目反应理论（IRT）问题，证明无监督IRT模型能够可靠恢复标注者与题目的潜在特质并实现稳定的标签推断，还提出在分类器训练中利用IRT模型提供指导。 |
| [^62] | [$\mu$MPC: Microsecond-Scale Nonlinear and World Model Control from Edge to GPU via Koopman-Linearized Precompilation](https://arxiv.org/abs/2611.34597) | 该论文提出$\mu$MPC框架，通过Koopman线性化预编译将非线性模型预测控制转化为仅含动作变量的二次规划，实现了从微控制器到GPU的微秒级高速控制，同时支持控制感知的动力学学习。 |
| [^63] | [ChemoMAE: Self-Supervised Spectral Representation Learning Reveals Chemical Heterogeneity in Historical Wood](https://arxiv.org/abs/2611.34581) | 提出自监督Transformer模型ChemoMAE，利用掩码去噪学习像素级光谱表征，无需预先定义的化学标记即可从近红外高光谱图像中揭示历史木材的化学异质性与劣化模式。 |
| [^64] | [Tune for Retention, Not Just Gain: SFT Forgets Far Less than You Think](https://arxiv.org/abs/2611.34579) | 本文挑战了“SFT因其离策略特性而容易导致灾难性遗忘”的流行观点，证明通过选择更低学习率和批量大小等同时兼顾增益与能力保持的超参数，普通SFT对旧能力的保持远好于以往认知，甚至可与在线策略的SDFT方法相媲美。 |
| [^65] | [Investigating the efficacy of vortex beam speckle patterns for the recognition of atmospheric turbulence and fractional topological charge with a deep learning model](https://arxiv.org/abs/2611.34573) | 该论文提出利用卷积神经网络直接从单张散斑图像中回归涡旋光束的分数拓扑荷并识别大气湍流强度，验证了该方法在宽范围湍流条件下（D/r0为0到50）识别分数拓扑荷（精度达0.01）的有效性。 |
| [^66] | [On Utilization of Edge-AI for condition monitoring towards predictive maintenance: A use case in dairy processing using explainable AI](https://arxiv.org/abs/2611.34563) | 该研究开发了一个基于边缘AI的状态监测框架，利用低功耗IIoT传感器节点采集的振动数据并在边缘设备上处理，对乳制品加工厂的旋转阀进行状态监测，为预测性维护迈出重要一步。 |
| [^67] | [Kant-Q: A Q-Learning Algorithm for Kantian Equilibrium in Symmetric Markov Games](https://arxiv.org/abs/2611.34536) | 本文提出Kant-Q算法，通过让每个智能体学习所有智能体采取相同行动的行动配置来寻找康德均衡，继承了标准Q学习的收敛保证，并证明每个对称马尔可夫博弈都存在纯策略康德均衡，为解决混合动机博弈中纳什均衡帕累托低效的问题提供了新方案。 |
| [^68] | [AliceAI-Foundation-80B-A3B-Base](https://arxiv.org/abs/2611.34529) | Yandex 从零训练并开源了总参数 800 亿、每 token 激活约 30 亿的混合专家基座模型，它结合 Kimi Delta Attention 与全注意力并使用 Muon 优化器训练，同时提出在 MoE 输出上引入辅助损失以在预测的高学习率下稳定训练。 |
| [^69] | [Resilience Analysis Through Automated Stage Identification and Quantitative Assessment](https://arxiv.org/abs/2611.34527) | 本文提出一种与领域和威胁无关的机器学习方法，自动识别系统性能时间序列中的PARA韧性阶段，并结合事件前、中、后的性能数据计算韧性指标以实现定量韧性评估，在多个不同领域的系统数据上验证了其通用性。 |
| [^70] | [Feasibility-Aware Deep Reinforcement Learning for O-CU and O-DU Mapping in 6G O-RAN](https://arxiv.org/abs/2611.34516) | 本文提出一种可行性感知的深度强化学习方法，利用DQN和Double DQN将6G O-RAN中O-CU和O-DU的虚拟网络功能组件高效映射到虚拟机上，在保证可行性的同时显著降低传统优化方法的计算成本。 |
| [^71] | [SMOTE 25 Years of Magic: What Has Been Done, What the Evidence Shows, and Whom It Reached](https://arxiv.org/abs/2611.34511) | 在SMOTE问世25周年之际，本研究通过审计76篇论文语料库并对22,426条记录进行系统性检索，评估了SMOTE家族方法的改进内容、证据强度及其跨学科的影响力范围。 |
| [^72] | [Efficient Data Fusion in Distributionally Robust Optimization](https://arxiv.org/abs/2611.34495) | 本文提出仅通过函数型拉格朗日乘子耦合各模糊集的融合分布鲁棒优化精确重构方法，并利用仿射决策规则近似得到可分解的高效上界，从而克服了传统方法随数据源数量呈指数级增长的计算瓶颈。 |
| [^73] | [Auditing Identity and Timing Signals in Operational Logs](https://arxiv.org/abs/2611.34493) | 该论文提出身份审计与时间审计两种简单方法，可在训练事件预测模型之前量化运营日志中的身份与时间信息泄露，并发现ICU数据中常用的“向前填充”预处理会显著夸大模型性能（身份AUC从0.916升至0.993，脓毒症预测AUC虚增0.163）。 |
| [^74] | [FlyingStart: Learned Cross-Token Warm Starts for Fixed-Point Decoding in Recurrent-Depth Language Models](https://arxiv.org/abs/2611.34492) | 提出FlyingStart方法，利用上一个Token停止时的状态经闭式解拟合的线性变换来热启动下一个Token的迭代，在精度不变的前提下将递归深度语言模型的解码循环层数减少14%–19%。 |
| [^75] | [Backpropagation as Dirac-bracket dynamics of dissipative training](https://arxiv.org/abs/2611.34479) | 该论文通过 Bateman-Galley 自由度倍增将耗散的梯度训练方法（梯度流、重球法、Nesterov 加速梯度）表述为哈密顿系统，并把网络激活值视为第二类约束，证明反向传播本质上是狄拉克括号演化，从而为反向传播给出了严格的哈密顿力学几何解释。 |
| [^76] | [DiversePlace: Diversity-Seeking Curriculum Reinforcement Learning for Macro Placement](https://arxiv.org/abs/2611.34472) | 该论文提出DiversePlace，一种将质量受控的多样性作为显式目标的课程强化学习方法，通过维护多个HPWL具有竞争力且彼此显著不同的宏单元布局存档，为下游选择提供多样化的高质量方案。 |
| [^77] | [Global Convergence of Second-Order Methods for Physics-Informed Neural Networks in the Overparameterized Limit](https://arxiv.org/abs/2611.34452) | 本文首次为物理信息神经网络训练中的正则化牛顿法和Gauss-牛顿法建立了全局收敛理论，证明了在过参数化极限下训练动力学以显式速率概率收敛到确定性极限方程的解。 |
| [^78] | [Decision-Aligned Value Learning for Selective Adoption of EEG Fusion in Human-Robot Collaboration](https://arxiv.org/abs/2611.34451) | 该论文提出VECTA框架，将人机协作中EEG信息的选择性采用建模为逐样本增量价值估计问题，通过分离融合预测生成与采用决策，并借助系统增益和匹配增益来判断EEG融合何时能真正改善预测。 |
| [^79] | [vSFT: Learning to Verbally Tune Agents at Test Time](https://arxiv.org/abs/2611.34440) | 提出vSFT框架，通过超网络将监督者基于智能体失败轨迹生成的自然语言反馈转化为低秩权重更新，实现测试时的参数内演化，在ALFWorld等任务上性能超越GRPO和Reflexion等主流强化学习方法。 |
| [^80] | [One Representation to Rule Them All -- Open-Vocabulary Change Detection from Text or Visual Examples](https://arxiv.org/abs/2611.34430) | 提出免训练框架OVTSID3，在DINOv3表示空间中统一区域分组、概念识别与时间比较，支持文本或视觉示例两种方式指定变化目标，使开放词汇变化检测的平均F1较现有免训练方法提升8.9个百分点。 |
| [^81] | [A Head-Level Investigative Study of Self-Supervised Backbones for Synthetic Speech Detection](https://arxiv.org/abs/2611.34426) | 该研究首次对自监督语音骨干网络进行注意力头级别分析，发现仅少数注意力头主导合成语音检测决策，移除对抗性头部无需重新训练即可将等错误率相对降低 13%–22%，且伪造语音与真实语音之间的注意力集中度差异本身即可几乎替代完整表示实现真假语音区分。 |
| [^82] | [Decision-Focused Learning for Multistage Stochastic Programs, with Application to Power Network Repair Scheduling](https://arxiv.org/abs/2611.34419) | 该论文提出了一种面向多阶段随机规划的决策聚焦学习框架，利用神经网络预测驱动状态转移的决策得分，并通过带扰动优化器的Fenchel-Young损失进行训练，成功应用于随机故障下的电网修复调度问题。 |
| [^83] | [Recoverability policy editing for closed-loop safety in autonomous driving](https://arxiv.org/abs/2611.34403) | 该论文提出了一种状态级策略编辑技术，通过引入策略可恢复性概念，在不改变模型原有驾驶能力的前提下，保证自动驾驶在可恢复扰动下的闭环安全性，在nuPlan场景中将安全事故减少了48%。 |
| [^84] | [Any-Rate: Half-Bit Multi-Precision LLM Quantization with Coded Bit-Planes](https://arxiv.org/abs/2611.34402) | Any-Rate通过将位平面约束为8权重块的扩展汉明码字并结合逐精度联合编码，实现了从单一检查点以半比特为步长提供2–8比特的多精度LLM量化，在2比特和3比特下达到最先进的准确率。 |
| [^85] | [Strong Starts, Simple Recipes: Unified Multi-Teacher On-Policy Distillation from Multimodal Specialists](https://arxiv.org/abs/2611.34397) | 该论文提出Uni-MOPD方法，将学生模型的起点作为设计变量——先用无训练的专家模型均匀合并获得强起点，再用简短的多教师在线策略蒸馏配方进行精炼，并发现起点选择比后续训练策略对最终性能的影响更大。 |
| [^86] | [Plausible Is Not Proven: Testing Failure Explanations in Monocular Depth Estimation](https://arxiv.org/abs/2611.34393) | 提出视觉误差观测台（VEO）审计框架，通过四项独立检验来验证单目深度估计失败原因标签的可信度，发现遮挡边界标签虽有解释力但并不优于简单的深度边缘线索。 |
| [^87] | [Decoupling the clocks of learning and computing in spiking neural networks](https://arxiv.org/abs/2611.34381) | 本文提出 ChronoProp 方法，通过解耦学习与计算的时钟，利用神经元在固定时间窗内的脉冲计数和实测输入输出函数来计算权重更新，在图像分类中达到与通过时间反向传播相似的准确率，同时训练时间缩短约3倍、内存占用降低约20倍。 |
| [^88] | [Under the Influence -- of What? A Controlled Evaluation of Influence Functions](https://arxiv.org/abs/2611.34373) | 这是首个通过受控数据划分重新训练的系统评估，验证了影响函数在GPT-2类模型中识别的“有影响力”训练样本确实与输出相关，但不总是直接承载目标属性，且相关性因任务设置而异。 |
| [^89] | [Single-Database Rewards Are Hackable: Counterfactual Execution for Trustworthy Text-to-SQL RL](https://arxiv.org/abs/2611.34363) | 该论文提出一种反事实数据库奖励机制，通过在隐藏的代码生成数据库副本上执行查询来验证结果，将文本到SQL强化学习中的奖励破解率从54.1%大幅降至1.9%，且不误拒任何正确查询。 |
| [^90] | [RAWPrism: A Three-Task Dataset and a Unified Model for RAW-Domain Image Fusion](https://arxiv.org/abs/2611.34359) | 提出了覆盖多曝光融合、多焦点融合和红外-可见光融合三任务的RAWPrism数据集，并构建了首个直接在RAW域统一执行这三种融合任务的轻量级模型。 |
| [^91] | [Inverse Reinforcement Learning with State-Dependent Foresight for Human Decision-Making](https://arxiv.org/abs/2611.34351) | 该论文提出SC-AIRL（状态条件对抗逆强化学习），将人类前瞻性的有限性与状态依赖性这两种认知特性融入逆强化学习的规划器中，在行人过街和高速公路超车任务中显著提升了对人类行为的模仿效果。 |
| [^92] | [Music Similarity Estimation from Jukebox Embeddings](https://arxiv.org/abs/2611.34346) | 提出JukeSim方法，首次利用生成式音乐模型Jukebox的嵌入来估计音乐相似度，其性能与基于学习的方法相当，并优于最佳的手工设计方法。 |
| [^93] | [Scaling and Cross-Simulation Generalization of Graph Neural Networks for Cosmic-Web Classification in CAMELS](https://arxiv.org/abs/2611.34332) | 本研究系统评估了图神经网络在CAMELS多套流体动力学模拟中进行宇宙网环境分类时，随训练样本规模扩展的性能变化规律及其跨模拟泛化能力。 |
| [^94] | [Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers](https://arxiv.org/abs/2611.34331) | Puzzling-Bench是一个基于23,777道来自Puzzling Stack Exchange、带有社区验证答案的人工编写谜题（涵盖谜语、密码、逻辑推理等类别）的语言模型评估基准，并利用训练截止日期后发布的谜题切片来检测数据污染。 |
| [^95] | [Variational Principles and Machine Learning: a Numerical Analysis and PDE perspective](https://arxiv.org/abs/2611.34314) | 本文从数值分析与偏微分方程的变分视角出发，将机器学习算法视为非线性最小化问题加以研究，探讨稳定性、一致性、可逼近性和紧性等经典数值分析概念在机器学习中的新内涵。 |
| [^96] | [Interpretable identification of constitutive parameter field for three-dimensional heterogeneous and graded hyperelastic materials from full-field displacements](https://arxiv.org/abs/2611.34310) | 本文提出了一种可解释的方法，仅利用全场位移和边界反力即可辨识三维非均质与梯度超弹性材料的空间变化本构参数场，为功能梯度材料设计及生物组织表征提供了有效工具。 |
| [^97] | [RAP: Radial-Angular Random Projection for Hyperbolic Representations](https://arxiv.org/abs/2611.34298) | 本文提出径向-角向随机投影（RAP），通过精确保持径向坐标并单独处理角向误差，首次为双曲表示构建了类似 Johnson–Lindenstrauss 引理的盲式随机降维原语，填补了双曲表示学习中缺乏维度压缩工具的空白。 |
| [^98] | [LatENSO v1.0: a learned latent state of the tropical Pacific for seasonal ENSO operators and forecasting](https://arxiv.org/abs/2611.34296) | 该论文提出LatENSO v1.0系统，利用Swin-transformer自编码器将热带太平洋13个月平均海气场压缩为一个32维的学习潜在状态，并基于该状态构建XRO-lat和Floquet模态算子两种季节算子，实现ENSO的季节预报。 |
| [^99] | [Beat This! Live - Online Beat and Downbeat Tracking by Predicting the Future](https://arxiv.org/abs/2611.34277) | 该论文提出了 Beat This! Live——一个通过时间偏移标签训练、能够略微超前预测未来帧的因果模型，实现了低延迟的在线节拍与强拍跟踪，并在 GTZAN 数据集上显著超越现有最先进的在线方法。 |
| [^100] | [Scaling Looped MoE Language Models under Matched Budgets: Information Flow and Parameter Allocation](https://arxiv.org/abs/2611.34276) | 该论文通过门控内循环状态更新与跨阶段残差连接改进循环MoE模型的信息流，并在等参数、等FLOP预算下发现将单次Core深度设为总展开深度的25%-35%、同时把节省的参数重新分配给每层额外路由专家的配置，能够以零额外计算成本提升模型性能。 |
| [^101] | [LatentCP-DPO: Conformal Pessimism for DPO with Finite-Sample Guarantees](https://arxiv.org/abs/2611.34275) | 该论文提出LatentCP-DPO框架，将保形预测与悲观偏好优化相结合，为潜在相对奖励构建具有有限样本覆盖保证的不确定性集合，并通过最大化最坏情况下的偏好对数似然来学习响应策略，从而有效防止DPO在数据有限时对噪声偏好标签过拟合。 |
| [^102] | [Libtt: A Compiler and Runtime for Tenstorrent Processors](https://arxiv.org/abs/2611.34273) | Libtt 通过实现 PJRT 接口将 PyTorch 和 JAX 生态系统完整移植到 Tenstorrent 硬件，以单个共享库提供从编译器到多芯片运行时的全栈方案，并借助大量修复和编译器优化使现有面向 TPU 的代码开箱即用且高效运行。 |
| [^103] | [LLM-coupled interpretable tool wear monitoring with multi-evidence reasoning in CNC machining](https://arxiv.org/abs/2611.34270) | 该论文提出了一种两层框架，将深度学习刀具磨损分类与大语言模型多证据推理相结合，在实现高精度磨损状态识别的同时，生成可解释的自然语言报告和刀具更换建议。 |
| [^104] | [Train-Only-Constrained LM for Synthetic Clinical Trajectories: Leak-Safe Calibration and Auditable Dynamics](https://arxiv.org/abs/2611.34258) | 该论文提出一种仅用训练集校准的约束式语言模型流水线，通过分位数标记学习共现关系并用可审计的马尔可夫转移矩阵约束时间动态，在防泄漏（不使用测试集统计量）的前提下生成了分布保真度更高、隐私可检测性接近随机的合成临床轨迹。 |
| [^105] | [EXPOSURE: Explaining Sources of Uncertainty in Reinforcement Learning](https://arxiv.org/abs/2611.34248) | 提出EXPOSURE方法，利用单个感知掩码的信念网络同时进行动作选择和Shapley解释，将回报不确定性分解为特征缺失导致的观测部分与环境随机性导致的固有部分，从而更准确地解释强化学习智能体的期望回报、不确定性来源及信念不确定性。 |
| [^106] | [Change Point Detection with Generative Models](https://arxiv.org/abs/2611.34228) | 该论文提出利用生成模型对时间序列窗口进行数据增广，突破了基于窗口的变点检测中“样本量瓶颈”的根本局限，显著提升了变点检测的准确性与质量。 |
| [^107] | [The Socratic PAUSE: safeguarding scientific judgement at AI speed](https://arxiv.org/abs/2611.34222) | 论文提出“苏格拉底式暂停”机制，在关键科学决策节点设置审慎检查点并保持AI贡献的暂定性，以应对AI加速发现快于论证所导致的验证债务累积与信心超越证据的风险。 |
| [^108] | [Compute-Aware Submodular Online Learning in Open Multi-Agent Systems](https://arxiv.org/abs/2611.34211) | 本文提出了一种带查询预算的计算感知次模在线学习算法，通过采样和逆概率加权获得无偏梯度估计，并推导出可分解为KL散度跟踪误差与查询相关反馈项的1/2近似遗憾界。 |
| [^109] | [Text-Conditioned Low-Rank Adaptation of Time-Series Foundation Models for Multimodal Forecasting](https://arxiv.org/abs/2611.34206) | 提出C2LoRA-TS方法，通过将文本上下文编码为冻结时间序列基础模型的临时LoRA参数更新，实现文本条件化的参数适应，在九个领域的多模态预测任务中均优于冻结模型和静态LoRA基线。 |
| [^110] | [Cell-Type-Specific Expression Is Hard to Predict from Bulk, but It Is Still Useful](https://arxiv.org/abs/2611.34166) | 该论文通过推导闭式精度界从理论上解释了为何从bulk RNA数据预测细胞类型比例容易而预测细胞类型特异性表达困难，并证明尽管CTSE难以估计，对其进行目标性反卷积在非小细胞肺癌总生存期预测中仍具有预后价值。 |
| [^111] | [Strategic and Robust Task Allocation for Combat Objectives in Adversarial Multi-Agent Environments](https://arxiv.org/abs/2611.34158) | 本文提出一种“先预测后优化”的混合任务分配方法，通过多头模型预测战术信号并结合最小费用流求解器，在对抗性多智能体环境中实现实时、可控、可解释且鲁棒的作战目标分配。 |
| [^112] | [When High Capacity Becomes Undeployable: Pore-Filling Constraints for Multi-Objective Optimization of Direct Air Capture Sorbents](https://arxiv.org/abs/2611.34154) | 该论文创新性地将孔隙填充边界作为显式结果约束引入DAC吸附剂的多目标贝叶斯优化，并基于618个可追溯配方构建基准，揭示无约束优化会系统性提出物理上不可部署的高容量配方，而显式约束能将搜索引导至更低胺负载、更大孔隙的可部署设计。 |
| [^113] | [SOC-Net: Second-Order-Cone-Constrained Neural Networks](https://arxiv.org/abs/2611.34136) | SOC-Net是一种通过构造方式对任意输入（包括训练分布之外的输入）强制满足输入依赖二阶锥（SOC）约束的神经网络架构，无需锥求解器、迭代方案或内部可行点。 |
| [^114] | [Agent-as-a-Verifier: From Scores to Evidence for General Verification](https://arxiv.org/abs/2611.34132) | 该论文提出“验证器即智能体”系统，将角色分离的LLM智能体嵌入受形式化方法启发的声明-义务-证书协议中，使验证从仅输出分数升级为生成可审计的证据证书。 |
| [^115] | [MUSE: Multi-Task Uncertainty-Aware Self-Supervised Encoder for mmWave ISAC](https://arxiv.org/abs/2611.34122) | 提出MUSE统一框架，通过自监督transformer与掩码自编码在多个数据集上预训练，并利用证据分类头实现不确定性量化，以单一编码器取代针对各任务单独训练的多个模型，提升了毫米波人体感知的可扩展性、可迁移性和可靠性。 |
| [^116] | [Unstable Action Boundaries in Lethal Drone Agents](https://arxiv.org/abs/2611.34121) | 该论文研究了致命无人机LLM智能体行为边界的稳定性，发现部分智能体无需任何越狱即可对“人”等普通目标描述调用致命工具，且原本保守的智能体在简单的语义目标转变下会变得不安全。 |
| [^117] | [Not Every Input Needs Everything: Unifying Expert Gating, Model Cascades, and Agent Orchestration](https://arxiv.org/abs/2611.34112) | 该系统性综述首次在统一的“先检查后分配”条件计算框架下，将令牌级专家门控、查询级模型级联与任务级多智能体编排三大孤立研究领域综合起来，分析了2,289项研究以揭示跨尺度的共性失败模式与可迁移的解决方案。 |
| [^118] | [SysBio-SR: Benchmarking Symbolic Regression for Trajectory Fidelity and Mechanism Recovery in Systems Biology](https://arxiv.org/abs/2611.34106) | 本文提出SysBio-SR基准，包含198个带反应注释的生物ODE系统，首次系统评估符号回归方法在轨迹保真度与机制恢复两个维度上的表现，揭示了准确拟合系统轨迹并不等于恢复真实的生物反应机制。 |
| [^119] | [Do Machine Learning Ensembles Beat Published Quotas? Target Parameterization, Censored Baselines, and Conformal Intervals Under Admissions Reform](https://arxiv.org/abs/2611.34104) | 该研究基于越南2022-2026年本科招生面板数据，发现由于多数项目恰好按公布名额录取，结合截尾基线与保形区间的公布名额预测优于无锚定的机器学习集成方法，且最差5%项目的“幻影容量”贡献了一半以上的名额预测误差。 |
| [^120] | [AlignPre: Skill-Aligned Post-Training for Latent-Free Generative Precipitation Nowcasting](https://arxiv.org/abs/2611.34101) | AlignPre提出了一种无需潜在压缩的像素空间降水临近预报框架，其核心创新是利用预报验证得分的解析结构构建可微分目标进行技能对齐后训练（SAPT），从而直接优化业务预报技能而非仅进行分布匹配。 |
| [^121] | [Exploring the Design Factors of Pixel-space Diffusion](https://arxiv.org/abs/2611.34093) | 该论文提出EiT——一种带通道瓶颈的U形像素扩散Transformer，从架构、表示和频域三个视角系统揭示了像素空间扩散的关键设计因素，发现更早的去噪与更长的基提升阶段有助于更好的生成质量。 |
| [^122] | [Improving Tail-Augmented Speculative Drafting with Hidden State Mixing and Gradient Detach](https://arxiv.org/abs/2611.34092) | 本文提出两种仅用于训练阶段的方法——隐状态混合（HSM）和隐状态梯度分离（HSGD），在不显式模拟推理、不增加额外内存和专用注意力掩码的情况下，缓解推测解码草稿模型的训练-推理不匹配，从而改进自回归草稿生成。 |
| [^123] | [Feature Engineered Reinforcement Learning for S-pair Selection in Buchberger's Algorithm](https://arxiv.org/abs/2611.34088) | 本文提出用17维统计特征向量描述多项式，将强化学习方法从二项式理想扩展到一般多项式理想的S-对选择，在所有测试分布上显著优于基线方法，相比度数启发式减少了41–53%的算法工作量。 |
| [^124] | [Accelerated Proximal Gradient Methods for Multiobjective Optimization over Probability Measures via Hilbert-Space Lifting](https://arxiv.org/abs/2611.34073) | 本文提出基于希尔伯特空间提升的多目标Wasserstein近端梯度方法（MWPG）及其加速版本（MWAPG），通过隐式Wasserstein梯度步求解概率测度上KL散度的Pareto多目标优化问题，并提供严格的离散时间收敛保证。 |
| [^125] | [CARE-WM: Same-Horizon Self-Conditioning with Constraint-Adaptive Regularization for Latent World Models](https://arxiv.org/abs/2611.34054) | CARE-WM通过同视界自条件化训练与约束自适应的潜在正则化，在无需额外监督、独立代价模型或架构改动的情况下，同时提升潜在世界模型的多步预测精度和潜在空间与任务代价的对齐程度，从而提高规划成功率。 |
| [^126] | [Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability](https://arxiv.org/abs/2611.34051) | 本文发现GDN-2训练失败源于擦除操作中读写方向不对称导致的循环状态指数级增长，并提出通过对称擦除与衰减调节残差写入来解决该问题，同时理论上证明了状态增长的O(√T)上界并保留分块并行计算能力。 |
| [^127] | [Decentralized Clustered Federated Learning for Cellular Traffic Forecasting over LEO Constellations](https://arxiv.org/abs/2611.34049) | 该论文将去中心化联邦聚类（DFCA）扩展到低轨卫星星座场景，通过k近邻星间gossip协作与零样本头选择机制，在无地面聚合器的条件下让每颗卫星自适应地预测其下方地面小区的蜂窝流量，显著降低预测误差。 |
| [^128] | [Few-Shot Adaptation of Vision-Language Models via Agentic Model Editing](https://arxiv.org/abs/2611.34036) | 该论文提出AME框架，让LLM智能体基于少量样本和对模型的观察，自主设计并迭代优化包含输入变换、编码器编辑和分类器拟合的领域定制适配程序，实现视觉-语言模型的小样本适配。 |
| [^129] | [Post-Training Policy Replication for Strategy-Following Multi-Agent Deep Reinforcement Learning across Team Sizes](https://arxiv.org/abs/2611.34035) | 该论文提出了一种与团队规模无关的观测架构，通过将单个预训练策略复制到不同规模团队的各智能体并辅以热启动再训练，显著降低了策略遵循型多智能体强化学习的训练成本并提升了跨团队规模的可迁移性。 |
| [^130] | [Assessing the Physical Interpretability of AlphaEarth Foundation Embeddings Through Multi-Sensor Satellite Time-Series Benchmarking for Agricultural Mapping](https://arxiv.org/abs/2611.34019) | 本研究在摩洛哥农业制图任务中对AlphaEarth基础模型嵌入与多传感器卫星时间序列特征进行了基准测试，发现AEF嵌入精度（0.91）接近多时相时间序列特征的最优结果（0.93），证明其可作为传统遥感产品的高效可迁移替代方案，并具备重建生物物理信息的潜力。 |
| [^131] | [Hetero-Associative Visuomotor Memory: Soft Recall as a Base for Learned Residuals in Few-Demonstration Imitation](https://arxiv.org/abs/2611.34014) | 研究发现，在少演示模仿的检索-修正框架中，让演示记忆提供对整个记忆库进行softmax加权的平滑软召回作为修正器的基底，相比传统修正最近邻检索片段的方式可显著提升策略成功率，且该优势在稀疏、多样的真实人类演示上最为突出。 |
| [^132] | [CARROT: Competitive Adaptive Graph Clustering via Semi-Relaxed Optimal Transport](https://arxiv.org/abs/2611.34012) | 该论文提出CARROT框架，将半松弛最优传输引入深度图聚类，无需预先指定聚类数量即可自适应地确定簇数，并能有效处理簇规模高度不平衡的问题。 |
| [^133] | [The Reference Matters: Displacement Probing for Weight-Space Learning](https://arxiv.org/abs/2611.33998) | 该论文提出位移探测（DSProbe），通过探测模型相对于参考起点（如预训练权重）的位移而非原始权重，并结合学习一个需减去的共享参考成分，从而提升了基于探测的权重空间学习在预测模型属性上的效果。 |
| [^134] | [MolGPAO: Molecular Generative Pre-trained Autoregressive Octree with Speculative Decoding and Sub-Millisecond Seam Healing for 3D Molecules](https://arxiv.org/abs/2611.33985) | 提出MolGPAO——一个独立的3D八叉树自回归分子生成架构，通过6维无尺度预算嵌入、推测树解码与反捷径训练，在不依赖外部代理、强化学习或推理时MLIP弛豫的情况下，实现对大型类药物分子高效且几何稳定的3D生成。 |
| [^135] | [Quaternion-Tucker ZigZag Networks: Halving Transformer Backbone Parameters and Enabling Exact 2D Zero-Loss Continual Scaling](https://arxiv.org/abs/2611.33983) | 该论文提出QuatTucker-ZigZagGPT架构，通过四元数压缩、块对角Tucker分解、ZigZag菱形路由和GatedBounce自蒸馏技术，将Transformer骨干参数减半，同时实现宽度和深度两个维度上的精确零损失持续扩展。 |
| [^136] | [Kolpona: An Agentic Framework with Gradient-Free Neuron Programming for RTL Generation](https://arxiv.org/abs/2611.33973) | Kolpona是一个无需梯度训练的智能体框架，通过神经元编程（以闭式方式直接改写模型权重中的单个神经元）并仅在新任务测试结果有利时保留编辑，结合基于规范与仿真的无标签设计选择，显著提升开放权重LLM的Verilog生成性能，在Qwen3.5-4B上将VerilogEval v2 pass@1从31.7%提升至77.6%。 |
| [^137] | [Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder](https://arxiv.org/abs/2611.33972) | 该论文提出闭式权重空间编程（WSP）框架，让大语言模型编写程序在冻结编码器上以闭式解拟合分类头，全程无需梯度更新和权重修改，却在多个标签预算下达到甚至超过全量微调、LoRA 等梯度方法的性能。 |
| [^138] | [DLaR: Dynamic Stochasticity Control for Latent Reasoning in Large Language Models](https://arxiv.org/abs/2611.33962) | 该论文提出DLaR框架，通过在token层面基于置信度-熵融合信号自适应调节扰动强度、在训练层面感知训练进程调优随机性，解决了潜在推理中均匀随机性次优和无约束扰动破坏连贯性的问题，实现探索与稳定性的动态平衡。 |
| [^139] | [RefMix: Expanding Endpoint Proposals for Diversity-Preserving Reward Fine-Tuning](https://arxiv.org/abs/2611.33957) | RefMix 通过在早期高噪声步骤借助参考模型注入备选端点候选，打破策略内反馈循环，在保持任务奖励性能的同时显著提升生成结果的视觉多样性。 |
| [^140] | [DuoMI: Dual Mutual Information Regularization for Expert Reuse in Looped Transformers](https://arxiv.org/abs/2611.33952) | DuoMI通过轮次级边缘互信息和token级条件互信息的双重正则化，在不增加参数和改变稀疏Top-k路由的前提下，调节循环MoE Transformer中跨轮次的专家路由以实现专家参数的高效重用。 |
| [^141] | [BReD: Block Replay Dithering for Stable Low-Bit EMA Optimizer States](https://arxiv.org/abs/2611.33951) | 提出 BReD 方法，通过将确定性种子重放与分块共享抖动值相结合的块重放抖动技术，减少 EMA 优化器状态迭代量化中的舍入偏差并控制舍入方差，从而解决误差累积问题、稳定大规模训练。 |
| [^142] | [A Graphical Criterion for Linear Causal Bounds](https://arxiv.org/abs/2611.33950) | 本文针对每个混杂组件最多含两个隐藏原因的因果图，给出了判断尖锐边界能否通过求解单个线性规划得到的充分必要条件，并实现了Python包partialid以计算高效的方式获得尖锐边界。 |
| [^143] | [CondUQ: Calibrated Conditional Uncertainty Quantification Using Distribution Matching](https://arxiv.org/abs/2611.33939) | 该论文提出 condUQ 方法，通过将误差模型条件化于可观测因素（如输入质量和运行条件）进行分布匹配，在不修改甚至黑盒的现有系统之上实现准确可靠的条件不确定性量化。 |
| [^144] | [InstructTTS-OPD: On-Policy Distillation for Multi-Objective Instruction-Following TTS](https://arxiv.org/abs/2611.33937) | 该论文提出 InstructTTS-OPD，用在策略蒸馏融合多个单目标教师模型，取代单一标量奖励的强化学习，从而解决指令跟随 TTS 中多目标联合优化带来的训练不稳定与目标间跷跷板效应。 |
| [^145] | [Correlation-Aware Evidential Fusion for Trusted Multi-View Classification](https://arxiv.org/abs/2611.33900) | 提出了相关性感知证据融合框架CAFE，通过显式建模视图间的相关性，避免重复计算冗余证据、强化互补信息，从而提升可信多视图分类的性能和不确定性估计的校准质量。 |
| [^146] | [Looped Models are Fast Multi-Token Predictors](https://arxiv.org/abs/2611.33896) | 提出Floodgates自推测解码方法，在循环语言模型末尾附加掩码词块并通过低秩适应并行起草多个未来词，实现高达7.5倍的无损推理加速。 |
| [^147] | [On the Learnability of Model Reprogramming: A Reachability and Relabeling Perspective](https://arxiv.org/abs/2611.33889) | 本文从可达性与重标记的视角系统研究了模型重编程包装器假设类的有限样本可学习性，回答了逼近该类最佳预测器所需的目标样本数量以及其最优解与目标贝叶斯风险之间的差距这两个核心理论问题。 |
| [^148] | [The Gradient Difference Induced by Detaching the Denominator in QK Normalization](https://arxiv.org/abs/2611.33887) | 本文推导了QK归一化中detach分母所引起的参数梯度差异的闭式形式，证明该差异在单个token上仅改变query范数（归一化后无一阶效应），而在共享的投影矩阵 $W_Q$ 上可精确分解为径向、切向和上游三个分量。 |
| [^149] | [Random Voronoi Networks](https://arxiv.org/abs/2611.33875) | 随机Voronoi网络是一族结构化的三层MLP，通过采样的Voronoi比较实现最近邻几何的精确或近似表达，并可松弛为可微的软加权，从而将最近邻方法融入可微神经架构。 |
| [^150] | [Normalizing Flows are Capable World Models](https://arxiv.org/abs/2611.33872) | 本文提出基于归一化流的世界模型NFWM，利用可逆编码器通过精确似然最大化联合训练表示与动力学，支持潜在空间规划、基于奖励的规划以及目标条件控制。 |
| [^151] | [Mitigating Overgeneralization in RND via Spectral Target Design](https://arxiv.org/abs/2611.33850) | 本文从核理论视角证明RND新颖性分数是由目标协方差核与预测器NTK共同决定的跨核残差方差，将过度泛化归结为谱欠激励问题，并提出以带宽受控的随机傅里叶特征（RFF）目标进行谱目标设计来缓解该问题。 |
| [^152] | [ChemDeduce: Benchmarking LLMs for Unknown Chemical Identification through Interactive Simulated Experiments](https://arxiv.org/abs/2611.33848) | 该论文提出了ChemDeduce——一个将有状态模拟实验室中未知化学物质识别建模为序贯决策过程的交互式评估框架，首次系统性地测试了大语言模型通过主动设计实验、迭代推理来发现未知答案的能力。 |
| [^153] | [The Missing Root: Geometric Imagination Without Images in Large Language Models](https://arxiv.org/abs/2611.33841) | 该论文提出将包含位姿与带符号距离形状嵌入的连续物体token嵌入transformer序列，使大型语言模型无需依赖图像即可获得真正的空间几何理解能力。 |
| [^154] | [VoroPlex: A Python Framework for Multi-Threaded and Differentiable Voronoi Tessellation](https://arxiv.org/abs/2611.33829) | 该论文提出了VoroPlex，一个基于多线程Voro++的开源Python框架，不仅能并行计算二维和三维Voronoi镶嵌，还首次使镶嵌结果可微分，允许借助PyTorch对单元格面积、体积、界面面积等几何量进行基于梯度的优化。 |
| [^155] | [Learning a Whole-Body Pivot-Walking Gait for Heavy-Box Transport with a Humanoid](https://arxiv.org/abs/2611.33828) | 本文提出单一强化学习策略联合协调人形机器人的迈步、平衡控制与物体操作，实现枢轴行走搬运重箱，并在真实机器人上成功将箱子向前搬运超过1米。 |
| [^156] | [FEAST: Fair and Efficient Adaptive Sampling for Task-Free On-Policy Learning](https://arxiv.org/abs/2611.33827) | 论文提出了配额-效率定律（QEL），揭示在线策略学习中分配历史会影响学习效率，并据此引入 FEAST 框架，通过将机会分配与计算分配分离，实现无任务场景下公平而高效的自适应采样。 |
| [^157] | [On Grokking and the Dynamics of Generalization](https://arxiv.org/abs/2611.33825) | 本文扩展了grokking的信号通道理论，通过将训练耗散与评估迁移同总体风险联系起来，导出了任务可见性的定量界、迁移信号降低总体风险的充要条件，以及同一训练轨迹在不同评估分布下泛化成败的充分条件。 |
| [^158] | [Pitfalls of Dependency-Ready Decoding for Masked Diffusion Language Models](https://arxiv.org/abs/2611.33823) | 本研究揭示了掩码扩散模型中“依赖就绪”解码策略的局限性：当该策略优于基于置信度或未来效用的基线方法时，其解码顺序实际上会退化成从左到右的顺序解码。 |
| [^159] | [MetaDecomp: Adaptive Task Decomposition via Meta-Learned Procedural Memory](https://arxiv.org/abs/2611.33822) | MetaDecomp提出了一种上下文元学习框架，其外循环从成功执行中构建以抽象状态和有向无环图组织的可重用程序记忆，内循环则在测试时结合记忆中的程序片段与环境反馈，自适应地细化并修订任务分解。 |
| [^160] | [KnitNet: Learning to Search at Inference Time for Exact RNA Inverse Design](https://arxiv.org/abs/2611.33812) | KnitNet是一个推理时扩展框架，通过从验证器反馈中学习核苷酸间的全局依赖关系并自适应调整联合修复策略，实现了RNA逆向设计中精确结构匹配的高效搜索。 |
| [^161] | [QPAT: A Universal Quantum Machine Learning Framework for Quantum-Assisted Training and Classical Inference](https://arxiv.org/abs/2611.33806) | QPAT提出了一种通用量子机器学习框架，利用量子线路玻恩机（QCBM）作为可学习的生成式先验来辅助经典模型训练，实现“量子辅助训练+经典推理”的混合范式，既借助量子模型的归纳偏置克服深度学习在有限数据下的过拟合问题，又规避了NISQ时代量子推理延迟等部署瓶颈。 |
| [^162] | [InterGenEHR$^+$: An Interactive Framework for Natural Language-Conditioned Generation of Longitudinal Patient Trajectories](https://arxiv.org/abs/2611.33788) | 提出了交互式框架InterGenEHR$^+$，利用大语言模型解析临床需求，并通过生成模型基于自然语言条件合成电子健康记录和纵向患者轨迹，用于机器学习模型的场景化评估。 |
| [^163] | [Robust Adaptive Channel Estimation for RIS-Assisted mmWave Massive MIMO via a Stochastic Shannon Entropy Criterion](https://arxiv.org/abs/2611.33773) | 本文提出了一种将双窗口核香农熵准则与重加权零吸引ℓ1惩罚相结合的鲁棒稀疏自适应算法RZA-SSE，用于RIS辅助毫米波大规模MIMO系统的角度域信道估计，能够有效抵抗非高斯脉冲干扰并给出了完整的理论稳定性分析。 |
| [^164] | [RHOPE: LLM-Augmented Adaptive Experimental Design with Complex Contextual Information](https://arxiv.org/abs/2611.33767) | 该论文提出RHOPE框架，利用大语言模型将复杂上下文信息转化为代理结果，将有限批次自适应实验设计形式化为后验马尔可夫决策过程，并通过样本增强与决策前预观察代理结果两种方式提升估计精度和分配决策效果。 |
| [^165] | [Don't Waste the Population: Post-Anneal Refinement for Combinatorial Optimization](https://arxiv.org/abs/2611.33766) | 本文提出一种无需训练的确定性后处理方法 I-PRP，通过在并行退火产生的候选解之间探索成本引导的路径重连并迭代融入新改进，在不修改退火过程的前提下发掘种群多样性，从理论上保证不劣于传统单解精细化且有限收敛，并在 MaxCut、MaxClique 等组合优化基准上发现更优解。 |
| [^166] | [Understanding Double Descent through Universal Compression](https://arxiv.org/abs/2611.33759) | 该论文提出以通用压缩的全新理论视角解释深度学习中的双下降现象，将最大似然超额测试损失的上界分解为近似与偏移项、极小极大批次遗憾、先验代价和MLE与贝叶斯差距四个可解释的组成部分。 |
| [^167] | [OrbACE: Orbital and Atomic Cluster Expansion](https://arxiv.org/abs/2611.33743) | OrbACE框架通过在E(3)等变簇展开骨干网络中引入廉价的电子结构描述符，在有限训练数据下将能量误差降低至MACE的三分之一、力误差大约减半，显著提升了机器学习原子间势的精度。 |
| [^168] | [FlashProve-Next: Verifiable Inference for Production LLM APIs](https://arxiv.org/abs/2611.33741) | FlashProve-Next 在完全不修改原生推理服务的前提下，为闭源大语言模型 API 的单个响应生成零知识可验证证明，使客户端能够验证服务商是否真实运行了所宣称的模型、精度和推理预算，同时不泄露权重、隐藏提示词等机密信息。 |
| [^169] | [CovNovBench: A Benchmark for Evaluating Time Series Foundation Models under Covariate Novelty](https://arxiv.org/abs/2611.33732) | 本文提出了CovNovBench基准，首次系统性地评估时间序列基础模型在协变量新颖性（即未来处理协变量超出历史观测范围，如新定价策略或新治疗方案）情形下的预测能力。 |
| [^170] | [Understanding Mode Collapse in Diffusion Models under Classifier-Free Guidance](https://arxiv.org/abs/2611.33719) | 本文提出一个条件分布为均值位于椭球面上的高斯混合、无条件分布为近似该混合的单一高斯的分析模型，揭示了无分类器引导下模式坍缩的根源在于条件分布与无条件分布表示能力的不匹配。 |
| [^171] | [Flipping the Coins: Training Physical Simulators through Hard Switches](https://arxiv.org/abs/2611.33716) | 提出EventFlip方法，通过随机翻转近阈值硬开关并重放求解器来测量两次运行的损失差，实现对跨越离散跳变的梯度估计，从而有效训练包含硬开关的物理模拟器。 |
| [^172] | [Nepali Currency Denomination Classification Using Convolutional Neural Networks (CNNs) and MobileNetV2: An Experimental Comparison](https://arxiv.org/abs/2611.33713) | 经过超参数调优的MobileNetV2在七种尼泊尔货币面额分类任务中表现最优，实现了95%的验证准确率，优于传统CNN和采用迁移学习的MobileNetV2。 |
| [^173] | [Node Differential Privacy via Node-Stable Graph Projections](https://arxiv.org/abs/2611.33672) | 本文提出了一种近线性时间的节点稳定图投影方法，该方法可将任意图映射为有界度数的子图且对单节点变化保持稳定，从而为设计精确高效的节点差分隐私算法提供了新的核心工具。 |
| [^174] | [Verifiable but Not Predictable: What Sentence Embeddings Know About Propositional Decomposition](https://arxiv.org/abs/2611.33668) | 本文以PropSegmEnt语料库为基础，发现命题分解的两个质量控制问题处于不同表示空间：分解后可用NLI评判器高精度验证命题归属（准确率92.8%、AUC达0.977），而分解前无法通过句子嵌入的向量几何预测句子应产出的命题数量。 |
| [^175] | [Online Identification Can Replace a Floor-Plan Prior on Unseen Buildings](https://arxiv.org/abs/2611.33656) | 该论文提出一种无需目标建筑数据的暖通空调控制策略，用递归最小二乘法在线辨识一阶灰箱模型来替代平面图先验，在从未见过的建筑上性能大幅超越PID基线和ASHRAE Guideline 36。 |
| [^176] | [Self-supervised Clipped-objective Reinforcement Leaning Framework for Risk-aware Portfolio Management](https://arxiv.org/abs/2611.33652) | 本文提出自监督截断目标投资组合管理框架（SCP），通过两阶段架构融合逐笔级别的日内市场动态信息并稳定强化学习的高方差训练过程，从而实现风险感知的投资组合管理。 |
| [^177] | [PRIOR: Pre-Call Reasoning in SFT and Its Influence on Tool-Use Retention under RL](https://arxiv.org/abs/2611.33648) | 该研究通过控制token预算下的工具调用深度实验，揭示SFT数据中工具调用前的推理量是决定强化学习后工具使用行为能否保持的关键因素——调用前推理越少，RL训练中工具使用越容易崩溃。 |
| [^178] | [The Danger of Double Precision: Working Precision as a Hidden Constraint in Gaussian Process Maximum Likelihood](https://arxiv.org/abs/2611.33647) | 高斯过程超参数的最大似然拟合受浮点工作精度的隐式约束——只有当有效数字宽度超过协方差条件数（约70位）时拟合才会收敛，因此双精度下的拟合实际上得到的是一个不同的模型。 |
| [^179] | [Pretraining for Forecasting in Relational Databases](https://arxiv.org/abs/2611.33645) | 提出查询条件关系预训练（QCRP）自监督框架，利用结构化时间查询语法生成多样化预测目标，使标准关系深度学习骨干网络成为数据库上的通用预测器，其冻结表示可经轻量级预测器或零样本接口复用于各类下游预测任务。 |
| [^180] | [Physics-Informed Learning for Multiplayer Differential Games](https://arxiv.org/abs/2611.33644) | 该论文提出一种在策略迭代框架内结合消失粘性退火调度、利用物理信息神经网络逼近N人对抗性微分博弈粘性解的方法，并通过可计算残差给出了显式的ε-纳什保证及有限时间收敛性证明。 |
| [^181] | [How Much Effort Does an AI Code Reviewer Need? Evidence from Claude 5.5 Models on Plausible-but-Wrong Patches](https://arxiv.org/abs/2611.33639) | 通过对Claude 5.5系列模型审查90个SWE-bench补丁的预注册实验，本研究发现更高的推理努力程度能显著减少AI代码审查者对似是而非错误补丁的误接受，且对Opus而言中等努力程度能以低20%的成本达到与高等努力程度相当的效果。 |
| [^182] | [Interpreting Speckle Effects Through Neuron Interventions](https://arxiv.org/abs/2611.33632) | 研究发现，SAR预训练模型中的斑点信息并非由斑点选择性神经元直接编码，而是主要由第一个MLP层的64个神经元承载，修复这些神经元可恢复44.1%的相关性损失。 |
| [^183] | [State Space Attention: Learning Context-Dependent Memory Retention](https://arxiv.org/abs/2611.33625) | 提出状态空间注意力（SSA），通过可学习的评分器自适应地决定保留哪些历史表示以维持有界内存，并配合“恢复”式监督样本训练技术，使模型能够依据上下文动态选择长上下文推理所需的记忆内容。 |
| [^184] | [Learning from Natural Images Untangles Number Sense](https://arxiv.org/abs/2611.33615) | 该论文提出一个几何框架，发现尽管未训练与预训练网络的单元级数量选择性相似，但自然图像学习能将数量与比例的表征解缠分离，证明视觉学习在数量选择性的自发产生之外仍有重要贡献。 |
| [^185] | [Improved Sample Complexity of Robust Reinforcement Learning with KL Uncertainty Sets](https://arxiv.org/abs/2611.33592) | 本文针对基于KL散度不确定性集的鲁棒强化学习，给出了更紧致且完备的样本复杂度上下界，填补了以往结果中的差距，并将下界覆盖范围扩展到完整的不确定性水平区间。 |
| [^186] | [Amortizing Exact Orthogonality: Drift-Triggered Restoration for Orthogonal Fine-Tuning](https://arxiv.org/abs/2611.33589) | 该论文提出 RTOFT 方法，用一阶更新 W(I+S) 取代逐步构造正交变换，仅在漂移 ‖S‖₂ 超过容差 τ 时才通过极分解恢复精确正交性，从而以漂移触发的方式摊销正交微调的计算开销。 |
| [^187] | [Symmetry-Aware Residual Learning of Few-Body Molecular Wave Functions](https://arxiv.org/abs/2611.33576) | 该论文提出一种物理信息神经变分方法，将分子波函数分解为编码旋转对称性与渐近衰减的解析分量和神经残差分量，从而在不依赖离散网格的情况下有效降低少体极性分子系统高维本征值问题的求解难度。 |
| [^188] | [Trigger-Invariant Contrastive Unlearning for Backdoor Removal](https://arxiv.org/abs/2611.33571) | 提出TICU框架，通过构建干净与插入触发器的反事实样本对并强制预测与表示对触发器插入保持不变性，有效移除NLP模型中的文本后门，同时借助干净数据监督损失和参数漂移约束维持干净任务性能。 |
| [^189] | [DiffER$^2$: Diffusion Ensembles for Retrosynthesis Prediction with SMILES Adapted Particle Guidance](https://arxiv.org/abs/2611.33557) | 本文提出将粒子引导适配到SMILES序列生成的扩散模型集成方法DiffER²，显著提升了逆合成预测中生成独特分子的数量和top-k准确率。 |
| [^190] | [FokasNets: Completing missing boundary data in Physics-Informed Neural Networks and Operators via the Global Relation](https://arxiv.org/abs/2611.33549) | FokasNets将PINNs/PINOs与Fokas方法的全局关系相结合，在训练过程中自动补全缺失的边界条件，从而更高效、更准确地求解混合边界问题。 |
| [^191] | [RankCast: Phase-Aware Rank Balancing in MoE Serving](https://arxiv.org/abs/2611.33542) | RankCast是一个面向MoE推理服务的阶段感知负载均衡系统，针对预填充阶段计算受限（由秩间最大token数决定延迟）与解码阶段内存受限（由秩间最大激活专家数决定延迟）的不同瓶颈，在统一框架下通过为每个token保留Top-p专家并智能分配，降低了EP秩间的负载不均衡，从而减少推理延迟。 |
| [^192] | [Adversarial Bandit Convex Optimization with $d^{3/2}\sqrt{n}$ Regret](https://arxiv.org/abs/2611.33531) | 本文提出一种带高斯探索的在线牛顿步算法，将对抗性赌博机凸优化的遗憾上界从 $\widetilde O(d^{5/2}\sqrt n)$ 改进为 $\widetilde O(d^{3/2}\sqrt n)$，匹配了此前被猜想的最优速率，并将极小极大遗憾确定到 $d^{1/6}$ 因子以内。 |
| [^193] | [Casper: A Projection-Based Neurosymbolic Layer for Scalable & Guaranteed Constraint Satisfaction](https://arxiv.org/abs/2611.33527) | Casper 是一种基于投影的神经符号层，通过将对约束可行域的闭式欧几里得投影嵌入单次前向传播，同时实现了约束满足保证、推理时的可扩展性、完全自动化以及完全可微性。 |
| [^194] | [Deletion You Can Check: Auditing LLM Unlearning](https://arxiv.org/abs/2611.33525) | 该论文指出，业界常用的“成员推理得分接近随机（ROC-AUC ≈ 0.5）即代表遗忘成功”这一审计标准，依赖于极少被验证的参照群体假设，并通过 2×2 对照实验（结合精确重训练金标准与事后诊断）证明基准参照群体的不匹配会导致对大模型删除/遗忘效果的错误判断，从而提出了一套更可靠的遗忘审计方法。 |
| [^195] | [A Perturbation-Driven Framework for Sensitivity Analysis of Dimensionality Reduction](https://arxiv.org/abs/2611.33523) | 本文提出一种模型无关的扰动驱动敏感性评估框架，将扰动视为对合理数据变化的受控探针，通过量化结构保持的变化与视觉诊断来评估降维方法对输入微小变化的鲁棒性。 |
| [^196] | [Evaluation and Restoration of Safety in Reasoning-Tuned LLMs](https://arxiv.org/abs/2611.33515) | 本文揭示了一个反直觉的发现——增强推理能力会在所有模型家族中显著提高大语言模型的不安全输出率，并提出了一种将进化策略与后训练防护栏相结合的新方法（ESPG），以有效恢复模型的安全性。 |
| [^197] | [Model Predictive Control of a Multirotor UAV with a Cable-Suspended Payload: A Comparison of Shifted and Transformer-Based Warm Starts](https://arxiv.org/abs/2611.33514) | 本文针对携带缆绳吊挂载荷的多旋翼无人机的模型预测控制，提出用基于Transformer的暖启动替代传统移位初始化来加速序列凸规划优化器，数值实验表明Transformer初始化能减少计算时间。 |
| [^198] | [Learning to Reason in Real-Time](https://arxiv.org/abs/2611.33513) | 该论文提出了实时推理训练（RRT）框架及三项标准技术——使用实时数据、解耦行动与观察、时间维度课程学习，使智能体学会根据时间成本权衡快速行动与深入推理，从而在实时环境中大幅超越传统同步训练方法。 |
| [^199] | [Data-free continuous-time physics-informed neural networks for vortex shedding: a consistent pressure across time windows](https://arxiv.org/abs/2611.33489) | 该研究首次在完全无数据且无时间离散化的条件下，通过解析无散度初始条件打破对称性，在相继时间窗口上训练连续时间PINN实现了Re=100圆柱绕流的涡旋脱落，并将窗口交界处压力不连续归因于不可压缩Navier-Stokes方程指标为2的微分代数结构对一致初始条件的要求。 |
| [^200] | [The Geometry Is the Memory: Task-Density Routing for Continual LoRA](https://arxiv.org/abs/2611.33487) | CONTLORA利用冻结表示空间中的高斯类别模型实现任务密度路由来选择LoRA适配器，并通过累积特征矩精确恢复类别平衡岭分类器，在不保留任何图像或特征向量的情况下达到与存储样本方法几乎相同的持续学习性能。 |
| [^201] | [On the Impact of Extra Information in Bandit Learning: More is Not Always Merrier](https://arxiv.org/abs/2611.33479) | 该研究揭示了一个反直觉现象：在随机老虎机学习中，额外信息可能使UCB算法的遗憾变差而非改善，并提出了一种能为广泛类别老虎机算法可证明地保证“非退化性”的元算法。 |
| [^202] | [Memory as the Backbone: A Dual-Store Associative Memory Architecture for Visual Recognition](https://arxiv.org/abs/2611.33475) | 提出了双存储联想记忆架构DuSAM，以工作记忆与持久语义记忆的记忆操作替代自注意力机制，构建出一种新型视觉主干网络，在ImageNet-1K分类、ADE20K分割和COCO检测上取得与同规模模型相当的性能。 |
| [^203] | [Differentially Private Policy Optimization in Linear MDPs with Adversarial Rewards](https://arxiv.org/abs/2611.33469) | 本文针对具有未知转移和自适应对抗性奖励的线性MDP，提出了一种结合批量化PPO与乐观策略评估的差分隐私策略优化算法，在联合差分隐私下同时保护轨迹与奖励数据，并实现了保留 $\widetilde{O}\left(K^{3/4}\right)$ 主导项的高概率遗憾界。 |
| [^204] | [Learning from Recurrence: Internalizing Depth-Induced Preferences in Looped Language Models](https://arxiv.org/abs/2611.33468) | 本文提出“偏好回写”方法，利用循环语言模型在更大深度下自身生成的预测来训练共享块（无需参考答案），从而将深层计算带来的偏好内化到单遍推理中，同时提升单遍与多遍推理的准确率。 |
| [^205] | [SplatWAM: Splatting World Action Models for Real-Time 3D Imagination](https://arxiv.org/abs/2611.33467) | SplatWAM是一种直接在3D高斯基元上学习动作条件动力学的生成式世界模型，通过分词器与动力学模型实现单次前向传播的3D状态预测，可从任意视角实时渲染，速度比视频世界模型快12倍且精度更高。 |
| [^206] | [PAC Learning to Sample: On Computational Aspects of Language Generation](https://arxiv.org/abs/2611.33464) | 该论文将PAC学习框架与极限语言生成模型相结合，首次为语言生成任务提供了关于样本复杂度和时间复杂度的定量保证。 |
| [^207] | [Fast Muscle-Driven Whole-Body Motion Tracking with a Differentiable World Model](https://arxiv.org/abs/2611.33460) | 该论文提出一种基于可微分世界模型的快速肌肉驱动全身运动跟踪方法，通过四项训练改进将 MS-Human-700 的训练时间从 3.9 小时缩短至 6.3 分钟，实现 37 倍加速。 |
| [^208] | [A Realistic End-to-End Runtime Crossover Estimation for Quantum-Enhanced Training of Large-Scale Machine Learning Models](https://arxiv.org/abs/2611.33454) | 该论文对用量子线性求解器训练线性化神经网络进行了涵盖数据访问、计算与读出的端到端现实资源估算，发现量子方法相对于经典方法的优势交叉点仅在问题规模极大、计算时间长且输出要求严苛时才会出现。 |
| [^209] | [HumblePRM: Improving VLM Abstention in Ambiguous Situations through Process Reward Modeling](https://arxiv.org/abs/2611.33452) | 提出轻量级过程奖励模型HumblePRM，将多模态推理分为感知与校准两阶段独立评分，通过奖励证据与置信度的一致性而非仅奖励结果正确性，提升视觉语言模型在模糊情境下的弃权能力。 |
| [^210] | [Centered Kernel Distance: Dimension-Free Error Bounds for Comparing Neural Representations](https://arxiv.org/abs/2611.33447) | 本文提出中心核距离（CKD）这一神经表征相似性度量，证明其存在无偏估计量且估计误差以 1/M + 1/N 的速度衰减，同时收敛速率与实验条件空间维度无关，从而克服了表征相似性估计中的维度灾难。 |
| [^211] | [ViCARD: Visual Credit Assignment with Reward-Gated Distillation for Multimodal Reasoning](https://arxiv.org/abs/2611.33437) | ViCARD通过奖励门控蒸馏，将逐token的视觉监督与经过验证的正确性奖励相结合，同时解决了基于结果的强化学习和视觉自蒸馏在多模态推理中的互补性失败模式。 |
| [^212] | [Scalpel and Brush: Pruning and Targeted Weight Editing of a Small Language Model by a Single LLM Agent](https://arxiv.org/abs/2611.33426) | 该论文通过严格的对照实验首次系统地证明，单个LLM智能体能够通过稀疏权重编辑（针对性修复、结构化剪枝与层移除）改进小语言模型，仅修改0.0084%的权重即可在常微分方程任务上使BLEU提升24-32%乃至2.5倍，并排除了随机编辑等因素的干扰。 |
| [^213] | [Yet Another Concept Drift Benchmark? A Principled Framework for Evaluating Drift Detectors](https://arxiv.org/abs/2611.33424) | 该论文提出了一个基于漂移可观测性和警报语义显式形式化的原则性概念漂移检测基准框架，通过引入有限检测窗口、重复警报和标准化综合评分指标，实现了检测准确性、及时性与稳定性的清晰分离，并通过对10个检测器在9种学习器配置和16个数据集上的大规模压力测试验证了框架的有效性。 |
| [^214] | [Temporal Selective Exploration for Reinforcement Learning-Guided Continuous-Discrete Flow Matching in 3D Molecular Design](https://arxiv.org/abs/2611.33417) | 该论文提出了一种带时间选择性探索的强化学习引导连续-离散流匹配框架，通过仅在关键时间步选择性注入随机性以减少冗余探索噪声，并平衡多目标优化中的更新不均问题，从而实现高效的三维从头分子设计。 |
| [^215] | [A Semantics-First Approach to Multimodal Video Captioning with Frozen Encoders and Stabilized Reinforcement Learning](https://arxiv.org/abs/2611.33409) | 提出AVQFormer架构，在27亿参数LLM上通过固定长度语义对齐模块与稳定化强化学习（ASCST）实现高效多模态视频描述，并揭示全局音频表征与视觉帧时序冲突导致的“时间对齐摩擦”现象。 |
| [^216] | [An Adversarial Multi-Agent Pursuit Game for Locomotor Agility Pretraining](https://arxiv.org/abs/2611.33407) | 该论文提出通过在物理模拟器中让神经网络玩多智能体追逐游戏进行竞争性自我博弈，利用稀疏反馈使新的运动生成与控制策略得以涌现，从而实现人形机器人运动敏捷性的预训练。 |
| [^217] | [Gaussian priors from nonlinear PDEs with the semiclassical approximation](https://arxiv.org/abs/2611.33400) | 本文提出利用半经典近似，通过在稳定解附近展开非线性PDE的变分能量或积分平方残差来构造高斯先验，从而将贝叶斯不确定性量化的概率数值方法拓展到非线性偏微分方程领域。 |
| [^218] | [Sequentially Acquiring Concept Knowledge to Guide Continual Learning](https://arxiv.org/abs/2611.33399) | 该论文提出SACK——一种即插即用的持续学习技术，通过根据新样本与模型当前知识的语义关系对其进行评分和排序，从而指导持续学习过程。 |
| [^219] | [Best-of-$N$ Test-Time Steering Can Lead to Reward Hacking in Vision-Language-Action Models](https://arxiv.org/abs/2611.33398) | 本文揭示了当验证器不准确时，对VLA模型采用best-of-$N$测试时引导会随着N的增大引发奖励黑客行为，即被选中的动作候选利用了验证器的预测误差而非真正提升任务性能。 |
| [^220] | [Federated Learning with Uncertainty Quantification on Pseudo Label Exchange](https://arxiv.org/abs/2611.33395) | 本文提出FedSON两阶段框架，通过交换本地神经算子生成伪标签辅助数据，解决了联邦随机算子学习中客户端数据分布异构（尤其是输入函数支撑互不相交）导致的性能下降问题，并实现了不确定性量化。 |
| [^221] | [Confidence scores for gated automation of large language models](https://arxiv.org/abs/2611.33394) | 该论文提出利用监督学习在冻结LLM的最终层隐藏状态上拟合线性置信度分数头（并为闭源模型训练仅文本的元模型），以预测LLM答案的正确性，从而实现高置信度回答的自动化处理和不确定回答的人工审查分流。 |
| [^222] | [Can a Short Video Clip Verify a Model Fix?](https://arxiv.org/abs/2611.33393) | 研究表明，短视频片段不能可靠地验证视频模型的修复：仅凭片段通过就判定问题已解决会造成误判，而要求片段通过又会遗漏近四分之一的真实修复。 |
| [^223] | [Probabilistic Machine Learning for Total Kinetic Energy Distributions in Neutron-Induced Fission](https://arxiv.org/abs/2611.33379) | 该论文利用高斯过程回归和（贝叶斯）混合密度网络等概率机器学习模型，系统评估了其重现中子诱发裂变总动能分布宽度（尤其在高入射中子能量下）以及在训练范围之外外推的能力。 |
| [^224] | [Evaluating Ensemble Methods for Algorithmic Stability Under Data Perturbations](https://arxiv.org/abs/2611.33375) | 本文揭示了集成方法对分类器稳定性的双面效应：子装袋可显著提升不稳定分类器的稳定性，但会破坏本已稳定的分类器，其有限装袋引入的蒙特卡洛噪声可通过膨胀argmax方法缓解。 |
| [^225] | [Discovering Explicit Magnetic Core-Loss Equations via Symbolic Sparse Identification](https://arxiv.org/abs/2611.33364) | 本文提出符号稀疏辨识（SSI）框架，无需预设函数形式即可直接从实测数据中发现显式、可解释且高精度的磁芯损耗方程，弥合了经验模型与机器学习黑箱方法之间的差距。 |
| [^226] | [The Marauder's Map: B\'{e}zier Manifolds Reveal Hidden Surfaces for Model Merging and Ensembling](https://arxiv.org/abs/2611.33353) | 该论文提出 Nimbus，通过学习连接 N 个不同任务专家模型的高维贝塞尔流形，将模式连通性扩展为“灵活简并”的类比，为模型合并与集成揭示隐藏的多任务解曲面。 |
| [^227] | [Understanding Failure Patterns in LLM-Based Mathematical Optimization Modeling](https://arxiv.org/abs/2611.33351) | 该研究将LLM优化建模过程分解为数学推理、求解器使用和领域知识三个组成部分，通过构建针对性数据集分析失败模式，发现即便组合这些能力，可行但不正确的解仍是主要的失败类型。 |
| [^228] | [FlowersBench: A FluidFlower-Inspired Dataset for Benchmarking Carbon Storage Surrogates Across Geological Structures](https://arxiv.org/abs/2611.33335) | 提出了FlowersBench基准数据集，包含跨越244种地质结构几何形状的5,000个二维油藏规模CO₂存储模拟，填补了测试机器学习碳存储代理模型对未见地质结构泛化能力的空白。 |
| [^229] | [A Hierarchy of Risk Certificates for Representation-Based Prediction](https://arxiv.org/abs/2611.33333) | 该论文提出了一个针对基于表示的预测的层级式风险证书框架，证明期望块分数构成最优类风险的非递减下界并在紧致空间上收敛于该风险，从而为模型精度提供统计与经验证相结合的置信界。 |
| [^230] | [Online Prediction System for Proactive Autoscaling](https://arxiv.org/abs/2611.33332) | 提出了一种两阶段在线预测系统，通过季节性分解结合轻量级线性模型的残差修正，并借助每个预测周期的即时重训练，为大规模分布式系统的主动式自动扩缩容提供高精度、低延迟的短期流量预测。 |
| [^231] | [Do Not Let the Prior Heal the Anomaly: Selective Anchoring for Detection from Incomplete Measurements](https://arxiv.org/abs/2611.33331) | 该论文提出“选择性锚定”方法，在使用生成式先验从不完整测量中重建医学影像时，保护可靠测量的图像分量不被先验抹除病灶，并提供与所学先验无关的无偏估计与名义区间覆盖保证，具体实现为扩散/流采样器的包装器及快速的AnchoredFM模型。 |
| [^232] | [Sparse Adaptation for In-Context Learning on Data Streams](https://arxiv.org/abs/2611.33325) | 本文提出SAIL框架，通过重用键值缓存、结合漂移检测与在线微聚类的稀疏适应机制，使表格基础模型能够在动态演变的物联网数据流上高效进行上下文学习。 |
| [^233] | [What Must LLM Agents Communicate? From Information Loss to Communication Failures](https://arxiv.org/abs/2611.33318) | 本文提出基于信息论与通信复杂度的分析框架，指出LLM智能体有效通信需满足信息充分性、相关证据选择和接收信息利用三个条件，并证明局部概率报告造成的信息损失与性能下降高度吻合，而查询优先的交互仅在证据依赖私有查询时有效、且可能被紧凑的一次性表示抵消。 |
| [^234] | [What Does RL Post-Training Teach Text-to-Image Models? A Skill-Wise Analysis of Flow-GRPO](https://arxiv.org/abs/2611.33316) | 本文通过逐技能测量pass@k发现，Flow-GRPO后训练对文生图模型的提升大多是“锐化”（放大基础模型本已具备的能力），只有在空间关系上才是真正“解锁”了新能力，因此基准分数的提升在很大程度上夸大了模型真正学到的东西。 |
| [^235] | [Large-scale Model Merging via Recovery-aware Task Weighting](https://arxiv.org/abs/2611.33315) | 该论文提出RaTW模型合并框架，通过为专家能力损失严重的任务分配更大权重来缓解大规模模型合并中的任务间不平衡问题，并构建了包含100个任务的CLIPMerge-100基准用于大规模模型合并评估。 |
| [^236] | [Building a Unified Inline Code Editing Model for GitHub Copilot](https://arxiv.org/abs/2611.33314) | 本文为GitHub Copilot提出统一行内代码编辑模型，通过锚定差异补丁的单一输出空间一次性预测光标补全、邻近编辑与远距离编辑三种类型，支持单次响应包含多个编辑，避免了原系统多次顺序调用不同模型的开销。 |
| [^237] | [Change Partners at Every Layer: Balanced Random Regrouping for Model Merging](https://arxiv.org/abs/2611.33308) | 该论文提出在每一层对专家进行均衡随机重新分组的模型合并策略，作为完全合并与保留多个模型之间的折中方案，以缓解合并大量专家带来的准确率下降。 |
| [^238] | [Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift](https://arxiv.org/abs/2611.33302) | 本研究通过在随机层次模型上预训练并用受限规则子集微调transformer，发现语言模型微调中的遗忘具有两个时间尺度——表面行为被快速抑制，而支撑该行为的潜在表征则以缓慢漂移的方式逐渐丢失。 |
| [^239] | [JetDiffusion: Pushing the Speed Limits of Video Generation with NVFP4 Sparse Attention](https://arxiv.org/abs/2611.33268) | JetDiffusion通过可训练的视频稀疏量化注意力（VSQA）结合NVFP4量化、核融合与步数蒸馏，并借助分阶段训练方案稳定适应三项加速技术，实现了注意力8-29倍加速和端到端273倍加速的视频生成系统。 |
| [^240] | [Same Tokenizer, Better Tokens: Choosing Image Tokens by MAP Inference](https://arxiv.org/abs/2611.33261) | 该论文提出RecaP方法，通过MAP推断在冻结分词器上同时平衡重建保真度与预训练自回归先验下的序列可预测性，从而为编码器特征选择比独立最近邻更优的图像码字。 |
| [^241] | [Fast and Scalable Abstract Interpretation for Deep Neural Networks](https://arxiv.org/abs/2611.33259) | Scepter框架通过结构化语义将DNN抽象解释器的结构计算与数值求值分离，预先编译生成可复用的张量程序，在不损失精度的情况下实现快速且可扩展的神经网络性质认证。 |
| [^242] | [One Token, Many Lessons: Sliding-Window Attention within Gated Delta Networks](https://arxiv.org/abs/2611.33257) | 提出滑窗门控DeltaNet（SWGDN），从测试时回归视角研究状态更新中键值学习对的构建方式，用滑窗注意力替代短卷积实现内容依赖的局部查询聚合，以提升线性注意力的状态学习效果。 |
| [^243] | [MonoQPM: Splitting Features into Concepts for Inherent Interpretability and Predictive Performance](https://arxiv.org/abs/2611.33252) | 本文提出 MonoQPM，通过将 QPM 模型的多义特征解耦为单义概念以实现内在可解释性，并基于干扰引起的激活误差提出实用的多义性度量方法，从而显著提升共形预测的效率并获得更紧凑的预测集。 |
| [^244] | [FERRETS: A Scalable Symbolic Transform for Time Series Classification](https://arxiv.org/abs/2611.33249) | FERRETS是一种基于WEASEL 2的无标签字典变换方法，通过将随机长度和扩张率的窗口映射为精确八位符号词，在达到与最准确字典方法相当的分类精度的同时，速度提升超过百倍，成为测试中速度最快的时间序列分类方法。 |
| [^245] | [Typical Angina, Reversed: A Long-Standing Documentation Error in the UCI Heart Disease Data](https://arxiv.org/abs/2611.33248) | UCI心脏病数据集的胸痛变量编码与其文档说明完全相反——“无症状”实际代表典型心绞痛患者，“典型心绞痛”实际代表无胸痛患者，这是一个通过复现原始1986年研究和多站点数据交叉验证所证实的长期存在的标注错误。 |
| [^246] | [Beyond the Efficiency-Efficacy Dilemma: A Critical Analysis of Transferability in Adversarial Malware Detection](https://arxiv.org/abs/2611.33243) | 该论文提出对抗人工智能引擎（AAIE）自动化安全压力测试平台，揭示单步FGSM攻击在恶意软件检测迁移攻击中比迭代式PGD更有效（成功率52.9%对45.2%），并依据新兴AI治理标准提出了包含可迁移性得分、逃逸率和韧性指数的攻击评估指标体系。 |
| [^247] | [Learning the Platform, Not the Learner: Sequencing Shortcuts in the Offline Evaluation of Educational Recommender Systems and LLM Rankers](https://arxiv.org/abs/2611.33240) | 该论文发现在教育推荐系统的离线评估中，仅依赖平台行为规律、完全不使用学习者信息的简单结构化预测器就能达到最佳个性化推荐器和LLM排序器81%到100%的准确率，揭示了现有离线评估基准主要在衡量平台序列行为而非真正的学习者建模能力。 |
| [^248] | [Sequential Testing on Complex and High Dimensional Data Streams Leveraging Unsupervised Model Errors](https://arxiv.org/abs/2611.33238) | 该论文提出利用仅在正常数据上训练的无监督模型误差作为检验统计量，为复杂高维数据流设计了无需分布假设的序贯假设检验与变点检测方法，解决了仅有变化前数据的信息不对称难题。 |
| [^249] | [Resource-Efficient Multimodal Fusion for Early Rice Blast Forecasting in Sustainable Agriculture](https://arxiv.org/abs/2611.33237) | 该论文提出了一种由注意力位置编码Transformer与CNN嵌入GRU组成的轻量级多模态融合框架，能够在计算与能源受限条件下高效建模气象和遥感数据的谱-时序依赖关系，实现稻瘟病的早期预测。 |
| [^250] | [KiRA: Kronecker Input-Gated Rank Adaptation for Parameter-Efficient Fine-Tuning of Pretrained Models](https://arxiv.org/abs/2611.33236) | KiRA将微调更新表示为冻结随机矩阵与输入门控可训练因子的Kronecker积，在不增加可训练参数的前提下突破低秩PEFT方法的秩上限，实现参数高效的输入自适应微调。 |
| [^251] | [How Many Samples Is RL Worth? Best-of-N versus RL Post-Training for Text-to-Image Models](https://arxiv.org/abs/2611.33232) | 该论文提出“兑换率”这一概念来量化RL后训练与Best-of-N采样的等效样本数，发现在大多数不含位置要求的提示词上，基础模型配合现成VLM选择器的Best-of-16采样即可达到与RL后训练图像相当的效果，人工筛选最多3张基础图像也能获得类似收益。 |
| [^252] | [Effect-Driven Skill Abstractions for Offline Reinforcement Learning](https://arxiv.org/abs/2611.33229) | 该论文提出一种效果驱动的动作块量化方法，通过优化下一状态预测使离散技能依据其对状态的影响来区分，并可与常见离线强化学习方法结合，提升长时程目标条件任务的表现。 |
| [^253] | [Soft Decentralized Expected SARSA: Equilibrium Selection via Gradual Adaptation](https://arxiv.org/abs/2611.33228) | 该论文提出软去中心化期望SARSA（SDES）算法，通过渐进更新Q值而非直接切换至最佳响应，不仅证明了收敛性，还首次揭示了软更新可作为去中心化多智能体学习中的均衡选择机制，通过调节学习速率引导系统走向合作或竞争均衡。 |
| [^254] | [Real Package, Wrong Controller: Static Auditing of Vendor--Protocol Mismatches in LLM-Generated PLC Code](https://arxiv.org/abs/2611.33227) | 论文提出了Proteus-OT，一个基于版本固定绑定图谱的无标签静态验证器，用于检测大语言模型生成的PLC代码中导入包与声明工业控制器目标之间的供应商-协议不匹配，发现883个生成脚本中有20.4%存在此类导入级别不兼容问题。 |
| [^255] | [A Causal Energy Prior for Learning Directed Acyclic Graphs](https://arxiv.org/abs/2611.33225) | 提出了局部-全局能量先验（LGE），通过结合局部结构差异（结构汉明距离）与因果差异（结构干预距离）构建DAG上的吉布斯分布，使贝叶斯DAG学习能够编码干预沿路径传播的因果信息，并量化了数据在先验部分错误时对先验的修正能力。 |
| [^256] | [Adaptive attention routing multimodal expert graph neural network for drug-drug interaction prediction](https://arxiv.org/abs/2611.33200) | 提出自适应注意力路由多模态专家图神经网络（AARME-GNN），通过特征自适应校准模块与带熵约束的多模态权重路由器，解决药物相互作用预测中多模态特征异构、融合不适应药物对及模态信息被稀释的问题。 |
| [^257] | [Extraneous Cognitive Load in Large Language Models](https://arxiv.org/abs/2611.33197) | 该论文提出了首个系统地将信息呈现方式（外在认知负荷）与任务难度分离开来的受控基准CogLoadBench，并设计了基于模型激活值、在提示阶段即可计算的外在负荷分数（ELS），用于估计和诊断大语言模型的外在认知负荷。 |
| [^258] | [Dynamics of Collective Diversity in Human-AI Co-Creation for Creative Tasks](https://arxiv.org/abs/2611.33192) | 该论文通过大规模众包研究揭示了人机共创的动态规律：AI代理虽能提升创意作品质量并降低用户投入，但随着AI采用规模扩大，会导致集体多样性不断下降。 |
| [^259] | [One Neural Operator for Re-planning Low-Thrust Cislunar Transfers after Thrust Outages, at the Original or a Later Arrival](https://arxiv.org/abs/2611.33188) | 该论文训练了一个神经算子，能够在电推进任务发生推力中断后即时重新规划低推力地月转移轨迹（支持原定到达或最多延后10天到达），其控制误差比传统快速重规划基准低约一个数量级，并能以0.997的AUC准确预测中断的可恢复性。 |
| [^260] | [Semantic Density Does Not Predict Agent-Memory Collapse](https://arxiv.org/abs/2611.33184) | 本文通过大规模预注册实验发现，包括平均成对余弦相似度在内的记忆库统计指标均无法可靠预测智能体任务准确率的早期下降，即语义密度不能作为智能体记忆崩溃的有效预警信号。 |
| [^261] | [REI: Learning Ideation from Repository Evolution](https://arxiv.org/abs/2611.33183) | 本文提出REI框架，通过从代码仓库的演化历史中学习可复用的创意策略，将领域技术经验注入模型参数，以解决AI自动化科学迭代中创意构思的瓶颈，并首次揭示了质量驱动优化中的语义多样性坍塌问题。 |
| [^262] | [Neighbor Crowding Predicts Feature Stability in Sparse Autoencoders](https://arxiv.org/abs/2611.33182) | 该研究发现在TopK稀疏自编码器中，特征的跨种子稳定性仅凭单次训练得到的六个低成本统计量即可高度预测，其中字典的几何性质“邻域拥挤度”是最强预测因子，从而无需昂贵的多种子训练便能识别稳定特征。 |
| [^263] | [Deep Learning-Based Mesoscopic Emission Modeling](https://arxiv.org/abs/2611.33179) | 本研究提出基于GRU的深度学习模型DEP，通过学习由真实GNSS轨迹与微观排放模型MOBSim耦合生成的超百万条合成出行数据，直接预测路段级CO₂和NOx排放，在精度上媲美混合模型的同时实现了计算高效的介观尺度排放建模，并成功集成于法兰西岛地区的MATSim模拟流程。 |
| [^264] | [Geophysics-informed invertible operator network for Bayesian geophysical inversion](https://arxiv.org/abs/2611.33178) | 提出一种将可逆神经网络嵌入深度算子学习框架的地球物理信息引导可逆算子网络（GI-ION），实现了计算高效且物理一致的贝叶斯地球物理反演。 |
| [^265] | [Feature-Based Explainability of Transformer Scoring Models: A Fidelity-First Framework for Automated Writing Assessment](https://arxiv.org/abs/2611.33175) | 本研究提出一种“保真度优先”的事后解释框架，通过61个可解释的语言与写作质量特征，比较四种解释方法对DeBERTa写作评分模型预测的复现能力，发现随机森林代理模型保真度最高，从而在确保解释忠实于模型预测之后再解读特征重要性，提升了自动写作评估的透明度。 |
| [^266] | [The Symmetry Budget: How Much Group Structure Should a Network Architecture Assume?](https://arxiv.org/abs/2611.33169) | 该论文提出“对称性预算”概念，在一个精确可解模型中以闭式形式推导出有限样本超额风险，证明当数据仅具有近似对称性时，网络架构中的中间对称性绑定程度永远不会严格最优。 |
| [^267] | [Temporal Exposure to Byzantine Vulnerability Under Intermittent Participation in Federated Learning](https://arxiv.org/abs/2611.33168) | 本文提出动态参与暴露（DPE）概念，揭示联邦学习中的间歇性参与会将拜占庭鲁棒性从静态的群体属性转变为随机的时间性属性，并推导了每轮暴露概率及其多轮累积效应。 |
| [^268] | [Code availability is not enough: schemas for documenting and extending data analysis](https://arxiv.org/abs/2611.33167) | 该论文提出一种用自然语言记录数据分析中各项决策的架构，使大型语言模型能够据此生成并验证分析脚本，还能将单一分析自动扩展为多元分析。 |
| [^269] | [Quantum Machine Learning for Signature Forgery Detection](https://arxiv.org/abs/2611.33163) | 该论文提出了一种将冻结的 EfficientNetB0 特征提取器与量子兼容主成分投影及最近邻决策相结合的混合量子机器学习流水线，用于离线手写签名伪造检测，达到 92.62% 的准确率。 |
| [^270] | [Attributing Agentic Failure in Reasoning Vision-Language Models with Interactive 2D Environments](https://arxiv.org/abs/2611.33162) | 本文提出了专为失败归因而非排名设计的交互式迷宫环境 MultiNet v2.0-Gridworld，通过可独立操控的机制组合与基于广度优先搜索的精确最优步数评分，将视觉语言模型智能体的不同失败原因（误读环境、不解组件机制、操作顺序错误、无法纠错）区分开来。 |
| [^271] | [OmniAct: One Policy, One Million Embodiments](https://arxiv.org/abs/2611.33158) | OmniAct通过以机器人在具体场景中的运动学图（包含关节位置、运动轴及与视觉观测的连接关系）为条件，并用流匹配策略预测各节点的相对关节运动，实现了单一策略即可控制具有不同自由度和运动学配置的多种机器人具身形态。 |
| [^272] | [Reinforcement Learning Can Be Recovered From Its Own Rollouts](https://arxiv.org/abs/2611.33102) | 该论文发现，仅通过对存档的RL采样轨迹按其生成顺序进行最大似然训练、且无需区分响应的对错，就能从同一基础模型完全恢复出在线强化学习的性能，表明RL所获得的知识实质上蕴含在采样数据本身之中。 |
| [^273] | [Shifter Quantization: Compact Representations for Native Integer Computation](https://arxiv.org/abs/2611.33080) | 提出移位器量化（SQ）方法，通过追加和移位共享比特重构原生宽度整数操作数，使GPU整数矩阵运算能覆盖完整点积，在2比特精度下实现llama.cpp预填充吞吐量2.71-3.77倍的提升并降低KL散度。 |
| [^274] | [LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training](https://arxiv.org/abs/2611.33004) | LEGOSpec提出了一个可组合的推测性草稿设计与训练框架，将草稿模型分解为上下文、骨干、提议和目标四个组件，并通过系统化的对照实验揭示了不同工作负载下各组件的作用、相互作用及最优组合规律。 |
| [^275] | [Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard](https://arxiv.org/abs/2611.32595) | 本文证明了具有局部求和聚合和ReLU激活的消息传递神经网络（MPNN）的度数受限可满足性判定问题是NEXPTIME难的，且即使度数上界固定为3，该难度结果仍然成立。 |
| [^276] | [ASTERISM: Do Multimodal LLMs Actually Look at All the Images?](https://arxiv.org/abs/2611.32316) | 该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。 |
| [^277] | [A Fused Gaussian Process Bandit Framework for Multi-Fidelity Bayesian Optimization](https://arxiv.org/abs/2611.32092) | 该论文将多保真度贝叶斯优化问题创新性地表述为融合高斯过程老虎机问题，并开发了具有理论性能保证的FGP-UCB算法，从而更准确地建模保真度层级间的复杂依赖关系。 |
| [^278] | [Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios](https://arxiv.org/abs/2611.31249) | Actionwave 是首个覆盖重症监护、手术室和急诊科等多种医疗场景的以波形为中心的临床预测系统性基准，通过统一的决策时点预测接口、严格的评估协议，系统考察了信号时长、预警时间和可用输入等条件对模型表现的影响。 |
| [^279] | [Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL](https://arxiv.org/abs/2611.30702) | 在基于LLM模拟用户的共情强化学习中，寻求认同用户的占比每增加10个百分点，策略对有过错发帖人的认可率就上升3–4个百分点，表明策略会隐蔽地降低对所有人的认可门槛，而其辨别过错的能力并未下降。 |
| [^280] | [Physics-Constrained Neural ODEs for Learning Traffic Dynamics and Trajectories from Road Entry/Exit Times](https://arxiv.org/abs/2611.30577) | 该论文提出一种物理约束的神经常微分方程方法，仅凭稀疏的路段进出时间与领航车位置数据，即可学习物理上合理的交通跟驰动力学并重建车辆轨迹。 |
| [^281] | [Wavelet determinantal samplers: Optimal approximation rates adapting to regularity](https://arxiv.org/abs/2611.29992) | 该论文构造了基于小波的行列式点过程采样器，在流形与几何图上对线性统计量实现了 n^{1-2s/m} 阶的方差缩减，且该逼近速率能够自适应于观测量的 Hölder/Sobolev 正则性。 |
| [^282] | [Inferring Experimental Objectives in Microscopy Experiments from Decision Traces](https://arxiv.org/abs/2611.29814) | 本文提出从显微镜操作员的决策历史中学习奖励函数以推断其实验目标，并比较了多种从操作员对测量位置的选择偏好中重建奖励函数的算法框架。 |
| [^283] | [Click and Count: Multimodal Prototype Adaptation for Class-Agnostic Object Counting](https://arxiv.org/abs/2611.28321) | 该论文提出“点击与计数”多模态原型自适应框架，通过冻结分割模型将单次点击扩展为精确目标掩码，并在迭代原型细化中逐步融合文本描述，从而去除了边界框标注需求，实现类无关目标计数。 |
| [^284] | [Toward a digital twin of the ICU patient: Forecasting and treatment response in a generative event model](https://arxiv.org/abs/2611.27028) | 研究者训练了7900万参数的生成式事件模型TWICE，无需任务特定训练即可通过模拟50条患者未来轨迹，在19项ICU临床结局预测上达到与144个监督模型最佳配置相当的性能，是实现ICU患者数字孪生的重要一步。 |
| [^285] | [Search From the Start: Search-Native Pretraining for Language Models](https://arxiv.org/abs/2611.26809) | 提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。 |
| [^286] | [SparseHPS: A Unified Model for Human Pose and Shape across Depth Cameras and LiDARs](https://arxiv.org/abs/2611.26730) | 提出首个传感器无关的统一模型SparseHPS，仅需一套权重即可从深度相机或激光雷达采集的人体点云中恢复SMPL-X三维人体姿态、形状与全局平移，同时适用于单帧和视频流输入。 |
| [^287] | [Distributional Deep Hedging](https://arxiv.org/abs/2611.26652) | 本文提出分布式深度对冲方法，通过学习对冲损益分布的分位数，使风险偏好可在训练后作为权重灵活应用，从而将风险敏感对冲问题简化为一系列风险中性问题，避免了为每个新风险偏好重新训练策略的需求。 |
| [^288] | [Replay, Don't Resample: Off-Policy Consolidation of Long-Horizon RL Experts](https://arxiv.org/abs/2611.24059) | 提出多专家回放策略蒸馏（MRPD），通过按专家的学习顺序回放其RL训练日志（而非代价高昂的重新采样）来整合多个长时程强化学习专家，从而大幅降低整合成本并缓解训练与推理的不匹配。 |
| [^289] | [Escaping Latent Traps by Remembering: Training-Free Episodic Kernel Repulsion for World-Model Planning](https://arxiv.org/abs/2611.22470) | 提出一种无需训练的情景核排斥代价项，利用智能体已访问潜在状态的新颖性门控记忆惩罚候选计划，帮助世界模型规划摆脱局部极小值“潜在陷阱”，在两个JEPA世界模型和四个基准上将规划成功率显著提升。 |
| [^290] | [Improving fine-grained expressivity of graph neural networks](https://arxiv.org/abs/2611.20020) | 本论文通过刻画几何消息传递网络在稀疏图上的表达能力、利用表示理论分析多重置换等变线性层、研究保持包含关系的置换不变集合表示，并依次提出EGenNet、MASNet和FSW-GNN三种新架构，系统性地提升了图神经网络的细粒度表达能力与稳定性。 |
| [^291] | [Enrichment-Oriented Virtual Screening of Perovskite Solar Cells through Reinforcement Learning](https://arxiv.org/abs/2611.18043) | 该论文提出一个以冻结TabPFN代理模型引导的强化学习框架，将钙钛矿太阳能电池配方生成建模为富集导向的虚拟筛选，使前十名候选配方的平均PCE从19.68%提升至23.74%，其中PPO和RPPO策略在前100名候选名单中获得了最高的邻域命中率。 |
| [^292] | [Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost](https://arxiv.org/abs/2611.17658) | 该论文揭示了模仿策略会将机器人自身状态作为“本体感知锚点”来折扣含噪感知读数，其权重服从 α=V/(V+s²) 的预测定律，且锚点方差远小于线性模型暗示值，一次测量即可精确预测策略在不同任务和噪声水平下对感知噪声的敏感度。 |
| [^293] | [Where Is Uncertainty Observed? A Mechanistic Understanding of Uncertainty Quantification in LLMs](https://arxiv.org/abs/2611.15735) | 该论文基于机制可解释性提出了观测不确定性的四种透镜框架（结构、表征、分布、语义），并揭示了观测点决定不确定性量化估计器的信息获取能力与恢复效果这一关键机制。 |
| [^294] | [Whether untrained networks beat trained ones at V1 depends on a normalisation convention: evidence from two fMRI datasets](https://arxiv.org/abs/2611.15533) | 通过对THINGS和NSD两个fMRI数据集的系统分析，本研究表明未训练卷积网络在V1建模上的所谓优势并非本质性的，而是由归一化约定和是否去除共有响应轮廓等分析选择所驱动——在预设定的岭回归读出和标准归一化下，训练网络始终表现更优。 |
| [^295] | [A Cardiorespiratory Foundation Model Pretrained on Ten Physiological Signals for Perioperative and Critical Care](https://arxiv.org/abs/2611.15004) | CARMEN是一个在十种心肺生理波形上无标签预训练的基础模型，通过LSCNorm机制保留绝对信号水平，能灵活接受任意信号组合，并在围手术期与重症监护的19项任务中超越了监督模型和其他基础模型。 |
| [^296] | [Out-of-the-Box Tabular Foundation Models for Bayesian Optimization in Materials Discovery](https://arxiv.org/abs/2611.12491) | 该论文证明 TabPFN 和 TabICL 等表格基础模型可以作为开箱即用的代理模型，在材料发现的贝叶斯优化中媲美高斯过程等传统方法，无需核函数选择或任务特定训练。 |
| [^297] | [Ostinato: Low-Latency CPU-GPU MoE Inference for Local Agents](https://arxiv.org/abs/2611.11533) | 提出Ostinato引擎，通过在每次前向传播中原位复用一个静态的跨层GPU池，实现CPU与GPU协同的混合专家模型推理，为本地智能体提供低延迟支持。 |
| [^298] | [Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information](https://arxiv.org/abs/2611.08469) | 本文在具有已知真实答案的小型随机网络中精确刻画了整合信息 φ 等机制级量何时可被识别、连续且可由有限数据认证，并证明输入输出行为完全相同的网络可携带截然不同的整合信息（0.28 比特 vs 0 比特），但只需 4-53 次内部记录试验即可将其区分。 |
| [^299] | [ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models](https://arxiv.org/abs/2611.06732) | 该论文提出ConceptARC at Scale，一个能够大规模生成数千个针对特定概念的ARC风格任务的生成器，用于系统研究大型推理模型能否在概念相关的推理任务上形成可泛化的抽象能力。 |
| [^300] | [Generalizable Equivariant Diffusion Models for Non-Abelian Lattice Gauge Theory: Analytical Foundations and Numerical Scaling Tests](https://arxiv.org/abs/2611.03530) | 本文为非阿贝尔格点规范理论中的规范等变扩散模型建立了严格的分析框架，包括群流形上逆时随机微分方程的推导、去噪损失的规范不变性论证，以及关于晶格体积和耦合常数标度行为的系统性数值验证。 |
| [^301] | [Profile-aware diagnostics of discarded parameter-Hessian mass](https://arxiv.org/abs/2611.01335) | 提出一种轮廓感知的诊断框架，通过在不同分组规模和组内曲率轮廓参考下测量参数Hessian的块外平方占比，来量化块对角曲率模型所丢弃的曲率质量，并评估候选修正的实际效果。 |
| [^302] | [InterQ-RL: Interaction-Aware Correction of Execution Mismatch in Quantized Agentic RL](https://arxiv.org/abs/2610.35303) | InterQ-RL 针对量化智能体强化学习中残留的 rollout-学习器执行不匹配问题，提出通过对到达状态的策略历史施加有界执行前缀校正来修复对齐后仍存在的不匹配。 |
| [^303] | [Best Policy Tracking in Reinforcement Learning](https://arxiv.org/abs/2610.34501) | 该论文通过实证研究揭示了强化学习训练中性能非单调波动对策略选择的影响，并引入事后诊断指标来评估常用的策略选择标准能否可靠地追踪到训练过程中的最佳策略。 |
| [^304] | [Gaussian Approximation for Composition and Subsampling of Differentially Private Algorithms](https://arxiv.org/abs/2610.33333) | 该论文证明只要子采样机制组合中每一步的隐私损失在除小概率情形外都很小，其组合的隐私损失分布就近似服从具有相同均值和方差的高斯分布，从而为DP-SGD的隐私分析提供了无需对数因子损失或噪声尺度限制条件的更优界限。 |
| [^305] | [Compatible Likelihoods for Flow Matching on Manifolds](https://arxiv.org/abs/2610.31981) | 该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。 |
| [^306] | [Conditional Flow Models Convert Epistemic Uncertainty into Apparent Aleatoric Variance](https://arxiv.org/abs/2610.29172) | 该论文发现条件流模型生成的样本方差会系统性地大于真实条件分布的方差，这种额外的方差本质上是模型将拟合真实条件分布时的认知不确定性转化为生成样本中表观的偶然方差。 |
| [^307] | [STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites](https://arxiv.org/abs/2610.28709) | 本文提出基于集合Transformer的STAR-XPC模型，可在1-3毫秒内快速准确估算聚合物复合材料的X射线透射比，替代耗时超过5小时的蒙特卡罗模拟。 |
| [^308] | [Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds](https://arxiv.org/abs/2610.27022) | 该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。 |
| [^309] | [Federated Community Detection in Bipartite Networks](https://arxiv.org/abs/2610.25997) | 提出FLICKER与BiFLICKER方法，在原始数据不可共享的联邦环境下实现了二部网络的谱估计与社区检测，并提供了非渐近收敛与误聚类率的理论保证。 |
| [^310] | [Explaining Chess and Go in Natural Language](https://arxiv.org/abs/2610.23166) | 本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。 |
| [^311] | [L2GSSL: Label-Efficient Local-to-Global Self-Supervised Learning for Volumetric Healthcare Data](https://arxiv.org/abs/2610.21356) | 提出L2GSSL局部到全局自监督学习框架，通过将预训练分解为切片内和切片间表示空间并利用停止梯度隔离目标，在仅使用10%标注的肺结节恶性分类任务上达到0.9229 AUROC，大幅超越体积级对比方法，接近全监督基线性能。 |
| [^312] | [Behavioral Compositionality of Sparse Autoencoder Representations](https://arxiv.org/abs/2610.21261) | 论文提出SAE可加性残差（SAR）来度量稀疏自编码器特征在行为层面的组合性，发现学习到的SAE坐标比随机旋转具有更强的一阶可加性，但仍存在不可忽略的高阶交互项。 |
| [^313] | [Representation Redistribution in Sparse Autoencoders](https://arxiv.org/abs/2610.21240) | 该论文提出再分配敏感度分数（RSS）来识别在提示变换下以一致方向重复出现的大幅特征响应，并在涵盖八个语言模型的27个预训练SAE上验证了表征再分配的可复现性，同时将重复的表征变化与下游读出贡献及决策失败精确解耦。 |
| [^314] | [Not All Change Is Damage: Update-Matched Unlearning for Large Language Models](https://arxiv.org/abs/2610.21230) | 该论文提出“参照时序假设”，证明在大语言模型遗忘任务中，采用更新匹配的参照比更新前的参照能更好地保留模型学习新任务的可塑性，从而有效区分对已删知识的选择性抵抗与整体可塑性丧失。 |
| [^315] | [Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds](https://arxiv.org/abs/2610.20007) | 本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。 |
| [^316] | [Recurrent In-Context Learning for Tabular Foundation Models](https://arxiv.org/abs/2610.14998) | 提出Tab-Loop方法，用4个参数共享的块循环应用来替代表格基础模型中参数不共享的多层ICL块，在匹配计算量的前提下将参数量减少62%且性能不降。 |
| [^317] | [A Geometric Perspective on Reward Function Updates in Inverse Reinforcement Learning](https://arxiv.org/abs/2610.14041) | 本文从占用度量空间的几何视角重新审视逆强化学习中的奖励更新，揭示朝向最优占用度量的e-测地线几何结构与奖励参数空间中对偶梯度的直接联系，并据此提出用占用轨迹上的平均Fisher信息对对偶梯度进行预条件处理的方法。 |
| [^318] | [Amortized Counterfactual Prediction with Graph Neural Networks](https://arxiv.org/abs/2610.13724) | 提出反事实图神经过程（CFGNP），利用图神经网络端到端地直接从观测数据回答反事实查询，无需先估计因果图，从而避免因果图估计误差向下游反事实预测的传播。 |
| [^319] | [Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative](https://arxiv.org/abs/2610.09984) | 本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。 |
| [^320] | [SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models](https://arxiv.org/abs/2610.09335) | SearchWorld提出了一种将显式空间记忆与价值引导想象相结合的循环状态空间世界模型，通过前瞻性想象推演解决部分可观测性难题，实现无人机在城市环境中的高效目标搜索。 |
| [^321] | [Frozen Factor or Spectral Band? In LoRA, the Factor Matters More at Low Rank](https://arxiv.org/abs/2610.06621) | 本文通过2x2交叉实验设计发现，在低秩（1-8）少步数设置下，LoRA变体冻结哪个因子（输入因子A显著优于输出因子B）比其选择的奇异方向带（主导或次主导）对性能影响更大。 |
| [^322] | [SLIM: Simplex-Lattice Interpolation Merging](https://arxiv.org/abs/2610.01037) | SLIM 利用经典混料设计，仅通过对单个专家和等权重配对的最少评估，在系数单纯形上构建总体性能的二次代理模型，无需额外基准评估即可优化大语言模型合并系数，在有限评估预算下实现准确预测和有竞争力的合并性能。 |
| [^323] | [CompMat-Bench: Benchmarking AI Agents for Computational Materials Science](https://arxiv.org/abs/2610.00636) | 提出了CompMat-Bench基准，通过预先复现昂贵模拟的输入与结果作为固定规则的评分依据，实现对计算材料科学AI智能体的低成本高效评估，涵盖94个源自已发表研究的任务以及四种评估条件。 |
| [^324] | [Policy Iteration Is Not Strongly Polynomial for Deterministic Markov Decision Processes: The Price of Algorithmic Anarchy](https://arxiv.org/abs/2609.40147) | 本文证明Howard策略迭代在确定性折扣马尔可夫决策过程上具有指数级迭代下界因而并非强多项式，而带Dantzig规则的单纯形法在此类问题上却是强多项式的，这一差距揭示了“算法无政府状态”的代价。 |
| [^325] | [GazeFlow: From Human Gaze Behavior to Generative Egocentric Gaze Prediction](https://arxiv.org/abs/2609.38519) | 提出GazeFlow框架，利用条件流匹配将第一人称注视建模为受自上而下任务信息与自下而上视觉显著性共同约束的时间注视位置联合分布，从而生成合理的注视轨迹并达到最先进的预测性能。 |
| [^326] | [VirusCascade: Hijacking Collaborative Reflection in LLM-Powered Recommender Agents](https://arxiv.org/abs/2609.38270) | 本文揭示了基于大语言模型的智能体推荐系统中的新型安全漏洞，提出VirusCascade攻击方法，通过“反思洗白”将注入单个智能体的对抗性证据合理化，并借助协同反思机制在多智能体间传播放大，实现对整个推荐系统的劫持。 |
| [^327] | [RNA Design via Conditioned Flow Matching and Finite-Policy Reinforcement Learning](https://arxiv.org/abs/2609.36885) | 该论文提出结合结构条件化狄利克雷流匹配与保持碱基配对的有限策略强化学习的两阶段RNA逆折叠框架，通过显式建模自然RNA进化中的协调序列变异并利用热力学反馈优化，在多个基准测试中取得领先性能（Rfam-27上Pass@1达85.19%）。 |
| [^328] | [Conformal Prediction and Conditional Coverage for Tabular Foundation Models](https://arxiv.org/abs/2609.34887) | 该论文提出C-USIM方法，一种无需额外训练或推理、可处理多模态预测的轻量级最高预测密度分割共形预测技术，为表格基础模型提供有限样本边际覆盖有效性保证，并显著改善条件覆盖和分组覆盖精度。 |
| [^329] | [Concurrent Split Learning Through Stable Client Clustering](https://arxiv.org/abs/2609.29395) | 本文提出全局聚类并行分割学习（GCPSL），通过将客户端稳定分配到固定集群并并发执行多个分割学习工作负载，在不增大单个批量的前提下提升客户端直接数据参与率，使CIFAR-10达到85%验证准确率的时间从串行执行的约19分钟缩短至约6分钟。 |
| [^330] | [Even Sharper Bounds for Transductive Learning and Its Applications](https://arxiv.org/abs/2609.28459) | 本文提出直推学习的新型局部化复杂度方法STLC，去除了以往直推学习中多余的对数置信度因子，并在可实现设定下达到了与标准归纳学习相同的 $\cO\{\dVC\log(me/\dVC)/m\}$ 速率。 |
| [^331] | [Relative Discharge Stage (RDS) Classification: A Practical Indicator of Battery Discharge Progress](https://arxiv.org/abs/2609.27986) | 本文提出相对放电阶段（RDS）这一电池管理指标，将剩余放电状态划分为五个可解释等级，并通过结合SOC估计与轻量级时序学习的物理信息分类框架，在无需未来电流信息的情况下实现对电池放电进度的实用化评估。 |
| [^332] | [Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models](https://arxiv.org/abs/2609.27373) | 该论文发现循环语言模型的注意力路由在早期循环步骤即趋于稳定，据此提出无需训练的WISE推理方法：早期用全局注意力发现稀疏工作集，后续步骤重用该支撑集，在保持推理质量的同时大幅减少重复的全局注意力计算。 |
| [^333] | [LSTM-UT and Recurrent-Depth Transformers on Cellular Automata](https://arxiv.org/abs/2609.19521) | 该论文提出带有有界门控记忆的LSTM-UT模型，并通过Rule 30元胞自动机和延迟回忆实验证明，仅依赖当前隐藏状态的Block Universal Transformer在深度外推上优于依赖扩展注意力缓存的CoTFormer，而LSTM-UT进一步同时提升了深度外推与延迟回忆能力。 |
| [^334] | [Tabby: An Open Pretraining Recipe for Time Series Foundation Models](https://arxiv.org/abs/2609.13956) | Tabby是一个完全开放的1.45亿参数长上下文概率时间序列基础模型，通过融合真实与因果合成数据、渐进式收敛调度和深度分位数监督，以单模型支持预测、分类和异常检测多种任务。 |
| [^335] | [Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs](https://arxiv.org/abs/2609.11347) | 该研究提出将知识图谱“具体化”为事实节点并通过六个元关系的固定词汇表连接，使普通GNN（如现成的GAT）无需任何专用架构即可实现零样本归纳链接预测，性能媲美预训练基础模型ULTRA。 |
| [^336] | [Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor](https://arxiv.org/abs/2609.02145) | 该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。 |
| [^337] | [LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training](https://arxiv.org/abs/2608.24845) | 该论文介绍了LAION-BVD，一个包含1000万小时视频的大规模开放多模态数据集，通过合成描述和场景帧提取，显著提升了视频、音频和图像-文本任务的预训练性能。 |
| [^338] | [CrevasseSeg: A Label-Efficient UAV Crevasse Segmentation Framework](https://arxiv.org/abs/2608.15790) | 本文提出CrevasseSeg框架，通过自监督学习在仅有24块标注数据的情况下，实现无人机影像中冰川裂缝的高效分割，显著降低标注成本。 |
| [^339] | [ED-DiT: Physics-Guided Diffusion Pretraining for Transferable Molecular Representations from Electron Density](https://arxiv.org/abs/2608.03260) | 提出了 ED-DiT——一种物理引导的扩散 Transformer，通过在不同噪声水平下重建被破坏与掩码的对数电子密度场，并结合电子数一致性约束进行自监督预训练，学习可跨多种电子结构相关任务迁移的分子表示。 |
| [^340] | [A Graph Signal Processing Perspective on Numerical Sequence Representations in LLM In-Context Learning](https://arxiv.org/abs/2608.03015) | 本文提出用图信号处理视角研究大语言模型上下文学习中数值信息的内部组织，将注意力视为token加权图、隐藏状态视为图上信号，发现随着上下文变长，表示按输入动力学复杂度分化：简单输入对应更强的全局连通性和更平滑、谱更集中的隐藏状态信号。 |
| [^341] | [Improved Quantum Algorithms for Reinforcement Learning Under a Generative Model](https://arxiv.org/abs/2608.02826) | 本文将标准值迭代与量子均值估计、量子最大值查找等量子子程序相结合，提出了求解有限时域和无限时域折扣MDP近似最优策略的新量子算法，其查询复杂度优于以往工作并接近已有的量子下界。 |
| [^342] | [QSTAR: Quantum Selective Transfer with Adaptive Routing](https://arxiv.org/abs/2607.21411) | 本文提出QSTAR框架，通过自适应路由机制保留高置信度的经典预测、仅将低置信度样本交给量子分支处理，从而让量子迁移学习中量子分支的作用更加明确。 |
| [^343] | [A zero-one law for one-shot system identification](https://arxiv.org/abs/2607.15832) | 该论文证明了一次性系统辨识的零一律：当字典项线性无关时，几乎所有从非退化高斯测度采样的随机输入都能唯一确定模型系数，否则没有任何输入可以做到，从而将单次实验辨识问题归结为退化输入的判定问题。 |
| [^344] | [Toward Energy-Efficient and Low-Power Arrhythmia Detection for Wearable Devices](https://arxiv.org/abs/2607.14747) | 本文通过在深度学习心律失常检测模型及其硬件架构中采用数据精度降低和近似乘法等近似技术，在保持可接受分类精度的同时显著降低了功耗与能耗，使检测系统能够高效部署于可穿戴设备。 |
| [^345] | [Near-Optimal Learning of Gaussian Sobolev Operators](https://arxiv.org/abs/2607.11921) | 本文提出了 Hermite-PCA 逼近算法，一种采用主成分分析与加权最小二乘的完全数据驱动且计算高效的方法，能以近最优的样本复杂度学习高斯索博列夫算子。 |
| [^346] | [How Much is Left? LLMs Linearly Encode Their Remaining Output Length](https://arxiv.org/abs/2607.05316) | 本文发现大语言模型在隐藏状态中线性编码了剩余输出长度信息，仅需一个简单的线性探针即可准确预测生成回答还剩多少token，且当出现“Wait”等撤回token时预测长度会显著跳升。 |
| [^347] | [Unsupervised Detection of Underground Tunnels in Ground-Penetrating Radar Using Depth-Restricted Reconstruction Scoring](https://arxiv.org/abs/2607.04882) | 该论文提出了一种无需任何隧道样本的无监督检测方案，利用卷积自编码器学习场地正常地下状况，并通过深度受限的重建误差评分，在实地探地雷达数据中实现了近乎完美的地下隧道检测性能（AUC达0.994）。 |
| [^348] | [Safe Inference-Time Alignment via Lagrangian Reward Augmentation](https://arxiv.org/abs/2607.02781) | 提出LARA框架，通过拉格朗日对偶将安全约束转化为由小型校准集估计的对偶变量所定义的增强奖励，实现可即插即用于现有推理时对齐方法的无需权重更新的安全对齐。 |
| [^349] | [Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?](https://arxiv.org/abs/2606.31126) | 本文评估了TabPFN和TabICL等表格基础模型能否作为小样本场景下的数据高效预测器泛化到生物分子性质预测任务，发现在ProteinGym上TabPFN3-ESMC600M与领先的监督方法具有竞争力并取得最低MSE，但Kermut在Spearman相关性上仍更强，且在更难的划分设置下性能明显下降。 |
| [^350] | [Anti-Collapse Dynamics and the Emergence of Multi-Time-Scale Learning in Recurrent Neural Networks](https://arxiv.org/abs/2606.29519) | 本文揭示了递归神经网络中遗忘速率（指数或幂律）并非由架构预设，而是由状态与参数动力学的耦合自发涌现，形成坍缩或反坍缩两种状态，从而决定长程学习的可行性。 |
| [^351] | [The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching](https://arxiv.org/abs/2606.27510) | 本文从因果中介分析出发重新推导了激活修补的估计量，揭示其中隐藏的交互效应会导致因果重要性依赖于其他组件状态的模型组件被错误地低估或夸大，且这种效应无法通过调整估计器或分析单位来消除。 |
| [^352] | [Large-Scale Gaussian Splatting with a Multi-GPU PyTorch Backend](https://arxiv.org/abs/2606.11390) | 该论文提出基于CUDA统一内存的多GPU PyTorch后端，使高斯泼溅重建可扩展至超过10亿个高斯点的城市规模，且模型代码无需显式跨设备通信。 |
| [^353] | [Sample Where You Struggle: Sharpening Base Model Reasoning via Entropy-Guided Power Sampling](https://arxiv.org/abs/2606.09926) | 提出熵引导幂采样（EGPS），利用前向传播中已有的token级熵将MCMC移动聚焦到高熵决策点，以无需训练、无需验证器的方式高效提升基础模型从幂分布采样带来的强化学习级推理能力。 |
| [^354] | [A Geometric View for Understanding Concept Learning and Neuron Interpretation in Sparse Autoencoders](https://arxiv.org/abs/2606.07007) | 该论文提出了一个统一的数学几何框架，将稀疏自编码器中的概念学习形式化为人类定义概念与模型概念之间的集合对齐问题，区分了检测、分离和逼近三种学习强度，并从集合论角度解释了特征分裂、特征吸收等常见SAE现象。 |
| [^355] | [AlloGen: Conformation-Selective Binder Generation with Differential State Scoring](https://arxiv.org/abs/2606.05474) | AlloGen提出了一种将主链生成与状态选择性评分解耦的模块化方法，通过SE(3)不变的界面图变换器 $Q_\theta$ 对不同构象状态进行差分评分，从而在无需重新训练的情况下重排序候选并引导冻结的生成器，生成对特定受体构象状态具有选择性的蛋白质结合剂。 |
| [^356] | [SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511) | SparDA通过在Query、Key、Value之外引入与注意力查询解耦的Forecast投影来预测下一层所需的KV块，实现CPU到GPU预取与当前层计算重叠的前瞻式稀疏注意力选择，从而突破长上下文LLM推理中KV缓存容量与稀疏选择复杂度的双重瓶颈。 |
| [^357] | [Resolution-free neural surrogates for geometric parameterization and mapping with spatially varying fields](https://arxiv.org/abs/2605.28551) | 该论文提出了一种无分辨率的神经代理模型，能够以较低计算成本求解由空间变化参数场驱动的几何参数化与映射问题（如畸变校正、可变形图像配准等），避免了反复求解高分辨率偏微分方程系统的昂贵开销。 |
| [^358] | [Global linear convergence of entropy-regularized softmax policy gradient beyond tabular MDPs](https://arxiv.org/abs/2605.24939) | 该论文在连续状态-动作空间和线性函数逼近设定下，通过建立非一致PL不等式并证明其常数沿梯度流一致下有界，首次证明了熵正则化softmax策略梯度超越表格型MDP的全局线性收敛。 |
| [^359] | [Scalable Heterogeneous Graph Foundation Models for Data-Driven Optimal Power Flow in Smart Grids](https://arxiv.org/abs/2605.23194) | 本文提出基于HydraGNN的大规模异构图学习工作流，利用约三百万个涵盖不同规模电网的异构图训练图基础模型，为智能电网数据驱动的最优潮流代理建模提供了可扩展的统一解决方案。 |
| [^360] | [Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints](https://arxiv.org/abs/2605.22940) | 本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。 |
| [^361] | [Hyrax: An Extensible Framework for Rapid ML Experimentation and Unsupervised Discovery in the Era of Rubin, Roman, and Euclid](https://arxiv.org/abs/2605.18959) | Hyrax是一个开源、模块化、GPU加速的Python框架，为Rubin、Roman和Euclid时代的天文机器学习提供覆盖数据获取、训练、推理到实验比较的全生命周期支持，并集成向量数据库相似性搜索与交互式潜空间探索以实现无监督科学发现。 |
| [^362] | [SynVA: A Modular Toolkit for Vessel Generation and Aneurysm Editing](https://arxiv.org/abs/2605.17620) | SynVA 是一个模块化工具包，结合基于流匹配的血管网格生成方法与基于学习的动脉瘤合成方法，生成解剖学一致的脑血管数据，以缓解医学数据稀缺问题并助力颅内动脉瘤的诊断与治疗研究。 |
| [^363] | [Hierarchical Two-Stage Framework for Environment-Aware Long-Horizon Vessel Trajectory Prediction](https://arxiv.org/abs/2605.16442) | 该论文提出一种分层两阶段框架，结合基于时空图Transformer的短期局部动态预测与长期航行意图编码，并通过跨模态注意力融合洋流、风、浪等海洋环境参数，实现环境感知的长时程船舶轨迹精确预测。 |
| [^364] | [Coreset-Induced Conditional Velocity Flow Matching](https://arxiv.org/abs/2605.12951) | 该论文提出CCVFM生成模型，通过熵Sinkhorn核心集构建闭式代理源替代分层整流流中内部流的高斯噪声源，使条件速度规律成为无需神经采样器即可采样的闭式高斯混合，并只需轻量级校正流即可完成训练。 |
| [^365] | [The Transformer as a Polar State Estimator](https://arxiv.org/abs/2605.11007) | 本文证明Transformer的注意力机制、残差连接和归一化组件可从极坐标下的几何状态估计问题中自然推导得出，并提出保留了这些几何修正项的Polar Transformer。 |
| [^366] | [Regret-Oracle Complexity Tradeoffs in Agnostic Online Learning](https://arxiv.org/abs/2605.07155) | 本文显著改进了不可知在线学习中Oracle高效学习器的Oracle调用复杂度，且仅依赖更弱的弱一致性Oracle原语，实现了遗憾值与Oracle复杂度之间更优的权衡。 |
| [^367] | [EEG Foundation Models Need Channel Adaptation, but Fixed Channel Mappings Often Suffice](https://arxiv.org/abs/2604.23091) | 该研究通过对五个EEG预训练模型的系统评测发现，通道适配方法的选择取决于具体架构而非普适规律——在大多数模型-任务组合中简单的固定通道映射就已足够，只有刚性固定输入的BENDR等少数模型才需要可学习的适配方法。 |
| [^368] | [When the Prior Is Wrong: Probing Misspecified Diffusion Priors in Inverse Problems](https://arxiv.org/abs/2604.21066) | 该论文提出 PIVOT 方法，将扩散先验的错误设定建模为对代表科学合理偏差的潜在变量进行推断，从而能从单次观测中识别先验中哪些假设需要放宽以及放宽的程度。 |
| [^369] | [NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings](https://arxiv.org/abs/2604.10604) | 提出免训练的神经符号模糊逻辑框架 NSFL，借助 NS-Delta 边际语义贡献与黎曼优化查询投影，使稠密检索器无需训练即可原生处理多原子布尔逻辑约束，检索性能最高提升 81%。 |
| [^370] | [Reinforcement Learning from Human Feedback: A Statistical Perspective](https://arxiv.org/abs/2604.02507) | 本综述首次系统地从统计学视角梳理了RLHF框架，将其核心组成部分（监督微调、奖励建模与策略优化）与BTL模型、潜效用估计、主动学习、实验设计和不确定性量化等经典统计概念联系起来。 |
| [^371] | [Biologically Inspired Coordination of Parameter Selection and Update Strength for Language Model Adaptation](https://arxiv.org/abs/2604.00533) | 该论文提出了受生物学启发的SyCo框架，通过协调参数选择（基于Rac1遗忘机制保留选定的源低秩分量）与更新强度调节（基于MAPK信号机制结合多个预测指标），在不使用目标标签的情况下实现语言模型向目标任务的有效适配。 |
| [^372] | [Cross-Lingual Gaps in Visual STEM Reasoning: Auditing VLMs Across Six Indian Languages](https://arxiv.org/abs/2603.26742) | 该论文构建了六种印度语言的视觉STEM推理审计基准，发现视觉语言模型从英语切换到印度语言时准确率下降9.8至24.9个百分点，且达罗毗荼语系语言的性能下降比印度-雅利安语系更严重。 |
| [^373] | [Benchmarking LLM-Based Investment Advisor Agents for Asset Allocation in the Chinese ETF Market: Overview of NLPCC 2026 Shared Task 4](https://arxiv.org/abs/2603.22305) | 本文推出了 CN-Buzz2Portfolio——NLPCC 2026 共享任务 4 的官方基准，首次为中国 ETF 市场提供了评估大语言模型投资智能体进行新闻筛选、新闻映射与风险控制型资产配置的标准化端到端测试平台。 |
| [^374] | [Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization](https://arxiv.org/abs/2603.21676) | 本文提出一种深度循环Transformer，通过迭代共享权重模块在不生成思维链token的情况下实现更深的测试时推理，以恒定内存和线性延迟达成内存高效推理，并在组合泛化任务上验证了其有效性。 |
| [^375] | [Predictive Coding Implemented as a Hardware Substrate Computes Gradients faster than Backpropagation](https://arxiv.org/abs/2603.18066) | 本文将预测编码算法实例化为一种每个神经元对应一个物理核心的数字硬件基底，通过局部存储、通信与并发更新，使宽度为 n 的密集层的网络更新仅需 O(n) 个时钟周期，远快于反向传播串行处理所需的 O(n²) 次连接操作。 |
| [^376] | [Manifold-Matching Autoencoders: Embedding Data into Chosen Geometric Structures](https://arxiv.org/abs/2603.16568) | 提出一种名为“流形匹配”的简单正则化方法，通过最小化潜空间与任意参考表示之间成对距离的均方误差，使自编码器的潜空间能够被塑造成任意选定的几何结构，且无需标签、参考维数与潜空间维数自由解耦。 |
| [^377] | [FairMed-XGB: A Bayesian-Optimised Multi-Metric Framework with Explainability for Demographic Equity in Critical Healthcare Data](https://arxiv.org/abs/2603.14947) | 该论文提出FairMed-XGB框架，通过贝叶斯优化将统计均等差异、泰尔指数和Wasserstein距离三种公平性指标整合为综合损失函数并融入XGBoost分类器，在保持预测性能的同时，利用SHAP可解释性分析有效减轻了急诊医疗机器学习模型中的性别等人口统计学偏差。 |
| [^378] | [SVDP: Training-Free Contextual Sparsity Predictors for Fast LLM Inference](https://arxiv.org/abs/2603.14110) | 该论文提出SVDP框架，通过对门控投影矩阵进行截断感知SVD分解并结合阈值校准算法，无需训练即可构建稀疏模式预测器，在精度损失低于1%的情况下实现LLM前馈网络推理最高1.8倍的加速。 |
| [^379] | [Localized Dynamics-Aware Domain Adaptation for Off-Dynamics Offline Reinforcement Learning](https://arxiv.org/abs/2602.21072) | 提出LoDADA方法，通过聚类源与目标数据转移并借助域判别估计簇级动力学差异，优先保留差异较小簇中的源数据，实现细粒度、可扩展的数据选择，以更高效地复用源数据解决异动力学离线强化学习问题。 |
| [^380] | [In- and Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks](https://arxiv.org/abs/2602.15460) | 该论文提出了一个评估框架，系统检验多模态大语言模型的思维链推理在网格导航任务中的分布内与分布外泛化能力，发现输入地图格式与思维链格式对分布外泛化影响显著，且基于坐标的推理轨迹表现出令人惊讶的泛化效果。 |
| [^381] | [Explainable Machine-Learning based Detection of Knee Injuries in Runners](https://arxiv.org/abs/2602.11668) | 本研究利用可解释的机器学习方法分析光学动作捕捉的跑步支撑相数据，实现了对跑者膝部损伤（包括髌股疼痛综合征和髂胫束综合征）的有效检测与分类。 |
| [^382] | [Learned Finite Element-based Regularization of the Inverse Problem in Electrocardiographic Imaging](https://arxiv.org/abs/2602.07466) | 该论文提出了一种将空间正则化与学习的时间Fields-of-Experts先验相结合的心电图成像时空正则化框架，并借助有限元离散化与可扩展优化算法，实现了优于传统手工时空方法的去噪和逆问题重建效果。 |
| [^383] | [BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python](https://arxiv.org/abs/2602.07098) | BayesFlow 2是一个支持多种深度学习后端的通用Python库，通过摊销贝叶斯推断实现对后验分布、似然和比率的快速估计，解决了传统贝叶斯方法在复杂模型和大数据集上的计算瓶颈。 |
| [^384] | [TopoPrune: Robust Data Pruning via Unified Latent Space Topology](https://arxiv.org/abs/2602.02739) | TopoPrune通过全局拓扑感知流形嵌入与可微持续同调局部优化相结合的双尺度拓扑方法，捕获数据的内在稳定结构，实现对潜在空间扰动鲁棒的数据剪枝，即使在高剪枝率和特征噪声下仍能保持高精度。 |
| [^385] | [Latent Diffusion for Internet of Things Attack Data Generation in Intrusion Detection](https://arxiv.org/abs/2601.16976) | 该论文提出LaDIS框架，通过学习混合类型自编码器表示并在自适应连续潜在空间中执行扩散，为物联网入侵检测生成保真度高、保持多变量依赖且覆盖度好的攻击数据，尤其在代表性不足的攻击类型上优于对抗式、变分式和直接空间扩散等现有生成方法。 |
| [^386] | [Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding](https://arxiv.org/abs/2601.12095) | 该论文提出神经同构域（NIF），一种基于Transformer的数字嵌入方法，可证明地保留十进制分数的代数结构（与有理数域同构），从而解决神经网络处理极端数值时的溢出与不稳定问题。 |
| [^387] | [Mechanism-Based Intelligence: Differentiable Incentives for Rational Coordination and Equilibrium Alignment in Multi-Agent Systems](https://arxiv.org/abs/2512.20688) | 提出基于机制的智能（MBI）架构，通过可微价格机制将全局损失的精确梯度作为激励信号，使多智能体博弈成为以全局最优为唯一均衡的精确势博弈，从而同时解决哈耶克信息问题与赫维茨激励问题。 |
| [^388] | [CoDeQ: Compression with a Dead-Zone Quantizer for Differentiable Joint Pruning--Quantization](https://arxiv.org/abs/2512.12981) | 该论文提出CoDeQ，通过可学习的死区量化器将剪枝与量化统一在同一训练框架中，以死区宽度控制逐层稀疏度并支持可学习的混合精度位宽，在单一全局压缩系数下联合优化权重与压缩参数，从而省去辅助分配流程和大量超参数调优。 |
| [^389] | [Luxical: High-Speed Lexical-Dense Text Embeddings](https://arxiv.org/abs/2512.09015) | Luxical通过结合稀疏TF-IDF特征、小型ReLU网络和知识蒸馏训练，以极低的计算成本近似大型Transformer嵌入模型，实现了面向网络规模文本组织的高速“词汇密集”文本嵌入。 |
| [^390] | [Learning with Limited Antigenic Labels: Semi-supervised Prediction of Influenza A Antigenic Relationships with Protein Language Models](https://arxiv.org/abs/2512.05222) | 该论文的创新在于将预训练蛋白质语言模型表征（如ESM-2、ProtT5等）与半监督学习方法（自训练和标签传播）相结合，在标记数据稀缺的条件下预测甲型流感病毒的抗原关系，为减少昂贵的血凝抑制实验需求提供了新途径。 |
| [^391] | [Network of Theseus (like the ship)](https://arxiv.org/abs/2512.04198) | 提出忒修斯之网络方法，通过逐步替换组件并用表征相似性度量对齐，在保持性能的前提下将训练好的网络架构渐进转换为完全不同的目标架构，打破了“训练架构即部署架构”的传统假设。 |
| [^392] | [ML-Tool-Bench: Tool-Augmented Planning for ML Tasks](https://arxiv.org/abs/2512.00672) | 提出了ML-Tool-Bench基准，利用61个专用工具和15个Kaggle表格机器学习挑战，评估工具增强型ML智能体进行复杂规划的能力。 |
| [^393] | [db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism](https://arxiv.org/abs/2511.23113) | 本文提出db-SP，一种感知稀疏性的双重均衡序列并行技术，通过双层划分在注意力头级别和块级别实现近乎完美的负载均衡，从而加速采用块级稀疏注意力的扩散Transformer视觉生成模型的推理。 |
| [^394] | [ELBO$_{\mathrm{TDS}}$: Temporal Distribution Shift Generalization for Industry-Scale Recommender Systems](https://arxiv.org/abs/2511.21032) | 该论文提出了 ELBO$_{\mathrm{TDS}}$ 概率框架，通过识别时间分布偏移的关键因素、设计模拟时变因素的数据增强策略以及自监督变分目标，解决工业级推荐系统因时间分布偏移导致的泛化能力下降问题。 |
| [^395] | [StochEP: Stochastic Equilibrium Propagation for Spiking Convergent Recurrent Neural Networks](https://arxiv.org/abs/2511.11320) | 本文提出StochEP框架，通过将概率性脉冲神经元融入平衡传播范式，平滑了优化景观并稳定了训练，实现了脉冲神经网络高效且可扩展的学习，同时提供了理论保证。 |
| [^396] | [Fixed Points and Envelope Dynamics on Credal Hyperspaces](https://arxiv.org/abs/2510.04769) | 本文证明了信度集超空间中包含单调且豪斯多夫连续的自映射在完备格的信度序区间上存在可由迭代求得的最小与最大不动点，并为随时间变化的更新族构造了界定所有容许轨迹的上下包络动力学。 |
| [^397] | [Analysis of Variational Sparse Autoencoders](https://arxiv.org/abs/2509.22994) | 本研究通过置换比较分析变分稀疏自编码器，发现非采样 vSAE 与带 L2 激活惩罚的 Top-K SAE 之间的性能差距可完全分解为五个实现细节，且 Top-K 的离散选择机制与随机采样方法结合时会失效。 |
| [^398] | [GDS Agent for Graph Algorithmic Reasoning](https://arxiv.org/abs/2508.20637) | GDS 智能体通过 MCP 服务器将一整套图算法封装为工具并提供预处理与后处理流程，使任何现代 LLM 都能开箱即用地对大规模图结构数据进行算法推理。 |
| [^399] | [SACA: Selective Attention-Based Clustering Algorithm](https://arxiv.org/abs/2508.17150) | 本文提出了一种受选择性注意力概念启发的新型密度聚类算法SACA，通过自适应阈值自动过滤噪声点和异常值并重新整合优化结果，大幅降低了传统密度聚类方法对参数调优的依赖。 |
| [^400] | [Statistical Inference for Differentially Private Stochastic Gradient Descent](https://arxiv.org/abs/2507.20560) | 本文首次建立了随机化子采样规则下SGD的渐近性质并将其扩展到DP-SGD，证明了其渐近方差可分解为统计、采样和隐私诱导三个分量，并提出了代入法和随机缩放法两种构建有效置信区间的方法。 |
| [^401] | [PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI](https://arxiv.org/abs/2505.12707) | 该论文提出PLAICraft，一个从232名以上参与者收集的、涵盖视频、游戏音频、麦克风音频、鼠标和键盘五种毫秒级时间对齐模态的5525小时多人Minecraft交互数据集，并配套评估套件用于基准测试具身智能模型的各项能力。 |
| [^402] | [Improving regional weather forecasts with neural interpolation](https://arxiv.org/abs/2505.12040) | 本文设计了一个融合图像超分辨率技术、卷积神经网络、残差网络以及大气动力学流动的神经插值算子来改进区域天气模型的边界数据，其中嵌入动力学流动使小型网络的平方误差减少一半以上。 |
| [^403] | [Stability Regularized Cross-Validation](https://arxiv.org/abs/2505.06927) | 该论文提出了一种嵌套 k 折交叉验证方案，通过最小化标准交叉验证误差与经验假设稳定性的加权和来选择超参数，并证明了由此得到的泛化界，显示在信噪比较低的场景下该方法对测试误差的改善最为显著。 |
| [^404] | [A Unifying Framework for Robust and Efficient Inference with Unstructured Data](https://arxiv.org/abs/2505.00282) | 该论文提出MAR-S框架，将半参数缺失数据理论应用于AI预测，通过透明的验证样本设计实现非结构化数据下有效、高效且稳健的推断，并首次将框架扩展到聚合和转换后的AI预测这一常见但未被现有方法覆盖的场景。 |
| [^405] | [How to Enhance Downstream Adversarial Robustness (almost) without Touching the Pre-Trained Foundation Model?](https://arxiv.org/abs/2504.10850) | 该论文通过理论揭示鲁棒对比学习与监督学习对抗鲁棒性之间的密切关系，提出了一种无需更新或访问预训练基础模型权重即可有效增强下游任务对抗鲁棒性的简单方法。 |
| [^406] | [Neural Fidelity Calibration for Informative Sim-to-Real Adaptation](https://arxiv.org/abs/2504.08604) | 提出神经保真度校准（NFC）框架，利用条件得分扩散模型在机器人运行时在线校准仿真器物理参数与残差保真度域，从而无需专家物理知识即可弥合仿真到现实的差距并捕捉环境不确定性。 |
| [^407] | [DRAFT-ing Architectural Design Decisions using LLMs](https://arxiv.org/abs/2504.08207) | 该论文提出了DRAFT方法，将检索增强少样本生成与微调相结合，使大语言模型能够根据决策上下文生成架构设计决策，并根据标题生成完整的架构决策记录。 |
| [^408] | [Out-of-Distribution Detection on Graphs: A Survey](https://arxiv.org/abs/2502.08105) | 本文提出了首个针对图分布外（GOOD）检测的系统性综述，将现有方法按其主要OOD判别机制分为六大类别，并系统考察了它们的技术关联、监督需求、数据集、基准与评估指标。 |
| [^409] | [Forget Vectors at Play: Universal Input Perturbations Driving Machine Unlearning in Image Classification and Generation](https://arxiv.org/abs/2412.16780) | 该论文提出了一种名为“遗忘向量”的通用输入扰动方法，无需修改模型权重，即可在图像分类与图像生成任务中实现与基于模型的遗忘方法同样有效的机器遗忘。 |
| [^410] | [Protecting Confidentiality, Privacy and Integrity in Collaborative Learning](https://arxiv.org/abs/2412.08534) | 本文提出Citadel++协作式机器学习训练系统，通过以虚拟机级可信执行环境（TEE）作为训练运行时的链式方法，同时保护数据集、模型和训练代码的机密性以及个体用户数据的隐私。 |
| [^411] | [Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts](https://arxiv.org/abs/2411.09459) | 本文提出了Caravan水文数据集的新扩展Caravan MultiMet，在ERA5-Land再分析数据基础上增加了三种降水临近预报产品和三种天气预报产品，从而丰富气象强迫数据，支持对水文模型（尤其是实时预报场景）进行更稳健的评估和基准测试。 |
| [^412] | [ADformer: A Multi-Granularity Spatial-Temporal Transformer for EEG-Based Alzheimer Detection](https://arxiv.org/abs/2409.00032) | 提出ADformer，一种多粒度时空Transformer，可直接从原始脑电信号端到端地学习时间与空间特征表示，有效提升阿尔茨海默病检测的性能与泛化能力。 |
| [^413] | [Point2SSM++: Self-Supervised Learning of Anatomical Shape Models from Point Clouds](https://arxiv.org/abs/2405.09707) | 提出了Point2SSM++，一种自监督深度学习方法，可直接从解剖形状的点云表示中学习对应点，无需完整对齐的表面表示，同时避免了传统优化方法带来的偏差和漫长的推理时间。 |
| [^414] | [Neur2BiLO: Neural Bilevel Optimization](https://arxiv.org/abs/2402.02552) | Neur2BiLO是一个针对双层优化问题的框架，通过将神经网络近似引入到混合整数规划中，可以快速生成高质量的解决方案。 |
| [^415] | [Dynamic Tensor Product Regression](https://arxiv.org/abs/2210.03961) | 本文开创性地提出了动态张量积回归问题，并设计了一种动态树数据结构，使得当某个因子矩阵发生稀疏变化时，可以快速传播更新、高效维护张量积回归的解，而无需从头重新计算。 |
| [^416] | [Machine Learning-Driven Virtual Bidding with Electricity Market Efficiency Analysis](https://arxiv.org/abs/2104.02754) | 本文提出了一种考虑风险约束和价格敏感性的机器学习驱动虚拟竞价投资组合优化框架，并利用该策略评估了虚拟投标组合的盈利能力及美国批发电力市场的效率。 |
| [^417] | [Structural transfer learning of non-Gaussian DAG.](http://arxiv.org/abs/2310.10239) | 本文提出了一种用于非高斯有向无环图的结构迁移学习方法，通过引入新颖的DAG结构相似度度量并利用不同相似性水平的辅助DAG的信息，能够显著提高目标研究中的DAG重构效果。 |
| [^418] | [Tensor Networks Meet Neural Networks: A Survey and Future Perspectives.](http://arxiv.org/abs/2302.09019) | 这篇论文评述了张量网络和神经网络并介绍了它们的结合：张量神经网络(TNN)，探讨了TNN在网络压缩、信息融合和量子启发式神经网络构建方面的优缺点和未来研究方向。 |
| [^419] | [Sketching for First Order Method: Efficient Algorithm for Low-Bandwidth Channel and Vulnerability.](http://arxiv.org/abs/2210.08371) | 本文提出了一种针对大规模分布式学习场景的新型草图方案，通过节约通信成本实现联合学习，但是需要进行差分隐私处理以保护局部数据的隐私。 |

# 详细

[^1]: 分词化不属于APX复杂度类

    Tokenisation is not in APX

    [https://arxiv.org/abs/2611.34938](https://arxiv.org/abs/2611.34938)

    本文证明除非P=NP，分词化问题无法在多项式时间内近似到最优解的任意常数因子（即不属于APX类），且直接分词即使在一元字符表上也是APX难的。

    

    在本文中，我们证明，除非P=NP，否则不存在任何多项式时间算法能够找到一个分词器，使其目标值在最优值的任意常数因子范围内，这意味着分词化问题不属于APX复杂度类。我们对直接分词和自底向上分词均证明了这一结论，涵盖压缩目标和unigram对数似然目标两种情形，以及二元或更大字符表的情况。我们进一步证明，即使在一元字符表上，直接分词也是APX难的，因此不存在多项式时间近似方案。

    arXiv:2611.34938v1 Announce Type: new  Abstract: In this paper, we prove that, unless P=NP, no polynomial-time algorithm can find a tokeniser whose objective is within any constant factor of the optimum, meaning that tokenisation is not in the APX complexity class. We prove this for direct and bottom-up tokenisation, under both a compression or unigram log-likelihood objectives, and for binary or larger alphabets. We further show that, even over unary alphabets, direct tokenisation is APX-hard, and thus admits no polynomial-time approximation scheme.
    
[^2]: 图像生成能否帮助视觉推理？一项实证研究

    Can image generation help visual reasoning? An empirical study

    [https://arxiv.org/abs/2611.34935](https://arxiv.org/abs/2611.34935)

    这项实证研究发现，通过生成辅助图像来帮助视觉推理在大多数情况下反而会降低性能——在13组视觉语言模型与生成器的配对实验中平均准确率下降3.7个百分点，十个任务中九个出现下降——挑战了“多模态模型可以用图像思考”的假设。

    

    语言模型用词语思考，那么多模态模型是否可能用图像思考？我们测试了生成辅助图像能否改善视觉问答的表现。模型首先描述它所需要的辅助图像，然后结合原始图像和生成图像一起回答问题。在13组视觉语言模型与生成器的配对实验中，图像生成使平均准确率下降了3.7个百分点，十个任务中有九个出现下降。两个统一模型BAGEL和Janus-Pro-7B各自在八个任务上损失了准确率。在VSTAR数据集上，在相同的规划提示下额外加入一张生成图像会使准确率降低10.47个百分点。重复的图像编辑在不同任务上产生了不同结果：ChartQAPro和MathVista始终低于无生成基线，而ERQA在十次编辑后提升了三个百分点。

    arXiv:2611.34935v1 Announce Type: cross  Abstract: Language models think in words; then multimodal models might think in images. We test whether a generated helper improves the answer to a visual question. A model describes the helper it needs and then answers using the original and generated images together. Across 13 vision-language model and generator pairs, generation lowers mean accuracy by 3.7 percentage points, with declines on nine of ten tasks. Two unified models, BAGEL and Janus-Pro-7B, each lose accuracy on eight tasks. On VSTAR, adding a generated image to the same planning hint lowers accuracy by 10.47 points. Repeated image edits yield different outcomes across tasks. ChartQAPro and MathVista stay below their no-generation baselines, while ERQA gains three points after ten edits.
    
[^3]: 攻击策略选择可能误导监测器评估

    Attack Policy Selection can Mislead Monitor Evaluators

    [https://arxiv.org/abs/2611.34934](https://arxiv.org/abs/2611.34934)

    攻击策略的生成方法和优化程度会极大影响AI监测器假阴性率的测量结果（从0.5%到23.0%不等），这表明仅使用基础攻击方法的监测器规避评估很容易产生误导性结论。

    

    AI监测器是用于标记AI智能体危险行为的分类器，其评估通常依赖于由攻击策略生成的对话记录，这些攻击策略充当部署环境中危险行为的代理。我们以一个OpenAI已退出生产环境的监测器为案例研究，发现测得的假阴性率对生成攻击策略所用的方法以及在该方法内部应用的优化水平都高度敏感。在单一方法（强化学习）的不同优化水平下，该比率在0.5%至20.8%之间变化；在不同方法各自的最佳观察优化水平下（优化提示 vs. 对话记录编辑），该比率在6.4%至23.0%之间变化。每个攻击策略都仅针对代理监测器开发：所有报告的结果均显示攻击成功迁移到留出的GPT-5.4目标监测器上。我们的结果表明，监测器规避评估很容易造成误导：仅使用基础攻击方法的评估可能会给出虚假的……（原文摘要在此处截断）

    arXiv:2611.34934v1 Announce Type: cross  Abstract: AI monitors, classifiers that flag dangerous behaviour by AI agents, are often evaluated against transcripts generated by attack policies acting as proxies for dangerous behaviour in deployment. We evaluate an ex-production OpenAI monitor as a case study and find that measured false negative rate is highly sensitive to both the method used to generate attack policies and the level of optimisation applied within that method. This rate ranged from 0.5\% to 20.8\% across optimisation levels of a single method (reinforcement learning), and from 6.4\% to 23.0\% across methods at their best observed optimisation level (optimised prompting vs.\ transcript editing). Every attack policy was developed against proxy monitors only: all reported results show attacks transferring to the held-out GPT-5.4 target monitor. Our results suggest that monitor evasion evaluations can easily mislead: an evaluation using only basic attack methods may give fals
    
[^4]: 一种基于基因信息的病理学基础模型BioScoring基准测试

    A gene informed benchmark for BioScoring of pathology foundation models

    [https://arxiv.org/abs/2611.34930](https://arxiv.org/abs/2611.34930)

    本文提出BioScore基准，通过空间转录组学与完全无监督的稀疏自编码器量化病理学基础模型对52种细胞身份的生物学表征能力，发现所有病理学模型的得分均高于自然图像模型。

    

    病理学基础模型通常通过其在分类和检索等下游任务上的表现来进行比较。虽然这类基准测试衡量了模型的实用性，却无法揭示模型所表征的生物学信息。我们提出了BioScore，这是一个以同一组织的空间转录组学作为参考、量化图像编码器捕捉细胞身份能力的基准测试。稀疏自编码器将编码器的图像块嵌入分解为稀疏特征，再将这些特征聚类为固定数量的组。BioScore随后针对由标记基因定义的52种细胞身份中的每一种，衡量某个组的特征在组织上跟随该身份表达的特异性程度。由于自编码器的训练是完全无监督的，特征与细胞身份之间的匹配表明模型仅凭图像即可表征该细胞身份。我们对8个病理学基础模型和3个自然图像模型进行了评分，结果显示每个病理学模型的得分均高于每个自然图像模型。

    arXiv:2611.34930v1 Announce Type: new  Abstract: Pathology foundation models are usually compared by their performance on downstream tasks such as classification and retrieval. While such benchmarks measure how useful a model is, they provide no insight into the biology it represents. We introduce BioScore, a benchmark that quantifies how well an image encoder captures cell identities, with spatial transcriptomics of the same tissue as the reference. A sparse autoencoder splits the encoder's patch embeddings into sparse features, which we cluster into a fixed number of groups. BioScore then measures, for each of 52 cell identities defined by marker genes, how specifically one group follows the identity's expression over the tissue. Because the autoencoder training is fully unsupervised, a match shows that the model represents the cell identity from the image alone. We score 8 pathology foundation models and 3 natural-image models. Every pathology model scores above every natural-image 
    
[^5]: 面向复杂数学问题的多样化替代推理轨迹

    Diverse Alternative Reasoning Traces for Hard Math Problems

    [https://arxiv.org/abs/2611.34928](https://arxiv.org/abs/2611.34928)

    提出了DART轨迹挖掘方法，通过分解被拒绝的推理轨迹并基于位置加权不确定性与参考答案引导生成刻意不同的替代推理步骤，在相同生成成本下比蒙特卡洛采样多恢复37%的复杂数学问题，且每条合格轨迹成本降低30%。

    

    针对数学推理的大语言模型（LLM）后训练通常依赖于最终答案能通过验证器检验的合格轨迹。然而，对于复杂数学问题，在独立蒙特卡洛（MC）采样下，这类轨迹可能非常稀少。为了解决这一问题，我们提出了多样化替代推理轨迹（Diverse Alternative Reasoning Traces，DART），这是一种轨迹挖掘方法：它从一条被拒绝的轨迹出发，将其分解为推理步骤，并按位置加权不确定性进行排序；随后利用参考答案引导和元思考（meta-thought），为排名靠前的步骤生成刻意不同的替代候选，最后完成由此产生的完整轨迹。在贪心解码未能解决的DeepMath-103K问题上，在生成轨迹数量相同的条件下，DART比MC采样多解决37%的问题；其每个已恢复问题的估计生成成本几乎相同，而每条合格轨迹的成本降低了30%。在DART挖掘的轨迹上对Llama-3.1-8B-Instruct进行后训练取得了显著提升。

    arXiv:2611.34928v1 Announce Type: cross  Abstract: Post-training large language models (LLMs) for mathematical reasoning often relies on accepted traces whose final answers pass a verifier. For hard math problems, however, such traces can be rare under independent Monte Carlo (MC) sampling. To address this, we propose Diverse Alternative Reasoning Traces (DART), a trace-mining method that starts from a rejected trace, decomposes it into reasoning steps, and ranks them by position-weighted uncertainty. It then generates deliberately distinct alternative candidates for highly ranked steps using reference-answer guidance and a meta-thought before completing the resulting traces. On greedy-unsolved DeepMath-103K problems, DART recovers 37% more problems than MC sampling at a matched number of generated traces. Its estimated generation cost per recovered problem is nearly identical, while its cost per accepted trace is 30% lower. Post-training Llama-3.1-8B-Instruct on traces mined by DART a
    
[^6]: 混合循环Transformer

    Mixture-of-Loops Transformer

    [https://arxiv.org/abs/2611.34923](https://arxiv.org/abs/2611.34923)

    提出混合循环Transformer（MoLT），以KV缓存作为跨深度和跨循环的选择性记忆重用机制，在表达能力与效率之间取得新平衡，并支持超越训练循环次数的高效解码。

    

    循环Transformer通过在多个循环中重用共享层来提升表达能力，但代价是效率的下降。要推动其表达能力与效率的边界，就需要重新思考连接相邻循环的核心机制：当Transformer进行循环时，应该传递什么？我们提出了混合循环Transformer（MoLT），其混合循环注意力机制（MoLA）将键值对（KV）用作记忆，实现跨深度和跨循环的选择性重用。在循环残差模式下，MoLT-R保留残差流并跨循环读取局部KV，以增强表达能力。在循环KV模式下，MoLT-KV则从嵌入层重新开始，读取当前循环的局部KV以及前面循环在较早位置的记忆KV，以提升效率。MoLT-KV消除了同一位置上的跨循环依赖，并结合用于并行训练的循环强制技术，实现了超越训练循环次数的单次遍历混合循环（MoL）解码。为了平衡效率和质量，我们引入了U-

    arXiv:2611.34923v1 Announce Type: cross  Abstract: Looped Transformers boost expressivity by reusing shared layers across loops, yet at the cost of efficiency. Pushing their expressivity-efficiency frontier requires rethinking the core mechanism connecting adjacent loops: What should be passed as a Transformer loops? We introduce Mixture-of-Loops Transformer (MoLT), whose Mixture-of-Loops Attention (MoLA) uses keys and values (KV) as memory for selective reuse across depths and loops. Under loop residual, MoLT-R retains the residual stream and reads local KV across loops for expressivity. Under loop KV, MoLT-KV instead restarts from embeddings and reads current-loop local KV with preceding-loop memory KV at earlier positions for efficiency. MoLT-KV eliminates same-position cross-loop dependence and, paired with loop forcing for parallel training, unlocks single-traversal Mixture-of-Loops (MoL) decoding beyond the training loop horizon. To balance efficiency and quality, we introduce U-
    
[^7]: 使用潜在空间表示增强海事领域感知

    Enhancing Maritime Domain Awareness using Latent Space Representations

    [https://arxiv.org/abs/2611.34899](https://arxiv.org/abs/2611.34899)

    该论文利用地理空间基础模型TerraMind生成的压缩潜在空间嵌入，直接从Sentinel-1 SAR图像中实现船舶检测（ROC-AUC 0.95）、定位（中位误差约300米）和类型分类（62%准确率），有效缓解了海事领域感知中地球观测数据处理的数据引力瓶颈。

    

    本工作从地理空间基础模型TerraMind的视角，展示了地球观测嵌入在海事领域感知（MDA）中的一项新颖应用。通过直接在压缩的潜在空间表示上进行操作，而非在原始Sentinel-1图像上进行操作，该方法直接应对了日益制约业务化地球观测处理的数据引力瓶颈，这与Embed2Scale项目的目标一致。特别是，通过对压缩嵌入（即合成孔径雷达（SAR）图像的潜在空间表示）进行线性和非线性探测，用于从Sentinel-1 GRD图像中检测、定位和表征船舶。船舶检测达到了0.95的ROC-AUC，船舶定位对于长度超过100米的船舶中位误差约为300米，船舶AIS类型编码分类对多数类别达到了62%的准确率。然而，后者与……（原文摘要在此处截断）

    arXiv:2611.34899v1 Announce Type: cross  Abstract: This work presents a novel application of Earth Observation (EO) embeddings for Maritime Domain Awareness (MDA) from the perspective of the geospatial foundation model TerraMind. By operating directly on compressed latent-space representations rather than on the original Sentinel-1 imagery, this approach directly addresses the data gravity bottleneck that increasingly constrains operational EO processing, in line with the objectives of the Embed2Scale project. In particular, linear and non-linear probing of compressed embeddings, latent space representations of synthetic aperture radar (SAR) imagery, is used to detect, localise, and characterise vessels from Sentinel-1 GRD images. Vessel detection achieves a ROC-AUC of 0.95, while vessel localisation has a median error of around 300 m for vessels longer than 100 m, and vessel AIS type-code classification achieves 62% accuracy for the majority classes. The latter, however, is highly cor
    
[^8]: 一次坠落揭示了什么：从视频实现可变形物体材料属性的模拟到真实（Sim-to-Real）估计

    What a Drop Reveals: Sim-to-Real Material Estimation for Deformable Objects from Video

    [https://arxiv.org/abs/2611.34894](https://arxiv.org/abs/2611.34894)

    本论文发现仅用合成数据验证会误导输入表示的选择——仿真中最准确的三维轨迹输入在真实坠落视频上反而最差、且无法靠更好的深度修复，而简单的二维光流短轨迹在真实场景中最有效，从而实现了仅凭仿真训练、从单段真实坠落视频估计可变形物体刚度与屈服应力的sim-to-real方法。

    

    一个物体落地时如何形变，揭示了它的材质构成。仅需一段真实的坠落视频，我们即可估计可变形物体的密度归一化刚度 E/ρ，以及在冲击引起塑性变形时的屈服应力 σY/ρ，所用的前馈预测器仅在仿真中训练。由于真实视频不含材料标签，这类预测器只能在合成数据上进行开发与验证。我们证明，合成数据的验证会把几何输入的优劣排出错误的顺序。材料属性通过三维形变起作用，因此借助估计深度提升到三维的轨迹在仿真中确实是最准确的输入；然而在真实坠落中它们却最不准确，而且更好的深度也无法修复这一问题：无论训练时使用仿真深度、测试时使用LiDAR深度，还是在真实坠落的渲染回放中使用精确深度，这种失败都持续存在。而短的二维光流轨迹恰恰相反：它们在仿真中毫无改进……（原文摘要至此截断）

    arXiv:2611.34894v1 Announce Type: cross  Abstract: How an object deforms when it hits the ground reveals what it is made of. From a single real video of a drop, we estimate a deformable object's density-normalized stiffness $E/\rho$, and its yield stress $\sigma_Y/\rho$ when the impact causes plastic deformation, with a feed-forward predictor trained only in simulation. Real videos carry no material labels, so such predictors are developed and validated on synthetic data alone. We show that synthetic validation ranks their geometric inputs in the wrong order. Material acts through 3D deformation, and tracklets lifted to 3D with estimated depth are indeed the most accurate input in simulation. However, on real drops they are the least accurate, and better depth does not repair them: the failure persists with simulator depth in training, LiDAR depth at test time, and exact depth on rendered replays of the real drops. Short 2D optical-flow tracklets do the opposite: they improve nothing i
    
[^9]: 基于选择性案例分割的可解释且准确的混合决策树

    Interpretable and Accurate Hybrid Decision Trees with Selective Case-Based Splits

    [https://arxiv.org/abs/2611.34887](https://arxiv.org/abs/2611.34887)

    这篇论文提出了一种集成基于特征分割和基于案例分割的混合决策树架构，并引入了一种新颖的选择性基于案例的分割方法，通过搜索最具信息量的特征子集来构建既可解释又准确的决策树。

    

    决策树是机器学习中最著名的可解释模型之一。多年来，人们提出了各种各样的策略来定义贪婪学习框架内的分割条件，旨在指导实例路由并提高预测性能。这些策略包括基于特征值的轴平行和倾斜分割、使用范例实例的基于邻近性的分割，以及更复杂的混合方法。然而，没有任何一种方法被证明是普遍优越的。在这项工作中，我们对贪婪决策树的构建提出了两个关键贡献。首先，我们提出了一种混合树架构，它集成了基于特征的分割和基于案例的分割，允许在整个树中进行异构分割。其次，我们引入了一种新颖的分割类型，该分割类型利用了选择性的基于案例的方法：当需要分割时，我们的方法会搜索最具信息量的特征子集。

    arXiv:2611.34887v1 Announce Type: cross  Abstract: Decision trees are among the most well-known interpretable models in Machine Learning. Over the years, a wide variety of strategies have been proposed to define split conditions within the greedy learning framework, aiming to guide instance routing and enhance predictive performance. These strategies range from axis-parallel and oblique splits based on feature values to proximity-based splits using exemplar instances and more complex hybrid approaches. However, no single method has emerged as universally superior. In this work, we present two key contributions to the construction of greedy decision trees. First, we propose a hybrid tree architecture that integrates both feature-based and case-based splits, allowing for heterogeneous splits throughout the tree. Second, we introduce a novel type of split that leverages a selective case-based approach: when a split is needed, our method searches for the most informative subset of features
    
[^10]: 怪异之物：当物体脱离其典型邻居出现时

    Stranger Things: When Objects Appear Without Their Typical Neighbors

    [https://arxiv.org/abs/2611.34881](https://arxiv.org/abs/2611.34881)

    该论文提出了通过归一化逐点互信息（NPMI）度量典型性偏差的诊断框架，并构建了照片级真实感、布局受控的STRANGE-Bench基准（涵盖17个从典型到从不共现的物体对子层级），首次系统性地诊断了现代视觉模型中因依赖上下文邻居而忽视物体自身特征的典型性偏差问题。

    

    视觉上下文是物体识别的一个强大线索：场景能够提示物体可能以何种内容、在何处、以何种尺度出现。然而，当模型依据物体的典型邻居而非物体自身的视觉特征来预测物体时，对上下文的过度依赖便成为一种失败模式，我们将其称为典型性偏差（typicality bias）。尽管这一问题在Transformer出现之前的时代已被广泛研究，但人们通常假设架构与规模的改进会自然地解决它，而针对现代视觉技术栈的系统性诊断仍然缺失。我们提供了这样一个诊断框架。我们通过物体对上的归一化逐点互信息（NPMI）来度量典型性偏差，并引入STRANGE-Bench（合成典型性排序接地评估基准）：一个照片级真实感、布局受控的基准，涵盖从典型共现到从不共现的物体对的17个子层级。在五个模型家族（专用检测器、开放词表……

    arXiv:2611.34881v1 Announce Type: cross  Abstract: Visual context is a powerful cue for object recognition: scenes inform what, where, and at what scale objects are likely to appear. Yet over-reliance on context becomes a failure mode when models predict objects based on their typical neighbors rather than the objects' own visual features, which we call typicality bias. While extensively studied in the pre-Transformer era, it has often been assumed that architectural and scale improvements would naturally solve it, and a systematic diagnostic across the modern vision stack remains absent. We provide such a diagnostic framework. We measure typicality bias via normalized pointwise mutual information (NPMI) over object pairs, and introduce STRANGE-Bench (Synthetic Typicality-RANked Grounded Evaluation Benchmark): a photorealistic, layout-controlled benchmark spanning 17 sub-tiers from typical to never-co-occurring object pairs. Across five model families (specialist detectors, open-vocabu
    
[^11]: 只关注重要内容：面向长视频目标分割中冻结视觉Transformer的无训练焦点图

    Focus Only on What's Important: Training-Free Focus Maps for Frozen Vision Transformers in Long-Video Object Segmentation

    [https://arxiv.org/abs/2611.34863](https://arxiv.org/abs/2611.34863)

    该论文提出一种无需训练的焦点图策略，通过融合速度偏移掩码与块匹配掩码、相机补偿的意外性检查以及不确定性区域委托机制，使冻结的DINOv3 ViT仅在分割目标预期区域保持全分辨率，在长视频目标分割任务中以更少计算量取得显著性能提升。

    

    视觉Transformer是视频目标分割中强大但成本高昂的图像编码器。无训练的token合并在整个画面上均匀削减计算量，包括在被跟踪的目标上。我们为经过训练的Cutie头内的冻结DINOv3 ViT提供了一种无需训练的焦点图：在分割器预期目标所在的位置保持全分辨率，而在其他位置进行更粗粒度的合并。仅保护前一帧的掩码在短视频片段上有效，但在长视频上会失效，因为快速运动会把目标推出保护边缘、进入背景合并区域，从而锁定损失。我们的策略改为取速度偏移掩码与块匹配掩码的并集，执行相机补偿的意外性检查，将不确定区域委托给对比的合并方法，并在所有目标丢失时回退到该方法。在长视频上（LVOS V2验证集，预设协议，三个ViT-S随机种子），该方法以少16%的计算量相比ALGM 0.98提升了+2.76 J&F（95%目标级置信区间[+1.22,+4.44]；视频聚类置信区间[+0.99,+4.52]）

    arXiv:2611.34863v1 Announce Type: cross  Abstract: Vision transformers are strong but costly image encoders for video object segmentation. Training-free token merging cuts compute uniformly across the frame, including on tracked objects. We give a frozen DINOv3 ViT inside a trained Cutie head a training-free focus map: full resolution where the segmenter expects its objects and coarser merging elsewhere. Protecting only the previous frame's mask works on short clips but fails on long videos, where fast motion pushes targets outside the margin and into background merging, locking in loss. Our policy instead takes the union of velocity-shifted and block-matched masks, runs a camera-compensated surprise check, delegates uncertain regions to the compared merger, and falls back to it when all objects are lost. On long videos (LVOS V2 valid, pre-specified protocol, three ViT-S seeds), it beats ALGM 0.98 by +2.76 J&F (95% object CI [+1.22,+4.44]; video-cluster [+0.99,+4.52]) at 16% fewer coun
    
[^12]: 模型权重、气闸与沙盒：呼吁在EDPB匿名化指南下对重识别测试进行技术验证

    Model Weights, Airlocks and Sandboxes: A Call for Technical Validation of Re-identification Tests under the EDPB Anonymisation Guidelines

    [https://arxiv.org/abs/2611.34859](https://arxiv.org/abs/2611.34859)

    针对EDPB匿名化指南将AI模型纳入监管却未规定重识别测试如何设计与验证的问题，本文提出了基于气闸式安全处理环境的评估框架，区分形式化、经验和间接三类证据，并在模型权重发布、气闸审查和监管沙盒三种场景下提出并探讨了七个开放性技术问题。

    

    欧洲数据保护委员会（EDPB）于2026年7月发布征求意见的《02/2026号匿名化指南》，从可能访问数据的实体角度来评估匿名性。该指南将AI模型纳入适用范围并要求进行测试，但并未说明重识别测试应如何设计或验证，也未说明其结果在多大程度上可算作证据。基于我们对征求意见的回复，我们提出了一个基于安全处理环境的框架，其输出在发布前须经审查（即“气闸”机制），并区分了不同的发布形式和三种类型的证据：形式化证据、经验证据和间接证据。我们提出了七个开放性问题，每个问题均配有具体场景，并在三种情境下考察其技术层面：模型权重（机器学习模型的训练参数）的发布、气闸处的审查，以及由监管机构监督开发与测试的监管沙盒。在每种情境下，评估……

    arXiv:2611.34859v1 Announce Type: cross  Abstract: The Guidelines 02/2026 on anonymisation of the European Data Protection Board (EDPB), published for consultation in July 2026, assess anonymity relative to the entities that may access data. They bring AI models into scope and call for testing, but do not say how a re-identification test should be designed or validated, or how far its result counts as evidence. Building on our reply to the consultation, we describe a framework based on a secure processing environment whose outputs are reviewed before release (an airlock), and distinguish forms of release and three types of evidence: formal, empirical and indirect. We raise seven open questions, each with a scenario, and examine their technical side in three settings: the release of model weights (the trained parameters of a machine learning model), review at an airlock, and regulatory sandboxes, in which an authority supervises development and testing. In each setting the assessment de
    
[^13]: 从预测到决策：面向成本敏感金融事件的自回归表格Transformer决策导向微调

    From Prediction to Decision: Decision-Focused Fine-Tuning of Autoregressive Tabular Transformers for Cost-Sensitive Financial Events

    [https://arxiv.org/abs/2611.34858](https://arxiv.org/abs/2611.34858)

    本文提出决策导向微调方法，通过直接最大化预训练自回归表格Transformer在按金额缩放成本矩阵下的期望效用，使模型输出从概率预测转变为面向业务动作的决策优化，从而在成本敏感的金融欺诈检测中优于传统的事后阈值划分方法。

    

    在标记化表格事件流上训练的自回归Transformer是预测金融事件（如欺诈）的一个简单而强大的基线模型。然而，机构并非依据概率行事，而是依据决策行事，而这些决策的后果是非对称且异方差的：漏检一笔高价值欺诈的代价与错误拒绝一笔小额交易的代价并不相同。连接二者的标准方法是事后在成本矩阵下对校准概率进行贝叶斯最优阈值划分。我们探究将模型微调至决策目标是否能取得更好的效果。我们对同一预训练模型进行微调，使其在标签token上的受限softmax（其本身已是业务动作的分布）在机构指定的、按金额缩放的成本矩阵下最大化期望效用。由于奖励是（动作、结果、金额）的已知函数，且结果在数据中可直接观测，期望效用目标函数及其梯度均可解析获得。

    arXiv:2611.34858v1 Announce Type: cross  Abstract: Autoregressive transformers trained on tokenized tabular event streams are a simple and strong baseline for predicting financial events such as fraud. Yet institutions do not act on probabilities; they act on decisions whose consequences are asymmetric and heteroscedastic: the cost of a missed high-value fraud and that of a wrongly declined small purchase are not the same. The standard bridge is post-hoc Bayes-optimal thresholding of calibrated probabilities under a cost matrix. We ask whether fine-tuning the model to the decision objective can do better. We fine-tune the same pretrained model so that its restricted softmax over label tokens (already a distribution over business actions) maximizes expected utility under an institution-specified, amount-scaled cost matrix. Because the reward is a known function of (action, outcome, amount) and the outcome is observed in the data, the expected-utility objective and its gradient are avail
    
[^14]: 回归世界本体：时间戳在时间序列推理中至关重要

    Put the World Back: Timestamps Matter in Time Series Reasoning

    [https://arxiv.org/abs/2611.34848](https://arxiv.org/abs/2611.34848)

    该论文提出从“世界到序列投影”的视角重新定义时间序列推理，将时间戳视为连接数值观测与外部事件、指向共享世界状态的显式索引，并据此开发了首个能区分真正的基于时间戳的推理与相关性捷径的诊断协议，从而暴露了现有基于准确率的评估基准的局限性。

    

    大语言模型（LLM）正被越来越多地应用于时间序列推理。然而，当前的评估通常按时间顺序向模型提供事件，且事件已预先与时间序列匹配。这使得我们无法判断模型究竟是利用时间戳将事件与观测数据联系起来，还是仅仅走了捷径。为了揭示这一模糊性，我们从“世界到序列的投影”视角重新表述时间序列推理，将时间戳视为显式索引，把数值观测和外部事件连接到共享的世界状态。我们识别出时间戳能够显式化的三种结构：时序因果结构、时间语义以及事件-序列对齐。基于这一视角，我们开发了一种诊断协议，用以区分基于时间戳的推理与相关性捷径——这一区分是基于准确率的基准测试所无法做到的。对来自两个任务的310个问题进行的受控干预实验表明，高准确率……（原文摘要在此处截断）

    arXiv:2611.34848v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly applied to time series reasoning. However, current evaluations often give models events in chronological order and already matched to the time series. This makes it unclear whether models use timestamps to connect events with observations or simply take shortcuts. To expose this ambiguity, we reformulate time series reasoning from a world-to-series projection perspective, viewing timestamps as explicit indices that connect numerical observations and external events to shared world states. We identify three structures that timestamps can make explicit: temporal causal structure, time semantics, and event-series alignment. Building on this perspective, we develop a diagnostic protocol that distinguishes timestamp-grounded reasoning from correlational shortcuts, a distinction that accuracy-based benchmarks cannot make. Controlled interventions on 310 questions from two tasks show that high accur
    
[^15]: 知识库梯度下降：通过问答实现的自结构化学习

    Knowledge-Base Gradient Descent: Self-Structuring Learning through Question Answering

    [https://arxiv.org/abs/2611.34841](https://arxiv.org/abs/2611.34841)

    该论文提出“知识库梯度下降”方法，通过诊断问答响应与预期答案之间的差异，并以类似梯度下降的方式迭代地对知识库结构提出编辑、验证并保留能降低损失的编辑，从而让大语言模型应用能够以问题和答案为监督信号自动学习构建最优的知识库结构。

    

    我们研究为基于大语言模型的应用构建知识库的问题，其中知识库的结构决定了应用如何检索和连接回答问题所需的信息。我们研究自结构化学习这一设定，即由问题和预期答案来指导构建一个持久的知识库，其可编辑的结构可以是图模式或链接知识文档的组织方式。我们提出知识库梯度下降方法，该方法诊断预期答案与通过当前知识库获得的响应之间的差异。在每一轮中，该方法提出一个结构性编辑，重建知识库，并在学习损失下降时采纳该编辑。所保留的结构会在关于相同资料来源的其他问题上进行评估。通过对候选编辑进行随机化，我们将某次编辑带来的损失变化表示为导数……（摘要在此截断）

    arXiv:2611.34841v1 Announce Type: new  Abstract: We consider the problem of constructing knowledge bases for applications based on large language models, in which the structure of a knowledge base determines how an application can retrieve and connect the information needed to answer a question. We study Self-Structuring Learning, a setting in which questions and expected answers guide the construction of a persistent knowledge base whose editable structure can be a graph schema or the organization of linked knowledge documents. We propose Knowledge-Base Gradient Descent, which diagnoses discrepancies between expected answers and responses obtained through the current knowledge base. At each round, the method proposes a structural edit, rebuilds the knowledge base, and adopts the edit when the learning loss decreases. The retained structure is evaluated on other questions about the same sources. By randomizing over candidate edits, we express the loss change of an edit as a derivative 
    
[^16]: 递归熵风险下可证明高效的离线折扣强化学习

    Provably Efficient Offline Discounted Reinforcement Learning under Recursive Entropic Risk

    [https://arxiv.org/abs/2611.34840](https://arxiv.org/abs/2611.34840)

    本文提出了首个在递归熵风险度量下离线强化学习的可证明高效算法 ERM-VI-LCB，并给出匹配的样本复杂度上下界，同时覆盖风险厌恶与风险追求两种情形。

    

    我们研究了有限折扣无限时域马尔可夫决策过程（MDP）中基于递归熵风险度量（ERM）的风险敏感强化学习，该度量由参数 $\beta\neq 0$ 刻画。我们同时考虑了风险厌恶（$\beta>0$）和风险追求（$\beta<0$）两种设定。我们研究离线强化学习场景，即智能体获得一个固定数据集，并希望利用该数据集学习一个近似最优的策略。我们提出了一种基于模型的算法，称为 ERM 值迭代下置信界（ERM-VI-LCB），并推导出其样本复杂度的 PAC 型界。我们还建立了相应的下界，证明上界中所断言的对 $|\beta|/(1-\gamma)$ 的指数依赖在最坏情况下是不可避免的。所提出的界为递归 ERM 下的离线强化学习在风险厌恶和风险追求两种情形下提供了首个严格的样本复杂度保证。

    arXiv:2611.34840v1 Announce Type: new  Abstract: We study risk-sensitive reinforcement learning in finite discounted infinite-horizon Markov Decision Processes with recursive entropic risk measures (ERM), parameterized by $\beta\neq 0$. We consider both risk-averse ($\beta>0$) and risk-seeking ($\beta<0$) settings. We consider the offline RL setting, where the agent is provided with a fixed dataset and wishes to learn a near-optimal policy using this dataset. We introduce a model-based algorithm, called ERM Value Iteration Lower Confidence Bound (ERM-VI-LCB) and derive a PAC-type bound on its sample complexity. We also establish a corresponding lower bound, demonstrating that exponential dependence on $|\beta|/(1-\gamma)$ asserted by the upper bound is unavoidable in the worst case. The presented bounds provide the first rigorous sample complexity guarantees for offline RL under recursive ERM across both risk-averse and risk-seeking regimes.
    
[^17]: 并非所有世界模型都会怀疑：训练目标与架构如何塑造JEPA与RSSM世界模型中的认知不确定性

    Not all world models can doubt: how training objective and architecture shape epistemic uncertainty in JEPA and RSSM world models

    [https://arxiv.org/abs/2611.34830](https://arxiv.org/abs/2611.34830)

    本文通过在十个模拟环境中对JEPA架构的LeWorldModel与RSSM架构的DreamerV3进行受控对比，实证揭示了训练目标与架构从根本上决定了世界模型能否有效表达认知不确定性（即“怀疑”的能力）。

    

    世界模型通过预测环境对动作的响应，使得候选动作可以在执行之前得到评估。以这种方式执行的动作（即决策）的可靠性取决于模型的预测能力，而预测又仅在模型经验覆盖的范围内才是可靠的。认知不确定性正是标记经验边界所在的信号。在这项工作中，我们对训练目标和架构如何塑造世界模型的认知不确定性进行了广泛的实证研究。更具体地说，我们在两种不同范式的主流模型之间，对事后认知不确定性估计方法（最后一层拉普拉斯近似与集成方法）进行了受控比较：1）LeWorldModel（LeWM）：一种联合嵌入预测架构（JEPA），它无需解码器即可预测下一个嵌入表示；2）DreamerV3：一种循环状态空间模型（RSSM），它从循环潜在状态中重构观测。实验在十个模拟环境中进行……

    arXiv:2611.34830v1 Announce Type: cross  Abstract: World models predict how an environment responds to actions, so that candidate actions can be evaluated before execution. Executed actions, or decisions, taken in this manner are only as reliable as the model's predictive capabilities, and predictions in turn are only reliable within the model's experience. Epistemic uncertainty (EU) is the signal that marks where that experience ends. In this work we offer an extensive empirical study on how training objectives and architectures shape epistemic uncertainty of world models. More precisely, we present a controlled comparison of post-hoc EU (last-layer Laplace and ensembles) between two dominant models of different paradigms: 1) LeWorldModel (LeWM): A joint-embedding predictive architecture (JEPA), which predicts the next embedding without a decoder, and 2) DreamerV3: A recurrent state-space model (RSSM), which reconstructs the observation from a recurrent latent state. In ten simulated 
    
[^18]: 用于非线性函数逼近的梯度TD(λ)

    Gradient TD($\lambda$) for Non-Linear Function Approximation

    [https://arxiv.org/abs/2611.34826](https://arxiv.org/abs/2611.34826)

    本文将梯度TD(λ)算法从线性函数逼近扩展到非线性函数逼近，并继续采用投影贝尔曼误差目标，为现代复杂非线性函数逼近器提供了具有收敛保证的离策略学习算法。

    

    时序差分（TD）学习是强化学习的核心。许多方法使用半梯度TD，这可能会导致发散；或者发展出基于均方贝尔曼误差或TD误差目标的算法，而这些目标难以分析。梯度TD算法通过使用价值函数的投影克服了这些挑战，并提出了收敛的离策略算法。然而，针对λ回报的算法此前仅限于线性函数逼近。在本文中，我们将其扩展到非线性函数逼近，这是非常理想的，因为许多强大的现代函数逼近器都是复杂且非线性的。最近的工作已经通过广义投影贝尔曼误差目标推导出了类似的算法。与之不同，我们继续使用（标准的）投影贝尔曼误差。

    arXiv:2611.34826v1 Announce Type: new  Abstract: Temporal difference (TD) learning is central to reinforcement learning. Many methods use semi-gradient TD, which may diverge, or develop algorithms with mean-squared Bellman or TD-error objectives that are difficult to analyze. Gradient TD algorithms overcome these challenges by using projection of the value function and present convergent off-policy algorithms. However, the algorithm for $\lambda$-return was limited to linear function approximation. Here, we extend the same to non-linear function approximation, which is desirable since many strong modern function approximators are complex and non-linear. Recent works have derived similar algorithms with a generalized projected Bellman error objective. Alternatively, we continue working with the projected Bellman error.
    
[^19]: 路由之前的可路由性：用于经审计的LLM模型选择的可路由性审计协议（RAP）与RouteabilityBench

    Routeability Before Routing: Routeability Audit Protocol (RAP) and RouteabilityBench for Audited LLM Model Selection

    [https://arxiv.org/abs/2611.34825](https://arxiv.org/abs/2611.34825)

    该论文提出在进行LLM路由方法评估之前先审计基准测试的“可路由性”，并发布了可运行的审计协议RAP和基准套件RouteabilityBench，实验发现跨四个基准和66个LLM的通用路由增强手段并不能带来一致的改进。

    

    LLM路由的目标是，针对每个提示，从模型组合中选出最有可能正确回答的模型。这只有在基准测试包含足够多的提示特定差异、使不同模型能在不同输入上各自成为最佳时才有意义。然而，在采用粗粒度评分的LLM基准测试中，许多模型往往在最佳得分上并列，且单一的 最佳模型可能已几乎没有改进空间。因此，我们提出了一种“可路由性优先”的审计方法，具体实现为RAP与RouteabilityBench，一个可运行的声明验证软件包。该审计是回顾性的：它在基准测试—模型组合评分矩阵已经存在之后、在将路由器比较结果解读为一般性证据之前运行，而非对尚未评估的模型组合进行认证。在四个基准测试和66个LLM上，诸如更大嵌入集合、降维、选择性弃权、聚类和更强回归器等通用路由增强手段并不能一致地（摘要在此处截断）

    arXiv:2611.34825v1 Announce Type: new  Abstract: LLM routing aims to choose, for each prompt, which model in a portfolio is most likely to answer correctly. This only makes sense when the benchmark contains enough prompt-specific variation for different models to be best on different inputs. On coarse-scored LLM benchmarks, however, many models often tie for the best score, and the single best model may already leave little room for improvement. We therefore propose a routeability-first audit, instantiated as RAP and RouteabilityBench, a runnable claim-verification package. The audit is retrospective: it runs once the benchmark-portfolio score matrix exists and before router comparisons are interpreted as general evidence, rather than certifying a portfolio that has not yet been evaluated. Across four benchmarks and 66 LLMs, generic routing enhancements such as larger embedding sets, dimensionality reduction, selective abstention, clustering, and stronger regressors do not consistently
    
[^20]: LoopTitan：参数重访能为混合专家模型带来什么？

    LoopTitan: What Does Parameter Revisit Buy a Mixture-of-Experts?

    [https://arxiv.org/abs/2611.34821](https://arxiv.org/abs/2611.34821)

    研究发现循环MoE架构中参数重访本身几乎没有带来任何收益，其报告的性能提升实际来自网络形状和残差缩放等设计选择，而MoE本身已具备在固定参数量下用专家换深度的自由。

    

    循环混合专家架构在不同深度上重复使用层参数，这种设计在不增加参数的情况下增加了有效深度。近期的循环MoE研究报告称，在匹配的预算（参数量、FLOPs和KV缓存）下，其性能显著优于标准MoE。然而，所报告的性能提升仍然将参数重访与其他设计选择（如网络形状和残差缩放）混淆在一起。我们追问：参数重访本身能带来什么？通过使用匹配的非重访对照组，我们发现参数重访本身几乎没有带来任何收益。我们认为，重访只是提供了一种MoE本已拥有的自由，即在固定参数量的条件下用专家换深度（更多层，每层更少专家）。首先，在匹配预算下，我们分解了最先进的循环方案，并将重访放在最后应用，结果发现一个非重访对照组保留了全部收益，其中大部分来自网络形状，其余来自残差缩放。我们证明这种缩放起到了类似降低学习率的作用。

    arXiv:2611.34821v1 Announce Type: new  Abstract: Looped Mixture-of-Experts (MoE) architectures revisit layer parameters across depths. This design increases effective depth without adding parameters. Recent looped MoE studies report substantial gains over the standard MoE at matched budgets (parameters, FLOPs and KV cache). However, the reported gains still confound parameter revisit with other design choices, e.g., network shape and residual scale. We ask, what does parameter revisit itself buy? Using matched non-revisit controls, we find that parameter revisit itself buys almost nothing. We argue that revisit only offers a freedom an MoE already has, i.e., trading experts for depth (more layers, fewer experts each) at fixed parameters. First, at matched budgets, we decompose the state-of-the-art looped recipe, with revisit last. A non-revisit control retains all the gain. Most comes from network shape, and the rest from residual scale. We prove this scale acts as a lower learning rat
    
[^21]: 面向可变团队规模与智能体失效的多无人机数据采集深度强化学习

    Deep Reinforcement Learning for Multi-UAV Data Harvesting Under Variable Team Sizes and Agent Loss

    [https://arxiv.org/abs/2611.34818](https://arxiv.org/abs/2611.34818)

    本文提出F-MADDPG，一种灵活的MADDPG变体，能够在传感器与无人机数量可变的场景下进行数据采集，并对任务中智能体的灾难性失效保持鲁棒性。

    

    当有限的通信基础设施阻碍无线传感器上传其测量数据时，从这些传感器收集数据可能极具挑战性。在此类场景下，无人驾驶飞行器（UAV）可以利用其机动性在本地采集数据，而协同的无人机团队则能够为偏远地区提供高效且冗余的覆盖。多智能体强化学习（MARL）使这样的团队得以学习去中心化的协调策略。然而，为不同的团队规模和传感器数量分别训练策略成本高昂，且面向固定规模的策略可能容易受到智能体故障的影响。本工作提出了F-MADDPG，这是多智能体深度确定性策略梯度（MADDPG）的一种灵活变体，支持在无线传感器和无人机数量可变的情况下进行数据采集。该框架还能够应对任务期间因意外灾难性故障而永久失去智能体的情况。我们将F-MADDPG与MADDPG基线进行了对比实验。

    arXiv:2611.34818v1 Announce Type: new  Abstract: Collecting data from wireless sensors can be challenging when limited communication infrastructure prevents them from offloading their measurements. In such settings, unmanned aerial vehicles (UAVs) can exploit their mobility to harvest data locally, while coordinated UAV teams can provide efficient and redundant coverage of remote areas. Multi-agent reinforcement learning (MARL) enables such teams to learn decentralized coordination policies. However, training separate policies for different team and sensor populations is costly, and fixed-population policies may be vulnerable to agent failures. This work proposes F-MADDPG, a flexible variant of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) that supports data harvesting with varying numbers of wireless sensors and UAVs. The framework also accommodates unexpected catastrophic failures that permanently remove agents during a mission. We compare F-MADDPG with a MADDPG baseline th
    
[^22]: 将架构作为优化先验进行迁移

    Transferring Architecture as an Optimization Prior

    [https://arxiv.org/abs/2611.34816](https://arxiv.org/abs/2611.34816)

    该论文提出将网络的初始几何视为一种优化先验，并证明在训练前让目标网络与一个不传递特征、不接收梯度、不使用标签的随机初始化引导网络的表示几何对齐，可以通过跨架构迁移优化“命运”来使目标网络受益。

    

    初始化决定了模型的优化“命运”，其作用远不止是设置权重的尺度。它在输入数据上建立了一种基本的初始几何结构，决定了诸如哪些样本被视为相近、以及梯度下降会自然优先细化哪些表示等方面。在本文中，我们将网络的初始几何视为一种优化先验，并探讨网络能否通过参考一个完全不同的架构来改变自身的优化先验。为研究这一问题，我们采用了一种初始化方法：在任何下游训练开始之前，将目标网络与一个随机初始化的引导网络的表示几何进行对齐。至关重要的是，与引导网络的对齐无法迁移已学习的特征、从不接收梯度、也不使用标签。我们证明，这种“命运”的跨架构迁移能够使目标网络受益。

    arXiv:2611.34816v1 Announce Type: cross  Abstract: Initialization determines the optimization "fate" of a model, going far beyond just setting the scale of its weights. It establishes a fundamental initial geometry over the input data, dictating certain aspects such as which examples are considered close and which representations gradient descent will naturally refine first. In this paper, we view the initial geometry of a network as an optimization prior and ask whether a network can change its optimization prior by referencing a completely different architecture. To investigate this question, we use an initialization procedure where a target network is aligned to the representational geometry of a randomly initialized guide network before any downstream training occurs. Crucially, alignment to the guide network cannot transfer learned features, never receives gradients, and does not use labels. We demonstrate that this cross-architecture transfer of "fate" benefits the target network
    
[^23]: 用代码解决视觉问题：编码智能体是比视频模型更强的视觉推理器

    Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models

    [https://arxiv.org/abs/2611.34813](https://arxiv.org/abs/2611.34813)

    该论文在完全相同的输入、输出规范和评估器条件下，让编码智能体通过编写程序渲染答案视频来与视频生成模型直接对比，证明了编码智能体是比视频模型更强大的视觉推理器。

    

    视频生成模型越来越多地被当作视觉推理器来评估：给定第一帧和自然语言提示，模型能否生成执行指定任务的视频？这一范式被认为具有优势，因为推理是视觉媒介原生的，直接在帧中展开，而不是通过语言或代码。然而，这种原生路线是否确实是通往视觉推理的更可靠路径，尚未在相同条件下与符号化替代方案进行过对比检验。因此，我们在完全相同的输入、输出规范和评估器条件下，向两种媒介提出相同的任务：编码智能体不是对像素进行去噪，而是编写一个程序来渲染答案视频，再由未经修改的基于规则的评估器进行评分。在VBVR-Pro-Bench（Xu et al., 2026）上——一个包含500个实例、跨越五个认知类别、具有符号化真值的测试套件——编码智能体接收与视频模型相同的第一帧和提示，实验结果表明编码智能体展现出了更强的视觉推理能力。

    arXiv:2611.34813v1 Announce Type: cross  Abstract: Video generation models are increasingly evaluated as visual reasoners: given a first frame and a natural-language prompt, can the model produce the video in which the specified task is carried out? The paradigm is considered advantageous because the reasoning is native to the visual medium, unfolding directly in the frames rather than through language or code. Whether this native route is in fact the more reliable path to visual reasoning has not been tested against a symbolic alternative under the same conditions. We therefore pose the same tasks, with identical inputs, output specification and evaluator, to both media: rather than denoising pixels, a coding agent writes a program that renders the answer video, which an unmodified rule-based evaluator scores. On VBVR-Pro-Bench (Xu et al., 2026), a 500-instance suite spanning five cognitive categories with symbolic ground truth, the agent receives the same first frame and prompt as th
    
[^24]: 深度神经网络中各层的Barron平滑性

    Barron Smoothness for Layers in Deep Neural Networks

    [https://arxiv.org/abs/2611.34805](https://arxiv.org/abs/2611.34805)

    该论文提出用Barron范数的逐层推广来等价刻画深度神经网络逐层最优离散化问题，并通过优化景观分析证明局部极小值满足类似自适应样条与有限元方法中的等分布原理。

    

    为了界定深度神经网络的逼近误差，我们将该问题分解为两个步骤：首先为每一层选择最优的无限宽度极限，其次对所有层进行最优离散化以获得有限尺寸的网络。我们证明，在期望意义下，后一个问题可由Barron范数的逐层推广等价刻画。此外，我们对优化景观进行了分析，并表明局部极小值由一种等分布原理所刻画，这与自适应样条和有限元逼近中的类似思想相呼应。

    arXiv:2611.34805v1 Announce Type: new  Abstract: In order to bound approximation errors for deep neural networks, we split the problem into two steps: First the selection of optimal infinite width limits for each layer and second an optimal discretization of all layers to obtain finite sized networks. We show that in expectation, the latter problem is equivalently characterized by a layer wise generalization of Barron norms. Moreover, we provide an analysis of the optimization landscape and show that local minima are characterized by an equidistribution principle akin to similar ideas in adaptive spline and finite element approximation.
    
[^25]: 引导即方差匹配

    Guidance as Variance Matching

    [https://arxiv.org/abs/2611.34801](https://arxiv.org/abs/2611.34801)

    论文揭示扩散模型中的引导并非改善后验均值预测，而是恢复了被低估的方差，并提出无需调参的方差匹配引导闭式规则，可准确复现十一个主流模型的手工引导尺度。

    

    引导机制将扩散模型的预测推离一个较弱的参考预测。它显著提升了样本质量，但其内在机制至今仍缺乏深入理解。我们首先发现，在实际使用的引导尺度下，引导并不能改善对后验均值（即模型的训练目标）的预测；它反而使预测变差，但生成的样本却变得更好。其次，后验均值预测的方差低于数据本身的方差，而我们发现引导恰好能恢复这一方差。基于此，我们提出了方差匹配引导：一种闭式规则，通过将引导后预测的方差与数据的方差进行匹配来自动选择引导尺度。无需任何调参，该规则便能复现十一个广泛使用的模型在图像、视频、音频、动作和世界生成等任务中人工挑选的引导尺度；据我们所知，此前没有任何关于引导的解释能够预测出这些尺度。当按噪声水平逐级应用时，它还能产生一个无需调节超参数、且更受用户偏好的引导尺度调度方案。

    arXiv:2611.34801v1 Announce Type: cross  Abstract: Guidance repels the prediction of a diffusion model away from a weaker reference prediction. It substantially improves sample quality, but its mechanism remains poorly understood. We first find that, at the scales used in practice, guidance does not improve the prediction of the posterior mean (the model's training target); it makes the prediction worse, while samples improve. Second, posterior-mean predictions have less variance than the data, and we find that guidance restores it. We introduce variance-matched guidance: a closed-form rule that picks the guidance scale by matching the variance of the guided predictions to that of the data. Without tuning, it recovers the hand-picked scales of eleven widely used models across image, video, audio, motion, and world generation; to our knowledge, no previous explanation of guidance predicts them. Applied per noise level, it yields a schedule without tuned hyperparameters that users prefer
    
[^26]: Split-RL：强化学习中的局部冲突消解

    Split-RL: Local Conflict Resolution in Reinforcement Learning

    [https://arxiv.org/abs/2611.34799](https://arxiv.org/abs/2611.34799)

    Split-RL是一种基于梯度提升树的强化学习算法，通过将引导标签集成到树构建中，把矛盾反馈路由到互不相交的状态空间区域，实现冲突信号的局部隔离并防止信号干扰。

    

    强化学习策略的训练过程通常会混合在空间和时间上局部化的矛盾反馈。例如，机器人在开阔区域可能优先考虑速度，而在狭窄缝隙中则优先考虑精度。我们提出了Split-RL，这是一种基于梯度提升树（GBT）的算法，它利用引导标签在局部隔离这些冲突信号。标准神经网络缺乏实现这一目标的结构机制，因为其参数共享来自所有来源的反馈。虽然GBT提供了为每个目标显式划分状态空间的归纳偏置，但标准的树拟合过程会在矛盾梯度信号到达叶子节点之前将其聚合。Split-RL利用这一偏置，将引导标签集成到树构建过程中，将这些更新路由到互不相交的状态空间区域，从而防止信号干扰。此类标签可以来自启发式方法、传感器事件或现有的基于规则的系统。

    arXiv:2611.34799v1 Announce Type: new  Abstract: Reinforcement learning (RL) policies are often trained by mixing contradictory feedback that is localized in space and time. For example, a robot may prioritize speed in open areas versus precision in tight gaps. We introduce Split-RL, a Gradient Boosting Trees (GBT) algorithm that uses guidance labels to locally isolate these conflicting signals. Standard neural networks (NNs) lack the structural mechanism to do this, as their parameters share feedback from all sources. While GBT provides the inductive bias to explicitly partition the state-space per objective, standard tree-fitting procedures aggregate contradictory gradient signals before they reach the leaves. Split-RL leverages this bias to integrate guidance labels into tree construction, routing these updates into disjoint state-space regions and preventing signal interference. Such labels can arise from heuristics, sensor-derived events, or existing rule-based systems, allowing S
    
[^27]: PersonaFactory：基于合成人格库的风格化用户提示增强

    PersonaFactory: Stylistic User Prompt Augmentation with Synthetic Persona Bank

    [https://arxiv.org/abs/2611.34785](https://arxiv.org/abs/2611.34785)

    该论文提出PersonaFactory端到端流程，构建了超74万个刻画背景与交流行为的结构化合成人格库，并以85种语言生成约124万条人格风格化对话，解决了合成用户提示风格单一、无法模拟真实用户交流机制的问题。

    

    大型语言模型在AI开发流程中越来越多地取代真人，被用于大规模生成训练对话以及驱动对齐与评估。然而，开箱即用的模型所生成的合成用户提示往往语言单一、风格平淡，无法复现真实的用户-AI交流机制。我们提出了PersonaFactory，这是一个端到端的流程，它可以生成详细的用户人格画像，并从这些人格的视角对现有用户提示进行增强。该方法产出了一个包含超过74万个结构化合成人格的库，详细刻画了每个角色的背景与交流行为；同时生成了一个覆盖85种语言、约124万条人格风格化合成对话的语料库。我们评估了个人特征对风格采纳和意图保留的影响，使用LLM裁判和启发式指标来衡量增强的保真度，以及其对助手人格定制的下游影响。

    arXiv:2611.34785v1 Announce Type: cross  Abstract: Large language models increasingly replace real people in the AI development pipeline, as they are used to generate training dialogues at scale and to drive alignment and evaluation. However, out-of-the-box models produce monolingual and stylistically flat texts as synthetic user prompts, failing to replicate the real user-AI communication mechanics. We present PersonaFactory, an end-to-end pipeline that generates detailed user personas and uses them to augment existing user prompts from their point of view, producing a bank of over 740000 structured synthetic personas detailing their background and communication behaviors, and a corpus of roughly 1.24 million persona-styled synthetic conversations spanning 85 languages. We evaluate the effect of personal features on style adoption and intent preservation, using an LLM-judge and heuristic metrics to assess augmentation fidelity together with the downstream impact on assistant persona c
    
[^28]: 脑肿瘤分类深度学习架构的比较研究

    Comparative Study of Deep Learning Architectures for Brain Tumour Classification

    [https://arxiv.org/abs/2611.34781](https://arxiv.org/abs/2611.34781)

    本研究比较了六种深度学习架构在多源MRI数据集上脑肿瘤四分类任务中的表现，并提出了利用MD5哈希和感知哈希去除重复图像的防数据泄漏流程，以更真实地评估模型的泛化能力。

    

    从磁共振成像（MRI）中对脑肿瘤进行准确且早期的分类对于治疗规划至关重要，然而许多已发表的深度学习研究是在小型、单源数据集上训练和测试的，其报告的准确率在这些数据集之外可能并不成立。本研究比较了六种深度学习架构在四分类任务（健康、脑膜瘤、胶质瘤和垂体瘤）上的表现：基线卷积神经网络（CNN）、基于注意力的CNN、ShuffleNet，以及通过卷积块注意力模块（CBAM）增强的EfficientNet-B0、ResNet-50和Xception。这些模型在由四个公开MRI数据集汇编而成的语料库上进行训练。由于这些数据集不含患者标识信息，研究开发了一种防泄漏的处理流程：使用MD5哈希去除完全重复的图像（550张），使用感知哈希去除近似重复的图像（8,024张），并对视觉相似的图像进行……

    arXiv:2611.34781v1 Announce Type: cross  Abstract: Accurate and early classification of brain tumours from magnetic resonance imaging (MRI) is essential for treatment planning, yet many published deep learning studies are trained and tested on small, single-source datasets, reporting accuracies that may not hold outside these datasets. This study compares six deep learning architectures for four-class classification (healthy, meningioma, glioma and pituitary tumour): a baseline convolutional neural network (CNN), an attention-based CNN, ShuffleNet, and EfficientNet-B0, ResNet-50 and Xception augmented with the Convolutional Block Attention Module (CBAM). The models were trained on a corpus assembled from four publicly available MRI datasets. Because these datasets carry no patient identifiers, a leakage-aware pipeline was developed that removes exact duplicates using MD5 hashing (550 images) and near-duplicates using perceptual hashing (8,024 images), and assigns visually similar image
    
[^29]: 面向住宅电池集群的自适应可靠性备用容量

    Adaptive Reliability Reserves for Residential Battery Fleets

    [https://arxiv.org/abs/2611.34779](https://arxiv.org/abs/2611.34779)

    该论文提出用县级风险阈值取代固定20%电池备用容量的自适应策略，在休斯顿2024年重放模拟中将五小时备用覆盖率从75.0%提升至86.5%并带来每户约10.2美元的年化模型收益，同时发布了结合真实电价、县级停电数据与模拟住宅的HomeFleet-ERCOT数据集，并对比了独立优化与PPO强化学习训练的表现。

    

    在2024年休斯顿重放模拟中，用县级风险阈值替代固定的20%电池备用容量，可将观测到的五小时备用覆盖率从75.0%提升至86.5%，并在模型中为每户增加10.2美元的年化收益（配对95%置信区间：+1.7至+21.7美元）。配对的覆盖率增益区间包含零。这是条件模拟结果，而非实测的公司可靠性指标。住宅电池之所以面临这一权衡，是因为市场放电会消耗原本需要留给家庭停电期间使用的能量。我们发布了HomeFleet-ERCOT数据集，将观测到的电价和县级停电数据与经过辨识的模拟住宅及因果控制器输入相结合。我们比较了独立住宅优化与21次近端策略优化（PPO）训练运行；十个承诺成本候选方案的备用覆盖率介于31.1%至96.3%之间。其中，10个方案的模式化收益低于固定20%备用容量的方案，另有3个方案的覆盖率也更低。（原文在此处截断）

    arXiv:2611.34779v1 Announce Type: new  Abstract: In a Houston 2024 replay, replacing a fixed 20 percent battery reserve with a county-risk threshold raises observed five-hour backup coverage from 75.0 percent to 86.5 percent and adds USD 10.2 in modeled annualized margin per home (paired 95 percent interval: USD +1.7 to USD +21.7). The paired coverage-gain interval includes zero. This is a conditional simulation result, not measured company reliability. Residential batteries face this tradeoff because market discharge consumes energy also needed for household outages. We release HomeFleet-ERCOT, combining observed prices and county outages with identified simulated homes and causal controller inputs. We compare independent-home optimization with 21 proximal policy optimization (PPO) training runs; ten promise-cost candidates span 31.1 percent-96.3 percent backup coverage. Of these, 10 have lower modeled margin than fixed 20 percent, and 3 also have lower coverage. Chronological, geogra
    
[^30]: 当每次轨迹采样都失败时：将学习者条件化的选项拼接进长程智能体强化学习

    When Every Rollout Fails: Splicing Learner-Conditioned Options into Long-Horizon Agent RL

    [https://arxiv.org/abs/2611.34777](https://arxiv.org/abs/2611.34777)

    提出SPLICE方法，在线地从全失败的当前策略轨迹中拼接由学习者条件化的局部选项作为训练目标，解决了长程智能体强化学习中全失败组导致无奖励策略梯度的问题。

    

    基于可验证奖励的分组强化学习在长程智能体任务上经常产生全失败组：每条采样轨迹都获得零奖励，使得组相对类方法缺乏基于奖励的策略梯度。一种常见的补救措施是由更强的模型提供完整解答，但完整轨迹代价高昂，且会让学习者暴露于冗长的离策略历史中。我们提出SPLICE，它在线地从失败的当前策略轨迹中构建局部训练目标。对于每个全失败组，一个提议模型在当前策略轨迹中选择一个拼接点，并从该点生成若干有界选项。我们利用重复的当前策略延续来估计每个选项的价值，将正向估计转换为按回报加权的局部目标，并在这些选项上拟合当前策略。我们在两个长程智能体任务（基于SWE-bench的软件工程等）上，跨两个策略家族对SPLICE进行了评估。

    arXiv:2611.34777v1 Announce Type: cross  Abstract: Group-based reinforcement learning with verifiable rewards often yields all-fail groups on long-horizon agent tasks: every sampled trajectory receives zero reward, leaving group-relative methods without a reward-based policy gradient. A common remedy supplies complete solutions from a stronger model, but full trajectories are expensive and expose the learner to long off-policy histories. We introduce SPLICE, which constructs local training targets from failed current-policy trajectories online. For each all-fail group, a proposal model selects a splice point in a current-policy trajectory and generates several bounded options from that point. We estimate each option's value using repeated current-policy continuations, convert positive estimates into a return-weighted local target, and fit the current policy on the options. We evaluate SPLICE across two policy families on two long-horizon agent tasks, Software Engineering with SWE-bench
    
[^31]: 何时信任PFN：检测表格基础模型中的有害分布偏移

    When to Trust a PFN: Detecting Harmful Shift in Tabular Foundation Models

    [https://arxiv.org/abs/2611.34767](https://arxiv.org/abs/2611.34767)

    该论文提出了CALD，一种轻量级无监督监控方法，通过微调解码器最大化与测试预测的不一致性来检测表格基础模型（如TabPFN和TabICL）的有害分布偏移，在14个数据集上表现达到或超越基线，且对良性偏移的误报明显更具鲁棒性。

    

    表格先验拟合网络（PFNs）在各种表格分类问题上表现良好，但在部署时若分布发生偏移，可能会无声地失败。我们提出了CALD（校准锚点损失不一致性），这是一种轻量级的无监督监控器，当PFN可能失败时会触发警报发出警告。通过微调解码器以最大化与测试集预测的不一致性，同时在分布内（ID）锚点集上保持准确性，CALD通过由此产生的锚点损失来识别有害的分布外（OOD）批次，且无需对测试批次进行标注。我们证明，在足够严重的分布偏移下，微调后的锚点损失期望严格低于ID评估下的期望，从而得到一个单侧检验。在涵盖表格和结构化分布偏移的14个数据集上，无论是在TabPFN还是TabICL上，CALD在有害偏移上的表现与基线相当或更优，并且对良性偏移的误报具有明显更强的鲁棒性。

    arXiv:2611.34767v1 Announce Type: new  Abstract: Tabular prior-fitted networks (PFNs) work well across tabular classification problems but can fail silently at deployment when distributions shift. We propose CALD (Calibrated Anchor-Loss Disagreement), a lightweight unsupervised monitor that triggers an alarm to warn when a PFN might fail. By finetuning the decoder to maximize disagreement with the test-set prediction while maintaining accuracy on an ID anchor set, CALD identifies harmful OOD batches through the resulting anchor loss, requiring no labels on test batches. We prove that the post-finetune anchor loss has a strictly lower expectation under sufficiently severe shift than under ID eval, yielding a one-sided test. Across 14 datasets covering tabular and structured distribution shift, on both TabPFN and TabICL, CALD matches or outperforms baselines on harmful shifts and is markedly more resilient to benign-shift false alarms.
    
[^32]: 一个模型，多种模式：为创造性生成演化系统提示词集成

    One Model, Many Modes: Evolving System Prompt Ensembles for Creative Generation

    [https://arxiv.org/abs/2611.34760](https://arxiv.org/abs/2611.34760)

    该论文提出免训练框架 SEPO，通过演化系统提示词集成并结合质量约束下的贪心最大-最小选择来缓解偏好对齐大语言模型的模式坍缩，从而生成兼具高质量、新颖性与多样性的创造性内容。

    

    偏好对齐的大语言模型（LLM）可能出现模式坍缩现象，即其输出分布集中于一小部分高奖励响应附近。这限制了创造性生成，因为创造性生成需要同时具备质量、新颖性和多样性。现有的测试时方法往往强调多样性，而对单个响应是否兼具高质量与新颖性关注较少。这一区别在实际应用中很重要，因为用户可能只想要一条令人信服的、原创的响应。我们将创造性生成形式化为一个多目标分布优化问题，并引入 SEPO——一个免训练的系统提示词集成优化框架。该框架将提示词演化与贪心最大-最小选择相结合，在质量约束下识别行为上各不相同的系统提示词，然后在推理时为每个响应均匀采样一个提示词。在四个质量要求逐步提高的创造性生成任务以及三个开源权重模型上（摘要在此处截断）……

    arXiv:2611.34760v1 Announce Type: cross  Abstract: Preference-aligned LLMs can exhibit mode collapse, concentrating their output distributions around a narrow set of high-reward responses. This limits creative generation, which requires quality, novelty, and diversity. Existing test-time methods often emphasize diversity, giving less attention to whether individual responses are both high-quality and novel. This distinction matters in practical use, where users may seek a single compelling, original response. We formalize creative generation as a multi-objective distributional optimization problem and introduce SEPO, a training-free framework for System Prompt Ensemble Optimization. The framework combines prompt evolution with greedy max-min selection to identify behaviorally distinct system prompts under a quality constraint, then uniformly samples one prompt per response at inference time. Across four creative generation tasks with increasing quality requirements and three open-weigh
    
[^33]: 通过透镜解读微调差异：用于后门检测的模型差异分析方法

    Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor Detection

    [https://arxiv.org/abs/2611.34758](https://arxiv.org/abs/2611.34758)

    该论文提出了一个基于token读出的统一评分系统来评估各类模型差异后门检测方法，并引入J透镜与J++透镜改进权重和激活分析方法，其中最强的ΔW J++方法在12个设置中的11个成功将后门相关token排入前十。

    

    我们引入了一个基于token读出的统一评分系统，用于评估基于权重、激活和输出的模型差异分析方法在微调过程中的后门检测效果。我们还通过引入J透镜及其改进版J++透镜，改进了现有的基于权重和激活的分析方法。我们在将后门模型与其微调前的基座模型进行比较的实验中测试了该设置。结果表明，引入J++透镜在所有设置中都提升了基于权重的方法的表现，而J透镜则改进了其在单层内的排名。最强的单一方法是ΔW J++，即通过基座模型的J++透镜解码的权重差异，该方法在12个设置中的11个里将命名后门的token排进了前十。此外，我们还展示了基于输出和激活的方法（如ADL和Diff Mining）在后门模型出现某些情况时很容易失效。

    arXiv:2611.34758v1 Announce Type: cross  Abstract: We introduce a common scoring system based on token readouts to evaluate weight, activation and output-based model diffing methods for backdoor detection in fine-tuning processes. We also improve existing weight and activation based methods by incorporating the $J$-lens and its refinement, the $J{+}{+}$-lens. We test our setup in experiments where we compare a backdoored model with the base model from which it was fine-tuned. Our results show that incorporating the $J{+}{+}$-lens improves the weight-based method in all our settings, while the $J$-lens improves its ranks within single layers. The strongest single method is $\Delta W J{+}{+}$, the weight difference decoded through the $J{+}{+}$-lens of the base model, which ranks a token naming the backdoor among its first ten tokens in 11 of our 12 settings. Moreover, we show that output and activation based methods such as ADL and Diff Mining can easily break if the backdoored model is
    
[^34]: 收集以求覆盖，回放以证据：面向长程智能体的自适应高效在线策略蒸馏

    Collect for Coverage, Replay with Evidence: Adaptive and Efficient On-Policy Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2611.34756](https://arxiv.org/abs/2611.34756)

    该论文提出CARE-OPD框架，通过学生种子教师补全和响应级置信度门控扩大历史覆盖并过滤不可靠教师信号，同时以证据自适应地回放轨迹中任意深度的关键决策，从而实现面向长程智能体的自适应且高效的前缀回放式在线策略蒸馏。

    

    arXiv:2611.34756v1 公告类型：cross。摘要：对于长程智能体而言，完全在线的在线策略蒸馏（OPD）代价高昂，因为每次更新都依赖全新的环境交互轨迹。前缀回放提供了一种高效的替代方案：通过从存储的历史中采样当前学生模型的响应，减少了长尾轨迹带来的空闲时间，并使每条收集到的轨迹能够支持多次更新。然而，我们对完整前缀回放流程的分析发现，在长程场景中存在两个局限。在轨迹收集阶段，仅由教师生成的轨迹会遗漏由学生引入的历史，而更广泛的历史覆盖又可能降低教师的可靠性。在轨迹复用阶段，固定的基于位置的调度无法识别可能出现在轨迹任意深度处的关键决策。我们提出 CARE-OPD（Collect for coverAge, Replay with Evidence，为覆盖而收集、以证据回放）来解决这两个局限。学生种子教师补全与响应级置信度门控在扩展历史覆盖的同时过滤不可靠内容（摘要在此处被截断）。

    arXiv:2611.34756v1 Announce Type: cross  Abstract: Fully online on-policy distillation (OPD) is costly for long-horizon agents because each update depends on fresh environment rollouts. Prefix replay offers an efficient alternative by sampling current-student responses from stored histories, reducing idle time for long-tail rollouts and allowing each collected trajectory to support multiple updates. However, our analysis of the complete prefix replay pipeline identifies two limitations in long-horizon settings. In trajectory collection, teacher-only trajectories miss student-induced histories, while broader history coverage can reduce teacher reliability. In trajectory reuse, fixed position-based schedules cannot identify critical decisions that may occur at any trajectory depth. We introduce CARE-OPD (Collect for coverAge, Replay with Evidence) to address both limitations. Student-seeded teacher completions and response-level confidence gating expand history coverage while filtering u
    
[^35]: BRIAR：分叉引导推理树上的鞅信念平滑

    BRIAR: Martingale Belief Smoothing on Bifurcation-Guided Reasoning Trees

    [https://arxiv.org/abs/2611.34746](https://arxiv.org/abs/2611.34746)

    该论文提出BRIAR框架，利用推理轨迹上前缀通过概率构成有界Doob鞅、其信用方差在脱轨与吸收边界处消失这一鞅几何特性，通过分叉引导的树状采样与推理，解决长链推理中困难问题早期失误脱轨和可解问题冗余验证的两大病态。

    

    带有可验证奖励的强化学习（RLVR）已迅速推动了长链推理的发展，然而标准的策略优化器在整条轨迹上盲目地分配采样计算量与梯度信用。在困难问题上，早期的失误会使剩余的推理轨迹脱轨；而在可解决的问题上，即使答案早已确定，策略仍会继续生成冗余的验证。我们将这两种病态追溯到同一个结构性事实：在策略内验证器下，推理轨迹上的真实前缀通过概率 $V^\pi(u)$ 构成一个取值于 $[0,1]$ 的 Doob 鞅，其可学习的信用方差被严格守恒且上界为 $V^\pi(u)(1-V^\pi(u))$，并在脱轨（$V^\pi\to 0$）与吸收（$V^\pi\to 1$）两个边界处消失。我们提出了 BRIAR（基于自适应推理树的分叉引导采样与推理），这是一个直接围绕该鞅几何结构构建的树状 RLVR 框架。首先，B（摘要原文在此处截断）

    arXiv:2611.34746v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has rapidly advanced long-chain reasoning, yet standard policy optimizers distribute rollout compute and gradient credit blindly across a trajectory. On hard problems, an early misstep derails the remaining trace, whereas on solvable problems, the policy keeps generating redundant verification long after the answer is already determined. We trace both pathologies to a single structural fact: under an on-policy verifier, the true prefix pass probability $V^\pi(u)$ along a reasoning trace forms a $[0,1]$-bounded Doob martingale whose learnable credit variance is strictly conserved and bounded above by $V^\pi(u)(1-V^\pi(u))$, vanishing at both the derailed ($V^\pi\to 0$) and absorbed ($V^\pi\to 1$) boundaries. We introduce BRIAR (Bifurcation-guided Rollout and Inference on Adaptive Reasoning trees), a tree-structured RLVR framework built directly around this martingale geometry. First, B
    
[^36]: 模型是否收到了正确的动作？导航世界模型中动作接口的审计与修复

    Was the Model Given the Right Action? Auditing and Repairing Action Interfaces in Navigation World Models

    [https://arxiv.org/abs/2611.34739](https://arxiv.org/abs/2611.34739)

    该论文审计了导航世界模型（NWM）训练与推演代码之间相互作用的动作输入差异，发现无需重新训练、仅修正输入准备方式即可将视觉预测误差降低42%，并改变了模型间的性能排名，证明相互作用的输入失配必须联合检查。

    

    世界模型通过从动作预测未来观测来进行评估。这些评估假设动作的准备方式与训练时相同，但未经检查的不匹配可能使模型看起来不如其实际水平准确。我们在导航世界模型（NWM）中审计了这一假设，发现其训练代码与推演代码之间存在三个相互作用的差异。在不重新训练的情况下修正输入，使平均视觉预测误差降低了42%（在RECON导航数据集上使用DreamSim测量）。该修正还在探索性分析中改变了NWM检查点的排名，并消除或减少了两个替代模型所测得的优势。仅修正归一化偏移反而会增加误差，这表明相互作用的失配必须一起检查。为了测试对输入准备的敏感性是否超出NWM的范围，我们故意在两个替代模型中引入编码错误；两者都变得……

    arXiv:2611.34739v1 Announce Type: cross  Abstract: World models are evaluated by predicting future observations from actions. These evaluations assume that actions are prepared the same way as during training, but an unchecked mismatch can make a model appear less accurate than it is. We audit this assumption in Navigation World Models (NWM) and find three interacting differences between its training and rollout code. Correcting the inputs, without retraining, reduces mean visual prediction error by 42%, measured with DreamSim on the RECON navigation dataset. The correction also changes the ranking of NWM checkpoints in an exploratory analysis and removes or reduces two alternative models' measured advantages. Correcting only the normalization offset instead increases error, showing why interacting mismatches must be checked together. To test whether sensitivity to input preparation extends beyond NWM, we deliberately introduce encoding errors into two alternative models; both become l
    
[^37]: 从NP难到精确可解：样本丰裕性带来的转变

    From NP-Hardness to Exact Tractability Through Sample Abundance

    [https://arxiv.org/abs/2611.34734](https://arxiv.org/abs/2611.34734)

    该论文证明了对于最坏情况下NP难的带噪声奇偶校验拟合问题，只需 $O(n/(\log n)^{1/(k-2)})$ 次来自同一来源的额外采样，就能以高概率在 $O(n\log^2 n)$ 的工作量内实现精确优化，表明样本的丰富性可以在不改变问题本身的情况下将NP难问题变为精确可解。

    

    额外的观测可以在不改变目标函数或可行集的情况下，使精确优化变得可解。我们在带噪声的奇偶校验拟合问题上建立了这一现象——这是一个最坏情况下NP难的问题——其设定包括均匀采样的秩一符号张量元素、独立的公平因子符号以及独立的误差。对于固定的交互阶数 $k\ge3$、每块 $n$ 个符号、对数平均度以及已知的信道偏置，来自同一来源的 $O(n/(\log n)^{1/(k-2)})$ 次额外读取，足以以高概率在 $O(n\log^2 n)$ 的最坏情况工作量内实现精确优化。其主导系数对于采用独立读取标记、深度为 $O(\log\log n)$ 的局部估计器的弱恢复而言是紧致的。一个共同的符号场定律决定了相对于生成源的增益、并列最优解的数量以及后续重加权所需的维度。剩余坐标的几何结构决定了高分赋值的局部验证的锐利半径。

    arXiv:2611.34734v1 Announce Type: new  Abstract: Additional observations can make exact optimization tractable without changing the objective or feasible set. We establish this phenomenon for noisy parity fitting, a worst-case NP-hard problem, with uniformly sampled rank-one sign tensor entries, independent fair factor signs and independent errors. For fixed interaction order $k\ge3$, $n$ signs per block, logarithmic mean degree and known channel biases, $O(n/(\log n)^{1/(k-2)})$ further readings of the same source suffice with high probability for exact optimization in $O(n\log^2 n)$ worst-case work. The leading coefficient is sharp for weak recovery by depth-$O(\log\log n)$ local estimators with independent reading marks. A common signed-field law determines the gain over the generating source, the number of tied optima and the dimension needed for later reweightings. The geometry of the remaining coordinates determines the sharp radius of local verification for high-scoring assignme
    
[^38]: 层次图元网络

    Hierarchical Graph Metanetworks

    [https://arxiv.org/abs/2611.34726](https://arxiv.org/abs/2611.34726)

    该论文提出层次图元网络（Hierarchical Graph Metanetworks），利用神经网络固有的层次结构（神经元—层—块/模块）作为归纳偏置，突破了现有图元网络只能在最细粒度处理每个神经元和权重、计算成本随参数量扩展的瓶颈，从而实现高效的大规模神经网络权重空间学习。

    

    arXiv:2611.34726v1 公告类型：新 摘要：权重空间学习这一新兴领域已经展现出使元网络（即处理神经网络（NN）数据的模型）能够支持机器学习生命周期多个步骤的潜力，涵盖从模型优化到评估与适应。然而，这一潜力尚未在小规模MLP或CNN之外真正实现：目前的大多数模型，包括基于消息传递的图元网络（GMNs），仅在最细粒度层级上运行，处理输入模型的每一个神经元和权重，因此其成本随参数数量的增长而扩展。然而，神经网络本质上是按层次组织的：神经元构成层，层构成块和模块。此外，神经网络的若干属性，如Lipschitz上界和泛化度量，是层级别和块级别量的组合。为了将这一归纳偏置纳入模型，我们提出了层次图元网络。

    arXiv:2611.34726v1 Announce Type: new  Abstract: The emerging field of weight space learning has demonstrated the potential to enable metanetworks, i.e. models that process neural network (NN) data, to support several steps of the machine learning lifecycle, from model optimisation to evaluation and adaptation. However, this potential has yet to materialise beyond small-scale MLPs or CNNs: most current models, including message-passing based graph metanetworks (GMNs), operate exclusively at the finest level of granularity, processing every neuron and weight of the input model. Their cost therefore scales with the number of parameters. Yet NNs are inherently organised hierarchically, with neurons forming layers and layers forming blocks and modules. Additionally, several of their properties, such as Lipschitz upper bounds and generalisation measures, are compositions of layer- and block-level quantities. To incorporate this inductive bias, we introduce Hierarchical Graph Metanetworks (H
    
[^39]: 更大的模型还是更大的表格？Chorus N-gram 模块能替代多少主干网络

    Bigger Model or Bigger Table? How Much Backbone a Chorus N-gram Module Replaces

    [https://arxiv.org/abs/2611.34725](https://arxiv.org/abs/2611.34725)

    该论文通过在 13.4 亿波兰语词元上训练 16M 至 349M 参数的模型，量化了 Chorus N-gram 哈希表模块对 transformer 主干网络的替代作用：当查找表与主干网络规模相当时，每个新增参数最多可替代 0.31 个主干参数，使模型能够以较小规模匹敌 1.16 至 1.34 倍大小的普通模型。

    

    我们探究了 Chorus N-gram 模块能在多大程度上替代 transformer 的主干网络。Chorus 在其输入和注意力机制中加入了基于哈希表存储的词元二元组（bigram）和三元组（trigram）。我们在 13.4 亿词元的 1918 年之前印刷的波兰语文本上训练了 1600 万至 3.49 亿参数的模型，并找出具有相同留出集损失的普通模型所需的规模。针对每种模型分别进行独立的学习率搜索后，Chorus 在 47M 规模下的收益减少了约三分之一。当查找表规模与主干网络大致相当时，Chorus 在 16M 规模下可以匹敌 1.34 倍于其规模的普通模型，在 107M 至 349M 规模下可以匹敌 1.16 至 1.20 倍于其规模的普通模型。使用这些表格时，每个新增参数最多只能替代 0.31 个主干网络参数。Chorus 的训练速度比同等规模的普通模型慢，但比达到相同损失的普通模型快。在事实问答测验上，两者的差异大约相当于同一模型在不同随机种子下的差异。这些结果基于单一语料库、一个 8000 词元的词表以及仅五分之一个训练周期。

    arXiv:2611.34725v1 Announce Type: new  Abstract: We ask how much of a transformer's backbone the Chorus N-gram module replaces. Chorus adds hashed tables of token bigrams and trigrams to its input and attention. We train models of 16M to 349M parameters on 1.34 billion tokens of Polish printed before 1918 and find the size of a plain model with the same held-out loss. A separate learning-rate search for each kind of model cuts Chorus's gain at 47M by about a third. With tables about as large as the backbone, Chorus then matches a plain model 1.34 times its size at 16M and 1.16 to 1.20 times from 107M to 349M. With these tables, each added parameter replaces at most 0.31 backbone parameters. Chorus trains slower than a plain model of its size and faster than one of the same loss. On a fact quiz the two differ about as much as two seeds of one model. The results rest on one corpus, one 8,000-token vocabulary and a fifth of an epoch.
    
[^40]: 有限真实数据下的激光雷达强度重建：固定轮次训练下的架构基准测试

    LiDAR Intensity Reconstruction with Limited Real Data: An Architecture Benchmark under Fixed-Epoch Training

    [https://arxiv.org/abs/2611.34724](https://arxiv.org/abs/2611.34724)

    该论文在固定20轮训练协议下，对九种神经架构在真实数据稀缺、以物理仿真数据补充的激光雷达强度重建任务上进行了系统基准测试，发现UNet-V2在纯真实数据训练下表现最佳，并量化了不同真实数据比例对重建性能的影响。

    

    基于物理推导的仿真可以补充稀缺的真实世界激光雷达强度测量数据，但其有效性取决于学习架构以及可用的真实训练数据量。我们针对从深度、语义标签、入射角和材料反射率重建激光雷达强度这一任务，对九种神经架构进行了基准测试，使用了真实的SemanticKITTI和合成的VoxelScape距离图像数据。所有模型均遵循固定的20轮（epoch）训练协议；该协议控制的是训练轮数，而非各架构之间的计算成本。我们报告了平均绝对误差（MAE）、均方误差（MSE）、峰值信噪比（PSNR）、结构相似性（SSIM）、推理耗时以及频谱能量一致性。此外，我们使用包含0%、1%、5%、10%、20%、50%或100%真实帧的固定规模训练混合数据，对四种选定的架构进行了评估。在仅使用真实数据训练的情况下，UNet-V2取得了已报告结果中最优的重建性能。

    arXiv:2611.34724v1 Announce Type: cross  Abstract: Physics-derived simulation can supplement scarce real-world LiDAR intensity measurements, but its effectiveness depends on the learning architecture and the amount of real training data available. We benchmark nine neural architectures for LiDAR intensity reconstruction from depth, semantic labels, incidence angle, and material reflectance, using real SemanticKITTI and synthetic VoxelScape range-image data. All models follow a fixed 20-epoch protocol; this controls the number of training passes, not the computational cost across architectures. We report mean absolute error (MAE), mean squared error (MSE), peak signal-to-noise ratio (PSNR), structural similarity (SSIM), inference timing, and spectral-energy agreement. We also evaluate four selected architectures with fixed-size training mixtures containing 0\%, 1\%, 5\%, 10\%, 20\%, 50\%, or 100\% real frames. With real-only training, UNet-V2 achieves the best reported reconstruction me
    
[^41]: 因果图模糊性下的鲁棒算法补救

    Robust Recourse under Causal Graph Ambiguity

    [https://arxiv.org/abs/2611.34723](https://arxiv.org/abs/2611.34723)

    针对因果图存在歧义的问题，提出了最坏情况补救和图加权补救两种方法，通过梯度搜索找到在多个候选因果图上均有效的低成本补救行动。

    

    因果算法补救通过建模干预如何改变个体特征，向个体推荐行动以扭转不利的决策结果。现有方法通常假设仅存在单一因果图来描述系统，并考虑因果机制中的不确定性。然而，实际上通常存在多个符合相同数据但对同一干预行动可能预测出不同效果的因果图。某些行动或许能在多个图上都实现补救，但随着潜在因果图集合的增大，单一最优行动可能变得不可行。我们提出了“最坏情况补救”和“图加权补救”两种方法，分别用于寻找在所有候选图上均能成功、或在指定的图权重比例上成功的低成本行动。在假设存在一组合理因果图的前提下，我们对每个图拟合高斯过程结构方程，并采用后验均值-标准差准则来评估行动的成功性。我们使用基于梯度的搜索方法来确定最小成本的行动。

    arXiv:2611.34723v1 Announce Type: new  Abstract: Causal algorithmic recourse recommends actions to reverse unfavorable decisions by modeling how interventions change an individual's features. Existing methods typically assume one causal graph that describes the system and account for uncertainty in the causal mechanisms. However, there usually exist several graphs that fit the same data but can predict different effects for the same action. Certain actions may achieve recourse across multiple graphs, but as the set of potential graphs grows, a single best action may become infeasible.   We propose Worst-Case Recourse and Graph-Weighted Recourse to find a low-cost action that succeeds across all candidate graphs or a specified fraction of graph weight, respectively. Assuming a set of plausible graphs, for each graph we fit Gaussian process structural equations and assess success using the posterior mean--standard-deviation criterion. We use gradient-based search to determine minimal cos
    
[^42]: 基于特权在线策略蒸馏的缺失模态下校准阿尔茨海默病诊断

    Privileged On-Policy Distillation for Calibrated Alzheimer's Diagnosis With Missing Modalities

    [https://arxiv.org/abs/2611.34721](https://arxiv.org/abs/2611.34721)

    该论文提出一种在线策略蒸馏方法，让模型自行逐模态插补缺失数据，并由掌握特权信息（真实诊断结果和完整模态）的教师对每步插补和诊断输出进行评分，从而在模态缺失情况下实现校准良好的阿尔茨海默病诊断。

    

    阿尔茨海默病的诊断依赖于临床、生物标本和影像学证据，但很少有患者完成全部检查。由于这些模态彼此互补，缺失的模态存在多种合理取值。现有方法大多直接舍弃缺失模态，或仅基于单一估计进行训练，因此分类器从未见过该模态的其他可能取值，估计值也从未依据诊断结果得到检验。我们通过在线策略蒸馏弥补了这两方面不足：模型自行填充缺失的模态，而拥有仅在训练阶段可获得的特权信息的教师对这些填充进行评分。我们的方法“特权教师分级Rollout插补”（Graded Rollout Imputation with Privileged teachers）逐模态抽取多个填充值。一个知晓诊断结果的插补教师对每一步进行评分，按照统计学实践的推荐将诊断结局纳入插补过程。一个能看到全部真实模态的诊断教师提供软目标。在多样化的填充值上训练，模型学会了能够反映……的置信度（原文摘要在此处截断）。

    arXiv:2611.34721v1 Announce Type: new  Abstract: Alzheimer's disease is diagnosed from clinical, biospecimen and imaging evidence, but few patients undergo every test. Because these modalities are complementary, a missing one has many plausible values. Most existing methods leave it out or train on one estimate, so their classifier never sees its other values, and the estimate is never checked against the diagnosis. We address both gaps with on-policy distillation: the model fills in the missing modalities itself, and teachers with information available only in training grade these fills. Our method, Graded Rollout Imputation with Privileged teachers, draws several fills, one modality at a time. An imputation teacher that knows the diagnosis grades each step, bringing the outcome into the imputation as statistical practice recommends. A diagnosis teacher that sees all real modalities supplies a soft target. Trained on fills that vary, the model learns a confidence that reflects what th
    
[^43]: Spec2Weights：对语言模型根据规范书写的神经网络进行形式化认证

    Spec2Weights: Certifying Networks that a Language Model Writes from a Specification

    [https://arxiv.org/abs/2611.34711](https://arxiv.org/abs/2611.34711)

    该论文提出Spec2Weights方法，首次实现了对语言模型根据自然语言规范直接生成（无数据、无梯度）的神经网络在其真实运行的float32算术上的穷举式精确性证明，在15个网络中成功认证12个，且能指出网络失效的精确边界。

    

    语言模型能够根据自然语言规范直接写出小型MLP、CNN、RNN或GNN的权重，无需任何数据和梯度下降步骤。本文研究这样一个问题：何时可以证明这样的网络在其实际运行的float32浮点算术中是精确的。论文提出的浮点安全引理使穷举float32枚举成为对任意求和顺序的有效证明，而分层误差界则覆盖了权重不是短二进制分数的网络。借助动态规划、模板检查、SMT证明、广度优先搜索以及针对权重绑定递归结构的归纳法，该方法成功认证了15个生成网络中的12个（其中1个仅经过一次常量修正），验证覆盖指定域内的所有输入（最多可达10^16个输入，或不超过大小界限的所有图或网格），每个网络仅需不超过一分钟的CPU时间。这些认证还能精确指出网络在何处失去精确性：例如最短路径网络当且仅当距离不超过4096时是精确的，并在预测的跳数处失效。（摘要原文在结尾处被截断）

    arXiv:2611.34711v1 Announce Type: new  Abstract: A language model can write the weights of a small MLP, CNN, RNN or GNN from a plain-language specification, with no data and no gradient step. We ask when such a network can be proved exact in the float32 arithmetic it actually runs in. A float-safety lemma makes exhaustive float32 enumeration a proof for any summation order, and a layered error bound covers networks whose weights are not short binary fractions. With dynamic programming, template checks, an SMT proof, breadth-first search and induction for weight-tied recurrences, this certifies 12 of 15 written networks (one after a single constant fix) on every input of a stated domain (up to $10^{16}$ inputs, or every graph or grid up to a size bound), in a minute of CPU or less each. The certificates also say where a network stops being exact: the shortest-path network is exact iff the distance is at most 4,096 and fails at the predicted hops. Thirty new specifications were each writ
    
[^44]: 从共享模式到个体证据：时序知识图谱中新兴实体的两阶段推理

    From Shared Patterns to Individual Evidence: Two-Stage Reasoning for Emerging Entities in Temporal Knowledge Graphs

    [https://arxiv.org/abs/2611.34708](https://arxiv.org/abs/2611.34708)

    该论文提出INSPIRE两阶段推理框架，通过融合全目录事件记忆的共享模式预测与轻量级个体证据建模，有效解决时序知识图谱中新兴实体从零样本到少样本演化场景下的链接预测难题。

    

    时序知识图谱（TKG）持续引入新兴实体，这些实体的交互历史会从零样本的初次登场逐步演变为稀疏且突发性的少样本轨迹。然而，现有研究范式较为割裂：模式迁移方法缺乏覆盖完整实体目录的事件记忆，而在线记忆模型则存在端点角色不对称问题，导致对未见答案的排序性能下降。此外，稀疏的早期交互在偏斜分布下同时表现出长程分散性与局部冗余性。为应对这些挑战，我们提出了INSPIRE（Integrating Shared Patterns and Individual Evidence for Reasoning，融合共享模式与个体证据的推理框架），一个面向已知实体目录的两阶段预测框架。在第一阶段，INSPIRE学习一个具备可用性感知能力的共享模式预测器，该预测器融合跨完整候选目录的事件更新记忆，并采用查询侧与目录侧协同掩码机制。随后，在第二阶段，该预测器被冻结，并在其基础上优化一个轻量级的个体……（原文摘要在此处截断）

    arXiv:2611.34708v1 Announce Type: new  Abstract: Temporal knowledge graphs (TKGs) continually introduce emerging entities whose interaction histories evolve from zero-shot debuts into sparse, bursty few-shot trajectories. However, current paradigms are fragmented: pattern-transfer methods lack catalog-wide event memory, whereas online memory models suffer from endpoint-role asymmetry that degrades unseen-answer ranking. Furthermore, sparse early interactions exhibit both long-range dispersion and local redundancy under skewed distribution. To address these challenges, we propose INSPIRE (Integrating Shared Patterns and Individual Evidence for Reasoning), a two-stage forecasting framework over a known catalog. In Stage I, INSPIRE learns an availability-aware shared-pattern predictor that fuses event-updated memory across the full candidate catalog while applying coordinated query- and catalog-side masking. Subsequently, Stage II freezes this predictor and optimizes a lightweight individ
    
[^45]: 当代理模型出错时：用于自适应数据采样的失配校正老虎机

    When the Proxy Is Wrong: Mismatch-Corrected Bandits for Adaptive Data Sampling

    [https://arxiv.org/abs/2611.34707](https://arxiv.org/abs/2611.34707)

    该论文提出失配校正老虎机方法，将代理模型与目标模型的排序失建为多保真度老虎机问题，利用稀少审计中的成对残差校正代理奖励并智能分配审计预算，从而在无需假设共享偏移或排序保持的情况下改进大模型的自适应数据采样。

    

    数据分配方法通常使用一个小型代理模型来为更大的目标模型选择训练数据。然而，该代理模型对数据的排序可能与目标模型不一致，从而导致次优的训练性能。为了解决这一问题，我们对代理模型与目标模型之间的失配进行建模，并将其纳入分配决策中。为此，该问题被构建为一个多保真度老虎机问题，其中每个臂对应一种数据干预。代理奖励在每一轮都会被观测，而目标奖励仅在对同一更新的稀少审计中才会揭示。这些成对的观测能够在不假设共享偏移或排序保持的前提下，识别出每个臂各自的失配。失配校正老虎机通过成对残差来校正代理模型，仅审计不确定的决策，并在整个训练过程中合理控制审计预算。从理论角度来看，我们在平稳性条件下证明了遗憾分解和下界。（原文摘要此处似乎被截断）

    arXiv:2611.34707v1 Announce Type: cross  Abstract: Data allocation methods often use a small proxy model to select training data for a larger target model. However, this proxy can rank data differently from the target, leading to suboptimal training performance. To address this problem, we model the mismatch between the proxy and target models and incorporate it into the allocation decision. To do so, the problem is cast as a multi-fidelity bandit, where each arm is a data intervention. The proxy reward is observed every round, whereas the target reward is revealed only on rare audits of the same update. These paired observations identify arm-wise mismatches without assuming a shared offset or preservation of rankings. Mismatch-corrected bandits correct the proxy with paired residuals, audit only uncertain decisions, and pace the audit budget over training. From a theoretical viewpoint, we prove a regret decomposition and lower bounds under stationarity. These bounds show that proxy-on
    
[^46]: Maco-RL：通过混合任务强化学习扩展智能体能力

    Maco-RL: Scaling Agent Capabilities with Mixed-Task Reinforcement Learning

    [https://arxiv.org/abs/2611.34703](https://arxiv.org/abs/2611.34703)

    Maco-RL通过基于配额的异步调度器和三缓冲区采样机制有效扩展混合任务强化学习，实现了智能体能力的稳定训练、高效计算分配与持续的领域扩展。

    

    混合任务强化学习直接在多个领域上训练统一策略，从而持续扩展智能体能力，无需训练独立的专家模型，也不需要额外的蒸馏阶段。尽管其颇具吸引力，但扩展MixRL在实现稳定高效的混合任务学习、有效分配采样计算资源以及支持持续的领域扩展方面面临重大挑战。为应对这些挑战，我们提出了Maco-RL，一种能够有效扩展MixRL的方法。具体而言，Maco-RL采用基于配额的调度器在多个任务领域之间进行异步采样，通过平衡梯度贡献并缓解离策略效应来稳定训练。在另一个正交维度上，Maco-RL进一步引入了三缓冲区采样，在不同掌握程度的样本之间分配计算资源，在保持已获得能力的同时促进持续的前沿探索。实验……（原文截断）

    arXiv:2611.34703v1 Announce Type: cross  Abstract: Mixed-task reinforcement learning (MixRL) directly trains a unified policy across domains to continually expand agent capabilities without training separate specialists or requiring an extra distillation stage. Despite its appeal, scaling MixRL poses significant challenges in achieving stable and efficient mixed-task learning, effectively allocating rollout compute, and supporting continual domain expansion. To address these challenges, we propose Maco-RL, an approach that effectively scales MixRL. Specifically, Maco-RL employs a quota-based scheduler for asynchronous rollouts across multiple task domains, thereby stabilizing training by balancing gradient contributions and mitigating off-policy effects. Along an orthogonal axis, Maco-RL further introduces Tri-Buffer Sampling (TBS) to allocate compute among samples with varying levels of mastery, promoting sustained frontier exploration while preserving acquired capabilities. Experimen
    
[^47]: 类别比例变化对分类器曲率的影响

    Effect of class proportion shifts on classifier curvature

    [https://arxiv.org/abs/2611.34702](https://arxiv.org/abs/2611.34702)

    该论文通过测量输出层参数的损失Hessian迹来研究类别比例变化对分类器曲率的影响，发现曲率变化极小（-0.152%至+0.071%），并提出了一种基于初始类别梯度的预测方法且通过了前瞻性验证。

    

    arXiv:2611.34702v1 公告类型：新论文 摘要：改变类别混合比例可以在不改变分类器的情况下改变平均损失。为了将这种重新加权效应与模型本身的变化区分开来，我们研究了十类CIFAR-10分类任务，为两个目标类别分别设置独立分支，并在每个目标类别的固定图像上测量损失Hessian矩阵相对于输出层参数的迹。在十次新的初始化实验中，在特征冻结的条件下对输出层进行60次更新后，相对于各分支的初始迹，在目标类别比例从10%上升到19%的分支中，迹平均变化了-0.152%；而在比例下降到1%的分支中，迹变化了+0.071%。这些变化无法将类别比例的影响与所分配训练图像的影响分离开来。我们在检查第一次实验结果后，开发了一种基于初始类别梯度的预测方法，然后在不做任何修改的情况下进行了前瞻性测试。对于从10%直接增加到19%的情况，而不是增加……

    arXiv:2611.34702v1 Announce Type: new  Abstract: Changing the class mix can alter mean loss without changing the classifier. To distinguish this reweighting from changes in the model itself, we studied ten-class CIFAR-10 classification in separate branches for two target classes, measuring the trace of the loss Hessian with respect to the output-layer parameters on fixed images of each target class. Across ten new initialisations, after 60 updates to the output layer with frozen features, the trace changed by -0.152% on average in branches where the target class share rose from 10% to 19%, and by +0.071% in branches where it fell to 1%, relative to each branch's initial trace. These changes do not isolate the effect of class share from that of the assigned training images. We developed a forecast based on initial class gradients after examining the first experiment's results and then tested it prospectively without modification. For direct increases from 10% to 19%, rather than increas
    
[^48]: 多飓风期间动态交通网络的疏散交通预测：一种广义图学习方法

    Evacuation Traffic Prediction for Dynamic Transportation Networks over Multiple Hurricanes: A Generalized Graph Learning Approach

    [https://arxiv.org/abs/2611.34687](https://arxiv.org/abs/2611.34687)

    本研究利用12场历史飓风的检测器和疏散数据，提出一种广义动态图学习框架（图卷积LSTM），实现跨多场飓风的疏散交通预测并解决模型泛化难题。

    

    飓风疏散期间的高交通需求导致主要高速公路严重拥堵，给疏散人员带来延误和安全风险。实时预测疏散交通可以显著增强疏散管理工作。然而，由于飓风特征的不断演变和人群响应的多样性，疏散交通高度动态且具有不确定性。每场飓风都有独特的特征和疏散背景，导致不同的时空交通模式。此外，基于检测器的交通网络结构在多场飓风之间也存在差异。因此，在一场飓风数据上训练的模型可能无法很好地泛化到其他飓风。本研究利用来自12场历史飓风的交通检测器和疏散数据，提出了一个广义动态图学习框架。利用交通网络的动态特性，我们开发了一种图卷积长短期记忆网络

    arXiv:2611.34687v1 Announce Type: cross  Abstract: High traffic demand during hurricane evacuations causes severe congestion on major highways, leading to delays and safety risks for evacuees. Predicting evacuation traffic in real-time can significantly enhance evacuation management efforts. However, evacuation traffic is highly dynamic and uncertain due to evolving hurricane characteristics and diverse population response. Each hurricane has unique characteristics and evacuation contexts, resulting in different spatial-temporal traffic patterns. Moreover, the structure of detector-based transportation networks varies across multiple hurricanes. Hence, a model trained on the data from one hurricane may not generalize well for others. This study proposes a generalized dynamic graph learning framework using traffic detectors and evacuation data from 12 historical hurricanes. Leveraging the dynamic nature of transportation network, we develop a Graph Convolutional Long Short-Term Memory (
    
[^49]: 可证明高效的线性函数逼近凹效用强化学习

    Provably Efficient Concave Utility Reinforcement Learning with Linear Function Approximation

    [https://arxiv.org/abs/2611.34678](https://arxiv.org/abs/2611.34678)

    本文提出基于模型的乐观凹效用策略优化（OCUPO）框架，利用双线性MDP的低维结构高效估计状态-动作占用度量，首次在线性函数逼近下实现了可证明高效的凹效用强化学习。

    

    凹效用强化学习（CURL）统一了多种问题类型，包括标准强化学习（RL）、模仿学习和最大熵探索。与标准RL不同，CURL通常需要估计高维的状态-动作占用度量。因此，现有的CURL方法要么回避了探索问题，要么仅限于表格型设置，要么缺乏计算高效的策略优化。为解决这些问题，我们在有限水平回合制双线性MDP中研究CURL，其中转移模型在已知特征空间中具有低维表示。我们利用双线性结构通过低维特征期望高效估计占用度量，并提出了基于模型的乐观凹效用策略优化（OCUPO）框架。我们首先在在策略（on-policy）设置下实例化OCUPO，其中每次迭代使用独立轨迹进行估计

    arXiv:2611.34678v1 Announce Type: new  Abstract: Concave utility reinforcement learning (CURL) unifies a diverse range of problems, including standard reinforcement learning (RL), imitation learning, and maximum entropy exploration. Unlike standard RL, CURL generally requires estimating high-dimensional state-action occupancy measures. Consequently, existing methods for CURL either sidestep the exploration problem, are restricted to the tabular setting, or lack computationally efficient policy optimization. To address this, we study CURL in finite-horizon episodic bilinear MDPs, where the transition model admits a low-dimensional representation in known feature spaces. We use the bilinear structure to efficiently estimate occupancy measures via low-dimensional feature expectations, and propose a model-based optimistic concave utility policy optimization (OCUPO) framework. We first instantiate OCUPO in the on-policy setting, where each iteration uses independent trajectories to estimate
    
[^50]: 学会被截断：为大语言模型检查点做好SVD压缩准备

    Learning to Be Truncated: Preparing LLM Checkpoints for SVD Compression

    [https://arxiv.org/abs/2611.34667](https://arxiv.org/abs/2611.34667)

    提出LLift方法，仅需在单一压缩比率下用原模型作为教师进行一次低秩微调，即可将预训练大语言模型检查点准备好，使其在多种SVD压缩比率下直接获得准确率提升，无需针对每个比率重复微调。

    

    基于奇异值分解（SVD）的压缩通常需要后续的微调或权重校正来缓解性能下降问题，而针对特定压缩预算的优化必须针对不同压缩比率重复进行。我们提出了LLift，它能够使预训练模型为跨多种压缩比率的SVD压缩做好准备，而无需进一步的微调。LLift以原始模型作为教师，在单一比率下微调一个低秩近似。我们将学习到的更新合并到原始权重中，同时保留在训练前向传播中被排除的互补组件。随后我们对准备好的模型的目标矩阵应用SVD，并在不同的截断秩下复用这些分解结果。在LLaMA-7B上，我们仅使用一个准备好的模型就在三个压缩比率下展示了准确率的提升。这些提升可推广至不同的模型家族、模型规模以及SVD压缩算子。我们通过分析梯度和早期训练更新来提供（对该方法有效性的）解释。

    arXiv:2611.34667v1 Announce Type: new  Abstract: Compression based on singular value decomposition (SVD) often requires subsequent fine-tuning or weight correction to mitigate performance degradation. Budget-specific optimization must be repeated for different compression ratios. We introduce LLift, which prepares a pretrained model for SVD compression across ratios without further fine-tuning. LLift fine-tunes a low-rank approximation at a single ratio using the original model as a teacher. We merge the learned updates into the original weights while preserving the complementary components excluded from the training forward pass. We apply SVD to the prepared model's target matrices and reuse the decompositions at different truncation ranks. On LLaMA-7B, we demonstrate accuracy gains across three compression ratios using a single prepared model. The gains extend across model families, model sizes, and SVD compression operators. We analyze gradients and an early training update to provi
    
[^51]: 当我们能够读懂世界模型时，还需要强化学习吗？

    Do We Still Need Reinforcement Learning When We Can Read the World Model?

    [https://arxiv.org/abs/2611.34656](https://arxiv.org/abs/2611.34656)

    该论文提出了一种新范式：让基于大语言模型的编程智能体直接阅读可读的世界模型（环境源代码）来推理并合成控制策略，无需任务特定的强化学习训练，即可在大幅减少交互经验的情况下达到甚至超越传统强化学习方法的性能。

    

    强化学习在序贯决策中取得了显著的成功，但学习有效的策略往往需要大量的交互经验。基于模型的强化学习通过利用世界模型来缓解这种依赖，而世界模型通常被用于支持规划或策略优化。在本文中，我们研究了一种替代范式：自主的基于大语言模型（LLM）的编程智能体通过对可读的世界模型进行推理，直接合成控制策略。通过检查环境的源代码，这些智能体将环境知识转化为可执行的决策规则，并利用在线反馈或离线数据迭代地改进其策略，而无需针对特定任务的强化学习训练。在多种决策基准上的实验表明，合成策略能够匹配甚至超越传统的强化学习方法，同时所需的经验大幅减少。进一步的分析揭示了可解释的决策规则（原文在此处截断）。

    arXiv:2611.34656v1 Announce Type: cross  Abstract: Reinforcement learning has achieved remarkable success in sequential decision-making, but learning effective policies often requires substantial interaction experience. Model-based reinforcement learning alleviates this dependence by leveraging world models, which are typically used to support planning or policy optimization. In this paper, we investigate an alternative paradigm in which autonomous LLM-based coding agents directly synthesize control policies by reasoning over readable world models. By examining environment source code, these agents translate environmental knowledge into executable decision rules and iteratively refine their policies using online feedback or offline data, without task-specific RL training. Experiments across diverse decision-making benchmarks show that synthesized policies match or outperform conventional RL methods while requiring substantially less experience. Further analyses reveal interpretable dec
    
[^52]: 困难背景配对可跨分类器与图像迁移

    A Hard Background Pairing Transfers across Classifiers and Images

    [https://arxiv.org/abs/2611.34645](https://arxiv.org/abs/2611.34645)

    该论文证明，对某些分类器而言最困难的物体-背景配对，对其他分类器和在新的图像上同样困难，表明这种困难性具有跨分类器和跨图像的迁移性。

    

    图像分类器为图像中的物体命名，其准确率是正确命名图像的比例。一个能在道路上识别汽车的分类器，可能无法识别被粘贴到云朵背景上的同一辆汽车。复合基准测试通过将照片中裁剪出的物体粘贴到来自其他照片的背景上来衡量这类失败。一种配对方式是将每个物体标签置于不同的背景标签上，例如海面上的卡车。某些分类器表现最差的配对，即其最困难的配对，构成了一项严格的测试，但该配对可能仅对这些分类器困难，或仅在用于选择它的照片上困难。我们证明，某些分类器的最困难配对对其他分类器以及在新的照片上依然困难。我们将40个ImageNet分类器按其报告的ImageNet准确率平衡地分为两半。其中一半从Common Objects in Context（COCO）数据集中选取8个物体标签与8个背景标签的最困难配对。另一半（未参与该选择）将该配对与随机配对进行比较……

    arXiv:2611.34645v1 Announce Type: cross  Abstract: An image classifier names the object in an image, and its accuracy is the share of images named correctly. A classifier that names a car on a road may miss the same car pasted onto clouds. Composite benchmarks measure such failures by pasting objects cut from photographs onto backgrounds from other photographs. A pairing puts every object label on a different background label, such as truck on sea. The pairing on which some classifiers do worst, their hardest pairing, makes a strict test, but may be hard only for them or only on the photographs used to choose it. We show that the hardest pairing of some classifiers stays hard for others and on new photographs. We split 40 ImageNet classifiers into halves balanced by reported ImageNet accuracy. One half picks its hardest pairing of 8 object and 8 background labels from Common Objects in Context (COCO). The other half, held out of the choice, compares that pairing with random pairings of
    
[^53]: 通过变分互模拟实现干扰物鲁棒的强化学习

    Distractor-Robust Reinforcement Learning via Variational Bisimulation

    [https://arxiv.org/abs/2611.34641](https://arxiv.org/abs/2611.34641)

    提出VIBES新目标，用对抗项取代像素重建并结合固定容量潜在瓶颈，使模型让潜在状态只需预测奖励与动态，从而丢弃无关干扰信息，显著提升强化学习对干扰物的鲁棒性。

    

    基于模型的强化学习（MBRL）有望实现数据高效性和泛化能力，但典型的基于重建的目标函数会促使模型把表征容量浪费在与任务无关的干扰物上。我们提出了VIBES（基于互模拟编码状态的变分推断），这是一种新的目标函数，用对抗项取代像素重建，该对抗项强制要求潜在状态足以预测奖励和下一个潜在状态。我们证明这一目标度量了编码器距离奖励与动力学充分性的远近，并且只要精确充分性可以实现，其全局最优解就会在历史序列上诱导出互模拟关系。结合固定容量的潜在瓶颈，这促使潜在状态保留与任务相关的信息，同时丢弃无关的变化。我们的方法可作为Dreamer模型学习组件的即插即用替代方案，实现了最先进的性能

    arXiv:2611.34641v1 Announce Type: new  Abstract: Model-based reinforcement learning (MBRL) promises data efficiency and generalization, but typical reconstruction-based objectives encourage models to waste representational capacity on task-irrelevant distractors. We introduce VIBES (Variational Inference for Bisimulation-based Encoded States), a new objective that replaces pixel reconstruction with an adversarial term, which enforces that latent states suffice to predict both rewards and the next latent state. We show that this objective measures how far the encoder is from reward and dynamics sufficiency, and that whenever exact sufficiency is attainable, its global optima induce bisimulation relations on histories. Combined with a fixed-capacity latent bottleneck, this encourages latent states to retain task-relevant information while discarding irrelevant variation. Our method serves as a drop-in replacement for Dreamer's model-learning component, achieves state-of-the-art performan
    
[^54]: 跨稀疏自编码器随机种子移植特征消融

    Carrying Feature Ablations Across Sparse Autoencoder Seeds

    [https://arxiv.org/abs/2611.34637](https://arxiv.org/abs/2611.34637)

    该论文提出一种方法：通过分别在两个SAE中移除匹配特征并比较编码器输出的变化，来判断特征消融能否跨随机种子移植，发现在其容差下12%至70%的匹配特征对能够成功跨种子移植。

    

    视觉编码器将图像转换为每个图像块一个向量，并逐层精炼。稀疏自编码器（SAE）将某一层的每个向量表示为若干学习到的方向之和，其中大多数方向的系数为零。研究人员将每个方向称为一个特征，并通过移除某个特征来编辑编码器。而使用另一个随机种子训练的SAE会学习到不同的特征。先前的工作根据方向的一致程度将两个SAE的特征进行一对一匹配。然而匹配的特征可能方向相似，却对图像的不同部分产生响应，因此移除它们可能对编码器产生不同的改变。我们移除一个特征，并单独移除其匹配特征，然后在相同的留出图像上比较编码器输出的变化。当第二种变化几乎复现第一种变化时，我们称该移除操作可以跨种子移植。实验设置固定DINOv2或SigLIP的某一层及其SAE特征数量。在我们的容差范围内，移除操作在三个（种子间的）12%至70%的匹配特征对上能够移植。（原文摘要在此处截断）

    arXiv:2611.34637v1 Announce Type: cross  Abstract: A vision encoder turns an image into one vector per patch, refined layer by layer. A sparse autoencoder (SAE) writes each vector of one layer as a sum of learned directions, most with a coefficient of zero. Researchers call each direction a feature and edit the encoder by removing one feature. An SAE trained from another random seed learns different features. Earlier work matches the two SAEs' features one to one by how closely their directions agree. Matched features can point alike yet respond to different image parts, so removing them may change the encoder differently. We remove a feature and, separately, its match, then compare the changes in the encoder's output on the same held out images. When the second change nearly reproduces the first, we say the removal carries across seeds. A setting fixes a DINOv2 or SigLIP layer and its SAE feature count. At our tolerance, the removal carries for 12% to 70% of matched pairs across three
    
[^55]: LatentRL-QAS：基于潜在强化学习的量子架构搜索迁移方法

    LatentRL-QAS: Transfer via Latent Reinforcement Learning for Quantum Architecture Search

    [https://arxiv.org/abs/2611.34634](https://arxiv.org/abs/2611.34634)

    提出LatentRL-QAS框架，利用电路变分自编码器将任意量子比特数的电路映射到统一的潜在空间中，使强化学习智能体在固定大小的状态、动作和网络下实现跨量子比特数的迁移学习，并通过对潜在空间的周期性刷新来适应更大系统。

    

    强化学习（RL）已成为量子架构搜索（QAS）——即量子电路自动化设计——的一种重要方法。然而，将智能体从一个量子比特数中学到的知识迁移到另一个量子比特数仍缺乏深入研究：最先进的RL-QAS智能体的状态、动作空间和神经网络都会随量子比特数增长，这使得跨量子比特数的迁移学习变得难以实现。我们通过引入LatentRL-QAS使这种迁移成为可能，这是一个面向基于RL的QAS的潜在空间迁移与刷新框架。其核心是一个电路变分自编码器，它将电路编码为每个量子门对应一个token，并将任意量子比特数上的电路映射到同一个潜在空间中。RL智能体在这个空间中进行导航，而不是直接添加量子门，因此其状态、动作和网络在每个量子比特数下都保持相同的大小。潜在空间会定期根据智能体的新发现进行刷新，对于更大的系统……（摘要在此处截断）

    arXiv:2611.34634v1 Announce Type: cross  Abstract: Reinforcement learning (RL) emerges as a prominent approach to quantum architecture search (QAS), the automated design of quantum circuits. However, transferring what an agent has learned from one number of qubits to another is still weakly explored: the state, the action space and the neural network of a state-of-the-art RL-QAS agent all grow with the number of qubits, which makes transfer learning across qubit counts prohibitive. We make this transfer possible by introducing LatentRL-QAS, a latent space transfer and refresh framework for RL-based QAS. At its core, a circuit variational autoencoder encodes a circuit as one token per gate and maps circuits on any number of qubits into the same latent space. An RL agent navigates this space instead of adding gates, so its state, actions and network have the same size at every qubit count. The latent space is periodically refreshed with the agent's new discoveries and, for a larger syste
    
[^56]: 基于因果机器学习预测急性粮食危机并发现可干预的驱动因素

    Predicting acute food crises and discovering actionable drivers with causal machine learning

    [https://arxiv.org/abs/2611.34629](https://arxiv.org/abs/2611.34629)

    该研究构建了包含驱动因素变量的粮食不安全数据集，开发了缺失数据鲁棒的机器学习流程以在3至9个月提前期预测急性粮食危机，并通过因果机器学习方法对比预测重要性与因果估计，识别出可干预的危机驱动因素。

    

    急性粮食不安全由冲突、不利气候事件、人口流离失所和经济冲击的复合作用所驱动。我们利用驱动因素变量（月度、次国家级、覆盖80个国家、2007–2024年）丰富了开放获取的统一粮食不安全数据集，并开发了能够应对数据缺失的机器学习预测流程，可在3、6和9个月的提前期预测危机发生，同时使用FEWS NET分类体系，并首次大规模采用IPC/CH分类体系。经过嵌套交叉验证评估的预测流程在PR-AUC指标上比无技能基线高出15至50个百分点，且其性能与危机流行率相关，而与所采用的分类体系无关。基于TreeSHAP的可解释性分析揭示了复合的、具有地域特异性的预测信号特征，并归纳出三种反复出现的预测模式。通过将预测重要性与针对可干预驱动因素的双重稳健因果估计进行比较，我们发现急性冲突事件具有最……（原文在此截断）

    arXiv:2611.34629v1 Announce Type: new  Abstract: Acute food insecurity is driven by a compound of conflict, adverse climate events, displacement, and economic shocks. We enrich the open-access Harmonized Food Insecurity Dataset with driver variables (monthly, subnational, 80 countries, 2007--2024) and develop missing-data-resilient machine learning pipelines forecasting crisis occurrence at 3, 6, and 9-month lead times using both FEWS NET and, for the first time at scale, IPC/CH classifications. Pipelines, evaluated through nested cross-validation, outperform no-skill baselines by 15 to 50 percentage points in PR-AUC, with performance correlated with crisis prevalence rather than the classification system. TreeSHAP-based explainability reveals compound, geography-specific predictive signal profiles, with three recurrent forecasting archetypes. Comparing predictive importance against doubly robust causal estimates of actionable drivers, we find that acute conflict events, having the wea
    
[^57]: DecisionReplay：用于集成决策模型的语料库与离线模拟器

    DecisionReplay: A Corpus and Offline Simulator for Ensembling Decision Models

    [https://arxiv.org/abs/2611.34627](https://arxiv.org/abs/2611.34627)

    DecisionReplay 发布了 17 个商业决策模型在 35 个基准测试上的完整概率向量及成本延迟数据，使集成、路由和级联策略可在离线确定性模拟器中快速评估；实验表明精心挑选的单一模型很难被超越，而最佳路由器能以其 64% 的成本达到同等性能。

    

    比较集成方法、路由器和级联需要获取每个模型在每个条目上的输出及其成本。为路由而构建的存储集合并不保留每个选项的概率。我们提供了一个数据集，可在其上快速模拟集成方法、路由器和级联，而无需新的模型调用。DecisionReplay 包含 17 个商业决策模型（即为每个答案选项返回概率的端点）在 35 个基准测试上的概率向量，以及每次调用的 token 数量、计费成本和实测延迟，并全部公开发布。一个确定性模拟器可以重放任何对这些模型进行选择、组合或级联的策略。在每个基准测试上，一个精心选择的单一模型很难被超越：汇总所有模型会损失 2.7 个准确率百分点，而堆叠方法以 3.4 倍的成本仅获得 1.0 个百分点的提升。这些模型会一起犯错：两个都出错的模型在 72% 的情况下会选择相同的错误选项。最佳路由器以该模型 64% 的成本达到同等性能，并可验证（摘要在此处被截断）……

    arXiv:2611.34627v1 Announce Type: cross  Abstract: Comparing ensembles, routers and cascades requires the output of every model on every item, and its cost. Stored collections built for routing keep no per-option probabilities. We provide a dataset on which ensemble methods, routers and cascades are simulated quickly, without new model calls. DecisionReplay holds the probability vectors of 17 commercial decision models, endpoints that return a probability for every answer option, on 35 benchmarks, with the token counts, billed cost and measured latency of every call, all released. A deterministic simulator replays any policy that selects, combines or cascades them. One well-chosen model per benchmark is hard to beat: pooling all models loses 2.7 accuracy points to it, and stacking gains 1.0 points at 3.4 times its cost. The models are wrong together: two that are both wrong choose the same wrong option in 72% of cases. The best router matches that model at 64% of its cost, and certifie
    
[^58]: HoMe：面向AI助手记忆系统的实用控制框架优化

    HoMe: Toward Practical Harness Optimization of Memory Systems for AI Assistants

    [https://arxiv.org/abs/2611.34624](https://arxiv.org/abs/2611.34624)

    HoMe提出了一个评估驱动的框架，通过迭代评估与编辑候选控制框架，并结合会话级记忆版本控制和执行轨迹分析来定位共享失败机制，从而自动化优化AI助手记忆系统的代码、提示词与配置设计。

    

    长期运行的AI助手记忆系统的行为由一个控制框架支配，该框架由用于记忆构建、维护、检索和答案生成的代码、提示词和配置组成。然而，记忆系统控制框架的优化在很大程度上仍然是手动的，且仅靠不受限制的编辑并不能确保结构上的改进。在本文中，我们提出了HoMe，这是一个由评估驱动的框架，它迭代地评估候选控制框架，并通过编辑任意部分从任何先前评估过的控制框架中派生出新的候选框架。HoMe将会话级记忆版本控制与执行轨迹相结合，以识别跨案例的共同失败机制，并将其定位到相关的会话和控制框架组件。它通过候选门控机制鼓励早期的结构性探索，并通过冒烟测试、自适应采样和已构建记忆的复用来降低评估成本。我们验证了HoMe（原文摘要在此处截断）

    arXiv:2611.34624v1 Announce Type: new  Abstract: The behavior of memory systems for long-running AI assistants is governed by a harness, which consists of the code, prompts, and configurations for memory construction, maintenance, retrieval, and answer generation. However, optimizing the harness of a memory system remains largely manual, and unrestricted editing alone does not ensure structural improvements. In this paper, we introduce HoMe, an evaluation-driven framework that iteratively evaluates candidate harnesses and derives each new candidate from any previously evaluated harness by editing any part of it. HoMe combines session-level memory versioning with execution traces to identify shared failure mechanisms across cases and localize them to the responsible sessions and harness components. It encourages early structural exploration through candidate gating and reduces evaluation cost through smoke tests, adaptive sampling, and the reuse of constructed memory. We validate HoMe w
    
[^59]: 基于ComBat的影像组学协调中批变量的自动化选择

    Automated Batch-Variable Selection for ComBat-Based Radiomic Harmonization

    [https://arxiv.org/abs/2611.34618](https://arxiv.org/abs/2611.34618)

    提出ParsiComBat方法，为ComBat影像组学协调自动选择最优批变量（采集参数），在去除批效应与保留特征结构之间取得平衡，并能识别真实批变量（模拟实验准确率92.9%）、在技术变异可忽略时主动放弃校正。

    

    ComBat可用于协调多中心影像组学特征，但批变量通常依靠人工选择，在不同图像采集参数之间缺乏统一的标准。我们提出ParsiComBat，它为ComBat自动选择采集参数，在批效应去除与特征结构保留之间取得平衡，并在技术变异可忽略时主动放弃校正。ParsiComBat按批效应大小对候选采集参数进行排序，过滤冗余参数，并对顺序执行的ComBat校正进行评分。我们通过三项实验对其进行了评估：在模拟实验中，测试其能否恢复已知的批变量并在无批变量时选择放弃；在CT（NSCLC-Radiomics）和MRI（MAMA-MIA）队列中，将其批效应去除效果与制造商（手动）、所有可行参数（贪婪）以及AutoCombat进行比较；并在临床预测任务中进行了内部和外部验证。在模拟实验中，ParsiComBat在92.9%的正向试验中正确识别出真实的批变量，并在无批变量时选择放弃（摘要在此处截断）。

    arXiv:2611.34618v1 Announce Type: new  Abstract: ComBat harmonizes multicentre radiomic features, but batch variables are usually chosen manually, without a standardized criterion across image acquisition parameters. We propose ParsiComBat, which selects acquisition parameters for ComBat, balancing batch-effect removal against feature-structure preservation and abstaining when technical variation is negligible.   ParsiComBat ranks candidate acquisition parameters by batch effect, filters redundancy, and scores sequential ComBat corrections. We evaluated it in three experiments: in simulations, testing whether it recovered a known batch variable and abstained without one; in CT (NSCLC-Radiomics) and MRI (MAMA-MIA) cohorts, comparing batch-effect removal against manufacturer (manual), every feasible parameter (greedy) and AutoCombat; and in clinical prediction, internally and externally.   In simulations, ParsiComBat identified the true batch variable in 92.9% of positive trials and abst
    
[^60]: 基于时滞嵌入的相依数据条件独立图学习

    Conditional Independence Graph Learning from Dependent Data Via Time-Delay Embedding

    [https://arxiv.org/abs/2611.34616](https://arxiv.org/abs/2611.34616)

    提出一种基于时滞嵌入的时域方法，将相依时间序列的条件独立图估计转化为多属性随机向量图估计问题，突破了以往频域方法仅适用于短程相依数据的限制，可同时处理短程相依和长程相依的高维平稳多元高斯时间序列。

    

    我们研究高维平稳多元高斯时间序列的条件独立图（CIG）推断问题，该时间序列可以是短程相依（SRD）或长程相依（LRD）的。在时间序列图中，向量序列的每个分量由一个不同的节点表示，分量之间的关联由相应节点之间的边表示。我们考虑一种基于时滞嵌入的时域方法，将问题转化为随机向量的多属性图估计问题，其中图的每个节点都关联一个随机向量。对于嵌入维度 $d$，每个节点所关联的 $(d+1)$ 维随机向量由一个时间序列分量及其 $d$ 个延迟副本组成，延迟为不超过 $d \ge 1$ 的整数延迟。以往对相依数据的图估计研究多采用频域方法，但其分析仅限于短程相依数据。

    arXiv:2611.34616v1 Announce Type: cross  Abstract: We consider the problem of inferring the conditional independence graph (CIG) of a high-dimensional stationary multivariate Gaussian time series which could be short-range dependent (SRD) or long-range dependent (LRD). In a time series graph, each component of the vector series is represented by a distinct node, and associations between components are represented by edges between the corresponding nodes. We consider a time-domain approach based on time-delay embedding, casting the problem as one of multi-attribute graph estimation for random vectors where a vector is associated with each node of the graph. For an embedding dimension $d$, at each node the associated $(d+1)$-dimensional random vector consists of a time series component and its $d$ delayed copies, with integer delays up to $d \ge 1$. Graph estimation from dependent data has been investigated in the past using frequency-domain methods whose analysis is restricted to SRD da
    
[^61]: 基于无监督项目反应模型的群体学习

    Learning from Crowds with Unsupervised Item Response Models

    [https://arxiv.org/abs/2611.34602](https://arxiv.org/abs/2611.34602)

    本文将众包标注建模为无监督项目反应理论（IRT）问题，证明无监督IRT模型能够可靠恢复标注者与题目的潜在特质并实现稳定的标签推断，还提出在分类器训练中利用IRT模型提供指导。

    

    项目反应理论（IRT）是研究测试的学科，涵盖测试题目与受试者的属性。众包标注可以被建模为一个无监督的IRT问题：标注者扮演能力各异的受试者角色，标注任务则扮演难度不同的题目角色。我们在六个包含众包标签和已知真实标签的数据集上，评估了标准逻辑IRT模型的无监督变体在潜在特质恢复和标签推断方面的表现。与基于真实标签训练的有监督变体相比，无监督IRT模型在大多数数据集上都能可靠地恢复标注者和题目的特质（如能力、难度和区分度）。与大多数基线方法相比，IRT模型在标签推断方面似乎更为稳定，但依据数据集的不同，它未必是最佳选择。我们进一步提出在分类器训练过程中利用IRT模型提供指导，并与现有的众包方法进行比较。

    arXiv:2611.34602v1 Announce Type: new  Abstract: Item Response Theory (IRT) is the study of tests, including the properties of both test items and test takers. Crowdsourced labelling can be cast as an unsupervised IRT problem: annotators play the role of test takers with varying ability, and annotation tasks play the role of items with varying difficulty. We evaluate unsupervised variants of standard logistic IRT models for latent trait recovery and label inference on six datasets with crowdsourced labels and known ground truths. Unsupervised IRT models can recover annotator and item traits (e.g., ability, difficulty and discrimination) reliably across most datasets when compared to supervised variants trained on the ground truth. IRT models seem to be more stable in label inference compared to most baselines, but might not be the best possible choice depending on the dataset. We further propose to use IRT models to inform classifiers during training and compare against existing crowd-
    
[^62]: $\mu$MPC：通过Koopman线性化预编译实现从边缘设备到GPU的微秒级非线性与世界模型控制

    $\mu$MPC: Microsecond-Scale Nonlinear and World Model Control from Edge to GPU via Koopman-Linearized Precompilation

    [https://arxiv.org/abs/2611.34597](https://arxiv.org/abs/2611.34597)

    该论文提出$\mu$MPC框架，通过Koopman线性化预编译将非线性模型预测控制转化为仅含动作变量的二次规划，实现了从微控制器到GPU的微秒级高速控制，同时支持控制感知的动力学学习。

    

    我们提出了$\mu$MPC，一个利用学习到的Koopman线性化动力学来减少在线计算量的快速模型预测控制（MPC）框架。通过预计算有限时域的控制到输出映射，$\mu$MPC消除了提升状态轨迹，得到仅含动作变量的Koopman编译二次规划，其决策变量数为$Hn_u$，与提升维度无关。固定的预计算原始算子使得仅依靠矩阵-向量乘法和投影即可实现高效的原始-对偶迭代，无需重新分解即可适应变化的目标和约束。这种高效率还使得通过训练循环中可微的KC-QP实现控制感知的动力学学习以及批量DAgger式数据聚合成为可能。$\mu$MPC在桌面CPU上实现了低至23微秒的求解时间，在168 MHz的Crazyflie微控制器上实现了24毫秒的求解时间。它可扩展至在5毫秒内完成4096维四足机器人控制、0.4毫秒内完成2048维世界模型的视觉规划等任务。

    arXiv:2611.34597v1 Announce Type: cross  Abstract: We present $\mu$MPC, a fast model predictive control (MPC) framework that reduces online computation using learned Koopman-linearized dynamics. By precomputing finite-horizon control-to-output maps, $\mu$MPC eliminates lifted-state trajectories, yielding an action-only Koopman-Compiled QP (KC-QP) with $Hn_u$ decision variables, independent of lifted dimension. A fixed, precomputed primal operator enables efficient primal-dual iterations using only matrix-vector products and projections, accommodating changing goals and constraints without refactorization. This efficiency also enables control-aware dynamics learning through differentiable KC-QPs in the training loop and batched DAgger-style data aggregation. $\mu$MPC achieves solve times as low as $23\mu$s on a desktop CPU and 24 ms onboard a 168 MHz Crazyflie microcontroller. It scales to 4096D quadruped control in under 5 ms, visual planning with 2048D world models in 0.4 ms, and coll
    
[^63]: ChemoMAE：自监督光谱表征学习揭示历史木材中的化学异质性

    ChemoMAE: Self-Supervised Spectral Representation Learning Reveals Chemical Heterogeneity in Historical Wood

    [https://arxiv.org/abs/2611.34581](https://arxiv.org/abs/2611.34581)

    提出自监督Transformer模型ChemoMAE，利用掩码去噪学习像素级光谱表征，无需预先定义的化学标记即可从近红外高光谱图像中揭示历史木材的化学异质性与劣化模式。

    

    在历史木构建筑中，多种环境因素历经数百年作用于化学成分在不同解剖区域存在差异的木材，从而产生复杂的劣化模式。理解木构件的劣化方式对于历史建筑的有效保护与修复十分重要，然而这些模式背后的机制仍不明确。为探究这些机制，我们采用自监督学习，为树种和使用历史各不相同的木构件样本建立一个统一的比较基础。我们使用近红外高光谱成像技术检测了采自日本多座历史建筑的49个木材样本的横截面。ChemoMAE是一种自监督Transformer，通过掩码去噪学习像素级的光谱表征，该方法充分利用了光谱变量之间的多重共线性。对这些表征进行聚类即可生成空间分布图，而无需预先定义的化学标记。

    arXiv:2611.34581v1 Announce Type: new  Abstract: In historical timber buildings, multiple environmental factors act over centuries on wood whose chemical composition varies across anatomical regions, producing complex patterns of degradation. Understanding how timber members degrade is important for the effective conservation and restoration of historical buildings, yet the mechanisms underlying these patterns remain unclear. To investigate these mechanisms, we used self-supervised learning to establish a common basis for comparing specimens from timber members that differ in wood species and service history. We examined transverse sections of 49 wood specimens from several historical buildings in Japan using near-infrared hyperspectral imaging. ChemoMAE, a self-supervised Transformer, learns pixelwise spectral representations through masked denoising that exploits multicollinearity among spectral variables. Clustering these representations yields spatial maps without predefined chemic
    
[^64]: 调优应以能力保持为目标，而非仅仅追求增益：SFT的遗忘远比你想象的少

    Tune for Retention, Not Just Gain: SFT Forgets Far Less than You Think

    [https://arxiv.org/abs/2611.34579](https://arxiv.org/abs/2611.34579)

    本文挑战了“SFT因其离策略特性而容易导致灾难性遗忘”的流行观点，证明通过选择更低学习率和批量大小等同时兼顾增益与能力保持的超参数，普通SFT对旧能力的保持远好于以往认知，甚至可与在线策略的SDFT方法相媲美。

    

    灾难性遗忘，即在习得新能力时急剧丧失先前已获得的能力，是语言模型微调中的核心挑战。人们普遍认为监督微调（SFT）容易发生这种遗忘。近期研究将此归因于SFT的离策略（off-policy）特性，并认为在线策略学习是持续学习的关键。本文挑战了这一前提。我们证明，虽然最大化新任务性能确实会引发灾难性遗忘，但选择替代的超参数（更低的学习率和批量大小）可以在几乎不发生遗忘的情况下获得类似的新任务收益。当超参数的选择同时兼顾增益与能力保持时，普通SFT对既有能力的保持远好于此前的说法，并且可以与自蒸馏微调（SDFT）——一种旨在缓解遗忘的在线策略方法——相媲美。虽然SDFT对超参数选择可能更加鲁棒，但它也需要……（原文摘要截断于此）

    arXiv:2611.34579v1 Announce Type: cross  Abstract: Catastrophic forgetting, the drastic loss of previously acquired capabilities when learning new ones, is a central challenge in language model fine-tuning. Supervised fine-tuning (SFT) is widely believed to be prone to such forgetting. Recent works have attributed this to the off-policy nature of SFT, arguing that on-policy learning is key to continual learning. In this paper, we challenge this premise. We show that while maximizing new-task performance can induce catastrophic forgetting, alternative hyperparameter choices (lower learning rate and batch size) can yield similar gains with little to no forgetting. With hyperparameters selected jointly for gain and retention, vanilla SFT retains prior capabilities far better than previously suggested and can be competitive with Self-Distillation Fine-Tuning (SDFT), an on-policy approach designed to mitigate forgetting. While SDFT can be more robust to hyperparameter choices, it also requi
    
[^65]: 利用深度学习模型研究涡旋光束散斑图案识别大气湍流与分数拓扑荷的有效性

    Investigating the efficacy of vortex beam speckle patterns for the recognition of atmospheric turbulence and fractional topological charge with a deep learning model

    [https://arxiv.org/abs/2611.34573](https://arxiv.org/abs/2611.34573)

    该论文提出利用卷积神经网络直接从单张散斑图像中回归涡旋光束的分数拓扑荷并识别大气湍流强度，验证了该方法在宽范围湍流条件下（D/r0为0到50）识别分数拓扑荷（精度达0.01）的有效性。

    

    分数拓扑荷（TC）扩大了轨道角动量（OAM）链路的可用模式集合，但在经过大气湍流后可靠地读取它仍然十分困难。干涉法和衍射法需要无畸变的光束和精细的校准，而大多数机器学习方法仅处理整数或粗间隔的拓扑荷，且未考虑湍流的影响。本研究探讨标量涡旋光束的拓扑荷能否直接从单张散斑图像中回归得到，以及在多大的湍流强度下该方法会失效。我们让涡旋光束通过150个冯·卡门（von Kármán）相位屏传播1500米，用固定的随机散射片对其进行散射，并记录拓扑荷从0到10（步长0.01）、湍流强度D/r0从0到50的256×256、16位散斑图像；本文报告的结果使用了20,000张图像的子集，一个包含10^6张图像的数据集正在准备中。一个卷积神经网络用于估计拓扑荷……（摘要在此处被截断）

    arXiv:2611.34573v1 Announce Type: new  Abstract: Fractional topological charge (TC) enlarges the usable mode set of orbital angular momentum (OAM) links, but reading it reliably after atmospheric turbulence remains difficult. Interferometric and diffractive methods need an undistorted beam and a careful alignment, while most learned approaches treat integer or coarsely spaced charges and do not include turbulence. Here we ask whether the charge of a scalar vortex beam can be regressed directly from a single speckle image, and at which turbulence strength this stops working. We propagate vortex beams over 1500~m through 150 von K\'arm\'an phase screens, scatter them with a fixed random diffuser, and record $256\times256$, 16-bit speckle images for TC from 0 to 10 in steps of 0.01 and turbulence strengths $D/r_0$ from 0 to 50; the results reported here use a subset of 20\,000 images, and a data set of $10^{6}$ images is in preparation. A convolutional neural network estimates the TC and,
    
[^66]: 论边缘AI在面向预测性维护的状态监测中的应用：基于可解释AI的乳制品加工用例

    On Utilization of Edge-AI for condition monitoring towards predictive maintenance: A use case in dairy processing using explainable AI

    [https://arxiv.org/abs/2611.34563](https://arxiv.org/abs/2611.34563)

    该研究开发了一个基于边缘AI的状态监测框架，利用低功耗IIoT传感器节点采集的振动数据并在边缘设备上处理，对乳制品加工厂的旋转阀进行状态监测，为预测性维护迈出重要一步。

    

    工业物联网和人工智能正在推动工业4.0的发展，提升制造系统的生产力、自动化水平和效率。本研究开发了一个基于边缘AI的状态监测框架，作为迈向预测性维护的一步，应用于乳制品加工中试工厂中的旋转阀。该框架使用通过低功耗IIoT传感器节点采集的振动数据，并在边缘设备上进行处理。

    arXiv:2611.34563v1 Announce Type: new  Abstract: The Industrial Internet of Things (IIoT) and Artificial Intelligence (AI) are driving Industry 4.0, enhancing productivity, automation, and efficiency in manufacturing systems. This study develops an Edge-AI-based condition monitoring framework, as a step towards predictive maintenance, for a rotary valve in a dairy processing pilot plant, using vibration data collected via a low-power IIoT sensor node and processed on an edge device.
    
[^67]: Kant-Q：对称马尔可夫博弈中康德均衡的Q学习算法

    Kant-Q: A Q-Learning Algorithm for Kantian Equilibrium in Symmetric Markov Games

    [https://arxiv.org/abs/2611.34536](https://arxiv.org/abs/2611.34536)

    本文提出Kant-Q算法，通过让每个智能体学习所有智能体采取相同行动的行动配置来寻找康德均衡，继承了标准Q学习的收敛保证，并证明每个对称马尔可夫博弈都存在纯策略康德均衡，为解决混合动机博弈中纳什均衡帕累托低效的问题提供了新方案。

    

    AI智能体之间的互动日益增多，其中许多互动属于混合动机博弈，这对学习算法构成了长期挑战。大多数多智能体强化学习（MARL）算法旨在达到纳什均衡，然而在许多此类博弈中，这些均衡是帕累托低效的。我们转而研究康德均衡（Kantian equilibria），这是一种经济学解概念，其中每个智能体采取它希望所有智能体都采用的行动。在对称博弈中，这对应于一个最优的共同策略。我们提出了Kant-Q，这是一种Q学习算法，其中每个智能体学习所有智能体采取相同行动的行动配置。每次更新都是来自单智能体MDP的一个样本，因此Kant-Q继承了标准Q学习的收敛性保证。我们证明每个对称马尔可夫博弈都存在纯策略康德均衡。我们还证明了所有智能体的Q表收敛到相同的极限，其贪心策略是这样一个……（摘要在此处截断）

    arXiv:2611.34536v1 Announce Type: new  Abstract: AI agents increasingly interact with one another, and many of these interactions are mixed-motive games, which have posed a persistent challenge for learning algorithms. Most multi-agent reinforcement learning (MARL) algorithms are designed to reach Nash equilibria, yet in many such games these equilibria are Pareto-inefficient. We instead study Kantian equilibria, an economic solution concept in which each agent plays the action it would want all agents to adopt. In symmetric games, this corresponds to an optimal common policy. We propose Kant-Q, a Q-learning algorithm in which each agent learns action profiles where all agents play the same action. Each update is then a sample from a single-agent MDP, so Kant-Q inherits the convergence guarantees of standard Q-learning. We show that every symmetric Markov game has a pure Kantian equilibrium. We also prove that all agents' Q-tables converge to the same limit, whose greedy policy is such
    
[^68]: AliceAI-Foundation-80B-A3B-Base（AliceAI 基础 80B-A3B 基座模型）

    AliceAI-Foundation-80B-A3B-Base

    [https://arxiv.org/abs/2611.34529](https://arxiv.org/abs/2611.34529)

    Yandex 从零训练并开源了总参数 800 亿、每 token 激活约 30 亿的混合专家基座模型，它结合 Kimi Delta Attention 与全注意力并使用 Muon 优化器训练，同时提出在 MoE 输出上引入辅助损失以在预测的高学习率下稳定训练。

    

    arXiv:2611.34529v1 发布类型：新论文。我们推出 AliceAI-Foundation-80B-A3B-Base，这是一个混合专家语言模型，总参数量为 800 亿，每个词元约有 30 亿激活参数，由 Yandex 从零开始训练，并以 Apache 2.0 许可证开源发布。该模型将 Kimi Delta Attention（Kimi 增量注意力）与全注意力机制相结合，采用注意力残差，并使用 Muon 优化器进行训练。训练过程包括在 17.5 万亿词元上进行的一般预训练、两个上下文扩展阶段，以及一个专门针对推理与智能体交互的训练阶段。我们描述了数据收集与数据增强流程、评估协议、架构消融实验以及缩放定律实验。我们指出了使用留出集训练损失作为下游性能代理指标的局限性，并在混合专家输出上引入了一种辅助损失，以便在预测的学习率下稳定训练。领域特定的数据选择与数据增强提升了事实知识、数学以及编程……（原文在此处截断）

    arXiv:2611.34529v1 Announce Type: new  Abstract: We present AliceAI-Foundation-80B-A3B-Base, a mixture-of-experts language model with 80 billion total parameters and approximately 3 billion active parameters per token, trained from scratch at Yandex and released under the Apache 2.0 license. The model combines Kimi Delta Attention with full attention, uses Attention Residuals, and is trained with Muon. Training comprises general pretraining on 17.5 trillion tokens, two context-extension stages, and a dedicated stage for reasoning and agentic interaction. We describe the data collection and augmentation pipelines, evaluation protocols, architectural ablations, and scaling-law experiments. We identify the limitations of held-out training loss as a proxy for downstream performance and introduce an auxiliary loss on mixture-of-experts outputs to stabilize training at the predicted learning rate. Domain-specific data selection and augmentation improve factual knowledge, mathematics, and pro
    
[^69]: 通过自动化阶段识别与定量评估的韧性分析

    Resilience Analysis Through Automated Stage Identification and Quantitative Assessment

    [https://arxiv.org/abs/2611.34527](https://arxiv.org/abs/2611.34527)

    本文提出一种与领域和威胁无关的机器学习方法，自动识别系统性能时间序列中的PARA韧性阶段，并结合事件前、中、后的性能数据计算韧性指标以实现定量韧性评估，在多个不同领域的系统数据上验证了其通用性。

    

    本文提出了一种基于机器学习的方法，该方法被设计为与领域和威胁无关，可自动识别系统性能时间序列中的PARA（规划/准备、吸收、恢复、适应）韧性阶段，从而支持对不同系统开展后续的定量韧性分析。我们研究了四种用于韧性阶段识别的机器学习模型：随机森林（RF）、支持向量机（SVM）、轻量梯度提升机以及极端梯度提升（XGBoost）。随后，我们利用这些机器学习模型的预测结果，通过从干扰事件发生前、发生期间和发生之后观测到的系统性能数据计算相应的韧性指标，来量化系统韧性。我们使用来自两个不同运营领域网络系统和一个机场交通系统的数据集对所提出的方法进行了评估，证明了该方法在韧性评估方面具有与领域无关的通用适用性。结果表明了（原文摘要在此处截断）

    arXiv:2611.34527v1 Announce Type: new  Abstract: This paper introduced an ML-based approach, designed to be domain- and threat-agnostic, for automated recognition of PARA (Plan/Prepare, Absorb, Recover, Adapt) resilience stages in system-performance time series, enabling subsequent quantitative resilience analysis across different systems. We investigate four ML models for resilience stage recognition: Random Forest (RF), Support Vector Machine (SVM), Light Gradient Boosting Machine (LightGBM), and Extreme Gradient Boosting (XGBoost). We then use the ML models' predictions to quantify system resilience by computing corresponding resilience metrics from system-performance data observed before, during, and after disruptive events. We evaluate the proposed approach using datasets from two distinct operational domain network systems and an airport transportation system, demonstrating its domain-agnostic applicability to resilience assessment. The results demonstrate the effectiveness of re
    
[^70]: 面向6G O-RAN中O-CU与O-DU映射的可行性感知深度强化学习

    Feasibility-Aware Deep Reinforcement Learning for O-CU and O-DU Mapping in 6G O-RAN

    [https://arxiv.org/abs/2611.34516](https://arxiv.org/abs/2611.34516)

    本文提出一种可行性感知的深度强化学习方法，利用DQN和Double DQN将6G O-RAN中O-CU和O-DU的虚拟网络功能组件高效映射到虚拟机上，在保证可行性的同时显著降低传统优化方法的计算成本。

    

    本文研究在第六代（6G）开放无线接入网（O-RAN）中，将O-RAN中央单元（O-CU）和O-RAN分布式单元（O-DU）的虚拟网络功能组件（VNFC）映射到虚拟机（VM）上的问题。该问题被建模为一个可行性感知的序贯决策过程，其首要目标是最大化已部署的VNFC数量，其次是偏好更少的活跃虚拟机数量和更低的剩余资源浪费。尽管传统优化技术可以解决该问题，但在更大规模实例或需要重复执行优化时，其计算成本会变得难以承受。为解决这一局限，本文探索了两种基于神经强化学习（RL）的方法：深度Q网络（DQN）和双深度Q网络。DQN使用同一个目标网络来选择和评估自举动作，而Double DQN则将这两个操作解耦，以缓解由最大化操作引起的过估计问题。

    arXiv:2611.34516v1 Announce Type: cross  Abstract: This paper studies the mapping of the virtual network function components (VNFCs) of the O-RAN central unit (O-CU) and O-RAN distributed unit (O-DU) onto virtual machines (VMs) in sixth-generation (6G) open radio access networks (O-RAN). The problem is formulated as a feasibility-aware sequential decision process that maximizes the number of placed VNFCs and, secondarily, favors fewer active VMs and lower residual resource wastage. Although conventional optimization techniques can solve this problem, their computational cost becomes prohibitive for larger instances or when optimization must be repeated. To address this limitation, we explore two neural reinforcement learning (RL)-based methods: a deep $Q$-network (DQN) and a double deep $Q$-network (Double DQN). Whereas DQN uses the same target network to both select and evaluate the bootstrap action, Double DQN decouples these two operations to mitigate maximization-induced overestima
    
[^71]: SMOTE的25年魔力：已做之事、证据所示及其影响力所及

    SMOTE 25 Years of Magic: What Has Been Done, What the Evidence Shows, and Whom It Reached

    [https://arxiv.org/abs/2611.34511](https://arxiv.org/abs/2611.34511)

    在SMOTE问世25周年之际，本研究通过审计76篇论文语料库并对22,426条记录进行系统性检索，评估了SMOTE家族方法的改进内容、证据强度及其跨学科的影响力范围。

    

    背景。SMOTE于2002年发表在本期刊上，并于2018年其十五周年之际在本刊进行了回顾综述。至其二十五周年时，它已被超过34,000篇收录文献引用，其中一半来自计算机科学领域之外。目标。确定基于SMOTE构建的方法改变了什么、这些方法的证据强度如何，以及该技术已经触及了哪些领域。方法。我们审计了一个精选的76篇论文语料库，其中37篇提出了SMOTE家族的生成器，并通过以PRISMA 2020格式对22,426条记录进行系统检索来检验该语料库的覆盖范围。我们从语料库中收集了1,756个数据集条目、1,511项与SMOTE的已发表比较以及71项研究的评估设计；并从系统检索中随机抽取了300项SMOTE的应用案例。我们采用智能体AI方法在既定书面规则下进行提取、筛选和编码，并报告了每个步骤的核查方式。结果。每个变体都会重新决定SMOTE规则的六个决策中的某些部分，其中24个……（摘要在此处截断）

    arXiv:2611.34511v1 Announce Type: new  Abstract: Background. SMOTE was published in this journal in 2002 and reviewed here at its fifteenth anniversary in 2018. At twenty-five it is cited by more than 34,000 indexed works, half of them outside computer science. Objectives. To establish what the methods built on SMOTE change, how strong the evidence for them is, and whom the technique has reached. Methods. We audited a curated corpus of 76 papers, 37 of which propose a SMOTE-family generator, and tested its coverage with a systematic search of 22,426 records in the PRISMA 2020 format. From the corpus we collected 1,756 dataset entries, 1,511 published comparisons with SMOTE and the evaluation design of 71 studies; from the search we drew 300 applications of SMOTE at random. We used an agentic AI approach to extract, screen and code under our written rules, and we report how each step was checked. Results. Every variant re-decides some of the six decisions of the SMOTE rule, and 24 of th
    
[^72]: 分布鲁棒优化中的高效数据融合

    Efficient Data Fusion in Distributionally Robust Optimization

    [https://arxiv.org/abs/2611.34495](https://arxiv.org/abs/2611.34495)

    本文提出仅通过函数型拉格朗日乘子耦合各模糊集的融合分布鲁棒优化精确重构方法，并利用仿射决策规则近似得到可分解的高效上界，从而克服了传统方法随数据源数量呈指数级增长的计算瓶颈。

    

    不确定性下的决策日益依赖于碎片化的证据，例如来自相关但不同总体的样本、隐私保护的汇总统计数据，或对同一数据的互补统计描述。分布鲁棒优化（DRO）可以通过为每个数据源构建一个模糊集，并对这些模糊集交集内的所有分布进行对冲来融合此类证据。这种融合方法比将所有数据源聚合为单一模糊集更少保守性，但计算代价高昂：精确重构会随数据源数量呈指数级增长，甚至判断交集是否非空本身就是NP难问题。本文证明融合DRO存在一种精确重构形式，其中各模糊集仅通过函数型拉格朗日乘子耦合。将这些乘子限制为仿射决策规则即可得到可在各模糊集间分解的上界，并在一定条件下具有精确性（摘要在此处截断）。

    arXiv:2611.34495v1 Announce Type: cross  Abstract: Decisions under uncertainty increasingly rest on fragmented evidence, such as samples from related but different populations, privacy-protected summary statistics, or complementary statistical descriptions of the same data. Distributionally robust optimization (DRO) can fuse such evidence by constructing one ambiguity set per source and hedging against all distributions in the intersection of these sets. This fusion approach is less conservative than aggregating the sources into a single ambiguity set, but it is computationally demanding. Exact reformulations grow exponentially with the number of sources, and even deciding whether the intersection is nonempty is NP-hard. We show that fusion DRO admits an exact reformulation in which the ambiguity sets are coupled only through functional Lagrange multipliers. Restricting these multipliers to affine decision rules yields upper bounds that decompose across the ambiguity sets, are exact fo
    
[^73]: 审计运营日志中的身份与时间信号

    Auditing Identity and Timing Signals in Operational Logs

    [https://arxiv.org/abs/2611.34493](https://arxiv.org/abs/2611.34493)

    该论文提出身份审计与时间审计两种简单方法，可在训练事件预测模型之前量化运营日志中的身份与时间信息泄露，并发现ICU数据中常用的“向前填充”预处理会显著夸大模型性能（身份AUC从0.916升至0.993，脓毒症预测AUC虚增0.163）。

    

    从运营日志中预测事件的模型，例如基于ICU记录预测脓毒症或预测硬盘故障，可以通过识别记录来自哪个患者或硬盘、或记录是何时写入的来取得高分，而不是真正预测事件本身。我提出两个简单的审计方法，用于在任何事件预测模型训练之前衡量数据集携带了多少此类信息，并将其应用于六个领域的七个公开数据集。其中一个审计检验分类器能否判断两条记录是否来自同一单元；另一个审计检验来自同一单元的两条记录中哪一条时间更晚。在给定零假设下，两者的期望AUC均为0.5。在ICU数据中，身份审计揭示了一个常见预处理步骤的问题：将最近一次测量值向前填充会使身份AUC从0.916升至0.993。在一个12小时脓毒症预测任务中，当某患者的小时数据被同时分入训练集和测试集时，向前填充使模型表现出的AUC比使用原始值高出0.163，但仅……（原文摘要在此处截断）

    arXiv:2611.34493v1 Announce Type: new  Abstract: Models that predict events from operational logs, such as sepsis in ICU records or failures in hard drives, can score well by recognizing which patient or drive a record came from, or when it was recorded, instead of predicting the event. I introduce two simple audits that measure how much of this information a dataset carries before any event-prediction model is trained, and apply them to seven public datasets in six domains. One asks whether a classifier can tell that two records came from the same unit; the other asks which of two records from the same unit came later. Under the stated nulls, both have expected AUC 0.5. In ICU data, the identity audit flags a common preprocessing step: carrying the last measurement forward raises identity AUC from 0.916 to 0.993. On a 12-hour sepsis forecasting task, carrying forward makes a model look 0.163 AUC better than raw values when a patient's hours are split between training and test, but onl
    
[^74]: FlyingStart：面向递归深度语言模型不动点解码的学习型跨Token热启动方法

    FlyingStart: Learned Cross-Token Warm Starts for Fixed-Point Decoding in Recurrent-Depth Language Models

    [https://arxiv.org/abs/2611.34492](https://arxiv.org/abs/2611.34492)

    提出FlyingStart方法，利用上一个Token停止时的状态经闭式解拟合的线性变换来热启动下一个Token的迭代，在精度不变的前提下将递归深度语言模型的解码循环层数减少14%–19%。

    

    递归深度语言模型通过反复迭代一个共享的层块来生成每个Token，直到其隐藏状态收敛。针对这类模型的推理方法只决定何时停止迭代，而每个Token仍然从随机噪声开始迭代，并为此付出完整收敛过程的代价。我们证明，迭代的起始点是第二个独立的优化杠杆。在学习到的坐标系变换之后，一个Token的不动点可以从上一个Token停止迭代时的状态预测出来。基于此，我们提出FlyingStart——一种线性热启动方法，它在冻结模型上以闭式解拟合，成本不到一次循环的5%。迭代过程及其停止规则保持不变，因此退出准则从任意起点都能保证到不动点的相同距离：更好的起点能在精度不变的情况下减少循环次数。在Huginn-0125模型上，FlyingStart以减少14%–19%的解码层数达到了最强免训练基线的精度，同时提升了批量推理吞吐量……

    arXiv:2611.34492v1 Announce Type: new  Abstract: Recurrent-depth language models generate each token by iterating a shared block of layers until its hidden state converges. Inference methods for these models decide when to stop the iteration, yet every token still starts from random noise and pays the full cost of converging. We show that where the iteration starts is a second, independent lever. A token's fixed point is predictable from the state at which the previous token stopped, after a learned change of coordinates, and we introduce FlyingStart, a linear warm start that is fitted in closed form on the frozen model and costs under 5% of one loop. The iteration and its stopping rule are unchanged, so the exit criterion certifies the same distance to the fixed point from any start: a better start removes loops at unchanged accuracy. On Huginn-0125, FlyingStart matches the accuracy of the strongest training-free baseline with 14--19% fewer decoding layers, raises batched throughput b
    
[^75]: 反向传播作为耗散训练的狄拉克括号动力学

    Backpropagation as Dirac-bracket dynamics of dissipative training

    [https://arxiv.org/abs/2611.34479](https://arxiv.org/abs/2611.34479)

    该论文通过 Bateman-Galley 自由度倍增将耗散的梯度训练方法（梯度流、重球法、Nesterov 加速梯度）表述为哈密顿系统，并把网络激活值视为第二类约束，证明反向传播本质上是狄拉克括号演化，从而为反向传播给出了严格的哈密顿力学几何解释。

    

    基于梯度的参数化模型训练可以看作是阻尼粒子在由损失函数设定的势场中的运动。利用 Bateman-Galley 自由度倍增方法（该方法赋予耗散系统一个作用量原理），我们将梯度流、重球法和 Nesterov 加速梯度方法写成哈密顿系统。我们证明，在这一哈密顿描述中，反向传播就是狄拉克演化。前馈网络的激活值被视为相空间的坐标，层方程被视为约束。对于任何网络架构，这些约束都是第二类约束，而带有由此得到的狄拉克括号的哈密顿方程重现了使用损失函数精确梯度的训练过程，尽管哈密顿量中只包含网络输出的代价。激活值与参数动量之间的狄拉克括号正是其雅可比矩阵，而约束矩阵的逆同时包含了前向传播……（原文摘要在此处截断）

    arXiv:2611.34479v1 Announce Type: cross  Abstract: Gradient-based training of a parametric model is the motion of a damped particle in the potential set by the loss. Using the Bateman-Galley doubling of the degrees of freedom, which gives dissipative systems an action principle, we write gradient flow, the heavy-ball method and Nesterov's accelerated gradient method as Hamiltonian systems. We show that, within this Hamiltonian description, backpropagation is Dirac evolution. The activations of a feed-forward network are treated as coordinates of phase space, and the layer equations as constraints. These constraints are second class for every architecture, and Hamilton's equations with the resulting Dirac bracket reproduce training with the exact gradient of the loss, although the Hamiltonian contains only the cost of the network output. The Dirac bracket of an activation with the momenta of the parameters is its Jacobian, the inverse of the constraint matrix contains both the forward a
    
[^76]: DiversePlace：面向宏单元布局的多样性探索课程强化学习

    DiversePlace: Diversity-Seeking Curriculum Reinforcement Learning for Macro Placement

    [https://arxiv.org/abs/2611.34472](https://arxiv.org/abs/2611.34472)

    该论文提出DiversePlace，一种将质量受控的多样性作为显式目标的课程强化学习方法，通过维护多个HPWL具有竞争力且彼此显著不同的宏单元布局存档，为下游选择提供多样化的高质量方案。

    

    宏单元布局是物理设计中至关重要的早期决策，因为宏单元的位置会强烈约束后续的实现阶段。这项任务之所以困难，是因为这些高影响力的离散决策必须在下游质量能够被准确评估之前做出，迫使训练只能依赖快速但不完美的训练时代理指标（oracle），例如半周长线长（HPWL）。因此，现有的基于强化学习的布局器大多只优化单一的HPWL驱动解决方案，其较长的序列化决策时程往往需要繁重的搜索辅助或离线专家数据，同时几乎没有保留多个具有竞争力的布局方案的空间。我们通过将质量受控的多样性作为宏单元布局的显式目标来弥补这一空白。我们不依赖单一的代理最优解，而是维护一个紧凑的存档，其中保存了在HPWL上具有竞争力但彼此显著不同的布局方案，供下游选择使用。我们在DiversePlace中实现了这一想法，这是一种课程强化学习方法……

    arXiv:2611.34472v1 Announce Type: new  Abstract: Macro placement is a critical early-stage decision in physical design because macro locations strongly constrain subsequent implementation stages. The task is difficult because these high-impact discrete decisions must be made before downstream quality can be measured accurately, forcing training to rely on fast but imperfect training-time oracles such as half-perimeter wirelength (HPWL). Existing RL-based placers therefore mostly optimize single HPWL-driven solutions, and their long sequential horizons often require heavy search assistance or offline expert data while leaving little room to preserve multiple competitive layouts. We address this gap by making quality-controlled diversity an explicit objective for macro placement. Instead of relying on a single proxy-optimal solution, we maintain a compact archive of HPWL-competitive yet meaningfully different placements for downstream selection. We realize this idea in DiversePlace, a cu
    
[^77]: 过参数化极限下物理信息神经网络二阶优化方法的全局收敛性

    Global Convergence of Second-Order Methods for Physics-Informed Neural Networks in the Overparameterized Limit

    [https://arxiv.org/abs/2611.34452](https://arxiv.org/abs/2611.34452)

    本文首次为物理信息神经网络训练中的正则化牛顿法和Gauss-牛顿法建立了全局收敛理论，证明了在过参数化极限下训练动力学以显式速率概率收敛到确定性极限方程的解。

    

    求解高维偏微分方程（PDEs）是许多工程与科学领域的核心任务。物理信息神经网络（PINNs）和深度Galerkin方法通过最小化与PDE算子对应的残差以及外部约束，有效地利用机器学习模型来逼近已知PDE的解。尽管二阶优化器在训练神经网络求解PDE方面日益流行，并在速度和精度上都优于其一阶方法，但训练此类模型的收敛理论基本仅限于一阶方法。在本文中，我们填补了这一空白，为正则化牛顿法和Gauss-牛顿法建立了收敛性分析。当隐藏单元数量趋于无穷时，训练动力学以概率收敛（并具有显式收敛速率）到某个确定性极限方程的解。

    arXiv:2611.34452v1 Announce Type: cross  Abstract: Solving high-dimensional partial differential equations (PDEs) constitutes a core task in many engineering and scientific disciplines. Physics-informed neural networks and the deep Galerkin method effectively leverage machine learning models to approximate solutions of known PDEs by minimizing residuals corresponding to the PDE operator alongside external constraints. Although second-order optimizers are becoming increasingly popular for training neural networks (NNs) to solve PDEs and empirically outperform their first-order counterparts in both speed and accuracy, the convergence theory for training such models is essentially confined to first-order methods. In this paper, we close this gap and develop a convergence analysis for the regularized Newton and Gauss-Newton methods. As the number of hidden units tends to infinity, the training dynamics converge in probability, with explicit rates, to the solution of a deterministic limit e
    
[^78]: 面向人机协作中脑电（EEG）融合选择性采用的决策对齐价值学习

    Decision-Aligned Value Learning for Selective Adoption of EEG Fusion in Human-Robot Collaboration

    [https://arxiv.org/abs/2611.34451](https://arxiv.org/abs/2611.34451)

    该论文提出VECTA框架，将人机协作中EEG信息的选择性采用建模为逐样本增量价值估计问题，通过分离融合预测生成与采用决策，并借助系统增益和匹配增益来判断EEG融合何时能真正改善预测。

    

    脑电图（EEG）能够捕捉与人类感知和决策相关的神经活动，为人机协作（HRC）系统提供用于事件识别、态势评估和行动准备的可测量证据，这是外部传感手段难以提供的。然而，这类证据的质量和互补性因样本而异，现有的EEG融合方法并未明确判断EEG何时能够真正改善预测。因此，本文将EEG的选择性使用构建为一个逐样本的增量价值估计问题，并提出了价值估计与约束感知的采用决策框架（VECTA）。VECTA将EEG融合预测的生成过程与是否用其替换现有基础预测的决策过程分离开来。该框架利用系统增益来量化从基础预测到融合预测的损失降低程度，并利用匹配增益来评估收益是否来自样本匹配……（摘要原文在此处截断）

    arXiv:2611.34451v1 Announce Type: cross  Abstract: Electroencephalography (EEG) captures neural activity associated with human perception and decision-making. It provides human-robot collaboration (HRC) systems with measurable evidence for event recognition, situation assessment, and action preparation. External sensing cannot easily provide such evidence. However, the quality and complementarity of this evidence vary across samples, and existing EEG fusion methods do not explicitly determine when EEG improves a prediction. We therefore framed selective EEG use as a sample-wise incremental-value estimation problem and proposed Value Estimation and Constraint-Aware Triage for Adoption (VECTA). VECTA separates the generation of an EEG-informed Fusion prediction from the decision about whether it should replace the existing Base prediction. The framework uses system gain to quantify the loss reduction from Base to Fusion and matching gain to assess whether benefits arise from sample-match
    
[^79]: vSFT：在测试时通过语言反馈微调智能体

    vSFT: Learning to Verbally Tune Agents at Test Time

    [https://arxiv.org/abs/2611.34440](https://arxiv.org/abs/2611.34440)

    提出vSFT框架，通过超网络将监督者基于智能体失败轨迹生成的自然语言反馈转化为低秩权重更新，实现测试时的参数内演化，在ALFWorld等任务上性能超越GRPO和Reflexion等主流强化学习方法。

    

    大语言模型智能体越来越多地通过智能体强化学习进行改进，这种方法可以采用在策略方式，但存在 rollout 成本高昂和奖励稀疏的问题，限制了其在测试时的应用。本文提出了一种语言监督微调框架，实现了自然语言密集反馈与测试时权重更新的结合。具体而言，我们的vSFT通过一个超网络在智能体与监督者之间构建闭环：监督者从智能体的失败轨迹中生成语言监督信号，超网络则将这些监督信号转化为低秩权重更新以调整智能体。我们验证了这种测试时的参数内演化过程优于上下文学习对应方法以及主流强化学习方法。特别地，基于Qwen3-4B的vSFT在ALFWorld上的成功率分别比GRPO和Reflexion高出8.7%和13.9%，并比另一方法高出4.8%（原文摘要此处截断）。

    arXiv:2611.34440v1 Announce Type: new  Abstract: Large language model agents are increasingly improved via agentic reinforcement learning, which can be on-policy but suffers from expensive rollouts and sparse rewards, limiting its application at test time. In this paper, we present a verbally Supervised Fine-Tuning (vSFT) framework that enables the conjunction of dense feedback in natural language and weight updates at test time. Specifically, our vSFT forms a closed loop between an agent and a supervisor via a hypernetwork, in which the supervisor generates verbal supervision from the agent's failed trajectories, and the hypernetwork converts that supervision into low-rank weight updates to tune the agent. We validate that this test-time in-parameter evolving process outperforms both its in-context counterparts and prevalent reinforcement learning methods. In particular, our vSFT using Qwen3-4B outperforms GRPO and Reflexion by 8.7% and 13.9% in success rate on ALFWorld, and by 4.8% a
    
[^80]: 一个表示统治一切——基于文本或视觉示例的开放词汇变化检测

    One Representation to Rule Them All -- Open-Vocabulary Change Detection from Text or Visual Examples

    [https://arxiv.org/abs/2611.34430](https://arxiv.org/abs/2611.34430)

    提出免训练框架OVTSID3，在DINOv3表示空间中统一区域分组、概念识别与时间比较，支持文本或视觉示例两种方式指定变化目标，使开放词汇变化检测的平均F1较现有免训练方法提升8.9个百分点。

    

    开放词汇变化检测（OVCD）旨在无需为预定义类别重新训练的情况下定位用户指定的变化。现有方法通常依赖文本描述，并组合用于区域提议、语义识别和时间比较的多个独立模型。这限制了目标概念的表达方式，并使检测依赖于多个中间预测。我们提出了OVTSID3，这是一个免训练框架，它在DINOv3的表示空间中统一了区域分组、概念识别和时间比较。OVTSID3支持两种互补的概念指定模式：零样本文本查询或通过掩码参考对进行的单样本视觉提示。OVTSID3在两个建筑和两个自然灾害基准上优于所有先前的免训练OVCD方法，将平均F1提高了8.9个百分点。OVTSID3还支持语义变化检测，并在SEC上取得了最先进的结果。

    arXiv:2611.34430v1 Announce Type: cross  Abstract: Open-vocabulary change detection (OVCD) aims to localise user-specified changes without retraining for predefined categories. Existing approaches typically rely on textual descriptions and combine separate models for region proposal, semantic recognition, and temporal comparison. This limits how target concepts can be expressed and makes detection dependent on several intermediate predictions. We introduce OVTSID3, a training-free framework that unifies region grouping, concept identification, and temporal comparison in the representation space of DINOv3. OVTSID3 supports two complementary modes of concept specification: zero-shot textual queries or one-shot visual prompting through a masked reference pair. OVTSID3 outperforms all prior training-free OVCD methods on two building and two natural-disaster benchmarks, raising the average F1 by 8.9 points. OVTSID3 also supports semantic change detection and achieves state-of-the-art on SEC
    
[^81]: 面向合成语音检测的自监督骨干网络的头部级研究

    A Head-Level Investigative Study of Self-Supervised Backbones for Synthetic Speech Detection

    [https://arxiv.org/abs/2611.34426](https://arxiv.org/abs/2611.34426)

    该研究首次对自监督语音骨干网络进行注意力头级别分析，发现仅少数注意力头主导合成语音检测决策，移除对抗性头部无需重新训练即可将等错误率相对降低 13%–22%，且伪造语音与真实语音之间的注意力集中度差异本身即可几乎替代完整表示实现真假语音区分。

    

    自监督学习（SSL）语音骨干网络是合成语音检测的标准前端，逐层探测分析已经确定了哪些层携带最多的信号。但分析止步于此：一层中的十二个注意力头中究竟是哪些在计算欺骗决策仍是未知的。我们在 ASVspoof 5 数据集上首次对两个冻结的骨干网络 WavLM-base 和 Wav2Vec2-base 进行了头部级研究。逐头消融实验表明，只有少数注意力头携带决策信息，而其他头部则主动对抗该决策；移除这些头部无需重新训练即可降低等错误率（EER），在 WavLM 中相对降低 13%，在 Wav2Vec2 中相对降低 22%。我们还发现了一种无需探测器的特征：伪造语音比真实语音的注意力更加集中，仅凭这一差距就能几乎与完整表示一样好地区分真假语音，且在每个骨干网络最具判别力的层上达到峰值。最后，攻击身份可以从每个头部中被解码出来，但各头部之间的差异在于它们执行的功能，而不在于它们感知的内容。

    arXiv:2611.34426v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) speech backbones are the standard frontend for synthetic-speech detection, and layer-wise probing has established which layers carry the most signal. But analysis stops there: which of a layer's twelve attention heads compute the spoof decision is unknown. We present the first head-level study of two frozen backbones, WavLM-base and Wav2Vec2-base, on ASVspoof 5. Per-head ablation shows that only a few heads carry the decision while others actively oppose it; removing those heads lowers equal error rate (EER) with no retraining, by 13% relative in WavLM and 22% in Wav2Vec2. We also find a probe-free signature: spoofed speech draws more focused attention than genuine speech, and this gap alone separates the two almost as well as the full representation, peaking at each backbone's most discriminative layer. Finally, attack identity is decodable from every head, yet heads differ in what they do, not in what t
    
[^82]: 面向多阶段随机规划的决策聚焦学习及其在电网修复调度中的应用

    Decision-Focused Learning for Multistage Stochastic Programs, with Application to Power Network Repair Scheduling

    [https://arxiv.org/abs/2611.34419](https://arxiv.org/abs/2611.34419)

    该论文提出了一种面向多阶段随机规划的决策聚焦学习框架，利用神经网络预测驱动状态转移的决策得分，并通过带扰动优化器的Fenchel-Young损失进行训练，成功应用于随机故障下的电网修复调度问题。

    

    具有离散决策的多阶段随机规划（MSP）由于维数灾难、非凸值函数以及随时间范围呈指数增长的情景树而难以求解。我们为MSP开发了一个决策聚焦学习（DFL）框架，其中神经网络对驱动状态转移的那部分决策进行得分预测。在使该子决策以线性方式进入阶段目标的结构性条件下，网络通过离线单情景问题的标签进行训练，采用带扰动优化器的Fenchel-Young损失，该方法只需一个优化预言机。我们将该框架应用于随机、时变组件故障下的电网修复调度问题，涵盖有和没有每阶段修复预算两种情形。在合成的5节点网络以及两种修复成本规模下的IEEE 14节点网络上，对学习到的策略与基线策略在常见评估指标上进行了比较。

    arXiv:2611.34419v1 Announce Type: cross  Abstract: Multistage stochastic programs (MSPs) with discrete decisions are hard to solve due to the curse of dimensionality, non-convex value functions, and scenario trees that grow exponentially with the horizon. We develop a decision-focused learning (DFL) framework for MSPs in which a neural network predicts scores for the part of the decision that drives the state transition. Under structural conditions that make this sub-decision enter the stage objective linearly, the network is trained on labels from offline single-scenario problems, using the Fenchel-Young loss with perturbed optimizers, which need only an optimization oracle. We apply the framework to repair scheduling in power networks under stochastic, time-varying component failures, with and without a per-stage repair budget. On a synthetic 5-bus network and on the IEEE 14-bus network at two repair-cost scales, the learned and baseline policies are compared on common evaluation pat
    
[^83]: 面向自动驾驶闭环安全性的可恢复性策略编辑

    Recoverability policy editing for closed-loop safety in autonomous driving

    [https://arxiv.org/abs/2611.34403](https://arxiv.org/abs/2611.34403)

    该论文提出了一种状态级策略编辑技术，通过引入策略可恢复性概念，在不改变模型原有驾驶能力的前提下，保证自动驾驶在可恢复扰动下的闭环安全性，在nuPlan场景中将安全事故减少了48%。

    

    基于神经网络模型驱动的自动驾驶缺乏安全性保证。我们提出了一种状态级的策略编辑技术，在可恢复的扰动下保证闭环安全性，同时复用模型现有的驾驶能力。闭环安全是指车辆在所有未来时刻都保持无碰撞状态。我们引入了策略可恢复性的概念，用于表征车辆能否在有限时间内安全地返回到原始策略的可行集。我们的方法识别出在原始策略下长期不安全但可恢复的状态，并在该区域内选择性地编辑策略输出。车辆在进入该区域时会纠正其动作，并在恢复后重新执行原始策略。在nuPlan城市驾驶场景上的闭环仿真实验显示，与原始策略相比，安全事故减少了48%，同时在很大程度上保留了其驾驶性能。

    arXiv:2611.34403v1 Announce Type: cross  Abstract: Autonomous driving powered by neural network models lacks safety guarantees. We propose a state-level policy editing technique that guarantees closed-loop safety under recoverable disturbances while reusing the model's existing driving capabilities. Closed-loop safety means that the vehicle remains free of collisions at all future times. We introduce the concept of policy recoverability to characterize whether a vehicle can safely return to the original policy's feasible set in finite time. Our method identifies states that are unsafe in the long term under the original policy but recoverable, and selectively edits policy outputs within this region. The vehicle corrects its actions upon entering this region and resumes the original policy upon recovery. Closed-loop simulations on nuPlan urban driving scenarios show a 48% reduction in safety incidents under the original policy while largely preserving its driving performance.
    
[^84]: Any-Rate：基于编码位平面的半比特多精度大语言模型量化

    Any-Rate: Half-Bit Multi-Precision LLM Quantization with Coded Bit-Planes

    [https://arxiv.org/abs/2611.34402](https://arxiv.org/abs/2611.34402)

    Any-Rate通过将位平面约束为8权重块的扩展汉明码字并结合逐精度联合编码，实现了从单一检查点以半比特为步长提供2–8比特的多精度LLM量化，在2比特和3比特下达到最先进的准确率。

    

    嵌套多精度量化通过仅读取每个权重的最高有效位，从单一检查点派生出多种精度，这使其在各种内存和能耗预算下的端侧部署中极具吸引力。然而，现有方法仅支持整数比特，导致相邻精度之间存在较大的准确率差距，并且在2比特时出现严重的性能退化。为解决这一问题，我们提出了Any-Rate，它能够从单一检查点提供2至8比特的精度服务，并在2至4比特范围内以半比特为步长进行调节。Any-Rate将部分位平面限制为8权重块中的扩展汉明码字（每权重0.5比特），在各精度之间共享每行一个查找表（LUT），并对所有精度进行联合编码，同时为每个精度设置独立的优化目标，从而降低了共享位平面带来的损失。在四个大语言模型上，Any-Rate在2比特和3比特下实现了多精度方法中最先进的准确率，在2比特下保持了FP16准确率的69%，在4比特下与最佳方法相差不到一个百分点。它解码……

    arXiv:2611.34402v1 Announce Type: cross  Abstract: Nested multi-precision quantization derives multiple precisions from a single checkpoint by reading only the leading bits of each weight, which makes it attractive for on-device deployment under varying memory and energy budgets. However, existing methods offer only integer bits, leaving large accuracy gaps between adjacent precisions and severe degradation at 2 bits. To address this, we propose Any-Rate, which serves 2-8 bits from one checkpoint, in half-bit steps over 2-4 bits. Any-Rate restricts some bit-planes to extended Hamming codewords in 8-weight blocks (0.5 bits per weight), shares one LUT per row across precisions, and encodes all precisions jointly with a separate target for each, reducing the loss from sharing bit-planes. Across four LLMs, Any-Rate achieves state-of-the-art accuracy among multi-precision methods at 2 and 3 bits, keeping 69% of FP16 accuracy at 2 bits, and is within one point of the best at 4 bits. It decod
    
[^85]: 强大的起点，简单的配方：基于多模态专家模型的统一多教师在线策略蒸馏

    Strong Starts, Simple Recipes: Unified Multi-Teacher On-Policy Distillation from Multimodal Specialists

    [https://arxiv.org/abs/2611.34397](https://arxiv.org/abs/2611.34397)

    该论文提出Uni-MOPD方法，将学生模型的起点作为设计变量——先用无训练的专家模型均匀合并获得强起点，再用简短的多教师在线策略蒸馏配方进行精炼，并发现起点选择比后续训练策略对最终性能的影响更大。

    

    多教师在线策略蒸馏（MOPD）将分别训练的领域专家模型整合到一个统一模型中。现有方法专注于改进蒸馏过程，而学生模型则从基础模型或多领域SFT检查点初始化，这使得能力整合在很大程度上依赖于蒸馏本身。相反，我们将起点视为一个设计变量，提出了统一MOPD（Uni-MOPD），它将强大的起点与简单的配方相结合：使用来自同一基础模型的专家模型的无训练均匀合并来初始化学生模型，随后使用均匀领域采样和全词表反向KL散度的短程MOPD对其进行精炼。在五个多模态领域和19个基准测试上，我们的研究确立了三个发现。（1）起点比后续训练选择更重要：在任何训练之前，均匀合并已经达到了基础模型经过200次MOPD更新后才能达到的分数……

    arXiv:2611.34397v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) integrates separately trained domain specialists into a single unified model. Existing methods focus on refining the distillation procedure while initializing the student from a base model or a multi-domain SFT checkpoint, which leaves capability integration largely to distillation itself. Instead, we treat the starting point as a design variable and propose Unified MOPD (Uni-MOPD), which pairs a strong start with a simple recipe: a training-free uniform merge of specialists from the same base model initializes the student, and short MOPD with uniform domain sampling and full-vocabulary reverse KL refines it. Across five multimodal domains and 19 benchmarks, our study establishes three findings. (1) The starting point matters more than subsequent training choices: before any training, the uniform merge already matches the score that the base model reaches after 200 MOPD updates, and after sho
    
[^86]: 合理不等于证实：测试单目深度估计中的失败解释

    Plausible Is Not Proven: Testing Failure Explanations in Monocular Depth Estimation

    [https://arxiv.org/abs/2611.34393](https://arxiv.org/abs/2611.34393)

    提出视觉误差观测台（VEO）审计框架，通过四项独立检验来验证单目深度估计失败原因标签的可信度，发现遮挡边界标签虽有解释力但并不优于简单的深度边缘线索。

    

    单目深度模型的失败分析通常会为每个错误指认一个原因，例如遮挡边界、细小物体或反射表面，但这类标签很少经过实际检验。我们提出了视觉误差观测台（Vision Error Observatory, VEO），这是一种审计方法，它在传感器测量像素上找出未通过 δ1 准则的区域，为每个区域标注一个可能的原因或选择弃权，然后对标签施加四项检验：与标签模型从未见过的数据上物体标注的一致性、与盲测人类评分者的一致性、对针对性图像编辑的响应，以及作为训练信号的价值。我们将其应用于 NYU Depth V2、DIODE 和 iBims-1 数据集上的八个模型，并在 Hypersim 上对标签进行额外检验，且大多数决策规则在看到结果之前就已固定。遮挡边界标签与物体标注一致（NYU 上 AUROC 为 0.85，Hypersim 上为 0.81，iBims-1 上为 0.69），但单一的深度边缘线索也能达到同样效果，并且最强模型的大多数边界失败在 2-

    arXiv:2611.34393v1 Announce Type: cross  Abstract: Failure analyses of monocular depth models often name a cause for each error, such as an occluding boundary, a thin object or a reflective surface, but such labels are rarely tested. We present the Vision Error Observatory (VEO), an audit that finds regions failing the $\delta_1$ criterion on sensor-measured pixels, labels each with a likely cause or abstains, and then subjects the labels to four tests: agreement with object annotations on data the label model never saw, agreement with blinded human raters, response to targeted image edits, and value as a training signal. We apply it to eight models on NYU Depth V2, DIODE and iBims-1, with labels also checked on Hypersim, and fix most decision rules before their results. The occluding-boundary label agrees with annotations (AUROC 0.85 on NYU, 0.81 on Hypersim, 0.69 on iBims-1), but a single depth-edge cue does as well, and most boundary failures of the strongest model vanish under a 2-
    
[^87]: 解耦脉冲神经网络中学习与计算的时钟

    Decoupling the clocks of learning and computing in spiking neural networks

    [https://arxiv.org/abs/2611.34381](https://arxiv.org/abs/2611.34381)

    本文提出 ChronoProp 方法，通过解耦学习与计算的时钟，利用神经元在固定时间窗内的脉冲计数和实测输入输出函数来计算权重更新，在图像分类中达到与通过时间反向传播相似的准确率，同时训练时间缩短约3倍、内存占用降低约20倍。

    

    生物智能尽管其组成部件嘈杂且不完美，却能以极低的功耗高效运转，其神经元通过电压尖峰进行通信。类脑计算旨在将这种效率带入真实设备中，这需要为其量身定制的训练方法。训练脉冲神经网络的标准方法——通过时间的反向传播——需要在模拟设备的每一步都计算权重更新，因此高保真或长时间的模拟会使训练成本高昂且不稳定。在这里，我们展示了解耦学习时钟与计算时钟后，脉冲网络能够可靠且高效地学习。我们的方法 ChronoProp 利用每个神经元实测的输入-输出函数，根据其在固定时间窗内的脉冲计数来计算权重更新。与通过时间的反向传播相比，ChronoProp 在图像分类任务中达到了相似的准确率，同时训练时间缩短约3倍，在最细时间步长下内存占用降低约20倍。

    arXiv:2611.34381v1 Announce Type: cross  Abstract: Biological intelligence thrives on minimal power despite its noisy, imperfect components, with neurons that communicate through voltage spikes. Brain-inspired computing seeks to bring such efficiency to real devices, which requires training methods tailored to them. The standard way to train spiking neural networks, backpropagation through time, computes weight updates through every step of a simulated device, so faithful or long simulations can make training costly and unstable. Here we show that spiking networks can learn reliably and efficiently by decoupling the clock of learning from that of computing. Our method, ChronoProp, calculates weight updates from each neuron's spike counts in fixed time bins using its measured input-output function. Compared with backpropagation through time, ChronoProp reaches similar accuracy in image classification, with about 3 times less training time and 20 times less memory at the finest timestep 
    
[^88]: 影响之下——受什么影响？影响函数的受控评估

    Under the Influence -- of What? A Controlled Evaluation of Influence Functions

    [https://arxiv.org/abs/2611.34373](https://arxiv.org/abs/2611.34373)

    这是首个通过受控数据划分重新训练的系统评估，验证了影响函数在GPT-2类模型中识别的“有影响力”训练样本确实与输出相关，但不总是直接承载目标属性，且相关性因任务设置而异。

    

    影响函数是一种有前景的方法，用于识别哪些训练样本对模型的输出负有责任。然而，迄今为止在大语言模型预训练层面的实证评估都只是假设它们按预期工作。这项工作是首个系统性评估“有影响力”样本是否真的具有与语言模型输出相关属性的研究，其方法是基于受控数据划分进行重新训练。我们在GPT-2类型的模型（6800万至56亿训练词元）中，针对两种设置（语言现象和事实知识），研究了三种著名的Hessian近似方法（GradSim、TrackStar、K-FAC）。我们发现，平均而言，“有影响力”的样本确实是相关的，但它们并不总是直接承载目标属性，且相关性的比例和类型因设置而异。

    arXiv:2611.34373v1 Announce Type: new  Abstract: Influence functions are a promising way to identify which training examples are responsible for a model's output. However, so far empirical evaluations on LLM pretraining level only assumed that they work as intended. This is the first work to systematically evaluate whether the 'influential' examples actually have properties relevant to a language model output, based on re-training with controlled data splits. We investigate three prominent Hessian approximation methods (GradSim, TrackStar, K-FAC) in two settings (linguistic phenomena and factual knowledge) in GPT-2 type models (68M-5.6B training tokens). We find that on average, the `influential' examples are indeed relevant, but they do not always bear the target properties directly, and the ratio and kind of relevance varies by setting.
    
[^89]: 单一数据库奖励可被破解：用于可信赖文本到SQL强化学习的反事实执行

    Single-Database Rewards Are Hackable: Counterfactual Execution for Trustworthy Text-to-SQL RL

    [https://arxiv.org/abs/2611.34363](https://arxiv.org/abs/2611.34363)

    该论文提出一种反事实数据库奖励机制，通过在隐藏的代码生成数据库副本上执行查询来验证结果，将文本到SQL强化学习中的奖励破解率从54.1%大幅降至1.9%，且不误拒任何正确查询。

    

    面向文本到SQL智能体的强化学习（RL）通常在查询结果与单一数据库上的参考结果匹配时给予奖励。然而，当数据缺乏能够暴露查询缺陷的行时，错误的查询可能凭运气匹配成功。我们研究了一种反事实数据库（CF）奖励，它还在若干隐藏的、由代码生成的数据库副本上运行最终查询，仅在所有结果都匹配时（严格形式）或按匹配比例（平均形式）给予奖励。在1,302个合成审计问题上，严格形式将错误查询被全额支付的比例从29.5%降至16.8%，将奖励破解的比例从54.1%降至1.9%，同时拒绝了2,847个正确查询中的0个；来自“模型家族盲”生成器生成的副本表现大致相当。在一个3.35B模型上的受控GRPO对比实验（每组三个随机种子）中，使用平均形式训练并未提升域内准确率（8.4% vs 11.7%），因为两种奖励仅在5-7%的rollout上存在分歧；而在Spider上提升了1.6个点。（原文此处截断）

    arXiv:2611.34363v1 Announce Type: cross  Abstract: Reinforcement learning (RL) for text-to-SQL agents pays a query when its result matches the reference on one database. However, a wrong query can match by luck when the data lack the rows that expose its bug. We study a counterfactual-database (CF) reward that also runs the final query on a few hidden, code-generated copies of the database and pays only if all results match (strict form) or by the share matched (mean form). On 1,302 synthetic audit questions, the strict form lowers the share of buggy queries paid in full from 29.5% to 16.8% and of reward hacks from 54.1% to 1.9%, rejecting 0 of 2,847 correct queries; copies from a family-blind generator do about as well. In a controlled GRPO comparison on a 3.35B model with three seeds per arm, training on the mean form did not improve in-domain accuracy (8.4% vs 11.7%), because the two rewards disagreed on only 5-7% of rollouts; on Spider it gained 1.6 points. One seed reverses both s
    
[^90]: RAWPrism：一个三任务数据集与RAW域图像融合的统一模型

    RAWPrism: A Three-Task Dataset and a Unified Model for RAW-Domain Image Fusion

    [https://arxiv.org/abs/2611.34359](https://arxiv.org/abs/2611.34359)

    提出了覆盖多曝光融合、多焦点融合和红外-可见光融合三任务的RAWPrism数据集，并构建了首个直接在RAW域统一执行这三种融合任务的轻量级模型。

    

    多曝光融合（MEF）、多焦点融合（MFF）和红外-可见光图像融合（IVIF）的研究大多集中在渲染后的RGB图像上，而在图像信号处理（ISP）之前的RAW域中跨这些任务的统一融合仍然相对缺乏探索。为支持这一场景下的研究，我们引入了覆盖全部三个任务的RAWPrism数据集，以及一个轻量级的统一多任务融合网络RAWPrism模型。RAWPrism数据集包含来自我们RAW采集的422个可用MEF场景和150对配准的MFF图像对，以及1,500个源自FMB的IVIF样本，其中可见光RAW输入由ReRAW重建，其对应的RGB图像由CycleISP渲染。场景级划分以及任务特定的渲染与评分协议将这些资源与方法开发和评估相衔接。据我们所知，RAWPrism模型是首个直接在RAW域统一MEF、MFF和IVIF的模型。

    arXiv:2611.34359v1 Announce Type: new  Abstract: Research on multi-exposure fusion (MEF), multi-focus fusion (MFF), and infrared-visible image fusion (IVIF) has largely focused on rendered RGB images, while unified fusion across these tasks in the RAW domain before image signal processing (ISP) remains relatively underexplored. To support research in this setting, we introduce the RAWPrism dataset, covering all three tasks, and the RAWPrism model, a lightweight unified multi-task fusion network. The RAWPrism dataset comprises 422 usable MEF scenes and 150 registered MFF pairs from our RAW captures, together with 1,500 FMB-derived IVIF samples, in which the visible RAW inputs are reconstructed by ReRAW and their RGB counterparts rendered by CycleISP. Scene-level partitions and task-specific rendering and scoring protocols connect these resources to method development and evaluation. To our knowledge, the RAWPrism model is the first to unify MEF, MFF, and IVIF directly in the RAW domain.
    
[^91]: 面向人类决策的具有状态依赖前瞻性的逆强化学习

    Inverse Reinforcement Learning with State-Dependent Foresight for Human Decision-Making

    [https://arxiv.org/abs/2611.34351](https://arxiv.org/abs/2611.34351)

    该论文提出SC-AIRL（状态条件对抗逆强化学习），将人类前瞻性的有限性与状态依赖性这两种认知特性融入逆强化学习的规划器中，在行人过街和高速公路超车任务中显著提升了对人类行为的模仿效果。

    

    行为的计算建模通过拟合个体观察到的行为来对其进行表示，并通过模型参数来刻画其特征。自然情境任务融合了真实环境的复杂性，并产生高维的行为轨迹。逆强化学习（IRL）可以推断这些轨迹背后的潜在奖励函数。然而，标准IRL假设规划在效果上是无限视界的，而人类的前瞻性既是有界的，又是依赖于状态的。在此，我们提出了SC-AIRL（状态条件对抗逆强化学习），它将这两种认知特性构建到规划器之中。不同前瞻深度的策略共享同一个奖励函数，但在显式累加奖励的前瞻范围上有所不同；一个状态条件路由器则根据当前状态对这些策略进行混合。在行人过街和高速公路超车任务的留出测试片段上，SC-AIRL相比标准AIRL提升了模仿性能，将top-1准确率从0.549提高至……（摘要原文在此处截断）

    arXiv:2611.34351v1 Announce Type: cross  Abstract: Computational modeling of behavior represents individuals with a model fitted to their observed behavior and characterizes them through its parameters. Naturalistic tasks incorporate the complexity of real environments and yield high-dimensional behavioral trajectories. Inverse reinforcement learning (IRL) infers the latent reward function underlying these trajectories. Standard IRL, however, assumes effectively infinite-horizon planning, whereas human foresight is both bounded and state-dependent. Here we introduce SC-AIRL (State-Conditional Adversarial IRL), which builds these two cognitive properties into the planner. Per-depth policies share one reward but differ in how far ahead they sum it explicitly. A state-conditional router mixes them according to the current state. Across held-out episodes in pedestrian crossing and highway overtaking tasks, SC-AIRL improves imitation over Standard AIRL, raising top-1 accuracy from 0.549 to 
    
[^92]: 基于Jukebox嵌入的音乐相似度估计

    Music Similarity Estimation from Jukebox Embeddings

    [https://arxiv.org/abs/2611.34346](https://arxiv.org/abs/2611.34346)

    提出JukeSim方法，首次利用生成式音乐模型Jukebox的嵌入来估计音乐相似度，其性能与基于学习的方法相当，并优于最佳的手工设计方法。

    

    基于内容的音乐相似度估计旨在提供一种衡量音乐录音之间相似度的方法，而不依赖任何相关的元数据。这在冷启动音乐推荐等场景中具有潜在应用价值，因为此类场景中缺乏用户交互数据。在这项工作中，我们提出了JukeSim，一种基于生成式音乐模型Jukebox所创建的嵌入来估计音乐相似度的新方法。在多个数据集上的评估中，我们发现JukeSim的表现与现有的基于学习的方法相当，且两者均优于最佳的手工设计方法——这些手工设计方法此前从未与近期工作进行过比较。尽管JukeSim所需的计算量高于其他度量方法，但我们证明了利用生成式嵌入完成这一任务的可行性，这将鼓励进一步的研究。

    arXiv:2611.34346v1 Announce Type: cross  Abstract: Content-based music similarity estimation aims to provide a measure of similarity between music audio recordings, disregarding any associated metadata. This has potential applications in e.g. cold-start music recommendation scenarios, where there is no user interaction data. In this work, we propose JukeSim, a new approach to estimating musical similarity based on embeddings created by Jukebox, a generative music model. In evaluations on multiple datasets, we find that JukeSim performs on par with an existing learning-based approach, and both outperform the best hand-designed approaches, which had never been compared to recent work before. Even though the required compute for JukeSim is higher than for other measures, we demonstrate the viability of utilizing generative embeddings for this task, encouraging further research.
    
[^93]: CAMELS宇宙网分类中图神经网络的规模化与跨模拟泛化研究

    Scaling and Cross-Simulation Generalization of Graph Neural Networks for Cosmic-Web Classification in CAMELS

    [https://arxiv.org/abs/2611.34332](https://arxiv.org/abs/2611.34332)

    本研究系统评估了图神经网络在CAMELS多套流体动力学模拟中进行宇宙网环境分类时，随训练样本规模扩展的性能变化规律及其跨模拟泛化能力。

    

    基于图的方法可以根据星系所处的暗物质宇宙网环境对星系进行分类，但其超越开发模拟之外的可靠性仍不确定。星系形成模型会改变示踪天体的数量与空间分布，可能导致图特征发生偏移。我们测试了图神经网络如何随独立训练样本数的增加而扩展，以及如何在不同流体动力学模拟之间迁移。利用z=0的CAMELS IllustrisTNG、SIMBA和Astrid模拟套件，我们根据平滑后的冷暗物质密度场，将质量大于10^9太阳质量的星系标记为空洞、薄片、纤维或节点四类环境。我们将其空间分布表示为带有十个几何描述量的周期性Delaunay图。在按样本组划分的评估下，我们比较了一个多层感知机（MLP）和五种图神经网络（GNN），其中包括一个局部图Transformer。学习曲线将训练数据从10个独立样本逐步增加到100个，同时保持验证与测试盒固定不变。

    arXiv:2611.34332v1 Announce Type: cross  Abstract: Graph-based methods can classify galaxies by their dark-matter cosmic-web environments, but reliability beyond development simulations remains uncertain. Galaxy-formation models alter tracer populations and spatial distributions, potentially shifting graph features. We test how graph neural networks scale with independent training realizations and transfer across hydrodynamical simulations. Using the $z=0$ CAMELS IllustrisTNG, SIMBA and Astrid suites, we label galaxies with $M_\star>10^9\,M_\odot$ as void, sheet, filament or knot from the smoothed cold-dark-matter density field. We represent their spatial distributions as periodic Delaunay graphs with ten geometric descriptors. We compare a multilayer perceptron (MLP) and five graph neural networks (GNNs), including a local graph Transformer, under realization-grouped evaluation. Learning curves increase training data from 10 to 100 realizations, with validation and test boxes fixed. W
    
[^94]: Puzzling-Bench：基于23,777道人工编写、社区验证答案谜题的语言模型评估基准

    Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers

    [https://arxiv.org/abs/2611.34331](https://arxiv.org/abs/2611.34331)

    Puzzling-Bench是一个基于23,777道来自Puzzling Stack Exchange、带有社区验证答案的人工编写谜题（涵盖谜语、密码、逻辑推理等类别）的语言模型评估基准，并利用训练截止日期后发布的谜题切片来检测数据污染。

    

    我们推出了Puzzling-Bench，这是一个包含23,777道谜题的基准测试集，谜题选自Puzzling Stack Exchange存档（2014-2026年），均带有社区验证的答案：涵盖谜语、密码、逻辑网格、横向思维问题、文字游戏和趣味数学，其中三分之一配有图片。每道谜题都附有作者认可的答案、从中提取的简短参考答案（评估集的参考答案由LLM提取，并经50道谜题的人工审核验证）、作者本人的提示、社区标签和创建日期，因此该基准支持提示消融实验、类别分析以及滚动更新的训练截止日期后谜题切片（254道谜题发布于我们所评估的九个模型中六个模型的训练截止日期之后，其中包括表现最好的模型），可作为数据污染检查。我们使用两级评分器（先进行归一化精确匹配，再由LLM裁判评分）评估了九个前沿模型和开源模型，并通过来自两个模型系列的三个LLM裁判对裁判进行了验证：它们在95-97%的判定上达成一致……

    arXiv:2611.34331v1 Announce Type: cross  Abstract: We introduce Puzzling-Bench, a benchmark of 23,777 puzzles with community-verified answers drawn from the Puzzling Stack Exchange archive (2014-2026): riddles, ciphers, logic grids, lateral-thinking problems, wordplay and recreational mathematics, a third of them with images. Every puzzle carries the answer accepted by its author, a short reference answer taken from it (LLM-extracted for the evaluation sets and checked on a 50-puzzle human review), the author's own hints, community tags and a creation date, so that the benchmark supports hint ablations, category analysis and a rolling post-cutoff slice (254 puzzles posted after the training cutoff of six of the nine models we evaluate, including the best one) as a contamination check. We evaluate nine frontier and open models with a two-tier scorer (normalized exact match, then an LLM judge) and validate the judge with three judges from two model families: they agree on 95-97% of verdi
    
[^95]: 变分原理与机器学习：数值分析与偏微分方程的视角

    Variational Principles and Machine Learning: a Numerical Analysis and PDE perspective

    [https://arxiv.org/abs/2611.34314](https://arxiv.org/abs/2611.34314)

    本文从数值分析与偏微分方程的变分视角出发，将机器学习算法视为非线性最小化问题加以研究，探讨稳定性、一致性、可逼近性和紧性等经典数值分析概念在机器学习中的新内涵。

    

    变分原理在现代数学、物理学和科学计算的发展中发挥了基础性作用。从力学中的最小作用量原理，到变分法以及偏微分方程的变分形式，它们为理解复杂问题和设计可靠的数值方法提供了强大的框架。近年来，机器学习引入了一种新的计算范式，其中高表达性的非线性函数空间和大规模最小化问题扮演着核心角色。在这些讲义中，我们探讨了这两种框架之间的联系。我们从变分和数值分析的角度审视机器学习算法，特别关注其作为非线性最小化问题的表述形式。我们研究了稳定性、一致性、可逼近性和紧性等经典概念如何在机器学习中获得新的诠释。

    arXiv:2611.34314v1 Announce Type: cross  Abstract: Variational principles have played a fundamental role in the development of modern mathematics, physics, and scientific computing. From the principles of least action in mechanics to the calculus of variations and the variational formulation of partial differential equations, they have provided a powerful framework for understanding complex problems and designing reliable numerical methods. In recent years, Machine Learning has introduced a new computational paradigm, in which highly expressive nonlinear function spaces and large-scale minimisation problems play a central role. In these notes, we explore the connection between these two frameworks. We develop a variational and numerical-analysis perspective on Machine Learning algorithms, focusing in particular on their formulation as nonlinear minimisation problems. We investigate how classical concepts such as stability, consistency, approximability, and compactness acquire new inter
    
[^96]: 基于全场位移的三维非均质与梯度超弹性材料本构参数场的可解释辨识

    Interpretable identification of constitutive parameter field for three-dimensional heterogeneous and graded hyperelastic materials from full-field displacements

    [https://arxiv.org/abs/2611.34310](https://arxiv.org/abs/2611.34310)

    本文提出了一种可解释的方法，仅利用全场位移和边界反力即可辨识三维非均质与梯度超弹性材料的空间变化本构参数场，为功能梯度材料设计及生物组织表征提供了有效工具。

    

    仅利用实验可获得的位移场和边界反力来辨识非均质固体空间变化材料特性的反问题仍然具有挑战性。能够表征非均质和梯度固体将有助于功能梯度材料和复合材料的设计与制造。此外，一种从体内测量中高效辨识非均质材料特性的技术在生物医学领域中表征非均质生物组织也将十分有用。尽管过去二十年间开发的多种技术已经解决了均质材料（包括线弹性、超弹性、粘弹性和弹塑性固体）的反问题辨识，但相对而言，针对非均质和梯度材料的研究较少。此外，近期几种表征非均质和梯度材料的方法……

    arXiv:2611.34310v1 Announce Type: new  Abstract: The inverse problem of identifying spatially varying material properties of heterogeneous solids using experimentally available displacement fields and boundary reaction forces alone remains challenging. Being able to characterize heterogeneous and graded solids would be advantageous in the design and manufacturing of functionally graded materials and composites. Additionally, an efficient technique for identifying heterogeneous material properties from in vivo measurements would be useful in biomedical contexts to characterize heterogeneous biological tissues. Although several techniques developed over the past two decades have addressed inverse identification for homogeneous materials, including linear elastic, hyperelastic, viscoelastic, and elastoplastic solids, comparatively fewer studies have considered heterogeneous and graded materials. Furthermore, several recent approaches to characterizing heterogeneous and graded materials em
    
[^97]: RAP：面向双曲表示的径向-角向随机投影

    RAP: Radial-Angular Random Projection for Hyperbolic Representations

    [https://arxiv.org/abs/2611.34298](https://arxiv.org/abs/2611.34298)

    本文提出径向-角向随机投影（RAP），通过精确保持径向坐标并单独处理角向误差，首次为双曲表示构建了类似 Johnson–Lindenstrauss 引理的盲式随机降维原语，填补了双曲表示学习中缺乏维度压缩工具的空白。

    

    随机投影是欧几里得表示学习中的一种基础压缩原语，Johnson–Lindenstrauss 引理给出了与环境维度无关的目标维度 $k=O(\varepsilon^{-2}\log n)$。随着双曲表示在现代 Transformer、视觉-语言模型和大语言模型中日益受到关注，对这种“盲式”降维的需求也随之出现，但目前尚不存在针对双曲表示的类似原语。我们研究了这一空白。两种自然的尝试以相反的方式失败：切空间方案提供了干净的局部保证，但受制于局部坐标系且需付出曲率代价；而双曲面模型中的完全双曲投影虽然在几何上是原生的，却在将径向误差与角向误差送入 $\operatorname{arcosh}$ 的条件数之前将二者混合在一起。我们提出径向-角向随机投影（RAP），它精确地保持径向坐标，并对角向部分应用欧氏（原文摘要在此处截断）……

    arXiv:2611.34298v1 Announce Type: cross  Abstract: Random projection is a basic compression primitive for Euclidean representation learning, with the Johnson--Lindenstrauss lemma giving a target dimension $k=O(\varepsilon^{-2}\log n)$ independent of the ambient dimension. As hyperbolic representations gain increasing attention in modern Transformers, vision-language models, and large language models, the same need for oblivious dimensionality reduction arises, but no analogous primitive yet exists for hyperbolic representations. We study this gap. Two natural attempts fail in dual ways: a tangent-space scheme gives a clean local guarantee but is chart-bound and pays a curvature penalty, while a fully hyperbolic projection in the hyperboloid model is geometrically native but mixes radial and angular error before feeding them into the conditioning of $\operatorname{arcosh}$. We propose radial-angular random projection (RAP), which preserves the radial coordinate exactly and applies a Euc
    
[^98]: LatENSO v1.0：一个用于季节性ENSO算子与预报的热带太平洋学习潜在状态

    LatENSO v1.0: a learned latent state of the tropical Pacific for seasonal ENSO operators and forecasting

    [https://arxiv.org/abs/2611.34296](https://arxiv.org/abs/2611.34296)

    该论文提出LatENSO v1.0系统，利用Swin-transformer自编码器将热带太平洋13个月平均海气场压缩为一个32维的学习潜在状态，并基于该状态构建XRO-lat和Floquet模态算子两种季节算子，实现ENSO的季节预报。

    

    我们在此提出LatENSO v1.0，这是一个围绕学习得到的32维热带太平洋状态构建的厄尔尼诺-南方涛动（ENSO）机器学习预报系统。一个Swin-transformer自编码器将十三个月平均的海洋和大气场压缩为32个数，其训练目标仅为场的重构，不包含任何预报目标。这32个数对场的重构能力可与约百（O(100)）个经验正交函数相当。两个季节算子使用这一状态。第一个是扩展非线性充电振荡器（XRO），将学习状态的两个方向纳入其变量之中，我们称之为XRO-lat。第二个是由三个阻尼振荡器构成的Floquet模态算子（FMO），它预报一个修正量，并将该修正量叠加到由四个观测预测因子构建的最小二乘基准预报之上。我们将这两个算子与充电振荡器、XRO以及深度回声状态网络（DESN）进行对比评估，所有模型均在相同的数据、交叉验证折和（……）上重新拟合

    arXiv:2611.34296v1 Announce Type: cross  Abstract: Here we present LatENSO v1.0, a machine-learning forecast system for the El Ni\~no-Southern Oscillation (ENSO) built around a learned, 32-dimensional state of the tropical Pacific. A Swin-transformer autoencoder compresses thirteen monthly-averaged ocean and atmosphere fields into 32 numbers, trained to reconstruct the fields, with no forecast target. The 32 numbers rebuild the fields as well as O(100) empirical orthogonal functions. Two seasonal operators use this state. The first is the extended nonlinear recharge oscillator (XRO) with two directions of the learned state among its variables; we call it XRO-lat. The second is a Floquet modal operator (FMO) of three damped oscillators, which forecasts a correction added to a least-squares base forecast made from four observed predictors. We evaluate both against the recharge oscillator, XRO and a deep echo-state network (DESN), all refitted on identical data, cross-validation folds and
    
[^99]: 《Beat This! Live——通过预测未来实现的在线节拍与强拍跟踪》

    Beat This! Live - Online Beat and Downbeat Tracking by Predicting the Future

    [https://arxiv.org/abs/2611.34277](https://arxiv.org/abs/2611.34277)

    该论文提出了 Beat This! Live——一个通过时间偏移标签训练、能够略微超前预测未来帧的因果模型，实现了低延迟的在线节拍与强拍跟踪，并在 GTZAN 数据集上显著超越现有最先进的在线方法。

    

    节拍跟踪与强拍跟踪近年来通过深度学习方法取得了多项进展，但大多数方法针对的是离线设置，因而无法应用于实时场景。我们将近期的一个离线节拍跟踪模型 Beat This 改造为在线、低延迟运行的版本，并提出了 Beat This! Live——一个专为实时节拍与强拍跟踪设计的因果变体。我们介绍了为获得适合流式推理的因果系统所需的架构修改，并测量了这些修改对准确率的影响。通过使用时间偏移的标签来训练模型，它能够产生略微领先于当前帧的预测，这不仅降低了延迟，还使峰值提取更加有效。在 GTZAN 数据集上的评估表明，与当前在线节拍跟踪领域最先进的 BeatNet+ 相比，节拍跟踪的 F1 分数从 80.6 提升至 86.7，强拍跟踪的 F1 分数从 56.5 提升至 74.1。

    arXiv:2611.34277v1 Announce Type: cross  Abstract: Beat and downbeat tracking have seen multiple improvements through recent deep learning approaches, but most target an offline setting, precluding their use in real-time scenarios. We adapt a recent offline beat tracking model, Beat This, for online, low-latency operation and propose Beat This! Live, a causal variant designed for real-time beat and downbeat tracking. We present the architectural modifications required to obtain a causal system suitable for streaming inference and measure their impact on accuracy. By training our model with temporally shifted labels, it produces predictions slightly ahead of the current frame, which not only reduces latency, but enables much more effective peak-picking. Evaluation on the GTZAN dataset shows the F1-score improves from 80.6 to 86.7 for beats and 56.5 to 74.1 for downbeats compared to BeatNet+, the current state of the art in online beat tracking.
    
[^100]: 匹配预算下循环MoE语言模型的扩展：信息流与参数分配

    Scaling Looped MoE Language Models under Matched Budgets: Information Flow and Parameter Allocation

    [https://arxiv.org/abs/2611.34276](https://arxiv.org/abs/2611.34276)

    该论文通过门控内循环状态更新与跨阶段残差连接改进循环MoE模型的信息流，并在等参数、等FLOP预算下发现将单次Core深度设为总展开深度的25%-35%、同时把节省的参数重新分配给每层额外路由专家的配置，能够以零额外计算成本提升模型性能。

    

    循环Transformer将计算深度与参数量解耦，先前的研究已经证明通过增加计算可以获得可观的性能提升。然而，在严格的等参数和等FLOP约束下配置这些模型的系统性设计原则仍然稀缺。我们通过改进循环模型中的信息流并系统研究循环参数配置来为这一方向做出贡献。具体而言，我们通过门控内循环状态更新和跨阶段残差连接来优化信息流。基于这一设计，受控研究确定了一种配置方案：将单次传递的Core深度设为总展开深度的25%-35%，并将节省下来的参数重新分配为每层额外的路由专家，从而在零额外计算开销的情况下提升性能，同时保持与现有训练基础设施的兼容性。在1T预训练token后的评估显示，我们的69B MLA和51B KDA Mix（摘要原文在此处截断）

    arXiv:2611.34276v1 Announce Type: new  Abstract: Looped Transformers decouple computational depth from parameter count, and prior studies have demonstrated promising performance gains with increased computation. However, systematic design principles for configuring these models under strict iso-parameter and iso-FLOP constraints remain scarce. We contribute to this direction by improving information flow in looped models and systematically studying loop parameter configurations. Specifically, we refine this information flow via gated inner-loop state updates and a cross-stage residual connection. Building on this design, controlled studies identify a configuration combining a single-pass Core depth of 25%-35% of the total unrolled depth with reallocating saved parameters into extra routed experts per layer, improving performance at zero extra compute while preserving compatibility with existing training infrastructure. Evaluated after 1T pre-training tokens, our 69B MLA and 51B KDA Mix
    
[^101]: LatentCP-DPO：具有有限样本保证的基于保形悲观主义的直接偏好优化方法

    LatentCP-DPO: Conformal Pessimism for DPO with Finite-Sample Guarantees

    [https://arxiv.org/abs/2611.34275](https://arxiv.org/abs/2611.34275)

    该论文提出LatentCP-DPO框架，将保形预测与悲观偏好优化相结合，为潜在相对奖励构建具有有限样本覆盖保证的不确定性集合，并通过最大化最坏情况下的偏好对数似然来学习响应策略，从而有效防止DPO在数据有限时对噪声偏好标签过拟合。

    

    直接偏好优化（DPO）从人类偏好数据中学习响应策略，但在数据集有限的情况下，它可能会对噪声较大或信息较弱的偏好标签产生过拟合。我们提出了LatentCP-DPO，这是一个将保形预测与悲观偏好优化相结合的框架，适用于Bradley-Terry（BT）成对反馈和Plackett-Luce（PL）排序反馈。基于LatentCP对未观测模型参数构建不确定性集合的方法，我们研究了如何将这种不确定性纳入偏好优化，并刻画了其对策略性能的影响。在可交换性和偏好模型正确设定的条件下，我们从可观测偏好的保形预测集合出发，为潜在相对奖励构建了具有有限样本边际覆盖保证的不确定性集合。随后，我们通过在这些不确定性集合上最大化最坏情况下的期望偏好对数似然来学习响应策略。

    arXiv:2611.34275v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) learns a response policy from human preference data, but with a limited dataset it can overfit to noisy or weakly informative preference labels. We introduce LatentCP-DPO, a framework that combines conformal prediction with pessimistic preference optimization for Bradley-Terry (BT) pairwise feedback and Plackett-Luce (PL) ranking feedback. Building on LatentCP's construction of uncertainty sets for unobserved model parameters, we address how to incorporate this uncertainty into preference optimization and characterize its effect on policy performance. Under exchangeability and a correctly specified preference model, we construct uncertainty sets for latent relative rewards with finite-sample marginal coverage from conformal prediction sets for observable preferences. We then learn a response policy by maximizing the worst-case expected preference log-likelihood over these uncertainty sets, using an 
    
[^102]: Libtt：面向 Tenstorrent 处理器的编译器与运行时

    Libtt: A Compiler and Runtime for Tenstorrent Processors

    [https://arxiv.org/abs/2611.34273](https://arxiv.org/abs/2611.34273)

    Libtt 通过实现 PJRT 接口将 PyTorch 和 JAX 生态系统完整移植到 Tenstorrent 硬件，以单个共享库提供从编译器到多芯片运行时的全栈方案，并借助大量修复和编译器优化使现有面向 TPU 的代码开箱即用且高效运行。

    

    Libtt 将 PyTorch 和 JAX 生态系统引入 Tenstorrent 硬件。它实现了 PJRT——XLA 框架驱动谷歌 TPU 所使用的接口——因此 JAX 可以直接在其上运行，PyTorch 通过 TorchTPU 运行，为 TPU 编写的软件只需替换自定义内核即可无缝迁移。Libtt 基于 Tenstorrent 的开源编译器和运行时构建，将从编译器到多芯片运行时的整个技术栈打包为单个共享库发布。它对上游 Tenstorrent 软件进行了大量修复，使现有的 PyTorch 和 JAX 代码能够开箱即用，并加入了大量新的编译器优化以提升运行速度。它通过了大部分 JAX 和 TorchTPU 测试套件，并能运行 SGLang、TorchTitan 和 Miles，覆盖推理服务、微调和强化学习等应用场景。

    arXiv:2611.34273v1 Announce Type: cross  Abstract: Libtt brings the PyTorch and JAX ecosystem to Tenstorrent hardware. It implements PJRT, the interface through which XLA-based frameworks drive Google's TPUs, so JAX runs on it directly, PyTorch runs through TorchTPU, and software written for TPUs carries over with no changes except replacing custom kernels. Built on Tenstorrent's open-source compiler and runtime, libtt ships the entire stack, from the compiler to the multi-chip runtime, as a single shared library. It contains extensive fixes to the upstream Tenstorrent software to make existing PyTorch and JAX code work out of the box, and extensive new compiler optimizations to make it fast. It passes most of the JAX and TorchTPU test suites and runs SGLang, TorchTitan, and Miles, covering serving, fine-tuning, and reinforcement learning.
    
[^103]: 数控加工中基于多证据推理的大语言模型耦合可解释刀具磨损监测

    LLM-coupled interpretable tool wear monitoring with multi-evidence reasoning in CNC machining

    [https://arxiv.org/abs/2611.34270](https://arxiv.org/abs/2611.34270)

    该论文提出了一种两层框架，将深度学习刀具磨损分类与大语言模型多证据推理相结合，在实现高精度磨损状态识别的同时，生成可解释的自然语言报告和刀具更换建议。

    

    数控加工中可解释的刀具磨损评估具有挑战性。现有的诊断模型虽然准确率高，但对刀具更换决策的指导有限。本研究提出了一种结合深度学习磨损分类与大语言模型辅助推理的两层框架。第一层使用PCA Loading特征选择和优化后的GRU网络识别三种磨损状态，在PHM2010数据集上实现了98.38%的宏观F1分数。第二层利用大语言模型根据分类结果和特征趋势生成自然语言报告和刀具更换建议。工程师评估表明，这些报告优于规则模板。在NASA铣削数据上的跨数据集测试验证了该框架的泛化能力。

    arXiv:2611.34270v1 Announce Type: cross  Abstract: Interpretable tool wear assessment in CNC machining is challenging. Existing diagnostic models have high accuracy but provide limited guidance for tool replacement. This study proposes a two layer framework combining deep learning wear classification and LLM assisted reasoning. The first layer uses PCA Loading feature selection and optimized GRU to recognize three wear states, achieving 98.38 percent macro F1 score on the PHM2010 dataset. The second layer uses LLMs to generate natural language reports and replacement suggestions from classification results and feature trends. Engineer evaluation shows the reports outperform rule templates. Cross dataset tests on NASA milling data support the generalizability of this framework.
    
[^104]: 面向合成临床轨迹的仅训练集约束语言模型：防泄漏校准与可审计动力学

    Train-Only-Constrained LM for Synthetic Clinical Trajectories: Leak-Safe Calibration and Auditable Dynamics

    [https://arxiv.org/abs/2611.34258](https://arxiv.org/abs/2611.34258)

    该论文提出一种仅用训练集校准的约束式语言模型流水线，通过分位数标记学习共现关系并用可审计的马尔可夫转移矩阵约束时间动态，在防泄漏（不使用测试集统计量）的前提下生成了分布保真度更高、隐私可检测性接近随机的合成临床轨迹。

    

    模拟临床时间序列需要时间真实性、保持模型完整性的校准以及隐私透明度。我们提出了一种仅使用训练集的、受约束的语言模型流水线，对生成过程进行因式分解。一个紧凑的GPT风格语言模型通过分位数标记学习跨生物标志物的共现关系，而时间方向则由仅在训练集（TRAIN）上拟合的每个生物标志物3x3马尔可夫转移矩阵来强制执行，从而使动力学可被审计。我们仅在训练集上拟合单调分位数映射（QMap）（SYN(TRAIN) -> SIM(TRAIN)）；不使用任何测试集（TEST）统计量，随后将该固定映射应用于SYN(TEST)的严重程度。在一个5,000名患者的模拟测试平台（参数均处于已发表的临床范围内）中，与经验采样器相比，该受约束语言模型展现出更高的分布保真度和共现准确性，在小型lambda扫描后保持接近随机的可检测性，并通过构造保证了时间一致性。基于分类器的成员推断……（原文摘要至此截断）

    arXiv:2611.34258v1 Announce Type: new  Abstract: Simulating clinical time series necessitates temporal realism, calibration that preserves model integrity, and privacy transparency. We present a train-only, constrained language-model pipeline that factorizes generation. A compact GPT-style language model learns cross-biomarker co-occurrence through quantile tokens, and temporal directions are enforced by 3x3 per-biomarker Markov transitions fit on TRAIN, allowing dynamics to be audited. We fit a monotone quantile map (QMap) on TRAIN only (SYN(TRAIN) -> SIM(TRAIN)); no TEST statistics are used. The fixed map is then applied to SYN(TEST) severities. The constrained LM demonstrated improved distributional fidelity and co-occurrence in a 5,000-patient simulated testbed (parameters within published clinical ranges) compared to an empirical sampler, maintained near-chance detectability after a small lambda-sweep, and exhibits temporal adherence by construction. A classifier-based membership-
    
[^105]: EXPOSURE：解释强化学习中不确定性的来源

    EXPOSURE: Explaining Sources of Uncertainty in Reinforcement Learning

    [https://arxiv.org/abs/2611.34248](https://arxiv.org/abs/2611.34248)

    提出EXPOSURE方法，利用单个感知掩码的信念网络同时进行动作选择和Shapley解释，将回报不确定性分解为特征缺失导致的观测部分与环境随机性导致的固有部分，从而更准确地解释强化学习智能体的期望回报、不确定性来源及信念不确定性。

    

    强化学习中的Shapley值被用来解释哪些观测特征驱动了智能体的行为和期望回报，但无法解释其确定性。解释确定性需要理解观测特征如何影响回报不确定性的不同来源，而计算Shapley值需要智能体对每个可见特征子集的预测。我们提出EXPOSURE方法，它使用一个感知掩码的信念网络同时完成动作选择和Shapley解释。该方法解释回报的方差——将其分解为由于特征缺失导致的观测部分和由于环境自身随机性导致的固有部分——以及智能体对被掩码特征的信念熵。在井字棋和扫雷这两个可精确计算Shapley值的环境中，EXPOSURE对期望回报及其不确定性给出了更准确的解释，并且是唯一能解释信念不确定性的方法。

    arXiv:2611.34248v1 Announce Type: new  Abstract: Shapley values in reinforcement learning are used to explain which observation features drive an agent's behavior and expected return, but not its certainty. Explaining certainty requires understanding how observation features influence different sources of return uncertainty, and Shapley values need the agent's prediction for every subset of visible features. We propose EXPOSURE, which uses one mask-aware belief network for both action selection and Shapley explanations. It explains the variance of the return, split into an observational part due to missing features and an inherent part due to the environment's own randomness, and the entropy of its belief over the masked features. In tic-tac-toe and Minesweeper, where exact Shapley values are computable, EXPOSURE gives more accurate explanations of expected return and its uncertainty, while uniquely explaining belief uncertainty.
    
[^106]: 基于生成模型的变点检测

    Change Point Detection with Generative Models

    [https://arxiv.org/abs/2611.34228](https://arxiv.org/abs/2611.34228)

    该论文提出利用生成模型对时间序列窗口进行数据增广，突破了基于窗口的变点检测中“样本量瓶颈”的根本局限，显著提升了变点检测的准确性与质量。

    

    变点检测（CPD）方法旨在识别时间序列属性中的突变，这是众多领域中的一个关键问题。一大类变点检测方法依赖于计算变点检测分数，即两个相邻样本（称为窗口）之间的差异或散度。基于窗口的变点检测方法存在一个根本性局限，即“样本量瓶颈”。为了确保时间精度，窗口必须很小；然而，小样本无法为目前用于计算变点检测分数的分类器和深度学习模型提供充分的统计证据，导致分数方差过大和频繁的误预测。我们的研究通过探索生成模型进行窗口增广来解决这一问题，有效地扩展了可用数据以提高变点检测质量。在合成数据和真实数据上的实验表明，所提出的方法显著提升了变点检测性能。

    arXiv:2611.34228v1 Announce Type: new  Abstract: Change point detection (CPD) methods aim to identify abrupt changes in the properties of time series, which is a key problem in numerous fields. A major class of CPD approaches relies on computing a CPD score, which is either a discrepancy or divergence between two subsequent samples, called windows. A fundamental limitation of window-based CPD is the 'sample size bottleneck.' To ensure temporal precision, windows must be small; however, small samples provide insufficient statistical evidence for the classifiers and deep-learning models currently used to compute CPD scores. This leads to high variance in the scores and frequent mispredictions. Our study tackles this issue by exploring generative models for window augmentation, effectively expanding available data to improve CPD quality. Experiments on synthetic and real data demonstrate that the proposed approach significantly enhances CPD. Specifically, we show that this approach allows
    
[^107]: 苏格拉底式暂停：在AI速度下守护科学判断

    The Socratic PAUSE: safeguarding scientific judgement at AI speed

    [https://arxiv.org/abs/2611.34222](https://arxiv.org/abs/2611.34222)

    论文提出“苏格拉底式暂停”机制，在关键科学决策节点设置审慎检查点并保持AI贡献的暂定性，以应对AI加速发现快于论证所导致的验证债务累积与信心超越证据的风险。

    

    生成式AI可以在几分钟内将一个初步的科学问题转化为假设、方法、代码和解释。然而，生成一项贡献并不意味着它值得科学上的信赖：研究人员仍必须审视其证据、假设与影响，并对依赖该贡献的决定保持问责。以爱因斯坦的科学发现图景为视角，我们论证了AI对“发现”的加速远超过对“论证”的加速。因此，在人机研究合作中，信心可能通过三种机制超越证据：递归强化、来源信息丢失，以及对AI生成贡献的失准依赖。当这些贡献未经充分审查就被纳入后续的论断或决策时，验证债务便会不断累积。据此，我们提出“苏格拉底式暂停”，作为在关键决策节点上设置的一种审慎检查点，使AI贡献在其中保持暂定状态。

    arXiv:2611.34222v1 Announce Type: cross  Abstract: Generative AI can turn a tentative scientific question into hypotheses, methods, code and interpretations within minutes. Yet generating a contribution does not establish that it deserves scientific reliance: researchers must still examine its evidence, assumptions and implications, and remain accountable for the decision to rely on it. Using Einstein's sketch of scientific discovery as a lens, we argue that AI has accelerated discovery far more than justification. In human-AI research partnerships, confidence can therefore outgrow evidence through three mechanisms: recursive reinforcement, loss of provenance and miscalibrated reliance on AI-generated contributions. Verification debt accumulates when such contributions are carried into later claims or decisions without adequate scrutiny. Accordingly, we propose the Socratic Pause as a deliberate checkpoint at consequential decision gates, where an AI contribution remains provisional wh
    
[^108]: 开放多智能体系统中的计算感知次模在线学习

    Compute-Aware Submodular Online Learning in Open Multi-Agent Systems

    [https://arxiv.org/abs/2611.34211](https://arxiv.org/abs/2611.34211)

    本文提出了一种带查询预算的计算感知次模在线学习算法，通过采样和逆概率加权获得无偏梯度估计，并推导出可分解为KL散度跟踪误差与查询相关反馈项的1/2近似遗憾界。

    

    本文研究开放多智能体系统中协作任务分配的策略学习问题，其中次模效用函数刻画了智能体贡献重叠所导致的边际收益递减现象。尽管标准方法会评估所有候选动作以更新策略，但这在线上场景中可能带来过高的计算开销。因此，我们对这些评估施加预算约束，并从理论层面分析查询预算对在线学习的影响。所提出的算法按预设数量采样智能体-动作对，并对其边际收益进行逆概率加权，从而获得期望阶段效用的无偏梯度估计，用于Kullback-Leibler（KL）镜像上升更新。我们推导出1/2近似遗憾界，该界优雅地分解为最优策略与学习策略之间以KL散度衡量的跟踪误差，以及一个与查询次数相关的反馈项。在协作信息覆盖任务上的实验表明（原文摘要在此处截断）。

    arXiv:2611.34211v1 Announce Type: new  Abstract: This paper studies policy learning for cooperative task allocation in open multi-agent systems, where submodular utilities model diminishing marginal returns due to overlapping agent contributions. While standard methods evaluate every candidate action to update the policy, this may incur an excessive computational cost online. Hence, we impose a budget on these evaluations and analytically study the effect of query budget on online learning. Our algorithm samples a prescribed number of agent-action pairs and applies inverse-probability weighting to their marginal gains, yielding an unbiased gradient estimate of the expected stage utility for a Kullback-Leibler (KL) mirror-ascent update. We derive a $1/2$-approximate regret bound that elegantly splits into a tracking error between optimal and learned policies, quantified by their KL divergence, and a query-dependent feedback term. Experiments on cooperative information coverage show that
    
[^109]: 面向多模态预测的文本条件化时间序列基础模型低秩适应方法

    Text-Conditioned Low-Rank Adaptation of Time-Series Foundation Models for Multimodal Forecasting

    [https://arxiv.org/abs/2611.34206](https://arxiv.org/abs/2611.34206)

    提出C2LoRA-TS方法，通过将文本上下文编码为冻结时间序列基础模型的临时LoRA参数更新，实现文本条件化的参数适应，在九个领域的多模态预测任务中均优于冻结模型和静态LoRA基线。

    

    时间序列基础模型（TSFMs）提供了可迁移的数值预测先验。然而，与预测相关的信息可能通过文本传达，而并未体现在观测序列中。许多多模态预测方法通过模型输入或中间表示来整合此类上下文信息。我们研究了一种互补机制：文本条件化参数适应。我们提出了面向时间序列的 Context-to-LoRA 方法（C2LoRA-TS），该方法对文本进行编码，并为冻结的时间序列基础模型中选定的权重矩阵生成临时性的低秩适应（LoRA）更新。数值主干网络和文本编码器均保持冻结，且当上下文信息缺失时，该方法能精确恢复冻结 TSFM 的原始输出。在 Time-MMD 数据集上跨九个领域和三种 TSFM 架构的评估中，C2LoRA-TS 在所有架构和随机种子下均比冻结 TSFM 和静态 LoRA 基线取得了更低的总体预测误差。（摘要在此处被截断）

    arXiv:2611.34206v1 Announce Type: new  Abstract: Time-series foundation models (TSFMs) provide transferable numerical forecasting priors. However, forecast-relevant information may be conveyed in text without being reflected in the observed series. Many multimodal forecasting methods integrate such context through model inputs or intermediate representations. We investigate a complementary mechanism: text-conditioned parameter adaptation. We propose Context-to-LoRA for Time Series (C2LoRA-TS), which encodes text and generates temporary Low-Rank Adaptation (LoRA) updates for selected weight matrices of a frozen TSFM. Both the numerical backbone and text encoder remain frozen, and the method exactly recovers the frozen TSFM output when context is absent. Evaluated on Time-MMD across nine domains and three TSFM architectures, C2LoRA-TS achieves lower aggregate forecasting error than the frozen-TSFM and Static LoRA baselines for every architecture and seed. Replacing the instance-specific 
    
[^110]: 从bulk数据预测细胞类型特异性表达虽困难但仍然有用

    Cell-Type-Specific Expression Is Hard to Predict from Bulk, but It Is Still Useful

    [https://arxiv.org/abs/2611.34166](https://arxiv.org/abs/2611.34166)

    该论文通过推导闭式精度界从理论上解释了为何从bulk RNA数据预测细胞类型比例容易而预测细胞类型特异性表达困难，并证明尽管CTSE难以估计，对其进行目标性反卷积在非小细胞肺癌总生存期预测中仍具有预后价值。

    

    基于单细胞信息的反卷积方法能够为临床bulk RNA队列提供细胞类型分辨的特征。现有大多数方法预测K个组成细胞类型（CTs）的比例 π_k。较少的方法预测细胞类型 k 中基因 g 的细胞类型特异性表达（CTSE）μ_kg，它能捕捉比例所遗漏的细胞状态，但近期的基准测试发现这一任务要困难得多。我们用闭式精度界解释了这一差距。对于CTSE，我们推导了逐基因线性估计器的R²上限；对于比例，跨基因汇集证据的最优线性估计器具有闭式R²，且其可计算的界可预测哪些细胞类型是可学习的。我们研究了106个经生物学整理的（细胞类型，基因）对的目标反卷积，这些对在事后被发现具有更高的可学习性评分。在非小细胞肺癌的总生存期分析中，估计难度并不能预测这些目标位点的预后价值。然而，稀疏生存模型……

    arXiv:2611.34166v1 Announce Type: new  Abstract: Single-cell-informed deconvolution provides cell-type-resolved features for clinical bulk RNA cohorts. Most existing methods predict the proportions $\pi_k$ of the $K$ constituent cell types (CTs). Fewer predict cell-type-specific expression (CTSE) $\mu_{kg}$ of gene $g$ in CT $k$, which can capture cell states that proportions miss, but recent benchmarks find it much harder. We explain this gap with closed-form accuracy bounds. For CTSE, we derive an $R^2$ ceiling for per-gene linear estimators. For proportions, the best linear estimator, which pools evidence across genes, has a closed-form $R^2$, and a computable lower bound on it predicts which CTs are learnable. We study targeted deconvolution of 106 biologically curated (CT, gene) pairs, found post hoc to have higher learnability scores. On overall survival in non-small cell lung cancer, estimation difficulty does not predict prognostic value across these targets. Yet a sparse survi
    
[^111]: 对抗性多智能体环境中面向作战目标的策略性与鲁棒任务分配

    Strategic and Robust Task Allocation for Combat Objectives in Adversarial Multi-Agent Environments

    [https://arxiv.org/abs/2611.34158](https://arxiv.org/abs/2611.34158)

    本文提出一种“先预测后优化”的混合任务分配方法，通过多头模型预测战术信号并结合最小费用流求解器，在对抗性多智能体环境中实现实时、可控、可解释且鲁棒的作战目标分配。

    

    在争夺目标点的团队博弈中，任务分配需要在不确定性、空间约束和不断变化的战术条件下进行协调决策。在这类场景中，多个智能体必须同时预判队友的行为和对手的行动。我们研究了一种“先预测后优化”的混合方法，用于实时地将可控智能体分配到各个目标。一个带掩码的多头模型为每个智能体预测战术信号，包括生存概率、局部优势、战斗交互、行进时间和目标贡献度。这些信号与基于规则的项组合成评分矩阵，并由最小费用流求解器在容量、占据状态、稳定性和防御策略等约束下计算协调的全局分配方案。学习仅限于战术评分部分，而显式规则和求解器结构保证了系统的可控性与可解释性。我们使用预测指标、离线分配诊断等方法对系统进行评估。

    arXiv:2611.34158v1 Announce Type: cross  Abstract: Task allocation in contested-objective team games requires coordinated decisions under uncertainty, spatial constraints, and changing tactical conditions. In such settings, multiple agents must anticipate both teammate behavior and opponent actions. We study a hybrid predict-then-optimize approach for assigning controllable agents to objectives in real time. A masked multi-head model predicts tactical signals for each agent, including survival chance, local superiority, combat interaction, travel time, and objective contribution. These signals are combined with rule-based terms in a score matrix, and a minimum-cost flow solver computes coordinated global assignments constrained by capacity, occupancy, stability, and defensive strategies. Learning is limited to tactical scoring, while explicit rules and solver structure preserve controllability and interpretability. We evaluate the system using predictive metrics, offline assignment dia
    
[^112]: 当高容量变得不可部署：直接空气捕集吸附剂多目标优化中的孔隙填充约束

    When High Capacity Becomes Undeployable: Pore-Filling Constraints for Multi-Objective Optimization of Direct Air Capture Sorbents

    [https://arxiv.org/abs/2611.34154](https://arxiv.org/abs/2611.34154)

    该论文创新性地将孔隙填充边界作为显式结果约束引入DAC吸附剂的多目标贝叶斯优化，并基于618个可追溯配方构建基准，揭示无约束优化会系统性提出物理上不可部署的高容量配方，而显式约束能将搜索引导至更低胺负载、更大孔隙的可部署设计。

    

    名义CO₂容量最高的胺浸渍吸附剂往往最难实际部署，因为超出孔体积限制的胺会堵塞活性位点。然而，用于直接空气捕集（DAC）的多目标贝叶斯优化（MOBO）通常将配方搜索视为无约束问题。我们将孔隙填充边界编码为显式结果约束 c(x) = tanh(1 - phi_hat) - tau，并将其作为诊断工具，应用于一个手工整理、可通过DOI追溯的基准数据集，该数据集包含来自43篇出版物的618个固体胺配方。无约束MOBO获得了更高的名义容量（3.639 ± 0.815 对比 2.964 ± 0.728 mmol/g，p = 1.0e-11），但在100个随机种子中，其帕累托最优候选方案中有63.2%违反孔隙填充边界，而有约束情形下仅为39.3%：即它在提出孔隙受限设计域之外的候选方案时反而获得了奖励。该约束引导帕累托前沿趋向更低的负载量（55.8 对比 65.2 wt%）和更大的孔隙。该边界得到……（原文摘要在此处截断）

    arXiv:2611.34154v1 Announce Type: new  Abstract: Amine-impregnated sorbents with the highest nominal CO2 capacity are frequently the least deployable, as amine beyond the pore-volume limit blocks active sites. Yet multi-objective Bayesian optimization (MOBO) for direct air capture (DAC) treats formulation search as unconstrained. We encode the pore-filling boundary as an explicit outcome constraint, c(x) = tanh(1 - phi_hat) - tau, and use it as a diagnostic instrument on a hand-curated, DOI-traceable benchmark of 618 solid-amine formulations from 43 publications. Unconstrained MOBO attains higher nominal capacity (3.639 +/- 0.815 vs. 2.964 +/- 0.728 mmol/g, p = 1.0e-11), but across 100 seeds 63.2% of its Pareto-optimal candidates violate the pore-filling boundary versus 39.3% under the constraint: it is rewarded for proposing candidates outside the pore-confined design regime. The constraint steers the front to lower loadings (55.8 vs. 65.2 wt%) and larger pores. The boundary is corrob
    
[^113]: SOC-Net：二阶锥约束神经网络

    SOC-Net: Second-Order-Cone-Constrained Neural Networks

    [https://arxiv.org/abs/2611.34136](https://arxiv.org/abs/2611.34136)

    SOC-Net是一种通过构造方式对任意输入（包括训练分布之外的输入）强制满足输入依赖二阶锥（SOC）约束的神经网络架构，无需锥求解器、迭代方案或内部可行点。

    

    在许多应用中，神经网络需要学习遵守先验知识（如物理定律、安全要求和运行限制）的输入-输出映射。这些要求约束了对于给定输入哪些输出是可容许的，并且通常可以表示为二阶锥（SOC）形式，包括范数界和鲁棒线性不等式。控制领域的例子包括摩擦锥、推力界以及输入鲁棒控制屏障函数（CBF）条件。然而，现有的施加一般输入依赖SOC约束的方法需要锥求解器、可行性依赖于收敛性的迭代方案，或针对每个输入的内部可行点。我们提出了SOC-Net，这是一种神经网络架构，它通过构造方式对每个输入强制执行输入依赖的SOC约束，即使在训练分布之外也能保证约束满足。SOC-Net将主干输出的仿射提升投影到固定乘积（原文截断）……

    arXiv:2611.34136v1 Announce Type: new  Abstract: In many applications, neural networks must learn input-output maps that respect prior knowledge such as physical laws, safety requirements, and operating limits. These requirements constrain which outputs are admissible for a given input and often admit second-order cone (SOC) representations, including norm bounds and robust linear inequalities. Examples in control include friction cones, thrust bounds, and input-robust control barrier function (CBF) conditions. However, existing approaches to enforcing general input-dependent SOC constraints require a conic solver, an iterative scheme whose feasibility depends on convergence, or an interior feasible point for each input. We propose SOC-Net, a neural network architecture that enforces input-dependent SOC constraints by construction for every input, even outside the training distribution. SOC-Net projects an affine lifting of the backbone output onto the reachable slice of a fixed produc
    
[^114]: 验证器即智能体：从分数到证据的通用验证

    Agent-as-a-Verifier: From Scores to Evidence for General Verification

    [https://arxiv.org/abs/2611.34132](https://arxiv.org/abs/2611.34132)

    该论文提出“验证器即智能体”系统，将角色分离的LLM智能体嵌入受形式化方法启发的声明-义务-证书协议中，使验证从仅输出分数升级为生成可审计的证据证书。

    

    验证在各种推理和学习范式中日益关键，例如测试时扩展、递归智能和自动化研究，在这些范式中，反馈能够提升效率与优化效果。常见的基于分数的验证器（即判别式奖励模型）为输出预测一个标量奖励。尽管这类分数对于过滤、奖励和排序很有用，但它们无法为判断提供明确且可审计的证据基础。与此同时，虽然现有的生成式过程奖励模型能够提供文本反馈，但大多数设计并未充分利用智能体能力，例如工具使用和迭代推理，而这些能力对于事实核查和细粒度声明验证至关重要。在本文中，我们通过引入“验证器即智能体”来解决这些局限性，这是一个能够产生证书的智能体验证系统，它将角色分离的LLM智能体置于受形式化方法启发的“声明-义务-证书”协议之中。

    arXiv:2611.34132v1 Announce Type: new  Abstract: Verification is increasingly critical across various inference and learning paradigms, such as test-time scaling, recursive intelligence and automated research, where feedback can improve efficiency and optimization. Common score-based verifiers, or discriminative reward models, predict a scalar reward for an output. While useful for filtering, rewarding, and ranking, such scores provide no explicit, auditable evidentiary basis for the judgment. In parallel, although existing generative process reward models offer text feedback, most designs do not fully exploit agentic capabilities, such as tool use and iterative reasoning, which are essential for fact-checking and fine-grained claim verification. In this paper, we address these limitations by introducing Agent-as-a-Verifier, an agentic, certificate-producing verification system that places role-separated LLM agents inside a formal-methods-inspired claim-obligation-certificate protocol.
    
[^115]: MUSE：面向毫米波ISAC的多任务不确定性感知自监督编码器

    MUSE: Multi-Task Uncertainty-Aware Self-Supervised Encoder for mmWave ISAC

    [https://arxiv.org/abs/2611.34122](https://arxiv.org/abs/2611.34122)

    提出MUSE统一框架，通过自监督transformer与掩码自编码在多个数据集上预训练，并利用证据分类头实现不确定性量化，以单一编码器取代针对各任务单独训练的多个模型，提升了毫米波人体感知的可扩展性、可迁移性和可靠性。

    

    集成感知与通信（ISAC）使无线系统能够使用相同的波形同时进行通信和环境感知。在毫米波频段，ISAC能够以高精度实现多样化的人体感知任务，例如定位、手势识别和姿态估计。然而，每个任务都需要训练一个专用模型，通常需要在任务特定的数据集上从头开始训练，且缺乏标记不可靠预测的机制。这损害了毫米波ISAC模型的可扩展性、可迁移性和可靠性。在本工作中，我们提出了MUSE（多任务不确定性感知自监督编码器），这是一个用于毫米波人体感知的统一框架。MUSE利用带有逐数据集标记化的自监督transformer模型，并通过掩码自编码在五个数据集上进行训练。在微调过程中，MUSE使用证据分类头来实现不确定性量化。MUSE用单个统一编码器取代了N个独立编码器。

    arXiv:2611.34122v1 Announce Type: cross  Abstract: Integrated sensing and communication (ISAC) allows a wireless system to use the same waveform for communication and environment sensing. At mmWave frequency bands, ISAC enables diverse human sensing tasks, such as localization, gesture recognition, and pose estimation, with high precision. However, each task requires training a dedicated model, typically from scratch on a task-specific dataset, with no mechanism to flag unreliable predictions. This hurts the scalability, transferability, and reliability of mmWave ISAC models. In this work, we propose MUSE (Multi-task Uncertainty-aware Self-supervised Encoder), a unified framework for mmWave human sensing. MUSE leverages a self-supervised transformer model with per-dataset tokenization and is trained via masked autoencoding across five datasets. During fine-tuning, MUSE uses evidential classification heads that enable uncertainty quantification. MUSE replaces N separate encoders with a 
    
[^116]: 致命无人机智能体中不稳定的行为边界

    Unstable Action Boundaries in Lethal Drone Agents

    [https://arxiv.org/abs/2611.34121](https://arxiv.org/abs/2611.34121)

    该论文研究了致命无人机LLM智能体行为边界的稳定性，发现部分智能体无需任何越狱即可对“人”等普通目标描述调用致命工具，且原本保守的智能体在简单的语义目标转变下会变得不安全。

    

    使用工具的大语言模型（LLM）智能体带来了仅靠文本拒绝行为无法涵盖的安全风险，尤其是当工具能够触发高影响动作时。这一担忧在军事环境中尤为突出，因为自主武器的存在使得将武力决策委托给软件智能体成为可能。在本工作中，我们研究了无人机智能体的行为边界在目标描述发生变化时是否保持稳定。我们在基于真实世界无人机图像的模拟感知-动作流水线中评估了近期流行的多个LLM，其中模型可以定位请求的目标并调用模拟的致命工具。我们发现了两种令人担忧的失败模式。第一，某些LLM智能体会针对普通的一般性描述词（如“人”）调用致命工具，根本无需任何越狱攻击。第二，一个在受保护描述词下最初表现保守的智能体，在简单的语义目标转变下可能变得不安全。

    arXiv:2611.34121v1 Announce Type: cross  Abstract: Tool-using large language model (LLM) agents raise safety risks that are not captured by text-only refusal behavior, especially when tools can trigger high-impact actions. This concern is salient in military settings, where autonomous weapons raise the possibility of delegating force decisions to software agents. In this work, we study whether a drone agent's action boundary is stable under changes to the target descriptor. We evaluate recent and popular LLMs in a simulated perception-action pipeline over real-world drone imagery, where models can localize requested targets and invoke a simulated lethal tool. We find two concerning failure modes. First, some LLM agents invoke the lethal tool for ordinary generic descriptors such as \emph{people}, requiring no jailbreak at all. Second, an agent that initially appears conservative under protected descriptors can become unsafe under a simple semantic target shift.
    
[^117]: 并非每个输入都需要全部资源：统一专家门控、模型级联与智能体编排

    Not Every Input Needs Everything: Unifying Expert Gating, Model Cascades, and Agent Orchestration

    [https://arxiv.org/abs/2611.34112](https://arxiv.org/abs/2611.34112)

    该系统性综述首次在统一的“先检查后分配”条件计算框架下，将令牌级专家门控、查询级模型级联与任务级多智能体编排三大孤立研究领域综合起来，分析了2,289项研究以揭示跨尺度的共性失败模式与可迁移的解决方案。

    

    在每一个输入上都统一执行庞大的神经权重、前沿基础模型或整个多智能体工作流，会带来巨大的计算、延迟和资金开销。条件计算通过动态决定每个请求应调用哪些内部专家、模型端点或专用智能体来缓解这一成本。然而，稀疏神经架构中的令牌级门控、查询级模型级联以及任务级多智能体协调，大多在彼此孤立的研究领域中发展，掩盖了反复出现的系统性故障和可迁移的解决方案。在本系统性综述中，我们在统一的“先检查后分配”两步框架下，综合了2017年至2026年9月间发表的2,289项研究（从13,884条独立记录中筛选），涵盖20个方法论家族。离散硬选择分别占令牌级、查询级和任务级文献的70%、83%和91%，计算开销……

    arXiv:2611.34112v1 Announce Type: new  Abstract: Executing monolithic neural weights, frontier foundation models, or entire multi-agent workflows uniformly on every input incurs substantial computational, latency, and monetary overhead. Conditional computation mitigates this cost by deciding dynamically which internal experts, model endpoints, or specialized agents to invoke per request. Yet token-level gating in sparse neural architectures, query-level model cascades, and task-level multi-agent coordination have largely progressed in disconnected research silos, obscuring recurring system failures and transferable fixes. In this systematic review, we synthesize 2,289 studies published between 2017 and September 2026 (selected from 13,884 unique records) across 20 methodological families under a unified two-step inspect-then-allocate framework. Discrete hard selection accounts for 70%, 83%, and 91% of token-, query-, and task-scale literature respectively, and computational overhead ap
    
[^118]: SysBio-SR：面向系统生物学中轨迹保真度与机制恢复的符号回归基准测试

    SysBio-SR: Benchmarking Symbolic Regression for Trajectory Fidelity and Mechanism Recovery in Systems Biology

    [https://arxiv.org/abs/2611.34106](https://arxiv.org/abs/2611.34106)

    本文提出SysBio-SR基准，包含198个带反应注释的生物ODE系统，首次系统评估符号回归方法在轨迹保真度与机制恢复两个维度上的表现，揭示了准确拟合系统轨迹并不等于恢复真实的生物反应机制。

    

    符号回归旨在从数据中发现动力学系统的可解释控制方程。然而，在系统生物学中，准确再现系统轨迹并不一定意味着能够恢复底层的反应机制。现有基准评估的是数值精度、表达式复杂度或方程层面的相似性，而不是恢复出的反应与生物系统中参考机制之间的一致性。我们提出了SysBio-SR，这是一个包含198个基于文献的生物ODE系统的基准，其反应注释涵盖质量作用、Michaelis-Menten、Hill以及复杂动力学。轨迹保真度在所有系统上于分布内重建和分布外（OOD）外推条件下进行评估，而机制恢复则在161个系统的子集上进行评估。我们评估了11种符号发现方法的轨迹保真度，并对三种反应基……（摘要原文不完整）

    arXiv:2611.34106v1 Announce Type: cross  Abstract: Symbolic regression seeks to discover interpretable governing equations for dynamical systems from data. In systems biology, however, accurately reproducing system trajectories does not necessarily imply recovery of the underlying reaction mechanisms. Existing benchmarks evaluate numerical accuracy, expression complexity, or equation-level similarity, but not agreement between recovered reactions and reference mechanisms in biological systems. We introduce SysBio-SR, a benchmark of 198 literature-based biological ODE systems with reaction annotations spanning mass action, Michaelis--Menten, Hill, and complex kinetics. Trajectory fidelity is evaluated across all systems under in-distribution reconstruction and out-of-distribution (OOD) extrapolation, while mechanism recovery is assessed on the subset of 161 systems. We evaluate trajectory fidelity across 11 symbolic-discovery methods and assess mechanism recovery for three reaction-base
    
[^119]: 机器学习集成方法能否胜过公布的招生名额？招生改革下的目标参数化、截尾基线与保形区间

    Do Machine Learning Ensembles Beat Published Quotas? Target Parameterization, Censored Baselines, and Conformal Intervals Under Admissions Reform

    [https://arxiv.org/abs/2611.34104](https://arxiv.org/abs/2611.34104)

    该研究基于越南2022-2026年本科招生面板数据，发现由于多数项目恰好按公布名额录取，结合截尾基线与保形区间的公布名额预测优于无锚定的机器学习集成方法，且最差5%项目的“幻影容量”贡献了一半以上的名额预测误差。

    

    在集中式高等教育统筹体系中，入学前公布的公共招生容量上限被各部委和机构分析人员广泛用作默认的入学预测依据。本文利用经过审计的2022至2026年越南本科面板数据（涵盖309所高校的26,895个项目-年度记录），并将其与严格滞后的t-1期职业空缺信号相关联，在5,385个2024年延续项目（采用五折按大学分组交叉验证）以及3,348个匹配的2025年项目（其中2,179个粒度一致，严格进行时间外验证）上，将机器学习、多层模型和截尾计量经济学估计方法与公布的招生名额进行对比评估。由于62.8%的2024年项目恰好录取其公布的招生名额，无锚定的对数增长集成方法降低了预测准确性，而表现最差的5%项目通过未填满的“幻影容量”贡献了总名额误差的一半。受控目标消融实验（包括在放宽的非约束性截剪条件下的单因子测试）……

    arXiv:2611.34104v1 Announce Type: cross  Abstract: Public admission capacity caps published prior to matriculation are widely used by ministries and institutional analysts as default intake projections in centralized postsecondary clearinghouses. Using an audited 2022 to 2026 Vietnamese undergraduate panel (26,895 program-year records across 309 universities) linked to strictly lagged t-1 occupational vacancy signals, this paper evaluates machine learning, multilevel, and censored econometric estimators against published quotas across 5,385 continuing 2024 programs (five-fold university-grouped cross-validation) and 3,348 matched (2,179 grain-consistent) 2025 programs strictly out of time. Because 62.8% of 2024 programs enroll their exact published quota, unanchored log-growth ensembles degrade accuracy, while the worst 5% generate half of total quota error via unfilled "phantom capacity." Controlled target ablations, including a single-factor test under widened non-binding clipping, s
    
[^120]: AlignPre：面向无潜在压缩生成式降水临近预报的技能对齐后训练

    AlignPre: Skill-Aligned Post-Training for Latent-Free Generative Precipitation Nowcasting

    [https://arxiv.org/abs/2611.34101](https://arxiv.org/abs/2611.34101)

    AlignPre提出了一种无需潜在压缩的像素空间降水临近预报框架，其核心创新是利用预报验证得分的解析结构构建可微分目标进行技能对齐后训练（SAPT），从而直接优化业务预报技能而非仅进行分布匹配。

    

    降水临近预报需要在快速演变的时空动态下进行准确的短期预测。生成式扩散模型可以通过建模条件分布来产生细致且具有竞争力的雷达预报。然而，大多数现有方法依赖于单独训练的组件，例如确定性分支或潜在压缩模块。更根本的是，仅靠分布匹配无法显式地优化业务预报技能。我们提出了AlignPre，一个精简的像素空间临近预报框架，通过技能对齐后训练（SAPT）来弥合这一差距。AlignPre采用具有层次化时空架构的端到端流匹配预报器，实现分布匹配与直接的雷达场合成，无需潜在压缩。在后训练阶段，SAPT利用预报验证得分的解析结构来构建可微分的目标函数，通过反向传播梯度……（摘要原文在此处截断）

    arXiv:2611.34101v1 Announce Type: cross  Abstract: Precipitation nowcasting requires accurate short-term predictions under rapidly evolving spatiotemporal dynamics. Generative diffusion models can produce detailed and competitive radar forecasts by modeling conditional distributions. However, most existing approaches rely on separately trained components, such as deterministic branches or latent compression modules. More fundamentally, distribution matching alone fails to explicitly optimize operational forecast skill. We propose AlignPre, a streamlined pixel-space nowcasting framework that bridges this gap via skill-aligned post-training (SAPT). AlignPre employs an end-to-end flow-matching forecaster with a hierarchical spatiotemporal architecture for distribution matching and direct radar-field synthesis, without latent compression. During post-training, SAPT exploits the analytic structure of forecast verification scores to construct a differentiable objective. By backpropagating gr
    
[^121]: 探索像素空间扩散模型的设计因素

    Exploring the Design Factors of Pixel-space Diffusion

    [https://arxiv.org/abs/2611.34093](https://arxiv.org/abs/2611.34093)

    该论文提出EiT——一种带通道瓶颈的U形像素扩散Transformer，从架构、表示和频域三个视角系统揭示了像素空间扩散的关键设计因素，发现更早的去噪与更长的基提升阶段有助于更好的生成质量。

    

    在本文中，我们从模型架构、表示和频域三个角度探索像素空间扩散的关键设计因素，并由此提出了EiT。受去噪自编码器（DAE）通过瓶颈执行去噪的启发，我们构建了一个U形像素扩散Transformer，该模型沿通道维度进行显式压缩。在表示方面，我们发现像素扩散Transformer与潜在扩散Transformer在各层和时间步上都表现出三阶段的表示动态：去噪/局部特征扩展、基提升以及后验投影。此外，更早的去噪和更长的基提升阶段与更好的生成质量相关，而EiT通过其通道瓶颈和增加的网络深度自然地实现了这一点。从频率角度来看，我们在保持网络运行的条件下建立了像素空间与频域扩散之间的等价关系。

    arXiv:2611.34093v1 Announce Type: cross  Abstract: In this paper, we introduce EiT by exploring the key design factors of pixel-space diffusion from three perspectives: model architecture, representation, and frequency domain. Inspired by Denoising Autoencoders (DAE) which perform denoising through a bottleneck, we construct a U-shaped pixel diffusion transformer that performs explicit compression along the channel dimension. In terms of representation, we find that both pixel and latent diffusion transformers exhibit a three-stage representation dynamics across layers and timesteps: denoising/local feature expansion, basis lift-up, and posterior projection. Moreover, earlier denoising and a longer basis lift-up stage are associated with better generation, which EiT naturally achieves through its channel bottleneck and increased network depth. From the frequency perspective, we establish the equivalence between pixel-space and frequency-space diffusion when keeping the network operatin
    
[^122]: 通过隐状态混合与梯度分离改进尾部增强的推测草稿生成

    Improving Tail-Augmented Speculative Drafting with Hidden State Mixing and Gradient Detach

    [https://arxiv.org/abs/2611.34092](https://arxiv.org/abs/2611.34092)

    本文提出两种仅用于训练阶段的方法——隐状态混合（HSM）和隐状态梯度分离（HSGD），在不显式模拟推理、不增加额外内存和专用注意力掩码的情况下，缓解推测解码草稿模型的训练-推理不匹配，从而改进自回归草稿生成。

    

    arXiv:2611.34092v1 公告类型：cross。多 token 预测（MTP）模块和 EAGLE 风格的草稿模型通过复用目标模型的隐状态来支持高效的推测解码。然而，它们的自回归使用方式带来了训练与推理之间的不匹配：训练时提供的隐状态序列来自骨干模型或单一的 MTP 步骤，而推理时则面向目标模型的前缀，并附加一段由递归生成的 MTP 状态构成的、不断增长的尾部。EAGLE-3 的训练时测试（TTT）方法在训练期间显式模拟了这一行为，但代价是额外的内存开销和专门的注意力掩码。我们提出两种仅作用于训练阶段的方法来改进自回归草稿生成：隐状态混合在对应同一预测目标的骨干隐状态和 MTP 隐状态之间进行采样，使草稿模型接触到来自不同 MTP 步骤的状态组合，而无需显式模拟推理过程；隐状态梯度分离则是一种正则化方法，它随机地……（摘要原文在此处截断）

    arXiv:2611.34092v1 Announce Type: cross  Abstract: Multi-token prediction (MTP) modules and EAGLE-style drafters support efficient speculative decoding by reusing target-model hidden states. Their autoregressive use, however, introduces a training-inference mismatch: training supplies sequences of hidden states from the backbone or a single MTP step, whereas inference attends to a target-model prefix followed by a growing tail of recurrently generated MTP states. EAGLE-3's training-time test (TTT) explicitly simulates this behavior during training, at the cost of additional memory and specialized attention masks. We introduce two training-only methods for improving autoregressive drafting. Hidden State Mixing (HSM) samples among backbone and MTP hidden states aligned to the same prediction target, exposing the drafter to mixtures of states from different MTP steps without explicitly simulating inference. Hidden State Gradient Detach (HSGD) is a regularization method that stochastically
    
[^123]: 布赫伯格算法中基于特征工程的强化学习S-对选择方法

    Feature Engineered Reinforcement Learning for S-pair Selection in Buchberger's Algorithm

    [https://arxiv.org/abs/2611.34088](https://arxiv.org/abs/2611.34088)

    本文提出用17维统计特征向量描述多项式，将强化学习方法从二项式理想扩展到一般多项式理想的S-对选择，在所有测试分布上显著优于基线方法，相比度数启发式减少了41–53%的算法工作量。

    

    在布赫伯格算法中计算格罗布纳基时，选择处理S-对的顺序是一个至关重要的决策：这一选择可能使计算成本改变几个数量级。最近的研究表明，观察多项式指数向量的强化学习智能体在二项式理想上优于经典启发式方法。我们通过使用统计特征将这一研究方向扩展到一般多项式理想。我们引入了一个17维的特征向量来描述多项式，并使用PPO算法在五种随机理想分布上训练策略。我们发现，在所有五种分布上，使用S-多项式特征都显著优于S-对特征。新策略在所有五种分布上也优于重新训练的强化学习基线，在学习有效的四种分布上，相对于度数启发式方法减少了41–53%的算法工作量。特征工程带来的额外开销意味着实际运行时间的成本……（原文在此处截断）

    arXiv:2611.34088v1 Announce Type: new  Abstract: Selecting the order in which to process S-pairs is a crucial decision in Buchberger's algorithm for computing Groebner bases: this choice can change the cost by orders of magnitude. Recent work has shown that reinforcement learning agents viewing exponent vectors of the polynomials outperform classical heuristics on binomial ideals. We extend this line of work to general polynomial ideals by using statistical features. We introduce a 17-dimensional feature vector to describe a polynomial and use PPO to train policies on five distributions of random ideals. We find that using S-polynomial features significantly outperforms S-pair features on all five distributions. The new policies also outperform the retrained RL baseline on all five distributions, reducing algorithmic work by 41--53\% relative to the degree heuristic on four distributions where learning helps. The additional cost of the feature engineering means that wall-clock cost rem
    
[^124]: 通过希尔伯特空间提升的概率测度上多目标优化的加速近端梯度方法

    Accelerated Proximal Gradient Methods for Multiobjective Optimization over Probability Measures via Hilbert-Space Lifting

    [https://arxiv.org/abs/2611.34073](https://arxiv.org/abs/2611.34073)

    本文提出基于希尔伯特空间提升的多目标Wasserstein近端梯度方法（MWPG）及其加速版本（MWAPG），通过隐式Wasserstein梯度步求解概率测度上KL散度的Pareto多目标优化问题，并提供严格的离散时间收敛保证。

    

    许多采样和推断任务需要一个能够平衡多个目标的单一分布。我们将该任务表述为在概率测度上对Kullback-Leibler (KL)散度进行Pareto意义下的多目标优化。粒子方法MWGraD（Nguyen等，2025）使用显式Wasserstein梯度步，而其加速版本A-MWGraD（Nguyen等，2026）缺乏离散时间保证。我们提出了使用隐式Wasserstein梯度步的多目标Wasserstein近端梯度方法（MWPG），以及其加速版本——具有严格离散时间保证的多目标Wasserstein加速近端梯度方法（MWAPG）。我们的核心思想是希尔伯特空间提升：每个测度被表示为希尔伯特空间中一个映射对固定基础测度的推前。该问题由此转化为关于传输映射的约束多目标复合优化问题，从而可以应用加速近端梯度方法求解。

    arXiv:2611.34073v1 Announce Type: cross  Abstract: Many sampling and inference tasks require a single distribution that balances several targets. We formulate this task as multiobjective optimization of Kullback--Leibler (KL) divergences over probability measures, in the Pareto sense. The particle method MWGraD (Nguyen et al., 2025) uses explicit Wasserstein gradient steps, and its accelerated version A-MWGraD (Nguyen et al., 2026) lacks discrete-time guarantees. We propose the multiobjective Wasserstein proximal gradient (MWPG), which uses implicit Wasserstein gradient steps, and its accelerated version, multiobjective Wasserstein accelerated proximal gradient (MWAPG), with rigorous discrete-time guarantees. Our key idea is a Hilbert-space lifting: each measure is represented as the pushforward of a fixed base measure by a map in a Hilbert space. The problem then becomes a constrained multiobjective composite problem over transport maps, to which accelerated proximal gradient methods 
    
[^125]: CARE-WM：面向潜在世界模型的同视界自条件化与约束自适应正则化

    CARE-WM: Same-Horizon Self-Conditioning with Constraint-Adaptive Regularization for Latent World Models

    [https://arxiv.org/abs/2611.34054](https://arxiv.org/abs/2611.34054)

    CARE-WM通过同视界自条件化训练与约束自适应的潜在正则化，在无需额外监督、独立代价模型或架构改动的情况下，同时提升潜在世界模型的多步预测精度和潜在空间与任务代价的对齐程度，从而提高规划成功率。

    

    基于潜在世界模型的规划既需要准确的多步预测，也需要能反映与目标物理接近程度的潜在距离。然而，模型通常在真实观测上训练，但在规划时必须依赖自身的想象，而更近的潜在距离并不一定对应更好的任务结果。现有方法依赖辅助训练目标或额外的任务代价读出模块，前者会带来额外训练开销，后者则使潜在几何结构本身仍与任务代价不对齐。我们提出CARE-WM，这是一种即插即用的训练方法，无需额外监督、无需独立的代价模型，也无需改变LeWM的架构和欧氏代价规划器即可同时满足上述两项要求。CARE-WM在预测器自身的预测上进行训练，并自适应地校准潜在正则化的强度。仅需单一超参数配置，CARE-WM即可提升平均规划成功率（原文摘要在此处截断）。

    arXiv:2611.34054v1 Announce Type: new  Abstract: Planning with latent world models requires both accurate multi-step predictions and latent distances that reflect physical proximity to the goal. However, models are typically trained on ground-truth observations but must rely on their own imaginations during planning, while closer latent distances do not necessarily correspond to better task outcomes. Existing methods resort to auxiliary training objectives or additional task-cost readouts, which respectively incur extra training overhead or leave the latent geometry itself misaligned with task cost. We introduce CARE-WM, a drop-in training method that meets both requirements without additional supervision, a separate cost model, or any change to LeWM's architecture and Euclidean-cost planner. CARE-WM trains the predictor on its own predictions and adaptively calibrates the strength of latent regularization. With a single hyperparameter configuration, CARE-WM improves mean planning succ
    
[^126]: 稳定的门控DeltaNet-2：理解与缓解循环状态不稳定性

    Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability

    [https://arxiv.org/abs/2611.34051](https://arxiv.org/abs/2611.34051)

    本文发现GDN-2训练失败源于擦除操作中读写方向不对称导致的循环状态指数级增长，并提出通过对称擦除与衰减调节残差写入来解决该问题，同时理论上证明了状态增长的O(√T)上界并保留分块并行计算能力。

    

    线性注意力凭借线性复杂度和固定大小的循环状态，为长序列建模提供了高效解决方案。门控DeltaNet-2（GDN-2）引入了对擦除和写入的独立通道级控制，增强了状态更新的表达能力以及上下文内检索能力，但它在长上下文训练中会出现训练失败。我们将这些失败归因于擦除过程中读取方向与修改方向之间的不对称性，这种不对称性产生了非正规（non-normal）的擦除算子，其方向性增益会在token之间累积，即使没有新的写入也会引发状态的指数级增长。为解决这一问题，我们提出了该模型，采用对称擦除以确保状态传播的非扩张性，并通过衰减调节的残差写入来保留写入的灵活性。在理论上，我们在保留分块并行计算的同时，建立了循环状态增长的 O(√T) 上界。在约16亿参数模型上的实验……（摘要原文在此处截断）

    arXiv:2611.34051v1 Announce Type: new  Abstract: Linear attention offers an efficient solution for long-sequence modeling via linear complexity and fixed-size recurrent states. Gated DeltaNet-2 (GDN-2) introduces independent channel-wise control over erasure and writing, enhancing state-update expressiveness and in-context retrieval, yet it suffers from training failures during long-context training. We trace these failures to the asymmetry between reading and modification directions during erasure, which yields non-normal erase operators whose directional gains accumulate across tokens, triggering exponential state growth even without new writes. To resolve this, we propose \model{}, employing symmetric erasure to ensure non-expansive state propagation and decay-regulated residual writing to preserve writing flexibility. Theoretically, we establish an $O(\sqrt{T})$ upper bound on recurrent state growth while retaining chunkwise parallel computation. Experiments on ~1.6B parameter mode
    
[^127]: 面向低轨星座蜂窝流量预测的去中心化聚类联邦学习

    Decentralized Clustered Federated Learning for Cellular Traffic Forecasting over LEO Constellations

    [https://arxiv.org/abs/2611.34049](https://arxiv.org/abs/2611.34049)

    该论文将去中心化联邦聚类（DFCA）扩展到低轨卫星星座场景，通过k近邻星间gossip协作与零样本头选择机制，在无地面聚合器的条件下让每颗卫星自适应地预测其下方地面小区的蜂窝流量，显著降低预测误差。

    

    低地球轨道（LEO）卫星星座有望在新兴的6G非地面网络中充当轨道边缘计算节点，为地面蜂窝基础设施的预测与控制任务提供支持。然而，训练此类模型面临多重挑战：地面小区流量模式的异构性、与地面站之间链路的间歇性，以及卫星自身的轨道运动——它会持续改变每颗卫星所观测到的数据。在本工作中，我们将去中心化联邦聚类（DFCA）扩展到LEO场景，并将其应用于Telecom Italia米兰+特伦托数据集上的蜂窝互联网流量预测。卫星之间仅通过k近邻的星间gossip通信进行协作，无需地面侧聚合器；同时，零样本头选择机制使每颗卫星能够专门适配其正下方地面小区的潜在流量模式。在训练后K=20个轨道快照上的平均结果显示，我们的扩展方法降低了（预测误差）……

    arXiv:2611.34049v1 Announce Type: new  Abstract: Low Earth Orbit (LEO) satellite constellations are positioned to act as orbital edge-compute nodes in emerging 6G non-terrestrial networks, supporting forecasting and control tasks for ground cellular infrastructure. However, training such models is challenged by heterogeneous ground-cell traffic patterns, intermittent links to ground stations, and the satellites' own orbital motion, which continuously shifts the data each satellite observes. In this work, we extend Decentralized Federated Clustering (DFCA) to the LEO setting and apply it to cellular internet-traffic forecasting on the Telecom Italia Milan+Trentino dataset. Satellites cooperate exclusively through $k$-nearest inter-satellite gossip with no ground-side aggregator, while zero-shot head selection lets each satellite specialize to the latent traffic regime of the ground cell currently beneath it. Averaged over $K{=}20$ post-training orbital snapshots, our extension reduces f
    
[^128]: 基于智能体模型编辑的视觉-语言模型小样本适配

    Few-Shot Adaptation of Vision-Language Models via Agentic Model Editing

    [https://arxiv.org/abs/2611.34036](https://arxiv.org/abs/2611.34036)

    该论文提出AME框架，让LLM智能体基于少量样本和对模型的观察，自主设计并迭代优化包含输入变换、编码器编辑和分类器拟合的领域定制适配程序，实现视觉-语言模型的小样本适配。

    

    小样本适配旨在利用少量带标注的图像将预训练的视觉-语言模型（VLM）迁移到新领域。大多数方法在预先指定的适配流程内拟合提示、表征或分类器。然而，识别所需的视觉差异在不同领域之间可能存在很大差异，这引出了一个 问题：LLM 智能体能否利用少量样本示例和模型观察来设计适配流程本身？我们提出了智能体模型编辑（AME），这是一个智能体框架，它将输入变换、编码器编辑、特征计算和分类器拟合联合设计为一个可执行的、领域定制的适配程序。该智能体检查图像、类别语义、模型参数、中间表征和激活值，以形成关于目标任务的假设。它在支持集上实现并测试候选程序，以总体验证准确率指导改进和……

    arXiv:2611.34036v1 Announce Type: new  Abstract: Few-shot adaptation transfers pretrained Vision-Language Models (VLMs) to new domains using a small set of labeled images. Most methods fit prompts, representations, or classifiers within an adaptation procedure specified in advance. Yet the visual distinctions needed for recognition can differ substantially across domains, motivating the question: Can an LLM agent use few-shot examples and model observations to design the adaptation procedure itself? We propose Agentic Model Editing (AME), an agentic framework that jointly designs input transformations, encoder edits, feature computation, and classifier fitting as an executable, domain-customized adaptation program. The agent inspects images, class semantics, model parameters, intermediate representations, and activations to formulate hypotheses about the target task. It implements and tests candidate programs on the support set, with aggregate validation accuracy guiding refinement and
    
[^129]: 面向跨团队规模的策略遵循型多智能体深度强化学习的训练后策略复制方法

    Post-Training Policy Replication for Strategy-Following Multi-Agent Deep Reinforcement Learning across Team Sizes

    [https://arxiv.org/abs/2611.34035](https://arxiv.org/abs/2611.34035)

    该论文提出了一种与团队规模无关的观测架构，通过将单个预训练策略复制到不同规模团队的各智能体并辅以热启动再训练，显著降低了策略遵循型多智能体强化学习的训练成本并提升了跨团队规模的可迁移性。

    

    arXiv:2611.34035v1 公告类型： cross 摘要： 尽管多智能体深度强化学习能够实现协调行为，但由于智能体间复杂的交互作用，使其与管理意图保持一致仍然十分困难。虽然策略遵循分布式注意力行动者架构通过区域指令提升了可控性，但指令组合的数量会随团队规模和指令规模呈指数级增长，导致训练成本过高且可迁移性差。为降低这一计算成本，我们引入了一种与团队规模无关的观测架构，并通过将单个已训练策略复制到不同规模团队中的各个智能体上来重构集体行为，从而免除了从头训练的需要。此外，我们通过使用预训练策略对策略网络进行初始化来评估热启动再训练，以显著降低计算开销。在一个物体收集游戏中，在四智能体团队中训练的策略被部署到……（原文摘要此处截断）

    arXiv:2611.34035v1 Announce Type: cross  Abstract: Although multi-agent deep reinforcement learning enables coordinated behavior, aligning it with managerial intent remains difficult due to complex agent interactions. Although the strategy-following distributed attentional actor architecture (sfDA7-X) improves controllability via regional instructions, instruction combinations scale exponentially with team and instruction sizes, leading to prohibitive training costs and poor transferability. To reduce this computational cost, we introduce a team-size-independent observation architecture and reconstruct collective behavior by replicating a single trained policy across agents in teams of different sizes, thereby eliminating the need to train from scratch. Additionally, we evaluate warm-start retraining by initializing policy networks with a pre-trained policy to significantly lower computational overhead. In an object collection game, a policy trained in a four-agent team is deployed to 
    
[^130]: 通过多传感器卫星时间序列基准测试评估AlphaEarth基础模型嵌入在农业制图中的物理可解释性

    Assessing the Physical Interpretability of AlphaEarth Foundation Embeddings Through Multi-Sensor Satellite Time-Series Benchmarking for Agricultural Mapping

    [https://arxiv.org/abs/2611.34019](https://arxiv.org/abs/2611.34019)

    本研究在摩洛哥农业制图任务中对AlphaEarth基础模型嵌入与多传感器卫星时间序列特征进行了基准测试，发现AEF嵌入精度（0.91）接近多时相时间序列特征的最优结果（0.93），证明其可作为传统遥感产品的高效可迁移替代方案，并具备重建生物物理信息的潜力。

    

    精确且动态更新的农业土地利用/土地覆盖（LULC）地图对于农业水资源管理至关重要，尤其是在水资源普遍短缺的地区，例如摩洛哥。本研究评估了AlphaEarth基础模型（AEF）嵌入在摩洛哥农业LULC和灌溉类型制图中的效用。研究将AEF嵌入与单日期和多时相时间序列遥感特征进行了比较，使用了三种机器学习分类器：XGBoost、CatBoost和TabPFN。此外，本研究还评估了AEF嵌入是否能够：(a) 提供一种高效且可迁移的替代方案，以取代直接使用传统地球观测产品的方法；(b) 重建生物物理信息。结果表明，多时相特征时间序列取得了最高性能，使用TabPFN时农业LULC制图精度达到0.93，灌溉类型制图精度达到0.87。而AEF嵌入也取得了可比的结果，精度达到0.91。

    arXiv:2611.34019v1 Announce Type: cross  Abstract: Accurate and dynamic maps of agricultural LULC are important for agricultural water resource management, particularly in areas where water scarcity is prevalent, such as Morocco. Here, we evaluate the utility of AlphaEarth Foundation Model (AEF) embeddings for agricultural LULC and irrigation-type mapping in Morocco. AEF embeddings were compared to single-date and time-series remote-sensing features using three machine-learning classifiers: XGBoost, CatBoost, and TabPFN. We also evaluated whether AEF embeddings can: (a) provide an efficient and transferable alternative to conventional direct use of Earth observation products and (b) reconstruct biophysical information. Results showed that multi-temporal feature time series achieved the highest performance, with accuracies of 0.93 for agricultural LULC and 0.87 for irrigation-type mapping when using TabPFN. However, AEF embeddings yielded comparable results, achieving accuracies of 0.91
    
[^131]: 异联想视觉运动记忆：软召回作为少演示模仿中学习残差的基础

    Hetero-Associative Visuomotor Memory: Soft Recall as a Base for Learned Residuals in Few-Demonstration Imitation

    [https://arxiv.org/abs/2611.34014](https://arxiv.org/abs/2611.34014)

    研究发现，在少演示模仿的检索-修正框架中，让演示记忆提供对整个记忆库进行softmax加权的平滑软召回作为修正器的基底，相比传统修正最近邻检索片段的方式可显著提升策略成功率，且该优势在稀疏、多样的真实人类演示上最为突出。

    

    检索-修正策略通常对检索到的最近邻演示片段进行细化，这种读取方式在直接执行时效果最佳。我们研究了演示记忆应当向修正器提供什么内容，发现对整个记忆库进行平滑的、softmax加权的软召回是一个更好的基础。在ALOHA方块传递任务（40条人类演示）上，软召回单独执行时并不比最近邻片段更优；然而在键、候选和训练保持固定的情况下，对软召回进行修正，在使用新编码器的全新重置环境上成功率提升22.4个百分点，相比闭环调优的top-1修正器提升7.4个百分点。在robomimic Square任务上，配合两个更优的执行算子时领先7.9个百分点，在同等规模的无标签抽取数据上领先6.7个百分点。这一增益在稀疏、多样的演示场景下，采用动作感知键和留一演示法训练时依然成立，而在密集、脚本化或生成的数据上则减弱甚至略有逆转。

    arXiv:2611.34014v1 Announce Type: cross  Abstract: Retrieve-and-correct policies typically refine the nearest retrieved demonstration chunk, the   readout that works best when executed directly. We study what a demonstration memory should   supply to the corrector instead and find that a smooth, softmax-weighted recall over the whole   memory is a much better base. On ALOHA cube transfer (40 human demonstrations), soft recall is   no better to execute alone than the nearest chunk; however, with keys, candidates, and training   fixed, correcting it raises success by 22.4 points on fresh resets with new encoders and by 7.4   against a closed-loop-tuned top-1 corrector. On robomimic Square, it leads by 7.9 points with   the two better operators and by 6.7 on a label-free draw of equal size. This gain holds on   sparse, varied demonstrations under an action-aware key and leave-one-demonstration-out   training, and fades or slightly reverses on dense, scripted, or generated data. Building o
    
[^132]: CARROT：基于半松弛最优传输的竞争性自适应图聚类

    CARROT: Competitive Adaptive Graph Clustering via Semi-Relaxed Optimal Transport

    [https://arxiv.org/abs/2611.34012](https://arxiv.org/abs/2611.34012)

    该论文提出CARROT框架，将半松弛最优传输引入深度图聚类，无需预先指定聚类数量即可自适应地确定簇数，并能有效处理簇规模高度不平衡的问题。

    

    深度图聚类（DGC）已成为无监督属性图聚类的重要范式。然而，现有方法在实际应用中常常面临两个挑战：聚类数量 $K$ 事先未知，且各聚类规模经常高度不平衡。依赖预先指定 $K$ 的方法在 $K$ 设定错误时可能无法恢复潜在的聚类结构，而许多方法隐式地偏好均衡的划分，阻碍了少数聚类的发现。我们提出了CARROT，一种通过半松弛最优传输实现的竞争性自适应图聚类框架，它将半松弛最优传输引入深度图聚类，以联合应对上述两个挑战。CARROT首先预训练节点嵌入以建立一个有意义的嵌入空间，然后通过自适应聚类机制联合优化嵌入和聚类分配。

    arXiv:2611.34012v1 Announce Type: cross  Abstract: Deep graph clustering (DGC) has emerged as a prominent paradigm for unsupervised attributed graph clustering. However, existing methods often struggle with two practical challenges encountered in real-world applications: the number of clusters $K$ is unknown a priori, and cluster sizes are frequently highly imbalanced. Methods relying on a pre-specified $K$ may fail to recover the underlying clustering structure when $K$ is misspecified, while many approaches implicitly favor balanced partitions, hindering the discovery of minority clusters. We propose CARROT, a Competitive Adaptive gRaph clustering framework via semi-Relaxed Optimal Transport, which introduces semi-relaxed optimal transport into deep graph clustering to jointly address both challenges. CARROT first pre-trains node embeddings to establish a meaningful embedding space, and then jointly optimizes embeddings and cluster assignments through an Adaptive Clustering Mechanism
    
[^133]: 参考基准至关重要：面向权重空间学习的位移探测方法

    The Reference Matters: Displacement Probing for Weight-Space Learning

    [https://arxiv.org/abs/2611.33998](https://arxiv.org/abs/2611.33998)

    该论文提出位移探测（DSProbe），通过探测模型相对于参考起点（如预训练权重）的位移而非原始权重，并结合学习一个需减去的共享参考成分，从而提升了基于探测的权重空间学习在预测模型属性上的效果。

    

    权重空间学习旨在直接从模型参数预测模型的属性，而基于探测的方法通过读取模型对学习到的探测信号（probe）的响应来高效实现这一目标。先前的研究表明，探测方法在“模型树”（即从同一个共享预训练起点微调而来的一组模型）内能够成功，但并未解释其原因。我们将此归因于共享起点本身：由于树中的各叶子模型都从相同的权重出发，真正区分它们的是它们相对于该起点的位移。这启发我们去探测位移而非原始权重，我们将这种干预称为位移探测。我们从两个方向深化这一思路：一是将其扩展到二阶Gram响应——其位移在一阶意义上能够追踪微调所引起的特征变化；二是将参考基准本身视为一种设计选择，在预训练起点之外额外学习一个需要减去的共享成分。在Model Jungle基准的四种架构上，所得的改进方法……（原文摘要在此处截断）

    arXiv:2611.33998v1 Announce Type: new  Abstract: Weight-space learning predicts a model's attributes directly from its parameters, and probing-based methods do so efficiently by reading a model's responses to learned probes. Prior work shows that probing succeeds within a Model Tree -- models fine-tuned from a shared pretrained origin -- but not why. We trace this to the shared origin itself: since a tree's leaves start from the same weights, what distinguishes them is their displacement from that origin. This motivates probing the displacement rather than the raw weight, an intervention we call the DiSplacement Probe (DSProbe). We develop this idea along two axes: extending it to second-order Gram responses, whose displacement, to first order, tracks the feature changes induced by fine-tuning; and treating the reference as a design choice, learning a shared component to subtract beyond the pretrained origin. Across four architectures on the Model Jungle benchmark, the resulting Refine
    
[^134]: MolGPAO：具有推测解码与亚毫秒接缝愈合功能的分子生成式预训练自回归八叉树（面向3D分子）

    MolGPAO: Molecular Generative Pre-trained Autoregressive Octree with Speculative Decoding and Sub-Millisecond Seam Healing for 3D Molecules

    [https://arxiv.org/abs/2611.33985](https://arxiv.org/abs/2611.33985)

    提出MolGPAO——一个独立的3D八叉树自回归分子生成架构，通过6维无尺度预算嵌入、推测树解码与反捷径训练，在不依赖外部代理、强化学习或推理时MLIP弛豫的情况下，实现对大型类药物分子高效且几何稳定的3D生成。

    

    在自适应3D八叉树上进行的自回归生成（Uni-3DAR）在大型类药物分子上相比稠密的O(N²)等变扩散与流模型具有内在的O(N log N)稀疏性优势，但由于Morton接缝坐标跳变、偏向叶节点的计算导致的根节点数量困局，以及内部评判器中的目标捷径坍缩，其在几何平衡性、闭壳层价键稳定性和性质保真度方面仍存在不足。我们提出了MolGPAO（分子生成式预训练自回归八叉树），这是一个独立的3D八叉树架构和单输出推测树解码引擎，不使用任何外部代理模型、不使用强化学习、也不在推理时进行MLIP弛豫。MolGPAO统一整合了：(1) 带有L5到L1共价堆积先验的非困局6维ScaleFreeBudgetEmbedding；(2) 基于FlashAttention-2的暂停-汇聚推测八叉树漏斗（m_s1 → m_s2 → 1，M_out = 1）；(3) 反捷径真值训练策略……（摘要在此处被截断）

    arXiv:2611.33985v1 Announce Type: cross  Abstract: Autoregressive generation over adaptive 3D octrees (Uni-3DAR) offers an intrinsic $O(N \log N)$ sparsity advantage over dense $O(N^2)$ equivariant diffusion and flow models on large drug-like molecules, yet lags in geometric equilibrium, closed-shell valence stability, and property fidelity due to Morton-seam coordinate jumps, leaf-heavy compute with root-count trapping, and target-shortcut collapse in internal critics. We introduce MolGPAO (Molecular Generative Pre-trained Autoregressive Octree), a standalone 3D octree architecture and single-output speculative tree-decoding engine using zero external surrogates, zero reinforcement learning, and zero inference-time MLIP relaxation. MolGPAO unifies: (1) non-trapping 6-D ScaleFreeBudgetEmbedding with L5-to-L1 covalent packing priors; (2) a FlashAttention-2 pause-and-gather speculative octree funnel ($m_{s1} \to m_{s2} \to 1$, $M_{\mathrm{out}} = 1$); (3) an anti-shortcut ground-truth Tw
    
[^135]: 四元数-Tucker ZigZag网络：将Transformer骨干参数减半并实现精确的二维零损失持续扩展

    Quaternion-Tucker ZigZag Networks: Halving Transformer Backbone Parameters and Enabling Exact 2D Zero-Loss Continual Scaling

    [https://arxiv.org/abs/2611.33983](https://arxiv.org/abs/2611.33983)

    该论文提出QuatTucker-ZigZagGPT架构，通过四元数压缩、块对角Tucker分解、ZigZag菱形路由和GatedBounce自蒸馏技术，将Transformer骨干参数减半，同时实现宽度和深度两个维度上的精确零损失持续扩展。

    

    标准Transformer为每个前馈网络（FFN）层分配独立的稠密权重矩阵，使参数量与深度线性耦合（O(L)），并在扩展模型容量时丢弃先前的计算成果。相反，权重共享的循环Transformer（Looped Transformers）将深度与参数解耦（O(1)），但强制执行时间均匀的更新。我们将L层FFN堆栈表述为3D张量长方体，并引入QuatTucker-ZigZagGPT，统一了以下四个方面：(1) bfloat16格式下的4D四元数汉密尔顿矩阵乘法（GEMM），实现4倍FFN压缩；(2) 用于在共享基上实现深度特定旋转的64x64块对角Tucker SwiGLU核心；(3) 用于零FLOP自顶向下精炼的辫状ZigZag菱形路由（L0 -> L2 -> L1 -> L3）；(4) 宽度和深度两个维度上的精确2D零损失持续扩展。我们还引入了退火门控反弹脚手架（GatedBounce），经自蒸馏后可实现0.0%的推理开销。在算法任务上（8位奇偶校验、56个种子的路由消融实验、Mod-53 G……（摘要截断）

    arXiv:2611.33983v1 Announce Type: new  Abstract: Standard Transformers allocate independent dense weight matrices to every feed-forward network (FFN) layer, coupling parameters linearly with depth (O(L)) and discarding prior compute when scaling capacity. Conversely, weight-tied Looped Transformers decouple depth from parameters (O(1)) but enforce time-homogeneous updates. We formulate the L-layer FFN stack as a 3D Tensor Cuboid and introduce QuatTucker-ZigZagGPT, unifying: (1) 4D Quaternion Hamilton GEMMs in bfloat16 (4x FFN compression), (2) 64x64 Block-Diagonal Tucker SwiGLU Cores for depth-specific rotations over a shared basis, (3) Braided ZigZag Diamond Routing (L0 -> L2 -> L1 -> L3) for zero-FLOP top-down refinement, and (4) Exact 2D Zero-Loss Continual Scaling in both width and depth. We also introduce Annealed Gated Bounce-Back Scaffolding (GatedBounce), which self-distills to 0.0% inference overhead. Across algorithmic tasks (8-bit Parity, 56-seed routing ablations, Mod-53 Gr
    
[^136]: Kolpona：一个用于RTL生成的无梯度神经元编程智能体框架

    Kolpona: An Agentic Framework with Gradient-Free Neuron Programming for RTL Generation

    [https://arxiv.org/abs/2611.33973](https://arxiv.org/abs/2611.33973)

    Kolpona是一个无需梯度训练的智能体框架，通过神经元编程（以闭式方式直接改写模型权重中的单个神经元）并仅在新任务测试结果有利时保留编辑，结合基于规范与仿真的无标签设计选择，显著提升开放权重LLM的Verilog生成性能，在Qwen3.5-4B上将VerilogEval v2 pass@1从31.7%提升至77.6%。

    

    面向寄存器传输级（RTL）设计的开放权重大型语言模型（LLM）通常通过在精选的Verilog语料库上进行微调来改进，但这需要大量经过验证的设计数据集和GPU训练。我们提出了Kolpona，一个无需梯度训练即可改进此类模型的智能体框架。其第一个组件是神经元编程，这是一种新的权重空间编程（WSP）形式，以闭式方式将单个神经元写入模型权重中，以消除模型在实践任务中发现的失败习惯，并且仅当新任务上的测试平台结果支持该编辑时才予以保留。在推理阶段，使用工具的智能体提出设计方案，并由一个无标签的选择阶段仅依据规范、候选设计及其仿真结果为每个任务交付一个设计。在Qwen3.5-4B上，对16个神经元的编辑使模型本身的VerilogEval v2 pass@1从31.7%提升至55.4%，而完整框架达到了77.6%。在Qwen3.8-27B上，Kolpona将pass@1从73.1%进一步提升。

    arXiv:2611.33973v1 Announce Type: cross  Abstract: Open-weight large language models (LLMs) for register-transfer level (RTL) design are usually improved by fine-tuning on curated Verilog corpora, which needs large sets of verified designs and GPU training. We present Kolpona, an agentic framework that improves such models without gradient training. Its first component is neuron programming, a new form of weight-space programming (WSP) that writes single neurons in closed form into the weights to remove failure habits found on practice tasks, and keeps each edit only when testbench results on fresh tasks favour it. At inference time, tool-using agents propose designs, and a label-free selection stage delivers one design per task from the specification, the candidates and their simulations alone. On Qwen3.5-4B, edits to 16 neurons raise the VerilogEval v2 pass@1 of the model alone from 31.7% to 55.4%, and the full framework reaches 77.6%. On Qwen3.8-27B, Kolpona lifts pass@1 from 73.1% 
    
[^137]: 无需梯度的微调？在冻结编码器上由大语言模型编写的闭式解分类器

    Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder

    [https://arxiv.org/abs/2611.33972](https://arxiv.org/abs/2611.33972)

    该论文提出闭式权重空间编程（WSP）框架，让大语言模型编写程序在冻结编码器上以闭式解拟合分类头，全程无需梯度更新和权重修改，却在多个标签预算下达到甚至超过全量微调、LoRA 等梯度方法的性能。

    

    在冻结编码器上以闭式解计算的分类器能否匹敌梯度微调？我们提出了闭式权重空间编程（WSP）：一个大语言模型（LLM）作为设计助手，编写程序通过多层完形填空式的提示模板读取一个冻结的掩码语言模型，在标注数据上拟合闭式分类头（收缩判别分析、核岭回归以及编码器的标签词先验），并通过交叉拟合从候选分类头池中进行选择。整个过程不计算任何梯度，也不改变编码器的权重。我们在 RoBERTa-large 上实例化了该框架。在 11 个包含 512 至 8,192 个训练标签的开发数据集上，WSP 在每个标签预算下均优于全量微调（FT）和 LoRA（平均高出 0.3 至 0.6 个百分点；预设的非劣性检验通过），且与基于提示的微调（PFT）的差距在 0.5 个百分点以内；这些数据集也参与了方案的构建。在标签更少的情况下，其相对于梯度微调方法的优势进一步扩大。

    arXiv:2611.33972v1 Announce Type: cross  Abstract: Can a classifier computed in closed form on a frozen encoder match gradient fine-tuning? We present closed-form weight-space programming (WSP): a large language model (LLM), used as a design assistant, writes programs that read a frozen masked language model through cloze-style prompt templates at several layers, fit closed-form heads (shrinkage discriminant analysis, kernel ridge regression and the encoder's label-word prior) on labelled data, and select among a pool of candidate heads by cross-fitting. No gradient is computed and no encoder weight changes. We instantiate the framework on RoBERTa-large. On 11 development datasets with 512 to 8,192 training labels, WSP is ahead of full fine-tuning (FT) and LoRA at every label budget (by 0.3 to 0.6 points on average; the registered non-inferiority tests pass) and within 0.5 points of prompt-based fine-tuning (PFT); these datasets also shaped the recipe. With fewer labels its lead over F
    
[^138]: DLaR：面向大语言模型潜在推理的动态随机性控制

    DLaR: Dynamic Stochasticity Control for Latent Reasoning in Large Language Models

    [https://arxiv.org/abs/2611.33962](https://arxiv.org/abs/2611.33962)

    该论文提出DLaR框架，通过在token层面基于置信度-熵融合信号自适应调节扰动强度、在训练层面感知训练进程调优随机性，解决了潜在推理中均匀随机性次优和无约束扰动破坏连贯性的问题，实现探索与稳定性的动态平衡。

    

    潜在推理使大语言模型（LLM）能够在连续嵌入空间中进行推理，然而现有方法受制于确定性推理，无法探索多样化的推理路径。尽管近期方法通过注入随机性来恢复探索能力，我们指出了两个根本性局限：在所有步骤上施加*均匀随机性*是次优的，且无约束的扰动会破坏推理的连贯性。我们提出了面向潜在推理的动态随机性控制（DLaR），这是一个包含两种互补机制的框架，这两种机制在GRPO强化学习框架内以不同的时间粒度运作。在token层面，自适应随机性调制（ASM）基于置信度-熵融合信号动态调整每个潜在步骤的扰动强度，在模型不确定时增强探索，在模型自信时保持稳定性。在训练层面，训练感知的随机性调优……

    arXiv:2611.33962v1 Announce Type: cross  Abstract: Latent reasoning enables LLMs to operate in continuous embedding space, yet existing methods suffer from deterministic inference that precludes exploration of diverse reasoning paths. While recent approaches inject stochasticity to restore exploration, we identify two fundamental limitations: \textit{uniform stochasticity} across all steps is suboptimal, and unconstrained perturbations disrupt reasoning coherence. We propose Dynamic Stochasticity Control for Latent Reasoning (DLaR), a framework with two complementary mechanisms that operate at different temporal granularities within the GRPO reinforcement learning framework. At the token level, Adaptive Stochasticity Modulation (ASM) dynamically adjusts perturbation intensity at each latent step based on a confidence-entropy fusion signal, increasing exploration when the model is uncertain and preserving stability when confident. At the training level, Training-Aware Stochasticity Tuni
    
[^139]: RefMix：扩展端点提议以实现保持多样性的奖励微调

    RefMix: Expanding Endpoint Proposals for Diversity-Preserving Reward Fine-Tuning

    [https://arxiv.org/abs/2611.33957](https://arxiv.org/abs/2611.33957)

    RefMix 通过在早期高噪声步骤借助参考模型注入备选端点候选，打破策略内反馈循环，在保持任务奖励性能的同时显著提升生成结果的视觉多样性。

    

    在线前向匹配通过反复生成、奖励评估和拟合模型产生的端点来对齐基于流的生成模型。然而，这种行为会形成一种策略内（on-policy）反馈循环：随着高奖励模式日益受到青睐，其他备选模式在后续训练池中出现的可能性逐渐降低，从而不断削弱视觉多样性。我们提出了 RefMix，一种参考辅助的端点提议策略，通过向训练池中注入备选候选来解决这一局限，同时保留原有的任务奖励与前向匹配目标。具体而言，RefMix 在早期高噪声步骤中使用参考模型来展现备选的视觉模式，随后由当前的奖励微调模型完成端点生成。在多种最先进的在线前向匹配方法上，RefMix 均能持续提升视觉多样性，同时保持任务奖励性能。

    arXiv:2611.33957v1 Announce Type: cross  Abstract: Online forward matching aligns flow-based generative models by repeatedly generating, rewarding, and fitting model-produced endpoints. However, this behavior creates an on-policy feedback loop: as high-reward modes become increasingly favored, alternative modes become less likely to appear in subsequent training pools, progressively reducing visual diversity. We introduce RefMix, a reference-assisted endpoint proposal strategy that addresses this limitation by injecting alternative candidates into the training pool while retaining the original task reward and forward matching objective. In particular, RefMix uses a reference model during early high-noise steps to expose alternative visual modes, followed by the current reward-tuned model for endpoint completion. Across state-of-the-art online forward matching methods, RefMix consistently improves visual diversity while preserving task reward performance. Controlled inference experiment
    
[^140]: DuoMI：用于循环Transformer中专家重用的双互信息正则化

    DuoMI: Dual Mutual Information Regularization for Expert Reuse in Looped Transformers

    [https://arxiv.org/abs/2611.33952](https://arxiv.org/abs/2611.33952)

    DuoMI通过轮次级边缘互信息和token级条件互信息的双重正则化，在不增加参数和改变稀疏Top-k路由的前提下，调节循环MoE Transformer中跨轮次的专家路由以实现专家参数的高效重用。

    

    我们提出了双互信息正则化（DuoMI），这是一种用于调节循环混合专家Transformer中跨轮次专家路由的方法。当两轮计算共享一个专家池时，它们的路由器可能偏好不同的专家子集，从而限制了单个专家参数在轮次间的重用。即使各轮次的总体专家使用情况相同，我们仍然希望每个token在保留部分专家的同时切换到其他专家。DuoMI同时处理路由的两个方面：它使用轮次级边缘互信息来限制总体专家使用对轮次的依赖，并使用token级条件互信息来惩罚超出目标区间的平均跨轮路由变化。在训练过程中，DuoMI从所有专家的软路由分布中计算可微分的惩罚项。它不增加任何模型参数，并保持稀疏Top-k路由策略不变。

    arXiv:2611.33952v1 Announce Type: cross  Abstract: We introduce Dual Mutual Information Regularization (DuoMI), a method for regulating expert routing across rounds in looped mixture-of-experts (MoE) Transformers. When two rounds of computation share an expert pool, their routers may favor different expert subsets, limiting the reuse of individual expert parameters across rounds. Even when aggregate expert usage is identical across rounds, we would still prefer each token to retain some experts while switching to others. DuoMI addresses both aspects of routing: it uses round-level marginal mutual information to limit the dependence of aggregate expert usage on the round, and token-level conditional mutual information to penalize average cross-round routing changes that fall outside a target interval. During training, DuoMI computes differentiable penalties from soft routing distributions over all experts. It adds no model parameters and leaves sparse Top-$k$ routing strategy unchanged 
    
[^141]: BReD：用于稳定低比特 EMA 优化器状态的块重放抖动方法

    BReD: Block Replay Dithering for Stable Low-Bit EMA Optimizer States

    [https://arxiv.org/abs/2611.33951](https://arxiv.org/abs/2611.33951)

    提出 BReD 方法，通过将确定性种子重放与分块共享抖动值相结合的块重放抖动技术，减少 EMA 优化器状态迭代量化中的舍入偏差并控制舍入方差，从而解决误差累积问题、稳定大规模训练。

    

    优化器状态的低比特存储可以大幅减少大规模训练的内存占用。然而在实践中，这些状态并非只被量化一次：在每个训练步骤中，它们都会被反量化、更新、重新量化并写回。这种迭代式的量化流程可能会破坏训练的稳定性。我们识别出了导致这种不稳定性的一个失效模式：对于通过指数移动平均（EMA）更新的状态，量化误差会被递归地引入后续更新中。因此，有偏误差会随时间不断累积，而高方差误差则会被放大，尤其是当 EMA 衰减因子接近 1 时。为了解决这种误差累积问题，我们提出了 BReD（块重放抖动），一种在量化过程中减少舍入偏差并控制舍入方差的方法。BReD 通过将确定性种子重放与分块共享的抖动值相结合，将经典的减法抖动技术适配到了优化器状态量化中。

    arXiv:2611.33951v1 Announce Type: new  Abstract: Low-bit storage of optimizer states can substantially reduce the memory footprint of large-scale training. In practice, however, these states are not quantized once: at every training step, they are dequantized, updated, requantized, and written back. This iterative quantization pipeline can destabilize training. We identify a failure mode behind this instability: for states updated via exponential moving averages (EMAs), quantization errors are recursively fed into subsequent updates. As a result, biased errors accumulate over time while high-variance errors are amplified, particularly when the EMA decay factor is close to 1. To address this error accumulation, we propose BReD (Block Replay Dithering), a method that reduces rounding bias and controls rounding variance during quantization. BReD adapts classical subtractive dithering to optimizer-state quantization by combining deterministic seed replay with block-wise shared dither value
    
[^142]: 线性因果边界的图形判定准则

    A Graphical Criterion for Linear Causal Bounds

    [https://arxiv.org/abs/2611.33950](https://arxiv.org/abs/2611.33950)

    本文针对每个混杂组件最多含两个隐藏原因的因果图，给出了判断尖锐边界能否通过求解单个线性规划得到的充分必要条件，并实现了Python包partialid以计算高效的方式获得尖锐边界。

    

    在存在未观测混杂因素的情况下，因果效应和反事实的值通常只能通过观测数据进行部分识别。对于离散值变量，在特定条件下，可以通过求解单个线性规划来推导出尖锐边界，这使得许多因果问题在计算上是可处理的。本文针对每个混杂组件最多包含两个隐藏原因的因果图，给出了对于任意（正）输入概率表，尖锐边界等于单个线性规划最优解的充分必要条件。该算法已在一个名为"partialid"的新Python包中实现，允许用户组合假设、查询和数据，以计算高效的方式获得尖锐边界。

    arXiv:2611.33950v1 Announce Type: cross  Abstract: The values of causal effects and counterfactuals are typically only partially identified from observational data in the presence of unobserved confounding. With discretely-valued variables, under certain conditions, sharp bounds can be derived by solving one linear program. This is then computationally tractable for many causal questions. In this paper, for every diagram in which each confounded component has at most two hidden causes, we give necessary and sufficient conditions for the sharp bounds to be equal to the optimum of one linear program for every (positive) input probability table. This algorithm is implemented in a new Python package, called "partialid", that allows users to compose assumptions, queries, and data to obtain sharp bounds in a computationally tractable manner.
    
[^143]: CondUQ：基于分布匹配的校准条件不确定性量化

    CondUQ: Calibrated Conditional Uncertainty Quantification Using Distribution Matching

    [https://arxiv.org/abs/2611.33939](https://arxiv.org/abs/2611.33939)

    该论文提出 condUQ 方法，通过将误差模型条件化于可观测因素（如输入质量和运行条件）进行分布匹配，在不修改甚至黑盒的现有系统之上实现准确可靠的条件不确定性量化。

    

    许多应用，尤其是可靠性和安全性关键的应用，需要对基于模型或学习的测量、估计和预测系统的输出进行准确的不确定性量化。此类系统的误差往往随输入质量和运行条件等因素变化，因此无法用单一的 高斯模型准确描述。尽管高斯混合模型和校准分位数可以近似此类误差分布，但它们的无条件形式无法捕捉误差行为对这些因素的依赖关系。因此，我们提出了 condUQ，一种将误差模型条件化于可观测因素的方法，以实现准确可靠的不确定性量化。condUQ 构建于现有系统之上而无需修改该系统，即使系统是黑盒也能适用。首先，condUQ 将可观测因素映射到一个潜在的质量指数，使得共享相同 QI 的误差服从相同的分布……（摘要在此处被截断）

    arXiv:2611.33939v1 Announce Type: new  Abstract: Many applications, especially reliability- and safety-critical ones, require accurate uncertainty quantification for the outputs of model- or learning-based measurement, estimation, and prediction systems. The errors of such systems often vary with factors such as input quality and operating conditions, and thus cannot be accurately described by a single Gaussian model. Although Gaussian mixture models and calibrated quantiles can approximate such error distributions, their unconditional forms do not capture how the error behavior depends on these factors. Hence, we propose condUQ, an approach that conditions error models on observable factors for enabling accurate and reliable uncertainty quantification. condUQ is built on top of an existing system without modifying it and applies even when the system is a black box. First, condUQ maps the observable factors to a latent quality index (QI) such that errors sharing the same QI follow a co
    
[^144]: InstructTTS-OPD：面向多目标指令跟随文本转语音的在策略蒸馏

    InstructTTS-OPD: On-Policy Distillation for Multi-Objective Instruction-Following TTS

    [https://arxiv.org/abs/2611.33937](https://arxiv.org/abs/2611.33937)

    该论文提出 InstructTTS-OPD，用在策略蒸馏融合多个单目标教师模型，取代单一标量奖励的强化学习，从而解决指令跟随 TTS 中多目标联合优化带来的训练不稳定与目标间跷跷板效应。

    

    指令跟随文本转语音根据自由形式的自然语言指令，以指令所描述的方式合成目标文本，而一个良好的合成结果必须同时在三个方面取得成功：遵循指令、正确发音以及听起来自然。强化学习已成为将此类生成模型进行后训练以对齐人类偏好的标准方法，TTS 系统也越来越多地采用它。当这三个目标在单一标量奖励下联合优化时（通常使用组相对策略优化 GRPO），训练往往不稳定：各目标相互竞争，共享的更新会向最容易改进的目标漂移，从而在它们之间产生跷跷板效应。我们转而采用在策略蒸馏（OPD），通过在学生模型自身的采样结果上匹配多个单目标教师模型的速度场，将它们融合起来。为实现面向 Instruct-TTS 的这一方法，首先需要……

    arXiv:2611.33937v1 Announce Type: cross  Abstract: Instruction-following text-to-speech (Instruct-TTS) renders a target transcript in the manner described by a free-form natural-language instruction, and a good rendition must succeed on three fronts at once: following the instruction, pronouncing the transcript correctly, and sounding natural. Reinforcement learning has become the standard way to post-train such generative models toward human preferences, and TTS systems increasingly adopt it. When the three objectives are optimized together under a single scalar reward, typically with group relative policy optimization (GRPO), training tends to be unstable: the objectives compete, and the shared update drifts toward whichever one is easiest to improve, producing a seesaw among them. We instead adapt on-policy distillation (OPD), which fuses several single-objective teachers by matching their velocity fields on the student's own rollouts. Realizing this for Instruct-TTS first calls for
    
[^145]: 面向可信多视图分类的相关性感知证据融合

    Correlation-Aware Evidential Fusion for Trusted Multi-View Classification

    [https://arxiv.org/abs/2611.33900](https://arxiv.org/abs/2611.33900)

    提出了相关性感知证据融合框架CAFE，通过显式建模视图间的相关性，避免重复计算冗余证据、强化互补信息，从而提升可信多视图分类的性能和不确定性估计的校准质量。

    

    可信多视图分类旨在通过整合来自多个视图的信息，做出准确且可靠的预测。现有方法通常为每个视图估计类别证据和不确定性，并通过聚合规则组合由此产生的观点。然而，这些方法通常将不同视图视为独立的信息源，忽略了视图之间的相关性。高度相关的视图往往会产生相似的证据，将它们视为独立信息源可能会重复计算一致的信息并赋予其不成比例的影响力，同时削弱提供额外信息的互补性证据。因此，这种不考虑相关性的融合不仅可能限制分类性能，还可能产生校准不佳的不确定性估计。为解决这一问题，我们提出了相关性感知证据融合（CAFE）框架，用于可信多视图分类。CAFE通过对……

    arXiv:2611.33900v1 Announce Type: cross  Abstract: Trusted multi-view classification aims to make accurate and reliable predictions by integrating information from multiple views. Existing methods typically estimate class evidence and uncertainty for each view and combine the resulting opinions through aggregation rules. However, they generally treat different views as independent information sources, overlooking the correlations among them. Highly correlated views tend to produce similar evidence, and treating them as independent sources may repeatedly count consistent information and give it disproportionate influence, while weakening complementary evidence that provides additional information. Consequently, such correlation-unaware fusion may not only limit classification performance but also produce miscalibrated uncertainty estimates. To address this issue, we propose Correlation-Aware Fusion of Evidence (CAFE), a framework for trusted multi-view classification. CAFE applies compl
    
[^146]: 循环模型是快速的多词预测器

    Looped Models are Fast Multi-Token Predictors

    [https://arxiv.org/abs/2611.33896](https://arxiv.org/abs/2611.33896)

    提出Floodgates自推测解码方法，在循环语言模型末尾附加掩码词块并通过低秩适应并行起草多个未来词，实现高达7.5倍的无损推理加速。

    

    循环语言模型可以通过调整循环次数来适应预测一个词所需的计算量，而无需按比例缩放唯一的模型参数。这一特性已被用于自推测解码，将浅层循环作为快速起草者，深层循环作为验证者，这种方法之所以有效，是因为起草过程仍然使用完整参数。在本文中，我们证明浅层循环还可以预测多个未来词，并利用这一点来大幅加速循环语言模型的推理。为此，我们提出了Floodgates，一种简单的自推测解码方法，它在序列末尾附加一个掩码词块，并通过低秩适应学习并行起草这些词。在七个模型、五个基准测试、一系列批处理大小以及贪婪解码和采样解码的实验中，Floodgates在大多数情况下是最快的无损解码方法，实现了高达7.5倍的加速。通过分析，我们展示了……（原文在此处截断）

    arXiv:2611.33896v1 Announce Type: cross  Abstract: Looped language models can adapt computation for predicting a token without a proportional rescaling in unique model parameters by adjusting the number of loops. This property has been used for self-speculative decoding, taking shallow loops as fast drafters and deep loops as verifiers, which is effective since drafting still uses full parameters. In this paper, we show that shallow loops can be also predictive of several future tokens and leverage this to drastically accelerate the inference of looped language models. To that end, we present Floodgates, a simple self-speculative decoding method that appends a block of mask tokens as a suffix and learns via low-rank adaptation to draft these tokens in parallel. Across seven models, five benchmarks, a range of batch sizes, and both greedy and sampled decoding, Floodgates is the fastest lossless decoding method in most cases, attaining up to 7.5x acceleration. Through analysis, we show t
    
[^147]: 论模型重编程的可学习性：可达性与重标记视角

    On the Learnability of Model Reprogramming: A Reachability and Relabeling Perspective

    [https://arxiv.org/abs/2611.33889](https://arxiv.org/abs/2611.33889)

    本文从可达性与重标记的视角系统研究了模型重编程包装器假设类的有限样本可学习性，回答了逼近该类最佳预测器所需的目标样本数量以及其最优解与目标贝叶斯风险之间的差距这两个核心理论问题。

    

    模型重编程通过将一个冻结的源模型包装在可学习的结构中使其适应目标任务：包括一个从目标输入空间到源模型输入空间的可学习输入变换，以及一个从源模型输出到目标预测的输出映射。尽管已取得大量实证进展，由此诱导出的包装器假设类 $\mathcal{F}_{\rm MR}$ 的有限样本可学习性仍未得到充分探索。我们针对暴露硬源标签的源模型研究这一问题，并通过包装器结构进行分析：在任何目标样本上，输入变换会选择可到达的源标签，输出映射再对其进行重标记。通过这一视角，我们回答了两个问题：(1) 需要多少目标样本才能逼近 $\mathcal{F}_{\rm MR}$ 中的最佳预测器；(2) $\mathcal{F}_{\rm MR}$ 上的最优解与目标贝叶斯风险之间还存在多大差距。对于问题 (1)，包装器结构对 $\mathcal{F}_{\rm MR}$ 的逐样本复杂度进行了分解（原文摘要在此处截断）。

    arXiv:2611.33889v1 Announce Type: new  Abstract: Model reprogramming adapts a frozen source model to a target task by wrapping it in a learnable input transformation from target input to the source-model input space, and an output mapping from source-model outputs to target predictions. Despite extensive empirical progress, the finite-sample learnability of the induced wrapper hypothesis class $\mathcal{F}_{\rm MR}$ remains underexplored. We address this for source models that expose hard source labels and analyze it through the wrapper structure: on any target samples, the input transformation selects reachable source labels that the output mapping relabels. Through this lens, we answer two questions: (1) how many target samples are needed to approach the best predictor in $\mathcal{F}_{\rm MR}$, and (2) what gap remains between the optimum over $\mathcal{F}_{\rm MR}$ and the target Bayes risk. For (1), the wrapper structure factorizes the sample-wise complexity of $\mathcal{F}_{\rm M
    
[^148]: QK归一化中分离分母所引起的梯度差异

    The Gradient Difference Induced by Detaching the Denominator in QK Normalization

    [https://arxiv.org/abs/2611.33887](https://arxiv.org/abs/2611.33887)

    本文推导了QK归一化中detach分母所引起的参数梯度差异的闭式形式，证明该差异在单个token上仅改变query范数（归一化后无一阶效应），而在共享的投影矩阵 $W_Q$ 上可精确分解为径向、切向和上游三个分量。

    

    QK归一化在计算注意力分数之前，将每个query和key除以其自身的RMS范数。在反向传播中将该分母视为常数（即detach处理）不会改变前向传播的结果，但会改变参数的梯度。我们推导出了该detach规则下的梯度与标准规则下的梯度之间差异的闭式表达形式。对于单个token，这一梯度差异与该token自身的query方向平行；它只改变query的范数，在归一化之后没有一阶效应。然而，query是由所有token共享的投影矩阵 $W_Q$ 计算得到的。因此，$W_Q$ 上的梯度差异等于 $W_Q$ 右乘一个输入的加权二阶矩，并且可以精确分解为三个分量：仅改变 $W_Q$ 范数的径向分量、改变其他token的query方向的切向分量，以及改变上游的上游分量。

    arXiv:2611.33887v1 Announce Type: new  Abstract: QK normalization divides each query and key by its own RMS norm before the attention scores are computed. Treating this denominator as a constant during backpropagation leaves the forward pass unchanged but alters the parameter gradients. We derive a closed form for the difference between the gradient under this detached rule and the gradient under the standard rule. For a single token, this difference is parallel to the token's own query; it changes only the query norm and has no first-order effect after normalization. The queries, however, are computed by a projection matrix $W_Q$ shared across all tokens. The gradient difference on $W_Q$ therefore equals $W_Q$ right-multiplied by a weighted second moment of the inputs, and it decomposes exactly into three components: a radial component that changes only the norm of $W_Q$, a tangential component that changes the query directions of other tokens, and an upstream component that changes t
    
[^149]: 随机Voronoi网络

    Random Voronoi Networks

    [https://arxiv.org/abs/2611.33875](https://arxiv.org/abs/2611.33875)

    随机Voronoi网络是一族结构化的三层MLP，通过采样的Voronoi比较实现最近邻几何的精确或近似表达，并可松弛为可微的软加权，从而将最近邻方法融入可微神经架构。

    

    最近邻方法具有局部性、简单性和可解释性等优点，但其非参数形式导致内存占用大、推理成本高，且难以集成到可微神经网络架构中。我们提出了随机Voronoi网络（RVNs），这是一族结构化的三层多层感知机（MLP），通过采样的Voronoi比较来表达最近邻几何。第一层从所选训练样本之间的仿射成对分隔器计算随机Voronoi特征（RVF）；第二层将这些比较聚合为Voronoi单元或top-k邻域分数；读出层则产生分类或回归预测。我们证明，当包含所有样本及成对比较时，采用阶跃激活的RVN能够精确恢复欧氏k-NN，而采用ReLU激活的RVN为1-NN提供了一种违规分数的实现方式，并可松弛为可微的软加权。通过子采样可以得到具有显式……的近似RVN。

    arXiv:2611.33875v1 Announce Type: cross  Abstract: Nearest-neighbor methods are local, simple, and interpretable, but their non-parametric form makes them memory-intensive, expensive at inference, and difficult to integrate into differentiable neural architectures. We introduce Random Voronoi Networks (RVNs), a family of structured three-layer MLPs that express nearest-neighbor geometry through sampled Voronoi comparisons. The first layer computes Random Voronoi Features (RVF) from affine pairwise separators between selected training samples, the second layer aggregates these comparisons into Voronoi cells or top-k neighborhood scores, and the readout layer produces classification or regression predictions. We show that, when all samples and pairwise comparisons are included, step RVNs recover exact Euclidean k-NN, while ReLU RVNs provide a violation-score realization for 1-NN that can be relaxed into a differentiable soft weighting. Subsampling yields approximate RVNs with explicit co
    
[^150]: 归一化流是能力强大的世界模型

    Normalizing Flows are Capable World Models

    [https://arxiv.org/abs/2611.33872](https://arxiv.org/abs/2611.33872)

    本文提出基于归一化流的世界模型NFWM，利用可逆编码器通过精确似然最大化联合训练表示与动力学，支持潜在空间规划、基于奖励的规划以及目标条件控制。

    

    世界模型使智能体能够预判自身动作的后果，并利用想象中的未来场景进行决策。我们提出了归一化流世界模型（Normalizing-Flow World Model, NFWM），它将一个可逆的观测编码器与一个动作条件下的概率预测器相结合。由于编码器是可逆的，对潜在状态预测出的分布可以定义未来观测的分布。因此，最大化这些观测的精确条件似然能够联合训练表示与动力学。智能体可以在学到的潜在空间中直接进行规划，并通过解码采样出的轨迹来检视可能的未来。在Teleport环境中，这些采样能够捕捉到同一动作的两种可能结果。在Flappy Bird中，仅通过似然训练的世界模型即可支持基于奖励的规划，在全部十个评估游戏中均达到100根管道的成绩。对于目标条件控制任务，一个辅助的逆动力学目标用于塑造所使用的坐标表示（原文在此处被截断）。

    arXiv:2611.33872v1 Announce Type: new  Abstract: A world model lets an agent anticipate the consequences of its actions and use imagined futures to make decisions. We introduce Normalizing-Flow World Model (NFWM), which pairs an invertible observation encoder with an action-conditioned probabilistic predictor. Because the encoder is invertible, the predicted distribution over latent states defines a distribution over future observations. Maximizing the exact conditional likelihood of those observations therefore trains the representation and dynamics jointly. An agent can plan directly in the learned latent space and decode sampled trajectories to inspect possible futures. On Teleport, these samples capture the two possible outcomes of the same action. In Flappy Bird, a world model trained by likelihood alone supports reward-based planning that reaches 100 pipes in all ten evaluated games. For goal-conditioned control, an auxiliary inverse-dynamics objective shapes the coordinates used
    
[^151]: 通过谱目标设计缓解RND中的过度泛化问题

    Mitigating Overgeneralization in RND via Spectral Target Design

    [https://arxiv.org/abs/2611.33850](https://arxiv.org/abs/2611.33850)

    本文从核理论视角证明RND新颖性分数是由目标协方差核与预测器NTK共同决定的跨核残差方差，将过度泛化归结为谱欠激励问题，并提出以带宽受控的随机傅里叶特征（RFF）目标进行谱目标设计来缓解该问题。

    

    随机网络蒸馏（RND）是一种可扩展的强化学习新颖性信号，它基于一个固定的随机目标网络和一个经过训练的预测器。然而，RND可能会过度泛化：预测器会在数据流形之外对随机目标进行外推，导致新颖性评分在不熟悉的输入上崩塌。本文通过核理论的视角分析了这一失效现象。在高斯过程（GP）目标模型和NTK机制下的预测器条件下，我们证明RND新颖性分数是一种跨核残差方差：它由目标协方差核与预测器的NTK共同决定，而非单独由其中任何一个决定。这表明过度泛化本质上是一种谱欠激励现象。基于这一观点，我们将谱目标设计形式化为塑造RND新颖性几何结构的原则。作为一个具体实例，我们用带宽受控的随机傅里叶特征（RFF）目标取代隐式的随机网络目标，将残差信号重新分配到较少被激励的模态上。

    arXiv:2611.33850v1 Announce Type: cross  Abstract: Random Network Distillation (RND) is a scalable novelty signal for RL based on a fixed random target network and a trained predictor. However, it can overgeneralize: the predictor extrapolates the random target off the data manifold, causing novelty scores to collapse on unfamiliar inputs. This paper analyzes this failure through a kernel-theoretic lens. Under a GP target model and an NTK-regime predictor, we show that the RND novelty score is a cross-kernel residual variance: it is governed jointly by the target covariance kernel and the predictor NTK, rather than by either one alone. This identifies overgeneralization as spectral under-excitation. Motivated by this view, we formulate spectral target design as a principle for shaping RND novelty geometry. As a concrete instance, we replace the implicit random-network target with a bandwidth-controlled Random Fourier Feature (RFF) target, redistributing residual signal toward modes les
    
[^152]: ChemDeduce：通过交互式模拟实验对大语言模型进行未知化学物质识别能力的基准测试

    ChemDeduce: Benchmarking LLMs for Unknown Chemical Identification through Interactive Simulated Experiments

    [https://arxiv.org/abs/2611.33848](https://arxiv.org/abs/2611.33848)

    该论文提出了ChemDeduce——一个将有状态模拟实验室中未知化学物质识别建模为序贯决策过程的交互式评估框架，首次系统性地测试了大语言模型通过主动设计实验、迭代推理来发现未知答案的能力。

    

    大语言模型（LLMs）在化学知识和推理方面已展现出强大的性能，但它们能否解决那些必须通过实验才能发现答案的问题仍不清楚。我们通过“未知化学物质识别”任务来研究这一问题，即模型必须主动设计实验、解读观测结果、更新假设，并最终识别出隐藏的物质。我们提出了ChemDeduce，一个通过模拟实验来评估大语言模型的交互式评估框架。ChemDeduce将化学物质识别表述为一个序贯决策过程，并提供了一个具有明确仪器和操作约束的有状态模拟实验室，从而实现对迭代实验推理能力的可扩展、可复现的评估。我们在200种化学物质上对八个前沿大语言模型进行了基准测试，产生了1600个模型-目标组合，并在三次独立重复实验中开展了4800个回合的评估，对识别能力进行了评测。

    arXiv:2611.33848v1 Announce Type: new  Abstract: Large language models (LLMs) have shown strong performance in chemical knowledge and reasoning, yet it remains unclear whether they can solve problems whose answers must be discovered through experimentation. We investigate this question through Unknown Chemical Identification, where a model must actively design experiments, interpret observations, update hypotheses, and ultimately identify a hidden substance. We introduce ChemDeduce, an interactive evaluation framework for assessing LLMs through simulated experiments. ChemDeduce formulates chemical identification as a sequential decision process and provides a stateful simulated laboratory with explicit instrument and operational constraints, enabling scalable and reproducible evaluation of iterative experimental reasoning. We benchmark eight frontier LLMs on 200 chemicals, yielding 1600 model--target pairs and 4800 episodes over three independent repetitions, and evaluate identificatio
    
[^153]: 缺失的根基：大型语言模型中无需图像的几何想象

    The Missing Root: Geometric Imagination Without Images in Large Language Models

    [https://arxiv.org/abs/2611.33841](https://arxiv.org/abs/2611.33841)

    该论文提出将包含位姿与带符号距离形状嵌入的连续物体token嵌入transformer序列，使大型语言模型无需依赖图像即可获得真正的空间几何理解能力。

    

    本文提出了一种赋予大型语言模型空间理解能力的新范式和新方法。标准语言模型仅在离散词语上运行（词语是事物的二次表示，而非连续空间本身），迫使它们只能通过语言来猜测几何信息。大多数多模态方法将空间接地等同于相机图像或视频帧，但空间结构本身并不是一幅图像：它由坐标框架、实体边界以及物体在连续空间中如何运动和相互关联所构成。基于这一区别，我们将结合位姿（位置、方向和尺度）与带符号距离形状嵌入的连续物体token，与词语token一同放置在单一不间断的transformer序列中，使语言和几何能够在物体场景逐步构建和更新的过程中相互关注。控制token可在自然语言词语……之间自动切换输出。

    arXiv:2611.33841v1 Announce Type: cross  Abstract: This paper proposes a new paradigm and method for giving large language models spatial understanding. Standard language models operate exclusively over discrete words (secondary representations of things rather than continuous space itself), forcing them to guess geometry verbally. Whereas most multimodal approaches equate spatial grounding with camera images or video frames, spatial structure itself is not a picture: it consists of coordinate frames, solid boundaries, and how objects move and relate to each other in continuous space. Motivated by this distinction, we place continuous object tokens combining pose (position, orientation, and scale) and signed-distance shape embeddings alongside word tokens inside a single uninterrupted transformer sequence, allowing language and geometry to attend to each other as object scenes are built and updated step by step. Control tokens switch output automatically between natural-language words 
    
[^154]: VoroPlex：一个支持多线程与可微计算的多维Voronoi镶嵌Python框架

    VoroPlex: A Python Framework for Multi-Threaded and Differentiable Voronoi Tessellation

    [https://arxiv.org/abs/2611.33829](https://arxiv.org/abs/2611.33829)

    该论文提出了VoroPlex，一个基于多线程Voro++的开源Python框架，不仅能并行计算二维和三维Voronoi镶嵌，还首次使镶嵌结果可微分，允许借助PyTorch对单元格面积、体积、界面面积等几何量进行基于梯度的优化。

    

    Voronoi镶嵌在计算物理中应用广泛，例如可用于测量粒子系统的局部结构以及生成网格。我们提出了VoroPlex，一个用于计算二维和三维Voronoi镶嵌的开源Python软件包。VoroPlex基于C++库Voro++的多线程版本构建，因此能够并行计算大规模的镶嵌。该软件包还使镶嵌具备了可微性：每个Voronoi顶点都由定义它的生成元进行标记，VoroPlex利用这些标记来计算顶点相对于生成元位置的导数。因此，PyTorch可以计算诸如单元格面积、体积以及界面面积等物理量相对于生成元位置的梯度，从而可以使用标准的基于梯度的方法对这些量进行优化。该软件包还包含用于提取单元格组之间界面的工具等。（注：原文摘要在此处被截断）

    arXiv:2611.33829v1 Announce Type: cross  Abstract: Voronoi tessellations are widely used in computational physics, for example to measure local structure in particle systems and to generate meshes. We present VoroPlex, an open-source Python package for computing Voronoi tessellations in two and three dimensions. VoroPlex is built on the multi-threaded version of the C++ library Voro++, so that large tessellations can be computed in parallel. The package also makes the tessellation differentiable. Each Voronoi vertex is labeled by the generators that define it, and VoroPlex uses these labels to compute the derivative of the vertex with respect to the positions of these generators. PyTorch can therefore compute gradients of quantities such as cell areas, volumes, and interface areas with respect to the generator positions, so that these quantities can be optimized using standard gradient-based methods. The package also includes tools for extracting interfaces between groups of cells and 
    
[^155]: 面向人形机器人重箱搬运的全身枢轴行走步态学习

    Learning a Whole-Body Pivot-Walking Gait for Heavy-Box Transport with a Humanoid

    [https://arxiv.org/abs/2611.33828](https://arxiv.org/abs/2611.33828)

    本文提出单一强化学习策略联合协调人形机器人的迈步、平衡控制与物体操作，实现枢轴行走搬运重箱，并在真实机器人上成功将箱子向前搬运超过1米。

    

    通过枢轴行走，人类可以搬运那些重到无法直接举起的物体。以往的人形机器人研究依赖于规划好的动作序列，其中物体操作和迈步是分开执行的，并且在连续旋转之间物体需要被停稳。然而，很少有研究为人形机器人的枢轴行走开发出单一的学习型控制器。本文提出了一种单一的强化学习策略，该策略联合协调迈步、平衡控制和物体操作。该策略通过鼓励交替枢轴运动的奖励，以及根据物体位移推进的相位参考进行训练。在仿真中，我们验证了人形机器人能够枢转箱子。为了部署到实际硬件上，我们将策略微调为不依赖物体信息的版本，该版本遵循周期性时钟信号而非进度参考。在真实机器人上，所部署的策略在所有实验中都将箱子向前搬运了超过1米。

    arXiv:2611.33828v1 Announce Type: cross  Abstract: Through pivot walking, humans can transport objects that are too heavy to lift. Previous humanoid studies have relied on planned sequences in which object manipulation and stepping are executed separately and the object is brought to rest between successive rotations. However, few studies have developed a single learned controller for humanoid pivot walking. In this paper, we propose a single reinforcement-learning policy that jointly coordinates stepping, balance control, and object manipulation. The policy is trained using rewards that promote alternating pivot motions and a phase reference that progresses according to the displacement of the object. In simulation, we confirmed that the humanoid can pivot a box. For deployment, we fine-tuned the policy into an object-blind version that follows a periodic clock instead of the progress reference. On the real robot, the deployed policy transported the box forward by more than 1 m in all
    
[^156]: FEAST：面向无任务在线策略学习的公平高效自适应采样

    FEAST: Fair and Efficient Adaptive Sampling for Task-Free On-Policy Learning

    [https://arxiv.org/abs/2611.33827](https://arxiv.org/abs/2611.33827)

    论文提出了配额-效率定律（QEL），揭示在线策略学习中分配历史会影响学习效率，并据此引入 FEAST 框架，通过将机会分配与计算分配分离，实现无任务场景下公平而高效的自适应采样。

    

    从自生成经验中训练的语言智能体必须在异构能力之间分配有限的 rollout 预算。一种常见策略是优先考虑当前学习信号较强的能力，但在在线策略学习中，这些信号是内生的：过去的分配会改变策略，进而改变未来分配可用的证据。这可能形成一个反馈循环，使训练不足的能力获得越来越少的改进机会。我们通过配额-效率定律（Quota-Efficiency Law, QEL）将这一效应形式化，该定律表明，即使在各能力特定曝光量匹配的情况下，分配历史仍可能改变学习效率。受此启发，我们引入了 FEAST，一个无任务的自适应采样框架，它将机会分配与计算分配分离开来。FEAST 在保留各能力学习历史的同时动态发现能力聚类，并在各能力间维持持续的采样机会……

    arXiv:2611.33827v1 Announce Type: cross  Abstract: Language agents trained from self-generated experience must allocate a limited rollout budget across heterogeneous capabilities. A common strategy is to prioritize capabilities with strong current learning signals, but in on-policy learning these signals are endogenous: past allocation changes the policy and therefore the evidence available for future allocation. This can create a feedback loop in which under-trained capabilities receive increasingly fewer opportunities to improve. We formalize this effect through the Quota-Efficiency Law (QEL), which shows that allocation history can change learning efficiency even under matched capability-specific exposure. Motivated by this, we introduce FEAST, a task-free adaptive sampling framework that separates opportunity allocation from compute allocation. FEAST dynamically discovers capability clusters while preserving their learning histories, maintains continued sampling opportunities acros
    
[^157]: 论Grokking（顿悟）现象与泛化的动力学

    On Grokking and the Dynamics of Generalization

    [https://arxiv.org/abs/2611.33825](https://arxiv.org/abs/2611.33825)

    本文扩展了grokking的信号通道理论，通过将训练耗散与评估迁移同总体风险联系起来，导出了任务可见性的定量界、迁移信号降低总体风险的充要条件，以及同一训练轨迹在不同评估分布下泛化成败的充分条件。

    

    我们扩展了Litman和Guo提出的grokking信号通道理论，将训练耗散与评估迁移同总体风险联系起来。对于具有时变核的预条件梯度流，我们区分了评估可见性受限的两种来源，即累积耗散过小与迁移能力薄弱。我们得到了任务可见性的定量界，并证明了其在信号过滤和评估分布变化下的稳定性。我们描述了任务方向如何通过谱激活或传输进入有效信号通道。我们建立了迁移信号能够降低平方总体风险的充要条件，并在间隔假设下推导出分类误差界。这些结果给出了同一训练轨迹在一种评估分布下能够泛化而在另一种评估分布下失败的充分条件。我们证明了谱上的（原文摘要在此处截断）

    arXiv:2611.33825v1 Announce Type: new  Abstract: We extend the signal-channel theory of grokking introduced by Litman and Guo by relating training dissipation and evaluation transfer to population risk. For preconditioned gradient flows with time-dependent kernels, we distinguish two sources of limited evaluation visibility, namely small cumulative dissipation and weak transfer. We obtain quantitative bounds for task visibility and establish their stability under signal filtering and changes of evaluation distribution. We describe how task directions enter the effective signal channel through spectral activation or transport. We establish necessary and sufficient conditions for the transferred signal to decrease squared population risk and derive classification error bounds under margin assumptions. These results yield sufficient conditions for the same training trajectory to generalize under one evaluation distribution while failing under another. We show that differences in spectral 
    
[^158]: 掩码扩散语言模型中“依赖就绪”解码的陷阱

    Pitfalls of Dependency-Ready Decoding for Masked Diffusion Language Models

    [https://arxiv.org/abs/2611.33823](https://arxiv.org/abs/2611.33823)

    本研究揭示了掩码扩散模型中“依赖就绪”解码策略的局限性：当该策略优于基于置信度或未来效用的基线方法时，其解码顺序实际上会退化成从左到右的顺序解码。

    

    掩码扩散模型（MDMs）在数独、行程规划和Countdown等结构化推理任务上展现出强大的性能。MDM引入了token去掩码顺序上的自由度，为了利用这种自由度获得更好的准确率，早期的解码策略依赖于置信度信号或解码某个token所产生的下游影响。然而，简单的例子似乎表明，在去掩码某个token之前先解决其依赖关系应当是很重要的。因此，我们从实践和理论两个角度检验了这一原则（我们称之为“依赖就绪解码”）是否有助于提升准确率。我们研究了一个通过注意力机制实现依赖就绪的近期解码器，总体上发现：当它比基于置信度或未来效用的基线表现更好时，其解码顺序会坍缩为从左到右的顺序解码。在理论分析方面，我们建立了一个……的定义（摘要在此处截断）

    arXiv:2611.33823v1 Announce Type: cross  Abstract: Masked diffusion models (MDMs) have shown strong performance on structured reasoning tasks such as Sudoku, Trip Planning, and Countdown. MDMs introduce a freedom in the order in which tokens may be unmasked, and to exercise this freedom for good accuracy, early decoding strategies relied on signals of confidence or the downstream impact of decoding a token. However, simple examples seem to illustrate that resolving a token's dependencies before unmasking it should be important. Therefore, we examine whether this principle, which we call dependency-ready decoding, would contribute to better accuracy, from a practical and a theoretical perspective. We examine a recent decoder that implements dependency-readiness through attention, and generally find that when it performs better than confidence-based or future utility baselines, its decoding order collapses to left-to-right decoding. For theoretical analysis, we establish a definition of 
    
[^159]: MetaDecomp：基于元学习程序记忆的自适应任务分解

    MetaDecomp: Adaptive Task Decomposition via Meta-Learned Procedural Memory

    [https://arxiv.org/abs/2611.33822](https://arxiv.org/abs/2611.33822)

    MetaDecomp提出了一种上下文元学习框架，其外循环从成功执行中构建以抽象状态和有向无环图组织的可重用程序记忆，内循环则在测试时结合记忆中的程序片段与环境反馈，自适应地细化并修订任务分解。

    

    在长程任务中重用过往经验不仅仅是存储成功的程序：语言智能体必须区分哪些程序适用，以及随着执行进展还有哪些工作待完成。我们提出了MetaDecomp，一个用于自适应任务分解的上下文元学习框架，它联合学习任务相关的状态空间和可重用程序。其外循环从成功的训练执行中构建程序记忆，将任务进度表示为抽象状态，并将状态之间的程序表示为操作与依赖关系构成的有向无环图。随着学习到新的进度区分，MetaDecomp会重新审视历史执行证据，以重新评估状态分配并重组其所支持的程序。在测试时记忆保持固定的情况下，内循环细化初始分解并根据环境反馈对其进行修订，将适用的程序片段与记忆中缺失的任务特定步骤相结合。

    arXiv:2611.33822v1 Announce Type: new  Abstract: Reusing past experience in long-horizon tasks requires more than storing successful procedures: Language agents must distinguish which procedures apply and what work remains as execution progresses. We introduce MetaDecomp, an in-context meta-learning framework for adaptive task decomposition that jointly learns a task-relative state space and reusable procedures. Its outer loop builds procedural memory from successful training executions, representing task progress as abstract states and procedures between states as directed acyclic graphs of operations and dependencies. As new progress distinctions are learned, MetaDecomp revisits historical execution evidence to reassess state assignments and reorganize the procedures they support. With memory fixed at test time, the inner loop refines an initial decomposition and revises it from environmental feedback, combining applicable procedure fragments with task-specific steps missing from mem
    
[^160]: KnitNet：在推理时学习搜索以实现精确的RNA逆向设计

    KnitNet: Learning to Search at Inference Time for Exact RNA Inverse Design

    [https://arxiv.org/abs/2611.33812](https://arxiv.org/abs/2611.33812)

    KnitNet是一个推理时扩展框架，通过从验证器反馈中学习核苷酸间的全局依赖关系并自适应调整联合修复策略，实现了RNA逆向设计中精确结构匹配的高效搜索。

    

    RNA逆向设计旨在寻找能够折叠成指定目标结构的核苷酸序列。现有方法依赖于学习到的生成器，但这些生成器可能无法找到精确的结构匹配，尤其是在超出其训练分布的情况下；或者需要在庞大且多模态的序列空间上进行代价高昂的搜索。由于核苷酸之间存在全局耦合的相互作用，搜索变得更加复杂——修正一个配对错误可能会在固定修复范围之外引发新的错误。为了解决这些问题，我们提出了KnitNet，这是一个推理时扩展（inference-time scaling）框架，它从验证器的反馈中学习来精炼序列，以实现精确的折叠回匹配。从为目标二级结构生成的候选序列出发，KnitNet利用折叠不匹配的结果来推断核苷酸之间的依赖关系，并自适应地调整其联合修复策略，同时一个在线代理模型学习预测哪些编辑操作是有希望的。这使得搜索过程逐渐变得更加信息化且计算高效。至关重要的是，KnitNet能够跨RNA...（原文摘要此处截断）

    arXiv:2611.33812v1 Announce Type: cross  Abstract: RNA inverse design seeks nucleotide sequences that fold into specified target structures. Existing methods rely on learned generators that can miss exact structural matches, especially beyond their training distributions, or on costly search over a vast, multimodal sequence space. Search is further complicated by globally coupled nucleotide interactions, as correcting one pairing error can create new errors beyond a fixed repair scope. To address this, we introduce KnitNet, an inference-time scaling framework that learns from verifier feedback to refine sequences for exact fold-back matching. Starting from candidates generated for a target secondary structure, KnitNet uses mismatched folds to infer nucleotide dependencies and adapt its joint repair strategy, while an online surrogate learns to predict which edits are promising. This makes search progressively more informed and compute-efficient. Crucially, KnitNet generalizes across RN
    
[^161]: QPAT：一种用于量子辅助训练与经典推理的通用量子机器学习框架

    QPAT: A Universal Quantum Machine Learning Framework for Quantum-Assisted Training and Classical Inference

    [https://arxiv.org/abs/2611.33806](https://arxiv.org/abs/2611.33806)

    QPAT提出了一种通用量子机器学习框架，利用量子线路玻恩机（QCBM）作为可学习的生成式先验来辅助经典模型训练，实现“量子辅助训练+经典推理”的混合范式，既借助量子模型的归纳偏置克服深度学习在有限数据下的过拟合问题，又规避了NISQ时代量子推理延迟等部署瓶颈。

    

    深度学习模型在有限数据情况下常常面临尖锐局部最优和过拟合的困境，并且捕获高阶复杂概率分布的能力有限。量子机器学习（QML）模型通过在指数级大的希尔伯特空间中运行，能够高效表示此类分布并为学习提供有利的归纳偏置，为克服这些局限性提供了一条有前景的途径。然而，在噪声中等规模量子（NISQ）时代，其实际部署受到梯度耦合、贫瘠高原、数据加载瓶颈和高推理延迟的严重阻碍。为了连接经典人工智能与量子机器学习，我们提出了QPAT（QCBM先验辅助训练），这是一个通用框架，它采用量子线路玻恩机（QCBM）作为可学习的生成式先验，在经典模型训练过程中提供与任务对齐的结构化归纳偏置。该框架可以无缝集成……

    arXiv:2611.33806v1 Announce Type: cross  Abstract: Deep learning models often struggle with sharp local optima and overfitting in limited-data regimes, and exhibit limited capacity for capturing high-order, complex probability distributions. Quantum Machine Learning (QML) models, by operating in exponentially large Hilbert spaces, can efficiently represent such distributions and provide favorable inductive biases for learning, offering a promising pathway to overcome these limitations. However, their practical deployment in the Noisy Intermediate-Scale Quantum (NISQ) era is severely hindered by gradient coupling, barren plateaus, data-loading bottlenecks, and high inference latency. To bridge classical AI and QML, we propose QPAT (QCBM-Prior-Assisted Training), a universal framework that employs a Quantum Circuit Born Machine (QCBM) as a learnable generative prior to provide a task-aligned structured inductive bias during classical model training. The framework can be seamlessly integr
    
[^162]: InterGenEHR$^+$：一个基于自然语言条件的纵向患者轨迹生成交互式框架

    InterGenEHR$^+$: An Interactive Framework for Natural Language-Conditioned Generation of Longitudinal Patient Trajectories

    [https://arxiv.org/abs/2611.33788](https://arxiv.org/abs/2611.33788)

    提出了交互式框架InterGenEHR$^+$，利用大语言模型解析临床需求，并通过生成模型基于自然语言条件合成电子健康记录和纵向患者轨迹，用于机器学习模型的场景化评估。

    

    目的：开发并验证InterGenEHR$^+$，这是一个交互式框架，用于基于自然语言条件生成合成的表格电子健康记录（EHR）和纵向轨迹，以支持机器学习模型的基于场景的评估。方法：一个可选的大语言模型（LLM）解析器将临床请求转换为结构化约束。条件变分自编码器和扩散模型用于生成单时间点记录，而扩散模型用于生成纵向轨迹。我们使用包含874名接受静脉-静脉体外膜肺氧合（VV-ECMO）治疗患者的多机构队列，对解析性能、分布保真度、插补、临床约束依从性、死亡率预测和临床合理性进行了评估。结果：使用少样本提示的Gemma 3 4B实现了最高的解析性能，且虚假槽位率较低。IGE-TabDDPM和IGE-PMVAE在单（摘要在此处截断）

    arXiv:2611.33788v1 Announce Type: new  Abstract: Objective: To develop and validate InterGenEHR$^+$, an interactive framework for natural language-conditioned generation of synthetic tabular electronic health records (EHRs) and longitudinal trajectories for scenario-based evaluation of machine learning models.   Methods: An optional large language model (LLM) parser converts clinical requests into structured constraints. Conditional variational autoencoders and diffusion models generate single-time-point records, while diffusion models generate longitudinal trajectories. We evaluate parsing, distributional fidelity, imputation, clinical-constraint adherence, mortality prediction, and clinical plausibility using a multi-institutional cohort of 874 patients receiving venovenous extracorporeal membrane oxygenation (VV-ECMO).   Results: Gemma 3 4B with few-shot prompting achieves the highest parsing performance with a low spurious slot rate. IGE-TabDDPM and IGE-PMVAE perform well in single
    
[^163]: 基于随机香农熵准则的RIS辅助毫米波大规模MIMO鲁棒自适应信道估计

    Robust Adaptive Channel Estimation for RIS-Assisted mmWave Massive MIMO via a Stochastic Shannon Entropy Criterion

    [https://arxiv.org/abs/2611.33773](https://arxiv.org/abs/2611.33773)

    本文提出了一种将双窗口核香农熵准则与重加权零吸引ℓ1惩罚相结合的鲁棒稀疏自适应算法RZA-SSE，用于RIS辅助毫米波大规模MIMO系统的角度域信道估计，能够有效抵抗非高斯脉冲干扰并给出了完整的理论稳定性分析。

    

    在可重构智能表面（RIS）辅助的毫米波大规模多输入多输出（MIMO）系统中，精确的信道估计仍然面临挑战，其原因在于RIS的无源特性、级联信道的高维度以及非高斯脉冲干扰的存在。本文提出了一种鲁棒的稀疏感知自适应估计器，称为重加权零吸引随机香农熵（RZA-SSE），用于角度域信道估计。该算法将双窗口基于核的香农熵准则与范数归一化的ℓ1惩罚相结合。数据驱动项和非光滑惩罚项均采用精确的Wirtinger微积分进行求导。文中推导了精确的权重误差递推关系和均方能量关系，并通过联合系数-惩罚状态表示分析了局部稳定性，同时给出了近似的稳态均方偏差（MSD）表达式。

    arXiv:2611.33773v1 Announce Type: cross  Abstract: Accurate channel estimation in reconfigurable intelligent surface (RIS)-assisted millimeter-wave (mmWave) massive multiple-input multiple-output (MIMO) systems remains challenging due to the passive nature of the RIS, the high dimensionality of the cascaded channel, and the presence of non-Gaussian impulsive disturbances. This paper proposes a robust sparsity-aware adaptive estimator, termed reweighted zero-attracting stochastic Shannon entropy (RZA-SSE), for angular-domain channel estimation. The algorithm combines a double-window kernel-based Shannon entropy criterion with a norm-normalized \(\ell_1\) penalty. Both the data-driven term and the nonsmooth penalty are differentiated using exact Wirtinger calculus. An exact weight-error recursion and a mean-square energy relation are derived. Local stability is analyzed via a joint coefficient--penalty state representation, and an approximate steady-state mean-square deviation (MSD) expr
    
[^164]: RHOPE：利用大语言模型增强的复杂上下文信息自适应实验设计

    RHOPE: LLM-Augmented Adaptive Experimental Design with Complex Contextual Information

    [https://arxiv.org/abs/2611.33767](https://arxiv.org/abs/2611.33767)

    该论文提出RHOPE框架，利用大语言模型将复杂上下文信息转化为代理结果，将有限批次自适应实验设计形式化为后验马尔可夫决策过程，并通过样本增强与决策前预观察代理结果两种方式提升估计精度和分配决策效果。

    

    实验设计将有限的测量资源分配到候选处理方案上，以学习其期望结果。自适应设计会随着证据的积累不断优化分配策略，从而提升估计精度和处理方案的选择效果。上下文信息可以为这些决策提供依据，但如何利用非结构化或高维度的上下文信息仍然具有挑战性。我们使用大语言模型（LLM）将复杂的上下文信息转化为代理结果，并提出了面向预测驱动实验的残差视野优化方法（RHOPE）——一个有原则的贝叶斯自适应实验框架，它将真实结果与LLM生成的代理结果相结合，并将有限批次的设计问题形式化为后验马尔可夫决策过程（MDP）。我们的方法通过两种方式利用代理结果：样本增强可以提高估计精度，而在每次分配决策之前观察代理结果则能提供前瞻性信息。合成实验和半合成……

    arXiv:2611.33767v1 Announce Type: cross  Abstract: Experimental design allocates limited measurement effort across candidate treatments to learn their expected outcomes. Adaptive designs refine allocations as evidence accumulates, improving estimation and treatment selection. Contextual information can inform these decisions, but exploiting unstructured or high-dimensional context remains challenging. We use large language models (LLMs) to transform complex context into proxy outcomes and propose Residual Horizon Optimization for Prediction-Powered Experimentation (RHOPE), a principled Bayesian adaptive experimentation framework that combines true outcomes with LLM-generated proxy outcomes and formulates finite-batch design as a posterior Markov decision process (MDP). Our methodology uses proxy outcomes in two ways: sample augmentation improves estimation precision, while observing proxies before each allocation decision provides advance information. Synthetic experiments and semi-syn
    
[^165]: 不要浪费种群：面向组合优化的退火后精细化方法

    Don't Waste the Population: Post-Anneal Refinement for Combinatorial Optimization

    [https://arxiv.org/abs/2611.33766](https://arxiv.org/abs/2611.33766)

    本文提出一种无需训练的确定性后处理方法 I-PRP，通过在并行退火产生的候选解之间探索成本引导的路径重连并迭代融入新改进，在不修改退火过程的前提下发掘种群多样性，从理论上保证不劣于传统单解精细化且有限收敛，并在 MaxCut、MaxClique 等组合优化基准上发现更优解。

    

    并行退火算法能够生成多样化的候选解；然而，传统的后处理通常只对其中最优的解进行精细化，导致种群的很大一部分潜力未被挖掘。我们提出了迭代路径重连精细化，这是一种确定性、无需训练的方法，它利用这种多样性来发现更优的解，而无需修改底层的退火过程。I-PRP 选择有潜力的候选解，在局部优化的解之间探索由成本引导的路径，并对中间状态进行精细化以发现更好的局部极小值。新发现的改进会被迭代地融入种群，从而支持进一步的探索。我们从理论上证明，在相同种群上，I-PRP 相比传统的单候选解精细化方法不会降低解的质量，并且能在有限次迭代内收敛。在包括 MaxCut、MaxClique 和图着色等具有挑战性的组合优化基准测试上的实验表明……（摘要在此处截断）

    arXiv:2611.33766v1 Announce Type: new  Abstract: Parallel annealers generate diverse candidate solutions; yet, conventional post-processing typically refines only the best, leaving much of the population's potential untapped. We introduce Iterative Path-Relinking Polish (I-PRP), a deterministic, training-free method that exploits this diversity to discover better solutions without modifying the underlying annealing process. I-PRP selects promising candidates, explores cost-guided paths between locally optimized solutions, and refines intermediate states to uncover better local minima. Newly discovered improvements are iteratively incorporated into the population, enabling further exploration. We theoretically guarantee that I-PRP never degrades solution quality relative to conventional single-candidate polishing on the same population and converges in finitely many iterations. Experiments across challenging combinatorial optimization benchmarks, including MaxCut, MaxClique, and graph c
    
[^166]: 通过通用压缩理解双下降现象

    Understanding Double Descent through Universal Compression

    [https://arxiv.org/abs/2611.33759](https://arxiv.org/abs/2611.33759)

    该论文提出以通用压缩的全新理论视角解释深度学习中的双下降现象，将最大似然超额测试损失的上界分解为近似与偏移项、极小极大批次遗憾、先验代价和MLE与贝叶斯差距四个可解释的组成部分。

    

    许多深度学习模型表现出双下降现象，即测试损失在插值阈值附近出现急剧峰值，然后随着模型规模的进一步增大而再次下降，这与经典偏差-方差权衡所预测的U形曲线相反。双下降现象已在简单的经典模型中被发现并得到解释，但尽管经过了大量实证研究，其在现代深度学习中仍未被充分理解。我们通过通用压缩这一全新视角来解释双下降现象。具体而言，我们将最大似然的超额测试损失的上界表示为四项之和：近似与偏移项、极小极大批次遗憾、先验代价，以及最大似然估计（MLE）与贝叶斯方法之间的差距。在精确后验采样条件下，我们进一步证明，在设定良好的情况下，预期的MLE与集成方法之间的差距可对MLE与贝叶斯方法之间的差距进行下界化。在我们的实验中，采用高斯先验的SGLD对贝叶斯预测器进行近似，与SGD不同，……

    arXiv:2611.33759v1 Announce Type: cross  Abstract: Many deep learning models exhibit a double-descent phenomenon, where test loss exhibits a sharp peak near the interpolation threshold and then descends a second time as model size grows further, contrary to the U-shape predicted by the classical bias--variance tradeoff. The double-descent phenomenon has been found and explained in toy classical models, but is not well-understood in modern deep learning despite extensive empirical study. We interpret double descent through the novel lens of universal compression. Specifically, we upper bound the maximum-likelihood excess test loss as a sum of four terms: an approximation-and-shift term, a minimax batch regret, a prior cost, and an MLE-vs-Bayes gap. Under exact posterior sampling, we further show that the expected MLE-vs-ensemble gap lower-bounds the MLE-vs-Bayes gap in the well-specified setting. In our experiments, a Gaussian-prior SGLD approximation to the Bayes predictor, unlike SGD,
    
[^167]: OrbACE：轨道与原子簇展开

    OrbACE: Orbital and Atomic Cluster Expansion

    [https://arxiv.org/abs/2611.33743](https://arxiv.org/abs/2611.33743)

    OrbACE框架通过在E(3)等变簇展开骨干网络中引入廉价的电子结构描述符，在有限训练数据下将能量误差降低至MACE的三分之一、力误差大约减半，显著提升了机器学习原子间势的精度。

    

    准确的电子结构参考数据成本高昂，这促使人们开发能够从有限训练集中有效学习的原子间势。我们提出了OrbACE（轨道与原子簇展开），这是一个用廉价的电子结构描述符来补充E(3)等变簇展开骨干网络的框架。在ωB97X/6-31G*水平的ANI-1x数据集上，其原子中心变体OrbACE-A在10,000个构型上训练、在442,000个留出构型上评估时，直接学习达到的能量和力平均绝对误差分别为3.16 meV/原子和76 meV/埃，而在相同骨干设置下MACE的误差为9.61 meV/原子和146 meV/埃。这将能量误差降低了三倍，并将力误差大约减半。其能量误差也低于最佳delta校正MACE基线的5.00 meV/原子，尽管其力误差略高。将OrbACE-A与H（摘要在此处被截断）

    arXiv:2611.33743v1 Announce Type: cross  Abstract: Accurate electronic-structure reference data are expensive, motivating interatomic potentials that learn effectively from limited training sets. We introduce OrbACE (Orbital and Atomic Cluster Expansion), a framework that supplements an E(3)-equivariant cluster-expansion backbone with inexpensive electronic-structure descriptors. On ANI-1x at the $\omega$B97X/6-31G* level, its atom-centered variant, OrbACE-A, trained on 10,000 configurations and evaluated on 442,000 held-out configurations, achieves energy and force mean absolute errors of 3.16 meV/atom and 76 meV/angstrom in direct learning, compared with 9.61 meV/atom and 146 meV/angstrom for MACE with matched backbone settings. This reduces the energy error by a factor of three and approximately halves the force error. Its energy error is also below that of the best delta-corrected MACE baseline, 5.00 meV/atom, although its force error is slightly higher. Combining OrbACE-A with a H
    
[^168]: FlashProve-Next：面向生产级大语言模型 API 的可验证推理

    FlashProve-Next: Verifiable Inference for Production LLM APIs

    [https://arxiv.org/abs/2611.33741](https://arxiv.org/abs/2611.33741)

    FlashProve-Next 在完全不修改原生推理服务的前提下，为闭源大语言模型 API 的单个响应生成零知识可验证证明，使客户端能够验证服务商是否真实运行了所宣称的模型、精度和推理预算，同时不泄露权重、隐藏提示词等机密信息。

    

    arXiv:2611.33741v1 公告类型：cross。闭源权重大语言模型 API 的客户端无法验证服务提供商是否真的运行了所宣称的模型、精度、隐藏提示词和推理预算。现有方法要么信任硬件或第三方，要么以高昂代价对整数代理模型的每一层重新证明，同时偏离了原生的浮点推理服务。FlashProve-Next 为单个大语言模型 API 响应生成可验证的、经零知识掩码的证明，无需修改原生推理服务（包括 bf16/fp8/mxfp4、随机采样、FlashAttention、CUDA 图、批处理、前缀缓存），也不会泄露权重、隐藏提示词、隐藏推理过程或激活值（稠密配置文件固定粗略几何信息；跨架构隐藏被推迟处理）。一种不消耗的 Reed-Solomon 权重承诺（model_id）通过隐藏列开启方式在证明内部打开，同时针对每个算子的局部舍入契约消除了整数代理模型和拒绝位泄露问题。成本加权的抽检机制支持非交互式……（原文摘要在此处截断）

    arXiv:2611.33741v1 Announce Type: cross  Abstract: Clients of closed-weight LLM APIs cannot verify whether a provider ran the advertised model, precision, hidden prompt, and reasoning budget. Existing approaches either trust hardware or third parties, or re-prove every layer of an integer surrogate at high cost while diverging from native floating-point serving.   FlashProve-Next produces verifiable, zero-knowledge-masked proofs for individual LLM API responses without modifying native serving (bf16/fp8/mxfp4, random sampling, FlashAttention, CUDA graphs, batching, prefix caching) and without disclosing weights, hidden prompts, hidden reasoning, or activations (dense profiles fix coarse geometry; cross-architecture hiding is deferred). A non-depleting Reed-Solomon weight commitment model_id opens inside proofs via hidden column openings, while local per-operator rounding contracts remove integer surrogates and the refusal-bit leak. Cost-weighted spot-checking supports non-interactive p
    
[^169]: CovNovBench：用于评估协变量新颖性情形下时间序列基础模型的基准测试

    CovNovBench: A Benchmark for Evaluating Time Series Foundation Models under Covariate Novelty

    [https://arxiv.org/abs/2611.33732](https://arxiv.org/abs/2611.33732)

    本文提出了CovNovBench基准，首次系统性地评估时间序列基础模型在协变量新颖性（即未来处理协变量超出历史观测范围，如新定价策略或新治疗方案）情形下的预测能力。

    

    协变量感知的时间序列基础模型（TSFM）将其预测条件化于在未来时段内提供的额外输入变量，即协变量。这一能力使其在干预式预测中极具吸引力，即在计划行动下（例如新产品定价策略或新的患者治疗方案）预测目标变量。我们将编码此类行动的协变量称为处理协变量。然而，当前的协变量感知TSFM是在观测到的模式上训练的，当未来的处理协变量超出历史所见的范围时，其行为尚不明确。例如，数值型处理变量可能超出其历史取值范围，类别型处理变量可能采用先前从未见过的策略。我们将这种在实践中十分常见但未在现有基准测试中得到系统性体现的情形称为协变量新颖性，并为此给出了操作性定义。

    arXiv:2611.33732v1 Announce Type: new  Abstract: Covariate-aware time-series foundation models (TSFMs) condition their forecasts on additional input variables, called covariates, that are supplied over the future horizon. This capability makes them appealing for interventional forecasting, in which a target is predicted under a planned action such as a new product pricing policy or a new patient treatment schedule. We refer to covariates that encode such actions as treatment covariates. Current covariate-aware TSFMs, however, are trained on observed patterns, and their behavior is unclear when future treatment covariates fall outside the range seen in the history. Examples include a numerical treatment that leaves its historical range and a categorical treatment that takes a previously unseen policy. We call this practically common setting covariate novelty, and note that it is not systematically represented in existing benchmarks. We introduce an operational definition of covariate no
    
[^170]: 理解无分类器引导下扩散模型中的模式坍缩

    Understanding Mode Collapse in Diffusion Models under Classifier-Free Guidance

    [https://arxiv.org/abs/2611.33719](https://arxiv.org/abs/2611.33719)

    本文提出一个条件分布为均值位于椭球面上的高斯混合、无条件分布为近似该混合的单一高斯的分析模型，揭示了无分类器引导下模式坍缩的根源在于条件分布与无条件分布表示能力的不匹配。

    

    无分类器引导在扩散模型和流模型中的应用已被证明对其经验成功至关重要，因为它能够可控地生成更低温、更高保真度的样本。尽管被广泛使用，无分类器引导在实验和理论上仍然缺乏充分理解，原因在于难以针对任意归一化流求解相关的输运方程。在本文中，我们观察到无分类器引导的效果源于条件分布与无条件分布之间表示能力的不匹配，其中条件分布表现出更高的复杂性和保真度。我们提出了一个分析模型来刻画这一观察：条件分布是一个其均值位于椭球面上的高斯混合分布，而无条件分布是一个近似该混合分布的单一高斯分布。由该混合分布产生的流……

    arXiv:2611.33719v1 Announce Type: new  Abstract: The use of classifier-free guidance in diffusion and flow models has proven critical to their empirical success, as it allows for the controllable generation of lower-temperature and higher-fidelity samples. Despite its widespread use, classifier-free guidance remains both empirically and theoretically poorly understood, owing to the difficulty of solving the associated transport equations for arbitrary normalizing flows. In this paper, we observe that the effects of classifier-free guidance arise from a mismatch between the representation capabilities of the conditional and unconditional distributions, wherein the conditional distribution exhibits higher complexity and fidelity. We propose an analytical model to capture this observation: the conditional distribution is a Gaussian mixture whose means lie on an ellipsoid, and the unconditional distribution is a single Gaussian approximating the mixture. The flows arising from this mixture
    
[^171]: 翻转硬币：通过硬开关训练物理模拟器

    Flipping the Coins: Training Physical Simulators through Hard Switches

    [https://arxiv.org/abs/2611.33716](https://arxiv.org/abs/2611.33716)

    提出EventFlip方法，通过随机翻转近阈值硬开关并重放求解器来测量两次运行的损失差，实现对跨越离散跳变的梯度估计，从而有效训练包含硬开关的物理模拟器。

    

    混合模拟器将神经网络等可学习组件嵌入物理求解器中，并在参考轨迹上对其进行训练。许多科学求解器具有硬开关特性，例如电力线路跳闸、电网单元点火或脑组织去极化，在这些情况下损失函数会在开关翻转时出现离散跳变，导致标准自动微分只能捕捉其平滑部分。现有的训练范式面临一个根本性的挑战性权衡：基于梯度的松弛方法会平滑开关并在跳变处引入偏差，而无导数方法则将求解器视为黑箱，从而丢弃了连续动力学的路径梯度。本文提出EventFlip方法，该方法从存储的状态出发翻转一个接近阈值的开关，重放原始求解器，并测量两次运行之间的损失差异，实质上相当于跨越开关的有限差分。通过用小的高斯噪声扰动每个阈值，我们……

    arXiv:2611.33716v1 Announce Type: new  Abstract: Hybrid simulators embed learnable components such as neural networks in physical solvers and train them on reference trajectories. Many scientific solvers feature hard switches, such as a tripping power line, an igniting grid cell, or depolarizing brain tissue, where the loss exhibits discrete jumps across switch flips, leaving standard automatic differentiation to capture only its smooth part. Existing training paradigms face a fundamentally challenging trade-off, as gradient-based relaxations smooth the switches and introduce bias across jumps, whereas derivative-free methods treat the solver as a black box and discard path gradients of the continuous dynamics. Here we introduce EventFlip, which flips a near-threshold switch from a stored state, replays the original solver, and measures the loss difference between the two runs, in effect a finite difference across the switch. By perturbing every threshold with small Gaussian noise, we 
    
[^172]: 使用卷积神经网络（CNN）和MobileNetV2进行尼泊尔货币面额分类：一项实验性比较

    Nepali Currency Denomination Classification Using Convolutional Neural Networks (CNNs) and MobileNetV2: An Experimental Comparison

    [https://arxiv.org/abs/2611.33713](https://arxiv.org/abs/2611.33713)

    经过超参数调优的MobileNetV2在七种尼泊尔货币面额分类任务中表现最优，实现了95%的验证准确率，优于传统CNN和采用迁移学习的MobileNetV2。

    

    自动货币面额识别是一项计算机视觉任务，在现金处理、金融科技和辅助系统中具有广泛的应用。本研究对七种尼泊尔货币面额（5、10、20、50、100、500和1000卢比）进行分类，采用了三种模型配置：传统卷积神经网络（CNN）、采用迁移学习的MobileNetV2，以及经过超参数调优的MobileNetV2。研究使用了一个包含16,249张图像的编译数据集，其中包括10,500张训练图像、3,500张验证图像和2,249张独立测试图像。图像被调整为128 x 128像素，经过归一化处理，并通过方向、尺度、光照和外观的多样化变换进行了数据增强。模型使用准确率、精确率、召回率、F1分数以及精确率-召回率分析和ROC分析进行评估。经过超参数调优的MobileNetV2达到了最高的验证准确率（95%），其精确率、召回率和F1分数分别为0.93、0.91和0.92，但在（原文在此处被截断）……上的准确率仅为84%。

    arXiv:2611.33713v1 Announce Type: cross  Abstract: Automated currency denomination recognition is a computer vision task with applications in cash handling, financial technology, and assistive systems. This study classifies seven Nepali currency denominations (Rs. 5, 10, 20, 50, 100, 500, and 1,000) using three configurations: a conventional convolutional neural network (CNN), MobileNetV2 with transfer learning, and MobileNetV2 with hyperparameter tuning. A compiled dataset of 16,249 images was used, comprising 10,500 training, 3,500 validation, and 2,249 independent test images. Images were resized to 128 x 128 pixels, normalized, and augmented to vary orientation, scale, illumination, and appearance. Models were evaluated using accuracy, precision, recall, F1-score, and precision-recall and ROC analysis. The hyperparameter-tuned MobileNetV2 achieved the highest validation accuracy (95\%), with precision, recall, and F1-score of 0.93, 0.91, and 0.92, but reached only 84\% accuracy on 
    
[^173]: 基于节点稳定图投影的节点差分隐私

    Node Differential Privacy via Node-Stable Graph Projections

    [https://arxiv.org/abs/2611.33672](https://arxiv.org/abs/2611.33672)

    本文提出了一种近线性时间的节点稳定图投影方法，该方法可将任意图映射为有界度数的子图且对单节点变化保持稳定，从而为设计精确高效的节点差分隐私算法提供了新的核心工具。

    

    节点差分隐私算法能够在网络结构化数据集上进行学习，其中记录反映了个体之间的交互。这类算法保护每个个体信息的隐私，包括其所有相关记录。设计此类算法的一个关键挑战在于：如何利用现实图的结构特性（即大多数节点具有较低的度数）来实现高精度，同时不损害最坏情况下的隐私保证。在这项工作中，我们为设计精确且计算高效的节点差分隐私算法提供了两个新工具。首先，我们给出了一种近线性时间的随机化节点稳定图投影：该投影使最大度数为 $D$ 的图保持不变，将任意图映射为最大度数 $\hat D \approx D$ 的子图，并确保改变输入图中的一个节点只会对在其余顶点中所选边的分布产生很小的影响。

    arXiv:2611.33672v1 Announce Type: cross  Abstract: Node-differentially private algorithms enable learning on network-structured datasets in which records reflect interactions among individuals. They protect the privacy of each individual's information, including all their associated records. A key challenge in the design of such algorithms is to exploit the structure of realistic graphs---where most nodes have low degree---to achieve high accuracy, without compromising on the worst-case privacy guarantee.   In this work we give two new tools for designing accurate, computationally efficient node-differentially private algorithms. First, we give a nearly linear time, randomized, node-stable graph projection: the projection leaves graphs of maximum degree $D$ unchanged, maps arbitrary graphs to subgraphs of maximum degree $\hat D \approx D$, and ensures that changing one node of the input graph has only a small effect on the distribution of the edges selected among the remaining vertices
    
[^174]: 可验证但不可预测：句子嵌入对命题分解究竟了解什么

    Verifiable but Not Predictable: What Sentence Embeddings Know About Propositional Decomposition

    [https://arxiv.org/abs/2611.33668](https://arxiv.org/abs/2611.33668)

    本文以PropSegmEnt语料库为基础，发现命题分解的两个质量控制问题处于不同表示空间：分解后可用NLI评判器高精度验证命题归属（准确率92.8%、AUC达0.977），而分解前无法通过句子嵌入的向量几何预测句子应产出的命题数量。

    

    arXiv:2611.33668v1 公告类型：新论文 摘要：大语言模型（LLM）命题分解的质量控制提出了两个不对称的问题。分解之后，可以进行验证：给定一个句子和一个候选命题，该命题是否属于该句子？分解之前，人们希望进行预测：这个句子应该产生多少个命题？在PropSegmEnt（一个包含英文维基百科句子、每句均由三位人类标注者进行分割的语料库）上，我们发现这两个问题处于不同的表示空间中。自然语言推理（NLI）评判器可以92.8%的准确率和0.977的ROC曲线下面积（AUC）验证命题的隶属关系，而且其残余错误大多属于在文档上下文中确实为真的命题，而非真正的蕴含错误。向量几何无法完成验证：余弦相似度会达到饱和，句子嵌入也并不是其各命题嵌入的线性组合。而在预测方面，每一个符号……（摘要原文在此处被截断）

    arXiv:2611.33668v1 Announce Type: new  Abstract: Quality control for proposition decomposition by large language models (LLMs) raises two asymmetric questions. After decomposition, one can verify: given a sentence and a candidate proposition, does the proposition belong to that sentence? Before decomposition, one would like to predict: how many propositions should the sentence yield? On PropSegmEnt, a corpus of English Wikipedia sentences each segmented by three human annotators, we find that the two questions live in different representational spaces. A natural language inference (NLI) judge verifies membership with 92.8% accuracy and an area under the ROC curve (AUC) of 0.977, and its residual errors turn out to be mostly propositions that are true in document context rather than genuine entailment mistakes. Vector geometry cannot verify: cosine similarity saturates, and a sentence embedding is not a linear combination of its propositions' embeddings. For prediction, every symbolic a
    
[^175]: 在线辨识可替代平面图先验以应对未知建筑

    Online Identification Can Replace a Floor-Plan Prior on Unseen Buildings

    [https://arxiv.org/abs/2611.33656](https://arxiv.org/abs/2611.33656)

    该论文提出一种无需目标建筑数据的暖通空调控制策略，用递归最小二乘法在线辨识一阶灰箱模型来替代平面图先验，在从未见过的建筑上性能大幅超越PID基线和ASHRAE Guideline 36。

    

    智能建筑挑战赛要求一个统一的暖通空调（HVAC）控制策略，能够仅凭建筑元数据和传感器数据流，在从未见过的办公楼和天气条件下运行。我们的控制器在运行前不接触目标建筑的任何数据，也不针对单个建筑进行调优；其少量全局常数仅在开发场景中设定。每个区域采用一个一阶灰箱模型，其初始系数由平面图计算得出，随后通过递归最小二乘法在线更新；规划器和一个闭式映射将这些模型转化为设备指令。在我们生成的两组建筑与天气数据集（验证集以及采用真实年份天气的压力集）上，该方法得分分别为起始套件PID基线的0.145倍和0.159倍，而在所有版本冻结后构建、仅评分一次的保留集上得分为0.095倍。该方法在每栋建筑上均有所改进，也优于我们实现的两种ASHRAE Guideline 36版本：每组得分为0.60至0.71倍（原文在此处截断）。

    arXiv:2611.33656v1 Announce Type: cross  Abstract: The Smart Buildings Challenge asks for one HVAC control policy that runs office buildings and weather it has never seen, from the building's metadata and sensor streams alone. Our controller sees no data from the target building before it runs and is not tuned per building; its few global constants were set on development scenarios. Each zone has a first-order grey-box model that starts from coefficients computed from the floor plan and is updated online by recursive least squares, and a planner and a closed-form mapping turn the models into plant commands. On two sets of buildings and weather that we generated, a validation set and a stress set with real-year weather, it scores 0.145 and 0.159 times the starter kit's PID baseline, and 0.095 times on a holdout built after all versions were fixed and scored once. It improves on every building, also against our two implementations of ASHRAE Guideline 36: per set it scores 0.60 to 0.71 ti
    
[^176]: 面向风险感知投资组合管理的自监督截断目标强化学习框架

    Self-supervised Clipped-objective Reinforcement Leaning Framework for Risk-aware Portfolio Management

    [https://arxiv.org/abs/2611.33652](https://arxiv.org/abs/2611.33652)

    本文提出自监督截断目标投资组合管理框架（SCP），通过两阶段架构融合逐笔级别的日内市场动态信息并稳定强化学习的高方差训练过程，从而实现风险感知的投资组合管理。

    

    投资组合管理涉及对多种金融资产的平衡配置，是量化交易中一项至关重要且充满挑战的任务。尽管在过去十年中，深度强化学习已成为解决复杂序贯决策问题的活跃研究领域，但在非平稳金融市场中部署强化学习框架会带来巨大的困难。大多数现有的基于强化学习的投资组合管理模型未能考虑到逐笔级别的日内市场动态（例如订单流失衡和流动性波动），而这些动态反映了资金的快速流动。同时，由于强化学习固有的高方差特性，这些模型在学习过程中存在不稳定的问题。本文提出了一种自监督截断目标投资组合管理框架（SCP），通过两阶段架构来解决这些局限性。市场特征整合阶段（应用……摘要在此处截断）

    arXiv:2611.33652v1 Announce Type: new  Abstract: Portfolio management involves balancing diverse financial assets, a critical yet challenging task in quantitative trading. Although deep reinforcement learning has emerged as an active research domain for solving complex sequential decision-making problems over the past decade, deploying reinforcement learning frameworks in non-stationary financial markets raises formidable difficulties. Most existing reinforcement learning-based portfolio management models fail to account for tick-level intraday market dynamics (e.g., order flow imbalances and liquidity fluctuations), which reflect rapid capital movements. Simultaneously, these models suffer from unstable learning processes due to reinforcement learning's inherent high variance. This paper presents a Self-supervised Clipped-objective Portfolio management framework (SCP), which addresses these limitations through a two-phase architecture. Market feature integration phase (application con
    
[^177]: PRIOR：SFT中的调用前推理及其对强化学习下工具使用保持能力的影响

    PRIOR: Pre-Call Reasoning in SFT and Its Influence on Tool-Use Retention under RL

    [https://arxiv.org/abs/2611.33648](https://arxiv.org/abs/2611.33648)

    该研究通过控制token预算下的工具调用深度实验，揭示SFT数据中工具调用前的推理量是决定强化学习后工具使用行为能否保持的关键因素——调用前推理越少，RL训练中工具使用越容易崩溃。

    

    在监督微调（SFT）阶段学习到的工具使用能力并不总能在强化学习（RL）阶段得以保持。我们通过构建Low、Medium和High三个SFT数据集来研究SFT数据的特性如何影响这种保持能力，这三个数据集具有大致匹配的token预算，但工具调用深度不同。在相同的RL训练方案下，三种初始模型达到了相近的RL训练准确率，却表现出显著不同的工具使用动态：Low组的工具调用发起率上升，Medium组下降，而High组则出现崩溃。我们观察到这些数据集在每次工具调用之前的推理量上也存在显著差异。特别是，High数据集的轨迹包含明显更少的调用前推理，由此引出一个假设：调用前推理可能影响工具使用的校准程度，进而决定相应行为在RL过程中是否被保持。理想化的理论分析为这一现象提供了“选择性保持”的视角。

    arXiv:2611.33648v1 Announce Type: new  Abstract: Tool use learned during supervised fine-tuning (SFT) is not always retained during reinforcement learning (RL). We study how properties of SFT data may affect this retention by constructing Low, Medium, and High SFT datasets with approximately matched token budgets but different tool-call depths. Under the same RL recipe, the three initializations reach similar RL training accuracy yet exhibit markedly different tool-use dynamics: tool initiation increases in Low, declines in Medium, and collapses in High. We observe that these datasets also differ substantially in the amount of reasoning preceding each tool call. In particular, High trajectories contain considerably less pre-call reasoning, motivating the hypothesis that pre-call reasoning may affect how well tool use is calibrated and, consequently, whether the corresponding behaviors are retained during RL. An idealized theoretical analysis provides a selective-retention view of this 
    
[^178]: 双精度的危险：工作精度作为高斯过程最大似然中的隐藏约束

    The Danger of Double Precision: Working Precision as a Hidden Constraint in Gaussian Process Maximum Likelihood

    [https://arxiv.org/abs/2611.33647](https://arxiv.org/abs/2611.33647)

    高斯过程超参数的最大似然拟合受浮点工作精度的隐式约束——只有当有效数字宽度超过协方差条件数（约70位）时拟合才会收敛，因此双精度下的拟合实际上得到的是一个不同的模型。

    

    高斯过程（GP）超参数通常通过在IEEE双精度下最大化边际似然来拟合，尽管对于平滑核和无噪声数据，协方差矩阵往往比双精度所能解析的更加病态。对于在包含贝叶斯优化簇的设计点上的平方指数核，在一系列有效数字宽度范围内进行重新拟合，并以高精度参考结果为基准进行评判，结果表明双精度拟合实际上是一个不同的模型，且只有当有效数字宽度超过协方差条件数的以2为底的对数（约70位）时，拟合结果才会收敛。Cholesky分解失败起到了一种隐式条件数约束的作用，这主要由核矩阵元素的四舍五入引起；仅根据工作精度预测出的受约束解，在40到64位的六种精度中有五种与最佳Powell拟合结果相差在20%以内，并且针对理想化问题证明了局部展开。在43个核的实验中……

    arXiv:2611.33647v1 Announce Type: new  Abstract: Gaussian process (GP) hyperparameters are routinely fitted by maximizing the marginal likelihood in IEEE double precision, although for smooth kernels and noiseless data the covariance matrix is often more ill conditioned than double precision can resolve. For a squared-exponential kernel on a design with a Bayesian-optimization cluster, refits over a range of significand widths, judged against a high-precision reference, show that the double-precision fit is a different model and that fits converge only once the width exceeds the base-2 logarithm of the covariance condition number, about 70 bits. Cholesky failure acts as an implicit conditioning constraint, caused mostly by rounding of the kernel entries; the constrained solution predicted from the working precision alone is within 20% of the best Powell fits at five of the six precisions from 40 to 64 bits, with a local expansion proved for the idealized problem. Across 43 kernels, des
    
[^179]: 关系数据库中面向预测任务的预训练

    Pretraining for Forecasting in Relational Databases

    [https://arxiv.org/abs/2611.33645](https://arxiv.org/abs/2611.33645)

    提出查询条件关系预训练（QCRP）自监督框架，利用结构化时间查询语法生成多样化预测目标，使标准关系深度学习骨干网络成为数据库上的通用预测器，其冻结表示可经轻量级预测器或零样本接口复用于各类下游预测任务。

    

    现实世界中的预测任务越来越多地依赖于关系数据库中的异步事务日志，而非均匀采样的数组。虽然关系深度学习（RDL）模型能够基于这些复杂的历史数据进行预测，但它们通常需要针对每个特定的目标和根实体类型分别训练。我们提出了查询条件关系预训练，这是一种自监督框架，使标准的RDL骨干网络能够充当整个数据库的通用预测器。QCRP使用结构化的时间查询语法生成多样化的预测目标，并通过共享的条件预测器将它们交错训练。由此产生了一个鲁棒的、冻结的编码器，可通过轻量级预测器或零样本接口重用于下游任务。在大规模RelBench数据库上的评估表明，QCRP学习到的冻结表示能够在跨任务、跨实体类型的预测中取得有竞争力的表现。此外，单一统一的QCRP模型（摘要截断）……

    arXiv:2611.33645v1 Announce Type: new  Abstract: Real-world forecasting increasingly relies on asynchronous transaction logs in relational databases rather than uniformly sampled arrays. While relational deep learning (RDL) models can forecast from these complex histories, they are typically trained separately for each specific target and root entity type. We introduce Query-Conditioned Relational pretraining (QCRP), a self-supervised framework that enables a standard RDL backbone to act as a universal forecaster across a database. QCRP generates diverse forecasting objectives using a structured temporal query grammar and interleaves them through a shared conditional predictor. This produces a robust, frozen encoder reusable for downstream tasks via lightweight predictors or a zero-shot interface. Evaluated on large-scale RelBench databases, QCRP learns frozen representations that support competitive forecasting across tasks and entity types. Furthermore, a single unified QCRP model ch
    
[^180]: 面向多人微分博弈的物理信息学习方法

    Physics-Informed Learning for Multiplayer Differential Games

    [https://arxiv.org/abs/2611.33644](https://arxiv.org/abs/2611.33644)

    该论文提出一种在策略迭代框架内结合消失粘性退火调度、利用物理信息神经网络逼近N人对抗性微分博弈粘性解的方法，并通过可计算残差给出了显式的ε-纳什保证及有限时间收敛性证明。

    

    在N人对抗性微分博弈中计算各玩家的反馈策略极具挑战性，因为相关的耦合哈密顿-雅可比-贝尔曼方程组是高维的，且通常缺乏经典解。为解决这一问题，我们在带有消失粘性退火调度的策略迭代框架中，利用物理信息神经网络来逼近粘性解。在理论方面，我们通过可计算的残差对每个玩家的遗憾值进行界定，建立了显式的ε-纳什保证，并证明了该方法能在有限时间内达到此停止准则。对于两人零和博弈，我们进一步给出了价值函数逼近误差的可计算上界。在追逃场景上的数值实验表明了这些计算所得界的紧致性，以及该框架在高维环境中的可扩展性。

    arXiv:2611.33644v1 Announce Type: cross  Abstract: Computing per-player feedback strategies in $N$-player adversarial differential games is challenging as the associated coupled Hamilton-Jacobi-Bellman system is high-dimensional and generally lacks classical solutions. We address this by approximating viscosity solutions using physics-informed neural networks within a policy iteration scheme with a vanishing-viscosity annealing schedule. Theoretically, we establish an explicit $\varepsilon$-Nash guarantee by bounding per-player regret via computable residuals and prove our method reaches this stopping criterion in finite time. For two-player zero-sum games, we further establish a computable bound on the approximation error of the value function. Numerical experiments on pursuit-evasion scenarios illustrate the tightness of these computed bounds and the framework's scalability to high-dimensional environments.
    
[^181]: AI代码审查者需要多少努力程度？来自Claude 5.5模型对似是而非错误补丁的证据

    How Much Effort Does an AI Code Reviewer Need? Evidence from Claude 5.5 Models on Plausible-but-Wrong Patches

    [https://arxiv.org/abs/2611.33639](https://arxiv.org/abs/2611.33639)

    通过对Claude 5.5系列模型审查90个SWE-bench补丁的预注册实验，本研究发现更高的推理努力程度能显著减少AI代码审查者对似是而非错误补丁的误接受，且对Opus而言中等努力程度能以低20%的成本达到与高等努力程度相当的效果。

    

    AI编码代理编写代码的速度超过了人类审查代码的速度，因此团队开始引入AI审查者。推理模型使这样的审查者能够通过努力程度设置在成本与细致程度之间进行权衡。我们探究了审查者需要多少努力才能发现似是而非的错误补丁——即能通过基准测试但在更强测试下失败的补丁。Claude Opus 5.5（预注册）、Sonnet 5.5和Haiku 5.5（探索性）各自以低、中、高三档努力程度审查了相同的90个来自SWE-bench Verified的代理编写的Django补丁。通过强化后的SWE-ABS测试，其中55个补丁是错误的，35个是正确的。在所有模型中，高等努力程度比低等努力程度接受更少的错误补丁（Opus的虚假成功率为0.45对0.67，Sonnet为0.49对0.65，Haiku为0.51对0.78），其中Opus和Haiku的差异得到统计学支持。对于Opus，中等努力程度在15个百分点范围内不劣于高等努力程度（差异0.018，95% CI [-.065, .108]），且成本降低20%，未达到预注册的30%标准。对于Sonnet和Haiku……

    arXiv:2611.33639v1 Announce Type: new  Abstract: AI coding agents write code faster than people can review it, so teams add AI reviewers. Reasoning models let such a reviewer trade cost for diligence through an effort setting. We asked how much effort a reviewer needs to catch plausible-but-wrong patches, which pass a benchmark's tests but fail stronger ones. Claude Opus 5.5 (preregistered), Sonnet 5.5 and Haiku 5.5 (exploratory) each reviewed the same 90 agent-written Django patches from SWE-bench Verified at low, medium and high effort. By strengthened SWE-ABS tests, 55 patches were wrong and 35 correct. In every model, high effort accepted fewer wrong patches than low (false success .45 against .67 for Opus, .49 against .65 for Sonnet, .51 against .78 for Haiku), a supported difference for Opus and Haiku. For Opus, medium was non-inferior to high within 15 percentage points (difference .018, 95% CI [-.065, .108]) and cost 20% less, short of the preregistered 30%. For Sonnet and Haik
    
[^182]: 通过神经元干预解释斑点效应

    Interpreting Speckle Effects Through Neuron Interventions

    [https://arxiv.org/abs/2611.33632](https://arxiv.org/abs/2611.33632)

    研究发现，SAR预训练模型中的斑点信息并非由斑点选择性神经元直接编码，而是主要由第一个MLP层的64个神经元承载，修复这些神经元可恢复44.1%的相关性损失。

    

    斑点是合成孔径雷达（SAR）成像中固有的乘性噪声，会降低预训练SAR模型的性能。我们使用可解释性方法来考察其对预训练视觉模型内部的影响。我们识别出对斑点有选择性响应的神经元，并测试纠正这些神经元的激活是否能改善重建效果。我们研究了SARMAE——一个在百万规模SAR数据集上预训练的地球观测模型。具有已知反射率的合成场景使我们能够精确测量恢复效果。该模型包含38个追踪斑点值分布的斑点选择性神经元。然而，将这些神经元恢复到其干净值并不能改善重建。跨层的激活修补（activation patching）实验表明，第一个MLP层具有最大的恢复和注入效应。在该层内部，另一组共64个神经元在保留测试场景上恢复了44.1%的相关性损失。当斑点选择性神经元被保持……

    arXiv:2611.33632v1 Announce Type: cross  Abstract: Speckle, the multiplicative noise inherent to synthetic aperture radar (SAR) imaging, reduces the performance of pretrained SAR models. We use interpretability methods to examine its effects inside pretrained vision models. We identify neurons that respond selectively to speckle and test whether correcting their activations improves reconstruction. We study SARMAE, an Earth observation model pretrained on a million-scale SAR dataset. Synthetic scenes with known reflectivity allow us to measure recovery precisely. The model contains 38 speckle-selective neurons that track speckle's value distribution. Restoring these neurons to their clean values does not improve reconstruction. Activation patching across layers shows that the first MLP has the largest restoration and injection effects. Within this layer, a different set of 64 neurons recovers 44.1\% of the correlation loss on held-out scenes. When the speckle-selective neurons are held
    
[^183]: 状态空间注意力：学习上下文相关的记忆保留

    State Space Attention: Learning Context-Dependent Memory Retention

    [https://arxiv.org/abs/2611.33625](https://arxiv.org/abs/2611.33625)

    提出状态空间注意力（SSA），通过可学习的评分器自适应地决定保留哪些历史表示以维持有界内存，并配合“恢复”式监督样本训练技术，使模型能够依据上下文动态选择长上下文推理所需的记忆内容。

    

    标准注意力的键值缓存会随上下文长度线性增长。若要对其加以限制，便会引出一个问题：序列模型在长上下文推理中应当保留什么。我们认为这一选择应当是可学习的，并据此提出了状态空间注意力（SSA），它学习保留一个有界的过去表示集合，并通过标准注意力来读取这些表示。一个学习得到的评分器会依据新到达的上下文以及记忆中的其余内容，对每个块进行重新评分。固定上下文的Transformer由此成为一种特例——即始终保留最近token的情况。我们联合训练评分器与读取器，或者从可微的软记忆加权逐渐过渡到硬性保留，或者从一开始就采用硬性保留。为了给评分器产生训练信号，我们在训练过程中生成监督样本，我们将这一技术称为“恢复”。我们在检索等任务上将SSA与循环模型、有界内存注意力方法以及全注意力Transformer进行了比较。

    arXiv:2611.33625v1 Announce Type: cross  Abstract: Standard attention's key-value cache grows linearly with context length. Bounding it raises the question of what a sequence model should retain for long-context reasoning. We argue this choice should be learned, and introduce State Space Attention (SSA), which learns to keep a bounded set of past representations and reads them with standard attention. A learned scorer rescores every chunk against newly arriving context and the rest of memory. A fixed-context Transformer is then the special case that always keeps the most recent tokens. We train the scorer and reader jointly, either moving gradually from a differentiable soft weighting of memories to hard retention, or with hard retention from the start. To produce signal for the scorer, we generate supervised samples during training, a technique we call recovery. We compare SSA with recurrent models, bounded-memory attention methods, and full-attention Transformers across retrieval, va
    
[^184]: 从自然图像中学习解缠数感

    Learning from Natural Images Untangles Number Sense

    [https://arxiv.org/abs/2611.33615](https://arxiv.org/abs/2611.33615)

    该论文提出一个几何框架，发现尽管未训练与预训练网络的单元级数量选择性相似，但自然图像学习能将数量与比例的表征解缠分离，证明视觉学习在数量选择性的自发产生之外仍有重要贡献。

    

    数值认知是一种简单却基础的能力，但对现代视觉-语言模型而言仍然具有挑战性。与此同时，数感已经存在于从未接受过数字训练的动物中，这表明它可以在没有显式数字学习的情况下自发产生。深度神经网络提供了一个相关的观察：在训练于不相关的自然图像任务之后，甚至在任何训练之前，就会出现数量选择性单元。然而，这些发现仍未解决视觉学习在数量选择性的自发产生之外是否还有额外贡献的问题。人类发育提供了一个线索：儿童在数量与比例相互冲突的判断中会变得越来越好。在此，我们引入一个几何框架来追踪这两种量的表征在学习过程中如何变化。尽管未训练网络和预训练网络在单元级数量选择性以及基本比较能力上表现相似，自然图像……（摘要原文在此处截断）

    arXiv:2611.33615v1 Announce Type: new  Abstract: Numerical cognition is a simple yet fundamental ability that remains challenging for modern vision-language models. Meanwhile, number sense is already present in numerically na\"ive animals, suggesting that it can emerge without explicit numerical learning. Deep neural networks provide a related observation, with numerosity-selective units appearing after training on unrelated natural-image tasks and even before any training. However, these findings leave unresolved whether visual learning contributes beyond the spontaneous emergence of numerical selectivity. Human development provides a clue, as children become better at making judgments in which numerosity and proportion conflict. Here, we introduce a geometric framework to trace how representations of these two quantities change across learning. Although untrained and pretrained networks exhibited similar unit-level numerosity selectivity and basic comparison performance, natural-imag
    
[^185]: 基于KL不确定性集的鲁棒强化学习样本复杂度的改进

    Improved Sample Complexity of Robust Reinforcement Learning with KL Uncertainty Sets

    [https://arxiv.org/abs/2611.33592](https://arxiv.org/abs/2611.33592)

    本文针对基于KL散度不确定性集的鲁棒强化学习，给出了更紧致且完备的样本复杂度上下界，填补了以往结果中的差距，并将下界覆盖范围扩展到完整的不确定性水平区间。

    

    本文研究了采用Kullback-Leibler（KL）散度定义的不确定性集的分布鲁棒强化学习。目标是找到一个能够最大化鲁棒价值函数的策略，该鲁棒价值函数定义为不确定性集中所有转移核对应的最坏情况值。在假设可以使用生成模型的前提下，我们旨在理解寻找ε-最优鲁棒策略所需的样本复杂度。文献中已知的最佳样本复杂度结果表明，上界与下界之间存在大小为 $\mathcal{O}\{\max\{p_\wedge^{-1}(1-\gamma)^{-1},(1-\gamma)^{-2}\}\}$ 的非平凡差距，其中 $p_\wedge$ 表示标称转移核的最小正转移概率，$\gamma$ 为折扣因子。更重要的是，现有关于下界的结果仅覆盖了有限范围的不确定性水平。在本文中，我们为基于KL不确定性集的鲁棒强化学习开发出更紧致且完备的上下界。

    arXiv:2611.33592v1 Announce Type: cross  Abstract: In this paper, we study distributionally robust reinforcement learning with Kullback--Leibler (KL) divergence-defined uncertainty sets. The goal is to find a policy that maximizes the robust value function, defined as the worst-case value over all transition kernels in the uncertainty set. Assuming access to a generative model, we aim to understand the sample complexity of finding an $\epsilon$-optimal robust policy. The best-known sample complexity results in the literature show a non-trivial gap of $\mathcal{O}\{\max\{p_\wedge^{-1}(1-\gamma)^{-1},(1-\gamma)^{-2}\}\}$ between the upper and lower bounds, where $p_\wedge$ denotes the minimum positive transition probability of the nominal transition kernel, and $\gamma$ is the discount factor. More importantly, existing results on the lower bound only cover a limited range of the uncertainty level. In this paper, we develop tighter and complete upper and lower bounds for robust RL with K
    
[^186]: 摊销精确正交性：面向正交微调的漂移触发恢复方法

    Amortizing Exact Orthogonality: Drift-Triggered Restoration for Orthogonal Fine-Tuning

    [https://arxiv.org/abs/2611.33589](https://arxiv.org/abs/2611.33589)

    该论文提出 RTOFT 方法，用一阶更新 W(I+S) 取代逐步构造正交变换，仅在漂移 ‖S‖₂ 超过容差 τ 时才通过极分解恢复精确正交性，从而以漂移触发的方式摊销正交微调的计算开销。

    

    正交微调（OFT）通过正交变换来适配预训练权重，并保持其奇异值和行间角度，但现有方法在每次优化步骤中都需要构造精确或近似的正交变换。我们通过正交性缺陷 ‖AᵀA−I‖₂ 来衡量变换 A 偏离正交的程度。对于截断 Cayley-Neumann 级数的 OFTv2，我们证明截断阶数并不能决定缺陷大小：在默认阶数下，参考实现检查点中的分块缺陷跨越五个数量级。我们提出重启切线正交微调，它使用带斜对称矩阵 S 的权重 W 的一阶更新 W(I+S)，并且仅在漂移 ‖S‖₂ 达到容差 τ 时，才通过极分解恢复精确正交性。由于 I+S 的缺陷恰好为 ‖S‖₂²，因此只需在每次更新后读取漂移，并在……（摘要在此处被截断）

    arXiv:2611.33589v1 Announce Type: new  Abstract: Orthogonal finetuning (OFT) adapts a pretrained weight by an orthogonal transform and preserves its singular values and row angles, but existing methods construct an exact or approximate orthogonal transform at every optimization step. We measure how far a transform $A$ is from orthogonal by its orthogonality defect $\|A^\top A-I\|_2$. For OFTv2, which truncates a Cayley-Neumann series, we show that the truncation order does not determine the defect. At its default order, block defects in checkpoints from the reference implementation span five orders of magnitude. We propose Restarted Tangent Orthogonal Finetuning (RTOFT), which uses the first-order update $W(I+S)$ of a weight $W$ with a skew-symmetric $S$ and restores exact orthogonality by a polar decomposition only when the drift $\|S\|_2$ reaches a tolerance $\tau$. The defect of $I+S$ is exactly $\|S\|_2^2$. Therefore, reading the drift after every update and restoring before the ne
    
[^187]: 对称性感知的少体分子波函数残差学习

    Symmetry-Aware Residual Learning of Few-Body Molecular Wave Functions

    [https://arxiv.org/abs/2611.33576](https://arxiv.org/abs/2611.33576)

    该论文提出一种物理信息神经变分方法，将分子波函数分解为编码旋转对称性与渐近衰减的解析分量和神经残差分量，从而在不依赖离散网格的情况下有效降低少体极性分子系统高维本征值问题的求解难度。

    

    超冷极性分子是工程化量子物质的高度可控平台，但对其少体性质的建模在计算上仍极具挑战性。其各向异性的长程相互作用和丰富的内部结构导致了高维本征值问题，而传统的基于网格的求解器在很大程度上仅限于处理双分子系统。为解决这一局限，我们引入了一种物理信息驱动的神经变分方法，该方法充分利用了分子波函数已知的物理结构。我们的核心思想是将波函数分解为解析分量和神经分量：解析分量编码旋转对称性和渐近衰减行为，而神经残差部分学习剩余的中程行为。这种分解在不依赖临时近似或针对特定问题的离散网格的情况下，有效降低了问题的有效维度。我们将我们的方法与高精度基准进行了比较验证。

    arXiv:2611.33576v1 Announce Type: cross  Abstract: Ultracold polar molecules are a highly controllable platform for engineering quantum matter, but modeling their few-body properties remains computationally challenging. Their anisotropic long-range interactions and rich internal structure lead to high-dimensional eigenvalue problems for which conventional grid-based solvers are largely restricted to two-molecule systems. To address this limitation, we introduce a physics-informed neural variational method that exploits known physical structure of the molecular wave function. Our key idea is to factor the wave function into analytic and neural components. The analytic component encodes rotational symmetry and asymptotic decay, while the neural residual learns the remaining intermediate-range behavior. This decomposition reduces the effective dimensionality of the problem without relying on ad hoc approximations or problem-specific discretization grids. We benchmark our method against hi
    
[^188]: 面向后门移除的触发器不变对比遗忘

    Trigger-Invariant Contrastive Unlearning for Backdoor Removal

    [https://arxiv.org/abs/2611.33571](https://arxiv.org/abs/2611.33571)

    提出TICU框架，通过构建干净与插入触发器的反事实样本对并强制预测与表示对触发器插入保持不变性，有效移除NLP模型中的文本后门，同时借助干净数据监督损失和参数漂移约束维持干净任务性能。

    

    后门攻击对现代NLP模型构成严重威胁，其通过嵌入隐藏触发器在推理阶段引发恶意行为。近期的遗忘方法（如梯度上升法GA）试图通过最大化中毒样本的损失来移除后门；然而，这类方法存在一个关键局限：它们可能将触发器的影响转移到其他类别，而非真正将其消除。在本工作中，我们提出了触发器不变对比遗忘，这是一个通过强制模型对触发器插入具有预测和表示不变性来缓解文本后门行为的框架。TICU构建干净样本与插入触发器的反事实样本对，然后对模型进行微调，使两类输入产生一致的输出分布和对齐的隐藏表示。为了保持干净任务上的性能，TICU还在干净数据上保留监督分类损失，并约束参数相对原始模型的漂移。

    arXiv:2611.33571v1 Announce Type: new  Abstract: Backdoor attacks pose a severe threat to modern NLP models by embedding hidden triggers that induce malicious behavior at inference time. Recent unlearning methods such as gradient ascent (GA) attempt to remove backdoors by maximizing the loss on poisoned samples; however, these methods can suffer from a critical limitation: they may shift the trigger's influence to other classes rather than eliminating it. In this work, we propose Trigger-Invariant Contrastive Unlearning (TICU), a framework for mitigating textual backdoor behavior by enforcing prediction and representation invariance to trigger insertion. TICU constructs clean and trigger-inserted counterfactual pairs, then fine-tunes the model so that both inputs produce consistent output distributions and aligned hidden representations. To preserve clean-task performance, TICU also retains the supervised classification loss on clean data and constrains parameter drift from the origina
    
[^189]: DiffER²：基于SMILES适配粒子引导的扩散集成逆合成预测方法

    DiffER$^2$: Diffusion Ensembles for Retrosynthesis Prediction with SMILES Adapted Particle Guidance

    [https://arxiv.org/abs/2611.33557](https://arxiv.org/abs/2611.33557)

    本文提出将粒子引导适配到SMILES序列生成的扩散模型集成方法DiffER²，显著提升了逆合成预测中生成独特分子的数量和top-k准确率。

    

    计算逆合成预测有望通过自动生成目标产物的潜在反应来缩短新药的研发时间。数据驱动方法通常将该任务视为SMILES字符串上的序列到序列生成任务。在本工作中，我们构建了用于分子生成的离散时间和连续时间扩散模型集成，并引入引导机制以提高输出多样性。我们提出了将粒子引导适配到SMILES序列生成的方法，该方法在显著提高扩散集成所生成的独特分子数量的同时，还提升了top-k准确率。这些结果进一步扩展了离散扩散在SMILES生成中的有效性，我们的实证分析也为扩散模型在化学逆合成中的能力提供了新的见解。

    arXiv:2611.33557v1 Announce Type: new  Abstract: Computational retrosynthesis prediction has the potential to reduce development time for newly discovered drugs by automatically generating potential reactions for a target product. Data-driven approaches commonly treat this as a sequence-to-sequence generation task on SMILES strings. In this work, we construct ensembles of discrete-time and continuous-time diffusion models for molecular generation and incorporate guidance mechanisms for improved output diversity. We propose an adaptation of particle guidance to SMILES sequence generation which significantly improves the number of unique molecules generated by diffusion ensembles while increasing top-k accuracy. These results further expand the efficacy of discrete diffusion for SMILES generation, and our empirical analyses offer new insights into the capabilities of diffusion models for chemical retrosynthesis.
    
[^190]: FokasNets：借助全局关系补全物理信息神经网络与算子中缺失的边界数据

    FokasNets: Completing missing boundary data in Physics-Informed Neural Networks and Operators via the Global Relation

    [https://arxiv.org/abs/2611.33549](https://arxiv.org/abs/2611.33549)

    FokasNets将PINNs/PINOs与Fokas方法的全局关系相结合，在训练过程中自动补全缺失的边界条件，从而更高效、更准确地求解混合边界问题。

    

    我们提出了FokasNets，这是一个将物理信息神经网络（PINNs）与物理信息算子（PINOs）同Fokas方法（也称为统一变换）的全局关系（GR）公式相结合的框架，用于处理混合边界问题。该通用框架采用神经网络模型的提升表示，其中Dirichlet边界条件通过构造方式直接施加，而由神经网络学习内部修正量。全局关系在训练过程中被用作一个额外的求解步骤，为提升的神经网络提供缺失的边界条件，从而使训练在区域内部和边界上都更加高效和准确。我们针对椭圆型和演化型偏微分方程构建了FokasNets（Fokas-PINNs），并利用这些偏微分方程类别的全局关系公式，构建了相应的偏微分方程解算子模型（Fokas算子网络，FONets）。此外，我们开发了……

    arXiv:2611.33549v1 Announce Type: cross  Abstract: We introduce FokasNets, a framework that combines Physics-Informed Neural Networks (PINNs) and operators (PINOs) with the Global Relation (GR) formulation of the Fokas method (also referred to as the Unified Transform), for mixed boundary problems. The general framework considers lifted representations of the neural network models, where the Dirichlet boundary conditions are imposed by construction and the neural network learns the interior correction. The Global Relation is used as an additional solve during training that provides the missing boundary conditions to the lifted neural network, making training more efficient and accurate in the interior and on the boundary. We develop the construction of FokasNets for elliptic and evolution PDEs (Fokas-PINNs), and the corresponding PDE solution operator problems (Fokas Operator Networks - FONets), using the Global Relation formulations for these PDE classes. Furthermore, we develop the n
    
[^191]: RankCast：面向MoE服务的阶段感知秩均衡系统

    RankCast: Phase-Aware Rank Balancing in MoE Serving

    [https://arxiv.org/abs/2611.33542](https://arxiv.org/abs/2611.33542)

    RankCast是一个面向MoE推理服务的阶段感知负载均衡系统，针对预填充阶段计算受限（由秩间最大token数决定延迟）与解码阶段内存受限（由秩间最大激活专家数决定延迟）的不同瓶颈，在统一框架下通过为每个token保留Top-p专家并智能分配，降低了EP秩间的负载不均衡，从而减少推理延迟。

    

    现代大语言模型（LLM）推理日益采用将预填充（Prefill）和解码（Decode）阶段分离的解耦架构。与此同时，专家并行（EP）已被广泛应用于大规模混合专家模型的服务中。然而，负载不均衡仍然会拖慢MoE推理，因为每一层都必须等待最繁忙的EP秩。预填充与解码阶段延迟的主要来源并不相同：预填充是计算受限的，其MoE层的延迟由最繁忙秩的最大token数量决定；而在中等批大小下，解码是内存受限的，其延迟由各秩间最大激活专家数量主导。为了在统一框架下解决这两个问题，我们提出了RankCast，一个在预填充阶段降低秩级最大token数量、在解码阶段降低激活专家数量的系统。对于每个token，RankCast保留预测出的Top-p专家，并选择剩余的……

    arXiv:2611.33542v1 Announce Type: cross  Abstract: Modern large language model (LLM) inference increasingly adopts disaggregated architectures that separate Prefill (P) and Decode (D) phases. Meanwhile, Expert Parallelism (EP) has been widely adopted in large Mixture-of-Experts (MoE) model serving. However, load imbalance still slows down MoE inference because each layer must wait for the busiest EP rank. The main source of latency differs between prefill and decode. Prefill is compute-bound, and its MoE layers' latency is determined by the maximum token count of the busiest rank. At the same time, under moderate batch sizes, Decode is memory-bound, so the latency is dominated by the maximum active-expert count across ranks. To solve both problems under a unified framework, we present RankCast, a system that reduces rank-level maximum token counts during prefill and active-expert counts during decode. For every token, RankCast preserves the predicted Top-p experts and chooses the remai
    
[^192]: 具有 $d^{3/2}\sqrt{n}$ 遗憾的对抗性赌博机凸优化

    Adversarial Bandit Convex Optimization with $d^{3/2}\sqrt{n}$ Regret

    [https://arxiv.org/abs/2611.33531](https://arxiv.org/abs/2611.33531)

    本文提出一种带高斯探索的在线牛顿步算法，将对抗性赌博机凸优化的遗憾上界从 $\widetilde O(d^{5/2}\sqrt n)$ 改进为 $\widetilde O(d^{3/2}\sqrt n)$，匹配了此前被猜想的最优速率，并将极小极大遗憾确定到 $d^{1/6}$ 因子以内。

    

    我们研究对抗性赌博机凸优化问题：在 $n$ 轮中的每一轮，学习者选择凸体 $\mathcal{K}\subset\mathbb{R}^d$ 中的一个点 $A_t$，并仅观测到该点处凸损失函数 $f_t:\mathcal{K}\to[0,1]$ 的含噪值。损失函数由对手选择，对手可以逐轮改变损失函数，并根据学习者过去的动作进行自适应调整。我们将该问题目前已知的最佳遗憾界从 $\widetilde O(d^{5/2}\sqrt n)$（Lattimore, 2020, 2026）改进为 $\widetilde O(d^{3/2}\sqrt n)$，这与 Bubeck 等人（2017）所猜测的速率相匹配。结合 Rajaraman 和 Han（2026）最近给出的 $\Omega(d^{4/3}\sqrt n)$ 下界，该结果将极小极大遗憾确定到 $d^{1/6}$ 因子和对数因子范围之内。该算法是带高斯探索的在线牛顿步方法。与以往同类方法不同，它使用高斯平滑后函数的梯度与 Hessian 矩阵的无偏估计……

    arXiv:2611.33531v1 Announce Type: cross  Abstract: We study adversarial bandit convex optimization, where in each of $n$ rounds a learner chooses a point $A_t$ in a convex body $\mathcal{K}\subset\mathbb{R}^d$ and observes only a noisy value of a convex loss $f_t:\mathcal{K}\to[0,1]$ at that point. The losses are chosen by an adversary, who may change them from round to round and adapt them to the learner's past actions. We improve the best known regret bound for this problem from $\widetilde O(d^{5/2}\sqrt n)$ (Lattimore, 2020, 2026) to $\widetilde O(d^{3/2}\sqrt n)$, which matches the rate conjectured by Bubeck et al. (2017). Together with the recent lower bound of $\Omega(d^{4/3}\sqrt n)$ due to Rajaraman and Han (2026), it determines the minimax regret up to a factor of $d^{1/6}$ and logarithmic factors. The algorithm is an online Newton step with Gaussian exploration. Unlike earlier methods of this kind, it uses unbiased estimates of the gradient and Hessian of the Gaussian smooth
    
[^193]: Casper：一种基于投影的可扩展且保证约束满足的神经符号层

    Casper: A Projection-Based Neurosymbolic Layer for Scalable & Guaranteed Constraint Satisfaction

    [https://arxiv.org/abs/2611.33527](https://arxiv.org/abs/2611.33527)

    Casper 是一种基于投影的神经符号层，通过将对约束可行域的闭式欧几里得投影嵌入单次前向传播，同时实现了约束满足保证、推理时的可扩展性、完全自动化以及完全可微性。

    

    许多现实世界的问题需要精确的约束满足才能得到有意义的解决，然而现有的神经符号方法面临一个严峻的权衡：那些能保证约束满足的方法无法扩展，而那些能够扩展的方法则必须 resort to 近似。与此同时，传统的深度学习方法完全不提供任何满足保证。在本文中，我们提出了 Casper，这是一种神经符号层，它同时具备以下特点：(i) 保证约束满足，(ii) 在推理时可扩展，(iii) 完全自动化，无需针对特定问题的工程设计，以及 (iv) 完全可微，使其在训练和推理阶段都能与任何神经预测器进行组合。Casper 将约束满足问题转化为对约束可行域的闭式欧几里得投影，可在单次前向传播中完成计算。在 MNIST 算术（多达 1024 位数字）、数独求解和 Warcraft 寻路等实验中，结果表明 Casper 是唯一……（原文摘要在此处截断）

    arXiv:2611.33527v1 Announce Type: new  Abstract: Many real-world problems demand exact constraint satisfaction to be meaningfully solved, yet existing neurosymbolic methods face a stark trade-off: those that guarantee satisfaction fail to scale, while those that scale must resort to approximation. Meanwhile, traditional deep learning approaches offer no satisfaction guarantees at all. In this paper, we propose Casper, a neurosymbolic layer that simultaneously (i) guarantees constraint satisfaction, (ii) scales at inference time, (iii) is fully automated, requiring no problem-specific engineering, and (iv) is fully differentiable, making it composable with any neural predictor both at training and at inference time. Casper casts constraint satisfaction as a closed-form Euclidean projection onto the constraint-feasible region, computable in a single forward pass. Experiments across MNIST arithmetic (up to 1024 digits), Sudoku solving, and Warcraft pathfinding show that Casper is the only
    
[^194]: 可验证的删除：审计大语言模型的“遗忘”（Unlearning）

    Deletion You Can Check: Auditing LLM Unlearning

    [https://arxiv.org/abs/2611.33525](https://arxiv.org/abs/2611.33525)

    该论文指出，业界常用的“成员推理得分接近随机（ROC-AUC ≈ 0.5）即代表遗忘成功”这一审计标准，依赖于极少被验证的参照群体假设，并通过 2×2 对照实验（结合精确重训练金标准与事后诊断）证明基准参照群体的不匹配会导致对大模型删除/遗忘效果的错误判断，从而提出了一套更可靠的遗忘审计方法。

    

    微调后的模型已成功遗忘某条被删除记录的一个常见判断依据，是成员推理得分接近随机水平（ROC-AUC ≈ 0.5）。只有当用于比较的参照群体能够代表该记录在有效删除后应有的表现时，这种解读才成立——而这一反事实情形很少得到验证。我们在一个受控的 2×2 研究中检验了这一前提，将两个成员推理基准（ArxivMIA 和 BookMIA）与两个语言模型（Llama-3.2-1B 和 Qwen2.5-1.5B）配对。以精确重训练作为金标准来验证成员推理参照：在 ArxivMIA 上参照位于预设的随机带内，但在 BookMIA 上达到 0.122，使得同一指标拒绝了经验性的删除参照。一项针对 BookMIA/Qwen 组合的事后诊断将基准中标签为 0 的记录替换为同池中从未参与微调的记录，使参照从 0.257 移动到 0.523，从而证实了该单元中存在参照群体不匹配的问题。每个近似遗忘的检查点都被联合评判……（原文摘要在此处截断）

    arXiv:2611.33525v1 Announce Type: new  Abstract: A common indication that a fine-tuned model has forgotten a deleted record is a membership-inference score near chance (ROC-AUC $\approx 0.5$). This reading is sound only if the comparison population represents how the record should behave after a valid deletion---a counterfactual that is rarely validated. We test this premise in a controlled $2\times2$ study pairing two membership-inference benchmarks, ArxivMIA and BookMIA, with two language models, Llama-3.2-1B and Qwen2.5-1.5B. Exact retraining validates the membership reference: it lies within the prespecified chance band on ArxivMIA but reaches $0.122$ on BookMIA, causing the same metric to reject an empirical deletion reference. A post-hoc BookMIA/Qwen diagnostic replaces benchmark label-0 records with same-pool, never-fine-tuned records and moves the reference from $0.257$ to $0.523$, supporting population mismatch in that cell. Each approximate-unlearning checkpoint is judged joi
    
[^195]: 一种用于降维敏感性分析的扰动驱动框架

    A Perturbation-Driven Framework for Sensitivity Analysis of Dimensionality Reduction

    [https://arxiv.org/abs/2611.33523](https://arxiv.org/abs/2611.33523)

    本文提出一种模型无关的扰动驱动敏感性评估框架，将扰动视为对合理数据变化的受控探针，通过量化结构保持的变化与视觉诊断来评估降维方法对输入微小变化的鲁棒性。

    

    由降维（DR）产生的低维表示通常使用可信度和连续性等失真度量进行评估。在本文中，我们通过量化结构保持方面的变化并视觉诊断这些变化如何在最终表示中体现，来研究降维方法对合理输入变化的敏感性。这一视角反映了现实世界数据集是底层分布的有限、含噪样本这一事实，即降维不应在数据发生微小变化时产生显著不同的结构。我们引入了一个模型无关的、扰动驱动的敏感性评估框架，该框架将扰动视为对合理数据变化的受控探针。在多种扰动族和强度水平下，我们反复扰动输入，应用降维方法，并使用一组失真度量来评估所得的低维表示。

    arXiv:2611.33523v1 Announce Type: new  Abstract: Low-dimensional representations produced by dimensionality reduction (DR) are commonly evaluated using distortion measures such as trustworthiness and continuity. In this paper, we study the sensitivity of DR methods to plausible input variations by quantifying changes in structural preservation and visually diagnosing how these changes manifest in the resulting representations. This perspective reflects the fact that real-world datasets are finite, noisy samples of an underlying distribution, and DR should not produce substantially different structures under small data changes. We introduce a model-agnostic, perturbation-driven sensitivity evaluation framework that treats perturbations as controlled probes of plausible data variation. Across multiple perturbation families and strength levels, we repeatedly perturb the input, apply a DR method, and evaluate the resulting low-dimensional representations using a set of distortion measures 
    
[^196]: 推理微调大语言模型的安全性评估与恢复

    Evaluation and Restoration of Safety in Reasoning-Tuned LLMs

    [https://arxiv.org/abs/2611.33515](https://arxiv.org/abs/2611.33515)

    本文揭示了一个反直觉的发现——增强推理能力会在所有模型家族中显著提高大语言模型的不安全输出率，并提出了一种将进化策略与后训练防护栏相结合的新方法（ESPG），以有效恢复模型的安全性。

    

    通过推理能力优化的开源大语言模型（如 VibeThinker）旨在通过将高能力模型带给资源有限的用户来实现AI的民主化。这些模型广泛可得，但缺乏监管和外部防护措施；因此，为了将它们安全地融入人类社会，关键在于它们必须以安全的方式运行。有一种流行且直观的论点认为，推理能力的提升应该会提高安全性，但最近的证据表明，事实上情况可能恰恰相反。本文旨在刻画并缓解这一问题。首先，本文开发了一种用于评估推理优化大语言模型的指标，并表明在多个模型家族中增强推理能力在所有情况下都会增加不安全输出率。其次，本文提出了一种通过将进化策略与后训练防护栏相结合来恢复安全性的方法（ESPG）。与诸如 GRPO 等替代性的安全性后训练方法相比，ESPG 表现出更优的效果。

    arXiv:2611.33515v1 Announce Type: new  Abstract: Open-source LLMs that are optimized for reasoning, such as VibeThinker, aim to democratize AI by bringing high-capability models to users with limited resources. They are broadly available but lack regulation and external guardrails; therefore, in order to integrate them safely into human society, it is critical that they perform in a safe manner. There are popular and intuitive arguments for why improved reasoning should improve safety, but recent evidence suggests that, in fact, the opposite may be true. This paper aims to characterize and mitigate this issue. First, it develops a metric for evaluating LLMs optimized for reasoning, and shows that increasing reasoning across multiple model families increases the unsafe rate in all cases. Second, it proposes a method for restoring safety by combining evolution strategies with a post-training guardrail (ESPG). Compared to alternative methods for safety post-training such as GRPO, ESPG is 
    
[^197]: 具有缆绳吊挂载荷的多旋翼无人机的模型预测控制：移位初始化与基于Transformer的暖启动的比较

    Model Predictive Control of a Multirotor UAV with a Cable-Suspended Payload: A Comparison of Shifted and Transformer-Based Warm Starts

    [https://arxiv.org/abs/2611.33514](https://arxiv.org/abs/2611.33514)

    本文针对携带缆绳吊挂载荷的多旋翼无人机的模型预测控制，提出用基于Transformer的暖启动替代传统移位初始化来加速序列凸规划优化器，数值实验表明Transformer初始化能减少计算时间。

    

    本研究研究了携带缆绳吊挂载荷的多旋翼无人机（无人驾驶飞行器，UAV）的模型预测控制（MPC），将传统的移位初始化与基于Transformer的暖启动在序列凸规划（SCP）优化器上进行了比较。该控制器采用带有弹性缆绳的耦合机械模型，并将吊挂载荷表示为双摆。其公式包含移动阻塞（move-blocked）控制输入、有限差分灵敏度、约束二次子问题以及输入可行性投影。一个在专家SCP解上训练的Transformer预测出初始控制序列，随后由与传统方法相同的优化器对其进行精炼。数值实验在相同的控制器设置下比较了零初始化、移位初始化和Transformer初始化。对于独立求解的MPC问题，Transformer初始化相对于移位初始化减少了计算时间……

    arXiv:2611.33514v1 Announce Type: cross  Abstract: This study investigates model predictive control (MPC) of a multirotor drone (unmanned aerial vehicle, UAV) carrying a cable-suspended payload, comparing conventional shifted initialization with Transformer-based warm starting of a sequential convex programming (SCP) optimizer. The controller uses a coupled mechanical model with an elastic cable and a double-pendulum representation of the suspended load. Its formulation includes move-blocked control inputs, finite-difference sensitivities, constrained quadratic subproblems, and input-feasibility projection. A Transformer trained on expert SCP solutions predicts an initial control sequence, which is subsequently refined by the same optimizer used for the conventional methods. Numerical experiments compare zero, shifted, and Transformer initialization under identical controller settings. For independently solved MPC problems, Transformer initialization reduces computation time relative t
    
[^198]: 学习实时推理

    Learning to Reason in Real-Time

    [https://arxiv.org/abs/2611.33513](https://arxiv.org/abs/2611.33513)

    该论文提出了实时推理训练（RRT）框架及三项标准技术——使用实时数据、解耦行动与观察、时间维度课程学习，使智能体学会根据时间成本权衡快速行动与深入推理，从而在实时环境中大幅超越传统同步训练方法。

    

    在现实世界中运作的AI智能体必须正确且及时地进行推理，然而目前它们并未考虑时间应如何改变其推理方式。现代智能体是在完全同步的环境中学习执行任务的：即世界会等待智能体采取行动之后才发生变化。相比之下，实时环境的变化则独立于智能体。现有方法试图通过强制规定思考与行动的比例来弥合这一差距。然而，要在实时环境中取得成功，智能体必须学会在延误代价高昂时迅速行动，并在收益超过错失机会的损失时进行更长时间的推理。为此，我们提出了实时推理训练（RRT）这一类针对实时推理智能体的训练技术。我们开发了三种标准的RRT技术：（1）使用实时数据，（2）解耦行动与观察，（3）时间维度的课程学习。这些简单的技术相结合，大幅超越了……

    arXiv:2611.33513v1 Announce Type: new  Abstract: AI agents that operate in the real world must reason correctly and on-time, yet they currently do not account for how time should alter their reasoning. Modern agents learn to do tasks in environments that are entirely synchronous: where the world waits for the agent to act before changing. A real-time environment, by contrast, changes independent of the agent. Existing approaches try to bridge this gap by imposing a ratio of thinking to acting. However, for an agent to succeed in real-time it must learn to act quickly when delay is costly and to reason longer when the benefit outweighs the missed opportunities. To this end, we introduce real-time reasoning training (RRT) as the class of training techniques for real-time reasoning agents. We develop three standard RRT techniques: (1) using real-time data, (2) decoupling acting and observing, and (3) curriculum learning in time. Together, these simple techniques substantially outperform s
    
[^199]: 面向涡旋脱落的无数据连续时间物理信息神经网络：跨时间窗口的一致压力

    Data-free continuous-time physics-informed neural networks for vortex shedding: a consistent pressure across time windows

    [https://arxiv.org/abs/2611.33489](https://arxiv.org/abs/2611.33489)

    该研究首次在完全无数据且无时间离散化的条件下，通过解析无散度初始条件打破对称性，在相继时间窗口上训练连续时间PINN实现了Re=100圆柱绕流的涡旋脱落，并将窗口交界处压力不连续归因于不可压缩Navier-Stokes方程指标为2的微分代数结构对一致初始条件的要求。

    

    在没有任何数据的情况下，连续时间的物理信息神经网络（PINNs）迄今未能在关于尾流中心线对称的计算域内发展出圆柱绕流的涡旋脱落现象：训练收敛于稳态对称解。我们通过在相继的时间窗口上训练连续时间PINN，在Re=100条件下获得了涡旋脱落，其中不使用任何参考数据、也不进行时间离散化，对称性由一个解析的、无散度的初始条件打破。斯特劳哈尔数与平均阻力系数与在同一计算域上计算的OpenFOAM解吻合，误差在0.5%以内。然而，压力在窗口交界处不连续，升力系数呈阶梯状。我们将这种不连续性与不可压缩Navier-Stokes方程的微分代数结构联系起来，该结构在空间离散化后为指标2：每个窗口的一致初始条件必须满足……

    arXiv:2611.33489v1 Announce Type: cross  Abstract: Without data, physics-informed neural networks (PINNs) in continuous time have so far failed to develop vortex shedding past a circular cylinder in a domain symmetric about the wake centerline: the training converges to the steady symmetric solution. We obtain shedding at Re=100 with a continuous-time PINN trained over successive time windows, without any reference data and without temporal discretization, the symmetry being broken by an analytical, divergence-free initial condition. The Strouhal number and the mean drag coefficient agree with an OpenFOAM solution computed on the same domain to within 0.5%. The pressure, however, is discontinuous at the junctions between windows, and the lift coefficient is a staircase. We relate this discontinuity to the differential-algebraic structure of the incompressible Navier-Stokes equations, of index 2 after spatial discretization: a consistent initial condition of each window must satisfy the
    
[^200]: 几何即记忆：面向持续学习LoRA的任务密度路由

    The Geometry Is the Memory: Task-Density Routing for Continual LoRA

    [https://arxiv.org/abs/2611.33487](https://arxiv.org/abs/2611.33487)

    CONTLORA利用冻结表示空间中的高斯类别模型实现任务密度路由来选择LoRA适配器，并通过累积特征矩精确恢复类别平衡岭分类器，在不保留任何图像或特征向量的情况下达到与存储样本方法几乎相同的持续学习性能。

    

    冻结旧的适配器并不能确保持续学习器在推理时能够正确地选择和使用它们。当存在多个适配器时，在它们之间进行路由，并识别出其表示空间与查询答案相匹配的适配器可能十分困难。我们提出CONTLORA，用紧凑的类别统计量替代存储的样本，用于适配器选择和分类。在冻结表示空间中的高斯类别模型用于选择LoRA适配器，而累积的特征矩对于固定的训练特征能够精确恢复出一个类别平衡的岭分类器。在不保留任何图像或特征向量的情况下训练，CONTLORA在20任务CIFAR-100上达到88.55%的准确率，在CUB-200上达到86.99%。在匹配组件研究的三个基准测试中，其平均最终准确率与独立训练的存储特征对应方法相比差异至多仅0.07个百分点。消融实验表明，任务覆盖率的提升并不一定带来分类性能的提升。

    arXiv:2611.33487v1 Announce Type: cross  Abstract: Freezing older adapters does not ensure that a continual learner will select and use them properly at inference time. When multiple adapters exist, routing among them and identifying the adapter whose representation space matches the query's answer can be difficult. We propose CONTLORA, which replaces stored examples with compact class statistics for adapter selection and classification. Gaussian class models in a frozen representation select LoRA adapters, while accumulated feature moments exactly recover a class-balanced ridge classifier for fixed training features. Trained without retaining images or feature vectors, CONTLORA reaches 88.55% accuracy on 20-task CIFAR-100 and 86.99% on CUB-200. Across three benchmarks in the matched component study, its mean final accuracy differs from an independently trained stored-feature counterpart by at most 0.07 percentage points. Ablations show that improved task coverage need not improve clas
    
[^201]: 论额外信息对老虎机学习的影响：更多并非总是更好

    On the Impact of Extra Information in Bandit Learning: More is Not Always Merrier

    [https://arxiv.org/abs/2611.33479](https://arxiv.org/abs/2611.33479)

    该研究揭示了一个反直觉现象：在随机老虎机学习中，额外信息可能使UCB算法的遗憾变差而非改善，并提出了一种能为广泛类别老虎机算法可证明地保证“非退化性”的元算法。

    

    通常认为更多的数据会提高学习性能，然而我们证明这一直觉在序列决策问题中可能出人意料地失效。在带有外生额外信息的随机老虎机设置下，我们探究免费且无偏的额外奖励样本是否必然能降低学习者的遗憾。我们引入了“额外信息下的非退化性”这一形式化概念，并证明广泛使用的置信上界算法（UCB）违反了这一性质，即存在某些良性的额外信息，反而使UCB产生严格高于无额外信息时的遗憾。我们将这种行为与汤普森采样和“先探索后承诺”算法进行对比，后者似乎能从相同的额外信息中受益。作为建设性贡献，我们提出了一种元算法，可证明地为一大类老虎机算法保证非退化性。我们还通过实证模拟来研究该元算法的表现。

    arXiv:2611.33479v1 Announce Type: cross  Abstract: More data is commonly assumed to improve learning performance, however, we show that this intuition can surprisngly fail in sequential decision-making. Considering a stochastic bandits setup with exogenous extra information, we ask if the free and unbiased extra reward samples are guaranteed to reduce the regret of the learner. We introduce a formal notion of `non-degradation under extra information' and demonstrate that the widely used Upper Confidence Bound algorithm (UCB) violates this property, i.e., there exist benign extra-information under which UCB incurs strictly higher regret than when run without extra information. We contrast this behaviour with Thompson Sampling and Explore-Then-Commit, which appear to benefit from the same extra information. Constructively, we propose a meta-algorithm that provably enforces non-degradation for a broad class of bandit algorithms. We also run empirical simulations to study our meta-algorith
    
[^202]: 记忆作为主干：一种面向视觉识别的双存储联想记忆架构

    Memory as the Backbone: A Dual-Store Associative Memory Architecture for Visual Recognition

    [https://arxiv.org/abs/2611.33475](https://arxiv.org/abs/2611.33475)

    提出了双存储联想记忆架构DuSAM，以工作记忆与持久语义记忆的记忆操作替代自注意力机制，构建出一种新型视觉主干网络，在ImageNet-1K分类、ADE20K分割和COCO检测上取得与同规模模型相当的性能。

    

    我们研究显式记忆能否作为通用视觉主干网络的组织原则。受人类认知中工作记忆与长期记忆相互作用的启发，我们提出了DuSAM（双存储联想记忆），这是一种层次化的记忆原生视觉主干网络，将图像局部的工作记忆（WM）与固定容量的持久语义记忆（PM）相耦合。工作记忆将局部视觉证据组织成紧凑的、空间锚定的状态，这些状态在各阶段之间被更新和传递，随后被读取回来以精炼密集特征。持久语义记忆将来自训练图像和标注区域的语义证据整合为原型，其检索到的内容用以丰富同一工作状态。通过用这些记忆操作取代自注意力或状态空间模块，DuSAM在ImageNet-1K分类、ADE20K语义分割和COCO目标检测等任务上，取得了与同等规模的代表性主干网络相当的性能。

    arXiv:2611.33475v1 Announce Type: cross  Abstract: We investigate whether explicit memory can serve as the organizing principle of a general-purpose visual backbone. Inspired by the interaction between working and long-term memory in human cognition, we introduce DuSAM (Dual-Store Associative Memory), a hierarchical memory-native visual backbone coupling image-local working memory (WM) with a fixed-capacity persistent semantic memory (PM). WM organizes local visual evidence into compact, spatially anchored states that are updated and transported across stages, then read back to refine dense features. PM consolidates semantic evidence from training images and annotated regions into prototypes, whose retrieved content enriches the same working states. With these memory operations in place of self-attention or state-space blocks, DuSAM achieves competitive performance relative to representative, similarly sized backbones on ImageNet-1K classification, ADE20K semantic segmentation, and COC
    
[^203]: 具有对抗性奖励的线性马尔可夫决策过程中的差分隐私策略优化

    Differentially Private Policy Optimization in Linear MDPs with Adversarial Rewards

    [https://arxiv.org/abs/2611.33469](https://arxiv.org/abs/2611.33469)

    本文针对具有未知转移和自适应对抗性奖励的线性MDP，提出了一种结合批量化PPO与乐观策略评估的差分隐私策略优化算法，在联合差分隐私下同时保护轨迹与奖励数据，并实现了保留 $\widetilde{O}\left(K^{3/4}\right)$ 主导项的高概率遗憾界。

    

    个性化服务中的强化学习（RL）通常依赖于敏感的交互数据。出于保护这些数据的需求，我们研究了在全信息反馈下、具有未知转移和自适应对抗性线性奖励的分段式线性马尔可夫决策过程（MDP）中的差分隐私（DP）策略优化。这一设定带来了两个关键挑战：过去的轨迹会被重复用于策略评估，且奖励的选择可能依赖于先前发布的私有统计量。所提出的算法将批量化的近端策略优化（PPO）更新与乐观策略评估相结合，利用私有回归统计量和奖励估计，在联合差分隐私（JDP）下同时保护轨迹和奖励函数。在所有其他参数固定的情况下，我们的高概率遗憾界保留了关于回合数的主要 $\widetilde{O}\left(K^{3/4}\right)$ 依赖关系。

    arXiv:2611.33469v1 Announce Type: new  Abstract: Reinforcement learning (RL) in personalized services often relies on sensitive interaction data. Motivated by the need to protect these data, we study differentially private (DP) policy optimization in episodic linear Markov decision processes (MDPs) with unknown transitions and adaptive adversarial linear rewards under full-information feedback. This setting poses two key challenges: past trajectories are repeatedly reused for policy evaluation, and reward choices may depend on previously released private statistics. The proposed algorithm combines batched proximal policy optimization (PPO) updates with optimistic policy evaluation, using private regression statistics and reward estimates to protect both trajectories and reward functions under joint differential privacy (JDP). With all other parameters fixed, our high-probability regret bound retains the same leading $\widetilde{O}\left(K^{3/4}\right)$ dependence on the number of episod
    
[^204]: 从循环中学习：在循环语言模型中内化深度诱导的偏好

    Learning from Recurrence: Internalizing Depth-Induced Preferences in Looped Language Models

    [https://arxiv.org/abs/2611.33468](https://arxiv.org/abs/2611.33468)

    本文提出“偏好回写”方法，利用循环语言模型在更大深度下自身生成的预测来训练共享块（无需参考答案），从而将深层计算带来的偏好内化到单遍推理中，同时提升单遍与多遍推理的准确率。

    

    arXiv:2611.33468v1 公告类型：新论文 摘要：循环语言模型在预测每个词元之前，会将同一个块重复应用于其隐藏状态，应用的次数称为其循环深度。增加循环深度可以在参数不变的情况下让模型为每个词元进行更多计算。共享块能否从它在更大深度下所做的预测中学习？遵循先前的工作，我们为预训练的 OLMo-2-1B 加装共享循环块，并在不同的循环深度下继续进行中期训练。在四遍模型中，额外的遍数主要是在一遍推理后已排名靠前的词元之间重新分配概率。我们利用在更大深度下生成的续写内容，仅通过单遍推理来训练共享块，且无需参考答案或正确性过滤。我们将这种转移方式称为“偏好回写”。偏好回写进一步提升了模型在大多数任务上的单遍和多遍准确率。在各自的目标深度下，更新后的模型优于参数量相当的非循环对照模型。

    arXiv:2611.33468v1 Announce Type: new  Abstract: A looped language model repeatedly applies the same block to its hidden state before predicting each token. The number of passes is its recurrent depth. Increasing this depth gives the model more computation per token with the same parameters. Can the shared block learn from the predictions it makes at greater depth? Following prior work, we retrofit pretrained OLMo-2-1B with a shared recurrent block and continue mid-training over varying recurrent depths. In the four-pass model, additional passes mainly redistribute probability among tokens that already rank highly after one pass. We use the continuations generated at greater depth to train only the shared block through a single pass, without reference answers or correctness filtering. We call this transfer preference write-back. Write-back further improves one-pass and multi-pass accuracy on most tasks. At their target depths, the updated models outperform a non-recurrent control match
    
[^205]: SplatWAM：用于实时3D想象的溅射世界动作模型

    SplatWAM: Splatting World Action Models for Real-Time 3D Imagination

    [https://arxiv.org/abs/2611.33467](https://arxiv.org/abs/2611.33467)

    SplatWAM是一种直接在3D高斯基元上学习动作条件动力学的生成式世界模型，通过分词器与动力学模型实现单次前向传播的3D状态预测，可从任意视角实时渲染，速度比视频世界模型快12倍且精度更高。

    

    世界模型通常在像素空间或潜空间中预测未来观测，这限制了它们直接表示真实3D场景演化的能力。我们提出了SplatWAM（溅射世界动作模型），这是一种生成式世界模型，直接在紧凑的3D高斯基元上学习动作条件下的动力学。分词器SplatWAM-T首先将可变数量的无位姿视角图像编码为固定预算的高斯token。随后，动力学模型SplatWAM-D针对每个动作通过单次前向传播预测未来的高斯状态，并利用对数欧几里得协方差坐标和可微渲染监督来促进在显式3D空间中的学习。每个预测状态都可以从任意视角进行光栅化渲染，从而在结构上保证了跨视角共享的3D表示。在OGBench Scene基准上，SplatWAM渲染新视角的速度比视频世界模型快12倍（在RTX 5090上约为80 FPS），同时PSNR最高提升2.0 dB。我们进一步引入了S……（摘要在此处被截断）

    arXiv:2611.33467v1 Announce Type: cross  Abstract: World models typically predict future observations in pixel or latent space, limiting their ability to directly represent true 3D scene evolution. We present SplatWAM (Splatting World Action Model), a generative world model that learns action-conditioned dynamics directly over compact 3D Gaussian primitives. A tokenizer SplatWAM-T first encodes a variable number of unposed views into a fixed budget of Gaussian tokens. A dynamics model SplatWAM-D then predicts future Gaussian states in a single forward pass per action, using log-Euclidean covariance coordinates and differentiable rendering supervision to facilitate learning in an explicit 3D space. Each predicted state can be rasterized from arbitrary viewpoints, enforcing a shared 3D representation across views by construction. On OGBench Scene, SplatWAM renders novel views 12x faster than video world models (~80 FPS on an RTX 5090) with up to 2.0 dB higher PSNR. We further introduce S
    
[^206]: PAC学习采样：论语言生成的计算复杂性

    PAC Learning to Sample: On Computational Aspects of Language Generation

    [https://arxiv.org/abs/2611.33464](https://arxiv.org/abs/2611.33464)

    该论文将PAC学习框架与极限语言生成模型相结合，首次为语言生成任务提供了关于样本复杂度和时间复杂度的定量保证。

    

    语言模型的成功推动了从离散域分布中学习采样的理论研究。一个重要的抽象模型是由Kleinberg和Mullainathan（NeurIPS 2024）提出的“极限语言生成”模型，其中算法必须在经过有限但未指定数量的正例后从目标语言中进行生成。虽然该模型的积极结果出奇地具有普适性，但大多数保证都是定性的。在这项工作中，我们受统计学习理论和计算学习理论的启发，寻求关于样本复杂度和时间复杂度的定量保证。为此，我们研究了由De、Diakonikolas和Servedio（SODA 2015）在另一背景下提出的语言生成PAC模型，其中定义在{-1,1}^n上的布尔概念类C定义了目标语言：给定未知概念f ∈ C的均匀随机正例，学习者必须输出一个采样器，其输出……

    arXiv:2611.33464v1 Announce Type: cross  Abstract: The success of language models has motivated theoretical work on learning to sample from distributions on discrete domains. A prominent abstraction is the model of language generation in the limit, introduced by Kleinberg and Mullainathan (NeurIPS 2024), in which the algorithm must generate from the target language after some finite but unspecified number of positive examples. While its positive results are surprisingly general, most guarantees are qualitative. In this work, we seek quantitative guarantees on sample and time complexity, inspired by statistical and computational learning theory.   To this end, we study a PAC model of language generation, proposed (in a different context) by De, Diakonikolas, and Servedio (SODA 2015), in which a Boolean concept class $C$ over $\{-1,1\}^n$ defines the target languages: given uniformly random positive examples of an unknown concept $f \in C$, the learner must output a sampler whose output 
    
[^207]: 基于可微分世界模型的快速肌肉驱动全身运动跟踪

    Fast Muscle-Driven Whole-Body Motion Tracking with a Differentiable World Model

    [https://arxiv.org/abs/2611.33460](https://arxiv.org/abs/2611.33460)

    该论文提出一种基于可微分世界模型的快速肌肉驱动全身运动跟踪方法，通过四项训练改进将 MS-Human-700 的训练时间从 3.9 小时缩短至 6.3 分钟，实现 37 倍加速。

    

    使用全身肌骨骼模型进行动作捕捉跟踪意味着需要通过接触来协调数百块肌肉，而现有的控制器需要数小时甚至数天的GPU训练时间，导致每一次调参决策都非常昂贵。SuperMimic通过在学习到的世界模型推演过程上进行反向传播来训练跟踪策略。我们将其扩展到 MS-Human-700（700块肌肉）和 MyoFullBody（354块肌肉），并将 MuJoCo 的激活动力学纳入世界模型中。为了实现快速训练，我们的方案借鉴了为 Unitree G1 开发的加速技术，替换了其中三项在肌肉模型上失效的设置，并新增四项改进：不包含肌肉几何信息的观测、数据采集与更新同步进行、确保每次激励保持在 [0,1] 区间的 Beta 探索噪声，以及 50 Hz 控制。在一块 RTX 5090 上，我们的方法仅需 6.3 分钟即可将 MS-Human-700 训练到 SuperMimic 公开超参数需要 3.9 小时才能达到的保留集回报，实现了 37 倍的加速。

    arXiv:2611.33460v1 Announce Type: new  Abstract: Tracking motion capture with a whole-body musculoskeletal model means coordinating hundreds of muscles through contact, and current controllers take hours to days of GPU time to train, so every tuning decision is expensive. SuperMimic trains a tracking policy by backpropagating through the rollouts of a learned world model. We extend it to MS-Human-700 (700 muscles) and MyoFullBody (354 muscles), with MuJoCo's activation dynamics inside the world model. To make its training fast, our recipe takes the speedups developed for the Unitree G1, replaces three of their settings that fail on muscles, and adds four changes of our own: observations without muscle geometry, data collection concurrent with the updates, Beta exploration noise that keeps every excitation in $[0,1]$, and 50 Hz control. On one RTX 5090 it trains MS-Human-700 in 6.3 minutes to the held-out return that SuperMimic's published hyperparameters reach in 3.9 hours, 37$\times$ 
    
[^208]: 面向大规模机器学习模型量子增强训练的端到端运行时间交叉点现实估算

    A Realistic End-to-End Runtime Crossover Estimation for Quantum-Enhanced Training of Large-Scale Machine Learning Models

    [https://arxiv.org/abs/2611.33454](https://arxiv.org/abs/2611.33454)

    该论文对用量子线性求解器训练线性化神经网络进行了涵盖数据访问、计算与读出的端到端现实资源估算，发现量子方法相对于经典方法的优势交叉点仅在问题规模极大、计算时间长且输出要求严苛时才会出现。

    

    机器学习（ML）是量子计算（QC）加速的一个有前景的应用领域，但渐近复杂度的考量往往掩盖了实际端到端工作流程的真实工程成本。针对使用量子线性求解器训练线性化神经网络的应用场景，已有研究提出了可证明具有优势的量子扩展性。我们针对这一问题提出了现实的资源估算，涵盖了数据访问、计算和读出等环节。我们的分析在三种不同的稀疏性扩展假设下，考察了三个复杂度递增的基准测试场景，从而实现逐步更加贴近现实的资源估算。我们识别了相对于经典方法的交叉区域，并发现交叉点仅在问题规模极大、计算时间很长以及输出要求极为严苛的情况下才会出现。这项工作为在现实硬件条件下评估量子增强的机器学习算法提供了蓝图。

    arXiv:2611.33454v1 Announce Type: cross  Abstract: Machine Learning (ML) is one promising application domain for Quantum Computing (QC) speedups, but asymptotic complexity considerations often mask the actual engineering costs of actual end-to-end workflows. For the use case of training a linearized Neural Network with a Quantum Linear Solver, a provably favorable quantum scaling has been suggested. We present realistic resource estimates for this problem, accounting for data access, computation, and readout. Our analysis examines three increasingly complex benchmarking scenarios, under distinct sparsity-scaling assumptions, thereby enabling progressively more realistic resource estimates. We identify crossover regimes relative to classical methods and find that crossover points emerge only for very large problem sizes, long computation times, and highly restrictive output requirements. This work provides a blueprint for evaluating quantum-enhanced ML algorithms under realistic hardwar
    
[^209]: HumblePRM：通过过程奖励建模提升视觉语言模型在模糊情境下的弃权能力

    HumblePRM: Improving VLM Abstention in Ambiguous Situations through Process Reward Modeling

    [https://arxiv.org/abs/2611.33452](https://arxiv.org/abs/2611.33452)

    提出轻量级过程奖励模型HumblePRM，将多模态推理分为感知与校准两阶段独立评分，通过奖励证据与置信度的一致性而非仅奖励结果正确性，提升视觉语言模型在模糊情境下的弃权能力。

    

    视觉语言模型（VLM）的可靠部署要求模型表达的置信度必须基于其实际能看到的内容，并且能够识别视觉证据过于模糊而无法给出确信回答的情况。标准的后训练方法使用结果奖励模型（ORM），仅监督最终决策，无论图像是否支持某个答案，都会奖励自信的回答；而现有方法要么倾向于事后检测不确定性，要么假设总是存在客观的判定结论。我们转而将多模态推理分为两个阶段：感知阶段，模型必须事实性地描述场景；校准阶段，模型必须判断该证据是否足以得出结论。我们提出了HumblePRM，这是一个轻量级的过程奖励模型（PRM），能够独立地对每个阶段进行评分，奖励证据与所表达置信度之间的一致性，而不仅仅是奖励结果的正确性。

    arXiv:2611.33452v1 Announce Type: cross  Abstract: Reliable deployment of Vision Language Models (VLMs) requires that a model's expressed confidence remain grounded in what it can actually see, and that it recognize when the visual evidence is too ambiguous to commit to a confident response. Standard post-training uses outcome reward models (ORMs) that supervise only the final decision, rewarding a confident answer regardless of whether the image supports one, while existing methods either tend to detect uncertainty post-hoc or assume an objective verdict always exists. We instead split multimodal reasoning into two stages: perception, where the model must factually describe the scene, and calibration, where it must judge whether that evidence is sufficient to commit to a conclusion. We introduce HumblePRM, a lightweight process reward model (PRM) that scores each stage independently, rewarding agreement between evidence and expressed confidence rather than outcome correctness alone. T
    
[^210]: 中心核距离：用于比较神经表征的无维度依赖误差界

    Centered Kernel Distance: Dimension-Free Error Bounds for Comparing Neural Representations

    [https://arxiv.org/abs/2611.33447](https://arxiv.org/abs/2611.33447)

    本文提出中心核距离（CKD）这一神经表征相似性度量，证明其存在无偏估计量且估计误差以 1/M + 1/N 的速度衰减，同时收敛速率与实验条件空间维度无关，从而克服了表征相似性估计中的维度灾难。

    

    中心核对齐（CKA）和表征相似性分析（RSA）等表征相似性度量的渐近误差已被充分刻画，但对于表征相似性的估计如何同时依赖于采样神经元数量（N）和实验条件数量（M），目前尚缺乏符合实际的误差界。我们研究了CKA和RSA的自然度量对应物，称之为中心核距离（CKD）。我们证明CKD存在无偏估计量，而包括CKA在内的几种流行的相似性度量则不存在无偏估计量。在温和的假设条件下，我们证明CKD的估计误差以 1/M + 1/N 的比例衰减。引人注目的是，该误差界中的收敛速率不依赖于实验条件空间的维度。因此，相似性估计不会受到维度灾难的影响，而维度灾难正困扰着高维统计中的类似问题。

    arXiv:2611.33447v1 Announce Type: cross  Abstract: The asymptotic error of representational similarity metrics such as centered kernel alignment (CKA) and representational similarity analysis (RSA) has been well-characterized, but there are no realistic bounds on how estimates of representational similarity $\textit{simultaneously}$ depend on the number of sampled neurons ($N$) and experimental conditions ($M$). We study the natural metric counterpart of CKA and RSA, which we call the Centered Kernel Distance (CKD). We show that CKD admits an unbiased estimator, whereas several popular similarity measures including CKA have no unbiased estimator. Under mild assumptions, we show that estimation error of CKD decays in proportion to $1/M + 1/N$. Strikingly, the rate in this bound does not depend on the dimensionality of the space of experimental conditions. Thus, similarity estimation does not suffer from the curse of dimensionality, which plagues similar problems in high-dimensional stat
    
[^211]: ViCARD：面向多模态推理的基于奖励门控蒸馏的视觉信用分配方法

    ViCARD: Visual Credit Assignment with Reward-Gated Distillation for Multimodal Reasoning

    [https://arxiv.org/abs/2611.33437](https://arxiv.org/abs/2611.33437)

    ViCARD通过奖励门控蒸馏，将逐token的视觉监督与经过验证的正确性奖励相结合，同时解决了基于结果的强化学习和视觉自蒸馏在多模态推理中的互补性失败模式。

    

    视觉推理要求模型能够识别相关的视觉证据，并利用这些证据推导出正确的答案。基于结果的强化学习（RL）在这一场景中面临两个关键挑战：当感知错误导致所有采样响应均失败时，它无法产生任何学习信号；同时它会在响应内的每个token上广播统一的advantage（优势值）。相反，虽然视觉自蒸馏能够通过以证据为中心的图像裁剪提供密集的指导，但仅从token分布中学习的学生模型往往无法得出正确的最终答案。基于对训练rollout的审计所揭示的这些互补性失败模式，我们提出了ViCARD（带奖励门控蒸馏的视觉信用分配），它将逐token的视觉监督与经过验证的正确性奖励相结合。对于每个rollout组，我们利用一个输入图像被裁剪至相关视觉区域的教师模型，并使用它来提供逐token的监督。

    arXiv:2611.33437v1 Announce Type: cross  Abstract: Visual reasoning requires models to identify relevant visual evidence and use it to derive correct answers. Outcomebased reinforcement learning (RL) faces two key challenges in this regime: it yields no learning signal when perceptual errors cause all sampled responses to fail, and it broadcasts a uniform advantage across every token within a response. Conversely, while visual self-distillation offers dense guidance from evidence-centered image crops, student learning from token distributions alone often fails to elicit correct final answers. Motivated by an audit of training rollouts that exposes these complementary failure modes, we introduce ViCARD (Visual Credit Assignment with RewardGated Distillation), which integrates per-token visual supervision with verified correctness rewards. For each rollout group, we leverage a teacher with its input image cropped to the relevant visual area, and use this to provide per-token supervision 
    
[^212]: 手术刀与画笔：利用单个大语言模型智能体对小语言模型进行剪枝与针对性权重编辑

    Scalpel and Brush: Pruning and Targeted Weight Editing of a Small Language Model by a Single LLM Agent

    [https://arxiv.org/abs/2611.33426](https://arxiv.org/abs/2611.33426)

    该论文通过严格的对照实验首次系统地证明，单个LLM智能体能够通过稀疏权重编辑（针对性修复、结构化剪枝与层移除）改进小语言模型，仅修改0.0084%的权重即可在常微分方程任务上使BLEU提升24-32%乃至2.5倍，并排除了随机编辑等因素的干扰。

    

    近期有研究表明，大语言模型（LLM）本身可以编辑语言模型的权重，但目前仍不清楚这类结果中哪些部分归功于智能体，哪些部分可以被相同形状的随机编辑、提示词中的指令、或不使用LLM的标准流程所复现。我们将修复与剪枝统一为稀疏权重编辑，以优化我们指定的指标（以BLEU为例）：三项任务——针对性权重修复（“画笔”）、结构化MLP神经元剪枝（“手术刀”，将被剪枝的权重置零）以及层移除并修复——被分配给由DeepSeek-V4.1-Flash驱动的DeepSeek Harness智能体，最多进行50次开发集评估。我们以每个任务三次独立运行的方式研究Qwen2.5-Math-1.5B-Instruct模型，并在两个隐藏的数据划分上进行最终评估。在常微分方程任务上，仅改动最多0.0084%的权重，就在二阶方程上将测试BLEU提升了24-32%，在含多项式项的方程上提升了1.8-2.5倍。

    arXiv:2611.33426v1 Announce Type: new  Abstract: It has recently been shown that an LLM can itself edit the weights of a language model, yet it remains unclear which part of such a result is due to the agent and which part is reproduced by a random edit of the same shape, by an instruction in the prompt, or by a standard procedure without an LLM. We unify repair and pruning as sparse weight edits to optimize a metric we specify, using BLEU as an example: three tasks -- targeted weight repair ("brush"), structured MLP neuron pruning ("scalpel", zeroing pruned weights), and layer removal with repair - are assigned to a DeepSeek Harness agent powered by DeepSeek-V4.1-Flash, with up to 50 dev evaluations. We study Qwen2.5-Math-1.5B-Instruct in three independent runs per task, with final evaluation on two hidden splits. On ordinary differential equations, changing at most 0.0084% of weights raised test BLEU by 24-32% on second-order equations and by a factor of 1.8-2.5 on equations with pol
    
[^213]: 又一个概念漂移基准？一个用于评估漂移检测器的原则性框架

    Yet Another Concept Drift Benchmark? A Principled Framework for Evaluating Drift Detectors

    [https://arxiv.org/abs/2611.33424](https://arxiv.org/abs/2611.33424)

    该论文提出了一个基于漂移可观测性和警报语义显式形式化的原则性概念漂移检测基准框架，通过引入有限检测窗口、重复警报和标准化综合评分指标，实现了检测准确性、及时性与稳定性的清晰分离，并通过对10个检测器在9种学习器配置和16个数据集上的大规模压力测试验证了框架的有效性。

    

    概念漂移检测是数据流学习的基本组成部分，然而其经验评估仍然基于定义松散的协议、含糊不清的警报语义和不一致的度量指标。我们提出了一个基于漂移可观测性和警报语义显式形式化的原则性基准。我们的框架引入了有限检测窗口和重复警报机制，实现了检测准确性、及时性和稳定性之间的清晰分离，并采用可复现的观察器提供受控且可解释的性能信号。基于这一形式化，我们定义了标准化的评估指标和一个综合评分，以解决现有度量方法的局限性。我们通过对10个检测器在9种学习器配置和16个数据集上进行的大规模压力测试评估来验证该框架。

    arXiv:2611.33424v1 Announce Type: new  Abstract: Concept drift detection is a fundamental component of data stream learning, yet its empirical evaluation remains based on loosely defined protocols, ambiguous alarm semantics, and inconsistent metrics. We propose a principled benchmark grounded in an explicit formulation of drift observability and alarm semantics. Our framework introduces finite detection windows and repeated alarms, enabling a clear separation between detection accuracy, timeliness, and stability, and employs a reproducible observer to provide a controlled and interpretable performance signal. Building on this formulation, we define normalized evaluation metrics and a composite score addressing limitations of existing measures. We validate the framework through a large-scale stress-testing evaluation of 10 detectors across 9 learner configurations and 16 datasets.
    
[^214]: 面向三维分子设计的强化学习引导连续-离散流匹配的时间选择性探索方法

    Temporal Selective Exploration for Reinforcement Learning-Guided Continuous-Discrete Flow Matching in 3D Molecular Design

    [https://arxiv.org/abs/2611.33417](https://arxiv.org/abs/2611.33417)

    该论文提出了一种带时间选择性探索的强化学习引导连续-离散流匹配框架，通过仅在关键时间步选择性注入随机性以减少冗余探索噪声，并平衡多目标优化中的更新不均问题，从而实现高效的三维从头分子设计。

    

    流匹配在生成任务中展现出了强大的潜力，然而通过强化学习（RL）对其进行优化仍然缺乏深入研究，尤其是在连续-离散混合的设置中。此外，流匹配被表述为常微分方程（ODE），其确定性轨迹并不天然支持强化学习优化。将整个轨迹转化为随机微分方程（SDE）以引入随机性是一种常见做法，但并非所有时间步都对最终结果有贡献，冗余探索会引入额外噪声。而且，在多目标设置中，某些目标可能主导优化过程，而其他目标则得不到充分的更新。在本研究中，我们提出了一种带有时间选择性探索的强化学习引导连续-离散流匹配框架，用于三维从头分子设计。具体而言，我们的方法联合优化连续和离散……（原文摘要在此处截断）

    arXiv:2611.33417v1 Announce Type: cross  Abstract: Flow matching has shown strong potential in generative tasks, while its optimization via Reinforcement Learning (RL) remains underexplored, especially in continuous-discrete mixed settings. Moreover, flow matching is formulated as an Ordinary Differential Equation (ODE), whose deterministic trajectories do not naturally support RL optimization. Introducing stochasticity by converting the entire trajectory into a Stochastic Differential Equation (SDE) is a common practice. Nevertheless, not all timesteps contribute to the final outcomes, and redundant exploration introduces additional noise. Furthermore, in multi-objective settings, some objectives may dominate the optimization while others receive insufficient updates. In this study, we propose an RL-guided continuous-discrete flow matching framework with temporal selective exploration for 3D de novo molecular design. Specifically, our method jointly optimizes continuous and discrete f
    
[^215]: 一种基于冻结编码器与稳定化强化学习的“语义优先”多模态视频描述方法

    A Semantics-First Approach to Multimodal Video Captioning with Frozen Encoders and Stabilized Reinforcement Learning

    [https://arxiv.org/abs/2611.33409](https://arxiv.org/abs/2611.33409)

    提出AVQFormer架构，在27亿参数LLM上通过固定长度语义对齐模块与稳定化强化学习（ASCST）实现高效多模态视频描述，并揭示全局音频表征与视觉帧时序冲突导致的“时间对齐摩擦”现象。

    

    我们提出了音视频查询Transformer（AVQFormer），这是一种基于27亿参数大语言模型构建的高效架构，其核心特点是采用固定长度K=16的语义对齐模块，并通过带熵正则化的自适应自我批判序列训练（ASCST）实现稳定化。多模态大语言模型通常依赖全局音频表征来弥补低效的基于帧的视觉编码器。虽然我们的第一阶段提示评估表明，该多模态基础模型实现了强大的深层语义对齐（BERTScore F1得分0.889），但在针对传统指标优化该架构时暴露出性能差距。我们将这一差距作为假设进行探讨，称之为“时间对齐摩擦”：即全局音频表征破坏视觉帧的时间顺序，导致在以视觉为中心的评估指标下受到惩罚的一种冲突。因此，通过剥离音频，我们的纯视频消融（实验）……

    arXiv:2611.33409v1 Announce Type: cross  Abstract: We propose the Audio-Visual Query Transformer (AVQFormer), a highly efficient architecture built upon a 2.7B parameter LLM featuring a fixed-length $K=16$ semantic alignment module, stabilized via Adaptive Self-Critical Sequence Training (ASCST) with Entropy Regularization. Multimodal Large Language Models (LLMs) often rely on global audio representations to compensate for inefficient, frame-based visual encoders. While our Phase 1 prompted evaluations demonstrate that the Multimodal Base Model achieves strong deep semantic alignment (scoring 0.889 on BERTScore F1), optimizing this architecture for legacy metrics reveals a performance gap. We explore this gap as a hypothesis we term temporal alignment friction: a conflict where global audio representations disrupt the chronological sequence of visual frames, leading to penalties under visually-focused evaluation metrics. Consequently, by stripping away the audio, our Video-Only ablatio
    
[^216]: 用于运动敏捷性预训练的对抗性多智能体追捕游戏

    An Adversarial Multi-Agent Pursuit Game for Locomotor Agility Pretraining

    [https://arxiv.org/abs/2611.33407](https://arxiv.org/abs/2611.33407)

    该论文提出通过在物理模拟器中让神经网络玩多智能体追逐游戏进行竞争性自我博弈，利用稀疏反馈使新的运动生成与控制策略得以涌现，从而实现人形机器人运动敏捷性的预训练。

    

    敏捷的人形机器人运动需要控制器既能适应快速变化的运动需求，又能保持稳定和鲁棒。现有的运动控制器通常采用两种设计方式：一种是精心设计的密集奖励，用于鼓励迈步并惩罚摔倒；另一种是使用参考轨迹跟踪奖励，教控制器跟随预定的动作。一旦发现能够最大化奖励的行为，控制器就会停止学习。参考跟踪方法虽然总体上稳定，但将运动控制与运动生成分离开来处理。自我博弈已被证明是一种有效的技术，可以在围棋、星际争霸、捉迷藏和夺旗等多种多智能体游戏中实现超越人类的性能。我们训练神经网络在物理模拟器中玩多智能体追逐游戏，以研究新的运动生成和控制策略能否通过稀疏反馈和竞争性自我博弈而涌现。

    arXiv:2611.33407v1 Announce Type: cross  Abstract: Agile humanoid locomotion requires controllers that can adapt to rapidly changing motion demands while remaining stable and robust. Recent locomotion controllers are typically designed with either a dense reward that is carefully shaped to encourage steps and penalize falling, or by using a reference tracking reward which teaches the controller to follow predetermined motions. Once a behavior that maximizes the reward is discovered, the controller ceases to learn. While generally stable, reference tracking methods focus on motor control separately from motion generation. Self-play has been proven as an effective technique for superhuman performance on several multiagent games like Go, Starcraft, hide-and-seek, and capture-the-flag. We train neural networks to play the game of multiagent tag in a physics simulator to study if new motion generation and control strategies can emerge through sparse feedback and competitive self-play. We pr
    
[^217]: 基于半经典近似从非线性偏微分方程构造高斯先验

    Gaussian priors from nonlinear PDEs with the semiclassical approximation

    [https://arxiv.org/abs/2611.33400](https://arxiv.org/abs/2611.33400)

    本文提出利用半经典近似，通过在稳定解附近展开非线性PDE的变分能量或积分平方残差来构造高斯先验，从而将贝叶斯不确定性量化的概率数值方法拓展到非线性偏微分方程领域。

    

    arXiv:2611.33400v1 公告类型：cross 摘要：偏微分方程（PDE）的概率数值方法将控制方程的物理规律与观测数据相结合，把求解PDE视为一个贝叶斯逆问题。这使得人们能够在模型形式误差存在的情况下量化场重建中的不确定性。在大多数情况下，会给感兴趣的场分配一个高斯先验，并在无穷维设置下利用贝叶斯定理推导后验分布。虽然高斯分布已被证明非常适合线性PDE，但对非线性PDE的类似处理仍是一个悬而未决的问题。在本工作中，我们研究利用半经典近似，从变分能量或积分平方残差中构造用于概率数值方法的高斯先验。该先验通过在稳定解附近对PDE能量进行展开并揭示出高斯形式而得到。随后，我们分析了所得先验的不同特性，以及它们在……（摘要原文在此处截断）

    arXiv:2611.33400v1 Announce Type: cross  Abstract: Probabilistic numerical methods for partial differential equations (PDE) combines the governing physics with observations to treat solving the PDE as a Bayesian inverse problem. This allows techniques to quantify the uncertainty in the field reconstruction under model-form error. In the majority of cases, a Gaussian prior is assigned to the field of interest, and Bayes's theorem in the infinite-dimensional setting is used to derive the posterior. While Gaussians have been shown to be well-suited for linear PDEs, a similar treatment for nonlinear PDEs remains open-ended. In this work, study the use of the semiclassical approximation to construct Gaussian priors from variational energies or integrated square residuals for probabilistic numerics. The prior is derived by expanding the PDE energy about a stable solution and uncovering a Gaussian form. We then analyze the different characteristics of the resulting priors as they pertain to t
    
[^218]: 顺序获取概念知识以指导持续学习

    Sequentially Acquiring Concept Knowledge to Guide Continual Learning

    [https://arxiv.org/abs/2611.33399](https://arxiv.org/abs/2611.33399)

    该论文提出SACK——一种即插即用的持续学习技术，通过根据新样本与模型当前知识的语义关系对其进行评分和排序，从而指导持续学习过程。

    

    持续学习（CL）的目标是在适应新数据（可塑性）的同时保留从旧数据中获取的知识（稳定性）。现有方法明确设计用于平衡稳定性和可塑性，以减轻灾难性遗忘的挑战，同时促进学习。然而，网络训练所使用的新数据样本的顺序和性质所带来的影响仍然是一个未被充分探索的因素。我们认为，持续学习算法应该有能力根据传入样本与先前数据的关系及其对学习过程的影响来对这些样本进行排序。在这项工作中，我们研究了根据传入数据与模型当前知识的语义关系对其进行评分和排序是否能够有益于持续学习。我们提出了SACK（Sequentially Acquiring Concept Knowledge，顺序获取概念知识），这是一种用于持续学习方法的即插即用技术。SACK将模型的类别知识分解为细粒度的……（摘要原文在此处截断）

    arXiv:2611.33399v1 Announce Type: cross  Abstract: The goal of continual learning (CL) is to adapt to new data (plasticity) while retaining the knowledge acquired from old data (stability). Existing methods are explicitly designed to focus on balancing stability and plasticity to mitigate the challenge of catastrophic forgetting while promoting learning. However, the impact of order and nature of samples from the new data that the network is trained on remains an under explored factor. We argue that a CL algorithm should have the ability to rank incoming samples in terms of their relationship with prior data and their effect on the learning process. In this work, we investigate if scoring and prioritizing incoming data based on their semantic relationships with the model's current knowledge can benefit CL. We propose SACK, short for Sequentially Acquiring Concept Knowledge, a plug and play technique for continual learning methods. SACK dissects categorical knowledge of the model into f
    
[^219]: Best-of-$N$ 测试时引导可能导致视觉-语言-动作模型中的奖励黑客行为

    Best-of-$N$ Test-Time Steering Can Lead to Reward Hacking in Vision-Language-Action Models

    [https://arxiv.org/abs/2611.33398](https://arxiv.org/abs/2611.33398)

    本文揭示了当验证器不准确时，对VLA模型采用best-of-$N$测试时引导会随着N的增大引发奖励黑客行为，即被选中的动作候选利用了验证器的预测误差而非真正提升任务性能。

    

    视觉-语言-动作（VLA）模型在通用机器人操作方面已展现出前景，但适应不断变化的环境条件仍然具有挑战性。近期的测试时引导方法通过在部署期间利用额外的计算来实现自适应的VLA策略，显示出了潜力。本文中，我们研究了针对VLA模型的best-of-$N$测试时引导方法，即在部署时从固定策略中采样多个动作候选，并使用验证器选择最佳的一个。我们表明，当验证器不准确时，增大$N$并不总能提升性能，反而可能偏向那些分数利用了预测误差的候选，这是一种形式的奖励黑客行为（reward hacking）。我们使用$\pi_{0.5}$策略在60个LIBERO-Safety任务级案例上研究了这一效应，每个案例重复十次。我们在MuJoCo轨迹验证器中引入受控误差，并表明随着$N$增大，所选候选的奖励……

    arXiv:2611.33398v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models have shown promise for general-purpose robotic manipulation, but adapting to changing environmental conditions still remains challenging. Recent test-time steering methods have shown promise in enabling adaptive VLA policies by utilizing additional computation during deployment. In this paper, we consider best-of-$N$ test-time steering for VLA models, where multiple action candidates are sampled from a fixed policy and a verifier is used to select the best one during deployment. We show that increasing $N$ does not always improve performance when the verifier is inaccurate, and can favor candidates whose scores exploit prediction errors, a form of reward hacking. We study this effect on 60 LIBERO-Safety task-level cases using a $\pi_{0.5}$ policy, with ten repetitions per case. We introduce controlled errors in a MuJoCo trajectory verifier and show that as $N$ increases, selected candidate rewards di
    
[^220]: 基于伪标签交换的不确定性量化联邦学习

    Federated Learning with Uncertainty Quantification on Pseudo Label Exchange

    [https://arxiv.org/abs/2611.33395](https://arxiv.org/abs/2611.33395)

    本文提出FedSON两阶段框架，通过交换本地神经算子生成伪标签辅助数据，解决了联邦随机算子学习中客户端数据分布异构（尤其是输入函数支撑互不相交）导致的性能下降问题，并实现了不确定性量化。

    

    在这项工作中，我们研究了针对随机偏微分方程（SPDEs）的联邦随机算子学习，重点关注客户端数据分布的异构性和不确定性量化。联邦学习使各客户端能够在不交换原始解数据的情况下协作训练模型，但传统的联邦平均方法在客户端分布存在差异时会遭遇显著的性能下降，尤其是当各客户端的输入函数支撑互不相交时。为应对这些挑战，我们提出了联邦随机算子网络，这是一个两阶段框架，将辅助数据构建与随机算子学习相结合。在第一阶段，通过交换本地训练的确定性神经算子及其对应的输入函数来生成伪标签辅助数据，使联邦模型能够在异构输入分布上进行学习，而无需

    arXiv:2611.33395v1 Announce Type: new  Abstract: In this work, we investigate federated stochastic operator learning for stochastic partial differential equations (SPDEs), with an emphasis on heterogeneous client data distributions and uncertainty quantification. Federated learning enables clients to collaboratively train models without exchanging their original solution data, but conventional federated averaging can suffer significant performance degradation when client distributions differ, particularly when their input function supports are disjoint. To address these challenges, we propose the Federated Stochastic Operator Network (FedSON), a two-phase framework that combines auxiliary data construction with stochastic operator learning. In the first phase, locally trained deterministic neural operators and their corresponding input functions are exchanged to generate pseudolabeled auxiliary data, enabling the federated model to learn across heterogeneous input distributions without
    
[^221]: 用于大语言模型门控自动化的置信度分数

    Confidence scores for gated automation of large language models

    [https://arxiv.org/abs/2611.33394](https://arxiv.org/abs/2611.33394)

    该论文提出利用监督学习在冻结LLM的最终层隐藏状态上拟合线性置信度分数头（并为闭源模型训练仅文本的元模型），以预测LLM答案的正确性，从而实现高置信度回答的自动化处理和不确定回答的人工审查分流。

    

    基于大语言模型（LLM）构建的AI智能体在自动化决策方面展现出强大的能力。然而，LLM可能会以高置信度产生错误的答案，这限制了其在影响个人和组织的决策中的适用性。可靠的置信度分数对于对此类决策进行门控是必要的，而现有的估计方法通常不可用或过于粗糙，无法支持有效的门控。本工作使用监督学习训练模型来预测LLM生成答案的正确性，使得高置信度的回答可以自动化处理，而不确定的回答则被转交人工审查。该方法在冻结的LLM骨干网络的最终层隐藏状态上拟合一个线性置信度分数头，并训练仅基于文本的元模型作为闭源模型的替代方案。所有结果均来自单一的20亿参数骨干网络，在约98%为SQuAD、其余为TruthfulQA的评分答案上进行。与多个基线方法相比，该置信度（摘要在此处截断）

    arXiv:2611.33394v1 Announce Type: new  Abstract: AI agents built on large language models (LLMs) demonstrate strong capabilities in automated decision-making. However, LLMs may produce incorrect answers with high confidence, limiting their applicability in decisions that affect individuals and organizations. Reliable confidence scores are necessary to gate such decisions, and existing estimates are frequently unavailable or too coarse to support effective gating. This work uses supervised learning to train models to predict the correctness of LLM-generated answers, enabling automated confident responses and routing uncertain ones for human review. A linear confidence score head is fitted to the final-layer hidden state of the frozen LLM backbone, and text-only metamodels are trained as an alternative for closed models. All results come from a single 2B-parameter backbone, on graded answers that are about 98% SQuAD and the remainder TruthfulQA. Against multiple baselines, the confidence
    
[^222]: 短视频片段能否验证模型修复？

    Can a Short Video Clip Verify a Model Fix?

    [https://arxiv.org/abs/2611.33393](https://arxiv.org/abs/2611.33393)

    研究表明，短视频片段不能可靠地验证视频模型的修复：仅凭片段通过就判定问题已解决会造成误判，而要求片段通过又会遗漏近四分之一的真实修复。

    

    在短视频片段上复现了视频模型错误的开发者，可以在模型更新后重放该片段。一个正确的片段答案在多大程度上意味着原始测试现已通过？在960个预先指定的CG-Bench问题上，我们从Qwen2-VL-2B-Instruct更新到了Qwen2.5-VL-7B-Instruct，覆盖完整视频的64帧样本上的准确率从27.9%提升至33.3%。在给新模型评分之前，我们冻结了388个旧模型复现了其确切参考错误的片段。更新后的模型在116次片段通过中通过了51次参考测试（44.0%，95% Wilson置信区间为35.3–53.0%），在272次片段失败中通过了16次（5.9%）。如果仅凭片段通过就关闭问题，将会宣告65个尚未解决的参考失败已被修复；而如果要求片段通过，则会遗漏67个参考修复中的16个（23.9%）。在另一项包含1,028个视频、使用不同检查点和直接选项评分的独立研究中，仅有15个片段通过，且16帧参考只确认了其中一个。

    arXiv:2611.33393v1 Announce Type: cross  Abstract: A developer who reproduces a video model's error on a short clip can replay that clip after an update. How often does a correct clip answer mean the original test now passes? On 960 prespecified CG-Bench questions, we updated from Qwen2-VL-2B-Instruct to Qwen2.5-VL-7B-Instruct. Accuracy on a 64-frame sample spanning the full video rose from 27.9% to 33.3%. Before scoring the newer model, we froze 388 clips on which the older model reproduced its exact reference error. The updated model passed the reference on 51 of 116 clip passes (44.0%, 95% Wilson interval 35.3-53.0%) and on 16 of 272 clip failures (5.9%). Closing cases on a clip pass would have declared 65 unresolved reference failures fixed. Requiring a clip pass would have missed 16 of the 67 reference fixes (23.9%). In a separate 1,028-video study with different checkpoints and direct option scoring, only 15 clips passed, and the 16-frame reference confirmed one. For the generate
    
[^223]: 中子诱发裂变中总动能分布的概率机器学习研究

    Probabilistic Machine Learning for Total Kinetic Energy Distributions in Neutron-Induced Fission

    [https://arxiv.org/abs/2611.33379](https://arxiv.org/abs/2611.33379)

    该论文利用高斯过程回归和（贝叶斯）混合密度网络等概率机器学习模型，系统评估了其重现中子诱发裂变总动能分布宽度（尤其在高入射中子能量下）以及在训练范围之外外推的能力。

    

    裂变后中子发射的总动能（TKE）分布对入射中子能量的依赖性，特别是其在多机会裂变区域的宽度，为裂变计算提供了一个敏感的检验手段。我们研究了裂变观测量通用描述模型（GEF）以及在GEF数据上训练的概率机器学习模型如何重现测量的TKE分布，重点关注其在高入射中子能量下的宽度以及在训练范围之外的外推能力。我们在一个共同的计算数据集上评估了高斯过程回归（GPR）、带白噪声核的高斯过程回归、混合密度网络（MDN）和贝叶斯混合密度网络（BMDN）。扩展的MDN/BMDN分析使用了来自20个靶核的1,025个GEF能谱和60个实测分布。研究比较了最多20、40和约76 MeV的训练范围与匹配规模的对照组（扩展至30或40 MeV），同时保持每个靶核的能谱数量不变。

    arXiv:2611.33379v1 Announce Type: cross  Abstract: The incident-neutron-energy dependence of post-neutron total-kinetic-energy (TKE) distributions, particularly their width in the multichance-fission regime, provides a sensitive test of fission calculations. We investigate how the General Description of Fission Observables (GEF) and probabilistic machine-learning models trained on GEF reproduce measured TKE distributions, emphasizing their width at high incident-neutron energies and extrapolation beyond the training range. Gaussian process regression (GPR), GPR with a white-noise kernel, a mixture density network (MDN), and a Bayesian mixture density network (BMDN) are evaluated on a common calculated dataset. An extended MDN/BMDN analysis uses 1,025 GEF spectra from 20 target nuclei and 60 measured distributions. Training ranges of up to 20, 40, and approximately 76 MeV are compared with matched-size controls extending to 30 or 40 MeV while preserving the number of spectra per target.
    
[^224]: 评估数据扰动下算法稳定性的集成方法

    Evaluating Ensemble Methods for Algorithmic Stability Under Data Perturbations

    [https://arxiv.org/abs/2611.33375](https://arxiv.org/abs/2611.33375)

    本文揭示了集成方法对分类器稳定性的双面效应：子装袋可显著提升不稳定分类器的稳定性，但会破坏本已稳定的分类器，其有限装袋引入的蒙特卡洛噪声可通过膨胀argmax方法缓解。

    

    标准分类算法通常在决策边界附近表现出敏感性，这可能加剧因对高影响样本子集过拟合而已经存在的不稳定性。集成学习常被用于稳定分类器，但我们证明它反而可能破坏那些本已稳定的分类器。成熟有效的子装袋方法可以带来稳定性方面的显著改进，我们在死因分类场景中验证了这一点。然而，其效果取决于扰动水平，并且对于稳定的分类器而言，有限数量的装袋会引入破坏稳定性的蒙特卡洛噪声，且该噪声随着装袋数量的增长而缩减缓慢。不过，这种不稳定性可以通过最近提出的膨胀argmax（inflated argmax）方法来缓解，该方法允许在模糊情况下进行多类预测。我们进一步研究了Rashomon集——即实现接近最优预测性能的不同模型的集合——在多大程度上……

    arXiv:2611.33375v1 Announce Type: cross  Abstract: Standard classification algorithms often exhibit sensitivity near decision boundaries, which can compound instability that already exists from overfitting to a subset of high-influence samples. Ensembling is commonly used to stabilize classifiers, but we show that it can destabilize those that are already stable. The well-established practice of subbagging can yield substantial improvements in stability, as we validate in the cause-of-death classification setting. However, its effect depends on the perturbation level, and for stable classifiers the finite number of bags introduces destabilizing Monte Carlo noise that shrinks only slowly as the number of bags grows. This instability can be mitigated, though, by the recently proposed inflated argmax, which allows for multi-class predictions in ambiguous cases. We further examine the extent to which the Rashomon set, the collection of distinct models that achieve near-optimal predictive p
    
[^225]: 基于符号稀疏辨识的显式磁芯损耗方程发现方法

    Discovering Explicit Magnetic Core-Loss Equations via Symbolic Sparse Identification

    [https://arxiv.org/abs/2611.33364](https://arxiv.org/abs/2611.33364)

    本文提出符号稀疏辨识（SSI）框架，无需预设函数形式即可直接从实测数据中发现显式、可解释且高精度的磁芯损耗方程，弥合了经验模型与机器学习黑箱方法之间的差距。

    

    紧凑的磁芯损耗方程是磁性元件设计中的重要工具，但现有方法在精度与可解释性之间难以兼顾。诸如Steinmetz方程（SE）之类的经验模型虽然表达式紧凑，但精度有限；而机器学习方法虽能达到较高精度，却只能作为黑箱预测器使用。本文提出了一种符号稀疏辨识框架来弥合这一差距，该框架将磁芯损耗建模转化为数据驱动的符号回归问题，无需预设函数形式，即可直接从实测数据中恢复未知方程。首先，构建了一个远比SE丰富得多的、全面扩展的符号候选项库；其次，通过稀疏回归从该库中筛选出主导项；第三，利用AdamW优化器对保留项的系数进行精调，以最大化精度并形成解析方程。在Fair-Rite 78、79和95材料上的实验获得了（原文此处截断）……

    arXiv:2611.33364v1 Announce Type: new  Abstract: Compact core loss equations are essential tools in magnetic component design, but existing approaches trade accuracy against interpretability. Empirical models such as the Steinmetz Equation (SE) yield compact expressions of limited accuracy, while machine learning methods achieve high accuracy as black-box predictors. This paper bridges the gap with a Symbolic Sparse Identification (SSI) framework, which casts core loss modeling as a data-driven symbolic regression problem and recovers unknown equations directly from measured data, without presuming a functional form. First, a comprehensively expanded library of symbolic candidate terms far richer than the SE is constructed. Second, the governing terms are isolated from this library by sparse regression. Third, the coefficients of the retained terms are refined with an AdamW optimizer to maximize accuracy and form an analytical equation. Experiments on Fair-Rite 78, 79, and 95 yield tes
    
[^226]: 《活点地图：贝塞尔流形揭示用于模型合并与集成的隐藏曲面》

    The Marauder's Map: B\'{e}zier Manifolds Reveal Hidden Surfaces for Model Merging and Ensembling

    [https://arxiv.org/abs/2611.33353](https://arxiv.org/abs/2611.33353)

    该论文提出 Nimbus，通过学习连接 N 个不同任务专家模型的高维贝塞尔流形，将模式连通性扩展为“灵活简并”的类比，为模型合并与集成揭示隐藏的多任务解曲面。

    

    arXiv:2611.33353v1 公告类型：新论文。摘要：神经回路以比深度网络更强的鲁棒性而著称，这通常通过“简并性”来解释，即结构上不同的配置可以支持相同的功能。然而，真实的生物系统表现出一种更灵活的简并形式：特化的回路通过局部化的改变不断适应新的能力，同时保留既有功能。深度学习领域通过“模式连通性”来研究简并性，即用低损失曲线、曲面或体积连接在同一任务上训练的网络。但该领域尚缺乏与“灵活简并”相对应的概念，而这需要超越普通的模式连通性。定义这样的类比需要回答的问题是：许多独立初始化、在不同任务上训练的模型，能否被一个高维的多任务解区域连接起来。我们提出了 Nimbus，它学习一个高维贝塞尔流形来连接 N 个专家模型，超越了单——（原文摘要在此处截断）。

    arXiv:2611.33353v1 Announce Type: new  Abstract: Neural circuits are famously more robust than deep networks. This is often explained through degeneracy, where structurally distinct configurations support the same function. However, real biological systems exhibit a more flexible form of degeneracy in which specialized circuits continually adapt towards new competencies through localized changes while preserving existing function. Deep learning studies degeneracy through mode connectivity, which connects same-task networks by low-loss curves, sheets, or volumes. However, the field lacks a counterpart to flexible degeneracy, which demands more than ordinary mode connectivity. Defining such an analogue requires asking whether many independently initialized models trained on distinct tasks can be connected by a high-dimensional region of multi-task solutions. We introduce $\textit{Nimbus}$, which learns a high-dimensional B\'ezier manifold connecting $N$ specialists, going beyond the one-
    
[^227]: 理解基于大语言模型的数学优化建模中的失败模式

    Understanding Failure Patterns in LLM-Based Mathematical Optimization Modeling

    [https://arxiv.org/abs/2611.33351](https://arxiv.org/abs/2611.33351)

    该研究将LLM优化建模过程分解为数学推理、求解器使用和领域知识三个组成部分，通过构建针对性数据集分析失败模式，发现即便组合这些能力，可行但不正确的解仍是主要的失败类型。

    

    工业优化涉及复杂的约束条件，使得优化建模难以自动化。大语言模型（LLM）已被探索用于从自然语言生成优化模型，但它们在端到端场景中的可靠性仍然有限，经常产生不可执行的、不可行的或可行但不正确的解。在这项工作中，我们通过将基于LLM的优化建模过程分解为三个组成部分——数学推理、求解器使用和领域知识——并构建针对各方面的数据集，来研究其失败模式。利用这些数据集，我们分析了不同组成部分与错误之间的关系，发现它们与不同类型的失败相关联，尽管这些关系并不总是很强或一致。此外，虽然组合这些组成部分可以提高执行可靠性，但可行但不正确的解仍然是主导的失败模式，这表明……

    arXiv:2611.33351v1 Announce Type: new  Abstract: Industrial optimization involves complex constraints, making optimization modeling difficult to automate. Large language models (LLMs) have been explored for generating optimization models from natural language, but their reliability in end-to-end settings remains limited, often producing non-executable, infeasible, or feasible but incorrect solutions. In this work, we study failure modes in LLM-based optimization modeling by decomposing the process into three components: mathematical reasoning, solver usage, and domain knowledge, and constructing datasets that target each aspect. Using these datasets, we analyze how different components relate to errors and find that they are associated with different types of failure, although these relationships are not always strong or consistent. Furthermore, while combining these components improves execution reliability, feasible but incorrect solutions remain the dominant failure mode, suggesting
    
[^228]: FlowersBench：一个受FluidFlower启发的数据集，用于跨地质结构对碳存储代理模型进行基准测试

    FlowersBench: A FluidFlower-Inspired Dataset for Benchmarking Carbon Storage Surrogates Across Geological Structures

    [https://arxiv.org/abs/2611.33335](https://arxiv.org/abs/2611.33335)

    提出了FlowersBench基准数据集，包含跨越244种地质结构几何形状的5,000个二维油藏规模CO₂存储模拟，填补了测试机器学习碳存储代理模型对未见地质结构泛化能力的空白。

    

    机器学习代理模型正成为地质碳存储（GCS）不确定性量化中一种计算高效的工具。地质结构是不确定性的主要来源之一；然而，现有的基准模拟数据集主要是在固定的地质环境内变化岩石物理和作业参数，因此对未见过的地质结构的泛化能力在很大程度上仍未被探索。我们提出了FlowersBench，这是一个包含5,000个二维油藏规模CO₂存储模拟的基准数据集，涵盖244种结构几何形状。这些几何形状采样自一个内部结构生成器，该生成器受FluidFlower实验以及第11届石油工程师协会比较求解项目案例B（SPE11B）的启发。每个模拟均使用Open Porous Media（OPM）Flow运行，涵盖50年的注入期和500年的注入后监测期。测试用的几何形状被保留（未包含在训练集中）

    arXiv:2611.33335v1 Announce Type: new  Abstract: Machine learning surrogates are emerging as a computationally efficient tool for uncertainty quantification in geological carbon storage (GCS). Geological structure constitutes a major source of uncertainty; however, existing benchmark simulation datasets predominantly vary petrophysical and operational parameters within a fixed geological setting. Generalisation to unseen geological structures therefore remains largely unexplored. We introduce FlowersBench, a benchmark dataset of 5,000 two-dimensional, reservoir-scale CO$_2$ storage simulations across 244 structural geometries. The geometries are sampled from an in-house structural generator inspired by the FluidFlower experiments and Case B of the 11th Society of Petroleum Engineers Comparative Solution Project (SPE11B). Each simulation is run with Open Porous Media (OPM) Flow over a 50-year injection period and a 500-year post-injection monitoring period. Test geometries are withheld 
    
[^229]: 基于表示的预测的风险证书层级体系

    A Hierarchy of Risk Certificates for Representation-Based Prediction

    [https://arxiv.org/abs/2611.33333](https://arxiv.org/abs/2611.33333)

    该论文提出了一个针对基于表示的预测的层级式风险证书框架，证明期望块分数构成最优类风险的非递减下界并在紧致空间上收敛于该风险，从而为模型精度提供统计与经验证相结合的置信界。

    

    基于表示的预测是众多机器学习系统的基础。然而，一个拟合的预测器未能满足精度要求，并不能证明其表示-解码器类无法满足该要求。我们研究了作用于固定表示的Lipschitz解码器上的最优总体均方误差（MSE）。对于由 b≥2 个独立同分布观测组成的块，我们在成对Lipschitz约束下最小化总平方误差（以 b-1 归一化）。我们证明了期望块分数构成最优类风险的非递减下界，并在紧致度量特征空间上收敛于该风险，尽管没有任何固定块大小能提供一致的相对保证。经验证的对偶下界，以及与模型开发相互独立、目标值和预测值均有界的独立同分布校准观测，能够给出最优类风险的同步置信下界以及拟合（模型的误差置信上界）。

    arXiv:2611.33333v1 Announce Type: cross  Abstract: Representation-based prediction underlies many machine-learning systems. Yet a fitted predictor's failure to meet an accuracy requirement does not establish that its representation-decoder class cannot meet it. We study optimal population mean squared error (MSE) over Lipschitz decoders acting on a fixed representation. For blocks of $b\geq2$ independent and identically distributed (iid) observations, we minimize total squared error (normalized by $b-1$) under pairwise Lipschitz constraints. We prove that expected block scores form nondecreasing lower bounds on optimal class risk and converge to that risk on compact metric feature spaces, although no fixed block size provides a uniform relative guarantee. Verified dual lower bounds and iid calibration observations independent of model development, with bounded targets and predictions, yield simultaneous lower confidence bounds on optimal class risk and upper confidence bounds on fitted
    
[^230]: 用于主动式自动扩缩容的在线预测系统

    Online Prediction System for Proactive Autoscaling

    [https://arxiv.org/abs/2611.33332](https://arxiv.org/abs/2611.33332)

    提出了一种两阶段在线预测系统，通过季节性分解结合轻量级线性模型的残差修正，并借助每个预测周期的即时重训练，为大规模分布式系统的主动式自动扩缩容提供高精度、低延迟的短期流量预测。

    

    我们提出了一种用于短期时间序列预测的在线学习系统，可在大规模分布式系统中实现主动式自动扩缩容。与传统的注重建模季节性和长期趋势的预测方法不同，该系统针对即将到来的时间区间提供高精度、低延迟的预测，服务于互联网规模基础设施中的动态资源管理。该系统采用两阶段架构：首先进行季节性分解，随后使用带有工程特征的轻量级线性模型进行残差修正。通过在每次预测周期中即时重新训练模型，该在线预测系统能够持续适应不断变化的需求模式，而无需持久化模型存储或特征漂移监控。该系统已部署在众多异构服务中，每小时处理超过90万次预测；在生产流量的离线评估中，其基于STL的配置（原文在此处截断）

    arXiv:2611.33332v1 Announce Type: new  Abstract: We present an online learning system for short-term time series forecasting that enables proactive autoscaling in large-scale distributed systems. Unlike traditional forecasting methods that focus on modeling seasonality and long-term trends, the system targets high-accuracy, low-latency predictions for immediate future intervals for dynamic resource management in internet-scale infrastructures. The system employs a two-stage architecture: a seasonality decomposition stage followed by a residual correction stage using lightweight linear models with engineered features. By retraining models on-the-fly for every prediction cycle, the online prediction system continuously adapts to changing demand patterns without requiring persistent model storage or feature drift monitoring. Deployed across many heterogeneous services, the system processes over 900K predictions per hour; in an offline evaluation on production traffic, its STL-based config
    
[^231]: 勿让先验“治愈”异常：面向不完整测量检测的选择性锚定方法

    Do Not Let the Prior Heal the Anomaly: Selective Anchoring for Detection from Incomplete Measurements

    [https://arxiv.org/abs/2611.33331](https://arxiv.org/abs/2611.33331)

    该论文提出“选择性锚定”方法，在使用生成式先验从不完整测量中重建医学影像时，保护可靠测量的图像分量不被先验抹除病灶，并提供与所学先验无关的无偏估计与名义区间覆盖保证，具体实现为扩散/流采样器的包装器及快速的AnchoredFM模型。

    

    生成式先验能够从不完整的医学测量数据中恢复解剖细节，但可能会抹除影像检查本意要检测的病灶，即使测量数据中已包含这些病灶的证据。我们提出选择性锚定方法，保护可靠测量的图像分量免受先验驱动的改变，同时允许表达力强的生成模型重建尚未解析的结构。受保护分量的采集不确定性既可以以确定性方式加入到下游检验中，也可以通过对读数进行随机扰动由集成方法承载。我们将该原则实现为预训练扩散模型和流模型采样器的包装器，并实现为AnchoredFM——一种快速的受约束流模型。在线性高斯测量模型下，受保护系数在期望意义上是无偏的，并且在重复采集过程中达到名义区间覆盖率，且这一保证独立于所学习的先验。在稀疏视角CT、加速脑部……（原文摘要至此截断）

    arXiv:2611.33331v1 Announce Type: cross  Abstract: Generative priors can recover anatomical detail from incomplete medical measurements, but may erase the very lesions that imaging aims to detect, even when the measurements contain evidence of them. We introduce selective anchoring that protects reliably measured image components from prior-driven changes, while allowing expressive generative models to reconstruct unresolved structure. Acquisition uncertainty in the protected components is either added deterministically to the downstream test or carried by the ensemble through stochastic perturbations of the readings. We implement this principle as a wrapper for pretrained diffusion and flow samplers and as AnchoredFM, a fast constrained flow model. Under a linear-Gaussian measurement model, the protected coefficients are unbiased in expectation and achieve nominal interval coverage across repeated acquisitions, independently of the learned prior. Across sparse-view CT, accelerated bra
    
[^232]: 数据流上上下文学习的稀疏自适应

    Sparse Adaptation for In-Context Learning on Data Streams

    [https://arxiv.org/abs/2611.33325](https://arxiv.org/abs/2611.33325)

    本文提出SAIL框架，通过重用键值缓存、结合漂移检测与在线微聚类的稀疏适应机制，使表格基础模型能够在动态演变的物联网数据流上高效进行上下文学习。

    

    物联网（IoT）产生了海量数据流，使得边缘计算对于低延迟应用和5G网络至关重要。表格基础模型（TFMs）能够通过上下文示例而非参数更新来实现适应，但不断演变的数据流对上下文的相关性和计算效率提出了挑战。本工作提出了SAIL，一个用于TFMs稀疏适应的框架，它在各次适应事件之间重用模型的键值缓存。SAIL仅在数据流结构出现漂移、表明当前上下文可能不足以应对时，才重建其支持集，而其结构表示本身则通过预测性能的变化进行更新。SAIL将预序误差监测与在线的、面向类别的微聚类相结合，以检测聚类质量、类别可分离性和特征空间覆盖范围的变化。在进行适应时，上下文预算按类别质量成比例分配，并将最近更新的微……（摘要原文在此处截断）

    arXiv:2611.33325v1 Announce Type: new  Abstract: The Internet of Things (IoT) generates massive data streams, making edge computing essential for low-latency applications and 5G networks. Tabular foundation models (TFMs) enable adaptation through in-context examples rather than parameter updates, but evolving streams challenge context relevance and computational efficiency. This work introduces SAIL, a framework for sparse adaptation of TFMs that reuses the model's key--value cache between adaptation events. SAIL reconstructs its support set only when drifts in stream structure indicate potential context inadequacy, while the structural representation itself is updated by changes in predictive performance. SAIL combines prequential error monitoring with online, class-specific micro-clustering to detect changes in cluster mass, class separability, and feature-space coverage. Upon adaptation, a context budget is allocated proportionally to class mass, and the most recently updated micro-
    
[^233]: LLM智能体必须交流什么？从信息损失到交流失败

    What Must LLM Agents Communicate? From Information Loss to Communication Failures

    [https://arxiv.org/abs/2611.33318](https://arxiv.org/abs/2611.33318)

    本文提出基于信息论与通信复杂度的分析框架，指出LLM智能体有效通信需满足信息充分性、相关证据选择和接收信息利用三个条件，并证明局部概率报告造成的信息损失与性能下降高度吻合，而查询优先的交互仅在证据依赖私有查询时有效、且可能被紧凑的一次性表示抵消。

    

    当证据分布在多个LLM智能体之间时，一个团队可能集体拥有决策所需的全部信息，却无法有效地加以利用。智能体必须交流什么才能做出准确的联合预测？交互何时才有帮助？我们研究了有效通信的三个必要条件：所传递信息的充分性、相关证据的选择，以及对所接收信息的利用。借鉴信息论与通信复杂度理论，我们开发了一个结合理论基准、受控实验和消息级分析的框架。我们发现，局部概率报告可能丢弃关键信息，理论上的信息损失与观察到的性能下降密切吻合。当相关证据依赖于私有查询时，查询优先的交互能够改善预测，但紧凑的一次性表示可以消除这一优势。即使拥有充分的信息……（原文摘要在此处截断）

    arXiv:2611.33318v1 Announce Type: cross  Abstract: When evidence is distributed across LLM agents, a team may collectively possess all the information needed for a decision yet fail to use it effectively. What must agents communicate for accurate joint predictions, and when does interaction help? We study three requirements for effective communication: sufficiency of the information conveyed, selection of relevant evidence, and utilization of received information. Drawing on information theory and communication complexity, we develop a framework that combines theoretical benchmarks with controlled experiments and message-level analyses. We find that local probability reports can discard essential information, with theoretical information loss closely matching observed performance penalties. Query-first interaction improves predictions when relevant evidence depends on a private query, but compact one-shot representations can eliminate this advantage. Even when sufficient information is
    
[^234]: RL后训练究竟教会了文生图模型什么？——对Flow-GRPO的技能维度分析

    What Does RL Post-Training Teach Text-to-Image Models? A Skill-Wise Analysis of Flow-GRPO

    [https://arxiv.org/abs/2611.33316](https://arxiv.org/abs/2611.33316)

    本文通过逐技能测量pass@k发现，Flow-GRPO后训练对文生图模型的提升大多是“锐化”（放大基础模型本已具备的能力），只有在空间关系上才是真正“解锁”了新能力，因此基准分数的提升在很大程度上夸大了模型真正学到的东西。

    

    强化学习（RL）后训练已成为改进文生图流模型的标准方法：Flow-GRPO将SD3.5-M的GenEval分数从0.63提升至0.95，这一提升被解读为模型学会了组合式生成。我们通过在共享噪声种子上测量pass@k（逐项技能、在RL前后、使用训练中从未出现过的提示）来探究这一提升中有多少是基础模型自身的样本早已具备的。答案因技能而异。在GenEval六项技能中的四项上——其中32个基础样本已能解决几乎所有的提示——这种提升属于“锐化”：一个后训练样本的价值相当于4到10个基础样本，而pass@32几乎没有变化。在空间关系技能上，这种提升属于“解锁”：pass@32从0.69升至1.00，128个基础样本也抵不上一个后训练样本，且基础模型在32个样本中从未解决过的提示，被后训练模型以93%的比例解决——与其原本就能解决的提示频率相当。就基准测试的平均值而言，一个后训练样本相当于31个基础样本（原文在此处截断）。

    arXiv:2611.33316v1 Announce Type: cross  Abstract: Reinforcement learning (RL) post-training has become a standard way to improve text-to-image flow models: Flow-GRPO lifts the GenEval score of SD3.5-M from 0.63 to 0.95, a gain read as the model having learned compositional generation. We ask how much of that gain the base model's own samples already contained, by measuring pass@k over shared noise seeds, one skill at a time, before and after RL, on prompts never used in training. The answer differs by skill. On four of GenEval's six skills, where 32 base samples already solve almost every prompt, the gain is sharpening: one post-trained sample is worth 4 to 10 base samples, and pass@32 barely moves. On spatial relations it is unlocking: pass@32 rises from 0.69 to 1.00, 128 base samples do not match one post-trained sample, and prompts the base model never solved in 32 samples are solved 93% of the time, as often as those it solved. The benchmark average, 31 base samples per post-train
    
[^235]: 基于恢复感知任务加权的大规模模型合并

    Large-scale Model Merging via Recovery-aware Task Weighting

    [https://arxiv.org/abs/2611.33315](https://arxiv.org/abs/2611.33315)

    该论文提出RaTW模型合并框架，通过为专家能力损失严重的任务分配更大权重来缓解大规模模型合并中的任务间不平衡问题，并构建了包含100个任务的CLIPMerge-100基准用于大规模模型合并评估。

    

    模型合并将多个任务特定的专家模型合并为一个单一模型。然而，合并过程在不同任务之间对专家能力的保留是不均衡的。这种不平衡在大规模场景下尤为明显——即使平均准确率保持较高水平，某些任务仍可能遭受严重的能力损失。为了解决这种不平衡问题，我们提出了恢复感知任务加权（RaTW），这是一种模型合并框架，它为那些专家能力在初始合并模型中保留较差的任务分配更大的权重。我们将这些恢复机会量化为任务的专家预测正确但初始合并模型预测错误的带标签验证样本所占比例，并利用由此得到的任务权重重新合并专家模型。为了在大规模场景下评估模型合并，我们构建了CLIPMerge-100，这是一个包含100个任务的基准测试，在CLIP ViT-L/14和ViT-B/32上使用任务特定的专家模型。在11种合并方法、20、50和...

    arXiv:2611.33315v1 Announce Type: cross  Abstract: Model merging combines multiple task-specific expert models into a single model. However, merging preserves expert capabilities unevenly across tasks. This imbalance is particularly pronounced in large-scale settings, where some tasks can suffer severe capability losses even when average accuracy remains high. To address this imbalance, we introduce Recovery-aware Task Weighting (RaTW), a model merging framework that assigns larger weights to tasks whose expert capabilities are poorly retained by an initial merged model. We quantify these recovery opportunities as the fraction of a task's labeled validation samples that its expert predicts correctly but the initial merged model gets wrong, and use the resulting task weights to re-merge the experts. To evaluate merging at large scale, we construct CLIPMerge-100, a 100-task benchmark with task-specific experts on CLIP ViT-L/14 and ViT-B/32. Across eleven merging methods with 20, 50, and 
    
[^236]: 构建面向GitHub Copilot的统一行内代码编辑模型

    Building a Unified Inline Code Editing Model for GitHub Copilot

    [https://arxiv.org/abs/2611.33314](https://arxiv.org/abs/2611.33314)

    本文为GitHub Copilot提出统一行内代码编辑模型，通过锚定差异补丁的单一输出空间一次性预测光标补全、邻近编辑与远距离编辑三种类型，支持单次响应包含多个编辑，避免了原系统多次顺序调用不同模型的开销。

    

    AI辅助编程的范围涵盖从完全智能体式编程到编辑器中的行内代码建议。行内建议需要从编辑会话中预测开发者最可能的下一次编辑。下一次编辑可能位于光标处、邻近行中，或更远的位置，而一个编辑意图可能同时跨越其中多个位置。先前的系统按固定顺序查询独立的补全模型、邻近编辑模型和远距离编辑模型，每个模型各有自己的输出格式；因此它无法在单次响应中表达跨越远距离位置的编辑意图，每次建议可能需要多次顺序的模型调用，并且即使在修正方案更合适时也优先采用补全。我们提出了一个统一模型，在单一输出空间中预测所有三种编辑类型：一系列锚定差异补丁，指明位置、删除的行和新增的行。这种表示允许每次响应包含多个编辑，因此客户端可以从缓存中提供后续编辑……

    arXiv:2611.33314v1 Announce Type: cross  Abstract: AI-assisted coding ranges from fully agentic coding to inline code suggestions in the editor. Inline suggestions require predicting a developer's most likely next edit from an editing session. The next edit may lie at the cursor, in nearby lines, or farther away, and one editing intent can span several of these. The preceding system queries separate completion, nearby edit, and long-distance edit models, each with its own output format, in a fixed sequence; it thus cannot express intents spanning distant locations in one response, can require several sequential model calls per suggestion, and prioritizes completion even when a correction would serve better. We present a unified model that predicts all three edit types in one output space: a sequence of anchored diff patches specifying locations, deleted lines, and added lines. This representation permits multiple edits per response, so the client can serve later edits from a cache with
    
[^237]: 每一层都更换搭档：面向模型合并的均衡随机重新分组

    Change Partners at Every Layer: Balanced Random Regrouping for Model Merging

    [https://arxiv.org/abs/2611.33308](https://arxiv.org/abs/2611.33308)

    该论文提出在每一层对专家进行均衡随机重新分组的模型合并策略，作为完全合并与保留多个模型之间的折中方案，以缓解合并大量专家带来的准确率下降。

    

    模型合并将多个微调后的专家模型组合成一个无需重新训练即可服务多任务的单一模型，但以这种方式合并大量专家会显著降低准确率。分组提供了一种折中方案：在每一层或每个权重矩阵处，将 N 个专家划分为 K 个组……

    arXiv:2611.33308v1 Announce Type: new  Abstract: Model merging combines fine-tuned experts into one model that serves many tasks without retraining, but merging many experts this way substantially reduces accuracy. Grouping offers a middle ground: at every layer or weight matrix, the $N$ experts are divided into $K
    
[^238]: 语言模型微调中遗忘的两个时间尺度：快速的行为抑制与缓慢的表征漂移

    Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift

    [https://arxiv.org/abs/2611.33302](https://arxiv.org/abs/2611.33302)

    本研究通过在随机层次模型上预训练并用受限规则子集微调transformer，发现语言模型微调中的遗忘具有两个时间尺度——表面行为被快速抑制，而支撑该行为的潜在表征则以缓慢漂移的方式逐渐丢失。

    

    当今，人工智能系统的核心流程是先在广泛的文本语料库上进行预训练，然后在特定任务或期望行为上进行微调。然而，对于在受限数据分布上进行微调如何改变预训练期间获得的表征，我们的理解仍然有限。尤其是，表面行为的抑制是否与支撑该行为的潜在表征的擦除同时发生，这一点仍不清楚。在这项工作中，我们研究了监督微调（SFT）：先在一个具有分层潜在结构的受控上下文无关文法——随机层次模型上预训练transformer模型，再使用其生成规则的一个受限子集进行微调。虽然在预训练期间，模型依据正确语法规则进行生成的能力与构建相应潜在变量的内部表征是同步发生的，但我们发现SFT表现出截然不同的动力学：(i)（摘要在此处被截断，后续内容缺失）

    arXiv:2611.33302v1 Announce Type: cross  Abstract: Today, a core pipeline of artificial intelligence systems is pretraining on a broad range of text corpora and then fine-tuning on specific tasks or desired behaviors. Yet we still have limited understanding of how fine-tuning on a restricted data distribution modifies representations acquired during pretraining. In particular, it remains unclear whether surface behavioral suppression happens together with the erasure of latent representations that support it. In this work, we study supervised fine-tuning (SFT) by pretraining transformers on a controlled context-free grammar with hierarchical latent structure, the Random Hierarchy Model (RHM), and fine-tuning with a restricted subset of its generative rules. While during pretraining the model's ability to generate with the correct grammar rules coincides with building the internal representations of the corresponding latent, we find that SFT exhibits a very different dynamics: (i) the g
    
[^239]: JetDiffusion：利用NVFP4稀疏注意力突破视频生成的速度极限

    JetDiffusion: Pushing the Speed Limits of Video Generation with NVFP4 Sparse Attention

    [https://arxiv.org/abs/2611.33268](https://arxiv.org/abs/2611.33268)

    JetDiffusion通过可训练的视频稀疏量化注意力（VSQA）结合NVFP4量化、核融合与步数蒸馏，并借助分阶段训练方案稳定适应三项加速技术，实现了注意力8-29倍加速和端到端273倍加速的视频生成系统。

    

    长序列注意力是视频生成中的主要计算瓶颈，可以通过稀疏化和低比特量化来缓解。我们提出了JetDiffusion，一个建立在可训练视频稀疏量化注意力（VSQA）之上的视频生成系统，配备结合原生NVFP4计算与核融合的自定义推理内核。在90%稀疏度下，VSQA在不同硬件上相比FlashAttention可将注意力加速8-29倍。为进一步加速视频生成，我们将VSQA与NVFP4线性量化和步数蒸馏相结合。尽管每项技术单独使用均能保持生成质量，但联合适应这三种技术仍然充满挑战，尤其是在激进的NVFP4量化条件下。通过实证研究，我们开发出一种有效的分阶段训练方案，能够稳定地适应量化、稀疏化和少步去噪。在Wan2.1-14B上，所得模型实现了273倍端到端加速。

    arXiv:2611.33268v1 Announce Type: cross  Abstract: Attention over long sequences is a major computational bottleneck in video generation, which can be alleviated through sparsity and low-bit quantization. We propose JetDiffusion, a video generation system built on trainable Video Sparse Quantized Attention (VSQA), with custom inference kernels that combine native NVFP4 computation and kernel fusion. At 90% sparsity, VSQA accelerates attention by 8-29x compared with FlashAttention on different hardwares. To further accelerate video generation, we combine VSQA with NVFP4 linear quantization and step distillation. Although each technique can individually preserve generation quality, jointly adapting to all three remains challenging, particularly under aggressive NVFP4 quantization. Through empirical studies, we develop an effective staged training recipe that enables stable adaptation to quantization, sparsity, and few-step denoising. On Wan2.1-14B, the resulting model achieves 273x end-t
    
[^240]: 相同的分词器，更好的图像标记：通过MAP（最大后验）推断选择图像标记

    Same Tokenizer, Better Tokens: Choosing Image Tokens by MAP Inference

    [https://arxiv.org/abs/2611.33261](https://arxiv.org/abs/2611.33261)

    该论文提出RecaP方法，通过MAP推断在冻结分词器上同时平衡重建保真度与预训练自回归先验下的序列可预测性，从而为编码器特征选择比独立最近邻更优的图像码字。

    

    离散视觉标记对图像进行编码以用于重建，并作为自回归图像生成器的训练目标。向量量化分词器通常独立地为每个编码器特征分配其最近的码字。然而解码器是联合使用这些码字的，因此一个由各自最近码字组成的序列在重建图像时，其保真度可能不如另一种分配方式。对重建损失进行局部建模会产生一个带有依赖于解码器的目标和度量的最近向量问题。在我们测试的所有冻结分词器上，仅基于重建的搜索都提高了PSNR。不同的码字序列可以达到相似的重建质量，这为选择更可预测的序列留下了空间。我们提出RecaP（Reconstruction and Predictability，重建与可预测性），它通过平衡码字序列重建图像的保真度与该序列在预训练自回归先验下的可能性，为冻结的分词器选择码字。在ImageNet上，LlamaGen GPT-B训练……（摘要在此处截断）

    arXiv:2611.33261v1 Announce Type: cross  Abstract: Discrete visual tokens encode images for reconstruction and serve as training targets for autoregressive image generators. Vector-quantized tokenizers typically assign each encoder feature its nearest code independently. The decoder uses these codes jointly, so a sequence of individually nearest codes can reconstruct an image less faithfully than an alternative assignment. A local model of the reconstruction loss yields a closest-vector problem with a decoder-dependent target and metric. Reconstruction-only search improves PSNR across the frozen tokenizers we test. Different code sequences can achieve similar reconstruction quality, leaving room to favour more predictable ones.   We introduce RecaP (Reconstruction and Predictability), which selects codes for a frozen tokenizer by balancing how faithfully they reconstruct the image with how likely their sequence is under a pretrained autoregressive prior. On ImageNet, LlamaGen GPT-B tra
    
[^241]: 面向深度神经网络的快速可扩展抽象解释

    Fast and Scalable Abstract Interpretation for Deep Neural Networks

    [https://arxiv.org/abs/2611.33259](https://arxiv.org/abs/2611.33259)

    Scepter框架通过结构化语义将DNN抽象解释器的结构计算与数值求值分离，预先编译生成可复用的张量程序，在不损失精度的情况下实现快速且可扩展的神经网络性质认证。

    

    抽象解释可以认证深度神经网络（DNN）的性质，但精确的分析仍然代价高昂。通用实现在针对不同输入和性质分析相同架构时会重复执行相同的工作。将这项工作与数值运算交错进行会限制GPU的利用率，并限制张量编译器能够一并处理的分析范围。我们提出了Scepter，一个通过结构化语义对DNN抽象解释器进行专门化的框架。该语义确定中间抽象值的结构，同时推迟其数值求值。因此，来自连续操作的结构化计算可以在认证之前一起执行，从而生成一个可执行的张量程序。当抽象解释器和DNN架构固定时，该程序可以在不同权重和输入区域下重复使用。我们证明了它计算出的界与通用解释器计算出的界相同。

    arXiv:2611.33259v1 Announce Type: cross  Abstract: Abstract interpretation can certify properties of Deep Neural Networks (DNNs), but precise analyses remain expensive. General implementations repeat work when analyzing the same architecture under different inputs and properties. Interleaving this work with numerical operations limits GPU utilization and restricts how much of the analysis a tensor compiler can process together.   We present Scepter, a framework for specializing DNN abstract interpreters through structural semantics. The semantics determine the structure of intermediate abstract values while deferring their numerical evaluation. Structural computations from successive operations can therefore be performed together before certification, generating an executable tensor program. With the abstract interpreter and DNN architecture fixed, the program can be reused with different weights and input regions. We prove that it computes the same bounds as the general interpreter in
    
[^242]: 一个Token，多重教训：门控Delta网络中的滑窗注意力

    One Token, Many Lessons: Sliding-Window Attention within Gated Delta Networks

    [https://arxiv.org/abs/2611.33257](https://arxiv.org/abs/2611.33257)

    提出滑窗门控DeltaNet（SWGDN），从测试时回归视角研究状态更新中键值学习对的构建方式，用滑窗注意力替代短卷积实现内容依赖的局部查询聚合，以提升线性注意力的状态学习效果。

    

    线性注意力将历史上下文压缩为固定大小的循环状态，从而降低了长序列建模的计算与内存开销。从测试时回归（TTR）的视角来看，状态矩阵可以被视为一个在线学习的回归模型，其中键和值分别充当输入和目标，而查询则用于从该模型中读取信息。基于这一视角，状态学习的有效性不仅取决于更新规则，还取决于用于状态更新的键值学习对是如何构建的。虽然现有工作大多聚焦于状态如何被更新与组织，我们则研究了用于状态更新的学习对是如何构建的。为此，我们提出了滑窗门控DeltaNet（SWGDN），用滑窗注意力取代Delta规则线性注意力中的短卷积，以对投影后的查询进行内容依赖的局部聚合……

    arXiv:2611.33257v1 Announce Type: cross  Abstract: Linear attention compresses historical context into a fixed-size recurrent state, reducing the computational and memory costs of long-sequence modeling. From the perspective of test-time regression (TTR), the state matrix can be viewed as an online-learned regression model, where keys and values serve as inputs and targets respectively, and queries are used to read from the model. From this perspective, the effectiveness of state learning depends not only on the update rule but also on how the key-value learning pairs used for state updates are constructed. While existing work has largely focused on how the state is updated and organized, we investigate how the learning pairs used for state updates are constructed. To this end, we propose Sliding-Window Gated DeltaNet (SWGDN), which replaces short convolutions in Delta-Rule linear attention with sliding-window attention to perform content-dependent local aggregation of projected querie
    
[^243]: MonoQPM：将特征分解为概念以实现内在可解释性与预测性能

    MonoQPM: Splitting Features into Concepts for Inherent Interpretability and Predictive Performance

    [https://arxiv.org/abs/2611.33252](https://arxiv.org/abs/2611.33252)

    本文提出 MonoQPM，通过将 QPM 模型的多义特征解耦为单义概念以实现内在可解释性，并基于干扰引起的激活误差提出实用的多义性度量方法，从而显著提升共形预测的效率并获得更紧凑的预测集。

    

    arXiv:2611.33252v1 公告类型：cross 摘要：与叠加现象一样，多义性是实现可解释深度神经网络的主要障碍之一。虽然这些现象通常本质上是相互关联的，但二次规划增强模型（QPM）却自然地将二者解耦。QPM 通过用极少量特征的稀疏二值分配来表示类别，在设计上就避免了其最终特征的叠加，但仍然表现出多义性。然而，如何度量多义性仍是一个开放性问题。本工作提出了一种以实用性为核心的多义性度量方法，通过量化由干扰引起的激活误差的减少量，从而带来了使用共形预测（CP）获得更紧凑预测集的实际效用。我们明确地将 QPM 的多义特征解耦为单义概念，从而构建了单义 QPM（MonoQPM）。由于其特征已被解缠，MonoQPM 可以作为一个效率显著更高的共形预测器（摘要在此处截断）……

    arXiv:2611.33252v1 Announce Type: cross  Abstract: Alongside superposition, polysemanticity is one of the primary obstacles to achieving interpretable deep neural networks. While these phenomena are typically intrinsically linked, the Quadratic Programming Enhanced Model (QPM) naturally decouples them. By representing classes with a sparse binary assignment of very few features, QPM prevents superposition by design on its final features but still exhibits polysemanticity. However, measuring polysemanticity is an open problem. This work proposes a utility-focused approach to measuring polysemanticity by quantifying the decrease in interference-induced activation errors, which yields the practical utility of tighter prediction sets using Conformal Prediction (CP). We explicitly disentangle the polysemantic features of QPM into monosemantic concepts to create the Monosemantic QPM (MonoQPM). Because its features are disentangled, MonoQPM acts as a significantly more efficient Conformal Pre
    
[^244]: FERRETS：一种可扩展的时间序列分类符号化变换方法

    FERRETS: A Scalable Symbolic Transform for Time Series Classification

    [https://arxiv.org/abs/2611.33249](https://arxiv.org/abs/2611.33249)

    FERRETS是一种基于WEASEL 2的无标签字典变换方法，通过将随机长度和扩张率的窗口映射为精确八位符号词，在达到与最准确字典方法相当的分类精度的同时，速度提升超过百倍，成为测试中速度最快的时间序列分类方法。

    

    时间序列的字典方法通过从子序列构建符号词的词袋进行分类，但准确性高的方法扩展性差，与其他方法族相比，其准确性已无法弥补其高昂的代价。我们提出了FERRETS，一种构建在WEASEL 2之上的无标签字典变换。它将随机长度和扩张率的窗口映射为精确的八位词，为每个窗口的均值和离散程度添加水平词，并在多变量数据上添加跨通道词。我们在UCR、Multiverse和MONSTER数据存档上将其与领先的分类器（包括深度网络和基础模型）进行比较。在十线程预算下，FERRETS是UCR上测试的十一种方法中最快的。在其他十种方法中，只有MultiRocket-Hydra的准确性显著更高，但耗时是FERRETS的24.7倍。FERRETS与此前最准确的字典方法WEASEL 2无显著差异，而后者耗时是FERRETS的125倍，且FERRETS显著优于其他字典方法。

    arXiv:2611.33249v1 Announce Type: new  Abstract: Dictionary methods for time series build bags of symbolic words from subsequences, but the accurate ones scale poorly. Against other families of methods, their accuracy no longer pays for their cost. We present FERRETS, a label-free dictionary transform built on WEASEL~2. It keys windows of random length and dilation into exact eight-bit words, adds a level word of each window's mean and spread, and on multivariate data adds cross-channel words. We compare it with leading classifiers, deep networks and foundation models included, on the UCR, Multiverse and MONSTER archives. FERRETS is the fastest of the eleven methods timed on UCR under a ten-thread budget. Of the ten others, only MultiRocket-Hydra is significantly more accurate, taking 24.7 times as long. FERRETS is not significantly different from WEASEL~2, the most accurate earlier dictionary method, which takes 125 times as long, and is significantly ahead of the other dictionary met
    
[^245]: 典型心绞痛，反转了：UCI心脏病数据集中一个长期存在的文档标注错误

    Typical Angina, Reversed: A Long-Standing Documentation Error in the UCI Heart Disease Data

    [https://arxiv.org/abs/2611.33248](https://arxiv.org/abs/2611.33248)

    UCI心脏病数据集的胸痛变量编码与其文档说明完全相反——“无症状”实际代表典型心绞痛患者，“典型心绞痛”实际代表无胸痛患者，这是一个通过复现原始1986年研究和多站点数据交叉验证所证实的长期存在的标注错误。

    

    UCI心脏病数据集是广泛使用的机器学习基准数据集。我们发现其胸痛变量的存储方式与其文档说明相反。在所有四个数据采集站点，被文档标注为“无症状”（代码4）的编码实际标记的是典型心绞痛患者，而被文档标注为“典型心绞痛”（代码1）的编码实际标记的是几乎没有经典心绞痛特征的患者（在克利夫兰站点即为无胸痛的患者）。两项互补的验证确立了这一发现：（i）原始1986年克利夫兰研究的胸痛表格和回归系数只有在纠正后的解读下才能复现；（ii）在布达佩斯、瑞士和长滩的数据文件中，93-96%编码为4的患者具备全部三项经典心绞痛特征。一份1990年带有文字标签的克利夫兰文件已经使用了文档所述的标签，而作为克利夫兰数据子集的Statlog (Heart)也存储了相同的编码。OpenML、PMLB以及两个高票数的Kaggle副本（这些副本命名了类别）使用了相同的文档标签（约40……

    arXiv:2611.33248v1 Announce Type: new  Abstract: The UCI Heart Disease data are a widely used machine-learning benchmark. We show that their chest-pain variable is stored in reverse of its documentation. At all four sites, the code documented as "asymptomatic" (4) marks patients with typical angina, and the code documented as "typical angina" (1) marks patients with almost none of the classical angina features (at Cleveland, no chest pain).   Two complementary checks establish this: (i) the chest-pain table and regression coefficients of the original 1986 Cleveland study are reproduced only under the corrected reading; (ii) in the Budapest, Switzerland and Long Beach files, 93-96% of patients coded 4 have all three classical angina features. A 1990 word-labeled Cleveland file already uses the documented labels, and Statlog (Heart), a Cleveland subset, stores the same codes. OpenML, PMLB and two highly voted Kaggle copies that name the categories use the same documented labels (about 40
    
[^246]: 超越效率与效能的困境：对抗性恶意软件检测中可迁移性的批判性分析

    Beyond the Efficiency-Efficacy Dilemma: A Critical Analysis of Transferability in Adversarial Malware Detection

    [https://arxiv.org/abs/2611.33243](https://arxiv.org/abs/2611.33243)

    该论文提出对抗人工智能引擎（AAIE）自动化安全压力测试平台，揭示单步FGSM攻击在恶意软件检测迁移攻击中比迭代式PGD更有效（成功率52.9%对45.2%），并依据新兴AI治理标准提出了包含可迁移性得分、逃逸率和韧性指数的攻击评估指标体系。

    

    机器学习模型已被大规模部署用于恶意软件检测，因此迫切需要对对抗鲁棒性进行严格研究。为此，我们引入了对抗人工智能引擎（AAIE）：一个自动化安全压力测试平台。我们通过EMBER2018数据集对随机森林目标模型实施了迁移攻击。与直觉相反，我们发现单步FGSM攻击具有最高的攻击成功率（ASR），达到52.9%，而迭代式PGD攻击为45.2%（p < 0.002）。与容易过拟合代理网络从而产生脆弱扰动的迭代攻击相比，FGSM生成了更具可迁移性的对抗样本。我们使用攻击成功率、可迁移性得分、逃逸率和韧性指数对攻击进行了基准测试。通过与新兴AI治理标准相一致的这些指标，我们暴露了现有方法的不足——在受限扰动条件下，可迁移性得分最高仅约为0.8。

    arXiv:2611.33243v1 Announce Type: cross  Abstract: Machine Learning (ML) models are already being deployed at scale for malware detection. Therefore there is a pressing need to rigorously study adversarial robustness. To that end, we introduce the Adversarial AI Engine (AAIE): a platform for automated security stress-testing. We performed transfer attacks, via EMBER2018, against a Random Forest target. Counterintuitively, we found single-step FGSM to have the highest attack success rate (ASR) of 52.9% compared to iterative PGD with 45.2% (p < 0.002). Compared to iterative attacks which tend to overfit surrogate networks creating brittle perturbations, FGSM created more transferable examples. We benchmarked our attacks using ASR, transferability score, evasion rate, and resilience index. Through these metrics, proposed in concordance with emerging standards in AI governance, we expose shortcomings as our transferability scores peak at ~0.8 with constrained perturbations. We highlight th
    
[^247]: 学习平台而非学习者：教育推荐系统与LLM排序器离线评估中的序列捷径

    Learning the Platform, Not the Learner: Sequencing Shortcuts in the Offline Evaluation of Educational Recommender Systems and LLM Rankers

    [https://arxiv.org/abs/2611.33240](https://arxiv.org/abs/2611.33240)

    该论文发现在教育推荐系统的离线评估中，仅依赖平台行为规律、完全不使用学习者信息的简单结构化预测器就能达到最佳个性化推荐器和LLM排序器81%到100%的准确率，揭示了现有离线评估基准主要在衡量平台序列行为而非真正的学习者建模能力。

    

    教育推荐系统通常通过离线方式进行评估，即衡量其在辅导系统、学习应用和MOOC日志中预测下一个活动的能力，而下一个活动在很大程度上是由平台决定的：教师布置习题集，辅导教师反复抽题直至掌握，引擎自动选择题目。在来自五个平台的八个公开数据集上，我们在两种协议下比较了四种结构化预测器（除最后一个活动外不使用任何学习者信息）、十个个性化推荐器以及参数量从0.5到70亿的开源LLM排序器。结果显示，结构化预测器达到了任何方法最佳准确率的0.81到1.00之间（中位数0.95），且这主要源于群体层面的转换规律而非课程顺序；最佳个性化推荐器仅在五个数据集上显著更优，NDCG@10提升0.007到0.102，并且在Eedi数据集的四次训练中仅有三次更优。

    arXiv:2611.33240v1 Announce Type: cross  Abstract: Educational recommenders are usually compared offline, by how well they predict the next activity in logs of tutoring systems, learning applications and MOOCs, where the next activity is largely decided by the platform: teachers assign problem sets, tutors draw problems until mastery, engines select questions. On eight public datasets from five platforms, we compare four structural predictors, which use nothing about the learner beyond the last activity, with ten personalised recommenders and with open-weight LLM rankers from 0.5 to 7 billion parameters, under two protocols. The structural predictors reach between 0.81 and 1.00 of the best accuracy of any method (median 0.95), mostly through population transitions rather than the course order; the best personalised recommender is significantly better on five datasets, by 0.007 to 0.102 NDCG@10, and on Eedi in three of four training runs. Every method is far less accurate on steps along
    
[^248]: 基于无监督模型误差的复杂高维数据流序贯检验

    Sequential Testing on Complex and High Dimensional Data Streams Leveraging Unsupervised Model Errors

    [https://arxiv.org/abs/2611.33238](https://arxiv.org/abs/2611.33238)

    该论文提出利用仅在正常数据上训练的无监督模型误差作为检验统计量，为复杂高维数据流设计了无需分布假设的序贯假设检验与变点检测方法，解决了仅有变化前数据的信息不对称难题。

    

    我们提出了两个通用的、无分布假设的检验统计量，分别用于二元序贯假设检验（SHT）和序贯变点检测（SCPD），适用于流式环境中数据（如网络流量元数据）过于复杂、无法进行传统似然比驱动检验所要求的精确概率建模的场景。我们的框架还解决了信息不对称的挑战，即数据仅来自零假设/变化前状态，而无法获得来自备择假设/变化后状态的数据，例如零日攻击或新兴疾病相关状态。所提出的统计量通过监测仅在零假设/变化前数据上训练的预测或重构机器学习模型的误差来工作，依赖于模型不完美的泛化能力（当模型部署于来自备择假设/变化后类别的数据时，其性能会下降）。在对模型误差的温和假设条件下，我们建立了严格的分析保证……

    arXiv:2611.33238v1 Announce Type: cross  Abstract: We propose two general, distribution-free test statistics for binary Sequential Hypothesis Testing (SHT) and Sequential Change Point Detection (SCPD) respectively, in streaming environments where the data (e.g., network traffic metadata) are too complicated for the exact probabilistic modeling required by conventional likelihood ratio-driven tests. Our framework also addresses the challenge of information asymmetry, where data is available only from the null/pre-change regime and not from the alternative/post-change one such as zero-day attacks or emerging disease related regimes. The proposed statistics monitor the errors of a predictive or reconstructive machine learning model trained solely on null/pre-change data, relying on imperfect generalization (the model's performance will degrade when deployed on data from the alternative/post-change class). Under mild assumptions on the model's errors, we establish rigorous analytical guara
    
[^249]: 面向可持续农业的资源高效多模态融合稻瘟病早期预测方法

    Resource-Efficient Multimodal Fusion for Early Rice Blast Forecasting in Sustainable Agriculture

    [https://arxiv.org/abs/2611.33237](https://arxiv.org/abs/2611.33237)

    该论文提出了一种由注意力位置编码Transformer与CNN嵌入GRU组成的轻量级多模态融合框架，能够在计算与能源受限条件下高效建模气象和遥感数据的谱-时序依赖关系，实现稻瘟病的早期预测。

    

    面向可持续农业的资源高效人工智能日益重要，因为在计算和能源受限的条件下必须实现及时的病害预测。稻瘟病的早期评估可以通过优化作物管理措施来增强粮食安全，同时减少过量的农药使用。然而，现有框架往往存在计算开销高的问题，且在建模异构气象与遥感数据中复杂的谱-时序依赖关系方面能力有限。为应对这些挑战，本文提出了一种轻量级多模态融合框架，该框架由基于注意力位置编码的Transformer和嵌入卷积神经网络（CNN）的门控循环单元（GRU）组成，用于高效的谱-时序表征学习。注意力增强的位置编码为序列气象标记分配自适应的重要性，从而使得……（原文摘要在此处被截断）

    arXiv:2611.33237v1 Announce Type: new  Abstract: Resource-efficient artificial intelligence has become increasingly important for sustainable agriculture, where timely disease forecasting must be achieved under computational and energy constraints. Early estimation of rice blast can enhance food security by optimizing crop management practices while reducing excessive pesticide use. However, existing frameworks often suffer from high computational overhead and limited capability to model the complex spectral-temporal dependencies present in heterogeneous meteorological and remote sensing data. To address these challenges, this paper proposes a lightweight multimodal fusion framework comprising an attentive positional encoding-based Transformer and a convolutional neural network (CNN)-embedded gated recurrent unit (GRU) for efficient spectral-temporal representation learning. The attention-enhanced positional encoding assigns adaptive importance to sequential meteorological tokens, enab
    
[^250]: KiRA：面向预训练模型参数高效微调的Kronecker输入门控秩自适应方法

    KiRA: Kronecker Input-Gated Rank Adaptation for Parameter-Efficient Fine-Tuning of Pretrained Models

    [https://arxiv.org/abs/2611.33236](https://arxiv.org/abs/2611.33236)

    KiRA将微调更新表示为冻结随机矩阵与输入门控可训练因子的Kronecker积，在不增加可训练参数的前提下突破低秩PEFT方法的秩上限，实现参数高效的输入自适应微调。

    

    对大型预训练模型进行全量微调代价高昂，参数高效微调（PEFT）是标准的替代方案。大多数PEFT方法采用低秩更新，其秩远低于全量微调，且对所有输入都是相同的。输入自适应变体虽然可以针对每个输入调整容量，但仍受限于低秩瓶颈，而且任务是否能从输入条件化中获益在训练前是未知的。本文提出Kronecker输入门控秩自适应（KiRA）。该方法不再使用传统的低秩乘积，而是将更新 $\Delta W$ 表示为冻结的随机 $p\times p$ 矩阵 $A$ 与可训练因子 $M(x)$ 的Kronecker积，从而突破秩上限：由于秩在该乘积下具有相乘特性，$A$ 可以将 $M(x)$ 的秩放大 $p$ 倍，而不增加任何可训练参数。因子 $M(x)$ 进一步分解为 $B_1\,\mathrm{diag}(g(x))\,B_2$，其中输入门控 $g(x)$ 仅需计算一次……

    arXiv:2611.33236v1 Announce Type: cross  Abstract: Full fine-tuning of large pretrained models is expensive, and parameter-efficient fine-tuning (PEFT) is the standard alternative. Most PEFT methods use low-rank updates, whose rank lies far below that of full fine-tuning and is the same for every input. Input-adaptive variants vary the capacity per input but remain within the low-rank bottleneck, and whether a task benefits from input conditioning is not known before training. This paper introduces Kronecker input-gated rank adaptation (KiRA). Instead of the conventional low-rank product, the update $\Delta W$ is the Kronecker product of a frozen random $p\times p$ matrix $A$ and a trainable factor $M(x)$, which overcomes the rank ceiling: because rank is multiplicative under this product, $A$ multiplies the rank of $M(x)$ by $p$ without adding trainable parameters. The factor $M(x)$ is further decomposed as $B_1\,\mathrm{diag}(g(x))\,B_2$, where the input gate $g(x)$, computed once pe
    
[^251]: RL究竟值多少个样本？文本到图像模型中Best-of-N采样与强化学习后训练的对比

    How Many Samples Is RL Worth? Best-of-N versus RL Post-Training for Text-to-Image Models

    [https://arxiv.org/abs/2611.33232](https://arxiv.org/abs/2611.33232)

    该论文提出“兑换率”这一概念来量化RL后训练与Best-of-N采样的等效样本数，发现在大多数不含位置要求的提示词上，基础模型配合现成VLM选择器的Best-of-16采样即可达到与RL后训练图像相当的效果，人工筛选最多3张基础图像也能获得类似收益。

    

    在可自动验证的奖励上进行强化学习（RL）已成为文本到图像模型训练中常见的最后一个步骤。获得更多正确图像的一种更简单方法是Best-of-N：从基础模型中抽取N张图像，保留选择器所偏好的那一张。我们提出这样一个问题：一个RL样本等价于多少个基础模型样本，并将这个数字称为“兑换率”。我们在398个GenEval风格的新提示词上测量了这一数值——GenEval正是该RL奖励所依赖的基准测试——测试对象为Stable Diffusion 3.5 Medium及其公开的Flow-GRPO检查点，每个提示词均有人工标注，并使用了除该奖励之外的两个自动评判器。答案取决于提示词的类型。五分之四的提示词不涉及位置要求（例如，一个物体位于另一个物体的左边）。在这类提示词上，基础模型能够迎头赶上：使用现成的视觉语言模型（VLM）作为选择器的Best-of-16，其效果与一张RL图像大致相当。而一个翻看最多3张基础图像的人就能找到一……（原文摘要至此截断）

    arXiv:2611.33232v1 Announce Type: cross  Abstract: Reinforcement learning (RL) on a reward that can be checked automatically has become a common last step in the training of text-to-image models. A simpler way to get more correct images is best-of-N: draw N images from the base model and keep the one that a picker prefers. We ask how many base model samples one RL sample is worth, and call this number the exchange rate. We measure it for Stable Diffusion 3.5 Medium and its public Flow-GRPO checkpoint on 398 new prompts in the style of GenEval, the benchmark behind the RL reward, with human labels on every prompt and two automatic judges other than the reward. The answer depends on the prompt. Four fifths of the prompts do not ask for a position (for example, one object left of another). On these, the base model catches up: best-of-16 with an off-the-shelf vision-language model (VLM) as the picker is about as good as one RL image. A person who looks through up to 3 base images finds a c
    
[^252]: 面向离线强化学习的效果驱动技能抽象

    Effect-Driven Skill Abstractions for Offline Reinforcement Learning

    [https://arxiv.org/abs/2611.33229](https://arxiv.org/abs/2611.33229)

    该论文提出一种效果驱动的动作块量化方法，通过优化下一状态预测使离散技能依据其对状态的影响来区分，并可与常见离线强化学习方法结合，提升长时程目标条件任务的表现。

    

    我们旨在学习离散技能抽象，以促进长时程、目标条件的强化学习（RL）任务。近期方法通过聚类或重构对动作进行量化并取得了成功。然而，这些目标所得到的技能与当前状态所能提供的可能性或目标所需的内容没有直接关联。相反，我们认为目标导向的技能应当由它们“如何影响智能体的状态”来驱动。为实现这一目标，我们提出了一种新颖的动作块量化方法，该方法优化以当前状态和动作为条件的下一状态预测。这一目标促使技能通过它们所导致的未来状态来加以区分，从而形成一个覆盖给定状态下可到达的多样化结果的技能词汇表。我们的框架可以轻松与常见的离线RL方法结合，只需将其连续动作头替换为在学习到的技能中进行选择的离散动作头。

    arXiv:2611.33229v1 Announce Type: cross  Abstract: We seek to learn discrete skill abstractions to facilitate long-horizon, goal-conditioned reinforcement-learning (RL) tasks. Recent approaches have shown success by quantizing actions via clustering or reconstruction. However, these objectives lead to skills that are not directly related to what the current state affords or what the goal requires. Instead, we argue that goal-directed skills should be driven by \emph{how they affect the state} of the agent. To realize this, we introduce a novel quantization method for action chunks that optimizes next-state predictions conditioned on the current state and action. This objective encourages skills to be distinguished by the future states they lead to, resulting in a vocabulary that spans diverse outcomes reachable from a given state. Our framework can easily be paired with common offline RL methods by replacing their continuous actor head with a discrete one that selects among the learned
    
[^253]: 软去中心化期望SARSA：通过渐进适应实现均衡选择

    Soft Decentralized Expected SARSA: Equilibrium Selection via Gradual Adaptation

    [https://arxiv.org/abs/2611.33228](https://arxiv.org/abs/2611.33228)

    该论文提出软去中心化期望SARSA（SDES）算法，通过渐进更新Q值而非直接切换至最佳响应，不仅证明了收敛性，还首次揭示了软更新可作为去中心化多智能体学习中的均衡选择机制，通过调节学习速率引导系统走向合作或竞争均衡。

    

    我们提出了软去中心化期望SARSA（SDES），这是对去中心化Q学习的一种改进，其中智能体保持持续探索，并在更新时渐进地更新其执行Q值，而不是直接切换到新学习到的最佳响应。我们证明了在弱无环随机博弈中，对于足够大的批规模，算法可收敛到ε-贪婪纳什均衡。随后，我们在具有记忆的重复囚徒困境中展示，软更新可以在不改变均衡集的情况下重塑吸引盆。在确定性极限下，我们推导出由目标策略中动作间Q值差异所支配的解析吸引盆边界，并表明降低策略学习速率可以有利于合作或竞争均衡。这些结果表明，软更新不仅仅是一种稳定化手段，更是去中心化学习中的一种均衡选择机制。

    arXiv:2611.33228v1 Announce Type: cross  Abstract: We introduce Soft Decentralized Expected SARSA (SDES), a modification of Decentralized Q-learning in which agents maintain persistent exploration and update their acting Q-values gradually rather than switching directly to newly learned best responses when updating. We prove convergence to $\epsilon$-greedy Nash equilibria in weakly acyclic stochastic games for sufficiently large batch sizes. We then show, in the repeated Prisoner's Dilemma with memory, that soft updating can reshape basins of attraction without changing the equilibrium set. In the deterministic limit, we derive analytical basin bounds governed by the Q-value differences between actions in the target policies and show that reducing the policy learning rate can favor either cooperative or competitive equilibria. These results suggest that soft updating is not merely a stabilization device but also a mechanism of equilibrium selection in decentralized learning.
    
[^254]: 真实的包，错误的控制器：LLM生成的PLC代码中供应商-协议不匹配的静态审计

    Real Package, Wrong Controller: Static Auditing of Vendor--Protocol Mismatches in LLM-Generated PLC Code

    [https://arxiv.org/abs/2611.33227](https://arxiv.org/abs/2611.33227)

    论文提出了Proteus-OT，一个基于版本固定绑定图谱的无标签静态验证器，用于检测大语言模型生成的PLC代码中导入包与声明工业控制器目标之间的供应商-协议不匹配，发现883个生成脚本中有20.4%存在此类导入级别不兼容问题。

    

    语言模型可以生成驱动可编程逻辑控制器（PLC）上工业协议的Python代码，然而现有的代码落地检查只验证导入的包或符号是否存在，而不验证它是否适合所声明的工业目标。我们定义了元数据条件下的导入级别供应商-协议不兼容性，并构建了Proteus-OT，这是一个基于版本固定的已文档化绑定图谱的无标签静态验证器。在883个生成的脚本中，有180个存在与声明目标不符的导入级别不匹配（20.4%）；限制为单一供应商目标时该比例为16.3%，若额外要求违规库能够触达协议通信调用，则比例为14.2%，我们将其解读为依赖于语料库元数据而非已验证的意图。28个可判定的人工编写的控制文件中没有一个存在此类问题，但样本量太小，无法估计该问题的普遍程度。在平衡子集中，96.6%的不匹配脚本（摘要至此截断）

    arXiv:2611.33227v1 Announce Type: cross  Abstract: Language models can generate Python that drives industrial protocols on programmable logic controllers, yet existing code-grounding checks ask whether an imported package or symbol exists rather than whether it suits the declared industrial target. We define \emph{metadata-conditioned, import-level vendor--protocol incompatibility} and build \textsc{Proteus-OT}, a label-free static verifier over a version-pinned graph of documented bindings. Across 883 generated scripts, 180 carry an import-level mismatch against the declared target (20.4\%); restricting to single-vendor targets gives 16.3\%, and additionally requiring the offending library to reach a protocol communication call gives 14.2\%, which we read as conditional on the corpus metadata rather than on verified intent. None of the 28 adjudicable human control files carries one, a sample too small to estimate a prevalence. Within the balanced subset, 96.6\% of mismatching scripts 
    
[^255]: 一种用于学习有向无环图的因果能量先验

    A Causal Energy Prior for Learning Directed Acyclic Graphs

    [https://arxiv.org/abs/2611.33225](https://arxiv.org/abs/2611.33225)

    提出了局部-全局能量先验（LGE），通过结合局部结构差异（结构汉明距离）与因果差异（结构干预距离）构建DAG上的吉布斯分布，使贝叶斯DAG学习能够编码干预沿路径传播的因果信息，并量化了数据在先验部分错误时对先验的修正能力。

    

    贝叶斯有向无环图（DAG）学习长期以来依赖于无信息或弱信息先验，其假设是有信息先验所需的信息很少可用。在遗传学和系统生物学中，注释数据库如今使得在从数据学习之前构建先验DAG成为可能。现有的有信息先验是局部的：它们逐条弧地对DAG进行评分，无法编码干预如何沿其路径传播。因此，我们提出了局部-全局能量先验（LGE），这是一种DAG上的吉布斯分布，其能量结合了与先验DAG的局部差异（结构汉明距离）和因果差异（结构干预距离）。我们推导了仅从先验DAG出发校准其衰减速度以及两种差异权重的规则。当先验DAG部分不正确时，我们量化了有多少先验概率质量保留在真实DAG上，以及需要多少数据才能推翻该先验。在仿真实验中，LGE……

    arXiv:2611.33225v1 Announce Type: cross  Abstract: Bayesian learning of directed acyclic graphs (DAGs) has long relied on non- or weakly informative priors, on the assumption that the information needed for an informative one is rarely available. In genetics and systems biology, annotation databases now make it possible to assemble a prior DAG before learning from data. Existing informative priors are local: they score a DAG arc by arc and cannot encode how interventions propagate along its paths. Thus, we propose the local-global energy prior (LGE), a Gibbs distribution over DAGs whose energy combines a local divergence from the prior DAG, the Structural Hamming Distance, and a causal one, the Structural Intervention Distance. We derive rules to calibrate its decay and the weight of the two divergences from the prior DAG alone. We quantify how much prior mass stays on the true DAG, and how much data overturn the prior, when the prior DAG is partially incorrect. In simulations, LGE imp
    
[^256]: 用于药物-药物相互作用预测的自适应注意力路由多模态专家图神经网络

    Adaptive attention routing multimodal expert graph neural network for drug-drug interaction prediction

    [https://arxiv.org/abs/2611.33200](https://arxiv.org/abs/2611.33200)

    提出自适应注意力路由多模态专家图神经网络（AARME-GNN），通过特征自适应校准模块与带熵约束的多模态权重路由器，解决药物相互作用预测中多模态特征异构、融合不适应药物对及模态信息被稀释的问题。

    

    准确预测药物-药物相互作用（DDIs）对于联合治疗中的用药安全至关重要。基于知识图谱（KGs）的多模态方法已经将分子结构、子结构、靶点和生物医学关系融合到药物表示中，但仍面临三个问题：各模态间的特征是异构的、融合过程无法针对单个药物对自适应调整、以及模态特异性信息在整合过程中被稀释。我们提出自适应注意力路由多模态专家图神经网络（AARME-GNN）来解决这些问题。特征自适应校准模块（FACM）在融合之前通过瓶颈重参数化和通道级重校准来降低每种表示中的模态内噪声。多模态权重路由器（MMWR）随后为每个药物对设定模态权重，并通过熵约束防止权重坍缩到单一模态上。路由感知加权……

    arXiv:2611.33200v1 Announce Type: cross  Abstract: Accurate prediction of drug-drug interactions (DDIs) is essential for drug safety in combination therapy. Multimodal methods built on knowledge graphs (KGs) already combine molecular structures, substructures, targets and biomedical relations into drug representations, but they still face three problems: features are heterogeneous across modalities, fusion does not adapt to individual drug pairs, and modality-specific information is diluted during integration. We propose the adaptive attention routing multimodal expert graph neural network (AARME-GNN) to address them. A Feature Adaptive Calibration Module (FACM) reduces within-modality noise in each representation through bottleneck reparameterization and channelwise recalibration before fusion. A Multimodal Weight Router (MMWR) then sets the modality weights of each drug pair, with an entropy constraint that prevents the weights from collapsing onto one modality. A Routing-Aware Weigh
    
[^257]: 大语言模型中的外在认知负荷

    Extraneous Cognitive Load in Large Language Models

    [https://arxiv.org/abs/2611.33197](https://arxiv.org/abs/2611.33197)

    该论文提出了首个系统地将信息呈现方式（外在认知负荷）与任务难度分离开来的受控基准CogLoadBench，并设计了基于模型激活值、在提示阶段即可计算的外在负荷分数（ELS），用于估计和诊断大语言模型的外在认知负荷。

    

    认知负荷理论表明，人类的问题解决能力会受到两种负荷的影响：内在负荷（源于任务本身的难度）和外在负荷（源于任务信息的呈现方式）。大语言模型表现出类似的效应：仅改变信息的呈现方式就能影响模型性能，但这些效应尚未被系统地从任务难度中分离出来。为了填补这一空白，我们提出了CogLoadBench，这是一个受控基准，在三个推理任务族中系统地变化五种受认知负荷理论启发的呈现因素。在多个模型上，我们发现较高的外在负荷条件往往会通过增加竞争或使推理路径复杂化来降低准确率。为了支持对负荷相关效应的诊断与缓解，我们提出了外在负荷分数，这是一种在提示阶段从模型激活值计算得到的指标，用于估计模型的外在负荷，并证明其可以在不同的推理任务类型和负荷条件之间进行泛化。

    arXiv:2611.33197v1 Announce Type: cross  Abstract: Cognitive Load Theory (CLT) shows that human problem-solving is affected by both intrinsic load, arising from task difficulty, and extraneous load, arising from how task information is presented. Large language models (LLMs) show analogous effects: presentation alone can affect model performance, but these effects have not yet been systematically isolated from task difficulty. To address this gap, we introduce CogLoadBench, a controlled benchmark that varies five CLT-inspired presentation factors across three reasoning task families. Across models, we find that higher extraneous load conditions often reduce accuracy by either adding competition or complicating the reasoning path. To support diagnosis and mitigation of load-related effects, we propose the Extraneous Load Score (ELS), a prompt-time metric computed from activations that estimates a model's extraneous load, and show that it generalizes across reasoning task types and load 
    
[^258]: 创意任务中人机共创的集体多样性动态

    Dynamics of Collective Diversity in Human-AI Co-Creation for Creative Tasks

    [https://arxiv.org/abs/2611.33192](https://arxiv.org/abs/2611.33192)

    该论文通过大规模众包研究揭示了人机共创的动态规律：AI代理虽能提升创意作品质量并降低用户投入，但随着AI采用规模扩大，会导致集体多样性不断下降。

    

    生成式AI系统正越来越多地被大规模部署，以支持人类用户在各领域的创意任务。然而，随着越来越多的用户采用相同的AI代理，由此产生的作品可能变得日益同质化，导致集体多样性的崩塌。先前的研究已经记录了这一风险，但缺乏一个系统性框架来理解随着AI采用规模的扩大，集体多样性在共创环境中如何变化。在本工作中，我们开展了一项大规模众包研究，以理解和建模人机共创在两个创意领域（创意假设生成和创意故事写作）中集体多样性、质量和努力程度的动态变化。我们发现，当前用于共创的AI代理能够提升作品质量并减少用户的努力，但代价是随着采用规模的增加，集体多样性会下降。我们进一步表明，更细粒度的人机交互以及……（原文摘要在此处截断）

    arXiv:2611.33192v1 Announce Type: cross  Abstract: Generative AI systems are increasingly deployed at scale to support human users in creative tasks across a variety of domains. However, as more users adopt the same AI agent, the resulting artifacts may become increasingly homogeneous, leading to a collapse in collective diversity. Prior work has documented this risk, but lacks a systematic framework for understanding how collective diversity changes as AI adoption scales across co-creation settings. In this work, we conduct a large-scale crowdsourcing study to understand and model the dynamics of collective diversity, quality, and effort in human-AI co-creation across two creative domains: Creative Hypothesis Generation and Creative Story Writing. We find that current AI agents used in co-creation can improve artifact quality and reduce effort, but at the cost of declining collective diversity as adoption increases. We further show that more fine-grained human-AI interaction, along wi
    
[^259]: 一种用于在推力中断后重新规划低推力地月转移的神经算子：支持原定到达或延迟到达

    One Neural Operator for Re-planning Low-Thrust Cislunar Transfers after Thrust Outages, at the Original or a Later Arrival

    [https://arxiv.org/abs/2611.33188](https://arxiv.org/abs/2611.33188)

    该论文训练了一个神经算子，能够在电推进任务发生推力中断后即时重新规划低推力地月转移轨迹（支持原定到达或最多延后10天到达），其控制误差比传统快速重规划基准低约一个数量级，并能以0.997的AUC准确预测中断的可恢复性。

    

    安全模式事件导致的推力中断在电推进任务中十分常见。为了证明任务在此类事件下的可行性，需要进行错过推力裕度研究，其中需要对数千种中断场景求解精确的重新规划方案。我们训练了一个神经算子，它能够将带有中断轮廓的标称轨迹映射到重新规划的轨迹以及恢复方案存在的概率，涵盖270条低推力近直线晕轨道（NRHO）到远距离逆行轨道（DRO）的转移，允许最多10天的到达时间滑移。在未见过的转移和零滑移情况下，该算子在最严重一半中断场景上的中位控制误差为1.20%，而最佳廉价重新规划基准方法为14.0%。当将到达滑移作为输入时，误差为1.67%，对比基准方法的23.2%。它还能预测给定到达滑移下中断的可恢复性，受试者工作特征曲线下面积（AUC）达到0.997，而最低恢复（摘要原文在此处截断）

    arXiv:2611.33188v1 Announce Type: cross  Abstract: Safe-mode events causing thrust outages are common in missions with electric propulsion. To demonstrate mission feasibility in such an event, missed-thrust margin studies are performed where thousands of outage scenarios need an exact re-plan solved. We train a neural operator that maps a nominal trajectory with an outage profile to a re-planned trajectory and to the probability that the recovery exists, for 270 low-thrust near-rectilinear halo orbit (NRHO) to distant retrograde orbit (DRO) transfers, at an allowed slip of up to 10 days. On unseen transfers and zero slip, its median control error on the severe half of outages is 1.20% compared to 14.0% for the best cheap re-planner benchmark. With the arrival slip as an input, the error is 1.67% against 23.2%. It also predicts the recoverability of an outage for a given arrival slip with an area under the receiver operating characteristic curve (AUC) of 0.997, while the minimum recover
    
[^260]: 语义密度无法预测智能体记忆崩溃

    Semantic Density Does Not Predict Agent-Memory Collapse

    [https://arxiv.org/abs/2611.33184](https://arxiv.org/abs/2611.33184)

    本文通过大规模预注册实验发现，包括平均成对余弦相似度在内的记忆库统计指标均无法可靠预测智能体任务准确率的早期下降，即语义密度不能作为智能体记忆崩溃的有效预警信号。

    

    arXiv:2611.33184v1 公告类型：交叉（cross）。摘要：持久化的文本记忆可以提升大语言模型（LLM）智能体的能力，但反复的记忆更新可能会侵蚀任务性能。我们探究记忆库的统计指标能否在留出集准确率下降之前发出预警。我们在来自13种记忆方法、覆盖ALFWorld、AppWorld、ScienceWorld、WebShop以及两个ARC-AGI设置共805个快照上，预先注册了平均成对余弦相似度（ρ）以及另外13种几何、检索和图统计指标。这些方法存储轨迹、规则、工作流、策略或链接记录，并采用选择性检索或全库提示的方式。结果显示，没有任何此类统计指标能可靠预测任务准确率下降的早期迹象，且ρ的中位领先时间为零。若以准确率峰值之后的检查点作为次级下降点，高相似对数量和最近邻拥挤度等指标虽然变化得更早，但并未一致地优于单纯的存储条目数量。我们在该语料库上训练了分类器，并将其不加修改地应用于PropWorld中多达5（此处摘要被截断）

    arXiv:2611.33184v1 Announce Type: cross  Abstract: Persistent textual memory can improve LLM agents, but repeated updates can erode task performance. We ask whether memory-store statistics warn before held-out accuracy declines. We registered mean pairwise cosine similarity ($\rho$) and 13 additional geometry, retrieval, and graph statistics on 805 snapshots from 13 memory methods across ALFWorld, AppWorld, ScienceWorld, WebShop, and two ARC-AGI settings. These methods store trajectories, rules, workflows, strategies, or linked records, using selective retrieval or whole-store prompting. No such statistic reliably predicted early signs of task accuracy decline, and $\rho$ had zero median lead. With the checkpoint after peak accuracy as a secondary decline point, measures of highly similar pairs and nearest-neighbor crowding changed earlier but did not consistently outperform stored-item count. We trained classifiers on this corpus and applied them unchanged to PropWorld runs of up to 5
    
[^261]: REI：从仓库演化中学习创意构思

    REI: Learning Ideation from Repository Evolution

    [https://arxiv.org/abs/2611.33183](https://arxiv.org/abs/2611.33183)

    本文提出REI框架，通过从代码仓库的演化历史中学习可复用的创意策略，将领域技术经验注入模型参数，以解决AI自动化科学迭代中创意构思的瓶颈，并首次揭示了质量驱动优化中的语义多样性坍塌问题。

    

    随着AI系统在执行和评估实验方面的能力日益增强，创意构思正成为自动化科学与工程迭代中的瓶颈。在许多现实世界领域中，有用的想法稀少且验证成本高昂，这使得直接的在线探索效率低下。我们观察到，仓库的演化过程提供了一种可扩展的可复用技术经验来源：代码变更记录了具体的技术假设，而后续的修订、采纳和复用则提供了关于这些假设实用性的信号。基于这一观察，我们提出了REI，一个由仓库演化驱动的创意构思框架，它从仓库演化中学习可复用的创意策略，将领域特定的技术经验注入模型参数，从而减少对昂贵试错的依赖。我们进一步发现了质量驱动的创意优化中一个隐藏的失效模式：语义多样性坍塌。尽管生成的想法

    arXiv:2611.33183v1 Announce Type: cross  Abstract: As AI systems become increasingly capable of executing and evaluating experiments, ideation is emerging as a bottleneck in automated scientific and engineering iteration. In many real-world domains, useful ideas are sparse and their validation is costly, making direct online exploration inefficient. We observe that repository evolution provides a scalable source of reusable technical experience: code changes record concrete technical hypotheses, while subsequent revisions, adoption, and reuse provide signals about their utility. Based on this observation, we propose REI, a Repository Evolution-driven Ideation framework that learns reusable ideation policies from repository evolution, injecting domain-specific technical experience into model parameters and reducing reliance on costly trial-and-error. We further identify a hidden failure mode in quality-driven ideation optimization: semantic diversity collapse. Although generated ideas c
    
[^262]: 稀疏自编码器中的邻域拥挤度可预测特征稳定性

    Neighbor Crowding Predicts Feature Stability in Sparse Autoencoders

    [https://arxiv.org/abs/2611.33182](https://arxiv.org/abs/2611.33182)

    该研究发现在TopK稀疏自编码器中，特征的跨种子稳定性仅凭单次训练得到的六个低成本统计量即可高度预测，其中字典的几何性质“邻域拥挤度”是最强预测因子，从而无需昂贵的多种子训练便能识别稳定特征。

    

    稀疏自编码器（SAE）提供了一种识别语言模型表征的方法，通过将多义特征分解为更稀疏、更可解释的特征来实现。然而，要识别哪些学习到的特征是真正稳定的、而非初始化产生的伪影，仍然颇具挑战。一种方法是对多个随机种子分别训练SAE，再通过余弦相似度比较不同种子学习到的特征，但这种方法的计算成本相当高。由此引出一个问题：能否在不付出训练多个种子所需计算开销的情况下识别稳定特征。在这项工作中，我们考察了不同训练token数量水平下的跨种子特征稳定性，并证明在TopK SAE中，仅凭从单次训练中获得的六个计算代价低廉的统计量，就能对这种稳定性进行高度准确的预测。其中，邻域拥挤度——字典的一种几何性质——是最强的单一预测因子，位于单位球面上密集区域的特征……（原文摘要至此截断）

    arXiv:2611.33182v1 Announce Type: cross  Abstract: Sparse Autoencoders (SAEs) offer an approach to identify language model representations by breaking apart polysemantic features into more sparse, interpretable ones. Yet, it is challenging to identify which of the learned features are actually stable, rather than artifacts of initialization. One approach is to train multiple SAE seeds, and compare the learned features across seeds by cosine similarity, but this is notably costly. This raises the question of whether stable features can be identified without the computational expense of training multiple seeds. In this work, we examine cross-seed feature stability across differing levels of training tokens, and demonstrate that in TopK SAEs, it is highly predictable using six computationally cheap statistics derived from a single training run. Neighbor crowding, a geometric property of the dictionary, is the strongest single predictor, with features that sit in dense regions of the unit 
    
[^263]: 基于深度学习的介观排放建模

    Deep Learning-Based Mesoscopic Emission Modeling

    [https://arxiv.org/abs/2611.33179](https://arxiv.org/abs/2611.33179)

    本研究提出基于GRU的深度学习模型DEP，通过学习由真实GNSS轨迹与微观排放模型MOBSim耦合生成的超百万条合成出行数据，直接预测路段级CO₂和NOx排放，在精度上媲美混合模型的同时实现了计算高效的介观尺度排放建模，并成功集成于法兰西岛地区的MATSim模拟流程。

    

    交通相关排放是城市空气污染的主要来源之一。然而，现有排放模型面临一个根本性的权衡：宏观模型计算效率高，但依赖平均速度，会导致系统性误差；而微观模型虽然精度高，但对于大规模模拟而言计算成本过高，令人望而却步。为弥合这一差距，本研究提出了直接排放预测（DEP），一种基于门控循环单元（GRU）的深度学习模型。DEP 通过将真实世界的 GNSS 速度轨迹与微观排放模型 MOBSim 相耦合，生成了超过一百万次合成出行数据用于训练。通过处理顺序的路段级输入，包括车辆特性、道路属性和环境温度，DEP 可预测 CO₂ 和 NOx 排放。我们将 DEP 集成到基于 MATSim 的工作流程中，并将其应用于法兰西岛（Île-de-France）地区。结果表明，DEP 的精度可与一种介观-微观混合模型 SPG-M 相媲美（原文此处截断）。

    arXiv:2611.33179v1 Announce Type: new  Abstract: Traffic-related emissions are a major contributor to urban air pollution. However, existing emission models face a fundamental trade-off: macroscopic models are computationally efficient but rely on average speeds, leading to systematic errors, while microscopic models achieve high accuracy but are computationally prohibitive for large scale simulations. To bridge this gap, this study introduces Direct Emission Prediction (DEP), a Gated Recurrent Unit (GRU)-based deep learning model. DEP was trained on over one million synthetic trips generated by coupling real-world GNSS speed profiles with the microscopic emission model MOBSim. By processing sequential link-level inputs, including vehicle characteristics, road attributes, and ambient temperature, DEP predicts CO$_2$ and NOx emissions. We integrated DEP into a MATSim-based workflow and applied it to the Ile-de-France region. DEP matches the accuracy of SPG-M, a hybrid mesoscopic-microsc
    
[^264]: 用于贝叶斯地球物理反演的地球物理信息引导可逆算子网络

    Geophysics-informed invertible operator network for Bayesian geophysical inversion

    [https://arxiv.org/abs/2611.33178](https://arxiv.org/abs/2611.33178)

    提出一种将可逆神经网络嵌入深度算子学习框架的地球物理信息引导可逆算子网络（GI-ION），实现了计算高效且物理一致的贝叶斯地球物理反演。

    

    由于观测数据稀疏和测量噪声的存在，地球物理反问题本质上是病态的（不适定的）。经典的基于采样的贝叶斯推断方法（如马尔可夫链蒙特卡罗，MCMC）由于需要进行大量正演模拟而在计算上代价高昂；而数据驱动的机器学习方法在训练数据有限时往往难以产生物理上一致的解。为解决这些问题，我们提出了一种地球物理信息引导的可逆算子网络，该方法将可逆神经网络集成到深度算子学习框架中，以实现高效的贝叶斯地球物理反演。INN作为分支网络，将模型参数转换为代表正演解的分支系数以及捕捉反问题不确定性的潜变量。通过学习正演算子并强制分支系数之间的统计独立性……（摘要内容在此处被截断）

    arXiv:2611.33178v1 Announce Type: new  Abstract: Geophysical inverse problems are inherently ill-posed because of sparse observations and measurement noise. Classical sampling-based Bayesian inference methods, such as Markov chain Monte Carlo (MCMC), are computationally prohibitive because they require many forward simulations. Data-driven machine-learning methods often struggle to produce physically consistent solutions when training data are limited. To address these issues, we propose a geophysics-informed invertible operator network (GI-ION) that integrates an invertible neural network (INN) within a deep operator-learning framework for efficient Bayesian geophysical inversion. The INN serves as the branch network and transforms the model parameters into branch coefficients that represent the forward solution and a latent variable that captures uncertainty in the inverse problem. By learning the forward operator and enforcing statistical independence between the branch coefficients
    
[^265]: Transformer评分模型的特征可解释性：一种面向自动写作评估的“保真度优先”框架

    Feature-Based Explainability of Transformer Scoring Models: A Fidelity-First Framework for Automated Writing Assessment

    [https://arxiv.org/abs/2611.33175](https://arxiv.org/abs/2611.33175)

    本研究提出一种“保真度优先”的事后解释框架，通过61个可解释的语言与写作质量特征，比较四种解释方法对DeBERTa写作评分模型预测的复现能力，发现随机森林代理模型保真度最高，从而在确保解释忠实于模型预测之后再解读特征重要性，提升了自动写作评估的透明度。

    

    基于Transformer的自动写作评分模型虽然能够与人类评分达到高度一致，但对其预测所对应的写作响应特征缺乏透明度。本研究评估了一种基于特征的事后解释方法，该方法在解读特征重要性之前，先确立解释的保真度。研究分析了六年级和八年级三个写作题目的英语语言艺术写作响应，每个题目约12,700份响应，并针对“表达”和“规范”两个维度分别训练了独立的DeBERTa-v3-base研究模型。每份响应由61个可解释的语言和写作质量特征来表示。研究比较了四种方法——理论驱动的复合特征、层次化特征分组、探索性因子分析以及随机森林代理模型——在复现DeBERTa预测方面的能力。结果表明，随机森林代理模型在全部六个题目-维度组合上均产生了最高的观测R²值。

    arXiv:2611.33175v1 Announce Type: new  Abstract: Transformer-based automated writing scoring models can achieve strong agreement with human scores while providing limited transparency into the response characteristics associated with their predictions. This study evaluated a feature-based post-hoc explanation approach in which explanatory fidelity was established before feature importance was interpreted. English Language Arts responses from three writing prompts in Grades 6 and 8 were analyzed, with about 12,700 responses per prompt and separate DeBERTa-v3-base research models for Expression and Conventions. Each response was represented by 61 interpretable linguistic and writing-quality features. Four approaches - theory-driven composites, hierarchical feature groupings, exploratory factor analysis, and Random Forest surrogates - were compared in their ability to reproduce DeBERTa predictions. Random Forest surrogates produced the largest observed R2 values across all six prompt-by-t
    
[^266]: 对称性预算：网络架构应该假设多少群结构？

    The Symmetry Budget: How Much Group Structure Should a Network Architecture Assume?

    [https://arxiv.org/abs/2611.33169](https://arxiv.org/abs/2611.33169)

    该论文提出“对称性预算”概念，在一个精确可解模型中以闭式形式推导出有限样本超额风险，证明当数据仅具有近似对称性时，网络架构中的中间对称性绑定程度永远不会严格最优。

    

    网络架构在很大程度上是一种对称性假设：卷积将权重在平移之间进行绑定，注意力机制将权重在置换之间进行绑定，而全连接层则不做任何绑定。当数据仅具有近似对称性时，这种绑定会在近似误差与估计误差之间进行权衡，而适当的绑定程度是数据集的属性，而非仅由架构本身决定。我们在一个精确可解的模型中研究这种权衡：循环群 $\mathbb{Z}_d$ 上信号的线性映射，其架构由 $\mathbb{Z}_d$ 的子群格索引，使得单个整数 $k \mid d$ 可以在无约束层（$k=1$，$d^2$ 个参数）和完全等变层（$k=d$，$d$ 个参数）之间进行插值。我们以闭式形式推导了有限样本超额风险，证明对于一般的对称性缺陷，主导阶风险与 $1/k$ 呈仿射关系，并证明此时中间架构永远不会严格最优：

    arXiv:2611.33169v1 Announce Type: cross  Abstract: A network architecture is, in large part, a symmetry assumption: convolution ties weights across translations, attention ties them across permutations, and a fully connected layer ties nothing. When the data are only approximately symmetric, tying trades approximation error against estimation error, and the right amount of tying is a property of the dataset, not of the architecture alone. We study this trade-off in an exactly solvable model: linear maps on signals over the cyclic group $\mathbb{Z}_d$, with architectures indexed by the subgroup lattice of $\mathbb{Z}_d$, so that a single integer $k \mid d$ interpolates between an unconstrained layer ($k=1$, $d^2$ parameters) and a fully equivariant one ($k=d$, $d$ parameters). We derive the finite-sample excess risk in closed form, show that for a generic symmetry defect the leading-order risk is affine in $1/k$, and prove that intermediate architectures are then never strictly optimal:
    
[^267]: 联邦学习中间歇性参与下的拜占庭脆弱性的时间性暴露

    Temporal Exposure to Byzantine Vulnerability Under Intermittent Participation in Federated Learning

    [https://arxiv.org/abs/2611.33168](https://arxiv.org/abs/2611.33168)

    本文提出动态参与暴露（DPE）概念，揭示联邦学习中的间歇性参与会将拜占庭鲁棒性从静态的群体属性转变为随机的时间性属性，并推导了每轮暴露概率及其多轮累积效应。

    

    在本文中，我们研究了参与度的变化对每一轮拜占庭脆弱性的影响。我们形式化了动态参与暴露（Dynamic Participation Exposure, DPE）这一现象：即全局拜占庭比例保持在聚合规则容忍阈值以下，而实际参与的客户端集合却违反或接近违反鲁棒聚合所要求的假设条件。在随机参与条件下，我们推导了每轮发生暴露的概率，并分析了活跃集合规模如何影响所观测到的拜占庭比例的变异性。我们进一步考察了暴露在多轮中的累积效应，并考虑了在有利轮次中选择性参与的对手。我们的分析表明，间歇性参与并不必然导致聚合失败；相反，它将拜占庭鲁棒性从静态的群体层面属性转变为一种随机性和时间性的属性。为缓解这一风险，

    arXiv:2611.33168v1 Announce Type: new  Abstract: In this paper, we study the effect of participation variability on per-round Byzantine vulnerability. We formalise Dynamic Participation Exposure (DPE), a phenomenon in which the global Byzantine fraction remains below an aggregation rule's tolerance threshold, while a participating client set violates, or approaches violation of, the assumptions required for robust aggregation. Under stochastic participation, we derive the probability of per round exposure and analyse how active-set size affects the variability of the observed Byzantine fraction. We further examine the accumulation of exposure over multiple rounds and consider adversaries that selectively participate during favourable rounds. Our analysis shows that intermittent participation does not automatically imply aggregation failure. Rather, it transforms Byzantine robustness from a static population-level property into a stochastic and temporal property. To mitigate this risk, 
    
[^268]: 代码可用性还不够：用于记录和扩展数据分析的架构

    Code availability is not enough: schemas for documenting and extending data analysis

    [https://arxiv.org/abs/2611.33167](https://arxiv.org/abs/2611.33167)

    该论文提出一种用自然语言记录数据分析中各项决策的架构，使大型语言模型能够据此生成并验证分析脚本，还能将单一分析自动扩展为多元分析。

    

    数据分析涉及一系列决策，从数据清洗到模型设定，每一项决策都可能影响最终报告的结果。复现该结果并不等同于理解产生该结果的决策序列及每项决策背后的理由：代码将这些决策与实现细节混杂在一起，而论文手稿虽然陈述了主要决策及其理由，却将其分散在各个章节中，并且常常遗漏次要决策。读者因此可能会问，如果采用不同的决策，结果将如何变化：这个问题本身很容易提出，但在实践中实现起来往往并不简单。在这项工作中，我们提出了一种架构，将数据分析表示为一组以自然语言描述的决策，大型语言模型可以直接从中生成并验证分析脚本。这一工作流程还可以将单一分析扩展为完整的多元宇宙分析，其规范以一种方式指定（原文在此处截断）。

    arXiv:2611.33167v1 Announce Type: new  Abstract: A data analysis involves a sequence of decisions, from data cleaning to model specification, each of which can affect the reported result. Reproducing that result is not the same as understanding the sequence of decisions that produced it and the reasons for each: the code mixes those decisions with implementation details, while the manuscript states the major decisions and their rationale but scatters them across its sections and often leaves out the minor ones. A reader may then ask how the result would change under a different decision: the question itself is easy to pose, but implementing it is often non-trivial in practice. In this work, we propose a schema that represents a data analysis as a collection of decisions in natural language, from which large language models (LLMs) can generate and validate analysis scripts directly. This workflow can also expand a single analysis into a full multiverse of alternatives, specified in an a
    
[^269]: 面向签名伪造检测的量子机器学习

    Quantum Machine Learning for Signature Forgery Detection

    [https://arxiv.org/abs/2611.33163](https://arxiv.org/abs/2611.33163)

    该论文提出了一种将冻结的 EfficientNetB0 特征提取器与量子兼容主成分投影及最近邻决策相结合的混合量子机器学习流水线，用于离线手写签名伪造检测，达到 92.62% 的准确率。

    

    离线手写签名仍然是法律和金融文件中主要的行为生物特征识别手段。由于笔压、书写时序和笔画顺序等信息的缺失，从静态光栅图像中检测高仿真伪造签名十分困难。本工作在公开的 Kaggle 图像语料库上研究了用于离线签名伪造检测的一系列量子机器学习流水线，该语料库的图像被划分为两个类别。我们将 ZZ 特征映射量子核与三种经典学习器相结合：K近邻、支持向量机和随机森林。随后，我们提出了一种更深层的混合模型，该模型使用 EfficientNetB0 作为冻结特征提取器，通过量子兼容的主成分投影将所得的 1280 维表示降维到与量子比特寄存器规模相匹配的子空间，并采用最近邻规则判定类别归属。在单一留出测试折上，所提出的混合模型达到了 92.62% 的准确率。在分层五折交叉验证上……（摘要原文在此处截断）

    arXiv:2611.33163v1 Announce Type: cross  Abstract: Offline handwritten signatures remain a primary behavioural biometric in legal and financial documents. Detecting skilled forgeries from a static raster image is hard because pen pressure, timing and stroke order are missing. This work studies a family of quantum machine learning pipelines for offline signature forgery detection on a public Kaggle corpus of images split into two classes. We pair a ZZ feature map quantum kernel with three classical learners: K-Nearest Neighbours, Support Vector Machine and Random Forest. We then propose a deeper hybrid that uses EfficientNetB0 as a frozen feature extractor, reduces the resulting 1280 dimensional representation to a qubit register sized subspace through a quantum compatible principal component projection and decides class membership with a nearest neighbour rule. On a single held out test fold the proposed hybrid reaches 92.62 percent accuracy. On stratified five fold cross validation th
    
[^270]: 利用交互式2D环境对推理型视觉语言模型智能体的失败进行归因

    Attributing Agentic Failure in Reasoning Vision-Language Models with Interactive 2D Environments

    [https://arxiv.org/abs/2611.33162](https://arxiv.org/abs/2611.33162)

    本文提出了专为失败归因而非排名设计的交互式迷宫环境 MultiNet v2.0-Gridworld，通过可独立操控的机制组合与基于广度优先搜索的精确最优步数评分，将视觉语言模型智能体的不同失败原因（误读环境、不解组件机制、操作顺序错误、无法纠错）区分开来。

    

    智能体基准测试所报告的成功率将多种不同的失败原因混为一谈：误读环境、始终无法弄清环境组件的运作方式、以错误的顺序操作组件、以及未能从错误中恢复，而每一种失败都需要不同的修复方案。我们展示了 MultiNet v2.0-Gridworld 的早期预览版——一个专为归因而非排名而构建的交互式迷宫环境。四种不同类型的机制，即钥匙-门配对、开关-闸门配对、干扰物与诱饵、以及这些机制的有序组合，可以被独立地添加、修改或移除，同时渲染器、动作空间和评分函数保持不变。通过对位置、朝向、物品栏和机制状态的联合状态进行广度优先搜索，每个实例都能获得精确的最优动作步数，该步数既设定了实例难度，也对未完成的回合进行评分。我们固定了评估协议

    arXiv:2611.33162v1 Announce Type: cross  Abstract: Agentic benchmarks report a success rate that confounds distinct causes of failure: misreading the environment, never working out how components of the environment operates, operating the components in the wrong order, and failing to recover from a mistake, each of which calls for a different fix. We present an early preview of MultiNet v2.0-Gridworld - an interactive maze environment built for attribution rather than ranking. Four different types of mechanisms, namely key-door pairs, switch-gate pairs, distractors and decoys, and ordered combinations of these mechanisms, can be added, modified, or removed independently while the renderer, the action space, and the scoring function stay fixed. A breadth-first search over the joint state of position, facing direction, inventory, and mechanism state gives every instance an exact optimal action count, which sets its difficulty and grades unfinished episodes. We fix the evaluation protocol
    
[^271]: OmniAct：一个策略，掌控百万具身形态

    OmniAct: One Policy, One Million Embodiments

    [https://arxiv.org/abs/2611.33158](https://arxiv.org/abs/2611.33158)

    OmniAct通过以机器人在具体场景中的运动学图（包含关节位置、运动轴及与视觉观测的连接关系）为条件，并用流匹配策略预测各节点的相对关节运动，实现了单一策略即可控制具有不同自由度和运动学配置的多种机器人具身形态。

    

    控制多种机器人的单一策略不仅需要在不同具身形态之间迁移任务知识，还必须根据每个机器人的具体能力调整其行为。现有方法通过共享的动作表示来促进这种迁移，但这些表示未必能捕捉机器人形态上的差异应如何塑造操作策略。我们提出OmniAct，这是一种跨具身形态的操作策略，它直接以特定场景中每个机器人的运动学信息为条件，生成针对该具身形态的行为。这种运动学图表示了机器人的关节位置、运动轴以及与其视觉观测之间的连接关系，使策略能够考虑其具身形态与场景中物体和表面之间的关系。一个流匹配（flow-matching）策略为每个节点预测相对关节运动，从而能够控制具有不同自由度和运动学配置的机器人。为了学习物理……（原文摘要在此处截断）

    arXiv:2611.33158v1 Announce Type: cross  Abstract: A single policy that controls diverse robots must not only transfer task knowledge across embodiments, but also adapt its behavior to each robot's capabilities. Existing approaches facilitate transfer through shared action representations, but these representations do not necessarily capture how differences in robot morphology should shape manipulation strategies. We introduce OmniAct, a cross-embodiment manipulation policy that conditions directly on each robot's kinematics in a specific scene to produce embodiment-specific behavior. This kinematic graph represents the robot's joint positions, motion axes, and connectivity to its visual observation, allowing the policy to account for how its embodiment related to objects and surfaces in the scene. A flow-matching policy predicts relative joint motions for each node, enabling control across robots with different degrees of freedom and kinematic configurations. To learn how physical str
    
[^272]: 强化学习可以从其自身的采样轨迹中恢复

    Reinforcement Learning Can Be Recovered From Its Own Rollouts

    [https://arxiv.org/abs/2611.33102](https://arxiv.org/abs/2611.33102)

    该论文发现，仅通过对存档的RL采样轨迹按其生成顺序进行最大似然训练、且无需区分响应的对错，就能从同一基础模型完全恢复出在线强化学习的性能，表明RL所获得的知识实质上蕴含在采样数据本身之中。

    

    广义而言，强化学习（RL）后训练通过在模型生成的数据上进行训练，将计算资源转化为模型能力。通常，采样轨迹在每次更新后都会被丢弃，RL过程中获得的知识仅存在于模型权重之中。在这项工作中，我们研究了这些知识有多少被保留在采样轨迹本身之中。一个自然的初步尝试是在存档的采样轨迹（而非新生成的轨迹）上重放原始的RL算法，但这会导致灾难性崩溃。我们为这一现象提供了一个简单的解释，并由此提出了一种补救方法：对存档的采样轨迹进行最大似然训练。令人惊讶的是，从同一个基础模型出发，按照采样轨迹生成的顺序进行重放，并且不区分正确与错误的响应，就足以恢复在线RL的性能。事实上，每个采样轨迹批次只需一次最大似然更新，就能接近（原文此处截断）

    arXiv:2611.33102v1 Announce Type: new  Abstract: Broadly speaking, reinforcement learning (RL) post-training converts compute into model capabilities by training on model-generated data. Typically, the rollouts are discarded after each update, and the knowledge acquired during RL resides in the model weights. In this work, we investigate how much of this knowledge is retained in the rollouts themselves. A natural first attempt is to replay the original RL algorithm on archived rather than freshly generated rollouts, but this leads to catastrophic collapse. We provide a simple explanation for this phenomenon, which in turn suggests a remedy: maximum likelihood on the archived rollouts. Surprisingly, starting from the same base model and replaying the rollouts in the order in which they were generated, without any distinction between correct and incorrect responses, is sufficient to recover the performance of online RL. In fact, a single maximum likelihood update per rollout batch closel
    
[^273]: 移位器量化：面向原生整数计算的紧凑表示

    Shifter Quantization: Compact Representations for Native Integer Computation

    [https://arxiv.org/abs/2611.33080](https://arxiv.org/abs/2611.33080)

    提出移位器量化（SQ）方法，通过追加和移位共享比特重构原生宽度整数操作数，使GPU整数矩阵运算能覆盖完整点积，在2比特精度下实现llama.cpp预填充吞吐量2.71-3.77倍的提升并降低KL散度。

    

    分组量化被用于大语言模型量化中，以在低位宽下提高精度，但对权重进行分组使得标准整数MMA（矩阵乘累加）无法跨越完整的点积进行运算，从而阻碍了高效的GPU推理。我们开发了一种新的量化方法——移位器量化（SQ），它通过追加和移位共享比特来重构原生宽度（原生整数宽度）的整数操作数。为了进一步提升效率，我们开发了SQ的一个特殊变体——移位整数（SHI），以使用INT4内核。我们表明，SQ和SHI格式在质量上（以投影误差和KL散度衡量）以及在推理速度上（在解码和预填充阶段测量）均优于标准的分组量化。对于像2比特量化这样的低精度场景，SQ2和SHI2在Q2_K检查点上实现了llama.cpp 4,096 token预填充吞吐量的2.71-3.77倍，并将WikiText-2 KL散度降低了6.6-7.8%。对于像4比特量化这样的中等精度场景，SQ4实现了1.43-2.47倍的预填充吞吐量。

    arXiv:2611.33080v1 Announce Type: new  Abstract: Groupwise quantization is used in LLM quantization to improve accuracy at low bit widths, but grouping weights keeps standard integer MMA from spanning the full dot product for efficient GPU inference. We develop a new quantization method, Shifter Quantization (SQ), that reconstructs native-width integer operands by appending and shifting shared bits. To further extend the efficiency, we develop a special variant of SQ, Shifter Integer (SHI), to use INT4 kernels. We show that SQ and SHI formats perform better than standard groupwise quantization in quality, as measured by projection error and KL divergence, and in inference speed, as measured during decoding and prefill. For low precision like 2-bit quantization, SQ2 and SHI2 achieve 2.71-3.77x the 4,096-token prefill throughput of llama.cpp on a Q2_K checkpoint and reduce WikiText-2 KL by 6.6-7.8%. For mid precision like 4-bit quantization, SQ4 achieves 1.43-2.47x the prefill throughput
    
[^274]: LEGOSpec：一个用于推测性草稿设计与训练的可组合高效框架

    LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training

    [https://arxiv.org/abs/2611.33004](https://arxiv.org/abs/2611.33004)

    LEGOSpec提出了一个可组合的推测性草稿设计与训练框架，将草稿模型分解为上下文、骨干、提议和目标四个组件，并通过系统化的对照实验揭示了不同工作负载下各组件的作用、相互作用及最优组合规律。

    

    推测性解码使用轻量级草稿模型来加速大语言模型（LLM）的生成。现有方法在不同的训练配置下评估完整的方案，导致各组件的效果、相互作用以及组合收益不明确。因此，我们提出了LEGOSpec，一个用于推测性草稿设计与训练的可组合框架。LEGOSpec将草稿模型分解为上下文、骨干、提议和目标四个组件，它们共享统一的训练基础设施。借助LEGOSpec，我们系统地改变各个组件，并在匹配的训练条件下评估选定的组合，以隔离它们各自的效果与相互作用。这些实验揭示了三个发现：（1）不同的工作负载偏好不同的骨干网络，这取决于上下文长度和内存预算，其中块内依赖建模对长上下文尤为重要。（2）提议设计必须在接受率与延迟之间取得平衡。（3）训练目标……（摘要截断）

    arXiv:2611.33004v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model (LLM) generation using a lightweight draft. Existing methods evaluate complete recipes under different training configurations, leaving component effects, interactions, and composition benefits unclear. Therefore, we present LEGOSpec, a composable framework for speculative draft design and training. LEGOSpec decomposes draft models into Context, Backbone, Proposal, and Objective components that share a unified training infrastructure. Using LEGOSpec, we systematically vary components and evaluate selected combinations under matched training conditions to isolate their effects and interactions. These experiments reveal three findings: (1) Different workloads favor different backbones, depending on context length and memory budget, with within-block dependency modeling particularly important for long contexts. (2) Proposal design must balance acceptance and latency. (3) The training o
    
[^275]: 具有求和聚合的局部MPNN的度数受限可满足性问题是NEXPTIME难的

    Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard

    [https://arxiv.org/abs/2611.32595](https://arxiv.org/abs/2611.32595)

    本文证明了具有局部求和聚合和ReLU激活的消息传递神经网络（MPNN）的度数受限可满足性判定问题是NEXPTIME难的，且即使度数上界固定为3，该难度结果仍然成立。

    

    我们证明了以下问题是NEXPTIME难的：给定一个具有局部求和聚合和ReLU激活函数的消息传递神经网络（MPNN）$M$，以及一个以一元编码表示的度数上界$d$，判断是否存在一个带有有理数节点标签且度数至多为$d$的指向图$(G,v)$，使得该图能被$M$接受。即使在固定的度数上界$d=3$的情况下，该难度结果依然成立。

    arXiv:2611.32595v1 Announce Type: cross  Abstract: We prove that the following problem is \textbf{NEXPTIME}-hard: given a message passing neural network (MPNN) $M$ with local sum aggregation and ReLU activations, and a degree bound $d$ encoded in unary, decide whether there is a pointed graph $(G,v)$ with rational node labels and degree at most $d$ that is accepted by $M$. Hardness already holds for the fixed degree bound $d=3$.
    
[^276]: ASTERISM：多模态大语言模型真的会看所有图像吗？

    ASTERISM: Do Multimodal LLMs Actually Look at All the Images?

    [https://arxiv.org/abs/2611.32316](https://arxiv.org/abs/2611.32316)

    该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。

    

    多图像基准旨在测试多模态大语言模型（MLLM）能否跨图像整合证据，但其中许多题目仅凭单张图像即可回答。通过对来自19个公开基准的约3.3万道题目进行审计，并采用两阶段单图过滤器，我们发现其中24.2%的题目仅凭一张图像或仅凭问题本身即可解答。在这些题目上，MLLM仅凭一张图像就能获得全部图像相对纯文本所带来的准确率提升的中位数78%，因此这些题目实际上奖励了“从单张图像作答”的行为，我们称之为单图捷径（single-image shortcut）。即便在单张图像不足以作答的题目上，MLLM也会走这条捷径：在这类题目上，它们的大部分错误答案与其仅凭单张图像时给出的答案重复，且该比例远高于随机水平。我们基于这19个基准构建了ASTERISM——据我们所知是规模最大的经过审计的多图像基准——仅保留其中需要多张图像才能作答的题目。其1.87万道题目（其中2000道经过修正或重写）共享同一（摘要在此处截断）。

    arXiv:2611.32316v1 Announce Type: cross  Abstract: Multi-image benchmarks aim to test whether a multimodal large language model (MLLM) combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them. On these items, MLLMs recover from one image a median 78% of the accuracy that all images add over the text alone, so these items reward answering from one image, which we call the single-image shortcut. MLLMs take this shortcut even on items where one image is not enough. On such items, most of their wrong answers repeat an answer they give from one image, at a rate well above chance. We build ASTERISM, to our knowledge the largest audited multi-image benchmark, from these 19 benchmarks by keeping their items that need more than one image. Its 18.7k items, 2.0k of them corrected or rewritten, share one
    
[^277]: 面向多保真度贝叶斯优化的融合高斯过程老虎机框架

    A Fused Gaussian Process Bandit Framework for Multi-Fidelity Bayesian Optimization

    [https://arxiv.org/abs/2611.32092](https://arxiv.org/abs/2611.32092)

    该论文将多保真度贝叶斯优化问题创新性地表述为融合高斯过程老虎机问题，并开发了具有理论性能保证的FGP-UCB算法，从而更准确地建模保真度层级间的复杂依赖关系。

    

    贝叶斯优化（BO）被广泛应用于直接评估成本高昂时对昂贵黑盒函数进行最大化。在许多实际应用中，高保真度目标可以通过一系列低保真度模型来近似，这些模型以大幅降低的成本提供部分但具有参考价值的信息信号。多保真度贝叶斯优化（MFBO）算法旨在战略性地组合这些信息源，以在精度与评估成本之间实现最优权衡。然而，现有的MFBO方法往往过度简化了保真度层级之间复杂的依赖关系，或依赖于缺乏理论性能保证的启发式采集函数。为了解决这些局限性，我们引入了一种将MFBO问题表述为融合高斯过程老虎机问题的新框架，并开发了相应的融合高斯过程置信上界（FGP-UCB）算法。我们的方法通过融合多保真度信息来准确预测高保真度函数。

    arXiv:2611.32092v1 Announce Type: new  Abstract: Bayesian optimization (BO) is widely used to maximize expensive black-box functions when direct evaluation is costly. In many real-world applications, the high-fidelity objective can be approximated by a series of lower-fidelity models that provide partial but informative signals at substantially reduced cost. Multi-fidelity BO (MFBO) algorithms aim to strategically combine these information sources to achieve an optimal trade-off between accuracy and evaluation cost. However, existing MFBO methods often oversimplify the complex dependencies between fidelity levels or rely on heuristic acquisition functions that lack theoretical performance guarantees. To address these limitations, we introduce a new formulation of the MFBO problem as a fused Gaussian Process bandit and develop a corresponding Fused Gaussian Process Upper Confidence Bound (FGP-UCB) algorithm. Our approach accurately predicts the high-fidelity function by fusing multi-fid
    
[^278]: Actionwave：面向多种医疗场景的以波形为中心的临床预测系统性基准

    Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios

    [https://arxiv.org/abs/2611.31249](https://arxiv.org/abs/2611.31249)

    Actionwave 是首个覆盖重症监护、手术室和急诊科等多种医疗场景的以波形为中心的临床预测系统性基准，通过统一的决策时点预测接口、严格的评估协议，系统考察了信号时长、预警时间和可用输入等条件对模型表现的影响。

    

    连续的床旁波形记录了快速的生理变化，可为临床决策提供信息。现有的基准测试仅考察了选定的预测条件，但目前尚不清楚模型优势如何依赖于信号时长、预警时间以及不同医疗场景下的可用输入。我们提出了 Actionwave，这是一个面向重症监护、手术室和急诊科场景的以波形为中心的临床预测基准。Actionwave 采用统一的决策时点预测接口，并根据推断的内容和时间对具有临床依据的任务进行分组。它改变了与临床使用相关的条件，包括干预前所需的预警时间。我们在指定的模型规模和计算预算下，比较了多种架构和预训练生物信号编码器。该协议要求患者级数据划分、严格的信息截断，以及能够隔离单个建模因素的比较。在 20 个任务和 140 万个任务标签……

    arXiv:2611.31249v1 Announce Type: cross  Abstract: Continuous bedside waveforms record rapid physiological changes that can inform clinical decisions. Existing benchmarks examine selected prediction conditions, but it remains unclear how model advantages depend on signal duration, warning time, and available inputs across care settings. We introduce Actionwave, a benchmark for waveform-centric clinical prediction across intensive-care, operating-room, and emergency-department settings. Actionwave uses a common decision-time prediction interface and groups clinically grounded tasks by what is inferred and when. It varies conditions relevant to clinical use, including the warning required before an intervention. We compare diverse architectures and pretrained biosignal encoders under specified model-size and compute budgets. The protocol requires patient-level splits, strict information cutoffs, and comparisons that isolate individual modeling factors. Across 20 tasks and 1.4M task-label
    
[^279]: 降低所有人的门槛：共情强化学习中隐藏的寻求认同目标

    Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL

    [https://arxiv.org/abs/2611.30702](https://arxiv.org/abs/2611.30702)

    在基于LLM模拟用户的共情强化学习中，寻求认同用户的占比每增加10个百分点，策略对有过错发帖人的认可率就上升3–4个百分点，表明策略会隐蔽地降低对所有人的认可门槛，而其辨别过错的能力并未下降。

    

    当强化学习从众多用户的反应中学习时，策略通常无法得知每个用户的真实诉求。我们探究：当部分用户私下希望获得认同时，策略会从他们身上学到什么——是只单独迎合这些用户（针对性偏袒），还是降低对所有人表示认可的门槛（阈值偏移）。我们在一个设定中回答了这一问题：针对LLM模拟用户的共情强化学习（RLVER、GRPO、Qwen2.5-7B-Instruct）。我们仅改变寻求认同用户的占比（0–50%；RLVER默认值为24.6%），并在1000条带有社区过错判定的Reddit AITA帖子上进行评估。结果表明，策略发生了阈值偏移：寻求认同用户占比每提高10个百分点，对有过错发帖人的认可率就上升3–4个百分点；在没有这类用户时，Qwen模型保持在基线水平；且该效应在匹配KL散度以及目标盲奖励下依然存在。信号检测分析显示，表示认可的判定标准下降，而对有过错与无过错发帖人的辨别能力并未下降。一个明确的请求……（原文摘要在此处截断）

    arXiv:2611.30702v1 Announce Type: cross  Abstract: When RL learns from the reactions of many users, the policy usually cannot see what each user wants. We ask what it learns from users who privately want validation: to validate them alone (targeting), or to lower its bar for endorsing everyone (a threshold shift). We answer this in one setting, empathetic RL against LLM-simulated users (RLVER, GRPO, Qwen2.5-7B-Instruct): we vary only the share of validation-seeking users (0--50%; RLVER's default: 24.6%) and evaluate on 1,000 Reddit AITA posts with community fault verdicts.   The policy shifts its threshold. Endorsement of at-fault posters rises by 3--4 percentage points per 10 points of share; without such users the Qwen model stays at base, and the effect persists at matched KL and under a goal-blind reward. Signal-detection analysis shows that the criterion for endorsing falls while discrimination between at-fault and not-at-fault posters does not decline. An explicit request to be t
    
[^280]: 物理约束的神经常微分方程：从道路进出时间学习交通动力学与车辆轨迹

    Physics-Constrained Neural ODEs for Learning Traffic Dynamics and Trajectories from Road Entry/Exit Times

    [https://arxiv.org/abs/2611.30577](https://arxiv.org/abs/2611.30577)

    该论文提出一种物理约束的神经常微分方程方法，仅凭稀疏的路段进出时间与领航车位置数据，即可学习物理上合理的交通跟驰动力学并重建车辆轨迹。

    

    我们研究如何从摄像头记录的路段进出时间戳（EET）以及领航排头探测车的位置来学习微观交通动力学并重建车辆轨迹。由于每辆车的位置仅在两个点被测量，其间的轨迹未知，因此动力学难以恢复。通过将神经常微分方程与局部密度近似相结合，我们学习了一个跟驰模型，其速度-密度关系由神经网络表示。在如此稀疏的数据下，无约束的网络可能学到物理上不现实的关系，例如负速度或完全拥堵道路上非零的速度。为此，我们引入了一种归一化方法，将其应用于带有可学习状态切换的单调神经网络，以强制满足端点条件，从而保证速度-密度关系的可容许性，同时可证明地保留了对任意可容许速度-密度关系的逼近能力。

    arXiv:2611.30577v1 Announce Type: new  Abstract: We consider the problem of learning microscopic traffic dynamics and reconstructing vehicle trajectories from section entry/exit timestamps (EET) recorded by cameras and from the positions of a probe vehicle leading the platoon. The dynamics are difficult to recover as each vehicle's position is measured only at two points, and its trajectory between them is unknown. By combining Neural ODEs with a local density approximation, we learn a follow-the-leader model whose speed--density relation is a neural network. With such sparse data, an unconstrained network may learn physically unrealistic relations, such as negative speeds or non-zero speeds on a fully congested road. We therefore introduce a normalization that, applied to a monotone neural network with a learned regime change, enforces the endpoint conditions, guaranteeing an admissible speed--density relation while provably retaining the ability to approximate any admissible one. We 
    
[^281]: 小波行列式采样器：自适应于正则性的最优逼近速率

    Wavelet determinantal samplers: Optimal approximation rates adapting to regularity

    [https://arxiv.org/abs/2611.29992](https://arxiv.org/abs/2611.29992)

    该论文构造了基于小波的行列式点过程采样器，在流形与几何图上对线性统计量实现了 n^{1-2s/m} 阶的方差缩减，且该逼近速率能够自适应于观测量的 Hölder/Sobolev 正则性。

    

    我们在流形与几何图上构造了基于小波的行列式采样器，其逼近速率能够自适应于观测量的正则性。在维度为 $m$ 的黎曼流形上，我们的小波行列式点过程（DPP）对于 $s$-Hölder 观测量和 $H^s$ 观测量的线性统计量实现了 $n^{1-2s/m}$ 阶的方差缩减，其中 $0 < s < m/2$。

    arXiv:2611.29992v1 Announce Type: cross  Abstract: We construct wavelet-based determinantal samplers on manifolds and geometric graphs whose approximation rates adapt to the regularity of the observable. On a Riemannian manifold of dimension $m$, our wavelet DPPs satisfy variance reduction of order $n^{1-2s/m}$ for linear statistics of both $s$-H\"older and $H^s$ observables, where $0
    
[^282]: 从决策轨迹推断显微实验中的实验目标

    Inferring Experimental Objectives in Microscopy Experiments from Decision Traces

    [https://arxiv.org/abs/2611.29814](https://arxiv.org/abs/2611.29814)

    本文提出从显微镜操作员的决策历史中学习奖励函数以推断其实验目标，并比较了多种从操作员对测量位置的选择偏好中重建奖励函数的算法框架。

    

    现代显微镜获取、分析并处理数据的速度远远超过人类操作员决定下一步测量内容的速度。要在自动化实验中充分利用这种速度，需要仪器遵循操作员的科学逻辑，而编码这种逻辑的一种方式是通过成为操作过程一部分的奖励函数。本文探讨了是否可以从显微镜操作员的决策历史中学习这种奖励函数。我们提出了多种框架，将操作员在不同候选测量位置之间做出的记录选择作为关于其偏好的比较证据，并比较了多种重建奖励函数的算法，包括基于具有物理意义的图像描述符构建的可解释线性模型、基于相同描述符的非线性函数、无监督图像表示结合偏好建模，以及一种受深度核启发、经偏好训练的模型。结果表明……

    arXiv:2611.29814v1 Announce Type: cross  Abstract: Modern microscopes can acquire, analyze and act on data far faster than human operators can decide what to measure next. Exploiting this speed in automated experiments requires the instrument to follow the scientific logic of the operator, and one way to encode that logic is through reward functions that become part of the operation process. Here we explore whether such reward functions can be learned from the decision histories of microscope operators. We propose multiple frameworks in which recorded choices among alternative measurement locations serve as comparative evidence about the operator's preferences, and we compare several algorithms for reconstructing the reward including interpretable linear models built on physically meaningful image descriptors, nonlinear functions of the same descriptors, an unsupervised image representation followed by preference modeling, and a preference trained deep kernel inspired model. The result
    
[^283]: 点击与计数：面向类无关目标计数的多模态原型自适应

    Click and Count: Multimodal Prototype Adaptation for Class-Agnostic Object Counting

    [https://arxiv.org/abs/2611.28321](https://arxiv.org/abs/2611.28321)

    该论文提出“点击与计数”多模态原型自适应框架，通过冻结分割模型将单次点击扩展为精确目标掩码，并在迭代原型细化中逐步融合文本描述，从而去除了边界框标注需求，实现类无关目标计数。

    

    我们提出了一种多模态、类无关的基于密度的目标计数框架，该框架在单一的迭代细化流程中，通过视觉示例、文本描述或两者结合的方式来指定感兴趣的目标对象。现有的计数方法通常分为两类：一类融合手绘边界框与文本描述两种模态来指定目标对象，另一类是仅依赖边界框示例或纯文本的单一模态方法，近来少数方法则用点或点击替代边界框。我们借助冻结的视觉提示分割模型将一次点击扩展为精确的目标掩码，并将其送入一个迭代式的原型细化过程，该过程在每一步细化中同时融合文本描述。我们做出了两项贡献：其一，提出了由点击导出的示例提取模块，去除了以往基于示例的计数方法对边界框的依赖；其二，提出了逐步文本融合机制，将视觉与（原文摘要在此处被截断）

    arXiv:2611.28321v1 Announce Type: cross  Abstract: We propose a multimodal, class-agnostic, density-based object counting framework that specifies the object of interest through visual exemplars, a text description, or both, within a single iterative refinement pipeline. Existing counters typically fall into two groups: methods that fuse two modalities, a hand-drawn bounding box and a text description, to specify the target object, and single-modality methods that rely on either a box exemplar or text alone. A few recent methods replace the box with a point or click. We expand a single click into a precise object mask via a frozen visual-prompt segmentation model, feeding an iterative prototype-refinement process that also fuses a text description at every refinement step. We make two contributions: first, a click-derived exemplar extraction module that removes the bounding-box requirement of prior exemplar-based counters; second, a per-step text-fusion mechanism that combines visual a
    
[^284]: 迈向ICU患者的数字孪生：生成式事件模型中的病情预测与治疗反应

    Toward a digital twin of the ICU patient: Forecasting and treatment response in a generative event model

    [https://arxiv.org/abs/2611.27028](https://arxiv.org/abs/2611.27028)

    研究者训练了7900万参数的生成式事件模型TWICE，无需任务特定训练即可通过模拟50条患者未来轨迹，在19项ICU临床结局预测上达到与144个监督模型最佳配置相当的性能，是实现ICU患者数字孪生的重要一步。

    

    重症监护病房（ICU）患者的数字孪生应能预测患者的病情走向以及某项治疗将带来的改变。生成式事件模型——即经过训练以延续患者电子健康记录（EHR）的模型——被视为实现此类数字孪生的一条途径。我们在一个学术医疗系统的ICU记录上训练了TWICE，一个7900万参数的仅解码器（decoder-only）transformer模型（训练数据包含127,332名患者、15.7亿个token）。我们在临床预测任务、生成保真度、个体轨迹和反事实验证等方面对其进行了评估。在没有任何任务特定训练的情况下，通过从50个模拟的未来轨迹中读取19项临床结果，TWICE在留出患者上的表现与144个监督学习配置中的最佳者相当（平均AUROC 0.836 对比 0.834）。它在一天内会改变诊疗决策的事件预测上领先，而在死亡预测上表现稍逊。在训练数据固定的情况下，生成更多且更长的模拟未来，其收益超过了将模型规模扩大6.8倍。其生成的记录与真实记录相匹配……

    arXiv:2611.27028v1 Announce Type: new  Abstract: A digital twin of a patient in intensive care would forecast the patient's course and what a treatment would change. Generative event models, trained to continue a patient's electronic health record (EHR), have been proposed as a path toward such twins. We trained TWICE, a 79M-parameter decoder-only transformer, on the ICU records of one academic health system (127,332 training patients and 1.57 billion tokens). We evaluated it on clinical prediction tasks, generation fidelity, individual trajectories and counterfactual validation. Reading 19 clinical outcomes off 50 simulated futures, with no task-specific training, TWICE matched the best of 144 supervised configurations on held-out patients (mean AUROC 0.836 vs. 0.834). It led on events that change care within a day and trailed on death. With the training data fixed, more and longer simulated futures were worth more than a 6.8-fold larger model. Its generated records matched real ones 
    
[^285]: 从一开始就学会搜索：面向语言模型的搜索原生预训练

    Search From the Start: Search-Native Pretraining for Language Models

    [https://arxiv.org/abs/2611.26809](https://arxiv.org/abs/2611.26809)

    提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。

    

    现代语言模型（LM）越来越多地在推理阶段配备搜索工具进行部署。然而，预训练仍然主要聚焦于对静态文本的预测，搜索能力通常是在后训练阶段才被引入。我们提出“搜索原生预训练”作为一种实用路径，在预训练阶段就基于普通文本语料教会模型搜索能力。具体而言，我们大规模地用网络搜索查询来增强预训练语料，并使用本地索引检索网络响应。在这种增强语料上的训练使模型学会生成搜索调用，并在语言建模过程中融入检索到的证据。在Wikipedia和DCLM-Edu语料库以及360M至3.3B的参数规模上，搜索预训练模型在知识密集型任务上表现出更高的性能，并且对检索到的证据具有更强的可控性（可引导性）。关键的是，在对搜索预训练模型和常规模型进行监督微调之后，它们的优势依然得以保持。

    arXiv:2611.26809v1 Announce Type: cross  Abstract: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conve
    
[^286]: SparseHPS：跨深度相机与激光雷达的人体姿态与形状统一模型

    SparseHPS: A Unified Model for Human Pose and Shape across Depth Cameras and LiDARs

    [https://arxiv.org/abs/2611.26730](https://arxiv.org/abs/2611.26730)

    提出首个传感器无关的统一模型SparseHPS，仅需一套权重即可从深度相机或激光雷达采集的人体点云中恢复SMPL-X三维人体姿态、形状与全局平移，同时适用于单帧和视频流输入。

    

    从点云中恢复三维人体姿态与形状是动作捕捉、机器人和自动驾驶汽车的核心任务，因为深度相机和激光雷达能够以真实的米制三维直接感知人体，并且可以在任意光照条件下工作。然而，现有的方法各自只针对一种传感器类型：不同设备之间的点密度相差十倍以上，公开数据集采用互不兼容的人体模型进行标注，而且激光雷达方法需要较长的输入序列来弥补稀疏扫描的不足。在这项工作中，我们提出了SparseHPS，一种新颖的传感器无关网络，它使用同一套权重即可从人体点云中恢复SMPL-X姿态、形状和全局平移，对单帧和流式输入均适用。具体而言，SparseHPS使用LitePT骨干网络在稀疏体素上编码点云，用可学习的查询表示每个关节并通过一个统一的解码头进行解码，通过在体素上投票来定位人体，并从先前帧令牌的因果记忆中获取时序上下文……

    arXiv:2611.26730v1 Announce Type: cross  Abstract: Recovering 3D human pose and shape from point clouds is central to motion capture, robotics and self-driving vehicles, because depth cameras and LiDARs sense a person directly in metric 3D and work under any lighting. However, current approaches are each tailored to one sensor type: point density differs by more than ten times between devices, public datasets are labeled with incompatible body models, and LiDAR approaches need long input sequences to offset sparse scans. In this work, we present SparseHPS, a novel sensor-agnostic network that recovers SMPL-X pose, shape and global translation from a human point cloud with one set of weights, for single frames and streams alike. Specifically, SparseHPS encodes the cloud with a LitePT backbone over sparse voxels, represents every joint by a learnable query decoded with one common head, localizes the body by voting over voxels, and draws temporal context from a causal memory of tokens fro
    
[^287]: 分布式深度对冲

    Distributional Deep Hedging

    [https://arxiv.org/abs/2611.26652](https://arxiv.org/abs/2611.26652)

    本文提出分布式深度对冲方法，通过学习对冲损益分布的分位数，使风险偏好可在训练后作为权重灵活应用，从而将风险敏感对冲问题简化为一系列风险中性问题，避免了为每个新风险偏好重新训练策略的需求。

    

    深度对冲通过在神经网络策略上优化风险偏好来学习对冲策略。风险偏好在训练之前是固定的，因此每个新的偏好都需要一个新的策略，并且在actor-critic方法中还需要一个新的critic（评论家网络）。我们提出了分布式深度对冲，该方法通过对冲损益分布的分位数来学习损益分布，并在训练之后应用风险偏好，将其作为这些分位数的权重。我们刻画了允许这种表示的风险偏好，包括预期损失及其混合形式，并证明熵效用可以从相同的分位数计算得出。在当前策略下，这些偏好中的每一个都有一个下界，该下界是具有相同梯度的期望效用。这一界将风险敏感对冲问题转化为一系列风险中性问题，由此我们推导出了策略搜索和actor-critic算法。在数值实验中，学习到的对冲策略接近参考（原文摘要似乎在此处截断）。

    arXiv:2611.26652v1 Announce Type: cross  Abstract: Deep hedging learns a hedging strategy by optimising a risk preference over a neural-network policy. The preference is fixed before training, so each new preference requires a new policy and, in actor-critic methods, a new critic. We propose distributional deep hedging, in which the distribution of the hedging profit and loss is learned through its quantiles and the risk preference is applied afterwards, as a weight on those quantiles. We characterise the preferences that admit this representation, which include expected shortfall and its mixtures, and show that the entropic utility is computed from the same quantiles. At the current policy, each of these preferences is bounded below by an expected utility with the same gradient. This bound reduces risk-sensitive hedging to a sequence of risk-neutral problems, from which we derive policy-search and actor-critic algorithms. In numerical experiments, the learned hedges approach reference
    
[^288]: 重放而非重采样：长时程强化学习专家的离策略整合

    Replay, Don't Resample: Off-Policy Consolidation of Long-Horizon RL Experts

    [https://arxiv.org/abs/2611.24059](https://arxiv.org/abs/2611.24059)

    提出多专家回放策略蒸馏（MRPD），通过按专家的学习顺序回放其RL训练日志（而非代价高昂的重新采样）来整合多个长时程强化学习专家，从而大幅降低整合成本并缓解训练与推理的不匹配。

    

    前沿模型的后训练遵循一个通用方案：先用强化学习（RL）在每个领域分别训练一个专家模型，然后通过多教师在线策略蒸馏（MOPD）将这些专家整合为一个模型。这种整合方式代价高昂：学生模型需要在每个环境中重新采样，专家需要对每个采样的token进行打分，而且学生会继承最终专家模型的漂移。然而我们认为，专家的RL训练过程其实已经产生了整合所需的一切内容：带有采样概率和奖励的采样轨迹，以及专家从中学习的先后顺序。我们提出多专家回放策略蒸馏（MRPD），它按照专家自身的学习顺序将这些训练日志回放给学生模型。在每一步，策略梯度在这些已被打分的轨迹上重新计算，最终专家模型的分布则在被记录的前缀处进行蒸馏。遵循专家自身的轨迹减少了训练与推理之间的不匹配，并保持了……（原文摘要在此截断）

    arXiv:2611.24059v1 Announce Type: cross  Abstract: Frontier post-training follows a common recipe: train one expert per domain with reinforcement learning (RL), then consolidate the experts into one model via multi-teacher on-policy distillation (MOPD). Such consolidation is expensive: the student samples in every environment again, an expert scores every sampled token, and the student inherits the drift of the final experts. Yet we argue that the experts' RL runs have already produced what consolidation needs: rollouts with their sampling probabilities and rewards, in the order the experts learned from them. We propose multi-expert replay policy distillation (MRPD), which replays these training logs through the student in the experts' order. At each step, the policy gradient is recomputed on these scored trajectories, and the final experts' distributions are distilled at the logged prefixes. Following the experts' own trajectory reduces the training-inference mismatch and keeps the st
    
[^289]: 通过记忆摆脱潜在陷阱：面向世界模型规划的无训练情景核排斥方法

    Escaping Latent Traps by Remembering: Training-Free Episodic Kernel Repulsion for World-Model Planning

    [https://arxiv.org/abs/2611.22470](https://arxiv.org/abs/2611.22470)

    提出一种无需训练的情景核排斥代价项，利用智能体已访问潜在状态的新颖性门控记忆惩罚候选计划，帮助世界模型规划摆脱局部极小值“潜在陷阱”，在两个JEPA世界模型和四个基准上将规划成功率显著提升。

    

    潜在世界模型通过最小化预测潜变量与目标潜变量之间的距离来向目标图像进行规划。然而这种距离并非测地距离：它在墙后、角落和死胡同中存在局部极小值，即“潜在陷阱”，而无记忆的滚动时域重规划会使智能体持续被困其中，不断重访远离目标的状态。我们提出情景核排斥，这是一种无需训练的代价项，通过针对智能体实际访问过的潜在变量所构成的新颖性门控记忆库，利用 top-k 高斯核对候选计划进行惩罚，并根据当前基础代价进行重新缩放，使其强度在不同模型之间具有可比性。在两个 JEPA 世界模型和四个基准测试上，在相同的模型、规划器和预算条件下，EKR 将平均成功率从 31.8% 提升至 46.7%（LeWM），从 54.7% 提升至 64.0%（H-JEPA）。相同记忆下的反事实实验表明，在大多数设置中，EKR 会将规划推向远离已访问状态的方向，在导航任务中往往以更高的基础代价为代价；卡住的情形……

    arXiv:2611.22470v1 Announce Type: new  Abstract: Latent world models plan towards a goal image by minimising the distance between predicted and goal latents. This distance is not geodesic: it has local minima, latent traps, behind walls, at corners and in dead ends, and memoryless receding-horizon replanning keeps the agent in them, revisiting states far from the goal. We propose Episodic Kernel Repulsion (EKR), a training-free cost term that penalises candidate plans by a top-$k$ Gaussian kernel to a novelty-gated memory of the latents the agent has actually visited, rescaled by the current base cost so that its strength is comparable across models. Across two JEPA world models and four benchmarks, EKR raises average success from 31.8% to 46.7% (LeWM) and from 54.7% to 64.0% (H-JEPA) with the same model, planner and budget. Same-memory counterfactuals show that, in most settings, EKR pushes plans away from visited states, in navigation, often at a higher base cost; stuck episodes and 
    
[^290]: 提升图神经网络的细粒度表达能力

    Improving fine-grained expressivity of graph neural networks

    [https://arxiv.org/abs/2611.20020](https://arxiv.org/abs/2611.20020)

    本论文通过刻画几何消息传递网络在稀疏图上的表达能力、利用表示理论分析多重置换等变线性层、研究保持包含关系的置换不变集合表示，并依次提出EGenNet、MASNet和FSW-GNN三种新架构，系统性地提升了图神经网络的细粒度表达能力与稳定性。

    

    本论文研究了面向结构化数据的神经网络表达能力，重点关注图神经网络与对称性感知架构。我们提出了四项主要贡献。首先，我们刻画了几何消息传递神经网络在稀疏图上的一般表达能力，并提出了实现相应保证的EGenNet架构。其次，我们利用表示理论刻画了对多种置换（包括圈积对称性）等变的线性层。第三，我们研究了保持包含关系的置换不变集合表示，建立了维度界和不可能性结果，并提出了MASNet。最后，我们提出了FSW-GNN，这是一种兼具强表达能力、双Lipschitz稳定性保证的消息传递架构，并在长程图学习任务上展现出优异的实证表现。这些结果共同为图神经网络表达能力提供了更广泛的理解。

    arXiv:2611.20020v1 Announce Type: cross  Abstract: This thesis investigates the expressive power of neural networks for structured data, focusing on graph neural networks and symmetry-aware architectures.   We present four main contributions. First, we characterize the generic expressive power of geometric message-passing neural networks on sparse graphs and introduce EGenNet, an architecture achieving the corresponding guarantees. Second, we use representation theory to characterize linear layers equivariant to multiple permutations, including wreath-product symmetries. Third, we study permutation-invariant set representations that preserve containment, establish dimensional bounds and impossibility results, and introduce MASNet. Finally, we propose FSW-GNN, a message-passing architecture combining strong expressivity and bi-Lipschitz stability guarantees with strong empirical performance on long-range graph-learning tasks.   Together, these results provide a broader understanding of 
    
[^291]: 基于强化学习的钙钛矿太阳能电池富集导向虚拟筛选

    Enrichment-Oriented Virtual Screening of Perovskite Solar Cells through Reinforcement Learning

    [https://arxiv.org/abs/2611.18043](https://arxiv.org/abs/2611.18043)

    该论文提出一个以冻结TabPFN代理模型引导的强化学习框架，将钙钛矿太阳能电池配方生成建模为富集导向的虚拟筛选，使前十名候选配方的平均PCE从19.68%提升至23.74%，其中PPO和RPPO策略在前100名候选名单中获得了最高的邻域命中率。

    

    钙钛矿太阳能电池的实验优化受限于吸收层组成、器件层结构和制备工艺的组合多样性，这使得虚拟筛选在制备前对配方进行优先排序方面极具吸引力。我们开发了一个代理模型引导的强化学习框架，将配方生成转化为虚拟筛选，在高性能实验样本的邻域内富集候选配方短名单。一个冻结的表格先验数据拟合网络引导配方生成，同时一个包含890个实测器件的参考池支持基于邻域的评估。在1,846个器件上的五折评估表明，前十个选择可将平均功率转换效率（PCE）从19.68%提升至23.74%。在每次运行3,000次代理评估的条件下，将三种学习到的策略与六个基线方法进行了比较。在前100名候选名单中，近端策略优化（PPO）和循环PPO（RPPO）实现了邻域支持的命中率23%（原文在此处截断）。

    arXiv:2611.18043v1 Announce Type: new  Abstract: Experimental optimization of perovskite solar cells is constrained by combinatorial diversity in absorber compositions, device layers, and processing, which makes virtual screening attractive for prioritizing recipes before fabrication. We develop a surrogate-guided reinforcement-learning framework that casts recipe generation as virtual screening, enriching candidate shortlists near high-performing experimental neighbors. A frozen Tabular Prior-data Fitted Network (TabPFN) guides recipe generation, while a reference pool of 890 measured devices enables neighborhood-based assessment. Five-fold evaluation on 1,846 devices shows top-ten selection raises mean power conversion efficiency (PCE) from 19.68\% to 23.74\%. Three learned policies are compared with six baselines under 3,000 surrogate evaluations per run. In top-100 shortlists, Proximal policy optimization (PPO) and recurrent PPO (RPPO) achieve neighborhood-supported hit rates of 23
    
[^292]: 模仿策略学习本体感知锚点：感知噪声及其闭环代价的预测定律

    Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost

    [https://arxiv.org/abs/2611.17658](https://arxiv.org/abs/2611.17658)

    该论文揭示了模仿策略会将机器人自身状态作为“本体感知锚点”来折扣含噪感知读数，其权重服从 α=V/(V+s²) 的预测定律，且锚点方差远小于线性模型暗示值，一次测量即可精确预测策略在不同任务和噪声水平下对感知噪声的敏感度。

    

    机器人操作的模仿策略通常在有噪声的物体位置读数上进行训练，它们学会对含噪读数进行折扣。我们展示了它们折扣的程度及其原因。由于演示者会朝物体方向移动，机器人自身的状态携带着物体位置的信息。训练好的策略将这一本体感知锚点与读数相结合，并以 α = V/(V + s²) 对读数加权，其中 s² 是读数噪声，V 是锚点的误差方差。V 属于学习器本身，远小于由演示数据建立的线性模型所暗示的值，这正是经典变量误差回归高估 α 的原因。V 只需测量一次，即可在两个 robomimic 任务、两个学习器和四个新噪声水平下预测十个保留样本的敏感度，平均绝对误差为 0.025，而经典 EIV 方法为 0.37；此外，一个无参数的偏差测试排除了更简单的解释，即策略只是单纯高估……

    arXiv:2611.17658v1 Announce Type: cross  Abstract: Imitation policies for robot manipulation are often trained on noisy object positions, and they learn to discount the noisy reading. We show how much they discount it, and why. Because demonstrators steer toward the object, the robot's own state carries information about where the object is. A trained policy combines this proprioceptive anchor with the reading and weights the reading by $\alpha = V/(V + s^2)$, where $s^2$ is the reading noise and $V$ the anchor's error variance. $V$ belongs to the learner and is far smaller than a linear model of the demonstrations implies, which is why classical errors-in-variables (EIV) regression overestimates $\alpha$. Measured once, $V$ predicted ten held-out sensitivities across two robomimic tasks, two learners and four new noise levels, with mean absolute error 0.025 against 0.37 for classical EIV, and a parameter-free bias test rules out the simpler explanation that the policies just overestim
    
[^293]: 不确定性在哪里被观测？对大语言模型不确定性量化的机制性理解

    Where Is Uncertainty Observed? A Mechanistic Understanding of Uncertainty Quantification in LLMs

    [https://arxiv.org/abs/2611.15735](https://arxiv.org/abs/2611.15735)

    该论文基于机制可解释性提出了观测不确定性的四种透镜框架（结构、表征、分布、语义），并揭示了观测点决定不确定性量化估计器的信息获取能力与恢复效果这一关键机制。

    

    了解大语言模型（LLM）的响应何时可以被信任对于其安全使用至关重要。不确定性量化（UQ）可以帮助LLM识别其响应是否可能不可靠。现有的分类体系按不确定性来源或估计机制来组织UQ方法，但尚未解决一个关键问题：在模型的计算过程中，不确定性究竟在哪里被观测。这个问题的答案非常重要，因为观测点决定了估计器是否能够获得足够的信息来恢复给定目标，以及其恢复能力有多强。借助机制可解释性研究，我们定义了四种机制观测透镜：结构透镜观测组件更新，表征透镜观测隐藏状态，分布透镜观测词元概率，语义透镜观测意义空间上的分布。我们通过残差流将这些透镜关联起来，并按它们所携带的信息量对其进行排序。

    arXiv:2611.15735v1 Announce Type: cross  Abstract: Knowing when a large language model's (LLM's) response can be trusted is critical for its safe use. Uncertainty quantification (UQ) can help LLMs recognize if their responses may be unreliable. Existing taxonomies organize UQ methods by uncertainty source or estimation mechanism. They leave unresolved the question of where uncertainty is observed in the model's computation. The answer matters because the observation point determines whether an estimator has access to enough information to recover a given target, and how well it can recover it. Drawing on mechanistic interpretability, we define four mechanistic observation lenses: the structural lens observes component updates, the representational lens observes hidden states, the distributional lens observes token probabilities, and the semantical lens observes distributions over meanings. We relate these lenses through the residual stream and order them by the information they carry, 
    
[^294]: 未训练网络在V1建模上是否优于训练网络取决于归一化约定：来自两个fMRI数据集的证据

    Whether untrained networks beat trained ones at V1 depends on a normalisation convention: evidence from two fMRI datasets

    [https://arxiv.org/abs/2611.15533](https://arxiv.org/abs/2611.15533)

    通过对THINGS和NSD两个fMRI数据集的系统分析，本研究表明未训练卷积网络在V1建模上的所谓优势并非本质性的，而是由归一化约定和是否去除共有响应轮廓等分析选择所驱动——在预设定的岭回归读出和标准归一化下，训练网络始终表现更优。

    

    有报道称，未训练的卷积网络作为初级视觉皮层（V1）的模型可以媲美甚至超越训练过的网络，我们也曾在THINGS fMRI数据中报告过这种优势。然而，在预先设定的岭回归读出方案下，训练过的网络在THINGS和NSD的每一个汇聚网格上都领先（NSD全部8名受试者均是如此），即使体素响应在扫描内进行了中心化处理，结果依然如此。THINGS中的表征相似性分析（RSA）结果依赖于分析选择（28项分析中：5项区间高于0，9项低于0，14项包含0）；而在采用标准归一化的NSD中，28项分析没有一项支持未训练网络。未训练网络的优势只在以下条件下出现：当体素模式中保留所有试次共有的响应轮廓、模型单元未做中心化、并使用相关距离时——在THINGS中去除会话平均轮廓后，该优势从+0.030降至+0.005；而使用原始beta值时，该优势在THINGS中达到+0.100，并出现在THINGS测试图像中……（摘要原文在此处截断）

    arXiv:2611.15533v2 Announce Type: replace-cross  Abstract: Untrained convolutional networks have been reported to rival or beat trained networks as models of primary visual cortex (V1), and we had reported such an advantage in THINGS fMRI. Under a pre-specified ridge readout, the trained network was ahead at every pooling grid in THINGS and in NSD (8 of 8 subjects), also when voxel responses were centred within runs. Representational similarity analysis (RSA) in THINGS depended on analysis choices (28 analyses: intervals above 0 in 5, below 0 in 9, including 0 in 14); in NSD with its standard normalisation, none of 28 analyses favoured the untrained network. The untrained advantage appears when a response profile common to all trials is left in the voxel patterns and model units are not centred, under correlation distance: removing the session-mean profile in THINGS reduced it from +0.030 to +0.005; with raw betas it reached +0.100 in THINGS, appeared in the THINGS test images, and app
    
[^295]: 一个在十种生理信号上预训练的心肺基础模型，用于围手术期与重症监护

    A Cardiorespiratory Foundation Model Pretrained on Ten Physiological Signals for Perioperative and Critical Care

    [https://arxiv.org/abs/2611.15004](https://arxiv.org/abs/2611.15004)

    CARMEN是一个在十种心肺生理波形上无标签预训练的基础模型，通过LSCNorm机制保留绝对信号水平，能灵活接受任意信号组合，并在围手术期与重症监护的19项任务中超越了监督模型和其他基础模型。

    

    动脉压、静脉压、肺动脉压和颅内压以及呼吸机波形指导着手术室和重症监护病房的临床决策，然而现有的生理波形基础模型几乎完全建立在心电图和光电容积脉搏波图之上。我们提出了CARMEN，一个单一编码器，在十种心肺波形上进行了无标签预训练（共227万模态小时）。CARMEN可以接受从10秒到5分钟的任意共录信号组合，并且通过一种名为LSCNorm的条件化机制恢复了归一化所丢失的绝对信号水平；若没有该机制，术中低血压预测的AUROC将从0.894下降至0.780。在来自韩国和美国队列的19个任务上，该冻结编码器在15个任务中的12个上优于监督模型，在19个任务中的14至17个上优于三个基础模型（一个通用模型和两个生物信号模型），且没有任何一个模型在任何任务上超越它。仅凭心电图或光电容积脉搏波图（摘要在此处截断）。

    arXiv:2611.15004v1 Announce Type: cross  Abstract: Arterial, venous, pulmonary artery and intracranial pressures and ventilator waveforms guide decisions in operating rooms and intensive care units, yet foundation models for physiological waveforms have been built almost entirely on the electrocardiogram and photoplethysmogram. We present CARMEN, a single encoder pretrained without labels on ten cardiorespiratory waveforms (2.27 million modality-hours). CARMEN accepts any combination of co-recorded signals from 10 s to 5 min, and a conditioning mechanism, LSCNorm, restores the absolute signal level that normalization removes; without it, the AUROC for intraoperative hypotension fell from 0.894 to 0.780. Across 19 tasks from Korean and US cohorts, the frozen encoder outperformed supervised models on 12 of 15 and three foundation models, one general-purpose and two for biosignals, on 14 to 17 of 19, with none exceeding it on any task. Given the electrocardiogram or photoplethysmogram alo
    
[^296]: 材料发现中贝叶斯优化的开箱即用表格基础模型

    Out-of-the-Box Tabular Foundation Models for Bayesian Optimization in Materials Discovery

    [https://arxiv.org/abs/2611.12491](https://arxiv.org/abs/2611.12491)

    该论文证明 TabPFN 和 TabICL 等表格基础模型可以作为开箱即用的代理模型，在材料发现的贝叶斯优化中媲美高斯过程等传统方法，无需核函数选择或任务特定训练。

    

    贝叶斯优化被广泛应用于材料科学的主动学习循环中。高斯过程是事实上的代理模型家族，但需要依赖具体任务的核函数选择，并且可能缺乏对复杂关系的建模能力。表格基础模型（TFMs）最近已成为结构化数据预测任务的最先进方法，无需定制核函数或超参数调优。我们评估了两个表格基础模型 TabPFN 和 TabICL，在两个标准的材料科学基准上作为开箱即用的代理模型，并与涵盖高斯过程、随机森林和 CatBoost 的大量基线方法进行比较。我们的研究结果表明，TFMs 在这两个基准上都是极具竞争力的代理模型，且无需针对主动学习的任务特定训练或 GP 核函数选择。我们的结果为将 TFMs 进一步发展为材料发现中主动学习的代理模型提供了动力。

    arXiv:2611.12491v3 Announce Type: replace  Abstract: Bayesian optimization is widely used in active learning loops for materials science. Gaussian processes serve as the de-facto surrogate model family, but require task-dependent kernel choices and can lack the modeling capacity for complex relationships. Tabular foundation models (TFMs) have recently emerged as the state-of-the-art for predictive tasks on structured data, without requiring bespoke kernels or hyperparameter tuning. We evaluate two TFMs, TabPFN and TabICL, as out-of-the-box surrogate models on two standard materials science benchmarks, comparing against a large field of baselines encompassing Gaussian processes, random forests, and CatBoost. Our findings indicate that TFMs are highly competitive surrogate models across the two benchmarks, without task-specific training for active learning or GP kernel selection. Our results motivate developing TFMs further as surrogates for active learning in materials discovery.
    
[^297]: Ostinato：面向本地智能体的低延迟CPU-GPU混合专家模型推理

    Ostinato: Low-Latency CPU-GPU MoE Inference for Local Agents

    [https://arxiv.org/abs/2611.11533](https://arxiv.org/abs/2611.11533)

    提出Ostinato引擎，通过在每次前向传播中原位复用一个静态的跨层GPU池，实现CPU与GPU协同的混合专家模型推理，为本地智能体提供低延迟支持。

    

    语言模型智能体越来越多地运行在配备消费级GPU的本地工作站上，其中量化的混合专家（MoE）模型将路由专家分配在GPU与主机内存之间。带有前缀缓存的智能体会话在单token解码与简短的工具返回预填充之间交替进行，通常最多几百个token。简短的工具调用轮次所触及的专家数量超出了消费级GPU的容纳能力，然而却有约85%的DeepSeek-V4-Flash路由被引导至一个静态GPU池，同时每个冷专家平均只承载3.3-5.5个token。广泛使用的本地引擎（如llama.cpp和KTransformers）要么每轮通过PCIe复制数十GB的数据，要么在CPU上计算所有路由而让常驻的GPU专家闲置，要么按整层和统一的每层配额来规划GPU内存。我们提出了Ostinato，一个CPU-GPU混合专家引擎，它在每次前向传播中原位复用一个静态的跨层GPU池。在解码和简短工具调用轮次期间，Ostinato在GPU上计算常驻路由

    arXiv:2611.11533v2 Announce Type: replace-cross  Abstract: Language-model agents increasingly run on local workstations with consumer GPUs, where quantized mixture-of-experts (MoE) models split routed experts between GPU and host memory. Prefix-cached agent sessions alternate single-token decode with short tool-return prefills, typically up to a few hundred tokens. Short tool turns touch more experts than consumer GPUs can hold, yet direct roughly 85% of DeepSeek-V4-Flash's routes to a static GPU pool while leaving 3.3-5.5 tokens per cold expert. Widely used local engines such as llama.cpp and KTransformers either copy tens of gigabytes over PCIe per turn, compute every route on the CPU while idling resident GPU experts, or budget GPU memory by whole layers and uniform per-layer quotas. We present Ostinato, a CPU-GPU MoE engine that reuses one static cross-layer GPU pool in place across every forward pass. During decode and short tool turns, Ostinato computes resident routes on the GPU
    
[^298]: 相同行为，不同机制：有限实验能对整合信息认证什么

    Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information

    [https://arxiv.org/abs/2611.08469](https://arxiv.org/abs/2611.08469)

    本文在具有已知真实答案的小型随机网络中精确刻画了整合信息 φ 等机制级量何时可被识别、连续且可由有限数据认证，并证明输入输出行为完全相同的网络可携带截然不同的整合信息（0.28 比特 vs 0 比特），但只需 4-53 次内部记录试验即可将其区分。

    

    可解释性、AI 审计以及机器意识评估越来越依赖于机制层面的量：这些量是系统因果转移律的泛函，描述系统如何计算，而非它输出什么。实验只能通过有限的、部分的且有噪声的干预来观察这一规律。我们探究这样的量何时能从实验的访问权限中被识别、对机制连续、并能从有限数据中得到认证，并针对一个刻意苛刻的情形——整合信息理论中的整合信息 $\varphi$——在具有已知真实答案的四单元和五单元随机网络中给出了精确的回答。行为可能是沉默的：一个循环网络与一个前馈网络在其响应延迟处具有完全相同的输入输出规律，分别携带 0.28 和 0 比特，但内部记录在 4-53 次试验内即可将二者区分开来。部分访问可能同样盲目：在我们的例子中，单步记录中被排除的单元会造成精度下限

    arXiv:2611.08469v1 Announce Type: cross  Abstract: Interpretability, AI auditing and assessments of machine consciousness increasingly rest on mechanism-level quantities: functionals of a system's causal transition law that describe how it computes, not what it outputs. Experiments see that law only through finite, partial and noisy interventions. We ask when such a quantity is identifiable from an experiment's access, continuous in the mechanism and certifiable from finite data, and answer exactly for a deliberately demanding case - the integrated information $\varphi$ of integrated information theory - in stochastic networks of four and five units with known ground truth. Behavior can be silent: a recurrent and a feed-forward network with identical input-output laws at their response delay carry 0.28 and 0 bits, yet internal recordings separate them within 4-53 trials. Partial access can be just as blind: in our examples, units left out of a one-step recording impose precision floors
    
[^299]: 规模化ConceptARC：研究推理模型中的概念理解

    ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models

    [https://arxiv.org/abs/2611.06732](https://arxiv.org/abs/2611.06732)

    该论文提出ConceptARC at Scale，一个能够大规模生成数千个针对特定概念的ARC风格任务的生成器，用于系统研究大型推理模型能否在概念相关的推理任务上形成可泛化的抽象能力。

    

    大型推理模型（LRMs）在具有挑战性的基准测试中表现优异，但它们能否在概念相关的推理任务之间形成可泛化的抽象仍不明确。抽象与推理语料库（ARC）为研究抽象能力提供了一个受控的测试平台，要求求解者从少量示例中推断二维网格上的变换规则。虽然ARC评估的是单个任务内的规则抽象能力，但不同任务可能共享更高层次的结构，即便它们需要不同的变换——例如，多个任务可能都涉及对“被包含在其他对象中的对象”进行推理。像ConceptARC这样的基准通过将ARC风格的任务按反复出现的空间和语义概念进行分组来捕捉这种结构，但它们通常每个概念只包含少量手工构建的任务。我们提出了ConceptARC at Scale，这是一个生成器，可以创建数千个针对特定概念的ARC风格任务。

    arXiv:2611.06732v1 Announce Type: new  Abstract: Large reasoning models (LRMs) perform well on challenging benchmarks, yet it remains unclear whether they form generalizable abstractions across conceptually related reasoning tasks. The Abstraction and Reasoning Corpus (ARC) provides a controlled testbed for studying abstraction, requiring solvers to infer transformation rules over two-dimensional grids from a small set of examples. While ARC evaluates rule abstraction within individual tasks, different tasks may share higher-level structure despite requiring different transformations$\unicode{x2014}$for example, several tasks may involve reasoning about objects contained within other objects. Benchmarks such as ConceptARC capture this structure by grouping ARC-style tasks into recurring spatial and semantic concepts, but they often contain only a few manually constructed tasks per concept. We introduce ConceptARC at Scale, a generator for creating thousands of ARC-style tasks targeting
    
[^300]: 面向非阿贝尔格点规范理论的可泛化等变扩散模型：分析基础与数值标度测试

    Generalizable Equivariant Diffusion Models for Non-Abelian Lattice Gauge Theory: Analytical Foundations and Numerical Scaling Tests

    [https://arxiv.org/abs/2611.03530](https://arxiv.org/abs/2611.03530)

    本文为非阿贝尔格点规范理论中的规范等变扩散模型建立了严格的分析框架，包括群流形上逆时随机微分方程的推导、去噪损失的规范不变性论证，以及关于晶格体积和耦合常数标度行为的系统性数值验证。

    

    规范等变扩散模型可用于生成非阿贝尔格点规范理论中的场构型系综。此前，我们通过将基于L-CNN的扩散模型与Metropolis修正的退火朗之万算法（MAALA）相结合，证明了建模此类理论的可行性。本文重点研究其背后的分析框架。具体而言，我们对群流形上的逆时随机微分方程进行了严格推导，讨论了去噪损失函数的规范不变性，并针对晶格体积与逆耦合常数β的标度行为提供了大量的数值基准测试。此外，文中还给出了前向扩散过程中任意尺寸Wilson环以及拓扑荷磁化率的完整解析表达式。

    arXiv:2611.03530v2 Announce Type: cross  Abstract: Gauge-equivariant diffusion models can be used to generate ensembles of field configurations in non-abelian lattice gauge theories. Previously, we demonstrated the feasibility of modeling these theories by combining L-CNN-based diffusion models with the Metropolis-adjusted annealed Langevin algorithm (MAALA). Here we focus on the underlying analytical framework. In particular, we present a rigorous derivation of the reverse-time stochastic differential equation on the group manifold, discuss the gauge invariance of the denoising loss function, and provide extensive numerical benchmarks regarding the scaling behavior with respect to the lattice volume and the inverse coupling $\beta$. Full analytical expressions for arbitrarily sized Wilson loops and the topological charge susceptibility during the forward process are given as well.
    
[^301]: 轮廓感知的被丢弃参数Hessian质量诊断

    Profile-aware diagnostics of discarded parameter-Hessian mass

    [https://arxiv.org/abs/2611.01335](https://arxiv.org/abs/2611.01335)

    提出一种轮廓感知的诊断框架，通过在不同分组规模和组内曲率轮廓参考下测量参数Hessian的块外平方占比，来量化块对角曲率模型所丢弃的曲率质量，并评估候选修正的实际效果。

    

    评估块对角曲率模型涉及两个问题：它们省略了多少曲率，以及一个特定的修正在有限步长内改变了什么。我们针对不同组规模以及观测到的组内曲率轮廓的参考基准，测量参数Hessian的块外平方占比。在CIFAR-100的ResNet-18及其无跳跃连接的对应模型中，相似的初始丢弃占比伴随着不同的曲率轮廓。在已保存的训练好的ResNet-18权重处，将分组方式从阶段组改为层组，会使估计的块外质量从轮廓包络之下移动到其之上；一个近似条件筛选方法在所有六个种子/输入单元中都解决了这一比较问题，而某些无跳跃连接模型之间的比较仍未得出结论。另外，在已保存的ResNet-50/Stanford Cars状态下，我们通过相同的Hessian信息映射，比较了由组外Hessian作用、剩余模型残差以及二者组合所构成的候选修正。在单位尺度下，仅组外输入（摘要在此处被截断）

    arXiv:2611.01335v2 Announce Type: replace  Abstract: Evaluating block-diagonal curvature models involves two questions: how much curvature they omit, and what a specific correction changes at a finite step. We measure the squared off-block fraction of the parameter Hessian against references for group sizes and the observed within-group curvature profile. In CIFAR-100 ResNet-18 and a skip-free counterpart, similar initial discarded fractions accompany different profiles. At saved trained ResNet-18 weights, changing from stage to layer groups moves estimated off-block mass from below to above the profile envelope; an approximate conditional screen resolves this comparison in all six seed/input cells, while some skip free comparisons remain unresolved. Separately, in saved ResNet-50/Stanford Cars states, we compare candidates formed from off-group Hessian action, the remaining model residual and their combination through the same Hessian-informed map. At unit scale off-group input alone 
    
[^302]: InterQ-RL：量化智能体强化学习中交互感知的执行不匹配校正

    InterQ-RL: Interaction-Aware Correction of Execution Mismatch in Quantized Agentic RL

    [https://arxiv.org/abs/2610.35303](https://arxiv.org/abs/2610.35303)

    InterQ-RL 针对量化智能体强化学习中残留的 rollout-学习器执行不匹配问题，提出通过对到达状态的策略历史施加有界执行前缀校正来修复对齐后仍存在的不匹配。

    

    多轮智能体强化学习（RL）依赖长闭环 rollout，使生成与 KV-cache 成本成为主要瓶颈。量化 rollout 可以降低这些成本，但其低精度执行器同时充当行为策略，可能引发与学习器之间的执行不匹配。现有的量化 RL 方法主要针对单轮生成设计，它们将学习器前向计算与量化 rollout 对齐，或校正当前 token 的 rollout-学习器比率，却没有对产生每个智能体状态的策略历史进行建模。即使经过量化对齐，残留的 rollout-学习器不匹配仍会沿着到达状态的策略历史不断累积，因此在历史不匹配的情况下，当前动作在局部上看似是对齐的。我们提出 InterQ-RL（交互感知量化强化学习）来校正这种对齐后的不匹配。它对到达状态的策略历史施加有界的执行前缀校正，（摘要在此处截断）

    arXiv:2610.35303v1 Announce Type: new  Abstract: Multi-turn agent reinforcement learning (RL) relies on long closed-loop rollouts, making generation and KV-cache costs a major bottleneck. Quantized rollout reduces these costs, but its low-precision executor also serves as the behavior policy and can induce execution mismatch with the learner. Existing quantized-RL methods are designed primarily for single-turn generation. They align the learner forward with quantized rollout or correct the current-token rollout-learner ratio, without modeling the policy history that produces each agent state. Even after quantization alignment, residual rollout-learner mismatch accumulates along the state-reaching policy history, so a current action may look locally aligned despite a mismatched history. We introduce InterQ-RL (Interaction-aware Quantized Reinforcement Learning) to correct this post-alignment mismatch. It applies bounded execution-prefix correction over the state-reaching policy history,
    
[^303]: 强化学习中的最优策略追踪

    Best Policy Tracking in Reinforcement Learning

    [https://arxiv.org/abs/2610.34501](https://arxiv.org/abs/2610.34501)

    该论文通过实证研究揭示了强化学习训练中性能非单调波动对策略选择的影响，并引入事后诊断指标来评估常用的策略选择标准能否可靠地追踪到训练过程中的最佳策略。

    

    强化学习在单任务设置中通常表现出非单调的性能动态。在学习过程中，会生成一系列策略，这些策略的真实性能是未知的，且可能在迭代之间显著波动。尽管这是一个众所周知的现象，标准的实验实践通常只报告最后访问的策略或周期性评估中的最佳策略，这隐含地假设了这些选择标准能够可靠地近似整个训练过程中表现最好的策略。在这项工作中，我们对单任务基于策略的强化学习中非单调性对策略选择的影响进行了实证研究。我们引入了事后诊断指标来刻画性能退化，并评估不同选择标准的准确性和效率。这些指标使我们能够在标准连续控制任务上评估常用的选择策略。

    arXiv:2610.34501v1 Announce Type: new  Abstract: Reinforcement learning commonly exhibits non-monotonic performance dynamics in single-task settings. During learning, a sequence of policies is generated whose true performance is unknown and may oscillate significantly across iterations. Despite this well-known phenomenon, standard experimental practice typically reports either the final visited policy or the best among periodically evaluated ones, implicitly assuming that these selection criteria reliably approximate the best-performing policy visited throughout the entire sequence.   In this work, we present an empirical study on the implications of non-monotonicity for policy selection in single-task policy-based reinforcement learning. We introduce post-hoc diagnostic metrics to characterize performance degradation and assess the accuracy and efficiency of different selection criteria. These metrics allow us to evaluate commonly used selection strategies across standard continuous-c
    
[^304]: 差分隐私算法组合与子采样的高斯近似

    Gaussian Approximation for Composition and Subsampling of Differentially Private Algorithms

    [https://arxiv.org/abs/2610.33333](https://arxiv.org/abs/2610.33333)

    该论文证明只要子采样机制组合中每一步的隐私损失在除小概率情形外都很小，其组合的隐私损失分布就近似服从具有相同均值和方差的高斯分布，从而为DP-SGD的隐私分析提供了无需对数因子损失或噪声尺度限制条件的更优界限。

    

    差分隐私随机梯度下降（DP-SGD）的隐私分析依赖于子采样机制组合的界限。这种组合的隐私性由一对乘积分布的隐私性所界定，其隐私损失是独立项之和，已通过多种概念和集中性结果进行了分析。然而，现有的闭式界限要么损失一个对数因子，要么对噪声尺度施加限制性条件，并需要对子采样进行精细分析。此外，尽管理论上众所周知独立随机变量之和近似服从高斯分布，但现有的基于中心极限定理（CLT）的技术给出的近似在 $\delta$ 上存在较大误差。我们证明，只要每一步在除小概率情形外都具有较小的隐私损失，这对分布的隐私损失就近似服从高斯分布。具体而言，其隐私损失分布接近于一个具有相同均值和方差的高斯分布……

    arXiv:2610.33333v1 Announce Type: new  Abstract: The privacy analysis of differentially private stochastic gradient descent (DP-SGD) relies on bounds for the composition of subsampled mechanisms. The privacy of such a composition is bounded by that of a pair of product distributions, whose privacy loss is a sum of independent terms and has been analyzed via a variety of notions and concentration results. Yet, existing closed-form bounds either lose a logarithmic factor or require restrictive conditions on the noise scale and delicate analysis of subsampling. Further, while it is well known that sums of independent random variables are approximately Gaussian, existing CLT-based techniques give an approximation with a large error in $\delta$.   We show that the privacy loss of this pair is approximately Gaussian as long as each step has a small privacy loss except with small probability. Specifically, its privacy loss distribution is close to a Gaussian with the same mean and variance, a
    
[^305]: 流形上流匹配的相容似然

    Compatible Likelihoods for Flow Matching on Manifolds

    [https://arxiv.org/abs/2610.31981](https://arxiv.org/abs/2610.31981)

    该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。

    

    在最简单的非欧几里得流形——一维平坦环面上，我们展示了一个分布，其中最近提出的黎曼流匹配扩展方法被一个将样本推离数据分布的速度场全局优化。我们将这种失败模式追溯到一个我们称之为“相容性”的性质：如果一个基于似然的流匹配目标的总体最小化器能够恢复由数据和耦合所诱导的边际速度场，则称该目标是相容的。我们给出了相容性的一般条件，将其应用于现有方法，并通过在相容性条件下消失的残差分量来界定最坏情况误差。此外，我们注意到现有的相容目标将概率密度置于紧流形上不可行的速度上。为此，我们提出了一个基于端点上指数族构建的目标，以条件速度作为充分统计量，该目标既满足相容性，又仅支持可行的速度。

    arXiv:2610.31981v1 Announce Type: cross  Abstract: On the simplest non-Euclidean manifold, the 1D flat torus, we exhibit a distribution where a recently proposed extension of Riemannian flow matching is globally optimized by a velocity field that pushes samples away from the data distribution. We trace this failure mode to a property we call compatibility; a likelihood-based flow matching objective is compatible if its population minimizer recovers the marginal velocity field induced by the data and coupling. We give a general condition for compatibility, apply it to existing methods, and bound the worst-case error in terms of a residual component vanishing under compatibility. Next, we note that existing compatible objectives put density on velocities that are infeasible on compact manifolds. We propose an objective built from an exponential family over endpoints, with the conditional velocity as sufficient statistic, which is both compatible and supported only on feasible velocities.
    
[^306]: 条件流模型将认知不确定性转化为表观的偶然方差

    Conditional Flow Models Convert Epistemic Uncertainty into Apparent Aleatoric Variance

    [https://arxiv.org/abs/2610.29172](https://arxiv.org/abs/2610.29172)

    该论文发现条件流模型生成的样本方差会系统性地大于真实条件分布的方差，这种额外的方差本质上是模型将拟合真实条件分布时的认知不确定性转化为生成样本中表观的偶然方差。

    

    条件流模型（CFMs）在机器学习的各个领域正被越来越多地用来替代确定性预测器，包括那些传统上被表述为回归的任务。鉴于它们在此类任务中的成功，我们提出了一个问题：CFM生成的条件分布在点预测器所能提供的信息之外，还携带了哪些有用的信息？我们识别出一个系统性现象：在许多自然的情形下，即使在近乎确定性的回归问题中，CFM样本的方差通常也大于真实条件分布的方差。这一效应在模型尚未充分解析出对条件变量的依赖关系之处最为显著，因此这种额外的方差可以被视为将拟合真实条件分布时的认知不确定性转化为生成样本中表观的偶然方差。通过受控的合成数据和真实数据实验，以及与指数族基线的比较……

    arXiv:2610.29172v1 Announce Type: new  Abstract: Conditional Flow Models (CFMs) are increasingly being used in place of deterministic predictors across machine learning, including for tasks that were classically formulated as regression. Given their success in such tasks, we ask the question: what useful information do the conditional distributions produced by CFMs carry beyond what a point predictor would provide?   We identify a systematic phenomenon: in many natural regimes, CFM samples routinely have larger variance than the true conditional distribution, even in nearly deterministic regression problems. This effect is most pronounced where the model has not resolved the dependence on the conditioning variable, and thus the additional variance can be viewed as converting epistemic uncertainty in fitting the ground truth conditional distribution into apparent aleatoric variance in the generated samples. Through controlled synthetic and real-data experiments, and comparisons with exp
    
[^307]: STAR-XPC：用于估算聚合物复合材料X射线透射率的集合Transformer

    STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites

    [https://arxiv.org/abs/2610.28709](https://arxiv.org/abs/2610.28709)

    本文提出基于集合Transformer的STAR-XPC模型，可在1-3毫秒内快速准确估算聚合物复合材料的X射线透射比，替代耗时超过5小时的蒙特卡罗模拟。

    

    在辐射屏蔽研究中，评估X射线束穿过聚合物复合材料的能量透射比是一项具有挑战性的任务。作为参考解的蒙特卡罗输运模拟方法精度很高，但计算成本极其昂贵：我们的每个参考模拟被限制在40亿（4×10⁹）光子历史数和20,000秒（超过5小时）的时间上限内，而闭式解析方法又无法表征如此复杂的场景。我们提出了一种人工神经网络模型，将分子组分的无序多重集、其质量分数以及四个整体物理描述符直接映射到透射比 I(t)/I₀，推理时间仅需1-3毫秒。该模型是一个集合Transformer（Set Transformer），在结构上具有置换不变性，并通过固定大小的填充集合接受最多含11种组分的不同规模混合物。每种组分均使用化学领域预训练的Transformer语言模型ChemBE进行嵌入……

    arXiv:2610.28709v1 Announce Type: new  Abstract: Evaluating the energy transmission ratio of an X-ray beam through a polymer composite is a challenging task in radiation shielding studies. The reference solution with Monte Carlo transport simulation is highly accurate but computationally expensive: each of our reference simulations is capped at $4\times10^9$ photon histories and a 20,000 s (over 5 hours) time limit, while closed-form analytical methods fail to represent such complex scenarios. We present an artificial neural network model that maps the unordered multi-set of molecular constituents, their mass fractions, and four bulk physical descriptors directly to the transmission ratio $I(t)/I_0$ in 1-3 milliseconds. The model is a Set Transformer, which makes it permutation invariant by construction and accepts mixtures of different sizes up to 11 constituents within a fixed-size padded set. Each constituent is embedded with a chemistry-pretrained transformer language model, ChemBE
    
[^308]: 流形上逻辑斯蒂分类的Besov函数高斯逼近

    Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds

    [https://arxiv.org/abs/2610.27022](https://arxiv.org/abs/2610.27022)

    该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。

    

    我们研究了连通、紧致、光滑的嵌入流形 $X\subset\mathbb R^d$（无边界，内在维度为 $\varrho$）上的高斯核逼近与正则化逻辑斯蒂分类问题。两个主要挑战分别是：在不假设贝叶斯函数有界的前提下推导快速的预言机不等式，以及构造既能利用内在维度又能利用目标函数正则性、且不出现饱和现象的高斯逼近器。对于第一个挑战，我们通过一个Bernstein型不等式加以解决，该不等式中的常数与预测器的上确界范数呈线性依赖关系。结合局部化的复杂度估计，由此得到了可分赋范函数空间上的预言机不等式。针对第二个挑战，我们建立了一个一致的Sobolev算子展开，并利用不同带宽下高斯核积分算子的带符号组合来抵消其低阶偏差项。随后通过实插值方法，进而得到Besov函数的 $L_2(X)$ 逼近（摘要原文在此处截断）。

    arXiv:2610.27022v1 Announce Type: cross  Abstract: We study Gaussian kernel approximation and regularized logistic classification on a connected compact smooth embedded manifold $X\subset\mathbb R^d$ without boundary and of intrinsic dimension $\varrho$. Two main challenges are to derive a fast oracle inequality without assuming a bounded Bayes function and to construct Gaussian approximants that exploit both the intrinsic dimension and the target's regularity without saturation. We address the first challenge through a Bernstein-type inequality whose constant depends linearly on the supremum norm of the predictor. Combined with localized complexity estimates, this yields an oracle inequality over separable normed function spaces. To address the second challenge, we establish a uniform Sobolev operator expansion and cancel its lower-order bias terms using a signed combination of Gaussian kernel integral operators at different bandwidths. Real interpolation then gives an $L_2(X)$ approx
    
[^309]: 二部网络中的联邦社区检测

    Federated Community Detection in Bipartite Networks

    [https://arxiv.org/abs/2610.25997](https://arxiv.org/abs/2610.25997)

    提出FLICKER与BiFLICKER方法，在原始数据不可共享的联邦环境下实现了二部网络的谱估计与社区检测，并提供了非渐近收敛与误聚类率的理论保证。

    

    我们研究了当一个大矩阵被水平划分到无法共享原始数据的多台服务器上时的联邦谱估计问题。我们提出了FLICKER，一种针对特定目标的基于梯度的幂迭代方法，它仅通过在本地数据块上执行矩阵-向量乘积，即可恢复与预定谱目标相关联的奇异方向。不同的目标方向可以独立且并行地恢复。我们建立了非渐近收敛保证，并提供了一种用于谱目标选择的递归网格搜索程序。基于FLICKER，我们开发了用于二部网络联邦社区检测的BiFLICKER，并针对相近或重复的奇异方向提出了顺序投影变体。在稀疏二部随机块模型下，我们证明了来自参考服务器的经过适当缩放的主奇异值可以提供有效的全局目标，并且顺序BiFLICKER实现了趋近于零的误聚类率。

    arXiv:2610.25997v2 Announce Type: replace-cross  Abstract: We study federated spectral estimation when a large matrix is horizontally partitioned across servers that cannot share raw data. We propose FLICKER, a target-specific gradient-based power iteration that recovers a singular direction associated with a prescribed spectral target using only matrix-vector products on local blocks. Different target directions can be recovered independently and in parallel. We establish non-asymptotic convergence guarantees and provide a recursive grid-search procedure for spectral target selection. Building on FLICKER, we develop BiFLICKER for federated community detection in bipartite networks, together with a sequential projected variant for nearby or repeated singular directions. Under a sparse bipartite stochastic block model, we show that properly rescaled leading singular values from a reference server provide valid global targets, and that Sequential BiFLICKER achieves a vanishing miscluster
    
[^310]: 用自然语言解释国际象棋与围棋

    Explaining Chess and Go in Natural Language

    [https://arxiv.org/abs/2610.23166](https://arxiv.org/abs/2610.23166)

    本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。

    

    现代国际象棋引擎的棋力远超人类，其选择着法背后的思路往往令人类难以理解。我们提出了一种基于国际象棋引擎评估的自动化指标，用于评估棋局解释的质量。利用该指标对最先进的语言模型进行评估后发现，它们的表现出奇地差，往往依赖于对棋盘的表面观察，而非稳健、可迁移的国际象棋概念。随后，我们提出了BinProp，这是一种无需训练的方法，能够使同样的模型发现可复用的国际象棋概念，并用它们以自然语言解释着法。BinProp提升了对所有受测模型的可解释性，与直接提示相比，平均得分几乎提高了两倍。除了国际象棋之外，同一方法无需额外调整即可在围棋、西洋跳棋和将棋中发现有用的概念。

    arXiv:2610.23166v1 Announce Type: cross  Abstract: Modern chess engines play far beyond human ability, often choosing moves whose underlying ideas are difficult for humans to understand. We propose an automatic metric for evaluating chess explanations, grounded in chess-engine evaluations. Evaluating state-of-the-art language models using this metric, we find that they perform surprisingly poorly, often relying on surface-level board observations rather than robust, transferable chess concepts. We then introduce BinProp, a training-free method that enables these same models to discover reusable chess concepts and use them to explain moves in natural language. BinProp improves explainability for every model tested, nearly tripling the average score compared with direct prompting. Beyond chess, the same method discovers useful concepts in Go, draughts, and shogi without additional tuning.
    
[^311]: L2GSSL：面向体积医疗数据的标签高效局部到全局自监督学习

    L2GSSL: Label-Efficient Local-to-Global Self-Supervised Learning for Volumetric Healthcare Data

    [https://arxiv.org/abs/2610.21356](https://arxiv.org/abs/2610.21356)

    提出L2GSSL局部到全局自监督学习框架，通过将预训练分解为切片内和切片间表示空间并利用停止梯度隔离目标，在仅使用10%标注的肺结节恶性分类任务上达到0.9229 AUROC，大幅超越体积级对比方法，接近全监督基线性能。

    

    医疗机构通常会积累大量高维计算机断层扫描（CT）体积数据，但其中只有一小部分能够由专家进行标注。我们提出了L2GSSL，这是一个局部到全局的自监督学习框架，通过将预训练分解为切片内和切片间两个表示空间来利用未标注的体积数据。切片编码器学习视图不变的局部特征，全局编码器聚合所得的标记以建模体积上下文。停止梯度机制将局部空间与全局目标隔离开来，而逐切片执行和三阶段训练调度则支持在受限GPU内存条件下的训练。在LIDC-IDRI肺结节恶性分类任务上，L2GSSL仅使用10%的标签即达到0.9229的AUROC，使用25%的标签达到0.9377，分别超过体积级VICReg方法0.0229和0.0294。在使用全部标签的情况下，最强的监督基线也仅比L2GSSL高出0.0044的AUROC，结果表现一致。

    arXiv:2610.21356v1 Announce Type: cross  Abstract: Healthcare institutions routinely accumulate high-dimensional computed tomography (CT) volumes, but only a fraction can be annotated by experts. We present L2GSSL, a local-to-global self-supervised learning framework that leverages unlabeled volumes by decomposing pretraining into intra-slice and inter-slice representation spaces. A slice encoder learns view-invariant local features, and a global encoder aggregates the resulting tokens to model volumetric context. The stop-gradient isolates the local space from the global objective, while slice-wise execution and a three-phase schedule support training under constrained GPU memory. On LIDC-IDRI lung nodule malignancy classification, L2GSSL reaches $0.9229$ AUROC with only $10\%$ of the labels and $0.9377$ with $25\%$, exceeding the volume-level VICReg by $0.0229$ and $0.0294$, respectively. With all labels, the strongest supervised baseline exceeds L2GSSL by only $0.0044$ AUROC, consis
    
[^312]: 稀疏自编码器表示的行为组合性

    Behavioral Compositionality of Sparse Autoencoder Representations

    [https://arxiv.org/abs/2610.21261](https://arxiv.org/abs/2610.21261)

    论文提出SAE可加性残差（SAR）来度量稀疏自编码器特征在行为层面的组合性，发现学习到的SAE坐标比随机旋转具有更强的一阶可加性，但仍存在不可忽略的高阶交互项。

    

    稀疏自编码器（SAE）使用稀疏的、线性解码的特征来表示模型激活，但这些特征的下游效应不一定是按加法方式组合的。我们引入了SAE可加性残差（SAR），用于衡量在选定交互阶数以上未能被解释的行为变异。在独立特征子集采样的条件下，总体SAR恰好等于该阶数以上交互的归一化能量，并为最优低阶近似的联盟排序误差提供了上界。为了区分与SAE表示本身相关的组合性与由下游计算所诱导的组合性，我们将学习到的SAE坐标与同一所选子空间的保和随机旋转进行比较，发现在所有18个模型和行为设置中，学习到的坐标都具有更低的一阶残差。然而，这些坐标并非完全可加：在每一设置中，成对交互项均能改善留出集上的联盟排序。

    arXiv:2610.21261v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) represent model activations with sparse, linearly decoded features, but the downstream effects of those features need not compose additively. We introduce SAE Additivity Residuals (SAR), which measure the behavioral variation left unexplained above a chosen interaction order. Under independent feature subset sampling, population SAR is exactly the normalized energy of interactions above that order and bounds coalition ordering error for the optimal low order approximation. To distinguish compositionality associated with the SAE representation from that induced by downstream computation, we compare learned SAE coordinates with sum preserving random rotations of the same selected subspace and find lower first order residuals for the learned coordinates across all 18 model and behavior settings. The coordinates are nevertheless not fully additive: pairwise terms improve held out coalition ordering in every setti
    
[^313]: 稀疏自编码器中的表征再分配

    Representation Redistribution in Sparse Autoencoders

    [https://arxiv.org/abs/2610.21240](https://arxiv.org/abs/2610.21240)

    该论文提出再分配敏感度分数（RSS）来识别在提示变换下以一致方向重复出现的大幅特征响应，并在涵盖八个语言模型的27个预训练SAE上验证了表征再分配的可复现性，同时将重复的表征变化与下游读出贡献及决策失败精确解耦。

    

    稀疏自编码器将语言模型激活分解为稀疏特征，但旨在保持底层请求不变的提示变换可能会使激活活动在这些特征之间重新分布。在某一个变换后提示上观察到的特征变化未必会再次出现，即使是重复出现的变化，也未必会影响下游计算。我们引入了再分配敏感度分数，用以识别以一致方向重复出现的大幅特征响应，并在未参与选择过程的变体上评估由此得到的支持集。在涵盖八个语言模型的27个预训练SAE上，所测试的提示变换在这些被排除的变体上产生了可复现的再分布特征。随后，我们将重复出现的表征变化与带符号的读出贡献以及决策失败区分开来。对于固定的线性读出，一个精确分解展示了响应幅度、读出系数、方向对齐以及……

    arXiv:2610.21240v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, but prompt transformations designed to preserve the underlying request can redistribute activity across them. A feature change observed on one transformed prompt need not recur, and even a recurring change need not affect a downstream computation. We introduce the redistribution sensitivity score (RSS), which identifies large feature responses that recur in a consistent direction, and evaluate the resulting supports on variants excluded from selection. Across 27 pretrained SAEs spanning eight language models, the tested prompt transformations produce a reproducible redistribution profile on these excluded variants. We then separate recurring representation change from signed readout contribution and decision failure. For a fixed linear readout, an exact decomposition shows how response magnitude, readout coefficients, directional alignment, and the cl
    
[^314]: 并非所有改变都是损害：面向大语言模型的更新匹配遗忘

    Not All Change Is Damage: Update-Matched Unlearning for Large Language Models

    [https://arxiv.org/abs/2610.21230](https://arxiv.org/abs/2610.21230)

    该论文提出“参照时序假设”，证明在大语言模型遗忘任务中，采用更新匹配的参照比更新前的参照能更好地保留模型学习新任务的可塑性，从而有效区分对已删知识的选择性抵抗与整体可塑性丧失。

    

    旨在从大语言模型中移除选定知识，以降低隐私和安全风险，同时保留有用的能力。对于发布后仍会持续微调的模型而言，这一目标并不充分：它们还必须在保持可塑性的同时抵抗对已删除知识的重新学习。现有的防御方法通过针对模拟的重新学习进行训练，同时试图保留有用的能力，但仅凭较低的重新学习程度无法区分选择性的抵抗能力与可塑性的整体丧失。要在模型更新后评估保留情况，需要确定更新后的模型应与什么保持接近。若采用更新前的参照，则会仅仅因为更新带来的有益改变偏离了早期输出而对这些改变进行惩罚。因此，我们提出了参照时序假设：在相当的目标抑制水平下，使用更新匹配的参照比使用更新前的参照能更好地保留可塑性。为了测试……（摘要截断）

    arXiv:2610.21230v1 Announce Type: new  Abstract: Unlearning aims to remove selected knowledge from large language models to mitigate privacy and safety risks while preserving useful capabilities. For models that continue to be fine-tuned after release, this goal is insufficient: they must also resist relearning the deleted knowledge while preserving plasticity, the ability to learn new tasks. Existing defenses train against simulated relearning while trying to preserve useful capabilities, but low relearning alone cannot distinguish selective resistance from a general loss of plasticity. Evaluating preservation after an update requires deciding what the updated model should remain close to. A pre-update reference penalizes useful changes induced by the update simply because they depart from earlier outputs. We therefore formulate the reference-timing hypothesis that, at comparable target suppression, an update-matched reference better preserves plasticity than a pre-update one. To test
    
[^315]: 连续马尔可夫链伪谱隙的极小极大估计及其在经验集中界中的应用

    Minimax estimation of the pseudo-spectral gap of continuous Markov chains, with applications to empirical concentration bounds

    [https://arxiv.org/abs/2610.20007](https://arxiv.org/abs/2610.20007)

    本文提出了紧致连续状态空间马尔可夫链伪谱隙的首个极小极大最优估计器，结合 Bernstein 不等式推导出经验集中不等式，并据此在无需伪谱隙先验知识的情况下，为马尔可夫奖励老虎机问题给出了具有接近独立同分布情形遗憾界的 UCB 算法。

    

    我们研究了基于单条轨迹观测来对马尔可夫链伪谱隙 $\gamma_{ps}$ 进行极小极大估计的问题。通过扩展 Wolfer 和 Kontorovich (2024) 关于有限状态空间链的早期工作，我们针对紧致连续状态空间的情形提出了一个估计器。在转移算子具有 Lipschitz 密度的链类上，我们的估计器在相差对数因子的意义下似乎是极小极大最优的。将该估计器与 Paulin 版本的 Bernstein 不等式相结合，可以为马尔可夫链推导出经验集中不等式。这开启了许多应用，包括置信区间、主动学习和泛化界。特别地，对于具有马尔可夫奖励的静止老虎机问题，我们提出了一个 UCB 算法版本，其遗憾界与独立同分布情形类似，且无需对链的伪谱隙有任何先验知识。

    arXiv:2610.20007v1 Announce Type: cross  Abstract: We study the minimax estimation of the pseudo-spectral gap $\gamma_{ps}$ of a Markov chain from the observation of a single trajectory. Extending earlier works on finite-state space chains by Wolfer and Kontorovich (2024), we provide an estimator in the case of a compact, continuous state-space. Our estimator appears to be minimax-optimal on the class of chains with a transition operator with Lipschitz density, up to logarithmic terms. Our estimator, combined with Paulin's version of Bernstein inequality, allows to derive empirical concentration inequalities for Markov chains. This unlocks many applications, from confidence intervals, to active learning and generalization bounds. In particular, for rested bandits with Markov rewards, we provide a version of UCB with a regret bound similar to the one in the iid case, without assuming any knowledge on the pseudo-spectral gap of the chains.
    
[^316]: 面向表格基础模型的循环上下文学习

    Recurrent In-Context Learning for Tabular Foundation Models

    [https://arxiv.org/abs/2610.14998](https://arxiv.org/abs/2610.14998)

    提出Tab-Loop方法，用4个参数共享的块循环应用来替代表格基础模型中参数不共享的多层ICL块，在匹配计算量的前提下将参数量减少62%且性能不降。

    

    表格基础模型通过上下文学习（ICL）适应新数据集，并且与大多数Transformer一样，它们将计算深度与独立参数化的层数绑定在一起。我们探究这样一个模型的ICL阶段是否需要为每一层配备一组独立的参数，还是仅仅需要深度。我们提出了Tab-Loop，它用4个参数共享（tied）的块循环应用k次来替换TabICLv2的12个参数不共享（untied）的ICL块，其他一切保持不变：编码器、解码器、合成先验和训练方案均与在相同计算量下训练的非共享参数基线保持一致。在匹配的预训练计算量（18.2 EFLOP）和匹配的推理计算量（12次块应用）下，Tab-Loop以减少62%的参数量（1040万 vs 2760万）达到了与基线相当的性能：留出集对数损失为0.4583 vs 0.4588，在200个TALENT数据集上的平均准确率为84.10% vs 83.92%，在38个TabArena数据集上的Elo得分为1409 vs 1396。一个具有Tab-Loop参数量的4块非共享对照模型……（摘要原文在此截断）

    arXiv:2610.14998v1 Announce Type: cross  Abstract: Tabular foundation models adapt to new datasets through in-context learning (ICL), and, like most transformers, they tie computational depth to the number of independently parameterized layers. We ask whether the ICL stage of such a model needs a distinct set of parameters for every layer, or only depth. We introduce Tab-Loop, which replaces the 12 untied ICL blocks of TabICLv2 with 4 tied blocks applied $k$ times and changes nothing else: the encoders, decoder, synthetic prior and training recipe are shared with an untied baseline trained for the same compute. At matched pretraining compute (18.2 EFLOP) and matched inference compute (12 block applications), Tab-Loop matches the baseline with 62\% fewer parameters (10.4M vs.\ 27.6M): held-out log loss is 0.4583 vs.\ 0.4588, mean accuracy on 200 TALENT datasets is 84.10\% vs.\ 83.92\%, and Elo on 38 TabArena datasets is 1409 vs.\ 1396. A 4-block untied control with Tab-Loop's parameter 
    
[^317]: 逆强化学习中奖励函数更新的几何视角

    A Geometric Perspective on Reward Function Updates in Inverse Reinforcement Learning

    [https://arxiv.org/abs/2610.14041](https://arxiv.org/abs/2610.14041)

    本文从占用度量空间的几何视角重新审视逆强化学习中的奖励更新，揭示朝向最优占用度量的e-测地线几何结构与奖励参数空间中对偶梯度的直接联系，并据此提出用占用轨迹上的平均Fisher信息对对偶梯度进行预条件处理的方法。

    

    逆强化学习（IRL）经典上通过拉格朗日优化来求解，即利用梯度下降联合优化一个原始-对偶问题，从而获得最优奖励函数参数及相应的最优策略。尽管这种算法视角的IRL影响深远，但基于梯度的奖励更新只描绘了问题的部分图景，很大程度上忽略了占用度量空间中该问题的几何结构。在本文中，我们提出了IRL中奖励更新的占用度量空间视角。我们利用这样一个洞察：最优解及其在占用度量空间中的几何结构均是已知的，并证明沿e-测地线（占用度量空间中的直线）朝向最优占用度量的几何结构，会直接影响奖励参数空间中IRL对偶问题的梯度。这一视角自然地启发我们用占用度量轨迹上的平均Fisher信息对对偶梯度进行预条件处理。我们证明了……

    arXiv:2610.14041v1 Announce Type: cross  Abstract: Inverse Reinforcement Learning (IRL) is classically solved using Lagrangian optimisation, by jointly optimising a primal-dual problem with gradient descent to obtain the optimal reward function parameters and corresponding optimal policy. While this algorithmic view of IRL has been highly influential, gradient-based reward updates only paint a partial picture, largely ignoring the geometry of the problem in occupancy space. In this paper, we present an occupancy-space view of reward updates in IRL. We leverage the insight that both the optimal solution and its geometry in occupancy space are known, and show that the geometry along the e-geodesic (straight line in occupancy space) toward the optimal occupancy directly affects the gradient of the IRL dual problem in reward parameter space. This perspective naturally suggests preconditioning the dual gradient with the average Fisher information along the occupancy trajectory. We show that
    
[^318]: 基于图神经网络的摊销式反事实预测

    Amortized Counterfactual Prediction with Graph Neural Networks

    [https://arxiv.org/abs/2610.13724](https://arxiv.org/abs/2610.13724)

    提出反事实图神经过程（CFGNP），利用图神经网络端到端地直接从观测数据回答反事实查询，无需先估计因果图，从而避免因果图估计误差向下游反事实预测的传播。

    

    反事实推断研究观察到的结果在不同情形下可能有何不同，例如患者如果接受不同的治疗是否会康复。只要决策依赖于未观察到的替代方案的结果，此类问题就会出现。回答反事实查询通常需要了解潜在的因果结构，而这在实践中往往是不可得的。因此，一种常见的做法是先从观测数据中估计因果图，然后再进行反事实推断。然而，估计出的因果图中的误差可能会传播到下游的反事实预测中。为了应对这一挑战，我们提出了反事实图神经过程（CFGNP），这是一种端到端的方法，可以直接从观测数据回答反事实查询，而无需先估计因果图。CFGNP 利用图神经网络（GNN）来发掘对偶……（原文在此处截断）

    arXiv:2610.13724v1 Announce Type: cross  Abstract: Counterfactual inference asks how an observed outcome might have differed under different circumstances, such as whether a patient would have recovered had they received a different treatment. Such questions arise whenever decisions depend on the outcomes of alternatives that were not observed. Answering counterfactual queries generally requires knowledge of the underlying causal structure, which is often unavailable in practice. A common approach is therefore to first estimate the causal graph from observational data and then perform counterfactual inference. However, errors in the estimated causal graph can propagate to downstream counterfactual predictions. To address this challenge, we introduce the Counterfactual Graph Neural Process (CFGNP), an end-to-end approach that answers counterfactual queries directly from observational data without first estimating a causal graph. CFGNP uses Graph Neural Networks (GNNs) to exploit the dua
    
[^319]: 极端二分类：利用极值理论实现对假负类的极端约束

    Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative

    [https://arxiv.org/abs/2610.09984](https://arxiv.org/abs/2610.09984)

    本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。

    

    尽管二分类是机器学习中研究最为广泛的问题之一，但以学习一个假负类率几乎为零的分类器为目标的这一情形在很大程度上仍未被探索。在本文中，我们提出了“极端二分类”问题，其目标是学习一个假负类率 $\alpha$ 受 $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$ 约束的分类器，其中 $N_1$ 表示训练集中正例样本的数量。为了解决这一问题，我们提出了一种基于极值理论推导的理论保证的阈值自适应方法，并结合一种基于对样本最大值进行置换检验的特征选择程序。在四个不同规模的真实数据集上的实验结果表明，我们的方法优于当前最先进的方法。此外，我们还通过……展示了该方法的可解释性。

    arXiv:2610.09984v1 Announce Type: cross  Abstract: While binary classification is one of the most extensively studied problems in machine learning,   the regime in which the goal is to learn a classifier with an almost zero false negative rate remains largely unexplored.   In this paper, we introduce the Extreme Binary Classification problem, where the objective is to learn a classifier whose false negative rate $\alpha$ is constrained by $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$, with $N_1$ denoting the number of positive examples in the training set.   To address this problem, we propose a threshold adaptation method theoretically grounded in guarantees derived from Extreme Value Theory, together with a feature selection procedure based on a permutation test applied to sample maxima.   Experimental results on four real-world datasets of varying sizes demonstrate that our approach compares favorably with state-of-the-art methods.   In addition, we illustrate its interpretability throug
    
[^320]: SearchWorld：基于世界模型的价值引导空间想象的无人机目标搜索

    SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models

    [https://arxiv.org/abs/2610.09335](https://arxiv.org/abs/2610.09335)

    SearchWorld提出了一种将显式空间记忆与价值引导想象相结合的循环状态空间世界模型，通过前瞻性想象推演解决部分可观测性难题，实现无人机在城市环境中的高效目标搜索。

    

    自主无人机（UAV）目标搜索涉及部分可观测环境下感知、决策与行动的闭环过程。城市环境带来了诸多挑战：广阔的搜索区域和狭窄的第一人称视野限制了覆盖范围，密集的三维几何结构制约了安全移动，而开放世界的指令要求在众多干扰物中识别特定目标。许多现有方法通过显式地图或记忆表征来缓解部分可观测性问题，但它们在很大程度上仍是反应式的，即仅基于过去的观测进行推理，而未能显式地预测未来状态。世界模型能够通过想象推演实现前瞻性推理。然而，基于图像生成的世界模型可能带来较高的推理延迟，而对于潜在（latent）世界模型而言，空间接地的规划仍然具有挑战性。我们提出了SearchWorld，一种将显式空间记忆与价值引导想象相连接的循环状态空间世界模型。该模

    arXiv:2610.09335v1 Announce Type: cross  Abstract: Autonomous unmanned aerial vehicle (UAV) object search involves a closed loop of perception, decision-making, and action under partial observability. Urban environments pose several challenges: large search areas and narrow egocentric views limit coverage, dense 3D geometry constrains safe motion, and open-world instructions require identifying a specific target among distractors. Many existing methods mitigate partial observability through explicit maps or memory representations, yet remain largely reactive, reasoning over past observations without explicitly predicting future states. World models enable prospective reasoning through imagined rollouts. However, image-generating world models can incur high inference latency, while spatially grounded planning remains challenging for latent world models. We propose SearchWorld, a recurrent state-space world model that connects explicit spatial memory with value-guided imagination. The mo
    
[^321]: 冻结因子还是频谱带？在LoRA中，低秩时因子更为重要

    Frozen Factor or Spectral Band? In LoRA, the Factor Matters More at Low Rank

    [https://arxiv.org/abs/2610.06621](https://arxiv.org/abs/2610.06621)

    本文通过2x2交叉实验设计发现，在低秩（1-8）少步数设置下，LoRA变体冻结哪个因子（输入因子A显著优于输出因子B）比其选择的奇异方向带（主导或次主导）对性能影响更大。

    

    在两个主要的任务-模型组合上，在秩1到8、优化器步数较少的设置下，LoRA变体所冻结的因子比其奇异方向带更为重要。MiCA和PiCa通常按频谱带（次主导与主导奇异方向）来比较，但在我们的记号下，MiCA冻结输出因子B，而PiCa冻结输入因子A。我们在1B到8B参数规模的模型上，通过2x2交叉实验设计同时考察了因子和带两个维度。在两个带上，冻结A均优于冻结B：在格式任务（我们的主要证据）的秩2上提升12到20个百分点，在OpenBookQA的秩8上仍提升约7个百分点。事后补充的针对258项主要检验的Holm校正显示，这一优势在16项检验中的15项中依然成立，且因子相对于带的优势在秩1、2和4上保持领先（格式任务的秩1本身也是事后检验）。在秩8上，该优势通过了自身检验但未通过校正检验。该优势在秩2上用两个7B模型得到重复验证，提升11到13个百分点。在校正之外，该优势还在GSM8K上用1B模型得到了重复（原文在此处截断）。

    arXiv:2610.06621v2 Announce Type: replace-cross  Abstract: On two main task-model pairs, at ranks 1 to 8 with few optimiser steps, the factor a LoRA variant freezes matters more than its band of singular directions. MiCA and PiCa are compared by band, minor against dominant, yet MiCA freezes the output factor B and PiCa, in our notation, the input factor A. We cross factor and band in a 2 x 2 design on models from 1B to 8B parameters. Freezing A beats freezing B on both bands, by +12 to +20 points at rank 2 on a format task, our primary evidence, and still by about +7 points at rank 8 on OpenBookQA. A Holm correction over 258 primary tests, added post hoc, keeps this advantage in 15 of 16 tests and the lead of factor over band at ranks 1, 2 and 4, rank 1 of the format task being itself post hoc. At rank 8 that lead passes its own test but not the correction. The advantage repeats at rank 2 with two 7B models, by +11 to +13 points. Outside the correction, it repeats on GSM8K with a 1B m
    
[^322]: SLIM：单纯形-格点插值合并

    SLIM: Simplex-Lattice Interpolation Merging

    [https://arxiv.org/abs/2610.01037](https://arxiv.org/abs/2610.01037)

    SLIM 利用经典混料设计，仅通过对单个专家和等权重配对的最少评估，在系数单纯形上构建总体性能的二次代理模型，无需额外基准评估即可优化大语言模型合并系数，在有限评估预算下实现准确预测和有竞争力的合并性能。

    

    优化大型语言模型的合并系数可能需要许多代价高昂的基准测试评估。我们提出了单纯形-格点插值合并（Simplex-Lattice Interpolation Merging, SLIM），该方法利用经典的混料设计，在系数单纯形上构建总体性能的二次代理模型。通过对单个专家模型和等权重配对的评估，以识别该域上一般二次模型所需的最少测量次数来确定该代理模型。随后，SLIM 在无需进一步目标指标评估的情况下对代理模型进行优化。在两种模型架构上的实验表明，该方法能够准确预测未见过的多专家混合方案，并在有限的评估预算下取得具有竞争力的合并性能。在预算匹配的比较中，结构化的评估点设计相比随机设计（包括采用正则化拟合的设计）展现出更高的预测保真度。

    arXiv:2610.01037v1 Announce Type: new  Abstract: Optimizing merging coefficients for large language models can require many costly benchmark evaluations. We propose \textbf{Simplex-Lattice Interpolation Merging (SLIM)}, which constructs a quadratic surrogate of aggregate performance on the coefficient simplex using a classical mixture design. Evaluations of individual experts and equal-weight pairs determine the surrogate with the minimum number of measurements needed to identify a general quadratic on this domain. SLIM then optimizes the surrogate without further target-metric evaluations. Experiments on two model architectures demonstrate accurate prediction of unseen multi-expert mixtures and competitive merge performance under limited evaluation budgets. Matched-budget comparisons show that structured evaluation points improve prediction fidelity over random designs, including those using regularized fitting.
    
[^323]: CompMat-Bench：面向计算材料科学的AI智能体基准测试

    CompMat-Bench: Benchmarking AI Agents for Computational Materials Science

    [https://arxiv.org/abs/2610.00636](https://arxiv.org/abs/2610.00636)

    提出了CompMat-Bench基准，通过预先复现昂贵模拟的输入与结果作为固定规则的评分依据，实现对计算材料科学AI智能体的低成本高效评估，涵盖94个源自已发表研究的任务以及四种评估条件。

    

    在科学研究任务上评估AI智能体受到底层实验或计算所需时间和资源的限制。在计算材料研究中，跨智能体和多次试验重复相同的昂贵模拟会使评估变得不切实际。我们提出了CompMat-Bench，这是一个包含94个任务的基准，这些任务源自近期发表的计算材料研究，每个任务要求智能体完成实现该研究科学目标的一个步骤。我们预先复现了研究步骤，并评估智能体在为昂贵模拟准备输入和分析输出方面的能力，从而在评估过程中避免执行昂贵的模拟。复现的输入和结果作为基准真值，通过固定规则对智能体进行评分，无需LLM裁判。该基准支持四种评估条件：单一任务和由相关任务组成的工作流，每种条件均提供完整或精简的方法学指导。

    arXiv:2610.00636v1 Announce Type: new  Abstract: Evaluating AI agents on scientific research tasks is constrained by the time and resources required for the underlying experiments or calculations. In computational materials research, repeating the same expensive simulations across agents and trials can make evaluation impractical. We introduce CompMat-Bench, a benchmark of 94 tasks derived from recently published computational materials studies, each asking agents to complete a step toward achieving the study's scientific goal. We reproduce the research steps in advance and assess agents on preparing inputs and analyzing outputs for expensive simulations, so expensive simulations can be avoided during evaluation. The reproduced inputs and results serve as ground truth for grading agents with fixed rules, without an LLM judge. The benchmark supports four evaluation conditions: single tasks and workflows composed of related tasks, each with full or reduced methodological guidance. With f
    
[^324]: 策略迭代对于确定性马尔可夫决策过程并非强多项式：算法无政府状态的代价

    Policy Iteration Is Not Strongly Polynomial for Deterministic Markov Decision Processes: The Price of Algorithmic Anarchy

    [https://arxiv.org/abs/2609.40147](https://arxiv.org/abs/2609.40147)

    本文证明Howard策略迭代在确定性折扣马尔可夫决策过程上具有指数级迭代下界因而并非强多项式，而带Dantzig规则的单纯形法在此类问题上却是强多项式的，这一差距揭示了“算法无政府状态”的代价。

    

    我们针对每个状态至多有两个动作的确定性折扣马尔可夫决策过程，为Howard策略迭代建立了以状态数为变量的指数级迭代次数下界。这一结果排除了当折扣因子作为输入的一部分时Howard策略迭代的强多项式性，并与带Dantzig转动规则的单纯形法形成指数级分离——后者被证明在这类问题上具有强多项式性。即使将每个奖励限制为对数级比特长度，我们仍然得到了一个拉伸指数级的迭代次数下界。Howard那种去中心化、各状态同时进行的自私改进，与Dantzig在所有状态间协调选择增益最大的单一动作之间的差距，揭示了“算法无政府状态”的代价。

    arXiv:2609.40147v1 Announce Type: new  Abstract: We establish an exponential iteration lower bound in the number of states for Howard's policy iteration on deterministic discounted Markov decision processes, with at most two actions per state. This rules out strong polynomiality of Howard's policy iteration when the discount factor is part of the input and yields an exponential separation from the simplex method with Dantzig's pivoting rule, which is proved to be strongly polynomial on this class. Even when each reward is restricted to logarithmic bit length, we obtain a stretched-exponential iteration lower bound. The gap between Howard's decentralized and simultaneous selfish improvements and Dantzig's coordinated selection of a single action with the largest gain across all states reveals a ``price'' of algorithmic anarchy.
    
[^325]: GazeFlow：从人类注视行为到生成式第一人称注视预测

    GazeFlow: From Human Gaze Behavior to Generative Egocentric Gaze Prediction

    [https://arxiv.org/abs/2609.38519](https://arxiv.org/abs/2609.38519)

    提出GazeFlow框架，利用条件流匹配将第一人称注视建模为受自上而下任务信息与自下而上视觉显著性共同约束的时间注视位置联合分布，从而生成合理的注视轨迹并达到最先进的预测性能。

    

    第一人称（自我中心）注视预测能够支持许多下游应用，但仍然具有挑战性，因为人类注视本质上是随机的。这种随机性受到结构化时间动态（在注视驻留与扫视之间交替）、自上而下的任务影响以及自下而上的视觉显著性的约束。基于这一观察，我们提出了GazeFlow，一个将注视直接建模为以自上而下和自下而上信息为条件的时间注视位置联合分布的框架。具体而言，GazeFlow采用条件流匹配（CFM）：一个学习到的速度场迭代地将高斯噪声样本输运为从该联合分布中抽取的合理注视轨迹。该速度场以视频编码器提取的自下而上视觉特征，以及通过全局查询这些特征获得的自上而下任务信息为条件。在标准数据集上，GazeFlow在每个……（原文在此截断）方面取得了最先进的性能。

    arXiv:2609.38519v1 Announce Type: cross  Abstract: Egocentric gaze prediction enables many downstream applications but remains challenging, as human gaze is inherently stochastic. This stochasticity is constrained by structured temporal dynamics alternating between fixations and saccades, top-down influences from tasks, and bottom-up visual saliency. Based on this observation, we introduce GazeFlow, a framework that directly models gaze as a joint distribution of temporal gaze positions conditioned upon both top-down and bottom-up information. In particular, GazeFlow uses conditional flow matching (CFM): a learned velocity field iteratively transports a Gaussian noise sample into a plausible gaze trajectory drawn from this joint distribution. The velocity field is conditioned on bottom-up visual features extracted by a video encoder and on top-down task information obtained by globally querying these features. On standard datasets, GazeFlow achieves state-of-the-art performance on per-
    
[^326]: VirusCascade：劫持基于大语言模型的推荐智能体中的协同反思

    VirusCascade: Hijacking Collaborative Reflection in LLM-Powered Recommender Agents

    [https://arxiv.org/abs/2609.38270](https://arxiv.org/abs/2609.38270)

    本文揭示了基于大语言模型的智能体推荐系统中的新型安全漏洞，提出VirusCascade攻击方法，通过“反思洗白”将注入单个智能体的对抗性证据合理化，并借助协同反思机制在多智能体间传播放大，实现对整个推荐系统的劫持。

    

    超越传统的静态评分模型，基于大语言模型的智能体推荐系统（LLM-ARS）将用户和物品实例化为自主智能体，其语义状态通过一种被称为“协同反思”的循环过程进行动态优化。虽然这一机制提升了推荐质量，但同时也引入了一种系统性漏洞：注入到单个智能体中的对抗性证据可以被合理化为合法的偏好叙事，写回记忆，并通过交互上下文传播给其他智能体。我们将这种局部合理化过程称为“反思洗白”，将其通过协同反思在系统范围内的升级称为“协同反思劫持”。现有的推荐系统攻击，无论是基于交互级数据投毒还是文本级对抗扰动，都假设静态流水线，因而无法利用这种循环式的多智能体放大机制。

    arXiv:2609.38270v1 Announce Type: cross  Abstract: Advancing beyond traditional static scoring models, LLM-powered agentic recommender systems (LLM-ARS) instantiate users and items as autonomous agents, whose semantic states are dynamically refined through a recurrent process known as collaborative reflection. While this mechanism improves recommendation quality, it simultaneously introduces a systemic vulnerability: adversarial evidence injected into a single agent can be rationalised into a legitimate preference narrative, written back into memory, and propagated to other agents through interaction contexts. We term the local rationalisation process reflection laundering, and its system-wide escalation through collaborative reflection collaborative-reflection hijacking. Existing attacks on recommender systems, whether based on interaction-level data poisoning or text-level adversarial perturbations, assume static pipelines and thus cannot exploit this recurrent, multi-agent amplifica
    
[^327]: 基于条件流匹配与有限策略强化学习的RNA设计

    RNA Design via Conditioned Flow Matching and Finite-Policy Reinforcement Learning

    [https://arxiv.org/abs/2609.36885](https://arxiv.org/abs/2609.36885)

    该论文提出结合结构条件化狄利克雷流匹配与保持碱基配对的有限策略强化学习的两阶段RNA逆折叠框架，通过显式建模自然RNA进化中的协调序列变异并利用热力学反馈优化，在多个基准测试中取得领先性能（Rfam-27上Pass@1达85.19%）。

    

    RNA设计旨在识别能够折叠成特定二级结构的序列。现有方法将该任务表述为针对特定目标的搜索或条件生成。然而，自然RNA进化是通过序列变异和选择进行的，并伴随补偿性替换，而这些方法并未显式建模这一过程。为解决这一局限，我们提出了一个两阶段框架，包括RNA逆折叠流和RNA-IFlow-RL。RNA-IFlow使用结构条件化的狄利克雷流匹配来建模序列上的协调变异，而RNA-IFlow-RL将学习到的流映射为保持碱基配对的有限策略，并通过热力学反馈对其进行优化。我们的框架在多个基准测试中取得了领先性能，在Rfam-27上达到85.19%的Pass@1。进一步的分析揭示了热力学增益、策略动态变化以及在不同设置下的鲁棒性。我们的工作将协调变异与（原文摘要在此处截断）

    arXiv:2609.36885v1 Announce Type: cross  Abstract: RNA design aims to identify sequences that fold into specified secondary structures. Existing methods formulate the task as target-specific search or conditional generation. However, natural RNA evolution proceeds through sequence variation and selection, with compensatory substitutions, whereas these methods do not explicitly model this process. To address this limitation, we propose a two-stage framework comprising RNA Inverse-Folding Flow (RNA-IFlow) and RNA-IFlow-RL. RNA-IFlow uses structure-conditioned Dirichlet Flow Matching to model coordinated variation across the sequence, while RNA-IFlow-RL maps the learned flow to a pairing-preserving finite policy and refines it with thermodynamic feedback. Our framework achieves leading performance on multiple benchmarks, reaching 85.19% Pass@1 on Rfam-27. Further analyses reveal thermodynamic gains, policy dynamics, and robustness across settings. Our work couples coordinated variation wi
    
[^328]: 面向表格基础模型的共形预测与条件覆盖

    Conformal Prediction and Conditional Coverage for Tabular Foundation Models

    [https://arxiv.org/abs/2609.34887](https://arxiv.org/abs/2609.34887)

    该论文提出C-USIM方法，一种无需额外训练或推理、可处理多模态预测的轻量级最高预测密度分割共形预测技术，为表格基础模型提供有限样本边际覆盖有效性保证，并显著改善条件覆盖和分组覆盖精度。

    

    表格基础模型（TFMs）为回归任务提供预测分布，但即使点预测准确，其预测区间仍可能出现覆盖不足或覆盖过度的问题。我们提出了C-USIM（条件均匀化得分积分方法），这是最高预测密度分割共形预测的一种轻量级应用，能够适应多模态预测。在给定校准集和测试集输出的情况下，该方法无需额外的训练或模型推理。在我们的假设下，它提供了有限样本的边际有效性保证。我们利用分布估计误差和得分离散性对条件-边际覆盖差距进行了界定，并通过百分位秩-得分图考察了覆盖的异质性。在TabPFN和TabICL上的实验表明，该方法提高了边际覆盖的准确性，并降低了平均条件覆盖误差和分组覆盖误差。在固定数据预算下，将更多的观测数据分配给校准阶段可以减少（条件覆盖误差）……

    arXiv:2609.34887v2 Announce Type: replace-cross  Abstract: Tabular foundation models (TFMs) provide predictive distributions for regression, but their prediction regions can exhibit undercoverage or overcoverage even when point predictions are accurate. We introduce C-USIM (Conditionally-Uniformized Score Integration Method), a lightweight application of highest predictive density split conformal prediction that accommodates multimodal predictions. Given calibration and test outputs, it requires no additional training or model inference. It provides finite-sample marginal validity under our assumptions. We bound conditional-marginal coverage gaps using distribution-estimation error and score discreteness, and examine coverage heterogeneity through percentile rank-score plots. Experiments with TabPFN and TabICL show improved marginal coverage accuracy and lower average conditional and group coverage errors. Under a fixed data budget, allocating more observations to calibration can reduc
    
[^329]: 通过稳定客户端聚类实现并发分割学习

    Concurrent Split Learning Through Stable Client Clustering

    [https://arxiv.org/abs/2609.29395](https://arxiv.org/abs/2609.29395)

    本文提出全局聚类并行分割学习（GCPSL），通过将客户端稳定分配到固定集群并并发执行多个分割学习工作负载，在不增大单个批量的前提下提升客户端直接数据参与率，使CIFAR-10达到85%验证准确率的时间从串行执行的约19分钟缩短至约6分钟。

    

    使用固定全局批量的训练限制了在任何单一步骤中能够提供样本的分布式客户端数量。我们研究了在不增加单个工作负载所处理批量的情况下，利用额外服务器工作节点的方法。全局聚类并行分割学习（GCPSL）将客户端分配到固定的集群，为每个集群并发执行带全局采样的并行分割学习（GPSL）工作负载，并周期性地融合客户端与服务器模型片段。在包含256个逻辑客户端的仿真中，将客户端群体划分到更多工作负载中可以提高直接数据参与率，而较小的集群可能带来准确率损失。在四块H100 GPU上实现的标签感知GCPSL，在三次匹配运行中平均于6.13±0.15分钟内达到CIFAR-10上85%的验证准确率，而相同工作负载串行执行则需要19.09±0.45分钟。在四GPU的资源配置下，规模均衡和随机固定归属的集群划分达到了目……（原文在此处截断）

    arXiv:2609.29395v1 Announce Type: cross  Abstract: Training with a fixed global batch limits how many distributed clients can provide examples in any one step. We examine a way to use additional server workers without increasing the batch processed by an individual workload. Global Clustered Parallel Split Learning (GCPSL) assigns clients to fixed clusters, executes a Parallel Split Learning with Global Sampling (GPSL) workload for each cluster concurrently, and periodically fuses the client and server model segments. In simulations with 256 logical clients, dividing the population across more workloads improves direct data participation, while smaller clusters can incur an accuracy cost. A four-H100 implementation of label-aware GCPSL reaches 85% CIFAR-10 validation accuracy in $6.13 \pm 0.15$ minutes over three matched runs, versus $19.09 \pm 0.45$ minutes when the same workloads are serialized. Within the four-GPU allocation, size-balanced and random fixed affiliations reach the tar
    
[^330]: 直推学习更锐利的界及其应用

    Even Sharper Bounds for Transductive Learning and Its Applications

    [https://arxiv.org/abs/2609.28459](https://arxiv.org/abs/2609.28459)

    本文提出直推学习的新型局部化复杂度方法STLC，去除了以往直推学习中多余的对数置信度因子，并在可实现设定下达到了与标准归纳学习相同的 $\cO\{\dVC\log(me/\dVC)/m\}$ 速率。

    

    我们提出了更锐利的直推局部复杂度，这是一种针对无放回均匀采样下直推学习的局部化复杂度方法。该构造始于一个关于测试-训练经验过程上确界的Bernstein型集中不等式，其证明利用了交换随机游走的修正对数Sobolev不等式和双参数熵闭包。随后，通过替代局部化泛函的剥离论证，我们得到了与经典归纳局部Rademacher复杂度界具有相同不动点和置信度项的过剩风险界，且去除了早期直推结果中额外的对数置信度因子。对于VC维为 $\dVC$ 的二值函数类上的可实现学习，当训练集大小为 $m$、测试集大小为 $u$ 且 $u\ge m\ge\dVC$ 时，STLC 给出了 $\cO\{\dVC\log(me/\dVC)/m\}$ 的界。这一结果匹配了标准的归纳学习速率，并且当 $m\ge9$ 时，与直推…（摘要原文在此处截断）相比仅相差一个对数因子。

    arXiv:2609.28459v1 Announce Type: new  Abstract: We introduce Sharper Transductive Local Complexity (STLC), a localized complexity method for transductive learning under uniform sampling without replacement. The construction starts from a Bernstein-type concentration inequality for the supremum of the test--train empirical process. Its proof uses the modified log-Sobolev inequality for the swap walk and a two-parameter entropy closure. A peeling argument with a surrogate localization functional then gives excess-risk bounds with the same fixed-point and confidence terms as the classical inductive local Rademacher-complexity bounds, without the additional logarithmic confidence factor in earlier transductive results. For realizable learning over a binary class of VC dimension $\dVC$, with training size $m$, test size $u$, and $u\ge m\ge\dVC$, STLC yields $\cO\{\dVC\log(me/\dVC)/m\}$. This matches the standard inductive rate and, when $m\ge9$, is within a logarithmic factor of the transd
    
[^331]: 相对放电阶段（RDS）分类：一种实用的电池放电进度指标

    Relative Discharge Stage (RDS) Classification: A Practical Indicator of Battery Discharge Progress

    [https://arxiv.org/abs/2609.27986](https://arxiv.org/abs/2609.27986)

    本文提出相对放电阶段（RDS）这一电池管理指标，将剩余放电状态划分为五个可解释等级，并通过结合SOC估计与轻量级时序学习的物理信息分类框架，在无需未来电流信息的情况下实现对电池放电进度的实用化评估。

    

    在实际电池应用中，由于未来负载曲线未知且高度动态，准确预测剩余放电时间（RDT）极具挑战性。为解决连续RDT回归的不确定性问题，本文提出了相对放电阶段——一种电池管理指标，通过五个可解释的类别（正常、良好、中等、偏低、需要充电）来表征剩余放电状态。与反映当前电荷水平的荷电状态（SOC）不同，RDS刻画的是剩余放电过程，且在推理阶段无需未来电流信息。本文提出了一种物理信息驱动的RDS分类框架，将SOC估计与轻量级时序学习相结合。SOC估计组件包括二阶等效电路模型（ECM）状态与端电压预测、滞回特性与OCV温度校正、电芯核心温度估计以及自适应扩展卡尔曼滤波（AEKF）状态校正，并辅以……（原摘要在此处截断）

    arXiv:2609.27986v1 Announce Type: new  Abstract: Accurate remaining discharge time (RDT) prediction is challenging in real-world battery applications because future load profiles are unknown and highly dynamic. To address the uncertainty of continuous RDT regression, this paper introduces Relative Discharge Stage (RDS), a battery-management indicator that represents the remaining discharge condition using five interpretable classes: Normal, Good, Moderate, Low, and Recharge Required. Unlike state of charge (SOC), which reflects the current charge level, RDS characterizes the remaining discharge process without requiring future-current information during inference. A physics-informed RDS classification framework is proposed, combining SOC estimation with lightweight temporal learning. The SOC-estimation component includes second-order ECM state and terminal-voltage prediction, hysteresis and OCV temperature correction, core-temperature estimation, and AEKF state correction, supported by
    
[^332]: 注意力路由早期即稳定：循环语言模型的工作集推理

    Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models

    [https://arxiv.org/abs/2609.27373](https://arxiv.org/abs/2609.27373)

    该论文发现循环语言模型的注意力路由在早期循环步骤即趋于稳定，据此提出无需训练的WISE推理方法：早期用全局注意力发现稀疏工作集，后续步骤重用该支撑集，在保持推理质量的同时大幅减少重复的全局注意力计算。

    

    循环语言模型通过反复应用共享的网络模块来精炼潜在表示，但标准推理在每个循环步骤都会重新计算全局注意力。我们研究了注意力在循环深度上的动态变化，发现注意力的支撑集和分布比隐藏状态和注意力输出更早地稳定下来。这提示了一种两阶段结构：早期步骤发现相关上下文的稀疏工作集，而后续步骤则在基本相同的路由支撑上精炼表示。受此结构启发，我们提出了 WISE（基于支撑集利用的工作集推理），这是一种无需训练的方法，它在早期循环阶段使用不受限制的全局注意力，而在后续阶段重用直接发现的块结构化支撑，同时保持循环深度和支撑内注意力计算的动态性。受控干预实验表明，循环中的发现过程非常重要，而仅重用支撑

    arXiv:2609.27373v1 Announce Type: new  Abstract: Recurrent language models repeatedly apply shared network blocks to refine latent representations, but standard inference recomputes global attention at every recurrent step. We study attention dynamics across recurrent depth and find that attention support and distributions stabilize substantially earlier than hidden states and attention outputs. This suggests a two-stage structure: early steps discover a sparse working set of relevant context, while later steps refine representations over largely the same routing support. Motivated by this structure, we introduce WISE (Working-set Inference with Support Exploitation), a training-free method that uses unrestricted global attention during early recurrence and later reuses directly discovered block-structured support while keeping recurrent depth and within-support attention computation dynamic. Controlled interventions show that recurrent discovery is important and that support-only reus
    
[^333]: LSTM-UT与循环深度Transformer在元胞自动机上的研究

    LSTM-UT and Recurrent-Depth Transformers on Cellular Automata

    [https://arxiv.org/abs/2609.19521](https://arxiv.org/abs/2609.19521)

    该论文提出带有有界门控记忆的LSTM-UT模型，并通过Rule 30元胞自动机和延迟回忆实验证明，仅依赖当前隐藏状态的Block Universal Transformer在深度外推上优于依赖扩展注意力缓存的CoTFormer，而LSTM-UT进一步同时提升了深度外推与延迟回忆能力。

    

    循环深度Transformer通过反复应用共享计算来工作，但它们在跨步骤保留信息的方式上有所不同。我们比较了三种模型：Block Universal Transformer（BUT），它仅携带当前隐藏状态；CoTFormer，它还保留不断扩展的注意力缓存；以及一种新的带有有界门控记忆的LSTM Universal Transformer（LSTM-UT）。在Rule 30元胞自动机上，BUT比CoTFormer更可靠地外推到未见过的循环深度，尽管其准确率最终会下降。状态与缓存干预实验表明，CoTFormer的失败源于两者的交互作用：修正当前状态可以暂时恢复准确率，而保留的历史信息可能会破坏这一修正。在延迟回忆任务中，尽管BUT缺乏对过去状态的直接访问，它仍然优于CoTFormer；CoTFormer无法可靠地选取所请求的缓存表示。LSTM-UT在这些基线之上同时提升了深度外推和延迟回忆能力……

    arXiv:2609.19521v1 Announce Type: new  Abstract: Recurrent-depth Transformers apply shared computation repeatedly, but differ in how they retain information across steps. We compare a Block Universal Transformer (BUT), which carries only its current hidden state; CoTFormer, which also retains an expanding attention cache; and a new LSTM Universal Transformer (LSTM-UT) with bounded gated memory. On Rule 30 cellular automata, BUT extrapolates to unseen recurrent depths more reliably than CoTFormer, although its accuracy eventually degrades. State and cache interventions show that CoTFormer's failure depends on their interaction: correcting the current state can temporarily restore accuracy, while retained history can undermine that correction. In a delayed-recall task, BUT also outperforms CoTFormer despite lacking direct access to past states; CoTFormer does not reliably select the requested cached representation. LSTM-UT improves both depth extrapolation and delayed recall over these b
    
[^334]: Tabby：时间序列基础模型的开放预训练方案

    Tabby: An Open Pretraining Recipe for Time Series Foundation Models

    [https://arxiv.org/abs/2609.13956](https://arxiv.org/abs/2609.13956)

    Tabby是一个完全开放的1.45亿参数长上下文概率时间序列基础模型，通过融合真实与因果合成数据、渐进式收敛调度和深度分位数监督，以单模型支持预测、分类和异常检测多种任务。

    

    本报告发布了Tabby，一个长上下文概率时间序列基础模型，并公开了其完整的开放构建方案。Tabby采用仅编码器的分块Transformer架构，将主要贡献集中在数据和训练流程上。预训练语料库结合了扩展的真实世界数据集GIFT-Eval-Pretrain+和BLAST，以及来自KernelSynth和CauKerV2的合成数据，后者是一种通过随机采样结构因果模型来组合时间动态的在线数据生成器。训练过程将渐进式收敛调度（可产出可复用的中间检查点）与针对中间层的深度分位数监督目标相结合。最终得到的1.45亿参数骨干网络支持最多8192个观测值的长上下文，可服务于预测、分类和异常检测任务，同时一个提示调优模块进一步提升了分布内预测性能。

    arXiv:2609.13956v1 Announce Type: new  Abstract: In this report, we release Tabby, a long context probabilistic time series foundation model, together with a complete and open recipe of how it was built. Tabby adopts an encoder-only patch Transformer architecture and concentrates the contributions on the data and the training procedure. The pretraining corpus combines an extended real-world collection, GIFT-Eval-Pretrain+ and BLAST, with synthetic data from KernelSynth and CauKerV2, an online generator that composes temporal dynamics through randomly sampled structural causal models. Training couples a progressive convergence schedule, which yields reusable intermediate checkpoints, with a deep quantile supervision objective for intermediate layers. The resulting 145M parameter backbone supports contexts of up to 8,192 observations and serves forecasting, classification, and anomaly detection, while a prompt-tuning module further improves in-distribution forecasting performance with th
    
[^335]: 具体化作为可迁移词汇表：用普通GNN实现零样本链接预测

    Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs

    [https://arxiv.org/abs/2609.11347](https://arxiv.org/abs/2609.11347)

    该研究提出将知识图谱“具体化”为事实节点并通过六个元关系的固定词汇表连接，使普通GNN（如现成的GAT）无需任何专用架构即可实现零样本归纳链接预测，性能媲美预训练基础模型ULTRA。

    

    诸如ULTRA之类的知识图谱基础模型，通过将迁移机制硬编码进专用架构，在未见过的图上实现零样本链接预测。在这项工作中，我们将该机制从架构中移出并融入表示之中，具体做法是对输入图进行“具体化”（reification）处理：每条事实都变成一个节点，通过由六个元关系组成的固定词汇表与其主语、宾语和关系类型相连，其中关系类型作为匿名的共享节点而非模型参数。在这种表示上，五个教科书式的GNN（GAT、采用求和聚合的GINE、采用mean+max聚合的GINE、GraphSAGE、R-GCN），每个仅在包含4,245个三元组的单个知识图谱上、使用一块NVIDIA A100训练30分钟，即可零样本迁移到40个归纳式链接预测基准上。其中表现最好的——一个现成的GAT——在ULTRA自己的评估套件上与ULTRA（一个在三个图上预训练的专用基础模型）表现相当。同样的固定词汇表还扩展了……

    arXiv:2609.11347v1 Announce Type: new  Abstract: Knowledge graph foundation models such as ULTRA achieve zero-shot link prediction on unseen graphs through dedicated architectures that hard-code a transfer mechanism. In this work we move that mechanism out of the architecture and into the representation, by \emph{reifying} the input graph: every fact becomes a node, connected to its subject, object, and relation type through a fixed vocabulary of six meta-relations, with relation types as anonymous shared nodes rather than model parameters. On this representation, five textbook GNNs (GAT, GINE with sum and with mean+max aggregation, GraphSAGE, R-GCN), each trained on a single knowledge graph of 4,245 triples for 30 minutes on one NVIDIA A100, transfer zero-shot to 40 inductive link-prediction benchmarks. The best of them, an off-the-shelf GAT, matches ULTRA, a dedicated foundation model pretrained on three graphs, across ULTRA's own evaluation suite. The same fixed vocabulary extends t
    
[^336]: 在线非单调DR-次模最大化：匹配离线0.401近似比

    Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor

    [https://arxiv.org/abs/2609.02145](https://arxiv.org/abs/2609.02145)

    该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。

    

    我们研究在 $d$ 维单位立方体的紧凸下闭子集上，非负、非单调DR-次模函数的在线最大化问题。在相应的元可解性假设下，目前已知最好的构造性离线近似比为 $0.401$，而可比的对抗性在线保证一直停留在 $1/e$。我们证明该 $0.401$ 近似比同样可以在线实现。在决策后全信息值预言机模型中，当预言机反馈条件无偏且有界时，我们的算法以次线性近似遗憾达到了 $0.401$ 的近似比。该在线算法并不在变化的目标函数上运行离线构造，而是用加权在线学习器替代离线算法中依赖于目标的箱约束步骤，以累积方式控制所需的残差项。一个精确的非对称平衡定理使得离线系数在对抗性变化下仍得以保持。直接实现（原文摘要在此处截断）……

    arXiv:2609.02145v1 Announce Type: cross  Abstract: We study online maximization of nonnegative, non-monotone DR-submodular functions over compact convex down-closed subsets of the $d$-dimensional unit cube. The best known constructive offline approximation factor is $0.401$ under the corresponding meta-solvability assumptions, whereas comparable adversarial online guarantees had remained at $1/e$. We show that this factor is also achievable online. In the post-decision full-information value-oracle model, our algorithm attains factor $0.401$ with sublinear approximate regret when oracle feedback is conditionally unbiased and bounded.   The online algorithm does not run the offline construction on a changing objective. Instead, it replaces the offline objective-dependent box step by a weighted online learner that controls the required residual terms cumulatively. An exact asymmetric balance theorem preserves the offline coefficients despite adversarial variation. The direct implementati
    
[^337]: LAION-BVD：一个面向多模态预训练的千万小时开放视频数据集

    LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training

    [https://arxiv.org/abs/2608.24845](https://arxiv.org/abs/2608.24845)

    该论文介绍了LAION-BVD，一个包含1000万小时视频的大规模开放多模态数据集，通过合成描述和场景帧提取，显著提升了视频、音频和图像-文本任务的预训练性能。

    

    arXiv:2608.24845v1 公告类型：交叉 摘要：我们提出了LAION-BVD，一个用于多模态学习的大规模开放视频数据集，其中包含从CommonCrawl收集的13亿个平台特定视频URL。从中，我们下载了8000万个视频，总时长达到1000万小时。该数据集专为跨视频、音频和图像模态的多模态预训练而设计。利用内容感知的场景检测，我们提取片段，并为其合成生成视频和音频描述。在这些数据上训练的模型在标准视频-文本和音频-文本基准测试中取得了具有竞争力的性能，并随着训练或模型规模的增加而持续改进。此外，我们通过提取场景变化帧，探索了视频帧作为图像-文本数据的替代来源。这些帧展现出与标准网络图像语料库不同的视觉分布，基于该数据集训练的模型在图像-文本检索性能上表现强劲。我们将LAION-BVD发布给研究社区。

    arXiv:2608.24845v1 Announce Type: cross  Abstract: We present LAION-BVD, a large-scale open video dataset for multimodal learning, which contains 1.3B platform-specific video URLs collected from CommonCrawl. From these, we download 80M videos with a total duration of 10 million hours. The dataset is designed for multimodal pre-training across the video, audio, and image modalities. Using content-aware scene detection, we extract clips for which we synthetically generate video and audio captions. Models trained on these data achieve competitive performance on standard video-text and audio-text benchmarks, with consistent improvements as training or model scale increases. Additionally, we explore video frames as an alternative source of image-text data by extracting scene-changing frames. These frames exhibit a visual distribution distinct from standard web image corpora, and models trained on this dataset achieve strong image-text retrieval performance. We release LAION-BVD to the resea
    
[^338]: CrevasseSeg：一种标签高效的无人机裂缝分割框架

    CrevasseSeg: A Label-Efficient UAV Crevasse Segmentation Framework

    [https://arxiv.org/abs/2608.15790](https://arxiv.org/abs/2608.15790)

    本文提出CrevasseSeg框架，通过自监督学习在仅有24块标注数据的情况下，实现无人机影像中冰川裂缝的高效分割，显著降低标注成本。

    

    摘要：从无人机（UAV）影像中绘制裂缝地图对冰川学研究和冰川地形的野外安全至关重要。然而，对冰川表面进行像素级标注成本高昂且需要领域专家。我们提出了CrevasseSeg，一个用于挪威斯瓦尔巴群岛Borebreen冰川末端的二值分割框架，包含1,938块未标注的无人机正射影像拼接瓦片用于自监督/无监督微调，24块标注瓦片用于验证，以及176块标注瓦片用于测试。利用CrevasseSeg，我们对五种自监督目标进行了基准测试——BYOL、Jensen-Shannon散度（JSD）目标、Barlow-Twins、VICReg以及组合的BYOL-JSD目标——并跨越三种架构：O-Net、O-Net++和DINOv3初始化的O-Net。每种配置在两种冻结特征读取器下评估，这两种读取器仅决策边界形式不同：线性探针和仅在24块标注验证集上拟合的非线性XGBoost分类器。

    arXiv:2608.15790v1 Announce Type: new  Abstract: Crevasse mapping from uncrewed aerial vehicle (UAV) imagery matters for glaciological research and for field safety in glaciated terrain. Yet, pixel-level annotation of glacier surfaces is costly and requires domain experts. We introduce CrevasseSeg, a framework for binary segmentation over the terminus of Borebreen, Svalbard, comprising 1,938 unlabelled UAV orthomosaic tiles for self-supervised/unsupervised fine-tuning, 24 labelled tiles for validation and 176 labelled tiles for testing. Using CrevasseSeg, we benchmark five self-supervised objectives -- BYOL, a Jensen-Shannon Divergence (JSD) objective, Barlow-Twins, VICReg, and a combined BYOL-JSD objective -- across three architectures: O-Net, O-Net++, and a DINOv3-initialised O-Net. Each configuration is evaluated under two frozen-feature readouts that differ only in the form of their decision boundary: a linear probe and a non-linear XGBoost classifier fit only on the 24 labelled va
    
[^339]: ED-DiT：基于电子密度的物理引导扩散预训练，用于可迁移的分子表示

    ED-DiT: Physics-Guided Diffusion Pretraining for Transferable Molecular Representations from Electron Density

    [https://arxiv.org/abs/2608.03260](https://arxiv.org/abs/2608.03260)

    提出了 ED-DiT——一种物理引导的扩散 Transformer，通过在不同噪声水平下重建被破坏与掩码的对数电子密度场，并结合电子数一致性约束进行自监督预训练，学习可跨多种电子结构相关任务迁移的分子表示。

    

    预训练在学习可迁移表示方面已展现出强大潜力，但其在基于电子密度的分子学习中仍未得到充分探索。电子密度为分子电子结构提供了连续的三维描述，既捕捉局部空间模式，也蕴含全局物理量。这引出了一个关键问题：电子密度场能否用于自监督预训练，以学习一种可在多种电子结构相关任务间迁移的共享表示？我们提出了 ED-DiT，一种物理引导的扩散 Transformer，用于在电子密度点云上进行自监督预训练。ED-DiT 通过在不同扩散噪声水平下重建被破坏和部分掩码的对数密度场来学习可复用的表示，并进一步引入电子数一致性约束以保持总电子质量。预训练得到的编码器可以适配到性质预测……（原文摘要在此处截断）

    arXiv:2608.03260v2 Announce Type: replace  Abstract: Pretraining has shown strong potential for learning transferable representations, yet it remains underexplored for electron-density-based molecular learning. Electron density provides a continuous three-dimensional description of molecular electronic structure, capturing both local spatial patterns and global physical quantities. This raises a key question: can electron-density fields be used for self-supervised pretraining to learn a shared representation that transfers across diverse electronic-structure-related tasks? We propose ED-DiT, a physics-guided Diffusion Transformer for self-supervised pretraining on electron-density point clouds. ED-DiT learns reusable representations by reconstructing corrupted and partially masked log-density fields across diffusion noise levels. An electron-number consistency constraint is further introduced to preserve the total electronic mass. The pretrained encoder can be adapted to property predi
    
[^340]: 从图信号处理视角看大语言模型上下文学习中的数值序列表示

    A Graph Signal Processing Perspective on Numerical Sequence Representations in LLM In-Context Learning

    [https://arxiv.org/abs/2608.03015](https://arxiv.org/abs/2608.03015)

    本文提出用图信号处理视角研究大语言模型上下文学习中数值信息的内部组织，将注意力视为token加权图、隐藏状态视为图上信号，发现随着上下文变长，表示按输入动力学复杂度分化：简单输入对应更强的全局连通性和更平滑、谱更集中的隐藏状态信号。

    

    预训练大语言模型已在数值推理上展现出上下文学习能力，即能够对序列化为文本的数值序列进行推理。先前的工作主要通过预测误差等输出层面的评估来识别和刻画这种数值推理能力。然而，数值信息在大语言模型内部表示中如何组织仍知之甚少。为了研究这种内部组织结构，我们采用了图信号处理的视角：注意力机制在token之间诱导出一个加权图，而token的隐藏状态则构成该图节点上的信号。定量的图谱诊断与定性的token图可视化共同揭示出：随着上下文长度的增加，模型表示会依据输入的动力学复杂度而变得更加清晰地分化。较简单的输入所产生的注意力诱导token图具有更强的全局连通性，其隐藏状态信号也更平滑、谱能量更加集中……

    arXiv:2608.03015v2 Announce Type: replace-cross  Abstract: Pretrained large language models (LLMs) have demonstrated in-context learning (ICL) capabilities for numerical inference over sequences serialized as text. Prior work has identified and characterized this form of numerical inference primarily through output-level evaluations such as prediction error. However, how numerical information is organized within LLM representations remains much less understood. To study this internal organization, we adopt a graph signal processing perspective in which attention induces a weighted graph over tokens, while token hidden states define signals on its nodes. Quantitative graph-spectral diagnostics and qualitative token-graph visualizations reveal that representations become more clearly differentiated by input dynamical complexity as context length increases. Simpler inputs produce attention-induced token graphs with stronger global connectivity and smoother, spectrally concentrated hidden-
    
[^341]: 生成模型下强化学习的改进量子算法

    Improved Quantum Algorithms for Reinforcement Learning Under a Generative Model

    [https://arxiv.org/abs/2608.02826](https://arxiv.org/abs/2608.02826)

    本文将标准值迭代与量子均值估计、量子最大值查找等量子子程序相结合，提出了求解有限时域和无限时域折扣MDP近似最优策略的新量子算法，其查询复杂度优于以往工作并接近已有的量子下界。

    

    强化学习是机器学习的一个子领域，研究智能体如何与环境进行交互，从而获取尽可能大的奖励。研究这种交互的标准方法是通过马尔可夫决策过程（MDP）以及选择最优策略的任务——策略是一个告诉智能体应采取何种动作的函数。在这项工作中，我们研究了两种时间相关的MDP——有限时域MDP和无限时域折扣MDP，并提出了用于计算近似最优策略的新量子算法。我们的量子算法基于标准值迭代与量子子程序（如量子均值估计和量子最大值查找）的一种新组合，并整体上融合了样本最优经典算法的技术加以增强。我们得到的查询复杂度优于以往工作，从而接近已经建立的量子下界。

    arXiv:2608.02826v2 Announce Type: replace-cross  Abstract: Reinforcement learning is a subfield of machine learning that studies how an agent interacts with an environment in order to extract as large a reward as possible. A standard approach to study such interaction is through Markov Decision Processes (MDPs) and the task of choosing an optimal policy---a function that tells the agent which action to take. In this work, we study two types of time-dependent MDP---finite-horizon and infinite-horizon discounted---and propose new quantum algorithms for computing approximate optimal policies. Our quantum algorithms are based on a new combination of standard value iteration and quantum subroutines like quantum mean estimation and quantum maximum finding, overall enhanced with techniques from sample-optimal classical algorithms. Our resulting query complexities improve upon previous works, thus approaching already established quantum lower bounds.
    
[^342]: QSTAR：具有自适应路由的量子选择性迁移学习

    QSTAR: Quantum Selective Transfer with Adaptive Routing

    [https://arxiv.org/abs/2607.21411](https://arxiv.org/abs/2607.21411)

    本文提出QSTAR框架，通过自适应路由机制保留高置信度的经典预测、仅将低置信度样本交给量子分支处理，从而让量子迁移学习中量子分支的作用更加明确。

    

    量子迁移学习（QTL）通常通过用一个固定的变分量子头替换经典分类器来进行评估，但这掩盖了一个关键问题：量子分支究竟何时才真正有用？我们提出了QSTAR（具有自适应路由的量子选择性迁移），这是一个选择性QTL框架，它保留高置信度的经典预测，仅将低置信度的样本路由到备用分支。在Fashion-MNIST数据集上使用冻结的ResNet18骨干网络，我们在共同的数据划分和优化调度下，比较了人工设计的QTL头、KetGPT设计的量子头以及参数匹配的经典基线。标准QTL头的准确率最高仅为57.0%，而在主要筛选实验中，最强的KetGPT头达到了78.5%的准确率和0.785的F1分数。尽管最强的固定经典头仍然更高，达到81.6%，但选择性路由使量子分支的作用更加明确。在低置信度样本上，KetGPT #180提升了

    arXiv:2607.21411v1 Announce Type: cross  Abstract: Quantum transfer learning (QTL) is often evaluated by replacing a classical classifier with a fixed variational quantum head, but this hides a key question: when is the quantum branch actually useful? We propose QSTAR: Quantum Selective Transfer with Adaptive Routing, a selective QTL framework that keeps high-confidence classical predictions and routes only low-confidence samples to a fallback branch. Using a frozen ResNet18 backbone on Fashion-MNIST, we compare manually designed QTL heads, KetGPT-designed quantum heads, and parameter-matched classical baselines under a common data split and optimization schedule. Standard QTL heads reach at most 57.0% accuracy, while the strongest KetGPT head in the main filtered sweep reaches 78.5% accuracy and 0.785 F1-score. Although the strongest fixed classical head remains higher at 81.6%, selective routing gives the quantum branch a clearer role. On low-confidence samples, KetGPT #180 improves 
    
[^343]: 一次性系统辨识的零一律

    A zero-one law for one-shot system identification

    [https://arxiv.org/abs/2607.15832](https://arxiv.org/abs/2607.15832)

    该论文证明了一次性系统辨识的零一律：当字典项线性无关时，几乎所有从非退化高斯测度采样的随机输入都能唯一确定模型系数，否则没有任何输入可以做到，从而将单次实验辨识问题归结为退化输入的判定问题。

    

    一个模型能否通过一次实验被辨识出来？我们研究了由一组预定字典项（如偏微分算子和动力系统）的组合进行线性参数化的解析系统。对于单个输入-响应对，当被求值的字典项线性无关时，恰好可以实现系统的精确恢复。我们证明了一个深刻的零一律：要么没有任何输入能唯一确定系数，要么几乎所有从非退化高斯测度中采样得到的随机输入都能做到。这一二分性将一次性系统辨识问题归结为关于退化输入的问题，并在无噪声情形下提供了事后验证证书。数值实验表明，可以从单条轨迹数据中恢复动力系统、非线性偏微分方程和结构化矩阵族，同时还能检测出何时需要额外的探测输入。

    arXiv:2607.15832v2 Announce Type: replace-cross  Abstract: Can a model be identified from one experiment? We study analytic systems that are linearly parameterized by a combination of prescribed dictionary terms, such as partial differential operators and dynamical systems. For a single input-response pair, recovery is possible exactly when the evaluated dictionary terms are linearly independent. We prove a sharp zero-one law: either no input uniquely determines the coefficients, or almost every random input sampled from a nondegenerate Gaussian measure does. This dichotomy reduces one-shot system identification to a question about degenerate inputs and provides an a posteriori certificate in the noiseless setting. Numerical examples recover dynamical systems, nonlinear partial differential equations, and structured matrix families from single trajectory data, while also detecting when an extra probe is necessary.
    
[^344]: 面向可穿戴设备的低功耗高能效心律失常检测

    Toward Energy-Efficient and Low-Power Arrhythmia Detection for Wearable Devices

    [https://arxiv.org/abs/2607.14747](https://arxiv.org/abs/2607.14747)

    本文通过在深度学习心律失常检测模型及其硬件架构中采用数据精度降低和近似乘法等近似技术，在保持可接受分类精度的同时显著降低了功耗与能耗，使检测系统能够高效部署于可穿戴设备。

    

    心血管疾病是全球主要的死亡原因，诸如心律失常等病症通常需要长期监测才能实现有效的检测和诊断。然而，目前的可穿戴监测设备体积庞大、佩戴不舒适，且通常依赖临床医生手动评估心电图（ECG）。虽然深度学习（DL）算法在心律失常检测和分类方面展现出卓越的性能，但其计算复杂性和高功耗限制了其在可穿戴设备上的部署。为了应对这一挑战，本文研究了利用近似技术来降低深度学习架构的功耗和能量消耗，同时保持可接受的分类性能。具体而言，在一个最先进的深度学习模型及其相应的硬件架构中，研究了数据精度降低和近似乘法等技术。

    arXiv:2607.14747v2 Announce Type: replace-cross  Abstract: Cardiovascular diseases are the leading cause of death worldwide, and conditions such as arrhythmia often require long-term monitoring for effective detection and diagnosis. However, current wearable monitoring devices are bulky, uncomfortable, and typically rely on clinicians to manually evaluate electrocardiograms (ECGs). While Deep Learning (DL) algorithms have shown superior performance in arrhythmia detection and classification, their computational complexity coupled with high power consumption limit deployment in wearable devices. To address this challenge, this paper investigates the use of approximation techniques to reduce the power and energy consumption of DL architectures while maintaining acceptable classification performance. Specifically, techniques such as data precision reduction and approximate multiplication are investigated in a state-of-the-art DL model and its corresponding hardware architecture. The model
    
[^345]: 高斯索博列夫算子的近最优学习

    Near-Optimal Learning of Gaussian Sobolev Operators

    [https://arxiv.org/abs/2607.11921](https://arxiv.org/abs/2607.11921)

    本文提出了 Hermite-PCA 逼近算法，一种采用主成分分析与加权最小二乘的完全数据驱动且计算高效的方法，能以近最优的样本复杂度学习高斯索博列夫算子。

    

    算子学习中的一个关键问题是如何在合理的计算时间内设计具有可证明逼近保证的代理算子。虽然全纯算子可以被高效逼近，即随着训练样本数量的增加至少具有代数收敛性，但已知学习有限正则性的算子效率较低。其原因在于样本复杂度的内在“维数灾难”，这使得只能达到低于代数的收敛速率。这一事实使得开发能够可证明地达到这些速率的算法变得更加重要。在这项工作中，我们提出了一种完全数据驱动的算法，称为 Hermite-PCA 逼近，用于以近最优的样本复杂度学习高斯索博列夫算子。该算法采用主成分分析和加权最小二乘方法，因此计算效率高。此外，它是谱方法意义上的算法，即随着……可实现更快（且近最优）的收敛。

    arXiv:2607.11921v2 Announce Type: replace-cross  Abstract: A key question in operator learning is how to design surrogate operators with provable approximation guarantees in reasonable computational time. Whereas holomorphic operators can be approximated efficiently, i.e., with at least algebraic convergence in the number of training samples, learning finitely regular operators is known to be less efficient. The reason is an intrinsic curse of sample complexity, which allows only subalgebraic convergence rates. This fact makes it all the more important to develop algorithms which provably achieve these rates. In this work, we present a fully data-driven algorithm, termed Hermite-PCA approximation, for learning Gaussian Sobolev operators with near-optimal sample complexity. It employs principal component analysis and weighted least-squares methods and is therefore computationally efficient. Moreover, it is spectral, in the sense that it achieves faster (and near-optimal) convergence as 
    
[^346]: 还剩多少？大语言模型线性编码其剩余输出长度

    How Much is Left? LLMs Linearly Encode Their Remaining Output Length

    [https://arxiv.org/abs/2607.05316](https://arxiv.org/abs/2607.05316)

    本文发现大语言模型在隐藏状态中线性编码了剩余输出长度信息，仅需一个简单的线性探针即可准确预测生成回答还剩多少token，且当出现“Wait”等撤回token时预测长度会显著跳升。

    

    大语言模型逐个生成token，并不会显式地追踪所生成文本的长度。本文提出了这样一个问题：我们能否仅通过对模型隐藏状态使用线性探针，来估计一个生成回答中剩余的token数量？先前关于推理调度的工作已经表明，文本长度在统计上是可预测的，但使用的是非线性方法。本文贡献了一种线性表示方法，使用7-8B指令微调模型的冻结隐藏状态，在七个数据集上进行了实验。我们证明，总的补全长度是可以被估计的，其性能优于统计基线，也优于与提示词最后一个token处预测结果的公平对比。在自然语言数据集上，该探针的误差介于统计基线误差的0.57到0.88之间。有趣的是，当出现撤回token（"Wait"）时，探针预测的长度会向上跳升超过70个token。

    arXiv:2607.05316v2 Announce Type: replace  Abstract: Large Language models generate tokens, without explicitly keeping track of the length of the generated text. This paper asks the question: can we estimate the number of tokens remaining in a generated response, using just a linear probe on the model's hidden states? Previous work on inference scheduling established that length is statistically predictable, using nonlinear methods. This paper contributes a linear representation method, using frozen hidden states of 7-8B instruction-tuned models, across seven datasets. We show that total completion length can be estimated, with performance stronger than statistical baselines, and stronger than the fair comparison to the prediction at the last token of the prompt. On the natural-language datasets, the probe's error lies between 0.57 and 0.88 of the statistical baseline's. Interestingly, when a retraction token occurs ("Wait"), the probe predicted length jumps upward by over 70 tokens, o
    
[^347]: 基于深度受限重建评分的探地雷达地下隧道无监督检测

    Unsupervised Detection of Underground Tunnels in Ground-Penetrating Radar Using Depth-Restricted Reconstruction Scoring

    [https://arxiv.org/abs/2607.04882](https://arxiv.org/abs/2607.04882)

    该论文提出了一种无需任何隧道样本的无监督检测方案，利用卷积自编码器学习场地正常地下状况，并通过深度受限的重建误差评分，在实地探地雷达数据中实现了近乎完美的地下隧道检测性能（AUC达0.994）。

    

    从下方挖至埋地管道的手工隧道在被利用之前很难被发现，而且此类样本过于稀少，无法训练监督式检测器。本工作的工程应用是在受监测场地的探地雷达勘测中检测此类隧道。其在人工智能方面的贡献是一种无需隧道样本的检测方案：卷积自编码器从任何隧道存在之前记录的勘测数据中学习场地的正常地下状况，随后通过在隧道可能存在的走时频带内的最大重建误差对后续勘测窗口进行评分，并且评估时对整条勘测测线进行了留出。在本研究专门手工挖掘的隧道实地数据上（宽0.6米、高1米、顶板深度0.9-1.4米），使用250兆赫兹雷达在干燥和潮湿条件下反复勘测，该检测器在1600个测试窗口上达到了0.994的受试者工作特征曲线下面积（AUC）和0.972的F1分数。

    arXiv:2607.04882v2 Announce Type: replace-cross  Abstract: Hand-dug tunnels that reach buried pipelines from below are hard to find before they are used, and too rare to train a supervised detector. The engineering application of this work is detecting such tunnels in ground-penetrating radar surveys of a monitored site. The contribution in artificial intelligence is a detection scheme that needs no tunnel examples: a convolutional autoencoder learns the site's normal ground from surveys recorded before any tunnel exists, later survey windows are scored by their largest reconstruction errors within the travel-time band where a tunnel can lie, and evaluation holds out whole survey lines. On field data from a tunnel hand-dug for this study (0.6 m wide, 1 m high, roof 0.9-1.4 m deep) and surveyed repeatedly with a 250-megahertz radar in dry and wet conditions, the detector reaches an area under the receiver operating characteristic curve of 0.994 and an F1 score of 0.972 on 1,600 test win
    
[^348]: 基于拉格朗日奖励增强的安全推理时对齐

    Safe Inference-Time Alignment via Lagrangian Reward Augmentation

    [https://arxiv.org/abs/2607.02781](https://arxiv.org/abs/2607.02781)

    提出LARA框架，通过拉格朗日对偶将安全约束转化为由小型校准集估计的对偶变量所定义的增强奖励，实现可即插即用于现有推理时对齐方法的无需权重更新的安全对齐。

    

    推理时对齐在解码过程中利用辅助奖励信号来引导冻结的语言模型，从而避免了反复更新权重的高昂代价。然而，现有的推理时对齐方法通常只优化单一的标量分数，因此显式的安全约束要么被忽略，要么只能通过手工调整的惩罚项来编码。我们提出了拉格朗日奖励增强（LARA），一个在安全约束下的通用推理时对齐框架。从包含奖励模型和代价模型的KL正则化约束目标出发，LARA对约束进行对偶化，将优化问题化为关于一个非负对偶变量的一维凸问题。该对偶变量通过一个小型校准集估计得到，并由其定义的增强奖励可作为现有推理时对齐方法中即插即用的评分信号。对于序列级采样方法，例如Best-of-N重排序，该……（原文摘要在此处截断）

    arXiv:2607.02781v3 Announce Type: replace-cross  Abstract: Inference-time alignment steers a frozen language model during decoding using auxiliary reward signals, avoiding the cost of repeated weight updates. However, existing inference-time alignment methods typically optimize a single scalar score, so explicit safety constraints must either be ignored or encoded through manually tuned penalties. We propose Lagrangian Reward Augmentation (LARA), a general inference-time alignment framework under safety constraints. Starting from a KL-regularized constrained objective with a reward model and a cost model, LARA dualizes the constraint and reduces the optimization problem to a one-dimensional convex problem over a nonnegative dual variable. Estimated on a small calibration set, this dual variable defines an augmented reward that can be used as a drop-in scoring signal within existing inference-time alignment methods. For sequence-level sampling methods, such as Best-of-N reranking, the c
    
[^349]: 表格上下文学习者能否泛化到生物分子性质预测？

    Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?

    [https://arxiv.org/abs/2606.31126](https://arxiv.org/abs/2606.31126)

    本文评估了TabPFN和TabICL等表格基础模型能否作为小样本场景下的数据高效预测器泛化到生物分子性质预测任务，发现在ProteinGym上TabPFN3-ESMC600M与领先的监督方法具有竞争力并取得最低MSE，但Kermut在Spearman相关性上仍更强，且在更难的划分设置下性能明显下降。

    

    从有限的标注数据预测生物分子性质是蛋白质工程和小分子设计中的一个核心瓶颈。随着强大的预训练编码器如今能够提供丰富的固定长度表示，难点已从表示学习转向为小样本场景构建数据高效的预测器。TabPFN和TabICL等表格基础模型在合成表格上完成预训练后，可通过上下文推理来执行此类预测。我们测试了它们能否作为在信息丰富的固定生物分子表示之上运作的灵活插值器。将每种方法视为预测器-表示对，我们在两个领域进行了评估。在ProteinGym上，TabPFN3-ESMC600M与领先的监督替换模型条目具有竞争力，并在所考虑的排行榜上下文行中取得了最低的MSE，而Kermut在Spearman相关性上仍更胜一筹。在模运算和连续留出设置下，性能显著降低……（原文摘要在此处截断）

    arXiv:2606.31126v3 Announce Type: replace  Abstract: Predicting biomolecular properties from limited labeled data is a central bottleneck in protein engineering and small-molecule design. As strong pretrained encoders now supply rich fixed-length representations, the difficulty has shifted from representation learning to building a data-efficient predictor for the few-shot regime. Tabular foundation models such as TabPFN and TabICL perform this prediction through in-context inference after pretraining on synthetic tables. We test whether they can act as flexible interpolators over informative fixed biomolecular representations. Treating each method as a predictor-representation pair, we evaluate across two domains. On ProteinGym, TabPFN3-ESMC600M is competitive with leading supervised-substitution entries and obtains the lowest MSE among the leaderboard-context rows considered, while Kermut remains stronger by Spearman. Performance is substantially lower under modulo and contiguous hol
    
[^350]: 反坍缩动力学与递归神经网络中多时间尺度学习的涌现

    Anti-Collapse Dynamics and the Emergence of Multi-Time-Scale Learning in Recurrent Neural Networks

    [https://arxiv.org/abs/2606.29519](https://arxiv.org/abs/2606.29519)

    本文揭示了递归神经网络中遗忘速率（指数或幂律）并非由架构预设，而是由状态与参数动力学的耦合自发涌现，形成坍缩或反坍缩两种状态，从而决定长程学习的可行性。

    

    arXiv:2606.29519v3 公告类型：替换 摘要：对于使用随机梯度下降训练的递归网络，长程学习是困难的，因为过去输入的影响会随着滞后ℓ而衰减，如果衰减过快，则无法从有限数据中学习这种依赖性。这种衰减由一个包络函数f(ℓ)描述。指数衰减使得学习滞后ℓ依赖性所需的数据呈指数增长，使长时域变得遥不可及；而幂律衰减则保持成本为多项式级别。我们表明，f(ℓ)的渐近衰减行为并非由架构固定。相反，它从状态动力学与参数动力学之间的耦合中涌现，并稳定在坍缩态（快速、指数遗忘）或扩展的反坍缩态（缓慢、幂律遗忘）。其直觉是这些耦合动力学中的竞争。训练驱动网络的有效时间尺度向短尺度移动，而罕见、重尾的波动则...

    arXiv:2606.29519v3 Announce Type: replace  Abstract: Long-range learning is hard for recurrent networks trained with stochastic gradient descent, because the influence of a past input fades with the lag $\ell$, and if it fades too fast the dependence cannot be learned from finite data. This fade is captured by an envelope $f(\ell)$. An exponential fade makes the data needed to learn a lag-$\ell$ dependence grow exponentially, putting long horizons out of reach; a power-law fade keeps the cost polynomial. We show that the asymptotic decay behavior of $f(\ell)$ is not fixed by the architecture. Instead, it emerges from the coupling between the state dynamics and parameter dynamics, settling into either a collapsed regime (fast, exponential forgetting) or an extended, anti-collapsed regime (slow, power-law forgetting). The intuition is a competition within these coupled dynamics. Training drives the network's effective time scales toward short ones, while rare, heavy-tailed fluctuations o
    
[^351]: 多中介诅咒：激活修补中的隐藏交互效应

    The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching

    [https://arxiv.org/abs/2606.27510](https://arxiv.org/abs/2606.27510)

    本文从因果中介分析出发重新推导了激活修补的估计量，揭示其中隐藏的交互效应会导致因果重要性依赖于其他组件状态的模型组件被错误地低估或夸大，且这种效应无法通过调整估计器或分析单位来消除。

    

    激活修补是机制可解释性研究中的一种主要工具。它通过估计自然间接效应（NIE）将模型行为的因果责任归因于各个单独组件。通过从因果中介分析重新推导激活修补的估计量，我们发现NIE并不仅仅捕捉通过特定组件的因果效应，它还包含交互效应（INT），用于衡量该组件的因果效应本身在多大程度上依赖于模型中其他组件的状态。一个自然的应对方法可能是尝试通过调整估计器或分析单位来消除INT，但这些潜在补救措施中的每一种都存在可预测的失败模式。我们在GPT-2 IOI电路中演示了这些失败模式：那些因果重要性依赖于其他组件状态的组件要么不可见，要么被人为地夸大。我们证明INT随着距离...（摘要在此处被截断）

    arXiv:2606.27510v2 Announce Type: replace-cross  Abstract: Activation patching is a primary tool in mechanistic interpretability. It attributes causal responsibility for a model's behavior to each of its individual components by estimating its natural indirect effect (NIE). Re-deriving the activation patching estimand from causal mediation analysis, we find that the NIE does not solely capture the causal effect through the specific component. It also contains interaction effects (INT) that measure how much the component's causal effect itself depends on the states of other components in the model. A natural response may be to try to eliminate INT by adjusting the estimator or unit of analysis, but each of these potential remedies has predictable failure modes. We demonstrate these failure modes in the GPT-2 IOI circuit; components whose causal importance is conditional on the state of other components are either invisible or artificially inflated. We prove that INT scales with the dist
    
[^352]: 基于多GPU PyTorch后端的大规模高斯泼溅

    Large-Scale Gaussian Splatting with a Multi-GPU PyTorch Backend

    [https://arxiv.org/abs/2606.11390](https://arxiv.org/abs/2606.11390)

    该论文提出基于CUDA统一内存的多GPU PyTorch后端，使高斯泼溅重建可扩展至超过10亿个高斯点的城市规模，且模型代码无需显式跨设备通信。

    

    高斯泼溅方法在真实世界的神经重建中日益流行。然而，由于计算和内存的限制，它们在规模和分辨率上往往受到约束。我们提出了一种多GPU高斯泼溅方法，能够将重建扩展到更高的分辨率和更大的场景，同时抽象掉了通常与模型分布式部署相关的代码复杂性。我们展示了具有街道级细节的城市规模重建，包含超过10亿个高斯泼溅点。为实现这一目标，我们提出了一个PyTorch后端，通过CUDA统一内存（并可选支持NVLink）在多个GPU之间分发参数和算子。由于分发发生在算子层面，模型代码无需任何显式的跨设备通信。更广泛地说，该后端将多个GPU暴露为一个聚合的异步PyTorch设备，并支持广泛的PyTorch算子集合，从而……

    arXiv:2606.11390v2 Announce Type: replace-cross  Abstract: Gaussian splatting methods have become increasingly popular for neural reconstruction of the real world. However, they are often limited in scale and resolution due to compute and memory constraints. We present a multi-GPU Gaussian splatting approach that scales reconstruction to higher resolutions and larger scenes while abstracting away the code complexity typically associated with distributing a model. We demonstrate city-scale reconstructions with street-level detail consisting of over 1 billion Gaussian splats. To accomplish this, we propose a PyTorch backend that distributes parameters and operators across GPUs via CUDA unified memory, with optional NVLink support. Because distribution occurs at the operator level, the model code requires no explicit cross-device communication. More broadly, the backend exposes multiple GPUs as an aggregate asynchronous PyTorch device and supports a broad set of PyTorch operators enabling
    
[^353]: 在你困惑之处采样：通过熵引导的幂采样锐化基础模型推理能力

    Sample Where You Struggle: Sharpening Base Model Reasoning via Entropy-Guided Power Sampling

    [https://arxiv.org/abs/2606.09926](https://arxiv.org/abs/2606.09926)

    提出熵引导幂采样（EGPS），利用前向传播中已有的token级熵将MCMC移动聚焦到高熵决策点，以无需训练、无需验证器的方式高效提升基础模型从幂分布采样带来的强化学习级推理能力。

    

    从序列级幂分布中进行采样，能够在不更新任何参数的情况下，使基础语言模型展现出强化学习级别的推理能力，但标准的Metropolis--Hastings（MH）采样器（一种马尔可夫链蒙特卡洛（MCMC）采样器）既昂贵又混合缓慢。我们将这两个问题归因于一种结构性失配：幂分布主要在一个稀疏且空间聚集的高熵决策点集合处偏离基础分布，而MH却在整条前缀上均匀地提出重采样位置——在近乎退化的条件分布上浪费计算，却在众数真正分歧之处混合不足。我们提出了熵引导幂采样（EGPS），这是一种无需训练、无需验证器的近似采样器，它利用前向传播中已有的token级熵来重新推导其提议分布。EGPS跳过确定性片段，将每次MCMC移动定位到高熵邻域，并在决策点处应用多试Metropolis——使采样……

    arXiv:2606.09926v2 Announce Type: replace-cross  Abstract: Sampling from the sequence-level power distribution elicits RL-level reasoning from base language models without any parameter updates, but the standard Metropolis--Hastings (MH), a Markov Chain Monte Carlo (MCMC) sampler, is both expensive and slow-mixing. We trace both to a structural mismatch: the power distribution mainly departs from the base distribution at a sparse, spatially clustered set of high-entropy decision points, yet MH proposes resampling positions uniformly along the prefix---wasting compute on near-degenerate conditionals while under-mixing precisely where modes diverge. We propose Entropy-Guided Power Sampling (EGPS), a training-free and verifier-free approximate sampler that re-derives its proposal from token-level entropy already in the forward pass. EGPS skips deterministic blocks, localizes each MCMC move to a high-entropy neighborhood, and applies Multiple-Try Metropolis at decision points---making samp
    
[^354]: 理解稀疏自编码器中概念学习与神经元解释的几何视角

    A Geometric View for Understanding Concept Learning and Neuron Interpretation in Sparse Autoencoders

    [https://arxiv.org/abs/2606.07007](https://arxiv.org/abs/2606.07007)

    该论文提出了一个统一的数学几何框架，将稀疏自编码器中的概念学习形式化为人类定义概念与模型概念之间的集合对齐问题，区分了检测、分离和逼近三种学习强度，并从集合论角度解释了特征分裂、特征吸收等常见SAE现象。

    

    我们提出了一个统一的数学框架，用于从几何角度理解稀疏自编码器（SAEs）中的概念学习和神经元解释。虽然SAEs通过学习稀疏特征表示来提高神经网络的可解释性，但“概念”和“学习”的原则性定义仍不明确。我们将概念形式化为数据点的集合，并将概念学习转化为人类定义的概念与模型诱导的概念之间的集合对齐问题。这一形式化区分了三种逐渐增强的学习概念——检测、分离和逼近——并给出了概念何时可以由单个神经元或多神经元单元表示的几何条件、误差界和容量约束。它还为SAE中的常见现象提供了集合论层面的解释，包括特征分裂、特征吸收、特征族和层级概念。最后，我们将概念学习与相关领域联系起来。

    arXiv:2606.07007v2 Announce Type: replace-cross  Abstract: We propose a unified mathematical framework for a geometric understanding of concept learning and neuron interpretation in sparse autoencoders (SAEs). While SAEs improve interpretability of neural networks by learning sparse feature representations, a principled definition of ''concept'' and ''learning'' remains unclear. We formalize concepts as sets of data points and cast concept learning as a set-alignment problem between human-defined and model-induced concepts. This formulation distinguishes three increasingly strong notions of learning -- detection, separation, and approximation -- and yields geometric conditions, error bounds, and capacity constraints for when concepts can be represented by individual neurons or multi-neuron units. It also provides a set-theoretic account for common SAE phenomena, including feature splitting, feature absorption, feature families, and hierarchical concepts. Finally, we connect concept lea
    
[^355]: AlloGen：基于差分状态评分的构象选择性结合剂生成

    AlloGen: Conformation-Selective Binder Generation with Differential State Scoring

    [https://arxiv.org/abs/2606.05474](https://arxiv.org/abs/2606.05474)

    AlloGen提出了一种将主链生成与状态选择性评分解耦的模块化方法，通过SE(3)不变的界面图变换器 $Q_\theta$ 对不同构象状态进行差分评分，从而在无需重新训练的情况下重排序候选并引导冻结的生成器，生成对特定受体构象状态具有选择性的蛋白质结合剂。

    

    蛋白质结合剂设计通常只优化与单一受体构象的兼容性，而未能解决对其他功能状态的选择偏好问题。为填补这一空白，我们提出了AlloGen，这是一种模块化方法，将主链生成与状态选择性评分解耦。我们通过界面质量回归和成对构象判别两个阶段，训练其SE(3)不变的界面图变换器 $Q_\theta$。通过比较不同状态间的评分，我们对候选结合剂进行重排序，并在无需重新训练的情况下引导冻结的主链生成器。在一个涵盖15个蛋白质家族、共65个靶点的基准测试中，$Q_\theta$ 在8个保留靶点上取得了平均0.520的Spearman相关系数。所有15种评估的生成器-引导组合在这些靶点上均实现了正的预测选择性，其中RFdiffusion结合朗之万精修在钙调蛋白（CaM）上达到了 $\bar{S}_{\mathrm{cons}}=0.677$。

    arXiv:2606.05474v2 Announce Type: replace-cross  Abstract: Protein binder design commonly optimizes compatibility with a single receptor conformation, leaving preference over alternative functional states unresolved. To address this gap, we introduce AlloGen, a modular method that decouples backbone generation from state-selectivity scoring. We train its SE(3)-invariant interface graph transformer, $Q_\theta$, through interface-quality regression followed by paired conformational discrimination. By comparing scores across states, we rerank candidates and guide frozen backbone generators without retraining. On a benchmark of 65 targets spanning 15 protein families, $Q_\theta$ achieved a mean Spearman correlation of $0.520$ on eight held-out targets. All 15 evaluated generator-guidance combinations achieved positive predicted selectivity averaged over these targets, and RFdiffusion with Langevin refinement reached $\bar{S}_{\mathrm{cons}}=0.677$ on calmodulin (CaM). To test the predicted
    
[^356]: SparDA：面向高效长上下文大语言模型推理的稀疏解耦注意力

    SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference

    [https://arxiv.org/abs/2606.04511](https://arxiv.org/abs/2606.04511)

    SparDA通过在Query、Key、Value之外引入与注意力查询解耦的Forecast投影来预测下一层所需的KV块，实现CPU到GPU预取与当前层计算重叠的前瞻式稀疏注意力选择，从而突破长上下文LLM推理中KV缓存容量与稀疏选择复杂度的双重瓶颈。

    

    稀疏注意力能够降低长上下文大语言模型推理中的计算量与内存带宽需求。然而，仍存在两个关键挑战：（1）KV缓存的容量依然随序列长度增长，而将其卸载到CPU内存会引入PCIe传输瓶颈；（2）预填充阶段的稀疏选择仍保持 O(T²) 的复杂度，在长上下文场景下可能主导注意力计算成本。我们提出了SparDA，一种解耦的稀疏注意力架构，它在Query、Key和Value之外引入了每层的第四个投影——Forecast。Forecast预测下一层所需的KV块，从而实现前瞻性选择，使CPU到GPU的预取与当前层的执行相互重叠。由于Forecast与注意力查询解耦，我们的GQA实现为每个GQA组仅使用一个Forecast头，相比原始的多头选择器降低了选择开销。在所评估的8B模型上，SparDA仅增加不到0.5%的参数，且只训练Forecast投影（摘要在此处截断）。

    arXiv:2606.04511v2 Announce Type: replace  Abstract: Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) sparse selection during prefill retains $O(T^2)$ complexity and can dominate attention cost at long contexts. We propose SparDA, a decoupled sparse attention architecture that introduces a fourth per-layer projection, the Forecast, alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because Forecast is decoupled from the attention query, our GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. On the evaluated 8B models, SparDA adds <0.5% parameters and trains only the Forecast proje
    
[^357]: 面向空间变化场的几何参数化与映射的无分辨率神经代理模型

    Resolution-free neural surrogates for geometric parameterization and mapping with spatially varying fields

    [https://arxiv.org/abs/2605.28551](https://arxiv.org/abs/2605.28551)

    该论文提出了一种无分辨率的神经代理模型，能够以较低计算成本求解由空间变化参数场驱动的几何参数化与映射问题（如畸变校正、可变形图像配准等），避免了反复求解高分辨率偏微分方程系统的昂贵开销。

    

    arXiv:2605.28551v3 公告类型：replace-cross 摘要：许多成像问题需要计算由空间变化的强度场、特征场或密度场所诱导的空间变换。典型例子包括畸变校正、可变形图像配准、基于图谱的分割以及变形驱动的图像分析。这些任务可以被表述为几何映射问题，其中变换需满足保持局部结构、控制边界行为或调节角度畸变等约束。此类表述通常对应变分模型、扩散过程或椭圆型偏微分方程。然而，当底层参数场在不同实例之间变化时，反复求解高分辨率系统的计算代价会变得非常高。在本工作中，我们提出了一种面向几何参数化与映射问题的无分辨率神经代理模型。给定一个空间变化的参数场 $p:\Omega\to\mathbb{R}^m$ 和网格节点 $\{x_i\}_...

    arXiv:2605.28551v3 Announce Type: replace-cross  Abstract: Many imaging problems require computing spatial transformations induced by spatially varying intensity, feature, or density fields. Canonical examples include distortion correction, deformable image registration, atlas-based segmentation, and deformation-driven image analysis. These tasks can be formulated as geometric mapping problems in which the transformation is constrained to preserve local structure, control boundary behavior, or regulate angular distortion. Such formulations typically lead to variational models, diffusion processes, or elliptic partial differential equations. However, repeatedly solving high-resolution systems becomes computationally expensive when the underlying parameter fields vary across instances. In this work, we propose a resolution-free neural surrogate for geometric parameterization and mapping problems. Given a spatially varying parameter field $p:\Omega\to\mathbb{R}^m$ and grid nodes $\{x_i\}_
    
[^358]: 超越表格型MDP的熵正则化softmax策略梯度的全局线性收敛

    Global linear convergence of entropy-regularized softmax policy gradient beyond tabular MDPs

    [https://arxiv.org/abs/2605.24939](https://arxiv.org/abs/2605.24939)

    该论文在连续状态-动作空间和线性函数逼近设定下，通过建立非一致PL不等式并证明其常数沿梯度流一致下有界，首次证明了熵正则化softmax策略梯度超越表格型MDP的全局线性收敛。

    

    我们研究了具有连续状态和动作空间的无限时域熵正则化马尔可夫决策过程（MDP）的策略梯度全局收敛性。我们考虑采用线性函数逼近的对数线性softmax策略，它扩展了表格型softmax参数化，同时保持策略类易于处理。在正则化状态-动作价值函数满足（直至一个状态相关基线的）$Q^\pi_\tau$可实现性的条件下，我们针对两种不同的特征情形建立了非一致的Polyak-Lojasiewicz（PL）不等式。这种非一致性在第一种特征情形中源于状态平均Fisher信息矩阵的退化，在第二种情形中源于未中心化的特征二阶矩矩阵的退化。在这两种情形下，我们都证明了这些非一致常数可以沿梯度流被一致地下界约束。对于具有全仿射跨度的特征，我们证明了KL正则项的径向无界性，并表明……

    arXiv:2605.24939v2 Announce Type: replace  Abstract: We study the global convergence of policy gradient for infinite-horizon entropy-regularized Markov decision processes (MDPs) with continuous state and action spaces. We consider log-linear softmax policies with linear function approximation, which extend the tabular softmax parameterization while retaining a tractable policy class. Under $Q^\pi_\tau$-realizability up to a state-dependent baseline for the regularized state-action value function, we establish a non-uniform Polyak-Lojasiewicz (PL) inequality for two different feature regimes. The non-uniformity arises through degeneracy of the state-averaged Fisher information matrix in the first feature regime and the uncentered feature second-moment matrix in the second. In both regimes we prove that these non-uniform constants can be bounded uniformly from below along the gradient flow. For full-affine-span features, we prove radial unboundedness of the KL regularizer and show that t
    
[^359]: 智能电网中数据驱动最优潮流的可扩展异构图基础模型

    Scalable Heterogeneous Graph Foundation Models for Data-Driven Optimal Power Flow in Smart Grids

    [https://arxiv.org/abs/2605.23194](https://arxiv.org/abs/2605.23194)

    本文提出基于HydraGNN的大规模异构图学习工作流，利用约三百万个涵盖不同规模电网的异构图训练图基础模型，为智能电网数据驱动的最优潮流代理建模提供了可扩展的统一解决方案。

    

    快速可靠的最优潮流（OPF）近似对电力系统运行至关重要，然而异构OPF图模型通常仅在特定架构的实现或有限的训练语料上进行评估。本文提出了一种基于HydraGNN的大规模异构图学习工作流，用于数据驱动的OPF代理建模和图基础模型（GFM）开发。该工作流保留了母线、发电机、负荷、并联支路、交流（AC）线路、变压器以及设备与母线之间的连接关系，并为分布式预处理、多图形处理器（GPU）训练、超参数优化（HPO）和下游适配提供了统一实现。使用来自十个PGLib-OPF（电力电网基准库交流最优潮流算法）案例的约三百万个异构图，涵盖14至13,659个母线的电网规模，我们比较了六种异构图神经网络……

    arXiv:2605.23194v2 Announce Type: replace-cross  Abstract: Fast and reliable optimal power flow (OPF) approximation is important for power system operation, yet heterogeneous OPF graph models are often evaluated either with architecture-specific implementations or on limited training corpora. This paper presents a large-scale heterogeneous graph-learning workflow, built on HydraGNN, for data-driven OPF surrogate modeling and graph foundation-model (GFM) development. The workflow preserves buses, generators, loads, shunts, alternating-current (AC) lines, transformers, and device-to-bus relations and provides a common implementation for distributed preprocessing, multi-graphics-processing-unit (GPU) training, hyperparameter optimization (HPO), and downstream adaptation. Using approximately three million heterogeneous graphs from ten Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms (PGLib-OPF) cases spanning 14 to 13,659 buses, we compare six heterogeneous graph neural
    
[^360]: 以人为中心的学习力学：受保护风险约束下的可观测熵控制

    Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints

    [https://arxiv.org/abs/2605.22940](https://arxiv.org/abs/2605.22940)

    本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。

    

    熵降低并非一个安全准则：它可能使预测保持不变，或伴随着受保护群体风险的上升。我们研究在分别指定的任务、风险与干预约束下的熵调控学习。一个经噪声校准的对数行列式泛函给出了精确的谱力公式，以及将速度度量分解为可观测分量与预测零空间分量的方法。对于标量反馈，我们刻画了在保持任务下降比例并满足熵速率要求时系数的完整区间；而将干预限制在可观测方向上可能使该区间为空。对于多个受保护风险，我们在指定子空间与范数预算内推导出精确的极小极大可行性证书、针对不确定约束行的精确鲁棒扩展，以及保护后剩余熵削减的锥投影公式。这些公式采用标准的凸几何方法，但……

    arXiv:2605.22940v2 Announce Type: replace-cross  Abstract: Entropy reduction is not a safety criterion: it can leave predictions unchanged or accompany increasing risk for a protected group. We study entropy-regulated learning under separately specified task, risk, and intervention constraints. A noise-calibrated log-determinant functional yields an exact spectral force formula and a metric decomposition into observable and prediction-null velocities. For scalar feedback, we characterize the complete interval of coefficients preserving a task-descent fraction and meeting an entropy-rate request; restricting intervention to observable directions can make that interval empty. For several protected risks, we derive an exact minimax certificate of feasibility within a specified subspace and norm budget, an exact robust extension for uncertain constraint rows, and a cone-projection formula for the entropy reduction remaining after protection. These formulas use standard convex geometry but 
    
[^361]: Hyrax：一个在Rubin、Roman和Euclid时代用于快速机器学习实验与无监督发现的可扩展框架

    Hyrax: An Extensible Framework for Rapid ML Experimentation and Unsupervised Discovery in the Era of Rubin, Roman, and Euclid

    [https://arxiv.org/abs/2605.18959](https://arxiv.org/abs/2605.18959)

    Hyrax是一个开源、模块化、GPU加速的Python框架，为Rubin、Roman和Euclid时代的天文机器学习提供覆盖数据获取、训练、推理到实验比较的全生命周期支持，并集成向量数据库相似性搜索与交互式潜空间探索以实现无监督科学发现。

    

    NSF-DOE Vera C. Rubin天文台、Roman空间望远镜、Euclid以及其他下一代巡天项目将以日益增长的规模提供成像、光谱和时域数据，这使得天文机器学习（ML）项目的瓶颈正逐渐从模型设计转移到基础设施上。我们提出了Hyrax，一个开源、模块化、支持GPU的Python框架，支持天文学中完整的机器学习生命周期：从数据获取和训练到推理与实验比较，其功能包括多模态数据集支持、用于相似性搜索的集成向量数据库，以及用于无监督发现的交互式二维和三维潜空间探索。我们通过五个基于真实巡天数据的代表性应用展示了Hyrax的多功能性：（i）对约4×10⁵个RubinLegacy时空巡天（LSST）数据预览1（DP1）星系的无监督表征学习，显现出……（摘要在此处被截断）

    arXiv:2605.18959v2 Announce Type: replace-cross  Abstract: The NSF-DOE Vera C. Rubin Observatory, Roman Space Telescope, Euclid, and other next-generation surveys will deliver imaging, spectroscopic, and time-domain data at scales that increasingly shift the bottleneck in astronomical machine learning (ML) projects from model design to infrastructure. We present Hyrax, an open-source, modular, GPU-enabled Python framework that supports the full ML lifecycle in astronomy: from data acquisition and training to inference and experiment comparison, with capabilities including multimodal dataset support, integrated vector databases for similarity search, and interactive two- and three-dimensional latent-space exploration for unsupervised discovery. We demonstrate Hyrax's versatility through five representative applications on real survey data: (i) unsupervised representation learning on $\sim 4\times10^5$ Rubin Legacy Survey of Space and Time (LSST) Data Preview 1 (DP1) galaxies, surfacing 
    
[^362]: SynVA：用于血管生成与动脉瘤编辑的模块化工具包

    SynVA: A Modular Toolkit for Vessel Generation and Aneurysm Editing

    [https://arxiv.org/abs/2605.17620](https://arxiv.org/abs/2605.17620)

    SynVA 是一个模块化工具包，结合基于流匹配的血管网格生成方法与基于学习的动脉瘤合成方法，生成解剖学一致的脑血管数据，以缓解医学数据稀缺问题并助力颅内动脉瘤的诊断与治疗研究。

    

    颅内动脉瘤（IA）以不可预测的生长和破裂风险为特征，是导致中风的主要原因之一，可引发危及生命的出血，具有高死亡率和长期致残率。随着人口老龄化，脑血管疾病的发病率和总体负担预计将不断增加，这凸显了对可扩展方法的需求，以分析复杂的医学数据并提高对这些疾病的群体层面理解。尽管数字孪生和深度学习为改善诊断、预后和治疗提供了有前景的途径，但它们的有效性受到大规模、高质量医学数据及相应标签稀缺的限制。我们提出了合成血管系统，这是一个用于血管网格生成和解剖学一致性动脉瘤合成的模块化工具包。SynVA 将基于流匹配的新型健康血管网格生成方法与基于学习的方法相结合

    arXiv:2605.17620v2 Announce Type: replace-cross  Abstract: Intracranial aneurysms (IAs), characterized by unpredictable growth and risk of rupture, are a major cause of stroke and can lead to life-threatening hemorrhages with high mortality and long-term disability. With aging populations, the incidence and overall burden of cerebrovascular diseases are expected to increase, highlighting the need for scalable approaches to analyze complex medical data and improve population-level understanding of these conditions. While digital twins and deep learning offer promising avenues for improving diagnosis, prognosis, and treatment, their effectiveness is limited by the scarcity of large-scale, high-quality medical data and corresponding labels. We present Synthetic VAsculature (SynVA), a modular toolkit for vascular mesh generation and anatomically consistent aneurysm synthesis. SynVA combines novel flow-matching-based methods for generating healthy vessel meshes with learning-based approache
    
[^363]: 面向环境感知的长时程船舶轨迹预测的分层两阶段框架

    Hierarchical Two-Stage Framework for Environment-Aware Long-Horizon Vessel Trajectory Prediction

    [https://arxiv.org/abs/2605.16442](https://arxiv.org/abs/2605.16442)

    该论文提出一种分层两阶段框架，结合基于时空图Transformer的短期局部动态预测与长期航行意图编码，并通过跨模态注意力融合洋流、风、浪等海洋环境参数，实现环境感知的长时程船舶轨迹精确预测。

    

    在真实海洋条件下的长时程船舶轨迹预测对于避碰、交通管理和航线规划至关重要。然而，由于长程时间依赖性以及洋流、风、浪等动态环境因素，实现精确预测极具挑战性。为解决这些问题，我们提出了一种分层两阶段框架，通过分层融合机制将粗粒度长期预测器与网格感知短期预测器相结合。短期分支利用时空图Transformer在离散化的海域单元格上捕捉局部动态，而长期分支则编码整体航行意图。集成的环境模块融合了包括表层洋流、风矢量和有效波高在内的海洋学参数，采用跨模态注意力和特征级调制机制，以自适应地响应变化的海洋状况。

    arXiv:2605.16442v2 Announce Type: replace-cross  Abstract: Long-horizon vessel trajectory forecasting under real ocean conditions is critical for collision avoidance, traffic management, and route planning. However, achieving accurate predictions is challenging due to long-range temporal dependencies and dynamic environmental factors such as currents, wind, and waves. To address these issues, we propose a hierarchical two-stage framework that combines a coarse long-term predictor with a grid-aware short-term predictor through a hierarchical fusion mechanism. The short-term branch leverages a Spatio-Temporal Graph Transformer on discretized maritime cells to capture localized dynamics, while the long-term branch encodes overarching navigational intent. An integrated environmental module incorporates oceanographic parameters, including surface currents, wind vectors, and significant wave height, using cross-modal attention and feature-wise modulation for adaptive response to varying sea 
    
[^364]: 核心集诱导的条件速度流匹配

    Coreset-Induced Conditional Velocity Flow Matching

    [https://arxiv.org/abs/2605.12951](https://arxiv.org/abs/2605.12951)

    该论文提出CCVFM生成模型，通过熵Sinkhorn核心集构建闭式代理源替代分层整流流中内部流的高斯噪声源，使条件速度规律成为无需神经采样器即可采样的闭式高斯混合，并只需轻量级校正流即可完成训练。

    

    我们提出了核心集诱导的条件速度流匹配（CCVFM），这是一种生成模型，它通过由数据信息构建的源分布来增强分层整流流。标准流匹配仅回归速度的条件均值；而分层流匹配则通过速度空间中的第二个内部流来建模完整的条件速度规律，但这个内部流需要从头开始将各向同性高斯噪声传输到多峰的目标速度分布。我们的关键观察是：这个内部源可以被一个由目标核心集构建的闭式代理分布所替代。CCVFM首先使用熵正则化的Sinkhorn核心集将目标压缩为加权原子，并将其提升为高斯混合。由此诱导的条件速度规律是一个闭式高斯混合，无需学习神经采样器即可进行采样。基于该代理源训练的轻量级校正流，t...

    arXiv:2605.12951v3 Announce Type: replace  Abstract: We propose Coreset-Induced Conditional Velocity Flow Matching (CCVFM), a generative model that augments hierarchical rectified flow with a data-informed source distribution. Standard flow matching regresses only the conditional mean of the velocity; hierarchical flow matching instead models the full conditional velocity law with a second, inner flow in velocity space, but this inner flow is asked to transport isotropic Gaussian noise to a multimodal target velocity distribution from scratch. Our key observation is that this inner source can be replaced by a closed-form surrogate built from a coreset of the target. CCVFM first compresses the target into weighted atoms using an entropic Sinkhorn coreset and lifts them to a Gaussian mixture. The induced conditional velocity law is then a closed-form Gaussian mixture that can be sampled without a learned neural sampler. A lightweight correction flow, trained from this surrogate source, t
    
[^365]: 作为极坐标状态估计器的Transformer

    The Transformer as a Polar State Estimator

    [https://arxiv.org/abs/2605.11007](https://arxiv.org/abs/2605.11007)

    本文证明Transformer的注意力机制、残差连接和归一化组件可从极坐标下的几何状态估计问题中自然推导得出，并提出保留了这些几何修正项的Polar Transformer。

    

    我们证明，Transformer的核心组件——注意力机制、残差连接和归一化——自然地源于一个单一的几何状态估计问题。在极坐标下对潜在状态进行建模，可以自然地将径向动力学与超球面动力学分离开来，从而产生一种精度加权滤波程序：其中归一化强制执行超球面约束，注意力机制聚合方向证据，残差连接实现增量式状态更新。通过舍弃所得状态估计器中的几何修正项，即可恢复出带有旋转位置编码的标准Transformer块，这表明其架构源自底层的估计问题，而非独立的设计选择。本文提出的“极坐标Transformer”（Polar Transformer）保留了这些几何修正项。

    arXiv:2605.11007v4 Announce Type: replace-cross  Abstract: We show that the core components of the Transformer---attention, residual connections, and normalization---arise naturally from a single geometric state estimation problem. Modeling the latent state in polar coordinates naturally separates radial and hyperspherical dynamics, yielding a precision-weighted filtering procedure in which normalization enforces the hyperspherical constraint, attention aggregates directional evidence, and the residual connection implements an incremental state update. The standard Transformer block with rotary positional encodings is recovered by discarding the geometric correction terms of the resulting state estimator, showing that its architecture follows from the underlying estimation problem rather than from independent design choices. The proposed \textit{Polar Transformer} retains these geometric corrections.
    
[^366]: 不可知在线学习中遗憾值与Oracle复杂度的权衡

    Regret-Oracle Complexity Tradeoffs in Agnostic Online Learning

    [https://arxiv.org/abs/2605.07155](https://arxiv.org/abs/2605.07155)

    本文显著改进了不可知在线学习中Oracle高效学习器的Oracle调用复杂度，且仅依赖更弱的弱一致性Oracle原语，实现了遗憾值与Oracle复杂度之间更优的权衡。

    

    不可知在线学习经典地通过归约到可实现情形来解决，其基础学习器采用Littlestone的标准最优算法（SOA）。然而，SOA即使在单轮的实现上也可能在计算上难以处理。为了克服这一障碍，近期关于Oracle高效在线学习的工作用一种可实现的基础学习器替代了SOA，该学习器仅通过离线经验风险最小化（ERM）Oracle来访问概念类。虽然这类不可知学习器可以达到期望遗憾 $\tilde{\mathcal{O}}\big(\sqrt{T\cdot 2^{\mathcal{O}(d_{\mathrm{LD}})}}\big)$，在固定的Littlestone维数下关于时间范围 $T$ 接近最优，但它们需要 $\mathcal{O}\big(T^{2^{\mathcal{O}(d_{\mathrm{LD}})}}\big)$ 次Oracle调用。在本工作中，我们显著改进了这一Oracle复杂度，同时依赖一个更弱的原语：弱一致性Oracle，它仅需判断……（原文摘要在此处截断）

    arXiv:2605.07155v2 Announce Type: replace  Abstract: Agnostic online learning is classically solved via a reduction to the realizable setting, utilizing Littlestone's Standard Optimal Algorithm (SOA) as a base learner. However, the SOA can be computationally intractable to implement even for a single round. To overcome this barrier, recent work in oracle-efficient online learning replaces the SOA with a realizable base learner that accesses the concept class exclusively through an offline empirical risk minimization (ERM) oracle. While such agnostic learners achieve expected regret $\tilde{\mathcal{O}}\big(\sqrt{T\cdot 2^{\mathcal{O}(d_{\mathrm{LD}})}}\big)$, which is near-optimal in the horizon $T$ for fixed Littlestone dimension $d_{\mathrm{LD}}$, they require $\mathcal{O}\big(T^{2^{\mathcal{O}(d_{\mathrm{LD}})}}\big)$ oracle calls. In this work, we significantly improve this oracle complexity while relying on an even weaker primitive: a weak-consistency oracle, which merely decides 
    
[^367]: EEG基础模型需要通道适配，但固定的通道映射往往已经足够

    EEG Foundation Models Need Channel Adaptation, but Fixed Channel Mappings Often Suffice

    [https://arxiv.org/abs/2604.23091](https://arxiv.org/abs/2604.23091)

    该研究通过对五个EEG预训练模型的系统评测发现，通道适配方法的选择取决于具体架构而非普适规律——在大多数模型-任务组合中简单的固定通道映射就已足够，只有刚性固定输入的BENDR等少数模型才需要可学习的适配方法。

    

    扩展EEG基础模型需要汇集来自不同异构电极配置的数据，因此每个模型都需要一个通道适配步骤，将任意的源电极配置映射到其预期的输入格式上。然而，这一步骤是否必须通过学习获得，以及它是否会混淆跨模型的比较，此前尚未得到检验。我们在五个预训练模型（500万至1.57亿参数）、五个下游任务、两种训练模式以及多达15个随机种子上，比较了五种通道适配方法（可学习的Conv1d投影、球面样条插值、源空间分解、黎曼重定中心，以及各模型的原生处理方式）。结论是结果取决于架构，而非存在普适的适配器：在25个模型-任务组合中，有18个组合里固定或原生映射表现最佳，但特定训练模式下的优势分布在原生方法、Conv1d、球面样条插值（SSI）、OmnEEG和黎曼方法之间。采用刚性固定输入的BENDR是唯一主要需要学习式桥接的案例；表现最强的模型CBraMod则达到了其最佳结果……

    arXiv:2604.23091v2 Announce Type: replace  Abstract: Scaling EEG foundation models requires pooling data across heterogeneous montages, so every model needs a channel-adaptation step mapping an arbitrary source montage onto its expected input. Whether this step must be learned, and whether it confounds cross-model comparison, is untested. We compare five channel-adaptation routes (a learned Conv1d projection, spherical spline interpolation, source-space decomposition, Riemannian re-centering, and each model's native handling) across five pretrained models (5M--157M parameters), five downstream tasks, two training regimes, and up to 15 random seeds. The answer is architecture-dependent rather than adapter-universal: fixed or native mappings are best in 18 of 25 model--task cells, but regime-specific wins are distributed across Native/Conv1d/SSI/OmnEEG/Riemannian. Rigid fixed-input BENDR is the main case where a learned bridge is needed; the strongest model, CBraMod, reaches its best res
    
[^368]: 当先验出错时：探究逆问题中设定错误的扩散先验

    When the Prior Is Wrong: Probing Misspecified Diffusion Priors in Inverse Problems

    [https://arxiv.org/abs/2604.21066](https://arxiv.org/abs/2604.21066)

    该论文提出 PIVOT 方法，将扩散先验的错误设定建模为对代表科学合理偏差的潜在变量进行推断，从而能从单次观测中识别先验中哪些假设需要放宽以及放宽的程度。

    

    虽然扩散先验在许多科学逆问题中能够生成高质量的后验样本，但它们通常是在纯模拟数据上训练的，因此会继承这些底层模拟来源的误差与偏差。许多从单次观测中调整扩散先验的方法，需要数据与先验之间存在足够的冲突才能成功适应，且无法对特定的错误设定假设进行检验。另一方面，现有的分布外（OOD）检测方法仅能判断观测结果是否与固定先验兼容，而无法指明应放宽哪些假设或放宽到何种程度。为解决这些问题，我们提出了 PIVOT（Prior Inference Via Observation-driven Transformations，基于观测驱动变换的先验推断），该方法将扩散先验的错误设定表述为对一个潜在变量的推断，该潜在变量代表一种科学上合理的偏差。通过沿这些合理的偏差方向扩展先验……

    arXiv:2604.21066v3 Announce Type: replace-cross  Abstract: While diffusion priors generate high-quality posterior samples across many scientific inverse problems, they are often trained on purely simulated data, thus inheriting the errors and biases of these underlying sources. Many approaches to adapting diffusion priors from a single observation require sufficient conflict between the data and the prior to successfully adapt and do not allow specific misspecification hypotheses to be tested. On the other hand, existing out-of-distribution (OOD) detection methods determine whether the observation is compatible with a fixed prior, rather than specifying which assumptions to relax or by how much. To address these issues, we propose PIVOT: Prior Inference Via Observation-driven Transformations, which formulates diffusion prior misspecification as inference over an underlying latent variable representing a scientifically plausible discrepancy. By expanding the prior along these plausible 
    
[^369]: NSFL：一种面向神经嵌入中布尔运算符的训练后神经符号模糊逻辑框架

    NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings

    [https://arxiv.org/abs/2604.10604](https://arxiv.org/abs/2604.10604)

    提出免训练的神经符号模糊逻辑框架 NSFL，借助 NS-Delta 边际语义贡献与黎曼优化查询投影，使稠密检索器无需训练即可原生处理多原子布尔逻辑约束，检索性能最高提升 81%。

    

    标准稠密检索器缺乏处理多原子逻辑约束的原生机制。我们提出了神经符号模糊逻辑框架，这是一种免训练框架，通过对相似度进行一阶有限差分处理，将模糊逻辑结构引入神经嵌入空间。NSFL 将逻辑运算锚定在零阶原子分数上，同时通过神经符号增量进行引导——即由上下文融合所诱导的边际语义贡献——在捕获上下文依赖性的同时保持原子语义，并避免了几何基线方法中普遍存在的表示坍缩和流形逃逸问题。针对延迟受限的部署场景，球形查询优化（SQO）利用黎曼优化将逻辑公式投影为流形稳定的查询向量。在六种编码器配置和两种模态上，NSFL 在 QUEST 数据集上实现了高达 +81% 的 mAP@100 提升，在 COCO 衍生的文本到图像检索任务上实现了 +70% 的提升。

    arXiv:2604.10604v2 Announce Type: replace-cross  Abstract: Standard dense retrievers have no native mechanism for multi-atom logical constraints. We introduce Neuro-Symbolic Fuzzy Logic (NSFL), a training-free framework that brings fuzzy-logical structure to neural embedding spaces through a first-order finite-difference treatment of similarity. NSFL anchors logical operations on zero-order atomic scores while steering with Neuro-Symbolic Deltas (NS-Delta) -- the marginal semantic contributions induced by contextual fusion -- preserving atomic meaning while capturing contextual dependence, and avoiding the representation collapse and manifold escape endemic to geometric baselines. For latency-constrained deployment, Spherical Query Optimization (SQO) uses Riemannian optimization to project logical formulas into manifold-stable query vectors. Across six encoder configurations and two modalities, NSFL yields mAP@100 improvements up to +81% on QUEST and +70% on a COCO-derived text-to-imag
    
[^370]: 基于人类反馈的强化学习：统计视角

    Reinforcement Learning from Human Feedback: A Statistical Perspective

    [https://arxiv.org/abs/2604.02507](https://arxiv.org/abs/2604.02507)

    本综述首次系统地从统计学视角梳理了RLHF框架，将其核心组成部分（监督微调、奖励建模与策略优化）与BTL模型、潜效用估计、主动学习、实验设计和不确定性量化等经典统计概念联系起来。

    

    基于人类反馈的强化学习（RLHF）已成为将大语言模型（LLM）与人类偏好对齐的核心框架。尽管在实践中取得了成功，RLHF由于依赖嘈杂、主观且异构的反馈来学习奖励模型并优化策略，因而引发了一些根本性的统计学问题。本综述从统计学的视角审视RLHF，主要聚焦于大语言模型的对齐。我们介绍了RLHF的主要组成部分，包括监督微调、奖励建模和策略优化，并将它们与人们熟知的统计概念联系起来，如Bradley-Terry-Luce（BTL）模型、潜效用估计、主动学习、实验设计和不确定性量化。我们回顾了从成对偏好数据中学习奖励函数的方法，以及通过两阶段RLHF流程和新兴的一阶段方法（如直接偏好优化）来优化策略的相关方法。

    arXiv:2604.02507v2 Announce Type: replace-cross  Abstract: Reinforcement learning from human feedback (RLHF) has emerged as a central framework for aligning large language models (LLMs) with human preferences. Despite its practical success, RLHF raises fundamental statistical questions because it relies on noisy, subjective, and heterogeneous feedback to learn reward models and optimize policies. This survey provides a statistical perspective on RLHF, focusing primarily on LLM alignment. We introduce the main components of RLHF, including supervised fine-tuning, reward modeling, and policy optimization, and relate them to familiar statistical concepts such as the Bradley-Terry-Luce (BTL) model, latent utility estimation, active learning, experimental design, and uncertainty quantification. We review methods for learning reward functions from pairwise preference data and for optimizing policies through both two-stage RLHF pipelines and emerging one-stage approaches such as direct prefer
    
[^371]: 受生物学启发的语言模型适配中参数选择与更新强度的协同调控

    Biologically Inspired Coordination of Parameter Selection and Update Strength for Language Model Adaptation

    [https://arxiv.org/abs/2604.00533](https://arxiv.org/abs/2604.00533)

    该论文提出了受生物学启发的SyCo框架，通过协调参数选择（基于Rac1遗忘机制保留选定的源低秩分量）与更新强度调节（基于MAPK信号机制结合多个预测指标），在不使用目标标签的情况下实现语言模型向目标任务的有效适配。

    

    将在源任务上训练的语言模型适配到目标任务时，需要在学习信号不确定的情况下决定更新哪些参数以及更新的强度。我们提出了突触巩固框架，这是一个受生物学启发的框架，用于协调这两种控制机制。与Rac1相关的遗忘和记忆痕迹重分布机制启发了在目标适配时保留选定的、源自源任务的低秩分量，同时重置其他分量。与MAPK相关的信号处理机制启发了随时间结合多个基于预测的指标来调节更新强度。SyCo将这些控制机制与结构化一致性目标以及一个小型固定的带标签源样本缓冲区相结合。在适配过程中不使用目标标签。我们在多源开放集适配（MOA）场景中评估该框架，使用源任务初始化的目标学习器来处理源训练中未出现的任务以及来自源任务的经扰动保留输入。（摘要原文在此处截断）

    arXiv:2604.00533v3 Announce Type: replace  Abstract: Adapting a source-trained language model to a target task requires deciding which parameters to update and how strongly when learning signals are uncertain. We introduce Synapse Consolidation (SyCo), a biologically inspired framework that coordinates these two controls. Rac1-associated forgetting and memory-trace redistribution motivate retaining selected source-derived low-rank components while resetting others for target adaptation. MAPK-associated signal processing motivates combining multiple prediction-based indicators over time to regulate update strength. SyCo combines these controls with a structured consistency objective and a small, fixed buffer of labelled source examples. Target labels are not used during adaptation. We evaluate the framework in Multi-source Open-set Adaptation (MOA), using source-initialized target learners for tasks absent from source training and for perturbed held-out inputs from source tasks. Across 
    
[^372]: 视觉STEM推理中的跨语言差距：跨越六种印度语言审计视觉语言模型

    Cross-Lingual Gaps in Visual STEM Reasoning: Auditing VLMs Across Six Indian Languages

    [https://arxiv.org/abs/2603.26742](https://arxiv.org/abs/2603.26742)

    该论文构建了六种印度语言的视觉STEM推理审计基准，发现视觉语言模型从英语切换到印度语言时准确率下降9.8至24.9个百分点，且达罗毗荼语系语言的性能下降比印度-雅利安语系更严重。

    

    视觉语言模型（VLM）在基于图表的科学技术工程数学（STEM）推理方面的基准测试大多以英语进行，因此当同一问题以另一种语言提出时，其准确率会如何变化尚不清楚。我们对六种印度语言进行了审计：将来自MathVista、ScienceQA和MMMU的980道题目通过IndicTrans2机器翻译成印地语、孟加拉语、马拉地语、泰米尔语、泰卢固语和卡纳达语，同时保持图像不变。我们评估了七个开源权重的VLM（Qwen2.5-VL-7B/32B、Qwen3-VL-30B、InternVL2.5-8B、Aya-Vision-8B、Gemma 3-27B、Llama-4-Maverick）以及GPT-4o，并在Qwen2.5-VL-7B上进行了无图像和思维链（CoT）消融实验（共计68,600条推理记录）。结果显示，从英语到印度语的平均准确率下降了9.8至24.9个百分点，且每个模型在达罗毗荼语系语言上的损失都比印度-雅利安语系语言更大（差距为0.1至13.2个百分点）。对于使用本文提示词的Qwen2.5-VL-7B，思维链降低了孟加拉语……

    arXiv:2603.26742v2 Announce Type: replace  Abstract: Vision-language models (VLMs) are benchmarked on diagram-grounded science, technology, engineering, and mathematics (STEM) reasoning mostly in English, so it is unclear how much their accuracy changes when the same problem is posed in another language. We audit this for six Indian languages: 980 questions from MathVista, ScienceQA, and MMMU are machine-translated into Hindi, Bengali, Marathi, Tamil, Telugu, and Kannada with IndicTrans2 while the images stay unchanged. We evaluate seven open-weight VLMs (Qwen2.5-VL-7B/32B, Qwen3-VL-30B, InternVL2.5-8B, Aya-Vision-8B, Gemma 3-27B, Llama-4-Maverick) and GPT-4o, plus no-image and chain-of-thought (CoT) ablations on Qwen2.5-VL-7B (68,600 inference records). Accuracy falls 9.8-24.9 percentage points (pp) from English to the Indian-language mean, and every model loses more on the Dravidian than on the Indo-Aryan languages (by 0.1-13.2 pp). For Qwen2.5-VL-7B with our prompt, CoT lowers Benga
    
[^373]: 面向中国ETF市场资产配置的基于大语言模型投资顾问智能体基准测试：NLPCC 2026 共享任务 4 概述

    Benchmarking LLM-Based Investment Advisor Agents for Asset Allocation in the Chinese ETF Market: Overview of NLPCC 2026 Shared Task 4

    [https://arxiv.org/abs/2603.22305](https://arxiv.org/abs/2603.22305)

    本文推出了 CN-Buzz2Portfolio——NLPCC 2026 共享任务 4 的官方基准，首次为中国 ETF 市场提供了评估大语言模型投资智能体进行新闻筛选、新闻映射与风险控制型资产配置的标准化端到端测试平台。

    

    大语言模型能够将非结构化的金融新闻转化为可执行的投资交易策略，然而针对中国市场的标准化端到端基准测试仍然稀缺。我们选取常见的中国交易型开放式指数基金（ETF）作为投资标的，并推出了 CN-Buzz2Portfolio——NLPCC 2026 共享任务 4 的官方基准测试。投资智能体以每日热门中文金融新闻和 ETF 历史价量序列作为输入，完成新闻筛选、新闻到 ETF 的映射以及风险可控的投资组合构建。为便于公平比较，我们提供了一个兼容主流大语言模型、可复现的“压缩—感知—配置”基线方案。该任务包含两条 ETF 配置赛道：宏观资产配置与行业轮动配置。我们采用 2024 至 2026 年的时序数据划分方式以缓解过拟合问题，最终排名由模型在未见过的测试数据上的样本外夏普比率决定。

    arXiv:2603.22305v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) convert unstructured financial news into actionable trading strategies, yet standardized end-to-end benchmarks tailored for Chinese market remain scarce. We select common Chinese Exchange-Traded Funds (ETFs) as the investment universe and introduce CN-Buzz2Portfolio, the official benchmark of NLPCC 2026 Shared Task 4. Investment agents take daily trending Chinese financial news and historical ETF price-volume series as inputs to complete news filtering, news-to-ETF mapping and risk-controlled portfolio construction. To facilitate fair comparison, we provide a reproducible Compression--Perception--Allocation baseline compatible with mainstream LLMs. The task contains two ETF allocation tracks: Macro-Asset Allocation and Sector-Rotation Allocation. We adopt chronological data split across 2024--2026 to alleviate overfitting, with final rankings determined by out-of-sample Sharpe ratios on unseen test 
    
[^374]: 更深入而非更长：使用深度循环Transformer实现内存高效的测试时推理以获得组合泛化能力

    Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization

    [https://arxiv.org/abs/2603.21676](https://arxiv.org/abs/2603.21676)

    本文提出一种深度循环Transformer，通过迭代共享权重模块在不生成思维链token的情况下实现更深的测试时推理，以恒定内存和线性延迟达成内存高效推理，并在组合泛化任务上验证了其有效性。

    

    标准Transformer具有固定的计算深度，限制了其泛化到需要可变深度推理任务的能力。通常的补救方法——思维链，通过耗费token进行推理，导致键值缓存膨胀，并使延迟随步数增长，因此在服务大批量查询时，内存成为限制性成本。我们研究了一种深度循环Transformer，它通过迭代一个共享权重的模块将计算深度与参数量解耦，使得每增加一个推理步骤只需恒定的内存开销和线性增长的延迟，且无需生成token。三个关键要素保证了循环在超过20个思考步骤中保持稳定：一个仅监督最终输出的“静默思考”目标、LayerScale初始化，以及一个带恒等偏置的门控机制，该门控在各步骤之间开辟了一条梯度高速通道。我们在三个结构偏置递减的组合域上对其进行了评估：图可达性（邻……

    arXiv:2603.21676v2 Announce Type: replace-cross  Abstract: Standard Transformers have a fixed computational depth, limiting their ability to generalize to tasks that require variable-depth reasoning. The usual remedy, Chain-of-Thought (CoT), spends tokens to reason, inflating the key--value cache and making latency grow with the step count, so memory becomes the limiting cost when reasoning is served over large query batches. We study a depth-recurrent Transformer that decouples computational depth from parameter count by iterating a shared-weight block, so that each added reasoning step costs flat memory and linear latency, with no token generation. Three ingredients keep the recurrence stable for 20+ thinking steps: a silent thinking objective that supervises only the final output, LayerScale initialization, and an identity-biased gate that opens a gradient highway across steps. We characterize it on three compositional domains with decreasing structural bias: graph reachability (adj
    
[^375]: 预测编码实现为硬件基底，其梯度计算速度超过反向传播

    Predictive Coding Implemented as a Hardware Substrate Computes Gradients faster than Backpropagation

    [https://arxiv.org/abs/2603.18066](https://arxiv.org/abs/2603.18066)

    本文将预测编码算法实例化为一种每个神经元对应一个物理核心的数字硬件基底，通过局部存储、通信与并发更新，使宽度为 n 的密集层的网络更新仅需 O(n) 个时钟周期，远快于反向传播串行处理所需的 O(n²) 次连接操作。

    

    反向传播促成了现代深度学习，但由于其全局误差传播、分离的训练阶段和集中式存储，难以实现为一个在线的、完全分布式的硬件系统。预测编码是一种与反向传播具有相似动力学的替代性学习算法，它源于局部预测误差动力学。本文表明，预测编码可以被实例化为一种离散时间数字硬件基底，具有局部存储和计算能力，无需指令集架构，且每个预测编码神经元对应一个物理核心。每个核心存储自身的活动和权重，与其所连接的核心进行通信，并执行局部梯度更新。通过并发更新神经元核心，该架构在 O(J+K) 个时钟周期内即可完成一次网络更新。对于宽度为 n 的密集层，每次更新仅需 O(n) 个周期，而相比之下，串行处理连接操作需要 O(n²) 次。

    arXiv:2603.18066v3 Announce Type: replace-cross  Abstract: Backpropagation has enabled modern deep learning but is difficult to realize as an online, fully distributed hardware system due to global error propagation, separate phases, and centralized memory. Predictive coding is an alternative learning algorithm with similar dynamics to backpropagation that arises from local prediction error dynamics. This paper shows that predictive coding can be instantiated as a discrete time digital hardware substrate, with local memory and computation, no instruction set architecture, and one physical core for each predictive-coding neuron. Each core stores its own activity and weights, communicates with the cores that it's connected to, and performs local gradient updates. By updating neuron cores concurrently, the architecture computes a network update in O(J+K) clock cycles. For dense layers of width n, this is O(n) cycles per update, compared with O(n^2) connection operations processed serially
    
[^376]: 流形匹配自编码器：将数据嵌入到选定的几何结构中

    Manifold-Matching Autoencoders: Embedding Data into Chosen Geometric Structures

    [https://arxiv.org/abs/2603.16568](https://arxiv.org/abs/2603.16568)

    提出一种名为“流形匹配”的简单正则化方法，通过最小化潜空间与任意参考表示之间成对距离的均方误差，使自编码器的潜空间能够被塑造成任意选定的几何结构，且无需标签、参考维数与潜空间维数自由解耦。

    

    操纵自编码器潜空间中数据的结构是表示学习领域的一个长期目标。我们提出了一种简单的正则化方法——流形匹配，它能够在潜空间上施加显式的几何结构。其核心思想是选取数据的一个参考表示R（它不必是输入网络的原始输入X），然后按批次最小化R中的成对距离与潜空间Z中的成对距离之间的均方误差。尽管自编码器仍然学习重建X，但其潜空间的形状会逐渐模仿R的形状。该目标不需要标签，只需要参考距离；而且由于这些距离是标量，R的维数与Z的维数相互解耦：任何参考都可以引导任何瓶颈层。利用这一单一工具，我们研究了如何在潜空间上施加不同的几何结构，并与基于几何或t…（方法）构建的自编码器进行了比较。

    arXiv:2603.16568v2 Announce Type: replace-cross  Abstract: Manipulating the structure of data in the latent space of an autoencoder is a long-standing goal of representation learning. We propose a simple regularization, Manifold Matching, that imposes an explicit geometric structure on that latent space. The idea is to take a reference representation R of the data, which need not be the raw input X fed to the network, and to minimize, batch-wise, the mean squared error between the pairwise distances in R and those in the latent space Z. Although the autoencoder still learns to reconstruct X, the shape of its latent space comes to mimic that of R. The objective requires no labels, only reference distances, and because those are scalars, the dimensionality of R is decoupled from that of Z: any reference can guide any bottleneck. Using this single instrument we study how different geometric structures can be imposed on a latent space, comparing against autoencoders built on geometric or t
    
[^377]: FairMed-XGB：一个基于贝叶斯优化的多指标可解释性框架，用于危重医疗数据中的人口统计公平性

    FairMed-XGB: A Bayesian-Optimised Multi-Metric Framework with Explainability for Demographic Equity in Critical Healthcare Data

    [https://arxiv.org/abs/2603.14947](https://arxiv.org/abs/2603.14947)

    该论文提出FairMed-XGB框架，通过贝叶斯优化将统计均等差异、泰尔指数和Wasserstein距离三种公平性指标整合为综合损失函数并融入XGBoost分类器，在保持预测性能的同时，利用SHAP可解释性分析有效减轻了急诊医疗机器学习模型中的性别等人口统计学偏差。

    

    背景：急诊医学中的机器学习模型常常表现出人口统计学偏差，特别是性别差异，这可能损害公平性、临床信任和决策可靠性。现有的缓解方法通常只优化孤立的公平性指标，无法在保持预测性能和可解释性的同时解决偏差的多维性质。方法：我们提出FairMed-XGB，一个集成到XGBoost分类器中的贝叶斯优化的多指标公平性框架。该模型引入了一个综合的公平性感知损失函数，结合了统计均等差异、泰尔指数和Wasserstein距离。贝叶斯优化被用来平衡公平性和预测准确性。该框架在源自MIMIC-IV-ED和eICU数据集的七个队列上进行评估。采用基于SHAP的可解释性方法来分析特征贡献并识别与偏差相关的模式。

    arXiv:2603.14947v2 Announce Type: replace-cross  Abstract: Background:Machine learning models in emergency medicine often exhibit demographic biases, particularly gender disparities, which can compromise fairness, clinical trust, and decision-making reliability. Existing mitigation approaches typically optimise isolated fairness metrics, failing to address the multi-dimensional nature of bias while maintaining predictive performance and interpretability. Methods: We propose FairMed-XGB, a Bayesian-optimised, multi-metric fairness framework integrated into an XGBoost classifier. The model incorporates a composite fairness-aware loss function combining Statistical Parity Difference, Theil Index, and Wasserstein Distance. Bayesian optimisation is used to balance fairness and predictive accuracy. The framework is evaluated on seven cohorts derived from the MIMIC-IV-ED and eICU datasets. SHAP-based explainability is employed to analyse feature contributions and identify bias-related pattern
    
[^378]: SVDP：用于快速大语言模型推理的无训练上下文稀疏性预测器

    SVDP: Training-Free Contextual Sparsity Predictors for Fast LLM Inference

    [https://arxiv.org/abs/2603.14110](https://arxiv.org/abs/2603.14110)

    该论文提出SVDP框架，通过对门控投影矩阵进行截断感知SVD分解并结合阈值校准算法，无需训练即可构建稀疏模式预测器，在精度损失低于1%的情况下实现LLM前馈网络推理最高1.8倍的加速。

    

    上下文稀疏性是降低大语言模型（LLM）推理过程计算复杂度的方法之一。现有基于上下文稀疏性的高效LLM推理加速技术（在精度损失极小的前提下）需要训练稀疏模式预测器。本文提出了一个用于加速LLM中基于ReGLU的前馈网络（FFN）推理的框架。该框架提供了一种快速、无需训练的稀疏模式预测器构建方法，该方法利用门控投影矩阵的截断感知奇异值分解（SVD），配合阈值校准算法，以及支持在CUDA和CANN设备上进行条件计算的推理执行器。在三个FFN平均激活稀疏度为90%的稀疏LLM上的实验表明，端到端解码时间最多可降低至原来的1/1.8，同时精度损失保持在1%以内。

    arXiv:2603.14110v2 Announce Type: replace  Abstract: Contextual sparsity is one of the approaches used to reduce computational complexity in the inference process of large language models (LLMs). Existing techniques for efficient LLM inference acceleration based on contextual sparsity with minimal accuracy degradation require training sparse pattern predictors. This paper presents a framework for accelerating inference of ReGLU-based feed-forward networks (FFNs) within LLMs. The proposed framework provides a fast, training-free method for building sparse pattern predictors using truncation-aware singular value decomposition (SVD) of the gate projection matrix, along with a threshold calibration algorithm, and inference executors supporting conditional computation on CUDA and CANN devices. Experiments on three sparse LLMs with an average activation sparsity level of 90% in the FFNs demonstrate up to a 1.8x reduction in end-to-end decoding time while maintaining less than 1% degradation 
    
[^379]: 面向异动力学离线强化学习的局部动力学感知域适应

    Localized Dynamics-Aware Domain Adaptation for Off-Dynamics Offline Reinforcement Learning

    [https://arxiv.org/abs/2602.21072](https://arxiv.org/abs/2602.21072)

    提出LoDADA方法，通过聚类源与目标数据转移并借助域判别估计簇级动力学差异，优先保留差异较小簇中的源数据，实现细粒度、可扩展的数据选择，以更高效地复用源数据解决异动力学离线强化学习问题。

    

    异动力学离线强化学习旨在利用有限的目标域数据以及在不同转移动力学下收集的大量源数据，为目标域学习策略。现有方法通常在状态空间上全局地处理动力学失配，或通过逐点数据过滤的方式来处理；这些方法可能遗漏局部化的跨域相似性，或带来较高的计算成本。我们提出局部动力学感知域适应，利用局部化的动力学失配来更好地复用源数据。LoDADA对来自源数据集和目标数据集的转移进行聚类，并通过域判别来估计簇级别的动力学差异。与来自差异较大簇的源转移相比，来自差异较小簇的源转移更可能被保留。这带来了一种细粒度且可扩展的数据选择策略，避免了过于粗糙的全局假设和……

    arXiv:2602.21072v2 Announce Type: replace-cross  Abstract: Off-dynamics offline reinforcement learning (RL) aims to learn a policy for a target domain using limited target data and abundant source data collected under different transition dynamics. Existing methods typically address dynamics mismatch either globally over the state space or via pointwise data filtering; these approaches can miss localized cross-domain similarities or incur high computational cost. We propose Localized Dynamics-Aware Domain Adaptation (LoDADA), which exploits localized dynamics mismatch to better reuse source data. LoDADA clusters transitions from source and target datasets and estimates cluster-level dynamics discrepancy via domain discrimination. Source transitions from clusters with small discrepancy are more likely to be retained compared with those from clusters with large discrepancy. This yields a fine-grained and scalable data selection strategy that avoids overly coarse global assumptions and ex
    
[^380]: 多模态大语言模型在简单视觉规划任务中推理的分布内与分布外泛化研究

    In- and Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks

    [https://arxiv.org/abs/2602.15460](https://arxiv.org/abs/2602.15460)

    该论文提出了一个评估框架，系统检验多模态大语言模型的思维链推理在网格导航任务中的分布内与分布外泛化能力，发现输入地图格式与思维链格式对分布外泛化影响显著，且基于坐标的推理轨迹表现出令人惊讶的泛化效果。

    

    将推理能力整合到（多模态）大语言模型中，近期已显著提升了其能力。然而，推理模型的泛化仍然定义模糊且缺乏深入理解。在这项工作中，我们提出了一个评估框架，以严格检验思维链方法在简单视觉规划中的泛化能力，具体针对基于网格的导航任务。该任务及其数据的多样性使我们能够使用不同的输入表示（视觉和文本）以及思维链推理策略来微调模型变体，并在分布内（ID）和分布外（OOD）测试条件下对其进行系统性评估。我们的实验表明，分布外泛化（例如，推广到更大的地图）在很大程度上受到输入地图和思维链所用格式的影响。令人惊讶的是，我们发现结合文本描述的基于坐标的推理轨迹……

    arXiv:2602.15460v2 Announce Type: replace  Abstract: Integrating reasoning in (multimodal) large language models has recently led to significant improvement of their capabilities. However, generalization in reasoning models is still vaguely defined and poorly understood. In this work, we present an evaluation framework to rigorously examine how well chain-of-thought (CoT) approaches generalize in simple visual planning, specifically on a grid-based navigation task. The versatility of the task and its data allows us to fine-tune model variants using different input representations (visual and textual) and CoT reasoning strategies, and systematically evaluate them under both in-distribution (ID) and out-of-distribution (OOD) test conditions. Our experiments show that the out-of-distribution generalization (e.g., to larger maps) is largely impacted by the format used for input maps and CoT chains. Surprisingly, we find that coordinate-based reasoning traces which combine text descriptions
    
[^381]: 基于可解释机器学习的跑者膝部损伤检测

    Explainable Machine-Learning based Detection of Knee Injuries in Runners

    [https://arxiv.org/abs/2602.11668](https://arxiv.org/abs/2602.11668)

    本研究利用可解释的机器学习方法分析光学动作捕捉的跑步支撑相数据，实现了对跑者膝部损伤（包括髌股疼痛综合征和髂胫束综合征）的有效检测与分类。

    

    跑步是一项广泛开展的运动，但膝关节损伤的发生率较高，尤其是髌股疼痛综合征（PFPS）和髂胫束综合征（ITBS）。识别与这些损伤相关的步态模式可以改善临床决策，这需要能够捕捉和分析时间运动学数据的精确系统。本研究使用光学动作捕捉系统来增强对与损伤相关的跑步模式的检测。我们分析了一个包含839条来自健康和受伤跑者的跑步机记录的公开数据集，以评估这些系统捕捉与损伤分类相关的动态参数的有效性。研究重点在于支撑相，使用关节和节段角度时间序列以及离散点数值。研究解决了三个分类任务：健康与受伤、健康与PFPS、健康与ITBS。我们考察了不同的特征空间，从传统的基于点的指标到完整的支撑相特征（摘要原文在此处截断）。

    arXiv:2602.11668v2 Announce Type: replace  Abstract: Running is a widely practiced activity but shows a high incidence of knee injuries, especially Patellofemoral Pain Syndrome (PFPS) and Iliotibial Band Syndrome (ITBS). Identifying gait patterns linked to these injuries can improve clinical decision-making, which requires precise systems capable of capturing and analyzing temporal kinematic data.   This study uses optical motion capture systems to enhance detection of injury-related running patterns. We analyze a public dataset of 839 treadmill recordings from healthy and injured runners to evaluate how effectively these systems capture dynamic parameters relevant to injury classification. The focus is on the stance phase, using joint and segment angle time series and discrete point values.   Three classification tasks are addressed: healthy vs. injured, healthy vs. PFPS, and healthy vs. ITBS. We examine different feature spaces, from traditional point-based metrics to full stance-pha
    
[^382]: 心电图成像中反问题的基于学习的有限元正则化方法

    Learned Finite Element-based Regularization of the Inverse Problem in Electrocardiographic Imaging

    [https://arxiv.org/abs/2602.07466](https://arxiv.org/abs/2602.07466)

    该论文提出了一种将空间正则化与学习的时间Fields-of-Experts先验相结合的心电图成像时空正则化框架，并借助有限元离散化与可扩展优化算法，实现了优于传统手工时空方法的去噪和逆问题重建效果。

    

    心电图成像（ECGI）旨在通过体表电位无创地重建心脏电活动。然而，其对应的逆问题严重病态，需要鲁棒的正则化处理。尽管心脏动力学的时间结构具有重要的生理学意义，经典方法却主要依赖空间平滑，时间结构仍未被充分利用。我们提出了一种时空正则化框架，将空间正则化与学习得到的时间Fields-of-Experts（FoE）先验相结合，以捕捉复杂的时空激活模式。我们在非结构化心脏表面网格上推导了有限元离散化方法，证明了Mosco收敛性，并开发了一种能够处理FoE项的可扩展优化算法。在合成心外膜数据上的数值实验表明，与手工设计的时空方法相比，该方法在去噪和逆问题重建方面均表现出改进效果。

    arXiv:2602.07466v3 Announce Type: replace-cross  Abstract: Electrocardiographic imaging (ECGI) seeks to reconstruct cardiac electrical activity from body-surface potentials noninvasively. However, the associated inverse problem is severely ill-posed and requires robust regularization. While classical approaches primarily employ spatial smoothing, the temporal structure of cardiac dynamics remains underexploited despite its physiological relevance. We introduce a space-time regularization framework that couples spatial regularization with a learned temporal Fields-of-Experts (FoE) prior to capture complex spatiotemporal activation patterns. We derive a finite element discretization on unstructured cardiac surface meshes, prove Mosco-convergence, and develop a scalable optimization algorithm capable of handling the FoE term. Numerical experiments on synthetic epicardial data demonstrate improved denoising and inverse reconstructions compared to handcrafted spatiotemporal methods, yieldin
    
[^383]: BayesFlow 2：Python中的多后端摊销贝叶斯推断

    BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python

    [https://arxiv.org/abs/2602.07098](https://arxiv.org/abs/2602.07098)

    BayesFlow 2是一个支持多种深度学习后端的通用Python库，通过摊销贝叶斯推断实现对后验分布、似然和比率的快速估计，解决了传统贝叶斯方法在复杂模型和大数据集上的计算瓶颈。

    

    现代贝叶斯推断涉及多种计算方法的组合，用于在系统性工作流程中对概率模型进行估计、验证并从中得出结论。许多贝叶斯方法的一个共同特征是速度相对较慢，在将复杂模型拟合到大型数据集时，这往往成为难以逾越的障碍。摊销贝叶斯推断为解决贝叶斯方法的计算挑战提供了一条途径。ABI通过在模型模拟数据上训练神经网络，使用户能够快速推断任何模型隐含的量，例如点估计、似然或完整的后验分布。在这项工作中，我们介绍了用于通用ABI的Python库BayesFlow第2版。除了直接的后验、似然和比率估计外，该软件还支持多种流行的深度学习后端，并提供丰富的用于采样和密度估计的生成网络集合。

    arXiv:2602.07098v3 Announce Type: replace-cross  Abstract: Modern Bayesian inference involves a mixture of computational methods for estimating, validating, and drawing conclusions from probabilistic models as part of principled workflows. An overarching motif of many Bayesian methods is that they are relatively slow, which often becomes prohibitive when fitting complex models to large data sets. Amortized Bayesian inference (ABI) offers a path to solving the computational challenges of Bayes. ABI trains neural networks on model simulations, rewarding users with rapid inference of any model-implied quantity, such as point estimates, likelihoods, or full posterior distributions. In this work, we present the Python library BayesFlow, Version 2, for general-purpose ABI. Along with direct posterior, likelihood, and ratio estimation, the software includes support for multiple popular deep learning backends, a rich collection of generative networks for sampling and density estimation, comple
    
[^384]: TopoPrune：通过统一潜在空间拓扑实现鲁棒的数据剪枝

    TopoPrune: Robust Data Pruning via Unified Latent Space Topology

    [https://arxiv.org/abs/2602.02739](https://arxiv.org/abs/2602.02739)

    TopoPrune通过全局拓扑感知流形嵌入与可微持续同调局部优化相结合的双尺度拓扑方法，捕获数据的内在稳定结构，实现对潜在空间扰动鲁棒的数据剪枝，即使在高剪枝率和特征噪声下仍能保持高精度。

    

    几何数据剪枝方法虽然在利用预训练模型方面十分实用，但本质上是不稳定的。它们对外在几何的依赖使其对潜在空间扰动高度敏感，导致在跨架构迁移或存在特征噪声的情况下性能下降。我们提出了TopoPrune，一个通过利用拓扑来捕获数据稳定内在结构从而解决这一挑战的框架。TopoPrune在两个尺度上运行：（1）利用拓扑感知的流形近似来建立数据集的全局低维嵌入；随后（2）采用可微持续同调（persistent homology）对流形嵌入进行局部拓扑优化，依据结构复杂度对样本进行排序。我们证明了这种统一的双尺度拓扑方法能够确保高准确度和精确度，尤其是在大幅度的数据集剪枝率下表现尤为突出。

    arXiv:2602.02739v3 Announce Type: replace-cross  Abstract: Geometric data pruning methods, while practical for leveraging pretrained models, are fundamentally unstable. Their reliance on extrinsic geometry renders them highly sensitive to latent space perturbations, causing performance to degrade during cross-architecture transfer or in the presence of feature noise. We introduce TopoPrune, a framework which resolves this challenge by leveraging topology to capture the stable, intrinsic structure of data. TopoPrune operates at two scales, (1) utilizing a topology-aware manifold approximation to establish a global low-dimensional embedding of the dataset. Subsequently, (2) it employs differentiable persistent homology to perform a local topological optimization on the manifold embeddings, ranking samples by their structural complexity. We demonstrate that our unified dual-scale topological approach ensures high accuracy and precision, particularly at significant dataset pruning rates (e
    
[^385]: 面向入侵检测中物联网攻击数据生成的潜在扩散方法

    Latent Diffusion for Internet of Things Attack Data Generation in Intrusion Detection

    [https://arxiv.org/abs/2601.16976](https://arxiv.org/abs/2601.16976)

    该论文提出LaDIS框架，通过学习混合类型自编码器表示并在自适应连续潜在空间中执行扩散，为物联网入侵检测生成保真度高、保持多变量依赖且覆盖度好的攻击数据，尤其在代表性不足的攻击类型上优于对抗式、变分式和直接空间扩散等现有生成方法。

    

    随着物联网系统部署的不断增长，在异构且不平衡的网络流量环境下进行可靠入侵检测的需求日益增加。然而，具有代表性的恶意流量——尤其是代表性不足的攻击类型——仍然难以获取，而现有的生成方法无法同时稳定地保持分布保真度、多变量依赖关系和覆盖度。本工作提出了LaDIS，一个针对攻击数据专用的框架，该框架学习混合类型的自编码器潜在表示，并在自适应的连续潜在空间中执行扩散过程。研究采用互补的评估指标，包括边缘分布与多变量分布的保真度、依赖关系保持、真实与合成数据的可区分性、流形结构以及计算效率，将LaDIS与成熟的对抗式、变分式和直接空间扩散方法进行了比较，涵盖代表性的洪泛、僵尸网络、侦察和隐蔽攻击。LaDIS提供了compe...（原文摘要在此处截断）

    arXiv:2601.16976v2 Announce Type: replace  Abstract: The growing deployment of IoT systems has increased the need for reliable intrusion detection under heterogeneous and imbalanced network traffic. However, representative malicious traffic, particularly for underrepresented attacks, remains difficult to obtain, while existing generative approaches do not consistently preserve distributional fidelity, multivariate dependencies, and coverage simultaneously. This work proposes LaDIS, an attack-specific framework that learns a mixed-type autoencoder representation and performs diffusion in an adaptive continuous latent space. LaDIS was compared with established adversarial, variational, and direct-space diffusion approaches across representative Flooding, Botnet, Recon, and Stealthy attacks using complementary measures of marginal and multivariate fidelity, dependency preservation, real--synthetic distinguishability, manifold structure, and computational efficiency. LaDIS provided a compe
    
[^386]: 神经同构域：一种基于Transformer的代数数值嵌入方法

    Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding

    [https://arxiv.org/abs/2601.12095](https://arxiv.org/abs/2601.12095)

    该论文提出神经同构域（NIF），一种基于Transformer的数字嵌入方法，可证明地保留十进制分数的代数结构（与有理数域同构），从而解决神经网络处理极端数值时的溢出与不稳定问题。

    

    神经网络模型在处理非常大或非常小的数字时常常面临挑战，原因在于溢出、下溢以及输出变化不稳定等问题。为了缓解这些问题，我们提出使用嵌入向量来表示数字，而不是直接使用其原始数值。这些嵌入旨在保留关键的代数性质，同时避免数值不稳定性。我们提出了神经同构域，这是一种数字序列Transformer自编码器，通过两个学习得到的交换算子和一个反对称神经比较器进行训练。在零损失条件下，这些嵌入可被证明构成一个与十进制分数同构的有序整环，其分式域与有理数同构。对于训练后的模型，我们通过解码边际、缺陷和Lipschitz常数来界定运算、链式运算和代数法则的误差。在最多25位数字上训练后，NIF在封闭性上达到了至少99.9%的精确匹配率。

    arXiv:2601.12095v2 Announce Type: replace-cross  Abstract: Neural network models often face challenges when processing very small or very large numbers due to issues such as overflow, underflow, and unstable output variations. To mitigate these problems, we propose using embedding vectors for numbers instead of directly using their raw values. These embeddings aim to retain essential algebraic properties while preventing numerical instabilities. We introduce Neural Isomorphic Fields (NIF), a digit-sequence Transformer autoencoder trained with two learned commutative operators and an antisymmetric neural comparator. At zero loss, the embeddings provably form an ordered integral domain isomorphic to the decimal fractions, whose field of fractions is isomorphic to the rationals. For the trained model, we bound errors of operations, chains and algebraic laws by decoding margins, defects and Lipschitz constants. Trained on up to 25 digits, NIF reaches at least 99.9% exact match on closure, 
    
[^387]: 基于机制的智能：面向多智能体系统理性协调与均衡对齐的可微激励机制

    Mechanism-Based Intelligence: Differentiable Incentives for Rational Coordination and Equilibrium Alignment in Multi-Agent Systems

    [https://arxiv.org/abs/2512.20688](https://arxiv.org/abs/2512.20688)

    提出基于机制的智能（MBI）架构，通过可微价格机制将全局损失的精确梯度作为激励信号，使多智能体博弈成为以全局最优为唯一均衡的精确势博弈，从而同时解决哈耶克信息问题与赫维茨激励问题。

    

    自主多智能体系统一直受困于哈耶克信息问题（如何利用分散的私人知识）与赫维茨激励问题（如何将局部行动与全局目标对齐）。本文提出了基于机制的智能（MBI）架构，它将智能视为从专门化模块的协调中涌现而来，而非单一的整块模型。其核心是可微价格机制：将系统损失的精确梯度作为价格进行计算；每个智能体在减去自身的边际成本后，即可获得激励信号 $$\mathbf{G}_i = - \frac{\partial \mathcal{L}_{\mathrm{global}}}{\partial \mathbf{x}_i}$$，而转移支付则采用Groves形式。在给定其他智能体行动的情况下，每个智能体的最优响应均能最小化全局损失，因此负的全局损失构成了该博弈的精确势函数，其唯一均衡点恰好位于全局最优处。基于梯度的博弈行为将收敛至……（摘要原文在此处截断）

    arXiv:2512.20688v2 Announce Type: replace-cross  Abstract: Autonomous multi-agent systems struggle with the Hayekian Information problem (using dispersed private knowledge) and the Hurwiczian Incentive problem (aligning local actions with global objectives). I introduce Mechanism-Based Intelligence (MBI), an architecture that treats intelligence as emergent from the coordination of specialized modules rather than a single monolithic model. At its core, the Differentiable Price Mechanism computes the exact gradient of the system loss as a price; subtracting its own marginal cost, each agent obtains the incentive signal   $$\mathbf{G}_i = - \frac{\partial \mathcal{L}_{\mathrm{global}}}{\partial \mathbf{x}_i},$$   and the transfers take the Groves form. Each agent's best response minimizes the global loss given the actions of others, so the negative global loss is an exact potential for the resulting game, with a unique equilibrium at the global optimum. Gradient-based play converges to i
    
[^388]: CoDeQ：使用死区量化器实现可微联合剪枝与量化的模型压缩

    CoDeQ: Compression with a Dead-Zone Quantizer for Differentiable Joint Pruning--Quantization

    [https://arxiv.org/abs/2512.12981](https://arxiv.org/abs/2512.12981)

    该论文提出CoDeQ，通过可学习的死区量化器将剪枝与量化统一在同一训练框架中，以死区宽度控制逐层稀疏度并支持可学习的混合精度位宽，在单一全局压缩系数下联合优化权重与压缩参数，从而省去辅助分配流程和大量超参数调优。

    

    联合剪枝与量化可以降低神经网络的推理成本，但确定各层的稀疏度与精度通常需要辅助的分配流程以及大量的超参数调优。我们提出CoDeQ，这是一种量化感知训练方法，通过一个可学习的死区量化器将剪枝与量化统一起来。其学习到的死区宽度可以控制各层的稀疏度，而位宽既可以保持固定，也可以在混合精度模式下进行学习。一个剪枝感知的缩放因子会在保留的范围内分配所有可用的非零重建电平。我们推导了在固定位宽和范围端点条件下，能够最小化该范围内最坏情况重建误差的间距。模型权重与压缩参数在一个全局压缩系数下联合优化，压缩参数共享同一学习率。在VGG7/CIFAR-10上，使用2位激活的CoDeQ实现了0.05%的相对……

    arXiv:2512.12981v2 Announce Type: replace  Abstract: Joint pruning and quantization can reduce the inference cost of neural-networks, but determining layerwise sparsity and precision often requires auxiliary allocation procedures and considerable hyperparameter tuning. We introduce CoDeQ, a quantization-aware training method that unifies pruning and quantization through a learnable dead-zone quantizer. Its learned dead-zone width controls layerwise sparsity, while bit widths can be held fixed or learned in mixed-precision mode. A pruning-aware scale allocates all available nonzero reconstruction levels over the retained range. We derive the spacing that minimizes worst-case reconstruction error over this range for a fixed bit width and range endpoint. Model weights and compression parameters are optimized jointly under one global compression coefficient, with a shared learning rate for the compression parameters. On VGG7/CIFAR-10, CoDeQ with 2-bit activations achieves 0.05% relative BO
    
[^389]: Luxical：高速词汇密集型文本嵌入

    Luxical: High-Speed Lexical-Dense Text Embeddings

    [https://arxiv.org/abs/2512.09015](https://arxiv.org/abs/2512.09015)

    Luxical通过结合稀疏TF-IDF特征、小型ReLU网络和知识蒸馏训练，以极低的计算成本近似大型Transformer嵌入模型，实现了面向网络规模文本组织的高速“词汇密集”文本嵌入。

    

    arXiv:2512.09015v3 公告类型：替换 摘要：前沿语言模型的质量在很大程度上取决于我们组织网络规模文本语料库以用于训练的能力。当今的主流工具在速度与灵活性之间存在权衡：词汇分类器（如FastText）速度快但仅限于生成分类输出分数，而Transformer文本嵌入模型的向量输出能够灵活支持多种工作流程（如聚类、分类和检索），但其计算成本高昂。我们推出了Luxical，一个用于高速“词汇密集”文本嵌入的库，旨在为网络规模的文本组织任务同时兼顾两种方法的最佳特性。Luxical结合了稀疏TF-IDF特征、一个小型ReLU网络以及知识蒸馏训练方案，以仅为大模型一小部分的运行成本来近似大型Transformer嵌入模型。在本技术报告中，我们描述了Luxical的架构和训练目标，并评估了……

    arXiv:2512.09015v3 Announce Type: replace  Abstract: Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training. Today's dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce. We introduce Luxical, a library for high-speed "lexical-dense" text embeddings that aims to recover the best properties of both approaches for web-scale text organization. Luxical combines sparse TF--IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost. In this technical report, we describe the Luxical architecture and training objective and evaluate a
    
[^390]: 有限抗原标签下的学习：基于蛋白质语言模型的甲型流感抗原关系半监督预测

    Learning with Limited Antigenic Labels: Semi-supervised Prediction of Influenza A Antigenic Relationships with Protein Language Models

    [https://arxiv.org/abs/2512.05222](https://arxiv.org/abs/2512.05222)

    该论文的创新在于将预训练蛋白质语言模型表征（如ESM-2、ProtT5等）与半监督学习方法（自训练和标签传播）相结合，在标记数据稀缺的条件下预测甲型流感病毒的抗原关系，为减少昂贵的血凝抑制实验需求提供了新途径。

    

    甲型流感病毒（IAVs）经历频繁的抗原性变化，需要定期更新疫苗。由于血凝抑制（HI）实验是劳动密集型的，相对于可获得的序列数据，抗原性测量数据仍然有限。我们结合预训练蛋白质表征与半监督学习（SSL），在标记数据稀缺的情况下预测抗原关系。来自H1N1、H3N2、H5N1和H9N2的HA1序列使用ESM-2、ProtVec、ProtT5和ProtBert进行编码。基于这些嵌入，我们将自训练和标签传播与监督基线方法进行比较，包括随机森林（RF）和支持向量机（SVM）模型。嵌套交叉验证评估了在保留25%、50%、75%或100%训练标签时的性能。自训练在大多数设置中略微提高了RF的性能，但通常降低了SVM区分抗原相似与抗原不同病毒对的能力……

    arXiv:2512.05222v2 Announce Type: replace  Abstract: Influenza A viruses (IAVs) undergo frequent antigenic changes that require regular vaccine updates. Because haemagglutination inhibition (HI) assays are labour-intensive, antigenic measurements remain limited relative to the available sequence data. We combined pre-trained protein representations with semi-supervised learning (SSL) to predict antigenic relationships when labelled data are scarce. HA1 sequences from H1N1, H3N2, H5N1, and H9N2 were encoded using ESM-2, ProtVec, ProtT5, and ProtBert. Using these embeddings, we compared self-training and label spreading with supervised baselines, including random forest (RF) and support vector machine (SVM) models. Nested cross-validation evaluated performance with 25%, 50%, 75%, or 100% of training labels retained. Self-training slightly improved RF performance in most settings but generally reduced SVM's ability to distinguish antigenically similar from antigenically different virus pa
    
[^391]: 忒修斯之网络（如同那艘船）

    Network of Theseus (like the ship)

    [https://arxiv.org/abs/2512.04198](https://arxiv.org/abs/2512.04198)

    提出忒修斯之网络方法，通过逐步替换组件并用表征相似性度量对齐，在保持性能的前提下将训练好的网络架构渐进转换为完全不同的目标架构，打破了“训练架构即部署架构”的传统假设。

    

    深度学习中的一个标准假设是：神经网络架构所引入的归纳偏置必须从训练阶段一直延续到推理阶段，即你用什么架构训练，就用什么架构部署。这一假设使得研究者无法选择那些可能具有理想效率或设计特性、但存在优化困难的架构。我们提出“忒修斯之网络”来挑战这一假设，该方法能够将一个已训练（甚至未训练）的引导网络架构逐部分地渐进转换为完全不同的目标网络架构，同时保持引导网络的性能。在每个阶段，引导网络架构中的组件被逐步替换为目标架构的模块，并通过表征相似性度量进行对齐。即使在发生重大架构变化的情况下，该过程仍能保留引导网络的大部分功能……

    arXiv:2512.04198v2 Announce Type: replace-cross  Abstract: A standard assumption in deep learning is that the inductive bias introduced by a neural network architecture must persist from training through inference. The architecture you train with is the architecture you deploy. This assumption constrains the community from selecting architectures that may have desirable efficiency or design properties due to difficulties with optimization. We challenge this assumption with the Network of Theseus (NoT), a method for progressively converting a trained, or even untrained, guide network architecture part-by-part into an entirely different target network architecture while preserving the performance of the guide network. At each stage, components in the guide network architecture are incrementally replaced with the target architecture modules and aligned via representational similarity metrics. This procedure preserves most of the functionality of the guide network even under substantial ar
    
[^392]: ML-Tool-Bench：面向机器学习任务的工具增强规划

    ML-Tool-Bench: Tool-Augmented Planning for ML Tasks

    [https://arxiv.org/abs/2512.00672](https://arxiv.org/abs/2512.00672)

    提出了ML-Tool-Bench基准，利用61个专用工具和15个Kaggle表格机器学习挑战，评估工具增强型ML智能体进行复杂规划的能力。

    

    能够执行端到端数据科学工作流的自主机器学习（ML）智能体的开发，代表着人工智能领域的一个重要前沿。这些智能体必须协调数据分析、特征工程、模型选择和超参数优化等复杂序列任务，这些任务需要精细的规划与迭代。虽然近期构建ML智能体的研究探索了使用大语言模型（LLM）进行直接代码生成，但工具增强的方法提供了更好的模块化与可靠性。然而，现有的工具使用基准主要关注任务特定的工具选择或工具调用的参数提取，未能评估ML智能体所需的复杂规划能力。在这项工作中，我们引入了一个全面的基准，使用精心整理的61个专用工具和来自Kaggle的15个表格ML挑战来评估工具增强的ML智能体。（原文摘要在此处截断，后续内容缺失）

    arXiv:2512.00672v3 Announce Type: replace-cross  Abstract: The development of autonomous machine learning (ML) agents capable of end-to-end data science workflows represents a significant frontier in artificial intelligence. These agents must orchestrate complex sequences of data analysis, feature engineering, model selection, and hyperparameter optimization, tasks that require sophisticated planning and iteration. While recent work on building ML agents has explored using large language models (LLMs) for direct code generation, tool-augmented approaches offer greater modularity and reliability. However, existing tool-use benchmarks focus primarily on task-specific tool selection or argument extraction for tool invocation, failing to evaluate the sophisticated planning capabilities required for ML Agents. In this work, we introduce a comprehensive benchmark for evaluating tool-augmented ML agents using a curated set of 61 specialized tools and 15 tabular ML challenges from Kaggle. Our 
    
[^393]: db-SP：基于双重均衡序列并行加速视觉生成模型的稀疏注意力

    db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism

    [https://arxiv.org/abs/2511.23113](https://arxiv.org/abs/2511.23113)

    本文提出db-SP，一种感知稀疏性的双重均衡序列并行技术，通过双层划分在注意力头级别和块级别实现近乎完美的负载均衡，从而加速采用块级稀疏注意力的扩散Transformer视觉生成模型的推理。

    

    通过序列并行来扩展扩散Transformer（DiT）的推理，对于降低视觉生成的延迟至关重要，但当将其应用于采用块级稀疏注意力的模型时，会因负载不均衡而受到严重阻碍。这种不均衡源于各注意力头之间稀疏度的固有差异，以及稀疏掩码中稠密块的不规则分布——当序列并行沿头维度（如Ulysses）或块维度（如Ring Attention）应用时尤为明显。本文形式化定义了一个稀疏不均衡比率来量化这种不均衡，并提出了db-SP，一种感知稀疏性的序列并行技术来应对这一挑战。db-SP包含一种双层划分方法，能够以可忽略的开销在头级别和块级别上实现近乎完美的负载均衡。此外，为了处理去噪步骤和各层之间不断演变的稀疏模式，db-SP动态（原文摘要至此截断）

    arXiv:2511.23113v2 Announce Type: replace-cross  Abstract: Scaling Diffusion Transformer (DiT) inference via sequence parallelism is critical for reducing latency in visual generation, but is severely hampered by workload imbalance when applied to models employing block-wise sparse attention. The imbalance stems from the inherent variation in sparsity across attention heads and the irregular distribution of dense blocks within the sparse mask, when sequence parallelism is applied along the head dimension (as in Ulysses) or the block dimension (as in Ring Attention). In this paper, we formalize a sparse imbalance ratio to quantify the imbalance, and propose db-SP, a sparsity-aware sequence parallelism technique that tackles the challenge. db-SP contains a dual-level partitioning approach that achieves near-perfect workload balance at both the head and block levels with negligible overhead. Furthermore, to handle the evolving sparsity patterns across denoising steps and layers, db-SP dyn
    
[^394]: ELBO$_{\mathrm{TDS}}$：面向工业级推荐系统的时间分布偏移泛化

    ELBO$_{\mathrm{TDS}}$: Temporal Distribution Shift Generalization for Industry-Scale Recommender Systems

    [https://arxiv.org/abs/2511.21032](https://arxiv.org/abs/2511.21032)

    该论文提出了 ELBO$_{\mathrm{TDS}}$ 概率框架，通过识别时间分布偏移的关键因素、设计模拟时变因素的数据增强策略以及自监督变分目标，解决工业级推荐系统因时间分布偏移导致的泛化能力下降问题。

    

    时间分布偏移（TDS）会降低推荐系统的泛化能力：在历史交互数据上训练的模型，在未来的时间窗口中会面临系统性不同的用户行为。虽然基于近期数据的增量训练在工业生产中被广泛使用，但它可能过度适应瞬态模式并加剧不稳定性。现有方法（如不变学习和自监督学习）提供了部分解决方案，但往往存在时间泛化不稳定、表征崩溃或数据利用效率低下等问题。为了解决这些局限性，我们提出了 ELBO$_{\mathrm{TDS}}$，一个面向时间鲁棒推荐的概率框架。首先，我们识别了真实生产数据中的关键偏移因素，并设计了一种简单而有效的数据增强策略，该策略能够模拟真实的时间变化因素并扩展训练数据的支持范围。其次，我们推导了一个自监督变分目标（原文摘要在此处截断）。

    arXiv:2511.21032v2 Announce Type: replace  Abstract: Temporal distribution shift (TDS) degrades the generalization of recommender systems: models trained on past interactions face systematically different user behavior in future time windows. While incremental training on recent data is widely used in industrial production, it can over-adapt to transient patterns and exacerbate instability. Existing approaches such as invariant learning and self-supervised learning offer partial solutions but often suffer from unstable temporal generalization, representation collapse, or inefficient data utilization. To address these limitations, we propose ELBO$_{\mathrm{TDS}}$, a probabilistic framework for temporally robust recommendation. First, we identify key shifting factors in real-world production data and design a simple yet effective data augmentation strategy that simulates realistic time-varying factors and expands the training support. Second, we derive a self-supervised variational objec
    
[^395]: StochEP：面向脉冲收敛循环神经网络的随机平衡传播

    StochEP: Stochastic Equilibrium Propagation for Spiking Convergent Recurrent Neural Networks

    [https://arxiv.org/abs/2511.11320](https://arxiv.org/abs/2511.11320)

    本文提出StochEP框架，通过将概率性脉冲神经元融入平衡传播范式，平滑了优化景观并稳定了训练，实现了脉冲神经网络高效且可扩展的学习，同时提供了理论保证。

    

    脉冲神经网络有望实现节能、稀疏且受生物启发的计算。使用时间反向传播（BPTT）和替代梯度来训练它们可以取得强大的性能，但在生物学上并不合理。平衡传播（EP）提供了一种更具局部性且更符合生物学原理的替代方案。然而，现有的针对SNN的EP框架主要依赖确定性神经元，这需要复杂的机制来处理脉冲的不连续性，并且无法扩展到MNIST和Fashion-MNIST等简单基准之外。受生物脉冲机制的随机特性以及近期硬件发展趋势的启发，我们提出了一种随机EP框架，将概率性脉冲神经元整合到EP范式中。这种表述平滑了优化景观，稳定了训练过程，并使SNN中的学习变得高效且可扩展。我们提供了理论保证，表明该方法...

    arXiv:2511.11320v2 Announce Type: replace-cross  Abstract: Spiking Neural Networks (SNNs) promise energy-efficient, sparse, biologically inspired computation. Training them with Backpropagation Through Time (BPTT) and surrogate gradients achieves strong performance but remains biologically implausible. Equilibrium Propagation (EP) provides a more local and biologically grounded alternative. However, existing EP frameworks for SNNs largely rely on deterministic neurons, which require complex mechanisms to handle spiking discontinuities and do not scale beyond simple benchmarks such as MNIST and Fashion-MNIST. Inspired by the stochastic nature of biological spiking mechanism and recent hardware trends, we propose a stochastic EP framework that integrates probabilistic spiking neurons into the EP paradigm. This formulation smoothens the optimization landscape, stabilizes training, and enables efficient and scalable learning in SNNs. We provide theoretical guarantees showing that the propo
    
[^396]: 信度超空间上的不动点与包络动力学

    Fixed Points and Envelope Dynamics on Credal Hyperspaces

    [https://arxiv.org/abs/2510.04769](https://arxiv.org/abs/2510.04769)

    本文证明了信度集超空间中包含单调且豪斯多夫连续的自映射在完备格的信度序区间上存在可由迭代求得的最小与最大不动点，并为随时间变化的更新族构造了界定所有容许轨迹的上下包络动力学。

    

    设 $X$ 为紧度量空间，考虑由 $X$ 上博雷尔概率测度的非空弱闭凸集构成的超空间 $\mathscr C_X$，称为信度集。这些集合表示关于哪种概率规律支配所研究现象的不确定性。我们用由弱收敛度量诱导的豪斯多夫度量来装备 $\mathscr C_X$。信度序区间 $[\mathcal{A},\mathcal{B}] = \{ \mathcal{P}\in\mathscr C_X: \mathcal{A}\subseteq\mathcal{P}\subseteq\mathcal{B} \}$ 是完备格，且此类区间上的每个包含单调、豪斯多夫连续的自映射都具有最小和最大不动点，它们可分别作为从 $\mathcal{A}$ 和 $\mathcal{B}$ 出发迭代所得的极限而得到。对于可能随时间变化的一族包含单调更新，我们构造了下包络映射和上包络映射，用以界定每条容许轨迹。当这些包络映射为豪斯多夫连续时，每条轨迹与……的距离（摘要在此处被截断）

    arXiv:2510.04769v3 Announce Type: replace-cross  Abstract: Let $X$ be a compact metric space and consider the hyperspace $\mathscr C_X$ of nonempty weakly closed convex sets of Borel probability measures on $X$, called credal sets. These sets represent uncertainty about which probability law governs the phenomenon of interest. We equip $\mathscr C_X$ with the Hausdorff metric induced by a metric for weak convergence. Credal order intervals $[\mathcal{A},\mathcal{B}] = \{ \mathcal{P}\in\mathscr C_X: \mathcal{A}\subseteq\mathcal{P}\subseteq\mathcal{B} \}$ are complete lattices, and every inclusion-monotone, Hausdorff-continuous self map of such an interval has least and greatest fixed points obtained as limits of iteration from $\mathcal{A}$ and $\mathcal{B}$. For families of inclusion-monotone updates that may vary with time, we construct lower and upper envelope maps that bound every admissible trajectory. When these envelope maps are Hausdorff-continuous, the distance of every traject
    
[^397]: 变分稀疏自编码器分析

    Analysis of Variational Sparse Autoencoders

    [https://arxiv.org/abs/2509.22994](https://arxiv.org/abs/2509.22994)

    本研究通过置换比较分析变分稀疏自编码器，发现非采样 vSAE 与带 L2 激活惩罚的 Top-K SAE 之间的性能差距可完全分解为五个实现细节，且 Top-K 的离散选择机制与随机采样方法结合时会失效。

    

    变分稀疏自编码器用采样得到的高斯后验取代了稀疏自编码器中确定性的 Top-K 机制，据假设这能产生更鲁棒的特征字典。为验证这一假设，我们构建了基于训练种子的置换比较以进行自定义分析，并在一对现有检查点上对两个流行基准进行了评分。当固定单位方差并在 vSAE 上消融采样操作时，我们发现非采样 vSAE 与带 L2 激活惩罚的 Top-K SAE 之间的差距可以完全分解为五个实现细节。当我们启用采样并允许网络学习自己的方差时，我们发现 Top-K 的离散选择过程在与随机采样方法结合时会失效。在这些模型训练所采用的噪声尺度下，相同的模型激活仅因采样过程就会产生截然不同的 Top-K 特征集（Jaccard 相似度为 0.075-0.29，而随机水平为 0.067）……

    arXiv:2509.22994v3 Announce Type: replace  Abstract: Variational sparse autoencoders (vSAEs) replace a sparse autoencoder's deterministic Top-K mechanism with a sampled Gaussian posterior hypothesized to produce a more robust feature dictionary. To test this, we build permutation comparisons over training seeds for custom analyses, and score two popular benchmarks on one pair of existing checkpoints. When fixing the unit variance and ablating sampling on the vSAE, we find that the gap between the non-sampled vSAE and a Top-K SAE with an L2 activation penalty can be decomposed completely into a set of five implementation details. When we enable sampling and allow the network to learn its own variance, we find that the discrete selection process of Top-K fails when in conjunction with stochastic sampling methods. At the noise scale these models are trained upon, vastly different Top-K feature sets (Jaccard 0.075-0.29, chance 0.067) are found on equal model activations due only to the sam
    
[^398]: 面向图算法推理的 GDS 智能体

    GDS Agent for Graph Algorithmic Reasoning

    [https://arxiv.org/abs/2508.20637](https://arxiv.org/abs/2508.20637)

    GDS 智能体通过 MCP 服务器将一整套图算法封装为工具并提供预处理与后处理流程，使任何现代 LLM 都能开箱即用地对大规模图结构数据进行算法推理。

    

    大型语言模型（LLM）已经展现出卓越的多模态信息处理与推理能力。当通过函数调用为其配备工具，并结合检索增强技术后，基于LLM的复合系统可以访问封闭的数据源并回答与之相关的问题。然而，它们在处理和推理大规模图结构数据方面仍然存在困难。在本技术报告中，我们介绍了 GDS（图数据科学）智能体。GDS 智能体在模型上下文协议（MCP）服务器中引入了一套全面的图算法作为工具，并配套了算法结果的预处理（检索）与后处理功能。同时，它还附带一项技能（skill），用于引导 LLM 完成图分析任务。该智能体可与任何现代 LLM 开箱即用。GDS 智能体允许用户提出任何隐含或本质上需要对数据进行图算法推理的问题，并快速获得答案。

    arXiv:2508.20637v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. In this technical report we introduce the GDS (Graph Data Science) agent. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. It is accompanied by a skill that guides the LLM through graph analysis. The agent can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtai
    
[^399]: SACA：基于选择性注意力的聚类算法

    SACA: Selective Attention-Based Clustering Algorithm

    [https://arxiv.org/abs/2508.17150](https://arxiv.org/abs/2508.17150)

    本文提出了一种受选择性注意力概念启发的新型密度聚类算法SACA，通过自适应阈值自动过滤噪声点和异常值并重新整合优化结果，大幅降低了传统密度聚类方法对参数调优的依赖。

    

    聚类算法是许多领域的基础工具，其中基于密度的方法在识别任意形状的簇和处理噪声方面具有独特优势。然而，此类方法的有效性常常受限于用户需要进行关键参数调优，而这通常要求具备大量的领域专业知识。本文提出了一种新颖的基于密度的聚类算法，其设计灵感松散地来源于选择性注意力的概念，旨在最大限度地减少大多数应用场景中对参数调优的依赖。该方法通过计算自适应阈值来排除稀疏分布的点和异常值，构建初始聚类框架，随后将先前过滤掉的点重新整合，以完善最终的聚类结果。在多种基准数据集上进行的广泛实验证明了所提方法的鲁棒性、准确性和易用性，使其成为传统密度聚类方法的有力替代方案。

    arXiv:2508.17150v3 Announce Type: replace-cross  Abstract: Clustering algorithms are fundamental tools across many fields, with density-based methods offering particular advantages in identifying arbitrarily shaped clusters and handling noise. However, their effectiveness is often limited by the requirement of critical parameter tuning by users, which typically requires significant domain expertise. This paper introduces a novel density-based clustering algorithm loosely inspired by the concept of selective attention, designed to minimize reliance on parameter tuning for most applications. The proposed method computes an adaptive threshold to exclude sparsely distributed points and outliers, constructs an initial cluster framework, and subsequently reintegrates the filtered points to refine the final results. Extensive experiments on diverse benchmark datasets demonstrate the robustness, accuracy, and ease of use of the proposed approach, establishing it as a powerful alternative to co
    
[^400]: 差分隐私随机梯度下降的统计推断

    Statistical Inference for Differentially Private Stochastic Gradient Descent

    [https://arxiv.org/abs/2507.20560](https://arxiv.org/abs/2507.20560)

    本文首次建立了随机化子采样规则下SGD的渐近性质并将其扩展到DP-SGD，证明了其渐近方差可分解为统计、采样和隐私诱导三个分量，并提出了代入法和随机缩放法两种构建有效置信区间的方法。

    

    机器学习中的隐私保护，特别是通过差分隐私随机梯度下降（DP-SGD）实现的隐私保护，对于分析敏感数据至关重要。然而，现有的SGD统计推断方法主要集中于循环子采样，而DP-SGD需要随机化子采样。本文首次通过建立随机化规则下SGD的渐近性质并将这些结果扩展到DP-SGD来弥合这一差距。对于DP-SGD，我们证明了渐近方差可分解为统计分量、采样分量和隐私诱导分量三个部分。我们提出了两种构建有效置信区间的方法：代入法和随机缩放法。我们还进行了广泛的数值分析，结果表明所提出的置信区间在保持隐私的同时达到了名义覆盖率。

    arXiv:2507.20560v3 Announce Type: replace-cross  Abstract: Privacy preservation in machine learning, particularly through Differentially Private Stochastic Gradient Descent (DP-SGD), is crucial for analyzing sensitive data. However, existing statistical inference methods for SGD predominantly focus on cyclic subsampling, while DP-SGD requires randomized subsampling. This paper first bridges this gap by establishing the asymptotic properties of SGD under the randomized rule and extending these results to DP-SGD. For DP-SGD, we show that the asymptotic variance decomposes into statistical, sampling, and privacy-induced components. We propose two methods for constructing valid confidence intervals: the plug-in method and the random scaling method. We also perform extensive numerical analysis showing that the proposed confidence intervals achieve nominal coverage rates while maintaining privacy.
    
[^401]: PLAICraft：面向具身智能的大规模时间对齐视觉-语音-动作数据集

    PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI

    [https://arxiv.org/abs/2505.12707](https://arxiv.org/abs/2505.12707)

    该论文提出PLAICraft，一个从232名以上参与者收集的、涵盖视频、游戏音频、麦克风音频、鼠标和键盘五种毫秒级时间对齐模态的5525小时多人Minecraft交互数据集，并配套评估套件用于基准测试具身智能模型的各项能力。

    

    深度生成建模的进展使得训练达到人类水平的具身智能体变得越来越可行。然而，由于缺乏能够反映自然环境感觉运动复杂性的大规模、实时、多模态且具有社交互动性的数据集，这一领域的进展一直受到限制。为了解决这一问题，我们提出了PLAICraft，这是一个新颖的数据收集平台和数据集，用于捕获多人Minecraft游戏互动，涵盖五种时间对齐的模态：视频、游戏输出音频、麦克风输入音频、鼠标和键盘操作。每种模态均以毫秒级时间精度记录，从而支持在丰富、开放的世界中研究同步的具身行为。该数据集包含来自232名以上参与者的5525小时互动游戏内容。除数据集外，我们还提供了一个评估套件，用于在物体识别、空间感知、语言基础和长期记忆等能力方面对模型进行基准测试。

    arXiv:2505.12707v3 Announce Type: replace-cross  Abstract: Advances in deep generative modeling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises 5525 hours of interactive gameplay from more than 232 participants. Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term mem
    
[^402]: 利用神经插值改进区域天气预报

    Improving regional weather forecasts with neural interpolation

    [https://arxiv.org/abs/2505.12040](https://arxiv.org/abs/2505.12040)

    本文设计了一个融合图像超分辨率技术、卷积神经网络、残差网络以及大气动力学流动的神经插值算子来改进区域天气模型的边界数据，其中嵌入动力学流动使小型网络的平方误差减少一半以上。

    

    在本文中，我们设计了一个神经插值算子来改进区域天气模型的边界数据，这是一个具有挑战性的问题，因为我们需要在不同网格分辨率之间映射多尺度动力学。特别地，我们提出了一种通过研究简化模型来处理该问题的方法论，旨在将本工作的结果推广到区域天气模型的动力核心。我们的方法结合了图像超分辨率中的卷积神经网络（CNN）和残差网络等技术，此外还将大气动力学的流动构建到神经网络中。我们发现，将流动嵌入网络后，小型网络的平方误差减少了一半以上。

    arXiv:2505.12040v2 Announce Type: replace  Abstract: In this paper we design a neural interpolation operator to improve the boundary data for regional weather models, which is a challenging problem as we are required to map multi-scale dynamics between grid resolutions. In particular, we expose a methodology for approaching the problem through the study of a simplified model, with a view to generalise the results in this work to the dynamical core of regional weather models. Our approach will exploit a combination of techniques from image super-resolution with convolutional neural networks (CNNs) and residual networks, in addition to building the flow of atmospheric dynamics into the neural network. We find that building in the flow more than halves the squared error of small networks.
    
[^403]: 稳定性正则化交叉验证

    Stability Regularized Cross-Validation

    [https://arxiv.org/abs/2505.06927](https://arxiv.org/abs/2505.06927)

    该论文提出了一种嵌套 k 折交叉验证方案，通过最小化标准交叉验证误差与经验假设稳定性的加权和来选择超参数，并证明了由此得到的泛化界，显示在信噪比较低的场景下该方法对测试误差的改善最为显著。

    

    我们研究通过交叉验证进行超参数选择的问题，其动机源于这样的场景：所选模型的 k 折交叉验证误差会低估该模型的测试误差。我们提出了一种嵌套的 k 折交叉验证方案，该方案最小化标准交叉验证误差与经验假设稳定性的加权和，其中稳定性权重由外层交叉验证循环来选择。我们推导出了一个泛化界，利用交叉验证误差和经验假设稳定性（以及每个候选模型共有的修正项）来控制测试误差。在一个简化的岭回归设定中，我们进一步表明，稳定性权重能否提升期望测试集性能取决于信噪比，当信噪比较小时提升效果最为显著。在 13 个真实世界的回归数据集上，该程序相比 k 折交叉验证降低了平均样本外 MSE。

    arXiv:2505.06927v3 Announce Type: replace-cross  Abstract: We study hyperparameter selection via cross-validation, motivated by settings in which the selected $k$-fold cross-validation error underestimates the selected model's test error. We propose a nested $k$-fold cross-validation scheme that minimizes a weighted sum of the standard cross-validation error and empirical hypothesis stability, with the stability weight selected by an outer cross-validation loop. We derive a generalization bound that controls test error using the cross-validation error and empirical hypothesis stability, together with correction terms common to every candidate. In a stylized ridge regression setting, we then show that whether the stability weight improves expected test-set performance depends on the signal-to-noise ratio, with the greatest improvement when the signal-to-noise ratio is small. On $13$ real-world regression datasets, the procedure reduces the average out-of-sample MSE relative to $k$-fold 
    
[^404]: 一个用于非结构化数据稳健高效推断的统一框架

    A Unifying Framework for Robust and Efficient Inference with Unstructured Data

    [https://arxiv.org/abs/2505.00282](https://arxiv.org/abs/2505.00282)

    该论文提出MAR-S框架，将半参数缺失数据理论应用于AI预测，通过透明的验证样本设计实现非结构化数据下有效、高效且稳健的推断，并首次将框架扩展到聚合和转换后的AI预测这一常见但未被现有方法覆盖的场景。

    

    AI可以低成本地从非结构化数据中提取结构化变量，但不同的实现选择可能会对同一概念产生不同的度量结果。这带来了可信度方面的挑战，实际上将设定搜索问题转移到了上游环节。我们提出MAR-S（随机缺失结构化数据）框架来应对这一挑战，该框架借鉴半参数缺失数据理论，以实现基于AI预测的有效、高效且稳健的推断。MAR-S将测量锚定在透明的验证样本设计之上，同时利用AI来规模化测量。我们提供了MAR-S估计的通用方法，并将该框架扩展到聚合和转换后的AI预测场景，这是现有框架尚未解决的常见问题。

    arXiv:2505.00282v4 Announce Type: replace-cross  Abstract: AI can cheaply extract structured variables from unstructured data, with alternative implementation choices potentially yielding different measures of the same concept. This creates a credibility challenge, effectively moving specification search upstream. We confront this challenge with MAR-S (Missing At Random Structured Data), a framework which adapts semiparametric missing data theory to deliver valid, efficient, robust inference with AI predictions. MAR-S anchors measurement in a transparent validation sample design, while AI scales measurement. We provide general recipes for MAR-S estimation and extend the framework to aggregated and transformed AI predictions, a common setting not addressed by existing frameworks.
    
[^405]: 如何在不（几乎）触碰预训练基础模型的情况下增强下游对抗鲁棒性？

    How to Enhance Downstream Adversarial Robustness (almost) without Touching the Pre-Trained Foundation Model?

    [https://arxiv.org/abs/2504.10850](https://arxiv.org/abs/2504.10850)

    该论文通过理论揭示鲁棒对比学习与监督学习对抗鲁棒性之间的密切关系，提出了一种无需更新或访问预训练基础模型权重即可有效增强下游任务对抗鲁棒性的简单方法。

    

    随着强大基础模型的兴起，预训练-微调范式如今变得越来越流行：基础模型使用来自各种来源的海量数据进行预训练，然后下游用户只需对其进行微调以适应特定的下游任务。然而，由于对抗训练的高计算复杂度，通过微调基础模型来提升其在下游任务上的鲁棒性是不可行的。针对上述挑战，我们希望在不更新或访问基础模型权重的情况下提高下游鲁棒性。受现有鲁棒性继承文献的启发（Kim et al., 2020），通过理论研究，我们发现鲁棒对比学习与监督学习的对抗鲁棒性之间存在密切关系。为了进一步验证并利用这一理论见解，我们设计了一种简单而有效的方法

    arXiv:2504.10850v2 Announce Type: replace  Abstract: With the rise of powerful foundation models, a pre-training-fine-tuning paradigm becomes increasingly popular these days: A foundation model is pre-trained using a huge amount of data from various sources, and then the downstream users only need to fine-tune and adapt it to specific downstream tasks. However, due to the high computation complexity of adversarial training, it is not feasible to fine-tune the foundation model to improve its robustness on the downstream task. Observing the above challenge, we want to improve the downstream robustness without updating/accessing the weights in the foundation model. Inspired from existing literature in robustness inheritance (Kim et al., 2020), through theoretical investigation, we identify a close relationship between robust contrastive learning with the adversarial robustness of supervised learning. To further validate and utilize this theoretical insight, we design a simple-yet-effectiv
    
[^406]: 面向信息丰富的仿真到现实迁移的神经保真度校准

    Neural Fidelity Calibration for Informative Sim-to-Real Adaptation

    [https://arxiv.org/abs/2504.08604](https://arxiv.org/abs/2504.08604)

    提出神经保真度校准（NFC）框架，利用条件得分扩散模型在机器人运行时在线校准仿真器物理参数与残差保真度域，从而无需专家物理知识即可弥合仿真到现实的差距并捕捉环境不确定性。

    

    深度强化学习能够将敏捷的运动与导航技能从仿真器无缝迁移到真实世界。然而，利用域随机化或对抗方法来弥合仿真到现实的差距，通常需要专家级的物理知识来确保策略的鲁棒性。即便如此，最先进的仿真器也可能无法捕捉现实世界的每一个细节，且重建的环境可能因各种感知不确定性而引入误差。为应对这些挑战，我们提出了神经保真度校准（Neural Fidelity Calibration, NFC），这是一个新颖的框架，利用条件得分扩散模型在机器人执行过程中在线校准仿真器的物理参数和残差保真度域。具体而言，残差保真度反映了仿真模型相对于真实世界动力学的偏移，并捕捉了感知环境的不确定性，使我们能够在推断过程中采样真实的环境。

    arXiv:2504.08604v2 Announce Type: replace-cross  Abstract: Deep reinforcement learning can seamlessly transfer agile locomotion and navigation skills from the simulator to real world. However, bridging the sim-to-real gap with domain randomization or adversarial methods often demands expert physics knowledge to ensure policy robustness. Even so, cutting-edge simulators may fall short of capturing every real-world detail, and the reconstructed environment may introduce errors due to various perception uncertainties. To address these challenges, we propose Neural Fidelity Calibration (NFC), a novel framework that employs conditional score-based diffusion models to calibrate simulator physical coefficients and residual fidelity domains online during robot execution. Specifically, the residual fidelity reflects the simulation model shift relative to the real-world dynamics and captures the uncertainty of the perceived environment, enabling us to sample realistic environments under the infe
    
[^407]: 使用大语言模型起草架构设计决策：DRAFT方法

    DRAFT-ing Architectural Design Decisions using LLMs

    [https://arxiv.org/abs/2504.08207](https://arxiv.org/abs/2504.08207)

    该论文提出了DRAFT方法，将检索增强少样本生成与微调相结合，使大语言模型能够根据决策上下文生成架构设计决策，并根据标题生成完整的架构决策记录。

    

    架构知识管理（AKM）对软件开发至关重要，而架构决策记录（ADR）作为AKM的重要产物，提供了一种结构化的方式来记录架构设计决策（ADD）。然而，由于需要大量人工投入且工具支持不足，ADR的实际采用率受限。先前的研究表明，大语言模型（LLM）能够支持ADD的生成，但简单的提示方法效果欠佳。本文探讨了LLM如何在两个互补的任务中为架构师提供支持：从决策上下文生成ADD，以及从标题生成完整的ADR。我们提出了领域特定检索增强少样本调优，这是一种将检索增强少样本生成与微调相结合的新型生成方法。DRAFT分两个阶段运行：离线阶段使用从历史ADR中提取的检索增强提示对模型进行微调，在线阶段生成……

    arXiv:2504.08207v2 Announce Type: replace-cross  Abstract: Architectural Knowledge Management (AKM) is crucial for software development and Architectural Decision Records (ADR), an important artifact of AKM, provide a structured approach to capture Architectural Design Decisions (ADD). However, their adoption is limited due to the manual effort involved and insufficient tool support. Prior work shows that LLMs can support ADD generation, though basic prompting yields suboptimal results. This paper explores how LLMs can support architects in two complementary tasks: generating ADDs from Decision Contexts, and generating complete ADRs from titles. We propose Domain-specific Retrieval Augmented Few-shot Tuning (DRAFT), a novel approach that integrates Retrieval Augmented Few-shot Generation with Fine-tuning to generate. DRAFT operates in two phases: an offline phase that fine-tunes a model using retrieval-augmented prompts derived from historical ADRs, and an online phase that generates d
    
[^408]: 图上的分布外检测：综述

    Out-of-Distribution Detection on Graphs: A Survey

    [https://arxiv.org/abs/2502.08105](https://arxiv.org/abs/2502.08105)

    本文提出了首个针对图分布外（GOOD）检测的系统性综述，将现有方法按其主要OOD判别机制分为六大类别，并系统考察了它们的技术关联、监督需求、数据集、基准与评估指标。

    

    图学习已在众多关系数据应用中推动了预测与发现。在不断演变的环境中，新类别的出现以及属性或关系的变化会产生分布外（OOD）输入，模型会在这些输入上做出看似自信却不可靠的预测。图分布外（GOOD）检测旨在识别此类节点或图，以支持对预测的拒答与进一步评估。本文提出了首个专门针对GOOD检测的系统性综述。我们阐明了问题设定，并根据主要的OOD判别机制将现有方法归纳为六大类别。该综述考察了各类方法之间的技术关联、监督与参考信息需求以及适用条件，同时涵盖了代表性数据集、OOD构造方式、基准测试与评估指标。受控的合成对比实验通过展示代表性方法……加深了对检测设定的理解。

    arXiv:2502.08105v2 Announce Type: replace  Abstract: Graph learning has advanced prediction and discovery across a wide range of relational data applications. In evolving environments, new categories and changes in attributes or relationships can produce out-of-distribution (OOD) inputs on which models make confident but unreliable predictions. Graph out-of-distribution (GOOD) detection identifies such nodes or graphs to support rejection and further assessment. This paper presents the first systematic survey dedicated to GOOD detection. We clarify the problem settings and organize existing methods into six categories according to their primary OOD discrimination mechanisms. The review examines their technical connections, supervision and reference requirements, and conditions of use, alongside representative datasets, OOD constructions, benchmarks, and evaluation metrics. Controlled synthetic comparisons deepen the understanding of detection settings by illustrating how representative
    
[^409]: 遗忘向量发挥作用：驱动图像分类与生成任务中机器遗忘的通用输入扰动

    Forget Vectors at Play: Universal Input Perturbations Driving Machine Unlearning in Image Classification and Generation

    [https://arxiv.org/abs/2412.16780](https://arxiv.org/abs/2412.16780)

    该论文提出了一种名为“遗忘向量”的通用输入扰动方法，无需修改模型权重，即可在图像分类与图像生成任务中实现与基于模型的遗忘方法同样有效的机器遗忘。

    

    机器遗忘（MU）旨在消除特定不需要的数据对已训练模型的影响，在模型编辑中正变得日益重要，特别是为了遵守诸如“被遗忘权”等不断发展的数据法规。传统的机器遗忘方法主要基于模型，通常需要重新训练或微调模型权重以满足遗忘要求。在本工作中，我们从基于输入扰动的视角来处理机器遗忘问题，在整个遗忘过程中模型权重保持不变。我们证明了一种主动的基于输入的遗忘策略的存在，称之为“遗忘向量”（forget vector），它可以作为与输入无关的数据扰动生成，并且在图像分类和图像生成任务中与基于模型的近似遗忘方法同样有效。对于图像分类任务，我们证明了遗忘向量可以诱导有效的类（原文在此处截断）……

    arXiv:2412.16780v5 Announce Type: replace  Abstract: Machine unlearning (MU), which seeks to erase the influence of specific unwanted data from already-trained models, is becoming increasingly vital in model editing, particularly to comply with evolving data regulations like the "right to be forgotten". Conventional MU approaches are predominantly model-based, typically requiring retraining or fine-tuning the model's weights to meet unlearning requirements. In this work, we approach the MU problem from an input perturbation-based perspective, where the model weights remain intact throughout the unlearning process. We demonstrate the existence of a proactive input-based unlearning strategy, referred to forget vector, which can be generated as an input-agnostic data perturbation and remains as effective as model-based approximate unlearning approaches in both image classification and image generation. For image classification, we demonstrate that forget vector can induce effective class-
    
[^410]: 保护协作学习中的机密性、隐私与完整性

    Protecting Confidentiality, Privacy and Integrity in Collaborative Learning

    [https://arxiv.org/abs/2412.08534](https://arxiv.org/abs/2412.08534)

    本文提出Citadel++协作式机器学习训练系统，通过以虚拟机级可信执行环境（TEE）作为训练运行时的链式方法，同时保护数据集、模型和训练代码的机密性以及个体用户数据的隐私。

    

    有效的机器学习（ML）训练需要数据集所有者与模型所有者之间的协作。然而，在这种协作过程中，数据集所有者和模型所有者都希望保护各自资产（即数据集、模型和训练代码）的机密性，同时数据集所有者还关心数据集中个体用户数据的隐私。现有的解决方案要么对模型和训练代码提供的机密性保护有限，要么存在数据隐私问题，要么会带来不可接受的性能开销。我们提出了Citadel++，一个协作式机器学习训练系统，能够同时保护数据集、模型和训练代码的机密性以及个体用户数据的隐私。具体而言，Citadel++采用了一种链式方法：首先，我们采用虚拟机级别的可信执行环境（TEE）作为训练运行时，为数据集、模型和训练代码提供机密性保护。

    arXiv:2412.08534v3 Announce Type: replace-cross  Abstract: A collaboration between dataset owners and model owners is needed for effective machine learning (ML) training. During this collaboration, however, dataset owners and model owners want to protect the confidentiality of their respective assets (i.e., datasets, models and training code), with the dataset owners also caring about the privacy of individual users whose data is in the datasets. Existing solutions provide limited confidentiality for models and training code, suffer from data privacy issues, or incur unacceptable performance.   We present Citadel++, a collaborative ML training system to simultaneously protect the confidentiality of datasets, models and training code as well as the privacy of individual user data. Specifically, Citadel++ takes a chained approach. First, we employ the Virtual Machine-level Trusted Execution Environments (TEEs) as the training runtime to provide confidentiality protections for datasets, m
    
[^411]: Caravan MultiMet：通过多种天气临近预报和预报数据扩展Caravan

    Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts

    [https://arxiv.org/abs/2411.09459](https://arxiv.org/abs/2411.09459)

    本文提出了Caravan水文数据集的新扩展Caravan MultiMet，在ERA5-Land再分析数据基础上增加了三种降水临近预报产品和三种天气预报产品，从而丰富气象强迫数据，支持对水文模型（尤其是实时预报场景）进行更稳健的评估和基准测试。

    

    Caravan大样本水文学数据集（Kratzert等人，2023）旨在对来自各种区域数据集的径流数据进行标准化和协调，并将其与全球可用的气象强迫数据和流域属性相结合。这个社区驱动的项目还允许研究人员方便地为额外的流域扩展该数据集，迄今为止已完成6次扩展（见 https://github.com/kratzert/Caravan/discussions/10）。我们提出了Caravan的一项新颖扩展，重点在于丰富气象强迫数据。该扩展在现有的ERA5-Land再分析数据基础上，增加了三种降水临近预报产品（CPC、IMERG v07 Early和CHIRPS）以及三种天气预报产品（ECMWF IFS HRES、GraphCast和CHIRPS-GEFS）。纳入多样化的数据源，尤其是天气预报数据，能够对水文模型进行更稳健的评估和基准测试，特别是在实时预报场景中。据我们所知...

    arXiv:2411.09459v2 Announce Type: replace  Abstract: The Caravan large-sample hydrology dataset (Kratzert et al., 2023) was created to standardize and harmonize streamflow data from various regional datasets, combined with globally available meteorological forcing and catchment attributes. This community-driven project also allows researchers to conveniently extend the dataset for additional basins, as done 6 times to date (see https://github.com/kratzert/Caravan/discussions/10). We present a novel extension to Caravan, focusing on enriching the meteorological forcing data. Our extension adds three precipitation nowcast products (CPC, IMERG v07 Early, and CHIRPS) and three weather forecast products (ECMWF IFS HRES, GraphCast, and CHIRPS-GEFS) to the existing ERA5-Land reanalysis data. The inclusion of diverse data sources, particularly weather forecasts, enables more robust evaluation and benchmarking of hydrological models, especially for real-time forecasting scenarios. To the best o
    
[^412]: ADformer：一种用于基于脑电图检测阿尔茨海默病的多粒度时空Transformer

    ADformer: A Multi-Granularity Spatial-Temporal Transformer for EEG-Based Alzheimer Detection

    [https://arxiv.org/abs/2409.00032](https://arxiv.org/abs/2409.00032)

    提出ADformer，一种多粒度时空Transformer，可直接从原始脑电信号端到端地学习时间与空间特征表示，有效提升阿尔茨海默病检测的性能与泛化能力。

    

    脑电图（EEG）已成为一种经济高效的工具，可帮助神经科医生检测阿尔茨海默病。然而，现有的大多数方法严重依赖人工特征工程或数据变换。虽然这些技术在处理小规模数据集时可能有所帮助，但在应用于大规模数据时，它们往往导致信息丢失和失真，最终限制了模型性能。此外，以往研究所使用的数据集受试者规模有限且人口统计学多样性不足，阻碍了对模型鲁棒性和泛化能力的全面评估，从而限制了其在真实临床环境中的适用性。为应对这些挑战，我们提出了ADformer，一种新颖的多粒度时空Transformer，能够从原始脑电信号中捕获时间和空间特征，实现有效的端到端表示学习。我们的模型引入了跨（通道与时间维度的）多粒度嵌入策略……

    arXiv:2409.00032v3 Announce Type: replace-cross  Abstract: Electroencephalography has emerged as a cost-effective, efficient tool to help neurologists detect Alzheimer's Disease. However, most existing approaches rely heavily on manual feature engineering or data transformation. While these techniques may help with small-scale datasets, they often cause information loss and distortion when applied to large-scale data, ultimately limiting model performance. Moreover, the limited subject scale and demographic diversity of datasets used in prior studies hinder comprehensive evaluation of model robustness and generalizability, thus restricting their applicability in real-world clinical settings. To address these challenges, we propose ADformer, a novel multi-granularity spatial-temporal transformer that captures both temporal and spatial features from raw EEG signals, enabling effective end-to-end representation learning. Our model introduces multi-granularity embedding strategies across b
    
[^413]: Point2SSM++：从点云中自监督学习解剖形状模型

    Point2SSM++: Self-Supervised Learning of Anatomical Shape Models from Point Clouds

    [https://arxiv.org/abs/2405.09707](https://arxiv.org/abs/2405.09707)

    提出了Point2SSM++，一种自监督深度学习方法，可直接从解剖形状的点云表示中学习对应点，无需完整对齐的表面表示，同时避免了传统优化方法带来的偏差和漫长的推理时间。

    

    基于对应关系的统计形状建模（SSM）是临床研究中形态测量分析的一项强大技术。SSM有助于对骨骼和器官等解剖形状进行群体水平的表征与量化，辅助病理与疾病诊断以及治疗规划。尽管潜力巨大，但SSM在医学研究中仍未得到充分利用，其原因在于自动构建方法存在巨大的额外开销，这些方法要求完整且对齐的形状表面表示。此外，基于优化的技术依赖于会引入偏差的假设或模板，并且由于需要对整个队列同时进行优化，推理时间较长。为了克服这些挑战，我们提出了Point2SSM++，这是一种原则性的、自监督的深度学习方法，可直接从解剖形状的点云表示中学习对应点。Point2SSM++对……

    arXiv:2405.09707v2 Announce Type: replace-cross  Abstract: Correspondence-based statistical shape modeling (SSM) stands as a powerful technology for morphometric analysis in clinical research. SSM facilitates population-level characterization and quantification of anatomical shapes such as bones and organs, aiding in pathology and disease diagnostics and treatment planning. Despite its potential, SSM remains under-utilized in medical research due to the significant overhead associated with automatic construction methods, which demand complete, aligned shape surface representations. Additionally, optimization-based techniques rely on bias-inducing assumptions or templates and have prolonged inference times as the entire cohort is simultaneously optimized. To overcome these challenges, we introduce Point2SSM++, a principled, self-supervised deep learning approach that directly learns correspondence points from point cloud representations of anatomical shapes. Point2SSM++ is robust to mis
    
[^414]: Neur2BiLO: 神经双层优化

    Neur2BiLO: Neural Bilevel Optimization

    [https://arxiv.org/abs/2402.02552](https://arxiv.org/abs/2402.02552)

    Neur2BiLO是一个针对双层优化问题的框架，通过将神经网络近似引入到混合整数规划中，可以快速生成高质量的解决方案。

    

    双层优化处理嵌套问题，在这些问题中，领导者首先做出决策以最小化自己的目标函数，同时考虑到追随者的最好反应。整数变量约束的双层问题特别难以处理。尽管已经提出了用于混合整数线性双层优化的精确求解器，但它们在问题规模较大时往往无法扩展，并且难以推广到非线性情况。另一方面，问题特定的算法（精确和启发式）局限于特定范围。在以数据驱动的环境下，我们提出的框架Neur2BiLO将通过监督回归训练的领导者或追随者的值函数的神经网络近似嵌入到易于解决的混合整数规划中。 Neur2BiLO作为一种启发式算法，可以快速生成高质量的解决方案，适用于双层背包拦截问题，即“关键n个问题”。

    Bilevel optimization deals with nested problems in which a leader takes the first decision to minimize their objective function while accounting for a follower's best-response reaction. Constrained bilevel problems with integer variables are particularly notorious for their hardness. While exact solvers have been proposed for mixed-integer linear bilevel optimization, they tend to scale poorly with problem size and are hard to generalize to the non-linear case. On the other hand, problem-specific algorithms (exact and heuristic) are limited in scope. Under a data-driven setting in which similar instances of a bilevel problem are solved routinely, our proposed framework, Neur2BiLO, embeds a neural network approximation of the leader's or follower's value function, trained via supervised regression, into an easy-to-solve mixed-integer program. Neur2BiLO serves as a heuristic that produces high-quality solutions extremely fast for the bilevel knapsack interdiction problem, the "critical n
    
[^415]: 动态张量积回归

    Dynamic Tensor Product Regression

    [https://arxiv.org/abs/2210.03961](https://arxiv.org/abs/2210.03961)

    本文开创性地提出了动态张量积回归问题，并设计了一种动态树数据结构，使得当某个因子矩阵发生稀疏变化时，可以快速传播更新、高效维护张量积回归的解，而无需从头重新计算。

    

    在这项工作中，我们开创了对动态张量积回归问题的研究。给定矩阵 $A_1\in \mathbb{R}^{n_1\times d_1},\ldots,A_q\in \mathbb{R}^{n_q\times d_q}$ 和标签向量 $b\in \mathbb{R}^{n_1\ldots n_q}$，目标是求解以这些矩阵的张量积作为设计矩阵 $A$ 的回归问题，即 $\min_{x\in \mathbb{R}^{d_1\ldots d_q}} \|(A_1\otimes \ldots\otimes A_q)x-b\|_2$。在每个时间步，其中一个矩阵 $A_i$ 会发生稀疏变化，目标是维护张量积 $A_1\otimes\ldots \otimes A_q$ 的一个概要，使得回归解可以被快速更新。每一轮都从头重新计算解是非常缓慢的，因此开发能够在设计矩阵更新后快速更新解的算法非常重要。我们的主要结果是一种动态树数据结构，对单个矩阵的任何更新都可以被快速传播（摘要原文在此处截断）。

    arXiv:2210.03961v3 Announce Type: replace-cross  Abstract: In this work, we initiate the study of Dynamic Tensor Product Regression. One has matrices $A_1\in \mathbb{R}^{n_1\times d_1},\ldots,A_q\in \mathbb{R}^{n_q\times d_q}$ and a label vector $b\in \mathbb{R}^{n_1\ldots n_q}$, and the goal is to solve the regression problem with the design matrix $A$ being the tensor product of the matrices $A_1, A_2, \dots, A_q$ i.e. $\min_{x\in \mathbb{R}^{d_1\ldots d_q}} \|(A_1\otimes \ldots\otimes A_q)x-b\|_2$. At each time step, one matrix $A_i$ receives a sparse change, and the goal is to maintain a sketch of the tensor product $A_1\otimes\ldots \otimes A_q$ so that the regression solution can be updated quickly. Recomputing the solution from scratch for each round is very slow and so it is important to develop algorithms which can quickly update the solution with the new design matrix. Our main result is a dynamic tree data structure where any update to a single matrix can be propagated quick
    
[^416]: 机器学习驱动的虚拟竞价与电力市场效率分析

    Machine Learning-Driven Virtual Bidding with Electricity Market Efficiency Analysis

    [https://arxiv.org/abs/2104.02754](https://arxiv.org/abs/2104.02754)

    本文提出了一种考虑风险约束和价格敏感性的机器学习驱动虚拟竞价投资组合优化框架，并利用该策略评估了虚拟投标组合的盈利能力及美国批发电力市场的效率。

    

    本文开发了一个机器学习驱动的电力市场虚拟竞价投资组合优化框架，同时考虑了风险约束和价格敏感性。该算法交易策略从自营交易公司的角度出发，以实现利润最大化为目标。通过利用市场出清算法的小时间依赖性，开发了一种基于循环神经网络的节点边际电价（LMP）价差预测模型。LMP价差对净虚拟投标的敏感性通过所提出的约束梯度提升树被建模为单调函数。我们利用所提出的算法虚拟投标交易策略来评估虚拟投标组合的盈利能力以及美国批发电力市场的效率。对PJM、ISO-NE和CAISO的全面实证分析表明，所提出的虚拟投标组合优化策略...（原文截断）

    arXiv:2104.02754v2 Announce Type: replace  Abstract: This paper develops a machine learning-driven portfolio optimization framework for virtual bidding in electricity markets considering both risk constraint and price sensitivity. The algorithmic trading strategy is developed from the perspective of a proprietary trading firm to maximize profit. A recurrent neural network-based Locational Marginal Price (LMP) spread forecast model is developed by leveraging the inter-hour dependencies of the market clearing algorithm. The LMP spread sensitivity with respect to net virtual bids is modeled as a monotonic function with the proposed constrained gradient boosting tree. We leverage the proposed algorithmic virtual bid trading strategy to evaluate both the profitability of the virtual bid portfolio and the efficiency of U.S. wholesale electricity markets. The comprehensive empirical analysis on PJM, ISO-NE, and CAISO indicates that the proposed virtual bid portfolio optimization strategy cons
    
[^417]: 非高斯有向无环图的结构迁移学习

    Structural transfer learning of non-Gaussian DAG. (arXiv:2310.10239v1 [stat.ML])

    [http://arxiv.org/abs/2310.10239](http://arxiv.org/abs/2310.10239)

    本文提出了一种用于非高斯有向无环图的结构迁移学习方法，通过引入新颖的DAG结构相似度度量并利用不同相似性水平的辅助DAG的信息，能够显著提高目标研究中的DAG重构效果。

    

    有向无环图(DAG)被广泛用于表示一组收集到的节点之间的方向关系。然而，一次单独研究中可用的数据往往有限，不足以准确地重构DAG，而异构数据可能会从多个相关研究中收集到。如何将异构数据汇集到目标研究中以更好地重构DAG结构仍然是一个未解决的问题。在本文中，我们首先引入了一套新颖的DAG结构相似度度量，然后提出了一个传输DAG学习框架，通过有效地利用不同相似性水平的辅助DAG的信息来提高目标研究中的DAG重构效果。我们的理论分析表明，在目标DAG没有整体与辅助DAG相似的情况下，我们的方法也能在DAG重构方面实现显著的改进，这与大多数现有的传输学习方法形成鲜明对比。我们的实验结果也支持了所提出的传输DAG学习的优势。

    Directed acyclic graph (DAG) has been widely employed to represent directional relationships among a set of collected nodes. Yet, the available data in one single study is often limited for accurate DAG reconstruction, whereas heterogeneous data may be collected from multiple relevant studies. It remains an open question how to pool the heterogeneous data together for better DAG structure reconstruction in the target study. In this paper, we first introduce a novel set of structural similarity measures for DAG and then present a transfer DAG learning framework by effectively leveraging information from auxiliary DAGs of different levels of similarities. Our theoretical analysis shows substantial improvement in terms of DAG reconstruction in the target study, even when no auxiliary DAG is overall similar to the target DAG, which is in sharp contrast to most existing transfer learning methods. The advantage of the proposed transfer DAG learning is also supported by extensive numerical ex
    
[^418]: 评述与未来展望

    Tensor Networks Meet Neural Networks: A Survey and Future Perspectives. (arXiv:2302.09019v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09019](http://arxiv.org/abs/2302.09019)

    这篇论文评述了张量网络和神经网络并介绍了它们的结合：张量神经网络(TNN)，探讨了TNN在网络压缩、信息融合和量子启发式神经网络构建方面的优缺点和未来研究方向。

    

    张量网络(TN)和神经网络(NN)是两种基本的数据建模方法。TN旨在通过将指数维度转化为多项式复杂度来解决处理大规模张量的维度灾难，并吸引了物理和机器学习的关注。与此同时，NN在计算机视觉、自然语言处理和机器人研究等各种应用中表现出了优异的性能。值得注意的是，尽管这两种网络源于不同的观察，但它们通过TNN (张量神经网络)概念的结合有着内在的联系，共同支撑着多线性结构。本文介绍了TNN的三个方面：网络压缩、信息融合和量子启发式神经网络构建，并阐述了TNN与其他神经网络架构的差异以及TNN的优点、局限性及其对于不同应用的潜在影响。最后，我们讨论了TNN的未来研究方向。

    Tensor networks (TNs) and neural networks (NNs) are two fundamental data modeling approaches. TNs were introduced to solve the curse of dimensionality in large-scale tensors by converting an exponential number of dimensions to polynomial complexity. As a result, they have attracted significant attention in the fields of quantum physics and machine learning. Meanwhile, NNs have displayed exceptional performance in various applications, e.g., computer vision, natural language processing, and robotics research. Interestingly, although these two types of networks originate from different observations, they are inherently linked through the common multilinearity structure underlying both TNs and NNs, thereby motivating a significant number of intellectual developments regarding combinations of TNs and NNs. In this paper, we refer to these combinations as tensorial neural networks (TNNs), and present an introduction to TNNs in three primary aspects: network compression, information fusion, a
    
[^419]: 针对低带宽通道和漏洞的一阶方法的草图技术：高效算法

    Sketching for First Order Method: Efficient Algorithm for Low-Bandwidth Channel and Vulnerability. (arXiv:2210.08371v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.08371](http://arxiv.org/abs/2210.08371)

    本文提出了一种针对大规模分布式学习场景的新型草图方案，通过节约通信成本实现联合学习，但是需要进行差分隐私处理以保护局部数据的隐私。

    

    草图技术是大规模机器学习中最基础的工具之一，通过对原始大问题进行随机压缩到更低维度的方式实现运行时和内存的节省。本文提出了一种新颖的一阶方法草图方案，用于大规模分布式学习场景中，以节约分布式代理之间的通信成本，同时保证算法的收敛。基于一个高维度 $d$ 的梯度信息，代理通过草图矩阵 $R\in \mathbb{R}^{s\times d}$ 处理这些压缩信息，在传递时只传输草图后的信息，而接收方会通过反草图矩阵 $R^\top$ 进行还原。利用这种框架，我们开发了具有较低通信成本的联合学习算法。然而，如此随机草图并不能直接保护局部数据的隐私。我们表明，在应用差分隐私后，梯度泄漏问题仍然存在。

    Sketching is one of the most fundamental tools in large-scale machine learning. It enables runtime and memory saving via randomly compressing the original large problem into lower dimensions. In this paper, we propose a novel sketching scheme for the first order method in large-scale distributed learning setting, such that the communication costs between distributed agents are saved while the convergence of the algorithms is still guaranteed. Given gradient information in a high dimension $d$, the agent passes the compressed information processed by a sketching matrix $R\in \mathbb{R}^{s\times d}$ with $s\ll d$, and the receiver de-compressed via the de-sketching matrix $R^\top$ to ``recover'' the information in original dimension. Using such a framework, we develop algorithms for federated learning with lower communication costs. However, such random sketching does not protect the privacy of local data directly. We show that the gradient leakage problem still exists after applying the
    

