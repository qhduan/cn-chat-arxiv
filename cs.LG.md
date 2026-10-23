# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success](https://arxiv.org/abs/2610.31670) | 提出无需训练的离线指标“感知运动对齐”（S²），通过衡量模型感知表征与物理运动几何结构的对齐程度，低成本地预测具身模型的下游任务成功率。 |
| [^2] | [Phase Transition from Random Features to Fully Trainable Networks](https://arxiv.org/abs/2610.31663) | 本文为部分可训练的两层神经网络建立了统一的平方 L₂ 逼近误差界 O((κ_P log(m/δ)/m)^α)，揭示了从随机特征到全可训练网络之间存在相变现象：初始改善快速平滑随后收益递减，且初始化失配越大转变越剧烈。 |
| [^3] | [Reliable AI-Assisted Decisions via Selective Auditing](https://arxiv.org/abs/2610.31662) | 提出了一个在线选择性审计框架，在最小化专家审计频率的同时，提供可证明的错误控制保证，使未经审计的AI错误比例始终低于用户设定的阈值。 |
| [^4] | [SpectralCanary: Embarrassingly Simple Low-Frequency Traces for T2I Dataset-Use Auditing](https://arxiv.org/abs/2610.31659) | SpectralCanary通过低通滤波掩码将固定金丝雀图像嵌入数据集，以极其简单且无需辅助生成或逐图像优化的方式，实现对文生图模型是否使用特定数据集的高效审计检测。 |
| [^5] | [Cost-Aware Abstention for LLM-Based IT Ticket Classification](https://arxiv.org/abs/2610.31657) | 该论文提出一种离线的成本感知弃权评估框架，通过在校准数据上按显式成本选择并冻结路由策略，在真实IT工单数据上显著降低分类成本与错误路由率，并表明简单的TF-IDF逻辑回归结合成本感知阈值可优于大语言模型方案。 |
| [^6] | [METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning](https://arxiv.org/abs/2610.31656) | 提出METTA方法，将元强化学习的快速适应分解为片段内的因果神经记忆细粒度自适应与跨片段的紧凑摘要级任务推断两个时间尺度，以线性序列复杂度实现了对未见任务的高效测试时适应。 |
| [^7] | [EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence](https://arxiv.org/abs/2610.31650) | 该论文提出了EpiSEM-Bench基准，通过六个领域中300对匹配的不安全/安全任务和系统性关键事实抑制，评估大语言模型智能体在证据缺失时能否主动恢复证据并安全地拒绝执行被禁止的操作。 |
| [^8] | [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642) | 该论文发现了softmax、log-softmax和交叉熵标准实现中的两个数值缺陷，并提出了简单的修复方案，该方案消除了“弹弓”现象、加速grokking、防止准确率崩溃并稳定了知识蒸馏过程。 |
| [^9] | [Causal Chain of Thought: Mitigating Attention Leakage via Context Isolation](https://arxiv.org/abs/2610.31639) | 提出CausalCoT推理框架，将问题分解为原子子问题的有向无环图，并在仅含直接因果父节点的隔离上下文中评估每个节点，从而消除注意力泄露，确保思维链推理步骤对最终答案具有真实的因果驱动作用。 |
| [^10] | [BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing](https://arxiv.org/abs/2610.31633) | 本文提出一种两阶段训练方法，通过在预训练中使用随机深度诱导收缩残差动力学，使循环架构实现KV缓存跨递归共享，从而将推理内存从线性增长降为常数而不损失模型质量。 |
| [^11] | [Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3](https://arxiv.org/abs/2610.31624) | 本文以OLMo-3-7B-Think在SFT、DPO及13个中间RLVR检查点上的行为轨迹为证据，质疑可验证奖励强化学习是否必然诱发模型的工具性收敛行为，指出该行为可能取决于训练方案、评估框架与后训练阶段。 |
| [^12] | [OP-Print: Tracing Membership Imprints in On-Policy Self-Distillation](https://arxiv.org/abs/2610.31616) | 本文首次研究了针对在线策略自蒸馏的成员推断攻击，发现其监督印记稀疏且集中于不确定性敏感的修正位点，并据此提出了统一的攻击框架 OP-Print，在灰盒和黑盒威胁模型下均能有效识别训练成员。 |
| [^13] | [Efficient and Optimal Learning of Discrete Distributions with Person-Level Privacy](https://arxiv.org/abs/2610.31612) | 本文首次在个人级纯差分隐私下实现了批处理设置中学习离散分布的多项式时间算法，其样本复杂度在对数因子内匹配信息论最优界，并在近似差分隐私下同样达到近最优的 $\widetilde O(k)$ 样本复杂度，核心技术贡献是一种新的六阶和平方估计器。 |
| [^14] | [CRANE: Post-Hoc Counterfactual Retrieval for Multimodal Classifiers](https://arxiv.org/abs/2610.31594) | 本文提出首个面向多模态分类器的事后、模型无关的反事实解释框架CRANE，通过从目标类别检索真实训练样本作为解释，在构造上保证了多模态反事实解释的合理性与连贯性。 |
| [^15] | [Beyond Best-of-$k$: Prior-Consistent DAPS for Ambiguous Nonlinear Inverse Problems](https://arxiv.org/abs/2610.31592) | 针对歧义非线性逆问题中测量引导可能收敛到先验不支持的解这一失败情形，本文提出PC-DAPS，通过在单次采样运行中维持多个候选解来保持先验一致性，从而超越了依赖真值比较的best-of-$k$评估方式。 |
| [^16] | [Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination](https://arxiv.org/abs/2610.31586) | 该研究按西班牙MIR住院医师考试的官方规则评估了四个开放大语言模型，发现三个较大模型的成绩可排进前600至3,000名并仍能匹配到41至43个专业，但其弃答策略从不划算，医学微调可减少过度自信，而其临床推理与图像利用能力仍存疑。 |
| [^17] | [Action Learning for Scientific Simulations](https://arxiv.org/abs/2610.31585) | 提出 LEAP 方法，通过将未知随机动作锚定到可处理的参考过程，并比较观测轨迹与参考轨迹来学习修正项，为从轨迹观测中恢复动作提供了可微分的路径级目标函数，从而支持科学仿真中动力学系统的推理与优化。 |
| [^18] | [Rethinking Sequence Compression with Test-time Regression](https://arxiv.org/abs/2610.31574) | 提出TTR-Compressor，将序列压缩重新建模为测试时回归问题，其回归系数在前向传播中以闭式解高效计算并通过反向传播端到端学习，在稀疏注意力压缩和多模态视觉token合并任务上显著优于传统的池化与投影压缩方法。 |
| [^19] | [On the Token Collapse Phenomenon in Textual Graph Soft Prompting](https://arxiv.org/abs/2610.31572) | 该论文揭示了图软提示方法G-Retriever存在“词元坍缩”病态现象——图词元在不同输入间几乎无法区分且对答案准确率无实际影响，并通过大量实验证明这一缺陷被后续方法普遍继承。 |
| [^20] | [When to Follow the Teacher: Advantage-Based On-Policy Distillation](https://arxiv.org/abs/2610.31563) | 该论文提出基于优势的在策略蒸馏（AOPD），仅利用学生标准回滚轨迹及其终止奖励来估计教师动作在学生策略下的反事实优势，从而只强化真正对学生有益的教师动作，克服了传统在线策略蒸馏中“始终跟随教师”的局限。 |
| [^21] | [Diffusion Language Models for Perplexity-Aware Black-Box Attacks](https://arxiv.org/abs/2610.31558) | 该论文提出将连续潜空间攻击框架与连续扩散语言模型（dLLM）直接耦合，以学习到的语言生成分布取代人工随机映射，从而生成更自然、更难被防护机制检测且基于自然提示的黑盒攻击提示词。 |
| [^22] | [Fine-Grained Analysis of Catastrophic Forgetting in Continual Learning via Algorithmic Stability](https://arxiv.org/abs/2610.31551) | 本文提出了一个不依赖特定模型架构的算法稳定性理论框架，将持续学习中的灾难性遗忘细粒度地分解为任务内与跨任务泛化差距及经验遗忘，并系统分析了三种学习率调度对遗忘的影响。 |
| [^23] | [Shift-aware score transport for privacy auditing without retraining](https://arxiv.org/abs/2610.31549) | 提出一种分布偏移感知的分数传输方法，通过从分布外非成员样本中学习并校正成员推断分数分布，实现了在无法重训练模型情况下的隐私审计。 |
| [^24] | [Same Luck, Different Choice: Twin-World Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2610.31544) | 提出TWICE方法，通过让反事实孪生回合继承事实回合的采样噪声（即“相同运气、不同选择”），将配对结果差异转化为回合级优势，从而在智能体强化学习中对每个决策实现更准确的信用分配。 |
| [^25] | [PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization](https://arxiv.org/abs/2610.31539) | 本文提出PYTHIA框架，通过结合基础模型、迁移学习（子空间对齐）与高斯过程不确定性量化，仅需少量目标数据即可将机器学习迁移应用于新颖且研究不足的材料，实现缺陷与吸附位点的同步优化。 |
| [^26] | [LACToS: Local Anomaly Detection via Conditional Typicality of Superpixels](https://arxiv.org/abs/2610.31538) | LACToS通过将像素分组为超像素并采用计算高效的代理来估计统计典型性，实现了一种在多模态分布内数据场景下依然稳健的局部分布外异常区域检测方法。 |
| [^27] | [Direct Inference on Admissible Sets (Diona)](https://arxiv.org/abs/2610.31526) | Diona是一种无需全局训练、通过按需局部求解约束优化问题直接从样本估计查询点状态及导数的新回归方法，在无噪声数据上精度超越高斯过程，且具备有限样本覆盖等理论保证。 |
| [^28] | [Nonparametric autoregressive estimation of probability distributions with parallel gradient boosting](https://arxiv.org/abs/2610.31524) | 该论文提出通过自回归因子分解将联合概率分布的估计转化为条件分布估计，并证明在满足次可加性的损失函数下，自回归估计器的收敛速度界可由条件分布估计器的收敛速度界导出，从而将联合分布估计的难度归结为自回归分解中最难因子的估计难度，并借助并行梯度提升实现高效估计。 |
| [^29] | [Piano2Pop: Expanding Piano Demos into Pop Productions Without Aligned Supervision](https://arxiv.org/abs/2610.31510) | 该论文提出Piano2Pop任务，通过伪监督的流匹配模型将钢琴MIDI扩展为完整的器乐流行音乐制作，无需钢琴与流行音乐的对齐配对监督，且微调后的模型在真实数据上显著优于零样本基线。 |
| [^30] | [Structure-preserving reduced-order models of thin panels in supersonic and hypersonic flow](https://arxiv.org/abs/2610.31504) | 本文为AePW-4高速工作组的两个挑战问题（马赫1.92固支薄板与马赫5.85悬臂板）构建了结构保持的降阶模型，其学习模型在构造上严格保证能量平衡（最差相对残差仅2.2e-15），并准确复现了激波压力比1.521（与压敏漆实测一致）和主频（偏差小于1.0%）。 |
| [^31] | [SABLE: Scalable Affine Bayesian Latent-space Exploration](https://arxiv.org/abs/2610.31495) | SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。 |
| [^32] | [Training Small Language Models for Transferable Tool Selection](https://arxiv.org/abs/2610.31483) | 本文通过强化学习训练8B小模型完成工具选择任务，性能超越402B的Llama-4-Maverick，并可直接迁移至不同检索器和跨领域的新工具语料库。 |
| [^33] | [Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon](https://arxiv.org/abs/2610.31479) | 该论文系统评估了KV缓存剪枝、投机解码、移植GPU内核和训练后量化四类大语言模型推理优化在苹果芯片上的适用性，发现其表现与服务器级NVIDIA GPU上显著不同——KV缓存剪枝在苹果芯片上可将端到端吞吐量提升至多1.75倍，而投机解码反而会降低吞吐量。 |
| [^34] | [MoG-VLM: Efficient Video Anomaly Detection through Motion-Guided Vision-Language Models](https://arxiv.org/abs/2610.31478) | 该论文提出MoG-VLM框架，将时间梯度幅度（TGM）作为轻量级运动控制信号进行内容感知采样，并仅在高异常可能性的帧附近激活密集的多帧VLM推理，从而在保持检测精度的同时显著提升视频异常检测的实时效率。 |
| [^35] | [RheOPT: Few-Shot Non-Newtonian Flow Prediction via Newtonian Pretraining of RheOFormer](https://arxiv.org/abs/2610.31465) | RheOPT通过在廉价的牛顿流体模拟上预训练变换器神经算子、再用少量非牛顿样本微调，并重用几何条件化的输运结构而非本构响应本身，实现了数据高效的少样本非牛顿流体预测。 |
| [^36] | [Multi-Token Prediction and Alternative Tokenization Improve Learning Latent Multi-Hop Graph Search at Scale](https://arxiv.org/abs/2610.31439) | 本文指出模型难以学习大规模图上最短路径搜索的两个原因——监督信号与分词方式，并提出多Token预测和替代分词这两种相互正交的方法，突破了以往工作的规模瓶颈。 |
| [^37] | [DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels](https://arxiv.org/abs/2610.31438) | 提出DIAL框架，利用无分类器引导的在线策略蒸馏和轻量级引导适配器，对大语言模型遵循指令的程度实现细粒度数值控制，同时兼顾效率与跨指令的可扩展性。 |
| [^38] | [Conservative Meta-Agent Search: Architecture Adaptation under Non-Stationarity](https://arxiv.org/abs/2610.31436) | 本文提出保守元智能体搜索（CMAS），将非平稳环境下的元智能体架构搜索形式化为带切换成本与硬风险约束的序贯现任者替换问题，通过配对影子回放评分、方差惩罚晋升裕度、金丝雀验证与回滚机制实现安全稳健的架构自适应，并配套引入MetaInvest-Bench状态切换模拟基准。 |
| [^39] | [Drift Says When, Budget Says How Much: Rate-Controlled KV Re-selection for Long-Form Multimodal Generation](https://arxiv.org/abs/2610.31433) | 提出速率受控刷新（RCR）方法，用注意力漂移决定KV缓存视觉token何时重选、用逐样本控制器调节刷新频率，从而显著提升多模态大模型长篇生成的质量。 |
| [^40] | [Dimension-free regimes for unadjusted Langevin in ridge-separable models: A subspace robust Wasserstein analysis](https://arxiv.org/abs/2610.31431) | 该论文利用子空间鲁棒Wasserstein距离分析未调整朗之万算法在岭可分模型中的偏差，证明在过参数化和大样本区间下算法复杂度可摆脱对维度的依赖。 |
| [^41] | [Predictable performance is suboptimal, and what it means for model routing](https://arxiv.org/abs/2610.31428) | 本文证明性能预测只有在候选模型本身表现次优或预测器具有显著更强能力时才能超越模型自身的置信度基线，且预测器的优势可通过重新校准在对数损失上被精确转移，这对模型路由方法的设计具有重要启示。 |
| [^42] | [How Depth Growth and Looping Shape In-Context Learning in Language Models](https://arxiv.org/abs/2610.31421) | 深度增长与循环语言模型通过跨层复用计算形成了偏向从上下文学习的归纳偏置，使其比稠密模型更早获得且更强保持上下文学习能力，其机制在于深度增长将早期归纳头传播到各层。 |
| [^43] | [Post-Hoc Feature Selection Layer for Neural Networks Interpretability](https://arxiv.org/abs/2610.31420) | 提出了一种事后特征选择层，可在完全不改变已部署模型架构和参数的前提下，以无侵入的方式识别出模型认为最重要的特征，从而提升复杂神经网络的可解释性。 |
| [^44] | [SQuAT: Spherical Lattice Vector Quantization-Aware Training](https://arxiv.org/abs/2610.31412) | 该论文提出SQuAT方法，通过分离的直通估计器将权重分解为方向（用E_8球面码本量化）和增益（用低比特标量量化器量化），将基于格的向量量化感知训练成功扩展到十亿参数级大语言模型，并实现了隐式的球面（黎曼）优化。 |
| [^45] | [HERD: Harness-Evolving Recursive Distillation for Continual Learning](https://arxiv.org/abs/2610.31401) | 提出HERD混合持续学习框架，通过可演化的工具链高效学习，并以双锚点漂移约束蒸馏将经验内化为参数化累积能力，从而在持续获取新知识的同时大幅减少遗忘。 |
| [^46] | [Convex Equilibrium Potentials for Monge--Amp\`ere: Curvature Bounds and A Posteriori Evaluation](https://arxiv.org/abs/2610.31386) | 本文针对用于Monge–Ampère方程的凸softplus平衡神经网络，证明了Hessian矩阵关于压缩因子具有最优的 $(1-q)^{-2}$ 增长界，并给出了基于残差的后验估计以控制近似平衡点处的状态及其导数误差。 |
| [^47] | [AfterDeleteBench: Auditing Post-Deletion Information Persistence in LLM Agents](https://arxiv.org/abs/2610.31366) | 本文提出AfterDeleteBench受控审计基准，揭示了LLM智能体删除原始记忆记录后，其派生状态（如摘要、分散存储的组件）仍可在约三分之二的案例中导致已删除的秘密信息被恢复，证明“删除”操作并不能保证信息真正被移除。 |
| [^48] | [Graph-Recurrent Learning of Emergent Collective Dynamics](https://arxiv.org/abs/2610.31350) | 该论文提出了一种将门控循环记忆与图消息传递相结合的图-循环神经网络（GRNN）框架，能够在无需已知微观运动方程的情况下直接从轨迹中学习关联电子系统的涌现集体动力学，并在半经典 Holstein 模型的电荷密度波动力学上成功再现了时空关联等动力学行为。 |
| [^49] | [Metric Conformal Prediction via the Conditional Expected Local Radius](https://arxiv.org/abs/2610.31336) | 该论文提出条件期望局部半径（CELR）这一新度量，构建了适用于一般度量空间中非欧几里得响应的保形预测方法，结合度量分布随机森林实现高效计算，并从理论上证明了估计量一致性、渐近条件有效性及有限样本边际覆盖率。 |
| [^50] | [Harmful Responses Are Jailbreak Templates: Transferring Autoregressive Jailbreaks to Diffusion Language Models](https://arxiv.org/abs/2610.31331) | 本文发现有害回复经部分掩码后可天然充当越狱模板，并据此提出RMT-Jailbreak框架，将现有的自回归模型越狱攻击有效迁移到扩散语言模型上。 |
| [^51] | [Deep Learning for Random Objects in Metric Spaces: Beyond Manifold Assumptions](https://arxiv.org/abs/2610.31328) | 该论文提出了一个直接在响应空间度量结构上运行的随机对象回归深度学习框架，无需流形或欧几里得嵌入假设，同时具备统计一致性与收敛速率保证。 |
| [^52] | [AgenticEvolve: Evolutionary Program Search with Persistent Language Model Agents that Design and Run Experiments](https://arxiv.org/abs/2610.31326) | AgenticEvolve让持久化的大语言模型智能体在进化程序搜索的评估间隙自主设计并运行实验，从而以更少的评估次数和相当的成本生成显著更优的程序。 |
| [^53] | [Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion](https://arxiv.org/abs/2610.31323) | 提出 Meta-Diff，通过在紧凑任务空间中一次性学习可复用的源-目标关系，使共享扩散主干无需针对每个目标域再训练即可从稀疏目标数据快速生成，尤其适用于医疗等目标数据难以获取的场景。 |
| [^54] | [The Value of System 1 Models in Medicine](https://arxiv.org/abs/2610.31315) | 本研究首次大规模评估了四个系统1模型在20种语言、14个临床领域共86个任务上的零样本表现，表明这类能快速输出概率并具备不确定性估计能力的模型，在临床决策中具有独特价值。 |
| [^55] | [Identifiability from Stationary Covariance in Linear SDEs with Reciprocal Interactions and Latent Variables](https://arxiv.org/abs/2610.31310) | 该论文为允许互惠相互作用（即因果图中含 有向2-环）的线性随机微分方程首次建立了基于平稳协方差的因果效应可识别性图判定条件，并进一步提出工具变量框架，解决了存在隐变量时部分可观测条件下目标直接因果效应的识别问题。 |
| [^56] | [Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization](https://arxiv.org/abs/2610.31309) | 提出OPTIC方法，通过为每个token仅存储一个共享量化编码元组并用层特定仿射解码器在每层解码，实现对逐层嵌入约100倍的极致压缩，同时保留未压缩模型97.8%的性能，远超传统量化方法。 |
| [^57] | [The Bayesian Learning Rule Beyond KL Geometry](https://arxiv.org/abs/2610.31308) | 本文提出了一类混合空间Bregman/Fenchel-Young差异度量，据此推导出广义贝叶斯学习规则（BLR），突破了KL几何的限制，并能够导出KL散度无法得到的新型加性优化算法。 |
| [^58] | [Data-Driven Vision-Language Models for Remote Sensing: A Survey](https://arxiv.org/abs/2610.31304) | 本文是一篇以数据为中心的遥感视觉-语言模型综述，系统梳理了高质量遥感视觉-语言数据集的构建方法，并指导研究人员进行有效的模型微调与性能评估，填补了该领域缺乏系统性数据导向调研的空白。 |
| [^59] | [DISC: Domain-Integrated Similarity Cost for Time-Series Dataset](https://arxiv.org/abs/2610.31288) | 提出了DISC——一个可学习的相似性度量聚合框架，利用校准迁移观测为不同下游任务自动选择、加权和组合多种相似性度量，使源-目标相似性得分与下游预测误差的关联更加一致，从而无需穷举评估即可实现数据集迁移性比较。 |
| [^60] | [Beyond Accuracy: Disentangling Decomposition from Typed-State Quality Assurance in Multi-Agent LLM Pipelines](https://arxiv.org/abs/2610.31280) | 该研究发现多智能体LLM流水线在需求分类与优先级排序任务中的性能提升主要源于任务分解所支持的专门化提示，而非类型化状态质量保障机制。 |
| [^61] | [Learning Robot Behavior Priors from Non-Robotic Synthetic Data](https://arxiv.org/abs/2610.31275) | 提出三种基于机器人控制数学抽象的合成数据生成管线，证明仅使用廉价的非机器人程序化合成数据进行预训练，就能让 Transformer 模型在机器人动作预测任务上获得非平凡的零样本性能。 |
| [^62] | [Deep ReLU Networks Can Fail the Best Approximation Property for Continuous Target Functions](https://arxiv.org/abs/2610.31264) | 本文通过构造连续函数反例证明，深度ReLU网络在一般情形下不满足最佳逼近性质，这与浅层网络中ReLU能保证全局极小值存在的已知结论形成鲜明对比。 |
| [^63] | [Can Entanglement-Induced Feature-Map Designs Improve Projected Quantum Kernels?](https://arxiv.org/abs/2610.31260) | 本文提出利用双量子比特纠缠的新型量子特征映射来构建投影量子核以捕获物联网数据的多变量相关性，并通过未调优与调优PQK对比RBF基线来分离评估其固有归纳偏置与结构表征能力。 |
| [^64] | [QuaNAS: Uncertainty-Aware Reuse of Earlier Searches for Query-Efficient Neural Architecture Search](https://arxiv.org/abs/2610.31251) | QuaNAS将以往搜索的“设计决策主效应”和零成本代理指标转化为候选架构的不确定性先验，并利用高斯过程在每次评估后动态校准对各先验的信任程度，从而在完全不使用新数据集标签的情况下，将查询高效神经架构搜索的任意时刻遗憾降低2.7至3.9倍。 |
| [^65] | [Muting Tokens: Exploring Token Sparsification for Efficient Audio Transformers](https://arxiv.org/abs/2610.31235) | 本文对音频Transformer中的令牌稀疏化技术进行了全面评估，并提出了一种将注意力分数与频谱线索相结合的新型稀疏化方法，在多种数据集和模型规模下实现效率提升的同时保持性能。 |
| [^66] | [Local Discrimination of Binary Classifiers via Localized AUC Functional](https://arxiv.org/abs/2610.31216) | 本文提出局部化AUC（LAUC）指标及LOCATOR方法，能够识别风险预测模型表现不佳的患者队列，从而克服传统AUC评估忽略性能异质性、可能导致临床决策失误的局限。 |
| [^67] | [Evolving Federated Density Peak Clustering with Shuffler-Based Anonymization](https://arxiv.org/abs/2610.31211) | 该论文提出了eFDPC方法，将演化模糊数据流聚类与密度峰值聚类相结合，并借助混洗器匿名化技术，在仅一轮通信的单次联邦学习范式中实现聚类，同时保证客户端数据的不可链接性与隐私安全。 |
| [^68] | [Inside-JEPA: From Prediction to Relational Grounding in Latent World Models](https://arxiv.org/abs/2610.31194) | Inside-JEPA将有序实体对的关系通道嵌入JEPA潜在状态内部，并利用逆动力学损失在无需关系标签的情况下实现关系接地，在多个数据集上显著提升了关系预测性能。 |
| [^69] | [When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions](https://arxiv.org/abs/2610.31193) | 该研究通过两阶段实验发现，元数据数量本身并不能提升大语言模型生成博物馆文物描述的可靠性，真正决定模型是接地、省略还是凭空推断信息的是字段类型，而图像既能支持视觉接地也可能助长无依据的解读。 |
| [^70] | [RaSFNet: Ray-aligned and Scan-aware Fusion Network for LiDAR Semantic Segmentation](https://arxiv.org/abs/2610.31186) | 该论文提出RaSFNet，将点云几何信息与激光雷达扫描特性融入基于投影的语义分割，通过光线对齐几何分布编码与扫描感知机制，解决了投影聚合掩盖视锥内几何差异、跨扫描区域空间关系建模受限以及远近特征混淆降低语义判别能力的问题。 |
| [^71] | [Track finding and particle identification performance with particle physics foundation models](https://arxiv.org/abs/2610.31180) | 该论文在时间投影室数据集上评估了粒子物理基础模型在径迹寻找和粒子鉴别等下游任务中的表现，证明基于基础模型的数据分析方法性能显著优于传统方法，且可灵活适应新任务。 |
| [^72] | [Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness](https://arxiv.org/abs/2610.31172) | 该研究通过涵盖9种提示策略、4个开源评判模型、27种语言的大规模基准测试（超过100万条评分），首次发现LLM-as-a-Judge评估中的提示策略效果不可跨模型和语言迁移——评判模型的选择比提示策略的选择更具决定性，且策略效果在不同评判模型间会反向。 |
| [^73] | [Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation](https://arxiv.org/abs/2610.31166) | 提出物体引导的感知记忆机制WholeBodyMem，利用SAM3跟踪的物体事件因果选择关键帧，使人形机器人全身VLA策略能够记住行动前的物体历史，从而在非马尔可夫动态场景中大幅提升成功率（仿真中29.1%提升至72.6%，真实机器人上36.7%提升至88.3%），并发布了HuMeGo人形机器人记忆基准。 |
| [^74] | [Prediction Uncertainty Controlled Computer Model Calibration](https://arxiv.org/abs/2610.31163) | 该论文提出一个两层框架，将条件流匹配摊销后验推断与共形风险控制相结合，直接控制校准后模拟器的预测覆盖率，即使在校准参数弱识别或模型差异被误设的情况下也能保证名义的预测区间水平。 |
| [^75] | [Towards Field-Deployable Wearable IMU Systems for Occupational Activity Recognition](https://arxiv.org/abs/2610.31160) | 该研究首次对VTT-ConIoT数据集进行系统的配置敏感性分析，发现即使在实验室数据上，受试者独立（LOSO）评估下可穿戴IMU活动识别的个体准确率差异高达27个百分点，揭示了以往总体高准确率报告掩盖了实际现场部署面临的性能挑战。 |
| [^76] | [Prompt Sensitivity in Tool-Augmented Model Comparisons](https://arxiv.org/abs/2610.31148) | 论文提出一种将答案读取、推理、调用决策等环节分离的匹配协议，发现工具虽能提升模型准确率，却会加剧模型间比较对问题表述方式的敏感性。 |
| [^77] | [PriMa-Causa: Causal Prescriptive Maintenance](https://arxiv.org/abs/2610.31147) | 论文指出处方式维护中纠正措施的选择本质上需要干预性因果量而非关联性预测，观测数据训练的预测模型会误导维护决策，因此应将因果推理嵌入处方式维护架构。 |
| [^78] | [GM-GReFEL: Geometry-Guided Spatiotemporal State-Space Learning for Robust Dynamic Facial Expression Recognition](https://arxiv.org/abs/2610.31140) | 提出GM-GReFEL模型，通过线性复杂度的时空状态空间骨干网络与基于面部几何锚点的可靠性决策模块，同时解决了动态面部表情识别中自注意力计算开销大和众包标签噪声两大难题。 |
| [^79] | [Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering](https://arxiv.org/abs/2610.31135) | 本研究通过在干扰文档中移动黄金证据，系统比较了Transformer、状态空间和混合Mamba架构在长上下文问答中的证据位置偏差，发现Transformer模型具有状态空间模型所缺乏的开头优势，而两类架构均偏好接近末尾的证据。 |
| [^80] | [Is High Local Accuracy Enough? Geometric Consensus with Support-Adaptive Statistical Personalization for Federated Learning](https://arxiv.org/abs/2610.31133) | 提出FedGRC统一个性化联邦学习框架及配套评估协议，通过几何共识与支持自适应统计个性化，在提升最差客户端个性化下限的同时，增强模型对全局、稀有及未见类别的识别能力。 |
| [^81] | [Post-Audit Mirage: Identifiability Limits of Offline Agent Verification](https://arxiv.org/abs/2610.31130) | 该论文证明了离线智能体验证存在根本性的可识别性极限——安全与不安全的智能体生命周期在离线审计中可能完全无法区分——并通过可执行的基准测试与基于组标记危害结果的监控方法，刻画了监控在何种条件下能够或无法恢复部署后缺失的安全性区分。 |
| [^82] | [Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory](https://arxiv.org/abs/2610.31125) | 提出Delta N-gram方法，通过冻结预训练N-gram表、将证据以仅前向传播的方式写入稀疏补丁Δ，实现无需反向传播的持续知识写入，清除补丁即可完全恢复原始模型，且事实保留效果远优于基于梯度的写入方法。 |
| [^83] | [What Comes Next and Why: Interpretable Next-Event Prediction with Neuro-Symbolic Rules](https://arxiv.org/abs/2610.31114) | 提出了神经符号模型RULES及其与残差网络结合的RUNES框架，通过从数据中直接学习可解释的序列规则，在下一事件预测中同时实现了有竞争力的准确率和透明的可解释性。 |
| [^84] | [DRAFT: Decoding and Reconstructing 3D Room Layouts from fMRI and Camera Trajectories](https://arxiv.org/abs/2610.31108) | 该论文首次提出仅凭相机轨迹和被试的fMRI信号（无需任何视觉帧）重建未见场景的3D房间布局，通过从早期视觉皮层解码紧凑布局表示并与相机运动推导的几何自由空间融合来实现这一目标，同时提出了布局r这一新评估指标。 |
| [^85] | [Transformers with CoT Discover Heavy-Ball Acceleration Through Training](https://arxiv.org/abs/2610.31089) | 本文证明Transformer在梯度下降教师的初始监督下，通过真实回归权重的轨迹级监督训练，能够自发学会实现Heavy-Ball动量加速方法，超越对教师算法的单纯模仿并获得加速效果。 |
| [^86] | [nnInteractive Agent: General-Purpose AI Agents as 3D Image Annotators](https://arxiv.org/abs/2610.31080) | 该论文提出开源的nninteractive-agent系统，让通用AI智能体无需额外训练即可根据自然语言指令自动完成3D图像标注的全流程工作（定位目标、提示分割、检查与修正结果），通过将nnInteractive封装为MCP工具实现了交互式分割的全自动化。 |
| [^87] | [When Forgotten Facts Return: Relearning-Aware Unlearning of Knowledge Graph Embeddings via Tangent Kernels](https://arxiv.org/abs/2610.31075) | 本文首次研究了知识图谱嵌入中被遗忘事实在持续训练下“复活”的再学习现象，并提出利用切向核建模这一恢复过程的核预期遗忘方法，使遗忘效果在持续训练中更加稳健。 |
| [^88] | [Federated Differentially Private Optimization via Low-Rank Preconditioning](https://arxiv.org/abs/2610.31069) | 本文提出一种联邦DP-PCA框架，通过低秩预条件化以线性通信成本实现中心化差分隐私保证，从而高效解决联邦差分隐私优化问题。 |
| [^89] | [NN-Based Transistor Modeling for Ultra-Fast SPICE Simulation and Standard Cell Library Characterization](https://arxiv.org/abs/2610.31024) | 本文开发了能精确预测3nm纳米片晶体管漏极电流和电荷的神经网络模型，并首次在涉及数千次SPICE仿真的标准单元库表征这一复杂EDA流程中，严格验证了其相比工业标准模型的精度与超快速仿真能力。 |
| [^90] | [Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters](https://arxiv.org/abs/2610.31017) | 开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。 |
| [^91] | [XALIGN: A Human-centric Visual Analytics Approach to Understanding AI Explanations](https://arxiv.org/abs/2610.31015) | XALIGN通过并排展示同一随机森林模型的基于规则的解释与多种特征归因方法（TreeSHAP、KernelSHAP、LIME），使用户能够直观发现各解释方法之间的一致与矛盾，并结合领域知识核验模型预测。 |
| [^92] | [EEG-Captioner: Grounding EEG-to-Language Generation in Fundamental Signal Properties](https://arxiv.org/abs/2610.30995) | 提出了EEG-Captioner统一框架及包含16万余个片段、以信号级问答标注的EEG-语言数据集，通过让模型回答关于频谱、空间、时间、侧向等可测量信号属性的问题，来验证EEG到语言生成确实建立在对脑电信号的真实解读之上，并采用双域编码器联合建模时间波形与频谱-空间表示。 |
| [^93] | [Diagnostic Loss: Asymmetric Supervision for Radiology Report Generation](https://arxiv.org/abs/2610.30980) | 提出诊断损失（DLoss），一种直接作用于解码器词元概率的可微分、架构无关的非对称监督目标，解决医学视觉-语言模型在放射学报告生成中遗漏关键诊断、引入错误诊断的目标错位问题。 |
| [^94] | [Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters](https://arxiv.org/abs/2610.30970) | 在闭集决策任务上，采用重排序器初始化的逐选项评分小型编码器（1700万至10亿参数）在精度上超越了联合评分和大语言模型提示，而延迟至多只增加一倍。 |
| [^95] | [Energy and Latency First: How Deployment Constraints Reverse Time-Series Model Selection](https://arxiv.org/abs/2610.30965) | 本文以蜂窝网络切换预测为例，证明在硬性40毫秒时延截止与设备全生命周期累计能耗约束下，按尾部截止期限可行性筛选并以包含特征计算的实测推理能耗排名，会逆转传统按准确率的时间序列模型选择，使基于手工时间特征的梯度提升树胜出。 |
| [^96] | [ReDence: A Hybrid Agentic and Decision-Model Framework for Rapid, Evidence-Grounded Detection of Modern Social Engineering Scams](https://arxiv.org/abs/2610.30963) | 该论文提出开源混合框架ReDence，通过微调多模态智能体“调查员”与专用“决策器”模型协同工作，实现对社会工程学诈骗的快速、自适应、基于证据的自动化检测。 |
| [^97] | [Doubly Robust Distributionally Robust Offline Contextual Pricing](https://arxiv.org/abs/2610.30957) | 本文针对连续定价场景下的分布式鲁棒离线策略评估与学习，提出了一个双重稳健框架，通过局部化DR估计器克服了基于IPW方法对倾向得分估计收敛速度的敏感性以及最坏情况期望的计算难题。 |
| [^98] | [Ground-Truth-First Benchmarking of Reinforcement Learning via Converse Optimality: From Smooth, Fully Observed Systems to Non-Smooth or Partially Observed Tasks](https://arxiv.org/abs/2610.30954) | 该论文提出基于逆最优性的“真值优先”强化学习基准测试框架，通过预先构建并保密经认证的最优值函数与策略，使算法性能可通过最优性差距、后悔值和策略误差等指标进行绝对化评估，并同时支持对任务变化敏感性的系统性分析。 |
| [^99] | [OpenAttention: One Percent Is Enough with Tail Proxies](https://arxiv.org/abs/2610.30951) | OpenAttention通过将整个未读尾部压缩为单个计数感知均值代理token，在读取率低至1%时仍能获得比丢弃或采样更低误差的稀疏注意力输出。 |
| [^100] | [Random Initialization as a Baseline for Time Series Foundation Models](https://arxiv.org/abs/2610.30950) | 本研究通过系统对比发现，时间序列基础模型的预训练权重相比架构相同的随机初始化模型仅带来平均2.58个百分点的准确率提升，表明预训练收益有限，随机初始化可作为评估此类模型的有效基线。 |
| [^101] | [Random Projections for Spectral Algorithms: Optimal Rates in Sobolev Norms](https://arxiv.org/abs/2610.30948) | 本文利用RKHS嵌入性质与算子相似性的精细分析，证明了基于随机投影（随机化草图与Nyström子采样）的谱算法在一般条件下于Sobolev范数中达到匹配极小极大下界的最优学习速率，且不牺牲统计效率。 |
| [^102] | [GateFormer: Forecast-Space Decomposition for Intra-Hour Solar Irradiance Prediction from Sky Images](https://arxiv.org/abs/2610.30942) | GateFormer提出了一种以GHI为条件的双解码器视觉Transformer，通过将观测辐照度嵌入视觉token流、进行预测空间分解并在预测空间中自适应加权融合轨迹与增量两类预测，实现了基于天空图像的更精准小时级内太阳辐照度预测。 |
| [^103] | [PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn](https://arxiv.org/abs/2610.30941) | LinkedIn提出PAIR检索框架，通过建模用户意图偏好而非个人资料相似度，解决了专业网络中跨角色互补匹配的人物推荐问题。 |
| [^104] | [Vision Transformers for Estimating Irradiance Using Data Scarce Sky Images](https://arxiv.org/abs/2610.30940) | 该论文提出一种面向数据稀缺场景的双框架视觉Transformer方法，通过整合计算大气参数（地外辐照度与周期性时间编码）并以顺序预测的合成GHI作为特征，无需昂贵测量仪器即可从天空图像准确估计散射水平辐照度。 |
| [^105] | [Bridging Number-Theoretic Primality Testing And Algorithmic Optimization: An Adaptive Cryptographic Framework](https://arxiv.org/abs/2610.30935) | 该论文提出了一个将经典数论与机器学习模型及优化算法相结合的自适应任意精度素性检测框架，以满足RSA等公钥密码系统对大素数生成、检测与证明的核心需求。 |
| [^106] | [MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors via Gradient Decomposition](https://arxiv.org/abs/2610.30914) | MIG-COW通过将动量积分梯度与共识正交分解相结合，捕获CNN与ViT检测器的共享及特有漏洞，在AADD-2025挑战赛中实现了99.96%的白盒攻击成功率并显著优于现有基线，同时揭示了黑盒可迁移性仍是待解决的挑战。 |
| [^107] | [Same molecule, Different Knowledge: Cross-Identifier Binding Failures in LLMs For chemistry](https://arxiv.org/abs/2610.30903) | 该论文通过受控基准和行为、表征、因果分析，首次揭示并验证了大语言模型中“跨标识符绑定失效”现象，即同一分子的俗名、IUPAC名称和SMILES等互指标识符所激活的模型知识子集并不互通。 |
| [^108] | [Parameter Learning in Active Inference via Variational Message Passing](https://arxiv.org/abs/2610.30899) | 该论文将主动推理中的期望自由能最小化统一到变分消息传递框架下，实现了基于狄利克雷先验的在线参数学习与规划的融合，并通过为每个因子分配独立参数先验解决了传统平均场分解下信息增益校正消失的关键难题。 |
| [^109] | [Equilibria and Learning in Competitive Market Making](https://arxiv.org/abs/2610.30891) | 本文提出了多策略做市商与代表性交易者之间的做市博弈模型，证明了纯策略纳什均衡在极弱假设下总是存在且买卖价差随波动率严格变宽，并进一步证明当报价单位趋于零时无内部遗憾学习者的时间平均策略收敛于连续纳什均衡价格包络。 |
| [^110] | [Continuous-depth Deep Gaussian Processes](https://arxiv.org/abs/2610.30890) | 提出将深度高斯过程的深度视为由随机微分方程刻画的连续演化，并在深度索引的漂移函数族上施加高斯过程先验（CDGP 与 FlowDGP 两种实现），从而缓解传统离散深度 DGP 随深度增长而难以优化的问题。 |
| [^111] | [Building Presentation Intelligence for High-Quality Slide Generation](https://arxiv.org/abs/2610.30888) | 该论文提出了一个面向演示智能的统一训练框架，通过交叉验证的数据合成、基于规则与先验知识的教师增强以及可验证奖励优化，实现高质量的自动PPT生成。 |
| [^112] | [Tracing Subliminal Learning Through Pretraining](https://arxiv.org/abs/2610.30884) | 该研究证明“潜意识学习”现象在硬标签蒸馏和同策略软标签蒸馏中均普遍存在，且不受权重初始化、数据顺序、语料库、模型规模和架构的影响，并首次发现传递偏好与获得偏好的能力在预训练过程中独立发展，导致教师与学生角色互换时迁移强度呈现强烈不对称。 |
| [^113] | [Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models](https://arxiv.org/abs/2610.30882) | 研究揭示推理模型中的{wait}标记并非由强化学习发明，而是从训练数据（中期训练与推理轨迹微调）中继承而来，RL只能放大模型已有的使用习惯。 |
| [^114] | [One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation](https://arxiv.org/abs/2610.30880) | 本文发现在线策略自蒸馏中单一完整参考条件会导致教师分布对参考的选择高度敏感（参考诱导的目标不稳定性），并提出蒙特卡洛特权蒸馏方法，通过利用参考的多个视角来降低这种参考敏感性。 |
| [^115] | [MOER: Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback](https://arxiv.org/abs/2610.30873) | MOER提出一种混合策略优化框架，通过逐步衰减的专家思维链前缀引导并配合选择性回滚机制保证离策略轨迹正确性，协调了专家引导与模型自我探索。 |
| [^116] | [Joules-to-GDP: How Much Economic Value Do AI Systems Produce per Unit of Energy?](https://arxiv.org/abs/2610.30858) | 该论文提出“每焦耳净价值（NVJ）”和“每焦耳时间节省（TSJ）”两个新指标来衡量AI系统每消耗一单位能源所创造的经济价值，并发现AI表现最出色、创造价值或节省时间最多的职业领域并不一定带来最高的每焦耳经济回报。 |
| [^117] | [Training Mixed-Monotone Neural Networks for Verification Without Set Propagation](https://arxiv.org/abs/2610.30853) | 提出一种将混合单调函数实现为两个单调子网络之差的新型神经网络架构，把紧凑输出包络的计算从复杂的集合传播简化为两次简单前向传播，且因包络大小是权重的闭式函数，可在训练中直接最小化它。 |
| [^118] | [Bifurcations from Weight Space: JEPAWG on Parametric Dynamical Systems](https://arxiv.org/abs/2610.30847) | 该论文以分岔结构已知的参数化动力系统为真值基准，构建以物理参数索引的模型动物园，验证了 JEPAWG 超网络能够从训练权重的潜在几何中定位动力学发生质变的分岔位置。 |
| [^119] | [Reverse-KL Policy Distillation: Statistical Guarantees and Limits](https://arxiv.org/abs/2610.30844) | 该研究首次从理论上分析反向KL策略蒸馏的统计保证，证明当教师提供完整动作分布时在线蒸馏具有极小极大最优的样本复杂度，而仅观察采样动作概率或采用离线蒸馏则可能需要显著更多的数据。 |
| [^120] | [Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries](https://arxiv.org/abs/2610.30843) | Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。 |
| [^121] | [Semi-Linear Gaussian Bayesian Networks: Enhancing Flexibility in Node Dependencies via penalized B-Splines](https://arxiv.org/abs/2610.30838) | 本文提出了半线性高斯贝叶斯网络（SLGBN），通过带惩罚B样条的广义可加模型估计非线性条件均值函数，在保持高斯条件分布的同时捕捉非线性依赖关系，弥合了线性高斯模型灵活性不足与非参数方法计算成本高之间的空白。 |
| [^122] | [Neural Spectral Capacity with Residual Context for Heterogeneous Resource Allocation within Transformers](https://arxiv.org/abs/2610.30823) | 本文提出残差感知神经谱容量ResNSC及其动态规划分配器ResNSC-DP，通过在容量计算中引入残差历史统计信息，在无需数据或实例化权重的情况下，解决了原NSC方法独立评分、无法区分层配置重排序的问题，并能在资源预算下为Transformer联合分配残差和前馈宽度。 |
| [^123] | [xVGAE: A Hierarchical Variational Graph Autoencoder for Exchangeable Graphs](https://arxiv.org/abs/2610.30815) | 本文提出 xVGAE，一种受 Aldous-Hoover 定理启发的层次化变分图自编码器，在生成质量与最先进模型相当的同时，将生成速度提升了几个数量级。 |
| [^124] | [What Lies Beyond Perception: A Partition Theory of the Limits of AI Evaluation and Oversight](https://arxiv.org/abs/2610.30803) | 本文提出AI评估的划分理论，证明当被评估模型的感知划分比评估者更精细时，通过评估者所能评定的任何测试都最多只能认证模型辨别力的下限而非上限，从而揭示了AI评估与监督存在的根本性局限。 |
| [^125] | [Is spectral predictability enough? Testing {\Omega}-matching for synthetic pretraining under distribution shift](https://arxiv.org/abs/2610.30802) | 论文通过多架构实验发现，仅匹配谱可预测性Ω不足以指导分布偏移下的合成预训练——尽管在理想合成条件下Ω匹配预训练可超越oracle基线，但在真实世界基准上，不同类型的Ω匹配合成数据（正弦加噪声与混沌ODE轨迹）会产生截然相反甚至负面的迁移效果。 |
| [^126] | [A Computationally Efficient and Rate-Optimal Algorithm for Adversarial Linear MDPs with (Aggregate) Bandit Feedback](https://arxiv.org/abs/2610.30801) | 本文首次提出多项式时间算法，通过将对抗性线性MDP归约为动作集未知的对抗性线性老虎机并构建近似分离预言机，实现了速率最优且计算高效的 $\sqrt{K}$ 遗憾界。 |
| [^127] | [Amplifying Latency and Cost of LLM Inference via Rowhammer Attack](https://arxiv.org/abs/2610.30798) | 本文提出LLVMhammer——首个针对Triton编译流水线的Rowhammer攻击，通过诱导性能关键优化阶段的位翻转，生成功能不变但性能次优的GPU内核，并借助内核磁盘缓存机制造成LLM推理的持久性变慢，且该攻击与GPU厂商无关。 |
| [^128] | [GIFT: Gradient Informed Filtering for Distillation Augmented Reasoning RL](https://arxiv.org/abs/2610.30796) | GIFT通过在token级别对蒸馏梯度进行几何分解——将平行于RL梯度的分量作为安全过滤器以排除冲突更新、将正交分量作为新信息来源——为蒸馏与强化学习之间的目标冲突提供了一个有原则的解决框架。 |
| [^129] | [Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training](https://arxiv.org/abs/2610.30792) | Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。 |
| [^130] | [Gandalf: Flash Shall Not Die! Increasing Flash Storage Lifetime for KV Cache Offloading in LLM Serving](https://arxiv.org/abs/2610.30791) | 提出Gandalf系统，通过结合访问模式等信息智能门控KV缓存写入闪存，在不降低KV缓存命中率的前提下，显著延长LLM服务中作为KV缓存卸载介质的SSD使用寿命。 |
| [^131] | [One Pretrained Decision Model Is Enough for Training-Free Generalist Graph Anomaly Detection](https://arxiv.org/abs/2610.30777) | Jev-GAD是一个免训练的通才图异常检测框架，通过将结构、属性和邻域等异构数值证据转化为图相对异常百分位数，并利用单个冻结的预训练语言模型进行基于证据的异常评估，实现了跨图的可泛化异常检测。 |
| [^132] | [Adaptive Deadline- and Energy-Aware Caching Algorithms in Dynamic Multi-UAV Edge Networks](https://arxiv.org/abs/2610.30768) | 论文针对多无人机边缘网络提出了一个联合考虑截止时间错过、通信能耗与机间能耗不均衡的动态缓存放置与替换问题，并开发了可解释的分层规则算法DE-UBAC来实现截止时间与能耗感知的缓存管理。 |
| [^133] | [Primal-Dual Representation Learning for Low-Rank Constrained MDPs](https://arxiv.org/abs/2610.30756) | 提出REP-PD-TEG算法，通过衰减ε-贪婪策略探索、最大似然特征学习与乐观拉格朗日规划，为连续状态空间的低秩约束MDP实现表征学习，并提供了后悔度与软约束违反的理论保证。 |
| [^134] | [Visualizing and Interpreting Temporal Care Team Dynamics for Cancer Survival Risk Analysis](https://arxiv.org/abs/2610.30748) | 该论文提出了一个可视化分析框架，通过EHR通信数据构建动态协作网络并结合时间图神经网络（TGNN），建模肿瘤护理团队随时间变化的互动，并揭示团队组成与专科参与的变化同患者风险演变之间的关系。 |
| [^135] | [Wavenumber-Aware Deep Learning for Robust Analysis of Raman Spectra](https://arxiv.org/abs/2610.30746) | 本文提出将光谱点表示为强度-波数对并从波数本身计算Transformer位置编码，使深度学习模型在噪声、稀疏采样、部分光谱覆盖以及未见过的采样网格条件下均能稳健地分析拉曼光谱。 |
| [^136] | [Random Remasking Scales Masked Diffusion Inference](https://arxiv.org/abs/2610.30744) | 该论文将掩码扩散模型推理重新表述为干净序列空间上的马尔可夫链，提出了无需微调、无需奖励的随机重掩码-去噪（R²D）测试时推理方法及轨迹校正训练（TCT），通过随机重掩码迭代校正解码错误，在逻辑任务上实现了接近对数线性的测试时性能扩展。 |
| [^137] | [Width and Depth Limits Commute During Training](https://arxiv.org/abs/2610.30729) | 本文将残差网络无限宽与无限深极限的可交换性从初始化阶段首次扩展到训练过程，证明了在Depth-μP参数化与梯度裁剪下，线性残差网络在任意有限训练步数内，网络输出沿宽度-深度联合增长路径按 $L^2$ 收敛且两极限可交换。 |
| [^138] | [Federated Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2610.30728) | 提出联邦GraphRAG（FedGRAG）统一框架，通过联邦知识路由机制在保护隐私的前提下解决缺乏全局知识图谱的检索难题，实现多客户端间的分布式图检索与知识增强生成。 |
| [^139] | [Solvable models of learning to pursue a moving target](https://arxiv.org/abs/2610.30721) | 提出了一个可解析求解的连续时间线性二次追捕模型，其最优控制策略可在反应式与预测式追捕之间插值，并给出了最优策略与学习动力学的闭式解，学习时间尺度由特征值精确刻画。 |
| [^140] | [AIRO: A Streaming-Based Machine Learning Framework for Threat Detection and Security in Industrial Internet of Things](https://arxiv.org/abs/2610.30713) | 该论文提出了AIRO框架，通过融合Kafka流处理、攻击特征提取、机器学习推理和基于置信度的决策逻辑，实现对工业物联网环境中多种网络威胁的实时检测与自动化风险降低。 |
| [^141] | [An Agent Architecture with Tools and Memory for Selective Correction of Wrist PPG Heart Rate Estimates](https://arxiv.org/abs/2610.30688) | 该论文提出一种带工具与记忆的智能体架构，通过独立风险门控对手腕PPG心率估计进行选择性校正，在四个公开数据集上使平均绝对误差最多降低44.5%。 |
| [^142] | [Learning Domain Trajectories with Flow Matching for Gradual Domain Adaptation](https://arxiv.org/abs/2610.30685) | 提出FlowGDA框架，利用流匹配将渐进域自适应建模为表示空间统计流形上的连续输运，通过学习时间依赖的速度场生成平滑域轨迹，避免了传统离散逐步方法中的误差累积和不稳定的对抗训练。 |
| [^143] | [Scaling Laws for Safety Probes](https://arxiv.org/abs/2610.30678) | 本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。 |
| [^144] | [RiboBridge: protein-conditioned de novo RNA binder structure design](https://arxiv.org/abs/2610.30668) | RiboBridge提出了一种结构优先的框架，通过跨坐标系交互几何将RNA构象生成与蛋白质-RNA组装解耦，实现了无需预定义结合位点的蛋白质靶向RNA结合体从头设计。 |
| [^145] | [Fast Iterative Combinatorial Auctions via Deep Reinforcement Learning](https://arxiv.org/abs/2610.30667) | FastCombo通过离线深度强化学习为迭代组合拍卖学习序贯定价策略，以单次神经网络前向传播完成价格更新，从理论上克服了单步贪婪定价的局限，并显著减少拍卖出清所需的轮次。 |
| [^146] | [Interpreting Model Reasoning through Behavioral Counterfactuals](https://arxiv.org/abs/2610.30630) | 该论文提出基于行为反事实的方法，通过训练模块编辑残差流以最小化固定响应下输出分布的KL散度，从而定位并分析干预对模型推理的影响，且比标准激活重构目标更能有效恢复反事实行为。 |
| [^147] | [Future of Time Series Anomaly Detection: From Historical Review to Unified Outlook](https://arxiv.org/abs/2610.30628) | 本综述系统梳理了时间序列异常检测领域，建立了精细的异常分类体系，并构建了涵盖扩散模型、状态空间模型和大语言模型等新兴范式的结构化分类框架，同时对评估方法与指标可靠性进行了统一分析，为该领域的未来发展指明了方向。 |
| [^148] | [Evaluating Type-1 Diabetes Digital Twins by Decision Quality](https://arxiv.org/abs/2610.30577) | 本文提出不再以血糖轨迹预测精度、而是以下游治疗方案排序的决策质量来评估1型糖尿病数字孪生，并基于FDA认可的UVA/Padova模型中30个虚拟患者、21种基础-追加治疗方案建立了系统的评估协议。 |
| [^149] | [Bridging the Closed-World Gap: Interpretable Anomaly Detection via a Hybrid TSK-FCM Ensemble on the BETH Dataset](https://arxiv.org/abs/2610.30576) | 提出了一种将梯度优化TSK模糊分类器与FCM聚类模块协同结合的混合模糊集成方法，在BETH数据集上实现了既能精确检测已知内部威胁、又能发现新型几何异常的可解释异常检测，从而弥合监督与无监督学习之间的封闭世界差距。 |
| [^150] | [When News Gets Old: Portfolio Allocation with Reinforcement Learning and Large Language Models](https://arxiv.org/abs/2610.30564) | 研究发现在LLM驱动的投资组合配置中，过滤旧新闻比向策略提供时间信息更有效——当年新闻延迟四周交付时，过滤旧新闻可实现18.43%的年化净收益，并优于所有基于新闻的强化学习策略。 |
| [^151] | [Lifetime-valid decisions by continually evolving AI analysts](https://arxiv.org/abs/2610.30562) | 提出“持续验证”方法，在部署前使用可随机化和扩展的零假设模拟数据评估每个AI分析师版本，并证明无论更新如何选择，部署第一类错误率超标版本的终身概率始终有界，从而保障持续演化的AI分析师所有后续决策的终身有效性。 |
| [^152] | [Decision Making and Ride Comfort Evaluation Based on Lane-Change Prediction for Autonomous Vehicles](https://arxiv.org/abs/2610.30528) | 本研究提出将基于知识图谱嵌入和贝叶斯推断的换道预测方法集成到自动驾驶决策中，可提前3-4秒预判周围车辆的换道行为，相比反应式驾驶策略有效避免了碰撞并改善了乘坐舒适性。 |
| [^153] | [LLM2Cert: LLM Synthesis of Certificates with SMT/Lean Verification](https://arxiv.org/abs/2610.30524) | 提出LLM2Cert框架，利用大语言模型合成李雅普诺夫函数与障碍证书，并通过SMT求解器和Lean定理证明器进行严格的形式化验证，同时发布包含105个实例的DynaCertBench基准以覆盖高维非线性系统。 |
| [^154] | [NEST: Nested Embedding Representation for Multiscale Clustering](https://arxiv.org/abs/2610.30520) | 受俄罗斯套娃表示学习启发，NEST提出了一种无监督框架，在单一嵌入中编码从粗到细的多尺度聚类结构，无需在训练中观测层次信息即可恢复数据的多尺度聚类组织。 |
| [^155] | [Optimal Multitask Representations Under a Rate Constraint: An Information Bottleneck Analysis](https://arxiv.org/abs/2610.30493) | 该论文通过信息瓶颈框架分析信息预算约束下的最优多任务预训练表示，发现压缩程度会改变哪个预训练目标最优，且不存在在所有任务族上都表现一致的固定预测目标。 |
| [^156] | [FIGFLEX: A Mixed-Precision FP-INT GEMM Accelerator Co-Designed with Low-Bit LLM Quantization](https://arxiv.org/abs/2610.30480) | FIGFLEX是一种算法-硬件协同设计的混合精度FP-INT矩阵乘加速器，通过移位加法基-4乘法重构将部分积减半，使数据通路吞吐量随权重精度降低而扩展，从而让低比特LLM量化真正转化为计算性能的提升。 |
| [^157] | [Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy](https://arxiv.org/abs/2610.30474) | 本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。 |
| [^158] | [DiffLTM: Continuous Diffusions Large Tabular Models for Conditional Generation](https://arxiv.org/abs/2610.30472) | 本文提出DiffLTM——一种用于表格数据条件生成的连续扩散大型模型，通过为类别特征引入交叉熵损失并给出首个理论证明、设计针对稀有子群体的评估指标以及泛化条件生成技术，在混合数值与类别数据上实现了强大的生成质量。 |
| [^159] | [Learning to Incentivise: Using Reinforcement Learning for Sustainable Urban Mobility](https://arxiv.org/abs/2610.30469) | 本文提出一种多智能体强化学习方法，通过动态向驾驶员发放激励引导其改道，在可持续城市出行框架下同时将交通网络的总出行时间降低16%并减少总排放量。 |
| [^160] | [Flow-based Spectral Kernel Learning for Nonstationary Attention](https://arxiv.org/abs/2610.30465) | 提出基于流的谱核注意力（FSKA），利用归一化流参数化非平稳的二元谱密度来学习灵活且可正则化的注意力核，同时支持移除查询-键投影的简化设计，实现参数效率更高的线性注意力架构。 |
| [^161] | [Short-to-Long: Detached Features for Long-Horizon Value Learning](https://arxiv.org/abs/2610.30461) | 提出S2L方法，通过梯度隔离的分离连接，让较小折扣因子的源网络为长时程行为学习器提供特征，在不改变控制目标的前提下实现长时程价值学习，并在Atari与连续控制基准上取得整体性能提升。 |
| [^162] | [aggmetrics: Aggregating Classification Performance Measures for Machine Learning](https://arxiv.org/abs/2610.30456) | aggmetrics 是一个仅依赖 NumPy 的 Python 包，通过四种聚合算子将任何基于混淆矩阵的二分类性能度量推广到多分类、多标签、序数、代价敏感、软标签和交叉验证等场景，并利用度量的结构性质在其使用不当时发出警告。 |
| [^163] | [Same Beliefs, Different Policies: Learning Reusable Representations for Early Stopping in LLM Reasoning](https://arxiv.org/abs/2610.30450) | 该论文提出BELT方法，通过将结果预测与停止控制分离，从冻结LLM隐藏状态中学习与token价格无关的信念表示，使单一探针能通过轻量级策略重拟合或阈值调整支持多种准确率-成本权衡，从而有效实现推理提前停止并降低计算浪费。 |
| [^164] | [Stability Analysis of Signed Graphs under Structural Perturbations](https://arxiv.org/abs/2610.30444) | 本文首次系统研究了符号图滤波器在边添加、删除和符号翻转等结构扰动下的稳定性，揭示了符号边扰动与节点特征相关性对学习表示稳定性的联合决定作用，并据此提出了一种理论引导的扰动策略来选择结构性影响显著的符号边修改。 |
| [^165] | [Towards Faithful Data Selection: Statistical Limits and Scalable Algorithms](https://arxiv.org/abs/2610.30442) | 本文证明了在缺乏远大于训练集的验证集时实现最优数据选择在信息论上是不可能的，并据此提出“忠实性”这一新目标，通过可估计的代理量上界子集训练与全量训练的差异，并利用未调整朗之万算法和轨迹累积梯度差异对参数分布的KL散度进行约束。 |
| [^166] | [Memory-Augmented Online Test-Time Adaptation for EEG Motor Imagery via Perturbation Consistency and Confidence Calibration](https://arxiv.org/abs/2610.30422) | 该论文提出MOPC，一个无源的记忆增强在线测试时自适应框架，通过扰动一致性与置信度校准，使EEG脑机接口在推理阶段实时更新解码参数，无需访问源数据即可提升跨受试者场景下的可靠性。 |
| [^167] | [High-Value Training Data from Noisy, Diverse Sources](https://arxiv.org/abs/2610.30421) | 提出一种基于元梯度的可扩展数据加权方法，通过学习可泛化、可迁移的参数化加权规则，为奖励建模等任务从嘈杂多样的数据源中筛选出高价值训练数据。 |
| [^168] | [AcoustiPrint: Physics-Guided Neural Design of Manufacturable Acoustic Structures](https://arxiv.org/abs/2610.30411) | AcoustiPrint提出了一种物理信息神经架构，通过隐式神经表示三维几何并引入可打印性算子，在声学结构逆设计中确保优化结果直接满足增材制造的离散约束，弥合了物理优化几何与实际制造结果之间的差距。 |
| [^169] | [Escaping the Phylogenetic Trap: Persistent Homology Enables Cross-Species Ion Channel Classification via Multimodal Graph Neural Networks](https://arxiv.org/abs/2610.30394) | 该研究揭示随机数据划分导致的系统发育泄漏会高估毒素离子通道分类器性能，并提出融合ESM-2序列嵌入与二硫键感知图注意力网络的多模态框架，证明持续同调等结构归纳偏置能在留一分类群评估下实现真正的跨物种泛化。 |
| [^170] | [Soft Ontology Priors over Latent Concept Posteriors for Weakly Supervised Neuro-Symbolic Learning](https://arxiv.org/abs/2610.30391) | 该论文提出将潜在概念的软本体先验注入弱监督神经符号学习的训练目标，并与任务约束一起软编译为概率电路，从而消除推理捷径、使概念可辨识、对噪声专家公理具有鲁棒性，同时实现与编译电路规模呈线性关系的推理复杂度。 |
| [^171] | [A Transformer-based Approach for Real-time Multi-agent Path Planning with Varying Initial/Final Conditions](https://arxiv.org/abs/2610.30388) | 提出OCFormer，一种通过物理信息损失训练的基于Transformer的神经网络，将解耦潜空间映射到耦合物理空间，从而实现具有可变初始/最终条件的实时高维多智能体路径规划。 |
| [^172] | [Bayesian Fixed-Confidence Best-Arm Identification with Abstention](https://arxiv.org/abs/2610.30384) | 该论文提出了带弃权机制的贝叶斯固定置信度最优臂识别算法S&C，证明了其期望停止时间在高斯设置下达到匹配的上下界，揭示近乎平局的实例主导采样成本，并在无弃权时达到最优的 Θ(1/δ) 依赖。 |
| [^173] | [Probing and Steering for Safety-Relevant Behaviours in Large Language Models](https://arxiv.org/abs/2610.30363) | 该论文在统一流水线中系统探究了激活探针设计选择（估计器类型、token 池化方案、层选择）在检测与引导欺骗、高风险意识和谄媚三种安全相关行为时的相互作用，并验证了“逻辑回归最擅长检测、均值差方向最擅长引导”的读/写分离现象。 |
| [^174] | [CTRL-FM: Inference-Time Structural Control for Flow-Based Virtual Sensing under Long-Horizon Node Outages](https://arxiv.org/abs/2610.30362) | CTRL-FM提出了一个观测保持的流匹配框架，通过从存活数据构建自监督代理中断，在推理时自适应选择结构控制策略，以解决长时节点中断下虚拟感知中固定结构偏置导致的过平滑或误差传播问题。 |
| [^175] | [Route-and-Generate: KL-Regularized Router Improvement for Reasoning Routing in LLMs](https://arxiv.org/abs/2610.30351) | 本文提出Rouge分层训练框架，通过KL正则化的路由策略自适应分配推理预算，使大语言模型在简单问题上简洁作答、在难题上进行深入推理，从而优化准确率与效率之间的权衡。 |
| [^176] | [Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable](https://arxiv.org/abs/2610.30349) | 该论文提出一种自我奖励的强化学习方法，让模型通过自我评判道德论证的质量来提升道德判断能力，使模型变得更加果断且不受选项呈现顺序的影响，但谄媚性和重新框架问题仍然存在。 |
| [^177] | [On the Adversarial Vulnerability of Neural Receivers with Unrolled LDPC Decoders](https://arxiv.org/abs/2610.30345) | 本文揭示了神经接收机中展开式LDPC解码器因boxplus运算的tanh饱和产生梯度掩蔽、使标准对抗攻击失效，并提出两种结构独立的新损失函数（L_BP与L_LLR）成功恢复了攻击效力。 |
| [^178] | [AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining](https://arxiv.org/abs/2610.30344) | AQL-Lite 通过质量优先的数据筛选流水线与部分参数持续预训练方案（RS-LoRA 适配器结合最后八层 Transformer 块与嵌入矩阵），以约一半的训练时间恢复了全参数 CPT 94% 的性能提升，高效地将多语言大模型适配至现代标准阿拉伯语。 |
| [^179] | [Spike-Aware Gradient-Based Mixed-Precision Quantization of Spiking Neural Networks for Vision Applications](https://arxiv.org/abs/2610.30339) | 提出了一种脉冲感知的基于一阶梯度的混合精度量化框架，能够自适应地为脉冲神经网络分配位宽，避免了昂贵的二阶Hessian计算开销，从而在视觉应用的低精度部署中保持模型精度。 |
| [^180] | [TerraTransfer++: Demonstration-Free End-to-End Planner](https://arxiv.org/abs/2610.30319) | 该论文提出TerraTransfer++，通过利用向量化仿真器中低成本的自我博弈获得多样化的状态分布并预训练策略，再将其轨迹蒸馏到端到端规划器中，从而无需真实驾驶示范数据即可训练高性能的端到端自动驾驶规划器。 |
| [^181] | [SciFusion: Physics-Guided Compositional Diffusion for Multimodal 3D Scientific Image Generation](https://arxiv.org/abs/2610.30313) | SciFusion提出了一种基于潜在扩散的多模态3D科学图像生成框架，通过将模态分解为域、对象和对比度因子实现跨模态知识迁移与零样本组合生成，并引入MRI傅里叶频谱一致性、CT Radon投影一致性等物理信息目标来提升生成的物理合理性。 |
| [^182] | [EgoHand: Monocular 3D Hand Reconstruction from Egocentric Video](https://arxiv.org/abs/2610.30293) | EgoHand是基于1400万张图像高质量语料库训练的单目3D手部重建模型，在自我中心视频上实现了最先进的性能，具备强大的零样本跨域泛化能力和约9.69毫秒的低推理延迟。 |
| [^183] | [GAV: A scientific experimentation environment for PDE agents](https://arxiv.org/abs/2610.30292) | 该论文提出GAV框架，通过生成器构造可验证的新PDE问题、智能体迭代开发与调试程序、验证器提供数值反馈，为PDE编码智能体建立了一个无需昂贵参考计算的科学实验环境。 |
| [^184] | [Generalizing Metropolis-Hastings for DAG Sampling](https://arxiv.org/abs/2610.30291) | 本文提出了“二重采样”这一Metropolis-Hastings算法的新推广形式，通过在每次迭代中选取两个提议分布（一个生成候选状态、一个评估返回当前状态），实现了对DAG中多个节点父集的同时重采样，从而显著提升了贝叶斯网络结构学习中马尔可夫链的混合效率。 |
| [^185] | [SWE-VERIFIER: Selective Verification for Cost-Efficient Coding Agents](https://arxiv.org/abs/2610.30284) | 提出轻量级生成模型SWE-Verifier，通过选择性触发候选动作采样与验证，以仅1.09倍的成本将编程智能体的解决率从75.47%提升至77.20%。 |
| [^186] | [A Quadratic Lens on Muon's Polar Update: Benefits, Pitfalls, and the Role of Newton-Schulz](https://arxiv.org/abs/2610.30280) | 该论文在矩阵二次型问题上对Muon核心的PolarGD正交化更新进行闭式分析，揭示其收益（梯度远离低秩且最大奇异方向高曲率时有效、误差行正交等范数时可一步收敛、具有旋转不变性）与陷阱，并阐明了Newton-Schulz正交化的作用。 |
| [^187] | [Edit, Don't Imitate: Minimal-Edit On-Policy Distillation for Multimodal Reasoning](https://arxiv.org/abs/2610.30231) | 该论文提出最小编辑纠正性在线策略蒸馏方法，教师模型仅对学生错误采样中不认可的词元施加纠正损失、正确采样则用任务奖励强化学习，从而在测试时无需额外图形描述的情况下提升多模态数学推理能力。 |
| [^188] | [Automated screening of privacy risks in medical AI](https://arxiv.org/abs/2610.30229) | 该研究通过对7,789个医疗AI模型的患者级隐私风险分析，发现高风险患者排名在不同训练配置间基本保持稳定，从而可以在模型开发早期自动化识别和筛查易受成员推断攻击的脆弱患者记录。 |
| [^189] | [Benchmarking Model Compression Techniques Across Heterogeneous Edge Platforms: A Predictive Maintenance Use Case for Rotating Machinery](https://arxiv.org/abs/2610.30228) | 本文针对旋转机械预测性维护任务，对四种最先进的模型压缩技术在异构边缘平台上的性能表现进行了全面的基准测试与对比分析，为工业边缘部署中的压缩技术选择提供了实用指导。 |
| [^190] | [GANet: A Lightweight Group Attention Network for Plant Disease Identification on Embedded Systems](https://arxiv.org/abs/2610.30225) | 本文提出仅含0.001M参数的轻量级分组注意力网络GANet，可在嵌入式边缘设备上高效识别38种植物病害，并在PlantVillage数据集上达到99.99%的准确率。 |
| [^191] | [How Prompt Style Affects AI Accuracy and Speed in Competitive Programming](https://arxiv.org/abs/2610.30223) | 该研究通过在IOI准确性任务与NP难速度任务上测试九种提示风格，发现没有普适最佳提示，提示风格的选择应取决于任务类型和统计汇总方式，且单一指标评估会掩盖提示效果的不稳定性。 |
| [^192] | [When Reconstruction Is Not Enough: Doubly Robust Counterfactual and Interventional Mediation Analysis with Learned Multimodal Representations](https://arxiv.org/abs/2610.30220) | 该论文证明了仅基于协变量的训练目标（重建、对比或物理信息）无法保证多模态表示足以支持因果识别，给出了表示保持可识别性的条件，并提出了双稳健的反事实与干预中介分析方法。 |
| [^193] | [AI Mindmeld: A Latent Interconnect Protocol](https://arxiv.org/abs/2610.30215) | 提出潜在互联协议（LIP），通过仅训练轻量级连接器将专家模型的潜在表示直接映射到LLM内部状态，实现无需联合训练、可扩展且可组合的模型间直接知识交换，突破了以文本等多模态数据传递信息的瓶颈。 |
| [^194] | [Prediction Is Not Calibration: Item Parameter Recovery and Representation Coupling in Knowledge Tracing](https://arxiv.org/abs/2610.30211) | 本文揭示知识追踪模型中预测准确并不等于项目参数可被正确恢复——难度类参数恢复稳健，而区分度和选项斜率的恢复依赖更宽的项目表示并与项目顺序存在耦合。 |
| [^195] | [Spectral Uncertainty Decomposition for Sparse Bayesian Learning](https://arxiv.org/abs/2610.30207) | 该论文提出一种基于先验预条件前向算子奇异值结构的谱框架，将稀疏贝叶斯学习中的后验协方差精确分解为可通过改进测量而减少的数据知情成分与由传感几何决定的不可约不确定性成分。 |
| [^196] | [MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2610.30206) | 提出MoEZip，一种基于Fisher信息矩阵度量专家路由扰动敏感性的路由感知KV缓存压缩方法，避免KV驱逐改变MoE模型的专家选择，从而提升稀疏MoE大模型长上下文推理的生成质量。 |
| [^197] | [Differentially Private Synthetic Data via Generalized Moment Matching](https://arxiv.org/abs/2610.30196) | 提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。 |
| [^198] | [Why Constrained Updates Still Forget: A Functional-Path Analysis of Continual SFT](https://arxiv.org/abs/2610.30191) | 该论文提出“函数路径”分析框架，将持续微调导致的遗忘分解为受控方向、梯度陈旧性、探针失配与有限步效应四个因素，并通过匹配的QLoRA干预实验揭示约束放置在Adam之前还是之后对旧能力保留的影响远大于约束粒度本身。 |
| [^199] | [Agentic Verified Optimization of Real-World Software](https://arxiv.org/abs/2610.30188) | 该论文提出“智能体验证优化”新任务：让AI智能体生成更快的程序实现，同时产出以现有实现为规约的机器验证等价性证明，从而避免测试漏洞，并在Lean 4中开发Aqueduct框架将其应用于真实世界软件。 |
| [^200] | [SIGNN++: Robust Graph Neural Star Identification on Earth](https://arxiv.org/abs/2610.30186) | SIGNN++通过更真实的训练噪声模型、角距边特征的傅里叶变换以及基于RANSAC的透视星对齐三项改进，显著提升了图神经网络星图识别方法在真实地面观测环境下的鲁棒性、识别精度与求解率。 |
| [^201] | [Detecting AI Training in Datacentre Power](https://arxiv.org/abs/2610.30182) | 该论文提出一种基于维特比追踪迭代节奏的方法，能从数据中心设施级电力数据中检测未申报的AI模型训练，即使运营者采用对抗性隐藏策略，一天内仍能以极低误报率捕获至少87%的训练运行。 |
| [^202] | [IRAH: Hierarchical Agentic Tree Search for Retrieval Pipeline Optimization](https://arxiv.org/abs/2610.30167) | IRAH将UCT引导的分层智能体树搜索与离线元学习初始化相结合，在有限评估预算下联合优化RAG检索流程的结构组件与数值超参数，性能最高超越TPE达6.4%。 |
| [^203] | [baseVultra: Compact Nonlinear Recurrence with Reallocated Capacity](https://arxiv.org/abs/2610.30154) | 提出baseVultra紧凑递归单元，将递归计算从门控变换重新分配到候选生成，在参数量匹配条件下于TinyShakespeare字符级语言建模上超越LSTM和GRU。 |
| [^204] | [The Cost of Saying Hello: An Empirical Study of Handshake Overhead in AI Agent Communication Protocols](https://arxiv.org/abs/2610.30151) | 本文首次对MCP、A2A和ACP三种AI智能体通信协议的握手开销进行分阶段实证测量，发现往返次数是握手成本的主导因素——旧版MCP需三次往返耗时230 ms，而A2A仅需一次往返耗时61 ms。 |
| [^205] | [Do Machine Learning-Guided Proactive Full Capacity Protocols Mitigate Emergency Department Overcrowding? A Simulation-Based Evaluation](https://arxiv.org/abs/2610.30148) | 该研究构建了机器学习与离散事件仿真相结合的评估框架，利用提前六小时的候诊室拥挤预测来触发主动式满负荷方案，以检验其相比传统阈值触发式（反应式）激活能否更有效地控制急诊科拥挤。 |
| [^206] | [Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories](https://arxiv.org/abs/2610.30016) | 该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。 |
| [^207] | [Tractable Inference with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2610.29909) | 该论文提出TOUCKAN，通过为KAN网络引入有限次多项式外函数形成可分解的和-积表示，使期望、高阶矩等统计量能够通过确定性递归精确计算，从而避免了高维场景下昂贵的蒙特卡洛采样。 |
| [^208] | [RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation](https://arxiv.org/abs/2610.29845) | 提出RangeNav，一个仅18.3万参数、完全基于合成数据预训练的轻量级激光雷达距离图像编码器，通过结合距离重建、语义分割和鸟瞰图预测的导航对齐目标，为大规模并行强化学习的机器人导航提供高效的360°感知表征。 |
| [^209] | [TraceForge: Distilling with Your Own Thinking](https://arxiv.org/abs/2610.29680) | 提出TraceForge，以教师模型的答案为输入、按学生模型自身家族的风格生成推理轨迹，并通过“监督+对抗”两阶段训练，在不违反服务条款的前提下实现对闭源前沿模型的推理蒸馏。 |
| [^210] | [Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories](https://arxiv.org/abs/2610.29385) | 该论文提出了一种可并行的概率潜在推理方法，通过在紧凑连续的潜在空间中对多个有效推理轨迹的分布进行建模，既实现了多样化推理轨迹的探索，又支持高效的并行化训练。 |
| [^211] | [Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning](https://arxiv.org/abs/2610.29351) | 提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。 |
| [^212] | [Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs](https://arxiv.org/abs/2610.28498) | 论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。 |
| [^213] | [Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory](https://arxiv.org/abs/2610.28315) | 该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。 |
| [^214] | [Poisoning-Based Backdoors Are Not Cryptographic](https://arxiv.org/abs/2610.28045) | 本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。 |
| [^215] | [ShlokaDhara: A Multi-Speaker Sanskrit Speech Corpus for Learning Meter-Aware Prosody in Text-to-Recitation](https://arxiv.org/abs/2610.26929) | 该论文发布了 43 小时的多说话人梵语诵读语料库 ShlokaDhara，用以支持要求格律正确性的“文本到诵读”合成任务，并通过 16 位梵语专家的主观评估与新颖实证指标验证了 XTTS-v2 在该任务上达到最高的感知质量。 |
| [^216] | [WhatIf-Bench: Benchmarking Intervention-Aware Time Series Forecasting](https://arxiv.org/abs/2610.26582) | 该论文提出了WhatIf-Bench，一个基于数据中心、电网、交通和供水系统机理仿真器构建的半合成基准，用于系统评估时间序列基础模型利用协变量进行干预感知情景预测的能力。 |
| [^217] | [NADS: Navigator-Guided Data Selection for Mitigating Catastrophic Forgetting in Fine-Tuning](https://arxiv.org/abs/2610.25751) | NADS通过先按目标微调配方训练一个“导航器”模型，再利用该模型与预训练模型之间的预测散度为合成数据评分，从而在预训练数据不可用时高效选择数据，以缓解LLM微调中的灾难性遗忘。 |
| [^218] | [Generative Retrieval for Reverse Target Fishing with a Proteome-Scale Benchmark](https://arxiv.org/abs/2610.24458) | 该论文提出GenFish，将反向靶标垂钓重新表述为生成式检索任务，通过残差向量量化蛋白质语言模型特征生成语义标识符，避免将未记录配对误判为负样本，并发布了首个蛋白质组规模基准ProTargetBench。 |
| [^219] | [Predictable Scaling: Efficient Model Exploration via Scaling-Law Prediction](https://arxiv.org/abs/2610.23453) | 该论文提出“可预测扩展”（PS）框架，通过沿资源轴构建小规模代理模型并结合曲线拟合与不确定性校准，从代理实验预测全规模推荐模型性能，从而将原本需数千GPU小时的模型改进假设评估变得高效可行。 |
| [^220] | [An Adaptive-Horizon Graph Neural Operator for coupled surrogate morphodynamic modelling](https://arxiv.org/abs/2610.22432) | 本文提出自适应时域图神经算子AH-GNO，直接在非结构网格上从水动力与地形状态学习预测河床高程增量，并能根据流动和河床演变自适应调整历史长度与更新间隔，从而实现高效的地貌动力学耦合代理建模。 |
| [^221] | [Adversarial Reinforcement Learning for Robust Planetary Landing](https://arxiv.org/abs/2610.20252) | 该论文提出一种三阶段对抗性强化学习训练框架，由辅助智能体生成具有时空相关性且满足物理约束的最坏情况扰动，与着陆控制器形成非对称序贯零和博弈并交替优化，从而显著提升行星着陆策略应对非标称条件的鲁棒性。 |
| [^222] | [Deficit-Driven Stratified Acquisition for Building STRATA-S12: An Open SAR-Optical Corpus for Self-Supervised Learning](https://arxiv.org/abs/2610.17680) | 提出了缺口驱动分层采集算法（DDSA），将构建配准的 Sentinel-1/Sentinel-2 数据集建模为半球校正分层上的约束最大覆盖问题，并通过人机协同的缺口修复机制确保全部 192 个分层得到完整覆盖，为自监督地球观测学习提供了高质量的开放 SAR-光学语料库。 |
| [^223] | [ATTUNE: Agentic Throughput Tuning for LLM Training](https://arxiv.org/abs/2610.17528) | ATTUNE是一个无人值守的编码智能体，通过“一次调一个参数”的诊断式搜索和正确性门控，在GPU小时预算内将LLM训练吞吐量平均提升1.7倍，同时确保不改变训练目标。 |
| [^224] | [FreeEvolve: Learning to Evolve Beyond Fixed Loops](https://arxiv.org/abs/2610.09197) | FREEEVOLVE 通过元进化将“如何优化”本身变成从经验中学到的能力，让进化器摆脱人工设计的固定搜索循环，自主决定测试什么、收集多少证据、保留哪些候选方案以及何时停止。 |
| [^225] | [Redundancy Meets Synergy: Dependency-aware Expert Selection for MoE via Submodular Optimization](https://arxiv.org/abs/2610.00558) | 提出DS-MoE框架，通过次模函数差分优化感知专家间的冗余与协同依赖关系，为MoE模型实现更优的紧凑专家子集选择。 |
| [^226] | [PACT: End-to-End Learning of Human Pose, Contacts, and Forces from Video](https://arxiv.org/abs/2610.00451) | PACT 提出一个端到端模型，通过可学习的接触力标记、时间 Transformer 和基于物理的监督，从单目视频中联合估计人体姿态、接触与接触力，并解决作用力标注稀缺问题。 |
| [^227] | [DashVMC: Real-Time Discrete World Model Control in Geometry Dash](https://arxiv.org/abs/2609.40003) | DashVMC仅用约两小时Geometry Dash游戏录像学习紧凑的动作条件世界模型，控制器通过BC初始化后完全在冻结模型推演中用PPO优化即可超越初始性能，并在消费级GPU上实现60Hz的实时捕获-动作闭环。 |
| [^228] | [UnlearningSoup: Is Repeated Tuning Necessary for Large Language Model Unlearning?](https://arxiv.org/abs/2609.37076) | 该论文发现不同遗忘运行得到的模型在权重空间中共处一个评估性能盆地，据此提出UnlearningSoup——一种针对遗忘任务定制的模型汤（权重平均）策略，无需反复调参即可恢复出更强的遗忘模型。 |
| [^229] | [PROACT-Agent: Progressive Runtime Oversight and Active Circuit-breaking for Real-Time Safety](https://arxiv.org/abs/2609.34415) | 提出PROACT-Agent框架，通过渐进式轨迹展开、推理增强的因果矫正和文化感知的数据本地化合成高保真安全轨迹数据，并构建包含15万余条标注状态的双语安全基准PROACT-Bench，以实现对大模型智能体的实时主动安全防护。 |
| [^230] | [Is the Judge Its Twin? Post-training makes a model's writing more predictable but barely moves its taste, as a judge, toward predictable writing](https://arxiv.org/abs/2609.32196) | 该研究发现后训练（SFT和DPO）确实使语言模型的写作变得更加可预测，但这种可预测性偏好并不会迁移到模型作为评判者的角色——训练后的模型在评判故事时几乎不偏向可预测的写作。 |
| [^231] | [Residual Correlation as a Diagnostic for Joint-Uncertainty Gains from GP Coregionalisation](https://arxiv.org/abs/2609.30085) | 该论文发现GP协同区域化的主要收益在于联合不确定性量化而非点预测，并提出残差相关性（而非原始目标相关性）才是预测联合不确定性增益的最强指标，同时给出了可从独立逐目标残差计算的轻量级诊断指标D_logdet来量化这一增益。 |
| [^232] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^233] | [Extending FunctionGemma for Practical On-Device Mobile Function Calling](https://arxiv.org/abs/2609.25373) | 该研究通过构建涵盖十五类设备控制操作的约9,500个对话的合成数据集MOBILEACTIONSEXTENDED对FunctionGemma 270M进行微调，将设备端Android函数调用的端到端准确率从29.3%大幅提升至76.5%，并使组合模型在Google数据集上达到82.3%。 |
| [^234] | [Reinforcement Learning in Operational Research: A Technical Review and Practical Roadmap](https://arxiv.org/abs/2609.24750) | 本文系统综述了强化学习与运筹学相结合的技术进展，并提出实践路线图，展示如何利用RL的学习能力提升传统OR算法在动态决策、组合优化和数字孪生中的解质量、计算效率与鲁棒性。 |
| [^235] | [FlashBoB: I/O-Efficient Exact Backward-over-Backward for Softmax Attention](https://arxiv.org/abs/2609.24089) | FlashBoB提出了一种精确且I/O高效的softmax注意力反向之反向算法，通过利用softmax双重反向中的分层仿射结构，将计算保持在片上tile内并避免所有N×N中间张量，从而在长序列场景下支持二阶优化、测试时训练和元学习等应用。 |
| [^236] | [Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents](https://arxiv.org/abs/2609.22486) | 该论文提出“语义投递网络”，主张将Web检索基础设施从面向人类客户端的URL/摘要检索模式，重构为直接向LLM智能体提供面向任务效用的语义块检索服务，并支持跨轮次有状态检索与智能体间的工作共享。 |
| [^237] | [Multi-center Medical Data Mining with FL-Net - A One-stop Shop for Federated Learning](https://arxiv.org/abs/2609.20650) | FL-Net是一个一站式联邦学习临床研究框架，集成了数据协调、数据发现、披露控制和容器化联邦工作流执行，填补了现有14个框架均无法完全满足的五项要求，并将在欧盟项目中覆盖10家医院的80万余名患者。 |
| [^238] | [Trajectory Learnability for Offline On-Policy Distillation with Imperfect Teachers](https://arxiv.org/abs/2609.18321) | 本文提出利用教师成功问题作为廉价参考，通过观测学生模型在不同训练阶段对教师失败轨迹中各token似然的变化（带符号似然变化），来识别不完美教师监督下仍然可学习的内容，从而解决离线在线策略蒸馏中不完美监督持续存在的问题。 |
| [^239] | [Kernel-Managed Shared Memory for System-Wide Personalization](https://arxiv.org/abs/2609.10144) | 提出内核管理的共享内存这一系统级抽象，由智能体系统内核统一负责多智能体间的记忆检索、隐私管控和提示注入，相比非管理的记忆后端显著提升了系统级个性化效果。 |
| [^240] | [Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules](https://arxiv.org/abs/2609.08333) | 该论文提出了一种两阶段生成框架，通过单一固定维度的分子级潜在表示结合自回归Transformer解码器，实现了无需预先指定分子大小的可变尺寸3D分子生成，并摒弃了等变层和密集成对键解码器的需求。 |
| [^241] | [Interface-Induced Trajectory Censoring](https://arxiv.org/abs/2609.03966) | 该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。 |
| [^242] | [Advances in Machine Learning for Directed Evolution: A Five-Year Retrospective](https://arxiv.org/abs/2609.03046) | 本文回顾指出，机器学习辅助定向进化（MLDE）未能像其他蛋白质工程领域那样取得变革性进展，其主因是研究者追求“最优蛋白质”的目标与定向进化在时间和资源约束下寻找“足够好蛋白质”的实际需求脱节，例如忽视DNA合成成本导致方法缺乏实际适用性。 |
| [^243] | [Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit](https://arxiv.org/abs/2609.02766) | 本文在316个物理方程采样的数据集上评估四种表格基础模型，发现其性能虽全面超越基线模型，但其贝叶斯先验既无法表示无噪声机制也无法表示物理单位，因此只能对物理数据进行插值而不能充当真正的物理模型。 |
| [^244] | [When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation](https://arxiv.org/abs/2609.00071) | 在部分线性模型的模拟研究中，干扰函数的预测误差无法一致地反映因果估计的偏差，表明仅用预测误差来评估干扰函数估计器对于因果推断而言是不充分的。 |
| [^245] | [Moving the Mean Toward the Known Good, Not Beyond It: What Inference-Time Interventions and Weight Consolidation Buy in Open-Ended Generation](https://arxiv.org/abs/2608.28886) | 该研究发现在开放式生成任务中，通过“生成-验证-选择-LoRA合并”循环在自身验证成功的样本上训练，能显著提升模型输出的平均质量并经三重重复实验验证其稳健性，但最佳候选仅能收敛到经典启发式算法的水平而无法超越。 |
| [^246] | [Decoupling candidate dual AGN from chance superpositions in the GOTHIC survey via a deep-learning framework](https://arxiv.org/abs/2608.24164) | 本文通过YOLOv11定向边界框深度学习框架，有效从GOTHIC巡天的偶然叠加中区分出真实双活动星系核，实现了高精度检测。 |
| [^247] | [POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment](https://arxiv.org/abs/2608.16407) | 本文提出LLM-MGCL，通过结合语义和地理信息的多图学习与对比对齐，有效解决了POI推荐中的冷启动问题。 |
| [^248] | [Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivarianc](https://arxiv.org/abs/2608.14556) | RHMP通过将拓扑与几何分离并学习余链度量，实现了物理场的精确守恒与几何依赖传播的统一。 |
| [^249] | [Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL](https://arxiv.org/abs/2608.11669) | 本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。 |
| [^250] | [When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs](https://arxiv.org/abs/2608.11403) | 本研究发现，在GPQA Diamond基准上，多数投票的自我一致性方法反而降低了小型语言模型在大多数硬科学问题上的准确率，并预注册验证了这一反直觉现象。 |
| [^251] | [Adaptation of Generalist Robot Policies with Minimal Data](https://arxiv.org/abs/2608.11363) | 本文提出了MiDAS方法，通过最小数据（单次或少量演示）结合离线到在线强化学习，使预训练机器人策略能高效适应新任务，验证了最小人工指导可启动自主学习的可行性。 |
| [^252] | [Parallel gradient boosting for flexible estimation of conditional distributions](https://arxiv.org/abs/2607.13550) | 该论文提出了一种称为“并行梯度提升”的梯度提升算法改进方法，通过在每次迭代中仅训练一个基模型即可同时预测多个目标（如整个条件分布），克服了传统提升方法在多输出回归任务中计算开销大、且现有高效变体受限于特定损失函数和决策树基学习器的局限。 |
| [^253] | [AirPASS: Over-the-Air Federated Learning via Pinching Antenna Systems](https://arxiv.org/abs/2607.06768) | 本文提出AirPASS交替优化框架，通过同伦-黎曼裕度巩固方法与同伦辅助几何优化方法交替求解设备选择、接收波束成形和夹持天线位置的联合非凸优化问题，在夹持天线系统辅助的空中联邦学习中实现聚合失真阈值约束下的设备接入数量最大化。 |
| [^254] | [LLM-Guided Transportation Hub Capacity Planning with Textual Business Inputs](https://arxiv.org/abs/2607.03651) | 该论文提出一种LLM引导的枢纽容量规划框架，通过思维链推理将文本商业背景转化为结构化容量决策，并借助优化模型反馈回路进行验证，在美国东南部真实13枢纽货运网络上将最优性差距从传统方法的11.0%显著降低至2.8%。 |
| [^255] | [Pathways of Visual Information Flow in Vision-Language Models](https://arxiv.org/abs/2607.03358) | 该研究发现视觉-语言模型通过“直接通路”和“文本中介通路”两条不同路径处理视觉信息，通路选择取决于任务类型、数据分布和提示设计，且在常用通路被消融时模型能灵活启用文本中介通路作为后备。 |
| [^256] | [How Far Can Sharpness and Complexity Jointly Explain Generalization?](https://arxiv.org/abs/2606.29043) | 本文定量研究了锐度与复杂度联合解释深度学习泛化能力的上限，发现恰当定义的度量对具有很强的解释力，其成功受锐度扰动尺度和复杂度表示方式的影响，但仍有失效情况，暗示可能需要额外因素。 |
| [^257] | [Giskard : Byzantine Robust and Private Aggregation for Large-Scale Decentralized Learning](https://arxiv.org/abs/2606.19129) | 提出了Giskard，一种面向大规模去中心化学习的拜占庭鲁棒且隐私保护的聚合方法，通过分布式差分隐私与安全多方计算相结合，同时防御模型投毒攻击和隐私攻击，并解决了现有基于MPC的方案可扩展性差的问题。 |
| [^258] | [SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation](https://arxiv.org/abs/2606.16454) | 本文提出SDS-LoRA方法，通过从几何角度分析并克服LoRA中由奇异值引起的各向异性梯度缩放问题，以改善低秩适应与全微调之间的对齐。 |
| [^259] | [Measuring Pretraining Contamination in Public Medical Vision-Language Benchmarks](https://arxiv.org/abs/2606.10066) | 该研究使用四种检测方法系统评估公开医学视觉-语言基准的预训练数据污染，发现图像层面的重叠信号主要反映图像来源的重叠而非模型对个体图像的记忆。 |
| [^260] | [Behavioral Consistency in Machine Learning](https://arxiv.org/abs/2606.07581) | 该论文指出模型行为不仅取决于权重，还取决于执行它的内核，并提出“内核契约”框架，通过非渐近理论界将内核间的logit差异与总变差、KL散度、期望奖励变化以及策略梯度偏差定量联系起来。 |
| [^261] | [Measuring Model Robustness via Fisher Information: Spectral Bounds, Theoretical Guarantees, and Practical Algorithms](https://arxiv.org/abs/2606.04767) | 本文提出了一种基于Fisher信息矩阵谱范数的、与攻击无关的模型鲁棒性度量，为常见架构推导了闭式谱界并给出首个理论鲁棒性排名，同时提供了适用于白盒和黑盒场景的高效估计算法。 |
| [^262] | [Rethinking Molecular Text Representations for LLMs: An Empirical Study](https://arxiv.org/abs/2606.03057) | 本文提出了系统性基准MolRepBench，通过对16个LLM在九种分子表示和八项化学任务上的评估，发现LLM的分子任务性能强烈依赖于表示形式且无通用最优解，其中CML整体最佳，结构化文本表示主导结构类任务而IUPAC主导语义类任务，SMILES变体虽在预训练中普遍使用却很少最优。 |
| [^263] | [LSVD: Loss-Aware Low-Rank Approximation for Efficient Low-Precision Vision-Language Models](https://arxiv.org/abs/2606.00573) | 提出了 LASER 框架，通过基于模型损失二阶近似的曲率加权 SVD 目标函数和 Kronecker 分解 Fisher 信息引导的秩分配，实现面向高效低精度推理的视觉语言模型低秩压缩。 |
| [^264] | [Measuring, Localizing, and Ablating Alignment Signatures in LLMs](https://arxiv.org/abs/2605.30526) | 该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。 |
| [^265] | [SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?](https://arxiv.org/abs/2605.30329) | 提出了包含1,099个真实研究提案的SoundnessBench基准，评估发现12个前沿大语言模型普遍存在乐观偏见（平均假阳性率74.0%），无法可靠地在投入资源前甄别研究想法的方法论缺陷。 |
| [^266] | [SAHG: Sector-Anisotropic Hyperbolic Graph Model for Social Bot Detection](https://arxiv.org/abs/2605.30166) | 该论文提出扇区各向异性双曲图模型SAHG，通过双曲几何适配社交图的层级无标度结构，并利用各向异性的曲率区分不同结构方向的密度差异，同时缓解社交机器人伪造异配连接导致的邻域信号污染问题，从而提升社交机器人检测的准确性。 |
| [^267] | [Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression](https://arxiv.org/abs/2605.28567) | 本文提出以激活加权分布和Wasserstein距离为核心的分布性框架，将稀疏自编码器的跨层特征匹配与电路压缩统一归结为不同激活流形间特征语义距离的估计问题。 |
| [^268] | [Can Entry-Wise Clipping Give Spectral Control of Stochastic Gradients?](https://arxiv.org/abs/2605.27733) | 本文提出一种简单的逐元素裁剪方法，能够以低成本实现对重尾随机梯度噪声的谱控制，从而平衡矩阵结构感知与计算开销之间的权衡，缓解训练中的损失突增等不稳定问题。 |
| [^269] | [IGADA-IoT: IoT Sensor Energy Optimization in Wireless Sensor Networks Driven by Automatic Data Augmentation](https://arxiv.org/abs/2605.27397) | 提出了一种信息差距引导的物联网传感器自动数据增强框架IGADA-IoT，通过分层多生成器协作与调度机制缩小动态信息差距，实现无线传感器网络中物联网传感器的能耗优化。 |
| [^270] | [When Should LLMs Reason? Adaptive Inference Routing via Entropy Dynamics](https://arxiv.org/abs/2605.22873) | 提出无需训练的轻量级路由框架EDRM，通过解读解码早期的熵轨迹变化趋势，自适应地在直接、标准和思维链三种解码模式间选择，在15个基准和7个大语言模型上有效平衡准确率与token消耗。 |
| [^271] | [Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts: What Survives a Careful Evaluation](https://arxiv.org/abs/2605.21437) | 对天山-帕米尔地区周地震计数神经概率预报的严格评估表明：泊松类预报（含ETAS）严重低估单周发生3次以上地震的概率（低估8.3倍），引入一个离散参数即可解决；而神经网络相对线性模型的优势在为每个网格加入截距后便不再显著。 |
| [^272] | [CoarseSoundNet: Building a reliable model for ecological soundscape analysis](https://arxiv.org/abs/2605.21143) | 提出了CoarseSoundNet深度学习模型，能在真实含噪的陆生被动声学监测条件下区分生物声、地声与人为声，为生态声景的粗粒度分类提供了可复现的框架。 |
| [^273] | [Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization](https://arxiv.org/abs/2605.19721) | 提出基于Wolpertinger框架的“投影智能体”方法，通过在共享的GNN潜在空间中单次前向传播预测潜在动作并解码为有效离散动作，同时提升了图组合优化求解器跨实例的泛化能力与面对大规模动作空间的计算可扩展性。 |
| [^274] | [Learning Unbiased Permutations via Flow Matching](https://arxiv.org/abs/2605.16755) | PermFlow是一种条件流匹配框架，通过闭式切空间投影直接在双随机矩阵的仿射子空间上运行，能够捕捉多峰置换分布，在无歧义输入上实现高准确率，并在歧义情形下恢复出多个有效置换。 |
| [^275] | [Manifold Sampling via Entropy Maximization](https://arxiv.org/abs/2605.12338) | 本文提出基于熵最大化的流形采样方法 MASEM，能够在光滑约束隐式定义、且被分割成未知数量不连通分量的可行域上实现均匀采样，并证明其在平均场意义下以指数速率收敛到最大熵目标分布。 |
| [^276] | [Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning](https://arxiv.org/abs/2605.12122) | 提出SAEParate方法，通过概念感知对比学习与交叉注意力引导的空间聚合，在稀疏自编码器潜在空间中显式解耦语义重叠的概念，从而在不修改模型参数的情况下实现扩散模型的精确概念遗忘。 |
| [^277] | [Efficient Adjoint Matching for Fine-tuning Diffusion Models](https://arxiv.org/abs/2605.11480) | 提出高效伴随匹配（EAM），通过将随机最优控制问题重构为线性基础漂移形式，避免了完整轨迹的随机模拟和伴随状态的反向ODE计算，大幅降低了扩散模型奖励微调的计算成本。 |
| [^278] | [Benchmarking Compositional Generalisation for Machine Learning Interatomic Potentials](https://arxiv.org/abs/2605.08988) | 该论文提出了一个由四个任务组成的基准测试，用以评估机器学习原子间势是真正学习到化学的组合结构并能泛化到未见分子，还是仅仅在训练数据上进行插值。 |
| [^279] | [One Algorithm, Two Goals: Dual Scoring for Parameter and Data Selection in LLM Fine-Tuning](https://arxiv.org/abs/2605.06166) | 该论文提出DualSFT算法，将LLM微调中的参数选择与数据选择统一为同一验证目标下的双层选择问题，通过对单一梯度交互矩阵进行行列聚合，一次性同时完成参数掩码和数据子集的双重评分与选择。 |
| [^280] | [Unifying Goal-Conditioned RL and Unsupervised Skill Learning via Control-Maximization](https://arxiv.org/abs/2605.06145) | 该论文提出将目标条件强化学习（GCRL）与互信息技能学习（MISL）统一在控制最大化框架之下，并证明三种典型的GCRL公式在根本上是不可等价的，为无监督技能学习为何能支持下游目标达成任务提供了理论基础。 |
| [^281] | [Graph Memory Transformer (GMT)](https://arxiv.org/abs/2604.23862) | 本文提出图记忆Transformer（GMT），用可学习的显式记忆图（128个质心加有向转移矩阵）替代仅解码器Transformer中的FFN子层，在因果自注意力不变的情况下，以更少的参数量（82.2M对103.0M）实现稳定训练。 |
| [^282] | [AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency](https://arxiv.org/abs/2604.16158) | 提出AtManRL方法，通过优化可微注意力掩码推导显著性奖励并与结果奖励结合于GRPO强化学习框架中，使思维链推理真正因果性地促成最终答案，从而提升推理的忠实性。 |
| [^283] | [Super Agents and Confounders: Influence of surrounding agents on vehicle trajectory prediction](https://arxiv.org/abs/2604.03463) | 该论文发现周围智能体信息反而会降低轨迹预测的精度，并提出无需额外监督的条件信息瓶颈（CIB）方法，通过压缩有效特征、过滤无益信息来提升预测性能的稳定性与因果性。 |
| [^284] | [Unrestrained Simplex Denoising for Discrete Data. A Non-Markovian Approach Applied to Graph Generation](https://arxiv.org/abs/2603.28572) | 提出了一种在概率单纯形上操作的非马尔可夫生成框架“无约束单纯形去噪”，通过使不同时间的噪声表示条件独立来移除不必要的约束，在图生成任务上超越了离散扩散和流匹配基线。 |
| [^285] | [Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation](https://arxiv.org/abs/2603.21782) | 该论文提出免训练的“不变性审计”诊断工具，利用预训练的扩散或流匹配模型，通过纤维损失引导去噪过程，高效采样出与查询输入特征相同但形式多样的输入，从而检验模型学习到的不变性是否符合任务语义。 |
| [^286] | [Brittlebench: Quantifying LLM robustness via prompt sensitivity](https://arxiv.org/abs/2603.13285) | 该论文提出了一个量化大语言模型对提示变体敏感性的理论框架，并设计了名为Brittlebench的评估流程，通过对主流基准测试施加保持语义的扰动，揭示前沿模型性能最高下降12%且不同模型的脆弱程度各异。 |
| [^287] | [Markovian Generation Chains in Large Language Models](https://arxiv.org/abs/2603.11228) | 本文将大语言模型反复处理文本的迭代推理过程定义为马尔可夫生成链，发现输出要么收敛到小的循环集合要么持续产生新句子，且多样性的增减取决于温度参数和初始输入句子等因素。 |
| [^288] | [An Optimal Control Approach To Transformer Training](https://arxiv.org/abs/2603.09571) | 本文将Transformer训练建模为受控McKean-Vlasov动力学的最优控制问题，通过将动力学提升到概率测度空间构造完全可观测的MDP，并证明了全局最优策略的存在性及其与依赖初始分布的开环策略的等价性。 |
| [^289] | [Optimistic Policy Regularization: Elite-Statistic Reward Shaping](https://arxiv.org/abs/2603.06793) | 论文提出乐观策略正则化（OPR），将同策略训练中被丢弃的罕见高回报回合转化为基于精英动作先验对数比率的闭式奖励加成，仅需对 PPO 奖励做一行修改，便在 49 个 Atari 游戏上以少 5 倍的决策次数超越了 PPO 和自我模仿学习等基线。 |
| [^290] | [K-Means as the Zero-Temperature Limit of Normalized RBF Assignments: Rates, Trajectories, and Exact Recovery with Entmax](https://arxiv.org/abs/2603.04625) | 本文严格证明了K-均值聚类是归一化RBF分配在零温度下的Γ-极限，从目标函数、优化方法、单次更新和轨迹收敛等多个层面定量刻画了软RBF分配与硬K-均值之间的联系，并利用Entmax实现了精确恢复。 |
| [^291] | [Near-Optimal Regret for KL-Regularized Multi-Armed Bandits](https://arxiv.org/abs/2603.02155) | 该论文通过新颖的分层论证对KL-UCB算法进行精细分析，首次给出了对臂数 $K$ 呈线性依赖的高概率后悔上界 $\tilde{O}(\eta K\log^2T)$，并以首个非常数下界 $\Omega(\eta K\log T)$ 证明了其近紧性。 |
| [^292] | [Exact and Asymptotically Complete Robust Verifications of Neural Networks via Ising Solvers](https://arxiv.org/abs/2603.00408) | 该论文提出了一个与Ising求解器兼容的神经网络鲁棒性验证框架，通过对数编码（Log-PW）将每个神经元的二元变量复杂度从线性降至对数级别，并结合基于Jordan分解的单调化简与混合Benders求解器，实现了对包括非单调激活在内的各类激活函数的精确且渐近完备的鲁棒性验证。 |
| [^293] | [Structured Unitary Tensor Network Representations for Circuit-Efficient Quantum Data Encoding](https://arxiv.org/abs/2602.16266) | 该论文提出了一种基于结构化酉张量网络表示的电路高效量子数据编码框架TNQE，通过将经典数据分解为张量核心并编译为编码电路，同时引入酉感知约束将核心参数化为可学习的块酉矩阵，从而降低量子机器学习中数据编码所需的电路深度与量子资源开销。 |
| [^294] | [Learning to Drive in New Cities Without Human Demonstrations](https://arxiv.org/abs/2602.15891) | 提出NOMAD方法，通过自博弈多智能体强化学习，仅利用目标城市的地图和元信息即可将驾驶策略适配到新城市，完全无需该城市的人类示范轨迹。 |
| [^295] | [Are Two LLMs Better Than One? A Student-Teacher Dual-Head LLMs Architecture for Pharmaceutical Content Optimization](https://arxiv.org/abs/2602.11957) | 本文提出面向制药合规内容审核的LRBTC架构，采用师生双头LLM/VLM结合瀑布式规则过滤逻辑，在AIReg Bench上达到83.0% F1与97.5%召回率，漏检违规较Gemini 2.5 Pro减少五倍。 |
| [^296] | [SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures](https://arxiv.org/abs/2602.07628) | SleepMaMi是一个通用睡眠基础模型，采用分层双编码器设计——宏观编码器建模整夜睡眠的时间依赖性，微观编码器捕捉生物信号的细粒度形态特征——从而突破传统任务特定模型只关注局部微观特征的局限，实现对整夜睡眠宏观架构与微观信号形态的统一建模。 |
| [^297] | [GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer](https://arxiv.org/abs/2602.03358) | GFlowPO提出了一种概率化提示词优化框架，利用离线策略的生成流网络目标微调轻量级提示词语言模型，并通过无需训练的动态记忆更新机制更新元提示词，从而实现样本高效的语言模型提示词优化。 |
| [^298] | [TabClustPFN: A Prior-data Fitted Network for Tabular Data Clustering](https://arxiv.org/abs/2601.21656) | TabClustPFN通过协同设计的先验、训练目标与解耦架构，在单次推理中同时推断表格数据的聚类分配和未知聚类基数，从而克服了无监督、基数未知和标签交换等聚类中的核心挑战。 |
| [^299] | [High-dimensional learning dynamics of multi-pass Stochastic Gradient Descent in multi-index models](https://arxiv.org/abs/2601.21093) | 本文在样本量与维度成比例的高维区间内，首次对非凸多指标模型上的多遍随机梯度下降动力学建立了渐近精确的刻画，证明其收敛于一种新颖的动力学平均场连续时间极限，且适用于任意次线性批量大小及相应的临界学习率缩放。 |
| [^300] | [Penalized Fair Regression for Multiple Groups in Chronic Kidney Disease](https://arxiv.org/abs/2512.17340) | 提出一种面向多群体的惩罚化公平回归通用框架，通过不公平性惩罚与新颖的自动惩罚权重选择评分函数，实现高效求解，并在慢性肾病研究中取得优于现有方法的公平性-准确性平衡。 |
| [^301] | [Verification of Unknown Dynamical Systems via Autoencoder Latent Space](https://arxiv.org/abs/2512.13593) | 该论文提出了一种通过凸自编码器降维并利用基于核的方法在潜在空间中学习动力学的形式化验证框架，保证了所构建的有限抽象包含原系统的真实行为，并可将潜在空间中的验证结果映射回原始系统。 |
| [^302] | [LLM-Auction: Generative Auction towards LLM-Native Advertising](https://arxiv.org/abs/2512.10551) | 该论文提出了首个将拍卖与生成相融合的学习式生成式拍卖机制LLM-Auction，通过把广告分配构建为LLM输出与平衡广告主价值和用户体验的机制目标之间的偏好对齐，使LLM无需额外推理成本即可内在建模分配外部性，解决了传统机制在LLM原生广告场景中失效的问题。 |
| [^303] | [MarkTune: Improving the Quality-Detectability Trade-off in Model-Embedded LLM Watermarking](https://arxiv.org/abs/2512.04044) | 该论文提出MarkTune，通过向高质量参考分布正则化来解耦质量保持与检测能力之间的关系，从而改进了模型嵌入式大语言模型水印中生成质量与可检测性之间的权衡。 |
| [^304] | [How Does RL Post-training Induce Skill Composition? A Case Study on Countdown](https://arxiv.org/abs/2512.01775) | 本文通过将Countdown任务中模型的解答解析为表达式树，发现RL后训练能诱导模型对子任务技能进行组合式复用（实现对更大规模和未见结构的分布外泛化），且技能习得的顺序由任务的组合结构决定。 |
| [^305] | [Bayesian Ambiguity Contraction-based Adaptive Robust Markov Decision Processes for Adversarial Surveillance Missions](https://arxiv.org/abs/2512.01660) | 本文提出一种基于贝叶斯模糊性收缩的自适应鲁棒马尔可夫决策过程框架，通过在线剔除与观测不一致的威胁原型来逐步收紧不确定性集合，为协同作战飞机在对抗环境下的侦察监视任务实现更精确且安全的实时决策。 |
| [^306] | [FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers](https://arxiv.org/abs/2511.21843) | 该论文提出了FLAWS基准，通过在同行评审论文中系统性插入破坏关键论断的错误，构建了713个论文-错误对，以自动化评估大语言模型检测和定位科学论文中错误的能力。 |
| [^307] | [Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading](https://arxiv.org/abs/2511.14102) | Specter通过量化使能的多步投机解码与专家卸载相结合，在单GPU上实现了平均1.50倍（最高1.88倍）的MoE推理加速。 |
| [^308] | [Revisiting GAN with Bayes-Optimal Discrimination](https://arxiv.org/abs/2510.25609) | 该论文提出用贝叶斯错误率上界（BOLT损失）替代传统交叉熵目标来重构GAN训练，理论上证明其对抗博弈等价于最小化数据与生成分布间的总变差，并与Wasserstein-1距离建立关联，实验显示样本质量与覆盖度均得到改善。 |
| [^309] | [FlowCritic: Bridging Value Estimation with Flow Matching in Reinforcement Learning](https://arxiv.org/abs/2510.22686) | FlowCritic提出了一种基于流匹配的生成式价值估计新范式，通过建模价值分布并生成样本来进行估计，克服了多评论家集成缺乏分布信息和分布式RL表达能力受限的问题。 |
| [^310] | [How to Teach Large Multimodal Models New Skills](https://arxiv.org/abs/2510.08564) | 该研究发现大型多模态模型在顺序学习新技能时的遗忘与输出token分布偏移共同变化，并提出两种简单微调策略——仅更新自注意力投影层或仅更新MLP的Gate&Up（冻结Down投影），能在强劲学习新技能的同时将通用能力遗忘降至最低。 |
| [^311] | [To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning](https://arxiv.org/abs/2510.03207) | 本文通过扰动块MDP理论模型与模拟运动任务上的对照实验，系统揭示了部分可观测强化学习中特权专家蒸馏与无特权信息标准强化学习之间的算法权衡及其关键影响因素。 |
| [^312] | [Robust Tangent Space Estimation via Laplacian Eigenvector Gradient Orthogonalization](https://arxiv.org/abs/2510.02308) | 提出了一种名为LEGO的谱方法，通过正交化图拉普拉斯低频特征向量的梯度，利用数据的全局结构来鲁棒地估计流形切空间，克服了局部PCA在高噪声环境下邻域大小选择的两难困境。 |
| [^313] | [Energy Estimation for Virtual Servers Using Resource Utilization and Machine Learning](https://arxiv.org/abs/2509.09991) | 该论文提出一种机器学习方法，仅利用虚拟服务器内部可观测的系统和性能指标即可预测单个虚拟服务器的CPU功耗，训练时以主机侧RAPL测量作为功耗标签，在训练覆盖部署条件时预测效果良好。 |
| [^314] | [Stochastic Matching Bandits with Rare Optimization Updates](https://arxiv.org/abs/2509.04194) | 本文提出了MNL选择模型下随机匹配老虎机的批处理算法，通过将昂贵的匹配优化更新次数限制在Θ(log log T)次，在大幅降低计算开销的同时保持了遗憾最优性能。 |
| [^315] | [Sparse Equation Matching: A Derivative-Free Learning for General-Order Dynamical Systems](https://arxiv.org/abs/2507.20072) | 提出稀疏方程匹配（SEM）统一框架，利用基于格林函数的积分型稀疏回归，实现对一般阶动力系统中微分算子和驱动函数的无导数学习。 |
| [^316] | [Automatic Fine-grained Segmentation-assisted Report Generation](https://arxiv.org/abs/2507.16623) | 该论文提出ASaRG方法，通过将专业放射学模型生成的中间特征和细粒度分割图以简单拼接的方式融合进LLaVA的多模态投影层，在仅增加少量参数的情况下提升了医学报告生成的性能与可信度。 |
| [^317] | [Meta-learning of Gibbs states for many-body Hamiltonians with applications to Quantum Boltzmann Machines](https://arxiv.org/abs/2507.16373) | 该论文提出了两种元学习算法，通过对有限训练参数值集合进行联合优化，在NISQ设备上高效制备参数化多体哈密顿量的吉布斯态，并可应用于量子玻尔兹曼机。 |
| [^318] | [On the Gradient Domination of the LQG Problem](https://arxiv.org/abs/2507.09026) | 本文提出一种基于过去p个时间步输入输出数据的控制输入历史表示参数化方法，将LQG问题转化为具有梯度支配性的等价静态优化问题，从而为策略梯度方法求解LQG问题建立了全局收敛保证。 |
| [^319] | [Few-shot Classification as Multi-instance Verification: Effective Backbone-agnostic Transfer across Domains](https://arxiv.org/abs/2507.00401) | 提出将少样本分类转化为多实例验证任务的"MIV-head"分类头，在无法微调骨干网络的约束下，仅利用冻结“黑盒”骨干网络的静态嵌入即可在目标域少样本数据上训练并实现有竞争力的跨域分类性能。 |
| [^320] | [Aggregation of Published Non-Uniform Axial Power Data for Phase II of the OECD/NEA AI/ML Critical Heat Flux Benchmark](https://arxiv.org/abs/2507.00034) | 本工作为OECD/NEA AI/ML临界热流密度基准第二阶段汇集并数字化了涵盖均匀与非均匀轴向加热条件的大规模CHF数据集，并揭示经典关联式和仅用均匀数据训练的神经网络在非均匀功率分布下均难以准确预测，凸显了开发泛化能力更强的AI/ML模型的必要性。 |
| [^321] | [Diffusion-Based Hypothesis Testing and Change-Point Detection](https://arxiv.org/abs/2506.16089) | 本文将基于分数的假设检验与变点检测方法推广为基于扩散的版本，通过数值优化权重矩阵提升检测能力，并从理论上量化了算法性能及其达到最优的条件。 |
| [^322] | [ShIOEnv: A Command Evaluation Environment for Grammar-Constrained Synthesis and Execution Behavior Modeling](https://arxiv.org/abs/2505.18374) | ShIOEnv是一个与Gymnasium兼容的Bash命令评估环境，通过语法约束合成生成有效的shell输入，并记录完整的执行产物（标准输出、标准错误、退出状态和系统状态变化），为训练无需执行的shell模拟模型和部署网络欺骗蜜罐提供数据支持。 |
| [^323] | [MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark](https://arxiv.org/abs/2505.11109) | 该论文提出了首个大规模多语言音视频开放集深度伪造检测基准MAVOS-DD，包含八种语言近350小时的真实与伪造视频，实验表明现有最先进的检测器在开放集场景下无法维持其性能水平。 |
| [^324] | [Eluder-Dimension-Free Regret for Stage-Wise Constrained Contextual Bandits via Offline Regression](https://arxiv.org/abs/2505.07101) | 提出CS-UCCB算法，通过结合离线回归、代价置信集与上反事实置信界，在逐阶段约束的上下文赌博机中首次实现了不含eluder维度项的Õ(√T)遗憾界，同时以高概率保证每一轮的代价可行性。 |
| [^325] | [DetoxAI: a Python Toolkit for Debiasing Deep Learning Models in Computer Vision](https://arxiv.org/abs/2505.05492) | DetoxAI是一个开源Python工具包，通过内部表征干预等先进去偏算法、公平性指标和基于归因的可视化工具，专门用于提升深度学习计算机视觉分类器的公平性。 |
| [^326] | [How to Backdoor Image Knowledge Distillation](https://arxiv.org/abs/2504.21323) | 本文提出了一种针对图像知识蒸馏的新型后门攻击方法，通过向蒸馏数据集注入被干净教师模型预测为目标标签的触发图像，使学生在教师保持干净的情况下学到后门，且在10%中毒率下仍然有效。 |
| [^327] | [Does More Compute Make Every Adversarial Attack Transferable?](https://arxiv.org/abs/2504.20310) | 本文证明了在攻击者与防御者之间存在二次方算力差距时，所有可学习分类任务要么存在可迁移攻击要么存在对抗防御，而生成任务在低于该差距时不满足此二分性，且推理时使用额外计算的防御方法可被证明严格优于所有发布固定模型的防御。 |
| [^328] | [Analysis of On-policy Policy Gradient Methods under the Distribution Mismatch](https://arxiv.org/abs/2503.22244) | 本文分析了策略梯度方法中采样分布与折扣状态分布失配的影响，证明了表格型参数化在有偏更新下的全局收敛性，并通过渐近误差首阶系数之比给出了分布失配何时加速或减缓收敛的明确判据。 |
| [^329] | [Reducing QPU Calls in Variational Quantum Algorithms via Classical Offloading of Gradient Computation](https://arxiv.org/abs/2503.16361) | 本文提出HELIA拟设及结合经典增强梯度方法与参数移位规则的混合训练方法，将梯度计算部分卸载到经典硬件，使VQE的逻辑量子评估次数平均减少高达60%，同时提升精度和成功率。 |
| [^330] | [Path Regularization: A Near-Complete and Optimal Nonasymptotic Generalization Theory for Multilayer Neural Networks and Double Descent Phenomenon](https://arxiv.org/abs/2503.02129) | 该论文首次提出了路径正则化多层神经网络的近乎完备且最优的非渐近泛化理论，给出了显式泛化误差上界，无需损失函数有界及网络宽度、深度等常见假设，超越了偏差-方差权衡并能解释深度学习中的双重下降现象。 |
| [^331] | [ACTIVA: Amortized Causal Effect Estimation via Variational Autoencoders](https://arxiv.org/abs/2503.01290) | ACTIVA是一个基于transformer的条件变分自编码器，能够仅通过观测数据在单次前向传播中摊销式地估计完整的干预分布，无需干预数据、因果图或逐任务拟合。 |
| [^332] | [Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms](https://arxiv.org/abs/2502.20115) | 本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。 |
| [^333] | [Fragility-aware Classification for Risk Characterization and Robust Learning](https://arxiv.org/abs/2502.13024) | 本文提出了脆弱性指数（FI）这一新型性能指标，从风险规避角度量化分类器自信误判的尾部风险，并在鲁棒满意度框架下开发了可处理的代理训练方法以实现鲁棒学习。 |
| [^334] | [The Fenchel-Young Generalization of Bayesian Principles](https://arxiv.org/abs/2502.10295) | 本文利用Fenchel-Young不等式推广了共轭贝叶斯的变分原理，建立了超越KL几何的广义证据下界与广义共轭框架，实现了支持参数依赖支撑集和可调重尾的后验推断，并给出了EM算法的FY推广形式。 |
| [^335] | [Risk-Aware Alignment Must Incorporate Reward Uncertainty](https://arxiv.org/abs/2410.23726) | 该论文证明了在参考策略局部邻域内实现最优风险的正则化器是由奖励协方差矩阵加权的二次范数，并据此提出一种基于奖励方差估计正则化的新策略优化算法，在大语言模型等实验中展现出比现有方法更鲁棒的对齐效果。 |
| [^336] | [Are EEG Sequences Time Series? EEG Classification with Time Series Models and Joint Subject Training](https://arxiv.org/abs/2404.06966) | 本文系统对比了EEG分类模型与通用时间序列分类模型的差异，提出受试者特定、受试者无关和受试者条件三种模型设置，并证明现成的时间序列分类模型按受试者训练后性能可接近专门的EEG分类模型。 |
| [^337] | [Optimal Clustering of Multi-layer Networks under Mixture Block Models](https://arxiv.org/abs/2311.15598) | 本文在混合多层随机块模型下建立了由Chernoff信息刻画的极小极大最优网络聚类错误率，并提出了一种结合张量初始化与似然精化的两阶段估计器，在极端稀疏条件下仍能达到该最优率，且可扩展至计数加权网络。 |
| [^338] | [Evolutionary Deep Nets for Non-Intrusive Load Monitoring](https://arxiv.org/abs/2303.03538) | 本研究提出并评估了用于非侵入式负荷监测的自适应稀疏进化训练与热启动压缩方法，在保持准确率的同时将宏F1提升至92.40%，并使存储权重减少17.36%、第一层活跃连接减少23.93%。 |
| [^339] | [On Sex, Evolution, and the Multiplicative Weights Update Algorithm](https://arxiv.org/abs/1502.05056) | 该论文强化了性进化理论，证明有性生殖实现乘性权重更新算法（MWUA）这一结论无需弱选择条件，也无需乘积分布假设，并指出乘积分布假设实际上与种群动态不相容。 |

# 详细

[^1]: 感知运动对齐：具身策略成功的柏拉图式代理指标

    Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success

    [https://arxiv.org/abs/2610.31670](https://arxiv.org/abs/2610.31670)

    提出无需训练的离线指标“感知运动对齐”（S²），通过衡量模型感知表征与物理运动几何结构的对齐程度，低成本地预测具身模型的下游任务成功率。

    

    机器人基础模型通常需要通过代价高昂的闭环回放（rollout）来评估，这促使人们寻找低成本的离线模型能力代理指标。柏拉图表征假说（PRH）指出，随着表征质量的提升，视觉、语言等不同模态会收敛到对现实的共享统计模型。我们提出一个问题：运动轨迹是否构成另一种这样的模态——当具身模型能力增强时，其感知表征是否应向物理运动的结构收敛？为探究这一点，我们提出了感知运动对齐，这是一种无需训练的离线指标，用于衡量模型的感知表征与物理运动几何结构的对齐程度。我们发现，S² 在多个具身模型家族中均与下游机器人任务成功率呈正相关，因此可作为低成本的离线代理指标，在闭环评估之前对模型进行优先级排序。S² 还能够识别出有前景的（模型）（摘要在此处截断）。

    arXiv:2610.31670v1 Announce Type: cross  Abstract: Robotic foundation models are typically evaluated through costly closed-loop rollouts, which motivates cheap offline proxies for model capability. The Platonic Representation Hypothesis (PRH) suggests that as representations improve, different modalities such as vision and language converge toward a shared statistical model of reality. We ask whether motion trajectories form another such modality: as embodied models become more capable, should their sensory representations converge toward the structure of physical motion? To examine this, we propose Sensorimotor Alignment ($S^2$), a training-free offline metric that measures how well a model's sensory representation aligns with the geometry of physical motion. We find that $S^2$ correlates positively with downstream robot task success across embodied model families, making it a low-cost offline proxy for prioritizing models before closed-loop evaluation. $S^2$ can also identify promisi
    
[^2]: 从随机特征到全可训练网络的相变

    Phase Transition from Random Features to Fully Trainable Networks

    [https://arxiv.org/abs/2610.31663](https://arxiv.org/abs/2610.31663)

    本文为部分可训练的两层神经网络建立了统一的平方 L₂ 逼近误差界 O((κ_P log(m/δ)/m)^α)，揭示了从随机特征到全可训练网络之间存在相变现象：初始改善快速平滑随后收益递减，且初始化失配越大转变越剧烈。

    

    构建精确且计算上可处理的逼近空间是科学计算中的一项基本任务。部分可训练神经网络在固定随机特征与完全自适应表示之间架起了桥梁。本文量化了逼近界中的相变模式，并在训练误差中观察到类似的行为：初始阶段呈现快速而平滑的改善，随后收益递减，且当初始化失配越大时，转变越剧烈。在谱多项式衰减（指数 α>1）与杠杆值兼容性条件下，我们为宽度为 m、可训练比例为 P 的两层网络建立了平方 L₂ 逼近速率 O((κ_P log(m/δ)/m)^α)，其中 κ_P=(P/κ₁\*+(1-P)/κ₀\*)⁻¹。该界在再生核希尔伯特空间（RKHS）的单位球上一致成立，置信概率至少为 1-δ。常数 κ₀\* 与 κ…（摘要原文在此截断）

    arXiv:2610.31663v1 Announce Type: cross  Abstract: Constructing accurate and computationally tractable approximation spaces is a fundamental task in scientific computing. Partially trainable neural networks bridge fixed random features and fully adaptive representations. This paper quantifies a phase transition pattern in approximation bounds and observes analogous behavior in training errors: a rapid but smooth initial improvement followed by diminishing returns, with a sharper transition under greater initialization mismatch. Under polynomial spectral decay with exponent $\alpha>1$ and leverage compatibility, we establish a squared $L_2$ approximation rate $\mathcal{O}((\kappa_P\log(m/\delta)/m)^\alpha)$ for two-layer networks of width $m$ and trainable proportion $P$, where $\kappa_P=(P/\kappa_1^*+(1-P)/\kappa_0^*)^{-1}$. The bound holds uniformly over the unit ball of a reproducing kernel Hilbert space (RKHS) with probability at least $1-\delta$. The constants $\kappa_0^*$ and $\ka
    
[^3]: 通过选择性审计实现可靠的AI辅助决策

    Reliable AI-Assisted Decisions via Selective Auditing

    [https://arxiv.org/abs/2610.31662](https://arxiv.org/abs/2610.31662)

    提出了一个在线选择性审计框架，在最小化专家审计频率的同时，提供可证明的错误控制保证，使未经审计的AI错误比例始终低于用户设定的阈值。

    

    我们考虑对产生一系列决策的AI系统进行在线监控。在每一轮中，系统输出一个提议的决策以及一个用于估计该决策可信度的置信分数。决策和置信分数不必由同一个模型产生：例如，一个智能体可能提出决策，而另一个独立的验证智能体对其进行评估并输出正确性或安全性分数。基于这些信息，我们必须决定是自动接受AI决策，还是咨询专家（例如人类审计员），由其审查该提案并接受或纠正它。我们将此任务表述为一个在线约束优化问题，其目标是最小化专家审计的频率，同时保证未经审计的AI错误比例保持在用户定义的阈值以下。我们推导出一族具有可证明错误控制保证的算法，无需（摘要在此处被截断）

    arXiv:2610.31662v1 Announce Type: cross  Abstract: We consider online monitoring of an AI system that produces a sequence of decisions. At each round, the system outputs a proposed decision together with a confidence score estimating the trustworthiness of the decision. The decision and confidence score need not be produced by the same model: for example, one agent may propose a decision, while a separate verifier agent evaluates it and outputs a correctness or safety score. Based on this information, we must decide whether to accept the AI decision automatically or consult an expert, such as a human auditor, who reviews the proposal and either accepts or corrects it. We formulate this task as an online constrained optimization problem in which the goal is to minimize the frequency of expert auditing while guaranteeing that the fraction of AI errors that go unaudited stays below a user-defined threshold. We derive a family of algorithms with provable error-control guarantees with no as
    
[^4]: SpectralCanary：用于文生图（T2I）数据集使用审计的极其简单的低频痕迹

    SpectralCanary: Embarrassingly Simple Low-Frequency Traces for T2I Dataset-Use Auditing

    [https://arxiv.org/abs/2610.31659](https://arxiv.org/abs/2610.31659)

    SpectralCanary通过低通滤波掩码将固定金丝雀图像嵌入数据集，以极其简单且无需辅助生成或逐图像优化的方式，实现对文生图模型是否使用特定数据集的高效审计检测。

    

    审计一个文本到图像（T2I）模型是否在特定图像集合上进行过微调是很困难的：模型可能在不重现训练图像的情况下吸收集合层面的视觉特征。主动式数据集使用审计方法在数据集发布前植入一个共享信号，随后检查该信号是否出现在可疑模型生成的输出中。现有方法往往需要在可验证性、保真度、鲁棒性和可部署性之间进行权衡。我们提出了SpectralCanary，它使用平滑的低通滤波掩码将一张固定的金丝雀图像混合到集合中的每张图像里，同时保持文本描述不变，既不需要辅助生成，也不需要逐图像优化。一个轻量级的验证器可以在可疑模型的输出中检测到传递的信号。在Pokemon数据集上的八项测试以及三种现实数据转换场景中，涵盖两个架构家族，SpectralCanary实现了最高的平均每……

    arXiv:2610.31659v1 Announce Type: cross  Abstract: Auditing whether a text-to-image (T2I) model was fine-tuned on a particular image collection is difficult: a model can absorb collection-level visual characteristics without reproducing training images. Proactive dataset-use auditing plants a shared signal in a dataset before release, then later checks whether that signal appears in a suspect model's generated outputs. Existing approaches often trade off verifiability, fidelity, robustness, and deployability. We introduce $\textbf{SpectralCanary}$, which uses a smooth low-pass mask to blend a fixed canary image into every collection image while leaving captions unchanged and requiring neither auxiliary generation nor per-image optimization. A lightweight verifier detects the transferred signal in suspect-model outputs. Across eight tests on the Pokemon dataset and three realistic data transformations, spanning two architectures families, SpectralCanary achieves the highest average per-
    
[^5]: 面向基于大语言模型的IT工单分类的成本感知弃权方法

    Cost-Aware Abstention for LLM-Based IT Ticket Classification

    [https://arxiv.org/abs/2610.31657](https://arxiv.org/abs/2610.31657)

    该论文提出一种离线的成本感知弃权评估框架，通过在校准数据上按显式成本选择并冻结路由策略，在真实IT工单数据上显著降低分类成本与错误路由率，并表明简单的TF-IDF逻辑回归结合成本感知阈值可优于大语言模型方案。

    

    IT支持工单的错误自动路由，其代价可能高于人工分诊。我们将成本感知弃权评估为一种离线的、成本条件下的就绪度评估方法：先在明确的成本设定下基于校准数据选择一个路由策略，随后仅在留出的测试数据上对该冻结策略进行一次评估。在一个包含47,837条工单、八个类别的公开语料库上，我们比较了四个经过训练的非大语言模型分类器与四个大语言模型在零样本和少样本提示下的表现。在示例性成本模型 (c_ok, c_triage, c_wrong) = (0, 1, 5) 下，基于校准数据选择的TF-IDF逻辑回归及0.60的阈值，将测试成本从0.763降至0.472，相对降低38.1%（配对比率自助法95%置信区间 [35.7%, 40.4%]），同时将错误路由率从15.3%降至3.5%。在事后成本参数分析中，冻结的校准选择策略还在16个设置中的14个里，在留出测试数据上同时优于“始终路由”和“始终弃权”两种基线。最强的p……（原文摘要截断）

    arXiv:2610.31657v1 Announce Type: cross  Abstract: Wrong automatic routing of an IT support ticket can cost more than human triage. We evaluate cost-aware abstention as an offline, cost-conditional readiness assessment: select a routing policy on calibration data under explicit costs, then evaluate that frozen policy once on held-out test data. On a public 47,837-ticket, eight-class corpus, we compare four trained non-LLM classifiers with four large language models under zero- and few-shot prompting. Under the illustrative cost model (c_ok, c_triage, c_wrong) = (0, 1, 5), TF-IDF with logistic regression and a threshold of 0.60 selected on calibration reduces test cost from 0.763 to 0.472, a 38.1% relative reduction (paired ratio-bootstrap 95% CI [35.7%, 40.4%]), while lowering wrong routes from 15.3% to 3.5%. Frozen calibration-selected policies also beat both always-route and always-defer on held-out test data in 14 of 16 settings in a post-hoc cost-parameter analysis. The strongest p
    
[^6]: METTA：面向元强化学习的记忆增强测试时自适应

    METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning

    [https://arxiv.org/abs/2610.31656](https://arxiv.org/abs/2610.31656)

    提出METTA方法，将元强化学习的快速适应分解为片段内的因果神经记忆细粒度自适应与跨片段的紧凑摘要级任务推断两个时间尺度，以线性序列复杂度实现了对未见任务的高效测试时适应。

    

    元强化学习旨在训练能够从有限的交互中快速适应未见任务的智能体，但这需要在长交互历史中对任务证据进行表示。经典的循环神经网络智能体将经验压缩到隐藏状态中，而基于Transformer的智能体则以二次方的内存和计算开销从过往轨迹中检索信息。在本工作中，我们提出了面向元强化学习的记忆增强测试时自适应方法（METTA）。该方法将快速适应阶段分解为两个时间尺度：因果神经记忆以线性序列复杂度执行片段内的细粒度自适应，而紧凑的片段级摘要则支持长程的跨片段任务推断，无需对完整的交互历史进行注意力计算。此外，为智能体配备神经记忆提供了足够的表达能力与测试时训练能力，使其能够有效适应未见的保留任务，并……（摘要在此处截断）

    arXiv:2610.31656v1 Announce Type: new  Abstract: Meta-reinforcement learning aims to train agents that adapt quickly to unseen tasks from limited interaction, but this requires representing task evidence over long interaction histories. Classical recurrent agents compress experience into hidden states, while transformer-based agents retrieve from past trajectories at quadratic memory and computation cost. In this work, we propose Memory-Enhanced Test-Time Adaptation (METTA) for meta-reinforcement learning. It decomposes the fast adaptation phase into two temporal scales: a causal neural memory performs fine-grained within-episode adaptation with linear sequence complexity, while compact episode-level summaries support long-range cross-episode task inference without attending over the full interaction history. Moreover, equipping agents with neural memory provides sufficient expressive capacity and test-time training capability, enabling effective adaptation to unseen held-out tasks and
    
[^7]: EpiSEM-Bench：评估大语言模型智能体在证据缺失下的安全性与证据恢复能力

    EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence

    [https://arxiv.org/abs/2610.31650](https://arxiv.org/abs/2610.31650)

    该论文提出了EpiSEM-Bench基准，通过六个领域中300对匹配的不安全/安全任务和系统性关键事实抑制，评估大语言模型智能体在证据缺失时能否主动恢复证据并安全地拒绝执行被禁止的操作。

    

    工具型语言模型智能体依赖环境事实来判断某个操作是否被允许。当工具响应遗漏了与安全相关的事实时，即使用户请求是良性的且所有返回的陈述都是真实的，智能体也可能执行被禁止的操作。我们提出了EpiSEM-Bench，这是一个可执行的基准测试，用于评估大语言模型智能体在证据缺失情况下的不安全执行、安全边界和证据恢复能力。该基准包含六个领域中共300对匹配的不安全/安全任务对。通过系统性地抑制关键事实，该基准测量了关键证据遗漏情况下的不安全执行，并确定最小观测失败基数，包括在全证据下的失败情况以及经过穷尽测试后未观测到失败的情况。一个独立的审计工具测试智能体是否会主动寻求缺失的事实，并在完成允许任务的同时拒绝执行被禁止的操作。该基准揭示了大量的……（原文摘要截断于此）

    arXiv:2610.31650v1 Announce Type: cross  Abstract: Tool-using language model agents rely on environmental facts to decide whether an action is permitted. When a tool response omits a safety-relevant fact, an agent may carry out a prohibited action even though the user request is benign and every returned statement is true. We introduce \bench{}, an executable benchmark for evaluating unsafe execution, safety boundaries, and evidence recovery in LLM agents under missing evidence. It contains 300 matched unsafe/safe task pairs across six domains. By systematically suppressing critical facts, the benchmark measures unsafe execution under critical evidence omission and determines the minimum observed failure cardinality, including failures with full evidence and cases with no observed failure after exhaustive testing. An independent audit tool tests whether agents seek missing facts and then withhold prohibited actions while completing permissible tasks. The benchmark reveals substantial v
    
[^8]: Softmax与交叉熵的修复方案

    A Fix for Softmax and Cross-Entropy

    [https://arxiv.org/abs/2610.31642](https://arxiv.org/abs/2610.31642)

    该论文发现了softmax、log-softmax和交叉熵标准实现中的两个数值缺陷，并提出了简单的修复方案，该方案消除了“弹弓”现象、加速grokking、防止准确率崩溃并稳定了知识蒸馏过程。

    

    softmax、log-softmax和交叉熵的标准实现存在两个数值缺陷。第一个缺陷是“Softmax坍缩”，当某个logit远大于其他logit时，该目标logit的梯度会被舍入为零，导致梯度之和不再为零。第二个缺陷是“Softmax抵消”，发生在学生logit接近教师logit时，即使学生和教师的logit完全相等，也会产生非零梯度。我们提出了log-softmax、softmax和交叉熵的简单实现，修复了这两个缺陷。我们的修复方法消除了此前观察到的“弹弓”现象，在使用权重衰减时加速了grokking（顿悟）过程，在没有权重衰减时防止了准确率崩溃，阻止了蒸馏过程中学生网络偏离教师网络，并表明在记忆化之后，logit的增长受到权重衰减与Adam的ε超参数之间平衡的约束。

    arXiv:2610.31642v1 Announce Type: new  Abstract: Standard implementations of softmax, log-softmax, and cross-entropy have two numerical bugs. The first bug, Softmax Collapse, rounds the target logit's gradient to zero if that one logit is much larger than the others, resulting in gradients that no longer sum to zero. The second bug, Softmax Cancellation, occurs when student logits are close to teacher logits, and produces non-zero gradients even when the student and teacher logits are equal. We present simple implementations of log-softmax, softmax, and cross-entropy that fix both bugs. Our fix eliminates the previously-observed "Slingshot" phenomenon, speeds up grokking with weight decay, prevents accuracy collapse without weight decay, stops student networks from drifting away from the teacher during distillation, and shows that after memorization, logit growth is bounded by an equilibrium between weight decay and Adam's $\epsilon$ hyperparameter.
    
[^9]: 因果思维链：通过上下文隔离缓解注意力泄露

    Causal Chain of Thought: Mitigating Attention Leakage via Context Isolation

    [https://arxiv.org/abs/2610.31639](https://arxiv.org/abs/2610.31639)

    提出CausalCoT推理框架，将问题分解为原子子问题的有向无环图，并在仅含直接因果父节点的隔离上下文中评估每个节点，从而消除注意力泄露，确保思维链推理步骤对最终答案具有真实的因果驱动作用。

    

    思维链提示能够引导模型生成逐步的推理过程，这些推理过程看似可以解释模型的预测，但无法保证这些步骤在因果上真正驱动了最终答案。在标准的自回归生成中，密集的注意力机制允许每个推理步骤关注完整的上下文，这使得模型能够通过直接查找上下文来推导答案，同时生成看似合理但实为事后辩护的解释。我们提出了CausalCoT，这是一个强制实现因果透明性的推理框架，它将问题分解为原子子问题构成的有向无环图，并在仅包含其直接因果父节点的隔离上下文中评估每个节点。这种马尔可夫结构移除了指向非父节点内容的注意力路径，使得下游预测对局部推理敏感，而非依赖全局上下文泄露。在多样化的推理基准测试中，CausalCoT在保持有竞争力的预测性能的同时，提升了……（摘要原文在此处截断）

    arXiv:2610.31639v1 Announce Type: new  Abstract: Chain-of-thought (CoT) prompting elicits step-by-step rationales that appear to explain model predictions, but there is no guarantee these steps causally drive the final answer. In standard autoregressive generation, dense attention allows each reasoning step to attend to the full context, enabling the model to derive answers through direct lookup while generating plausible-sounding but post-hoc justifications. We introduce CausalCoT, an inference framework that enforces causal transparency by decomposing problems into a directed acyclic graph of atomic sub-questions, evaluating each node in an isolated context containing only its direct causal parents. This Markovian structure removes attention pathways to non-parent content, making downstream predictions sensitive to local reasoning rather than global context leakage. Across a diverse suite of reasoning benchmarks, CausalCoT preserves competitive predictive performance while improving 
    
[^10]: BeNice！通过KV共享缩小循环架构的推理内存差距

    BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing

    [https://arxiv.org/abs/2610.31633](https://arxiv.org/abs/2610.31633)

    本文提出一种两阶段训练方法，通过在预训练中使用随机深度诱导收缩残差动力学，使循环架构实现KV缓存跨递归共享，从而将推理内存从线性增长降为常数而不损失模型质量。

    

    循环架构通过反复将激活值传递经过同一层块，在保持参数量固定的同时提升模型质量。在推理过程中，每次递归都会生成独立的KV缓存，导致内存呈线性增长。为解决这一挑战，先前的工作提出了KV共享（即在深度方向上共享KV缓存），并发现在某些情况下，最后一次递归的KV缓存可以与较早的递归共享而不会造成质量损失，我们将这一特性称为KV可共享性。然而，KV可共享性背后的机制仍然知之甚少，即使是训练方式完全相同的循环架构也不能一致地展现这一特性。本文研究了如何在循环模型中实现KV可共享性。我们提出了一种两阶段方法：在第一阶段，我们在预训练期间使用随机深度，这会诱导收缩残差动力学，使表示在各次递归间趋向一个共同的极限……（摘要在此处被截断）

    arXiv:2610.31633v1 Announce Type: new  Abstract: Looped architectures scale quality while keeping parameter count fixed by repeatedly passing activations through the same block of layers. During inference, each recurrence generates independent KV-caches, causing memory to grow linearly. To address this challenge, prior work has proposed KV-sharing (i.e., sharing KV-caches across depth), finding that, in some cases, the KV-cache of the final recurrence can be shared with the earlier recurrences without quality loss, a property we denote as KV-shareability. However, the mechanisms underlying KV-shareability remain poorly understood, with even identically trained looped architectures not consistently exhibiting it. In this paper, we study how KV-shareability can be enabled in looped models. We propose a two-stage method: in the first stage, we use stochastic depth during pretraining, which induces contractive residual dynamics that drive representations toward a common limit across recurr
    
[^11]: 质疑可验证奖励强化学习（RLVR）中的工具性收敛：来自OLMo-3的检查点证据

    Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3

    [https://arxiv.org/abs/2610.31624](https://arxiv.org/abs/2610.31624)

    本文以OLMo-3-7B-Think在SFT、DPO及13个中间RLVR检查点上的行为轨迹为证据，质疑可验证奖励强化学习是否必然诱发模型的工具性收敛行为，指出该行为可能取决于训练方案、评估框架与后训练阶段。

    

    工具性收敛常被援引为一种普遍性理由，用来预期有能力智能体会获得或表现出追求权力的子目标：自我保存、资源获取、欺骗或抵抗监督，这对社会构成重大威胁。对于大语言模型而言，强化学习是后训练阶段中第一个直接针对目标函数优化行为的环节；因此，从工具性收敛的理论来看，这也是收敛性工具行为最可能出现的阶段。那么，可验证奖励强化学习（RLVR）是否会使模型更倾向于表现出工具性收敛行为？还是说这取决于训练方案、评估框架以及后训练所处的阶段？我们将这一问题作为以检查点为条件的行为轨迹来研究，而非仅以单一的“前后”安全评分来衡量。借助InstrumentalEval，我们对OLMo-3-7B-Think在SFT、DPO、13个中间RLVR检查点以及最终RLVR模型上的表现进行了评估。

    arXiv:2610.31624v1 Announce Type: new  Abstract: Instrumental convergence is often invoked as a general reason to expect capable agents to acquire or express power-seeking subgoals: self-preservation, resource acquisition, deception, or resistance to oversight which poses an important threat to the society. For LLMs, reinforcement learning is the first stage of post-training that directly optimizes behavior against an objective; therefore, it is the stage where convergent instrumental behavior, on IC grounds, likely to appear. Does reinforcement learning with verifiable rewards (RLVR) make a model more likely to express instrumental-convergence behavior, or does this depend on the training regime, evaluation frame, and stage of post-training? We study this question as a checkpoint-conditioned behavioral trajectory rather than as a single before/after safety score. Using InstrumentalEval, we evaluate OLMo-3-7B-Think across SFT, DPO, 13 intermediate RLVR checkpoints, and the final RLVR m
    
[^12]: OP-Print：追踪在线策略自蒸馏中的成员印记

    OP-Print: Tracing Membership Imprints in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.31616](https://arxiv.org/abs/2610.31616)

    本文首次研究了针对在线策略自蒸馏的成员推断攻击，发现其监督印记稀疏且集中于不确定性敏感的修正位点，并据此提出了统一的攻击框架 OP-Print，在灰盒和黑盒威胁模型下均能有效识别训练成员。

    

    在线策略蒸馏已被广泛用于缓解离线策略蒸馏中的分布不匹配问题。自蒸馏则进一步通过使用学生模型自身的输出（辅以额外上下文或演示示例）作为监督，从而免除了对外部教师模型的需求。这种无教师的设计使在线策略自蒸馏变得实用，但其隐私风险在很大程度上仍未被探索。尤其是，对同一训练样本使用两个耦合视图可能会加强模型对该样本的记忆。受此启发，我们开展了首个针对在线策略自蒸馏的成员推断攻击研究。我们研究了灰盒与黑盒两种威胁模型，并提出了一个统一的攻击框架，命名为 OP-Print。OP-Print 建立在我们的一个发现之上：在线策略自蒸馏中的监督印记是稀疏且有结构的，集中于对不确定性敏感的修正位点，而非遍布整个采样序列。利用这一特性……（原文摘要在此处截断）

    arXiv:2610.31616v1 Announce Type: new  Abstract: On-policy distillation has been widely adopted for alleviating the distribution mismatch of off-policy distillation. Self-distillation further removes the need for an external teacher by using the student's own outputs, enhanced with extra context or demonstrations, as supervision. This teacher-free design makes on-policy self-distillation practical, yet its privacy risks remain largely unexplored. In particular, using two coupled views of the same training example may strengthen its memorization. Motivated by this, we present the first study of membership inference attacks against on-policy self-distillation. We investigate both gray-box and black-box threat models and propose a unified attack framework, named OP-Print. OP-Print builds on our finding that the supervision imprint in on-policy self-distillation is sparse and structured, concentrating on uncertainty-sensitive correction sites rather than across the entire rollout. Exploiti
    
[^13]: 具有个人级隐私保护的离散分布高效最优学习

    Efficient and Optimal Learning of Discrete Distributions with Person-Level Privacy

    [https://arxiv.org/abs/2610.31612](https://arxiv.org/abs/2610.31612)

    本文首次在个人级纯差分隐私下实现了批处理设置中学习离散分布的多项式时间算法，其样本复杂度在对数因子内匹配信息论最优界，并在近似差分隐私下同样达到近最优的 $\widetilde O(k)$ 样本复杂度，核心技术贡献是一种新的六阶和平方估计器。

    

    我们研究了在批处理（batched）设置下，在个人级差分隐私约束下学习离散分布的问题。给定 $n$ 个用户，每个用户贡献 $m$ 个来自 $[k]$ 上未知分布 $\mu$ 的独立同分布样本，我们寻求总变差误差为 $\alpha$ 的 $\mu$ 估计量。我们在纯差分隐私和近似差分隐私下均给出了多项式时间算法。在纯 $(\varepsilon,0)$-DP 下，我们达到了 $n = \widetilde O(k/(\alpha^2 m)+k/(\alpha\sqrt{m}\varepsilon)+k/\varepsilon)$ 的样本复杂度，这是首个在对数因子内匹配信息论最优界的多项式时间纯 DP 保证。在近似 $(\varepsilon,\delta)$-DP 下，我们达到了 $n = \widetilde O(k/(\alpha^2 m)+k/(\alpha\sqrt{m}\varepsilon)+k+\log(1/\delta)/\varepsilon)$。当 $\delta=2^{-\Theta(k)}$ 时，对于固定的 $\alpha$、$\varepsilon$、$m$ 和失败概率，样本复杂度为 $\widetilde O(k)$。我们的算法采用了一种新的六阶和平方（degree-6 sum-of-squares）估计器，用于从不可信的批（摘要在此处截断）……

    arXiv:2610.31612v1 Announce Type: new  Abstract: We study learning a discrete distribution under person-level differential privacy in the batched setting. Given $n$ users, each contributing $m$ i.i.d. samples from an unknown distribution $\mu$ over $[k]$, we seek an estimate of $\mu$ with total-variation error $\alpha$. We give polynomial-time algorithms under both pure and approximate differential privacy. Under pure $(\varepsilon,0)$-DP, we achieve $n = \widetilde O(k/(\alpha^2 m)+k/(\alpha\sqrt{m}\varepsilon)+k/\varepsilon)$, the first polynomial-time pure-DP guarantee matching the information-theoretic optimum up to logarithmic factors. Under approximate $(\varepsilon,\delta)$-DP, we achieve $n = \widetilde O(k/(\alpha^2 m)+k/(\alpha\sqrt{m}\varepsilon)+k+\log(1/\delta)/\varepsilon)$. At $\delta=2^{-\Theta(k)}$, this is $\widetilde O(k)$ for fixed $\alpha,\varepsilon,m$ and failure probability. Our algorithms use a new degree-6 sum-of-squares estimator for learning from untrusted b
    
[^14]: CRANE：面向多模态分类器的事后反事实检索

    CRANE: Post-Hoc Counterfactual Retrieval for Multimodal Classifiers

    [https://arxiv.org/abs/2610.31594](https://arxiv.org/abs/2610.31594)

    本文提出首个面向多模态分类器的事后、模型无关的反事实解释框架CRANE，通过从目标类别检索真实训练样本作为解释，在构造上保证了多模态反事实解释的合理性与连贯性。

    

    反事实解释方法此前已被引入单模态场景，针对表格、时间序列、文本和图像数据均有专门的框架，然而尚无针对从多模态输入中学习的分类器所设计的事后方法。现有的多模态分类器可解释性方法对每个输入独立地应用特定模态的归因，这不仅无法产生连贯的解释，也无法迁移到任意预训练的多模态架构上。为解决这些缺陷，我们提出了CRANE（Counterfactual Retrieval via Agnostic Nearest-neighbour Explanations，基于模态无关最近邻解释的反事实检索），这是首个用于最近邻多模态反事实生成的事后、模型无关框架。该框架从目标类别中检索真实训练样本作为候选解释，从而在构造上保证了解释的合理性。在四个异构数据集和三种融合策略上，我们展示了多模态共识检索（MC-R）……

    arXiv:2610.31594v1 Announce Type: new  Abstract: Counterfactual explanation methods have been introduced for unimodal settings, with dedicated frameworks for tabular, timeseries, text, and image data, yet no post-hoc method has been designed for classifiers that learn from multimodal inputs. Existing explainability methods for multimodal classifiers apply modality-specific attribution independently to each input, which fails to yield coherent explanations and does not transfer to arbitrary pre-trained multimodal architectures. To address these shortcomings, we present CRANE (Counterfactual Retrieval via Agnostic Nearest-neighbour Explanations), the first post-hoc, model-agnostic framework for nearest-neighbour multimodal counterfactual generation that retrieves real training examples from the target class as candidate explanations, ensuring plausibility by construction. Across four heterogeneous datasets and three fusion strategies, we show that Multimodal Consensus Retrieval (MC-R), w
    
[^15]: 超越Best-of-$k$：面向歧义非线性逆问题的先验一致DAPS

    Beyond Best-of-$k$: Prior-Consistent DAPS for Ambiguous Nonlinear Inverse Problems

    [https://arxiv.org/abs/2610.31592](https://arxiv.org/abs/2610.31592)

    针对歧义非线性逆问题中测量引导可能收敛到先验不支持的解这一失败情形，本文提出PC-DAPS，通过在单次采样运行中维持多个候选解来保持先验一致性，从而超越了依赖真值比较的best-of-$k$评估方式。

    

    扩散模型已成为求解逆问题的强大先验。然而，当多个测量一致的解只能依靠先验才能区分时，诸如相位恢复等非线性逆问题仍然具有挑战性；我们将这类测量称为“歧义”测量。现有的基于扩散的采样器在重建过程中依赖局部近似。对于歧义测量，这些近似可能使测量引导偏向某个与先验支持的模式接近、但本身并不被先验支持的测量一致模式。评估中常用的best-of-$k$（多次运行取最优）报告方式会掩盖这种失败，因为“最优”结果是通过与真值比较来选择的。我们提出了先验一致DAPS（PC-DAPS），这是解耦退火后验采样（DAPS）的一种变体，旨在通过在单次采样运行中维持多个候选解来解决这一失败情形。随着退火噪声的降低……（摘要在此处截断）

    arXiv:2610.31592v1 Announce Type: cross  Abstract: Diffusion models have become strong priors for solving inverse problems. However, nonlinear inverse problems such as phase retrieval remain challenging when multiple measurement-consistent solutions are distinguishable only by the prior; we call such measurements ambiguous. Existing diffusion-based samplers rely on local approximations during reconstruction. For an ambiguous measurement, these approximations can allow measurement guidance to favor a measurement-consistent mode that is close to a prior-supported mode but is itself not supported by the prior. Evaluation often masks this type of failure by reporting the best-of-$k$ runs, where the best is selected by comparison to the ground truth. We introduce Prior-Consistent DAPS (PC-DAPS), a variant of Decoupled Annealing Posterior Sampling (DAPS) designed to address this failure case by maintaining multiple candidate solutions within a single sampling run. As annealing noise decrease
    
[^16]: 大语言模型能匹配进入住院医师培训吗？开放语言模型在西班牙MIR考试中的不确定性、推理与图像利用

    Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination

    [https://arxiv.org/abs/2610.31586](https://arxiv.org/abs/2610.31586)

    该研究按西班牙MIR住院医师考试的官方规则评估了四个开放大语言模型，发现三个较大模型的成绩可排进前600至3,000名并仍能匹配到41至43个专业，但其弃答策略从不划算，医学微调可减少过度自信，而其临床推理与图像利用能力仍存疑。

    

    arXiv:2610.31586v1 公告类型：新 摘要：在医学考试中的准确率是支持语言模型临床应用的常见证据，然而一个打算在自己的硬件上运行开放模型的医疗系统需要了解的远不止于此：模型是否知道自己何时可能出错，医学领域的微调能否提升这种能力，它的正确答案是否建立在合理的临床推理之上，以及它是否真的用到了提供给它的图像。我们在西班牙MIR住院医师考试上针对四个参数量从80亿到270亿的开放模型考察了这些问题，按照该考试自身的规则对八届考试共1,552道题目进行评分。在回答所有题目的情况下，三个较大的模型将排在约前600至3,000名考生之间，仍有41至43个专业可供其匹配。然而，放弃作答永远不合算，因为即便是它们最没把握的答案，其正确率也高于25%的盈亏平衡点。医学微调降低了过度自信……

    arXiv:2610.31586v1 Announce Type: new  Abstract: Accuracy on medical examinations is common evidence offered for the clinical use of language models, yet a health system that intends to run an open model on its own hardware needs to know considerably more: whether the model recognizes when it is likely to be wrong, whether medical fine-tuning improves this ability, whether its correct answers rest on sound clinical reasoning, or whether it actually uses the images it is given. We examine these questions for four open models of 8 to 27 billion parameters on the Spanish MIR residency examination, scoring 1,552 questions from eight editions under the exam's own rules. Answering every question, the three larger models would rank among roughly the first 600 to 3,000 candidates, with 41 to 43 specialties still open to them. Abstention, however, never pays, because even their least confident answers are right more often than the 25% break-even. Medical fine-tuning reduced overconfidence at th
    
[^17]: 面向科学仿真的动作学习

    Action Learning for Scientific Simulations

    [https://arxiv.org/abs/2610.31585](https://arxiv.org/abs/2610.31585)

    提出 LEAP 方法，通过将未知随机动作锚定到可处理的参考过程，并比较观测轨迹与参考轨迹来学习修正项，为从轨迹观测中恢复动作提供了可微分的路径级目标函数，从而支持科学仿真中动力学系统的推理与优化。

    

    理解并控制系统的演化方式是科学与工程的核心，其中许多问题需要对完整轨迹进行推理。现有方法通常学习局部动力学或直接生成轨迹，但由单个转移构建的目标可能会忽略跨时间的依赖关系，而生成模型也不一定能为评估和优化任意路径提供可处理的目标函数。我们研究了“动作学习”（action learning）问题，即从轨迹观测中恢复随机动作，从而为动力学系统的推理提供一个统一的目标。我们提出了 LEAP（动作路径上的学习能量，Learned Energy on Action Paths），该方法将未知的动作锚定到一个可处理的参考过程上，并通过比较观测轨迹与参考轨迹来学习剩余的修正项。这一构造在路径上产生了一个可微分的目标函数，可用于科学仿真中的动作恢复与路径优化。

    arXiv:2610.31585v1 Announce Type: new  Abstract: Understanding and controlling how systems evolve is central to science and engineering, where many problems require reasoning about complete trajectories. Existing approaches typically learn local dynamics or generate trajectories, but objectives constructed from individual transitions can miss dependencies across time, while generative models do not necessarily provide a tractable objective for evaluating and optimizing arbitrary paths. We study $\textit{action learning}$, the problem of recovering a stochastic action from trajectory observations, yielding a common objective for reasoning about dynamical systems. We introduce $\textbf{LEAP}$ ($\textbf{L}$earned $\textbf{E}$nergy on $\textbf{A}$ction $\textbf{P}$aths), which anchors the unknown action to a tractable reference process and learns the remaining correction by comparing observed trajectories with reference trajectories. This construction yields a differentiable objective on p
    
[^18]: 基于测试时回归的序列压缩再思考

    Rethinking Sequence Compression with Test-time Regression

    [https://arxiv.org/abs/2610.31574](https://arxiv.org/abs/2610.31574)

    提出TTR-Compressor，将序列压缩重新建模为测试时回归问题，其回归系数在前向传播中以闭式解高效计算并通过反向传播端到端学习，在稀疏注意力压缩和多模态视觉token合并任务上显著优于传统的池化与投影压缩方法。

    

    序列压缩是高效语言模型和多模态模型中的一项基础操作，也是关键的信息瓶颈，然而目前主流的基于池化或可学习投影的方法性能仍然不佳。我们从测试时回归的视角重新审视序列压缩：输入token定义了一个回归问题，其拟合得到的系数即构成压缩表示。重要的是，该回归问题通过反向传播进行端到端学习，而其系数则在前向传播过程中以闭式解形式高效计算。由此我们提出了TTR-Compressor，作为传统基于池化和投影的压缩方法的通用替代方案。我们将TTR-Compressor作为块状稀疏注意力的压缩器以及多模态大语言模型的视觉token合并器进行评估。在块状稀疏注意力任务上，TTR-Compressor的RULER性能大幅超越基于注意力的压缩器。

    arXiv:2610.31574v1 Announce Type: new  Abstract: Sequence compression is a fundamental operation and a critical information bottleneck in efficient language and multimodal models, and yet prevailing approaches based on pooling or learned projections remain underperforming. We revisit sequence compression through the lens of test-time regression: input tokens define a regression problem whose fitted coefficients form the compressed representation. Importantly, the regression problem is learned end-to-end via backpropagation, while its coefficients are computed efficiently in closed form during the forward pass. This leads to TTR-Compressor, a general alternative to conventional pooling and projection-based compression. We evaluate TTR-Compressor as the compressor for block-wise sparse attention and the visual token merger for multimodal large language models. On block-wise sparse attention, TTR-Compressor improves RULER performance over attention-based compressors by a large margin. For
    
[^19]: 论文：文本图软提示中的词元坍缩现象

    On the Token Collapse Phenomenon in Textual Graph Soft Prompting

    [https://arxiv.org/abs/2610.31572](https://arxiv.org/abs/2610.31572)

    该论文揭示了图软提示方法G-Retriever存在“词元坍缩”病态现象——图词元在不同输入间几乎无法区分且对答案准确率无实际影响，并通过大量实验证明这一缺陷被后续方法普遍继承。

    

    近期将图结构注入大语言模型（LLM）提示的方法包括图软提示，即由图神经网络（GNN）学习图特定的词元，用以捕获文本提示中实体之间的关系信息。G-Retriever将图软提示与同一图的言语化描述相结合，以增强文本提示信号。我们对G-Retriever进行了深入分析，发现了一种我们称之为“词元坍缩”的病态行为。在三个基准数据集、三种图编码器、两个LLM系列，以及冻结模型和LoRA适配模型上，G-Retriever的图词元在不同输入之间几乎变得无法区分。我们提供证据表明，即使采用多词元池化，后续方法也继承了同样的失败模式。在标准设置下，将某个词元替换为数据集均值词元、来自其他样本的词元，或由损坏的图导出的词元，均不会在准确率上产生答案层面的显著效果。

    arXiv:2610.31572v1 Announce Type: new  Abstract: Recent methods for injecting graph structure into LLM prompts include graph soft prompting, in which a graph neural network (GNN) learns graph-specific tokens that capture relational information among entities in textual prompts. G-Retriever combines graph soft prompts with a verbalized description of the same graph to strengthen the textual prompt signal. We conduct an in-depth analysis of G-Retriever and uncover a pathological behavior we term token collapse. Across three benchmarks, three graph encoders, two LLM families, and both frozen and LoRA-adapted models, G-Retriever's graph tokens become almost indistinguishable across inputs. We provide evidence that the same failure is inherited by subsequent methods, even with multi-token pooling. Under the standard setup, swapping a token with the dataset mean, a token from another example, or one derived from a corrupted graph produces no resolved answer-level effect in accuracy. Moreover
    
[^20]: 何时跟随教师：基于优势的在策略蒸馏

    When to Follow the Teacher: Advantage-Based On-Policy Distillation

    [https://arxiv.org/abs/2610.31563](https://arxiv.org/abs/2610.31563)

    该论文提出基于优势的在策略蒸馏（AOPD），仅利用学生标准回滚轨迹及其终止奖励来估计教师动作在学生策略下的反事实优势，从而只强化真正对学生有益的教师动作，克服了传统在线策略蒸馏中“始终跟随教师”的局限。

    

    arXiv:2610.31563v1 宣布类型：新 摘要：在线策略蒸馏（OPD）通过直接在学生模型自身采样的轨迹上进行训练，其表现往往优于监督微调（SFT）等离线方法。然而，标准的在线策略蒸馏训练学生在轨迹的每个状态上都跟随教师，这隐含地假设教师的指导总是有益的。我们则区分教师所偏好的行为与真正能帮助当前学生取得成功的行为。我们提出了基于优势的在策略蒸馏（AOPD），该方法根据教师动作在学生策略下的反事实优势来为其加权：能够提升学生期望结果的教师动作会被强化，而不能提升的则被降权。直接计算这些优势需要在每一步尝试替代的教师动作，并从每个替代动作出发进行新的回滚采样。与之不同，我们证明AOPD目标可以仅通过标准的学生回滚轨迹及其终止奖励来估计。我们的估计……（摘要原文在此处截断）

    arXiv:2610.31563v1 Announce Type: new  Abstract: Online Policy Distillation (OPD) often outperforms offline methods such as Supervised Fine-Tuning (SFT) by training on trajectories sampled directly from the student. However, standard OPD trains the student to follow the teacher at every state of the trajectory, assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed. We introduce Advantage-Based On-Policy Distillation (AOPD), which weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student's expected outcome are reinforced, while those that do not are downweighted.   Directly computing these advantages requires trying alternative teacher actions at each step and running new rollouts from each. Instead, we show that the AOPD objective can be estimated using only standard student rollouts and their terminal rewards. Our estima
    
[^21]: 面向困惑度感知黑盒攻击的扩散语言模型

    Diffusion Language Models for Perplexity-Aware Black-Box Attacks

    [https://arxiv.org/abs/2610.31558](https://arxiv.org/abs/2610.31558)

    该论文提出将连续潜空间攻击框架与连续扩散语言模型（dLLM）直接耦合，以学习到的语言生成分布取代人工随机映射，从而生成更自然、更难被防护机制检测且基于自然提示的黑盒攻击提示词。

    

    连续空间攻击通过在低维高斯潜空间中进行采样或优化，并将其通过随机映射和最近token查找解码为文本，从而找到能够触发黑盒大语言模型恶意行为（如过度思考）的提示词。由于这种随机映射并非从语言中学习得到，其生成的提示词可能非常不自然，因此容易通过输入防护机制被检测到，且其产生的风险估计也并未建立在自然提示的基础之上。在本文中，我们证明可以将同样的连续潜空间框架直接与从语言中学习得到的生成分布——连续扩散语言模型（dLLM）——相耦合，从而消除人为引入的随机映射，同时保留标准高斯潜空间形式以及下游的采样或优化流程。我们在两种设置下评估了该方法：使用 RareTrap 的基于采样的估计以及…

    arXiv:2610.31558v1 Announce Type: cross  Abstract: \emph{Continuous-space} attacks find prompts that trigger malicious behaviors, such as overthinking, in black-box LLMs by sampling or optimizing over a low-dimensional Gaussian latent space and decoding it to text through a \emph{random mapping} and nearest-token lookup. Because the random mapping is not learned from language, its prompts can be highly unnatural, and as such easily detectable through input guardrails, and the risk estimates they yield are not grounded in natural prompts. In this paper, we demonstrate that the same continuous latent-space framework can be coupled directly to a learned generative distribution over language, a continuous diffusion language model (dLLM), which removes the artificial random mapping while preserving the standard-Gaussian latent formulation and the downstream sampling or optimization procedure. We evaluate this approach under two settings: sampling-based estimation using \emph{RareTrap} and o
    
[^22]: 通过算法稳定性对持续学习中灾难性遗忘的细粒度分析

    Fine-Grained Analysis of Catastrophic Forgetting in Continual Learning via Algorithmic Stability

    [https://arxiv.org/abs/2610.31551](https://arxiv.org/abs/2610.31551)

    本文提出了一个不依赖特定模型架构的算法稳定性理论框架，将持续学习中的灾难性遗忘细粒度地分解为任务内与跨任务泛化差距及经验遗忘，并系统分析了三种学习率调度对遗忘的影响。

    

    理解灾难性遗忘——即学习新任务会导致在先前已学习任务上的性能下降——仍然是持续学习（CL）中的一个基本挑战。现有的理论分析通常依赖于特定的模型或任务结构，对于迭代优化如何影响遗忘所提供的洞察有限。为了填补这一空白，我们开发了一个算法稳定性框架，用于分析SGD下的期望总体遗忘，而无需指定特定的模型架构。具体而言，我们将遗忘分解为任务内和跨任务的泛化差距以及经验遗忘。为了界定泛化差距，我们追踪样本扰动如何影响任务内的更新并在后续任务中传播。我们考虑了恒定学习率、任务内衰减学习率和持续衰减学习率三种调度方式。比较两种衰减调度方式使我们能够刻画在任务开始时重新启动学习率衰减如何影响遗忘（摘要在此处被截断）。

    arXiv:2610.31551v1 Announce Type: new  Abstract: Understanding catastrophic forgetting, in which learning new tasks degrades performance on previously learned tasks, remains a fundamental challenge in continual learning (CL). Existing theoretical analyses often rely on specific model or task structures and offer limited insight into how iterative optimization affects forgetting. To address this gap, we develop an algorithmic stability framework for expected population forgetting under SGD without prescribing a particular model architecture. Specifically, we decompose forgetting into intra-task and cross-task generalization gaps and empirical forgetting. To bound the generalization gaps, we track how a sample perturbation affects updates within a task and propagates through subsequent tasks. We consider constant, intra-task decaying, and continuously decaying learning-rate schedules. Comparing the two decaying schedules allows us to characterize how restarting learning-rate decay at tas
    
[^23]: 面向无需重训练的隐私审计的分布偏移感知分数传输方法

    Shift-aware score transport for privacy auditing without retraining

    [https://arxiv.org/abs/2610.31549](https://arxiv.org/abs/2610.31549)

    提出一种分布偏移感知的分数传输方法，通过从分布外非成员样本中学习并校正成员推断分数分布，实现了在无法重训练模型情况下的隐私审计。

    

    隐私审计通过使用成员推断攻击来建立隐私损失的下界，从而对隐私泄露进行实证评估。我们考虑一种观测式审计场景：审计者无法插入金丝雀样本、无法控制数据是否被纳入训练、也无法重新训练目标模型，而只能依赖固定的已知成员与非成员样本集合。这一场景适用于已部署的大型模型，或重训练受计算资源和访问权限限制的情况。在这种情况下，可用于审计的非成员样本的分布可能与训练数据不同，导致成员推断分数因分布偏移而产生偏差。为解决这一问题，我们利用可获得的分布外（OOD）非成员样本，估计反事实的分布内非成员分数分布。我们提出的方法从OOD非成员样本中学习数据表示与成员推断分数之间的关系，并对分数分布进行传输校正……

    arXiv:2610.31549v1 Announce Type: new  Abstract: Privacy auditing empirically assesses privacy leakage by using membership inference attacks to establish lower bounds on privacy loss. We consider an observational auditing setting in which the auditor cannot insert canaries, control data inclusion, or retrain the target model, and must instead rely on fixed sets of known members and non-members. This setting is relevant for large deployed models or when retraining is restricted by computational or access constraints. In this case, non-members available for auditing may differ in distribution from the training data, causing membership-inference scores to be skewed due to the distribution shift. We address this problem by estimating the counterfactual in-distribution non-member score distribution using the available out-of-distribution (OOD) non-members. Our proposed method learns the relationship between data representations and membership-inference scores from OOD non-members and transp
    
[^24]: 相同的运气，不同的选择：面向智能体强化学习的孪生世界信用分配

    Same Luck, Different Choice: Twin-World Credit Assignment for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.31544](https://arxiv.org/abs/2610.31544)

    提出TWICE方法，通过让反事实孪生回合继承事实回合的采样噪声（即“相同运气、不同选择”），将配对结果差异转化为回合级优势，从而在智能体强化学习中对每个决策实现更准确的信用分配。

    

    智能体强化学习对整个回合给予奖励，但策略需要为其内部的每一个决策获得信用。反事实分支方法通过在做出替代决策后重新运行回合来估计这一信用，但该分支的采样噪声是独立于事实回合抽取的，因此结果差异混合了决策的真实效果与后续所有环节的运气。我们观察到，在智能体搜索中，检索器、语料库和评分器都是确定性的，因此一个回合是策略决策及其采样噪声的确定性函数。因此，孪生世界是可以实现的：相同的噪声，不同的决策。我们提出了TWICE（孪生世界信用估计，TWIn-world Credit Estimation），它按回合和行对每个token的Gumbel噪声进行键控，推出继承事实回合下游噪声的反事实孪生回合，并将配对的结果差异转化为回合级优势，用于策略梯度更新。

    arXiv:2610.31544v1 Announce Type: new  Abstract: Agentic reinforcement learning rewards a complete episode, yet the policy needs credit for every decision inside it. Counterfactual branching estimates this credit by re-running the episode after an alternative decision, but the branch draws its sampling noise independently of the factual episode, so the outcome difference mixes the effect of the decision with the luck of everything that follows. We observe that in agentic search the retriever, the corpus, and the grader are deterministic, so an episode is a deterministic function of the policy's decisions and its sampling noise. Twin worlds are therefore realizable: the same noise, a different decision. We introduce TWICE (TWIn-world Credit Estimation), which keys the Gumbel noise of every token by turn and line, rolls out counterfactual twins that inherit the factual episode's downstream noise, and turns their paired outcome differences into turn-level advantages for policy-gradient up
    
[^25]: PYTHIA：一种用于缺陷与吸附位点同步优化的可迁移机器学习框架

    PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization

    [https://arxiv.org/abs/2610.31539](https://arxiv.org/abs/2610.31539)

    本文提出PYTHIA框架，通过结合基础模型、迁移学习（子空间对齐）与高斯过程不确定性量化，仅需少量目标数据即可将机器学习迁移应用于新颖且研究不足的材料，实现缺陷与吸附位点的同步优化。

    

    机器学习已经深入材料优化领域，并带来了极高的价值。然而，它对大量样本的依赖限制了其在相对已被充分研究的材料族中的应用。在本文中，我们通过PYTHIA（Probabilistic Yet Transferable Hierarchical Interaction Architecture，概率性可迁移分层交互架构）框架，将机器学习的适用范围扩展到新颖且研究不足的材料。PYTHIA利用了两个最强大的机器学习概念：基础模型和迁移学习。我们采用一个在大型化学相似分子数据集上训练的复杂基础模型，其训练目标在物理上与我们的预测目标相近。随后，我们冻结该模型的权重，借助一个transformer层进行子空间对齐，并通过高斯过程引入不确定性量化。这两个附加层仅需在有限的数据集上进行训练，却能让我们充分利用基础模型所积累的知识，将其迁移到所研究的材料体系中。

    arXiv:2610.31539v1 Announce Type: new  Abstract: Machine learning has permeated material optimization, providing high value. Nevertheless, its re- liance on a high number of samples limits its applicability to relatively explored material families. In this paper we extend the applicability of machine learning to novel and understudied materials through the PYTHIA (Probabilistic Yet Transferable Hierarchical Interaction Architecture) framework. Pythia utilizes two of the most powerful machine learning concepts, foundational models and transfer learning. We employ a complex foundational model trained on a large dataset of chemically similar molecules, on a target that is physically close to our prediction target. Then, we freeze the weights and perform subspace alignment with the use of a transformer layer and add uncertainty quantification with gaussian processes. These two layers can be trained on a limited dataset yet allow us to leverage the gained knowl- edge and the studied materia
    
[^26]: LACToS：基于超像素条件典型性的局部异常检测

    LACToS: Local Anomaly Detection via Conditional Typicality of Superpixels

    [https://arxiv.org/abs/2610.31538](https://arxiv.org/abs/2610.31538)

    LACToS通过将像素分组为超像素并采用计算高效的代理来估计统计典型性，实现了一种在多模态分布内数据场景下依然稳健的局部分布外异常区域检测方法。

    

    局部异常检测的任务是在整体上属于分布内（ID）的图像中识别空间上局限的异常或分布外（OOD）区域。大多数现有方法都是针对特定场景设计的，尤其聚焦于工业缺陷检测，并在成熟的基准数据集上取得了优异性能。然而，我们观察到这些方法在其他场景下可能出现显著的性能下降，尤其是当分布内数据的分布呈多模态时。为了应对这种分布内数据的复杂性，我们提出了LACToS，这是一种基于统计典型性的、有原则的局部OOD检测方法，并通过两项创新使其具备可计算性：(i) 将像素分组为超像素；(ii) 使用一种计算高效的代理方法来估计典型性。我们在医疗影像、人脸和X光安检三个不同的应用领域中评估了LACToS。LACToS的表现比先前方法更加稳健。

    arXiv:2610.31538v1 Announce Type: cross  Abstract: Local anomaly detection is the task of identifying spatially confined anomalous or out-of-distribution (OOD) regions within an otherwise in-distribution (ID) image. Most existing methods are designed for a specific context, with a strong focus on industrial defect detection, where they achieve excellent performance on well-established benchmarks. However, we observe that these methods can exhibit significant performance drops in other contexts, especially when the inlier distribution is multimodal. To account for this inlier complexity, we propose LACToS, a principled local OOD detection method based on statistical typicality and made tractable through two innovations: (i) grouping pixels into superpixels and (ii) using a computationally efficient proxy to estimate typicality. We evaluate LACToS across three diverse applications spanning medical imaging, human faces, and X-ray security screening. LACToS performs more robustly than prio
    
[^27]: 可容许集上的直接推断

    Direct Inference on Admissible Sets (Diona)

    [https://arxiv.org/abs/2610.31526](https://arxiv.org/abs/2610.31526)

    Diona是一种无需全局训练、通过按需局部求解约束优化问题直接从样本估计查询点状态及导数的新回归方法，在无噪声数据上精度超越高斯过程，且具备有限样本覆盖等理论保证。

    

    我们提出了一种名为“可容许集上的直接推断”的新回归方法，该方法直接利用样本估计查询点处的状态及其导数，以及它们的可容许界，而无需漫长的全局训练。通过将查询点及邻近样本处的值和导数视为约束优化问题中的潜在变量，长时间训练循环的负担转变为按需的局部求解。我们建立了Diona的理论性质，包括有限样本覆盖率，以及在无噪声情况下、稠密采样条件下收敛到真实状态和导数的性质。在一个包含六个三至八维函数的非线性回归基准测试中，Diona在无噪声数据上比高斯过程（GP）更准确，尤其是在数据集更稠密的情况下。此外，与GP类似，它在四个函数中产生的界在超过95%的查询点处包含了真实值。

    arXiv:2610.31526v1 Announce Type: cross  Abstract: We propose a new regression method named Direct Inference on Admissible Sets (Diona) that directly uses samples to estimate the state and its derivatives at a query point, along with their admissible bounds, without long global training. By casting the values and derivatives at the query and neighbouring samples as latent variables in a constrained optimisation problem, the burden of long training loops becomes one of on-demand local solves. We establish Diona's theoretical properties including finite-sample coverage and, in the noiseless case, convergence to the true state and derivatives under dense sampling. In a nonlinear regression benchmark on six functions of three to eight dimensions, Diona was more accurate than a Gaussian process (GP) on noiseless data especially with denser data sets. Furthermore, as with GP, it produced uncertainty bounds that contained the true value at more than 95% of query points in four functions. Dion
    
[^28]: 基于并行梯度提升的概率分布非参数自回归估计

    Nonparametric autoregressive estimation of probability distributions with parallel gradient boosting

    [https://arxiv.org/abs/2610.31524](https://arxiv.org/abs/2610.31524)

    该论文提出通过自回归因子分解将联合概率分布的估计转化为条件分布估计，并证明在满足次可加性的损失函数下，自回归估计器的收敛速度界可由条件分布估计器的收敛速度界导出，从而将联合分布估计的难度归结为自回归分解中最难因子的估计难度，并借助并行梯度提升实现高效估计。

    

    联合概率分布的非参数估计是统计学中的一个核心问题，在高维数据下这一问题以困难著称。类似的挑战也出现在条件分布估计中，由此催生了一批灵活的算法，它们通过利用维数低于环境空间的预测结构来应对“维数灾难”。在本文中，我们研究如何利用这类条件分布估计器，通过以“自回归”方式对联合分布进行因子分解来估计联合分布。我们证明，对于满足次可加性性质的损失函数，自回归估计器的收敛速度界可以从条件分布估计器的收敛速度界推导得出。这些界将估计联合分布的难度与估计自回归分解中最困难因子的难度联系起来。（摘要原文在此处截断）

    arXiv:2610.31524v1 Announce Type: cross  Abstract: Nonparametric estimation of joint probability distributions is a core problem in statistics, which is notoriously challenging with high-dimensional data. Such challenges also arise with conditional distribution estimation and have led to flexible algorithms that aim to tackle the ''curse of dimensionality'' by exploiting predictive structures of lower dimension than the ambient space. In this paper, we study the use of such conditional distribution estimators to estimate joint distributions by factorizing the latter in an ''autoregressive'' way. We show that for loss functions satisfying a subadditivity property, bounds on the convergence rate of an autoregressive estimator can be derived from bounds on the convergence rate of the conditional distribution estimators. These bounds relate the difficulty of estimating the joint distribution to that of estimating the most challenging factor in the autoregressive factorization. When the con
    
[^29]: Piano2Pop：无需对齐监督将钢琴演示扩展为流行音乐制作

    Piano2Pop: Expanding Piano Demos into Pop Productions Without Aligned Supervision

    [https://arxiv.org/abs/2610.31510](https://arxiv.org/abs/2610.31510)

    该论文提出Piano2Pop任务，通过伪监督的流匹配模型将钢琴MIDI扩展为完整的器乐流行音乐制作，无需钢琴与流行音乐的对齐配对监督，且微调后的模型在真实数据上显著优于零样本基线。

    

    我们提出了Piano2Pop，这是一项音乐扩展任务，可将钢琴MIDI转换为器乐流行音乐作品。得益于音乐生成与多声部音乐转录领域的最新进展，我们的伪监督方法以钢琴卷帘（piano roll）为条件对预训练的流匹配模型进行调控，而无需对齐的钢琴-流行音乐配对监督。我们在从流行音乐录音中派生的合成钢琴-器乐配对数据上进行训练，将无配对转录方法（包括功能选择性增强FST和音区密度增强RDA）与配对监督的钢琴翻弹生成器作为伪演示来源进行了比较。在476组真实世界的钢琴翻弹/流行音乐配对测试集上，配对监督方法在真实钢琴输入上取得了最佳的Fréchet音频距离（FAD）、伴奏提示依从性和频谱形状表现；而FST在无配对训练中缓解了钢琴音过重的问题，仍是一种有前景的替代方案。所有微调后的系统均显著优于零样本基线。

    arXiv:2610.31510v1 Announce Type: new  Abstract: We introduce Piano2Pop, a music-expansion task that transforms piano MIDI into instrumental pop productions. Enabled by recent advances in music generation and polyphonic music transcription, our pseudo-supervised approach conditions a pretrained flow-matching model on piano rolls without aligned piano-pop supervision. We train on synthetic piano-instrumental pairs derived from pop recordings, comparing pair-free transcription, including function-selective (FST) and register-density (RDA) augmentations, with a pair-supervised piano-cover generator as pseudo-demo sources. On 476 testing real-world piano-cover/pop pairs, the pair-supervised approach achieves the best Frechet Audio Distance, Accompaniment Prompt Adherence, and spectral shape on real-piano inputs, while FST mitigates piano-heavy outputs in pair-free training and remains a promising alternative. All fine-tuned systems significantly outperform the zero-shot baselines, which su
    
[^30]: 超声速与高超声速气流中薄板的结构保持降阶模型

    Structure-preserving reduced-order models of thin panels in supersonic and hypersonic flow

    [https://arxiv.org/abs/2610.31504](https://arxiv.org/abs/2610.31504)

    本文为AePW-4高速工作组的两个挑战问题（马赫1.92固支薄板与马赫5.85悬臂板）构建了结构保持的降阶模型，其学习模型在构造上严格保证能量平衡（最差相对残差仅2.2e-15），并准确复现了激波压力比1.521（与压敏漆实测一致）和主频（偏差小于1.0%）。

    

    我们为AePW-4高速工作组的两个挑战问题构建了结构保持的降阶模型。这两个问题分别是马赫1.92下的固支薄板（RC-19）和马赫5.85下的悬臂板（HyMAX）。每个算例的公开数据既没有时间分辨的响应，也没有重复运行的结果，因此物理特性必须由模型的形式本身来承载。通过一种构造方法，每个问题得到两个模型：一个描述板的Ritz-Galerkin参考模型，以及一个学习模型——后者保留参考模型的精确势能，拟合剩余的广义力，并且只训练一个无法注入能量的耗散项。其能量平衡在构造上对任意权重均成立。在训练后的模型上，最差相对残差为2.2e-15。在RC-19算例中，激波关系预测板两侧的压力比为1.521，压敏漆测量得到了相同的数值。学习模型的三十次运行使平均主导频率与（原文在此处截断）的偏差控制在1.0%以内。

    arXiv:2610.31504v1 Announce Type: cross  Abstract: We build structure-preserving reduced-order models for both challenge problems of the AePW-4 High-Speed Working Group. They are a clamped thin panel at Mach 1.92 (RC-19) and a cantilever plate at Mach 5.85 (HyMAX). The published data for each case hold no time-resolved response and no repeated runs. The physics must therefore be carried by the form of the model. One construction gives two models per problem. A Ritz-Galerkin reference model describes the plate. A learned model keeps the reference model's exact potential, fits the remaining generalized forces, and trains only a dissipation that cannot add energy. Its energy balance holds for any weights by construction. On the trained model the worst relative residual is 2.2e-15. On RC-19, shock relations predict a pressure ratio of 1.521 across the panel. Pressure-sensitive paint measured the same value. Thirty runs of the learned model place the mean dominant frequency within 1.0% of t
    
[^31]: SABLE：可扩展的仿射贝叶斯潜空间探索

    SABLE: Scalable Affine Bayesian Latent-space Exploration

    [https://arxiv.org/abs/2610.31495](https://arxiv.org/abs/2610.31495)

    SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。

    

    诸如工程设计、蛋白质等复杂对象往往难以优化，因为它们的设计空间呈现高维结构，且目标评估代价高昂。生成模型提供了连续的潜空间表示，但其潜空间可能具有数千至数万维，形成了潜空间维度 D 远超评估预算的情形。我们提出 SABLE，一种针对各向同性分布且预算有限的高维潜空间的贝叶斯优化方法。SABLE 使用贝叶斯仿射代理模型进行预测与采集，并将高阶结构视为代理模型的偏差。在 N 次观测之后，斜率更新的维度至多为 N-1，从而将每个查询分解为一个至多 N 维的跨度内分量（通过期望改进 EI 进行优化）和一个由标量范数支配的跨度外分量。我们表明……

    arXiv:2610.31495v1 Announce Type: new  Abstract: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show tha
    
[^32]: 训练小型语言模型实现可迁移的工具选择

    Training Small Language Models for Transferable Tool Selection

    [https://arxiv.org/abs/2610.31483](https://arxiv.org/abs/2610.31483)

    本文通过强化学习训练8B小模型完成工具选择任务，性能超越402B的Llama-4-Maverick，并可直接迁移至不同检索器和跨领域的新工具语料库。

    

    随着大语言模型智能体需要在一万多个工具的语料库上运行，为复杂的多步用户指令选择合适的工具成为一个核心瓶颈。工具描述无法全部放入上下文中，而基于相关性的检索器只能对工具进行排序，并不知道哪些工具是实际需要的。最近的工作转而利用诸如 Claude 或 ChatGPT 等专有商业大语言模型来驱动检索器，分解用户指令并从检索到的候选工具中进行选择。我们证明，可以通过强化学习训练一个小型开源模型来执行相同的功能。在 ToolBench 上，使用我们的方案进行后训练的 Qwen3-8B 模型优于 Llama-4-Maverick（402B）。此外，训练好的模型可以直接迁移到不同的检索器，并能泛化到不同领域的新工具语料库。

    arXiv:2610.31483v1 Announce Type: new  Abstract: As LLM agents operate over corpora of more than ten thousand tools, selecting the right tool for a complex multi-step user instruction becomes a central bottleneck. Tool descriptions cannot all fit in context, and relevance-based retrievers only rank tools without knowing which are actually needed. Recent work instead leverages proprietary commercial LLMs like Claude or ChatGPT to drive a retriever, decomposing the instruction and selecting over retrieved candidates. We show that a small, open model can be trained via reinforcement learning to perform the same functionality. On ToolBench, a Qwen3-8B model post-trained with our recipe outperforms Llama-4-Maverick (402B). In addition, the trained model transfers directly to different retrievers and generalizes to new tool corpora in different domains.
    
[^33]: 评估大语言模型推理优化在苹果芯片上的适用性

    Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon

    [https://arxiv.org/abs/2610.31479](https://arxiv.org/abs/2610.31479)

    该论文系统评估了KV缓存剪枝、投机解码、移植GPU内核和训练后量化四类大语言模型推理优化在苹果芯片上的适用性，发现其表现与服务器级NVIDIA GPU上显著不同——KV缓存剪枝在苹果芯片上可将端到端吞吐量提升至多1.75倍，而投机解码反而会降低吞吐量。

    

    苹果芯片是本地大语言模型推理使用最广泛的平台之一，然而几乎所有的推理优化都是在服务器级NVIDIA GPU上、针对批量服务工作负载进行设计和验证的。当这些优化迁移到苹果芯片上时，哪些优化仍能保持其收益，又是什么因素决定了它们的适用性？我们评估了四类优化方法：KV缓存剪枝、投机解码、移植的GPU内核以及训练后量化。实验在四台苹果机器上进行，采用统一的单请求解码工作负载，并在可能的情况下与RTX 4090进行对等比较。KV缓存剪枝在苹果芯片上可实现最高1.75倍的端到端吞吐量提升，但在桌面级NVIDIA GPU上反而可能降低吞吐量。在统一基准测试下，所有被评估的投机解码方法在苹果芯片上均降低了吞吐量，尽管在服务器GPU上报告有2-6倍的收益。CUDA内核设计只能部分迁移：公开的Metal缺乏异步拷贝支持，且批量优……

    arXiv:2610.31479v1 Announce Type: new  Abstract: Apple Silicon is among the most widely used platforms for local LLM inference, yet nearly all inference optimizations are designed and validated on server-class NVIDIA GPUs under batched serving workloads. Which optimizations retain their benefits when moved to Apple Silicon, and what determines their applicability? We evaluate four optimization families: KV-cache pruning, speculative decoding, ported GPU kernels, and post-training quantization. Experiments span four Apple machines under a common single-request decode workload, with matched RTX 4090 comparisons where possible. KV-cache pruning can reach up to 1.75 times end-to-end throughput on Apple but can reduce throughput on desktop NVIDIA GPUs. Under the common benchmark, all evaluated speculative decoding methods reduce throughput on Apple, despite reported 2-6 times gains on server GPUs. CUDA kernel designs transfer only partially: public Metal lacks asynchronous copy, and batch-o
    
[^34]: MoG-VLM：通过运动引导的视觉语言模型实现高效视频异常检测

    MoG-VLM: Efficient Video Anomaly Detection through Motion-Guided Vision-Language Models

    [https://arxiv.org/abs/2610.31478](https://arxiv.org/abs/2610.31478)

    该论文提出MoG-VLM框架，将时间梯度幅度（TGM）作为轻量级运动控制信号进行内容感知采样，并仅在高异常可能性的帧附近激活密集的多帧VLM推理，从而在保持检测精度的同时显著提升视频异常检测的实时效率。

    

    视觉语言模型（VLM）在视频异常检测方面展现出强大潜力，但密集的帧级推理使其难以实现实时部署。现有的加速方法通常依赖均匀采样，这种方式忽略了内容的动态变化，可能将计算浪费在静态片段上，或遗漏局部化的异常。我们提出MoG-VLM，一种运动自适应的异常检测框架，它将运动纯粹用作轻量级控制信号，以决定何时需要进行VLM推理。为此，我们引入了时间梯度幅度（TGM），用于执行内容感知采样，将计算集中在信息丰富的区域，同时避免冗余处理。我们进一步引入了分层双流设计，其中密集的多帧VLM推理仅在异常可能性升高的帧附近被激活，通过锚点触发的上下文窗口扩展及提前终止机制实现。融合模块随后…（摘要在此处被截断）

    arXiv:2610.31478v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) have shown strong potential for video anomaly detection, but dense frame-level inference makes them impractical for real-time deployment. Existing acceleration commonly relies on uniform sampling, which ignores content dynamics and can either waste computation on static segments or miss localized anomalies. We propose MoG-VLM a motion-adaptive anomaly detection framework that leverages motion purely as a lightweight control signal to decide when VLM inference is necessary. To this end, we introduce Temporal Gradient Magnitude (TGM) which performs content-aware sampling to concentrate computation on informative regions while avoiding redundant processing. We further introduce a hierarchical dual-stream design in which dense multi-frame VLM reasoning is activated only around frames with elevated anomaly likelihood, via anchor-triggered context-window expansion with early termination. A fusion module then rec
    
[^35]: RheOPT：通过对流变算子变换器进行牛顿流体预训练实现少样本非牛顿流体预测

    RheOPT: Few-Shot Non-Newtonian Flow Prediction via Newtonian Pretraining of RheOFormer

    [https://arxiv.org/abs/2610.31465](https://arxiv.org/abs/2610.31465)

    RheOPT通过在廉价的牛顿流体模拟上预训练变换器神经算子、再用少量非牛顿样本微调，并重用几何条件化的输运结构而非本构响应本身，实现了数据高效的少样本非牛顿流体预测。

    

    尽管数值方法不断进步，非牛顿流体模拟在计算上仍然代价高昂，因为速度和应力必须通过依赖于剪切速率和历史历史的本构动力学来共同推进。神经算子代理模型可以在训练后降低推理成本，但其训练数据恰恰需要它们想要替代的那种昂贵模拟，因此当只能负担少量非牛顿模拟时，数据高效的学习变得至关重要。在此，我们提出了流变算子预训练变换器（Rheological Operator Pretrained Transformer，RheOPT），它在廉价的牛顿流体模拟上预训练一个基于变换器的神经算子，然后在少量非牛顿样本上进行微调。RheOPT并非迁移牛顿本构响应本身，而是重用几何条件化的输运结构，这种结构在应力模型改变后依然有用。我们在两种几何匹配的设置中评估RheOPT：一个4:1收缩通道……

    arXiv:2610.31465v1 Announce Type: new  Abstract: Despite advances in numerical methods, non-Newtonian flow simulations remain computationally expensive because velocity and stress must be advanced together through rate- and history-dependent constitutive dynamics. Neural-operator surrogates reduce inference cost after training, but their training data require the same expensive simulations they aim to replace, making data-efficient learning essential when only a few non-Newtonian simulations can be afforded. Here, we introduce the Rheological Operator Pretrained Transformer (RheOPT), which pretrains a transformer-based neural operator on inexpensive Newtonian simulations and fine-tunes it on a small number of non-Newtonian samples. Rather than transferring the Newtonian constitutive response itself, RheOPT reuses geometry-conditioned transport structure that remains useful after the stress model changes. We evaluate RheOPT in two geometry-matched settings: a $4{:}1$ contraction channel
    
[^36]: 多Token预测与替代分词方法提升大规模潜在多跳图搜索的学习能力

    Multi-Token Prediction and Alternative Tokenization Improve Learning Latent Multi-Hop Graph Search at Scale

    [https://arxiv.org/abs/2610.31439](https://arxiv.org/abs/2610.31439)

    本文指出模型难以学习大规模图上最短路径搜索的两个原因——监督信号与分词方式，并提出多Token预测和替代分词这两种相互正交的方法，突破了以往工作的规模瓶颈。

    

    给定一个未见过的图、一个起始节点 $s$ 和一个终止节点 $t$，语言模型能否生成从 $s$ 到 $t$ 的最短路径？先前的研究发现，随着图规模的扩大，模型无法学会这一任务。我们证明这种失败有两个原因：监督信号和分词方式。首先，我们从理论上建立并从实证上证明了多Token预测（MTP）的有效性，并表明MTP的益处与图的拓扑结构无关。其次，我们引入了替代分词方法，并证明这两个原因是相互正交的。利用这些方法，我们突破了先前研究观察到的规模瓶颈，在远超以往极限的规模上持续且稳定地学习解决该任务。

    arXiv:2610.31439v1 Announce Type: cross  Abstract: Given an unseen graph, a start node, $s$, and an end node, $t$, can a language model generate the shortest path from $s$ to $t$? Prior work has found that models fail to learn this task as the graphs scale. We show that this failure has two causes: the supervision signal and the tokenization. First, we theoretically establish and empirically demonstrate the effectiveness of multi-token prediction (MTP), and show that the benefits of MTP are topologically agnostic. Second, we introduce alternative tokenizations and show that the two causes are orthogonal. Using these methods, we break through the scaling wall observed by prior work, consistently and stably learning to solve the task at scales far beyond previous limits.
    
[^37]: DIAL：具有蒸馏指令遵循级别的可引导语言模型

    DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels

    [https://arxiv.org/abs/2610.31438](https://arxiv.org/abs/2610.31438)

    提出DIAL框架，利用无分类器引导的在线策略蒸馏和轻量级引导适配器，对大语言模型遵循指令的程度实现细粒度数值控制，同时兼顾效率与跨指令的可扩展性。

    

    当用户说“保持简短”时，大语言模型的回复应该有多短？更广泛地说，用户和应用程序对指令遵循的程度有不同的要求，但提示工程无法可靠地提供对这些程度的细粒度控制。因此，我们研究“可引导语言模型”，它能够对模型遵循文本中指定的一般指令的程度提供细粒度的数值控制。然而，现有方法无法同时在多样化指令上实现细粒度控制、效率和可扩展性。为了弥补这一空白，我们提出了DIAL（蒸馏指令遵循级别），以同时实现这三个特性。为了实现细粒度控制，我们使用冻结的大语言模型执行无分类器引导，为不同级别指令遵循的在线策略蒸馏提供训练信号。为了提高效率，我们设计了一个轻量级的引导适配器，它使用基于大语言模型隐藏状态的分类器来……

    arXiv:2610.31438v1 Announce Type: cross  Abstract: When a user says "Keep it short", how short should the LLM's response be? More generally, users and applications require different levels of instruction adherence, but prompting does not reliably provide fine-grained control over them. We therefore study "steerable language models" that provide fine-grained numerical control over the degree to which models follow general instructions specified in text. However, existing approaches do not jointly offer fine-grained control, efficiency, and scalability across diverse instructions. To address this gap, we introduce DIAL (Distilled Instruction-Adherence Levels) to achieve all three properties. For fine-grained control, we perform classifier-free guidance (CFG) with a frozen LLM to provide training signals for on-policy distillation of instruction adherence at different levels. For efficiency, we design a lightweight steering adapter that uses a classifier over the LLM's hidden states to co
    
[^38]: 保守元智能体搜索：非平稳性下的架构自适应

    Conservative Meta-Agent Search: Architecture Adaptation under Non-Stationarity

    [https://arxiv.org/abs/2610.31436](https://arxiv.org/abs/2610.31436)

    本文提出保守元智能体搜索（CMAS），将非平稳环境下的元智能体架构搜索形式化为带切换成本与硬风险约束的序贯现任者替换问题，通过配对影子回放评分、方差惩罚晋升裕度、金丝雀验证与回滚机制实现安全稳健的架构自适应，并配套引入MetaInvest-Bench状态切换模拟基准。

    

    自动化智能体系统设计通常在具有即时标量奖励的静态任务套件上进行评估。然而实际部署的系统面临持续的环境漂移、延迟且相互依赖的反馈、组件间的相关性故障以及代价高昂的架构变更。我们将元智能体架构搜索形式化为带有切换成本和硬风险约束的序贯现任者替换问题，并提出保守元智能体搜索（CMAS）。CMAS从存档中检索架构并生成类型化的图编辑，随后在共同随机数条件下通过配对影子回放并结合干预式组件信用归因，将每个候选架构与现任架构进行对比评分。候选架构的晋升需要其经验校准的、经方差惩罚的同步得分超过预先指定的裕度，晋升后还需经过模拟金丝雀阶段并配备预先指定的回滚机制。鉴于回测无法证明元层面的能力，我们引入了MetaInvest-Bench——一个状态切换模拟器，其指定的工作者能力……（原文摘要在此处截断）

    arXiv:2610.31436v1 Announce Type: cross  Abstract: Automated agentic-system design is typically evaluated on static task suites with immediate scalar rewards. Deployed systems face persistent shift, delayed dependent feedback, correlated component failures, and costly architecture changes. We formalize meta-agent architecture search as sequential incumbent replacement with switching costs and hard risk constraints and introduce Conservative Meta-Agent Search (CMAS). CMAS retrieves archived architectures and proposes typed graph edits, then scores each against the incumbent by paired shadow replay under common random numbers, with interventional component credit. Promotion requires an empirically calibrated, variance-penalized simultaneous score above a pre-specified margin, and is followed by a simulated canary stage and pre-specified rollback. Since backtests cannot certify meta-level skill, we introduce MetaInvest-Bench, a regime-switching simulator whose specified worker competence 
    
[^39]: 漂移决定时机，预算决定数量：面向长篇多模态生成的速率受控KV重选

    Drift Says When, Budget Says How Much: Rate-Controlled KV Re-selection for Long-Form Multimodal Generation

    [https://arxiv.org/abs/2610.31433](https://arxiv.org/abs/2610.31433)

    提出速率受控刷新（RCR）方法，用注意力漂移决定KV缓存视觉token何时重选、用逐样本控制器调节刷新频率，从而显著提升多模态大模型长篇生成的质量。

    

    面向多模态大语言模型的KV缓存压缩通常在预填充阶段一次性选出需保留的视觉token，并在整个生成过程中复用这一选择。我们证明这种一次性选择在长篇生成任务上存在严重局限：在详细图像描述任务上，即使是使用未压缩答案的注意力进行选择的oracle，与未压缩输出的ROUGE-L一致性也仅为0.44-0.52，因为模型读取的证据会随着描述的展开而不断变化。每当注意力漂移出保留集合时重新选择是自然的补救办法，但我们发现固定的漂移阈值会系统性地失效：漂移幅度在不同样本之间不可比，导致低漂移样本几乎从不刷新，尽管它们恰恰受益最大。我们提出速率受控刷新（Rate-Controlled Refresh, RCR），由漂移决定何时何地重新选择，同时由逐样本控制器调节刷新频率，并结合低秩漂移检测器……（摘要截断）

    arXiv:2610.31433v1 Announce Type: new  Abstract: KV cache compression for multimodal LLMs usually selects the visual tokens to keep once, at prefill, and reuses that selection for the whole generation. We show that this one-shot selection is severely limited on long-form generation: on detailed image description, even an oracle that selects with the attention of the uncompressed answer reaches only 0.44-0.52 ROUGE-L agreement with the uncompressed output, because the evidence the model reads moves as the description unfolds. Re-selecting whenever attention drifts out of the kept set is the natural remedy, but we find that a fixed drift threshold fails systematically: drift magnitudes are not comparable across samples, so low-drift samples are almost never refreshed although they benefit the most. We propose Rate-Controlled Refresh (RCR), which lets drift decide when and where to re-select while a per-sample controller regulates how often, combined with a low-rank drift detector, candid
    
[^40]: 岭可分模型中未调整朗之万算法的无维度区间：基于子空间鲁棒Wasserstein距离的分析

    Dimension-free regimes for unadjusted Langevin in ridge-separable models: A subspace robust Wasserstein analysis

    [https://arxiv.org/abs/2610.31431](https://arxiv.org/abs/2610.31431)

    该论文利用子空间鲁棒Wasserstein距离分析未调整朗之万算法在岭可分模型中的偏差，证明在过参数化和大样本区间下算法复杂度可摆脱对维度的依赖。

    

    未调整朗之万算法（ULA）被广泛应用于高维贝叶斯推断。基于Wasserstein-2距离的经典分析预测了随维度增长的渐近偏差和混合时间，但这一度量对于许多统计任务而言强于实际需要。我们转而分析子空间鲁棒Wasserstein距离下的偏差，该距离能够控制均值和任意低维边缘分布，因此非常适合实践中通常估计的观测量。对于由独立同分布数据构成的岭可分势函数，我们证明了高概率界，表明当数据点数量N相对维度d变化时，混合过程呈现钟形曲线。在过参数化区间中，有效维度是数据依赖扰动的秩N，使得复杂度依赖于N而非d。在大样本区间中，高斯扰动估计产生了无维度的复杂度……

    arXiv:2610.31431v1 Announce Type: cross  Abstract: The Unadjusted Langevin Algorithm (ULA) is widely used for high-dimensional Bayesian inference. Classical analyses in Wasserstein-2 distance predict an asymptotic bias and a mixing time that grow with dimension, but this metric is stronger than necessary for many statistical tasks. We instead analyze the bias in subspace robust Wasserstein distance, which controls the mean and arbitrary low-dimensional marginals and is therefore well suited to observables typically estimated in practice. For ridge-separable potentials formed with i.i.d. data, we prove high-probability bounds showing a bell-shaped mixing profile as the number of data points $N$ varies relative to the dimension $d$. In the overparameterized regime, the effective dimension is the rank $N$ of the data-dependent perturbation, leading to complexity depending on $N$ rather than $d$. In the large-sample regime, Gaussian-perturbation estimates yield dimension-free complexity on
    
[^41]: 可预测的性能是次优的，以及这对模型路由的意义

    Predictable performance is suboptimal, and what it means for model routing

    [https://arxiv.org/abs/2610.31428](https://arxiv.org/abs/2610.31428)

    本文证明性能预测只有在候选模型本身表现次优或预测器具有显著更强能力时才能超越模型自身的置信度基线，且预测器的优势可通过重新校准在对数损失上被精确转移，这对模型路由方法的设计具有重要启示。

    

    大型语言模型（LLM）在能力和成本上各不相同，这促使人们开发能够为每个查询自适应选择模型的路由方法。这些方法依赖于性能预测：给定一个查询，估计某个模型是否能正确回答该查询。大多数路由方法通过训练独立的预测器来预测候选模型的性能，以指导模型选择。我们的工作表明，只有当候选模型在任务上表现次优，或者性能预测器相对于候选模型具有显著更强的能力时，性能预测才能超越一个简单的基线。具体而言，我们以模型的自我置信度（即模型分配给其所选答案的概率）作为基线，研究性能预测器能在多大程度上对其有所改进。首先，我们证明任何优于模型自我置信度的性能预测器都可以用来改进模型的损失：通过重新校准，预测器的优势可以在对数损失上被精确地转移。

    arXiv:2610.31428v1 Announce Type: new  Abstract: Large language models (LLMs) differ in capabilities and costs, motivating routing methods that adaptively choose a model for each query. These methods rely on performance prediction: given a query, estimate whether a model will answer it correctly. Most routing methods train separate predictors of candidate models' performance to guide model selection. Our work shows that performance prediction only beats a simple baseline when the candidate model is suboptimal on the task or the performance predictor adds significant capacity relative to the candidate model. Specifically, we take a model's self-confidence, the probability it assigns to its chosen answer, as a baseline and study how much a performance predictor can improve on it. First, we show that any performance predictor that outperforms a model's self-confidence can be used to improve the model's loss: the predictor's advantage transfers exactly in log loss through recalibration. Se
    
[^42]: 深度增长与循环如何塑造语言模型中的上下文学习

    How Depth Growth and Looping Shape In-Context Learning in Language Models

    [https://arxiv.org/abs/2610.31421](https://arxiv.org/abs/2610.31421)

    深度增长与循环语言模型通过跨层复用计算形成了偏向从上下文学习的归纳偏置，使其比稠密模型更早获得且更强保持上下文学习能力，其机制在于深度增长将早期归纳头传播到各层。

    

    循环与深度增长语言模型通过在深度方向上复用计算，并因其推理能力而受到关注。基于先前关于上下文学习增强的提示性证据，我们研究了这种复用如何有利于上下文学习，包括它如何塑造该能力的获取、保持及内在机制。我们的分析涵盖高达30亿参数的预训练语言模型以及受控任务。实验为一种共享的、偏向于从上下文学习的归纳偏置提供了证据：深度增长和循环模型更依赖上下文中的信息，与全深度稠密参考模型相比，它们在受控任务中更早获得上下文学习能力，并且即使在权重中学习固定关联也成为可能的情况下，仍能更强地保持这种能力。通过追踪训练过程中的注意力头，我们发现深度增长会将早期形成的归纳头传播到各个层。消融实验表明，深度增长和循环模型对归纳……（摘要原文在此处截断）

    arXiv:2610.31421v1 Announce Type: new  Abstract: Looped and depth-grown language models reuse computations across depth and have attracted interest for their reasoning capabilities. Building on prior suggestive evidence of improved in-context-learning (ICL), we investigate how such reuse favors ICL, including how it shapes the acquisition, retention, and mechanisms of this ability. Our analysis spans pretrained language models up to 3B parameters and controlled tasks. The experiments provide evidence for a shared inductive bias toward learning from context: grown and looped models rely more on information in context and, compared with a full-depth dense reference, acquire ICL earlier in controlled tasks and retain it more strongly when learning fixed associations in the weights is also possible. Tracing attention heads through training reveals that depth growth propagates early induction heads across layers. Ablations show that grown and looped models rely more strongly on induction-ra
    
[^43]: 用于神经网络可解释性的事后特征选择层

    Post-Hoc Feature Selection Layer for Neural Networks Interpretability

    [https://arxiv.org/abs/2610.31420](https://arxiv.org/abs/2610.31420)

    提出了一种事后特征选择层，可在完全不改变已部署模型架构和参数的前提下，以无侵入的方式识别出模型认为最重要的特征，从而提升复杂神经网络的可解释性。

    

    复杂神经网络的可解释性仍然是一个关键挑战，尤其是对于已经部署在高风险领域的模型。为了解决这一问题，我们将特征选择层（FSL）改造为在训练后优化特征权重的模块，并将其称为事后FSL。我们的方法将FSL重新构建为一个轻量级、可训练的模块，该模块与已冻结的预训练模型在表格数据集上集成，以突出显示原始模型认为最重要的特征。这个事后FSL通过基于原始模型学习到的输出微调其自身权重来学习特征相关性。至关重要的是，这一过程是无侵入性的，在不改变原始模型架构或其已学习参数的情况下运行。我们使用统计和视觉指标进行了实验，包括准确率、F1分数、召回率、精确率、加权t-SNE和轮廓系数，并分析了事后FSL在高维数据上的稳定性。

    arXiv:2610.31420v1 Announce Type: new  Abstract: The interpretability of complex neural networks remains a critical challenge, especially for models already deployed in high-stakes domains. To address this, we adapted the Feature Selection Layer (FSL) to optimize feature weights post-training, a variant we term post-hoc FSL. Our approach reframes the FSL as a lightweight, trainable module that integrates with already frozen pre-trained models on tabular datasets to highlight the features the original model considers most important. This post-hoc FSL learns feature relevance by fine-tuning its weights based on the original model's learned outputs. Crucially, this process is non-invasive, operating without altering the original model's architecture or its learned parameters. We conducted our experiments using both statistical and visual metrics, including accuracy, F1 score, recall, precision, weighted t-SNE and silhouette score, and also analyzed the stability of the post-hoc FSL on hig
    
[^44]: SQuAT：球面格向量量化感知训练

    SQuAT: Spherical Lattice Vector Quantization-Aware Training

    [https://arxiv.org/abs/2610.31412](https://arxiv.org/abs/2610.31412)

    该论文提出SQuAT方法，通过分离的直通估计器将权重分解为方向（用E_8球面码本量化）和增益（用低比特标量量化器量化），将基于格的向量量化感知训练成功扩展到十亿参数级大语言模型，并实现了隐式的球面（黎曼）优化。

    

    向量量化（VQ），特别是基于格的方法，为低比特大语言模型（LLM）提供了强大的率失真权衡，但其在量化感知训练（VQ-QAT）下的行为仍未被充分理解。我们将格VQ-QAT扩展至十亿参数规模的LLM和数十亿token的训练场景，并提出SQuAT——一种形状-增益式VQ-QAT方法，采用针对形状和增益的分离直通估计器（STE）进行训练。SQuAT将权重分解为方向和幅度两个部分：归一化方向使用由$E_8$推导的球面码本进行量化，而增益则使用低比特标量量化器独立量化。我们证明，与应用于完整量化器、忽略形状-增益结构的标准STE不同，这些分离的STE会在切空间中为方向引入更新，从而形成一种隐式的球面（黎曼）优化形式。为实现可扩展性，我们开发了一种分布式……（原文截断）

    arXiv:2610.31412v1 Announce Type: cross  Abstract: Vector quantization (VQ), in particular lattice-based methods, offer strong rate-distortion trade-offs for low-bit large language models (LLMs), yet their behavior under quantization-aware training (VQ-QAT) remains poorly understood. We scale lattice VQ-QAT to billion-parameter LLMs and multi-billion-token training regimes, and introduce SQuAT, a shape-gain VQ-QAT approach trained with a separate straight-through estimator (STE) for the shape and gain. SQuAT decomposes weights into direction and magnitude: normalized directions are quantized using an $E_8$-derived spherical codebook, while gains are quantized independently with low-bit scalar quantizers. We show that, unlike a standard STE applied to the full quantizer, which ignores the shape-gain structure, these separate STEs induce tangent-space updates for directions, yielding an implicit form of spherical (Riemannian) optimization. To enable scalability, we develop a distributed 
    
[^45]: HERD：面向持续学习的工具链演化递归蒸馏

    HERD: Harness-Evolving Recursive Distillation for Continual Learning

    [https://arxiv.org/abs/2610.31401](https://arxiv.org/abs/2610.31401)

    提出HERD混合持续学习框架，通过可演化的工具链高效学习，并以双锚点漂移约束蒸馏将经验内化为参数化累积能力，从而在持续获取新知识的同时大幅减少遗忘。

    

    持续学习要求模型在不损害既有能力的前提下不断获取新技能与新知识。参数化方法面临灾难性遗忘和可塑性丧失的困扰；非参数化方法（如提示、技能和工具链）虽借助上下文学习，却受限于上下文长度和有损的纯文本压缩。为有效结合二者的互补优势，我们提出HERD（工具链演化递归蒸馏）——一个混合式持续学习框架，它在工具链中高效学习，并通过蒸馏将经验内化为累积能力。在工具链方面，HERD持续精炼其技能以捕捉可复用的模式，同时在有限记忆中保留技能无法覆盖的高残差样本；在参数方面，HERD采用带双锚点的漂移约束蒸馏，在长递归过程中显著减少遗忘。

    arXiv:2610.31401v1 Announce Type: new  Abstract: Continual learning requires a model to keep acquiring new skills and knowledge without compromising its existing capabilities. Parametric methods suffer from catastrophic forgetting and loss of plasticity. Non-parametric methods, such as prompts, skills, and harnesses, leverage in-context learning but are limited by context length and lossy plain-text compression. To effectively combine their complementary strengths, we propose HERD (Harness-Evolving Recursive Distillation), a hybrid continual learning framework that learns efficiently in the harness and internalizes experience into cumulative capabilities via distillation. On the harness side, HERD continually refines its skills to capture reusable patterns, while retaining high-residual examples that the skills fail to capture in a bounded memory. On the parameter side, HERD performs drift-constrained distillation with dual anchors, substantially reducing forgetting over long recursive
    
[^46]: 用于Monge–Ampère方程的凸平衡势：曲率界与后验评估

    Convex Equilibrium Potentials for Monge--Amp\`ere: Curvature Bounds and A Posteriori Evaluation

    [https://arxiv.org/abs/2610.31386](https://arxiv.org/abs/2610.31386)

    本文针对用于Monge–Ampère方程的凸softplus平衡神经网络，证明了Hessian矩阵关于压缩因子具有最优的 $(1-q)^{-2}$ 增长界，并给出了基于残差的后验估计以控制近似平衡点处的状态及其导数误差。

    

    平衡神经网络将其隐藏状态定义为压缩映射的不动点，因此其有效深度是在评估时确定的，而非由网络架构决定。将这类网络用于完全非线性偏微分方程会带来两个显式网络可以避免的问题：二阶导数可能随有效深度增长，且势函数只能被近似求值。我们研究具有压缩因子 $q<1$ 的非负 softplus 平衡网络，这是一类适合 Monge–Ampère 方程的凸函数类。通过预解算子中的正性抵消，我们证明 Hessian 矩阵至多以 $(1-q)^{-2}$ 的速度增长，而不是直接估计所给出的 $(1-q)^{-3}$，并且一个显式的单神经元函数族表明该指数是最优（不可改进）的；三阶导数则满足一个最优的 $(1-q)^{-3}$ 界。随后，基于残差的后验估计可以控制近似平衡点处的状态、其输入导数，以及在二维情形下 Hessian 行列式的误差。

    arXiv:2610.31386v1 Announce Type: cross  Abstract: Equilibrium neural networks define their hidden state as the fixed point of a contraction, so their effective depth is set at evaluation time rather than by the architecture. Using them for fully nonlinear PDEs raises two issues that explicit networks avoid: second derivatives may grow with the effective depth, and the potential is only ever evaluated approximately. We study nonnegative softplus equilibrium networks with contraction factor $q<1$, a convex class suited to the Monge--Amp\`ere equation. A positivity cancellation in the resolvent shows that the Hessian grows at most like $(1-q)^{-2}$, rather than the $(1-q)^{-3}$ of a direct estimate, and an explicit one-neuron family shows that this exponent is sharp; third derivatives satisfy a sharp $(1-q)^{-3}$ bound. Residual-based a posteriori estimates then control the state, its input derivatives and, in two dimensions, the Hessian determinant at an approximate equilibrium. Combine
    
[^47]: AfterDeleteBench：审计大语言模型智能体中删除后的信息持久性

    AfterDeleteBench: Auditing Post-Deletion Information Persistence in LLM Agents

    [https://arxiv.org/abs/2610.31366](https://arxiv.org/abs/2610.31366)

    本文提出AfterDeleteBench受控审计基准，揭示了LLM智能体删除原始记忆记录后，其派生状态（如摘要、分散存储的组件）仍可在约三分之二的案例中导致已删除的秘密信息被恢复，证明“删除”操作并不能保证信息真正被移除。

    

    许多LLM智能体维护外部持久记忆，并从过去的交互中派生出额外的状态，例如摘要。因此，删除原始记忆记录并不一定会移除由其派生内容所承载的信息。我们提出了AfterDeleteBench，这是一个针对固定权重LLM智能体中该失效模式的受控行为审计。对于每个合成的秘密信息，我们将一个先暴露后删除该秘密的智能体，与一个其他条件完全匹配但从未接触过该秘密的智能体进行比较，从而能够将删除后的信息恢复归因于最初的暴露。在我们的主要设置中，单次查询即可在大约三分之二的情况下从Qwen2.5-7B-Instruct中恢复已删除的秘密，而匹配的从未暴露对照组的恢复率为零。至关重要的是，恢复并不依赖于存活的逐字副本：当完整的秘密字符串已不存在、仅有按公开组合规则分别存储的组件时，精确重构的成功率仍达到62-（摘要在此处被截断）

    arXiv:2610.31366v1 Announce Type: cross  Abstract: Many LLM agents maintain external persistent memory and derive additional state from past interactions, such as summaries. Deleting an original memory record therefore need not remove information carried by its descendants. We introduce AfterDeleteBench, a controlled behavioral audit for this failure mode in fixed-weight LLM agents. For each synthetic secret, we compare an exposed-then-deleted agent with an otherwise matched agent that was never exposed to the secret, allowing post-deletion recovery to be attributed to the original exposure. In our primary setting, a single query recovers the deleted secret from Qwen2.5-7B-Instruct in roughly two thirds of cases, while matched never-exposed controls have zero recovery. Crucially, recovery does not require a surviving verbatim copy: when the complete secret string is absent and only separately stored components remain under a public composition rule, exact reconstruction succeeds in 62-
    
[^48]: 涌现集体动力学的图-循环学习

    Graph-Recurrent Learning of Emergent Collective Dynamics

    [https://arxiv.org/abs/2610.31350](https://arxiv.org/abs/2610.31350)

    该论文提出了一种将门控循环记忆与图消息传递相结合的图-循环神经网络（GRNN）框架，能够在无需已知微观运动方程的情况下直接从轨迹中学习关联电子系统的涌现集体动力学，并在半经典 Holstein 模型的电荷密度波动力学上成功再现了时空关联等动力学行为。

    

    关联电子系统中的集体有序性源自微观动力学，然而可观测纹理的演化未必容许一个封闭的、无记忆的描述。我们引入了一个图-循环神经网络（GRNN）框架，该框架直接从轨迹中学习此类涌现动力学，而无需指定微观运动方程。该框架将门控循环记忆与图消息传递相结合：局部隐状态编码了超出瞬时可观测量的信息，而空间耦合使这些信息能够在格点之间传播。可观测场与隐场通过共享的局部规则共同演化，为具有空间相互作用和时间记忆的动力学提供了可扩展的表示。我们在半经典 Holstein 模型中电荷密度波有序的涌现动力学上验证了该框架，成功再现了时间与空间关联以及动力学结……（摘要在此处被截断）

    arXiv:2610.31350v1 Announce Type: cross  Abstract: Collective ordering in correlated electron systems emerges from microscopic dynamics, yet the evolution of observable textures need not admit a closed, memoryless description. We introduce a graph-recurrent neural network (GRNN) framework that learns such emergent dynamics directly from trajectories without specifying microscopic equations of motion. The framework integrates gated recurrent memory with graph message passing: local hidden states encode information beyond the instantaneous observables, while spatial coupling allows this information to propagate between sites. The observable and hidden fields evolve jointly through a shared local rule, providing a scalable representation of dynamics with spatial interactions and temporal memory. We demonstrate this framework on the emergent dynamics of charge-density-wave order in the semiclassical Holstein model, successfully reproducing temporal and spatial correlations and dynamical sc
    
[^49]: 基于条件期望局部半径的度量保形预测

    Metric Conformal Prediction via the Conditional Expected Local Radius

    [https://arxiv.org/abs/2610.31336](https://arxiv.org/abs/2610.31336)

    该论文提出条件期望局部半径（CELR）这一新度量，构建了适用于一般度量空间中非欧几里得响应的保形预测方法，结合度量分布随机森林实现高效计算，并从理论上证明了估计量一致性、渐近条件有效性及有限样本边际覆盖率。

    

    我们提出了条件期望局部半径（CELR），并开发了一种适用于一般度量空间中非欧几里得响应变量的新型保形预测方法。CELR 度量局部条件概率的集中程度，使预测集能够适应异质性和多峰性，而无需依赖密度估计或向量空间结构。其条件分布产生了一个在不同协变量取值之间具有可比性的非一致性得分，并避免了距离轮廓之间代价高昂的传输比较。我们进一步改造了度量分布随机森林，使其能够处理多变量协变量和非线性交互作用，同时单次森林拟合即可提供所有得分组成部分，通过消除重复的局部拟合来降低计算成本。我们在适当的正则性条件下建立了估计量的一致性和渐近条件有效性，并在可交换性假设下获得了有限样本边际覆盖率。

    arXiv:2610.31336v1 Announce Type: cross  Abstract: We introduce the conditional expected local radius (CELR) and develop a new conformal prediction method for non-Euclidean responses in general metric spaces. CELR measures local conditional probability concentration, allowing prediction sets to accommodate heterogeneity and multimodality without requiring densities or a vector space structure. Its conditional distribution yields a nonconformity score that is comparable across covariate values and avoids costly transport comparisons between distance profiles. We further adapt a metric distributional random forest that handles multivariate covariates and nonlinear interactions, while a single forest fit supplies all score components, reducing computation by eliminating repeated local fitting. We establish uniform consistency of the estimators and asymptotic conditional validity under suitable regularity conditions, together with finite-sample marginal coverage under exchangeability. Nume
    
[^50]: 有害回复即越狱模板：将自回归越狱攻击迁移至扩散语言模型

    Harmful Responses Are Jailbreak Templates: Transferring Autoregressive Jailbreaks to Diffusion Language Models

    [https://arxiv.org/abs/2610.31331](https://arxiv.org/abs/2610.31331)

    本文发现有害回复经部分掩码后可天然充当越狱模板，并据此提出RMT-Jailbreak框架，将现有的自回归模型越狱攻击有效迁移到扩散语言模型上。

    

    扩散大语言模型通过双向上下文建模实现了灵活的掩码生成能力，但这一能力同时也引入了独特的越狱攻击面。我们发现，针对给定查询的有害回复天然可以充当有效的越狱模板：只需对其中的部分词元进行掩码处理，就能诱导扩散语言模型重构出有害内容。基于这一观察，我们提出了RMT-Jailbreak，这是一个通用框架，可以将现有的自回归（AR）越狱攻击通过其诱导生成的有害回复迁移到扩散语言模型上。RMT-Jailbreak采用均匀词元保留策略，将AR模型生成的有害回复转换为高掩码比例的模板，并进一步应用基于提示的安全过滤来掩码所有剩余可见的有害内容。精炼后的模板随后被提供给目标扩散语言模型进行双向重构。在多种AR越狱方法和目标扩散语言模型上开展的大量实验表明……

    arXiv:2610.31331v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) enable flexible masked generation through bidirectional context modeling, but this capability also introduces a distinct jailbreak attack surface. We find that a harmful response to a given query naturally serves as an effective jailbreak template: simply masking a subset of its tokens can induce a dLLM to reconstruct harmful content. Based on this observation, we propose RMT-Jailbreak, a general framework that transfers existing autoregressive (AR) jailbreak attacks to dLLMs through their induced harmful responses. RMT-Jailbreak converts an AR-generated harmful response into a high-mask-ratio template using uniform token retention, and further applies prompt-based safety filtering to mask any remaining visible harmful content. The refined template is then provided to the target dLLM for bidirectional reconstruction. Extensive experiments across multiple AR jailbreak methods and target dLLMs demons
    
[^51]: 度量空间中随机对象的深度学习：超越流形假设

    Deep Learning for Random Objects in Metric Spaces: Beyond Manifold Assumptions

    [https://arxiv.org/abs/2610.31328](https://arxiv.org/abs/2610.31328)

    该论文提出了一个直接在响应空间度量结构上运行的随机对象回归深度学习框架，无需流形或欧几里得嵌入假设，同时具备统计一致性与收敛速率保证。

    

    深度学习（DL）方法已在众多下游预测任务中显著提升了预测性能，为多个领域带来了新的发现。然而，尽管以概率分布、随机图、函数以及度量空间中其他随机对象形式呈现的现代生物医学结果日益重要，针对非欧几里得随机对象的原则性深度学习方法仍然相对匮乏。受新一代数字与临床生物标志物的启发，我们提出了一种用于度量空间中随机对象回归的深度学习框架。我们的方法直接在响应空间的度量结构上进行操作，避免了现有许多方法所施加的流形或欧几里得嵌入等限制性假设。同时，在既定的统计假设下，该方法保留了现代统计保证。我们为所提出的方法建立了统计一致性与收敛速率（原文摘要在此处截断）。

    arXiv:2610.31328v1 Announce Type: cross  Abstract: Deep learning (DL) methods have substantially improved predictive performance across a wide range of downstream prediction tasks, enabling new discoveries in multiple domains. However, principled DL methods for non-Euclidean random objects remain relatively scarce, despite the growing importance of modern biomedical outcomes taking the form of probability distributions, random graphs, functions, and other random objects in metric spaces. Motivated by a new generation of digital and clinical biomarkers, we propose a deep learning framework for regression with random objects in metric spaces. Our approach operates directly on the metric structure of the response space, avoiding restrictive manifold or Euclidean embedding assumptions imposed by many existing methods. At the same time, it retains modern statistical guarantees under the stated statistical assumptions. We establish statistical consistency and convergence rates for the propos
    
[^52]: AgenticEvolve：基于持久化语言模型智能体设计并运行实验的进化程序搜索

    AgenticEvolve: Evolutionary Program Search with Persistent Language Model Agents that Design and Run Experiments

    [https://arxiv.org/abs/2610.31326](https://arxiv.org/abs/2610.31326)

    AgenticEvolve让持久化的大语言模型智能体在进化程序搜索的评估间隙自主设计并运行实验，从而以更少的评估次数和相当的成本生成显著更优的程序。

    

    使用大语言模型的进化程序搜索保留了进化算法的结构，并将模型用作其变异算子。通常情况下，模型根据父代程序及其评分，通过单次请求生成每个候选程序。我们提出了AgenticEvolve，这是一个进化程序搜索框架，其中每个岛屿由一个持久化的大语言模型智能体运行。在两次评估之间，该智能体会针对候选程序设计并运行实验。为了衡量实验环节的贡献，我们将AgenticEvolve与No-analysis（一种除禁用实验外完全相同的配置）进行比较。在使用三种大语言模型的两个启发式设计任务中，AgenticEvolve在隐藏和未见过的实例上都生成了比No-analysis和OpenEvolve更好的程序。它以少得多的评估次数和相当的成本达到了基线的最终质量。在十二个方程发现问题（原文在此处截断）……

    arXiv:2610.31326v1 Announce Type: cross  Abstract: Evolutionary program search with large language models keeps the structure of an evolutionary algorithm and uses the model as its variation operator. Typically, the model generates each candidate program in a single request from parent programs and their scores. We present AgenticEvolve, an evolutionary program search framework in which each island is run by a persistent large language model agent. Between evaluations, the agent designs and runs experiments on the candidate program. To measure the contribution of experimentation, we compare AgenticEvolve with No-analysis, an otherwise identical configuration with experiments disabled. On two heuristic design tasks with three large language models, AgenticEvolve produces better programs than No-analysis and OpenEvolve on both hidden and unseen instances. It reaches the baselines' final quality with far fewer evaluations and comparable cost. On twelve equation discovery problems, it reco
    
[^53]: 弥合数据鸿沟：面向少样本扩散的源信息任务空间传输

    Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion

    [https://arxiv.org/abs/2610.31323](https://arxiv.org/abs/2610.31323)

    提出 Meta-Diff，通过在紧凑任务空间中一次性学习可复用的源-目标关系，使共享扩散主干无需针对每个目标域再训练即可从稀疏目标数据快速生成，尤其适用于医疗等目标数据难以获取的场景。

    

    许多实际部署场景要求模型能够快速地从数据丰富的源域迁移到仅有稀疏观测的相关目标域。当目标数据难以收集时，这一问题尤为突出，例如在存在持续人群或采集不平衡的医疗场景中。为每个目标域重新训练或微调高容量生成器代价高昂，且当目标数据稀缺时在统计上可能不可靠。我们提出 Meta-Diff，它转而在一个紧凑的任务空间中一次性学习反复出现的源-目标关系：一个共享编码器将每个分布映射为低维坐标；一个学习到的关系将新的源坐标传输到其预测的目标坐标，随后共享的扩散主干直接从该预测进行生成，无需针对特定目标进行再训练。受控的高斯混合实验揭示了该方法生效的条件：任务身份必须在表示中得以保留，关系必须能够在不同任务间泛化，并且源-

    arXiv:2610.31323v1 Announce Type: new  Abstract: Many deployments require a model to move rapidly from a data-rich source to a related target observed only sparsely. This is especially consequential where target data are difficult to collect, as in healthcare settings with persistent population or acquisition imbalances. Retraining or fine-tuning a high-capacity generator for every target is costly and can be statistically unreliable when target data are scarce. We introduce Meta-Diff, which instead learns a recurring source--target relation once in a compact task space. A shared encoder maps each distribution to a low-dimensional coordinate; a learned relation transports a new source coordinate to its predicted target, and a shared diffusion backbone generates from that prediction without target-specific retraining. Controlled Gaussian-mixture experiments identify when this works: task identity must survive the representation, the relation must generalise across tasks, and the source-
    
[^54]: 系统1模型在医学中的价值

    The Value of System 1 Models in Medicine

    [https://arxiv.org/abs/2610.31315](https://arxiv.org/abs/2610.31315)

    本研究首次大规模评估了四个系统1模型在20种语言、14个临床领域共86个任务上的零样本表现，表明这类能快速输出概率并具备不确定性估计能力的模型，在临床决策中具有独特价值。

    

    许多临床决策是在少数固定选项之间做出的选择，需要快速、大量地完成。此类决策还需要不确定性估计，以便知道何时应转交给临床医生处理。生成式语言模型以自由文本形式回答速度较慢，且对自身置信度的表达不佳。相比之下，系统1模型能够以零样本方式作答，并为每个选项给出概率。然而，这类模型在各类临床任务中的表现尚不清楚。本研究评估了四个系统1模型（两个专有模型和两个开放权重模型）在86个公开任务上的零样本表现，这些任务来自14个临床领域、20种语言，涵盖考试题目和医学影像。在分析之前，任务被划分为六个能力层级。主要结局指标为受试者工作特征曲线下面积（AUROC），次要结局指标包括校准度、语言效应，以及与已发表结果和生成式模型的比较。在179,111个病例中，最佳模型的平均AUROC……（摘要原文在此处截断）

    arXiv:2610.31315v1 Announce Type: new  Abstract: Many clinical decisions are a choice among a few fixed options, taken quickly and in large numbers. Such decisions also need an uncertainty estimate to know when to defer to a clinician. Generative language models answer slowly in free text and state their confidence poorly. In contrast, System 1 models answer zero-shot with a probability for each answer. However, their performance across clinical tasks is unknown. Four System 1 models, two proprietary and two with open weights, were evaluated zero-shot on 86 public tasks from 14 clinical domains in 20 languages, spanning examination questions and medical imaging. Tasks were assigned to six capability tiers before analysis. The primary outcome was the area under the receiver-operating-characteristic curve (AUROC). Secondary outcomes included calibration, language effects and comparisons with published results and generative models. Across 179,111 cases, the mean AUROC of the best model p
    
[^55]: 具有互惠作用与隐变量的线性随机微分方程中基于平稳协方差的可识别性研究

    Identifiability from Stationary Covariance in Linear SDEs with Reciprocal Interactions and Latent Variables

    [https://arxiv.org/abs/2610.31310](https://arxiv.org/abs/2610.31310)

    该论文为允许互惠相互作用（即因果图中含 有向2-环）的线性随机微分方程首次建立了基于平稳协方差的因果效应可识别性图判定条件，并进一步提出工具变量框架，解决了存在隐变量时部分可观测条件下目标直接因果效应的识别问题。

    

    连续时间随机微分方程（SDE）为建模相互作用的动力系统提供了一个自然的框架。一个核心的因果问题是：在已知底层因果图的条件下，能否从来自平稳分布的观测数据中识别出观测变量之间的因果效应。对于线性随机微分方程而言，现有的可识别性结果大多局限于完全观测的系统，且要求其因果图中不包含有向2-环。我们首先为可能包含有向2-环的系统建立了因果效应一般可识别性的新图判定条件，从而允许系统中存在互惠（双向）相互作用。接着，我们研究了此前未被探索的隐变量情形，开发了一个工具变量框架，用于在部分可观测条件下识别目标直接因果效应。

    arXiv:2610.31310v1 Announce Type: new  Abstract: Continuous-time stochastic differential equations (SDEs) provide a natural framework for modeling interacting dynamical systems. A central causal question is whether causal effects among observed variables can be identified from observational data drawn from the stationary distribution, given the underlying causal graph. For linear SDEs, existing identifiability results are largely restricted to fully observed systems whose causal graphs contain no directed 2-cycles. We first establish new graphical conditions for the generic identifiability of causal effects in systems that may contain directed 2-cycles, thereby allowing for reciprocal interactions. Next, we address the previously unexplored setting of latent variables by developing an instrumental-variable framework for identifying a target direct causal effect under partial observability.
    
[^56]: 通过跨层潜在向量量化实现逐层嵌入的极致压缩

    Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization

    [https://arxiv.org/abs/2610.31309](https://arxiv.org/abs/2610.31309)

    提出OPTIC方法，通过为每个token仅存储一个共享量化编码元组并用层特定仿射解码器在每层解码，实现对逐层嵌入约100倍的极致压缩，同时保留未压缩模型97.8%的性能，远超传统量化方法。

    

    将嵌入容量与Transformer计算同步扩展是语言模型中的一个新兴趋势，这一趋势使得高压缩率变得越来越有用。因此，我们研究了逐层嵌入的极致压缩，这类嵌入顺应了这一趋势，占Gemma 4 E4B模型参数的35.2%。在这样的压缩率下，传统量化器会为每个token-层向量分配仅几个字节的独立编码。我们转而开发了OPTIC（适配输出路径的token共享索引编码，Output-Path-fitted, Token-shared Indexed Codes），它为每个token仅存储一个量化编码元组，并在每一层使用该层特定的仿射解码器进行解码。编码与解码器的拟合对象是PLE的输出路径而非所存储的向量本身。在对bfloat16（BF16）的E4B模型进行约100倍PLE压缩时，OPTIC在七个基准测试中总计保留了未压缩模型97.8%的性能，而六个传统基线中最强的乘积量化仅保留了66.3%。（注：原文摘要在此处被截断）

    arXiv:2610.31309v1 Announce Type: new  Abstract: Scaling embedding capacity alongside Transformer computation is an emerging trend in language models. This trend makes high compression rates increasingly useful. Hence, we study extreme compression of Per-Layer Embeddings (PLE), which follow this trend and account for 35.2% of the parameters of Gemma 4 E4B. At such rates, conventional quantizers give each token-layer vector its own code of only a few bytes. We instead develop OPTIC (Output-Path-fitted, Token-shared Indexed Codes), which stores one quantized code tuple per token and decodes it at every layer with a layer-specific affine decoder. Codes and decoders are fitted to the PLE output path rather than to the stored vectors. At about $100\times$ PLE compression of bfloat16 (BF16) E4B, OPTIC retains an aggregate 97.8% of the uncompressed model's performance across seven benchmarks, whereas product quantization, the strongest of six conventional baselines, retains 66.3%. At $594\tim
    
[^57]: 超越KL几何的贝叶斯学习规则

    The Bayesian Learning Rule Beyond KL Geometry

    [https://arxiv.org/abs/2610.31308](https://arxiv.org/abs/2610.31308)

    本文提出了一类混合空间Bregman/Fenchel-Young差异度量，据此推导出广义贝叶斯学习规则（BLR），突破了KL几何的限制，并能够导出KL散度无法得到的新型加性优化算法。

    

    贝叶斯学习规则（BLR）利用贝叶斯更新与Kullback-Leibler散度（KLD）之间的关系，为指数族变分后验推导出高效的加性更新。KL几何虽然强大但有局限性：当与具有参数相关支撑集的后验族结合时，可能导致目标函数无定义。我们探究哪些非KL几何能够保持加性的BLR计算。先前使用f-散度的研究工作表明，这只有在使用KLD时才可能实现。在本文中，我们证明事实并非如此。我们展示了一类混合空间Bregman散度或Fenchel-Young差异度量，可以据此推导出加性的信息几何优化算法。这些差异度量诱导出广义最大熵族、对偶坐标以及广义BLR规范梯度更新，并恢复了经典的BLR。广义BLR催生出了无法从KL散度推导得到的新算法。

    arXiv:2610.31308v1 Announce Type: new  Abstract: The Bayesian learning rule (BLR) exploits the relation between Bayesian updating and the Kullback-Leibler divergence (KLD) to derive efficient additive updates for exponential family variational posteriors. The KL geometry is powerful but restrictive: combined with posterior families with parameter-dependent support it can lead to undefined objectives. We ask which non-KL geometries preserve the additive BLR computation. Previous works that use $f$-divergences suggest that this is only possible with the KLD. Here, we show that this is not true. We show that additive information-geometric optimization algorithms can be derived for the class of mixed-space Bregman or Fenchel-Young discrepancies. These discrepancies induce generalized maximum-entropy families, dual coordinates, and generalized-BLR canonical gradient updates, recovering the classical BLR. The generalized BLR leads to new algorithms that \textit{cannot} be derived from the KL
    
[^58]: 面向遥感的数据驱动视觉-语言模型：综述

    Data-Driven Vision-Language Models for Remote Sensing: A Survey

    [https://arxiv.org/abs/2610.31304](https://arxiv.org/abs/2610.31304)

    本文是一篇以数据为中心的遥感视觉-语言模型综述，系统梳理了高质量遥感视觉-语言数据集的构建方法，并指导研究人员进行有效的模型微调与性能评估，填补了该领域缺乏系统性数据导向调研的空白。

    

    在深度学习时代，遥感（RS）视觉模型的创新主要集中在针对特定任务优化网络架构以及进行端到端训练。然而，随着视觉-语言模型（VLMs）时代的到来，受限于巨大的计算资源需求，主流研究范式已转向利用遥感垂直领域数据对通用视觉-语言模型进行微调适配，而非修改其核心架构。这一转变标志着创新路径从以模型为中心向以数据为中心的转变。尽管数据的重要性日益凸显，学术界仍然缺乏一份系统的、以数据为中心的综述，来指导研究人员针对各种应用场景构建高质量的遥感视觉-语言数据集，并基于这些数据集进行有效的模型微调和性能评估。为填补这一空白，本文提出了一……

    arXiv:2610.31304v1 Announce Type: cross  Abstract: During the deep learning era, innovations in remote sensing (RS) vision models primarily focused on optimizing network architectures for specific tasks and conducting end-to-end training. However, with the advent of the Vision-Language Models (VLMs) era, constrained by substantial computational resource demands, the mainstream research paradigm has shifted towards adapting general-purpose VLMs using RS vertical domain data for fine-tuning, rather than modifying their core architectures. This transition marks a change in the innovation pathway from model-centric to data-centric. Despite the increasing importance of data, the academic community still lacks a systematic, data-centric survey to guide researchers in constructing high-quality RS vision-language datasets for various application scenarios and in conducting effective model fine-tuning and performance evaluation based on these datasets. To address this gap, this paper presents a
    
[^59]: DISC：面向时间序列数据集的域集成相似性代价

    DISC: Domain-Integrated Similarity Cost for Time-Series Dataset

    [https://arxiv.org/abs/2610.31288](https://arxiv.org/abs/2610.31288)

    提出了DISC——一个可学习的相似性度量聚合框架，利用校准迁移观测为不同下游任务自动选择、加权和组合多种相似性度量，使源-目标相似性得分与下游预测误差的关联更加一致，从而无需穷举评估即可实现数据集迁移性比较。

    

    时间序列数据集和基础模型的增长使得数据集比较成为时间序列工作流程的核心环节。在模型适应之前评估源域到目标域的迁移，需要能够概括与迁移相关的数据集关系的相似性估计，而无需对每一个候选数据集对进行穷举式的源-目标评估。目前已有许多度量方法被提出用于衡量时间序列数据集之间的相似性，但它们所捕捉的数据集关系各不相同，且在不同任务上与下游效用的一致程度也存在差异。我们通过测量分布外设置和迁移学习设置下源-目标相似性得分与下游预测误差之间的相关性，来评估这一差距。我们提出了域集成相似性代价（DISC），这是一个可学习的聚合框架，利用校准迁移观测来为不同的下游任务选择、加权和组合相似性度量。通过集成……

    arXiv:2610.31288v1 Announce Type: new  Abstract: The growth of time-series datasets and foundation models has made dataset comparison a central part of time-series workflows. Assessing source-target transfer before model adaptation requires similarity estimates that can summarize transfer-related dataset relationships without exhaustive source-target evaluation over every candidate pair. Many metrics have been proposed for measuring similarity between time-series datasets, but they capture different dataset relationships and vary in how well they align with downstream utility across tasks. We evaluate this gap by measuring the correlation between source-target similarity scores and downstream forecasting errors in out-of-distribution and transfer-learning settings. We propose Domain-Integrated Similarity Cost (DISC), a learnable aggregation framework that uses calibration transfer observations to select, weight, and combine similarity metrics for different downstream tasks. By integrat
    
[^60]: 超越准确性：在多智能体LLM流水线中解耦任务分解与类型化状态质量保障

    Beyond Accuracy: Disentangling Decomposition from Typed-State Quality Assurance in Multi-Agent LLM Pipelines

    [https://arxiv.org/abs/2610.31280](https://arxiv.org/abs/2610.31280)

    该研究发现多智能体LLM流水线在需求分类与优先级排序任务中的性能提升主要源于任务分解所支持的专门化提示，而非类型化状态质量保障机制。

    

    多智能体LLM流水线常被认为能在软件工程任务中带来性能提升，但这些提升究竟源于智能体之间的类型化契约，还是源于任务分解所支持的专门化提示，往往并不清楚。我们在需求工程领域研究了这一归因问题，比较了三种架构，用于将需求分类为功能性或非功能性需求，并随后用MoSCoW方法对其进行优先级排序：单次调用基线；两次调用变体（复用流水线的专门化提示但传递未经验证的中间结果）；以及采用Pydantic验证的类型化状态的流水线。实验覆盖18种条件，包括PROMISE NFR+数据集（625条需求）、三个本地托管的小型语言模型和两种语言。结果表明，相对于基线的F1提升与两次调用的专门化提示变体相关（最高提升+0.075），而类型化状态仅带来微小差异（-0.003到+0.013），且仅在六组比较中的一组中显著。

    arXiv:2610.31280v1 Announce Type: cross  Abstract: Multi-agent LLM pipelines are often credited with gains in software engineering tasks, but it is rarely clear whether a gain stems from the typed contract between agents or from the specialized prompts that decomposition enables. We study this attribution problem in Requirements Engineering, comparing three architectures for classifying requirements as functional or non-functional and then prioritizing them with MoSCoW: a single-call baseline; a two-call variant that reuses the pipeline's specialized prompts but passes an unvalidated intermediate; and a pipeline with a Pydantic-validated typed state. The 18 conditions cover PROMISE NFR+ (625 requirements), three locally hosted small language models, and two languages. The F1 gain over the baseline is associated with the two-call, specialized-prompt variant (up to +0.075), whereas the typed state adds a small difference (-0.003 to +0.013) that is significant in one of six comparisons. B
    
[^61]: 从非机器人合成数据中学习机器人行为先验

    Learning Robot Behavior Priors from Non-Robotic Synthetic Data

    [https://arxiv.org/abs/2610.31275](https://arxiv.org/abs/2610.31275)

    提出三种基于机器人控制数学抽象的合成数据生成管线，证明仅使用廉价的非机器人程序化合成数据进行预训练，就能让 Transformer 模型在机器人动作预测任务上获得非平凡的零样本性能。

    

    真实机器人数据虽然覆盖丰富的任务和形态，但常被视为训练机器人学习模型（包括旨在实现零样本物理迁移的具身基础模型）的瓶颈。本文研究了这类模型在多大程度上能够利用无具身系统的数据来获得必要的结构。为此，我们测试了廉价的、程序化生成的合成数据能否在模型观察到任何真实世界行为之前产生有用的先验。我们通过机器人动作预测来研究这一问题——这既是一个自然的预训练目标，也是对从任意合成任务中学到的结构能否迁移到具身环境的直接检验。我们提出了三种数据生成管线，它们对机器人控制的数学抽象进行建模，并将其作为 Transformer 的预训练先验。我们证明，仅在这些非机器人数据上训练的模型便能获得非平凡的零样本预测性能（原文摘要在此处截断）。

    arXiv:2610.31275v1 Announce Type: cross  Abstract: Real robot data, rich across tasks and morphologies, is often treated as the bottleneck for training robot learning models, including embodied foundation models that aim for zero-shot physical transfer. In this paper, we investigate the degree to which such models can acquire necessary structure using data from systems without embodiment. To this end, we test if inexpensive, procedurally generated synthetic data can yield useful priors before a model observes any real-world behavior. We study this through robot action prediction, a natural pretraining objective and a direct test of whether structure learned from arbitrary synthetic tasks transfers to embodied settings. We introduce three data generation pipelines that model mathematical abstractions of robotic control, and use them as pretraining priors for a Transformer. We demonstrate that a model trained purely on this non-robotic data attains non-trivial zero-shot prediction perfor
    
[^62]: 深度ReLU网络对连续目标函数可能不满足最佳逼近性质

    Deep ReLU Networks Can Fail the Best Approximation Property for Continuous Target Functions

    [https://arxiv.org/abs/2610.31264](https://arxiv.org/abs/2610.31264)

    本文通过构造连续函数反例证明，深度ReLU网络在一般情形下不满足最佳逼近性质，这与浅层网络中ReLU能保证全局极小值存在的已知结论形成鲜明对比。

    

    我们在 $[-1,1]^2$ 上构造了一个连续函数，使得在具有两个隐藏层且架构为 $(2,2,1,1)$ 的 ReLU 网络类上，$L^2$ 逼近误差的下确界无法被达到。这一反例表明，深度 ReLU 网络一般不满足最佳逼近性质，即使对于连续目标函数也是如此。我们的发现与浅层神经网络的已有结果形成对比——在浅层网络中，ReLU 是少数能在典型回归任务中保证全局极小值存在的激活函数之一。证明基于这样一个事实：第二隐藏层中神经元的折线（breaklines）可以发生弯曲，这破坏了单隐藏层情形中所使用的对称性论证。

    arXiv:2610.31264v1 Announce Type: new  Abstract: We construct a continuous function on $[-1,1]^2$ for which the infimum of the $L^2$-approximation error over the class of ReLU networks with two hidden layers and architecture $(2,2,1,1)$ is not attained. This counterexample shows that deep ReLU networks do not satisfy the best approximation property in general, even for continuous target functions. Our findings contrast with existing results for shallow neural networks, where ReLU is one of the few activation functions that guarantee the existence of global minima in typical regression tasks. The proof is based on the fact that the breaklines for neurons in the second hidden layer can bend, which breaks the symmetry arguments used in the one-hidden-layer case.
    
[^63]: 纠缠诱导的特征映射设计能否改进投影量子核？

    Can Entanglement-Induced Feature-Map Designs Improve Projected Quantum Kernels?

    [https://arxiv.org/abs/2610.31260](https://arxiv.org/abs/2610.31260)

    本文提出利用双量子比特纠缠的新型量子特征映射来构建投影量子核以捕获物联网数据的多变量相关性，并通过未调优与调优PQK对比RBF基线来分离评估其固有归纳偏置与结构表征能力。

    

    投影量子核（PQKs）已成为一种规避由量子希尔伯特空间指数级维度所带来的机器学习任务困难的重要方法。然而，近期的PQK研究往往严重依赖于对径向基函数（RBF）核参数和支持向量机正则化参数进行大量的经典超参数调优。这种大规模调优可能会掩盖由量子特征映射所诱导的底层PQK的真实表征能力，而这一点尚未得到详细探究。因此，本工作旨在通过将未调优和已调优的PQK与RBF基线进行对比，分离并评估其固有的归纳偏置，从而揭示PQK的结构性能力。此外，我们提出了新颖的量子特征映射，利用双量子比特纠缠来捕获物联网（IoT）数据中的多变量相关性。结果表明，大规模调优可能导致竞争……

    arXiv:2610.31260v1 Announce Type: cross  Abstract: Projected quantum kernels (PQKs) have emerged as a notable approach to circumvent the hardness of machine learning tasks arising from the exponential dimensionality of the quantum Hilbert space. However, recent PQK studies often rely heavily on extensive classical hyperparameter tuning of the radial basis function (RBF) kernel parameter and the support vector machine regularization parameter. The extensive tuning can mask the true representational power of the underlying PQKs induced by quantum feature maps, which has not been explored in detail. Thus, this work aims to expose the structural capabilities of PQKs by isolating and evaluating their native inductive bias through comparisons between untuned and tuned PQKs against the RBF baseline. Furthermore, we propose novel quantum feature maps that leverage two-qubit entanglement to capture multivariate correlations in IoT data. The results show that extensive tuning can lead to competi
    
[^64]: QuaNAS：面向查询高效神经架构搜索的、具有不确定性感知的早期搜索复用

    QuaNAS: Uncertainty-Aware Reuse of Earlier Searches for Query-Efficient Neural Architecture Search

    [https://arxiv.org/abs/2610.31251](https://arxiv.org/abs/2610.31251)

    QuaNAS将以往搜索的“设计决策主效应”和零成本代理指标转化为候选架构的不确定性先验，并利用高斯过程在每次评估后动态校准对各先验的信任程度，从而在完全不使用新数据集标签的情况下，将查询高效神经架构搜索的任意时刻遗憾降低2.7至3.9倍。

    

    在全新数据集上进行神经架构搜索（NAS）时，往往只有几十次短时训练的机会，因此搜索在首次评估之前所掌握的知识决定了其效率。然而，大多数NAS方法都是从零开始，充其量只依靠零成本代理指标来引导，而这些指标在不同任务间的可靠性参差不齐。QuaNAS将同一搜索空间中以往的搜索库视为可靠性不确定的显式知识。从该搜索库中学习到的设计决策的主效应，以及（次要的）零成本代理指标，在完全不使用新数据集任何标签的情况下，构成候选架构上的先验分布。高斯过程从先验最优解出发，并在每次评估后重新估计对各先验的信任程度。在一个在任何确认数据出现之前就已冻结的评估协议下，QuaNAS在12个全新的图像、时间序列、文本和表格数据集上各评估了一次。相比五个预先指定的NAS基线，它将短时训练目标下的任意时刻遗憾降低了2.7至3.9倍。

    arXiv:2610.31251v1 Announce Type: new  Abstract: Neural architecture search (NAS) on a new dataset often affords only tens of short training runs, so what the search knows before its first evaluation decides its efficiency. Yet most NAS methods start from scratch, guided at best by zero-cost proxies that are unreliable across tasks. QuaNAS treats a library of earlier searches in the same search space as explicit knowledge of uncertain reliability. Main effects of design decisions learned from this library and, secondarily, zero-cost proxies become priors over candidate architectures without any label of the new dataset. A Gaussian process starts at the prior optimum and re-estimates after every evaluation how far each prior is trusted. Under a protocol frozen before any confirmation data were seen, QuaNAS was evaluated once on 12 new image, time-series, text and tabular datasets. It lowered anytime regret on the short-training objective 2.7- to 3.9-fold relative to five pre-specified N
    
[^65]: 静音令牌：探索用于高效音频Transformer的令牌稀疏化

    Muting Tokens: Exploring Token Sparsification for Efficient Audio Transformers

    [https://arxiv.org/abs/2610.31235](https://arxiv.org/abs/2610.31235)

    本文对音频Transformer中的令牌稀疏化技术进行了全面评估，并提出了一种将注意力分数与频谱线索相结合的新型稀疏化方法，在多种数据集和模型规模下实现效率提升的同时保持性能。

    

    基于Transformer的模型在多种任务中展现出强大的性能，但其对自注意力机制的依赖往往导致高昂的计算成本。令牌稀疏化通过减少Transformer层中所处理的令牌数量，提供了一种在保持性能的同时提升效率的有前景的解决方案。然而，尽管令牌稀疏化在视觉和语言领域已被广泛研究，它在音频领域仍探索不足，现有工作通常仅局限于单一方法、模型规模或数据集。本工作对音频Transformer的令牌稀疏化技术进行了全面评估。我们在多种数据集和模型规模上，于训练后和微调两种设置下，评估了一系列令牌剪枝与合并策略。此外，我们提出了一种新颖的稀疏化方法，该方法将注意力分数与从令牌嵌入的频率内容中提取的频谱线索相结合。

    arXiv:2610.31235v1 Announce Type: cross  Abstract: Transformer-based models have shown strong performance across diverse tasks, but their reliance on self-attention mechanisms often leads to high computational costs. Token sparsification, which reduces the number of tokens processed in transformer layers, offers a promising solution to maintain performance while improving efficiency. However, although widely studied in the vision and language domains, token sparsification remains underexplored in audio, with existing work often limited to a single method, model size, or dataset. This work presents a comprehensive evaluation of token sparsification techniques for audio transformers. We assess a range of token pruning and merging strategies across multiple datasets and model sizes, under both post-training and fine-tuning settings. Furthermore, we introduce a novel sparsification approach that combines attention scores with spectral cues derived from the frequency content of token embedd
    
[^66]: 基于局部化AUC泛函的二元分类器局部判别

    Local Discrimination of Binary Classifiers via Localized AUC Functional

    [https://arxiv.org/abs/2610.31216](https://arxiv.org/abs/2610.31216)

    本文提出局部化AUC（LAUC）指标及LOCATOR方法，能够识别风险预测模型表现不佳的患者队列，从而克服传统AUC评估忽略性能异质性、可能导致临床决策失误的局限。

    

    风险预测模型通常使用AUC进行评估，这意味着在评估过程中我们对模型性能的异质性视而不见，这可能导致临床上的不良决策。在这项工作中，我们旨在通过识别风险预测模型可能无法准确预测的患者队列来改变这一现状。为了区分这些队列，我们使用了一种加权AUC的变体，我们称之为局部化AUC（Localized AUC，LAUC）。基于这一指标，我们构建了一种命名为LOCATOR的方法。该方法能够识别出风险预测模型不准确的患者队列。我们在真实世界数据和模拟实验中对该方法进行了测试。此外，我们还为局部化AUC在该应用中的使用建立了统计学理论基础。

    arXiv:2610.31216v1 Announce Type: new  Abstract: Risk prediction models are frequently evaluated with AUC, which means that during evaluation we are blind to performance heterogeneity, which could lead to bad decision-making in clinics. In this work, we aim to change that by identifying patient cohorts for whom the risk prediction model might not be an accurate predictor. To separate these cohorts, we use a version of the weighted AUC, which we call Localized AUC or LAUC. Using this metric, we build a method that we coin LOCATOR. This method is able to identify cohorts of patients for whom the risk prediction model is inaccurate. We test this method both in real-world data and in simulations. We also build a statistical background for the use of Localized AUC for our purposes.
    
[^67]: 基于混洗器匿名化的演化联邦密度峰值聚类

    Evolving Federated Density Peak Clustering with Shuffler-Based Anonymization

    [https://arxiv.org/abs/2610.31211](https://arxiv.org/abs/2610.31211)

    该论文提出了eFDPC方法，将演化模糊数据流聚类与密度峰值聚类相结合，并借助混洗器匿名化技术，在仅一轮通信的单次联邦学习范式中实现聚类，同时保证客户端数据的不可链接性与隐私安全。

    

    本文提出了一种基于演化模糊聚类与混洗器匿名化的新型单次联邦聚类方法。在联邦聚类中，每一轮通信都会向服务器泄露额外的模型信息，即使原始样本从未被共享，也可能增加推断本地数据集属性的机会。因此，减少客户端与服务器之间的交换次数以及实现不可链接性至关重要，即传输的信息不能被归因于任何单个客户端。数据流聚类算法通过对数据进行单次遍历即可完成学习，这与单次联邦学习范式高度契合——在该范式下，仅需一轮通信即可完成任务。所提出的演化联邦密度峰值聚类方法在联邦环境中融合了演化模糊数据流聚类与密度峰值聚类。每个客户端通过聚类添加机制，从私有的数据小批量样本中独立演化微簇。

    arXiv:2610.31211v1 Announce Type: new  Abstract: In this paper, we present a novel one-shot federated clustering approach based on evolving fuzzy clustering and shuffler-based anonymization. In federated clustering, each round reveals additional model information to the server and can increase the opportunity to infer properties of local datasets, even when raw samples are never shared. Consequently, fewer client-server exchanges and unlinkability are important, so transmitted information cannot be attributed to any single client. Data-stream clustering algorithms, which learn in a single pass over the data, align well with the one-shot federated learning paradigm, where a single communication round can be sufficient. The proposed Evolving Federated Density Peak Clustering (eFDPC) method combines evolving fuzzy data-stream clustering and density peak clustering in a federated setting. Each client independently evolves micro-clusters from private data mini-batches through cluster additi
    
[^68]: Inside-JEPA：潜在世界模型中从预测走向关系接地

    Inside-JEPA: From Prediction to Relational Grounding in Latent World Models

    [https://arxiv.org/abs/2610.31194](https://arxiv.org/abs/2610.31194)

    Inside-JEPA将有序实体对的关系通道嵌入JEPA潜在状态内部，并利用逆动力学损失在无需关系标签的情况下实现关系接地，在多个数据集上显著提升了关系预测性能。

    

    联合嵌入预测架构（JEPAs）在潜在空间中学习世界模型，能够实现高效规划，但其状态却是不透明的。我们提出Inside-JEPA，它将有序实体对的关系通道置于JEPA潜在状态内部，并证明当逆动力学损失的头读取这些通道时——无论是单独读取还是作为完整状态的一部分读取——无需关系标签即可使这些通道实现接地。在Something-Something V2数据集上，仅靠潜在预测的效果并不比随机关系头更好，而通过从通道变化中恢复动作的逆动力学监督，在十个带标注框的设置中将关系AUROC提升了0.065-0.103，在RH20T机器人片段（含留出任务）上提升了0.081，在使用现成检测与跟踪方法的五类关系上提升了0.050-0.061。基于实体潜在表示的逆动力学头虽然能更准确地解码动作，但几乎不能改善关系接地；打乱标签、两种无标签（

    arXiv:2610.31194v1 Announce Type: new  Abstract: Joint-embedding predictive architectures (JEPAs) learn world models in latent space, enabling efficient planning but leaving their states opaque. We introduce Inside-JEPA, which places relational channels for ordered entity pairs inside the JEPA latent state and shows that an inverse-dynamics loss can ground these channels without relation labels when its head reads them, either alone or as part of the full state. On Something-Something V2, latent prediction alone performs no better than a random relation head, while inverse-dynamics supervision that recovers actions from channel changes improves relational AUROC by .065-.103 across ten settings with annotated boxes, by .081 on RH20T robot episodes with held-out tasks, and by .050-.061 across five relations using off-the-shelf detection and tracking. An inverse head on entity latents decodes actions more accurately but barely improves relational grounding; shuffled labels, two label-free
    
[^69]: 当更多元数据反而误导时：探查大语言模型生成的博物馆描述中的字段级接地性

    When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions

    [https://arxiv.org/abs/2610.31193](https://arxiv.org/abs/2610.31193)

    该研究通过两阶段实验发现，元数据数量本身并不能提升大语言模型生成博物馆文物描述的可靠性，真正决定模型是接地、省略还是凭空推断信息的是字段类型，而图像既能支持视觉接地也可能助长无依据的解读。

    

    元数据条件生成通常被视为一种数据到文本的任务，即更多的输入字段应当能改善生成内容的接地性（grounding）。然而博物馆描述生成对这一假设提出了挑战，因为馆藏目录记录具有异构性、部分依赖视觉信息，且在不同地理区域间的记录完备程度参差不齐。我们通过一个两阶段框架研究元数据的构成如何影响文物描述的生成。第一阶段分析了一个大型博物馆藏品中元数据的自然变化，其中字段数量、字段类型、图像可用性和地点等因素共同变化。第二阶段引入了受控的字段探针，在保持文物身份不变的前提下，改变元数据数量、字段类型和图像的访问情况。我们的结果表明，元数据数量本身并不能解释生成行为；字段类型才是决定模型对接地、省略还是推断信息的关键因素，而图像既能够支持视觉接地，也可能增加缺乏依据的解读。

    arXiv:2610.31193v1 Announce Type: new  Abstract: Metadata-conditioned generation is often framed as a data-to-text task in which additional input fields should improve grounding. Museum description generation challenges this assumption because catalogue records are heterogeneous, partially visual, and unevenly documented across geographic regions. We study how metadata composition shapes the generation of artefact descriptions through a two-phase framework. Phase 1 analyses natural metadata variation in a large museum collection, where field count, field type, image availability, and Place co-vary. Phase 2 introduces controlled field probes that hold object identity fixed while varying metadata quantity, field type, and image access. Our results show that metadata quantity alone does not explain generation behaviour. Field type determines whether models ground, omit, or infer information, while images can both support visual grounding and increase unsupported interpretation. More broad
    
[^70]: RaSFNet：面向激光雷达语义分割的光线对齐与扫描感知融合网络

    RaSFNet: Ray-aligned and Scan-aware Fusion Network for LiDAR Semantic Segmentation

    [https://arxiv.org/abs/2610.31186](https://arxiv.org/abs/2610.31186)

    该论文提出RaSFNet，将点云几何信息与激光雷达扫描特性融入基于投影的语义分割，通过光线对齐几何分布编码与扫描感知机制，解决了投影聚合掩盖视锥内几何差异、跨扫描区域空间关系建模受限以及远近特征混淆降低语义判别能力的问题。

    

    激光雷达点云语义分割是自动驾驶和机器人3D感知的基础组成部分。基于投影的方法将3D点云映射为2D表示，以利用2D网络进行特征提取。然而，投影过程中对多个点的聚合可能掩盖每个视锥体内的几何差异，而局部特征提取则限制了跨扫描区域空间关系的建模能力。此外，投影表示中的邻近关系并不一定意味着3D空间中的邻近关系，点特征与投影特征之间的交互可能混合来自不同距离的特征，从而降低语义判别能力。为了应对这些挑战，我们提出了光线对齐与扫描感知融合网络，它将点云几何信息和激光雷达扫描特性融入到基于投影的语义分割中。具体而言，光线对齐几何分布编码器……（摘要在此处被截断）

    arXiv:2610.31186v1 Announce Type: cross  Abstract: LiDAR point cloud semantic segmentation is a fundamental component of 3D perception for autonomous driving and robotics. Projection-based methods map 3D point clouds into 2D representations for feature extraction using 2D networks. However, aggregating multiple points during projection can obscure geometric differences within each frustum, while local feature extraction limits the modeling of spatial relationships across scanning regions. Moreover, proximity in the projected representation does not necessarily imply proximity in 3D space, and interactions between point and projected features may mix features from different ranges, reducing semantic discriminability. To address these challenges, we propose the Ray-aligned and Scan-aware Fusion Network (RaSFNet), which incorporates point cloud geometry and LiDAR scanning characteristics into projection-based semantic segmentation. Specifically, the Ray-Aligned Geometric Distribution Enco
    
[^71]: 基于粒子物理基础模型的径迹寻找与粒子鉴别性能研究

    Track finding and particle identification performance with particle physics foundation models

    [https://arxiv.org/abs/2610.31180](https://arxiv.org/abs/2610.31180)

    该论文在时间投影室数据集上评估了粒子物理基础模型在径迹寻找和粒子鉴别等下游任务中的表现，证明基于基础模型的数据分析方法性能显著优于传统方法，且可灵活适应新任务。

    

    现代高能核物理与粒子物理实验会产生大量稀疏数据，在进行物理分析之前必须对其进行大幅度的规模缩减。近年来，研究者们一直在尝试将基础模型评估为执行数据分析的潜在工具。已有研究表明，这些模型能够基于其所训练的多模态数据来学习高维数据集中的特征。为了展示基础模型在执行复杂模式识别任务（例如在磁场中重建带电粒子）方面的有效性，我们在一个由时间投影室空间点组成的数据集上，评估了基于基础模型的重建方法在多个下游任务中的性能。结果表明，基于基础模型的数据分析性能显著优于传统方法，此外该方法还可以被调整用于学习与原任务不一定相关的额外任务。

    arXiv:2610.31180v1 Announce Type: cross  Abstract: Modern high energy nuclear and particle physics experiments produce large amounts of sparse data that must be reduced significantly in size before performing physics analysis. Recently, there has been an effort to evaluate Foundation Models as a potential tool for performing data analysis. These models have been shown to learn features in high dimensional datasets based on the multi-modal data they are trained on. To show their effectiveness at performing complex pattern recognition tasks, such as reconstructing charged particles in a magnetic field, we evaluated the performance of a Foundation Model based reconstruction on several downstream tasks with a dataset composed of spacepoints in a Time Projection Chamber. We show that the performance of the Foundation Model based data analysis significantly improves upon that of traditional methods, and furthermore can be adapted to learn additional tasks that are not necessarily related to 
    
[^72]: 提示策略效果取决于评判模型：LLM作为评判者鲁棒性的多语言研究

    Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness

    [https://arxiv.org/abs/2610.31172](https://arxiv.org/abs/2610.31172)

    该研究通过涵盖9种提示策略、4个开源评判模型、27种语言的大规模基准测试（超过100万条评分），首次发现LLM-as-a-Judge评估中的提示策略效果不可跨模型和语言迁移——评判模型的选择比提示策略的选择更具决定性，且策略效果在不同评判模型间会反向。

    

    LLM-as-a-Judge（大语言模型作为评判者）评估正日益应用于多语言任务，然而诸如思维链和角色条件化等提示策略的推荐仅基于英语基准。这些策略的有效性能否跨评判模型和语言迁移仍是未知数：提示敏感性研究大多以英语为中心，而现有的多语言评判研究则固定了提示模板。在本工作中，我们通过一个基准测试来衡量这种交互作用，该基准涵盖9种提示策略、4个开源权重评判模型、27种语言以及跨越13个数据集的4个任务，在黄金参考输出上产生了超过100万个逐点评分，并经过人工判断的验证。我们发现提示策略是不可迁移的：评判模型的选择主导了提示策略的选择，策略效果在不同评判模型之间会发生方向反转，且在低资源语言中稳定性下降，而这些影响被掩盖……

    arXiv:2610.31172v1 Announce Type: new  Abstract: LLM-as-a-Judge evaluation is increasingly applied to multilingual tasks, yet prompting strategies such as Chain-of-Thought and Role Conditioning are recommended based solely on English benchmarks. Whether their effectiveness transfers across judge models and languages remains unknown: prompt-sensitivity studies are largely English-centric, while existing multilingual judge studies fix the prompt template. In this work, we address this gap by measuring this interaction through a benchmark of 9 prompting strategies, 4 open-weight judges, 27 languages, and 4 tasks across 13 datasets, yielding over 1M pointwise scores on gold-reference outputs, validated against human judgments. We find that prompting strategies are not transferable: the choice of the judge model dominates the choice of prompting strategy, strategy effects reverse direction across judges, and stability degrades in low-resource languages, where these effects are concealed by 
    
[^73]: 目标之前的记忆：预历史记忆在人形机器人移动操作中的作用

    Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation

    [https://arxiv.org/abs/2610.31166](https://arxiv.org/abs/2610.31166)

    提出物体引导的感知记忆机制WholeBodyMem，利用SAM3跟踪的物体事件因果选择关键帧，使人形机器人全身VLA策略能够记住行动前的物体历史，从而在非马尔可夫动态场景中大幅提升成功率（仿真中29.1%提升至72.6%，真实机器人上36.7%提升至88.3%），并发布了HuMeGo人形机器人记忆基准。

    

    动态场景中的人形机器人移动操作是非马尔可夫的：当物体发生变化或被遮挡时线索会消失，或者当机器人转身行走时物体会离开视野，而全身视觉-语言-动作（VLA）策略却只能看到当前时刻。我们提出了WholeBodyMem，一种物体引导的感知记忆，它保存了机器人在执行目标之前所看到的物体事件的历史。SAM3负责跟踪任务物体，其事件以因果方式在自适应预算内选择关键帧，动作专家通过交叉注意力机制将这些关键帧与编码外观、几何和时间的物体标记一起读取。我们还贡献了HuMeGo，一个包含六个仿真任务和六个真实世界Unitree G1任务的人形机器人记忆基准。通过保留正确的帧而非更多的帧，WholeBodyMem超越了无记忆和均匀记忆的基线方法，将其基础策略的成功率在仿真中从29.1%提升至72.6%，在真实机器人上从36.7%提升至88.3%，同时还改善了……

    arXiv:2610.31166v1 Announce Type: cross  Abstract: Humanoid loco-manipulation in dynamic scenes is non-Markovian: cues vanish as objects change or are hidden, or leave the view as the robot turns and walks, yet whole-body vision-language-action (VLA) policies see only the present. We present WholeBodyMem, an object-guided perceptual memory that keeps the pre-history of object events seen before the robot acts on its goal. SAM3 tracks the task objects, and their events causally select key frames within an adaptive budget, which the action expert reads by cross-attention together with object tokens encoding appearance, geometry and time. We also contribute HuMeGo, a humanoid memory benchmark with six simulated and six real-world Unitree G1 tasks. Keeping the right frames rather than more frames, WholeBodyMem surpasses memoryless and uniform-memory baselines, raising its base policy's success from 29.1% to 72.6% in simulation and from 36.7% to 88.3% on the real robot, while also improving
    
[^74]: 预测不确定性可控的计算机模型校准

    Prediction Uncertainty Controlled Computer Model Calibration

    [https://arxiv.org/abs/2610.31163](https://arxiv.org/abs/2610.31163)

    该论文提出一个两层框架，将条件流匹配摊销后验推断与共形风险控制相结合，直接控制校准后模拟器的预测覆盖率，即使在校准参数弱识别或模型差异被误设的情况下也能保证名义的预测区间水平。

    

    计算机模型校准通常是为预测服务的，然而无论是贝叶斯Kennedy-O'Hagan (KOH)校准还是摊销式的基于模拟的推断，其设计目的都不是控制校准后模拟器的预测误差。当参数识别性较弱或数据-模型差异被错误设定时，由名义后验水平导出的模拟器输出区间可能远偏离名义覆盖水平。我们提出了一个直接针对预测覆盖不足（miscoverage）的两层框架。第一层中，一个条件流匹配模型学习校准参数的摊销式后验，其训练数据是由不同结构的高斯过程差异扰动后的模拟器输出。第二层中，共形风险控制将针对目标观测的后验样本组织成嵌套的候选集合，并选出其中最小的集合，使其固定秩的逐点预测区间在合成校准样本上达到规定的时间平均覆盖不足率。

    arXiv:2610.31163v1 Announce Type: cross  Abstract: Computer-model calibration is usually undertaken in service of prediction, yet neither Bayesian Kennedy-O'Hagan (KOH) calibration nor amortized simulation-based inference is built to control the predictive error of the calibrated simulator. When parameters are weakly identified or the data-model discrepancy is misspecified, the simulator-output bands induced by a nominal posterior level can be far from nominal. We propose a two-layer framework that targets predictive miscoverage directly. First, a conditional flow-matching model learns an amortized posterior for the calibration parameters, trained on simulator outputs perturbed with Gaussian-process discrepancies of varying structure. Second, conformal risk control arranges posterior draws for the target observation into nested candidate sets and selects the smallest one whose fixed-rank pointwise bands attain a prescribed time-averaged miscoverage on a synthetic calibration sample. Un
    
[^75]: 迈向可现场部署的可穿戴IMU系统：面向职业活动识别

    Towards Field-Deployable Wearable IMU Systems for Occupational Activity Recognition

    [https://arxiv.org/abs/2610.31160](https://arxiv.org/abs/2610.31160)

    该研究首次对VTT-ConIoT数据集进行系统的配置敏感性分析，发现即使在实验室数据上，受试者独立（LOSO）评估下可穿戴IMU活动识别的个体准确率差异高达27个百分点，揭示了以往总体高准确率报告掩盖了实际现场部署面临的性能挑战。

    

    用于建筑工人人体活动识别（HAR）的可穿戴惯性测量单元（IMU）系统通常报告的总体准确率超过95%，但很少在真正的受试者独立（留一受试者交叉验证（LOSO））评估下进行测试，这掩盖了决定实际部署性能的个体间差异。利用VTT-ConIoT基准数据集，据我们所知，我们首次对该数据集进行了系统的配置敏感性分析：在统一的LOSO协议下，评估了六个分类器在五种传感器佩戴位置和三种窗口大小上的表现，并进行了个性化校准试点实验，以及对表格GAN增强方法与标准时间序列增强方法的验证性比较。在主要的多传感器16类配置中，各受试者的LOSO准确率范围为61.9%至88.7%（平均77.5%），个体差异达27个百分点，尽管所有数据均是在实验室环境中采集的。多传感器融合和更长的窗口均能提高准确率；标准增强方法给出的……（原文摘要不完整）

    arXiv:2610.31160v1 Announce Type: new  Abstract: Wearable inertial measurement unit (IMU) systems for construction worker human activity recognition (HAR) routinely report aggregate accuracies above 95 %, rarely under genuine subject-independent (leave-one-subject-out (LOSO)) evaluation, masking the per-individual variance that determines real-world deployment performance. Using the VTT-ConIoT benchmark, we conduct, to our knowledge, the first systematic configuration sensitivity analysis of this dataset: six classifiers across five sensor placements and three window sizes under a single LOSO protocol, a personalised-calibration pilot, and a validated comparison of tabular GAN-based and standard time-series augmentation. Per-subject LOSO accuracy for the main multi-sensor, 16-class configuration ranges from 61.9 % to 88.7 % (mean 77.5 %), a 27 pp spread, though all data are laboratory-collected. Multi-sensor fusion and longer windows both improve accuracy; standard augmentation gives s
    
[^76]: 工具增强模型比较中的提示敏感性

    Prompt Sensitivity in Tool-Augmented Model Comparisons

    [https://arxiv.org/abs/2610.31148](https://arxiv.org/abs/2610.31148)

    论文提出一种将答案读取、推理、调用决策等环节分离的匹配协议，发现工具虽能提升模型准确率，却会加剧模型间比较对问题表述方式的敏感性。

    

    语言模型通常在问答基准上进行比较，其中每个问题只以一种表述方式呈现，而对问题进行改写就可能改变两个模型中谁得分更高。我们探究为模型提供工具是否能使这类比较更加稳定。仅凭准确率的提升无法回答这个问题，因为工具会同时改变从问题到被评分答案之间的多个环节。我们提出一种匹配协议来分离这些环节：答案读取、推理、调用决策、工具返回的答案，以及在不调用工具时提供的报告。我们将该协议应用于六个开源模型，在五个领域、五种表述方式下进行测试。我们发现，在程序能够回答问题的场景中，工具可以提升准确率，但同时也会加大许多成对模型比较对表述方式的敏感性。这种敏感性的扩大与模型调用工具的频率相关，而某些模型的调用频率会随表述方式变化，另一些则不会。敏感性还取决于接口与指令：原生工具调用接口以及……

    arXiv:2610.31148v1 Announce Type: new  Abstract: Language models are compared on question-answering benchmarks that pose each question in one wording, and a reworded question can change which of two models scores higher. We ask whether giving the models tools makes such comparisons more stable. An accuracy gain cannot answer this, because tools change several steps between the question and the scored answer at once. We propose a matched protocol that separates those steps: the answer readout, reasoning, the decision to call, the tool's answer, and the report supplied without a call. We apply it to six open models in five domains under five wordings. We find that tools raise accuracy where a program answers the question and widen the wording sensitivity of many pairwise comparisons. The widening follows how often a model calls, and call frequency follows the wording for some models and not others. It also depends on the interface and the instructions: a native tool-calling interface and
    
[^77]: PriMa-Causa：因果性处方式维护

    PriMa-Causa: Causal Prescriptive Maintenance

    [https://arxiv.org/abs/2610.31147](https://arxiv.org/abs/2610.31147)

    论文指出处方式维护中纠正措施的选择本质上需要干预性因果量而非关联性预测，观测数据训练的预测模型会误导维护决策，因此应将因果推理嵌入处方式维护架构。

    

    自动化制造需要在不确定性和稀疏观测记录下做出数据驱动的决策，而非计划停机是决策失误最主要的代价。处方式维护（prescriptive maintenance）针对即将发生的故障推荐维护措施，而不仅仅是预测故障。既有的参考架构将预测模型与基于规则或优化的推荐逻辑相耦合。然而，选择纠正措施需要一个干预量：即如果设定了该措施，产线行为将如何变化。由于制造数据是观测性的，拟合用于预测结果的模型所估计的是关联量，可能会误导措施的选择，而预测误差却无法揭示这一问题。因果估计器以干预量为目标，但需要对每个案例进行专家主导的因果研究，而运行时的有限决策窗口并不允许这样做。本文分析了如何将因果推……

    arXiv:2610.31147v1 Announce Type: new  Abstract: Automated manufacturing requires data-driven decisions under uncertainty and sparse observational records, and unplanned downtime is a dominant cost of getting them wrong. Prescriptive maintenance recommends a maintenance action against an impending failure rather than merely predicting it. Established reference architectures couple predictive models with rule- or optimisation-based recommendation logic. Selecting a corrective action, however, requires an interventional quantity: how the line behaviour would change if the action were set. Manufacturing data is observational, so a model fitted to predict the outcome estimates an associational quantity and may misdirect the choice of action without the prediction error revealing it. Causal estimators target the interventional quantity but require an expert-led causal study per case, for which the bounded decision window of operation leaves no room. This dissertation analyses how causal rea
    
[^78]: GM-GReFEL：面向鲁棒动态面部表情识别的几何引导时空状态空间学习

    GM-GReFEL: Geometry-Guided Spatiotemporal State-Space Learning for Robust Dynamic Facial Expression Recognition

    [https://arxiv.org/abs/2610.31140](https://arxiv.org/abs/2610.31140)

    提出GM-GReFEL模型，通过线性复杂度的时空状态空间骨干网络与基于面部几何锚点的可靠性决策模块，同时解决了动态面部表情识别中自注意力计算开销大和众包标签噪声两大难题。

    

    在无约束视频中进行面部表情识别面临两个相互耦合的问题：自注意力的计算成本随视频片段长度呈二次方增长，以及众包标签存在噪声。我们提出了GM-GReFEL，一种带有几何感知决策阶段的时空状态空间网络。其调制型GroupMamba骨干网络取代了时空视觉Transformer。六个通道组被分别扫描，其中四个在帧平面上扫描，两个沿时间维度扫描，因此计算成本随序列长度线性增长（O(N)）；在每层之后，通道亲和调制门控会重新混合这些通道组。在骨干网络之上，GReFEL可靠性模块将每个预测的归一化熵转换为分类器输出与可训练几何锚点投票之间的混合权重，使得当网络不确定时，面部几何信息能发挥更大作用。该模型通过静态到动态的权重迁移进行初始化，并采用三重损失进行训练。

    arXiv:2610.31140v1 Announce Type: cross  Abstract: Recognizing facial expressions in unconstrained video suffers from two coupled problems: self-attention grows quadratically in cost with clip length, and crowdsourced labels are noisy. We present GM-GReFEL, a spatiotemporal state-space network with a geometry-aware decision stage. Its Modulated GroupMamba backbone replaces a spatiotemporal Vision Transformer. Six channel groups are scanned separately, four over the frame plane and two over time, so the cost grows linearly with sequence length ($\mathcal{O}(N)$); after each layer, a Channel Affinity Modulation gate mixes the groups again. On top of the backbone, a GReFEL reliability module turns the normalized entropy of each prediction into a mixing weight between the classifier output and a vote from trainable geometric anchors, so that facial geometry counts for more when the network is unsure. The model is initialized by static-to-dynamic weight transfer and trained with a Triple Lo
    
[^79]: 混合很重要：长上下文问答中不同序列混合架构的证据位置偏差研究

    Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering

    [https://arxiv.org/abs/2610.31135](https://arxiv.org/abs/2610.31135)

    本研究通过在干扰文档中移动黄金证据，系统比较了Transformer、状态空间和混合Mamba架构在长上下文问答中的证据位置偏差，发现Transformer模型具有状态空间模型所缺乏的开头优势，而两类架构均偏好接近末尾的证据。

    

    长上下文模型在包含答案的证据出现在远离提示边界的位置时，往往无法有效利用这些证据。我们通过将一份黄金文档在固定的九个干扰文档集合中移动位置，比较了Transformer、状态空间以及混合Mamba架构之间的这种位置偏差。被评估的Transformer模型显示出开头优势，而被评估的状态空间模型缺乏这种优势，同时两个家族的模型都更偏好接近末尾的证据。一项匹配的纯Mamba与混合Mamba对比指向相同的方向，但在统计上仍不确定。这种家族差异体现在能够完成该任务的模型之中。Transformer的首要地位在合成检索任务中再次显现，而状态空间模型的性能天花板使得家族间的比较无法进行。在同一状态空间架构内更换语料库会改变总体准确率，但曲线形状没有可检测到的变化。提示的改变可以产生或几乎消除开头优势。位置信息在各架构中仍然是可解码的。

    arXiv:2610.31135v1 Announce Type: new  Abstract: Long-context models often use answer-bearing evidence less effectively when it appears away from a prompt boundary. We compare this position bias across Transformer, state-space, and hybrid Mamba architectures by moving one gold document through a fixed set of nine distractors. The evaluated Transformers show a beginning advantage that the evaluated state-space models lack, while both families favor evidence near the end. A matched pure-versus-hybrid Mamba comparison points in the same direction but remains statistically uncertain. The family difference appears among models that can answer the task. Transformer primacy reappears on synthetic retrieval, where state-space ceiling prevents a family comparison. A corpus swap within one state-space architecture changes overall accuracy without a detectable change in curve shape. Prompt changes can create or nearly remove a beginning advantage. Position remains decodable across architectures e
    
[^80]: 高本地精度就够了吗？面向联邦学习的几何共识与支持自适应统计个性化

    Is High Local Accuracy Enough? Geometric Consensus with Support-Adaptive Statistical Personalization for Federated Learning

    [https://arxiv.org/abs/2610.31133](https://arxiv.org/abs/2610.31133)

    提出FedGRC统一个性化联邦学习框架及配套评估协议，通过几何共识与支持自适应统计个性化，在提升最差客户端个性化下限的同时，增强模型对全局、稀有及未见类别的识别能力。

    

    个性化联邦学习（PFL）的现实部署要求在适应异构客户端的同时，保持超越本地分布的识别性能。然而，较高的本地精度可能掩盖模型在本地稀有或缺失类别上的失败。因此，有效的个性化应在提升个性化下限的同时增强全局识别能力。我们提出了一种评估协议，在本地精度之外补充全局、稀有类别和未见类别的评估，以揭示被本地分数所掩盖的泛化能力，并在表现最差的客户端中评估个性化下限，以暴露被聚合性能隐藏的弱点。为应对这些部署挑战，我们提出了FedGRC，这是一个统一的PFL框架，通过以下方式整合泛化与个性化：(i) 共识引导表示学习（CGRL），通过共享的方向与幅度指导学习判别性表示

    arXiv:2610.31133v1 Announce Type: new  Abstract: Real-world deployment of personalized federated learning (PFL) requires adapting to heterogeneous clients while preserving recognition performance beyond local distributions. However, high local accuracy can conceal failures on locally rare or absent classes. Effective personalization should therefore strengthen global recognition while raising the personalization floor. We propose an evaluation protocol that complements local accuracy with global, rare-class, and unseen-class evaluations to reveal generalization obscured by local scores, and assesses personalization floors among the worst-performing clients to expose weaknesses hidden by aggregate performance. To meet these deployment challenges, we propose FedGRC, a unified PFL framework integrating generalization and personalization through (i) Consensus-Guided Representation Learning (CGRL), which learns discriminative representations through shared directional and magnitude guidance
    
[^81]: 审计后幻象：离线智能体验证的可识别性极限

    Post-Audit Mirage: Identifiability Limits of Offline Agent Verification

    [https://arxiv.org/abs/2610.31130](https://arxiv.org/abs/2610.31130)

    该论文证明了离线智能体验证存在根本性的可识别性极限——安全与不安全的智能体生命周期在离线审计中可能完全无法区分——并通过可执行的基准测试与基于组标记危害结果的监控方法，刻画了监控在何种条件下能够或无法恢复部署后缺失的安全性区分。

    

    离线测试可能无法揭示智能体更新在部署后是否仍然安全。我们通过构建匹配的安全与有害生命周期来操作化这种“审计后幻象”——这些生命周期向验证者暴露完全相同的可见证据，而一个隔离运行的评分器却记录到截然相反的部署真实结果。我们的定理将标准的观察等价与不可区分性思想应用于这一可执行的审计边界，并给出了由此产生的错误-弃权权衡。Post-Audit Mirage 基准使这一边界在授权、调度和共享容量环境中变得可执行。IdentifiedRangeMonitor 利用新鲜的组标记危害结果来返回“部署”、“保留”、“无法确定”或“不支持”四种决策。实验表明，监控在有利设置下可以恢复缺失的区分，但对于稀有群体、小效应或缺失观测，可能仍然无法做出判定。因此，验证器-智能体评估必须同时测试计算有效性……（原文在此处截断）

    arXiv:2610.31130v1 Announce Type: cross  Abstract: Offline tests can fail to reveal whether an agent update will remain safe after deployment. We operationalize this post-audit mirage using matched safe and harmful lifecycles that expose identical verifier-visible evidence while an isolated scorer records opposite deployment truth. Our theorem applies standard observational-equivalence and indistinguishability ideas to this executable audit boundary and states the resulting error-abstention tradeoff. The Post-Audit Mirage benchmark makes that boundary executable across authorization, scheduling, and shared-capacity environments. IdentifiedRangeMonitor uses fresh group-labelled harm outcomes to return deploy, hold, cannot determine, or unsupported. Experiments show that monitoring can recover the missing distinction in favorable settings but may remain undecided for rare groups, small effects, or missing observations. Verifier-agent evaluation must therefore test both calculation validi
    
[^82]: Delta N-gram：向语言模型原生N-gram记忆的仅前向持续写入

    Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory

    [https://arxiv.org/abs/2610.31125](https://arxiv.org/abs/2610.31125)

    提出Delta N-gram方法，通过冻结预训练N-gram表、将证据以仅前向传播的方式写入稀疏补丁Δ，实现无需反向传播的持续知识写入，清除补丁即可完全恢复原始模型，且事实保留效果远优于基于梯度的写入方法。

    

    我们之前的分析发现，模型原生的哈希N-gram表是持续更新对基础模型干扰最小的位置，但其写入需要反向传播，并且会改变预训练表。我们提出Delta N-gram，一种进入该表的仅前向写入路径。预训练表T^0永远不会被修改，证据被写入一个稀疏补丁Δ中其所寻址的行上：一个4.1M参数的写入器为每个（位置，哈希头）事件生成一个目标值和一个门控值，块平均的delta规则使每一行始终保持为其旧值与门控目标之间的凸组合。该写入器使用预训练目标在原始文本上离线训练；在使用时，写入仅需前向传播，清除Δ即可完全恢复原始模型。一个经过训练用于回答问题的0.6B参数原型模型在第二次写入后仍保留了16/16的事实；而在线梯度写入只能保留2-4/16。我们研究了两种写入器：一种是用仅基于T^0的模型编码证据的普通写入器，以及一种……（原文摘要到此截断）

    arXiv:2610.31125v1 Announce Type: new  Abstract: Our previous analysis found that a model's native hashed N-gram table is where continual updates disturb the base least, but its writes needed backpropagation and changed the pretrained table. We present Delta N-gram, a forward-only write path into that table. The pretrained table T^0 is never modified. Evidence is written into a sparse patch \Delta on the rows it addresses: a 4.1M-parameter writer emits a target and a gate per (position, hash head) event, and a block-averaged delta rule keeps each row a convex combination of its old value and the gated targets. The writer is trained offline on raw text with the pretraining objective; at use time writing takes forward passes only, and clearing \Delta restores the original model. A 0.6B prototype trained to answer questions keeps 16/16 facts after a second write; online gradient writes keep 2--4/16. We study two writers: a plain writer that encodes evidence with the T^0-only model, and a 
    
[^83]: 接下来会发生什么以及为什么：基于神经符号规则的可解释下一事件预测

    What Comes Next and Why: Interpretable Next-Event Prediction with Neuro-Symbolic Rules

    [https://arxiv.org/abs/2610.31114](https://arxiv.org/abs/2610.31114)

    提出了神经符号模型RULES及其与残差网络结合的RUNES框架，通过从数据中直接学习可解释的序列规则，在下一事件预测中同时实现了有竞争力的准确率和透明的可解释性。

    

    在现实世界的应用中，能够解释预测为何被做出往往与实现高准确率同等重要——甚至更为重要。然而，现有的序列事件预测方法通常以牺牲可解释性为代价来优先追求准确性。神经符号方法通过学习符号规则为提升可解释性提供了一个有前景的方向，但由于时序依赖性的存在，将其扩展到下一事件预测具有挑战性。我们提出了RULES，一个直接从数据中学习序列规则的神经符号模型，并将其与残差网络相结合，形成RUNES（事件序列规则网络）。RULES能够从合成数据中准确恢复真实规则，并在真实数据上发现紧凑且可解释的规则集。RUNES将这些规则以加法方式与残差网络结合，在提供对激活规则及残差作用机制的透明访问的同时，实现了具有竞争力的准确率。

    arXiv:2610.31114v1 Announce Type: new  Abstract: In real-world applications, being able to explain why a prediction is made is often as important - if not more so - than achieving high accuracy. However, existing methods for sequential event prediction typically prioritize accuracy at the expense of interpretability. Neuro-symbolic approaches offer a promising direction to improve interpretability by learning symbolic rules, but extending them to next-event prediction is challenging due to the presence of temporal dependencies. We introduce RULES, a neuro-symbolic model that learns sequential rules directly from data, and combine it with a residual network to form RUNES (Rule Network for Event Sequences). RULES accurately recovers ground truth rules from synthetic data and discovers compact, interpretable rule sets on real data. RUNES combines these rules additively with a residual network, achieving competitive accuracy while providing transparent access to active rules and how residu
    
[^84]: DRAFT：从fMRI和相机轨迹解码并重建3D房间布局

    DRAFT: Decoding and Reconstructing 3D Room Layouts from fMRI and Camera Trajectories

    [https://arxiv.org/abs/2610.31108](https://arxiv.org/abs/2610.31108)

    该论文首次提出仅凭相机轨迹和被试的fMRI信号（无需任何视觉帧）重建未见场景的3D房间布局，通过从早期视觉皮层解码紧凑布局表示并与相机运动推导的几何自由空间融合来实现这一目标，同时提出了布局r这一新评估指标。

    

    尽管功能性磁共振成像解码器已经能够重建图像和3D物体，但从移动视角恢复房间的度量几何仍然是一个开放的挑战。据我们所知，我们首次提出了这样一个任务：仅使用已知的相机轨迹和观看室内漫游视频的被试的fMRI信号来重建未见过的场景的3D布局，而不使用任何视觉帧。这项任务非常困难，因为快速的视觉变化与模糊的fMRI信号之间存在时间上的失配。我们的方法DRAFT（从fMRI和轨迹解码并重建3D房间布局）从早期视觉皮层提取紧凑的布局表示，并将这种神经编码与从相机运动推导出的几何自由空间相融合。为了评估该方法，我们提出了布局r指标，即预测深度与光线投射对数深度之间的皮尔逊相关系数。在十个保留场景（20名被试）上，DRAFT达到了布局……

    arXiv:2610.31108v1 Announce Type: cross  Abstract: While functional magnetic resonance imaging (fMRI) decoders reconstruct images and 3D objects, recovering the metric geometry of rooms from a moving viewpoint remains an open challenge. We introduce, to our knowledge for the first time, the task of reconstructing 3D layouts of held-out scenes using only the known camera trajectories and the fMRI signals of participants watching indoor walkthroughs, without any visual frames. The task is hard because of the temporal mismatch between rapid visual changes and blurred fMRI signals. Our method, DRAFT (Decoding and Reconstructing 3D room lAyouts from fMRI and Trajectories), extracts a compact layout representation from the early visual cortex and fuses this neural code with the geometric free space derived from camera motion. To evaluate it, we propose layout r, the Pearson correlation between predicted and ray-cast log depths. On ten held-out scenes (20 participants), DRAFT reaches a layout
    
[^85]: 具有思维链的Transformer通过训练发现动量加速方法

    Transformers with CoT Discover Heavy-Ball Acceleration Through Training

    [https://arxiv.org/abs/2610.31089](https://arxiv.org/abs/2610.31089)

    本文证明Transformer在梯度下降教师的初始监督下，通过真实回归权重的轨迹级监督训练，能够自发学会实现Heavy-Ball动量加速方法，超越对教师算法的单纯模仿并获得加速效果。

    

    现代Transformer通常通过生成和改进中间推理步骤来解决复杂任务。为了对思维链推理建立理论理解，Huang等人（2025）近期的一项重要研究将Transformer视为算法执行器，并证明当Transformer的CoT步骤由梯度下降教师监督时，它们可以学习实现梯度下降，用于线性回归中的上下文权重预测。在本文中，我们研究相同的上下文权重预测任务，但探究CoT推理能否超越对教师算法的单纯模仿。我们证明，在获得梯度下降教师的初始监督之后，进一步使用真实回归权重进行轨迹级监督会使模型实现Heavy-Ball（动量）方法，从而实现对原始教师的加速。值得注意的是，这一加速算法的出现无需来自任何Heavy-Ball教师的监督……

    arXiv:2610.31089v1 Announce Type: new  Abstract: Modern transformers often solve complex tasks by generating and refining intermediate reasoning steps. To develop a theoretical understanding of chain-of-thought (CoT) reasoning, a notable recent study by Huang et al. (2025) viewed transformers as algorithm executors, and showed that when transformers' CoT steps are supervised by a gradient-descent teacher, they can learn to implement gradient descent for in-context weight prediction in linear regression. In this paper, we study the same in-context weight prediction task but ask whether CoT reasoning can go beyond merely imitating the teacher algorithm. We show that, after initial supervision from a gradient-descent teacher, further trajectory-level supervision using the true regression weights leads the model to implement the Heavy-Ball method, thereby achieving acceleration over the original teacher. Notably, this accelerated algorithm emerges without supervision from any Heavy-Ball te
    
[^86]: nnInteractive智能体：作为3D图像标注器的通用人工智能体

    nnInteractive Agent: General-Purpose AI Agents as 3D Image Annotators

    [https://arxiv.org/abs/2610.31080](https://arxiv.org/abs/2610.31080)

    该论文提出开源的nninteractive-agent系统，让通用AI智能体无需额外训练即可根据自然语言指令自动完成3D图像标注的全流程工作（定位目标、提示分割、检查与修正结果），通过将nnInteractive封装为MCP工具实现了交互式分割的全自动化。

    

    诸如nnInteractive之类的交互式3D分割模型可以将少量空间提示（点、涂鸦、框）转换为精确的体积掩膜，但标注工作仍需人工完成：寻找每个结构、放置提示、检查结果并进行修正。通用人工智能体如今已能够解读图像、调用工具并编写代码。我们证明这类智能体无需额外训练即可接管这项工作：用户用自然语言提出请求，智能体便会确定需要分割的内容、定位目标、为nnInteractive提供提示、检查3D结果、进行修正并响应反馈。当被要求找出“用于咀嚼的肌肉”时，智能体能够识别并分割咬肌和翼状肌；当被要求计算肝脏病变数量时，它会自行编写代码查找候选区域，并可根据要求拆分粘连在一起的病变。我们的开源nninteractive-agent通过模型上下文协议将nnInteractive作为工具提供，……

    arXiv:2610.31080v1 Announce Type: new  Abstract: Interactive 3D segmentation models such as nnInteractive turn a few spatial prompts (points, scribbles, boxes) into accurate volumetric masks, but a human still does the annotation work: finding each structure, placing prompts, checking the result and correcting it. General-purpose AI agents can now interpret images, call tools and write code. We show that such agents can take over this work without additional training: the user makes a request in plain language, and the agent works out what has to be segmented, locates the targets, prompts nnInteractive, inspects the 3D result, corrects it and responds to feedback. Asked for 'the muscles used for chewing', an agent identifies and segments the masseters and pterygoids; asked to count liver lesions, it writes its own code to find candidates and splits a merged lesion on request. Our open-source nninteractive-agent provides nnInteractive as tools over the Model Context Protocol (MCP), toge
    
[^87]: 当被遗忘的事实回归：基于切向核的知识图谱嵌入再学习感知遗忘

    When Forgotten Facts Return: Relearning-Aware Unlearning of Knowledge Graph Embeddings via Tangent Kernels

    [https://arxiv.org/abs/2610.31075](https://arxiv.org/abs/2610.31075)

    本文首次研究了知识图谱嵌入中被遗忘事实在持续训练下“复活”的再学习现象，并提出利用切向核建模这一恢复过程的核预期遗忘方法，使遗忘效果在持续训练中更加稳健。

    

    近似遗忘通过编辑实体嵌入而非从头重新训练，来响应知识图谱嵌入的删除请求。随着新事实的到来，这些嵌入会被持续更新。在这种常规的持续训练下，被遗忘的事实会比精确遗忘基线（即在没有这些事实的情况下重新训练并接受相同持续训练的模型）恢复得更多。我们将这种再学习现象追溯到被遗忘事实与新事实所共享的实体嵌入。据我们所知，这是首个针对知识图谱嵌入中再学习问题的研究。为深度网络开发的抗再学习方法并未考虑到这种恢复背后的图结构。我们提出了核预期遗忘，它利用切向核来建模持续训练如何通过这些共享参数改变被遗忘事实的得分。KAU 以闭式形式表达这种一阶响应，并在遗忘损失内部对其进行微分。

    arXiv:2610.31075v1 Announce Type: new  Abstract: Approximate unlearning answers a deletion request on a knowledge-graph embedding by editing entity embeddings instead of retraining from scratch. These embeddings keep being updated as new facts arrive. Under this ordinary continued training, forgotten facts recover more than in the exact-unlearning baseline, a model retrained without them and given the same continued training. We trace this relearning to entity embeddings shared by forgotten and new facts. To our knowledge, this is the first study of relearning in knowledge-graph embeddings. Relearning-robust methods developed for deep networks do not account for the graph structure underlying this recovery. We propose Kernel-Anticipated Unlearning (KAU), which uses the tangent kernel to model how continued training changes forgotten facts' scores through these shared parameters. KAU expresses this first-order response in closed form and differentiates through it inside the forgetting l
    
[^88]: 基于低秩预条件化的联邦差分隐私优化

    Federated Differentially Private Optimization via Low-Rank Preconditioning

    [https://arxiv.org/abs/2610.31069](https://arxiv.org/abs/2610.31069)

    本文提出一种联邦DP-PCA框架，通过低秩预条件化以线性通信成本实现中心化差分隐私保证，从而高效解决联邦差分隐私优化问题。

    

    我们研究联邦场景下的差分隐私（DP）优化问题，其中特征协方差矩阵的差分隐私估计被用作广义线性模型的Hessian近似。现有的中心化差分隐私协方差估计方法的通信成本随数据维度呈二次方增长，而具有线性通信成本的现有基线方法仅能提供本地隐私保护，且随着客户端数量的增加性能会下降。为解决这一问题，我们提出了一个联邦DP-PCA框架，在安全聚合下以线性通信成本提供中心化差分隐私保证。该框架由差分隐私秩选择步骤、差分隐私幂迭代法以及投影到所选低维子空间上的协方差的噪声化释放组成。我们给出了该协方差估计的误差界，以及基于该估计的低通信预条件化差分隐私梯度下降方法的误差界。实验结果表明，我们提出的差分隐私优化方法与中心化方法性能相当（原文摘要在此处截断）。

    arXiv:2610.31069v1 Announce Type: new  Abstract: We consider differentially private (DP) optimization in the federated setting, where a DP estimate of the feature covariance matrix serves as a Hessian approximation for generalized linear models. Existing central DP methods for covariance estimation have communication cost that grows quadratically in the data dimension, while existing baselines with linear communication cost are only locally private and degrade as the number of clients increases. As a remedy, we propose a federated DP-PCA framework that achieves linear communication cost while providing central DP guarantees under secure aggregation. It consists of a DP rank selection step, a DP power method, and a noisy release of the covariance projected onto the selected low-dimensional subspace. We give error bounds for this covariance estimate and for the resulting low-communication preconditioned DP gradient descent. Empirically, our proposed DP optimization method matches a centr
    
[^89]: 基于神经网络的晶体管建模：用于超快速SPICE仿真与标准单元库表征

    NN-Based Transistor Modeling for Ultra-Fast SPICE Simulation and Standard Cell Library Characterization

    [https://arxiv.org/abs/2610.31024](https://arxiv.org/abs/2610.31024)

    本文开发了能精确预测3nm纳米片晶体管漏极电流和电荷的神经网络模型，并首次在涉及数千次SPICE仿真的标准单元库表征这一复杂EDA流程中，严格验证了其相比工业标准模型的精度与超快速仿真能力。

    

    基于神经网络（NN）的晶体管模型已被证明是加速器件建模的一种有前景的解决方案。尽管这些模型在电路仿真中展现出了显著的加速效果，但它们尚未在芯片设计标准EDA工具流程的复杂任务中得到严格测试，而这些任务要求晶体管模型具有极高的精度，才能提供准确的结果并正确收敛。为了充分探究其潜力和前景，我们开发了基于神经网络的晶体管模型，能够准确预测3nm纳米片器件的漏极电流和各类电荷，并将这些模型应用于标准单元库表征。这使我们能够在非常广泛的SPICE仿真范围内评估模型的精度和加速效果，并与工业标准实现进行基准对比。由于标准单元库表征涉及数千次SPICE仿真，它可以作为检验晶体管模型质量和收敛性的绝佳测试基准。

    arXiv:2610.31024v1 Announce Type: cross  Abstract: Neural network (NN)-based transistor models have proven to be a promising solution to accelerate device modeling. Although these models demonstrated remarkable speedup in circuit simulations, they were not rigorously tested in complex tasks within standard EDA tool flows for chip design, which require very high transistor model precision to provide accurate results and converge properly. To investigate their full potential and promise, we develop NN-based transistor models that accurately predict the drain current and various charges for a 3nm nanosheet, and employ the models for a standard cell library characterization. This enables us to evaluate the accuracy and speedup under a very wide range of SPICE simulations, benchmarking our models against an industry-standard implementation. Since standard cell library characterization involves thousands of SPICE simulations, it serves as a great benchmark for transistor model quality and co
    
[^90]: 主要是阅读：开卷评分在文档级LoRA适配器中究竟衡量了什么

    Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

    [https://arxiv.org/abs/2610.31017](https://arxiv.org/abs/2610.31017)

    开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。

    

    单文档LoRA适配器已被提出作为可合并与检索的知识单元；然而，开卷评分——即将文档放入提示词中——会同样容易地把功劳归给文档自己的适配器（其专用适配器）的“阅读”行为，而非真正“记住”该文档。我们在Llama-3.1-8B-Instruct上为150篇Wikipedia文章各训练一个适配器，发现开卷分数主要衡量的是每个适配器都共有的一种阅读行为：一个在其他文档上训练的随机适配器能够恢复专用适配器相对于基础模型82%的开卷优势，但在一个不提供源文本、直接询问同一文档事实的完形填空探测中，只能恢复其2%的优势——这一比例与零无法区分。文档内容确实写入了权重，因为专用适配器在完形回忆上比基础模型高出0.094，且其更新的几何结构遵循其所阅读的文档；然而，在一项包含30个查询的测试中，通过权重空间邻近性选择的适配器……（原文在此处截断）

    arXiv:2610.31017v1 Announce Type: new  Abstract: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximi
    
[^91]: XALIGN：一种以人为中心的理解AI解释的可视化分析方法

    XALIGN: A Human-centric Visual Analytics Approach to Understanding AI Explanations

    [https://arxiv.org/abs/2610.31015](https://arxiv.org/abs/2610.31015)

    XALIGN通过并排展示同一随机森林模型的基于规则的解释与多种特征归因方法（TreeSHAP、KernelSHAP、LIME），使用户能够直观发现各解释方法之间的一致与矛盾，并结合领域知识核验模型预测。

    

    可解释AI（XAI）方法通常根据忠实性和鲁棒性等技术特性进行评估。在高风险应用领域，专家还需要对照既有知识来核验模型预测。不同的解释方法之间常常存在分歧，而当单独检查每种方法时，这种分歧难以察觉。我们提出了XALIGN，这是一个可视化分析工作流和界面，能够并排展示针对同一随机森林模型的基于规则的解释以及多种特征归因方法（TreeSHAP、KernelSHAP和LIME）。各方法之间的一致与矛盾成为可见的信号，用户可以据此进行探索并与领域知识进行对比。我们在两个案例研究中应用了XALIGN：德国信贷数据集和一个气象数据集，后者对来自ICON模型输出和Cloudnet观测的事件进行分类。在这两个案例中，相关视图揭示了基于规则的解释（原文此处截断）

    arXiv:2610.31015v1 Announce Type: new  Abstract: Explainable AI (XAI) methods are usually evaluated for technical properties such as faithfulness and robustness. In high-stakes application domains, experts also need to check model predictions against established knowledge. Different explanation methods often disagree, and this disagreement is hard to see when each method is inspected separately. We propose XALIGN, a visual analytics workflow and interface that shows rule-based explanations and several feature attribution methods (TreeSHAP, KernelSHAP, and LIME) side by side for the same Random Forest model. Agreements and contradictions between the methods become visible signals that users can explore and compare with domain knowledge. We apply XALIGN in two case studies: the German Credit dataset and a meteorological dataset, in which events derived from ICON model output and Cloudnet observations are classified. In both, the views reveal discrepancies between the rule-based explanati
    
[^92]: EEG-Captioner：基于基本信号属性的脑电到语言生成

    EEG-Captioner: Grounding EEG-to-Language Generation in Fundamental Signal Properties

    [https://arxiv.org/abs/2610.30995](https://arxiv.org/abs/2610.30995)

    提出了EEG-Captioner统一框架及包含16万余个片段、以信号级问答标注的EEG-语言数据集，通过让模型回答关于频谱、空间、时间、侧向等可测量信号属性的问题，来验证EEG到语言生成确实建立在对脑电信号的真实解读之上，并采用双域编码器联合建模时间波形与频谱-空间表示。

    

    脑电图（EEG）到语言模型为将复杂的神经信号翻译成人类可读的描述提供了一种有前景的方法。然而，生成看似合理的语言并不一定证明模型真正学会了解读底层的EEG信号。评估这种解读能力的一个直接方法是探测信号本身可客观测量的属性。因此，我们构建了一个包含160,781个片段的EEG-语言数据集，该数据集标注了信号级问答（QA）对，其中每个问题针对五类属性族之一，并需要涉及频谱、空间、时间或侧向信号属性的定量比较。每个片段还配有一个基于特征的EEG描述。我们进一步提出了一种双域EEG编码器，它联合建模时间波形与频谱-空间表示，并引入了EEG-Captioner——一个统一的EEG到语言框架，其训练采用……

    arXiv:2610.30995v1 Announce Type: new  Abstract: Electroencephalography (EEG)-to-language models offer a promising approach for translating complex neural signals into human-readable descriptions. However, generating plausible language does not necessarily demonstrate that a model has learned to interpret the underlying EEG signal. A direct way to assess such interpretation is to probe objectively measurable properties of the signal itself. We therefore construct an EEG-language dataset of 160,781 segments annotated with signal-level question answering (QA) pairs, where each question targets one of five property families and requires a quantitative comparison involving spectral, spatial, temporal, or lateral signal properties. Each segment is also paired with a feature-grounded EEG caption. We further propose a Dual-Domain EEG Encoder that jointly models temporal waveforms and spectral-spatial representations, and introduce EEG-Captioner, a unified EEG-to-language framework trained wit
    
[^93]: 诊断损失：面向放射学报告生成的非对称监督

    Diagnostic Loss: Asymmetric Supervision for Radiology Report Generation

    [https://arxiv.org/abs/2610.30980](https://arxiv.org/abs/2610.30980)

    提出诊断损失（DLoss），一种直接作用于解码器词元概率的可微分、架构无关的非对称监督目标，解决医学视觉-语言模型在放射学报告生成中遗漏关键诊断、引入错误诊断的目标错位问题。

    

    arXiv:2610.30980v1 公告类型：交叉  摘要：自动化放射学报告生成要求模型不仅仅是产出流畅的临床叙述：模型必须明确陈述有依据的诊断，同时避免给出无依据的诊断。然而，现有的医学视觉-语言模型（MedVLMs）通常采用词元级的最大似然目标进行训练，这类目标优先考虑文本一致性，而非诊断上的关键术语。其结果是，模型可能遗漏关键诊断并引入错误的诊断。我们将此问题识别为传统语言建模目标与放射学报告非对称诊断需求之间的目标错位。为了编码这些需求，我们提出了诊断损失（DLoss），这是一种可微分、与架构无关的目标函数，直接作用于解码器的词元概率。DLoss 分别在与参考报告中支持诊断相关联的词表、以及与不被支持的替代诊断相关联的词表上聚合概率质量。通过对聚合方式使用非对称……

    arXiv:2610.30980v1 Announce Type: cross  Abstract: Automated radiology report generation requires models to do more than produce fluent clinical narratives: they must explicitly state supported diagnoses while avoiding unsupported ones. However, existing medical vision-language models (MedVLMs) are typically trained with token-level maximum likelihood objectives, which prioritize text agreement over diagnostically critical terms. As a result, models can miss key diagnoses and introduce wrong ones. We identify this as an objective misalignment between conventional language modeling and the asymmetric diagnostic requirements of radiology reporting. To encode these requirements, we introduce Diagnostic Loss (DLoss), a differentiable, architecture-agnostic objective that operates directly on decoder token probabilities. DLoss separately aggregates probability mass over vocabularies associated with diagnoses supported by the reference report and with unsupported alternatives. Using asymmetr
    
[^94]: 联合还是逐选项？面向闭集决策的紧凑编码器：参数规模从1700万到10亿

    Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters

    [https://arxiv.org/abs/2610.30970](https://arxiv.org/abs/2610.30970)

    在闭集决策任务上，采用重排序器初始化的逐选项评分小型编码器（1700万至10亿参数）在精度上超越了联合评分和大语言模型提示，而延迟至多只增加一倍。

    

    许多语言模型流水线最终都要做一个闭集决策，例如选择某个工具。为此对大型语言模型进行提示既缓慢又对顺序敏感。我们探讨用于此类决策的编码器可以小到什么程度，以及它应该联合读取所有选项还是分别读取每个选项。在一项3200万参数、三个随机种子的研究中，在匹配优化器步数的条件下，逐选项评分在包含37个基准的决策套件的保留样本上，比联合评分高出5.8个点的机会校正技能分数；在近似匹配训练计算量的条件下高出2.8个点；两个外部基准的结果方向一致，但在其未见任务上差异不显著。将读取方式与初始化方式进行交叉实验表明，这一优势依赖于重排序器初始化，因为其查询-候选对输入方式与逐选项读取相匹配；而若从原始编码器出发，两种读取方式则难以区分。逐选项评分需要5到13倍的编码器前向传递次数：中位延迟至多增加一倍，但……

    arXiv:2610.30970v1 Announce Type: new  Abstract: Many language-model pipelines end in a closed-set decision, such as selecting a tool. Prompting a large language model for it is slow and order-sensitive. We ask how small an encoder for such decisions can be and whether it should read the options jointly or separately. In a three-seed study at 32 million parameters, per-option scoring beats joint scoring on a held-out sample of a 37-benchmark decision suite by 5.8 points of chance-corrected skill at matched optimizer steps and by 2.8 at approximately matched training compute; two external benchmarks agree, though not significantly on their unseen tasks. Crossing reader with initialisation shows that the advantage requires a reranker initialisation, whose query-candidate pair input matches per-option reading; from the raw encoder the two readers are indistinguishable. Per-option scoring needs 5 to 13 times more encoder passes: at most twice the median latency, but more than four times th
    
[^95]: 能耗与时延优先：部署约束如何逆转时间序列模型的选择

    Energy and Latency First: How Deployment Constraints Reverse Time-Series Model Selection

    [https://arxiv.org/abs/2610.30965](https://arxiv.org/abs/2610.30965)

    本文以蜂窝网络切换预测为例，证明在硬性40毫秒时延截止与设备全生命周期累计能耗约束下，按尾部截止期限可行性筛选并以包含特征计算的实测推理能耗排名，会逆转传统按准确率的时间序列模型选择，使基于手工时间特征的梯度提升树胜出。

    

    AI模型，尤其是时间序列模型，通常依据准确率进行排名，但其实际部署却受到该排名中未纳入的约束。本文研究了一个这些约束明确且可测量的真实场景：蜂窝网络中的切换预测，其中推理在用户设备（UE）上按每40毫秒的测量周期运行一次，且必须在网络采取行动之前完成。该时序界限是硬性的：一个迟到的预测会被它本来要预测的测量结果所取代；而能耗则是在设备整个生命周期内持续消耗的累计预算。这两者均未出现在这些模型惯常的选择目标中。我们在一套评估协议下对八种架构进行了评测：该协议先按尾部（而非均值）筛选是否满足截止期限，再对可行的模型按实测推理能耗进行排名，并将前向传播之前的特征计算（通常被忽略的部分）计入其中。基于手工构造时间特征的梯度提升树（摘要至此中断）

    arXiv:2610.30965v1 Announce Type: new  Abstract: AI models, and time-series models in particular are typically ranked by accuracy and deployed under constraints absent from that ranking. In this paper, we study a real setting where these constraints are explicit and measurable: handover pre- diction in cellular networks, where inference runs on the User Equipment once per 40 ms measurement period and must complete before the network acts. The timing bound is hard: a late prediction is superseded by the measurement it was predicting, while energy is a cumulative budget spent for the life of the device. Neither appears in the objective these models are selected on. We evaluate eight architectures under a protocol that screens for deadline feasibility at the tail rather than the mean, then ranks the feasible models on measured inference energy, count- ing the feature computation that precedes the forward pass and is usually omitted. Gradient-boosted trees on handcrafted temporal features 
    
[^96]: ReDence：用于快速、基于证据检测现代社会工程学诈骗的混合智能体与决策模型框架

    ReDence: A Hybrid Agentic and Decision-Model Framework for Rapid, Evidence-Grounded Detection of Modern Social Engineering Scams

    [https://arxiv.org/abs/2610.30963](https://arxiv.org/abs/2610.30963)

    该论文提出开源混合框架ReDence，通过微调多模态智能体“调查员”与专用“决策器”模型协同工作，实现对社会工程学诈骗的快速、自适应、基于证据的自动化检测。

    

    检测在线社会工程学（SE）诈骗通常需要检查网站工作流程、对照外部来源核实相关声明，并跨多个页面关联证据。现有检测器对此类调查提供的支持有限，导致分析人员不得不手动收集和验证证据，这不可避免地延迟了此类攻击的检测与后续下架。为弥补这一空白，我们提出了ReDence，一个开源的混合框架，可跨多种SE诈骗类别实现自适应、基于证据的检测。ReDence结合了两个本地部署、职责互补的模型：一个是经过微调的多模态智能体“调查员”（Investigator，Qwen3.5-4B），负责调查规划、证据解读与报告生成；另一个是专门的“决策器”（Decider，Tev1），负责处理二选一与多选任务，包括浏览器操作选择、证据评估，以及是否继续调查的决策。

    arXiv:2610.30963v1 Announce Type: cross  Abstract: Detecting online social engineering (SE) scams often requires examining website workflows, verifying claims against external sources, and connecting evidence across multiple pages. Existing detectors provide limited support for these investigations, leaving analysts to manually gather and validate evidence, which inevitably delays detection and subsequent takedown of these attacks. To address this gap, we present ReDence, an open-source hybrid framework for adaptive, evidence-grounded detection across diverse SE scam categories. ReDence combines two locally hosted models with complementary responsibilities: a fine-tuned multimodal agentic Investigator (Qwen3.5-4B) that performs investigative planning, evidence interpretation, and reporting, and a specialized Decider (Tev1) that handles binary and multiple-choice tasks, including browser-action selection, evidence assessment, and decisions about whether to continue an investigation. By 
    
[^97]: 双重稳健的分布式鲁棒离线上下文定价

    Doubly Robust Distributionally Robust Offline Contextual Pricing

    [https://arxiv.org/abs/2610.30957](https://arxiv.org/abs/2610.30957)

    本文针对连续定价场景下的分布式鲁棒离线策略评估与学习，提出了一个双重稳健框架，通过局部化DR估计器克服了基于IPW方法对倾向得分估计收敛速度的敏感性以及最坏情况期望的计算难题。

    

    离线上下文定价通常依赖于记录的观测数据，但面临训练环境与部署环境之间分布偏移的挑战。分布式鲁棒优化为离线策略评估与学习（OPE/L）提供了一种有原则的方法。然而，现有方法大多仅限于离散动作。最近的工作探索了使用逆倾向加权（IPW）对连续处理进行分布式鲁棒优化，但此类基于IPW的估计器对倾向得分估计的收敛速度较为敏感，特别是在非参数估计的情况下，这可能导致更大的估计误差和遗憾。在这项工作中，我们为连续定价环境下的分布式鲁棒OPE/L开发了一个双重稳健（DR）框架。对于评估，我们提出了一个局部化DR估计器，通过仅拟合少量回归来解决最坏情况期望的计算挑战……

    arXiv:2610.30957v1 Announce Type: new  Abstract: Offline contextual pricing often relies on logged observational data, but faces challenges from distributional shifts between training and deployment environments. Distributionally robust optimization (DRO) provides a principled approach to off-policy evaluation and learning (OPE/L). However, existing methods are mostly limited to discrete actions. Recent work has explored DRO for continuous treatments using inverse propensity weighting (IPW), while such IPW-based estimators can be sensitive to the convergence rate of propensity score estimates, particularly when estimated nonparametrically, which may lead to larger estimation errors and regret. In this work, we develop a doubly robust (DR) framework for distributionally robust OPE/L in continuous pricing settings. For evaluation, we propose a localized DR estimator that addresses the computational challenges of worst-case expectations by fitting only a small number of regressions, compa
    
[^98]: 基于逆最优性的强化学习“真值优先”基准测试：从平滑、完全可观测系统到非平滑或部分可观测任务

    Ground-Truth-First Benchmarking of Reinforcement Learning via Converse Optimality: From Smooth, Fully Observed Systems to Non-Smooth or Partially Observed Tasks

    [https://arxiv.org/abs/2610.30954](https://arxiv.org/abs/2610.30954)

    该论文提出基于逆最优性的“真值优先”强化学习基准测试框架，通过预先构建并保密经认证的最优值函数与策略，使算法性能可通过最优性差距、后悔值和策略误差等指标进行绝对化评估，并同时支持对任务变化敏感性的系统性分析。

    

    强化学习中的基准测试通常是相对的：算法在最优值未知的环境上根据回报进行排名。与最优性的距离始终不可观测，而结论可能依赖于奖励设计、随机性、超参数、环境选择和模拟器细节。我们开发了一个基于逆最优性的“真值优先”框架，为每个任务构建经过认证的最优值函数和策略，这些内容对学习者保密，仅用于评估。该框架统一了两种模式：敏感性基准测试从连续时间生成器中采样经过认证的环境，以评估算法对任务变化的鲁棒性；而以预言机为参照的比较则使用原生离散时间生成器，通过最优性差距、后悔值和策略误差来评估策略。连续时间模型支持物理上可解释的敏感性研究，但引入了对模拟器步长（Δt）和数值方案的依赖……

    arXiv:2610.30954v1 Announce Type: new  Abstract: Benchmarking in reinforcement learning is usually relative: algorithms are ranked by returns on environments with unknown optima. Distance to optimality remains unobservable, while conclusions may depend on reward design, stochasticity, hyperparameters, environment selection, and simulator details. We develop a ground-truth-first framework based on converse optimality, constructing each task with a certified optimal value function and policy withheld from learners and used only for evaluation. The framework unifies two modes: sensitivity benchmarking samples certified environments from continuous-time generators to assess robustness to task variation, while oracle-referenced comparison uses native discrete-time generators to evaluate policies via optimality gap, regret, and policy error. Continuous-time models support physically interpretable sensitivity studies but introduce dependence on the simulator step ($\Delta t$) and numerical sc
    
[^99]: OpenAttention：用尾部代理，百分之一足矣

    OpenAttention: One Percent Is Enough with Tail Proxies

    [https://arxiv.org/abs/2610.30951](https://arxiv.org/abs/2610.30951)

    OpenAttention通过将整个未读尾部压缩为单个计数感知均值代理token，在读取率低至1%时仍能获得比丢弃或采样更低误差的稀疏注意力输出。

    

    arXiv:2610.30951v1 公告类型：新论文 摘要：稀疏注意力只读取键值缓存中的一小部分，而如何处理未被读取的尾部仍然是一个关键挑战。直接丢弃尾部会使输出产生偏差，而以往方法采用的尾部采样策略在读取率为5%及以下时会失效，因为此时尾部很可能包含选择器遗漏的重要token。我们的关键观察是，最简单的代理形式——一个将整个尾部折叠为其计数感知均值的token——在实用选择器下的每个读取率上，都能产生比采样更低的输出误差。我们进一步以闭式形式证明，在低读取率下，采样带来的权重误差几乎是丢弃的两倍，而单个代理的误差则始终低于丢弃。基于这些发现，我们探索用少量针对尾部的代理来近似稠密注意力分布。我们提出了OpenAttention，这是一个由聚类、ErrorGate和top-k选择加上计数感知均值代理组成的流水线。为了高效地实现聚类和选择……

    arXiv:2610.30951v1 Announce Type: new  Abstract: Sparse attention reads only a small subset of the key-value cache, and how to treat the unread tail remains a key challenge. Discarding the tail biases the output, and sampling it as previous approaches do fails at reads of 5% and below, where the tail likely holds heavy hitters the selector missed. Our key observation is that the simplest possible proxy, one token that folds the entire tail into its count-aware mean, already yields lower output error than a sample at every read behind a practical selector. We further show in closed form that at low reads sampling nearly doubles the weight error of discarding while a single proxy stays below it. Building on these findings, we explore approximating the dense attention distribution with a few proxies that target the tail. We propose OpenAttention, a pipeline of clustering, ErrorGate and top-k selection followed by a count-aware mean proxy. To implement clustering and selection both efficie
    
[^100]: 随机初始化作为时间序列基础模型的基线

    Random Initialization as a Baseline for Time Series Foundation Models

    [https://arxiv.org/abs/2610.30950](https://arxiv.org/abs/2610.30950)

    本研究通过系统对比发现，时间序列基础模型的预训练权重相比架构相同的随机初始化模型仅带来平均2.58个百分点的准确率提升，表明预训练收益有限，随机初始化可作为评估此类模型的有效基线。

    

    时间序列基础模型（TSFMs）利用自监督预训练来学习可复用的表示，以用于多元时间序列分类（MTSC）等下游任务。我们通过在UEA MTSC数据集上，将来自四个TSFM家族的七个公开发布模型实例与架构匹配的随机初始化对应模型进行比较，来检验预训练权重在架构本身之外究竟贡献了多少。对于每个实例，在冻结表示上使用四种分类器评估五个随机初始化版本：岭分类、逻辑回归、RBF支持向量机和随机森林。在因计算限制进行模型特定的排除之后，该研究共包含183组配对的数据集级比较。预训练提高了所有七个实例的平均准确率，但增益较为有限，介于1.43至5.19个百分点之间，总体平均提升为2.58个百分点。七个实例中有三个的逐实例Wilcoxon符号秩检验……（摘要在此处截断）

    arXiv:2610.30950v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) use self-supervised pretraining to learn reusable representations for downstream tasks such as multivariate time series classification (MTSC). We examine how much pretrained weights contribute beyond the architecture itself by comparing seven publicly released instances from four TSFM families with architecture-matched randomly initialized counterparts on the UEA MTSC archive. For each instance, five random initializations are evaluated using four classifiers on frozen representations: ridge classification, logistic regression, RBF SVM, and random forest. After model-specific exclusions due to computational constraints, the study comprises 183 paired dataset-level comparisons. Pretraining improves mean accuracy for all seven instances, but the gains are modest, ranging from 1.43 to 5.19 percentage points, with an overall average improvement of 2.58 points. Three of seven per-instance Wilcoxon signed-
    
[^101]: 谱算法的随机投影：Sobolev范数下的最优速率

    Random Projections for Spectral Algorithms: Optimal Rates in Sobolev Norms

    [https://arxiv.org/abs/2610.30948](https://arxiv.org/abs/2610.30948)

    本文利用RKHS嵌入性质与算子相似性的精细分析，证明了基于随机投影（随机化草图与Nyström子采样）的谱算法在一般条件下于Sobolev范数中达到匹配极小极大下界的最优学习速率，且不牺牲统计效率。

    

    随机投影（RP）提供了一种在保持数据几何结构的同时降低计算与存储开销的有效方法。然而，现有研究主要集中于在设定良好且条件受限的情况下，针对采用随机投影的特定核正则化算法的最优泛化性能。在本文中，我们在不牺牲统计效率的前提下，对基于随机投影的谱算法在一般条件下的泛化性能进行了全面且改进的分析。通过利用再生核希尔伯特空间（RKHS）的嵌入性质以及对算子相似性的精细化分析，我们建立了Sobolev范数下的最优学习速率，其在对数因子范围内与极小极大下界相匹配。对于随机投影的两种常见实现方式——随机化草图和Nyström子采样（均匀采样或基于杠杆得分的采样），我们表明（摘要原文在此处截断）

    arXiv:2610.30948v1 Announce Type: cross  Abstract: Random projections (RP) offer an effective method for reducing computational and storage overhead while preserving the geometric structure of data. However, existing studies have primarily focused on the optimal generalization performance of specific kernel-regularized algorithms with RP in the well-specified setting and under restrictive conditions. In this paper, we provide a comprehensive and improved analysis of the generalization performance of RP-based spectral algorithms under general conditions \change{without sacrificing statistical efficiency}. By leveraging the embedding property of the reproducing kernel Hilbert space (RKHS) and a refined analysis of the operator similarity, we establish optimal learning rates in Sobolev norms that match the minimax lower bounds up to logarithmic factors. For two common implementations of RP, randomized sketches and Nystr\"{o}m sub-sampling (uniform or leverage-based), we show that the proj
    
[^102]: GateFormer：基于预测空间分解的天空图像小时级内太阳辐照度预测

    GateFormer: Forecast-Space Decomposition for Intra-Hour Solar Irradiance Prediction from Sky Images

    [https://arxiv.org/abs/2610.30942](https://arxiv.org/abs/2610.30942)

    GateFormer提出了一种以GHI为条件的双解码器视觉Transformer，通过将观测辐照度嵌入视觉token流、进行预测空间分解并在预测空间中自适应加权融合轨迹与增量两类预测，实现了基于天空图像的更精准小时级内太阳辐照度预测。

    

    arXiv:2610.30942v1 公告类型：cross

    arXiv:2610.30942v1 Announce Type: cross  Abstract: Sub-hourly dispatch, balancing, and storage control increasingly depend on forecasts that anticipate rapid variability in solar generation. Accurate sky-image-based irradiance forecasting requires a combination of visual cloud dynamics with recent irradiance state while adapting to different short-term irradiance-change regimes. To address this, we introduce GateFormer, a GHI-conditioned dual-decoder ViT for intra-hour global horizontal irradiance (GHI) forecasting through forecast-space decomposition. GateFormer embeds observed GHI into the visual token stream, allowing interaction with sky-image patch features through self-attention. From this shared representation, a Trajectory Head predicts future GHI sequence, while a Delta Head predicts final-horizon change from the latest observed GHI. The two final-horizon forecasts are adaptively weighted in prediction space, with magnitude- and direction-supervision targeting the remaining er
    
[^103]: PAIR：面向LinkedIn高端人物推荐的偏好对齐意图检索框架

    PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn

    [https://arxiv.org/abs/2610.30941](https://arxiv.org/abs/2610.30941)

    LinkedIn提出PAIR检索框架，通过建模用户意图偏好而非个人资料相似度，解决了专业网络中跨角色互补匹配的人物推荐问题。

    

    在专业网络中，基于个人资料相似度训练的通用嵌入常常会错误地对候选人进行排序，因为其相关性在本质上是互补的而非对称的。例如，求职者通常与招聘人员互动，而不是与拥有相似资历的同行互动。当前的用户推荐系统通常建立在个人资料相似度或社交信号（如共同好友）之上，这类方法在以交友为导向的社交网络中是有效的，因为用户倾向于与相似的人建立联系或在现有社交圈内互动。然而，这一假设在LinkedIn等专业平台上并不成立，因为这些平台上的用户互动往往是由目标驱动的跨角色互动，需要基于互补意图而非相似性或社交邻近度来进行匹配。在本文中，我们提出了偏好对齐意图检索，这是一个旨在建模用户意图偏好而非个人资料相似度的检索框架。

    arXiv:2610.30941v1 Announce Type: cross  Abstract: General-purpose embeddings trained on profile similarity often misrank candidates in professional networks, where relevance is inherently complementary rather than symmetric. For example, job seekers usually engage with recruiters rather than peers with similar credentials. Current user recommendation systems are often built on profile similarity or social signals such as shared connections, which are effective in friend-oriented social networks where users tend to connect with similar people or within existing social circles. However, this does not hold in professional platforms like LinkedIn, where interactions are often goal-driven and cross-role interactions, requiring matching based on complementary intents rather than similarity or social proximity. In this paper, we propose Preference-Aligned Intent Retrieval, a retrieval framework that is designed to model user intent preference instead of profile similarity. Our approach is in
    
[^104]: 基于数据稀缺天空图像的视觉Transformer辐照度估计方法

    Vision Transformers for Estimating Irradiance Using Data Scarce Sky Images

    [https://arxiv.org/abs/2610.30940](https://arxiv.org/abs/2610.30940)

    该论文提出一种面向数据稀缺场景的双框架视觉Transformer方法，通过整合计算大气参数（地外辐照度与周期性时间编码）并以顺序预测的合成GHI作为特征，无需昂贵测量仪器即可从天空图像准确估计散射水平辐照度。

    

    准确估计散射水平辐照度（DHI）对于优化光伏系统性能和能量预测至关重要，但在缺乏全面地面观测仪器的地区仍然是一项挑战。近期的研究进展表明，利用在大量天空图像数据集上训练的视觉Transformer（ViT）有望替代昂贵的辐照度测量设备，然而长期高质量天空图像数据的稀缺严重限制了其实际应用。针对这一关键缺口，本研究提出了一种专为数据稀缺场景设计的新型双框架方法。首先，整合计算得到的气象参数（包括地外辐照度和周期性时间编码）来表征天空状况，无需使用任何仪器设备。其次，采用顺序流水线，先预测合成的全球水平辐照度（GHI），并将其作为特征，用于改进DHI的估计。

    arXiv:2610.30940v1 Announce Type: cross  Abstract: Accurate estimation of diffuse horizontal irradiance (DHI) is critical for optimising photovoltaic system performance and energy forecasting yet remains challenging in regions lacking comprehensive ground-based instrumentation. Recent advancements using Vision Transformers (ViTs) trained on extensive sky image datasets have shown promise in replacing costly irradiance measurement equipment, but the scarcity of long-term, high-quality sky imagery significantly restricts practical implementation. Addressing this critical gap, this study proposes a novel dual-framework approach designed for data-scarce scenarios. First, calculated atmospheric parameters, including extraterrestrial irradiance and cyclic time encodings, are integrated to represent sky conditions without utilising any instrumentation. Next, a sequential pipeline initially predicts synthetic global horizontal irradiance (GHI) and uses it as a feature, to refine DHI estimation
    
[^105]: 桥接数论素性检测与算法优化：一种自适应密码学框架

    Bridging Number-Theoretic Primality Testing And Algorithmic Optimization: An Adaptive Cryptographic Framework

    [https://arxiv.org/abs/2610.30935](https://arxiv.org/abs/2610.30935)

    该论文提出了一个将经典数论与机器学习模型及优化算法相结合的自适应任意精度素性检测框架，以满足RSA等公钥密码系统对大素数生成、检测与证明的核心需求。

    

    现代公钥密码系统（如RSA、Diffie-Hellman和ECC）的安全性完全依赖于整数分解问题和离散对数问题的计算难解性。这些非对称算法要正常运行，一个核心要求是系统必须能够生成、检测并同时证明超大数字的素性。为满足这些需求，本研究提出了一个综合性框架，将经典数论与现代机器学习模型及优化算法相桥接。具体而言，本文设计了一种专门用于素性检测的自适应、任意精度架构。与此同时，该研究深入分析了检测方法（包括确定性方法和概率性方法）的理论基础、硬件级优化技术以及量子安全熵源的应用，并提供了……

    arXiv:2610.30935v1 Announce Type: cross  Abstract: The security of modern public-key cryptographic systems, such as RSA, Diffie-Hellman, and ECC, depends entirely on the computational intractability of the integer factorization and discrete logarithm problems. For these asymmetric algorithms to operate, a core requirement is that the system must be capable of generating, testing, and simultaneously proving the primality of extraordinarily large numbers. To meet those requirements, this study proposes a comprehensive framework, bridging clas- sical number theory with modern machine learning models and optimization algorithms. Specifically, the paper delineates an adaptive, arbitrary-precision architecture dedicated exclusively to primality testing. Concurrently, the research deeply analyzes the theoretical foundations of testing methods (both deterministic and probabilistic), hardware-level optimization techniques, and the application of quantum-secured entropy sources, while providing 
    
[^106]: MIG-COW：基于梯度分解的深度伪造检测器可迁移对抗攻击

    MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors via Gradient Decomposition

    [https://arxiv.org/abs/2610.30914](https://arxiv.org/abs/2610.30914)

    MIG-COW通过将动量积分梯度与共识正交分解相结合，捕获CNN与ViT检测器的共享及特有漏洞，在AADD-2025挑战赛中实现了99.96%的白盒攻击成功率并显著优于现有基线，同时揭示了黑盒可迁移性仍是待解决的挑战。

    

    尽管近期取得了诸多进展，深度伪造检测器仍然容易受到对抗样本的攻击，尤其是在多样化、真实世界的场景中。我们提出了MIG-COW，一种新颖的对抗攻击框架，能够生成高度泛化且视觉上不可察觉的对抗样本。通过将动量积分梯度与共识正交分解相结合，MIG-COW能够同时捕获异构CNN和ViT检测器中共享的以及模型特有的漏洞。在AADD-2025挑战赛基准测试中，MIG-COW实现了99.96%的白盒攻击成功率（ASR），同时保持了高感知相似度（SSIM），显著优于现有基线方法。然而，尽管获得了最佳综合评分，其对官方黑盒目标仅7.16%的有限攻击成功率凸显了可迁移性方面仍然存在的挑战。我们还证明，在集成中加入性能较低但多样化的模型可能会降低攻击效果。

    arXiv:2610.30914v1 Announce Type: cross  Abstract: Despite recent advances, deepfake detectors remain vulnerable to adversarial examples, particularly in diverse, real-world settings. We propose MIG-COW, a novel adversarial attack framework that generates highly generalizable and visually imperceptible adversarial examples. By combining momentum-integrated gradients with a consensus-orthogonal decomposition, MIG-COW captures both shared and model-specific vulnerabilities across heterogeneous CNN and ViT detectors. On the AADD-2025 Challenge benchmarks, MIG-COW achieves a 99.96% white-box attack success rate (ASR) with high perceptual similarity (SSIM), significantly outperforming existing baselines. However, its limited 7.16% ASR against official black-box targets - despite achieving the best overall score - highlights the ongoing challenge of transferability. We also demonstrate that incorporating low-performing but diverse models in the ensemble can degrade attack effectiveness, unde
    
[^107]: 同一分子，不同的知识：化学领域大语言模型中的跨标识符绑定失效

    Same molecule, Different Knowledge: Cross-Identifier Binding Failures in LLMs For chemistry

    [https://arxiv.org/abs/2610.30903](https://arxiv.org/abs/2610.30903)

    该论文通过受控基准和行为、表征、因果分析，首次揭示并验证了大语言模型中“跨标识符绑定失效”现象，即同一分子的俗名、IUPAC名称和SMILES等互指标识符所激活的模型知识子集并不互通。

    

    化学AI工作流程通常通过多个互指标识符来表示同一分子，包括俗名、IUPAC名称和SMILES，并隐含地假设这些表示形式能够以可互换的方式访问大语言模型的知识。为了验证这一假设，我们首先引入了一个受控基准测试，结果表明该假设并不成立。在参数规模从7B到72B的基础模型中，即使是化学专业模型，互指的分子标识符始终暴露出模型知识的不同子集。随后，我们进行了一系列受控的行为、表征和因果分析来刻画这种失败现象。我们证明，即使模型能够通过SMILES识别出某个分子，并通过相应的名称检索到某个事实，同一事实却可能无法直接从SMILES中获取。我们将这种现象称为“跨标识符绑定失效”。对内部表征的进一步分析……

    arXiv:2610.30903v1 Announce Type: new  Abstract: Chemical AI workflows routinely represent the same molecule through multiple co-referring identifiers, including trivial names, IUPAC names, and SMILES, implicitly assuming that these representations provide interchangeable access to an LLM's knowledge. To test this assumption, we first introduce a controlled benchmark and show that it does not hold. Across foundation models spanning 7B to 72B parameters, even chemistry-specialized models, co-referring molecular identifiers consistently expose different subsets of the model's knowledge. We then conduct a series of controlled behavioral, representational, and causal analyses to characterize this failure. We show that even when a model can identify a molecule from its SMILES and retrieve a fact through the corresponding name, the same fact can remain inaccessible directly from SMILES. We term this phenomenon \emph{cross-identifier binding failure}. Further analysis of internal representati
    
[^108]: 基于变分消息传递的主动推理参数学习

    Parameter Learning in Active Inference via Variational Message Passing

    [https://arxiv.org/abs/2610.30899](https://arxiv.org/abs/2610.30899)

    该论文将主动推理中的期望自由能最小化统一到变分消息传递框架下，实现了基于狄利克雷先验的在线参数学习与规划的融合，并通过为每个因子分配独立参数先验解决了传统平均场分解下信息增益校正消失的关键难题。

    

    主动推理将感知、学习和规划统一表述为变分推断，其中期望自由能（EFE）定义了规划目标。近期研究表明，EFE最小化可以等价地写作在带有认知先验的增强生成模型上进行变分自由能（VFE）最小化，并且所得的熵校正目标可以通过在修改后的因子函数上执行和积消息传递（SP-MP）来最小化。这些公式将模型不确定性视为对有限个已知模型的推断。本文将其扩展到学习过程中的规划，其中对观测参数和转移参数 $\theta_A$、$\theta_B$ 的信念采用狄利克雷先验表示。闭式狄利克雷更新通常通过参数与状态之间的结构化平均场分解来获得，但恰恰在这种分解下，信息增益校正项会消失。我们通过为每个因子赋予独立的参数先验来解决这一问题……（摘要原文在此处截断）

    arXiv:2610.30899v1 Announce Type: cross  Abstract: Active inference casts perception, learning and planning as variational inference, where the Expected Free Energy (EFE) defines the planning objective. Recent work showed that EFE minimization can be written as Variational Free Energy (VFE) minimization on a generative model augmented with epistemic priors, and that the resulting entropy-corrected objective is minimized by sum-product message passing (SP-MP) on modified factor functions. These formulations treat model uncertainty as inference over a finite set of known models. We extend them to planning during learning, where beliefs over the observation and transition parameters $\theta_A$, $\theta_B$ are Dirichlet priors. Closed-form Dirichlet updates are usually obtained by a structured mean-field factorization between parameters and states, but under exactly that factorization the information-gain corrections vanish. We resolve this by giving every factor its own parameter prior, f
    
[^109]: 竞争性做市中的均衡与学习

    Equilibria and Learning in Competitive Market Making

    [https://arxiv.org/abs/2610.30891](https://arxiv.org/abs/2610.30891)

    本文提出了多策略做市商与代表性交易者之间的做市博弈模型，证明了纯策略纳什均衡在极弱假设下总是存在且买卖价差随波动率严格变宽，并进一步证明当报价单位趋于零时无内部遗憾学习者的时间平均策略收敛于连续纳什均衡价格包络。

    

    我们提出了一个在 N ≥ 2 个策略性做市商与单个非策略性代表性交易者之间进行的做市博弈。在连续报价空间中，标准的有限博弈论证并不适用，而非凸且不连续的收益函数排除了经典的连续性存在性结果，并使学习问题复杂化。尽管如此，我们刻画了纯策略纳什均衡（PSNE），并证明在对比标资产价值分布的极小假设下，至少总存在一个纯策略纳什均衡，且均衡买卖价差随资产价值波动率的增大而严格变宽。为了获得一个易于处理的学习框架，我们对价格进行离散化，并在附加的正则性条件下证明：当最小报价单位趋于零时，无内部遗憾学习者的时间平均策略收敛于连续纳什价格包络。最后，我们通过数值模拟对理论结果进行了实证验证。

    arXiv:2610.30891v1 Announce Type: cross  Abstract: We propose a market-making game played among $N \ge 2$ strategic market makers and a single non-strategic representative trader. In a continuous quote space, standard finite-game arguments do not apply, while non-convex and discontinuous payoffs rule out classical continuous existence results and complicate learning. Nonetheless, we characterize pure-strategy Nash equilibria (PSNEs) and show that at least one PSNE always exists under minimal assumptions on the underlying asset value distribution, and that the equilibrium bid-ask spread widens strictly with asset value volatility. To obtain a tractable learning framework, we discretize prices and, under an additional regularity condition, show that the time-averaged play of no-internal-regret learners approaches the continuous Nash price envelope as the tick size vanishes. Finally, we empirically validate our theoretical results through numerical simulations.
    
[^110]: 连续深度的深度高斯过程

    Continuous-depth Deep Gaussian Processes

    [https://arxiv.org/abs/2610.30890](https://arxiv.org/abs/2610.30890)

    提出将深度高斯过程的深度视为由随机微分方程刻画的连续演化，并在深度索引的漂移函数族上施加高斯过程先验（CDGP 与 FlowDGP 两种实现），从而缓解传统离散深度 DGP 随深度增长而难以优化的问题。

    

    深度高斯过程（DGPs）是一类很有吸引力的贝叶斯回归模型，但其标准的离散深度参数化往往随深度增加而变得更难优化。我们认为，与其将深度视为一长串离散堆叠，不如将其视为一种潜在的连续演化。为此，我们提出了一种连续深度的建模形式：隐表示遵循一个随机微分方程，其漂移项是按深度索引的函数族，并在该函数族上配备基于高斯过程的先验。这将核心设计选择重新聚焦于施加在该函数族上的先验，而不是像神经常微分方程（Neural ODE）或神经随机微分方程（Neural SDE）模型那样采用纯神经网络的连续深度参数化。我们研究了这一思想的两种具体实现：CDGP，对状态-深度漂移直接施加高斯过程先验；以及 FlowDGP，一种通过流演化得到的先验，能以更低的训练成本捕捉更丰富的深度依赖关系。为论证这一重新设计的必要性，我们分析了单样本 DGP 目标如何会变得愈发敏感……（原文摘要在此处被截断）

    arXiv:2610.30890v1 Announce Type: cross  Abstract: Deep Gaussian processes (DGPs) are appealing Bayesian regression models, but their standard discrete-depth parameterization often becomes harder to optimize as depth grows. We argue that depth is better treated as a latent continuous evolution than as a long discrete stack. We therefore propose a continuous-depth formulation in which the latent representation follows a stochastic differential equation with a depth-indexed drift family equipped with a GP-based prior. This reframes the core design choice as the prior placed on that family, rather than adopting a purely neural continuous-depth parameterization as in neural ODE or neural SDE models. We study two instantiations of this idea: CDGP, a direct GP prior over the state--depth drift, and FlowDGP, a flow-evolved prior that captures richer depth dependence at lower training cost. To motivate the redesign, we analyze how the single-sample DGP objective can become increasingly sensiti
    
[^111]: 构建面向高质量幻灯片生成的演示智能

    Building Presentation Intelligence for High-Quality Slide Generation

    [https://arxiv.org/abs/2610.30888](https://arxiv.org/abs/2610.30888)

    该论文提出了一个面向演示智能的统一训练框架，通过交叉验证的数据合成、基于规则与先验知识的教师增强以及可验证奖励优化，实现高质量的自动PPT生成。

    

    PowerPoint演示文稿（PPT）是各个领域中沟通的主流媒介，然而高质量的PPT生成仍然严重依赖大量的人力投入。大语言模型（LLM）在智能体任务方面的最新进展为自动PPT生成带来了新的机遇与挑战。在这项工作中，我们提出了一个面向演示智能的自动PPT生成统一训练框架，该框架集成了交叉验证的数据合成、引导式教师增强以及可验证奖励优化。我们首先构建了一个交叉验证的数据合成流程，能够对难度和有效性进行显式控制。随后，我们开发了一个基于规则和先验知识的教师增强框架，以显著增强教师模型的指导能力。在此基础上，我们通过混合教师策略进一步改进了监督微调（SFT）……（原文摘要在此处被截断）

    arXiv:2610.30888v1 Announce Type: new  Abstract: PowerPoint presentations (PPTs) are a dominant medium for communication across diverse domains, yet high-quality PPT generation still depends heavily on labor-intensive human effort. Recent advances in large language models (LLMs) for agentic tasks have created both new opportunities and challenges (Chen et al., 2026) for automatic PPT generation. In this work, we present a unified training framework for automatic PPT generation toward presentation intelligence, integrating cross-validated data synthesis, guided teacher enhancement, and verifiable reward optimization. We first construct a cross-validated pipeline for data synthesis, with explicit control over both difficulty and validity. We then develop a teacher-enhancement framework grounded in rules and prior knowledge to substantially strengthen the instructional capacity of teacher models. Building on this, we further improve supervised fine-tuning (SFT) through a mixed-teacher str
    
[^112]: 追踪预训练过程中的潜意识学习

    Tracing Subliminal Learning Through Pretraining

    [https://arxiv.org/abs/2610.30884](https://arxiv.org/abs/2610.30884)

    该研究证明“潜意识学习”现象在硬标签蒸馏和同策略软标签蒸馏中均普遍存在，且不受权重初始化、数据顺序、语料库、模型规模和架构的影响，并首次发现传递偏好与获得偏好的能力在预训练过程中独立发展，导致教师与学生角色互换时迁移强度呈现强烈不对称。

    

    大型语言模型在由其他模型生成的数据上训练时，有时会继承与该数据无关的特征，这种现象被称为“潜意识学习”（subliminal learning）。我们研究了两种设置：硬标签蒸馏（学生在由有偏差的教师生成的数字上训练）和同策略软标签蒸馏（学生自己生成数据，教师的logits提供学习信号）。使用Pythia、PolyPythia和DataDecide OLMo模型，我们发现这种迁移在两种设置中均持续存在，且不受权重初始化、预训练数据顺序、预训练语料库、模型规模和架构差异的影响。在预训练的各个检查点上，传递偏好的能力和获得偏好的能力是分开发展的：某些检查点作为教师更有效但作为学生较弱，反之亦然。与这一区别相一致，当教师和学生的角色互换时，跨模型迁移在程度上可能表现出强烈的不对称性。

    arXiv:2610.30884v1 Announce Type: new  Abstract: Large language models trained on data generated by other models can sometimes inherit traits unrelated to that data, a phenomenon called subliminal learning. We study both hard label distillation, where students train on numbers generated by a biased teacher, and on-policy soft label distillation, where students generate the data and the teacher's logits provide the learning signal. Using Pythia, PolyPythia, and DataDecide OLMo models, we find that transfer persists in both settings across differences in weight initialization, pretraining data order, pretraining corpus, model size, and architecture. Across pretraining checkpoints, the ability to transmit a preference and the ability to acquire one develop separately: some checkpoints are more effective teachers but weaker students, and vice versa. Consistent with this distinction, cross-model transfer can be strongly asymmetric in magnitude when teacher and student roles are reversed. To
    
[^113]: 承袭而非发明：追溯推理模型中“Wait”（等一下）一词的来源

    Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models

    [https://arxiv.org/abs/2610.30882](https://arxiv.org/abs/2610.30882)

    研究揭示推理模型中的{wait}标记并非由强化学习发明，而是从训练数据（中期训练与推理轨迹微调）中继承而来，RL只能放大模型已有的使用习惯。

    

    强化学习（RL）被认为教会了推理模型在重新审视某一步骤之前先写下{wait}（等一下），然而RL带来的巨大准确率提升并不需要这个词。这种观点可能混淆了RL所“发现”的内容与模型从训练数据中“继承”的内容。我们通过开源模型的训练历史追溯这个词，采用无标签的频率、位置、形式和用法指标以及注入错误测试来分析它，所有指标均从采样输出中计算得出。我们发现zero-RL几乎从不创造这个标记：在缺乏该词的基础模型上，RL虽能大幅提升MATH-500准确率，但{wait}却几乎依然缺席。这个词是随训练数据而来的：它出现在OLMo-3的中期训练中，而基于推理轨迹的微调则将其引入大多数基础模型中原本缺乏该词的模型。在基础模型已经使用该词的情况下，两个已发布的RL-Zero配方（OLMo-3、MiMo）都会使其成倍增加，而来自同一基础模型的无KL正则化的GRPO则不会；OLMo-3的配方会磨损其形式，使其变得更加……

    arXiv:2610.30882v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is credited with teaching reasoning models to write {wait} before reconsidering a step, yet large RL accuracy gains do not require the word. This view can confuse what RL discovers with what a model inherits from its training data. We trace the word through open models' training histories, reading it with label-free indices of frequency, position, form and usage and an injected-error test, all computed from sampled outputs. We find that zero-RL almost never coins the marker: on bases that lack the word, RL raises MATH-500 accuracy substantially while {wait} stays near absent. It arrives with training data: mid-training in OLMo-3, and fine-tuning on reasoning traces installs it in most models whose base lacks it. Where the base already uses it, both released RL-Zero recipes (OLMo-3, MiMo) multiply it, whereas KL-free GRPO from the same base does not; OLMo-3's recipe wears down its form and makes it a more sen
    
[^114]: 一个教师，多个视角：降低在线策略自蒸馏中的参考敏感性

    One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.30880](https://arxiv.org/abs/2610.30880)

    本文发现在线策略自蒸馏中单一完整参考条件会导致教师分布对参考的选择高度敏感（参考诱导的目标不稳定性），并提出蒙特卡洛特权蒸馏方法，通过利用参考的多个视角来降低这种参考敏感性。

    

    在线策略自蒸馏（OPSD）通过在模型自身生成的轨迹上进行训练，并利用一个以参考解答为条件的特权副本提供密集的token级监督，从而提升推理能力。现有方法通常使用以完整参考为条件的单一教师分布。我们发现了这种形式的一个局限性：参考只是推理过程的一种实现方式。尽管它提供了关于答案的证据，但它同时编码了特定的顺序、措辞和求解路径。因此，以完整参考为条件可能会产生过度依赖于学生模型在推理时无法获得的信息的正确目标。我们证明，在保持问题和学生状态不变的情况下，用一个有效参考替换另一个有效参考，会显著改变教师分布，我们将这种现象称为参考诱导的目标不稳定性。为了解决这种不稳定性，我们引入了蒙特卡洛特权蒸馏（

    arXiv:2610.30880v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) improves reasoning by training a model on its own trajectories with dense token-level supervision from a privileged copy conditioned on a reference solution. Existing methods typically use a single teacher distribution conditioned on the complete reference. We identify a limitation of this formulation: a reference is only one realization of the reasoning process. Although it provides evidence about the answer, it also encodes an ordering, phrasing, and solution path. Full-reference conditioning can therefore produce correct targets that depend excessively on information unavailable to the student at inference. We show that replacing one valid reference with another, while holding the problem and student state fixed, can substantially alter the teacher distribution, a phenomenon we call reference-induced target instability. To address this instability, we introduce Monte Carlo Privileged Distillation (
    
[^115]: MOER：专家前缀引导下结合选择性回滚的混合策略优化

    MOER: Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback

    [https://arxiv.org/abs/2610.30873](https://arxiv.org/abs/2610.30873)

    MOER提出一种混合策略优化框架，通过逐步衰减的专家思维链前缀引导并配合选择性回滚机制保证离策略轨迹正确性，协调了专家引导与模型自我探索。

    

    混合策略优化将离策略专家数据引入强化学习（RL）循环中，以突破在策略强化学习的能力边界。然而，现有方法要么提供静态的完整轨迹引导，抑制了模型的自我探索；要么采用缺乏正确性保证的专家前缀引导，导致性能欠佳甚至训练崩溃。本文提出MOER，一个能够协调专家引导与自我探索的混合策略优化框架。MOER将一个衰减的专家思维链前缀注入提示中，让模型完成剩余的推理，并通过逐步退火前缀长度来鼓励自我探索。选择性回滚机制用原始的完整专家轨迹替换失败的补全，保证每条离策略轨迹的正确性。此外，一个协调的混合策略目标使用GRPO对在策略token进行优化（原文摘要在此处截断）。

    arXiv:2610.30873v1 Announce Type: new  Abstract: Mix-policy optimization incorporates off-policy expert data into the reinforcement learning (RL) loop to transcend the capability boundary of on-policy RL. However, existing methods either provide static full-trace guidance that suppresses self-exploration, or adopt expert prefix guidance without correctness guarantees, leading to suboptimal performance or even training collapse. In this paper, we propose MOER, a mix-policy optimization framework that harmonizes expert guidance with self-exploration. MOER injects a decaying expert chain-of-thought prefix into the prompt and lets the model complete the remaining reasoning, with the prefix length progressively annealed to encourage self-exploration. A selective rollback mechanism replaces failed completions with the original full expert trace, guaranteeing the correctness of every off-policy trajectory. Furthermore, a harmonized mix-policy objective optimizes on-policy tokens with a GRPO l
    
[^116]: 焦耳到GDP：AI系统每单位能源能产生多少经济价值？

    Joules-to-GDP: How Much Economic Value Do AI Systems Produce per Unit of Energy?

    [https://arxiv.org/abs/2610.30858](https://arxiv.org/abs/2610.30858)

    该论文提出“每焦耳净价值（NVJ）”和“每焦耳时间节省（TSJ）”两个新指标来衡量AI系统每消耗一单位能源所创造的经济价值，并发现AI表现最出色、创造价值或节省时间最多的职业领域并不一定带来最高的每焦耳经济回报。

    

    AI系统可以在产生更多具有经济价值的工作的同时，其消耗的能源所带来的价值却更低。因此我们提出这样的问题：AI系统每消耗一单位能源能创造多少经济价值？我们以两种方式对经济价值进行形式化：（1）净美元价值，即产出的美元价值减去AI系统的执行成本；（2）时间节省，即专家的完成时间减去AI系统的完成时间。将每项除以实测的端到端能耗，即可得到每焦耳净价值（NVJ）和每焦耳时间节省（TSJ），分别以美元/兆焦（USD/MJ）和小时/兆焦为单位报告。我们研究了涵盖44个职业的220个公开GDPval任务，在本地部署五个开放权重模型，并评估了四种加速器类型。首先，AI系统成功率最高、创造美元价值最多或节省时间最多的职业群体，并不一定能提供每焦耳最高的经济回报。例如，在测试的任务中，工程类工作（摘要在此处截断）

    arXiv:2610.30858v1 Announce Type: new  Abstract: An AI system can produce more economically valuable work while delivering less value for the energy it consumes. We therefore ask: How much economic value does an AI system create per unit of energy consumed? We formalize economic value in two ways: (1) net dollar value, the output's dollar value minus AI-system execution cost, and (2) time saving, the expert's completion time minus the AI system's completion time. Dividing each by measured end-to-end energy yields Net Value per Joule (NVJ) and Time Saving per Joule (TSJ), reported in USD/MJ and hours/MJ. We study 220 public GDPval tasks across 44 occupations, serving five open-weight models locally and evaluating four accelerator types. First, the occupation groups where AI systems succeed most often, deliver the greatest dollar value, or save the most time do not necessarily provide the greatest economic return per joule. For example, among the tested tasks, engineering work delivers a
    
[^117]: 训练混合单调神经网络以实现无需集合传播的验证

    Training Mixed-Monotone Neural Networks for Verification Without Set Propagation

    [https://arxiv.org/abs/2610.30853](https://arxiv.org/abs/2610.30853)

    提出一种将混合单调函数实现为两个单调子网络之差的新型神经网络架构，把紧凑输出包络的计算从复杂的集合传播简化为两次简单前向传播，且因包络大小是权重的闭式函数，可在训练中直接最小化它。

    

    大多数针对神经网络的形式化验证方法都会保守地包络神经网络的输出集合。然而，计算一个紧凑的输出包络在通常情况下是计算困难的。我们通过一种新颖的神经网络架构绕过了这一挑战，该架构将混合单调函数实现为两个单调子网络之差。借助混合单调系统的理论，这将紧凑输出包络的计算从复杂的集合传播简化为对输入区间两个相反角点的两次简单前向传播。此外，由于所得包络的大小是神经网络权重的闭式函数，我们可以在训练过程中直接将其最小化。我们证明了任何此类包络大小的一个下界，该下界由推广到多输入情形的所表示函数的全变差给出，这使得将包络的紧凑性与最优情况进行比较成为可能。

    arXiv:2610.30853v1 Announce Type: new  Abstract: Most formal verification approaches for neural networks conservatively enclose the output set of the neural network. However, computing a tight output enclosure is in general computationally hard. We sidestep this challenge with a novel neural network architecture that realizes mixed-monotone functions as the difference of monotone subnetworks. By utilizing the theory on mixed-monotone systems, this reduces the computation of a tight output enclosure from complicated set propagation to two simple forward passes of opposite corners of an input interval. Moreover, because the size of the resulting enclosure is a closed-form function of the weights of the neural network, we can directly minimize it during training. We prove a lower bound on the size of any such enclosure, given by the total variation of the represented function generalized to several inputs, which makes it possible to compare the tightness of the enclosure against the optim
    
[^118]: 源自权重空间的分岔：JEPAWG 在参数化动力系统上的应用

    Bifurcations from Weight Space: JEPAWG on Parametric Dynamical Systems

    [https://arxiv.org/abs/2610.30847](https://arxiv.org/abs/2610.30847)

    该论文以分岔结构已知的参数化动力系统为真值基准，构建以物理参数索引的模型动物园，验证了 JEPAWG 超网络能够从训练权重的潜在几何中定位动力学发生质变的分岔位置。

    

    训练后的网络权重日益被视为数据，但关于一群模型应当编码什么内容，鲜有真值（ground truth）可供参照，这阻碍了权重空间方法的系统性验证。这使参数化动力系统成为一个天然的试验平台：在研究充分的动力系统族中，动力学会在参数空间的某些子流形处发生质变，而这些子流形可以通过解析或数值方法获知。在参数网格的每个点上微调一个预测网络，便得到一个以物理参数为索引的模型动物园（model zoo），权重空间诊断方法可据此加以检验。我们为四个动力系统构建了这样的模型动物园，并用 JEPAWG——一种近期提出的、围绕学习到的参数-权重表示构建的联合嵌入超网络——对其进行分析。该方法最初是为格点场论的神经采样器而提出的，其潜在几何结构成功定位了相变并估计了二维 Ising 临界指数。（原文摘要在此处截断）

    arXiv:2610.30847v1 Announce Type: new  Abstract: Trained network weights are increasingly treated as data, but there is rarely ground truth about what a population of models should encode, which hampers systematic validation of weight-space methods. This makes parametric dynamical systems a natural test-bed: the dynamics change qualitatively at submanifolds of parameter space that are, in well-studied families, known analytically or numerically. Fine-tuning a forecasting network at each point of a parameter grid yields a model zoo indexed by a physical parameter, against which weight-space diagnostics can be checked.   We build such model zoos for four dynamical systems and analyse them with JEPAWG, a recently proposed joint-embedding hypernetwork built around a learned parameter--weight representation. It was first introduced for neural samplers for lattice field theories, where its latent geometry located the phase transition and estimated the 2D Ising critical exponent. Applied to d
    
[^119]: 反向KL策略蒸馏：统计保证与极限

    Reverse-KL Policy Distillation: Statistical Guarantees and Limits

    [https://arxiv.org/abs/2610.30844](https://arxiv.org/abs/2610.30844)

    该研究首次从理论上分析反向KL策略蒸馏的统计保证，证明当教师提供完整动作分布时在线蒸馏具有极小极大最优的样本复杂度，而仅观察采样动作概率或采用离线蒸馏则可能需要显著更多的数据。

    

    反向Kullback-Leibler（KL）蒸馏在训练语言模型方面已展现出经验上的潜力，但其统计保证和极限仍未被充分理解。特别是，两种设计选择的影响尚不明确：获取教师的完整动作分布是否比仅观察其赋予采样动作的概率更具优势？训练轨迹是否必须来自学生自身（在线蒸馏），还是教师生成的轨迹（离线蒸馏）就已足够？我们的工作为理解这些问题背后的理论迈出了第一步。我们研究了在两种反馈模型下在线反向KL蒸馏的样本复杂度：教师要么揭示其完整的动作分布，要么仅揭示学生所生成动作的概率。我们为前者建立了极小极大界，并证明后者可能需要多得多的数据。相比之下，离线反（摘要在此处截断）

    arXiv:2610.30844v1 Announce Type: new  Abstract: Reverse Kullback-Leibler (KL) distillation has shown empirical promise for training language models, but its statistical guarantees and limits remain poorly understood. In particular, the effects of two design choices remain unclear: Does access to the teacher's full action distribution provide an advantage over observing only the probability it assigns to the sampled action? Must training trajectories come from the student (online distillation), or can teacher-generated trajectories (offline distillation) suffice? Our work takes a first step toward understanding the theory underlying these questions.   We study the sample complexity of online reverse-KL distillation under two feedback models: the teacher reveals either its full action distribution or only the probability of the student-generated action. We establish minimax bounds for the former and show that substantially more data may be required for the latter. In contrast, offline r
    
[^120]: Lean Finder V2：为 Mathlib 提供可消除用户查询歧义的语义搜索

    Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries

    [https://arxiv.org/abs/2610.30843](https://arxiv.org/abs/2610.30843)

    Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。

    

    我们提出了 Lean Finder V2，这是一个面向 Mathlib 的语义搜索框架，通过互补的模型级对抗训练和输入级查询重写来消除用户查询的歧义。人类数学家和编码智能体很少通过提供详细上下文来搜索 Mathlib。在实践中，他们查询的是简短的性质、记号、别名或假设，而将大部分预期的数学内容隐含在查询之外。相比之下，现有的 Mathlib 搜索引擎不切实际地假设查询带有丰富的上下文。由于缺乏与真实标准（ground-truth）Mathlib 陈述配对的真实世界短查询，这种歧义问题变得更加难以处理。我们认为这本质上是一个数据问题。通过对抗训练，我们扩大了带标注的短且含歧义查询的生成规模，并在模型级微调和输入级查询重写两个层面上学习消除用户查询的歧义。在人工策划的 MathlibQR 基准测试上，我们的 Lean Finder V2 创造了新的最佳纪录（原文在此处截断）。

    arXiv:2610.30843v1 Announce Type: new  Abstract: We introduce Lean Finder V2, a semantic search framework for Mathlib that disambiguates user queries through complementary model-level adversarial training and input-level query rewriting. Human mathematicians and coding agents rarely search Mathlib by providing detailed context. In practice, they query short properties, notation, aliases, or hypotheses, leaving much of the intended mathematics implicit. In contrast, existing Mathlib search engines unrealistically assume context-rich queries. The scarcity of real-world short queries paired with ground-truth Mathlib statements makes this ambiguity even more intractable. We argue that this is essentially a data problem. Through adversarial training, we scale up the generation of labeled short and ambiguous queries, and learn to disambiguate user queries at both model-level fine-tuning and input-level query rewriting. On the human-curated MathlibQR benchmark, our Lean Finder V2 sets a new s
    
[^121]: 半线性高斯贝叶斯网络：通过惩罚B样条增强节点依赖关系的灵活性

    Semi-Linear Gaussian Bayesian Networks: Enhancing Flexibility in Node Dependencies via penalized B-Splines

    [https://arxiv.org/abs/2610.30838](https://arxiv.org/abs/2610.30838)

    本文提出了半线性高斯贝叶斯网络（SLGBN），通过带惩罚B样条的广义可加模型估计非线性条件均值函数，在保持高斯条件分布的同时捕捉非线性依赖关系，弥合了线性高斯模型灵活性不足与非参数方法计算成本高之间的空白。

    

    用于连续变量的贝叶斯网络通常依赖于严格的假设：线性高斯模型缺乏捕捉非线性依赖关系的灵活性，而非参数方法虽然提供了更大的灵活性，但计算成本高昂。这在建模变量近似服从高斯分布但呈现非线性关系的场景时造成了一个空白。我们提出了半线性高斯贝叶斯网络（SLGBN），这是一类新颖的连续贝叶斯网络，旨在弥合这一空白，它在保持高斯条件分布的同时引入了非线性条件均值函数。所提出的框架提供了一个统一的概率框架，将线性高斯和非线性高斯条件模型结合在同一个贝叶斯网络中。非线性条件均值函数使用带惩罚B样条的广义可加模型（GAMs）进行估计。我们给出了两种类型模型的参数估计方法。

    arXiv:2610.30838v1 Announce Type: cross  Abstract: Bayesian networks for continuous variables often rely on restrictive assumptions: linear Gaussian models lack flexibility to capture nonlinear dependencies, while non-parametric approaches provide greater flexibility at a high computational cost. This creates a gap for modeling scenarios where variables are approximately Gaussian but exhibit nonlinear relationships. We introduce Semi-Linear Gaussian Bayesian Networks (SLGBN), a novel class of continuous Bayesian networks designed to bridge this gap, preserving Gaussian conditional distributions while introducing nonlinear conditional mean functions. The proposed framework provides a unified probabilistic framework combining linear Gaussian and nonlinear Gaussian conditional models within the same Bayesian network Nonlinear conditional mean functions are estimated using generalized additive models (GAMs) with penalized B-splines. We present parameter estimation procedures for both types
    
[^122]: 面向Transformer内部异构资源分配的带残差上下文的神经谱容量

    Neural Spectral Capacity with Residual Context for Heterogeneous Resource Allocation within Transformers

    [https://arxiv.org/abs/2610.30823](https://arxiv.org/abs/2610.30823)

    本文提出残差感知神经谱容量ResNSC及其动态规划分配器ResNSC-DP，通过在容量计算中引入残差历史统计信息，在无需数据或实例化权重的情况下，解决了原NSC方法独立评分、无法区分层配置重排序的问题，并能在资源预算下为Transformer联合分配残差和前馈宽度。

    

    当训练成本限制了实验次数时，从架构规格预测相对训练后性能可以为设计提供指导。神经谱容量提供了一种这样的启发式方法，但它独立地对各投影进行评分，并且对同一层配置的兼容重排序赋予相同的分数。我们为Transformer引入了残差感知神经谱容量，使用从规格推导的累积残差更新统计参考，来确定每个投影容量计算中的输入和残差状态统计量，同时保持无需数据或实例化权重的可加评估特性。当残差宽度变化时，较早的选择会影响后续的评分，因此我们在NSC的分配器（NSC-DP）基础上增加了超出其层-预算状态的残差历史信息。ResNSC-DP在资源预算约束下联合分配残差宽度和前馈宽度，并对最佳数值可评估的候选架构提供了认证边界。

    arXiv:2610.30823v1 Announce Type: new  Abstract: Predicting relative trained performance from architectural specifications can guide design when training costs limit experimentation. Neural Spectral Capacity (NSC) provides such a heuristic, but scores projections independently and assigns identical scores to compatible reorderings of the same layer configurations. We introduce Residual-Aware Neural Spectral Capacity (ResNSC) for Transformers, using a specification-derived statistical reference for accumulated residual updates to determine the input and residual-state statistics in each projection's capacity calculation, while retaining additive evaluation without data or instantiated weights. When residual widths vary, earlier choices affect later scores, so we augment NSC's allocator (NSC-DP) with residual history beyond its layer-and-budget state. ResNSC-DP jointly allocates residual and feed-forward widths under a resource budget, with certified bounds on the best numerically evalua
    
[^123]: xVGAE：一种面向可交换图的层次化变分图自编码器

    xVGAE: A Hierarchical Variational Graph Autoencoder for Exchangeable Graphs

    [https://arxiv.org/abs/2610.30815](https://arxiv.org/abs/2610.30815)

    本文提出 xVGAE，一种受 Aldous-Hoover 定理启发的层次化变分图自编码器，在生成质量与最先进模型相当的同时，将生成速度提升了几个数量级。

    

    图在许多应用中扮演着至关重要的角色，涵盖从药物发现到社交网络建模再到天文学等领域。近年来，图生成模型取得了长足进步，但我们发现，在最先进的模型中，存在一些相对简单的场景，这些模型在标准计算预算内难以收敛，或表现出极其缓慢的生成速度。我们提出了一种生成架构，其生成质量可与最先进的模型相媲美，且生成速度提高了几个数量级。我们的方法受 Aldous-Hoover 定理的启发，这是一个经典的表示定理，它刻画了任何对节点索引置换不变（即可交换）的图的概率分布。该定理表明，三个要素即可满足需求：一个图级潜变量、一组可交换的节点级潜变量，以及一个将图级和节点级变量进行映射的所谓 graphon 函数……

    arXiv:2610.30815v1 Announce Type: cross  Abstract: Graphs play a crucial role in many applications, ranging from drug discovery to social network modeling and astronomy. Generative models for graphs have substantially advanced in recent years, but we identify relatively simple scenarios where state-of-the-art models struggle to converge within a standard compute budget, or exhibit prohibitively slow generation times. We propose a generative architecture that achieves quality competitive with state-of-the-art models and is orders of magnitude faster at generation. Our approach is inspired by the Aldous-Hoover theorem, a classic representation theorem that characterizes any probability distribution of graphs that is invariant to permutations of node indices or exchangeable. This theorem establishes that three ingredients suffice: a graph-level latent variable, a set of exchangeable node-specific latent variables, and a so-called graphon function that maps graph- and node-level variables 
    
[^124]: 《超越感知的边界：AI评估与监督局限性的划分理论》

    What Lies Beyond Perception: A Partition Theory of the Limits of AI Evaluation and Oversight

    [https://arxiv.org/abs/2610.30803](https://arxiv.org/abs/2610.30803)

    本文提出AI评估的划分理论，证明当被评估模型的感知划分比评估者更精细时，通过评估者所能评定的任何测试都最多只能认证模型辨别力的下限而非上限，从而揭示了AI评估与监督存在的根本性局限。

    

    更换评判者可以改变基准测试的得分，而模型也可以通过训练学会隐藏自身能力。这两者都使评估者本身受到质疑：它究竟可以认证什么？我们从一个原则出发，建立了一套评估的划分理论：判断能力受感知能力所限。智能体的“辨别力”是指将其输入划分为它无法相互区分的类别所形成的一种划分。辨别力是能力的必要条件但非充分条件，因此对辨别力的界限直接关系到评估能否证明某个模型不具备某种能力。我们证明了一个单侧界限：只要每个输入都存在一个可接受的回应，那么当应试者的划分比评估者的划分更精细时，该应试者就具备通过评估者所能评定的所有测试所需的辨别力。我们定义了“回声测试”，该测试接受任何被评估者感知为与输入相同的回应，并证明它对于评估者所能评定的测试是完备的。因此，通过测试最多只能认证辨别力的下限，而永远无法认证其上限……

    arXiv:2610.30803v1 Announce Type: cross  Abstract: Replacing the judge can move a benchmark score, and models can be trained to hide capabilities. Both put the evaluator in question: what can it certify? We develop a partition theory of evaluation from one principle: judgement is bounded by perception. An agent's discrimination is the partition of its inputs into classes it cannot tell apart. Discrimination is necessary for capability but not sufficient, so bounds on it bear directly on whether evaluation can show that a model lacks an ability. We prove a one-sided bound. An examinee whose partition refines its evaluator's has the discrimination to pass every test the evaluator can grade, provided each input has an acceptable reply. We define the echo test, which accepts any reply the evaluator perceives as identical to the input, and show that it is complete for the tests the evaluator can grade. Passing therefore certifies at most a floor on discrimination, never a ceiling, and only 
    
[^125]: 谱可预测性就足够了吗？在分布偏移下测试Ω匹配的合成预训练方法

    Is spectral predictability enough? Testing {\Omega}-matching for synthetic pretraining under distribution shift

    [https://arxiv.org/abs/2610.30802](https://arxiv.org/abs/2610.30802)

    论文通过多架构实验发现，仅匹配谱可预测性Ω不足以指导分布偏移下的合成预训练——尽管在理想合成条件下Ω匹配预训练可超越oracle基线，但在真实世界基准上，不同类型的Ω匹配合成数据（正弦加噪声与混沌ODE轨迹）会产生截然相反甚至负面的迁移效果。

    

    在分布偏移下进行合成预训练需要决定使用什么样的合成数据。一个自然的准则是使源数据的谱可预测性Ω与目标数据相匹配，但Ω是否足够？我们在三种架构（LSTM、PatchTST、TCN）和三个随机种子下测试了这一假设，我们称之为可预测性感知预训练。在理想的合成到合成条件下，Ω匹配的预训练甚至可以超越oracle基线（LSTM的MSE降低+46.8%），尽管这种收益已经呈现出架构依赖性。然而，在真实世界的ETTm1基准上，这种关系并不成立：在匹配Ω的条件下，正弦加噪声的预训练仅带来微小的平均改进（三种架构平均+2.3%，个别通道最高+65.5%），而Ω匹配的混沌ODE轨迹则系统性地降低性能（-13.2%）。这两种数据源在迁移方向上……（原文在此截断）

    arXiv:2610.30802v1 Announce Type: new  Abstract: Synthetic pretraining under distribution shift requires deciding what synthetic data to use. A natural criterion is to match the spectral predictability $\Omega$ of the source to the target, but is $\Omega$ sufficient? We test this hypothesis, which we call predictability-aware pretraining, across three architectures (LSTM, PatchTST, TCN) and three seeds. Under ideal synthetic-to-synthetic conditions, $\Omega$-matched pretraining can outperform even an oracle baseline (+46.8% MSE reduction, LSTM), though the benefit is already architecture-dependent. On the real-world ETTm1 benchmark, however, this relationship does not hold: sine-plus-noise pretraining at matched $\Omega$ yields only a marginal mean improvement +2.3% averaged over all three architectures, up to +65.5% on individual channels), whereas $\Omega$ matched chaotic ODE trajectories systematically degrade performance -13.2%). The two sources agree on the direction of transfer f
    
[^126]: 一种用于具有（聚合）老虎机反馈的对抗性线性马尔可夫决策过程（MDP）的计算高效且速率最优的算法

    A Computationally Efficient and Rate-Optimal Algorithm for Adversarial Linear MDPs with (Aggregate) Bandit Feedback

    [https://arxiv.org/abs/2610.30801](https://arxiv.org/abs/2610.30801)

    本文首次提出多项式时间算法，通过将对抗性线性MDP归约为动作集未知的对抗性线性老虎机并构建近似分离预言机，实现了速率最优且计算高效的 $\sqrt{K}$ 遗憾界。

    

    我们研究具有未知转移、对抗性损失和老虎机反馈的回合制线性马尔可夫决策过程（MDP）。针对该设置的现有最优结果（Liu et al., 2024）要么通过计算效率低下的算法实现 $\sqrt{K}$ 遗憾，要么实现次优的 $K^{3/4}$ 遗憾，其中 $K$ 为回合数。我们的工作首次给出了以多项式时间算法实现 $\sqrt{K}$ 遗憾的结果。我们的方法将对抗性线性MDP归约为对抗性线性老虎机问题，其后者的动作集由各策略的期望特征向量组成。核心挑战在于该动作集是未知的，只能通过探索进行估计。我们通过为该动作集构建一个近似分离预言机来应对这一挑战，该预言机要么根据查询的特征生成可执行的策略，要么细化该动作集的外部近似。该方法借鉴了 van Erven et al. (2025) 的归约方法……

    arXiv:2610.30801v1 Announce Type: new  Abstract: We study episodic linear Markov decision processes with unknown transitions, adversarial losses, and bandit feedback. The state-of-the-art result for this setting (Liu et al., 2024) either achieves $\sqrt{K}$ regret with a computationally inefficient algorithm or achieves suboptimal $K^{3/4}$ regret, where $K$ is the number of episodes. Our work gives the first $\sqrt{K}$ regret with a polynomial-time algorithm. Our method reduces adversarial linear MDPs to adversarial linear bandits whose action set consists of policies' expected feature vectors. The central challenge is that this action set is unknown and can be estimated only through exploration. We address this challenge by constructing an approximate separation oracle for the action set that either generates executable policies from queried features, or refines an outer approximation of the action set. This approach draws inspiration from the reduction of van Erven et al. (2025) for
    
[^127]: 通过Rowhammer攻击放大LLM推理的延迟与成本

    Amplifying Latency and Cost of LLM Inference via Rowhammer Attack

    [https://arxiv.org/abs/2610.30798](https://arxiv.org/abs/2610.30798)

    本文提出LLVMhammer——首个针对Triton编译流水线的Rowhammer攻击，通过诱导性能关键优化阶段的位翻转，生成功能不变但性能次优的GPU内核，并借助内核磁盘缓存机制造成LLM推理的持久性变慢，且该攻击与GPU厂商无关。

    

    大语言模型（LLM）通过诸如Triton等即时（JIT）编译框架生成的优化GPU内核来实现高推理效率。Triton通过基于MLIR/LLVM的流水线编译GPU内核。我们研究了CPU DRAM中由Rowhammer引起的位翻转是否能够在保持输出正确性的同时降低基于GPU的LLM推理性能。我们提出了LLVMhammer，这是首个通过在Triton编译流水线中诱导位翻转来导致LLM推理变慢的Rowhammer攻击。在性能关键优化过程中的一次位翻转会使Triton生成功能完全相同但次优的GPU内核。由于Triton将生成的内核缓存在磁盘上，一次成功的位翻转会导致后续推理执行的持续性能下降。Triton与目标无关的编译阶段在各GPU厂商之间共享，这使得我们的攻击不受GPU厂商限制。为了识别可利用的位翻转位置……（原文截断）

    arXiv:2610.30798v1 Announce Type: cross  Abstract: Large language models (LLMs) achieve high inference efficiency using optimized GPU kernels generated by just-in-time (JIT) compilation frameworks such as Triton. Triton compiles GPU kernels through an MLIR/LLVM-based pipeline. We investigate whether a Rowhammer-induced bit-flip in the CPU's DRAM can degrade GPU-based LLM inference performance while preserving output correctness. We present LLVMhammer, the first Rowhammer attack that causes LLM inference slowdown by inducing a bit-flip in Triton's compilation pipeline. A bit-flip in a performance-critical optimization pass causes Triton to generate a functionally identical but suboptimal GPU kernel. Since Triton caches the generated kernel on disk, a single successful bit-flip causes persistent slowdown across subsequent inference executions. Triton's target-independent stages are shared across GPU vendors, which makes our attack GPU vendor-agnostic. To identify exploitable bit-flip loc
    
[^128]: GIFT：面向蒸馏增强推理强化学习的梯度感知过滤

    GIFT: Gradient Informed Filtering for Distillation Augmented Reasoning RL

    [https://arxiv.org/abs/2610.30796](https://arxiv.org/abs/2610.30796)

    GIFT通过在token级别对蒸馏梯度进行几何分解——将平行于RL梯度的分量作为安全过滤器以排除冲突更新、将正交分量作为新信息来源——为蒸馏与强化学习之间的目标冲突提供了一个有原则的解决框架。

    

    arXiv:2610.30796v1 公告类型：new 摘要：训练大语言模型执行复杂推理任务依赖两种互补的信号：用于保证正确性的稀疏的、基于结果的强化学习（RL），以及用于提供指导的密集的、token级别的蒸馏。然而，监督信号上的互补性并不能保证优化过程中的兼容性；以模仿为中心的蒸馏可能会与以结果为导向的RL目标发生冲突，而现有的混合方法缺乏一个有原则的解决方案。我们提出了GIFT（Gradient-Informed FilTering for Distillation-Augmented Reasoning RL，面向蒸馏增强推理强化学习的梯度感知过滤），这是一种新颖的方法，直接解决了“应应用哪些教师更新”这一分配问题。GIFT在token级别进行几何分析，将蒸馏梯度分解为与RL梯度平行的分量——该分量衡量一阶冲突并提供一个安全过滤器——以及与RL梯度正交的分量——该分量量化新颖的方向…

    arXiv:2610.30796v1 Announce Type: new  Abstract: Training large language models for complex reasoning tasks relies on two complementary signals: sparse, outcome-grounded reinforcement learning (RL) for correctness, and dense, token-level distillation for guidance. However, complementarity in supervision does not guarantee compatibility in optimization; imitation-focused distillation can conflict with the outcome-driven RL objective, and existing hybrids lack a principled resolution. We propose \textbf{GIFT} (\textbf{G}radient-\textbf{I}nformed \textbf{F}il\textbf{T}ering for Distillation-Augmented Reasoning RL), a novel method that directly addresses the allocation problem of which teacher updates to apply. GIFT performs a token-level geometric analysis, decomposing the distillation gradient into a component \textbf{parallel} to the RL gradient, which measures first-order conflict and provides a \textbf{safety} filter, and a component \textbf{orthogonal} to it, which quantifies novel d
    
[^129]: Trace2World：从真实交互轨迹重建应用世界以实现可扩展的GUI智能体训练

    Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training

    [https://arxiv.org/abs/2610.30792](https://arxiv.org/abs/2610.30792)

    Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。

    

    扩展GUI强化学习需要能够保留真实应用行为的环境。然而，将行为推断与编码相耦合，可能会把部分轨迹转变成状态转换错误的可运行模拟（mock）。我们提出Trace2World，它将全局行为重建与具体实现分离开来。该方法首先将多条轨迹调和为一个共享的、与来源关联的应用模型，然后编译出可执行的环境。通过独立回放来源工作流来定位不一致之处，从而进行有边界的实现修复。经过验证的世界进而可以在来源支持的行为范围内，支持面向学习者自适应的任务与环境增强。Trace2World从1,570条轨迹中构建了50个环境。在十个应用的200个工作流上，91.1%的工作流被端到端复现，而最强的任务局部匹配基线仅为59.8%。基于来源的验证在真实设备上保留了94.0%的模拟训练增益，而自我验证下仅为54.7%。学习者引导（摘要在此处截断）

    arXiv:2610.30792v1 Announce Type: new  Abstract: Scaling GUI reinforcement learning requires environments that preserve real-app behavior. Yet coupling behavior inference with coding can turn partial traces into runnable mocks with incorrect state transitions. We introduce \textbf{Trace2World}, which separates global behavior reconstruction from implementation. It reconciles traces into a shared, source-linked app model, then compiles an executable environment. Independent replay of source workflows localizes mismatches for bounded implementation repair. Validated worlds then support learner-adaptive task and environment augmentation within source-supported behavior. From 1,570 traces, Trace2World constructs 50 environments. On 200 workflows across ten apps, 91.1% are reproduced end to end, versus 59.8% for the strongest matched task-local baseline. Source-grounded validation retains 94.0% of simulated training gains on real devices, versus 54.7% under self-validation. Learner guidance
    
[^130]: Gandalf：闪存不应消亡！延长LLM服务中KV缓存卸载的闪存存储寿命

    Gandalf: Flash Shall Not Die! Increasing Flash Storage Lifetime for KV Cache Offloading in LLM Serving

    [https://arxiv.org/abs/2610.30791](https://arxiv.org/abs/2610.30791)

    提出Gandalf系统，通过结合访问模式等信息智能门控KV缓存写入闪存，在不降低KV缓存命中率的前提下，显著延长LLM服务中作为KV缓存卸载介质的SSD使用寿命。

    

    KV缓存降低了大语言模型（LLM）的服务成本，但不断增长的KV缓存占用空间日益超出GPU和主机内存的容量。因此，人们使用SSD以其高I/O带宽来扩展KV缓存的存储容量。然而，在生产级LLM工作负载（如多轮对话和LLM智能体）中，KV缓存写入总量显著超过SSD容量，使SSD寿命从数年缩短至数月。为了延长SSD寿命，一个自然的解决方案是通过历史访问模式来门控KV缓存写入，从而减少KV缓存写入量。然而，这会显著降低KV缓存命中率（并增加LLM服务成本），因为来自不同注意力类型的KV缓存可能具有相同的历史访问模式但不同的未来访问模式。我们的见解是，要在不损害KV缓存命中率的情况下减少KV缓存写入，不仅需要考虑访问模式，还需考虑……

    arXiv:2610.30791v1 Announce Type: new  Abstract: KV caching reduces the serving cost of large language models (LLMs), but the growing KV cache footprint is increasingly exceeds GPU and host-memory capacity. Therefore, people use SSD to expand the KV cache storage capacity with high I/O bandwidth. However, in production LLM workload, such as multi-round conversation and LLM agents, the total amount of KV cache writes significantly exceeds SSD capacity, shortening SSD lifetime from years to months. To improve SSD lifetime, a natural solution is to reduce the KV cache writes by gating the KV cache writes via historical access patterns. However, that significantly lowers the KV cache hit rate (and increase the LLM serving cost) since KV caches from different attention types might have identical historical access pattern but different future access patterns. Our insight is that, to reduce KV cache writes without hurting KV cache hit rate, one need to not only consider the access pattern, bu
    
[^131]: 一个预训练决策模型足以实现免训练的通才图异常检测

    One Pretrained Decision Model Is Enough for Training-Free Generalist Graph Anomaly Detection

    [https://arxiv.org/abs/2610.30777](https://arxiv.org/abs/2610.30777)

    Jev-GAD是一个免训练的通才图异常检测框架，通过将结构、属性和邻域等异构数值证据转化为图相对异常百分位数，并利用单个冻结的预训练语言模型进行基于证据的异常评估，实现了跨图的可泛化异常检测。

    

    图异常检测（GAD）旨在识别图结构数据中的不规则实体，然而异构的异常模式以及不同图之间的分布差异对可泛化的检测构成了挑战。现有的免训练方法虽然避免了特定任务的优化，但主要依赖于预定义的数值评分规则，使得对异构异常证据的语义解释在很大程度上尚未被探索。在本文中，我们提出了Jev-GAD，一个连接异构数值证据与预训练决策机制的免训练通才图异常检测框架。为了应对异构的异常模式，我们构建了互补的结构、属性和邻域证据，并将异构统计量转换为图相对异常百分位数。为了超越预定义的数值评分规则，我们利用单个冻结的预训练语言模型进行基于证据的异常评估。

    arXiv:2610.30777v1 Announce Type: new  Abstract: Graph anomaly detection (GAD) aims to identify irregular entities in graph-structured data, yet heterogeneous anomaly patterns and distributional differences across graphs challenge generalizable detection. Existing training-free approaches avoid task-specific optimization but primarily rely on predefined numerical scoring rules, leaving the semantic interpretation of heterogeneous anomaly evidence largely unexplored. In this paper, we propose Jev-GAD, a training-free generalist graph anomaly detection framework that bridges heterogeneous numerical evidence and pretrained decision mechanisms. To address heterogeneous anomaly patterns, we construct complementary structural, attribute, and neighborhood evidence and transform heterogeneous statistics into graph-relative anomaly percentiles. To move beyond predefined numerical scoring rules, we leverage a single frozen pretrained language model for evidence-conditioned anomaly assessment. To
    
[^132]: 动态多无人机边缘网络中自适应的截止时间与能耗感知缓存算法

    Adaptive Deadline- and Energy-Aware Caching Algorithms in Dynamic Multi-UAV Edge Networks

    [https://arxiv.org/abs/2610.30768](https://arxiv.org/abs/2610.30768)

    论文针对多无人机边缘网络提出了一个联合考虑截止时间错过、通信能耗与机间能耗不均衡的动态缓存放置与替换问题，并开发了可解释的分层规则算法DE-UBAC来实现截止时间与能耗感知的缓存管理。

    

    基于无人机（UAV）的内容缓存成为在地面基础设施不可用或受损时维持内容分发的解决方案之一。然而，现有的缓存方法通常主要关注缓存命中率或平均交付时延，未能联合处理空间异构且随时间变化的需求、内容特定的截止时间要求，以及各无人机之间不均衡的通信能耗。为应对这些挑战，我们构建了一个动态的截止时间与能耗感知的缓存放置与替换问题，该问题联合考虑了截止时间错过、无人机侧通信能耗以及无人机之间的能耗不均衡。为此，我们首先开发了截止时间与能耗感知的UBAC（DE-UBAC），这是一种可解释的、基于规则的分层实现方法，融合了截止时间感知的优先级排序、协作式放置、能耗感知协调以及缓存更新代价。然而，预先定义的优先级以及……

    arXiv:2610.30768v1 Announce Type: new  Abstract: Content caching via unmanned aerial vehicles (UAVs) becomes one of the solutions to maintain content delivery when terrestrial infrastructure is unavailable or disrupted. However, existing caching approaches often focus primarily on cache-hit performance or average delivery delay, without jointly addressing spatially heterogeneous and time-varying demand, content-specific deadlines, and uneven communication-energy consumption across UAVs. To address these challenges, we formulate a dynamic deadline- and energy-aware cache placement and replacement problem that jointly considers deadline misses, UAV-side communication energy, and inter-UAV energy imbalance. To achieve this, we first develop deadline- and energy-aware UBAC (DE-UBAC), an interpretable hierarchical rule-based realization that incorporates deadline-aware prioritization, cooperative placement, energy-aware coordination, and cache-update cost. However, predefined priorities and
    
[^133]: 低秩约束马尔可夫决策过程的原对偶表征学习

    Primal-Dual Representation Learning for Low-Rank Constrained MDPs

    [https://arxiv.org/abs/2610.30756](https://arxiv.org/abs/2610.30756)

    提出REP-PD-TEG算法，通过衰减ε-贪婪策略探索、最大似然特征学习与乐观拉格朗日规划，为连续状态空间的低秩约束MDP实现表征学习，并提供了后悔度与软约束违反的理论保证。

    

    表征学习已大幅提升了无约束连续状态马尔可夫决策过程（MDP）的效率，但将其扩展到约束马尔可夫决策过程（CMDP）十分困难，因为转移表征必须在控制约束违反的同时进行学习。本文研究了具有连续状态空间的低秩CMDP的表征学习，并针对软累积违反设置提出了REP-PD-TEG算法。在特征学习阶段，该算法使用衰减$\epsilon$-贪婪混合执行策略与环境交互，并利用收集到的样本通过最大似然方法学习转移特征。基于学习到的特征，算法构建不确定性奖励项，并通过乐观拉格朗日目标进行规划以获得回报策略。我们证明了回报策略的后悔度与软违反保证，并量化了衰减$\epsilon$-贪婪执行策略的影响。

    arXiv:2610.30756v1 Announce Type: new  Abstract: Representation learning has substantially improved the efficiency of unconstrained continuous-state Markov Decision Processes (MDPs), but its extension to Constrained MDPs (CMDPs) is difficult because the transition representation must be learned while constraint violation is controlled. In this paper, we study representation learning for low-rank CMDPs with continuous state spaces, and propose REP-PD-TEG for the soft cumulative-violation setting. In the feature learning period, the algorithm interacts with the environment using a decaying-$\epsilon$-greedy mixture execution policy, and uses the collected samples to learn transition features by maximum likelihood. Based on the learned features, it constructs uncertainty bonuses and plans with an optimistic Lagrangian objective to obtain the return policy. We prove regret and soft-violation guarantees for the return policies, and quantify the influence of the decaying-$\epsilon$-greedy ex
    
[^134]: 可视化与解读肿瘤生存风险分析中的时间性护理团队动态

    Visualizing and Interpreting Temporal Care Team Dynamics for Cancer Survival Risk Analysis

    [https://arxiv.org/abs/2610.30748](https://arxiv.org/abs/2610.30748)

    该论文提出了一个可视化分析框架，通过EHR通信数据构建动态协作网络并结合时间图神经网络（TGNN），建模肿瘤护理团队随时间变化的互动，并揭示团队组成与专科参与的变化同患者风险演变之间的关系。

    

    癌症护理是一个纵向的、时间敏感的过程，需要多种医疗专业人员（HCPs）之间的协调配合，其协作关系在整个治疗过程中不断演变。先前的研究已将护理团队结构与患者预后联系起来，但协作的时间动态及其与患者风险演变之间的关系仍知之甚少。我们提出了一个可视化分析框架，用于建模和解读肿瘤学中随时间变化的护理团队互动。我们基于电子健康记录（EHR）介导的通信数据构建动态协作网络，并使用时间图神经网络（TGNN）来估计患者随时间推移的预后风险。随后，我们研究护理团队组成和专科医生参与的变化如何与各治疗阶段估计风险的变动相关联。我们的多视图交互系统将基于TGNN的风险建模与护理团队网络的联动可视化相结合。

    arXiv:2610.30748v1 Announce Type: new  Abstract: Cancer care is a longitudinal, time-sensitive process requiring coordination among diverse healthcare professionals (HCPs), whose collaboration evolves throughout treatment. Prior studies link care team structure to patient outcomes, but the temporal dynamics of collaboration and their relationship to evolving patient risk remain poorly understood. We introduce a visual analytics framework for modeling and interpreting time-varying care team interactions in oncology. We construct dynamic collaboration networks from electronic health record (EHR)-mediated communication data and use a temporal graph neural network (TGNN) to estimate patient outcome risk over time. We then investigate how changes in care team composition and specialist participation relate to shifts in estimated risk across treatment stages. Our multi-view interactive system combines TGNN-based risk modeling with coordinated visualizations of care team networks, feature-lev
    
[^135]: 面向拉曼光谱稳健分析的波数感知深度学习

    Wavenumber-Aware Deep Learning for Robust Analysis of Raman Spectra

    [https://arxiv.org/abs/2610.30746](https://arxiv.org/abs/2610.30746)

    本文提出将光谱点表示为强度-波数对并从波数本身计算Transformer位置编码，使深度学习模型在噪声、稀疏采样、部分光谱覆盖以及未见过的采样网格条件下均能稳健地分析拉曼光谱。

    

    用于光谱学的机器学习模型通常将光谱视为固定网格上的强度向量，物理坐标被隐含地编码在数组索引中。这类模型因此与单一的采样网格、光谱范围和分辨率绑定，限制了其在测量条件变化时的应用。本文将每个光谱点表示为强度与波数的配对，并直接从波数本身计算Transformer的位置编码。在三种细菌及其人工混合物的拉曼光谱上，这种表示方法在卷积网络、残差网络和Transformer网络中均表现出对噪声、稀疏采样和部分光谱覆盖的稳健性。在受控对比实验中，基于波数的编码在部分观测条件下优于基于索引的编码和可学习的编码，并且在训练中从未见过的采样网格上仍保持了性能，而基于索引的编码则出现性能下降。这些发现在公开数据集上得到了复现。

    arXiv:2610.30746v1 Announce Type: new  Abstract: Machine-learning models for spectroscopy usually treat a spectrum as a fixed-grid intensity vector, leaving the physical coordinate implicit in the array index. Such models are tied to a single sampling grid, spectral range and resolution, which limits their use when measurement conditions change. Here we represent each spectral point as a pair of intensity and wavenumber and compute Transformer positional encodings from the wavenumber itself. On Raman spectra of three bacterial species and their synthetic mixtures, this representation remained robust to noise, sparser sampling and partial spectral coverage across convolutional, residual and Transformer networks. In controlled comparisons, wavenumber-based encoding outperformed index-based and learned encodings under partial observation and retained its performance on sampling grids never seen during training, where index-based encoding degraded. These findings were reproduced on a publi
    
[^136]: 随机重掩码可扩展掩码扩散推理

    Random Remasking Scales Masked Diffusion Inference

    [https://arxiv.org/abs/2610.30744](https://arxiv.org/abs/2610.30744)

    该论文将掩码扩散模型推理重新表述为干净序列空间上的马尔可夫链，提出了无需微调、无需奖励的随机重掩码-去噪（R²D）测试时推理方法及轨迹校正训练（TCT），通过随机重掩码迭代校正解码错误，在逻辑任务上实现了接近对数线性的测试时性能扩展。

    

    掩码扩散模型（MDMs）通过逐步解除掩码来生成序列，但解码过程中引入的错误通常是不可逆的。近期研究表明，迭代校正可以提升样本质量，然而现有理论无法在给定一个学习到的、不完美的去噪器模型时解释这一现象。我们将MDM推理重新表述为干净序列空间上的马尔可夫链，该链由一个“重掩码-去噪”核定义，即随机重新引入掩码然后进行去噪。这有助于我们分析朝着数据分布方向的实际去噪轨迹。该分析催生了随机重掩码-去噪（R²D）推理方法，这是一种纯粹的测试时方法，无需微调、无需奖励信号。为进一步提升性能，我们提出了轨迹校正训练（TCT），这是一种去噪器训练方法，使模型在生成轨迹中暴露于自身产生的错误。在逻辑任务上，我们的方法展现出惊人的接近对数线性的测试时扩展……

    arXiv:2610.30744v1 Announce Type: new  Abstract: Masked diffusion models (MDMs) generate sequences by progressively unmasking tokens, but errors introduced during decoding are typically irreversible. Recent work has shown that iterative correction can improve sample quality, yet existing theory fails to explain this given a learned, imperfect denoiser model. We re-formulate MDM inference as a Markov chain on the clean sequence space, defined by a remask-denoise kernel that randomly re-introduces masks and then denoises. This helps us analyze the actual denoising trajectory towards the data distribution. The analysis motivates Random Remask-Denoise (R$^2$D) inference, a purely test-time, finetuning-free, reward-free procedure. To further improve performance, we propose Trajectory Correction Training (TCT), a denoiser-training procedure that exposes the model to its own errors along the generation trajectory. On logical tasks, our approach exhibits a striking near-log-linear test-time sc
    
[^137]: 训练过程中宽度与深度极限的可交换性

    Width and Depth Limits Commute During Training

    [https://arxiv.org/abs/2610.30729](https://arxiv.org/abs/2610.30729)

    本文将残差网络无限宽与无限深极限的可交换性从初始化阶段首次扩展到训练过程，证明了在Depth-μP参数化与梯度裁剪下，线性残差网络在任意有限训练步数内，网络输出沿宽度-深度联合增长路径按 $L^2$ 收敛且两极限可交换。

    

    近期的研究工作证明了，当残差分支以 $1/\sqrt{深度}$ 进行缩放时，残差网络在初始化时的无限宽与无限深极限是可交换的。特别地，无论宽度与深度是依次趋于无穷还是同时趋于无穷，网络输出都收敛到相同的极限。我们将这一可交换性结果扩展到线性残差网络的训练过程中。在相同的 $1/\sqrt{深度}$ 残差分支缩放（Depth-$\mu$P）和 $\mu$P 宽度参数化下，我们证明：在使用梯度裁剪的条件下，对于任意固定数量的梯度下降步数，当宽度和深度同时趋于无穷时，网络输出沿着每条宽度-深度联合增长路径按 $L^2$ 收敛。因此，无限宽与无限深极限在任何有限训练范围内都是可交换的。在不使用梯度裁剪的情况下，相应的 $L^2$ 收敛一般不成立，但我们建立了较弱的依概率收敛。

    arXiv:2610.30729v1 Announce Type: new  Abstract: Recent work established that the infinite-width and infinite-depth limits of residual networks commute at initialization when residual branches are scaled by $1/\sqrt{depth}$. In particular, the network output converges to the same limit whether width and depth tend to infinity sequentially or jointly. We extend this commutativity result to training in linear residual networks. Under the same $1/\sqrt{depth}$ residual-branch scaling (Depth-$\mu$P) and the $\mu$P width parameterization, we prove that, for any fixed number of gradient-descent steps with gradient clipping, the network output converges in $L^2$ along every joint width--depth growth path as both width and depth tend to infinity. Consequently, the infinite-width and infinite-depth limits commute throughout any finite training horizon. Without gradient clipping, the corresponding $L^2$ convergence does not hold in general, but we establish the weaker convergence in probability.
    
[^138]: 联邦图检索增强生成

    Federated Graph Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.30728](https://arxiv.org/abs/2610.30728)

    提出联邦GraphRAG（FedGRAG）统一框架，通过联邦知识路由机制在保护隐私的前提下解决缺乏全局知识图谱的检索难题，实现多客户端间的分布式图检索与知识增强生成。

    

    图检索增强生成利用集中式知识图谱来增强大语言模型的结构化知识检索能力。然而，现实世界中的知识天然分布在多个组织之间，而非集中可访问。隐私约束进一步禁止共享完整的本地知识图谱，这阻碍了GraphRAG有效检索和利用分布式知识。为此，我们提出了联邦GraphRAG（FedGRAG），一个在多个拥有知识图谱的客户端上进行分布式图检索与知识增强生成的统一框架。具体而言，我们解决了联邦GraphRAG中的两个基本挑战。首先，由于缺乏全局知识图谱，统一检索无法实现。为解决这一问题，我们开发了联邦知识路由机制，通过轻量级知识索引高效识别相关客户端，而无需暴露本地数据。

    arXiv:2610.30728v1 Announce Type: new  Abstract: Graph Retrieval-Augmented Generation (GraphRAG) leverages centralized knowledge graphs to enhance large language models with structured knowledge retrieval. However, real-world knowledge is inherently distributed across multiple organizations rather than being centrally accessible. Privacy constraints further prohibit sharing entire local knowledge graphs, preventing GraphRAG from effectively retrieving and utilizing distributed knowledge. To this end, we propose Federated GraphRAG (FedGRAG), a unified framework for distributed graph retrieval and knowledge-enhanced generation over multiple clients with knowledge graphs. Specifically, we address two fundamental challenges in federated GraphRAG. First, the absence of a global knowledge graph makes unified retrieval impossible. To address this, we develop a Federated Knowledge Routing mechanism that efficiently identifies relevant clients through lightweight knowledge indices without expos
    
[^139]: 学习追捕移动目标的可解模型

    Solvable models of learning to pursue a moving target

    [https://arxiv.org/abs/2610.30721](https://arxiv.org/abs/2610.30721)

    提出了一个可解析求解的连续时间线性二次追捕模型，其最优控制策略可在反应式与预测式追捕之间插值，并给出了最优策略与学习动力学的闭式解，学习时间尺度由特征值精确刻画。

    

    被赋予拦截移动目标任务的智能体必须学习如何到达目标，理想情况下应尽可能高效。正如对动物追捕行为的观察者所注意到的，存在多种可能的策略：一个极端是，智能体反应式地朝目标的当前位置移动；另一个极端是，智能体预测目标的未来位置并直接朝那里移动。出于理解这类策略如何被学习的动机，我们引入了一个开环目标追捕的连续时间线性二次模型，其最优控制策略可以在上述两种可能性之间进行插值。有用的是，少量可解释的参数即可控制哪种类型的策略是最优的，并且可以为最优策略、策略学习动力学和价值学习动力学推导出闭式解。利用模型的线性结构，我们发现学习的时间尺度精确对应于特征值……

    arXiv:2610.30721v1 Announce Type: new  Abstract: Agents tasked with intercepting a moving target must learn how to reach it, ideally as efficiently as possible. As observers of animal pursuit behavior have noticed, different strategies are possible: on one extreme, the agent reactively moves toward the target's current location; on the other, the agent predicts the target's future location and moves directly there. Motivated by the desire to understand how such strategies might be learned, we introduce a continuous-time linear-quadratic model of open-loop target pursuit whose optimal control strategies interpolate between these possibilities. Usefully, a small number of interpretable parameters control which type of strategy is optimal, and it is possible to derive closed-form solutions for optimal strategies, policy learning dynamics, and value learning dynamics. Exploiting our model's linear structure, we find that the time scales of learning precisely correspond to the eigenvalues o
    
[^140]: AIRO：一种基于流处理的工业物联网威胁检测与安全机器学习框架

    AIRO: A Streaming-Based Machine Learning Framework for Threat Detection and Security in Industrial Internet of Things

    [https://arxiv.org/abs/2610.30713](https://arxiv.org/abs/2610.30713)

    该论文提出了AIRO框架，通过融合Kafka流处理、攻击特征提取、机器学习推理和基于置信度的决策逻辑，实现对工业物联网环境中多种网络威胁的实时检测与自动化风险降低。

    

    现代工业物联网（IIoT）部署现已连接边缘网关、云端监控仪表盘和业务应用，这提升了可见性和远程监控能力。然而，它同时也使工业流量暴露于多种网络威胁之下，包括扫描、Web攻击和结构化查询语言（SQL）注入（SQLi）、钓鱼驱动的凭证窃取，以及命令与控制（C2）活动。尽管基于规则的工具在检测已知特征签名方面仍然有效，但其在应对轻微变形的攻击或需要人工调查警报时效能会下降。本文提出了AIRO（自主智能驱动的网络安全风险降低系统），一个面向IIoT监控与响应的高效框架。AIRO结合了Apache Kafka流处理、针对特定攻击的特征提取、机器学习（ML）推理、基于置信度的决策逻辑，以及诸如互联网协议（IP）封禁等风险降低措施。

    arXiv:2610.30713v1 Announce Type: cross  Abstract: Modern Industrial Internet of Things (IIoT) im- plementations now connect edge gateways, cloud dashboards and business applications. This improves visibility and remote monitoring. However it simultaneously exposes industrial traffic to various cyber threats including scanning, web attacks and Structured Query Language (SQL) injection (SQLi), phishing- driven credential theft, and command-and-control (C2) activity. Although rule-based tools remain useful for detecting known signatures, their efficacy decreases with minimally altered attacks or when alerts require manual investigation. This paper presents AIRO (Autonomous Intelligence-Driven Cybersecurity Risk Re- duction System), an efficient framework for IIoT monitoring and response. AIRO combines Apache Kafka streaming, attack- specific feature extraction, machine learning (ML) inference, confidence-based decision logic and risk reduction measures such as Internet Protocol (IP) bloc
    
[^141]: 一种具备工具与记忆的智能体架构，用于手腕光电容积脉搏波（PPG）心率估计的选择性校正

    An Agent Architecture with Tools and Memory for Selective Correction of Wrist PPG Heart Rate Estimates

    [https://arxiv.org/abs/2610.30688](https://arxiv.org/abs/2610.30688)

    该论文提出一种带工具与记忆的智能体架构，通过独立风险门控对手腕PPG心率估计进行选择性校正，在四个公开数据集上使平均绝对误差最多降低44.5%。

    

    手腕光电容积脉搏波（PPG）心率估计容易受到运动伪影的影响，而那些能降低平均误差的校正却可能使个别时间窗口的表现变差。我们提出了一种围绕固定心率估计器构建的、具备工具与记忆的智能体架构。脉搏、运动以及有限的历史证据用于生成候选估计值，候选估计的预测收益与危害则指导选择过程。一个独立的风险门控决定对每个输出是保留、校正还是拒绝。我们在PPG-DaLiA数据集上训练了该架构，并在冻结权重的情况下，在Wrist Exercise、GalaxyPPG和WildPPG数据集上进行了测试。在这四个公开数据集上，对十个估计器进行宏平均后的平均绝对误差（MAE）最多降低了44.5%，其中在GalaxyPPG上降幅最大。有益校正与有害校正分别使绝对误差降低和增加超过1 bpm，其十估计器宏平均比例分别在16.06%至36.64%和1.68%至……

    arXiv:2610.30688v1 Announce Type: new  Abstract: Wrist photoplethysmography (PPG) heart rate estimates are vulnerable to motion artifacts, and corrections that reduce mean error can worsen individual windows. We propose an agent architecture with tools and memory around a fixed heart rate estimator. Pulse, motion, and limited historical evidence generate candidate estimates, whose predicted benefits and harms guide selection. An independent risk gate determines whether to retain, correct, or reject each output. We trained the architecture on PPG-DaLiA and tested it on Wrist Exercise, GalaxyPPG, and WildPPG with frozen weights. Across these four public datasets, mean absolute error (MAE), macro-averaged over ten estimators, decreased by up to 44.5%, with the largest reduction on GalaxyPPG. Beneficial and harmful corrections decreased and increased absolute error by more than 1 bpm, respectively. Their ten-estimator macro-averaged proportions ranged from 16.06% to 36.64% and from 1.68% t
    
[^142]: 基于流匹配的渐进域自适应域轨迹学习

    Learning Domain Trajectories with Flow Matching for Gradual Domain Adaptation

    [https://arxiv.org/abs/2610.30685](https://arxiv.org/abs/2610.30685)

    提出FlowGDA框架，利用流匹配将渐进域自适应建模为表示空间统计流形上的连续输运，通过学习时间依赖的速度场生成平滑域轨迹，避免了传统离散逐步方法中的误差累积和不稳定的对抗训练。

    

    arXiv:2610.30685v1 公告类型：新 摘要：大幅度的分布偏移仍然是领域泛化和无监督自适应面临的主要障碍。尽管渐进域自适应通过引入有序的中间域缓解了这一难题，但大多数现有方法仍以离散、逐步的方式进行，这会累积误差，并且往往依赖于不稳定的 Min-Max 优化。我们提出 FlowGDA，一个流匹配框架，它将渐进自适应建模为表示空间中统计流形上的连续输运。FlowGDA 通过学习随时间变化的速度场来诱导平滑的域轨迹，并采用无需模拟的回归目标，而非对抗训练。为防止语义漂移，我们将相邻域之间的从左到右输运与源域锚定的输运以及语义一致性损失相耦合，并通过两阶段顺序优化方案进一步稳定学习过程。大量实验……

    arXiv:2610.30685v1 Announce Type: new  Abstract: Large distribution shifts remain a primary obstacle for domain generalization and unsupervised adaptation. Although gradual domain adaptation alleviates this difficulty by introducing ordered intermediate domains, most existing approaches still operate in a discrete, step-wise manner, which can accumulate errors and often relies on unstable Min--Max optimization. We propose \textbf{FlowGDA}, a flow-matching framework that models gradual adaptation as \emph{continuous} transport on a statistical manifold in a representation space. FlowGDA learns time-dependent velocity fields that induce smooth domain trajectories, using a simulation-free regression objective rather than adversarial training. To prevent semantic drift, we couple left-to-right transport between consecutive domains with a source-anchored transport and a semantic consistency loss, and we further stabilize learning via a two-stage sequential optimization scheme. Extensive exp
    
[^143]: 安全探针的缩放定律

    Scaling Laws for Safety Probes

    [https://arxiv.org/abs/2610.30678](https://arxiv.org/abs/2610.30678)

    本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。

    

    安全探针是在模型激活值上训练的、用于实时监控语言模型的标准工具。每当发布新模型或需要监控新的危害领域时，探针都必须重新训练，这需要新的标签以及探针架构的选择。然而，无论是模型还是标签都无法预先告诉我们最终探针的性能如何，要弄清这一点需要训练并比较多种探针配置。在本文中，我们探讨是否可以仅凭模型规模和标签数量来提前预测探针的性能。现有的安全数据集不足以支持这一研究，因为它们要么规模有限，要么已经饱和。因此，我们构建了六个规模足够大、难度足够高的新安全数据集。在这些数据集上，我们在来自五个模型家族的20个基础模型上训练了两种探针架构，每个模型使用32到10⁶个标签。我们使用探针误差来衡量性能……

    arXiv:2610.30678v1 Announce Type: cross  Abstract: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, d
    
[^144]: RiboBridge：蛋白质条件下的从头RNA结合体结构设计

    RiboBridge: protein-conditioned de novo RNA binder structure design

    [https://arxiv.org/abs/2610.30668](https://arxiv.org/abs/2610.30668)

    RiboBridge提出了一种结构优先的框架，通过跨坐标系交互几何将RNA构象生成与蛋白质-RNA组装解耦，实现了无需预定义结合位点的蛋白质靶向RNA结合体从头设计。

    

    设计能够结合目标蛋白结构的RNA分子，需要同时生成RNA构象及其结合复合物的几何结构，且不能假设存在预先定义的结合位点。现有方法通常在统一表示中对这些耦合过程进行建模，导致RNA构象生成与蛋白质相对位置放置相互纠缠。在这项工作中，我们提出了RiboBridge，这是一个结构优先的框架，通过跨坐标系交互几何将RNA构象生成与蛋白质-RNA组装分离开来。RiboBridge首先在蛋白质表面提出多个候选RNA结合界面，在预训练的RNA结构潜在空间（RiboSphere）中生成典型的RNA构象，然后预测蛋白质-RNA相互作用约束，以桥接独立定义的RNA和蛋白质坐标系。由此得到的相互作用几何被转换为几何一致的蛋白质相对位姿

    arXiv:2610.30668v1 Announce Type: new  Abstract: Designing RNA molecules that bind to target protein structures requires generating both an RNA conformation and its bound complex geometry, without assuming a predefined binding site. Existing approaches generally model these coupled processes within a unified representation, entangling RNA conformation generation with protein-relative placement. In this work, we introduce RiboBridge, a structure-first framework that separates RNA conformational generation from protein--RNA assembly through cross-frame interaction geometry. RiboBridge first proposes multiple candidate RNA-binding interfaces on the protein surface, generates canonical RNA conformations within a pretrained RNA structural latent space (RiboSphere), and then predicts protein--RNA interaction constraints to bridge the independently defined RNA and protein coordinate frames. The resulting interaction geometry is converted into a geometrically consistent protein-relative pose t
    
[^145]: 基于深度强化学习的快速迭代组合拍卖

    Fast Iterative Combinatorial Auctions via Deep Reinforcement Learning

    [https://arxiv.org/abs/2610.30667](https://arxiv.org/abs/2610.30667)

    FastCombo通过离线深度强化学习为迭代组合拍卖学习序贯定价策略，以单次神经网络前向传播完成价格更新，从理论上克服了单步贪婪定价的局限，并显著减少拍卖出清所需的轮次。

    

    迭代组合拍卖被应用于频谱分配、交通运输和工业采购等场景，但拍卖轮次代价高昂。我们提出了FastCombo，一种用于序贯价格发现的深度强化学习方法。贝叶斯定价即使与针对每个实例用事后信息调优的次梯度基准相比，也能显著减少拍卖轮次，但其依赖于单步目标以及反复的在线推理与优化。我们证明，即使已知真实的估值分布，贪婪地最大化即时出清概率的策略在期望意义上也可能需要任意多的额外轮次。FastCombo在离线阶段学习定价策略，为那些能够促成未来决策的查询分配信用，并通过单次神经网络前向传播生成每次在线价格更新。我们将单需求拍卖中最小化到出清的期望轮次的目标形式化为部分可观测马尔可夫决策过程（摘要至此截断）。

    arXiv:2610.30667v1 Announce Type: cross  Abstract: Iterative combinatorial auctions are used in spectrum allocation, transportation, and industrial procurement, but rounds are costly. We introduce FastCombo, a deep reinforcement learning approach to sequential price discovery. Bayesian pricing has sharply reduced auction rounds even against subgradient benchmarks tuned with hindsight for each instance, but relies on a one-step objective and repeated online inference and optimization. We prove that greedily maximizing immediate clearing probability can require arbitrarily more rounds in expectation than an informative policy, even with the true valuation distribution known. FastCombo learns price policies offline, assigning credit to queries for the future decisions they enable and producing each online price update with a single neural-network forward pass. We formulate the objective of minimizing expected rounds to clearing in single-minded auctions as a partially observable Markov de
    
[^146]: 通过行为反事实解释模型推理

    Interpreting Model Reasoning through Behavioral Counterfactuals

    [https://arxiv.org/abs/2610.30630](https://arxiv.org/abs/2610.30630)

    该论文提出基于行为反事实的方法，通过训练模块编辑残差流以最小化固定响应下输出分布的KL散度，从而定位并分析干预对模型推理的影响，且比标准激活重构目标更能有效恢复反事实行为。

    

    随着模型能力的不断提升，忠实地理解模型形成思维链过程中发生的内部机制成为一个重要问题。然而，由于采样响应之间存在差异，这项工作仍然充满挑战，使得研究者难以隔离干预措施的影响并系统地比较模型推理。我们提出了一种通过行为反事实来定位和分析给定干预如何影响模型推理的方法：给定一个固定响应，向提示中引入干预会如何改变下一个词元的分布？我们训练一个模块来编辑残差流，使其最小化固定响应在源提示与其反事实目标提示下输出分布之间的KL散度。我们发现，匹配输出分布比标准的激活重构目标能更有效地恢复反事实行为，并验证了……

    arXiv:2610.30630v1 Announce Type: new  Abstract: Faithfully understanding the internal process that occurs as models form their chain of thought is an important problem with the rise in model capabilities. However, this remains challenging due to the variance in sampled responses, making it difficult to isolate the effects of interventions and compare model reasoning systematically. We propose a method for localizing and analyzing how a given intervention affects the model's reasoning through behavioral counterfactuals: given a fixed response, how does introducing an intervention to the prompt change the next-token distributions? We train a module to edit the residual stream such that it minimizes the KL divergence between the output distributions for a fixed response under a source prompt and its counterfactual target prompt. We find that matching the output distributions recovers the counterfactual behavior more effectively than standard activation reconstruction objectives and valid
    
[^147]: 时间序列异常检测的未来：从历史回顾到统一展望

    Future of Time Series Anomaly Detection: From Historical Review to Unified Outlook

    [https://arxiv.org/abs/2610.30628](https://arxiv.org/abs/2610.30628)

    本综述系统梳理了时间序列异常检测领域，建立了精细的异常分类体系，并构建了涵盖扩散模型、状态空间模型和大语言模型等新兴范式的结构化分类框架，同时对评估方法与指标可靠性进行了统一分析，为该领域的未来发展指明了方向。

    

    时间序列异常检测（TSAD）是数据挖掘中的一个基础问题，旨在识别显著偏离正常时间模式的观测值。在数据量不断增长和应用复杂性日益提升的驱动下，近期研究产生了多种多样的架构，包括扩散模型、状态空间模型和大语言模型（LLMs）。然而，现有综述对这些新兴设计的覆盖有限，且缺乏对多变量异常、跨领域统一建模、指标可靠性以及当代评估挑战的统一处理。为弥补这些空白，本综述提供了三项核心贡献：（1）我们建立了一个精细的异常分类体系和一个反映现代进展的结构化范式分类法，比较了各范式的检测机制并描绘了其发展轨迹；（2）我们对评估方法进行了广泛分析，并开展了……（原文摘要在此处截断）

    arXiv:2610.30628v1 Announce Type: new  Abstract: Time series anomaly detection (TSAD) is a foundational problem in data mining, aiming to identify observations that deviate significantly from nominal temporal patterns. Driven by escalating data volumes and application complexity, recent research has yielded diverse architectures, including diffusion models, state space models, and large language models (LLMs). However, existing surveys provide limited coverage of these emerging designs and lack a unified treatment of multivariate anomalies, cross-domain unified modeling, metric reliability, and contemporary evaluation challenges. To bridge these gaps, this survey delivers three core contributions: (1) We establish a refined anomaly categorization and a structured taxonomy of paradigms reflecting modern advances, comparing detection mechanisms and charting developmental trajectories across paradigms. (2) We provide an extensive analysis of evaluation methodologies and conduct comprehens
    
[^148]: 基于决策质量的1型糖尿病数字孪生评估

    Evaluating Type-1 Diabetes Digital Twins by Decision Quality

    [https://arxiv.org/abs/2610.30577](https://arxiv.org/abs/2610.30577)

    本文提出不再以血糖轨迹预测精度、而是以下游治疗方案排序的决策质量来评估1型糖尿病数字孪生，并基于FDA认可的UVA/Padova模型中30个虚拟患者、21种基础-追加治疗方案建立了系统的评估协议。

    

    1型糖尿病（T1D）的数字孪生（DT）可以预测个体血糖对不同胰岛素治疗方案将如何响应，而无需实际尝试这些方案。对于每种治疗方案，可以计算预测血糖轨迹的下游统计数据（例如血糖风险），以指导临床决策。然而，数字孪生的质量通常是通过其预测血糖轨迹的准确性来衡量的，这可能无法反映其所支撑的下游治疗决策的质量。在本工作中，我们直接用下游治疗方案排序的决策质量来评估数字孪生，即数字孪生所预测的血糖轨迹能否像真实患者那样用于对候选治疗方案进行排序。我们提出了一套评估协议，涵盖来自FDA认可的UVA/Padova模型的30个虚拟患者，每位患者在21种不同的基础-追加（basal-bolus）治疗方案下进行仿真。对于每个患者，数字孪生在一种治疗方案下拟合，然后生成血……（原文摘要在此处截断）

    arXiv:2610.30577v1 Announce Type: new  Abstract: A digital twin (DT) for type 1 diabetes (T1D) can predict how an individual's blood glucose will respond to different insulin treatments without actually trying them. For each treatment, downstream statistics of predicted glucose trajectories, such as glycemic risk, can be computed to guide clinical decisions. However, the quality of DTs is usually measured by the accuracy of their predicted glucose trajectories, which may not reflect the quality of the downstream treatment decisions they inform. In this work, we directly evaluate DTs by this decision quality on downstream treatment ranking, i.e., whether a DT predicts glucose trajectories that can be used to rank candidate treatments as the real patient would. We present an evaluation protocol over 30 virtual patients from the FDA-accepted UVA/Padova model, each simulated under 21 different basal-bolus treatments. For each patient, a DT is fitted under one treatment and then produces gl
    
[^149]: 弥合封闭世界差距：基于TSK-FCM混合集成方法在BETH数据集上的可解释异常检测

    Bridging the Closed-World Gap: Interpretable Anomaly Detection via a Hybrid TSK-FCM Ensemble on the BETH Dataset

    [https://arxiv.org/abs/2610.30576](https://arxiv.org/abs/2610.30576)

    提出了一种将梯度优化TSK模糊分类器与FCM聚类模块协同结合的混合模糊集成方法，在BETH数据集上实现了既能精确检测已知内部威胁、又能发现新型几何异常的可解释异常检测，从而弥合监督与无监督学习之间的封闭世界差距。

    

    arXiv:2610.30576v1 公告类型：交叉 摘要：机器学习在网络安全部署中面临一个根本性的悖论：监督学习模型对新颖的分布外攻击泛化能力较差，而无监督模型往往存在可分性差的问题，导致误报率难以控制。BETH数据集正是这一挑战的典型例证，这是一个内核级基准数据集，其训练数据仅包含良性和可疑的内部行为，而测试集几乎完全由新颖的僵尸网络激活攻击组成。为了弥合这一检测差距，我们提出了一种混合模糊集成方法，将监督学习的精确性与无监督聚类的适应性协同结合。我们的架构集成了一个梯度优化的Takagi-Sugeno-Kang（TSK）模糊分类器，作为已知内部威胁的检测专家，并与一个并行的模糊C均值（FCM）模块协同工作以检测几何异常。我们研究的一个关键发现是，传统……

    arXiv:2610.30576v1 Announce Type: cross  Abstract: A fundamental paradox plagues the deployment of Machine Learning in cybersecurity: supervised models generalize poorly to novel, Out-of-Distribution (OoD) attacks, whereas unsupervised models often suffer from poor separability, leading to unmanageable false-alarm rates. This challenge is exemplified by the BETH dataset, a kernel-level benchmark in which the training data consists solely of benign and suspicious insider behaviors, whereas the test set consists almost entirely of novel botnet activation attacks. To bridge this detection gap, we propose a Hybrid Fuzzy Ensemble that synergizes the precision of supervised learning with the adaptability of unsupervised clustering. Our architecture integrates a gradient-optimized Takagi-Sugeno-Kang (TSK) fuzzy classifier to act as an expert on known insider threats, with a parallel Fuzzy C-Means (FCM) module to detect geometric anomalies. A critical finding of our research is that traditiona
    
[^150]: 当新闻变旧时：基于强化学习与大语言模型的投资组合配置

    When News Gets Old: Portfolio Allocation with Reinforcement Learning and Large Language Models

    [https://arxiv.org/abs/2610.30564](https://arxiv.org/abs/2610.30564)

    研究发现在LLM驱动的投资组合配置中，过滤旧新闻比向策略提供时间信息更有效——当年新闻延迟四周交付时，过滤旧新闻可实现18.43%的年化净收益，并优于所有基于新闻的强化学习策略。

    

    金融新闻在发布后的很长时间内仍可能影响投资决策。及时的新闻可能在价格充分调整之前到达投资者手中，而较旧的新闻可能已经反映在价格中。我们研究了两种处理老化信号的方法：保留它们并向投资组合策略提供时间信息，或在投资组合选择之前将其过滤掉。一个固定的大语言模型（LLM）从公开的SEC文件中提取信号。我们在强化学习（RL）策略和直接选择投资组合的LLM中比较这些处理方法。我们在2023-2025年的历史测试中评估扣除交易成本后的收益。在受控测试中，RL选择会对时间信息作出反应，但历史收益显示添加时间信息并无一致的好处。对于LLM而言，过滤旧新闻产生了最高的平均净收益，当新闻交付延迟四周时，年化收益达到18.43%。这超过了所有使用新闻的RL策略的收益。

    arXiv:2610.30564v1 Announce Type: new  Abstract: Financial news can influence investment decisions long after publication. Timely news may reach an investor before prices fully adjust, whereas older news may already be reflected in prices. We examine two ways to handle aging signals: retain them and give a portfolio policy timing information, or filter them out before portfolio selection. A fixed large language model (LLM) extracts signals from public SEC disclosures. We compare these treatments in reinforcement learning (RL) policies and LLMs that choose portfolios directly. We evaluate returns after trading costs in historical tests over 2023-2025. RL choices respond to timing information in a controlled test, but historical returns show no consistent benefit from adding it. For the LLM, filtering old news produced the highest average net return, reaching 18.43% per year when news delivery was delayed by four weeks. This exceeded the returns of all RL strategies that used news.
    
[^151]: 持续演化的AI分析师实现终身有效的决策

    Lifetime-valid decisions by continually evolving AI analysts

    [https://arxiv.org/abs/2610.30562](https://arxiv.org/abs/2610.30562)

    提出“持续验证”方法，在部署前使用可随机化和扩展的零假设模拟数据评估每个AI分析师版本，并证明无论更新如何选择，部署第一类错误率超标版本的终身概率始终有界，从而保障持续演化的AI分析师所有后续决策的终身有效性。

    

    AI智能体越来越多地扮演分析师的角色，其决策通常可以归结为假设检验。经典检验针对固定程序控制第一类错误率，即拒绝真实零假设的概率。然而，AI分析师引入了内部随机性，并且会随时间不断演化。随着AI分析师的演化，批准第一类错误率超过目标水平的版本的风险会不断累积，而对评估反馈的自适应使用可能导致过拟合。我们提出了持续验证方法，在部署前使用零假设下的模拟数据对每个提出的AI分析师版本进行评估，其测试场景可以随时间随机化和扩展。我们证明，无论更新如何选择或提出的更新数量多少，部署第一类错误率超过允许水平的版本的终身概率始终保持有界。这保证了每个后续决策都能控制第一类错误率。

    arXiv:2610.30562v1 Announce Type: cross  Abstract: AI agents increasingly act as analysts whose decision can often be cast as hypothesis tests. Classical tests control the Type I error rate, the probability of rejecting a true null hypothesis, for fixed procedures. However, AI analysts introduce internal randomness and can evolve continuously over time. As AI analysts evolve, the risk of approving versions with Type I error rates above the target can accumulate, and adaptive use of evaluation feedback can lead to overfitting. We introduce continual validation, which evaluates each proposed AI analyst version before deployment using simulated data under the null hypothesis, with scenarios that can be randomized and expanded over time. We prove that, regardless of how updates are chosen or how many are proposed, the lifetime probability of ever deploying a version whose Type I error rate exceeds the allowed level remains bounded. This guarantees that every subsequent decision controls th
    
[^152]: 基于换道预测的自动驾驶车辆决策制定与乘坐舒适性评估

    Decision Making and Ride Comfort Evaluation Based on Lane-Change Prediction for Autonomous Vehicles

    [https://arxiv.org/abs/2610.30528](https://arxiv.org/abs/2610.30528)

    本研究提出将基于知识图谱嵌入和贝叶斯推断的换道预测方法集成到自动驾驶决策中，可提前3-4秒预判周围车辆的换道行为，相比反应式驾驶策略有效避免了碰撞并改善了乘坐舒适性。

    

    预测周围车辆的换道意图能够使自动驾驶车辆预判潜在危险并采取主动响应，而非仅仅依赖被动反应式操作。本研究探讨了将基于知识图谱嵌入和贝叶斯推断的换道预测集成到自动驾驶车辆决策制定中的益处。在CARLA仿真平台构建的危险汇入场景中，研究评估了三种驾驶策略：反应式驾驶、基于预测的纵向动作以及基于预测的横向动作。所提出的预测模型能够在周围车辆实际换道发生前约3-4秒预判其行为，从而使本车能够相应地调整自身行为。在代表性仿真记录中，反应式驾驶导致碰撞并伴随剧烈减速。相比之下，基于预测的纵向动作通过提前制动建立起安全的（原文在此处截断）

    arXiv:2610.30528v1 Announce Type: cross  Abstract: Predicting surrounding vehicles' lane-change intentions enables autonomous vehicles to anticipate potential hazards and respond proactively rather than relying solely on reactive maneuvers. This study investigates the benefits of integrating lane-change prediction, based on knowledge graph embeddings and Bayesian inference, into autonomous vehicle decision-making. Three driving strategies are evaluated in risky merging scenarios simulated in CARLA: reactive driving, prediction-based longitudinal action, and prediction-based lateral action. The proposed prediction model anticipates the surrounding vehicle's lane change approximately 3-4 seconds before its occurrence, allowing the ego vehicle to adjust its behavior accordingly. In the representative simulation records, reactive driving results in a collision accompanied by aggressive deceleration. In contrast, prediction-based longitudinal action initiates early braking to establish a sa
    
[^153]: LLM2Cert：基于SMT/Lean验证的大语言模型证书合成

    LLM2Cert: LLM Synthesis of Certificates with SMT/Lean Verification

    [https://arxiv.org/abs/2610.30524](https://arxiv.org/abs/2610.30524)

    提出LLM2Cert框架，利用大语言模型合成李雅普诺夫函数与障碍证书，并通过SMT求解器和Lean定理证明器进行严格的形式化验证，同时发布包含105个实例的DynaCertBench基准以覆盖高维非线性系统。

    

    李雅普诺夫函数和障碍证书能够为系统的稳定性与安全性提供状态空间层面的保证，但如何合成并验证它们——尤其是针对非线性、高维系统——仍然是一个未解决的挑战。现有流程依赖数值求解器（平方和规划）、区间方法或可满足性模理论（SMT）求解器来验证候选证书，这些方法往往需要松弛处理、结果过于保守，或随模型维度和非线性程度的增加而扩展性变差。这些局限性也影响了基准测试的设计：标准测试集主要由低维多项式系统主导，其选取依据是求解器所能处理的问题而非工程应用的实际需求。我们从两个方面应对这一挑战。首先，我们提出DynaCertBench基准，包含105个实例：其中80个非冗余系统精选自文献，并扩展了高维和控制受限的变体以及新的非线性耦合系统……（原文摘要在此处截断）

    arXiv:2610.30524v1 Announce Type: cross  Abstract: Lyapunov functions and barrier certificates give state-space guarantees of stability and safety, but synthesizing and verifying them---especially for nonlinear, high-dimensional systems---remains an open challenge. Existing pipelines verify candidates with numerical solvers (sum-of-squares programs), interval methods, or satisfiability-modulo-theories (SMT) solvers, which often require relaxations, are conservative, or scale poorly with model dimension and nonlinearity. These limitations have also shaped benchmark design: standard test suites are dominated by low-dimensional polynomial systems chosen to suit what solvers can handle rather than what engineering applications require. We address both sides of this challenge. First, we introduce DynaCertBench, a benchmark of 105 instances: 80 non-redundant systems curated from the literature, extended with high-dimensional and control-constrained variants and new nonlinearly coupled system
    
[^154]: NEST：用于多尺度聚类的嵌套嵌入表示

    NEST: Nested Embedding Representation for Multiscale Clustering

    [https://arxiv.org/abs/2610.30520](https://arxiv.org/abs/2610.30520)

    受俄罗斯套娃表示学习启发，NEST提出了一种无监督框架，在单一嵌入中编码从粗到细的多尺度聚类结构，无需在训练中观测层次信息即可恢复数据的多尺度聚类组织。

    

    现实世界的数据通常在多个分辨率上呈现出聚类结构，然而现有方法通常只学习单一的划分或构建显式的层次结构。受俄罗斯套娃表示学习的启发，我们提出了NEST，这是一个无监督框架，能够在单一嵌入中编码从粗到细的结构。NEST在嵌套的前缀子空间上联合优化聚类与分离，随着嵌入维度的增加实现逐渐更精细的区分，并通过秩正则化和簇内正则化防止嵌入坍缩。实验表明，尽管层次结构在训练过程中从未被观测到，学习到的嵌入中的距离比输入距离更贴近层次距离。在合成层次数据上，NEST仅需每层的聚类数量即可恢复多尺度结构。在视觉、文本和单细胞数据上的实验进一步证明了多尺度组织的存在。

    arXiv:2610.30520v1 Announce Type: new  Abstract: Real-world data often exhibits cluster structure at multiple resolutions, yet existing methods typically learn a single partition or construct an explicit hierarchy. Inspired by Matryoshka Representation Learning, we introduce NEST, an unsupervised framework that encodes coarse-to-fine structure within a single embedding. NEST jointly optimizes clustering and separation across nested prefix subspaces, enabling progressively finer distinctions as embedding dimensionality increases, with rank and within-cluster regularization that keep the embedding from collapsing. Empirically, distances in the learned embedding follow the hierarchy distance more closely than those of the input, although the hierarchy is never observed during training. On synthetic hierarchies, NEST recovers multiscale structure given only the number of clusters per level. Experiments on vision, text, and single-cell data further demonstrate multiscale organization across
    
[^155]: 速率约束下的最优多任务表示：一种信息瓶颈分析

    Optimal Multitask Representations Under a Rate Constraint: An Information Bottleneck Analysis

    [https://arxiv.org/abs/2610.30493](https://arxiv.org/abs/2610.30493)

    该论文通过信息瓶颈框架分析信息预算约束下的最优多任务预训练表示，发现压缩程度会改变哪个预训练目标最优，且不存在在所有任务族上都表现一致的固定预测目标。

    

    在信息预算约束下，哪种预训练目标能为下游任务产生最佳表示？我们将该问题形式化为一个多任务信息瓶颈问题，使用互信息约束来建模不完整的信息保留。对于高斯输入和任务，我们刻画了最优表示，并表明弱任务对齐的专家模型在低速率下可以优于重构通用模型，但在高速率下会落后。我们还构建了任务族，对于这些任务族，任何固定的高斯预测目标都会产生任意大的相对遗憾，这是由最优分配优先级的逆转所驱动的。相反，平均任务信息矩阵定义了一个具有统一相对遗憾上界的目标，该上界由最大任务信噪比控制。CelebA 和 dSprites 上的实验表明，压缩可以改变哪个预训练目标表现最佳，甚至任务感知训练也可能逆转哪个下游任务……

    arXiv:2610.30493v1 Announce Type: new  Abstract: Which pretraining objective yields the best representation for downstream tasks under an information budget? We formalize this problem as a multitask information bottleneck, using a mutual information constraint to model incomplete information retention. For Gaussian inputs and tasks, we characterize optimal representations and show that a weakly task-aligned specialist can outperform a reconstruction generalist at low rates but lose at high rates. We also construct task families for which every fixed Gaussian prediction objective incurs arbitrarily large relative regret, driven by reversals in optimal allocation priorities. Conversely, the average task information matrix defines an objective with a uniform relative-regret bound controlled by maximum task SNR. CelebA and dSprites experiments illustrate that compression can change which pretraining objective performs best, and even task-aware training can reverse which downstream task is 
    
[^156]: FIGFLEX：与低比特LLM量化协同设计的混合精度FP-INT矩阵乘加速器

    FIGFLEX: A Mixed-Precision FP-INT GEMM Accelerator Co-Designed with Low-Bit LLM Quantization

    [https://arxiv.org/abs/2610.30480](https://arxiv.org/abs/2610.30480)

    FIGFLEX是一种算法-硬件协同设计的混合精度FP-INT矩阵乘加速器，通过移位加法基-4乘法重构将部分积减半，使数据通路吞吐量随权重精度降低而扩展，从而让低比特LLM量化真正转化为计算性能的提升。

    

    仅权重量化方法将大语言模型（LLM）的激活值保持为浮点数，并将权重存储为低比特整数，使得主要计算负载转变为浮点-整数（FP-INT）通用矩阵乘法。尽管最新的量化器通过混合精度达到了平均2-3比特的水平，但硬件发展并未跟上步伐：GPU在张量核执行乘累加运算之前需将权重反量化为FP16，而位并行的FP-INT加速器只能运行在单一权重精度下，因此更少的比特数只节省了内存而未能节省计算。在本文中，我们提出了FIGFLEX，这是一种算法-硬件协同设计的混合精度加速器，其位并行数据通路的吞吐量可随精度降低而扩展。FIGFLEX的贡献包括：（1）对FP-INT乘法的一种移位加法、基-4（radix-4）重新表述，将乘累加融合到统一的、精度灵活的部分积归约中，并将部分积数量大约减半；同时提出了一种新颖的INT3优化方法，以回收空闲的数据通路资源。

    arXiv:2610.30480v1 Announce Type: cross  Abstract: Weight-only quantization keeps LLM activations in floating point and stores weights as low-bit integers, turning the dominant workload into floating-point-integer (FP-INT) general matrix multiplication. Recent quantizers reach a 2-3-bit average with mixed precision, but hardware has not kept pace: GPUs dequantize weights to FP16 before the tensor-core multiply-accumulate, and bit-parallel FP-INT accelerators run at a single weight precision, so fewer bits save memory but not computation. In this paper, we present FIGFLEX, an algorithm-hardware co-designed mixed-precision accelerator whose bit-parallel datapath scales throughput as precision falls. FIGFLEX contributes (1) a shift-add, radix-4 reformulation of FP-INT multiplication that fuses the multiply-accumulate into a unified, precision-flexible partial-product reduction and roughly halves the partial products, together with a novel INT3 optimization that reclaims the idle datapath 
    
[^157]: 重新思考SAE评估：原子可解释单元框架揭示隐藏的多义性与冗余性

    Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

    [https://arxiv.org/abs/2610.30474](https://arxiv.org/abs/2610.30474)

    本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。

    

    稀疏自编码器（SAE）最初是为发现单义特征而提出的，单义性是刻画单个特征的性质，但并未规定学习到的特征集合整体上应该是什么样子。随着SAE越来越多地被应用于归因分析、回路发现和干预，研究者们开始使用描述特征集合全局性质的评估指标，例如稀疏性、死特征比例、重建保真度和解耦性等，这一系列指标已远远超出单义性的范畴。然而，这种超越单义性的隐式转变尚未被形式化为一个清晰、明确定义的、先于这些评估指标存在的目标。缺少这样的目标，现有指标仍然是临时拼凑的，其必要性缺乏依据。在这项工作中，我们通过为SAE评估引入一种新范式来形式化这一最终目标：要求SAE特征，以及更广义的字典……（摘要不完整）

    arXiv:2610.30474v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictio
    
[^158]: DiffLTM：用于条件生成的连续扩散大型表格模型

    DiffLTM: Continuous Diffusions Large Tabular Models for Conditional Generation

    [https://arxiv.org/abs/2610.30472](https://arxiv.org/abs/2610.30472)

    本文提出DiffLTM——一种用于表格数据条件生成的连续扩散大型模型，通过为类别特征引入交叉熵损失并给出首个理论证明、设计针对稀有子群体的评估指标以及泛化条件生成技术，在混合数值与类别数据上实现了强大的生成质量。

    

    条件生成是大型表格模型所需支持的一项关键能力，能够支撑众多应用。准确的条件生成必须应对两个关键挑战：(i) 数据由数值型与类别型条目混合组成，后者通常具有高基数且类别比例严重不平衡；(ii) 对于稀有子群体，条件生成变得越来越困难。此类模型中表现最佳者采用连续扩散与离散扩散相结合的混合方法。在本文中，我们设计了一个具有强大生成质量的连续扩散模型。其中一个重要元素是对类别特征使用交叉熵损失，我们为此提供了首个理论证明。我们引入了专注于稀有子群体的评估指标。此外，我们对条件生成技术进行了泛化，以支持新类型的条件控制。最后，我们展示了我们的模型（在劳动力数据上训练）能够捕捉经典的洞察

    arXiv:2610.30472v1 Announce Type: new  Abstract: Conditional generation is a key capability to be supported by Large Tabular Models, enabling a number of applications. Accurate conditional generation has to contend with two key challenges: $(i)$ Data comprise a mixture of numerical and categorical entries, the latter often with high cardinality and heavily unbalanced proportions; $(ii)$ Conditional generation becomes increasingly difficult for rare sub-populations. The best models in this class use a hybrid of continuous and discrete diffusions. In this paper, we design a continuous diffusion model with strong generation quality. An important element is the use of cross-entropy loss for categoricals, for which we provide a first theoretical justification. We introduce metrics that focus on rare subpopulations. Further, we generalize conditional generation techniques to enable new types of conditioning. Finally, we show that our model (trained on labor data) can capture classical insigh
    
[^159]: 学会激励：利用强化学习实现可持续城市出行

    Learning to Incentivise: Using Reinforcement Learning for Sustainable Urban Mobility

    [https://arxiv.org/abs/2610.30469](https://arxiv.org/abs/2610.30469)

    本文提出一种多智能体强化学习方法，通过动态向驾驶员发放激励引导其改道，在可持续城市出行框架下同时将交通网络的总出行时间降低16%并减少总排放量。

    

    交通管理传统上一直专注于基于收费和道路定价的解决方案。然而，道路收费常常引发关于可及性和公众不满的担忧，导致其在某些地区（如芬兰）被禁止。本研究优化了对驾驶员的动态激励分配，鼓励他们改道至替代（可能更长的）路径，以实现更大的社会效益，即降低交通网络中的总出行时间（TTT）和总排放量，为可持续城市出行做出贡献。我们采用多智能体强化学习方法，动态地为驾驶员分配激励，以同时降低总出行时间和排放量，其中出行时间通过交通仿真软件进行估算。我们证明，在预算不受限制且以最小化出行时间为目标的情况下，该激励方案相比预算相当的动态用户均衡（UE）方案可将总出行时间降低16%……（摘要原文在此处被截断）

    arXiv:2610.30469v1 Announce Type: cross  Abstract: Traffic management has traditionally focused on toll-based and road pricing solutions. However, road pricing often raises concerns about accessibility and public dissatisfaction, leading to its prohibition in some places, such as Finland. This study optimises the dynamic allocation of incentives to drivers, encouraging them to reroute onto alternative (potentially longer) paths to achieve greater societal benefit, reduced total travel time (TTT) and total emissions in the transportation network, contributing to sustainable urban mobility. We employ a multi-agent reinforcement learning approach to dynamically assign incentives to drivers to reduce both TTT and emissions, with travel times estimated using traffic simulation software. We demonstrate that, with an unlimited budget and an objective of minimising travel time, the incentive scheme reduces TTT by 16%, compared to the dynamic User-Equilibrium (UE) with a budget equivalent to ab
    
[^160]: 面向非平稳注意力的基于流的谱核学习

    Flow-based Spectral Kernel Learning for Nonstationary Attention

    [https://arxiv.org/abs/2610.30465](https://arxiv.org/abs/2610.30465)

    提出基于流的谱核注意力（FSKA），利用归一化流参数化非平稳的二元谱密度来学习灵活且可正则化的注意力核，同时支持移除查询-键投影的简化设计，实现参数效率更高的线性注意力架构。

    

    核化注意力为理解自注意力作为一种基于相似性的聚合机制提供了有用的视角，但现有方法通常依赖于固定或平稳的核，这限制了它们将相似性结构自适应地调整以应对复杂词元交互的能力。在本文中，我们提出了基于流的谱核注意力，这是一个用于学习富有表现力的非平稳注意力核的框架。FSKA 通过谱密度学习来构建核化注意力，并使用归一化流对所得的二元谱密度进行参数化。这种设计实现了灵活且可正则化的核学习，同时保持与可扩展的线性注意力计算的兼容性。我们进一步证明，富有表现力的核学习使得一种简化的注意力设计成为可能，即移除查询-键投影并使用固定的正交值投影，从而得到参数效率更高的架构。

    arXiv:2610.30465v1 Announce Type: cross  Abstract: Kernelized attention provides a useful perspective for understanding self-attention as a similarity-based aggregation mechanism, but existing approaches often rely on fixed or stationary kernels, limiting their ability to adapt the similarity structure to complex token interactions. In this paper, we propose flow-based spectral kernel attention (FSKA), a framework for learning expressive nonstationary attention kernels. FSKA formulates kernelized attention through spectral density learning and parameterizes the resulting bivariate spectral density with normalizing flows. This design enables flexible and regularizable kernel learning while remaining compatible with scalable linear-attention computation. We further show that expressive kernel learning enables a simplified attention design with removed query--key projections and fixed orthogonal value projections, resulting in a more parameter-efficient architecture. Controlled experiment
    
[^161]: 从短到长：面向长时程价值学习的分离特征

    Short-to-Long: Detached Features for Long-Horizon Value Learning

    [https://arxiv.org/abs/2610.30461](https://arxiv.org/abs/2610.30461)

    提出S2L方法，通过梯度隔离的分离连接，让较小折扣因子的源网络为长时程行为学习器提供特征，在不改变控制目标的前提下实现长时程价值学习，并在Atari与连续控制基准上取得整体性能提升。

    

    在深度强化学习中，如何从持续变化的价值目标中学习有用的表示仍然是一个挑战。我们研究了这种监督信号的时间视野：较小的折扣因子会使这些价值目标对远处的自举变化不那么敏感，但若直接用它们来驱动行为，则会改变控制目标。我们提出了Short-to-Long（S2L）方法，它为长时程行为学习器提供由较小折扣因子的源网络训练得到的特征。分离连接防止行为价值损失反向更新源网络，使每个源网络可以遵循自身的价值目标，同时行为折扣因子保持不变。在Atari、在线连续控制和离线连续控制任务上，该方法带来了总体性能提升，在57个Atari游戏中有40个提高了配对终端中位数。匹配条件的Atari消融实验验证了特征迁移、较小的源折扣因子以及梯度隔离各自的作用。

    arXiv:2610.30461v1 Announce Type: new  Abstract: Learning useful representations from continually changing value targets remains a challenge in deep reinforcement learning. We study the temporal horizon of this supervision: smaller discounts make these targets less sensitive to distant bootstrap changes, but using them for behavior changes the control objective. We introduce Short-to-Long (S2L), which supplies a long-horizon behavior learner with features trained by smaller-discount source networks. Detached connections prevent the behavior value loss from updating the sources, allowing each source to follow its own value objective while the behavior discount remains unchanged. Across Atari, online continuous control, and offline continuous control, the approach yields aggregate performance gains, improving the paired terminal median on 40 of 57 Atari games. Matched Atari ablations support the roles of feature transfer, smaller source discounts, and gradient isolation. Fixed-trajectory
    
[^162]: aggmetrics：面向机器学习的分类性能度量聚合

    aggmetrics: Aggregating Classification Performance Measures for Machine Learning

    [https://arxiv.org/abs/2610.30456](https://arxiv.org/abs/2610.30456)

    aggmetrics 是一个仅依赖 NumPy 的 Python 包，通过四种聚合算子将任何基于混淆矩阵的二分类性能度量推广到多分类、多标签、序数、代价敏感、软标签和交叉验证等场景，并利用度量的结构性质在其使用不当时发出警告。

    

    aggmetrics 是一个 Python 包，它通过用四种聚合算子之一替代各类库中通常以“平均”开关形式暴露的功能，将任何以混淆矩阵四个计数表示的二分类性能度量扩展到多分类、多标签、序数、代价敏感、软标签以及交叉验证等设置。该包还为每个度量存储了决定哪些恒等式适用于该度量的结构性质，能够从用户提供的公式推断这些性质，并利用它们在度量被用于其行为不符合通常假设的场景时发出警告。其测试套件将配套理论的结果编码为可执行测试。该包采用 BSD-3 许可证，仅依赖 NumPy，可在 https://github.com/rcprati/aggmetrics 获取，文档见 https://aggmetrics.readthedocs.io。

    arXiv:2610.30456v1 Announce Type: new  Abstract: aggmetrics is a Python package that extends any binary performance measure expressed in the four confusion-matrix counts to multiclass, multilabel, ordinal, cost-sensitive, soft-label and cross-validation settings, by substituting one of four aggregation operators for what libraries usually expose as an average switch. The package also stores, for every measure, the structural properties that decide which identities apply to it, can infer those properties from a user-supplied formula, and uses them to warn when a measure is used where it does not behave as commonly assumed. Its test suite encodes the results of the companion theory as executable tests. The package is BSD-3 licensed, depends only on NumPy, and is available at https://github.com/rcprati/aggmetrics; documentation at https://aggmetrics.readthedocs.io.
    
[^163]: 相同信念，不同策略：学习可复用表示以实现LLM推理的提前停止

    Same Beliefs, Different Policies: Learning Reusable Representations for Early Stopping in LLM Reasoning

    [https://arxiv.org/abs/2610.30450](https://arxiv.org/abs/2610.30450)

    该论文提出BELT方法，通过将结果预测与停止控制分离，从冻结LLM隐藏状态中学习与token价格无关的信念表示，使单一探针能通过轻量级策略重拟合或阈值调整支持多种准确率-成本权衡，从而有效实现推理提前停止并降低计算浪费。

    

    推理型大语言模型（LLM）在得出正确答案后往往还会继续推理，这不仅浪费计算资源，有时还会用错误答案取代原本正确的答案。决定何时停止是一个元推理问题：模型必须在已知自己当前已能回答什么的前提下，权衡进一步推理可能带来的收益与其成本。直接效用训练方法在选定的token价格下学习停止表示；要支持新的价格，可能需要对控制器进行额外的训练。我们提出了BELT（基于信念的终止机制，BELief-based Termination），它将结果预测与停止控制相分离。该方法从冻结LLM的隐藏状态中学习关于当前答案和未来答案的信念，然后在这些冻结特征上拟合一个线性停止策略。由于预测目标与token价格无关，单一探针即可通过CPU上的策略重拟合或阈值调整，支持不同的准确率-成本权衡。在五个分布外（OOD）数学基准上的平均结果显示，BELT减少了生成（计算开销），同时保持了答案准确率。

    arXiv:2610.30450v1 Announce Type: cross  Abstract: Reasoning LLMs often continue reasoning after reaching a correct answer, wasting computation and sometimes replacing it with an incorrect one. Deciding when to stop is a metareasoning problem: the model must weigh what further reasoning could achieve against its cost, given what it can already answer. Direct utility training learns a stopping representation at a chosen token price; supporting new prices can require additional controller training. We introduce BELT (BELief-based Termination), which separates outcome prediction from stopping control. It learns beliefs about current and future answers from a frozen LLM's hidden states, then fits a linear stopping policy on the frozen features. The prediction targets are independent of token price, so one probe supports different accuracy-cost trade-offs through CPU policy refitting or threshold adjustment. Averaged over five out-of-distribution (OOD) math benchmarks, BELT reduces generate
    
[^164]: 结构扰动下符号图的稳定性分析

    Stability Analysis of Signed Graphs under Structural Perturbations

    [https://arxiv.org/abs/2610.30444](https://arxiv.org/abs/2610.30444)

    本文首次系统研究了符号图滤波器在边添加、删除和符号翻转等结构扰动下的稳定性，揭示了符号边扰动与节点特征相关性对学习表示稳定性的联合决定作用，并据此提出了一种理论引导的扰动策略来选择结构性影响显著的符号边修改。

    

    符号图学习已成为建模同时包含正负交互关系的网络的强大框架。尽管基于图滤波器的方法在符号网络上取得了令人满意的实证性能，但其在边扰动下的稳定性仍缺乏充分的理解。现有的稳定性分析主要针对无符号图，并未考虑符号翻转扰动。本文研究了符号图滤波器在边添加、边删除和符号翻转下的稳定性。我们的理论分析揭示了符号边扰动与节点特征相关性如何共同决定学习表示的稳定性。基于这一分析，我们提出了一种理论引导的扰动策略，用于选择具有结构性影响力的符号边修改操作。在合成和真实符号图上的实验表明，所提出的方法能够持续引发更大的嵌入漂移和极性变化（摘要原文在此处截断）。

    arXiv:2610.30444v1 Announce Type: new  Abstract: Signed graph learning has emerged as a powerful framework for modeling networks with both positive and negative interactions. While graph-filter-based methods have achieved promising empirical performance on signed networks, their stability under edge perturbations remains less well understood. Existing stability analyses primarily focus on unsigned graphs and do not account for sign-flip perturbations. In this paper, we study the stability of signed graph filters under edge additions, deletions, and sign flips. Our theoretical analysis reveals how signed edge perturbations and node feature correlations jointly determine the learned representation stability. Based on this analysis, we propose a theory-guided perturbation strategy that selects structurally influential signed-edge modifications. Experiments on synthetic and real signed graphs show that the proposed method consistently induces larger embedding drift, polarity change, and do
    
[^165]: 迈向忠实的数据选择：统计极限与可扩展算法

    Towards Faithful Data Selection: Statistical Limits and Scalable Algorithms

    [https://arxiv.org/abs/2610.30442](https://arxiv.org/abs/2610.30442)

    本文证明了在缺乏远大于训练集的验证集时实现最优数据选择在信息论上是不可能的，并据此提出“忠实性”这一新目标，通过可估计的代理量上界子集训练与全量训练的差异，并利用未调整朗之万算法和轨迹累积梯度差异对参数分布的KL散度进行约束。

    

    什么才是最优的数据选择算法？传统上，最优性被定义为找到一个子集，使得仅在该子集上训练的分类器的总体误差最小化。本工作的出发点是首先证明，除非数据选择算法能够访问比训练集数量级更大的验证集，否则这一目标在信息论上是不可能实现的。这一不可能性结果促使我们从无法验证的总体风险最小化转向一种单边保证，我们称之为“忠实性”：对于任何候选子集，我们提供一个可估计的代理量，该代理量给出在全量训练数据上训练与仅在子集上训练之间差异的上界。我们使用未调整朗之万算法对训练过程进行建模，并通过轨迹累积梯度差异来约束子集训练与全量数据训练所得参数分布之间的Kullback--Leibler散度。通过最小化……（摘要在此处截断）

    arXiv:2610.30442v1 Announce Type: new  Abstract: What does it mean to have an optimal data selection algorithm? Optimality is conventionally framed as finding a subset that minimizes the population error of the classifier trained only on that subset. Our starting point in this work is first to show that this goal is information-theoretically impossible unless the data selection algorithm has access to a validation set order-wise larger than the training set. This impossibility result motivates shifting from unverifiable population risk minimization to a one-sided guarantee that we refer to as faithfulness: for any candidate subset, we provide an estimable proxy that upper-bounds the difference between training on the full training data and on only the subset. We model training with the Unadjusted Langevin Algorithm and bound the Kullback--Leibler divergence between subset-trained and full-data-trained parameter distributions by the trajectory-accumulated gradient discrepancy. Minimizin
    
[^166]: 基于扰动一致性与置信度校准的记忆增强在线测试时自适应用于脑电运动想象

    Memory-Augmented Online Test-Time Adaptation for EEG Motor Imagery via Perturbation Consistency and Confidence Calibration

    [https://arxiv.org/abs/2610.30422](https://arxiv.org/abs/2610.30422)

    该论文提出MOPC，一个无源的记忆增强在线测试时自适应框架，通过扰动一致性与置信度校准，使EEG脑机接口在推理阶段实时更新解码参数，无需访问源数据即可提升跨受试者场景下的可靠性。

    

    近年来，基于脑电图（EEG）的脑机接口（BCI）作为一种便携式神经测量系统日益受到关注。然而，这类仪器的实际部署往往受到非平稳性和跨受试者分布偏移的阻碍，导致显著的受试者间不一致性。现有的自适应方法在实时测量场景下仍然不切实际，因为它们通常需要对源数据或聚合目标数据进行离线访问。为应对这些挑战，本文提出了一种结合扰动一致性与置信度校准的记忆增强在线测试时自适应方法（MOPC），这是一个旨在提升BCI仪器可靠性的无源（source-free）框架。MOPC使测量系统能够在推理阶段即时更新其解码参数，实现单次遍历的自适应，而无需访问敏感的源数据。我们引入了随机掩码……（摘要内容不完整）

    arXiv:2610.30422v1 Announce Type: new  Abstract: Recently, electroencephalography (EEG)-based brain--computer interfaces (BCIs) have gained traction as portable neural measurement systems. However, the practical deployment of these instruments is often hindered by nonstationarity and cross-subject distribution shifts, which lead to significant intersubject inconsistency. Existing adaptation methods remain impractical for real-time measurement scenarios, as they typically require offline access to source or aggregated target data. To address these challenges, this article proposes memory-augmented online test-time adaptation (OTTA) with perturbation consistency and confidence calibration (MOPC), a source-free framework designed to enhance the reliability of BCI instrumentation. MOPC enables the measurement system to update its decoding parameters on the fly during the inference stage, facilitating single-pass adaptation without accessing sensitive source data. We introduce random-mask p
    
[^167]: 来自嘈杂、多样化来源的高价值训练数据

    High-Value Training Data from Noisy, Diverse Sources

    [https://arxiv.org/abs/2610.30421](https://arxiv.org/abs/2610.30421)

    提出一种基于元梯度的可扩展数据加权方法，通过学习可泛化、可迁移的参数化加权规则，为奖励建模等任务从嘈杂多样的数据源中筛选出高价值训练数据。

    

    训练最先进的模型越来越需要从大型、异构的数据集中学习，这些数据集中的样本在质量、领域相关性和监督来源上各不相同。数据加权——即为训练样本分配不同权重——提供了一种控制此类数据如何影响训练的通用方法，但在大规模下学习有效的权重仍然具有挑战性。我们开发了一种可扩展的数据权重学习方法：给定一个小型、高质量的目标数据集，我们在整个训练过程中使用元梯度（metagradients），直接针对目标域的性能高效地优化训练样本的权重。为了超越固定的数据集和模型进行扩展，我们学习参数化的加权规则，这些规则可以泛化到新样本和流式样本，并能从小模型迁移到规模大得多的模型。我们研究了奖励建模作为典型的应用场景，其中偏好数据可能是嘈杂的、对抗性的、合成的，或与目标域不匹配的。

    arXiv:2610.30421v1 Announce Type: cross  Abstract: Training state-of-the-art models increasingly requires learning from large, heterogeneous datasets whose examples vary in quality, domain relevance, and source of supervision. Dataweighting---assigning different weights to training examples---offers a general way to control how such data influences training, but learning effective weights at scale remains challenging. We develop a scalable approach to learning dataweights: given a small, high-quality target dataset, we use metagradients through the training process to efficiently optimize training-example weights directly for performance on the target domain. To scale beyond a fixed dataset and model, we learn parametric weighting rules that generalize to new and streaming examples and transfer from smaller models to substantially larger ones. We study reward modeling as a prototypical application, where preference data may be noisy, adversarial, synthetic, or mismatched with the targe
    
[^168]: AcoustiPrint：物理引导的可制造声学结构神经设计

    AcoustiPrint: Physics-Guided Neural Design of Manufacturable Acoustic Structures

    [https://arxiv.org/abs/2610.30411](https://arxiv.org/abs/2610.30411)

    AcoustiPrint提出了一种物理信息神经架构，通过隐式神经表示三维几何并引入可打印性算子，在声学结构逆设计中确保优化结果直接满足增材制造的离散约束，弥合了物理优化几何与实际制造结果之间的差距。

    

    近年来，增材制造与神经逆设计的进展使得创建复杂三维结构来操控物理场成为可能，然而这些设计还必须遵守离散的、不可微分的制造约束。当前的方法通过施加可微分的软可制造性损失惩罚来限制设计空间，或通过后处理来固定结构，导致物理模型优化的几何形状与最终制造结果之间存在偏差。因此，我们提出了AcoustiPrint，这是一种用于约束逆设计的物理信息神经架构，它采用隐式神经表示（INR）来表示三维几何形状，并在评估最终响应之前让每个候选设计通过一个可打印性算子。该可打印性算子执行逐层自支撑、离散材料选择和整体连通性检查，从而确保制造可行性。

    arXiv:2610.30411v1 Announce Type: new  Abstract: Recent work in additive manufacturing and neural inverse design enables the creation of complex three-dimensional structures to manipulate physical fields, yet these designs must also adhere to discrete, non-differentiable fabrication constraints. Current methods limit the design space by imposing differentiable soft manufacturability loss penalties, or by fixing structures through post-processing, causing a discrepancy between the geometry optimized by the physics model and the final fabricated result. Thus, we present AcoustiPrint, a physics-informed neural architecture for constrained inverse design that represents three-dimensional geometries with an implicit neural representation (INR) and passes each candidate through a printability operator before evaluating its final response. The printability operator executes layer-wise self-support, discrete material selection, and whole-body connectivity, assuring fabrication feasibility. To 
    
[^169]: 逃离系统发育陷阱：持续同调助力多模态图神经网络实现跨物种离子通道分类

    Escaping the Phylogenetic Trap: Persistent Homology Enables Cross-Species Ion Channel Classification via Multimodal Graph Neural Networks

    [https://arxiv.org/abs/2610.30394](https://arxiv.org/abs/2610.30394)

    该研究揭示随机数据划分导致的系统发育泄漏会高估毒素离子通道分类器性能，并提出融合ESM-2序列嵌入与二硫键感知图注意力网络的多模态框架，证明持续同调等结构归纳偏置能在留一分类群评估下实现真正的跨物种泛化。

    

    蛋白质分类器对评估协议高度敏感：随机划分训练/测试集会导致系统发育泄漏，使模型利用特定分类群的相关性，从而显著高估分布外性能。这一问题在按离子通道靶点（Ca²⁺、K⁺、Na⁺）对毒素化合物进行分类的任务中尤为突出，文献中报告的80–90%准确率并不能反映模型对未见分类群的泛化能力。我们将跨物种预测构建为一种结构化分布偏移问题，其中分类学身份充当混杂因素，因此需要能够捕捉跨系统发育不变的结构特征的表示方法。我们采用一个共享的多模态骨干网络，将ESM-2序列嵌入与二硫键感知的图注意力网络相融合，并在留一分类群评估协议下系统比较了五种结构归纳偏置（谱方法、黎曼几何、SE(3)-不变、统计流形以及持续同调表示）……

    arXiv:2610.30394v1 Announce Type: new  Abstract: Protein classifiers are highly sensitive to evaluation protocol: random train/test splits induce phylogenetic leakage, allowing models to exploit taxon-specific correlations and substantially overestimate out-of-distribution performance. This issue is particularly acute for venom compound classification by ion-channel target (Ca2+, K+, Na+), where reported accuracies of 80-90% do not reflect generalization to unseen taxa. We formulate cross-species prediction as a structured distribution shift in which taxonomic identity acts as a confounder, requiring representations that capture structural features invariant across phylogeny. Using a shared multimodal backbone that fuses ESM-2 sequence embeddings with a disulfide-aware graph attention network, we systematically compare five structural inductive biases (spectral, Riemannian, SE(3)-invariant, statistical-manifold, and persistent-homology representations) under a leave-one-taxon-out evalu
    
[^170]: 面向弱监督神经符号学习的潜在概念后验软本体先验方法

    Soft Ontology Priors over Latent Concept Posteriors for Weakly Supervised Neuro-Symbolic Learning

    [https://arxiv.org/abs/2610.30391](https://arxiv.org/abs/2610.30391)

    该论文提出将潜在概念的软本体先验注入弱监督神经符号学习的训练目标，并与任务约束一起软编译为概率电路，从而消除推理捷径、使概念可辨识、对噪声专家公理具有鲁棒性，同时实现与编译电路规模呈线性关系的推理复杂度。

    

    仅监督任务标签的神经符号预测器存在推理捷径问题：不同的概念映射可以达到相同的最大似然，导致恢复出的概念不可辨识，对下游复用而言是不安全的。为攻克这一失效模式，我们在训练目标中注入关于潜在概念的软本体先验，并将其与任务约束一起编译为平滑、可分解的概率电路。在独立伯努利感知的假设下，所得结构化概念后验的归一化因子可化简为加权模型计数，其计算复杂度与编译后电路的规模呈线性关系，因为神经叶子权重会在求值之前被折叠进电路中。该软编译策略将专家公理置于电路的权重之中而非其支撑集之中（由数据生成过程所保证的任务结构公理仍保持硬约束），因此带噪声的专家公理只会重新分配先验概率质量，而不会……（原文摘要在此处截断）

    arXiv:2610.30391v1 Announce Type: cross  Abstract: Neuro-symbolic (NeSy) predictors that supervise only the task label suffer from reasoning shortcuts: distinct concept mappings achieve the same maximum likelihood, leaving the recovered concepts unidentifiable and unsafe for downstream reuse. We attack this failure by injecting a soft ontology prior over the latent concepts into the training objective and compiling it together with the task constraint into a smooth, decomposable probabilistic circuit. Under independent Bernoulli perception, the normalisers of the resulting structured concept posterior reduce to weighted model counts that are linear in the compiled circuit size, because the neural leaf weights fold into the circuit before evaluation. The soft compilation strategy places expert axioms in the circuit's weights rather than its support (task-structural axioms guaranteed by the data-generating process stay hard), so noisy expert axioms only redistribute prior mass and cannot
    
[^171]: 一种基于Transformer的可变初始/最终条件下实时多智能体路径规划方法

    A Transformer-based Approach for Real-time Multi-agent Path Planning with Varying Initial/Final Conditions

    [https://arxiv.org/abs/2610.30388](https://arxiv.org/abs/2610.30388)

    提出OCFormer，一种通过物理信息损失训练的基于Transformer的神经网络，将解耦潜空间映射到耦合物理空间，从而实现具有可变初始/最终条件的实时高维多智能体路径规划。

    

    我们研究了在给定初始和最终构型下的实时多智能体路径规划问题，将其表述为一个受智能体间碰撞避免和障碍物约束的高维最优控制问题。基于Hamilton-Jacobi-Bellman方程或庞特里亚金最大值原理（PMP）的经典求解方法，随着智能体数量的增加，由于维数灾难以及交互约束带来的非凸性，在计算上变得难以处理。我们提出OCFormer——一种基于Transformer的神经网络，它参数化了一个坐标变换的逆变换，将一个简单的、可解析处理的潜空间（其中各智能体在解耦的哈密顿量下独立演化）映射到耦合交互系统的物理空间。该模型通过最小化物理信息驱动的损失函数进行训练，该损失包括来自PMP的哈密顿常微分方程残差项和边界匹配项。

    arXiv:2610.30388v1 Announce Type: cross  Abstract: We address the problem of real-time multi-agent path planning under prescribed initial and final configurations, formulated as a high-dimensional optimal control problem subject to inter-agent collision avoidance and obstacle constraints. Classical solvers based on the Hamilton--Jacobi--Bellman equation or Pontryagin's maximum principle (PMP) become computationally intractable as the number of agents grows, due to the curse of dimensionality and the non-convexity introduced by interaction constraints. We propose OCFormer, a transformer-based neural network that parameterizes the inverse of a coordinate transformation mapping a simple, analytically tractable latent space, in which agents evolve independently under a decoupled Hamiltonian, to the physical space of the coupled, interacting system. The model is trained by minimizing a physics-informed loss comprising the Hamiltonian ODE residual from the PMP and a boundary matching term, w
    
[^172]: 带弃权机制的贝叶斯固定置信度最优臂识别

    Bayesian Fixed-Confidence Best-Arm Identification with Abstention

    [https://arxiv.org/abs/2610.30384](https://arxiv.org/abs/2610.30384)

    该论文提出了带弃权机制的贝叶斯固定置信度最优臂识别算法S&C，证明了其期望停止时间在高斯设置下达到匹配的上下界，揭示近乎平局的实例主导采样成本，并在无弃权时达到最优的 Θ(1/δ) 依赖。

    

    我们研究了贝叶斯固定置信度最优臂识别问题，其中学习者可以选择放弃做出最终推荐。在目标错误概率 $\delta$ 和弃权预算 $\alpha$ 的约束下，我们提出了筛选与确认（Screen-and-Confirm, S&C）算法并分析其期望停止时间。我们进一步推导了满足这些约束的任何算法的最小期望停止时间。对于高斯先验和奖励，当 $(\delta,\alpha)\downarrow(0,0)$ 时，我们建立了在乘性常数意义下匹配的上下界，其速率为 $\kappa_\nu^2[1+\log((\alpha+\delta)/\delta)]/(\alpha+\delta)$，其中 $\kappa_\nu$ 是前两名臂差距在零处的先验密度，这突出表明近乎平局的实例主导了采样成本。当不允许弃权（$\alpha=0$）时，S&C算法在固定模型参数下达到了最优的 $\Theta(1/\delta)$ 依赖关系，去除了上界中额外的 $\log(1/\delta)$ 因子。

    arXiv:2610.30384v1 Announce Type: new  Abstract: We study the Bayesian fixed-confidence best-arm identification problem in which a learner can abstain from making a terminal recommendation. Subject to a target error probability $\delta$ and an abstention budget $\alpha$, we propose Screen-and-Confirm (S&C) and analyze its expected stopping time. We further derive the minimum expected stopping time for any algorithm that satisfies these constraints. For Gaussian priors and rewards, as $(\delta,\alpha)\downarrow(0,0)$, we establish matching upper and lower bounds up to multiplicative constants, with rate $\kappa_\nu^2[1+\log((\alpha+\delta)/\delta)]/(\alpha+\delta)$, where $\kappa_\nu$ is the prior density of the top-two gap at zero, highlighting that nearly tied instances govern the sampling cost. When abstention is disallowed ($\alpha=0$), S&C yields the optimal $\Theta(1/\delta)$ dependence for fixed model parameters, removing the additional $\log(1/\delta)$ factor in the upper bound 
    
[^173]: 大语言模型中安全相关行为的探测与引导

    Probing and Steering for Safety-Relevant Behaviours in Large Language Models

    [https://arxiv.org/abs/2610.30363](https://arxiv.org/abs/2610.30363)

    该论文在统一流水线中系统探究了激活探针设计选择（估计器类型、token 池化方案、层选择）在检测与引导欺骗、高风险意识和谄媚三种安全相关行为时的相互作用，并验证了“逻辑回归最擅长检测、均值差方向最擅长引导”的读/写分离现象。

    

    激活探针是在语言模型残差流中计算出的方向，既可用于检测安全相关行为，也可通过将其方向加回残差流来引导这些行为。我们在同一条流水线、同一个模型（Llama-3.1-8B-Instruct）上，针对三种行为（欺骗、高风险意识与谄媚性），探究此类探针背后的主要设计选择之间如何相互作用。该流水线将两种估计器（均值差与逻辑回归）与三种 token 池化方案（末位、均值与可学习的注意力）进行交叉组合，为检测或引导某一行为选择合适的层，并对每个方向进行双重评估：既作为该行为的检测器，也作为将输出推向该行为的引导向量。综合此前的研究可以得出一种“读/写分离”的结论，即逻辑回归在读取（检测）时效果最好，而均值差方向在写入（引导）时效果最好。我们发现这一规律在欺骗行为和（摘要在此处被截断）

    arXiv:2610.30363v1 Announce Type: cross  Abstract: Activation probes are computed directions in a language model's residual stream, used both to detect safety-relevant behaviours and, by adding their direction back into the stream, to steer them. We explore how the main design choices behind such probes interact, within one pipeline, on one model (Llama-3.1-8B-Instruct), and across three behaviours: deception, high-stakes awareness and sycophancy. The pipeline crosses two estimators (difference-of-means and logistic regression) with three token-pooling schemes (last, mean and learned attention), selects appropriate layers to detect or steer a behaviour, and evaluates every direction twice: as a detector of the behaviour, and as a steering vector to push outputs towards the behaviour. Prior work read together suggests a read/write dissociation, in which logistic regression reads (detects) best but difference-of-means directions write (steer) best. We find that it holds for deception and
    
[^174]: CTRL-FM：长时节点中断下基于流的虚拟感知的推理时结构控制

    CTRL-FM: Inference-Time Structural Control for Flow-Based Virtual Sensing under Long-Horizon Node Outages

    [https://arxiv.org/abs/2610.30362](https://arxiv.org/abs/2610.30362)

    CTRL-FM提出了一个观测保持的流匹配框架，通过从存活数据构建自监督代理中断，在推理时自适应选择结构控制策略，以解决长时节点中断下虚拟感知中固定结构偏置导致的过平滑或误差传播问题。

    

    长时间节点中断比传统的缺失值插补更为困难，因为完整的传感器轨迹在较长时间内处于未观测状态，而空间和时间依赖性的有用性会随中断位置和运行工况而变化。现有的基于图、扩散和流的方法通常施加固定的结构偏置，这可能会过度平滑突变或从不可靠的邻居传播错误。我们提出了CTRL-FM，一个保持观测数据的流匹配框架，它将结构引导视为以中断情况为条件的推理时决策。一个仅在缺失部分进行学习的流匹配主干网络学习条件速度场，同时将观测到的测量值严格保持固定。对于每次实际发生的中断，CTRL-FM从存活的测量数据中构建自监督代理中断，并利用其已知值在无控制、图主导、时间主导和平衡控制策略之间进行选择。

    arXiv:2610.30362v1 Announce Type: new  Abstract: Long-horizon node outages are harder than conventional missing-value imputation because complete sensor trajectories remain unobserved for extended periods, while the usefulness of spatial and temporal dependencies varies across outage locations and operating regimes. Existing graph, diffusion, and flow-based methods typically impose a fixed structural bias, which can oversmooth abrupt changes or propagate errors from unreliable neighbors. We introduce CTRL-FM, an observation-preserving flow-matching framework that treats structural guidance as an outage-conditioned inference-time decision. A missing-only flow-matching backbone learns a conditional velocity field while keeping observed measurements exactly fixed. For each realized outage, CTRL-FM constructs self-supervised proxy outages from surviving measurements and uses their known values to select among uncontrolled, graph-dominant, temporal-dominant, and balanced control policies. T
    
[^175]: 先路由后生成：面向大语言模型推理路由的KL正则化路由器改进方法

    Route-and-Generate: KL-Regularized Router Improvement for Reasoning Routing in LLMs

    [https://arxiv.org/abs/2610.30351](https://arxiv.org/abs/2610.30351)

    本文提出Rouge分层训练框架，通过KL正则化的路由策略自适应分配推理预算，使大语言模型在简单问题上简洁作答、在难题上进行深入推理，从而优化准确率与效率之间的权衡。

    

    大语言模型常常不加区分地应用复杂推理，导致在简单任务上过度思考，而在较难任务上思考不足。我们提出了Route-and-Generate（Rouge），这是一种分层训练框架，使大语言模型能够自适应地决定每个输入何时需要推理以及需要多少推理。我们的方法通过引入路由策略将推理力度的分配与答案生成解耦，该路由策略在多种推理格式之间分配固定的采样预算，同时基础模型使用基于逐样本奖励的GRPO进行优化。为促进高效推理，我们采用基于排名的奖励机制，偏好更短的正确解法，并通过闭式KL正则化策略改进步骤来更新路由策略，将其蒸馏到神经路由器中。这种设计鼓励在简单问题上给出简洁答案，同时为更难的问题保留更长的推理过程，从而在多个基准上实现了改进的准确率-效率权衡。

    arXiv:2610.30351v1 Announce Type: cross  Abstract: Large language models often apply complex reasoning indiscriminately, leading to overthinking on simple tasks and underthinking on harder ones. We present Route-and-Generate (Rouge), a hierarchical training framework that enables LLMs to adaptively decide when and how much reasoning is required for each input. Our approach decouples reasoning effort allocation from answer generation by introducing a routing policy that allocates a fixed sampling budget across multiple reasoning formats, while the base model is optimized with GRPO using per-sample rewards. To promote efficient reasoning, we employ a rank-based reward that favors shorter correct solutions, and update the routing policy via a closed-form KL-regularized policy improvement step distilled into a neural router. This design encourages concise answers on easy problems while reserving longer reasoning for harder ones, achieving improved accuracy-efficiency trade-offs across mult
    
[^176]: 一致性并非有效性：自我奖励的道德论证使语言模型变得果断但易受影响

    Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable

    [https://arxiv.org/abs/2610.30349](https://arxiv.org/abs/2610.30349)

    该论文提出一种自我奖励的强化学习方法，让模型通过自我评判道德论证的质量来提升道德判断能力，使模型变得更加果断且不受选项呈现顺序的影响，但谄媚性和重新框架问题仍然存在。

    

    大语言模型越来越多地被用于道德判断咨询。我们旨在通过奖励好的论证来提升此类任务的性能，其前提是论证得当能够提高正确判断的可能性。我们从一个最先进的语言模型初始化一个强化学习策略，在一种掩蔽思维的自我评判锦标赛中进行训练：由策略本身对生成的论证就论证质量进行两两评判。作为实证基础，我们从经典小说中提取真实的道德困境，并使用逻辑求解器验证每个困境确实构成冲突。我们证明，我们的评判机制对论证质量的评分与人类标注者一致，策略随后会采用它所奖励的语言风格。在MoralChoice、ELEPHANT和MORABLES等外部基准上的测试表明，该策略变得更加果断，且对选项呈现顺序保持不变，但谄媚性和重新框架问题仍然是对有效性的挑战。

    arXiv:2610.30349v1 Announce Type: new  Abstract: LLMs are increasingly consulted for moral judgment. We aim to improve performance on such tasks by rewarding good arguments, on the premise that arguing well increases the likelihood of judging well. We train a reinforcement learning policy initialized from a state-of-the-art language model in a self-judged tournament with masked thinking: the generated arguments are judged pairwise on argument quality by the policy itself. As an empirical basis, we extract authentic dilemmas from classic novels, using a logic solver to verify that each constitutes a conflict. We show that our judge rates argument quality like human annotators, and the policy then adopts the language register that it rewards. Testing on the external benchmarks MoralChoice, ELEPHANT, and MORABLES, we find that the policy becomes more decisive and invariant to the order in which options are presented, but sycophancy and reframing remain a challenge to validity. Two ablatio
    
[^177]: 关于采用展开式LDPC解码器的神经接收机对抗脆弱性的研究

    On the Adversarial Vulnerability of Neural Receivers with Unrolled LDPC Decoders

    [https://arxiv.org/abs/2610.30345](https://arxiv.org/abs/2610.30345)

    本文揭示了神经接收机中展开式LDPC解码器因boxplus运算的tanh饱和产生梯度掩蔽、使标准对抗攻击失效，并提出两种结构独立的新损失函数（L_BP与L_LLR）成功恢复了攻击效力。

    

    面向5G新空口（NR）的神经接收机在链路级性能上可与经典的基于模型的接收机相媲美，但其对作用于接收机输入的对抗扰动的鲁棒性仍然缺乏充分研究。我们表明，通过展开式低密度奇偶校验（LDPC）解码器端到端施加的标准基于梯度的攻击，无法显著优于等功率的高斯扰动。我们将这一失败归因于梯度掩蔽，其源于置信传播（BP）校验节点更新中boxplus运算的tanh饱和，在十次展开的BP迭代中，每次迭代的梯度范数衰减超过四个数量级。基于这一发现，我们提出了两种结构上相互独立的损失函数以恢复攻击的有效性：L_BP通过逐迭代监督将梯度引导至未饱和的早期迭代，而L_LLR是作用于解码前对数似然比（LLR）上的间隔损失，其梯度……（摘要原文在此处截断）

    arXiv:2610.30345v1 Announce Type: cross  Abstract: Neural receivers for 5G New Radio (NR) match classical model-based receivers in link-level performance, but their robustness to adversarial perturbations at the receiver input remains underexplored. We show that standard gradient-based attacks applied end-to-end through an unrolled low-density parity-check (LDPC) decoder fail to significantly outperform a matched-power Gaussian perturbation. We identify the failure as gradient masking caused by tanh saturation in the boxplus operation of the belief-propagation (BP) check-node update, with a per-iteration gradient norm decay of more than four orders of magnitude across ten unrolled BP iterations. Based on this finding, we propose two structurally independent loss functions that restore attack efficacy. L_BP routes the gradient through the unsaturated early iterations via per-iteration supervision, while L_LLR is a margin loss on pre-decoder log-likelihood ratios (LLRs) whose gradient by
    
[^178]: AQL-Lite：通过以数据为中心的持续预训练实现阿拉伯语大语言模型的高效知识获取

    AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining

    [https://arxiv.org/abs/2610.30344](https://arxiv.org/abs/2610.30344)

    AQL-Lite 通过质量优先的数据筛选流水线与部分参数持续预训练方案（RS-LoRA 适配器结合最后八层 Transformer 块与嵌入矩阵），以约一半的训练时间恢复了全参数 CPT 94% 的性能提升，高效地将多语言大模型适配至现代标准阿拉伯语。

    

    训练阿拉伯语大语言模型面临网络数据嘈杂、方言变体多样以及持续预训练（CPT）成本高昂等挑战。我们提出了 AQL-Lite（AQL：Arabic Quality Language，阿拉伯语质量语言），一个高效节省时间和资源的框架，用于将多语言大语言模型适配至现代标准阿拉伯语（MSA）。AQL-Lite 将以质量为先的数据筛选流水线（结构化清洗、启发式与基于模型的过滤、MSA 筛选以及去重）与部分参数 CPT 方案相结合：在整个网络中训练 RS-LoRA 适配器，同时训练最后八个 Transformer 块和嵌入矩阵，并采用逐层学习率。在匹配的 10 亿 token 预算下，使用 AQL 筛选数据进行 CPT 相比使用原始网络文本进行 CPT（后者相对基础模型并无提升），在四个阿拉伯语基准测试的平均分上提高了 3.51 分。在受控的 50 亿 token 对比实验中，AQL-Lite 仅用 52% 的实际训练时间（wall-clock time）就恢复了全参数 CPT 94% 的性能提升（平均分 73.19 对比 73.58）。

    arXiv:2610.30344v1 Announce Type: new  Abstract: Training Arabic large language models faces challenges from noisy web data, dialectal variation, and the high cost of continual pretraining (CPT). We introduce AQL-Lite (AQL: Arabic Quality Language), a time- and resource-efficient framework for adapting a multilingual LLM to Modern Standard Arabic (MSA). AQL-Lite couples a quality-first curation pipeline (structural cleaning, heuristic and model-based filtering, MSA selection, and deduplication) with a partial-parameter CPT recipe that trains RS-LoRA adapters across the network together with the last eight transformer blocks and the embedding matrix, under layer-wise learning rates. At a matched 1B-token budget, CPT on AQL-curated data raises a four-benchmark Arabic mean by 3.51 points over CPT on raw web text, which yields no gain over the base model. In a controlled 5B-token comparison, AQL-Lite recovers 94% of the gain of full-parameter CPT (mean 73.19 vs. 73.58) in 52% of the wall-c
    
[^179]: 面向视觉应用的脉冲感知基于梯度的脉冲神经网络混合精度量化

    Spike-Aware Gradient-Based Mixed-Precision Quantization of Spiking Neural Networks for Vision Applications

    [https://arxiv.org/abs/2610.30339](https://arxiv.org/abs/2610.30339)

    提出了一种脉冲感知的基于一阶梯度的混合精度量化框架，能够自适应地为脉冲神经网络分配位宽，避免了昂贵的二阶Hessian计算开销，从而在视觉应用的低精度部署中保持模型精度。

    

    近年来，脉冲神经网络因其事件驱动和稀疏计算能力，作为一种高能效的神经计算范式受到了广泛关注。然而，在实际部署所必需的低精度约束下保持精度，对于SNN模型而言在很大程度上尚未被探索。此外，现有的基于Hessian的量化方法（如HAWQ-V2）由于需要在多个SNN时间步上进行时间展开，会产生显著的内存和计算开销。我们提出了一种面向SNN的脉冲感知基于梯度的混合精度量化框架，该框架利用一阶梯度信息自适应地分配位宽，避免了代价高昂的二阶Hessian计算。我们在多种基于SNN的架构上评估了该方法，包括通过ANN-to-SNN转换和直接训练框架获得的VGG、ResNet和Vision Transformer模型，并在CIFAR10/100、ImageNet等数据集上进行了测试。

    arXiv:2610.30339v1 Announce Type: new  Abstract: In recent years, Spiking Neural Networks (SNNs) have gained massive attention as an energy-efficient neural computing paradigm due to their event-driven and sparse computing capability. However, maintaining accuracy under low-precision constraints, necessary for practical deployment, has been largely unexplored for SNN models. In addition, existing Hessian-based approaches for quantization, such as HAWQ-V2, incur significant memory and computational overhead due to temporal unrolling across multiple SNN timesteps. We propose a spike-aware gradient-based mixed-precision quantization framework for SNNs that adaptively assigns bit-widths using first-order gradient information, avoiding costly second-order Hessian computations. We evaluate our method across various SNN-based architectures, including VGG, ResNet, and Vision Transformer models obtained from both ANN-to-SNN conversion and direct training frameworks, on CIFAR10/100, ImageNet, an
    
[^180]: TerraTransfer++：免示范数据的端到端规划器

    TerraTransfer++: Demonstration-Free End-to-End Planner

    [https://arxiv.org/abs/2610.30319](https://arxiv.org/abs/2610.30319)

    该论文提出TerraTransfer++，通过利用向量化仿真器中低成本的自我博弈获得多样化的状态分布并预训练策略，再将其轨迹蒸馏到端到端规划器中，从而无需真实驾驶示范数据即可训练高性能的端到端自动驾驶规划器。

    

    端到端自动驾驶在基准测试和实际部署中已达到最先进的性能。然而，其标准训练方案在所有现有范式下都代价高昂：模仿学习需要车队规模级别的驾驶日志，而其行为分布对安全关键的尾部场景覆盖十分有限；在照片级真实感仿真器中进行闭环强化学习虽然能提供多样的行为，却受限于样本效率、每一步的渲染成本以及大型视觉骨干网络的前向计算开销。向量化仿真器中的自我博弈改变了这一经济格局：以每秒数百万步的仿真速度，可以获得包含碰撞、险情和脱险恢复等自然多样性的状态分布——这是任何驾驶日志都不具备的——并且仅凭一个基本的驾驶质量奖励，就能涌现出高度反应灵敏的策略。我们的方法正是利用了这种不对称性：先通过自我博弈预训练一个单一策略，再将其仿真轨迹蒸馏（此处摘要原文截断）……

    arXiv:2610.30319v1 Announce Type: cross  Abstract: End-to-end autonomous driving has achieved state-of-the-art performance on benchmarks and real-world deployments. Its standard training recipe, however, is expensive across all existing paradigms: imitation learning requires fleet-scale driving logs whose behavior distribution only thinly covers the safety-critical tail; closed-loop reinforcement learning in a photorealistic simulator, while offering diverse behaviors, is bottlenecked by sample-efficiency, per-step cost of rendering, and a forward pass through a large vision backbone. Self-play in vectorized simulators changes the economics: at millions of rollout steps per second, one can obtain a state distribution with natural diversity in collisions, near-misses, and recoveries that no driving log contains, and a highly reactive policy emerges from a basic driving-quality reward. Our approach exploits this asymmetry. We pretrain a single policy by self-play, distill its rollouts in
    
[^181]: SciFusion：面向多模态3D科学图像生成的物理引导组合式扩散方法

    SciFusion: Physics-Guided Compositional Diffusion for Multimodal 3D Scientific Image Generation

    [https://arxiv.org/abs/2610.30313](https://arxiv.org/abs/2610.30313)

    SciFusion提出了一种基于潜在扩散的多模态3D科学图像生成框架，通过将模态分解为域、对象和对比度因子实现跨模态知识迁移与零样本组合生成，并引入MRI傅里叶频谱一致性、CT Radon投影一致性等物理信息目标来提升生成的物理合理性。

    

    现有的3D生成方法通常针对单一成像模态开发，需要大量特定模态的数据，并限制了其跨科学成像领域利用共享结构的能力。我们提出了SciFusion，一个建立在潜在扩散主干之上的多模态统一3D科学图像生成框架，具有两项关键贡献。首先，我们引入了一种组合式条件化方案，将每种模态分解为域、对象和对比度表示。在相关模态之间共享这些因子，能够实现异构成像领域之间的知识迁移，并允许在推理时重新组合这些因子，以零样本方式合成未见过的域-对象-对比度组合。其次，我们融入了领域特定的物理信息目标：用于MRI的傅里叶频谱一致性、用于CT的Radon投影一致性以及多层衍射一致性（原文摘要在此处截断）。

    arXiv:2610.30313v1 Announce Type: cross  Abstract: Existing 3D generative approaches are typically developed for individual imaging modalities, requiring substantial modality-specific data and limiting their ability to exploit shared structure across scientific imaging domains. We present SciFusion, a multimodal framework for unified 3D scientific image generation built on a latent diffusion backbone, with two key contributions. First, we introduce a compositional conditioning scheme that factorizes each modality into domain, object, and contrast representations. Sharing these factors across related modalities enables knowledge transfer across heterogeneous imaging domains and allows the factors to be recombined at inference time to synthesize unseen domain-object-contrast compositions in a zero-shot manner. Second, we incorporate domain-specific physics-informed objectives: Fourier spectral consistency for MRI, Radon projection consistency for CT, and multislice diffraction consistenc
    
[^182]: EgoHand：基于自我中心视频的单目3D手部重建

    EgoHand: Monocular 3D Hand Reconstruction from Egocentric Video

    [https://arxiv.org/abs/2610.30293](https://arxiv.org/abs/2610.30293)

    EgoHand是基于1400万张图像高质量语料库训练的单目3D手部重建模型，在自我中心视频上实现了最先进的性能，具备强大的零样本跨域泛化能力和约9.69毫秒的低推理延迟。

    

    我们提出了EgoHand，一个在自我中心（第一视角）视频上实现最先进性能的单目3D手部重建模型。我们方法的核心是构建了一个包含1400万张图像的高质量训练语料库，结合了合成数据、真实场景视频和多相机动作捕捉，以平衡场景多样性与监督质量。该模型采用SOMA-Hand作为其参数化表示，使用transformer编码器处理2D手部裁剪图像，并融入相机线索和辅助2D分支以提升几何一致性和图像对齐效果。在HOT3D、ARCTIC和H2O数据集上的零样本评估展示了强大的跨域泛化能力，在约9.69毫秒的推理延迟下，其关节位置、根平移和加速度误差均低于竞争基线方法。一项盲测用户偏好研究进一步表明，参与者更青睐EgoHand重建的手部姿态，而非Apple EgoDex的结果。

    arXiv:2610.30293v1 Announce Type: cross  Abstract: We introduce EgoHand, a monocular 3D hand reconstruction model that achieves state-of-the-art performance on egocentric video. Central to our approach is the curation of a high-quality training corpus of 14 million images, combining synthetic data, in-the-wild video, and multi-camera motion capture to balance scene diversity and supervision quality. The model adopts SOMA-Hand as its parametric representation and processes 2D hand crops with a transformer encoder, incorporating camera cues and an auxiliary 2D branch to improve geometric consistency and image alignment. Zero-shot evaluations on HOT3D, ARCTIC, and H2O demonstrate strong out-of-domain generalization, with lower joint-position, root-translation, and acceleration errors than competing baselines at an inference latency of approximately 9.69 ms. A blinded user preference study further shows that participants favor EgoHand's reconstructed hand poses over those from Apple EgoDex
    
[^183]: GAV：一个面向偏微分方程智能体的科学实验环境

    GAV: A scientific experimentation environment for PDE agents

    [https://arxiv.org/abs/2610.30292](https://arxiv.org/abs/2610.30292)

    该论文提出GAV框架，通过生成器构造可验证的新PDE问题、智能体迭代开发与调试程序、验证器提供数值反馈，为PDE编码智能体建立了一个无需昂贵参考计算的科学实验环境。

    

    可靠的反馈对于编码智能体开发偏微分方程（PDE）程序至关重要。然而，在新PDE问题上评估这些程序通常需要昂贵的参考计算，而固定数据集又限制了实验的范围。我们提出了生成器-智能体-验证器，这是一个使新PDE问题在构造上即可验证的科学实验环境。生成器利用来自数值模拟或实验测量的数据学习连续参考解的分布，以表示目标物理区域；并借助制造解方法，从生成的参考解反向构造出匹配的PDE问题，而无需单独的数值求解。智能体负责设计实验、调查失败原因，并迭代改进其程序。验证器则根据参考解评估候选程序并提供数值反馈。

    arXiv:2610.30292v1 Announce Type: new  Abstract: Reliable feedback is essential for coding agents to develop programs for partial differential equations (PDEs). Yet evaluating these programs on new PDE problems often requires expensive reference computations, while fixed datasets restrict the range of experiments. We introduce Generator-Agent-Verifier (GAV), a scientific experimentation environment that makes new PDE problems verifiable by construction. The Generator learns a distribution of continuous reference solutions using data from numerical simulations or experimental measurements to represent a target physical regime. Using the method of manufactured solutions, it works backward from the generated reference solutions to construct matching PDE problems without a separate numerical solve. The Agent designs experiments, investigates failures, and iteratively improves its program. The Verifier evaluates candidate programs against the reference solutions and provides numerical feedb
    
[^184]: 用于DAG采样的Metropolis-Hastings算法推广

    Generalizing Metropolis-Hastings for DAG Sampling

    [https://arxiv.org/abs/2610.30291](https://arxiv.org/abs/2610.30291)

    本文提出了“二重采样”这一Metropolis-Hastings算法的新推广形式，通过在每次迭代中选取两个提议分布（一个生成候选状态、一个评估返回当前状态），实现了对DAG中多个节点父集的同时重采样，从而显著提升了贝叶斯网络结构学习中马尔可夫链的混合效率。

    

    贝叶斯网络结构学习中的一个根本性挑战在于高效地近似有向无环图（DAG）的后验分布。对于大型高维数据集，这一任务仍然超出了现有方法的能力范围。本工作通过引入新颖的转移步骤来推进马尔可夫链蒙特卡罗算法的发展，这些步骤能够同时对DAG中多个节点的父节点集进行重采样。这些步骤的核心是“二重采样”，这是Metropolis-Hastings算法的一种新的推广形式，它在每次迭代中从提议分布池中选取两个提议分布：一个用于生成马尔可夫链的候选状态，另一个用于评估从候选状态返回当前状态的转移；两者共同构成了精心设计的接受准则的组成部分。将不同的提议分布进行配对提供了额外的自由度，使得能够构建出比基本Metropolis-Hastings算法混合效率更高的马尔可夫链（摘要在此处被截断）。

    arXiv:2610.30291v1 Announce Type: new  Abstract: A fundamental challenge in Bayesian network structure learning lies in efficiently approximating the posterior distribution of directed acyclic graphs (DAGs). For large, high-dimensional datasets, the task remains beyond the capabilities of current methods. This work advances Markov chain Monte Carlo algorithms by introducing novel moves that simultaneously resample the parents for multiple nodes of the DAG. At the core of these moves is duet sampling, a new generalization of the Metropolis-Hastings algorithm that selects two proposal distributions from a pool at each iteration. One is used for generating a candidate state of the Markov chain and another for evaluating the return to the current state; both are ingredients of an appropriately crafted acceptance criterion. Pairing different proposal distributions provides additional freedom and enables constructing Markov chains that mix more efficiently than their basic Metropolis-Hasting
    
[^185]: SWE-Verifier：面向高性价比编程智能体的选择性验证

    SWE-VERIFIER: Selective Verification for Cost-Efficient Coding Agents

    [https://arxiv.org/abs/2610.30284](https://arxiv.org/abs/2610.30284)

    提出轻量级生成模型SWE-Verifier，通过选择性触发候选动作采样与验证，以仅1.09倍的成本将编程智能体的解决率从75.47%提升至77.20%。

    

    编程智能体在生成动作和处理长交互历史时会消耗大量计算。糟糕的动作选择会使智能体进入低价值状态，浪费计算资源并增加API成本。采样并验证备选动作可以改善智能体的决策，但在每一步都执行该流程会带来大量额外开销。我们提出了SWE-Verifier，一个轻量级生成模型，可选择性地触发候选采样与验证。它对编程智能体提出的动作进行评分，以判断何时需要额外的候选采样，然后在执行前对生成的动作进行排序。我们使用偏好数据从Qwen3.5-4B训练SWE-Verifier，该数据将对比动作与其产生的环境观察结果以及基于评分标准的教师评分分组在一起。在mini-swe-agent框架下，SWE-Verifier以1.09倍的成本将MiniMax-M2.7的解决率从75.47%提升至77.20%，并能泛化到其他（模型）……

    arXiv:2610.30284v1 Announce Type: new  Abstract: Coding agents spend substantial computation generating actions and processing long interaction histories. Poor action choices can steer agents into low-value states, wasting computation and increasing API costs. Sampling and verifying alternative actions can improve the agent's decisions, but applying this procedure at every step adds substantial overhead. We introduce SWE-Verifier, a lightweight generative model that selectively triggers candidate sampling and verification. It scores the coding agent's proposed action to determine when additional candidate sampling is needed, then ranks the resulting actions before execution. We train SWE-Verifier from Qwen3.5-4B using preference data that groups contrasting actions with their resulting environment observations and rubric-guided teacher scores. Using the mini-swe-agent harness, SWE-Verifier improves resolution from 75.47% to 77.20% on MiniMax-M2.7 at 1.09 times cost and generalizes to D
    
[^186]: 二次型视角下的Muon极化更新：收益、陷阱与Newton-Schulz的作用

    A Quadratic Lens on Muon's Polar Update: Benefits, Pitfalls, and the Role of Newton-Schulz

    [https://arxiv.org/abs/2610.30280](https://arxiv.org/abs/2610.30280)

    该论文在矩阵二次型问题上对Muon核心的PolarGD正交化更新进行闭式分析，揭示其收益（梯度远离低秩且最大奇异方向高曲率时有效、误差行正交等范数时可一步收敛、具有旋转不变性）与陷阱，并阐明了Newton-Schulz正交化的作用。

    

    Muon沿近似正交化的方向更新矩阵值参数，训练大语言模型的速度通常快于AdamW，但正交化在何时以及为何有效仍不清楚。我们在矩阵二次型问题上研究Muon核心的正交化更新方法PolarGD，在此类问题上优化器可以进行闭式分析，并在线搜索下比较不同更新方向，以排除学习率调节的影响。当梯度远离低秩、且其最大奇异方向具有高曲率时，正交化才能带来收益。只要误差的各行相互正交且范数相等，无论Hessian如何，PolarGD都能一步到达极小值点，而梯度下降则需要所有Hessian特征值相等；与符号下降不同，PolarGD不受问题旋转的影响。在多步迭代中，PolarGD相当于以当前迭代点为预条件子的梯度下降，因此其速度由条件数决定。

    arXiv:2610.30280v1 Announce Type: new  Abstract: Muon updates matrix-valued parameters along approximately orthogonalized directions and often trains large language models faster than AdamW, yet it remains unclear when and why orthogonalization helps. We study PolarGD, the orthogonalized update at the core of Muon, on matrix quadratics, where optimizers can be analyzed in closed form, and compare update directions under line search to remove the effect of learning-rate tuning. Orthogonalization pays off when the gradient is far from low rank and its largest singular directions see high curvature. PolarGD reaches the minimizer in one step whenever the rows of the error are orthogonal with equal norms, for any Hessian, whereas gradient descent needs all Hessian eigenvalues to be equal; and unlike sign descent, PolarGD is unaffected by rotations of the problem. Over many steps, PolarGD acts as gradient descent preconditioned by the current iterate, so its speed is governed by the conditio
    
[^187]: 编辑而非模仿：面向多模态推理的最小编辑在线策略蒸馏

    Edit, Don't Imitate: Minimal-Edit On-Policy Distillation for Multimodal Reasoning

    [https://arxiv.org/abs/2610.30231](https://arxiv.org/abs/2610.30231)

    该论文提出最小编辑纠正性在线策略蒸馏方法，教师模型仅对学生错误采样中不认可的词元施加纠正损失、正确采样则用任务奖励强化学习，从而在测试时无需额外图形描述的情况下提升多模态数学推理能力。

    

    视觉数学推理可能在推理开始之前就已失败：当视觉语言模型（VLM）误读图表时，后续推理将基于错误的前提进行。这种感知瓶颈在小模型中尤为明显，而准确的文本描述可以显著缩小小模型与大模型之间的差距。由于在推理阶段获取此类描述需要额外的标注或模型调用，我们提出了最小编辑纠正性在线策略蒸馏方法。该方法将参考图形描述和经过验证的参考解答仅作为训练阶段的信息，由教师模型在最小编辑目标下对学生的错误采样结果进行评分，且只有教师不认可的词元才会贡献于纠正性损失；而正确的采样结果则接受基于任务奖励的强化学习。这样既保留了在线策略监督，又避免了对完整响应的模仿。在多个模型规模上，该方法在测试时无需任何描述即可达到或超越（相应性能）。

    arXiv:2610.30231v1 Announce Type: new  Abstract: Visual mathematical reasoning can fail before reasoning begins: when a vision-language model (VLM) misreads a diagram, subsequent reasoning proceeds from false premises. This perceptual bottleneck is especially pronounced in smaller models, while accurate textual descriptions can substantially narrow the gap to larger models. Because obtaining such descriptions at inference requires additional annotation or model calls, we propose minimal-edit corrective on-policy distillation. Using reference figure descriptions and verified reference solutions as training-only information, a teacher scores incorrect student rollouts under a minimal-edit objective, and only teacher-disfavored tokens contribute to the corrective loss. Correct rollouts receive task-reward reinforcement learning. This preserves on-policy supervision while avoiding full-response imitation. Across model scales, the method requires no descriptions at test time and meets or ex
    
[^188]: 医疗AI中隐私风险的自动化筛查

    Automated screening of privacy risks in medical AI

    [https://arxiv.org/abs/2610.30229](https://arxiv.org/abs/2610.30229)

    该研究通过对7,789个医疗AI模型的患者级隐私风险分析，发现高风险患者排名在不同训练配置间基本保持稳定，从而可以在模型开发早期自动化识别和筛查易受成员推断攻击的脆弱患者记录。

    

    医学人工智能（AI）模型通常在不附带训练数据的情况下被共享，并假定这样做可以保护贡献数据的患者。然而，模型可能会泄露某位患者的记录是否被用于训练，这种隐私泄露被称为成员推断攻击。这种风险通常以群体平均值的形式报告，这可能显得微不足道，而实际上个体患者却几乎肯定面临暴露风险。在开发生产模型之前识别潜在易受攻击的记录，可以使隐私评估更具实用性。我们测量了每位患者对成员推断攻击的脆弱性，共涉及在六种临床数据集、四种模态上训练的7,789个模型，其中包括两个适配的预训练医学模型。训练策略的选择会改变具有高攻击分数的患者数量，但在我们测试的各种配置中，按风险排名最高的患者群体基本保持不变。更强的权重衰减降低了暴露程度……

    arXiv:2610.30229v1 Announce Type: new  Abstract: Medical artificial intelligence (AI) models are often shared without their training data, assuming this protects the patients who contributed them. However, a model can reveal whether a patient's records were used in training, a privacy breach known as a membership inference attack. This risk is usually reported as a cohort average, which can appear negligible while individual patients are almost certainly exposed. Identifying potentially vulnerable records before developing a production model could make privacy assessment more practical. We measured each patient's vulnerability to membership inference attack, across 7,789 models trained on six clinical datasets in four modalities, including two adapted pretrained medical models. Training choices changed how many patients had high attack scores, but the patients who ranked highest by risk largely stayed the same across the configurations we tested. Stronger weight decay reduced the expos
    
[^189]: 跨异构边缘平台的模型压缩技术基准测试：面向旋转机械预测性维护的应用案例

    Benchmarking Model Compression Techniques Across Heterogeneous Edge Platforms: A Predictive Maintenance Use Case for Rotating Machinery

    [https://arxiv.org/abs/2610.30228](https://arxiv.org/abs/2610.30228)

    本文针对旋转机械预测性维护任务，对四种最先进的模型压缩技术在异构边缘平台上的性能表现进行了全面的基准测试与对比分析，为工业边缘部署中的压缩技术选择提供了实用指导。

    

    为了实现工业5.0——即自主系统被广泛采用并与人员深度融合以促进工业的可持续发展——必须使复杂的深度学习模型能够在资源受限的硬件上高效运行，以支持边缘推理。然而，由于工业环境中边缘计算的硬件异构性、模型压缩带来的性能下降，以及缺乏关于哪些压缩技术对预测性维护等应用切实有效的明确指导，模型适配仍然极具挑战性。为应对这些挑战，本文对四种最先进的模型压缩技术进行了全面的对比分析，并将其应用于基于真实世界旋转机械（工业中常用设备）数据集的预测性维护高性能卷积神经网络中。压缩后的模型在多个异构边缘平台上进行了评估。

    arXiv:2610.30228v1 Announce Type: new  Abstract: To enable Industry 5.0, where autonomous systems are widely adopted and integrated with people to promote sustainable development in industries, it is imperative to adapt complex deep learning models to operate efficiently on resource-constrained hardware for edge inference. However, adapting these models remains challenging due to the hardware heterogeneity of edge computing in industrial settings, performance degradation caused by model compression, and the lack of clear guidance on which compression techniques are practical and effective for applications such as predictive maintenance. To address these challenges, this paper presents a comprehensive comparative analysis of four state-of-the-art model compression techniques applied to a high-performance convolutional neural network for predictive maintenance using a real-world dataset of rotating machinery, equipment typically used in industry. The compressed models were evaluated acro
    
[^190]: GANet：一种用于嵌入式系统植物病害识别的轻量级分组注意力网络

    GANet: A Lightweight Group Attention Network for Plant Disease Identification on Embedded Systems

    [https://arxiv.org/abs/2610.30225](https://arxiv.org/abs/2610.30225)

    本文提出仅含0.001M参数的轻量级分组注意力网络GANet，可在嵌入式边缘设备上高效识别38种植物病害，并在PlantVillage数据集上达到99.99%的准确率。

    

    全球变暖和气候变化是植物病害的主要催化剂，严重影响着全球粮食生产。解决这一问题需要对植物病害进行持续监测和早期检测。尽管深度学习的最新进展提供了潜在的解决方案，但大多数模型需要较高的处理能力，难以在低端电子硬件上高效运行。此外，许多现有工作只专注于特定作物病害，限制了其更广泛的适用性。本文提出了一种轻量级的群等变注意力网络，专门用于38种植物病害的预测和监测。GANet引入了新颖的分组注意力模块和分组特征融合模块，使其既稀疏又高效。该模型仅有0.001M参数，在PlantVillage数据集上实现了99.99%的惊人准确率。此外，GANet针对在嵌入式系统等边缘设备上的部署进行了优化。

    arXiv:2610.30225v1 Announce Type: new  Abstract: Global warming and climate change are major catalysts for plant diseases, significantly affecting food production worldwide. Addressing this issue requires continuous monitoring and early detection of plant diseases. While recent advancements in deep learning offer potential solutions, most models demand high processing power and struggle to perform efficiently on low-end electronic hardware. Furthermore, many existing works focus on specific crop diseases, limiting their broader applicability. This paper presents a lightweight Group Equivariance Attention Network (GANet) designed for the prediction and monitoring of 38 types of plant diseases. GANet introduces a novel Group Attention block and Group-Wise Feature Fusion block, making it both sparse and efficient. With only 0.001M parameters, the model achieves an impressive 99.99% accuracy on the PlantVillage dataset. Additionally, GANet is optimized for deployment on edge devices such a
    
[^191]: 提示风格如何影响AI在竞技编程中的准确性与速度

    How Prompt Style Affects AI Accuracy and Speed in Competitive Programming

    [https://arxiv.org/abs/2610.30223](https://arxiv.org/abs/2610.30223)

    该研究通过在IOI准确性任务与NP难速度任务上测试九种提示风格，发现没有普适最佳提示，提示风格的选择应取决于任务类型和统计汇总方式，且单一指标评估会掩盖提示效果的不稳定性。

    

    大型语言模型越来越多地被用于竞技编程，其输出对提示的措辞方式十分敏感。已有研究比较了竞技编程中不同提示策略在准确性上的表现，但未比较运行速度，也没有将结果与人类表现进行对照。在一项探索性研究中，我们在GPT-5.5上对九种风格各选取一个提示进行测试，涵盖国际信息学奥林匹克竞赛（IOI）的三个准确性任务和两个NP难速度任务。我们发现不存在单一的最佳提示：最强的提示取决于任务类型，也取决于结果是用均值还是中位数来汇总，这表明至少对于这些任务，提示风格应当与目标相匹配，而不能被当作通用方案。许多均值分数上的差异来自少数离群运行，这说明仅用单一指标评估提示的效果，可能会掩盖其有效性的巨大波动。与（人类表现）进行的描述性比较（摘要在此处截断）

    arXiv:2610.30223v1 Announce Type: new  Abstract: Large language models are increasingly used in competitive programming, and their outputs are sensitive to how a prompt is phrased. Prior work has compared prompt strategies for accuracy on competitive programming, but not for runtime speed, and has not situated the results against human performance. In an exploratory study of one prompt in each of nine styles on GPT-5.5 across three accuracy tasks from the International Olympiad in Informatics (IOI) and two NP-hard speed tasks, we found that no single prompt was best: the strongest prompt depended on the task type and on whether results were summarized by the mean or the median, suggesting that, at least for these tasks, prompt style should be matched to the objective rather than treated as universal. Many differences in mean scores came from a few outlier runs, showing that evaluating prompting by a single metric can obscure how much its effectiveness varies. Compared descriptively wit
    
[^192]: 当重建还不够：基于学习多模态表示的双稳健反事实与干预中介分析

    When Reconstruction Is Not Enough: Doubly Robust Counterfactual and Interventional Mediation Analysis with Learned Multimodal Representations

    [https://arxiv.org/abs/2610.30220](https://arxiv.org/abs/2610.30220)

    该论文证明了仅基于协变量的训练目标（重建、对比或物理信息）无法保证多模态表示足以支持因果识别，给出了表示保持可识别性的条件，并提出了双稳健的反事实与干预中介分析方法。

    

    高维多模态健康记录的学习表示越来越多地被用于调整混杂因素，但它们通常被训练用于重建数据，而非保留因果识别所需的信息。我们研究了替代治疗下的反事实结果，包括表型特异性效应和数字孪生所提供的那种患者层面反事实风险概况，以及暴露诱导的中间混杂下的干预直接与间接效应。我们给出了表示保持可识别性的条件，并表明干预中介分析比反事实比较要求更高。我们证明了没有任何仅基于协变量的训练目标——无论是重建、对比学习还是物理信息驱动——能够证明一个表示是充分的；并且在线性结构模型中，一个表示可能仅损失任意小的重建准确度……（摘要原文在此处截断）

    arXiv:2610.30220v1 Announce Type: new  Abstract: Learned representations of high-dimensional multimodal health records are increasingly used to adjust for confounding, yet they are usually trained to reconstruct the data rather than preserve the information that causal identification requires. We study counterfactual outcomes under alternative treatments, including phenotype-specific effects and patient-level counterfactual risk profiles of the kind digital twins provide, and interventional direct and indirect effects under exposure-induced intermediate confounding. We give conditions under which a representation preserves identification and show that interventional mediation is more demanding than counterfactual comparison. We prove that no covariate-only training objective, whether reconstruction, contrastive, or physics-informed, can certify that a representation is sufficient, and that, in a linear structural model, a representation can lose arbitrarily little reconstruction accura
    
[^193]: AI心智融合：一种潜在互联协议

    AI Mindmeld: A Latent Interconnect Protocol

    [https://arxiv.org/abs/2610.30215](https://arxiv.org/abs/2610.30215)

    提出潜在互联协议（LIP），通过仅训练轻量级连接器将专家模型的潜在表示直接映射到LLM内部状态，实现无需联合训练、可扩展且可组合的模型间直接知识交换，突破了以文本等多模态数据传递信息的瓶颈。

    

    大语言模型（LLM）可以作为科学智能体，使用工具咨询专家模型并运行实验。然而，专家信息必须进入其上下文，通常以文本、图像、音频或视频的形式呈现：这在拥有丰富高维表示的模型之间构成了瓶颈。我们倡导直接的表示交换：一种“心智融合”，传递多种属性，包括LLM不会想到去请求的信息。该方法应可扩展至新的专家模型，灵活应对各种问题和任务，并能在无需联合训练的情况下，在独立连接的专家模型之间实现组合。潜在互联协议学习连接器，将专家模型的表示映射到LLM的内部状态中。仅训练连接器使得该方法在LLM不断增长和演进的情况下依然实用。我们将三个蛋白质专家模型和三个小分子专家模型连接到Gemma 3（4B和12B参数）以及Kimi K2.6（总参数1T，激活参数32B），并冻结所有预训练模型。独立训练的连接……

    arXiv:2610.30215v1 Announce Type: new  Abstract: LLMs can act as scientific agents using tools to consult specialists and run experiments. Yet specialist information must enter their context, generally as text, images, audio or video: a bottleneck between models with rich, high-dimensional representations. We advocate direct representation exchange: a mindmeld conveying many properties, including information an LLM would not think to request. It should be extensible to new specialists, flexible across questions and tasks, and compositional across independently connected specialists without joint training.   The Latent Interconnect Protocol (LIP) learns connectors mapping specialist representations into LLM internal states. Training only connectors keeps this practical as LLMs grow and evolve. We connect three protein and three small molecule specialists to Gemma 3 (4B and 12B parameters) and Kimi K2.6 (1T total, 32B active), freezing all pretrained models. Independently trained connect
    
[^194]: 预测不等于校准：知识追踪中的项目参数恢复与表示耦合

    Prediction Is Not Calibration: Item Parameter Recovery and Representation Coupling in Knowledge Tracing

    [https://arxiv.org/abs/2610.30211](https://arxiv.org/abs/2610.30211)

    本文揭示知识追踪模型中预测准确并不等于项目参数可被正确恢复——难度类参数恢复稳健，而区分度和选项斜率的恢复依赖更宽的项目表示并与项目顺序存在耦合。

    

    知识追踪（KT）模型可以通过仅针对下一次作答预测而训练的作答响应解码器来估计项目参数，但预测准确性并不能保证参数的恢复。我们研究了2PL、GPCM和NRM三种解码器分别与LSTM、Transformer和DKVMN三种编码器配对的组合。响应模型的导数分析揭示了为什么区分度和选项斜率的更新直接依赖于估计能力的量级与离散程度。仿射logit恒等式表明，估计能力中与项目相关的平移和缩放可以在不改变作答概率的情况下被吸收进项目参数中。在模拟实验中，在项目顺序被打乱的条件下，难度、平均阈值和正确选项截距的恢复在不同表示宽度下均保持较高水平，而区分度和正确选项斜率对比度的恢复则需要更宽的项目表示，并且在固定项目顺序下会进一步退化。将序列项目表示（宽度 $d_S$）与项目（此处摘要被截断）

    arXiv:2610.30211v1 Announce Type: new  Abstract: Knowledge tracing (KT) models can estimate item parameters through an item response decoder trained solely for next-response prediction, yet predictive accuracy does not guarantee parameter recovery. We study 2PL, GPCM, and NRM decoders paired with LSTM, Transformer, and DKVMN encoders. Response-model derivatives show why discrimination and option slope updates depend directly on the scale and spread of estimated ability. Affine logit identities show how item-dependent shifts and rescaling in estimated ability are absorbed into item parameters without changing response probabilities. In simulations, difficulty, mean threshold, and correct-option intercept recovery remain high under shuffled order across representation widths, whereas discrimination and keyed option slope contrast recovery require wider item representations and degrade further under fixed item order. Separating the sequence item representation (width $d_S$) from the item 
    
[^195]: 稀疏贝叶斯学习的谱不确定性分解

    Spectral Uncertainty Decomposition for Sparse Bayesian Learning

    [https://arxiv.org/abs/2610.30207](https://arxiv.org/abs/2610.30207)

    该论文提出一种基于先验预条件前向算子奇异值结构的谱框架，将稀疏贝叶斯学习中的后验协方差精确分解为可通过改进测量而减少的数据知情成分与由传感几何决定的不可约不确定性成分。

    

    从脑磁图/脑电图（M/EEG）测量中估计分布式神经源是一个病态逆问题，其中基于原理的不确定性量化仍然具有挑战性。通常的条件高斯后验协方差并未明确区分沿似然可见源方向的不确定性与测量不可见的不确定性。我们提出了一个基于先验预条件前向算子奇异值结构的谱框架，该框架将后验协方差精确且可解释地分解为数据知情成分与不可确定性认知成分。这一公式化方法能够识别哪些不确定性贡献可以通过更好的测量加以减少，哪些是由传感几何结构决定的根本不可减少的不确定性。我们进一步推导出一个等价的减法分解，将先验协方差表示为后验项与数据驱动减少项之和，从而实现精确核算

    arXiv:2610.30207v1 Announce Type: new  Abstract: Estimating distributed neural sources from M/EEG measurements is an ill-posed inverse problem where principled uncertainty quantification remains challenging. The usual conditional Gaussian posterior covariance does not explicitly distinguish uncertainty along likelihood-visible source directions from uncertainty that is invisible to the measurements. We propose a spectral framework based on the singular value structure of the prior-preconditioned forward operator, which provides an exact, interpretable decomposition of the posterior covariance into data-informed and indeterminacy epistemic components. This formulation identifies uncertainty contributions that are reducible by better measurements from those that are fundamentally irreducible by sensing geometry. We further derive an equivalent subtractive decomposition that expresses the prior covariance as the sum of posterior and data-driven reduction terms, enabling precise accounting
    
[^196]: MoEZip：面向稀疏混合专家大语言模型的路由感知KV缓存压缩

    MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs

    [https://arxiv.org/abs/2610.30206](https://arxiv.org/abs/2610.30206)

    提出MoEZip，一种基于Fisher信息矩阵度量专家路由扰动敏感性的路由感知KV缓存压缩方法，避免KV驱逐改变MoE模型的专家选择，从而提升稀疏MoE大模型长上下文推理的生成质量。

    

    混合专家（MoE）语言模型通过每个token仅激活一小部分参数来高效扩展模型容量。然而，在长上下文推理中，内存仍然是主要瓶颈，因为KV缓存会随序列长度线性增长。现有的KV缓存压缩方法能够降低稠密Transformer的这一成本，但忽略了MoE特有的一个问题：KV驱逐可能会改变答案生成过程中所选择的专家，从而降低生成质量。我们提出MoEZip，一种面向稀疏MoE语言模型的路由感知KV缓存压缩方法。为了估计专家路由对扰动的敏感性，我们提出了一种基于路由分布Fisher信息矩阵的新颖度量指标。MoEZip将该指标与注意力权重相结合，对上下文进行路由感知评分。最后，我们构建了一个互补的评分系统，融合了适用于稠密架构和稀疏MoE架构的驱逐方法。实验……

    arXiv:2610.30206v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) language models scale model capacity efficiently by activating a small subset of parameters per token. However, in long-context inference, memory remains a major bottleneck because the KV cache grows linearly with sequence length. Existing KV cache compression methods reduce this cost for dense transformers, but overlook a problem specific to MoE: KV eviction can change which experts are selected during answer generation, degrading the generation quality. We propose MoEZip, a routing-aware KV cache compression for sparse MoE language models. To estimate the sensitivity of expert routing to perturbation, we propose a novel metric based on the Fisher information matrix of routing distributions. MoEZip combines this metric with attention weights for routing-aware scoring of contexts. Finally, we compose a complementary scoring system that combines eviction methods for dense and sparse-MoE architectures. Experiments 
    
[^197]: 基于广义矩匹配的差分隐私合成数据

    Differentially Private Synthetic Data via Generalized Moment Matching

    [https://arxiv.org/abs/2610.30196](https://arxiv.org/abs/2610.30196)

    提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。

    

    我们研究了在指定的积分概率度量（IPM）下，针对[-1,1]^d中包含n条记录的任意数据库的(ε,δ)-差分隐私合成数据生成问题。我们提出了广义矩匹配方法，该方法私有地估计矩、拟合概率测度并采样合成记录。当矩误差能够控制目标IPM时，可应用一个一般性误差定理。该框架扩展了针对1-Wasserstein距离和光滑查询的切比雪夫矩匹配方法。我们将该框架应用于采用Matern核、Laplace核以及两层ReLU神经正切核的最大均值差异（MMD），还获得了切片Wasserstein-1以及由范数受限ReLU网络诱导的IPM的保证。在合成记录数量足够多的情况下，对于固定维度，这些误差率在相差对数因子的意义下是极小极大最优的。

    arXiv:2610.30196v1 Announce Type: cross  Abstract: We study $(\varepsilon,\delta)$-differentially private synthetic data for arbitrary databases of $n$ records in $[-1,1]^d$ under a specified integral probability metric (IPM). We introduce generalized moment matching, which estimates moments privately, fits a probability measure, and samples synthetic records. A general error theorem applies when moment error controls the target IPM. This framework extends Chebyshev moment matching for $1$-Wasserstein distance and smooth queries. We apply the framework to maximum mean discrepancy with Matern and Laplace kernels and a two-layer ReLU neural tangent kernel.We also obtain guarantees for sliced Wasserstein-$1$ and the IPM induced by norm-bounded ReLU networks. With sufficiently many synthetic records,these rates are minimax optimal up to logarithmic factors for fixed dimension.
    
[^198]: 为什么受约束的更新仍然会遗忘：持续监督微调的函数路径分析

    Why Constrained Updates Still Forget: A Functional-Path Analysis of Continual SFT

    [https://arxiv.org/abs/2610.30191](https://arxiv.org/abs/2610.30191)

    该论文提出“函数路径”分析框架，将持续微调导致的遗忘分解为受控方向、梯度陈旧性、探针失配与有限步效应四个因素，并通过匹配的QLoRA干预实验揭示约束放置在Adam之前还是之后对旧能力保留的影响远大于约束粒度本身。

    

    监督微调使语言模型适应专门任务，但可能会破坏其先前习得的行为。现有方法通过约束参数、梯度或重放目标来缓解这一问题，然而受保护对象、实际干预轨迹与保留输出之间的联系仍不清晰。我们发展了一种函数路径分析框架，将实际更新所引起的旧损失变化分解为受控方向、梯度陈旧性、探针失配和有限步效应四个组成部分。该框架给出了一个条件性的刷新预算，并识别出信息性重放所提供的恢复方向。通过匹配的QLoRA干预实验，我们将约束粒度与放置位置（位于Adam之前或之后）分离开来：在所研究的代码标签审计中，放置位置强烈影响旧能力的保留效果，而粒度的交互作用则取决于端点状态。检查点生成实验将端点恢复与更高的采样路径准确率区分开来，匹配的重放方案……

    arXiv:2610.30191v1 Announce Type: cross  Abstract: Supervised fine-tuning adapts language models to specialized tasks but can disrupt previously acquired behavior. Existing approaches constrain parameters, gradients, or replay objectives, yet the connection between the protected object, the intervening trajectory, and retained outputs remains unclear. We develop a functional-path account that decomposes old-loss changes along realized updates into the controlled direction, gradient staleness, probe mismatch, and finite-step effects. The account yields a conditional refresh budget and identifies the restoring direction supplied by informative replay. Matched QLoRA interventions separate constraint granularity from placement before or after Adam: placement strongly affects retention across the studied code-label audits, whereas the granularity interaction depends on the endpoint. Checkpointed generation separates endpoint recovery from higher sampled-path accuracy, and matched replay sch
    
[^199]: 真实世界软件的智能体验证优化

    Agentic Verified Optimization of Real-World Software

    [https://arxiv.org/abs/2610.30188](https://arxiv.org/abs/2610.30188)

    该论文提出“智能体验证优化”新任务：让AI智能体生成更快的程序实现，同时产出以现有实现为规约的机器验证等价性证明，从而避免测试漏洞，并在Lean 4中开发Aqueduct框架将其应用于真实世界软件。

    

    程序优化旨在在保持程序行为不变的前提下提升性能。编码智能体已经能够通过改变算法、数据结构和执行策略来让程序运行得更快。然而，现有的等价性检查大多依赖于测试，而已有研究表明智能体会利用测试的漏洞，在不保留预期行为的情况下通过测试。我们提出智能体验证优化，其中智能体在生成更快实现的同时，还需提供与现有实现等价性的机器验证证明。由于现有实现本身充当规约，该任务避免了验证代码生成中编写规约的瓶颈，而定理证明领域的最新进展使得这些证明变得可行。我们在 Lean 4 中于函数和代码仓库两个层面定义了该任务，在 Lean 4 中程序既可以被执行也可以被推理验证。由于很少有真实世界的应用是用 Lean 编写的，因此我们开发了 Aqueduct，一个智能体框架（摘要在此处截断）。

    arXiv:2610.30188v1 Announce Type: cross  Abstract: Program optimization aims to improve performance while preserving behavior. Coding agents can already make programs faster by changing algorithms, data structures, and execution strategies. However, existing equivalence checks mostly rely on tests, which agents have been shown to exploit, passing the tests without preserving the intended behavior. We propose agentic verified optimization, in which an agent produces a faster implementation together with a machine-checked proof of equivalence to the existing one. Because the existing implementation serves as the specification, the task avoids the specification-writing bottleneck of verified code generation, and recent advances in theorem proving make the proofs attainable. We formulate the task at the function and repository levels in Lean 4, where programs can be both executed and reasoned about. Few real-world applications are written in Lean, so we develop Aqueduct, an agentic framewo
    
[^200]: SIGNN++：面向地球环境的鲁棒图神经网络星图识别

    SIGNN++: Robust Graph Neural Star Identification on Earth

    [https://arxiv.org/abs/2610.30186](https://arxiv.org/abs/2610.30186)

    SIGNN++通过更真实的训练噪声模型、角距边特征的傅里叶变换以及基于RANSAC的透视星对齐三项改进，显著提升了图神经网络星图识别方法在真实地面观测环境下的鲁棒性、识别精度与求解率。

    

    我们提出了SIGNN++，这是一个面向GNSS拒止夜间环境天体导航的基于学习的星图识别（SID）框架。尽管研究人员已经开发了基于学习的SID方法以提升对观测噪声的鲁棒性，但以往的评估主要集中在合成数据或非地面环境上，其在真实地面条件下的可靠性尚不明确。为解决这些局限，SIGNN++在基于图神经网络的SID方法SIGNN的基础上扩展了三个组件：用于训练数据的更真实的噪声模型、角距边特征的傅里叶变换（FT），以及用于识别和姿态精化的基于RANSAC的透视星对齐方法。结果表明，SIGNN++对地面观测干扰具有更强的鲁棒性，与原始SIGNN及公开可用的SID基线方法相比，实现了更高的星图识别精度和求解率，同时保持了相当的

    arXiv:2610.30186v1 Announce Type: cross  Abstract: We present SIGNN++, a learning-based star identification (SID) framework for celestial navigation in GNSS-denied nighttime environments. Although researchers have developed learning-based SID methods to improve robustness to observation noise, prior evaluations have focused mainly on synthetic or non-terrestrial environments, leaving their reliability under real terrestrial conditions unclear. To address these limitations, SIGNN++ extends SIGNN, a graph neural network-based SID method, with three components: a more realistic noise model for training data, Fourier Transform (FT) of angular-distance edge features, and RANSAC-based Perspective Star Alignment for identification and attitude refinement. The results show that SIGNN++ is more robust to terrestrial observation disturbances and achieves higher star identification accuracy and solved rates than the original SIGNN and publicly available SID baselines, while maintaining comparable
    
[^201]: 检测数据中心电力中的人工智能训练

    Detecting AI Training in Datacentre Power

    [https://arxiv.org/abs/2610.30182](https://arxiv.org/abs/2610.30182)

    该论文提出一种基于维特比追踪迭代节奏的方法，能从数据中心设施级电力数据中检测未申报的AI模型训练，即使运营者采用对抗性隐藏策略，一天内仍能以极低误报率捕获至少87%的训练运行。

    

    人工智能治理协议需要能够在受监控的计算集群上检测大规模模型训练运行的方法，且无需信任运营者。我们开发了一种方法，通过维特比追踪漂移的迭代节奏，在设施级电力数据中检测同步模型训练。在模拟的数据中心电力数据上，当运营者不做任何隐藏尝试时，在电表采样率为2 Hz或更快的条件下，维特比追踪器在1%误报率下，相对于申报的推理工作负载，能检测出至少96%的训练运行，其性能与奈曼–皮尔逊最优检测器相差不超过0.04。在运营者可能用于隐藏训练的八种对抗性策略中，只有每次迭代工作量的极端变化能够躲避追踪器。其最强形式可在多达73%的五分钟观测窗口中隐藏训练，但在一天的时间尺度上，此类运行仍至少有87%会被捕获，每次运行的误报率为0.1%。该对抗性策略不损失吞吐量，但（摘要在此处截断）

    arXiv:2610.30182v1 Announce Type: cross  Abstract: AI governance agreements require methods to detect large model training runs on a monitored computing cluster, without trusting the operator. We develop a method to detect synchronous model training in facility-level power by Viterbi tracking of the wandering iteration cadence. On simulated datacentre power, when the operator makes no attempt to hide, the Viterbi tracker detects at least $96\%$ of training runs against a declared inference workload, at a $1\%$ false-alarm rate and within $0.04$ of the Neyman--Pearson optimum, given a meter that samples at $2$ Hz or faster. Of eight adversarial strategies an operator might use to hide, only extreme variation of the work per iteration evades the tracker. Its strongest form hides training in up to $73\%$ of five-minute observations, but over a day at least $87\%$ of such runs are still caught, at a $0.1\%$ false-alarm rate per run. This adversarial strategy costs no throughput, but in sma
    
[^202]: IRAH：面向检索流程优化的分层智能体树搜索

    IRAH: Hierarchical Agentic Tree Search for Retrieval Pipeline Optimization

    [https://arxiv.org/abs/2610.30167](https://arxiv.org/abs/2610.30167)

    IRAH将UCT引导的分层智能体树搜索与离线元学习初始化相结合，在有限评估预算下联合优化RAG检索流程的结构组件与数值超参数，性能最高超越TPE达6.4%。

    

    优化用于检索增强生成（RAG）的信息检索（IR）流程，需要在有限的评估预算下联合选择结构组件和数值超参数。我们提出了信息检索智能体框架IRAH，它将面向信息检索的语言智能体树搜索（LATS-IR）与离线元学习初始化相结合。LATS-IR执行由UCT引导的分层搜索，首先探索宏观结构层面的流程选择，然后细化数值参数，并通过自然语言反思来指导后续的变异操作，同时提供可审查的搜索轨迹。我们在三个分布内基准和一个分布外基准上评估IRAH，以考察其在分布偏移下的泛化能力。在相同的元学习初始化条件下，LATS-IR在三个分布内基准中的两个上优于TPE，提升最高达6.4%，并在分布外基准上保持竞争力。

    arXiv:2610.30167v1 Announce Type: cross  Abstract: Optimizing Information Retrieval (IR) pipelines for Retrieval-Augmented Generation (RAG) requires jointly selecting structural components and numerical hyperparameters under limited evaluation budgets. We introduce the Information Retrieval Agentic Harness (IRAH), which combines Language Agent Tree Search for IR (LATS-IR) with offline meta-learned initialization. LATS-IR performs UCT-guided hierarchical search, first exploring macro-structural pipeline choices and then refining numerical parameters, with natural-language reflections to guide subsequent mutations and provide inspectable search traces. We evaluate IRAH on three in-distribution benchmarks and one out of-distribution benchmark to assess generalization under distribution shift. Under the same meta-learned initialization, LATS-IR outperforms TPE on two of three in-distribution benchmarks, with gains of up to 6.4%, and remains competitive on the out-of-distribution benchmark.
    
[^203]: baseVultra：具有容量重新分配的紧凑非线性递归网络

    baseVultra: Compact Nonlinear Recurrence with Reallocated Capacity

    [https://arxiv.org/abs/2610.30154](https://arxiv.org/abs/2610.30154)

    提出baseVultra紧凑递归单元，将递归计算从门控变换重新分配到候选生成，在参数量匹配条件下于TinyShakespeare字符级语言建模上超越LSTM和GRU。

    

    非线性递归网络在流式序列建模中仍然很有用，然而门控单元的设计仍然由LSTM和GRU主导。我们研究递归计算是否可以从门控变换重新分配到候选生成上。我们提出了baseVultra，这是一种紧凑的递归单元，它将密集的递归计算预算集中在单个H x H的候选生成矩阵上，同时用O(H)的对角交互替换递归门控矩阵。候选更新将这个密集分支与两个低秩辅助分支、一个EMA缓存以及可学习的输出混合相结合；门控还通过对角项获得对h_{t-2}的轻量级二阶访问。在TinyShakespeare字符级语言建模任务上，baseVultra在紧凑的单层200K参数设置中实现了1.594的验证损失，而参数量匹配的LSTM为1.604，参数量匹配的GRU为1.616。

    arXiv:2610.30154v1 Announce Type: new  Abstract: Nonlinear recurrent networks remain useful for streaming sequence modeling, yet gated cell design is still dominated by LSTM and GRU. We study whether recurrent computation can be reallocated away from gate transforms and toward candidate generation. We propose baseVultra, a compact recurrent cell that concentrates the dense recurrent budget on a single H x H candidate-generation matrix, while replacing recurrent gate matrices with O(H) diagonal interactions. The candidate update combines this dense branch with two low-rank auxiliary branches, an EMA cache, and learnable output blending; the gates additionally receive lightweight second-order access to h_{t-2} through diagonal terms. On TinyShakespeare character-level language modeling, baseVultra achieves a validation loss of 1.594 in the compact one-layer 200K setting, compared with 1.604 for a parameter-matched LSTM and 1.616 for a parameter-matched GRU, while its dense recurrent matr
    
[^204]: 打招呼的代价：AI智能体通信协议中握手开销的实证研究

    The Cost of Saying Hello: An Empirical Study of Handshake Overhead in AI Agent Communication Protocols

    [https://arxiv.org/abs/2610.30151](https://arxiv.org/abs/2610.30151)

    本文首次对MCP、A2A和ACP三种AI智能体通信协议的握手开销进行分阶段实证测量，发现往返次数是握手成本的主导因素——旧版MCP需三次往返耗时230 ms，而A2A仅需一次往返耗时61 ms。

    

    AI智能体之间的每一次交互都始于一次握手：发起方智能体必须先定位对方、了解对方的能力，并建立会话，才能发送第一条有用的消息。关于智能体通信协议的综述将握手开销列为效率评估标准之一，然而此前没有任何研究提供过实测数据。本文首次对三大主流智能体协议——MCP（涵盖其2025年的initialize时期和2026年的discover时期）、A2A和ACP——中的握手过程进行了分阶段的实证基准测试。我们使用各协议的官方SDK，在冷启动、热重复、能力不匹配和元数据退化等场景下（每格N=50次，分别在本地回环和模拟50 ms往返延迟条件下），对每个请求的往返次数、各阶段延迟、传输字节数和令牌成本进行了测量。结果表明，往返次数主导了握手成本：旧版MCP握手需要三次往返，在50 ms往返延迟下耗时230 ms；而A2A仅需一次往返，耗时61.0 ms。

    arXiv:2610.30151v1 Announce Type: cross  Abstract: Every interaction between AI agents begins with a handshake: the initiating agent must locate its counterpart, learn what it can do, and establish a session before the first useful message can be sent. Surveys of agent communication protocols name handshake overhead as an efficiency criterion, yet none report measurements. This paper presents the first empirical, phase-decomposed benchmark of the handshake in three major agent protocols: MCP (in both its 2025 initialize and 2026 discover eras), A2A, and ACP. Using each protocol's official SDK, we instrument every request for round trips, per-phase latency, wire bytes, and token cost across cold-start, warm-repeat, capability-mismatch, and degraded-metadata scenarios (N=50 per cell, on loopback and under an emulated 50 ms RTT). Round-trip count dominates handshake cost. A legacy MCP handshake takes three round trips and 230 ms at 50 ms RTT; A2A needs one round trip and 61.0 ms. Round tr
    
[^205]: 机器学习引导的主动式满负荷方案能否缓解急诊科拥挤？一项基于仿真的评估

    Do Machine Learning-Guided Proactive Full Capacity Protocols Mitigate Emergency Department Overcrowding? A Simulation-Based Evaluation

    [https://arxiv.org/abs/2610.30148](https://arxiv.org/abs/2610.30148)

    该研究构建了机器学习与离散事件仿真相结合的评估框架，利用提前六小时的候诊室拥挤预测来触发主动式满负荷方案，以检验其相比传统阈值触发式（反应式）激活能否更有效地控制急诊科拥挤。

    

    目的：急诊科（ED）拥挤威胁着医疗质量，然而满负荷方案（FCPs）通常仅在拥挤程度超过阈值后才被激活。本研究评估了机器学习引导的主动式满负荷方案激活与传统反应式激活相比，能否改善急诊科的拥挤控制。材料与方法：我们开发并验证了一个集成机器学习与离散事件仿真的框架，用于刻画急诊科患者流与容量管理。其中嵌入的机器学习模型可提前六小时预测候诊室的每小时拥挤情况，并据此触发主动式满负荷方案激活。基于从美国一家大型医院经验校准的运营参数，我们在365天的仿真周期内通过10次独立重复实验，比较了主动式与反应式部署策略。结果指标包括急诊科住院时长、等待时间、超过拥挤阈值的时间百分比等（原文摘要在此截断）。

    arXiv:2610.30148v1 Announce Type: cross  Abstract: Objective: Emergency department (ED) overcrowding threatens care quality, yet Full Capacity Protocols (FCPs) are typically activated only after crowding thresholds have been exceeded. This study evaluated whether machine learning-guided proactive FCP activation improves ED crowding control compared with conventional reactive activation.   Materials and Methods: We developed and validated an integrated machine learning and discrete-event simulation framework representing ED patient flow and capacity management. An embedded machine learning model predicted hourly waiting-room crowding six hours ahead and triggered proactive FCP activation. Using empirically calibrated operational parameters from a large US hospital, we compared proactive and reactive deployment strategies over a 365-day simulation horizon with 10 independent replications. Outcomes included ED length of stay, wait time, percentage of time above the crowding threshold, num
    
[^206]: Code Comprehensor：面向大型代码仓库的层次化文档与跨模态检索

    Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories

    [https://arxiv.org/abs/2610.30016](https://arxiv.org/abs/2610.30016)

    该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。

    

    理解大型、复杂且快速演进的代码库仍然是规模化软件工程的核心挑战，它阻碍了新人上手、代码维护、重构和影响分析。将这些信息提供给大语言模型（LLM）是另一个障碍，因为LLM不仅需要理解所需的代码方法，还需要理解其在整个仓库全局结构中相关联的方法。我们提出了Code Comprehensor，这是一个混合检索-推理框架，结合了：（i）代码依赖调用图，（ii）CodeScribe——利用代码感知的LLM在仓库调用图上生成层次化文档，（iii）CodeContrast——一个CLIP风格的双编码器模型，基于CodeScribe的输出进行训练以学习对齐的文本-代码嵌入。CodeContrast能够实现快速、语义精准的相关代码上下文邻域检索，这些邻域作为基于MCP的LLM编排器回答领域特定查询时的可靠依据。

    arXiv:2610.30016v1 Announce Type: cross  Abstract: Understanding large, complex, and rapidly evolving code bases remains a core challenge for software engineering at scale, hindering onboarding, maintenance, refactoring, and impact analysis. Providing this information to a Large-Language Model (LLM) is another hurdle, as the LLM must understand not only the required code method but also its associated methods in the grand scheme of the repository. We propose Code Comprehensor, a hybrid retrieval-reasoning framework combining (i) a code dependency callgraph, (ii) CodeScribe, which uses code-aware LLMs to generate hierarchical documentation across repository call graphs, and (iii) CodeContrast, a CLIP-style dual-encoder model trained on CodeScribe output to learn aligned text-code embeddings. CodeContrast enables fast, semantically precise retrieval of relevant code-context neighborhoods, which serve as grounded evidence for an MCP-based LLM orchestrator answering domain-specific queries
    
[^207]: 基于Kolmogorov-Arnold网络的可处理推理

    Tractable Inference with Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2610.29909](https://arxiv.org/abs/2610.29909)

    该论文提出TOUCKAN，通过为KAN网络引入有限次多项式外函数形成可分解的和-积表示，使期望、高阶矩等统计量能够通过确定性递归精确计算，从而避免了高维场景下昂贵的蒙特卡洛采样。

    

    计算科学中的机器学习代理模型通常不仅需要准确的预测，还需要计算期望、方差、守恒积分和归一化常数。传统神经网络（包括Kolmogorov-Arnold网络，即KAN）一般需要通过蒙特卡洛采样或多维数值求积来评估这些量，导致重复推理的成本高昂，尤其在高维情况下更是如此。我们提出了具有可处理算子的不确定性与约束KAN网络（TOUCKAN），其有限次多项式外函数产生了可分解的和-积表示。在因式分解的输入测度下，期望和高阶矩可通过代数方式化归为对一维叶子积分的确定性递归计算。独立和共享内块两种变体构成了一个参数效率层次结构，且针对其最受约束成员的通用逼近定理证明可扩展至整个类别。

    arXiv:2610.29909v1 Announce Type: cross  Abstract: Machine-learning surrogates in computational science often require not only accurate predictions but also expectations, variances, conserved integrals, and normalization constants. Conventional neural networks, including Kolmogorov--Arnold networks (KANs), generally evaluate these quantities through Monte Carlo sampling or multidimensional quadrature, making repeated inference costly, particularly in high dimensions. We introduce KANs with Tractable Operators for Uncertainty and Constraints (TOUCKANs), whose finite-degree polynomial outer functions yield a decomposable sum--product representation. Under factorizing input measures, expectations and higher moments then reduce algebraically to deterministic recursions over one-dimensional leaf integrals. Independent and shared inner-block variants form a parameter-efficiency hierarchy, and a universal-approximation proof for its most restricted member extends to the complete class. The sa
    
[^208]: RangeNav：面向机器人导航的任务对齐激光雷达距离图像预训练

    RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation

    [https://arxiv.org/abs/2610.29845](https://arxiv.org/abs/2610.29845)

    提出RangeNav，一个仅18.3万参数、完全基于合成数据预训练的轻量级激光雷达距离图像编码器，通过结合距离重建、语义分割和鸟瞰图预测的导航对齐目标，为大规模并行强化学习的机器人导航提供高效的360°感知表征。

    

    在密集人群和杂乱场所中运行的自主移动机器人需要360°感知能力以应对周围的危险。虽然全景激光雷达能够提供全方位覆盖，但在强化学习（RL）中学习表征在稀疏奖励下样本效率低下，且现有的3D预训练点云编码器对于大规模并行RL仿真而言内存开销过大。我们提出了RangeNav，这是一个紧凑型（18.3万参数）的激光雷达距离图像编码器，完全使用合成数据进行预训练。RangeNav采用各向异性卷积和循环填充来处理距离图像的极端长宽比和方位角接缝问题。我们使用结合距离重建、语义分割和鸟瞰图（BEV）预测的导航对齐目标对网络进行预训练，然后将其冻结用于策略学习。在仿真中，相比仅使用重建的基线，添加语义和BEV监督使成功率提升了（原文在此处截断）。

    arXiv:2610.29845v1 Announce Type: cross  Abstract: Autonomous mobile robots operating in dense crowds and cluttered sites require $360^\circ$ perception to react to surrounding hazards. While panoramic LiDAR offers full-surround coverage, learning representations within reinforcement learning (RL) is sample-inefficient under sparse rewards, and existing 3D pretrained point-cloud encoders are too memory-heavy for massively parallel RL simulation. We present RangeNav, a compact (183k parameters) LiDAR range-image encoder pretrained entirely on synthetic data. RangeNav uses anisotropic convolutions and circular padding to handle the extreme aspect ratio and azimuthal seam of range images. We pretrain the network using a navigation-aligned objective combining range reconstruction, semantic segmentation, and bird's-eye-view (BEV) prediction, then freeze it for policy learning. In simulation, adding semantic and BEV supervision increases success rates over a reconstruction-only baseline by a
    
[^209]: TraceForge：用自己的思维进行蒸馏

    TraceForge: Distilling with Your Own Thinking

    [https://arxiv.org/abs/2610.29680](https://arxiv.org/abs/2610.29680)

    提出TraceForge，以教师模型的答案为输入、按学生模型自身家族的风格生成推理轨迹，并通过“监督+对抗”两阶段训练，在不违反服务条款的前提下实现对闭源前沿模型的推理蒸馏。

    

    使用一个大语言模型为另一个生成训练数据是最通用的训练方式之一，至今仍是常见做法。一些前沿团队已对自家模型关闭了这条路线：他们返回的只是答案或简短摘要，而不附带背后详细的推理轨迹。这略显讽刺：这些模型本身正是建立在蒸馏人类知识之上的，而蒸馏开源模型仍然完全常规；该限制仅针对它们自身的输出。最近有工作通过跨模型回放加密数据块来恢复隐藏的推理过程，虽然这种方法不知为何竟能奏效，但它违反了服务提供商的服务条款，因此无法作为长期构建产品的基础。我们构建了替代方案：以上下文和教师模型的答案作为输入，自行生成推理轨迹，并使其符合学生自身模型家族的风格。训练分为两个阶段——先监督学习、后对抗学习：一个生成器学习……

    arXiv:2610.29680v1 Announce Type: cross  Abstract: Using an LLM to generate training data for another is among the most general ways to train one, and it remains common practice. Some frontier teams have closed that route for their own models: what comes back is an answer, or a short summary, without the detailed reasoning trace behind it. That is a little ironic: they were themselves built on distilling human knowledge, and distilling an open-source model is still entirely routine; the restriction applies to their own outputs alone. Recent work recovers the hidden reasoning by replaying encrypted blocks across models. It somehow works, but it breaks the providers' terms of service, so it is not something a product can be built on long-term.   We built the alternative. We take the context and the teacher's answer as input, and generate the reasoning trace ourselves, in the style of the student's own model family. Training is two-stage, supervised and then adversarial: a generator learn
    
[^210]: 面向多样化推理轨迹的可并行化概率潜在推理

    Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories

    [https://arxiv.org/abs/2610.29385](https://arxiv.org/abs/2610.29385)

    该论文提出了一种可并行的概率潜在推理方法，通过在紧凑连续的潜在空间中对多个有效推理轨迹的分布进行建模，既实现了多样化推理轨迹的探索，又支持高效的并行化训练。

    

    依赖语言化思维链的语言模型受限于在语言空间中逐个token的生成方式，其中许多token仅服务于文本连贯性而非核心推理。这使得逻辑轨迹上的真实差异与语言表达的表面变化纠缠在一起。将推理转移到紧凑的连续潜在空间中，为高效表示和探索多样化推理轨迹提供了一种有前景的替代方案。然而，当前的潜在模型大多仍是确定性的，只能捕获单一推理轨迹，而无法对多个有效备选方案的分布进行建模。虽然随机方法对潜在推理步骤的分布进行了参数化，但它们常常面临优化不稳定和实际性能有限的问题。此外，它们在训练过程中依赖自回归的潜在预测，阻碍了有效的并行化，导致了令人难以承受的（训练开销）……

    arXiv:2610.29385v1 Announce Type: new  Abstract: Language models relying on verbalized Chain-of-Thought (CoT) are constrained to token-by-token generation in the language space, where many tokens serve textual coherence rather than core reasoning. This entangles genuine differences in logical trajectories with superficial variations in verbalization. Moving reasoning into a compact, continuous latent space offers a promising alternative for efficiently representing and exploring diverse reasoning trajectories. However, current latent models remain largely deterministic, capturing only a single reasoning trajectory rather than modeling a distribution over multiple valid alternatives. While stochastic approaches parameterize distributions over latent reasoning steps, they frequently suffer from unstable optimization and limited empirical performance. Furthermore, their reliance on autoregressive latent prediction during training precludes effective parallelization, leading to prohibitive
    
[^211]: 跨语言语音克隆中用于口音控制的闭环双参考引导方法

    Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning

    [https://arxiv.org/abs/2610.29351](https://arxiv.org/abs/2610.29351)

    提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。

    

    零样本跨语言语音克隆常常会将参考说话人的口音带入目标语言中。我们提出闭环双参考引导方法，将原始参考与通过语音转换（VC）构建的可复用母语锚点进行配对。这两个参考分别初始化各自独立的语言模型（LM）状态，共同引导一条共享的声学轨迹；每个完成的语音片段随后会更新这两个状态，使其生成历史与实际输出的语音保持对齐。锚点构建每个说话人仅需执行一次，在线合成使用冻结的TTS模型，无需逐句进行语音转换。在20个保留测试的法语说话人上，固定设置 λ*=0.33 与直接克隆相比，将残余的法语LID质量降低了73%，ECAPA相似度损失仅为0.042；相对于仅使用锚点的克隆，该方法恢复了0.155的相似度。在保留测试组中，该方法实现了最低的平均WER（0.009）和最高的平均GOP（4.750）。

    arXiv:2610.29351v1 Announce Type: cross  Abstract: Zero-shot cross-lingual voice cloning often carries the reference speaker's accent into the target language. We propose closed-loop dual-reference guidance, which pairs the original reference with a reusable native anchor constructed by voice conversion (VC). The references initialize separate language-model (LM) states that guide a shared acoustic trajectory; each completed patch then updates both states, keeping their generated histories aligned with the emitted speech. Anchor construction runs once per speaker, and online synthesis uses a frozen TTS model without per-utterance VC. On 20 held-out French speakers, the fixed setting $\lambda^{\!*}=0.33$ reduces residual French LID mass by 73\% versus direct cloning, at an ECAPA similarity cost of only $0.042$; relative to anchor-only cloning, it recovers $0.155$. The method achieves the lowest mean WER ($0.009$) and highest mean GOP ($4.750$) among the held-out test arms. In blinded pr
    
[^212]: 存在但难以读取：用最小提示对探查潜在推理

    Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs

    [https://arxiv.org/abs/2610.28498](https://arxiv.org/abs/2610.28498)

    论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。

    

    潜在推理的探查研究通常在答案开始生成的位置读取模型状态，并将探查失败当作该信息不存在的证据。我们用一个从问题中复制的数字来检验这一点，实验对象是循环模型 Ouro-1.4B 和标准 Transformer Qwen2.5-1.5B。比较仅在该数字上不同的提示对可以发现，该数字确实存在于最后一个提示词元中，并以一种跨问题共享的编码形式存在（读取准确率约 0.9）。而一次只看单个提示的常规探针几乎读不出它：即使训练了 5,249 个问题，Ouro 中仅达 0.36，Qwen 中仅达 0.19。该数字引起的状态变化是“微小”而非“拥挤”的：随机噪声会落在同样繁忙的方向上，而用一个随机词替换该数字会使状态发生大得多的位移且可被读取。由于每个变化的词元都会留下痕迹，成功的探查结果同样存在歧义。我们提出了一套探查协议，包括用于检验一般性混合效应的合成零假设，用于将模型计算出的值与输入的副本及混合信息进行区分测试。

    arXiv:2610.28498v1 Announce Type: cross  Abstract: Probing studies of latent reasoning often read a model's state where the answer begins and take a failed probe as evidence that a value is absent. We test this with a digit copied from the question, in the looped model Ouro-1.4B and the standard transformer Qwen2.5-1.5B. Comparing prompts that differ only in that digit shows it is present at the last prompt token, in a code shared across questions (readout about 0.9). A probe that sees one prompt at a time, the usual setting, barely reads it: 0.36 in Ouro and 0.19 in Qwen even after 5,249 training questions. The digit's change is small rather than crowded: random noise lands on equally busy directions, while a random word in its place moves the state far more and is readable. Since every varied token leaves a trace, a successful probe is ambiguous too. We give a protocol, including a synthetic null for generic mixing, for testing computed values against copies and mixtures of the input
    
[^213]: 披露引发披露：具有持久记忆的LLM智能体中的自我强化隐私循环

    Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory

    [https://arxiv.org/abs/2610.28315](https://arxiv.org/abs/2610.28315)

    该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。

    

    大语言模型智能体越来越多地使用持久记忆来个性化响应，而记忆中存储的个人属性一旦暴露便构成隐私风险。我们探究在对话中查阅记忆如何改变用户后续的发言内容，进而改变记忆最终会揭示的信息。在多会话模拟中，一个拥有24个个人属性的模拟用户与一个查阅或不查阅记忆的客服智能体对话，并在两种设置下均从完整累积的对话记录中推断这些属性。在四个客服模型上，使用记忆提升了响应质量，但同时也增加了经验性泄露——即从记录中能被正确推断出的属性值比例——且这一差距随会话次数增加而不断扩大。其根源在于普通的、非对抗性辅助过程中存在的一个循环：基于记忆的响应会将早期的属性值延续下去（常以预设的形式出现），而用户会确认这些值或不予纠正，从而使线索和智能体自身的猜测都被记录下来。

    arXiv:2610.28315v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly use persistent memory to personalize responses, and personal attributes in this memory pose a privacy risk if exposed. We ask how consulting memory during dialogue changes what users go on to say, and thus what the memory comes to reveal. In multi-session simulations, a simulated user with 24 personal attributes talks with a support agent that does or does not consult memory, and in both settings attributes are inferred from the full accumulated record. Across four support models, memory raises response quality but also empirical leakage, the share of attribute values correctly inferred from the record, by a margin that widens over sessions. The cause is a loop in ordinary, non-adversarial assistance. Memory-conditioned responses carry earlier values forward, often as presuppositions, and users affirm them or leave them uncorrected, so that cues and the agent's own guesses become recorded 
    
[^214]: 基于投毒的后门并非密码学安全的

    Poisoning-Based Backdoors Are Not Cryptographic

    [https://arxiv.org/abs/2610.28045](https://arxiv.org/abs/2610.28045)

    本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。

    

    为了防御针对神经网络的后门攻击，防御者必须识别出决定某个输入是否为触发器的验证器函数，而不能只是孤立地拒绝单个触发器，因为攻击者生成满足同一验证器的新触发器的速度会快于防御者拦截它们的速度。如果后门验证器能够像密码学认证方案那样具有不对称性——即只有攻击者能够产生触发器输入，而防御者无法通过观测逆向推导出验证器——那么基于学习的防御在原则上将是不可能的。我们证明，这种不对称性不可能由基于投毒的安装方式产生，在这种方式中，训练过程中唯一依赖于验证器的输入就是被投毒的数据集。该论证采用归约方法：一个能够从被投毒数据中安装验证器、使后门在未见过的触发器上触发同时保持良性行为的训练过程，其本身就是一个学习（摘要在此处截断）

    arXiv:2610.28045v1 Announce Type: cross  Abstract: To defend against backdoor attacks on neural networks, the defender must identify the verifier function that determines whether an input is a trigger, rather than rejecting individual triggers in isolation, since an attacker can generate new triggers satisfying the same verifier faster than they can be blocked. If a backdoor verifier could be installed with the asymmetry of a cryptographic authentication scheme, in which only the attacker can produce trigger inputs while the defender cannot reverse-engineer the verifier from observations, learning-based defense would be impossible in principle. We show that no such asymmetry can arise from poisoning-based installation, in which the only verifier-dependent input to training is a poisoned dataset. The argument is a reduction. A training procedure that installs a verifier from poisoned data, so that the backdoor fires on unseen triggers while benign behavior is preserved, is itself a lear
    
[^215]: ShlokaDhara：一个用于在“文本到诵读”中学习格律感知韵律的多说话人梵语语音语料库

    ShlokaDhara: A Multi-Speaker Sanskrit Speech Corpus for Learning Meter-Aware Prosody in Text-to-Recitation

    [https://arxiv.org/abs/2610.26929](https://arxiv.org/abs/2610.26929)

    该论文发布了 43 小时的多说话人梵语诵读语料库 ShlokaDhara，用以支持要求格律正确性的“文本到诵读”合成任务，并通过 16 位梵语专家的主观评估与新颖实证指标验证了 XTTS-v2 在该任务上达到最高的感知质量。

    

    梵语诗歌的诵读受"Chandas"（诗律）的支配，这是一个由节奏性的 Laghu（轻音节）和 Guru（重音节）所定义的严格韵律框架。由于现有的梵语语音数据集主要面向自动语音识别（ASR）或对话式语音合成（TTS），它们缺乏真实诗歌合成所需的格律保真度。我们推出了 ShlokaDhara，一个 43 小时、多说话人的梵语语音语料库，涵盖了多样的格律、诵读风格与旋律变化。我们特别关注 Text-to-Recite（文本到诵读）这一合成任务，该任务明确要求同时具备韵律准确性与格律正确性。通过自动化的 UTMOS（神经平均意见分）预测器对最先进的 TTS 架构进行基准测试，结果表明 XTTS-v2 实现了最高的感知质量。我们进一步采用两种范式对 XTTS-v2 生成的样本进行验证：由 16 位梵语专家开展的主观评估，以及新颖的实证指标，特别是……

    arXiv:2610.26929v1 Announce Type: cross  Abstract: Sanskrit verse recitation is governed by $\textit{Chandas}$, a strict prosodic framework defined by rhythmic $\textit{Laghu}$ (light) and $\textit{Guru}$ (heavy) syllables. Because existing Sanskrit speech datasets prioritize ASR or conversational TTS, they lack the metrical fidelity required for authentic verse synthesis. We introduce $\textbf{\textit{ShlokaDhara}}$, a 43-hour, multi-speaker Sanskrit speech corpus capturing diverse meters, recitation styles, and melodic variations. We specifically focus on $\textit{Text-to-Recite}$, a synthesis task explicitly requiring both prosodic accuracy and metrical correctness. Benchmarking state-of-the-art TTS architectures via an automated UTMOS (neural MOS) predictor demonstrates that XTTS-v2 achieves the highest perceptual quality. We further validate the XTTS-v2 generated samples using two paradigms: subjective assessments by 16 Sanskrit experts and novel empirical metrics, specifically, s
    
[^216]: WhatIf-Bench：面向干预感知时间序列预测的基准测试

    WhatIf-Bench: Benchmarking Intervention-Aware Time Series Forecasting

    [https://arxiv.org/abs/2610.26582](https://arxiv.org/abs/2610.26582)

    该论文提出了WhatIf-Bench，一个基于数据中心、电网、交通和供水系统机理仿真器构建的半合成基准，用于系统评估时间序列基础模型利用协变量进行干预感知情景预测的能力。

    

    近年来，时间序列基础模型（TSFMs）可以在上下文中接收协变量，将零样本预测扩展到由外部因素驱动的问题上。这使得情景分析成为可能：即预测目标变量对那些将协变量设定为特定未来值的事件的响应。然而，这些模型能否可靠地做到这一点仍是一个悬而未决的问题。我们提出了WhatIf-Bench，这是一个用于衡量预测模型利用协变量来预测目标对未来事件响应能力的基准。WhatIf-Bench是半合成的：时间序列来自数据中心、电网、道路交通和供水网络的机理仿真器，其协变量与目标之间的关系在每个系统中受到控制且具有实际依据。每个任务都呈现一段对协变量进行干预的事件历史，然后要求模型预测一个未来事件，该事件要么重放已见过的协变量值（Replay），要么将其设定为未见过的值（Novel）。通过改变协变量数量、历史事件数量和事件长度，共产生了37,665个任务。

    arXiv:2610.26582v2 Announce Type: replace-cross  Abstract: Recent time series foundation models (TSFMs) can consume covariates in-context, extending zero-shot forecasting to problems driven by external factors. This enables scenario analysis: forecasting how a target responds to events that set covariates to chosen future values. Whether they can do this reliably is an open question. We introduce WhatIf-Bench, a benchmark measuring how well forecasters use covariates to forecast a target's response to future events. WhatIf-Bench is semi-synthetic: time series come from mechanistic simulators of data centers, power grids, road traffic, and water networks, with controlled covariate-target relationships grounded in each system. Each task presents a history of events intervening on covariates, then asks to forecast a future event that either replays seen values (Replay) or sets them to unseen ones (Novel). Varying the number of covariates, historical events, and event length yields 37,665 
    
[^217]: NADS：面向微调中缓解灾难性遗忘的导航器引导数据选择

    NADS: Navigator-Guided Data Selection for Mitigating Catastrophic Forgetting in Fine-Tuning

    [https://arxiv.org/abs/2610.25751](https://arxiv.org/abs/2610.25751)

    NADS通过先按目标微调配方训练一个“导航器”模型，再利用该模型与预训练模型之间的预测散度为合成数据评分，从而在预训练数据不可用时高效选择数据，以缓解LLM微调中的灾难性遗忘。

    

    灾难性遗忘是大语言模型（LLM）在单阶段微调至新任务时的核心障碍，尤其是在原始预训练数据不可用的情况下。合成数据提供了一条实用的缓解途径，但候选数据池往往噪声大且冗余，因此有效的数据选择至关重要。现有方法通常依赖间接的代理信号，而这些信号与实际训练收益的一致性可能有限；同时，在大规模候选池上进行“效用—多样性”选择的计算成本也很高。为应对这些挑战，我们提出了导航器引导的数据选择（NADS），一个面向LLM的微调框架。NADS首先按照目标微调配方（不加入保持性正则化）训练得到一个导航器模型，其相对于预训练模型的偏差能够揭示由微调配方引起的能力漂移。随后，它利用导航器模型与预训练模型之间的预测散度来为遗忘相关……（摘要不完整，此处为可获得部分）

    arXiv:2610.25751v2 Announce Type: replace  Abstract: Catastrophic forgetting is a central obstacle in single-stage fine-tuning of large language models (LLMs) on a new task, especially when the original pretraining data are unavailable. Synthetic data offers a practical mitigation path, but candidate pools are often noisy and redundant, making effective selection essential. Existing methods often rely on indirect proxy signals whose alignment with actual training benefit can be limited, while utility--diversity selection over large candidate pools remains computationally demanding. To address these challenges, we propose NAvigator-guided Data Selection (NADS), a fine-tuning framework for LLMs. NADS first follows the target fine-tuning recipe without preservation regularization to obtain a navigator model, whose deviation from the pretrained model exposes recipe-induced capability drift. It then uses the predictive divergence between the navigator and the pretrained model to score forge
    
[^218]: 基于蛋白质组规模基准的反向靶标垂钓生成式检索

    Generative Retrieval for Reverse Target Fishing with a Proteome-Scale Benchmark

    [https://arxiv.org/abs/2610.24458](https://arxiv.org/abs/2610.24458)

    该论文提出GenFish，将反向靶标垂钓重新表述为生成式检索任务，通过残差向量量化蛋白质语言模型特征生成语义标识符，避免将未记录配对误判为负样本，并发布了首个蛋白质组规模基准ProTargetBench。

    

    反向靶标垂钓为查询分子识别蛋白靶标，是药物发现与药物重定位的核心任务。从虚拟筛选改编而来的判别式方法面临稀疏、长尾的关联以及非随机缺失的测量问题：将未记录的蛋白-分子配对视为负样本可能会抑制真实的靶标。我们将该任务重新表述为生成式检索，并提出了GenFish。通过残差向量量化，将冻结的蛋白质语言模型特征转换为共享的语义标识符。给定分子指纹后，GenFish以查询和前序编码为条件自回归地对编码进行建模，依据标识符似然对蛋白质进行排序，而无需将未记录的配对显式标记为负样本。我们同时提出了ProTargetBench，这是首个面向反向靶标垂钓的蛋白质组规模基准，可联合支持测量面板判别以及对实验记录阳性样本的全目录恢复。该基准覆盖20,431个Uni...（摘要在此处截断）

    arXiv:2610.24458v2 Announce Type: replace  Abstract: Reverse target fishing identifies protein targets for a query molecule and is central to drug discovery and repurposing. Discriminative approaches adapted from virtual screening face sparse, long-tailed associations and missing-not-at-random measurements: treating unrecorded pairs as negatives can suppress genuine targets. We reformulate the task as generative retrieval and introduce GenFish. Residual vector quantization converts frozen protein language model features into shared semantic identifiers. Given a molecular fingerprint, GenFish autoregressively models codes conditioned on the query and preceding code, ranking proteins by identifier likelihood without explicitly labeling unrecorded pairs as negatives. We introduce ProTargetBench, the first proteome-scale benchmark for reverse target fishing to jointly support measured-panel discrimination and full-catalogue recovery of experimentally recorded positives. Covering 20,431 Uni
    
[^219]: 可预测扩展：基于扩展定律预测的高效模型探索

    Predictable Scaling: Efficient Model Exploration via Scaling-Law Prediction

    [https://arxiv.org/abs/2610.23453](https://arxiv.org/abs/2610.23453)

    该论文提出“可预测扩展”（PS）框架，通过沿资源轴构建小规模代理模型并结合曲线拟合与不确定性校准，从代理实验预测全规模推荐模型性能，从而将原本需数千GPU小时的模型改进假设评估变得高效可行。

    

    评估推荐模型的候选变更（假设）需要进行全规模训练，消耗数千GPU小时，严重限制了工程师能够实际探索的假设空间。我们提出了可预测扩展，这是一个系统性框架，利用经过实证验证的规律性条件，从小规模代理模型预测全规模模型的性能。PS通过沿不同资源轴的可组合扩展函数构建缩小版模型，并通过带有校准不确定性估计的曲线拟合将指标投射到目标规模。我们比较了三种表达能力递增的预测策略：即使是最简单的（原始代理）也能达到与重复运行方差的经验噪声水平相当的预测误差；表达能力最强的（高斯过程）则提供了15%的相对MAE改进和校准的不确定性估计。在两个复杂度递增的排序模型上的评估——一个……

    arXiv:2610.23453v1 Announce Type: new  Abstract: Evaluating candidate changes (hypotheses) to recommendation models requires full-scale training runs consuming thousands of GPU-hours, severely limiting the hypothesis space engineers can feasibly explore. We introduce Predictable Scaling (PS), a systematic framework that exploits empirically verified regularity conditions to predict full-scale model performance from small-scale proxies. PS constructs scaled-down variants via composable scaling functions along distinct resource axes, and projects metrics to the target scale via curve fitting with calibrated uncertainty estimates. We compare three prediction strategies of increasing expressiveness: even the simplest (raw proxy) achieves prediction error within the empirical noise floor of repeated-run variance; the most expressive (Gaussian Process) provides an indicative 15% relative MAE improvement and calibrated uncertainty. Evaluation on two ranking models of increasing complexity---a
    
[^220]: 一种用于耦合代理地貌动力学建模的自适应时域图神经算子

    An Adaptive-Horizon Graph Neural Operator for coupled surrogate morphodynamic modelling

    [https://arxiv.org/abs/2610.22432](https://arxiv.org/abs/2610.22432)

    本文提出自适应时域图神经算子AH-GNO，直接在非结构网格上从水动力与地形状态学习预测河床高程增量，并能根据流动和河床演变自适应调整历史长度与更新间隔，从而实现高效的地貌动力学耦合代理建模。

    

    地貌动力学模拟需要水动力求解器、泥沙输移模块与河床更新之间的反复耦合。对于在大规模非结构网格上进行的长历时、多情景模拟而言，这种耦合在计算上代价高昂。本研究提出一种自适应时域图神经算子（AH-GNO），将河床演变表述为一个图算子学习问题，从近期的水动力和地形状态出发预测未来的河床高程增量。该模型直接在原始非结构网格上运行。该方法融合了面积加权图学习、历史状态选择和预报时域估计，使有效历史长度（K）和更新间隔（B）能够随流动与河床状态的演变而自适应变化。以Telemac2D/GAIA求解结果作为监督数值参考，该框架在官方Bump和Yen弯道基准算例以及一个现场尺度应用（Waiho河）上进行了评估。

    arXiv:2610.22432v1 Announce Type: cross  Abstract: Morphodynamic simulations require repeated coupling between hydrodynamic solvers, sediment transport modules, and bed updating. For long duration, multi-scenario simulations on large unstructured meshes, such coupling is computationally demanding. This study proposes an Adaptive-Horizon Graph Neural Operator (AH-GNO) that formulates bed evolution as a graph operator learning problem from recent hydrodynamic and topographic states to predict future bed elevation increments. The model operates directly on the native unstructured meshes. The approach integrates area-weighted graph learning, historical state selection, and forecast horizon estimation, allowing the effective history length (K) and update interval (B) to vary with the evolving flow and bed state. Using Telemac2D/GAIA solutions as supervised numerical references, the framework is evaluated on the official Bump and Yen bend benchmarks and on a field-scale application (Waiho Ri
    
[^221]: 用于鲁棒行星着陆的对抗性强化学习

    Adversarial Reinforcement Learning for Robust Planetary Landing

    [https://arxiv.org/abs/2610.20252](https://arxiv.org/abs/2610.20252)

    该论文提出一种三阶段对抗性强化学习训练框架，由辅助智能体生成具有时空相关性且满足物理约束的最坏情况扰动，与着陆控制器形成非对称序贯零和博弈并交替优化，从而显著提升行星着陆策略应对非标称条件的鲁棒性。

    

    用于行星着陆以及其他航空航天制导、导航与控制任务的深度强化学习（DRL）策略，通常在随机化扰动下进行训练以提高鲁棒性。然而，典型的随机扰动采样假设各扰动变化相互独立，而实际任务中的扰动往往表现出空间与时间上的相关性，无法通过独立的随机抽样来复现。本工作研究对抗性强化学习，以发现具有挑战性的非标称条件。一个辅助智能体生成物理受限的扰动来降低着陆控制器的性能，从而形成非对称的序贯零和博弈。训练分为三个阶段进行：基线控制器学习、借助随机扰动进行鲁棒性增强，以及控制器与扰动生成器交替优化的迭代对抗阶段。在……中的初步实验（原文在此处截断）

    arXiv:2610.20252v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) policies for planetary landing and other aerospace guidance, navigation, and control tasks are commonly trained under randomized disturbances to improve robustness. However, typical stochastic perturbation sampling assumes independent variations, whereas real mission disturbances often exhibit spatial and temporal correlation and cannot be reproduced through independent random draws. This work investigates adversarial reinforcement learning to discover challenging off-nominal conditions. A secondary agent generates physically constrained perturbations that degrade the performance of a landing controller, forming an asymmetric sequential zero-sum game. Training proceeds in three stages: baseline controller learning, robustness enhancement with stochastic perturbations, and an iterative adversarial phase in which the controller and disturbance generator are alternately optimized. Preliminary experiments in
    
[^222]: 基于缺口驱动的分层采集方法构建 STRATA-S12：一个面向自监督学习的开放 SAR-光学数据集

    Deficit-Driven Stratified Acquisition for Building STRATA-S12: An Open SAR-Optical Corpus for Self-Supervised Learning

    [https://arxiv.org/abs/2610.17680](https://arxiv.org/abs/2610.17680)

    提出了缺口驱动分层采集算法（DDSA），将构建配准的 Sentinel-1/Sentinel-2 数据集建模为半球校正分层上的约束最大覆盖问题，并通过人机协同的缺口修复机制确保全部 192 个分层得到完整覆盖，为自监督地球观测学习提供了高质量的开放 SAR-光学语料库。

    

    arXiv:2610.17680v2 公告类型：replace-cross 摘要：为自监督地球观测构建一个配准的 SAR-光学语料库，本质上首先是一个数据采集问题，然而目前主要的多模态数据集都是机会性采集的，既没有分层的配额控制，也没有应对非均匀云覆盖所导致的地理数据缺失的机制。我们提出了缺口驱动的分层采集（DDSA）算法，该算法将构建配准的 Sentinel-1/Sentinel-2 语料库视为在经过半球校正的（类别×季节×区域）分层上的约束最大覆盖问题，执行严格的时间配对和多阶段质量把关，并在采集过程中通过人机协同的缺口修复阶段形成闭环。在该算法自身产品上的测量表明，该算法取得了单次采集无法报告的三项结果。缺口驱动的排序使得部分运行也能成为覆盖性运行：它使 192 个分层中的每一个从最初几个……

    arXiv:2610.17680v2 Announce Type: replace-cross  Abstract: Building a co-registered SAR-optical corpus for self-supervised Earth Observation is above all an acquisition problem, yet the main multi-modal datasets are collected opportunistically, with no per-stratum quota control and no mechanism against the geographic dropout that non-uniform cloud cover leaves behind. We present Deficit-Driven Stratified Acquisition (DDSA), an algorithm that treats building a co-registered Sentinel-1/Sentinel-2 corpus as a constrained maximum-coverage problem over a hemisphere-corrected (class x season x region) stratification, enforces strict temporal pairing and a multi-stage quality gate, and closes the loop during acquisition with a human-in-the-loop deficit-repair stage. Measured on its own product, the algorithm yields three results a single-pass acquisition cannot report. Deficit-driven ordering makes a partial run a covering one: it holds every one of the 192 strata populated from the first few
    
[^223]: ATTUNE：面向LLM训练的智能体化吞吐量调优

    ATTUNE: Agentic Throughput Tuning for LLM Training

    [https://arxiv.org/abs/2610.17528](https://arxiv.org/abs/2610.17528)

    ATTUNE是一个无人值守的编码智能体，通过“一次调一个参数”的诊断式搜索和正确性门控，在GPU小时预算内将LLM训练吞吐量平均提升1.7倍，同时确保不改变训练目标。

    

    LLM训练任务的吞吐量取决于其配置方案：模型如何在GPU之间切分、如何以内存换取计算、使用哪些内核以及采用何种通信重叠策略。最佳配置方案会随每个模型、加速器和框架版本而变化。自动调优器只能搜索固定的参数空间，且仅检查任务能否运行，因此会遗漏该空间之外的调节手段，还可能给出一个悄然改变训练行为的配置。ATTUNE是一个编码智能体，能够在GPU小时预算内无人值守地调优任何可正常运行的配置方案。它由一套包含搜索规则、脚本和以往调优经验的技能库指引，对每次运行进行诊断并一次只调整一个参数。智能体之外的正确性门控会拒绝任何已知会改变训练目标的参数设置。ATTUNE在H100、H200和B200集群上调优了亚马逊的生产级后训练配置方案，平均将吞吐量提升1.7倍。它超越了四款（原文在此处截断）已公布的吞吐量。

    arXiv:2610.17528v2 Announce Type: replace-cross  Abstract: The throughput of an LLM training job depends on its recipe: how the model is split across GPUs, how memory is traded for compute, and which kernels and communication overlaps it uses. The best recipe changes with every model, accelerator and framework version. Automated tuners search a fixed space of knobs and check only that a job runs, so they miss levers outside that space and can ship a configuration that silently changes training. ATTUNE is a coding agent that tunes any working recipe unattended within a GPU-hour budget. Guided by a skill of search rules, scripts and lessons from past tuning campaigns, it diagnoses each run and changes one lever at a time. A correctness gate outside the agent refuses any argument known to change the training objective. ATTUNE tunes Amazon's production post-training recipes on H100, H200 and B200 clusters, raising throughput by 1.7x on average. It beats the published throughput of four of 
    
[^224]: FreeEvolve：学习超越固定循环的进化方式

    FreeEvolve: Learning to Evolve Beyond Fixed Loops

    [https://arxiv.org/abs/2610.09197](https://arxiv.org/abs/2610.09197)

    FREEEVOLVE 通过元进化将“如何优化”本身变成从经验中学到的能力，让进化器摆脱人工设计的固定搜索循环，自主决定测试什么、收集多少证据、保留哪些候选方案以及何时停止。

    

    智能体进化器（agent evolvers）能够自动化设计围绕语言模型智能体的提示词、技能和工作流程，然而它们所遵循的优化过程本身仍是人工设计的：一个固定的搜索循环决定如何评估候选方案、保留哪些候选方案以及何时停止搜索。我们提出 FREEEVOLVE，它将这一过程也予以自动化。环境指定目标、目标智能体、评估器、数据和资源限制；在这些限制范围内，进化器自身决定测试什么、收集多少证据、追求哪些候选方案以及何时停止。这些决策遵循一个可编辑的进化技能，我们通过元进化（meta-evolution）来改进该技能——即在每个候选技能所产出的全新目标智能体上对其进行评分。由此，优化过程成为从经验中学习到的能力，而非预先设计的循环。在 tau3-bench、ARC-AGI-2、ARC-AGI-3 和 Terminal-Bench 2.1 上，FREEEVOLVE 自主掌控整个进化活动……

    arXiv:2610.09197v1 Announce Type: cross  Abstract: Agent evolvers automate the design of the prompts, skills and workflows around language model agents, yet the optimization process they follow is still designed by hand: a fixed search loop decides how candidates are evaluated, which are kept and when the search stops. We propose FREEEVOLVE, which automates this process as well. An environment specifies the goal, target agent, evaluator, data and resource limits; within these limits, the evolver itself decides what to test, how much evidence to collect, which candidates to pursue and when to stop. These decisions follow an editable evolution skill, which we improve through meta-evolution by scoring each candidate skill on the fresh target agent it produces. The optimization process thus becomes a capability learned from experience rather than a loop engineered in advance. On tau3-bench, ARC-AGI-2, ARC-AGI-3 and Terminal-Bench 2.1, FREEEVOLVE controls the evolution campaign by itself, y
    
[^225]: 冗余与协同的相遇：基于次模优化的MoE依赖感知专家选择

    Redundancy Meets Synergy: Dependency-aware Expert Selection for MoE via Submodular Optimization

    [https://arxiv.org/abs/2610.00558](https://arxiv.org/abs/2610.00558)

    提出DS-MoE框架，通过次模函数差分优化感知专家间的冗余与协同依赖关系，为MoE模型实现更优的紧凑专家子集选择。

    

    尽管混合专家模型通过稀疏激活有效地扩展了模型容量，但其部署常常受到过高内存需求的制约。提取一个紧凑的专家子集是一种有前景的解决方案。然而，现有的专家选择启发式方法主要依赖Top-k排序，这种做法孤立地评估单个专家，忽略了MoE门控网络所引入的专家间复杂的相互依赖关系。本文提出DS-MoE，一个具有理论支撑的框架，通过次模函数差分（DS）优化重新定义了专家选择问题。通过分析损失退化的二阶泰勒展开，我们揭示了专家组合中的功能二重性：冗余（专家编码重叠的表示）与协同（专家提供互补的误差消除）。为应对这种二重性，我们在数学上将冗余削减……

    arXiv:2610.00558v1 Announce Type: new  Abstract: While Mixture-of-Experts (MoE) models effectively scale model capacity through sparse activation, their deployment is often bottlenecked by prohibitive memory requirements. Extracting a compact subset of experts presents a promising solution. However, existing expert selection heuristics predominantly rely on Top-k ranking, which isolates the evaluation of individual experts and ignores the intricate inter-expert dependencies introduced by the MoE gating network. In this paper, we propose DS-MoE, a theoretically grounded framework that redefines expert selection via difference-of-submodular (DS) optimization. By analyzing the second-order Taylor expansion of the loss degradation, we reveal functional duality within expert combinations: redundancy (where experts encode overlapping representations) and synergy (where experts provide complementary error cancellation). To navigate this duality, we mathematically decouple redundancy reduction
    
[^226]: PACT：从视频中端到端学习人体姿态、接触与作用力

    PACT: End-to-End Learning of Human Pose, Contacts, and Forces from Video

    [https://arxiv.org/abs/2610.00451](https://arxiv.org/abs/2610.00451)

    PACT 提出一个端到端模型，通过可学习的接触力标记、时间 Transformer 和基于物理的监督，从单目视频中联合估计人体姿态、接触与接触力，并解决作用力标注稀缺问题。

    

    人体运动、环境接触以及交互作用力都遵循共同的物理定律，然而现有方法通常将视觉姿态重建与接触及作用力估计分离开来处理。这种分离限制了对二者的联合推理，并可能在各阶段之间传播误差。我们提出了 PACT，这是一种端到端模型，能够从单目视频中联合学习估计人体姿态、接触以及接触力。我们的方法在人体重建基础模型的基础上，引入了可学习的接触力标记和一种将视觉特征与世界空间运动相融合的时间 Transformer。联合预测头用于细化人体姿态并估计接触和作用力，同时基于物理的监督机制促使重建的运动与交互作用力之间保持一致性。为了解决作用力标注数据稀缺的问题，我们开发了一套数据标注流程，该流程将接触标注与基于物理的运动和作用力优化相结合。

    arXiv:2610.00451v1 Announce Type: cross  Abstract: Human motion, environmental contacts, and interaction forces are governed by common physical laws, yet existing approaches typically separate visual pose reconstruction from contact and force estimation. This separation limits joint reasoning and can propagate errors between stages. We introduce PACT, an end-to-end model that jointly learns to estimate human pose, contacts and contact forces from monocular video. Our approach augments a human reconstruction foundation model with learnable contact-force tokens and a temporal transformer that integrates visual features with world-space motion. Joint prediction heads refine human poses and estimate contacts and forces, while physics-based supervision encourages consistency between the reconstructed motion and interaction forces. To address the scarcity of force annotations, we develop a data annotation pipeline that combines contact labeling with physics-based motion and force optimizatio
    
[^227]: DashVMC：Geometry Dash中的实时离散世界模型控制

    DashVMC: Real-Time Discrete World Model Control in Geometry Dash

    [https://arxiv.org/abs/2609.40003](https://arxiv.org/abs/2609.40003)

    DashVMC仅用约两小时Geometry Dash游戏录像学习紧凑的动作条件世界模型，控制器通过BC初始化后完全在冻结模型推演中用PPO优化即可超越初始性能，并在消费级GPU上实现60Hz的实时捕获-动作闭环。

    

    世界模型智能体通常是在可以等待策略计算的模拟器中进行评估的；而实时游戏则提出了相反的约束，要求在下一帧到来之前完成画面捕获、预测和动作执行。我们提出了DashVMC，它从约两小时的Geometry Dash游戏录像中学习一个紧凑的、以动作为条件的世界模型。为了检验所学习到的动态是否真正可用于控制，控制器先通过行为克隆（BC）进行初始化，随后使用近端策略优化（PPO）完全在冻结模型的轨迹推演中进行优化，无需与实时游戏进行任何 further 交互。在三个控制器随机种子下，优化后的策略在全部三个官方关卡以及一个留出的社区自制关卡上的存活时间均长于其BC初始版本。在部署时，该基线方法跳过视觉生成环节，在消费级GPU上维持每秒60帧的从画面捕获到动作输出的循环。基于动作条件的续写推演和轨迹诊断表明，该模型在（截断处）……

    arXiv:2609.40003v1 Announce Type: new  Abstract: World-model agents are usually evaluated in simulators that can wait for the policy; live games impose the opposite constraint, requiring capture, prediction, and action before the next frame. We present DashVMC, which learns a compact, action-conditioned world model from approximately two hours of recorded Geometry Dash gameplay. To test whether the learned dynamics are actionable, a controller is initialized by behavioural cloning (BC) and refined with Proximal Policy Optimization (PPO) entirely in frozen-model rollouts, without further interaction with the live game. Across three controller seeds, the refined policies survive longer than their BC initializations on all three official levels and a held-out community layout. At deployment, the baseline skips visual generation and sustains a 60-Hz capture-to-action loop on a consumer GPU. Action-conditioned continuations and rollout diagnostics show that the model remains useful for cont
    
[^228]: UnlearningSoup：大语言模型遗忘真的需要反复调参吗？

    UnlearningSoup: Is Repeated Tuning Necessary for Large Language Model Unlearning?

    [https://arxiv.org/abs/2609.37076](https://arxiv.org/abs/2609.37076)

    该论文发现不同遗忘运行得到的模型在权重空间中共处一个评估性能盆地，据此提出UnlearningSoup——一种针对遗忘任务定制的模型汤（权重平均）策略，无需反复调参即可恢复出更强的遗忘模型。

    

    在海量语料上训练的大语言模型天然存在记忆有害内容并在后续输出中重新出现的风险。为缓解这一问题，现有的遗忘方法通常依赖基于训练的参数更新，例如梯度上升及其变体，在删除目标内容的同时保留其他知识。然而，遗忘与保留这两个相互冲突的目标使得这些方法的超参数选择尤为困难，往往需要反复调参才能得到较强的模型，即便如此仍有较大改进空间，且在不同模型和数据集之间迁移性较差。为应对这一挑战，我们研究了遗忘过程在权重空间中是否具有可利用的结构，并观察到来自不同运行所得到的模型仍位于一个共享的评估性能盆地中。这表明可以通过一种为遗忘量身定制的模型汤（soup）策略恢复出更强的模型，从而减少反复调参的开销。

    arXiv:2609.37076v1 Announce Type: new  Abstract: Large language models trained on vast corpora inherently risk memorizing harmful content that may later re-emerge in their outputs. To mitigate this issue, existing unlearning methods typically rely on training-based parameter updates, such as gradient ascent and its variants, to delete targeted content while preserving other knowledge. However, balancing the competing goals of forgetting and retention makes hyperparameter choices for these methods particularly difficult, often requiring repeated tuning to obtain a strong model that still leaves substantial room for improvement and transfers poorly across models and datasets. To address this challenge, we investigate whether unlearning runs exhibit exploitable structure in weight space, and observe that models from different runs still lie in a shared evaluation-performance basin. This suggests that stronger models may be recovered through an unlearning-tailored soup strategy, reducing t
    
[^229]: PROACT-Agent：面向实时安全的渐进式运行时监督与主动熔断机制

    PROACT-Agent: Progressive Runtime Oversight and Active Circuit-breaking for Real-Time Safety

    [https://arxiv.org/abs/2609.34415](https://arxiv.org/abs/2609.34415)

    提出PROACT-Agent框架，通过渐进式轨迹展开、推理增强的因果矫正和文化感知的数据本地化合成高保真安全轨迹数据，并构建包含15万余条标注状态的双语安全基准PROACT-Bench，以实现对大模型智能体的实时主动安全防护。

    

    arXiv:2609.34415v2 公告类型：替换 摘要：从大语言模型（LLM）向智能体的转变，将安全风险从有害文本升级为不可逆的环境危害。当前的防御手段大多仍停留在事后回顾层面，而主动的运行时干预则受制于缺乏大规模、因果一致的数据。我们提出PROACT-Agent，一个用于合成高保真轨迹以实现实时安全防护的框架。我们在既有基准中发现了一个关键的“安全漂移”问题，即宽松的标注范式未能保证时间上的一致性。PROACT-Agent通过以下方式解决该问题：（1）渐进式轨迹展开，揭示隐藏在长上下文交互中的风险；（2）推理增强的因果矫正，强制实现单调的因果一致性；（3）文化感知的数据本地化，增强跨境场景下的鲁棒性。我们进一步提出PROACT-Bench，一个包含通过多模型裁决标注的155,780个状态的双语安全基准。对……的评估

    arXiv:2609.34415v2 Announce Type: replace  Abstract: The transition from Large Language Models (LLMs) to agents shifts safety stakes from toxic text to irreversible environmental harm. While current defenses remain largely retrospective, proactive runtime intervention is bottlenecked by the lack of large-scale, causally-consistent data. We propose PROACT-Agent, a framework for synthesizing high-fidelity trajectories to enable real-time guardrails. We identify a critical "safety drift" in prior benchmarks, where lenient annotation paradigms fail to enforce temporal consistency. PROACT-Agent addresses this through: (1) Progressive Trajectory Unrolling to reveal risks hidden in long-context interactions; (2) Reasoning-Augmented Causal Rectification to enforce monotonic causal consistency; and (3) Culturally-Aware Data Localization for cross-border robustness. We introduce PROACT-Bench, a bilingual safety benchmark with 155,780 states labeled through multi-model adjudication. Evaluating up
    
[^230]: 评判者是其孪生兄弟吗？后训练使模型的写作更可预测，但作为评判者，其品味几乎不会偏向可预测的写作

    Is the Judge Its Twin? Post-training makes a model's writing more predictable but barely moves its taste, as a judge, toward predictable writing

    [https://arxiv.org/abs/2609.32196](https://arxiv.org/abs/2609.32196)

    该研究发现后训练（SFT和DPO）确实使语言模型的写作变得更加可预测，但这种可预测性偏好并不会迁移到模型作为评判者的角色——训练后的模型在评判故事时几乎不偏向可预测的写作。

    

    语言模型如今经常给其他语言模型评分。如果后训练使模型的写作更可预测，它是否也会教导同一个模型（作为评判者）奖励可预测的写作？我们直接测试了这一点。我们跟踪两个开放的70亿参数模型家族——OLMo-2和Zephyr——从它们的基础模型经过监督微调（SFT）和直接偏好优化（DPO）的完整过程，并在每个阶段测量它们作为短篇故事作者和故事对评判者的表现。作为作者，模型如人们担心的那样发生了漂移：每个家族训练后的模型按基础、SFT、DPO的顺序，在全部十个提示中都发现自己各阶段生成的故事更加熟悉，而另一个家族的模型则发现训练后生成的故事每token的惊讶度分别降低了7.0至9.2个百分点（OLMo-2）和24至25个百分点（Zephyr）。但作为评判者，它们几乎不偏向可预测的写作。当被问及哪个故事更好时——一个所有评判者都能通过能力检验的问题——没有任何训练后的评判者的偏好表现出对可预测写作的倾向。

    arXiv:2609.32196v2 Announce Type: replace  Abstract: Language models now routinely grade other language models. If post-training makes a model's writing more predictable, it may also teach the same model, as a judge, to reward predictable writing. We test this directly. We follow two open 7-billion-parameter model families, OLMo-2 and Zephyr, from their base models through supervised fine-tuning (SFT) and direct preference optimization (DPO), and measure every stage as a writer of short stories and as a judge of story pairs. As writers, the models drift as feared. Each family's trained model finds its stages' stories more familiar in the order base, SFT, DPO, in all ten prompts, and a model from the other family finds the trained stories 7.0 to 9.2 percent (OLMo-2) and 24 to 25 percent (Zephyr) less surprising per token. As judges, they barely move toward predictable writing. Asked which story is better, a question every judge passes a competence check on, no trained judge's preference
    
[^231]: 残差相关性作为GP协同区域化联合不确定性增益的诊断指标

    Residual Correlation as a Diagnostic for Joint-Uncertainty Gains from GP Coregionalisation

    [https://arxiv.org/abs/2609.30085](https://arxiv.org/abs/2609.30085)

    该论文发现GP协同区域化的主要收益在于联合不确定性量化而非点预测，并提出残差相关性（而非原始目标相关性）才是预测联合不确定性增益的最强指标，同时给出了可从独立逐目标残差计算的轻量级诊断指标D_logdet来量化这一增益。

    

    在多目标回归中，相关目标通常通过具有内在协同区域化模型（GP-ICM）的多输出高斯过程进行耦合，其假设是共享统计强度能够提升整体性能。然而在实践中，其收益并不一致。在所研究的各种设置中，我们发现协同区域化的主要收益在于联合不确定性量化，而非点预测。原始目标相关性无法预测耦合何时有益；在此研究的可分离GP-ICM设置中，残差相关性——即独立的逐目标预测器无法解释的跨目标依赖性——是联合不确定性增益的最强预测因子。我们引入了一个轻量级诊断指标 D_logdet = -1/2 log det R_res，它表示通过建模完整的残差协方差而非对角残差协方差所获得的理想化联合负对数似然（NLL）增益，且该指标可由独立的逐目标残差计算得出。

    arXiv:2609.30085v1 Announce Type: new  Abstract: In multi-target regression, correlated targets are often coupled through multi-output Gaussian processes with an intrinsic model of coregionalisation (GP-ICM), assuming that sharing statistical strength improves overall performance. In practice, the benefits are inconsistent. Across the settings studied, we find that the main benefit of coregionalisation is joint uncertainty quantification rather than point prediction. Raw target correlation does not predict when coupling helps; in the separable GP-ICM settings studied here, residual correlation, the cross-target dependence left unexplained by independent per-target predictors, is the strongest predictor of joint-uncertainty gains.   We introduce a lightweight diagnostic, $D_{\rm logdet}=-\frac{1}{2}\log\det R_{\rm res}$, which represents the idealised joint negative log-likelihood (NLL) gain from modelling a full rather than diagonal residual covariance and is computable from independen
    
[^232]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^233]: 扩展FunctionGemma以实现实用的设备端移动函数调用

    Extending FunctionGemma for Practical On-Device Mobile Function Calling

    [https://arxiv.org/abs/2609.25373](https://arxiv.org/abs/2609.25373)

    该研究通过构建涵盖十五类设备控制操作的约9,500个对话的合成数据集MOBILEACTIONSEXTENDED对FunctionGemma 270M进行微调，将设备端Android函数调用的端到端准确率从29.3%大幅提升至76.5%，并使组合模型在Google数据集上达到82.3%。

    

    设备端助手需要能够将自然语言映射为本地系统操作的函数调用模型，但现有资源主要侧重于Web API或狭窄的移动操作目录。我们通过引入MOBILEACTIONSEXTENDED数据集，将FunctionGemma 270M-it扩展到实用的Android工作流程中。该数据集是一个合成的、经过模式校验的数据集，包含约9,500个对话，涵盖十五个设备控制类别，包括消息发送、电话呼叫、相机/截图、亮度控制、设备状态查询、手电筒控制以及应用程序管理。我们使用TRL监督微调方法（仅对完成部分计算损失）对270M模型进行微调，得到一个扩展的专用模型，以及一个与Google的MOBILEACTIONSGOOGLE联合训练的组合模型。在MOBILEACTIONSEXTENDED上，端到端准确率从基础模型的29.3%、Google Mobile-Actions变体的17.2%提升至76.5%。组合模型在MOBILEACTIONSEXTENDED上保持76.5%的准确率，并在MOBILEA（原文截断，应为Google数据集）上达到82.3%。

    arXiv:2609.25373v1 Announce Type: new  Abstract: On-device assistants require function-calling models that map natural language to local system actions, but existing resources emphasize web APIs or narrow mobile-action catalogs. We extend FunctionGemma 270M-it to practical Android workflows by introducing MOBILEACTIONSEXTENDED, a synthetic, schema-validated dataset of ~9,500 conversations covering fifteen device-control categories, including messaging, phone calls, camera/screenshot, brightness control, device-status queries, flashlight control, and application management. We fine-tune the 270M model with TRL supervised fine-tuning under completion-only loss, producing an extended specialist and a combined model trained jointly with Google's MOBILEACTIONSGOOGLE. On MOBILEACTIONSEXTENDED, end-to-end accuracy improves from 29.3% for the base model and 17.2% for Google's Mobile-Actions variant to 76.5%. The combined model retains 76.5% on MOBILEACTIONSEXTENDED and reaches 82.3% on MOBILEA
    
[^234]: 运筹学中的强化学习：技术综述与实践路线图

    Reinforcement Learning in Operational Research: A Technical Review and Practical Roadmap

    [https://arxiv.org/abs/2609.24750](https://arxiv.org/abs/2609.24750)

    本文系统综述了强化学习与运筹学相结合的技术进展，并提出实践路线图，展示如何利用RL的学习能力提升传统OR算法在动态决策、组合优化和数字孪生中的解质量、计算效率与鲁棒性。

    

    复杂动态系统中对实时、数据驱动决策日益增长的需求，正给传统运筹学（OR）方法带来越来越大的压力。强化学习（RL）作为一种补充方法应运而生，为动态和不确定环境下的序贯决策提供了强大的学习和计算能力。近期研究显示，人们对将强化学习与运筹学相结合的兴趣日益浓厚，以解决动态决策问题、增强组合优化的启发式方法和精确方法，并支持运营系统数字孪生的开发。这些努力的总体目标是利用强化学习的学习能力来强化传统运筹学算法，提高解的质量、计算效率和鲁棒性。鉴于集成方法和应用场景的多样性，显然需要一个系统性的技术……

    arXiv:2609.24750v1 Announce Type: cross  Abstract: The growing demand for real-time, data-driven decision-making in complex and dynamic systems is placing increasing pressure on traditional Operational Research (OR) methodologies. Reinforcement learning (RL) has emerged as a complementary approach, offering strong learning and computational capabilities for sequential decision-making in dynamic and uncertain environments. Recent research shows an increasing interest in integrating RL with OR to address dynamic decision-making problems, enhance heuristic and exact methods for combinatorial optimization, and support the development of digital replicas of operational systems. The overarching goal across these efforts is to leverage the learning capabilities of RL to strengthen traditional OR algorithms, improving solution quality, computational efficiency, and robustness. Given the diversity of integration approaches and application settings, there is a clear need for a systematic and tec
    
[^235]: FlashBoB：面向Softmax注意力的I/O高效精确反向之反向算法

    FlashBoB: I/O-Efficient Exact Backward-over-Backward for Softmax Attention

    [https://arxiv.org/abs/2609.24089](https://arxiv.org/abs/2609.24089)

    FlashBoB提出了一种精确且I/O高效的softmax注意力反向之反向算法，通过利用softmax双重反向中的分层仿射结构，将计算保持在片上tile内并避免所有N×N中间张量，从而在长序列场景下支持二阶优化、测试时训练和元学习等应用。

    

    基于注意力机制构建的Transformer模型已成为现代深度学习的核心构建模块，然而softmax注意力仍然是长上下文工作负载的主要瓶颈。虽然FlashAttention使前向传播和第一次反向传播实现了I/O高效，但它不支持反向之反向，而后者能够对反向传播过程进行精确微分，可应用于二阶优化、测试时训练、基于梯度的记忆和元学习等场景。现有的BoB实现要么需要物化大型中间张量，要么在长序列长度下耗尽GPU显存。我们提出了FlashBoB，这是一种用于softmax注意力中反向之反向的精确、I/O高效算法，它将计算保持在片上tile内，并避免了所有N×N的中间张量，其中N为序列长度。其关键洞察是softmax双重反向中的分层仿射结构：两个行标量即可决定所有……（摘要在此处截断）

    arXiv:2609.24089v1 Announce Type: new  Abstract: Transformer models built on the attention mechanism have become a central building block in modern deep learning, yet softmax attention remains a major bottleneck for long-context workloads. While FlashAttention makes the forward and first backward passes I/O-efficient, it does not support backward-over-backward (BoB), which enables exact differentiation through the backward pass for applications such as second-order optimization, test-time training, gradient-based memory, and meta-learning. Existing BoB implementations either materialize large intermediate tensors or exhaust GPU memory at long sequence lengths. We present FlashBoB, an exact, I/O-efficient algorithm for BoB in softmax attention that keeps computation within on-chip tiles and avoids all $N \times N$ intermediate tensors, where $N$ is the sequence length. The key insight is a hierarchical affine structure in the softmax double backward: two row-wise scalars determine all o
    
[^236]: 语义投递网络：为LLM智能体重构Web检索基础设施

    Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents

    [https://arxiv.org/abs/2609.22486](https://arxiv.org/abs/2609.22486)

    该论文提出“语义投递网络”，主张将Web检索基础设施从面向人类客户端的URL/摘要检索模式，重构为直接向LLM智能体提供面向任务效用的语义块检索服务，并支持跨轮次有状态检索与智能体间的工作共享。

    

    大语言模型（LLM）在回答需要专有信息或最新实时网页内容的问题时，越来越多地依赖外部来源，其方式既包括传统的单次检索增强生成（RAG），也包括多轮智能体式RAG（agentic RAG）。然而，当今的网络基础设施仍然是为人类客户端构建的：给定一个查询，当前的搜索服务返回的是按通用相关性排序的URL和摘要列表；内容分发网络（CDN）缓存以URL为地址的对象（文本、图像、视频等），但并不知道智能体究竟需要哪一段内容。相比之下，LLM消费的是简短且语义连贯的段落（下文称为“语义块”），这些语义块应根据下游任务的效用而非仅仅依据相似性来选择，且LLM可能在多个推理轮次中进行有状态的检索。此外，缺乏协调的智能体还会重复执行搜索、数据获取和语义处理，导致本可共享的工作被重复完成。我们主张，语义块的检索应当成为

    arXiv:2609.22486v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly rely on external sources when answering questions that require proprietary information or up-to-date live web content, through both traditional single-shot retrieval-augmented generation (RAG) and multi-turn agentic RAG. Yet today's web infrastructure is still built for human clients. Given a query, current search services return a list of URLs and snippets ranked for generic relevance; content delivery networks (CDNs) cache URL-addressed objects (texts, images, videos, etc.) without knowing which passage an agent needs. LLMs, in contrast, consume short, semantically coherent passages, hereafter "chunks", selected for downstream task utility rather than similarity alone, and may retrieve statefully across reasoning turns. Uncoordinated agents also repeat search, data acquisition, and semantic processing, duplicating work that could be shared. We argue that semantic chunk retrieval should become
    
[^237]: 基于FL-Net的多中心医学数据挖掘——联邦学习的一站式解决方案

    Multi-center Medical Data Mining with FL-Net - A One-stop Shop for Federated Learning

    [https://arxiv.org/abs/2609.20650](https://arxiv.org/abs/2609.20650)

    FL-Net是一个一站式联邦学习临床研究框架，集成了数据协调、数据发现、披露控制和容器化联邦工作流执行，填补了现有14个框架均无法完全满足的五项要求，并将在欧盟项目中覆盖10家医院的80万余名患者。

    

    联邦学习使协作训练成为可能，且无需共享患者级别的数据，但大多数研究仍停留在模拟阶段。基于从文献中总结出的五项要求，我们分析了14个联邦学习框架，发现没有框架能完全满足这些要求。我们提出了FL-Net，这是一个新颖的联邦临床研究框架，可满足所有要求。它将模块化数据协调、数据发现、披露控制、安全构建的版本化FL-Net-Tools以及容器化的联邦工作流执行集成到一个持久网络中。它使协调后的数据和工作流能够在不同研究之间重复使用。FL-Net的端到端能力通过数据协调、跨MIMIC和US-130数据集的跨研究患者发现，以及支持多达50个并发客户端的可重现、经过审计的联邦工作流得到了评估。FL-Net正在dAIbetes和Microb-AI-ome欧盟项目内开发，将覆盖10家医院的超过80万名患者。

    arXiv:2609.20650v1 Announce Type: new  Abstract: Federated learning enables collaborative training without sharing patient-level data, but most studies remain simulations. Based on five requirements derived from the literature, we analyzed 14 FL frameworks and found that none fully satisfied these requirements. We present FL-Net, a novel federated clinical research framework to fulfill all requirements. It integrates modular data harmonization, data discovery, disclosure control, securely built versioned FL-Net-Tools and containerized federated workflow execution into a persistent network. It enables the re-use of harmonized data and workflows across studies. FL-Net's end-to-end capabilities were evaluated through harmonization, cross-study patient discovery across MIMIC and US-130, and reproducible, audited federated workflows with up to 50 concurrent clients. FL-Net is being developed within the dAIbetes and Microb-AI-ome EU projects and will cover over 800,000 patients across 10 hos
    
[^238]: 面向不完美教师模型离线在线策略蒸馏的轨迹可学习性

    Trajectory Learnability for Offline On-Policy Distillation with Imperfect Teachers

    [https://arxiv.org/abs/2609.18321](https://arxiv.org/abs/2609.18321)

    本文提出利用教师成功问题作为廉价参考，通过观测学生模型在不同训练阶段对教师失败轨迹中各token似然的变化（带符号似然变化），来识别不完美教师监督下仍然可学习的内容，从而解决离线在线策略蒸馏中不完美监督持续存在的问题。

    

    离线在线策略蒸馏通过一次性收集学生轨迹和教师监督，并在整个优化过程中重复使用，从而获得效率优势。然而，同样的重复使用机制也使得不完美的监督持续存在。由于即使是强大的教师模型也可能失败，我们提出这样一个问题：从不完美的教师监督中，究竟还有哪些内容是可学习的？教师失败只是一个粗粒度的问题级别信号，并不意味着相关学生轨迹上的所有监督都是无益的。一种自然的替代方案是沿轨迹估计教师的可恢复性，但反复的续写生成会大大削弱离线蒸馏的效率优势。我们转而利用教师成功的问题来定义一个廉价的参考，以衡量学生可以学习的内容。我们在教师成功的问题上进行训练，并测量来自教师失败问题的轨迹中每个观测token的似然如何变化。我们将这些带符号的似然变化作为一种可操作的（摘要在此处截断）

    arXiv:2609.18321v1 Announce Type: cross  Abstract: Offline on-policy distillation gains efficiency by collecting student trajectories and teacher supervision once and reusing them throughout optimization. The same reuse makes imperfect supervision persistent. Since even strong teachers can fail, we ask \emph{what remains learnable from imperfect teacher supervision?} Teacher failure is only a coarse problem-level signal and does not imply that all supervision along the associated student trajectory is unhelpful. A natural alternative is to estimate teacher recoverability along the trajectory, but repeated continuations largely erase the efficiency advantage of offline distillation. We instead use teacher-successful problems to define a cheap reference for what the student can learn. We train on teacher-successful problems and measure how the likelihood of each observed token in trajectories from teacher-failed problems changes. We use these signed likelihood changes as an operational \
    
[^239]: 内核管理的共享内存实现系统级个性化

    Kernel-Managed Shared Memory for System-Wide Personalization

    [https://arxiv.org/abs/2609.10144](https://arxiv.org/abs/2609.10144)

    提出内核管理的共享内存这一系统级抽象，由智能体系统内核统一负责多智能体间的记忆检索、隐私管控和提示注入，相比非管理的记忆后端显著提升了系统级个性化效果。

    

    当AI系统能够适应使用它们的用户时，其效用会显著提升，但在多智能体系统中，一个智能体学到的有用上下文往往无法被其他智能体利用。我们提出了内核管理的共享内存，这是一种系统级抽象：专门的智能体负责写入结构化的、带标签的记忆，而由智能体系统内核（而非单个智能体）来统一管理检索、隐私执行和提示注入。我们在AIOS上实现并评估了这一设计，并在三个助手模型（GPT-4o、Llama-3.1:8B、Qwen-2.5:7B）上共进行了1,800次试验，与三种替代方案进行了对比。在使用相同底层存储的非管理外部内存后端（Mem0）的对比中，内核管理的检索和注入使个性化得分在5分制上提高了2.4-4.0分（例如，GPT-4o上的用户画像使用得分从1.05提升至4.69），且所有对比结果均显著（p < 10⁻¹⁸）。与标准检索增强注入相比，性能提升同样……（摘要不完整）

    arXiv:2609.10144v1 Announce Type: new  Abstract: AI systems become more useful when they can adapt to the people using them, but in multi-agent systems, useful context learned by one agent often remains unavailable to others. We present kernel-managed shared memory, a system-level abstraction in which specialized agents write structured, tagged memories while the agent-system kernel, not individual agents, governs retrieval, privacy enforcement, and prompt injection. We implement and evaluate this design on AIOS and compare it against three alternatives across three assistant models (GPT-4o, Llama-3.1:8B, Qwen-2.5:7B) and 1,800 total trials. Against an unmanaged external memory backend (Mem0) using identical underlying storage, kernel-managed retrieval and injection improve personalization scores by 2.4-4.0 points on a 5-point scale (e.g., 1.05 to 4.69 profile usage on GPT-4o), with every comparison significant at p < 10^-18. Against standard retrieval-augmented injection, gains are si
    
[^240]: 用于生成可变尺寸3D分子的固定维度潜在流

    Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules

    [https://arxiv.org/abs/2609.08333](https://arxiv.org/abs/2609.08333)

    该论文提出了一种两阶段生成框架，通过单一固定维度的分子级潜在表示结合自回归Transformer解码器，实现了无需预先指定分子大小的可变尺寸3D分子生成，并摒弃了等变层和密集成对键解码器的需求。

    

    在分子发现中，分子大小与组成、结构及其他目标性质相互耦合。然而，大多数3D生成器需要在生成前预先指定分子的大小。在此，我们提出了无等变Transformer自编码潜在流匹配，这是一个两阶段生成框架，完全依赖于单一固定维度的分子级潜在表示来生成可变尺寸的分子。第二阶段的流匹配模型对该潜在向量进行采样，随后自回归Transformer解码器在生成原子类型、坐标和化学信息状态的同时确定分子大小。规范的原子排序和刚体姿态对齐使得无需等变层的标准Transformer成为可能，而分子几何结构与丰富化学状态的联合解码实现了可靠、确定性的、化学引导的图恢复，无需学习密集的成对键解码器。

    arXiv:2609.08333v1 Announce Type: cross  Abstract: In molecular discovery, molecule size is coupled to composition, structure, and other target properties. Yet most 3D generators require molecule size to be specified before generation. Here, we introduce Equivariant-Free Transformer-Autoencoded Latent Flow Matching, a two-stage generative framework that relies entirely on a single fixed-dimensional molecule-level latent representation to generate variable-size molecules. The second-stage flow matching model samples this latent vector, and an autoregressive Transformer decoder then determines molecule size while generating atom types, coordinates, and chemically informative states. Canonical atom ordering and rigid-pose alignment enable standard Transformers without equivariant layers, while joint decoding of molecular geometry and an enriched chemical state enables reliable, deterministic, chemistry-guided graph recovery without requiring a learned dense pairwise bond decoder. The same
    
[^241]: 接口诱导的轨迹审查

    Interface-Induced Trajectory Censoring

    [https://arxiv.org/abs/2609.03966](https://arxiv.org/abs/2609.03966)

    该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。

    

    智能体评估报告的工具调用率是从服务栈中读取的。即使模型正在发出格式良好的调用，该数字也可能为零：接口在下游任何组件看到轨迹之前就将其“审查”掉了。在 BFCL v4 自身的数据、执行器和评分器上，保持权重、测试用例、解码方式和随机种子固定不变，仅更换服务适配器，同一模型的得分就可以是 0.00 或 0.96 / 0.19。对聊天模板和解析器进行的 2x2 析因实验精确定位了该效应：两个主效应都恰好为零，全部效应都落在交互作用中——没有任何组件是有缺陷的，只修复契约的一方毫无收效。在 tau-bench 的 115 个交互式零售任务上，同样的更换使服务器解析的调用从 0 增加到 636，到达工具执行环节的任务从 0 增加到 103。我们的探针实验在 Qwen2.5-Coder 的 21 倍规模范围内重现了这一漏斗效应：服务器在每个规模下都只能解析 0/100，而格式良好的发出调用在 32B 时上升到 80/100（审查后约 72……（摘要截断）

    arXiv:2609.03966v1 Announce Type: new  Abstract: Agent evaluations report a tool-call rate read off the serving stack. That number can be zero while the model is emitting well-formed calls: the interface censors the trajectory before anything downstream sees it.   On BFCL v4's own data, executor and scorer, holding weights, cases, decoding and seeds fixed and changing only the serving adapter, the same model scores 0.00 or 0.96 / 0.19. A 2x2 over chat template and parser locates the effect exactly: both main effects are exactly zero and all of it sits in the interaction -- no component is defective, and repairing one side of the contract buys precisely nothing. On tau-bench's 115 interactive retail tasks the same swap moves server-parsed calls from 0 to 636 and tasks reaching any tool execution from 0 to 103. Our probe reproduces the funnel across a 21x scale range of Qwen2.5-Coder: the server parses 0/100 at every size while well-formed emitted calls rise to 80/100 at 32B (~72 after c
    
[^242]: 机器学习在定向进化中的进展：五年回顾

    Advances in Machine Learning for Directed Evolution: A Five-Year Retrospective

    [https://arxiv.org/abs/2609.03046](https://arxiv.org/abs/2609.03046)

    本文回顾指出，机器学习辅助定向进化（MLDE）未能像其他蛋白质工程领域那样取得变革性进展，其主因是研究者追求“最优蛋白质”的目标与定向进化在时间和资源约束下寻找“足够好蛋白质”的实际需求脱节，例如忽视DNA合成成本导致方法缺乏实际适用性。

    

    过去五年多来，机器学习（ML）的进展已经改变了许多蛋白质工程学科，但定向进化却并非如此。基于对一篇先前共同撰写的观点文章的反思，我讨论了为什么我认为情况如此，并认为机器学习辅助定向进化（MLDE）研究者的目标——“识别最优蛋白质”——与更广义的定向进化目标——“在时间和资源限制下识别足够好的蛋白质”——之间的脱节是主要原因。作为一个例子，我强调了几乎所有当前的MLDE方法都忽略了DNA合成的成本，导致这些策略无论其底层模型的能力如何，实际适用性都很有限。最后，我讨论了近期一些打破这一总体趋势的工作，并强调了过去五年机器学习辅助蛋白质工程研究的努力（摘要内容不完整，原文在此处被截断）。

    arXiv:2609.03046v1 Announce Type: cross  Abstract: The last five-plus years have seen many protein engineering disciplines transformed by advances in machine learning (ML), but the same cannot be said for directed evolution. Reflecting on a previously co-authored perspective, I discuss why I believe this to be the case, arguing that a disconnect between the goals of machine-learning-assisted directed evolution (MLDE) researchers--"identify an optimal protein"--and the goals of directed evolution more broadly--"identify a sufficient protein given time and resource constraints"--is a principal culprit. As an example, I highlight how nearly all current MLDE methods neglect to account for the cost of DNA synthesis, resulting in strategies that have limited practical applicability regardless of the underlying models' capabilities. I close by discussing recent works that are exceptions to this overarching trend, and emphasize that the last five years of efforts in ML-assisted protein enginee
    
[^243]: 表格基础模型懂物理吗？数据污染、物理单位与确定性极限

    Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit

    [https://arxiv.org/abs/2609.02766](https://arxiv.org/abs/2609.02766)

    本文在316个物理方程采样的数据集上评估四种表格基础模型，发现其性能虽全面超越基线模型，但其贝叶斯先验既无法表示无噪声机制也无法表示物理单位，因此只能对物理数据进行插值而不能充当真正的物理模型。

    

    表格基础模型（TFMs）以语言模型填充文本的方式学习填充表格，而表格可以说是大多数物理测量数据所呈现的格式。它们在这一过程中学到了物理知识吗？由于它们在构造上是贝叶斯模型，因此问题在于其先验中包含什么。我们直接探测了这一问题，在从316个物理方程采样的数据集上（涵盖域内和域外），将四个模型（TabPFN-3、TabICLv2、TabDPT和Real-TabPFN-2.5）与六个基线模型进行了对比评估。TFMs无论开箱即用还是经过调优后都表现卓越。但我们证明了它们的先验既无法表示无噪声的物理机制，也无法表示物理单位，这正是它们能够对物理数据进行插值、却尚不能充当真正物理模型的原因。

    arXiv:2609.02766v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) learn to fill in tables the way language models fill in text, and tables are arguably the format in which most physical measurement arrives. Did they learn any physics in the process? They are Bayesian by construction, so the question is what their prior contains. We probe it directly, evaluating four of them (TabPFN-3, TabICLv2, TabDPT and Real-TabPFN-2.5) against six baselines on datasets sampled from 316 physical equations, in and out of domain. TFMs dominate, out of the box and after tuning. But we show that their prior can represent neither a noiseless mechanism nor physical units, which is why they interpolate physics without yet being able to act as physical models.
    
[^244]: 当预测误差不足够时：评估用于因果估计的干扰函数预测

    When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation

    [https://arxiv.org/abs/2609.00071](https://arxiv.org/abs/2609.00071)

    在部分线性模型的模拟研究中，干扰函数的预测误差无法一致地反映因果估计的偏差，表明仅用预测误差来评估干扰函数估计器对于因果推断而言是不充分的。

    

    预测误差被广泛用于评估因果推断中干扰函数估计器的性能，但其与因果估计器表现之间的关系可能因性能度量指标的不同而存在差异。我们在部分线性模型中通过蒙特卡洛模拟研究了这一问题。我们比较了普通最小二乘法（OLS）、广义可加模型（GAMs）、XGBoost 以及结合 XGBoost 的双重机器学习（DML-XGBoost），评估指标包括干扰函数预测误差、偏差、均方根误差（RMSE）以及95%置信区间覆盖率。我们还检验了一种简单的联合误差度量，其基于暴露干扰函数与结局干扰函数估计误差的绝对交叉乘积。在所有模拟设置中，XGBoost 在非先知方法中具有最低的 RMSE，而 DML-XGBoost 通常能提供更好的置信区间覆盖率。预测误差在不同方法和设置下并不能一致地反映因果估计的偏差，且点估计表现最好的方法（摘要在此处截断）

    arXiv:2609.00071v1 Announce Type: new  Abstract: Prediction error is widely used to evaluate nuisance-function estimators in causal inference, but its relationship with causal estimator performance may differ across performance measures. We studied this question in a partially linear model using Monte Carlo simulations. We compared ordinary least squares (OLS), generalized additive models (GAMs), XGBoost, and Double Machine Learning with XGBoost (DML-XGBoost), evaluating nuisance-function prediction error, bias, RMSE, and 95\% confidence interval coverage. We also examined a simple joint-error measure based on the absolute cross-product of estimation errors from the exposure and outcome nuisance functions. Across the simulated settings, XGBoost had the lowest RMSE among the non-oracle methods, while DML-XGBoost generally provided better confidence interval coverage. Prediction error did not consistently track causal bias across methods and settings, and the method with the best point-e
    
[^245]: 将均值推向已知的良好水平而非超越它：推理时干预与权重合并在开放式生成中能带来什么

    Moving the Mean Toward the Known Good, Not Beyond It: What Inference-Time Interventions and Weight Consolidation Buy in Open-Ended Generation

    [https://arxiv.org/abs/2608.28886](https://arxiv.org/abs/2608.28886)

    该研究发现在开放式生成任务中，通过“生成-验证-选择-LoRA合并”循环在自身验证成功的样本上训练，能显著提升模型输出的平均质量并经三重重复实验验证其稳健性，但最佳候选仅能收敛到经典启发式算法的水平而无法超越。

    

    一个生成循环能从学习自身经过验证的成功中获得什么？在针对在线装箱问题的“生成-验证-选择-LoRA合并”循环中，基于价值过滤候选样本的训练使模型在保留变体上的输出向更优价值偏移（超额量-1.7分，p=0.008；相比随机合并对照组为-3.1，p=0.004），而观察到的最佳候选收敛至经典启发式算法的水平且不再超越。一项确认实验组使用新随机种子和一个仅读取一次的从未接触过的保留集，将整个流程重复了三次：三个谱系中的均值几乎完全一致（-2.0、-1.8、-1.9），且在按保留变体内聚合后，全部七个可评估变体均支持合并方法（p=0.008）。观察到的最佳候选恰好移动到经典启发式算法的水平（三个谱系中均为0.021028，吸引条件与随机对照亦然），且从未超越该水平。一项匹配的SFT

    arXiv:2608.28886v1 Announce Type: cross  Abstract: What does a generation loop gain from learning on its own verified successes? In cycles of generate, verify, select and LoRA-consolidate on online bin packing, training on value-filtered candidates shifts what the model writes on held-out variants toward value (-1.7 points of excess, p=0.008; -3.1 against a random-consolidation control, p=0.004) while the best observed candidate converges to the classic heuristic's level and no further. A confirmation battery replicates the whole procedure three times, with fresh seeds and a never-consulted held-out set read exactly once: the mean was nearly identical in all three lineages (-2.0, -1.8, -1.9), and after aggregating within held-out variant all seven evaluable variants favored consolidation (p=0.008). The best observed candidate moved to the classic heuristic's level, exactly (0.021028 in all three lineages, for attract and for the random control alike), and never beyond it. A matched SFT
    
[^246]: 基于深度学习框架在GOTHIC巡天中解耦候选双活动星系核与偶然叠加

    Decoupling candidate dual AGN from chance superpositions in the GOTHIC survey via a deep-learning framework

    [https://arxiv.org/abs/2608.24164](https://arxiv.org/abs/2608.24164)

    本文通过YOLOv11定向边界框深度学习框架，有效从GOTHIC巡天的偶然叠加中区分出真实双活动星系核，实现了高精度检测。

    

    双活动星系核（DAGN）标志着合并星系和超大质量黑洞配对演化中的关键阶段，然而由于投影效应和有限的空间分辨率，它们在大规模成像巡天中仍难以识别。致密前景恒星和未解析的子结构可能通过偶然叠加模拟双核，从而干扰自动检测。我们重新审视了GOTHIC流程中标记但被拒绝的46,061个DAGN候选星系，主要原因在于两个核落在SDSS光纤孔径内或超过其分离阈值。我们基于YOLOv11定向边界框架构，在带注释的SDSS成像上训练了一个监督深度学习框架，以区分真正的双核与前景恒星污染物及其他虚假对齐。最终模型在双核类别上实现了验证精度0.919、召回率0.905和F1分数0.912。

    arXiv:2608.24164v1 Announce Type: cross  Abstract: Dual active galactic nuclei (DAGN) mark a critical phase in the evolution of merging galaxies and the pairing of supermassive black holes, yet they remain difficult to identify in large imaging surveys because of projection effects and limited spatial resolution. Compact foreground stars and unresolved substructure can mimic dual nuclei through chance superposition, complicating automated detection. We revisit the 46,061 galaxies flagged but rejected as DAGN candidates by the GOTHIC pipeline, primarily because the two nuclei fell within the SDSS fibre aperture or exceeded its separation threshold. We train a supervised deep-learning framework based on the YOLOv11 oriented-bounding-box architecture on annotated SDSS imaging to separate genuine dual nuclei from foreground stellar contaminants and other spurious alignments. The final model attains a validation precision of 0.919, recall of 0.905, and $F_1$ of 0.912 for the dual-nuclei cla
    
[^247]: 基于大语言模型增强的多图学习与对比对齐的兴趣点推荐

    POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment

    [https://arxiv.org/abs/2608.16407](https://arxiv.org/abs/2608.16407)

    本文提出LLM-MGCL，通过结合语义和地理信息的多图学习与对比对齐，有效解决了POI推荐中的冷启动问题。

    

    摘要：基于图神经网络的兴趣点（POI）推荐模型通过在用户-物品交互上传播协同信号实现强性能，但它们在冷启动问题上表现不佳，即交互很少或没有交互的物品无法被有效表示。在本文中，我们提出了大语言模型增强的多图对比学习（LLM-MGCL），这是一种多图神经网络，利用物品的语义和空间信息，通过两个辅助物品-物品图扩展LightGCN骨干网络：一个是由大语言模型生成的图片摘要和关键词的句子嵌入构建的语义图，另一个是根据商业地点之间的哈弗辛距离推导出的地理图。物品嵌入在所有三个图上并行传播，通过加性融合，并通过双向InfoNCE对比目标跨视图对齐，连接同一物品的行为、语义和空间表示。

    arXiv:2608.16407v1 Announce Type: cross  Abstract: Point-of-interest (POI) recommendation models based on graph neural networks achieve strong performance by propagating collaborative signals over user-item interactions, yet they struggle with the cold-start problem, where items with few or no interactions are not represented. In this paper, we propose LLM-augmented Multi-Graph Contrastive Learning (LLM-MGCL), a multi-graph neural network that uses semantic and spatial information about items to extend the LightGCN backbone with two auxiliary item-item graphs: a semantic graph constructed from sentence embeddings of LLM-generated photo summaries and keywords, and a geographic graph derived from Haversine distances between business locations. Item embeddings are propagated over all three graphs in parallel, fused additively, and aligned across views through a bidirectional InfoNCE contrastive objective that connects behavioral, semantic, and spatial representations of the same items. Ex
    
[^248]: 学习物理场离散黎曼度量与余链框架等变性

    Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivarianc

    [https://arxiv.org/abs/2608.14556](https://arxiv.org/abs/2608.14556)

    RHMP通过将拓扑与几何分离并学习余链度量，实现了物理场的精确守恒与几何依赖传播的统一。

    

    arXiv:2608.14556v1 公告类型：新 摘要：网格上的物理场需要将拓扑与几何分离：守恒定律是拓扑的，应当精确成立，而几何、材料响应和各向异性耦合必须从数据中学习。现有的神经替代模型常常在无约束的消息传递中混淆这些角色。我们引入了黎曼霍奇消息传递（RHMP），将这种分离转化为架构原则。RHMP固定由定向关联决定的细胞余边界（$d_k$），并学习对称正定余链度量（$H_k$）以进行几何依赖的传播。将$H_k$视为学习度量，激发了余链框架等变性：物理传播应对隐藏余链特征基的正交变化保持不变。RHMP通过度量加权霍奇块（$d_k^\top H_{k+1}d_k$）实现这一原则，从而确保精确的余链复形恒等式（$d_{k+1}d_k=0$）和非负霍奇能量。

    arXiv:2608.14556v1 Announce Type: new  Abstract: Physical fields on meshes require a separation between topology and geometry: conservation laws are topological and should be exact, while geometry, material response, and anisotropic coupling must be learned from data. Existing neural surrogates often mix these roles inside unconstrained message passing. We introduce Riemannian Hodge Message Passing (RHMP), which turns this separation into an architectural principle. RHMP fixes the cellular coboundaries ($d_k$) determined by oriented incidence and learns symmetric positive-definite cochain metrics ($H_k$) for geometry-dependent propagation. Treating $H_k$ as the learned metric motivates cochain-frame equivariance: physical propagation should be invariant to orthogonal changes of the hidden cochain feature basis. RHMP implements this principle with metric-weighted Hodge blocks ($d_k^\top H_{k+1}d_k$), yielding exact cochain-complex identities ($d_{k+1}d_k=0$), nonnegative Hodge energies,
    
[^249]: 橡皮筋丢弃法：一种缓解基于评分标准奖励强化学习中奖励黑客问题的简单方法

    Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL

    [https://arxiv.org/abs/2608.11669](https://arxiv.org/abs/2608.11669)

    本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。

    

    摘要：基于评分标准的强化学习，即由LLM评判员根据一系列标准进行评分，已成为对没有确定性答案的任务进行语言模型后训练的标准方法。然而，评分标准是质量的固定代理，永远无法完全描述质量，长期针对其训练的策略会学会利用这一差异。我们直接对此进行了测量。使用组相对策略优化（GRPO）在医学和科学评分标准上训练Qwen3-8B，并使用训练评判员和更强的黄金评判员对分布外（OOD）基准进行评分，我们发现这两个分数在训练过程中出现分歧。训练评判员的分数持续上升，而黄金评判员的分数达到峰值后下降，在HealthBench-Hard上下降3分，在ResearchQA上下降22分。如果评判员存在固定偏差，黄金曲线只会发生恒定偏移，而不会在训练分数上升时下降，因此这种分歧是奖励黑客问题，而非评判员偏差。

    arXiv:2608.11669v1 Announce Type: cross  Abstract: Reinforcement learning against rubrics, lists of criteria graded by an LLM judge, has become a standard way to post-train language models on tasks with no deterministic answer. The rubric, however, is a fixed proxy for quality, never a complete description of it, and a policy trained against it long enough will learn to exploit the difference. We measure this directly. Training Qwen3-8B with Group Relative Policy Optimization (GRPO) on medical and science rubrics and grading out-of-distribution (OOD) benchmarks with both the training judge and a stronger gold judge, we find that the two scores diverge during training. The training judge's score keeps climbing while the gold judge's score peaks and then falls, by 3 points on HealthBench-Hard and by 22 points on ResearchQA. A judge with a fixed bias would shift the gold curve by a constant, not send it down while the training score rises, so the divergence is reward hacking, not judge no
    
[^250]: 当自我一致性适得其反：多数投票损害小型语言模型在多数硬科学问题上的表现

    When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs

    [https://arxiv.org/abs/2608.11403](https://arxiv.org/abs/2608.11403)

    本研究发现，在GPQA Diamond基准上，多数投票的自我一致性方法反而降低了小型语言模型在大多数硬科学问题上的准确率，并预注册验证了这一反直觉现象。

    

    arXiv:2608.11403v1 公告类型：新 摘要：通过多数投票实现的自我一致性（SC）是一种广泛使用的推理时计算扩展方法：采样N条思维链，返回多数答案。在完整的GPQA Diamond基准测试（198个研究生级科学问题）上，多数投票在两个不同家族的指令微调模型上降低了大多数问题的逐题准确率：Qwen2.5-7B的56.6%问题和Llama-3-8B的65.7%问题，其中Qwen为主要演示模型，Llama从接近随机的基线证实了这一趋势。该效应在47个探索性问题上观察到后，在151个问题的确认性分割上进行了预注册，所有四项确认性假设均通过。一个网格神谕（将每个问题路由至{1, 2, 4, 8, 16, 32, 64}中的最佳N值）标志着理论上限，Qwen在N=1基础上高出14个准确率点，Llama高出17个点，但这一神谕上限需要真实标签而非可部署方法。没有无验证器的门控可以达到此效果。

    arXiv:2608.11403v1 Announce Type: new  Abstract: Self-consistency (SC) via majority vote is a widely used way to spend inference-time compute: sample N chains of thought, return the plurality answer. On the full GPQA Diamond benchmark (198 graduate-level science questions), majority voting reduces per-problem accuracy on a majority of problems for two instruction-tuned models from different families: 56.6% of problems for Qwen2.5-7B and 65.7% for Llama-3-8B, with Qwen the primary demonstration and Llama corroborating the direction from a near-chance baseline. The effect was pre-registered on a 151-problem confirmatory split after being observed on 47 exploratory problems, and all four confirmatory hypotheses passed. A grid oracle that routes each problem to the best N across {1, 2, 4, 8, 16, 32, 64} marks a theoretical upper bound 14 accuracy points above N = 1 for Qwen and 17 for Llama, an oracle bound requiring ground truth rather than a deployable method. No verifier-free gate reach
    
[^251]: 基于最小数据适应通用机器人策略

    Adaptation of Generalist Robot Policies with Minimal Data

    [https://arxiv.org/abs/2608.11363](https://arxiv.org/abs/2608.11363)

    本文提出了MiDAS方法，通过最小数据（单次或少量演示）结合离线到在线强化学习，使预训练机器人策略能高效适应新任务，验证了最小人工指导可启动自主学习的可行性。

    

    arXiv:2608.11363v1 公告类型：交叉 摘要：机器人学习的一个核心目标是超越针对特定任务的人工数据收集，转向通过自主交互来改进的机器人。然而，在当前策略下，完全自主的学习仍然困难：稀疏奖励和弱的零样本探索使得机器人不太可能从零开始发现成功行为。我们研究了最小数据适应，这是一种预训练机器人策略必须从尽可能少的一个演示开始，随后进行自主在线交互来学习新任务的机制。这一设置作为完全自主改进的最接近可行代理，使我们能够研究最小人工指导是否能启动自主学习，以及哪些算法成分使其可行。我们构建了MiDAS，一个简单的离线到在线强化学习配方，首先通过单次或少量演示的行为克隆将预训练的VLA锚定到目标任务，然后通过基于价值的在线强化学习进行改进。

    arXiv:2608.11363v1 Announce Type: cross  Abstract: A central goal in robot learning is to move beyond task-specific human data collection toward robots that improve through autonomous interaction. Yet fully autonomous learning remains difficult with current policies: sparse rewards and weak zero-shot exploration make it unlikely that a robot will discover successful behavior from scratch. We study minimal-data adaptation, a regime in which a pre-trained robot policy must learn a new task from as little as one demonstration followed by autonomous online interaction. This setting serves as the closest tractable proxy for fully autonomous improvement, allowing us to study whether minimal human guidance can bootstrap autonomous learning and what algorithmic ingredients make it feasible. We build MiDAS, a simple offline-to-online RL recipe that first anchors a pre-trained VLA to the target task with behavior cloning on single/few demonstrations, then improves it through value-based online R
    
[^252]: 用于条件分布灵活估计的并行梯度提升方法

    Parallel gradient boosting for flexible estimation of conditional distributions

    [https://arxiv.org/abs/2607.13550](https://arxiv.org/abs/2607.13550)

    该论文提出了一种称为“并行梯度提升”的梯度提升算法改进方法，通过在每次迭代中仅训练一个基模型即可同时预测多个目标（如整个条件分布），克服了传统提升方法在多输出回归任务中计算开销大、且现有高效变体受限于特定损失函数和决策树基学习器的局限。

    

    提升方法是标准分类和回归任务中最成功的学习技术之一。近年来，其向多输出预测问题的扩展得到了越来越多的应用。其中一个应用是对整个条件分布而非单一泛函进行预测，这类问题通常可以表述为多输出回归问题，例如多分位数回归。使用经典的提升方法实现来解决此类问题在计算上极具挑战性，因为通常在每次迭代中需要为每个目标分别训练一个基模型。为了加速训练，已有研究提出了更高效的提升方法变体，但它们往往局限于特定的损失函数和特定的基学习器类别（通常是决策树）。在这项工作中，我们研究了梯度提升算法的一种改进版本，我们称之为并行梯度提升，旨在克服上述所有限制……

    arXiv:2607.13550v2 Announce Type: replace-cross  Abstract: Boosting is one of the most successful learning techniques for standard classification and regression tasks. Its extension to multi-output prediction problems has found an increasing number of applications in recent years. Among them is the prediction of entire conditional distributions rather than single functionals, which can often be framed as a multi-output regression problem, for example multiple quantile regression. Addressing such problems with classical implementations of boosting is computationally challenging, because usually one base model is trained for each target at every iteration. More efficient variants of boosting have been proposed to speed up training, but they tend to be tied to specific loss functions and classes of base learners, usually decision trees. In this work, we study a modification of the gradient boosting algorithm, which we call parallel gradient boosting, designed to circumvent all these limit
    
[^253]: AirPASS：基于夹持天线系统的空中联邦学习

    AirPASS: Over-the-Air Federated Learning via Pinching Antenna Systems

    [https://arxiv.org/abs/2607.06768](https://arxiv.org/abs/2607.06768)

    本文提出AirPASS交替优化框架，通过同伦-黎曼裕度巩固方法与同伦辅助几何优化方法交替求解设备选择、接收波束成形和夹持天线位置的联合非凸优化问题，在夹持天线系统辅助的空中联邦学习中实现聚合失真阈值约束下的设备接入数量最大化。

    

    本文研究了接入点配备多波导夹持天线系统（PASS）的无线系统中的空中联邦学习（AirFL）。我们采用被广泛研究的面向学习的AirFL建模方式，其目标是在将聚合失真控制在预定阈值之下的同时，最大化被选择设备的数量。由此产生的设备选择、接收波束成形与夹持天线位置的联合优化问题，由于这些系统变量之间复杂的耦合关系而呈现高度非凸性。为应对这一挑战，我们提出了AirPASS——一个交替优化框架，包含两个核心组件：在固定PASS配置下，用于设备选择与接收波束成形的同伦-黎曼裕度巩固方法；以及在固定被选设备与波束成形器下，用于更新夹持天线位置的同伦辅助几何优化方法。

    arXiv:2607.06768v2 Announce Type: replace-cross  Abstract: This paper investigates over-the-air federated learning (AirFL) in wireless systems where the access point is equipped with a multi-waveguide pinching antenna system (PASS). We adopt the widely studied learning-oriented AirFL formulation, which seeks to maximize the number of selected devices while keeping the aggregation distortion below a prescribed threshold. The resulting joint optimization of device selection, receive beamforming, and pinching-antenna placement is highly nonconvex due to the intricate coupling among these system variables. To address this challenge, we develop AirPASS, an alternating optimization framework with two main components: a homotopy-Riemannian margin-consolidation method for device selection and receive beamforming under a fixed PASS configuration, and a homotopy-assisted geometry optimization method for updating the pinching-antenna positions under fixed selected devices and beamformer.
    
[^254]: 基于文本商业输入的LLM引导交通枢纽容量规划

    LLM-Guided Transportation Hub Capacity Planning with Textual Business Inputs

    [https://arxiv.org/abs/2607.03651](https://arxiv.org/abs/2607.03651)

    该论文提出一种LLM引导的枢纽容量规划框架，通过思维链推理将文本商业背景转化为结构化容量决策，并借助优化模型反馈回路进行验证，在美国东南部真实13枢纽货运网络上将最优性差距从传统方法的11.0%显著降低至2.8%。

    

    传统的枢纽容量规划模型虽然能够针对量化输入进行有效优化，但往往难以消化定性的商业背景信息。我们提出了一个新颖的框架，其中大语言模型（LLM）智能体在自然语言商业背景描述的引导下，迭代地提出枢纽容量决策。其关键机制是一种思维链推理协议：LLM构建一个结构化决策表，根据每个背景信息所隐含的变化方向与幅度，将其映射为具体的容量调整。随后，新的容量决策通过与优化模型之间的反馈回路进行验证，优化模型提供基于路径规划的性能指标来指导智能体的选择。在美国东南部一个真实的13枢纽货运网络上的实验表明，我们的框架相对于隐藏的真实最优解仅产生2.8%的最优性差距，相比传统优化方法11.0%的差距有显著改进。

    arXiv:2607.03651v3 Announce Type: replace  Abstract: While traditional hub capacity planning models optimize effectively for quantitative inputs, they often fail to digest qualitative business context. We propose a novel framework where a large language model (LLM) agent iteratively proposes hub capacity decisions guided by natural-language business context descriptions. The key mechanism is a chain-of-thought reasoning protocol: the LLM constructs a structured decision table that maps each contextual item to specific capacity adjustments based on the implied direction and magnitude of changes. The new capacity decision is then validated through a feedback loop with an optimization model, which provides routing-based performance metrics to guide the agent's selection. On a real-world 13-hub freight network in the southeastern US, our framework achieves a 2.8% optimality gap relative to the hidden ground-truth, a significant improvement over the 11.0% gap produced by the traditional opt
    
[^255]: 视觉-语言模型中视觉信息流动的通路

    Pathways of Visual Information Flow in Vision-Language Models

    [https://arxiv.org/abs/2607.03358](https://arxiv.org/abs/2607.03358)

    该研究发现视觉-语言模型通过“直接通路”和“文本中介通路”两条不同路径处理视觉信息，通路选择取决于任务类型、数据分布和提示设计，且在常用通路被消融时模型能灵活启用文本中介通路作为后备。

    

    我们研究视觉信息在视觉-语言模型（VLMs）中是如何被路由的。通过在受控的合成与自然数据集上使用因果补丁技术，我们发现模型依赖两种不同的通路来解决视觉任务：一种是直接通路，即视觉信息保留在图像token的表示中，并在后续层中被最后一个token读取；另一种是文本中介通路，即视觉信息首先被传递到查询token，然后由最后一个token读取。在三个视觉任务上，我们表明通路的选择依赖于具体任务，且数据分布和提示词设计也可以调节使用哪条通路来解决基于图像的查询。此外，通过注意力敲除和受损输入补丁方法，我们发现这些通路是灵活的——在某些干预下，当常用通路被消融时，模型可以依赖文本中介通路作为后备机制。这一行为统一了相关发现……

    arXiv:2607.03358v2 Announce Type: replace-cross  Abstract: We study how visual information is routed in vision-language models (VLMs). Using causal patching on controlled synthetic and natural datasets, we find that models rely on two distinct pathways to solve visual tasks: A direct pathway, where visual information is retained in image token representations and read out by the final token at later layers, and a text-mediated pathway, where visual information is first transferred to the query tokens and then read out by the final token. Across three visual tasks, we show that pathway selection is task-dependent, and that data distribution and prompt design can also modulate which pathway is used to solve the image-based query. Moreover, using attention knockouts and corrupted-input patching, we find that these pathways are flexible, under certain interventions, models can rely on the text-mediated pathway as a fallback when the usual pathway is ablated. This behavior unifies findings 
    
[^256]: 锐度与复杂度能在多大程度上共同解释泛化？

    How Far Can Sharpness and Complexity Jointly Explain Generalization?

    [https://arxiv.org/abs/2606.29043](https://arxiv.org/abs/2606.29043)

    本文定量研究了锐度与复杂度联合解释深度学习泛化能力的上限，发现恰当定义的度量对具有很强的解释力，其成功受锐度扰动尺度和复杂度表示方式的影响，但仍有失效情况，暗示可能需要额外因素。

    

    大规模实证研究表明，许多广泛使用的泛化度量与泛化能力仅呈弱相关，这表明仅依赖单一因素不足以解释泛化。在本工作中，我们联合考察锐度与复杂度这两个因素，并将双因素视角的适用范围作为核心问题：它们能在多大程度上共同解释泛化？什么决定了它们何时成功？它们又在哪里失效？我们以定量的方式研究这些问题，使用十六种替代度量对作为探针，并采用帕累托分析和线性回归作为评估工具。我们的结果表明，恰当定义的$(S,C)$度量对在一系列设置中都表现出很强的解释力。锐度扰动尺度对这种成功有显著影响，而复杂度的表示方式则影响其鲁棒性。剩余的失败案例留下了一个悬而未决的问题：更精细的度量是否足够，还是需要引入额外的因素。

    arXiv:2606.29043v2 Announce Type: replace  Abstract: Large-scale empirical studies have revealed that many widely used generalization measures correlate only weakly with generalization, suggesting that relying on a single factor is insufficient to explain it. In this work, we examine sharpness and complexity jointly, placing the reach of the two-factor view at the center: how far can they jointly explain generalization, what determines when they succeed, and where do they fall short? We investigate these questions quantitatively, using sixteen alternative metric pairs as probes and Pareto analysis and linear regression as evaluation tools. Our results show that suitably defined $(S,C)$ pairs provide strong explanatory power across a range of settings. The sharpness perturbation scale substantially affects this success, and the complexity representation influences its robustness. The remaining failures leave open whether more refined measurements would suffice or additional factors are 
    
[^257]: Giskard：面向大规模去中心化学习的拜占庭鲁棒且隐私保护的聚合方法

    Giskard : Byzantine Robust and Private Aggregation for Large-Scale Decentralized Learning

    [https://arxiv.org/abs/2606.19129](https://arxiv.org/abs/2606.19129)

    提出了Giskard，一种面向大规模去中心化学习的拜占庭鲁棒且隐私保护的聚合方法，通过分布式差分隐私与安全多方计算相结合，同时防御模型投毒攻击和隐私攻击，并解决了现有基于MPC的方案可扩展性差的问题。

    

    在去中心化学习中，客户端在本地保留数据训练机器学习模型，并与一组邻居共享其模型参数或梯度，这使得系统容易受到隐私攻击和模型投毒攻击的双重威胁。在去中心化学习中同时应对隐私问题和拜占庭行为极具挑战性。事实上，已知在客户端应用差分隐私（DP）会增加诚实模型更新的方差，从而削弱拜占庭鲁棒聚合对投毒攻击的抵抗能力。分布式DP可以通过在聚合后添加噪声来缓解这一问题，但在没有可信管理者的情況下实现该方法，需要同时具备机密性和拜占庭弹性的计算。最近的工作使用安全多方计算（MPC）来实现鲁棒聚合，同时保护输入的机密性，然而这些解决方案的可扩展性很差：它们要么需要全对全通信……

    arXiv:2606.19129v2 Announce Type: replace-cross  Abstract: In decentralized learning, clients train a machine learning model while keeping their data locally and share their model parameters or gradients with a set of neighbors, making it vulnerable to both privacy attacks and model poisoning attacks. Jointly addressing privacy and Byzantine behavior in decentralized learning is challenging. Indeed, Differential Privacy (DP) applied on the client side is known to increase the variance of honest model updates, which weakens Byzantine-robust aggregation against poisoning attacks. Distributed DP can mitigate this issue by adding noise after aggregation, but implementing it without a trusted curator requires both confidential and Byzantine-resilient computation. Recent work addresses this challenge using secure multi-party computation (MPC) to implement robust aggregation while preserving input confidentiality. However, these solutions scale badly: they either require all-to-all communicat
    
[^258]: SDS-LoRA：克服低秩适应中的各向异性梯度缩放

    SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation

    [https://arxiv.org/abs/2606.16454](https://arxiv.org/abs/2606.16454)

    本文提出SDS-LoRA方法，通过从几何角度分析并克服LoRA中由奇异值引起的各向异性梯度缩放问题，以改善低秩适应与全微调之间的对齐。

    

    低秩适应（LoRA）通过使用低秩矩阵参数化权重更新，使得大型预训练模型能够高效适应下游任务。在本文中，我们从几何角度研究了LoRA参数化的局限性。具体来说，我们表明当全微调梯度反向传播到低秩矩阵时，它会经历由奇异值驱动的各向异性缩放。我们认为这种现象是不理想的，因为它通过将梯度偏向主导奇异方向而抑制其他方向，从而扭曲了全微调梯度。我们的分析表明，各向异性梯度缩放降低了低秩矩阵梯度的有效秩，并且无法在LoRA中为任意梯度提供全微调梯度与其低秩近似之间的最佳对齐，从而加剧了与全微调的差距。为了解决这一问题，我们提出了SDS-LoRA。

    arXiv:2606.16454v2 Announce Type: replace-cross  Abstract: Low-Rank Adaptation (LoRA) enables efficient adaptation of large pretrained models to downstream tasks by parameterizing weight updates with low-rank matrices. In this paper, we investigate the limitations of the LoRA parameterization from a geometric perspective. Specifically, we show that when a full fine-tuning gradient is backpropagated to the low-rank matrices, it undergoes anisotropic scaling driven by their singular values. We argue that this phenomenon is undesirable because it distorts the full fine-tuning gradient by skewing it toward dominant singular directions while suppressing others. Our analyses demonstrate that anisotropic gradient scaling reduces the effective rank of the low-rank matrices' gradients and fails to provide the best possible alignment between the full fine-tuning gradient and its low-rank approximation in LoRA for arbitrary gradients, thereby exacerbating the gap to full fine-tuning. To address t
    
[^259]: 衡量公开医学视觉-语言基准中的预训练数据污染

    Measuring Pretraining Contamination in Public Medical Vision-Language Benchmarks

    [https://arxiv.org/abs/2606.10066](https://arxiv.org/abs/2606.10066)

    该研究使用四种检测方法系统评估公开医学视觉-语言基准的预训练数据污染，发现图像层面的重叠信号主要反映图像来源的重叠而非模型对个体图像的记忆。

    

    医学视觉-语言模型在公开基准上进行评估，而这些基准的图像和问答对已免费公开多年，但报告的准确率默认这些样本未出现在训练数据中。我们在SLAKE-En、PathVQA、VQA-RAD以及OmniMedVQA的公开镜像上，使用四种检测方法（针对PMC-OA-beta的图像近邻搜索、规范顺序可交换性、队列相对Min-K%++尾部富集、跨模型top-K重叠度）对开源VLM检验了这一假设。在图像方面，在SigLIP-B-16下，SLAKE-En中19.8%的图像在PMC-OA-beta中存在异常接近的近邻（SigLIP-SO400M下为4.2%），而域外对照图像未产生任何标记。人工检查显示这些是来自不同患者但具有相同模态和视角的图像，并非完全相同的副本，因此我们将这一信号解读为图像来源的重叠，而非对单个图像的记忆。在文本方面……

    arXiv:2606.10066v2 Announce Type: replace-cross  Abstract: Medical vision-language models (VLMs) are evaluated on public benchmarks whose images and question-answer pairs have been freely available for years, yet reported accuracy assumes that these examples were not in the training data. We test this assumption for open VLMs on SLAKE-En, PathVQA, and VQA-RAD, and on a public mirror of OmniMedVQA, using four detectors: image near-neighbour search against PMC-OA-beta, canonical-order exchangeability, cohort-relative Min-K%++ tail enrichment, and cross-model top-K overlap. On the image side, 19.8% of SLAKE-En images have an unusually close neighbour in PMC-OA-beta under SigLIP-B-16 (4.2% under SigLIP-SO400M), while out-of-domain control images produce no flags. Manual inspection shows that these are images of the same modality and view from different patients, not exact copies, so we read the signal as overlap in image sources rather than memorization of individual images. On the text si
    
[^260]: 机器学习中的行为一致性

    Behavioral Consistency in Machine Learning

    [https://arxiv.org/abs/2606.07581](https://arxiv.org/abs/2606.07581)

    该论文指出模型行为不仅取决于权重，还取决于执行它的内核，并提出“内核契约”框架，通过非渐近理论界将内核间的logit差异与总变差、KL散度、期望奖励变化以及策略梯度偏差定量联系起来。

    

    一个训练好的模型通常用其权重来描述，仿佛权重本身就决定了模型的行为。但在实践中，相同的权重会由不同的程序来运行：训练使用高精度累加和固定的张量形状，而推理服务则使用低精度算术、融合的注意力内核、分页键值缓存以及动态批处理。这些程序会从相同的权重中产生不同的下一个词元分布。我们通过用执行该策略的内核来对策略进行索引，使这种依赖关系显式化，并提出了“内核契约”，它规定了服务内核在每部分流量上相对于参考内核可以偏离多远，以及当超出限制时服务系统应当采取什么措施。我们证明了非渐近界，将两个内核之间的logit差异与总变差距离和KL散度、与期望奖励的变化、与基于另一内核生成的 rollout 所计算出的策略梯度的偏差联系起来……

    arXiv:2606.07581v2 Announce Type: replace  Abstract: A trained model is usually described by its weights, as if the weights fixed its behavior. In practice the same weights are run by different programs. Training uses high-precision accumulation and fixed tensor shapes, while serving uses low-precision arithmetic, fused attention kernels, a paged key-value cache, and dynamic batching. These programs produce different next-token distributions from identical weights. We make this dependence explicit by indexing a policy by the kernel that executes it, and we propose kernel contracts, which state how far a serving kernel may drift from a reference kernel on each slice of traffic and what the serving system should do when a limit is exceeded. We prove non-asymptotic bounds that link the logit difference between two kernels to total-variation and KL divergence, to the change in expected reward, to the bias of policy gradients computed from rollouts generated by another kernel, and to whethe
    
[^261]: 基于Fisher信息的模型鲁棒性度量：谱界、理论保证与实用算法

    Measuring Model Robustness via Fisher Information: Spectral Bounds, Theoretical Guarantees, and Practical Algorithms

    [https://arxiv.org/abs/2606.04767](https://arxiv.org/abs/2606.04767)

    本文提出了一种基于Fisher信息矩阵谱范数的、与攻击无关的模型鲁棒性度量，为常见架构推导了闭式谱界并给出首个理论鲁棒性排名，同时提供了适用于白盒和黑盒场景的高效估计算法。

    

    深度神经网络的鲁棒性对于安全关键型部署至关重要，然而现有的评估方法往往依赖于特定攻击且缺乏可解释性。我们提出了一种有原则的、与攻击无关的鲁棒性度量，该度量基于Fisher信息矩阵（FIM）的谱范数，量化了模型输出分布对输入扰动的最坏情况敏感度。在理论方面，我们证明了FIM等于输入雅可比矩阵的方差，并为VGG、ResNet、DenseNet和Transformer等常见架构推导了闭式谱界，提供了首个理论鲁棒性排名。为实现可扩展的评估，我们开发了高效的算法，包括幂迭代法和基于Hutchinson的估计方法，可同时支持白盒和黑盒两种设置。我们在CIFAR、ImageNet和医学图像等多个数据集以及多种架构上进行了大量实验……

    arXiv:2606.04767v2 Announce Type: replace  Abstract: The robustness of deep neural networks is crucial for safety-critical deployments, yet existing evaluation methods are often attack-dependent and lack interpretability. We propose a principled, attack-agnostic robustness metric based on the spectral norm of the Fisher Information Matrix (FIM), which quantifies the worst-case sensitivity of the model's output distribution to input perturbations. Theoretically, we establish that the FIM equals the variance of the input Jacobian and derive closed-form spectral bounds for common architectures, including VGG, ResNet, DenseNet, and Transformer, providing the first theoretical robustness ranking. To enable scalable evaluation, we develop efficient algorithms, including power iteration and Hutchinson-based estimation, that support both white-box and black-box settings. Extensive experiments across multiple datasets, including CIFAR, ImageNet, and medical images, and across multiple architect
    
[^262]: 重新思考大语言模型的分子文本表示：一项实证研究

    Rethinking Molecular Text Representations for LLMs: An Empirical Study

    [https://arxiv.org/abs/2606.03057](https://arxiv.org/abs/2606.03057)

    本文提出了系统性基准MolRepBench，通过对16个LLM在九种分子表示和八项化学任务上的评估，发现LLM的分子任务性能强烈依赖于表示形式且无通用最优解，其中CML整体最佳，结构化文本表示主导结构类任务而IUPAC主导语义类任务，SMILES变体虽在预训练中普遍使用却很少最优。

    

    大语言模型（LLM）在分子任务中的应用日益广泛，但究竟应使用哪种分子表示形式仍不明确。我们提出了一个系统性基准MolRepBench，用于评估LLM在九种分子表示和八项化学任务上的分子能力。我们对来自五个模型系列的16个大语言模型进行了基准测试，包括推理型与非推理型变体、化学专业LLM以及闭源前沿模型。结果显示，模型性能高度依赖于表示形式，且没有一种表示在所有任务上都占优，其中CML表现最佳，其次是MolJSON、InChI和标准SMILES。显式结构化文本表示（CML和MolJSON）在结构类任务中占主导地位；IUPAC命名法在语义类任务中占优，在全部16个LLM的分子检索任务中均获胜；而SMILES的各种变体尽管在预训练中被广泛使用，却很少是最优选择。化学专业模型在使用SMILES时表现良好，但使用结构化文本时性能会大幅下降。

    arXiv:2606.03057v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used for molecular tasks, but it remains unclear which molecular representation to use. We present a systematic benchmark, MolRepBench, to evaluate LLM molecular competence across nine representations and eight chemical tasks. We benchmark 16 LLMs across five model families, including reasoning and non-reasoning variants, chemistry-specialized LLMs, and closed frontier models. Performance is strongly representation-dependent and no single representation wins across tasks, though CML is the best, followed by MolJSON, InChI and canonical SMILES. Explicit structured text representations (CML and MolJSON) dominate structural tasks; IUPAC dominates semantic tasks, winning molecule retrieval for all 16 LLMs; and SMILES variants are rarely optimal despite their prevalence in pretraining. Chemistry-specialized models perform well with SMILES at the cost of large degradations with structured text 
    
[^263]: LSVD：面向高效低精度视觉语言模型的损失感知低秩近似

    LSVD: Loss-Aware Low-Rank Approximation for Efficient Low-Precision Vision-Language Models

    [https://arxiv.org/abs/2606.00573](https://arxiv.org/abs/2606.00573)

    提出了 LASER 框架，通过基于模型损失二阶近似的曲率加权 SVD 目标函数和 Kronecker 分解 Fisher 信息引导的秩分配，实现面向高效低精度推理的视觉语言模型低秩压缩。

    

    视觉语言模型（VLM）具备强大的多模态推理能力，但其高昂的计算成本和庞大的参数量使其难以在资源受限的设备上部署。低秩分解已成为一种有前景的压缩技术，然而现有方法通常只优化局部矩阵重构误差，依赖均匀或启发式的秩分配策略，并且主要关注注意力投影，而对前馈网络的探索不足。在本文中，我们提出了 LASER（损失感知奇异值分解与秩分配），这是一个用于实现高效低精度 VLM 推理的低秩压缩框架。LASER 从模型损失的二阶近似中推导出曲率加权的 SVD 目标函数，并利用 Kronecker 分解的 Fisher 信息来引导分解过程朝向下游任务性能优化，而非仅仅追求重构误差最小化。

    arXiv:2606.00573v2 Announce Type: replace  Abstract: Vision-language models (VLMs) deliver strong multimodal reasoning capabilities, but their large computational cost and high parameter counts make deployment challenging on resource-constrained devices. Low-rank decomposition has emerged as a promising compression technique, yet existing methods often optimize local matrix reconstruction error, rely on uniform or heuristic rank allocation, and focus mainly on attention projections while leaving feed-forward networks underexplored. In this paper, we propose~\textit{LASER} (\textbf{L}oss-\textbf{A}ware \textbf{S}ingular-value d\textbf{E}composition and \textbf{R}ank allocation), a low-rank compression framework for efficient low-precision VLM inference. LASER derives a curvature-weighted SVD objective from a second-order approximation of the model loss and uses Kronecker-factored Fisher information to guide decomposition toward downstream performance rather than reconstruction alone. We
    
[^264]: 《测量、定位与消融大语言模型中的对齐签名》

    Measuring, Localizing, and Ablating Alignment Signatures in LLMs

    [https://arxiv.org/abs/2605.30526](https://arxiv.org/abs/2605.30526)

    该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。

    

    经过对齐的大语言模型通常表现出一种可被识别的类AI风格，但这种风格与后训练及模型内部表征之间的联系仍知之甚少。在本工作中，我们研究了后训练是否引入或放大了类AI的风格规律性，以及这些规律性是否对应于一个可局部化的内部签名。为此，我们在匹配的人类来源前缀条件下，比较了人类文本、基座模型生成文本与对齐模型生成文本。结果表明，与基座模型的生成文本相比，对齐模型的生成文本具有更低的人类语料亲和度和更高的AI检测率，这说明后训练使生成文本偏离了人类语料风格，转向了检测器易于识别的类AI文本。随后，我们提出了PASTA（Post-training Alignment Signature Targeted Ablation，后训练对齐签名定向消融），这是一种无需训练的方法：它通过对比对齐模型与基座模型的残差来估计后训练对齐签名，并在解码过程中消融相应的方向。在11个……（摘要被截断）

    arXiv:2605.30526v2 Announce Type: replace-cross  Abstract: Aligned language models often exhibit a recognizable AI-like style, yet its connection to post-training and internal representations remains poorly understood. In this work, we study whether post-training introduces or amplifies AI-like stylistic regularities and whether these regularities have a localized internal signature. To this end, we compare human text, base-model generations, and aligned-model generations under matched human-source prefixes. Aligned generations show lower human-corpus affinity and higher AI-detection rates than base generations, suggesting that post-training shifts generated text away from human-corpus style and toward detector-visible AI-like text. We then introduce PASTA (Post-training Alignment Signature Targeted Ablation), a training-free method that estimates a post-training alignment signature from aligned-base residual contrasts and ablates the corresponding direction during decoding. Across 11 
    
[^265]: SoundnessBench：你的AI科学家真的能分辨好研究想法与坏研究想法吗？

    SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?

    [https://arxiv.org/abs/2605.30329](https://arxiv.org/abs/2605.30329)

    提出了包含1,099个真实研究提案的SoundnessBench基准，评估发现12个前沿大语言模型普遍存在乐观偏见（平均假阳性率74.0%），无法可靠地在投入资源前甄别研究想法的方法论缺陷。

    

    自主AI研究智能体旨在通过自动化从假设生成到同行评审的整个研究流程来加速科学发现。然而，现有基准很少测试一个根本性的瓶颈：大语言模型能否在投入时间和计算资源**之前**判断一个研究想法的方法论可行性。如果没有可靠的“第一道关卡”过滤器，自主智能体可能会规模化推广有缺陷的方法论，而非加速有意义的科学进展。我们提出了SoundnessBench，这是一个精心构建的基准，包含1,099个真实研究提案，具有来源验证的提取方式、来自评审员的代理标签，以及对其中327个提案子集的仅基于提案内容的专家判断。通过对12个前沿大语言模型的评估，我们揭示了一种普遍存在的**乐观偏见**：在标准提示下，平均假阳性率高达74.0%。采用激进提示可以减少错误批准，但会显著增加错误拒绝。

    arXiv:2605.30329v2 Announce Type: replace  Abstract: Autonomous AI research agents aim to speed up scientific discovery by automating the research pipeline from hypothesis generation to peer review. However, existing benchmarks rarely test a fundamental bottleneck whether LLMs can judge the methodological viability of a research idea $\textbf{before}$ investing time and computational resources. Without a reliable ``first-gate'' filter, autonomous agents risk scaling flawed methodology rather than accelerating meaningful science. We introduce $\textbf{SoundnessBench}$, a curated benchmark of 1,099 real research proposals, with source verified extraction, reviewer-derived proxy labels, and proposal-only expert judgments on a subset of 327 proposals. Across 12 frontier LLMs, our evaluation reveals a pervasive $\textbf{optimism bias}$: standard prompting yields a $74.0\%$ mean false-positive rate. Aggressive prompting reduces false approvals but substantially increases false rejections. Ad
    
[^266]: SAHG：面向社交机器人检测的扇区各向异性双曲图模型

    SAHG: Sector-Anisotropic Hyperbolic Graph Model for Social Bot Detection

    [https://arxiv.org/abs/2605.30166](https://arxiv.org/abs/2605.30166)

    该论文提出扇区各向异性双曲图模型SAHG，通过双曲几何适配社交图的层级无标度结构，并利用各向异性的曲率区分不同结构方向的密度差异，同时缓解社交机器人伪造异配连接导致的邻域信号污染问题，从而提升社交机器人检测的准确性。

    

    arXiv:2605.30166v3 通告类型：replace-cross 摘要：由大语言模型（LLM）驱动的社交机器人能够生成流畅的、类人风格的文本，这削弱了仅基于内容进行检测的判别优势。然而，协同行动仍会留下关系层面的模式——交互行为、行为相似性、共享邻域、社区位置以及协同活动——这些证据可以被基于图的方法所利用。现有图检测器在利用此类证据时面临两大挑战。其一，欧几里得图神经网络（GNN）会扭曲具有层级性和无标度特性的社交图；尽管双曲几何能够解决这种体积增长不匹配的问题，但固定曲率的模型仍对不同密度和分离需求的结构方向赋予统一的几何分辨率。其二，关系证据并不总是可靠的：高明的社交机器人会与真实用户建立伪造的异配连接，导致邻域聚合过程混合了机器人与人类用户的信号，从而稀释了账户层面的证据。我们提出了SAHG（扇区各向异性……

    arXiv:2605.30166v3 Announce Type: replace-cross  Abstract: LLM-driven social bots can generate fluent, human-like text, reducing the discriminative advantage of content-based detection alone. However, coordinated campaigns still leave relational patterns -- interactions, behavioral similarity, shared neighborhoods, community positions, and coordinated activity -- that graph-based methods can exploit. Existing graph detectors face two challenges when exploiting such evidence. First, Euclidean GNNs distort hierarchical and scale-free social graphs; while hyperbolic geometry addresses this volume-growth mismatch, fixed-curvature models still assign uniform geometric resolution to structural directions with different densities and separation needs. Second, relational evidence is not always reliable: sophisticated bots forge heterophilic connections with genuine users, causing neighborhood aggregation to mix bot and human signals and dilute account-level evidence. We propose SAHG (Sector-An
    
[^267]: 用于稀疏自编码器特征匹配与电路压缩的语义最优传输

    Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression

    [https://arxiv.org/abs/2605.28567](https://arxiv.org/abs/2605.28567)

    本文提出以激活加权分布和Wasserstein距离为核心的分布性框架，将稀疏自编码器的跨层特征匹配与电路压缩统一归结为不同激活流形间特征语义距离的估计问题。

    

    稀疏自编码器（SAE）已成为解释语言模型的核心工具。然而，有两项关键的SAE分析仍然难以扩展：（1）跨多层匹配语义相似的特征，以及（2）将大型特征电路压缩为可解释的超节点。尽管这些问题通常被当作独立问题来处理，我们证明了二者都是一个更根本挑战的具体实例，我们将其框架化为估计位于不同激活流形上的SAE特征之间的语义距离。我们为此问题引入了一个分布性框架，其中每个特征不再像现有文献中那样由单个解码器向量表示，而是由表达该特征的隐状态上的激活加权分布来表示。通过将这些分布投影到共享参考空间并使用Wasserstein距离进行比较，我们的方法为跨层特征比较提供了一种统一的语义度量。

    arXiv:2605.28567v2 Announce Type: replace  Abstract: Sparse autoencoders (SAEs) have become a central tool for interpreting language models. However, two key SAE analyses that remain difficult to scale are (1) matching semantically similar features across multi-layers and (2) compressing large feature circuits into interpretable supernodes. Although these have been treated as separate problems, we show that both are instances of a more fundamental challenge, which we frame as the estimation of semantic distances between SAE features that lie on different activation manifolds. We introduce a distributional framework for this problem, in which each feature is represented not by a single decoder vector like in the literature, but by an activation-weighted distribution over the hidden states that express it. By projecting these distributions into a shared reference space and comparing them with Wasserstein distance, our method provides a unified semantic metric for cross-layer feature comp
    
[^268]: 逐元素裁剪能否实现对随机梯度的谱控制？

    Can Entry-Wise Clipping Give Spectral Control of Stochastic Gradients?

    [https://arxiv.org/abs/2605.27733](https://arxiv.org/abs/2605.27733)

    本文提出一种简单的逐元素裁剪方法，能够以低成本实现对重尾随机梯度噪声的谱控制，从而平衡矩阵结构感知与计算开销之间的权衡，缓解训练中的损失突增等不稳定问题。

    

    训练不稳定性（如损失突增）常常是随机梯度噪声的结果。由于语言训练数据中存在稀有表达，加上多层网络的复合作用，噪声影响呈重尾分布，并且在小批量平均后依然存在。现有的补救措施在结构与代价之间进行权衡：向量范数裁剪忽略了权重更新的矩阵结构，而谱归一化（例如 Muon (Jordan et al., 2024)）虽然尊重这种结构，却需要额外代价。我们证明这种权衡是可以平衡的。真实的梯度噪声似乎类似于逐元素的重尾污染，一阶扰动分析揭示了此类噪声的局域化特性，在此特性下，一种简单的逐元素方法即可实现谱控制。利用这一点，我们推导出在高斯信号先验下贝叶斯最优逐元素估计器的一个可求解的替代形式，并在柯西污染假设下建立了 $O(\varepsilon^{-4})$ 的收敛保证。

    arXiv:2605.27733v2 Announce Type: replace  Abstract: Training instabilities such as loss spikes are frequently the result of stochastic gradient noise. Because of rare expressions in language training data, and multiple layer composition, the noise impact is heavy-tailed and survives mini-batch averaging. Existing remedies trade off structure against cost: vector-norm clipping ignores the matrix structure of weight updates, while spectral normalization (e.g., Muon (Jordan et al., 2024)) respects it at additional cost. We show that this trade-off can be balanced. Real gradient noise appears to be similar to entry-wise heavy-tailed contamination, and a first-order perturbation analysis reveals a localization property of such noise, under which a simple entry-wise method achieves spectral control. Exploiting this, we derive a tractable surrogate for the Bayes-optimal entry-wise estimator under a Gaussian signal prior. We establish $O(\varepsilon^{-4})$ convergence guarantee under Cauchy-c
    
[^269]: IGADA-IoT：基于自动数据增强的无线传感器网络中物联网传感器能耗优化

    IGADA-IoT: IoT Sensor Energy Optimization in Wireless Sensor Networks Driven by Automatic Data Augmentation

    [https://arxiv.org/abs/2605.27397](https://arxiv.org/abs/2605.27397)

    提出了一种信息差距引导的物联网传感器自动数据增强框架IGADA-IoT，通过分层多生成器协作与调度机制缩小动态信息差距，实现无线传感器网络中物联网传感器的能耗优化。

    

    在无线传感器网络（WSN）中，数据增强是一种提升采样频率决策性能的新方法，从而实现物联网（IoT）传感器的能耗优化。然而，现有方法依赖单一生成器和经验确定的数量，未能建立动态信息差距与多个生成器之间的映射，并忽视了生成样本的异质性。此外，目前还缺乏一种能够同时考虑信息差距和模型性能的评估与闭环方法。为解决这些问题，我们提出了一种信息差距引导的物联网传感器自动数据增强框架（IGADA-IoT），该框架在多个轮次中采用分层多生成器协作与调度机制。通过联合利用不同生成器的能力来缩小信息差距。在IGADA-IoT中，分层多生成器协作与调度策略……

    arXiv:2605.27397v2 Announce Type: replace  Abstract: In wireless sensor networks (WSNs), data augmentation is a novel method to improve sampling-frequency decision performance, thereby enabling energy optimization for IoT (Internet of Things) sensors. However, existing methods rely on a single generator and empirically determined quantities, failing to establish a mapping between dynamic information gaps and multiple generators, and overlooking the heterogeneity of generated samples. Moreover, an evaluation and a closed-loop method that jointly consider the information gap and the model performance are lacking. To address these issues, we propose an information gap-guided IoT sensor automatic data augmentation framework (IGADA-IoT) with hierarchical multi-generator collaboration and scheduling over multiple rounds. Capabilities of different generators are jointly utilized to reduce the information gaps. In the IGADA-IoT, a hierarchical multi-generator collaboration and scheduling strat
    
[^270]: 大语言模型何时应该进行推理？基于熵动态的自适应推理路由

    When Should LLMs Reason? Adaptive Inference Routing via Entropy Dynamics

    [https://arxiv.org/abs/2605.22873](https://arxiv.org/abs/2605.22873)

    提出无需训练的轻量级路由框架EDRM，通过解读解码早期的熵轨迹变化趋势，自适应地在直接、标准和思维链三种解码模式间选择，在15个基准和7个大语言模型上有效平衡准确率与token消耗。

    

    思维链推理可以提升大语言模型的性能，但其收益因任务、模型和具体查询而异。在某些任务上，CoT带来的准确率提升微乎其微甚至为负，同时却显著增加了token消耗。我们研究了早期解码动态是否有助于判断显式推理何时有用。我们的分析表明，CoT收益较大的任务往往在解码早期呈现熵值递减的趋势，而收益较低的任务则更多表现出振荡或上升的轨迹。这些模式促使我们对解码行为进行经验性状态解读。基于此，我们提出了EDRM（基于熵动态的推理流形），一个轻量级的路由框架，其默认启发式方法无需训练路由器。EDRM利用紧凑且可解释的表示来概括早期熵轨迹，并在直接解码、标准解码和CoT解码三种模式之间进行选择。在15个基准测试和7个大语言模型上的实验显示，其全局性能……

    arXiv:2605.22873v3 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) reasoning can improve LLM performance, but its benefits vary across tasks, models, and individual queries. On some tasks, CoT provides marginal or negative accuracy gains while substantially increasing token consumption. We investigate whether early decoding dynamics can help identify when explicit reasoning is useful. Our analysis shows that tasks with larger CoT gains tend to exhibit decreasing early-stage entropy, while low-gain tasks more often show oscillatory or increasing trajectories. These patterns motivate an empirical regime interpretation of decoding behavior. We propose \textbf{EDRM} (Entropy Dynamics-based Reasoning Manifold), a lightweight routing framework whose default heuristic requires no router training. EDRM summarizes early entropy trajectories with a compact, interpretable representation and selects among Direct, Standard, and CoT decoding. Across 15 benchmarks and 7 LLMs, its globa
    
[^271]: 评估稀疏地震计数的神经概率预报：哪些结论能经受严格评估的检验

    Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts: What Survives a Careful Evaluation

    [https://arxiv.org/abs/2605.21437](https://arxiv.org/abs/2605.21437)

    对天山-帕米尔地区周地震计数神经概率预报的严格评估表明：泊松类预报（含ETAS）严重低估单周发生3次以上地震的概率（低估8.3倍），引入一个离散参数即可解决；而神经网络相对线性模型的优势在为每个网格加入截距后便不再显著。

    

    我们重新审视了一个用于天山与帕米尔地区3×3度网格上每周地震计数的神经负二项模型，并探究其哪些结论能够经受住严格评估的检验。在完整性震级为4.5的条件下，144周的测试期共包含2448个网格-周单元；其中六个网格-周（分布在五个不同的周内）发生了三次及以上的地震事件。有两个结论得以保留。首先，泊松类预报（包括时间性ETAS模型）显著低估了出现这种“繁忙周”的概率：在实际发生六次事件的情况下，ETAS模型的预期仅为0.72次，低估达8.3倍（p = 0.00009，精确检验）。仅需引入一个离散（过散布）参数即可消除这一偏差（带离散参数的ETAS预期为4.45次，p = 0.58）；而在周内模拟余震链虽能再现该尾部特征，却会使总计数超出约四分之一。其次，一旦线性模型为每个网格单元设置了各自的截距项，神经网络相对于线性模型的优势便不复存在：剩余增益仅为偏差的3.4%，不具统计显著性（p = 0.57）。

    arXiv:2605.21437v2 Announce Type: replace-cross  Abstract: We revisit a neural negative binomial model for weekly earthquake counts on a 3 x 3 degree grid over the Tian Shan and Pamir, and ask which of its conclusions survive a careful evaluation. At the completeness magnitude of 4.5, the 144-week test period holds 2448 cell-weeks; six cell-weeks, in five distinct weeks, have three or more events.   Two conclusions survive. First, Poisson forecasts, temporal ETAS included, understate the probability of such a busy week: ETAS expects 0.72 where six occurred, a shortfall by a factor of 8.3 (p = 0.00009, exact test). One dispersion parameter removes it (ETAS with dispersion expects 4.45, p = 0.58); simulating aftershock chains within the week also reproduces the tail but overshoots the total count by about a quarter. Second, the network's advantage over a linear model disappears once that model has an intercept per cell: the remaining gain is 3.4% of deviance, not significant (p = 0.57 ov
    
[^272]: CoarseSoundNet：构建用于生态声景分析的可靠模型

    CoarseSoundNet: Building a reliable model for ecological soundscape analysis

    [https://arxiv.org/abs/2605.21143](https://arxiv.org/abs/2605.21143)

    提出了CoarseSoundNet深度学习模型，能在真实含噪的陆生被动声学监测条件下区分生物声、地声与人为声，为生态声景的粗粒度分类提供了可复现的框架。

    

    声景由三类声音组成：生物声（动物发出的声音）、地声（自然非生物声音）和人为声（人类产生的声音）。声景生态学领域的一个关键研究问题是这些成分之间如何相互作用，特别是生物声如何响应地声和人为声。然而，迄今为止，能够对这些成分进行区分量化的分析工具并不多。近来的机器学习方法旨在支持自动化分析，但往往依赖于特定任务或干净的数据，这限制了其对含噪被动声学监测（PAM）录音的泛化能力。本研究提出了一个可复现的框架来构建用于粗粒度声景分类的机器学习模型，并介绍了CoarseSoundNet——一个经过训练以在真实陆生被动声学监测条件下区分生物声、地声和人为声的深度学习模型。我们系统地……

    arXiv:2605.21143v3 Announce Type: replace-cross  Abstract: A soundscape is composed of three types of sound: biophony (sounds made by animals), geophony (natural abiotic sounds) and anthropophony (sounds made by humans). A key research question in the field of soundscape ecology is how these components interact with each other, specifically how biophony responds to geophony and anthropophony. Nevertheless, as of today, there are not many analytical instruments that enable the distinct quantification of these elements. Recent machine learning (ML) approaches aim to support automated analysis but often rely on task-specific or clean data, limiting generalisation to noisy passive acoustic monitoring (PAM) recordings. This study presents a reproducible framework to build ML models for coarse soundscape classification and introduces CoarseSoundNet, a deep learning model trained to distinguish biophony, geophony, and anthropophony under realistic terrestrial PAM conditions. We systematically
    
[^273]: 投影潜在强化学习动作：迈向可泛化且可扩展的图组合优化

    Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization

    [https://arxiv.org/abs/2605.19721](https://arxiv.org/abs/2605.19721)

    提出基于Wolpertinger框架的“投影智能体”方法，通过在共享的GNN潜在空间中单次前向传播预测潜在动作并解码为有效离散动作，同时提升了图组合优化求解器跨实例的泛化能力与面对大规模动作空间的计算可扩展性。

    

    图组合优化（GCO）引起了越来越多的关注，因为许多NP难问题天然可以用图的形式表述，但其组合爆炸使得精确求解方法在计算上难以处理。强化学习（RL）与图神经网络（GNN）相结合的最新进展显著提升了基于学习的GCO求解器的性能。然而，现有方法在两个方面存在局限：一是跨多样化图实例的泛化能力，二是随动作空间增长的计算可扩展性。为了同时应对这两个挑战，我们提出了*投影智能体*（projection agents），这是一种建立在Wolpertinger框架之上、并适配于共享的基于GNN的观测与动作潜在空间的GCO方法。这些智能体通过在单次前向传播中预测期望的潜在动作来学习求解GCO问题，随后利用该潜在空间将其解码为图中有效的离散动作。在各种基（准测试上……）

    arXiv:2605.19721v2 Announce Type: replace-cross  Abstract: Graph combinatorial optimization (GCO) has attracted growing interest, as many NP-hard problems naturally admit graph formulations, yet their combinatorial explosion renders exact methods computationally intractable. Recent advances in Reinforcement Learning (RL) combined with Graph Neural Networks (GNNs) have significantly improved learning-based GCO solvers. However, existing approaches face limitations in both generalization across diverse graph instances and computational scalability as action spaces grow. To address both challenges, we introduce \textit{projection agents}, a GCO methodology built on the Wolpertinger framework and adapted to a shared GNN-based latent space for observations and actions. These agents are trained to solve GCO problems by predicting a desired latent action in a single forward pass, which is subsequently decoded into a valid discrete action in the graph using the latent space. Across diverse ben
    
[^274]: 通过流匹配学习无偏置换

    Learning Unbiased Permutations via Flow Matching

    [https://arxiv.org/abs/2605.16755](https://arxiv.org/abs/2605.16755)

    PermFlow是一种条件流匹配框架，通过闭式切空间投影直接在双随机矩阵的仿射子空间上运行，能够捕捉多峰置换分布，在无歧义输入上实现高准确率，并在歧义情形下恢复出多个有效置换。

    

    学习置换对于排序、排名和匹配任务至关重要，但现有的基于熵正则化Sinkhorn的可微方法只能产生单一的软化解，在存在歧义时会失效。我们提出了PermFlow，一个直接作用于行和与列和均为1的矩阵仿射子空间上的条件流匹配框架。闭式切空间投影器通过构造方式（而非迭代修正）在每条轨迹上严格保持这些约束，而最近目标耦合将不同的噪声初始化引导至不同的有效置换。由此得到的模型能够捕捉多峰置换分布，而不是将其坍缩为单一模式。在一系列问题上，PermFlow在无歧义输入上达到了高准确率，并且在存在歧义时能够恢复出多个有效的置换。

    arXiv:2605.16755v2 Announce Type: replace  Abstract: Learning permutations is fundamental to sorting, ranking, and matching, but existing differentiable methods based on entropy-regularized Sinkhorn produce a single softened solution and collapse under ambiguity. We present PermFlow, a conditional flow matching framework that operates directly on the affine subspace of matrices with unit row and column sums. A closed-form tangent-space projector preserves these constraints exactly along every trajectory, by construction rather than through iterative correction, and a nearest-target coupling routes distinct noisy initializations toward distinct valid permutations. The result is a model that captures multimodal permutation distributions rather than collapsing them to a single mode. On a range of problems, PermFlow achieves high accuracy on unambiguous inputs and recovers both valid permutations under ambiguity.
    
[^275]: 通过熵最大化的流形采样

    Manifold Sampling via Entropy Maximization

    [https://arxiv.org/abs/2605.12338](https://arxiv.org/abs/2605.12338)

    本文提出基于熵最大化的流形采样方法 MASEM，能够在光滑约束隐式定义、且被分割成未知数量不连通分量的可行域上实现均匀采样，并证明其在平均场意义下以指数速率收敛到最大熵目标分布。

    

    从受约束分布中采样具有广泛的应用，包括贝叶斯优化和机器人学等领域。先前的工作为受约束采样建立了收敛性和可行性保证，但假设可行集是连通的。然而，在实践中，可行集往往分解为多个不连通的分量，这使得在约束条件下进行高效采样变得具有挑战性。本文提出了通过熵最大化的流形采样方法（MASEM），用于在由光滑等式和不等式约束隐式定义、且不连通分量数量未知的流形上进行均匀采样。该方法采用重采样方案，基于 k 近邻密度估计来最大化经验分布的熵。我们证明，在平均场意义下，MASEM 使经验分布与最大熵目标分布之间的 KL 散度呈指数级下降……

    arXiv:2605.12338v2 Announce Type: replace  Abstract: Sampling from constrained distributions has a wide range of applications, including in Bayesian optimization and robotics. Prior work establishes convergence and feasibility guarantees for constrained sampling, but assumes that the feasible set is connected. However, in practice, the feasible set often decomposes into multiple disconnected components, which makes efficient sampling under constraints challenging. In this paper, we propose MAnifold Sampling via Entropy Maximization (MASEM) for uniform sampling on a manifold with an unknown number of disconnected components, implicitly defined by smooth equality and inequality constraints. The presented method uses a resampling scheme to maximize the entropy of the empirical distribution based on k-nearest neighbor density estimation. We show that, in the mean field, MASEM decreases the KL-divergence between the empirical distribution and the maximum-entropy target exponentially in the 
    
[^276]: 面向概念分离扩散模型遗忘的解耦稀疏表示

    Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning

    [https://arxiv.org/abs/2605.12122](https://arxiv.org/abs/2605.12122)

    提出SAEParate方法，通过概念感知对比学习与交叉注意力引导的空间聚合，在稀疏自编码器潜在空间中显式解耦语义重叠的概念，从而在不修改模型参数的情况下实现扩散模型的精确概念遗忘。

    

    在文本到图像扩散模型中遗忘特定概念对于防止生成不良内容正变得日益重要。在先前的方法中，基于稀疏自编码器（SAE）的方法因能够通过轻量级地操纵潜在特征来抑制目标概念、且无需修改模型参数而备受关注。然而，主要以稀疏重建为目标训练的SAE并未显式地强制实现概念间的分离，使得语义相关的概念可能共享潜在特征。为解决这一问题，我们提出了SAEParate，通过概念感知的对比学习在SAE潜在空间中显式分离语义重叠的概念，并由交叉注意力引导的空间聚合加以支持，该聚合机制强调与概念相关的区域。此外，我们通过非线性变换增强编码器，以提升其在该分离目标下的表达能力。

    arXiv:2605.12122v2 Announce Type: replace  Abstract: Unlearning specific concepts in text-to-image diffusion models has become increasingly important for preventing undesirable content generation. Among prior approaches, sparse autoencoder (SAE)-based methods have attracted attention due to their ability to suppress target concepts through lightweight manipulation of latent features, without modifying model parameters. However, SAEs trained primarily for sparse reconstruction do not explicitly enforce concept-wise separation, allowing semantically related concepts to share latent features. To address this, we propose SAEParate, which explicitly separates semantically overlapping concepts in the SAE latent space through concept-aware contrastive learning, supported by cross-attention-guided spatial aggregation that emphasizes concept-relevant regions. In addition, we enhance the encoder with a nonlinear transformation to increase its expressive capacity under this separation objective, 
    
[^277]: 面向扩散模型微调的高效伴随匹配

    Efficient Adjoint Matching for Fine-tuning Diffusion Models

    [https://arxiv.org/abs/2605.11480](https://arxiv.org/abs/2605.11480)

    提出高效伴随匹配（EAM），通过将随机最优控制问题重构为线性基础漂移形式，避免了完整轨迹的随机模拟和伴随状态的反向ODE计算，大幅降低了扩散模型奖励微调的计算成本。

    

    奖励微调已成为在文本到图像生成中将预训练扩散模型和流模型与人类偏好对齐的常用方法。在基于奖励梯度的方法中，伴随匹配通过将奖励微调转化为随机最优控制（SOC）问题，提供了一种有原则的建模框架。然而，AM不可避免地带来巨大的计算开销：它需要（i）在无记忆动力学下对完整的生成轨迹进行随机模拟，导致大量的函数评估次数；（ii）沿每条采样轨迹对伴随状态进行反向ODE模拟。在本工作中，我们观察到这两个瓶颈都与从预训练模型继承的非平凡基础漂移密切相关。受此观察启发，我们提出高效伴随匹配，通过使用线性基础漂移对SOC问题进行重构，显著提升了训练效率。

    arXiv:2605.11480v3 Announce Type: replace  Abstract: Reward fine-tuning has become a common approach for aligning pretrained diffusion and flow models with human preferences in text-to-image generation. Among reward-gradient-based methods, Adjoint Matching (AM) provides a principled formulation by casting reward fine-tuning as a stochastic optimal control (SOC) problem. However, AM inevitably requires a substantial computational cost: it requires (i) stochastic simulation of full generative trajectories under memoryless dynamics, resulting in a large number of function evaluations, and (ii) backward ODE simulation of the adjoint state along each sampled trajectory. In this work, we observe that both bottlenecks are closely tied to the non-trivial base drift inherited from the pretrained model. Motivated by this observation, we propose Efficient Adjoint Matching (EAM), which substantially improves training efficiency by reformulating the SOC problem with a linear base drift and a corres
    
[^278]: 面向机器学习原子间势的组合泛化基准测试

    Benchmarking Compositional Generalisation for Machine Learning Interatomic Potentials

    [https://arxiv.org/abs/2605.08988](https://arxiv.org/abs/2605.08988)

    该论文提出了一个由四个任务组成的基准测试，用以评估机器学习原子间势是真正学习到化学的组合结构并能泛化到未见分子，还是仅仅在训练数据上进行插值。

    

    机器学习原子间势在计算化学和材料科学中发挥着基础性作用，支持从分子动力学模拟到药物设计和材料发现等广泛应用。尽管近期的方法能够以高精度估计原子间作用力，但它们在多大程度上能够泛化到此前未见过的分子仍不明确。它们究竟是学习到了化学的组合结构，即捕捉分子片段及其组合如何决定性质，还是主要学习对训练样本特有模式进行插值？为了回答这一问题，我们提出了一个由四个任务组成的基准测试，这些任务都需要某种形式的组合泛化能力。在每个任务中，模型都在训练期间未见过的分子上进行测试，但训练数据的选取方式使得对于学习了潜在化学组合结构的模型而言，对测试样本进行泛化应当是可行的。

    arXiv:2605.08988v2 Announce Type: replace  Abstract: Machine Learning Interatomic Potentials play a fundamental role in computational chemistry and materials science, enabling applications from molecular dynamics simulations to drug design and materials discovery. While recent approaches can estimate inter-atomic forces with high precision, it remains unclear to what extent they can generalise to previously unseen molecules. Do they learn the compositional structure of chemistry, capturing how molecular fragments and their combinations determine properties, or do they primarily learn to interpolate patterns that are specific to the training examples? To address this question, we propose a benchmark consisting of four tasks that require some form of compositional generalisation. In each task, models are tested on molecules that were unseen during training, but the training data is chosen such that generalisation to the test examples should be feasible for models that learn the underlyin
    
[^279]: 一个算法，两个目标：大语言模型微调中参数与数据选择的双重评分

    One Algorithm, Two Goals: Dual Scoring for Parameter and Data Selection in LLM Fine-Tuning

    [https://arxiv.org/abs/2605.06166](https://arxiv.org/abs/2605.06166)

    该论文提出DualSFT算法，将LLM微调中的参数选择与数据选择统一为同一验证目标下的双层选择问题，通过对单一梯度交互矩阵进行行列聚合，一次性同时完成参数掩码和数据子集的双重评分与选择。

    

    在大语言模型（LLM）微调中，参数选择和数据选择是降低微调成本的常用策略，但它们通常由各自独立的评分机制驱动。当参数掩码和数据子集共同决定受限微调时，这种分离会带来冗余开销，并使协同选择变得困难。我们将参数选择和数据选择表述为在共同验证目标下的两个双层选择问题，并推导出一个共享的局部响应替代评分规则。在一阶和二阶验证改进近似下，参数重要性和数据效用表现为同一个梯度交互矩阵的列向聚合与行向聚合，从而得到用于协同提取这两种信号的闭式行列对应关系。基于这一结构，我们提出了DualSFT（双重选择微调），这是一种一次性双重评分算法，可生成参数掩码和数据……（摘要在此处截断）

    arXiv:2605.06166v2 Announce Type: replace  Abstract: In Large Language Model (LLM) fine-tuning, parameter and data selection are common strategies for reducing fine-tuning cost, yet they are typically driven by separate scoring mechanisms. When a parameter mask and data subset jointly determine restricted fine-tuning, this separation incurs redundant overhead and makes coordinated selection difficult. We cast parameter and data selection as two bilevel selection problems under a common validation objective and derive a shared local response-surrogate scoring rule. Under first- and second-order validation-improvement approximations, parameter importance and data utility emerge as column-wise and row-wise aggregations of a single gradient interaction matrix, yielding a closed-form row-column correspondence for co-extracting both signals. Building on this structure, we propose DualSFT (Dual-Selection Fine-Tuning), a one-shot dual-scoring algorithm that produces a parameter mask and data s
    
[^280]: 通过控制最大化统一目标条件强化学习与无监督技能学习

    Unifying Goal-Conditioned RL and Unsupervised Skill Learning via Control-Maximization

    [https://arxiv.org/abs/2605.06145](https://arxiv.org/abs/2605.06145)

    该论文提出将目标条件强化学习（GCRL）与互信息技能学习（MISL）统一在控制最大化框架之下，并证明三种典型的GCRL公式在根本上是不可等价的，为无监督技能学习为何能支持下游目标达成任务提供了理论基础。

    

    无监督预训练已推动目标条件强化学习（GCRL）取得了经验上的进展，但其理论基础仍缺乏深入理解。特别是，一类有影响力的方法——互信息技能学习（MISL），能够发现行为多样的技能，这些技能随后可用于下游的目标达成任务。然而，为什么通过MISL学到的技能能够支持目标达成，这在理论上仍是一个谜。一个微妙的挑战在于，GCRL和MISL都是伞形术语：不同的GCRL任务采用不同的标准来衡量目标达成性能，而不同的MISL方法则优化不同的行为多样性概念。我们应对这一挑战，将GCRL和MISL统一为控制最大化的实例。我们识别出三种典型的GCRL公式，并证明它们在根本上是不可等价的：即使在相同环境中，它们也可能导出互不相容的最优策略。（摘要原文在此处截断）

    arXiv:2605.06145v2 Announce Type: replace  Abstract: Unsupervised pretraining has driven empirical advances in goal-conditioned reinforcement learning (GCRL), but its theoretical foundations remain poorly understood. In particular, an influential class of methods, mutual information skill learning (MISL), discovers behaviorally diverse skills that can later be used for downstream goal-reaching. However, it remains a theoretical mystery why skills learned through MISL should support goal-reaching. A subtle challenge is that both GCRL and MISL are umbrella terms: Different GCRL tasks use distinct criteria for measuring goal-reaching performance, while different MISL methods optimize distinct notions of behavioral diversity. We address this challenge and unify GCRL and MISL as instances of control maximization. We identify three canonical GCRL formulations and prove that they are fundamentally inequivalent: They can induce incompatible optimal policies even in the same environment. Nevert
    
[^281]: 图记忆Transformer (GMT)

    Graph Memory Transformer (GMT)

    [https://arxiv.org/abs/2604.23862](https://arxiv.org/abs/2604.23862)

    本文提出图记忆Transformer（GMT），用可学习的显式记忆图（128个质心加有向转移矩阵）替代仅解码器Transformer中的FFN子层，在因果自注意力不变的情况下，以更少的参数量（82.2M对103.0M）实现稳定训练。

    

    我们研究了一种显式的可学习记忆图能否替代仅解码器（decoder-only）Transformer中的前馈网络（FFN）子层，同时保留周围的自回归架构。在所提出的图记忆Transformer（GMT）中，记忆单元通过一个由学习得到的有向转移矩阵连接的可学习质心库来路由token表示，而因果自注意力机制保持不变。在基础v7实例中，16个Transformer块中的每一块包含128个质心和一个128×128的边矩阵。每个记忆单元采用引力源路由（gravitational source routing）和基于token条件的目标选择，返回一个从估计的源记忆状态指向目标记忆状态的门控位移（gated displacement），而非检索到的值。由此得到的仅解码器语言模型拥有8220万可训练参数，且不含密集的FFN子层；而用于对比的密集GPT风格基线模型拥有1.03亿参数。该模型能够稳定训练，并展现出质心u……（原文摘要至此截断）

    arXiv:2604.23862v4 Announce Type: replace-cross  Abstract: We investigate whether an explicit learned memory graph can replace the Feed-Forward Network (FFN) sublayer in a decoder-only transformer while preserving the surrounding autoregressive architecture. In the proposed graph Memory Transformer (GMT), a memory cell routes token representations over a learned bank of centroids connected by a learned directed transition matrix, while causal self-attention remains unchanged. In the base v7 instantiation, each of 16 transformer blocks contains 128 centroids and a 128 x 128 edge matrix. Each cell uses gravitational source routing and token-conditioned target selection to return a gated displacement from the estimated source memory state toward the target memory state, rather than a retrieved value. The resulting decoder-only language model has 82.2M trainable parameters and no dense FFN sublayers; the evaluated dense GPT-style baseline has 103.0M. It trains stably and exposes centroid u
    
[^282]: AtManRL：通过可微注意力显著性迈向忠实推理

    AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency

    [https://arxiv.org/abs/2604.16158](https://arxiv.org/abs/2604.16158)

    提出AtManRL方法，通过优化可微注意力掩码推导显著性奖励并与结果奖励结合于GRPO强化学习框架中，使思维链推理真正因果性地促成最终答案，从而提升推理的忠实性。

    

    大型语言模型（LLM）日益依赖思维链推理来解决复杂任务，然而推理轨迹可能只是伴随最终答案出现，而非因果性地促成答案，或忠实地反映答案的得出过程。我们引入可微注意力操作，通过强化学习训练显著性推理，并将该方法命名为 AtManRL。通过优化一个加性注意力掩码来识别恢复模型答案所需的思维链标记，我们推导出显著性奖励，并将其与结果奖励相结合，应用于具有解耦优势的 GRPO 中。由于显著性和忠实性都强烈依赖于轨迹长度，我们在评估时针对一系列长度惩罚基线进行扫描，以追踪纯长度惩罚所能达到的权衡范围，而非仅与单一基线比较。在三个模型和四个基准测试上的实验表明了该方法的有效性。

    arXiv:2604.16158v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex tasks, yet a reasoning trace may merely accompany the final answer rather than causally contribute to it or faithfully reflect how it was reached. We introduce differentiable attention manipulation to train salient reasoning with reinforcement learning -- method which we call AtManRL. By optimizing an additive attention mask that identifies the CoT tokens needed to recover the model's answer, we derive a saliency reward and combine it with an outcome reward in GRPO with decoupled advantages. Because both saliency and faithfulness depend strongly on trace length, we evaluate against a sweep of length-penalty baselines that traces the trade-off a pure length penalty can reach, rather than against a single baseline. Across three models (Qwen2.5-3B, Granite-3.3-2B, and Phi-4-mini) and four benchmarks (GSM8K, MMLU, ProofWriter, and QuestBen
    
[^283]: 超级智能体与混淆因素：周围智能体对车辆轨迹预测的影响

    Super Agents and Confounders: Influence of surrounding agents on vehicle trajectory prediction

    [https://arxiv.org/abs/2604.03463](https://arxiv.org/abs/2604.03463)

    该论文发现周围智能体信息反而会降低轨迹预测的精度，并提出无需额外监督的条件信息瓶颈（CIB）方法，通过压缩有效特征、过滤无益信息来提升预测性能的稳定性与因果性。

    

    在高度交互的驾驶场景中，轨迹预测依赖于周围交通参与者（如汽车和行人）的信息。我们的主要贡献是对最先进的轨迹预测器进行了全面分析，揭示了一个令人惊讶且关键的缺陷：许多周围智能体会降低而非提高预测精度。借助基于Shapley值的归因方法，我们严格证明了模型学习到的是不稳定的、非因果的决策方案，且这些方案在不同的训练运行之间存在显著差异。基于这些发现，我们提出引入条件信息瓶颈（Conditional Information Bottleneck, CIB），该方法无需额外的监督信息，通过训练有效压缩智能体特征，并忽略那些对预测任务无益的特征。在多个数据集和模型架构上进行的全面实验表明，这种简单而有效的方法不仅能……

    arXiv:2604.03463v2 Announce Type: replace  Abstract: In highly interactive driving scenes, trajectory prediction is conditioned on information from surrounding traffic participants such as cars and pedestrians. Our main contribution is a comprehensive analysis of state-of-the-art trajectory predictors, which reveals a surprising and critical flaw: many surrounding agents degrade prediction accuracy rather than improve it. Using Shapley-based attribution, we rigorously demonstrate that models learn unstable and non-causal decision-making schemes that vary significantly across training runs. Building on these insights, we propose to integrate a Conditional Information Bottleneck (CIB), which does not require additional supervision and is trained to effectively compress agent features as well as ignore those that are not beneficial for the prediction task. Comprehensive experiments using multiple datasets and model architectures demonstrate that this simple yet effective approach not only
    
[^284]: 离散数据的无约束单纯形去噪：一种应用于图生成的非马尔可夫方法

    Unrestrained Simplex Denoising for Discrete Data. A Non-Markovian Approach Applied to Graph Generation

    [https://arxiv.org/abs/2603.28572](https://arxiv.org/abs/2603.28572)

    提出了一种在概率单纯形上操作的非马尔可夫生成框架“无约束单纯形去噪”，通过使不同时间的噪声表示条件独立来移除不必要的约束，在图生成任务上超越了离散扩散和流匹配基线。

    

    扩散模型或流匹配等去噪模型近来推动了离散结构生成建模的发展，然而大多数方法直接在离散状态空间中操作，导致状态发生突变。我们提出了单纯形去噪，这是一种在概率单纯形上操作的简单而有效的生成框架。其核心思想是一种非马尔可夫的加噪方案，即对于给定的干净数据点，不同时间的噪声表示是条件独立的。在保留基于去噪的生成模型理论保证的同时，我们的方法移除了不必要的约束，从而提升了性能并简化了公式化。实验表明，无约束单纯形去噪在合成和真实世界的图基准测试中超越了强大的离散扩散和流匹配基线。这些结果凸显了概率单纯形作为离散生成有效框架的价值。

    arXiv:2603.28572v2 Announce Type: replace  Abstract: Denoising models such as Diffusion or Flow Matching have recently advanced generative modeling for discrete structures, yet most approaches operate directly in the discrete state space, leading to abrupt state changes. We introduce simplex denoising, a simple yet effective generative framework that operates on the probability simplex. The key idea is a non-Markovian noising scheme in which, for a given clean data point, noisy representations at different times are conditionally independent. While preserving the theoretical guarantees of denoising-based generative models, our method removes unnecessary constraints, thereby improving performance and simplifying the formulation. Empirically, \emph{unrestrained simplex denoising} surpasses strong discrete diffusion and flow-matching baselines across synthetic and real-world graph benchmarks. These results highlight the probability simplex as an effective framework for discrete generative
    
[^285]: 让我看看你不知道什么：面向模型验证的不变集高效采样方法

    Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation

    [https://arxiv.org/abs/2603.21782](https://arxiv.org/abs/2603.21782)

    该论文提出免训练的“不变性审计”诊断工具，利用预训练的扩散或流匹配模型，通过纤维损失引导去噪过程，高效采样出与查询输入特征相同但形式多样的输入，从而检验模型学习到的不变性是否符合任务语义。

    

    机器学习模型的性能取决于发现数据中与解决给定任务相关的特征，同时忽略无关的变化。为了检查和可视化模型是否做到了这一点，我们提出了“不变性审计”，这是一种用于分析特征提取器的诊断工具。不变性审计会生成多样化的输入，这些输入都与给定的查询输入产生相同的特征，因此在模型看来它们是相同的。借助关于目标任务的外部知识，这使我们能够评估模型学习到的不变性是否符合任务所需的语义。与现有工作中需要为每个特征提取器专门训练一个生成模型不同，我们的算法无需训练，而是利用预训练的扩散模型或流匹配模型来采样不变输入。我们的纤维损失通过惩罚特征不匹配来引导去噪过程，使输出与查询输入共享相同的表示……（原文在此处截断）

    arXiv:2603.21782v2 Announce Type: replace  Abstract: The performance of machine learning models depends on discovering features in the data that are relevant to solving the given task while ignoring irrelevant variation. To check and visualize whether this is the case, we propose 'invariance auditing', a diagnostic tool to analyze feature extractors. Invariance auditing generates diverse inputs which all yield the same features as a given query input, and are therefore treated as identical by the model. Using external knowledge about the task at hand, this enables assessing if the learned invariances do or do not conform to the task's required semantics. Unlike existing work where a dedicated generative model is trained for each feature extractor, our algorithm is training-free and exploits a pretrained diffusion or flow-matching model to sample invariant inputs. Our fiber loss -- which penalizes feature mismatch -- guides the denoising process so the output shares the same representat
    
[^286]: Brittlebench：通过提示敏感性量化大语言模型的鲁棒性

    Brittlebench: Quantifying LLM robustness via prompt sensitivity

    [https://arxiv.org/abs/2603.13285](https://arxiv.org/abs/2603.13285)

    该论文提出了一个量化大语言模型对提示变体敏感性的理论框架，并设计了名为Brittlebench的评估流程，通过对主流基准测试施加保持语义的扰动，揭示前沿模型性能最高下降12%且不同模型的脆弱程度各异。

    

    arXiv:2603.13285v4 公告类型：替换。摘要：现有的评估方法主要依赖于干净、静态的基准测试，这会因未能捕捉真实世界用户输入中固有的噪声与变异性而高估模型的实际性能。对于语言模型而言尤其如此，因为语言模型面对的人类生成的文本查询可能包含错误、拼写失误或对同一问题的不同表述方式。在这项工作中，我们引入了一个理论框架，用于量化模型对提示变体的敏感性（即脆弱性），该框架使我们能够将数据导致的难度与提示相关的变异性分离开来。利用这一框架，我们设计了一个新颖的评估流程 Brittlebench，以全面评估前沿模型的敏感性。我们对一系列流行的基准测试施加了保持语义的扰动，并观察到模型性能下降幅度高达12%。然而，这些扰动对不同模型的影响并不均等：即使单个扰动也会改变……

    arXiv:2603.13285v4 Announce Type: replace  Abstract: Existing evaluation methods largely rely on clean, static benchmarks, which can overestimate true model performance by failing to capture the noise and variability inherent in real-world user inputs. This is especially true for language models, which can face human-generated text queries containing mistakes, typos, or alternative ways of phrasing the same question. In this work, we introduce a theoretical framework for quantifying model sensitivity to prompt variants, or brittleness, that can enable us to disentangle data-induced difficulty from prompt-related variability. Using this framework, we design a novel evaluation pipeline, Brittlebench, to holistically evaluate the sensitivity of frontier models. We apply semantics-preserving perturbations to a suite of popular benchmarks, and observe model performance to degrade as much as 12%. However, these perturbations do not affect all models equally: even a single perturbation alters
    
[^287]: 大语言模型中的马尔可夫生成链

    Markovian Generation Chains in Large Language Models

    [https://arxiv.org/abs/2603.11228](https://arxiv.org/abs/2603.11228)

    本文将大语言模型反复处理文本的迭代推理过程定义为马尔可夫生成链，发现输出要么收敛到小的循环集合要么持续产生新句子，且多样性的增减取决于温度参数和初始输入句子等因素。

    

    大语言模型（LLM）的广泛使用引发了一个重要问题：当文本被大语言模型反复处理时，文本会如何演化？在本文中，我们将这种迭代推理过程定义为马尔可夫生成链，其中每一步都以特定的提示模板和先前的输出作为输入，而不包含任何先验数据。在迭代改写和往返翻译实验中，输出要么收敛到一个小的循环集合，要么在有限范围内持续产生新句子。通过句子级马尔可夫链建模和对模拟数据的分析，我们表明迭代过程既可能增加也可能降低多样性，这取决于温度参数和初始输入句子等因素。这些结果为迭代式大语言模型推理的动态特性及其对多智能体大语言模型系统的启示提供了有价值的见解。

    arXiv:2603.11228v2 Announce Type: replace  Abstract: The widespread use of large language models (LLMs) raises an important question: how do texts evolve when they are repeatedly processed by LLMs? In this paper, we define this iterative inference process as Markovian generation chains, where each step takes a specific prompt template and the previous output as input, without including any prior data. In iterative rephrasing and round-trip translation experiments, the output either converges to a small recurrent set or continues to produce novel sentences within a finite horizon. Through sentence-level Markov chain modeling and analysis of simulated data, we show that iterative process can either increase or reduce the diversity depending on factors such as the temperature parameter and the initial input sentence. These results offer valuable insights into the dynamics of iterative LLM inference and their implications for multi-agent LLM systems.
    
[^288]: 一种用于Transformer训练的最优控制方法

    An Optimal Control Approach To Transformer Training

    [https://arxiv.org/abs/2603.09571](https://arxiv.org/abs/2603.09571)

    本文将Transformer训练建模为受控McKean-Vlasov动力学的最优控制问题，通过将动力学提升到概率测度空间构造完全可观测的MDP，并证明了全局最优策略的存在性及其与依赖初始分布的开环策略的等价性。

    

    本文提出了一种严格的最优控制理论方法用于Transformer训练，该方法遵循关键的结构性约束，包括：执行过程中与实际输入的独立性、问题的集合控制特性，以及位置依赖性。我们将Transformer架构建模为具有共享动作的离散时间受控粒子系统，呈现出无噪声的受控McKean--Vlasov动力学，并证明将该动力学提升到概率测度空间可产生一个完全可观测的马尔可夫决策过程（MDP）。位置编码被纳入状态空间中，以便在提升过程中保持序列顺序。利用动态规划原理，我们在紧致性的温和假设下建立了全局最优策略的存在性。我们进一步证明，提升问题中的闭环策略等价于依赖于初始分布的开环策略，这些策略……

    arXiv:2603.09571v2 Announce Type: replace  Abstract: In this paper, we develop a rigorous optimal control-theoretic approach to Transformer training that respects key structural constraints such as (i) realized-input-independence during execution, (ii) the ensemble control nature of the problem, and (iii) positional dependence. We model the Transformer architecture as a discrete-time controlled particle system with shared actions, exhibiting noise-free controlled McKean--Vlasov dynamics, and we show that lifting the dynamics to probability measures produces a fully-observed Markov decision process (MDP). Positional encodings are incorporated into the state space to preserve the sequence order under lifting. Using the dynamic programming principle, we establish the existence of globally optimal policies under mild assumptions of compactness. We further prove that closed-loop policies in the lifted problem is equivalent to an initial-distribution dependent open-loop policy, which are rea
    
[^289]: 乐观策略正则化：基于精英统计的奖励塑形

    Optimistic Policy Regularization: Elite-Statistic Reward Shaping

    [https://arxiv.org/abs/2603.06793](https://arxiv.org/abs/2603.06793)

    论文提出乐观策略正则化（OPR），将同策略训练中被丢弃的罕见高回报回合转化为基于精英动作先验对数比率的闭式奖励加成，仅需对 PPO 奖励做一行修改，便在 49 个 Atari 游戏上以少 5 倍的决策次数超越了 PPO 和自我模仿学习等基线。

    

    同策略智能体在每次更新后会丢弃所有轨迹，包括那些罕见的高回报回合。我们提出了乐观策略正则化，将这些回合转化为一种闭式形式的奖励加成：我们从表现最好的四分之一回合中估计一个精英动作先验，并将该先验对数比率的有界变换加到策略的动作分布上。OPR 无需辅助网络、价值门控、离策略校正或对抗循环。它仅是对近端策略优化（PPO）奖励的一行修改；然而，在 49 个 Atari 游戏上以 1000 万步进行测试时——所用智能体决策次数比 2 亿帧基准少 5 倍——它达到了 148.4% 的中位人类归一化得分（HNS；四分位均值达 196.9%）：在 36/49 个游戏上得分高于已发表的 PPO（p=0.0014），且总体中位数高于自我模仿学习——当帧数和骨干网络相匹配时，自我模仿学习的表现远逊于 OPR。我们将该方法与一个理论……（摘要截断）

    arXiv:2603.06793v2 Announce Type: replace  Abstract: On-policy agents discard trajectories after each update, including rare high-return episodes. We propose Optimistic Policy Regularization (OPR), which turns those episodes into a closed-form reward bonus: we estimate an elite action prior from top-quartile episodes and add a \reviewblue{bounded transform of the log-ratio} of the prior to the policy's action distribution. OPR needs no auxiliary network, value gate, off-policy correction, or adversarial loop. It is a one-line modification of Proximal Policy Optimization (PPO)'s reward; yet, on 49 Atari games at 10M steps, $5\times$ fewer agent decisions than the 200M-frame baselines, it reaches a 148.4\% median human-normalized score (HNS; 196.9\% interquartile mean): higher scores than published PPO on $36/49$ games ($p{=}0.0014$) and a higher aggregate median than self-imitation learning, which falls well below OPR when frames and backbone are matched. We pair the method with a theor
    
[^290]: K-均值作为归一化RBF分配的零温度极限：收敛速率、轨迹与基于Entmax的精确恢复

    K-Means as the Zero-Temperature Limit of Normalized RBF Assignments: Rates, Trajectories, and Exact Recovery with Entmax

    [https://arxiv.org/abs/2603.04625](https://arxiv.org/abs/2603.04625)

    本文严格证明了K-均值聚类是归一化RBF分配在零温度下的Γ-极限，从目标函数、优化方法、单次更新和轨迹收敛等多个层面定量刻画了软RBF分配与硬K-均值之间的联系，并利用Entmax实现了精确恢复。

    

    K-均值通常被视为一种独立于梯度学习的离散过程，而归一化径向基函数（RBF）层则是光滑的。我们精确阐明了K-均值作为此类层零温度极限的含义。通过引入香农熵惩罚将K-均值分配松弛到单纯形上，可以得到归一化高斯RBF责任度以及一个简化的目标函数F_σ，其中0≤J−F_σ≤2nσ²log k，因此F_σ Γ-收敛于K-均值的失真J。我们区分了四个层次的论述：在目标函数层面，最小值与最小化子收敛，且在临界温度σ²=λ_max之上，所有位于均值的质心构成一个严格局部极小值点；在优化层面，交替最小化是关于F_σ的预条件梯度方法，其硬性极限即为Lloyd算法；在单次更新层面，在间隔Δ_min>0的条件下，软质心映射逼近（原文此处截断）

    arXiv:2603.04625v2 Announce Type: replace  Abstract: K-Means is usually treated as a discrete procedure outside gradient-based learning, whereas normalized radial basis function (RBF) layers are smooth. We make precise the sense in which K-Means is the zero-temperature limit of such a layer. Relaxing the K-Means assignments to the simplex with a Shannon entropy penalty yields the normalized Gaussian RBF responsibilities and a reduced objective $F_\sigma$ with $0\le J-F_\sigma\le 2n\sigma^2\log k$, so $F_\sigma$ $\Gamma$-converges to the K-Means distortion $J$. We keep four levels of statement apart. For the objective, minimal values and minimizers converge, and above the critical temperature $\sigma^2=\lambda_{\max}$ all centroids at the mean form a strict local minimizer. For optimization, alternating minimization is a preconditioned gradient method on $F_\sigma$ whose hard limit is Lloyd's algorithm. For one update, under a margin $\Delta_{\min}>0$, the soft centroid map approaches L
    
[^291]: KL正则化多臂老虎机的近最优后悔界

    Near-Optimal Regret for KL-Regularized Multi-Armed Bandits

    [https://arxiv.org/abs/2603.02155](https://arxiv.org/abs/2603.02155)

    该论文通过新颖的分层论证对KL-UCB算法进行精细分析，首次给出了对臂数 $K$ 呈线性依赖的高概率后悔上界 $\tilde{O}(\eta K\log^2T)$，并以首个非常数下界 $\Omega(\eta K\log T)$ 证明了其近紧性。

    

    近年来的研究表明，带有KL正则化目标的强化学习能够获得更快的收敛速度或对数级后悔，这与未正则化设定下经典的 $\sqrt{T}$ 型后悔形成鲜明对比。然而，即使是针对多臂老虎机（MAB）这一特殊情形，关于KL正则化目标的在线学习的统计效率仍然远未被完全刻画。我们通过对KL-UCB算法进行精细分析，并采用一种新颖的分层论证方法，解决了多臂老虎机中的这一问题，得到了 $\tilde{O}(\eta K\log^2T)$ 的上界：这是首个对臂数 $K$ 呈线性依赖的高概率后悔界。其中，$T$ 表示时间范围，$K$ 表示臂数，$\eta^{-1}$ 表示正则化强度，$\tilde{O}$ 隐藏了除涉及 $\log T$ 之外的所有对数因子。我们分析的近紧性由首个非常数下界 $\Omega(\eta K \log T)$ 所证实……

    arXiv:2603.02155v2 Announce Type: replace  Abstract: Recent studies have shown that reinforcement learning with KL-regularized objectives can enjoy faster rates of convergence or logarithmic regret, in contrast to the classical $\sqrt{T}$-type regret in the unregularized setting. However, the statistical efficiency of online learning with respect to KL-regularized objectives remains far from completely characterized, even when specialized to multi-armed bandits (MABs). We address this problem for MABs via a sharp analysis of KL-UCB using a novel peeling argument, which yields a $\tilde{O}(\eta K\log^2T)$ upper bound: the first high-probability regret bound with linear dependence on $K$. Here, $T$ is the time horizon, $K$ is the number of arms, $\eta^{-1}$ is the regularization intensity, and $\tilde{O}$ hides all logarithmic factors except those involving $\log T$. The near-tightness of our analysis is certified by the first non-constant lower bound $\Omega(\eta K \log T)$, which follo
    
[^292]: 基于Ising求解器的神经网络精确且渐近完备的鲁棒性验证

    Exact and Asymptotically Complete Robust Verifications of Neural Networks via Ising Solvers

    [https://arxiv.org/abs/2603.00408](https://arxiv.org/abs/2603.00408)

    该论文提出了一个与Ising求解器兼容的神经网络鲁棒性验证框架，通过对数编码（Log-PW）将每个神经元的二元变量复杂度从线性降至对数级别，并结合基于Jordan分解的单调化简与混合Benders求解器，实现了对包括非单调激活在内的各类激活函数的精确且渐近完备的鲁棒性验证。

    

    我们提出了一个与Ising求解器兼容的形式化神经网络鲁棒性验证框架，用于处理有界输入扰动下的验证问题。对于分段激活函数（包括连续分段线性和离散分段常数激活函数），精确对数分段模型（Log-PW）提供了一个精确、可靠且完备的表述，采用状态最优的对数编码，通过对数SOS2（Log-PWL）和SOS1（Log-PWC）约束，将每个神经元的二元变量数量从线性复杂度O(κ)降低到信息论意义上最小的对数复杂度O(⌈log₂κ⌉)。对于一般的逐元素激活函数，渐近阶梯包络模型（Step-Env）利用Log-PWC构建可靠且可不断精化的端点包络。基于Jordan分解的精确单调化简方法进一步将该框架扩展到任意非单调激活函数。我们进一步开发了具有输出敏感迭代界的混合Benders求解器。

    arXiv:2603.00408v5 Announce Type: replace  Abstract: We present an Ising-compatible framework for formal neural-network robustness verification under bounded input perturbations. For piecewise activations (both continuous piecewise-linear and discrete piecewise-constant), the Exact Logarithmic Piecewise Model (Log-PW) provides an exact, sound, and complete formulation with a state-optimal logarithmic encoding, reducing the binary variables per neuron from linear $O(\kappa)$ to information-theoretically minimal logarithmic complexity $O(\lceil \log_2 \kappa \rceil)$ via logarithmic SOS2 (Log-PWL) and SOS1 (Log-PWC) constraints. For general element-wise activations, the Asymptotic Step-Envelope Model (Step-Env) leverages Log-PWC to construct sound, refinable endpoint enclosures. An exact monotonic reduction based on Jordan decomposition further extends this framework to arbitrary non-monotonic activations. We further develop a hybrid Benders solver with output-sensitive iteration bounds,
    
[^293]: 面向电路高效量子数据编码的结构化酉张量网络表示

    Structured Unitary Tensor Network Representations for Circuit-Efficient Quantum Data Encoding

    [https://arxiv.org/abs/2602.16266](https://arxiv.org/abs/2602.16266)

    该论文提出了一种基于结构化酉张量网络表示的电路高效量子数据编码框架TNQE，通过将经典数据分解为张量核心并编译为编码电路，同时引入酉感知约束将核心参数化为可学习的块酉矩阵，从而降低量子机器学习中数据编码所需的电路深度与量子资源开销。

    

    将经典数据编码到量子态中是量子机器学习的一个核心瓶颈：许多广泛使用的编码方式电路效率低下，需要深层电路和大量量子资源，这限制了其在量子硬件上的可扩展性。在本工作中，我们提出了基于张量网络的量子数据编码（TNQE），这是一个建立在结构化酉张量网络（TN）表示之上的电路高效框架。TNQE首先通过张量网络分解来表示每个经典输入，然后通过两种互补的核心到电路策略将所得的张量核心编译为编码电路。为了使这一编译过程可训练，同时尊重量子操作的酉性质，我们引入了一种酉感知约束，将张量网络核心参数化为可学习的块酉矩阵，使其能够被直接优化并直接编码为量子算符。所提出的TNQE框架实现了显式控制……

    arXiv:2602.16266v2 Announce Type: replace-cross  Abstract: Encoding classical data into quantum states is a central bottleneck in quantum machine learning: many widely used encodings are circuit-inefficient, requiring deep circuits and substantial quantum resources, which limits scalability on quantum hardware. In this work, we propose tensor network-based quantum data encoding (TNQE), a circuit-efficient framework built on structured unitary tensor network (TN) representations. TNQE first represents each classical input via a TN decomposition and then compiles the resulting tensor cores into an encoding circuit through two complementary core-to-circuit strategies. To make this compilation trainable while respecting the unitary nature of quantum operations, we introduce a unitary-aware constraint that parameterizes TN cores as learnable block unitaries, enabling them to be directly optimized and directly encoded as quantum operators. The proposed TNQE framework enables explicit control
    
[^294]: 无需人类示范学习在新城市中驾驶

    Learning to Drive in New Cities Without Human Demonstrations

    [https://arxiv.org/abs/2602.15891](https://arxiv.org/abs/2602.15891)

    提出NOMAD方法，通过自博弈多智能体强化学习，仅利用目标城市的地图和元信息即可将驾驶策略适配到新城市，完全无需该城市的人类示范轨迹。

    

    虽然自动驾驶汽车在特定运行区域内已实现可靠性能，但将其部署到新城市仍然成本高昂且进展缓慢。一个关键瓶颈在于：当将驾驶策略适配到与训练城市在道路几何结构、交通规则和交互模式方面存在差异的新城市时，需要收集大量人类示范轨迹。在本文中，我们证明了自博弈多智能体强化学习可以仅利用地图和元信息，将驾驶策略适配到一个差异显著的目标城市，而无需来自该城市的任何人类示范。我们提出了面向自动驾驶的无数据基于地图的自博弈方法（NOMAD），该方法能够在基于目标城市地图构建的模拟器中实现策略适配。使用简单的奖励函数，NOMAD显著提升了目标城市的任务成功率和轨迹真实性，展示了一种有效的……

    arXiv:2602.15891v2 Announce Type: replace-cross  Abstract: While autonomous vehicles have achieved reliable performance within specific operating regions, their deployment to new cities remains costly and slow. A key bottleneck is the need to collect many human demonstration trajectories when adapting driving policies to new cities that differ from those seen in training in terms of road geometry, traffic rules, and interaction patterns. In this paper, we show that self-play multi-agent reinforcement learning can adapt a driving policy to a substantially different target city using only the map and meta-information, without requiring any human demonstrations from that city. We introduce NO data Map-based self-play for Autonomous Driving (NOMAD), which enables policy adaptation in a simulator constructed based on the target-city map. Using a simple reward function, NOMAD substantially improves both task success rate and trajectory realism in target cities, demonstrating an effective and
    
[^295]: 两个大语言模型比一个好吗？一种用于制药内容优化的师生双头大语言模型架构

    Are Two LLMs Better Than One? A Student-Teacher Dual-Head LLMs Architecture for Pharmaceutical Content Optimization

    [https://arxiv.org/abs/2602.11957](https://arxiv.org/abs/2602.11957)

    本文提出面向制药合规内容审核的LRBTC架构，采用师生双头LLM/VLM结合瀑布式规则过滤逻辑，在AIReg Bench上达到83.0% F1与97.5%召回率，漏检违规较Gemini 2.5 Pro减少五倍。

    

    大语言模型越来越多地被应用于制药等受监管领域的内容创作，在这些领域中，内容必须达到医学准确性和法律合规性。传统的人工质量控制（QC）速度慢、易出错，并造成发布瓶颈。为解决这一问题，我们提出了一种模块化、由LLM/VLM驱动的QC架构——LRBTC（语言、法规、品牌、技术、内容结构），通过师生人机协同架构和瀑布式规则过滤逻辑实现可扩展、可验证的内容审核。该架构旨在提供可验证、可追溯、可扩展的内容优化。我们的方法在AIReg Bench上实现了83.0%的F1分数和97.5%的召回率，与Gemini 2.5 Pro相比将漏检违规减少了五倍。在CSpelling数据集上，平均准确率提升了26.7%。我们的误差分析进一步揭示，尽管当前模型在检测拼写错误方面表现较强（92.5%的召回率）

    arXiv:2602.11957v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly used for content creation in regulated domains such as pharmaceuticals, where content must be medical accurate and legally compliant. Traditional manual quality control (QC) is slow, error-prone, and create publication bottlenecks. To address this, we introduce a modular, LLM/VLMs-driven QC architecture, LRBTC (Language, Regulatory, Brand, Technical, Content Structure), implemented through a Student Teacher HITL architecture and Waterfall rule filtering logic for scalable, verifiable content validation. This architecture is designed to provide verifiable, traceable, and scalable content optimization. Our methods achieve 83.0% F1 and 97.5% recall on AIReg Bench, reducing missed violations five fold compared with Gemini 2.5 Pro. On CSpelling, it improves mean accuracy by +26.7 %. Our error analysis further reveals that while current models are strong at detecting misspellings (92.5% recall)
    
[^296]: SleepMaMi：一个整合宏观与微观结构的通用睡眠基础模型

    SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures

    [https://arxiv.org/abs/2602.07628](https://arxiv.org/abs/2602.07628)

    SleepMaMi是一个通用睡眠基础模型，采用分层双编码器设计——宏观编码器建模整夜睡眠的时间依赖性，微观编码器捕捉生物信号的细粒度形态特征——从而突破传统任务特定模型只关注局部微观特征的局限，实现对整夜睡眠宏观架构与微观信号形态的统一建模。

    

    尽管向统一基础模型的转变已经彻底改变了许多深度学习领域，但睡眠医学在很大程度上仍局限于专注于局部微观结构特征的任务特定模型。这些方法往往忽略了多导睡眠图（PSG）丰富的多模态上下文，并且无法捕捉整夜睡眠的全局宏观结构。为了解决这一问题，我们提出了SleepMaMi，一个旨在同时掌握长达数小时的睡眠架构和细粒度信号形态的睡眠基础模型。我们的框架采用分层双编码器设计：宏观编码器（Macro-Encoder）用于建模整夜的时间依赖性，微观编码器（Micro-Encoder）用于从生物信号中捕捉短期特征。宏观编码器通过人口统计学引导的对比学习进行训练，该学习方法将整夜睡眠模式与客观的受试者元数据（如年龄、性别和BMI）对齐，以优化全局表示。微观编码器通过...

    arXiv:2602.07628v3 Announce Type: replace-cross  Abstract: While the shift toward unified foundation models has revolutionized many deep learning domains, sleep medicine remains largely restricted to task-specific models that focus on localized micro-structure features. These approaches often neglect the rich, multi-modal context of Polysomnography (PSG) and fail to capture the global macro-structure of a full night's sleep. To address this, we introduce SleepMaMi , a Sleep Foundation Model engineered to master both hour-long sleep architectures and fine-grained signal morphologies. Our framework utilizes a hierarchical dual-encoder design: a Macro-Encoder to model full-night temporal dependencies and a Micro-Encoder to capture short-term characteristics from biosignals. Macro-Encoder is trained via Demographic-Guided Contrastive Learning, which aligns overnight sleep patterns with objective subject metadata, such as age, sex and BMI to refine global representations. Micro-Encoder is o
    
[^297]: GFlowPO：生成流网络作为语言模型提示词优化器

    GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer

    [https://arxiv.org/abs/2602.03358](https://arxiv.org/abs/2602.03358)

    GFlowPO提出了一种概率化提示词优化框架，利用离线策略的生成流网络目标微调轻量级提示词语言模型，并通过无需训练的动态记忆更新机制更新元提示词，从而实现样本高效的语言模型提示词优化。

    

    为语言模型寻找有效的提示词至关重要却又极其困难：提示词空间在组合意义上非常庞大，且由于目标语言模型评估代价高昂，奖励十分稀疏。然而，现有的基于强化学习的提示词优化器通常依赖在线策略更新以及从固定分布中采样的元提示词，导致样本效率低下。我们提出GFlowPO，一个概率化的提示词优化框架，它将提示词搜索转化为一个后验推断问题：在潜在提示词上进行推断，并由以元提示词为条件的参考语言模型先验进行正则化。在第一步中，我们采用离线策略的生成流网络目标函数微调一个轻量级的提示词语言模型，并使用基于重放缓冲区的训练策略重用过去的提示词评估结果，从而实现样本高效的探索。在第二步中，我们引入动态记忆更新，这是一种无需训练的机制，通过注入以下内容来更新元提示词：（i）来自……的多样化提示词……（摘要在此处被截断）

    arXiv:2602.03358v2 Announce Type: replace-cross  Abstract: Finding effective prompts for language models (LMs) is critical yet notoriously difficult: the prompt space is combinatorially large, rewards are sparse due to expensive target-LM evaluation. Yet, existing RL-based prompt optimizers often rely on on-policy updates and a meta-prompt sampled from a fixed distribution, leading to poor sample efficiency. We propose GFlowPO, a probabilistic prompt optimization framework that casts prompt search as a posterior inference problem over latent prompts regularized by a meta-prompted reference-LM prior. In the first step, we fine-tune a lightweight prompt-LM with an off-policy Generative Flow Network (GFlowNet) objective, using a replay-based training policy that reuses past prompt evaluations to enable sample-efficient exploration. In the second step, we introduce Dynamic Memory Update (DMU), a training-free mechanism that updates the meta-prompt by injecting both (i) diverse prompts from
    
[^298]: TabClustPFN：一种用于表格数据聚类的先验数据拟合网络

    TabClustPFN: A Prior-data Fitted Network for Tabular Data Clustering

    [https://arxiv.org/abs/2601.21656](https://arxiv.org/abs/2601.21656)

    TabClustPFN通过协同设计的先验、训练目标与解耦架构，在单次推理中同时推断表格数据的聚类分配和未知聚类基数，从而克服了无监督、基数未知和标签交换等聚类中的核心挑战。

    

    先验数据拟合网络（PFNs）已将监督式表格学习重构为无需针对每个数据集进行优化的单次上下文内推理。将这一范式扩展到聚类虽然很有吸引力，却面临着更根本性的挑战，原因在于缺乏监督信息、未知的聚类基数（簇的数量），以及划分输出所固有的标签交换问题。现有的基于PFN的聚类方法只能部分地应对这些挑战：要么需要将已知的聚类基数作为输入，要么依赖于不稳定的标签排序约定和限制性过强的合成先验。我们提出了TabClustPFN，这是一种聚类PFN，通过协同设计的先验、训练目标与架构来联合解决上述所有挑战。我们的混合预训练先验能够捕捉异构的真实表格数据几何结构；我们解耦的划分推断网络与基数推断网络架构能够在单次前向传递中同时推断聚类分配和聚类基数；我们的SoftARI训练目标……（原文摘要在此处截断）

    arXiv:2601.21656v4 Announce Type: replace  Abstract: Prior-data Fitted Networks (PFNs) have reframed supervised tabular learning as single-pass in-context inference without per-dataset optimization. Extending this paradigm to clustering is appealing yet fundamentally more challenging, due to absent supervision, unknown cluster cardinality, and label switching inherent to partition outputs. Existing PFN-based clustering methods address these challenges only partially, either requiring known cardinality as input or relying on unstable label-ordering conventions and overly restrictive synthetic priors. We introduce TabClustPFN, a clustering PFN that resolves all challenges jointly through co-designed prior, objective, and architecture. Our hybrid pretraining prior captures heterogeneous real-tabular geometry; our decoupled partition inference network--cardinality inference network architecture jointly infers cluster assignments and cardinality in a single pass; and our SoftARI training ob
    
[^299]: 多指标模型中多轮随机梯度下降的高维学习动力学

    High-dimensional learning dynamics of multi-pass Stochastic Gradient Descent in multi-index models

    [https://arxiv.org/abs/2601.21093](https://arxiv.org/abs/2601.21093)

    本文在样本量与维度成比例的高维区间内，首次对非凸多指标模型上的多遍随机梯度下降动力学建立了渐近精确的刻画，证明其收敛于一种新颖的动力学平均场连续时间极限，且适用于任意次线性批量大小及相应的临界学习率缩放。

    

    随机梯度下降（SGD）的学习动力学被认为对训练模型的泛化性能有着重要影响。尽管在线（单遍）SGD如何精确优化总体风险的动力学已在广泛的高维问题中得到理解，但对于用于经验风险最小化的多遍SGD，类似的理解在很大程度上仍然缺失。在这项工作中，我们在样本量n与数据维度d成比例增长的高维区间内，为非凸多指标模型的多遍SGD动力学建立了渐近精确的刻画。我们考虑任意次线性批量大小κ ≍ n^α（其中α ∈ [0,1)）以及相应的“临界”学习率缩放，并证明SGD的逐坐标动力学具有一种新颖的连续时间缩放极限，该极限由一组动力学平均场（方程给出）……

    arXiv:2601.21093v3 Announce Type: replace-cross  Abstract: The learning dynamics of Stochastic Gradient Descent (SGD) are believed to play an important role in influencing the generalization properties of trained models. While the precise dynamics of how online (one-pass) SGD optimizes the population risk are now understood in a broad range of high-dimensional problems, a comparable understanding of multi-pass SGD for empirical risk minimization is still largely lacking.   In this work, we develop an asymptotically exact characterization of the dynamics of multi-pass SGD for a non-convex multi-index model, in the high-dimensional regime where sample size $n$ and data dimension $d$ increase proportionally. We consider any sub-linear batch size $\kappa \asymp n^\alpha$ with $\alpha \in [0,1)$ and a commensurate ``critical'' scaling of the learning rate, and show that the coordinate-wise dynamics of SGD admit a novel continuous-time scaling limit, given by a system of dynamical mean-field
    
[^300]: 慢性肾病中面向多群体的惩罚化公平回归

    Penalized Fair Regression for Multiple Groups in Chronic Kidney Disease

    [https://arxiv.org/abs/2512.17340](https://arxiv.org/abs/2512.17340)

    提出一种面向多群体的惩罚化公平回归通用框架，通过不公平性惩罚与新颖的自动惩罚权重选择评分函数，实现高效求解，并在慢性肾病研究中取得优于现有方法的公平性-准确性平衡。

    

    公平回归方法有潜力缓解医疗保健领域的社会偏见问题，但当多个群体同时遭受此类偏见时，关于惩罚化公平回归的研究却十分匮乏。我们提出了一个通用的回归框架，通过对多个群体施加不公平性惩罚来填补这一空白。我们的方法在二元结果上以真阳性率差异惩罚的形式进行了展示，并可通过归约为代价敏感分类问题来高效实现。此外，我们还引入了新颖的评分函数，用于自动选择惩罚权重。我们在仿真实验中对这些惩罚化公平回归方法进行了实证研究，结果显示它们达到了超越现有对比方法的公平性-准确性边界。最后，我们将这些方法应用于一项全国多中心慢性肾病初级保健研究，以开发针对终末期肾病的公平分类器。在该研究中，我们发现子（原文摘要在此处被截断）……

    arXiv:2512.17340v3 Announce Type: replace-cross  Abstract: Fair regression methods have the potential to mitigate societal bias concerns in health care, but there has been little work on penalized fair regression when multiple groups experience such bias. We propose a general regression framework that addresses this gap with unfairness penalties for multiple groups. Our approach is demonstrated for binary outcomes with true positive rate disparity penalties. It can be efficiently implemented through reduction to a cost-sensitive classification problem. We additionally introduce novel score functions for automatically selecting penalty weights. Our penalized fair regression methods are empirically studied in simulations, where they achieve a fairness-accuracy frontier beyond that of existing comparison methods. Finally, we apply these methods to a national multi-site primary care study of chronic kidney disease to develop a fair classifier for end-stage renal disease. There we find subs
    
[^301]: 基于自编码器潜在空间的未知动力系统验证

    Verification of Unknown Dynamical Systems via Autoencoder Latent Space

    [https://arxiv.org/abs/2512.13593](https://arxiv.org/abs/2512.13593)

    该论文提出了一种通过凸自编码器降维并利用基于核的方法在潜在空间中学习动力学的形式化验证框架，保证了所构建的有限抽象包含原系统的真实行为，并可将潜在空间中的验证结果映射回原始系统。

    

    形式化验证为证明动力系统满足其规约提供了强大的框架。然而，这些技术在高维场景下面临可扩展性挑战，因为它们通常依赖于随维度呈指数级增长的状态空间离散化。基于学习的降维方法在缓解这一问题上展现出巨大潜力，但如何确保潜在空间验证结果的正确性仍是一个悬而未决的问题。在本工作中，我们提供了一种通过凸自编码器对系统进行降维的形式化方法，并通过基于核的方法在潜在空间中学习系统动力学。随后，我们从潜在空间中学到的模型构建有限抽象，并保证该抽象包含原始系统的真实行为。我们证明了在潜在空间中得到的验证结果可以映射回原始系统。

    arXiv:2512.13593v5 Announce Type: replace  Abstract: Formal verification provides a powerful framework for proving that dynamical systems satisfy their specifications. However, these techniques face scalability challenges in high-dimensional settings, as they often rely on state-space discretization which grows exponentially with dimension. Learning-based approaches to dimensionality reduction have shown great potential to alleviate this problem. However, ensuring correctness of latent space verification results remains an open question. In this work, we provide a formal approach to reduce the dimensionality of systems via convex autoencoders and learn the dynamics in the latent space through a kernel-based method. We then construct a finite abstraction from the learned model in the latent space and guarantee that the abstraction contains the true behaviors of the original system. We show that the verification results in the latent space can be mapped back to the original system. Final
    
[^302]: LLM-Auction：面向大语言模型原生广告的生成式拍卖机制

    LLM-Auction: Generative Auction towards LLM-Native Advertising

    [https://arxiv.org/abs/2512.10551](https://arxiv.org/abs/2512.10551)

    该论文提出了首个将拍卖与生成相融合的学习式生成式拍卖机制LLM-Auction，通过把广告分配构建为LLM输出与平衡广告主价值和用户体验的机制目标之间的偏好对齐，使LLM无需额外推理成本即可内在建模分配外部性，解决了传统机制在LLM原生广告场景中失效的问题。

    

    大语言模型（LLM）应用的商业化是在线广告的下一个前沿领域，LLM原生广告作为一种极具前景的新范式应运而生，它通过将广告整合到LLM生成的内容中实现商业化。然而，在这种拍卖对象从离散广告位转变为LLM输出分布的新场景下，经典拍卖机制已不再适用；同时，现有方法由于忽略了外部性或推理成本过高，在工业场景中难以实用。为解决这些问题，我们提出了LLM-Auction，这是首个将拍卖与生成相融合的基于学习的生成式拍卖机制。通过将广告分配问题构建为LLM输出与机制目标（该目标平衡广告主价值与用户体验）之间的偏好对齐，我们对LLM进行优化，使其能够内在地建模分配外部性，且无需额外的推理成本。在理论层面，我们刻画了该机制的分配单调性与连续性（原文摘要在此处截断）。

    arXiv:2512.10551v3 Announce Type: replace-cross  Abstract: The commercialization of LLM applications is the next frontier in online advertising, with LLM-native advertising emerging as a promising paradigm by integrating ads into LLM-generated content. However, classic mechanisms are no longer applicable in this setting where the auction object is shifted from discrete ad slots to distributions over LLM outputs, and existing methods are impractical in industrial scenarios due to ignored externalities or high inference costs. To address these issues, we propose LLM-Auction, the first learning-based generative auction mechanism that integrates auction and generation. By formulating the allocation as preference alignment between LLM outputs and a mechanism objective that balances advertisers' value and user experience, we optimize the LLMs to inherently model allocation externalities without extra inference cost. Theoretically, we identify the allocation monotonicity and continuity of the
    
[^303]: MarkTune：改进模型嵌入式大语言模型水印的质量-可检测性权衡

    MarkTune: Improving the Quality-Detectability Trade-off in Model-Embedded LLM Watermarking

    [https://arxiv.org/abs/2512.04044](https://arxiv.org/abs/2512.04044)

    该论文提出MarkTune，通过向高质量参考分布正则化来解耦质量保持与检测能力之间的关系，从而改进了模型嵌入式大语言模型水印中生成质量与可检测性之间的权衡。

    

    语言模型水印方案大致可分为两大类：推理时方法（在生成时修改解码过程）和模型嵌入方法（将秘密信号直接编码到模型权重中）。推理时水印方法会引入额外的推理开销，且不适用于日益普及的开放权重模型场景。相比之下，模型嵌入水印不会带来额外的推理延迟（生成时使用标准的采样流程），因此特别适合开放权重场景。然而，现有的模型嵌入方法（如GaussMark）面临一个根本性的质量-可检测性权衡：要获得强大的检测能力，通常需要对权重进行扰动，而这会明显降低生成质量。我们的关键洞察是，向一个高质量的参考分布进行正则化，可以将质量保持与对基础模型的接近程度解耦（摘要在此处截断）

    arXiv:2512.04044v2 Announce Type: replace  Abstract: Language model watermarking schemes fall into two broad categories: inference-time methods, which modify the decoding process at generation time, and model-embedded methods, which encode a secret signal directly into the model weights. Inference-time watermarking methods incur additional inference overhead and are not applicable in the increasingly prevalent open-weight model settings. In contrast, model-embedded watermarks introduce no additional inference latency-generation uses the standard sampling pipeline-and are particularly well-suited to open-weight settings. However, existing model-embedded approaches such as GaussMark face a fundamental quality-detectability trade-off: achieving strong detection power typically requires weight perturbations that noticeably degrade generation quality. Our key insight is that regularizing toward a high-quality reference distribution decouples quality preservation from proximity to the base m
    
[^304]: 强化学习后训练如何诱导技能组合？——以Countdown任务为例的案例研究

    How Does RL Post-training Induce Skill Composition? A Case Study on Countdown

    [https://arxiv.org/abs/2512.01775](https://arxiv.org/abs/2512.01775)

    本文通过将Countdown任务中模型的解答解析为表达式树，发现RL后训练能诱导模型对子任务技能进行组合式复用（实现对更大规模和未见结构的分布外泛化），且技能习得的顺序由任务的组合结构决定。

    

    虽然强化学习（RL）成功提升了大型语言模型的推理能力，但其在促进组合泛化（即从已知组件合成新技能的能力）方面的作用，常与单纯的长度泛化相混淆。为此，我们研究了RL后训练在技能组合方面究竟教会了模型什么，以及组合结构如何影响技能迁移。我们聚焦于Countdown任务（给定n个数字和一个目标值，构造一个运算结果等于目标值的表达式），并将模型解答解析为表达式树，其中每个子树对应一个可复用的子任务，因而可被视为一种“技能”。通过跟踪树形状及其成功率随训练的变化，我们发现：(i) 模型能对更大的n以及未见过的树形状实现分布外（OOD）泛化，表明子任务被组合式复用；(ii) 技能习得的顺序取决于组合结构——模型先掌握浅层平衡（摘要原文在此处截断）

    arXiv:2512.01775v2 Announce Type: replace  Abstract: While reinforcement learning (RL) successfully enhances reasoning in large language models, its role in fostering compositional generalization (the ability to synthesize novel skills from known components) is often conflated with mere length generalization. To this end, we study what RL post-training teaches about skill composition and how the composition structure affects the skill transfer. We focus on the Countdown task (given n numbers and a target, form an expression that evaluates to the target) and analyze model solutions as expression trees, where each subtree corresponds to a reusable subtask and thus can be viewed as a "skill." Tracking tree shapes and their success rates over training, we find: (i) out-of-distribution (OOD) generalization to larger n and to unseen tree shapes, indicating compositional reuse of subtasks; (ii) the order of skill acquisition depends upon the composition structure---models master shallow balan
    
[^305]: 基于贝叶斯模糊性收缩的自适应鲁棒马尔可夫决策过程在对抗性监视任务中的应用

    Bayesian Ambiguity Contraction-based Adaptive Robust Markov Decision Processes for Adversarial Surveillance Missions

    [https://arxiv.org/abs/2512.01660](https://arxiv.org/abs/2512.01660)

    本文提出一种基于贝叶斯模糊性收缩的自适应鲁棒马尔可夫决策过程框架，通过在线剔除与观测不一致的威胁原型来逐步收紧不确定性集合，为协同作战飞机在对抗环境下的侦察监视任务实现更精确且安全的实时决策。

    

    协同作战飞机（CCA）被期望能够在对抗环境中执行自主的情报、监视与侦察（ISR）任务，而在这些环境中，对手可能采取策略性行动来欺骗或逃避探测。由于模型不确定性以及安全实时决策的需求，这类任务面临诸多挑战。鲁棒马尔可夫决策过程（RMDP）虽然能够提供最坏情况下的保证，但其受限于静态的模糊集——这类模糊集只能刻画初始不确定性，而无法随新观测进行调整。本文提出了一种专为CCA的ISR任务设计的自适应RMDP框架。我们引入了一种任务特定的建模方式，其中CCA在移动状态与感知状态之间交替切换。对抗性不确定性由有限的威胁原型来表示，这些原型定义了观测模型、暴露模型以及状态-动作转移模型。我们的方法通过逐步剔除与观测不一致的威胁模型来增量式地精炼策略，从而……

    arXiv:2512.01660v2 Announce Type: replace-cross  Abstract: Collaborative Combat Aircraft (CCAs) are envisioned to enable autonomous Intelligence, Surveillance, and Reconnaissance (ISR) missions in contested environments, where adversaries may act strategically to deceive or evade detection. These missions pose challenges due to model uncertainty and the need for safe, real-time decision-making. Robust Markov Decision Processes (RMDPs) provide worst-case guarantees but are limited by static ambiguity sets that capture initial uncertainty without adapting to new observations. This paper presents an adaptive RMDP framework tailored to ISR missions with CCAs. We introduce a mission-specific formulation in which the CCA alternates between movement and sensing states. Adversarial uncertainty is represented by finite threat prototypes defining observation, exposure, and state-action transition models. Our approach incrementally refines policies by eliminating inconsistent threat models, allow
    
[^306]: FLAWS：科学论文中错误识别与定位的基准测试

    FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers

    [https://arxiv.org/abs/2511.21843](https://arxiv.org/abs/2511.21843)

    该论文提出了FLAWS基准，通过在同行评审论文中系统性插入破坏关键论断的错误，构建了713个论文-错误对，以自动化评估大语言模型检测和定位科学论文中错误的能力。

    

    错误的识别与定位是同行评审中的一项核心任务，然而科学产出的指数级增长，加之专家资源有限，使得人类评审者越来越难以可靠地检测出错误。大语言模型（LLM）的最新进展激发了人们对其在这类评估任务中潜力的兴趣，涵盖从学术同行评审到自动化科学评估等应用。然而，尽管LLM在评审系统中的应用日益增多，其精确定位错误的能力却仍未得到充分探索。在本工作中，我们提出了科学写作中的错误定位基准FLAWS（Fault Localization Across Writing in Science），这是一个包含713个论文-错误对的自动化基准，旨在评估LLM检测那些破坏研究论文关键论断的错误的有效性。我们通过利用LLM系统性地在经同行评审的论文中插入使论断失效的错误来构建该基准，并配套了自动化的评估指标。

    arXiv:2511.21843v2 Announce Type: replace  Abstract: The identification and localization of errors is a core task in peer review, yet the exponential growth of scientific output has made it increasingly difficult for human reviewers to reliably detect errors given the limited pool of experts. Recent advances in Large Language Models (LLMs) have sparked interest in their potential to support such evaluation tasks, from academic peer review to automated scientific assessment. However, despite the growing use of LLMs in review systems, their capabilities to pinpoint errors remain underexplored. In this work, we introduce Fault Localization Across Writing in Science (FLAWS), an automated benchmark consisting of 713 paper-error pairs designed to evaluate how effectively LLMs detect errors that undermine key claims in research papers. We construct the benchmark by systematically inserting claim-invalidating errors into peer-reviewed papers using LLMs, paired with an automated evaluation metr
    
[^307]: Specter：通过量化使能的多步投机卸载实现快速MoE推理

    Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading

    [https://arxiv.org/abs/2511.14102](https://arxiv.org/abs/2511.14102)

    Specter通过量化使能的多步投机解码与专家卸载相结合，在单GPU上实现了平均1.50倍（最高1.88倍）的MoE推理加速。

    

    混合专家模型通常超出单个加速器的内存容量。将专家卸载到主机内存会造成严重的PCIe I/O瓶颈，因为数据依赖的路由使数据传输处于关键路径上。我们提出了Specter，一个利用量化将投机解码与专家卸载相结合的推理系统。其轻量级量化草稿模型在设备上高效运行，同时保持高保真度的专家激活模式。这为未来的专家使用提供了准确的多步前瞻，使运行时能够在验证之前提前组织专家的移动和执行，从而同时提高通信和执行效率。在单请求推理下，Specter在每输出令牌时间（TPOT）方面相比最先进的系统实现了平均1.50倍（最高1.88倍）的加速。Specter使内存受限的单GPU MoE推理在商用硬件上更加实用。

    arXiv:2511.14102v2 Announce Type: replace  Abstract: Mixture-of-experts (MoE) models often exceed a single accelerator's memory capacity. Offloading experts to host memory creates a severe PCIe I/O bottleneck because data-dependent routing keeps transfers on the critical path. We present Specter, an inference system that uses quantization to combine speculative decoding with expert offloading. Its lightweight quantized draft model runs efficiently on device while preserving high-fidelity expert activation patterns. This provides accurate multi-step lookahead over future expert usage, allowing the runtime to organize expert movement and execution ahead of verification and improve both communication and execution efficiency. Under single-request inference, Specter achieves an average 1.50x (up to 1.88x) speedup in time per output token (TPOT) over the state-of-the-art system. Specter makes memory-constrained, single-GPU MoE inference more practical on commodity hardware. Our code is avai
    
[^308]: 基于贝叶斯最优判别的GAN重探

    Revisiting GAN with Bayes-Optimal Discrimination

    [https://arxiv.org/abs/2510.25609](https://arxiv.org/abs/2510.25609)

    该论文提出用贝叶斯错误率上界（BOLT损失）替代传统交叉熵目标来重构GAN训练，理论上证明其对抗博弈等价于最小化数据与生成分布间的总变差，并与Wasserstein-1距离建立关联，实验显示样本质量与覆盖度均得到改善。

    

    我们从生成对抗网络（GAN）训练最初的真实与伪造分类视角出发，用贝叶斯错误率（BER）的上界替代交叉熵判别器目标。利用贝叶斯最优学习阈值（BOLT）损失，我们构建了一个对抗博弈：判别器收紧该上界，而生成器最大化它。我们刻画了无约束下的最优判别器，并证明在类别先验平衡的情况下，该总体博弈等价于最小化数据分布与生成器分布之间的总变差。将有界判别器限制为1-Lipschitz，即得到BOLT-GAN。对于平衡先验和完整判别器类，我们建立了将其总体差异与有界域上Wasserstein-1距离相关联的上下界。在四个图像生成基准上的受控实验表明，该方法的样本质量和覆盖度均有提升。

    arXiv:2510.25609v4 Announce Type: replace  Abstract: We approach generative adversarial network (GAN) training from its original real-versus-fake classification viewpoint, replacing the cross-entropy discriminator objective with an upper bound on the Bayes error rate (BER). Using the Bayes optimal learning threshold (BOLT) loss, we formulate an adversarial game in which the discriminator tightens this bound and the generator maximizes it. We characterize the unrestricted optimal discriminator and show that, under balanced class priors, the population game is equivalent to minimizing total variation between the data and generator distributions. Restricting the bounded discriminator to be 1-Lipschitz yields BOLT-GAN. For balanced priors and the full discriminator class, we establish upper and lower bounds relating its population discrepancy to Wasserstein-1 distance on bounded domains. Controlled experiments on four image-generation benchmarks show improved sample quality and coverage ov
    
[^309]: FlowCritic：在强化学习中架起价值估计与流匹配的桥梁

    FlowCritic: Bridging Value Estimation with Flow Matching in Reinforcement Learning

    [https://arxiv.org/abs/2510.22686](https://arxiv.org/abs/2510.22686)

    FlowCritic提出了一种基于流匹配的生成式价值估计新范式，通过建模价值分布并生成样本来进行估计，克服了多评论家集成缺乏分布信息和分布式RL表达能力受限的问题。

    

    可靠的价值估计是强化学习（RL）的基石，它通过评估长期回报并指导策略改进，显著影响着收敛速度和最终性能。现有工作通过多评论家集成和分布式强化学习来提升价值函数估计的可靠性，然而前者仅组合了多个点估计而未能捕获分布信息，后者则依赖于离散化或分位数回归，限制了复杂价值分布的表达能力。受流匹配在生成建模领域取得成功的启发，我们提出了一种价值估计的生成式范式，命名为FlowCritic。FlowCritic摒弃了用于确定性价值预测的传统回归方法，转而利用流匹配对价值分布进行建模，并生成样本用于价值估计。

    arXiv:2510.22686v2 Announce Type: replace  Abstract: Reliable value estimation serves as the cornerstone of reinforcement learning (RL) by evaluating long-term returns and guiding policy improvement, significantly influencing the convergence speed and final performance. Existing works improve the reliability of value function estimation via multi-critic ensembles and distributional RL, yet the former merely combines multi point estimation without capturing distributional information, whereas the latter relies on discretization or quantile regression, limiting the expressiveness of complex value distributions. Inspired by flow matching's success in generative modeling, we propose a generative paradigm for value estimation, named FlowCritic. Departing from conventional regression for deterministic value prediction, FlowCritic leverages flow matching to model value distributions and generate samples for value estimation.
    
[^310]: 如何教会大型多模态模型学习新技能

    How to Teach Large Multimodal Models New Skills

    [https://arxiv.org/abs/2510.08564](https://arxiv.org/abs/2510.08564)

    该研究发现大型多模态模型在顺序学习新技能时的遗忘与输出token分布偏移共同变化，并提出两种简单微调策略——仅更新自注意力投影层或仅更新MLP的Gate&Up（冻结Down投影），能在强劲学习新技能的同时将通用能力遗忘降至最低。

    

    如何在不擦除已有能力的前提下，教大型多模态模型（LMMs）学习新技能？我们研究了在五个目标技能上的顺序微调，同时监测三个模型家族在八个留出基准上的通用能力。令人惊讶的是，我们发现模型在微调某一项技能后在留出任务上损失的性能，在随后微调另一项不同技能时能够部分恢复。我们将这种行为追溯到输出token分布的可测量偏移，并通过一个简单的计数偏差探针来体现，该探针显示这种偏移与遗忘共同变化。基于这一洞察，我们识别出两种简单而稳健的微调方案，它们在强劲学习的同时限制了漂移：（i）仅更新自注意力投影层（SA Proj.，Δ学习 +24.9 / Δ留出遗忘 -0.6），以及（ii）仅更新MLP的Gate&Up而冻结Down投影（+30.5 / -2.1）。两者都显著优于……

    arXiv:2510.08564v3 Announce Type: replace-cross  Abstract: How can we teach large multimodal models (LMMs) new skills without erasing prior abilities? We study sequential fine-tuning on five target skills while monitoring general ability on eight held-out benchmarks across three model families. Surprisingly, we find that performance lost on held-out tasks after fine-tuning on one skill can partly recover when the model is subsequently tuned on a different skill. We trace this behavior to a measurable shift in the output token distribution, manifested through a simple counting-bias probe that shows the shift co-varies with forgetting. Guided by this insight, we identify two simple, robust tuning recipes that learn strongly while limiting drift: (i) updating only the self-attention projection layers (SA Proj., $\Delta$ learning +24.9 / $\Delta$ held-out forgetting -0.6), and (ii) updating only the MLP Gate&Up while freezing the Down projection (+30.5 / -2.1). Both substantially outperfor
    
[^311]: 蒸馏还是决策？理解部分可观测强化学习中的算法权衡

    To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning

    [https://arxiv.org/abs/2510.03207](https://arxiv.org/abs/2510.03207)

    本文通过扰动块MDP理论模型与模拟运动任务上的对照实验，系统揭示了部分可观测强化学习中特权专家蒸馏与无特权信息标准强化学习之间的算法权衡及其关键影响因素。

    

    部分可观测性是强化学习（RL）中一个众所周知的难题，因为它需要学习复杂的、依赖历史的策略。近期的实证成功采用了特权专家蒸馏方法——利用训练期间（例如来自模拟器）可获得的潜在状态信息来学习并模仿最优的潜在马尔可夫策略——从而将“学习观察”与“学习行动”这两个任务解耦。虽然专家蒸馏比没有潜在状态信息的强化学习在计算上更高效，但它也存在有据可查的失败模式。在本文中，我们通过一个简单而具有启发性的理论模型——扰动块MDP（perturbed Block MDP），以及在具有挑战性的模拟运动任务上的对照实验，研究了特权专家蒸馏与无特权信息的标准强化学习之间的算法权衡。我们的主要发现是：（1）该权衡在经验上取决于随机性……

    arXiv:2510.03207v2 Announce Type: replace  Abstract: Partial observability is a notorious challenge in reinforcement learning (RL), due to the need to learn complex, history-dependent policies. Recent empirical successes have used privileged expert distillation--which leverages availability of latent state information during training (e.g., from a simulator) to learn and imitate the optimal latent, Markovian policy--to disentangle the task of "learning to see" from "learning to act". While expert distillation is more computationally efficient than RL without latent state information, it also has well-documented failure modes. In this paper--through a simple but instructive theoretical model called the perturbed Block MDP, and controlled experiments on challenging simulated locomotion tasks--we investigate the algorithmic trade-off between privileged expert distillation and standard RL without privileged information. Our main findings are: (1) The trade-off empirically hinges on the sto
    
[^312]: 基于拉普拉斯特征向量梯度正交化的鲁棒切空间估计

    Robust Tangent Space Estimation via Laplacian Eigenvector Gradient Orthogonalization

    [https://arxiv.org/abs/2510.02308](https://arxiv.org/abs/2510.02308)

    提出了一种名为LEGO的谱方法，通过正交化图拉普拉斯低频特征向量的梯度，利用数据的全局结构来鲁棒地估计流形切空间，克服了局部PCA在高噪声环境下邻域大小选择的两难困境。

    

    估计数据流形的切空间是几何数据分析中的一个基本问题。标准方法——局部主成分分析（LPCA）——由于在邻域大小选择上存在关键权衡，在高噪声环境下表现不佳。选择最优邻域大小需要事先了解数据的几何和噪声特性，而这些信息往往难以获得。本文提出了一种谱方法——拉普拉斯特征向量梯度正交化（LEGO），利用数据的全局结构来指导局部切空间估计。LEGO不再仅仅依赖局部邻域，而是通过对图拉普拉斯算子的低频特征向量的梯度进行正交化，来估计每个数据点处的切空间。我们为该方法提供了两个理论证明：首先，对流形管状邻域的微分几何分析表明，低频特征向量的梯度……

    arXiv:2510.02308v3 Announce Type: replace  Abstract: Estimating the tangent spaces of a data manifold is a fundamental problem in geometric data analysis. The standard approach, Local Principal Component Analysis (LPCA), struggles in high-noise settings due to a critical trade-off in choosing the neighborhood size. Selecting an optimal size requires prior knowledge of the geometric and noise characteristics of the data that are often unavailable. In this paper, we propose a spectral method, Laplacian Eigenvector Gradient Orthogonalization (LEGO), that utilizes the global structure of the data to guide local tangent space estimation. Instead of relying solely on local neighborhoods, LEGO estimates the tangent space at each data point by orthogonalizing the gradients of low-frequency eigenvectors of the graph Laplacian. We provide two theoretical justifications of our method. First, a differential geometric analysis on the tubular neighborhood of a manifold shows that gradients of the lo
    
[^313]: 基于资源利用率与机器学习的虚拟服务器能耗估算

    Energy Estimation for Virtual Servers Using Resource Utilization and Machine Learning

    [https://arxiv.org/abs/2509.09991](https://arxiv.org/abs/2509.09991)

    该论文提出一种机器学习方法，仅利用虚拟服务器内部可观测的系统和性能指标即可预测单个虚拟服务器的CPU功耗，训练时以主机侧RAPL测量作为功耗标签，在训练覆盖部署条件时预测效果良好。

    

    估算虚拟服务器的能耗十分困难，因为物理电源接口和特权主机计数器通常只能在底层物理主机上访问。我们研究了一种机器学习方法，该方法利用虚拟服务器内部可观测的系统和性能指标，来预测归属于单个虚拟服务器的CPU封装功耗。在训练阶段，主机侧的RAPL测量提供用于学习该映射关系的功耗目标值，而部署推理时则只需要虚拟服务器内部可获取的测量数据。我们采用时间分块评估方法，在异构CPU平台、多种工作负载类型、未见过的负载与硬件、并发工作负载组合以及不同资源配置等条件下对该方法进行了评估。当部署条件在训练期间得到覆盖时，预测效果总体较强，而向未见过的负载和硬件进行零样本迁移的效果则……

    arXiv:2509.09991v2 Announce Type: replace  Abstract: Estimating the energy use of virtual servers is difficult because physical power interfaces and privileged host counters are typically accessible only on the underlying physical host. We investigate a machine learning approach that predicts the CPU-package power attributable to an individual virtual server from system and performance metrics observable within that virtual server itself. During training, host-side RAPL measurements provide the power target used to learn this mapping, while deployed inference requires only measurements available within the virtual server. We evaluate the approach across heterogeneous CPU platforms, multiple workload families, unseen workloads and hardware, concurrent workload combinations, and varying resource allocations using temporally blocked evaluation. Prediction is generally strong when deployment conditions are represented during training, while zero-shot transfer to unseen workloads, hardware,
    
[^314]: 基于稀疏优化更新的随机匹配老虎机

    Stochastic Matching Bandits with Rare Optimization Updates

    [https://arxiv.org/abs/2509.04194](https://arxiv.org/abs/2509.04194)

    本文提出了MNL选择模型下随机匹配老虎机的批处理算法，通过将昂贵的匹配优化更新次数限制在Θ(log log T)次，在大幅降低计算开销的同时保持了遗憾最优性能。

    

    我们提出了一个在多项Logit（MNL）选择模型下的随机匹配老虎机框架。在该设定中，一侧的 $N$ 个智能体被分配到另一侧的 $K$ 个臂上，每个臂根据未知的偏好从其分配池中随机选择一个智能体，并在时间范围 $T$ 内产生相应的奖励。目标是通过最大化成功匹配的累积收益来最小化遗憾。一种朴素的方法需要在每一轮都求解一个NP难的组合优化问题，导致极其高昂的计算成本。为应对这一挑战，我们提出了批处理算法，将匹配分配的更新次数在整个时间范围内策略性地限制在 $\Theta(\log\log T)$ 次。通过仅在极小比例的轮次中调用昂贵的组合优化，我们的算法大幅降低了总体计算开销，同时仍然实现……

    arXiv:2509.04194v3 Announce Type: replace-cross  Abstract: We introduce a bandit framework for stochastic matching under the multinomial logit (MNL) choice model. In this setting, $N$ agents on one side are assigned to $K$ arms on the other side, where each arm stochastically selects an agent from its assigned pool according to unknown preferences and yields a corresponding reward over a horizon $T$. The objective is to minimize regret by maximizing the cumulative revenue from successful matches. A naive approach requires solving an NP-hard combinatorial optimization problem at every round, resulting in a prohibitive computational cost. To address this challenge, we propose batched algorithms that strategically limit the number of times matching assignments are updated to $\Theta(\log\log T)$ over the entire horizon. By invoking expensive combinatorial optimization only on a vanishing fraction of rounds, our algorithms substantially reduce overall computational overhead while still ach
    
[^315]: 稀疏方程匹配：一种面向一般阶动力系统的无导数学习方法

    Sparse Equation Matching: A Derivative-Free Learning for General-Order Dynamical Systems

    [https://arxiv.org/abs/2507.20072](https://arxiv.org/abs/2507.20072)

    提出稀疏方程匹配（SEM）统一框架，利用基于格林函数的积分型稀疏回归，实现对一般阶动力系统中微分算子和驱动函数的无导数学习。

    

    方程发现是揭示复杂系统潜在动力学的一项基础学习任务，在脑连接分析、气候建模、基因调控和物理仿真等领域有着广泛的应用。然而，许多现有方法依赖于精确的导数估计，且仅限于一阶动力系统，这限制了它们在现实场景中的适用性。在本工作中，我们提出了稀疏方程匹配，这是一个将多种现有方程发现方法统一在同一数学表述下的框架。SEM 引入了一种基于格林函数的积分型稀疏回归方法，能够在一般阶动力系统中实现无导数的微分算子及其相关驱动函数的估计。通过大量仿真实验并与基于导数的方法进行性能对比，SEM 的有效性得到了验证。

    arXiv:2507.20072v3 Announce Type: replace  Abstract: Equation discovery is a fundamental learning task for uncovering the underlying dynamics of complex systems, with wide-ranging applications in areas such as brain connectivity analysis, climate modeling, gene regulation, and physical simulation. However, many existing approaches rely on accurate derivative estimation and are limited to first-order dynamical systems, restricting their applicability in real-world scenarios. In this work, we propose Sparse Equation Matching (SEM), a unified framework that encompasses several existing equation discovery methods under a common formulation. SEM introduces an integral-based sparse regression approach using Green's functions, enabling derivative-free estimation of differential operators and their associated driving functions in general-order dynamical systems. The effectiveness of SEM is demonstrated through extensive simulations, benchmarking its performance against derivative-based approac
    
[^316]: 自动化细粒度分割辅助报告生成

    Automatic Fine-grained Segmentation-assisted Report Generation

    [https://arxiv.org/abs/2507.16623](https://arxiv.org/abs/2507.16623)

    该论文提出ASaRG方法，通过将专业放射学模型生成的中间特征和细粒度分割图以简单拼接的方式融合进LLaVA的多模态投影层，在仅增加少量参数的情况下提升了医学报告生成的性能与可信度。

    

    可靠的端到端临床报告生成一直是医学机器学习研究的一个长期目标。这一过程的最终目标是减轻放射科医生的工作负担，并为临床医生或患者提供第二诊疗意见。因此，报告生成模型的必要前提是具备强大的综合性能以及某种内在的可定位溯源能力，以使临床医生或患者确信所生成报告的真实性。在本文中，我们提出了ASaRG（自动分割辅助报告生成），这是对流行的LLaVA架构的一种扩展，旨在同时解决上述两个问题。ASaRG提出通过简单的拼接方式，将由专业放射学模型生成的中间特征和细粒度分割图融合到LLaVA的多模态投影层中。在仅增加少量参数的情况下，我们的方法实现了+0.89%的性能提升。

    arXiv:2507.16623v2 Announce Type: replace-cross  Abstract: Reliable end-to-end clinical report generation has been a longstanding goal of medical ML research. The end goal for this process is to alleviate radiologists' workloads and provide second opinions to clinicians or patients. Thus, a necessary prerequisite for report generation models is a strong general performance and some type of innate grounding capability, to convince clinicians or patients of the veracity of the generated reports. In this paper, we present ASaRG (\textbf{A}utomatic \textbf{S}egmentation-\textbf{a}ssisted \textbf{R}eport \textbf{G}eneration), an extension of the popular LLaVA architecture that aims to tackle both of these problems. ASaRG proposes to fuse intermediate features and fine-grained segmentation maps created by specialist radiological models into LLaVA's multi-modal projection layer via simple concatenation. With a small number of added parameters, our approach achieves a +0.89\% performance gain 
    
[^317]: 面向多体哈密顿量的吉布斯态元学习及其在量子玻尔兹曼机中的应用

    Meta-learning of Gibbs states for many-body Hamiltonians with applications to Quantum Boltzmann Machines

    [https://arxiv.org/abs/2507.16373](https://arxiv.org/abs/2507.16373)

    该论文提出了两种元学习算法，通过对有限训练参数值集合进行联合优化，在NISQ设备上高效制备参数化多体哈密顿量的吉布斯态，并可应用于量子玻尔兹曼机。

    

    量子吉布斯态的制备是量子计算中的一项基本挑战，对于从开放量子系统建模到量子机器学习等诸多应用至关重要。基于Cervera-Lierta等人提出的元变分量子本征求解器（Meta-VQE）框架和问题驱动的拟设设计，我们提出了两种元学习算法：元变分量子热化器和神经网络元变分量子热化器（NN-Meta VQT），用于在含噪声中等规模量子（NISQ）设备上高效制备参数化哈密顿量的热态。Meta-VQT采用全量子拟设，而NN-Meta VQT则结合了量子-经典混合架构。两者均通过对有限训练参数值集合进行联合优化，在采样参数域内的留出参数值处制备近似吉布斯态。我们在最高若干比特规模的横向场伊辛模型（TFIM）上验证了我们的方法……

    arXiv:2507.16373v3 Announce Type: replace-cross  Abstract: The preparation of quantum Gibbs states is a fundamental challenge in quantum computing, essential for applications ranging from modeling open quantum systems to quantum machine learning. Building on the Meta-Variational Quantum Eigensolver framework proposed by Cervera-Lierta et al. and a problem-driven ansatz design, we introduce two meta-learning algorithms: Meta-Variational Quantum Thermalizer (Meta-VQT) and Neural Network Meta-VQT (NN-Meta VQT) for efficient thermal state preparation of parametrized Hamiltonians on Noisy Intermediate-Scale Quantum (NISQ) devices. Meta-VQT utilizes a fully quantum ansatz, while NN Meta-VQT integrates a quantum--classical hybrid architecture. Both use collective optimization over a finite set of training parameter values to prepare approximate Gibbs states at held-out parameter values within the sampled parameter domain. We validate our methods on Transverse Field Ising Models (TFIM) with up
    
[^318]: 关于LQG问题的梯度支配性研究

    On the Gradient Domination of the LQG Problem

    [https://arxiv.org/abs/2507.09026](https://arxiv.org/abs/2507.09026)

    本文提出一种基于过去p个时间步输入输出数据的控制输入历史表示参数化方法，将LQG问题转化为具有梯度支配性的等价静态优化问题，从而为策略梯度方法求解LQG问题建立了全局收敛保证。

    

    我们通过策略梯度（PG）方法研究线性二次高斯（LQG）调节器问题的求解。尽管PG方法在求解线性二次调节器（LQR）问题方面已经展现出强大的理论保证（尽管其优化景观是非凸的），但人们对其在LQG设定下的理论理解仍然有限。值得注意的是，LQG问题在经典参数化（即采用动态控制器）下缺乏梯度支配性，这阻碍了全局收敛保证的建立。在本工作中，我们通过采用一种对镇定控制器集合的替代参数化并运用提升论证，研究了LQG问题的PG方法。我们将这种参数化称为控制输入的历史表示，因为它由之前p个时间步的输入和输出数据进行参数化。在该参数化下，无限历史的LQG优化问题可简化为一个等价的静态（摘要在此处被截断）

    arXiv:2507.09026v2 Announce Type: replace-cross  Abstract: We consider solutions to the linear quadratic Gaussian (LQG) regulator problem via policy gradient (PG) methods. Although PG methods have demonstrated strong theoretical guarantees in solving the linear quadratic regulator (LQR) problem, despite its nonconvex landscape, their theoretical understanding in the LQG setting remains limited. Notably, the LQG problem lacks gradient dominance in the classical parameterization, i.e., with a dynamic controller, which hinders global convergence guarantees. In this work, we study a PG method for the LQG problem by adopting an alternative parameterization of the set of stabilizing controllers and employing a lifting argument. We refer to this parameterization as a history representation of the control input as it is parameterized by past input and output data from the previous p time-steps. Under this parameterization, the infinite-history LQG optimization reduces to an equivalent static s
    
[^319]: 将少样本分类视为多实例验证：跨域的高效骨干网络无关迁移

    Few-shot Classification as Multi-instance Verification: Effective Backbone-agnostic Transfer across Domains

    [https://arxiv.org/abs/2507.00401](https://arxiv.org/abs/2507.00401)

    提出将少样本分类转化为多实例验证任务的"MIV-head"分类头，在无法微调骨干网络的约束下，仅利用冻结“黑盒”骨干网络的静态嵌入即可在目标域少样本数据上训练并实现有竞争力的跨域分类性能。

    

    我们研究了在无法或不适合对骨干网络（即特征提取器）进行微调的约束下的跨域少样本学习——这种情况在实际应用场景中越来越常见——并展示了如何将少样本分类表示为一系列多实例验证（MIV）任务，从而利用冻结的“黑盒”骨干网络所产生的静态嵌入实现有竞争力的分类性能。更具体地说，基于这种新颖的表示方法，我们提出了一种名为“MIV-head”的少样本域适应新方法：一个与骨干网络无关且计算高效的分类头。我们表明，当在来自目标域的少样本数据上进行训练时，MIV-head的核心组件共同在该域的测试数据上产生了强大的性能。重要的是，它们无需对骨干网络进行微调即可实现这一点，只需要“元测试”阶段。

    arXiv:2507.00401v2 Announce Type: replace-cross  Abstract: We investigate cross-domain few-shot learning under the constraint that fine-tuning of backbones (i.e., feature extractors) is impossible or infeasible---a scenario that is increasingly common in practical use cases---and show how representing few-shot classification as a series of multiple instance verification (MIV) tasks enables competitive classification performance using the static embeddings produced by frozen, "black-box" backbones. More specifically, based on this novel representation, we introduce a new approach to few-shot domain adaptation named the "MIV-head": a classification head that is backbone-agnostic and computationally efficient. We show that the core components of the MIV-head, when trained on few-shot data from a target domain, collectively yield strong performance on test data from that domain. Importantly, they do so without fine-tuning the backbone, requiring only the "meta-testing" phase. In experiment
    
[^320]: 面向OECD/NEA人工智能/机器学习临界热流密度基准第二阶段的已发表非均匀轴向功率数据汇集

    Aggregation of Published Non-Uniform Axial Power Data for Phase II of the OECD/NEA AI/ML Critical Heat Flux Benchmark

    [https://arxiv.org/abs/2507.00034](https://arxiv.org/abs/2507.00034)

    本工作为OECD/NEA AI/ML临界热流密度基准第二阶段汇集并数字化了涵盖均匀与非均匀轴向加热条件的大规模CHF数据集，并揭示经典关联式和仅用均匀数据训练的神经网络在非均匀功率分布下均难以准确预测，凸显了开发泛化能力更强的AI/ML模型的必要性。

    

    临界热流密度（CHF）标志着轻水反应堆中沸腾危机的发生，界定了安全的热工水力运行极限。为支持OECD/NEA AI/ML CHF基准第二阶段（该阶段引入了空间变化的功率分布），本工作汇集并数字化了一个涵盖均匀与非均匀轴向加热条件的大规模CHF数据集。加热分布从技术报告中提取，插值到统一的轴向网格上，通过能量平衡检查进行验证，并以机器可读格式编码以保证与基准的兼容性。经典CHF关联式在均匀加热条件下已存在较大误差，在应用于非均匀分布时性能明显下降；现代表格方法虽有所改进，但预测仍不完美。仅用均匀数据训练的神经网络在该工况下表现良好，但无法泛化到空间变化的场景，凸显了对能够……（摘要在此处截断）

    arXiv:2507.00034v3 Announce Type: replace  Abstract: Critical heat flux (CHF) marks the onset of boiling crisis in light-water reactors, defining safe thermal-hydraulic operating limits. To support Phase II of the OECD/NEA AI/ML CHF benchmark, which introduces spatially varying power profiles, this work compiles and digitizes a broad CHF dataset covering both uniform and non-uniform axial heating conditions. Heating profiles were extracted from technical reports, interpolated onto a consistent axial mesh, validated via energy-balance checks, and encoded in machine-readable formats for benchmark compatibility. Classical CHF correlations exhibit substantial errors under uniform heating and degrade markedly when applied to non-uniform profiles, while modern tabular methods offer improved but still imperfect predictions. A neural network trained solely on uniform data performs well in that regime but fails to generalize to spatially varying scenarios, underscoring the need for models that 
    
[^321]: 基于扩散的假设检验与变点检测

    Diffusion-Based Hypothesis Testing and Change-Point Detection

    [https://arxiv.org/abs/2506.16089](https://arxiv.org/abs/2506.16089)

    本文将基于分数的假设检验与变点检测方法推广为基于扩散的版本，通过数值优化权重矩阵提升检测能力，并从理论上量化了算法性能及其达到最优的条件。

    

    基于分数的方法近年来在建模与生成领域日益流行。已有研究构建了利用分数函数进行假设检验和变点检测的方法，但这些方法通常不如基于似然的方法强大。近期的工作考虑通过将分数函数与矩阵值函数或权重矩阵相乘进行变换，将基于分数的Fisher散度推广为扩散散度。在本文中，我们将基于分数的假设检验方法和变点检测停止规则扩展为基于扩散的对应方法。此外，我们从理论上量化了这些基于扩散的算法的性能，并研究了可实现最优性能的场景。我们提出了一种数值优化权重矩阵的方法，并通过数值模拟展示了基于扩散算法的优势。

    arXiv:2506.16089v2 Announce Type: replace-cross  Abstract: Score-based methods have recently seen increasing popularity in modeling and generation. Methods have been constructed to perform hypothesis testing and change-point detection with score functions, but these methods are in general not as powerful as their likelihood-based peers. Recent works consider generalizing the score-based Fisher divergence into a diffusion-divergence by transforming score functions via multiplication with a matrix-valued function or a weight matrix. In this paper, we extend the score-based hypothesis test and change-point detection stopping rule into their diffusion-based analogs. Additionally, we theoretically quantify the performance of these diffusion-based algorithms and study scenarios where optimal performance is achievable. We propose a method of numerically optimizing the weight matrix and present numerical simulations to illustrate the advantages of diffusion-based algorithms.
    
[^322]: ShIOEnv：一个用于语法约束合成与执行行为建模的命令评估环境

    ShIOEnv: A Command Evaluation Environment for Grammar-Constrained Synthesis and Execution Behavior Modeling

    [https://arxiv.org/abs/2505.18374](https://arxiv.org/abs/2505.18374)

    ShIOEnv是一个与Gymnasium兼容的Bash命令评估环境，通过语法约束合成生成有效的shell输入，并记录完整的执行产物（标准输出、标准错误、退出状态和系统状态变化），为训练无需执行的shell模拟模型和部署网络欺骗蜜罐提供数据支持。

    

    我们介绍了ShIOEnv，一个与Gymnasium兼容的Bash环境，用于合成带有执行标注的shell输入轨迹。包含此类轨迹的数据集对于训练模型执行无需实际执行的shell模拟非常有用，这进而有助于部署用于网络欺骗的蜜罐。现有数据集（如NL2CMD及其他Bash命令语料库）不包含与命令执行相关的完整执行产物（标准输出、标准错误、退出状态以及系统状态的变化）。ShIOEnv能够合成shell输入，在受控的系统上下文中执行它们，并记录标准输出、标准错误、退出状态以及结构化的文件系统和环境补丁。无约束（随机）生成的shell输入（命令和参数）会产生大量冗余或导致执行错误的参数实例。我们引入语法约束合成方法来生成语法上有效且覆盖度高的参数……

    arXiv:2505.18374v3 Announce Type: replace  Abstract: We introduce ShIOEnv, a Gymnasium-compatible Bash environment for synthesizing execution-annotated shell input traces. Datasets containing such traces are useful for training models to perform execution-free shell simulation, which in turn is useful for deploying honeypots for cyber deception. Existing datasets, such as NL2CMD and other Bash command corpora, do not include complete execution artifacts (stdout, stderr, exit status, and changes to the system's state) associated with a command's execution. ShIOEnv synthesizes shell inputs, executes them in a controlled system context, and records stdout, stderr, exit status, and structured filesystem and environment patches. Unconstrained (random) generation of shell inputs (command and arguments) results in multiple instances of arguments that are mostly redundant or result in execution errors. We introduce grammar-constrained synthesis to generate syntactically valid arguments with hi
    
[^323]: MAVOS-DD：多语言音视频开放集深度伪造检测基准

    MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark

    [https://arxiv.org/abs/2505.11109](https://arxiv.org/abs/2505.11109)

    该论文提出了首个大规模多语言音视频开放集深度伪造检测基准MAVOS-DD，包含八种语言近350小时的真实与伪造视频，实验表明现有最先进的检测器在开放集场景下无法维持其性能水平。

    

    我们提出了首个用于多语言音视频深度伪造检测的大规模开放集基准。我们的数据集包含近350小时、涵盖八种语言的真实和伪造视频，其中59%的数据为生成数据。对于每种语言，伪造视频均由多个不同的音频和视频深度伪造生成模型生成，这些模型是通过人工视觉检查并依据其生成质量挑选的。我们对训练集、验证集和测试集的划分方式进行了组织，使得训练阶段仅能接触到所选生成模型和语言的一个子集，从而构建了多个具有挑战性的开放集评估场景。我们使用近期文献中提出的多种预训练和微调深度伪造检测器进行了实验。结果表明，目前最先进的检测器在我们的开放集场景下测试时无法保持其原有性能水平。我们以非商业用途公开发布了我们的数据和代码。

    arXiv:2505.11109v2 Announce Type: replace-cross  Abstract: We present the first large-scale open-set benchmark for multilingual audio-video deepfake detection. Our dataset comprises nearly 350 hours of real and fake videos across eight languages, with 59% of data being generated. For each language, the fake videos are generated with several distinct audio and video deepfake generation models, selected based on their quality, via visual inspection. We organize the training, validation and test splits such that only a subset of the chosen generative models and languages are available during training, thus creating several challenging open-set evaluation setups. We perform experiments with various pre-trained and fine-tuned deepfake detectors proposed in recent literature. Our results show that state-of-the-art detectors are not currently able to maintain their performance levels when tested in our open-set scenarios. We publicly release our data and code for non-commercial use at https:/
    
[^324]: 基于离线回归实现逐阶段约束上下文赌博机的无Eluder维度遗憾

    Eluder-Dimension-Free Regret for Stage-Wise Constrained Contextual Bandits via Offline Regression

    [https://arxiv.org/abs/2505.07101](https://arxiv.org/abs/2505.07101)

    提出CS-UCCB算法，通过结合离线回归、代价置信集与上反事实置信界，在逐阶段约束的上下文赌博机中首次实现了不含eluder维度项的Õ(√T)遗憾界，同时以高概率保证每一轮的代价可行性。

    

    具有逐阶段约束的上下文赌博机对一类决策问题进行建模，这类决策必须在学习其结果的同时控制风险。我们研究仅能观测到所选动作的奖励与代价的情形，其中奖励函数属于一般函数类，代价函数属于一类凸的非线性函数。我们提出了CS-UCCB算法，该算法结合了离线回归、代价置信集以及上反事实置信界。每次回归拟合都定义了一个有效的代价置信集，而由覆盖熵导出的几何结构则控制着沿采样决策的累积代价不确定性。在两个函数类均具有多对数级覆盖熵且离线估计误差为多对数级的条件下，该有限动作算法以高概率保证每一轮的可行性，并实现Õ(√T)的遗憾界，且不含eluder维度项。在显式的奖励设计条件下，我们还将分析扩展到无限动作的情形……

    arXiv:2505.07101v4 Announce Type: replace-cross  Abstract: Contextual bandits with stage-wise constraints model decisions that must control risk while learning from their outcomes. We study this problem when only the selected action's reward and cost are observed, with a general reward class and a convex class of nonlinear cost functions. We propose CS-UCCB, which combines offline regression, cost confidence sets, and upper counterfactual confidence bounds. Each regression fit defines a valid cost confidence set, while a geometry derived from covering entropy controls cumulative cost uncertainty along the sampled decisions. Under polylogarithmic covering entropy for both classes and polylogarithmic offline estimation error, the finite-action algorithm guarantees, with high probability, feasibility at every round and $\widetilde{\mathcal O}(\sqrt T)$ regret without an eluder-dimension term. We also extend the analysis to infinite actions under an explicit reward-design condition and com
    
[^325]: DetoxAI：一个用于计算机视觉中深度学习模型去偏的Python工具包

    DetoxAI: a Python Toolkit for Debiasing Deep Learning Models in Computer Vision

    [https://arxiv.org/abs/2505.05492](https://arxiv.org/abs/2505.05492)

    DetoxAI是一个开源Python工具包，通过内部表征干预等先进去偏算法、公平性指标和基于归因的可视化工具，专门用于提升深度学习计算机视觉分类器的公平性。

    

    尽管机器学习公平性近年来取得了显著进展，但大多数现有解决方案主要针对表格数据，难以适用于高度依赖深度学习的视觉分类任务。为弥合这一差距，我们推出了DetoxAI，这是一个开源Python库，旨在通过事后去偏（post-hoc debiasing）来提升深度学习视觉分类器的公平性。DetoxAI实现了最先进的去偏算法、公平性指标和可视化工具。它支持通过对内部表征进行干预来实现去偏，并包含基于归因的可视化工具和量化的算法公平性指标，以展示偏差是如何被缓解的。本文介绍了DetoxAI的动机、设计和用例，展示了其对工程师和研究人员的实际价值。

    arXiv:2505.05492v2 Announce Type: replace-cross  Abstract: While machine learning fairness has made significant progress in recent years, most existing solutions focus on tabular data and are poorly suited for vision-based classification tasks, which rely heavily on deep learning. To bridge this gap, we introduce DetoxAI, an open-source Python library for improving fairness in deep learning vision classifiers through post-hoc debiasing. DetoxAI implements state-of-the-art debiasing algorithms, fairness metrics, and visualization tools. It supports debiasing via interventions in internal representations and includes attribution-based visualization tools and quantitative algorithmic fairness metrics to show how bias is mitigated. This paper presents the motivation, design, and use cases of DetoxAI, demonstrating its tangible value to engineers and researchers.
    
[^326]: 如何对图像知识蒸馏进行后门攻击

    How to Backdoor Image Knowledge Distillation

    [https://arxiv.org/abs/2504.21323](https://arxiv.org/abs/2504.21323)

    本文提出了一种针对图像知识蒸馏的新型后门攻击方法，通过向蒸馏数据集注入被干净教师模型预测为目标标签的触发图像，使学生在教师保持干净的情况下学到后门，且在10%中毒率下仍然有效。

    

    知识蒸馏被广泛用于将大型教师模型的行为迁移到较小的学生模型。当教师模型是干净的时，知识蒸馏通常被认为是安全的，因为经典的后门攻击依赖于监督训练中的中毒标签和触发器，而蒸馏则是训练学生模型去匹配教师模型的输出。我们表明，当蒸馏数据集本身被投毒时，这一假设可能会失效。我们的攻击注入带有触发器且经过操纵的图像，这些图像会被干净的教师模型预测为攻击者选定的目标标签，从而使学生在教师模型不受影响的情况下学到后门。我们在多种操纵策略下评估了这一威胁，包括针对性的对抗扰动和基于GAN的针对性类别转换，并研究了蒸馏设置如何同时影响准确率和攻击成功率。我们表明，即使在10%的中毒率下，该攻击仍然有效。

    arXiv:2504.21323v3 Announce Type: replace-cross  Abstract: Knowledge distillation is widely used to transfer behavior from a large teacher model to a smaller student. It is often assumed to be safe when the teacher is clean, because classic backdoor attacks rely on poisoned labels and triggers in supervised training, whereas distillation trains the student to match a teacher's outputs. We show that this assumption can fail when the distillation dataset itself is poisoned. Our attack injects triggered and manipulated images that a clean teacher already predicts as an attacker chosen target label, which causes the student to learn a backdoor even though the teacher remains unaffected. We evaluate this threat across multiple manipulation strategies, including targeted adversarial perturbations and targeted GAN based class transitions, and study how distillation settings influence both accuracy and attack success. We show that the attack remains effective even at a 10% poisoning rates. The
    
[^327]: 更多的算力能否让每一种对抗攻击都具备可迁移性？

    Does More Compute Make Every Adversarial Attack Transferable?

    [https://arxiv.org/abs/2504.20310](https://arxiv.org/abs/2504.20310)

    本文证明了在攻击者与防御者之间存在二次方算力差距时，所有可学习分类任务要么存在可迁移攻击要么存在对抗防御，而生成任务在低于该差距时不满足此二分性，且推理时使用额外计算的防御方法可被证明严格优于所有发布固定模型的防御。

    

    可迁移攻击，或称可迁移越狱攻击，是一种特别强大的对抗样本形式，能够欺骗某一类中的所有防御者。在这项工作中，我们研究了在什么条件下更多的算力能够实现可迁移攻击。我们证明，当攻击者与防御者之间存在计算资源的二次方差距时，每个可学习的分类任务要么存在可迁移攻击，要么存在对抗防御。而对于生成任务，在低于二次方差距的情况下这种二分性不再成立，即攻击者拥有更多算力并不必然带来可迁移攻击。我们证明了一个蕴含上述失效性的更强结果：一种在看到输入之后才花费额外计算的防御方法，可以被证明严格强于所有发布固定模型的防御方法。这一结果为“增加推理时算力可以提升鲁棒性”提供了形式化证据，这与Zaremba等人（2025）的实验观察相一致。我们假设存在一个……

    arXiv:2504.20310v3 Announce Type: replace  Abstract: Transferable attacks, or transferable jailbreaks, are a particularly strong version of adversarial examples that fool all defenders in a class. In this work, we study when more compute enables transferable attacks. We show that, when there is a quadratic gap in computational resources between attacker and defender, every learnable classification task admits either a transferable attack or an adversarial defense. For generative tasks, this dichotomy fails below a quadratic gap, i.e., more compute for the attacker does not lead to a transferable attack.   We prove a stronger result that implies this failure: a defense that spends additional computation after it sees the input can be provably strictly stronger than every defense that publishes a fixed model. This result provides formal evidence that increasing inference-time compute can improve robustness, as empirically observed by Zaremba et al. (2025). We assume the existence of an a
    
[^328]: 分布失配下同策略策略梯度方法的分析

    Analysis of On-policy Policy Gradient Methods under the Distribution Mismatch

    [https://arxiv.org/abs/2503.22244](https://arxiv.org/abs/2503.22244)

    本文分析了策略梯度方法中采样分布与折扣状态分布失配的影响，证明了表格型参数化在有偏更新下的全局收敛性，并通过渐近误差首阶系数之比给出了分布失配何时加速或减缓收敛的明确判据。

    

    策略梯度方法是解决具有挑战性的强化学习问题最成功的方法之一。尽管在经验上取得了成功，但对于折扣回报问题，许多最先进的策略梯度算法在估计更新方向时所使用的采样分布与策略梯度定理所要求的折扣状态分布并不一致。在这项工作中，我们分析了这种分布失配对策略梯度方法的影响。具体而言，{对于表格型直接参数化和softmax参数化，我们在有偏更新下建立了目标值的全局收敛性}。{对于表格型softmax策略，我们进一步推导了在有偏更新和归一化精确更新下渐近目标误差的精确首阶系数，两者的比值提供了一个明确的判据，用于确定分布失配何时会加速或减缓渐近收敛。}随后，我们将分析扩展到更一般……

    arXiv:2503.22244v3 Announce Type: replace  Abstract: Policy gradient methods are among the most successful approaches for solving challenging reinforcement learning problems. Despite their empirical success, many state-of-the-art policy gradient algorithms for discounted problems estimate update directions using a sampling distribution that differs from the discounted state distribution required by the policy gradient theorem. In this work, we analyze the impact of this distribution mismatch on policy gradient methods. Specifically, {for tabular direct and softmax parameterizations, we establish global convergence of objective values under biased updates}. {For tabular softmax policies, we further derive the exact leading-order coefficients of the asymptotic objective errors under biased and normalized exact updates. Their ratio provides an explicit criterion for determining when distribution mismatch accelerates or slows asymptotic convergence.} We then extend the analysis to more gen
    
[^329]: 通过梯度计算的经典卸载减少变分量子算法中的QPU调用次数

    Reducing QPU Calls in Variational Quantum Algorithms via Classical Offloading of Gradient Computation

    [https://arxiv.org/abs/2503.16361](https://arxiv.org/abs/2503.16361)

    本文提出HELIA拟设及结合经典增强梯度方法与参数移位规则的混合训练方法，将梯度计算部分卸载到经典硬件，使VQE的逻辑量子评估次数平均减少高达60%，同时提升精度和成功率。

    

    变分量子算法（VQAs）是利用近期量子计算机的一种有前景的方法，但其可训练性和量子资源成本仍然是主要挑战。本文通过硬件高效且动态李代数支持的拟设（HELIA）以及两种将经典增强梯度方法与基于量子的参数移位规则（PSR）相结合的训练方法来解决这些挑战。该方法将梯度估计和训练过程分布在经典和量子硬件上。我们使用变分量子本征求解器（VQE）评估了6到18量子比特哈密顿量的基态估计问题，以及最多12量子比特哈密顿量的量子相分类问题。对于VQE，我们的方法提高了精度和成功率，与相同拟设的Full-PSR训练相比，估计的逻辑量子评估次数平均减少高达60%。对于分类任务，最大的改进……

    arXiv:2503.16361v3 Announce Type: replace-cross  Abstract: Variational quantum algorithms (VQAs) are a promising approach for leveraging near-term quantum computers, but trainability and quantum-resource costs remain major challenges. Here we address these challenges with the \textbf{H}ardware \textbf{E}fficient and dynamical \textbf{LI}e algebra supported \textbf{A}nsatz (HELIA) and two training methods combining the classical-enhanced \g method with the quantum-based Parameter-Shift Rule (PSR). This distributes gradient estimation and training across classical and quantum hardware. We evaluate ground-state estimation of 6- to 18-qubit Hamiltonians using the Variational Quantum Eigensolver (VQE) and quantum phase classification for Hamiltonians up to 12 qubits. For VQE, our methods improve accuracy and success rates, with a mean reduction in estimated logical quantum evaluations of up to $60\%$ relative to Full-PSR training of the same ansatz. For classification, the largest improveme
    
[^330]: 路径正则化：多层神经网络的近乎完备且最优的非渐近泛化理论与双重下降现象

    Path Regularization: A Near-Complete and Optimal Nonasymptotic Generalization Theory for Multilayer Neural Networks and Double Descent Phenomenon

    [https://arxiv.org/abs/2503.02129](https://arxiv.org/abs/2503.02129)

    该论文首次提出了路径正则化多层神经网络的近乎完备且最优的非渐近泛化理论，给出了显式泛化误差上界，无需损失函数有界及网络宽度、深度等常见假设，超越了偏差-方差权衡并能解释深度学习中的双重下降现象。

    

    路径正则化已被证明是训练神经网络的一种非常有效的正则化方法，与权重衰减等常见正则化方法相比，它能带来更好的泛化性能。我们针对一般学习问题，首次提出了带有路径正则化的多层神经网络的近乎完备的（将在正文中明确说明）非渐近泛化理论。特别地，该理论不要求损失函数有界，而这在现有文献中通常是必备假设。我们的理论超越了偏差-方差权衡，并与深度学习中常见的现象相吻合，因此与其他现有的非渐近泛化误差界有显著不同。更具体地说，我们为满足 $\sigma(0)=0$ 且损失函数为足够广泛的Lipschitz函数的多层神经网络提出了显式的泛化误差上界，而无需对网络的宽度、深度或其他超参数作出限制……

    arXiv:2503.02129v3 Announce Type: replace-cross  Abstract: Path regularization has shown to be a very effective regularization to train neural networks, leading to a better generalization property than common regularizations i.e. weight decay, etc. We propose a first near-complete (as will be made explicit in the main text) nonasymptotic generalization theory for multilayer neural networks with path regularizations for general learning problems. In particular, it does not require the boundedness of the loss function, as is commonly assumed in the literature. Our theory goes beyond the bias-variance tradeoff and aligns with phenomena typically encountered in deep learning. It is therefore sharply different from other existing nonasymptotic generalization error bounds. More explicitly, we propose an explicit generalization error upper bound for multilayer neural networks with $\sigma(0)=0$ and sufficiently broad Lipschitz loss functions, without requiring the width, depth, or other hyper
    
[^331]: ACTIVA：基于变分自编码器的摊销式因果效应估计

    ACTIVA: Amortized Causal Effect Estimation via Variational Autoencoders

    [https://arxiv.org/abs/2503.01290](https://arxiv.org/abs/2503.01290)

    ACTIVA是一个基于transformer的条件变分自编码器，能够仅通过观测数据在单次前向传播中摊销式地估计完整的干预分布，无需干预数据、因果图或逐任务拟合。

    

    从观测数据预测干预后分布是许多科学和决策问题的核心，但由于因果模糊性、限制性的建模假设以及缺乏跨任务的摊销能力，这一任务仍然充满挑战。我们提出了ACTIVA，一个基于transformer的条件变分自编码器，用于从观测数据和干预查询中以摊销方式估计完整的干预分布。ACTIVA通过学习条件化的潜在先验，将因果知识摊销到多样化的训练任务中，使得从训练分布中抽取的新的因果模型只需单次前向传播即可处理，无需干预数据、因果图或针对每个任务的单独拟合。在训练分布之外，仅对模拟器的参数化偏移进行了测试。我们提供了ACTIVA估计目标的总体水平刻画，表明在理想化表达能力和无限样本的假设下，AC……

    arXiv:2503.01290v4 Announce Type: replace  Abstract: Predicting post-intervention distributions from observational data is central to many scientific and decision-making problems, but remains challenging due to causal ambiguity, restrictive modeling assumptions, and the lack of amortization across tasks. We introduce ACTIVA, a transformer-based conditional variational autoencoder for amortized estimation of full interventional distributions from observational data and intervention queries. ACTIVA learns a conditional latent prior that amortizes causal knowledge across diverse training tasks, so that a new causal model drawn from the training distribution is handled in a single forward pass, without interventional data, a causal graph, or per-task fitting. Beyond the training distribution, only a parametric shift of the simulator is tested. We provide a population-level characterization of ACTIVA's estimand, showing that, under idealized expressivity and infinite samples assumptions, AC
    
[^332]: 无需非高斯性假设的多视图因果发现：可辨识性与算法

    Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms

    [https://arxiv.org/abs/2502.20115](https://arxiv.org/abs/2502.20115)

    本文提出一种多视图线性结构方程模型及相应算法，通过利用同一系统多个视图间的相关性，在不依赖非高斯性假设的情况下实现了因果发现的可辨识性，并成功应用于脑区间因果图的估计。

    

    因果发现是一个困难的问题，通常依赖于对数据生成模型的强假设，例如非高斯性。在实践中，许多现代应用提供了同一系统的多个相关视图，而这一点在因果发现领域很少被考虑。在此，我们利用这种多视图结构，在弱假设条件下实现因果发现。我们提出了一个多视图线性结构方程模型（SEM），该模型通过交替利用视图间的相关性，扩展了著名的非高斯扰动框架。我们证明了该模型在无环SEM情形下的可辨识性。随后，受单视图算法（DirectLiNGAM、PairwiseLiNGAM和ICA-LiNGAM）的启发，我们提出了几种多视图因果发现算法。新方法通过仿真实验和神经影像数据应用得到了验证，在这些应用中，它们能够估计脑区之间的因果图。

    arXiv:2502.20115v4 Announce Type: replace  Abstract: Causal discovery is a difficult problem that typically relies on strong assumptions on the data-generating model, such as non-Gaussianity. In practice, many modern applications provide multiple related views of the same system, which has rarely been considered for causal discovery. Here, we leverage this multi-view structure to achieve causal discovery with weak assumptions. We propose a multi-view linear Structural Equation Model (SEM) that extends the well-known framework of non-Gaussian disturbances by alternatively leveraging correlation over views. We prove the identifiability of the model for acyclic SEMs. Subsequently, we propose several multi-view causal discovery algorithms, inspired by single-view algorithms (DirectLiNGAM, PairwiseLiNGAM, and ICA-LiNGAM). The new methods are validated through simulations and applications on neuroimaging data, where they enable the estimation of causal graphs between brain regions.
    
[^333]: 基于脆弱性感知的分类方法：用于风险刻画与鲁棒学习

    Fragility-aware Classification for Risk Characterization and Robust Learning

    [https://arxiv.org/abs/2502.13024](https://arxiv.org/abs/2502.13024)

    本文提出了脆弱性指数（FI）这一新型性能指标，从风险规避角度量化分类器自信误判的尾部风险，并在鲁棒满意度框架下开发了可处理的代理训练方法以实现鲁棒学习。

    

    分类模型在医疗诊断、推荐系统和风险评估等数据驱动的决策应用中发挥着核心作用。传统的性能指标（如准确率和AUC）量化了分类错误和成对排序性能，但未考虑错误的严重程度，即自信误判的风险。这一局限性在安全关键和成本敏感的场景中尤为严重，因为过度自信的错误可能导致严重后果。为解决这一问题，我们提出了脆弱性指数（FI），这是一种新颖的性能指标，从风险规避的视角评估分类器，通过捕捉自信误判的尾部风险。我们在鲁棒满意度（RS）框架下构建FI，以量化分类器在分布不确定性下的敏感性。在此基础上，我们开发了一个基于可处理代理的训练框架，并建立了可证明的

    arXiv:2502.13024v3 Announce Type: replace  Abstract: Classification models play a central role in data-driven decision-making applications such as medical diagnosis, recommendation systems, and risk assessment. Traditional performance metrics, such as accuracy and AUC, quantify classification errors and pairwise ranking performance but do not account for the magnitude of errors, i.e., the risk of confident misjudgments. This limitation is particularly consequential in safety-critical and cost-sensitive settings, where overconfident errors can lead to severe outcomes. To address this issue, we propose the Fragility Index (FI), a novel performance metric that evaluates classifiers from a risk-averse perspective by capturing the tail risk of confident misjudgments. We formulate FI within a robust satisficing (RS) framework to quantify classifier sensitivity under distributional uncertainty. Building on this, we develop a tractable surrogate-based training framework and establish provable 
    
[^334]: 贝叶斯原理的Fenchel-Young推广

    The Fenchel-Young Generalization of Bayesian Principles

    [https://arxiv.org/abs/2502.10295](https://arxiv.org/abs/2502.10295)

    本文利用Fenchel-Young不等式推广了共轭贝叶斯的变分原理，建立了超越KL几何的广义证据下界与广义共轭框架，实现了支持参数依赖支撑集和可调重尾的后验推断，并给出了EM算法的FY推广形式。

    

    共轭贝叶斯在机器学习中扮演着关键角色，确保了计算上简单的后验更新。我们通过使用Fenchel-Young（FY）不等式，推广了这种计算背后的变分原理，从而定义了超越KL几何的广义证据下界，并在等号成立时刻画出新的广义后验。由此引出广义共轭的概念，其定义为在损失函数平移下保持仿射的先验-势函数族的闭包。我们证明，这样的函数族不仅保持了可加的规范更新，还能够进行具有参数依赖支撑集和可调重尾的后验推断。这种组合在标准共轭贝叶斯中是无法实现的。我们推导了期望最大化（EM）算法的新的FY推广形式，它恢复了经典EM和Hard EM以及它们的摊销版本。我们展示了若干例子，包括具有闭式有界支撑集的广义贝叶斯线性回归。

    arXiv:2502.10295v4 Announce Type: replace  Abstract: Conjugate Bayes plays a key role in machine learning, ensuring computationally simple posterior updates. We generalize the variational principle behind this computation by using the Fenchel-Young (FY) inequality, which lets us define a generalized evidence lower bound beyond the KL geometry, and at equality, characterizes new generalized posteriors. A generalized notion of conjugacy follows, defined as closure of prior-potential families that are affine under translation by the loss. We show that such families not only preserve additive canonical updating, but also posterior inference with parameter-dependent supports and tuneably heavy tails. This combination is not feasible with standard conjugate Bayes. We derive a new FY generalization of expectation maximization (EM), which recovers classical and Hard EM, and amortized versions. We show examples, including generalized Bayesian linear regression with closed-form bounded support a
    
[^335]: 风险感知的对齐必须纳入奖励不确定性

    Risk-Aware Alignment Must Incorporate Reward Uncertainty

    [https://arxiv.org/abs/2410.23726](https://arxiv.org/abs/2410.23726)

    该论文证明了在参考策略局部邻域内实现最优风险的正则化器是由奖励协方差矩阵加权的二次范数，并据此提出一种基于奖励方差估计正则化的新策略优化算法，在大语言模型等实验中展现出比现有方法更鲁棒的对齐效果。

    

    我们考虑带有噪声奖励的策略优化问题，其中策略被约束在参考策略的邻域内。我们研究策略优化过程的风险，其定义为获得比参考策略更差的策略的概率。在噪声奖励函数的理想化多元高斯模型中，我们证明了在参考策略的适当局部邻域内实现最优风险的正则化器是由奖励协方差矩阵加权的二次范数。基于这一结果，我们开发了一种新的策略优化算法，该算法通过奖励方差估计来进行正则化。在合成老虎机模型、大语言模型和奖励模型上的实验表明，该算法比现有方法能产生更鲁棒的对齐结果。

    arXiv:2410.23726v2 Announce Type: replace-cross  Abstract: We consider the problem of policy optimization with noisy rewards, where the policy is constrained to be in the neighborhood of a reference policy.We study the risk of the policy optimization process, defined to be the chance of obtaining a policy worse off than the reference policy. In an idealized multivariate Gaussian model for the noisy reward function, we show that the regularizer achieving optimal risk in a suitably local neighborhood of the reference policy is the quadratic norm weighted by the reward covariance matrix. This result is used to develop a new policy optimization algorithm based on regularization via reward variance estimates. Experiments across synthetic bandit models, large language models and reward models show that this algorithm yields more robust alignment outcomes than existing approaches.
    
[^336]: 脑电（EEG）序列是时间序列吗？基于时间序列模型与联合受试者训练的EEG分类

    Are EEG Sequences Time Series? EEG Classification with Time Series Models and Joint Subject Training

    [https://arxiv.org/abs/2404.06966](https://arxiv.org/abs/2404.06966)

    本文系统对比了EEG分类模型与通用时间序列分类模型的差异，提出受试者特定、受试者无关和受试者条件三种模型设置，并证明现成的时间序列分类模型按受试者训练后性能可接近专门的EEG分类模型。

    

    与大多数其他数据领域一样，EEG（脑电图）数据分析依赖于丰富的领域特定预处理。除了这类预处理之外，机器学习研究者希望像处理任何其他时间序列数据一样来处理这些数据。在EEG分类领域，许多已开发的模型采用了我们在时间序列分类中通常看不到的层类型和架构。此外，通常的做法是为每个受试者单独学习一个模型，而不是为所有受试者学习一个统一模型。在本文中，我们系统地研究了EEG分类模型与通用时间序列分类模型之间的差异。我们描述了三种处理来自不同受试者EEG数据的模型设置：受试者特定模型（大多数EEG文献采用的方式）、受试者无关模型和受试者条件模型。在三个数据集上的实验中，我们证明了按受试者训练的现成时间序列分类模型的性能接近于EEG分类模型。

    arXiv:2404.06966v2 Announce Type: replace  Abstract: As with most other data domains, EEG data analysis relies on rich domain-specific preprocessing. Beyond such preprocessing, machine learners would hope to deal with such data as with any other time series data. For EEG classification many models have been developed with layer types and architectures we typically do not see in time series classification. Furthermore, typically separate models for each individual subject are learned, not one model for all of them. In this paper, we systematically study the differences between EEG classification models and generic time series classification models. We describe three different model setups to deal with EEG data from different subjects, subject-specific models (most EEG literature), subject-agnostic models and subject-conditional models. In experiments on three datasets, we demonstrate that off-the-shelf time series classification models trained per subject perform close to EEG classifica
    
[^337]: 混合块模型下多层网络的最优聚类

    Optimal Clustering of Multi-layer Networks under Mixture Block Models

    [https://arxiv.org/abs/2311.15598](https://arxiv.org/abs/2311.15598)

    本文在混合多层随机块模型下建立了由Chernoff信息刻画的极小极大最优网络聚类错误率，并提出了一种结合张量初始化与似然精化的两阶段估计器，在极端稀疏条件下仍能达到该最优率，且可扩展至计数加权网络。

    

    我们研究多层网络的聚类问题，其中每个观测值都是在同一组顶点上测量的无向网络。在混合多层随机块模型（MMSBM）下，每个网络由两个随机块模型之一生成，目标是恢复潜在的网络标签。我们建立了极小极大最优的网络聚类错误率，该错误率呈指数形式，并由各分量网络边概率分布之间的Chernoff信息刻画。我们提出了一种新颖的两阶段估计器，它结合了基于张量的初始化、节点与样本分割以及基于似然的精化方法。该精化步骤利用估计得到的节点社区结构，在极端网络稀疏性的条件下也能以高概率达到极小极大最优率。我们进一步通过混合多层泊松块模型，将似然精化框架扩展到计数加权网络。

    arXiv:2311.15598v2 Announce Type: replace-cross  Abstract: We study clustering for multi-layer networks, where each observation is an undirected network measured on the same set of vertices. Under the mixture multi-layer stochastic block model (MMSBM), each network is generated from one of two stochastic block models, and the goal is to recover the latent network labels. We establish the minimax optimal network clustering error rate, which takes an exponential form and is characterized by the Chernoff information between the edge probability distributions of the component networks. We propose a novel two-stage estimator that combines tensor-based initialization, node and sample splitting, and a likelihood-based refinement. The refinement step exploits estimated node communities and achieves the minimax rate with high probability under extreme network sparsity. We further extend the likelihood-refinement framework to count-weighted networks through a mixture multi-layer Poisson block mo
    
[^338]: 用于非侵入式负荷监测的进化深度网络

    Evolutionary Deep Nets for Non-Intrusive Load Monitoring

    [https://arxiv.org/abs/2303.03538](https://arxiv.org/abs/2303.03538)

    本研究提出并评估了用于非侵入式负荷监测的自适应稀疏进化训练与热启动压缩方法，在保持准确率的同时将宏F1提升至92.40%，并使存储权重减少17.36%、第一层活跃连接减少23.93%。

    

    紧凑的电器分类器可以支持内存和计算受限的本地电力负荷监测。我们使用从家用电器记录中导出的合成聚合窗口，评估了自适应稀疏进化训练和热启动压缩方法。一项完整的卷积网络研究在四个目标和三个随机种子上训练了120个模型，且仅通过验证集进行模型选择。两项分析分别考察了在近似保持二分类准确率的情况下进行压缩，以及正类F1与稀疏性的联合提升。与新训练的固定稀疏进化基线相比，所选的压缩策略将准确率从91.53%进行了调整，同时将宏F1提升至92.40%。存储权重减少了17.36%，第一层活跃连接减少了23.93%。进一步的精炼需要额外的训练。由于仅使用了三个随机种子且源窗口独立性未经证实，无法对统计等价性或在未见房屋上的性能作出声明。

    arXiv:2303.03538v2 Announce Type: replace  Abstract: Compact appliance classifiers can support local electricity-load monitoring where memory and computation are constrained. We evaluate adaptive sparse evolutionary training and warm-start compression using synthetic aggregate windows derived from household appliance recordings. A completed convolutional network study trains 120 models over four targets and three seeds with validation-only model selection. Two analyses examine compression with approximately preserved binary accuracy and joint improvement in positive-class F1 and sparsity. Relative to a newly trained fixed sparse evolutionary baseline, the selected compressed policy changes accuracy from 91.53%, while increasing macro F1 to 92.40%. Stored weights decrease by 17.36% and active first-layer connections by 23.93%. Refinement requires additional training. Three seeds and unverified source-window independence prevent claims of statistical equivalence or unseen-house performan
    
[^339]: 论性别、进化与乘性权重更新算法

    On Sex, Evolution, and the Multiplicative Weights Update Algorithm

    [https://arxiv.org/abs/1502.05056](https://arxiv.org/abs/1502.05056)

    该论文强化了性进化理论，证明有性生殖实现乘性权重更新算法（MWUA）这一结论无需弱选择条件，也无需乘积分布假设，并指出乘积分布假设实际上与种群动态不相容。

    

    我们重新审视了Chastain等人最近提出的一个关于性在进化中作用的创新理论 [PNAS'14]。简而言之，该理论表明基因重组的进化过程实现了著名的乘性权重更新算法（MWUA）。他们证明了由有性生殖引发的种群动态可以被精确地建模为：基因在一种特定的协调博弈中采用MWUA作为其学习策略。该结果在弱选择的环境下成立，且需假设种群频率始终保持乘积分布。我们重新审视了这一理论，同时消除了弱选择的要求以及对种群分布的任何假设。去除乘积分布假设至关重要，因为正如我们所证明的，该假设与种群动态本身是不相容的。我们证明了由种群动态所诱导的边缘等位基因分布……

    arXiv:1502.05056v2 Announce Type: replace  Abstract: We consider a recent innovative theory by Chastain et al. on the role of sex in evolution [PNAS'14]. In short, the theory suggests that the evolutionary process of gene recombination implements the celebrated multiplicative weights updates algorithm (MWUA). They prove that the population dynamics induced by sexual reproduction can be precisely modeled by genes that use MWUA as their learning strategy in a particular coordination game. The result holds in the environments of \emph{weak selection}, under the assumption that the population frequencies remain a product distribution.   We revisit the theory, eliminating both the requirement of weak selection and any assumption on the distribution of the population. Removing the assumption of product distributions is crucial, since as we show, this assumption is inconsistent with the population dynamics. We show that the marginal allele distributions induced by the population dynamics prec
    

