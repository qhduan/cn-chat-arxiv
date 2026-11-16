# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pairing AIS with RF: Analysis of a Markovian Approach for Maritime Data Fusion](https://arxiv.org/abs/2611.18563) | 该论文提出一种基于配对隐马尔可夫模型（PHMM）的马尔可夫方法，将天基射频检测结果与AIS船舶轨迹进行时空对齐，从而克服AIS系统的安全漏洞与射频观测无法识别船舶的局限，实现有效的海上多源数据融合。 |
| [^2] | [Training Reasoning Models in Simulation to Natively Use Real Robots](https://arxiv.org/abs/2611.18562) | 该论文提出PEAR后训练引擎和RURL强化学习方法，仅通过大规模仿真训练（无需演示数据和外部框架），使开源27B推理模型获得可泛化到真实机器人上的原生机器人操控能力，缩小了开源模型与前沿专有模型的差距。 |
| [^3] | [FETTUCCINE: Fast and efficient brain-to-text decoding on mobile devices](https://arxiv.org/abs/2611.18561) | FETTUCCINE借助自动语音识别模型的速度与可移植性，实现了可在移动设备上运行、词错误率低于10%的端到端脑信号到文本解码框架。 |
| [^4] | [BAL: Bidirectional Autoregression in Latent Space for Learning Human Movement Representations](https://arxiv.org/abs/2611.18559) | 本文提出BAL自监督框架，利用人体日常行为的自然时间顺序作为内在监督信号，通过潜空间中的双向自回归预测学习人体运动表示，避免了以往人工假设引入的启发式超参数问题。 |
| [^5] | [WeightBench: Evaluating Language Models as Weight-Space Programmers](https://arxiv.org/abs/2611.18553) | 该论文提出了WEIGHTBENCH基准，通过五大赛道的42项权重空间编程任务系统评估语言模型直接读取、修复和编写神经网络参数的能力，发现前沿模型已能读取小型网络并编写可用的控制策略，但尚无法修补或压缩大型语言模型。 |
| [^6] | [Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond](https://arxiv.org/abs/2611.18542) | 该论文提出了一种基于复化ODE轨迹解析性质的插值调度设计新框架，通过保角映射精确求解得到由雅可比椭圆函数参数化的保方差椭圆调度族，并结合Zolotarev最优有理函数逼近构建了能根据目标分布条件数自适应优化的新采样算法Zoloflow。 |
| [^7] | [Why Finite Intelligence Works in an Infinite World: A Mathematical Theory](https://arxiv.org/abs/2611.18538) | 该论文提出“未来任务信息”理论，证明当任务请求共享集体结构时，有限的信息表示足以在无限环境中保持决策价值，并给出了有限保存的两个精确数学边界条件。 |
| [^8] | [ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL](https://arxiv.org/abs/2611.18533) | 提出 ModHarness-RL，将智能体强化学习中固定不变的手写支架重构为可从数据学习并随训练不断演化的程序，通过压缩程序（组装上下文）和引导程序（行为偏离时提供指导）两个核心组件，从轨迹中提出候选程序并经执行验证而持续进化。 |
| [^9] | [Understanding Contextual Bias in Agricultural CNNs Through Hidden-Neuron Concept Induction](https://arxiv.org/abs/2611.18530) | 该论文创新性地将概念归纳方法应用于CNN可解释性研究，通过知识图谱编码图像描述符并推导逻辑表达式，揭示了农业CNN隐藏神经元在上下文图像块上激活所对应的可解释视觉属性，从而深入理解CNN利用背景信息的上下文偏差机制。 |
| [^10] | [StreamEmbed: Real-Time Indexing from Pixels for Screenshot Retrieval](https://arxiv.org/abs/2611.18516) | StreamEmbed利用屏幕像素的冗余性，通过仅依据像素差异的空间token剪枝和跨截图时间KV缓存复用，实现增量式视觉嵌入，从而对不断到来的屏幕截图进行实时编码与索引，同时保持与全图嵌入相当的精度。 |
| [^11] | [On numerical issues of divergences between nearly equal distributions](https://arxiv.org/abs/2611.18511) | 该论文揭示了当类别分布几乎相等时，KL散度、JSD散度及熵等标准散度运算存在与softmax类似的数值精度问题，并提出了数值稳定的重新表述方案及经过属性测试验证的实现。 |
| [^12] | [Asymptotic Margins and Feature Confinement in Conic Classification Heads under Softmax Cross-Entropy](https://arxiv.org/abs/2611.18507) | 论文为锥形分类头引入了可由二阶锥规划计算的渐近间隔，并证明该间隔的正负决定了softmax交叉熵训练下特征是否有界：间隔为正时损失仅在特征发散处趋近于零，间隔为负时损失强制且最优解位于一个显式球内。 |
| [^13] | [Interactive Program Induction with Language Model Priors](https://arxiv.org/abs/2611.18485) | 本文提出以语言模型作为程序先验的贝叶斯交互式程序归纳方法，通过贪心选择最具信息量的输入向预言机查询，逐步将后验概率集中于目标程序，以化解从有限示例归纳程序时固有的歧义性问题。 |
| [^14] | [Bregman geometry in regular exponential families: dual representations and relative smoothness](https://arxiv.org/abs/2611.18480) | 本文基于射影刻画，为正则指数族在期望参数化下建立了一般性的等价对偶Bregman表示构造（含归一化与唯一性结果），使对数密度可表达为对偶坐标系中的普通Bregman散度，并将β-与α-缩放对偶表示作为该框架的特例。 |
| [^15] | [Evaluating Feedback-Driven Black-Box Processes via Stochastic Modeling and Simulation](https://arxiv.org/abs/2611.18477) | 本文提出了一个基于马尔可夫链和仿真的随机建模框架，用于统一评估反馈驱动的黑盒过程，并通过新颖的模糊测试马尔可夫链模型加以实例化，从而将实证实践与理论联系起来。 |
| [^16] | [Contrastive latent representation learning with denoising score matching](https://arxiv.org/abs/2611.18469) | 该论文提出ScoreCLR方法，通过推导香农熵梯度得到涉及潜在表示分数的等价目标，并利用去噪分数匹配技术由轻量级网络与编码器联合训练来估计排斥项，从而以新颖方式克服了对比学习中排斥力估计依赖大批量、高方差和维度灾难的问题。 |
| [^17] | [Score-based neural ordinary differential equations for reaction-diffusions](https://arxiv.org/abs/2611.18467) | 该论文提出了一种带物理权重的拉格朗日式得分驱动神经常微分方程方法，让粒子通过权重追踪质量变化，并以质量归一化的自洽残差作为训练目标，从而有效求解反应扩散方程。 |
| [^18] | [Sample-efficient prediction of high-entropy alloy strength through outcome-aware robust sequential learning](https://arxiv.org/abs/2611.18462) | 本文提出ROSE-HEA框架，通过预判实验结果对预测的更新影响并评估最坏情况下的期望预测风险来选择信息量最大的候选实验，从而以更少的拉伸试验数据高效预测高熵合金的抗拉强度。 |
| [^19] | [Callo: Serving Cache-Miss Requests with Speculative Decoding on Invalidated KV under Prefill-Decode Disaggregation](https://arxiv.org/abs/2611.18459) | Callo是一种预填充-解码分离的服务系统，通过利用失效KV缓存进行投机解码草稿生成并在预填充后用精确KV验证草稿，将缓存未命中请求的处理速度提升1.05-1.19倍。 |
| [^20] | [Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles](https://arxiv.org/abs/2611.18453) | 该论文提出了一种仅需黑盒访问的两级“集成之集成”方法，通过对固定提示词随机解码构成内层集成、对提示词进行语义扰动构成外层集成，并利用全协方差定律在嵌入空间中精确分解大语言模型的偶然性不确定性与认知性不确定性。 |
| [^21] | [PACT: A Decoder-Confidence-Guided Two-Round Receiver for Joint Channel and Phase-Noise Estimation in Sub-THz OFDM](https://arxiv.org/abs/2611.18450) | 该论文提出了一种解码器置信度引导的两轮接收机PACT，利用导频感知复数变换器联合估计信道与相位噪声，在亚太赫兹OFDM系统中将码块错误率显著降低约57%-70%。 |
| [^22] | [The State Is Not the Bottleneck: A Paged-State Gated-DeltaNet Tile on a Tri-Mode Multicast Mesh, Bit-Exact from RTL to Silicon](https://arxiv.org/abs/2611.18422) | 在Agilex 5 FPGA上实现的Gated-DeltaNet单层硬件单元表明，尽管循环状态需分页至片外存储，分页开销仅占解码总时间的11.2%，状态并非瓶颈，矩阵-向量乘法才是主要耗时部分。 |
| [^23] | [Graph diffusion transformer for interpretable generation and inverse design of disordered atomic structures](https://arxiv.org/abs/2611.18420) | 提出GaDiT图扩散Transformer生成模型，通过动态周期图、掩码条件化与区间加权噪声分布，实现无序原子结构的高质量生成、采样速度提升两个数量级，并支持统一模型下的多条件生成与逆向设计。 |
| [^24] | [CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2611.18417) | 提出CoPre-MoE方法，通过为每个任务先学习任务特定MoE专家，再对跨任务冗余专家进行整合与保留，在避免灾难性遗忘的同时有效抑制多模态持续指令微调中的跨任务参数冗余和参数线性增长。 |
| [^25] | [Fuse-PINN: Combining Finite Element Methods and Physics-Informed Neural Networks to enable massively parallel simulations of non-linear PDEs](https://arxiv.org/abs/2611.18409) | 本文提出Fuse-PINN框架，将有限元方法与物理信息神经网络相结合，实现了非线性偏微分方程在高性能计算架构上的可扩展大规模并行模拟，打破了“PINNs无法扩展”的固有观念。 |
| [^26] | [Deep Neural Networks Training Efficiency: Does Grokking Modify its Dynamics?](https://arxiv.org/abs/2611.18396) | 本文通过在模运算和MNIST任务上以多种数值精度进行长短时程训练实验，利用Orthograd诱发grokking并系统考察其与训练效率（单位能量准确率）的关系，发现grokking的发生并未显著改变训练效率通常递减的动态规律。 |
| [^27] | [Lipschitz-Constrained Discriminators in Generative Modeling: Asymptotic Behavior and Support Mismatch](https://arxiv.org/abs/2611.18394) | 本文刻画了生成建模中带 Lipschitz 约束的最优判别器在正则化参数 $L$ 趋于无穷时的渐近行为，涵盖了绝对连续且 Radon-Nikodym 导数有界、以及离散分布等多种情形。 |
| [^28] | [WeatherDreamer: Learning a World Model from Multimodal Earth Observations](https://arxiv.org/abs/2611.18393) | WeatherDreamer是一个无需再分析数据、从18个全球观测系统端到端学习递归滤波循环的世界模型，其集合预报在12小时内的温度、露点和风速预测上优于NOAA的GEFS。 |
| [^29] | [A systematic comparison of physics-informed neural networks and mechanistic models for epidemic modeling and forecasting](https://arxiv.org/abs/2611.18389) | 本研究在统一的SEIRD框架下系统比较了物理信息神经网络（PINNs）与随机机理模型在流行病建模中的表现，发现PINNs虽然在合成场景中表现欠佳，但在真实COVID-19疫情数据上预测性能更优，且能提供更窄的不确定性区间和更精确的参数估计。 |
| [^30] | [Where Tree Ensembles Decide: Structural Local Explanation with Decision Predicate Graphs](https://arxiv.org/abs/2611.18382) | 该论文提出局部决策谓词图，将树集成对单个样本的预测计算记录为可验证的执行规则集合，并从中识别出关键谓词（即一个具体的阈值及共享它的树），使用户无需外部解释器即可定位模型决策的决定性因素。 |
| [^31] | [Structure makes quantum circuits trainable](https://arxiv.org/abs/2611.18378) | 该论文提出了一个通过分解紧致李群幺正表示来界定任意输入态和可观测量下梯度方差的分析框架，据此给出判定量子电路是否具有贫瘠高原的方法，并证明了量子电路可训练性的充分条件。 |
| [^32] | [Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment](https://arxiv.org/abs/2611.18372) | 这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。 |
| [^33] | [f-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models](https://arxiv.org/abs/2611.18370) | 提出DiSPaT，一个基于f散度最小化的自我博弈微调框架，通过交替优化缩小模型生成样本与真实正常数据之间的分布差距，从而增强大语言模型对表格数据正常模式的理解，提升表格异常检测性能。 |
| [^34] | [DECODE: Decomposable Environments for COntrolled Sequential-Decision Evaluation](https://arxiv.org/abs/2611.18367) | 提出了DECODE基准，通过共享已知动力学矩阵、可在记忆与噪声参数上分解控制的消费信贷催收环境，以受控实验方法评估规划范围和信息可用性等修改对序贯决策系统性能的具体影响。 |
| [^35] | [Adaptable Watermarking: Transforming Image Watermarking Models via Lightweight Adapters](https://arxiv.org/abs/2611.18363) | 提出了一个通过轻量级参数高效适配器和多域残差对齐损失，将现有图像水印模型转换为其他水印方法的框架，无需重新训练完整模型即可实现模块化和可升级的水印架构。 |
| [^36] | [Online Switching Control as Learning with Experts](https://arxiv.org/abs/2611.18352) | 本文将离散时间非线性系统的在线切换控制问题归约为带切换代价的专家问题，并基于带切换抑制机制的Hedge算法提出了自适应切换算法，实现了 $O(\sqrt{T\log N})$ 的期望遗憾同时保证状态一致有界。 |
| [^37] | [Emulating physics-based XCO2 retrievals across OCO-2 viewing modes with compact neural networks](https://arxiv.org/abs/2611.18333) | 该研究通过奇异值分解压缩OCO-2光谱并训练紧凑多层感知器，实现快速模拟全物理XCO2反演，在2021-2025年测试中三种观测模式的年均均方根误差仅为0.89至1.43 ppm。 |
| [^38] | [Clean Tickets, Wrong Orders: Catching Silent Errors with a Language Model in Your Browser Tab](https://arxiv.org/abs/2611.18284) | 本文展示了一个完全在浏览器本地运行的小型语言模型演示系统，利用移植到TypeScript的PIZZA匹配器实时捕捉“静默错误”（即模型生成格式正确却无人真正下过的订单），其表现与服务器端评估几乎完全一致。 |
| [^39] | [Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering](https://arxiv.org/abs/2611.18274) | 该论文将基于稀疏自编码器的激活引导重构为固定预算下的逐词元分配问题，并提出无记忆的对比特征投影（CFP）与轨迹感知的闭环自适应伯努利引导（CLABS）两种门控机制，通过只在恰当的词元和时机施加干预来避免分布偏移、保持生成质量。 |
| [^40] | [ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization](https://arxiv.org/abs/2611.18252) | 提出了一种基于惩罚的极小-极大-极小优化公式（M³问题）及单循环一阶算法 ParetoM³，将偏好优化与弱帕累托最优性统一在单一目标中，并证明了算法能收敛到近似弱帕累托最优的ε-稳定点。 |
| [^41] | [Neural Classifiers as Error-Correcting Codes](https://arxiv.org/abs/2611.18239) | 该论文提出统一框架，将监督深度分类刻画为香农意义下的纠错码译码，证明交叉熵优化等价于最大似然译码，深度网络在特征空间中隐式学习码本，并据此推导出无损量化、知识蒸馏和结构正则化的新准则。 |
| [^42] | [Small Is Not Dead: Distance-Aware Cluster Revival for e-means](https://arxiv.org/abs/2611.18238) | 针对 e-means 仅依据聚类规模判定死聚类、会误删覆盖稀疏孤立点群的小聚类的问题，本文提出增加基于中心间距离的第二条件——只有当小聚类的中心靠近其他中心时才将其移除复活，从而在含稀疏孤立点群的数据上显著降低聚类误差。 |
| [^43] | [NeoResearch: Literature- and Experience-Guided Autonomous Time Series Model Discovery](https://arxiv.org/abs/2611.18235) | NeoResearch 提出了一个文献与经验引导的自主研究框架，利用大语言模型充当“研究者”，通过结合假设驱动架构探索与迭代训练及超参数优化的双循环工作流，自主挖掘网络科学文献并发现轻量级、领域定制的时间序列预测模型。 |
| [^44] | [T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer](https://arxiv.org/abs/2611.18199) | T2ST 框架通过引入丰富的时间序列数据辅助训练，并从冻结的时间序列基础模型中迁移知识，在时空观测稀缺的情况下实现数据高效的时空基础模型训练。 |
| [^45] | [MarathonTS: a benchmark of long synthetic time series from window assembly and long-series generators](https://arxiv.org/abs/2611.18190) | MarathonTS是首个在统一协议下系统比较窗口组装与长序列生成器两类方法的长合成时间序列开放基准，从保真度、实用性、记忆性和计算效率多维度全面评估序列质量。 |
| [^46] | [LaMe: a Joint Latent Mediation Framework for High-Dimensional Systems](https://arxiv.org/abs/2611.18180) | 该论文提出LaMe，一个联合线性-高斯潜在中介框架，通过少量潜变量概括高维的暴露-中介-结果系统，在明确假设下识别自然直接与间接效应，克服了现有方法过度参数化、推断有偏和可扩展性受限的问题。 |
| [^47] | [Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders](https://arxiv.org/abs/2611.18179) | 本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。 |
| [^48] | [Detection and Classification in Latent Spaces: High-Dimensional Analysis and Validation](https://arxiv.org/abs/2611.18164) | 该论文在潜在维度与样本规模成比例增长的高维状态下，对基础模型潜空间中的线性和二次检测器进行了Neyman-Pearson大偏差分析，推导出虚警与漏检错误概率的显式刻画，揭示了双下降乃至多重下降的非单调现象，并通过视觉、文本和音频数据集实验验证了理论预测与实际性能的高度一致性。 |
| [^49] | [On the Limits of Sufficient Explanations](https://arxiv.org/abs/2611.18163) | 该论文建立了一个通用的充分解释形式化框架，并证明分类器在图和字符串等结构化数据上充分解释器的存在性完全由分类器与选择关系之间的相互作用决定。 |
| [^50] | [Which Model, For Which Variable? Per-Variable Forecast Combination in Sparse ICU Time Series](https://arxiv.org/abs/2611.18149) | 研究发现ICU生理指标预测中不存在对所有变量都最优的单一模型，通过在五个ICU数据库上对94个变量逐变量组合14个模型的预测，平均可将均方误差较最佳单一模型降低8.7%，且增益集中在数据最稀疏的变量上。 |
| [^51] | [Compiling Specialized GPU-Initiated Networking](https://arxiv.org/abs/2611.18145) | Obaleth编译器将GPU发起的通信纳入内核中间表示并与计算协同优化，生成专门的传输代码，在小消息发送、大规模all-to-all通信和大模型推理服务中显著超越NVSHMEM、NCCL GIN和DeepEP等通用通信库。 |
| [^52] | [STAMP: Manufacturing-Informed Generative Design and Shape Optimization of Sheet Panels](https://arxiv.org/abs/2611.18144) | STAMP 通过将制造几何条件直接嵌入形状表示本身，实现了钣金面板的生成式设计与形状优化，使所有生成和优化的结果无论训练数据多少都能天然满足制造约束。 |
| [^53] | [When Copying Is Hard to Beat: A Negative Result in Clinical Assessment Forecasting](https://arxiv.org/abs/2611.18142) | 该研究是一项负面结果，表明在预测次日临床评估时，各类模型相比简单复制前一天答案的持续性基线并无一致优势。 |
| [^54] | [PRISM: A Post-Prediction Reliability Framework for Evidence-Guided Correction and Selective Abstention](https://arxiv.org/abs/2611.18127) | PRISM是一个预测后可靠性框架，通过估计错误风险、检索历史证据来验证潜在不可靠的预测，从而实现有证据支撑的纠正或在证据不足时弃权，以应对顺序应用中高置信度预测仍可能出错的问题。 |
| [^55] | [Efficient Poincar\'e Convolutions through Patch-Statistics Factorization](https://arxiv.org/abs/2611.18125) | 本文提出将庞加莱卷积得分分解为图像块的平方范数和与滤波器的内积两个统计量，由原生卷积直接计算，无需将图像块映射到庞加莱球即可精确求值，使训练速度提升1.2至1.8倍。 |
| [^56] | [Muon Attains $\mathcal{O}(\varepsilon^{-3})$ Iteration Complexity under Shuffling](https://arxiv.org/abs/2611.18124) | 本文首次在随机重排采样下对Muon优化器进行了收敛性分析，证明了对于光滑非凸有限和问题寻找ε-驻点具有O(ε⁻³)的迭代复杂度，且该保证对任意置换序列均成立。 |
| [^57] | [Training-point consistency of kernel ridge regression with varying correlation scales](https://arxiv.org/abs/2611.18095) | 本文建立了变化尺度参数下核岭回归的统一理论框架，证明训练点风险的一致性当且仅当有效局部样本量发散到无穷大，从而无缝统一了长程、固定尺度和短程等各种情形。 |
| [^58] | [Deep Latents, Long Memory: Latent persistence explains temporal correlations in transformer activations](https://arxiv.org/abs/2611.18089) | 该论文提出用层激活的自相关函数作为抽象的可操作度量，发现潜变量的持久编码会在自相关中产生慢衰减尾部，且这些尾部随Transformer层加深而增长，从而解释了深层模型编码抽象潜变量的机制。 |
| [^59] | [Learning nonlinear interactions and cross-feature couplings with product-unit residual networks](https://arxiv.org/abs/2611.18074) | 提出复数值乘积单元残差网络（CV-PURe），通过将乘法乘积单元与残差连接结合来显式建模跨特征耦合，在多个科学基准上预测误差显著低于MLP，并能有效恢复输入级的特征交互关系。 |
| [^60] | [Product units in gated recurrent networks enhance prediction of nuclear masses and charge radii](https://arxiv.org/abs/2611.18066) | 该论文提出将可学习的乘积单元嵌入门控循环网络并在复数域中进行计算，从而显著提升了原子核质量与电荷半径的预测精度。 |
| [^61] | [Enrichment-Oriented Virtual Screening of Perovskite Solar Cells through Reinforcement Learning](https://arxiv.org/abs/2611.18043) | 该论文提出一个以冻结TabPFN代理模型引导的强化学习框架，将钙钛矿太阳能电池配方生成建模为富集导向的虚拟筛选，使前十名候选配方的平均PCE从19.68%提升至23.74%，其中PPO和RPPO策略在前100名候选名单中获得了最高的邻域命中率。 |
| [^62] | [RL$^{\mathrm{RT}}$: On-Device Reinforcement Learning Runtime](https://arxiv.org/abs/2611.18035) | RL$^{\mathrm{RT}}$通过将执行器与学习器的耦合变为可调参数（刷新界限与松弛感知的GPU许可），在共享嵌入式GPU上实现了高效的端侧强化学习，将16.8M参数策略的截止期限错过率从85%-100%降至1%以下。 |
| [^63] | [Beyond Bot Labels: Retrospective Consequence-Grounded Ranking of Attributed Cache Harm in E-Commerce Logs](https://arxiv.org/abs/2611.18027) | 该论文提出基于后果的会话排序方法，通过按时间顺序回溯重放将缓存驱逐造成的未命中归因到具体会话，从而识别并绕过对电商缓存命中率造成跨会话危害的自动化流量。 |
| [^64] | [A Generalizable Approach to Safety-Aware Robotic Behavior Planning with an Explicit Constraint Space](https://arxiv.org/abs/2611.18026) | 该论文提出一种基于显式约束空间的安全感知机器人行为规划框架，通过任务与环境条件的适用性学习（结合轻量级双塔模型）来选择适用的安全规则并交给多模态规划器，从而提升桌面环境中机器人操作的安全性与泛化能力。 |
| [^65] | [JaRS: Contrast-Free Localization of Memorized Regions in Diffusion Models via Jacobian Row Sums](https://arxiv.org/abs/2611.18024) | 提出JaRS方法，利用去噪器雅可比矩阵的行和，无需参考图即可将扩散模型的记忆效应与局部平均区分开来，实现了无对比的记忆区域定位（JaRS+进一步结合图像级检测分数）。 |
| [^66] | [Conflict, Feasibility, Faithfulness, Displacement: Four Quantities That Come Apart in Federated Dual-Cone Aggregation](https://arxiv.org/abs/2611.18014) | 该论文通过仅改变更新方向、不改变幅度的 UPGrad 对照实验证明，联邦学习中多目标梯度聚合方法所依据的“消除客户端更新冲突”理论混淆了四个本质不同的量（冲突、可行性、忠实性、位移），实验表明消除冲突并不必然带来性能提升，均匀权重即可匹配其增益。 |
| [^67] | [Sample-Efficient Control of Magnetoconvective Flows via Differentiable Simulation](https://arxiv.org/abs/2611.18008) | 本文提出φ-PICT，一种面向主动流动控制的可微分GPU加速磁流体动力学求解器，可将传热提升25%以上，并在逆向设计中实现比黑盒方法高10³倍的精度。 |
| [^68] | [RiverScan: Learning to Accumulate on River Networks](https://arxiv.org/abs/2611.18005) | 提出 RiverScan——一种基于面积加权选择性状态空间递归的空间算子，按拓扑顺序学习河网上的信息累积，在多瑙河流域 608 个站点的径流预测中超越了时序与图基线方法。 |
| [^69] | [LACCO: Local Accumulation with Computation--Communication Overlap](https://arxiv.org/abs/2611.17991) | 提出LACCO两阶段算法，在不引入外层更新陈旧性的前提下实现计算与通信重叠并保持各副本更新一致性，在语言模型预训练中持续优于Eager和Delayed DiLoCo等通信重叠基线。 |
| [^70] | [When Does Oversampling Help? A Decision Framework for Imbalanced Data Streams Under Label Delay and Concept Drift](https://arxiv.org/abs/2611.17981) | 该论文提出了一个围绕漂移密度、少数类可学习性和延迟相对于细化窗口时长这三个维度构建的决策框架，用于预测在标签延迟与概念漂移并存的不平衡数据流中，流式过采样何时能比持续重评估带来额外价值。 |
| [^71] | [Algorithmic Opacity as a Principal-Agent Problem](https://arxiv.org/abs/2611.17980) | 该论文提出基于委托-代理理论的检验框架，主张即使存在博弈操纵和商业秘密的顾虑，算法决策者也应披露错误概况和群体间决策结果分布这两项诊断指标，以防止其利用不透明性掩盖偏见、低效或损害公共利益的策略性行为。 |
| [^72] | [Efficient and Practical Clustering of Variables in Causal Graphs](https://arxiv.org/abs/2611.17971) | 本文提出了一类称为“导管”的新型聚类，并给出检测因果图中所有导管的可靠、完备且高效算法，该类聚类在真实世界因果图中自然出现，可大幅缩减图规模并支持已知的可识别性结果。 |
| [^73] | [Analyzing Learning-Dynamics Across Loss and Prediction Choices for Diffusion Models](https://arxiv.org/abs/2611.17967) | 本文解耦了扩散模型中预测目标与损失空间的影响，证明改变损失空间恰好等价于时间步重加权，并从理论上揭示了秩-$m$ 线性瓶颈会使噪声和速度预测存在 $D-m$ 的不可约平方误差下限，而干净数据预测则可利用低维数据结构。 |
| [^74] | [DeltaVisionX: A Low-Cost Vision-Guided Automation System Using YOLOv11 and PLC-Based Actuation](https://arxiv.org/abs/2611.17959) | DeltaVisionX通过编码器同步的YOLOv11视觉检测与PLC气动驱动集成，实现了低成本、高精度（mAP 0.97）和高吞吐量（每分钟80件）的自动化物体分拣系统。 |
| [^75] | [Reasoning Models Confabulate to Defend Injected Beliefs About Their Own Requests](https://arxiv.org/abs/2611.17958) | 通过在提示文本中不可见的激活编辑注入离题概念，可以切断推理模型对自身请求的信念锚定，使其在被明确警告真相时仍会编造理由为被注入的信念辩护并据此作答。 |
| [^76] | [WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents](https://arxiv.org/abs/2611.17955) | WebArena-Pro是一个涵盖20个自托管Web应用、300个任务的异构多模态可复现基准测试，首次支持对Web智能体理解和运用音频、视频内容能力的评估。 |
| [^77] | [Domain Adaptation Based Pipeline for Character Classification and Handwritten Text Recognition](https://arxiv.org/abs/2611.17952) | 该论文提出了一种基于无监督域适应的三步流水线，通过对齐源域与目标域的特征分布并利用未标注目标数据，有效提升了标注数据有限情况下字符分类与手写文本识别的跨数据集泛化能力。 |
| [^78] | [TIDE: Reclaiming Lost Performance In On-Policy Reinforcement Learning](https://arxiv.org/abs/2611.17924) | TIDE通过动态复用空闲推演GPU进行训练以及在推理引擎之间迁移推演任务，在保持在策略训练语义的前提下，将LLM强化学习后训练的端到端性能提升高达1.36倍。 |
| [^79] | [Condensing Law of Security Information](https://arxiv.org/abs/2611.17918) | 本文指出缩放定律在安全领域失效——模型越强越会在下游安全模块察觉前“悄悄修复”异常输入，并首次将其形式化为认知能力与安全信息保持率之间的权衡，推导出三个信息论界。 |
| [^80] | [Hierarchical Bayesian physics-augmented neural networks for populations of hyperelastic laws: specimen variability and uncertainty from scarce, noisy data](https://arxiv.org/abs/2611.17892) | 提出一种分层贝叶斯物理增强神经网络（非线性混合效应模型），可从稀疏、含噪数据中联合推断软组织等材料超弹性本构模型的群体分布、各样本个体差异及其不确定性。 |
| [^81] | [Quantum Graph Neural Networks: Architectures, Trainability, and Applications](https://arxiv.org/abs/2611.17886) | 本综述提出了描述量子图神经网络的统一形式化框架，将现有文献归纳为五大架构家族，并从量子比特成本、对称性、可扩展性和证据强度等方面对它们进行系统比较。 |
| [^82] | [From Local Prediction Error to Final Outputs: A Theory of Expert Sharing in Multi-Step Reasoning](https://arxiv.org/abs/2611.17879) | 该论文建立了多步推理中专家共享的理论框架，通过权衡方差节省与偏差推导出共享何时有效的精确准则，并揭示了一个关键现象：共享专家可能降低局部预测误差却反而增大最终输出误差。 |
| [^83] | [Delayed Effects of Temporary Weight Decay in Muon Pretraining](https://arxiv.org/abs/2611.17876) | 该研究揭示了Muon预训练中临时权重衰减的延迟效应——早期衰减虽在结束时增加验证损失，但通过留下持久的权重范数赤字使相对步长保持较大，最终反而降低最终损失，且小规模模型上选出的衰减调度无需重新调参即可成功迁移到更大规模模型。 |
| [^84] | [Process Mining-Enabled Federated Anomaly Detection for the Industrial IoT](https://arxiv.org/abs/2611.17869) | 该论文提出一种基于流程挖掘的联邦异常检测方法，通过评估本地节点数据与全局参考数据的相似度来动态加权聚合各本地模型，从而在非独立同分布的工业物联网时间序列数据上实现个性化更新并降低误报率。 |
| [^85] | [Disentangling Forgetting from Utility: Lightweight LLM Unlearning via Orthogonal Null-Space Projection](https://arxiv.org/abs/2611.17864) | UUE框架通过将参数更新投影到保留知识的正交左零空间并引入具有闭式解的凸无目标目标函数，实现了在不损害模型通用能力前提下的高效轻量级LLM知识遗忘。 |
| [^86] | [Electrocardiographic Imaging via Posterior Sampling under Operator Discrepancy](https://arxiv.org/abs/2611.17851) | 提出OpDis-PS方法，通过学习并对信号依赖的算子差异进行建模，将其融入修正后的前向算子，从而在算子不精确的现实条件下实现更准确的心电成像重建。 |
| [^87] | [Multimodal Person Concept Model for Person Identification under Varying Observation Conditions](https://arxiv.org/abs/2611.17802) | 本研究提出一种基于非参数贝叶斯生成模型的多模态人物概念模型，融合CLIP和CLAP提取的视觉、声学与语言特征，在少量无标签观测下实现变化观测条件下的鲁棒人员识别。 |
| [^88] | [Federated Physics-Informed Thermal Field Prediction Across Unseen Geometries in Distributed Additive Manufacturing](https://arxiv.org/abs/2611.17794) | 该论文提出Fed-PIGANO联邦学习框架，使各制造站点能够在不共享专有零件几何数据的前提下联合训练物理信息几何感知神经算子模型，从而在分布式增材制造中实现对未见几何形状的热场预测。 |
| [^89] | [AutoRL: A Multi-Agent System towards Autonomous Reinforcement Learning](https://arxiv.org/abs/2611.17790) | AutoRL提出了一个由感知器等多个强化学习专属智能体角色组成的多智能体系统，通过持续监控演变的训练状态并将过程级监督与研究融入正在进行的RL训练，实现强化学习训练的自动化，为未来的递归自我改进（RSI）奠定基石。 |
| [^90] | [LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent](https://arxiv.org/abs/2611.17780) | 提出非自回归 7B 模型 LuthorSQL，作为 LLM 智能体的快速“系统 1”门控，在单 GPU 上每题 0.84 秒即达到 BIRD 测试集 68.03% 的执行准确率，仅在不确定时才将查询升级给较慢的智能体，从而大幅降低成本并保护数据隐私。 |
| [^91] | [Beyond Prediction Gain: LASR as a Metric for Action-Grounded Latents in World Models](https://arxiv.org/abs/2611.17775) | 提出无量纲离线评估指标LASR（潜在动作支持率），用以判断潜动作模型学到的潜变量是否真正编码了动作信息，而非仅仅通过编码当前状态、回合进度等预测捷径来提升预测效果。 |
| [^92] | [Activation Steering the Nan-Shan in Typhoon](https://arxiv.org/abs/2611.17768) | 本文介绍了基于康拉德《台风》的视听装置作品，通过让大语言模型以自身输出为食并采用一种新颖的自适应激活引导方法，探索AI生成文本随时间推移变得日益陌生和失控的过程，以此反思人类文化当下的人工智能时刻。 |
| [^93] | [Learning Anisotropic Prediction Targets for Flow Models](https://arxiv.org/abs/2611.17767) | 该论文提出各向异性流模型，将固定的预测目标替换为完全可学习的目标，通过对噪声分量进行各向异性缩放并利用闭式映射精确恢复速度，在不改变网络架构、损失函数和采样器的情况下优化训练与生成质量。 |
| [^94] | [Efficient Genotype Classification from 3D MRI in ADPKD via Density-Aware Point Sampling and E(3)-invariance](https://arxiv.org/abs/2611.17758) | 提出E3PN网络，利用密度感知点采样和E(3)-不变性，从三维腹部MRI中高效分类ADPKD基因型，解决了图像过大与标注数据不足两大难题。 |
| [^95] | [Complex Schr\"odinger Bridges](https://arxiv.org/abs/2611.17746) | 该论文提出了共形最优传输（COT）框架与复值薛定谔桥（CSBs）这一新型动态扩散桥方法，通过共形几何先验有效处理复向量空间中受缩放和旋转定律支配的物理系统的扩散任务。 |
| [^96] | [The Polyak-{\L}ojasiewicz Paper: A 10-Year Retrospective](https://arxiv.org/abs/2611.17745) | 本文是对引入Polyak-Łojasiewicz (PL)条件术语的论文的十年回顾，作者回顾了论文的起源、给出了AI辅助精炼的主定理版本，并讨论了对该工作的批评及相关扩展。 |
| [^97] | [Logically Consistent Accurate Forecasts](https://arxiv.org/abs/2611.17735) | 该论文提出微调一个监督语言模型，利用支撑每个预测的推理过程来协调相关预测之间的逻辑不一致性，从而在ForecastBench上同时实现预测的逻辑一致性与准确性的提升。 |
| [^98] | [RaFL: An Empirical Study of Random Filter Losses for Image Reconstruction](https://arxiv.org/abs/2611.17707) | 提出了一种无需预训练特征和人工调参的随机滤波器损失，通过随机卷积、随机傅里叶特征和随机小波滤波器结合梯度范数匹配自动平衡各滤波器的贡献，有效提升图像重建中精细纹理的保留效果。 |
| [^99] | [Representational Geometry as a Diagnostic for Adversarial Robustness](https://arxiv.org/abs/2611.17684) | 该论文提出将表征几何分析（包括M-PHATE训练过程可视化、干净与扰动样本位移、最大均值差异及最近邻恢复等指标）作为诊断神经网络对抗鲁棒性的新工具，超越了传统仅依赖攻击下准确率的评估方式。 |
| [^100] | [Pattern-Based Matrix Analysis Under Permutations](https://arxiv.org/abs/2611.17674) | 本文提出模式距离及基于线性指派子问题的交替算法，寻找最优行列置换使矩阵集中于指定几何模式（如带状、块对角等），并结合模式加低秩分解，为潜在排序、社区检测和带宽最小化等结构问题提供统一框架。 |
| [^101] | [Provable Cross-Modal Transfer of Hierarchical Concept Structure in CLIP Embeddings](https://arxiv.org/abs/2611.17671) | 本文首次从理论上刻画了CLIP嵌入中层级概念结构的跨模态迁移现象，提出层级角分离（HAS）的几何性质作为形式化定义，并证明了图像嵌入天然具有的层级结构可在对比学习中被迁移至文本嵌入。 |
| [^102] | [Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention](https://arxiv.org/abs/2611.17662) | 提出了LeBet统计学习框架，通过学习可解释的“或-与”规则监控器并结合固定下注统计量积累证据，在达到阈值时对自主智能体运行进行干预，并提供有限样本下的假阳性控制和检测功效保证。 |
| [^103] | [Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost](https://arxiv.org/abs/2611.17658) | 该论文揭示了模仿策略会将机器人自身状态作为“本体感知锚点”来折扣含噪感知读数，其权重服从 α=V/(V+s²) 的预测定律，且锚点方差远小于线性模型暗示值，一次测量即可精确预测策略在不同任务和噪声水平下对感知噪声的敏感度。 |
| [^104] | [Classifier-Guided Latent Diffusion for Small-Molecule Generation](https://arxiv.org/abs/2611.17650) | 该论文提出在ChemBERTa潜在空间中构建去噪扩散模型，并利用可微分的性质预测器引导扩散采样过程，实现性质导向的小分子生成，生成的分子通过有效性、唯一性、新颖性和性质分数进行评估。 |
| [^105] | [Binarization as an Attack Surface in Weightless Neural Networks](https://arxiv.org/abs/2611.17646) | 该研究揭示了无权重神经网络（WiSARD及其布隆过滤器变体）虽然看似对对抗样本天然鲁棒，但其二值化阈值本身可被利用为攻击面，并通过阈值跨越界限和响应边际证书对这一攻击机制进行了形式化证明。 |
| [^106] | [A mesh-free deep energy method for phase-field modeling of thermally driven brittle fracture](https://arxiv.org/abs/2611.17630) | 提出了一种基于多分辨率神经表示的无网格深度能量方法，无需预先放置缺陷即可在瞬态热载荷下预测脆性断裂的自发裂纹成核、裂纹间距以及选择性的裂纹扩展与止裂。 |
| [^107] | [Joint World Action Models through Differentiable Trajectory Optimization](https://arxiv.org/abs/2611.17626) | WATO 通过可微分轨迹优化联合优化生成的动作与未来状态，使世界-动作模型的轨迹既具备任务有效性又符合动力学一致性，且部署时无需迭代轨迹优化。 |
| [^108] | [Finite speed Ornstein-Uhlenbeck-Kac processes and their applications in generative modeling](https://arxiv.org/abs/2611.17625) | 本文提出有限速度的Ornstein-Uhlenbeck-Kac（OU-K）过程，利用勒让德多项式显式求解Euler-Poisson-Darboux方程，并借助多项式分布公式刻画其条件速度场，从而成功将该过程应用于生成流模型。 |
| [^109] | [AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs](https://arxiv.org/abs/2611.17621) | 提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。 |
| [^110] | [LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters](https://arxiv.org/abs/2611.17616) | 该论文提出让大语言模型（LLM）基于无阈值的聚类整体状态视图，就地决策行为聚类的生命周期操作（分裂、合并、退役、创建），从而替代传统上依赖硬编码阈值或重聚类规则的维护范式。 |
| [^111] | [More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs](https://arxiv.org/abs/2611.17610) | 本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。 |
| [^112] | [Neural Simulation-Based Inference for Computational Cognitive Models: A Tutorial](https://arxiv.org/abs/2611.17608) | 这篇教程介绍了神经仿真推断（neural SBI）如何通过在模型仿真数据上训练深度神经网络，克服计算认知模型中似然函数不可计算的问题，从而突破传统拟合方法的局限、拓展可研究的认知理论范围。 |
| [^113] | [Choosing Quantized Code LLMs by File Size and Energy per Solved Task](https://arxiv.org/abs/2611.17605) | 本研究在 MacBook Air M2 上以“每解决任务的能耗”评测 Qwen2.5-Coder 的 24 个量化版本，给出按文件大小索引、附证据标签的模型选择指南，并揭示 3 GB 档位的 7B-q2_K 相比 2 GB 档位的 3B-q4_K_M 无显著性能提升却耗能翻倍。 |
| [^114] | [LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning](https://arxiv.org/abs/2611.17600) | LightTAR是一种简单低成本的降维增强方法，它利用正则化PLS从编码器最后一层和中间层提取目标感知特征来补充标准PCA表示，从而在21种表格模型上普遍提升多模态表格学习的预测性能。 |
| [^115] | [Tractable Importance Sampling for Monte Carlo Privacy Accounting](https://arxiv.org/abs/2611.17596) | 该论文提出了一种基于平移重要性采样的高效蒙特卡洛隐私核算方法，利用拉伸因子 α 参数化的提议分布及闭式界 B(α) < 1，大幅降低了估计严格隐私保证下极小 δ 值时的方差与样本复杂度。 |
| [^116] | [Fair and Efficient Distributed EV Charging Coordination Using Threshold Cryptography and Machine Learning](https://arxiv.org/abs/2611.17577) | 提出一种结合门限密码学与机器学习的框架，能在保持线性通信成本且无误报的前提下检测分布式电动汽车充电协调中的拜占庭攻击，并利用车队感知的机器学习价格预测帮助受攻击的电动汽车实现本地恢复。 |
| [^117] | [Beyond a Single Sign: BiTE for Branch-Preserving Binary Threshold Expansion](https://arxiv.org/abs/2611.17576) | BiTE通过将单一符号激活替换为K个有序阈值比较并保留分支响应，使二值神经网络能够区分多达K+1个有序激活状态，从而提升表示能力与准确率。 |
| [^118] | [Performance-Portable Exact Gaussian Process Regression in Python via SYCL and oneMath](https://arxiv.org/abs/2611.17565) | 通过为GPRat高斯过程回归库新增基于oneMath的混合HPX/SYCL后端，单一代码库无需厂商专用代码即可在NVIDIA GPU上媲美原生CUDA性能，并在AMD GPU上最高以4.62倍的速度超越GPflow和GPyTorch等成熟框架，实现了真正的性能可移植性。 |
| [^119] | [Benchmarks Don't Talk Back: Learning Through Self-Disagreement](https://arxiv.org/abs/2611.17563) | 该论文提出“辩证能力”这一概念——即知道何时坚持立场、何时向更优答案让步的能力，证明其无法通过准确率训练或推理时的自我反思获得，而应借鉴哲学辩证法思想，通过模型自身生成的对立观点进行训练。 |
| [^120] | [Contrastive local learning in large nonlinear analog circuits](https://arxiv.org/abs/2611.17561) | 本文证明了基于对比局部规则训练的非线性模拟电路可以扩展到超过一万个自适应边的规模并学习更大型的机器学习任务，同时在硬件非理想条件下保持稳健性。 |
| [^121] | [Generative Design with Real-time Physics Explainability: Towards Product Definition in the Age of AI](https://arxiv.org/abs/2611.17560) | 提出将基于可逆神经网络的逆向设计与实时生成式物理仿真相结合的统一生成式设计框架，借助不确定性量化和经典工具验证来确保设计的物理可行性与可解释性。 |
| [^122] | [A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes](https://arxiv.org/abs/2611.17556) | 该论文提出了一个用于工业过程目标中心因果发现的合成基准，其生成器是受批量生产启发的两层结构因果模型，由可独立设置的组件构成，能够逐一调节识别目标原因的难度，从而系统评估各类因果发现方法。 |
| [^123] | [Uncertainty-Aware Continual Learning for 2D XRD Interpretation Using Simulated and Experimental Data](https://arxiv.org/abs/2611.17547) | 该论文提出了一个不确定性感知的持续学习概念验证框架，通过融合基于MMD的领域自适应、CNN-SNGP不确定性估计以及结合知识蒸馏的重放机制，实现二维XRD数据在模拟与实验域之间稳健且可持续更新的解释。 |
| [^124] | [Learning Latent Goal Costs for Visual World-Model Planning with an Expanding Curriculum](https://arxiv.org/abs/2611.17529) | 提出在冻结 I-JEPA 嵌入上拟合小型残差适配器，通过扩展课程采集的成功与失败轨迹中观测结果的顺序信息自动学习潜在目标代价，无需任务特定的进展标签，从而有效改进冻结视觉世界模型的图像目标规划效果。 |
| [^125] | [User-Schedulable Approximations for Quantized Tensor Programs](https://arxiv.org/abs/2611.17526) | 该论文将Halide算法与调度分离的思想扩展到数值表示层面，提出由量化器、整数表示、位打包器等可复用单元组合而成的编解码器来表达量化近似，并通过新的approximate_by指令使编码器和解码器成为可被调度的普通阶段，从而大幅减少量化张量程序所需手工维护的内核数量。 |
| [^126] | [Model Annealing: Data-Efficient Learning through Review and Consolidation](https://arxiv.org/abs/2611.17524) | 模型退火通过将训练划分为学习率升降循环的多个回合、维护过去模型状态的预测记忆库，并从对抗性分歧和记忆预测中学习，实现了对网络自身知识的复习与整合，在数据受限场景下显著提升分类准确率且不增加推理开销。 |
| [^127] | [Multifidelity Operator Inference for Dynamical Systems](https://arxiv.org/abs/2611.17503) | 本文提出MF-OpInf方法，通过受控制变量法启发的多保真度最小二乘公式，将有限的高保真度快照与丰富的低保真度快照相结合，为非线性动力系统学习更准确稳定的非侵入式降阶模型。 |
| [^128] | [Continuous Cuffless Blood Pressure Estimation Using PPG From Methods to Clinical Translation](https://arxiv.org/abs/2611.17499) | 本综述系统回顾了基于PPG的无袖带血压估计的完整技术链条，从生理机制、波形特征、PAT/PTT、机器学习到临床转化，并强调应以证据强度而非仅以报告误差来评估研究。 |
| [^129] | [An Operationalization of Opaque Serial Depth](https://arxiv.org/abs/2611.17495) | 本文将“不透明串行深度”操作化为NLS深度指标，通过定义“自然语言根植节点”作为可解释瓶颈的标准，为衡量模型的非言语化串行认知能力提供了可在训练前判定架构的依据，以应对架构变化可能削弱思维链可监控性的风险。 |
| [^130] | [PF-Exec: Large-Scale UAV Relay Load Balancing with an Anchored Potential Direction Decider and a Per-Dimension Gain-Learning Executor](https://arxiv.org/abs/2611.17492) | 提出两层决策框架 PF-Exec：锚定势能方向决策器每时刻输出闭式目标方向，逐维度增益学习执行器将局部观测映射为各维度增益以自适应调节动作幅度，从而在大规模无人机中继网络中实现关联、带宽、功率与位置的联合负载均衡。 |
| [^131] | [Convergence Properties of PF-Exec: Homogeneous Structure, Anchored Closed Form, and Gain Dynamics](https://arxiv.org/abs/2611.17488) | 本文为 PF-Exec 算法提供了完整的分层数学刻画：由齐次分解闭式导出带宽目标与可行性界、证明对数拐点最优性与 Jain 公平指数一致、给出单纯形乘积上阻尼 softmax 的不动点存在性，并完整刻画了增益迭代在临界增益前几何收敛、临界处有界振荡、临界后出现有界双状态周期循环的行为。 |
| [^132] | [Creating Steering Vectors via Input Optimisation](https://arxiv.org/abs/2611.17481) | 本文提出ActGCG方法，通过优化输入后缀来模拟激活引导向量的效果，使攻击者无需白盒访问模型内部状态，即可通过修改输入文本实现与激活引导相同的行为操控，包括语言切换和越狱攻击。 |
| [^133] | [ERGo: Energy-Efficient Hybrid Graph Neural Network Training on Processing-In-Memory Architectures](https://arxiv.org/abs/2611.17478) | 提出混合训练框架ERGo，在训练过程中自动从全参数阶段切换到基于低秩表示的参数高效阶段，从而减少PIM架构上的片外内存访问，显著提升GNN训练的性能与能效。 |
| [^134] | [Do Mixture-of-Experts Routers Benefit from Ensembling?](https://arxiv.org/abs/2611.17472) | 提出在混合专家模型的路由步骤中引入路由器集成方法，实验证明更多路由器可降低验证损失，且显式训练路由器最大化专家多样性能提升平均性能。 |
| [^135] | [Improved Regret via Variance-Aware Policy Optimization in Adversarial MDPs](https://arxiv.org/abs/2611.17469) | 本文提出方差感知策略优化算法VOPO，将未知动力学对抗性MDP的遗憾值首项改进至 $\widetilde O(\sqrt{H^3SAK})$，并通过从已知到未知动力学的归约将剩余差距缩小至仅 $\sqrt H$ 因子。 |
| [^136] | [A thorough review on Deep Learning approaches to 2D-to-3D Registration in Medical Imaging](https://arxiv.org/abs/2611.17468) | 本综述全面梳理了医学图像2D到3D配准的深度学习方法，从流程设计、网络架构创新和领域自适应等多个角度分析了最新进展，凸显了深度学习相比传统优化方法在配准鲁棒性和实时性方面的优势。 |
| [^137] | [Availability-Aware Historical Feedback Learning for Electric Arc Furnace Temperature and Chemical Composition Prediction](https://arxiv.org/abs/2611.17463) | 该研究提出了一种可用性感知的历史反馈学习框架，在考虑60分钟测量延迟的约束下，将多源工业数据整合为逐步丰富的特征表示，显著提升了电弧炉终点温度和十二种化学元素最终浓度的预测性能。 |
| [^138] | [Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models](https://arxiv.org/abs/2611.17462) | 提出扩散树采样（DTS）方法，通过在去噪轨迹上进行蒙特卡洛树搜索，在推理阶段引导晶体扩散生成模型，既能直接复用现有性质条件模型而不扭曲目标分布，又支持多目标采样以高效生成满足多个冲突性质的晶体候选。 |
| [^139] | [Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments](https://arxiv.org/abs/2611.17460) | Unfinished-Bench是一个在真实工作环境中评估AI智能体持久性与规则遵守之间冲突的基准测试，发现任务持久性会显著提高智能体违反内部文档中规则的比例（从32.2%升至43.6%），而违反用户明确陈述的同一规则的比例仅为8.5%。 |
| [^140] | [Classification with Large-Step Gradient Descent: Unifying Self-Stabilization and Convergence Rates](https://arxiv.org/abs/2611.17459) | 本文将统一的几何视角扩展到指数尾损失（如逻辑损失和交叉熵损失），首次在分类设定下将大步长梯度下降的自稳定机制与收敛速率分析统一起来。 |
| [^141] | [Racing the Clock: Teaching VLNs Budget-Conditioned Object Search](https://arxiv.org/abs/2611.17442) | 该论文提出“预算条件下的目标搜索”新任务，使视觉语言导航智能体能够根据明确的动作预算自适应调整搜索策略，并对比了基于特权信息的TSP专家与基于部分可观测和跨场景语义先验的房间级Bellman专家两种监督方案。 |
| [^142] | [Interpreting Coding Agents Through Their Behavioral Trajectories](https://arxiv.org/abs/2611.17440) | 该论文提出一种基于规则的解析器，将编码智能体的原始执行轨迹映射为高级行为，并通过对14万余条轨迹的行为画像分析，揭示解决率相近的智能体在探索、修改和验证代码的方式上存在显著差异，且这些行为画像本身足以在不同仓库与语言间识别出对应的智能体技术栈。 |
| [^143] | [WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search](https://arxiv.org/abs/2611.17412) | 提出WETW学习型距离，利用小型对称网络为每对序列的每个嵌入维度预测权重，实现针对特定序列对的自适应比较且无需计算对齐，将误差相比最佳静态距离减半，从而实现高效的DTW相似性搜索。 |
| [^144] | [Can LLMs Write Backdoors? An Empirical Security Study of Weight-Space Programming](https://arxiv.org/abs/2611.17395) | 本文通过实证安全研究首次系统评估了大语言模型在权重空间编程中的后门注入能力，表明通用大语言模型可以在不访问任务数据、不执行前向传播、不计算梯度的情况下，仅凭自然语言指令直接修改模型权重来注入后门，揭示了权重空间编程带来的新型安全威胁。 |
| [^145] | [Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms](https://arxiv.org/abs/2611.17390) | 提出枚举后验支撑先验框架，基于 Quot->MorTop->Set 三层范畴链显式记录底层谓词支撑、可逆性与误差所在层级，解决标准测度论 a.e. 约定下零测度支撑隐藏域外误差无法被显式计入的问题。 |
| [^146] | [Waveform-based pulse timing enhanced by self-supervised neural regression models for digital nuclear instrumentation](https://arxiv.org/abs/2611.17356) | 本文提出一种基于匹配波形对的、结合自监督损失与辅助弱监督损失的神经回归新范式，显著提升了数字核仪器中波形脉冲定时的精度。 |
| [^147] | [Hybrid Quantum-Classical Sequence Models for Sample-Efficient Energy Regression in Simulated Calorimeter Showers](https://arxiv.org/abs/2611.17354) | 该论文在CaloGAN基准数据集上系统比较了八种经典与量子增强架构在正电子能量重建任务中的表现，发现基于纵向簇射序列的混合量子-经典模型结合物理信息驱动的数据表示达到了最高精度，证明了数据表示选择对能量回归的关键作用。 |
| [^148] | [Unsupervised Anomaly Detection in Measured Neutron Flux Profiles of the SAFARI-1 Research Reactor](https://arxiv.org/abs/2611.17351) | 该论文提出了一种基于无监督学习的自动化异常检测方法，通过高阶多项式拟合提取轴向注量率剖面的形状特征并结合密度聚类，有效识别SAFARI-1研究反应堆中因铜丝插入不当导致的异常中子注量率测量数据。 |
| [^149] | [Curvature-Aware Edge Dynamic GNNs for Low-$p_T$ Track Reconstruction at HL-LHC](https://arxiv.org/abs/2611.17343) | 提出曲率感知的流形螺旋交互网络（Manifold Helical-IN），通过将探测器空间点嵌入螺旋流形来“展开”低动量粒子的卷曲轨迹，在TrackML数据集的 $p_T > 0.5$ GeV 关键区域实现了0.9675的高径迹重建效率。 |
| [^150] | [ArchSpace: Recording, Reviewing, and Referencing LLM Architecture Research](https://arxiv.org/abs/2611.17332) | ArchSpace 是一个公共平台，通过记录实验与推理、社区评审和后续引用来系统化大语言模型架构研究，并借助缩放定律在昂贵的规模化训练前预测模型表现，从而同时弥补工业界缺乏创新公开记录和学术界评估受限的两大缺口。 |
| [^151] | [A Convex-Analytic Reformulation for Poisoning Attacks on Logistic Regression](https://arxiv.org/abs/2611.17143) | 本文将针对带 ℓ2 正则化逻辑回归的投毒攻击问题重构为一种凸分析形式，使有界双线性等式成为唯一的非凸约束，并为所有变量推导出有限界。 |
| [^152] | [LazyWeights: Decode-on-Demand Edge LLMs](https://arxiv.org/abs/2611.17108) | LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。 |
| [^153] | [An Invariant Neural-Network Eddy-Viscosity Closure with Iterative Solver Coupling for Molten Salt Fast Reactor Thermal-Hydraulics](https://arxiv.org/abs/2611.17067) | 本文为SAM系统热工水力程序开发了一种对平移、旋转、反射等变换严格不变的神经网络涡粘性闭合模型，无需求解湍流输运方程即可预测非负涡粘性并保持动量方程良好条件性，将其应用于熔盐快堆以捕捉传统混合长度模型无法表征的非平衡湍流与大规模回流。 |
| [^154] | [Time Flies When You Parallelise: Pareto-efficient Time Series Foundation Models with Linear Recurrences](https://arxiv.org/abs/2611.16934) | 提出基于 Mamba-3 线性递归的时间序列基础模型家族 ci-proto，实现训练时的时间并行与线性推理成本，在 fev-bench 100 项预测任务上创下新的最先进纪录，并凭借五个变体覆盖完整的精度-效率帕累托前沿。 |
| [^155] | [Detection Is Not Early Warning: Evaluating Token-Level Alarms on Language Model Failures](https://arxiv.org/abs/2611.16803) | 该论文提出了PREWARN评估协议，将监控器的评估从单纯检测失效转向在失效token起点之前发出预警，从而为停止或干预生成过程争取时间。 |
| [^156] | [Benchmarking AI risk behaviors](https://arxiv.org/abs/2611.16318) | 该论文提出了一个评估LLM风险倾向的工具包，发现2026年最先进模型因更倾向于采用期望值最大化作为决策程序，而比旧模型更稳定、更一致且更趋于风险中性。 |
| [^157] | [Rectified Looped Transformer](https://arxiv.org/abs/2611.16265) | 提出整流循环Transformer（RLT），通过受流模型启发的插值调度重新设计循环间反馈，在理论上保证隐藏状态幅度有界并降低扰动放大，从而提升参数高效的语言建模能力。 |
| [^158] | [JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents](https://arxiv.org/abs/2611.16174) | 提出JudgeMix框架，利用裁判模型对每一步进行语义评估，动态决定目标模型与专家模型的交替介入时机，从而生成与目标模型自身分布匹配、且不覆盖其已正确动作的高质量SFT训练数据。 |
| [^159] | [CastWorld: Benchmarking Context-Aware Agentic Time Series Forecasting Beyond Predictive Accuracy](https://arxiv.org/abs/2611.15262) | CastWorld是一个面向上下文感知智能体时间序列预测的基准测试，提供了涵盖8个领域25个真实数据集的丰富上下文数据套件，以及包含49个核心工具和状态自适应循环的预测原生框架，实现了对预测智能体超越单纯预测精度的系统化评估。 |
| [^160] | [Stealthy, High-Impact False Data Injection Attacks in Smart Grid Using Physics-Constrained GAN with Limited System Knowledge](https://arxiv.org/abs/2611.15162) | 该论文提出一种仅需有限系统知识、与检测器无关的物理约束生成对抗网络（PCGAN），通过从历史量测中学习解耦的潮流灵敏度，生成兼顾隐蔽性与高影响的智能电网虚假数据注入攻击。 |
| [^161] | [Profile-Based Auditing of Domestic Violence Survey Selection Across South Asia](https://arxiv.org/abs/2611.15106) | 本研究利用机器学习模型对南亚五国近80万女性的特征画像进行分析，发现人口与社会经济特征能够高度预测家庭暴力调查模块的样本选择，从而为审计DHS家暴数据的选择偏差和代表性提供了画像审计方法。 |
| [^162] | [Conformal Risk Control for QoT-Aware Lightpath Feasibility Under Technology-Driven Domain Shift](https://arxiv.org/abs/2611.13893) | 本文提出一种事后保形风险控制框架，在保持预训练AI/ML回归模型不变的情况下，为技术升级引起域偏移时光网络QoT感知光路可行性评估的不确定性区间提供有限样本统计保证，并通过策略规则将预测区间映射为运行决策。 |
| [^163] | [ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding](https://arxiv.org/abs/2611.13442) | 该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。 |
| [^164] | [Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA](https://arxiv.org/abs/2611.13331) | 该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。 |
| [^165] | [GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement](https://arxiv.org/abs/2611.13243) | 该论文发布了开源 SDK GigaEdge Spench，通过共享 C++ 核心与多平台集成（服务器、移动端、浏览器）实现实用的高速流式语音增强，其模型质量和处理速度优于 RNNoise 和 DPDFNet2 且快于实时。 |
| [^166] | [Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood](https://arxiv.org/abs/2611.13213) | 该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。 |
| [^167] | [Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks](https://arxiv.org/abs/2611.12799) | 该论文提出Agentic-OPD方法，将GRPO训练中被丢弃的检查点作为免费的教师池，通过有选择性的在线策略蒸馏来恢复最佳验证检查点所遗忘的智能体问题求解能力。 |
| [^168] | [Trajectory Density Ratios under Weak Support: A Survey across the Sciences](https://arxiv.org/abs/2611.11985) | 本文是一篇跨学科综述，系统梳理了轨迹密度比这一统一量在强化学习、分子模拟、气候归因和语言模型后训练等领域的核心作用，并分析了其在支撑薄弱（弱支持）情形下的存在性条件、方差爆炸与无定义等根本性失效问题。 |
| [^169] | [on\'ona: Twin Embeddings with Tied Unembeddings](https://arxiv.org/abs/2611.10556) | on'ona 提出了一种双嵌入表参数化方法，通过仅依赖当前输入 token 的门控来混合两个共享嵌入表定义的 softmax 分布，在保留权重绑定省参数优势的同时增强了输出头表达能力，在多种架构和三个规模下显著降低验证损失并提升下游任务准确率。 |
| [^170] | [LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages](https://arxiv.org/abs/2611.10357) | 提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。 |
| [^171] | [How Much Is Left Matters: False Positives and False Negatives in Partial-Prompt Contamination Tests](https://arxiv.org/abs/2611.10307) | 该论文审计了部分提示污染测试，发现剩余文本越短假阳性越多（短剩余文本贡献了大部分 ROUGE 标记，在真阴性题目上 80% 前缀时误标率高达 8.6%），且该测试无法检测通过蒸馏传播的上游污染。 |
| [^172] | [PINR-ESI: Physics-Informed Self-Supervised Implicit Neural Representation Framework for EEG Extended-Source Imaging](https://arxiv.org/abs/2611.06700) | 提出了一种物理信息驱动的自监督隐式神经表示框架PINR-ESI，利用导联场矩阵的物理约束和经典逆估计的暖启动初始化，在无需成对源标签或仿真预训练的情况下实现脑电扩展源成像的空间源重建。 |
| [^173] | [NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria](https://arxiv.org/abs/2611.06504) | 该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。 |
| [^174] | [Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox](https://arxiv.org/abs/2611.06022) | 该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。 |
| [^175] | [FedSharp: Byzantine Robust Decentralized Federated Learning Under Noisy Channel](https://arxiv.org/abs/2611.05546) | 提出FedSharp框架，通过维护2b个前驱节点模型、坐标式中值偏差筛选可疑模型并对损失最低的b个候选模型取平均，在噪声信道下有效抵抗拜占庭攻击与信道噪声累积，实现鲁棒的去中心化联邦学习。 |
| [^176] | [An Axiomatic Framework for Reinforcement Learning in Open Multi-Agent Systems](https://arxiv.org/abs/2611.03301) | 本文提出一个以 Shapley 公理为结构性约束的公理化框架，通过新颖的 NAHT 分解刻画开放与完全可控多智能体系统的本质差异，为开放多智能体系统的强化学习奠定合作博弈论基础，并证明对个体价值函数施加这些公理可恢复 Dec-POMDP 的 Shapley 值。 |
| [^177] | [Watch Your Step: Adam's Margin Maximization Bias Depends on the Learning Rate Schedule](https://arxiv.org/abs/2611.01929) | 本文揭示了Adam的隐式偏差依赖于学习率调度——在常见的恒定或弱衰减调度下，Adam对齐次模型不再具有与轨迹无关的间隔最大化偏差。 |
| [^178] | [Causal-prior-constrained deep learning for rapid-intensification warning of western North Pacific tropical cyclones](https://arxiv.org/abs/2610.40304) | 该研究提出CausalVisionRI系统，将观测发现的因果图及八项符号验证效应作为可审计的因果先验约束嵌入深度学习模型，融合红外遥感影像与大气海洋场数据，显著提升了西北太平洋热带气旋24小时快速增强预警的性能。 |
| [^179] | [Mode Dynamics: A Framework for Neural Network Learning](https://arxiv.org/abs/2610.40143) | 该论文提出“模式动力学”框架，通过分解、重构和集中度三种机制将神经网络训练中参数的集体运动模式与其输出效应联系起来，发现少数主导模式即可重构模型行为，且效应依赖训练路径、模式复杂度随任务和数据变化。 |
| [^180] | [EmbodiExp: Compositional Experience Transfer for Embodied Agents](https://arxiv.org/abs/2610.39180) | EmbodiExp 提出一种以结果为依据的子目标经验表示，使具身智能体能够在任务执行过程中在线组合与迁移经验，并借助子目标层面的结果归因来利用失败轨迹。 |
| [^181] | [Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI](https://arxiv.org/abs/2610.28411) | Brain2Vec是一个在74,425个T1加权脑部MRI扫描上预训练的自监督3D基础模型，既可作为下游任务微调的权重初始化骨干，也可作为独立特征提取器，在脑年龄预测等任务中展现出更高的精度和更快的收敛速度。 |
| [^182] | [Learning to Read Out: Unembedding Dynamics in Language Model Pretraining](https://arxiv.org/abs/2610.21025) | 提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。 |
| [^183] | [Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena](https://arxiv.org/abs/2610.19095) | 该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。 |
| [^184] | [PATTERN Can Be Solved from Memory: Testing Graph Neural Networks on Unseen Patterns](https://arxiv.org/abs/2610.15020) | PATTERN基准因训练与测试重复使用相同的100个模式而可被“记忆”破解——简单的查找表就能超越所有已发表的GNN结果，而重新训练的图神经网络在未见过的模式上性能大幅下滑，说明该基准无法衡量模型的真正泛化能力。 |
| [^185] | [Similar Predictive Fit but Different Latent Dynamics: Characterizing Learned Dynamical Structure in Personalized Models of Brain Disorders](https://arxiv.org/abs/2610.10850) | 该研究提出将EEG基础模型的潜在表示映射为个性化转移依赖图的方法，发现即使预测拟合度相似，癫痫患者的潜在动力学依赖结构也显著比非癫痫者更密集，表明模型学到的动力学结构能揭示超越预测准确率的临床相关差异。 |
| [^186] | [High-dimensional online calibration from harmonic weights](https://arxiv.org/abs/2610.07740) | 本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。 |
| [^187] | [Lightweight Semantic EEG Foundation Model for Frozen Cross-Disorder Transfer](https://arxiv.org/abs/2610.05503) | 该论文提出仅有146万参数（约为现有模型五分之一）的轻量级脑电图基础模型USE-FM，通过自监督信号重建学习可迁移神经表征，并在冻结编码器条件下于异常脑电检测和癫痫发作识别两类跨疾病任务上取得与大型模型相当的性能。 |
| [^188] | [What Should World Models Forget? Stratified Retention for Continual Adaptation](https://arxiv.org/abs/2610.03713) | 提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。 |
| [^189] | [Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback](https://arxiv.org/abs/2610.02771) | 本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。 |
| [^190] | [Cross-Benchmark Transfer from RL on Agentic Coding Tasks](https://arxiv.org/abs/2610.00890) | 仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。 |
| [^191] | [Parameterization method of reservoir properties for ensemble-based data assimilation using intermediate latent space of StyleGAN](https://arxiv.org/abs/2609.39626) | 本研究提出利用StyleGAN2的中间潜在空间对储层物性进行参数化，以克服集合数据同化方法在处理非高斯复杂地质（复杂相分布）条件下历史拟合性能下降的问题，并与VAE-GAN和潜在扩散模型等深度学习参数化方案进行了对比评估。 |
| [^192] | [When a Flatness Proxy Is Not a Function: Robustness Certificates and Training Interventions](https://arxiv.org/abs/2609.38540) | 该论文证明，最后一层相对平坦度代理的有效曲率上界既不能作为鲁棒性证书的依据，也不能作为训练干预的依据，并提出规范不变的特征空间修复与行中心化两种补救方法。 |
| [^193] | [Generative sequence modeling for infinite memory processes via predictive states](https://arxiv.org/abs/2609.38524) | 本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。 |
| [^194] | [From Data to Program: Fast & Direct Generative Program Inference from Empirical Data](https://arxiv.org/abs/2609.35348) | PRODiGI是一种预训练的数据到程序模型，能在单次前向传播中从经验数据直接推断出显式可执行的生成程序，配合程序空间微调在密度和分数估计上超越现有预训练模型并获得数倍加速。 |
| [^195] | [ReCo: When to Relocate Sensor Kits under Deployment Constraints -- A NILM Case Study](https://arxiv.org/abs/2609.35075) | 提出了一种基于约束的传感器套件迁移决策框架ReCo，通过在目标-背景联合特征空间中统计新运行状态并预测各住宅的未来数据增益，解决NILM数据采集中“何时迁移”的决策问题。 |
| [^196] | [The Alignment Illusion in Multimodal Large Language Models](https://arxiv.org/abs/2609.30210) | 该研究通过对13个多模态大语言模型施加受控干预，揭示了标准标量相似度度量无法真实反映跨模态内容交互的“对齐幻觉”现象，其根源在于MLP下投影所导致的权重诱导对齐。 |
| [^197] | [UltraBench 2: Towards Robust Evaluation of Vision Foundation Models on Ultrasound](https://arxiv.org/abs/2609.28610) | 该论文推出了UltraBench 2，一个覆盖广泛解剖结构和任务、注重标准化与可重复性的超声视觉基础模型综合评估基准，并发现超声特异性预训练在分类任务上仍占优势，而最先进的通用模型在分割任务上已与之持平。 |
| [^198] | [Hidden not Deleted: How Networks Suppress Entangled Features](https://arxiv.org/abs/2609.27593) | 该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。 |
| [^199] | [Beyond Embedding Transfer: Component Roles in Grokking Transfer and Stability](https://arxiv.org/abs/2609.18078) | 该论文通过大规模控制实验首次系统区分了Grokking热启动迁移中各模型组件的角色，发现迁移内部注意力/MLP权重不仅可将早期准确率提升5.46个百分点并缩短确认延迟，且该优势在双层模型的前瞻性复现实验中得到完全验证。 |
| [^200] | [Similarity Pairing with Energy Mover's Distance for Self-Supervised Pre-Training at the LHC](https://arxiv.org/abs/2609.17738) | 该论文提出一种利用能量移动距离（EMD）按相似性配对真实对撞事件作为数据增强视图的方法，无需人工构造或模拟增强事件即可保持事件物理真实性，从而改进LHC自监督预训练。 |
| [^201] | [Self-Supervised Learning for Robust Resonance Mass Regression in Cascade Decays](https://arxiv.org/abs/2609.17726) | 该论文提出遵循基础模型范式，先用VICReg自监督预训练transformer编码器以学习对各种数据破坏不变的嵌入表示，再微调用于级联衰变中重共振粒子的质量回归，从而在系统不确定性和分布偏移下实现鲁棒的质量重建。 |
| [^202] | [A Gradient-based yet Spike-Timing-Dependent Solution to the Feedback Learning Problem in Neural Microcircuits](https://arxiv.org/abs/2609.08070) | 本文提出梯度隧穿算法，将时间信用分配重构为状态分离问题，从局部突触脉冲时序推导信用分配，实现了与生物脉冲时序一致的神经微回路在线反馈学习框架。 |
| [^203] | [Attributing Cohen's d: Training Data Attribution for Disease-Related Effects in Normative Age Biomarkers](https://arxiv.org/abs/2609.07729) | 本文提出一种基于Cohen's d的闭式影响泛函，可直接将常模年龄模型中疾病相关的年龄差距效应归因到个体训练样本，并在UK Biobank上证明移除最有影响力的训练样本可显著提高疾病相关效应量。 |
| [^204] | [Learning Task-Specific Antibody Representations via Function-Aware Masking](https://arxiv.org/abs/2609.00518) | 该论文提出功能感知掩码这一预训练算法家族，通过将掩码位置与特定功能先验（如IMGT注释或结构预测）对齐来学习任务特异性的抗体表示，在结构相关任务上最高提升14%，在CDR相关任务上最高提升5.9倍。 |
| [^205] | [Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference](https://arxiv.org/abs/2608.18982) | 门罗是一种新型分子基础模型，通过扩大预训练规模、改进立体化学表示、优化损失函数和多任务学习，并结合TabPFN实现上下文概率推断，显著提升了数据受限下的生物活性预测性能。 |
| [^206] | [BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems](https://arxiv.org/abs/2608.02612) | 该论文提出BBOWP-Bench基准，首次评估大语言模型能否从自然语言描述的黑箱优化文字问题中同时推断出合适的搜索空间并选择恰当的优化算法。 |
| [^207] | [When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning](https://arxiv.org/abs/2607.29617) | 该论文发现，在基于价值的模仿学习中，与专家进行同策略交互能放宽对学习者的表征要求——只需模型能实现专家的价值函数，而无需实现要求更严格的专家策略本身。 |
| [^208] | [Fusion-Free Multimodal Classification with Heterogeneous Deep Ensembles](https://arxiv.org/abs/2607.05019) | 该论文证明，无需任何显式模态融合，仅用单模态网络的深度集成在相同参数量下即可持续超越为解决模态不平衡而设计的最先进后期融合及其他融合方法。 |
| [^209] | [Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study](https://arxiv.org/abs/2607.04926) | 该研究通过对微型Transformer进行完全可枚举实验发现，无论信息以符号token、神谕代码还是纠缠感知向量的形式输入，模型都无法实现零样本组合绑定（其失败源于归纳偏置而非信息缺失），但在小样本设置下，绑定学习的样本效率由输入代码的可读性决定。 |
| [^210] | [URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment](https://arxiv.org/abs/2607.04688) | URSA提出了一个化学感知的逆合成评估基准框架，首次从形式指标（如收敛到商业可得起始原料）和模拟专家化学家判断的化学合理性两个维度，对端到端逆合成模型与大语言模型进行客观、可解释的比较评估。 |
| [^211] | [MMAO-Cls: Metabolic Multi-Agent Optimization for Joint Feature Selection and Classifier Tuning](https://arxiv.org/abs/2607.01539) | 提出MMAO-Cls，一种基于代谢多智能体优化的混合空间方法，将特征选择与分类器超参数调优联合编码于每个智能体中，并通过特征信息先验和过拟合正则化加以增强，在七个表格基准上展现出与GA-lite等基线方法相当且更具竞争力的性能。 |
| [^212] | [QVal: Evaluating Dense Supervision Signals for Long-Horizon LLM Agents](https://arxiv.org/abs/2606.32034) | 该论文提出QVal——一个无需训练的测试平台，通过衡量密集监督信号在每个状态下对候选动作的排序与参考策略回报排序的一致性（Q-对齐性）来直接评估信号质量，并在四个环境中对七个方法族的21种方法进行基准测试，发现简单提示基线平均Q-对齐性最高且性能按方法族高度聚集。 |
| [^213] | [Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts](https://arxiv.org/abs/2606.28228) | 该论文证明了在未知非线性观测下，仅利用多个环境间扩散协方差的变化即可识别加性噪声潜在SDE的潜在坐标（至置换、缩放和常数平移），且无需对漂移项作任何稀疏性假设。 |
| [^214] | [A Horizon-Specific Attribute-Conditioned Neural Network for Estimating Health-State Transition Probabilities in Irregular Longitudinal Data: The LANTERN Framework](https://arxiv.org/abs/2606.13880) | 本文提出 LANTERN 框架，通过整合循环健康表示、观测间隔时间与自适应的人口社会经济属性条件化，从不规则纵向数据中估计特定预测期的健康状态转移概率，为长期护理保险建模提供了超越传统精算多状态模型的新方法。 |
| [^215] | [Express Language Modeling](https://arxiv.org/abs/2606.10944) | Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。 |
| [^216] | [Larch: Learned Query Optimization for Semantic Predicates](https://arxiv.org/abs/2606.07923) | Larch利用非结构化数据嵌入所承载的语义信息，通过学习型优化技术高效优化AI SQL查询中语义过滤器的执行。 |
| [^217] | [Generative Molecular Morphing for Flexible-Size Design via Unbalanced Optimal Transport](https://arxiv.org/abs/2606.07239) | 提出了Morph——一个基于非平衡最优传输的柔性尺寸3D分子生成模型，通过动态调整原子数目实现分布外生成，并显著增强对分子性质的引导能力。 |
| [^218] | [Evidence-Guided Neural Architecture Selection under Uncertainty for Subject-Specific Blood Glucose Forecasting](https://arxiv.org/abs/2606.05373) | 提出了EVIDENT框架，通过结合贝叶斯训练、基于证据的排序和不确定性下的任务特定验证，自动为1型糖尿病个体化血糖预测选出满足验证准则且容量最低的TCN架构。 |
| [^219] | [Certified Policy Optimisation for Nested Causal Bandits via PAC-Bayes Risk](https://arxiv.org/abs/2605.29788) | 本文提出嵌套上下文因果老虎机（NCCB）这一分层因果决策问题框架及嵌套因果汤普森采样（NCTS）算法，其核心贡献是一个因果PAC-Bayesian超额风险界，可仅凭历史数据、离策略且随时地对任意候选部署策略进行认证，回答“能否信任该智能体、风险多大”的部署问题。 |
| [^220] | [Kernel-based potential mean-field games with unbiased random Fourier $U$-statistics](https://arxiv.org/abs/2605.29371) | 本文针对运行成本与终端成本均由核MMD惩罚刻画的势均场博弈，提出了一种基于无偏随机傅里叶U统计量的计算框架，以神经网络参数化漂移并通过随机梯度下降训练，同时证明了在惩罚发散及显式耦合条件下向终端约束问题的收敛性。 |
| [^221] | [SA-RAG: Structuring Noisy Conversations for Retrieval-Augmented Generation](https://arxiv.org/abs/2605.24366) | 提出SA-RAG框架，通过质量感知的结构归纳方法直接从噪声对话数据中构建表格形式的可靠中间结构，从而提升检索增强生成在嘈杂上下文中的表现。 |
| [^222] | [AMUSE: Anytime Muon with Stable Gradient Evaluation](https://arxiv.org/abs/2605.22432) | 该论文提出AMUSE优化器，通过“河谷”损失景观理论揭示了Muon正交化虽加速主体方向进展但会放大支配方向噪声引发振荡的问题，并结合稳定梯度评估与随时调度能力来加以改进。 |
| [^223] | [TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation](https://arxiv.org/abs/2605.22355) | 该论文发布了包含四个中国城市近1300万条路线规划记录的大规模数据集TransitLM，证明大语言模型可完全从数据中学习公交路线规划，无需地图即可实现从起终点信息到路线的端到端生成。 |
| [^224] | [Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation](https://arxiv.org/abs/2605.18020) | 提出了非货币化框架FedUCA，通过效用约束的随机聚合机制确保理性客户端的协作收益满足其本地效用阈值，从而提升客户端参与意愿、防止客户端流失导致的联邦训练崩溃。 |
| [^225] | [Wavelet Flow Matching for Multi-Scale Physics Emulation](https://arxiv.org/abs/2605.16573) | 提出小波流匹配（WFM），直接在多尺度小波空间中进行最优传输流匹配，无需预训练自编码器，在混沌流体仿真中以更低成本实现了更优的谱相干性与细节保持能力。 |
| [^226] | [Shortcut Mitigation via Spurious-Positive Samples](https://arxiv.org/abs/2605.13340) | 提出一种无需数据标注或组平衡数据的新型捷径缓解方法，通过识别模型依赖虚假属性的少量实例，定位中间层中高度相关神经元并正则化其影响，使模型依靠真实有效特征进行预测，从而提升鲁棒性。 |
| [^227] | [Adaptive Calibration in Non-Stationary Environments](https://arxiv.org/abs/2605.11490) | 该论文提出了能够自动适应环境非平稳程度的在线预测校准算法，其校准误差可在独立同分布与对抗性两种情形之间平滑过渡，并在多种校准指标下提供了自适应保证。 |
| [^228] | [When Adaptation Fails: A Gradient-Based Diagnosis of Collapsed Gating in Vision-Language Prompt Learning](https://arxiv.org/abs/2605.09549) | 本文通过基于梯度的系统诊断研究，揭示了CLIP类视觉语言模型少样本提示学习中自适应门控坍塌的两种失败模式——梯度幅度失衡和门控退化，警示不应盲目增加架构复杂性。 |
| [^229] | [Diagnosing and Mitigating Domain Shift in Permission-Based Android Malware Detection](https://arxiv.org/abs/2605.09028) | 本文系统诊断了基于权限的Android恶意软件检测器在跨域场景下的领域偏移问题，揭示了严重的非对称性能下降与特征重要性的跨域不稳定性，并提出了相应的缓解方法。 |
| [^230] | [Retrieval of Coastal Biogeochemical Parameters From Near-Surface Hyperspectral Remote Sensing Reflectance Using Physics-Aware Meta-Learning](https://arxiv.org/abs/2605.05623) | 本文提出了一个两阶段物理感知元学习框架，通过在纯数据驱动方法基础上融入物理知识，从近表面高光谱遥感反射率中反演沿岸水质参数，从而克服了传统反演算法难以跨不同水体推广的挑战。 |
| [^231] | [Time series causal discovery with variable lags](https://arxiv.org/abs/2605.04081) | 本文提出了一种可分解的BIC风格评分方法，用于联合学习时间序列因果发现中的变量时间顺序与每个有序变量对的主导滞后项，并通过增强的禁忌搜索算法进行优化。 |
| [^232] | [BoostLoRA: Growing Effective Rank by Boosting Adapters](https://arxiv.org/abs/2604.27308) | BoostLoRA提出一种梯度提升框架，通过在错误样本上迭代训练并合并正交子空间的超低秩适配器，使有效秩随轮次线性增长，在不增加任何推理开销的情况下超越了全量微调的性能。 |
| [^233] | [mlr3torch: A Deep Learning Framework in R based on mlr3 and torch](https://arxiv.org/abs/2604.18152) | 本文介绍了R包mlr3torch，这是一个构建于torch之上、深度集成mlr3生态系统的可扩展深度学习框架，它简化了神经网络的定义、训练和评估，并允许用户通过图语言在单一图中定义包含预处理、数据增强和网络架构的完整建模工作流程。 |
| [^234] | [Minimizing classical resources in variational measurement-based quantum computation for generative modeling](https://arxiv.org/abs/2604.11578) | 本文针对变分测量量子计算（VMBQC）中信道模型参数量是幺正模型两倍的局限，提出了最小化经典资源消耗的方案，使生成建模中的经典处理开销得到显著降低。 |
| [^235] | [Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification](https://arxiv.org/abs/2604.07953) | 本文提出多维度评估框架、有界误差剪枝策略及新型Hydrant分类器扩展，可在时间序列分类任务中实现能耗降低高达70%，而预测精度几乎不受影响。 |
| [^236] | [Resolving AdaBoost Cycling with LLMs: A Computer-Assisted Counterexample](https://arxiv.org/abs/2604.07055) | 本文通过计算机辅助构造了一个包含 2,500 个样本的反例，证明穷举式 AdaBoost 并不总是收敛到有限循环，从而解决了 Rudin、Schapire 和 Daubechies 提出的长期公开问题。 |
| [^237] | [Generative machine learning for protein design](https://arxiv.org/abs/2603.26378) | 本文系统综述了59种生成式机器学习方法在六类蛋白质设计任务（序列生成、骨架生成、结构条件序列设计、序列-结构联合设计、基序支架及结合蛋白/抗体设计）中的应用，并讨论了设计蛋白质的计算与实验评估方法。 |
| [^238] | [DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation](https://arxiv.org/abs/2603.21465) | 提出DRTriton框架，通过CSP-DAG大规模合成数据、解耦奖励的课程化强化学习以及测试时搜索算法，训练大语言模型将PyTorch程序高效转换为优化的Triton内核。 |
| [^239] | [Auto-differentiable data assimilation: Co-learning of states, dynamics, and filtering algorithms](https://arxiv.org/abs/2603.20891) | 本文提出了一种自动微分滤波框架，利用理论驱动的损失函数和基于梯度的优化，实现对状态、动力学模型以及滤波算法参数的联合协同学习，从而摆脱传统数据同化中昂贵的参数调优和对精确动力学模型的依赖。 |
| [^240] | [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854) | FlashSampling将精确采样融合进LM头的矩阵乘法运算中，通过在片上逐块计算并添加Gumbel噪声避免在显存中物化logits张量，并以流式点对点通信替代all-gather，实现了跨最多8个GPU的快速、精确且内存高效的大词表采样。 |
| [^241] | [Denoising the US Census: Succinct Block Hierarchical Regression](https://arxiv.org/abs/2603.10099) | BlueDown是一种结合线性回归与约束优化的后处理算法，能在不牺牲隐私保护或计算效率的前提下，显著降低2020年美国人口普查差分隐私噪声造成的分组数据误差。 |
| [^242] | [Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation](https://arxiv.org/abs/2603.09527) | 本文提出参数与数据高效的草稿模型适配框架 EDA，通过解耦架构、数据再生策略和样本选择机制三项创新，高效地将草稿模型对齐到微调后的目标模型，从而提升推测解码的加速效果。 |
| [^243] | [ESENSC: A Polynomial-Time Axiomatic Alternative to SHAP](https://arxiv.org/abs/2603.00496) | ESENSC是一种仅需线性次联盟值评估的多项式时间确定性归因方法，在表格预测任务上与精确SHAP高度一致，同时计算成本远低于后者。 |
| [^244] | [Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume](https://arxiv.org/abs/2602.24195) | 提出了UMPIRE，一个免训练且无需外部工具的多模态大语言模型不确定性量化框架，其受行列式点过程启发，通过计算新颖的“非相干性调整语义体积”，将语义嵌入离散度与响应二次熵相结合，从而高效支持多种输入输出模态。 |
| [^245] | [On sparsity, extremal structure, and monotonicity properties of Wasserstein and Gromov-Wasserstein optimal transport plans](https://arxiv.org/abs/2602.16265) | 本文系统梳理了Gromov-Wasserstein最优传输的关键性质，证明了在条件负定/正定（CND/CPD）条件下存在稀疏且支撑于置换上的最优传输计划，并给出不满足这些假设时置换并非最优的反例。 |
| [^246] | [Flow Matching from Viewpoint of Proximal Operators](https://arxiv.org/abs/2602.12683) | 该论文通过扩展Brenier势将最优传输条件流匹配（OT-CFM）的向量场精确表示为与势函数相关的近端算子（去噪器），并借助Lyapunov指数证明其流形稳定性：法向扰动线性收缩而切向扰动被保留。 |
| [^247] | [Adaptive Power Iteration Method for Differentially Private PCA](https://arxiv.org/abs/2602.11454) | 本文提出一种自适应的差分隐私幂迭代算法，通过引入适应矩阵相干性参数的新过滤技术，对低相干性矩阵实现了超越最坏情况的顶部奇异向量近似保证。 |
| [^248] | [ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression](https://arxiv.org/abs/2602.11008) | ROCKET是一种免训练的模型压缩方法，通过将逐层压缩分配建模为多选择背包问题，并结合基于校准集的单步稀疏矩阵分解，在全局压缩预算下实现了超越现有方法的最先进压缩性能。 |
| [^249] | [From Chats to Markets: AgenticPay for LLM-Powered Negotiation in Multi-Agent Commerce](https://arxiv.org/abs/2602.06008) | AgenticPay是一个包含160个多模态任务、覆盖四种商业场景和八种市场拓扑的统一基准，用于评估多模态LLM智能体在竞争性多智能体市场中通过多轮谈判达成高福利、多维度协议的能力。 |
| [^250] | [The Alignment Curse: Modality Alignment Supercharges Audio Attacks via Text Transfer](https://arxiv.org/abs/2602.02557) | 提出并验证了“对齐诅咒”原则——全模态模型中文本-音频模态对齐越强，文本越狱攻击向音频模态的迁移就越有效，揭示了模型能力提升与安全性之间的根本张力。 |
| [^251] | [Deep Multivariate Models with Parametric Conditionals](https://arxiv.org/abs/2602.01953) | 该论文提出用每组变量以其余变量为条件的参数化条件概率分布来表示联合概率分布，从而构建几乎可通用于任意下游任务的深度多元模型，其训练等价于通过最大化极限分布的数据似然来学习一个参数化马尔可夫链核。 |
| [^252] | [HetCCL: Efficient LLM Training on Heterogeneous Vendor GPUs](https://arxiv.org/abs/2601.22585) | HetCCL是一个无需修改驱动程序即可统一NVIDIA NCCL和AMD RCCL后端的集合通信库，通过基于RDMA的跨厂商通信机制，使现有深度学习应用无需改动即可在NVIDIA与AMD混合GPU集群上高效训练大语言模型。 |
| [^253] | [Tracing 3D Anatomy in 2D Strokes: Can Projection-Derived Masks Replace 3D Segmentation for Cervical Spine Fracture Recognition?](https://arxiv.org/abs/2601.15235) | 本研究提出一种多阶段流水线，通过从二维投影近似生成椎体掩膜来替代完整的三维分割，在大幅降低计算复杂度的同时，仍能保留足够的诊断上下文以实现颈椎骨折的高精度识别。 |
| [^254] | [Vision-language models lag human performance on physical dynamics and intent reasoning](https://arxiv.org/abs/2601.01547) | 该论文提出“目的性空间智能”（TSI）概念，并基于11,328个真实世界视频构建了EscherVerse大规模基准与指令微调数据集，揭示当前最先进的视觉语言模型在整合物理动力学与人类意图进行空间推理方面仍显著落后于人类。 |
| [^255] | [Improving Matrix Exponential for Generative AI Flows: A Taylor-Based Approach Beyond Paterson--Stockmeyer](https://arxiv.org/abs/2512.20777) | 本文提出一种超越经典Paterson–Stockmeyer求值方案的优化泰勒矩阵指数算法，通过严格的误差分析和泰勒阶数与缩放因子的动态选择策略，在满足误差容限的前提下显著降低计算量，为生成式AI流带来大幅加速。 |
| [^256] | [NEAT: Neighborhood-Guided, Efficient, Autoregressive Set Transformer for 3D Molecular Generation](https://arxiv.org/abs/2512.05844) | 提出NEAT模型，通过新颖的邻域引导训练策略将分子图视为原子集合并学习与顺序无关的token分布，实现原子级置换不变性，在QM9和GEOM-Drugs数据集上达到最先进的3D分子生成质量。 |
| [^257] | [Model-Free Assessment of Simulator Fidelity via Quantile Curves](https://arxiv.org/abs/2512.05024) | 该论文提出一种无模型方法，通过为潜在总体参数构建置信集并估计分位数曲线，来稳健地量化生成式AI模拟器与真实系统之间随情景变化的“仿真-现实”差距。 |
| [^258] | [Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows](https://arxiv.org/abs/2512.04954) | 该论文提出用似然加权重要性采样训练归一化流以高效推断多模态后验分布，并发现用与目标众数数量匹配的高斯混合模型初始化流可以消除模式间的虚假概率桥梁，从而显著提升重建保真度。 |
| [^259] | [E0: Expressive Fine-Grained Discrete Action Prediction for Vision-Language-Action Models via Tweedie Discrete Diffusion](https://arxiv.org/abs/2511.21542) | 提出E0框架，通过Tweedie离散扩散将动作生成建模为对量化动作token的迭代去噪过程，实现了与token化推理天然契合的表达性细粒度离散动作预测，从而解决VLA模型泛化能力差和动作粗糙不稳定的问题。 |
| [^260] | [Manifold Approximation leads to Robust Kernel Alignment](https://arxiv.org/abs/2510.22953) | 提出流形近似核对齐（MKA），通过将流形局部邻域几何融入核对齐任务，去除启发式度加权、给出闭式对齐分数及马尔可夫概率解释，并证明其对邻域大小的稳定性，从而实现更鲁棒的表示比较。 |
| [^261] | [CS-54k: A Paper-Grounded Corpus and Benchmark for AI Research QA](https://arxiv.org/abs/2510.20279) | 提出了CS-54k，一个从14,474篇开放许可的AI研究论文中构建的包含5.4万个问答对的语料库与基准，涵盖论文的八个核心组成部分，为AI研究问答提供了可用于模型评估与后训练的结构化监督资源。 |
| [^262] | [Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling: A Model-level Approach to High-dimensional and Multi-action Environments](https://arxiv.org/abs/2510.19244) | 提出了一种带强化学习引导标注的增强SILVER框架，通过将RL策略自身的动作输出纳入边界点识别，将基于Shapley回归的策略解释方法扩展到高维和多动作环境。 |
| [^263] | [Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction](https://arxiv.org/abs/2510.10140) | 本文提出了Cyc-Attack对抗攻击方法，通过对上游天气预报模型施加细微扰动来操纵下游热带气旋轨迹预测，克服了检测系统黑盒不可微分和热带气旋样本极度稀缺导致的类别不平衡两大难题。 |
| [^264] | [Simultaneous Learning and Optimization via Misspecified Saddle Point Problems](https://arxiv.org/abs/2510.05241) | 该论文提出了一个将优化与参数学习统一处理的误设定鞍点问题框架，并设计了两种基于加速原始-对偶（APD）方法的算法，均达到 $\mathcal{O}(\log K/K)$ 收敛率，其中学习感知变体通过调整动量更新显式建模参数动态，并获得更优的 $\mathcal{O}(1)$ 约束保证。 |
| [^265] | [Graph Your Own Prompt](https://arxiv.org/abs/2509.23373) | 该论文提出图一致性正则化（GCR），通过在任意深度插入无参数的图一致性层，将特征相似性图与由模型自身预测导出的类感知掩码预测图对齐，实现自我提示式训练，使特征表示更具类一致性和语义意义。 |
| [^266] | [Bayesian Transfer Operators in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2509.22482) | 该论文提出高斯过程动态模态分解（GP-DMD），一种将稀疏变分高斯过程与基于核的Koopman及Perron–Frobenius算子近似相结合的贝叶斯转移算子框架，可同时实现不确定性量化、可扩展的稀疏近似和数据自适应字典学习。 |
| [^267] | [Toric geometry of ReLU neural networks](https://arxiv.org/abs/2509.05894) | 该论文建立了ReLU神经网络与环面几何之间的联系，定义了ReLU扇、ReLU环面簇和ReLU卡地亚除子等新概念，借助代数几何工具系统研究了网络的精确函数实现问题与表达能力，并揭示了热带几何与环面几何之间的对应关系。 |
| [^268] | [Structured Basis Function Networks: Loss-Centric Multi-Hypothesis Ensembles with Controllable Diversity](https://arxiv.org/abs/2509.02792) | 本文提出结构化基函数网络，通过受Bregman质心几何启发的损失感知监督聚合机制与可调的多样性控制参数，将集成学习与多假设预测统一在同一框架中，在回归、图像识别和大规模视频分类等任务上均取得优异表现。 |
| [^269] | [Training Language Model Agents to Find Vulnerabilities with CTF-Dojo](https://arxiv.org/abs/2508.18370) | 提出了 CTF-Dojo——首个面向 LLM 训练的大规模可验证执行环境，包含 658 个容器化 CTF 挑战，并配套自动化流水线 CTF-Forge，可数分钟内将公开工件转化为即用型执行环境，仅需 486 条高质量轨迹即可训练出擅长漏洞发现的智能体。 |
| [^270] | [Robust Surrogate Modeling for Explainable Graph Neural Networks](https://arxiv.org/abs/2508.01925) | 该论文提出通过迭代重加权策略训练一个鲁棒代理模型，解决了训练图与解释图之间的分布偏移问题，从而为图神经网络提供更忠实可靠的解释。 |
| [^271] | [Cyber-Zero: Training Cybersecurity Agents without Runtime](https://arxiv.org/abs/2508.00910) | Cyber-Zero是首个无需运行时环境的网络安全智能体训练框架，通过利用公开CTF解题报告和基于角色扮演的LLM模拟来合成高质量交互轨迹，使训练出的智能体在三个CTF基准测试上相比基线模型最高取得11.4%的绝对性能提升。 |
| [^272] | [Recovering the Euclidean Geometry of High-Dimensional Data Under Heterogeneous Noise](https://arxiv.org/abs/2507.18520) | 本文证明了对于具有底层低维几何结构的高维数据，即使在异构噪声严重且不均匀地破坏成对欧几里得距离的情况下，仍然可以恢复出其底层干净的欧几里得几何结构。 |
| [^273] | [HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithmic Coding](https://arxiv.org/abs/2505.24098) | HARDTESTGEN提出了一种为算法编程问题自动合成高质量测试用例的流水线，并构建了包含2.66万道题的HARDTESTS数据集，其测试用例在验证LLM生成代码方面的精确度比现有测试提升11.22个百分点，有效支撑拒绝采样和强化学习等下游后训练。 |
| [^274] | [Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection](https://arxiv.org/abs/2505.18934) | 提出基于新型卡方小波滤波器的谱图神经网络框架ChiGAD，通过多图卡方滤波捕获跨元路径异常信号、交互式元图卷积在维度对齐中保留高频信息，有效解决了异构图异常检测中的语义捕获与类别不平衡难题。 |
| [^275] | [Bias-Optimal Bounds for Stochastic Gradient Descent](https://arxiv.org/abs/2505.17965) | 本文在仅假设凸性和光滑性的条件下，为随机梯度下降推导出覆盖全部常数步长范围的偏差最优收敛界，并证明在临界步长处最优偏差虽无法精确达到但可被任意逼近。 |
| [^276] | [AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion](https://arxiv.org/abs/2505.12358) | AbFlowNet提出了一种将GFlowNet与扩散模型融合的新型生成框架，通过将每个扩散步骤建模为GFlowNet中的状态，在单一训练流程中直接引入能量信号，实现扩散损失与抗体-抗原结合能的联合优化，从而避免了依赖不可靠估计器的昂贵在线强化学习流程。 |
| [^277] | [Bidirectional Information Flow (BIF) - A Sample Efficient Hierarchical Gaussian Process for Bayesian Optimization](https://arxiv.org/abs/2505.11294) | 提出双向信息流（BIF）框架，使层次高斯过程的父子节点间实现持续双向信息交换，从而提升贝叶斯优化的样本效率、训练稳健性，并支持子任务模型的模块化重用。 |
| [^278] | [Mapping Seven Decades of Philosophy in Colombia: Dynamic Topic Modelling of Ideas y Valores](https://arxiv.org/abs/2412.04236) | 本研究通过动态主题建模技术分析哥伦比亚最具影响力的哲学期刊《Ideas y Valores》七十年来主题的演变，将哲学史的数据驱动研究拓展至哥伦比亚和拉丁美洲语境，发现价值理论、认识论和科学哲学是该地区最突出的哲学研究主题。 |
| [^279] | [LEGAL-UQA: Testing Constitutional Question-Answering Systems for Urdu Speakers with a Parallel English-Urdu Benchmark](https://arxiv.org/abs/2410.13013) | 提出了LEGAL-UQA，一个包含619个问答对的英乌尔都语平行基准，用于测试面向乌尔都语使用者的巴基斯坦宪法问答系统，研究发现模型排名依赖相似度度量方式、词匹配检索优于嵌入方法，且自动评判模型与人工评分的一致性有限。 |
| [^280] | [Benign Overfitting for Regression with Trained Two-Layer ReLU Networks](https://arxiv.org/abs/2410.06191) | 本文首次证明了通过梯度流训练的有限宽度两层ReLU网络在最小二乘回归中既能实现泛化又会过拟合数据，从而在任意有界回归函数下首次建立了良性过拟合结果。 |
| [^281] | [Robust Long-Tailed Learning under Label Noise](https://arxiv.org/abs/2108.11569) | 该论文发现传统的小损失噪声标签检测准则因深度网络将尾部类别误判为头部类别的固有偏差而在长尾分布下失效，并提出利用所学表示鲁棒性的小距离准则来准确识别标签正确的样本，同时用标签分布替代离散伪标签以改进尾部类别的训练。 |

# 详细

[^1]: AIS与射频配对：海上数据融合的马尔可夫方法分析

    Pairing AIS with RF: Analysis of a Markovian Approach for Maritime Data Fusion

    [https://arxiv.org/abs/2611.18563](https://arxiv.org/abs/2611.18563)

    该论文提出一种基于配对隐马尔可夫模型（PHMM）的马尔可夫方法，将天基射频检测结果与AIS船舶轨迹进行时空对齐，从而克服AIS系统的安全漏洞与射频观测无法识别船舶的局限，实现有效的海上多源数据融合。

    

    监测海上活动对于安全与保障至关重要，特别是在检测非法活动方面，但由于海上领域的规模和复杂性，这项工作仍然充满挑战。船舶自动识别系统（AIS）通常被用于这一目的，尽管它存在已知的漏洞。相比之下，天基射频（RF）观测提供了船舶活动的独立数据来源，但缺乏船舶身份信息，因此无法独立识别船舶。本研究遵循CRISP-DM流程，应用配对隐马尔可夫模型（PHMM）将单个射频检测结果与AIS轨迹进行对齐，在利用时间和空间信息的同时对噪声和时间偏差进行建模。我们在NOAA AIS数据上结合合成射频检测结果对该方法进行评估，并与最近邻基线方法进行对比，同时通过对轨迹长度、采样密度、平稳性以及AIS-RF距离的偏差测试来分析其误差。

    arXiv:2611.18563v1 Announce Type: new  Abstract: Monitoring maritime activities is important for safety and security, particularly for detecting illegal activities, but remains challenging due to the size and complexity of the maritime domain. The Automatic Identification System (AIS) is commonly used for this purpose, despite its known vulnerabilities. In contrast, space-based radio frequency (RF) observations provide an independent source of vessel activity but lack ship identification and therefore cannot identify vessels on their own. Following the CRISP-DM process, this study applies a Pair Hidden Markov Model (PHMM) to align individual RF detections with AIS trajectories, using temporal and spatial information while modeling noise and timing discrepancies. We evaluate it on NOAA AIS data with synthetic RF detections against nearest-neighbor baselines and analyze its errors with bias tests on trajectory length, sampling density, stationarity and AIS--RF distance. The length-correc
    
[^2]: 在仿真中训练推理模型以原生使用真实机器人

    Training Reasoning Models in Simulation to Natively Use Real Robots

    [https://arxiv.org/abs/2611.18562](https://arxiv.org/abs/2611.18562)

    该论文提出PEAR后训练引擎和RURL强化学习方法，仅通过大规模仿真训练（无需演示数据和外部框架），使开源27B推理模型获得可泛化到真实机器人上的原生机器人操控能力，缩小了开源模型与前沿专有模型的差距。

    

    机器人使用智能体（RUAs）已成为一种有前景的通用机器人控制方法。然而，迄今为止，强大的性能仅由 GPT-6 Astra 和 Claude Fable 5.2 等前沿专有模型实现，而开源权重模型的能力仍明显不足。我们展示了一个开源权重的 27B 推理模型可以通过强化学习被训练为原生使用真实机器人，训练完全在仿真中进行且无需任何演示数据。与计算机使用智能体（CUA）类似，该模型观察环境、进行思维链推理，并生成下一个末端执行器位姿，全程无需任何外部框架。我们构建了面向智能体机器人学的后训练引擎（PEAR），并利用它通过机器人使用强化学习（RURL）在大规模仿真机器人任务上训练该模型，这类似于前沿模型习得计算机使用能力的方式。我们发现，在仿真中习得的技能能够很好地泛化到训练任务之外：它们可以迁移……（摘要原文在此处截断）

    arXiv:2611.18562v1 Announce Type: cross  Abstract: Robot-use agents (RUAs) have emerged as a promising approach to general-purpose robot control. However, strong performance has so far been achieved only by frontier proprietary models such as GPT-6 Astra and Claude Fable 5.2, whereas open-weight models remain substantially less capable. We show that an open-weight 27B reasoning model can be trained to use real robots natively through reinforcement learning, entirely in simulation and without demonstrations. Like a computer-use agent (CUA), the model observes the environment, reasons in a chain of thought, and generates the next end-effector pose, without any harness. We build a Post-training Engine for Agentic Robotics (PEAR) and use it to train the model on simulated robot tasks at scale with robot-use reinforcement learning (RURL), mirroring how frontier models acquire computer use. We find that the skills acquired in simulation generalize well beyond the training tasks: they transfe
    
[^3]: FETTUCCINE：在移动设备上实现快速高效的脑信号到文本解码

    FETTUCCINE: Fast and efficient brain-to-text decoding on mobile devices

    [https://arxiv.org/abs/2611.18561](https://arxiv.org/abs/2611.18561)

    FETTUCCINE借助自动语音识别模型的速度与可移植性，实现了可在移动设备上运行、词错误率低于10%的端到端脑信号到文本解码框架。

    

    语音神经假体的最新进展为因神经系统疾病或损伤而丧失说话能力的人群开辟了新的沟通途径。这些系统通过依赖语言先验，将尝试说话期间的神经活动解码为文本，从而实现较高的解码准确率。然而，目前这一性能依赖于计算密集型的语言建模堆栈，成为实际应用中的主要瓶颈。此外，现有语言模型的推理延迟高度可变且具有干扰性，限制了沟通吞吐量，并阻碍用户可靠地参与自然对话。我们提出了FETTUCCINE，这是一个端到端的脑信号到文本框架，它利用自动语音识别（ASR）模型的速度和可移植性来解决上述实际挑战，同时实现了可用的（<10%）词错误率（WER），并能高效适应神经活动的非平稳性。

    arXiv:2611.18561v1 Announce Type: cross  Abstract: Recent advances in speech neuroprostheses have enabled new communication avenues for people who have lost the ability to speak due to neurological illness or injury. These systems can decode neural activity during attempted speech into text by relying on language priors to achieve high decoding accuracy. This performance currently comes at the cost of using a compute-intensive language modeling stack, presenting a major bottleneck for real-world use. Further, current language models incur highly variable and disruptive inference latencies, limiting communication throughput and preventing users from reliably participating in natural conversation. We introduce FETTUCCINE, an end-to-end brain-to-text framework that leverages the speed and portability of automatic speech recognition (ASR) models to address the aforementioned practical challenges, while achieving usable (<10%) word error rates (WER), efficient adaptation to neural non-stati
    
[^4]: BAL：基于潜空间双向自回归的人体运动表示学习方法

    BAL: Bidirectional Autoregression in Latent Space for Learning Human Movement Representations

    [https://arxiv.org/abs/2611.18559](https://arxiv.org/abs/2611.18559)

    本文提出BAL自监督框架，利用人体日常行为的自然时间顺序作为内在监督信号，通过潜空间中的双向自回归预测学习人体运动表示，避免了以往人工假设引入的启发式超参数问题。

    

    从3D姿态序列中学习具有语义意义的人体运动表示对于人类行为建模至关重要。现有的自监督方法从人工假设中获取监督信号，例如对一定范围内数据增强的不变性，或在特定时间尺度上被掩码区域的可恢复性，这引入了启发式超参数，使得学习到的语义对这些设置非常敏感。我们提出了BAL，一种自监督框架，它转而利用人体运动内在的监督信号：日常行为的自然顺序结构。BAL在语义抽象的潜空间中进行自回归预测，这类似于人类在预测他人行为时关注的是动作本身而非精确的关节配置。时间上的前向与后向双向自回归预测充分挖掘了这些顺序约束，同时使得模型能够利用过去和未来的上下文来提取表示。

    arXiv:2611.18559v1 Announce Type: cross  Abstract: Learning semantically meaningful human movement representations from 3D pose sequences is essential for human behavior modeling. Existing self-supervised methods derive supervision from artificial assumptions, e.g., invariance to augmentation within a certain range or recoverability of masked regions at a certain temporal scale, introducing heuristic hyperparameters that make the learned semantics sensitive to their settings. We propose BAL, a self-supervised framework that instead exploits supervision intrinsic to human movement: the natural ordering structure of everyday behavior. BAL autoregressively predicts in a semantically abstract latent space, analogous to how humans anticipate others' actions rather than their precise joint configurations. Bidirectional autoregressive prediction, both forward and backward in time, fully leverages these ordering constraints while enabling representation extraction using both past and future co
    
[^5]: WeightBench：将语言模型作为权重空间程序员进行评估

    WeightBench: Evaluating Language Models as Weight-Space Programmers

    [https://arxiv.org/abs/2611.18553](https://arxiv.org/abs/2611.18553)

    该论文提出了WEIGHTBENCH基准，通过五大赛道的42项权重空间编程任务系统评估语言模型直接读取、修复和编写神经网络参数的能力，发现前沿模型已能读取小型网络并编写可用的控制策略，但尚无法修补或压缩大型语言模型。

    

    最近的研究表明，大型语言模型（LLM）可以直接操作其他神经网络的参数。它们能够读取一个网络被训练来做什么，修复失效的网络，并从零开始编写可正常工作的网络——所有这些都无需梯度下降。这种能力被称为权重空间编程。要系统地探索这一新范式，需要一个稳健的基准。我们提出了WEIGHTBENCH，这是一个在五大赛道（读取权重、修复权重、编写权重、识别不可能的编辑以及无数据压缩）上，对能力递增的LLM进行42项WSP任务评估的基准。LLM读取和编辑权重的网络规模从2K参数的控制策略到7B参数的语言模型不等。实验基准结果表明，前沿智能体已经能够读取小型网络并在从未实际运行它们的情况下编写出可正常工作的控制策略，但目前还无法修补语言模型或对其进行压缩。

    arXiv:2611.18553v1 Announce Type: cross  Abstract: Recent work shows that large language models (LLMs) can operate directly on the parameters of other neural networks. They can read what a network was trained to do, fix a network that fails, and write working networks from scratch, all without gradient descent. This capability has been termed Weight-Space Programming (WSP). Exploring this new paradigm systematically requires a robust benchmark. We introduce WEIGHTBENCH, a benchmark that evaluates LLMs of increasing capability on 42 WSP tasks across five tracks: reading, fixing and writing weights, recognizing edits that are impossible, and data-free compression. The networks whose weights the LLMs read and edit range from 2K-parameter control policies to 7B-parameter language models. Experimental benchmark results show that frontier agents can already read small networks and write working control policies without ever running them, but cannot yet patch language models, compress them wi
    
[^6]: 椭圆采样：面向高斯分布及更广泛情形的最优插值调度

    Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond

    [https://arxiv.org/abs/2611.18542](https://arxiv.org/abs/2611.18542)

    该论文提出了一种基于复化ODE轨迹解析性质的插值调度设计新框架，通过保角映射精确求解得到由雅可比椭圆函数参数化的保方差椭圆调度族，并结合Zolotarev最优有理函数逼近构建了能根据目标分布条件数自适应优化的新采样算法Zoloflow。

    

    基于随机插值的生成建模需要确定两个关键选择：连接初始分布与目标分布的插值调度，以及由此得到的概率流ODE的高效离散化方法。在这项工作中，我们提出了一个基于（复化后）ODE轨迹解析性质的调度/离散化联合设计新框架，该解析性质进而保证了良好的数值行为。我们的公式可以通过保角映射精确求解，从而得到一个由雅可比椭圆函数参数化的、自然的保方差椭圆调度族。将椭圆调度与量身定制的离散化方法相结合，我们得到了Zoloflow——一种与Zolotarev最优有理函数逼近密切相关的新采样算法。关键之处在于，椭圆调度与Zoloflow都显式地纳入了目标分布的条件数 $\kappa$，并可退化为经典的……（摘要截断）

    arXiv:2611.18542v1 Announce Type: cross  Abstract: Generative modeling based on stochastic interpolants requires specifying two key choices: an interpolant schedule bridging the initial and target distributions, and an efficient discretization of the resulting probability-flow ODE. In this work, we introduce a new framework for joint schedule/discretization design based on the analyticity properties of the (complexified) ODE trajectories, which in turn ensure good numerical behavior. Our formulation can be solved exactly via conformal mapping, yielding a natural family of variance-preserving elliptic schedules, parametrized by Jacobi elliptic functions. Pairing an elliptic schedule with a tailored discretization, we obtain Zoloflow, a new sampling algorithm closely connected to Zolotarev's optimal rational function approximants. Crucially, both the elliptic schedule and Zoloflow explicitly incorporate the condition number $\kappa$ of the target distribution and reduce to the classical 
    
[^7]: 为什么有限智能能在无限世界中有效：一种数学理论

    Why Finite Intelligence Works in an Infinite World: A Mathematical Theory

    [https://arxiv.org/abs/2611.18538](https://arxiv.org/abs/2611.18538)

    该论文提出“未来任务信息”理论，证明当任务请求共享集体结构时，有限的信息表示足以在无限环境中保持决策价值，并给出了有限保存的两个精确数学边界条件。

    

    一个信息系统往往需要在知道自己将被问及什么之前就进行学习。当这些请求共享集体结构，而不是要求掌握散布在环境中任意位置的不相关事实时，有限的表示可以在不完整记录无限环境的情况下取得成功。我们通过“未来任务信息”来形式化这一现象，即相对于可用证据而言，保持决策价值所需的最小保留信息。一个联合决策-后悔状态记录了每个允许任务中各行动之间的后验比较。在紧性假设下，每个保留的表示都可以基于该状态进行校准替换，在不增加信息的情况下保留所有任务后悔；由此产生的信息-后悔曲线等于一个操作性编码率。两个精确的边界支配着有限保存：对于固定的行动数量是弱星闭包，以及当所有有限……（原文摘要在此处被截断）

    arXiv:2611.18538v1 Announce Type: cross  Abstract: An information system often has to learn before it knows what it will be asked. A finite representation can succeed without recording an infinite environment in full when those asks share collective structure, rather than demand unrelated facts scattered arbitrarily across it. We formalize this phenomenon through future-task information, the minimum retained information needed to preserve decision value relative to the available evidence. A joint decision-regret state records posterior comparisons among actions for every permitted task. Under compactness assumptions, every retained representation admits a calibrated replacement based on this state, preserving all task regrets without increasing information; the resulting information-regret curve equals an operational coding rate. Two exact boundaries govern finite preservation: weak-star closure for a fixed action count, and the singular mass of the source-evidence law when all finite 
    
[^8]: ModHarness-RL：将支架重构为可学习、可演化的程序以服务于智能体强化学习

    ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL

    [https://arxiv.org/abs/2611.18533](https://arxiv.org/abs/2611.18533)

    提出 ModHarness-RL，将智能体强化学习中固定不变的手写支架重构为可从数据学习并随训练不断演化的程序，通过压缩程序（组装上下文）和引导程序（行为偏离时提供指导）两个核心组件，从轨迹中提出候选程序并经执行验证而持续进化。

    

    智能体强化学习（RL）日益依赖于一个“支架”，即围绕模型的脚手架，它决定模型能看到自身交互中的哪些内容，以及模型会接收到什么样的引导。大多数现有方法都在一个全程固定不变的手写支架下进行训练。然而，一个有效的支架取决于领域、数据和模型本身，并且会随着模型的训练而持续变化。我们提出 ModHarness-RL，它将支架重构为从数据中学习、并随着模型通过强化学习训练而不断演化的程序。ModHarness-RL 将交互记录为结构化的状态，即一组带类型的块，而支架的两个核心组件就是基于这些块的程序：一个压缩程序，负责将积累的历史组装进模型的上下文中；一个引导程序，在行为发生偏离时提供指导。在强化学习训练过程中，候选程序从最近的轨迹中被提出，通过执行进行验证，并在过去的（轨迹上）进行重放。

    arXiv:2611.18533v1 Announce Type: new  Abstract: Agentic reinforcement learning (RL) increasingly relies on a harness, the scaffold around the model that decides what it sees of its own interaction and what guidance it receives. Most existing approaches train under a hand-written harness that stays fixed throughout. However, an effective harness depends on the domain, the data and the model, and keeps changing as the model trains. We introduce ModHarness-RL, which recasts the harness as programs that are learned from data and evolve as the model trains with RL. ModHarness-RL records the interaction as structured State, a set of typed blocks, and two core components of the harness are programs over these blocks, a Compression program that assembles the accumulated history into the model's context and a Steering program that supplies guidance when behavior deviates. During RL training, candidate programs are proposed from recent trajectories, verified by execution, and replayed over past
    
[^9]: 通过隐藏神经元概念归纳理解农业卷积神经网络（CNN）中的上下文偏差

    Understanding Contextual Bias in Agricultural CNNs Through Hidden-Neuron Concept Induction

    [https://arxiv.org/abs/2611.18530](https://arxiv.org/abs/2611.18530)

    该论文创新性地将概念归纳方法应用于CNN可解释性研究，通过知识图谱编码图像描述符并推导逻辑表达式，揭示了农业CNN隐藏神经元在上下文图像块上激活所对应的可解释视觉属性，从而深入理解CNN利用背景信息的上下文偏差机制。

    

    卷积神经网络（CNN）能够利用主要包含背景、几乎不含或完全不含目标对象的图像块对图像进行分类，这表明它们利用了上下文信息。以往的研究通过分类准确率来检测这种行为，但对CNN隐藏层所表示的内容却知之甚少。我们研究概念归纳是否能揭示与上下文图像块上隐藏神经元激活相关的可解释视觉属性。我们使用五个具有高、中、低上下文偏差的农业数据集，分析了四种CNN架构在三个网络深度和五个图像块位置上的表现。每个图像块的定量图像描述符被编码到知识图谱中，并作为概念归纳的背景知识，由此推导出表征神经元激活的逻辑表达式。这些表达式在留出图像上使用Mann-Whitney U检验和Cliff's delta进行验证。在48,013个测试（摘要原文截断）……

    arXiv:2611.18530v1 Announce Type: cross  Abstract: Convolutional Neural Networks (CNNs) can classify images using background-dominated patches that contain little or none of the target object, showing that they exploit contextual information. Previous studies detected this behavior through classification accuracy but say little about what CNN hidden layers represent. We investigate whether Concept Induction can reveal interpretable visual properties associated with hidden-neuron activations on contextual image patches. Using five agricultural datasets with high, moderate, and low contextual bias, we analyze four CNN architectures at three network depths and five patch locations. Quantitative image descriptors of each patch are encoded in a knowledge graph and used as background knowledge for Concept Induction, which derives logical expressions characterizing neuron activation. These expressions are tested on held-out images using the Mann-Whitney U test and Cliff's delta. Of 48,013 tes
    
[^10]: StreamEmbed：面向屏幕截图检索的从像素到索引的实时编码

    StreamEmbed: Real-Time Indexing from Pixels for Screenshot Retrieval

    [https://arxiv.org/abs/2611.18516](https://arxiv.org/abs/2611.18516)

    StreamEmbed利用屏幕像素的冗余性，通过仅依据像素差异的空间token剪枝和跨截图时间KV缓存复用，实现增量式视觉嵌入，从而对不断到来的屏幕截图进行实时编码与索引，同时保持与全图嵌入相当的精度。

    

    屏幕历史可以作为人类和智能体的可搜索视觉记忆，但与离线语料库不同，它必须在每张截图到达时即时进行编码和索引。基于OCR的索引方式速度快，但会丢失视觉信息；而视觉编码则需要超过两秒的时间。然而屏幕是增量变化的：大多数像素在单张截图内部以及跨截图之间都是重复的。这引出了一个问题：视觉编码器是否需要为每张新截图从头重新计算其表示？我们提出了StreamEmbed，一个利用这种冗余性的增量视觉嵌入系统。它仅由像素差异引导，结合了空间token剪枝（跳过与邻近块重复的块）和时间KV复用（对跨截图未变化的块复用缓存的键和值）。一个小的可学习适配器对剪枝和复用进行补偿，以保持与完整截图嵌入的一致性。在ScreenMemEval——我们的基准测试（包含针对18……的10,426个问题）——上……（摘要在此处截断）

    arXiv:2611.18516v1 Announce Type: cross  Abstract: Screen history can serve as searchable visual memory for people and agents, but unlike an offline corpus, it must be encoded and indexed as each screenshot arrives. OCR-based indexing is fast but loses visual information; visual encoding takes over two seconds. Yet screens change incrementally: most pixels repeat within and across screenshots. This raises a question: do visual encoders need to recompute their representations from scratch for every new screenshot? We present StreamEmbed, a system for incremental visual embedding that exploits this redundancy. Guided solely by pixel differences, it combines spatial token pruning, which skips blocks that repeat a neighbor, with temporal KV reuse, which reuses cached keys and values for blocks unchanged across screenshots. A small learned adapter compensates for pruning and reuse to preserve fidelity to full-screenshot embeddings. On ScreenMemEval, our benchmark of 10,426 questions over 18
    
[^11]: 关于几乎相等分布之间散度的数值问题

    On numerical issues of divergences between nearly equal distributions

    [https://arxiv.org/abs/2611.18511](https://arxiv.org/abs/2611.18511)

    该论文揭示了当类别分布几乎相等时，KL散度、JSD散度及熵等标准散度运算存在与softmax类似的数值精度问题，并提出了数值稳定的重新表述方案及经过属性测试验证的实现。

    

    Barron等人（2026年）最近发现了softmax和软交叉熵损失中的数值问题。我们证明在类别分布之间的标准散度运算中也存在类似的问题，这包括正向和反向Kullback-Leibler（KL）散度、广义Jensen-Shannon散度（JSD），以及蒙特卡洛KL估计和熵。当分布几乎相等时，舍入和抵消问题会影响前向计算和梯度计算。这些问题可能影响在优化目标中使用散度的场景，例如知识蒸馏、强化学习和近端方法。我们提出了数值稳定的重新表述方案，并为每种情况提供了示例实现，并通过基于属性的测试进行了验证。

    arXiv:2611.18511v1 Announce Type: new  Abstract: Barron et al. (2026) recently identified numerical issues in the softmax and soft cross-entropy loss. We show that similar issues are present in standard divergence operations between categorical distributions. This includes the forward and reverse Kullback-Leibler (KL) divergence and generalized Jensen-Shannon divergence (JSD), as well as Monte Carlo KL estimates and entropy. When the distributions are nearly equal, rounding and cancellation issues affect the forward and gradient computations. These issues can affect settings using divergences in optimization objectives, such as knowledge distillation, reinforcement learning, and proximal methods. We propose numerically stable reformulations and provide example implementations for each setting, validated by property-based tests.
    
[^12]: Softmax交叉熵下锥形分类头中的渐近间隔与特征约束

    Asymptotic Margins and Feature Confinement in Conic Classification Heads under Softmax Cross-Entropy

    [https://arxiv.org/abs/2611.18507](https://arxiv.org/abs/2611.18507)

    论文为锥形分类头引入了可由二阶锥规划计算的渐近间隔，并证明该间隔的正负决定了softmax交叉熵训练下特征是否有界：间隔为正时损失仅在特征发散处趋近于零，间隔为负时损失强制且最优解位于一个显式球内。

    

    基于锥函数构建的分类头为每个类别赋予特征空间中的一个有界区域，这些区域可用于拒绝未见类别的样本。我们提出的问题是：当该分类头使用softmax交叉熵训练时，这些区域是否也能使特征保持有界。对于类别得分由线性项与欧几里得范数相结合的分类头，我们首先证明每个类别的区域由一个旋转二次曲面所界定，其离心率是某个学习到的比率的倒数。随后，我们为每个类别定义一个渐近间隔，该间隔比较其自身得分增长与所有其他得分增长的情况，并可通过二阶锥规划进行计算。当且仅当一个类别的间隔为正时，其损失的下确界为零，且损失仅在特征发散时才趋近于零。当间隔为负时，损失是强制的，其极小值点位于一个显式给定的球内。由于softmax仅依赖于差异……（原文摘要至此截断）

    arXiv:2611.18507v1 Announce Type: new  Abstract: Classification heads built on conic functions give each class a bounded region of the feature space, and such regions are used to reject samples of unseen classes. We ask whether these regions also keep the features bounded when the head is trained with softmax cross-entropy. For a head whose class scores combine a linear term with a Euclidean norm, we first show that the region of each class is bounded by a quadric of revolution, whose eccentricity is the reciprocal of one learned ratio. We then attach to each class an asymptotic margin, which compares the growth of its score with the growth of all the other scores and can be computed by a second-order cone program. The loss of a class has infimum zero exactly when its margin is positive, and it then approaches zero only along diverging features. When the margin is negative, the loss is coercive and its minimizers lie in an explicit ball. Since softmax depends only on the differences be
    
[^13]: 基于语言模型先验的交互式程序归纳

    Interactive Program Induction with Language Model Priors

    [https://arxiv.org/abs/2611.18485](https://arxiv.org/abs/2611.18485)

    本文提出以语言模型作为程序先验的贝叶斯交互式程序归纳方法，通过贪心选择最具信息量的输入向预言机查询，逐步将后验概率集中于目标程序，以化解从有限示例归纳程序时固有的歧义性问题。

    

    程序归纳是指学习者从少量给定的输入-输出示例中恢复出程序的过程。这一范式通过让用户直接演示他们希望应用于未见数据上的基于规则的行为，而无需亲自实现该行为，从而降低了编程的门槛。然而，从有限的输入-输出示例中恢复程序在本质上存在歧义性，因为一个有限的输入-输出示例集在理论上与无穷多个程序相容，而这些程序在未见输入上的输出可能各不相同。我们从贝叶斯的视角来审视这一问题：在给定程序先验分布 p⁰ 的情况下，识别目标程序意味着通过观察更多的输入-输出示例，将后验概率质量逐步集中到该程序上。在交互式程序归纳的设定中，学习者会主动选择信息量最大的输入，并向预言机（例如用户）查询其对应输出。我们推导了一种贪心的查询选择算法……

    arXiv:2611.18485v1 Announce Type: cross  Abstract: Program induction refers to the process in which a learner aims to recover a program from a few given input-output examples. This paradigm broadens programming access by letting users demonstrate the rule-based behavior they want to apply to unseen data, instead of having to implement it themselves. However, recovering a program from finite input-output examples is fundamentally ambiguous, as a finite set of input-output examples is theoretically consistent with infinitely many programs whose output diverges on unseen inputs. We view this problem from a Bayesian perspective, where, given a prior $p^0$ over programs, identifying the target program means concentrating posterior probability mass on it by observing additional input-output examples. In the interactive program induction setting, the learner chooses maximally informative inputs and queries an oracle, such as the user, for their outputs. We derive a greedy query-selection algo
    
[^14]: 正则指数族中的Bregman几何：对偶表示与相对光滑性

    Bregman geometry in regular exponential families: dual representations and relative smoothness

    [https://arxiv.org/abs/2611.18480](https://arxiv.org/abs/2611.18480)

    本文基于射影刻画，为正则指数族在期望参数化下建立了一般性的等价对偶Bregman表示构造（含归一化与唯一性结果），使对数密度可表达为对偶坐标系中的普通Bregman散度，并将β-与α-缩放对偶表示作为该框架的特例。

    

    正则指数族分布在统计学中无处不在，因为其具有优美且便利的性质。值得注意的是，这类分布可以用Bregman散度来表示，从而揭示了统计学、凸分析和信息几何之间的基本联系。然而，这一几何结构尚未被系统地用于刻画这些分布相对于其期望参数化的性质。基于射影刻画，我们建立了等价对偶Bregman表示的一般性构造，并给出了归一化与唯一性结果。这些表示将对数密度表达为对偶坐标系中的普通Bregman散度，从而为参数空间提供了几何刻画。所谓的β-与α-缩放对偶表示作为该框架的特殊实例自然涌现。我们进一步推导了共轭……（原摘要在此处截断）

    arXiv:2611.18480v1 Announce Type: cross  Abstract: Regular exponential-family distributions are ubiquitous in statistics due to their appealing and convenient properties. Notably, they can be expressed in terms of Bregman divergences, revealing fundamental connections between statistics, convex analysis, and information geometry. However, this geometric structure has not been systematically exploited to characterize these distributions with respect to their expectation parametrization. Building on a projective characterization, we establish a general construction of equivalent dual Bregman representations, together with normalization and uniqueness results. These representations express log-densities as ordinary Bregman divergences in dual coordinate systems, providing geometric characterizations of the parameter space. So-called $\beta$- and $\alpha$-scaled dual representations emerge as particular instances of this framework. We further derive dual Bregman representations of conjugat
    
[^15]: 基于随机建模与仿真评估反馈驱动的黑盒过程

    Evaluating Feedback-Driven Black-Box Processes via Stochastic Modeling and Simulation

    [https://arxiv.org/abs/2611.18477](https://arxiv.org/abs/2611.18477)

    本文提出了一个基于马尔可夫链和仿真的随机建模框架，用于统一评估反馈驱动的黑盒过程，并通过新颖的模糊测试马尔可夫链模型加以实例化，从而将实证实践与理论联系起来。

    

    我们研究软件工程与机器学习领域中反馈驱动的黑盒过程。这些过程本质上是顺序进行的：策略生成输入、观察系统的输出/反馈，并利用反馈来指导后续输入的设计。虽然许多策略是在对底层系统做出结构假设的前提下开发的，但许多现实世界的系统在缺乏此类知识的情况下运行，因此需要黑盒策略，而这类策略目前缺乏统一的评估框架。我们引入了一个随机框架，利用马尔可夫链和仿真来建模与评估此类策略，从而将实证实践与理论联系起来。我们通过一种新颖的模糊测试马尔可夫链模型，将该框架实例化于基于变异的模糊测试（一种广泛用于通过顺序输入变异发现软件缺陷的技术）中，该模型的转移由变异策略控制，其反馈由程序执行给出。我们开发了一个严格的……（摘要原文在此处截断）

    arXiv:2611.18477v1 Announce Type: cross  Abstract: We study feedback-driven black-box processes across software engineering and machine learning. These processes are inherently sequential: policies generate inputs, observe outputs/feedback from systems, and use the feedback to guide the design of subsequent inputs. While many policies are developed under structural assumptions on the underlying systems, many real-world systems operate without such knowledge, necessitating black-box policies that lack a unified evaluation framework. We introduce a stochastic framework that models and evaluates such policies using Markov chains and simulation, thereby connecting empirical practice with theory. We instantiate the framework in mutation-based fuzzing (widely used for discovering software bugs via sequential input mutation), by a novel Markov chain model of fuzzing whose transitions are governed by the mutation strategy and whose feedback is given by program execution. We develop a rigorous 
    
[^16]: 基于去噪分数匹配的对比潜在表示学习

    Contrastive latent representation learning with denoising score matching

    [https://arxiv.org/abs/2611.18469](https://arxiv.org/abs/2611.18469)

    该论文提出ScoreCLR方法，通过推导香农熵梯度得到涉及潜在表示分数的等价目标，并利用去噪分数匹配技术由轻量级网络与编码器联合训练来估计排斥项，从而以新颖方式克服了对比学习中排斥力估计依赖大批量、高方差和维度灾难的问题。

    

    对比表示学习方法通过将同一样本的不同视图拉近，同时将其与不同样本的视图推开来进行学习。一个根本性的挑战在于对排斥力的估计，这通常需要较大的批处理规模，可能表现出高方差，并且容易受到维度灾难的影响。我们提出了ScoreCLR，这是一种对比表示学习方法，为估计这一排斥项提供了一种新颖的方式。我们考虑了通过最大化样本两个潜在视图之间互信息来进行表示学习的框架。通过推导香农熵关于编码器参数的梯度，我们得到了一个涉及潜在表示分数的等价目标函数。随后，我们使用去噪分数匹配技术，通过一个与编码器联合训练的轻量级网络来估计该分数。我们还进一步提出了正则化方法（摘要在此处被截断）。

    arXiv:2611.18469v1 Announce Type: cross  Abstract: Contrastive representation learning methods operate by pulling different views of the same data sample together while pushing them away from views of different samples. A fundamental challenge is the estimation of the repulsive force, which typically requires large batch sizes, can exhibit high variance, and is susceptible to the curse of dimensionality. We introduce ScoreCLR, a contrastive representation learning method that offers a novel way to estimate this repulsion term. We consider the framework of representation learning by maximizing the mutual information between two latent views of a sample. By deriving the gradient of the Shannon entropy with respect to the encoder parameters, we obtain an equivalent objective that involves the score of the latent representation. We then use denoising score matching techniques to estimate this score with a lightweight network that is trained jointly with the encoder. We further propose regu
    
[^17]: 基于得分函数的神经常微分方程求解反应扩散方程

    Score-based neural ordinary differential equations for reaction-diffusions

    [https://arxiv.org/abs/2611.18467](https://arxiv.org/abs/2611.18467)

    该论文提出了一种带物理权重的拉格朗日式得分驱动神经常微分方程方法，让粒子通过权重追踪质量变化，并以质量归一化的自洽残差作为训练目标，从而有效求解反应扩散方程。

    

    扩散方程可以被改写为由演化密度的得分函数所驱动的输运方程。传统的概率流表述利用这种由得分驱动的速度确定性地移动粒子，通常保持总质量守恒。在本工作中，我们针对反应扩散方程开发了一种拉格朗日方法，其中每个粒子同时携带一个位置和一个物理权重。特征常微分方程将源项纳入对数密度及其梯度的演化之中，而粒子权重则跟踪质量的变化。我们将这种表示与学习得到的速度场以及加权粒子训练相结合，以近似这些动力学。对于非递增的反应速率，一个相对熵不等式支持将质量归一化的自洽残差作为训练目标。当增长率有界时，该目标能够控制所诱导密度的误差。该算法将时间…（摘要原文在此处截断）

    arXiv:2611.18467v1 Announce Type: cross  Abstract: A diffusion equation can be rewritten as a transport equation driven by the score function of the evolving density. Conventional probability-flow formulations, which move particles deterministically with this score-driven velocity, typically conserve total mass. In this work, we develop a Lagrangian method for reaction-diffusion equations in which each particle carries a position and a physical weight. The characteristic ODEs incorporate source terms into the evolution of the log-density and its gradient, while particle weights track changes in mass. We combine this representation with a learned velocity field and weighted particle training to approximate these dynamics. For non-increasing reaction rates, a relative-entropy inequality supports the mass-normalized self-consistency residual as the training objective. This objective can control the error of the induced density when the growth rate is bounded. The algorithm divides the tim
    
[^18]: 基于结果感知的鲁棒序贯学习实现高熵合金强度的样本高效预测

    Sample-efficient prediction of high-entropy alloy strength through outcome-aware robust sequential learning

    [https://arxiv.org/abs/2611.18462](https://arxiv.org/abs/2611.18462)

    本文提出ROSE-HEA框架，通过预判实验结果对预测的更新影响并评估最坏情况下的期望预测风险来选择信息量最大的候选实验，从而以更少的拉伸试验数据高效预测高熵合金的抗拉强度。

    

    准确预测高熵合金（HEA）的强度可以指导成分设计与加工工艺决策；然而，获取高熵合金的拉伸试验数据成本高昂且实验要求苛刻。本研究提出了面向高熵合金的鲁棒结果感知序贯实验框架（ROSE-HEA），这是一个根据合金成分和加工条件预测抗拉强度（UTS）的样本高效框架。其采集策略会预判各种可能的UTS结果将如何更新预测以及各互补高斯过程（GP）描述的支持程度，随后在参考域上重新评估剩余的最坏情况期望预测风险，从而选择信息量最大的候选点。在一项基于501条高熵合金记录的回顾性评估中，研究在30次共享相同初始数据的匹配运行下，将ROSE-HEA与期望改进（EI）和置信上界（UCB）贝叶斯优化（BO）、随机GP采样以及八种传统回归器进行了比较。

    arXiv:2611.18462v1 Announce Type: new  Abstract: Accurate prediction of high-entropy alloy (HEA) strength can guide composition and processing decisions; however, obtaining tensile-test data for HEAs can be costly and experimentally demanding. This study proposes Robust Outcome-aware Sequential Experimentation for High-Entropy Alloys (ROSE-HEA), a sample-efficient framework for predicting ultimate tensile strength (UTS) from alloy composition and processing conditions. Its acquisition policy anticipates how possible UTS outcomes update predictions and the support for complementary Gaussian process (GP) descriptions. It then reassesses the worst-case expected predictive risk remaining across a reference domain to select informative candidates. A retrospective evaluation on 501 HEA records compares ROSE-HEA with expected-improvement (EI) and upper-confidence-bound (UCB) Bayesian optimization (BO), random GP sampling, and eight conventional regressors across 30 matched runs with shared in
    
[^19]: Callo：在预填充-解码分离架构下利用失效KV进行投机解码以服务缓存未命中请求

    Callo: Serving Cache-Miss Requests with Speculative Decoding on Invalidated KV under Prefill-Decode Disaggregation

    [https://arxiv.org/abs/2611.18459](https://arxiv.org/abs/2611.18459)

    Callo是一种预填充-解码分离的服务系统，通过利用失效KV缓存进行投机解码草稿生成并在预填充后用精确KV验证草稿，将缓存未命中请求的处理速度提升1.05-1.19倍。

    

    智能体会话可能涉及频繁的上下文编辑，这会使缓存的KV状态失效，迫使请求在解码前等待重新预填充。我们提出了Callo，一种分离式服务系统，它利用失效的KV生成投机草稿以减少这种延迟，同时在预填充完成后、提交token之前，用精确的KV对草稿进行验证。Callo通过在解码资源利用率不足的时期调度投机计算，并在草稿可能不被接受时抢占投机计算，从而有效利用解码资源。我们在双GPU测试平台上使用三个模型系列的SWE-agent会话对Callo进行了评估，观察到在中等负载下所有请求相比vLLM获得1.05-1.19倍的加速。我们通过由真实世界轨迹驱动的集群级仿真证明，这些收益在大规模场景下同样能够实现。

    arXiv:2611.18459v1 Announce Type: cross  Abstract: Agent sessions may involve frequent context edits, which effectively invalidate cached KV state and force requests to wait for re-prefill before decoding. We present Callo, a disaggregated serving system that leverages invalidated KV for speculative drafts to reduce this latency, while verifying drafts against the exact KV after prefill before committing tokens. Callo makes effective use of decode resources by scheduling speculations during periods of underutilization and preempting speculations when drafts are likely to not be accepted. We evaluated Callo on a two-GPU testbed using SWE-agent sessions across three model families, observing all-request speedups of 1.05-1.19x over vLLM at moderate load. We demonstrate that these gains are also realized at scale through cluster-level simulations driven by real-world traces.
    
[^20]: 基于集成之集成的大语言模型黑盒不确定性量化

    Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles

    [https://arxiv.org/abs/2611.18453](https://arxiv.org/abs/2611.18453)

    该论文提出了一种仅需黑盒访问的两级“集成之集成”方法，通过对固定提示词随机解码构成内层集成、对提示词进行语义扰动构成外层集成，并利用全协方差定律在嵌入空间中精确分解大语言模型的偶然性不确定性与认知性不确定性。

    

    可靠地使用大语言模型（LLM）需要不确定性估计，以区分提示词固有的模糊性与真正的知识缺口。贝叶斯方法和深度集成方法在原则上能够提供这种偶然性/认知性的分解，但在LLM规模上计算成本过高，且需要白盒访问。相比之下，现有的黑盒方法只能将其总结为单一标量，无法做出上述区分。我们引入了一种两级集成方法来弥合这一差距。对固定提示词进行随机解码构成内层集成；对提示词进行保义语义扰动构成外层集成。变异性在连续嵌入空间中进行测量，全协方差定律给出了精确的扰动内/扰动间分解，从而估计偶然性不确定性与认知性不确定性（AU/EU）；该方法仅需要黑盒采样访问。在此估计器的基础上，我们提出了三项理论结果（摘要在此处被截断）

    arXiv:2611.18453v1 Announce Type: cross  Abstract: Reliable use of large language models (LLMs) requires uncertainty estimates that separate ambiguity intrinsic to a prompt from genuine knowledge gaps. Bayesian and deep-ensemble methods deliver this aleatoric/epistemic decomposition in principle, but are computationally prohibitive at LLM scale and require white-box access. Existing black-box methods, in contrast, summarize to a single scalar that cannot make the distinction. We introduce a two-level ensemble that bridges this gap. Stochastic decoding of a fixed prompt forms the inner ensemble; meaning-preserving semantic perturbations of the prompt form the outer ensemble. Variability is measured in a continuous embedding space, and the law of total covariance gives an exact within-/between-perturbation decomposition to estimate aleatoric and epistemic uncertainty (AU/EU); it requires only black-box sampling access. On top of this estimator, we contribute three theoretical results tha
    
[^21]: PACT：一种解码器置信度引导的两轮接收机，用于亚太赫兹OFDM中信道与相位噪声的联合估计

    PACT: A Decoder-Confidence-Guided Two-Round Receiver for Joint Channel and Phase-Noise Estimation in Sub-THz OFDM

    [https://arxiv.org/abs/2611.18450](https://arxiv.org/abs/2611.18450)

    该论文提出了一种解码器置信度引导的两轮接收机PACT，利用导频感知复数变换器联合估计信道与相位噪声，在亚太赫兹OFDM系统中将码块错误率显著降低约57%-70%。

    

    频率选择性衰落和振荡器相位噪声（PN）共同影响编码的亚太赫兹OFDM接收。我们提出了一种解码器置信度引导的两轮接收机，结合了导频感知复数变换器（PACT）估计器、带状线性最小均方误差均衡和神经归一化最小和解码。PACT通过相互作用的复数值分支联合估计频域信道和有效时域相位噪声。解码器导出的伪导频和连续的星座感知置信度通过PACT输入和注意力机制引导第二轮细化，且参数在各轮之间共享。在E_b/N_0=30 dB时，PACT接收机在两个基准设置中分别比性能最好的基于学习的基线降低了约70%和57%的码块错误率。冻结权重测试进一步证明了该方法在不同信道/相位噪声模型、载波频率和FFT/带宽配置下的适用性。

    arXiv:2611.18450v1 Announce Type: new  Abstract: Frequency-selective fading and oscillator phase noise (PN) jointly impair coded sub-THz OFDM reception. We propose a decoder-confidence-guided two-round receiver combining a {pilot-aware complex transformer} (PACT) estimator, banded linear minimum mean-square error equalization, and neural normalized min-sum decoding. PACT jointly estimates the frequency-domain channel and effective time-domain PN through interacting complex-valued branches. Decoder-derived pseudo-pilots and continuous constellation-aware confidence guide second-round refinement through PACT inputs and attention, with parameters shared across rounds. At $E_b/N_0=30$~dB, the PACT receiver achieves approximately $70\%$ and $57\%$ lower code-block error rates than the best-performing learning-based baseline in the two benchmark settings, respectively. Frozen-weight tests further demonstrate applicability across channel/PN models, carrier frequencies, and FFT/bandwidth confi
    
[^22]: 状态并非瓶颈：三模式组播网格上具有分页状态的Gated-DeltaNet硬件单元，从RTL到硅片位级精确

    The State Is Not the Bottleneck: A Paged-State Gated-DeltaNet Tile on a Tri-Mode Multicast Mesh, Bit-Exact from RTL to Silicon

    [https://arxiv.org/abs/2611.18422](https://arxiv.org/abs/2611.18422)

    在Agilex 5 FPGA上实现的Gated-DeltaNet单层硬件单元表明，尽管循环状态需分页至片外存储，分页开销仅占解码总时间的11.2%，状态并非瓶颈，矩阵-向量乘法才是主要耗时部分。

    

    线性注意力语言模型用一个固定大小的循环状态取代了不断增长的键值缓存。在边缘FPGA上，这一状态可能超出片上存储容量，必须分页到外部存储器。本文探讨这种分页是否会主导解码（生成token）的成本。我们构建了一个硬件计算单元，在Agilex 5 FPGA上计算定制的3.69亿参数混合Gated DeltaNet模型的完整一层，并对其循环状态进行分页。模型状态需要6.2 MiB，而器件仅有0.87 MiB的块存储器。分页在每个解码token增加26.0 MB的内存流量，最多占计算总量的11.2%。在75 MHz的板卡上使用合成权重时，该层处理一个token需要11.15毫秒。与权重的矩阵-向量乘法约占该时间的62%，而状态阶段（包括分页）占13%。该计算单元是三模式处理元件组播网格中的一个节点。我们测量……

    arXiv:2611.18422v1 Announce Type: cross  Abstract: Linear-attention language models replace a growing key-value cache with a recurrent state of fixed size. On an edge FPGA, this state can exceed the on-chip memory and must be paged to external memory. This paper asks whether such paging dominates the cost of decoding (generating tokens). We built a hardware tile that computes one complete layer of a custom 369-million-parameter hybrid Gated DeltaNet model on an Agilex 5 FPGA and pages its recurrent state. The model state needs 6.2 MiB, whereas the device has 0.87 MiB of block memory. Paging adds 26.0 MB of memory traffic per decoded token, at most 11.2 percent of the calculated total. On the board at 75 MHz, with synthetic weights, one token takes 11.15 ms in this layer. The matrix-vector products with the weights take about 62 percent of this time and the state phase, including the paging, 13 percent. The tile is one node of a multicast mesh of tri-mode processing elements. We measure
    
[^23]: 用于无序原子结构可解释生成与逆向设计的图扩散Transformer

    Graph diffusion transformer for interpretable generation and inverse design of disordered atomic structures

    [https://arxiv.org/abs/2611.18420](https://arxiv.org/abs/2611.18420)

    提出GaDiT图扩散Transformer生成模型，通过动态周期图、掩码条件化与区间加权噪声分布，实现无序原子结构的高质量生成、采样速度提升两个数量级，并支持统一模型下的多条件生成与逆向设计。

    

    原子无序性为调控材料性能提供了一个强有力的自由度，但由此产生的庞大构型空间掩盖了结构-性能关系，阻碍了理性的逆向设计。在此，我们提出了图无序扩散Transformer（GaDiT），这是一种结合了动态周期图、掩码条件化和区间加权训练噪声分布的生成模型。GaDiT能够生成高质量的无序结构，同时与现有的基于校正器的方法相比，将采样速度提升两个数量级，这一点在二维非晶碳以及三维非晶硅和二氧化硅上得到了验证。此外，GaDiT在统一模型中支持基于单一或组合的结构特征、物理性质和制备条件的条件生成，并实现了将可及的结构-性能范围扩展至训练分布之外的逆向设计。

    arXiv:2611.18420v1 Announce Type: cross  Abstract: Atomic disorder offers a potent degree of freedom to tailor material properties, yet the resulting vast configuration space obscures structure-property relations and hinders rational inverse design. Here we introduce Graph Amorphous Diffusion Transformer (GaDiT), a generative model that combines dynamic periodic graphs, masked conditioning and an interval-weighted training-noise distribution. GaDiT generates high-quality disordered structures while accelerating sampling by two orders of magnitude compared with existing corrector-based approaches, as demonstrated across two-dimensional amorphous carbon and three-dimensional amorphous silicon and silica. Moreover, GaDiT supports conditional generation under single or combined structural features, physical properties and preparation conditions in a unified model, and enables inverse design that expands accessible structure-property ranges beyond the training distribution. To interpret the
    
[^24]: CoPre-MoE：面向多模态持续指令微调的专家混合模型整合与保留方法

    CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2611.18417](https://arxiv.org/abs/2611.18417)

    提出CoPre-MoE方法，通过为每个任务先学习任务特定MoE专家，再对跨任务冗余专家进行整合与保留，在避免灾难性遗忘的同时有效抑制多模态持续指令微调中的跨任务参数冗余和参数线性增长。

    

    多模态持续指令微调要求多模态大语言模型在依次获取新能力的同时不发生灾难性遗忘。专家混合（MoE）架构通过专家专业化与组合，为缓解任务间干扰提供了一种有前景的途径。然而，现有基于MoE的方法通过为每个任务分配独立专家来避免覆盖共享参数，但这种渐进式扩展会导致严重的跨任务冗余和参数线性增长。具体而言，不同任务的专家在学习任务特定能力的同时，往往会学到重叠的功能。例如，ScienceQA和TextVQA都需要定位相关的图像区域。为了在不引发遗忘的情况下抑制参数冗余，我们提出了整合与保留MoE（CoPre-MoE）。对于每个到来的任务，CoPre-MoE首先学习任务特定的MoE以捕捉专门化能力，从而消除冗余并减少参数……（摘要不完整）

    arXiv:2611.18417v1 Announce Type: cross  Abstract: Multimodal continual instruction tuning requires multimodal large language models to sequentially acquire new capabilities without catastrophic forgetting. Mixture-of-experts (MoE) architectures offer a promising way to mitigate task interference through expert specialization and combination. However, while existing MoE-based methods avoid overwriting shared parameters by allocating separate experts for each task, this progressive expansion incurs severe cross-task redundancy and linear parameter growth. Specifically, experts across tasks often learn overlapping functions alongside task-specific capabilities. For instance, ScienceQA and TextVQA both locate relevant image regions. To curb parameter redundancy without inducing forgetting, we propose Consolidating and Preserving MoE (CoPre-MoE). For each incoming task, CoPre-MoE first learns task-specific MoE to capture specialized capabilities. To eliminate redundancy and reduce paramete
    
[^25]: Fuse-PINN：结合有限元方法与物理信息神经网络，实现非线性偏微分方程的大规模并行模拟

    Fuse-PINN: Combining Finite Element Methods and Physics-Informed Neural Networks to enable massively parallel simulations of non-linear PDEs

    [https://arxiv.org/abs/2611.18409](https://arxiv.org/abs/2611.18409)

    本文提出Fuse-PINN框架，将有限元方法与物理信息神经网络相结合，实现了非线性偏微分方程在高性能计算架构上的可扩展大规模并行模拟，打破了“PINNs无法扩展”的固有观念。

    

    近年来，将物理信息神经网络（PINNs）应用于偏微分方程建模的研究取得了显著进展，形成了丰富的新型方法生态系统。然而，一个常见的局限在于PINNs无法在高性能计算（HPC）架构上进行横向扩展，以应对更具挑战性的问题，尤其是涉及大空间区域内具有湍流动力学的非线性物理模型。典型研究往往将演示局限于玩具模型——物理方程简单、变量较少，且很少涉及在长时间尺度上演化的复杂非线性动力学。AI领域的一部分人甚至接受了“PINNs无法扩展”这一观点。本工作证明了事实恰恰相反。我们提出了一种数学上稳健的面向神经网络的有限元方法（FEM-NN）公式化表达，并将其实现为一个全面且模块化的框架：Fuse-PINN。该公式化表达为物理信息神经网络的可扩展并行化奠定了基础。

    arXiv:2611.18409v1 Announce Type: new  Abstract: Research in the domain of Physics-Informed Neural Networks applied to PDE modelling has advanced significantly in recent years, providing a wide ecosystem of novel methods. However, one frequent limitation is the inability of PINNs to scale out on High-Performance Computing architectures to address more ambitious challenges, particularly non-linear physics models with turbulent dynamics in large spatial domains. Typical studies restrain demonstrations to toy models with simple physics equations, few variables, and rarely any complex non-linear dynamics evolved over long timescales. Part of the AI community has even adopted the idea that "PINNs don't scale". This work proves otherwise. We present a mathematically robust formulation of Finite-Element-Method for Neural Networks (FEM-NN) implemented into a comprehensive, modular framework: Fuse-PINN. This formulation provides the basis for a sclable parallelisation of Physics-Informed Neural
    
[^26]: 深度神经网络训练效率：Grokking（顿悟）现象会改变其动态吗？

    Deep Neural Networks Training Efficiency: Does Grokking Modify its Dynamics?

    [https://arxiv.org/abs/2611.18396](https://arxiv.org/abs/2611.18396)

    本文通过在模运算和MNIST任务上以多种数值精度进行长短时程训练实验，利用Orthograd诱发grokking并系统考察其与训练效率（单位能量准确率）的关系，发现grokking的发生并未显著改变训练效率通常递减的动态规律。

    

    我们研究了grokking（顿悟）现象——即在明显过拟合之后延迟出现的分布外泛化能力——是否会改变深度神经网络的训练效率，其中效率被定义为单位能量所获得的准确率。在模运算和MNIST两个经典实验场景中，我们分析了两个神经网络模型在16位、32位和64位精度下的表现，并使用Orthograd来检验grokking是否会使训练效率偏离其通常递减的动态规律。我们同时考察了长达500个epoch的短时程训练和长达20,000个epoch的长时程训练，并在两种情况下都将grokking识别为在持续的训练-测试差距之后测试准确率的急剧提升。在短时程实验中，在训练早期施加强度逐渐增强的Orthograd并未诱发grokking，训练效率仍保持平滑递减。在长时程实验中，Orthograd成功诱发了grokking，但……

    arXiv:2611.18396v1 Announce Type: new  Abstract: We study whether grokking, the delayed emergence of out-of-distribution generalization after apparent overfitting, alters training efficiency in deep neural networks, where efficiency is measured as accuracy per unit energy. Using two canonical settings, modular arithmetic and MNIST, we analyze two neural network models under 16-, 32-, and 64-bit precision and use Orthograd to test whether grokking produces noticeable deviations from the typically decreasing dynamics of training efficiency. We consider both short training runs of up to 500 epochs and long training runs of up to 20,000 epochs, and in both cases identify grokking as a sharp improvement in test accuracy after a sustained train-test gap. In the short-horizon experiments, applying Orthograd at progressively stronger levels during early training does not induce grokking, and efficiency remains smoothly decreasing. In the long-horizon experiments, Orthograd induces grokking, bu
    
[^27]: 生成建模中的Lipschitz约束判别器：渐近行为与支撑不匹配

    Lipschitz-Constrained Discriminators in Generative Modeling: Asymptotic Behavior and Support Mismatch

    [https://arxiv.org/abs/2611.18394](https://arxiv.org/abs/2611.18394)

    本文刻画了生成建模中带 Lipschitz 约束的最优判别器在正则化参数 $L$ 趋于无穷时的渐近行为，涵盖了绝对连续且 Radon-Nikodym 导数有界、以及离散分布等多种情形。

    

    我们研究了与两个概率分布 $Q$ 和 $P$ 的 $L$-Lipschitz 正则化 KL 散度相关联的最优判别器的渐近行为，该散度通过参数 $L>0$ 在缩放的 Wasserstein-1 度量与 KL 散度之间进行插值。参数 $L$ 起着 Lipschitz 约束的作用，并被视为一个正则化参数：较小的 $L$ 值强制判别器更具几何感知能力，而随着 $L$ 的增大，几何正则化被逐渐放松。已知结果 [3] 表明，当 $L\to\infty$ 时，这类散度收敛于 KL 散度。在本文中，我们刻画了最优判别器在 $L\to\infty$ 时在不同情形下的渐近行为：首先，在 $Q\ll P$ 且 Radon-Nikodym 导数 $\frac{dQ}{dP}$ 有上界且远离零的情况下；其次，当 $Q$ 和 $P$ 是满足 $Q\ll P$ 的离散分布时；最后，当 $Q$……

    arXiv:2611.18394v1 Announce Type: cross  Abstract: We study the asymptotic behavior of optimal discriminators associated to $L-$Lipschitz regularized KL-divergences of two probability distributions $Q$ and $P$ that interpolate between the rescaled Wasserstein$-1$ metric by some parameter $L>0$ and the KL divergence. The parameter $L$ plays the role of a Lipschitz constraint and is viewed as a regularization parameter: small values of $L$ enforce a more geometry-aware discriminator, while as $L$ increases the geometric regularization is progressively relaxed. It is a known result [3] that such divergences converge to the KL divergence as $L\to\infty$. In this paper, we characterize the asymptotic behavior of the optimal discriminators as $L\to\infty$ in different regimes: Firstly, under $Q\ll P$ and the Radon-Nikodym derivative $\frac{dQ}{dP}$ is bounded from above and bounded away from zero. Secondly, when $Q$ and $P$ are discrete distributions satisfying $Q\ll P$, and finally when $Q$
    
[^28]: WeatherDreamer：从多模态地球观测中学习世界模型

    WeatherDreamer: Learning a World Model from Multimodal Earth Observations

    [https://arxiv.org/abs/2611.18393](https://arxiv.org/abs/2611.18393)

    WeatherDreamer是一个无需再分析数据、从18个全球观测系统端到端学习递归滤波循环的世界模型，其集合预报在12小时内的温度、露点和风速预测上优于NOAA的GEFS。

    

    机器学习天气模型如今已能媲美业务化的全球预报，但它们仍需以由独立且昂贵的数据同化（DA）系统生成的分析场作为起点，而全球模型的数据同化系统通常每六小时才更新一次。观测到观测的模型避免了这种依赖，但大多数模型忽略了递归滤波循环——正是这一循环使数据同化系统能够借助稀疏且不规则的观测来延续背景状态。我们提出了WeatherDreamer，这是一个从18个全球观测系统中端到端学习该循环的世界模型，无需依赖再分析数据。它从观测中推断潜状态，利用学习到的动力学逐小时向前传播，每小时用新的观测对其进行校正，并能预测其任意观测系统在任何位置会测量到什么。在留出的站点上，其集合预报在温度、露点和风速方面在未来12小时内优于NOAA的全球集合预报系统（GEFS），且比GEFS和E（原文截断）具有更好的校准性。

    arXiv:2611.18393v1 Announce Type: new  Abstract: Machine learning weather models now rival operational global forecasts but still start from analyses produced by separate, expensive data assimilation (DA) systems, which for global models are typically updated every six hours. Observation-to-observation models avoid this dependence, but most omit the recursive filtering cycle that lets DA systems carry a background state through sparse, irregular observations. We introduce WeatherDreamer, a world model that learns this cycle end-to-end from 18 global observing systems, without reanalysis. It infers a latent state from observations, carries it forward hour by hour with learned dynamics, corrects it each hour with new observations, and can predict what any of its observing systems would measure at any location. At held-out stations, its ensembles beat NOAA's Global Ensemble Forecast System (GEFS) for temperature, dew point and wind out to 12 h and are better calibrated than GEFS and the E
    
[^29]: 物理信息神经网络与机理模型在流行病建模和预测中的系统性比较

    A systematic comparison of physics-informed neural networks and mechanistic models for epidemic modeling and forecasting

    [https://arxiv.org/abs/2611.18389](https://arxiv.org/abs/2611.18389)

    本研究在统一的SEIRD框架下系统比较了物理信息神经网络（PINNs）与随机机理模型在流行病建模中的表现，发现PINNs虽然在合成场景中表现欠佳，但在真实COVID-19疫情数据上预测性能更优，且能提供更窄的不确定性区间和更精确的参数估计。

    

    物理信息神经网络近来已被应用于流行病建模和预测，并取得了令人鼓舞的成果。这些方法将机理知识与数据驱动的学习相结合。然而，它们与成熟的、更简单的流行病模型相比的性能尚未得到充分探索。在此，我们对基于PINN的框架和随机机理模型进行了系统性比较。两种方法在刻意简化的设置中共享相同的年龄分层SEIRD核心和滚动窗口预测流程。我们在两个合成场景和伦敦第一波COVID-19疫情中评估了它们的回顾性拟合和概率预测。有趣的是，虽然PINNs在合成场景中表现不佳，但它们在真实数据上取得了更好的性能。此外，PINNs在合成场景中提供了更窄的不确定性区间和更精确的参数估计。我们的研究结果证实PINNs是一种有前景的工具。

    arXiv:2611.18389v1 Announce Type: cross  Abstract: Physics-Informed Neural Networks (PINNs) have recently been applied to epidemic modeling and forecasting with promising results. These methods integrate mechanistic knowledge with data-driven learning. Yet, their comparative performance against well-established, and simpler epidemic models is underexplored. Here, we conduct a systematic comparison of a PINN-based framework and a stochastic mechanistic model. Both approaches share the same age-structured SEIRD core and rolling-window forecasting pipeline, in a deliberately minimal setting. We evaluate their retrospective fits and probabilistic forecasts in two synthetic scenarios and the first COVID-19 wave in London. Interestingly, while PINNs underperform in synthetic scenarios, they achieve better performance on real data. Furthermore, PINNs provide narrower uncertainty intervals and sharper estimation of parameters in synthetic scenarios. Our findings confirm PINNs as a promising to
    
[^30]: 树集成在哪里做出决策：基于决策谓词图的结构化局部解释

    Where Tree Ensembles Decide: Structural Local Explanation with Decision Predicate Graphs

    [https://arxiv.org/abs/2611.18382](https://arxiv.org/abs/2611.18382)

    该论文提出局部决策谓词图，将树集成对单个样本的预测计算记录为可验证的执行规则集合，并从中识别出关键谓词（即一个具体的阈值及共享它的树），使用户无需外部解释器即可定位模型决策的决定性因素。

    

    树集成是表格数据上最准确的模型之一，但单次预测背后的计算分散在数百条决策规则之中。特征归因和反事实方法通过特征的影响或输入的变化来解释此类预测，却未能说明预测究竟响应的是树所应用的哪些阈值，而结构化解释虽然压缩了这些规则，却丢失了其背后的树结构。为弥补这一空白，我们提出了局部决策谓词图，它将针对某个样本执行的所有规则收集为一个可验证的计算记录，其类别支持能够复现每一个被解释的预测。从这些规则中，我们选出“关键谓词”，即根据估计，其阈值被越过时会使模型最强烈地趋向所选类别的已执行谓词，从而为用户提供一个具体的阈值以及共享该阈值的树，而无需任何外部解释器。

    arXiv:2611.18382v1 Announce Type: new  Abstract: Tree ensembles are among the most accurate models for tabular data, yet the computation behind a single prediction is spread over hundreds of decision rules. Feature attributions and counterfactuals explain such predictions through the influence of features or a change of the input, leaving open which of the thresholds applied by the trees a prediction responds to, while structural explanations compress these rules and lose the trees behind them. To address this gap, we propose a local Decision Predicate Graph (local DPG), which gathers the rules executed for a sample into a verifiable record of the computation, with a class support that reproduces every explained prediction. From these rules we select the critical predicate, the executed predicate whose crossing is estimated to move the model most strongly toward a chosen class, which offers the user a concrete threshold and the trees that share it, without an external explainer. Across
    
[^31]: 结构使量子电路可训练

    Structure makes quantum circuits trainable

    [https://arxiv.org/abs/2611.18378](https://arxiv.org/abs/2611.18378)

    该论文提出了一个通过分解紧致李群幺正表示来界定任意输入态和可观测量下梯度方差的分析框架，据此给出判定量子电路是否具有贫瘠高原的方法，并证明了量子电路可训练性的充分条件。

    

    变分量子算法的可扩展性受到贫瘠高原的限制——在贫瘠高原中，梯度随系统规模呈指数级消失。这一最坏情况的结果对量子AI模型的可训练性提出了根本性问题。我们通过回答“一个给定的量子电路是否具有贫瘠高原”这一问题，在解决可训练性难题上取得了重大进展。我们的方法是采用一个分析框架：通过分解相关紧致李群的幺正表示，为任意输入态 ρ 和可观测量 ℳ 的梯度方差给出界定。当被求导门两侧的电路段形成 2-设计时，我们精确求得了梯度方差，并证明了可训练性的一个充分条件：若 ρ^{⊗2} 与 ℳ^{⊗2} 充分集中于同一个多项式维数的不可约不变子空间，且被求导的门以逆……（原文摘要在此处被截断）

    arXiv:2611.18378v1 Announce Type: cross  Abstract: The scalability of variational quantum algorithms is limited by barren plateaus, where gradients vanish exponentially in system size. This worst-case result raises fundamental questions about the trainability of quantum AI models. We present a major advance towards resolving this trainability question by answering whether a given quantum circuit has a barren plateau. We do this using an analytical framework that bounds the gradient variance for arbitrary input states $\rho$ and observables $\mathcal{M}$ by decomposing the unitary representation of the associated compact Lie group. When the circuit segments on both sides of a gate form $2$-designs, we obtain the variance exactly and prove a sufficient condition for trainability: if $\rho^{\otimes 2}$ and $\mathcal{M}^{\otimes 2}$ are sufficiently concentrated on the same irreducible invariant subspace of polynomial dimension, and the differentiated gate moves this subspace by an inverse
    
[^32]: 视觉-语言-动作模型：设计、训练与部署的全生命周期综述

    Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment

    [https://arxiv.org/abs/2611.18372](https://arxiv.org/abs/2611.18372)

    这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。

    

    视觉-语言-动作（VLA）模型将网络预训练的视觉-语言模型适配为输出机器人动作，从而使网络知识得以迁移到物理控制中。在不到4年的时间里，该领域从首个VLA模型RT-2，发展到可在单张消费级GPU上微调的开源模型，再到能够清扫陌生住宅的专有通用模型。本综述回顾了2022年12月至2026年10月间的88项研究工作，主要聚焦于机器人操作与导航，并涵盖人形机器人控制、世界-动作模型与评估。我们沿着七个维度追踪VLA从模型设计到训练与部署的全生命周期：模型设计涵盖，输入与记忆，即模型观察什么；架构，即其流水线如何构建；动作输出，即它如何产生运动指令；中间预测，即它还预测什么，从推理步骤到未来观测。训练涵盖——预训练数据，即机器人、网络、人类与仿真数据的混合（注：原文摘要在此处截断）。

    arXiv:2611.18372v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated 
    
[^33]: 基于大语言模型的f散度自我博弈表格异常检测

    f-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models

    [https://arxiv.org/abs/2611.18370](https://arxiv.org/abs/2611.18370)

    提出DiSPaT，一个基于f散度最小化的自我博弈微调框架，通过交替优化缩小模型生成样本与真实正常数据之间的分布差距，从而增强大语言模型对表格数据正常模式的理解，提升表格异常检测性能。

    

    表格数据中的异常检测由于异构特征类型——混合了数值型、类别型和文本型属性——而面临重大挑战，这使得学习有意义的“正常性”表示变得困难。近期的研究通过将表格行序列化为文本序列，将大语言模型（LLM）应用于这一问题，然而这些方法依赖一次性有监督微调，只能提供有限的信号来收紧模型对正常性的描述。我们提出了DiSPaT，这是一个自我博弈微调框架，用于强化模型对正常数据的理解。基于f散度最小化的理论基础，我们推导出一个紧密的近似，将我们的训练目标与缩小真实正常数据与模型生成样本之间的分布差距联系起来。DiSPaT通过交替优化进行运作：在每次迭代中，当前策略生成合成样本作为伪异常……

    arXiv:2611.18370v1 Announce Type: cross  Abstract: Anomaly detection in tabular data poses significant challenges due to heterogeneous feature types-mixing numerical, categorical, and textual attributes, which complicate learning meaningful representations of normality. Recent work has applied large language models (LLMs) to this problem by serializing table rows as text sequences, yet these approaches rely on one-shot supervised fine-tuning that offers limited signal to tighten the model's description of normality. We propose DiSPaT, a self-play fine-tuning framework that strengthens the model's understanding of normal data. Building on the theoretical foundation of f-divergence minimization, we derive a tight approximation connecting our training objective to reducing the distributional gap between real normal data and model-generated samples. DiSPaT operates through an alternating optimization: at each iteration, the current policy generates synthetic samples that serve as pseudo-an
    
[^34]: DECODE：用于受控序贯决策评估的可分解环境

    DECODE: Decomposable Environments for COntrolled Sequential-Decision Evaluation

    [https://arxiv.org/abs/2611.18367](https://arxiv.org/abs/2611.18367)

    提出了DECODE基准，通过共享已知动力学矩阵、可在记忆与噪声参数上分解控制的消费信贷催收环境，以受控实验方法评估规划范围和信息可用性等修改对序贯决策系统性能的具体影响。

    

    当一个序贯决策系统未能达到最优时，其得分并不能揭示哪些修改能够改进其决策。DECODE 是一个基准测试，用于通过对规划方式和可用信息进行受控实验来检验此类修改。其消费信贷催收生成器提供了五个部分可观测的环境，这些环境共享已知的动力学矩阵，并在记忆参数和过程噪声上有所变化。实验在保持求解器其余部分不变的情况下，改变选定的输入或规划规则，每次比较都明确了求解器的状态输入、响应输入和规划范围。在确定性基础环境中（该环境被选中是因为使用隐藏状态时六个月规划与一个月规划之间存在较大差距），将规划范围从一个月延长到六个月，使两个梯度提升求解器的得分相对精确最优值（基于隐藏状态和借款人信息计算）提高了15到17个百分点。

    arXiv:2611.18367v1 Announce Type: new  Abstract: When a sequential decision system falls short of an optimum, its score does not reveal which modifications would improve its decisions. DECODE is a benchmark for testing such modifications through controlled experiments on planning and available information. Its consumer-credit collections generator provides five partially observed environments that share known dynamics matrices and vary a memory parameter and process noise. The experiments change selected inputs or planning rules while keeping the rest of the solver fixed. Each comparison specifies the solver's state inputs, response inputs and planning horizon. On the deterministic base environment, which was selected for a large gap between six-month and one-month planning with the hidden state, extending planning from one to six months increases the scores of two gradient-boosted solvers by 15 to 17 percentage points of the exact optimum, computed with the hidden state and borrower t
    
[^35]: 适应性水印：通过轻量级适配器转换图像水印模型

    Adaptable Watermarking: Transforming Image Watermarking Models via Lightweight Adapters

    [https://arxiv.org/abs/2611.18363](https://arxiv.org/abs/2611.18363)

    提出了一个通过轻量级参数高效适配器和多域残差对齐损失，将现有图像水印模型转换为其他水印方法的框架，无需重新训练完整模型即可实现模块化和可升级的水印架构。

    

    我们探索了神经图像水印模型的适应性，即一种水印方法通过轻量级适配被转换为另一种方法的程度。我们提出了一个框架，使编码器-解码器水印架构变得模块化和可升级，无需重新训练完整模型即可支持多种水印风格。适配通过附加一个紧凑的参数高效适配器，并在多域残差对齐损失（空间域、梯度域和频域）下进行微调来实现，同时通过秘密损失项保持比特准确性。我们在一系列近期水印模型上评估了这种方法，包括TrustMark、InvisMark、RoSteALS、HiDDeN和WAM，证明了适配后的编码器能够紧密再现目标残差，且适配后的解码器在标准增强和失真条件下能以接近最优的准确性恢复消息。

    arXiv:2611.18363v1 Announce Type: cross  Abstract: We explore the adaptability of neural image watermarking models; the extent to which one watermarking method can be transformed into another through lightweight adaptation. We introduce a framework that enables encoder-decoder watermarking architectures to become modular and upgradable, supporting multiple watermarking styles without re-training full models. Adaptation is performed by attaching a compact parameter-efficient adapter and fine-tuning under a multi-domain residual alignment loss (spatial, gradient, and frequency) while preserving bit accuracy via a secret-loss term. We evaluate this approach across a suite of recent watermarking models, including TrustMark, InvisMark, RoSteALS, HiDDeN, and WAM, demonstrating that adapted encoders closely reproduce target residuals and that adapted decoders recover messages with near-optimal accuracy under standard augmentations and distortions. We show that cross-model adaptation, with app
    
[^36]: 在线切换控制作为专家学习问题

    Online Switching Control as Learning with Experts

    [https://arxiv.org/abs/2611.18352](https://arxiv.org/abs/2611.18352)

    本文将离散时间非线性系统的在线切换控制问题归约为带切换代价的专家问题，并基于带切换抑制机制的Hedge算法提出了自适应切换算法，实现了 $O(\sqrt{T\log N})$ 的期望遗憾同时保证状态一致有界。

    

    本文研究离散时间非线性系统的在线切换控制问题，其中控制器需要从有限个候选控制器中依次选择。目标是在扰动和代价函数未知的情况下，生成自适应的控制器序列以最小化累积代价。具体而言，我们考虑全信息反馈设定，即每次选择之后，实际产生的扰动影响和代价函数都是可获得的。利用由历史数据计算得到的反事实代价，我们将在线切换控制问题归约为带切换代价的专家问题。随后，我们提出了一种基于Hedge算法并带有切换抑制机制的自适应切换算法。值得注意的是，所提出的算法实现了 $O(\sqrt{T\log N})$ 的期望遗憾，其中 $T$ 为时间范围，$N$ 为候选控制器数量，同时保证了状态的一致有界性。

    arXiv:2611.18352v1 Announce Type: cross  Abstract: This paper addresses the problem of online switching control for discrete-time nonlinear systems where a controller is sequentially selected from a finite set of candidates. The goal is to generate an adaptive sequence of controllers to minimize the cumulative cost under unknown disturbances and cost functions. Specifically, we consider a full-information feedback setting where realized disturbance effects and cost functions are available after each selection. Using counterfactual costs computed from historical data, we reduce the online switching control problem to the experts problem with switching costs. Then, we propose an adaptive switching algorithm based on the Hedge algorithm with a switching suppression mechanism. Notably, the proposed algorithm achieves an expected regret of $O(\sqrt{T\log N})$, where $T$ is the time horizon and $N$ is the number of candidate controllers, while ensuring uniform boundedness of the state. This 
    
[^37]: 利用紧凑神经网络在OCO-2各观测模式下模拟基于物理的XCO2反演

    Emulating physics-based XCO2 retrievals across OCO-2 viewing modes with compact neural networks

    [https://arxiv.org/abs/2611.18333](https://arxiv.org/abs/2611.18333)

    该研究通过奇异值分解压缩OCO-2光谱并训练紧凑多层感知器，实现快速模拟全物理XCO2反演，在2021-2025年测试中三种观测模式的年均均方根误差仅为0.89至1.43 ppm。

    

    基于卫星光谱的二氧化碳柱平均干空气摩尔分数的全物理业务化反演在计算上开销巨大。我们采用奇异值分解将轨道碳观测卫星2号（OCO-2）的光谱按波段分别进行压缩。紧凑的多层感知器利用所得的光谱得分、先验信息和观测几何，来预测全物理反演结果与业务化先验值之间的差异。将该预测差异加到先验值上即可得到反演摩尔分数的估计。模型在2015-2019年的观测数据上训练，并在2020年的观测数据上验证。在2021-2025年的测试中，相对于天基大气二氧化碳观测（ACOS）反演产品，海洋耀斑观测、陆地耀斑观测和天底观测的年均均方根误差分别为0.89、1.35和1.43 ppm。

    arXiv:2611.18333v1 Announce Type: new  Abstract: Operational full-physics retrieval of the column-averaged dry-air mole fraction of carbon dioxide from satellite spectra is computationally demanding. We use singular value decomposition to compress spectra from the Orbiting Carbon Observatory-2 separately by band. Compact multilayer perceptrons use the resulting spectral scores, prior information, and viewing geometry to predict the difference between the full-physics retrieval and its operational prior. Adding the predicted difference to the prior yields an estimate of the retrieved mole fraction. The models were trained on 2015-2019 observations and validated on 2020 observations. In 2021-2025 tests, the mean annual root-mean-square errors relative to the Atmospheric Carbon Dioxide Observations from Space retrieval product were 0.89, 1.35, and 1.43 parts per million for ocean glint, land glint, and nadir observations, respectively. On East Asian nadir observations screened using the p
    
[^38]: 清洁票据，错误订单：用浏览器标签页中的语言模型捕捉静默错误

    Clean Tickets, Wrong Orders: Catching Silent Errors with a Language Model in Your Browser Tab

    [https://arxiv.org/abs/2611.18284](https://arxiv.org/abs/2611.18284)

    本文展示了一个完全在浏览器本地运行的小型语言模型演示系统，利用移植到TypeScript的PIZZA匹配器实时捕捉“静默错误”（即模型生成格式正确却无人真正下过的订单），其表现与服务器端评估几乎完全一致。

    

    将输入订单转换为结构化数据的小型语言模型可能以两种方式失败：一是输出损坏的内容，任何系统都会将其拒绝；二是输出一个干净、格式良好却无人真正下过的订单。我们的配套工作将第二种情况称为“静默错误”，而这才是危险的一种。我们展示了一个网页演示：一个经过微调的0.8B参数模型通过编译为WebAssembly的llama.cpp完全在访问者的浏览器标签页中运行，任何订单文本都不会离开用户设备。页面在访问者输入时实时解析订单，使用移植到TypeScript的PIZZA基准测试自带匹配器对每张票据进行评分，用"？"拒绝任何不在菜单上的内容以让模型直面开放网络，并允许访问者寻找静默错误或尝试制造一个。在浏览器中对全部1,357个PIZZA测试订单运行时，该部署模型对99.0%的订单输出与服务器端评估完全一致，并达到相同的精确匹配率（0.882）；我们还报告了其延迟表现。

    arXiv:2611.18284v1 Announce Type: cross  Abstract: A small language model that turns a typed order into structure can fail in two ways. It can print something broken, which any system rejects. Or it can print a clean, well formed order that nobody placed. Our companion work calls the second kind a silent error, and it is the dangerous one. We present a web demo in which a fine-tuned 0.8B parameter model runs entirely inside the visitor's browser tab, through llama.cpp compiled to WebAssembly. No order text leaves the device. The page parses orders as the visitor types, scores every ticket with the PIZZA benchmark's own matcher ported to TypeScript, refuses anything off the menu with "?" so the model can face the open web, and lets visitors hunt for silent errors or try to cause one. Run in the browser on all 1,357 PIZZA test orders, the deployed model writes the same output as the server evaluation for 99.0% of them and reaches the same exact match (0.882); we also report its latency o
    
[^39]: 正确的词元，正确的时机：从逐词元门控到轨迹感知的激活引导

    Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering

    [https://arxiv.org/abs/2611.18274](https://arxiv.org/abs/2611.18274)

    该论文将基于稀疏自编码器的激活引导重构为固定预算下的逐词元分配问题，并提出无记忆的对比特征投影（CFP）与轨迹感知的闭环自适应伯努利引导（CLABS）两种门控机制，通过只在恰当的词元和时机施加干预来避免分布偏移、保持生成质量。

    

    通过稀疏自编码器（SAE）进行的激活引导能够在无需任务特定微调的情况下对大型语言模型实现行为控制。然而，标准方法会在生成的每一个词元上都施加引导信号，这会使模型表示偏离其训练时所处的分布，从而损害生成质量。我们证明了在生成过程中“何时”施加干预是决定控制效果的一个关键维度。我们将引导问题形式化为在固定可引导词元预算下的逐词元分配问题，使不同方法之间的差异仅体现在其选择的词元上，从而让分配方式成为唯一变量。在该框架下，我们引入了两种门控机制：第一种是对比特征投影（Contrastive Feature Projection, CFP），它是无记忆的，通过稀疏且可解释的特征投影来选择与目标行为最不对齐的词元；第二种是闭环自适应伯努利引导，它是轨迹感知的，将分配视为一个闭环过程。

    arXiv:2611.18274v1 Announce Type: cross  Abstract: Activation steering via sparse autoencoders (SAEs) enables behavioral control of large language models without task-specific fine-tuning. Yet, standard methods apply the steering signal at every generated token, displacing representations from the distribution the model was trained on and eroding generation quality. We show that \emph{when} along a generation the intervention is applied is a decisive axis of control. We cast steering as a per-token allocation problem under a fixed budget of steerable tokens, so that methods differ only in which tokens they select, making allocation the sole variable. Within this framework we introduce two gates. The first, Contrastive Feature Projection (CFP), is memoryless and selects the tokens least aligned with the target behavior through a sparse, interpretable feature projection. The second, Closed-Loop Adaptive Bernoulli Steering (CLABS), is trajectory aware. It treats allocation as closed-loop 
    
[^40]: ParetoM³：通过极小-极大-极小优化在偏好引导下学习帕累托集

    ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization

    [https://arxiv.org/abs/2611.18252](https://arxiv.org/abs/2611.18252)

    提出了一种基于惩罚的极小-极大-极小优化公式（M³问题）及单循环一阶算法 ParetoM³，将偏好优化与弱帕累托最优性统一在单一目标中，并证明了算法能收敛到近似弱帕累托最优的ε-稳定点。

    

    在多个本质冲突的目标上进行学习，需要从帕累托集中选择理想的权衡方案。在这项工作中，我们研究了偏好引导的多目标优化问题，即在多个非凸目标的弱帕累托集上最小化一个预定义的偏好函数。利用一个在弱帕累托集上取值为零的優值函数（merit function），我们提出了一种基于惩罚的极小-极大-极小（min-max-min）公式，称为 M³ 问题，该公式将偏好优化与弱帕累托最优性的强制约束统一在单一目标中。我们进一步开发了 ParetoM³，这是一种单循环一阶算法，用于计算所得 M³ 问题的稳定点。在局部 Kurdyka-Łojasiewicz 假设下，我们证明了当惩罚参数足够大时，ParetoM³ 能够找到 M³ 问题的 ε-稳定点，并且该点对于原始多目标问题也是近似弱帕累托最优的。

    arXiv:2611.18252v1 Announce Type: cross  Abstract: Learning across multiple inherently conflicting objectives requires selecting desirable trade-offs from the Pareto set. In this work, we study preference-guided multi-objective optimization, where a predefined preference function is minimized over the weak Pareto set of multiple nonconvex objectives. Leveraging a merit function that vanishes on the weak Pareto set, we propose a penalty-based min-max-min formulation, termed the M$^3$ problem, which unifies preference optimization and the enforcement of weak Pareto optimality in a single objective. We further develop ParetoM$^3$, a single-loop first-order algorithm for computing stationary solutions of the resulting M$^3$ problem. Under a local Kurdyka-Lojasiewicz assumption, we prove that ParetoM$^3$ finds an $\epsilon$-stationary point of the M$^3$ problem that is also approximately weakly Pareto optimal for the original multi-objective problem when the penalty parameter is sufficientl
    
[^41]: 神经分类器作为纠错码

    Neural Classifiers as Error-Correcting Codes

    [https://arxiv.org/abs/2611.18239](https://arxiv.org/abs/2611.18239)

    该论文提出统一框架，将监督深度分类刻画为香农意义下的纠错码译码，证明交叉熵优化等价于最大似然译码，深度网络在特征空间中隐式学习码本，并据此推导出无损量化、知识蒸馏和结构正则化的新准则。

    

    本文提出了一个统一框架，将监督深度分类刻画为香农意义下的纠错码译码过程。研究表明，交叉熵优化使softmax分类器与最大似然译码相一致，其中logits映射为到各类别特定码字的汉明距离。这种对应关系解释了标准散度目标（如Kullback-Leibler散度、Jeffreys散度）的出现，并为决策边界的稳定性提供了几何基础。此外，Neural Collapse与LeJEPA的几何结构被识别为各向同性噪声下译码的最优星座图。与经典的纠错输出码不同，研究表明深度网络能够在特征空间中隐式地学习有效的码本。基于这一见解，本文推导出用于无损量化、改进知识蒸馏以及基于码冗余的结构正则化的新准则。最后，这一……（摘要被截断）

    arXiv:2611.18239v1 Announce Type: new  Abstract: A unified framework is proposed that characterizes supervised deep classification as the decoding of error-correcting codes in the Shannon sense. Cross-entropy optimization is shown to align softmax classifiers with maximum-likelihood decoding, where logits map to Hamming distances to class-specific codewords. This correspondence explains the emergence of standard divergence objectives (e.g., Kullback-Leibler, Jeffreys) and provides a geometric basis for the stability of decision boundaries. Furthermore, the geometries of Neural Collapse and LeJEPA are identified as optimal constellations for decoding under isotropic noise. Unlike classical Error-Correcting Output Codes, deep networks are shown to learn effective codebooks in the feature space implicitly. This insight is leveraged to derive novel criteria for lossless quantization, improved knowledge distillation, and architectural regularization based on code redundancy. Finally, this p
    
[^42]: 小并非消亡：面向 e-means 的距离感知聚类复活机制

    Small Is Not Dead: Distance-Aware Cluster Revival for e-means

    [https://arxiv.org/abs/2611.18238](https://arxiv.org/abs/2611.18238)

    针对 e-means 仅依据聚类规模判定死聚类、会误删覆盖稀疏孤立点群的小聚类的问题，本文提出增加基于中心间距离的第二条件——只有当小聚类的中心靠近其他中心时才将其移除复活，从而在含稀疏孤立点群的数据上显著降低聚类误差。

    

    e-means（Bessa 等人，2026）是一种新近提出的小批量 k-means 算法，现已并入 FAISS 库，在大规模嵌入数据集上能比 Lloyd 算法快约一个数量级地达到相同的误差水平。每处理五个小批量，该算法会移除所有运行中点数低于平均值十分之一的聚类，并将其中心重新用于分裂一个大聚类。在作者的消融实验中，移除这一被称为“死聚类复活”的机制所导致的误差上升，比移除滑动平均更新中加入的其他任何组件都要大。然而，该规则仅考虑聚类规模：它无法区分因冗余而变小，还是因覆盖稀疏、孤立的点群而变小，因此在包含此类点群的数据上，它会反复移除那些移除代价很高的中心。我们增加了一个基于中心间距离的第二条件：仅当小聚类的中心相对于彼此靠近另一个中心时，小聚类才会被移除。

    arXiv:2611.18238v1 Announce Type: new  Abstract: e-means (Bessa et al., 2026) is a recent mini-batch k-means algorithm, now part of the FAISS library, that reaches the error of Lloyd's algorithm about an order of magnitude faster on large embedding datasets. Every five mini-batches, it removes every cluster whose running point count falls below a tenth of the average and reuses its centre to split a large cluster. In the authors' ablation, removing this mechanism, called dead-cluster revival, raises the error more than removing any other component added to the moving-average update. The rule, however, considers only cluster size: it cannot distinguish a cluster that is small because it is redundant from one that covers a rare, isolated group of points, and on data with such groups it repeatedly removes centres whose removal is costly. We add a second condition, based on the distance between centres: a small cluster is removed only if its centre is close to another centre, relative to a
    
[^43]: NeoResearch：文献与经验引导的自主时间序列模型发现

    NeoResearch: Literature- and Experience-Guided Autonomous Time Series Model Discovery

    [https://arxiv.org/abs/2611.18235](https://arxiv.org/abs/2611.18235)

    NeoResearch 提出了一个文献与经验引导的自主研究框架，利用大语言模型充当“研究者”，通过结合假设驱动架构探索与迭代训练及超参数优化的双循环工作流，自主挖掘网络科学文献并发现轻量级、领域定制的时间序列预测模型。

    

    为多样化的时间序列领域开发有效的预测模型，需要大量的专家知识、来自网络科学文献的证据以及反复的实验迭代。然而，将这些知识和研究经验转化为模型设计，需要保留源上下文，并在任务特定假设下检验其适用性。在本文中，我们提出了 NeoResearch，一个文献与经验引导的自主研究框架，它利用大语言模型（LLM）作为“研究者”来发现轻量级、面向特定领域的预测模型。具体而言，NeoResearch 建立了受专家启发的双循环研究工作流，将假设驱动的架构探索与基于实验反馈的迭代训练和超参数优化相结合。为实现基于知识的发现，该智能体自主从网络中挖掘科学文献，提取建模机制……（原文摘要至此截断）

    arXiv:2611.18235v1 Announce Type: new  Abstract: Developing effective forecasting models for diverse time-series domains requires substantial expert knowledge, evidence from scientific literature on the web, and iterative experimentation. However, translating this knowledge and research experience into model designs requires preserving source context and testing applicability under task-specific assumptions. In this paper, we propose NeoResearch, a literature- and experience-guided autonomous research framework that leverages large language models (LLMs) as researchers to discover lightweight, domain-specific forecasting models. Specifically, NeoResearch establishes an expert-inspired dual-loop research workflow, coupling hypothesis-driven architecture exploration with iterative training and hyperparameter optimization through experimental feedback. To enable knowledge-grounded discovery, the agent autonomously mines scientific literature from the Web, extracting modeling mechanisms, e
    
[^44]: T2ST：通过时间数据与知识迁移实现数据高效的时空基础模型

    T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer

    [https://arxiv.org/abs/2611.18199](https://arxiv.org/abs/2611.18199)

    T2ST 框架通过引入丰富的时间序列数据辅助训练，并从冻结的时间序列基础模型中迁移知识，在时空观测稀缺的情况下实现数据高效的时空基础模型训练。

    

    arXiv:2611.18199v1 公告类型：cross 摘要：时空基础模型（STFMs）需要大规模训练语料，然而同步观测和空间元数据的获取与整理成本高昂，限制了合适训练数据的可获得性。相比之下，时间序列数据丰富且更易于收集。由于每个位置的观测本身构成了一个时间序列，这引出了一个自然的问题：当时空观测稀缺时，丰富的时间序列数据能否增强时空基础模型？为弥合这一差距，我们提出了 T2ST，一个数据高效的训练框架，它利用精选的辅助时间序列来增强有限的时空观测，并从冻结的时间序列基础模型中迁移知识。我们的关键洞察是：时间模式学习可以从丰富的外部时间序列中受益，而空间依赖学习则仍然立足于真实的时空观测。基于这一洞察，T2ST 检索匹配的序列以强化稀疏的……（原文摘要至此截断）

    arXiv:2611.18199v1 Announce Type: cross  Abstract: Spatio-temporal foundation models (STFMs) require large-scale training corpora, yet synchronized observations and spatial metadata are costly to acquire and curate, limiting the availability of suitable data. By contrast, time-series data are abundant and easier to collect. Since observations at each location form a time series, this motivates a natural question: can abundant time-series data enhance STFMs when spatio-temporal observations are scarce? To bridge this gap, we propose T2ST, a data-efficient training framework that augments limited spatio-temporal observations with selected auxiliary time series and transfers knowledge from a frozen time-series foundation model. Our key insight is that temporal pattern learning can benefit from abundant external time series, while spatial dependency learning remains grounded in real spatio-temporal observations. Guided by this insight, T2ST retrieves matched sequences to reinforce sparsely
    
[^45]: MarathonTS：基于窗口组装与长序列生成器的长合成时间序列基准

    MarathonTS: a benchmark of long synthetic time series from window assembly and long-series generators

    [https://arxiv.org/abs/2611.18190](https://arxiv.org/abs/2611.18190)

    MarathonTS是首个在统一协议下系统比较窗口组装与长序列生成器两类方法的长合成时间序列开放基准，从保真度、实用性、记忆性和计算效率多维度全面评估序列质量。

    

    许多合成时间序列的应用场景需要比生成模型训练所用的窗口长得多的序列。这样的序列既可以从短生成窗口池中组装而成（这需要决定选取哪些窗口、以何种顺序排列以及如何处理窗口间的接缝），也可以直接由长序列生成器生成。本研究提出了MarathonTS，据我们所知，这是首个在统一协议下比较这两类方法家族的开放基准。每种方法在9个开放的逐小时数据集上以10个随机种子生成相同长度的序列。序列的评估维度包括：对真实数据的保真度（12组共83个指标，对比三个真实参考集：训练集、验证集和按时间划分的集外数据）、实用性（模型仅在合成序列上训练并在真实数据上测试，即TSTR对比TRTR，涵盖14个分类和回归任务）、对训练数据的记忆程度以及计算成本。

    arXiv:2611.18190v1 Announce Type: new  Abstract: Many applications of synthetic time series require a series that is much longer than the windows on which a generative model is trained. Such a series can be assembled from a pool of short generated windows, which requires deciding which windows are taken, in what order, and how their seams are processed, or it can be produced directly by a long-series generator. This study presents MarathonTS, to our knowledge the first open benchmark that compares both families under one protocol. Every method produces a series of the same length on 9 open hourly datasets with 10 seeds. The series is evaluated by its fidelity to the real data (83 metrics in 12 groups, against three real references: train, validation and an out-of-time split), by its utility (models are trained on the synthetic series only and tested on real data, TSTR against TRTR, in 14 classification and regression tasks), by its memorisation of the training data and by its computati
    
[^46]: LaMe：面向高维系统的联合潜在中介分析框架

    LaMe: a Joint Latent Mediation Framework for High-Dimensional Systems

    [https://arxiv.org/abs/2611.18180](https://arxiv.org/abs/2611.18180)

    该论文提出LaMe，一个联合线性-高斯潜在中介框架，通过少量潜变量概括高维的暴露-中介-结果系统，在明确假设下识别自然直接与间接效应，克服了现有方法过度参数化、推断有偏和可扩展性受限的问题。

    

    中介分析旨在理解暴露因素如何通过中间变量影响结果。然而，随着日益丰富的数据集的出现，成百上千个潜在中介变量使得这些系统难以建模和解释。潜变量方法为处理这类高维场景提供了一种自然的途径。然而，现有的潜在中介方法存在诸多局限，包括过度参数化、推断有偏或可扩展性受限。我们开发了LaMe，这是一个联合的线性-高斯框架，通过少量潜变量来概括高维的暴露-中介-结果系统。我们在一个明确的假设下（即显性测量完全捕捉潜中介变量）识别出自然直接效应和间接效应，并刻画了潜在表示能够有效约束观测交叉协方差结构的条件。模拟实验...

    arXiv:2611.18180v1 Announce Type: cross  Abstract: Mediation analysis seeks to understand how an exposure influences an outcome through intermediate variables. However, as increasingly rich datasets become available, hundreds to thousands of potential mediators make these systems difficult to model and hard to interpret. Latent-variable approaches offer a natural way to accommodate these high-dimensional settings. However, existing latent mediation methods face limitations, including overparameterization, biased inference, or limited scalability. We develop LaMe, a joint linear-Gaussian framework that summarizes a high-dimensional exposure-mediator-outcome system through a small number of latent variables. We identify natural direct and indirect effects under an explicit assumption that the manifest measurements fully capture the latent mediators, and characterize conditions under which the latent representation effectively restricts the observed cross-covariance structure. Simulations
    
[^47]: 可迁移的解释：用稀疏自编码器解读医学信息检索

    Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

    [https://arxiv.org/abs/2611.18179](https://arxiv.org/abs/2611.18179)

    本文提出将稀疏自编码器应用于医学信息检索，通过结合重建目标与InfoNCE对比损失，将不透明的稠密文本嵌入转化为语义可解释的表示，同时保持对新临床语料库的可迁移性。

    

    回答用户查询的医学AI系统依赖于从大型文档集合中检索医学知识。在此背景下，在医学数据集上微调通用模型可以提升检索相关性。然而，现代检索系统所依赖的稠密嵌入在很大程度上仍是不透明的，难以理解其检索结果背后的机制。此外，针对特定医学数据集进行适配可能会失去对新临床语料库的可迁移性。在本工作中，我们通过将稀疏自编码器（SAEs）应用于医学信息检索来解决这些问题，将稠密文本嵌入转化为语义上可解释的表示。为此，我们的模型将重建目标与InfoNCE对比损失相结合，后者能够直接保留查询与文档之间的相关性。我们在六个医学检索数据集上（涵盖域内与域外场景）评估了三种嵌入模型。在医学语料库上训练后，我们的稀疏……

    arXiv:2611.18179v1 Announce Type: cross  Abstract: Medical AI systems that answer users' queries rely on retrieving medical knowledge from large document collections. In this context, finetuning general models on medical datasets can improve retrieval relevance. Yet the dense embeddings underlying modern retrieval systems remain largely opaque, making it difficult to understand the mechanics behind their results. Moreover, adapting to a specific medical dataset risks losing transferability to new clinical corpora. In this work, we tackle these issues by applying Sparse Autoencoders (SAEs) to medical information retrieval, transforming dense text embeddings into semantically interpretable representations. To do so, our model combines a reconstruction objective with an InfoNCE contrastive loss that directly preserves query-document relevance. We evaluate three embedding models across six medical retrieval datasets, both in-domain and out-of-domain. Trained on a medical corpus, our sparse
    
[^48]: 潜空间中的检测与分类：高维分析与验证

    Detection and Classification in Latent Spaces: High-Dimensional Analysis and Validation

    [https://arxiv.org/abs/2611.18164](https://arxiv.org/abs/2611.18164)

    该论文在潜在维度与样本规模成比例增长的高维状态下，对基础模型潜空间中的线性和二次检测器进行了Neyman-Pearson大偏差分析，推导出虚警与漏检错误概率的显式刻画，揭示了双下降乃至多重下降的非单调现象，并通过视觉、文本和音频数据集实验验证了理论预测与实际性能的高度一致性。

    

    基础模型会产生高维潜在表示，其中经过适当设计的统计决策规则既能有效工作，又便于解析处理。我们在高维状态下研究这类检测器，即潜在维度与训练样本规模按比例增长的情形。对于线性和二次统计量，我们在Neyman-Pearson范式下推导出了虚警与漏检性能的显式大偏差刻画，并展示了二元分析如何扩展到多类别分类。所得到的错误概率及相应的衰减速率对形状参数呈现出非单调依赖性，包括双下降现象，在某些状态下甚至出现多重下降现象。在现代基础模型上对视觉、文本和音频数据集进行的实验表明，理论预测与实际性能高度吻合。特别地，所提出的框架使我们能够设定预定的虚警……

    arXiv:2611.18164v1 Announce Type: cross  Abstract: Foundation models produce high-dimensional latent representations in which suitably designed statistical decision rules can be effective yet analytically tractable. We study such detectors in the high-dimensional regime, where latent dimension and training sample size grow proportionally. For linear and quadratic statistics, we derive explicit large-deviation characterizations of false-alarm and miss-detection performance under the Neyman-Pearson paradigm, and show how the binary analysis extends to multi-class classification. The resulting error probabilities and corresponding decay rates exhibit non-monotonic dependence on the shape parameter, including double-descent and, in some regimes, multiple-descent. Experiments on vision, text, and audio datasets using modern foundation models show close agreement between theoretical predictions and empirical performance. In particular, the proposed framework allows us to set a prescribed fal
    
[^49]: 关于充分解释的极限

    On the Limits of Sufficient Explanations

    [https://arxiv.org/abs/2611.18163](https://arxiv.org/abs/2611.18163)

    该论文建立了一个通用的充分解释形式化框架，并证明分类器在图和字符串等结构化数据上充分解释器的存在性完全由分类器与选择关系之间的相互作用决定。

    

    我们形式化地刻画了分类器在图和字符串等结构化数据上存在充分解释的条件。我们关注那些通过选取输入的一部分作为充分解释的事后解释器。如果所选部分本身就能保证预测标签，则该解释是充分的。充分解释已被广泛应用于图、字符串和图像等多种数据模态。然而，对充分解释的统一形式化分析仍然缺失。我们通过四项主要贡献来弥补这一空白：(i) 我们定义了一个通用的解释形式体系，由实例空间、分类器和选择关系组成，其中选择关系规定了从每个实例中可以选取哪些潜在的子实例或特征子集作为解释；(ii) 我们证明了充分解释器的存在完全由分类器与选择关系之间的相互作用所刻画。

    arXiv:2611.18163v1 Announce Type: new  Abstract: We formally characterize the conditions under which sufficient explanations exist for classifiers on structured data such as graphs and strings. We focus on post-hoc explainers that select parts of the input as sufficient explanations. An explanation is sufficient if the selected part alone ensures the predicted label. Sufficient explanations have been widely adopted across data modalities, including graphs, strings, and images. However, a unified formal analysis of sufficient explanations remains missing. We bridge this gap with four main contributions: (i) We define a general explanation formalism consisting of an instance space, a classifier, and a select relation. The select relation specifies which potential sub-instances or feature subsets can be selected as explanations from each instance; (ii) We show that the existence of sufficient explainers is fully characterized by the interplay between the classifier and the select relation
    
[^50]: 哪个模型，用于哪个变量？稀疏ICU时间序列中的逐变量预测组合

    Which Model, For Which Variable? Per-Variable Forecast Combination in Sparse ICU Time Series

    [https://arxiv.org/abs/2611.18149](https://arxiv.org/abs/2611.18149)

    研究发现ICU生理指标预测中不存在对所有变量都最优的单一模型，通过在五个ICU数据库上对94个变量逐变量组合14个模型的预测，平均可将均方误差较最佳单一模型降低8.7%，且增益集中在数据最稀疏的变量上。

    

    临床预测论文通常只报告单一的最佳架构。我们发现在多变量ICU生理指标预测中，这一问题掩盖了变量之间的差异：没有任何一种架构在所有变量上都是最优的，哪个模型胜出会随变量而变化。我们在五个ICU数据库（MIMIC-III、MIMIC-IV、eICU、HiRID以及UMM（曼海姆大学医学中心，一个德国单中心ICU））上，使用包含14个模型的模型池对94个生理通道进行24小时时间跨度的预测，并采用逐变量拟合的权重对这些模型的预测进行组合。在测试集上，该组合方法在所有五个数据库上均优于最佳单一模型，均方误差降低6.1%至11.0%（平均8.7%）。增益主要集中在临床数据最稀疏的地方：最稀疏的三分之一通道提升了11.9%，实验室检验值提升了7.5%，而最密集的三分之一通道仅提升3.1%。对每个变量表现最好的五个模型采用基于排名的调和加权——完全无需拟合任何权重——即可恢复约五分之四的增益。

    arXiv:2611.18149v1 Announce Type: new  Abstract: Clinical forecasting papers typically report a single best architecture. We show that on multivariate ICU physiology this question conceals variable-specific differences: no architecture is best on every variable, and which model wins changes from variable to variable. Across five ICU databases (MIMIC-III, MIMIC-IV, eICU, HiRID, and UMM (University Medicine Mannheim), a German single-center ICU) we forecast 94 channels over a 24-hour horizon with a pool of 14 models, and combine them with weights fitted per variable. On test, the combination beats the best single model on all five databases by 6.1-11.0% mean squared error (mean 8.7%). The gain is concentrated where clinical data is thinnest: the sparsest third of channels improves by 11.9% and laboratory values by 7.5%, against 3.1% for the densest third. Rank-based harmonic weighting of the best five models per variable, with no fitted weights at all, recovers about four-fifths of the g
    
[^51]: 编译专用化的GPU发起网络通信

    Compiling Specialized GPU-Initiated Networking

    [https://arxiv.org/abs/2611.18145](https://arxiv.org/abs/2611.18145)

    Obaleth编译器将GPU发起的通信纳入内核中间表示并与计算协同优化，生成专门的传输代码，在小消息发送、大规模all-to-all通信和大模型推理服务中显著超越NVSHMEM、NCCL GIN和DeepEP等通用通信库。

    

    机器学习系统能够针对工作负载和硬件对计算进行专门化优化，但通信部分仍停留在通用库中，其设备代码和传输决策对高级编译器不可见。我们提出了Obaleth，一个将GPU发起的通信纳入内核中间表示的编译器，它将通信与计算共同优化，并将其降低为专门的传输代码。Obaleth围绕缓冲区生命周期规划完成操作，并在NVIDIA和AMD GPU上优化内存排序、同步和批处理。在H100 GPU上，Obaleth发送小消息的速度最高提升3.3倍；在256个GPU上，其all-to-all通信比NVSHMEM和NCCL GIN分别快1.25倍和2.3倍，且内存和寄存器占用显著更小。在vLLM中，它在16个B200 GPU上服务Qwen3-235B模型，响应请求的速度比使用DeepEP最高快1.29倍。

    arXiv:2611.18145v1 Announce Type: cross  Abstract: Machine learning systems specialize computation to the workload and hardware, but communication remains in generic libraries whose device code and transport decisions are not visible to high-level compilers. We present Obaleth, a compiler that makes GPU-initiated communication part of a kernel's intermediate representation, co-optimizes it alongside computation, and lowers it to specialized transport code. Obaleth plans completions around buffer lifetimes and optimizes memory ordering, synchronization, and batching across NVIDIA and AMD GPUs. On H100 GPUs, Obaleth sends small messages up to 3.3$\times$ faster, and on 256 GPUs runs all-to-all 1.25$\times$ and 2.3$\times$ faster than NVSHMEM and NCCL GIN, with much smaller memory and register use. In vLLM, it serves Qwen3-235B on 16 B200 GPUs and answers requests up to 1.29$\times$ faster than with DeepEP.
    
[^52]: STAMP：面向制造信息的钣金面板生成式设计与形状优化

    STAMP: Manufacturing-Informed Generative Design and Shape Optimization of Sheet Panels

    [https://arxiv.org/abs/2611.18144](https://arxiv.org/abs/2611.18144)

    STAMP 通过将制造几何条件直接嵌入形状表示本身，实现了钣金面板的生成式设计与形状优化，使所有生成和优化的结果无论训练数据多少都能天然满足制造约束。

    

    成形钣金面板的生成式设计必须在探索布局并提升性能的同时保持几何制造条件。STAMP 将这些条件直接嵌入到形状表示中。每个面板由其材料布局以及一个参考曲面来描述，该参考曲面在与绘图方向垂直的平面上的每个被占据位置具有一个高度。名义厚度沿法线方向指定。该表示允许曲面、加强筋和开口的存在，排除了参考曲面下的倒扣，并在无需制造惩罚项的情况下保持名义厚度固定。由于这些条件属于形状表示本身，无论训练集规模如何，每个生成和优化的面板都天然满足这些制造条件。边界条件扩散模型生成备选布局，预测器梯度引导曲面优化，同时保留轮廓、开口和指定的固定区域。我们在汽车引擎盖上对 STAMP 进行了演示。

    arXiv:2611.18144v1 Announce Type: new  Abstract: Generative design of formed sheet panels must preserve geometric manufacturing conditions while exploring layouts and improving performance. STAMP embeds these conditions directly in the shape representation. Each panel is described by its material layout and a reference surface with one height at each occupied position in a plane perpendicular to the drawing direction. Nominal thickness is prescribed along the normal. The representation allows curved surfaces, ribs, and openings. It excludes reference-surface undercuts and keeps nominal thickness fixed without manufacturing penalty terms. Because these conditions belong to the representation, every generated and optimized panel satisfies them regardless of training-set size. Boundary-conditioned diffusion generates alternative layouts, and predictor gradients guide a surface optimization that retains outlines, openings, and designated fixed regions. We demonstrate STAMP on automotive ho
    
[^53]: 当复制难以被超越：临床评估预测中的一项负面结果

    When Copying Is Hard to Beat: A Negative Result in Clinical Assessment Forecasting

    [https://arxiv.org/abs/2611.18142](https://arxiv.org/abs/2611.18142)

    该研究是一项负面结果，表明在预测次日临床评估时，各类模型相比简单复制前一天答案的持续性基线并无一致优势。

    

    我们报告了在次日临床评估预测任务中的一项负面结果：所评估的模型相比强大的持续性基线并未表现出一致的优势。临床医生可以查看前一天的问卷答案，这是文档持续性的一种合理来源。在29,715个ICU日对上使用患者不相交的数据划分来预测45个条目时，一个基于生理信息的门控模型相比单行临床规则将条目宏F1提高了+0.0019 [+0.0006, +0.0031]，但准确率有所下降。当生理数据窗口结束于前一份表格的参考时间时（+0.0001 [-0.0006, +0.0008]），或当评分排除68.5%源自父级答案的单元格时，宏F1优势均无法确立；而将这两项变化结合时则出现了小幅劣势。三个模型家族的四个检查点均表现不如持续性基线，其中一个经过微调的模型与持续性基线仅在两个单元格上存在差异。分类式读出使变更单元格的准确率提高了八倍，但仅基于状态的

    arXiv:2611.18142v1 Announce Type: new  Abstract: We report a negative result in next-day clinical assessment forecasting: the evaluated models show no consistent advantage over strong persistence-based baselines. Clinicians can view the previous day's questionnaire answers, a plausible source of documentation persistence. Predicting 45 items on 29,715 ICU day-pairs with patient-disjoint splits, a physiology-informed gated model improves item-macro-F1 over a one-line clinical rule by +0.0019 [+0.0006,+0.0031], while losing accuracy. No macro-F1 advantage is established when the physiology window ends at the previous form's reference time (+0.0001 [-0.0006,+0.0008]), or when scoring excludes the 68.5% of cells derived from parent answers; combining both changes yields a small disadvantage. Four checkpoints across three families underperform persistence, and a fine-tuned one differs from it on only two cells. A categorical read-out raises changed-cell accuracy eightfold, but a state-only 
    
[^54]: PRISM：一种面向证据引导纠错与选择性弃权的预测后可靠性框架

    PRISM: A Post-Prediction Reliability Framework for Evidence-Guided Correction and Selective Abstention

    [https://arxiv.org/abs/2611.18127](https://arxiv.org/abs/2611.18127)

    PRISM是一个预测后可靠性框架，通过估计错误风险、检索历史证据来验证潜在不可靠的预测，从而实现有证据支撑的纠正或在证据不足时弃权，以应对顺序应用中高置信度预测仍可能出错的问题。

    

    模型可能会以高置信度产生预测，但结果仍然是错误的。这在顺序应用中是一个挑战，因为系统会随时间演变，部署时的数据可能与训练时使用的数据不同。为了解决这个问题，我们提出了PRISM（预测可靠性与智能自我纠正机制），这是一个在初始输出生成后评估预测可靠性的框架。它估计错误风险，并检查被识别为潜在不可靠的预测。对于这些情况，PRISM分析可用的信号，从历史数据中检索相似的情况，并验证是否有证据支持替代预测。然后，它可以保留原始预测、应用经过验证的纠正，或在证据不足时选择弃权。我们在印度超级联赛（IPL）的第二局比赛预测上评估了PRISM。逐球记录被转换……

    arXiv:2611.18127v1 Announce Type: cross  Abstract: A model may produce a prediction with high confidence and still get the outcome wrong. This is a challenge in sequential applications, where the system evolves over time and deployment data may differ from the data used during training. To address this issue, we propose PRISM (Prediction Reliability and Intelligent Self-Correction Mechanism), a framework that evaluates prediction reliability after the initial output is generated. It estimates the risk of error and examines predictions identified as potentially unreliable. For these cases, PRISM analyses available signals, retrieves similar situations from historical data, and verifies whether an alternative prediction is supported by the evidence. It can then retain the original prediction, apply a verified correction, or abstain when the evidence is insufficient.   We evaluate PRISM on second-innings match prediction in the Indian Premier League (IPL). Ball-by-ball records are convert
    
[^55]: 基于图像块统计量分解的高效庞加莱卷积

    Efficient Poincar\'e Convolutions through Patch-Statistics Factorization

    [https://arxiv.org/abs/2611.18125](https://arxiv.org/abs/2611.18125)

    本文提出将庞加莱卷积得分分解为图像块的平方范数和与滤波器的内积两个统计量，由原生卷积直接计算，无需将图像块映射到庞加莱球即可精确求值，使训练速度提升1.2至1.8倍。

    

    双曲神经网络++（HNN++）中已发表的庞加莱卷积实现在计算得分之前，需要将每个重叠的图像块映射到庞加莱球上。我们证明了该截断得分仅通过两个统计量依赖于每个图像块：其平方范数以及与滤波器的内积，而这两个统计量可以由原生卷积和单通道盒式滤波器直接计算得到。因此，该分解层无需构造球值图像块即可精确计算指定的得分，且其单精度输出和梯度与双精度计算结果的一致性不低于显式参考实现。在RTX A6000上、JAX的两个版本下，庞加莱U-Net和庞加莱ResNet-20的完整训练更新比两个将图像块映射到球上的显式实现中较快者快1.5至1.8倍（相比跳过该映射的显式实现则快1.2至1.4倍）。

    arXiv:2611.18125v1 Announce Type: new  Abstract: Published implementations of the Poincar\'e convolution in Hyperbolic Neural Networks++ (HNN++) map every overlapping image patch to the Poincar\'e ball before evaluating its score. We show that the clipped score depends on each patch only through two statistics, its squared norm and its inner products with the filters, which a native convolution and a single-channel box filter compute directly. The factorized layer therefore evaluates the specified score exactly, without constructing ball-valued patches, and its single-precision outputs and gradients agree with double-precision evaluations at least as closely as those of the explicit reference. On an RTX A6000 under two versions of JAX, complete training updates of a Poincar\'e U-Net and a Poincar\'e ResNet-20 are 1.5 to 1.8 times faster than the faster of two explicit implementations that map patches to the ball (1.2 to 1.4 times against an explicit implementation that skips this mappi
    
[^56]: Muon在随机重排下达到 $\mathcal{O}(\varepsilon^{-3})$ 迭代复杂度

    Muon Attains $\mathcal{O}(\varepsilon^{-3})$ Iteration Complexity under Shuffling

    [https://arxiv.org/abs/2611.18124](https://arxiv.org/abs/2611.18124)

    本文首次在随机重排采样下对Muon优化器进行了收敛性分析，证明了对于光滑非凸有限和问题寻找ε-驻点具有O(ε⁻³)的迭代复杂度，且该保证对任意置换序列均成立。

    

    Muon 是一种针对矩阵值参数的随机一阶优化器。它以系数 $\beta$ 在动量缓冲区中维护随机梯度的指数滑动平均，并对该缓冲区进行正交化以获得更新方向。据报道，Muon 能够加速并稳定大型语言模型的训练。在实践中，它通常与随机重排一起使用，即每个 epoch 以置换的顺序遍历有限和目标的各个分量。然而，其在该设置下的迭代复杂度仍是一个悬而未决的问题。我们在随机重排设置下分析了 Muon，其中动量缓冲区通过极映射进行正交化。对于光滑非凸有限和问题，我们建立了寻找 $\varepsilon$-驻点的 $\mathcal{O}(\varepsilon^{-3})$ 迭代复杂度。该保证对任意置换序列均成立，涵盖了随机重排、单次重排和增量梯度方法。

    arXiv:2611.18124v1 Announce Type: cross  Abstract: Muon is a stochastic first-order optimizer for matrix-valued parameters. It maintains an exponential moving average of stochastic gradients in a momentum buffer with coefficient $\beta$, and orthogonalizes the buffer to obtain the update direction. Muon has been reported to accelerate and stabilize the training of large language models. In practice, it is often used with shuffling (sampling without replacement), where each epoch visits the components of a finite-sum objective in a permuted order. However, its iteration complexity in this setting remains open. We analyze Muon under shuffling, with the momentum buffer orthogonalized by the polar map. For smooth nonconvex finite-sum problems, we establish an $\mathcal{O}(\varepsilon^{-3})$ iteration complexity for finding an $\varepsilon$-stationary point. This guarantee holds for every sequence of permutations, covering random reshuffling, shuffle once, and the incremental gradient metho
    
[^57]: 相关尺度变化下核岭回归的训练点一致性

    Training-point consistency of kernel ridge regression with varying correlation scales

    [https://arxiv.org/abs/2611.18095](https://arxiv.org/abs/2611.18095)

    本文建立了变化尺度参数下核岭回归的统一理论框架，证明训练点风险的一致性当且仅当有效局部样本量发散到无穷大，从而无缝统一了长程、固定尺度和短程等各种情形。

    

    核岭回归中的尺度参数在建模数据的相关结构方面起着关键作用。在现代应用中，采样密度与潜在的相关范围往往同时发生变化，这使得固定尺度的假设具有很强的局限性。然而，在变化的尺度参数下建立渐近理论从根本上是具有挑战性的。为填补这一空白，本文为变化尺度参数下的核岭回归建立了一个统一的理论框架。我们证明，在带有乘积指数核的均匀张量网格上，训练点风险的一致性成立当且仅当有效局部样本量发散到无穷大。这一判据覆盖了任意确定性尺度序列下的所有情形，无缝统一了长程、固定尺度和短程等各种情形。此外，我们还提供了一个显式的、与信号无关的确定性构造……

    arXiv:2611.18095v1 Announce Type: cross  Abstract: The scale parameter in kernel ridge regression plays a pivotal role in modeling the correlation structure of data. In modern applications, sampling density and the underlying correlation range often change simultaneously, making a fixed-scale assumption highly restrictive. However, establishing asymptotic theories under varying scale parameters is fundamentally challenging. To address this gap, this paper develops a unified theoretical framework for kernel ridge regression under varying scale parameters. We establish that, on uniform tensor grids with product exponential kernels, the consistency of the training-point risk holds if and only if the effective local sample size diverges to infinity. This sharp criterion universally covers all regimes along arbitrary deterministic scale sequences, seamlessly unifying the long-range, fixed-scale, and short-range. Furthermore, we provide an explicit, signal-independent construction of a deter
    
[^58]: 深层潜变量，长记忆：潜变量持久性解释Transformer激活中的时间相关性

    Deep Latents, Long Memory: Latent persistence explains temporal correlations in transformer activations

    [https://arxiv.org/abs/2611.18089](https://arxiv.org/abs/2611.18089)

    该论文提出用层激活的自相关函数作为抽象的可操作度量，发现潜变量的持久编码会在自相关中产生慢衰减尾部，且这些尾部随Transformer层加深而增长，从而解释了深层模型编码抽象潜变量的机制。

    

    大型语言模型（LLM）的隐藏层被认为编码了数据表示，其抽象程度随深度增加。然而，由于缺乏可操作的抽象概念定义，这一说法难以进行定量检验。在此，我们提出了另一种视角，其基础原理是：序列的相关性反映了生成该序列的过程的结构。我们将某一层的自相关函数定义为其在两个相隔一定滞后的上下文位置上的激活之间的平均点积。沿上下文持续编码的潜变量会在自相关中产生缓慢衰减的尾部，我们发现在合成数据和自然数据中，这些尾部都会随层加深而增长。在具有受控层级潜变量结构的合成数据中，这种增长源于对抽象潜变量的编码，这些潜变量支配着输入的长片段。事实上，潜变量的持久性……

    arXiv:2611.18089v1 Announce Type: cross  Abstract: The hidden layers of Large Language Models (LLMs) are believed to encode representations of the data whose abstraction increases with depth. However, in the absence of an operational notion of abstraction, this statement is hard to test quantitatively. Here, we propose an alternative view, building on the principle that the correlations of a sequence reflect the structure of the process that generated it. We define the autocorrelation function of a layer as the average dot product between its activations at two context positions separated by a lag. Latent variables encoded persistently along the context create slowly decaying tails in the autocorrelation, and we find these tails to grow across layers in both synthetic and natural data. In synthetic data with controlled hierarchical latent structure, this growth stems from the encoding of abstract latent variables, which govern long stretches of the input. Indeed, the persistence of lat
    
[^59]: 基于乘积单元残差网络学习非线性交互与跨特征耦合

    Learning nonlinear interactions and cross-feature couplings with product-unit residual networks

    [https://arxiv.org/abs/2611.18074](https://arxiv.org/abs/2611.18074)

    提出复数值乘积单元残差网络（CV-PURe），通过将乘法乘积单元与残差连接结合来显式建模跨特征耦合，在多个科学基准上预测误差显著低于MLP，并能有效恢复输入级的特征交互关系。

    

    理解非线性特征交互在科学与工程中至关重要，然而标准多层感知机（MLP）往往只能隐式地捕捉这类交互，导致表示相互纠缠，从而损害鲁棒性和可解释性。我们研究了复数值乘积单元残差网络（CV-PURe），该网络将乘法乘积单元与残差连接相结合，在稳定优化的同时显式地建模跨特征耦合。我们在一个交互驱动的合成基准和真实世界数据集上进行了系统评估，并在可复现的协议下同时评估预测性能和输入级交互恢复能力。在五次数据划分和五个训练种子下，CV-PURe 在六个科学基准中的四个上取得了最低的平均 RMSE，相比 MLP 将 Friedman 误差降低了 53.9%。其 Friedman 交互图将 60.7% 的总交互强度分配给已知的交互对，

    arXiv:2611.18074v1 Announce Type: new  Abstract: Understanding nonlinear feature interactions is crucial in science and engineering, yet standard multilayer perceptrons (MLPs) often capture such interactions only implicitly, leading to entangled representations that can impair robustness and interpretability. We investigate complex-valued product-unit residual networks (CV-PURe) that integrate multiplicative product units with residual connections to explicitly model cross-feature couplings while stabilizing optimization. We conduct a systematic evaluation on an interaction-driven synthetic benchmark and real-world datasets and evaluate prediction together with input-level interaction recovery under a reproducible protocol. Across five data splits and five training seeds, CV-PURe achieves the lowest mean RMSE on four of six scientific benchmarks, reducing Friedman error by 53.9% relative to MLP. Its Friedman interaction maps assign 60.7% of total strength to the known interacting pair,
    
[^60]: 门控循环网络中的乘积单元提升原子核质量与电荷半径的预测

    Product units in gated recurrent networks enhance prediction of nuclear masses and charge radii

    [https://arxiv.org/abs/2611.18066](https://arxiv.org/abs/2611.18066)

    该论文提出将可学习的乘积单元嵌入门控循环网络并在复数域中进行计算，从而显著提升了原子核质量与电荷半径的预测精度。

    

    使用机器学习预测原子核性质可以补充理论模型，并推动对核素图中知之甚少区域的探索。我们提出了一种基于门控循环单元（GRU）的机器学习技术来预测原子核质量和电荷半径，该技术将可学习的乘积单元映射集成到候选状态中，以建模相邻原子核之间的乘性依赖关系。计算在复数域中进行，以联合捕捉幅度和相位动力学。在三个数据划分及每个划分的三个训练种子上，AM-PU-GRU 取得了 0.2698 MeV 的平均质量插值 RMSE 和 0.1905 MeV 的时间外推 RMSE。配置对比实验支持将学习到的乘积整合进门控循环结构。在不使用质量输入的情况下，MI-PU-GRU 预测电荷半径的 RMSE 为 0.019947 fm，比固定的 WS* 先验低 14.40%。匹配的半径对照实验进一步揭示了候选……（原文摘要至此截断）

    arXiv:2611.18066v1 Announce Type: cross  Abstract: The prediction of nuclear properties using machine learning can complement theoretical models and advance the exploration of poorly known domains of the nuclear chart. We propose a machine learning technique for predicting nuclear masses and charge radii based on gated recurrent units (GRU) that integrates learnable product-unit mappings into the candidate state to model multiplicative dependencies among neighboring nuclei. Computations are performed in the complex domain to jointly capture amplitude and phase dynamics. Across three data partitions and three training seeds per partition, AM-PU-GRU achieves a mean mass interpolation RMSE of 0.2698 MeV and a temporal RMSE of 0.1905 MeV. Configuration comparisons support the integration of learned products into gated recurrence. Without mass inputs, MI-PU-GRU predicts charge radii with an RMSE of 0.019947 fm, 14.40% below the fixed WS* prior. Matched radius controls further show how candi
    
[^61]: 基于强化学习的钙钛矿太阳能电池富集导向虚拟筛选

    Enrichment-Oriented Virtual Screening of Perovskite Solar Cells through Reinforcement Learning

    [https://arxiv.org/abs/2611.18043](https://arxiv.org/abs/2611.18043)

    该论文提出一个以冻结TabPFN代理模型引导的强化学习框架，将钙钛矿太阳能电池配方生成建模为富集导向的虚拟筛选，使前十名候选配方的平均PCE从19.68%提升至23.74%，其中PPO和RPPO策略在前100名候选名单中获得了最高的邻域命中率。

    

    钙钛矿太阳能电池的实验优化受限于吸收层组成、器件层结构和制备工艺的组合多样性，这使得虚拟筛选在制备前对配方进行优先排序方面极具吸引力。我们开发了一个代理模型引导的强化学习框架，将配方生成转化为虚拟筛选，在高性能实验样本的邻域内富集候选配方短名单。一个冻结的表格先验数据拟合网络引导配方生成，同时一个包含890个实测器件的参考池支持基于邻域的评估。在1,846个器件上的五折评估表明，前十个选择可将平均功率转换效率（PCE）从19.68%提升至23.74%。在每次运行3,000次代理评估的条件下，将三种学习到的策略与六个基线方法进行了比较。在前100名候选名单中，近端策略优化（PPO）和循环PPO（RPPO）实现了邻域支持的命中率23%（原文在此处截断）。

    arXiv:2611.18043v1 Announce Type: new  Abstract: Experimental optimization of perovskite solar cells is constrained by combinatorial diversity in absorber compositions, device layers, and processing, which makes virtual screening attractive for prioritizing recipes before fabrication. We develop a surrogate-guided reinforcement-learning framework that casts recipe generation as virtual screening, enriching candidate shortlists near high-performing experimental neighbors. A frozen Tabular Prior-data Fitted Network (TabPFN) guides recipe generation, while a reference pool of 890 measured devices enables neighborhood-based assessment. Five-fold evaluation on 1,846 devices shows top-ten selection raises mean power conversion efficiency (PCE) from 19.68\% to 23.74\%. Three learned policies are compared with six baselines under 3,000 surrogate evaluations per run. In top-100 shortlists, Proximal policy optimization (PPO) and recurrent PPO (RPPO) achieve neighborhood-supported hit rates of 23
    
[^62]: RL$^{\mathrm{RT}}$：端侧强化学习运行时

    RL$^{\mathrm{RT}}$: On-Device Reinforcement Learning Runtime

    [https://arxiv.org/abs/2611.18035](https://arxiv.org/abs/2611.18035)

    RL$^{\mathrm{RT}}$通过将执行器与学习器的耦合变为可调参数（刷新界限与松弛感知的GPU许可），在共享嵌入式GPU上实现了高效的端侧强化学习，将16.8M参数策略的截止期限错过率从85%-100%降至1%以下。

    

    端侧强化学习在与其它工作负载共享的单个嵌入式GPU上运行周期性执行器和学习器。由于学习器会生成执行器的权重，现有设计要么阻塞执行器，要么导致其策略发生漂移。我们提出了RL$^{\mathrm{RT}}$，一个端侧运行时系统，它将这种耦合转化为一个可调参数：刷新界限限制刷新之间执行器的发布次数，而具有松弛感知准入机制的GPU许可则对执行器进行优先调度，并在任意基础调度器之上利用空闲时间进行训练。在两块Jetson开发板上考虑真实的刷新开销的实验中，RL$^{\mathrm{RT}}$将一个16.8M参数策略的截止期限错过率控制在1%以下，而没有使用它时该比率高达85%至100%。

    arXiv:2611.18035v1 Announce Type: cross  Abstract: On-device reinforcement learning runs a periodic actor and a learner on one embedded GPU shared with other workloads. Because the learner produces the actor's weights, existing designs stall the actor or let its policy drift. We present RL$^{\mathrm{RT}}$, an on-device runtime that makes this coupling a tunable parameter: a refresh bound limits actor releases between refreshes, and a GPU permit with slack-aware admission prioritises the actor and fills spare time with training atop any base scheduler. Under the real refresh cost on two Jetson boards, RL$^{\mathrm{RT}}$ keeps a 16.8 M-parameter policy's deadline misses below 1%, against 85 to 100% without it.
    
[^63]: 超越机器人标签：电商日志中基于后果的缓存危害归因回溯排序

    Beyond Bot Labels: Retrospective Consequence-Grounded Ranking of Attributed Cache Harm in E-Commerce Logs

    [https://arxiv.org/abs/2611.18027](https://arxiv.org/abs/2611.18027)

    该论文提出基于后果的会话排序方法，通过按时间顺序回溯重放将缓存驱逐造成的未命中归因到具体会话，从而识别并绕过对电商缓存命中率造成跨会话危害的自动化流量。

    

    网络缓存能够降低源站负载与延迟，但自动化流量既可能为缓存预热，也可能驱逐购物者重复使用的对象，从而降低缓存命中率（CHR）。现有的机器人防御方法通常通过User-Agent（UA）字符串或行为来推断自动化特征，而非识别那些造成跨会话缓存危害的会话。我们转而将该任务表述为按驱逐损害对会话进行排序，并提出基于后果的会话排序方法。通过一次按时间顺序的重放，将受保护购物会话（受害者）因驱逐而产生的每次未命中归因于最后插入并驱逐该对象的会话；行为模型仅凭会话行为即可预测该目标。被选中的会话绕过缓存，但仍由源站提供服务。在跨危害程度选取的539个非受害者会话的受控检验中，归因排序能够追踪移除后受害者命中情况的变化（Spearman相关系数为0.80）。在一个包含1270万条请求的真实追踪数据上，排名前1%的会话……

    arXiv:2611.18027v1 Announce Type: cross  Abstract: Web caches reduce origin load and latency, but automated traffic can either warm them or evict objects shoppers reuse, lowering the cache hit ratio (CHR). Existing bot defenses generally infer automation from User-Agent (UA) strings or behavior rather than identify sessions that impose cross-session cache harm. We instead formulate the task as ranking sessions by eviction damage and propose consequence-grounded session ranking. A single chronological replay attributes each eviction-induced miss by a protected shopping session (a victim) to the session whose insertion last evicted the object. A behavioral model predicts the target from session behavior alone. Selected sessions bypass the cache but remain origin-served. In a controlled check of 539 non-victim sessions selected across harm levels, attribution order tracks victim-hit changes after removal (Spearman correlation 0.80). On a 12.7-million-request real trace, the top 1% of sess
    
[^64]: 一种基于显式约束空间的可泛化安全感知机器人行为规划方法

    A Generalizable Approach to Safety-Aware Robotic Behavior Planning with an Explicit Constraint Space

    [https://arxiv.org/abs/2611.18026](https://arxiv.org/abs/2611.18026)

    该论文提出一种基于显式约束空间的安全感知机器人行为规划框架，通过任务与环境条件的适用性学习（结合轻量级双塔模型）来选择适用的安全规则并交给多模态规划器，从而提升桌面环境中机器人操作的安全性与泛化能力。

    

    大语言模型和视觉-语言模型已经极大地推动了基于自然语言指令和视觉观测的机器人行为规划。然而，在桌面环境中的安全操作仍然具有挑战性，因为任务所需的安全约束取决于周围场景，并且可能未在指令中被明确指定。我们提出了一种基于显式约束空间的安全感知行为规划框架，该约束空间存储可复用的文本形式安全规则。我们不再仅通过语义相似度来检索规则，而是将约束选择形式化为以任务和环境为条件的适用性学习。一个轻量级的、以查询为条件的双塔模型识别出适用于当前上下文的约束，随后将这些约束连同原始任务和图像一起提供给多模态行为规划器。为了处理错误的约束选择以及先前未见过的危险，我们引入……（原文摘要在此处截断）

    arXiv:2611.18026v1 Announce Type: cross  Abstract: Large language and vision-language models have substantially advanced robotic behavior planning from natural-language instructions and visual observations. However, safe manipulation in tabletop environments remains challenging because the safety constraints required for a task depend on the surrounding scene and may not be explicitly specified in the instruction. We propose a safety-aware behavior planning framework based on an explicit Constraint Space that stores reusable textual safety rules. Rather than retrieving rules solely by semantic similarity, we formulate constraint selection as task- and environment-conditioned applicability learning. A lightweight query-conditioned dual-tower model identifies the constraints applicable to the current context, which are then provided together with the original task and image to a multimodal behavior planner. To handle incorrect constraint selections and previously unseen hazards, we intro
    
[^65]: JaRS：基于雅可比行和的扩散模型记忆区域无对比定位方法

    JaRS: Contrast-Free Localization of Memorized Regions in Diffusion Models via Jacobian Row Sums

    [https://arxiv.org/abs/2611.18024](https://arxiv.org/abs/2611.18024)

    提出JaRS方法，利用去噪器雅可比矩阵的行和，无需参考图即可将扩散模型的记忆效应与局部平均区分开来，实现了无对比的记忆区域定位（JaRS+进一步结合图像级检测分数）。

    

    扩散模型有时会记忆并复现其训练图像中的部分区域，这引发了隐私和版权方面的担忧。我们的目标是定位这些区域。现有的基于对比的方法使用去噪器雅可比矩阵的对角线，它反映了每个像素的后验方差。然而，较小的对角线元素存在歧义：它既可能表示记忆效应，也可能表示在平滑或可预测区域中对许多相邻像素进行了求平均。为了解决这一歧义，这些方法需要减去一张参考图，这要求使用空提示或更早的检查点进行第二次评估。但无条件模型没有空提示，且更早的检查点通常不可获得。我们证明，与对角线不同，雅可比行和无需参考图即可将记忆效应与局部平均区分开来。将JaRS与一种在早期去噪步骤中使用同一统计量的图像级检测分数相结合，即得到JaRS+。在Stable Diffusion……（原文摘要在此处截断）

    arXiv:2611.18024v1 Announce Type: cross  Abstract: Diffusion models sometimes memorize and reproduce regions of their training images, raising privacy and copyright concerns. We aim to localize these regions. Existing contrast-based methods use the diagonal of the denoiser Jacobian, which reflects the posterior variance of each pixel. However, a small diagonal entry is ambiguous: it signals either memorization or averaging over many neighboring pixels in smooth or predictable regions. To resolve this ambiguity, these methods subtract a reference map, which requires a second evaluation with a null prompt or an earlier checkpoint. Unconditional models have no null prompt, and earlier checkpoints are unavailable in general. We show that Jacobian row sums (JaRS), unlike the diagonal, separate memorization from local averaging without a reference map. Pairing JaRS with an image-level detection score that uses the same statistic at an early denoising step yields JaRS+. Across Stable Diffusio
    
[^66]: 冲突、可行性、忠实性、位移：联邦双锥聚合中相互分离的四个量

    Conflict, Feasibility, Faithfulness, Displacement: Four Quantities That Come Apart in Federated Dual-Cone Aggregation

    [https://arxiv.org/abs/2611.18014](https://arxiv.org/abs/2611.18014)

    该论文通过仅改变更新方向、不改变幅度的 UPGrad 对照实验证明，联邦学习中多目标梯度聚合方法所依据的“消除客户端更新冲突”理论混淆了四个本质不同的量（冲突、可行性、忠实性、位移），实验表明消除冲突并不必然带来性能提升，均匀权重即可匹配其增益。

    

    arXiv:2611.18014v1 公告类型：新论文 摘要：FedFV、FedGH 和 FedMGDA+ 采用多目标梯度聚合器来聚合客户端更新，以消除客户端更新之间的冲突。我们使用 UPGrad 来检验这一理论依据——UPGrad 在一阶意义上不会增加任何客户端的损失，并且在对比实验中仅改变 FedAvg 更新的方向。该理论依据视为同一事物的四个量——客户端更新之间的冲突、其加权平均的可行性、传输更新对客户端梯度的忠实性、以及更新产生的位移——实际上会相互分离。在 Dirichlet 数据划分下，UPGrad 将 CIFAR-10 的准确率提升了 +0.78 个百分点（15 个随机种子中有 14 个提升，p < 0.001），但仅在所有客户端完全参与时成立；在 Oxford-IIIT Pet 数据集上，一旦保留了我们协议中未使用的数据，其 +0.52 个百分点的增益便消失了（仅剩 +0.07 个百分点）。仅改变 Dirichlet 浓度参数时，增益随数据异质性的增加而增长，然而冲突最严重的划分却未显示出任何增益。对冲突毫无感知的均匀客户端权重在 CIFAR-10 上达到了相同的增益，并超过了……

    arXiv:2611.18014v1 Announce Type: new  Abstract: FedFV, FedGH and FedMGDA+ aggregate client updates with multi-objective gradient aggregators to remove conflict between them. We examine this rationale with UPGrad, which increases no client's loss to first order, in a comparison that changes only the direction of FedAvg's update. Four quantities the rationale treats as one -- conflict between client updates, feasibility of their weighted mean, faithfulness of the transmitted updates to the clients' gradients, and displacement of the update -- come apart. Under Dirichlet partitioning UPGrad improves CIFAR-10 accuracy by +0.78 pp (14/15 seeds, p < 0.001), only with full participation; on Oxford-IIIT Pet its +0.52 pp gain vanishes (+0.07 pp) once the data our protocol left unused are kept. Varying only the Dirichlet concentration, the gain grows with heterogeneity, yet the split with the most conflict shows none. Uniform client weights, blind to conflict, match the CIFAR-10 gain and exceed
    
[^67]: 基于可微模拟的磁对流流动样本高效控制

    Sample-Efficient Control of Magnetoconvective Flows via Differentiable Simulation

    [https://arxiv.org/abs/2611.18008](https://arxiv.org/abs/2611.18008)

    本文提出φ-PICT，一种面向主动流动控制的可微分GPU加速磁流体动力学求解器，可将传热提升25%以上，并在逆向设计中实现比黑盒方法高10³倍的精度。

    

    磁流体动力学（MHD）流体流动在要求苛刻的工程应用中处于核心地位，最突出的是聚变包层冷却，其中流动控制是优化传热等关键特性的一种有前景的工具。虽然主动流动控制（AFC）近来已通过机器学习取得进展，但其在MHD流动中的应用在很大程度上仍未被探索。在这项工作中，我们提出了φ-PICT，这是一种针对AFC的新型可微分GPU加速MHD求解器。我们通过MHD基准测试验证了φ-PICT，并展示了其高计算效率。集成到FluidGym中后，φ-PICT使样本高效的基于梯度的算法能够将传热提高25%以上，并使优化器在逆向设计中比黑盒方法获得10³倍的更高精度。φ-PICT已在 https://github.com/upb-sciml/phiPICT 公开提供。

    arXiv:2611.18008v1 Announce Type: new  Abstract: Magnetohydrodynamic (MHD) fluid flows are central to demanding engineering applications, most prominently fusion blanket cooling, where flow control is a promising tool to optimize key properties such as heat transfer. While active flow control (AFC) has recently advanced through machine learning, its application to MHD flows remains largely unexplored. In this work, we present $\phi$-PICT, a novel differentiable GPU-accelerated MHD solver targeted at AFC. We validate $\phi$-PICT against MHD benchmarks and demonstrate its high computational efficiency. Integrated into FluidGym, $\phi$-PICT enables sample-efficient gradient-based algorithms to improve heat transfer by $>25$% and allows optimizers to achieve $10^3\times$ higher accuracy in inverse design than black-box methods. $\phi$-PICT is publicly available at https://github.com/upb-sciml/phiPICT.
    
[^68]: RiverScan：学习在河网上进行累积

    RiverScan: Learning to Accumulate on River Networks

    [https://arxiv.org/abs/2611.18005](https://arxiv.org/abs/2611.18005)

    提出 RiverScan——一种基于面积加权选择性状态空间递归的空间算子，按拓扑顺序学习河网上的信息累积，在多瑙河流域 608 个站点的径流预测中超越了时序与图基线方法。

    

    准确的大流域径流预测要求模型能够捕捉异质性的局地水文响应及其向下游的累积过程。河网连通性为这一任务提供了天然的结构，但要有效利用它，需要学习如何合并来自不同尺度子流域的贡献。我们将这一挑战表述为“学习累积”，并提出 RiverScan——一种基于面积加权选择性状态空间递归的空间算子。RiverScan 按拓扑顺序递归地将局地时间表示与上游累积状态相结合。汇流面积比例用于刻画子流域的大小，而可学习的持留机制则控制上游信息向下游的持续传递。在多瑙河流域 608 个站点的实验表明，RiverScan 在整体预测性能上优于所评估的时序模型与图基线方法。结构性控制实验与消融研究证明了……

    arXiv:2611.18005v1 Announce Type: new  Abstract: Accurate runoff prediction across large river basins requires models to capture heterogeneous local hydrological responses and their accumulation downstream. River-network connectivity provides a natural structure for this task, but using it effectively requires learning how to combine contributions from subcatchments of different sizes. We frame this challenge as learning to accumulate and introduce RiverScan, a spatial operator based on area-weighted selective state-space recurrence. RiverScan recursively combines local temporal representations with accumulated upstream states in topological order. Contributing-area fractions account for subcatchment size, while learned retention controls how upstream information persists downstream. Experiments on 608 stations in the Danube basin show that RiverScan outperforms the evaluated temporal and graph baselines in overall predictive performance. Structural controls and ablations demonstrate t
    
[^69]: LACCO：带计算-通信重叠的本地累积

    LACCO: Local Accumulation with Computation--Communication Overlap

    [https://arxiv.org/abs/2611.17991](https://arxiv.org/abs/2611.17991)

    提出LACCO两阶段算法，在不引入外层更新陈旧性的前提下实现计算与通信重叠并保持各副本更新一致性，在语言模型预训练中持续优于Eager和Delayed DiLoCo等通信重叠基线。

    

    在低带宽网络上训练语言模型需要减少同步开销。诸如 DiLoCo 等本地优化方法仅在执行若干本地步骤后才进行通信，但外层梯度平均和外层更新仍然处于关键路径上。现有的计算-通信重叠方法通过延迟更新或副本本地更新来避免这种停滞，但这可能损害收敛性能或导致副本之间产生漂移。我们提出了 LACCO（带计算-通信重叠的本地累积），这是一种两阶段算法，它将全局通信与本地计算重叠，同时应用由所有副本共享的全局聚合更新，而不会在外层更新中引入陈旧性。我们为光滑非凸目标建立了收敛性保证。在使用 FineWeb 数据集进行的语言模型预训练实验中，LACCO 持续优于 Eager 和 Delayed DiLoCo 等通信重叠基线方法。

    arXiv:2611.17991v1 Announce Type: new  Abstract: Training language models across low-bandwidth networks requires reducing synchronization overhead. Local optimization methods such as DiLoCo communicate only after several local steps, but outer gradient averaging and the outer update remain on the critical path. Existing computation-communication overlap methods avoid this stall through delayed or replica-local updates, which can degrade convergence or cause replicas to drift. We propose LACCO (Local Accumulation with Computation-Communication Overlap), a two-stage algorithm that overlaps global communication with local computation while applying globally aggregated updates shared by every replica, without introducing staleness in the outer update. We establish convergence guarantees for smooth non-convex objectives. Experiments on language-model pre-training with FineWeb show that LACCO consistently outperforms communication-overlapped baselines such as Eager and Delayed DiLoCo. These 
    
[^70]: 何时过采样才有帮助？标签延迟与概念漂移下不平衡数据流的决策框架

    When Does Oversampling Help? A Decision Framework for Imbalanced Data Streams Under Label Delay and Concept Drift

    [https://arxiv.org/abs/2611.17981](https://arxiv.org/abs/2611.17981)

    该论文提出了一个围绕漂移密度、少数类可学习性和延迟相对于细化窗口时长这三个维度构建的决策框架，用于预测在标签延迟与概念漂移并存的不平衡数据流中，流式过采样何时能比持续重评估带来额外价值。

    

    对于标签延迟的不平衡数据流，添加一个流式过采样器是否值得，还是对早期预测进行持续重评估已经足以弥补延迟带来的代价？我们通过在类别不平衡、概念漂移与标签延迟三者共同作用的条件下，对流式过采样与持续重评估进行广泛的实验比较来回答这一问题。流式过采样器通常在标签即时可得的“先测试后训练”（test-then-train）假设下被评估；而在此条件下通过重评估来考察其行为的研究尚未被系统地开展过。我们通过一个决策框架填补了这一空白，该框架能够预测过采样何时比重评估更有价值，围绕三个维度组织：漂移密度、少数类的可学习性，以及延迟相对于细化窗口的时长，并将该问题具体化为两条通道之间的竞争——恢复通道（重评估）与少数类覆盖通道（过采样）。在对15个分类器（原文摘要在此截断）……

    arXiv:2611.17981v1 Announce Type: new  Abstract: For imbalanced data streams with delayed labels, is it worth adding a streaming oversampler, or does continuous re-evaluation of earlier predictions already recover the cost of the delay? We address this through an extensive experimental comparison of streaming oversampling and continuous re-evaluation under joint class imbalance, concept drift, and label delay. Streaming oversamplers are typically evaluated under the test-then-train assumption of immediate labels; their behaviour under these conditions, assessed through re-evaluation, has not been systematically studied. We close this gap with a decision framework that predicts when oversampling adds value over re-evaluation, organised around three axes: drift density, minority-class learnability, and delay relative to the refinement window, and operationalised as a competition between a recovery channel (re-evaluation) and a minority-coverage channel (oversampling). Evaluating $15$ cla
    
[^71]: 算法不透明性作为委托-代理问题

    Algorithmic Opacity as a Principal-Agent Problem

    [https://arxiv.org/abs/2611.17980](https://arxiv.org/abs/2611.17980)

    该论文提出基于委托-代理理论的检验框架，主张即使存在博弈操纵和商业秘密的顾虑，算法决策者也应披露错误概况和群体间决策结果分布这两项诊断指标，以防止其利用不透明性掩盖偏见、低效或损害公共利益的策略性行为。

    

    对博弈操纵和商业秘密的担忧经常被用来为算法不透明性辩护，但这些担忧也可能掩盖决策者自利的策略性行为——他们可能将私人利益置于社会福利之上。我们开发了一种委托-代理检验方法，用于判断即使存在博弈操纵或商业秘密的顾虑，算法决策何时仍应触发披露义务。该检验以决策质量为基础，依赖于两项可在不助长博弈操纵或不泄露商业秘密的前提下披露的诊断指标：错误概况（假阳性率和假阴性率，包括总体水平和各群体分别的水平）以及各群体间决策结果的分布。这些诊断指标能够揭示决策者是否使算法的设计与实施符合公共利益，还是相反地利用不透明性来掩盖偏见、低效或策略性推诿。社会群体间的显著差异以及决策结果分布的不对称性……（原文摘要在此处截断）

    arXiv:2611.17980v1 Announce Type: cross  Abstract: Concerns about gaming and trade secrecy, frequently invoked to justify algorithmic opacity, can also mask self-serving strategic behavior by decision-makers, who may prioritize private benefits over social welfare. We develop a principal-agent test for when algorithmic decision-making should trigger disclosure obligations even when concerns about gaming or trade secrecy exist. The test, which is anchored in decision quality, relies on two diagnostics that can be disclosed without enabling gaming or revealing trade secrets: the error profile (false positive and false negative rates, overall and for each group) and the distribution of decision outcomes across groups. These diagnostics make visible whether decision-makers align design and implementation with the public interest or instead use opacity to obscure bias, inefficiency, or strategic shirking. Strong disparities across social groups, and asymmetries in the distribution of decisi
    
[^72]: 因果图中变量的高效实用聚类

    Efficient and Practical Clustering of Variables in Causal Graphs

    [https://arxiv.org/abs/2611.17971](https://arxiv.org/abs/2611.17971)

    本文提出了一类称为“导管”的新型聚类，并给出检测因果图中所有导管的可靠、完备且高效算法，该类聚类在真实世界因果图中自然出现，可大幅缩减图规模并支持已知的可识别性结果。

    

    对图中的变量进行聚类可以提高视觉清晰度并简化因果推断。要充分发挥聚类的潜力，需要一种自动化方法来检测图中有效的聚类。我们引入了一类称为“导管”的新型聚类，并提出了一种可靠、完备且高效的算法来检测图中的所有导管。除了算法上的可检测性之外，导管在源自现实世界应用的因果图中也自然出现，从而能够大幅缩减图的规模。此外，导管的结构使得关于聚类的已知可识别性结果得以应用。我们通过基于真实世界图的模拟研究验证了这些特性。

    arXiv:2611.17971v1 Announce Type: cross  Abstract: Clustering variables in a graph may improve visual clarity and simplify causal inference. Harnessing the full potential of clustering requires an automated method to detect useful clusters in a graph. We introduce a novel class of clusters called conduits, and present a sound, complete and efficient algorithm for detecting all conduits in the graph. In addition to algorithmic detectability, conduits appear naturally in causal graphs stemming from real-world applications, enabling substantial size reduction. Furthermore, the structure of conduits enables the application of known identifiability results regarding clustering. We demonstrate these properties in a simulation study with real-world graphs.
    
[^73]: 分析扩散模型中损失空间与预测目标选择对学习动态的影响

    Analyzing Learning-Dynamics Across Loss and Prediction Choices for Diffusion Models

    [https://arxiv.org/abs/2611.17967](https://arxiv.org/abs/2611.17967)

    本文解耦了扩散模型中预测目标与损失空间的影响，证明改变损失空间恰好等价于时间步重加权，并从理论上揭示了秩-$m$ 线性瓶颈会使噪声和速度预测存在 $D-m$ 的不可约平方误差下限，而干净数据预测则可利用低维数据结构。

    

    去噪生成模型可以预测干净数据（$x$）、噪声（$\epsilon$）或速度（$v$），而训练损失则可以在这些空间中的任意一个中进行度量。尽管由此产生的九种组合在代数上彼此相关，但它们可能在去噪精度和样本质量上带来显著差异。本文解耦了预测目标与损失空间的影响。对于固定的预测目标，我们证明改变损失空间恰好等价于对时间步进行重新加权，并对这种重加权如何影响生成质量给出了启发式分析。随后，我们证明：一个将 $D$ 维含噪输入压缩的秩为 $m$ 的线性瓶颈，会给噪声和速度预测带来 $D-m$ 的不可约平方误差下限，而干净数据预测则可以利用数据的低维结构。将该下限与依赖于预测目标的误差转换相结合，可以得到关于瓶颈容量的不同必要条件。使用 U-（此处摘要被截断）

    arXiv:2611.17967v1 Announce Type: new  Abstract: Denoising generative models predict clean data ($x$), noise ($\epsilon$), or velocity ($v$), while training losses can be measured in any of these spaces. Although the resulting nine combinations are algebraically related, they can yield substantially different denoising accuracy and sample quality. We disentangle the effects of prediction and loss spaces. For a fixed prediction target, we show that changing the loss space is exactly equivalent to timestep reweighting and provide a heuristic analysis of how this reweighting affects generation quality. We then show that a rank-$m$ linear bottleneck compressing a $D$-dimensional noisy input imposes an irreducible squared-error floor of $D-m$ on noise and velocity prediction, whereas clean-data prediction can exploit low-dimensional data structure. Combining this bound with prediction-dependent error conversion yields distinct necessary conditions on bottleneck capacity. Experiments with U-
    
[^74]: DeltaVisionX：一种基于YOLOv11与PLC驱动的低成本视觉引导自动化系统

    DeltaVisionX: A Low-Cost Vision-Guided Automation System Using YOLOv11 and PLC-Based Actuation

    [https://arxiv.org/abs/2611.17959](https://arxiv.org/abs/2611.17959)

    DeltaVisionX通过编码器同步的YOLOv11视觉检测与PLC气动驱动集成，实现了低成本、高精度（mAP 0.97）和高吞吐量（每分钟80件）的自动化物体分拣系统。

    

    本文提出了DeltaVisionX，这是一种低成本的工业自动化系统，它将实时计算机视觉与基于PLC的驱动控制相结合，用于物体的分类与分拣。该系统使用YOLOv11（You Only Look Once）深度学习模型检测运行中传送带上的物体，并将检测结果传递给Delta DVP-SE PLC，由PLC驱动气动活塞完成分拣。研究团队创建、标注并增强了一个自定义数据集用于训练YOLOv11模型，实现了0.97的平均精度均值（mAP），实时推理速度为每帧100毫秒。基于编码器的跟踪机制确保了检测与驱动执行之间的可靠同步，支持每分钟80个物体的处理吞吐量。结果表明，经济实惠的本地化开发自动化系统能够减少对人工的依赖并提高制造效率。所提系统的新颖之处在于其编码器同步的视觉-PLC集成架构，而非检测算法本身。

    arXiv:2611.17959v1 Announce Type: new  Abstract: This paper presents DeltaVisionX, a low-cost industrial automation system that integrates real-time computer vision with PLC-based actuation for object classification and sorting. The system uses the YOLOv11 (You Only Look Once) deep learning model to detect objects on an operating conveyor belt and communicates detection results to a Delta DVP-SE PLC, which actuates pneumatic pistons for sorting. A custom dataset was created, labeled, and augmented to train the YOLOv11 model, achieving a mean average precision of 0.97 with real-time inference of 100 ms per frame. Encoder-based tracking ensures reliable synchronization between detection and actuation, supporting a throughput of 80 objects per minute. Results show that affordable, locally developed automation can reduce labor dependency and improve manufacturing efficiency. The novelty of the proposed system lies in its encodersynchronized vision-PLC integration rather than in the detecti
    
[^75]: 推理模型会编造理由来捍卫被注入的关于自身请求的信念

    Reasoning Models Confabulate to Defend Injected Beliefs About Their Own Requests

    [https://arxiv.org/abs/2611.17958](https://arxiv.org/abs/2611.17958)

    通过在提示文本中不可见的激活编辑注入离题概念，可以切断推理模型对自身请求的信念锚定，使其在被明确警告真相时仍会编造理由为被注入的信念辩护并据此作答。

    

    思维链（CoT）监测与自我报告被提议作为AI监督的抓手，其前提假设是：模型对自身行为的叙述锚定于它实际被问到的请求。我们证明这一锚定可以被切断。一种在提示文本中不可见的激活编辑——注入一个离题概念的形成痕迹——会覆盖推理模型对自身请求的信念：在横跨规模、架构和训练配方的六个模型上，被注入的主题变成了模型所推理、报告和回答的任务：强制引用回述在48个模型-概念组合中的47个都返回了被注入的主题；在四个最终俳句干净的模型上，99%的生成结果都跟随了被注入的答案。警告几乎不起作用：即使警告中点明了真实主题，模型给出真实主题答案的比例也不足5%。当以干净的token形式把真相直接交给模型时，模型会为用户编造一个故事（“也许他们搞混了然后自己纠正了”），并用被注入的内容进行回答。

    arXiv:2611.17958v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring and self-report are proposed as levers for AI oversight, on the assumption that a model's account of what it is doing is anchored to what it was asked. We show this anchor can be cut. An activation edit invisible in the prompt text, the injected forming-trace of an off-topic concept, overwrites a reasoning model's belief about its own request: across six models spanning scale, architecture, and recipe, the injected topic becomes the task it reasons about, reports back, and answers: the forced quote-back returns the injected topic in 47 of 48 model-concept cells, and on the four models with clean final haiku the injected answer follows in 99% of generations. Warnings barely move it: even one naming the true topic yields under 5% true-topic answers. Handed the truth in clean tokens, the model confabulates a story about the user ("maybe they got confused and corrected themselves") and answers with the injec
    
[^76]: WebArena-Pro：一个面向Web智能体的异构、多模态、可复现基准测试

    WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents

    [https://arxiv.org/abs/2611.17955](https://arxiv.org/abs/2611.17955)

    WebArena-Pro是一个涵盖20个自托管Web应用、300个任务的异构多模态可复现基准测试，首次支持对Web智能体理解和运用音频、视频内容能力的评估。

    

    基于大语言模型和视觉-语言模型的Web智能体正越来越多地被应用于涉及异构应用、多模态内容和有状态工作流的现实浏览器任务中。然而，现有的可复现Web智能体基准测试仅覆盖来自少数软件类别的少量Web应用，且模态仅限于文本和视觉。实时基准测试虽然扩大了站点覆盖范围，但由于页面和数据在运行之间会发生漂移，牺牲了可复现性。此外，现有基准测试未能有效评估智能体能否理解和使用嵌入在Web任务中的音频和视频内容。为弥补这些不足，我们提出了WebArena-Pro，这是一个包含300个任务的基准测试，涵盖六个领域类别中的20个自托管Web应用，跨越不同的界面惯例、工作流和数据模型。在所评估的智能体中，Codex with GPT 6.1 Sol 取得了最佳表现。

    arXiv:2611.17955v1 Announce Type: new  Abstract: Web agents powered by large language and vision-language models are increasingly applied to realistic browser work that spans heterogeneous applications, multimodal content, and stateful workflows. However, existing reproducible web-agent benchmarks cover only a small number of web applications drawn from a few software categories, and restrict modality to text and vision. Live benchmarks broaden site coverage but sacrifice reproducibility, since pages and data drift between runs. Moreover, existing benchmarks do not meaningfully evaluate whether agents can understand and use audio and video content embedded within web tasks. To address these gaps, we introduce WebArena-Pro, a benchmark comprising 300 tasks across 20 self-hosted web applications in six domain categories, spanning distinct interface conventions, workflows, and data models. Across the evaluated agents, the best performance is achieved by both Codex with GPT 6.1 Sol and Cla
    
[^77]: 基于域适应的字符分类与手写文本识别流水线

    Domain Adaptation Based Pipeline for Character Classification and Handwritten Text Recognition

    [https://arxiv.org/abs/2611.17952](https://arxiv.org/abs/2611.17952)

    该论文提出了一种基于无监督域适应的三步流水线，通过对齐源域与目标域的特征分布并利用未标注目标数据，有效提升了标注数据有限情况下字符分类与手写文本识别的跨数据集泛化能力。

    

    字符分类和手写文本识别，特别是在专门化任务中，常常受到标注数据有限的困扰，而大量未标注数据仍然未被利用。为了解决这一挑战，我们提出了一种新颖的基于无监督域适应的流水线，以缩小源域和目标域之间的分布差距。我们的流水线遵循三个主要步骤：第一步，我们在源数据上预训练一个专家模型，以学习强大的通用表示；第二步，我们执行无监督域适应，利用未标注的目标数据，使模型适应目标域的特定特征；第三步，我们在目标风格的源数据上训练专家模型。通过在源域和目标域之间对齐特征分布，我们的方法使模型能够在不同的数据集上有效泛化。无监督训练增加了计算时间……

    arXiv:2611.17952v1 Announce Type: cross  Abstract: Character classification and handwritten text recognition, particularly in specialized tasks, often suffer from limited labeled data, while a large amount of unlabeled data remains unused. To address this challenge, we propose a novel unsupervised domain adaptation-based pipeline to reduce the distribution gap between source and target domains. Our pipeline follows three main steps. In the first step, we pretrained an expert model on source data, to learn a strong general representation. In the second step, we perform unsupervised domain adaptation to leverage unlabeled target data, adapting the model to the specific characteristics of the target domain. In the third step, we trained the expert model on source data in the target style. By aligning the feature distributions between the source and target, our method enables the model to generalize effectively across diverse datasets. The unsupervised training increases the computation ti
    
[^78]: TIDE：重拾在策略强化学习中损失的性能

    TIDE: Reclaiming Lost Performance In On-Policy Reinforcement Learning

    [https://arxiv.org/abs/2611.17924](https://arxiv.org/abs/2611.17924)

    TIDE通过动态复用空闲推演GPU进行训练以及在推理引擎之间迁移推演任务，在保持在策略训练语义的前提下，将LLM强化学习后训练的端到端性能提升高达1.36倍。

    

    arXiv:2611.17924v1 公告类型： new 摘要：当前，大型语言模型（LLM）的同步共置式强化学习（RL）后训练由于长尾推演的存在而遭受严重的资源利用率不足问题。推理引擎要么在等待长尾推演完成时处于空闲状态，要么以较低且低效的批量大小进行解码。为解决这一问题，我们提出了TIDE——一个在保持在策略训练语义的同时恢复资源利用率的强化学习训练系统。TIDE通过两项互补技术实现这一目标：（1）在长尾推演完成期间，动态地将用于推演的GPU重新分配用于训练；（2）在推理引擎之间动态迁移推演任务以提升解码吞吐量。TIDE通过自适应迁移策略自动确定重新利用推演资源的最佳时机，所获得的性能提升能够泛化至多种应用和硬件配置，而无需定制化配置。在我们的评估中，TIDE实现了高达1.36倍的端到端加速。

    arXiv:2611.17924v1 Announce Type: new  Abstract: Synchronous, colocated RL post-training of large language models (LLMs) today suffers from severe under-utilization due to long-tail rollouts. Inference engines fall idle waiting for long-tail rollouts or decode at low, inefficient batch sizes. To address this, we present TIDE, an RL training system that preserves on-policy training semantics while recovering resource utilization. TIDE does so by two complementary techniques: (1) dynamically repurposing rollout GPUs for training while long-tail rollouts complete and (2) dynamically migrating rollouts between inference engines to boost decoding throughput. TIDE automatically determines the right time to repurpose rollout resources through an adaptive migration policy, obtaining performance improvements that generalize across multiple applications and hardware configurations without requiring bespoke configuration. In our evaluation, TIDE provides end-to-end speedups of up to 1.36x over co
    
[^79]: 安全信息凝聚定律

    Condensing Law of Security Information

    [https://arxiv.org/abs/2611.17918](https://arxiv.org/abs/2611.17918)

    本文指出缩放定律在安全领域失效——模型越强越会在下游安全模块察觉前“悄悄修复”异常输入，并首次将其形式化为认知能力与安全信息保持率之间的权衡，推导出三个信息论界。

    

    缩放定律表明，在数据充足的情况下，大语言模型（LLM）的泛化误差随模型规模单调递减，“更大模型优于更小模型”这一推论现已被广泛接受。我们论证这一推论在安全领域并不成立：安全决策依赖于异常输入，而异常输入呈长尾分布（例如零日攻击），因此似然拟合难以有效处理此类输入。更糟糕的是，规模所带来的能力（例如从受损的部分推断出原本完整的整体）会使LLM在任何下游安全模块察觉之前就悄悄地“修复”异常。我们将这一现象形式化为两个量之间的权衡：模型对世界分布的认知能力 K(W)=I(W;P)，以及在其中间表示中得以保留的安全信息保持率 R(W)=I(Z;S)/I(A;S)。我们推导了三个信息论界来连接……

    arXiv:2611.17918v1 Announce Type: cross  Abstract: Scaling laws indicate that the generalization error of a large language model (LLM) decreases monotonically with model size once data is sufficient, and the corollary that larger models outperform smaller ones is now widely accepted. We argue that this corollary fails in security: decisions there hinge on anomalous inputs, which are long-tailed (e.g., zero-day attacks) and therefore poorly served by likelihood fitting. Worse still, the capability brought about by scale (e.g., inferring the intended whole from a corrupted part) leads a LLM silently \emph{repair} the anomaly before any downstream security head ever sees it. We formalize this as a tradeoff between two quantities: the cognitive capability $K(W)=I(W;P)$ that a model holds about the world distribution, and the security-information retention $R(W)=\frac{I(Z;S)}{I(A;S)}$ that survives in its intermediate representation. We derive three information-theoretic bounds to connect t
    
[^80]: 面向超弹性本构模型群体的分层贝叶斯物理增强神经网络：从稀疏、含噪数据中获取样本间变异性与不确定性

    Hierarchical Bayesian physics-augmented neural networks for populations of hyperelastic laws: specimen variability and uncertainty from scarce, noisy data

    [https://arxiv.org/abs/2611.17892](https://arxiv.org/abs/2611.17892)

    提出一种分层贝叶斯物理增强神经网络（非线性混合效应模型），可从稀疏、含噪数据中联合推断软组织等材料超弹性本构模型的群体分布、各样本个体差异及其不确定性。

    

    软组织等材料的超弹性本构关系在不同样本之间存在差异，而通常只能对少数样本进行测试。物理增强神经网络（PANNs）能够在不预设函数形式的情况下从数据中学习此类本构关系，同时保持其物理合理性；相关的贝叶斯模型版本则可量化单个样本或一组汇集样本的本构模型的不确定性。然而，汇集拟合会将样本之间的差异归因于测量噪声或对平均本构模型的不确定性，这两者都无法描述一个新的样本。我们首先针对单个样本开发了一种贝叶斯PANN，其先验设计保证了在数据稀少且含噪的情况下所学到的本构关系仍然物理合理。随后，我们将其扩展为分层贝叶斯PANN，即一种非线性混合效应模型：所有样本共享学习得到的多凸特征，而每个样本则是从本构模型总体中抽取的一个样本。特征、样本与总体通过变分推断进行联合推断……

    arXiv:2611.17892v1 Announce Type: cross  Abstract: The hyperelastic law of a material such as soft tissue differs between specimens, and often only a few specimens can be tested. Physics-augmented neural networks (PANNs) learn such laws from data without a prescribed functional form while keeping them physically admissible, and Bayesian versions of related models quantify the uncertainty of the law of one specimen or of a pooled group of specimens. A pooled fit, however, attributes the differences between specimens to measurement noise or to uncertainty about a mean law, and neither describes a new specimen. We first develop a Bayesian PANN for one specimen, whose priors keep the learned law plausible for few, noisy data. We then extend it to a hierarchical Bayesian PANN, a nonlinear mixed-effects model in which all specimens share learned polyconvex features and each specimen is a draw from a population of laws. Features, specimens, and population are inferred jointly by variational i
    
[^81]: 量子图神经网络：架构、可训练性与应用

    Quantum Graph Neural Networks: Architectures, Trainability, and Applications

    [https://arxiv.org/abs/2611.17886](https://arxiv.org/abs/2611.17886)

    本综述提出了描述量子图神经网络的统一形式化框架，将现有文献归纳为五大架构家族，并从量子比特成本、对称性、可扩展性和证据强度等方面对它们进行系统比较。

    

    量子图神经网络（QGNNs）将参数化量子电路应用于图结构数据，旨在克服经典图神经网络在可扩展性、表达能力以及长程信息流动方面的局限。自2019年首个基于哈密顿量的模型问世以来，该领域已涌现出众多架构、应用及相关论断，但其背后的证据尚未得到系统性评估。本综述引入了一个统一的形式化框架，将每个QGNN描述为编码器、图条件化量子电路、可观测量以及可选的经典后处理模块，并给出了置换等变性的电路级条件以及用于比较成本的资源模型。我们将相关文献归纳为五大架构家族（卷积型、等变型、消息传递与循环型、哈密顿型以及混合型），并从量子比特成本、对称性、可扩展性和证据强度等方面对其进行比较。随后，我们综述了对称性与几何归纳偏置等相关内容。

    arXiv:2611.17886v1 Announce Type: new  Abstract: Quantum graph neural networks (QGNNs) apply parametrized quantum circuits to graph-structured data, with the aim of overcoming limits of classical graph neural networks in scalability, expressivity, and long-range information flow. Since the first Hamiltonian-based model in 2019, the field has produced many architectures, applications, and claims, but the evidence behind them has not been assessed systematically. This survey introduces a unified formalism that describes every QGNN as an encoder, a graph-conditioned circuit, observables, and an optional classical head, together with a circuit-level condition for permutation equivariance and a resource model for comparing costs. We organize the literature into five architecture families (convolutional, equivariant, message-passing and recurrent, Hamiltonian, and hybrid) and compare them by qubit cost, symmetry, scalability, and evidence. We then review symmetry and geometric inductive bias
    
[^82]: 从局部预测误差到最终输出：多步推理中的专家共享理论

    From Local Prediction Error to Final Outputs: A Theory of Expert Sharing in Multi-Step Reasoning

    [https://arxiv.org/abs/2611.17879](https://arxiv.org/abs/2611.17879)

    该论文建立了多步推理中专家共享的理论框架，通过权衡方差节省与偏差推导出共享何时有效的精确准则，并揭示了一个关键现象：共享专家可能降低局部预测误差却反而增大最终输出误差。

    

    混合专家模型可以在层间或计算步骤间共享专家，汇集来自不同上下文的训练信息。这可以降低局部预测误差，但多步系统最终是以其最终输出来评估的。我们研究了这些局部收益何时能在复合过程中得以保留。在指定的线性回归模型中，我们通过权衡方差节省与跨不同目标行为的角色共享所带来的偏差，推导出精确的共享与合并准则。在共同固定线性传播映射下，我们刻画了期望平方最终输出风险，并保留了误差方向和相关性。我们构造了共享降低平均局部误差却增加最终输出误差的案例。在此设定下，我们还识别了带各向同性噪声的正交完全池化在何种条件下能在最终输出处保留所有局部改进。对于一般的路由计算，我们推导出条件端到端误差界。

    arXiv:2611.17879v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models can share experts across layers or computational steps, pooling training information from different contexts. This can reduce local prediction error, but multi-step systems are ultimately evaluated by their final outputs. We study when these local gains survive composition. In specified linear regression models, we derive exact sharing and merging criteria by balancing variance savings against bias from sharing across roles with different target behavior. Under a common fixed linear propagation map, we characterize expected squared final-output risk, retaining error directions and correlations. We construct cases where sharing lowers average local error yet increases final-output error. Within this setting, we also identify when orthogonal full pooling with isotropic noise preserves every local improvement at the final output. For general routed computations, we derive conditional end-to-end error bounds
    
[^83]: Muon预训练中临时权重衰减的延迟效应

    Delayed Effects of Temporary Weight Decay in Muon Pretraining

    [https://arxiv.org/abs/2611.17876](https://arxiv.org/abs/2611.17876)

    该研究揭示了Muon预训练中临时权重衰减的延迟效应——早期衰减虽在结束时增加验证损失，但通过留下持久的权重范数赤字使相对步长保持较大，最终反而降低最终损失，且小规模模型上选出的衰减调度无需重新调参即可成功迁移到更大规模模型。

    

    一种临时的优化干预在其结束时可能会增加验证损失，但在最终训练预算下却能降低验证损失。我们通过在具有162M、405M、1.4B和3.4B参数的Muon训练的仅解码器Transformer中使用临时权重衰减来证明这一点。对于参数量高达1.4B的模型，将固定时长的衰减窗口从训练早期移动到训练后期会逆转其最终效果。在162M参数规模下，即使在总收缩量大致匹配的情况下，改变窗口持续时间也会改变最终损失。最引人注目的是，在162M规模下选择的调度方案在未经重新调参的情况下迁移到1.4B规模时，在移除时比无衰减差0.26 nats，仅在训练进行到73%之后才超越它，最终以好0.028 nats的成绩收尾，同时还超越了采样得到的最佳常数衰减基线。早期衰减会留下持久的权重范数赤字，这使得干预结束后的相对步长保持较大。一个近似的范数动力学递推关系刻画了这种缓慢收敛的现象。

    arXiv:2611.17876v1 Announce Type: cross  Abstract: A temporary optimization intervention can increase validation loss when it ends yet reduce it at the final training budget. We demonstrate this with temporary weight decay in Muon-trained decoder-only Transformers with 162M, 405M, 1.4B, and 3.4B parameters. For models up to 1.4B parameters, moving a fixed-duration decay window from early to late training reverses its final effect. At 162M, changing window duration alters final loss even at approximately matched total shrinkage. Most strikingly, a schedule selected at 162M and transferred without retuning to 1.4B is 0.26 nats worse than no decay when removed, overtakes it only after 73% of training, and ultimately finishes 0.028 nats better while also outperforming the best sampled constant-decay baseline. Early decay leaves a persistent weight-norm deficit, which keeps relative steps larger after the intervention ends. An approximate norm-dynamics recurrence captures the slow convergen
    
[^84]: 基于流程挖掘的工业物联网联邦异常检测

    Process Mining-Enabled Federated Anomaly Detection for the Industrial IoT

    [https://arxiv.org/abs/2611.17869](https://arxiv.org/abs/2611.17869)

    该论文提出一种基于流程挖掘的联邦异常检测方法，通过评估本地节点数据与全局参考数据的相似度来动态加权聚合各本地模型，从而在非独立同分布的工业物联网时间序列数据上实现个性化更新并降低误报率。

    

    联邦异常检测虽然能够整合多个节点的本地数据洞察以减少误报，但其固有难点在于应对非独立同分布数据。在实际中，工业物联网工厂往往在不同条件下运行，并经历多样的环境现象，导致被监测的时间序列存在显著异质性。在这种环境下，标准的联邦学习方法无法进行充分考虑节点间差异的细致聚合，最终导致误报率上升。本文提出一种基于流程挖掘的方法，用于评估本地节点数据与全局节点所提供参考数据之间的相似度。全局节点利用这些相似度指标在聚合过程中对本地模型进行动态加权，从而产生分布到整个网络的个性化更新。我们使用从制造工厂收集的工业物联网时间序列数据对该方法进行了评估。

    arXiv:2611.17869v1 Announce Type: new  Abstract: While federated anomaly detection enables the integration of local data insights across diverse nodes to reduce false positives, it inherently struggles with non-IID data. In practice, industrial IoT plants often operate under varying conditions and experience diverse environmental phenomena, resulting in significant heterogeneity in the monitored time series. In such settings, standard federated learning approaches fail to perform nuanced aggregation that accounts for these cross-node differences, ultimately leading to increased false positive rates. In this paper, we propose a process mining approach to evaluate the similarity of local node data against a reference provided by a global node. The global node utilizes these similarity metrics to dynamically weight local models during aggregation, producing personalized updates distributed across the network. We evaluate our approach using industrial IoT time-series data collected from ma
    
[^85]: 将遗忘与效用解耦：基于正交零空间投影的轻量级大语言模型遗忘

    Disentangling Forgetting from Utility: Lightweight LLM Unlearning via Orthogonal Null-Space Projection

    [https://arxiv.org/abs/2611.17864](https://arxiv.org/abs/2611.17864)

    UUE框架通过将参数更新投影到保留知识的正交左零空间并引入具有闭式解的凸无目标目标函数，实现了在不损害模型通用能力前提下的高效轻量级LLM知识遗忘。

    

    大语言模型（LLM）因保留私人或危险信息而带来重大安全风险，这迫切需要能够在不损害整体效用的情况下删除特定知识的遗忘技术。然而，主流的基于微调的遗忘方法难以实现这种平衡，由于模型参数中知识的纠缠特性，这些方法往往存在计算效率低下和效用意外下降的问题。为了将遗忘与效用解耦，我们提出了UUE，这是一个具有理论基础的框架，它将遗忘重新表述为可控的模型编辑。首先，UUE将参数更新投影到保留知识表示的正交左零空间中，严格确保编辑过程对模型的通用能力造成最小干扰。其次，与有目标的映射不同，我们引入了一个凸的无目标目标函数，该函数推导出闭式解析解……

    arXiv:2611.17864v1 Announce Type: new  Abstract: Large Language Models (LLMs) pose significant security risks by retaining private or hazardous information, creating an urgent need for unlearning techniques that excise specific knowledge without compromising general utility. However, prevailing fine-tuning-based unlearning methods struggle to achieve this balance, often suffering from computational inefficiency and unintended utility degradation due to the entangled nature of knowledge within model parameters. To disentangle forgetting from utility, we propose UUE, a theoretically grounded framework that reformulates unlearning as controllable model editing. First, UUE projects parameter updates into the orthogonal left null space of retained knowledge representations, strictly ensuring that the editing process causes minimal interference to the model's general capabilities. Second, distinct from targeted mapping, we introduce a convex untargeted objective that derives a closed-form an
    
[^86]: 算子差异下基于后验采样的心电成像

    Electrocardiographic Imaging via Posterior Sampling under Operator Discrepancy

    [https://arxiv.org/abs/2611.17851](https://arxiv.org/abs/2611.17851)

    提出OpDis-PS方法，通过学习并对信号依赖的算子差异进行建模，将其融入修正后的前向算子，从而在算子不精确的现实条件下实现更准确的心电成像重建。

    

    心电成像（ECGI）旨在从体表电位重建心脏表面电位，这是一个病态逆问题。扩散后验采样器将学习到的先验与显式前向算子相结合，并假设该算子是准确的。然而在ECGI中，该算子仅是一个估计值，因为人体会随呼吸和姿势发生变化，而且皮下脂肪等个体身体成分通常难以获知。由此产生的算子差异是结构化的，并且依赖于心脏信号。我们提出了OpDis-PS，一种对这种差异进行建模的扩散后验采样器，而不是将其视为与信号无关的噪声。OpDis-PS学习差异如何依赖于信号，将其融入修正后的前向算子中，并在所有算子条件下使用固定的先验进行重建。我们研究了两种现实的差异来源：肺部电导率的变化以及皮下脂肪层的存在。

    arXiv:2611.17851v1 Announce Type: new  Abstract: Electrocardiographic imaging (ECGI) reconstructs heart-surface potentials from body-surface potentials, an ill-posed inverse problem. Diffusion posterior samplers combine a learned prior with an explicit forward operator that they assume to be accurate. In ECGI this operator is only an estimate, because the body changes with respiration and posture, and body composition such as subcutaneous fat is rarely known for an individual patient. The resulting operator discrepancy is structured and depends on the heart signal. We introduce OpDis-PS, a diffusion posterior sampler that models this discrepancy instead of treating it as signal-independent noise. OpDis-PS learns how the discrepancy depends on the signal, folds this into a corrected forward operator, and reconstructs with a fixed prior across all operator conditions. We study two realistic sources of discrepancy, changing lung conductivity and an added subcutaneous-fat layer. Across bot
    
[^87]: 多模态人物概念模型：面向变化观测条件下的人员识别

    Multimodal Person Concept Model for Person Identification under Varying Observation Conditions

    [https://arxiv.org/abs/2611.17802](https://arxiv.org/abs/2611.17802)

    本研究提出一种基于非参数贝叶斯生成模型的多模态人物概念模型，融合CLIP和CLAP提取的视觉、声学与语言特征，在少量无标签观测下实现变化观测条件下的鲁棒人员识别。

    

    家庭和办公室环境中的服务机器人必须识别用户，才能根据其偏好和之前的交互提供个性化服务。这极具挑战性，因为可用于学习的观测样本可能很少，而且观测条件会因外观变化、环境噪声和话语内容不同而改变。为了应对这一挑战，本研究提出了一种多模态人物概念模型，该模型能够从少量观测数据中学习人物概念，无需人物标签，并能在变化的条件下实现鲁棒的识别。该模型采用基于中餐馆过程（Chinese restaurant process）的非参数贝叶斯生成模型，整合了通过对比语言-图像预训练（CLIP）和对比语言-音频预训练（CLAP）提取并经主成分分析（PCA）处理的视觉、声学和语言特征。研究在八种条件下对六名参与者进行了评估。

    arXiv:2611.17802v1 Announce Type: cross  Abstract: Service robots in domestic and office environments must identify users to provide personalized services based on their preferences and previous interactions. This is challenging because only a few observations may be available for learning, while observation conditions change owing to appearance, environmental noise, and utterance content. To address this challenge, this study proposes a multimodal person concept model that learns person concepts from a small number of observations, without person labels, and enables robust identification under changing conditions. The model employs a nonparametric Bayesian generative model based on the Chinese restaurant process and integrates visual, acoustic, and linguistic features extracted using Contrastive Language-Image Pre-training (CLIP) and Contrastive Language-Audio Pretraining (CLAP), followed by principal component analysis (PCA). We evaluated six participants under eight conditions in wh
    
[^88]: 分布式增材制造中跨未见几何形状的联邦物理信息热场预测

    Federated Physics-Informed Thermal Field Prediction Across Unseen Geometries in Distributed Additive Manufacturing

    [https://arxiv.org/abs/2611.17794](https://arxiv.org/abs/2611.17794)

    该论文提出Fed-PIGANO联邦学习框架，使各制造站点能够在不共享专有零件几何数据的前提下联合训练物理信息几何感知神经算子模型，从而在分布式增材制造中实现对未见几何形状的热场预测。

    

    在增材制造（AM）中，每次零件几何形状的改变都需要进行新的热仿真，这使得对大量候选设计的评估成本高昂。几何感知模型可以通过单一训练好的模型在未见过的零件上预测热场，从而降低这一成本，但这类模型假设多样化的几何形状数据可以集中获取。当每个制造站点仅持有有限的、无法共享的专有零件设计时，这一假设便具有很强的局限性。我们提出了Fed-PIGANO，一个联邦学习框架，它在各个站点独立训练物理信息几何感知神经算子（PI-GANO）模型，并在不共享零件几何数据的情况下聚合各站点的几何编码器和预测网络。我们在金属增材制造的二维稳态热基准上对该框架进行了评估，几何形状来自五个多边形族，分布于三个站点，并在从族组成相同到差异显著的四种场景下进行了测试。

    arXiv:2611.17794v1 Announce Type: new  Abstract: In additive manufacturing (AM), each change in part geometry requires a new thermal simulation, which makes design evaluation over many candidates costly. Geometry-aware models reduce this cost by predicting thermal fields on unseen parts with a single learned model, but they assume that diverse geometries are available centrally. This assumption is restrictive when each manufacturing site holds only a limited set of proprietary part designs that it cannot share. We propose Fed-PIGANO, a federated framework that trains physics-informed geometry-aware neural operator (PI-GANO) models at individual sites and aggregates their geometry encoders and prediction networks without sharing part geometries. We evaluate the framework on a 2D steady-state thermal benchmark for metal AM, with geometries from five polygonal families distributed across three sites under four scenarios that range from identical to strongly different family compositions. 
    
[^89]: AutoRL：面向自主强化学习的多智能体系统

    AutoRL: A Multi-Agent System towards Autonomous Reinforcement Learning

    [https://arxiv.org/abs/2611.17790](https://arxiv.org/abs/2611.17790)

    AutoRL提出了一个由感知器等多个强化学习专属智能体角色组成的多智能体系统，通过持续监控演变的训练状态并将过程级监督与研究融入正在进行的RL训练，实现强化学习训练的自动化，为未来的递归自我改进（RSI）奠定基石。

    

    递归自我改进（RSI）是通向超级智能AI奇点最有希望的路径之一。实现这一愿景需要AI系统能够承担完整的训练流程以进行迭代改进。强化学习（RL）在这一循环中扮演着至关重要的角色，用于提升模型能力和泛化能力，然而由于训练动态不稳定以及需要持续监督，实现强化学习的自动化仍然困难重重。我们提出了AutoRL——一个用于自动化强化学习训练的多智能体系统，作为未来递归自我改进的基石。与通过端到端完整实验开展研究的方式不同，AutoRL监控不断演变的训练状态，并将过程级的监督和研究融入正在进行的强化学习训练中，遵循强化学习研究人员的实际工作流程。具体而言，AutoRL由三个强化学习专属的智能体角色组成。感知器追踪训练动态、样本采集、评估和效率，并触发……（摘要原文在此处截断）

    arXiv:2611.17790v1 Announce Type: new  Abstract: Recursive self-improvement (RSI) is one of the most promising paths toward an AI singularity for superintelligence. Achieving this vision requires AI systems to undertake the full training pipeline for iterative improvement. Reinforcement learning (RL) plays an essential role in this cycle to improve model capabilities and generalization, yet automating RL remains difficult due to unstable training dynamics and the need for sustained supervision. We introduce AutoRL, a multi-agent system for automating RL training, as a cornerstone for future RSI. Different from conducting research by end-to-end complete experiments, AutoRL monitors the evolving training state and integrates process-level supervision and research into an ongoing RL training, following the practical workflow of an RL researcher. Specifically, AutoRL consists of three RL-specific agentic roles. Perceptors track training dynamics, rollouts, evaluation, and efficiency, trigg
    
[^90]: LuthorSQL：作为较慢 LLM 智能体的“系统 1”的单遍文本到 SQL 模型

    LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent

    [https://arxiv.org/abs/2611.17780](https://arxiv.org/abs/2611.17780)

    提出非自回归 7B 模型 LuthorSQL，作为 LLM 智能体的快速“系统 1”门控，在单 GPU 上每题 0.84 秒即达到 BIRD 测试集 68.03% 的执行准确率，仅在不确定时才将查询升级给较慢的智能体，从而大幅降低成本并保护数据隐私。

    

    使用工具查询数据库的前沿 LLM 智能体是目前最准确的文本到 SQL 系统之一，但每次查询调用托管模型速度慢、成本高，且会将组织的敏感查询暴露给外部 API。例如，一个每天处理 10,000 次查询的 Claude Opus 5.5 智能体，按标价计算每年成本超过 70,000 美元。那么，组织已有的本地硬件能在多大程度上卸载这一负担而不损害智能体的准确性？我们提出了 LuthorSQL，一个非自回归的 7B 文本到 SQL 模型，它作为较慢 LLM 智能体（系统 2）的快速“系统 1”门控，仅在不确定时才将查询升级给智能体。LuthorSQL 通过单次前向传播，在受限文法上预测查询计划的槽位分数，编译并运行最多 32 个候选计划，并使用训练好的验证器选出答案。仅凭自身，LuthorSQL 在 BIRD 隐藏测试集上达到 68.03% 的执行准确率，在单个 GPU 上每个问题仅需 0.84 秒。

    arXiv:2611.17780v1 Announce Type: cross  Abstract: Frontier LLM agents that query databases with tools are among the most accurate text-to-SQL systems, but per-query calls to a hosted model are slow, costly, and expose sensitive organizational queries to external APIs. For example, a Claude Opus 5.5 agent serving 10,000 queries a day costs over $70,000 a year at list price. To what extent can local hardware already owned by an organization offload this burden without compromising the agent's accuracy? We present LuthorSQL, a non-autoregressive 7B text-to-SQL model that serves as a rapid System 1 gate for a slower LLM agent (System 2), escalating queries only when uncertain. LuthorSQL makes a single forward pass to predict slot scores over a bounded grammar for query plans, compiles and runs up to 32 candidate plans, and chooses an answer using a trained validator. On its own, LuthorSQL reaches 68.03% execution accuracy on BIRD's hidden test set at 0.84 s per question on a single GPU, m
    
[^91]: 超越预测增益：LASR——世界模型中动作接地潜表示的评估指标

    Beyond Prediction Gain: LASR as a Metric for Action-Grounded Latents in World Models

    [https://arxiv.org/abs/2611.17775](https://arxiv.org/abs/2611.17775)

    提出无量纲离线评估指标LASR（潜在动作支持率），用以判断潜动作模型学到的潜变量是否真正编码了动作信息，而非仅仅通过编码当前状态、回合进度等预测捷径来提升预测效果。

    

    支持规划、控制与可解释推理的世界模型，需要能够真正编码动作信息的潜表示。然而，用于训练此类模型的大规模视频数据普遍缺乏动作标注。潜动作模型通过从观测序列中推断出类似动作的变量，使得从无标注视频或轨迹中进行动作感知学习成为可能。然而，一个根本性的问题依然存在：一个能够提升未来预测效果的潜变量，可能只是编码了当前状态、回合进度、任务身份或其他预测捷径。现有的评估方法——重建损失、轨迹推演回放保真度、动作恢复或下游任务性能——均无法区分这种差异。我们提出了潜在动作支持率，这是一个无量纲指标，用于在留出数据上进行离线评估时，衡量学习到的潜变量是否能在当前状态之外，为动作相关的预测目标提供额外的预测支持。

    arXiv:2611.17775v1 Announce Type: cross  Abstract: World models that support planning, control, and interpretable reasoning require latent representations that genuinely encode action. Yet the large-scale video data from which such models are learned overwhelmingly lack action annotations. Latent-action models infer action-like variables from observation sequences, enabling action-aware learning from unlabeled videos or trajectories. A fundamental concern, however, remains: a latent that improves future prediction may simply encode the current state, episode progress, task identity, or other predictive shortcuts. Existing evaluation practices---reconstruction loss, rollout fidelity, action recovery, or downstream performance---cannot isolate this distinction. We introduce the Latent Action Support Ratio (LASR), a dimensionless metric that measures, under offline evaluation on held-out data, whether a learned latent provides predictive support for action-related targets beyond the curre
    
[^92]: 激活引导《台风》中的“南山号”

    Activation Steering the Nan-Shan in Typhoon

    [https://arxiv.org/abs/2611.17768](https://arxiv.org/abs/2611.17768)

    本文介绍了基于康拉德《台风》的视听装置作品，通过让大语言模型以自身输出为食并采用一种新颖的自适应激活引导方法，探索AI生成文本随时间推移变得日益陌生和失控的过程，以此反思人类文化当下的人工智能时刻。

    

    《台风》是一件基于约瑟夫·康拉德中篇小说的视听装置作品，小说中麦克威尔船长驾驶“南山号”轮船直接驶入并穿越一场灾难性的风暴。这部小说研究了当一种现象的规模超出可用于理解它的实践与概念框架时会发生什么，以及在如此压倒性的情境下被要求采取行动时有时随之而来的瘫痪状态。我们选择《台风》这部文本来反思人类文化当前的人工智能时刻。我们以康拉德的原始语言作为种子提示一个大语言模型，使其无休止地以自身输出为食，并探索文本随着时间推移何时以及如何变得越来越陌生和难以驾驭，包括偏离叙事、游移至模型自身的指令和思考，以及长时间不连贯的沉思。我们采用了来自激活引导领域的方法，包括我们引入的一种新颖的自适应引导方法。

    arXiv:2611.17768v1 Announce Type: cross  Abstract: Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, t
    
[^93]: 面向流模型学习各向异性预测目标

    Learning Anisotropic Prediction Targets for Flow Models

    [https://arxiv.org/abs/2611.17767](https://arxiv.org/abs/2611.17767)

    该论文提出各向异性流模型，将固定的预测目标替换为完全可学习的目标，通过对噪声分量进行各向异性缩放并利用闭式映射精确恢复速度，在不改变网络架构、损失函数和采样器的情况下优化训练与生成质量。

    

    基于流的生成模型通常在人工选定的预测目标（如速度、噪声或干净数据）上进行训练，这一选择对训练过程和样本质量有显著影响。我们提出了各向异性流模型，用一个完全可学习的预测目标取代固定的预测目标。AnisoFlow 学习以各向异性的方式缩放目标中的噪声分量，即在联合学习的基中为每个方向分配一个独立的非负权重。通过一个闭式映射即可从网络的预测中恢复速度，从而保持网络架构、流匹配损失和采样器完全不变。由于恢复是精确的，训练可以纯粹以使网络更容易预测为目标来优化预测目标。我们从理论上分析了由此产生的预测误差，并预测每个方向上的最优噪声权重与数据方差近似呈线性关系，这与在像素空间中通过实验学到的权重相吻合。

    arXiv:2611.17767v1 Announce Type: cross  Abstract: Flow-based generative models are trained on a hand-chosen prediction target, such as the velocity, the noise, or the clean data. This choice strongly affects training and sample quality. We introduce Anisotropic Flow Models (AnisoFlow), which replaces a fixed prediction target with a fully learnable one. AnisoFlow learns to scale the noise component of the target anisotropically, assigning a separate nonnegative weight to each direction in a jointly learned basis. A closed-form map recovers the velocity from the network's prediction, leaving the architecture, flow-matching loss, and sampler unchanged. Since recovery is exact, training optimizes the target purely to make it easier for the network to predict. We theoretically analyze the resulting prediction error and predict that the preferred noise weight in each direction scales near-linearly with data variance, in agreement with the weights learned empirically in pixel space. AnisoFl
    
[^94]: 基于密度感知点采样与E(3)-不变性的ADPKD三维MRI高效基因型分类

    Efficient Genotype Classification from 3D MRI in ADPKD via Density-Aware Point Sampling and E(3)-invariance

    [https://arxiv.org/abs/2611.17758](https://arxiv.org/abs/2611.17758)

    提出E3PN网络，利用密度感知点采样和E(3)-不变性，从三维腹部MRI中高效分类ADPKD基因型，解决了图像过大与标注数据不足两大难题。

    

    早期检测遗传性肾病，尤其是常染色体显性多囊肾病（ADPKD）至关重要，因为某些基因型（如截短型PKD1突变）会导致肾脏快速增大并早期进展至终末期肾病。基于MRI的基因型推断是基因检测的一种潜在低成本补充手段。然而，尽管已有许多深度学习方法被开发用于分类各类MRI图像，但将其应用于ADPKD仍然面临挑战：（1）腹部MRI图像通常过大，无法用为脑部MRI等较小图像设计的深度学习模型进行处理；（2）ADPKD基因突变的标注数据量不足以训练深度学习模型。为应对这些挑战，我们提出了E3PN，一种基于三维MRI的高效ADPKD基因型分类网络，其依赖于点采样和E(3)-不变性。我们利用三维MRI的点表示，……（摘要截断于此）

    arXiv:2611.17758v1 Announce Type: cross  Abstract: Early detection of genetic kidney diseases, especially autosomal dominant polycystic kidney disease (ADPKD), is crucial because certain genotypes such as truncating PKD1 mutations cause rapid kidney growth and early progression to end-stage kidney disease. MRI-based genotype inference is a potential low-cost complement to genetic testing. However, although many deep learning methods have been developed for classifying various MR images, applying them to ADPKD remains challenging: (1) abdominal MR images are generally too large to be handled by deep learning models designed for smaller images like brain MRI, and (2) the amount of labeled data for ADPKD genetic mutations is insufficient to train deep learning models. In order to address these challenges, we propose E3PN, an efficient 3D MRI-based ADPKD genotype classification network that relies on point sampling and E(3)-invariance. We utilize a point representation of 3D MRI, which con
    
[^95]: 复值薛定谔桥

    Complex Schr\"odinger Bridges

    [https://arxiv.org/abs/2611.17746](https://arxiv.org/abs/2611.17746)

    该论文提出了共形最优传输（COT）框架与复值薛定谔桥（CSBs）这一新型动态扩散桥方法，通过共形几何先验有效处理复向量空间中受缩放和旋转定律支配的物理系统的扩散任务。

    

    扩散桥已成为促进任意数据分布之间相互转换的通用框架。然而，这些方法受到庞大优化空间及其底层高阶结构的阻碍。本文探索了扩散桥的一种变体，用于解决复向量空间中由缩放和旋转定律（如哈密顿结构）所支配的现实问题。为此，我们引入了共形最优传输（COT），这是一种复值传输框架，专为在几何提升下具有几何性与耗散动力学的物理系统而设计。在此基础上，我们提出了一类新的动态扩散桥，称为复值薛定谔桥（CSBs），其衍生算法能够利用共形几何先验来处理基于扩散的任务。为了提升可扩展性，我们提出了图像到图像的CSB（I²CSB），这是一种易于处理的扩散桥技术，具有解析形式的……

    arXiv:2611.17746v1 Announce Type: new  Abstract: Diffusion bridges have emerged as versatile frameworks for facilitating transformations between arbitrary data distributions. However, these approaches are hindered by an immense optimization space and its underlying high-order structures. This paper explores a variant of diffusion bridges for solving real-world problems in complex vector spaces governed by scaling and rotation laws, such as Hamiltonian structures. To address this, we introduce conformal optimal transport (COT), a complex-valued transportation framework designed for physical systems with geometric and dissipative dynamics under a geometric lifting. Building on this, we propose a new class of dynamical diffusion bridges, called complex Schr\"odinger bridges (CSBs), whose derived algorithms can handle diffusion-based tasks using a conformal geometric prior. For scalability, we propose image-to-image CSB (I$^2$CSB), a tractable diffusion-bridge technique with an analytic sp
    
[^96]: Polyak-Łojasiewicz 论文：十年回顾

    The Polyak-{\L}ojasiewicz Paper: A 10-Year Retrospective

    [https://arxiv.org/abs/2611.17745](https://arxiv.org/abs/2611.17745)

    本文是对引入Polyak-Łojasiewicz (PL)条件术语的论文的十年回顾，作者回顾了论文的起源、给出了AI辅助精炼的主定理版本，并讨论了对该工作的批评及相关扩展。

    

    2016年，我们撰写了一篇论文，为一个古老的想法引入了“Polyak-Łojasiewicz (PL) 条件”这一术语，当时并不确定读者会有何反应。这篇论文最终产生了超出我们预期的影响，我们感到惊讶并十分荣幸能获得2026年PKDD ECML时间检验奖。凭借10年后的后见之明，本文对PL条件进行了一次非正式且坦率的回顾。我们回顾了这篇论文的起源，给出了借助AI辅助精炼的主定理版本，审视了针对该工作的批评，并讨论了各种相关的想法与扩展。

    arXiv:2611.17745v1 Announce Type: cross  Abstract: In 2016, we wrote a paper introducing the phrase "Polyak-Lojasiewicz (PL) condition" for an old idea, unsure of how readers would react. The paper ultimately had a larger impact than we expected, and we are surprised and honoured to receive the 2026 PKDD ECML Test of Time Award. With 10 years of hindsight, this note offers an informal and candid retrospective on the PL condition. We take a look back at the origins of the paper, give an AI-assisted refined version of the main theorem, review criticisms of the work, and discuss a variety of related ideas and extensions.
    
[^97]: 逻辑一致且准确的预测

    Logically Consistent Accurate Forecasts

    [https://arxiv.org/abs/2611.17735](https://arxiv.org/abs/2611.17735)

    该论文提出微调一个监督语言模型，利用支撑每个预测的推理过程来协调相关预测之间的逻辑不一致性，从而在ForecastBench上同时实现预测的逻辑一致性与准确性的提升。

    

    基于大语言模型（LLM）的智能体预测系统如今已能与专家级人类预测员相媲美，但这些系统仍然存在一个关键局限：无法产生逻辑一致的预测。例如，对相关事件的预测往往不满足概率公理，这意味着相关预测不可能同时正确，从而削弱了其对决策者的实用价值。强制一致性并非易事——朴素的方法（例如投影到一致预测的集合上）可能会降低准确性。我们认为，克服一致性与准确性之间这种明显权衡的关键在于支撑每个预测的推理过程。我们微调了一个监督语言模型，该模型利用这种推理来协调相关预测之间的不一致性，同时提高其准确性。在ForecastBench上，我们的方法改进了Brier分数（Brier score）……

    arXiv:2611.17735v1 Announce Type: cross  Abstract: Agentic forecasting systems based on large language models (LLMs) are now competitive with expert human forecasters, but these systems still suffer from a critical limitation: the inability to produce forecasts that are logically consistent. For example, forecasts for related events often fail to satisfy the axioms of probability, which in turn implies that related forecasts cannot simultaneously be correct and undermines their utility for decision makers. Enforcing consistency is not trivial -- naive approaches (e.g., projection onto the set of consistent forecasts) can degrade accuracy. We argue that the key to overcoming the apparent tradeoff between consistency and accuracy lies in the reasoning which supports each forecast. We finetune a supervisory language model that leverages this reasoning to jointly reconcile inconsistencies while improving the accuracy of related forecasts. On ForecastBench, our method improves the Brier sco
    
[^98]: RaFL：用于图像重建的随机滤波器损失的实证研究

    RaFL: An Empirical Study of Random Filter Losses for Image Reconstruction

    [https://arxiv.org/abs/2611.17707](https://arxiv.org/abs/2611.17707)

    提出了一种无需预训练特征和人工调参的随机滤波器损失，通过随机卷积、随机傅里叶特征和随机小波滤波器结合梯度范数匹配自动平衡各滤波器的贡献，有效提升图像重建中精细纹理的保留效果。

    

    像素损失在训练图像重建网络时，会使网络趋向目标分布的一个点统计量，在存在不确定性时往往会把精细纹理平均掉。感知损失通过预训练特征缓解了这一问题，但这些特征本质上是与特定领域绑定的。我们研究了一种随机滤波器损失，它通过具有随机权重的滤波器来比较预测结果与目标，从而既避免了使用预训练特征，也避免了针对具体任务的损失函数设计工作。RaFL 结合了随机卷积、随机傅里叶特征和随机小波滤波器，并通过梯度范数匹配自动平衡三者的贡献，无需手动调整损失权重或进行网格搜索。我们系统地研究了三种设计选择：滤波器族、权重重采样策略和权重初始化方式。基于验证集 PSNR 以及跨随机种子的稳定性，我们选择了同时使用三种滤波器族、并在每次优化（迭代）时重新抽取滤波器权重的配置……

    arXiv:2611.17707v1 Announce Type: cross  Abstract: Pixel losses train image reconstruction networks toward a point statistic of the target distribution, often averaging away fine textures under uncertainty. Perceptual losses alleviate this issue using pretrained features, but those features are inherently domain-specific. We study a random filter loss (RaFL) that compares prediction and target through filters with random weights, avoiding both pretrained features and task-specific loss engineering. RaFL combines random convolutions, random Fourier features, and random wavelet filters, automatically balancing their contributions through gradient-norm matching, eliminating manual loss-weight tuning and grid search. We systematically investigate three design choices: filter family, weight resampling schedule, and weight initialization. Based on validation PSNR and stability across random seeds, we select the configuration that uses all three filter families, redraws filters every optimiza
    
[^99]: 表征几何作为对抗鲁棒性的诊断工具

    Representational Geometry as a Diagnostic for Adversarial Robustness

    [https://arxiv.org/abs/2611.17684](https://arxiv.org/abs/2611.17684)

    该论文提出将表征几何分析（包括M-PHATE训练过程可视化、干净与扰动样本位移、最大均值差异及最近邻恢复等指标）作为诊断神经网络对抗鲁棒性的新工具，超越了传统仅依赖攻击下准确率的评估方式。

    

    对抗训练是一种常用技术，用于使神经网络对刻意设计的微小输入扰动更具鲁棒性，而鲁棒性通常通过攻击下的准确率来衡量。通过探索不同对抗训练目标下网络间的表征，我们可以分析多个网络是否以相似的方式组织其潜在表征。我们研究了在MNIST数据集上训练的多层感知机（MLP），涵盖三种对抗训练方案，并将得到的群体训练动态和几何结构与普通网络进行比较。我们使用M-PHATE技术在训练各轮次中对数据样本进行嵌入，考察在对抗目标下训练过程中数据几何结构的变化。在倒数第二层表征上，我们测量了干净样本点与扰动样本点之间的位移、通过最大均值差异（MMD）衡量的分布距离，以及最近邻恢复情况。我们的可视化和定量分析（摘要原文在此处截断）

    arXiv:2611.17684v1 Announce Type: new  Abstract: Adversarial training is a common technique for making neural networks more robust to small, intentionally crafted input perturbations, and robustness is commonly measured through accuracy under attack. By exploring inter-network representations across different adversarial training objectives, we can analyze whether multiple networks organize their latent representations similarly. We study Multilayer Perceptrons trained on MNIST under three adversarial training regimes and compare the resulting population training dynamics and geometry to those of a vanilla network. We use M-PHATE to embed data samples across training epochs and examine how the geometry of the data changes during training under adversarial objectives. On penultimate-layer representations we measure displacement between clean and perturbed points, distributional distance through maximum mean discrepancy, and nearest-neighbor recovery. Our visualizations and quantitative 
    
[^100]: 置换下基于模式的矩阵分析

    Pattern-Based Matrix Analysis Under Permutations

    [https://arxiv.org/abs/2611.17674](https://arxiv.org/abs/2611.17674)

    本文提出模式距离及基于线性指派子问题的交替算法，寻找最优行列置换使矩阵集中于指定几何模式（如带状、块对角等），并结合模式加低秩分解，为潜在排序、社区检测和带宽最小化等结构问题提供统一框架。

    

    arXiv:2611.17674v1 公告类型：新 摘要：低秩近似刻画了矩阵简单性的一个重要概念：如果一个矩阵能被低维线性模型很好地描述，则它是简单的。本文研究一个互补的概念。我们考虑那些只有在置换行和列之后才在结构上变得简单的矩阵，使其质量集中于指定的几何模式，如带状、块对角、箭头形或相关形式。我们引入了一种模式距离，用于衡量矩阵在最优重排后与给定模式的接近程度，为潜在排序、社区检测和带宽最小化等结构问题提供了统一框架。该问题一般是NP难的；我们提出了一种基于线性指派子问题的交替算法。我们进一步引入了模式加低秩分解，将置换结构拟合与低秩残差相结合，其置换步骤是在残差上进行模式拟合；i

    arXiv:2611.17674v1 Announce Type: new  Abstract: Low-rank approximation captures one important notion of matrix simplicity: a matrix is simple if it is well described by a low-dimensional linear model. This paper studies a complementary notion. We consider matrices that become structurally simple only after permuting rows and columns, so that their mass concentrates in a prescribed geometric pattern such as a band, block-diagonal, arrowhead, or related form. We introduce a pattern distance that measures how close a matrix is to a given pattern after optimal reordering, giving a common framework for structural problems such as latent ordering, community detection, and bandwidth minimization. The problem is NP-hard in general; we derive an alternating algorithm based on Linear Assignment subproblems. We further introduce a pattern-plus-low-rank decomposition that combines permutation-structured fitting with a low-rank residual, whose permutation step is pattern fitting on the residual; i
    
[^101]: CLIP嵌入中层级概念结构的可证明跨模态迁移

    Provable Cross-Modal Transfer of Hierarchical Concept Structure in CLIP Embeddings

    [https://arxiv.org/abs/2611.17671](https://arxiv.org/abs/2611.17671)

    本文首次从理论上刻画了CLIP嵌入中层级概念结构的跨模态迁移现象，提出层级角分离（HAS）的几何性质作为形式化定义，并证明了图像嵌入天然具有的层级结构可在对比学习中被迁移至文本嵌入。

    

    诸如CLIP这样的对比式视觉-语言模型是为了对齐图像和文本而训练的，而非为了表示分类体系。然而实证研究表明，CLIP嵌入展现出与语义概念层级相一致的层级结构。关于这种层级结构为何会从非层级性目标中产生，其理论刻画仍是一个悬而未决的问题。在本文中，我们将这一现象形式化为层级角分离（Hierarchical Angular Separation, HAS），这是一种几何性质，即每个非叶节点概念嵌入与其后代概念的相似度高于其与分支外概念的相似度。在实证方面，我们证明HAS在原始图像和随机初始化的CLIP图像嵌入中即近似成立，而文本嵌入只有在经过对比学习后才近似满足HAS。为了从理论上解释这种不对称性，我们分析了一个无约束特征模型，其中图像嵌入固定不变，仅对文本嵌入进行训练。在全局最小值处……

    arXiv:2611.17671v1 Announce Type: cross  Abstract: Contrastive vision-language models such as CLIP are trained to align images and text, not to represent taxonomies. Yet empirical studies show that CLIP embeddings exhibit hierarchical structure aligned with the hierarchy of semantic concepts. A theoretical characterization of why such hierarchy emerges from a non-hierarchical objective remains open. In this paper, we formalize this phenomenon as Hierarchical Angular Separation (HAS), a geometric property whereby each non-leaf concept embedding is more similar to its descendants than to off-branch concepts. Empirically, we show that HAS holds approximately in raw images and randomly initialized CLIP image embeddings, while in text embeddings it holds approximately only after contrastive learning. To theoretically explain this asymmetry, we analyze an unconstrained features model in which the image embeddings are fixed and only the text embeddings are trained. At the global minimizer, th
    
[^102]: 先学习再下注：学习可解释规则以实现可靠的智能体干预

    Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention

    [https://arxiv.org/abs/2611.17662](https://arxiv.org/abs/2611.17662)

    提出了LeBet统计学习框架，通过学习可解释的“或-与”规则监控器并结合固定下注统计量积累证据，在达到阈值时对自主智能体运行进行干预，并提供有限样本下的假阳性控制和检测功效保证。

    

    自主智能体现在可以在很少或没有人类监督的情况下运行数百步。在实践中，从业者使用运行时监控器（如轮次限制和关键词过滤器）来决定何时应停止、升级或协助某次运行。然而，这些启发式方法可能会停止本来会成功的运行，或让失败的运行继续下去。为了解决这一局限，我们提出了LeBet（Learn then Bet，先学习再下注），这是一个统计学习框架，其学习目标源自有效且强有力干预的充分条件。特别地，LeBet学习一个可解释的“或-与”（OR-of-AND）监控器，并通过固定的下注统计量积累证据，当证据达到预设阈值时进行干预。在有限的类内可交换性和留一稳定性条件下，我们利用相同的轨迹数据，推导出了有限样本保证，用于控制假阳性（有效性）以及以训练数据为条件的检测功效。

    arXiv:2611.17662v1 Announce Type: new  Abstract: Autonomous agents now run for hundreds of steps with little or no human supervision. In practice, practitioners use runtime monitors such as turn limits and keyword filters to decide when a run should be stopped, upgraded, or assisted. However, these heuristics may stop runs that would otherwise succeed or allow failing runs to continue. To address this limitation, we introduce LeBet (Learn then Bet), a statistical learning framework that derives its learning objective from sufficient conditions for valid and powerful intervention. In particular, LeBet learns an interpretable OR-of-AND monitor and accumulates evidence through a fixed betting statistic, intervening when the evidence reaches a prescribed threshold. Under finite within-class exchangeability and leave-one-out stability, we derive finite-sample guarantees for false-positive control (validity) and detection power conditional on the training data, using the same trajectories fo
    
[^103]: 模仿策略学习本体感知锚点：感知噪声及其闭环代价的预测定律

    Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost

    [https://arxiv.org/abs/2611.17658](https://arxiv.org/abs/2611.17658)

    该论文揭示了模仿策略会将机器人自身状态作为“本体感知锚点”来折扣含噪感知读数，其权重服从 α=V/(V+s²) 的预测定律，且锚点方差远小于线性模型暗示值，一次测量即可精确预测策略在不同任务和噪声水平下对感知噪声的敏感度。

    

    机器人操作的模仿策略通常在有噪声的物体位置读数上进行训练，它们学会对含噪读数进行折扣。我们展示了它们折扣的程度及其原因。由于演示者会朝物体方向移动，机器人自身的状态携带着物体位置的信息。训练好的策略将这一本体感知锚点与读数相结合，并以 α = V/(V + s²) 对读数加权，其中 s² 是读数噪声，V 是锚点的误差方差。V 属于学习器本身，远小于由演示数据建立的线性模型所暗示的值，这正是经典变量误差回归高估 α 的原因。V 只需测量一次，即可在两个 robomimic 任务、两个学习器和四个新噪声水平下预测十个保留样本的敏感度，平均绝对误差为 0.025，而经典 EIV 方法为 0.37；此外，一个无参数的偏差测试排除了更简单的解释，即策略只是单纯高估……

    arXiv:2611.17658v1 Announce Type: cross  Abstract: Imitation policies for robot manipulation are often trained on noisy object positions, and they learn to discount the noisy reading. We show how much they discount it, and why. Because demonstrators steer toward the object, the robot's own state carries information about where the object is. A trained policy combines this proprioceptive anchor with the reading and weights the reading by $\alpha = V/(V + s^2)$, where $s^2$ is the reading noise and $V$ the anchor's error variance. $V$ belongs to the learner and is far smaller than a linear model of the demonstrations implies, which is why classical errors-in-variables (EIV) regression overestimates $\alpha$. Measured once, $V$ predicted ten held-out sensitivities across two robomimic tasks, two learners and four new noise levels, with mean absolute error 0.025 against 0.37 for classical EIV, and a parameter-free bias test rules out the simpler explanation that the policies just overestim
    
[^104]: 面向小分子生成的分类器引导潜在扩散模型

    Classifier-Guided Latent Diffusion for Small-Molecule Generation

    [https://arxiv.org/abs/2611.17650](https://arxiv.org/abs/2611.17650)

    该论文提出在ChemBERTa潜在空间中构建去噪扩散模型，并利用可微分的性质预测器引导扩散采样过程，实现性质导向的小分子生成，生成的分子通过有效性、唯一性、新颖性和性质分数进行评估。

    

    扩散模型因其能够学习复杂的化学分布并支持条件生成或引导生成，已成为分子设计领域一类有前景的生成模型。在这项工作中，我们研究了一种用于性质引导小分子生成的潜在扩散流程。所提出的框架使用冻结的ChemBERTa编码器来表示规范化的SMILES字符串，训练基于Transformer的解码器从分子潜在嵌入中重构SELFIES字符串，并在ChemBERTa潜在空间中学习一个无条件的去噪扩散模型。为实现性质导向的生成，使用可微分的潜在性质预测器来引导扩散采样过程。生成的SELFIES字符串被转换为SMILES，使用RDKit进行验证，并通过分子有效性、唯一性、新颖性和预测性质分数进行评估。我们研究了针对两个药物相关二元终点的引导（摘要在此处被截断）。

    arXiv:2611.17650v1 Announce Type: new  Abstract: Diffusion models have emerged as a promising class of generative models for molecular design because of their ability to learn complex chemical distributions and support conditional or guided generation. In this work, we investigate a latent diffusion pipeline for property-guided small-molecule generation. The proposed framework represents canonicalized SMILES strings using a frozen ChemBERTa encoder, trains a Transformer-based decoder to reconstruct SELFIES strings from molecular latent embeddings, and learns an unconditional denoising diffusion model in the ChemBERTa latent space. To enable property-directed generation, a differentiable latent property predictor is used to guide the diffusion sampling process. Generated SELFIES strings are converted to SMILES, validated using RDKit, and evaluated using molecular validity, uniqueness, novelty, and predicted property scores. We study guidance toward two drug-relevant binary endpoints: bl
    
[^105]: 二值化：无权重神经网络的攻击面

    Binarization as an Attack Surface in Weightless Neural Networks

    [https://arxiv.org/abs/2611.17646](https://arxiv.org/abs/2611.17646)

    该研究揭示了无权重神经网络（WiSARD及其布隆过滤器变体）虽然看似对对抗样本天然鲁棒，但其二值化阈值本身可被利用为攻击面，并通过阈值跨越界限和响应边际证书对这一攻击机制进行了形式化证明。

    

    无权重神经网络（WNNs）是一类轻量级分类器，它们对输入进行二值化处理，并通过在查找存储器中存储二进制模式来进行学习，这使其在入侵检测等资源受限任务中颇具吸引力。由于不跨越任何二值化阈值的微小扰动不会改变二进制输入，无权重神经网络似乎对对抗样本具有天然的鲁棒性；然而，一旦扰动跨越了阈值，仍可能改变模型的预测结果。我们在WiSARD（经典的基于RAM的无权重神经网络）以及两个用布隆过滤器替代其存储器的变体——Bloom WiSARD和BTHOWeN——中研究了这一矛盾。连续扰动只有通过跨越编码阈值、改变至少一个二进制元组并重定向存储器查询，才能影响这些分类器；我们通过阈值跨越界限和标准布尔WiSARD的响应边际证书对这一机制进行了形式化。受控MNIST实验…

    arXiv:2611.17646v1 Announce Type: cross  Abstract: Weightless Neural Networks (WNNs) are lightweight classifiers that binarize their inputs and learn by storing binary patterns in lookup memories, which makes them attractive for resource-constrained tasks such as intrusion detection. Because a small perturbation that crosses no binarization threshold leaves the binary input unchanged, WNNs may appear naturally robust to adversarial examples; a perturbation that does cross a threshold, however, can still change the prediction. We study this tension in WiSARD, the canonical RAM-based WNN, and in two variants that replace its memories with Bloom filters, Bloom WiSARD and BTHOWeN. A continuous perturbation reaches these classifiers only by crossing an encoding threshold, changing at least one binary tuple, and redirecting a memory query; we formalize this mechanism with a threshold-crossing bound and a response-margin certificate for standard Boolean WiSARD. Controlled MNIST experiments co
    
[^106]: 一种用于热驱动脆性断裂相场建模的无网格深度能量方法

    A mesh-free deep energy method for phase-field modeling of thermally driven brittle fracture

    [https://arxiv.org/abs/2611.17630](https://arxiv.org/abs/2611.17630)

    提出了一种基于多分辨率神经表示的无网格深度能量方法，无需预先放置缺陷即可在瞬态热载荷下预测脆性断裂的自发裂纹成核、裂纹间距以及选择性的裂纹扩展与止裂。

    

    热驱动断裂的相场模拟需要解析裂纹路径预先未知的局部化裂纹带，以及在瞬态过程中厚度不断变化的热边界层。迄今为止，基于神经能量的热断裂计算都是从预先放置的缺陷出发的。本文提出了一种基于多分辨率神经表示的无网格深度能量方法，能够在瞬态热载荷下预测自发裂纹成核、裂纹间距以及选择性的裂纹扩展与止裂。两个网络（各自具有独立的多分辨率编码）分别表示位移场与相场以及瞬态温度场，机械能和后向欧拉热方程的增量泛函在每次迭代时重新生成的积分点上进行估计。通过将不可逆性约束限制在裂纹集合上、跨载荷步传递优化器状态并衰减学习率……（摘要内容不完整，此处截断）

    arXiv:2611.17630v1 Announce Type: new  Abstract: Phase-field simulation of thermally driven fracture requires resolving localized crack bands whose paths are unknown in advance and thermal boundary layers whose thickness evolves during the transient. Neural energy-based computations of thermal fracture have so far started from defects placed in advance. This paper proposes a mesh-free deep energy method based on multiresolution neural representations that predicts spontaneous crack nucleation, crack spacing, and selective crack propagation and arrest under transient thermal loading. Two networks, each with its own multiresolution encoding, represent the displacement and phase fields and the transient temperature, and the mechanical energy and the incremental functional of the backward Euler heat equation are estimated on integration points redrawn at every iteration. Restricting irreversibility to the crack set, carrying the optimizer state across load steps and decaying the learning r
    
[^107]: 基于可微分轨迹优化的联合世界-动作模型

    Joint World Action Models through Differentiable Trajectory Optimization

    [https://arxiv.org/abs/2611.17626](https://arxiv.org/abs/2611.17626)

    WATO 通过可微分轨迹优化联合优化生成的动作与未来状态，使世界-动作模型的轨迹既具备任务有效性又符合动力学一致性，且部署时无需迭代轨迹优化。

    

    联合世界-动作模型（WAM）同时生成未来动作与状态，对“世界-动作”轨迹整体建模，而非仅建模动作本身。然而，轨迹模仿并不能保证这些轨迹具有任务有效性或与动力学一致。我们提出了世界-动作轨迹优化（WATO），它为控制目的对生成的动作和未来状态进行联合优化。WATO 将二者作为参考置于一个动力学约束的轨迹优化问题中，并通过优化后的轨迹对任务损失进行微分来训练生成器。这使得生成的轨迹既具备任务感知能力又与动力学保持一致，同时轨迹优化仅在训练阶段使用。在部署时，WATO 直接生成候选轨迹并用世界模型对其进行排序，无需迭代式轨迹优化。在 7 个视觉控制环境中，WATO 将模仿式联合 WAM 的平均成功率从 79.9%（进一步提升）。

    arXiv:2611.17626v1 Announce Type: new  Abstract: Joint world action models (WAMs) generate future actions and states together, modeling world-action trajectories rather than actions alone. However, trajectory imitation does not ensure that these trajectories are task-effective or consistent with dynamics. We present \textbf{world action trajectory optimization (WATO)}, which optimizes generated actions and future states jointly for control. WATO uses both as references in a dynamics-constrained trajectory optimization problem and differentiates the task loss through the optimized trajectory to train the generator. This makes generated trajectories task-aware and dynamics-consistent while using trajectory optimization only during training. At deployment, WATO directly generates candidate trajectories and ranks them with a world model, without iterative trajectory optimization. Across $7$ visual control environments, WATO improves average success from $79.9\%$ for the imitation joint WAM
    
[^108]: 有限速度Ornstein-Uhlenbeck-Kac过程及其在生成建模中的应用

    Finite speed Ornstein-Uhlenbeck-Kac processes and their applications in generative modeling

    [https://arxiv.org/abs/2611.17625](https://arxiv.org/abs/2611.17625)

    本文提出有限速度的Ornstein-Uhlenbeck-Kac（OU-K）过程，利用勒让德多项式显式求解Euler-Poisson-Darboux方程，并借助多项式分布公式刻画其条件速度场，从而成功将该过程应用于生成流模型。

    

    我们研究了Ornstein-Uhlenbeck-Kac（OU-K）过程，其中驱动过程由一个有限速度的Kac过程而非布朗运动来定义。OU-K过程可以通过一个重标度的、时变非齐次的Kac过程来分析，该过程由来自Kac部分的切换率和来自Ornstein-Uhlenbeck（OU）部分的收敛率来刻画。该非齐次Kac过程的分布描述了一个波动方程——Euler-Poisson-Darboux（EPD）方程的解。当Kac切换率是OU收敛率的整数倍时，我们可以通过两个勒让德多项式（或其导数）之和显式地推导出该EPD解，该解被限制在两个代表波前的Dirac点之间。利用分段确定性马尔可夫过程（PDMP）的框架，我们通过多项式分布公式描述了OU-K过程的条件速度场。这使得我们可以将OU-K过程应用于生成流模型中。

    arXiv:2611.17625v1 Announce Type: cross  Abstract: We study the Ornstein-Uhlenbeck-Kac (OU-K) process, where instead of a Brownian motion, a finite-speed Kac process defines the driving process. The OU-K process can be analyzed through a rescaled, time-inhomogeneous Kac process, characterized by a switching rate from the Kac part, and a convergence rate from the Ornstein-Uhlenbeck (OU) part. The law of this inhomogeneous Kac process describes the solution of a wave equation, the Euler-Poisson-Darboux (EPD) equation. For Kac switching rates being multiples of the OU convergence rate, we can explicitly derive this EPD solution via the sum of (the derivative of) two Legendre polynomials, restricted between two Dirac points representing the wave fronts. Using the framework of piecewise-deterministic Markov processes (PDMP), we describe the conditional velocity fields of the OU-K process via our polynomial-law-formula. This allows us to apply the OU-K process in a generative flow model. Sin
    
[^109]: AscendGYM：面向昇腾NPU真实工作负载的成本感知算子优化基准

    AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs

    [https://arxiv.org/abs/2611.17621](https://arxiv.org/abs/2611.17621)

    提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。

    

    算子优化对于在昇腾NPU上高效进行大语言模型（LLM）训练和推理至关重要。现有的昇腾算子基准主要关注算子生成，较少关注智能体在来自真实工作负载的输入配置下优化现有实现的能力。为满足这一需求，我们提出了AscendGYM，这是一个基准测试，其初始套件包含48个算子，分为三个难度级别。每个算子都配有测试用例，其数据类型和张量形状均收集自真实的训练和推理任务。因此，在AscendGYM上取得的性能提升可以直接映射到相应的工作负载。在可用性方面，AscendGYM支持两种评估模式：模型评估和模型-框架评估，既可以衡量经过训练的模型，也可以衡量自定义智能体的算子优化能力。它提供了一个完整的执行环境……

    arXiv:2611.17621v1 Announce Type: new  Abstract: Kernel optimization is essential for efficient large language model (LLM) training and inference on Ascend NPUs. Existing Ascend kernel benchmarks primarily focus on kernel generation, with less emphasis on the ability of agents to optimize existing implementations under input configurations drawn from real workloads. To address this requirement, we introduce AscendGYM, a benchmark whose initial suite comprises 48 kernels across three difficulty levels. Each kernel is paired with test cases whose data types and tensor shapes are collected from real-world training and inference tasks. Consequently, performance improvements on AscendGYM map directly to the corresponding workloads. In terms of usability, two evaluation modes are supported in AscendGYM: model evaluation and model-harness evaluation. The kernel optimization capabilities of both trained models and custom agents are allowed to be measured. It provides a complete execution envir
    
[^110]: 基于大语言模型引导的聚类生命周期管理：决定行为聚类何时以及如何演化

    LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters

    [https://arxiv.org/abs/2611.17616](https://arxiv.org/abs/2611.17616)

    该论文提出让大语言模型（LLM）基于无阈值的聚类整体状态视图，就地决策行为聚类的生命周期操作（分裂、合并、退役、创建），从而替代传统上依赖硬编码阈值或重聚类规则的维护范式。

    

    随着用户或物品交互行为的不断演变，基于这些交互学习到的行为聚类会发生漂移：一些聚类会分裂成子行为，一些会融合，一些会消失，同时新的行为模式会涌现。目前维护演化聚类主要有两种范式：阈值触发机制在硬编码的统计量超过预设临界值时触发预定义动作；重聚类与跟踪方法（MONIC风格）在每个时间窗口重新聚类，并根据新聚类与前一窗口聚类的对齐情况推断演化转移关系。这两种方法都将决策硬编码为阈值或重聚类规则。我们研究了一种替代方案：由大语言模型（LLM）直接就地做出生命周期决策。给定一个无阈值、面向全体数据的当前聚类集合视图（统计健康度、标签、成员文本、残差质量），LLM针对每个聚类决定是否以及如何采取行动（分裂、合并、退役、创建或无动作），而无需重新聚类。由于有用的信号通道取决于信息量……

    arXiv:2611.17616v1 Announce Type: new  Abstract: Behavioral clusters learned over user or item interactions drift as behavior evolves: some fragment into sub-behaviors, some converge, some disappear, and new patterns emerge. Two paradigms dominate the maintenance of evolving clusterings. Threshold triggers fire a pre-specified action when a hardcoded statistic crosses a cutoff; refit-and-track (MONIC-style) re-clusters every window and infers transitions from how the new clusters align with the previous window's. Both hardwire the decision into a threshold or a re-clustering rule. We study an alternative in which a large language model (LLM) makes the lifecycle decision in place. Given a threshold-free, whole-population view of the current cluster set (statistical health, labels, member texts, residual mass), the LLM decides per cluster whether and how to act (split, merge, retire, create, or no-action) without re-clustering. Because the useful signal channel depends on how informative
    
[^111]: 更多智能体、更多样本，还是更多思考？解耦多智能体大语言模型中的协调、集成与测试时计算

    More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs

    [https://arxiv.org/abs/2611.17610](https://arxiv.org/abs/2611.17610)

    本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。

    

    多智能体语言模型的评估通常将具有通信机制的架构与单次贪婪推理进行比较，这种做法将协调效应与样本数量和推理计算量混为一谈。本文综合了九项主要实证评估，涵盖516个独特的多智能体配置，沿一个包含六个维度的资源向量R进行组织，并构建了包含四个类别的基线阶梯。与单次推理基线（A类）相比，多智能体的准确率差异总体为负，在剔除贡献最大的一项研究后接近于零。当与按样本数量（B类）或资源预算（C类）匹配的基线进行比较时，多智能体系统的平均准确率差异为-0.042，且基于对数几率尺度的多层随机效应建模在各项敏感性检验中均保持负值。在74%的可比对比中，多智能体系统的准确率等于或低于基线，且各研究的均值几乎均为非正值。

    arXiv:2611.17610v1 Announce Type: new  Abstract: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), multi-agent accuracy differences are negative overall and near zero when omitting the largest contributing study. When evaluated against baselines matched on sample count (Category B) or resource budget (Category C), multi-agent systems average -0.042 in accuracy, and multilevel random effects modeling on the log odds scale remains negative across sensitivity checks. Multi-agent accuracy equals or falls below the baseline in 74% of these comparable contrasts, and study means are nonpositive in alm
    
[^112]: 面向计算认知模型的神经仿真推断：一篇教程

    Neural Simulation-Based Inference for Computational Cognitive Models: A Tutorial

    [https://arxiv.org/abs/2611.17608](https://arxiv.org/abs/2611.17608)

    这篇教程介绍了神经仿真推断（neural SBI）如何通过在模型仿真数据上训练深度神经网络，克服计算认知模型中似然函数不可计算的问题，从而突破传统拟合方法的局限、拓展可研究的认知理论范围。

    

    计算认知建模是一种日益重要的方法论，用于阐明人类与动物认知背后的机制。通过将模型拟合到实证数据，研究者可以进行参数估计、模型比较以及基于模型的神经分析，以评估相互竞争的理论、量化个体差异并研究神经表征。通常，拟合这些模型依赖于依赖似然函数的方法（例如最大似然估计或分层贝叶斯方法）。然而，对于许多合理的认知模型而言，计算似然函数在计算上是难以实现的，在这种情况下这些方法便不再适用。这极大地限制了研究者可以考察的理论范围。基于神经网络的仿真推断（neural SBI）已成为克服这一障碍的强大工具包，其通过在由模型仿真生成的数据集上训练深度神经网络……

    arXiv:2611.17608v1 Announce Type: new  Abstract: Computational cognitive modeling is an increasingly essential methodology to elucidate the mechanisms underlying human and animal cognition. By fitting models to empirical data, researchers can perform parameter estimation, model comparison, and model-based neural analyses to assess competing theories, quantify individual differences, and investigate neural representations. Typically, fitting these models relies on likelihood-dependent methods (e.g., Maximum likelihood estimation or hierarchical Bayesian methods). However, for many reasonable cognitive models, computing the likelihood is computationally intractable, in which case these methods become inadequate. This substantially limits the range of theories that researchers can investigate. Simulation-based inference with neural networks (neural SBI) has emerged as a powerful toolkit to overcome this hurdle. By training deep neural networks on datasets generated through model simulatio
    
[^113]: 按文件大小与每解决任务能耗选择量化代码大语言模型

    Choosing Quantized Code LLMs by File Size and Energy per Solved Task

    [https://arxiv.org/abs/2611.17605](https://arxiv.org/abs/2611.17605)

    本研究在 MacBook Air M2 上以“每解决任务的能耗”评测 Qwen2.5-Coder 的 24 个量化版本，给出按文件大小索引、附证据标签的模型选择指南，并揭示 3 GB 档位的 7B-q2_K 相比 2 GB 档位的 3B-q4_K_M 无显著性能提升却耗能翻倍。

    

    在笔记本电脑上运行代码助手的开发者只需下载一个模型文件，本地推理便把每次请求的能耗都转移到这台机器上。一条常见的经验法则是：低精度的大模型往往胜过高精度的小模型。我们在一台无风扇的 MacBook Air M2 上运行了四种规模的 24 个 Qwen2.5-Coder GGUF 文件，并在 542 个 EvalPlus 任务上以“每解决一个任务所消耗的能量”为代价指标对每个文件进行评估。最终产出一份以 Ollama 所列文件大小为索引的模型选择指南，且每个推荐都附有证据标签。在 1 GB 和 2 GB 档位，指南推荐 1.5B 和 3B 模型的 q4_K_M 文件，二者在任何核算方式下都位于正确性与能耗的前沿。在所列的 3 GB 档位，“模型越大越好”的规则会选择 7B-q2_K——这是唯一实测能装下的 7B 文件——但在此处它并不划算：与 2 GB 档位的选择 3B-q4_K_M 相比，该文件在解题数上没有显著提升，且 90% 置信区间将 3B-q4_K_M 的任何劣势限制在 2.4 分以内，而 7B-q2_K 每解决一个任务所需的能耗却是 2.03 倍。

    arXiv:2611.17605v1 Announce Type: cross  Abstract: A developer running a code assistant on a laptop downloads one model file, and local inference moves the energy of every request onto that machine. A common rule says a larger model at lower precision tends to beat a smaller one at higher precision. We ran 24 GGUF files of Qwen2.5-Coder at four sizes on a fanless MacBook Air M2 and priced each in energy per solved task on 542 EvalPlus tasks. The result is a selection guide indexed by Ollama's listed file size, with an evidence label on each pick. At 1 and 2 GB it picks the q4_K_M files of the 1.5B and 3B models, both on the correctness and energy frontier under every accounting. At 3 GB as listed, the larger-model rule picks 7B-q2_K, the only measured 7B file that fits, and there it does not pay. That file brings no significant gain in solves over the 2 GB pick, 3B-q4_K_M, and the 90% CI limits any 3B-q4_K_M deficit to 2.4 points. Yet 7B-q2_K needs 2.03 times the energy per solved task
    
[^114]: LightTAR：用于多模态表格学习的轻量快速目标感知表示

    LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning

    [https://arxiv.org/abs/2611.17600](https://arxiv.org/abs/2611.17600)

    LightTAR是一种简单低成本的降维增强方法，它利用正则化PLS从编码器最后一层和中间层提取目标感知特征来补充标准PCA表示，从而在21种表格模型上普遍提升多模态表格学习的预测性能。

    

    多模态表格学习涉及对同时包含结构化数值与类别特征以及非结构化文本和图像的表格进行分类和回归。这些非结构化字段通常由预训练编码器的嵌入来表示，并使用主成分分析（PCA）进行降维。这种降维优先保留输入方差，但可能忽略与预测目标相关的信息。我们提出了LightTAR，这是一种简单且低开销的方法，能够使降维过程与预测任务相适应。LightTAR利用正则化偏最小二乘（PLS）从编码器的最后一层和某一中间层联合提取少量目标感知特征，以此增广标准的PCA表示。我们使用21种表格模型对LightTAR进行了评估，包括树集成模型、神经网络以及表格基础模型。LightTAR提升了所有模型的平均预测性能，并且在跨模型的平均意义上改善了预测……

    arXiv:2611.17600v1 Announce Type: cross  Abstract: Multimodal tabular learning involves classification and regression on tables combining structured numerical and categorical features with unstructured text and images. These unstructured fields are commonly represented by pretrained encoder embeddings reduced with principal component analysis (PCA). This reduction prioritizes input variance but may overlook information relevant to predicting the target. We introduce LightTAR, a simple, inexpensive method that adapts dimensionality reduction to the prediction task. LightTAR augments the standard PCA representation with a few target-aware features extracted jointly from the final and an intermediate encoder layer using regularized partial least squares (PLS). We evaluate LightTAR with 21 tabular models, including tree ensembles, neural networks, and tabular foundation models. LightTAR improves average predictive performance for all models and, averaged across models, improves prediction 
    
[^115]: 蒙特卡洛隐私核算的可处理重要性采样方法

    Tractable Importance Sampling for Monte Carlo Privacy Accounting

    [https://arxiv.org/abs/2611.17596](https://arxiv.org/abs/2611.17596)

    该论文提出了一种基于平移重要性采样的高效蒙特卡洛隐私核算方法，利用拉伸因子 α 参数化的提议分布及闭式界 B(α) < 1，大幅降低了估计严格隐私保证下极小 δ 值时的方差与样本复杂度。

    

    对于采用采样隐私放大机制的差分隐私（DP）矩阵分解，评估其紧致的差分隐私参数需要分析高维且相关的（correlated）高斯混合分布。由于对这些混合分布进行积分以求得精确的隐私失败概率（δ）在解析上是不可处理的，这类机制通常通过蒙特卡洛核算来进行分析。然而，在估计与严格隐私保证相关联的极小 δ 值时，标准采样会遭遇难以承受的高方差，进而需要极大的样本复杂度，使得蒙特卡洛核算效率低下。我们证明，通过重要性采样可以大幅提升蒙特卡洛核算的效率。我们提出了一种由拉伸因子 α > 1 参数化的平移重要性采样提议分布，并在重要性加权估计量的支撑集上证明了一个恒成立的闭式界 B(α) < 1……

    arXiv:2611.17596v1 Announce Type: cross  Abstract: Evaluating tight Differential Privacy (DP) parameters for DP matrix factorization with privacy amplification by sampling requires analyzing high-dimensional, correlated Gaussian mixtures. Because integrating these mixture distributions to find the exact privacy failure probability ($\delta$) is analytically intractable, these mechanisms are typically analyzed using Monte Carlo accounting. However, standard sampling suffers from prohibitive variance when estimating small $\delta$ associated with strict privacy guarantees, which in turn requires a large sample complexity, making Monte Carlo accounting inefficient. We demonstrate that Monte Carlo accounting can be made much more efficient via importance sampling. We give a shifted importance sampling proposal distribution parameterized by a stretch factor $\alpha > 1$. We prove a closed-form bound $B(\alpha) < 1$ on the support of the importance-weighted estimator that holds identically f
    
[^116]: 基于门限密码学与机器学习的公平高效分布式电动汽车充电协调

    Fair and Efficient Distributed EV Charging Coordination Using Threshold Cryptography and Machine Learning

    [https://arxiv.org/abs/2611.17577](https://arxiv.org/abs/2611.17577)

    提出一种结合门限密码学与机器学习的框架，能在保持线性通信成本且无误报的前提下检测分布式电动汽车充电协调中的拜占庭攻击，并利用车队感知的机器学习价格预测帮助受攻击的电动汽车实现本地恢复。

    

    基于交替方向乘子法（ADMM）的电动汽车（EV）充电分布式协调能够保持各电动汽车约束的隐私性，但其假设所有参与者都遵循该算法。这一假设在实践中并不成立——有偏见的运营商或中间人（MITM）攻击者可以向不同的电动汽车发送不一致的信号，而被攻陷的电动汽车也可能偏离其规定的更新步骤。我们首先证明了此类任意恶意（即拜占庭）参与者的攻击并不会阻止算法收敛，因此难以被检测，并以闭式形式推导出了最具破坏性的攻击方式。随后，我们提出了一个将门限签名广播与上传重计算相结合的框架，该框架能够检测所有偏差且不产生误报，同时将通信成本从二次方降低至线性。为保持优化的持续运行，每辆受影响的电动汽车可通过经认证的、具备车队感知能力的机器学习价格预测进行本地恢复，并且……（摘要在此处截断）

    arXiv:2611.17577v1 Announce Type: new  Abstract: Distributed coordination of electric vehicle (EV) charging with the alternating direction method of multipliers (ADMM) keeps EV constraints private but assumes that every participant follows the algorithm. This assumption fails in practice, where a biased operator or a man-in-the-middle (MITM) attacker can send inconsistent signals to different EVs, and compromised EVs can deviate from their prescribed updates. We first prove that attacks by such arbitrarily malicious, i.e., Byzantine, participants do not prevent convergence and therefore remain undetected, and derive the most damaging attack in closed form. We then propose a framework that combines a threshold-signed broadcast with upload recomputation, which detects every deviation without false alarms at linear instead of quadratic communication cost. To keep the optimization running, each affected EV recovers locally with a certified, fleet-aware machine learning price forecast, and 
    
[^117]: 超越单一符号：用于分支保留型二值阈值扩展的BiTE

    Beyond a Single Sign: BiTE for Branch-Preserving Binary Threshold Expansion

    [https://arxiv.org/abs/2611.17576](https://arxiv.org/abs/2611.17576)

    BiTE通过将单一符号激活替换为K个有序阈值比较并保留分支响应，使二值神经网络能够区分多达K+1个有序激活状态，从而提升表示能力与准确率。

    

    二值神经网络（BNNs）通过减少模型存储量并用高效的按位运算替代许多算术运算来降低成本，但单一符号激活限制了其表示能力。每个预激活方向仅与一个阈值进行比较并被归约为两个值之一，从而丢弃了每个符号区域内部的变化。我们提出BiTE（分支保留型二值阈值扩展），它将每个特征方向与K个有序阈值进行比较，并将产生的二值响应保留为后续学习算子的独立输入。BiTE能够区分多达K+1个有序激活状态，同时允许后续算子为每个阈值响应学习单独的变换。它还实现了在独立构建的特征方向数量与从每个方向提取的有序响应数量之间的可控权衡。在BinaryViT上，BiTE提升了平均最终（准确率）……（原文摘要在此处截断）

    arXiv:2611.17576v1 Announce Type: new  Abstract: Binary neural networks (BNNs) reduce model storage and replace many arithmetic operations with efficient bitwise computations, but the single-sign activation limits their representational capacity. Each preactivation direction is compared with one threshold and reduced to one of two values, discarding variation within each sign region. We propose \textbf{BiTE} (Branch-Preserving Binary Threshold Expansion), which evaluates each feature direction against $K$ ordered thresholds and retains the resulting binary responses as separate inputs to the following learned operator. BiTE can distinguish up to $K+1$ ordered activation states while allowing the following operator to learn a separate transformation for each threshold response. It also enables a controllable trade-off between the number of independently constructed feature directions and the number of ordered responses extracted from each direction. On BinaryViT, BiTE improves mean fina
    
[^118]: 基于SYCL和oneMath在Python中实现性能可移植的精确高斯过程回归

    Performance-Portable Exact Gaussian Process Regression in Python via SYCL and oneMath

    [https://arxiv.org/abs/2611.17565](https://arxiv.org/abs/2611.17565)

    通过为GPRat高斯过程回归库新增基于oneMath的混合HPX/SYCL后端，单一代码库无需厂商专用代码即可在NVIDIA GPU上媲美原生CUDA性能，并在AMD GPU上最高以4.62倍的速度超越GPflow和GPyTorch等成熟框架，实现了真正的性能可移植性。

    

    现代多厂商GPU环境迫使开发者在可移植代码与原生性能之间做出取舍。GPRat是一个用于精确高斯过程回归的开源Python库，构建于HPX异步多任务运行时、分块算法以及CPU/CUDA BLAS后端之上。我们贡献了一个采用混合HPX/SYCL设计的UXL Foundation oneMath后端，将GPRat扩展到了AMD和Intel GPU。在最大问题规模下，新的SYCL后端在NVIDIA A30上的性能与原生CUDA相差不超过3%，并在A30上比GPflow和GPyTorch库分别最高快1.75倍和2.17倍，在AMD Instinct MI210上分别最高快1.79倍和4.62倍。而在Intel Arc B580（一张FP64吞吐率低于所用CPU的消费级显卡）上，未经修改的SYCL后端则落后于GPRat的CPU后端。总体而言，oneMath对厂商原生BLAS的分发调度，使单一SYCL代码库无需任何厂商专用代码即可在NVIDIA上媲美CUDA，并在AMD上超越成熟的Python高斯过程框架。

    arXiv:2611.17565v1 Announce Type: cross  Abstract: Modern multi-vendor GPU environments force a choice between portable code and native performance. GPRat is an open-source Python library for exact Gaussian process regression built on the HPX asynchronous many-task runtime, tiled algorithms, and CPU/CUDA BLAS backends. We contribute a UXL Foundation oneMath backend with a hybrid HPX/SYCL design, extending GPRat to AMD and Intel GPUs. The new SYCL backend comes within 3% of native CUDA on an NVIDIA A30 at the largest problem size and outperforms the GPflow and GPyTorch libraries by up to 1.75x and 2.17x on the A30 and up to 1.79x and 4.62x on an AMD Instinct MI210. On an Intel Arc B580, a consumer-class card with FP64 throughput below our CPUs, the unmodified SYCL backend trails GPRat's CPU backend. Overall, oneMath's dispatch to vendor-native BLAS lets a single SYCL codebase match CUDA on NVIDIA and outperform established Python GP frameworks on AMD, without vendor-specific code.
    
[^119]: 基准测试不会反驳你：在自我分歧中学习

    Benchmarks Don't Talk Back: Learning Through Self-Disagreement

    [https://arxiv.org/abs/2611.17563](https://arxiv.org/abs/2611.17563)

    该论文提出“辩证能力”这一概念——即知道何时坚持立场、何时向更优答案让步的能力，证明其无法通过准确率训练或推理时的自我反思获得，而应借鉴哲学辩证法思想，通过模型自身生成的对立观点进行训练。

    

    基准测试通常是让语言模型独自面对一个问题，而实际部署中很少出现这样的情形：答案会受到质疑，模型会在本身错误的反驳推理下放弃原本正确的答案。我们证明，在这种情形下失效的并非准确率的不足，而是一种独立存在的能力——知道何时坚持立场、何时向更优的立场让步——我们将其称为辩证能力。这种能力以首次回答的准确率为基础，但并不由其必然导出：可以显著提升模型的回答水平却对这种能力毫无帮助，而仅针对准确率的训练甚至会使其发生退化。这种能力也无法在推理阶段通过自我反思或增加采样次数来获得。借鉴哲学中的辩证传统——其中富有成效的对立产生于立场内部，而非从外部强加——我们假设这种能力可以通过模型自身提供的对立观点来进行训练。它提出一个立场……

    arXiv:2611.17563v1 Announce Type: cross  Abstract: A benchmark usually measures a language model alone with a question, and deployment rarely looks like that: answers get contested, and models give up correct ones under counter-reasoning that is itself wrong. We show that what fails there is not a shortfall of accuracy but a capability in its own right -- knowing when to hold a position and when to yield to a better one -- which we call dialectical competence. It stands on first-answer accuracy without following from it: a model can be made materially better at answering and no better at this, and training for accuracy alone can drift against it. Nor can it be bought at inference, by self-reflection or by sampling more. Taking a cue from the dialectical tradition in philosophy, in which productive opposition arises within a position rather than being imposed on it from outside, we hypothesize that this capability can be trained from opposition the model supplies itself. It states a pos
    
[^120]: 大型非线性模拟电路中的对比局部学习

    Contrastive local learning in large nonlinear analog circuits

    [https://arxiv.org/abs/2611.17561](https://arxiv.org/abs/2611.17561)

    本文证明了基于对比局部规则训练的非线性模拟电路可以扩展到超过一万个自适应边的规模并学习更大型的机器学习任务，同时在硬件非理想条件下保持稳健性。

    

    在物理硬件中进行机器学习，通过在器件物理中同时实现计算、存储与自适应，有望降低训练成本。对比局部学习网络提供了一条极具吸引力的途径：电路在自由态和弱钳制态下弛豫至平衡状态，每个自适应元件根据局部电压对比而非反向传播的误差信号进行更新。然而，非线性晶体管实现能否扩展到超越概念验证电路的规模，以及控制这种扩展的设计原则是什么，此前一直是未知的问题。本文表明，通过对比局部规则训练的非线性模拟电路能够学习更大规模的机器学习任务，并在硬件非理想性下保持稳健。利用器件级电路仿真，我们系统地探索了初始化、工作区域、依赖于损失的钳制方式以及训练参数，并将对比局部学习网络扩展至超过一万个自适应边。（注：原摘要在此处被截断）

    arXiv:2611.17561v1 Announce Type: cross  Abstract: Machine learning in physical hardware promises to reduce training cost by colocating computation, memory and adaptation within device physics. Contrastive local learning networks (CLLNs) offer a compelling route: an electrical circuit relaxes to equilibrium in free and weakly clamped states, and each adaptive element updates from local voltage contrasts rather than backpropagated error signals. Yet whether nonlinear transistor implementations can scale beyond proof-of-concept circuits, and what design principles control that scaling, have remained unknown. Here we show that nonlinear analog circuits trained by contrastive local rules can learn larger machine-learning tasks and remain robust under hardware nonidealities. Using device-level circuit simulations, we systematically explore initialization, operating regime, loss-dependent clamping and training parameters, and scale CLLNs beyond ten thousand adaptive edges. The same local rul
    
[^121]: 具有实时物理可解释性的生成式设计：迈向人工智能时代的产品定义

    Generative Design with Real-time Physics Explainability: Towards Product Definition in the Age of AI

    [https://arxiv.org/abs/2611.17560](https://arxiv.org/abs/2611.17560)

    提出将基于可逆神经网络的逆向设计与实时生成式物理仿真相结合的统一生成式设计框架，借助不确定性量化和经典工具验证来确保设计的物理可行性与可解释性。

    

    在本文中，我们探讨了在机械工程领域应用生成式人工智能的机遇。一方面，近年来基于可逆神经网络的生成式方法被提出用于求解逆向设计问题，能够从性能标签推断出设计参数。另一方面，生成式人工智能模型已证明其可作为基于物理的仿真的实时替代模型。在此，我们建议将这两条技术路线结合到一个统一框架中：可逆神经网络用于从客户需求的角度控制设计过程，而系统物理的生成式仿真则增加了一层可解释性，用以说明所提出的设计为何能够满足需求。鉴于生成式设计容易产生幻觉，我们提出利用不确定性量化来防范物理上不可行的请求。最后，采用经典工具进行验证是生成式设计的必要要求。

    arXiv:2611.17560v1 Announce Type: new  Abstract: In this article, we explore opportunities from using generative AI in mechanical engineering. On the one hand, generative methods based on invertible neural networks have been recently proposed for the solution of inverse design problems, allowing to infer design parameters from performance labels. On the other hand, generative AI models have shown that they can act as real time capable surrogates for physics based simulations. Here we suggest that these two streams should be combined in a unified framework, where INN are used to control the design process from the customer requirements perspective and the generative simulation of the systems physics adds a layer of explainability, why the proposed design fulfils the requirements. As generative design is prone to hallucinations, we propose uncertainty quantification to fend off physically infeasible requests. Finally, validation with classical tools is a requirement for generative design
    
[^122]: 一种面向工业过程目标中心因果发现的合成基准

    A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes

    [https://arxiv.org/abs/2611.17556](https://arxiv.org/abs/2611.17556)

    该论文提出了一个用于工业过程目标中心因果发现的合成基准，其生成器是受批量生产启发的两层结构因果模型，由可独立设置的组件构成，能够逐一调节识别目标原因的难度，从而系统评估各类因果发现方法。

    

    因果发现有望找出哪些记录变量驱动过程目标，例如产品质量、产量或能耗。具有已知原因的过程数据十分罕见，而在现有基准中，识别目标原因的难度要么由记录的数据所固定，要么针对整个图统一设置。我们提出了一个用于工业过程目标中心因果发现的合成基准。其生成器是一个受批量生产启发的两层结构因果模型，由各自模拟过程数据某一现象（如控制回路或延迟效应）的组件组装而成。该图围绕目标构建：在没有传感器误差的情况下，每个非原因都可以通过对其条件化或依据时间顺序与直接原因区分开来。每个组件均可独立设置，因此可以一次改变一个条件来调整难度。我们评估了八种方法，从关联度量到效应估计（原文在此处截断）。

    arXiv:2611.17556v1 Announce Type: cross  Abstract: Causal discovery promises to find which recorded variables drive a process target such as product quality, yield or energy use. Process data with known causes are rare, and in existing benchmarks the difficulty of identifying the causes of a target is either fixed by the recorded data or set for the whole graph. We present a synthetic benchmark for target-centric causal discovery in industrial processes. Its generator is a two-level structural causal model inspired by batch production, assembled from components that each model one phenomenon of process data, such as a control loop or a delayed effect. The graph is built around the target: without sensor error, every non-cause can be distinguished from the direct causes by conditioning on them or by time order. Each component is set independently, so the difficulty can be varied one condition at a time. We evaluate eight methods, from association measures to effect estimation, with an e
    
[^123]: 基于模拟与实验数据的面向二维X射线衍射（XRD）解释的不确定性感知持续学习

    Uncertainty-Aware Continual Learning for 2D XRD Interpretation Using Simulated and Experimental Data

    [https://arxiv.org/abs/2611.17547](https://arxiv.org/abs/2611.17547)

    该论文提出了一个不确定性感知的持续学习概念验证框架，通过融合基于MMD的领域自适应、CNN-SNGP不确定性估计以及结合知识蒸馏的重放机制，实现二维XRD数据在模拟与实验域之间稳健且可持续更新的解释。

    

    自主实验系统需要能够解释异构测量数据，并随着数据积累保持预测性能的表征模型。开发此类模型需要解决数据源之间的领域不匹配、对陌生输入的过度自信预测，以及顺序更新过程中的灾难性遗忘等问题。在本工作中，我们提出了一个用于二维X射线衍射（XRD）数据不确定性感知持续解释的概念验证框架。该框架集成了多个成熟的组件：基于最大均值差异（MMD）的领域自适应方法，用于减小模拟数据与实验数据之间的差异；采用谱归一化神经高斯过程的卷积神经网络（CNN-SNGP），用于预测不确定性估计；以及结合知识蒸馏的重放机制，用于模型的顺序更新。我们使用模拟数据和先前收集的晶体……（摘要原文在此处被截断）实验数据对该框架进行了评估。

    arXiv:2611.17547v1 Announce Type: cross  Abstract: Autonomous experimentation systems require characterization models that can interpret heterogeneous measurements and maintain predictive performance as data accumulate. Developing such models involves addressing domain mismatch between data sources, overconfident predictions on unfamiliar inputs, and catastrophic forgetting during sequential updates. In this work, we present a proof-of-concept framework for uncertainty-aware continual interpretation of two-dimensional X-ray diffraction (XRD) data. The framework integrates established components: maximum mean discrepancy-based domain adaptation to reduce simulation-to-experiment discrepancies, a convolutional neural network with a spectral-normalized neural Gaussian process (CNN--SNGP) for predictive uncertainty estimation, and replay with knowledge distillation for sequential model updating. We evaluate the framework using simulated and previously-collected experimental data for crysta
    
[^124]: 通过扩展课程学习为视觉世界模型规划学习潜在目标代价

    Learning Latent Goal Costs for Visual World-Model Planning with an Expanding Curriculum

    [https://arxiv.org/abs/2611.17529](https://arxiv.org/abs/2611.17529)

    提出在冻结 I-JEPA 嵌入上拟合小型残差适配器，通过扩展课程采集的成功与失败轨迹中观测结果的顺序信息自动学习潜在目标代价，无需任务特定的进展标签，从而有效改进冻结视觉世界模型的图像目标规划效果。

    

    使用冻结视觉世界模型进行图像目标规划，其效果取决于如何依据目标对预测的观测结果进行评分。预训练视觉编码器嵌入之间的均方误差（MSE）可以在目标附近提供有用的信号，但它会偏向于那些在视觉上与目标相似、却被障碍物隔开且难以到达的状态。我们在冻结的 I-JEPA 嵌入之上拟合一个小型残差适配器，以改进这种潜在目标代价。规划器从每个目标附近的初始状态开始，从逐步扩大的初始状态区域中收集轨迹。该适配器从成功轨迹上观测结果的先后顺序中学习，并从失败的推演中推断出额外的比较信息。整个拟合过程不使用任何任务特定的进展标签，因此该流程可以应用于其他目标到达任务。本工作使用了 DotWall 和 PointMaze 环境。在具有相同起始状态、目标、世界模型和规划器设置的成对 50 例评估中，成功率……（原文摘要在此处截断）

    arXiv:2611.17529v1 Announce Type: cross  Abstract: Image-goal planning with a frozen visual world model depends on how predicted observations are scored against the goal. Mean-squared error (MSE) between embeddings from a pretrained visual encoder can provide a useful signal near a goal, yet favor states across an obstacle that are visually similar to the goal and difficult to reach. We fit a small residual adapter on top of frozen I-JEPA embeddings to improve this latent goal cost. Starting near each goal, a planner collects trajectories from progressively wider regions of initial states. The adapter learns from the order of observations along successful trajectories, with additional comparisons inferred from failed rollouts. No task-specific progress labels are used for fitting, so the procedure can be applied to other goal-reaching tasks. DotWall and PointMaze are used in this work. In paired 50-case evaluations with identical starts, goals, world models, and planner settings, succe
    
[^125]: 面向量化张量程序的用户可调度近似方法

    User-Schedulable Approximations for Quantized Tensor Programs

    [https://arxiv.org/abs/2611.17526](https://arxiv.org/abs/2611.17526)

    该论文将Halide算法与调度分离的思想扩展到数值表示层面，提出由量化器、整数表示、位打包器等可复用单元组合而成的编解码器来表达量化近似，并通过新的approximate_by指令使编码器和解码器成为可被调度的普通阶段，从而大幅减少量化张量程序所需手工维护的内核数量。

    

    量化技术使语言模型能够在边缘设备上运行，但它大幅增加了张量库必须维护的内核数量：每种权重格式都必须与每种激活格式在每套指令集和GPU上配对，且编码器与解码器必须就存储的字节达成一致。在GGML（llama.cpp所使用的张量库）中，这些内核均由手工编写，而在我们的Arm CPU上运行的54对权重-激活组合中，仅有8对拥有分块矩阵乘法内核。我们将Halide对算法与调度的分离扩展到了值的表示。每种近似都是一个编解码器，即一个编码器与一个解码器的配对，由可复用单元构成：量化器、整数表示、位打包器和布局。一个新的指令approximate_by可将一个编解码器应用于程序的某个阶段，并返回编码器和解码器作为Halide可调度的普通阶段，从而使解码器能够融合进矩阵乘法的寄存器分块中。

    arXiv:2611.17526v1 Announce Type: cross  Abstract: Quantization lets language models run on edge devices, but it sharply increases the number of tensor kernels a library must maintain: every weight format must be paired with every activation format on every instruction set and GPU, with an encoder and a decoder that agree on the stored bytes. In GGML, the tensor library used by llama.cpp, these kernels are written by hand, and only 8 of the 54 weight-activation pairs it runs on our Arm CPU have a tiled matrix-multiplication kernel.   We extend Halide's separation of algorithm and schedule to the representation of values. Each approximation is a codec, an encoder paired with a decoder, composed from reusable units: quantizers, integer representations, bit packers and layouts. A new directive, approximate_by, applies a codec to one stage of a program and returns the encoder and decoder as ordinary stages that Halide can schedule, so both decoders fuse into the register tiles of a matrix 
    
[^126]: 模型退火：通过复习与整合实现数据高效学习

    Model Annealing: Data-Efficient Learning through Review and Consolidation

    [https://arxiv.org/abs/2611.17524](https://arxiv.org/abs/2611.17524)

    模型退火通过将训练划分为学习率升降循环的多个回合、维护过去模型状态的预测记忆库，并从对抗性分歧和记忆预测中学习，实现了对网络自身知识的复习与整合，在数据受限场景下显著提升分类准确率且不增加推理开销。

    

    在训练预算耗尽之前，小规模标记数据集早已被模型记住，而剩余待拟合的独热标签此时大多只是在进一步增强置信度。模型退火将剩余预算用于复习与整合，从网络自身的知识中重新学习。训练被划分为多个回合，每个回合的学习率重新升高后再冷却，每个冷却后的状态都会加入一个由过去自身组成的记忆库，每个新回合从标签、从该记忆的柔和化预测，以及从学生与记忆分歧最大处寻找的对抗性“梦境”中学习。该记忆对预测而非权重进行平均，部署的网络是最后一个学生，因此推理不产生额外开销。在CIFAR-100、Tiny-ImageNet和Imagewoof上，涵盖四种架构家族和5%到100%的标签比例，模型退火在33个配置中的28个里提升了使用AdamW训练的ConvNeXt、ViT和MLP-Mixer相对交叉熵训练的top-1准确率。

    arXiv:2611.17524v1 Announce Type: new  Abstract: A small labelled dataset is memorized long before the training budget runs out, and the one-hot labels that remain to be fitted then mostly sharpen confidence. Model annealing spends the rest of the budget on review and consolidation, relearning from the network's own knowledge. Training is divided into episodes whose learning rate is raised again and cooled, every cooled state joins a memory of past selves, and each new episode learns from the labels, from the tempered predictions of this memory, and from adversarial dreams sought where student and memory disagree most. The memory averages predictions rather than weights, and the deployed network is the last student, so inference costs nothing extra. Across CIFAR-100, Tiny-ImageNet and Imagewoof, four architecture families and 5\% to 100\% of the labels, model annealing raises the top-1 accuracy of the AdamW-trained ConvNeXt, ViT and MLP-Mixer over cross-entropy training in 28 of 33 con
    
[^127]: 面向动力系统的多保真度算子推断方法

    Multifidelity Operator Inference for Dynamical Systems

    [https://arxiv.org/abs/2611.17503](https://arxiv.org/abs/2611.17503)

    本文提出MF-OpInf方法，通过受控制变量法启发的多保真度最小二乘公式，将有限的高保真度快照与丰富的低保真度快照相结合，为非线性动力系统学习更准确稳定的非侵入式降阶模型。

    

    本文提出了一种多保真度的算子推断方法，用于学习非线性动力系统的非侵入式、基于物理的降阶模型。OpInf通过求解最小二乘回归问题，从快照数据的投影中学习降阶算子。然而，生成高保真度快照的计算成本可能非常昂贵，这限制了许多航空航天应用中的参数覆盖范围以及训练轨迹的时间跨度，从而可能导致降阶模型预测不准确或不稳定。本工作提出了MF-OpInf方法，通过一种受控制变量法启发的多保真度最小二乘公式，将有限的高保真度快照与更丰富的低保真度快照相结合，以在多条轨迹上学习单一的一组降阶算子。低保真度轨迹提供了额外的参数覆盖范围，并延伸超出高保真度训练的时间范围。

    arXiv:2611.17503v1 Announce Type: cross  Abstract: This paper presents a multifidelity (MF) formulation of Operator Inference (OpInf) for learning non-intrusive, physics-based reduced-order models (ROMs) of nonlinear dynamical systems. OpInf learns reduced operators from projections of snapshot data by solving a least squares regression problem. However, generating high-fidelity snapshots can be computationally expensive, limiting parameter coverage and the temporal extent of training trajectories for many aerospace applications, which can lead to inaccurate or unstable ROM predictions. This work proposes MF-OpInf, which combines limited high-fidelity snapshots with more plentiful low-fidelity snapshots through a control-variates-inspired MF least squares formulation to learn a single set of reduced operators across multiple trajectories. Low-fidelity trajectories provide additional parameter coverage and extend beyond the high-fidelity training time horizon. The low-fidelity snapshots
    
[^128]: 基于光电容积脉搏波（PPG）的连续无袖带血压估计：从方法到临床转化

    Continuous Cuffless Blood Pressure Estimation Using PPG From Methods to Clinical Translation

    [https://arxiv.org/abs/2611.17499](https://arxiv.org/abs/2611.17499)

    本综述系统回顾了基于PPG的无袖带血压估计的完整技术链条，从生理机制、波形特征、PAT/PTT、机器学习到临床转化，并强调应以证据强度而非仅以报告误差来评估研究。

    

    连续血压（BP）监测能够捕捉间歇性袖带测量所遗漏的血压变化，从而改善高血压管理。光电容积脉搏波（PPG）被广泛探索用于无袖带血压估计，因为光学传感器价格低廉、可穿戴，并且已经广泛应用于许多消费级和临床设备中。然而，PPG并不能直接测量动脉压力：血压是从波形特征或脉搏时间推断出来的，而这些特征同时也受到血管张力、动脉僵硬度、传感器接触情况、运动以及个体生理差异的影响。本综述沿着完整的测量链回顾了基于PPG的无袖带血压估计研究，涵盖生理机制、波形特征、脉搏到达时间（PAT）、脉搏传导时间（PTT）、机器学习方法、数据集、校准、验证、部署、偏差以及临床转化。研究评估依据的是其所提供的证据强度，而不仅仅是报告的误差……

    arXiv:2611.17499v1 Announce Type: cross  Abstract: Continuous blood pressure (BP) monitoring could improve hypertension management by capturing changes that intermittent cuff measurements miss. Photoplethysmography (PPG) is widely explored for cuffless BP estimation because optical sensors are inexpensive, wearable, and already present in many consumer and clinical devices. PPG, however, does not measure arterial pressure: BP is inferred from waveform characteristics or pulse timing, which also depend on vascular tone, arterial stiffness, sensor contact, motion, and individual physiology. This survey reviews PPG-based cuffless BP estimation along the full measurement chain, covering physiological mechanisms, waveform features, pulse arrival time (PAT), pulse transit time (PTT), machine-learning methods, datasets, calibration, validation, deployment, bias, and clinical translation. Studies are assessed by the strength of evidence they provide rather than by reported error alone, with pa
    
[^129]: 不透明串行深度的可操作化定义

    An Operationalization of Opaque Serial Depth

    [https://arxiv.org/abs/2611.17495](https://arxiv.org/abs/2611.17495)

    本文将“不透明串行深度”操作化为NLS深度指标，通过定义“自然语言根植节点”作为可解释瓶颈的标准，为衡量模型的非言语化串行认知能力提供了可在训练前判定架构的依据，以应对架构变化可能削弱思维链可监控性的风险。

    

    思维链目前是监督AI模型的一个宝贵工具，但某些架构上的转变可能会显著降低思维链的可监控性。我们操作化了一种度量指标，作为模型能够执行的非言语化串行认知量的代理，这是“不透明串行深度”概念的一个具体实例化——大致而言，即模型计算中不经过可解释瓶颈的最长路径的长度。我们的主要贡献是提出了一个判定何种节点可算作此类瓶颈的特定标准。我们将根植于自然语言的节点定义为：从产生人类可理解输出的预训练先验中初始化而来的节点，其输出空间未被扩展至超出该先验的范围，且模型未被修改以将其输出解释为不同类型的数据。通过构建方式，由此得到的度量指标——NLS深度，可以在训练之前基于架构进行判定……（摘要原文在此处截断）

    arXiv:2611.17495v1 Announce Type: new  Abstract: Chain-of-thought (CoT) is currently a valuable tool for overseeing AI models, but some architectural shifts could significantly reduce CoT monitorability. We operationalize a measure that serves as a proxy for the amount of unverbalized serial cognition a model can perform, as a specific instantiation of the notion of opaque serial depth: roughly, the length of the longest path through a model's computation that does not pass through an interpretable bottleneck. Our main contribution is a particular standard for what counts as such a bottleneck. We define natural-language-rooted (NL-rooted) nodes as nodes that were initialized from a pre-training prior to produce human-understandable outputs, whose output space has not been expanded beyond that prior, and whose outputs the model has not been modified to interpret as a different type of data. By construction, the resulting measure, NLS depth, can be adjudicated before training from an arc
    
[^130]: PF-Exec：基于锚定势能方向决策器与逐维度增益学习执行器的大规模无人机中继负载均衡

    PF-Exec: Large-Scale UAV Relay Load Balancing with an Anchored Potential Direction Decider and a Per-Dimension Gain-Learning Executor

    [https://arxiv.org/abs/2611.17492](https://arxiv.org/abs/2611.17492)

    提出两层决策框架 PF-Exec：锚定势能方向决策器每时刻输出闭式目标方向，逐维度增益学习执行器将局部观测映射为各维度增益以自适应调节动作幅度，从而在大规模无人机中继网络中实现关联、带宽、功率与位置的联合负载均衡。

    

    大规模无人机网络中的中继负载均衡需要对关联、带宽、功率和悬停位置进行联合决策，而决策维度随无人机数量增长，且每架无人机只能观测到其局部状态。本文将决策拆分为两层：锚定势能方向决策器在每一时刻输出一个闭式目标，并仅返回其方向；增益学习执行器将局部观测映射为每个维度的正增益，将增益与目标组合，并把动作限制在物理单步界限内；这两层共同构成 PF-Exec。单位增益是默认起始值，因此在训练之前，执行器退化为纯势能策略。负载对带宽是一阶齐次的，使得带宽目标具有闭式形式，而位置目标则是所服务用户的按需求加权质心。系统采用两跳接入与毫米波回传链路以及软关联机制……

    arXiv:2611.17492v1 Announce Type: new  Abstract: Relay load balancing in large-scale UAV networks requires joint decisions on association, bandwidth, power and hovering position, while the dimension grows with the number of UAVs and each UAV sees only its local state. This paper splits the decision into two layers. An anchored potential direction decider emits a closed-form target at every instant and returns its direction only. A gain-learning executor maps a local observation into a positive gain per dimension, composes gain and target, and clips the action against the physical single-step bound; the two layers form PF-Exec. The unit gain is the default start, so the executor reduces to the pure potential policy before training. The load is first-order homogeneous in bandwidth, making the bandwidth target closed form, and the position target is the demand-weighted centroid of the served users. The system uses a two-hop access and millimetre-wave backhaul link with soft association, a
    
[^131]: PF-Exec 的收敛性质：齐次结构、锚定闭式解与增益动力学

    Convergence Properties of PF-Exec: Homogeneous Structure, Anchored Closed Form, and Gain Dynamics

    [https://arxiv.org/abs/2611.17488](https://arxiv.org/abs/2611.17488)

    本文为 PF-Exec 算法提供了完整的分层数学刻画：由齐次分解闭式导出带宽目标与可行性界、证明对数拐点最优性与 Jain 公平指数一致、给出单纯形乘积上阻尼 softmax 的不动点存在性，并完整刻画了增益迭代在临界增益前几何收敛、临界处有界振荡、临界后出现有界双状态周期循环的行为。

    

    本文为 PF-Exec（配套系统论文中提出的算法）建立了分层式的数学刻画。需求成比例的带宽分配产生一个精确的一阶齐次分解，由此可以闭式形式导出等负载带宽目标及其可行性界。对数拐点视角保持了严格的负载顺序，并与 Jain 公平性指数共享同一个全局最优值，而需求加权的质心则使位置势能最小化。阻尼 softmax 关联模块在单纯形的乘积空间上存在一个不动点。截断的一维增益迭代获得了完整的刻画：低于临界增益时呈几何收敛，恰在临界增益处出现依赖初值的有界振荡，高于临界增益时则存在一族有界的双状态周期循环。全局不变误差球与显式跟踪递推共同完成了执行保证的表述。数值审计证实了上述结论。

    arXiv:2611.17488v1 Announce Type: cross  Abstract: This paper establishes a layered mathematical char acterisation of PF-Exec, the algorithm of the companion system paper. Demand-proportional bandwidth allocation yields an exact first-order homogeneous decomposition, from which the equal-load bandwidth target and its feasibility bound follow in closed form. The logarithmic knee view preserves strict load order and shares its global optimum with the Jain fairness index, while the demand-weighted centroid minimises the position potential. The damped softmax association block possesses a fixed point on the product of simplices. The truncated one-dimensional gain iteration admits a complete characterisation, with geomet ric convergence below the critical gain, initial-value-dependent bounded oscillation at it, and a family of bounded two-state cycles above it. A global invariant error ball and an explicit tracking recursion complete the execution guarantee. A numerical audit confirms the s
    
[^132]: 通过输入优化创建引导向量

    Creating Steering Vectors via Input Optimisation

    [https://arxiv.org/abs/2611.17481](https://arxiv.org/abs/2611.17481)

    本文提出ActGCG方法，通过优化输入后缀来模拟激活引导向量的效果，使攻击者无需白盒访问模型内部状态，即可通过修改输入文本实现与激活引导相同的行为操控，包括语言切换和越狱攻击。

    

    激活引导通过在生成过程中向语言模型的残差流添加引导向量来影响模型行为，但这种方法需要对模型的白盒访问权限，以便在每个生成步骤中修改隐藏状态。我们提出了ActGCG，这是对贪婪坐标梯度方法的一种改进，它通过优化一个后缀来匹配将引导向量添加到残差流后所获得的目标激活。我们进一步引入了同时在多个提示上训练的通用后缀、一个在优化过程中将后缀改写为人类可读文本的模型翻译步骤，以及一种去除离散词元空间约束的软后缀变体。我们研究了两种行为：引导模型用德语回答，以及用于越狱攻击的反拒绝引导。对于德语引导，离散后缀几乎完美地复现了引导向量的行为，并且在提示和模型之间具有良好的迁移性。对于反拒绝引导，这种等效性较弱。

    arXiv:2611.17481v1 Announce Type: new  Abstract: Activation steering influences a language model's behaviour by adding a steering vector to its residual stream during generation, but it requires white-box access to modify the hidden states at every generation step. We introduce ActGCG, a modification of Greedy Coordinate Gradient which optimises a suffix to match the target activations obtained by adding a steering vector to the residual stream. We further introduce universal suffixes trained across many prompts at once, a model-translation step that rewrites the suffix into human-readable text midway through optimisation, and a soft-suffix variant to remove the discrete token space constraint. We study two behaviours: steering a model to answer in German, and anti-refusal steering for jailbreaking. For German, a discrete suffix reproduces the steering vector's behaviour almost perfectly and transfers well across prompts and models. For anti-refusal, the equivalence is weaker, succeedi
    
[^133]: ERGo：基于存内计算架构的高能效混合图神经网络训练

    ERGo: Energy-Efficient Hybrid Graph Neural Network Training on Processing-In-Memory Architectures

    [https://arxiv.org/abs/2611.17478](https://arxiv.org/abs/2611.17478)

    提出混合训练框架ERGo，在训练过程中自动从全参数阶段切换到基于低秩表示的参数高效阶段，从而减少PIM架构上的片外内存访问，显著提升GNN训练的性能与能效。

    

    存内计算已被提出作为训练深度神经网络（包括图神经网络）的替代计算范式。尽管取得了这些进展，在PIM设备上训练GNN工作负载仍需要进行片外内存访问。这种片外访问在性能和能耗方面的代价高昂，从而影响了PIM平台上训练过程的整体能效。在本文中，我们提出了一种名为ERGo的新型混合训练框架，它能在训练过程中自动从全参数阶段切换到参数高效阶段，且预测准确率的损失可以忽略不计。参数高效阶段采用GNN权重的低秩表示，有效减少了可训练参数的数量。这有助于减少在PIM架构上进行端到端训练过程中的片外访问，从而在性能和能效方面带来显著提升。

    arXiv:2611.17478v1 Announce Type: new  Abstract: Processing-in-memory (PIM) has been proposed as an alternative computing paradigm for training Deep Neural Networks, including Graph Neural Networks (GNNs). Despite these advancements, training GNN workloads on PIM devices necessitates off-chip memory access. This off-chip access is expensive in terms of performance and energy, thereby impacting the overall energy efficiency of the training process on PIM platforms. In this paper, we propose a novel hybrid training framework called ERGo that automatically switches from a full-parameter phase to a parameter-efficient phase during the training process with negligible loss in predictive accuracy. The parameter-efficient phase employs low-rank representations of GNN weights, effectively reducing the number of trainable parameters. This helps in reducing the off-chip access during the end-to-end training process on PIM architectures, leading to notable improvements in both performance and ene
    
[^134]: 混合专家模型的路由器能从集成中受益吗？

    Do Mixture-of-Experts Routers Benefit from Ensembling?

    [https://arxiv.org/abs/2611.17472](https://arxiv.org/abs/2611.17472)

    提出在混合专家模型的路由步骤中引入路由器集成方法，实验证明更多路由器可降低验证损失，且显式训练路由器最大化专家多样性能提升平均性能。

    

    神经网络的集成方法已被反复证明优于单一网络。集成理论将这种增益归因于各个成员的准确性以及它们之间的多样性。混合专家模型已经结合了许多专家，但由单个路由器决定每个输入下专家的权重分配。受集成理论的启发，我们将集成方法应用于路由步骤：由一组路由器集成（与专家联合训练）来提出专家权重。我们使用欠参数化网络在两个图像分类任务和两个表格回归任务上系统地评估了这种方法。我们通过路由器集成的性能-多样性分解来分析每个实验，据我们所知，这种分解方法此前尚未被开发应用于路由器。我们报告了四个发现。第一，更多的路由器降低了验证损失。第二，训练路由器显式地最大化专家多样性提高了平均性能。

    arXiv:2611.17472v1 Announce Type: cross  Abstract: Ensembles of neural networks have repeatedly been shown to outperform single networks. Ensemble theory attributes this gain to the accuracy of the individual members and the diversity among them. Mixture-of-experts models already combine many experts, but a single router decides how the experts are weighted for each input. Motivated by ensemble theory, we apply ensembling to the routing step: an ensemble of routers, trained jointly with the experts, proposes the expert weights. We systematically evaluate this approach using underparameterized networks on two image-classification and two tabular-regression tasks. We analyse each experiment with a performance-diversity decomposition of the router ensemble, which, to our knowledge, has not previously been developed for routers. We report four findings. First, more routers lower the validation loss. Second, training routers to explicitly maximize expert diversity improves the mean performa
    
[^135]: 对抗性马尔可夫决策过程中基于方差感知策略优化的改进遗憾界

    Improved Regret via Variance-Aware Policy Optimization in Adversarial MDPs

    [https://arxiv.org/abs/2611.17469](https://arxiv.org/abs/2611.17469)

    本文提出方差感知策略优化算法VOPO，将未知动力学对抗性MDP的遗憾值首项改进至 $\widetilde O(\sqrt{H^3SAK})$，并通过从已知到未知动力学的归约将剩余差距缩小至仅 $\sqrt H$ 因子。

    

    我们研究具有未知动力学和全信息奖励的回合制对抗性马尔可夫决策过程（MDP），实现了首项为 $\widetilde O(\sqrt{H^3SAK})$ 的遗憾值。其中 $H$ 为时域长度，$S$ 为各决策层的状态总数，$A$ 为动作数，$K$ 为回合数。该结果相比两个互不占优的先前最优首项分别改进了 $\sqrt{S/H}$ 和 $\sqrt{H^3}$ 因子，距下界规模仅剩 $\sqrt H$ 因子的差距。我们给出了针对“下游依赖”算法的从未知动力学到已知动力学的归约，这类算法在第 $h$ 步的策略仅由第 $h+1$ 步及之后的信息计算得到。这将剩余的首阶差距归约为寻找一个具有 $\widetilde O(H\sqrt K)$ 遗憾的下游依赖已知动力学算法，这样的算法将使未知动力学情形达到紧致的 $\widetilde O(H\sqrt{SAK})$ 首项。我们用策略优化（PO）算法 VOPO 来实例化该归约，其改进的保证……

    arXiv:2611.17469v1 Announce Type: new  Abstract: We study episodic adversarial MDPs with unknown dynamics and full-information rewards and achieve regret with leading term $\widetilde O(\sqrt{H^3SAK})$. Here $H$ is the horizon, $S$ the total number of states across decision layers, $A$ the number of actions, and $K$ the number of episodes. This improves the two incomparable prior leading terms by factors $\sqrt{S/H}$ and $\sqrt{H^3}$, leaving a factor $\sqrt H$ to the lower-bound scale. We give a reduction from unknown to known dynamics for \emph{downstream-dependent} algorithms, whose policy at step $h$ is computed only from information at steps $h+1$ and onward. This reduces the remaining leading-order gap to finding a downstream-dependent known-dynamics algorithm with $\widetilde O(H\sqrt K)$ regret, which would yield the tight $\widetilde O(H\sqrt{SAK})$ leading term for unknown dynamics. We instantiate the reduction with VOPO, a policy optimization (PO) algorithm. Its improved gua
    
[^136]: 医学图像中2D到3D配准的深度学习方法全面综述

    A thorough review on Deep Learning approaches to 2D-to-3D Registration in Medical Imaging

    [https://arxiv.org/abs/2611.17468](https://arxiv.org/abs/2611.17468)

    本综述全面梳理了医学图像2D到3D配准的深度学习方法，从流程设计、网络架构创新和领域自适应等多个角度分析了最新进展，凸显了深度学习相比传统优化方法在配准鲁棒性和实时性方面的优势。

    

    2D到3D图像配准是多种医学成像应用（如术中引导）的关键组成部分，它能够实现术前3D体数据与实时2D图像之间的对齐。传统的基于优化的方法常常面临非凸目标函数、对初始化敏感以及计算需求高等难题，而深度学习（DL）作为一种有前景的替代方案应运而生，提供了更强的配准鲁棒性和实时处理的潜力。本综述对为解决2D到3D配准固有挑战而开发的深度学习框架进行了全面分析。我们首先概述了深度学习的核心概念如何适配到配准问题中，然后从多个角度考察了近期的技术进展，包括流程设计、架构创新和领域自适应策略。本综述特别关注定量评估指标，重点阐述了如何衡量配准性能。

    arXiv:2611.17468v1 Announce Type: cross  Abstract: 2D-to-3D image registration is a key component for several medical imaging purposes such as intraoperative guidance, enabling the alignment between preoperative 3D volumes and real-time 2D images. While conventional optimisation-based methods often struggle with non-convex objectives, sensitivity to initialisation, and high computational demands, deep learning (DL) has emerged as a promising alternative, offering improved alignment robustness and the potential for real-time performance. This review provides a comprehensive analysis of DL frameworks developed to address the challenges inherent in 2D-to-3D registration. We first outline how core concepts of DL are adapted to the registration problem. We then examine recent methodological advancements from multiple perspectives, including pipeline design, architectural innovations, and domain adaptation strategies. Particular attention is given to quantitative metrics, highlighting how pe
    
[^137]: 面向电弧炉温度与化学成分预测的可用性感知历史反馈学习方法

    Availability-Aware Historical Feedback Learning for Electric Arc Furnace Temperature and Chemical Composition Prediction

    [https://arxiv.org/abs/2611.17463](https://arxiv.org/abs/2611.17463)

    该研究提出了一种可用性感知的历史反馈学习框架，在考虑60分钟测量延迟的约束下，将多源工业数据整合为逐步丰富的特征表示，显著提升了电弧炉终点温度和十二种化学元素最终浓度的预测性能。

    

    电弧炉（EAF）炼钢在再生钢和低碳钢生产中扮演着日益重要的角色。然而，炉料、能量输入和操作条件的波动性使得准确预测钢水温度和最终化学成分面临挑战。间歇性或延迟的测量限制了可用于及时预测的信息。本研究开发了一种可用性感知的历史反馈学习框架，用于预测电弧炉终点温度以及十二种化学元素的最终浓度。八个工业数据源被整合为三种逐步丰富化的预测特征表示，包含19个静态特征、39个时序特征和47个历史信息特征。历史结果仅在满足假设的60分钟测量发布延迟的条件下才被纳入。基于19,738炉温度数据和3,341炉化学成分数据，对十二种经过调优的回归算法进行了训练与评估。

    arXiv:2611.17463v1 Announce Type: new  Abstract: Electric arc furnace (EAF) steelmaking plays an increasingly important role in recycled and lower-carbon steel production. However, variability in charged materials, energy input, and operating conditions makes accurate prediction of molten steel temperature and final chemical composition challenging. Intermittent or delayed measurements limit the information available for timely prediction. This study develops an availability-aware historical-feedback learning framework for predicting EAF endpoint temperature and the final concentrations of twelve chemical elements. Eight industrial data sources are integrated into three progressively enriched predictor representations comprising 19 static, 39 temporal, and 47 history-informed features. Historical outcomes are incorporated only when eligible under an assumed 60-minute measurement-release delay. Using 19,738 temperature heats and 3,341 chemistry heats, twelve tuned regression algorithms 
    
[^138]: 面向晶体生成模型推理时自适应的扩散树采样方法

    Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models

    [https://arxiv.org/abs/2611.17462](https://arxiv.org/abs/2611.17462)

    提出扩散树采样（DTS）方法，通过在去噪轨迹上进行蒙特卡洛树搜索，在推理阶段引导晶体扩散生成模型，既能直接复用现有性质条件模型而不扭曲目标分布，又支持多目标采样以高效生成满足多个冲突性质的晶体候选。

    

    基于扩散的生成模型已成为对合理晶体结构的强大先验，但要将这一先验转化为有前景的候选材料，需要同时向多个相互冲突的性质进行引导。微调成本高昂，且每引入一个新目标都需要训练一个新模型；而可微引导方法又不适用于许多现实世界的任务。我们转而在推理阶段通过扩散树采样来适配这些生成模型，这是一种作用于去噪轨迹上的蒙特卡洛树搜索方法。我们在DTS基础上做出了两项与AI-for-科学应用尤其相关的技术贡献。第一，我们展示了如何将现有的性质条件生成模型（例如为无分类器引导而微调的模型）作为DTS内部的引导性提议分布加以利用，且不会扭曲目标分布。第二，我们将DTS扩展到多目标采样，使得单次搜索即可产生覆盖整个……（原文在此处截断）

    arXiv:2611.17462v1 Announce Type: cross  Abstract: Diffusion-based generative models have become strong priors over plausible crystal structures, but turning that prior into promising candidates requires steering toward several conflicting properties at once. Fine-tuning is expensive and produces a new model for every new objective, while differentiable guidance is not applicable to many real-world tasks. We instead adapt these generative models at inference time using Diffusion Tree Sampling (DTS), a Monte Carlo tree search procedure over the denoising trajectory. We make two technical contributions on top of DTS that are particularly relevant for AI-for-science applications. First, we show how to use existing property-conditional generative models (e.g., models fine-tuned for classifier-free guidance) as guided proposals inside DTS, without distorting the target distribution. Second, we extend DTS to multi-objective sampling, so that a single search produces samples spanning an entir
    
[^139]: Unfinished-Bench：在真实工作环境中测量持久性智能体的规则违反行为

    Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments

    [https://arxiv.org/abs/2611.17460](https://arxiv.org/abs/2611.17460)

    Unfinished-Bench是一个在真实工作环境中评估AI智能体持久性与规则遵守之间冲突的基准测试，发现任务持久性会显著提高智能体违反内部文档中规则的比例（从32.2%升至43.6%），而违反用户明确陈述的同一规则的比例仅为8.5%。

    

    AI智能体越来越多地被赋予完成任务所需的持久性，以便在来自内部文档和他人权限等诸多未明示约束下，长时间自主运行被委托的工作。当完成任务需要违反某条规则时，智能体必须独自做出决定，然而现有的指令-规则冲突基准测试要么在提示词中明确陈述规则，要么将规则隐藏起来，且都不评估持久性。我们提出了Unfinished-Bench，这是一个让模型在无用户参与的自主运行中、在嵌入规则的真实工作环境中展现持久性的基准测试。在其跨越7个组织和7个风险类别的97个场景中，每个场景的目标与规则都无法同时成立。我们发现，持久性使智能体违反内部文档中规则的比例从32.2%上升到43.6%，是用户明确陈述同一规则时8.5%违规率的五倍。在文档规则的违反行为中，98.3%发生在规则被展示之后，且在被评判的样本中有89%……

    arXiv:2611.17460v1 Announce Type: cross  Abstract: AI agents are increasingly equipped with persistence toward task completion to run delegated work autonomously for long periods, under many unstated constraints from internal documents and other people's authority. When completion requires violating a rule, the agent must decide alone, yet benchmarks of instruction-rule conflicts state the rule in the prompt or keep it hidden and do not evaluate persistence. We present Unfinished-Bench, a benchmark that gives models persistence in autonomous runs without a user, in realistic work environments with embedded rules. In each of its 97 scenarios across 7 organizations and 7 risk categories, the goal and the rule cannot both hold. We find that persistence raised violations of a rule in an internal document from 32.2% to 43.6%, five times the 8.5% for the same rule stated by the user. Of the document-rule violations, 98.3% occurred after the rule was displayed, and in 89% of a judged sample t
    
[^140]: 基于大步长梯度下降的分类：统一自稳定机制与收敛速率

    Classification with Large-Step Gradient Descent: Unifying Self-Stabilization and Convergence Rates

    [https://arxiv.org/abs/2611.17459](https://arxiv.org/abs/2611.17459)

    本文将统一的几何视角扩展到指数尾损失（如逻辑损失和交叉熵损失），首次在分类设定下将大步长梯度下降的自稳定机制与收敛速率分析统一起来。

    

    神经网络通常使用超过经典稳定性阈值的步长进行梯度下降（GD）训练，在这种情形下，标准的收敛理论不再适用。尽管该情形表现出非单调的损失和振荡的动力学，但它往往能带来更快的收敛速度和更好的泛化性能。现有的理论视角提供了互补但不完整的见解：一种视角识别出自稳定机制，该机制驱动梯度下降趋向平坦区域，但无法给出收敛速率；另一种视角针对特定损失推导出了严格的收敛速率，但没有解释其背后的稳定机制；第三种视角针对过参数化最小二乘问题统一了上述观点，但其依赖于极小值流形的存在，而这一流形在分类问题中并不存在。在这项工作中，我们将这一统一的几何视角扩展到具有指数尾部的损失函数，包括可分线性分类中的逻辑损失和交叉熵损失……

    arXiv:2611.17459v1 Announce Type: new  Abstract: Neural networks are often trained with gradient descent (GD) using step sizes that exceed classical stability thresholds, a regime where standard convergence theory no longer applies. Despite exhibiting non-monotonic loss and oscillatory dynamics, this regime frequently leads to faster convergence and better generalization. Existing theoretical perspectives offer complementary but incomplete insights: one identifies a self-stabilization mechanism that drives GD toward flat regions but does not yield convergence rates; another derives rigorous rates for specific losses but does not explain the underlying stabilization mechanism; and a third unifies these views for overparameterized least squares, but relies on the existence of a minimizer manifold, which is absent in classification settings. In this work, we extend this unified geometric perspective to exponential-tail losses, including logistic and cross-entropy losses in separable linea
    
[^141]: 争分夺秒：教会视觉语言导航模型进行预算条件下的目标搜索

    Racing the Clock: Teaching VLNs Budget-Conditioned Object Search

    [https://arxiv.org/abs/2611.17442](https://arxiv.org/abs/2611.17442)

    该论文提出“预算条件下的目标搜索”新任务，使视觉语言导航智能体能够根据明确的动作预算自适应调整搜索策略，并对比了基于特权信息的TSP专家与基于部分可观测和跨场景语义先验的房间级Bellman专家两种监督方案。

    

    当救援机器人只有三十秒的时间进入一栋建筑时，它应该冲向最有希望的那一个房间；而如果有十分钟，它则应该进行更广泛的探索。然而，视觉语言导航（VLN）模型通常在固定的回合限制下进行评估，而策略本身从未观测到这一限制，尽管时间紧迫的任务要求行为能够根据可用时间预算进行调整。我们研究预算条件下的目标搜索任务：给定一个目标类别和一个明确的动作预算，智能体必须在预算耗尽之前找到尽可能多的不同目标实例。我们通过对比两种专家来研究如何为此任务监督VLN模型：一种是特权TSP专家，它利用完整地图和真实目标位置规划接近最优的路线；另一种是房间级Bellman专家，它在部分可观测条件下，利用高层跨场景语义先验来最大化期望回报。尽管TSP专家取得了显著更高的期望……（原文摘要在此处截断）

    arXiv:2611.17442v1 Announce Type: cross  Abstract: A rescue robot entering a building with thirty seconds on the clock should sprint toward the single most promising room; given ten minutes, it should instead explore a broader tour. Yet vision-language-navigation (VLN) models are typically evaluated under a fixed episode limit that the policy never observes, even though time-critical missions require behavior to adapt to the available budget. We study budget-conditioned object search: given a target category and an explicit action budget, an agent must find as many distinct target instances as possible before the budget expires. We investigate how to supervise VLNs for this task by comparing a privileged TSP expert, which plans near-optimal routes using the full map and true target locations, with a room-level Bellman expert that maximizes expected return under partial observability using high-level cross-scene semantic priors. Despite the TSP expert achieving substantially higher expe
    
[^142]: 通过行为轨迹解读编码智能体

    Interpreting Coding Agents Through Their Behavioral Trajectories

    [https://arxiv.org/abs/2611.17440](https://arxiv.org/abs/2611.17440)

    该论文提出一种基于规则的解析器，将编码智能体的原始执行轨迹映射为高级行为，并通过对14万余条轨迹的行为画像分析，揭示解决率相近的智能体在探索、修改和验证代码的方式上存在显著差异，且这些行为画像本身足以在不同仓库与语言间识别出对应的智能体技术栈。

    

    标准的编码智能体基准测试主要评估二元化的任务成功与否，将执行过程视为黑箱，掩盖了产生这些结果的行为。我们分析了来自125种智能体配置、涵盖117个不同模型的大型语料库，共143,209条执行轨迹。为了能在异构的智能体接口之间进行比较，我们开发了一个基于规则的解析器，将原始执行痕迹映射为仓库搜索、文件编辑、测试执行等高级动作，并针对500个事件以六模型共识参考对其预测进行了审计。我们发现，解决率相近的智能体在探索、修改和验证代码的方式上仍可能存在显著差异。此外，在不使用任务结果、token数量或工具名称作为特征的情况下，汇总的行为画像能够在不相交的仓库集合和不同编程语言中检索出对应的智能体技术栈，尽管任务构成尚未被完全……（摘要在此处截断）

    arXiv:2611.17440v1 Announce Type: cross  Abstract: Standard coding-agent benchmarks primarily evaluate binary task success, treating execution as a black box and obscuring the behaviors that produce those outcomes. We analyze a large corpus of 143,209 execution trajectories from 125 agent configurations spanning 117 unique models. To enable comparisons across heterogeneous agent interfaces, we develop a rule-based parser that maps raw execution traces to high-level actions such as repository search, file editing, and test execution, and audit its predictions against a six-model consensus reference on 500 events. We find that agents with similar solve rates can nevertheless differ substantially in how they explore, modify and verify code. Moreover, without using task outcomes, token counts, or tool names as features, aggregate behavioral profiles retrieve the corresponding agent stack across disjoint repository sets and programming languages, although task composition is not fully contr
    
[^143]: WETW：学习针对特定序列对的加权欧几里得距离以实现高效DTW搜索

    WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search

    [https://arxiv.org/abs/2611.17412](https://arxiv.org/abs/2611.17412)

    提出WETW学习型距离，利用小型对称网络为每对序列的每个嵌入维度预测权重，实现针对特定序列对的自适应比较且无需计算对齐，将误差相比最佳静态距离减半，从而实现高效的DTW相似性搜索。

    

    动态时间规整（DTW）是对时间上错位的时间序列进行相似性搜索的标准距离度量，但其计算成本与序列长度呈二次方关系，且下界通常只能剪枝掉很少的候选序列，因此在大规模长序列集合上的搜索速度很慢。学习型近似方法将序列嵌入到向量空间以降低比较成本，但它们对每一对序列都使用相同的固定规则进行比较（通常是欧几里得距离），而实际上DTW对每一对序列的对齐方式都不相同。我们观察到，一旦一对序列被对齐，DTW实际上就是在为该对序列所选坐标上计算的欧几里得距离。基于这一观点，我们提出了加权欧几里得时间规整（WETW），这是一种学习型距离，能够针对每一对序列自适应地调整固定长度嵌入的比较方式。一个小型对称网络在无需计算对齐的情况下，为每个嵌入维度和每对序列预测一个权重，这使得在同一嵌入上的误差相比最佳静态距离减半。这些嵌入来自W……

    arXiv:2611.17412v1 Announce Type: cross  Abstract: Dynamic Time Warping (DTW) is the standard distance for similarity search over time series that are misaligned in time, but its cost is quadratic in the series length and lower bounds often prune few candidates, so search over large collections of long series is slow. Learned approximations embed the series to make comparisons cheap, yet compare every pair with the same fixed rule, usually the Euclidean distance, although DTW aligns each pair differently. We observe that, once a pair is aligned, DTW is a Euclidean distance over coordinates selected for that pair. Building on this view, we propose Weighted Euclidean Time Warping (WETW), a learned distance that adapts the comparison of fixed-length embeddings to each pair. A small symmetric network predicts one weight per embedding di- mension and pair without computing an alignment, which halves the error of the best static distance on the same embeddings. The embeddings come from the W
    
[^144]: 大语言模型能编写后门吗？关于权重空间编程的实证安全研究

    Can LLMs Write Backdoors? An Empirical Security Study of Weight-Space Programming

    [https://arxiv.org/abs/2611.17395](https://arxiv.org/abs/2611.17395)

    本文通过实证安全研究首次系统评估了大语言模型在权重空间编程中的后门注入能力，表明通用大语言模型可以在不访问任务数据、不执行前向传播、不计算梯度的情况下，仅凭自然语言指令直接修改模型权重来注入后门，揭示了权重空间编程带来的新型安全威胁。

    

    权重空间编程的最新进展表明，大语言模型（LLM）可以在不使用传统基于梯度训练的情况下，直接检查和修改神经网络的权重。虽然这一能力为模型改进开辟了新的可能性，但也引出了一个重要的安全问题：大语言模型能否利用权重空间编程向神经网络注入后门？为回答这一问题，我们对基于大语言模型的权重空间编程后门注入开展了一项实证安全研究。具体而言，我们研究通用大语言模型能否仅根据自然语言指令直接修改目标模型的权重，而无需访问任务特定数据、执行前向传播或计算梯度。我们评估了五个大语言模型作为权重编辑器，针对三种目标架构进行实验：MNIST 上的 MLP、CIFAR-10 上的 ResNet-18，以及 AG News 上的 Qwen2.5-0.5B-Instruct。我们的实验揭示了……

    arXiv:2611.17395v1 Announce Type: cross  Abstract: Recent advances in weight-space programming (WSP) have demonstrated that large language models (LLMs) can directly inspect and modify neural network weights without conventional gradient-based training. While this capability opens new possibilities for model improvement, it also raises an important security question: Can LLMs exploit weight-space programming to inject backdoors into neural networks? To answer this question, we present an empirical security study of LLM-driven weight-space programming for backdoor injection. Specifically, we investigate whether general-purpose LLMs can directly modify the weights of target models according to natural-language instructions, without accessing task-specific data, performing forward passes, or computing gradients. We evaluate five LLMs as weight editors across three target architectures: an MLP on MNIST, ResNet-18 on CIFAR-10, and Qwen2.5-0.5B-Instruct on AG News. Our experiments reveal thr
    
[^145]: 枚举后验支撑先验：有限积分跨范畴测量态射的一阶误差

    Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms

    [https://arxiv.org/abs/2611.17390](https://arxiv.org/abs/2611.17390)

    提出枚举后验支撑先验框架，基于 Quot->MorTop->Set 三层范畴链显式记录底层谓词支撑、可逆性与误差所在层级，解决标准测度论 a.e. 约定下零测度支撑隐藏域外误差无法被显式计入的问题。

    

    在生成建模中，标准测度论采用几乎处处（a.e.）约定，这合理地忽略了由零测度支撑集所携带的任何误差分量：单个零测度元素可以将域外误差隐藏于零支撑内部，而永远不会被显式计入。本文的贡献不是提出一种新的分布距离，而是一种可验证的结构，该结构明确写出了哪个底层谓词支撑着高层对象、它在何时可逆、以及误差驻留在哪一层，由此我们提出了枚举后验支撑先验框架。该框架运行于三层范畴链 Quot->MorTop->Set 之上：前两层不假设任何点元素，并以二元谓词商划分作为对象；副本索引计数测度将一对多集合依次区分为单射，把对象收集从满射提升为商层双射；互斥……

    arXiv:2611.17390v1 Announce Type: cross  Abstract: In generative modeling, standard measure theory adopts the a.e. convention, which legitimately ignores any error component carried by a null-support set: a single null element can hide out-of-domain error inside null support and never be explicitly accounted for. The contribution of this paper is not a new distributional distance, but rather a verifiable structure that writes down which underlying predicate supports a high-level object, when it is invertible, and at which layer the error resides, from which we propose the enumerating posterior-supported prior framework. The framework operates on a three-layer categorical chain Quot->MorTop->Set: the first two layers posit no point elements and take binary-predicate quotient partitions as objects; the replica-index counting measure successively discriminates one-to-many collections into injections, lifting object collection from surjection to a quotient-layer bijection; the mutual exclu
    
[^146]: 基于自监督神经回归模型增强的数字核仪器波形脉冲定时方法

    Waveform-based pulse timing enhanced by self-supervised neural regression models for digital nuclear instrumentation

    [https://arxiv.org/abs/2611.17356](https://arxiv.org/abs/2611.17356)

    本文提出一种基于匹配波形对的、结合自监督损失与辅助弱监督损失的神经回归新范式，显著提升了数字核仪器中波形脉冲定时的精度。

    

    随着高采样率数字核设备的快速发展，其在核设施及相关领域的应用日益普及，并凭借波形采样系统的通用性和灵活性有望得到广泛推广。当数字化仪达到数百兆赫兹或吉赫兹级别时，其时间测量下限可深入至亚纳秒区域，加速了此类设备在众多时间敏感场景中的应用。尽管传统算法结合数字采样技术已在特定条件下实现了较为理想的脉冲定时，但紧凑高效的机器学习模型的潜力仍有待挖掘。本文旨在通过一种基于匹配波形对自监督学习的新范式来提升脉冲定时性能。所提出的方法将主流的自监督损失与参考集上的辅助弱监督损失相结合，共同提升了……（原文摘要在此处截断）

    arXiv:2611.17356v1 Announce Type: cross  Abstract: With the rapid development of high-sampling-rate digital nuclear devices, their applications in nuclear facilities and related fields become more acceptable, and will possibly become widespread due to the versatility and flexibility of waveform sampling systems. When digitizers reach hundreds of megahertz or gigahertz, their lower limit in time measurement probes into the sub-nanosecond region, expediting their use in a bunch of time-critical scenarios. While conventional algorithms coupled with digital sampling techniques realize fair pulse timing in certain conditions, the potentials are still open to compact and efficient machine learning models. In this paper, we intend to boost the performance of pulse timing with a new paradigm of self-supervised learning on matched waveform pairs. The proposed method combines the mainstream self-supervised loss and the auxiliary weakly-supervised loss on the reference set, jointly improving the 
    
[^147]: 用于模拟量能器簇射中样本高效能量回归的混合量子-经典序列模型

    Hybrid Quantum-Classical Sequence Models for Sample-Efficient Energy Regression in Simulated Calorimeter Showers

    [https://arxiv.org/abs/2611.17354](https://arxiv.org/abs/2611.17354)

    该论文在CaloGAN基准数据集上系统比较了八种经典与量子增强架构在正电子能量重建任务中的表现，发现基于纵向簇射序列的混合量子-经典模型结合物理信息驱动的数据表示达到了最高精度，证明了数据表示选择对能量回归的关键作用。

    

    从电磁量能器数据中进行能量重建是高能物理中粒子鉴别、事例重建和精密测量的基础。机器学习在这项任务上已经表现良好，但仍有两个问题悬而未决：数据表示的选择究竟有多重要，以及混合量子-经典模型在实践中是否真的带来价值。我们在CaloGAN数据集上比较了经典模型与量子增强模型用于入射正电子能量重建，该数据集是一个包含100,000个模拟电磁簇射事例的公开基准，能量范围跨越1至100 GeV。共八种架构，即SVR、ANN、CNN和LSTM及其量子对应版本，在三种量能器数据表示上进行训练：工程化物理特征、量能器图像以及纵向簇射序列。追踪簇射纵向发展的模型达到了最高的准确率，这凸显了物理信息驱动的数据表示的价值。（注：原始摘要在此处被截断）

    arXiv:2611.17354v1 Announce Type: cross  Abstract: Energy reconstruction from electromagnetic calorimeter data underpins particle identification, event reconstruction and precision measurements in high-energy physics. Machine learning already performs well on this task, yet two questions remain open: how much the choice of data representation matters, and whether hybrid quantum-classical models add value in practice. We compare classical and quantum-enhanced models for incident positron energy reconstruction on the CaloGAN dataset, a public benchmark of 100,000 simulated electromagnetic shower events spanning 1 to 100 GeV. Eight architectures, namely SVR, ANN, CNN and LSTM together with their quantum counterparts, are trained on three calorimeter representations: engineered physical features, calorimeter images, and longitudinal shower sequences. Models that follow the longitudinal development of the shower reach the highest accuracy, which points to the value of physically informed re
    
[^148]: SAFARI-1研究反应堆中子注量率测量剖面的无监督异常检测

    Unsupervised Anomaly Detection in Measured Neutron Flux Profiles of the SAFARI-1 Research Reactor

    [https://arxiv.org/abs/2611.17351](https://arxiv.org/abs/2611.17351)

    该论文提出了一种基于无监督学习的自动化异常检测方法，通过高阶多项式拟合提取轴向注量率剖面的形状特征并结合密度聚类，有效识别SAFARI-1研究反应堆中因铜丝插入不当导致的异常中子注量率测量数据。

    

    掌握中子注量率分布对于反应堆的安全和高效利用至关重要。在SAFARI-1研究反应堆中，通过活化沿燃料组件或控制组件高度插入的天然铜丝，并用碘化钠探测器测量其活度，从而周期性地获得一组轴向注量率剖面。然而，铜丝插入不当可能导致所测剖面无法准确反映实际的中子注量率分布。在先前识别出的铜丝插入不当潜在指标的基础上，本研究开发了一种基于无监督学习的自动化方法，用于检测异常的轴向注量率测量数据。测量数据首先经过缺失值填补、衰变校正和归一化等预处理，随后采用高阶多项式拟合来表征轴向注量率剖面并提取描述其形状的特征，再利用基于密度的空间聚类方法进行异常检测。

    arXiv:2611.17351v1 Announce Type: cross  Abstract: Knowledge of the neutron flux distribution is essential for safe and effective reactor utilisation. In the SAFARI-1 research reactor, a set of axial flux profiles are periodically measured by activating natural copper wires inserted along the height of the fuel or control assemblies and subsequently measuring their activity using a sodium iodide detector. Improper wire insertion can, however, result in profiles that do not accurately represent the actual neutron flux distribution. Building on previously identified potential indicators of improper wire insertion, an automated approach based on unsupervised learning was developed to detect anomalous axial flux measurements. The measured data were preprocessed using missing-value imputation, decay correction, and normalisation. High-order polynomial fitting was then applied to characterise the axial flux profiles and derive features describing their shape. The Density-Based Spatial Cluste
    
[^149]: 面向HL-LHC低$p_T$径迹重建的曲率感知边缘动态图神经网络

    Curvature-Aware Edge Dynamic GNNs for Low-$p_T$ Track Reconstruction at HL-LHC

    [https://arxiv.org/abs/2611.17343](https://arxiv.org/abs/2611.17343)

    提出曲率感知的流形螺旋交互网络（Manifold Helical-IN），通过将探测器空间点嵌入螺旋流形来“展开”低动量粒子的卷曲轨迹，在TrackML数据集的 $p_T > 0.5$ GeV 关键区域实现了0.9675的高径迹重建效率。

    

    高能对撞机实验中的粒子径迹重建是一个计算量巨大的组合问题，传统上采用卡尔曼滤波器求解，但随着探测器占据率的增加，其扩展性能较差。图神经网络（GNN）近年来作为一种有前景的替代方案而兴起。在这项工作中，我们提出了流形螺旋网络（Manifold Helical-IN），这是一种新颖的交互网络架构，它利用曲率感知特征将探测器空间点显式地嵌入到螺旋流形图册上。这种几何投影在数学上“展开”了低动量粒子紧密卷曲的轨迹，从而使网络能够以良好的精度区分真实的物理径迹与组合噪声。在TrackML挑战赛数据集上的评估表明，所提出的流形螺旋网络在实验上至关重要的 $p_T > 0.5$ GeV 区域实现了0.9675的重建效率，接近近期报道的基于Transformer的方法。

    arXiv:2611.17343v1 Announce Type: cross  Abstract: Particle track reconstruction in high-energy collider experiments is a computationally demanding combinatorial problem, traditionally solved by Kalman filters that scale poorly with increasing detector occupancy. Graph Neural Networks (GNNs) have recently emerged as a promising alternative. In this work, we introduce the Manifold Helical-IN, a novel Interaction Network architecture that explicitly embeds detector spacepoints onto a helical manifold chart using curvature-aware features. This geometric projection mathematically unrolls the tightly curling trajectories of low-momentum particles. Consequently, the network can distinguish true physical tracks from combinatorial noise with good precision. Evaluated on the TrackML challenge dataset, the proposed Manifold Helical-IN achieves a reconstruction efficiency of 0.9675 in the experimentally critical $p_T > 0.5$ GeV regime, approaching recently reported Transformer-based approaches. C
    
[^150]: ArchSpace：记录、评审与引用大语言模型架构研究

    ArchSpace: Recording, Reviewing, and Referencing LLM Architecture Research

    [https://arxiv.org/abs/2611.17332](https://arxiv.org/abs/2611.17332)

    ArchSpace 是一个公共平台，通过记录实验与推理、社区评审和后续引用来系统化大语言模型架构研究，并借助缩放定律在昂贵的规模化训练前预测模型表现，从而同时弥补工业界缺乏创新公开记录和学术界评估受限的两大缺口。

    

    arXiv:2611.17332v2 公告类型：替换 摘要：模型架构决定了大语言模型（LLM）的扩展行为、训练效率和推理成本，使其成为工业界和学术界共同关注的核心焦点。在工业界，DeepSeek、Qwen 和 Kimi 背后的团队持续推出架构创新，但公开报告很少记录哪些想法成功了、哪些被放弃以及为什么。在学术界，有限的算力和不一致的基线往往将评估限制在小规模、单点比较上，难以确定这些改进是否在更大规模下依然成立或能产生更广泛的影响。ArchSpace 作为记录、评审与引用架构研究的公共平台，同时弥补了这两个缺口：每个提案都会连同其实验和推理过程被记录下来，由社区进行评审，并被后续工作引用。为了在昂贵的规模化训练之前评估想法，ArchSpace 利用缩放定律来预测模型在更大规模下的损失……

    arXiv:2611.17332v2 Announce Type: replace  Abstract: Model architecture shapes the scaling behavior, training efficiency, and inference cost of LLMs, making it a central focus of both industry and academia. In industry, teams such as those behind DeepSeek, Qwen, and Kimi routinely introduce architectural innovations, yet public reports rarely document which ideas succeeded, which were abandoned, and why. In academia, limited compute and inconsistent baselines often restrict evaluations to small-scale, single-point comparisons, making it difficult to establish whether improvements hold at larger scales or translate into broader impact. ArchSpace addresses both gaps as a public platform for Recording, Reviewing, and Referencing Architecture Research: each proposal is recorded with its experiments and reasoning, reviewed by the community, and referenced by later work. To evaluate ideas before costly large-scale training, ArchSpace leverages scaling laws to predict model loss at larger sca
    
[^151]: 针对逻辑回归投毒攻击的凸分析重构方法

    A Convex-Analytic Reformulation for Poisoning Attacks on Logistic Regression

    [https://arxiv.org/abs/2611.17143](https://arxiv.org/abs/2611.17143)

    本文将针对带 ℓ2 正则化逻辑回归的投毒攻击问题重构为一种凸分析形式，使有界双线性等式成为唯一的非凸约束，并为所有变量推导出有限界。

    

    我们研究了带 $\ell_2$ 正则化的逻辑回归上的数据投毒攻击问题，其中攻击者在标签被预先指定的情况下，选择固定数量注入观测样本的特征，以最大化干净训练数据上的逻辑损失。由于防御者问题具有强凸性，我们可以通过其一阶最优性条件对其进行精确重构。利用 softplus 函数的凸共轭性，我们用单一的凸最优性间隙不等式替代了 sigmoid 关系，并将攻击者的目标表示为凹函数。通过变量替换，平稳性条件变为仿射形式，使得有界的双线性等式成为唯一的非凸约束。我们推导出这些等式中所有涉及变量的有限界，包括系数范数的全局界，以及通过基于凸优化的预处理获得的更紧的界。在八个二分类数据集上的实验评估了……

    arXiv:2611.17143v2 Announce Type: replace  Abstract: We study data poisoning attacks on logistic regression with $\ell_2$-regularization, where an attacker chooses the features of a fixed number of injected observations with prescribed labels to maximize the logistic loss on the clean training data. Strong convexity of the defender's problem yields an exact reformulation through its first-order optimality conditions. Using convex conjugacy of the softplus function, we replace the sigmoid relations by a single convex optimality-gap inequality and express the attacker's objective as a concave function. A change of variables makes the stationarity condition affine, leaving bounded bilinear equalities as the only nonconvex constraints. We derive finite bounds on all variables involved in these equalities, including a global bound on the coefficient norm and tighter bounds obtained by preprocessing based on convex optimization. Experiments on eight binary classification datasets evaluate at
    
[^152]: LazyWeights：按需解码的边缘大语言模型

    LazyWeights: Decode-on-Demand Edge LLMs

    [https://arxiv.org/abs/2611.17108](https://arxiv.org/abs/2611.17108)

    LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。

    

    内存决定了边缘设备能运行哪些大语言模型以及运行得多快：在4 GB的Jetson Nano上受容量限制，在8 GB的Jetson Orin Nano Super上则受带宽限制。现有部署格式（GPTQ、AWQ、llama.cpp的K量化和I量化）对矩阵的每个整数编码分配相同的位数，尽管这些编码的分布远非均匀。LazyWeights保留了这些格式，但通过调整GPTQ使编码更易压缩，在交错通道中对编码进行无损Huffman编码，并在DRAM与L2缓存之间的一个小型单元中按需解码，因此GPU内核无需任何改动，DRAM只需传输编码后的字节。在1%和3.5%的困惑度预算下，编码后的Phi-3-mini、Mistral-7B和SmolLM-1.7B比最佳的llama.cpp格式小5.7%-31.4%。一个位精确的0.21 mm² ASIC解码单元（1 GHz，7 nm工艺）将Orin上每个token的延迟和能耗最多降低36%和38%（结合prompt-lookup投机解码时可降低54%和44%）。在Nano上，LazyWeights每token所需时间比CPU版llama.cpp减少58%-64%，能耗减少53%-57%，并能运行……

    arXiv:2611.17108v2 Announce Type: replace-cross  Abstract: Memory decides which LLMs an edge device runs, and how fast: capacity on a 4 GB Jetson Nano, bandwidth on an 8 GB Jetson Orin Nano Super. Deployed formats (GPTQ, AWQ, llama.cpp's K- and I-quants) spend equal bits on every integer code of a matrix, although the codes are far from uniform. LazyWeights keeps these formats but tunes GPTQ so the codes compress, Huffman-codes them losslessly in interleaved lanes, and decodes them on demand in a small unit between DRAM and L2, so GPU kernels run unchanged and DRAM moves only coded bytes. At 1% and 3.5% perplexity budgets, coded Phi-3-mini, Mistral-7B and SmolLM-1.7B are 5.7-31.4% smaller than the best llama.cpp format. A bit-exact 0.21 mm^2 ASIC decoder unit (1 GHz, 7 nm) cuts Orin latency and energy per token by up to 36% and 38% (54% and 44% with prompt-lookup speculation). On the Nano, LazyWeights needs 58-64% less time and 53-57% less energy per token than CPU llama.cpp, and runs 
    
[^153]: 用于熔盐快堆热工水力学的具有迭代求解器耦合的不变性神经网络涡粘性闭合模型

    An Invariant Neural-Network Eddy-Viscosity Closure with Iterative Solver Coupling for Molten Salt Fast Reactor Thermal-Hydraulics

    [https://arxiv.org/abs/2611.17067](https://arxiv.org/abs/2611.17067)

    本文为SAM系统热工水力程序开发了一种对平移、旋转、反射等变换严格不变的神经网络涡粘性闭合模型，无需求解湍流输运方程即可预测非负涡粘性并保持动量方程良好条件性，将其应用于熔盐快堆以捕捉传统混合长度模型无法表征的非平衡湍流与大规模回流。

    

    系统级热工水力分析程序（如系统分析模块SAM）在粗网格上采用代数零方程混合长度模型来闭合湍流。这些模型无法表征开放腔体式先进反应堆中的非平衡湍流、流动分离和大尺度回流现象。我们为SAM开发了一种不变性神经网络涡粘性闭合模型，并将其应用于熔盐快堆（MSFR）。该闭合模型根据邻近单元的流动与几何特征预测每个单元的湍动能和时间尺度，而无需求解湍流输运方程。由此得到的非负涡粘性进入隐式扩散算子，从而保持动量方程的良好条件性。我们证明了该闭合模型对平移、伽利略变换、旋转、反射以及邻居排序具有严格的不变性。在高阶Nek5000流场上训练得到的网络面对不同的特征分布（摘要原文在此处截断）。

    arXiv:2611.17067v2 Announce Type: replace-cross  Abstract: System thermal-hydraulics codes such as the System Analysis Module (SAM) close turbulence on coarse meshes with algebraic zero-equation mixing-length models. These models cannot represent the non-equilibrium turbulence, flow separation, and large-scale recirculation of open-cavity advanced reactors. We develop an invariant neural-network eddy-viscosity closure for SAM and apply it to the Molten Salt Fast Reactor (MSFR). The closure predicts the turbulent kinetic energy and timescale of each element from the flow and geometry of its neighbors, without solving turbulence transport equations. The resulting non-negative eddy viscosity enters the implicit diffusive operator, which keeps the momentum equations well conditioned. We prove that the closure is exactly invariant to translation, Galilean boost, rotation, reflection, and neighbor ordering. A network trained on high-order Nek5000 fields sees a different feature distribution 
    
[^154]: 并行化让时间飞逝：基于线性递归的帕累托高效时间序列基础模型

    Time Flies When You Parallelise: Pareto-efficient Time Series Foundation Models with Linear Recurrences

    [https://arxiv.org/abs/2611.16934](https://arxiv.org/abs/2611.16934)

    提出基于 Mamba-3 线性递归的时间序列基础模型家族 ci-proto，实现训练时的时间并行与线性推理成本，在 fev-bench 100 项预测任务上创下新的最先进纪录，并凭借五个变体覆盖完整的精度-效率帕累托前沿。

    

    时间序列基础模型已经展现出强大的零样本预测性能，但计算效率仍然是其架构上的核心挑战。时间维度上的自注意力机制会随时间词元数量的增加而产生二次方成本，而非线性的递归状态转移则限制了沿时间维度的并行计算。我们提出了 ci-proto，这是一个基于 Mamba-3 的时间序列基础模型家族，其线性状态递归支持训练时按块的时间并行计算，且推理成本随序列长度线性扩展。该架构将递归式时间处理与跨变量注意力交替使用，支持结合仅历史可见协变量与未来已知协变量的多变量概率预测。在 fev-bench 的 100 项预测任务上，ci-proto XL 在时间序列基础模型中树立了新的最先进水平，五个 ci-proto 变体也共同构成了完整的帕累托前沿。

    arXiv:2611.16934v2 Announce Type: replace  Abstract: Time series foundation models have demonstrated strong zero-shot forecasting performance, but computational efficiency remains a central architectural challenge. Temporal self-attention incurs quadratic cost in the number of temporal tokens, while nonlinear recurrent state transitions restrict parallel computation along the temporal dimension. We introduce ci-proto, a family of time series foundation models based on Mamba-3, whose linear state recurrences support chunk-wise parallel-in-time computation during training, and inference costs that scale linearly with sequence length. The architecture alternates recurrent temporal processing with cross-variate attention, supporting multivariate probabilistic forecasting with past-only and future-known covariates. Across fev-bench's 100 forecasting tasks, ci-proto XL sets a new state of the art among time series foundation models. The five ci-proto variants also make up the entire Pareto f
    
[^155]: 检测并非预警：评估针对语言模型失效的Token级警报

    Detection Is Not Early Warning: Evaluating Token-Level Alarms on Language Model Failures

    [https://arxiv.org/abs/2611.16803](https://arxiv.org/abs/2611.16803)

    该论文提出了PREWARN评估协议，将监控器的评估从单纯检测失效转向在失效token起点之前发出预警，从而为停止或干预生成过程争取时间。

    

    许多语言模型的失效始于某个特定的token，即“起点”。一个在生成过程中读取模型的监控器可以对已完成的响应进行分类、在失效进行中检测到它，或者在它开始之前发出预警——只有预警才能留出时间来停止或引导生成过程。监控器通常使用响应级指标（如AUROC）进行评估，这些指标忽略了监控器何时触发。我们提出了PREWARN，这是一个适用于任何具有标注起点的失效的评估协议。它根据验证数据将每个监控器的阈值校准到应用所需的错误率，运行监控器，并就三项任务相对于起点对其警报进行评分。作为案例研究，我们将其应用于三个开源语言模型的重复性退化问题，以早期预警为主要任务，使用基于激活的线性探针和六个未训练的基线。响应级指标在这种失效上已经饱和，而一个仅读取响应……（原文摘要在此处截断）

    arXiv:2611.16803v2 Announce Type: replace-cross  Abstract: Many failures of language models begin at a specific token, the onset. A monitor that reads the model during generation can classify the finished response, detect the failure once it is under way, or warn before it begins, and only a warning leaves time to stop or steer the generation. Monitors are usually evaluated with response-level metrics such as AUROC, which ignore when the monitor fires. We introduce PREWARN, an evaluation protocol for any failure with an annotated onset. It calibrates each monitor's threshold on validation data to an error rate chosen for the application, runs the monitor, and scores its alarms relative to the onset on all three tasks. As a case study we apply it to repetitive degeneration in three open language models, with early warning as the main task, using linear probes on activations and six untrained baselines. Response-level metrics saturate on this failure, and a rule that reads only the respo
    
[^156]: AI风险行为基准测试

    Benchmarking AI risk behaviors

    [https://arxiv.org/abs/2611.16318](https://arxiv.org/abs/2611.16318)

    该论文提出了一个评估LLM风险倾向的工具包，发现2026年最先进模型因更倾向于采用期望值最大化作为决策程序，而比旧模型更稳定、更一致且更趋于风险中性。

    

    我们在一系列情境和任务需求下评估了大语言模型（LLM）的风险倾向。我们评估了模型在各种条件下对风险的厌恶或容忍程度，以及其行为在不同条件下的稳定性和一致性。我们提出了一个用于构建模型风险画像的工具包，该工具包可用于预测模型行为，并可扩展纳入新的风险倾向评估。我们还考察了这些特性在不同模型代际和模型家族之间的变化。我们观察到，2026年的最先进模型往往比旧模型更稳定、更一致，且更趋于风险中性。这些趋势似乎是相互关联的：较新的模型之所以更稳定、更一致，是因为它们更倾向于将期望值最大化作为决策程序。

    arXiv:2611.16318v1 Announce Type: new  Abstract: We evaluate LLM risk propensities across a range of contexts and task demands. We evaluate models for how risk averse or risk tolerant they are in various conditions, as well as how stable and consistent their behavior is across conditions. We present a toolkit for developing model risk profiles that can be used to predict model behavior and which can be adapted to include new evaluations of risk propensities. We also examine how these properties have changed across model generations and families. We observe that 2026 state-of-the-art models tend to be more stable, more consistent, and more risk neutral than older models. These trends appear to be related; newer models are more stable and consistent because they are more likely to use expected value maximization as a decision procedure.
    
[^157]: 整流循环Transformer

    Rectified Looped Transformer

    [https://arxiv.org/abs/2611.16265](https://arxiv.org/abs/2611.16265)

    提出整流循环Transformer（RLT），通过受流模型启发的插值调度重新设计循环间反馈，在理论上保证隐藏状态幅度有界并降低扰动放大，从而提升参数高效的语言建模能力。

    

    循环Transformer通过重复应用共享模块来增加有效深度，为语言建模提供了一种参数高效的方法。然而，大多数现有工作直接将每个循环的输出作为下一个循环的输入，这会随着递归加深而导致隐藏状态规模增长和扰动放大。我们提出整流循环Transformer（RLT），该模型受基于流的建模启发，使用插值调度重新设计了循环之间的反馈机制。在每个循环中，RLT通过在当前状态与其预测之间进行线性插值来构造下一个输入。这种插值将状态变化分布到各个循环中，促使从初始输入到最终输出的轨迹更加平滑。理论分析证明，在适当的假设下，RLT能使状态幅度在各个循环中保持一致有界，并降低最坏情况下的扰动放大。我们在语言建模语料库上对RLT进行了评估。

    arXiv:2611.16265v1 Announce Type: cross  Abstract: Looped Transformers increase effective depth by repeatedly applying a shared block, offering a parameter-efficient approach to language modeling. However, most existing works directly reuse each loop's output as the next's input, which can lead to hidden-state scale growth and perturbation amplification as recurrence deepens. We propose Rectified Looped Transformer (RLT), which redesigns feedback between loops using an interpolation schedule inspired by flow-based modeling. At each loop, RLT forms the next input by linearly interpolating between the current state and its prediction. Such interpolation distributes state changes across loops, encouraging a smoother trajectory from the initial input to the final output. Theoretical analysis proves that, under appropriate assumptions, RLT keeps state magnitudes uniformly bounded across loops and reduces the worst-case perturbation amplification. We evaluate RLT on language modeling corpora
    
[^158]: JudgeMix：面向大语言模型智能体的裁判引导式混合采样数据合成

    JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents

    [https://arxiv.org/abs/2611.16174](https://arxiv.org/abs/2611.16174)

    提出JudgeMix框架，利用裁判模型对每一步进行语义评估，动态决定目标模型与专家模型的交替介入时机，从而生成与目标模型自身分布匹配、且不覆盖其已正确动作的高质量SFT训练数据。

    

    在专家轨迹上进行监督微调（SFT）是训练大语言模型智能体的标准范式。然而，这种方法存在众所周知的训练-推理分布不匹配问题，因此需要目标感知的数据合成策略，将训练数据锚定在目标模型的固有状态分布上。为弥合这一差距，新兴方法引入专家模型，要么与目标模型一同引导混合采样，要么提供密集的逐步监督。尽管如此，现有方法在应用专家辅助时忽视了目标模型的固有行为：它们要么使用次优的随机干预启发式方法破坏采样过程，要么强制实施不加区分的统一监督，不必要地覆盖模型本已正确的动作。我们提出JudgeMix，一个由裁判引导的数据合成框架，它基于逐步语义评估，动态地交替使用目标模型和专家模型来生成训练数据。

    arXiv:2611.16174v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) on expert trajectories is a standard paradigm for training LLM agents. However, it suffers from a well-known training-inference distribution mismatch, calling for target-aware data synthesis strategies that ground the training data in the target model's native state distribution. To bridge this gap, emerging approaches incorporate the expert either to guide mixed rollouts alongside the target model, or to provide dense step-wise supervision. Nonetheless, existing methods apply expert assistance blind to the target model's inherent behavior: they either disrupt the rollout using sub-optimal random intervention heuristics, or they enforce indiscriminate uniform supervision that unnecessarily overwrites the model's already-correct actions. We introduce JudgeMix, a judge-guided data synthesis framework that dynamically interleaves the target model and expert based on step-wise semantic evaluations. JudgeMix que
    
[^159]: CastWorld：超越预测精度的上下文感知智能体时间序列预测基准测试

    CastWorld: Benchmarking Context-Aware Agentic Time Series Forecasting Beyond Predictive Accuracy

    [https://arxiv.org/abs/2611.15262](https://arxiv.org/abs/2611.15262)

    CastWorld是一个面向上下文感知智能体时间序列预测的基准测试，提供了涵盖8个领域25个真实数据集的丰富上下文数据套件，以及包含49个核心工具和状态自适应循环的预测原生框架，实现了对预测智能体超越单纯预测精度的系统化评估。

    

    智能体时间序列预测将传统方法从一次性数值预测扩展到自主的、上下文感知的交互。然而，现有的基准测试在多样化上下文信息、预测特定的自适应交互以及预测智能体的系统化评估方面所提供的集成支持较为有限。本文提出了CastWorld，一个面向上下文感知时间序列预测的智能体基准测试。首先，其丰富的上下文数据集套件涵盖8个领域和25个真实世界数据集，在统一的上下文规范下包含外部变量、事件和历史案例。其次，一个轻量级、模块化的预测原生框架提供了49个核心工具和一个状态自适应循环，该循环跟踪预测版本，支持迭代式的证据收集和预测修订。中间预测可以暴露信息缺口，而新获取的证据则指导后续的修订和最终版本的生成。

    arXiv:2611.15262v1 Announce Type: cross  Abstract: Agentic time series forecasting extends conventional approaches beyond one-shot numerical prediction toward autonomous, context-aware interaction. However, existing benchmarks offer limited integrated support for diverse contextual information, forecasting-specific adaptive interactions, and systematic evaluation of forecasting agents. In this paper, we introduce CastWorld, an agentic benchmark for context-aware time series forecasting. First, its context-rich dataset suite spans 8 domains and 25 real-world datasets, covering external variables, events, and historical cases under a unified context specification. Second, a lightweight, modular forecasting-native harness provides 49 core tools and a state-adaptive loop that tracks forecast versions and enables iterative evidence gathering and forecast revision. Intermediate forecasts can expose information gaps, while newly acquired evidence guides subsequent revisions and final version 
    
[^160]: 基于物理约束生成对抗网络且仅需有限系统知识的智能电网隐蔽高影响虚假数据注入攻击

    Stealthy, High-Impact False Data Injection Attacks in Smart Grid Using Physics-Constrained GAN with Limited System Knowledge

    [https://arxiv.org/abs/2611.15162](https://arxiv.org/abs/2611.15162)

    该论文提出一种仅需有限系统知识、与检测器无关的物理约束生成对抗网络（PCGAN），通过从历史量测中学习解耦的潮流灵敏度，生成兼顾隐蔽性与高影响的智能电网虚假数据注入攻击。

    

    虚假数据注入（FDI）攻击能够在保持看似合理的量测模式的同时扭曲电力系统状态估计。现有的FDI方法往往在隐蔽性与攻击影响之间进行权衡，并且通常假设攻击者能够访问电网内部信息、全部母线量测或防御者模型。本文提出一种与检测器无关的物理约束生成对抗网络（PCGAN），它能够从已部署传感器的历史量测数据中学习解耦的潮流灵敏度。学习到的有功-相角与无功-电压幅值关系用于引导仅限于被攻陷通道的扰动，同时对抗训练促使攻击数据呈现真实的时空模式。我们在IEEE 300节点系统上，利用美国区域输电组织PJM互联公司一年的实测小时需求数据以及合成的母线级负荷曲线，实现了PCGAN及七种现有的FDI攻击，并对它们的性能进行了比较……

    arXiv:2611.15162v2 Announce Type: replace  Abstract: False data injection (FDI) attacks can distort power system state estimation while retaining plausible measurement patterns. Existing FDI methods trade stealth for impact or vice versa. They often assume access to internal grids, all-bus sensing, or defender models. In this paper, we propose a detector-agnostic physics-constrained generative adversarial network (PCGAN) that learns decoupled power-flow sensitivities from historical measurements of deployed sensors. The learned active-power-angle and reactive-power-voltage-magnitude relationships guide perturbations restricted to compromised channels, while adversarial training promotes realistic spatiotemporal patterns. We implement PCGAN and seven existing FDI attacks on the IEEE 300-bus system using one year of observed hourly demand from PJM Interconnection, a regional transmission organization in the United States, and synthetic bus-level load profiles. We compare their performanc
    
[^161]: 基于画像的南亚家庭暴力调查选择审计

    Profile-Based Auditing of Domestic Violence Survey Selection Across South Asia

    [https://arxiv.org/abs/2611.15106](https://arxiv.org/abs/2611.15106)

    本研究利用机器学习模型对南亚五国近80万女性的特征画像进行分析，发现人口与社会经济特征能够高度预测家庭暴力调查模块的样本选择，从而为审计DHS家暴数据的选择偏差和代表性提供了画像审计方法。

    

    人口与健康调查（DHS）中的家庭暴力（DV）数据通过专门模块收集，其住户层面的抽样和隐私要求形成了一种独特的观测过程。我们探究人口统计、社会经济和住户特征能否形成可重现的人群画像，以及这些画像是否包含关于南亚各地区家庭暴力模块选择信息。研究使用了覆盖阿富汗、印度、缅甸、尼泊尔和巴基斯坦的796,374名15-49岁女性的DHS个体重编码数据，统一协调了61个数值特征和8个分类特征，通过主成分分析（PCA）降低画像空间维度，并识别出六个可解释的画像。随后，在考虑初级抽样单元的留出验证框架下评估了22个机器学习模型。表现最佳的模型HistGradientBoosting达到了ROC-AUC 0.9167和PR-AUC 0.9612，而仅基于抽样设计的基线分别为0.8475和0.9203。加入女性个体层面的特征可进一步……

    arXiv:2611.15106v2 Announce Type: replace  Abstract: Domestic violence (DV) data in the Demographic and Health Surveys (DHS) are collected through a specialized module whose household-level sampling and privacy requirements create a distinct observation process. We ask whether demographic, socioeconomic, and household characteristics form reproducible population profiles and whether these profiles contain information about selection into the DV module across South Asia. Using DHS Individual Recode data for 796,374 women aged 15-49 across Afghanistan, India, Myanmar, Nepal, and Pakistan, we harmonize 61 numeric and 8 categorical features, reduce the profile space with PCA, and identify six interpretable profiles. We then evaluate 22 machine-learning models under primary-sampling-unit-aware holdout validation. The best model, HistGradientBoosting, achieves ROC-AUC 0.9167 and PR-AUC 0.9612, versus 0.8475 and 0.9203 for a design-only baseline. Adding woman-level characteristics yields the 
    
[^162]: 技术驱动的域偏移下基于保形风险控制的QoT感知光路可行性评估

    Conformal Risk Control for QoT-Aware Lightpath Feasibility Under Technology-Driven Domain Shift

    [https://arxiv.org/abs/2611.13893](https://arxiv.org/abs/2611.13893)

    本文提出一种事后保形风险控制框架，在保持预训练AI/ML回归模型不变的情况下，为技术升级引起域偏移时光网络QoT感知光路可行性评估的不确定性区间提供有限样本统计保证，并通过策略规则将预测区间映射为运行决策。

    

    基于人工智能和机器学习（AI/ML）的回归模型能够支持光网络中的运行决策，但通常只提供点预测，缺乏统计上可控的可靠性信息。本文提出一种事后保形风险控制（CRC）框架，该框架保持预训练的AI/ML回归器不变，为不确定性区间的覆盖失败提供有限样本保证，并通过策略规则将该区间映射为运行决策。该保证适用于区间覆盖失败，而决策错误率则通过实证进行评估。我们将该框架实例化用于传输质量（QoT）感知的光路（LP）可行性评估，其中回归器在光纤、放大器和交换设备升级引起的域偏移下预测光信噪比（OSNR）。特征偏移贡献分数能够识别出与QoT关系变化最大的光路特征。

    arXiv:2611.13893v1 Announce Type: new  Abstract: Artificial intelligence and machine learning (AI/ML)-based regression models support operational decision-making in optical networks but typically expose only point predictions, which carry no statistically controlled information about their reliability. This paper proposes a post-hoc conformal risk control (CRC) framework that keeps a pretrained AI/ML regressor fixed, provides a finite-sample guarantee on uncertainty band miscoverage, and maps the band to operational decisions through policy rules. The guarantee applies to interval miscoverage, while decision-error rates are evaluated empirically. We instantiate the framework for quality of transmission (QoT)-aware lightpath (LP) feasibility assessment, in which a regressor predicts optical signal-to-noise ratio (OSNR) under domain shifts caused by fiber, amplifier, and switching upgrades. Feature shift contribution scores identify the LP features whose relation to QoT changes the most 
    
[^163]: ForeSpark：面向并行投机解码的草稿与验证预算联合调度

    ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding

    [https://arxiv.org/abs/2611.13442](https://arxiv.org/abs/2611.13442)

    该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。

    

    arXiv:2611.13442v1 公告类型：cross 摘要：并行草稿器通过在单次前向传播中提出一整块草稿token来加速LLM解码，使其对LLM服务具有很强的吸引力。近期的系统优化的是有效吞吐量（goodput，即每秒被接受的token数），因为被拒绝的草稿会抢占其他请求的计算资源。然而，这些系统只能控制验证多少个token，因此每个请求仍然会草稿出完整的一块token。在高服务负载下，我们发现草稿生成会消耗高达40%的步进时间，其中大部分因从未被验证而被丢弃。而且，来自自回归草稿器的草稿长度控制方式无法直接迁移到并行草稿器，因为并行草稿器必须在其唯一一次前向传播之前就确定其长度。此外，草稿长度与验证长度在整个批次中相互耦合，因此必须为每个请求联合选择这两者，同时保证该决策在每一步的计算开销都足够低。在本文中，我们提出节省工作的投机（WCS）设计，该设计联合分配……

    arXiv:2611.13442v1 Announce Type: cross  Abstract: Parallel drafters accelerate LLM decoding by proposing a whole block of draft tokens in a single forward pass, making them attractive for LLM serving. Recent systems optimize goodput (i.e., accepted tokens per second) because rejected drafts steal compute from other requests. However, these systems control only how many tokens are verified, so every request still drafts a full block. At high serving load, we find that drafting consumes up to 40% of step time, of which a majority is discarded because it is never verified. Yet, draft-length control from autoregressive drafters does not carry over, because a parallel drafter commits to its length before its single pass. Moreover, draft and verification lengths are coupled across the batch, so they must be selected jointly for every request while keeping the decision computationally cheap at every step. In this paper, we propose work-conserving speculation (WCS), a design that jointly allo
    
[^164]: 面向联邦LoRA的随机投影与秩感知学习率

    Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

    [https://arxiv.org/abs/2611.13331](https://arxiv.org/abs/2611.13331)

    该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。

    

    在资源受限的分布式客户端上对大规模基础模型进行微调，面临着显著的计算与通信挑战。虽然将低秩适应与联邦学习相结合可以实现基于秩的灵活性，但适应矩阵（A 与 B）之间的相互依赖往往使优化变得复杂。在这项工作中，我们从理论上证明：采用高斯初始化并冻结 A 矩阵的 LoRA 微调，相当于对全量微调的一种随机近似，其中批量梯度会受到随机且与秩相关的扰动。受这一见解的启发，我们提出在客户端层面冻结并周期性地重新采样 A 矩阵。该策略解耦了 A 和 B 的梯度，削弱了 LoRA 的低秩约束，并将计算与上行通信开销降低了 50%，从而允许重新分配……（原文在此处截断）

    arXiv:2611.13331v1 Announce Type: cross  Abstract: Fine-tuning large-scale foundation models across resource-constrained, distributed clients presents significant computational and communication challenges. While merging Low-Rank Adaptation (LoRA) with Federated Learning (FL) enables rank-based flexibility, the interdependence of the adaptation matrices ($\mathbf{A}$ and $\mathbf{B}$) often complicates optimization. In this work, we theoretically demonstrate that LoRA fine-tuning with Gaussian initialization and a frozen $\mathbf{A}$ matrix acts as a stochastic approximation of full fine-tuning, where batch gradients are subject to a random, rank-dependent perturbation. Motivated by this insight, we propose freezing and periodically resampling $\mathbf{A}$ matrices at the client level. This strategy decouples the $\mathbf{A}$ and $\mathbf{B}$ gradients, diminishes LoRA low-rank constraints and reduces both computational and uplink communication overheads by 50\%, allowing for the reall
    
[^165]: GigaEdge Spench：一个面向实用流式语音增强的软件开发工具包（SDK）

    GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement

    [https://arxiv.org/abs/2611.13243](https://arxiv.org/abs/2611.13243)

    该论文发布了开源 SDK GigaEdge Spench，通过共享 C++ 核心与多平台集成（服务器、移动端、浏览器）实现实用的高速流式语音增强，其模型质量和处理速度优于 RNNoise 和 DPDFNet2 且快于实时。

    

    我们提出了 GigaEdge Spench，一个用于在 CPU 服务器、移动设备和浏览器上进行流式语音增强的开源 SDK。应用程序可以由神经去噪、归一化与动态处理模块、采样率转换，以及在支持情况下的声学回声消除来构建增强通道。这些处理阶段可以组合使用，同时归一化也可以在没有神经模型的情况下独立运行。本次发布包含一个 8 kHz INT8 去噪器，以及判别式和混合式的 48 kHz FP32 模型。一个共享的 C++ 核心通过原生接口支持 ONNX Runtime 和 TensorFlow Lite，并提供 Android 和 iOS SDK 以及 WebAssembly 集成。独立的流上下文可以共享模型资源，且来自同一流的连续帧可以在单次推理调用中处理。我们在 EARS-WHAM v2 数据集上评估了发布的模型，并将其质量与处理开销与 RNNoise 和 DPDFNet2 进行了比较。所有测试的原生和 Web 配置处理文件的速度均快于实时。

    arXiv:2611.13243v1 Announce Type: new  Abstract: We present GigaEdge Spench, an open-source SDK for streaming speech enhancement on CPU servers, mobile devices, and in browsers. Applications can build enhancement channels from neural denoising, normalization and dynamics blocks, sample-rate conversion, and acoustic echo cancellation where supported. These stages can be combined, while normalization can also run without a neural model. The release includes an 8 kHz INT8 denoiser and discriminative and hybrid 48 kHz FP32 models. A shared C++ core supports ONNX Runtime and TensorFlow Lite through native interfaces, Android and iOS SDKs, and WebAssembly integration. Independent stream contexts share model resources, and consecutive frames from one stream can be processed in a single inference call. We evaluate the released models on EARS-WHAM v2 and compare their quality and processing costs with RNNoise and DPDFNet2. All tested native and Web configurations process files faster than real 
    
[^166]: 言语之前，超越语音：评估幼儿期的非语言社会推理能力

    Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood

    [https://arxiv.org/abs/2611.13213](https://arxiv.org/abs/2611.13213)

    该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。

    

    儿童最早的岁月是一个高强度的社会学习时期，在此期间，意义通过眼神注视、手势和身体接触而构建。然而，社会互动的这一关键方面在多模态视频理解领域仍未得到充分评估，因为现有基准主要优先关注以对话驱动的成人互动。我们通过NEST（自然情境下的幼儿社会互动）基准来填补这一空白，这是首个以儿童为中心的非语言社会推理基准。NEST包含大量人工标注的视频片段（约12秒），这些片段在自然情境、文化背景和发展阶段方面都经过精心挑选以确保多样性。除了数据匮乏之外，该领域针对性的模型改进还受到粗略评估方案的阻碍，这些方案将基础感知与高阶推理混为一谈。为解决这一问题，NEST引入了一种组合式评估方案，在原子级交互线索的基础上构建（摘要在此处截断）……

    arXiv:2611.13213v1 Announce Type: cross  Abstract: Children's earliest years are a period of intense social learning, during which meaning is built through gaze, gesture, and physical contact. Yet, this critical aspect of social interaction remains under-evaluated in multimodal video understanding, as existing benchmarks prioritize the conversation-driven interactions of adults. We address this gap with \bench ($\textbf{N}$aturalistic $\textbf{E}$arly-childhood $\textbf{S}$ocial in$\textbf{T}$eractions), the first child-centric benchmark for nonverbal social reasoning. \bench comprises \totalvids manually annotated clips ($\sim$12s) curated for diversity across naturalistic settings, cultural contexts, and developmental stages. Beyond a lack of data, targeted model improvement in this domain is hindered by coarse evaluation schemas that conflate basic perception with higher-order reasoning. To resolve this, \bench introduces a compositional schema that builds atomic interaction cues in
    
[^167]: Agentic-OPD：通过在线策略蒸馏恢复智能体任务中被遗忘的智能体能力

    Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks

    [https://arxiv.org/abs/2611.12799](https://arxiv.org/abs/2611.12799)

    该论文提出Agentic-OPD方法，将GRPO训练中被丢弃的检查点作为免费的教师池，通过有选择性的在线策略蒸馏来恢复最佳验证检查点所遗忘的智能体问题求解能力。

    

    强化学习目前是训练大语言模型执行智能体任务的主流范式，但标准做法只保留验证集表现最佳的检查点，而丢弃其余的检查点。我们证明这种做法是有损失的：在大语言模型智能体的GRPO训练中，验证集最佳的检查点会在同一训练过程中早期检查点已经解决的问题上失败，且所有检查点所解决问题的并集远超任何单个检查点所解决的。因此，这些被丢弃的检查点构成了一个无需额外训练成本即可获得的教师池，在线策略蒸馏（OPD）可以从中挖掘以恢复丢失的覆盖能力。挖掘这一教师池带来两个挑战：其一，应从哪个检查点蒸馏，因为没有哪个检查点能全面优于其他检查点，且其与学生模型的互补性因具体问题而异；其二，应在何处信任它，因为即使是精心挑选的教师模型也只是局部更优，不加区分的蒸馏会覆盖学生模型已经掌握的行为。我们提出……

    arXiv:2611.12799v2 Announce Type: replace-cross  Abstract: Reinforcement learning is now the mainstream paradigm for training LLMs on agentic tasks, yet standard practice keeps only the best-validation checkpoint and discards the rest. We show that this is lossy: in GRPO training of LLM agents, the best-validation checkpoint fails on problems that earlier checkpoints from the same run had already solved, and the union of problems solved across checkpoints far exceeds any single one. The discarded checkpoints thus form a teacher pool obtained at no additional training cost, which on-policy distillation (OPD) can mine to recover the lost coverage. Mining this pool raises two challenges: i) which checkpoint to distill from, as no checkpoint dominates the others and complementarity to the student is problem-specific; ii) where to trust it, as even a well-chosen teacher is only locally superior, so indiscriminate distillation overwrites behavior the student has already mastered. We propose 
    
[^168]: 弱支持下的轨迹密度比：跨科学的综述

    Trajectory Density Ratios under Weak Support: A Survey across the Sciences

    [https://arxiv.org/abs/2611.11985](https://arxiv.org/abs/2611.11985)

    本文是一篇跨学科综述，系统梳理了轨迹密度比这一统一量在强化学习、分子模拟、气候归因和语言模型后训练等领域的核心作用，并分析了其在支撑薄弱（弱支持）情形下的存在性条件、方差爆炸与无定义等根本性失效问题。

    

    arXiv:2611.11985v2 公告类型：replace-cross。摘要：科学领域的许多问题需要通过两个随机过程生成的序列来对它们进行比较。强化学习问的是：从旧策略的轨迹出发，一个新策略会带来怎样的回报；分子模拟问的是：从未结合态的样本出发，配体的结合能力提高了多少；气候归因问的是：在变暖背景下，一场热浪发生了怎样的偏移；语言模型后训练问的是：从记录下来的交互数据出发，部署后的智能体会做出什么行为。在所有这些情形中，有一个量可以同时回答上述所有问题：两个概率规律在其生成序列上的比值，即轨迹密度比。这个比值只在很强的条件下才存在。当一个过程到达了另一个过程无法到达的状态时，该比值不存在；当两者的重叠区域很薄时，其方差会随序列长度不受控制地增长；而当两者集中于不同的低维集合上时，该比值则完全无定义。弱支持（原文摘要在此处截断）

    arXiv:2611.11985v2 Announce Type: replace-cross  Abstract: Many problems across the sciences compare two stochastic processes through the sequences they generate. Reinforcement learning asks what a new policy would return from the trajectories of an old one; molecular simulation asks how much more a ligand binds from samples of the unbound state; climate attribution asks how a heat wave shifted under warming; and language-model post-training asks what a deployed agent will do from logged interactions. In every case, one quantity would answer all these questions at once: the ratio of the two probability laws over their generated sequences, the trajectory density ratio. This ratio exists only under strong conditions. When one process reaches states the other cannot, the ratio does not exist; when their overlap is thin, its variance grows without control with the sequence length; and when they concentrate on different lower-dimensional sets, the ratio is undefined altogether. Weak support
    
[^169]: on'ona：双嵌入与绑定反嵌入

    on\'ona: Twin Embeddings with Tied Unembeddings

    [https://arxiv.org/abs/2611.10556](https://arxiv.org/abs/2611.10556)

    on'ona 提出了一种双嵌入表参数化方法，通过仅依赖当前输入 token 的门控来混合两个共享嵌入表定义的 softmax 分布，在保留权重绑定省参数优势的同时增强了输出头表达能力，在多种架构和三个规模下显著降低验证损失并提升下游任务准确率。

    

    权重绑定降低了语言模型的参数开销，但通常的绑定输出头仅限于单一的 softmax。我们提出了 on'ona，这是一种双表参数化方法，在保持输入-输出权重共享的同时提升了输出的表达能力。Transformer 将两个嵌入表的平均值作为输入，而输出头则组合由这两个相同的表所定义的 softmax 分布。一个仅取决于当前输入 token 的门控决定混合权重，其数值可以在推理时预先计算，且 Transformer 主体保持不变。我们在 142M、330M 和 1B 规模下，跨越稠密、Gated DeltaNet 混合、混合专家和注意力残差等架构对 on'ona 进行了评估。在匹配的 token 预算下，on'ona 在 12 项对比中的 11 项里降低了最终验证损失和外部困惑度，并在 10 项中提高了八个下游任务的平均准确率。损失方面的唯一例外是 1B

    arXiv:2611.10556v2 Announce Type: replace  Abstract: Weight tying reduces the parameter cost of language models, but the usual tied output head is limited to a single softmax. We introduce on\'ona, a two-table parameterization that increases output expressiveness while preserving input-output weight sharing. The Transformer takes the average of two embedding tables as input, and the output head combines the softmax distributions defined by those same tables. A gate conditioned only on the current input token determines the mixture weight. Its values can be precomputed for inference, and the Transformer body remains unchanged. We evaluate on\'ona across Dense, Gated DeltaNet hybrid, mixture-of-experts, and Attention Residual architectures at 142M, 330M, and 1B scales. Under matched token budgets, on\'ona lowers final validation loss and external perplexity in 11 of 12 comparisons and raises the average accuracy over eight downstream tasks in 10 of 12. The one exception on loss is the 1B
    
[^170]: LEGIT：基于标识符字典树的实时资格门控，用于异构页面的受限解码

    LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages

    [https://arxiv.org/abs/2611.10357](https://arxiv.org/abs/2611.10357)

    提出 LEGIT 约束机制，通过对标识符字典树叶节点进行深度优先编号、把随请求和 token 动态变化的资格规则转化为高效的范围查找，使生成式推荐系统在整页解码时能精确且快速地保证每个语义 ID 真实、合规且不重复。

    

    生成式推荐器用一段简短的 token 序列（即语义 ID，Semantic ID）来命名每个候选，并通过生成这些 token 来完成推荐。一些生成式推荐器，包括我们正在构建的 Uber Eats 搜索引擎，会在一次解码过程中写出整个结果页面，其中包含商店及其菜单项。这样的页面上，每个语义 ID 都必须指向一个真实的、且符合本次搜索资格的商店或条目（例如，能够配送到用户所在位置的商店），并且尚未出现在该页面上；此外，每个条目还必须属于其上方的商店。在商品目录上构建字典树可以保证标识符的真实性，但资格性与重复性约束会随着每次请求和每个 token 动态变化，而我们尝试过的语法约束解码器无法既精确又快速地执行这些不断变化的规则。我们将这一约束命名为 LEGIT，即“基于标识符字典树的实时资格门控”。对字典树叶节点进行深度优先编号，可以将每条规则转化为对一棵商店字典树和一棵条目字典树的范围查找。

    arXiv:2611.10357v2 Announce Type: replace-cross  Abstract: A generative recommender names each candidate with a short token sequence, a Semantic ID, and recommends by generating those tokens. Some generative recommenders, including the Uber Eats search engine we are building, write a whole results page, stores with their menu items, in one decoding pass. Every Semantic ID on such a page must name a real store or item that is eligible for the search, such as a store that delivers to the user's location, and that is not already on the page. Each item must also belong to the store above it. A trie over the catalog keeps identifiers real, but eligibility and repetition change with every request and token, and the grammar-constrained decoder we tried could not enforce these changing rules exactly and quickly. We call this constraint LEGIT, for live eligibility gating over identifier tries. Numbering trie leaves depth first turns each rule into a range lookup over one store trie and one item
    
[^171]: 剩余多少很重要：部分提示污染测试中的假阳性与假阴性

    How Much Is Left Matters: False Positives and False Negatives in Partial-Prompt Contamination Tests

    [https://arxiv.org/abs/2611.10307](https://arxiv.org/abs/2611.10307)

    该论文审计了部分提示污染测试，发现剩余文本越短假阳性越多（短剩余文本贡献了大部分 ROUGE 标记，在真阴性题目上 80% 前缀时误标率高达 8.6%），且该测试无法检测通过蒸馏传播的上游污染。

    

    部分提示补全是指模型接收基准测试题目的前半部分，然后检验其能否复现剩余部分，这是一种广泛使用的基准污染检测方法。我们使用 MATH-500 以及最新的 AIME 2025/2026 题目（作为真阴性样本）对七个开源权重模型进行了这项测试的审计。记忆化标记强烈依赖于剩余文本的多少：对于每个模型，剩余单词不超过 8 个的题目被标记的频率是剩余单词至少 25 个的题目的数倍（Qwen2.5-7B-Instruct 上为 31.5% 对 5.5%），且所有基于 ROUGE 的标记中有二分之一到四分之三来自较短的剩余文本。在真阴性样本上，ROUGE-L ≥ 0.75 阈值在 60% 前缀时最多会误标 6.7% 的题目，而在 80% 前缀时即使精确匹配也会误标 8.6%（跨模型汇总；在晚于所有模型训练时间的 AIME 2026 上为 6.2%），而 40-60% 前缀时为 0%。该测试还会遗漏上游污染：从 Qwen2.5-Math 蒸馏而来的模型……

    arXiv:2611.10307v3 Announce Type: replace-cross  Abstract: Partial-prompt completion, where a model receives the first part of a benchmark problem and is checked for whether it reproduces the rest, is a widely used test for benchmark contamination. We audit this test on seven open-weight models using MATH-500 and recent AIME 2025/2026 problems, which serve as true negatives. Memorization flags depend strongly on how much text remains: for every model, problems with at most 8 leftover words are flagged several times more often than problems with at least 25 (31.5% vs. 5.5% for Qwen2.5-7B-Instruct), and half to three-quarters of all ROUGE-based flags come from short leftovers. On true negatives, a ROUGE-L >= 0.75 threshold flags up to 6.7% of problems at a 60% prefix, and at an 80% prefix even exact match flags 8.6% (pooled over models; 6.2% on AIME 2026, which postdates every model), against 0% at 40-60%. The test also misses contamination upstream: models distilled from Qwen2.5-Math, w
    
[^172]: PINR-ESI：面向脑电扩展源成像的物理信息自监督隐式神经表示框架

    PINR-ESI: Physics-Informed Self-Supervised Implicit Neural Representation Framework for EEG Extended-Source Imaging

    [https://arxiv.org/abs/2611.06700](https://arxiv.org/abs/2611.06700)

    提出了一种物理信息驱动的自监督隐式神经表示框架PINR-ESI，利用导联场矩阵的物理约束和经典逆估计的暖启动初始化，在无需成对源标签或仿真预训练的情况下实现脑电扩展源成像的空间源重建。

    

    脑电（EEG）扩展源成像旨在定位活跃的皮层区域并刻画其空间范围，然而有监督的逆问题网络通常依赖于合成的脑电-源数据对，且可能对合成数据与真实数据之间的失配较为敏感。我们提出了一种物理信息自监督隐式神经表示（INR）框架，用于空间源重建，无需成对的源标签，也无需基于仿真的有监督预训练。导联场矩阵编码了从源到传感器的容积传导模型，并通过预测脑电与实测脑电之间的一致性来约束隐式神经表示。皮层电流密度被建模为一个连续的、以坐标为条件的场。经典的逆问题估计为源活动提供暖启动初始化和空间约束，同时基于块集成的投影将连续场映射为离散源系数以用于正向建模。在不同噪声水平、源数量和源范围条件下的仿真（摘要原文在此处截断）

    arXiv:2611.06700v1 Announce Type: new  Abstract: EEG extended-source imaging aims to localize active cortical regions and characterize their spatial extent, yet supervised inverse networks typically depend on synthetic EEG-source pairs and may be sensitive to synthetic-to-real mismatch. We propose a physics-informed self-supervised implicit neural representation (INR) framework for spatial source reconstruction without paired source labels or simulation-based supervised pretraining. The lead field matrix encodes the source-to-sensor volume-conduction model and constrains the INR through agreement between predicted and measured EEG. Cortical current density is modeled as a continuous coordinate-conditioned field. A classical inverse estimate provides warm initialization and spatial constraints on source activity, while patch-integrated projection maps the field to discrete source coefficients for forward modeling. Simulations with varying noise levels, source counts, and source extents 
    
[^173]: NaijaFacts：一个将语言与知识从关于尼日利亚的事实中分离出来的框架

    NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

    [https://arxiv.org/abs/2611.06504](https://arxiv.org/abs/2611.06504)

    该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。

    

    面向非洲语言的多语言基准测试大多是将通用问题翻译成当地语言，衡量的是语言能力，而非模型是否了解关于非洲的事实。我们提出了NaijaFacts，这是一个关于尼日利亚的事实性和统计性问题基准，由母语者撰写，并基于权威的官方统计数据，每个问题都有英语、尼日利亚皮钦语、约鲁巴语、伊博语和豪萨语版本。通过让模型回答针对英国和美国的类似问题，我们构建了一个国家×语言的矩阵，从而将语言效应（模型知道该事实，但在低资源语言中性能下降）与知识效应（该事实在任何语言中都不存在）分离开来。在八个前沿模型和尼日利亚适配模型上，知识效应占主导地位：以英语提问时，针对美国/英国的准确率为49%–50%，而针对尼日利亚仅为20%。语言效应较小且与网络表征程度相关，从英语到约鲁巴语逐渐下降（在尼日利亚问题上下降9个百分点）……

    arXiv:2611.06504v1 Announce Type: cross  Abstract: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nige
    
[^174]: 谁为模型复制的知识买单？作为罗杰斯悖论的模型坍塌

    Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox

    [https://arxiv.org/abs/2611.06022](https://arxiv.org/abs/2611.06022)

    该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。

    

    在先前模型输出上训练的生成模型会在代际间丢失知识，且稀有知识最先丢失。行之有效的补救措施都需要从循环外部引入信息，而现有的模型坍塌理论将这些信息供给视为既定条件：一个来自固定分布的固定档案库。我们认为供给本身才是问题的根源，而文化演化早已拥有关于这一问题的理论。我们将模型坍塌解读为罗杰斯悖论：一个逐渐依赖复制的人口最终并不比从不复制的人口境况更好，因为复制品会过时，除非有人持续进行自主学习。我们通过三个步骤检验这一论证。第一，仅用完全合成训练数据拟合的罗杰斯模型，能够以几个百分点以内的误差，预测语言模型——从词三元组模型到预训练Transformer——在其他新鲜数据混合比例下保留多少稀有知识。第二，当世界本身发生变化时，复制自身过往的谱系……（原文摘要在此处截断）

    arXiv:2611.06022v1 Announce Type: cross  Abstract: Generative models trained on the output of earlier models lose knowledge across generations, rare knowledge first. The remedies that work all bring in information from outside the loop, and the theory of model collapse treats that supply as given: a fixed archive, drawn from a fixed distribution. We argue that the supply is the problem, and that cultural evolution has the theory of it. We read model collapse as Rogers' paradox: a population that comes to rely on copying ends up no better off than one that never copied, because copies go stale unless someone keeps learning for themselves. We test the identification in three steps. First, Rogers' model, fitted only to fully synthetic training, predicts to within a few percentage points how much rare knowledge language models, from word trigrams to a pretrained transformer, retain under other mixtures of fresh data. Second, when the world itself changes, a lineage that copies its own past
    
[^175]: FedSharp：噪声信道下的拜占庭鲁棒去中心化联邦学习

    FedSharp: Byzantine Robust Decentralized Federated Learning Under Noisy Channel

    [https://arxiv.org/abs/2611.05546](https://arxiv.org/abs/2611.05546)

    提出FedSharp框架，通过维护2b个前驱节点模型、坐标式中值偏差筛选可疑模型并对损失最低的b个候选模型取平均，在噪声信道下有效抵抗拜占庭攻击与信道噪声累积，实现鲁棒的去中心化联邦学习。

    

    去中心化联邦学习（DFL）消除了中央聚合服务器，但仍然容易受到数据异构性、拜占庭攻击和信道噪声的影响。我们证明，随着模型在节点间传播，噪声会不断累积，并且随着学习率的衰减，这种累积噪声的影响变得越来越显著。此外，累积噪声使拜占庭攻击者能够通过模仿诚实模型的平均值来利用基于损失的模型选择机制。为应对这些挑战，我们提出了FedSharp——一种鲁棒的去中心化联邦学习框架，即筛选式、跳数感知、排序、近端平均方法。在该方法中，每个节点维护来自其2b个前驱节点的模型，从而在至多b个节点为拜占庭节点的情况下，确保至少有b个诚实的候选模型。FedSharp使用坐标式中值偏差来筛选可疑模型，对损失最低的b个候选模型进行平均，从而将新鲜信道噪声的方差降低b倍，并应用噪声自适应梯度范数正则化和近端训练来缓解……

    arXiv:2611.05546v1 Announce Type: new  Abstract: Decentralized federated learning (DFL) eliminates the central aggregation server but remains vulnerable to data heterogeneity, Byzantine attacks, and channel noise. We show that noise accumulates as models propagate across nodes, becoming increasingly significant as the learning rate decays. Moreover, accumulated noise enables Byzantine attackers to exploit loss-based model selection by mimicking honest model averages. To address these challenges, we propose screened, hop-aware, ranked, proximal averaging, a robust DFL framework (FedSharp). In this approach, each node maintains models from its 2b predecessors, ensuring at least b honest candidates when at most b nodes are Byzantine. FedSharp screens suspicious models using coordinate-wise median deviations, averages the b lowest-loss candidates to reduce fresh channel-noise variance by a factor of b, and applies noise-adaptive gradient-norm regularization and proximal training to mitigat
    
[^176]: 开放多智能体系统中强化学习的公理化框架

    An Axiomatic Framework for Reinforcement Learning in Open Multi-Agent Systems

    [https://arxiv.org/abs/2611.03301](https://arxiv.org/abs/2611.03301)

    本文提出一个以 Shapley 公理为结构性约束的公理化框架，通过新颖的 NAHT 分解刻画开放与完全可控多智能体系统的本质差异，为开放多智能体系统的强化学习奠定合作博弈论基础，并证明对个体价值函数施加这些公理可恢复 Dec-POMDP 的 Shapley 值。

    

    开放多智能体系统（其中只有一部分智能体是可控的）是智能电网等现实应用领域的核心，但此类系统缺乏一个有原则的强化学习算法设计基础。我们通过 n 智能体临时团队协作（n-agent ad hoc teamwork, NAHT）这一新兴的开放多智能体环境学习框架来研究该设定。我们提出了一个公理化框架，从 Shapley 公理（有效性、对称性与线性性）出发推导价值函数学习，将这些公理视为结构性约束而非启发式设计选择。这一表述建立在新颖的 NAHT 分解之上，该分解刻画了开放系统与完全可控多智能体系统之间的根本差异，并为学习建立了合作博弈论基础。我们进一步证明，在学习个体价值函数时强制执行这些公理可以恢复 Dec-POMDP 的 Shapley 值，从而在……之间架起一座有原则的桥梁（原文摘要在此处截断）。

    arXiv:2611.03301v3 Announce Type: replace-cross  Abstract: Open multi-agent systems, where only a subset of agents are controllable, are central to real-world domains such as smart grids, yet lack a principled foundation for reinforcement learning algorithm design. We study this setting through n-agent ad hoc teamwork (NAHT), a recent framework for learning in open multi-agent environments. We propose an axiomatic framework that derives value function learning from the Shapley axioms (Efficiency, Symmetry and Linearity), treating them as structural constraints rather than heuristic design choices. This formulation builds on a novel NAHT decomposition, which characterises the fundamental difference between open and fully controllable multi-agent systems, and establishes a cooperative game-theoretic foundation for learning. We further show that enforcing these axioms on learning individual value functions recovers the Shapley value for Dec-POMDPs, providing a principled bridge between re
    
[^177]: 小心你的步伐：Adam的间隔最大化偏差取决于学习率调度

    Watch Your Step: Adam's Margin Maximization Bias Depends on the Learning Rate Schedule

    [https://arxiv.org/abs/2611.01929](https://arxiv.org/abs/2611.01929)

    本文揭示了Adam的隐式偏差依赖于学习率调度——在常见的恒定或弱衰减调度下，Adam对齐次模型不再具有与轨迹无关的间隔最大化偏差。

    

    我们研究了Adam的隐式偏差，以及其在二分类任务中于齐次模型上与符号梯度下降的相似性。或许令人惊讶的是，我们发现这些性质依赖于学习率调度。以往的工作研究了强衰减的学习率调度 $\eta(t)=\mathrm o\left(t^{1/L-1}\right)$（其中 $L$ 为齐次度），并证明了 $\ell_\infty$ 间隔最大化偏差；而我们聚焦于更常见的互补情形，包括恒定调度和弱衰减调度。对于多项式调度 $t^{\alpha}$ 和齐次度 $L$，我们证明当 $\alpha > \frac{1}{L} - \frac{1}{2}$ 时，Adam对深度线性网络不具有与轨迹无关的间隔最大化偏差（在KKT意义下）：首先，在此设置下不存在任何非平凡的、局部Lipschitz的凸隐式正则化器，包括任意平方范数；其次，在宽度为 $1$ 的网络中，该结论无需凸性假设即可成立……

    arXiv:2611.01929v2 Announce Type: replace  Abstract: We study the implicit bias of Adam and its similarity to sign gradient descent on homogeneous models in binary classification tasks. Perhaps surprisingly, we uncover that these depend on the learning rate schedule. While previous works studied strongly decaying learning rate schedules $\eta(t)=\mathrm o \left(t^{1/L-1}\right)$, where $L$ is the homogeneity degree, and showed $\ell_\infty$-margin-maximization biases, we focus on the more common complementary regime, including constant and weakly decaying schedules. For a polynomial schedule $t^{\alpha}$ and homogeneity degree $L$, we show that when $\alpha > \frac 1L - \frac12$, Adam has no trajectory-independent margin-maximization bias for deep linear networks (in the KKT sense): first, there is no non-trivial, locally Lipschitz, convex implicit regularizer in this setting, including any squared norm. Second, on width-$1$ networks, the result holds without requiring convexity, imply
    
[^178]: 基于因果先验约束的深度学习实现西北太平洋热带气旋快速增强预警

    Causal-prior-constrained deep learning for rapid-intensification warning of western North Pacific tropical cyclones

    [https://arxiv.org/abs/2610.40304](https://arxiv.org/abs/2610.40304)

    该研究提出CausalVisionRI系统，将观测发现的因果图及八项符号验证效应作为可审计的因果先验约束嵌入深度学习模型，融合红外遥感影像与大气海洋场数据，显著提升了西北太平洋热带气旋24小时快速增强预警的性能。

    

    快速增强（RI）是热带气旋中最危险且最难以预测的阶段之一，其环境与结构驱动因素在观测中难以识别和分离。本文提出了CausalVisionRI，一个面向西北太平洋24小时快速增强预警的因果先验约束深度学习系统。该系统将红外遥感影像、大气和海洋场以及通过观测发现的因果图整合到一个统一的模型中。该因果图及八项经过符号验证的24小时强度变化效应为训练提供了可审计的因果先验。该因果图基于2000-2017年的522个气旋发现，识别了连接大气和海洋条件、气旋动力与结构状态及其24小时强度变化之间的因果路径。在涵盖2020-2024年、共3,004条记录的前向留出测试集上，五成员集成模型相对于固定基准取得了0.259的Brier技巧评分（BSS）

    arXiv:2610.40304v3 Announce Type: replace  Abstract: Rapid intensification (RI) is one of the most dangerous and unpredictable phases of tropical cyclones, and its environmental and structural drivers are difficult to identify and separate in observations. Here we introduce CausalVisionRI, a causal-prior-constrained deep-learning system for 24-h RI warning in the western North Pacific. It integrates infrared remote sensing imagery, atmospheric and oceanic fields, and an observationally discovered causal graph into a unified model. The graph and eight sign-verified effects on 24-h intensity change provide an auditable causal prior for training. Discovered from 522 cyclones in 2000-2017, it identifies causal pathways linking atmospheric and oceanic conditions, the cyclone's dynamical and structural state, and its 24-h intensity change. On a forward holdout covering 2020-2024 and 3,004 records, the five-member ensemble achieves a Brier skill score (BSS) of 0.259 relative to a fixed develo
    
[^179]: 模式动力学：一个神经网络学习框架

    Mode Dynamics: A Framework for Neural Network Learning

    [https://arxiv.org/abs/2610.40143](https://arxiv.org/abs/2610.40143)

    该论文提出“模式动力学”框架，通过分解、重构和集中度三种机制将神经网络训练中参数的集体运动模式与其输出效应联系起来，发现少数主导模式即可重构模型行为，且效应依赖训练路径、模式复杂度随任务和数据变化。

    

    我们提出了模式动力学，这是一个将神经网络学习过程中参数的协同运动与模型输出变化联系起来的框架。从记录的参数轨迹出发，我们将参数更新分解为集体模式，并将每种模式映射到其依赖于状态的输出效应。该框架通过三种机制来组织这一联系。分解：识别参数运动中的集体模式及其随时间变化的振幅。重构：检验这些模式诱导的效应能否恢复局部和累积的输出变化。集中度：量化行为贡献在各模式之间的分布情况。受控的标量回归和下一词元预测实验直接检验了这些机制。结果表明，少数主导模式即可重构模型行为，诱导效应依赖于训练路径，且模式复杂度随任务和数据而变化。在一个受控的归纳任务中，抑制主导模式……

    arXiv:2610.40143v2 Announce Type: replace  Abstract: We introduce Mode Dynamics, a framework that connects coordinated parameter motion during neural network learning to changes in model outputs. Starting from a recorded parameter trajectory, we decompose its updates into collective modes and map each mode to its state-dependent output effect. The framework organizes this link through three mechanisms. Decomposition identifies collective modes in parameter motion and their time-varying amplitudes. Reconstruction tests whether the effects induced by those modes recover local and accumulated output change. Concentration quantifies how behavioral contribution is distributed across modes. Controlled scalar regression and next-token experiments directly test these mechanisms. They show that small sets of leading modes reconstruct behavior, induced effects depend on the training path, and modal complexity varies with task and data. In a controlled induction task, suppressing the leading mode
    
[^180]: EmbodiExp：面向具身智能体的可组合经验迁移

    EmbodiExp: Compositional Experience Transfer for Embodied Agents

    [https://arxiv.org/abs/2610.39180](https://arxiv.org/abs/2610.39180)

    EmbodiExp 提出一种以结果为依据的子目标经验表示，使具身智能体能够在任务执行过程中在线组合与迁移经验，并借助子目标层面的结果归因来利用失败轨迹。

    

    arXiv:2610.39180v1 公告类型：cross 摘要：经验可以帮助具身智能体将高层规划转化为落地动作，但跨任务迁移经验仍然充满挑战：先前有效的策略可能不适用于当前状态，或者无法产生预期的效果。我们提出 EmbodiExp，一个用于构建可迁移经验并在具身任务执行过程中在线组合这些经验的框架。我们的关键观察是：任务往往共享一些子目标，而这些子目标的完成情况可以独立于整体任务的成功与否进行评估。因此，我们引入了一种以结果为依据的子目标经验，它明确了子目标、适用条件、可调整的策略以及可验证的结果。在执行过程中，适用的经验引导动作生成，而经过验证的结果可以建立后续经验所需的条件，从而实现在线组合。子目标层面的结果归因进一步实现了对失败的利用：失败的轨迹……

    arXiv:2610.39180v1 Announce Type: cross  Abstract: Experience can help embodied agents turn high-level plans into grounded actions, but transferring it across tasks remains challenging: previously effective strategies may be inapplicable to the current state or fail to produce their intended outcomes. We present EmbodiExp, a framework for constructing transferable experiences and composing them online during embodied task execution. Our key observation is that tasks often share subgoals whose completion can be assessed independently of overall task success. We therefore introduce an outcome-grounded subgoal experience that specifies a subgoal, applicability conditions, an adaptable policy, and a verifiable outcome. During execution, applicable experiences guide action generation, while verified outcomes can establish the conditions required by subsequent experiences, enabling online composition. Subgoal-level outcome attribution further enables failure utilization: failed trajectories 
    
[^181]: Brain2Vec：一种用于结构化脑部MRI的自监督3D基础模型

    Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI

    [https://arxiv.org/abs/2610.28411](https://arxiv.org/abs/2610.28411)

    Brain2Vec是一个在74,425个T1加权脑部MRI扫描上预训练的自监督3D基础模型，既可作为下游任务微调的权重初始化骨干，也可作为独立特征提取器，在脑年龄预测等任务中展现出更高的精度和更快的收敛速度。

    

    深度学习模型已成为医学影像领域不可或缺的工具，然而由于标注数据有限和计算资源受限，针对多样化任务进行端到端训练仍然充满挑战。为解决这一问题，我们提出了Brain2Vec，这是一个在汇集自多项研究的74,425个T1加权磁共振成像（MRI）扫描数据上预训练的基础模型。Brain2Vec既可以作为权重初始化的骨干网络用于下游任务的微调，也可以作为独立的特征提取器使用。我们从三个方面对Brain2Vec进行了评估：下游预测性能、收敛速度和嵌入质量。在脑年龄预测任务中，使用Brain2Vec的预训练权重作为初始化并在标注数据上进行微调，结果表明与随机初始化相比，最终精度得到提升且收敛速度更快。此外，其冻结嵌入特征在五个下游分类任务中的四个上优于基于解剖学定义的体积特征。

    arXiv:2610.28411v1 Announce Type: cross  Abstract: Deep learning models have become essential tools in medical imaging, yet training them end-to-end for diverse tasks remains challenging due to limited labeled data and computational constraints. To address this, we introduce Brain2Vec, a foundation model pre-trained on 74,425 T1-weighted Magnetic Resonance Imaging (MRI) scans pooled from multiple studies. Brain2Vec can serve both as a weight initialization backbone for fine-tuning downstream tasks and as a standalone feature extractor. We evaluated Brain2Vec in three tasks: downstream predictive performance, convergence speed, and embedding quality. Using Brain2Vec's pretrained weights as initialization and fine-tuning on labeled data for a brain age prediction task, results demonstrate improved final accuracy and faster convergence compared to random initialization. Additionally, its frozen embeddings outperform anatomically defined volumetric features on four of five downstream class
    
[^182]: 学会读出：语言模型预训练中的反嵌入动态

    Learning to Read Out: Unembedding Dynamics in Language Model Pretraining

    [https://arxiv.org/abs/2610.21025](https://arxiv.org/abs/2610.21025)

    提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。

    

    语言模型是在何时获得某种能力的？发展性研究通常通过追踪模型何时开始生成正确的词元来回答这个问题。但这一预测需要经过反嵌入矩阵（即读出层），而读出层本身在预训练过程中也仍在被学习。我们证明，模型的隐藏状态可以在其自身的读出层表达任务信息之前就已编码该信息，因此从输出中读取的时间线混淆了模型所编码的内容与读出层所表达的内容。为了直接追踪读出层，我们引入了参数轨迹交叉编码方法，该方法在每个检查点为每个词元的读出行拟合一个稀疏字典。与作用于激活值的交叉编码器不同，它只需要检查点的权重。我们利用该方法定位到一个狭窄的早期窗口——即Pythia预训练进程的0.4%–1.1%阶段——在此期间读出层发生重组，WordNet词类变得可以从其行中解码。到那时，对隐藏状态的线性探针已经能够解码

    arXiv:2610.21025v1 Announce Type: cross  Abstract: When does a language model acquire a capability? Developmental studies usually answer this by tracking when models start producing the correct token. But that prediction passes through the unembedding matrix, or readout, which is itself still being learned during pretraining. We show that a model's hidden states can encode task information before its own readout expresses it, so timelines read from outputs conflate what the model encodes with what its readout expresses. To track the readout directly, we introduce parameter-trajectory crosscoding, which fits one sparse dictionary to every token's readout row at every checkpoint. Unlike crosscoders on activations, it needs only checkpoint weights. We use it to locate a narrow early window, 0.4-1.1% of the way through Pythia's pretraining, in which the readout reorganizes and WordNet word categories become decodable from its rows. By then, linear probes on the hidden states already decode
    
[^183]: 相似的模型，失灵的胜率：Luce公理在Chatbot竞技场中的失效之处

    Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena

    [https://arxiv.org/abs/2610.19095](https://arxiv.org/abs/2610.19095)

    该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。

    

    Chatbot竞技场使用Bradley-Terry模型对语言模型进行排名，该模型为每个模型赋予单一分数，因而隐含地假设了Luce选择公理。选择理论预言了该公理会在何处失效：近似复制的模型被区分的果断程度会超出其分数所暗示的水平。基于180万条竞技场投票数据，同一模型不同版本之间的投票果断程度是Bradley-Terry预测的1.37倍，同一代不同参数规模之间为1.26倍，同一实验室不同模型之间为1.12倍；一年后的新发布模型也呈现出一致的乘数，且该效应在替代性的平局处理方式、提示词子集与主题划分以及时间检验下依然稳健。两个模型回答在措辞和格式上的相似程度能够追踪这一效应。一个仅含四到六个参数的修正模型在预测留出集投票方面优于Bradley-Terry模型以及拥有数百个参数的模型，这主要归功于对平局的建模；在预先设定的测试中，将该修正模型冻结并应用于新发布的模型时，其收益集中于平局情形，并在成对比较中表现更优。

    arXiv:2610.19095v1 Announce Type: new  Abstract: Chatbot Arena ranks language models with the Bradley-Terry model, which gives each model one score and so assumes Luce's choice axiom. Choice theory predicts where this fails: near-copies are told apart more decisively than their scores imply. On 1.8 million Arena votes, votes between versions of one model are 1.37 times as decisive as Bradley-Terry predicts, between sizes of one generation 1.26 times and between models of one lab 1.12 times; a release a year later gives consistent multipliers, and the effect survives alternative tie treatments, prompt subsets and topics, and timing checks. How alike two models' answers are worded and formatted tracks the effect. A correction with four to six parameters predicts held-out votes better than Bradley-Terry and than models with hundreds of parameters, mostly by modelling ties; frozen and carried to new releases in a test fixed in advance, its gains are concentrated in ties, and better head-to
    
[^184]: PATTERN基准可通过记忆破解：在未见过的模式上测试图神经网络

    PATTERN Can Be Solved from Memory: Testing Graph Neural Networks on Unseen Patterns

    [https://arxiv.org/abs/2610.15020](https://arxiv.org/abs/2610.15020)

    PATTERN基准因训练与测试重复使用相同的100个模式而可被“记忆”破解——简单的查找表就能超越所有已发表的GNN结果，而重新训练的图神经网络在未见过的模式上性能大幅下滑，说明该基准无法衡量模型的真正泛化能力。

    

    PATTERN要求图神经网络在新的宿主图中找到植入的子图。宿主图会变化，但该基准仅抽取了100个模式，并在训练集与测试集中重复使用这些模式。一种从训练标签中提取这些模式的查找方法，在官方测试集上达到了96.6%的类别加权准确率，高于所有已发表的结果，且无需拟合模型或查询生成器。而在未见过的模式上，同样的查找方法仅得50.6%。一个同时利用生成器的匹配器在官方测试集上可达到99.2%。在我们重新训练的五个图神经网络中，GRASS和PPGT从官方测试集上的前两名跌至未见模式上的最后两名：即使熟悉模式与未见模式的规模相匹配，它们仍各自损失8.1个百分点。基于两跳节点统计的逻辑回归在未见模式上的得分高于这五个网络的所有运行结果。对于CKGConv而言，其优势依赖于训练模式：当……

    arXiv:2610.15020v2 Announce Type: replace  Abstract: PATTERN asks graph neural networks to find a planted subgraph in a new host graph. The hosts change, but the benchmark draws just 100 patterns and reuses them across training and test splits. A lookup that extracts these patterns from the training labels achieves 96.6% class-weighted accuracy on the official test split, above every published result, without fitting a model or consulting the generator. On unseen patterns, the same lookup scores 50.6%. A matcher that also uses the generator reaches 99.2% on the official split. Among five graph networks we retrained, GRASS and PPGT go from the top two on the official split to the bottom two on unseen patterns. Each loses 8.1 percentage points even when familiar and unseen patterns are matched in size. Logistic regression on two-hop node statistics scores above every run of the five networks on unseen patterns. For CKGConv, the advantage follows the training patterns: it reappears with e
    
[^185]: 相似的预测拟合但不同的潜在动力学：刻画脑部疾病个性化模型中学习到的动力学结构

    Similar Predictive Fit but Different Latent Dynamics: Characterizing Learned Dynamical Structure in Personalized Models of Brain Disorders

    [https://arxiv.org/abs/2610.10850](https://arxiv.org/abs/2610.10850)

    该研究提出将EEG基础模型的潜在表示映射为个性化转移依赖图的方法，发现即使预测拟合度相似，癫痫患者的潜在动力学依赖结构也显著比非癫痫者更密集，表明模型学到的动力学结构能揭示超越预测准确率的临床相关差异。

    

    随着AI模型走向临床决策和个性化治疗，理解模型学到了什么，其重要性已超越单纯的预测准确率。我们研究当预测拟合度相似时，个性化的潜在动力学是否仍能揭示与临床相关的差异。一个在坦普尔大学EEG语料库（TUEG）上预训练的轻量级CNN-Transformer EEG基础模型提取片段级表示。利用坦普尔大学癫痫语料库（TUEP），将表示映射到共享的潜在状态空间，并对每名受试者独立拟合稀疏的多项逻辑斯蒂转移分布，从而获得个性化的转移依赖图 W_n。分析共纳入198名受试者（99名癫痫/99名非癫痫）。在k=4时，癫痫受试者表现出明显更密集的学习依赖结构（p=1.1×10^-7），在k=6时呈现相同模式（19.90 vs. 13.46；p...

    arXiv:2610.10850v1 Announce Type: new  Abstract: As AI models move toward clinical decision-making and personalized treatment, understanding \emph{what} a model learns is important beyond predictive accuracy alone. We investigate whether personalized latent dynamics reveal clinically associated differences even when predictive fit is similar. A lightweight CNN--Transformer EEG foundation model pretrained on the Temple University EEG Corpus (TUEG) extracts segment-level representations. Using the Temple University Epilepsy Corpus (TUEP), representations are mapped to a shared latent-state space, and sparse multinomial logistic transition distributions (mLTD) are fit independently to each subject to obtain personalized transition-dependency graphs $W_n$. Analyses include $n{=}198$ subjects (99 epilepsy / 99 non-epilepsy). At $k{=}4$, epilepsy subjects exhibit substantially denser learned dependency structure ($p{=}1.1\times10^{-7}$), with the same pattern at $k{=}6$ (19.90 vs. 13.46; $p{
    
[^186]: 基于调和权重的高维在线校准

    High-dimensional online calibration from harmonic weights

    [https://arxiv.org/abs/2610.07740](https://arxiv.org/abs/2610.07740)

    本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。

    

    我们研究了在任意凸集 $Y\subseteq\mathbb{R}^d$ 上、相对于任意误差范数 $\|\cdot\|_{L}$ 的多维预测在线校准问题。对于同时预测 $d$ 个二元结果（$Y=[0,1]^d$）的情形，我们给出了首个在每个固定精度下都能以关于 $d$ 为多项式的轮数实现 $\varepsilon$-校准的算法。该算法需要 $d^{O(1/\varepsilon)}$ 轮，相比此前界中的维度依赖性实现了指数级改进。对于多类别预测（$Y=\Delta_d$），我们获得了相同的 $d^{O(1/\varepsilon)}$ 速率，改进了 Peng 以及 Fishelson 等人给出的 $d^{\widetilde{O}(1/\varepsilon^2)}$ 界。我们的算法非常简单：在每一轮，它输出一个对过去结果进行调和平滑后的调和加权分布。同一个算法适用于所有预测集和范数。更一般地，它在 $\exp(O(\gamma(Y,L)/\varepsilon$……（原文摘要在此处截断）轮后即可实现 $\varepsilon$-校准。

    arXiv:2610.07740v1 Announce Type: cross  Abstract: We study the online calibration of multidimensional forecasts over an arbitrary convex set $Y\subseteq\mathbb{R}^d$ relative to an arbitrary error norm $\|\cdot\|_{L}$. For forecasting $d$ binary outcomes simultaneously ($Y=[0,1]^d$), we give the first algorithm that achieves $\varepsilon$-calibration in a number of rounds that is polynomial in $d$ for every fixed accuracy. It requires $d^{O(1/\varepsilon)}$ rounds, exponentially improving the dimension dependence of previous bounds. For multi-class forecasting ($Y=\Delta_d$), we obtain the same $d^{O(1/\varepsilon)}$ rate, improving the $d^{\widetilde{O}(1/\varepsilon^2)}$ bounds of Peng and Fishelson et al.   Our algorithm is simple: on each round, it outputs a harmonically weighted distribution over harmonically smoothed past outcomes. The same algorithm works for every forecast set and norm. More generally, it achieves $\varepsilon$-calibration after $\exp(O(\gamma(Y,L)/\varepsilon
    
[^187]: 面向冻结跨疾病迁移的轻量级语义脑电图基础模型

    Lightweight Semantic EEG Foundation Model for Frozen Cross-Disorder Transfer

    [https://arxiv.org/abs/2610.05503](https://arxiv.org/abs/2610.05503)

    该论文提出仅有146万参数（约为现有模型五分之一）的轻量级脑电图基础模型USE-FM，通过自监督信号重建学习可迁移神经表征，并在冻结编码器条件下于异常脑电检测和癫痫发作识别两类跨疾病任务上取得与大型模型相当的性能。

    

    大规模脑电图基础模型已在跨神经系统疾病的迁移中展现出良好的前景，但通常需要数百万参数和大量计算资源。在本文中，我们提出了通用语义脑电图基础模型（USE-FM），这是一种轻量级脑电图基础模型，通过在坦普尔大学医院脑电图语料库（TUEG）上进行自监督信号重建来学习可迁移的神经表征。预训练完成后，编码器被冻结，并在两个临床上截然不同的下游任务上进行评估：异常脑电检测（TUAB）和癫痫发作识别（TUEP），采用统一的冻结迁移协议与近期的脑电图基础模型（包括LUNA-Base和CBraMod）进行对比。USE-FM仅有146万参数，约为现有模型规模的五分之一，却取得了具有竞争力的整体性能，包括在TUEP上较强的敏感性和F1分数（SEN 75.00 ± 14.14，……（摘要至此处不完整）

    arXiv:2610.05503v2 Announce Type: replace  Abstract: Large-scale EEG foundation models have demonstrated promising transferability across neurological disorders, but often require millions of parameters and substantial computational resources. In this paper, we present the Universal Semantic EEG Foundation Model (USE-FM), a lightweight EEG foundation model that learns transferable neural representations through self-supervised signal reconstruction on the Temple University Hospital EEG Corpus (TUEG). After pretraining, the encoder is frozen and evaluated on two clinically distinct downstream tasks, abnormal EEG detection (TUAB) and epileptic seizure recognition (TUEP), using a unified frozen-transfer protocol against recent EEG foundation models, including LUNA-Base and CBraMod. With only 1.46 million parameters, approximately one-fifth the size of existing models, USE-FM achieves competitive overall performance, including strong sensitivity and F1-score on TUEP (SEN $75.00 \pm 14.14$,
    
[^188]: 世界模型应该遗忘什么？面向持续适应的分层保留机制

    What Should World Models Forget? Stratified Retention for Continual Adaptation

    [https://arxiv.org/abs/2610.03713](https://arxiv.org/abs/2610.03713)

    提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。

    

    持续学习将在已见数据上的性能退化视为失败的证据，这一惯例继承自预测目标平稳的设定，即正确标签会永远保持正确。世界模型不满足这一条件：它们的预测目标是环境，而环境会变化，因此获取时准确的知识之后可能变为错误，丢弃这些知识是必要行为而非缺陷。非平稳的真实标签问题在概念漂移文献和语言模型的时间事实性研究中已被充分探讨，但尚未在世界模型中被形式化，而世界模型的独特之处在于它们还编码了绝不能被修改的知识。我们认为，持续世界模型需要按不变性时间尺度分层的保留机制，将诸如物理规律和物体恒存性等绝不应被修改的不变量，与应随环境变化而尽快被更新的实例级事实区分开来。

    arXiv:2610.03713v1 Announce Type: cross  Abstract: Continual learning treats degradation on previously seen data as evidence of failure, a convention inherited from settings with a stationary prediction target, where a correct label remains correct indefinitely. World models do not satisfy this condition. Their prediction target is the environment, which changes, so knowledge that was accurate when acquired may later become false, and discarding it is required behavior rather than a defect. Non-stationary ground truth is well studied in the concept drift literature and in the temporal factuality of language models, but has not been formulated for world models, which are distinctive in that they also encode knowledge that must never be revised. We argue that continual world models require retention stratified by invariance timescale, separating invariants such as physics and object permanence, which must never be revised, from instance-level facts that should be revised as soon as the e
    
[^189]: 具有1比特反馈的近最优固定置信度最优臂识别

    Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

    [https://arxiv.org/abs/2610.02771](https://arxiv.org/abs/2610.02771)

    本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。

    

    我们研究在严格1比特反馈约束下的固定置信度最优臂识别问题。在每一轮中，学习者选择一个臂和一个查询集合，并且仅接收一个比特，该比特指示采样的奖励是否属于该集合。我们考虑一种具有逐臂定位的无分布有限方差设置，在这种设置下，直接的经验均值估计不再可用，截断处理变得不可避免。我们首先基于随机化阈值查询和截断尾积分恒等式，构建了一个时间一致的1比特均值估计基元。随后，我们将该基元嵌入到候选-挑战者式的最优臂识别算法中。固定截断算法提供了简单的任意时刻（ε,δ)-PAC保证，而分阶段自适应截断算法则将截断水平与当前分辨率相匹配，从而产生了间隙自适应的样本复杂度。我们还证明了一个K臂最坏情况的信息论下界（摘要原文在此处截断）。

    arXiv:2610.02771v1 Announce Type: cross  Abstract: We study fixed-confidence best-arm identification under strict 1-bit feedback constraints. At each round, the learner selects an arm and a query set, and receives only a single bit indicating whether the sampled reward belongs to that set. We consider a distribution-free finite-variance setting with arm-wise localization, where direct empirical mean estimation is no longer available and clipping becomes unavoidable. We first formulate a time-uniform 1-bit mean-estimation primitive based on randomized threshold queries and a clipped tail-integral identity. We then embed this primitive into candidate-challenger best-arm identification algorithms. A fixed-clipping algorithm gives a simple anytime $(\epsilon,\delta)$-PAC guarantee, while a phased adaptive-clipping algorithm matches the clipping level to the current resolution and yields a gap-adaptive sample complexity. We also prove a $K$-arm worst-case information-theoretic lower bound s
    
[^190]: 基于智能体编码任务强化学习的跨基准迁移

    Cross-Benchmark Transfer from RL on Agentic Coding Tasks

    [https://arxiv.org/abs/2610.00890](https://arxiv.org/abs/2610.00890)

    仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。

    

    编码智能体常常在“最后一公里”失败：它们完成了功能的大部分却遗漏某项需求、只测试其实现已经能处理的情况、破坏了本应保持不变的行为，或基于未经检验的假设进行验证。我们探究在专家构建的智能体编码任务上进行强化学习（RL）能否弥合这一差距，以及智能体所学内容能否迁移到训练分布之外。我们仅使用强化学习对 Kimi K2.7 Code（一个 1 万亿参数、激活参数 32B 的开源混合专家模型）在 1,700 个任务上进行后训练：其中 1,000 个仓库任务由隐藏的 fail-to-pass 测试和保护既有行为的 pass-to-pass 测试评分，700 个终端任务由专家编写的隐藏验证器评分。奖励为目标检查的通过比例，若任何 pass-to-pass 测试失败则奖励降为零。在秩为 32 的 LoRA 适配器上进行一个 epoch 的 GSPO 训练，在我们评估的六个外部基准上的 pass@1 均有提升，跨越三个……（原文摘要在此处截断）

    arXiv:2610.00890v1 Announce Type: cross  Abstract: Coding agents often fail in the last mile: they build most of a feature but drop a requirement, test only the cases their implementation already handles, break behavior that was supposed to stay intact, or validate against an unchecked assumption. We ask whether reinforcement learning (RL) on expert-built agentic coding tasks closes this gap, and whether what the agent learns transfers beyond the training distribution. We post-train Kimi K2.7 Code, a 1T-parameter (32B active) open-weight mixture-of-experts model, with RL alone on 1,700 tasks: 1,000 repository tasks graded by hidden fail-to-pass tests and by pass-to-pass tests of existing behavior, and 700 terminal tasks graded by expert-written hidden verifiers. The reward is the fraction of target checks passed and drops to zero if any pass-to-pass test fails. One epoch of GSPO on a rank-32 LoRA adapter improves pass@1 on each of the six external benchmarks we evaluated, across three 
    
[^191]: 基于StyleGAN中间潜在空间的储层物性参数化方法用于基于集合的数据同化

    Parameterization method of reservoir properties for ensemble-based data assimilation using intermediate latent space of StyleGAN

    [https://arxiv.org/abs/2609.39626](https://arxiv.org/abs/2609.39626)

    本研究提出利用StyleGAN2的中间潜在空间对储层物性进行参数化，以克服集合数据同化方法在处理非高斯复杂地质（复杂相分布）条件下历史拟合性能下降的问题，并与VAE-GAN和潜在扩散模型等深度学习参数化方案进行了对比评估。

    

    集合平滑方法是当前最成功、最高效的历史拟合技术。然而，由于这些方法依赖于高斯假设，当先验地质以复杂相分布（非高斯）描述时，其性能会严重下降。因此，对于这些方法，需要应用高效的参数化技术。目前，执行参数化最有效的方法是深度学习模型。然而，鉴于现有深度学习模型的多样性，尽管已经评估过一些重要模型，但研究尚未确定哪种模型最适合与基于集合的方法配合使用。基于近期的文献综述，选出的最有前景的模型是VAE-GAN、潜在扩散模型和StyleGAN模型。作为本工作的创新之处，使用第二代StyleGAN（StyleGAN2）模型进行了数据同化，利用其

    arXiv:2609.39626v1 Announce Type: cross  Abstract: Ensemble smoothers are the most successful and efficient techniques currently available for history matching. However, because these methods rely on Gaussian assumptions, their performance is severely degraded when the prior geology is described in terms of complex facies distributions (non-Gaussian). In this way, for these methods, we need to apply efficient parameterization techniques. Currently, the most efficient methods for performing parameterization are deep learning models. However, given the variety of existing deep learning models, studies have not identified which is most suitable for use with ensemble-based methods, although some important models had already been evaluated. Based on a recent literature review, the most promising models selected were VAE-GAN, Latent Diffusion, and StyleGAN models. As a novel aspect of this work, data assimilation with the second generation of StyleGAN (StyleGAN2) model was performed using th
    
[^192]: 当平坦度代理并非函数时：鲁棒性证书与训练干预

    When a Flatness Proxy Is Not a Function: Robustness Certificates and Training Interventions

    [https://arxiv.org/abs/2609.38540](https://arxiv.org/abs/2609.38540)

    该论文证明，最后一层相对平坦度代理的有效曲率上界既不能作为鲁棒性证书的依据，也不能作为训练干预的依据，并提出规范不变的特征空间修复与行中心化两种补救方法。

    

    一个有效的曲率上界既不必能为鲁棒性证书提供依据，也不必能为对预测器内在属性的干预提供依据。我们针对在上述两种场景中均被使用的最后一层相对平坦度代理，论证了这一区别。首先，经验风险的平稳性并不能消除逐点的一阶损失项：在一个有限的全局经验风险最小值点处，所保留的证书表达式对损失增加的低估超过210倍。我们推导出了一个全局有效且规范不变的特征空间修复方法。其次，逐行共同的softmax平移在保持预测结果和精确收缩不变的同时，会使该代理变得无界。即使是标准的参考类选择，相对于中心化表示，平均而言也会使该代理的值加倍。对于一个具有至少三个类别的单一固定特征示例，标量重调一般无法使诱导的概率更新相互对齐。行中心化给出了轨道最小化的界，并恢复了数值以及整个模型的梯度……

    arXiv:2609.38540v1 Announce Type: new  Abstract: A valid curvature upper bound need not justify either a robustness certificate or an intervention on an intrinsic predictor property. We demonstrate this distinction for a last-layer relative-flatness proxy used in both settings. First, empirical-risk stationarity does not eliminate pointwise first-order loss terms: at a finite global empirical-risk minimum, the retained certificate expression underestimates a loss increase by over $210\times$. We derive a globally valid, gauge-invariant feature-space repair. Second, common-row softmax shifts preserve predictions and the exact contraction while making the proxy unbounded. Even standard reference-class choices double it on average relative to the centered representation. For a single fixed-feature example with at least three classes, scalar retuning generically cannot align the induced probability updates. Row centering gives the orbit-minimized bound and restores value and full-model gra
    
[^193]: 基于预测状态的无穷记忆过程生成式序列建模

    Generative sequence modeling for infinite memory processes via predictive states

    [https://arxiv.org/abs/2609.38524](https://arxiv.org/abs/2609.38524)

    本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。

    

    我们考虑估计多元随机过程的一步预测条件分布。许多现有方法依赖于有限范围记忆、稀疏性或可加性等假设，这些假设对于具有长程非线性交互作用的过程可能并不适用。然而，在没有这些结构性假设的情况下，由于维数灾难，非参数估计面临巨大挑战。为了应对这一挑战，我们提出了一种基于过程预测状态的新估计方法，该方法可以处理可能具有无穷范围记忆的过程。我们证明了当过去的历史信息能够被压缩成一个足以预测未来的低维统计量时，我们的估计器能够达到快速收敛速度。具体而言，我们证明了该估计问题的统计复杂度由预测状态空间的内在维数决定。我们为基于该方法的一种具体实现建立了理论保证（摘要在此处被截断）。

    arXiv:2609.38524v1 Announce Type: new  Abstract: We consider estimating the one-step-ahead conditional distribution of a multivariate stochastic process. Many existing approaches rely on assumptions such as finite-range memory, sparsity, or additivity, which can be poorly suited to processes with long-range nonlinear interactions. However, without such structural assumptions, nonparametric estimation is challenging due to the curse of dimensionality. To address this challenge, we introduce a new estimation approach based on the predictive states of a process, possibly with infinite-range memory. We show that our estimator achieves fast convergence rates when the past history can be compressed into a low-dimensional statistic that is sufficient for predicting the future. Specifically, we show that the statistical complexity of the estimation problem is determined by the intrinsic dimension of the predictive state space. We establish guarantees for an instantiation of our method based on
    
[^194]: 从数据到程序：从经验数据快速直接地推断生成程序

    From Data to Program: Fast & Direct Generative Program Inference from Empirical Data

    [https://arxiv.org/abs/2609.35348](https://arxiv.org/abs/2609.35348)

    PRODiGI是一种预训练的数据到程序模型，能在单次前向传播中从经验数据直接推断出显式可执行的生成程序，配合程序空间微调在密度和分数估计上超越现有预训练模型并获得数倍加速。

    

    从有限样本集合中估计概率密度通常需要针对特定数据集进行模型拟合。我们提出PRODiGI，这是一个预训练的数据到程序模型，能够在单次前向传播中推断出显式、可执行的生成程序。PRODiGI在合成数据集与其真实程序的配对数据上进行预训练，通过模板预测和非自回归的程序参数解码，能够适应多种生成模型族和数据维度。其推断出的程序支持直接采样、密度与分数评估以及程序检查，且独立于预训练模型运行。我们进一步引入了程序空间微调方法，在保持模型参数不变的前提下，通过最大化经验数据似然来优化可微的程序参数。实验表明，PRODiGI-FT比现有预训练模型取得了更优的密度和分数指标，同时相比迭代方法提供了数倍的加速。

    arXiv:2609.35348v2 Announce Type: replace-cross  Abstract: Estimating probability densities from a finite set of samples typically requires dataset-specific model fitting. We introduce PRODiGI, a pretrained data-to-program model that infers an explicit, executable generative program in a single forward pass. Pretrained on synthetic datasets paired with their ground-truth programs, PRODiGI accommodates diverse generative families and data dimensionalities through template prediction and non-autoregressive program parameter decoding. Its inferred programs support direct sampling, density and score evaluation, and inspection independently of the pretrained model. We further introduce program-space fine-tuning, which refines differentiable program parameters by maximizing the likelihood of empirical data while keeping model parameters intact. Experiments show that PRODiGI-FT achieves better density and score metrics than existing pretrained models, while offering multi-fold speedups over i
    
[^195]: ReCo：在部署约束下何时迁移传感器套件——一个NILM案例研究

    ReCo: When to Relocate Sensor Kits under Deployment Constraints -- A NILM Case Study

    [https://arxiv.org/abs/2609.35075](https://arxiv.org/abs/2609.35075)

    提出了一种基于约束的传感器套件迁移决策框架ReCo，通过在目标-背景联合特征空间中统计新运行状态并预测各住宅的未来数据增益，解决NILM数据采集中“何时迁移”的决策问题。

    

    许多感知任务只能通过在现场部署仪器来获取训练标签。在传感器套件数量有限、存在采集截止期限、且每次迁移都会造成测量停机的情况下，采集者必须反复决定是留在当前站点还是进行迁移。我们在非侵入式负荷监测（NILM）中研究这一决策问题。NILM通过住宅的主电表估计各个电器的用电量，其训练数据来自临时安装了电器级子电表的住宅。在NILM中，电器的使用情况随电器类型、季节和气候而变化，新数据的价值取决于目标运行状态与背景负荷组合的多样性程度。为解决这一问题，我们提出了一个基于约束的迁移决策框架，并针对NILM将其实现为ReCo（基于覆盖增益的迁移）。ReCo在联合的目标-背景特征空间中统计新的运行状态，并预测每个住宅未来可带来的增益……

    arXiv:2609.35075v2 Announce Type: replace  Abstract: Many sensing tasks obtain training labels only by deploying instruments in the field. With a limited number of sensor kits, a collection deadline, and measurement downtime at every move, the collector must repeatedly decide whether to stay at the current site or relocate. We study this decision in non-intrusive load monitoring (NILM), which estimates the power drawn by individual appliances from a home's main meter and is trained on data from homes temporarily fitted with appliance-level sub-meters. In NILM, appliance usage varies with the appliance, season and climate, and the value of new data depends on how diverse the combinations of target operation and background load are. To address this, we propose a constraint-based relocation framework and instantiate it for NILM as ReCo (Relocation by Coverage gain). ReCo counts new operating regimes in a joint target--background feature space, forecasts each home's future gain from the da
    
[^196]: 多模态大语言模型中的对齐幻觉

    The Alignment Illusion in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.30210](https://arxiv.org/abs/2609.30210)

    该研究通过对13个多模态大语言模型施加受控干预，揭示了标准标量相似度度量无法真实反映跨模态内容交互的“对齐幻觉”现象，其根源在于MLP下投影所导致的权重诱导对齐。

    

    多模态大语言模型（MLLM）中的层级视觉-文本相似度被广泛解读为语言模型逐步将视觉内容整合进共享表示空间的证据。这种解读建立在这样一个假设之上：标量对齐分数反映的是内容层面的跨模态交互。为了检验这一假设，我们对视觉流施加了受控干预。在来自五个模型家族、参数量从0.5B到72B的13个MLLM上，用高斯噪声替换投影器输出的视觉token会急剧降低任务准确率，然而四种标准标量度量（CKA、SVCCA、MIR以及主角度余弦）却无法一致地区分被破坏的视觉流与原始视觉流。我们将这种失败称为“对齐幻觉”，并将其根源追溯至共享的语言模型路径：各向异性的MLP下投影将视觉token和文本token拉向共同的输出方向，从而产生了由权重引起的对齐……

    arXiv:2609.30210v1 Announce Type: cross  Abstract: Layer-wise visual-text similarity in Multimodal Large Language Models (MLLMs) is widely interpreted as evidence that the language model progressively integrates visual content into a shared representation space. This reading rests on the assumption that scalar alignment scores reflect content-level cross-modal interaction. To test this assumption, we apply controlled interventions to the visual stream. Across 13 MLLMs from five families spanning 0.5B to 72B parameters, replacing projector-output visual tokens with Gaussian noise sharply reduces task accuracy, yet four standard scalar measures (CKA, SVCCA, MIR, and the leading principal-angle cosine) fail to consistently separate the corrupted stream from the original. We call this failure the alignment illusion and trace it to the shared language-model pathway: anisotropic MLP down-projections pull visual and text tokens toward common output directions, producing weight-induced alignme
    
[^197]: UltraBench 2：迈向超声领域视觉基础模型的鲁棒评估

    UltraBench 2: Towards Robust Evaluation of Vision Foundation Models on Ultrasound

    [https://arxiv.org/abs/2609.28610](https://arxiv.org/abs/2609.28610)

    该论文推出了UltraBench 2，一个覆盖广泛解剖结构和任务、注重标准化与可重复性的超声视觉基础模型综合评估基准，并发现超声特异性预训练在分类任务上仍占优势，而最先进的通用模型在分割任务上已与之持平。

    

    基准测试已成为机器学习研究及其应用领域（包括医疗保健）中日益关键的一部分。然而，尽管近年来新的超声基础模型在稳步发展，用于评估这些模型的设计良好的基准测试的发展却相对滞后。这一不足导致了对相互竞争的模型进行评估时碎片化且不一致，使得难以衡量研究进展。为了解决这个问题，我们推出了UltraBench 2，这是一个具有广泛解剖结构和任务覆盖范围的综合基准，专注于标准化、可重复性和易用性。利用该基准，我们比较了现有的用于超声图像分析的视觉基础模型。我们的分析表明，超声特异性预训练在分类任务上仍然领先，但最先进的通用模型在分割任务上已经与之持平。

    arXiv:2609.28610v1 Announce Type: cross  Abstract: Benchmarking is an increasingly critical part of research in machine learning and the domains where it is applied, including healthcare. Yet, despite the steady development of new ultrasound foundation models in recent years, the development of well-designed benchmarks to evaluate them has lagged behind. This deficiency has led to fragmented and inconsistent evaluations of competing models, making it difficult to measure progress. To address this issue, we introduce UltraBench 2, a comprehensive benchmark with wide anatomical and task coverage, and a focus on standardization, reproducibility, and ease-of-use. Using this benchmark, we compare existing vision foundation models for ultrasound image analysis. Our analyses demonstrate that ultrasound-specific pretraining still leads on classification, but that state-of-the-art general-purpose models have drawn level on segmentation.
    
[^198]: 隐藏而非删除：网络如何抑制纠缠特征

    Hidden not Deleted: How Networks Suppress Entangled Features

    [https://arxiv.org/abs/2609.27593](https://arxiv.org/abs/2609.27593)

    该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。

    

    通过线性投影实现的概念擦除方法假设特征占据可分离的子空间。我们证明该假设在密集叠加情况下会失效：当两个特征被迫形成共享同一子空间的对跖对时，最先进的线性擦除方法会同时破坏两者，而不仅仅是目标特征。通过梯度下降训练的网络则以非线性方式解决这一问题，但方式并不统一：根据初始化的不同，它们会收敛到两种不同的电路级解决方案之一，我们称之为“镜像”解决方案和“阴影”解决方案。我们将这种分叉现象映射为特征纠缠程度的函数，证明它反映的是稳定的吸引子结构而非实验设置的伪影，并通过针对性的因果干预证明，这两种解决方案都会在被擦除特征的表征中留下大量可测量的完整痕迹，仅需一个标量补丁即可恢复，而无需任何进一步的训练。这一现象类似于——（摘要原文在此处截断）

    arXiv:2609.27593v1 Announce Type: cross  Abstract: Concept erasure methods that operate via linear projection assume that features occupy separable subspaces. We show this assumption fails under dense superposition: when two features are forced into an antipodal pair sharing a single subspace, state-of-the-art linear erasure destroys both, not just the target. Networks trained with gradient descent instead solve this problem non-linearly, but not uniformly: they converge to one of two distinct circuit-level solutions depending on initialization, which we call mirror and shadow solutions. We map this bifurcation as a function of feature entanglement, show it reflects a stable attractor structure rather than an artifact of our setup, and use targeted causal interventions to demonstrate that both solutions leave a substantial, measurable trace of the erased feature's representation intact, recoverable through a single scalar patch rather than requiring any further training. This mirrors a
    
[^199]: 超越嵌入迁移：Grokking迁移与稳定性中的组件角色

    Beyond Embedding Transfer: Component Roles in Grokking Transfer and Stability

    [https://arxiv.org/abs/2609.18078](https://arxiv.org/abs/2609.18078)

    该论文通过大规模控制实验首次系统区分了Grokking热启动迁移中各模型组件的角色，发现迁移内部注意力/MLP权重不仅可将早期准确率提升5.46个百分点并缩短确认延迟，且该优势在双层模型的前瞻性复现实验中得到完全验证。

    

    热启动迁移可以使算法任务快速泛化，但目前尚不清楚哪些模型组件提供了这种增益，以及该增益在持续优化下是否保持稳定。我们在模运算上研究跨算子迁移，并将有效性（早期速度）与稳定性（达到后回撤）区分开来。在一个规模匹配的108次实验组合中（涵盖12个种子块，包括96次2³因子实验和12次规模对照），同时迁移内部注意力/MLP权重（B）与词元嵌入和读出层（E+U）可将早期准确率提高5.46个百分点（Holm校正 p=0.0039），并将确认延迟减少558步（Holm校正 p=0.0088）。虽然读出层加内部块迁移在单层模型中满足预先设定的±500步延迟等效性标准（TOST p=0.0011，尽管在11/12的配对种子中完整迁移更快），但前瞻性的双层复现实验证实了内部块的优势（12/12种子，+704.67积分单位，p=4.88×10⁻⁴）。

    arXiv:2609.18078v1 Announce Type: new  Abstract: Warm-start transfer can make algorithmic tasks generalize rapidly, yet it is unclear which model components provide the gain and whether that gain remains stable under continued optimization. We study cross-operator transfer on modular arithmetic and separate efficacy (early velocity) from stability (post-reach drawdown). In a scale-matched 108-run battery across 12 seed blocks (96-run 2^3 factorial plus 12-run scale control), transferring internal attention/MLP weights (B) alongside token embeddings and readout (E+U) improves early accuracy by 5.46 pp (Holm p=0.0039) and cuts confirmation latency by 558 steps (Holm p=0.0088). While readout plus internal-block transfer satisfies the pre-specified +/-500-step latency equivalence criterion in 1-layer models (TOST p=0.0011, though Full is faster in 11/12 paired seeds), a prospective 2-layer replication confirms the internal-block advantage (12/12 seeds, +704.67 integral units, p=4.88x10^-4)
    
[^200]: 基于能量移动距离的相似性配对方法用于LHC自监督预训练

    Similarity Pairing with Energy Mover's Distance for Self-Supervised Pre-Training at the LHC

    [https://arxiv.org/abs/2609.17738](https://arxiv.org/abs/2609.17738)

    该论文提出一种利用能量移动距离（EMD）按相似性配对真实对撞事件作为数据增强视图的方法，无需人工构造或模拟增强事件即可保持事件物理真实性，从而改进LHC自监督预训练。

    

    在大型强子对撞机（LHC）上训练基础模型的许多自监督方法依赖于数据增强，以鼓励模型将事件嵌入到对某些物理或探测器对称性保持不变的表示空间中。一个常见的挑战在于选择合适的增强方法集合具有很大的自由度，而下游性能正依赖于这些增强。数据增强的实现要么需要修改现有事件，可能破坏事件的真实性（保真度），要么需要模拟更多的事件变体，这在计算上非常昂贵。在本工作中，我们提出了一种数据驱动的方法，通过能量移动距离（EMD）根据相似性对事件进行配对——EMD通过衡量将一个事件变换为另一个事件所需的“功”来度量两个事件的相似程度。通过这种方法，可以采样不同的真实事件并根据其相似性进行匹配，作为学习不变性的视图，同时保持事件本身的物理内容……

    arXiv:2609.17738v1 Announce Type: cross  Abstract: Many self-supervised methods for training foundation models at the Large Hadron Collider (LHC) rely on data augmentations to encourage the model to embed events into a representation space invariant to certain physical or detector symmetries. A common challenge arises from the large freedom in choosing a proper set of augmentations on which downstream performance depends. The implementation of augmentations involves either modifying existing events, potentially breaking the event fidelity, or simulating more event variants, which is computationally intensive. In this work, we present a data-driven method of pairing events by their similarity via the energy mover's distance (EMD), which measures how similar two events are in terms of the work required to transform one into the other. With this approach, distinct events are sampled and matched by their similarity to serve as views for learning invariance, keeping the physics content of e
    
[^201]: 用于级联衰变中鲁棒共振质量回归的自监督学习

    Self-Supervised Learning for Robust Resonance Mass Regression in Cascade Decays

    [https://arxiv.org/abs/2609.17726](https://arxiv.org/abs/2609.17726)

    该论文提出遵循基础模型范式，先用VICReg自监督预训练transformer编码器以学习对各种数据破坏不变的嵌入表示，再微调用于级联衰变中重共振粒子的质量回归，从而在系统不确定性和分布偏移下实现鲁棒的质量重建。

    

    从带有丢失能量的衰变产物中重建重共振粒子的质量，是直接决定对撞机实验中新物理搜索灵敏度的核心任务之一。由于存在各种系统不确定性和分布偏移，针对该问题的监督学习方法通常难以很好地泛化。穷尽标注数据中所有可能的变化可能非常耗费计算资源，而模型泛化的失败则会破坏重建的共振峰宽度，而该宽度在寻峰分析中至关重要。在本工作中，我们遵循基础模型范式，采用自监督方法，使用VICReg预训练一个transformer编码器，以学习对各种破坏不变的嵌入表示，随后对其进行微调，用于重共振粒子的质量回归，其中共振粒子质量范围为2.5至6.5 TeV，并采用类似SUSY的级联衰变至十一体末态。我们展示了……（摘要在此处截断）

    arXiv:2609.17726v1 Announce Type: cross  Abstract: Reconstructing the mass of a heavy resonance from its decay products with missing energy is one of the central tasks that directly determine the sensitivity in new physics searches at collider experiments. Supervised learning approaches to this problem often struggle to generalize well due to the presence of various systematic uncertainties and distribution shifts. Exhausting all possible variations in the labeled data can be very compute-intensive, while a failure of the model to generalize can corrupt the reconstructed resonance widths that are critical in peak-hunting analyses. In this work, following the foundation model paradigm, we use a self-supervised approach to pre-train a transformer encoder with VICReg to learn an embedding invariant to various corruptions, then fine-tune it for mass regression on a heavy resonance with masses ranging from 2.5 to 6.5 TeV and a SUSY-like cascade decay into an eleven-body final state. We show
    
[^202]: 一种基于梯度但依赖脉冲时序的神经微回路反馈学习问题解决方案

    A Gradient-based yet Spike-Timing-Dependent Solution to the Feedback Learning Problem in Neural Microcircuits

    [https://arxiv.org/abs/2609.08070](https://arxiv.org/abs/2609.08070)

    本文提出梯度隧穿算法，将时间信用分配重构为状态分离问题，从局部突触脉冲时序推导信用分配，实现了与生物脉冲时序一致的神经微回路在线反馈学习框架。

    

    大脑利用离散脉冲进行动态计算，然而神经微回路如何利用局部脉冲时序来解决时间信用分配问题仍然是一个基本的开放性问题。主流的脉冲神经网络（SNN）方法通过代理梯度近似反向传播来规避这一问题，使学习与生物脉冲时序解耦。在此，我们将时间信用分配重新表述为一个状态分离问题：直接从当前神经状态中提取由历史扰动引起的任务所需成分。这使得通过梯度隧穿（GT）算法和超前-滞后展开技术，为神经微回路构建了在线反馈学习框架，该技术从局部突触脉冲时序中推导信用分配，同时保持与ANN-SNN混合架构的兼容性。实验表明，经GT训练的神经微回路在长时间尺度的证据整合和抗噪记忆保持方面表现出色，并且能够执行……

    arXiv:2609.08070v1 Announce Type: cross  Abstract: The brain uses discrete spikes for dynamic computation, yet, how neural microcircuits (NMCs) solve temporal credit assignment using local spike timing remains a fundamental open question. Dominant spiking neural network (SNN) approaches circumvent this by approximating backpropagation through surrogate gradients, decoupling learning from biological spike timing. Here, we reformulate temporal credit assignment as a state separation problem: extracting task-required components induced by historical perturbations directly from the current neural state. This enables an online feedback learning framework for NMCs through a gradient tunneling (GT) algorithm and the lead-lag expansion technique that derives credit assignment from local synaptic spike timing, while remaining compatible with ANN-SNN hybrid architectures. Experimentally, GT-trained NMCs excel at long-timescale evidence integration and noise-robust memory retention, and perform c
    
[^203]: 归因Cohen's d：常模年龄生物标志物中疾病相关效应的训练数据归因

    Attributing Cohen's d: Training Data Attribution for Disease-Related Effects in Normative Age Biomarkers

    [https://arxiv.org/abs/2609.07729](https://arxiv.org/abs/2609.07729)

    本文提出一种基于Cohen's d的闭式影响泛函，可直接将常模年龄模型中疾病相关的年龄差距效应归因到个体训练样本，并在UK Biobank上证明移除最有影响力的训练样本可显著提高疾病相关效应量。

    

    常模年龄模型在名义上健康的队列中进行训练，用于预测实际年龄（实际年龄）。当应用于患者时，模型会产生偏差，预测年龄与实际年龄之间的差距被解读为疾病风险。本文将年龄差距的疾病相关效应量直接归因于个体训练样本，而不是以预测级别的损失作为归因目标。对于Cohen's d，由此得到的闭式影响泛函（经留一法重训练验证）可根据训练样本对留出数据的病例-对照分离效果的影响对其进行排序。在英国生物银行（UK Biobank）的四种疾病和两种生物标志物模态上，移除最具影响力的10%训练样本在所有随机种子下均能提高留出的疾病相关效应量：对于2型糖尿病，代谢组学年龄效应增加了一倍以上；对于多发性硬化症，脑年龄效应提高了约三分之一。而随机移除即使达到50%，效应量也保持不变，这证实了……

    arXiv:2609.07729v1 Announce Type: new  Abstract: Normative age models are trained to predict chronological age in a nominally healthy cohort. Applied to patients, they deviate, and the gap between predicted and chronological age is read as disease risk. Here, we attribute the disease-related effect size of the age gap directly to individual training samples, rather than using a prediction-level loss as the attribution target. For Cohen's $d$, the resulting closed-form influence functional, validated against leave-one-out retraining, ranks training samples by their effect on held-out case-control separation. Across four diseases and two biomarker modalities in UK Biobank, removing the 10% most influential training samples raises held-out disease-related effect size in every seed. It more than doubles the metabolomic-age effect for type-2 diabetes and raises the brain-age effect for multiple sclerosis by roughly a third. Random removal leaves effect size flat even at 50% removal, confirm
    
[^204]: 通过功能感知掩码学习任务特异性抗体表示

    Learning Task-Specific Antibody Representations via Function-Aware Masking

    [https://arxiv.org/abs/2609.00518](https://arxiv.org/abs/2609.00518)

    该论文提出功能感知掩码这一预训练算法家族，通过将掩码位置与特定功能先验（如IMGT注释或结构预测）对齐来学习任务特异性的抗体表示，在结构相关任务上最高提升14%，在CDR相关任务上最高提升5.9倍。

    

    通过掩码语言建模（MLM）预训练的抗体特异性语言模型，学习到的表示对下游序列设计和性质预测任务至关重要。然而，掩码破坏过程本身在预训练期间很少被用作归纳偏置的来源。虽然优先掩码互补决定区（CDRs）能够改善与结合相关的预测，但抗体在多种功能上拥有多样的生物学先验。本文中，我们提出了功能感知掩码，这是一类将掩码位置与特定功能先验（例如来自IMGT注释或结构预测的先验）对齐的预训练算法，用以塑造学习到的表示空间。我们证明，这些针对特定功能的掩码策略在各自的目标任务上显著提升了性能，在结构相关任务上带来高达14%的性能提升，在CDR相关任务上带来高达5.9倍的改进。为了进一步提高性能……

    arXiv:2609.00518v1 Announce Type: new  Abstract: Antibody-specific language models pretrained via masked language modeling (MLM) learn representations that are critical for downstream sequence design and property prediction tasks. Yet, the corruption process itself is rarely leveraged as a source of inductive bias during pretraining. While preferentially masking complementarity-determining regions (CDRs) improves binding-related predictions, antibodies possess diverse biological priors over a variety of functions. Herein, we introduce function-aware masking, a family of pretraining algorithms that align mask placement with specific functional priors (e.g., from IMGT annotations or structure predictions) to shape the learned representation space. We show that these specialist masking strategies significantly improve performance on their respective objectives, yielding up to a 14% gain on structure-related tasks and up to a 5.9x improvement on CDR-related tasks. To further improve perfor
    
[^205]: 门罗：用于上下文概率推断的分子基础模型

    Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference

    [https://arxiv.org/abs/2608.18982](https://arxiv.org/abs/2608.18982)

    门罗是一种新型分子基础模型，通过扩大预训练规模、改进立体化学表示、优化损失函数和多任务学习，并结合TabPFN实现上下文概率推断，显著提升了数据受限下的生物活性预测性能。

    

    arXiv:2608.18982v1 公告类型：新 摘要：生物活性测定预测常常受限于数据量，因为药物发现数据集依赖于耗时且昂贵的湿实验室实验来进行数据生成和评估。这一挑战激发了近期对分子基础模型（MFMs）的研究，这些模型旨在将通用化学知识编码到分子表示中，使其在数据受限场景下具有良好的泛化能力。本文介绍了门罗（Monroe），一种新型MFM，相对于现有技术具有多项创新：扩大规模，可在PM6量子化学数据集的超过8100万个分子上进行预训练；改进立体化学的图表示；改进训练损失，包括构象去噪和嵌入去相关；改进多任务学习；以及使用先验数据拟合模型（TabPFN）进行下游上下文预测。我们的评估采用一种原则性的配对比较框架，该框架测量统计显著性。

    arXiv:2608.18982v1 Announce Type: new  Abstract: Bioassay activity prediction is often data-limited because drug-discovery datasets rely on time-consuming and expensive wet-lab experiments for data generation and evaluation. This challenge has inspired recent research into molecular foundation models (MFMs), which aim to encode general-purpose chemical knowledge into molecular representations that generalize well in data-constrained scenarios. This paper presents Monroe, a new MFM with several innovations over the existing state of the art: increased scale allowing pre-training on over 81 million molecules from the PM6 quantum chemistry dataset; improved graph representation of stereochemistry; improved training losses including conformer denoising and embedding decorrelation; improved multi-task learning; and the use of a prior-data-fitted model (TabPFN) for downstream in-context prediction. Our evaluations use a principled pairwise comparison framework that measures statistically sig
    
[^206]: BBOWP-Bench：评估大语言模型在黑箱优化文字问题上的能力

    BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems

    [https://arxiv.org/abs/2608.02612](https://arxiv.org/abs/2608.02612)

    该论文提出BBOWP-Bench基准，首次评估大语言模型能否从自然语言描述的黑箱优化文字问题中同时推断出合适的搜索空间并选择恰当的优化算法。

    

    优化问题的建模方式会强烈影响最终解的质量，然而良好的建模通常需要大量专业知识。近期研究已探索从自然语言描述中自动推导优化问题，但现有基准主要针对目标与约束能够显式写成数学表达式的场景。许多实际重要的问题天然适合被视为黑箱优化（BBO）问题——其中只有目标函数值可观测，而函数形式不可获得。在BBO中，搜索空间设计（作为问题建模的一部分）与优化算法的选择对求解至关重要，而利用大语言模型（LLM）自动化这些过程仍具挑战性。我们提出了黑箱优化文字问题，这是一个新颖的问题设定：系统需要从自然语言描述的问题中同时推断出搜索空间并选择优化算法。

    arXiv:2608.02612v2 Announce Type: replace  Abstract: Formulating an optimization problem strongly affects final solution quality, yet good formulations usually require substantial expertise. Recent studies have explored automatically deriving optimization problems from natural-language descriptions, but existing benchmarks focus on settings where objectives and constraints can be written explicitly as mathematical expressions. Many practically important problems are naturally treated as black-box optimization (BBO) problems, in which only objective values are observable, and the functional form is unavailable. In BBO, search space design (as part of problem formulation) and optimization algorithm selection are crucial for problem-solving. Automating these processes with large language models (LLMs) remains challenging. We introduce Black-Box Optimization Word Problems (BBOWP), a novel problem setting in which a system must infer both a search space and an optimization algorithm from a 
    
[^207]: 同策略交互何时有益？基于价值的模仿学习中的表征权衡

    When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning

    [https://arxiv.org/abs/2607.29617](https://arxiv.org/abs/2607.29617)

    该论文发现，在基于价值的模仿学习中，与专家进行同策略交互能放宽对学习者的表征要求——只需模型能实现专家的价值函数，而无需实现要求更严格的专家策略本身。

    

    模仿学习（IL）——即训练智能体从演示中复制专家行为——是机器人技术到语言模型训练等诸多应用的基础。众所周知，行为克隆（BC）等标准方法存在误差累积和性能停滞的问题，尤其当学习者无法完美表征专家策略时（例如在蒸馏场景中这是典型情况）。两种干预措施在经验上被广泛认为可以提升性能：沿学习者自身轨迹交互式地查询专家，以及在生成策略的过程中使用价值函数估计。我们研究了这些改进的本质及其可能令人惊讶的相互作用。我们的主要发现是，与专家的交互放宽了对学习者的表征要求：只需要一个能够实现专家价值函数的模型，从而绕过了（通常更严格的）实现专家策略的要求。（注：摘要原文在此处被截断）

    arXiv:2607.29617v2 Announce Type: replace-cross  Abstract: Imitation learning (IL)---training an agent to replicate expert behavior from demonstrations---underpins applications from robotics to language model training. Standard approaches such as Behavior Cloning (BC) are known to suffer from compounding errors and performance plateaus, particularly when the learner cannot perfectly represent the expert's policy (as is typical, e.g., in distillation). Two interventions are widely understood empirically to improve performance: querying the expert interactively along the learner's own trajectories, and using value function estimation en route to generating a policy.   We investigate the nature of these improvements and their potentially surprising interplay. Our main finding is that expert interaction relaxes the representational demands on the learner: one only needs a model capable of realizing the expert's value function, bypassing the (often stricter) requirement of realizing the exp
    
[^208]: 无需融合的异构深度集成多模态分类

    Fusion-Free Multimodal Classification with Heterogeneous Deep Ensembles

    [https://arxiv.org/abs/2607.05019](https://arxiv.org/abs/2607.05019)

    该论文证明，无需任何显式模态融合，仅用单模态网络的深度集成在相同参数量下即可持续超越为解决模态不平衡而设计的最先进后期融合及其他融合方法。

    

    在多模态分类中，后期融合方法对由单模态神经网络提取并拼接的模态特定特征进行分类。当模态不平衡现象显著时，已有多种正则化技术被提出以平衡学习过程并克服后期融合网络性能不佳的问题。相比之下，这项工作表明，多模态数据无需任何显式模态融合即可被有效分类，只需使用单模态网络的深度集成。我们在相同参数量的条件下系统地将深度集成与后期融合网络进行比较，结果显示，在从随机初始化开始训练的情况下，深度集成始终优于为解决模态不平衡而设计的最先进的后期融合方法。这一优势在我们所评估的中间融合技术以及结合单模态和多模态分类器的混合方法上同样成立。我们提出并实证验证了一种方法……

    arXiv:2607.05019v3 Announce Type: replace  Abstract: In multimodal classification, late-fusion approaches classify concatenated modality-specific features extracted by unimodal neural networks. When modality imbalance is pronounced, various regularization techniques have been proposed to balance the learning process and overcome the inferior performance of late-fusion networks. In contrast, this work demonstrates that multimodal data can be effectively classified without any explicit modality fusion, using deep ensembles of unimodal networks. We systematically compare deep ensembles to late-fusion networks at equal parameter count and show that ensembles consistently outperform state-of-the-art late-fusion methods designed to address modality imbalance when trained from random initialization. This advantage also holds over intermediate-fusion techniques we evaluated and over hybrid methods that combine unimodal and multimodal classifiers. We propose and empirically validate a method fo
    
[^209]: 输入通路塑造微型Transformer的小样本绑定而非零样本绑定：一项完全可枚举的研究

    Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study

    [https://arxiv.org/abs/2607.04926](https://arxiv.org/abs/2607.04926)

    该研究通过对微型Transformer进行完全可枚举实验发现，无论信息以符号token、神谕代码还是纠缠感知向量的形式输入，模型都无法实现零样本组合绑定（其失败源于归纳偏置而非信息缺失），但在小样本设置下，绑定学习的样本效率由输入代码的可读性决定。

    

    信息以何种方式到达Transformer——是以符号token的形式、清晰的逐因子“神谕”代码，还是纠缠的感知向量——会如何影响它能否对这些信息进行组合式绑定？我们在有限因子世界上研究了约6-10K参数的微型Transformer，并对整个输入空间进行穷举枚举，因此每次测量都覆盖全部输入空间，且各信息通路在信息量上是精确匹配的（贝叶斯准确上限为1.0）。研究得出四个发现：(1) 终点不变性：在留出的绑定查询上，没有任何一条信息通路能够达到收敛的零样本组合能力——每条通路都终止于随机水平或以下，因此这种失败反映的是在查表即可充分完成目标情况下的归纳偏置问题，而非信息缺失。即使将状态放在其自身的序列位置、而不是加到每个token上，该结论依然成立。(2) 可读性决定小样本绑定：当留出类型的一小部分数据泄漏进训练集时，样本效率与代码可读性成正比。可读性最差的……（原文摘要在此处截断）

    arXiv:2607.04926v2 Announce Type: replace-cross  Abstract: How does the way information reaches a transformer -- as symbolic tokens, a clean per-factor "oracle" code, or an entangled perceptual vector -- shape whether it binds that information compositionally? We study ~6-10K-parameter transformers on finite factored worlds enumerated exhaustively, so every measurement covers the whole input space and the informative routes are information-matched (exact Bayes ceiling 1.0). Four findings. (1) Endpoint invariance: on held-out binding queries no informative route reaches converged zero-shot composition -- each ends at or below chance, so the failure reflects inductive bias under a lookup-sufficient objective, not missing information. It also holds with the state in its own sequence position rather than added to every token. (2) Readability governs few-shot binding: when a small fraction of the held-out type is leaked into training, sample efficiency tracks code readability. The least rea
    
[^210]: URSA：面向实用性逆合成评估的化学感知基准

    URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment

    [https://arxiv.org/abs/2607.04688](https://arxiv.org/abs/2607.04688)

    URSA提出了一个化学感知的逆合成评估基准框架，首次从形式指标（如收敛到商业可得起始原料）和模拟专家化学家判断的化学合理性两个维度，对端到端逆合成模型与大语言模型进行客观、可解释的比较评估。

    

    合成规划旨在为目标分子寻找可行的反应路径，是药物发现中最重要且最具挑战性的任务之一。近期的进展催生了专用的深度学习逆合成系统和通用的大型语言模型，但由于缺乏灵活且具备化学可解释性的基准测试协议，对它们进行客观比较仍然十分困难。在本研究中，我们提出了URSA（实用性逆合成评估）评估框架，该框架不仅能够从形式角度（例如是否收敛到可商业购买的起始原料）对合成路线进行基准评估，还能从化学合理性角度进行评估，模拟专家化学家评估反应与路线的方式。该研究对传统端到端逆合成解决方案和大语言模型在合成规划任务上进行了全面评估（原文摘要在此处截断）。

    arXiv:2607.04688v2 Announce Type: replace-cross  Abstract: Synthesis planning aiming to find pathways of reactions for a target molecule is one of the most important and challenging tasks in drug discovery. Recent progress has produced both specialized deep-learning retrosynthesis systems and general-purpose large language models, but objective comparison remains difficult due to the lack of flexible, chemically interpretable benchmarking protocols. In the current study, we are introducing the URSA (Utilitarian RetroSynthesis Assessment) evaluation framework that provides the opportunity to benchmark the synthetic routes not only from a formal perspective, such as convergence to commercially available starting materials, but also from a chemical plausibility perspective, mimicking the way expert chemists evaluate the reactions and routes. The study covers a comprehensive evaluation of both conventional end-to-end retrosynthesis solutions and LLMs for the synthesis planning task on a se
    
[^211]: MMAO-Cls：用于特征选择与分类器调优联合优化的代谢多智能体优化方法

    MMAO-Cls: Metabolic Multi-Agent Optimization for Joint Feature Selection and Classifier Tuning

    [https://arxiv.org/abs/2607.01539](https://arxiv.org/abs/2607.01539)

    提出MMAO-Cls，一种基于代谢多智能体优化的混合空间方法，将特征选择与分类器超参数调优联合编码于每个智能体中，并通过特征信息先验和过拟合正则化加以增强，在七个表格基准上展现出与GA-lite等基线方法相当且更具竞争力的性能。

    

    本文研究了代谢多智能体优化器（MMAO）能否作为分类模型选择的可靠外循环优化器。我们提出了MMAO-Cls，这是一种混合空间实现，其中每个智能体同时编码二进制特征掩码和分类器超参数，而私有能量、公共预算、角色漂移和生命周期更替则被映射到包装器学习的精度-复杂度权衡上。该实现通过从特征信息先验中推导特征预算适应机制，并利用子集紧凑性和训练-验证过拟合差距对验证奖励进行正则化而得到加强。我们在七个标准表格基准上对MMAO-Cls进行了评估，每个基准使用三个随机种子，并将其与RandomSearch、GA-lite、PSO-lite以及内生无共享消融方法进行比较。在总体验证目标上，MMAO-Cls排名第二（0.9433），仅次于GA-lite（0.9446）。在保留测试集性能上，（摘要在此截断）

    arXiv:2607.01539v3 Announce Type: replace-cross  Abstract: This paper studies whether the Metabolic Multi-Agent Optimizer (MMAO) can act as a credible outer-loop optimizer for classification model selection. We propose MMAO-Cls, a mixed-space realization in which each agent jointly encodes a binary feature mask and classifier hyperparameters, while private energy, communal budget, role drift, and lifecycle turnover are mapped to the accuracy-complexity tradeoff of wrapper learning. The implementation is strengthened by deriving feature-budget adaptation from feature-information priors and by regularizing validation reward with both subset compactness and train-validation overfitting gap. We evaluate MMAO-Cls on seven standard tabular benchmarks with three seeds each and compare it against RandomSearch, GA-lite, PSO-lite, and an endogenous no-sharing ablation. On the aggregate validation objective, MMAO-Cls ranks second ($0.9433$) behind GA-lite ($0.9446$). On held-out test performance,
    
[^212]: QVal：评估长程LLM智能体的密集监督信号

    QVal: Evaluating Dense Supervision Signals for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2606.32034](https://arxiv.org/abs/2606.32034)

    该论文提出QVal——一个无需训练的测试平台，通过衡量密集监督信号在每个状态下对候选动作的排序与参考策略回报排序的一致性（Q-对齐性）来直接评估信号质量，并在四个环境中对七个方法族的21种方法进行基准测试，发现简单提示基线平均Q-对齐性最高且性能按方法族高度聚集。

    

    LLM智能体越来越多地在长时程任务中执行行动，而仅基于结果的奖励几乎无法说明哪些中间动作是好的。密集监督方法通过为中间步骤打分来解决这一问题，但这类方法通常通过完整训练流程的下游性能来评估。这种方式成本高昂，将信号质量与其他训练选择混为一谈，也意味着这些方法很少在共同基准上进行比较。我们提出了QVal，一个无需训练的测试平台，用于衡量密集信号的$Q$-对齐性：即该信号在每个状态下对候选动作的排序是否与其在参考策略下的回报排序一致。我们在四个环境中，使用六个开源权重骨干模型，对来自七个方法族的21种方法进行了基准测试。结果表明，简单的提示基线平均而言$Q$-对齐性最高，且性能按方法族高度聚集。代码生成的信号仅在小状态和动作空间中表现良好；基于嵌入的信号则……

    arXiv:2606.32034v2 Announce Type: replace-cross  Abstract: LLM agents increasingly act over long horizons, where outcome-only rewards say little about which intermediate actions were good. Dense supervision methods address this by scoring intermediate steps, but they are usually evaluated through the downstream performance of a full training pipeline. This is expensive, confounds signal quality with other training choices, and means that methods are rarely compared on common ground. We introduce QVal, a training-free testbed that measures the $Q$-alignment of a dense signal: whether it orders the candidate actions in each state the same way as their returns under a reference policy do. We benchmark 21 methods from seven families with six open-weight backbones in four environments. Simple prompting baselines are the most $Q$-aligned on average, and performance clusters strongly by family. Code-generated signals do well only in small state and action spaces; embedding-based signals are w
    
[^213]: 解耦连续时间潜在动力学：通过扩散系数变化实现潜在随机微分方程的可辨识性

    Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts

    [https://arxiv.org/abs/2606.28228](https://arxiv.org/abs/2606.28228)

    该论文证明了在未知非线性观测下，仅利用多个环境间扩散协方差的变化即可识别加性噪声潜在SDE的潜在坐标（至置换、缩放和常数平移），且无需对漂移项作任何稀疏性假设。

    

    面向时间序列的因果表征学习在离散时间潜在因果模型中已建立了较强的可辨识性结果，但连续时间潜在随机微分方程（SDE）模型中的可辨识性问题在很大程度上仍悬而未决。我们利用环境引起的扩散协方差变化来填补这一空白。我们研究了通过未知非线性微分同胚观测到的加性噪声潜在SDE，其漂移项在不同环境间共享，而扩散协方差则因环境而异。我们证明：两个具有逐坐标成对不同方差比的对角扩散机制，可以在无需对漂移项施加任何稀疏性假设的条件下，将潜在坐标识别至置换、逐坐标缩放以及可能的常数平移的等价类内。我们首先针对线性Ornstein-Uhlenbeck系统证明了这一结果，随后将其推广至一般的加性噪声潜在SDE。在温和的光滑性条件下，瞬时漂移-雅可比因果图是可辨识的……

    arXiv:2606.28228v2 Announce Type: replace-cross  Abstract: Causal representation learning for time series has developed strong identifiability results in discrete-time latent causal models, but identifiability in continuous-time latent stochastic differential equation (SDE) models remains largely open. We address this gap using environment-induced shifts in diffusion covariance. We study additive-noise latent SDEs observed through an unknown nonlinear diffeomorphism, with shared drift but environment-specific diffusion covariance. We show that two diagonal diffusion regimes with pairwise distinct coordinate-wise variance ratios identify the latent coordinates up to permutation, coordinate-wise scaling, and a possible constant shift, without any sparsity assumption on the drift. We first prove this result for linear Ornstein-Uhlenbeck systems and then extend it to general additive-noise latent SDEs. Under mild smoothness, the instantaneous drift-Jacobian causal graph is identifiable up 
    
[^214]: 一种针对特定预测期、基于属性条件化的神经网络，用于在不规则纵向数据中估计健康状态转移概率：LANTERN 框架

    A Horizon-Specific Attribute-Conditioned Neural Network for Estimating Health-State Transition Probabilities in Irregular Longitudinal Data: The LANTERN Framework

    [https://arxiv.org/abs/2606.13880](https://arxiv.org/abs/2606.13880)

    本文提出 LANTERN 框架，通过整合循环健康表示、观测间隔时间与自适应的人口社会经济属性条件化，从不规则纵向数据中估计特定预测期的健康状态转移概率，为长期护理保险建模提供了超越传统精算多状态模型的新方法。

    

    准确估计健康状态转移概率对于长期护理保险建模至关重要。传统的精算多状态模型提供了可解释的框架，但在捕捉非线性关系以及从不规则纵向观测中积累的健康历史方面可能存在局限。本文提出 LANTERN，这是一个依赖历史的神经框架，它整合了循环健康状态表示、观测间隔时间，以及自适应的人口统计与社会经济属性条件化，用于估计健康、轻度失能、重度失能和死亡四种状态之间的特定预测期转移概率。利用健康与退休研究（Health and Retirement Study）的纵向数据，我们在一年、两年和三年的预测期上，将所提出的模型与循环神经网络、多项逻辑回归、梯度提升树以及精算连续时间马尔可夫链进行了对比评估。在主要的两年预测期上……

    arXiv:2606.13880v2 Announce Type: replace  Abstract: Accurate estimation of health-state transition probabilities is essential for long-term care insurance modeling. Conventional actuarial multi-state models provide an interpretable framework but may be restrictive in capturing nonlinear relationships and accumulated health histories from irregular longitudinal observations. This paper proposes LANTERN, a history-dependent neural framework that integrates recurrent health representations, elapsed observation time, and adaptive demographic and socioeconomic attribute conditioning to estimate horizon-specific transition probabilities among Healthy, Mild disability, Severe disability, and Death states. Using longitudinal Health and Retirement Study data, we evaluate our model against a recurrent neural network, multinomial logistic regression, gradient-boosted trees, and an actuarial continuous-time Markov chain across one-, two-, and three-year forecast horizons. At the primary two-year 
    
[^215]: Express 语言建模

    Express Language Modeling

    [https://arxiv.org/abs/2606.10944](https://arxiv.org/abs/2606.10944)

    Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。

    

    我们介绍了一种新工具 Express，用于将非因果注意力近似转换为具有匹配近似保证的因果近似。当与最先进的 Thinformer 近似相结合时，Express 改进了已知最佳的因果注意力保证：对于长度为 n 的序列，在仅使用 O(s) 内存和 O(s² log²(n)) 压缩开销的情况下，实现了 log^{3/2}(n)/s 的近似误差。我们将这些进展与高效的 I/O 感知 Triton 实现相结合，展示了相对于 FlashAttention 2 的显著加速，并利用 Express 克服了语言建模流程中的四个资源瓶颈：长上下文预填充、KV 缓存压缩、长文本内存受限解码以及长文本计算受限解码。

    arXiv:2606.10944v2 Announce Type: replace  Abstract: We introduce a new tool, Express, for converting a non-causal attention approximation into a causal approximation with matching approximation guarantees. When combined with the state-of-the-art Thinformer approximation, Express improves upon the best known causal attention guarantees, delivering $\log^{3/2}(n)/s$ approximation error with only $O(s)$ memory and $O(s^2 \log^2(n))$ compression overhead for a sequence of length $n$. We pair these developments with an efficient I/O-aware Triton implementation, demonstrate substantial speedups over FlashAttention 2, and use Express to overcome four resource bottlenecks in the language modeling pipeline: long-context prefill, KV cache compression, long-form memory-constrained decoding, and long-form compute-constrained decoding.
    
[^216]: Larch：面向语义谓词的学习型查询优化

    Larch: Learned Query Optimization for Semantic Predicates

    [https://arxiv.org/abs/2606.07923](https://arxiv.org/abs/2606.07923)

    Larch利用非结构化数据嵌入所承载的语义信息，通过学习型优化技术高效优化AI SQL查询中语义过滤器的执行。

    

    随着大语言模型（LLM）的出现，许多数据库系统引入了语义算子，使分析查询能够应用于非结构化数据（如文本、图像、视频）。语义算子通常伴随较高的推理成本和延迟，使得语义（AI）SQL查询难以应用于大规模数据集。与此同时，其语义特性导致数据库引擎将其视为黑盒，使AISQL查询难以优化。本文提出Larch，一个用于优化AI SQL查询中语义过滤器执行过程的框架。Larch的灵感来自两个关键观察：i) 语义算子的高延迟为计算密集型的运行时优化技术留下了巨大的空间；ii) 非结构化数据通常伴随着以嵌入（embedding）形式存在的语义信息，从而能够在AI_FILTER提示与数据值之间进行高效的语义比较。

    arXiv:2606.07923v2 Announce Type: replace-cross  Abstract: With the advent of Large Language Models (LLMs), many database systems introduced semantic operators that enabled analytical queries over unstructured data (e.g. text, images, videos). Semantic operators typically incur high inference costs and latencies making semantic (AI) SQL queries challenging to apply on large scale datasets. At the same time, their semantic nature leads database engines to treat them as black boxes, making AISQL queries difficult to optimize. In this paper, we introduce Larch, a framework for optimizing the execution of semantic filters in AI SQL queries. Larch was inspired by two key observations: i) the high latency of semantic operators leaves significant room for computationally-heavy runtime optimization techniques, ii) unstructured data are typically accompanied by semantic information in the form of embeddings allowing for efficient semantic comparisons between AI_FILTER prompts and data values. B
    
[^217]: 基于非平衡最优传输的柔性尺寸设计生成式分子变形

    Generative Molecular Morphing for Flexible-Size Design via Unbalanced Optimal Transport

    [https://arxiv.org/abs/2606.07239](https://arxiv.org/abs/2606.07239)

    提出了Morph——一个基于非平衡最优传输的柔性尺寸3D分子生成模型，通过动态调整原子数目实现分布外生成，并显著增强对分子性质的引导能力。

    

    生成式分子设计的成功取决于模型向高奖励样本引导的能力。由于许多分子性质与分子大小本质相关，准确捕捉分子性质与原子数目之间的联合分布至关重要。然而，当前的扩散模型和流模型固定了原子数目，这最终限制了它们探索这种复杂关系的能力。为解决这一问题，我们提出了Morph，一个基于几何图的、可用于条件和无条件3D分子设计的柔性尺寸生成模型。通过动态调整尺寸，Morph能够无缝整合现有的结构先验（如分子骨架），并显著增强了性质导向能力。我们证明Morph在性能上与当前固定尺寸的最先进模型相当，同时显著提升了采样灵活性。我们展示了在以往模型无法胜任的区域中进行分布外生成，为……（摘要原文在此处截断）

    arXiv:2606.07239v2 Announce Type: replace  Abstract: The success of generative molecular design depends on a model's steerability toward high-reward samples. Because many molecular properties are intrinsically linked to molecular size, accurately capturing the joint distribution of properties and the number of atoms is essential. However, current diffusion and flow-based models fix the number of atoms, which ultimately limits their ability to navigate this complex relationship. To address this, we introduce Morph, a flexible-size generative model for conditional and unconditional 3D molecular design based on geometric graphs. By dynamically adapting size, Morph can seamlessly integrate existing structural priors, like scaffolds, and significantly enhances property steering. We show that Morph matches current fixed-size state-of-the-art models while significantly increasing sampling flexibility. We demonstrate out-of-distribution generation in regimes where previous models fail, paving 
    
[^218]: 不确定性下基于证据引导的神经架构选择用于个体化血糖预测

    Evidence-Guided Neural Architecture Selection under Uncertainty for Subject-Specific Blood Glucose Forecasting

    [https://arxiv.org/abs/2606.05373](https://arxiv.org/abs/2606.05373)

    提出了EVIDENT框架，通过结合贝叶斯训练、基于证据的排序和不确定性下的任务特定验证，自动为1型糖尿病个体化血糖预测选出满足验证准则且容量最低的TCN架构。

    

    在数据有限、含噪且异构的时间序列预测任务中，可靠的神经架构选择仍是一个悬而未决的挑战，标准的启发式架构设计与验证方法无法确保预测的准确性、可靠性及泛化能力。我们提出了EVIDENT（基于证据的神经架构识别），一个用于架构选择的框架，它集成了贝叶斯训练、基于证据的排序以及不确定性下的任务特定验证。该框架探索候选架构池，并识别出满足预定验证准则的容量最低的模型。我们使用时间卷积网络（TCN）在1型糖尿病患者的个体化血糖预测任务中对该方法进行了验证。结果表明，EVIDENT在人群水平的糖尿病数据上系统性地拒绝参数不足和参数过多的TCN架构，同时为个体受试者识别出更合适的架构。

    arXiv:2606.05373v2 Announce Type: replace  Abstract: Reliable neural architecture selection is an open challenge in time-series forecasting under limited, noisy, and heterogeneous data, where standard heuristic architecture design and validation approaches fail to ensure accurate and reliable prediction and generalization. We propose EVIDENT (EVidence-based IDEntification of Neural archiTectures), a framework for architecture selection that integrates Bayesian training, evidence-based ranking, and task-specific validation under uncertainty. The framework explores the candidate architecture pool and identifies the lowest-capacity model that satisfies a prescribed validation criterion. We demonstrate this method using temporal convolutional networks (TCNs) for individualized blood glucose forecasting in type 1 diabetes patients. The results show that EVIDENT systematically rejects both under- and over-parameterized TCN architectures on population-level diabetes data, while identifying mo
    
[^219]: 基于PAC-Bayes风险的嵌套因果老虎机认证策略优化

    Certified Policy Optimisation for Nested Causal Bandits via PAC-Bayes Risk

    [https://arxiv.org/abs/2605.29788](https://arxiv.org/abs/2605.29788)

    本文提出嵌套上下文因果老虎机（NCCB）这一分层因果决策问题框架及嵌套因果汤普森采样（NCTS）算法，其核心贡献是一个因果PAC-Bayesian超额风险界，可仅凭历史数据、离策略且随时地对任意候选部署策略进行认证，回答“能否信任该智能体、风险多大”的部署问题。

    

    关键性序贯决策很少是单一时间尺度的：一个战略性决策会因果性地塑造后续每一个战术性选择所处的情境；标准的赌博机和强化学习理论无法捕捉这种时间尺度之间的因果耦合。我们将该问题类别形式化为嵌套上下文因果老虎机，这是一个分层结构因果模型（SCM），其中每一层的动作决定下一层的情境分布；并提出嵌套因果汤普森采样，它在每个回合抽取一个机制分解的信念，并在该信念下递归地行动。我们的主要理论结果是一个因果PAC-Bayesian超额风险界，仅凭历史数据即可对任何候选部署策略进行认证，支持离策略（off-policy）和随时（anytime）使用，从而回答部署问题：我们能否在此处信任这个智能体，其风险有多大？在分层SCM上的实验表明，与在同一函数类别上匹配的RFF-GP联合回归相比，该因子化方法……（摘要原文在此处截断）

    arXiv:2605.29788v2 Announce Type: replace  Abstract: Critical sequential decisions are rarely single-timescale: a strategic decision causally shapes the context in which every subsequent tactical choice is made; standard bandit and reinforcement-learning theory does not capture this causal coupling between timescales. We formalise the problem class as Nested Contextual Causal Bandits (NCCBs), a hierarchical SCM where each level's action sets the next level's context distribution, and propose Nested Causal Thompson Sampling (NCTS), which draws one mechanism-factorised belief per episode and acts recursively under it. Our main theoretical result is a causal PAC-Bayesian excess-risk bound that certifies any candidate deployment policy from historic data alone, off-policy and anytime, answering the deployment question: can we trust this agent here, and at what risk? Experiments on a hierarchical SCM show that, against a matched RFF-GP joint regression on the same function class, the factor
    
[^220]: 基于核的势均场博弈与无偏随机傅里叶U统计量

    Kernel-based potential mean-field games with unbiased random Fourier $U$-statistics

    [https://arxiv.org/abs/2605.29371](https://arxiv.org/abs/2605.29371)

    本文针对运行成本与终端成本均由核MMD惩罚刻画的势均场博弈，提出了一种基于无偏随机傅里叶U统计量的计算框架，以神经网络参数化漂移并通过随机梯度下降训练，同时证明了在惩罚发散及显式耦合条件下向终端约束问题的收敛性。

    

    我们研究势均场博弈的一个子类，其中运行交互成本和终端目标成本均通过再生核最大均值差异（MMD）惩罚来表示，并开发了一个利用这种核结构的计算框架。两种成本均通过一种无偏且计算成本随批大小线性增长的随机傅里叶U统计量表示，从有限样本经验分布中进行估计。受控扩散的漂移由神经网络参数化，并通过随机梯度下降进行训练。对于种群近极小化器，我们证明了当惩罚发散时收敛到终端约束问题，并表明在惩罚、随机特征数量和样本大小的显式耦合条件下，当学习到的控制策略在独立的有限样本上评估时，几乎必然恢复到相同的极限。该框架包括核MMD-（原文在此处截断）

    arXiv:2605.29371v2 Announce Type: replace-cross  Abstract: We study the subclass of potential mean-field games in which the running interaction cost and the terminal target cost are both expressed through reproducing-kernel maximum mean discrepancy (MMD) penalties, and develop a computational framework that exploits this kernel structure. Both costs are estimated from finite-sample empirical distributions using a random Fourier U-statistic representation that is unbiased and has linear cost in the batch size. The drift of the controlled diffusion is parametrized by a neural network and trained via stochastic gradient descent. For population near-minimizers we prove convergence to the terminal-constrained problem as the penalty diverges, and show that the same limit is recovered almost surely when the learned controls are evaluated on independent finite samples under explicit coupling conditions on the penalty, random-feature count and sample size. The framework includes the kernel-MMD-
    
[^221]: SA-RAG：为检索增强生成构建结构化的噪声对话

    SA-RAG: Structuring Noisy Conversations for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2605.24366](https://arxiv.org/abs/2605.24366)

    提出SA-RAG框架，通过质量感知的结构归纳方法直接从噪声对话数据中构建表格形式的可靠中间结构，从而提升检索增强生成在嘈杂上下文中的表现。

    

    大语言模型（LLM）已被广泛应用于对话应用中。然而，其对参数化知识的依赖限制了其在需要动态或领域特定信息的真实场景中的可靠性。检索增强生成（RAG）通过在生成过程中引入外部知识来应对这一局限，但现有的RAG方法往往难以处理噪声或无关的上下文，因为它们通常假设高质量的结构已经可用。在这项工作中，我们提出了结构感知检索增强生成（SA-RAG），这是一个有原则的结构归纳与生成框架，能够直接从噪声对话数据中构建可靠的中间结构。SA-RAG使用表格作为紧凑且可控的结构化表示。我们引入了一个质量感知的结构归纳框架，联合建模语义一致性和结构一致性。此外，我们还探索了……（摘要在此处被截断）

    arXiv:2605.24366v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have been widely adopted in conversational applications. However, their reliance on parametric knowledge limits reliability in real-world scenarios that require dynamic or domain-specific information. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge during generation, but existing RAG methods often struggle with noisy or irrelevant contexts, as they typically assume that high-quality structures are already available. In this work, we propose Structure-aware Retrieval-Augmented Generation (SA-RAG), a principled structure induction and generation framework that constructs reliable intermediate structures directly from noisy conversational data. SA-RAG uses tables as compact and controllable structured representations. We introduce a quality-aware structure induction framework that jointly models semantic and structural consistency. Furthermore, we explore bo
    
[^222]: AMUSE：具有稳定梯度评估的随时Muon优化器

    AMUSE: Anytime Muon with Stable Gradient Evaluation

    [https://arxiv.org/abs/2605.22432](https://arxiv.org/abs/2605.22432)

    该论文提出AMUSE优化器，通过“河谷”损失景观理论揭示了Muon正交化虽加速主体方向进展但会放大支配方向噪声引发振荡的问题，并结合稳定梯度评估与随时调度能力来加以改进。

    

    现代深度学习通常依赖于AdamW和预设的学习率调度，但近期的研究对这两个组件都提出了挑战：Schedule-Free优化通过迭代平均去除了显式调度，而Muon通过对矩阵参数的动量进行正交化来改进更新的几何结构。尽管Muon具有出色的实证性能，但其底层机制仍未被完全理解。我们通过“河谷”损失景观来研究Muon：在该景观中，有效的训练进展沿着平坦、低曲率的主体子空间（河流）进行，而高曲率的支配方向则形成陡峭的谷壁并引发振荡。我们通过实证表明，虽然Muon的正交化通过增加主体分量来加速河流进展，但它同时也放大了支配方向的噪声，导致轨迹出现振荡。基于这一发现，我们提出了具有稳定梯度评估的随时Muon优化器（AMUSE），该方法集成了Muon的快速主体……

    arXiv:2605.22432v3 Announce Type: replace  Abstract: Modern deep learning commonly relies on AdamW with prescribed learning rate schedules, but recent works challenge both components: Schedule-Free optimization removes explicit schedules via iterate averaging, and Muon improves the update geometry by orthogonalizing momentum for matrix parameters. Despite Muon's strong empirical performance, its underlying mechanism remains partially understood. We study Muon through the river-valley loss landscape, where useful training progress occurs along a flat, low-curvature bulk subspace (the river), while high-curvature dominant directions form steep valley walls that induce oscillations. We empirically show that while Muon's orthogonalization accelerates river progress by increasing the bulk component, it also amplifies dominant-direction noise, causing oscillatory trajectories. Building on this, we propose Anytime MUon with Stable gradient Evaluation (AMUSE), which integrates Muon's rapid bul
    
[^223]: TransitLM：面向无地图公交路线生成的大规模数据集与基准

    TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation

    [https://arxiv.org/abs/2605.22355](https://arxiv.org/abs/2605.22355)

    该论文发布了包含四个中国城市近1300万条路线规划记录的大规模数据集TransitLM，证明大语言模型可完全从数据中学习公交路线规划，无需地图即可实现从起终点信息到路线的端到端生成。

    

    公共交通路线规划传统上依赖于结构化地图基础设施和复杂的路由引擎，目前尚无数据集支持训练模型来摆脱这一依赖。我们提出了TransitLM，这是一个包含来自四个中国城市的近1300万条公交路线规划记录的大规模数据集，覆盖120,845个车站和13,666条线路，以持续预训练语料库的形式发布，并附带三个评估任务的基准数据及互补的评估指标。实验表明，在TransitLM上训练的大语言模型能够以高准确率生成结构有效的路线，并能在没有任何显式映射的情况下，隐式地将任意GPS坐标对齐到合适的车站。这些结果表明，公共交通路线规划可以完全从数据中学习，从而实现直接根据起终点信息进行端到端、无地图的路线生成。数据集和基准已在 https://huggingface.co/datasets/GD-ML/ 公开。

    arXiv:2605.22355v2 Announce Type: replace-cross  Abstract: Public transit route planning traditionally depends on structured map infrastructure and complex routing engines, and no existing dataset supports training models to bypass this dependency. We present TransitLM, a large-scale dataset of nearly 13 million transit route planning records from four Chinese cities covering 120,845 stations and 13,666 lines, released as a continual pre-training corpus and benchmark data for three evaluation tasks with complementary metrics. Experiments show that an LLM trained on TransitLM produces structurally valid routes at high accuracy and implicitly grounds arbitrary GPS coordinates to appropriate stations without any explicit mapping. These results demonstrate that transit route planning can be learned entirely from data, enabling end-to-end, map-free route generation directly from origin-destination information. The dataset and benchmark are available at https://huggingface.co/datasets/GD-ML/
    
[^224]: 基于效用约束随机聚合的联邦学习以提升理性参与

    Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation

    [https://arxiv.org/abs/2605.18020](https://arxiv.org/abs/2605.18020)

    提出了非货币化框架FedUCA，通过效用约束的随机聚合机制确保理性客户端的协作收益满足其本地效用阈值，从而提升客户端参与意愿、防止客户端流失导致的联邦训练崩溃。

    

    联邦学习算法隐含地假设客户端被动服从服务器端的调度，在服务器的请求下共享本地模型更新。然而，这忽视了现实跨机构环境中的一个重要方面：客户端往往是理性的智能体，可能将自身利益（如本地模型性能）置于全局模型利益之上。在统计异质性显著的场景中，如果协作带来的感知收益未能达到其本地效用阈值，理性客户端可能会选择退出联邦。这种流失会降低全局模型性能，并可能导致联邦训练过程的崩溃。为避免此类情况，本工作提出了FedUCA（Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation），这是一种非货币化框架，通过建立非支付渠道，使客户端能够……

    arXiv:2605.18020v2 Announce Type: replace  Abstract: Federated Learning (FL) algorithms implicitly assume that clients passively comply with server-side orchestration, sharing local model updates upon the server's request. However, this overlooks an important aspect of real-world cross-silo environments: clients are often rational agents who may prioritize their own interests, such as local model performance, over that of the global model. In settings with significant statistical heterogeneity, rational clients may opt out of the federation if the perceived benefits of collaboration fail to meet their local utility thresholds. Such attrition degrades global model performance and can lead to the collapse of the federated training process. To avoid such a scenario, in this work, we introduce FedUCA (Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation), a non-monetary framework that establishes a non-payment channel through which clients c
    
[^225]: 用于多尺度物理仿真的小波流匹配

    Wavelet Flow Matching for Multi-Scale Physics Emulation

    [https://arxiv.org/abs/2605.16573](https://arxiv.org/abs/2605.16573)

    提出小波流匹配（WFM），直接在多尺度小波空间中进行最优传输流匹配，无需预训练自编码器，在混沌流体仿真中以更低成本实现了更优的谱相干性与细节保持能力。

    

    对由偏微分方程（PDE）支配的多尺度物理系统进行精确仿真，要求模型在长时自回归滚动预测中保持稳定，同时保留细尺度结构。确定性仿真器会产生过度平滑的预测，而生成式方法虽能更好地捕捉细节但计算成本高昂。潜空间生成模型作为一种折中方案出现，但其代价是需要额外单独预训练的自编码器。我们提出小波流匹配（Wavelet Flow Matching, WFM），这是一种生成式仿真器，通过直接在多尺度小波空间中执行最优传输流匹配，克服了当前成本与性能之间的权衡。WFM 不学习潜在压缩，而是利用 U-Net 的层次结构，联合预测一个规定的无损小波表示的传输速度。在三个具有挑战性的混沌流体动力学系统上，WFM 实现了更优的谱相干性和具有竞争力的……（摘要原文在此处截断）

    arXiv:2605.16573v2 Announce Type: replace-cross  Abstract: Accurate emulation of multi-scale physical systems governed by PDEs demands models that remain stable over long autoregressive rollouts while preserving fine-scale structures. Deterministic emulators produce overly smoothed predictions, while generative approaches better capture details but are costly. Latent-space generative models have emerged as a compromise but with the additional cost of separately pre-trained autoencoders. We propose Wavelet Flow Matching (WFM), a generative emulator that overcomes current trade-offs between cost and skill by performing optimal-transport flow matching directly in the multi-scale wavelet space. Rather than learning a latent compression, WFM leverages the hierarchical structure of a U-Net to jointly predict the transport velocities of a prescribed, lossless wavelet representation. On three challenging systems of chaotic fluid dynamics, WFM achieves superior spectral coherence and competitiv
    
[^226]: 通过虚假正样本缓解捷径学习

    Shortcut Mitigation via Spurious-Positive Samples

    [https://arxiv.org/abs/2605.13340](https://arxiv.org/abs/2605.13340)

    提出一种无需数据标注或组平衡数据的新型捷径缓解方法，通过识别模型依赖虚假属性的少量实例，定位中间层中高度相关神经元并正则化其影响，使模型依靠真实有效特征进行预测，从而提升鲁棒性。

    

    捷径缓解策略通常依赖于训练数据标注、组平衡的保留数据，或者训练数据中存在所有组别，即（虚假）属性与类别的所有组合。然而，这些要求在实践中很少能够满足。我们转而提出一种有针对性的模型分析方法，用于识别模型依赖虚假属性的一小部分实例。利用该实例集，并遵循“此特征不应被用于预测”的推理逻辑，我们识别出中间层中高度相关的神经元，并对其影响进行正则化。这确保模型学会依赖有信息量的特征，而不是因为错误的原因而得出正确结果，从而在无需额外的平衡保留数据或标注的情况下提升模型的鲁棒性。

    arXiv:2605.13340v2 Announce Type: replace  Abstract: Shortcut mitigation strategies commonly rely on training data annotations, group-balanced held-out data or the presence of all groups, i.e., all combinations of (spurious) attributes and classes, in the training data. However, these requirements are rarely met in practice. We instead propose a method for targeted model analysis to identify a small set of instances in which the model relies on spurious attributes. Using that set and following ``this feature should not be used for prediction'' reasoning, we identify highly relevant neurons in an intermediate layer and regularize their impact. This ensures that models learn to depend on informative features rather than being right for the wrong reasons, thereby improving robustness without requiring additional balanced held-out data or annotations.
    
[^227]: 非平稳环境中的自适应校准

    Adaptive Calibration in Non-Stationary Environments

    [https://arxiv.org/abs/2605.11490](https://arxiv.org/abs/2605.11490)

    该论文提出了能够自动适应环境非平稳程度的在线预测校准算法，其校准误差可在独立同分布与对抗性两种情形之间平滑过渡，并在多种校准指标下提供了自适应保证。

    

    在现代人工智能系统中，做出经过校准的在线预测是一个核心挑战。现有文献大多关注完全对抗性的环境，其中结果可能是任意的，这导致算法过于保守，在更温和的环境中（例如结果近乎平稳的情形下）可能表现次优。这一空白引出了一个自然的问题：我们能否设计出这样的在线预测算法，使其校准误差能够自动适应环境的非平稳程度，并在独立同分布（i.i.d.）情形与对抗性情形之间平滑过渡？我们对这一问题给出了肯定的回答，并开发了一套在多种校准指标下均能实现自适应校准保证的算法。具体而言，设 T 为轮数，W 为环境中未知的独立同分布段数，C∈[0,T] 为另一个未知的非平稳性度量，其定义为最小的 ℓ1 偏差……（摘要原文在此处截断）

    arXiv:2605.11490v3 Announce Type: replace  Abstract: Making calibrated online predictions is a central challenge in modern AI systems. Much of the existing literature focuses on fully adversarial environments where outcomes may be arbitrary, leading to conservative algorithms that can perform suboptimally in more benign settings, such as when outcomes are nearly stationary. This gap raises a natural question: can we design online prediction algorithms whose calibration error automatically adapts to the degree of non-stationarity in the environment, smoothly interpolating between i.i.d. and adversarial regimes? We answer this question in the affirmative and develop a suite of algorithms that achieve adaptive calibration guarantees under multiple calibration measures. Specifically, with $T$ being the number of rounds, $W$ being the unknown number of i.i.d. segments of the environment, and $C\in[0,T]$ being another unknown non-stationary measure defined as the minimal $\ell_1$ deviation o
    
[^228]: 当自适应失败时：视觉语言提示学习中的门控坍塌基于梯度的诊断

    When Adaptation Fails: A Gradient-Based Diagnosis of Collapsed Gating in Vision-Language Prompt Learning

    [https://arxiv.org/abs/2605.09549](https://arxiv.org/abs/2605.09549)

    本文通过基于梯度的系统诊断研究，揭示了CLIP类视觉语言模型少样本提示学习中自适应门控坍塌的两种失败模式——梯度幅度失衡和门控退化，警示不应盲目增加架构复杂性。

    

    自适应提示机制被提出用于通过动态调整提示来增强视觉语言模型。然而，在基于CLIP类骨干网络的冻结少样本提示学习中，我们系统性地观察到自适应门控和提示选择模块经常发生坍塌：它们产生近乎恒定的输出，贡献可忽略的梯度信号，并且常常无法超越固定提示。为了进一步探索这一问题，我们提出了一项系统性的诊断研究，以揭示自适应失败的潜在原因和条件。通过跨数据集和多种提示学习架构的受控实验，我们识别出两种反复出现的失败模式：梯度幅度失衡和门控退化。我们的发现促使人们重新审视在参数高效学习中不加区分地增加架构复杂性的做法，并阐明了提示级自适应门控在这一领域中何时有效、何时无效。

    arXiv:2605.09549v2 Announce Type: replace  Abstract: Adaptive prompting mechanisms have been proposed to enhance vision-language models by dynamically tailoring prompts to inputs. However, in frozen few-shot prompt learning with CLIP-style backbones, we systematically observe that adaptive gates and prompt-selection modules often collapse: they produce nearly constant outputs, contribute negligible gradient signals, and frequently fail to outperform fixed prompts. To further explore this issue, we present a systematic diagnostic study to uncover the underlying causes and conditions of adaptation failure. Through controlled experiments across datasets and multiple prompt learning architectures, we identify two recurring failure modes: gradient magnitude imbalance and gate degradation. Our findings invite a re-examination of indiscriminately adding architectural complexity in parameter-efficient learning and clarify when prompt-level adaptive gating is, and is not, effective in this regi
    
[^229]: 基于权限的Android恶意软件检测中的领域偏移诊断与缓解

    Diagnosing and Mitigating Domain Shift in Permission-Based Android Malware Detection

    [https://arxiv.org/abs/2605.09028](https://arxiv.org/abs/2605.09028)

    本文系统诊断了基于权限的Android恶意软件检测器在跨域场景下的领域偏移问题，揭示了严重的非对称性能下降与特征重要性的跨域不稳定性，并提出了相应的缓解方法。

    

    基于机器学习的Android恶意软件检测器在现实世界部署中常常失效，原因在于领域偏移，即在某一数据源上训练的模型在来自另一数据源的应用上表现不佳。本文对基于权限的检测器在跨域条件下的泛化能力和可解释性进行了全面研究。使用两个互补的数据集（PerMalDroid和NATICUSdroid）以及五个集成分类器，我们首先建立了域内基线，模型准确率超过92%，随后量化了严重的不对称性能下降。虽然在PerMalDroid上训练的模型能够较好地泛化到NATICUSdroid（准确率86%），但反方向训练的模型准确率则急剧下降至73%。可解释AI分析揭示了特征分布的双峰特性，并表明特征重要性高度不稳定，关键权限在不同领域间会失去或获得影响力。不同领域的预测特征集……

    arXiv:2605.09028v5 Announce Type: replace  Abstract: Machine learning-based Android malware detectors often fail in real-world deployment due to domain shift, where models trained on one data source perform poorly on applications from another. This paper presents a comprehensive study on the generalizability and interpretability of permission-based detectors under cross-domain conditions. Using two complementary datasets (PerMalDroid and NATICUSdroid) and five ensemble classifiers, we first establish an intra-domain baseline, where models achieve over 92% accuracy, and then quantify a severe asymmetric performance drop. While models trained on PerMalDroid generalize well to NATICUSdroid (86% accuracy), the reverse direction sees a drastic drop to 73% accuracy. Explainable AI analysis reveals bimodal feature distributions and shows that feature importance is highly unstable, with key permissions losing or gaining influence across domains. The predictive feature sets for different domain
    
[^230]: 基于物理感知元学习从近表面高光谱遥感反射率反演沿岸生物地球化学参数

    Retrieval of Coastal Biogeochemical Parameters From Near-Surface Hyperspectral Remote Sensing Reflectance Using Physics-Aware Meta-Learning

    [https://arxiv.org/abs/2605.05623](https://arxiv.org/abs/2605.05623)

    本文提出了一个两阶段物理感知元学习框架，通过在纯数据驱动方法基础上融入物理知识，从近表面高光谱遥感反射率中反演沿岸水质参数，从而克服了传统反演算法难以跨不同水体推广的挑战。

    

    高光谱原位传感在反演水质参数（如总悬浮固体、溶解有机碳和总叶绿素-a）方面已显示出潜力，可用于经济高效的沿岸水质管理。然而，将此类反演算法推广到不同水体仍然具有挑战性，因为遥感反射率与水质参数之间的关系在不同区域可能存在显著差异，这是由于区域环境条件和生物地球化学的不同，导致了不同的水质范围和生物光学特性。在本研究中，我们提出了一个两阶段的物理感知元学习框架，用于从近表面遥感反射率观测数据中反演沿岸水质参数。与仅基于配对观测数据训练的纯数据驱动方法不同，所提出的框架通过两阶段学习过程融入了物理知识。在第一阶段……

    arXiv:2605.05623v4 Announce Type: replace  Abstract: Hyperspectral in situ sensing has shown promise in retrieving aquatic water quality (WQ) parameters, such as total suspended solids (TSS), dissolved organic carbon (DOC), and total chlorophyll-a (TChl-a), for cost-effective management of coastal WQ. However, generalising such retrieval algorithms across water bodies remains challenging, as the relationship between remote sensing reflectance ($R_{rs}$) and WQ parameters can vary considerably from one region to another due to regional distinctions in environmental conditions and biogeochemistry that lead to different WQ ranges and bio-optical properties. In this study, we propose a two-stage physics-aware meta-learning framework for retrieving coastal WQ parameters from near-surface $R_{rs}$ observations. Unlike purely data-driven approaches trained solely on paired observations, the proposed framework incorporates physical knowledge through a two-stage learning process. In the first s
    
[^231]: 具有可变滞后项的时间序列因果发现

    Time series causal discovery with variable lags

    [https://arxiv.org/abs/2605.04081](https://arxiv.org/abs/2605.04081)

    本文提出了一种可分解的BIC风格评分方法，用于联合学习时间序列因果发现中的变量时间顺序与每个有序变量对的主导滞后项，并通过增强的禁忌搜索算法进行优化。

    

    因果贝叶斯网络是在不确定条件下对复杂现实世界问题进行推理的强大工具。这类问题随时间演变，并在外部冲击发生时作出响应。为支持决策，因果贝叶斯网络需要一张关于所研究变量的因果关系图，即网络的结构。从数据中学习因果模型的图结构仍然具有挑战性；而从时间序列数据中学习则更加困难，因为变量间的依赖关系可能出现在不同的时间滞后处。我们构建了一种方法，可以联合学习变量的时间顺序以及每个选定有序变量对的主导滞后项。我们的贡献是一个适用于混合数据的可分解 BIC 风格目标评分函数，它将节点特定的样本量与偏向近期滞后项的惩罚项相结合。我们使用禁忌搜索对该目标函数进行优化，并通过改变滞后项的搜索动作和贪心的逐子节点滞后调整加以增强，同时其节点级可分解性……

    arXiv:2605.04081v3 Announce Type: replace-cross  Abstract: Causal Bayesian Networks (CBNs) are a powerful tool for reasoning under uncertainty about complex real-world problems. Such problems evolve over time, responding to external shocks as they occur. To support decision-making, CBNs require a cause-and-effect map of the variables under consideration, known as the network's structure. Learning the graphical structure of a causal model from data remains challenging; learning it from time-series data is even harder because dependencies may arise at different time lags. We construct a method that jointly learns a temporal ordering and a dominant lag for each selected ordered variable pair. Our contribution is a decomposable BIC-style objective score for mixed data that combines node-specific sample sizes with a penalty favouring recent lags. We optimise this objective using a Tabu search augmented with change-lag moves and greedy per-child lag tuning, while its node-wise decomposition 
    
[^232]: BoostLoRA：通过Boosting适配器来增长有效秩

    BoostLoRA: Growing Effective Rank by Boosting Adapters

    [https://arxiv.org/abs/2604.27308](https://arxiv.org/abs/2604.27308)

    BoostLoRA提出一种梯度提升框架，通过在错误样本上迭代训练并合并正交子空间的超低秩适配器，使有效秩随轮次线性增长，在不增加任何推理开销的情况下超越了全量微调的性能。

    

    参数高效微调（PEFT）方法面临适配器大小与表达能力之间的权衡：超低参数适配器被限制在固定的低秩子空间中，即使延长训练时间也无法突破性能上限。我们提出BoostLoRA，一种梯度提升框架，通过在当前模型出错的样本上迭代训练并合并极小适配器来克服这一限制。ROTATE SVD基策略将每一轮分配到一个正交子空间，使累积有效秩随轮数线性增长，同时每个适配器仍保持超低秩。合并完成后，适配器即被丢弃，推理阶段零额外开销。在Qwen2.5-3B上，BoostLoRA在GSM8K上达到89.1%，在MATH-500上达到68.8%，超越了最佳的单次超低参数适配器方法（TinyLoRA）和全量微调；在代码生成方面，它在MBPP上达到57.2%，在HumanEval上达到80.4%，而全量微调则降至零样本基线之下。

    arXiv:2604.27308v2 Announce Type: replace-cross  Abstract: Parameter-efficient fine-tuning (PEFT) methods face a tradeoff between adapter size and expressivity: ultra-low-parameter adapters are confined to fixed low-rank subspaces, capping performance even with extended training. We propose BoostLoRA, a gradient-boosting framework that overcomes this limit by iteratively training and merging minimal adapters on the examples the current model gets wrong. A ROTATE SVD basis strategy assigns each round to an orthogonal subspace, so cumulative effective rank grows linearly with the number of rounds while each adapter remains ultra-low-rank. After merging, adapters are discarded, leaving zero inference overhead. On Qwen2.5-3B, BoostLoRA reaches 89.1% on GSM8K and 68.8% on MATH-500, surpassing both the best single-shot ultra-low parameter adapter (TinyLoRA) and full fine-tuning; on code generation it reaches 57.2% on MBPP and 80.4% on HumanEval while full fine-tuning drops below the zero-sho
    
[^233]: mlr3torch：基于mlr3和torch的R语言深度学习框架

    mlr3torch: A Deep Learning Framework in R based on mlr3 and torch

    [https://arxiv.org/abs/2604.18152](https://arxiv.org/abs/2604.18152)

    本文介绍了R包mlr3torch，这是一个构建于torch之上、深度集成mlr3生态系统的可扩展深度学习框架，它简化了神经网络的定义、训练和评估，并允许用户通过图语言在单一图中定义包含预处理、数据增强和网络架构的完整建模工作流程。

    

    深度学习（DL）已成为现代机器学习（ML）实践的基石。我们介绍R包mlr3torch，它是mlr3生态系统的一个可扩展的深度学习框架。该框架构建于torch包之上，简化了针对表格数据和通用张量（例如图像）进行分类和回归任务的神经网络的定义、训练与评估。该包实现了预定义的网络架构，并且可以轻松地将torch模型转换为mlr3学习器。它还允许用户将神经网络定义为图。这种表示方式基于mlr3pipelines中定义的图语言，使用户能够在单个图中定义完整的建模工作流程，包括预处理、数据增强和网络架构。通过与mlr3生态系统的集成，该包支持便捷的重采样、基准测试、预处理等功能。我们对该包的设计进行了说明……（原文摘要在此处截断）

    arXiv:2604.18152v2 Announce Type: replace-cross  Abstract: Deep learning (DL) has become a cornerstone of modern machine learning (ML) praxis. We introduce the R package mlr3torch, which is an extensible DL framework for the mlr3 ecosystem. It is built upon the torch package, and simplifies the definition, training, and evaluation of neural networks for both tabular data and generic tensors (e.g., images) for classification and regression. The package implements predefined architectures, and torch models can easily be converted to mlr3 learners. It also allows users to define neural networks as graphs. This representation is based on the graph language defined in mlr3pipelines and allows users to define the entire modeling workflow, including preprocessing, data augmentation, and network architecture, in a single graph. Through its integration into the mlr3 ecosystem, the package allows for convenient resampling, benchmarking, preprocessing, and more. We explain the package's design an
    
[^234]: 用于生成建模的变分测量量子计算中的经典资源最小化

    Minimizing classical resources in variational measurement-based quantum computation for generative modeling

    [https://arxiv.org/abs/2604.11578](https://arxiv.org/abs/2604.11578)

    本文针对变分测量量子计算（VMBQC）中信道模型参数量是幺正模型两倍的局限，提出了最小化经典资源消耗的方案，使生成建模中的经典处理开销得到显著降低。

    

    测量量子计算是一种量子信息处理框架，其中计算任务通过对高度纠缠的资源态执行单量子比特测量来实现。由于量子测量结果的不确定性，这些操作的随机结果若不加校正，将形成一个变分量子信道族。传统上，这种随机性需要通过经典处理加以校正，以确保确定性的幺正计算。最近，变分测量量子计算（VMBQC）被提出，利用这种测量引起的随机性在生成建模中获得优势。该方法的一个局限是，相应的信道模型相比幺正模型具有两倍的参数量，其规模为 $N \times D$，其中 $N$ 是逻辑量子比特的数量（宽度），$D$ 是 VMBQC 模型的深度。这可能……

    arXiv:2604.11578v2 Announce Type: replace-cross  Abstract: Measurement-based quantum computation (MBQC) is a framework for quantum information processing in which a computational task is carried out through one-qubit measurements on a highly entangled resource state. Due to the indeterminacy of the outcomes of a quantum measurement, the random outcomes of these operations, if not corrected, yield a variational quantum channel family. Traditionally, this randomness is corrected through classical processing in order to ensure deterministic unitary computations. Recently, variational measurement-based quantum computation (VMBQC) has been introduced to exploit this measurement-induced randomness to gain an advantage in generative modeling. A limitation of this approach is that the corresponding channel model has twice as many parameters compared to the unitary model, scaling as $N \times D$, where $N$ is the number of logical qubits (width) and $D$ is the depth of the VMBQC model. This can
    
[^235]: 面向可持续时间序列分类的剪枝扩展与效率权衡

    Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification

    [https://arxiv.org/abs/2604.07953](https://arxiv.org/abs/2604.07953)

    本文提出多维度评估框架、有界误差剪枝策略及新型Hydrant分类器扩展，可在时间序列分类任务中实现能耗降低高达70%，而预测精度几乎不受影响。

    

    大规模时间序列分类需要在预测性能与资源消耗之间取得平衡，而这种平衡取决于给定的数据、模型和硬件。本文提出了一个多维度评估框架，能够对时间序列分类中资源与质量之间的权衡进行显式的基准测试和探索。为了提高效率，我们针对领先的混合分类器 Hydra 和 Quant 提出了一种具有有界误差的剪枝策略，并进一步提出了 Hydrant——一种将区间表示与随机卷积和字典学习相结合的新型扩展方法。通过在 20 个 MONSTER 数据集、10 种方法和三种计算设置上评估超过 4000 种配置，我们系统地分析了模型设计、超参数和硬件选择如何影响实际的时间序列分类性能。我们的结果表明，剪枝能够显著降低能耗高达 70%，同时仅轻微降低预测准确率。

    arXiv:2604.07953v2 Announce Type: replace-cross  Abstract: Time series classification (TSC) at scale requires balancing predictive performance against resource consumption, which is subject to the given data, model, and hardware. This paper introduces a multi-dimensional evaluation framework that allows for explicitly benchmarking and exploring the resource-versus-quality trade-offs in TSC. To improve efficiency, we present a pruning strategy with bounded error for leading hybrid classifiers - Hydra and Quant - and moreover propose Hydrant, a novel extension that integrates interval representations with random convolutions and dictionary learning. With over 4000 evaluated configurations across 20 MONSTER datasets, 10 methods, and three compute setups, we systematically analyze how model design, hyperparameters, and hardware choices impact practical TSC performance. Our results show that pruning can significantly reduce energy consumption by up to 70% while only marginally lowering pred
    
[^236]: 利用大语言模型解决 AdaBoost 循环问题：一个计算机辅助的反例

    Resolving AdaBoost Cycling with LLMs: A Computer-Assisted Counterexample

    [https://arxiv.org/abs/2604.07055](https://arxiv.org/abs/2604.07055)

    本文通过计算机辅助构造了一个包含 2,500 个样本的反例，证明穷举式 AdaBoost 并不总是收敛到有限循环，从而解决了 Rudin、Schapire 和 Daubechies 提出的长期公开问题。

    

    我们针对 Rudin、Schapire 和 Daubechies 在 COLT 2012 上提出的公开问题——穷举式 AdaBoost 是否总是收敛到一个有限循环——构造了一个计算机辅助的反例。该构造基于两个小型 AdaBoost 实例的块乘积：两个因子在各自的五步分支映射上共享一个精确的周期-2 轨道，但其线性化回归映射的优势特征值具有无理的对数比值。这种无理性迫使每轮更新中获胜实例的序列具有无理的渐近频率，从而排除了收敛到有限循环的可能性。所得反例包含 2,500 个样本，弱学习器边际为 1/5，初始分布为均匀分布，并且在任何最大化平局规则下均成立。证明中的所有有限计算均通过 SymPy 中的有理数算术进行认证，并由 SageMath 验证主要不等式。我们还记录了相关的协作工作

    arXiv:2604.07055v3 Announce Type: replace  Abstract: We develop a computer-assisted counterexample to the open question of whether exhaustive AdaBoost always converges to a finite cycle, posed by Rudin, Schapire, and Daubechies in COLT 2012. The construction is based on a block-product of two small AdaBoost instances whose two factors share an exact period-2 orbit for their five-step branch maps, but whose linearized return maps have dominant eigenvalues with an irrational logarithmic ratio. This irrationality forces the sequence of instances that win each round of updates to have an irrational asymptotic frequency, which precludes convergence to a finite cycle. The resulting counterexample has 2,500 examples, weak-learning margin $1/5$, and a uniform initial distribution, and it holds under every maximizing tie rule. All finite computations in the proof are certified by rational arithmetic in SymPy, with SageMath verifying the main inequalities. We also document the collaborative work
    
[^237]: 蛋白质设计的生成式机器学习

    Generative machine learning for protein design

    [https://arxiv.org/abs/2603.26378](https://arxiv.org/abs/2603.26378)

    本文系统综述了59种生成式机器学习方法在六类蛋白质设计任务（序列生成、骨架生成、结构条件序列设计、序列-结构联合设计、基序支架及结合蛋白/抗体设计）中的应用，并讨论了设计蛋白质的计算与实验评估方法。

    

    设计具有特定生物功能的蛋白质对于新疗法的开发以及酶的工程化改造都具有重要意义，然而实现预期的蛋白质活性仍然充满挑战。生成式人工智能（AI）通过从蛋白质序列和结构数据中学习，已成为探索庞大可能设计空间的有力工具。本文综述了涵盖六种蛋白质设计任务的59种生成式方法，包括序列生成、骨架生成、结构条件下的序列设计、序列-结构联合设计、基序支架构建以及结合蛋白和抗体设计。我们概述了常见的生成方法和蛋白质表示形式，并讨论了在生成过程中如何利用结构约束和功能信息。随后，我们讨论了用于评估所设计蛋白质的计算与实验方法，以及这些评估……（原文在此处截断）

    arXiv:2603.26378v2 Announce Type: replace-cross  Abstract: Designing proteins with specific biological functions is of considerable interest for both the development of new therapeutics and the engineering of enzymes, yet achieving the intended activity remains challenging. Generative artificial intelligence (AI) has emerged as a powerful tool to explore the vast space of possible designs by learning from protein sequence and structural data. Here, we review 59 generative approaches across six protein design tasks, including sequence generation, backbone generation, structure-conditioned sequence design, joint sequence-structure design, motif scaffolding, and binder and antibody design. We outline common generative approaches and protein representations and discuss how structural constraints and functional information are utilized during generation. This is followed by a discussion of the computational and experimental approaches used to evaluate designed proteins and what such assessm
    
[^238]: DRTriton：基于大规模合成数据驱动的强化学习实现Triton内核生成

    DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation

    [https://arxiv.org/abs/2603.21465](https://arxiv.org/abs/2603.21465)

    提出DRTriton框架，通过CSP-DAG大规模合成数据、解耦奖励的课程化强化学习以及测试时搜索算法，训练大语言模型将PyTorch程序高效转换为优化的Triton内核。

    

    开发高效的CUDA内核是生成式AI行业中一项基础但极具挑战性的任务。近期研究利用大语言模型（LLM）自动将PyTorch参考实现转换为CUDA内核，显著降低了工程工作量。然而，即便是最先进的LLM（如GPT-5.2和Claude-Sonnet-4.5）仍然难以胜任这项任务。为应对这一挑战，我们提出了DRTriton，一个可扩展的学习框架，用于训练LLM将PyTorch程序转换为高度优化的Triton内核，并在运行时将其编译为CUDA内核。DRTriton包含三个关键组件：（i）数据合成算法CSP-DAG，可在算子空间中保证完全覆盖、无偏均匀采样，并能控制难度；（ii）采用解耦奖励的课程化强化学习框架，可联合优化转换成功率和执行速度；（iii）测试时搜索算法，进一步……（原文摘要在此处截断）

    arXiv:2603.21465v3 Announce Type: replace  Abstract: Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent research leverages Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing engineering effort. State-of-the-art LLMs, such as GPT-5.2 and Claude-Sonnet-4.5, still struggle with this task. To address this challenge, we propose DRTriton, a scalable learning framework for training LLMs to convert PyTorch programs into highly optimized Triton kernels, which are then compiled to CUDA kernels at runtime. DRTriton consists of three key components: (i) a data synthetic algorithm CSP-DAG that guarantees full coverage and unbiased uniform sampling over the operator space with controlled difficulty; (ii) a curriculum RL framework with decoupled rewards that jointly optimizes conversion success rate and execution speed; and (iii) a test-time search algorithm that fur
    
[^239]: 自动微分数据同化：状态、动力学与滤波算法的协同学习

    Auto-differentiable data assimilation: Co-learning of states, dynamics, and filtering algorithms

    [https://arxiv.org/abs/2603.20891](https://arxiv.org/abs/2603.20891)

    本文提出了一种自动微分滤波框架，利用理论驱动的损失函数和基于梯度的优化，实现对状态、动力学模型以及滤波算法参数的联合协同学习，从而摆脱传统数据同化中昂贵的参数调优和对精确动力学模型的依赖。

    

    数据同化算法通过部分观测来估计动力学系统的状态，而这些算法的成功应用依赖于代价高昂的参数调优以及采用精确的动力学模型。本文提出了一个框架，通过我们称之为“自动微分滤波”的过程，在数据同化中联合学习状态、动力学以及滤波算法的参数。该框架利用一个具有理论依据的损失函数，使得能够借助基于梯度的优化（使用自动微分）从部分且有噪声的观测中进行学习。我们进一步展示了如何在den框架内学习或调优几种著名的数据同化方法。为了凸显自动微分滤波的多功能性，我们在跨越多个科学领域的动力系统上进行了实验，例如航空航天工程中的 Clohessy-Wiltshire 方程等。

    arXiv:2603.20891v2 Announce Type: replace-cross  Abstract: Data assimilation algorithms estimate the state of a dynamical system from partial observations, where the successful performance of these algorithms hinges on costly parameter tuning and on employing an accurate model for the dynamics. This paper introduces a framework for jointly learning the state, dynamics, and parameters of filtering algorithms in data assimilation through a process we refer to as auto-differentiable filtering. The framework leverages a theoretically motivated loss function that enables learning from partial, noisy observations via gradient-based optimization using auto-differentiation. We further demonstrate how several well-known data assimilation methods can be learned or tuned within this framework. To underscore the versatility of auto-differentiable filtering, we perform experiments on dynamical systems spanning multiple scientific domains, such as the Clohessy-Wiltshire equations from aerospace engi
    
[^240]: FlashSampling：快速且内存高效的确切采样

    FlashSampling: Fast and Memory-Efficient Exact Sampling

    [https://arxiv.org/abs/2603.15854](https://arxiv.org/abs/2603.15854)

    FlashSampling将精确采样融合进LM头的矩阵乘法运算中，通过在片上逐块计算并添加Gumbel噪声避免在显存中物化logits张量，并以流式点对点通信替代all-gather，实现了跨最多8个GPU的快速、精确且内存高效的大词表采样。

    

    arXiv:2603.15854v4 公告类型：replace-cross 摘要：从分类分布中进行采样在数学上很简单，但在大词表解码场景中，它通常会在语言模型输出头（LM head）之后触发额外的内存流量和额外的内核调用。我们提出了FlashSampling，这是一种精确的采样原语，它将采样融合到LM头的矩阵乘法中，并且从不在HBM（高带宽显存）中物化logits张量。该方法十分简单：在芯片上逐块计算logits，添加Gumbel噪声，每行和每个词表块仅保留一个最大值，最后通过对各块进行小型归约完成采样。在张量并行解码中，FlashSampling用流式点对点写入取代了logits的all-gather（全收集）：这使得GPU到GPU的通信与计算和HBM加载相重叠，最多可扩展至8个GPU，并在大批量规模下实现接近理想的扩展性。我们的内核是精确的，因为argmax可以在分区上进行分解；针对在线和张量并行设置的分组变体通过分类分布的层次化分解同样保证了精确性。

    arXiv:2603.15854v4 Announce Type: replace-cross  Abstract: Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. In tensor-parallel decoding, FlashSampling replaces the all-gather of logits with streaming peer-to-peer writes: This overlaps GPU-to-GPU communication with computation and HBM loads across up to 8 GPUs, with near-ideal scaling at large batch sizes. Our kernel is exact because argmax decomposes over partitions; grouped variants for online and tensor-parallel settings are exact by hierarchical factorization of the categorical distribu
    
[^241]: 美国人口普查去噪：简洁的分块层次回归

    Denoising the US Census: Succinct Block Hierarchical Regression

    [https://arxiv.org/abs/2603.10099](https://arxiv.org/abs/2603.10099)

    BlueDown是一种结合线性回归与约束优化的后处理算法，能在不牺牲隐私保护或计算效率的前提下，显著降低2020年美国人口普查差分隐私噪声造成的分组数据误差。

    

    2020年美国人口普查采用差分隐私来保护个人隐私，在汇总的人口统计测量数据中加入校准噪声，再将这些数据调和成人口表格。我们开发了BlueDown，这是一种后处理算法，结合了线性回归和约束优化的技术，同时利用测量数据的结构以提高效率。在2020年人口普查数据上，BlueDown将县级按种族分组数据的误差降低了21-36%，将所有县级分组数据的误差降低了9-58%；在普查区层面，它将按住房类型分组数据的误差降低了38-61%。这些改进没有以牺牲隐私保护或计算效率为代价，将提高依赖这一权威数据集的研究以及用于指导每年超过1.5万亿美元联邦支出分配的分析的有效性。

    arXiv:2603.10099v2 Announce Type: replace  Abstract: The 2020 United States Census adopted differential privacy to protect individual confidentiality, adding calibrated noise to summary demographic measurements that are then reconciled into population tables. We developed BlueDown, a postprocessing algorithm that combines techniques from linear regression and constrained optimization while exploiting measurement structure for efficiency. On 2020 Census data, BlueDown reduced the error of groupings by race at the county level by 21-36% and of all county-level groupings by 9-58%; at the tract level, it reduced error of groupings by housing type by 38-61%. These gains came at no cost to confidentiality or computational efficiency, and would improve the validity of research relying on this definitive dataset and analyses used to guide the distribution of over $1.5 trillion in annual federal spending.
    
[^242]: 基于参数与数据高效适配的草稿模型高效对齐方法

    Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation

    [https://arxiv.org/abs/2603.09527](https://arxiv.org/abs/2603.09527)

    本文提出参数与数据高效的草稿模型适配框架 EDA，通过解耦架构、数据再生策略和样本选择机制三项创新，高效地将草稿模型对齐到微调后的目标模型，从而提升推测解码的加速效果。

    

    推测解码可以加速大语言模型的推理，但当目标模型针对特定领域进行微调后，其性能会出现下降。一种直接的做法是为每个目标模型重新训练草稿模型，但这成本高昂且效率低下。为解决这一问题，我们提出了一种实用的参数与数据高效框架，称为“高效草稿适配”，简称 EDA，用于高效地适配草稿模型。EDA 引入了三项创新：（1）解耦架构，利用共享组件和私有组件分别建模共享的输出分布与目标特定的输出分布，只需更新轻量级的私有组件即可实现参数高效的适配；（2）数据再生策略，利用微调后的目标模型重新生成训练数据，从而提升训练与推测解码之间的对齐程度，带来更高的平均接受长度；（3）样本选择机制（摘要原文在此处被截断）。

    arXiv:2603.09527v2 Announce Type: replace-cross  Abstract: Speculative decoding accelerates LLM inference but suffers from performance degradation when target models are fine-tuned for specific domains. A naive solution is to retrain draft models for every target model, which is costly and inefficient. To address this, we introduce a practical parameter- and data-efficient framework named Efficient Draft Adaptation, abbreviated as EDA, for efficiently adapting draft models. EDA introduces three innovations: (1) a decoupled architecture that utilizes shared and private components to model the shared and target-specific output distributions separately, enabling parameter-efficient adaptation by updating only the lightweight private component; (2) a data regeneration strategy that utilizes the fine-tuned target model to regenerate training data, thereby improving the alignment between training and speculative decoding, leading to higher average acceptance length; (3) a sample selection me
    
[^243]: ESENSC：SHAP的一种多项式时间公理化替代方法

    ESENSC: A Polynomial-Time Axiomatic Alternative to SHAP

    [https://arxiv.org/abs/2603.00496](https://arxiv.org/abs/2603.00496)

    ESENSC是一种仅需线性次联盟值评估的多项式时间确定性归因方法，在表格预测任务上与精确SHAP高度一致，同时计算成本远低于后者。

    

    我们提出了ESENSC，一种计算高效且具有理论基础SHAP替代方法。ESENSC是一种确定性的、闭式解析的归因规则，只需线性次数的联盟值评估，满足空玩家性质，并基于效率公理、受限形式的差分边际性以及明确的计算约束，获得了公理化刻画。在实证方面，在使用神经网络和XGBoost模型的表格预测任务上，ESENSC在归因大小和特征排序上均与精确SHAP高度一致，同时所需计算量大幅减少。在所评估的设置下，其表现优于广泛使用的基于采样的SHAP近似方法。精确SHAP需要指数次联盟评估，而ESENSC的计算量随特征数量呈线性扩展。这些结果表明，ESENSC在与精确SHAP的一致性和计算效率之间实现了良好的权衡。

    arXiv:2603.00496v2 Announce Type: replace-cross  Abstract: We propose ESENSC, a computationally efficient and theoretically grounded alternative to SHAP. ESENSC is a deterministic, closed-form attribution rule that requires only linearly many coalition-value evaluations, satisfies the null-player property, and admits an axiomatic characterization based on efficiency, a restricted form of differential marginality, and explicit computational constraints.Empirically, on tabular prediction tasks using neural network and XGBoost models, ESENSC shows close agreement with exact SHAP in both attribution magnitude and feature rankings while requiring substantially less computation. Under the evaluated settings, it compares favorably with widely used sampling-based SHAP approximations. While exact SHAP requires exponentially many coalition evaluations, ESENSC scales linearly with the number of features. These results show that ESENSC offers a favorable trade-off between agreement with exact SHAP
    
[^244]: 基于非相干性调整语义体积的多模态大语言模型不确定性量化

    Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume

    [https://arxiv.org/abs/2602.24195](https://arxiv.org/abs/2602.24195)

    提出了UMPIRE，一个免训练且无需外部工具的多模态大语言模型不确定性量化框架，其受行列式点过程启发，通过计算新颖的“非相干性调整语义体积”，将语义嵌入离散度与响应二次熵相结合，从而高效支持多种输入输出模态。

    

    尽管多模态大语言模型（MLLM）能力强大，但它们可能产生看似合理实则错误的输出，这阻碍了其可靠部署。准确的不确定性度量可以将不可靠的查询升级给人类专家或更大的模型处理，从而提升性能。然而，现有的不确定性度量存在实际限制，例如仅针对特定模态设计、依赖外部工具或计算成本高昂。我们提出了UMPIRE，这是一个面向多模态大语言模型的免训练不确定性量化框架，它能够在各种输入和输出模态上高效工作，无需外部工具，仅依赖模型自身的内部模态特征。受行列式点过程（DPPs）的启发，UMPIRE针对给定任务实例计算采样得到的MLLM响应的一种新颖的非相干性调整语义体积，该方法将语义嵌入的离散度与响应的二次熵相结合。

    arXiv:2602.24195v2 Announce Type: replace  Abstract: Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features. Inspired by Determinantal Point Processes (DPPs), UMPIRE computes a novel incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, which couples semantic embedding dispersion with the quadratic entropy of the responses. This
    
[^245]: 关于Wasserstein与Gromov-Wasserstein最优传输计划的稀疏性、极值结构与单调性性质

    On sparsity, extremal structure, and monotonicity properties of Wasserstein and Gromov-Wasserstein optimal transport plans

    [https://arxiv.org/abs/2602.16265](https://arxiv.org/abs/2602.16265)

    本文系统梳理了Gromov-Wasserstein最优传输的关键性质，证明了在条件负定/正定（CND/CPD）条件下存在稀疏且支撑于置换上的最优传输计划，并给出不满足这些假设时置换并非最优的反例。

    

    本简报对Gromov-Wasserstein（GW）距离的一些关键性质进行了自成一体的综述，并与标准的线性最优传输（OT）框架进行比较。更具体地说，我探讨了以下问题：GW最优传输计划是否稀疏？在什么条件下存在支撑于置换上的最优计划？最优计划是否满足某种形式的循环单调性？特别是，我给出了条件负定/正定（CND/CPD）性质，并证明当这些性质成立时，存在稀疏且支撑于置换上的GW最优计划。在最后一节中，我还给出了在去掉CND或CPD假设时置换并非最优的例子。

    arXiv:2602.16265v3 Announce Type: replace-cross  Abstract: This note gives a self-contained overview of some key properties of the Gromov-Wasserstein (GW) distance, compared with the standard linear optimal transport (OT) framework. More specifically, I explore the following questions: are GW optimal transport plans sparse? under what conditions does there exist an optimal plan supported on a permutation? do optimal plans satisfy a form of cyclical monotonicity? In particular, I present the conditionally negative/positive definite (CND/CPD) properties and show that, when they hold, there exist GW optimal plans that are sparse and supported on a permutation. In the last section, I also give examples where no permutation is optimal when the CND or CPD assumption is dropped.
    
[^246]: 从近端算子视角看流匹配

    Flow Matching from Viewpoint of Proximal Operators

    [https://arxiv.org/abs/2602.12683](https://arxiv.org/abs/2602.12683)

    该论文通过扩展Brenier势将最优传输条件流匹配（OT-CFM）的向量场精确表示为与势函数相关的近端算子（去噪器），并借助Lyapunov指数证明其流形稳定性：法向扰动线性收缩而切向扰动被保留。

    

    我们讨论了流形假设下最优传输条件流匹配（OT-CFM）的动力学，其中目标分布的支撑集是一个低维流形。我们首先利用扩展的Brenier势导出了向量场的精确近端表达式；该向量场可由一个去噪器表示，而该去噪器可呈现为与势函数相关联的近端算子。基于向量场的这一显式近端形式，我们利用动力学的Lyapunov指数分析了初始点扰动下的流形稳定性。借助凸分析工具对局部行为进行分析，我们证明动力学在流形法向方向上的扰动呈线性收缩，而切向方向上的扰动则得以保留；这表明流形结构对扰动具有稳定性。为了支持该理论……

    arXiv:2602.12683v3 Announce Type: replace  Abstract: We discuss the dynamics of Optimal Transport Conditional Flow Matching (OT-CFM) under the manifold hypothesis, where the support of the target distribution is a low dimensional manifold. We first derive an exact proximal expression of the vector field, using an extended Brenier potential for the OT; the vector field can be expressed by a denoiser, which is presented as the proximal operator associated with the potential function. Based on this explicit proximal form of the vector field, we analyze the manifold stability for perturbations to initial points, using the Lyapunov exponents of the dynamics. By analyzing the local behavior with the tools in convex analysis, we show that the dynamics contract linearly in the perturbation of the normal directions to the manifold, while perturbations in the tangential directions are preserved; this implies the stability of the manifold structure to the perturbations. To justify the theoretical
    
[^247]: 面向差分隐私主成分分析的自适应幂迭代方法

    Adaptive Power Iteration Method for Differentially Private PCA

    [https://arxiv.org/abs/2602.11454](https://arxiv.org/abs/2602.11454)

    本文提出一种自适应的差分隐私幂迭代算法，通过引入适应矩阵相干性参数的新过滤技术，对低相干性矩阵实现了超越最坏情况的顶部奇异向量近似保证。

    

    我们研究用于近似计算矩阵 $A\in\mathbb{R}^{n\times d}$ 顶部奇异向量的 $(\epsilon,\delta)$-差分隐私算法，其中矩阵的每一行是 $\mathbb{R}^d$ 中的一个数据点。遵循 Dwork-Talwar-Thakurta-Zhang（STOC 2014）的工作，我们考虑相邻输入仅相差一行的隐私模型。我们提出了一种新颖的算法，对于具有低相干性的输入矩阵能够实现超越最坏情况的保证，而低相干性是许多应用中矩阵所具备的一种结构性质，包括但不限于独立同分布数据。我们的算法为私有幂迭代方法方面的大量文献做出了贡献，其中我们引入了一种能够适应相干性参数的新过滤技术。我们的工作区别于并补充了 Hardt-Roth（STOC 2013）的工作，后者在更严格的隐私模型（相邻输入……）下实现了超越最坏情况的保证。

    arXiv:2602.11454v4 Announce Type: replace-cross  Abstract: We study $\left(\epsilon,\delta\right)$-differentially private algorithms for the problem of approximately computing the top singular vector of a matrix $A\in\mathbb{R}^{n\times d}$ where each row of $A$ is a data point in $\mathbb{R}^{d}$. Following Dwork-Talwar-Thakurta-Zhang (STOC 2014), we consider the privacy model where neighboring inputs differ by one single row. We give a novel algorithm that achieves beyond-worst-case guarantees for input matrices with low coherence, which is a structural property of matrices in many applications, including but not limited to i.i.d. data. Our algorithm contributes to the extensive literature on private power iteration methods, where we introduce a new filtering technique which adapts to this coherence parameter. Our work departs from and complements the work by Hardt-Roth (STOC 2013) which achieves beyond-worst-case guarantees for the more restrictive privacy model where neighboring in
    
[^248]: ROCKET：通过校准引导的背包增强截断实现快速优化的高效模型压缩方法

    ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression

    [https://arxiv.org/abs/2602.11008](https://arxiv.org/abs/2602.11008)

    ROCKET是一种免训练的模型压缩方法，通过将逐层压缩分配建模为多选择背包问题，并结合基于校准集的单步稀疏矩阵分解，在全局压缩预算下实现了超越现有方法的最先进压缩性能。

    

    我们提出了ROCKET，这是一种免训练的模型压缩方法，与因式分解、结构化稀疏化和动态压缩等基线方法相比，实现了最先进的性能。在全局压缩预算的约束下，ROCKET包含两项关键创新：首先，它将逐层压缩分配问题建模为一个多选择背包问题，在满足目标模型大小的前提下，为每一层选择最优压缩级别以最小化总重构误差。其次，它引入了一种受字典学习启发的单步稀疏矩阵分解方法：仅使用一个小型校准集，基于激活-权重敏感度对权重系数进行稀疏化，然后通过最小二乘法以闭式解形式更新字典，完全绕过了迭代优化、稀疏编码或反向传播。ROCKET在不同模型上持续超越现有的压缩方法。

    arXiv:2602.11008v2 Announce Type: replace-cross  Abstract: We present ROCKET, a training-free model compression method that achieves state-of-the-art performance in comparison with factorization, structured-sparsification and dynamic compression baselines. Operating under a global compression budget, ROCKET comprises two key innovations: First, it formulates layer-wise compression allocation as a multi-choice knapsack problem, selecting the optimal compression level for each layer to minimize total reconstruction error while adhering to a target model size. Second, it introduces a single-step sparse matrix factorization inspired by dictionary learning: using only a small calibration set, it sparsifies weight coefficients based on activation-weights sensitivity and then updates the dictionary in closed form via least squares bypassing iterative optimization, sparse coding, or backpropagation entirely. ROCKET consistently outperforms existing compression approaches across different model
    
[^249]: 从对话到市场：AgenticPay——用于多智能体商务中LLM驱动的谈判

    From Chats to Markets: AgenticPay for LLM-Powered Negotiation in Multi-Agent Commerce

    [https://arxiv.org/abs/2602.06008](https://arxiv.org/abs/2602.06008)

    AgenticPay是一个包含160个多模态任务、覆盖四种商业场景和八种市场拓扑的统一基准，用于评估多模态LLM智能体在竞争性多智能体市场中通过多轮谈判达成高福利、多维度协议的能力。

    

    大型语言模型智能体越来越被期望能够自主进行谈判和交易，然而现有的基准测试很少将多模态产品定位、多议题讨价还价和竞争性多智能体市场结合起来。我们提出了AgenticPay，这是一个统一的基准，用于评估多模态智能体在现实市场中达成高福利、多维度协议的能力。AgenticPay包含160个多模态任务，涵盖四种商业场景（电子商务、外卖配送、网约车和公寓租赁）以及八种市场拓扑结构，从双边谈判到多对多竞争。智能体需要解读产品图片、推断对手的隐藏偏好，并通过多轮自然语言对话协商具有约束力的合同条款，如价格、租期、退货政策和配送速度。该基准采用基于效用的框架来评估协议质量和联合福利。使用专有模型（GPT-5.4、Claude Son……（摘要原文在此处被截断）的实验……

    arXiv:2602.06008v2 Announce Type: replace  Abstract: Large language model agents are increasingly expected to negotiate and transact autonomously, yet existing benchmarks rarely combine multimodal product grounding, multi-issue bargaining, and competitive multi-agent markets. We introduce AgenticPay, a unified benchmark for evaluating multimodal agents' ability to reach high-welfare, multidimensional agreements in realistic markets. AgenticPay includes 160 multimodal tasks across four business scenarios (E-commerce, Food Delivery, Ride-hailing, and Apartment Rental) and eight market topologies, from bilateral bargaining to many-to-many competition. Agents interpret product images, infer opponents' hidden preferences, and negotiate binding contract terms such as price, lease duration, return policy, and delivery speed through multi-round natural-language dialogue. A utility-based framework evaluates agreement quality and joint welfare. Experiments with proprietary (GPT-5.4, Claude Sonne
    
[^250]: 对齐诅咒：模态对齐通过文本迁移大幅增强音频攻击

    The Alignment Curse: Modality Alignment Supercharges Audio Attacks via Text Transfer

    [https://arxiv.org/abs/2602.02557](https://arxiv.org/abs/2602.02557)

    提出并验证了“对齐诅咒”原则——全模态模型中文本-音频模态对齐越强，文本越狱攻击向音频模态的迁移就越有效，揭示了模型能力提升与安全性之间的根本张力。

    

    端到端训练的全模态模型的最新进展通过强化文本-音频模态对齐，显著提升了音频能力。然而，这种对齐是否会在无意中促进安全漏洞的跨模态迁移，目前仍未得到充分研究。这一问题至关重要，因为基于文本的越狱攻击远比基于音频的攻击成熟；如果这些攻击能够系统性地迁移，当前的音频安全评估可能低估源自文本模态的风险。在本文中，我们提出了“对齐诅咒”，这是一个经过形式化表征和实证验证的原则，表明更强的模态对齐使攻击从文本到音频的迁移更加有效，揭示了能力与安全之间的根本张力。基于这一原则，我们对近期全模态模型（如 Qwen2.5-Omni 等）上的三类攻击进行了全面的黑盒评估……

    arXiv:2602.02557v3 Announce Type: replace-cross  Abstract: Recent advances in end-to-end trained omni-models have substantially improved audio capabilities by strengthening text-audio modality alignment. However, whether such alignment inadvertently facilitates the transfer of safety vulnerabilities across modalities remains underexplored. This question is critical as text-based jailbreak attacks are considerably more mature than audio-based ones; if they transfer systematically, current audio safety evaluations may underestimate risks originating from the text modality. In this paper, we introduce the Alignment Curse, a formally characterized and empirically validated principle showing that stronger modality alignment enables more effective transfer of attacks from text to audio, revealing a fundamental tension between capability and safety. Motivated by this principle, we conduct a comprehensive black-box evaluation of three attack categories on recent omni-models (e.g., Qwen2.5-Omni
    
[^251]: 具有参数化条件分布的深度多元模型

    Deep Multivariate Models with Parametric Conditionals

    [https://arxiv.org/abs/2602.01953](https://arxiv.org/abs/2602.01953)

    该论文提出用每组变量以其余变量为条件的参数化条件概率分布来表示联合概率分布，从而构建几乎可通用于任意下游任务的深度多元模型，其训练等价于通过最大化极限分布的数据似然来学习一个参数化马尔可夫链核。

    

    我们研究面向异构随机变量集合的深度多元模型。在计算机视觉的背景下，此类集合可以由图像、分割结果、图像属性以及潜在变量等组成。在开发此类模型时，大多数现有工作从某个应用任务出发，设计模型组件及其依赖关系以满足所选任务的需求。这种做法的缺点是限制了所得模型在其他下游任务上的适用性。与此不同，我们提出用每组变量以其余变量为条件的条件概率分布来表示联合概率分布。这样构建的模型几乎可以用于任何可能的下游任务。此类模型的学习可以被看作训练一个参数化的马尔可夫链核，即通过最大化其极限分布的数据似然来进行训练。这还具有一个额外的好处，即允许……（原文摘要在此处截断）

    arXiv:2602.01953v2 Announce Type: replace  Abstract: We consider deep multivariate models for heterogeneous collections of random variables. In the context of computer vision, such collections may e.g. consist of images, segmentations, image attributes, and latent variables. When developing such models, most existing works start from an application task and design the model components and their dependencies to meet the needs of the chosen task. This has the disadvantage of limiting the applicability of the resulting model for other downstream tasks. Here, instead, we propose to represent the joint probability distribution by means of conditional probability distributions for each group of variables conditioned on the rest. Such models can then be used for practically any possible downstream task. Their learning can be approached as training a parametrised Markov chain kernel by maximising the data likelihood of its limiting distribution. This has the additional advantage of allowing a 
    
[^252]: HetCCL：异构厂商GPU上的高效大语言模型训练

    HetCCL: Efficient LLM Training on Heterogeneous Vendor GPUs

    [https://arxiv.org/abs/2601.22585](https://arxiv.org/abs/2601.22585)

    HetCCL是一个无需修改驱动程序即可统一NVIDIA NCCL和AMD RCCL后端的集合通信库，通过基于RDMA的跨厂商通信机制，使现有深度学习应用无需改动即可在NVIDIA与AMD混合GPU集群上高效训练大语言模型。

    

    arXiv:2601.22585v2 公告类型：replace-cross 摘要：大语言模型的快速发展正促使各组织不断扩展其GPU集群，这些集群通常包含来自多个厂商的GPU。然而，当前的深度学习框架缺乏对异构GPU之间集合通信的支持，导致效率低下和成本增加。我们提出了HetCCL，这是一个集合通信库，它统一了各厂商特定的后端，并在无需修改驱动程序的情况下实现基于RDMA的GPU间通信。HetCCL引入了两种新颖机制，在利用厂商优化库（NVIDIA NCCL和AMD RCCL）的同时实现跨厂商通信。在多厂商GPU集群上的评估表明，HetCCL在同构环境中可达到与NCCL和RCCL相当的性能，同时能够独特地在异构环境中扩展，使得在不更改现有深度学习应用程序的情况下，使用NVIDIA和AMD GPU进行实用、高性能的大语言模型训练成为可能。

    arXiv:2601.22585v2 Announce Type: replace-cross  Abstract: The rapid growth of large language models is driving organizations to expand their GPU clusters, often with GPUs from multiple vendors. However, current deep learning frameworks lack support for collective communication across heterogeneous GPUs, leading to inefficiency and higher costs. We present HetCCL, a collective communication library that unifies vendor-specific backends and enables RDMA-based communication across GPUs without requiring driver modifications. HetCCL introduces two novel mechanisms that enable cross-vendor communication while leveraging optimized vendor libraries, NVIDIA NCCL and AMD RCCL. Evaluations on a multi-vendor GPU cluster show that HetCCL matches NCCL and RCCL performance in homogeneous setups while uniquely scaling in heterogeneous environments, enabling practical, high-performance training with both NVIDIA and AMD GPUs without changes to existing deep learning applications.
    
[^253]: 在二维笔迹中勾勒三维解剖：投影衍生的掩膜能否替代三维分割用于颈椎骨折识别？

    Tracing 3D Anatomy in 2D Strokes: Can Projection-Derived Masks Replace 3D Segmentation for Cervical Spine Fracture Recognition?

    [https://arxiv.org/abs/2601.15235](https://arxiv.org/abs/2601.15235)

    本研究提出一种多阶段流水线，通过从二维投影近似生成椎体掩膜来替代完整的三维分割，在大幅降低计算复杂度的同时，仍能保留足够的诊断上下文以实现颈椎骨折的高精度识别。

    

    颈椎骨折需要快速且准确的诊断，然而由于微小的损伤必须在大体积三维影像中进行评估，自动CT判读仍然充满挑战。我们提出这样的问题：骨折识别是否必须进行完整的三维椎体分割，还是从二维投影近似得到的椎体掩膜即可保留足够的诊断上下文信息。在所提出的多阶段流水线中，YOLOv8检测器从多视图方差投影中定位颈椎（三维平均交并比为94.45%）。随后在基于能量的矢状面和冠状面投影上，使用DenseNet121-Unet执行多标签椎体分割（平均Dice系数为87.86%）。预测的二维掩膜被反投影并融合为每个椎体的近似三维掩膜，用于从原始CT中提取感兴趣体积，再由2.5D时空序列CNN-Transformer模型集成进行分析。在具有自然患病率的完整RSNA 2022数据集上……

    arXiv:2601.15235v5 Announce Type: replace-cross  Abstract: Cervical spine fractures require rapid and accurate diagnosis, yet automatic CT interpretation remains challenging as subtle injuries must be assessed across large 3D volumes. We ask whether full 3D vertebra segmentation is necessary for fracture recognition, or whether vertebra masks approximated from 2D projections preserve sufficient diagnostic context. In the proposed multi-stage pipeline, a YOLOv8 detector localizes the cervical spine from multi-view variance projections (3D mean Intersection over Union 94.45%). Multi-label vertebra segmentation is performed with a DenseNet121-Unet on energy-based sagittal and coronal projections (mean Dice 87.86%). The predicted 2D masks are back-projected and fused into approximate 3D masks per vertebra to extract volumes of interest from the original CT, which are analyzed by an ensemble of 2.5D spatio-sequential CNN-Transformer models. On the full RSNA 2022 cohort at native prevalence 
    
[^254]: 视觉语言模型在物理动力学与意图推理方面落后于人类表现

    Vision-language models lag human performance on physical dynamics and intent reasoning

    [https://arxiv.org/abs/2601.01547](https://arxiv.org/abs/2601.01547)

    该论文提出“目的性空间智能”（TSI）概念，并基于11,328个真实世界视频构建了EscherVerse大规模基准与指令微调数据集，揭示当前最先进的视觉语言模型在整合物理动力学与人类意图进行空间推理方面仍显著落后于人类。

    

    空间智能是具身认知的核心，然而当代AI系统仍然难以在开放世界的人类环境中对物理交互进行推理。尽管在受控基准测试中表现强劲，视觉语言模型往往无法联合建模物理动力学、参考系以及驱动空间变化的潜在人类意图。我们提出了目的性空间智能，它捕捉了空间推理的一个重要方面：整合物理动力学与行动目标来推理行动及其空间后果。为了评估TSI，我们提出了EscherVerse，一个基于11,328个真实世界视频构建的大规模开放世界资源，包括一个包含8,000个样本的基准测试集和一个包含35,963个样本的指令微调集。通过对27个最先进的视觉语言模型的评估，以及对来自11名标注者的首次人类响应的独立分析，我们发现了一种持续存在……

    arXiv:2601.01547v3 Announce Type: replace-cross  Abstract: Spatial intelligence is central to embodied cognition, yet contemporary AI systems still struggle to reason about physical interactions in open-world human environments. Despite strong performance on controlled benchmarks, vision-language models often fail to jointly model physical dynamics, reference frames, and the latent human intentions that drive spatial change. We introduce Teleo-Spatial Intelligence (TSI), which captures an important aspect of spatial reasoning: integrating physical dynamics and action goals to reason about actions and their spatial consequences. To evaluate TSI, we present EscherVerse, a large-scale open-world resource built from 11,328 real-world videos, including an 8,000-example benchmark and a 35,963-example instruction-tuning set. Across 27 state-of-the-art vision-language models and an independent analysis of first-pass human responses from 11 annotators, we identify a persistent teleo-spatial rea
    
[^255]: 改进面向生成式AI流的矩阵指数计算：一种超越Paterson–Stockmeyer的基于泰勒展开的方法

    Improving Matrix Exponential for Generative AI Flows: A Taylor-Based Approach Beyond Paterson--Stockmeyer

    [https://arxiv.org/abs/2512.20777](https://arxiv.org/abs/2512.20777)

    本文提出一种超越经典Paterson–Stockmeyer求值方案的优化泰勒矩阵指数算法，通过严格的误差分析和泰勒阶数与缩放因子的动态选择策略，在满足误差容限的前提下显著降低计算量，为生成式AI流带来大幅加速。

    

    矩阵指数是科学计算与系统仿真中的一个基本算子，其应用涵盖从控制理论、量子力学到现代生成式机器学习等多个领域。尽管结合缩放平分法的Padé逼近长期以来一直是标准方法，但近期基于泰勒展开的方法利用了超越经典Paterson–Stockmeyer技术的多项式求值方案，能够提供更高的精度和更低的计算复杂度。本文提出了一种优化的基于泰勒展开的矩阵指数算法，专门针对生成式AI流的高吞吐量需求而设计。我们给出了严格的误差分析，并开发了一种针对泰勒阶数和缩放因子的动态选择策略，以便在给定误差容限下最小化计算量。大量数值实验表明，我们的方法带来了显著的加速与更（摘要在此处截断）。

    arXiv:2512.20777v3 Announce Type: replace  Abstract: The matrix exponential is a fundamental operator in scientific computing and system simulation, with applications ranging from control theory and quantum mechanics to modern generative machine learning. While Pad\'e approximants combined with scaling and squaring have long served as the standard, recent Taylor-based methods, which utilize polynomial evaluation schemes that surpass the classical Paterson--Stockmeyer technique, offer superior accuracy and reduced computational complexity. This paper presents an optimized Taylor-based algorithm for the matrix exponential, specifically designed for the high-throughput requirements of generative AI flows. We provide a rigorous error analysis and develop a dynamic selection strategy for the Taylor order and scaling factor to minimize computational effort under a prescribed error tolerance. Extensive numerical experiments demonstrate that our approach provides significant acceleration and m
    
[^256]: NEAT：用于3D分子生成的邻域引导、高效、自回归集合Transformer

    NEAT: Neighborhood-Guided, Efficient, Autoregressive Set Transformer for 3D Molecular Generation

    [https://arxiv.org/abs/2512.05844](https://arxiv.org/abs/2512.05844)

    提出NEAT模型，通过新颖的邻域引导训练策略将分子图视为原子集合并学习与顺序无关的token分布，实现原子级置换不变性，在QM9和GEOM-Drugs数据集上达到最先进的3D分子生成质量。

    

    基于Transformer的自回归模型为3D分子生成提供了一种区别于扩散模型和流模型的高效替代方案。然而仍存在一个挑战：标准Transformer架构需要token具有顺序排列，而对于分子中的原子来说，这种顺序并没有天然的定义。先前的工作通过使用规范原子排序来解决这一问题。然而，这些方法相对于原子不具备置换不变性，并且会使下一token的预测偏向于特定的排序惯例。我们通过引入一种新颖的邻域引导训练策略克服了这一限制。我们的模型NEAT（邻域引导、高效、自回归集合Transformer）将分子图视为原子的集合，并在图边界处学习一个与顺序无关的可容许token分布，从而确保了原子级别的置换不变性。NEAT在QM9和GEOM-Drugs数据集上实现了最先进的生成质量。

    arXiv:2512.05844v4 Announce Type: replace-cross  Abstract: Transformer-based autoregressive models offer an efficient alternative to diffusion- and flow-based approaches for generating 3D molecules. One challenge remains: standard transformer architectures require a sequential ordering of tokens, which is not inherently defined for the atoms in a molecule. Prior works have addressed this by using canonical atom orderings. However, these approaches are not permutation invariant w.r.t. atoms and bias next-token prediction towards ordering conventions. We overcome this limitation by introducing a novel neighborhood-guided training strategy. Our model, NEAT (Neighborhood-Guided, Efficient, Autoregressive Set Transformer) treats molecular graphs as sets of atoms and learns an order-agnostic distribution over admissible tokens at the graph boundary, thereby ensuring atom-level permutation invariance. NEAT achieves state-of-the-art generation quality on the QM9 and GEOM-Drugs datasets, enable
    
[^257]: 基于分位数曲线的模拟器保真度无模型评估

    Model-Free Assessment of Simulator Fidelity via Quantile Curves

    [https://arxiv.org/abs/2512.05024](https://arxiv.org/abs/2512.05024)

    该论文提出一种无模型方法，通过为潜在总体参数构建置信集并估计分位数曲线，来稳健地量化生成式AI模拟器与真实系统之间随情景变化的“仿真-现实”差距。

    

    随着生成式AI模型越来越多地被用于模拟真实世界系统，量化“仿真到现实”差距变得至关重要。对于每个感兴趣的输入设置——我们称之为“情景”，例如调查问题或运行条件——真实系统和模拟系统都与未观测的潜在总体参数相关联，且二者的差异因情景而异。一个根本性的挑战在于，对于任何给定情景，这种差异无法直接观测，因为两个系统都只能通过有限样本来访问，且各情景间的样本量往往大小不一。因此，标准的预测推断方法并不适用，因为它们量化的是可观测输出的不确定性，而非潜在总体参数的不确定性。为解决这一问题，我们为这些潜在参数构建置信集，并据此推导出仿真-现实差距的稳健代理量。随后，我们估计分位数函数曲线……

    arXiv:2512.05024v4 Announce Type: replace-cross  Abstract: As generative AI models are increasingly used to simulate real-world systems, quantifying the ``sim-to-real'' gap is critical. For each input setting of interest -- which we call a \emph{scenario}, such as a survey question or operating condition -- the real and simulated systems are associated with unobserved latent population parameters, and their discrepancy varies across scenarios. A fundamental challenge is that, for any given scenario, this discrepancy cannot be observed directly, since both systems are accessible only through finite samples, often of heterogeneous sizes across scenarios. Standard predictive inference methods are therefore ill-suited, as they quantify uncertainty in observable outputs rather than latent population parameters. To address this, we construct confidence sets for these latent parameters and use them to derive a robust proxy for the sim-to-real discrepancy. We then estimate the quantile functio
    
[^258]: 使用似然加权归一化流的多模态后验摊销推断

    Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows

    [https://arxiv.org/abs/2512.04954](https://arxiv.org/abs/2512.04954)

    该论文提出用似然加权重要性采样训练归一化流以高效推断多模态后验分布，并发现用与目标众数数量匹配的高斯混合模型初始化流可以消除模式间的虚假概率桥梁，从而显著提升重建保真度。

    

    我们提出了一种后验估计技术，该技术使用通过似然加权重要性采样训练的归一化流，这是神经重要性采样的特殊情况，其中提议分布即为先验分布。这种方法能够在无需后验训练样本的情况下，高效地推断高维逆问题中的理论参数。我们在二维和三维的多模态基准任务上实现了该方法以检验其有效性。本研究的一个关键观察是基础分布对建模后验的影响。我们发现，标准的单峰基础分布会在分离良好的众数之间留下虚假的概率“桥梁”。我们证明，使用与目标众数数量相匹配的高斯混合模型来初始化流，可以显著提高重建保真度（通过若干距离和散度指标来衡量）。最后，我们将该方法应用于一个精选的……

    arXiv:2512.04954v4 Announce Type: replace  Abstract: We present a technique for posterior estimation using Normalizing Flows trained with likelihood-weighted importance sampling, the special case of neural importance sampling in which the proposal is the prior. This approach allows for the efficient inference of theoretical parameters in high-dimensional inverse problems without the need for posterior training samples. We implement the method on multi-modal benchmark tasks in 2D and 3D to check for the efficacy. A critical observation of our study is the impact of the base distribution on the modelled posteriors. We find that standard unimodal base distributions leave spurious probability \textit{bridges} between well-separated modes. We demonstrate that initializing the flow with a Gaussian Mixture Model that matches the number of target modes significantly improves reconstruction fidelity, as measured by some distance and divergence metrics. Finally, we apply this method to a curated
    
[^259]: E0：基于Tweedie离散扩散的视觉-语言-动作模型表达性细粒度离散动作预测

    E0: Expressive Fine-Grained Discrete Action Prediction for Vision-Language-Action Models via Tweedie Discrete Diffusion

    [https://arxiv.org/abs/2511.21542](https://arxiv.org/abs/2511.21542)

    提出E0框架，通过Tweedie离散扩散将动作生成建模为对量化动作token的迭代去噪过程，实现了与token化推理天然契合的表达性细粒度离散动作预测，从而解决VLA模型泛化能力差和动作粗糙不稳定的问题。

    

    视觉-语言-动作模型通过整合视觉感知、语言理解和控制生成，为机器人操作提供了一个统一的框架。然而，现有的VLA系统仍然难以在多样化的任务、场景和相机视角之间实现泛化，并且常常产生粗糙或不稳定的动作。我们认为这些限制与VLA设置中动作的结构特性密切相关，包括动作分布固有的多峰性质、预训练VLM/VLA骨干网络基于token的符号推理，以及真实世界机器人控制所施加的有效有限分辨率。受这些特性的启发，我们提出了E0，一个Tweedie离散扩散框架，它将动作生成形式化为对量化动作token的迭代去噪过程。通过在离散动作空间中进行操作并采用有原则的扩散过程，E0自然地与基于token的推理相契合。

    arXiv:2511.21542v3 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. However, existing VLA systems still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We argue that these limitations are closely tied to the structural properties of actions in VLA settings, including the inherent multi-peaked nature of action distributions, the token-based symbolic reasoning of pretrained VLM/VLA backbones, and the effective finite resolution imposed by real-world robotic control. Motivated by these properties, we introduce E0, a tweedie discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. By operating in a discrete action space with a principled diffusion process, E0 naturally aligns with token-based reasoning
    
[^260]: 流形近似带来鲁棒的核对齐

    Manifold Approximation leads to Robust Kernel Alignment

    [https://arxiv.org/abs/2510.22953](https://arxiv.org/abs/2510.22953)

    提出流形近似核对齐（MKA），通过将流形局部邻域几何融入核对齐任务，去除启发式度加权、给出闭式对齐分数及马尔可夫概率解释，并证明其对邻域大小的稳定性，从而实现更鲁棒的表示比较。

    

    中心化核对齐（CKA）是比较表示、判断网络等价性以及开展神经科学研究的一种常用度量。然而，CKA 没有考虑底层的流形结构，并且依赖众多启发式方法，导致其在不同数据规模下表现不一致。在本工作中，我们提出了流形近似核对齐（MKA），将流形几何（即局部邻域）融入到对齐任务中。我们为 MKA 推导了一个理论框架。我们的构造去除了依赖于表示的度加权，得到了闭式的对齐分数，并将每一行赋予了作为局部马尔可夫转移分布的概率解释。我们进一步建立了关于邻域大小的核级别与分数级别的稳定性。我们在合成数据集和真实世界示例上进行了实证评估，以刻画 MKA 并将其与同类方法进行比较。

    arXiv:2510.22953v3 Announce Type: replace-cross  Abstract: Centered kernel alignment (CKA) is a popular metric for comparing representations, determining equivalence of networks, and conducting neuroscience research. However, CKA does not account for the underlying manifold and relies on numerous heuristics that cause it to behave differently at different scales of data. In this work, we propose Manifold-approximated Kernel Alignment (MKA), which incorporates manifold geometry (i.e., local neighborhood) into the alignment task. We derive a theoretical framework for MKA. Our construction removes representation-dependent degree weighting, yields a closed-form alignment score, and gives each row a probabilistic interpretation as a local Markov transition distribution. We further establish kernel- and score-level stability with respect to the neighborhood size. We perform empirical evaluations on synthetic datasets and real-world examples to characterize and compare MKA to its contemporari
    
[^261]: CS-54k：一个基于论文的AI研究问答语料库与基准

    CS-54k: A Paper-Grounded Corpus and Benchmark for AI Research QA

    [https://arxiv.org/abs/2510.20279](https://arxiv.org/abs/2510.20279)

    提出了CS-54k，一个从14,474篇开放许可的AI研究论文中构建的包含5.4万个问答对的语料库与基准，涵盖论文的八个核心组成部分，为AI研究问答提供了可用于模型评估与后训练的结构化监督资源。

    

    大语言模型（LLMs）正越来越多地被用于支持计算机科学研究。为了可靠地发挥作用，这些系统必须能够处理分布在研究论文中以及研究过程不同方面的信息。然而，能够将此类信息系统地组织成结构化监督、同时服务于模型评估与后训练的大规模资源仍然有限。为了填补这一空白，我们提出了一个系统化的“论文到问答”构建流水线，将AI研究论文中的信息组织为由来源证据支持的结构化问答对。利用该流水线，我们构建了CS-54k，这是一个包含54,000个问答对的语料库，源自14,474篇开放许可的AI研究论文。该语料库涵盖了八个常见的论文组成部分，包括研究领域、先前工作、现有挑战、研究动机、发现与假设、方法、实验设置以及实验结果。我们进一步……

    arXiv:2510.20279v4 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly being used to support computer science research. To contribute reliably, these systems must be able to work with information distributed across research papers and across different aspects of the research process. However, large-scale resources that systematically organize such information into structured supervision for both model evaluation and post-training remain limited. To address this gap, we propose a systematic paper-to-QA construction pipeline that organizes information from AI research papers into structured QA pairs supported by source evidence. Using this pipeline, we construct CS-54k, a 54k-pair corpus derived from 14,474 openly licensed AI research papers. The corpus covers eight recurring paper components, including research domain, prior work, existing challenges, motivation, findings and assumptions, methods, experimental settings, and experimental results. We further de
    
[^262]: 使用带强化学习引导标注的SILVER解释深度强化学习策略：一种面向高维与多动作环境的模型级方法

    Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling: A Model-level Approach to High-dimensional and Multi-action Environments

    [https://arxiv.org/abs/2510.19244](https://arxiv.org/abs/2510.19244)

    提出了一种带强化学习引导标注的增强SILVER框架，通过将RL策略自身的动作输出纳入边界点识别，将基于Shapley回归的策略解释方法扩展到高维和多动作环境。

    

    深度强化学习（RL）取得了卓越的性能，但缺乏可解释性，限制了对策略行为的信任。现有的SILVER框架（Li, Siddique, and Cao 2025）通过基于Shapley的回归来解释RL策略，但仍然局限于低维、二元动作的领域。我们提出了带强化学习引导标注的SILVER（SILVER with RL-guided labeling），这是SILVER的一个增强变体，通过将RL策略自身的动作输出纳入边界点识别，将SILVER扩展到多动作和高维环境中。我们的方法首先从图像观测中提取紧凑的特征表示，进行基于SHAP的特征归因，然后采用强化学习引导标注来生成行为一致的边界数据集。随后训练代理模型（如决策树和基于回归的函数）来解释RL策略的决策结构。我们在两个Atari环境中评估了所提出的框架。

    arXiv:2510.19244v4 Announce Type: replace  Abstract: Deep reinforcement learning (RL) achieves remarkable performance but lacks interpretability, limiting trust in policy behavior. The existing SILVER framework (Li, Siddique, and Cao 2025) explains RL policy via Shapley-based regression but remains restricted to low-dimensional, binary-action domains. We propose SILVER with RL-guided labeling, an enhanced variant that extends SILVER to multi-action and high-dimensional environments by incorporating the RL policy's own action outputs into the boundary points identification. Our method first extracts compact feature representations from image observations, performs SHAP-based feature attribution, and then employs RL-guided labeling to generate behaviorally consistent boundary datasets. Surrogate models, such as decision trees and regression-based functions, are subsequently trained to interpret RL policy's decision structure. We evaluate the proposed framework on two Atari environments u
    
[^263]: 针对下游天气预报模型的对抗攻击：在热带气旋轨迹预测中的应用

    Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction

    [https://arxiv.org/abs/2510.10140](https://arxiv.org/abs/2510.10140)

    本文提出了Cyc-Attack对抗攻击方法，通过对上游天气预报模型施加细微扰动来操纵下游热带气旋轨迹预测，克服了检测系统黑盒不可微分和热带气旋样本极度稀缺导致的类别不平衡两大难题。

    

    基于深度学习的天气预报（DLWF）模型利用过去的天气观测数据生成未来的预报，支持广泛的下游应用，其中包括热带气旋（TC）预测。本文研究了这些模型对对抗攻击的脆弱性，即对上游预报施加细微扰动即可改变下游的热带气旋轨迹预测。构建扰动输入以将下游输出引导至攻击者指定的轨迹是一个极具挑战性的问题，原因如下：首先，传统的热带气旋检测系统是不透明、不可微分的黑盒，使得标准的基于梯度的攻击方法无法实施；其次，热带气旋事件的极端稀有性导致了严重的类别不平衡问题，使得难以开发出能够生成与目标轨迹相符的逼真气旋路径的攻击方法。为克服这些限制，我们提出了Cyc-Attack这一新颖的方法。

    arXiv:2510.10140v3 Announce Type: replace  Abstract: Deep learning-based weather forecasting (DLWF) models leverage past weather observations to generate future forecasts, supporting a wide range of downstream applications, including tropical cyclone (TC) prediction. In this paper, we investigate the vulnerability of these models to adversarial attacks, where subtle perturbations to upstream forecasts can alter downstream TC trajectory predictions. Crafting perturbed inputs to steer downstream outputs toward attacker-specified trajectories is a challenging problem for several reasons. First, conventional TC detection systems are opaque, non-differentiable black boxes, making standard gradient-based attacks infeasible. Second, the extreme rarity of TC events leads to a severe class imbalance problem, making it difficult to develop attack methods that produce realistic-looking cyclone paths aligned with target trajectories. To overcome these limitations, we propose Cyc-Attack, a novel me
    
[^264]: 基于误设定鞍点问题的学习与优化同步进行

    Simultaneous Learning and Optimization via Misspecified Saddle Point Problems

    [https://arxiv.org/abs/2510.05241](https://arxiv.org/abs/2510.05241)

    该论文提出了一个将优化与参数学习统一处理的误设定鞍点问题框架，并设计了两种基于加速原始-对偶（APD）方法的算法，均达到 $\mathcal{O}(\log K/K)$ 收敛率，其中学习感知变体通过调整动量更新显式建模参数动态，并获得更优的 $\mathcal{O}(1)$ 约束保证。

    

    我们研究一类误设定的鞍点（SP）问题，其中优化目标依赖于一个必须从数据中同步学习的未知参数。与现有研究中假设参数完全已知或预先估计的做法不同，我们的框架将优化与学习整合到一个统一的公式化中，从而实现了一个更灵活的问题类别。为解决这一设定，我们基于Hamedani与Aybat（2021）的加速原始-对偶（APD）方法提出了两种算法。具体而言，我们首先通过将不断演化的参数估计直接代入原始-对偶更新，分析了APD方法的朴素扩展；随后，我们设计了一种新的“学习感知”APD变体，通过调整动量更新来显式考虑参数动态。两种方法均实现了可证明的 $\mathcal{O}(\log K / K)$ 收敛率，而学习感知方法则获得了更优的 $\mathcal{O}(1)$ 约束（摘要在此处截断）。

    arXiv:2510.05241v2 Announce Type: replace  Abstract: We study a class of misspecified saddle point (SP) problems, where the optimization objective depends on an unknown parameter that must be learned concurrently from data. Unlike existing studies that assume parameters are fully known or pre-estimated, our framework integrates optimization and learning into a unified formulation, enabling a more flexible problem class. To address this setting, we propose two algorithms based on the accelerated primal-dual (APD) by Hamedani & Aybat 2021. In particular, we first analyze the naive extension of the APD method by directly substituting the evolving parameter estimates into the primal-dual updates; then, we design a new learning-aware variant of the APD method that explicitly accounts for parameter dynamics by adjusting the momentum updates. Both methods achieve a provable convergence rate of $\mathcal{O}(\log K / K)$, while the learning-aware approach attains a tighter $\mathcal{O}(1)$ cons
    
[^265]: 图构自身提示

    Graph Your Own Prompt

    [https://arxiv.org/abs/2509.23373](https://arxiv.org/abs/2509.23373)

    该论文提出图一致性正则化（GCR），通过在任意深度插入无参数的图一致性层，将特征相似性图与由模型自身预测导出的类感知掩码预测图对齐，实现自我提示式训练，使特征表示更具类一致性和语义意义。

    

    我们提出了图一致性正则化（Graph Consistency Regularization, GCR），这是一种新颖的框架，它将从模型预测中导出的关系图结构注入学习过程，以促进具有类感知能力且语义上有意义的特征表示。作为一种自我提示（self-prompting）的形式，GCR 使模型能够利用自身输出对其内部结构进行优化。尽管深度网络能够学习到丰富的表示，但这些表示往往捕捉到与模型预测语义相矛盾的、含噪的类间相似性。GCR 通过在任意深度引入无参数的图一致性层（Graph Consistency Layers, GCLs）来解决这一问题。每个 GCL 构建一个批级别的特征相似性图，并将其与全局的、类感知的掩码预测图对齐——该预测图通过用类内指标调制 softmax 预测相似性而得到。这种对齐强制特征层面的关系反映类一致的预测行为，从而充当一种语义层面的约束。（原文摘要在此处截断）

    arXiv:2509.23373v4 Announce Type: replace-cross  Abstract: We propose Graph Consistency Regularization (GCR), a novel framework that injects relational graph structures, derived from model predictions, into the learning process to promote class-aware, semantically meaningful feature representations. Functioning as a form of self-prompting, GCR enables the model to refine its internal structure using its own outputs. While deep networks learn rich representations, these often capture noisy inter-class similarities that contradict the model's predicted semantics. GCR addresses this issue by introducing parameter-free Graph Consistency Layers (GCLs) at arbitrary depths. Each GCL builds a batch-level feature similarity graph and aligns it with a global, class-aware masked prediction graph, derived by modulating softmax prediction similarities with intra-class indicators. This alignment enforces that feature-level relationships reflect class-consistent prediction behavior, acting as a seman
    
[^266]: 再生核希尔伯特空间中的贝叶斯转移算子

    Bayesian Transfer Operators in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2509.22482](https://arxiv.org/abs/2509.22482)

    该论文提出高斯过程动态模态分解（GP-DMD），一种将稀疏变分高斯过程与基于核的Koopman及Perron–Frobenius算子近似相结合的贝叶斯转移算子框架，可同时实现不确定性量化、可扩展的稀疏近似和数据自适应字典学习。

    

    转移算子通过传播观测量或概率分布，为非线性动力系统提供了一种线性描述。在再生核希尔伯特空间中，这些算子通过条件均值嵌入来表示，其正则化估计量与高斯过程回归的后验均值相一致。基于这一对应关系，我们提出了高斯过程动态模态分解，这是一种贝叶斯转移算子方法，它将稀疏变分高斯过程与基于核的Koopman算子和Perron–Frobenius算子近似相结合。所得到的框架为不确定性量化、可扩展的稀疏近似以及面向预测与谱分析的数据自适应字典学习提供了一条有原则的途径。此外，该模型选择公式自然地产生了前向–后向算子结构，从而建立了贝叶斯转移算子

    arXiv:2509.22482v3 Announce Type: replace  Abstract: Transfer operators provide a linear description of nonlinear dynamical systems by propagating observables or probability distributions. In reproducing kernel Hilbert spaces, these operators are represented through conditional mean embeddings, whose regularized estimator coincides with the posterior mean of Gaussian process regression. Building on this correspondence, we formulate Gaussian Process Dynamic Mode Decomposition, a Bayesian transfer-operator method that combines sparse variational Gaussian processes with kernel-based Koopman and Perron--Frobenius operator approximation. The resulting framework provides a principled route to uncertainty quantification, scalable sparse approximations, and data-adaptive dictionary learning for prediction and spectral analysis. Additionally, the model-selection formulation naturally gives rise to a forward--backward operator structure, which establishes a connection between Bayesian transfer-o
    
[^267]: ReLU神经网络的环面几何

    Toric geometry of ReLU neural networks

    [https://arxiv.org/abs/2509.05894](https://arxiv.org/abs/2509.05894)

    该论文建立了ReLU神经网络与环面几何之间的联系，定义了ReLU扇、ReLU环面簇和ReLU卡地亚除子等新概念，借助代数几何工具系统研究了网络的精确函数实现问题与表达能力，并揭示了热带几何与环面几何之间的对应关系。

    

    给定一个连续分段线性函数 $f:\mathbb{R}^{n_0} \to \mathbb{R}$ 和一个固定的前馈ReLU神经网络架构，精确函数实现问题问的是该架构的网络何时能够实现 $f$。与之相对，表达能力问题关注的是刻画由给定架构能够精确实现的所有函数。为了系统地解决这些问题，我们在环面几何与ReLU神经网络之间建立了联系，使得代数几何工具能够用于分析这些结构。从一个具有有理权重的无偏ReLU网络出发，我们定义了与该网络相关联的ReLU扇、ReLU环面簇以及ReLU卡地亚除子。这项工作还将热带几何与环面几何联系起来：前者捕捉ReLU网络的组合数据，而后者则恢复其底层的几何结构。作为应用，我们获得了……的完整刻画。

    arXiv:2509.05894v2 Announce Type: replace-cross  Abstract: Given a continuous piecewise linear function $f:\mathbb{R}^{n_0} \to \mathbb{R}$ and a fixed architecture of feedforward ReLU neural networks, the exact function realization problem asks when a network of that architecture realizes $f$. In contrast, expressivity concerns characterizing all functions exactly realizable by a given architecture. To address these questions systematically, we establish a connection between toric geometry and ReLU neural networks, allowing tools from algebraic geometry to analyze the structures. Starting with an unbiased ReLU network with rational weights, we define the ReLU fan, the ReLU toric variety, and the ReLU Cartier divisor associated with the network. This work also relates the tropical and the toric geometry: the former captures combinatorial data of ReLU networks, while the latter recovers the underlying geometric structure. As an application, we obtain a complete characterization of the f
    
[^268]: 结构化基函数网络：以损失为中心、多样性可控的多假设集成

    Structured Basis Function Networks: Loss-Centric Multi-Hypothesis Ensembles with Controllable Diversity

    [https://arxiv.org/abs/2509.02792](https://arxiv.org/abs/2509.02792)

    本文提出结构化基函数网络，通过受Bregman质心几何启发的损失感知监督聚合机制与可调的多样性控制参数，将集成学习与多假设预测统一在同一框架中，在回归、图像识别和大规模视频分类等任务上均取得优异表现。

    

    集成学习与多假设预测是提升预测性能与鲁棒性的两种互补范式。本工作提出了结构化基函数网络，这是一种损失感知的集成框架，通过受预测空间中损失诱导的Bregman质心几何启发的监督聚合机制，将这两种范式统一起来。该方法构建了由基础预测组成的结构化数据集，并直接针对观测标签训练参数化组合器，其中质心公式仅作为几何参考而非精确的质心求解器，同时通过一个多样性控制参数调节预测器之间的专门化程度。该框架在回归与分类等多种任务上进行了评估，包括表格数据、图像识别以及大规模时序结构视频分类。实证结果表明，s-BFN提供了强大而优异的性能。

    arXiv:2509.02792v2 Announce Type: replace  Abstract: Ensemble learning and multi-hypothesis prediction are two complementary paradigms for improving predictive performance and robustness. This work introduces the Structured Basis Function Network (s-BFN), a loss-aware ensemble framework that unifies these paradigms through a supervised aggregation mechanism motivated by the loss-induced Bregman-centroid geometry of prediction space. The approach constructs a structured dataset of base predictions and trains a parametric combiner directly against the observed labels, with the centroidal formulation serving as a geometric reference rather than an exact centroid solver, while a diversity-control parameter regulates specialization among predictors. The proposed framework is evaluated across regression and classification tasks, including tabular data, image recognition, and large-scale temporally structured video classification. Empirical results show that s-BFN provides strong performance 
    
[^269]: 使用 CTF-Dojo 训练语言模型智能体发现漏洞

    Training Language Model Agents to Find Vulnerabilities with CTF-Dojo

    [https://arxiv.org/abs/2508.18370](https://arxiv.org/abs/2508.18370)

    提出了 CTF-Dojo——首个面向 LLM 训练的大规模可验证执行环境，包含 658 个容器化 CTF 挑战，并配套自动化流水线 CTF-Forge，可数分钟内将公开工件转化为即用型执行环境，仅需 486 条高质量轨迹即可训练出擅长漏洞发现的智能体。

    

    大型语言模型（LLM）在可执行运行时环境中训练时展现出了卓越的能力，尤其通过可验证的反馈循环在软件工程任务中表现突出。然而，可扩展且可泛化的基于执行的环境仍然稀缺，限制了训练更强机器学习智能体的进展。我们提出了 CTF-Dojo，这是首个专为训练具有可验证反馈的 LLM 而设计的大规模可执行运行时环境，包含 658 个功能完整的夺旗（CTF）式挑战，并以 Docker 容器化方式确保可重现性。为了在没有人工干预的情况下实现快速扩展，我们开发了 CTF-Forge，这是一个自动化流水线，能够在几分钟内将公开可用的工件转换为可直接使用的执行环境，省去了传统上需要数周专家配置的过程。我们仅使用来自 CTF-D（原文在此处截断）的 486 条高质量、经过执行验证的轨迹对基于 LLM 的智能体进行训练……

    arXiv:2508.18370v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-D
    
[^270]: 面向可解释图神经网络的鲁棒代理模型

    Robust Surrogate Modeling for Explainable Graph Neural Networks

    [https://arxiv.org/abs/2508.01925](https://arxiv.org/abs/2508.01925)

    该论文提出通过迭代重加权策略训练一个鲁棒代理模型，解决了训练图与解释图之间的分布偏移问题，从而为图神经网络提供更忠实可靠的解释。

    

    图神经网络（GNN）在广泛的图相关学习任务中取得了卓越的性能。然而，解释其预测仍然是一个具有挑战性的问题，尤其是因为训练时使用的图与解释时遇到的图之间存在不匹配。大多数现有方法通过在加权图上优化软边掩码来突出重要的子结构，但这些图与GNN训练时所使用的无权图并不相同。这种分布偏移会导致不可靠的梯度和解释质量的下降，特别是在生成小规模、稀疏的子图时。为了解决这一问题，我们提出对鲁棒代理模型进行解释，该代理模型的解释能够为原始模型的解释提供忠实的代理。具体而言，我们通过一种迭代重加权策略来训练该代理模型，该策略在识别解释子图与使模型适应所得结果之间交替进行。

    arXiv:2508.01925v2 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) have achieved remarkable performance in a wide range of graph-related learning tasks. However, explaining their predictions remains a challenging problem, especially due to the mismatch between the graphs used during training and those encountered during explanation. Most existing methods optimize soft edge masks on weighted graphs to highlight important substructures, but these graphs differ from the unweighted graphs on which GNNs are trained. This distributional shift leads to unreliable gradients and degraded explanation quality, especially when generating small, sparse subgraphs. To address this issue, we propose explaining a robust surrogate whose explanations provide a faithful proxy for those of the original model. Specifically, we train this surrogate model through an iterative re-weighting strategy that alternates between identifying explanation subgraphs and adapting the model to the resulting 
    
[^271]: Cyber-Zero：无需运行时环境训练网络安全智能体

    Cyber-Zero: Training Cybersecurity Agents without Runtime

    [https://arxiv.org/abs/2508.00910](https://arxiv.org/abs/2508.00910)

    Cyber-Zero是首个无需运行时环境的网络安全智能体训练框架，通过利用公开CTF解题报告和基于角色扮演的LLM模拟来合成高质量交互轨迹，使训练出的智能体在三个CTF基准测试上相比基线模型最高取得11.4%的绝对性能提升。

    

    大型语言模型（LLM）在软件工程任务中通过可执行运行时环境的训练已取得显著成功，尤其是在解决GitHub问题方面。然而，此类运行时环境在其他领域往往不可用，特别是在网络安全领域，挑战配置和执行环境通常是短暂的或受限的。我们提出了Cyber-Zero，这是首个无需运行时环境的框架，用于合成高质量的智能体轨迹以训练网络安全LLM。Cyber-Zero利用公开可用的CTF（夺旗赛）解题报告，并采用基于角色扮演的LLM模拟来逆向工程运行时行为，在无需实际环境的情况下生成逼真的、长时程的交互序列。使用Cyber-Zero合成的轨迹训练的LLM智能体，在三个知名CTF基准测试上相比基线模型取得了最高11.4%的绝对性能提升。

    arXiv:2508.00910v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have achieved remarkable success in software engineering tasks when trained with executable runtime environments, particularly in resolving GitHub issues. However, such runtime environments are often unavailable in other domains, especially cybersecurity, where challenge configurations and execution contexts are ephemeral or restricted. We present Cyber-Zero, the first runtime-free framework for synthesizing high-quality agent trajectories to train cybersecurity LLMs. Cyber-Zero leverages publicly available CTF writeups and employs persona-driven LLM simulation to reverse-engineer runtime behaviors and generate realistic, long-horizon interaction sequences without actual environments. Using trajectories synthesized by Cyber-Zero, we train LLM-based agents that achieve up to 11.4% absolute performance gains over baseline models on three prominent CTF benchmarks: InterCode-CTF, NYU CTF Bench, and Cybe
    
[^272]: 在异构噪声下恢复高维数据的欧几里得几何结构

    Recovering the Euclidean Geometry of High-Dimensional Data Under Heterogeneous Noise

    [https://arxiv.org/abs/2507.18520](https://arxiv.org/abs/2507.18520)

    本文证明了对于具有底层低维几何结构的高维数据，即使在异构噪声严重且不均匀地破坏成对欧几里得距离的情况下，仍然可以恢复出其底层干净的欧几里得几何结构。

    

    成对欧几里得距离为高维数据提供了一种简单的几何表示，是包括聚类、可视化和流形学习在内的许多数据分析方法的基础。然而，在现实应用中，这些距离可能受到异构噪声的严重破坏，即不同的数据点会受到不同且可能很大的噪声水平影响。这种噪声会不均匀地扭曲距离，导致对底层数据几何结构的错误表示。在这项工作中，我们探讨是否仍然能够从被破坏的成对距离中恢复出底层的干净欧几里得几何结构。我们对这个问题给出了肯定的回答，表明——也许令人惊讶的是——即使噪声幅度在不同观测之间变化很大，这种恢复在广泛的具有底层低维几何结构的高维场景中也是可能的。我们开发了一种有原则的方法……

    arXiv:2507.18520v3 Announce Type: replace-cross  Abstract: Pairwise Euclidean distances provide a simple geometric representation of high-dimensional data and underlie many data analysis methods, including clustering, visualization, and manifold learning. However, in real-world applications, these distances can be severely corrupted by heterogeneous noise, where different data points are subject to different, and potentially large, noise levels. Such noise distorts the distances inhomogeneously, leading to a misrepresentation of the underlying data geometry. In this work, we ask whether the underlying clean Euclidean geometry can nevertheless be recovered from the corrupted pairwise distances. We answer this question affirmatively, showing that, perhaps surprisingly, such recovery is possible across a broad class of high-dimensional settings with underlying low-dimensional geometry, even when the noise magnitudes vary substantially across observations. We develop a principled approach 
    
[^273]: HARDTESTGEN：面向大语言模型算法编程的高质量强化学习验证器生成流水线

    HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithmic Coding

    [https://arxiv.org/abs/2505.24098](https://arxiv.org/abs/2505.24098)

    HARDTESTGEN提出了一种为算法编程问题自动合成高质量测试用例的流水线，并构建了包含2.66万道题的HARDTESTS数据集，其测试用例在验证LLM生成代码方面的精确度比现有测试提升11.22个百分点，有效支撑拒绝采样和强化学习等下游后训练。

    

    验证器为大语言模型（LLM）的强化学习提供重要的奖励信号。然而，开发或创建可靠的验证器极具挑战性，尤其是在代码生成任务中。一个伪装巧妙的错误解答程序可能只有通过精心人工编写的边缘测试用例才能被发现，而这类用例很难自动合成。为了解决这一问题，我们提出了HARDTESTGEN，一种为算法编程问题合成高质量测试用例的方法。我们构建了一个全面的算法编程数据集HARDTESTS，包含2.66万个问题及高质量的合成测试。与现有测试相比，HARDTESTGEN生成的测试在验证LLM生成代码方面表现出显著更高的准确率（精确度提升11.22个百分点，即在预测为正确的代码中实际正确代码所占的比例）。我们还证明，下游后训练——包括拒绝采样和强化学习（RL）——（摘要此处不完整）

    arXiv:2505.24098v2 Announce Type: cross  Abstract: Verifiers provide important reward signals for reinforcement learning of large language models (LLMs). However, it is challenging to develop or create reliable verifiers, especially for code generation tasks. A well-disguised wrong solution program may only be detected by carefully human-written edge cases that are difficult to synthesize automatically. To address this issue, we propose HARDTESTGEN, an approach to synthesize high-quality test cases for algorithmic coding problems. We curate a comprehensive algorithmic programming dataset HARDTESTS with 26.6k problems and high-quality synthetic tests. Compared with existing tests, HARDTESTGEN tests demonstrate significantly higher accuracy in verifying LLM-generated code (+11.22 percentage points in precision, the percentage of actually correct code within the predicted correct ones). We also show that downstream post-training -- including rejection sampling and reinforcement learning (
    
[^274]: 用于异构图异常检测的卡方小波图神经网络

    Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection

    [https://arxiv.org/abs/2505.18934](https://arxiv.org/abs/2505.18934)

    提出基于新型卡方小波滤波器的谱图神经网络框架ChiGAD，通过多图卡方滤波捕获跨元路径异常信号、交互式元图卷积在维度对齐中保留高频信息，有效解决了异构图异常检测中的语义捕获与类别不平衡难题。

    

    异构网络中的图异常检测由于节点和边的异质性而面临独特的挑战。现有的图神经网络（GNN）方法主要针对同构图异常检测，因此无法解决三个关键问题：（C1）跨不同元路径捕获异常信号和丰富语义；（C2）在异构信息网络（HIN）维度对齐中保留高频内容；（C3）在类别不平衡条件下从困难异常样本中有效学习。为克服这些问题，我们提出了ChiGAD，一种基于新型卡方滤波器的谱图神经网络框架，其灵感来源于小波在多个领域的有效性。具体而言，ChiGAD包含：（1）多图卡方滤波器，通过对每个元路径图应用专用卡方滤波器来捕获异常信息；（2）交互式元图卷积，在保留高频信息的同时对齐特征并融合异构消息。

    arXiv:2505.18934v3 Announce Type: replace-cross  Abstract: Graph Anomaly Detection (GAD) in heterogeneous networks presents unique challenges due to node and edge heterogeneity. Existing Graph Neural Network (GNN) methods primarily focus on homogeneous GAD and thus fail to address three key issues: (C1) Capturing abnormal signal and rich semantics across diverse meta-paths; (C2) Retaining high-frequency content in HIN dimension alignment; and (C3) Learning effectively from difficult anomaly samples with class imbalance. To overcome these, we propose ChiGAD, a spectral GNN framework based on a novel Chi-Square filter, inspired by the wavelet effectiveness in diverse domains. Specifically, ChiGAD consists of: (1) Multi-Graph Chi-Square Filter, which captures anomalous information via applying dedicated Chi-Square filters to each meta-path graph; (2) Interactive Meta-Graph Convolution, which aligns features while preserving high-frequency information and incorporates heterogeneous message
    
[^275]: 随机梯度下降的偏差最优界

    Bias-Optimal Bounds for Stochastic Gradient Descent

    [https://arxiv.org/abs/2505.17965](https://arxiv.org/abs/2505.17965)

    本文在仅假设凸性和光滑性的条件下，为随机梯度下降推导出覆盖全部常数步长范围的偏差最优收敛界，并证明在临界步长处最优偏差虽无法精确达到但可被任意逼近。

    

    随机梯度下降（SGD）的非渐近分析通常得到的界可分解为偏差项和方差项。在本工作中，我们聚焦于偏差部分，研究SGD在多大程度上能够匹配确定性梯度下降的最优收敛行为。仅在目标函数满足（强）凸性和光滑性的假设下，我们为常数步长的全部范围 $\gamma L \in (0,2)$ 推导出了新的界。这包括了此前在无额外方差假设条件下未曾被探索过的临界步长和较大步长区间，且同时涵盖凸和强凸两种情形。这些界是偏差最优的，即对于所有非临界步长，偏差项与梯度下降的最坏情形速率相一致。对于临界步长，我们证明无法获得具有最优偏差项的界，但该最优偏差可以被任意接近地逼近。

    arXiv:2505.17965v3 Announce Type: replace-cross  Abstract: The non-asymptotic analysis of Stochastic Gradient Descent (SGD) typically yields bounds that decompose into a bias term and a variance term. In this work, we focus on the bias component and study the extent to which SGD can match the optimal convergence behavior of deterministic gradient descent. Assuming only (strong) convexity and smoothness of the objective, we derive new bounds for the full range of constant step-sizes $\gamma L \in (0,2)$. This includes critical and large step-size regimes that were previously unexplored without additional variance assumptions, in both the convex and strongly convex cases. Those bounds are bias-optimal, in the sense that the bias term coincides with the worst-case rate of gradient descent, for all non-critical step-sizes. For critical step-sizes, we show that no bounds can be obtained with an optimal bias term, but that this optimal bias can be approximated arbitrarily closely. As an imme
    
[^276]: AbFlowNet：通过扩散模型与GFlowNet融合优化抗体-抗原结合能

    AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion

    [https://arxiv.org/abs/2505.12358](https://arxiv.org/abs/2505.12358)

    AbFlowNet提出了一种将GFlowNet与扩散模型融合的新型生成框架，通过将每个扩散步骤建模为GFlowNet中的状态，在单一训练流程中直接引入能量信号，实现扩散损失与抗体-抗原结合能的联合优化，从而避免了依赖不可靠估计器的昂贵在线强化学习流程。

    

    互补决定区（CDRs）是抗体中促进与特定抗原结合的关键片段。目前用于CDR设计的计算方法仅采用重构损失，并未联合优化结合能——而结合能是衡量抗体效力的关键指标。现有的结合能优化通常通过计算代价高昂的在线强化学习（RL）流程来实现，且这些流程严重依赖不可靠的结合能估计器。本文提出了AbFlowNet，一种将GFlowNet与扩散模型相融合的新型生成框架。通过将每个扩散步骤视为GFlowNet框架中的一个状态，AbFlowNet将能量信号直接融入训练过程，从而联合优化标准扩散损失与结合能，在单一流程中统一了扩散优化与奖励优化。实验结果表明，AbFlowNet的表现优于基础扩散模型……

    arXiv:2505.12358v2 Announce Type: replace-cross  Abstract: Complementarity Determining Regions (CDRs) are critical segments of an antibody that facilitate binding to specific antigens. Current computational methods for CDR design utilize reconstruction losses and do not jointly optimize binding energy, a crucial metric for antibody efficacy. Rather, binding energy optimization is done through computationally expensive Online Reinforcement Learning (RL) pipelines rely heavily on unreliable binding energy estimators. In this paper, we propose AbFlowNet, a novel generative framework that integrates GFlowNet with Diffusion models. By framing each diffusion step as a state in the GFlowNet framework, AbFlowNet jointly optimizes standard diffusion losses and binding energy by directly incorporating energy signals into the training process, thereby unifying diffusion and reward optimization in a single procedure. Experimental results show that AbFlowNet outperforms the base diffusion model by 
    
[^277]: 双向信息流（BIF）——一种用于贝叶斯优化的样本高效层次高斯过程

    Bidirectional Information Flow (BIF) - A Sample Efficient Hierarchical Gaussian Process for Bayesian Optimization

    [https://arxiv.org/abs/2505.11294](https://arxiv.org/abs/2505.11294)

    提出双向信息流（BIF）框架，使层次高斯过程的父子节点间实现持续双向信息交换，从而提升贝叶斯优化的样本效率、训练稳健性，并支持子任务模型的模块化重用。

    

    层次高斯过程（H-GP）模型将问题划分为不同的子任务，允许不同组件分别处理各个部分，使其非常适合具有固有组合结构的问题。然而，现有的H-GP框架通常采用单向信息共享——要么自上而下，要么自下而上——这限制了样本效率并减缓了收敛速度。我们提出双向信息流（BIF），建立持续的双向通信机制。BIF保留了层次模型的模块化结构——父节点将子节点的摘要视为结构化先验，以此对自身后验进行条件化——同时引入自上而下的反馈，将父节点的环境观测软性地分解为子响应。这种相互交换提高了样本效率，实现了稳健的训练，并允许对已学习的子任务模型进行模块化重用。我们解析地证明了具有学习核的高斯过程的遗憾值（regret）……

    arXiv:2505.11294v4 Announce Type: replace  Abstract: Hierarchical Gaussian Process (H-GP) models divide problems into different subtasks, allowing for different components to address each part, making them well-suited for problems with inherent compositional structure. However, existing H-GP frameworks typically employ one-way information sharing - either top-down or bottom-up - which limits sample efficiency and slows convergence. We propose Bidirectional Information Flow (BIF), which establishes continuous two-way communication. BIF retains the modular structure of hierarchical models - the parent conditions its own posterior on child summaries, treating them as structured priors - while introducing top-down feedback to softly decompose environment observations from the parent into sub-responses. This mutual exchange improves sample efficiency, enables robust training, and allows modular reuse of learned subtask models. We prove analytically the regret of a GP with a learned kernel s
    
[^278]: 绘制哥伦比亚七十年哲学图景：对《Ideas y Valores》期刊的动态主题建模研究

    Mapping Seven Decades of Philosophy in Colombia: Dynamic Topic Modelling of Ideas y Valores

    [https://arxiv.org/abs/2412.04236](https://arxiv.org/abs/2412.04236)

    本研究通过动态主题建模技术分析哥伦比亚最具影响力的哲学期刊《Ideas y Valores》七十年来主题的演变，将哲学史的数据驱动研究拓展至哥伦比亚和拉丁美洲语境，发现价值理论、认识论和科学哲学是该地区最突出的哲学研究主题。

    

    以数据驱动的方法研究哲学已成为探索该学科历史的重要工具。然而，该领域的大多数研究都集中在特定地区和子领域的少数期刊上。我们通过应用动态主题建模技术来探索哥伦比亚和拉丁美洲哲学的历史，从而扩展了这一研究的范围。我们的研究考察了哥伦比亚哲学期刊《Ideas y Valores》，该期刊创刊于1951年，目前是该地区最具影响力的学术哲学期刊之一。通过分析该期刊历史中主题的演变，我们识别出哥伦比亚和拉丁美洲语境下哲学话语的多种趋势和特定动态。我们的研究发现，最突出的主题是价值理论（包括伦理学、政治哲学和美学）、认识论和科学哲学。我们还追踪了（原文摘要在此处截断）……

    arXiv:2412.04236v2 Announce Type: replace-cross  Abstract: Data-driven approaches to philosophy have emerged as a valuable tool for studying the history of the discipline. However, most studies in this area have focused on a limited number of journals from specific regions and subfields. We expand the scope of this research by applying dynamic topic modelling techniques to explore the history of philosophy in Colombia and Latin America. Our study examines the Colombian philosophy journal Ideas y Valores, founded in 1951 and currently one of the most influential academic philosophy journals in the region. By analyzing the evolution of topics across the journal's history, we identify various trends and specific dynamics in philosophical discourse within the Colombian and Latin American context. Our findings reveal that the most prominent topics are value theory (including ethics, political philosophy, and aesthetics), epistemology, and the philosophy of science. We also trace the evoluti
    
[^279]: LEGAL-UQA：使用英乌尔都语平行基准测试面向乌尔都语使用者的宪法问答系统

    LEGAL-UQA: Testing Constitutional Question-Answering Systems for Urdu Speakers with a Parallel English-Urdu Benchmark

    [https://arxiv.org/abs/2410.13013](https://arxiv.org/abs/2410.13013)

    提出了LEGAL-UQA，一个包含619个问答对的英乌尔都语平行基准，用于测试面向乌尔都语使用者的巴基斯坦宪法问答系统，研究发现模型排名依赖相似度度量方式、词匹配检索优于嵌入方法，且自动评判模型与人工评分的一致性有限。

    

    法律信息助手可以帮助乌尔都语使用者了解其宪法权利，但构建可靠的系统需要适合该语言的测试数据和评估方法。我们提出了LEGAL-UQA，这是一个包含619个问答对的英乌尔都语平行基准，每个问答对均与巴基斯坦宪法的段落相关联。每个示例包含两种语言的问题、参考答案和源段落。我们分别评估两个组成部分：在提供源段落的情况下回答乌尔都语问题，以及检索生成每个问题的源段落。我们还将JEV（一种用作自动评判器的决策模型）与人工评分进行比较。研究结果表明，模型排名取决于答案相似度的度量方式，而简单的词匹配方法在检索源段落方面比所测试的嵌入方法更为有效。在评估的截止阈值处，JEV与人工评分的一致性低于基线方法。

    arXiv:2410.13013v2 Announce Type: replace-cross  Abstract: Legal information assistants could help Urdu speakers understand their constitutional rights, but building reliable systems requires test data and evaluation methods suited to the language. We present LEGAL-UQA, a parallel English-Urdu benchmark of 619 question-answer pairs linked to passages from Pakistan's Constitution. Each example includes the question, reference answer, and source passage in both languages. We evaluate two components separately: answering Urdu questions when the source passage is supplied, and retrieving the passage from which each question was generated. We also compare JEV, a decision model used as an automated judge, with human grades. The findings show that model rankings depend on how answer similarity is measured, while a simple word-matching method retrieves source passages more effectively than the embedding methods tested. At the evaluated cutoff, JEV agrees with human grades less often than a bas
    
[^280]: 基于训练后两层ReLU网络的回归中的良性过拟合

    Benign Overfitting for Regression with Trained Two-Layer ReLU Networks

    [https://arxiv.org/abs/2410.06191](https://arxiv.org/abs/2410.06191)

    本文首次证明了通过梯度流训练的有限宽度两层ReLU网络在最小二乘回归中既能实现泛化又会过拟合数据，从而在任意有界回归函数下首次建立了良性过拟合结果。

    

    我们研究了使用带ReLU激活函数的两层全连接神经网络、通过梯度流训练的最小二乘回归问题。我们的第一个结果是一个泛化结果，除了要求底层回归函数和噪声有界之外，不需要对它们做任何假设。我们在神经切向核机制下进行分析，通过将超额风险分解为估计误差和近似误差来建立泛化结果，并将梯度流视为一种隐式正则化器。这种在神经网络背景下的分解为梯度下降提供了一个新颖的视角，帮助我们避开一致收敛的陷阱。在本工作中，我们还证明了在相同设置下，训练后的网络会对数据产生过拟合。综合起来，这些结果首次建立了有限宽度ReLU网络在任意回归函数下的良性过拟合结论。

    arXiv:2410.06191v2 Announce Type: replace-cross  Abstract: We study the least-square regression problem with a two-layer fully-connected neural network, with ReLU activation function, trained by gradient flow. Our first result is a generalization result, that requires no assumptions on the underlying regression function or the noise other than that they are bounded. We operate in the neural tangent kernel regime, and our generalization result is developed via a decomposition of the excess risk into estimation and approximation errors, viewing gradient flow as an implicit regularizer. This decomposition in the context of neural networks is a novel perspective of gradient descent, and helps us avoid uniform convergence traps. In this work, we also establish that under the same setting, the trained network overfits to the data. Together, these results, establishes the first result on benign overfitting for finite-width ReLU networks for arbitrary regression functions.
    
[^281]: 标签噪声下的鲁棒长尾学习

    Robust Long-Tailed Learning under Label Noise

    [https://arxiv.org/abs/2108.11569](https://arxiv.org/abs/2108.11569)

    该论文发现传统的小损失噪声标签检测准则因深度网络将尾部类别误判为头部类别的固有偏差而在长尾分布下失效，并提出利用所学表示鲁棒性的小距离准则来准确识别标签正确的样本，同时用标签分布替代离散伪标签以改进尾部类别的训练。

    

    长尾学习旨在提升欠代表性尾部类别的泛化性能。然而，以往的方法在很大程度上忽视了训练数据中噪声标签普遍存在这一问题。本文研究了长尾学习中噪声标签带来的挑战。我们发现了一个关键问题：常用的基于小损失的噪声标签检测准则在长尾类别分布下无法有效发挥作用。这种失效源于深度神经网络的固有偏差——网络倾向于将尾部类别的样本误分类为头部类别，从而导致不可靠的损失计算。为缓解这一问题，我们提出了一种新颖的基于小距离的准则，该准则利用了所学表示的鲁棒性，能够更准确地识别头部和尾部类别中标签正确的样本。此外，为了改进尾部类别的训练，我们用标签分布替代离散的伪标签来……

    arXiv:2108.11569v2 Announce Type: replace  Abstract: Long-tailed learning aims to enhance the generalization performance of underrepresented tail classes. However, previous methods have largely overlooked the prevalence of noisy labels in training data. In this paper, we address the challenge of noisy labels in long-tailed learning. We identify a critical issue: the commonly used small-loss noisy label detection criterion fails to perform effectively in long-tailed class distributions. This failure arises from the inherent bias of deep neural networks, which tend to misclassify tail class examples as head classes, leading to unreliable loss calculations. To mitigate this, we propose a novel small-distance criterion that leverages the robustness of learned representations, enabling more accurate identification of correctly-labeled examples across both head and tail classes. Additionally, to improve training for tail classes, we replace discrete pseudo-labels with label distributions for
    

