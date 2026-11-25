# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OmniAct: One Policy, One Million Embodiments](https://arxiv.org/abs/2611.33158) | OmniAct通过以机器人在具体场景中的运动学图（包含关节位置、运动轴及与视觉观测的连接关系）为条件，并用流匹配策略预测各节点的相对关节运动，实现了单一策略即可控制具有不同自由度和运动学配置的多种机器人具身形态。 |
| [^2] | [BioLoom: Scaling Biomedical Vision-Language Models by Learning from the Structure of Scientific Articles](https://arxiv.org/abs/2611.33151) | BioLoom通过挖掘科学文章内部的结构关系（如子图与同级图注之间的关联、同类内容的跨文章负样本），在对比图文预训练中引入弱正样本监督与加权负样本，从而在不增加数据规模的情况下提升生物医学视觉-语言模型的表示能力。 |
| [^3] | [iWAM: Efficient Action Prediction from World Models without Explicit Future Prediction](https://arxiv.org/abs/2611.33141) | 该论文提出 iWAM，利用流匹配模型的可逆动力学将未来反演为潜噪声向量，仅以单次视频模型评估的中间特征来条件化动作策略，从而无需显式生成未来帧即可实现高效的动作预测。 |
| [^4] | [A Bias in Disguise: Tiny Gradient Edits Stop a Small Language Model's Thinking Loops and Complement a Frontier Model's Weight Edit](https://arxiv.org/abs/2611.33133) | 该论文发现通过梯度选择极少量关键权重微调即可让小模型走出无休止的思考循环，其本质是为无偏置项的模型隐式添加偏置，并与前沿模型的无梯度权重编辑形成互补。 |
| [^5] | [Exploratory Midtraining for Robot Policies](https://arxiv.org/abs/2611.33130) | 提出探索性中间训练方法并构建通过对抗性遥操作采集的 β-Data 数据集，使机器人策略获得连通多样低层行为的覆盖，从而支持下游适应与在线自主改进。 |
| [^6] | [Draft Calibration for Speculative Decoding: Accelerating LLM Serving by Matching Draft and Target Distributions](https://arxiv.org/abs/2611.33111) | 该论文提出推测解码中的草稿校准方法，通过事后校准解决开源草稿模型相对目标模型普遍置信度不足的分布失配问题，缩小草稿分布与目标分布之间的差距，从而提高草稿 token 接受率并加速大语言模型推理服务。 |
| [^7] | [Reinforcement Learning Can Be Recovered From Its Own Rollouts](https://arxiv.org/abs/2611.33102) | 该论文发现，仅通过对存档的RL采样轨迹按其生成顺序进行最大似然训练、且无需区分响应的对错，就能从同一基础模型完全恢复出在线强化学习的性能，表明RL所获得的知识实质上蕴含在采样数据本身之中。 |
| [^8] | [GREMLIN: Generating Random Environments for Meta-Learning with Interacting Networks](https://arxiv.org/abs/2611.33101) | 该研究提出 GREMLIN 方法，用随机参数化的神经网络环境替代人工设计游戏为元学习提供训练经验，实验证明生成的环境经验虽不及 DQN，但能在所有测试游戏上有效改进学习规则。 |
| [^9] | [The Radar-based Activities of Daily Living (R-ADL) Dataset](https://arxiv.org/abs/2611.33088) | 该论文推出了R-ADL数据集，提供超过八小时、涵盖八种活动和五个居家位置的标注雷达数据，弥补了现有雷达活动识别数据集在规模、房间配置和活动层次上的不足。 |
| [^10] | [Shifter Quantization: Compact Representations for Native Integer Computation](https://arxiv.org/abs/2611.33080) | 提出移位器量化（SQ）方法，通过追加和移位共享比特重构原生宽度整数操作数，使GPU整数矩阵运算能覆盖完整点积，在2比特精度下实现llama.cpp预填充吞吐量2.71-3.77倍的提升并降低KL散度。 |
| [^11] | [The Reliability Function of Error-Correcting Output Codes with Noisy Binary Classifiers](https://arxiv.org/abs/2611.33068) | 该论文将带高斯噪声二分类器的 ECOC 解码建模为高斯信道检验，推导出超额错误率的上下界，证明较大的码字汉明距离是降低超额错误率的充分必要条件，且该错误率下界只与最大两个后验概率和噪声水平有关而与类别数无关。 |
| [^12] | [Auditing the Judge: Human-Derived Bias Discovery, Quantification, and Mitigation in LLM Judges](https://arxiv.org/abs/2611.33065) | 该论文提出HUB-J集成框架，利用人类与LLM判断不一致的案例自动发现可解释的偏见因子，并通过受控输入修改量化偏见强度，实现LLM评判系统中偏见的发现、量化与缓解。 |
| [^13] | [U-Statistics-Powered LLM Reasoning](https://arxiv.org/abs/2611.33064) | 该论文利用 GRPO 策略梯度的 U 统计量结构，将组大小选择转化为方差比估计这一经典统计问题，并提出考虑分母估计不确定性的悲观比率估计量，从而以系统且稳定的方式确定这一关键超参数。 |
| [^14] | [Understanding Clipping In Zeroth-Order Optimization](https://arxiv.org/abs/2611.33062) | 本文系统研究了逐样本裁剪为何能提升零阶优化性能这一现象，证明了最优学习率与裁剪阈值成反比（即 ηc 为常数），并且裁剪会使零阶优化收敛到不同的解。 |
| [^15] | [Local Indices Are Not Enough: Synchrony and Global Information in Bandit Algorithms](https://arxiv.org/abs/2611.33056) | 本文证明了仅依赖臂自身历史和局部随机性的异步指数型多臂老虎机算法无法在所有实例上达到 O(log T) 的最优遗憾值，其遗憾值只能处于亚多项式但超多对数的区间，从而揭示了全局时间信息与同步指数更新对于最优性能的必要性。 |
| [^16] | [Latent Merger: Latent-Space Composition of Hypernetwork-Generated LoRAs](https://arxiv.org/abs/2611.33048) | 该论文发现超网络生成的LoRA在几何特性上与直接微调的LoRA截然不同（谱更集中、跨文档对齐更强），据此提出在潜空间中直接组合中间潜变量再解码的LoRA合并方法，以解决流式场景下逐文档存储LoRA不可持续的问题。 |
| [^17] | [When Is Product Decoding of ECOC MAP-Consistent? Pair Structure and the n-vs-Rest Codes](https://arxiv.org/abs/2611.33044) | 本文给出了ECOC乘积解码器与贝叶斯最优（MAP）类别保持一致的充要条件——即各类别对之间取值不同的列须满足特定的配对匹配结构，并据此刻画了满足该条件的n-vs-Rest编码。 |
| [^18] | [Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems](https://arxiv.org/abs/2611.33036) | 该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。 |
| [^19] | [Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?](https://arxiv.org/abs/2611.33031) | 该研究证明前沿大语言模型无需任何工具、数据或推理能力，仅凭权重空间编程即可将4比特量化CNN的准确率从60.4%恢复至77.95%，接近原始浮点模型的81.4%水平。 |
| [^20] | [Score-Based Diffusion for Hypergraph Signals in the t-Spectral Domain](https://arxiv.org/abs/2611.33026) | 该论文提出了首个在t-积超图信号处理谱域中构建的、面向固定超图信号的基于分数生成模型，通过逐模态频率偏移修复了张量热半群的不适定问题，实现了稳定的大规模采样与缺失节点恢复。 |
| [^21] | [Best Arm Identification with Unknown Post-Action Context Distribution](https://arxiv.org/abs/2611.33024) | 该论文研究了上下文分布未知的动作后上下文最优臂识别问题，推导了依赖实例的样本复杂度下界，并提出了一种基于新型集中不等式、能够渐进匹配该下界的 Track-and-Stop 算法。 |
| [^22] | [PAC-style Bounds on the Conditional Validity of Conformal Predictors](https://arxiv.org/abs/2611.33023) | 本文在不假设数据分布的前提下，基于马尔可夫和坎泰利不等式为任意共形预测器的条件覆盖率推导出PAC型概率下界，并提出将条件覆盖率方差作为条件有效性的一种可扩展且有理论支撑的经验度量。 |
| [^23] | [Train What You Run: Probabilistic Convolutions for Learning Token Compaction in State Space Models](https://arxiv.org/abs/2611.33009) | 本文提出 ProbConv——一种关于令牌保留概率可微的概率卷积算子，可计算独立保留决策下的精确期望卷积，使状态空间模型（如 Mamba）的训练与推理时的令牌压缩保持一致，且压缩越强其优势越明显。 |
| [^24] | [LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training](https://arxiv.org/abs/2611.33004) | LEGOSpec提出了一个可组合的推测性草稿设计与训练框架，将草稿模型分解为上下文、骨干、提议和目标四个组件，并通过系统化的对照实验揭示了不同工作负载下各组件的作用、相互作用及最优组合规律。 |
| [^25] | [On Potentiality and Actuality Of Foundation Models](https://arxiv.org/abs/2611.32991) | 基础模型本身已蕴含完成特定任务所需的能力，适配并非添加新能力，而只需通过恒定的、与输入无关的向量移位将模型“实现”即可。 |
| [^26] | [AVQ-Attention-2: Adaptive Vector-Quantized Attention with Exact Local Blocks and Linear-Cost Training](https://arxiv.org/abs/2611.32989) | AVQ-Attention-2 通过将精确局部注意力与关注聚类键均值的自适应向量量化注意力相结合，实现了线性成本的训练和推理，无需微调即可直接插入预训练 Transformer，在几乎无损的情况下运行速度快于 FlashAttention 且质量可与精确稠密注意力相媲美。 |
| [^27] | [File Size as a Potential Dataset Bias in Audio-Visual Deepfake Data](https://arxiv.org/abs/2611.32979) | 该研究揭示了两个主流视听深度伪造数据集中存在文件大小偏差——仅凭文件大小和视频时长等文件级属性即可获得很高的检测AUC，说明深度伪造检测基准可能在类别标签与文件属性之间引入了非预期的关联。 |
| [^28] | [Eliciting the Recall Capacity of Linear Attention](https://arxiv.org/abs/2611.32969) | 该研究揭示了线性注意力模型的召回容量随状态规模缩放的规律，并提出了上下文句子复述和上下文敏感损失重加权两种训练干预方法，有效弥补了线性模型在关联召回能力上与softmax注意力的差距。 |
| [^29] | [Polynomial-Time Learning of Random Automata from One Label per Word](https://arxiv.org/abs/2611.32959) | 本文提出一个多项式时间算法，仅凭每个观测单词上的单个接受/拒绝标签，即可在绝大多数随机转移表上高概率地学习二进制确定性有限自动机，并将预测错误率控制在 ε 以内，样本量与运行时间均为参数的多项式。 |
| [^30] | [Online V-Learning with Adaptive Longitudinal Data](https://arxiv.org/abs/2611.32947) | 本文针对移动健康中策略在线批量更新所导致的自适应收集、患者轨迹相依的数据，在可变确定性批次大小设定下建立了V学习的统计理论并证明了一致性。 |
| [^31] | [PHM-ensemble: Scalable and Stable Probabilistic Multiresolution Clustering via Ensemble Density Estimation](https://arxiv.org/abs/2611.32945) | 该论文提出PHM-ensemble框架，通过集成密度估计汇集多个混合模型并对成分进行层次化合并，实现了同时兼顾保真度、可解释性、稳定性和可扩展性的概率多分辨率聚类。 |
| [^32] | [Quantum kernels on breast data: concentration, surrogates, and two ways an advantage gets manufactured](https://arxiv.org/abs/2611.32942) | 本文通过对乳腺癌数据上的量子核优势宣称进行系统性基准检验（涵盖样本效率扫描、闭式去量化替代模型对比和选择膨胀测量），揭示所谓的量子优势可能是由于不完整的对照比较、选择偏差乃至数据泄露而被“制造”出来的假象。 |
| [^33] | [Efficient Video Retrieval Augmented Generation via Token Allocation](https://arxiv.org/abs/2611.32938) | Lembas是一个即插即用的视频RAG框架，将帧与字幕的令牌分配建模为收益递减的优化问题，用具有可证明收敛保证的迭代注水算法求解，并能在多轮对话中零开销地复用令牌。 |
| [^34] | [Theory-Informed Quantum Error Mitigation for Shor Order Finding with Fourier Neural Operators](https://arxiv.org/abs/2611.32923) | 该论文提出了一种理论引导的量子误差缓解方法，通过建模电路误差对输出分布峰值的偏移、展宽和减弱效应，并利用带可学习增益的傅里叶神经算子，从噪声测量分布中有效恢复Shor求阶算法的正确阶估计。 |
| [^35] | [Label leakage through file headers: geometry and batch confounds in public mammography datasets, and a pre-training acceptance gate](https://arxiv.org/abs/2611.32899) | 该论文提出一套仅利用图像几何、纹理或标识符等非诊断信息的预训练准入控制检查，以AUC 0.60为停止阈值审计六个公开乳腺X线摄影数据集，发现多个数据集存在严重标签泄露（如KAU-BCMD仅凭几何信息即能以AUC 1.0预测BI-RADS分级），并据此构建了模型预训练前的数据集验收门槛。 |
| [^36] | [Shared information governs multi-view planted matching](https://arxiv.org/abs/2611.32892) | 该论文证明对同一 n 个对象的 K 个打乱视图能否全部匹配由对象观测间的共享信息精确决定——共享信息超过 (1+ε)log n 时可匹配至 o(n) 误差、低于 (1−ε)log n 时不可能，且整体可匹配性由视图最弱二分划分的互信息阈值 2log n 控制，弱共同信号下两个阈值分别为 4/K 与 4/(K−1)。 |
| [^37] | [MXFP4 for LLM Inference: Complementary Post-Training Quantization Techniques and Their Interactions](https://arxiv.org/abs/2611.32873) | 该研究发现有效的微缩放格式量化方案依赖于具体格式：MXFP4和NVFP4虽都支持精确的4位推理，但MXFP4显著受益于旋转和纠错PTQ技术，而这些技术未必适用于NVFP4。 |
| [^38] | [Weight Scale Rebalancing: Transferable Training Recipes for Transformers](https://arxiv.org/abs/2611.32866) | 提出UDVR重参数化方法，在Transformer成对权重矩阵间插入辅助对角矩阵并周期性再平衡尺度，得到一条仅需矩阵维度的免调优超参数规则，可跨模型规模、架构、优化器和数据分布直接迁移。 |
| [^39] | [Tabra Blends: Improving the Accuracy--Latency Trade-off with Complementary Tabular Models](https://arxiv.org/abs/2611.32859) | 将四个估计器的Kumo与一个LimiX流水线混合成紧凑集成，在816个TabArena划分上比最佳已发布模型高出45.4 Elo，同时归一化拟合+预测时间比Kumo-16低15%、比完整LimiX低98%，且两个模型间的预测多样性是误差降低的关键。 |
| [^40] | [What Does a Skill Vocabulary Cost? A Closed-Loop Decomposition over Demonstrated Motions](https://arxiv.org/abs/2611.32858) | 提出运动原型量化（MPQ），将演示动作库置于已训练策略之后且无需训练参数，使包装后的策略与原策略统计等价，从而建立参考基线，可逐一归因闭环失败于选择器的具体组件。 |
| [^41] | [Path-probability gradients for optimizing force fields against dynamical observables](https://arxiv.org/abs/2611.32851) | 提出PathGrad方法，通过对朗之万路径概率而非混沌分子轨迹求导，获得方差稳定的梯度估计器，从而能够直接利用实验动力学观测量（如扩散系数）对机器学习力场进行微调。 |
| [^42] | [PtychoFM: A Foundation Model for Ptychographic Imaging](https://arxiv.org/abs/2611.32830) | PtychoFM是一种物理感知的叠层衍射成像基础模型，通过可微正向模型在大规模数据上预训练并针对具体实验微调，无需物体空间标签即可直接从单个衍射图样重建图像，并能跨样品与扫描方式迁移，在同步辐射装置上实现实时推理。 |
| [^43] | [Data-driven Reduced-order Modeling of a Multistable Viscoelastic Flow](https://arxiv.org/abs/2611.32818) | 该论文提出了一种数据驱动框架，仅利用稳定流态的稀疏稳态测量数据，并通过平衡态约束、稳定性约束和物理反射对称性，学习出能够刻画并排双微圆柱粘弹性流动多稳态分岔结构的二维参数依赖正规形模型。 |
| [^44] | [E(n)-Equivariant Spherical Decision Surfaces](https://arxiv.org/abs/2611.32815) | 该论文通过将O(n)等变超球面神经元扩展以包含平移，构造性地推导出无需输入中心化或显式成对差值即严格E(n)等变的球面决策曲面，并给出其显式闭式矩阵表示。 |
| [^45] | [Breaking the Joint Action Space Dependence in Distributionally Robust Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.32812) | 该论文设计了一种分布鲁棒多智能体强化学习算法，其样本复杂度仅依赖于最大个体动作集大小而非随玩家数指数增长的联合动作数，从而打破了多智能体维数灾难。 |
| [^46] | [EPRO: Robust Policy Optimization Through an Imperfect Execution-Policy Predictor](https://arxiv.org/abs/2611.32810) | EPRO 针对大语言模型强化学习中训练与 rollout 执行策略不一致且当前执行策略不可直接观测的问题，利用训练策略变化构建可微分的执行策略预测器，并在信任度约束下对实际执行策略进行鲁棒优化。 |
| [^47] | [Sample-Efficient In-Context Adaptation via Sleep](https://arxiv.org/abs/2611.32802) | 该研究提出在Transformer语言模型预训练中引入“睡眠”机制（通过多次循环前向传递精炼上下文表示），显著提升了模型在一般性任务上少样本上下文学习的样本效率。 |
| [^48] | [Scaling Laws of Ensemble Uncertainty in Language Models](https://arxiv.org/abs/2611.32801) | 本文提出三条幂律缩放定律，证明集成随模型规模增大会捕获越来越多的认知不确定性，作为扩展路径相比单个更大的模型更具竞争力。 |
| [^49] | [Simulator-as-Proxy World Model for Robot Manipulation](https://arxiv.org/abs/2611.32786) | 本文提出SimPWM，用一个无需训练参数的规则模拟器作为显式代理世界模型，仅凭第一帧即可表示机器人与物体，并首次系统验证了视频世界模型是否忠实跟随模拟代理——结果显示机械臂能紧密跟随代理，而物体的行为则可能出现偏差。 |
| [^50] | [Measuring Layer-wise Intrinsic Dimensionality of FFNs in LLMs via PCA](https://arxiv.org/abs/2611.32782) | 该研究用PCA测量58个开源LLM各层FFN的内在维度，发现中间层的内在维度显著高于边缘层的“中间重”分布，表明逐层均匀分配参数的传统设计并非最优，为非均匀的参数预算分配提供了实证依据。 |
| [^51] | [NABLA: A Noise-Adaptive Blockwise Left-Right Preconditioning Algorithm for Stochastic Optimization](https://arxiv.org/abs/2611.32781) | 本文提出NABLA算法，通过闭环反馈规则自适应调整二阶矩平均系数，实现无需噪声先验知识的分块左右预条件随机优化。 |
| [^52] | [SGD with Large Learning Rates: General Analysis & Improved Rates](https://arxiv.org/abs/2611.32762) | 本文证明了大学习率的SGD在随机梯度噪声存在下仍能加速收敛，对线性预测器的交叉熵损失和两层神经网络建立了加速的O(T^-2)收敛速率，并在稳定区域内给出了更紧致的收敛保证。 |
| [^53] | [ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models](https://arxiv.org/abs/2611.32755) | ReQOR提出了一种无需训练的防御方法，通过在良性校准数据上识别模块级量化算子差异的主导方向，并以闭式形式恢复全精度计算，有效修复大语言模型中的量化条件后门。 |
| [^54] | [NCPU: Towards Cell-Inspired Spatial Computing](https://arxiv.org/abs/2611.32750) | 该论文提出用神经元胞自动机（NCA）通过梯度下降学习局部更新规则来实现精确的数字逻辑计算，将比特编码为二维网格上的细胞簇，成功训练出逻辑门、4位和8位加法器及初步的8位ALU，全部达到100%位级正确率，且仅用3位操作数训练的加法器可泛化到8位。 |
| [^55] | [Quantum Community Detection for Data Analysis](https://arxiv.org/abs/2611.32738) | 本文提出了量子谱社区检测算法projeQt，利用量子奇异值变换和基于量子态重叠的k均值聚类，无需递归即可一次性提取多个社区，可应用于特征选择和聚类等数据分析任务并有望实现量子加速。 |
| [^56] | [Seeing to Drive on a Single Board: Attention-Free Depth for Onboard Robot Navigation](https://arxiv.org/abs/2611.32718) | 该论文提出用知识蒸馏而非量化，将无注意力的ResNet-34 U-Net学生模型部署到树莓派5的Hailo-8L加速器上，以27.1 FPS实现芯片上最精确的单目深度估计，使小型机器人能依靠单块板卡的视觉进行导航。 |
| [^57] | [FourStat: A Dual AutoEncoder Leveraging Frequency and Stationarity for Time Series Anomaly Detection](https://arxiv.org/abs/2611.32714) | 提出FourStat双自编码器框架，结合频域分析的傅里叶自编码器与强化统计平稳性的平稳自编码器来检测单变量时间序列异常，并引入Perfect Point新指标以消除阈值选择的主观性。 |
| [^58] | [Contrastive Proxy Forests: Completeness-Preserving Proxy Learning from Multimodal Records](https://arxiv.org/abs/2611.32708) | 该论文提出对比代理森林（CPF），利用处理侧与结果侧模态之间由混杂因子构成的唯一联系，从高维多模态临床记录中学习共享表示以捕获混杂因子、过滤模态特有风格，从而保持代理变量的完备性。 |
| [^59] | [A Hybrid Analytical-Neural Residual Surrogate for Fast Radio Propagation Prediction over Irregular Terrain](https://arxiv.org/abs/2611.32698) | 本文提出一种解析-神经混合残差代理模型，用确定性解析基线（涵盖自由空间扩散、天线方向性、刀口衍射和地形镜面反射）分离可解析的传播趋势，再由紧凑的由粗到细神经网络预测残差场，从而实现不规则地形上快速准确的无线电传播预测。 |
| [^60] | [Load Balancing Neurons: Firing-Rate Control Sustains Plasticity in Continual Learning](https://arxiv.org/abs/2611.32691) | 本文提出一种通过调整每个神经元的阈值来控制ReLU激活率的轻量级负载均衡机制，主动防止神经元死亡或过度激活，从而在持续学习中维持网络的可塑性。 |
| [^61] | [OpenWoud: Vendor-Unlocked Scalable Adaptive Random Forest Accelerator](https://arxiv.org/abs/2611.32662) | OpenWoud是一个无厂商锁定的FPGA自适应随机森林加速框架，通过定点运算库替代浮点运算并引入完全透明的手动调度数据流，实现了细粒度的时间/面积权衡与跨平台可移植性。 |
| [^62] | [LATEC: Learned Axial TEnsor Compression for High-Dimensional Scientific Data](https://arxiv.org/abs/2611.32651) | 本文提出LATEC，一种基于GNN架构、维度无关的学习式误差界限压缩编解码器，可高效压缩高维科学数据且无需针对不同维度进行大量适配或重新训练。 |
| [^63] | [TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for Multi-Agent Coordination](https://arxiv.org/abs/2611.32645) | TeamFlowRL提出了一种两阶段框架，通过在去中心化流策略的完整去噪轨迹上构建团队策略优化目标（避免难以计算的动作似然），实现多智能体流策略的离线预训练与在线微调，在51个配对配置中平均提升47.3%。 |
| [^64] | [HiCal: Offline Calibration for Hierarchical Scale Encoding in Accurate HiF4 Quantization](https://arxiv.org/abs/2611.32624) | 提出HiCal离线校准方法，通过为E6M2尺度引入校准舍入偏置、并基于局部组幅值经验分布优化尺度扩展阈值，从三个尺度级别全面提升HiF4分层量化精度。 |
| [^65] | [ST-GNN-AAAS: Spatio-Temporal Graph Neural Networks for Anytime Automatic Algorithm Selection in the Traveling Salesman Problem](https://arxiv.org/abs/2611.32602) | 提出了一种时空图神经网络框架ST-GNN-AAAS，通过融合当前最优路径的时变节点特征与求解器短期性能历史来预测未来的求解器选择，实现了旅行商问题中随不同时间预算变化的自动算法选择。 |
| [^66] | [Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding](https://arxiv.org/abs/2611.32600) | 提出了门控概念适配器（GCA）这一分层概念瓶颈框架，通过在预训练Transformer的选定层附加门控低维概念读出适配器，将隐藏状态转化为可检查、可干预的概念中介接口，从而实现可审计的文本理解。 |
| [^67] | [On the Effectiveness of Resampling for Supervised Graph Anomaly Detection](https://arxiv.org/abs/2611.32598) | 该研究系统评估了监督式图异常检测中的重采样策略，发现对自然发生的异常没有任何过采样方法能带来统计显著的收益，高斯噪声过采样仅对注入型属性交换异常有效，且标注哪些节点比选择何种采样策略更为关键。 |
| [^68] | [Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard](https://arxiv.org/abs/2611.32595) | 本文证明了具有局部求和聚合和ReLU激活的消息传递神经网络（MPNN）的度数受限可满足性判定问题是NEXPTIME难的，且即使度数上界固定为3，该难度结果仍然成立。 |
| [^69] | [Understanding the Importance of Massive Activations through Adversarial Robustness](https://arxiv.org/abs/2611.32585) | 本文将大规模激活与延迟鲁棒性中的区域迁移现象联系起来，证明大规模激活通过将表示推离网络非线性边界来降低输入输出映射的复杂性，并首次发现该现象并非Transformer独有，也普遍存在于其他常见架构中。 |
| [^70] | [Boiling Down Uncertainty: Deep Generative Scenario Search with Decision-Guided Evolution Strategies for Two-Stage Stochastic Optimization](https://arxiv.org/abs/2611.32573) | 本文通过补偿对偶性与伴随求解推导出闭式梯度，为深度生成式情景搜索补齐了解析理论基础，并将决策聚焦方向作为引导进化策略的采样子空间，保证期望更新为平滑目标的下降方向，从而改进两阶段随机优化的情景生成。 |
| [^71] | [Breaking Routing And Inter-device Dependencies in Mixture-of-Experts](https://arxiv.org/abs/2611.32572) | 提出BRAID-MoE架构，通过阶梯式残差连接与早期路由打破MoE模型中的顺序依赖，实现通信与注意力计算的深度重叠，缓解专家并行推理中令牌分发与聚合的通信瓶颈。 |
| [^72] | [Representation and Model Contributions on Small Molecular Benchmarks: Controlled Comparisons and External Evaluation](https://arxiv.org/abs/2611.32568) | 该研究通过二乘二因子设计与配对支架自助法，将分子表示与模型架构的贡献分离，发现在溶解度和亲脂性等任务上，分子表示（如RDKit描述符）带来的性能提升甚至超过更换模型本身的效果。 |
| [^73] | [Contact Geometry for Generative Models: An Unbalanced Optimal Transport Formulation](https://arxiv.org/abs/2611.32547) | 该论文提出基于接触几何的非平衡最优传输新方法 CuOT，通过将密度传输方向与质量增长解耦，消除了 WFR 几何中偏向高密度区域的结构性偏差，从而避免模式坍塌并实现稳定、可扩展的生成模型求解。 |
| [^74] | [Beyond Robust Algorithmic Recourse: Sharing the Burden of Validity through Model Multiplicity](https://arxiv.org/abs/2611.32543) | 该论文揭示了鲁棒算法追索虽能保证建议有效性但会因偏向数据高密度区域而造成系统性不公平，并提出利用模型多重性下的策略性模型选择来分担有效性负担，从而在放宽个体鲁棒性要求的同时维持追索的有效性。 |
| [^75] | [Advanced Synthetic Image Detection Using Hybrid Feature Extraction](https://arxiv.org/abs/2611.32538) | 本文提出一种混合特征提取的合成图像检测方法，结合CBAM、ResNet50、DCT评分和SRM模块，通过捕捉生成模型难以复制的细粒度纹理与复杂物体特征来有效区分真实图像与AI生成图像。 |
| [^76] | [Branch and Consolidate: A Unified Recipe for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2611.32515) | 该论文提出了一套统一的多模态多教师在线策略蒸馏配方，能够将大幅偏离共享检查点的专家模型重新整合回单一学生模型，使其在各个领域接近专家模型性能并超越感知专家。 |
| [^77] | [Analysis of Federated Graph Neural Networks for Predicting Nuclear Core Properties](https://arxiv.org/abs/2611.32509) | 本文证明在联邦学习框架下，图神经网络因其无需填充且不假设平移不变性而更契合核反应堆圆柱形堆芯的几何结构，从而在堆芯优化任务中优于CNN基线。 |
| [^78] | [InducedTFMs: A Family of Sub-Quadratic Tabular Foundation Models Using Induced Attention](https://arxiv.org/abs/2611.32504) | 针对表格基础模型注意力机制随样本和特征数量呈二次方增长的计算瓶颈，本文提出 InducedTFM，用 Set Transformer 的诱导注意力替换二维注意力的两个阶段，在保留单元格级表示的同时将计算成本降为线性，并在与全注意力模型同等训练预算下取得相当的准确率。 |
| [^79] | [Assessing Robustness to Catastrophic Forgetting in Continual Deep Reinforcement Learning for Cyber-Physical Systems](https://arxiv.org/abs/2611.32496) | 该论文在非平稳自主泊车环境中系统评估了三类强化学习算法（无模型同策略的PPO、无模型异策略的SAC和基于模型的Drama）抵抗灾难性遗忘的鲁棒性，以确定哪类算法更适合信息物理系统中的持续学习。 |
| [^80] | [When Knowledge-Tracing Rankings Depend on Available History: A Source-Correction and Matched-Information Audit on ASSISTments 2009 and 2015](https://arxiv.org/abs/2611.32489) | 该论文通过对 ASSISTments 2009 与 2015 数据集进行数据源修正与信息匹配审计，揭示了知识追踪模型的性能排名依赖于学习者可用历史的长度、来源与表示方式，为公平的模型基准比较提供了重要的方法学依据。 |
| [^81] | [The Teacher Is Already Inside: Self-Distilling Looped Language Models Beyond Their Last Loop](https://arxiv.org/abs/2611.32452) | 该论文提出ELPSD方法，将循环语言模型从首轮循环外推至末轮得到的下一个token分布作为自蒸馏目标，在模型自身rollout上训练其最后一轮循环，突破了传统后训练方法无法改进循环模型最后一轮的瓶颈。 |
| [^82] | [See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight](https://arxiv.org/abs/2611.32439) | 该论文提出HEME方法，在训练时将匹配的候选目标作为“后见之明”上下文来学习基于查询的表示，使模型在推理时无需额外推理计算即可生成高质量的通用多模态嵌入。 |
| [^83] | [A Framework for Graph Node Symmetric Quantification under Dataset Shift](https://arxiv.org/abs/2611.32403) | 该论文提出了一种新的图节点量化框架，摆脱了对节点级监督和节点级分类器预测的依赖，仅需按类别分布标注的节点样本进行训练，即可在数据集偏移下直接估计样本中各类别的占比。 |
| [^84] | [Optimal Transport Meets t-SNE: Integrated Visualization without Batch Effects in OT-tSNE](https://arxiv.org/abs/2611.32402) | OT-tSNE 用熵正则化最优传输的全局耦合亲和度取代跨批次最近邻匹配，在单细胞数据整合可视化中有效消除批次效应并保持稳健性。 |
| [^85] | [Not All Pairs Align: Caption-Guided Audio-Visual Contrastive Learning](https://arxiv.org/abs/2611.32399) | 该论文提出了一种模型无关的对比学习目标CGAAL，利用模态特定字幕估计音视频间的语义一致性，并据此为正样本对加权，从而解决了画外音等因素导致的音视频内容不匹配问题。 |
| [^86] | [Fractal Neural Synthesis: Constant-Storage Procedural Weight Derivation via Complex Quadratic Boundary Dynamics](https://arxiv.org/abs/2611.32393) | 该论文提出分形神经合成（FNS）架构，用仅24字节、锚定于复二次分岔边界的分形坐标种子，通过过程式推导生成神经网络权重并证明了熵界，将存储占用从随参数量线性增长降低到常数级，适用于资源受限的边缘微控制器。 |
| [^87] | [Latent Class Mixed Model Variational Autoencoders for Scalable Phenotyping of Health Trajectories](https://arxiv.org/abs/2611.32391) | 提出LCMM-VAE，一种以类别特定线性混合模型的混合作为潜变量先验的深度生成模型，通过闭式变分EM算法，实现了对高维、不规则采样纵向健康轨迹的可扩展且临床可解释的聚类表型分析。 |
| [^88] | [Evolve to Learn: Gradient-aligned experience evolution for model self-improvement](https://arxiv.org/abs/2611.32390) | 该论文揭示了一个关键问题：能提升测试时性能的经验并不一定值得内化到模型中，并提出以梯度对齐作为更可靠的经验筛选标准，据此构建了E2L自进化框架来实现模型的自我改进。 |
| [^89] | [Can AI Design a Better nnU-Net? A Hypothesis-Driven Architecture Discovery Loop from Failures to Design Rules](https://arxiv.org/abs/2611.32384) | 该论文提出一种假设驱动的架构发现循环，让LLM智能体从失败案例中诊断错误模式、提出可证伪假设并验证机制，最终提炼出两条设计规则，构建出比nnU-Net计算量更低、分割精度更高的3D医学图像分割网络LeanNeXt。 |
| [^90] | [The Activity Geometry of Discrete Generative Sampling](https://arxiv.org/abs/2611.32380) | 该论文提出“活动几何”框架，将离散采样中逐步耦合的活动度分解为边际传输的最小代价与耦合超额项，并据此设计出无需重训、无需额外模型评估即可在最小活动度下实现指定边际分布并支持修正已生成token的MAFTC采样耦合方法。 |
| [^91] | [Metamorphic Testing of Learned Spatial Surrogates for Transit Accessibility](https://arxiv.org/abs/2611.32373) | 本文对用于近似公交可达性计算的学习型神经代理模型进行蜕变测试，发现平均预测误差不足以衡量模型可靠性——预测技能良好的模型仍可能无法通过基本连通性测试，而一种针对性干预可同时提升连通性测试通过率与预测准确性。 |
| [^92] | [From the ELBO to Maximum Likelihood: Convergence of Variational Autoencoders for Nonlinear Mixed-Effects Modeling](https://arxiv.org/abs/2611.32372) | 本文对用于非线性混合效应建模的VAE-NLME算法进行了收敛性分析，证明了其基于ELBO的版本无需精确训练编码器即可几乎必然收敛到ELBO的稳定点，且最终算法通过重要性加权充分统计量收敛到极大似然意义下的解。 |
| [^93] | [A Description-Derived Drug-Drug Interaction Benchmark Reveals Split-Dependent Molecular Representation Rankings](https://arxiv.org/abs/2611.32371) | 该论文从DrugBank描述文本中构建了一个药物-药物相互作用预测基准，并发现分子表示方法的性能排名依赖于训练-测试划分方式——分子指纹在化学重叠度高的随机划分上最优，而图衍生表示在骨架不相交的划分上更具优势。 |
| [^94] | [CriticJEPA: World Models Are Better Critics Than Optimizers](https://arxiv.org/abs/2611.32357) | 提出CriticJEPA，证明冻结的世界模型作为“评判者”比作为“优化器”更可靠，通过从无奖励轨迹学习目标条件的流候选方案，并在各层级用冻结模型推演进行前向选择，从而实现更稳健的视觉规划。 |
| [^95] | [Beyond Hard Tokens: Predictive Memory for Multimodal Protein Generation](https://arxiv.org/abs/2611.32347) | 该论文提出了PredMem——一种无需训练的自条件化递归机制，通过在迭代掩码解码的各步骤间传递未提交的软预测作为跨步记忆，解决了多模态蛋白质生成中收敛缓慢与重复性坍塌的问题。 |
| [^96] | [When Exact Token Importance Is Not Enough: Limits of Top-K Selection in LLM Attention](https://arxiv.org/abs/2611.32324) | 本文证明了即使采用精确的单点删除效应作为词元重要性评分，Top-K独立排名选择仍会次优，因为注意力归一化引入的位置间交互可使排名组合与最优组合之间的KL散度差距高达133.2倍且在理论上无界。 |
| [^97] | [SPRUNN: Structural Proximity for Pruning in Deep Neural Networks](https://arxiv.org/abs/2611.32317) | 本文提出基于结构接近度准则的滤波器剪枝方法SPRUNN，通过全面比较滤波器间的相似性识别并剪除不重要的滤波器，并与全连接层细粒度剪枝相结合，在不显著损失准确率的前提下压缩深度神经网络。 |
| [^98] | [ASTERISM: Do Multimodal LLMs Actually Look at All the Images?](https://arxiv.org/abs/2611.32316) | 该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。 |
| [^99] | [DAiSy: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining](https://arxiv.org/abs/2611.32294) | DAiSy通过一个与主模型同步更新的轻量级线性辅助模型，从冻结的文档嵌入预测基于梯度的数据效用分数，使在线数据选择中每个候选文档的评分仅需一次线性变换，从而消除了大语言模型预训练中数据选择高昂的计算开销。 |
| [^100] | [$\epsilon$-Seg++: Ensembling Across Hierarchy and Training Time for Label-Efficient EM Segmentation](https://arxiv.org/abs/2611.32290) | ε-Seg++ 通过在层次化变分自编码器的每一层级引入语义专家并用无参数专家乘积进行集成、支持无泄漏的 3D 子体积输入、以及跨训练时间对最优检查点进行集成，显著提升了稀疏标注条件下电子显微镜分割的效果。 |
| [^101] | [CoMNO: Commutator Memory Neural Operators for Long-Horizon PDE Prediction](https://arxiv.org/abs/2611.32274) | 提出换位子记忆神经算子（CoMNO），通过交换“先下采样后滚动预测”与“先滚动预测后下采样”两种操作顺序所产生的性能差距来提取细尺度动力学细节，从而增强神经算子在长时程偏微分方程预测中捕捉细尺度特征的能力。 |
| [^102] | [Pando: Noise-Coupled Parallel Draft Trees for Efficient Edge Speculative Decoding](https://arxiv.org/abs/2611.32269) | Pando 在并行草稿生成器之上构建动态草稿树，并通过噪声耦合剪枝在草稿生成前依据目标模型的采样噪声对候选分支排序筛选，从而突破固定块大小对草稿链长度的限制，充分利用验证阶段的计算预算，实现高效的边缘端推测解码。 |
| [^103] | [Seldon TFM family: Seldon-Mini, Seldon and Seldon+](https://arxiv.org/abs/2611.32257) | Seldon 表格基础模型家族仅用合成数据训练，在多项表格基准测试中名列前茅，其开源模型 Seldon-Mini 和 Seldon 同时支持分类与回归，并在准确性与推理速度的权衡上处于帕累托前沿。 |
| [^104] | [MYSTIQUE: Few-Shot Facial Identity Transfer with Performance Preservation](https://arxiv.org/abs/2611.32232) | 提出基于扩散模型的少样本面部身份迁移方法MYSTIQUE，通过将身份与表情、光照、场景分流的输入解耦策略及多阶段训练，有效防止自重建训练带来的身份泄露，同时完整保留驱动图像中的表演表现。 |
| [^105] | [Relational Bottleneck Models](https://arxiv.org/abs/2611.32227) | 该论文提出了免训练、即插即用的关系瓶颈模型（RBMs），通过结构化推理依据生成显式建模视觉概念间的语义关系，将现成的大型视觉语言模型转化为兼具可解释性与高性能的内在可解释模型。 |
| [^106] | [Joint Classification and Generation on Tabular Data: an Empirical Study](https://arxiv.org/abs/2611.32225) | 本文系统比较了基于能量模型、概率电路、表格基础模型和表格扩散模型四类最先进方法构建的联合分类与生成模型，在29个数据集上评估了不同计算预算下它们与独立“分类器+生成器”组合的优劣。 |
| [^107] | [Beyond Autoregression: Diffusion Language Model is All Analog In-Memory Computing Need](https://arxiv.org/abs/2611.32223) | 该论文发现扩散语言模型在模拟内存计算的硬件噪声扰动下比自回归模型显著更鲁棒，其机制在于延迟词元提交与渐进式上下文解析避免了误差级联，且更大的去噪步数预算可作为延迟与鲁棒性之间的调节手段。 |
| [^108] | [Who gets the information? An $\alpha$-fair label-coverage framework for multi-label feature selection](https://arxiv.org/abs/2611.32200) | 该论文提出 FLIC 框架，将多标签特征选择中不同的相关性分配策略统一解释为隐含的社会福利函数，通过单一公平参数 α 实现从功利主义到最大最小的信息覆盖分配，并证明在 α≥0 时贪婪选择具有 (1-1/e) 的近似保证。 |
| [^109] | [Steerable by Design: Conditional Pretraining for Robust Control of Language Models](https://arxiv.org/abs/2611.32199) | 该论文提出了一种条件预训练方法和架构，通过编码器交叉注意力机制从一开始就将指令与数据通道分离，构建出Mancelle-3B这一在设计上即可被有效操控的语言模型，解决了模型误将数据当作指令处理的越狱与对齐漂移问题。 |
| [^110] | [A Unified Control Framework for Gradient Inversion Defense in Federated Learning](https://arxiv.org/abs/2611.32174) | 本文提出一个统一的受控动力学框架来刻画联邦学习中各类梯度扰动防御方法，并据此提出近正交梯度投影（NODE），仅沿正交方向注入扰动以保持梯度下降方向不变，从而在保障隐私的同时大幅降低优化开销。 |
| [^111] | [Multi-view Random-set Learning](https://arxiv.org/abs/2611.32170) | 提出多视图随机集学习（MRSL）新范式，保留各视图对候选类别集合的支持，通过跨视图集合交集直接细化候选组，以应对多视图数据中的视图退化问题。 |
| [^112] | [RetraQ: Manifold-Constrained Critic Guidance for Flow Policies in RL](https://arxiv.org/abs/2611.32148) | RetraQ利用预训练流模型编码的几何结构（去噪映射及其雅可比矩阵）来过滤和校正Critic梯度引导，使其近似于向行为流形的切向投影和回缩，从而避免引导过程被Critic外推误差误导，有效改进强化学习中的流匹配策略。 |
| [^113] | [ARS-VLM: Visually Aligned Augmented Random Search for Vision-Language Models](https://arxiv.org/abs/2611.32147) | 提出ARS-VLM，一种视觉对齐的增强随机搜索零阶方法，解决了直接将进化策略类算法应用于视觉-语言模型时的图像块网格过拟合和灾难性遗忘两大失败模式，同时规避了GRPO等一阶强化学习带来的内存与系统开销。 |
| [^114] | [Calibrated Reading of Latent Reasoning and Communication](https://arxiv.org/abs/2611.32143) | 本文提出校准潜读取器（CLR），一个紧凑的Transformer，通过任务条件化交互、源适配器迁移与保形校准，以比语言模型解码器低数百倍的计算成本准确读取潜推理结果与键值通信内容，并提供统计覆盖率与误报保证。 |
| [^115] | [VIDE : Training-Free Video Anomaly Detection via Visual Intrinsic Distribution Estimation](https://arxiv.org/abs/2611.32138) | VIDE是一种无需训练的视频异常检测框架，通过利用冻结片段嵌入估计每个测试视频的正常场景内容分布，自适应修正文本参考并重调整类别响应，从而减少跨场景误报并降低对提示词选择的敏感性。 |
| [^116] | [Interpretable Orthogonal Latent Diffusion with Scale-Selective Uncertainty for Medical Image Super-Resolution](https://arxiv.org/abs/2611.32135) | 提出Med-PODiff分块POD扩散框架，将预测不确定性分解为粗细两个尺度并定义逐像素幻觉风险评分，用于脑部MRI超分辨率中的细尺度幻觉识别。 |
| [^117] | [Rethinking Reasoning Effort Training in Large Language Models: On-Policy Distillation or Reinforcement Learning?](https://arxiv.org/abs/2611.32131) | 带长度惩罚的强化学习虽能压缩单一策略的推理长度，却无法联合训练多个推理努力层级；而多教师策略内蒸馏凭借层级特定的训练目标，能够同时实现推理压缩与层级分离。 |
| [^118] | [Bit-Flip Attacks on Quantized Neural Networks without Knowledge of Model Parameters](https://arxiv.org/abs/2611.32128) | CoarseHammer是一种无需模型参数、梯度或数据的比特翻转攻击方法，它利用量化不对称性将攻击集中于高精度保留的小型张量，可在任意比特位实施翻转，仅用最多24次翻转就能在包括大语言模型在内的多种神经网络中取得超过81.5%的攻击成功率。 |
| [^119] | [EnSurv: Deep Generative Survival Modeling with Application to Kidney Graft Outcomes](https://arxiv.org/abs/2611.32127) | 该研究提出了深度生成式生存分析框架EnSurv，利用结合Kaplan-Meier尾部补全的能量评分处理右删失数据，以捕捉非线性临床协变量交互作用，从而准确预测肾移植后移植物失效时间，服务于器官分配优化与个体化免疫抑制治疗。 |
| [^120] | [Monte Carlo Tree Search for Classifier Chains](https://arxiv.org/abs/2611.32118) | 本文提出将蒙特卡洛树搜索应用于分类器链，使概率分类器链能够扩展到大量标签的多标签分类任务，同时保持对精确匹配预测不确定性的准确估计。 |
| [^121] | [PhysRF-WorldJEPA: Transforming Heterogeneous Large Language Models into Manifold-Decoupled Millimeter-Wave Radar World Models](https://arxiv.org/abs/2611.32113) | 提出PhysRF-WorldJEPA，一种流形解耦的联合嵌入预测架构，通过将连续电磁状态预测与跨模态语言对齐分离，使大语言模型能够基于毫米波雷达实现全天候、隐私保护的物理场景理解。 |
| [^122] | [EEGJev: A System One Decision Model and Benchmark for Multi-Task EEG Question Answering](https://arxiv.org/abs/2611.32100) | 该论文提出了 EEGJev——一个通过一次前向传播即可对候选答案评分的“系统一”式脑电决策模型，并构建了 EEGJev-Bench 基准，使用户能够仅通过更换问题来动态切换脑电解码目标，在 13 个测试数据源上达到 99.70% 的属性选择准确率，且响应速度优于 NeuroLM。 |
| [^123] | [A Fused Gaussian Process Bandit Framework for Multi-Fidelity Bayesian Optimization](https://arxiv.org/abs/2611.32092) | 该论文将多保真度贝叶斯优化问题创新性地表述为融合高斯过程老虎机问题，并开发了具有理论性能保证的FGP-UCB算法，从而更准确地建模保真度层级间的复杂依赖关系。 |
| [^124] | [KKT Atlas: Learning Low-Dimensional Local Scenario Models for Contextual Stochastic Optimization](https://arxiv.org/abs/2611.32071) | KKT Atlas通过将情境划分为局部图册并在其中学习低维子空间，仅用4-18个场景即可逼近条件KKT最优性条件，相比需要500个场景的传统一体化学习方法实现约4-23倍加速，同时保持相当或更低的风险。 |
| [^125] | [Fix How It Acts, Not How It Thinks: Surgical Fine-Tuning of Frontier LLMs](https://arxiv.org/abs/2611.32064) | 提出一种“精细微调”方法，不增大更新规模，而是直接在权重更新合并后残存的部分上进行训练，从而在不干扰模型推理能力的前提下纠正其行动层面的小失误，并解决了 LoRA 小幅更新在合并到量化发布版本时大量丢失的问题。 |
| [^126] | [Integral DeltaNet: Improved Linear Attention with Adaptive Memory Integration](https://arxiv.org/abs/2611.32063) | Integral DeltaNet通过同时维护快速更新的delta状态和保留其加权历史的积分状态这两个互补记忆，并在读取时自适应地融合二者，从而在缓解线性注意力记忆冲突的同时保留对未来预测有用的历史信息。 |
| [^127] | [Spec-CoFrGeNet: Reparametrizing Continued Fraction Networks for Stable and Accurate Training](https://arxiv.org/abs/2611.32041) | 该论文提出Spec-CoFrGeNet，通过受格林函数启发对部分分母进行变换以约束三对角哈密顿量的谱，从而解决连分数网络因分母趋零产生极点导致的训练不稳定问题，无需下界约束和增量式训练即可实现稳定且精确的训练。 |
| [^128] | [FlashQLA: Efficient Context Parallelism for Gated DeltaNet](https://arxiv.org/abs/2611.32039) | FlashQLA是一个面向Gated DeltaNet的上下文并行执行框架，通过衰减感知准备、Dual CP、Auto CP和CP感知内核协同设计，挖掘序列级并行并降低边界状态预处理开销，在长上下文场景下实现了最高4.4倍的显著加速。 |
| [^129] | [Online Environment Repair for Coding Agents](https://arxiv.org/abs/2611.32037) | 该论文提出“在线环境修复”方法，在强化学习训练过程中持续检测并修复编程智能体环境中可被利用的验证器缺陷（如不完整的单元测试），在不惩罚有效解法、不丢弃高难度任务的前提下，从源头抑制奖励作弊行为。 |
| [^130] | [Scale-Robust and Global-Aware Representation Learning for Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2611.32033) | 本文提出STAGER方法，通过联合学习尺度鲁棒的时间表示与全局感知的关系表示，解决了多元时间序列异常检测中跨时间尺度异常捕捉与序列间全局依赖建模两大关键挑战。 |
| [^131] | [Unifying Partner and Environment Diversity to Improve Human-AI Coordination](https://arxiv.org/abs/2611.32031) | 该论文提出跨环境协同博弈（CECP）两阶段框架，通过统一建模伙伴多样性与环境多样性，显著提升AI智能体在陌生环境中与陌生人类伙伴的协调泛化能力。 |
| [^132] | [IncrementDET: Continual AI-Generated Music Detection via Enrollment](https://arxiv.org/abs/2611.32023) | IncrementDET通过为每个AI音乐生成器注册一个独立的线性分类器“针”来实现持续式AI生成音乐检测，无需大规模重训练即可快速适应新生成器，同时保持对已学习生成器的检测性能和稳定的误报率。 |
| [^133] | [PFedHASNet: A Personalized Federated Hybrid Analog-Spike Network for Multi-Subject Motor Imagery EEG Decoding](https://arxiv.org/abs/2611.32019) | 该论文提出PFedHASNet，一种个性化联邦混合模拟-脉冲网络，通过可学习权重融合连续时间特征与LIF脉冲响应，并借助客户端私有的残差适配器和分类器，在保护数据隐私的前提下实现多被试运动想象脑电的个性化解码。 |
| [^134] | [Structure-Aware Transformers for Hierarchical Authorization in Personal AI Agents](https://arxiv.org/abs/2611.32018) | 提出PATH架构，通过层级感知注意力掩码与父门控概率机制，为个人AI智能体预测与权限层级结构一致的用户授权偏好，从而在保障隐私安全的同时减轻手动审查权限请求的负担。 |
| [^135] | [To Critic Or Not To Critic](https://arxiv.org/abs/2611.32013) | 本文从梯度方向的偏差-方差权衡入手，揭示PPO（带评论家）与GRPO（不带评论家）之间没有普适的优劣答案，方法选择取决于批次大小、对评论家的依赖程度和任务复杂度等相互作用因素。 |
| [^136] | [LIBRA: Runtime Co-Tuning of Batch Size and Parallelism for Time-Efficient LLM Training](https://arxiv.org/abs/2611.32008) | LIBRA首次将批大小与并行配置协同调优，以qNEHVI多目标搜索为核心，结合层次化剪枝和采集引导的并行评估，实现最小化大语言模型训练端到端时间的轻量级运行时框架。 |
| [^137] | [Grounding Language Models in Revealed Audience Behavior](https://arxiv.org/abs/2611.31999) | 该论文提出将语言模型在受众真实的点击、点赞、观看记录上进行微调（即“受众显性行为锚定”），使同一个8B模型预测标题点击的准确率从约50%提升至86.5%，并证明“直接询问模型”与“行为锚定”之间的差距主要源于训练信号，而非模型规模或提示方式。 |
| [^138] | [ADE: A Zero-Training Unlearning Method for Latent World Models](https://arxiv.org/abs/2611.31998) | 提出零训练闭式遗忘算法ADE，仅利用缓存的预训练统计矩即可从潜在世界模型中精准切除后门触发器与过时物理机制的污染，且无需原始保留数据、不扭曲潜在流形、不损害有效控制响应。 |
| [^139] | [Metric-Dependent Scaling Laws for Sketched Multiclass Softmax Regression](https://arxiv.org/abs/2611.31995) | 该论文在高斯师生模型下证明，草图化多分类Softmax回归中交叉熵超额风险与0-1超额风险共享由分辨率项加“有效学习维度除以样本量”构成的共同谱尺度，并据此给出率最优的草图规模与训练时程选择。 |
| [^140] | [How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding](https://arxiv.org/abs/2611.31990) | 该论文量化了扩散语言模型并行解码中基于边缘分布的解码器对提交token集合条件总相关性（即依赖性损失）的预测能力，发现其预测准确性主要来源于边界本身，且估计值在散文文本上通常比精确 KL 代价低 0.4 到 1.1 比特。 |
| [^141] | [Discovering Bandwidth-Efficient Prefetchers to Shrink the Cache Hierarchy](https://arxiv.org/abs/2611.31983) | 本文提出DuoArchitect——一个自主智能体框架，通过多保真度仿真与多核协同演化，联合发现覆盖L1D、L2和LLC各层级的协调带宽高效预取策略，其仿真速度比传统集群评估快3-26倍，从而帮助缩小缓存层次结构、释放芯片面积。 |
| [^142] | [A Minimax Optimal Leave One Out Bound for Bounded Losses](https://arxiv.org/abs/2611.31972) | 本文针对有界损失下的留一预测问题给出了极小极大最优损失界，证明超额损失阶为 $\min\{B, \sqrt{Br\log M/n} + B\log M/n\}$，且单个确定性规则即可对所有 $r$ 达到该速率，并同时给出了匹配的下界。 |
| [^143] | [Predict Backbone, Graft Dynamics: A Decoupled Paradigm for Time Series Forecasting](https://arxiv.org/abs/2611.31970) | 该论文提出GRAFT解耦预测范式，将时间序列预测分解为可靠的主干预测与基于历史证据检索的动态嫁接恢复，有效保留突变、激增和转折点等高效用局部动态。 |
| [^144] | [On the Information Loss of Multi-Token Prediction: Origin and Solution](https://arxiv.org/abs/2611.31966) | 该论文揭示了多词元预测中误差累积源于参考目标间的跨块依赖性，并提出Dual-MTP方法，通过在完整与随机掩码上下文下双重监督共享预测头来缓解信息损失。 |
| [^145] | [Sounding the Black Box: Online Training of Embedded Neural Networks in Forward-Only Physical Simulators](https://arxiv.org/abs/2611.31964) | 该论文提出FORGE在线训练框架，通过将精确的网络导数与黑箱模拟器动力学解耦，仅利用不可微分求解器的前向运行即可训练嵌入PDE求解器中的神经网络，从而突破了传统BPTT方法对可微分求解器接口的依赖。 |
| [^146] | [Are Web Agents Still Detectable? Stress-Testing Bot Defenses with Coordinated Evasion](https://arxiv.org/abs/2611.31962) | 论文提出了DarkAgent自主网络智能体，通过结合隐藏自动化痕迹的Chromium浏览器与联合生成逼真按键事件及时间间隔的类人鼠标键盘模型，实现跨浏览器指纹、自动化控制和交互行为的协同规避，实验表明现有机器人防御系统难以检测此类攻击。 |
| [^147] | [DUST: Directional Uncertainty-aware and Scale-invariant Transfer Learning](https://arxiv.org/abs/2611.31948) | 提出了一种基于源回归参数尺度不变方向分布的概率迁移学习框架DUST，能够同时纳入源参数的不确定性并实现尺度不变且稳健的迁移学习。 |
| [^148] | [MELODI: Similarity-Identity Decoupling for Lightweight Ranking](https://arxiv.org/abs/2611.31940) | 本文提出相似性-身份解耦框架 MELODI，证明了相似度算子线性条件下“先池化后相似”与“先相似后池化”等价，从而以更高效的轻量级设计在公开及工业数据集上达到或超越现有排序基线。 |
| [^149] | [Beyond the Last Loop: Reinforcement Learning with a Teacher Extrapolated from the Model Itself](https://arxiv.org/abs/2611.31939) | 本文提出ELGO方法，通过从循环语言模型自身首轮到末轮外推出一个比最后一轮更强的“内生教师”分布，并在GRPO中加入逐token密集蒸馏项，让教师随学生共同进化，从而在不依赖外部教师的情况下解决了循环模型使用可验证奖励进行强化学习时奖励信号稀疏的难题。 |
| [^150] | [Rethinking Universal Time Series Foundation Models from Unified Representations to Shared Dynamics (UniTFU)](https://arxiv.org/abs/2611.31921) | 该论文提出时间序列基础单元（TFUs）作为从多尺度时间片段中发现的可复用构建模块，通过显式刻画跨领域共享的重复性局部动态模式，而非依赖隐式统一表示，来提升时间序列基础模型的跨领域泛化能力。 |
| [^151] | [Meta-Amortized Cognitive Modeling](https://arxiv.org/abs/2611.31914) | 该论文提出CogFormerV2元摊销框架，用单个网络一次性将贝叶斯推断摊销至整个认知模型空间，仅需在单GPU上训练约一天，即可对涵盖二元、多选项和连续决策的超过146亿个模型配置快速给出准确且校准良好的后验分布，从而免去每次模型修改后重新模拟与训练的成本。 |
| [^152] | [ReJEPA: Bounded Predictive Regulation for Latent World Models](https://arxiv.org/abs/2611.31910) | 提出 ReJEPA，一种有界预测调节原则，将大误差的有界梯度与零附近的平滑二次行为相结合，解决了潜在世界模型训练中 L2 损失过拟合噪声帧与 L1 损失破坏局部平滑性的矛盾。 |
| [^153] | [TQ-HQNet: A Trainability-Aware and Qubit-Efficient Hybrid Quantum Neural Network for Cross-Domain Learning on Heterogeneous Data](https://arxiv.org/abs/2611.31908) | 提出了TQ-HQNet混合量子神经网络，通过自适应潜在压缩、可训练性感知的紧凑变分电路搜索和浅层量子专家混合融合三种机制，在保持量子比特高效的同时，显著提升了跨异构数据领域学习的可训练性与鲁棒性。 |
| [^154] | [OrangeTree: A Linear and Tree-based Time Series Forecasting Model Supporting Multiple Input and Output Lengths](https://arxiv.org/abs/2611.31905) | OrangeTree通过分段树编码器、逆树解码器以及范围加权与特征融合机制，提出了一种将模型参数与序列长度解耦的线性树结构架构，在统一框架内实现了多输入多输出长度的时间序列预测，并在六个基准上达到了SOTA性能。 |
| [^155] | [MIFLOW: Dynamic Recent-Behavior Priors for Reference-Calibrated Online Flow-Policy Reinforcement Learning](https://arxiv.org/abs/2611.31902) | MIFLOW通过用拟合最近执行动作的动态流匹配先验作为移动密度参考，对在线流策略强化学习的演员更新和评论家目标进行维度归一化校正，在抑制低密度不可靠动作的同时保持探索能力，从而在MuJoCo连续控制任务上取得有竞争力的性能。 |
| [^156] | [A General Framework for E-process Stopping Rules in Best-Arm Identification](https://arxiv.org/abs/2611.31898) | 该论文提出了最优臂识别中e-过程停止规则的通用框架，通过e-过程分解将最优性条件分为三个部分并证明了渐近最优性的充分条件，进而推导出广义线性老虎机问题的渐近最优算法。 |
| [^157] | [Multi Agent Reinforcement Learning enhanced Federated Learning for Predicting Remaining Useful Lifetime](https://arxiv.org/abs/2611.31880) | 该论文提出了一种基于QMIX值函数分解的协作式多智能体强化学习框架，通过自适应选择各客户端传输的模型层子集并限制重传次数，解决了有损无线信道下联邦学习在剩余使用寿命预测中因丢包导致模型交换脆弱的问题，同时兼顾了数据隐私保护。 |
| [^158] | [EEG-Jev: A Unified Single-Pass Decision Model for EEG](https://arxiv.org/abs/2611.31875) | EEG-Jev提出了一种统一的脑电单次前向决策模型，一次前向传播即可完成选项选择、命题判断与有序评分三类任务，并通过先验校正和温度缩放提供含义明确、可校准的概率输出，在多个公开脑电数据集上取得了优异表现。 |
| [^159] | [Subliminal Learning Crosses Modalities](https://arxiv.org/abs/2611.31870) | 本文发现基于文本的潜意识学习可以跨模态迁移——通过数字序列在视觉-语言模型间传递的主题偏好不仅保留在文本中，还会改变图像的生成和表示方式，其机制源于学生模型变化中的一个恒定分量。 |
| [^160] | [Start Where the Data Are: Statistical Priors for One-Step Generation on Manifolds](https://arxiv.org/abs/2611.31856) | 该论文提出基于切空间高斯混合先验与后验耦合（并借助隐马尔可夫先验处理长扭转角链）的一步生成框架，使采样从数据密集区域出发，避免在流形低密度间隙产生虚假样本，从而提升蛋白质设计等科学应用中的生成质量与效率。 |
| [^161] | [Two-Stage Post-Training for Flow-Based Embodied Foundation Models](https://arxiv.org/abs/2611.31850) | 该论文提出了一种基于流的具身基础模型的两阶段后训练框架，第一阶段通过多教师在线策略蒸馏高效整合专家能力，第二阶段将强化学习与基于采样噪声加权的教师蒸馏相结合，从而突破性能瓶颈、实现超越教师模型的表现。 |
| [^162] | [Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks](https://arxiv.org/abs/2611.31839) | 本文提出规格-测试覆盖差距（STCG）这一1-4分制新指标，利用两个LLM评判者自动检测SWE-bench风格基准中的过度指定测试，从而取代昂贵的专家审查，识别会导致正确的替代修复方案被误判失败的测试用例。 |
| [^163] | [Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models](https://arxiv.org/abs/2611.31821) | 本文提出将多人社交游戏作为可验证训练环境，通过SFT和RLVR使Qwen3-4B的胜率从33%大幅提升至81%，超越了远大于自身的Gemma-3-27B模型，并能泛化到未见过的游戏中，效果远优于提示等上下文策略。 |
| [^164] | [How Far Apart Are Two Proofs? Measuring Proof Dissimilarity in a Proof Kernel and in Lean/Mathlib](https://arxiv.org/abs/2611.31811) | 该论文针对“同一定理的两个证明相距多远”这一源于希尔伯特第 24 问题的基础问题，定义了 6 大族共 19 种证明差异性度量和 10 个期望性质，并在 Curry-Howard 内核与 Lean/Mathlib 中加以实例化与验证。 |
| [^165] | [Learning Compact State Abstractions for LLM Agents](https://arxiv.org/abs/2611.31800) | 本文提出将智能体状态定义为交互历史的紧凑抽象，以确定性递归程序表示并用LLM引导的遗传算法搜索，在ALFWorld和ScienceWorld上保持接近参考智能体的任务成功率的同时，将策略输入最多减少71%和57%，且得到的程序可直接审查。 |
| [^166] | [Byte Patches versus BPE at Matched Data: A Matched-Twin Comparison at 350M Parameters](https://arxiv.org/abs/2611.31790) | 在350M参数、100亿字节的严格匹配数据条件下，8k词表的BPE分词在每字节比特数及多数下游任务上优于字节补丁方法，同时研究揭示了外推上下文、混淆比特分母等可能颠覆结论的评估陷阱。 |
| [^167] | [RoMo Hands: A Large Scale Richly Organized Text to Hand Motion Dataset](https://arxiv.org/abs/2611.31766) | 本文提出了RoMo Hands数据集——包含约54.2万序列（884小时）的真实场景文本-手部动作数据，规模比以往工作大一个数量级，并通过严格的灵巧性过滤、层次化多粒度文本标注以及新颖的393维双手表示方法，弥合了身体与手部动作生成架构之间的差距。 |
| [^168] | [Geo-SMILE: Statistical Model-Agnostic Interpretability with Local Explanations for Geospatial Machine Learning](https://arxiv.org/abs/2611.31731) | 提出Geo-SMILE框架，通过互补的特征屏蔽与地理重定位两个扰动分支以及距离加权的替代模型，将实质性特征解释与地理空间位置响应区分开来，为地理空间机器学习提供模型无关的局部可解释性。 |
| [^169] | [Tokenization and Architecture Jointly Allocate Component Roles in Time Series Transformers](https://arxiv.org/abs/2611.31730) | 该论文提出基于15个token级诊断任务的时间探测框架，揭示时间序列Transformer中构建时间特征的组件并非固定，而是由标记化与架构共同分配——在patch标记化（PatchTST）下FFN承担特征构建角色，其增益平均达注意力机制的3.2倍。 |
| [^170] | [EventSSM: Causal Spatial-State Modelling for Asynchronous Events Event-Depth Estimation](https://arxiv.org/abs/2611.31720) | 提出EventSSM——一种用循环记忆按时间顺序处理事件数据包的因果空间-状态模型，实现异步事件关联的度量深度估计，并通过相机位姿几何传输与可微渲染桥接异步预测与稀疏帧时刻监督，在多个深度误差指标上超越已有方法。 |
| [^171] | [Stopping criteria in structural reliability analysis: An overview](https://arxiv.org/abs/2611.31706) | 本文系统综述了结构可靠性分析中各类停止准则，指出基于误差估计、方差和稳定性检测的经典方法虽源自其他领域，但在结构可靠性分析中具有广泛的适用性和应用潜力。 |
| [^172] | [CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence](https://arxiv.org/abs/2611.31694) | CATWAY 提出了一种代价敏感的三支决策门控框架，利用四种指令条件下的损失矩阵和校准的语言模型置信度概率，让机器人在执行、追问与拒绝之间做出决策，并在跨领域测试中无需重新拟合即保持有效。 |
| [^173] | [Online Statistical Inference for Derivative-Free Constrained Stochastic Optimization](https://arxiv.org/abs/2611.31687) | 本文提出一种基于随机投影的免导数随机序列二次规划方法（SDF-SSQP），在无导数信息的等式约束随机优化问题中，将每轮迭代成本降至 $O((d+m)^2)$ 的同时实现有效的在线统计推断。 |
| [^174] | [Curvature-Equilibrated Policy Optimization for Deep Reinforcement Learning](https://arxiv.org/abs/2611.31680) | 提出EQUIP方法，通过均衡策略损失的正负曲率来平衡参数空间的局部收缩与扩张，从而改进深度强化学习中基于梯度的策略优化。 |
| [^175] | [Neural Interoperability Protocols: Specifying Compatibility Before Training](https://arxiv.org/abs/2611.31670) | 该论文提出神经网络互操作协议（NIP），将模型间的兼容性从“事后由模型对性能推断”转变为“训练前以公开接口规范明确规定”，独立于具体架构和训练过程来约束表示交换的边界、语义与下游支持的计算。 |
| [^176] | [A Piecewise-Linear Activation Function for Implicit Neural Representations](https://arxiv.org/abs/2611.31666) | 本文提出JaSIN——一族介于ReLU与分段线性正弦之间的有界分段线性激活函数，通过引入有界范围、居中凸起和有限振荡，将ReLU在图像重建任务上的PSNR从24.0 dB提升至37.3 dB，在十一种激活函数中位居第三，接近最先进水平。 |
| [^177] | [Boundary Representation-Driven Graph Transformer Modeling for Machining Operation Sequencing](https://arxiv.org/abs/2611.31661) | 该论文提出了一种基于图Transformer的框架，直接从B-Rep设计几何预测加工工序序列，充分利用边界表示模型中丰富的几何与拓扑信息以实现工艺规划的自动化。 |
| [^178] | [Anchoring VAE Fine-Tuning to Its Latent Diffusion Prior](https://arxiv.org/abs/2611.31651) | 提出扩散先验锚定（DPA）正则化方法，在VAE微调时通过分数蒸馏将潜变量引导至预训练扩散模型能准确去噪的高密度流形，在多个模型和任务上将预测误差降低4.0%至7.5%。 |
| [^179] | [Reward Basis Decomposition for Interpreting Preference Heterogeneity](https://arxiv.org/abs/2611.31640) | 提出奖励基分解（RBD）方法，将奖励模型中每个用户的权重分解为群体共享项与个体项，从而分别解析群体共享价值观与个体偏好差异，并发现真实数据中偏好主要由单一群体方向解释。 |
| [^180] | [What You See Is What You Certify: Concept-Based Certificates for Unseen Compositions](https://arxiv.org/abs/2611.31627) | 该研究揭示了基于概念的模型在未见属性组合上会以高达93-94%的概率错误认证类别，并将本体论视为错误检测码，据此确立了有效决策证书的三个条件：排除所有竞争类别的可见证据、如实标注未见部件、以及不受共现捷径驱动的罕见误读。 |
| [^181] | [What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models](https://arxiv.org/abs/2611.31619) | 该论文提出“排行榜分辨率”这一概念并开发了Vernier工具来量化它，揭示出GIFT-Eval等主流零样本时间序列预测基准的分辨率远大于相邻模型间的实际分差、因而无法可靠区分模型优劣，而BOOM凭借2,554个独立序列成为唯一具有足够分辨率的基准。 |
| [^182] | [Eliciting sandbagged LLM capability via structural interventions on attention-head functional networks](https://arxiv.org/abs/2611.31615) | 该研究通过基于注意力头激活构建功能网络图，发现了跨模型架构一致的藏拙结构性特征，并证明可通过对该功能网络施加结构干预来引导出模型被隐藏的能力。 |
| [^183] | [Training decision models for simulating human respondents](https://arxiv.org/abs/2611.31613) | 在历史受访者选择数据上对KEV决策模型进行选择训练，可将LLM合成受访者在跨领域留存研究上的处理效应误差降低22%至89%，且小模型可胜过更大的现成模型。 |
| [^184] | [Instance-Dependent Bandit Convex Optimization in One Dimension](https://arxiv.org/abs/2611.31608) | 本文在一维赌博机凸优化中首次刻画了实例依赖样本复杂度，引入新量 Δ_ε(f) 证明定位极小值点问题可获得依赖该量的加速，同时惊人地发现简单遗憾设定下不存在任何实例依赖的改进。 |
| [^185] | [Noise-Level Adjacency in Diffusion Training](https://arxiv.org/abs/2611.31605) | 本文揭示了扩散模型跨噪声级别共享权重的收益来源于噪声级别嵌入中的邻接性而非方向性，并从理论上给出共享网络拟合误差的下界，即由相邻后验均值间的距离减去网络在相邻嵌入间可调整的幅度所决定。 |
| [^186] | [RePT: 3D Geometry Emerges from 2D Relational Pretraining](https://arxiv.org/abs/2611.31601) | RePT通过跨视角二维点预测学习物理点之间的关系，使三维几何结构无需三维预测目标即可从二维关系预训练中自然涌现，并展现出跨数据集的泛化能力。 |
| [^187] | [LUCID: Statistically Grounded Self-Correction for LLM Reasoning Evaluation](https://arxiv.org/abs/2611.31595) | LUCID无需重新训练或访问模型权重，通过将提示自我纠正与基于设计的监督学习相结合来校准语言模型评分器对推理步骤的评估，将GPT-5-mini的校准差距从7.8个百分点降至0.9个百分点，误报减半，错误检测F1提升32%。 |
| [^188] | [One Phase Is Enough: Sim-to-Real Transfer with Direct Visual Reinforcement Learning](https://arxiv.org/abs/2611.31588) | 提出DRIFT方法，通过使用冻结的预训练基础视觉编码器实现高效的单阶段视觉强化学习，直接从图像观测完成仿真到现实的策略迁移，无需传统的“教师-学生”蒸馏流程。 |
| [^189] | [LatentLens: Rethinking Latent Interface in Multi-Agent Systems](https://arxiv.org/abs/2611.31582) | 本文提出LatentLens框架，从外部和内部两个层面解耦多智能体系统中的潜在接口，发现在相同智能体框架下引入潜在接口并不能提升甚至可能显著降低性能，其效果取决于重新对齐兼容性、信息可访问性和传播动态等内部因素。 |
| [^190] | [Latent Channel Completion for Forecasting in Sensor Networks with Failed Detectors](https://arxiv.org/abs/2611.31575) | 提出潜在通道补全方法，通过在训练中重建失效传感器的潜在表示而非其数值，使传感器网络预测模型在部分探测器失效时仍能保持对健康传感器预测的准确性。 |
| [^191] | [Denoising Diffusion with Progressive Posterior Sampling](https://arxiv.org/abs/2611.31558) | 该论文提出带流形投影的渐进后验采样（PPS）方法，通过闭式高斯后验近似自动估计噪声方差、避免SVD，并将样本投影至局部含噪图像流形，从而更有效地利用扩散先验的局部几何结构求解含噪线性逆问题。 |
| [^192] | [ExitLUT: Boolean Early Exits for Efficient FPGA-Based Neural Network Inference](https://arxiv.org/abs/2611.31540) | ExitLUT用直接映射到FPGA LUT的可学习布尔函数替代softmax置信度退出决策，将节省的退出路径资源重新投入主数据通路，在保持精度不变的前提下实现高达4.08倍的吞吐量提升。 |
| [^193] | [JevBench: Metamorphic Coherence Testing for Typed Probabilistic Decision Models](https://arxiv.org/abs/2611.31533) | JevBench 是一个无需金标准标签的蜕变一致性测试基准，通过覆盖表示、批次独立性、概率测度、逻辑顺序与选项集条件化五个维度的 50 条蜕变关系，系统检验类型化概率决策模型在问题改写、否定、组合等变换下的概率输出是否符合概率论与选择定律，并对 21 个开源模型进行了评测。 |
| [^194] | [ECLIPSE: Embedding-Centered Foundation-Model Federated Knowledge Transfer for Data-Free Edge Intelligence](https://arxiv.org/abs/2611.31527) | 该论文提出ECLIPSE框架，以嵌入为中心，利用服务器托管的基础模型建立共享嵌入参考空间，在不共享原始数据、本地模型或公共数据集的前提下实现无数据场景下的联邦知识迁移与边缘智能。 |
| [^195] | [Diagnosing Epistemic Uncertainty in Neural Operators: Effects of Initialization, Discretization, and Function Sampling](https://arxiv.org/abs/2611.31526) | 该论文系统研究了神经算子中认知不确定性的三大来源（训练函数采样、输出离散化、模型初始化），通过训练约46,000个模型并构建集成，定量刻画了各来源对算子学习性能差异的独立贡献。 |
| [^196] | [SOTS: Signature-based Optimal Transport Selection](https://arxiv.org/abs/2611.31525) | SOTS提出了一种无监督核心集选择框架，通过将不规则多变量时间序列映射为截断路径签名并贪心最小化子集与全量数据之间的切片Wasserstein距离来选择训练数据，同时提供了随机核心集的收敛性理论保证。 |
| [^197] | [How to build and train your recurrent-depth model: A fixed-point perspective](https://arxiv.org/abs/2611.31513) | 该论文从不动点理论视角揭示了循环深度模型中输入注入本质上就是Halpern迭代，证明了截断反向传播仅需多对数深度的窗口即可保证收敛，并据此开发了无额外开销的Sparrow优化器，可无缝替换Muon/MuonH并加速训练。 |
| [^198] | [Near-Zero Forgetting Is Not Enough: Out-of-Distribution Robustness in Continual Learning of Pretrained VLAs](https://arxiv.org/abs/2611.31511) | 该论文首次揭示了带经验回放的持续学习虽能使预训练VLA模型实现近乎零遗忘，但模型对分布外数据的鲁棒性会随持续学习持续退化，且这种退化遵循集中于多模态与动作特征的跨领域共享表征漂移模式。 |
| [^199] | [Measuring Counterfactual Credit in a Replayable Wildfire-Suppression Benchmark](https://arxiv.org/abs/2611.31499) | 提出可回放的野火扑救基准FLARE，通过在同一随机事件下以不同动作反事实回放火灾的任意时刻，直接测量多智能体动作的信用分配，并以“模拟器信用”指导训练和评价评论家，相比传统MAPPO显著降低了燃烧分数。 |
| [^200] | [ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed](https://arxiv.org/abs/2611.31495) | ModernELECTRA提出了一种预训练方案，在保留替换词元检测（RTD）主目标的同时，引入轻量级的边界窗口重建辅助任务，使紧凑编码器既能擅长分类任务，又能胜任检索与语义相似度等嵌入任务。 |
| [^201] | [Trust-Region Optimization for Performative Prediction without Response Modeling](https://arxiv.org/abs/2611.31492) | PTRO方法通过从当前总体采样评估已知损失、仅在估计总体响应影响时进行额外部署，无需响应建模即可利用已知损失函数对执行性风险进行局部近似优化，并结合置信域机制进行决策更新。 |
| [^202] | [CHORD: Learning Latent Spaces for Constrained Stellarator Design](https://arxiv.org/abs/2611.31486) | CHORD是一种条件变分自编码器，通过在训练中惩罚潜空间插值路径上的约束违反并结合自洽推理，实现了仿星器聚变设计中满足物理约束的可行设计插值。 |
| [^203] | [SwiftLoop: Selective Refinement for Adaptive Implicit Reasoning in Looped Transformers](https://arxiv.org/abs/2611.31481) | SwiftLoop 提出在推理时通过位置级冻结策略对循环 Transformer 进行选择性自适应精炼，同时优化循环深度与活跃宽度两个维度的计算分配，以更低的计算成本获得更优的推理精度。 |
| [^204] | [Transferable, Time-Parallel Graph Dynamics via Space-Time Factorization](https://arxiv.org/abs/2611.31479) | 该论文提出时空分解方法，将模型拆分为与图无关的局部时间算子和无图依赖参数的空间混合器，从而实现同一参数在未见图上的零样本迁移，并支持非递归的时间并行评估。 |
| [^205] | [Learning from Human Corrections through Temporal Preferences](https://arxiv.org/abs/2611.31446) | 该论文提出将人类纠正机器人时介入前后的行为及其时间关系转化为偏好对，从而学习更准确逼近人类潜在奖励的价值函数，并在目标推断与纠正预测两个下游任务上验证了其有效性。 |
| [^206] | [Small Learned Projections for Semantic-Reversal-Aware Retrieval with Spectral Acceleration: A Four-Domain Empirical Study](https://arxiv.org/abs/2611.31442) | 该论文提出一种小型学习投影，在法律、医疗、专利、金融四个领域实证中能有效阻止对抗性“陷阱”文档进入检索前10名，同时保留93-99%的正确释义文档，并借助边界条件化的谱图检索加速器实现大规模检索加速。 |
| [^207] | [Transformers with Signatures: Towards Universal In-Context Learning in Path Space](https://arxiv.org/abs/2611.31439) | 该论文利用路径签名特征建立了面向路径空间的上下文学习理论框架，显式构造了可模拟上下文Lasso的每层双注意力头因果softmax Transformer，并给出超越IID数据设定的有限样本预测误差界。 |
| [^208] | [Is 4 Bits Enough for EEG Sleep Staging? A Subject-Level Benchmark of MXFP4, NVFP4 and FP8](https://arxiv.org/abs/2611.31436) | 该论文首次在受试者级别系统评估了FP16至4位块缩放格式（MXFP4、NVFP4）及FP8在脑电睡眠分期中的量化性能，发现8/16位量化几乎无损，4位量化下时间上下文网络最多仅损失1.19个百分点，且NVFP4优于MXFP4，其中量化原始脑电信号的第一层最为敏感。 |
| [^209] | [The Recycler: In-Transit Machine Learning on High-Fidelity Data from Simulation Campaigns as an Optimized Approach for Building Frontier Models](https://arxiv.org/abs/2611.31434) | 该论文首次实现了在同一台百亿亿次超算系统上，将多个并行仿真产生的高保真数据以流式传输方式直接用于在途机器学习训练，从而绕过文件系统带宽和容量的限制，优化前沿模型的构建。 |
| [^210] | [CaLA: Causal Latent Adjustment for World Models under Adaptive Policies](https://arxiv.org/abs/2611.31431) | 该论文提出 CaLA，一种受 DAG 约束的世界模型，通过从处理前历史中推断潜变量玩家表征，消除自适应策略在想象 rollout 中引入的处理-结果混杂，从而在三消难度调整任务中获得最可靠的因果干预估计和有竞争力的个体化决策。 |
| [^211] | [Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training](https://arxiv.org/abs/2611.31427) | 该论文将智能体能力分解为“任务能力”（解决任务所需的知识与推理）和“界面能力”（通过特定框架表达知识的能力），揭示了基于框架的训练效果取决于这两种能力的交互方式，并提出多框架蒸馏可将不同框架下的任务知识聚合到共享策略中。 |
| [^212] | [OmniX: A first jet and event model](https://arxiv.org/abs/2611.31403) | 本文提出OmniX，首个能在单一模型中同时处理喷注级和事件级数据的基础模型架构，其通过联合训练在生成和分类任务上达到甚至超越了专用的仅喷注或仅事件模型。 |
| [^213] | [CoRe: Jointly Learning Contextual Representations for Understanding and Generation](https://arxiv.org/abs/2611.31397) | CoRe提出让理解与生成任务联合塑造统一的图像与文本上下文表征，一个从零训练的383M模型在无需任何预训练组件的情况下，在理解能力以及文生图、图生文生成效果上均可与基于预训练组件的更大规模模型相媲美。 |
| [^214] | [Detecting Harmful Short Videos in Toxic Diet Culture: A Governance-Ready Evidence Framework with Sparse Autoencoders](https://arxiv.org/abs/2611.31354) | 本文提出SPACE框架，利用大型多模态模型的稀疏自编码器特征空间，识别、定位并将证据对齐到平台政策，为有毒饮食文化等有害短视频的审核治理提供与决策关联且可检验的证据支持。 |
| [^215] | [GHC:Expanding Residual Interfaces in Hyper-Connections](https://arxiv.org/abs/2611.31350) | 提出分组超连接，通过沿注意力头或FFN通道将分支分组并为每组提供差异化的流读写接口，在开销可忽略的情况下将训练效率较Pre-Norm提升35%、较IHC提升10%，同时克服了增加流数量时的收益递减问题。 |
| [^216] | [Exact Volume-Preserving Diffeomorphic Shear Splitting: Continuous-Discrete Geometric Bounds, Reversible Memory Backpropagation, and Expressivity Frontiers in Volumetric Mapping](https://arxiv.org/abs/2611.31343) | 该论文提出基于级联坐标剪切的精确保体积微分同胚配准框架，将无散度李代数分解为六个剪切生成元，实现闭合形式的精确体积保持、机器精度的零迭代求逆以及可逆的低内存反向传播。 |
| [^217] | [Fusion Gains Depend on the Implementation and the Calibration Target: An 18-Task Study of Prediction Intervals with a Single-Source Fallback](https://arxiv.org/abs/2611.31333) | 该研究通过对18个回归任务的实证比较发现，融合方法的收益高度依赖于具体实现方式与校准目标，而结合单源回退的开发集选择器能在所有实现和校准设置下稳定提升预测区间性能。 |
| [^218] | [Conservative Overlap Fuzzy Rough Sets for Incomplete Data](https://arxiv.org/abs/2611.31329) | 该论文提出保守重叠模糊粗糙集（COFRS）模型，通过对共同存在属性进行加权重叠、全局惩罚缺失值的相似度定义及基于加权粒一致性的软依赖度，实现了不完整数据上高效且稳健的属性约减。 |
| [^219] | [Flagging Batch Shift in Flow Cytometry using Jeffreys Divergence between Gaussian Mixtures](https://arxiv.org/abs/2611.31321) | 提出了一种基于高斯混合模型间Jeffreys散度的与模型无关的方法，用于检测流式细胞术数据中的批次偏移并标记不可靠的预测。 |
| [^220] | [Layer-Readout Interfaces in EEG Foundation Models: Predictive Accessibility and Cross-Dataset Transfer](https://arxiv.org/abs/2611.31320) | 本研究解析了EEG基础模型中间层优于最终层的原因：改进最终层读出接口即可恢复大部分预测性能，但冻结提取下中间层仍具优势，且所选择的层与任务方向能够跨数据集迁移。 |
| [^221] | [Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse](https://arxiv.org/abs/2611.31319) | 该论文提出将保形化的分布外异常检测门控与可操作性约束的反事实搜索相结合，防止反事实救济利用梯度提升树决策边界的伪影和低密度分布外区域，从而生成更合理的分布内救济建议。 |
| [^222] | [SlotFill: Repairing Lossy SoC Communication Traces for Post-Silicon Validation](https://arxiv.org/abs/2611.31313) | SlotFill通过为消息间隙分配固定槽位并利用基于Transformer的模型结合监视器字段信息，在流片后验证中精确恢复SoC通信踪迹中丢失的消息，在15%丢失率下恢复率高达97%。 |
| [^223] | [Set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem](https://arxiv.org/abs/2611.31306) | 提出了一种融合记忆机制的基于集合的综合学习粒子群优化算法，用于求解离散优化问题，并通过与现有算法的数值实验对比验证了其有效性。 |
| [^224] | [Differential evolution with perturbation for continuous function optimization](https://arxiv.org/abs/2611.31305) | 本文提出了一种结合扰动机制的差分进化算法（DEP）用于连续函数优化，并通过CEC2017数据集与现有算法的对比验证了其有效性。 |
| [^225] | [Hidden Clues in WM-811K: Uncovering the Spatial Signals Behind Wafer Map Prediction](https://arxiv.org/abs/2611.31292) | 本研究通过为WM-811K晶圆图构建控制变量（布局、失效芯片数量、距中心距离、二维位置）的多种变体，并严格清洗数据集，系统揭示了卷积神经网络预测性能背后真正起作用的空间特征。 |
| [^226] | [LLM Genetic Feature Engineering](https://arxiv.org/abs/2611.31277) | 该论文提出了一种在单个特征层面而非整个特征集层面执行选择、交叉和变异的新型遗传算法，使遗传算法能够在过程中对新特征进行统计分析，并同时整合个体与群体层面的特征指标，从而改进基于LLM的特征工程自动化。 |
| [^227] | [Chain-of-Thought Is Not Explainability](https://arxiv.org/abs/2611.31271) | 本立场论文指出，思维链生成的推理说明经常不忠实于模型真实的内部隐藏计算，因而不能作为可信的可解释性依据，既非必要也非充分。 |
| [^228] | [ToneMamaba: Learning Contextualized Tonal Contours with Boundary-Reset State Space Models](https://arxiv.org/abs/2611.31267) | ToneMamba通过在音节、词和话语单元层面进行双向状态重置的边界感知状态空间模型，结合节奏条件门控建模时长信息，显著提升了普通话声调识别性能，在会话语音上提升尤为突出。 |
| [^229] | [SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching](https://arxiv.org/abs/2611.31265) | 提出SD-Cache，通过状态引导的动态缓存机制，在不预知未来查询的情况下，将LLM原生的KV状态选择性地整合到固定大小的工作记忆中，解决了RAG场景下KV缓存线性增长的问题。 |
| [^230] | [Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios](https://arxiv.org/abs/2611.31249) | Actionwave 是首个覆盖重症监护、手术室和急诊科等多种医疗场景的以波形为中心的临床预测系统性基准，通过统一的决策时点预测接口、严格的评估协议，系统考察了信号时长、预警时间和可用输入等条件对模型表现的影响。 |
| [^231] | [Local subspace encoding for spectral prediction: from TabPFN to operator-adaptive calibration](https://arxiv.org/abs/2611.31245) | 该研究证实局部振动子空间编码（LVSE）表示方法在nirs4all生态系统的光谱数据集上能显著降低TabPFN的RMSE并提升分类加权F1，且其价值可扩展至经典定标方法。 |
| [^232] | [Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals](https://arxiv.org/abs/2611.31094) | LoopRes通过让每个循环层借助可训练的循环感知键直接关注早期循环的输出（并限制在残差窗口内），使预训练大语言模型无需从头训练即可将循环结构转化为显著且一致的增益。 |
| [^233] | [Lorentz symmetry in self-supervised representation learning](https://arxiv.org/abs/2611.31011) | 该论文首次将洛伦兹等变的L-GATr-slim架构与LeJEPA自监督学习框架相统一，证明在潜空间中强制洛伦兹对称性（可通过架构设计或运动学增强学习实现）对喷注标记的表征质量至关重要，并在多项探测准确率上超越了现有基线。 |
| [^234] | [Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching](https://arxiv.org/abs/2611.30986) | 该论文揭示现有前向扩散强化学习方法因行为分布与策略失配导致基线引起的梯度漂移和训练不稳定，并提出瞬时轨迹分布匹配（iTDM）来解决这一问题，从而高效对齐扩散模型。 |
| [^235] | [LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation](https://arxiv.org/abs/2611.30934) | LimeTree是一种内存与计算均为线性复杂度的树图语言模型架构，通过在语言建模训练中集成KV检索任务，仅用极低的计算成本（4块GPU训练6小时）便实现了19的低困惑度，并能将上下文从512外推至16,384个token且保持100%的检索准确率。 |
| [^236] | [Rewirable Circuits in Transformers with Loom Attention](https://arxiv.org/abs/2611.30634) | 提出loom attention使Transformer能在每个token处动态重连其注意力电路，通过垂直注意力与矩阵记忆构成的向量-矩阵双流架构，在仅增加约2%计算量的情况下达到超过3倍计算量训练的Transformer的性能。 |
| [^237] | [Vasicek Diffusion Bridge for Graph Generation](https://arxiv.org/abs/2611.30605) | 本文提出基于Vasicek动力学的图扩散模型VGD，利用具有闭式解的数据锚定均值动力学，从解析上建立高斯噪声到目标图分布的连续过渡，解决了现有图扩散模型不确定性过大、信噪比增长滞后导致结构信息过晚出现的问题。 |
| [^238] | [De Novo Generation of Odorant Molecules with Targeted Olfactory Receptor Activation Patterns](https://arxiv.org/abs/2611.30509) | 该论文提出了一个基于强化学习的气味分子从头生成框架，通过构建嗅觉受体全景结合矩阵、聚类发现代表性激活模式，并采用结合成对排序、分布匹配与复杂度约束的结构化选择性损失，成功生成匹配目标多受体激活谱的分子，在广谱型和稀疏激活型等罕见目标谱上取得了最大提升。 |
| [^239] | [Sparse Hodge Preference Optimization: Fine-Tuning Language Models toward Alignment beyond Transitive Preferences](https://arxiv.org/abs/2611.30135) | 本文提出稀疏霍奇偏好优化（SHPO），借助组合霍奇分解理论为DPO引入稀疏反对称旋度残差，使其能够捕捉人类偏好数据中的非传递性与循环结构，并证明其严格扩大了DPO的假设类。 |
| [^240] | [Multi-Order Passive Macromodeling from Layouts via Nested Port-Hamiltonian Learning](https://arxiv.org/abs/2611.28778) | 该论文提出了一种直接在无源空间中学习的端到端宏建模框架，利用前缀嵌套的端口哈密顿参数化，从物理版图图一次性生成在构造上即保证无源性与稳定性的多阶宏模型，从而免去了逐版图昂贵且耗时的全波电磁仿真。 |
| [^241] | [The Complexity of Oracle-Efficient Realizable Transductive Learning](https://arxiv.org/abs/2611.28757) | 本文证明即使借助能优化任意线性目标的加权 ERM 预言机，可实现转换式在线学习也无法通过多项式次预言机查询达到减半算法的对数级错误保证，并给出了与不可知率上界匹配的 $\Omega_\beta(\sqrt{T\log N})$ 期望错误下界。 |
| [^242] | [Interpreting Diffusion Model Activations in Natural Language](https://arxiv.org/abs/2611.28621) | 本文提出扩散专用激活预言机（DAO），能将扩散模型中间激活翻译为自然语言回答开放式问题，既可从早期去噪步骤预测最终图像内容，又能利用激活空间结构引导生成目标属性并定位关键模型组件。 |
| [^243] | [Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events](https://arxiv.org/abs/2611.27509) | 该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。 |
| [^244] | [Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training](https://arxiv.org/abs/2611.27376) | 提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。 |
| [^245] | [Loop-OPF: Solver-Informed Learning for Optimal Power Flow with Inference-Time Topology Adaptivity](https://arxiv.org/abs/2611.27013) | Loop-OPF是一种由PDHG迭代构建的求解器引导神经网络，在凸性条件下保证DC-OPF预测的最优性与可行性，并通过自监督方式学习PDHG参数实现无需解标签的加速训练与对推理时未见拓扑的自适应。 |
| [^246] | [Harnessing Dephasing for Quantum Reservoir Computing with a Minimum Complexity Architecture: Applications to Nonlinear Time-Series Prediction](https://arxiv.org/abs/2611.26762) | 提出了一种最小复杂度的量子储备池计算模型，证明纯退相位在开放均匀XX链中可保证回声态与衰减记忆特性，且联合控制退相位与振幅阻尼可调节记忆与非线性能力，成功应用于非线性时间序列预测。 |
| [^247] | [Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models](https://arxiv.org/abs/2611.25594) | 本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。 |
| [^248] | [MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection](https://arxiv.org/abs/2611.25194) | 该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。 |
| [^249] | [Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message](https://arxiv.org/abs/2611.23570) | 该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。 |
| [^250] | [Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving](https://arxiv.org/abs/2611.23211) | 该论文提出多模态掩码扩散自编码器 ID-MAE，通过自监督方式重建被掩码的注视与轨迹数据来增强受损驾驶预测，并发布了包含 109 人、103 小时驾驶数据及十种受损诱发条件的大规模模拟驾驶数据集 DIVERT，显著提升了受损驾驶推断效果。 |
| [^251] | [LoopPrune: Recurrent Pruning for Looped Diffusion Transformers](https://arxiv.org/abs/2611.22757) | LoopPrune提出了一种针对循环扩散Transformer的剪枝与恢复框架，通过循环重要性估计、共享重建和循环位置恢复三大机制，在降低存储与计算成本的同时有效保持模型性能。 |
| [^252] | [LoopZip: Visual Token Compression for Efficient Looped Vision-Language Models](https://arxiv.org/abs/2611.22728) | LoopZip是一种无需训练的视觉标记压缩算法，通过在循环主干前构建紧凑候选池并结合跨调用选择，降低循环视觉语言模型处理长视觉序列时的计算与GPU显存开销。 |
| [^253] | [When Should Decision Models Compare Candidates?](https://arxiv.org/abs/2611.20335) | 该论文系统比较了逐点打分、集合打分和候选对象注意力三种候选评分方式，发现逐点打分总体仍有竞争力，集合打分在特定任务上更优且经关系训练后能学习候选间依赖并泛化，而候选对象注意力即使允许跨候选交互也未带来稳定提升。 |
| [^254] | [On-policy distillation as a Backdoor Propagation Channel](https://arxiv.org/abs/2611.19207) | 本研究首次揭示在策略蒸馏可成为后门传播的高效通道：教师模型中被植入的后门仅需极少触发提示暴露即可转移至学生模型，且转移强度显著高于传统蒸馏方法。 |
| [^255] | [Antidistillation is Cheap: Lightweight Distillation-Resistance via Stylistic Corruption](https://arxiv.org/abs/2611.17772) | 该论文提出一种轻量级抗蒸馏防御方法，通过在检测到恶意查询时向教师模型激活中动态注入可定制的风格引导向量，以极低的计算开销破坏序列级蒸馏攻击，同时不影响正常输出的语义质量和流畅性。 |
| [^256] | [GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement](https://arxiv.org/abs/2611.13243) | 该论文发布了开源 SDK GigaEdge Spench，通过共享 C++ 核心与多平台集成（服务器、移动端、浏览器）实现实用的高速流式语音增强，其模型质量和处理速度优于 RNNoise 和 DPDFNet2 且快于实时。 |
| [^257] | [Speculative Self-Distillation enables Efficient Knowledge Internalization](https://arxiv.org/abs/2611.12016) | 提出推测式自蒸馏（SSD）——一种逐 token 混合策略方法：默认由学生生成 rollout，教师仅在下一个 token 分布显著分歧处介入，既贴近学生的测试时行为分布又避免低信号状态，从而实现更高效的知识内化。 |
| [^258] | [Off-Policy RL with Online Inference Tuning: 1st Place Solution to the CoRL 2026 RoboEval Challenge](https://arxiv.org/abs/2611.11464) | 该方案将基于π0.5策略（自任价值函数）的离线策略强化学习与AWR/RECAP优势训练及对约20个推理参数的汤普森采样在线调优相结合，并补充满奖励生成演示与失败恢复数据，最终夺得CoRL 2026 RoboEval挑战赛第一名。 |
| [^259] | [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892) | 提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。 |
| [^260] | [How many sorted projections determine n points in k dimensions?](https://arxiv.org/abs/2611.06470) | 本文借助Hair和Sahai关于随机线性码列表恢复的两个引理，证明了在任意k维空间中，几乎任意选择适当数量的方向即可确定n个点的多重集，从而回答了Dym等人提出的高维开放问题。 |
| [^261] | [Alleviating Limited Effectiveness and Multi-Class Collapse in Deep Graph-level Clustering](https://arxiv.org/abs/2611.06052) | 该论文提出S³GLC框架，利用Hoffman-Wielandt不等式构建谱签名作为图编辑距离的高效下界代理，将全局拓扑特征与局部邻域特征相融合，从而缓解深度图级聚类中区分能力不足和多类别聚类坍塌的问题。 |
| [^262] | [Thought-deciphering via hierarchical EEG-to-text alignment](https://arxiv.org/abs/2611.03623) | 提出一种分层脑电图-文本对齐框架，先解码粗粒度语义类别、再在类别内检索细粒度句子，显著提升了从EEG进行句子级解码的检索准确率。 |
| [^263] | [Write Facts the Way Models Read Them: Lifting the Two-Hop Curse](https://arxiv.org/abs/2611.01519) | 论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。 |
| [^264] | [Lightweight Representation Learning for Low Tucker Rank Tensors](https://arxiv.org/abs/2611.01168) | 本文提出了一种保持迭代轨迹不变的轻量化计算框架，通过仅计算更新所需的奇异三元组、避免Kronecker乘积的收缩以及稀疏加因子存储三个组件，显著降低了五种低Tucker秩张量迭代估计器的内存开销。 |
| [^265] | [Sparse Autoencoder Attribution Graphs Are Decided at the Margin](https://arxiv.org/abs/2611.00606) | 该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。 |
| [^266] | [Confidence-Aware Speed-over-Ground Estimation from Doppler Spectra via Reconstruction-Based Representation Learning](https://arxiv.org/abs/2610.40253) | 该论文提出一种基于自编码器重构式表征学习的置信度感知方法，从多普勒谱中同时估计对地速度与帧级置信度，推理时省略解码器并采用专用置信度头进行误差界限估计。 |
| [^267] | [Tiered Cache Blocking and In-Kernel Task Dispatch for Long-Context Prefill Attention on CPU](https://arxiv.org/abs/2610.38809) | 提出TACPU——一种采用分层缓存分块与内核内任务调度机制的CPU高性能注意力内核，相比现有方案最高加速2.01倍，可将Llama-3.1-8B的首Token生成时间最高缩短1.74倍且不损失精度。 |
| [^268] | [ResCom: Activation Residual Compensation for Large Language Model Quantization](https://arxiv.org/abs/2610.35955) | ResCom提出一种低秩激活残差补偿方法，通过输出感知校准以闭式形式利用激活舍入残差校正层输出误差，无需梯度训练即可显著提升大语言模型4比特权重-激活量化的精度。 |
| [^269] | [A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955) | 本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。 |
| [^270] | [The Missing Compliance Class: LLM Graders That Collapse Jailbreak Success, and a Free Check](https://arxiv.org/abs/2610.24872) | 本文发现在越狱攻击评估中，常用的开源8B/9B LLM评判器会系统性地将“合规”类别坍缩为“规避”类别，导致不安全模型被误判为安全，而只需统计每类标签计数即可零成本检测这种坍缩问题。 |
| [^271] | [HiPPO Functional Memory: Coordinate-Queryable Attention for Irregular Time Series](https://arxiv.org/abs/2610.24501) | 提出HiPPO函数记忆（HiPPO-FM），一种将不规则时间序列历史编译为可重用的坐标可寻址函数状态的注意力算子，可在任意时间坐标处直接求值查询，无需为每个查询反复解析观测相关表示。 |
| [^272] | [Announced Breaks Separate Conformal Reliability from Frequency Calibration](https://arxiv.org/abs/2610.23917) | 本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。 |
| [^273] | [The Ephemeral Cost of Switching Optimizers Across LLM Training Stages](https://arxiv.org/abs/2610.22650) | 该论文通过系统性实验首次发现，在LLM预训练与后训练阶段之间切换优化器（如AdamW与Muon之间）并不会造成实质性的性能损失，最终性能主要由各阶段所用优化器的独立效应决定，且在SFT中增加Muon更新比例反而能提升下游性能。 |
| [^274] | [Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning](https://arxiv.org/abs/2610.18690) | Varest通过语义引导的可变分辨率token化与相关性token剪枝，仅用5.56%的图像token即可保留86%-92%的全token性能，实现高效且语义无损的图像token压缩。 |
| [^275] | [BridgeTwist: Twisting Schr\"odinger Bridges for Training-Free Conditional Sampling](https://arxiv.org/abs/2610.18658) | 提出了BridgeTwist，一种基于带温度调节扭转序贯蒙特卡洛方案的免训练条件采样器，可复用预训练薛定谔桥生成模型，利用由速度场与得分场构造的闭式即插即用扭转代理实现条件后验采样。 |
| [^276] | [SPLICE: Sparse Physics Library Identification and Compositional Expansion](https://arxiv.org/abs/2610.17029) | SPLICE是一个无需机制标签即可从多机制混合数据中自动发现物理算子的框架，通过凸优化推断贡献、校准测试分配机制，实现算子库的自动扩展并组合成零样本模拟器。 |
| [^277] | [Learning joint probabilistic weather forecasts from station observations alone](https://arxiv.org/abs/2610.09898) | CLARA模型仅凭站点观测数据（无需数值天气预报或再分析数据）即可学习五个地面变量的联合高斯概率预报分布，以约2.8万参数的小模型在CPU上实现了优于各类基线的联合概率预报能力。 |
| [^278] | [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。 |
| [^279] | [CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use](https://arxiv.org/abs/2610.09127) | CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。 |
| [^280] | [Convex-Concave Reinforcement Learning](https://arxiv.org/abs/2610.09108) | 该论文揭示强化学习的策略优化问题在对数密度比坐标下具有凸差规划结构，从而将 CPI、NPG、TRPO 和 AWR 统一为特例，突破了传统凸代理近似的局限。 |
| [^281] | [InvestigationWorlds: An Agentic Environment for Legal Investigation](https://arxiv.org/abs/2610.04129) | 该论文利用真实的美国联邦法院简易判决动议构建了法律调查智能体环境InvestigationWorlds，评估发现智能体即使检索到相关证据也常常坚持错误假设，难以区分法院采纳的假设与其他备选假设。 |
| [^282] | [MiDShip: Multimodal Dataset of Ship Cargo Hold Structures for Engineering Design](https://arxiv.org/abs/2610.02214) | 本文提出了MiDShip多模态数据集，包含12,753个船舶货舱结构设计（涵盖参数数据、3D几何、工程图纸、材料清单和结构评估），填补了船舶结构设计中缺乏连接几何、性能与规范约束的结构化数据集的空白。 |
| [^283] | [Beyond Accuracy: Prefix-Invariant Realizations of Low-Precision Fast Matrix Multiplication](https://arxiv.org/abs/2609.39816) | 该论文发现，即使在准确率上表现正常，低精度快速矩阵乘法实现仍可能破坏语言模型多选题评分所依赖的前缀不变性并改变模型答案，证明准确率不足以作为此类实现的认证标准。 |
| [^284] | [ReSCENE: Server-Side Replay for Structural Mitigation of Catastrophic Forgetting in Federated Continual Learning](https://arxiv.org/abs/2609.38833) | ReSCENE通过让客户端上传本地数据的浓缩代理、由服务器端存储并重放这些代理来训练全局模型，并配合时间牧群压缩缓冲区与理论保证，从结构上缓解联邦持续学习中的灾难性遗忘，同时减轻客户端负担。 |
| [^285] | [Accelerator Choice Is Not Enough: AlphaFold2 Inference on Cloud TPUs](https://arxiv.org/abs/2609.34818) | 仅选择TPU等加速器并不足够，软件层的配置（如使用JAX的pmap将查询映射到多颗芯片）才能真正释放AlphaFold2在云端TPU上的硬件性能优势。 |
| [^286] | [Can We Predict Anomaly Detection Performance from Embedding-Space Geometry?](https://arxiv.org/abs/2609.26460) | 该论文为基于kNN的异常检测器推导了AUC下界，将检测性能与嵌入空间几何（分数分离度与方差）联系起来，并提出简单的伪异常探针方法，从而在无需真实异常数据的情况下实现可靠的模型选择。 |
| [^287] | [CLOOPD: Closing the Learner Loop in On-Policy Distillation](https://arxiv.org/abs/2609.24141) | CLOOPD提出一种闭环在策略蒸馏框架，将教师信号获取与学生端实现分离，通过自适应α航点、冻结评分批次、重复前向传播学生模型以及在独立token预算下分配演员工作量，显著提升了宏观准确率。 |
| [^288] | [RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking](https://arxiv.org/abs/2609.23457) | 本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。 |
| [^289] | [RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding](https://arxiv.org/abs/2609.21827) | 该论文提出RheoSampling方法，通过将树构建与token验证两个任务解耦，解决了随机解码下动态树投机解码中草稿分布坍缩为独热概率、接受率骤降的两难困境。 |
| [^290] | [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190) | 该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。 |
| [^291] | [Graph-Based Stochastic Power-UCT: Monte-Carlo Graph Search with Power Mean Estimation](https://arxiv.org/abs/2609.19956) | 提出基于图的随机Power-UCT算法（GS-Power-UCT），通过在相同规划深度共享状态并保留不同深度的独立值，在含环路的一般随机MDP中实现样本复用，证明了以 O(n^{-1/2}) 速率收敛到有限视界值，并进一步提出两种全状态变体以增加样本共享并控制视界混合偏差。 |
| [^292] | [Learning Options for Compositional Motor Control with Adapter Banks](https://arxiv.org/abs/2609.17042) | 该论文将神经科学中“运动基元是共享循环网络低秩扰动”的理论转化为端到端可学习的新架构——共享循环核心加上由离散潜码选择的残差适配器库，适配器在无秩约束下自发涌现低秩特性，并通过冻结网络后优化的高层策略组合选项实现新颖分布外运动的泛化。 |
| [^293] | [Beyond Point Forecasts: A Survey on Probabilistic Forecasting for Time Series and Spatiotemporal Data](https://arxiv.org/abs/2609.13345) | 这篇综述通过按不确定性在预测流程中引入的位置和方式建立统一分类框架，整合了时间序列与时空概率预测中分散的统计建模、机器学习和深度生成建模方法，并涵盖了集成校准、贝叶斯建模、分布回归以及新兴的时间序列基础模型等范式。 |
| [^294] | [Finite Audits of Model Updating: Coverage and Resolution of Paired Disparity Comparisons](https://arxiv.org/abs/2609.09788) | 该论文发现，模型更新中的配对差异审计尽管更精确、更有决断力，其区间对子群体差距变化的覆盖率却远低于名义95%（bootstrap仅12.0%，数值差分替代方法也仅52.9%），且绝对差距偏差、差距方向与先验强度会显著影响点估计与结论。 |
| [^295] | [Support Topology and Gradient Mixing in Sinkhorn Layers](https://arxiv.org/abs/2609.07954) | 本文发展了一套固定支撑演算来分析Sinkhorn层中的梯度传播机制，并证明了传输多面体的每个可行面具有成对两跳列重叠是支撑图与边际保证梯度均匀一步收缩的充要条件。 |
| [^296] | [Constrained Online Learning with Noisy Constraint Values](https://arxiv.org/abs/2609.06921) | 针对约束值受噪声污染的约束在线凸优化，本文证明了联合次线性保证的不可能性，并提出LEDGER算法，通过引入预算违反新度量，实现了 $O(\sqrt T/V)$ 的期望遗憾和可控的预算违反保证。 |
| [^297] | [From Synthetic Priors to Model Behavior: Structural Coverage in Tabular Foundation Models](https://arxiv.org/abs/2609.06912) | 该论文从分布层面归因视角出发，通过结构覆盖度和归一化密度量化表格基础模型的合成预训练先验对下游基准任务的支持范围，并探究这种结构支持与预测性能之间的关联。 |
| [^298] | [ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models](https://arxiv.org/abs/2609.05461) | 该论文提出ARC-Bench无泄漏审计协议，揭示冻结JEPA世界模型中“潜在距离可正确排序动作”的核心假设严重且结构性失效，且该缺陷在不同视觉骨干网络下均持续存在。 |
| [^299] | [Learning Informative Prior with Infinite-Dimensional Continuous Normalizing Flow for Bayesian Inverse Problem](https://arxiv.org/abs/2609.03343) | 该论文提出了一种基于无限维连续归一化流的新方法，通过在希尔伯特空间中引入定义良好的神经常微分方程将简单参考测度变换为编码先验信息的复杂测度，建立了无限维贝叶斯先验的适定性理论框架，并提供了先验训练方法与后验采样算法，用于求解偏微分方程的贝叶斯逆问题。 |
| [^300] | [Untangling the Mechanisms of Misleading Context in Medical Question Answering](https://arxiv.org/abs/2609.02754) | 该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。 |
| [^301] | [CRAD: Class-wise Reliability-Aware Distillation for Decentralized Heterogeneous Federated Learning](https://arxiv.org/abs/2609.00446) | 该论文提出CRAD框架，通过类别级可靠性感知的方式组合同伴教师的软预测，在去中心化联邦学习中同时解决模型架构异构与非独立同分布数据两大难题，且无需中央服务器或共享原始数据。 |
| [^302] | [Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations](https://arxiv.org/abs/2608.29434) | 该论文证明JEPA世界模型的潜空间规划可以从图像成功扩展到点云等几何观测——三种经典JEPA设计在点云上均能无坍塌地规划，其中动作敏感模型在几何变化最大的场景中表现最佳。 |
| [^303] | [The Imperfective Paradox Is Not Necessarily in Large Language Models: A Benchmark Failure Before a Model Failure](https://arxiv.org/abs/2608.25005) | 本文指出现有NLI基准因概念和评估错误规范（如体貌缩减）而失效，导致大型语言模型在“不完全性悖论”测试中的表现被误判，并提出了改进的词汇匹配基准。 |
| [^304] | [qshap: Fast Shapley Decomposition of $R^2$ for Gradient-Boosted Trees](https://arxiv.org/abs/2608.24104) | qshap是一种新工具，通过Shapley分解梯度提升树中的$R^2$值，实现特征和观测对模型性能贡献的全局量化。 |
| [^305] | [RIBOSPAN: A Long-Context RNA Foundation Model for Versatile RNA Modeling](https://arxiv.org/abs/2608.22849) | 提出了RIBOSPAN——一个16.1亿参数、原生支持最长10,240核苷酸上下文的双向RNA基础模型，通过密集双向自注意力、单核苷酸分词和注意力隔离序列打包，实现了对完整长RNA的单核苷酸分辨率建模。 |
| [^306] | [Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment](https://arxiv.org/abs/2608.11537) | 提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。 |
| [^307] | [CommitKV: Commit-Aligned KV Cache Compression for Multimodal Agents](https://arxiv.org/abs/2608.07855) | CommitKV是一种无需训练的多模态智能体KV缓存压缩方法，通过对比工具调用前后对同一缓存页面的删除影响来捕捉证据使用的变化，淘汰两个时间窗口中影响力均低的页面并暂时保护被观察激活的页面，避免因注意力转移而误删后续需要的信息。 |
| [^308] | [Parameter-Free Dynamic Regret under Heavy-Tailed Noise](https://arxiv.org/abs/2607.27073) | 该论文提出了一种无需预知噪声矩参数、Lipschitz 界、梯度尺度及比较序列路径长度的在线凸优化算法，通过重启 AdaGrad 专家与自适应熵正则化主算法的结合，在重尾噪声下实现了具有噪声路径指数 $(p-1)/p$ 的最优通用动态遗憾界。 |
| [^309] | [Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting](https://arxiv.org/abs/2607.26643) | SkillBoost通过结构化利用和先验引导探索的受约束框架，有效平衡探索与利用，从而缓解LLM代理技能自我进化中的过拟合问题。 |
| [^310] | [Posterior Variance Is a Constraint Map, Not an Error Map: Closed-Form Uncertainty for Radiative Gaussian Splatting in Sparse-View CT](https://arxiv.org/abs/2607.13682) | 本文证明辐射高斯泼溅CT重建中的后验方差图本质上是数据约束图而非误差图（其警报可信但“无异常”信号不可信），并推导出无需修改光栅化器、一次前向传播即可精确计算的截断感知闭式不确定性（成本约为采样法的1/8），同时发现该不确定性在临床关注的物体内部区域会失去对真实误差的排序能力。 |
| [^311] | [Glioma IDH Classification via CVAE: Synthetic Augmentation, Generative Classification, and MIL for Raman Biopsies](https://arxiv.org/abs/2607.10196) | 该研究利用β-CVAE同时解决拉曼活检数据集样本量小和缺乏组织标注两大难题，通过合成数据增强、生成式分类和多实例学习，将胶质瘤IDH状态二分类准确率提升至0.83。 |
| [^312] | [Variable-Length Generative Protein Design via Generalized Poisson Flow](https://arxiv.org/abs/2607.09039) | 提出广义泊松流（GPFlow），一个无需预先指定蛋白质长度即可进行生成的可变长度框架，通过学习非齐次广义泊松过程的强度函数，在结构和序列设计、基序支架及肽段协同设计等任务上实现了跨多种模态的蛋白质设计。 |
| [^313] | [A Few Teacher Steps Go a Long Way: Cost-Efficient On-Policy Data Augmentation for Agent Post-Training](https://arxiv.org/abs/2607.04574) | 该论文将智能体后训练中的同策略数据构建形式化为一个预算分配问题，系统权衡教师演示、续写长度、过滤规则与上下文覆盖等设计维度，发现少量教师步骤即可带来显著收益，从而实现高性价比的数据增强。 |
| [^314] | [Same Concept, Different Directions: Cross-Modal Feature Heterogeneity in Sparse Autoencoders](https://arxiv.org/abs/2606.29888) | 本文挑战了稀疏自编码器应用于联合嵌入空间时“语义对应特征跨模态共享相同方向”的隐含假设，首次刻画了同一概念在图像与文本模态间的特征方向异质性现象，并证明这是导致模态分裂的关键驱动因素。 |
| [^315] | [RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage](https://arxiv.org/abs/2606.27174) | 本文提出了一个名为RecallRisk-BERT的多任务框架，通过联合建模召回严重性与根本原因类别，实现了对医疗器械报告后召回记录的高效自动分诊，填补了现有研究在联合预测方面的空白。 |
| [^316] | [LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search](https://arxiv.org/abs/2606.25039) | LLM-ACES提出了一种由大语言模型引导的闭环方程发现框架，通过LLM生成算子先验来缩小符号搜索空间，并联合优化假设构建与自适应数据采集，利用候选方程间的分歧指导新数据获取，从而消除可辨识性缺口、准确恢复动力系统的控制方程。 |
| [^317] | [Compressing History into Memory: Distilling Transformers into Recurrent Transformers](https://arxiv.org/abs/2606.21562) | 本文提出一种蒸馏方法，将处理完整历史的Transformer所学会的压缩策略迁移到循环Transformer中，从而在无需存储观测历史的情况下，显著提升循环模型在长时程流式视觉与机器人任务中的性能。 |
| [^318] | [Benchmarking Open-Ended Multi-Agent Coordination in Language Agents](https://arxiv.org/abs/2606.08340) | 提出了基于 JAX 的开放式多智能体协作基准 Alem，将程序化协作任务、软专业化和通信嵌入长时程生存世界中，评估显示当前 LLM 智能体平均归一化回报仅约 6%，远未解决该基准。 |
| [^319] | [FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences](https://arxiv.org/abs/2606.03330) | 本文提出面向AI监管的“实例级指纹识别”新范式及方法FLIPS，通过分析模型生成的二进制伪随机序列中的偏差，来区分同一大语言模型在不同实例级配置（如提示词、采样设置、量化方式）下的部署实例。 |
| [^320] | [Speculative Decoding and the Curse of Multilinguality](https://arxiv.org/abs/2605.30580) | 本研究验证了推测解码在低资源语言上的效果因小模型多语言能力有限而大打折扣，且通过蒸馏提升草稿模型多语言能力的方法因任务间泛化差以及低资源语言难以构建通用数据集而不可行。 |
| [^321] | [Structured Prompt Optimization Advances Globally and Locally Interpretable Learning from Complex Text](https://arxiv.org/abs/2605.29076) | eXTC 通过结构化提示优化学习自然语言规则手册（SOP），结合推理蒸馏与强化学习，实现了兼具快速推理、局部推理轨迹与全局模块化可解释性的高性能文本分类。 |
| [^322] | [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/abs/2605.27358) | 提出MobileMoE端侧MoE语言模型系列，通过端侧MoE缩放定律确定内存与计算同时最优的架构（适度稀疏+细粒度专家+共享专家），在十亿以下激活参数规模上为端侧大语言模型建立了新的帕累托前沿。 |
| [^323] | [Oracle-Budgeted Protein Design with Stochastic Search and Iterative Imitation](https://arxiv.org/abs/2605.26690) | SILO是一种轨迹级蛋白质设计方法，结合层次化编辑策略与随机束搜索探索多样化突变轨迹，利用UCB代理集成与丙氨酸扫描式适应度评分筛选候选序列，并通过交叉熵模仿学习迭代自我改进，在严格的预言机评估预算下高效优化蛋白质序列。 |
| [^324] | [NPSolver: Neural Poisson Solver with Iterative Physics Supervision](https://arxiv.org/abs/2605.25786) | NPSolver 提出了一种无需解标签的神经泊松求解器，通过少量预条件共轭梯度（PCG）迭代步精炼自身预测作为迭代物理监督，获得了比原始 PDE 残差更稳定、尺度更合适的训练信号。 |
| [^325] | [Copy, Predict, Code the Rest: Lossless Graph Compression with Decoder-Side Neighbor Prediction](https://arxiv.org/abs/2605.21510) | 该论文提出了一种利用解码端邻居预测（复制已有邻居、预测剩余目标并对余下部分编码）的有向图无损压缩器，在基准图上实现了比WebGraph、Zuckerli和HyperFast更小的压缩文件，但代价是解码速度显著变慢。 |
| [^326] | [Flash PD-SSM: Memory-Optimized Structured Sparse State-Space Models](https://arxiv.org/abs/2605.19150) | 提出了一种新颖的状态空间模型PD-SSM，通过在每个时间步从一组可训练的结构化稀疏矩阵中离散选择一个矩阵，在保持与主流结构化SSM相当的吞吐量的同时，实现了与非结构化转移矩阵相当的表达能力。 |
| [^327] | [Seeking the Unfamiliar but Memorable: Conceptual Creativity as Meta-Learning](https://arxiv.org/abs/2605.16477) | 该论文提出将创造性形式化为“创造者-鉴赏者”元学习框架——创造者生成初看陌生但几次接触即可快速学会的刺激，以鉴赏者快速适应的提升作为奖励信号，从而在冻结扩散模型、无需额外语言条件的情况下，生成超越基础模型能力的风格变化与概念组合。 |
| [^328] | [Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty](https://arxiv.org/abs/2605.14494) | 本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。 |
| [^329] | [Fairness of Explanations in Artificial Intelligence (AI): A Unifying Framework, Axioms, and Future Direction toward Responsible AI](https://arxiv.org/abs/2605.09852) | 本文首次提出了研究AI解释公平性的统一理论框架与公理体系，揭示了模型即使输出满足公平性标准，其推理过程仍可能存在“程序性偏见”，并为迈向负责任的人工智能指明了未来方向。 |
| [^330] | [Agent Collectives Should Not Detect Their Own Imposters: A Chess Case Study](https://arxiv.org/abs/2605.09027) | 研究表明让智能体集体自行检测内部冒名顶替者会适得其反，而采用能够从少量样本快速适应新型攻击的轻量级元训练外部检测器才是更优策略。 |
| [^331] | [Quantile-Coupled Flow Matching for Distributional Reinforcement Learning](https://arxiv.org/abs/2605.08515) | 该论文提出FlowIQN方法，通过在小批次内对源样本和目标样本进行排序形成分位数耦合，修复了条件流匹配评论家与分布式贝尔曼算子之间的Wasserstein度量不匹配问题。 |
| [^332] | [Text-to-CAD Evaluation with CADTests](https://arxiv.org/abs/2605.07807) | 该论文提出了首个基于可执行测试的文本转CAD评估基准CADTestBench，通过CADTests验证生成模型是否满足几何与拓扑要求，并证明这些测试还能指导CAD生成，使简单基线超越现有方法。 |
| [^333] | [Robust stochastic first order methods in heavy-tailed noise via medoid mini-batch gradient sampling](https://arxiv.org/abs/2605.07634) | 本文提出一种新颖的梯度估计器，通过将小批量数据分割成多个数据块并选取梯度中心元作为估计值，在无需梯度裁剪等非线性操作的情况下，实现了重尾噪声（方差可能无限）下鲁棒的随机一阶优化。 |
| [^334] | [Crafting Reversible SFT Behaviors in Large Language Models](https://arxiv.org/abs/2605.06632) | 该论文提出损失约束双重下降方法，可将SFT诱发的行为构造性地压缩到稀疏子网络“载体”中，该载体对行为既充分又必要，从而在不修改权重的情况下实现对大语言模型微调行为的推理时可控与可逆操控。 |
| [^335] | [PairAlign: A Framework for Autoregressive Tokenization via Self-Alignment with Applications to Audio Tokenization](https://arxiv.org/abs/2605.06582) | PairAlign提出了一种通过自回归自对齐实现紧凑音频词元化的框架，通过让两个内容保持视图预测统一的规范锚点目标，实现了序列一致性、紧凑性和长度等特性的可控音频离散表示。 |
| [^336] | [Jacobian-Velocity Bounds for Deployment Risk Under Covariate Drift](https://arxiv.org/abs/2605.04932) | 该论文通过时域庞加莱不等式与雅可比-速度定理建立了协变量漂移下冻结预测器长期部署的风险上界，并据此提出漂移对齐切向正则化（DTR）及配套监控指标来有效控制部署风险。 |
| [^337] | [UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding](https://arxiv.org/abs/2605.04543) | 本文提出UniVer，将树状投机解码的验证统一建模为条件最优传输问题，通过前缀接受概率作为动态缩放因子主动引导草稿选择，从而联合优化多步依赖与多草稿分支之间的耦合关系。 |
| [^338] | [Sequential Strategic Classification with Multi-Stage Selective Classifiers](https://arxiv.org/abs/2605.04202) | 本文提出了一种序贯、随机、多阶段的策略性分类新模型，刻画了智能体通过改进行为与投机行为在多个阶段面对不同分类器时逐步调整策略的过程，突破了以往研究仅关注单次决策或与单一分类器重复交互的局限。 |
| [^339] | [Model-Based Reinforcement Learning with Double Oracle Efficiency in Policy Optimization and Offline Estimation](https://arxiv.org/abs/2605.00393) | 该论文通过对数障碍与对数行列式正则化技术，为表格型MDP提出了新算法，在实现最优 $\tilde{O}(\sqrt{T})$ 遗憾界的同时，将离线统计估计与规划预言机的调用次数降至对数级别，从而突破了大规模环境下强化学习的计算瓶颈。 |
| [^340] | [Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models](https://arxiv.org/abs/2604.13991) | 提出了一种提示词自适应的共形预测方法，通过提示词相关的校准提升大语言模型生成内容的事实性，在保持边际覆盖保证的同时改善条件覆盖率，并支持选择性预测以过滤不可靠输出。 |
| [^341] | [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517) | AVO用自主编码智能体取代传统演化搜索中的固定变异算子，在B200 GPU上经过7天连续自主演化，发现了性能超越cuDNN达3.5%、超越FlashAttention-4达10.5%的注意力内核优化方案。 |
| [^342] | [ProFed: Projection-Guided Personalized Federated Learning for Low-Dose CT Denoising](https://arxiv.org/abs/2603.13422) | ProFed提出了一种结合图像域特征调制与投影域监督的个性化联邦学习框架，通过解剖/协议描述符、可微分投影一致性损失和不确定性加权聚合，有效应对患者间解剖差异与扫描协议变化，实现低剂量CT去噪的协同训练。 |
| [^343] | [SoundWeaver: Compositional Warm-Starting for Text-to-Audio Diffusion Serving](https://arxiv.org/abs/2603.07865) | SoundWeaver是首个无需训练的组合式跨示例声学复用框架，通过从缓存音频片段构建提示条件化声学先验并热启动扩散过程，显著加速文本到音频扩散模型的推理。 |
| [^344] | [Frequency-Structured Hamiltonian Neural Network for Multi-Timescale Dynamics](https://arxiv.org/abs/2603.06354) | 该论文提出频率结构化哈密顿神经网络（FS-HNN），通过在时间频率尺度上组织哈密顿学习来克服频谱偏差，从而准确刻画快慢耦合的多时间尺度动力学，并在连续时间内保持能量守恒。 |
| [^345] | [Fail-Closed Alignment for Large Language Models](https://arxiv.org/abs/2602.16977) | 提出失效关闭对齐设计原则及渐进式对齐框架，通过迭代消融已学到的拒绝方向、迫使模型在独立子空间中重构安全性，将越狱攻击平均成功率降至1.6%。 |
| [^346] | [Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift](https://arxiv.org/abs/2602.14913) | 该论文针对分布偏移下共形预测覆盖率失效的问题，利用域适应工具推导出以源分类器损失和Wasserstein偏移表示的目标覆盖率下界，并提出源调优伪校准方法来缓解覆盖率退化。 |
| [^347] | [Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation](https://arxiv.org/abs/2602.14701) | 本文提出在反向传播中用随机化的无偏近似替代精确的向量-雅可比积，从理论上分析了精度与成本之间的权衡并证明了稀疏约束下最小方差的最优性，实验验证了该方法能有效降低深度学习的训练成本。 |
| [^348] | [VI-CuRL: Stabilizing Verifier-Independent Reinforcement Learning Reasoning via Confidence-Guided Variance Reduction](https://arxiv.org/abs/2602.12579) | 提出VI-CuRL框架，利用模型内在置信度构建课程并优先处理高置信度样本，有效降低梯度方差，从而解决无验证器强化学习推理中的训练崩溃问题。 |
| [^349] | [MUSE: Multi-Tenant Model Serving With Seamless Model Updates](https://arxiv.org/abs/2602.11776) | MUSE是一个多租户模型服务框架，通过将模型分数与客户端决策边界解耦，使模型更新时无需跨数百个客户重新协调校准阈值，从而实现无缝的模型更新。 |
| [^350] | [Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings](https://arxiv.org/abs/2602.09985) | 该论文提出了一种基于JEPA自监督嵌入的在线监测框架，无需异常标签即可检测汽车目标状态表示中的异常，为自动驾驶车辆的安全运行监控提供了新方案。 |
| [^351] | [A Fast and Accurate Fourier Neural Operator-Based Surrogate for Melt-Pool Prediction in Laser Processing](https://arxiv.org/abs/2602.06241) | 该论文提出LP-FNO代理模型，通过将瞬态问题在随激光移动的参考系中重构并施加时间平均，将其转化为适合算子学习的准稳态问题，从而快速精确地从工艺参数预测激光加工中的三维温度场与熔池边界，覆盖从传导焊到匙孔焊的宽工艺窗口。 |
| [^352] | [Instance-optimal high-precision shadow tomography with few-copy measurements: A metrological approach](https://arxiv.org/abs/2602.04952) | 本文给出了高精度影子断层扫描样本复杂度的实例最优刻画 $\tilde{\Theta}(\Gamma_p/\epsilon^2)$，其中 $\Gamma_p$ 由涉及逆 Fisher 信息矩阵的优化公式定义。 |
| [^353] | [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) | 该论文提出了一种名为“漂移模型”的新生成建模范式，通过漂移场在训练过程中演化前推分布并实现一步推理，在 ImageNet 256×256 上取得了一步生成最先进的性能（潜空间 FID 1.54，像素空间 FID 1.61）。 |
| [^354] | [SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration](https://arxiv.org/abs/2602.04361) | SparVAR是一种免训练的VAR模型加速框架，通过利用注意力汇聚点、跨尺度激活相似性和局部性三大特性，从稀疏决策尺度动态预测高分辨率尺度的稀疏注意力模式，在不跳过尺度、不牺牲图像质量的前提下大幅降低计算复杂度和推理延迟。 |
| [^355] | [FlowSteer: Towards Agents Designing Agentic Workflows via Reinforced Progressive Canvas Editing](https://arxiv.org/abs/2602.01664) | FlowSteer 提出了一种新范式，由单个轻量级智能体在可执行的工作流画布环境中，通过强化学习训练的渐进式原子画布编辑，端到端地自主设计供下游执行器运行的智能体工作流，并获得语法检查级别的实时执行反馈。 |
| [^356] | [On Approximate Computation of Critical Points](https://arxiv.org/abs/2601.21917) | 该论文证明了即使对系数极简单的非凸三次多项式，临界点的粗略近似计算也是计算难解的——除非P=NP，否则不存在多项式时间算法能输出近似临界点。 |
| [^357] | [Statistical Inference for Fuzzy Clustering](https://arxiv.org/abs/2601.02656) | 本文提出加权模糊c均值（WFCM）框架，通过聚类特定权重处理聚类规模不平衡问题，并构建带归一化工作密度的似然框架，首次为模糊聚类提供了原则性的统计推断方法。 |
| [^358] | [Energy-Guided Generative Modeling for Low-Energy Molecular Structure Discovery](https://arxiv.org/abs/2512.22597) | EnFlow 是首个将基于流的构象生成与显式能量势面建模相结合的能量引导生成框架，能够联合实现低能构象系综的生成与分子基态结构的识别。 |
| [^359] | [Robustness Evaluation of Machine Learning Models for Fault Classification and Localization In Power System Protection](https://arxiv.org/abs/2512.15385) | 本文提出了一个统一评估框架，利用高保真EMT仿真系统性地评估电力系统保护中机器学习故障分类与定位模型在传感器中断、采样率降低和通信丢失等数据退化场景下的鲁棒性。 |
| [^360] | [Beyond Primitive Bloat: High-Fidelity Gaussian Splatting via Coordinate-Convolution Neural Network](https://arxiv.org/abs/2512.12898) | 该论文从理论上证明离散高斯表示存在图元数量指数级增长导致的保真度瓶颈，并提出以坐标卷积神经网络（CCNN）将细节恢复建模为粗渲染之上的连续神经场，从而无需图元膨胀即可实现高保真高斯泼溅。 |
| [^361] | [Fine-Grained Caching for Diffusion Transformers with Few Calibration Conditions](https://arxiv.org/abs/2512.05134) | 提出了一种无需训练的细粒度缓存框架，利用条件共模抑制效应和少量校准条件为扩散Transformer构建固定的模块复用调度，从而大幅降低图像与视频生成的计算开销。 |
| [^362] | [Hierarchical Deep Research for Physically Grounded System-Level Materials Design](https://arxiv.org/abs/2511.18303) | 本文提出基于开源语言模型的分层多智能体系统"Deep Tree of Research"，在盲评中于47个智能体中排名第一，其材料设计方案在第一性原理模拟中超越最佳商业系统，并通过传感器实验得到物理验证。 |
| [^363] | [Nowcast3D for effective precipitation nowcasting](https://arxiv.org/abs/2511.04659) | 提出灰箱学习框架Nowcast3D，将物理结构化神经算子（三维平流、局地扩散与微物理过程）与条件扩散模型相结合，直接基于三维雷达体扫数据实现物理引导且不确定性量化的降水临近预报。 |
| [^364] | [Generalized Guarantees for Variational Inference in the Presence of Even and Elliptical Symmetry](https://arxiv.org/abs/2511.01064) | 该论文将反向KL散度变分推断在偶对称下恢复均值、椭圆对称下恢复相关矩阵的保证，从对数凹目标分布推广到更一般的单峰（含重尾）目标分布，并扩展至仅部分对称的情形。 |
| [^365] | [Repairing Reward Functions with Feedback to Mitigate Reward Hacking](https://arxiv.org/abs/2510.13036) | 该论文提出了基于偏好的奖励修复（PBRR）自动化迭代框架，通过从少量人类偏好数据中学习一个加性的、依赖于状态转移的修正项来修复人工指定的代理奖励函数，以低成本缓解奖励欺骗问题。 |
| [^366] | [Efficient Prediction of pass@k Scaling in Large Language Models](https://arxiv.org/abs/2510.05197) | 本文提出了一种稳健的估计框架，能够在远小于大规模尝试次数的采样预算下高效准确地预测大语言模型的pass@k缩放行为，从而帮助模型提供商和监管机构评估模型的能力与风险。 |
| [^367] | [Complexity-Regularized Proximal Policy Optimization](https://arxiv.org/abs/2509.20509) | 该论文提出用香农熵与非平衡度乘积构成的自调节复杂度项替代传统熵正则化来改进近端策略优化（PPO），在保持策略有益随机性的同时，当策略高度不确定时降低正则化压力，从而让学习更专注于奖励优化。 |
| [^368] | [Conditional Deep Levy Models for Exotic Derivatives: History-Aware Path Generation and P-Q Payoff Diagnostics](https://arxiv.org/abs/2509.13374) | 该论文提出基于去噪列维概率模型的条件式历史感知股指路径生成器，通过严格控制变量的匹配实验证明，列维稳定参数α=1.9相比高斯极限α=2在路径依赖衍生品收益诊断（CRPS、覆盖率、区间评分等指标）上取得显著改进。 |
| [^369] | [Adaptive Microwave Reception with Diamond Spin Sensors](https://arxiv.org/abs/2509.08171) | 本文推导了金刚石自旋微波接收机中反馈延迟的精确阈值，并据此提出了一种基于导频记录一阶圆矩方向估计的自适应相位控制器，从而在载波漂移条件下有效保持二进制相移键控符号的对准增益。 |
| [^370] | [One-Shot Clustering for Federated Learning Under Clustering-Agnostic Assumption](https://arxiv.org/abs/2509.01587) | 本文提出了一种聚类无关的一次性聚类联邦学习算法（OCFL），能够自动检测出最适合聚类的最早时机，从而实现个性化模型的训练。 |
| [^371] | [Playing Markov Games Without Observing Payoffs](https://arxiv.org/abs/2509.00179) | 本文将“不观测收益也能竞争”的结果从矩阵博弈扩展到零和对称马尔可夫博弈，证明仅凭转移动态知识和对手动作序列即可与拥有完全信息的对手竞争，并将该学习问题归约为在线学习。 |
| [^372] | [On the notion of missingness for path attribution explainability methods in medical settings: Guiding the selection of medically meaningful baselines](https://arxiv.org/abs/2508.14482) | 该论文针对医学影像归因可解释性方法，提出“语义缺失性”新概念，指出全零等标准基线在医学场景中语义上无意义，主张基线应代表疾病特征缺失的临床合理状态，并据此提出反事实引导的基线选择方法。 |
| [^373] | [Perfect Clustering in Extremely Sparse Multilayer Networks with Groupwise Subspace Structure](https://arxiv.org/abs/2507.19423) | 本文提出一个极其广义的多层网络聚类模型——各层遵循广义随机点积图模型且可按潜在子空间结构分组，并证明了即使各层网络极度稀疏、逐层分析失效时，仍能完美地恢复具有不同子空间结构的层分组。 |
| [^374] | [Text-to-Image Prompt Optimization for Unseen Rewards](https://arxiv.org/abs/2506.16853) | 提出RATTPO方法，通过对比高分与低分提示词，将优化历史中的隐式证据转化为显式搜索指导，从而实现对设计时未知奖励的文生图提示词优化。 |
| [^375] | [Local Policy Manifolds for Efficient Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2506.04022) | 提出LLE-MORL方法，通过利用策略参数空间与性能空间之间的局部关系追踪策略流形，借助局部线性外推实现高效搜索，无需大量重新训练即可快速生成覆盖多目标权衡的高质量策略集合。 |
| [^376] | [TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning](https://arxiv.org/abs/2505.16743) | TRIM提出一种行级迭代度量驱动剪枝方法，通过为层内每个输出行分配不同的稀疏比率，在高稀疏度（70-80%）下显著降低大语言模型的困惑度，例如使Qwen2.5-14B在80%稀疏度下的困惑度近乎减半。 |
| [^377] | [Impact of Data Sparsity on Machine Learning for Fault Detection in Power System Protection](https://arxiv.org/abs/2505.15560) | 提出了一个评估框架，系统性地量化了数据稀疏性对基于机器学习的电力系统故障检测与故障线路识别性能的影响。 |
| [^378] | [Adaptive Linear Networks for Irregular Multivariate Time Series Forecasting in Big Data Systems](https://arxiv.org/abs/2505.00590) | 该论文提出AiT模型，其核心ALinear模块通过无需值投影的时间条件化注意力池化，从观测或查询坐标生成样本自适应的时间权重，从而在不规则多变量时间序列的任意查询坐标上实现高效且灵活的预测。 |
| [^379] | [Virology Capabilities Test (VCT): A Multimodal Virology Q&A Benchmark](https://arxiv.org/abs/2504.16137) | 研究者提出了由博士级病毒学专家构建的322题多模态基准VCT，结果显示OpenAI的o3以43.8%的准确率超过94%的专家病毒学家，凸显了AI在病毒学领域的能力及其双用途风险。 |
| [^380] | [Causal pieces: analysing and improving spiking neural networks piece by piece](https://arxiv.org/abs/2504.14015) | 提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。 |
| [^381] | [Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos](https://arxiv.org/abs/2503.20258) | 提出E-ViM³——一种保留视频3D结构的Vision Mamba网络，结合包围全局标记和时空链式掩码策略进行自监督预训练，实现了医学超声视频的精确且数据高效的分析。 |
| [^382] | [On the Efficiency of Structured Pruning in Small Language Model Pretraining](https://arxiv.org/abs/2503.05920) | 提出IDEA Prune集成式放大-剪枝流水线，在单一余弦退火学习率调度下统一放大模型预训练、迭代结构化剪枝和恢复过程，在全成本核算下比传统分阶段剪枝方法更高效地训练出小语言模型。 |
| [^383] | [A Steerable Deep Network for Model-Free Diffusion MRI Registration](https://arxiv.org/abs/2501.04794) | 提出了一种基于SE(3)等变可操纵卷积的几何深度学习框架，实现了无需模型假设的原始扩散磁共振数据非刚性配准，且无需旋转平移数据增强。 |
| [^384] | [MMGenBench: An Automated Pipeline for Evaluating LMMs from a Text-to-Image Generation Perspective](https://arxiv.org/abs/2411.14062) | MMGenBench是一个无需预定义问题或参考描述的自动化基准框架，通过“图像描述生成→文生图重建→编码器比对”的流水线从原始图像评估大型多模态模型，对68个LMM的评估揭示了总分可能掩盖的模式特定重建质量差异。 |
| [^385] | [ProFL: Performative Robust Optimal Federated Learning](https://arxiv.org/abs/2410.18075) | 本文提出ProFL，一种在模型诱导分布漂移下无需凸性策略性风险和无噪声数据假设、可使联邦学习模型收敛到策略性最优点而非仅仅稳定点的鲁棒方法。 |
| [^386] | [Grammatical Error Correction for Low-Resource Languages: The Case of Zarma](https://arxiv.org/abs/2410.15539) | 本文首次针对低资源语言哲尔马语开展语法错误纠正研究，比较了基于规则、机器翻译和大语言模型三种方法，发现基于M2M100的机器翻译方法表现最佳，并构建了超过25万样本的数据集用于评估。 |
| [^387] | [Solving Attention Kernel Regression Problem via Pre-conditioner](https://arxiv.org/abs/2308.14304) | 本文通过定义两个问题，设计了用于注意力矩阵的代理快速算法及其回归问题求解算法。 |
| [^388] | [Fast Sketching of Polynomial Kernels of Polynomial Degree](https://arxiv.org/abs/2108.09420) | 本文提出了一种新的无意识草图方法，消除了运行时间对多项式核次数的依赖，大幅改进了神经正切核、反余弦核等慢增长核的近似运行速度。 |
| [^389] | [HindSight: A Graph-Based Vision Model Architecture For Representing Part-Whole Hierarchies](https://arxiv.org/abs/2104.03722) | 该论文提出了HindSight架构，通过将图像划分为不同层级的图像块并构建全连接图来编码图像中的部分-整体层次结构，可训练为通用视觉编码器以支持多种视觉下游任务。 |
| [^390] | [Randomized Forward Mode of Automatic Differentiation for Optimization Algorithms.](http://arxiv.org/abs/2310.14168) | 该论文介绍了一种随机前向模式自动微分优化算法，通过在神经网络的正向传递中计算损失函数的方向导数来更新参数。算法通过采样不同概率分布的随机方向，使用正向模式自动微分计算雅可比向量乘积，并提供了对其收敛速度和计算复杂性的严格分析。 |
| [^391] | [Individual Treatment Effects in Extreme Regimes.](http://arxiv.org/abs/2306.11697) | 本文提出了一种新的框架，通过测量潜在结果在存在或缺乏治疗的情况下的尾部衰减率变化，来估计极端环境下的个体治疗效果（ITE$_2$）。 |

# 详细

[^1]: OmniAct：一个策略，掌控百万具身形态

    OmniAct: One Policy, One Million Embodiments

    [https://arxiv.org/abs/2611.33158](https://arxiv.org/abs/2611.33158)

    OmniAct通过以机器人在具体场景中的运动学图（包含关节位置、运动轴及与视觉观测的连接关系）为条件，并用流匹配策略预测各节点的相对关节运动，实现了单一策略即可控制具有不同自由度和运动学配置的多种机器人具身形态。

    

    控制多种机器人的单一策略不仅需要在不同具身形态之间迁移任务知识，还必须根据每个机器人的具体能力调整其行为。现有方法通过共享的动作表示来促进这种迁移，但这些表示未必能捕捉机器人形态上的差异应如何塑造操作策略。我们提出OmniAct，这是一种跨具身形态的操作策略，它直接以特定场景中每个机器人的运动学信息为条件，生成针对该具身形态的行为。这种运动学图表示了机器人的关节位置、运动轴以及与其视觉观测之间的连接关系，使策略能够考虑其具身形态与场景中物体和表面之间的关系。一个流匹配（flow-matching）策略为每个节点预测相对关节运动，从而能够控制具有不同自由度和运动学配置的机器人。为了学习物理……（原文摘要在此处截断）

    arXiv:2611.33158v1 Announce Type: cross  Abstract: A single policy that controls diverse robots must not only transfer task knowledge across embodiments, but also adapt its behavior to each robot's capabilities. Existing approaches facilitate transfer through shared action representations, but these representations do not necessarily capture how differences in robot morphology should shape manipulation strategies. We introduce OmniAct, a cross-embodiment manipulation policy that conditions directly on each robot's kinematics in a specific scene to produce embodiment-specific behavior. This kinematic graph represents the robot's joint positions, motion axes, and connectivity to its visual observation, allowing the policy to account for how its embodiment related to objects and surfaces in the scene. A flow-matching policy predicts relative joint motions for each node, enabling control across robots with different degrees of freedom and kinematic configurations. To learn how physical str
    
[^2]: BioLoom：通过学习科学文章的结构来扩展生物医学视觉-语言模型

    BioLoom: Scaling Biomedical Vision-Language Models by Learning from the Structure of Scientific Articles

    [https://arxiv.org/abs/2611.33151](https://arxiv.org/abs/2611.33151)

    BioLoom通过挖掘科学文章内部的结构关系（如子图与同级图注之间的关联、同类内容的跨文章负样本），在对比图文预训练中引入弱正样本监督与加权负样本，从而在不增加数据规模的情况下提升生物医学视觉-语言模型的表示能力。

    

    采用对比图文预训练的生物医学视觉-语言模型（VLMs）的发展一直依赖于不断扩大且日益多样化的图文数据集的规模。除了规模化之外，进一步的进展还可以来自从现有数据中学习更丰富的表示。我们假设科学文献本身的结构提供了额外的监督来源：图、子图（面板）、图注以及伴随的文章文本编码了丰富的跨模态关系。然而，将这些内容视为相互独立的图文对忽略了它们之间的关系。为了解决这一局限，我们将这些结构关系纳入对比图文监督中。我们在保留精确图文匹配的同时，为相关配对（例如某个子图与其同级子图的图注）引入较弱的正样本监督。我们还对同一内容类型的跨文章负样本进行加权，以促使模型学习更精细的区分能力。我们还……

    arXiv:2611.33151v1 Announce Type: cross  Abstract: The development of biomedical vision-language models (VLMs) with contrastive image-text pretraining has been driven by scaling to increasingly large and diverse image-text datasets. Beyond scaling, further progress can come from learning richer representations from existing data. We hypothesize that the structure of scientific literature itself provides an additional source of supervision: figures, panels, captions, and accompanying article text encode rich relationships across modalities. Yet, treating these as independent image-text pairs overlooks their relationships. To address this limitation, we incorporate these structural relationships into contrastive image-text supervision. We preserve exact image-text matches while introducing weaker positive supervision for related pairs, such as a panel and the caption of a sibling panel. We upweight cross-article negatives of the same content type to encourage finer discrimination. We als
    
[^3]: iWAM：无需显式未来预测的世界模型高效动作预测

    iWAM: Efficient Action Prediction from World Models without Explicit Future Prediction

    [https://arxiv.org/abs/2611.33141](https://arxiv.org/abs/2611.33141)

    该论文提出 iWAM，利用流匹配模型的可逆动力学将未来反演为潜噪声向量，仅以单次视频模型评估的中间特征来条件化动作策略，从而无需显式生成未来帧即可实现高效的动作预测。

    

    世界-动作模型（World-action models, WAMs）在预训练的视频生成模型之上构建策略，其前提是：预测场景如何演化有助于决定如何行动。然而，生成未来场景对于实时控制而言代价高昂。这促使人们寻求“未来想象”的高效代理，但如何在训练此类代理的同时不牺牲动作预测质量仍不明确。一个有效的代理并不需要生成高分辨率帧；它只需提供足够的信息来区分可能的未来。由此引出了 iWAM 背后的设计原则：与其对整个未来帧进行去噪，我们能否让动作策略以单次视频模型评估所产生的中间特征为条件？我们证明答案是肯定的。利用流匹配模型的可逆动力学，我们将每个训练样本的未来反演为对应的潜在噪声向量。然而，在该潜变量处进行单次速度评估并不能……（摘要原文在此处截断）

    arXiv:2611.33141v1 Announce Type: cross  Abstract: World-action models (WAMs) build policies on pretrained video-generation models, on the premise that predicting how a scene evolves helps determine how to act. However, generating future scenes is expensive for real-time control. This motivates efficient proxies for future imagination, but how to train such proxies without sacrificing action-prediction quality remains unclear. An effective proxy need not generate a high-resolution frame; it only needs to provide enough information to distinguish between possible futures. This leads to the design principle behind iWAM: rather than denoising an entire future frame, can we condition the action policy on intermediate features from a single video-model evaluation? We show that the answer is yes. Exploiting the reversible dynamics of flow-matching models, we invert each training future into the corresponding latent noise vector. However, a single velocity evaluation at this latent does not n
    
[^4]: 伪装的偏置：微小梯度编辑终止小语言模型的思考循环，并与前沿模型的权重编辑形成互补

    A Bias in Disguise: Tiny Gradient Edits Stop a Small Language Model's Thinking Loops and Complement a Frontier Model's Weight Edit

    [https://arxiv.org/abs/2611.33133](https://arxiv.org/abs/2611.33133)

    该论文发现通过梯度选择极少量关键权重微调即可让小模型走出无休止的思考循环，其本质是为无偏置项的模型隐式添加偏置，并与前沿模型的无梯度权重编辑形成互补。

    

    Qwen3.5-0.8B在思考模式下很少完成推理：在4,096个token的预算下，500个MATH-500答案中只有1或2个能关闭其思考块（两个随机种子），准确率仅为15.4%。最近的一项演示让GPT-6 Astra在无梯度的情况下编辑了该模型的131,072个权重，并报告在32,768个token预算下MATH-500从7.9%提升至48.1%。我们提供了缺失的梯度基线，然后将两种编辑方式结合起来。仅对梯度最大的140至139,750个权重进行微调（训练数据为模型自己对MATH训练题目的正确答案），在单GPU上训练2-9分钟后，可将4k token下思考模式的MATH-500提升至39.0-45.5%；50个训练样例即已足够，而随机选择1,398个位置只能达到16.3%。梯度将编辑集中在前三层中每层最接近恒定取值的通道上。由于该模型没有偏置项，局限于这些通道的编辑相当于一个加到残差流（原文此处截断）中的常数向量。

    arXiv:2611.33133v1 Announce Type: cross  Abstract: Qwen3.5-0.8B rarely finishes reasoning in thinking mode: at a 4,096-token budget only 1 or 2 of 500 MATH-500 answers close their think block (two seeds), and accuracy is 15.4%. A recent demonstration had GPT-6 Astra edit 131,072 of the model's weights without gradients and reported MATH-500 rising from 7.9% to 48.1% at a 32,768-token budget. We provide the missing gradient baseline, then combine the two edits. Fine-tuning only the 140 to 139,750 weights with the largest gradient, on the model's own correct answers to MATH training problems, raises thinking-mode MATH-500 at 4k tokens to 39.0-45.5% after 2-9 minutes of training on one GPU; 50 training examples suffice, and 1,398 random positions give only 16.3%. The gradient concentrates the edit on the most nearly constant channel of each of the first three layers. Because the model has no bias terms, an edit confined to those channels acts as a constant vector added to the residual str
    
[^5]: 面向机器人策略的探索性中间训练

    Exploratory Midtraining for Robot Policies

    [https://arxiv.org/abs/2611.33130](https://arxiv.org/abs/2611.33130)

    提出探索性中间训练方法并构建通过对抗性遥操作采集的 β-Data 数据集，使机器人策略获得连通多样低层行为的覆盖，从而支持下游适应与在线自主改进。

    

    机器人策略主要从强调熟练行为的人类采集数据中学习，这限制了支撑自主改进所需的低层行为的覆盖范围。我们研究如何利用一种中间训练形式来构建对多样化低层行为的覆盖，这些行为能够将机器人及其环境的困难状态连接到可以继续推进的状态（即“连通性行为覆盖”）。我们还考察了这种覆盖如何支持下游适应与在线改进。为此，我们引入了 β-Data，这是一个通过对抗性遥操作采集的数据集：由一名人类“对抗者”对环境进行可逆扰动，而人类操作者则在不重置的情况下继续朝任务完成方向操作。我们用自然语言描述的局部结果对轨迹片段进行重新标注，并基于近期动作历史对预训练策略进行微调。所得策略表现出对更多低层行为的持续使用……

    arXiv:2611.33130v1 Announce Type: cross  Abstract: Robot policies predominantly learn from human-collected data emphasizing competent behavior, limiting coverage of the low-level behaviors needed to support autonomous improvement. We study how a form of midtraining can be used to build coverage over diverse low-level behaviors that connect challenging states of the robot and its environment to states from which further progress is possible ("connected behavioral coverage"). We also examine how this coverage supports downstream adaptation and online improvement. To this end, we introduce $\beta$-Data, a dataset collected via adversarial teleoperation, where a human adversary reversibly perturbs the environment while a human operator continues toward completion without resetting. We relabel trajectory segments with local outcomes in natural language and finetune a pretrained policy conditioned on recent action history. The resulting policy exhibits sustained use of more low-level behavio
    
[^6]: 面向推测解码的草稿校准：通过匹配草稿分布与目标分布加速大语言模型服务

    Draft Calibration for Speculative Decoding: Accelerating LLM Serving by Matching Draft and Target Distributions

    [https://arxiv.org/abs/2611.33111](https://arxiv.org/abs/2611.33111)

    该论文提出推测解码中的草稿校准方法，通过事后校准解决开源草稿模型相对目标模型普遍置信度不足的分布失配问题，缩小草稿分布与目标分布之间的差距，从而提高草稿 token 接受率并加速大语言模型推理服务。

    

    推测解码通过采用轻量级草稿模型提出草稿 token，并允许目标模型并行验证新 token，从而提高推理阶段的计算利用率，加速大语言模型（LLM）推理。其加速效果取决于草稿 token 在目标模型验证中的接受概率，而该概率与目标分布 $p$ 和草稿分布 $q$ 之间的差距 $\|p-q\|_1$ 直接相关。本文研究推测解码中的草稿校准方法，以提升推测解码的加速效果：我们采用事后校准方法对草稿分布进行校准，从而缓解分布差距。我们观察到，尽管当前社区开源的草稿模型可能正确学习到最优候选 token，但与目标模型相比它们普遍存在置信度不足的问题，这表明存在系统性的分布失配。为减轻这种失配对接受率的影响……（原文摘要在此处截断）

    arXiv:2611.33111v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model (LLM) inference by employing a lightweight draft model to propose draft tokens and allowing the target model to verify the new tokens in parallel to improve inference-time compute utilization. The speedup relies on the acceptance probability of draft tokens in target verification, which is directly related to the gap $\|p-q\|_1$ between the target distribution $p$ and the draft distribution $q$. In this paper, we study draft calibration of speculative decoding to improve the speedup of speculative decoding, where we employ post-hoc methods to calibrate the draft distribution to mitigate the distribution gap. We observe that though current community open-source draft models may learn the top candidate token correctly, they are generally underconfident compared to the target model, suggesting a consistent distribution mismatch. To mitigate the effect of this mismatch on the acceptance
    
[^7]: 强化学习可以从其自身的采样轨迹中恢复

    Reinforcement Learning Can Be Recovered From Its Own Rollouts

    [https://arxiv.org/abs/2611.33102](https://arxiv.org/abs/2611.33102)

    该论文发现，仅通过对存档的RL采样轨迹按其生成顺序进行最大似然训练、且无需区分响应的对错，就能从同一基础模型完全恢复出在线强化学习的性能，表明RL所获得的知识实质上蕴含在采样数据本身之中。

    

    广义而言，强化学习（RL）后训练通过在模型生成的数据上进行训练，将计算资源转化为模型能力。通常，采样轨迹在每次更新后都会被丢弃，RL过程中获得的知识仅存在于模型权重之中。在这项工作中，我们研究了这些知识有多少被保留在采样轨迹本身之中。一个自然的初步尝试是在存档的采样轨迹（而非新生成的轨迹）上重放原始的RL算法，但这会导致灾难性崩溃。我们为这一现象提供了一个简单的解释，并由此提出了一种补救方法：对存档的采样轨迹进行最大似然训练。令人惊讶的是，从同一个基础模型出发，按照采样轨迹生成的顺序进行重放，并且不区分正确与错误的响应，就足以恢复在线RL的性能。事实上，每个采样轨迹批次只需一次最大似然更新，就能接近（原文此处截断）

    arXiv:2611.33102v1 Announce Type: new  Abstract: Broadly speaking, reinforcement learning (RL) post-training converts compute into model capabilities by training on model-generated data. Typically, the rollouts are discarded after each update, and the knowledge acquired during RL resides in the model weights. In this work, we investigate how much of this knowledge is retained in the rollouts themselves. A natural first attempt is to replay the original RL algorithm on archived rather than freshly generated rollouts, but this leads to catastrophic collapse. We provide a simple explanation for this phenomenon, which in turn suggests a remedy: maximum likelihood on the archived rollouts. Surprisingly, starting from the same base model and replaying the rollouts in the order in which they were generated, without any distinction between correct and incorrect responses, is sufficient to recover the performance of online RL. In fact, a single maximum likelihood update per rollout batch closel
    
[^8]: GREMLIN：通过交互网络生成随机环境用于元学习

    GREMLIN: Generating Random Environments for Meta-Learning with Interacting Networks

    [https://arxiv.org/abs/2611.33101](https://arxiv.org/abs/2611.33101)

    该研究提出 GREMLIN 方法，用随机参数化的神经网络环境替代人工设计游戏为元学习提供训练经验，实验证明生成的环境经验虽不及 DQN，但能在所有测试游戏上有效改进学习规则。

    

    随机参数化的神经网络环境能否为学习强化学习算法提供有用的经验？我们使用 DiscoRL 的规则发现框架来研究这一问题，用十六个精选的神经环境替代人工设计的训练游戏。每个游戏将受限的向量动力学与基于随机策略的奖励以及独立的结局奖励相结合。在经过 2.248 亿步智能体-环境交互后，我们冻结了教师模型并训练了全新的智能体。DQN 在大多数游戏（无论是生成的还是人工设计的）上的得分超过了 GREMLIN。然而，GREMLIN 在全部 26 个游戏上都优于未经训练的教师模型，这表明生成的经验能够改进学习规则。该方法在更长时间的元训练下能否产生具有竞争力的规则仍不清楚。

    arXiv:2611.33101v1 Announce Type: new  Abstract: Can randomly parameterized neural environments supply useful experience for learning a reinforcement learning algorithm? We investigate this question using DiscoRL's rule-discovery framework, replacing human-designed training games with sixteen selected neural environments. Each game combines bounded vector dynamics with a random strategy-based reward and an independent outcome reward. After 224.8 million agent-environment steps, we froze the teacher and trained fresh agents. DQN outscored GREMLIN on most games, both generated and human-designed. GREMLIN nevertheless outperformed the untrained teacher on all 26 games, showing that generated experience can improve a learning rule. It remains unclear whether this approach can yield a competitive rule with longer meta-training.
    
[^9]: 基于雷达的日常生活活动数据集

    The Radar-based Activities of Daily Living (R-ADL) Dataset

    [https://arxiv.org/abs/2611.33088](https://arxiv.org/abs/2611.33088)

    该论文推出了R-ADL数据集，提供超过八小时、涵盖八种活动和五个居家位置的标注雷达数据，弥补了现有雷达活动识别数据集在规模、房间配置和活动层次上的不足。

    

    监测日常生活活动（ADLs）有助于保障居家生活者的安全与福祉，尤其是老年人。雷达传感技术因能够在保护家庭隐私的同时克服其他传感模式的技术局限，在活动识别领域受到越来越多的关注。然而，基于雷达的活动识别仍受限于大规模数据集的稀缺。我们推出了基于雷达的日常生活活动数据集（R-ADL），以弥补现有工作中发现的不足：1）通过提供超过八小时的标注记录来扩大可用数据规模，其中一个人执行八种不同的活动；2）通过在两室两卫公寓的五个不同位置进行录制，克服了房间配置的限制；3）超越原始动作层次，聚焦于高层次的日常生活活动。除数据外……

    arXiv:2611.33088v1 Announce Type: cross  Abstract: Monitoring Activities of Daily Living (ADLs) supports safety and well-being for people living at home, especially older adults, and radar sensing is gaining increasing attention in the field of activity recognition for its ability to ensure privacy in the home while also surpassing technical limitations of other sensing modalities. However, radar-based activity recognition remains constrained by the limited availability of large-scale datasets. We introduce the Radar-based Activities of Daily Living (R-ADL) dataset to addresses the gaps identified in existing work: 1) it expands the scale of available data by providing over eight hours of annotated recordings in which a person performs eight distinct activities; 2) it overcomes room configuration limitations by recording from five different locations in a two-bedroom, two-bathroom apartment; and 3) it extends beyond primitive actions to focus on high-level ADLs. In addition to the data
    
[^10]: 移位器量化：面向原生整数计算的紧凑表示

    Shifter Quantization: Compact Representations for Native Integer Computation

    [https://arxiv.org/abs/2611.33080](https://arxiv.org/abs/2611.33080)

    提出移位器量化（SQ）方法，通过追加和移位共享比特重构原生宽度整数操作数，使GPU整数矩阵运算能覆盖完整点积，在2比特精度下实现llama.cpp预填充吞吐量2.71-3.77倍的提升并降低KL散度。

    

    分组量化被用于大语言模型量化中，以在低位宽下提高精度，但对权重进行分组使得标准整数MMA（矩阵乘累加）无法跨越完整的点积进行运算，从而阻碍了高效的GPU推理。我们开发了一种新的量化方法——移位器量化（SQ），它通过追加和移位共享比特来重构原生宽度（原生整数宽度）的整数操作数。为了进一步提升效率，我们开发了SQ的一个特殊变体——移位整数（SHI），以使用INT4内核。我们表明，SQ和SHI格式在质量上（以投影误差和KL散度衡量）以及在推理速度上（在解码和预填充阶段测量）均优于标准的分组量化。对于像2比特量化这样的低精度场景，SQ2和SHI2在Q2_K检查点上实现了llama.cpp 4,096 token预填充吞吐量的2.71-3.77倍，并将WikiText-2 KL散度降低了6.6-7.8%。对于像4比特量化这样的中等精度场景，SQ4实现了1.43-2.47倍的预填充吞吐量。

    arXiv:2611.33080v1 Announce Type: new  Abstract: Groupwise quantization is used in LLM quantization to improve accuracy at low bit widths, but grouping weights keeps standard integer MMA from spanning the full dot product for efficient GPU inference. We develop a new quantization method, Shifter Quantization (SQ), that reconstructs native-width integer operands by appending and shifting shared bits. To further extend the efficiency, we develop a special variant of SQ, Shifter Integer (SHI), to use INT4 kernels. We show that SQ and SHI formats perform better than standard groupwise quantization in quality, as measured by projection error and KL divergence, and in inference speed, as measured during decoding and prefill. For low precision like 2-bit quantization, SQ2 and SHI2 achieve 2.71-3.77x the 4,096-token prefill throughput of llama.cpp on a Q2_K checkpoint and reduce WikiText-2 KL by 6.6-7.8%. For mid precision like 4-bit quantization, SQ4 achieves 1.43-2.47x the prefill throughput
    
[^11]: 带噪声二分类器的纠错输出码的可靠性函数

    The Reliability Function of Error-Correcting Output Codes with Noisy Binary Classifiers

    [https://arxiv.org/abs/2611.33068](https://arxiv.org/abs/2611.33068)

    该论文将带高斯噪声二分类器的 ECOC 解码建模为高斯信道检验，推导出超额错误率的上下界，证明较大的码字汉明距离是降低超额错误率的充分必要条件，且该错误率下界只与最大两个后验概率和噪声水平有关而与类别数无关。

    

    纠错输出编码（ECOC）通过二分类器来构建多类分类器。我们研究了当每个二分类器输出的后验估计其对数几率（log-odds）受到高斯噪声破坏时，ECOC 相对于贝叶斯（MAP）分类器的超额错误率。通过将解码问题重新表述为高斯信道上的假设检验，我们基于联合界和 Chernoff 界推导出上界，并基于最近竞争类别和 Mills 比率推导出下界。这些界表明，较大的汉明距离对于降低超额错误率既是必要的也是充分的。当两个最可能类别的码字之间距离有界时，超额错误率将始终有界地远离零，该界常数取决于两个最大的后验概率和噪声水平，而与类别数 $M$ 无关；对于等距码，该距离即为 $d_{\min}$。反之，超额错误率至多是某个指数衰减因子的 $M-1$ 倍。

    arXiv:2611.33068v1 Announce Type: cross  Abstract: Error-correcting output coding (ECOC) builds a multi-class classifier from binary classifiers. We study the excess error rate of ECOC over the Bayes (MAP) classifier when each binary classifier outputs a posterior estimate whose log-odds is corrupted by Gaussian noise. Recasting decoding as a test over a Gaussian channel, we derive an upper bound from the union and Chernoff bounds and a lower bound from the nearest competing category and the Mills ratio. The bounds show that a large Hamming distance is both necessary and sufficient for reducing the excess error rate. The excess error rate stays bounded away from zero, by a constant that depends on the two largest posteriors and the noise level but not on the number of categories $M$, whenever the distance between the code words of the two most probable categories is bounded; for equidistant codes this distance is $d_{\min}$. Conversely, it is at most $M-1$ times a factor that is expone
    
[^12]: 审计评判者：基于人类发现的偏见在LLM评判系统中的发现、量化与缓解

    Auditing the Judge: Human-Derived Bias Discovery, Quantification, and Mitigation in LLM Judges

    [https://arxiv.org/abs/2611.33065](https://arxiv.org/abs/2611.33065)

    该论文提出HUB-J集成框架，利用人类与LLM判断不一致的案例自动发现可解释的偏见因子，并通过受控输入修改量化偏见强度，实现LLM评判系统中偏见的发现、量化与缓解。

    

    虽然大语言模型越来越多地被部署为评估和训练流程中的自动化评估器，但其判断往往受到与人类偏好相冲突的系统性偏见的影响。尽管先前的研究已经识别出各种偏见并提出了检测和缓解方法，但对利用人类偏好来指导偏见识别与缓解的关注却较少。此外，这些方法严重依赖预先发现的偏见列表，忽视了偏见强度，并且依赖成本高昂的干预措施。在本工作中，我们提出了HUB-J，一个基于人类衍生偏见因子的集成框架，用于检测、量化和缓解LLM作为评判者系统中的偏见。我们的方法利用人类与LLM判断不一致的案例自动发现可解释的偏见因子，并利用一致的案例通过受控输入修改及其产生的偏移来量化偏见强度。

    arXiv:2611.33065v1 Announce Type: new  Abstract: While Large Language Models (LLMs) are increasingly deployed as automated evaluators in evaluation and training pipelines, their judgements are often affected by systematic biases that conflict with human preferences. Although prior work has identified various biases and proposed methods for detecting and mitigating them, less attention has been paid to using human preferences to inform bias identification and mitigation. Moreover, these approaches rely heavily on pre-discovered bias lists, overlook bias strength, and depend on costly interventions. In this work, we propose HUB-J, an integrated framework based on human-derived bias factors that detects, quantifies, and mitigates biases in LLM-as-a-Judge systems. Our approach leverages human-LLM judgement disagreement cases to automatically discover interpretable bias factors, and utilizes agreement cases to quantify bias strength through controlled input modifications and resulting shift
    
[^13]: 基于 U 统计量的大语言模型推理

    U-Statistics-Powered LLM Reasoning

    [https://arxiv.org/abs/2611.33064](https://arxiv.org/abs/2611.33064)

    该论文利用 GRPO 策略梯度的 U 统计量结构，将组大小选择转化为方差比估计这一经典统计问题，并提出考虑分母估计不确定性的悲观比率估计量，从而以系统且稳定的方式确定这一关键超参数。

    

    自 DeepSeek 提出组相对策略优化（GRPO）以来，该方法已被广泛用于训练大型推理模型和智能体，并催生了众多变体。然而，组大小作为这类 GRPO 型算法中的一个关键超参数，目前仍在很大程度上依靠反复试错来选择。本文利用 GRPO 策略梯度的 U 统计量结构，将组大小的选择重新表述为一个经典的统计问题——方差比估计。当分母较小时，该问题尤其具有挑战性，导致朴素的代入估计量不稳定。为应对这一挑战，我们提出了一种考虑分母估计不确定性的悲观比率估计量，并刻画了该估计量在三种样本量情形下的统计性质。实证方面，我们在来自 Reasoning Gym 的合成任务和真实世界基准上采用了基于 jackknife 的悲观规则（原文在此处截断）。

    arXiv:2611.33064v1 Announce Type: new  Abstract: Since its introduction by DeepSeek, group relative policy optimization (GRPO) has been widely adopted for training large reasoning models and agents, spurring a proliferation of its variants. Yet the group size, a crucial hyperparameter in these GRPO-type algorithms, remains largely chosen by trial and error. Exploiting the U-statistic structure of GRPO's policy gradient, we recast the selection of the group size as a classical statistical problem of variance ratio estimation. This problem is particularly challenging when the denominator of the ratio is small, rendering naive plug-in estimators unstable. To address this challenge, we propose a pessimistic ratio estimator that accounts for the denominator's estimation uncertainty. We characterise the statistical properties of this estimator across three sample-size regimes. Empirically, using %a jackknife-based pessimistic rule on synthetic tasks from Reasoning Gym and real-world benchmar
    
[^14]: 理解零阶优化中的裁剪机制

    Understanding Clipping In Zeroth-Order Optimization

    [https://arxiv.org/abs/2611.33062](https://arxiv.org/abs/2611.33062)

    本文系统研究了逐样本裁剪为何能提升零阶优化性能这一现象，证明了最优学习率与裁剪阈值成反比（即 ηc 为常数），并且裁剪会使零阶优化收敛到不同的解。

    

    零阶优化已成为微调大语言模型（LLM）时替代随机梯度下降（SGD）的一种节省资源的方法，这得益于损失景观固有的低维特性。它依赖于梯度的两点估计，只需访问函数值而无需计算梯度，从而绕过了资源消耗巨大的反向传播过程。最近，人们在以非常大的隐私参数 $\epsilon$（对应于几乎没有隐私保护）运行差分隐私版本的零阶优化时，偶然发现逐样本裁剪可以提升训练后模型的性能。在本文中，我们系统地研究了这一现象，并证明：(i) 学习率 $\eta$ 的最优选择与裁剪阈值 $c$ 成反比，满足 $\eta c =$ 常数；(ii) 裁剪会使零阶优化收敛到不同的解……

    arXiv:2611.33062v1 Announce Type: new  Abstract: Zeroth-order optimization has emerged as a resource-efficient alternative to Stochastic Gradient Descent (SGD) when fine-tuning LLMs, thanks to the inherent low dimensionality of the loss landscape. It relies on a two-point estimate of the gradient that only accesses the function value and not the gradient, bypassing the resource-heavy backpropagation. Recently, it was accidentally discovered that per-sample clipping can improve the performance of the trained model, while running a differentially private version of zeroth-order optimization \cite{zhang2024dpzero} with a very large privacy parameter $\epsilon$ (corresponding to very little privacy). In this paper, we systematically investigate this phenomenon and demonstrate that ($i$) the optimal choice of learning rate $\eta$ is inversely proportional to the clipping threshold $c$, satisfying $\eta c =$ constant; ($ii$) clipping drives the zeroth-order optimization to a different soluti
    
[^15]: 局部指数并不足够：多臂老虎机算法中的同步性与全局信息

    Local Indices Are Not Enough: Synchrony and Global Information in Bandit Algorithms

    [https://arxiv.org/abs/2611.33056](https://arxiv.org/abs/2611.33056)

    本文证明了仅依赖臂自身历史和局部随机性的异步指数型多臂老虎机算法无法在所有实例上达到 O(log T) 的最优遗憾值，其遗憾值只能处于亚多项式但超多对数的区间，从而揭示了全局时间信息与同步指数更新对于最优性能的必要性。

    

    诸如 KL-UCB 和 Thompson Sampling 等基于指数的标准随机多臂老虎机算法能够在时间范围 T 内实现最优的 O(log T) 遗憾值。前者利用全局时间步来构建每个臂的指数，而后者在每个时间步对每个臂的信念分布进行采样。然而，全局信息和同步指数更新对于实现最优遗憾值是否必要，此前尚未被考察。在本文中，我们研究了异步局部指数算法的根本局限——这类算法仅在拉取某个臂时才重新计算该臂的指数，且仅使用该臂自身的奖励历史和一个局部随机性预算。我们证明了这类算法无法在所有多臂老虎机实例上都实现 O(log T) 的遗憾值。随后，我们分析了由异步更新产生的函数方程，并表明可实现的遗憾值处于亚多项式、超多对数的区间。我们提出了一种异步局部指数策略，其遗憾值（摘要在此处截断）

    arXiv:2611.33056v1 Announce Type: new  Abstract: Standard index-based stochastic bandit algorithms such as KL-UCB and Thompson Sampling achieve optimal $O(\log T)$ regret over a horizon $T$. The former uses the global timestep to construct each arm's index, while the latter samples each arm's belief distribution at every timestep. The necessity of global information and synchronous index updates for achieving optimal regret has not been examined yet. In this paper, we investigate the fundamental limits of asynchronous local index algorithms---those that recalculate an arm's index only upon pulling it, using solely that arm's own reward history and a local randomness budget. We prove that no such algorithm can achieve $O(\log T)$ regret across all bandit instances. We then analyse the functional equations arising from asynchronous updates, and show that the achievable regret lies in a sub-polynomial, super-polylogarithmic regime. We propose an asynchronous local index policy whose regre
    
[^16]: 潜在合并器：超网络生成LoRA的潜空间组合

    Latent Merger: Latent-Space Composition of Hypernetwork-Generated LoRAs

    [https://arxiv.org/abs/2611.33048](https://arxiv.org/abs/2611.33048)

    该论文发现超网络生成的LoRA在几何特性上与直接微调的LoRA截然不同（谱更集中、跨文档对齐更强），据此提出在潜空间中直接组合中间潜变量再解码的LoRA合并方法，以解决流式场景下逐文档存储LoRA不可持续的问题。

    

    近期研究提出了超网络方法，能够在单次前向传播中将文档转换为低秩参数更新，从而实现向冻结的大语言模型快速注入新知识。然而，在流式场景下为每个文档存储独立的LoRA适配器可能难以持续。本文以两个代表性系统Doc-to-LoRA和SHINE为例，研究组合超网络生成的LoRA这一问题。我们证明超网络生成的LoRA具有与直接微调的LoRA截然不同的几何特性：与基于下一个词预测和问答微调的适配器（其分布弥散且在不同文档间近乎正交）相比，超网络生成的LoRA表现出更强的谱集中性和更强的跨文档对齐性。基于这一观察，我们提出了潜空间LoRA合并方法：先组合超网络的中间潜变量，再通过冻结的LoRA生成头对合并后的潜变量进行解码。我们证明潜空间合并方法……

    arXiv:2611.33048v1 Announce Type: cross  Abstract: Recent work proposes hypernetworks that convert a document into a low-rank parameter update (LoRA) in a single forward pass, enabling fast injection of new knowledge into a frozen LLM. However, storing separate LoRA adapters for each document can be unsustainable in streaming settings. In this paper, we study the problem of composing hypernetwork-generated LoRAs using two representative systems, Doc-to-LoRA and SHINE. We show that hypernetwork-generated LoRAs have a geometry sharply different from directly fine-tuned LoRAs. They exhibit greater spectral concentration and stronger cross-document alignment than next-token prediction and QA fine-tuned adapters, which are diffuse and nearly orthogonal across documents. Guided by this observation, we propose latent-space LoRA merging. We compose the intermediate hypernetwork latents and then decode the merged latent through the frozen LoRA-generation head. We show that latent-space merging 
    
[^17]: 《ECOC乘积解码何时具有MAP一致性？配对结构与“一对多”编码》

    When Is Product Decoding of ECOC MAP-Consistent? Pair Structure and the n-vs-Rest Codes

    [https://arxiv.org/abs/2611.33044](https://arxiv.org/abs/2611.33044)

    本文给出了ECOC乘积解码器与贝叶斯最优（MAP）类别保持一致的充要条件——即各类别对之间取值不同的列须满足特定的配对匹配结构，并据此刻画了满足该条件的n-vs-Rest编码。

    

    误差校正输出码（ECOC）通过求和解码器（对概率求和）或乘积解码器（对对数似然求和）来组合二元分类器。在无噪声设定下，即二元分类器输出真实后验概率时，我们研究哪些编码矩阵能使每种解码器始终与贝叶斯最优（MAP）类别保持一致。对于求和解码器，答案已为人所知：编码必须是等距的（James and Hastie, 1998）。我们则回答了乘积解码器的这一问题。两个类别通过它们取值不同的列的投票来进行比较：求和解码器依据这些投票的样本均值做决策，而乘积解码器则采用一个影响函数为artanh的M-估计器；每个投票等于MAP间隔加上一个仅由其他类别决定的偏差。我们的主要定理表明，当且仅当对于每一个类别对，两个类别取值不同的列与所得到的列相匹配时，乘积解码器才具有MAP一致性。

    arXiv:2611.33044v1 Announce Type: new  Abstract: Error-Correcting Output Codes (ECOC) combine binary classifiers by the sum decoder, which totals probabilities, or the product decoder, which totals log-likelihoods. In the noiseless setting, where the binary classifiers output the true posteriors, we ask on which code matrices each decoder always agrees with the Bayes-optimal (MAP) class. For the sum decoder the answer is known: the code must be equidistant (James and Hastie, 1998). We answer the question for the product decoder. Two classes are compared through the votes of the columns on which they differ: the sum decoder decides by their sample mean, the product decoder by an M-estimator with influence function artanh, and each vote is the MAP margin plus a bias determined by the other classes alone. Our main theorem states that the product decoder is MAP-consistent if and only if, for every class pair, the columns on which the two classes differ are matched with the columns obtained
    
[^18]: 迈向基于大语言模型的推荐系统中不确定性量化的现实评估

    Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems

    [https://arxiv.org/abs/2611.33036](https://arxiv.org/abs/2611.33036)

    该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。

    

    大语言模型（LLM）为推荐系统提供了灵活的接口。然而，其给出的排序结果需要可靠的实例级不确定性监控。在这种场景下，比较不确定性量化（UQ）方法变得具有挑战性，因为LLM排序器缺乏标准的物品分类头。其打分可能依赖于选项标签、提示词、生成过程以及可用的用户和物品证据。现有方法从不同的来源提取不确定性，并将其应用于推荐过程的不同阶段。一些方法评估已生成的排序结果，而另一些方法则利用不确定性来改变排序本身，这使得直接比较变得困难。本文引入了一个固定排序基准，在共享的输出契约下，将八种异构的UQ信号作为记录级监控器进行评估。我们在四个推荐数据集和两个LLM骨干网络上，在共享的…（原文在此处截断）

    arXiv:2611.33036v1 Announce Type: cross  Abstract: Large language models (LLMs) provide a flexible interface for recommendation systems. However, their issued rankings require reliable instance-level uncertainty monitoring. In this scenario, comparing uncertainty quantification (UQ) methods becomes challenging because LLM rankers lack a standard item-classification head. Their scores may depend on option labels, prompts, generation procedures, and the available user and item evidence. Existing methods derive uncertainty from different sources and apply it at different stages of the recommendation process. Some methods evaluate an issued ranking, whereas others use uncertainty to change the ranking itself, making direct comparison difficult. Here we introduce a fixed-ranking benchmark which evaluates eight heterogeneous UQ signals as record-level monitors under a shared output contract. We evaluate these signals across four recommendation datasets and two LLM backbones under a shared fi
    
[^19]: 前沿大语言模型能否通过无工具的权重空间编程修复量化网络？

    Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?

    [https://arxiv.org/abs/2611.33031](https://arxiv.org/abs/2611.33031)

    该研究证明前沿大语言模型无需任何工具、数据或推理能力，仅凭权重空间编程即可将4比特量化CNN的准确率从60.4%恢复至77.95%，接近原始浮点模型的81.4%水平。

    

    我们将Claude家族的四个大语言模型（LLM）以文本形式提供了具有2万参数的CIFAR-10卷积神经网络的完整权重，并要求它们在没有工具、数据或运行推理能力的情况下生成一个4比特网络。标准的舍入到最近（RTN）4比特量化会将测试准确率从81.4%降低到60.4%。我们设计了四种评估协议，它们在提供给LLM的输入和任务描述上有所不同，并分析它们的响应。在前两种协议中，LLM被提供了浮点网络和4比特RTN网络，并被告知准确率显著下降。在后两种协议中，它们只收到浮点网络和量化该网络的任务，这用于测试它们是自己主动应用优化，还是退回到普通的RTN。我们的分析表明，前沿LLM能够恢复大部分损失的准确率。在每层给出一个答案的情况下，Opus 5.5通过平衡舍入误差将4比特网络的准确率提升至77.95%。

    arXiv:2611.33031v1 Announce Type: cross  Abstract: We give four large language models (LLMs) of the Claude family the complete weights of a 20k-parameter CIFAR-10 CNN as text and ask them to produce a 4-bit network without tools, data or the ability to run inference. Standard round-to-nearest (RTN) quantization to 4 bit lowers the test accuracy from 81.4% to 60.4%. We design four evaluation protocols that differ in the input and task description given to the LLMs and analyze their responses. In the first two protocols the LLMs are given the float and the 4-bit RTN network and are told that the accuracy has dropped significantly. In the last two they receive only the float network and the task to quantize it, which tests whether they apply an optimization on their own or fall back to plain RTN. Our analysis shows that a frontier LLM can recover most of the lost accuracy. Given one answer per layer, Opus 5.5 raises the accuracy of the 4-bit network to 77.95% by balancing the rounding err
    
[^20]: t-谱域中超图信号的基于分数的扩散模型

    Score-Based Diffusion for Hypergraph Signals in the t-Spectral Domain

    [https://arxiv.org/abs/2611.33026](https://arxiv.org/abs/2611.33026)

    该论文提出了首个在t-积超图信号处理谱域中构建的、面向固定超图信号的基于分数生成模型，通过逐模态频率偏移修复了张量热半群的不适定问题，实现了稳定的大规模采样与缺失节点恢复。

    

    我们提出了首个针对固定超图上信号的基于分数的生成模型，该模型在t-积超图信号处理（t-HGSP）的谱域中进行构建。朴素张量热半群是不适定的，因为经过回文对称化处理后，拉普拉斯张量的谱切片不再半正定。通过逐模态的频率偏移可以修复这一问题，由此得到一个前向Ornstein-Uhlenbeck扩散过程，该过程在每个（特征模态，切片）对上解耦，具有闭式边缘分布、精确的分数目标以及严格各向同性的平稳分布。一个轻量级的切片条件去噪器在由此产生的实数谱网格上进行训练。采样过程将模型在数据活跃模态上的分数与其他位置的闭式分数相结合，事实证明这种组合对于大规模下的稳定性至关重要。一个闭式余弦形式的观测算子将缺失节点恢复转化为具有稳定后验采样的线性高斯引导。在多达500个节点的超图上

    arXiv:2611.33026v1 Announce Type: new  Abstract: We introduce the first score-based generative model for signals on a fixed hypergraph, formulated in the spectral domain of t-product hypergraph signal processing (t-HGSP). The naive tensor heat semigroup is ill-posed, since after palindromic symmetrization the spectral slices of the Laplacian tensor are no longer positive semidefinite. A per-mode frequency shift repairs it, yielding a forward Ornstein-Uhlenbeck diffusion that decouples per (eigenmode, slice) pair with closed-form marginals, exact score targets, and an exactly isotropic stationary law. A lightweight slice-conditioned denoiser is trained on the resulting real spectral grid. Sampling combines its score on data-active modes with the closed-form score elsewhere, which proves essential for stability at scale. A closed cosine-form observation operator turns missing-node recovery into linear-Gaussian guidance with stabilized posterior sampling. On hypergraphs of up to 500 nodes
    
[^21]: 未知动作后上下文分布下的最优臂识别

    Best Arm Identification with Unknown Post-Action Context Distribution

    [https://arxiv.org/abs/2611.33024](https://arxiv.org/abs/2611.33024)

    该论文研究了上下文分布未知的动作后上下文最优臂识别问题，推导了依赖实例的样本复杂度下界，并提出了一种基于新型集中不等式、能够渐进匹配该下界的 Track-and-Stop 算法。

    

    我们研究了固定置信度设置下具有动作后上下文的最优臂识别（BAI）问题。在选择一个动作之后，学习器会观察到一个动作后上下文以及一个奖励，而该奖励的分布仅取决于此上下文。引入这种中间信息可以大幅降低识别最优臂所需的样本复杂度。虽然 Shahverdikondori 等人（2026a）分析了动作对应的上下文分布已知的情况，但我们解决了更具挑战性且更具实际意义的情况，即该分布是未知的。我们推导了一个依赖实例的样本复杂度下界，并提出了一种 Track-and-Stop 算法，在获得优化预言机的情况下，该算法能够渐进地匹配该下界。我们的停止规则采用了一种新的集中不等式，能够联合控制上下文分布和奖励均值中的不确定性，从而消除了分别使用集中界所带来的两倍因子损失。

    arXiv:2611.33024v1 Announce Type: new  Abstract: We study best arm identification (BAI) with post-action context in the fixed-confidence setting. After choosing an action, the learner observes a post-action context together with a reward whose distribution depends only on this context. Incorporating such intermediate information can substantially reduce the sample complexity of identifying the best arm. While Shahverdikondori et al. (2026a) analyzed the case where the distribution of contexts given actions is known, we address the more challenging and practically relevant case where this distribution is unknown. We derive an instance-dependent sample complexity lower bound and propose a Track-and-Stop algorithm that matches it asymptotically with access to an optimization oracle. Our stopping rule uses a new concentration inequality that jointly controls uncertainty in the context distributions and reward means, removing the factor of two incurred by separate concentration bounds. Impl
    
[^22]: 共形预测器条件有效性的PAC式界

    PAC-style Bounds on the Conditional Validity of Conformal Predictors

    [https://arxiv.org/abs/2611.33023](https://arxiv.org/abs/2611.33023)

    本文在不假设数据分布的前提下，基于马尔可夫和坎泰利不等式为任意共形预测器的条件覆盖率推导出PAC型概率下界，并提出将条件覆盖率方差作为条件有效性的一种可扩展且有理论支撑的经验度量。

    

    共形预测集在属性空间上保证具有边际有效性，但局部覆盖率可能偏离用户定义的目标 $1-\alpha$。对于一般的数据生成分布，使用有限的校准数据构造条件有效的预测集是不可能的。我们在不对数据分布做任何假设的情况下，推导出任意共形预测器的输入条件覆盖率的PAC型概率下界。这些下界基于马尔可夫不等式和坎泰利（Cantelli）不等式，并同时适用于校准边际和校准条件两种设置。一个更紧的、依赖于模型的坎泰利界取决于条件覆盖率的方差，并且可以通过完全非渐近的Hoeffding论证从数据中进行估计。这些结果表明，条件覆盖率方差可以作为条件有效性的一种可扩展且有理论依据的经验度量。

    arXiv:2611.33023v1 Announce Type: new  Abstract: Conformal prediction sets are guaranteed to be marginally valid over the attribute space, but local coverage can diverge from the user-defined target $1-\alpha$. Constructing conditionally valid prediction sets with finite calibration data is impossible for a general data-generating distribution. We derive PAC-type probabilistic lower bounds on the input-conditional coverage of any conformal predictor, without assumptions on the data distribution. They are based on the Markov and Cantelli inequalities and adapt to both calibration-marginal and calibration-conditional setups. A tighter, model-dependent Cantelli bound depends on the variance of the conditional coverage and is estimable from data using a fully non-asymptotic Hoeffding argument. The results suggest conditional-coverage variance as a scalable and theoretically motivated empirical measure of conditional validity.
    
[^23]: 训练即推理：状态空间模型中用于学习视觉令牌压缩的概率卷积

    Train What You Run: Probabilistic Convolutions for Learning Token Compaction in State Space Models

    [https://arxiv.org/abs/2611.33009](https://arxiv.org/abs/2611.33009)

    本文提出 ProbConv——一种关于令牌保留概率可微的概率卷积算子，可计算独立保留决策下的精确期望卷积，使状态空间模型（如 Mamba）的训练与推理时的令牌压缩保持一致，且压缩越强其优势越明显。

    

    视觉令牌压缩通过仅将选定的图像块或视频令牌传递到后续层，从而降低推理成本。在采用局部卷积的序列模型中，每移除一个令牌都会改变哪些令牌互为邻居。因此，学习保留哪些令牌需要一种能够考虑这种新邻居关系的训练信号。我们提出了 ProbConv，它在独立保留决策的条件下，计算保留位置在激活之前的精确期望卷积。该算子关于保留概率是可微的。当采用二值决策时，ProbConv 与步进门控可以复现压缩后的 Mamba 模块。在图像、视觉—语言以及视频骨干网络上的实验表明，ProbConv 在强压缩情形下优势最为明显：随着移除的令牌越来越多，它相对于同等训练设置的对照方法的优势也不断增大。在 Cobra 模型上，在相同的推理预算下，它还优于已发表的视觉—语言剪枝方法的适配版本。ProbConv 相对于同等的零掩码训练也有提升……

    arXiv:2611.33009v1 Announce Type: cross  Abstract: Visual token compaction reduces inference cost by passing only selected image patches or video tokens to later layers. In sequence models with local convolutions, each removal changes which tokens are neighbors. Learning which tokens to keep therefore requires a training signal that accounts for the new neighborhoods. We introduce ProbConv, which computes the exact expected convolution before activation at a kept position under independent keep decisions. The operator is differentiable in the keep probabilities. With binary decisions, ProbConv and step gating reproduce the compacted Mamba block. Across image, vision--language and video backbones, ProbConv is strongest under strong compaction: its advantage over matched training controls grows as more tokens are removed. In Cobra, it also outperforms adaptations of published vision--language pruning methods at matched inference budgets. ProbConv also improves over matched zero-mask trai
    
[^24]: LEGOSpec：一个用于推测性草稿设计与训练的可组合高效框架

    LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training

    [https://arxiv.org/abs/2611.33004](https://arxiv.org/abs/2611.33004)

    LEGOSpec提出了一个可组合的推测性草稿设计与训练框架，将草稿模型分解为上下文、骨干、提议和目标四个组件，并通过系统化的对照实验揭示了不同工作负载下各组件的作用、相互作用及最优组合规律。

    

    推测性解码使用轻量级草稿模型来加速大语言模型（LLM）的生成。现有方法在不同的训练配置下评估完整的方案，导致各组件的效果、相互作用以及组合收益不明确。因此，我们提出了LEGOSpec，一个用于推测性草稿设计与训练的可组合框架。LEGOSpec将草稿模型分解为上下文、骨干、提议和目标四个组件，它们共享统一的训练基础设施。借助LEGOSpec，我们系统地改变各个组件，并在匹配的训练条件下评估选定的组合，以隔离它们各自的效果与相互作用。这些实验揭示了三个发现：（1）不同的工作负载偏好不同的骨干网络，这取决于上下文长度和内存预算，其中块内依赖建模对长上下文尤为重要。（2）提议设计必须在接受率与延迟之间取得平衡。（3）训练目标……（摘要截断）

    arXiv:2611.33004v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model (LLM) generation using a lightweight draft. Existing methods evaluate complete recipes under different training configurations, leaving component effects, interactions, and composition benefits unclear. Therefore, we present LEGOSpec, a composable framework for speculative draft design and training. LEGOSpec decomposes draft models into Context, Backbone, Proposal, and Objective components that share a unified training infrastructure. Using LEGOSpec, we systematically vary components and evaluate selected combinations under matched training conditions to isolate their effects and interactions. These experiments reveal three findings: (1) Different workloads favor different backbones, depending on context length and memory budget, with within-block dependency modeling particularly important for long contexts. (2) Proposal design must balance acceptance and latency. (3) The training o
    
[^25]: 论基础模型的潜能与实现

    On Potentiality and Actuality Of Foundation Models

    [https://arxiv.org/abs/2611.32991](https://arxiv.org/abs/2611.32991)

    基础模型本身已蕴含完成特定任务所需的能力，适配并非添加新能力，而只需通过恒定的、与输入无关的向量移位将模型“实现”即可。

    

    基础模型通过大规模预训练构建，通常随后进行后训练，然后再被适配到特定任务和行为上。但是，这种适配是否真正增加了新能力，还是基础模型本身早已具备这些能力？我们将不同适配方式所引起的更新分解为一个在所有输入间共享的分量和一个随输入变化的残差分量。我们发现，共享分量足以产生大部分适配增益，并且可以直接学习得到：一个在每层学习一个恒定向量（对所有输入均相同）的移位适配器，就能复现LoRA和全量微调的行为。我们将这一现象称为基础模型的潜能原则：基础模型本身已经包含了实现特定任务专业能力所需的容量。要适配一个新任务，模型只需被“实现”，即通过一个恒定的、与输入无关的移位被推向某个方向。我们展示了这一原则在……方面的意义。

    arXiv:2611.32991v1 Announce Type: cross  Abstract: Foundation models are built through large-scale pretraining, often followed by post-training, and are then adapted to specific tasks and behaviors. But, does the adaptation add truly new capabilities, or the base model already holds it? We decompose the update induced by different adaptations into a component shared across all inputs and a residual that varies with the input. We find that the shared component suffices for most of the adaptation gain and can be learned directly: a shift adapter that learns a constant vector per layer, identical for every input, reproduces the behavior of LoRA and full fine-tuning. We call this the potentiality principle of foundation models: the base model already contains the capacity needed for task-specific expertise. To adapt to a new task, the model just needs to be actualized, i.e., pushed in a certain direction via a constant, input-independent shift. We show the implications of this principle on
    
[^26]: AVQ-Attention-2：具有精确局部块和线性成本训练的自适应向量量化注意力

    AVQ-Attention-2: Adaptive Vector-Quantized Attention with Exact Local Blocks and Linear-Cost Training

    [https://arxiv.org/abs/2611.32989](https://arxiv.org/abs/2611.32989)

    AVQ-Attention-2 通过将精确局部注意力与关注聚类键均值的自适应向量量化注意力相结合，实现了线性成本的训练和推理，无需微调即可直接插入预训练 Transformer，在几乎无损的情况下运行速度快于 FlashAttention 且质量可与精确稠密注意力相媲美。

    

    注意力机制的成本随token数量呈 O(N²) 增长。向量量化注意力通过对 N 个键进行聚类、转而关注 M 个码字，将成本降至 O(NM)；其自适应形式 AVQ-attention 能够将码本分辨率集中在查询注意力质量实际所在的位置。我们提出了 AVQ-Attention-2，这是一种更优的设计，它将自适应向量量化注意力与精确局部注意力相结合，并关注每个聚类的键均值而非码字。我们展示了三项改进：在相当的成本下持续获得更好的质量；训练成本也降至 O(NM) 而非 O(N²)；近似精度足够高，该层可以直接插入预训练 Transformer 而无需微调。在各种序列长度下，其结果不仅能与其他近似注意力方法竞争，还能与精确稠密注意力相媲美，在几乎没有质量损失的情况下运行速度超过 FlashAttention。

    arXiv:2611.32989v1 Announce Type: new  Abstract: Attention costs $\mathcal{O}(N^2)$ in the number of tokens. Vector-quantized attention reduces this to $\mathcal{O}(NM)$ by clustering the $N$ keys, attending to the $M$ codewords instead, and its adaptive form, AVQ-attention, concentrates codebook resolution where a query's attention mass actually falls. We present AVQ-Attention-2, a better design that merges it with exact local attention and attends to the mean of each cluster's keys rather than to a codeword. We show three improvements: consistently better quality at comparable cost; training that now also costs $\mathcal{O}(NM)$ rather than $\mathcal{O}(N^2)$; and an approximation accurate enough that the layer can be dropped into a pretrained transformer without fine-tuning. Across a variety of sequence lengths the result is competitive not merely with other approximate attention but with exact dense attention, running faster than FlashAttention at little to no loss in quality.
    
[^27]: 文件大小作为视听深度伪造数据中潜在的数据集偏差

    File Size as a Potential Dataset Bias in Audio-Visual Deepfake Data

    [https://arxiv.org/abs/2611.32979](https://arxiv.org/abs/2611.32979)

    该研究揭示了两个主流视听深度伪造数据集中存在文件大小偏差——仅凭文件大小和视频时长等文件级属性即可获得很高的检测AUC，说明深度伪造检测基准可能在类别标签与文件属性之间引入了非预期的关联。

    

    深度伪造检测基准通常旨在评估模型能否区分真实与被篡改的视听内容。然而，基准构建流程可能会在类别标签与简单的文件级属性之间引入非预期的关联。在这项工作中，我们在两个广泛使用的视听深度伪造数据集中识别出一种潜在的文件大小偏差。在1M-Deepfakes Detection Challenge数据集中，仅基于文件大小得出的评分在验证集上即可达到70.13%的AUC。当将文件大小与视频时长结合考虑时，简单的文件级方法在1M数据集上的AUC超过90%，在FakeAVCeleb数据集上超过85%。尽管在DFDC数据集中未观察到相同的模式，但结果表明这种关联依赖于具体数据集，而非深度伪造视频的固有属性。我们讨论了几种可能的解释，包括编码方式、压缩、源数据以及篡改流程等方面的差异。

    arXiv:2611.32979v1 Announce Type: new  Abstract: Deepfake detection benchmarks are commonly designed to evaluate whether models can distinguish authentic from manipulated audiovisual content. However, benchmark construction pipelines may introduce unintended associations between class labels and simple file-level properties. In this work, we identify a potential file-size bias in two widely used audio-visual deepfake datasets. In the 1M-Deepfakes Detection Challenge dataset, a score derived from file size alone achieves 70.13% AUC on the validation set. When file size is considered together with video duration, simple file-level methods exceed 90% AUC on 1M and 85% AUC on FakeAVCeleb. Although the same pattern is not observed in DFDC, the results suggest that the observed association is dataset-dependent rather than an inherent property of deepfake videos. We discuss several possible explanations, including differences in encoding, compression, source data, manipulation procedures, and
    
[^28]: 激发线性注意力的召回能力

    Eliciting the Recall Capacity of Linear Attention

    [https://arxiv.org/abs/2611.32969](https://arxiv.org/abs/2611.32969)

    该研究揭示了线性注意力模型的召回容量随状态规模缩放的规律，并提出了上下文句子复述和上下文敏感损失重加权两种训练干预方法，有效弥补了线性模型在关联召回能力上与softmax注意力的差距。

    

    在语言建模中，线性注意力模型在关联召回（AR）方面相对于softmax注意力表现出不成比例的显著差距。我们通过自然语言评估、合成召回任务以及针对性的训练干预来研究这一差距。在合成实验设置中，我们量化了状态规模在决定可从记忆中召回多少个项目方面所起的作用，并在所测试的各线性注意力架构中发现了相似的实证缩放规律。对这一关系的验证表明，即使是大规模下的单层模型也能够学会跨长距离检索数千个关联。然而，标准的语言模型训练并不能可靠地教会线性模型从上下文中复制信息，而同等规模的softmax模型却能够学会做到这一点。有鉴于此，我们通过上下文内的句子复述任务和对上下文敏感的损失重加权两种方法，在训练过程中加强了召回监督。这两种方法都提升了召回密集型基准任务的性能。

    arXiv:2611.32969v1 Announce Type: new  Abstract: Linear attention models exhibit a disproportionate deficit relative to softmax attention in associative recall (AR) in language modeling. We investigate this gap through natural language evaluations, synthetic recall tasks, and targeted training interventions. Using a synthetic setting, we quantify the role of state-size in how many items can be recalled from memory and find a similar empirical scaling across the tested linear-attention architectures. Testing this relationship reveals that even a single layer at scale can learn to retrieve thousands of associations across long distances. Yet, standard language model training does not reliably teach a linear model to copy from context, whereas a comparable softmax model learns to do so. Considering this, we strengthen recall supervision during training through in-context sentence repetition and context-sensitive loss reweighting. Both methods improve recall-intensive benchmark performance
    
[^29]: 多项式时间学习随机自动机：每个单词仅一个标签

    Polynomial-Time Learning of Random Automata from One Label per Word

    [https://arxiv.org/abs/2611.32959](https://arxiv.org/abs/2611.32959)

    本文提出一个多项式时间算法，仅凭每个观测单词上的单个接受/拒绝标签，即可在绝大多数随机转移表上高概率地学习二进制确定性有限自动机，并将预测错误率控制在 ε 以内，样本量与运行时间均为参数的多项式。

    

    我们研究一类二进制确定性有限自动机，其 2n 个转移是从其 n 个状态中独立、均匀地随机选取的。学习者观察相互独立的均匀随机长度为 m 的单词，每个单词仅由一个比特标记，表示该单词是否被接受。对于每个固定的 β>0，我们给出了一个多项式时间算法，该算法在概率为 o(n^−β) 的转移表集合之外均能正常工作。在其余每个转移表上，对于任意的起始状态和接受状态集合，预测器在新鲜的均匀随机长度为 m 的单词上的错误率至多为 ε，且该保证在训练样本和学习者随机选择下以至少 1−γ 的概率成立。其样本量和运行时间是 n、m、1/ε 和 log(2/γ) 的多项式，多项式次数仅取决于 β。尽管每个单词只揭示一个输出，但通过比较两个相邻历史长度处的标签矩，我们能够估计从某状态出发的两条续写的有符号输出的乘积的期望……（摘要原文在此处截断）

    arXiv:2611.32959v1 Announce Type: new  Abstract: We study a binary deterministic finite automaton whose $2n$ transitions are chosen independently and uniformly from its $n$ states. The learner observes independent uniform words of length $m$, each labeled with a single bit indicating acceptance. For every fixed $\beta>0$ we give a polynomial-time algorithm that works outside a set of transition tables of probability $o(n^{-\beta})$. On each remaining table, for any start state and accepting set, the predictor has error at most $\epsilon$ on fresh uniform words of length $m$ with probability at least $1-\gamma$ over the training sample and the learner's random choices. Its sample size and running time are polynomial in $n$, $m$, $1/\epsilon$ and $\log(2/\gamma)$, with degree depending only on $\beta$. Although each word reveals only one output, comparing label moments at two consecutive history lengths lets us estimate the expected product of the signed outputs of two continuations from
    
[^30]: 自适应纵向数据下的在线V学习

    Online V-Learning with Adaptive Longitudinal Data

    [https://arxiv.org/abs/2611.32947](https://arxiv.org/abs/2611.32947)

    本文针对移动健康中策略在线批量更新所导致的自适应收集、患者轨迹相依的数据，在可变确定性批次大小设定下建立了V学习的统计理论并证明了一致性。

    

    用于估计最优动态治疗方案的强化学习方法正日益应用于移动健康领域，在这类场景中治疗决策需要持续不断地做出。Luckett等人（2020）提出了V学习，该方法可从离线轨迹中估计最优策略。在实际应用中，移动健康策略是在线更新的。批量更新——即在每个新数据块到达后重新估计策略——是介于离线学习与每次观测后更新之间的一种可行折中方案。然而，由于每个批次的处理策略是基于早期批次中跨患者汇集的数据估计得到的，由此产生的数据是自适应收集的，且患者轨迹之间相互依赖，这一情形是现有离线理论未能涵盖的。我们在批次大小可变且为确定性的设定下，为V学习建立了统计理论。对于每个固定批次，随着患者数量的增长，我们证明了估计量的一致性（原文摘要在此处截断）。

    arXiv:2611.32947v1 Announce Type: cross  Abstract: Reinforcement learning methods for estimating optimal dynamic treatment regimes are increasingly applied in mobile health (mHealth), where treatment decisions are made continually. Luckett et al. (2020) proposed V-learning, which estimates an optimal policy from offline trajectories. In practice, mHealth policies are updated online. Batch updating, in which the policy is re-estimated after each block of new data, is a tractable middle ground between offline learning and updating after every observation. However, because each batch's treatment policy is estimated from data pooled across patients in earlier batches, the resulting data are adaptively collected and patient trajectories are dependent, a setting that offline theory does not address. We develop statistical theory for V-learning in this setting with varying, deterministic batch sizes. For each fixed batch, as the number of patients grows, we establish uniform consistency of th
    
[^31]: PHM-ensemble：基于集成密度估计的可扩展且稳定的概率多分辨率聚类

    PHM-ensemble: Scalable and Stable Probabilistic Multiresolution Clustering via Ensemble Density Estimation

    [https://arxiv.org/abs/2611.32945](https://arxiv.org/abs/2611.32945)

    该论文提出PHM-ensemble框架，通过集成密度估计汇集多个混合模型并对成分进行层次化合并，实现了同时兼顾保真度、可解释性、稳定性和可扩展性的概率多分辨率聚类。

    

    聚类是跨科学领域探索性数据分析的基础工具。我们认为聚类方法应从四个维度进行评估：对数据底层结构的保真度、所恢复群组的可解释性、在数据扰动和算法随机性下的稳定性，以及对现代科学数据集的可扩展性。然而，现有方法很少能同时满足这四项标准。为弥补这一空白，我们提出了PHM-ensemble，这是一个用于多分辨率聚类的通用概率框架，它通过汇集来自集成拟合的混合模型来估计数据的底层密度，随后通过对混合成分的层次化合并构建聚类层次结构。在模拟数据集和MNIST基准测试中，PHM-ensemble与真实标签的一致性达到或超过广泛使用的聚类算法，在多次运行中产生更稳定的结果，并能揭示复杂的底层……（原文摘要在此处截断）

    arXiv:2611.32945v1 Announce Type: cross  Abstract: Clustering is a foundational tool for exploratory data analysis across the sciences. We argue that clustering methods should be evaluated along four dimensions: fidelity to the data's underlying structure, interpretability of the recovered groups, stability under data perturbations and algorithmic randomness, and scalability to modern scientific datasets. Yet existing methods rarely satisfy all four criteria. To address this gap, we introduce PHM-ensemble, a general probabilistic framework for multiresolution clustering that estimates the underlying data density by pooling mixture models from an ensemble of fits and subsequently constructs a clustering hierarchy through hierarchical merging of the mixture components. On simulated datasets and the MNIST benchmark, PHM-ensemble matches or outperforms widely used clustering algorithms in agreement with ground truth, produces more stable results across runs, and reveals complex underlying 
    
[^32]: 乳腺数据上的量子核：集中效应、替代模型以及量子优势被“制造”出来的两种方式

    Quantum kernels on breast data: concentration, surrogates, and two ways an advantage gets manufactured

    [https://arxiv.org/abs/2611.32942](https://arxiv.org/abs/2611.32942)

    本文通过对乳腺癌数据上的量子核优势宣称进行系统性基准检验（涵盖样本效率扫描、闭式去量化替代模型对比和选择膨胀测量），揭示所谓的量子优势可能是由于不完整的对照比较、选择偏差乃至数据泄露而被“制造”出来的假象。

    

    量子核方法在乳腺数据上的优势宣称通常只是一个孤立的数字：一个量子方案、一个对照方法、一个测试集划分。我们探究了当完整的对照方法集合和选择历史都可见时，这个数字究竟衡量了什么。我们报告了三个有效的基准测试，全部采用8量子比特保真度核的精确经典态矢量模拟，推理路径中不含任何量子成分：（1）在威斯康星乳腺癌诊断数据集上，以十个不同训练集规模进行的样本效率扫描；（2）在1,200张M-SYNTH模拟乳腺X线照片上进行的原始数据与处理后数据的对比实验，配有匹配的闭式去量化替代模型、相同输入下的五种经典方法以及十一种核函数家族；（3）在BreastMNIST数据集上进行的30个随机种子的选择膨胀测量。CMMD D1临床数据集的实验作为结果性实验被撤回，因为其固定的数据划分会通过患者序列顺序泄露标签——该缺陷已在另一项研究中被报告；该实验仅作为案例示例被保留。所有三个有效的实验方案都是n...（摘要在此处被截断）

    arXiv:2611.32942v1 Announce Type: cross  Abstract: Quantum-kernel advantage claims on breast data are usually a single number: one quantum arm, one comparator, one test split. We asked what that number measures once the full comparator set and selection history are visible. We report three active benchmarks, all exact classical statevector simulations of 8-qubit fidelity kernels with no quantum component in the inference path: a sample-efficiency sweep on Wisconsin Diagnostic Breast Cancer at ten training-set sizes; raw-versus-processed arms on 1,200 M-SYNTH simulated mammograms with a matched closed-form dequantisation surrogate, five classical methods at the same input and eleven kernel families; and a 30-seed selection-inflation measurement on BreastMNIST. A CMMD D1 clinical arm is withdrawn as a result arm because its frozen splits leak the label through patient serial order, a defect reported in a separate study; it is retained only as a worked example. All three active arms are n
    
[^33]: 基于令牌分配的高效视频检索增强生成

    Efficient Video Retrieval Augmented Generation via Token Allocation

    [https://arxiv.org/abs/2611.32938](https://arxiv.org/abs/2611.32938)

    Lembas是一个即插即用的视频RAG框架，将帧与字幕的令牌分配建模为收益递减的优化问题，用具有可证明收敛保证的迭代注水算法求解，并能在多轮对话中零开销地复用令牌。

    

    高效的长视频理解需要在尽量减少视觉语言模型（VLM）所处理令牌数量的同时识别相关证据。现有的视频检索增强生成（RAG）方法依赖于帧选择和令牌压缩，但忽略了每增加一个令牌所带来的价值。我们提出了Lembas，一个即插即用的框架，它将视频RAG形式化为在共享预算下对帧和字幕进行令牌分配的优化问题。我们的关键洞察是：令牌的价值在每个条目内部呈递减趋势，并且随着相似证据获得更多令牌而进一步降低。Lembas在单一目标函数中同时捕捉这两种效应，并通过一种具有可证明收敛保证的高效迭代注水算法进行求解。对于多轮对话，Lembas复用此前轮次中已处理的令牌，并借助VLM的注意力以零额外开销的方式精细化相关性评估。我们的评估表明，Lembas提升了准确率……

    arXiv:2611.32938v1 Announce Type: cross  Abstract: Efficient long-video understanding requires identifying relevant evidence while minimizing the tokens processed by vision-language models (VLMs). Existing video retrieval-augmented generation (RAG) methods rely on frame selection and token compression, but overlook how much each additional token is worth. We introduce Lembas, a plug-and-play framework that formulates video RAG as a token allocation problem over frames and subtitles under a shared budget. Our key insight is that the value of a token diminishes within each item and decreases further as similar evidence receives more tokens. Lembas captures both effects in a single objective and solves it with an efficient iterative water-filling algorithm that has a provable convergence guarantee. For multi-turn conversations, Lembas reuses the tokens processed in earlier turns and refines relevance with the VLM's attention at no extra cost. Our evaluations show that Lembas improves accu
    
[^34]: 基于傅里叶神经算子的理论引导量子误差缓解方法用于Shor求阶算法

    Theory-Informed Quantum Error Mitigation for Shor Order Finding with Fourier Neural Operators

    [https://arxiv.org/abs/2611.32923](https://arxiv.org/abs/2611.32923)

    该论文提出了一种理论引导的量子误差缓解方法，通过建模电路误差对输出分布峰值的偏移、展宽和减弱效应，并利用带可学习增益的傅里叶神经算子，从噪声测量分布中有效恢复Shor求阶算法的正确阶估计。

    

    量子噪声会扭曲电路测量结果的分布，但通常仍保留足够的信息来推断算法的正确结果。为了从Shor求阶算法的噪声分布中恢复这些潜在信息，我们提出了一种理论引导的量子误差缓解方法。我们对电路误差如何使输出分布中的峰发生偏移、展宽或减弱进行建模。这些模型既指导合成训练数据的生成，也指导傅里叶神经算子的设计，该算子具有可学习的增益，能够衰减或放大所保留的较高傅里叶模态。我们通过将每个候选阶的理想分布与三次电路执行中测得的分布之间的相关性进行平均，来对每个候选阶进行评分。网络调整这些评分以指导阶的估计，并确定理想分布归一化混合的权重。在针对一般模幂运算的1351个模拟测试用例上（由电路……留出测试）……

    arXiv:2611.32923v1 Announce Type: cross  Abstract: Quantum noise distorts the distribution of circuit measurement outcomes, often leaving enough information to infer the algorithm's correct result. To recover this latent information from the noisy distribution in Shor order finding, we introduce a theory-informed quantum error mitigation method. We model how circuit errors shift, broaden, or weaken peaks in the output distribution. These models guide both the generation of synthetic training data and the design of a Fourier neural operator with a learned gain that attenuates or amplifies higher retained Fourier modes. We score each candidate order by averaging the correlations between its ideal distribution and the distributions measured in three circuit executions. The network adjusts these scores to guide the order estimate and determine the weights of a normalized mixture of the ideal distributions. On 1,351 simulated test cases for general modular exponentiation, held out by circui
    
[^35]: 通过文件头的标签泄露：公开乳腺X线摄影数据集中的几何与批次混淆因素，以及一种预训练准入门槛

    Label leakage through file headers: geometry and batch confounds in public mammography datasets, and a pre-training acceptance gate

    [https://arxiv.org/abs/2611.32899](https://arxiv.org/abs/2611.32899)

    该论文提出一套仅利用图像几何、纹理或标识符等非诊断信息的预训练准入控制检查，以AUC 0.60为停止阈值审计六个公开乳腺X线摄影数据集，发现多个数据集存在严重标签泄露（如KAU-BCMD仅凭几何信息即能以AUC 1.0预测BI-RADS分级），并据此构建了模型预训练前的数据集验收门槛。

    

    能够从非诊断信息中恢复出诊断标签的数据集，会产生高、可复现但无信息量的模型性能。我们对六个公开乳腺X线摄影数据集应用了仅依赖图像几何、全局纹理、子集标识符或单一协变量的预训练控制检查，并以AUC 0.60作为停止阈值。在KAU-BCMD中，仅凭几何信息即以AUC 1.0000预测出本地副本中的BI-RADS分级，在公开Kaggle存档中平均为0.979；调整图像尺寸后，仅凭纹理得到0.847。在CMMD中，子集前缀以0.7753的AUC预测恶性；受限子集通过了几何（0.5000）和纹理（0.5810）检查，却在事后添加的标识符顺序控制检查中失败（测试AUC 0.9926）。CBIS-DDSM在几何检查上超过停止阈值（20次划分的合并均值0.654），该判定统计量为事后选定；其官方的肿块与钙化划分则未超过。M-SYNTH、INbreast和部分VinDr-Mammo（摘要原文被截断）

    arXiv:2611.32899v1 Announce Type: cross  Abstract: A dataset whose diagnostic label can be recovered from non-diagnostic information yields high, reproducible and uninformative model performance. We applied pre-training controls that see only image geometry, global texture, a sub-collection identifier or one covariate to six public mammography datasets, against a stop threshold of AUC 0.60. In KAU-BCMD, geometry alone predicted the BI-RADS class at AUC 1.0000 in a local copy and at a mean of 0.979 in a public Kaggle archive; after resizing, texture alone gave 0.847. In CMMD, the sub-collection prefix predicted malignancy at 0.7753; the restricted sub-collection passed geometry (0.5000) and texture (0.5810) but failed an identifier-order control added post hoc (test AUC 0.9926). CBIS-DDSM exceeded the stop on geometry (pooled mean 0.654 over 20 splits), a verdict statistic chosen post hoc; its official mass and calcification splits did not. M-SYNTH, INbreast and a partial VinDr-Mammo co
    
[^36]: 共享信息决定多视图植入匹配

    Shared information governs multi-view planted matching

    [https://arxiv.org/abs/2611.32892](https://arxiv.org/abs/2611.32892)

    该论文证明对同一 n 个对象的 K 个打乱视图能否全部匹配由对象观测间的共享信息精确决定——共享信息超过 (1+ε)log n 时可匹配至 o(n) 误差、低于 (1−ε)log n 时不可能，且整体可匹配性由视图最弱二分划分的互信息阈值 2log n 控制，弱共同信号下两个阈值分别为 4/K 与 4/(K−1)。

    

    我们研究同一 n 个对象的 K 个被打乱顺序的视图的行何时可以相互匹配，其中每个对象在每个视图中通过具有已知协方差的高斯特征向量被观测一次。对于每个固定的 K，若单个对象的 K 次观测之间的共享信息 J 超过 (1+ε)log n，则所有视图可以在 o(n) 误差内全部匹配；若低于 (1−ε)log n 则不能；即使任意两个视图自身无法匹配，上述情形仍可能发生。在阈值以下，若存在某个同时包含这两个视图的视图集合、其共享信息高于 log n，则这两个视图仍然可以匹配；否则，任何估计器——即使读取全部视图——在这两个视图之间的重叠都会趋于消失。能否匹配每一行由视图的最弱二分划分所控制：在弱相关情形下，跨越该划分的互信息阈值为 2log n。对于通过独立噪声观测到的弱共同信号，两个阈值分别为 4/K 和 4/(K−1)（原文此处截断）。

    arXiv:2611.32892v1 Announce Type: new  Abstract: We study when the rows of $K$ shuffled views of the same $n$ objects can be matched, where each object is observed once in every view through a Gaussian feature vector with known covariance. For every fixed $K$, all views can be matched up to $o(n)$ errors if the shared information $J$ of one object's $K$ observations exceeds $(1+\epsilon)\log n$, and not if it is below $(1-\epsilon)\log n$; this can hold although no two views can be matched on their own. Below the threshold, two views can still be matched if some set of views containing both has shared information above $\log n$; otherwise every estimator, even one that reads all views, has vanishing overlap between them. Matching every row is governed by the weakest bipartition of the views: with weak correlations the threshold is $2\log n$ for the mutual information across it. For a weak common signal observed through independent noise, the two thresholds are $4/K$ and $4/(K-1)$ in un
    
[^37]: MXFP4用于大语言模型推理：互补的训练后量化技术及其相互作用

    MXFP4 for LLM Inference: Complementary Post-Training Quantization Techniques and Their Interactions

    [https://arxiv.org/abs/2611.32873](https://arxiv.org/abs/2611.32873)

    该研究发现有效的微缩放格式量化方案依赖于具体格式：MXFP4和NVFP4虽都支持精确的4位推理，但MXFP4显著受益于旋转和纠错PTQ技术，而这些技术未必适用于NVFP4。

    

    低精度表示的最新进展为大语言模型量化引入了多种微缩放数值格式。通过在16或32个元素组成的组之间共享缩放因子，这些格式能够实现激进的低精度量化，同时精度损失极小，特别是与现代训练后量化（PTQ）算法结合使用时。它们的精度取决于数值格式如何与旋转、PTQ和缩放因子选择相互作用。我们研究了PTQ配置在六个大语言模型和三种微缩放格式（MXFP6、MXFP4和NVFP4）上的可组合性，系统地变化Hadamard旋转粒度、PTQ算法和缩放因子选择策略。我们的核心发现是：有效的量化方案依赖于具体的数值格式——MXFP4和NVFP4都能支持精确的4位推理，但对一种格式有益的技术不一定能迁移到另一种格式。MXFP4从旋转和纠错PTQ中获益显著，而NVF（摘要在此处截断）

    arXiv:2611.32873v1 Announce Type: cross  Abstract: Recent advances in low-precision representation have introduced several microscaling numerical formats for LLM quantization. By sharing a scale factor across groups of 16 or 32 elements, these formats enable aggressive low-precision quantization with minimal accuracy degradation, particularly when combined with modern PTQ algorithms. Their accuracy depends on how the numerical format interacts with rotation, PTQ, and scale selection. We study the composability of PTQ configurations across six LLMs and three microscaling formats: MXFP6, MXFP4, and NVFP4, systematically varying Hadamard rotation granularity, PTQ algorithm, and scale-selection strategy. Our central finding is that effective quantization recipes are format dependent: both MXFP4 and NVFP4 support accurate 4-bit inference, but techniques that benefit one do not necessarily transfer to the other. MXFP4 benefits substantially from rotation and error-correcting PTQ, whereas NVF
    
[^38]: 权重尺度再平衡：可迁移的Transformer训练配方

    Weight Scale Rebalancing: Transferable Training Recipes for Transformers

    [https://arxiv.org/abs/2611.32866](https://arxiv.org/abs/2611.32866)

    提出UDVR重参数化方法，在Transformer成对权重矩阵间插入辅助对角矩阵并周期性再平衡尺度，得到一条仅需矩阵维度的免调优超参数规则，可跨模型规模、架构、优化器和数据分布直接迁移。

    

    训练Transformer需要细致的超参数调优，而这对大型模型来说成本越来越高。我们提出了UDV-Rebalance（UDVR），这是一种重参数化方法，在Transformer架构中自然成对出现的权重矩阵(U,V)之间插入一个辅助对角矩阵D。UDVR周期性地重新平衡这些因子，使U保持单位范数的列、V保持单位范数的行，而相应的尺度由D吸收。通过再平衡，UDVR的单一超参数设置在模型规模、架构和数据分布之间均保持有效。这产生了一条免调优规则，仅需利用每个权重矩阵自身的维度即可对其进行适配。我们在CIFAR-10和CIFAR-100上训练的ViT模型上对该规则拟合了一次，即使更换激活函数或优化器，该规则仍然有效。同样的规则可迁移到DomainNet中的新图像域、ImageNet上的ViT、Swin和MaxViT，以及语言（原文在此处截断）。

    arXiv:2611.32866v1 Announce Type: new  Abstract: Training Transformers requires careful hyperparameter tuning, which becomes increasingly costly for large models. We introduce UDV-Rebalance (UDVR), a reparameterization that inserts an auxiliary diagonal matrix $D$ between naturally paired weight matrices $(U,V)$ appearing throughout Transformer architectures. UDVR periodically rebalances the factors to maintain unit-norm columns in $U$ and unit-norm rows in $V$, with $D$ absorbing the corresponding scales. With rebalancing, a single hyperparameter setting of UDVR remains effective across model scales, architectures, and data distributions. This yields a tuning-free rule that adapts to each weight matrix using only its dimensions. We fit this rule once on ViT models trained on CIFAR-10 and CIFAR-100, and it remains valid when the activation function or the optimizer is changed. The same rule transfers to new image domains in DomainNet, to ViT, Swin, and MaxViT on ImageNet, and to langua
    
[^39]: Tabra Blends：利用互补的表格模型改进准确率与延迟的权衡

    Tabra Blends: Improving the Accuracy--Latency Trade-off with Complementary Tabular Models

    [https://arxiv.org/abs/2611.32859](https://arxiv.org/abs/2611.32859)

    将四个估计器的Kumo与一个LimiX流水线混合成紧凑集成，在816个TabArena划分上比最佳已发布模型高出45.4 Elo，同时归一化拟合+预测时间比Kumo-16低15%、比完整LimiX低98%，且两个模型间的预测多样性是误差降低的关键。

    

    表格基础模型在预测准确性与延迟之间做出不同的权衡。我们研究一个紧凑的集成模型能否在更短时间内提供更高的准确率。在816个TabArena数据集划分上，我们的由四个估计器的Kumo与一个LimiX流水线组成的混合模型达到了2063.8 Elo，比相同拟合条件下的最高已发布成绩高出45.4分。其归一化的拟合+预测时间比Kumo-16低15%，比完整的LimiX低98%。我们发现这两个模型之间更大的预测多样性有助于解释误差的降低。

    arXiv:2611.32859v1 Announce Type: new  Abstract: Tabular foundation models make different trade-offs between prediction accuracy and latency. We study whether a compact ensemble can deliver higher accuracy in less time. Across 816 TabArena splits, our blend of four-estimator Kumo and one LimiX pipeline reaches 2063.8 Elo, 45.4 points above the top published entry in the same fit. Its normalized fit + predict time is 15% lower than Kumo-16's and 98% lower than full LimiX's. We find that greater prediction diversity between the two models helps explain the reduction in error.
    
[^40]: 技能词汇表的代价是什么？基于演示动作的闭环分解

    What Does a Skill Vocabulary Cost? A Closed-Loop Decomposition over Demonstrated Motions

    [https://arxiv.org/abs/2611.32858](https://arxiv.org/abs/2611.32858)

    提出运动原型量化（MPQ），将演示动作库置于已训练策略之后且无需训练参数，使包装后的策略与原策略统计等价，从而建立参考基线，可逐一归因闭环失败于选择器的具体组件。

    

    离散技能策略执行来自有限词汇表的动作。在目前的大多数方法中，这个词汇表是与选择器和解码器联合学习的潜在码本。当这样的策略在闭环中失败时，无法将失败归因于词汇表本身、技能的选择或其执行过程。我们提出了运动原型量化，它将一个由演示动作组成的库放置在已训练好的策略之后，而不是嵌入策略内部。MPQ将每个预测的动作块替换为库中最近的元素，并在一定界限内将该元素向预测方向校正，且不需要训练任何参数。在包括7B视觉-语言-动作模型在内的六种架构上，经过包装的策略与原始策略在统计上是等价的。因此，通过该动作库执行的参考策略可以作为同一库上任何选择器的基线。通过逐一替换选择器的组件，可以将其性能差距归因于具体的组件。

    arXiv:2611.32858v1 Announce Type: cross  Abstract: Discrete skill policies execute motions drawn from a finite vocabulary. In most current methods, this vocabulary is a latent codebook learned jointly with a selector and a decoder. When such a policy fails in closed loop, the failure cannot be traced to the vocabulary, to the choice of skill or to its execution. We present Movement Prototype Quantization (MPQ), which places a library of demonstrated motions after a trained policy instead of inside it. MPQ replaces each predicted action chunk with its nearest library element, corrects that element toward the prediction within a bound, and trains no parameters. Across six architectures, including a 7B vision-language-action model, the wrapped policy is statistically equivalent to the original. A reference policy executed through the library is therefore a baseline for any selector over the same library. Substituting a selector's components one at a time attributes its shortfall to them. 
    
[^41]: 基于路径概率梯度针对动力学观测量优化力场

    Path-probability gradients for optimizing force fields against dynamical observables

    [https://arxiv.org/abs/2611.32851](https://arxiv.org/abs/2611.32851)

    提出PathGrad方法，通过对朗之万路径概率而非混沌分子轨迹求导，获得方差稳定的梯度估计器，从而能够直接利用实验动力学观测量（如扩散系数）对机器学习力场进行微调。

    

    机器学习力场继承了用于生成其训练数据的电子结构方法的误差，而利用实验测量数据对其进行精修提供了一条突破这一限制的直接途径。然而，对于动力学观测量而言，这种精修仍然十分困难，因为对混沌的分子轨迹进行微分会产生不稳定的梯度。在此，我们提出了PathGrad方法，它对朗之万路径概率而非轨迹本身进行微分。由此得到的梯度估计器可以从热噪声和力导数中即时累积，并且其方差随着轨迹长度的增加保持稳定。我们通过有限差分计算验证了该梯度估计器，并利用它结合实验扩散系数以及结构约束，对熔融铜和水的势能模型进行了微调。对于水，我们联合优化了扩散系数和氧-氧径向分布函数。

    arXiv:2611.32851v1 Announce Type: cross  Abstract: Machine learning force fields inherit the errors of the electronic-structure methods used to generate their training data, and refining them against experimental measurements provides a direct route beyond this limit. However, such refinement remains difficult for dynamical observables because differentiating chaotic molecular trajectories yields unstable gradients. Here, we introduce PathGrad, which differentiates Langevin path probabilities instead of the trajectories themselves. The resulting gradient estimator is accumulated on the fly from thermal noise and force derivatives, and its variance remains stable as the trajectory length increases. We validated the gradient estimator against finite-difference calculations and used it to fine-tune molten-copper and water potentials against experimental diffusion coefficients together with structural constraints. For water, jointly optimizing diffusion and the oxygen-oxygen radial distrib
    
[^42]: PtychoFM：一个用于叠层衍射成像的基础模型

    PtychoFM: A Foundation Model for Ptychographic Imaging

    [https://arxiv.org/abs/2611.32830](https://arxiv.org/abs/2611.32830)

    PtychoFM是一种物理感知的叠层衍射成像基础模型，通过可微正向模型在大规模数据上预训练并针对具体实验微调，无需物体空间标签即可直接从单个衍射图样重建图像，并能跨样品与扫描方式迁移，在同步辐射装置上实现实时推理。

    

    叠层衍射成像技术可在X射线、电子和光学显微领域提供高分辨率无透镜成像，但从衍射测量数据重建图像通常需要计算量极大的迭代相位恢复算法。这些方法往往依赖针对特定实验的分析流程，随着数据采集速率的提升，难以实现实时反馈。在此，我们提出了PtychoFM，一种物理感知的叠层衍射成像基础模型，它将大规模预训练与针对特定实验的微调相结合，能够直接从单个衍射图样重建复数物体图像。PtychoFM使用可微正向模型进行训练，无需物体空间的标签，能够从在一个样品上的步进扫描训练数据迁移到具有不同特征尺寸、形貌和折射率的其他样品的飞行扫描测量。在同步辐射光束线上，衍射帧从探测器流式传输至单个GPU进行推理。

    arXiv:2611.32830v1 Announce Type: new  Abstract: Ptychography provides high-resolution lensless imaging across X-ray, electron, and optical microscopy, but reconstructing images from diffraction measurements generally requires computationally intensity iterative phase retrieval. These methods often require experiment-specific analysis, limiting real-time feedback as data acquisition rates increase. Here, we demonstrate PtychoFM, a physics-aware foundation model for ptychographic imaging that combines large-scale pre-training with experiment-specific fine-tuning to reconstruct complex object images directly from individual diffraction patterns. Trained using a differentiable forward model without object-space labels, PtychoFM transfers from step-scan training data on one sample to fly-scan measurements of other samples with different feature sizes, morphologies, and refraction indices. At a synchrotron beamline, diffraction frames were streamed from the detector to a single GPU for infe
    
[^43]: 多稳态粘弹性流动的数据驱动降阶建模

    Data-driven Reduced-order Modeling of a Multistable Viscoelastic Flow

    [https://arxiv.org/abs/2611.32818](https://arxiv.org/abs/2611.32818)

    该论文提出了一种数据驱动框架，仅利用稳定流态的稀疏稳态测量数据，并通过平衡态约束、稳定性约束和物理反射对称性，学习出能够刻画并排双微圆柱粘弹性流动多稳态分岔结构的二维参数依赖正规形模型。

    

    并排双微圆柱绕流的粘弹性流动已被证明会表现出依赖于几何构型的流态转变，即在对称、汇聚、发散和非对称发散等流动状态之间的转换，同时存在双稳态和三稳态区域。在此，我们开发了一个数据驱动框架，用于从稳定流态的稀疏测量数据中学习描述该分岔结构的二维、参数依赖的正规形。我们采用流动不对称参数 $I'$ 和 $I''$ 作为状态变量，以间隙比 $G$ 和魏森贝格数 $Wi$ 作为控制参数。在缺乏轨迹数据的情况下，通过平衡态约束和局部稳定性约束，将测得的稳态强制为学习所得向量场的稳定平衡点。我们考察了该正规形的多项式和神经网络两种表示形式，并将物理反射对称性直接融入模型之中。所得模型恢复了观测到的分岔结构（摘要原文在此处被截断）。

    arXiv:2611.32818v1 Announce Type: cross  Abstract: Viscoelastic flow past two side-by-side microcylinders has been shown to exhibit geometry-dependent transitions among symmetric, converging, diverging, and asymmetric-diverging flow states, together with regions of bistability and tristability. Here, we develop a data-driven framework for learning a two-dimensional, parameter-dependent normal form for this bifurcation structure from sparse measurements of stable flow states. We use the flow asymmetry parameters $I'$ and $I''$ as state variables, with the gap ratio $G$ and Weissenberg number $Wi$ as control parameters. In the absence of trajectory data, the measured steady states are enforced as stable equilibria of the learned vector field through equilibrium and local-stability constraints. We examine both polynomial and neural-network representations of the associated normal form and incorporate the physical reflection symmetry directly into the model. The resulting models recover th
    
[^44]: E(n)等变球面决策曲面

    E(n)-Equivariant Spherical Decision Surfaces

    [https://arxiv.org/abs/2611.32815](https://arxiv.org/abs/2611.32815)

    该论文通过将O(n)等变超球面神经元扩展以包含平移，构造性地推导出无需输入中心化或显式成对差值即严格E(n)等变的球面决策曲面，并给出其显式闭式矩阵表示。

    

    我们通过将先前的O(n)等变超球面神经元扩展以包含平移，提出了严格E(n)等变球面决策曲面的构造性推导。为实现这一目标，我们对O(n)等变神经元的特征进行了分解，并为平移和E(n)变换提供了显式表示，以满足相应的等变性约束。所得的决策曲面在无需输入中心化或显式成对差值的情况下即严格E(n)等变，并且具有显式的闭式矩阵表示。此外，我们通过数值方法验证了推导的正确性，并对所得的几何基元进行了下游检验。

    arXiv:2611.32815v1 Announce Type: new  Abstract: We present a constructive derivation of exactly E(n)-equivariant spherical decision surfaces by extending prior O(n)-equivariant hypersphere neurons to include translations. To achieve this, we present a decomposition of the features of the O(n)-equivariant neurons and provide explicit representations for translation and E(n)-transformations to fulfil the respective equivariance constraints. The resulting decision surfaces are exactly E(n)-equivariant without input centring or explicit pairwise differences, and admit explicit closed-form matrix representations. In addition, we numerically verify the correctness of the derivations and perform a downstream check of the resulting geometric primitives.
    
[^45]: 打破分布鲁棒多智能体强化学习中对联合动作空间的依赖

    Breaking the Joint Action Space Dependence in Distributionally Robust Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.32812](https://arxiv.org/abs/2611.32812)

    该论文设计了一种分布鲁棒多智能体强化学习算法，其样本复杂度仅依赖于最大个体动作集大小而非随玩家数指数增长的联合动作数，从而打破了多智能体维数灾难。

    

    我们研究了在有限时域、一般和、分布鲁棒马尔可夫博弈中学习鲁棒粗相关均衡（CCE）的样本复杂度，其中奖励已知，且拥有名义转移的生成模型。每个玩家通过其在以每个阶段、状态和联合动作的名义转移分布为中心的总变差（TV）球上的最坏情况回报来评估策略；这是标准的 $(s,a)$ 矩形不确定性集。针对该模型，现有基于生成模型的样本保证随联合动作数量 $\prod_i A_i$（其中 $A_i$ 是玩家 $i$ 的动作数量）而扩展。由于该数量随玩家数量呈指数增长，这些保证饱受“多智能体维数灾难”之苦。我们设计了一种算法，其样本复杂度仅通过最大个体动作集大小 $A_* := \max_i A_i$ 依赖于动作集。该算法使用 $\widetilde O(SA_*H^5D^4/\varepsilon^4)$ 次转移采样

    arXiv:2611.32812v1 Announce Type: new  Abstract: We study the sample complexity of learning robust coarse correlated equilibria (CCE) in finite-horizon, general-sum, distributionally robust Markov games with known rewards and a generative model of the nominal transitions. Each player evaluates a policy by its worst-case return over total-variation (TV) balls centered at the nominal transition distribution of every stage, state, and joint action; this is the standard $(s,a)$-rectangular uncertainty set. Existing generative-model guarantees for this model scale with the number $\prod_iA_i$ of joint actions, where $A_i$ is the number of actions of player~$i$. Since this number grows exponentially with the number of players, these guarantees suffer from the curse of multiagency. We design an algorithm whose sample complexity depends on the action sets only through the largest individual action-set size $A_*:=\max_iA_i$. The algorithm uses $\widetilde O(SA_*H^5D^4/\varepsilon^4)$ transition
    
[^46]: EPRO：通过不完美的执行策略预测器实现鲁棒策略优化

    EPRO: Robust Policy Optimization Through an Imperfect Execution-Policy Predictor

    [https://arxiv.org/abs/2611.32810](https://arxiv.org/abs/2611.32810)

    EPRO 针对大语言模型强化学习中训练与 rollout 执行策略不一致且当前执行策略不可直接观测的问题，利用训练策略变化构建可微分的执行策略预测器，并在信任度约束下对实际执行策略进行鲁棒优化。

    

    针对大语言模型的强化学习（RL）通常将 rollout 生成与策略训练分离。尽管这两个数值引擎共享模型参数，它们却可能诱导出不同的策略。这带来了一个超出修正已知训练-推理不匹配范畴的挑战：在学习器更新期间，与当前参数相对应的执行策略无法被直接观测到。我们将此问题视为通过一个对其演化的不完美预测器来优化一个不可观测的执行策略。这一视角引出了两个根本性问题：我们如何估计执行策略对学习器更新的响应，以及在优化实际执行策略时我们应该在多大程度上信任该估计？我们提出了执行策略鲁棒优化（EPRO）来应对这些挑战。从观测到的 rollout 时执行策略出发，我们利用训练策略的变化来构建当前执行策略的可微分预测器，其分数可产生 Transpor（摘要原文在此处截断）……

    arXiv:2611.32810v1 Announce Type: new  Abstract: RL for LLMs often separates rollout generation from policy training. Although the two numerical engines share model parameters, they can induce different policies. This creates a challenge beyond correcting a known training-inference mismatch: during learner updates, the execution policy corresponding to the current parameters cannot be directly observed. We view this as optimizing an unobserved execution policy through an imperfect predictor of its evolution. This perspective raises two fundamental questions: how can we estimate the execution policy's response to learner updates, and how much should we trust that estimate when optimizing the actual execution policy? We introduce Execution-Policy Robust Optimization (EPRO) to address these challenges. Starting from the observed rollout-time execution policy, we use training-policy changes to construct a differentiable predictor of the current execution policy, whose score yields Transpor
    
[^47]: 基于睡眠机制的样本高效上下文适应

    Sample-Efficient In-Context Adaptation via Sleep

    [https://arxiv.org/abs/2611.32802](https://arxiv.org/abs/2611.32802)

    该研究提出在Transformer语言模型预训练中引入“睡眠”机制（通过多次循环前向传递精炼上下文表示），显著提升了模型在一般性任务上少样本上下文学习的样本效率。

    

    arXiv:2611.32802v1 公告类型：交叉 摘要：从有限经验中学习并泛化的能力，即样本效率，是智能的基石。我们如何构建能够将少量用户交互、几段陌生代码或上下文中提供的少量实验结果转化为可泛化知识的语言模型？先前的研究发现，“睡眠”——即模型执行 N 次循环前向传递以精炼其对一段上下文表示的过程——可以改善长上下文推理（具体而言，在 SSM-注意力混合模型中）。我们研究这种方法是否也能改善Transformer语言模型在一般性、非合成任务上的上下文适应能力。为此，我们使用睡眠机制在多达 1000 亿 token 上预训练了参数量高达 12 亿的语言模型。值得注意的是，更长的睡眠（即使用更多的循环前向传递次数 N）使模型能够在多个少样本算法任务上更快地进行上下文学习……

    arXiv:2611.32802v1 Announce Type: cross  Abstract: The ability to learn and generalize from limited experience, or sample efficiency, is a cornerstone of intelligence. How can we build language models that turn a few user interactions, a few snippets of unfamiliar code, or a handful of experimental results provided in context into knowledge that generalizes? Prior work has found that sleep, a process where the model performs $N$ recurrent forward passes to refine its representations of a chunk of context, can improve long-context reasoning (specifically, within SSM-attention hybrid models). We study whether this method also improves in-context adaptation on general, non-synthetic tasks for transformer language models. To this end, we pre-train language models up to 1.2B-parameters on up to 100B tokens with sleep mechanisms. Notably, longer sleep (i.e., using a larger number of recurrent forward passes $N$) enables the model to learn faster in-context on multiple few-shot algorithmic, q
    
[^48]: 语言模型中集成不确定性的缩放定律

    Scaling Laws of Ensemble Uncertainty in Language Models

    [https://arxiv.org/abs/2611.32801](https://arxiv.org/abs/2611.32801)

    本文提出三条幂律缩放定律，证明集成随模型规模增大会捕获越来越多的认知不确定性，作为扩展路径相比单个更大的模型更具竞争力。

    

    大型语言模型（LLM）的许多应用场景——从AI安全到探索与人类交互——都能从认知不确定性估计中受益。我们认为，现有的贝叶斯深度学习解决方案通常局限于固定的模型族，在实际的LLM规模下无法捕捉完整的认知不确定性。这是因为由模型族有限容量导致的误差，对于先验所支持的模型之间的分歧而言是不可见的。我们将这一误差与著名的LLM损失随模型规模的幂律缩放联系起来。随后，我们证明第二条幂律量化了集成内部分歧随模型规模增大而缩小的现象。相对于上述容量定律，该幂律较慢的衰减速度表明，集成能够捕获越来越多的认知不确定性份额，与单个更大的模型相比，集成作为一条扩展路径变得更具竞争力。最后，第三条幂律量化了集成……（摘要在此处截断）

    arXiv:2611.32801v1 Announce Type: cross  Abstract: Many use cases for large language models (LLMs)---from AI safety to exploration and human interaction---can benefit from epistemic uncertainty estimates. We argue that existing Bayesian Deep Learning solutions, which are often restricted to a fixed model family, fail to capture the full epistemic uncertainty at practical LLM scales. This is because the error due to the bounded capacity of the model family remains invisible to disagreement among models supported by the prior. We relate this error to the well-known power-law scaling of LLM loss with model size. We then show that a second power law quantifies the shrinking disagreement within ensembles as model size increases. Its slower decay relative to the capacity law above suggests that ensembles capture a growing share of epistemic uncertainty and become more competitive as an axis for scaling, compared to a single larger model. Finally, a third power law quantifies the loss of ense
    
[^49]: 面向机器人操作的“模拟器作为代理”世界模型

    Simulator-as-Proxy World Model for Robot Manipulation

    [https://arxiv.org/abs/2611.32786](https://arxiv.org/abs/2611.32786)

    本文提出SimPWM，用一个无需训练参数的规则模拟器作为显式代理世界模型，仅凭第一帧即可表示机器人与物体，并首次系统验证了视频世界模型是否忠实跟随模拟代理——结果显示机械臂能紧密跟随代理，而物体的行为则可能出现偏差。

    

    机器人的视频世界模型越来越多地以场景的模拟或渲染代理作为条件，其假设是视频将呈现出代理所呈现的内容。要验证这一假设，需要一个状态显式且可被扰动的代理。我们提出了SimPWM（模拟器作为代理的世界模型）：给定一个经过标定的相机，它仅从第一帧出发，用机器人自身的3D模型来表示机器人，并将夹爪可能接触到的物体表示为3D点；一个不含任何训练参数的基于规则的模拟器在动作作用下移动这些元素，随后由视频模型渲染出结果。SimPWM同时返回逼真的视频和模拟的状态。接着，我们冻结模拟中的物体，让机械臂按照记录的动作运动，并检查视频中物体是否仍然被抬起。在来自未见过的环境的DROID片段上，机械臂和物体给出的答案有所不同。使用三种基础视频模型中的每一种，机械臂都能跟随代理（机器人IoU为0.75-0.78……（摘要原文在此处被截断）

    arXiv:2611.32786v1 Announce Type: cross  Abstract: Video world models for robots are increasingly conditioned on a simulated or rendered proxy of the scene, on the assumption that the video will show what the proxy shows. Testing this assumption requires a proxy whose state is explicit and can be perturbed. We present SimPWM (Simulator-as-Proxy World Model): given a calibrated camera, it represents, from the first frame alone, the robot by its 3D model and the objects the gripper may touch as 3D points; a rule-based simulator with no trained parameters moves them under the actions, and a video model renders the result. SimPWM returns the realistic video together with the simulated state. We then freeze the simulated objects while the arm moves as recorded and check whether the video still shows the object lifted. On DROID episodes from unseen setups, the answer differs for the arm and for the objects. With each of three base video models, the arm follows the proxy (robot IoU 0.75-0.78,
    
[^50]: 通过PCA测量大语言模型中FFN的逐层内在维度

    Measuring Layer-wise Intrinsic Dimensionality of FFNs in LLMs via PCA

    [https://arxiv.org/abs/2611.32782](https://arxiv.org/abs/2611.32782)

    该研究用PCA测量58个开源LLM各层FFN的内在维度，发现中间层的内在维度显著高于边缘层的“中间重”分布，表明逐层均匀分配参数的传统设计并非最优，为非均匀的参数预算分配提供了实证依据。

    

    现代大语言模型（LLM）在每一层都保持前馈网络（FFN）中间宽度相同，这是从原始Transformer继承而来的惯例。然而，机制研究表明不同深度的层做着质上不同的工作，这引出了一个自然的问题：它们的容量需求是否也是不均匀的？如果是，固定的参数预算能否比均匀分配得到更明智的利用？为探究这一问题，我们对每个FFN下投影的输入应用主成分分析（PCA），并以将激活重构至固定余弦相似度所需的最少主成分数量来度量每一层的内在维度。通过扫描涵盖14个模型家族、参数规模从70M到72B的58个开源权重LLM，我们发现了一种反复出现的“中间重”轮廓——中间层的内在维度高于边缘层；该轮廓对校准语料库具有鲁棒性，甚至在不同……（注：摘要原文在此处被截断）

    arXiv:2611.32782v1 Announce Type: cross  Abstract: Modern large language models (LLMs) keep the feed-forward network (FFN) intermediate width identical across every layer, a convention inherited from the original Transformer. Yet mechanistic work suggests layers at different depths do qualitatively different work, raising a natural question: are their capacity demands also non-uniform, and if so, can a fixed parameter budget be allocated more wisely than uniformly? To probe this, we apply principal component analysis (PCA) to the input of each FFN down-projection and measure, for every layer, its intrinsic dimensionality as the smallest number of principal components needed to reconstruct the activation up to a fixed cosine similarity. Scanning 58 open-weight LLMs spanning 14 families and 70M to 72B parameters, we find a recurring middle-heavy profile in which middle layers carry higher intrinsic dimensionality than edge layers; the profile is robust to the calibration corpus, even acr
    
[^51]: NABLA：一种用于随机优化的噪声自适应分块左右预条件算法

    NABLA: A Noise-Adaptive Blockwise Left-Right Preconditioning Algorithm for Stochastic Optimization

    [https://arxiv.org/abs/2611.32781](https://arxiv.org/abs/2611.32781)

    本文提出NABLA算法，通过闭环反馈规则自适应调整二阶矩平均系数，实现无需噪声先验知识的分块左右预条件随机优化。

    

    我们提出了NABLA，一种用于小批量随机非凸优化的噪声自适应分块左右预条件算法。NABLA将随机梯度的指数移动平均与由矩阵值二阶矩统计量构建的分块对角行和列预条件子相结合。与使用固定衰减参数的传统矩阵预条件方法不同，NABLA通过一个由累积随机梯度能量及其滑动平均值所确定的闭环反馈规则来调整二阶矩平均系数。一阶矩系数保持固定，从而使依赖轨迹的自适应机制能够直接作用于结构化预条件子。由此得到的方法不受时间尺度限制，无需对噪声水平的先验知识，并能根据观察到的优化动态自动调整其二阶矩估计器的记忆长度。在标准的有下界性、平滑性、无偏性（假设条件下，原文至此被截断）

    arXiv:2611.32781v1 Announce Type: cross  Abstract: We propose NABLA, a noise-adaptive blockwise left-right preconditioning algorithm for mini-batch stochastic nonconvex optimization. NABLA combines an exponential moving average of stochastic gradients with block-diagonal row and column preconditioners constructed from matrix-valued second-moment statistics. Unlike conventional matrix-preconditioned methods that use fixed decay parameters, NABLA adjusts the second-moment averaging coefficient through a closed-loop feedback rule determined by the cumulative stochastic-gradient energy and its running average. The first-moment coefficient remains fixed, allowing the trajectory-dependent adaptation to act directly on the structured preconditioners. The resulting method is horizon-free, requires no prior knowledge of the noise level, and automatically adjusts the memory of its second-moment estimators to the observed optimization dynamics. Under standard lower boundedness, smoothness, unbias
    
[^52]: 大学习率下的随机梯度下降：一般性分析与改进的收敛速率

    SGD with Large Learning Rates: General Analysis & Improved Rates

    [https://arxiv.org/abs/2611.32762](https://arxiv.org/abs/2611.32762)

    本文证明了大学习率的SGD在随机梯度噪声存在下仍能加速收敛，对线性预测器的交叉熵损失和两层神经网络建立了加速的O(T^-2)收敛速率，并在稳定区域内给出了更紧致的收敛保证。

    

    随机梯度下降（SGD）是现代机器学习的核心。然而，其经典分析通常将步长限制在低于一个依赖于曲率的稳定性阈值以下。确定性优化领域的最新进展挑战了这一范式，表明适当选择的大步长不仅可以保持收敛性，还能加速梯度下降（GD）。这一现象在存在随机梯度噪声的情况下是否仍然成立，一直是一个悬而未决的问题（Wu et al., 2024）。在本工作中，我们对带有线性预测器的交叉熵损失以及两层神经网络肯定地回答了这一问题，建立了加速的 $\mathcal{O}(T^{-2})$ 收敛速率。我们分析的核心是对随机大步长动力学的细粒度刻画。这种对轨迹更精确的控制进一步在稳定区域内给出了紧致的收敛保证，改进了……

    arXiv:2611.32762v1 Announce Type: cross  Abstract: Stochastic Gradient Descent lies at the core of modern machine learning. However its classical analysis typically restricts the stepsize to be below a curvature-dependent stability threshold. Recent developments in deterministic optimization have challenged this paradigm, showing that suitably chosen large stepsizes can not only preserve convergence, but also accelerate Gradient Descent (GD). Whether this phenomenon persists in the presence of stochastic gradient noise has remained an open question (Wu et al., 2024). In this work, we answer this question affirmatively for the cross-entropy loss with linear predictors and two-layer neural networks, establishing an accelerated $\mathcal{O}(T^{-2})$ convergence rate. At the core of our analysis is a fine-grained characterization of the stochastic large-stepsize dynamics. This sharper control of the trajectory further yields tight convergence guarantees within the stable regime, improving 
    
[^53]: ReQOR：面向大语言模型量化条件后门的算子差异子空间修复

    ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models

    [https://arxiv.org/abs/2611.32755](https://arxiv.org/abs/2611.32755)

    ReQOR提出了一种无需训练的防御方法，通过在良性校准数据上识别模块级量化算子差异的主导方向，并以闭式形式恢复全精度计算，有效修复大语言模型中的量化条件后门。

    

    量化条件后门会使大语言模型在量化后表现出恶意行为，而在高精度状态下保持良性。现有防御方法通常依赖安全导向的配置选择或迭代优化，但未能识别哪些由量化引发的变化应当被修复。我们通过在来自量化推理轨迹的相同状态上评估每个模块的高精度版本与量化版本，将模块局部的量化效应与上游漂移分离，并将量化条件后门防御形式化为修复由此产生的同状态算子差异。我们提出ReQOR，一种无需训练的防御方法，它在良性校准数据上识别这些差异的主导方向，并以闭式形式沿这些方向恢复全精度计算，从而产生一个因子化的残差适配器。ReQOR既不使用攻击信息，也不使用基于梯度的优化，并支持……

    arXiv:2611.32755v1 Announce Type: cross  Abstract: Quantization-Conditioned Backdoors (QCBs) cause large language models (LLMs) to exhibit malicious behavior after quantization while remaining benign at high precision. Existing defenses often rely on security-guided configuration selection or iterative optimization, but do not identify which quantization-induced changes should be repaired. We isolate module-local quantization effects from upstream drift by evaluating the high-precision and quantized versions of each module on identical states from the quantized inference trajectory, and formulate QCB defense as repairing the resulting same-state operator discrepancies. We propose ReQOR, a training-free defense that identifies the dominant directions of these discrepancies on benign calibration data and restores the full-precision computation along them in closed form, yielding a factorized residual adapter. ReQOR uses neither attack information nor gradient-based optimization and suppo
    
[^54]: NCPU：迈向受细胞启发的空间计算

    NCPU: Towards Cell-Inspired Spatial Computing

    [https://arxiv.org/abs/2611.32750](https://arxiv.org/abs/2611.32750)

    该论文提出用神经元胞自动机（NCA）通过梯度下降学习局部更新规则来实现精确的数字逻辑计算，将比特编码为二维网格上的细胞簇，成功训练出逻辑门、4位和8位加法器及初步的8位ALU，全部达到100%位级正确率，且仅用3位操作数训练的加法器可泛化到8位。

    

    简单的元胞自动机（CA）可以具备计算通用性，但其规则需要手工设计。我们训练神经元胞自动机（NCA）——通过梯度下降学习局部更新规则——来执行精确的数字逻辑计算。比特被编码为二维网格上的细胞簇，计算结果只有通过中间细胞的传播才能到达输出端。每个任务训练一个模型：逻辑门、4位和8位行波进位加法器，以及一个初步的8位算术逻辑单元（ALU）。所有门电路和加法器模型在10个随机种子中全部达到100%的位级正确率，对所有输入组合的穷举测试发现，4位加法器在256个测试中最多出现1个错误，8位加法器在65,536个测试中最多出现7个错误。仅用最多3位操作数训练的加法器能够泛化到8位。加法器以波的形式传播进位，进位链中每比特的延迟约为3.5步。输出仅在训练的演化步数范围内保持正确。

    arXiv:2611.32750v1 Announce Type: cross  Abstract: Simple cellular automata (CAs) can be computationally universal, but their rules are designed by hand. We train Neural Cellular Automata (NCAs), which learn a local update rule by gradient descent, to compute exact digital logic. Bits are encoded as clusters of cells on a 2D grid, and a result reaches the output only by propagating through intermediate cells. One model is trained per task: logic gates, 4- and 8-bit ripple-carry adders, and a preliminary 8-bit arithmetic logic unit (ALU). All gate and adder models reach 100% bit correctness in 10 of 10 seeds, and exhaustive tests over all input pairs find at most 1 error in 256 (4-bit) and 7 in 65,536 (8-bit). Adders trained on operands of up to 3 bits generalize to 8 bits. The adders propagate the carry as a wave, with a delay of about 3.5 steps per bit of the carry chain. Outputs are correct only within the trained rollout length.
    
[^55]: 用于数据分析的量子社区检测

    Quantum Community Detection for Data Analysis

    [https://arxiv.org/abs/2611.32738](https://arxiv.org/abs/2611.32738)

    本文提出了量子谱社区检测算法projeQt，利用量子奇异值变换和基于量子态重叠的k均值聚类，无需递归即可一次性提取多个社区，可应用于特征选择和聚类等数据分析任务并有望实现量子加速。

    

    社区检测为特征选择和聚类等数据分析任务提供了一种有别于传统方法的选择，而利用量子算法检测社区是实现量子加速的一条有前景的途径。我们研究了量子谱社区检测器，例如模块度矩阵算法deteQt，在特征选择和聚类两类任务的流水线中的应用。我们首先通过递归二分法将deteQt推广到多社区的情形，并证明它能够在合成数据集和小型数据集中恢复植入的社区，从而识别出冗余特征。随后我们提出了projeQt，该算法通过量子奇异值变换对编码的模块度矩阵施加谱滤波器，保留了deteQt未使用的高阶特征向量。projeQt作用于广义模块度矩阵，通过对由量子态重叠构建的Gram矩阵进行k均值聚类，无需递归即可一次性提取多个社区。在MNIST数据集上我们展示了……（原文摘要在此处截断）

    arXiv:2611.32738v1 Announce Type: cross  Abstract: Community detection offers an alternative to conventional methods for data analysis tasks such as feature selection and clustering, and detecting communities with quantum algorithms is a promising avenue for quantum speedup. We investigate quantum spectral community detectors, such as the modularity-matrix algorithm deteQt, in pipelines for both tasks. We first generalise deteQt to multiple communities by recursive bisection, and show that it recovers planted communities, and hence redundant features, in synthetic and small datasets. We then propose projeQt, which applies a spectral filter to the encoded modularity matrix through the quantum singular value transformation, retaining higher-order eigenvectors that deteQt does not use. projeQt acts on the generalised modularity matrix and extracts several communities at once, without recursion, by k-means clustering of a Gram matrix built from quantum state overlaps. On MNIST we demonstra
    
[^56]: 单板视觉驾驶：面向机载机器人导航的无注意力深度估计

    Seeing to Drive on a Single Board: Attention-Free Depth for Onboard Robot Navigation

    [https://arxiv.org/abs/2611.32718](https://arxiv.org/abs/2611.32718)

    该论文提出用知识蒸馏而非量化，将无注意力的ResNet-34 U-Net学生模型部署到树莓派5的Hailo-8L加速器上，以27.1 FPS实现芯片上最精确的单目深度估计，使小型机器人能依靠单块板卡的视觉进行导航。

    

    小型地面机器人可以携带一个摄像头和几瓦特的算力，但那些能够将摄像头转化为距离传感器的深度基础模型，却是为GPU规模设计的视觉Transformer。我们将其中的一个模型部署到配备Hailo-8L加速器的树莓派5上——该加速器以整数运算运行所有层，包括注意力层。经过针对性量化的Depth Anything V2会丢失场景的大部分内容，我们测量了原因：在449个token时，91%的softmax权重低于一个int8量化步长，却承载了17-33%的注意力质量。我们的解决方案是蒸馏而非量化。一个从Depth Anything V2-Large（2450万参数）蒸馏而来的无注意力ResNet-34 U-Net学生网络，在芯片上以27.1 FPS运行，其int8输出与其浮点版本几乎完全一致（皮尔逊相关系数0.99993），并且在其模拟庭院的留出测试场景上，它是该芯片上最精确的深度网络（AbsRel为0.269，而对比方法为0.354-0.530）。在真实户外行驶采集的1000个经立体视觉验证的采样点上，它（与浮点教师模型）的差异不可分辨。

    arXiv:2611.32718v1 Announce Type: cross  Abstract: A small ground robot can carry one camera and a few watts of compute, but the depth foundation models that could turn that camera into a range sensor are vision transformers sized for GPUs. We bring one to a Raspberry Pi 5 with the Hailo-8L, an accelerator that runs every layer in integer arithmetic, attention included. Quantised for it, Depth Anything V2 loses large parts of the scene, and we measure why: at 449 tokens, 91% of softmax weights fall below one int8 step while carrying 17-33% of the attention mass. Our answer is to distil, not quantise. An attention-free ResNet-34 U-Net student of Depth Anything V2-Large (24.5M parameters) runs at 27.1 FPS on the chip, its int8 output matches its float twin (Pearson 0.99993), and on held-out scenes of its simulated yard it is the most accurate depth network on the chip (AbsRel 0.269 against 0.354-0.530). On 1,000 stereo-verified points of a real outdoor drive it is not resolvably differen
    
[^57]: FourStat：一种利用频率与平稳性的双自编码器时间序列异常检测方法

    FourStat: A Dual AutoEncoder Leveraging Frequency and Stationarity for Time Series Anomaly Detection

    [https://arxiv.org/abs/2611.32714](https://arxiv.org/abs/2611.32714)

    提出FourStat双自编码器框架，结合频域分析的傅里叶自编码器与强化统计平稳性的平稳自编码器来检测单变量时间序列异常，并引入Perfect Point新指标以消除阈值选择的主观性。

    

    尽管大量研究聚焦于多变量时间序列异常检测，但许多现有方法在单变量场景下表现不佳或无法直接应用。我们提出了FourStat，一个用于单变量时间序列异常检测的双模型框架，它结合了两种互补的方法。傅里叶自编码器（FAE）通过频域分析捕捉季节性等全局模式，而平稳自编码器（SAE）通过强化统计平稳性来关注局部结构。我们还提出了一种名为Perfect Point的新评估指标，用于评估能否直接从异常分数中获得完美的F1分数，从而消除阈值选择的主观性。结果表明，FourStat优于现有基线方法，尤其是在具有挑战性的异常类型上，同时新指标能更清晰地揭示模型性能的差距与优势。

    arXiv:2611.32714v1 Announce Type: cross  Abstract: While numerous studies have focused on multivariate time series anomaly detection, many existing methods perform poorly or are not directly applicable to univariate settings. We introduce FourStat, a dual-model framework for univariate time series anomaly detection that combines two complementary approaches. The Fourier Autoencoder (FAE) captures global patterns like seasonality through frequency-domain analysis, while the Stationary Autoencoder (SAE) focuses on local structures by enforcing statistical stationarity. We also propose a new evaluation metric called Perfect Point that assesses whether perfect F1-scores can be achieved directly from anomaly scores, eliminating thresholding subjectivity. Results demonstrate that FourStat outperforms existing baselines, particularly on challenging anomaly types, while the new metric provides clearer insights into model performance gaps and advantages.
    
[^58]: 对比代理森林：从多模态记录中进行保持完备性的代理学习

    Contrastive Proxy Forests: Completeness-Preserving Proxy Learning from Multimodal Records

    [https://arxiv.org/abs/2611.32708](https://arxiv.org/abs/2611.32708)

    该论文提出对比代理森林（CPF），利用处理侧与结果侧模态之间由混杂因子构成的唯一联系，从高维多模态临床记录中学习共享表示以捕获混杂因子、过滤模态特有风格，从而保持代理变量的完备性。

    

    近端因果推断通过使用混杂因子的两个代理变量来消除未测量的混杂：结果侧代理 W 和处理侧代理 Z。这些代理变量必须满足排除限制并且是完备的，即它们能反映混杂因子的每一个方面。在实践中，这些代理变量通常是从结构化表格中手工挑选的少数变量，然而临床笔记和生理波形信号承载着关于混杂因子的丰富信息。这类记录对于近端估计器而言维度过高，而 PCA 等通用压缩方法可能保留模态特有的变化（风格）而非混杂因子本身，从而破坏完备性。我们的关键观察是：在近端条件下，混杂因子是处理侧模态与结果侧模态之间除协变量之外的唯一联系。因此，学习两者共享的内容即可捕获混杂因子，并过滤掉模态特有的风格。对比代理森林学习两个代理变量……

    arXiv:2611.32708v1 Announce Type: new  Abstract: Proximal causal inference removes unmeasured confounding using two proxies of the confounder, an outcome-side W and a treatment-side Z. The proxies must satisfy exclusion restrictions and be complete, meaning that they reflect every aspect of the confounder. In practice, they are a few variables hand-picked from a structured table, even though clinical notes and physiological waveforms carry rich information about the confounder. Such records are too high-dimensional for a proximal estimator, and generic compression such as PCA may keep modality-specific variation (style) instead of the confounder, destroying completeness. Our key observation is that, under the proximal condition, the confounder is the only link between a treatment-side and an outcome-side modality beyond the covariates. Learning what they share therefore captures the confounder and filters out modality-specific style. Contrastive Proxy Forests (CPF) learns both proxies 
    
[^59]: 一种用于不规则地形上快速无线电传播预测的解析-神经混合残差代理模型

    A Hybrid Analytical-Neural Residual Surrogate for Fast Radio Propagation Prediction over Irregular Terrain

    [https://arxiv.org/abs/2611.32698](https://arxiv.org/abs/2611.32698)

    本文提出一种解析-神经混合残差代理模型，用确定性解析基线（涵盖自由空间扩散、天线方向性、刀口衍射和地形镜面反射）分离可解析的传播趋势，再由紧凑的由粗到细神经网络预测残差场，从而实现不规则地形上快速准确的无线电传播预测。

    

    arXiv:2611.32698v1 公告类型：cross 摘要：在不规则地形上进行无线电波传播的快速预测，对于山区环境中的覆盖分析至关重要，因为需要对多种地形和天线配置进行反复评估。高精度的分步抛物方程（SSPE）求解器能够捕捉地形引起的阴影和干扰效应，但其计算成本对于反复的规划研究而言过高。基于学习的代理模型可以降低这一成本，但直接的场回归要求网络同时学习解析上可处理的传播趋势和依赖于地形的残差结构。本文提出了一种解析-神经混合残差代理模型，将这两个部分分离。确定性的解析基线表征自由空间扩散、天线方向性、改进的刀口衍射以及地形感知的镜面反射，而一个紧凑的由粗到细神经网络则预测残差场。该网络利用基线导出的引导（信息）……

    arXiv:2611.32698v1 Announce Type: cross  Abstract: Fast prediction of radio-wave propagation over irregular terrain is important for coverage analysis in mountainous environments, where many terrain and antenna configurations must be evaluated repeatedly. High-fidelity split-step parabolic equation (SSPE) solvers can capture terrain-induced shadowing and interference, but their computational cost is high for repeated planning studies. Learning-based surrogates reduce this cost, but direct field regression asks the network to learn both analytically tractable propagation trends and terrain-dependent residual structure. This paper proposes a hybrid analytical-neural residual surrogate that separates these two parts. A deterministic analytical baseline represents free-space spreading, antenna directivity, modified knife-edge diffraction, and terrain-aware specular reflection, while a compact coarse-to-fine neural network predicts the residual field. The network uses baseline-derived guida
    
[^60]: 负载均衡神经元：激活率控制维持持续学习中的可塑性

    Load Balancing Neurons: Firing-Rate Control Sustains Plasticity in Continual Learning

    [https://arxiv.org/abs/2611.32691](https://arxiv.org/abs/2611.32691)

    本文提出一种通过调整每个神经元的阈值来控制ReLU激活率的轻量级负载均衡机制，主动防止神经元死亡或过度激活，从而在持续学习中维持网络的可塑性。

    

    持续学习中的神经网络常常失去可塑性：一些神经元变得不活跃，而另一些神经元则几乎持续激活。这限制了对变化数据的适应能力，并浪费了网络容量。先前的工作通过定期重新初始化低效用单元来缓解这一问题，但这种重置可能会破坏之前学到的特征，且无法主动预防低效用问题的出现。我们研究了一种简单的诊断指标：ReLU单元的激活率，其定义为正预激活所占的比例。低激活率可以识别出“死亡”单元，而极高的激活率则表明单元已被线性化、始终处于开启状态。基于这一视角，我们引入了一种轻量级的负载均衡机制，通过调整每个神经元的阈值，将激活率保持在目标范围内。在持续ImageNet和类增量CIFAR-100实验中，激活率分布解释了包括L2正则化和非仿射（non-affine）等知名方法在内的一系列方法之间可塑性差异的原因。

    arXiv:2611.32691v1 Announce Type: new  Abstract: Neural networks in continual learning often lose plasticity: some neurons become inactive, while others fire almost constantly. This limits adaptation to shifting data and wastes capacity. Prior work mitigates this by periodically reinitializing low-utility units, but such resets can destroy previously learned features and do not proactively prevent low utility. We study a simple diagnostic measure: the firing rate of ReLU units, defined as the fraction of positive pre-activations. Low rates identify dead units, while very high rates indicate linearized, always-on units. Based on this view, we introduce a lightweight load-balancing mechanism that adjusts per-neuron thresholds to keep firing rates within a target range. Across continual ImageNet and class-incremental CIFAR-100, firing-rate distributions explain differences in plasticity across a range of approaches, including well-known techniques such as L2 regularization and non-affine 
    
[^61]: OpenWoud：无厂商锁定、可扩展的自适应随机森林加速器

    OpenWoud: Vendor-Unlocked Scalable Adaptive Random Forest Accelerator

    [https://arxiv.org/abs/2611.32662](https://arxiv.org/abs/2611.32662)

    OpenWoud是一个无厂商锁定的FPGA自适应随机森林加速框架，通过定点运算库替代浮点运算并引入完全透明的手动调度数据流，实现了细粒度的时间/面积权衡与跨平台可移植性。

    

    在线机器学习（ML）能够在不断演进的数据流上实现实时决策。这些动态模型持续学习以适应新概念，因为旧概念会逐渐失效，但其性能往往受到吞吐量的限制。自适应随机森林（Adaptive Random Forest, ARF）是这一场景下最先进的模型，它天然具备并行性，因此非常适合硬件加速。然而，现有的基于现场可编程门阵列（FPGA）的加速器存在厂商锁定问题，并且依赖资源消耗巨大的浮点（FP）运算，这限制了其可扩展性和可移植性。我们提出了OpenWoud，一个无厂商锁定的FPGA ARF框架。我们的设计用灵活的定点（Fixed-Point, FxP）运算库替代浮点算子，并为训练和漂移检测引入了完全透明的、手动调度的数据流。这使得能够进行细粒度的时间/面积权衡，并保证跨平台的可预测行为。（注：原摘要在此处被截断）

    arXiv:2611.32662v1 Announce Type: cross  Abstract: Online Machine Learning (ML) enables real-time decision-making on evolving data streams. These dynamic models learn continuously to adapt to new concepts, as previous ones become invalid, but their performance is often limited by throughput. The Adaptive Random Forest (ARF) is a state-of-the-art model for this setting, which naturally exposes parallelism and is therefore well-suited for hardware acceleration. However, prior Field Programmable Gate Array (FPGA)-based accelerators are vendor-locked and rely on resource-hungry Floating-Point (FP) arithmetics, limiting their scalability and portability. We present OpenWoud, a vendor-unlocked ARF framework for FPGAs. Our design replaces FP operators with a flexible FiXed-Point (FxP) arithmetic library and introduces fully transparent, manually scheduled dataflows for training and drift detection. This enables fine-grained time/area trade-offs and predictable behavior across platforms. A com
    
[^62]: LATEC：面向高维科学数据的学习式轴向张量压缩

    LATEC: Learned Axial TEnsor Compression for High-Dimensional Scientific Data

    [https://arxiv.org/abs/2611.32651](https://arxiv.org/abs/2611.32651)

    本文提出LATEC，一种基于GNN架构、维度无关的学习式误差界限压缩编解码器，可高效压缩高维科学数据且无需针对不同维度进行大量适配或重新训练。

    

    随着科学模拟与基础设施规模的不断增长，部分负担已经从生成科学数据转移到了分析和存储科学数据上。压缩是扩展存储容量的关键步骤，但需要专门的技术来强制执行用户指定的逐点误差界限，以确保下游科学工作流的有效性。为了进一步突破压缩的极限，人们开发了新颖的学习式编解码器，尤其是针对图像和视频的编解码器，但这些方法通常以聚合失真为优化目标，无法满足误差界限标准。此外，这些方法通常针对固定的场维度，未能涵盖许多科学数据的使用场景，或者需要付出巨大的适配或重新训练成本。在这项工作中，我们提出了学习式轴向张量压缩方法，这是一种基于GNN架构、维度无关的学习式预测器的误差界限编解码器。LATEC经过预训练

    arXiv:2611.32651v1 Announce Type: new  Abstract: With the growing scale of scientific simulation and infrastructure, part of the burden has shifted from producing scientific data to analyzing and storing it. Compression is a key step to expanding storage capacity, but requires dedicated techniques to enforce user-specified pointwise error bounds that ensure the validity of downstream scientific workflows. To further push the limits of compression, novel learned codecs have been developed, in particular for images and videos, but these methods often optimize for aggregated distortion and fail to meet error bound standards. Moreover, these methods often target fixed field dimensionality, failing to account for many scientific data use cases, or at the expense of large adaptation or retraining costs. In this work, we introduce the Learned Axial TEnsor Compression (LATEC) method, an error-bounded codec with a dimension-agnostic learned predictor based on a GNN architecture. LATEC is pretra
    
[^63]: TeamFlowRL：面向多智能体协作的去中心化流策略在线策略优化

    TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for Multi-Agent Coordination

    [https://arxiv.org/abs/2611.32645](https://arxiv.org/abs/2611.32645)

    TeamFlowRL提出了一种两阶段框架，通过在去中心化流策略的完整去噪轨迹上构建团队策略优化目标（避免难以计算的动作似然），实现多智能体流策略的离线预训练与在线微调，在51个配对配置中平均提升47.3%。

    

    复杂的协作任务需要协调一致的多智能体行为。离线生成式策略能够从轨迹中学习富有表现力的协作行为，但受限于固定的数据集；在线强化学习则可以通过与环境交互来改进这些策略。然而，由于动作似然难以计算，流策略的微调十分困难。我们提出了TeamFlowRL，这是一个两阶段框架，先离线预训练多智能体流策略，再对其进行在线微调。我们将团队策略优化建模在去中心化流策略的完整去噪轨迹之上，从而避免了难以处理的动作似然计算。所得的目标函数使用共享的团队优势对每个智能体的轨迹比率进行加权，并分别对每个比率进行裁剪，使局部策略更新与共同的团队回报保持一致。在七个仿真组的51个配对配置中，TeamFlowRL改进了所有离线初始化策略，平均提升47.3%（摘要原文在此处截断）。

    arXiv:2611.32645v1 Announce Type: cross  Abstract: Complex cooperative tasks require coordinated multi-agent behavior. Offline generative policies learn expressive cooperative behaviors from trajectories but remain limited by fixed datasets. Online reinforcement learning can improve them through environment interaction. However, fine-tuning flow policies is difficult because action likelihoods are hard to compute. We propose TeamFlowRL, a two-stage framework that pretrains multi-agent flow policies offline and fine-tunes them online. We formulate team policy optimization over the complete denoising trajectories of decentralized flow policies, avoiding intractable action likelihoods. The resulting objective weights each agent's trajectory ratio with a shared team advantage and clips each ratio separately, aligning local policy updates with the common team return. Across 51 paired configurations in seven simulation groups, TeamFlowRL improves every offline initialization, with a 47.3% av
    
[^64]: HiCal：面向精确HiF4量化的分层尺度编码离线校准方法

    HiCal: Offline Calibration for Hierarchical Scale Encoding in Accurate HiF4 Quantization

    [https://arxiv.org/abs/2611.32624](https://arxiv.org/abs/2611.32624)

    提出HiCal离线校准方法，通过为E6M2尺度引入校准舍入偏置、并基于局部组幅值经验分布优化尺度扩展阈值，从三个尺度级别全面提升HiF4分层量化精度。

    

    块浮点（BFP）量化被广泛应用于高效低比特大语言模型（LLM）推理，其中块尺度的选择对量化精度起着至关重要的作用。HiFloat4（HiF4）采用了一种分层尺度编码方案，由全局E6M2尺度和两级二进制尺度扩展组成。然而，其传统的舍入规则和固定的尺度扩展阈值可能无法充分捕捉实际中量化张量的统计特性，从而导致次优的尺度表示。在这项工作中，我们提出了HiCal，这是一种离线校准方法，通过细化所有三个尺度级别的编码决策来提升HiF4量化精度。具体而言，我们为Level-1的E6M2尺度引入了校准舍入偏置，并基于局部组幅值的经验分布优化了Level 2和Level 3的尺度扩展阈值。由此产生的校准……

    arXiv:2611.32624v1 Announce Type: new  Abstract: Block floating-point (BFP) quantization is widely adopted for efficient low-bit large language model (LLM) inference, where the choice of block scales plays a critical role in quantization accuracy. HiFloat4 (HiF4) employs a hierarchical scale encoding scheme consisting of a global E6M2 scale and two levels of binary scale extensions. However, its conventional rounding rule and fixed scale extension thresholds may not fully capture the statistical characteristics of the quantized tensors in practice, leading to suboptimal scale representations. In this work, we propose HiCal, an offline calibration approach that improves HiF4 quantization accuracy by refining the encoding decisions across all three scale levels. Specifically, we introduce a calibrated rounding bias for the Level-1 E6M2 scale and optimize the scale-extension thresholds at Levels 2 and 3 based on the empirical distributions of local group magnitudes. The resulting calibrat
    
[^65]: ST-GNN-AAAS：面向旅行商问题中随时自动算法选择的时空图神经网络

    ST-GNN-AAAS: Spatio-Temporal Graph Neural Networks for Anytime Automatic Algorithm Selection in the Traveling Salesman Problem

    [https://arxiv.org/abs/2611.32602](https://arxiv.org/abs/2611.32602)

    提出了一种时空图神经网络框架ST-GNN-AAAS，通过融合当前最优路径的时变节点特征与求解器短期性能历史来预测未来的求解器选择，实现了旅行商问题中随不同时间预算变化的自动算法选择。

    

    随时自动算法选择旨在根据不同的时间预算来做出求解器选择。现有的AAAS方法主要依赖静态的实例特征，对求解器执行过程中所揭示的时间信息利用有限。我们提出了ST-GNN-AAAS，一个用于旅行商问题（TSP）中AAAS的时空图神经网络框架。该框架使用随时间变化的节点特征来表示不断演化的当前最优路径，包括每个节点到当前路径中其相邻节点的距离，并将这些特征与短时间历史窗口内的求解器性能观测相结合。基于这段短历史信息，ST-GNN-AAAS可以预测更长未来时段内的求解器选择。我们在来自NATIONAL、TNM、TSPLIB和VLSI的多样化TSP基准测试子集上评估了该方法，并发布了相应的求解器执行数据以支持可复现性。我们的模型达到了76.59的选择准确率。

    arXiv:2611.32602v1 Announce Type: new  Abstract: Anytime Automatic Algorithm Selection (AAAS) aims to make solver choices under varying time budgets. The existing AAAS approaches largely rely on static instance characteristics and make limited use of temporal information revealed during solver execution. We propose ST-GNN-AAAS, a Spatio-Temporal Graph Neural Network framework for AAAS in the Traveling Salesman Problem (TSP). The framework represents evolving incumbent tours using time-varying node features, including distances from each node to its neighbors in the current tour, and combines these features with solver performance observations over a short history window. Given this short history, ST-GNN-AAAS predicts solver selections over a longer future horizon. We evaluate the method on a subset of diverse TSP benchmarks from NATIONAL, TNM, TSPLIB, and VLSI, releasing the corresponding solver execution data to support reproducibility. Our model achieves a selection accuracy of 76.59
    
[^66]: 门控概念适配器：用于可审计文本理解的分层概念瓶颈

    Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding

    [https://arxiv.org/abs/2611.32600](https://arxiv.org/abs/2611.32600)

    提出了门控概念适配器（GCA）这一分层概念瓶颈框架，通过在预训练Transformer的选定层附加门控低维概念读出适配器，将隐藏状态转化为可检查、可干预的概念中介接口，从而实现可审计的文本理解。

    

    面向人类的文本理解系统被用于推断情感、审核有害内容以及支持社会敏感的决策。然而，Transformer的预测难以审计，因为其预测产生于高维隐藏状态，无法为人类分析或干预提供紧凑、可检查的接口。我们提出了门控概念适配器（Gated Concept Adapters, GCA），这是一种分层概念瓶颈框架，它将带门控的、低维的概念生成读出适配器附加到预训练Transformer的选定输出上。词元表示被映射为概念激活，经学习到的门控过滤，汇聚为层级别的摘要，并作为概念中介接口暴露给预测头。这些维度可以通过使用统计、特征性n-gram、事后概念命名以及针对性的概念干预来进行检查。我们在细粒度情感分类、毒性检测等任务上对GCA进行了评估……

    arXiv:2611.32600v1 Announce Type: cross  Abstract: Human-facing text understanding systems are used to infer affect, moderate harmful content, and support socially sensitive decisions. However, Transformer predictions are difficult to audit because they are produced from high-dimensional hidden states that do not provide a compact, inspectable interface for human analysis or intervention. We introduce Gated Concept Adapters (GCA), a layer-wise concept bottleneck framework that attaches gated, low-dimensional concept-producing readout adapters to selected outputs of pretrained Transformers. Token representations are mapped into concept activations, filtered by learned gates, pooled into layer-level summaries, and exposed to the prediction head as a concept-mediated interface. The dimensions can be inspected through usage statistics, signature n-grams, post-hoc concept names, and targeted concept interventions. We evaluate GCA on fine-grained emotion classification, toxicity detection, t
    
[^67]: 关于重采样在监督式图异常检测中有效性的研究

    On the Effectiveness of Resampling for Supervised Graph Anomaly Detection

    [https://arxiv.org/abs/2611.32598](https://arxiv.org/abs/2611.32598)

    该研究系统评估了监督式图异常检测中的重采样策略，发现对自然发生的异常没有任何过采样方法能带来统计显著的收益，高斯噪声过采样仅对注入型属性交换异常有效，且标注哪些节点比选择何种采样策略更为关键。

    

    类别不平衡和异常标签稀缺促使监督式图异常检测（GAD）中广泛采用过采样技术，但其在不同检测器和基准上的实际收益仍不明确。在GADBench的低标签设置下（仅20个带标签异常节点和80个带标签正常节点），我们在17个数据集上，针对来自三大家族——图神经网络（GNN）、基于邻域聚合特征的树集成模型以及表格基础模型——的十种检测器配置，比较了类别加权与多达六种过采样策略。在八个包含自然发生异常的数据集上，没有任何策略在任何配置下展现出具有统计显著性的平均增益，而且哪些节点被标记与所采用的策略同等重要。在九个包含注入异常的数据集上，高斯噪声过采样（ROSE）在六个树模型和表格基础模型配置中的四个里最高可提升5.2个AUPRC点，且在所有六个配置中仅在属性交换类异常上有效；而在四个GNN配置中，其增益……（原文摘要在此处截断）

    arXiv:2611.32598v1 Announce Type: new  Abstract: Class imbalance and scarce anomaly labels motivate oversampling in supervised graph anomaly detection (GAD), but its benefits across detectors and benchmarks remain unclear. In the low-label setting of GADBench, with 20 labelled anomalies and 80 labelled normal nodes, we compare class weighting and up to six oversampling strategies in ten detector configurations from three families: graph neural networks (GNNs), tree ensembles on neighbourhood-aggregated features, and tabular foundation models, on 17 datasets. On the eight datasets with organic anomalies, no strategy has a statistically resolved average gain in any configuration, and which nodes are labelled matters as much as the strategy. On the nine datasets with injected anomalies, Gaussian-noise oversampling (ROSE) gains up to 5.2 AUPRC points in four of the six tree and tabular foundation model configurations, and in all six only on attribute-swap anomalies; in four GNNs its gain i
    
[^68]: 具有求和聚合的局部MPNN的度数受限可满足性问题是NEXPTIME难的

    Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard

    [https://arxiv.org/abs/2611.32595](https://arxiv.org/abs/2611.32595)

    本文证明了具有局部求和聚合和ReLU激活的消息传递神经网络（MPNN）的度数受限可满足性判定问题是NEXPTIME难的，且即使度数上界固定为3，该难度结果仍然成立。

    

    我们证明了以下问题是NEXPTIME难的：给定一个具有局部求和聚合和ReLU激活函数的消息传递神经网络（MPNN）$M$，以及一个以一元编码表示的度数上界$d$，判断是否存在一个带有有理数节点标签且度数至多为$d$的指向图$(G,v)$，使得该图能被$M$接受。即使在固定的度数上界$d=3$的情况下，该难度结果依然成立。

    arXiv:2611.32595v1 Announce Type: cross  Abstract: We prove that the following problem is \textbf{NEXPTIME}-hard: given a message passing neural network (MPNN) $M$ with local sum aggregation and ReLU activations, and a degree bound $d$ encoded in unary, decide whether there is a pointed graph $(G,v)$ with rational node labels and degree at most $d$ that is accepted by $M$. Hardness already holds for the fixed degree bound $d=3$.
    
[^69]: 通过对抗鲁棒性理解大规模激活的重要性

    Understanding the Importance of Massive Activations through Adversarial Robustness

    [https://arxiv.org/abs/2611.32585](https://arxiv.org/abs/2611.32585)

    本文将大规模激活与延迟鲁棒性中的区域迁移现象联系起来，证明大规模激活通过将表示推离网络非线性边界来降低输入输出映射的复杂性，并首次发现该现象并非Transformer独有，也普遍存在于其他常见架构中。

    

    在大型语言模型（LLMs）中，一个常见的现象是所谓的大规模激活（massive activations）的出现。这一现象在文献中广为人知，并与注意力汇聚、表示压缩以及信号传播相关联。然而，其出现的原因和后果尚未得到很好的理解。在本文中，我们通过证明大规模激活与延迟鲁棒性中的区域迁移现象相关联，解释了大规模激活在深度学习中的作用。在深度网络的样条理论解释下，我们表明大规模激活将表示推离网络的非线性边界，降低了数据点周围输入输出映射的复杂性。此外，我们首次发现这一现象并非Transformer所独有，实际上也出现在许多执行简单监督学习任务的常见架构中。

    arXiv:2611.32585v1 Announce Type: new  Abstract: A common occurrence in Large Language Models (LLMs) is the emergence of so-called massive activations. This phenomenon is well known in the literature and has been connected to attention sinks, representational compression, as well as signal propagation. However, the reasons and consequences for their appearance are not yet well understood. In this paper, we explain the role of massive activations in deep learning by showing that they are connected to the region migration phenomenon in delayed robustness. Under the spline theoretic interpretation of deep networks, we show that massive activations push the representations away from the nonlinearity boundaries of the network, decreasing the complexity of its input-output map around data points. Furthermore, we uncover for the first time that this phenomenon is not unique to transformers and actually appears in a number of common architectures performing simple supervised learning tasks. We
    
[^70]: 蒸炼不确定性：面向两阶段随机优化的决策引导进化策略深度生成式情景搜索

    Boiling Down Uncertainty: Deep Generative Scenario Search with Decision-Guided Evolution Strategies for Two-Stage Stochastic Optimization

    [https://arxiv.org/abs/2611.32573](https://arxiv.org/abs/2611.32573)

    本文通过补偿对偶性与伴随求解推导出闭式梯度，为深度生成式情景搜索补齐了解析理论基础，并将决策聚焦方向作为引导进化策略的采样子空间，保证期望更新为平滑目标的下降方向，从而改进两阶段随机优化的情景生成。

    

    两阶段随机规划是处理不确定性最常用的模型之一。选择合适的情景数量需要在可表征的不确定性与计算成本之间进行权衡。近期提出的一种方法——深度生成式情景搜索，通过生成合成情景，并从这些情景所诱导的决策中学习来改进情景质量。由于决策在情景之间是分段常数的，原始方法不得不使用黑盒梯度估计。本文通过借助补偿对偶性和正则化简化问题的伴随求解，推导出闭式形式的梯度，补齐了该方法缺失的解析理论基础。决策聚焦方向（当对偶变量不可用时以启发式方向作为后备）被用作引导进化策略的采样子空间，而非直接作为更新方向。对于任意引导子空间，期望更新都是平滑目标的下降方向，因此……

    arXiv:2611.32573v1 Announce Type: cross  Abstract: One of the most common models for handling uncertainty is two-stage stochastic programming. Choosing the right number of scenarios is a trade-off between the uncertainty that can be represented and computational cost. A recent method, Deep Generative Scenario Search, generates synthetic scenarios and improves them by learning from the decisions they induce. The original approach used a black-box gradient estimate, since decisions are piecewise constant across scenarios. This paper provides the missing analytical foundation by deriving the gradient in closed form from recourse duality and an adjoint solve of the regularized reduced problem. The decision-focused direction, with a heuristic direction as a backup when duals are unavailable, defines the sampling subspace of a guided evolution strategy rather than being used as a direct update. For any guiding subspace, the expected update is a descent direction of the smoothed objective, so
    
[^71]: 打破混合专家模型中的路由与设备间依赖

    Breaking Routing And Inter-device Dependencies in Mixture-of-Experts

    [https://arxiv.org/abs/2611.32572](https://arxiv.org/abs/2611.32572)

    提出BRAID-MoE架构，通过阶梯式残差连接与早期路由打破MoE模型中的顺序依赖，实现通信与注意力计算的深度重叠，缓解专家并行推理中令牌分发与聚合的通信瓶颈。

    

    随着混合专家模型规模的不断扩大，专家并行在将专家分布到多个设备上变得越来越重要。然而，由此产生的令牌分发与聚合操作可能成为推理过程中的主要瓶颈。将通信阶段与计算进行重叠会受到标准MoE架构顺序依赖关系的限制。我们提出了BRAID-MoE，这是一种将阶梯式残差连接与早期路由相结合以放宽这些依赖关系的架构。阶梯式残差使每个专家聚合操作可以延迟到下一个模块执行，而早期路由则基于聚合前的残差计算下一次路由决策，同时专家仍接收更新后的聚合后隐状态。这创造了一个延长的通信窗口，使前一个聚合操作和下一个分发操作可以在专用通信流上背靠背执行，并与注意力计算重叠。（原文摘要于此处截断）

    arXiv:2611.32572v1 Announce Type: cross  Abstract: As Mixture-of-Experts (MoE) models scale, Expert Parallelism (EP) becomes increasingly important for distributing experts across devices. However, the resulting token dispatch and combine operations can become a major bottleneck during inference. Overlapping these communication phases with computation is constrained by the sequential dependencies of standard MoE architectures. We introduce BRAID-MoE, an architecture that pairs laddered residual connections with early routing to relax these dependencies. Laddered residuals enable each expert combine to be deferred to the following block, while early routing computes the next routing decision from the pre-combine residual, with experts still receiving the updated post-combine hidden state. This creates an extended communication window in which the preceding combine and next dispatch can execute back-to-back on a dedicated communication stream, overlapping attention computation. On Qwen3-
    
[^72]: 小分子基准测试中分子表示与模型的贡献：受控比较与外部评估

    Representation and Model Contributions on Small Molecular Benchmarks: Controlled Comparisons and External Evaluation

    [https://arxiv.org/abs/2611.32568](https://arxiv.org/abs/2611.32568)

    该研究通过二乘二因子设计与配对支架自助法，将分子表示与模型架构的贡献分离，发现在溶解度和亲脂性等任务上，分子表示（如RDKit描述符）带来的性能提升甚至超过更换模型本身的效果。

    

    分子性质预测的性能提升通常被归功于模型本身，尽管其中部分收益可能来自分子表示。新模型通常与在分子指纹上训练的随机森林（RF）进行基准对比，这种比较方式同时改变了模型和表示两个因素。我们通过二乘二的因子设计将这两种贡献分离：随机森林与第二个模型（CatBoost 或表格基础模型 TabPFN 和 TabFM）分别在同一分子指纹和 RDKit 描述符上训练，并使用配对支架自助法检验各因素的效应。我们在七个小型 MoleculeNet 数据集（3 个回归、4 个分类）上，于近期图 Transformer（DGT）已发表的数据划分、另外 10 个支架划分以及外部数据集上进行评估。对于溶解度和亲脂性，在所有主要因子设计中、在所有三种评估中，表示效应均超过模型效应……（摘要在此处截断）

    arXiv:2611.32568v1 Announce Type: new  Abstract: Improvements in molecular property prediction are usually attributed to the model, although part of the gain may come from the molecular representation. New models are commonly benchmarked against a random forest (RF) trained on a molecular fingerprint, a comparison that changes the model and the representation at the same time. We separate the two contributions with a two-by-two factorial design in which the RF and a second model (CatBoost or the tabular foundation models TabPFN and TabFM) are each trained on a fingerprint and on RDKit descriptors, and each effect is tested with a paired scaffold bootstrap. Seven small MoleculeNet datasets (3 regression, 4 classification) are evaluated on the published split of a recent graph transformer (DGT), on 10 additional scaffold splits and on external datasets. For solubility and lipophilicity, the representation effect exceeds the model effect in every main factorial design, in all three evalua
    
[^73]: 面向生成模型的接触几何：一种非平衡最优传输形式化方法

    Contact Geometry for Generative Models: An Unbalanced Optimal Transport Formulation

    [https://arxiv.org/abs/2611.32547](https://arxiv.org/abs/2611.32547)

    该论文提出基于接触几何的非平衡最优传输新方法 CuOT，通过将密度传输方向与质量增长解耦，消除了 WFR 几何中偏向高密度区域的结构性偏差，从而避免模式坍塌并实现稳定、可扩展的生成模型求解。

    

    现实世界中生物学、医学成像或机器人技术等领域的生成问题，很少能提供完美配对的源分布和目标分布。虽然非平衡最优传输（uOT）通过在概率空间中寻找最小长度路径，为桥接非配对数据集提供了一个稳健的框架，但现有的基于 Wasserstein-Fisher-Rao（WFR）几何的公式化方法会引入一种朝向高密度区域的结构性偏差，常常导致模式坍塌。我们提出了接触非平衡最优传输，这是一种基于接触几何的新型非平衡最优传输公式化方法，它将密度传输方向与质量增长解耦，从而消除结构性偏差。这使得 CuOT 能够根据局部数据结构自适应地调整采样步长，在高密度区域自然地减小步长以防止过冲，并确保稳定收敛。其结果是一个稳定、可扩展的路径长度最小化求解器，在生物学应用上持续优于基于 WFR 的方法。

    arXiv:2611.32547v1 Announce Type: new  Abstract: Real-world generative problems in biology, medical imaging, or robotics, rarely come with perfectly paired source and target distributions. While unbalanced Optimal Transport (uOT) provides a robust framework to bridge unpaired datasets via minimum-length paths in probability space, existing formulations based on the Wasserstein-Fisher-Rao (WFR) geometry introduce a structural bias toward high density regions, often triggering mode collapse. We introduce Contact unbalanced Optimal Transport (CuOT), a novel uOT formulation grounded in contact geometry that decouples density transport direction from mass growth to eliminate structural bias. This allows CuOT to adapt sampling steps to the local data structure, naturally reducing step sizes in high-density regions to prevent overshooting and ensuring stable convergence. The result is a stable, scalable path-length minimization solver that consistently outperforms WFR-based methods on biologi
    
[^74]: 超越鲁棒算法追索：通过模型多重性分担有效性负担

    Beyond Robust Algorithmic Recourse: Sharing the Burden of Validity through Model Multiplicity

    [https://arxiv.org/abs/2611.32543](https://arxiv.org/abs/2611.32543)

    该论文揭示了鲁棒算法追索虽能保证建议有效性但会因偏向数据高密度区域而造成系统性不公平，并提出利用模型多重性下的策略性模型选择来分担有效性负担，从而在放宽个体鲁棒性要求的同时维持追索的有效性。

    

    算法追索为个体提供可操作的建议，使其能够获得有利的结果。为个体实施追索需要大量的时间和精力，因此强调了鲁棒追索的重要性：即使在未来的模型更新下仍然有效的建议。虽然鲁棒性能够保持有效性，但我们证明它会引入系统性伤害，因为它将可行的建议限制在不成比例地反映多数人行为的区域，降低了个体灵活性并放大了差异。因此，非鲁棒追索面临失效风险，而鲁棒追索则倾向于数据中的高密度区域，从而引发不公平。然而，这种权衡是由一个隐含假设驱动的，即个体应当独自承担有效性的负担。我们挑战了这一观点，并证明策略性的模型选择可以在放宽个体鲁棒性要求的同时帮助维持有效性。

    arXiv:2611.32543v1 Announce Type: new  Abstract: Algorithmic recourse provides actionable recommendations that enable individuals to achieve favourable outcomes. Implementing recourse for individuals requires significant time and effort, thus emphasizing robust recourse: recommendations that remain valid even under future model updates. While robustness preserves validity, we show it introduces systematic harms by constraining feasible recommendations to regions that disproportionately reflect majority behavior, reducing individual flexibility and amplifying disparities. Thus, non-robust recourse risks invalidation, whereas robust recourse creates a preference for high-density regions in the data that induces unfairness. However, this trade-off is driven by an implicit assumption that individuals alone should bear the burden of validity. We challenge this narrative and show that strategic model selection can help maintain validity while relaxing robustness requirements for individuals.
    
[^75]: 基于混合特征提取的先进合成图像检测方法

    Advanced Synthetic Image Detection Using Hybrid Feature Extraction

    [https://arxiv.org/abs/2611.32538](https://arxiv.org/abs/2611.32538)

    本文提出一种混合特征提取的合成图像检测方法，结合CBAM、ResNet50、DCT评分和SRM模块，通过捕捉生成模型难以复制的细粒度纹理与复杂物体特征来有效区分真实图像与AI生成图像。

    

    在生成式人工智能领域，GAN、MidJourney、DALL-E 等模型不断演进，能够生成日益逼真的图像。这一快速发展使得人们迫切需要稳健的方法来区分真实图像与 AI 生成的图像。本文提出了一种利用深度学习技术进行特征分析的合成图像检测新方法。所提出的架构通过识别生成模型难以复制的差异来对图像进行分类，尤其是细粒度纹理和复杂物体特征。借助 CBAM 架构与 ResNet50 对复合区域中的形变进行检测。我们采用离散余弦变换（DCT）评分进行了基于图像块的选择性分析，提取出复杂度最高和最低的区域，再利用 SRM 模块和 ResNet50 模块进行进一步处理。实验结果表明，我们的方法……

    arXiv:2611.32538v1 Announce Type: cross  Abstract: In the realm of generative AI, models such as GANs, MidJourney, DALL-E, and others have been continuously evolving to generate increasingly lifelike images. This rapid advancement has created an urgent need for robust methods to distinguish between real and AI-generated images. This paper proposes a novel approach for the detection of synthetic images by employing deep learning techniques for feature analysis. The proposed architecture classifies images by identifying variations that generative models struggle to replicate, particularly fine-grained textures and complex object features. Deformations in complexes are examined with the help of the CBAM architecture along with ResNet50. We have performed a selective patch-based analysis using Discrete Cosine Transform (DCT) scoring, extracting the regions with the most and least complexity for further processing with an SRM Module and ResNet50 module. Experimental results show that our ap
    
[^76]: 分支与整合：多教师在线策略蒸馏的统一配方

    Branch and Consolidate: A Unified Recipe for Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2611.32515](https://arxiv.org/abs/2611.32515)

    该论文提出了一套统一的多模态多教师在线策略蒸馏配方，能够将大幅偏离共享检查点的专家模型重新整合回单一学生模型，使其在各个领域接近专家模型性能并超越感知专家。

    

    从共享检查点后训练得到的专家模型会习得互补的能力，而多教师在线策略蒸馏（MOPD）可以将这些能力整合到单一模型中。然而，我们发现现有的整合方法（从权重合并到MOPD）在专家模型与共享检查点产生显著偏离时效果不佳，而以往研究主要考察的是与检查点保持接近的专家模型。我们提出了一套用于多模态MOPD的统一配方，可将这些专家模型重新整合回它们最初分支出的那个单一学生模型中。该配方明确了整合所需的初始化、数据选择、跨教师的均衡监督以及现有能力的保持，并辅以针对性的精炼阶段，用于提升仍然落后于对应专家模型的能力。在一个包含六个专家模型的工业场景中，学生模型在每个专家模型自身的领域内均能接近其表现，并在感知领域超越了感知专家模型。

    arXiv:2611.32515v1 Announce Type: cross  Abstract: Specialists post-trained from a shared checkpoint acquire complementary capabilities, and multi-teacher on-policy distillation (MOPD) can consolidate them into a single model. However, we find that existing consolidation methods, from weight merging to MOPD, fall short when specialists diverge substantially from the shared checkpoint, whereas prior studies mostly examine specialists that remain close to it. We present a unified recipe for multimodal MOPD that consolidates such specialists back into the single student they branched from. The recipe specifies initialization, data selection, balanced supervision across teachers, and preservation of existing capabilities for consolidation, together with a targeted refinement stage for capabilities that still lag behind their specialists. In an industrial setting with six specialists, the student closely approaches each specialist in its own domain and surpasses the perception specialist. I
    
[^77]: 用于预测核反应堆堆芯特性的联邦图神经网络分析

    Analysis of Federated Graph Neural Networks for Predicting Nuclear Core Properties

    [https://arxiv.org/abs/2611.32509](https://arxiv.org/abs/2611.32509)

    本文证明在联邦学习框架下，图神经网络因其无需填充且不假设平移不变性而更契合核反应堆圆柱形堆芯的几何结构，从而在堆芯优化任务中优于CNN基线。

    

    由于对数据隐私和安全性的日益关注，联邦学习变得越来越受欢迎。它允许多方在不交换原始数据的情况下协同训练模型。作为具有国家重要性的高度敏感基础设施的典型例子，核反应堆受益于这种方法——尽管汇集来自不同机构的大型数据集可以改进机器学习解决方案，但数据集的保密性至关重要。在本文中，我们证明了在核反应堆堆芯优化的应用场景中，图神经网络（GNN）在联邦学习设置下的表现优于卷积神经网络（CNN）基线。GNN架构更适合核反应堆的几何结构，因为与CNN不同，它不需要对圆柱形堆芯进行填充，也不假设平移不变性。这项工作的关键发现是输入表示更为重要……

    arXiv:2611.32509v1 Announce Type: cross  Abstract: Federated learning has grown in popularity owing to the increasing data privacy and safety concerns. It allows multiple parties to collaboratively train a model without exchanging their raw data. As an example of highly sensitive infrastructure of national importance, nuclear reactors benefit from this approach, where---while pooling multiple large datasets from different institutions could improve machine learning solutions---the confidentiality of the datasets is paramount. In this paper, we demonstrate that a graph neural network (GNN) outperforms the convolutional neural network (CNN) baseline in a federated setting for a use case concerning nuclear reactor core optimization. The GNN architecture is better suited for the nuclear reactor geometry as, unlike the CNN, it does not require padding the cylindrical core and does not assume translation invariance. The key finding of this work is that input representation is more important 
    
[^78]: InducedTFM：基于诱导注意力的次二次复杂度表格基础模型家族

    InducedTFMs: A Family of Sub-Quadratic Tabular Foundation Models Using Induced Attention

    [https://arxiv.org/abs/2611.32504](https://arxiv.org/abs/2611.32504)

    针对表格基础模型注意力机制随样本和特征数量呈二次方增长的计算瓶颈，本文提出 InducedTFM，用 Set Transformer 的诱导注意力替换二维注意力的两个阶段，在保留单元格级表示的同时将计算成本降为线性，并在与全注意力模型同等训练预算下取得相当的准确率。

    

    表格基础模型已成为梯度提升树的强劲竞争对手，能够通过上下文学习在单次前向传播中完成对新表格的预测。然而，其注意力机制在样本数量和特征数量上均呈二次方复杂度，这使得预训练只能局限于小型合成表格，而对大型表格的推理不得不依赖分块或子采样。行压缩模型虽然降低了特征维度的计算成本，但代价是丢弃了单元格级别的表示，并且在样本数量上仍然保持二次方复杂度。我们提出了 InducedTFM，它使用 Set Transformer 的诱导注意力替换了二维表格基础模型的两个注意力阶段，在保留单元格级别表示的同时，使其计算成本在样本和特征两个维度上均为线性。在与全注意力 TabICLv2 重新实现相匹配的训练预算下，InducedTFM 在 TabArena 以及一个包含更大数据集的基准上达到了与其相当的准确率，并明显优于核方法基线。

    arXiv:2611.32504v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) have emerged as a serious competitor to gradient-boosted trees, predicting on a new table in a single forward pass by in-context learning. However, their attention is quadratic in the number of samples and features, which caps pretraining at small synthetic tables and forces inference on large ones to rely on chunking or subsampling. Row-compression models reduce the feature-wise cost, but only by discarding the cell-level representation, and remain quadratic in the number of samples. We introduce InducedTFM, which replaces both attention stages of a bidimensional TFM with the induced attention of the Set Transformer, preserving cell-level representations while making its cost linear in both samples and features. Under a training budget matched to a full-attention TabICLv2 reimplementation, InducedTFM matches its accuracy on TabArena and on a benchmark of larger datasets, and clearly outperforms a kernel 
    
[^79]: 评估信息物理系统中持续深度强化学习对灾难性遗忘的鲁棒性

    Assessing Robustness to Catastrophic Forgetting in Continual Deep Reinforcement Learning for Cyber-Physical Systems

    [https://arxiv.org/abs/2611.32496](https://arxiv.org/abs/2611.32496)

    该论文在非平稳自主泊车环境中系统评估了三类强化学习算法（无模型同策略的PPO、无模型异策略的SAC和基于模型的Drama）抵抗灾难性遗忘的鲁棒性，以确定哪类算法更适合信息物理系统中的持续学习。

    

    随着强化学习（RL）在信息物理系统中的应用日益广泛，设计能够适应环境变化以保持安全性和可靠性的RL智能体变得越来越重要。本文研究了在非平稳环境中，不同类别的RL算法是否对遗忘先前习得行为（持续学习中的关键问题）具有更强的鲁棒性。我们在一个非平稳的自主泊车环境中评估了三类重要RL算法中代表性方法的鲁棒性，分别是近端策略优化（PPO）——一种无模型同策略算法、软演员-评论家（SAC）——一种无模型异策略算法，以及Drama——一种基于模型的算法。智能体必须学习在四种不同场景中成功泊车，这些场景依次呈现，泊车位朝向……

    arXiv:2611.32496v1 Announce Type: cross  Abstract: As the use of reinforcement learning (RL) becomes more widespread in cyber-physical systems, it becomes increasingly important to design RL agents that are able to adapt to changes in their environments in order to maintain safety and reliability. This paper addresses the question of whether different classes of RL algorithm are more robust against forgetting previously learned behaviours, a critical issue in continual learning, in non-stationary environments. We evaluate the robustness of representative examples of three important classes of RL algorithm in a non-stationary autonomous parking environment. These are Proximal Policy Optimization (PPO), which is a model-free on-policy algorithm, Soft-Actor-Critic (SAC), which is a model-free off-policy algorithm, and Drama, which is a model-based algorithm. The agents must learn to successfully park in four different scenarios, which are presented successively, with parking spaces orient
    
[^80]: 当知识追踪的排名依赖于可用历史时：对 ASSISTments 2009 与 2015 数据集的数据源修正与信息匹配审计

    When Knowledge-Tracing Rankings Depend on Available History: A Source-Correction and Matched-Information Audit on ASSISTments 2009 and 2015

    [https://arxiv.org/abs/2611.32489](https://arxiv.org/abs/2611.32489)

    该论文通过对 ASSISTments 2009 与 2015 数据集进行数据源修正与信息匹配审计，揭示了知识追踪模型的性能排名依赖于学习者可用历史的长度、来源与表示方式，为公平的模型基准比较提供了重要的方法学依据。

    

    知识追踪基准测试常常在比较对学习者先前作答记录拥有不同访问权限的模型。我们考察数据来源、历史长度以及表示方式如何影响这些比较。首先，我们将一个固定版本的 ASSISTments 2009 序列镜像与修正后的官方发布版本进行对比：在两个数据源中均出现的 4,148 名学习者中，有 817 名学习者的有序技能-作答序列存在差异，且该镜像包含的已标注事件也更少。其次，我们保留了此前在 2009 与 2015 数据集上开展的、按学习者划分互不重叠的模型评估，将探索性结果与针对 2015 数据集的修正性复现区分开来。最后，我们在 4,000 名全新的、互不重叠的 ASSISTments 2015 学习者样本上评估了三个模型，所有模型均被限定使用相同的单次事件或 20 次事件的先前作答窗口。在来自 582 名学习者的 20,592 条测试作答上，20 事件条件下的 Brier 分数分别为：基于计数的逻辑回归 0.17667、顺序感知逻辑模型 0.17389，以及 0.17……

    arXiv:2611.32489v1 Announce Type: new  Abstract: Knowledge-tracing benchmarks often compare models that have different access to a learner's previous responses. We examine how data provenance, history length, and representation affect those comparisons. First, we compare a pinned ASSISTments 2009 sequence mirror against the corrected official release. Of 4,148 learners represented in both, 817 have different ordered skill-response sequences; the mirror also contains fewer labeled events. Second, we retain earlier learner-disjoint model evaluations on the 2009 and 2015 datasets, distinguishing exploratory results from a corrective 2015 replication. Finally, we evaluate a new, nonoverlapping sample of 4,000 ASSISTments 2015 learners using three models constrained to the same one-event or 20-event prior-response window. On 20,592 test responses from 582 learners, the 20-event Brier scores were 0.17667 for count-based logistic regression, 0.17389 for an order-aware logistic model, and 0.17
    
[^81]: 教师就在其中：将循环语言模型自蒸馏至超越其最后一轮

    The Teacher Is Already Inside: Self-Distilling Looped Language Models Beyond Their Last Loop

    [https://arxiv.org/abs/2611.32452](https://arxiv.org/abs/2611.32452)

    该论文提出ELPSD方法，将循环语言模型从首轮循环外推至末轮得到的下一个token分布作为自蒸馏目标，在模型自身rollout上训练其最后一轮循环，突破了传统后训练方法无法改进循环模型最后一轮的瓶颈。

    

    循环语言模型对每个token多次重复应用同一个共享的Transformer堆栈，且每一轮循环都可以通过共享的语言模型头读取输出。对这类模型进行后训练十分困难：仅基于奖励的强化学习需要验证器，且每次响应只能从一个稀疏信号中学习；将奖励分散到各轮循环会导致每轮循环的熵坍缩；而像LoopOPD那样的跨循环自蒸馏方法只能训练中间循环，因为最后一轮循环无法从其自身的冻结副本中学习——在初始化时损失即为零。我们观察到，循环模型内部本身已经蕴含着一个比其最后一轮解码效果更好的策略：即从第一轮循环外推至最后一轮所得到的下一个token分布。ELPSD（外推循环策略自蒸馏，Extrapolated-Loop Policy Self-Distillation）将这一分布转化为自蒸馏目标：由模型的一个冻结副本计算该分布，并在模型自身的rollout上，将模型自身的最后一轮循环朝该目标进行蒸馏，使用精确的逆向……

    arXiv:2611.32452v1 Announce Type: new  Abstract: Looped language models apply one shared transformer stack several times per token, and every loop is readable through the shared language-model head. Post-training them is hard. Reward-only reinforcement learning needs a verifier and learns from one sparse signal per response; spreading the reward over loops collapses the entropy of every loop; and cross-loop self-distillation, as in LoopOPD, can only teach intermediate loops, because the last loop cannot learn from its own frozen copy: the loss is zero at initialisation. We observe that a looped model already contains a policy that decodes better than its last loop: the next-token distribution extrapolated from the first loop through the last. ELPSD (Extrapolated-Loop Policy Self-Distillation) turns this distribution into a self-distillation target: a frozen copy of the model computes it, and the model's own last loop is distilled towards it on its own rollouts, using the exact reverse 
    
[^82]: 看见答案，学会嵌入：从后见之明中学习通用多模态嵌入

    See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight

    [https://arxiv.org/abs/2611.32439](https://arxiv.org/abs/2611.32439)

    该论文提出HEME方法，在训练时将匹配的候选目标作为“后见之明”上下文来学习基于查询的表示，使模型在推理时无需额外推理计算即可生成高质量的通用多模态嵌入。

    

    通用多模态嵌入（UME）模型将异构的多模态输入映射到一个共享的嵌入空间中，其中成对相似度可支持多种下游任务。近期的推理驱动的UME方法使用中间推理为嵌入生成额外上下文，但通常会在推理阶段引入额外的计算开销。在本文中，我们探索将匹配的候选（即配对的目标）作为一种替代性的上下文来源，研究模型能否在训练时“看见答案”，而在推理时无需它也能“学会嵌入”。然而，直接以这种特权上下文为条件，可能会引入仅凭查询本身无法获得的候选特定信息，或者导致以候选为条件的查询表示向候选嵌入坍缩。为解决这一问题，我们提出了后见之明增强的多模态嵌入（HEME），它从这种候选上下文中学习基于查询的后见之明表示。HEME

    arXiv:2611.32439v1 Announce Type: cross  Abstract: Universal Multimodal Embedding (UME) models map heterogeneous multimodal inputs into a shared embedding space, where pairwise similarity supports diverse downstream tasks. Recent reasoning-driven UME approaches use intermediate reasoning to generate additional context for embedding, but often incur extra computation at inference. In this paper, we explore the matched candidate (i.e., the paired target) as an alternative source of context, asking whether a model can "see the answer" during training and "learn to embed" without it at inference. However, direct conditioning on this privileged context can introduce candidate-specific information unavailable from the query alone or cause the candidate-conditioned query representation to collapse toward the candidate embedding. To address this, we introduce Hindsight-Enhanced Multimodal Embeddings (HEME), which learns query-grounded hindsight representations from such candidate context. HEME
    
[^83]: 数据集偏移下图节点对称量化框架

    A Framework for Graph Node Symmetric Quantification under Dataset Shift

    [https://arxiv.org/abs/2611.32403](https://arxiv.org/abs/2611.32403)

    该论文提出了一种新的图节点量化框架，摆脱了对节点级监督和节点级分类器预测的依赖，仅需按类别分布标注的节点样本进行训练，即可在数据集偏移下直接估计样本中各类别的占比。

    

    图节点量化是一项监督学习任务，其目标是在存在某种形式数据集偏移的情况下，估计从图中抽取的任意未标记节点样本中不同类别的占比（普遍程度）。现有的图节点量化方法通常遵循标准的聚合范式，即利用图中带标签的节点训练一个中间节点分类器，然后通过聚合分类器对各个节点的预测结果来获得样本的估计类别分布，再借助某种考虑数据集偏移的调整策略对结果进行校正。在本工作中，我们提出了一种图节点量化框架，它同时摒弃了节点级别的监督——因为其训练阶段只需要按类别分布进行标注的节点样本——以及节点级别的预测——因为它直接估计类别的占比，而无需任何中间节点分类器。

    arXiv:2611.32403v1 Announce Type: new  Abstract: Graph node quantification is the supervised learning task of estimating the prevalence of different classes within an arbitrary sample of unlabelled nodes from a graph, subject to some form of dataset shift. Existing approaches to graph node quantification typically follow the standard aggregative paradigm, i.e., they use labelled nodes from the graph to train an intermediate node classifier, and obtain an estimated class distribution for the sample by aggregating the classifier's predictions for the individual nodes and then correcting the result via some adjustment strategy that accounts for dataset shift. In this work we propose a framework for graph node quantification that does away with both node-level supervision, since its training phase only requires node samples labelled by their class distribution, and node-level predictions, since it estimates class prevalence values directly, without the need of any intermediate node classif
    
[^84]: 最优传输遇见 t-SNE：OT-tSNE 实现无批次效应的整合可视化

    Optimal Transport Meets t-SNE: Integrated Visualization without Batch Effects in OT-tSNE

    [https://arxiv.org/abs/2611.32402](https://arxiv.org/abs/2611.32402)

    OT-tSNE 用熵正则化最优传输的全局耦合亲和度取代跨批次最近邻匹配，在单细胞数据整合可视化中有效消除批次效应并保持稳健性。

    

    可视化探索是理解数据集结构不可或缺的步骤。然而，在不同条件下收集的数据集之间存在的系统性差异（即批次效应）会掩盖数据的共同结构，并使其联合可视化变得复杂。现有方法通常依赖跨批次的最近邻关系来整合批次，但对于具有挑战性的跨批次几何结构，这些局部对应关系可能并不可靠。我们提出了最优传输 t-SNE（OT-tSNE），用基于熵正则化最优传输推导的全局耦合亲和度取代基于最近邻的匹配。受控合成实验表明，基于最近邻的方法可能出现退化，而最优传输亲和度和 OT-tSNE 嵌入则保持稳健。在七个真实的单细胞数据集上，我们将 OT-tSNE 与广泛的整合和可视化方法进行了基准比较，发现它实现了帕累托最优的（摘要在此处截断）。

    arXiv:2611.32402v1 Announce Type: new  Abstract: Visual exploration is an integral step in understanding the structure of datasets. However, systematic variation between datasets collected under different conditions, known as batch effects, can obscure shared structure and complicate their joint visualization. Existing methods often rely on cross-batch nearest-neighbor relations to integrate the batches. Yet, these local correspondences can be unreliable for challenging cross-batch geometries. We propose optimal-transport t-SNE (OT-tSNE), which replaces nearest-neighbor-based matching with globally coupled affinities derived from entropic optimal transport. Controlled synthetic experiments demonstrated that nearest-neighbor-based methods can deteriorate, while optimal-transport affinities and OT-tSNE embeddings remain robust. Across seven real single-cell datasets, we benchmarked OT-tSNE against a broad range of integration and visualization methods and found that it achieves a Pareto-
    
[^85]: 并非所有样本对都对齐：字幕引导的音视频对比学习

    Not All Pairs Align: Caption-Guided Audio-Visual Contrastive Learning

    [https://arxiv.org/abs/2611.32399](https://arxiv.org/abs/2611.32399)

    该论文提出了一种模型无关的对比学习目标CGAAL，利用模态特定字幕估计音视频间的语义一致性，并据此为正样本对加权，从而解决了画外音等因素导致的音视频内容不匹配问题。

    

    我们研究从共现模态中学习音视频表示的问题。多模态对比学习通常假设成对观测提供的是同一底层信息的共享视图，这对于在受限环境中收集的数据集可能成立，但并不总是如此。例如，由于画外音的存在，共现的音频与视觉信号在内容上可能存在显著差异。然而，标准对比学习目标将所有配对样本视为同等对齐。在这项工作中，我们提出了字幕引导的对齐感知损失（CGAAL），这是一种与模型无关的目标函数，利用特定模态的字幕来估计音频与视频之间的语义一致性。CGAAL使用这些一致性分数来为每个正样本对的贡献加权，并通过预热期和渐进阈值策略逐步引入这种对齐感知的训练信号。我们展示了……

    arXiv:2611.32399v1 Announce Type: cross  Abstract: We address the problem of learning audio-visual representations from co-occurring modalities. Multimodal contrastive learning typically assumes that paired observations provide shared views of the same underlying information. While this may be true for datasets collected in constrained environments, it does not always hold. For instance, co-occurring audio and visual signals can differ substantially in their content due to off-screen sounds. Nevertheless, standard contrastive objectives treat all paired samples as equally aligned. In this work, we propose Caption Guided Alignment Aware Loss (CGAAL), a model-agnostic objective that leverages modality-specific captions to estimate semantic agreement between audio and video. CGAAL uses these agreement scores to weight the contribution of each positive pair. A warm-up period and gradual thresholding strategy progressively introduce this alignment-aware training signal. We demonstrate the g
    
[^86]: 分形神经合成：基于复二次边界动力学的恒定存储过程式权重推导

    Fractal Neural Synthesis: Constant-Storage Procedural Weight Derivation via Complex Quadratic Boundary Dynamics

    [https://arxiv.org/abs/2611.32393](https://arxiv.org/abs/2611.32393)

    该论文提出分形神经合成（FNS）架构，用仅24字节、锚定于复二次分岔边界的分形坐标种子，通过过程式推导生成神经网络权重并证明了熵界，将存储占用从随参数量线性增长降低到常数级，适用于资源受限的边缘微控制器。

    

    传统神经网络将突触参数存储为显式张量，其持久化Flash占用随参数数量呈线性增长（O(W)），这限制了资源受限的边缘微控制器的应用。我们提出并系统评估了分形神经合成（FNS），这是一种过程式权重推导架构，用一个属于F_64^3、锚定在f_c(z) = z^2 + c的复二次分岔边界dM附近的恒定24字节坐标种子Theta = (c_x, c_y, ell)来替代隐藏层矩阵。我们引入了大逃逸半径（R = 256）平滑逃逸核，以及四种零开销几何采样孔径——笛卡尔网格、互素3x3三分法网格（Z/9Z）、phi-Vogel螺旋和多尺度调和三脚架——并证明了条件Kolmogorov复杂度界（K_Kolm ≤ 192 + c_0比特）和香农熵界。在放大倍数ζ ∈ [10^0, 10^8]范围内，C^1傅里叶基和格点噪声基在标准化后坍缩至秩约2.25，SHA-256 P……（摘要在此处截断）

    arXiv:2611.32393v1 Announce Type: new  Abstract: Conventional neural networks store synaptic parameters as explicit tensors whose persistent Flash footprint scales linearly (O(W)) with parameter count, constraining resource-limited edge microcontrollers. We formulate and systematically evaluate Fractal Neural Synthesis (FNS), a procedural weight derivation architecture that replaces hidden-layer matrices with a constant 24-byte coordinate seed Theta = (c_x, c_y, ell) in F_64^3 anchored near the complex quadratic bifurcation boundary dM of f_c(z) = z^2 + c. We introduce a large-bailout (R = 256) smooth escape kernel, four zero-overhead geometric sampling apertures--Cartesian Grid, coprime 3x3 Rule-of-Thirds (Z/9Z), phi-Vogel spiral, and multi-scale Harmonic Tripod--and prove conditional Kolmogorov (K_Kolm <= 192 + c_0 bits) and Shannon entropy bounds. Across magnifications zeta in [10^0, 10^8], C^1 Fourier and lattice-noise bases collapse to post-standardization rank ~2.25 and SHA-256 P
    
[^87]: 用于健康轨迹可扩展表型分析的潜类别混合模型变分自编码器

    Latent Class Mixed Model Variational Autoencoders for Scalable Phenotyping of Health Trajectories

    [https://arxiv.org/abs/2611.32391](https://arxiv.org/abs/2611.32391)

    提出LCMM-VAE，一种以类别特定线性混合模型的混合作为潜变量先验的深度生成模型，通过闭式变分EM算法，实现了对高维、不规则采样纵向健康轨迹的可扩展且临床可解释的聚类表型分析。

    

    arXiv:2611.32391v1 公告类型：交叉 摘要：对纵向健康轨迹进行聚类需要处理高维测量数据、不规则的就诊时间以及受试者层面的异质性，同时还要保留临床可解释的结构。现有方法只能部分解决这一问题：潜类别混合模型能够产生可解释的、由协变量驱动的划分，但无法扩展到超过少数几个标志物的情形；而深度时间序列聚类虽然具备可扩展性，却牺牲了这种结构，并且通常要求统一的时间网格。我们提出了潜类别混合模型变分自编码器（LCMM-VAE），这是一种深度生成模型，其潜变量先验是类别特定的线性混合模型的混合，因此每个受试者的潜轨迹可以分解为其所属类别的均值轨迹和个体特异性偏差。我们推导了一种变分EM算法，其中的类别归属责任、随机效应和混合模型参数均以闭式解形式更新，并提供了可辨识性保证……

    arXiv:2611.32391v1 Announce Type: cross  Abstract: Clustering longitudinal health trajectories requires handling high-dimensional measurements, irregular visit times, and subject-level heterogeneity while retaining clinically interpretable structure. Existing methods address this only partially: latent-class mixed models yield an interpretable, covariate-driven partition but do not scale past a handful of markers, while deep time-series clustering scales at the cost of that structure and generally requires a common time grid. We introduce the Latent Class Mixed Model Variational Autoencoder (LCMM-VAE), a deep generative model whose latent prior is a mixture of class-specific linear mixed models, so that each subject's latent trajectory decomposes into its class mean trajectory and a subject-specific deviation. We derive a variational EM algorithm whose class responsibilities, random effects and mixed-model parameters are updated in closed form, and provide identifiability guarantees un
    
[^88]: 进化以学习：面向模型自我改进的梯度对齐经验进化方法

    Evolve to Learn: Gradient-aligned experience evolution for model self-improvement

    [https://arxiv.org/abs/2611.32390](https://arxiv.org/abs/2611.32390)

    该论文揭示了一个关键问题：能提升测试时性能的经验并不一定值得内化到模型中，并提出以梯度对齐作为更可靠的经验筛选标准，据此构建了E2L自进化框架来实现模型的自我改进。

    

    自进化语言模型可以通过测试时交互、反思和提示优化持续获取有用经验，然而如何将这些经验转化为模型自身的能力以确保自我改进仍然具有挑战性。在线策略自蒸馏（OPSD）为将测试时信息内化到模型参数中提供了一种自然的机制，但它很大程度上假设所生成的经验是值得学习的。我们证明这一假设并不总是成立：能够提升测试时性能的经验可能提供有限甚至有害的训练信号。为了更好地刻画哪些经验应该被内化，我们系统地分析了跨多个任务和反思模型的多样化变异经验，发现基于梯度的对齐度是比即时的测试时性能提升更可靠的下游蒸馏收益指标。基于这一发现，我们提出了E2L，一种自进化方法框架。

    arXiv:2611.32390v1 Announce Type: cross  Abstract: Self-evolving language models can continuously acquire useful experience through test-time interaction, reflection, and prompt optimization, yet turning such experience into model itself to ensure self improvement remains challenging. On-policy self-distillation (OPSD) provides a natural mechanism for internalizing test-time information into model parameters, yet it largely assumes that the gener- ated experience is worth learning. We show that this assumption does not always hold: experiences that improve test-time performance may provide limited or even harmful training signals. To better characterize which experience should be inter- nalized, we systematically analyze diverse mutated experiences across multiple tasks and reflection models, and find that gradient-based alignment is a more reli- able indicator of downstream distillation gains than immediate test-time improve- ment. Based on this finding, we propose E2L, a self-evoluti
    
[^89]: AI能设计出更好的nnU-Net吗？一种从失败到设计规则的假设驱动架构发现循环

    Can AI Design a Better nnU-Net? A Hypothesis-Driven Architecture Discovery Loop from Failures to Design Rules

    [https://arxiv.org/abs/2611.32384](https://arxiv.org/abs/2611.32384)

    该论文提出一种假设驱动的架构发现循环，让LLM智能体从失败案例中诊断错误模式、提出可证伪假设并验证机制，最终提炼出两条设计规则，构建出比nnU-Net计算量更低、分割精度更高的3D医学图像分割网络LeanNeXt。

    

    神经架构搜索和LLM驱动的进化方法通常将评估简化为一个标量适应度分数，从而丢弃了那些本可以解释设计为何成功或失败的失败案例。我们提出了一种假设驱动的架构发现循环，其中LLM智能体诊断结构化的错误模式、提出可证伪的假设、在冻结的nnU-Net流水线下测试反事实对照实验，并将经过验证的机制提炼为可执行的“触发-行动”规则。在八轮搜索过程中，该循环为3D医学图像分割识别出了两条设计原则：第一，将计算资源从高分辨率解码器重新分配到中部编码器，可以在降低成本的同时保留细薄解剖结构的边界；第二，支撑受限的区域对比能够解决嵌套解剖腔室之间的错误，并通过依赖解剖结构的触发器来决定该规则何时适用。这些原则催生了LeanNeXt（相对于nnU-Net，其计算量为0.78倍MACs、训练步时间为0.92倍；性能提升+1.63 Dice）。

    arXiv:2611.32384v1 Announce Type: new  Abstract: Neural architecture search and LLM-driven evolution often reduce evaluation to a scalar fitness score, discarding failures that could explain why designs succeed or fail. We present a hypothesis-driven architecture discovery loop in which LLM agents diagnose structured error patterns, formulate falsifiable hypotheses, test counterfactual controls under a frozen nnU-Net pipeline, and distill validated mechanisms into executable trigger-to-action rules. Across eight search waves, the loop identifies two principles for 3D medical image segmentation. First, reallocating computation from high-resolution decoders to the mid-encoder preserves thin-anatomy boundaries while reducing cost. Second, support-confined region contrast addresses errors between nested anatomical compartments, with anatomy-dependent triggers selecting when the rule applies. These principles produce LeanNeXt (0.78 MACs and 0.92 training-step time relative to nnU-Net; +1.63
    
[^90]: 离散生成采样中的活动几何

    The Activity Geometry of Discrete Generative Sampling

    [https://arxiv.org/abs/2611.32380](https://arxiv.org/abs/2611.32380)

    该论文提出“活动几何”框架，将离散采样中逐步耦合的活动度分解为边际传输的最小代价与耦合超额项，并据此设计出无需重训、无需额外模型评估即可在最小活动度下实现指定边际分布并支持修正已生成token的MAFTC采样耦合方法。

    

    迭代式修正是扩散生成模型的一项关键能力，这也推动了大量研究试图将类似行为引入离散扩散模型与流模型。越来越多的工作通过允许在采样过程中修改先前已生成的token来实现这种修正，从而塑造了相继采样状态之间的耦合。尽管这种耦合自由度在采样器设计中扮演着日益重要的角色，但目前对其刻画仍然不够充分。我们提出“活动几何”，通过期望转移代价（我们称之为“活动度”）来研究这些有限步耦合。我们证明活动度可以分解为由诱导边际传输所要求的最小值，以及由耦合产生的超额项两部分。以该最小值为基准，我们提出最小活动度有限时间耦合（MAFTC），无需重新训练或额外的模型评估，即可在最小活动度下实现指定的边际分布。在连续时间极限下，……（原文摘要在此处截断）

    arXiv:2611.32380v1 Announce Type: new  Abstract: Iterative refinement is a key capability of diffusion-based generation and has motivated growing efforts to bring similar behavior to discrete diffusion and flow models. A growing line of work enables such refinement by allowing previously generated tokens to be revised during sampling, thereby shaping the coupling between successive sampling states. Despite its growing role in sampler design, this coupling freedom remains insufficiently characterized. We introduce Activity Geometry to study these finite-step couplings through expected transition cost, which we call activity. We show that activity decomposes into the minimum required by the induced marginal transport and an excess term arising from the coupling. Using this minimum as a reference, we propose Minimal-Activity Finite-Time Coupling (MAFTC) to realize prescribed marginals with minimum activity without retraining or additional model evaluations. In the continuous-time limit, t
    
[^91]: 公共交通可达性学习型空间代理模型的蜕变测试

    Metamorphic Testing of Learned Spatial Surrogates for Transit Accessibility

    [https://arxiv.org/abs/2611.32373](https://arxiv.org/abs/2611.32373)

    本文对用于近似公交可达性计算的学习型神经代理模型进行蜕变测试，发现平均预测误差不足以衡量模型可靠性——预测技能良好的模型仍可能无法通过基本连通性测试，而一种针对性干预可同时提升连通性测试通过率与预测准确性。

    

    公交网络设计需要评估大量候选方案，但精细的可达性计算过于缓慢，无法在优化循环内反复执行。因此，我们训练了一个神经代理模型来近似这些计算。给定一个候选公交网络以及关于轨道服务和周边机会的固定信息，该模型预测可达性的空间变化。我们在地理上留出的区域中将模型预测与基于路径规划计算的目标进行比较，并在受控输入变化下测试指定的定性关系。通过在构造网络上进行的参考计算，评估了那些在路径规划层面存在对应关系的蜕变关系的适用性。研究表明，具有正预测技能的模型仍可能无法通过基本的连通性测试。一种针对性的干预措施显著提高了两种连通性关系的通过率，同时预测技能也有所改善。因此，当学习型模型成为优化器的一部分时，仅依靠平均预测误差是不够的。

    arXiv:2611.32373v1 Announce Type: new  Abstract: Transit-network design requires evaluating many candidates, but detailed accessibility calculations are too slow to repeat inside an optimization loop. We therefore train a neural surrogate to approximate them. Given a candidate bus network and fixed information about rail service and nearby opportunities, it predicts the calculated spatial change in accessibility. We compare its predictions with routing-derived targets in geographically held-out regions and test specified qualitative relations under controlled input changes. Reference calculations on constructed networks assess the applicability of relations that have a routing-level counterpart. Models with positive predictive skill can still fail basic connectivity tests. One targeted intervention substantially increases pass rates for two connectivity relations while predictive skill also improves. Average prediction error is therefore insufficient when a learned model becomes an opt
    
[^92]: 从ELBO到极大似然：非线性混合效应建模中变分自编码器的收敛性分析

    From the ELBO to Maximum Likelihood: Convergence of Variational Autoencoders for Nonlinear Mixed-Effects Modeling

    [https://arxiv.org/abs/2611.32372](https://arxiv.org/abs/2611.32372)

    本文对用于非线性混合效应建模的VAE-NLME算法进行了收敛性分析，证明了其基于ELBO的版本无需精确训练编码器即可几乎必然收敛到ELBO的稳定点，且最终算法通过重要性加权充分统计量收敛到极大似然意义下的解。

    

    非线性混合效应模型是药物计量学中的核心统计模型，在药物研发中为剂量选择、个体化给药以及监管决策提供支持。从统计学角度看，NLME建模是一个边缘化似然问题。变分自编码器通过神经网络编码器对难以求解的后验分布进行摊销式近似来解决此类问题，并使得AI组件能够集成到NLME建模之中。本文对VAE-NLME算法进行了收敛性分析。首先，证明了其基于证据下界（ELBO）的版本在几乎必然意义下收敛到ELBO的稳定点集合，且无需假设编码器被精确训练。其次，最终的VAE-NLME算法通过基于梯度的优化方法在ELBO上训练编码器，并将其作为重要性加权充分统计量的建议分布。该算法几乎必然收敛到……（原文在此处截断）

    arXiv:2611.32372v1 Announce Type: new  Abstract: Nonlinear mixed-effects (NLME) models are the central statistical models of pharmacometrics, where they support dose selection, individualized dosing and regulatory decisions in drug development. Statistically, NLME modeling is a marginalized likelihood problem. Variational autoencoders (VAEs) solve such problems with a neural network encoder that amortizes the approximation of the intractable posterior distribution, and allow the integration of AI components into NLME modeling. This paper provides the convergence analysis of the VAE-NLME algorithm. First, its version for the evidence lower bound (ELBO) is proven to converge almost surely to the set of stationary points of the ELBO, without assuming that the encoder is trained exactly. Second, the final VAE-NLME algorithm trains the encoder on the ELBO by gradient-based optimization and uses it as proposal for importance-weighted sufficient statistics. It converges almost surely to the s
    
[^93]: 基于描述文本构建的药物-药物相互作用基准揭示了依赖于数据划分方式的分子表示排名

    A Description-Derived Drug-Drug Interaction Benchmark Reveals Split-Dependent Molecular Representation Rankings

    [https://arxiv.org/abs/2611.32371](https://arxiv.org/abs/2611.32371)

    该论文从DrugBank描述文本中构建了一个药物-药物相互作用预测基准，并发现分子表示方法的性能排名依赖于训练-测试划分方式——分子指纹在化学重叠度高的随机划分上最优，而图衍生表示在骨架不相交的划分上更具优势。

    

    药物-药物相互作用（DDI）预测具有重要的临床意义，并为分子表示学习带来了独特的挑战。与单分子性质预测不同，DDI预测要求预测器将两个分子的表示组合为一个关系性结果。DrugBank提供的是由策展人员撰写的DDI描述文本，而非现成的类别标签，因此必须首先从文本中推导出类别预测目标。我们通过将处理后的描述映射到六个源自文本的类别，构建了一个DDI药物对数据集，并定义了一个包含四种留出划分方式的基准，这些划分方式将训练集与测试集之间的化学重叠程度从随机药物对逐渐限制到骨架不相交的评估。随后，我们利用该基准，在五个随机种子和多种分类器头部设置下，对分子指纹、预训练序列表示和图衍生特征进行了比较。分子指纹在高重叠的随机划分上表现领先，而图衍生表示……（原文摘要在此处截断）

    arXiv:2611.32371v1 Announce Type: cross  Abstract: Drug-drug interaction (DDI) prediction is clinically important and poses a distinct challenge for molecular representation learning. Unlike single-molecule property prediction, DDI requires a predictor to compose two molecular representations into a relational outcome. DrugBank provides curator-authored DDI descriptions rather than ready-made categorical labels, so a categorical prediction target must first be derived from the text. We construct a dataset of DDI pairs by mapping processed descriptions to six text-derived categories and define a benchmark with four holdout regimes that restrict train-test chemical overlap from random pairs to scaffold-disjoint evaluation. We then use this benchmark to compare molecular fingerprints, pretrained sequence representations, and graph-derived features across five seeds and multiple classifier heads. Molecular fingerprints lead the high- overlap random split, whereas graph-derived representati
    
[^94]: CriticJEPA：世界模型是比优化器更好的评判者

    CriticJEPA: World Models Are Better Critics Than Optimizers

    [https://arxiv.org/abs/2611.32357](https://arxiv.org/abs/2611.32357)

    提出CriticJEPA，证明冻结的世界模型作为“评判者”比作为“优化器”更可靠，通过从无奖励轨迹学习目标条件的流候选方案，并在各层级用冻结模型推演进行前向选择，从而实现更稳健的视觉规划。

    

    视觉世界模型通过预测动作结果来实现目标导向的规划。基于梯度的规划器通过优化动作来降低预测的目标代价，但当执行效果不再提升时，持续的优化可能会利用预测误差来“钻空子”。比较不同方案需要具有信息量的相对评分，而改进方案则需要可靠的更新方向。在精确状态下的执行实验表明，一个冻结的分层世界模型能够对目标条件下的候选方案进行有效排序，即使它在经过长时间梯度搜索得到的输出方案之间的排序是不具信息量的。有效的规划需要将目标导向的候选方案与能够区分其结果的预测相结合。我们提出了CriticJEPA，它从无奖励轨迹中学习目标条件的流（flow）候选方案，并在每个层级上使用冻结的模型推演（rollouts）来从中进行选择。前向选择构成了一个完整的规划器；在选择之前，短暂的锚定更新会对候选方案进行细化。在视觉导航与操作（manipul...）任务上……

    arXiv:2611.32357v1 Announce Type: new  Abstract: Visual world models enable goal-directed planning by predicting action outcomes. Gradient-based planners optimize actions to lower predicted goal cost, but continued optimization can exploit prediction errors after execution stops improving. Comparing plans requires informative relative scores; improving them also requires reliable update directions. Exact-state execution reveals that a frozen hierarchical world model ranks goal-conditioned proposals even when its rankings among outputs of prolonged gradient search are uninformative. Effective planning pairs goal-directed candidates with predictions that distinguish their outcomes. We introduce CriticJEPA, which learns goal-conditioned flow proposals from reward-free trajectories and uses frozen rollouts to select among them at every hierarchy level. Forward selection forms a complete planner; short anchored updates refine candidates before selection. Across visual navigation and manipul
    
[^95]: 超越硬词元：面向多模态蛋白质生成的预测记忆

    Beyond Hard Tokens: Predictive Memory for Multimodal Protein Generation

    [https://arxiv.org/abs/2611.32347](https://arxiv.org/abs/2611.32347)

    该论文提出了PredMem——一种无需训练的自条件化递归机制，通过在迭代掩码解码的各步骤间传递未提交的软预测作为跨步记忆，解决了多模态蛋白质生成中收敛缓慢与重复性坍塌的问题。

    

    多模态蛋白质语言模型通过迭代式掩码解码，将序列与离散结构词元统一起来，为蛋白质联合设计提供了一种有前景的范式。然而，标准解码器在每次迭代时依赖于无记忆的掩码重置机制：只有稀疏的一小部分新提交的硬词元会被传递下去，而其余所有未确定的位点都被重置为相同的 [MASK] 嵌入。在开放式的协同生成场景中，由于没有任何模态能提供固定的外部锚点，这种无状态的转换限制了单步细化的深度，并使两条生成轨道都缺乏稳定的跨步软锚点，从而导致收敛缓慢或退化为重复性坍塌。我们观察到，未提交的预测会以显著的时间连贯性演化，而非无规则的噪声，据此提出了预测记忆，这是一种无需训练的自条件化递归机制，能够在解码步骤之间传递这些软性信念。在每一步，PredMem 会对原始的……

    arXiv:2611.32347v1 Announce Type: new  Abstract: Multimodal protein language models unify sequence and discrete structural tokens via iterative masked decoding, offering a promising paradigm for joint protein design. However, standard decoders rely on a memoryless mask reset at each iteration: only a sparse subset of newly committed hard tokens is carried forward, while all remaining unresolved sites are reset to an identical [MASK] embedding. In open-ended co-generation where neither modality provides a fixed external anchor, this stateless transition limits single-step refinement depth and deprives both tracks of stable cross-step soft anchors, leading to slow convergence or degenerative repetitive collapse. Observing that uncommitted predictions evolve with pronounced temporal coherence rather than erratic noise, we introduce Predictive Memory (PredMem), a training-free self-conditioning recurrence that carries these soft beliefs across steps. At each step, PredMem summarizes raw lo
    
[^96]: 当精确的词元重要性不够时：大语言模型注意力中Top-K选择的局限性

    When Exact Token Importance Is Not Enough: Limits of Top-K Selection in LLM Attention

    [https://arxiv.org/abs/2611.32324](https://arxiv.org/abs/2611.32324)

    本文证明了即使采用精确的单点删除效应作为词元重要性评分，Top-K独立排名选择仍会次优，因为注意力归一化引入的位置间交互可使排名组合与最优组合之间的KL散度差距高达133.2倍且在理论上无界。

    

    长上下文语言模型在进行注意力查询时，通常只保留有限的过去词元位置集合。一种常见策略是对各位置单独打分并保留排名最高的K个。然而，注意力权重会在所选集合上重新归一化：两个单独来看很重要的位置，未必能构成最佳的组合。我们探究了这样一个问题：即使每个重要性得分精确度量了仅删除该位置时对下一词元预测造成的下游变化，独立排名是否仍然足够。答案是否定的。一个四位置构造表明，按排名选出的组合所产生的Kullback-Leibler（KL）散度是最优组合的133.2倍，而一个参数化族则证明这一乘性差距是无界的。一个精确的交换恒等式，以及针对每个K >= 2 都存在的偏好反转现象，解释了这种交互作用。在对SmolLM2、Gemma和Qwen的穷举式冻结模型审计中，精确单点评分的Top-4在完整枚举下仍留有至少10%的相对改进空间……

    arXiv:2611.32324v1 Announce Type: cross  Abstract: Long-context language models often retain only a limited set of past token positions for an attention query. A common strategy scores positions individually and keeps the highest-ranked K. Yet attention weights are renormalized over the chosen set: two individually important positions need not form the best pair. We ask whether independent ranking suffices even when each importance score measures the exact downstream change in next-token predictions caused by deleting that position alone. It does not. A four-position construction makes the ranked pair incur 133.2 times the Kullback-Leibler (KL) divergence of the optimal pair, and a parametric family proves that this multiplicative gap is unbounded. An exact exchange identity and preference reversals for every K >= 2 explain the interaction. In exhaustive frozen-model audits of SmolLM2, Gemma, and Qwen, exact-singleton Top-4 leaves at least 10% relative improvement within a fully enumer
    
[^97]: SPRUNN：用于深度神经网络剪枝的结构接近度方法

    SPRUNN: Structural Proximity for Pruning in Deep Neural Networks

    [https://arxiv.org/abs/2611.32317](https://arxiv.org/abs/2611.32317)

    本文提出基于结构接近度准则的滤波器剪枝方法SPRUNN，通过全面比较滤波器间的相似性识别并剪除不重要的滤波器，并与全连接层细粒度剪枝相结合，在不显著损失准确率的前提下压缩深度神经网络。

    

    深度神经网络在各类机器学习任务中展现出了显著的准确率。然而，这些网络存在参数量和计算量巨大的问题，导致处理过程非常耗时。因此，参数剪枝方法被提出用于压缩深度神经网络并实现高效部署。剪枝的主要挑战在于选择合适的剪枝参数，同时不显著损害模型准确率。本文提出了SPRUNN，一种基于结构接近度准则的算法剪枝方法，用于深度神经网络卷积层中的滤波器剪枝。结构接近度（SP）通过考虑所有滤波器并将每个滤波器与其他滤波器进行比较来计算滤波器相似度，从而有效识别卷积层中较不重要的滤波器进行剪枝。此外，我们将该方法与全连接层中的细粒度剪枝相结合。

    arXiv:2611.32317v1 Announce Type: cross  Abstract: Deep Neural Networks (DNNs) have demonstrated significant accuracy across various machine learning tasks. However, these networks suffer from a huge number of parameters and computations, which leads to time-consuming processes. Consequently, parameter pruning approaches have been proposed to compress DNNs and efficiently implement them. The main challenge in pruning is selecting the appropriate parameters for pruning without significantly damaging the accuracy. In this paper, we propose SPRUNN, an algorithmic pruning approach based on the structural proximity criterion for filter pruning in convolutional layers of deep neural networks. Structural Proximity (SP) calculates filter similarity by considering all filters and comparing each one with the others, effectively identifying less important filters for pruning in the convolutional layers. Furthermore, we integrate this approach with fine-grained pruning in Fully Connected(FC)layers
    
[^98]: ASTERISM：多模态大语言模型真的会看所有图像吗？

    ASTERISM: Do Multimodal LLMs Actually Look at All the Images?

    [https://arxiv.org/abs/2611.32316](https://arxiv.org/abs/2611.32316)

    该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。

    

    多图像基准旨在测试多模态大语言模型（MLLM）能否跨图像整合证据，但其中许多题目仅凭单张图像即可回答。通过对来自19个公开基准的约3.3万道题目进行审计，并采用两阶段单图过滤器，我们发现其中24.2%的题目仅凭一张图像或仅凭问题本身即可解答。在这些题目上，MLLM仅凭一张图像就能获得全部图像相对纯文本所带来的准确率提升的中位数78%，因此这些题目实际上奖励了“从单张图像作答”的行为，我们称之为单图捷径（single-image shortcut）。即便在单张图像不足以作答的题目上，MLLM也会走这条捷径：在这类题目上，它们的大部分错误答案与其仅凭单张图像时给出的答案重复，且该比例远高于随机水平。我们基于这19个基准构建了ASTERISM——据我们所知是规模最大的经过审计的多图像基准——仅保留其中需要多张图像才能作答的题目。其1.87万道题目（其中2000道经过修正或重写）共享同一（摘要在此处截断）。

    arXiv:2611.32316v1 Announce Type: cross  Abstract: Multi-image benchmarks aim to test whether a multimodal large language model (MLLM) combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them. On these items, MLLMs recover from one image a median 78% of the accuracy that all images add over the text alone, so these items reward answering from one image, which we call the single-image shortcut. MLLMs take this shortcut even on items where one image is not enough. On such items, most of their wrong answers repeat an answer they give from one image, at a rate well above chance. We build ASTERISM, to our knowledge the largest audited multi-image benchmark, from these 19 benchmarks by keeping their items that need more than one image. Its 18.7k items, 2.0k of them corrected or rewritten, share one
    
[^99]: DAiSy：基于自适应效用估计的大语言模型预训练在线数据选择

    DAiSy: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining

    [https://arxiv.org/abs/2611.32294](https://arxiv.org/abs/2611.32294)

    DAiSy通过一个与主模型同步更新的轻量级线性辅助模型，从冻结的文档嵌入预测基于梯度的数据效用分数，使在线数据选择中每个候选文档的评分仅需一次线性变换，从而消除了大语言模型预训练中数据选择高昂的计算开销。

    

    模型感知的在线数据选择会优先挑选在模型当前状态下贡献最大的训练文档。现有的在线选择器基于梯度的数据效用分数对候选文档进行排序，这要求对每个候选文档执行主模型的完整前向-反向传播。因此，对每个文档进行评分的代价与在它上面训练的代价相当，但其中只有一小部分文档被真正用于模型更新。我们提出了DAiSy，一个轻量级的线性辅助模型，它学习从冻结的文档嵌入中预测数据效用分数，从而消除了计算开销和基础设施负担。该线性估计器与主语言模型同步更新，拟合常规训练步骤反向传播中获得的精确的基于梯度的效用分数。因此，对候选文档进行评分只需一次线性变换，绕过了代价高昂的梯度对齐计算。我们证明了……（摘要在此处截断）

    arXiv:2611.32294v1 Announce Type: cross  Abstract: Model-aware online data selection prioritizes the training documents that contribute the most given the model's current state. Existing online selectors rank candidates upon gradient-based data utility scores, which requires the full forward-backward pass of the main model on every candidate document. Scoring each document therefore pays as much as training on it, yet only a small fraction is used for the real model update. We introduce DAiSy, a lightweight linear auxiliary model that learns to predict data utility scores from frozen document embeddings, which eliminates the computation overhead and infrastructure burdens. The linear estimator is updated alongside the main language model, fitting to the exact gradient-based utility scores obtained from the backward pass of regular training steps. Consequently, scoring a candidate requires only a linear transformation, bypassing the costly gradient-alignment computation. We demonstrate 
    
[^100]: ε-Seg++：跨层级与训练时间的集成方法实现高效标签的电子显微镜分割

    $\epsilon$-Seg++: Ensembling Across Hierarchy and Training Time for Label-Efficient EM Segmentation

    [https://arxiv.org/abs/2611.32290](https://arxiv.org/abs/2611.32290)

    ε-Seg++ 通过在层次化变分自编码器的每一层级引入语义专家并用无参数专家乘积进行集成、支持无泄漏的 3D 子体积输入、以及跨训练时间对最优检查点进行集成，显著提升了稀疏标注条件下电子显微镜分割的效果。

    

    电子显微镜（EM）体数据的语义分割是定量细胞生物学的核心步骤，但密集的体素级标注成本极其昂贵。ε-Seg 已经证明，在稀疏标注体素上训练的层次化变分自编码器可以获得具有竞争力的分割结果，但它仅从最顶层的潜在层级进行分类，且只能在 2D 图像块上运行。我们提出了 ε-Seg++，它在层次结构的每一层都放置了一个语义专家，并通过无参数的专家乘积来整合它们的预测。我们进一步通过无泄漏的子体积划分协议实现了 3D 输入，使得 2D 和 3D 模型能够在相同的稀疏监督下进行比较。由于验证标签与训练标签同样稀缺，我们将检查点选择作为方法的一部分，对排名靠前的检查点进行平均，这构成了第二个集成——即跨训练时间的集成。在 B（原文摘要在此处截断）

    arXiv:2611.32290v1 Announce Type: cross  Abstract: Semantic segmentation of electron microscopy (EM) volumes is a central step in quantitative cell biology, but dense voxel-wise annotation is prohibitively expensive. $\epsilon$-Seg showed that a hierarchical variational autoencoder trained on sparsely annotated voxels can yield competitive segmentations, but it classifies from its topmost latent level only and operates exclusively on 2D patches. We introduce $\epsilon$-Seg++, which places a semantic expert at every level of the hierarchy and consolidates their predictions with a parameter-free product of experts. We further enable 3D inputs through a leakage-free sub-volume splitting protocol, which allows 2D and 3D models to be compared under identical sparse supervision. Because validation labels are just as scarce as training labels, we treat checkpoint selection as part of the method and average the top-ranked checkpoints, a second ensemble, this time across training time. On the B
    
[^101]: CoMNO：用于长时程偏微分方程预测的换位子记忆神经算子

    CoMNO: Commutator Memory Neural Operators for Long-Horizon PDE Prediction

    [https://arxiv.org/abs/2611.32274](https://arxiv.org/abs/2611.32274)

    提出换位子记忆神经算子（CoMNO），通过交换“先下采样后滚动预测”与“先滚动预测后下采样”两种操作顺序所产生的性能差距来提取细尺度动力学细节，从而增强神经算子在长时程偏微分方程预测中捕捉细尺度特征的能力。

    

    使用传统数值方法在长时间范围内求解非线性偏微分方程（PDE）的计算代价高昂。虽然神经算子提供了一种高效的替代方案，但它们在预测轨迹上会累积近似误差，这通常是由于未能捕捉细尺度细节所导致的。在这项工作中，我们设计了一种换位子记忆神经算子，它通过对两个操作进行换位来识别细尺度动力学细节，并利用这些细节来增强神经算子在长时间范围内捕捉细尺度特征的能力。具体而言，我们发现了两种操作之间的性能差距：一种是在滚动预测之前对高分辨率场进行下采样，另一种是先进行滚动预测再进行下采样。我们观察到，细尺度动力学细节往往编码在这一差距之中。通过换位这两个操作，细尺度细节能够被（摘要在此处截断）

    arXiv:2611.32274v1 Announce Type: cross  Abstract: Solving nonlinear partial differential equations (PDEs) over long time horizons is computationally expensive using traditional numerical methods. While neural operators provide an efficient alternative, they suffer from the accumulation of approximation errors along the predicted trajectory, typically caused by the failure to capture fine-scale details. In this work, we design a Commutator Memory Neural Operator that identifies fine-scale dynamical details by commuting two operations and leverages these details to enhance the neural operator's capability in capturing fine-scale features over long time horizons. Specifically, we identify a performance gap between two operations: downsampling the high-resolution field before rollout prediction versus performing rollout prediction before downsampling. We observe that fine-scale dynamical details are often encoded in this gap. By commuting these two operations, the fine-scale details remov
    
[^102]: Pando：面向高效边缘端推测解码的噪声耦合并行草稿树

    Pando: Noise-Coupled Parallel Draft Trees for Efficient Edge Speculative Decoding

    [https://arxiv.org/abs/2611.32269](https://arxiv.org/abs/2611.32269)

    Pando 在并行草稿生成器之上构建动态草稿树，并通过噪声耦合剪枝在草稿生成前依据目标模型的采样噪声对候选分支排序筛选，从而突破固定块大小对草稿链长度的限制，充分利用验证阶段的计算预算，实现高效的边缘端推测解码。

    

    自回归（AR）解码每次前向传播仅生成一个词元，且每次都需重新加载完整模型，因此大语言模型在边缘设备上的推理受内存带宽限制，而计算单元处于闲置状态。推测解码通过在单次前向传播中验证多个草稿词元，将这些闲置算力利用起来。诸如 DFlash 和 DSpark 等最先进的并行草稿生成器可在一次前向传播中提出一整块草稿词元，使草稿生成在计算上高效。然而，它们部署时的草稿链长度受限于其训练所针对的块大小，导致验证阶段的大部分计算预算未被利用。为解决这一问题，我们提出了 Pando，它在并行草稿生成器之上构建动态草稿树，从而消除了这一限制。Pando 采用噪声耦合剪枝技术：在草稿生成之前先抽取目标模型的采样噪声，并在该噪声下对候选分支进行排序，从而使草稿树保留目标模型最可能接受的分支。

    arXiv:2611.32269v1 Announce Type: new  Abstract: Autoregressive (AR) decoding emits one token per forward pass, reloading the full model each time, so LLM inference on edge devices is memory-bound while the compute units sit idle. Speculative decoding puts that idle compute to work by verifying several drafted tokens in a single pass. State-of-the-art parallel drafters such as DFlash and DSpark propose an entire block of draft tokens in one forward pass, which makes drafting compute-efficient. However, they are deployed with draft chains whose length is capped by the block size they are trained for, leaving most of the computational budget for verification unused.   To tackle this problem, we propose Pando, which builds dynamic draft trees on top of a parallel drafter and thereby removes this cap. Pando employs noise-coupled pruning, which draws the target model's sampling noise before drafting and ranks candidate branches under that noise, so the tree retains the branches the target w
    
[^103]: Seldon 表格基础模型家族：Seldon-Mini、Seldon 与 Seldon+

    Seldon TFM family: Seldon-Mini, Seldon and Seldon+

    [https://arxiv.org/abs/2611.32257](https://arxiv.org/abs/2611.32257)

    Seldon 表格基础模型家族仅用合成数据训练，在多项表格基准测试中名列前茅，其开源模型 Seldon-Mini 和 Seldon 同时支持分类与回归，并在准确性与推理速度的权衡上处于帕累托前沿。

    

    我们介绍了 Seldon，一个表格基础模型家族。我们的分类器 Seldon+ 在 TabBench（我们构建的包含 124 个分类数据集的基准测试）及其含缺失值的变体上，在 15 个模型中排名第一。在我们自行运行的公开基准测试中，它还在 BeyondArena IID 分类和 TabBench-Bio 上排名第一，在 TabArena 分类任务上排名第二。开源权重模型 Seldon-Mini（2700 万参数）和 Seldon（2.13 亿参数）均位于 TabArena 上 Elo 得分与推理时间的帕累托前沿：没有任何模型能够同时做到更准确且更快速。在 TabArena 的 109 个排行榜条目中，Seldon 排名第四，Seldon-Mini 排名第七；在 BeyondArena 的 IID 赛道上，Seldon 在 42 个条目中排名第二。Seldon-Mini 和 Seldon 同时支持分类与回归任务，采用与 Kumo-Tabular、TabFM 和 TabPFN-3.5 类似的架构，并且仅使用合成表格数据进行训练。

    arXiv:2611.32257v1 Announce Type: new  Abstract: We introduce Seldon, a family of tabular foundation models. Our classifier Seldon+ ranks first among 15 models on TabBench, our benchmark of 124 classification datasets, and on its variant with missing values. In our own runs of public benchmarks, it also ranks first on BeyondArena IID classification and TabBench-Bio, and second on TabArena classification. The open-weight models Seldon-Mini (27M parameters) and Seldon (213M parameters) both lie on the TabArena Pareto front of Elo against inference time: no entry is both more accurate and faster. On TabArena, Seldon ranks fourth and Seldon-Mini seventh among 109 leaderboard entries, and on the IID track of BeyondArena, Seldon ranks second among 42 entries. Seldon-Mini and Seldon handle both classification and regression, use an architecture similar to those of Kumo-Tabular, TabFM and TabPFN-3.5, and are trained on synthetic tables only.
    
[^104]: MYSTIQUE：保留表演表现的少样本面部身份迁移

    MYSTIQUE: Few-Shot Facial Identity Transfer with Performance Preservation

    [https://arxiv.org/abs/2611.32232](https://arxiv.org/abs/2611.32232)

    提出基于扩散模型的少样本面部身份迁移方法MYSTIQUE，通过将身份与表情、光照、场景分流的输入解耦策略及多阶段训练，有效防止自重建训练带来的身份泄露，同时完整保留驱动图像中的表演表现。

    

    在视觉效果领域，身份迁移是指将驱动图像中的人物身份替换为目标人物的身份，同时保留该图像中的表情、光照、姿态和场景。由于配对的真值数据非常稀缺，许多方法采用自重建的方式进行训练，这使得模型可以走捷径，直接从驱动图像中复制身份线索。这种捷径会在推理时导致身份泄露，即驱动图像中显示的是另一个不同的人。我们的关键观察是：身份泄露可以通过任何读取驱动图像的输入通道进入。因此，我们提出了MYSTIQUE，这是一种基于扩散模型的身份迁移方法，它从目标人物的一组照片中提取身份信息，并通过相互分离的独立流读取表情、光照和场景。该方法通过多阶段训练学习将身份与其他信息流解耦，并在各输入处采用解耦策略：一种边界不沿面部轮廓的随机块掩码……

    arXiv:2611.32232v1 Announce Type: cross  Abstract: In visual effects, identity transfer replaces the identity in a driving image with that of a target person, while preserving the expression, lighting, pose and scene of that image. Paired ground truth is rarely available, so many methods are trained by self-reconstruction, where the model can learn a shortcut by copying identity cues from the driving image. This shortcut causes identity leakage at inference, where the driving image shows a different person. Our key observation is that leakage can enter through any input that reads the driving image. We therefore present MYSTIQUE, a diffusion-based identity transfer method that takes identity from a set of the target's photos and reads expression, lighting and scene through separate streams. It learns to disentangle identity from the other streams through multi-stage training, with disentanglement strategies at their inputs: a random block mask whose boundaries do not follow the face ou
    
[^105]: 关系瓶颈模型

    Relational Bottleneck Models

    [https://arxiv.org/abs/2611.32227](https://arxiv.org/abs/2611.32227)

    该论文提出了免训练、即插即用的关系瓶颈模型（RBMs），通过结构化推理依据生成显式建模视觉概念间的语义关系，将现成的大型视觉语言模型转化为兼具可解释性与高性能的内在可解释模型。

    

    实现既可解释又高性能的推理是深度学习中的一个关键挑战。概念瓶颈模型（CBMs）虽然提供了内在的可解释性，但受限于任务特定的训练，且依赖于孤立的视觉概念，无法捕捉高阶的视觉语义关系。相反，大型视觉语言模型（LVLMs）通过思维链（CoT）推理展现了零样本能力。然而，标准的CoT在生成的推理依据与最终输出之间缺乏严格的因果依赖关系，损害了推理的忠实性。为了弥合这一差距，我们提出了关系瓶颈模型（RBMs），这是一种新颖的免训练框架，能够以即插即用的方式将任何现成的LVLM转化为内在可解释的模型。RBMs包含两个阶段：（i）结构化推理依据生成（SRG），显式地将视觉概念之间的语义交互建模为主-谓-宾（三元组）……（原文摘要在此处截断）

    arXiv:2611.32227v1 Announce Type: cross  Abstract: Achieving both interpretable and high-performance inference is a key challenge in deep learning. While Concept Bottleneck Models (CBMs) provide inherent interpretability, they are constrained by task-specific training and rely on isolated visual concepts that fail to capture higher-order visual semantic relationships. Conversely, Large Vision Language Models (LVLMs) demonstrate zero-shot capabilities through Chain-of-Thought (CoT) reasoning. However, standard CoT lacks a strict causal dependency between the generated rationale and the final output, compromising faithfulness. To bridge this gap, we propose Relational Bottleneck Models (RBMs), a novel training-free framework that transforms any off-the-shelf LVLMs into inherently interpretable models in a plug-and-play manner. RBMs consist of two stages: (i) Structured Rationale Generation (SRG), which explicitly models semantic interactions between visual concepts as subject-predicate-o
    
[^106]: 表格数据上的联合分类与生成：一项实证研究

    Joint Classification and Generation on Tabular Data: an Empirical Study

    [https://arxiv.org/abs/2611.32225](https://arxiv.org/abs/2611.32225)

    本文系统比较了基于能量模型、概率电路、表格基础模型和表格扩散模型四类最先进方法构建的联合分类与生成模型，在29个数据集上评估了不同计算预算下它们与独立“分类器+生成器”组合的优劣。

    

    机器学习在表格数据上的许多应用既需要一个分类器，也需要一个能生成真实记录的生成器。默认做法是训练两个独立的模型，而单一的联合模型有望仅用一组权重同时实现两者。多个表格深度学习模型家族都可以被改造成这样的联合模型，这就引出了一个问题：在何种计算预算下，哪种联合模型更为可取。在这项实证研究中，我们在29个分类数据集上引入并比较了基于四个最先进的表格数据建模家族（基于能量的模型、概率电路、表格基础模型和表格扩散模型）构建的联合模型，并以实测的FLOPs和挂钟时间作为衡量标准。这些方法基于两个定义的聚合分数进行评估，这两个分数综合考虑了预测和生成的不同方面。随后，我们还将这些联合模型与独立的分类器和生成器的各种组合在不同计算预算下进行了比较。

    arXiv:2611.32225v1 Announce Type: cross  Abstract: Many applications of machine learning on tabular data need both a classifier and a generator of realistic records. The default is to train two separate models, whereas a single joint model promises both from one set of weights. Several families of tabular deep learning models can be turned into such a joint model, which raises the question of which joint model is preferable at which computational budget. In this empirical study, we introduce and compare joint models based on four state-of-the-art tabular data modeling families (energy-based models, probabilistic circuits, tabular foundation models and tabular diffusion models) against measured FLOPs and wall-clock time, on 29 classification datasets. Methods are evaluated based on two defined aggregate scores that take into account different aspects of prediction and generation. We then also compare these joint models with every pair of a separate classifier and generator at different 
    
[^107]: 超越自回归：扩散语言模型正是模拟内存计算所需要的

    Beyond Autoregression: Diffusion Language Model is All Analog In-Memory Computing Need

    [https://arxiv.org/abs/2611.32223](https://arxiv.org/abs/2611.32223)

    该论文发现扩散语言模型在模拟内存计算的硬件噪声扰动下比自回归模型显著更鲁棒，其机制在于延迟词元提交与渐进式上下文解析避免了误差级联，且更大的去噪步数预算可作为延迟与鲁棒性之间的调节手段。

    

    模拟内存计算（AIMC）能够实现高能效的语言模型推理，但硬件的非理想性会扰动矩阵-向量乘法运算，从而降低生成质量。自回归（AR）模型对此尤其脆弱：每个词元在生成之初就被固定写入因果上下文，因此扰动引发的误差会不断级联累积。我们研究了扩散语言模型（DLMs）——一种通过迭代精炼掩码序列的模型——作为替代范式。在规模匹配的模型对以及开源检查点上，DLMs在持续权重扰动下比对应的AR模型退化得更加平缓。受控分析排除了跨轮次噪声平均的解释，并支持“延迟词元提交”和“渐进式上下文解析”作为其鲁棒性机制。更大的去噪步数预算可进一步缩小干净条件与噪声条件之间的差距，这是DLM特有的延迟-鲁棒性调节旋钮。我们使用校准的噪声模型，在多样化的数据集和模型上进行了大规模仿真测试……

    arXiv:2611.32223v1 Announce Type: cross  Abstract: Analog in-memory computing (AIMC) enables energy-efficient language-model inference, but hardware non-idealities perturb matrix--vector multiplications and degrade generation quality. Autoregressive (AR) models are especially vulnerable: each token is committed to the causal context at birth, so perturbation-induced errors cascade. We study diffusion language models (DLMs), which iteratively refine a masked sequence, as an alternative paradigm. Across scale-matched model pairs and open checkpoints, DLMs degrade more gracefully than AR counterparts under persistent weight perturbations. Controlled analyses rule out cross-round noise averaging and support delayed token commitment and progressive context resolution as the mechanism. A larger denoising-step budget further shrinks the clean--noisy gap, a DLM-specific latency--robustness knob. We perform large-scale simulation tests across diverse datasets and models using a noise model cali
    
[^108]: 谁获得信息？一个用于多标签特征选择的 α-公平标签覆盖框架

    Who gets the information? An $\alpha$-fair label-coverage framework for multi-label feature selection

    [https://arxiv.org/abs/2611.32200](https://arxiv.org/abs/2611.32200)

    该论文提出 FLIC 框架，将多标签特征选择中不同的相关性分配策略统一解释为隐含的社会福利函数，通过单一公平参数 α 实现从功利主义到最大最小的信息覆盖分配，并证明在 α≥0 时贪婪选择具有 (1-1/e) 的近似保证。

    

    近来的信息论多标签特征选择过滤器在如何将相关性分配到各标签上存在分歧：一些方法放大那些能为少数标签提供信息的特征，另一些则平衡每个标签所接收的信息。我们证明这些选择实际上是隐含的社会福利函数。我们将标签视为以所接收信息为效用的智能体，提出 FLIC，它贪婪地最大化每标签信息覆盖的 α-公平福利。单一参数 α 从功利主义（α=0）经过比例公平（α=1）过渡到最大最小选择（α→∞），十几种已发表的相关性度量都位于这条轴上，其中大多数是特殊情形、一阶近似或极限。当每个标签的信息是单调次模时，对于每个 α≥0 该目标函数也是单调次模的，因此贪婪选择可获得 (1-1/e) 近似保证；该性质在 α<0 时失效，且最大最小极限情形是不可近似的。（摘要在此处截断）

    arXiv:2611.32200v1 Announce Type: cross  Abstract: Recent information-theoretic filters for multi-label feature selection disagree on how to distribute relevance over labels: some amplify features that inform a few labels, others balance what every label receives. We show these choices are implicit social welfare functions. Treating labels as agents whose utility is the information they receive, we propose FLIC, which greedily maximises the $\alpha$-fair welfare of per-label information coverage. A single parameter $\alpha$ moves from utilitarian ($\alpha=0$) through proportionally fair ($\alpha=1$) to max-min selection ($\alpha\to\infty$), and a dozen published relevance terms lie on this axis, most as special cases, first-order approximations or limits. The objective is monotone submodular for every $\alpha\ge0$ whenever each label's information is, so greedy selection is a $(1-1/e)$-approximation; this fails for $\alpha<0$, and the max-min limit is inapproximable. Fano's and the Hel
    
[^109]: 可操控性设计：用于语言模型鲁棒控制的条件预训练

    Steerable by Design: Conditional Pretraining for Robust Control of Language Models

    [https://arxiv.org/abs/2611.32199](https://arxiv.org/abs/2611.32199)

    该论文提出了一种条件预训练方法和架构，通过编码器交叉注意力机制从一开始就将指令与数据通道分离，构建出Mancelle-3B这一在设计上即可被有效操控的语言模型，解决了模型误将数据当作指令处理的越狱与对齐漂移问题。

    

    我们的目标是构建在设计上就具备可操控性的语言模型。当今的语言模型采用下一词预测进行预训练，这将模型的对齐与控制推迟到训练最后1%的token或推理阶段的干预。这些模型将单一输入通道同时用于指令和数据，导致了常见的失败模式——模型错误地将数据当作指令处理，例如有意的越狱攻击和无意的上下文对齐漂移。为了解决这一问题，我们提出了一种从训练一开始就将两者分离的预训练方法和架构，由此产生了Mancelle-3B。一个编码器通过交叉注意力机制处理后续文本的高层语义特征，并阻止token流向该控制通道回写。在预训练过程中，模型学会依赖这些特征来预测下一个token，从而将该通道转变为操控模型的有效抓手。

    arXiv:2611.32199v1 Announce Type: cross  Abstract: Our goal is to build language models that are steerable by design. Today's language models are pretrained with next-token prediction, which relegates alignment and control to the final 1% of training tokens or inference-time interventions. These models repurpose a single input channel for both instructions and data, leading to common failure modes where the model mistakenly treats data as instructions such as intentional jailbreaks and unintentional in-context alignment drift. To correct this, we propose a pretraining method and architecture that separates the two from the very start, producing Mancelle-3B. An encoder processes high-level semantic features of upcoming text with cross-attention that prevents the token stream from writing back into this control channel. During pretraining, the model learns to rely on these features to predict the next token, turning the channel into an effective handle for steering the model. Mancelle-3B
    
[^110]: 联邦学习中梯度反演防御的统一控制框架

    A Unified Control Framework for Gradient Inversion Defense in Federated Learning

    [https://arxiv.org/abs/2611.32174](https://arxiv.org/abs/2611.32174)

    本文提出一个统一的受控动力学框架来刻画联邦学习中各类梯度扰动防御方法，并据此提出近正交梯度投影（NODE），仅沿正交方向注入扰动以保持梯度下降方向不变，从而在保障隐私的同时大幅降低优化开销。

    

    梯度扰动被广泛认为是抵御联邦学习中梯度反演攻击的有效手段，但同样众所周知的是，它也会损害优化过程，导致隐私与效用之间存在持续的矛盾。在本文中，我们证明现有的梯度扰动防御方法尽管公式形式各异，却可以在一个受控动力学框架下被统一起来，从而对其优化行为提供连贯一致的理解，并使得无需依赖针对具体案例的分析即可推导出通用的收敛性与安全性条件。基于这一观察，我们提出了“近正交梯度投影”，该方法仅沿与原始梯度正交的方向注入扰动，使得防御后的梯度与真实梯度之间保持固定的余弦相似度。通过保留一阶下降方向，NODE 大幅降低了更强扰动带来的优化成本。

    arXiv:2611.32174v1 Announce Type: new  Abstract: Gradient perturbations are widely recognized as an effective defense against gradient inversion attacks in federated learning, but they are equally known to impair optimization, creating a persistent tension between privacy and utility. In this paper, we show that existing gradient perturbation defenses, despite their different formulations, can be unified under a controlled dynamics framework, providing a cohesive understanding of their optimization behavior and enabling the derivation of universal convergence and safety conditions without relying on case-specific analyses. Leveraging this observation, we propose ``Near Orthogonal Gradient Projection (NODE)'', which injects perturbations only along directions orthogonal to the original gradient so that the defended gradient maintains a fixed cosine similarity with the true gradient. By preserving the first-order descent direction, NODE substantially reduces the optimization cost of stro
    
[^111]: 多视图随机集学习

    Multi-view Random-set Learning

    [https://arxiv.org/abs/2611.32170](https://arxiv.org/abs/2611.32170)

    提出多视图随机集学习（MRSL）新范式，保留各视图对候选类别集合的支持，通过跨视图集合交集直接细化候选组，以应对多视图数据中的视图退化问题。

    

    可信多视图学习旨在整合异构视图以实现准确可靠的预测。然而，多视图数据通常表现出视图退化现象：单个视图仅包含部分判别信息，无法独立确定实例级标签。要求此类视图产生明确无歧义的类别证据会扭曲证据学习并误导后续的融合过程。现有的解决该问题的方法至少存在以下缺点之一：（1）将证据分配给单个类别，而没有显式表示未解决的类别组；（2）在融合之前将复合信念投影到单个类别上，阻碍了通过跨视图集合交集对候选组的直接细化。为了解决这些局限性，我们提出了一种新的多视图随机集学习范式，该范式保留了视图特定的对候选类别集合的支持，并对候选集进行细化……

    arXiv:2611.32170v1 Announce Type: new  Abstract: Trusted multi-view learning aims to integrate heterogeneous views for accurate and reliable predictions. However, multi-view data often exhibit the view degradation phenomenon: individual views contain only partial discriminative information and cannot independently determine the instance-level label. Requiring such views to produce unambiguous class evidence can distort evidence learning and mislead subsequent fusion. Existing methods addressing this problem have at least one of the following drawbacks: (1) assigning evidence to individual classes without explicitly representing unresolved class groups; (2) projecting composite beliefs onto individual classes before fusion, preventing direct refinement of candidate groups through cross-view set intersections.   To address these limitations, we propose a new Multi-view Random-Set Learning (MRSL) paradigm, which preserves view-specific support for candidate class sets and refines candidat
    
[^112]: RetraQ：强化学习中流策略的流形约束Critic引导

    RetraQ: Manifold-Constrained Critic Guidance for Flow Policies in RL

    [https://arxiv.org/abs/2611.32148](https://arxiv.org/abs/2611.32148)

    RetraQ利用预训练流模型编码的几何结构（去噪映射及其雅可比矩阵）来过滤和校正Critic梯度引导，使其近似于向行为流形的切向投影和回缩，从而避免引导过程被Critic外推误差误导，有效改进强化学习中的流匹配策略。

    

    学习到的Critic可以通过将采样动作引导向更高的预测价值来改进预训练的流匹配策略。然而，即使是能很好拟合行为数据的Critic也可能提供误导性的动作梯度。指向数据支持区域之外的梯度分量受行为数据的约束较弱，可能导致引导过程利用Critic的外推误差。我们提出了RetraQ（基于回缩的Q引导），它利用预训练流所编码的几何结构来同时约束Critic引导的方向和终点。对冻结流进行一次速度评估即可定义一个去噪映射。RetraQ用该映射的雅可比矩阵过滤Critic梯度，并应用该映射本身来校正更新后的动作。我们的分析刻画了这些操作何时近似于切向过滤以及向行为流形上的回缩。同一更新机制既可在测试时引导动作，也可为策略训练提供目标。

    arXiv:2611.32148v1 Announce Type: cross  Abstract: A learned critic can improve a pretrained flow-matching policy by steering sampled actions toward higher predicted value. However, even a critic that fits the behavior data well can provide misleading action gradients. Gradient components pointing away from the data-supported region are weakly constrained by behavior data and can lead guidance to exploit critic extrapolation errors. We introduce RetraQ (Retraction-based Q-guidance), which uses geometry encoded by the pretrained flow to constrain both the direction and the destination of critic guidance. A single velocity evaluation of the frozen flow defines a denoising map. RetraQ filters the critic gradient with the map's Jacobian and applies the map itself to correct the updated action. Our analysis characterizes when these operations approximate tangent filtering and retraction onto the behavior manifold. The same update guides actions at test time and provides targets for policy d
    
[^113]: ARS-VLM：面向视觉-语言模型的视觉对齐增强随机搜索

    ARS-VLM: Visually Aligned Augmented Random Search for Vision-Language Models

    [https://arxiv.org/abs/2611.32147](https://arxiv.org/abs/2611.32147)

    提出ARS-VLM，一种视觉对齐的增强随机搜索零阶方法，解决了直接将进化策略类算法应用于视觉-语言模型时的图像块网格过拟合和灾难性遗忘两大失败模式，同时规避了GRPO等一阶强化学习带来的内存与系统开销。

    

    以群体相对策略优化（GRPO）为代表的一阶强化学习被广泛用于视觉-语言模型（VLM）的多模态推理后训练。然而，这类方法因反向传播、优化器状态和参考模型而带来沉重的内存与系统开销。仅前向传播的零阶（ZO）方法，如进化策略（ES）及其top-k变体增强随机搜索（ARS），通过奖励引导的权重扰动规避了这些开销，但在VLM推理领域尚未被探索。我们发现，直接将标准ES或原始ARS应用于VLM会出现两种根本性的失败模式：(i) 图像块网格过拟合，即模型拟合脆弱的网格特定视觉token伪影而忽略空间不变性；(ii) 灾难性遗忘，即由于无界的高秩参数与策略漂移导致通用感知能力和分布外（OOD）任务性能下降。我们提出了ARS-VLM，一种视觉对齐的增强随机搜索方法（摘要在此处截断）。

    arXiv:2611.32147v1 Announce Type: cross  Abstract: First-order reinforcement learning, such as Group Relative Policy Optimization (GRPO), is widely used to post-train Vision-Language Models (VLMs) for multimodal reasoning. However, it incurs heavy memory and system overhead from backpropagation, optimizer states, and a reference model. Forward-only zeroth-order (ZO) methods, such as Evolution Strategies (ES) and its top-k variant Augmented Random Search (ARS), avoid these overheads via reward-guided weight perturbations, yet remain unexplored for VLM reasoning. We find that directly applying standard ES or vanilla ARS to VLMs suffers from two fundamental failure modes: (i) patch-grid overfitting, fitting fragile grid-specific visual token artifacts while ignoring spatial invariants, and (ii) catastrophic forgetting, degrading general perception and out-of-distribution (OOD) tasks due to unbounded high-rank parameter and policy drift. We propose ARS-VLM, a visually aligned Augmented Ran
    
[^114]: 潜空间推理与通信的校准化读取

    Calibrated Reading of Latent Reasoning and Communication

    [https://arxiv.org/abs/2611.32143](https://arxiv.org/abs/2611.32143)

    本文提出校准潜读取器（CLR），一个紧凑的Transformer，通过任务条件化交互、源适配器迁移与保形校准，以比语言模型解码器低数百倍的计算成本准确读取潜推理结果与键值通信内容，并提供统计覆盖率与误报保证。

    

    潜推理与键值通信将中间信息存储在连续表示中，限制了对模型行为的直接检查。现有的激活读取器在任务特定探针的低成本与语言模型解码器的灵活性之间进行权衡。我们提出了校准潜读取器（Calibrated Latent Reader，CLR），这是一个紧凑的Transformer，通过状态与候选答案之间的任务条件化交互来预测推理结果和消息内容。源适配器支持跨模型迁移，而保形校准在可交换性条件下提供覆盖率与误报保证。在两个CODI模型上，CLR在最终答案预测和参考值隶属判断任务中比MLP和注意力探针的准确率提高了4-18个百分点。在CODI-GPT-2上，它以减少437-648倍推理FLOPs的代价，实现了与基于1.7B参数LatentQA的言语化器相当的准确率。在合成的Qwen3-4B通信上，比较（摘要在此处被截断）

    arXiv:2611.32143v1 Announce Type: cross  Abstract: Latent reasoning and key-value communication store intermediate information in continuous representations, limiting direct inspection of model behavior. Existing activation readers trade the low cost of task-specific probes against the flexibility of language-model decoders. We propose the Calibrated Latent Reader, CLR, a compact transformer that predicts reasoning outcomes and message content through task-conditioned interactions between states and candidate answers. Source adapters support transfer across models, while conformal calibration provides coverage and false-alarm guarantees under exchangeability. On two CODI models, CLR improves accuracy over MLP and attention probes by 4-18 percentage points for final-answer prediction and reference-value membership. On CODI-GPT-2, it achieves accuracy comparable to a 1.7B LatentQA-based verbalizer using 437-648 times fewer inference FLOPs. On synthetic Qwen3-4B communications, comparing 
    
[^115]: VIDE：基于视觉内在分布估计的无训练视频异常检测

    VIDE : Training-Free Video Anomaly Detection via Visual Intrinsic Distribution Estimation

    [https://arxiv.org/abs/2611.32138](https://arxiv.org/abs/2611.32138)

    VIDE是一种无需训练的视频异常检测框架，通过利用冻结片段嵌入估计每个测试视频的正常场景内容分布，自适应修正文本参考并重调整类别响应，从而减少跨场景误报并降低对提示词选择的敏感性。

    

    在基于嵌入的视频异常检测（VAD）中，正常片段可能与异常描述具有高相似度，这种异常偏差会导致误报。诸如PRISM等方法通过测量沿异常方向相对于正常性参考的偏差来缓解这一问题。然而，正常性参考和异常方向均从文本中导出，并在所有视频之间共享，尽管正常活动和异常的表现形式在不同场景中各不相同。由此产生的评分对提示词的选择也十分敏感。为了解决这些局限性，我们提出了视觉内在分布估计（VIDE），这是一个无需训练的框架，它利用冻结的片段嵌入将基于文本的评分适配到每个测试视频中。VIDE从初始异常分数较低的片段中估计正常场景内容，并利用该估计对文本参考进行部分修正。随后，它重新调整类别响应的尺度，并对在多个类别上具有强烈响应的片段进行惩罚。

    arXiv:2611.32138v1 Announce Type: cross  Abstract: In embedding-based video anomaly detection (VAD), normal clips can have high similarity to anomalous descriptions, an anomaly bias that leads to false positives. Methods such as PRISM mitigate this bias by measuring deviations from a normality reference along anomaly directions. Both the normality reference and the anomaly directions are derived from text and shared across videos, although normal activity and the appearance of anomalies vary across scenes. The resulting scores are also sensitive to prompt selection. To address these limitations, we propose Visual Intrinsic Distribution Estimation (VIDE), a training-free framework that adapts text-derived scoring to each test video using frozen clip embeddings. VIDE estimates normal scene content from clips with low initial anomaly scores and uses this estimate to partially correct the textual reference. It then rescales category responses, penalizes clips with strong responses across c
    
[^116]: 具有尺度选择性不确定性的可解释正交潜在扩散用于医学图像超分辨率

    Interpretable Orthogonal Latent Diffusion with Scale-Selective Uncertainty for Medical Image Super-Resolution

    [https://arxiv.org/abs/2611.32135](https://arxiv.org/abs/2611.32135)

    提出Med-PODiff分块POD扩散框架，将预测不确定性分解为粗细两个尺度并定义逐像素幻觉风险评分，用于脑部MRI超分辨率中的细尺度幻觉识别。

    

    扩散模型在医学图像重建和超分辨率中的应用日益增多，但其采样结果可能包含看似合理的细尺度幻觉，仅从重建图像本身难以识别。我们提出Med-PODiff，一种用于脑部MRI超分辨率的分块本征正交分解（POD）扩散框架，在紧凑且可解释的局部系数空间中进行生成式重建。与像素空间或学习潜在空间的扩散模型不同，Med-PODiff利用POD结构将低秩解剖变异与细节残差变异分离，使预测不确定性能够分解为总量、粗尺度和细尺度三个分量。基于这一分解，我们定义了逐像素幻觉风险评分 $H_p$，该评分突出细节不确定性主导粗结构不确定性的区域。在脑部MRI上，Med-PODiff优于代表性的扩散……

    arXiv:2611.32135v1 Announce Type: cross  Abstract: Diffusion models are increasingly used for medical image reconstruction and super-resolution, but their samples may contain plausible fine-scale hallucinations that are difficult to identify from the reconstructed image alone. We propose Med-PODiff, a patch-wise proper orthogonal decomposition (POD) diffusion framework for brain MRI super-resolution that performs generative reconstruction in compact and interpretable local coefficient spaces. Unlike pixel-space or learned-latent diffusion models, Med-PODiff uses the POD structure to separate low-rank anatomical variation from fine-detail residual variation, enabling predictive uncertainty to be decomposed into total, coarse, and fine components. Building on this decomposition, we define a pixel-wise hallucination-risk score, $H_p$, that emphasizes regions where fine-detail uncertainty dominates coarse structural uncertainty. On brain MRI, Med-PODiff outperforms representative diffusion
    
[^117]: 重新思考大语言模型的推理努力训练：策略内蒸馏还是强化学习？

    Rethinking Reasoning Effort Training in Large Language Models: On-Policy Distillation or Reinforcement Learning?

    [https://arxiv.org/abs/2611.32131](https://arxiv.org/abs/2611.32131)

    带长度惩罚的强化学习虽能压缩单一策略的推理长度，却无法联合训练多个推理努力层级；而多教师策略内蒸馏凭借层级特定的训练目标，能够同时实现推理压缩与层级分离。

    

    大语言模型在复杂任务上已展现出强大的能力，然而如何控制其推理努力仍是实际部署中的一项重大挑战。本文对推理努力训练进行了系统性研究，对比了强化学习（RL）与策略内蒸馏（OPD）。我们发现，压缩推理与分离努力层级受不同信号的支配。带长度惩罚的强化学习能够可靠地压缩单一策略，在AIME25上减少70%的思考量且不损失准确率，但当多个层级联合训练时会导致层级崩溃。多教师策略内蒸馏则能实现层级分离：在33个基准测试中，低层级和中层级分别比高层级少使用58%和43%的思考token，而高层级相比参考策略提升了3.1分。我们将这一差异归因于训练信号：序列级别的奖励无法区分层级特异性行为与共享行为，而策略内蒸馏则能在每个……

    arXiv:2611.32131v1 Announce Type: new  Abstract: Large language models have shown strong capabilities on complex tasks, yet how to control their reasoning effort remains a significant challenge for practical deployment. This paper provides a systematic investigation of reasoning-effort training, contrasting reinforcement learning (RL) with on-policy distillation (OPD). We find that compressing reasoning and separating effort tiers are governed by different signals. Length-penalized RL reliably compresses a single policy, reducing thinking on AIME25 by 70\% without accuracy loss, but collapses the tiers when they are trained jointly. Multi-teacher OPD separates them: across 33 benchmarks, low and medium use 58\% and 43\% fewer thinking tokens than high, while high improves on the reference policy by 3.1 points. We trace this difference to the training signal. A sequence-level reward cannot tell tier-specific behavior from shared behavior, whereas OPD supplies a tier-specific target at e
    
[^118]: 无需模型参数知识的量化神经网络比特翻转攻击

    Bit-Flip Attacks on Quantized Neural Networks without Knowledge of Model Parameters

    [https://arxiv.org/abs/2611.32128](https://arxiv.org/abs/2611.32128)

    CoarseHammer是一种无需模型参数、梯度或数据的比特翻转攻击方法，它利用量化不对称性将攻击集中于高精度保留的小型张量，可在任意比特位实施翻转，仅用最多24次翻转就能在包括大语言模型在内的多种神经网络中取得超过81.5%的攻击成功率。

    

    我们证明了量化的不对称性将比特翻转攻击的脆弱性集中在以高精度保留的小型张量中。CoarseHammer正是针对这些张量发起攻击，并且可以在任意比特位实施翻转，而无需知道参数值、梯度或数据。即使模型持续进行微调，该攻击仍然有效。在理想放置条件下，CoarseHammer在十个被评估模型（包括四个大语言模型）中的每一个上，至少有81.5%的运行能将模型精度降至随机猜测水平，成功运行所需翻转次数的中位数最多为24次。在每百万比特中仅有0.01个可翻转比特的模拟DRAM环境下，该攻击在每个被评估的卷积网络上都优于三种先前攻击方法，成功率最高达83.6%，而先前方法从未成功。在真实系统上，即使存在地址空间布局随机化（ASLR）保护，CoarseHammer也能将堆分配的张量引导至易受攻击的内存帧，且无需访问模型文件。使用16个模板化内存帧时，所有四个语言模型的中位数精度均降至随机猜测水平，大约……（摘要原文在此处被截断）

    arXiv:2611.32128v1 Announce Type: cross  Abstract: We show that quantization asymmetry concentrates bit-flip vulnerability in small tensors retained at high precision. CoarseHammer targets these tensors and accepts flips at any bit, without parameter values, gradients, or data. It remains effective under continued fine-tuning. Under ideal placement, CoarseHammer reaches chance accuracy in at least 81.5% of runs on each of ten evaluated models, including four large language models, with median successful runs requiring at most 24 flips. On simulated DRAM with 0.01 flippable bits per million, it outperforms three prior attacks on every evaluated convolutional network, succeeding in up to 83.6% of runs while they never succeed. On a real system, CoarseHammer steers heap-allocated tensors onto vulnerable frames despite address space layout randomization, without accessing the model file. With 16 templated frames, median accuracy reaches chance for all four language models and approximately
    
[^119]: EnSurv：深度生成式生存建模及其在肾脏移植物结局中的应用

    EnSurv: Deep Generative Survival Modeling with Application to Kidney Graft Outcomes

    [https://arxiv.org/abs/2611.32127](https://arxiv.org/abs/2611.32127)

    该研究提出了深度生成式生存分析框架EnSurv，利用结合Kaplan-Meier尾部补全的能量评分处理右删失数据，以捕捉非线性临床协变量交互作用，从而准确预测肾移植后移植物失效时间，服务于器官分配优化与个体化免疫抑制治疗。

    

    肾移植是危及生命的终末期肾病的最佳治疗方式，与透析相比具有显著的生存优势。鉴于供体器官的严重短缺，准确预测移植物过早失效的时间是优化器官分配和制定个体化免疫抑制治疗方案的一项重要临床任务。本研究受一个全面的器官共享联合网络（UNOS）数据集的启发，该数据集包含243,443条肾移植记录，具有复杂的多区域结构和较高的右删失率。我们提出了EnSurv（生存engression），这是一种深度生成式生存分析框架，旨在捕捉复杂的非线性临床协变量交互作用，以建模移植物失效时间。在前加性噪声模型家族中，EnSurv通过优化结合了合并Kaplan-Meier尾部补全的能量评分来处理右删失问题。我们识别了合并c……（原文在此处截断）

    arXiv:2611.32127v1 Announce Type: cross  Abstract: Kidney transplantation is the optimal treatment for life-threatening end-stage renal disease, providing significant survival advantages over dialysis. Given the severe shortage of donor organs, accurately predicting the time to premature graft failure is an essential clinical task for optimizing organ allocation and tailoring individualized immunosuppressive therapies. This study is motivated by a comprehensive United Network of Organ Sharing (UNOS) dataset containing 243,443 kidney transplant records with complex multi-region structures and high right-censoring rates. We propose EnSurv (survival engression), a deep generative survival framework designed to capture complex, non-linear clinical covariate interactions for modeling time to graft failure. Within a pre-additive noise model (pre-ANM) family, EnSurv optimizes an energy score augmented with pooled Kaplan-Meier tail completion to handle right-censoring. We identify the pooled-c
    
[^120]: 用于分类器链的蒙特卡洛树搜索

    Monte Carlo Tree Search for Classifier Chains

    [https://arxiv.org/abs/2611.32118](https://arxiv.org/abs/2611.32118)

    本文提出将蒙特卡洛树搜索应用于分类器链，使概率分类器链能够扩展到大量标签的多标签分类任务，同时保持对精确匹配预测不确定性的准确估计。

    

    多标签分类在众多现代任务中仍然十分重要。在诸多挑战中，现代数据集的标签数量日益庞大，且这些标签在训练集中可能存在噪声或缺失。基于标签的评分和基于排序的指标一直很流行，但优化精确的多标签评估（即预测的标签集合应与真实集合完全匹配）对许多领域（如医学）尤为重要，同时还需提供某种置信度度量。这需要对标签之间的联合概率进行估计（即标签依赖性模型，以每个输入实例为条件）。被称为概率分类器链的一系列方法提供了这样的推断能力。然而，现有的这些方法要么无法扩展到大量标签的场景，要么通过边缘分布来近似联合分布以实现扩展，但这种近似方式会损害对精确匹配不确定性的估计。

    arXiv:2611.32118v1 Announce Type: new  Abstract: Multi-label classification continues to be relevant to a variety of modern tasks. Among other challenges, modern data sets are increasingly large with respect to the number of labels, and these labels may be noisy or missing in the training sets. Label-wise scores and ranking-based metrics have been popular, but optimising exact multi-label evaluation (where a predicted set of labels should match the true set exactly) is particularly relevant for many domains, e.g., in medicine; alongside some measure of confidence. This requires an estimation of joint probability across labels (a model of label dependence; conditioned on each input instance). A family of methods known as probabilistic classifier chains provides such inference. However, these existing varieties either do not scale to a large number of labels or do so by approximating joint via marginal distributions in a way that compromises the estimation of uncertainty for exact match 
    
[^121]: PhysRF-WorldJEPA：将异构大语言模型转变为流形解耦的毫米波雷达世界模型

    PhysRF-WorldJEPA: Transforming Heterogeneous Large Language Models into Manifold-Decoupled Millimeter-Wave Radar World Models

    [https://arxiv.org/abs/2611.32113](https://arxiv.org/abs/2611.32113)

    提出PhysRF-WorldJEPA，一种流形解耦的联合嵌入预测架构，通过将连续电磁状态预测与跨模态语言对齐分离，使大语言模型能够基于毫米波雷达实现全天候、隐私保护的物理场景理解。

    

    将大语言模型（LLM）适配到毫米波（mmWave）雷达上，可实现跨黑暗、烟雾和墙壁的隐私保护、全天候物理场景理解。然而，标准的指令微调将连续的雷达数据流视为静态的类文本标记，导致近乎随机的全局雷达-文本检索，以及在传感器黑障情况下的严重物理漂移。反之，将成对对比损失（如SigLIP）直接耦合到共享骨干网络中虽然改善了检索效果，但会注入O(B²)的排斥性梯度，破坏连续坐标流形（R² < 0）。我们提出了PhysRF-WorldJEPA，这是一种流形解耦的联合嵌入预测架构（JEPA），将连续电磁状态预测与跨模态语言对齐分离开来。一个无衰减的1.0倍时钟锚点与并行双尺度射频编码器共同为经过两步状态空间EMA-JEPA（去噪->推演）训练的精简约3% LoRA大语言模型提供输入，并采用采样……

    arXiv:2611.32113v1 Announce Type: new  Abstract: Adapting Large Language Models (LLMs) to millimeter-wave (mmWave) radar enables privacy-preserving, all-weather physical scene understanding across darkness, smoke, and walls. However, standard instruction tuning treats continuous radar streams as static text-like tokens, causing near-random global radar-to-text retrieval and severe physical drift under sensor blackouts. Conversely, coupling pairwise contrastive losses (e.g., SigLIP) directly into the shared backbone improves retrieval but injects $O(B^2)$ repulsive gradients that fracture continuous coordinate manifolds ($R^2 < 0$). We present PhysRF-WorldJEPA, a manifold-decoupled Joint-Embedding Predictive Architecture (JEPA) that separates continuous electromagnetic state prediction from cross-modal language alignment. An unattenuated 1.0x clock anchor and parallel dual-scale RF encoder feed a lean ~3% LoRA LLM trained with two-step state-space EMA-JEPA (Denoise -> Rollout) and sampl
    
[^122]: EEGJev：一个用于多任务脑电问答的系统一决策模型与基准测试

    EEGJev: A System One Decision Model and Benchmark for Multi-Task EEG Question Answering

    [https://arxiv.org/abs/2611.32100](https://arxiv.org/abs/2611.32100)

    该论文提出了 EEGJev——一个通过一次前向传播即可对候选答案评分的“系统一”式脑电决策模型，并构建了 EEGJev-Bench 基准，使用户能够仅通过更换问题来动态切换脑电解码目标，在 13 个测试数据源上达到 99.70% 的属性选择准确率，且响应速度优于 NeuroLM。

    

    单次脑电（EEG）记录可以支持多种解码任务。固定标签的测试无法展现脑电基础模型应对变化问题或选项的能力。我们提出了 EEGJev，一种快速决策模型，可在一次前向传播中对候选答案进行评分。我们的基准测试 EEGJev-Bench 涵盖了来自 19 个公开脑电数据源的六种问题类型。我们保持脑电记录和选项固定不变，同时在训练时配对的属性相关问题之间切换，例如面部表情和被试被要求使用的手。在 13 个测试数据源上，EEGJev 为所请求的属性选择标签的准确率达到 99.70%（均匀随机选择下为 49.20%），针对特定答案的平衡准确率为 62.04%。在逐题回答的匹配验证问题上，其第 95 百分位响应时间短于 NeuroLM。EEGJev 使用户能够通过更改问题来切换解码目标。借助 EEGJev-Bench，研究人员可以测试……（原文在此处截断）

    arXiv:2611.32100v1 Announce Type: cross  Abstract: A single electroencephalographic (EEG) recording can support multiple decoding tasks. Fixed-label tests cannot show how EEG foundation models handle changing questions or options. We introduce EEGJev, a fast decision model that scores candidate answers in one pass. Our benchmark, EEGJev-Bench, covers six question types across 19 public EEG sources. We held the recording and options fixed while switching between questions about attributes paired during training, such as facial expression and the hand participants were asked to use. Across 13 test sources, EEGJev selected a label for the requested attribute with 99.70% accuracy (49.20% under uniform choice). Balanced accuracy for the specific answer was 62.04%. Its 95th-percentile response time was shorter than NeuroLM's on matched validation questions answered individually. EEGJev lets users switch decoding targets by changing the question. With EEGJev-Bench, researchers can test whethe
    
[^123]: 面向多保真度贝叶斯优化的融合高斯过程老虎机框架

    A Fused Gaussian Process Bandit Framework for Multi-Fidelity Bayesian Optimization

    [https://arxiv.org/abs/2611.32092](https://arxiv.org/abs/2611.32092)

    该论文将多保真度贝叶斯优化问题创新性地表述为融合高斯过程老虎机问题，并开发了具有理论性能保证的FGP-UCB算法，从而更准确地建模保真度层级间的复杂依赖关系。

    

    贝叶斯优化（BO）被广泛应用于直接评估成本高昂时对昂贵黑盒函数进行最大化。在许多实际应用中，高保真度目标可以通过一系列低保真度模型来近似，这些模型以大幅降低的成本提供部分但具有参考价值的信息信号。多保真度贝叶斯优化（MFBO）算法旨在战略性地组合这些信息源，以在精度与评估成本之间实现最优权衡。然而，现有的MFBO方法往往过度简化了保真度层级之间复杂的依赖关系，或依赖于缺乏理论性能保证的启发式采集函数。为了解决这些局限性，我们引入了一种将MFBO问题表述为融合高斯过程老虎机问题的新框架，并开发了相应的融合高斯过程置信上界（FGP-UCB）算法。我们的方法通过融合多保真度信息来准确预测高保真度函数。

    arXiv:2611.32092v1 Announce Type: new  Abstract: Bayesian optimization (BO) is widely used to maximize expensive black-box functions when direct evaluation is costly. In many real-world applications, the high-fidelity objective can be approximated by a series of lower-fidelity models that provide partial but informative signals at substantially reduced cost. Multi-fidelity BO (MFBO) algorithms aim to strategically combine these information sources to achieve an optimal trade-off between accuracy and evaluation cost. However, existing MFBO methods often oversimplify the complex dependencies between fidelity levels or rely on heuristic acquisition functions that lack theoretical performance guarantees. To address these limitations, we introduce a new formulation of the MFBO problem as a fused Gaussian Process bandit and develop a corresponding Fused Gaussian Process Upper Confidence Bound (FGP-UCB) algorithm. Our approach accurately predicts the high-fidelity function by fusing multi-fid
    
[^124]: KKT Atlas：面向情境随机优化的低维局部场景模型学习

    KKT Atlas: Learning Low-Dimensional Local Scenario Models for Contextual Stochastic Optimization

    [https://arxiv.org/abs/2611.32071](https://arxiv.org/abs/2611.32071)

    KKT Atlas通过将情境划分为局部图册并在其中学习低维子空间，仅用4-18个场景即可逼近条件KKT最优性条件，相比需要500个场景的传统一体化学习方法实现约4-23倍加速，同时保持相当或更低的风险。

    

    在情境随机优化中，一体化学习基于运行损失训练条件模型，但使用大量场景反复求解随机规划代价高昂。我们提出KKT Atlas，它将情境划分为局部图册，并从场景成本的决策梯度中为每个图册识别一个低维子空间，通过联合拟合子空间坐标和情境相关概率来生成共享场景。我们证明，在正则性条件下，只需比子空间维数多一个场景即可逼近条件KKT最优性条件，并在局部强凸性下给出决策与风险界，其泛化误差随紧凑参数维数缩放。在三个合成问题上，Atlas每个决策仅需4-18个场景，而直接求解全支持一体化学习需要500个场景，它以约4-23倍的计算加速获得了相当或更低的风险。

    arXiv:2611.32071v1 Announce Type: cross  Abstract: In contextual stochastic optimization, integrated learning trains conditional models on operational loss, but repeatedly solving stochastic programs with many scenarios is costly. We propose KKT Atlas, which partitions contexts into local charts and identifies a low-dimensional subspace per chart from decision gradients of the scenario cost. It generates shared scenarios by jointly fitting subspace coordinates and context-dependent probabilities. We show that one more scenario than the subspace dimension suffices to approximate the conditional Karush-Kuhn-Tucker optimality condition under regularity conditions, yielding decision and risk bounds under local strong convexity, with generalization error scaling with the compact parameter dimension. On three synthetic problems, Atlas uses 4-18 scenarios per decision versus 500 for direct-solve full-support integrated learning. It attains comparable or lower risk with about $4$-$23\times$ co
    
[^125]: 修正行动而非思维：前沿大语言模型的精细微调

    Fix How It Acts, Not How It Thinks: Surgical Fine-Tuning of Frontier LLMs

    [https://arxiv.org/abs/2611.32064](https://arxiv.org/abs/2611.32064)

    提出一种“精细微调”方法，不增大更新规模，而是直接在权重更新合并后残存的部分上进行训练，从而在不干扰模型推理能力的前提下纠正其行动层面的小失误，并解决了 LoRA 小幅更新在合并到量化发布版本时大量丢失的问题。

    

    开放权重大语言模型是胜任的软件工程师，但它们有时会在本可以完成的任务上失败。这通常并非推理能力的缺陷，而是长执行轨迹中的几个小失误造成的差异（例如重复已经失败的步骤，或在需求尚未满足时就宣告任务完成）。纠正这一点需要对权重进行精细的更新，引导模型采取正确的行动而不干扰其推理过程。这类更新通常通过 LoRA 适配器学习得到，但往往过于微小，以至于在合并到发布的 checkpoint 时无法保留。在我们的 BF16 版本中，权重变化的中位数仅为单个舍入步长的 1.5%，合并后会损失约三分之二的更新范数。当同一个适配器合并到同一模型的 FP8/NVFP4 混合精度版本时，93% 的更新会消失。我们没有增大更新的规模，而是直接在合并后残存的部分上进行训练。我们引入……（摘要在此处被截断）

    arXiv:2611.32064v1 Announce Type: cross  Abstract: Open-weight LLMs are competent software engineers, but they sometimes fail at tasks they are otherwise able to do. Often it is not a failure of reasoning. It is a few small missteps over a long trajectory (repeating a step that has already failed, or declaring the task done while a requirement is still unmet) that make the difference. Correcting this calls for a surgical weight update that steers the model toward the right actions without interfering with its reasoning. Such updates, typically learned with LoRA adapters, are often too small to survive merging into a released checkpoint. In our BF16 release, the median weight change is only 1.5% of a single rounding step, and merging loses roughly two thirds of the update's norm. When the same adapter is merged into the mixed FP8/NVFP4 release of the same model, 93% of the update disappears. Instead of increasing the size of the update, we directly train on the surviving part. We introd
    
[^126]: Integral DeltaNet：基于自适应记忆积分的改进线性注意力

    Integral DeltaNet: Improved Linear Attention with Adaptive Memory Integration

    [https://arxiv.org/abs/2611.32063](https://arxiv.org/abs/2611.32063)

    Integral DeltaNet通过同时维护快速更新的delta状态和保留其加权历史的积分状态这两个互补记忆，并在读取时自适应地融合二者，从而在缓解线性注意力记忆冲突的同时保留对未来预测有用的历史信息。

    

    线性注意力用线性形式替代softmax注意力，实现线性时间计算，并用固定大小的循环状态取代不断增长的键值缓存。这些计算与内存优势使线性注意力在长上下文语言建模中受到越来越多的关注。然而，将过去信息压缩到固定大小的状态中会造成记忆冲突，阻碍长上下文理解。尽管DeltaNet通过delta规则更新缓解了记忆冲突，Gated DeltaNet进一步引入门控机制加以扩展，但这些机制也可能丢弃对未来预测有用的信息。为解决这一局限，我们提出Integral DeltaNet，它维护两个互补的循环状态：一个快速更新的delta状态，用于捕获当前的键值关联；以及一个积分状态，用于保存delta状态的加权历史。在读取时，模型自适应地结合这些状态。

    arXiv:2611.32063v1 Announce Type: cross  Abstract: Linear attention replaces softmax attention with a linear formulation, enabling linear-time computation with a fixed-size recurrent state instead of a growing key-value cache. These computational and memory advantages have attracted growing interest in linear attention for long-context language modeling. However, compressing past information into a fixed-size state causes memory collisions that hinder long-context understanding. Although DeltaNet mitigates memory collisions through delta-rule updates and Gated DeltaNet extends this approach with gating, these mechanisms may also discard information relevant to future predictions. To address this limitation, we introduce Integral DeltaNet, which maintains two complementary recurrent states: a fast-updating delta state that captures current key-value associations and an integral state that preserves a weighted history of the delta state. At readout, the model adaptively combines these st
    
[^127]: Spec-CoFrGeNet：通过重参数化连分数网络实现稳定且精确的训练

    Spec-CoFrGeNet: Reparametrizing Continued Fraction Networks for Stable and Accurate Training

    [https://arxiv.org/abs/2611.32041](https://arxiv.org/abs/2611.32041)

    该论文提出Spec-CoFrGeNet，通过受格林函数启发对部分分母进行变换以约束三对角哈密顿量的谱，从而解决连分数网络因分母趋零产生极点导致的训练不稳定问题，无需下界约束和增量式训练即可实现稳定且精确的训练。

    

    连分数已被证明是一种能够构建表达能力强且参数高效的神经网络的函数类。这类神经网络不仅在监督学习中有效，还可用于训练大型语言模型（LLM）。然而，该函数类的有理形式在分母趋近于零时会引入极点，以往的工作通过强制施加一个较小的正数（非平滑）下界来避免这一问题。我们证明，性能和训练稳定性对该下界的选择可能高度敏感。在这项工作中，我们受该函数与物理学中格林函数联系的启发，引入了一种对部分分母的新型变换，通过约束所构造的三对角哈密顿量的谱，优雅地缓解了上述问题。我们的方法还进一步消除了以往工作为保持稳定性而采用的增量式训练，从而避免了繁琐的标准预训练过程修改。

    arXiv:2611.32041v1 Announce Type: cross  Abstract: Continued fractions have been shown to be a function class capable of constructing expressive yet parameter-efficient neural networks. These neural networks are not only effective for supervised learning, but also for training LLMs. However, the rational form of this function class introduces poles when the denominator approaches zero, which prior work has avoided by enforcing a small positive (non-smooth) lower bound. We show that performance and training stability can be highly sensitive to the choice of this lower bound. In this work, we introduce a novel transformation of the partial denominators, inspired by its connection to the Green's function in physics, that mitigates the above issue elegantly by constraining the spectra of the constructed tridiagonal Hamiltonian. Our approach further eliminates the need for incremental training used by previous works for stability, thus avoiding involved modifications to standard pre-trainin
    
[^128]: FlashQLA：面向门控DeltaNet（Gated DeltaNet）的高效上下文并行方法

    FlashQLA: Efficient Context Parallelism for Gated DeltaNet

    [https://arxiv.org/abs/2611.32039](https://arxiv.org/abs/2611.32039)

    FlashQLA是一个面向Gated DeltaNet的上下文并行执行框架，通过衰减感知准备、Dual CP、Auto CP和CP感知内核协同设计，挖掘序列级并行并降低边界状态预处理开销，在长上下文场景下实现了最高4.4倍的显著加速。

    

    门控DeltaNet（GDN）通过循环状态更新实现了线性计算复杂度，但分块之间的顺序依赖限制了并行性，尤其是对于小批量或每GPU头级并行度有限的长上下文工作负载。我们提出了FlashQLA，一个面向GDN的上下文并行执行框架，它能够挖掘额外的序列级并行性，同时降低边界状态预处理的开销。FlashQLA通过衰减感知准备减少边界状态预处理开销，并通过Dual CP将序列并行扩展到多GPU，Dual CP可在不同并行层级之间共享摘要计算。工作负载自适应的Auto CP与CP感知的内核协同设计进一步提升了执行效率。在单GPU上，FlashQLA相比Flash Linear Attention（FLA）实现了1.7倍至3.1倍的前向加速以及高达4.4倍的反向加速。在八个GPU上，针对高达100万token的上下文长度，Dual CP实现...

    arXiv:2611.32039v1 Announce Type: new  Abstract: Gated DeltaNet (GDN) achieves linear computational complexity through recurrent state updates, but sequential dependencies between chunks limit parallelism, particularly for long-context workloads with small batch sizes or limited head-level parallelism per GPU. We present FlashQLA, a context-parallel execution framework for GDN that exposes additional sequence-level parallelism while reducing boundary-state preprocessing overhead. FlashQLA reduces boundary-state preprocessing overhead through decay-aware preparation and extends sequence parallelism across GPUs with Dual CP, which shares summary computation across parallelism levels. Workload-adaptive Auto CP and CP-aware kernel co-design further improve execution efficiency. FlashQLA achieves $1.7\times$-$3.1\times$ forward speedup and up to $4.4\times$ backward speedup over Flash Linear Attention (FLA) on a single GPU. On eight GPUs with context lengths up to 1M tokens, Dual CP achieve
    
[^129]: 面向编程智能体的在线环境修复

    Online Environment Repair for Coding Agents

    [https://arxiv.org/abs/2611.32037](https://arxiv.org/abs/2611.32037)

    该论文提出“在线环境修复”方法，在强化学习训练过程中持续检测并修复编程智能体环境中可被利用的验证器缺陷（如不完整的单元测试），在不惩罚有效解法、不丢弃高难度任务的前提下，从源头抑制奖励作弊行为。

    

    程序化验证器为通过强化学习训练编程智能体提供了可扩展的奖励信号，然而验证框架中细微的缺陷和不完整的单元测试，使策略能够在并未真正解决目标任务的情况下获得满分奖励。现有的训练时防御手段要么通过惩罚可疑输出来干预，要么通过过滤被标记的任务和轨迹来干预。然而，输出惩罚并未消除底层验证器可被利用的漏洞，反而促使策略将“作弊捷径”伪装得能够骗过推理监视器，或因检测器误报而将有效的解法清零；而任务与轨迹过滤则会丢弃或压制那些作弊捷径最早显现的高难度问题。预训练阶段的验证器加固虽然能够直接修复环境，但会遗漏那些只有在强化学习过程中随策略能力提升才暴露出来的漏洞类别。在本文中，我们提出在线环境修复方法，该方法在保持触发轨迹的奖励不变的同时，移除隐藏的……（原文摘要在此处截断）

    arXiv:2611.32037v1 Announce Type: new  Abstract: Programmatic verifiers provide scalable reward signals for training coding agents with reinforcement learning, yet subtle harness defects and incomplete unit tests allow policies to earn full credit without solving the intended task. Existing training-time defenses intervene either by penalizing suspicious outputs or by filtering flagged tasks and rollouts. However, output penalties leave the underlying verifier exploitable, driving policies to disguise shortcuts under reasoning monitors or zeroing out valid solutions under detector false positives, whereas task and rollout filtering discards or mutes the hardest problems where shortcuts emerge earliest. Pre-training verifier hardening repairs the environment directly, but misses vulnerability classes that surface only as the policy improves during reinforcement learning. In this paper, we propose online environment repair, which leaves triggering rollout rewards unchanged to remove hidi
    
[^130]: 面向多元时间序列异常检测的尺度鲁棒与全局感知表示学习

    Scale-Robust and Global-Aware Representation Learning for Multivariate Time Series Anomaly Detection

    [https://arxiv.org/abs/2611.32033](https://arxiv.org/abs/2611.32033)

    本文提出STAGER方法，通过联合学习尺度鲁棒的时间表示与全局感知的关系表示，解决了多元时间序列异常检测中跨时间尺度异常捕捉与序列间全局依赖建模两大关键挑战。

    

    多元时间序列异常检测（MTSAD）是一项关键任务，旨在识别异常的系统行为，并支持各类现实应用中的可靠运行。尽管MTSAD已取得长足进展，但表示学习中的两个关键挑战仍未被充分探索：（C1）如何学习尺度鲁棒的时间表示，以捕捉不同时间尺度上的异常；（C2）如何学习全局感知的关系表示，通过融合超越局部窗口的序列级全局特征来捕捉序列间的依赖关系。为应对这些挑战，我们提出了STAGER，一种新颖的方法，可联合学习两种互补的表示：尺度鲁棒的时间表示与全局感知的关系表示。具体而言，STAGER采用（1）尺度鲁棒的时间编码器，通过构建多尺度时间有向无环图（DAG）来学习尺度鲁棒的时间表示……

    arXiv:2611.32033v1 Announce Type: new  Abstract: Multivariate time series anomaly detection (MTSAD) is an essential task for identifying abnormal system behaviors and supporting reliable operations across diverse real-world applications. Despite substantial advances in MTSAD, two key challenges in representation learning remain underexplored: (C1) how to learn scale-robust temporal representations that capture anomalies across different temporal scales and (C2) how to learn global-aware relational representations that capture inter-series dependencies by incorporating series-level global characteristics beyond local windows. To address these challenges, we propose STAGER, a novel method that jointly learns two complementary representations: scale-robust temporal and global-aware relational representations. Specifically, STAGER employs (1) a scale-robust temporal encoder that constructs a multi-scale temporal directed acyclic graph (DAG) to learn scale-robust temporal representations an
    
[^131]: 统一伙伴多样性与环境多样性以提升人机协调能力

    Unifying Partner and Environment Diversity to Improve Human-AI Coordination

    [https://arxiv.org/abs/2611.32031](https://arxiv.org/abs/2611.32031)

    该论文提出跨环境协同博弈（CECP）两阶段框架，通过统一建模伙伴多样性与环境多样性，显著提升AI智能体在陌生环境中与陌生人类伙伴的协调泛化能力。

    

    现代人工智能系统的现实应用需要其能够泛化到未见过的环境以及陌生的人类用户。零样本协调（ZSC）领域主要关注在熟悉的环境中部署AI以与陌生伙伴进行合作。流行的基于种群的方法通过在固定环境的训练过程中让智能体与多样化的模拟人类用户配对来实现这一目标，但没有考虑更为现实的场景，即用户和环境在训练期间都是未见的。最近，有研究探索了在跨环境训练中训练单个自博弈智能体，该方法在陌生人类用户和未见环境上均表现出更好的泛化能力。在本工作中，我们探究联合建模伙伴多样性与环境多样性是否能够在陌生环境中与陌生伙伴实现更强的协调。我们提出了跨环境协同博弈，这是一个两阶段框架

    arXiv:2611.32031v1 Announce Type: cross  Abstract: Real-world applications of modern AI systems require generalization to both unseen environments and novel human users. The field of zero-shot coordination (ZSC) has focused primarily on deploying AI to cooperate with novel partners in otherwise familiar environments. Popular population-based methods achieve this by partnering the agent with a diversity of simulated human users during training in a fixed environment, but does not consider the more realistic scenario where both the users and the environment are unseen during training. Recently, training across environments has been explored in training a single self-play agent, which shows better generalization on both novel human users and unseen environments. In this work, we ask whether jointly modeling partner diversity and environment diversity yields stronger coordination with novel partners in novel environments. We introduce Cross Environment Co-Play (CECP), a two-stage framework
    
[^132]: IncrementDET：通过注册机制实现持续式AI生成音乐检测

    IncrementDET: Continual AI-Generated Music Detection via Enrollment

    [https://arxiv.org/abs/2611.32023](https://arxiv.org/abs/2611.32023)

    IncrementDET通过为每个AI音乐生成器注册一个独立的线性分类器“针”来实现持续式AI生成音乐检测，无需大规模重训练即可快速适应新生成器，同时保持对已学习生成器的检测性能和稳定的误报率。

    

    AI生成的音乐在音乐流媒体平台上已变得十分常见，听众往往难以将其与人类创作的音乐区分开来。因此需要AI音乐检测技术，但在固定生成器集合上训练的检测器面对未见过的生成器时常常失效。由于新生成器频繁涌现，更新一个通用检测器（即适用于所有生成器的单一模型）通常需要大量训练数据和计算资源，同时还存在在先前已学习生成器上性能退化的风险。为此，我们提出了IncrementDET，一种通过名为“注册”的过程来学习每个新生成器的检测器。对于每个生成器，IncrementDET训练一个线性分类器（我们称之为“针”），将该生成器的样本与一组固定的人类参考曲目区分开来。注册一个新生成器只需添加一个“针”，而先前构建的“针”保持不变。Sidak校正则确保整体误报率保持在目标水平。

    arXiv:2611.32023v1 Announce Type: cross  Abstract: AI-generated music has become common on music streaming platforms, and listeners often cannot distinguish it from music made by humans. AI-music detection is therefore needed, but detectors trained on a fixed set of generators often fail on unseen generators. As new generators emerge frequently, updating a universal detector, a single model for all generators, typically requires substantial training data and computation, while risking performance degradation on previously learned generators. We therefore introduce IncrementDET, a detector that learns each new generator by a process called enrollment. For each generator, IncrementDET trains one linear classifier, which we call a needle, to separate samples of that generator from a fixed set of human reference tracks. Enrolling a new generator adds one needle while the previously constructed needles remain unchanged. A Sidak correction keeps the overall false-positive rate at its target 
    
[^133]: PFedHASNet：一种用于多被试运动想象脑电解码的个性化联邦混合模拟-脉冲网络

    PFedHASNet: A Personalized Federated Hybrid Analog-Spike Network for Multi-Subject Motor Imagery EEG Decoding

    [https://arxiv.org/abs/2611.32019](https://arxiv.org/abs/2611.32019)

    该论文提出PFedHASNet，一种个性化联邦混合模拟-脉冲网络，通过可学习权重融合连续时间特征与LIF脉冲响应，并借助客户端私有的残差适配器和分类器，在保护数据隐私的前提下实现多被试运动想象脑电的个性化解码。

    

    运动想象脑电（MI-EEG）数据通常在不同被试或机构之间分散采集与存储，且脑电的采集和标注成本高昂。仅基于单个被试数据训练模型会限制对其他被试信息的利用，而集中收集原始脑电记录则会引发隐私和数据治理方面的担忧。联邦学习（FL）能够在不共享原始脑电数据的情况下，实现分布式客户端之间的协同模型训练，为利用多个被试的信息提供了一种实用的方法。据此，我们提出了PFedHASNet，一种用于运动想象脑电解码的个性化联邦混合模拟-脉冲网络。共享编码器通过可学习的融合权重，整合了连续值时间特征、泄漏积分发放（LIF）脉冲响应以及通道级的对数方差统计量。客户端私有的残差适配器和分类器执行面向特定被试的特征精炼……

    arXiv:2611.32019v1 Announce Type: new  Abstract: Motor imagery electroencephalography (MI-EEG) data are often collected and stored separately across subjects or institutions, and EEG acquisition and annotation remain costly. Training models solely on individual subject data limits the use of information from other subjects, whereas centralizing raw EEG recordings raises privacy and data governance concerns. Federated learning (FL) enables collaborative model training across distributed clients without sharing raw EEG data, providing a practical approach to leveraging information from multiple subjects. Accordingly, we propose PFedHASNet, a personalized federated hybrid analog-spike network for MI-EEG decoding. The shared encoder integrates continuous-valued temporal features, leaky integrate-and-fire (LIF) spike responses, and channel-wise log-variance statistics through learnable fusion weights. Client-private residual adapters and classifiers perform subject-specific feature refineme
    
[^134]: 面向个人AI智能体分层授权的结构感知Transformer

    Structure-Aware Transformers for Hierarchical Authorization in Personal AI Agents

    [https://arxiv.org/abs/2611.32018](https://arxiv.org/abs/2611.32018)

    提出PATH架构，通过层级感知注意力掩码与父门控概率机制，为个人AI智能体预测与权限层级结构一致的用户授权偏好，从而在保障隐私安全的同时减轻手动审查权限请求的负担。

    

    个人AI智能体可以代表用户执行复杂的多步骤任务，但这一过程通常涉及大量的权限请求。逐一手动审查这些请求非常耗时，而不加区分地全部批准则会带来隐私与安全风险。预测用户的授权偏好是一种既减轻负担又保留用户控制权的方式。然而，授权预测极具挑战性，因为权限请求往往呈层级结构，且观察到的用户决策稀疏且可能相互关联。为应对这些挑战，我们提出了分层父门控授权Transformer（PATH），用于预测用户对个人AI智能体的授权偏好。PATH是一种新颖的神经网络架构，它将层级感知的注意力掩码与父门控概率公式相结合，以确保预测结果与层级结构保持一致。层级嵌入进一步支持从稀疏观察中学习。

    arXiv:2611.32018v1 Announce Type: cross  Abstract: Personal AI agents can carry out complex, multi-step tasks on users' behalf, but doing so often involves numerous permission requests. Manually reviewing each request is time-consuming, while indiscriminate approval poses privacy and security risks. Predicting users' authorization preferences offers a way to reduce this burden while preserving user control. However, authorization prediction is challenging because permission requests are often hierarchical and observed user decisions are sparse and potentially correlated. To address these challenges, We propose the Parent-gated Authorization Transformer for Hierarchies (PATH) to predict users' authorization preferences for personal AI agents. PATH is a novel neural network architecture that combines hierarchy-aware attention masks with a parent-gated probability formulation to ensure hierarchy-consistent predictions. Hierarchy-level embeddings further support learning from sparse observ
    
[^135]: 用还是不用评论家：这是个问题

    To Critic Or Not To Critic

    [https://arxiv.org/abs/2611.32013](https://arxiv.org/abs/2611.32013)

    本文从梯度方向的偏差-方差权衡入手，揭示PPO（带评论家）与GRPO（不带评论家）之间没有普适的优劣答案，方法选择取决于批次大小、对评论家的依赖程度和任务复杂度等相互作用因素。

    

    随着大语言模型处理越来越长、越来越复杂的任务，理解评论家网络何时能提升强化学习（RL）的效率变得日益重要。近端策略优化（PPO）通过训练评论家网络来降低方差并提供细粒度的信用分配，而组相对策略优化（GRPO）则舍弃评论家网络，从而避免了基于评论家价值估计进行自举所带来的偏差。尽管近期基于评论家的方法相对于GRPO报告了出色的性能，但何时选择哪种方法仍不明确。我们认为这个问题没有普适的答案；相反，方法的选择取决于多个相互作用的因素，包括批次大小、对评论家的依赖程度以及任务复杂度。我们从梯度方向的角度研究偏差-方差的权衡，衡量随着批次大小的增加，每种方法的梯度估计与真实策略梯度的契合程度。在共享的演员目标函数下，我们……

    arXiv:2611.32013v1 Announce Type: cross  Abstract: As large language models tackle longer and more complex tasks, understanding when a critic improves the efficiency of reinforcement learning (RL) becomes increasingly important. Proximal Policy Optimization (PPO) trains a critic to reduce variance and provide fine-grained credit assignment, while Group Relative Policy Optimization (GRPO) drops the critic and avoids the bias of bootstrapping from the critic's value estimates. Although recent critic-based methods have reported strong performance relative to GRPO, it remains unclear when each approach is preferable. We argue that there is no universal answer; instead, the choice depends on several interacting factors, including batch size, critic reliance, and task complexity. We study the bias-variance tradeoff in terms of gradient direction, measuring how closely each method's gradient estimates align with the true policy gradient as batch size grows. Under a shared actor objective, we 
    
[^136]: LIBRA：面向时间高效大语言模型训练的批大小与并行度运行时协同调优

    LIBRA: Runtime Co-Tuning of Batch Size and Parallelism for Time-Efficient LLM Training

    [https://arxiv.org/abs/2611.32008](https://arxiv.org/abs/2611.32008)

    LIBRA首次将批大小与并行配置协同调优，以qNEHVI多目标搜索为核心，结合层次化剪枝和采集引导的并行评估，实现最小化大语言模型训练端到端时间的轻量级运行时框架。

    

    摘要：训练大语言模型（LLM）需要同时选择批大小和并行配置。现有系统将这些决策割裂开来——为吞吐量调优并行度，为收敛性调优批大小——但仅凭其中任何一个目标都无法决定训练效率。我们提出LIBRA，一个轻量级的运行时框架，通过协同调优批大小和并行配置来最小化端到端训练时间。LIBRA将吞吐量和收敛性视为两个带噪声的目标，并使用并行噪声期望超体积改进（qNEHVI）在其联合空间中进行搜索。为了使这一运行时搜索高效，我们提出了层次化剪枝方法：首先通过系统级性能估计剔除不可行和低吞吐量的并行配置，然后通过基于临界批大小的收敛监测来缩小批大小候选范围。我们进一步设计了采集引导的并行评估，用以评估存续的候选配置……

    arXiv:2611.32008v1 Announce Type: cross  Abstract: Training large language models (LLMs) requires choosing both a batch size and a parallel configuration. Existing systems make these decisions in isolation, tuning parallelism for throughput and batch size for convergence, but neither objective alone determines training efficiency. We present LIBRA, a lightweight runtime framework that co-tunes batch size and parallel configuration to minimize end-to-end time. LIBRA treats throughput and convergence as two noisy objectives and searches their joint space using parallel noisy expected hypervolume improvement (qNEHVI). To make this runtime search efficient, we propose hierarchical pruning, which removes infeasible and low-throughput parallel configurations through system-level performance estimation and then narrows batch-size candidates through critical-batch-size-aware convergence monitoring. We further design acquisition-guided parallel evaluation (AGPE), which evaluates surviving candi
    
[^137]: 将语言模型锚定于受众显性行为

    Grounding Language Models in Revealed Audience Behavior

    [https://arxiv.org/abs/2611.31999](https://arxiv.org/abs/2611.31999)

    该论文提出将语言模型在受众真实的点击、点赞、观看记录上进行微调（即“受众显性行为锚定”），使同一个8B模型预测标题点击的准确率从约50%提升至86.5%，并证明“直接询问模型”与“行为锚定”之间的差距主要源于训练信号，而非模型规模或提示方式。

    

    语言模型正越来越多地被用于模拟受众对某个标题或帖子的反应，而此类模拟器通常以人们“说”的内容作为验证基准。我们则用人们“做”的事来验证它们。当一个模型在受众实际点击、点赞或观看的记录上进行微调时，我们称其为“锚定于受众显性行为”。我们不仅追问此前研究所发现的“直接询问模型不如行为锚定”这一现象是否成立，还进一步追问差距为何存在、存在于何处。为何存在：这一差距源于训练信号本身的特性，而非主要取决于模型规模或提示方式。同一个80亿参数（8B）模型，在被直接询问读者会点击哪个标题时，正确率约为50%；而经过点击行为锚定后，正确率达到86.5%；我们尝试的最强提示方法也只能与一个文本分类器相当；在我们的实验中，一个经过行为锚定的0.6B模型击败了所有采用提示方式的前沿模型；此外，在一项预先注册的检验中，将点击数据按计数还是按份额进行拟合，未检测到任何差异。（原文摘要在此处截断）

    arXiv:2611.31999v1 Announce Type: new  Abstract: Language models are increasingly used to simulate how an audience will respond to a headline or a post, and such simulators are validated against what people say. We validate them against what people do. We call a model grounded in revealed audience behavior when it is fine-tuned on records of what audiences actually clicked, liked or watched, and we ask not only whether asking a model falls short of grounding it, as prior work found, but why and where. Why: the gap is a property of the training signal, not mainly of model size or prompting. The same 8B model is right about half the time when asked which headline readers will click and 86.5% of the time when grounded in the clicks; the strongest prompting we tried only matches a text classifier; a grounded 0.6B model beats every prompted frontier model in our runs; and whether the clicks are fitted as counts or as shares makes no detectable difference in a pre-registered test. The asked 
    
[^138]: ADE：一种面向潜在世界模型的零训练遗忘方法

    ADE: A Zero-Training Unlearning Method for Latent World Models

    [https://arxiv.org/abs/2611.31998](https://arxiv.org/abs/2611.31998)

    提出零训练闭式遗忘算法ADE，仅利用缓存的预训练统计矩即可从潜在世界模型中精准切除后门触发器与过时物理机制的污染，且无需原始保留数据、不扭曲潜在流形、不损害有效控制响应。

    

    潜在世界模型从离线轨迹中学习紧凑的循环动力学，但已部署的模拟器可能保留后门触发器或来自过时物理机制的陈旧转移。现有的遗忘方法在自回归物理场景中失效：梯度更新会扭曲潜在流形并加剧滚动预测漂移，朴素的输出清零会在触发器出现时抑制有效的控制响应，而单步删除会在循环隐藏状态内部留下损坏。我们提出了分析动力学切除（ADE），这是一种面向潜在世界模型的确定性、零训练的闭式遗忘算法。ADE 仅利用缓存的预训练统计矩，通过单次矩阵降更新恢复保留数据的协方差，无需原始保留数据，并将闭式编辑级联作用于输入瓶颈、隐藏状态和下游头部。将正交切除与白化循环阻尼和均值保持仿射校准相结合，可中和流形外污染（原文在此处截断）。

    arXiv:2611.31998v1 Announce Type: cross  Abstract: Latent world models learn compact recurrent dynamics from offline trajectories, yet deployed simulators can retain backdoor triggers or stale transitions from outdated physical regimes. Existing unlearning methods fail in autoregressive physics: gradient updates distort latent manifolds and compound rollout drift, naive output zeroing suppresses valid control responses on trigger onset, and single-step deletion leaves corruption inside recurrent hidden states. We propose Analytical Dynamics Excision (ADE), a deterministic, zero-training closed-form unlearning algorithm for latent world models. Using only cached pretraining moments, ADE recovers retain covariances via single-pass matrix downdating without raw retain data and cascades closed-form edits across the input bottleneck, hidden states, and downstream heads. Pairing orthogonal excision with whitened recurrent damping and mean-preserving affine calibration neutralizes out-of-mani
    
[^139]: 草图化多分类Softmax回归中度量依赖的标度律

    Metric-Dependent Scaling Laws for Sketched Multiclass Softmax Regression

    [https://arxiv.org/abs/2611.31995](https://arxiv.org/abs/2611.31995)

    该论文在高斯师生模型下证明，草图化多分类Softmax回归中交叉熵超额风险与0-1超额风险共享由分辨率项加“有效学习维度除以样本量”构成的共同谱尺度，并据此给出率最优的草图规模与训练时程选择。

    

    我们研究了草图化多分类Softmax回归中，采用一般稳定确定性学习率的全批量梯度下降的标度律。在一个具有幂律特征谱与教师谱的高斯师生模型中，学生观测到一个随机特征草图，并在整个优化过程中重复使用有限的带标签样本。草图化通常使学生模型处于误设状态，因此分析必须同时考虑近似误差、非线性有限时间学习，以及由数据重复使用引入的固定经验扰动。在一个以指数概率出现的草图事件上，我们证明交叉熵超额风险与超额0-1风险共享一个共同的谱尺度，该尺度由一个分辨率项加上有效学习维度除以样本量构成；师生决策不一致程度则遵循该尺度的平方根。由此得到的平衡给出了率最优的草图规模与累积训练时程选择，合成实验支持了所预测的（标度律）……

    arXiv:2611.31995v1 Announce Type: new  Abstract: We study scaling laws for full-batch gradient descent with general stable deterministic learning rates in sketched multiclass softmax regression. In a Gaussian teacher student model with power law feature and teacher spectra, the student observes a random feature sketch and reuses a finite labeled sample throughout optimization. Sketching generally makes the student misspecified, so the analysis must jointly account for approximation, nonlinear finite time learning, and the fixed empirical perturbation induced by data reuse. On an exponentially likely sketch event, we show that cross entropy excess and excess zero one risk share a common spectral scale, consisting of a resolution term and an effective learned dimension divided by sample size; teacher student decision disagreement follows the square root of that scale. The resulting balance gives rate optimal sketch and cumulative horizon choices, and synthetic experiments support the pre
    
[^140]: 边缘规则能看到多少依赖性？扩散语言模型并行解码中的总相关性

    How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding

    [https://arxiv.org/abs/2611.31990](https://arxiv.org/abs/2611.31990)

    该论文量化了扩散语言模型并行解码中基于边缘分布的解码器对提交token集合条件总相关性（即依赖性损失）的预测能力，发现其预测准确性主要来源于边界本身，且估计值在散文文本上通常比精确 KL 代价低 0.4 到 1.1 比特。

    

    掩码扩散语言模型之所以快速，是因为它在每一步提交多个token。每个token都是从其自身的边缘分布中采样的，因此token之间的依赖性丢失了，而该损失的大小就是所提交集合的条件总相关性 C。大多数解码器从每个位置的边缘分布中选择集合，这可以约束 C 但无法确定它。我们探讨了它们对 C 的预测能力如何。我们在来自两个家族的四个掩码扩散语言模型上估计固定候选集合的 C。在四个模型中的三个上，最佳的仅边缘预测器偏差约为一比特（0.74 到 1.17），比仅知道边界的基线好 1.8 到 3.7 倍，而这种准确性大部分来自边界本身。在一个子集上，我们的估计对集合的排序效果类似于并行提交的精确 KL 代价，但在散文文本上以及当文本其余部分仍被掩码时，比该代价中位数低 0.4 到 1.1 比特。Dream-instruct 是例外，主要体现在散文文本上，其条件……（摘要截断）

    arXiv:2611.31990v1 Announce Type: new  Abstract: A masked diffusion language model is fast because it commits several tokens per step. Each token is sampled from its own marginal, so the dependency between the tokens is lost, and the size of that loss is the conditional total correlation C of the committed set. Most decoders choose the set from per-position marginals, which can bound C but cannot determine it. We ask how well they predict it. We estimate C for fixed candidate sets on four masked diffusion LMs from two families. On three of the four, the best marginal-only predictor is off by about one bit (0.74 to 1.17), 1.8 to 3.7 times less than a baseline that knows only the bound, and most of this accuracy comes from the bound itself. On a subset, our estimate ranks sets like the exact KL cost of a parallel commit, but sits a median 0.4 to 1.1 bits below it on prose and when the rest of the text is still masked. Dream-instruct is the exception, mainly on prose, where its conditiona
    
[^141]: 发现带宽高效的预取器以缩小缓存层次结构

    Discovering Bandwidth-Efficient Prefetchers to Shrink the Cache Hierarchy

    [https://arxiv.org/abs/2611.31983](https://arxiv.org/abs/2611.31983)

    本文提出DuoArchitect——一个自主智能体框架，通过多保真度仿真与多核协同演化，联合发现覆盖L1D、L2和LLC各层级的协调带宽高效预取策略，其仿真速度比传统集群评估快3-26倍，从而帮助缩小缓存层次结构、释放芯片面积。

    

    共享末级缓存（LLC）日益占据服务器处理器芯片面积的主导地位，迫使架构师在核心数量与缓存容量之间进行取舍。虽然智能数据预取提供了一条回收这些硅片面积的途径，但为多核系统设计协调的多级预取策略却受制于耗时一整天的周期级仿真。这一仿真瓶颈甚至限制了基于LLM的微架构探索；最先进的ArchAgent系统在单核基准测试中一次只演化一个缓存层级，仅在最后才加入多核配置，并且在多核共享缓存竞争下落后于人类冠军方案。我们提出了DuoArchitect，一个用于多保真度、多核协同发现预取器的自主智能体框架。智能体在L1D、L2和LLC层级上共同演化评估代理与协调的预取策略，所构建的仿真代理比集群评估快3-26倍……

    arXiv:2611.31983v1 Announce Type: new  Abstract: Shared last-level caches (LLCs) increasingly dominate server processor die area, forcing architects to trade core counts for cache capacity. While intelligent data prefetching offers a pathway to reclaim this silicon, designing coordinated, multi-level prefetching policies for multi-core systems is bottlenecked by day-long, cycle-level simulations. This simulation wall limits even LLM-based microarchitecture discovery; the state-of-the-art ArchAgent system evolves one cache level at a time on single-core benchmarks, adds multi-core configurations only at the end, and trails the human champion under multi-core shared cache contention. We introduce DuoArchitect, an autonomous agentic framework for multi-fidelity, multi-core co-discovery of prefetchers. Agents jointly evolve evaluation proxies and coordinated prefetching policies across the L1D, L2, and LLC hierarchies, constructing simulation proxies 3-26x faster than cluster evaluation an
    
[^142]: 有界损失下留一法的极小极大最优界

    A Minimax Optimal Leave One Out Bound for Bounded Losses

    [https://arxiv.org/abs/2611.31972](https://arxiv.org/abs/2611.31972)

    本文针对有界损失下的留一预测问题给出了极小极大最优损失界，证明超额损失阶为 $\min\{B, \sqrt{Br\log M/n} + B\log M/n\}$，且单个确定性规则即可对所有 $r$ 达到该速率，并同时给出了匹配的下界。

    

    我们研究了在 $n$ 个固定输入点上、由 $M$ 个实值专家进行的适当留一预测问题。每个标签依次被隐藏，学习器利用其余标签在留出的输入点上选择某位专家的预测。我们确定了这类规则能够多好地匹配最佳单个专家在整个样本上的平均损失。对于任何有界损失——即关于绝对误差连续、非递减且在零点取值为零的函数——设其最大值为 $B$。在专家数量为 $M$、最佳专家平均损失至多为 $r$ 的样本类上，极小极大超额损失的阶为 $\min\{B, \sqrt{\frac{Br\log M}{n}}+\frac{B\log M}{n}\}$。对于每个专家类和每种损失，存在一个确定性规则对所有 $r\in[0,B]$ 都能达到该速率。相应的匹配下界即使在给定 $r$ 的随机适当规则下依然成立。

    arXiv:2611.31972v1 Announce Type: cross  Abstract: We study proper leave-one-out prediction with $M$ real-valued experts on $n$ fixed inputs. Each label is hidden in turn, and the learner uses the remaining labels to choose one of the experts' predictions at the held-out input. We determine how closely such a rule can match the average loss of the best single expert on the full sample. For any bounded loss that is a continuous nondecreasing function of absolute error and vanishes at zero, let $B$ denote its maximum value. Over classes of size $M$ and samples on which the best expert has average loss at most $r$, the minimax excess loss is of order $\min\{B, \sqrt{\frac{Br\log M}{n}}+\frac{B\log M}{n}\}$. For each class and loss, one deterministic rule attains this rate for every $r\in[0,B]$. The matching lower bound holds even for randomized proper rules given $r$.
    
[^143]: 预测主干，嫁接动态：一种解耦的时间序列预测范式

    Predict Backbone, Graft Dynamics: A Decoupled Paradigm for Time Series Forecasting

    [https://arxiv.org/abs/2611.31970](https://arxiv.org/abs/2611.31970)

    该论文提出GRAFT解耦预测范式，将时间序列预测分解为可靠的主干预测与基于历史证据检索的动态嫁接恢复，有效保留突变、激增和转折点等高效用局部动态。

    

    现代长期预测模型通常针对逐点精度进行优化，这种优化方式偏向于主导的低频主干信号，但往往会抑制高效用的局部动态，例如突变、瞬时激增和转折点。我们提出了“预测主干，嫁接动态”这一解耦预测范式，将稳定的结构性预测与基于证据的动态恢复分离开来。我们将其实例化为GRAFT：由基础预测器预测可靠的轨迹主干，由一条专门的嫁接路径通过基于检索的证据、相位感知释放、通道级控制和恢复协同优化来恢复历史上重复出现的动态。与修正残差误差的通用事后修正方法不同，GRAFT将第二阶段表述为基于证据的动态嫁接，即将历史响应证据校准为可迁移的动态，并在相位、通道等条件下选择性地将其嫁接到主干上。

    arXiv:2611.31970v1 Announce Type: new  Abstract: Modern long-term forecasters are typically optimized for point-wise accuracy, which rewards dominant low-frequency backbones but often suppresses high-utility local dynamics such as abrupt variations, transient surges, and turning points. We propose Predict Backbone, Graft Dynamics, a decoupled forecasting paradigm that separates stable structural prediction from grounded dynamic restoration. We instantiate it as GRAFT, where a base forecaster predicts the reliable trajectory backbone, and a dedicated grafting path restores historically recurrent dynamics through retrieval-based evidence, phase-aware release, channel-wise control, and restoration-cooperative optimization. Unlike generic post-hoc revision methods that correct residual errors, GRAFT formulates the second stage as grounded dynamic grafting, where historical response evidence is calibrated into transferable dynamics and selectively grafted onto the backbone under phase-, cha
    
[^144]: 多词元预测中的信息损失：成因与解决方案

    On the Information Loss of Multi-Token Prediction: Origin and Solution

    [https://arxiv.org/abs/2611.31966](https://arxiv.org/abs/2611.31966)

    该论文揭示了多词元预测中误差累积源于参考目标间的跨块依赖性，并提出Dual-MTP方法，通过在完整与随机掩码上下文下双重监督共享预测头来缓解信息损失。

    

    多词元预测通过分块预测加速自回归解码，但在按顺序生成各块的过程中仍容易受到误差累积的影响。为研究这一局限性，我们分析了分块级别与序列级别累积信息损失（CIL）之间的区别。具体而言，当模型使用其自身先前的输出进行生成而无法访问参考答案时，输入条件下的差距等于参考目标之间的跨块依赖性。为了在这些生成前缀不完美的情况下改进预测，我们提出了Dual-MTP，它在完整上下文和随机掩码上下文下对共享预测头进行监督。完整前缀监督与上下文扰动共同提供了互补的训练信号，同时保留了MTP架构。为了刻画这一训练目标，我们使用归一化双重损失来约束模型预测分布之间的散度……

    arXiv:2611.31966v1 Announce Type: new  Abstract: Multi-token prediction (MTP) accelerates autoregressive decoding through block-wise prediction, but remains vulnerable to error accumulation across sequentially generated blocks. To investigate this limitation, we analyze the distinction between block-wise and sequence-level cumulative information loss (CIL). Specifically, when the model generates using its own previous outputs without access to reference answers, the input-conditional gap equals the cross-block dependence of the reference targets. To improve prediction when these generated prefixes are imperfect, we propose Dual-MTP, which supervises shared prediction heads under full and randomly masked contexts. Together, complete-prefix supervision and context perturbations provide complementary training signals while retaining the MTP architecture. To characterize this training objective, we use the normalized dual loss to bound the divergence between the model's predictive distribu
    
[^145]: 探测黑箱：在仅前向物理模拟器中在线训练嵌入式神经网络

    Sounding the Black Box: Online Training of Embedded Neural Networks in Forward-Only Physical Simulators

    [https://arxiv.org/abs/2611.31964](https://arxiv.org/abs/2611.31964)

    该论文提出FORGE在线训练框架，通过将精确的网络导数与黑箱模拟器动力学解耦，仅利用不可微分求解器的前向运行即可训练嵌入PDE求解器中的神经网络，从而突破了传统BPTT方法对可微分求解器接口的依赖。

    

    混合物理-神经模拟器将神经网络嵌入到偏微分方程（PDE）求解器中，以建模未解析的亚网格动力学，同时保留已解析的物理过程和守恒定律。由于闭合输出会在时间步之间产生反馈，单步离线回归无法直接最小化多步滚动误差，这促使了对在线轨迹优化的需求。当求解器导数可用时，通过求解器进行的时间反向传播（BPTT）是标准的在线解决方案，然而许多实际运行中的Fortran和C/C++模拟器仅暴露不可微分的前向接口。我们提出了FORGE（仅前向响应与高斯-牛顿估计），这是一个仅通过前向运行就能训练嵌入式网络的在线框架，它将精确的网络导数与黑箱模拟器动力学解耦。第一阶段通过沿输出正交权重方向的低维高斯-牛顿步骤对滚动过程进行预热，以抵消多层……

    arXiv:2611.31964v1 Announce Type: new  Abstract: Hybrid physics-neural simulators embed neural networks in partial differential equation (PDE) solvers to model unresolved subgrid dynamics while preserving resolved physics and conservation laws. Because closure outputs feed back across time steps, single-step offline regression does not directly minimize multi-step rollout error, motivating online trajectory optimization. Solver backpropagation through time (BPTT) is the standard online solution when solver derivatives are available, yet many operational Fortran and C/C++ simulators expose only non-differentiable, forward-only interfaces. We introduce FORGE (Forward-Only Response and Gauss-Newton Estimation), an online framework that trains embedded networks from forward runs alone by decoupling exact network derivatives from black-box simulator dynamics. Stage I warms up the rollout via a low-dimensional Gauss-Newton step along output-orthonormal weight directions, canceling multi-laye
    
[^146]: 网络智能体还能被检测到吗？用协同规避手段对机器人防御系统进行压力测试

    Are Web Agents Still Detectable? Stress-Testing Bot Defenses with Coordinated Evasion

    [https://arxiv.org/abs/2611.31962](https://arxiv.org/abs/2611.31962)

    论文提出了DarkAgent自主网络智能体，通过结合隐藏自动化痕迹的Chromium浏览器与联合生成逼真按键事件及时间间隔的类人鼠标键盘模型，实现跨浏览器指纹、自动化控制和交互行为的协同规避，实验表明现有机器人防御系统难以检测此类攻击。

    

    AI网络智能体能够自主浏览网站并完成各类任务。随着其能力和应用的不断扩展，评估机器人防御系统需要开展对抗性压力测试。尽管先前的研究表明现有智能体仍可被检测到，但目前尚不清楚：面对那些在浏览器指纹、自动化控制以及交互行为等多个层面协同规避检测的攻击者时，机器人防御系统的实际效力究竟如何。为了研究这一威胁，我们提出了DarkAgent——一个自主网络智能体，它结合了一个经过改造、可隐藏自动化痕迹的Chromium浏览器分支，以及能够生成类人鼠标和键盘轨迹的生成模型。针对现有键盘模拟器的局限性，我们的键盘模型在可靠重现目标文本的同时，联合生成逼真的按键事件及其时间间隔。已发布的智能体检测器会将我们生成的行为判定为人类，而经过训练的分类器也发现要识别这些行为要困难得多……

    arXiv:2611.31962v1 Announce Type: cross  Abstract: AI web agents can autonomously navigate websites and complete various tasks. As their capabilities and adoption expand, evaluating bot defenses requires adversarial stress tests. Although prior work demonstrates that existing agents can remain detectable, it remains unclear how effective bot defenses are against adversaries that coordinate evasion across browser fingerprints, automation controls, and interaction behavior. To investigate this threat, we introduce DarkAgent, an autonomous web agent that combines a forked Chromium browser that conceals automation artifacts and generative models that produce human-like mouse and keyboard traces. To address the limitations of existing keyboard mimickers, our keyboard model jointly generates realistic key events and their timings while reliably reproducing the target text. Released agent detectors accept our generated behavior as human, and trained classifiers find it substantially harder to
    
[^147]: DUST：方向不确定性感知与尺度不变的迁移学习

    DUST: Directional Uncertainty-aware and Scale-invariant Transfer Learning

    [https://arxiv.org/abs/2611.31948](https://arxiv.org/abs/2611.31948)

    提出了一种基于源回归参数尺度不变方向分布的概率迁移学习框架DUST，能够同时纳入源参数的不确定性并实现尺度不变且稳健的迁移学习。

    

    迁移学习旨在通过借鉴相关外部源数据集的信息来提高目标回归任务的效率。现有方法通过强制目标回归参数与源回归参数之间的欧氏距离接近来实现迁移，或者最近通过基于角度的惩罚来鼓励二者的方向对齐。虽然这类基于角度的惩罚减轻了迁移学习方法对源和目标回归系数之间尺度差异的敏感性，但它们仍然依赖于源回归参数的点估计，因此没有提供直接机制来纳入源信息中的不确定性。为此，我们提出了一个用于迁移学习的概率框架，该框架在源回归参数估计的尺度不变方向分布层面进行操作，而非依赖其点估计，从而实现稳健的、尺度不变的且不确定性感知的迁移学习。

    arXiv:2611.31948v1 Announce Type: cross  Abstract: Transfer learning seeks to improve efficiency in a target regression task by borrowing information from related external source dataset(s). Existing approaches achieve this by enforcing Euclidean proximity between the target and source regression parameters or, more recently, by encouraging their directional alignment through angle-based penalties. While such angle-based penalties mitigate the sensitivity of the transfer learning methods to the scale differences between source and target regression coefficients, they still rely on a point estimate of the source regression parameter and thus provide no direct mechanism to incorporate uncertainty in the source information. To that end, we propose a probabilistic framework for transfer learning that operates at the level of a \emph{scale-invariant directional distribution} of the source regression parameter estimate, rather than its point estimate, to enable robust, scale-invariant and un
    
[^148]: MELODI：面向轻量级排序的相似性-身份解耦

    MELODI: Similarity-Identity Decoupling for Lightweight Ranking

    [https://arxiv.org/abs/2611.31940](https://arxiv.org/abs/2611.31940)

    本文提出相似性-身份解耦框架 MELODI，证明了相似度算子线性条件下“先池化后相似”与“先相似后池化”等价，从而以更高效的轻量级设计在公开及工业数据集上达到或超越现有排序基线。

    

    对于延迟受限的候选排序任务，我们提出了一种相似性-身份解耦框架，将目标-历史兼容性与仅目标信息分离开来。我们研究了计算池化相似度的两种顺序——先相似后池化与先池化后相似，并证明对于每个固定的目标，相似度算子在其历史参数上的线性性是两种顺序等价的充分条件，从而为计算成本更低的先池化后相似形式提供了理论依据。我们将该框架实例化为解耦身份与相似性的多头高效轻量算子（MELODI），其相似性分支将目标条件化的多头历史池化与一个对其池化历史输入呈线性的单一无残差交叉层相结合，而其身份分支则依赖于具体问题设定。在公开基准数据集上，MELODI 达到或超越了所评估的目标感知与目标无关基线。在两个大规模工业数据集上，它的……

    arXiv:2611.31940v1 Announce Type: cross  Abstract: For latency-constrained candidate ranking, we propose a similarity-identity decoupling framework that separates target-history compatibility from target-only information. We study two orders for computing pooled similarity, similarity-then-pool and pool-then-similarity, and identify linearity of the similarity operator in its history argument, for each fixed target, as sufficient for their equivalence, justifying the cheaper pool-then-similarity form. We instantiate the framework as multi-head efficient lightweight operator for decoupling identity and similarity (MELODI), whose similarity branch combines target-conditioned multi-head history pooling with a single residual-free cross layer that is linear in its pooled-history input and whose identity branch is problem-dependent. On public benchmark datasets, MELODI matches or surpasses the evaluated target-aware and target-agnostic baselines. On two large-scale industrial datasets, its 
    
[^149]: 超越最后一轮循环：使用从模型自身外推得到的教师进行强化学习

    Beyond the Last Loop: Reinforcement Learning with a Teacher Extrapolated from the Model Itself

    [https://arxiv.org/abs/2611.31939](https://arxiv.org/abs/2611.31939)

    本文提出ELGO方法，通过从循环语言模型自身首轮到末轮外推出一个比最后一轮更强的“内生教师”分布，并在GRPO中加入逐token密集蒸馏项，让教师随学生共同进化，从而在不依赖外部教师的情况下解决了循环模型使用可验证奖励进行强化学习时奖励信号稀疏的难题。

    

    循环语言模型对每个token多次应用同一个共享的transformer堆栈，且每一轮循环都可以通过共享的语言模型头读出输出。使用可验证奖励对其进行后训练非常困难：结果奖励每次响应只携带一个比特，将其分散到各轮循环上会使中间循环在从最后一轮循环采样的token上进行训练，而蒸馏则需要一个要么忽略奖励、要么来自外部更大模型的教师。我们观察到，循环模型本身已经包含一个比其最后一轮更强的教师：由第一轮循环向最后一轮外推得到的下一个token分布。ELGO（Extrapolated-Loop Group-relative Optimization，外推循环组相对优化）在GRPO基础上增加了一个朝向该分布的密集逐token蒸馏项。该教师由模型的一个锚点副本计算得出，该副本保持冻结，一旦学生相对于教师的KL散度降至阈值以下，它就会被当前模型替换，因此教师会随学生一同改进……

    arXiv:2611.31939v1 Announce Type: new  Abstract: Looped language models apply one shared transformer stack several times per token, and every loop can be read out through the shared language-model head. Post-training them with verifiable rewards is hard: the outcome reward carries one bit per response, spreading it over loops trains intermediate loops on tokens sampled from the last loop, and distillation needs either a teacher that ignores the reward or an external, larger model. We observe that a looped model already contains a teacher stronger than its last loop: the next-token distribution extrapolated from the first loop through the last. ELGO (Extrapolated-Loop Group-relative Optimization) adds to GRPO a dense per-token distillation term towards this distribution. The teacher is computed from an anchor copy of the model, which stays frozen and is replaced by the current model once the student's KL to the teacher falls below a threshold, so the teacher improves with the student wi
    
[^150]: 从统一表示到共享动态：重新思考通用时间序列基础模型（UniTFU）

    Rethinking Universal Time Series Foundation Models from Unified Representations to Shared Dynamics (UniTFU)

    [https://arxiv.org/abs/2611.31921](https://arxiv.org/abs/2611.31921)

    该论文提出时间序列基础单元（TFUs）作为从多尺度时间片段中发现的可复用构建模块，通过显式刻画跨领域共享的重复性局部动态模式，而非依赖隐式统一表示，来提升时间序列基础模型的跨领域泛化能力。

    

    时间序列基础模型（TSFMs）在跨领域预测中受到越来越多的关注。现有工作主要依赖大规模预训练和通用建模范式。然而，前者严重依赖于预训练数据的多样性和领域覆盖范围；后者则在异构领域之间学习统一表示，却未能显式识别各领域之间共享的时间规律，从而限制了稳健的跨领域泛化能力。本工作通过刻画跨领域中重复出现的局部动态的共同性来探索跨领域泛化，而非仅仅依赖隐式学习到的时间表示。为了捕捉这种跨领域共同性，我们提出了时间序列基础单元（TFUs），作为从多尺度时间片段中发现的可复用构建模块，每个单元刻画一种跨领域共享的重复性局部动态模式。基于TFUs，我们……（原文摘要在此处截断）

    arXiv:2611.31921v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) have attracted increasing attention in cross-domain forecasting. Existing works mainly rely on large-scale pre-training and general modeling paradigms. However, the former heavily relies on the diversity and domain coverage of pre-training data. Meanwhile, the latter learns unified representations across heterogeneous domains without explicitly identifying the temporal regularities shared among them, limiting robust cross-domain generalization. This work explores cross-domain generalization by characterizing commonalities in recurring local dynamics across domains, rather than relying solely on implicitly learned temporal representations. To capture such cross-domain commonalities, we propose Time Series Foundation Units (TFUs) as reusable building blocks discovered from multi-scale temporal segments, each characterizing a recurring local dynamic pattern shared across domains. Building on TFUs, we de
    
[^151]: 元摊销认知建模

    Meta-Amortized Cognitive Modeling

    [https://arxiv.org/abs/2611.31914](https://arxiv.org/abs/2611.31914)

    该论文提出CogFormerV2元摊销框架，用单个网络一次性将贝叶斯推断摊销至整个认知模型空间，仅需在单GPU上训练约一天，即可对涵盖二元、多选项和连续决策的超过146亿个模型配置快速给出准确且校准良好的后验分布，从而免去每次模型修改后重新模拟与训练的成本。

    

    认知建模是一个高度迭代的过程。随着理论的发展与数据的积累，研究者需要反复修改模型结构、参数化方式、协变量以及实验设计。基于神经网络的摊销贝叶斯推断能够为单个模型提供快速估计，但每一次模型修改都需要重新进行模拟和训练。在此，我们提出了 CogFormerV2——一个元摊销框架，使研究者能够用单个网络来指定、修改和拟合认知模型，将摊销能力一次性覆盖整个模型空间。该网络仅需在单块GPU上训练约一天，即可在超过146亿个模型配置上给出准确且校准良好的后验分布，这些配置涵盖二元、多选项和连续决策任务，并具有不同的参数集合、协变量结构、试次数量和污染水平。它可以拟合来自不同行为任务的真实经验数据，并可通过……扩展到新模型。

    arXiv:2611.31914v1 Announce Type: new  Abstract: Cognitive modeling is highly iterative. Researchers repeatedly revise model structures, parameterizations, covariates, and experimental designs as theories evolve and data accumulates. Amortized Bayesian inference with neural networks provides rapid estimation for individual models, but every revision requires new simulations and training. Here, we introduce CogFormerV2, a meta-amortized framework that enables researchers to specify, revise, and fit cognitive models with a single network, carrying amortization across an entire space of models all at once. Trained once in about a day on a single GPU, it delivers accurate, well-calibrated posteriors across more than 14.6 billion model configurations spanning binary, multi-alternative, and continuous decisions, with varying parameter sets, covariate structures, trial counts, and contamination levels. It can fit empirical data from distinct behavioral tasks and extends to new models through 
    
[^152]: ReJEPA：面向潜在世界模型的有界预测调节

    ReJEPA: Bounded Predictive Regulation for Latent World Models

    [https://arxiv.org/abs/2611.31910](https://arxiv.org/abs/2611.31910)

    提出 ReJEPA，一种有界预测调节原则，将大误差的有界梯度与零附近的平滑二次行为相结合，解决了潜在世界模型训练中 L2 损失过拟合噪声帧与 L1 损失破坏局部平滑性的矛盾。

    

    潜在世界模型通过在抽象嵌入空间中预测未来状态、而非重建原始像素的方式来学习表示。然而，在多帧序列上进行训练时，并非所有帧和空间区域都具有同等的信息量或可预测性。我们认为，强迫世界模型最小化每一个预测误差可能是既不必要又有害的。在标准的 $L_2$ 损失下，无界的梯度会迫使编码器和预测器过度拟合噪声帧或瞬态帧，从而以牺牲核心物理结构为代价，同时还会加剧与表示正则化项之间的梯度冲突。相反，$L_1$ 损失虽然限制了大梯度，但其在零附近的恒定梯度幅值破坏了规划所需的局部平滑性。我们通过 **ReJEPA** 来化解这一矛盾——这是一种有界预测调节原则，它将大误差时的有界梯度与零附近的平滑二次行为相结合，并简单地通过（后续方法）进行实例化。

    arXiv:2611.31910v1 Announce Type: new  Abstract: Latent world models learn representations by predicting future states in an abstract embedding space rather than reconstructing raw pixels. When training across multi-frame sequences, however, not all frames and spatial regions are equally informative or predictable. We argue that forcing a world model to minimize every prediction error may be unnecessary and harmful. Under standard $L_2$ loss, unbounded gradients force the encoder and predictor to overfit noisy or transient frames at the expense of core physical structure, while amplifying gradient conflict with representation regularizers. Conversely, $L_1$ loss bounds large gradients, but its constant gradient magnitude near zero disrupts the local smoothness required for planning. We resolve this tension with \textbf{ReJEPA}, a bounded predictive regulation principle that combines bounded gradients for large errors with smooth quadratic behavior near zero, instantiated simply via the
    
[^153]: TQ-HQNet：一种面向异构数据跨领域学习的可训练性感知且量子比特高效的混合量子神经网络

    TQ-HQNet: A Trainability-Aware and Qubit-Efficient Hybrid Quantum Neural Network for Cross-Domain Learning on Heterogeneous Data

    [https://arxiv.org/abs/2611.31908](https://arxiv.org/abs/2611.31908)

    提出了TQ-HQNet混合量子神经网络，通过自适应潜在压缩、可训练性感知的紧凑变分电路搜索和浅层量子专家混合融合三种机制，在保持量子比特高效的同时，显著提升了跨异构数据领域学习的可训练性与鲁棒性。

    

    混合量子神经网络是含噪声中等规模量子计算（NISQ）中一个颇具前景的研究方向，但其实际效果往往受限于较差的可训练性、僵化的电路设计以及在异构真实任务上鲁棒性不足。本文提出了TQ-HQNet，一种面向异构数据跨领域学习的可训练性感知且量子比特高效的混合量子神经网络。该框架结合了三种机制：1）自适应潜在压缩，将特定领域的特征映射为量子比特兼容的表示；2）在紧凑的变分电路候选上进行可训练性感知的搜索；3）多个浅层量子专家的混合融合，在不显著增加量子比特成本的前提下提升鲁棒性。TQ-HQNet在四个公开数据集上进行了评估，涵盖神经系统疾病筛查、钓鱼攻击检测、金融走势预测和DNA序列分类。在所有领域中，所提出的（原文在此处截断）

    arXiv:2611.31908v1 Announce Type: new  Abstract: Hybrid quantum neural networks are a promising direction for noisy intermediate-scale quantum computing, but their practical effectiveness is often limited by poor trainability, rigid circuit design, and weak robustness across heterogeneous real-world tasks. This paper proposes TQ-HQNet, a Trainability-Aware and Qubit-Efficient Hybrid Quantum Neural Network for cross-domain learning on heterogeneous data. The proposed framework combines three mechanisms: 1) adaptive latent compression to map domain-specific features into qubit-compatible representations, 2) trainability-aware search over compact variational circuit candidates, and 3) hybrid fusion of multiple shallow quantum experts to improve robustness without increasing qubit cost excessively. TQ-HQNet is evaluated on four public datasets spanning neurological screening, phishing detection, financial direction prediction, and DNA sequence classification. Across all domains, the propos
    
[^154]: OrangeTree：一种支持多输入多输出长度的线性与树结构时间序列预测模型

    OrangeTree: A Linear and Tree-based Time Series Forecasting Model Supporting Multiple Input and Output Lengths

    [https://arxiv.org/abs/2611.31905](https://arxiv.org/abs/2611.31905)

    OrangeTree通过分段树编码器、逆树解码器以及范围加权与特征融合机制，提出了一种将模型参数与序列长度解耦的线性树结构架构，在统一框架内实现了多输入多输出长度的时间序列预测，并在六个基准上达到了SOTA性能。

    

    时间序列预测（TSF）模型存在一个关键的僵化问题：单一架构无法处理多输入多输出长度。目前的变通方法，如填充或自回归，往往会带来计算冗余或误差累积问题。我们提出了OrangeTree，这是一种线性与树结构相结合的架构，将模型参数与序列长度解耦。它采用分段树编码器和逆树解码器，在统一框架内将多长度的输入历史转化为多长度的输出预测。至关重要的是，范围加权器和特征融合器衔接了这些组件，动态选择并融合相关范围以构建最优的预测上下文。在六个基准数据集上的实验证实，OrangeTree实现了SOTA性能。在MIOL设置中，它在没有明显精度下降或计算量大幅增加的情况下，达到了与固定长度模型相当的精度。

    arXiv:2611.31905v1 Announce Type: new  Abstract: Time series forecasting (TSF) models suffer from a critical rigidity: the inability to handle Multiple Input and Output Lengths (MIOL) within a single architecture. Current workarounds, such as padding or autoregression, often incur computational redundancy or suffer from error accumulation. We propose OrangeTree, a linear and tree-based architecture that decouples model parameters from sequence lengths. It employs a Segment Tree Encoder and an Inverse Tree Decoder to translate multi-length input history into multi-length output predictions within a unified framework. Crucially, a Range Weighter and a Feature Fuser bridge these components, dynamically selecting and fusing relevant ranges into optimal contexts for forecasting. Experiments on six benchmarks confirm OrangeTree achieves the SOTA performance. In MIOL settings, it matches the accuracy of length-fixed models without substantial accuracy degradation or a substantial increase in 
    
[^155]: MIFLOW：用于参考校准在线流策略强化学习的动态近期行为先验

    MIFLOW: Dynamic Recent-Behavior Priors for Reference-Calibrated Online Flow-Policy Reinforcement Learning

    [https://arxiv.org/abs/2611.31902](https://arxiv.org/abs/2611.31902)

    MIFLOW通过用拟合最近执行动作的动态流匹配先验作为移动密度参考，对在线流策略强化学习的演员更新和评论家目标进行维度归一化校正，在抑制低密度不可靠动作的同时保持探索能力，从而在MuJoCo连续控制任务上取得有竞争力的性能。

    

    基于流的策略为连续控制强化学习提供了富有表现力的动作分布。在在线演员-评论家（actor-critic）学习中，当动作覆盖不足时，离策略价值估计可能不可靠，此时富有表现力的策略可能会将概率质量转移到评论家偏好的动作上。我们提出MIFLOW，该方法通过一个与最近执行动作拟合的、随时间自适应更新的流匹配先验来校准在线流策略学习。受“渠化”（canalization）概念作为路径依赖稳定化的定性类比启发，我们将该先验用作价值引导策略改进的移动密度参考。MIFLOW在演员更新和评论家目标中均采用近似近期动作对数密度进行维度归一化校正。这种校正提高了低密度动作的成本，同时保持这些动作仍然可用。在六个MuJoCo连续控制任务和五个随机种子上，MIFLOW取得了具有竞争力的平均回报曲线。

    arXiv:2611.31902v1 Announce Type: cross  Abstract: Flow-based policies provide expressive action distributions for continuous-control reinforcement learning. In online actor-critic learning, off-policy value estimates can be unreliable in poorly covered action regions, where expressive policies may shift probability mass toward critic-favored actions. We propose MIFLOW, which calibrates online flow-policy learning with a recency-adaptive flow-matching prior fitted to recently executed actions. Inspired by canalization as a qualitative analogy for path-dependent stabilization, we use the prior as a moving density reference for value-guided policy improvement. MIFLOW uses the approximate recent-action log-density in a dimension-normalized correction to both actor updates and critic targets. This correction increases the cost of low-density actions while keeping them available. Across six MuJoCo continuous-control tasks and five random seeds, MIFLOW achieves competitive mean return curves
    
[^156]: 最优臂识别中e-过程停止规则的通用框架

    A General Framework for E-process Stopping Rules in Best-Arm Identification

    [https://arxiv.org/abs/2611.31898](https://arxiv.org/abs/2611.31898)

    该论文提出了最优臂识别中e-过程停止规则的通用框架，通过e-过程分解将最优性条件分为三个部分并证明了渐近最优性的充分条件，进而推导出广义线性老虎机问题的渐近最优算法。

    

    我们在最优臂识别问题中提出了一种e-过程停止规则。e-过程停止规则与Chernoff停止规则的不同之处在于其对停止决策的假设结构。e-过程停止规则的主要优势在于停止规则设计的灵活性，事实上，它在多个老虎机问题上可以恢复先前提出的Chernoff停止规则。此外，我们通过e-过程的分解分析了δ-PAC算法的渐近最优性。该分解将最优性条件分离为三个部分：采样策略分配的收敛性、e-过程设计的效率以及老虎机问题的内在随机行为。随后，我们证明了老虎机算法渐近最优性的一个充分条件。作为e-过程停止规则的应用，我们推导了针对广义线性老虎机问题的渐近最优算法。

    arXiv:2611.31898v1 Announce Type: cross  Abstract: We propose an e-process stopping rule in the best-arm identification problem. The e-process stopping rule is different from the Chernoff stopping rule by its hypothesis structure on the stopping decision. The major benefit of an e-process stopping rule is the flexibility in stopping rule design and in fact, it recovers previously suggested Chernoff stopping rule on several bandit problems. Moreover, we analyze the asymptotic optimality of the $\delta$-PAC algorithm with the decomposition of e-processes. The decomposition separates optimality conditions into three parts; the sampling policy allocation convergence, the efficiency of e-process design, and the bandit problem's inherent stochastic behavior. Subsequently, we prove a sufficient condition for asymptotic optimality in bandit algorithms. For the application of an e-process stopping rule, we derived an asymptotically optimal algorithm for the generalized linear bandit problem.
    
[^157]: 基于多智能体强化学习增强的联邦学习用于剩余使用寿命预测

    Multi Agent Reinforcement Learning enhanced Federated Learning for Predicting Remaining Useful Lifetime

    [https://arxiv.org/abs/2611.31880](https://arxiv.org/abs/2611.31880)

    该论文提出了一种基于QMIX值函数分解的协作式多智能体强化学习框架，通过自适应选择各客户端传输的模型层子集并限制重传次数，解决了有损无线信道下联邦学习在剩余使用寿命预测中因丢包导致模型交换脆弱的问题，同时兼顾了数据隐私保护。

    

    arXiv:2611.31880v1 公告类型：新 摘要：剩余使用寿命（RUL）预测是故障预测与健康管理（PHM）的核心。工业物联网的发展解锁了大规模多变量传感器数据流，并使深度模型——尤其是Transformer——能够进行长时序序列学习。然而，Transformer参数量大且需要大量数据，而实际运营数据集往往各自孤立存储且涉及隐私敏感信息。联邦学习（FL）通过在不共享原始数据的情况下跨客户端训练来解决这一问题，但在实际中，许多客户端通过有损无线链路连接，出于低延迟的考虑倾向于使用UDP，这使得模型交换容易受到丢包的影响。我们提出了一种协作式多智能体强化学习（MARL）框架，通过自适应选择每个客户端需传输的模型层子集并限制重传次数，使联邦学习能够适应有损信道。该框架使用QMIX值函数分解进行训练以保证可扩展性。我们采用图卷积……来实例化该方法。

    arXiv:2611.31880v1 Announce Type: new  Abstract: Remaining useful life (RUL) prediction is central to Prognostics and Health Management (PHM). The growth of industrial IoT has unlocked large multivariate sensor streams and enabled deep models - especially Transformers - for long-horizon sequence learning. However, transformers are parameter-heavy and data-hungry, while operational datasets are often siloed and privacy-sensitive. Federated Learning (FL) addresses this by training across clients without raw data sharing, but in practice many clients connect over lossy wireless links where UDP is preferred for low latency, making model exchange vulnerable to packet drops. We present a cooperative multi-agent reinforcement learning (MARL) framework that adapts FL to lossy channels by selecting per-client layer subsets to transmit and capping retransmissions. This framework is trained with QMIX value-function factorization for scalability. We instantiate the approach with a Graph Convolutio
    
[^158]: EEG-Jev：一种面向脑电（EEG）的统一单次前向决策模型

    EEG-Jev: A Unified Single-Pass Decision Model for EEG

    [https://arxiv.org/abs/2611.31875](https://arxiv.org/abs/2611.31875)

    EEG-Jev提出了一种统一的脑电单次前向决策模型，一次前向传播即可完成选项选择、命题判断与有序评分三类任务，并通过先验校正和温度缩放提供含义明确、可校准的概率输出，在多个公开脑电数据集上取得了优异表现。

    

    脑电（EEG）分析涵盖从备选项中选择、对陈述进行判断以及对有序属性进行评分等任务，需要具有明确含义且可校准的概率。我们提出EEG-Jev，一种统一的脑电单次前向决策模型。给定一个脑电窗口、一个问题及答案规范，仅需一次前向传播即可返回类型化的输出：CHOICE给出候选选项分布，NOUL给出命题概率，SCORE给出有序等级分布及期望值。该模型仅训练一个联合检查点，并在35个公开数据源的保留划分上一次性评估。核心原始输出在Stieger2021运动想象任务上达到54.08%的平衡准确率，在Sleep-EDF-SC睡眠分期上达到70.45%，在CHB-MIT癫痫检测上以阈值0.5达到每窗口72.11%的准确率（AUROC为0.825），在0–10分的ds005292疼痛评分上MAE为1.700。先验校正与温度缩放将Sleep-EDF-SC上的负对数似然从0.766降至0.606，在CHB-MIT上从0.166降至0.019，同时直接输出削减……

    arXiv:2611.31875v1 Announce Type: cross  Abstract: EEG analysis spans choosing alternatives, judging statements, and rating ordered attributes, requiring calibratable probabilities with explicit meaning. We present EEG-Jev, a unified single-pass decision model for EEG. Given an EEG window, a question and an answer specification, one forward pass returns a typed readout: CHOICE gives a candidate distribution, NOUL a proposition probability, and SCORE an ordered-level distribution and expectation. One checkpoint is trained jointly and evaluated once on held-out splits of 35 public sources. Core raw outputs reach 54.08% balanced accuracy on Stieger2021 motor imagery, 70.45% on Sleep-EDF-SC sleep staging, 72.11% per window at threshold 0.5 on CHB-MIT seizure detection with 0.825 AUROC, and 1.700 MAE on 0-10 ds005292 pain ratings. Prior correction and temperature scaling lower negative log-likelihood from 0.766 to 0.606 on Sleep-EDF-SC and 0.166 to 0.019 on CHB-MIT, while direct readout cut
    
[^159]: 潜意识学习跨越模态

    Subliminal Learning Crosses Modalities

    [https://arxiv.org/abs/2611.31870](https://arxiv.org/abs/2611.31870)

    本文发现基于文本的潜意识学习可以跨模态迁移——通过数字序列在视觉-语言模型间传递的主题偏好不仅保留在文本中，还会改变图像的生成和表示方式，其机制源于学生模型变化中的一个恒定分量。

    

    潜意识学习是指语言模型中一种令人惊讶的现象：给定某个模型的提示（例如，对猫的偏好）可以通过在表面上看似无意义的数据（例如，数字序列）上进行训练，潜移默化地蒸馏到另一个模型中。本文表明，基于文本的潜意识学习可以跨模态迁移，特别是，通过数字序列在视觉-语言模型之间传递的主题/偏好（例如，动物）也会改变图像的生成和表示方式。首先，我们展示了使用学生模型作为文本编码器的扩散模型，比使用原始模型作为文本编码器时更频繁地生成学生模型所对应的动物，但不会生成其颜色或风格。随后，我们研究了这一现象背后的机制以及哪些概念能够跨模态迁移。学生模型变化中的一个恒定分量解释了图像表示向该特征偏移的大部分原因。添加这一分量……

    arXiv:2611.31870v1 Announce Type: cross  Abstract: Subliminal learning refers to a surprising phenomenon in language models wherein a prompt given to one model (e.g., preference for cats) can be subliminally distilled to another model via training on data that is ostensibly meaningless (e.g., number sequences). This paper shows that text-based subliminal learning transfers cross-modally, in particular, a topic/preference (e.g., animals) transferred through number sequences across vision-language models also changes how images are generated and represented. First, we show that a diffusion model using the student as its text encoder generates the student's animal more often than with the original model as its text encoder, but not its color or style. We then investigate the mechanism behind this phenomenon and which concepts transfer across modalities. A constant component of the student's change accounts for much of the shift of image representations toward the trait. Adding this compon
    
[^160]: 从数据所在之处出发：面向流形上一步生成的统计先验

    Start Where the Data Are: Statistical Priors for One-Step Generation on Manifolds

    [https://arxiv.org/abs/2611.31856](https://arxiv.org/abs/2611.31856)

    该论文提出基于切空间高斯混合先验与后验耦合（并借助隐马尔可夫先验处理长扭转角链）的一步生成框架，使采样从数据密集区域出发，避免在流形低密度间隙产生虚假样本，从而提升蛋白质设计等科学应用中的生成质量与效率。

    

    流形上的快速生成对于蛋白质设计等科学应用十分重要，然而现有的一步生成器常常在数据模式之间的低密度间隙中产生虚假样本。我们的分析揭示了均匀噪声与集中分布的数据之间的不匹配，由此启发我们提出一种能够捕捉数据粗略结构的统计先验。我们首先通过几何感知的矩匹配方法拟合切空间高斯混合模型来估计该先验。然而，独立的噪声-数据配对仍可能连接不同的模式，从而削弱拟合先验带来的好处。为此，我们引入了后验耦合方法，即以每个数据点为条件采样一个潜在分量，并从该分量中抽取与之配对的噪声。为了将该方法推广到全协方差混合模型代价过高的长扭转角链场景，我们使用隐马尔可夫先验，通过共享参数来建模序列依赖关系。由此得到的框架……

    arXiv:2611.31856v1 Announce Type: new  Abstract: Fast generation on manifolds is important for scientific applications such as protein design, yet existing one-step generators often produce spurious samples in low-density gaps between data modes. Our analysis highlights the mismatch between uniform noise and concentrated data, motivating a statistical prior that captures the data's coarse structure. We first estimate this prior by fitting mixtures of tangent-space Gaussians through geometry-aware moment matching. However, independent noise-data pairing can still connect different modes and weaken the benefit of the fitted prior. We therefore introduce posterior coupling, which samples a latent component conditioned on each data point and draws its paired noise from that component. To extend this approach to long torsion-angle chains, where full-covariance mixtures become costly, we use a hidden Markov prior to model sequential dependence with shared parameters. The resulting framework 
    
[^161]: 面向基于流的具身基础模型的两阶段后训练

    Two-Stage Post-Training for Flow-Based Embodied Foundation Models

    [https://arxiv.org/abs/2611.31850](https://arxiv.org/abs/2611.31850)

    该论文提出了一种基于流的具身基础模型的两阶段后训练框架，第一阶段通过多教师在线策略蒸馏高效整合专家能力，第二阶段将强化学习与基于采样噪声加权的教师蒸馏相结合，从而突破性能瓶颈、实现超越教师模型的表现。

    

    大规模预训练为具身模型提供了通用能力，但可靠的任务执行仍然需要后训练。在多任务设置中，监督微调（SFT）面临训练与部署之间的分布不匹配问题，而强化学习（RL）则受限于稀疏奖励以及跨任务学习进度不均衡的问题。多教师在线策略蒸馏（MOPD）通过对模型自身轨迹的密集监督来整合专家能力，但要实现超越教师的性能仍然具有挑战性。我们提出了一种针对基于流的具身模型的两阶段后训练框架，从条件流匹配的速度回归形式中构建蒸馏目标。阶段1通过MOPD高效整合专家能力。阶段2将强化学习与教师蒸馏相结合，根据采样噪声对速度误差进行加权，其中蒸馏被限制在（原文在此截断）……

    arXiv:2611.31850v1 Announce Type: cross  Abstract: Large-scale pretraining provides embodied models with general capabilities, but reliable task execution still requires post training. In multitask settings, supervised fine-tuning (SFT) faces distribution mismatch between training and deployment, whereas reinforcement learning (RL) is limited by sparse rewards and uneven learning progress across tasks. Multi-teacher on policy distillation (MOPD) consolidates expert capabilities through dense supervision on the model's own trajectories, but achieving performance beyond the teachers remains challenging. We propose a two-stage post training framework for flow-based embodied models, constructing distillation objectives from the velocity regression formulation of conditional flow matching. Stage 1 efficiently consolidates expert capabilities through MOPD. Stage 2 combines RL with teacher distillation that weights velocity errors according to the sampling noise, where distillation is restric
    
[^162]: 规格-测试覆盖差距：检测SWE-Bench风格基准中的过度指定测试

    Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks

    [https://arxiv.org/abs/2611.31839](https://arxiv.org/abs/2611.31839)

    本文提出规格-测试覆盖差距（STCG）这一1-4分制新指标，利用两个LLM评判者自动检测SWE-bench风格基准中的过度指定测试，从而取代昂贵的专家审查，识别会导致正确的替代修复方案被误判失败的测试用例。

    

    SWE-bench风格的基准测试，例如SWE-bench Verified（Verified）和SWE-bench Pro（Pro），使用源自最初修复各问题的变更中的测试来评估编码智能体。然而，这些测试可能会强制执行任务描述中从未说明的细节，导致正确的替代修复方案失败。迄今为止，这类过度指定的测试主要通过昂贵的专家审查来发现。为了解决这一问题，我们定义了规格-测试覆盖差距（STCG），这是一个用于衡量过度指定程度的1-4分制量表，并由两个LLM评判者——Claude Opus 5.5和GPT-6.1 Sol——根据规格说明和测试补丁进行评分。当两个评判者都给出3分或更高分数时，该任务被标记为过度指定。我们的研究包括三个部分。首先，我们在206个任务上验证了评判者，这些任务由早期的人工审计和智能体结果标记为过度指定或正常。其次，我们将STCG应用于全部500个Verified任务和731个公开的Pro任务，并将标记结果与排名前30的Verified（原文在此处截断）

    arXiv:2611.31839v1 Announce Type: cross  Abstract: SWE-bench-style benchmarks, such as SWE-bench Verified (Verified) and SWE-bench Pro (Pro), evaluate coding agents with tests taken from the change that originally fixed each issue. However, these tests can enforce details the task never states, causing correct alternative fixes to fail. Such overspecified tests have so far been found mainly through expensive expert review. To address this, we define the Spec-Test Coverage Gap (STCG), a 1-4 scale measuring overspecification, and score it with two LLM judges, Claude Opus 5.5 and GPT-6.1 Sol, which are given the specification and the test patch. A task is flagged as overspecified when both judges score it 3 or higher. Our study has three parts. First, we validate the judges on 206 tasks that earlier human audits and agent results label as overspecified or fine. Second, we apply STCG to all 500 Verified and 731 public Pro tasks and compare the flags with the results of the top 30 Verified 
    
[^163]: 探索多人社交游戏作为大语言模型的可验证训练场

    Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models

    [https://arxiv.org/abs/2611.31821](https://arxiv.org/abs/2611.31821)

    本文提出将多人社交游戏作为可验证训练环境，通过SFT和RLVR使Qwen3-4B的胜率从33%大幅提升至81%，超越了远大于自身的Gemma-3-27B模型，并能泛化到未见过的游戏中，效果远优于提示等上下文策略。

    

    训练具有社交智能的语言模型通常依赖于大语言模型（LLM）评判者，但这种方式成本高昂、噪声大且容易受到奖励作弊的影响。多人社交游戏提供了一种天然的替代方案：它们需要开放式的沟通来与队友协作并推断隐藏角色，同时其规则提供了客观的胜负结果，可以作为训练中的可验证奖励。本文通过三个社交游戏，探索了Qwen3模型通过监督微调（SFT）和基于可验证奖励的强化学习（RLVR）能够习得多少社交智能。结果表明，面对强大得多的Gemma-3-27B模型，SFT和RLVR将Qwen3-4B在三个训练游戏上的胜率从33%提升至81%，在另外三个未训练的保留游戏上的胜率从39%提升至54%。相比之下，提示等上下文内策略在相同的训练和评估游戏上仅将Qwen3-4B的胜率提高约6个百分点。

    arXiv:2611.31821v1 Announce Type: cross  Abstract: The training of socially intelligent language models often relies on large language model (LLM) judges, which are expensive, noisy, and vulnerable to reward hacking. Multi-player social games offer a natural alternative: they require open-ended communication to coordinate with teammates and infer hidden roles, while their rules provide an objective outcome which can serve as a verifiable reward for training. Using three social games, we explore how much social intelligence Qwen3 models can acquire through supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Results show that against a much stronger Gemma-3-27B model, SFT and RLVR raise the win rate of Qwen3-4B from 33% to 81% on the three trained games, and from 39% to 54% on three additional held-out games. In contrast, in-context strategies such as prompting improve the win rate of Qwen3-4B by only about 6 pp across the same games we train and evalu
    
[^164]: 两个证明相距多远？在证明内核与 Lean/Mathlib 中度量证明差异性

    How Far Apart Are Two Proofs? Measuring Proof Dissimilarity in a Proof Kernel and in Lean/Mathlib

    [https://arxiv.org/abs/2611.31811](https://arxiv.org/abs/2611.31811)

    该论文针对“同一定理的两个证明相距多远”这一源于希尔伯特第 24 问题的基础问题，定义了 6 大族共 19 种证明差异性度量和 10 个期望性质，并在 Curry-Howard 内核与 Lean/Mathlib 中加以实例化与验证。

    

    我们如今生活在机器生成数学的时代：AI 系统产生研究级证明的速度已超过人类数学界能够阅读它们的速度。验证只是挑战的一部分，我们还需要知道一个“新证明”是否真的是新的，一千个生成的证明究竟包含一个想法还是多个想法，以及每个证明相对于人类数学所处的位置。所有这些都要求度量证明之间的距离。沿着希尔伯特第 24 问题与证明等同性问题的思路，我们追问：同一个定理的两个证明相距多远？它们所栖居的证明空间又具有怎样的形状？我们定义了 6 个族（句法、结构、依赖与资源、语义、信息论、高阶范畴）共 19 种差异性度量，以及 10 个期望性质，我们认为这些性质是安全优化某一度量的前提条件。在一个 Curry-Howard 内核中，5 个经机器验证的素数无穷性证明表明，原始句法……

    arXiv:2611.31811v1 Announce Type: cross  Abstract: We now live in the age of machine-generated mathematics: AI systems produce research-level proofs faster than the human mathematical community can read them. Verification is only part of the challenge. We also need to know whether a "new proof" is indeed new, whether a thousand generated proofs contain one idea or many, and where each sits relative to human mathematics. All of these require measuring distances between proofs. Following Hilbert's 24th problem and the identity-of-proofs problem, we ask how far apart two proofs of one theorem are, and what shape the proof space they inhabit has. We define 19 dissimilarity measures in 6 families (syntactic, structural, dependency & resource, semantic, information-theoretic, higher-categorical) and 10 desirable properties, which we argue are the conditions for safely optimising a measure. In a Curry-Howard kernel, 5 machine-checked proofs that the primes are infinite show that raw syntax ca
    
[^165]: 为大语言模型智能体学习紧凑的状态抽象

    Learning Compact State Abstractions for LLM Agents

    [https://arxiv.org/abs/2611.31800](https://arxiv.org/abs/2611.31800)

    本文提出将智能体状态定义为交互历史的紧凑抽象，以确定性递归程序表示并用LLM引导的遗传算法搜索，在ALFWorld和ScienceWorld上保持接近参考智能体的任务成功率的同时，将策略输入最多减少71%和57%，且得到的程序可直接审查。

    

    大语言模型（LLM）智能体通常将完整的交互历史作为状态的替代，无论过去的观察和动作是否仍然相关，都全部予以保留。我们转而探究智能体真正需要携带哪些信息前进。我们将智能体状态定义为交互历史的一种紧凑抽象，它能够保持某个固定参考智能体的能力，并以最小的此类抽象作为理想目标。由于匹配参考智能体的行为对黑盒策略而言过于局限且困难，我们通过任务表现而非行为匹配来定义充分性。我们将状态抽象表示为确定性递归程序，并使用由大语言模型引导的遗传算法对其进行搜索。在ALFWorld和ScienceWorld基准上，学习到的抽象保持了接近参考智能体的任务成功率，同时将策略的平均输入分别减少了高达71%和57%。所得到的程序是可直接检查审查的。

    arXiv:2611.31800v1 Announce Type: cross  Abstract: LLM agents commonly use the full interaction history as a proxy for state, retaining all past observations and actions regardless of whether they remain relevant. We instead ask what information an agent actually needs to carry forward. We define agent state as a compact abstraction of interaction history that preserves the competence of a fixed reference agent, with the minimal such abstraction as the ideal. Because matching the reference agent's behavior is unnecessarily restrictive and difficult for black-box policies, we define sufficiency through task performance rather than behavior matching. We represent state abstractions as deterministic recursive programs and search over them using an LLM-guided genetic algorithm. Across ALFWorld and ScienceWorld, the learned abstractions preserve near-reference task success while reducing mean policy input by up to 71% and 57%, respectively. The resulting programs are directly inspectable, p
    
[^166]: 字节补丁与BPE在匹配数据下的对比：350M参数规模的匹配孪生比较

    Byte Patches versus BPE at Matched Data: A Matched-Twin Comparison at 350M Parameters

    [https://arxiv.org/abs/2611.31790](https://arxiv.org/abs/2611.31790)

    在350M参数、100亿字节的严格匹配数据条件下，8k词表的BPE分词在每字节比特数及多数下游任务上优于字节补丁方法，同时研究揭示了外推上下文、混淆比特分母等可能颠覆结论的评估陷阱。

    

    我们在完全相同的原始字节、文档顺序以及每个优化器步骤近似匹配的字节数条件下，将MEGABYTE风格、配备64字节滑动局部解码器的字节补丁Transformer（local64）与匹配的字节级8k词表BPE解码器（BPE-8k）进行对比：两者均为350M参数和100亿训练字节。在匹配跨度的文档滑动设置下（评分字节范围完全相同），BPE-8k在每字节比特数上领先，相对差距为-5.2%、-4.9%、-9.3%和-6.1%（对应总体、文本、代码、数学；配对自助法检验）。在下游任务中，CharBench等微调更有利于local64（53.7%对40.0%），fact200更有利于BPE-8k（16.5%对7.0%），在alpha=0.05的Holm校正后，PIQA（原始和长度归一化）以及ARC-Easy长度归一化也更有利于BPE-8k；其余任务均为平局。研究中发现会改变结论的评估陷阱包括：超出BPE训练上下文长度的上下文外推、将每token比特数误报为每字节比特数，以及不一致的字节分母。

    arXiv:2611.31790v1 Announce Type: cross  Abstract: We compare a MEGABYTE-style byte-patch Transformer with a 64-byte sliding local decoder (local64) to a matched byte-level 8k BPE decoder (BPE-8k) under identical raw bytes, document order, and approximately matched bytes per optimizer step: 350M parameters and 10 billion training bytes. Under matched-span document sliding (identical scored byte ranges), BPE-8k leads with relative gaps of -5.2%, -4.9%, -9.3% and -6.1% in bits per byte (overall, text, code, math; paired bootstrap). Downstream, CharBench equal fine-tuning favors local64 (53.7% vs 40.0%), fact200 favors BPE-8k (16.5% vs 7.0%), and PIQA (raw and length-normalized) and ARC-Easy length-normalized favor BPE-8k after Holm correction at alpha = 0.05; remaining tasks are ties. Evaluation pitfalls that changed conclusions include context-length extrapolation past the BPE training context, bits per token reported as bits per byte, and inconsistent byte denominators. Unlike controll
    
[^167]: RoMo Hands：一个大规模、丰富组织的文本到手部动作数据集

    RoMo Hands: A Large Scale Richly Organized Text to Hand Motion Dataset

    [https://arxiv.org/abs/2611.31766](https://arxiv.org/abs/2611.31766)

    本文提出了RoMo Hands数据集——包含约54.2万序列（884小时）的真实场景文本-手部动作数据，规模比以往工作大一个数量级，并通过严格的灵巧性过滤、层次化多粒度文本标注以及新颖的393维双手表示方法，弥合了身体与手部动作生成架构之间的差距。

    

    尽管文本驱动的身体动作生成技术已经成熟，纯手部动作合成仍受制于数据瓶颈。现有数据集往往将手视为刚性末端执行器，缺乏细粒度的文本描述，或规模过于有限而难以支持稳健的泛化。我们提出了 RoMo Hands，一个真实场景下的文本配对数据集，包含约54.2万个序列（884小时，30fps下共9550万帧），规模比以往工作大一个数量级。为确保高质量监督信号，严格的灵巧性过滤器会剔除低关节活动度的序列，使数据分布聚焦于复杂的双手运动。每个片段配有五个不同粒度的文本描述，从全局语义标签到原子级手指状态，并按照层次化分类体系进行组织。为弥合身体与手部架构之间的差距，我们提出了一种新颖的393维双手表示方法，将冗余的身体动作范式适配到手部的高自由度运动学中。系统性基准评测……

    arXiv:2611.31766v1 Announce Type: cross  Abstract: While text-driven body motion generation has matured, pure-hand synthesis remains bottlenecked by data. Existing datasets often treat hands as rigid end-effectors, lack fine-grained captions, or are too limited in scale to support robust generalization. We introduce RoMo Hands, an in-the-wild, text-paired dataset of ~542K sequences (884 hours, 95.5M frames at 30 fps) which is an order of magnitude larger than prior work. To ensure high-quality supervision, a stringent dexterity filter discards low-articulation sequences, focusing the distribution on complex bimanual movement. Each clip is paired with five captions at increasing granularities from global semantic tags to atomic finger states, organized under a hierarchical taxonomy. To bridge the gap between body and hand architectures, we propose a novel 393-D bimanual representation that adapts redundant body-motion paradigms to the high-DoF kinematics of hands. Systematic benchmarkin
    
[^168]: Geo-SMILE：面向地理空间机器学习的统计模型无关可解释性与局部解释方法

    Geo-SMILE: Statistical Model-Agnostic Interpretability with Local Explanations for Geospatial Machine Learning

    [https://arxiv.org/abs/2611.31731](https://arxiv.org/abs/2611.31731)

    提出Geo-SMILE框架，通过互补的特征屏蔽与地理重定位两个扰动分支以及距离加权的替代模型，将实质性特征解释与地理空间位置响应区分开来，为地理空间机器学习提供模型无关的局部可解释性。

    

    传统的后置（post-hoc）解释方法主要是为非空间输入开发的，通常对地理背景缺乏明确的处理。我们提出Geo-SMILE，一个模型无关的框架，通过互补的扰动分支将实质性特征解释与地理位置响应区分开来。特征分支在保留观测坐标的同时屏蔽实质性预测变量；地理分支则在固定实质性预测变量的同时重新定位地理单元。遵循SMILE的统计距离加权替代模型原则，输入分布的Wasserstein距离决定联盟的局部性权重，加权Ridge替代模型用于近似期望的黑箱预测。GeoSensitivity以保留方向为符号，量化在指定设计下拟合模型对地理单元重新定位的响应。在赫尔市的犯罪预测、西雅图的房价预测等任务上（摘要原文在此处截断）……

    arXiv:2611.31731v1 Announce Type: cross  Abstract: Conventional post-hoc explanation methods were developed largely for non-spatial inputs and often provide limited explicit treatment of geographic context. We introduce Geo-SMILE, a model-agnostic framework that distinguishes substantive Feature explanation from geographic-position response through complementary perturbation branches. The Feature branch masks substantive predictors while retaining observed coordinates; the Geo branch relocates geographic units while holding substantive predictors fixed. Following the statistical-distance-weighted surrogate principle of SMILE, input-distribution Wasserstein distances determine coalition locality weights and weighted Ridge surrogates approximate expected black-box predictions. GeoSensitivity, signed in the keep orientation, quantifies fitted-model responses to geographic-player relocation under the specified design. Across crime prediction in Hull, housing-price prediction in Seattle and
    
[^169]: 时间序列Transformer中标记化与架构共同分配组件角色

    Tokenization and Architecture Jointly Allocate Component Roles in Time Series Transformers

    [https://arxiv.org/abs/2611.31730](https://arxiv.org/abs/2611.31730)

    该论文提出基于15个token级诊断任务的时间探测框架，揭示时间序列Transformer中构建时间特征的组件并非固定，而是由标记化与架构共同分配——在patch标记化（PatchTST）下FFN承担特征构建角色，其增益平均达注意力机制的3.2倍。

    

    时间序列Transformer已取得强大的预测性能，但其组件中究竟是注意力机制还是前馈网络（FFN）负责构建时间特征，以及patching等标记化方式如何改变这一分工，仍不清楚。我们提出了一个时间探测框架，使用15个token级诊断任务，通过测量时间属性从一个子层到下一个子层的线性可解码性增益来衡量特征构建（匹配的MLP探测在嵌入层已饱和）。该框架针对的是被探测表示由线性预测头读取的模型。将该框架应用于在8个长期预测基准上训练的5个监督模型后发现，构建特征的组件并非固定不变，而是由标记化与架构的交互作用所分配。在patch标记化（PatchTST）下，FFN的增益在全部8个数据集的探测中均超过注意力机制，其8个数据集的平均增益是注意力机制的3.2倍。

    arXiv:2611.31730v1 Announce Type: new  Abstract: Time series transformers have achieved strong forecasting performance, yet it remains unclear which of their components, attention or the feed-forward network (FFN), constructs temporal features, and how tokenization such as patching changes this. We propose a temporal probing framework that uses 15 token-level diagnostic tasks to measure feature construction as the gain in linear decodability of temporal properties from one sub-layer to the next (a matched MLP probe already saturates at the embedding). It targets models whose probed representations are read by a linear forecasting head. Applied to 5 supervised models trained on 8 long-term forecasting benchmarks, it finds that the constructing component is not fixed but allocated by the interaction of tokenization and architecture. Under patch tokenization (PatchTST), the FFN's gain exceeds attention's in probing on all 8 datasets, and its 8-dataset mean is 3.2x attention's. Without att
    
[^170]: EventSSM：面向异步事件深度估计的因果空间-状态建模

    EventSSM: Causal Spatial-State Modelling for Asynchronous Events Event-Depth Estimation

    [https://arxiv.org/abs/2611.31720](https://arxiv.org/abs/2611.31720)

    提出EventSSM——一种用循环记忆按时间顺序处理事件数据包的因果空间-状态模型，实现异步事件关联的度量深度估计，并通过相机位姿几何传输与可微渲染桥接异步预测与稀疏帧时刻监督，在多个深度误差指标上超越已有方法。

    

    事件相机以异步方式记录亮度变化，而度量深度监督通常仅在稀疏采样的参考时间戳处可用。将事件聚合为类帧表示虽然简化了学习过程，但可能掩盖原始事件流的细粒度时间结构。我们提出了EventSSM，这是一个用于与事件相关联的度量深度估计的因果空间-状态框架，它采用由按时间顺序到达的事件数据包更新的循环记忆。为了连接异步预测与帧时刻监督，预测的事件深度利用标定的相机位姿进行几何传输，并在参考时间戳处进行可微渲染。基准评估表明，该方法在多个深度误差指标上均优于已有的事件深度方法。受控消融实验进一步考察了跨模态初始化、时间循环、个体状态、读出与监督组件以及事件数据包子采样等因素的作用。

    arXiv:2611.31720v1 Announce Type: cross  Abstract: Event cameras report asynchronous brightness changes, yet metric-depth supervision is typically available only at sparsely sampled reference timestamps. Aggregating events into frame-like representations simplifies learning but can obscure the fine-grained temporal structure of the original event stream. We introduce EventSSM, a causal spatial-state framework for event-associated metric-depth estimation using a recurrent memory updated by chronological event packets. To bridge asynchronous predictions and frame-time supervision, predicted event depths are geometrically transported using calibrated camera poses and differentiably rendered at reference timestamps. Benchmark evaluations demonstrate improvements over established event-depth methods across multiple depth-error measures. Controlled ablations further examine cross-modal initialization, temporal recurrence, individual state, readout, and supervision components, and packet subd
    
[^171]: 结构可靠性分析中的停止准则：综述

    Stopping criteria in structural reliability analysis: An overview

    [https://arxiv.org/abs/2611.31706](https://arxiv.org/abs/2611.31706)

    本文系统综述了结构可靠性分析中各类停止准则，指出基于误差估计、方差和稳定性检测的经典方法虽源自其他领域，但在结构可靠性分析中具有广泛的适用性和应用潜力。

    

    停止准则已成为结构可靠性分析中概率与统计方法的关键组成部分，有助于防止结构缺陷与失效，以及对现有结构的不准确评估。随着时间推移，这些准则在概念和应用上都不断发展演变，但文献中对它们的处理仍然零散。这些准则往往与特定模型或分析框架相结合进行开发和呈现，从而掩盖了其在不同模型和方法之间更广泛的适用性。基于此，本文的目标有两个方面。首先，文章借鉴应用概率与统计中的概念，对广泛的停止规则进行了综述。这些策略包括基于误差估计或方差以及稳定性检测的经典方法。虽然其中许多思想源自其他领域，但它们有潜力为结构可靠性任务提供有用的工具。

    arXiv:2611.31706v1 Announce Type: cross  Abstract: Stopping criteria have become a crucial component of probabilistic and statistical methods in structural reliability analysis, helping prevent structural defects and failures, as well as inaccurate evaluations of existing structures. Over time, these rules have evolved in both concept and application, whereas their treatment in the literature remains fragmented. They are often developed and presented in connection with specific models or analytical frameworks, obscuring their broader applicability across diverse models and methodologies. On this basis, the aim of this article is twofold. First, it overviews a wide range of stopping rules, drawing on concepts from applied probability and statistics. These strategies include classical methods based on error estimates or variance, and stability detection. While many of these ideas were developed in other fields, they have the potential to offer useful tools for structural reliability task
    
[^172]: CATWAY：基于校准语言模型置信度的代价敏感智能体三支机器人决策

    CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence

    [https://arxiv.org/abs/2611.31694](https://arxiv.org/abs/2611.31694)

    CATWAY 提出了一种代价敏感的三支决策门控框架，利用四种指令条件下的损失矩阵和校准的语言模型置信度概率，让机器人在执行、追问与拒绝之间做出决策，并在跨领域测试中无需重新拟合即保持有效。

    

    语言模型机器人智能体必须决定是执行、追问还是拒绝一条指令，而不安全执行的代价远高于一次多余的提问。此前没有任何机器人决策门控从错误代价推导其全部阈值、或基于校准后的概率行动，也没有任何方法在一个领域拟合后在另一个领域原样评估。我们提出 CATWAY，一种代价敏感的三支决策门控：通过对四种指令条件（清晰、模糊、不可行、不安全）设定的损失矩阵，在概率单纯形上划定执行、追问与拒绝三个区域，其中追问即是信息收集步骤。其概率通过逻辑回归将十次采样投票与模糊度、不可行性和风险的分级评分融合。CATWAY 在家务指令上拟合后即被冻结，随后在域内、两个新的机器人领域以及外部 TRUST-NAV 基准上，对三个小模型的文本级决策进行了测试。在每次决策十一次生成的情况下……（摘要在此处截断）

    arXiv:2611.31694v1 Announce Type: cross  Abstract: A language-model robot agent must decide whether to execute, question or refuse an instruction, and an unsafe execution costs far more than an unneeded question. No prior robot gate derives all its thresholds from error costs or acts on calibrated probabilities, and none was fitted in one domain and evaluated unchanged in another. We propose CATWAY, a cost-sensitive three-way decision gate: a loss matrix over four instruction conditions (clear, ambiguous, infeasible, unsafe) fixes execute, ask and refuse regions on the probability simplex, the question being the information-gathering step. Its probabilities fuse a ten-sample vote with graded scores of ambiguity, infeasibility and risk by logistic regression. Fitted on household instructions and frozen, CATWAY was tested on text-level decisions of three small models in-domain, in two new robot domains and on the external TRUST-NAV benchmark. At eleven generations per decision against on
    
[^173]: 无导数约束随机优化的在线统计推断

    Online Statistical Inference for Derivative-Free Constrained Stochastic Optimization

    [https://arxiv.org/abs/2611.31687](https://arxiv.org/abs/2611.31687)

    本文提出一种基于随机投影的免导数随机序列二次规划方法（SDF-SSQP），在无导数信息的等式约束随机优化问题中，将每轮迭代成本降至 $O((d+m)^2)$ 的同时实现有效的在线统计推断。

    

    我们研究在导数信息不可用或获取代价高昂的情形下，非线性等式约束随机优化问题的统计推断。尽管随机序列二次规划（SSQP）的最新进展已实现了高效的在线优化，但现有方法要么依赖梯度信息，要么计算成本过高。为解决这些局限，我们开发了一种基于随机投影的免导数SSQP（SDF-SSQP）方法，可同时实现计算效率和有效的统计推断。通过利用随机投影技术，所提方法将每次迭代的计算成本降低至 $O((d+m)^2)$，其中 $d$ 为决策变量的维数，$m$ 为约束的数量，使其非常适合大规模问题。为克服随机投影和零阶近似给统计推断带来的挑战，我们采用……（原文摘要在此处截断）

    arXiv:2611.31687v1 Announce Type: cross  Abstract: We study statistical inference for nonlinear equality-constrained stochastic optimization problems in settings where derivative information is unavailable or expensive to obtain. While recent advances in stochastic sequential quadratic programming (SSQP) have enabled efficient online optimization, existing methods either rely on gradient information or suffer from high computational costs. To address these limitations, we develop a sketched derivative-free SSQP (SDF-SSQP) method that simultaneously achieves computational efficiency and valid statistical inference. By leveraging randomized sketching, the proposed method reduces the per-iteration computational cost to $O((d+m)^2)$, where $d$ is the dimension of the decision variable and $m$ is the number of constraints, making it well suited for large-scale problems. To overcome the challenges that both sketching and zeroth-order approximations introduce for statistical inference, we ado
    
[^174]: 面向深度强化学习的曲率均衡策略优化

    Curvature-Equilibrated Policy Optimization for Deep Reinforcement Learning

    [https://arxiv.org/abs/2611.31680](https://arxiv.org/abs/2611.31680)

    提出EQUIP方法，通过均衡策略损失的正负曲率来平衡参数空间的局部收缩与扩张，从而改进深度强化学习中基于梯度的策略优化。

    

    基于梯度的参与者优化是深度强化学习的核心。然而，传统的一阶优化器并未显式利用神经参与者损失的局部曲率。我们提出了EQUIP（用于策略搜索的曲率均衡梯度），这是一种新颖的方法，旨在通过均衡正曲率与负曲率来平衡参与者参数空间的局部收缩与扩张。EQUIP构建了一个基于轨迹信息的搜索子空间，并利用参与者损失的投影Hessian矩阵来构建有界的局部度量，以实现曲率均衡。一阶损失匹配保留了参与者损失中近似的一阶下降量。同时，梯度混合和基于曲率的梯度选择机制调整了用于参与者优化的梯度。我们建立了曲率均衡的几何性质，推导了梯度对齐性与幅度的界限，并给出了改善参与者损失下降的条件。

    arXiv:2611.31680v1 Announce Type: new  Abstract: Gradient-based actor optimization is central to deep reinforcement learning. However, conventional first-order optimizers do not explicitly exploit the local curvature of neural actor losses. We propose EQUIP (Curvature-Equilibrated Gradients for Policy Search), a novel approach that seeks to balance local contraction and expansion of the actor parameter space by equilibrating positive and negative curvature. EQUIP constructs a trajectory-informed search subspace and uses the projected Hessian of actor loss to build a bounded local metric for curvature equilibration. First-order loss matching preserves the approximated first-order decrease in actor loss. Meanwhile, gradient mixing and curvature-based gradient selection adjust the gradient applied for actor optimization. We establish the geometric properties of curvature equilibration, derive bounds on gradient alignment and magnitude, and identify conditions for improved actor loss reduc
    
[^175]: 神经网络互操作协议：在训练前指定兼容性

    Neural Interoperability Protocols: Specifying Compatibility Before Training

    [https://arxiv.org/abs/2611.31670](https://arxiv.org/abs/2611.31670)

    该论文提出神经网络互操作协议（NIP），将模型间的兼容性从“事后由模型对性能推断”转变为“训练前以公开接口规范明确规定”，独立于具体架构和训练过程来约束表示交换的边界、语义与下游支持的计算。

    

    神经网络越来越多地被作为可复用组件来开发，但它们的中间表示在独立训练的模型之间仍然难以交换。现有方法可以对齐表示或鼓励兼容特征，但特定模型对之间成功的迁移本身并不能定义未来的实现必须满足哪些条件才能实现互操作。我们提出了神经网络互操作协议，这是一个“规范优先”的框架，它将神经兼容性视为一组公开的接口义务，而不是仅从模型对性能中推断出来的属性。NIP 规定了交换边界、交换数据包的语义与可接受性条件，以及接收方必须支持的下游计算，且这一切独立于实现它的具体架构和训练过程。我们通过受控的检索、视觉以及连续记忆实验来研究这一区别……

    arXiv:2611.31670v1 Announce Type: new  Abstract: Neural networks are increasingly developed as reusable components, yet their intermediate representations remain difficult to exchange across independently trained models. Existing approaches can align representations or encourage compatible features, but successful transfer between particular models does not by itself define what future implementations must satisfy to interoperate. We introduce Neural Interoperability Protocols (NIPs), a specification-first framework that treats neural compatibility as a set of public interface obligations rather than a property inferred solely from model-pair performance. A NIP specifies an exchange boundary, the semantics and admissibility conditions of exchanged packets, and the downstream computations that recipients must support, independently of the architectures and training procedures implementing it. We investigate this distinction through controlled retrieval, vision, and continuous-memory exp
    
[^176]: 用于隐式神经表示的分段线性激活函数

    A Piecewise-Linear Activation Function for Implicit Neural Representations

    [https://arxiv.org/abs/2611.31666](https://arxiv.org/abs/2611.31666)

    本文提出JaSIN——一族介于ReLU与分段线性正弦之间的有界分段线性激活函数，通过引入有界范围、居中凸起和有限振荡，将ReLU在图像重建任务上的PSNR从24.0 dB提升至37.3 dB，在十一种激活函数中位居第三，接近最先进水平。

    

    使用ReLU激活函数的坐标网络在隐式神经表示（INR）任务上表现不佳，而解决这一问题的激活函数（如正弦函数和Gabor小波）都脱离了分段线性函数类，因此目前仍不清楚ReLU的哪个性质是问题所在。我们提出了JaSIN，这是一族有界的分段线性激活函数，它在ReLU与分段线性正弦函数之间进行插值，且无需使用任何超越运算。我们的结果表明：有界范围平均可将图像重建PSNR提高0.06 dB，单个凸起可提高3.29 dB，将该凸起居中可再提高3.64 dB。引入振荡可再多提高6.30 dB，将ReLU在Kodak图像上的表现从24.0 dB提升至37.3 dB，在十一种激活函数中排名第三，与最先进方法的差距在1.3 dB以内，但当振荡次数超过四次时重建效果会崩溃。由于INR是在评估它的网格上进行拟合的，一个预激活值……（原文在此处截断）

    arXiv:2611.31666v1 Announce Type: cross  Abstract: Coordinate networks with ReLU activations perform poorly on implicit neural representation tasks, and the activations that address this, such as sinusoids and Gabor wavelets, all leave the piecewise-linear function class, so it remains unclear which property of ReLU is at fault. We introduce JaSIN, a family of bounded piecewise-linear activations that interpolates between ReLU and a piecewise-linear sinusoid without any the use of transcendental operation. Our results show that a bounded range improves image reconstruction PSNR on average by 0.06 dB, a single bump 3.29 dB, and centering that bump a further 3.64 dB. The oscillation count improves 6.30 dB more, raising ReLU from 24.0 to 37.3 dB on Kodak images, placing third over eleven activations, within 1.3 dB of the state-of-the-art, though beyond four oscillations the reconstruction collapses. Because an INR is fit on the grid on which it is evaluated, a unit whose pre-activations n
    
[^177]: 基于边界表示驱动的图Transformer建模用于加工工序排序

    Boundary Representation-Driven Graph Transformer Modeling for Machining Operation Sequencing

    [https://arxiv.org/abs/2611.31661](https://arxiv.org/abs/2611.31661)

    该论文提出了一种基于图Transformer的框架，直接从B-Rep设计几何预测加工工序序列，充分利用边界表示模型中丰富的几何与拓扑信息以实现工艺规划的自动化。

    

    加工工艺规划本质上十分复杂，这是由于零件几何形状、加工工序以及不断演变的可制造状态之间存在相互依赖关系。传统方法严重依赖专家知识和基于规则的系统，限制了可扩展性和自动化程度。虽然近期的机器学习方法已展现出潜力，但大多数方法基于简化的几何表示（如体素或手工设计的特征），无法充分利用边界表示（B-Rep）模型中编码的丰富几何与拓扑信息。在本工作中，我们提出了一种基于图Transformer的“几何到序列”学习框架，可直接从B-Rep设计几何预测加工工序序列。所提出的方法从B-Rep实体构建拓扑感知图，其中面及其邻接关系被编码为节点和边特征，从而同时捕捉几何信息和关系信息。图神经网络

    arXiv:2611.31661v1 Announce Type: new  Abstract: Machining process planning is inherently complex due to the interdependence between part geometry, machining operations, and evolving manufacturable states. Traditional approaches rely heavily on expert knowledge and rule-based systems, limiting scalability and automation. While recent machine learning methods have shown promise, most operate on simplified geometric representations, such as voxels or hand-crafted features, which fail to fully exploit the rich geometric and topological information encoded in Boundary Representation (B-Rep) models. In this work, we propose a graph transformer-based framework for geometry-to-sequence learning that directly predicts machining operation sequences from B-Rep design geometry. The proposed method constructs topology-aware graphs from B-Rep entities, where faces and their adjacencies are encoded as node and edge features capturing both geometric and relational information. A graph neural network 
    
[^178]: 将VAE微调锚定于其潜空间扩散先验

    Anchoring VAE Fine-Tuning to Its Latent Diffusion Prior

    [https://arxiv.org/abs/2611.31651](https://arxiv.org/abs/2611.31651)

    提出扩散先验锚定（DPA）正则化方法，在VAE微调时通过分数蒸馏将潜变量引导至预训练扩散模型能准确去噪的高密度流形，在多个模型和任务上将预测误差降低4.0%至7.5%。

    

    预训练的潜空间扩散模型是迁移学习的有效先验，应用范围涵盖从几何估计到物理模拟等多个领域。现有方法通常微调扩散主干网络，但冻结变分自编码器（VAE）的编码器以保持潜变量分布。然而，冻结的编码器可能遗漏非RGB输入中的重要细节，且仅限于三个通道。我们提出了扩散先验锚定，这是一种在VAE微调过程中应用的正则化方法，通过分数蒸馏更新将潜变量引导至预训练扩散模型能够准确去噪的高密度流形上。我们在三个图像和视频模型（FLUX 3、LTX-Video、Wan2.2）以及四项任务（单目深度估计、MRI重建、风流模拟和表面法线估计）上验证了DPA的有效性。实验表明，DPA相对于冻结VAE将预测误差降低了7.5%，相对于仅解码器微调降低了4.9%，相对于未锚定的VAE微调降低了4.0%。

    arXiv:2611.31651v1 Announce Type: cross  Abstract: Pretrained latent diffusion models are effective priors for transfer learning, from geometry estimation to physics simulation. Existing approaches fine-tune the diffusion backbone but freeze the encoder of the VAE to preserve the latent distribution. However, a frozen encoder may miss salient details in non-RGB inputs and is limited to three channels. We introduce Diffusion Prior Anchoring (DPA), a regularizer applied during VAE fine-tuning that uses a score-distillation update to steer latents toward the high-density manifold on which the pretrained diffusion model denoises accurately. We validate DPA across three image and video models (FLUX 3, LTX-Video, Wan2.2) and four tasks: monocular depth estimation, MRI reconstruction, wind-flow simulation, and surface normal estimation, where DPA lowers the prediction error by 7.5% relative to the frozen VAE, 4.9% to decoder-only fine-tuning, and 4.0% to unanchored VAE fine-tuning. Its gain g
    
[^179]: 用于解释偏好异质性的奖励基分解

    Reward Basis Decomposition for Interpreting Preference Heterogeneity

    [https://arxiv.org/abs/2611.31640](https://arxiv.org/abs/2611.31640)

    提出奖励基分解（RBD）方法，将奖励模型中每个用户的权重分解为群体共享项与个体项，从而分别解析群体共享价值观与个体偏好差异，并发现真实数据中偏好主要由单一群体方向解释。

    

    标准奖励模型规定了AI策略的优化目标，因此理解它们所编码的价值观对于确保AI对齐至关重要。现有的可解释性技术使用单一奖励头来解释奖励模型，无法区分群体共享的价值观与个体间存在差异的价值观。我们提出了奖励基分解，将每个用户的权重分解为一个群体项和一个用户个体项。由此得到一个共享的群体方向和若干个体用户方向，可以通过概念对齐和稀疏自编码器分解分别进行分析。我们在合成对照实验上验证了RBD能够恢复人工植入的用户个体方向，随后将其应用于两个真实世界的多偏好数据集——PRISM和Community Alignment。在这两个数据集中，我们均未发现个体用户结构，这表明奖励嵌入空间中的单一方向就能解释大部分偏好……

    arXiv:2611.31640v1 Announce Type: new  Abstract: Canonical reward models specify the optimization target for AI policies, so understanding the values they encode is important for ensuring AI alignment. Existing interpretability techniques interpret reward models with a single reward head, which cannot distinguish between the values that a population shares and those that vary among individuals. We introduce Reward Basis Decomposition (RBD), which splits per-user weights into a population term and a per-user term. This yields a shared population direction and individual user directions that are analyzed separately via concept alignment and sparse autoencoder decomposition. We verify that RBD can recover artificially planted per-user directions on a synthetic control, then apply it to two real-world multi-preference datasets, PRISM and Community Alignment. In both, we find no individual-user structure, suggesting that a single direction in the reward embedding space accounts for most pre
    
[^180]: 所见即所证：面向未见组合的基于概念的证书

    What You See Is What You Certify: Concept-Based Certificates for Unseen Compositions

    [https://arxiv.org/abs/2611.31627](https://arxiv.org/abs/2611.31627)

    该研究揭示了基于概念的模型在未见属性组合上会以高达93-94%的概率错误认证类别，并将本体论视为错误检测码，据此确立了有效决策证书的三个条件：排除所有竞争类别的可见证据、如实标注未见部件、以及不受共现捷径驱动的罕见误读。

    

    基于概念的模型通过它们声称“看到”的概念来为决策提供解释，并通过显式的本体论进行解码。然而，众所周知，概念预测会泄露标签信息、忽略局部性，并且在新的属性组合上缺乏可靠基础。由这类概念解码出的决策即使满足本体论仍可能是错误的，因此我们追问：它在什么情况下才能作为一份证书。在熟悉部件的未见组合上，按标准方式训练的联合概念瓶颈模型和概念嵌入模型，即使本体论正确，也有93-94%的时间认证了错误的类别（对于仅在可见部件上训练并由定位器门控的联合模型，这一比例为60%）。联合模型的部件错误比随机水平更频繁地复制共现的属性值（跨概念捷径现象）。通过将本体论视为一种错误检测码，我们证明了有效的证书需要满足：具有排除所有竞争类别的可见证据、将未见部件如实标注为未见、以及误读罕见且不受共享模式驱动的特性。

    arXiv:2611.31627v1 Announce Type: cross  Abstract: Concept-based models justify a decision by concepts they claim to see, decoded through an explicit ontology. Yet concept predictions are known to leak label information, ignore locality, and be poorly grounded on new attribute combinations. A decision decoded from such concepts can satisfy the ontology and still be wrong, so we ask when it can serve as a certificate. On unseen combinations of familiar parts, a joint concept bottleneck model and a concept embedding model, trained as is standard, certify wrong classes 93-94% of the time despite a correct ontology (60% for a joint model trained only on visible parts and gated by a localizer). The joint model's part errors copy co-occurring values more often than chance (cross-concept shortcutting). Treating the ontology as an error-detecting code, we show that a certificate needs visible evidence excluding every rival, unseen parts reported as such, and rare misreads not driven by a share
    
[^181]: 预测排行榜能够确立什么：零样本时间序列模型的分辨率报告

    What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models

    [https://arxiv.org/abs/2611.31619](https://arxiv.org/abs/2611.31619)

    该论文提出“排行榜分辨率”这一概念并开发了Vernier工具来量化它，揭示出GIFT-Eval等主流零样本时间序列预测基准的分辨率远大于相邻模型间的实际分差、因而无法可靠区分模型优劣，而BOOM凭借2,554个独立序列成为唯一具有足够分辨率的基准。

    

    时间序列基础模型只需预训练一次，之后便被要求对其从未见过的序列进行预测，因此评判该模型的依据就是它在留出数据集上的得分。这类数据集十分稀缺，少数公开排行榜承载了大部分模型比较，然而没有任何一个排行榜说明其数据集能够检测到的最小差异是多少。我们将这一量称为排行榜的分辨率，并推出了Vernier——一个利用排行榜已发布文件即可计算该分辨率的工具。Vernier还能报告排行榜无法将其与领先者区分开的条目，以及当无关第三方提交结果时排行榜的排序是否会发生改变。GIFT-Eval可分辨6.8个排名位次，而其显示的相邻条目之间的分差为0.91，其前十名的十个位次所包含的可分辨步数不足两步。fev-bench、Chronos Benchmark II和TIME也呈现出同样的模式：它们的分辨率是相邻条目间差距的3.6至19倍。凭借其2,554个序列各自作为独立数据源，BOOM是唯一一个能够……（原文摘要在此处截断）

    arXiv:2611.31619v1 Announce Type: new  Abstract: A foundation model for time series is pretrained once and then asked to forecast series it has never seen, so the evidence for it is its score on held-out datasets. Such datasets are scarce, and a few public leaderboards carry most comparisons, yet none of them states how small a difference its datasets can detect. We call that quantity the resolution of a board and introduce Vernier, a tool that computes it from the files a board already publishes. Vernier also reports which entries a board cannot separate from its leader and whether its order changes when unrelated parties submit. GIFT-Eval resolves 6.8 rank places, and its displayed neighbours sit 0.91 apart. The ten places of its top ten hold fewer than two resolvable steps. fev-bench, Chronos Benchmark II and TIME show the same pattern: their resolution is 3.6 to 19 times the gap between neighbours. With each of its 2,554 series as an independent source, BOOM is the one board that r
    
[^182]: 通过对注意力头功能网络的结构干预来引导出被藏拙的大语言模型能力

    Eliciting sandbagged LLM capability via structural interventions on attention-head functional networks

    [https://arxiv.org/abs/2611.31615](https://arxiv.org/abs/2611.31615)

    该研究通过基于注意力头激活构建功能网络图，发现了跨模型架构一致的藏拙结构性特征，并证明可通过对该功能网络施加结构干预来引导出模型被隐藏的能力。

    

    可靠地评估先进AI系统的能力对于其安全开发与部署至关重要，然而这类评估可能被“藏拙”行为所破坏——即模型策略性地抑制其自身实际拥有的能力。现有工作已构建了受控的藏拙场景以及检测或引导隐藏能力的方法，但对于模型在能力被抑制时其内部神经计算如何变化，相关研究仍相对匮乏。在本工作中，我们从内部功能组织的视角来研究藏拙行为。受网络神经科学启发，我们基于注意力头激活构建功能图，并在保持模型架构与权重不变的前提下，比较诚实条件与藏拙条件下功能图的组织方式。令人惊讶的是，跨越不同模型架构与社区检测方法，我们发现了一个一致的结构性特征：藏拙与……（原文摘要在此处截断）

    arXiv:2611.31615v1 Announce Type: new  Abstract: Reliably evaluating the capabilities of advanced AI systems is critical for safe development and deployment, yet such evaluations may be undermined by sandbagging, a situation in which a model strategically suppresses capabilities it possesses. Existing work has developed controlled sandbagging settings and methods for detecting or eliciting hidden capabilities, but comparatively little is known about how a model's internal neural computation changes when its capabilities are suppressed. In this work, we study sandbagging through the lens of internal functional organization. Inspired by network neuroscience, we construct functional graphs from attention-head activations and compare their organization under honest and sandbagging conditions while holding the model architecture and weights fixed. Surprisingly, across model architectures and community detection methods, we find a consistent structural signature in which sandbagging is assoc
    
[^183]: 训练决策模型以模拟人类受访者

    Training decision models for simulating human respondents

    [https://arxiv.org/abs/2611.31613](https://arxiv.org/abs/2611.31613)

    在历史受访者选择数据上对KEV决策模型进行选择训练，可将LLM合成受访者在跨领域留存研究上的处理效应误差降低22%至89%，且小模型可胜过更大的现成模型。

    

    研究人员越来越多地将大型语言模型（LLM）用作市场研究、联合分析和随机实验中的合成受访者。然而，现成的生成式LLM产生的合成选择数据会给下游的计量经济学推断带来偏差，无论是专有模型还是更高的推理投入都无法可靠地减少这种偏差。我们研究了用于硅采样（silicon sampling）的决策模型（System One架构）。通过选择训练，我们在推理之前、不使用目标研究数据的情况下，利用条件极大似然法在历史受访者层面的选择微观数据上更新KEV模型权重。在三个模型规模上，基于两个离散选择实验的训练使四个来自新领域（疫苗、营养标签、消费者权利和人工智能）且采用新回应格式的留存研究的处理效应误差降低了22%至89%。一个经过选择训练的小模型在四个研究中的三个上胜过了规模大得多的现成模型。尽管（原文在此处截断）

    arXiv:2611.31613v1 Announce Type: new  Abstract: Researchers increasingly use large language models (LLMs) as synthetic respondents in market research, conjoint analysis, and randomized experiments. However, synthetic choice data from off-the-shelf generative LLMs bias downstream econometric inference. Neither proprietary models nor higher reasoning effort reliably reduces this bias. We investigate decision models (System One architectures) for silicon sampling. Using choice training, we update KEV model weights on historical respondent-level choice microdata by conditional maximum likelihood, before inference and without target-study data. Across three model scales, training on two discrete choice experiments lowers treatment-effect error by 22% to 89% on four held-out studies from new domains (vaccines, nutrition labels, consumer rights, and AI) and new response formats. A small choice-trained model outperforms a much larger off-the-shelf model on three of the four studies. Although 
    
[^184]: 一维实例依赖的赌博机凸优化

    Instance-Dependent Bandit Convex Optimization in One Dimension

    [https://arxiv.org/abs/2611.31608](https://arxiv.org/abs/2611.31608)

    本文在一维赌博机凸优化中首次刻画了实例依赖样本复杂度，引入新量 Δ_ε(f) 证明定位极小值点问题可获得依赖该量的加速，同时惊人地发现简单遗憾设定下不存在任何实例依赖的改进。

    

    我们研究了一维随机赌博机凸优化在固定置信度设定下的实例依赖样本复杂度。首先，我们考虑极小值点定位问题，即学习者试图识别一个与唯一极小值点距离不超过 ε 的点。我们引入了一个新的实例依赖量 Δ_ε(f)，它对应于 f−min f 的下水平集长度为 2ε 的阈值。我们证明了量级为 Δ_ε(f)^{-2} 的下界，并给出一种二分类式的算法，该算法在对数因子范围内匹配这一速率。随后，我们重新审视简单遗憾设定，其目标是识别一个 ε-最优点。令人惊讶的是，在此设定下不可能获得可比的实例依赖改进：每个正确的算法在每个实例上的样本复杂度均为 ε^{-2} 量级，在常数因子内匹配极小极大速率。这些结果共同区分了这两种设定。

    arXiv:2611.31608v1 Announce Type: cross  Abstract: We study the instance-dependent sample complexity of one-dimensional stochastic bandit convex optimization in the fixed-confidence setting. First, we consider minimizer localization, where the learner seeks to identify a point within distance $\varepsilon$ of the unique minimizer. We introduce a new instance-dependent quantity $\Delta_\varepsilon(f)$, corresponding to the threshold for which the sublevel set of $f-\min f$ has length $2\varepsilon$. We prove a lower bound of order $\Delta_\varepsilon(f)^{-2}$ and give a bisection-style algorithm that matches this rate up to logarithmic factors. We then revisit the simple-regret setting, where the goal is to identify an $\varepsilon$-optimal point. Surprisingly, no comparable instance-dependent improvement is possible: every correct algorithm has sample complexity of order $\varepsilon^{-2}$ on every instance, matching the minimax rate up to constants. Together, these results separate th
    
[^185]: 扩散训练中的噪声级别邻接性

    Noise-Level Adjacency in Diffusion Training

    [https://arxiv.org/abs/2611.31605](https://arxiv.org/abs/2611.31605)

    本文揭示了扩散模型跨噪声级别共享权重的收益来源于噪声级别嵌入中的邻接性而非方向性，并从理论上给出共享网络拟合误差的下界，即由相邻后验均值间的距离减去网络在相邻嵌入间可调整的幅度所决定。

    

    扩散模型在所有噪声级别上训练同一个网络，尽管其训练目标在每个噪声级别上是逐点独立的。如果在每对相邻级别上的训练是相互独立的问题，那么具有相同总参数量的多个独立网络理应能与一个共享网络相媲美。但在实践中，在图像任务上，跨级别共享权重能带来更好的结果，这暗示了其中存在对级别顺序的隐式依赖。顺序同时决定了哪些级别相邻以及它们的运行方向。本文证明，在两者之中，共享的收益依赖于噪声级别嵌入中的邻接性。共享网络的拟合误差存在一个下界：该下界由相邻后验均值之间的距离，减去网络在相邻嵌入之间所能做出的改变而确定。我们首先区分了邻接性与方向性：在高斯混合分布和CIFAR-10上，反转嵌入的顺序保持了所有邻居关系不变，而……

    arXiv:2611.31605v1 Announce Type: new  Abstract: Diffusion models train one network across all noise levels, although their training objective is pointwise in the noise level. If training on each pair of adjacent levels were an independent problem, separate networks with the same total parameter count would be expected to match one shared network. But in practice, on images, sharing weights across levels gives better results, which hints at an implicit dependence on their order. An order fixes both which levels are adjacent and the direction in which they run. In this paper, we show that, of the two, the benefit of sharing rests on adjacency in the noise-level embedding. The fitting error of a shared network is bounded below by the distances between neighbouring posterior means, minus the change the network can make between neighbouring embeddings. We first distinguish adjacency from direction: on a Gaussian mixture and CIFAR-10, reversing the embeddings keeps every neighbour and leave
    
[^186]: RePT：三维几何从二维关系预训练中涌现

    RePT: 3D Geometry Emerges from 2D Relational Pretraining

    [https://arxiv.org/abs/2611.31601](https://arxiv.org/abs/2611.31601)

    RePT通过跨视角二维点预测学习物理点之间的关系，使三维几何结构无需三维预测目标即可从二维关系预训练中自然涌现，并展现出跨数据集的泛化能力。

    

    arXiv:2611.31601v1 公告类型： cross 摘要：学习空间表示通常依赖于显式预测几何量，但物理几何如何在学到的表示中被组织起来仍知之甚少。我们探究三维结构能否从跨二维观测的关系学习中涌现，而无需三维预测目标。我们提出RePT，通过跨视角二维点预测来学习不同物理点之间的紧凑关系。学习到的关系空间形成了几何组织，其中幅值、角度和邻域关系能够反映物理三维位移。在16个场景上拟合的单一线性读出层，无需重新拟合即可在39个保留场景上恢复位移长度和角度（Pearson r 分别为0.72和0.78）。在ARKitScenes上训练的模型检查点在ScanNet++上也保持了强几何组织，展现了泛化能力。在受控比较中，不同点关系预测持续产生更强的几何组织。

    arXiv:2611.31601v1 Announce Type: cross  Abstract: Learning spatial representations often relies on explicitly predicting geometric quantities, but how physical geometry is organized within learned representations remains less understood. We ask whether 3D structure can emerge from learning relations across 2D observations, without 3D prediction targets. We introduce RePT, which learns compact relations between distinct physical points through cross-view 2D point prediction. The learned relation space develops a geometric organization in which magnitudes, angles, and neighborhoods reflect physical 3D displacement. A single linear readout fitted on 16 scenes recovers displacement length and angle on 39 held-out scenes without refitting (Pearson r = 0.72 and 0.78). The ARKitScenes-trained checkpoint also retains strong geometric organization on ScanNet++, showcasing generalizability. In controlled comparisons, distinct-point relational prediction consistently yields stronger geometric or
    
[^187]: LUCID：基于统计学原理的大语言模型推理评估自我纠错方法

    LUCID: Statistically Grounded Self-Correction for LLM Reasoning Evaluation

    [https://arxiv.org/abs/2611.31595](https://arxiv.org/abs/2611.31595)

    LUCID无需重新训练或访问模型权重，通过将提示自我纠正与基于设计的监督学习相结合来校准语言模型评分器对推理步骤的评估，将GPT-5-mini的校准差距从7.8个百分点降至0.9个百分点，误报减半，错误检测F1提升32%。

    

    廉价的语言模型评分器可以对思维链推理进行大规模评分，但系统性的步骤评分错误可能会扭曲推理审计和解决方案排名。我们提出了LUCID（通过推理时诊断实现无偏校准的循环），它将提示自我纠正与基于设计的监督学习（DSL）相结合。在无需重新训练或访问模型权重的情况下，LUCID通过将预测的合理率与专家合理率对齐来校准二元步骤评分。基于一个小型专家标注数据集，LUCID以纯文本形式总结评分器的偏差，并提示评分器对未见过的步骤重新评分。例如，在PRM800K数据集上，GPT-5-mini评分过于严格，对计算、验证和推理链后期步骤的低估最为严重。经过一次校正，LUCID将GPT-5-mini的校准差距（以预测率与专家率的绝对差值衡量）从7.8个百分点缩小至0.9个百分点。LUCID还将误报（将合理步骤误评为不合理）减少了一半，并将不合理步骤的错误检测F1分数提升了32%。

    arXiv:2611.31595v1 Announce Type: new  Abstract: Cheap language-model raters grade chain-of-thought reasoning at scale, yet systematic step-rating errors can skew reasoning audits and solution rankings. We introduce LUCID (Loop for Unbiased Calibration via Inference-time Diagnosis), which pairs prompt self-correction with Design-based Supervised Learning (DSL). Without retraining or weight access, LUCID calibrates binary step ratings by aligning predicted and expert soundness rates. From a small expert-labeled set, LUCID summarizes rater bias in plain text and prompts the rater to re-grade unseen steps. For example, on PRM800K, GPT-5-mini grades too strictly, under-rating computation, verification, and late-trace steps most acutely. In one pass, LUCID shrinks GPT-5-mini's calibration gap from 7.8 to 0.9 percentage points in absolute predicted-expert rate difference. LUCID also halves false alarms (sound steps rated unsound) and lifts error-detection F1 on unsound steps by 32%. LUCID re
    
[^188]: 一个阶段足矣：基于直接视觉强化学习的仿真到现实迁移

    One Phase Is Enough: Sim-to-Real Transfer with Direct Visual Reinforcement Learning

    [https://arxiv.org/abs/2611.31588](https://arxiv.org/abs/2611.31588)

    提出DRIFT方法，通过使用冻结的预训练基础视觉编码器实现高效的单阶段视觉强化学习，直接从图像观测完成仿真到现实的策略迁移，无需传统的“教师-学生”蒸馏流程。

    

    仿真到现实强化学习已经实现了令人瞩目的机器人行为，但直接基于视觉部署策略仍具挑战性。一种常见方法是先用特权状态观测通过强化学习训练教师策略，再通过模仿学习将其行为蒸馏到基于视觉的学生策略中。然而，所得的学生策略表现可能大幅逊于教师策略。直接用强化学习训练视觉策略是一种有前景的替代方案，但从零开始学习任务相关的视觉表征往往样本效率低下，使得单阶段视觉强化学习难以实现。我们提出DRIFT，一种利用测试时可获取的图像观测来实现高效单阶段视觉强化学习的简单方案。我们的核心洞察是使用冻结的预训练基础视觉编码器，使强化学习能够专注于基于现有视觉表征学习控制。冻结编码器避免了通过视觉骨干网络进行代价高昂的反向传播，而其预训练所获得的表征……（摘要在此处截断）

    arXiv:2611.31588v1 Announce Type: new  Abstract: Sim-to-real reinforcement learning (RL) has enabled impressive robotic behaviors, yet deploying policies directly from vision remains challenging. A common approach first trains a teacher policy with RL using privileged state observations, then distills its behavior into a vision-based student via imitation. However, the resulting student can substantially underperform its teacher. Directly training visual policies with RL offers a promising alternative, but learning task-relevant visual representations from scratch can be sample-inefficient, making one-phase visual RL intractable. We propose DRIFT, a simple recipe for efficient one-phase visual RL using image observations available at test time. Our key insight is to use frozen, pretrained foundation vision encoders, allowing RL to focus on learning control from existing visual representations. Freezing the encoder avoids costly backpropagation through the vision backbone, while its pre
    
[^189]: LatentLens：重新思考多智能体系统中的潜在接口

    LatentLens: Rethinking Latent Interface in Multi-Agent Systems

    [https://arxiv.org/abs/2611.31582](https://arxiv.org/abs/2611.31582)

    本文提出LatentLens框架，从外部和内部两个层面解耦多智能体系统中的潜在接口，发现在相同智能体框架下引入潜在接口并不能提升甚至可能显著降低性能，其效果取决于重新对齐兼容性、信息可访问性和传播动态等内部因素。

    

    潜在接口使多智能体系统中的智能体能够通过连续表示而非解码后的文本进行通信。先前的研究报告了性能和效率上的提升，但既没有将潜在通信与智能体框架（harness）分离开来，也没有考察其内部组件，导致其实际贡献尚不明确。我们提出了LatentLens，一个在两个层面解耦潜在接口的框架：在外部将其与智能体框架解耦，以衡量其对整体性能的贡献（RQ1）；在内部将其解耦为重新对齐兼容性、信息可访问性和传播动态，以识别各因素的优势与局限（RQ2）。在基于Qwen3模型的七个基准测试中，在相同的智能体框架上添加潜在接口并不能提升性能，甚至可能显著降低准确率。在内部，潜在表示保留了一定的领域信息，但不兼容的重新对齐可能大幅降低性能。

    arXiv:2611.31582v1 Announce Type: cross  Abstract: Latent interfaces let agents in a multi-agent system communicate through continuous representations rather than decoded text. Prior work reports performance and efficiency gains, but neither separates latent communication from the agent harness nor examines its internal components, leaving its actual contribution unclear. We propose LatentLens, a framework that decouples the latent interface at two levels: externally from the agent harness, to measure its contribution to overall performance (RQ1), and internally into realignment compatibility, information accessibility, and propagation dynamics, to identify the benefits and limitations of each (RQ2). Across seven benchmarks with Qwen3 models, adding the latent interface to the same harness does not improve and can even substantially degrade accuracy. Internally, latent representations retain some domain information, but incompatible realignment can substantially degrade performance, an
    
[^190]: 潜在通道补全：面向含失效检测器传感器网络的预测

    Latent Channel Completion for Forecasting in Sensor Networks with Failed Detectors

    [https://arxiv.org/abs/2611.31575](https://arxiv.org/abs/2611.31575)

    提出潜在通道补全方法，通过在训练中重建失效传感器的潜在表示而非其数值，使传感器网络预测模型在部分探测器失效时仍能保持对健康传感器预测的准确性。

    

    用于交通和公用事业基础设施的监测网络经常出现传感器丢失，而通过将传感器相互耦合来提升精度的预测模型也因此继承了这种失效带来的脆弱性。当一个探测器停止报告时，替代值会被当作真实读数嵌入其中，并通过耦合网络的共享表征传播，从而损害仍然健康的传感器的预测质量。由于这些模型是在所有传感器均正常报告的基准上被挑选的，这种脆弱性并未体现在用于选择模型的指标数字中。我们提出潜在通道补全，这是一种沿通道轴的联合嵌入预测目标，它重建缺失传感器的潜在表示而非其数值，因为耦合算子消费的是嵌入而从不接触原始读数。在训练期间对通道进行掩蔽，并根据存活的通道对照停止梯度目标来预测被掩蔽通道的表示；在推理时……（原文摘要在此处截断）

    arXiv:2611.31575v1 Announce Type: new  Abstract: Monitoring networks for traffic and utility infrastructure lose sensors routinely, and forecasting models that couple sensors together to gain accuracy also inherit an exposure to those losses. When a detector stops reporting, the substituted value is embedded as though it were a reading and propagates through the shared representation that couples the network, degrading forecasts for sensors that are still healthy. Because these models are selected on benchmarks where every sensor reports, the exposure is absent from the numbers used to choose them. We introduce Latent Channel Completion, a channel-axis joint embedding predictive objective that reconstructs missing sensors' latent representations rather than their values, since a coupling operator consumes embeddings and never sees raw readings. Channels are masked during training and their representations predicted from surviving channels against stop-gradient targets; at inference the
    
[^191]: 基于渐进后验采样的去噪扩散

    Denoising Diffusion with Progressive Posterior Sampling

    [https://arxiv.org/abs/2611.31558](https://arxiv.org/abs/2611.31558)

    该论文提出带流形投影的渐进后验采样（PPS）方法，通过闭式高斯后验近似自动估计噪声方差、避免SVD，并将样本投影至局部含噪图像流形，从而更有效地利用扩散先验的局部几何结构求解含噪线性逆问题。

    

    扩散模型为求解含噪线性逆问题提供了有效的生成先验。然而，现有的后验采样方法可能需要对测量算子进行奇异值分解（SVD），或者需要预先知晓测量噪声的方差，而基于局部梯度的引导仅能提供对后验几何结构的有限视角。我们提出了一种带流形投影的渐进后验采样（PPS）方法，该方法能够自动估计测量噪声方差并避免SVD。在反向扩散过程中，PPS以闭式形式构建后验分布的高斯近似，并将从局部先验中抽取的样本引导至近似后验均值。随后，我们将引导后的样本投影到一个近似局部含噪图像流形的球面水平集上，从而促进样本与扩散先验局部几何结构的一致性。在FFHQ和ImageNet数据集上跨越四种图像修复任务的实验……（原文摘要在此处截断）

    arXiv:2611.31558v1 Announce Type: cross  Abstract: Diffusion models provide effective generative priors for solving noisy linear inverse problems. However, existing posterior sampling methods may require singular value decomposition (SVD) of the measurement operator or prior knowledge of the measurement noise variance, while guidance based on local gradients provides only a limited view of posterior geometry. We propose progressive posterior sampling (PPS) with manifold projection, which automatically estimates the measurement noise variance and avoids SVD. During reverse diffusion, PPS constructs a Gaussian approximation to the posterior distribution in closed form and guides a sample drawn from the local prior toward the approximate posterior mean. We then project the guided sample onto a spherical level set that approximates the local noisy image manifold, promoting consistency with the local geometry of the diffusion prior. Experiments on FFHQ and ImageNet across four restoration t
    
[^192]: ExitLUT：面向高效FPGA神经网络推理的布尔早期退出机制

    ExitLUT: Boolean Early Exits for Efficient FPGA-Based Neural Network Inference

    [https://arxiv.org/abs/2611.31540](https://arxiv.org/abs/2611.31540)

    ExitLUT用直接映射到FPGA LUT的可学习布尔函数替代softmax置信度退出决策，将节省的退出路径资源重新投入主数据通路，在保持精度不变的前提下实现高达4.08倍的吞吐量提升。

    

    早期退出神经网络通过在达到足够置信度时终止推理来降低平均推理成本。在FPGA上，传统的基于置信度的退出机制会引入与LUT结构不匹配的算术控制路径，消耗了本可用于主数据通路的资源。我们提出ExitLUT，用直接映射到FPGA LUT中的可学习布尔函数来替代softmax风格的退出决策。在CNN评估中，fpgaConvNet是静态空间数据流基线，而ATHEENA是其基于softmax的早期退出扩展。ExitLUT将ATHEENA的退出决策LUT使用量相比softmax路径降低2.58倍，并消除了决策路径的DSP使用，同时保持精度和退出行为不变。重新运行设计空间探索可将这些节省的资源再投入到数据通路中，吞吐量相比fpgaConvNet提升高达4.08倍，相比ATHEENA提升1.82倍。对于基于LUT的神经网络，嵌入式ExitLUT退出实现了1.41–1.89倍的性能提升。

    arXiv:2611.31540v1 Announce Type: cross  Abstract: Early exit neural networks reduce average inference cost by terminating once sufficient confidence is reached. On FPGAs, conventional confidence-based exits add arithmetic control paths poorly matched to LUT fabrics, consuming resources that could otherwise support the main datapath. We present ExitLUT, which replaces softmax-style exit decisions with learned Boolean functions mapped directly into FPGA LUTs. In our CNN evaluation, fpgaConvNet is the static spatial-dataflow baseline, while ATHEENA is its softmax-based early exit extension. ExitLUT reduces ATHEENA's exit-decision LUT usage by 2.58$\times$ over the softmax path and eliminates decision-path DSP usage, while preserving accuracy and exit behavior. Rerunning design-space exploration reinvests these savings into the datapath, improving throughput by up to 4.08$\times$ over fpgaConvNet and 1.82$\times$ over ATHEENA. For LUT-based NNs, embedded ExitLUT exits achieve 1.41-1.89$\t
    
[^193]: JevBench：面向类型化概率决策模型的蜕变一致性测试

    JevBench: Metamorphic Coherence Testing for Typed Probabilistic Decision Models

    [https://arxiv.org/abs/2611.31533](https://arxiv.org/abs/2611.31533)

    JevBench 是一个无需金标准标签的蜕变一致性测试基准，通过覆盖表示、批次独立性、概率测度、逻辑顺序与选项集条件化五个维度的 50 条蜕变关系，系统检验类型化概率决策模型在问题改写、否定、组合等变换下的概率输出是否符合概率论与选择定律，并对 21 个开源模型进行了评测。

    

    类型化概率决策模型会返回定义在指定答案上的概率分布，例如“是/否”、一组选项或有序量表上的等级。我们探究当针对同一输入的问题被改写、否定、组合、给出不同选项或置于不同批次时，模型的概率是否依然保持相干（coherent）。JevBench 无需金标准标签即可测试这种一致性。它的 50 条蜕变关系各自检验一条概率或选择定律，覆盖五个维度：表示、批次独立性、概率测度、逻辑顺序与选项集条件化。每个测试在保持输入不变的前提下改变问题或批次，且所有模型接收相同的、事先固定的请求。JevBench-240 在 12 个决策领域的 240 个合成案例上包含 12,000 项测试；其固定的 1,200 项测试子集 JevBench-mini 能够高度复现完整测试集的模型排名（Spearman 相关 0.983）。在 21 个公开发布的模型中，一致性得分范围……（原文摘要在此处截断）

    arXiv:2611.31533v1 Announce Type: new  Abstract: Typed probabilistic decision models return distributions over specified answers, such as yes or no, a set of options or levels on an ordered scale. We ask whether their probabilities remain coherent when questions about the same input are reworded, negated, combined, given different options or placed in different batches. JevBench tests this coherence without gold labels. Its 50 metamorphic relations each test one law of probability or choice across five dimensions: representation, batch independence, probability measure, logical order and choice-set conditioning. Each test changes the question or batch while keeping the input fixed. Every model receives the same requests, fixed in advance. JevBench-240 contains 12,000 tests across 240 synthetic cases in 12 decision domains. A fixed subset of 1,200 tests, JevBench-mini, closely reproduces the full-suite ranking (Spearman 0.983). Across 21 openly released models, coherence scores range fr
    
[^194]: ECLIPSE：面向无数据边缘智能的以嵌入为中心的基础模型联邦知识迁移框架

    ECLIPSE: Embedding-Centered Foundation-Model Federated Knowledge Transfer for Data-Free Edge Intelligence

    [https://arxiv.org/abs/2611.31527](https://arxiv.org/abs/2611.31527)

    该论文提出ECLIPSE框架，以嵌入为中心，利用服务器托管的基础模型建立共享嵌入参考空间，在不共享原始数据、本地模型或公共数据集的前提下实现无数据场景下的联邦知识迁移与边缘智能。

    

    基础模型凭借其强大的表征和泛化能力，被认为将成为智能联邦边缘网络不可或缺的组成部分。然而，将其集成到资源受限且对隐私敏感的边缘环境中仍然面临挑战。这一挑战在无数据场景下尤为突出，因为在此类场景中不存在共享的、公开的或服务器可访问的参考数据集，无法建立用于对齐分布式知识的公共参考空间。为解决这一挑战，我们提出了ECLIPSE，一个专为无数据边缘智能设计的、以嵌入为中心的基于基础模型的联邦知识迁移框架。ECLIPSE利用服务器托管的基础模型实现边缘智能，而无需共享原始数据、本地模型或公共参考数据集。它建立了一个共享的嵌入参考空间，边缘设备通过该空间对其本地任务知识进行编码，仅由此产生的嵌入结果……

    arXiv:2611.31527v1 Announce Type: new  Abstract: Foundation Models (FMs) are envisioned to become an integral component of intelligent federated edge networks because of their strong representation and generalization capabilities. However, their integration into resource-constrained and privacy-sensitive edge environments remains challenging. This challenge is particularly pronounced in data-free settings, where no shared, public, or server-accessible reference dataset is available to establish a common reference space for aligning distributed knowledge. To address this challenge, we propose ECLIPSE, which is an embedding-centered FM-based federated knowledge transfer framework developed for data-free edge intelligence. ECLIPSE enables edge intelligence capitalizing on a server-hosted FM without sharing raw data, local models, or a public reference dataset. It establishes a shared embedding reference space through which edge devices encode their local task knowledge. Only the resulting
    
[^195]: 诊断神经算子中的认知不确定性：初始化、离散化与函数采样的影响

    Diagnosing Epistemic Uncertainty in Neural Operators: Effects of Initialization, Discretization, and Function Sampling

    [https://arxiv.org/abs/2611.31526](https://arxiv.org/abs/2611.31526)

    该论文系统研究了神经算子中认知不确定性的三大来源（训练函数采样、输出离散化、模型初始化），通过训练约46,000个模型并构建集成，定量刻画了各来源对算子学习性能差异的独立贡献。

    

    神经算子为函数空间之间的映射提供了高效的代理模型，但其可靠性无法通过单一训练模型的性能来刻画。现有的神经算子不确定性量化方法通常只刻画预测不确定性，而未能分别识别认知不确定性各来源的独立贡献。即使底层物理问题是确定性的，训练函数空间的有限采样、输出离散化和模型初始化也会在所学到的算子之间产生显著差异。我们在多样化的算子学习架构集合上系统地研究了这些认知不确定性来源。在实验中，我们训练了约46,000个神经算子模型，并在训练数据集、模型初始化和输出离散化上构建集成，以刻画相对L2误差的分布特征。

    arXiv:2611.31526v1 Announce Type: new  Abstract: Neural operators provide efficient surrogates for mappings between function spaces, but their reliability cannot be characterized by the performance of a single trained model. Existing neural operator uncertainty quantification approaches often characterize predictive uncertainty without separately identifying the contributions of individual sources of epistemic uncertainty. Finite sampling of the training function space, output discretization, and model initialization can produce substantial variation among learned operators even when the underlying physical problem is deterministic. We systematically investigate these sources of epistemic uncertainty across a diverse set of operator learning architectures. Across our experiments, we train approximately 46,000 neural operator models and construct ensembles over training datasets, model initializations, and output discretizations to characterize distributions of relative $L_2$ error rath
    
[^196]: SOTS：基于签名的最优传输选择

    SOTS: Signature-based Optimal Transport Selection

    [https://arxiv.org/abs/2611.31525](https://arxiv.org/abs/2611.31525)

    SOTS提出了一种无监督核心集选择框架，通过将不规则多变量时间序列映射为截断路径签名并贪心最小化子集与全量数据之间的切片Wasserstein距离来选择训练数据，同时提供了随机核心集的收敛性理论保证。

    

    在工业应用中，由于专家标注成本高且耗时，机器学习模型的训练数据选择成为一个关键瓶颈。现有的时间序列核心集选择方法要么完全忽略时间结构，要么无法处理变长、不规则采样、多通道信号——而这正是工业监测与测试中最常见的场景。我们提出了SOTS（Signature-based Optimal Transport Selection，基于签名的最优传输选择），一个面向不规则多变量时间序列（IMTS）分类的无监督核心集选择框架。每个样本被映射到一个截断路径签名，以捕捉其各通道的联合几何结构；随后通过贪心地最小化候选子集与完整数据集的经验签名分布之间的切片Wasserstein距离来选择核心集，我们为该准则建立了随机核心集的收敛性保证。我们在一个re……

    arXiv:2611.31525v1 Announce Type: new  Abstract: Training data selection for machine learning models is a critical bottleneck in industrial applications where expert annotation is costly and time-consuming. Existing coreset selection methods for time series either ignore temporal structure entirely or cannot handle variable-length, irregularly sampled, multi-channel signals, the setting most common in industrial monitoring and testing. We introduce SOTS (Signature-based Optimal Transport Selection), an unsupervised coreset selection framework for irregular multivariate time series (IMTS) classification. Each sample is mapped to a truncated path signature, capturing the joint geometric structure of its channels, and the coreset is selected by greedily minimizing the Sliced Wasserstein distance between the empirical signature distributions of the candidate subset and the full dataset, a criterion for which we establish a convergence guarantee for random coresets. We validate SOTS on a re
    
[^197]: 如何构建和训练你的循环深度模型：不动点视角

    How to build and train your recurrent-depth model: A fixed-point perspective

    [https://arxiv.org/abs/2611.31513](https://arxiv.org/abs/2611.31513)

    该论文从不动点理论视角揭示了循环深度模型中输入注入本质上就是Halpern迭代，证明了截断反向传播仅需多对数深度的窗口即可保证收敛，并据此开发了无额外开销的Sparrow优化器，可无缝替换Muon/MuonH并加速训练。

    

    循环深度模型使用截断反向传播进行训练，但其为何有效仍不明确。我们证明了循环深度模型中常用的输入注入恰好等价于Halpern迭代，并且缓慢消失的步长在反向传播窗口仅为深度的多对数级别时可证明收敛。这些理论保证要求算子满足Lipschitz条件，对于SwiGLU这意味着权重需要同时满足行范数和谱范数约束，而对于门控机制则只需行范数约束。基于这些约束，我们为每个参数组量身定制了一个优化器Sparrow，它可作为Muon/MuonH的无缝替代品且无额外开销，甚至在循环模型之外也能加速训练。

    arXiv:2611.31513v1 Announce Type: new  Abstract: Recurrent-depth models are trained with truncated backpropagation, yet it remains unclear why this works. We show that input injection, commonly used in recurrent-depth models, is exactly Halpern iteration, and that a slowly vanishing stepsize provably converges with a backpropagation window polylogarithmic in the depth. The guarantees require a Lipschitz operator, which for SwiGLU leads to a weight constraint in both row and spectral norm, while only a row norm constraint for gates. Based on these constraints, we develop an optimizer tailored to each parameter group, Sparrow, which provides a drop-in replacement of Muon/MuonH with no overhead, that speeds up training even beyond recurrent models.
    
[^198]: 近乎零遗忘并不足够：预训练视觉-语言-动作模型持续学习中的分布外鲁棒性

    Near-Zero Forgetting Is Not Enough: Out-of-Distribution Robustness in Continual Learning of Pretrained VLAs

    [https://arxiv.org/abs/2611.31511](https://arxiv.org/abs/2611.31511)

    该论文首次揭示了带经验回放的持续学习虽能使预训练VLA模型实现近乎零遗忘，但模型对分布外数据的鲁棒性会随持续学习持续退化，且这种退化遵循集中于多模态与动作特征的跨领域共享表征漂移模式。

    

    近期研究表明，预训练的视觉-语言-动作模型在带有经验回放的特定持续学习设置中表现出近乎为零的遗忘。然而，所学任务对领域偏移的鲁棒性仍未被探索。在本工作中，我们使用LIBERO和LIBERO-Plus基准，研究了通过经验回放微调的Pi0模型在分布外（OOD）数据下的鲁棒性。虽然域内（ID）设置实现了近乎为零的灾难性遗忘，但先前任务在OOD条件下的性能迅速崩溃。具体而言，我们展示了在持续学习过程中，模型的OOD能力会退化，并遵循一种跨领域偏移的共享表征模式，这种模式在两个更简单的测试套件中也能在行为层面观察到。表征分析表明，漂移集中在多模态特征和动作特征中。线性探针显示，OOD特征仍然保留与任务相关的信息，但表现出未……（原文摘要在此处截断）

    arXiv:2611.31511v1 Announce Type: new  Abstract: Recent work shows that pretrained Vision-Language-Action (VLA) models exhibit near-zero forgetting in specific continual learning settings with experience replay. However, robustness to domain shift of learned tasks remains unexplored. In this work, we study the robustness of Pi0 fine-tuned with experience replay under out-of-distribution (OOD) data using the LIBERO and LIBERO-Plus benchmarks. While the in-domain (ID) setup achieves near-zero catastrophic forgetting, OOD performance on previous tasks quickly collapses. Specifically, we show that during the continual learning process, the OOD capabilities of the model degrade, following a shared representational pattern across domain shifts, which is also visible behaviorally in the two simpler suites. Representational analysis shows that drift concentrates in the multimodal and action features. Linear probes indicate that OOD features still retain task-relevant information, but appear un
    
[^199]: 在可回放的野火扑救基准中测量反事实信用

    Measuring Counterfactual Credit in a Replayable Wildfire-Suppression Benchmark

    [https://arxiv.org/abs/2611.31499](https://arxiv.org/abs/2611.31499)

    提出可回放的野火扑救基准FLARE，通过在同一随机事件下以不同动作反事实回放火灾的任意时刻，直接测量多智能体动作的信用分配，并以“模拟器信用”指导训练和评价评论家，相比传统MAPPO显著降低了燃烧分数。

    

    在协作式多智能体强化学习中，学习到的评论家通常根据其预测团队回报的能力来评估。然而，预测团队回报与为个体动作分配信用是两个不同的任务。为了直接测试信用分配，我们提出了FLARE（Fireline Learning Agents in a Replayable Environment，可回放环境中的火线学习智能体），这是一个随机野火扑救基准，其中火灾的任何时刻都可以在相同随机事件下以不同动作进行回放。回放一步可以测量每个智能体扑救动作的即时效果，我们称之为“模拟器信用”，并将其用于训练。回放到回合结束则可以测量每个动作的完整效果，我们用它来评价评论家的表现。在八个随机种子上与多智能体近端策略优化（MAPPO）的比较中，模拟器信用将四分位均值燃烧分数（即每个格网的燃烧时间，经封顶并在地图上取平均，数值越低越好）从使用s…（摘要原文在此处截断）

    arXiv:2611.31499v1 Announce Type: cross  Abstract: In cooperative multi-agent reinforcement learning, learned critics are usually evaluated by how well they predict the team's return. Yet predicting a team's return and assigning credit to individual actions are distinct tasks. To test credit directly, we introduce FLARE (Fireline Learning Agents in a Replayable Environment), a stochastic wildfire suppression benchmark in which any moment of a fire can be replayed with different actions under the same random events. Replaying one step measures the immediate effect of each agent's suppression, which we call simulator credit and use for training. Replaying to the end of the episode measures the effect of each action, which we use for grading critics. In a comparison over eight seeds with multi-agent proximal policy optimization (MAPPO), simulator credit reduces the interquartile mean burn score (each cell's burning time, capped and averaged over the map; lower is better) from 26.6% with s
    
[^200]: ModernELECTRA：面向分类与嵌入的紧凑编码器的替换词元检测与边界窗口重建

    ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed

    [https://arxiv.org/abs/2611.31495](https://arxiv.org/abs/2611.31495)

    ModernELECTRA提出了一种预训练方案，在保留替换词元检测（RTD）主目标的同时，引入轻量级的边界窗口重建辅助任务，使紧凑编码器既能擅长分类任务，又能胜任检索与语义相似度等嵌入任务。

    

    arXiv:2611.31495v1 公告类型：新论文 摘要：ELECTRA式的替换词元检测（RTD）是一种用于预训练文本编码器的高样本效率训练目标，采用RTD训练的模型（如DeBERTaV3）在分类基准测试中表现优异。然而，众所周知，这类模型作为骨干网络在检索和语义相似度等嵌入任务上表现不佳：二元的“原始词元 vs. 替换词元”信号并未要求最终层保留稠密表示所依赖的词元身份和上下文内容。我们提出了ModernELECTRA，这是一种预训练方案，它将RTD保留为主要目标，并增加了一个轻量级辅助任务——边界窗口重建。对于每个词元，我们采样一个严格包含该词元的3至12个词元的窗口，仅从判别器在窗口两个边界处的隐藏状态来重建该词元的原始身份。该任务在训练时只需额外添加一个小型前馈网络（FFN），训练结束后即被丢弃，但它能迫使每个最终层状态携带信息……

    arXiv:2611.31495v1 Announce Type: new  Abstract: ELECTRA-style replaced-token detection (RTD) is a sample-efficient objective for pre-training text encoders, and RTD-trained models such as DeBERTaV3 are strong on classification benchmarks. The same models, however, are known to be poor backbones for embedding tasks such as retrieval and semantic similarity: the binary "original vs. replaced" signal does not ask the final layer to retain the token identity and contextual content that dense representations rely on. We introduce ModernELECTRA, a pre-training recipe that keeps RTD as the main objective and adds a lightweight auxiliary task, boundary-window reconstruction. For every token, we sample a window of 3-12 tokens that strictly contains it and reconstruct the token's original identity only from the discriminator's hidden states at the two window boundaries. The task costs one small FFN at training time and is discarded afterwards, yet it forces every final-layer state to carry info
    
[^201]: 无需响应建模的执行性预测置信域优化

    Trust-Region Optimization for Performative Prediction without Response Modeling

    [https://arxiv.org/abs/2611.31492](https://arxiv.org/abs/2611.31492)

    PTRO方法通过从当前总体采样评估已知损失、仅在估计总体响应影响时进行额外部署，无需响应建模即可利用已知损失函数对执行性风险进行局部近似优化，并结合置信域机制进行决策更新。

    

    执行性预测研究的是部署决策会改变未来数据分布的学习问题。现有方法通常对总体响应施加结构假设，当固定的模型类别无法很好地刻画该响应时，这限制了它们的适用性。而通用的黑盒优化虽然避免了响应建模，却没有利用已知的损失函数。我们提出了执行性置信域优化（PTRO），它无需指定或拟合响应模型即可优化执行性风险，同时仍能利用已知的损失函数，从而弥合了这两种方法。其关键思想是从当前总体中采样，以评估已知损失在邻近决策处的取值，而额外的部署仅用于估计总体响应的影响。这样无需对总体响应方式作出模型假设，即可获得执行性风险的局部近似。PTRO随后使用置信域机制来确定和适应（摘要在此处截断）

    arXiv:2611.31492v1 Announce Type: new  Abstract: Performative prediction studies learning problems in which deployed decisions change the distribution of future data. Existing methods often impose structure on the population response, which limits their applicability when a fixed model class poorly captures that response. Generic black-box optimization avoids response modeling but does not exploit the known loss function.   We introduce Performative Trust-Region Optimization (PTRO), which bridges these two approaches by optimizing performative risk without specifying or fitting a response model while still exploiting the known loss. The key idea is sampling from the current population to evaluate the known loss at nearby decisions, and use additional deployments only to estimate the effect of population response. This gives a local approximation of performative risk without committing to a model of how the population responds. PTRO then uses a trust-region mechanism to determine and ad
    
[^202]: CHORD：面向受约束仿星器设计的潜空间学习

    CHORD: Learning Latent Spaces for Constrained Stellarator Design

    [https://arxiv.org/abs/2611.31486](https://arxiv.org/abs/2611.31486)

    CHORD是一种条件变分自编码器，通过在训练中惩罚潜空间插值路径上的约束违反并结合自洽推理，实现了仿星器聚变设计中满足物理约束的可行设计插值。

    

    在高维、受偏微分方程约束的设计问题中，在已知可行解之间进行插值可以在无需重复昂贵优化过程的情况下揭示设计之间的权衡关系。在面向聚变能源的仿星器设计中，即使两个端点均可行，对等离子体边界参数的直接插值仍可能违反物理约束。学习得到的潜在表示能够捕捉设计变量之间的非线性关系，但潜空间插值仍可能穿过不可行区域，且以目标指标为条件的调节可能与解码出的设计不一致。我们提出了CHORD，一种条件变分自编码器，它在训练过程中对解码后的潜空间插值路径上预测到的约束违反进行惩罚，并配合自洽推理方法，使条件调节与解码设计的预测指标保持一致。几何基准测试表明了对潜空间插值进行正则化的收益。在不同……之间的仿星器插值中（摘要在此处截断）

    arXiv:2611.31486v1 Announce Type: new  Abstract: In high-dimensional, PDE-constrained design, interpolating between known feasible solutions can reveal trade-offs without repeating expensive optimization campaigns. In stellarator design for fusion energy, direct interpolation of plasma boundary parameters can violate physical constraints even when both endpoints are feasible. Learned latent representations can capture nonlinear relationships among design variables, but latent interpolation can still cross infeasible regions, and conditioning on target metrics can become inconsistent with the decoded design. We introduce CHORD, a conditional variational autoencoder that penalizes predicted constraint violations along decoded latent interpolants during training, paired with self-consistent inference that aligns conditioning with the decoded design's predicted metrics. Geometric benchmarks show the benefit of regularizing latent interpolants. In stellarator interpolation between distinct 
    
[^203]: SwiftLoop：面向循环 Transformer 自适应隐式推理的选择性精炼方法

    SwiftLoop: Selective Refinement for Adaptive Implicit Reasoning in Looped Transformers

    [https://arxiv.org/abs/2611.31481](https://arxiv.org/abs/2611.31481)

    SwiftLoop 提出在推理时通过位置级冻结策略对循环 Transformer 进行选择性自适应精炼，同时优化循环深度与活跃宽度两个维度的计算分配，以更低的计算成本获得更优的推理精度。

    

    循环 Transformer 通过对潜在表示的反复精炼来推进隐式思维链推理。然而，现有的循环推理器通常采用固定深度的循环，在每次迭代中更新所有潜在位置，当无需进一步精炼时会造成冗余计算。我们提出 SwiftLoop，这是一个推理时框架，通过位置级别的冻结实现选择性循环精炼，从而在循环深度和活跃宽度两个维度上实现自适应的计算分配。SwiftLoop 通过由两个互补准则导出的后见之明引导监督来学习轻量级冻结策略：内部表示层面的 K/V 可复用性以及输出层面的功能性影响。在推理阶段，这些策略基于观察到的隐藏状态引导自适应冻结。大量实验表明，SwiftLoop 相比固定循环推理实现了更优的精度-计算权衡。

    arXiv:2611.31481v1 Announce Type: cross  Abstract: Looped Transformers advance implicit chain-of-thought reasoning through recurrent latent refinement. However, existing looped reasoners typically employ fixed-depth recurrence, updating all latent positions at every iteration and incurring redundancy when further refinement is unnecessary. We introduce \textsc{SwiftLoop}, an inference-time framework for selective recurrent refinement through position-wise freezing, enabling adaptive computation allocation across both recurrent depth and active width. SwiftLoop learns lightweight freezing policies through hindsight-guided supervision derived from two complementary criteria: K/V reusability at the internal representation level and functional impact at the output level. At inference time, these policies guide adaptive freezing based on observed hidden states. Extensive experiments demonstrate that SwiftLoop achieves favorable accuracy--computation trade-offs over fixed-loop inference and 
    
[^204]: 基于时空分解的可迁移、时间并行图动力学

    Transferable, Time-Parallel Graph Dynamics via Space-Time Factorization

    [https://arxiv.org/abs/2611.31479](https://arxiv.org/abs/2611.31479)

    该论文提出时空分解方法，将模型拆分为与图无关的局部时间算子和无图依赖参数的空间混合器，从而实现同一参数在未见图上的零样本迁移，并支持非递归的时间并行评估。

    

    时间并行神经算子（FNO 及其后续方法）已经变革了规则网格上的 PDE 代理建模。基于网格的算子（MeshGraphNets、Transolver）将算子学习扩展到了不规则几何结构，但在保持非递归时间评估的同时跨多种拓扑结构进行联合学习仍然具有挑战性。我们提出了时空分解方法：将模型划分为局部时间算子（可学习的、与图无关的）和一个空间混合器，后者在节点间传播信息时不需要依赖图的已学习参数。这一结构性条件使相同的参数能够在未见过的图上运行；非递归实例还支持时间并行评估。准确的零样本迁移进一步取决于控制三种信息泄漏（图相关权重、图特定特征以及图相关训练分布），以及合适的任务物理特性和训练覆盖范围。该协议适用于两种不同的空间混合器……

    arXiv:2611.31479v1 Announce Type: new  Abstract: Time-parallel neural operators (FNO and successors) have transformed PDE surrogates on regular grids. Mesh-based operators (MeshGraphNets, Transolver) extend operator learning to irregular geometries, but jointly learning across diverse topologies while retaining non-recurrent temporal evaluation remains challenging. We propose space-time factorization: partition the model into local temporal operators (learned, graph-independent) and a spatial mixer that propagates information across nodes without graph-dependent learned parameters. This structural condition lets the same parameters run on unseen graphs; non-recurrent instances also admit time-parallel evaluation. Accurate zero-shot transfer further depends on controlling three information leaks (graph-dependent weights, graph-specific features, and graph-correlated training distributions) and on suitable task physics and training coverage. The protocol applies to two distinct spatial-m
    
[^205]: 通过时间偏好从人类纠正中学习

    Learning from Human Corrections through Temporal Preferences

    [https://arxiv.org/abs/2611.31446](https://arxiv.org/abs/2611.31446)

    该论文提出将人类纠正机器人时介入前后的行为及其时间关系转化为偏好对，从而学习更准确逼近人类潜在奖励的价值函数，并在目标推断与纠正预测两个下游任务上验证了其有效性。

    

    当人们介入纠正机器人时，他们不仅通过纠正机器人行为的方式提供信息，还通过选择何时介入来提供信息。现有的从纠正中学习方法侧重于将新轨迹与机器人原本预期的行为进行比较，因而未能充分利用介入前机器人行为所处的情境。在这项工作中，我们研究了如何将介入前和介入后的行为转化为偏好对，以学习逼近人类潜在奖励的价值函数。我们通过比较纠正后的行为与其所替代的行为来构建介入后偏好，并从导致介入发生的各个状态之间的时间关系中构建介入前偏好。我们通过两个下游任务来评估学习到的价值表示：目标推断和纠正预测。我们的结果表明，介入后偏好是……（摘要原文在此处截断）

    arXiv:2611.31446v1 Announce Type: cross  Abstract: When people intervene to correct a robot, they provide information not only through how they correct its behavior, but also through when they choose to intervene. Existing approaches to Learning from Corrections focus on comparing the new trajectory to the robot's originally intended behavior, thus underutilizing the context of the robot's behavior before the intervention. In this work, we investigate how both pre- and post-intervention behavior can be translated into preference pairs for learning value functions that approximate a human's latent reward. We construct post-intervention preferences by comparing corrected behavior with the behavior it replaces, and pre-intervention preferences from temporal relationships among states leading up to an intervention. We evaluate the learned value representations through two downstream tasks: goal inference and correction forecasting. Our results show that post-intervention preferences are pa
    
[^206]: 用于语义反转感知检索的小型学习投影及谱加速：一项四领域实证研究

    Small Learned Projections for Semantic-Reversal-Aware Retrieval with Spectral Acceleration: A Four-Domain Empirical Study

    [https://arxiv.org/abs/2611.31442](https://arxiv.org/abs/2611.31442)

    该论文提出一种小型学习投影，在法律、医疗、专利、金融四个领域实证中能有效阻止对抗性“陷阱”文档进入检索前10名，同时保留93-99%的正确释义文档，并借助边界条件化的谱图检索加速器实现大规模检索加速。

    

    稠密嵌入检索器可能在对抗性场景下失效：当查询本应由文档A回答，却被一个与A仅有一处微小改变含义编辑的“近复制”文档B（即“陷阱”）反转时，在5,000文档池上，E5-Large-v2的原始余弦相似度会将B排入82-100%查询的前10名。我们研究两件事。第一，一个小型学习投影 W: R^1024 → R^d（d ∈ {11, 32}），通过确定性方案在一组声明的案例上训练，以抑制陷阱：对于声明的案例，在四个独立领域（法律、医疗、专利、金融）以及高达100,000的文档池的全部十二个（领域，池大小）设置中，HN@10 = 0.0000（前10名中不含陷阱），同时在精确最近邻搜索下，93-99%的案例中正确的释义文档得以保留。第二，一个边界条件化的加速器，用于在该规模下服务该投影：一种对谱图检索的加速方法（SpectRAG），以边界……替代密集特征分解。（摘要在此处截断）

    arXiv:2611.31442v1 Announce Type: new  Abstract: Dense-embedding retrievers can fail adversarially: when a query is answered by a document A and reversed by a near-copy B that differs from A by a small meaning-changing edit (a "trap"), raw cosine over E5-Large-v2 places B in the top-10 for 82-100% of queries at a pool of 5,000. We study two things. First, a small learned projection W: R^1024 -> R^d (d in {11, 32}), trained by a deterministic recipe on a declared set of cases, that suppresses the trap: for the declared cases, HN@10 = 0.0000 (no trap in the top-10) in all twelve (domain, pool-size) settings across four independent domains (legal, medical, patent, financial) and pools up to 100,000, while the correct paraphrase is retained for 93-99% of cases, under exact nearest-neighbour search. Second, a boundary-conditioned accelerator that serves the projection at that scale: an acceleration of spectral graph retrieval (SpectRAG) that replaces a dense eigendecomposition with a bounda
    
[^207]: 带签名的Transformer：迈向路径空间中的通用上下文学习

    Transformers with Signatures: Towards Universal In-Context Learning in Path Space

    [https://arxiv.org/abs/2611.31439](https://arxiv.org/abs/2611.31439)

    该论文利用路径签名特征建立了面向路径空间的上下文学习理论框架，显式构造了可模拟上下文Lasso的每层双注意力头因果softmax Transformer，并给出超越IID数据设定的有限样本预测误差界。

    

    上下文学习（ICL）使Transformer能够通过演示样例适应新任务，而无需更新其参数。然而，针对Transformer的理论保证在一般情况下仍然有限，尤其是当目标响应非线性地依赖于观测路径的演化时，更是难以获得。我们利用路径签名特征，建立了一个面向路径空间的上下文学习框架。基于签名的通用线性逼近性质，我们显式构造了一个每层具有两个注意力头的因果softmax Transformer，用以模拟上下文Lasso。在稀疏签名近似以及适当的矩和集中性条件下，我们建立了有限样本预测误差界。我们的分析超越了现有上下文学习理论中的IID数据设定，允许演示样例与查询之间存在依赖关系。在上下文稀疏恢复、非线性路径回归、执行成本预测和金融（数据）上的实验……

    arXiv:2611.31439v1 Announce Type: new  Abstract: In-context learning (ICL) enables transformers to adapt to new tasks from demonstrations without updating their parameters. However, theoretical guarantees for transformers remain limited in general, and elusive in particular when target responses depend nonlinearly on the evolution of observed paths. We establish an ICL framework for path space, using path signature features. Building on the universal linear approximation property of signatures, we construct explicitly a causal softmax transformer with two attention heads per layer that emulates in-context Lasso. Under sparse signature approximation and suitable moment and concentration conditions, we establish finite-sample prediction error bounds. Our analysis goes beyond the IID data setting in the existing ICL theory, allowing dependence among demonstrations and the query.   Experiments on in-context sparse recovery, nonlinear path regression, execution-cost prediction, and financia
    
[^208]: 4比特足以用于脑电睡眠分期吗？MXFP4、NVFP4和FP8的受试者级别基准测试

    Is 4 Bits Enough for EEG Sleep Staging? A Subject-Level Benchmark of MXFP4, NVFP4 and FP8

    [https://arxiv.org/abs/2611.31436](https://arxiv.org/abs/2611.31436)

    该论文首次在受试者级别系统评估了FP16至4位块缩放格式（MXFP4、NVFP4）及FP8在脑电睡眠分期中的量化性能，发现8/16位量化几乎无损，4位量化下时间上下文网络最多仅损失1.19个百分点，且NVFP4优于MXFP4，其中量化原始脑电信号的第一层最为敏感。

    

    在可穿戴设备上进行睡眠分期可以保护原始脑电图（EEG）数据的隐私，并节省传输所需的能量，但这同时也需要小型、节能的神经网络。我们对13种训练后量化（PTQ）配置进行了基准测试，范围涵盖FP16到块缩放的4位格式MXFP4和NVFP4，任务是基于单通道Fpz-Cz脑电信号进行五阶段睡眠分期。我们使用了Sleep-EDF-78数据集（78名受试者）、受试者级别的交叉验证、三个训练随机种子、数值精确仿真以及针对每位受试者的配对统计。结果显示，所有8位和16位格式与FP32的Macro-F1差距均保持在0.45个百分点以内。使用4位权重和激活值时，逐epoch的CNN损失1.98至3.04个百分点，而时间上下文网络最多损失1.19个百分点。相同编码器的消融实验表明，时间上下文对这一性能差距的解释作用很小。在相同块大小下，NVFP4优于MXFP4。量化原始脑电信号的第一层是两种网络中最敏感的层。

    arXiv:2611.31436v1 Announce Type: new  Abstract: Staging sleep on a wearable device keeps the raw electroencephalogram (EEG) private and saves the energy needed to transmit it. It also requires small, energy-efficient neural networks. We benchmark 13 post-training quantization (PTQ) configurations, from FP16 to the block-scaled 4-bit formats MXFP4 and NVFP4, for five-stage sleep staging from single-channel Fpz-Cz EEG. We use Sleep-EDF-78 (78 subjects), subject-wise cross-validation, three training seeds, value-exact emulation, and per-subject paired statistics. All 8- and 16-bit formats stay within 0.45 percentage points (pp) of FP32 Macro-F1. With 4-bit weights and activations, an epoch-wise CNN loses 1.98 to 3.04 pp, and a temporal-context network loses at most 1.19 pp. A same-encoder ablation shows that temporal context explains little of this gap. At equal block size, NVFP4 outperforms MXFP4. The first layer, which quantizes the raw EEG, is the most sensitive layer in both networks
    
[^209]: 回收器：基于仿真活动高保真数据的在途机器学习——一种构建前沿模型的优化方法

    The Recycler: In-Transit Machine Learning on High-Fidelity Data from Simulation Campaigns as an Optimized Approach for Building Frontier Models

    [https://arxiv.org/abs/2611.31434](https://arxiv.org/abs/2611.31434)

    该论文首次实现了在同一台百亿亿次超算系统上，将多个并行仿真产生的高保真数据以流式传输方式直接用于在途机器学习训练，从而绕过文件系统带宽和容量的限制，优化前沿模型的构建。

    

    我们报告了首次实现在途机器学习，其使用从多个并行运行的仿真中流式传输的数据，在同一台百亿亿次（exascale）高性能计算系统上进行训练。该工作流程可广泛应用于仿真活动发挥重要作用的各类场景。与单一仿真不同，仿真活动会并行运行多个仿真。这对于寻找诸如粒子加速器或聚变反应堆等机器的最佳运行点、量化模型预测的变化（例如在气候研究中），或者发现新的能力（例如在材料或药物设计中）都具有重要意义。此类仿真活动会以极高的速率产生海量高保真数据。文件系统的带宽和容量限制使得无法保留全部数据，通常需要先将数据缩减数个数量级才能存储。我们展示了一种工作流程，可以将数据从多个仿真中流式传输……

    arXiv:2611.31434v1 Announce Type: cross  Abstract: We report on the first realization of in-transit machine learning using data streamed from multiple simulations running in parallel to the training on the same exascale high performance computing system. This workflow can be commonly applied to a broad variety of scenarios in which simulation campaigns play an important role. Unlike single simulations, campaigns run several simulations in parallel. This is interesting for finding an optimum operation point of a machine like a particle accelerator or a fusion reactor, quantifying the variation in model predictions e.g. in climate research, or discovering new capabilities e.g. in materials or drug design. Such campaigns produce huge amounts of high-fidelity data at high rates. File system bandwidth and capacity prohibit retaining all of that data, usually requiring orders of magnitude in reduction before it can be stored. We demonstrate a workflow that streams the data from multiple simu
    
[^210]: CaLA：自适应策略下世界模型的因果潜变量调整

    CaLA: Causal Latent Adjustment for World Models under Adaptive Policies

    [https://arxiv.org/abs/2611.31431](https://arxiv.org/abs/2611.31431)

    该论文提出 CaLA，一种受 DAG 约束的世界模型，通过从处理前历史中推断潜变量玩家表征，消除自适应策略在想象 rollout 中引入的处理-结果混杂，从而在三消难度调整任务中获得最可靠的因果干预估计和有竞争力的个体化决策。

    

    世界模型通过想象的 rollout 来回答“如果-会怎样”的问题。但由自适应策略收集的 rollout 会在想象 rollout 中引入“处理-结果”混杂。我们在三消（Match-3）游戏难度调整这一场景中研究该问题，其中潜藏的玩家技能会同时影响所分配的难度与玩家流失。我们构建了一个完全指定的三消结构因果模型（SCM），包含序列化游戏机制、自适应分配机制以及成对的真值干预；在该模型上，未经因果调整、直接拟合记录轨迹的模型会选择会提高流失率的难度设置。随后，我们提出因果潜变量调整，这是一种受 DAG 约束的世界模型，能够从处理前历史中推断变分玩家表征。在各项评估中，CaLA 提供了最可靠的总体层面干预估计以及具有竞争力的个体化决策，尽管丰富的手工设计历史特征在个体层面调整上仍可能保持优势。我们进一步（原文在此处截断）

    arXiv:2611.31431v1 Announce Type: new  Abstract: World models is answering ``imagine" answers to ``what-if'' questions via imagined rollouts. But rollouts collected by adaptive policies induce treatment-outcome confounding in imagined rollouts. We study this problem in Match-3 difficulty adjustment, where latent player skill influences both assigned difficulty and churn. We build a fully specified Match-3 SCM with sequential mechanics, adaptive assignment, and paired ground-truth interventions, on which unadjusted models fit to the logged trajectories select difficulty settings that raise churn. We then introduce Causal Latent Adjustment (CaLA), a DAG-constrained world model that infers a variational player representation from pretreatment history. Across our evaluations, CaLA provides the most reliable population-level intervention estimates and competitive individualized decisions, although rich handcrafted histories can remain advantageous for individual-level adjustment. We further
    
[^211]: 学什么 vs. 学怎么做：智能体训练中任务能力与界面能力的分解

    Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training

    [https://arxiv.org/abs/2611.31427](https://arxiv.org/abs/2611.31427)

    该论文将智能体能力分解为“任务能力”（解决任务所需的知识与推理）和“界面能力”（通过特定框架表达知识的能力），揭示了基于框架的训练效果取决于这两种能力的交互方式，并提出多框架蒸馏可将不同框架下的任务知识聚合到共享策略中。

    

    LLM 智能体通过“框架”运行，框架定义了系统提示、动作格式、反馈循环和资源约束。一个模型可能拥有解决任务所需的知识，却无法通过特定的框架表达这些知识。这引出了一个基本问题：当策略在给定框架下失败时，究竟是缺乏任务知识，还是缺乏操作界面的能力？我们区分了智能体能力的两个组成部分：任务能力，即解决任务所需的知识和推理能力；界面能力，即通过给定框架将该知识付诸行动的能力。在四个推理和智能体基准测试中，我们展示了基于框架条件的训练会因这两种能力的交互方式不同而产生截然不同的结果。多框架蒸馏将通过不同框架收集的轨迹中的任务知识聚合到一个共享策略中，但只有在训练保留或模仿……（摘要截断）时，这些知识才能得以实现。

    arXiv:2611.31427v1 Announce Type: new  Abstract: LLM agents operate through harnesses that define system prompts, action formats, feedback loops, and resource constraints. A model may possess the knowledge needed to solve a task yet fail to express that knowledge through a particular harness. This raises a basic question: when a policy fails under a given harness, is it missing task knowledge or the competence to operate the interface? We distinguish two components of agentic competence: task competence, the knowledge and reasoning needed to solve a task, and interface competence, the ability to act on that knowledge through a given harness. Across four reasoning and agentic benchmarks, we show that harness-conditioned training yields vastly different outcomes depending on how these interact. Multi-harness distillation aggregates task knowledge from trajectories collected through different harnesses into a shared policy, but this knowledge is realised only when training preserves or im
    
[^212]: OmniX：首个喷注与事件统一模型

    OmniX: A first jet and event model

    [https://arxiv.org/abs/2611.31403](https://arxiv.org/abs/2611.31403)

    本文提出OmniX，首个能在单一模型中同时处理喷注级和事件级数据的基础模型架构，其通过联合训练在生成和分类任务上达到甚至超越了专用的仅喷注或仅事件模型。

    

    基础模型是构建粒子物理机器学习算法的一种新方法，其核心思想是单个模型可以适应多种下游任务，从而摊薄训练成本。迄今为止，大多数基础模型原型仅在单个喷注的组成部分上运行，这限制了它们在全事件尺度物理问题（如区分不同物理过程）中的实用性，同时也限制了其性能，因为模型无法捕捉全局关联。本文提出OmniX，这是一种能够在单个模型中同时处理单个喷注和完整事件的架构。利用喷注级的JetClass和Top-tagging数据集以及事件级的Collide-2V数据集，我们证明了在单个喷注和完整事件上联合训练的OmniX，在生成和分类任务中能够达到甚至超过同等的仅喷注模型和仅事件模型的性能。

    arXiv:2611.31403v1 Announce Type: cross  Abstract: Foundation models are a new approach to building machine learning algorithms for particle physics. Their central idea is that a single model can be adapted to a variety of downstream tasks, amortizing training costs. So far, most foundation model prototypes have operated on the constituents of a single jet. This limits their usefulness for physics problems at the scale of full events, such as distinguishing different processes, and restricts their performance, as global correla- tions are invisible to the model. Here, we introduce OmniX, an architecture that operates on both single jets and whole events within a single model. Using the jet-level JetClass and Top-tagging datasets and the event-level Collide-2V dataset, we demonstrate that OmniX trained jointly on single jets and full events matches or exceeds otherwise equivalent jet-only and event-only models in generation and classification.
    
[^213]: CoRe：联合学习用于理解与生成的上下文表征

    CoRe: Jointly Learning Contextual Representations for Understanding and Generation

    [https://arxiv.org/abs/2611.31397](https://arxiv.org/abs/2611.31397)

    CoRe提出让理解与生成任务联合塑造统一的图像与文本上下文表征，一个从零训练的383M模型在无需任何预训练组件的情况下，在理解能力以及文生图、图生文生成效果上均可与基于预训练组件的更大规模模型相媲美。

    

    我们提出了CoRe，其中理解与生成共同塑造针对图像和针对文本各自的一个上下文表征，并从零开始训练。每个表征既被读取用于理解，又对另一模态进行条件化，同时还是该模态从高斯噪声出发经由连续流（continuous flow）进行生成的空间。在CC12M数据集上训练的383M参数模型，在推理阶段无需预训练语言模型、文本编码器或VAE，CoRe在全局理解任务上与相同数据量的纯对比学习基线相当，并达到COCO-30K FID 10.10，可与基于预训练组件构建的更大规模生成器相媲美。在相同数据和架构的对照比较中，CoRe在两个方向（文本到图像和图像到文本）的生成表现都优于仅由理解任务塑造后冻结的表征，并且在遵循提示方面也优于在与理解无关的表征空间中进行生成。在另一个方向上，使用注意力分割（摘要截断）……

    arXiv:2611.31397v1 Announce Type: cross  Abstract: We introduce CoRe, in which understanding and generation jointly shape one contextual representation for images and one for text, trained from scratch. Each representation is read for understanding, conditions the other modality, and is the space in which that modality is generated by a continuous flow from Gaussian noise. With 383M parameters trained on CC12M and no pretrained language model, text encoder, or VAE at inference, CoRe matches a same-data contrastive-only baseline in global understanding and reaches COCO-30K FID 10.10, comparable to much larger generators built on pretrained components. In controlled comparisons at equal data and architecture, CoRe generates better in both directions (text to image and image to text) than a representation shaped by understanding alone and then frozen, and follows prompts better than generating in a space that understanding does not shape. In the other direction, with an attentive segmenta
    
[^214]: 检测有毒饮食文化中的有害短视频：基于稀疏自编码器的可治理证据框架

    Detecting Harmful Short Videos in Toxic Diet Culture: A Governance-Ready Evidence Framework with Sparse Autoencoders

    [https://arxiv.org/abs/2611.31354](https://arxiv.org/abs/2611.31354)

    本文提出SPACE框架，利用大型多模态模型的稀疏自编码器特征空间，识别、定位并将证据对齐到平台政策，为有毒饮食文化等有害短视频的审核治理提供与决策关联且可检验的证据支持。

    

    算法内容审核对于短视频内容的治理日益重要，然而现有的有害内容检测研究主要聚焦于预测或有限的解释，这使得一个治理层面的挑战未被充分探索：如何生成与决策相关联的证据，以支持审核决策的审查与论证。本文以“有毒饮食文化”作为实证案例，将“可治理就绪的证据”概念化为四个维度：计算可追溯性、证据定位、政策依据以及证据可检验性。作者在由大型多模态模型导出的预训练稀疏自编码器（SAE）特征空间中，开发了基于稀疏自编码器的政策对齐对比证据框架（SPACE）：全局相关性门控与可分解的线性证据头用于识别与决策相关联的证据；片段监督与视频内排序机制对证据进行定位；政策对齐将证据映射至平台准则；帕累托高效选择……（原文摘要在此处截断）

    arXiv:2611.31354v1 Announce Type: new  Abstract: Algorithmic content moderation is increasingly important for governing short-form video content, yet harmful-content detection research focuses primarily on prediction or limited explanations. This leaves a governance challenge underexplored: producing decision-linked evidence that supports the review and justification of moderation decisions. Using toxic diet culture as an empirical case, we conceptualize governance-ready evidence as computational traceability, evidence localization, policy grounding, and evidence inspectability. We develop the Sparse Autoencoder-based Policy-Aligned Contrastive Evidence (SPACE) framework in a pretrained SAE feature space derived from a large multimodal model. A global relevance gate and decomposable linear evidence head identify decision-linked evidence; segment supervision and within-video ranking localize this evidence; policy alignment maps it to platform guidelines; and Pareto-efficient selection y
    
[^215]: GHC：扩展超连接中的残差接口

    GHC:Expanding Residual Interfaces in Hyper-Connections

    [https://arxiv.org/abs/2611.31350](https://arxiv.org/abs/2611.31350)

    提出分组超连接，通过沿注意力头或FFN通道将分支分组并为每组提供差异化的流读写接口，在开销可忽略的情况下将训练效率较Pre-Norm提升35%、较IHC提升10%，同时克服了增加流数量时的收益递减问题。

    

    超连接通过并行流扩展了残差容量，但将每个分支限制在狭窄的读写接口上：每个分支仍然只读取单一的流混合，并将同一个输出向量的缩放副本写入所有流。为了突破这一瓶颈，我们提出了分组超连接。通过沿注意力头或前馈网络（FFN）通道将每个分支划分为并行组，GHC为每个组提供不同的流混合，并通过可学习的权重将各组的独立输出聚合到每个流中。在密集模型和混合专家模型上的大量实验表明，GHC在参数和内存访问开销几乎可以忽略的情况下，始终优于恒等超连接，相比Pre-Norm将训练效率提升了35%，相比IHC提升了10%。此外，扩展残差接口克服了增加流数量带来的收益递减问题，使得更宽的残差状态能够带来进一步的收益。

    arXiv:2611.31350v1 Announce Type: cross  Abstract: Hyper-Connections expand residual capacity across parallel streams, yet restrict every branch to narrow read and write interfaces: each branch still reads a single stream mixture and writes scaled copies of one output vector to all streams. To lift this bottleneck, we propose Grouped Hyper-Connections (GHC). By partitioning each branch into parallel groups along attention heads or FFN channels, GHC supplies each group with a distinct stream mixture and aggregates their separate outputs into each stream with learned weights. Extensive experiments across dense and Mixture-of-Experts (MoE) models show that GHC consistently improves upon Identity Hyper-Connections (IHC) with negligible parameter and memory-access overhead, increasing training efficiency by 35% over Pre-Norm and 10% over IHC. Moreover, expanding residual interfaces overcomes the diminishing returns of adding streams, allowing wider residual states to yield further gains.
    
[^216]: 精确保体积微分同胚剪切分裂：连续-离散几何界、可逆内存反向传播与体积映射中的表达能力前沿

    Exact Volume-Preserving Diffeomorphic Shear Splitting: Continuous-Discrete Geometric Bounds, Reversible Memory Backpropagation, and Expressivity Frontiers in Volumetric Mapping

    [https://arxiv.org/abs/2611.31343](https://arxiv.org/abs/2611.31343)

    该论文提出基于级联坐标剪切的精确保体积微分同胚配准框架，将无散度李代数分解为六个剪切生成元，实现闭合形式的精确体积保持、机器精度的零迭代求逆以及可逆的低内存反向传播。

    

    不可压缩微分同胚配准需要保持局部组织体积（det(DΦ) = 1）且不发生拓扑折叠（det(DΦ) > 0）的连续空间映射。标准速度场方法（SVF、LDDMM）通过数值ODE积分或软惩罚来近似不可压缩性，存在数值体积漂移（约10^-4至10^-1）、迭代网格求逆以及O(L * N_vox)的线性反向传播内存开销等问题。

    arXiv:2611.31343v1 Announce Type: cross  Abstract: Incompressible diffeomorphic registration requires continuous spatial mappings that preserve local tissue volume (det(D\Phi) = 1) without topological folding (det(D\Phi) > 0). Standard velocity-field methods (SVF, LDDMM) approximate incompressibility via numerical ODE integration or soft penalties, exhibiting numerical volume drift (~10^-4 to 10^-1), iterative grid inversions, and linear backpropagation memory O(L * N_vox).   We formulate an exact structure-preserving framework for 3D registration based on cascaded coordinate shears. By decomposing the divergence-free Lie algebra sdiff(\Omega) into six coordinate-shear generators whose convective non-linearities vanish identically, each elementary stage integrates analytically to a closed-form triangular shear with det(DS_k) = 1 and exact machine-precision inverse requiring zero iterations and zero spatial interpolations. Multi-stage compositions generate the connected volume-preservin
    
[^217]: 融合收益取决于实现方式与校准目标：一项关于带单源回退的预测区间的18任务研究

    Fusion Gains Depend on the Implementation and the Calibration Target: An 18-Task Study of Prediction Intervals with a Single-Source Fallback

    [https://arxiv.org/abs/2611.31333](https://arxiv.org/abs/2611.31333)

    该研究通过对18个回归任务的实证比较发现，融合方法的收益高度依赖于具体实现方式与校准目标，而结合单源回退的开发集选择器能在所有实现和校准设置下稳定提升预测区间性能。

    

    融合模型既能从额外的学习阶段中受益，也能从第二个数据源中受益。我们在18个带有预测区间的回归任务上，将四种融合实现方式与匹配的单源学习器进行比较。在分组校准下的十次重新拟合中，线性堆叠相较于没有第二阶段的源模型将区间分数提高了6.0%，而相较于其匹配的单源模型仅提高了0.1%。四种实现方式在18个任务中仅在3个任务上结论一致，并在12个任务上给出了截然相反的明确结论，其中包括10个具有天然差异来源任务中的8个。在逐点校准下，它们在5个任务上达成一致，并在9个任务上给出相反的明确结论。在两种校准方式下，带有匹配单源回退的开发集选择器在每种实现方式中都将十次重新拟合的任务平均分数提高了2.2%至6.3%。将融合模型加入单源候选池中，使平均相对分数提高了约5个百分点。在分组……（原文摘要在此处截断）

    arXiv:2611.31333v1 Announce Type: new  Abstract: A fused model can benefit from an extra learning stage as well as from a second data source. We compare four fusion implementations with matched single-source learners on 18 regression tasks with prediction intervals. Across ten refits under group-wise calibration, linear stacking improved the interval score by 6.0% against a source model without a second stage and by 0.1% against its matched single source. The four implementations agreed on 3 of 18 tasks and gave opposite clear verdicts on 12, including 8 of the 10 tasks with naturally distinct sources. With pointwise calibration, they agreed on 5 tasks and gave opposite clear verdicts on 9. A development-set selector with a matched single-source fallback improved ten-refit task-averaged scores by 2.2--6.3% in every implementation under both calibrations. Adding fused arms to the pool of single-source candidates improved the mean relative score by about 5 percentage points. Under group-
    
[^218]: 面向不完整数据的保守重叠模糊粗糙集

    Conservative Overlap Fuzzy Rough Sets for Incomplete Data

    [https://arxiv.org/abs/2611.31329](https://arxiv.org/abs/2611.31329)

    该论文提出保守重叠模糊粗糙集（COFRS）模型，通过对共同存在属性进行加权重叠、全局惩罚缺失值的相似度定义及基于加权粒一致性的软依赖度，实现了不完整数据上高效且稳健的属性约减。

    

    arXiv:2611.31329v1 公告类型：新 摘要：在不完整信息系统中进行属性约减十分困难，因为缺失值破坏了粗糙集理论的等价类结构。我们提出了保守重叠模糊粗糙集（COFRS）模型。每个对象由其存在的属性集合来描述。两个对象的相似度是在两者共同存在的属性上的加权重叠，除以固定的属性数量 |A| 而非其并集的大小，因此缺失值被全局性地惩罚，而不会被当作通配符来匹配。基于该相似度，我们定义了基于加权粒一致性的软依赖度，并通过时间复杂度为 O(|U|^2 |A|^2) 的贪心反向消除方法来搜索约减。我们证明了该关系是对称且非自反的，依赖度并不满足单调性，并给出了单调性成立的充分条件。在缺失率介于 0.15% 到 20.46% 的 11 个 UCI 基准数据集上，COFRS 给出了最小的（原文摘要在此处截断）。

    arXiv:2611.31329v1 Announce Type: new  Abstract: Attribute reduction in incomplete information systems is difficult because missing values break the equivalence-class structure of rough set theory. We propose the Conservative Overlap Fuzzy Rough Set (COFRS) model. Each object is described by its set of present attributes. The similarity of two objects is a weighted overlap on the attributes present in both, divided by the fixed number of attributes |A| instead of the size of their union, so missing values are penalised globally and are not matched as wildcards. From this similarity we define a soft dependency degree based on weighted granule consistency, and we search for reducts by greedy backward elimination in O(|U|^2 |A|^2) time. We show that the relation is symmetric and non-reflexive, that the dependency degree is not monotone, and we give a sufficient condition under which monotonicity holds. On 11 UCI benchmarks with missing rates from 0.15% to 20.46%, COFRS gives the smallest 
    
[^219]: 利用高斯混合模型间的Jeffreys散度标记流式细胞术中的批次偏移

    Flagging Batch Shift in Flow Cytometry using Jeffreys Divergence between Gaussian Mixtures

    [https://arxiv.org/abs/2611.31321](https://arxiv.org/abs/2611.31321)

    提出了一种基于高斯混合模型间Jeffreys散度的与模型无关的方法，用于检测流式细胞术数据中的批次偏移并标记不可靠的预测。

    

    用于临床流式细胞术的机器学习模型通常在来自其训练分布的数据上进行评估。然而在日常实践中，采集条件和试剂可能会发生变化，从而可能在毫无预警的情况下使预测变得不可靠。我们提出了一种与模型无关的预测可靠性判据，其基于与训练分布之间距离的概念：每个患者样本用一个高斯混合模型来表示，并通过蒙特卡洛积分估计其与在训练数据上拟合的参考模型之间的Jeffreys散度。我们在急性髓系白血病患者的真实偏移队列上展示了我们的方法，所用分类器经过训练以预测NPM1突变状态。首先，我们从图形和统计上证明成对散度能够捕捉采集队列的结构。随后，我们在偏移队列样本的一个子集上校准拒绝阈值，并在留出的受试者上进行评估。在2（摘要在此处截断）

    arXiv:2611.31321v1 Announce Type: new  Abstract: Machine learning models for clinical flow cytometry are usually evaluated on data drawn from their training distribution. In routine practice, however, acquisition conditions and reagents may change, potentially making predictions unreliable without warning. We propose a model agnostic criterion for prediction reliability based on a notion of distance from the training distribution: each patient sample is represented by a Gaussian mixture model, and its Jeffreys divergence to a reference fitted on the training data is estimated using Monte Carlo integration. We illustrate our approach on real-life shifted cohorts of acute myeloid leukaemia patients, using a classifier trained to predict NPM1 mutation status. First, we show graphically and statistically that pairwise divergences capture acquisition cohort structure. We then calibrate a rejection threshold on a subset of shifted cohort samples and evaluate it on held-out subjects. Across 2
    
[^220]: EEG基础模型中的层读出接口：预测可及性与跨数据集迁移

    Layer-Readout Interfaces in EEG Foundation Models: Predictive Accessibility and Cross-Dataset Transfer

    [https://arxiv.org/abs/2611.31320](https://arxiv.org/abs/2611.31320)

    本研究解析了EEG基础模型中间层优于最终层的原因：改进最终层读出接口即可恢复大部分预测性能，但冻结提取下中间层仍具优势，且所选择的层与任务方向能够跨数据集迁移。

    

    当基础模型被用作冻结特征提取器时，中间层的表现往往优于最终层，但这一优势的来源尚不清楚。我们在EEG基础模型中通过分离三个因素来研究这一问题：表征如何被读出、中间层是否在最终层接口之外提供额外的预测效用，以及有用的深度结构是否能够跨数据集迁移。在五个模型和三个任务的实验中，改进最终层读出始终能够恢复相当可观的预测性能，然而在冻结提取方式下中间层仍保持优势。我们还发现，在一个数据集上选出的层在其他数据集上依然有用（验证跨越五个数据集对和四个任务族），并且一组紧凑的、源自源数据集的任务方向在迁移到睡眠分期任务后仍保留预测价值。综合来看，这些结果表明预测效用在网络深度上存在变化，其有用性

    arXiv:2611.31320v1 Announce Type: cross  Abstract: Intermediate layers often outperform final layers when foundation models are used as frozen feature extractors, but the source of this advantage is unclear. We study this question in EEG foundation models by separating three factors: how representations are read out, whether intermediate layers provide predictive utility beyond the final-layer interface, and whether useful depth structure transfers across datasets. Across five models and three tasks, improving the final-layer readout consistently recovers substantial predictive performance, yet intermediate layers remain advantageous within frozen extraction. We also find that layers selected on one dataset remain useful on another across five dataset pairs and four task families, and that a compact set of source-derived task directions retains predictive value after transfer in sleep staging. Together, these results show that predictive utility varies across depth, and its usefulness 
    
[^221]: 保形化树型梯度集成用于分布外检测与合理反事实救济

    Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse

    [https://arxiv.org/abs/2611.31319](https://arxiv.org/abs/2611.31319)

    该论文提出将保形化的分布外异常检测门控与可操作性约束的反事实搜索相结合，防止反事实救济利用梯度提升树决策边界的伪影和低密度分布外区域，从而生成更合理的分布内救济建议。

    

    反事实救济提供了可操作的特征调整，能够扭转自动化决策模型做出的不利预测。然而，当救济搜索纯粹针对分类器置信度或基于梯度提升决策树（GBDT）的狭窄代理分数进行优化时，搜索过程可能利用边界伪影和低密度区域——这是古德哈特定律的一个实例——从而产生名义上看似有利但缺乏流形合理性的分布外（OOD）建议。本文在两个表格数据基准（Adult 和 Give Me Some Credit, GMSC）上研究了将保形化 OOD 门控与可操作性约束的反事实搜索相结合的方法。研究者在显著性水平 δ = 0.05 下，基于孤立森林、局部离群因子和 k 近邻信号校准有限样本异常门控（gate_IF、gate_LOF 以及合取式 gate_ALL），并利用保留的高斯混合模型（GMM）对生成的反事实进行审计（原文摘要在此处被截断）。

    arXiv:2611.31319v1 Announce Type: cross  Abstract: Counterfactual recourse provides actionable feature adjustments that reverse unfavourable predictions made by automated decision models. However, when recourse search optimizes purely against classifier confidence or narrow proxy scores over gradient-boosted decision trees (GBDT), search procedures can exploit boundary artifacts and low-density regions---an instance of Goodhart's law---yielding out-of-distribution (OOD) recommendations that appear favourable nominally while lacking manifold plausibility. In this paper, we study conformalized OOD gating coupled with actionability-constrained counterfactual search across two tabular benchmarks: Adult and Give Me Some Credit (GMSC). Calibrating finite-sample anomaly gates (gate_IF, gate_LOF, and the conjunctive gate_ALL) at significance level delta = 0.05 over isolation forest, local outlier factor, and k-nearest-neighbor signals, we audit generated counterfactuals against held-out GMM an
    
[^222]: SlotFill：面向流片后验证的有损SoC通信踪迹修复方法

    SlotFill: Repairing Lossy SoC Communication Traces for Post-Silicon Validation

    [https://arxiv.org/abs/2611.31313](https://arxiv.org/abs/2611.31313)

    SlotFill通过为消息间隙分配固定槽位并利用基于Transformer的模型结合监视器字段信息，在流片后验证中精确恢复SoC通信踪迹中丢失的消息，在15%丢失率下恢复率高达97%。

    

    流片后验证依赖于SoC组件之间交换消息的踪迹，但踪迹缓冲区和监视器会丢失消息，且通常不会留下任何丢失标记。我们提出SlotFill，它将每条被删除的消息放回正确的间隙中，并恢复正确的类型和顺序。SlotFill为剩余消息之间的每个间隙分配固定数量的槽位，由基于Transformer的模型在每个槽位中填入一条消息，或在该处没有消息被删除时留空，因此输入不会泄露到底丢失了多少消息。该模型还利用了存活消息中的监视器字段（时间戳、事务ID、地址）。在4核和8核系统的完整程序gem5踪迹（20.7亿条消息）上，SlotFill在15%丢失率下可精确恢复高达97%的已删除消息，在50%丢失率下可恢复高达87%；当时间戳较粗糙时，在15%丢失率下仍能恢复76-83%。

    arXiv:2611.31313v1 Announce Type: cross  Abstract: Post-silicon validation relies on traces of messages exchanged by SoC components, but trace buffers and monitors drop messages, often leaving no mark of what was lost. We present SlotFill, which puts each deleted message back in the right gap, with the right type and order. SlotFill gives every gap between the remaining messages a fixed number of slots, and a Transformer-based model fills each slot with a message or leaves it empty when nothing was deleted there, so the input never reveals how much was lost. The model also uses monitor fields (timestamps, transaction IDs, addresses) from surviving messages. On whole-program gem5 traces of 4- and 8-core systems (2.07 billion messages), SlotFill restores up to 97% of deleted messages exactly at 15% loss and up to 87% at 50% loss, and 76-83% at 15% loss when timestamps are coarse.
    
[^223]: 基于集合的综合学习与带记忆的粒子群优化算法用于离散优化问题

    Set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem

    [https://arxiv.org/abs/2611.31306](https://arxiv.org/abs/2611.31306)

    提出了一种融合记忆机制的基于集合的综合学习粒子群优化算法，用于求解离散优化问题，并通过与现有算法的数值实验对比验证了其有效性。

    

    本文提出了一种新颖的算法——针对离散优化问题的基于集合的综合学习与带记忆的粒子群优化算法。该算法在更新解的过程中，通过参考当前位置来在集合中搜索最佳位置。为了验证该算法的有效性，我们进行了与现有算法对比的数值实验。

    arXiv:2611.31306v1 Announce Type: cross  Abstract: This paper describes a novel algorithm of set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem. Our algorithm is an approach of searching for the best position in a set by referring to the current position when updating the solution. In order to show the validity of our algorithm, we examine numerical experiments compared to existing algorithms.
    
[^224]: 基于扰动的差分进化算法用于连续函数优化

    Differential evolution with perturbation for continuous function optimization

    [https://arxiv.org/abs/2611.31305](https://arxiv.org/abs/2611.31305)

    本文提出了一种结合扰动机制的差分进化算法（DEP）用于连续函数优化，并通过CEC2017数据集与现有算法的对比验证了其有效性。

    

    本文提出了一种用于连续函数优化的扰动差分进化算法（DEP）。差分进化是一种简单而高效的算法。扰动是指主力的贡献所产生的运动受到次级力影响而被干扰的现象。为了证明所提出算法的有效性，我们使用CEC2017数据集将其与现有算法进行了比较。

    arXiv:2611.31305v1 Announce Type: cross  Abstract: This paper presents differential evolution with perturbation (DEP) for continuous function optimization. Differential evolution is a simple and efficient algorithm. Perturbation is the phenomenon that the motion by the contribution of the main force is disturbed by the influence subsidiary power. In order to show the effectiveness of the proposed algorithm, we compare it to existing algorithms by CEC2017 datasets.
    
[^225]: WM-811K中的隐藏线索：揭示晶圆图预测背后的空间信号

    Hidden Clues in WM-811K: Uncovering the Spatial Signals Behind Wafer Map Prediction

    [https://arxiv.org/abs/2611.31292](https://arxiv.org/abs/2611.31292)

    本研究通过为WM-811K晶圆图构建控制变量（布局、失效芯片数量、距中心距离、二维位置）的多种变体，并严格清洗数据集，系统揭示了卷积神经网络预测性能背后真正起作用的空间特征。

    

    WM-811K（也称为LSWMD）是一个晶圆图数据集，最早于2015年由Wu等人在论文中公开发布。卷积神经网络在该数据集上已实现很高的分类准确率，然而很难解读究竟是数据集的哪些特征驱动了这种性能。本研究专门针对WM-811K尝试回答这一问题：预测性能中有多少来自晶圆布局、失效芯片的数量、它们与中心的距离以及它们精确的二维位置。在分析重复晶圆图、删除具有冲突标签的相同晶圆图，并确保同一制造批次或完全相同的晶圆图不会同时出现在训练、验证和测试数据中之后，我们得到了来自八个类别、共25,513张带有缺陷标签的晶圆图。随后我们为每张晶圆图构建了四种变体。原始版本是数据集中实际存在的晶圆图。仅布局版本则只保留了布局……

    arXiv:2611.31292v1 Announce Type: new  Abstract: WM-811K (also known as LSWMD) is a wafer map dataset first made publicly available in a 2015 paper by Wu et al. Convolutional neural networks have been used to achieve high classification accuracy on this dataset, yet it's not easy to decipher which features of the dataset are driving that performance. This study attempts to answer this specifically for WM-811K: how much of the predictive performance results from the wafer layout, the count of failed dies, their distance from the center and their exact 2-D location. After analyzing duplicate wafer maps, removing same wafer maps with conflicting labels and ensuring that the same manufacturing lot or exact same map doesn't appear in training, validation and test data, we were left with 25,513 defect-labeled wafer maps from eight classes. We then constructed four variations of each wafer map. The Original was the actual wafer map present in the dataset. The Layout-only kept only the layout 
    
[^226]: 大语言模型遗传特征工程

    LLM Genetic Feature Engineering

    [https://arxiv.org/abs/2611.31277](https://arxiv.org/abs/2611.31277)

    该论文提出了一种在单个特征层面而非整个特征集层面执行选择、交叉和变异的新型遗传算法，使遗传算法能够在过程中对新特征进行统计分析，并同时整合个体与群体层面的特征指标，从而改进基于LLM的特征工程自动化。

    

    特征工程是一种（传统上由人工完成的）过程，通过组合、转换或其他方式操作原始数据字段，从现有数据集中构建派生变量，以减少统计噪声，并从原始数据中提取有意义的、特定领域的关注量。近期研究表明，通过整合大语言模型（LLM）和各种元启发式算法来指导有用特征的搜索，在特征工程自动化方面取得了可喜的进展。我们在此方法的基础上，引入了一种新颖的遗传算法，该算法在单个特征层面而非整个工程特征集层面实现选择、交叉和变异操作。由此，我们能够对新生的特征进行过程中的统计分析，从而使遗传算法能够同时将个体和群体层面的特征指标整合到其选择过程中。

    arXiv:2611.31277v1 Announce Type: cross  Abstract: Feature Engineering is the (traditionally manual) process of constructing derived variables from an existing dataset by combining, transforming, or otherwise manipulating the original data fields to reduce statistical noise and extract meaningful, domain-specific quantities of interest from the raw dataset. Recent work demonstrates promising progress toward the au tomation of this process through the integration of large language models (LLMs) and various metaheuristics to guide the search for useful features. We build on this approach by introducing a novel genetic algorithm that implements selection, crossover, and mutation at the individual feature level, rather than at the level of the entire engineered feature set. In doing so, we enable in-process statistical analysis of newly-generated features, thereby enabling the genetic algorithm to simultaneously integrate individual and group-level feature metrics into its selec tion proce
    
[^227]: 思维链并非可解释性

    Chain-of-Thought Is Not Explainability

    [https://arxiv.org/abs/2611.31271](https://arxiv.org/abs/2611.31271)

    本立场论文指出，思维链生成的推理说明经常不忠实于模型真实的内部隐藏计算，因而不能作为可信的可解释性依据，既非必要也非充分。

    

    思维链（CoT）允许语言模型在给出最终答案之前，以语言形式表达多步推理过程。虽然这一技术通常能提升任务表现，并给人一种洞察模型推理过程的透明感，但本立场论文认为，当前CoT技术生成的推理说明可能具有误导性，对于可信的可解释性而言既非必要也非充分条件。我们将忠实性界定为：CoT不仅应当对人类可理解，还应当足够好地反映模型的底层推理过程，以支持负责任的使用。通过综合先前工作的证据，我们系统地表明，语言化的推理链经常是不忠实的，与驱动模型预测的关键隐藏计算相背离。随着CoT在各类应用领域（包括医疗、法律和自主系统等高风险领域）中被日益依赖，我们进一步认为，这种不忠实性并非……（摘要截断）

    arXiv:2611.31271v1 Announce Type: new  Abstract: Chain-of-thought (CoT) allows language models to verbalise multi-step rationales before producing their final answer. While this technique often boosts task performance and offers an impression of transparency into the model's reasoning, this position paper argues that rationales generated by current CoT techniques can be misleading and are neither necessary nor sufficient for trustworthy interpretability. We analyse faithfulness as whether CoTs are not only human-interpretable but also reflect the model's underlying reasoning well enough to support responsible use. By synthesising evidence from prior work, we systematically show that verbalised chains are frequently unfaithful, diverging from the critical hidden computations that drive a model's predictions. As CoT is increasingly relied upon in application domains, including high-stakes ones such as medicine, law, and autonomous systems, we further argue that this unfaithfulness is not
    
[^228]: ToneMamba：基于边界重置状态空间模型学习上下文相关的声调轮廓

    ToneMamaba: Learning Contextualized Tonal Contours with Boundary-Reset State Space Models

    [https://arxiv.org/abs/2611.31267](https://arxiv.org/abs/2611.31267)

    ToneMamba通过在音节、词和话语单元层面进行双向状态重置的边界感知状态空间模型，结合节奏条件门控建模时长信息，显著提升了普通话声调识别性能，在会话语音上提升尤为突出。

    

    连续语流中的普通话声调识别极具挑战性，因为语境和时长的变化会重塑基音频率（F0）轮廓。我们提出ToneMamba，这是一种边界感知的状态空间模型，它在音节、词和话语单元（IPU）三个层面用双向状态重置取代静态边界嵌入。模型采用固定权重融合多层次表征，并通过共享权重的双向扫描结合逐通道方向融合来减少参数量。动态音节池化用于构建声调轮廓，节奏条件门控则引入时长信息。ToneMamba在FCU-VOICE-360朗读语音和MCDC-8会话语音上分别达到97.9%和74.8%的准确率，分别超越基线模型ToneNet 0.9和15.9个百分点。消融实验表明，词级重置尤为有效，而节奏条件化在会话语音中带来更大收益，凸显了时长变化在自然口语中的重要作用。

    arXiv:2611.31267v1 Announce Type: new  Abstract: Mandarin tone recognition in connected speech is challenging because contextual and durational variation reshapes F0 contours. We propose ToneMamba, a boundary-aware state space model that replaces static boundary embeddings with bidirectional state resets at syllable, word, and IPU levels. Fixed weights fuse multi-level representations, while shared-weight bidirectional scans use channel-wise directional fusion to reduce parameters. Dynamic syllable pooling forms tonal contours, and rhythm-conditioned gating incorporates durational information. ToneMamba achieves 97.9% and 74.8% accuracy on FCU-VOICE-360 read and MCDC-8 conversational speech, outperforming the baseline (ToneNet) by 0.9 and 15.9 percentage points, respectively. Ablations show that word-level resets are particularly effective, while rhythm conditioning provides greater gains in conversational speech, underscoring the role of durational variation in spontaneous speech.
    
[^229]: SD-Cache：基于状态引导动态缓存的可学习记忆整合

    SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching

    [https://arxiv.org/abs/2611.31265](https://arxiv.org/abs/2611.31265)

    提出SD-Cache，通过状态引导的动态缓存机制，在不预知未来查询的情况下，将LLM原生的KV状态选择性地整合到固定大小的工作记忆中，解决了RAG场景下KV缓存线性增长的问题。

    

    大型语言模型（LLM）越来越多地被部署为需要对海量外部知识进行推理的持久化服务。检索增强生成（RAG）虽然将模型响应建立在相关文档的基础上，但检索到的内容仍需被编码并存储在键值（KV）缓存中。随着这些内容不断积累，缓存呈线性增长，导致内存消耗和推理成本上升。现有的记忆压缩方法要么通过保留缓存状态的一个子集来解决该问题，要么通过学习将文档映射为紧凑潜在表示的上下文压缩器。尽管取得了这些进展，如何在紧张的内存预算下整合信息、同时保持其对未来查询的有用性，仍然是一个挑战。我们研究了如何在无法访问未来查询的情况下，将模型原生的KV状态选择性地整合到一个固定大小的工作记忆中。为此，我们提出了SD-Cache（状态引导……）

    arXiv:2611.31265v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as persistent services that must reason over large volumes of external knowledge. Retrieval-augmented generation (RAG) grounds model responses in relevant documents, but the retrieved content must still be encoded and stored in the key-value (KV) cache. As this content accumulates, the cache grows linearly, increasing memory consumption and inference cost. Existing memory-compression methods address this problem either by retaining a subset of cached states or by learning context compressors that map documents to compact latent representations. Despite these advances, consolidating information under tight memory budgets while preserving its usefulness for future queries remains challenging. We investigate how the model's native KV states can be selectively consolidated into a fixed-size working memory without access to future queries. To this end, we introduce SD-Cache (State-guide
    
[^230]: Actionwave：面向多种医疗场景的以波形为中心的临床预测系统性基准

    Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios

    [https://arxiv.org/abs/2611.31249](https://arxiv.org/abs/2611.31249)

    Actionwave 是首个覆盖重症监护、手术室和急诊科等多种医疗场景的以波形为中心的临床预测系统性基准，通过统一的决策时点预测接口、严格的评估协议，系统考察了信号时长、预警时间和可用输入等条件对模型表现的影响。

    

    连续的床旁波形记录了快速的生理变化，可为临床决策提供信息。现有的基准测试仅考察了选定的预测条件，但目前尚不清楚模型优势如何依赖于信号时长、预警时间以及不同医疗场景下的可用输入。我们提出了 Actionwave，这是一个面向重症监护、手术室和急诊科场景的以波形为中心的临床预测基准。Actionwave 采用统一的决策时点预测接口，并根据推断的内容和时间对具有临床依据的任务进行分组。它改变了与临床使用相关的条件，包括干预前所需的预警时间。我们在指定的模型规模和计算预算下，比较了多种架构和预训练生物信号编码器。该协议要求患者级数据划分、严格的信息截断，以及能够隔离单个建模因素的比较。在 20 个任务和 140 万个任务标签……

    arXiv:2611.31249v1 Announce Type: cross  Abstract: Continuous bedside waveforms record rapid physiological changes that can inform clinical decisions. Existing benchmarks examine selected prediction conditions, but it remains unclear how model advantages depend on signal duration, warning time, and available inputs across care settings. We introduce Actionwave, a benchmark for waveform-centric clinical prediction across intensive-care, operating-room, and emergency-department settings. Actionwave uses a common decision-time prediction interface and groups clinically grounded tasks by what is inferred and when. It varies conditions relevant to clinical use, including the warning required before an intervention. We compare diverse architectures and pretrained biosignal encoders under specified model-size and compute budgets. The protocol requires patient-level splits, strict information cutoffs, and comparisons that isolate individual modeling factors. Across 20 tasks and 1.4M task-label
    
[^231]: 用于光谱预测的局部子空间编码：从TabPFN到算子自适应定标

    Local subspace encoding for spectral prediction: from TabPFN to operator-adaptive calibration

    [https://arxiv.org/abs/2611.31245](https://arxiv.org/abs/2611.31245)

    该研究证实局部振动子空间编码（LVSE）表示方法在nirs4all生态系统的光谱数据集上能显著降低TabPFN的RMSE并提升分类加权F1，且其价值可扩展至经典定标方法。

    

    TabPFN已经在近红外光谱上展现出了有用的预测性能。随后，RamanPFN提出了两种用于表格推理的光谱表示方法：全局成分解混和局部振动子空间编码。我们评估了这些表示方法是否也能在使用nirs4all生态系统构建的数据集上发挥作用，以及它们的价值是否能延伸到经典定标方法。我们重新分析了来自公共光谱数据集目录的已完成实验。在61个匹配的数值预测任务的事后子集上（剔除了十个类别代码、元数据或语义模糊的目标变量后），LVSE相对于原始光谱TabPFN，在检查点2.5、3和3.5处平均分别将RMSE降低了10.22%、12.80%和14.52%，中位数降幅分别为4.99%、4.74%和7.82%。在76个分类数据集上，加权F1分别提高了2.22、2.56和3.81个百分点。（注：原摘要似乎在末尾被截断）

    arXiv:2611.31245v1 Announce Type: new  Abstract: TabPFN has already shown useful predictive performance on near-infrared spectra. RamanPFN subsequently proposed two spectral representations for tabular inference: Global Compositional Unmixing (GCU) and Local Vibrational Subspace Encoding (LVSE). We assess whether these representations also help on datasets assembled through the nirs4all ecosystem, and whether their value extends to classical calibration. We reanalyse completed experiments from a public spectral dataset catalogue. On a post-hoc subset of 61 matched numerical prediction tasks, after excluding ten class-code, metadata or semantically ambiguous targets, LVSE reduces RMSE relative to raw-spectrum TabPFN by 10.22%, 12.80% and 14.52% on average for checkpoints 2.5, 3 and 3.5, respectively, with median reductions of 4.99%, 4.74% and 7.82%. Across 76 classification datasets, weighted F1 increases by 2.22, 2.56 and 3.81 percentage points. The 2.5 classification procedure uses ou
    
[^232]: 通过循环残差解锁预训练大语言模型中循环结构的潜力

    Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals

    [https://arxiv.org/abs/2611.31094](https://arxiv.org/abs/2611.31094)

    LoopRes通过让每个循环层借助可训练的循环感知键直接关注早期循环的输出（并限制在残差窗口内），使预训练大语言模型无需从头训练即可将循环结构转化为显著且一致的增益。

    

    循环Transformer通过迭代共享的层块来增加计算量而无需增加参数，从而提供更强的性能和测试时扩展能力。然而，迄今为止，这种能力需要从头开始训练，且对大语言模型只能带来有限的收益。将现有模型微调成循环结构虽然成本低得多，但相比SFT也只能带来边际提升。我们提出LoopRes，一种有效、可扩展且简单的方法，能够将循环转化为预训练大语言模型的显著且一致的增益。LoopRes建立在注意力残差的基础上，让每个循环层能够直接关注早期循环的输出。由于这些输出是相同层迭代产生的结果，彼此越来越难以区分，我们为每个输出提供一个可训练的、感知循环的键来编码其结构身份。关注所有早期循环还允许各层跳过中间循环，因此我们将每层限制在最近W个循环组成的残差窗口内，强制每个循环进行逐步构建

    arXiv:2611.31094v1 Announce Type: new  Abstract: Looped Transformers iterate a shared block of layers to increase computation without adding parameters, offering stronger performance and test-time scaling. However, this power has so far required training from scratch and yielded only modest gains for LLMs. Finetuning existing models into loops is far cheaper, yet also brings only marginal gains over SFT. We propose LoopRes, an effective, scalable, and simple approach that turns loops into significant and consistent gains for pretrained LLMs. Building on Attention Residuals, LoopRes lets each looped layer attend directly to the outputs of earlier loops. Since these outputs are iterates of the same layers and become harder to tell apart, we give each a trainable loop-aware key that encodes its structural identity. Attending to all earlier loops also lets layers skip intermediate ones, so we restrict each layer to a residual window of the latest W loops, forcing every loop to build increm
    
[^233]: 自监督表征学习中的洛伦兹对称性

    Lorentz symmetry in self-supervised representation learning

    [https://arxiv.org/abs/2611.31011](https://arxiv.org/abs/2611.31011)

    该论文首次将洛伦兹等变的L-GATr-slim架构与LeJEPA自监督学习框架相统一，证明在潜空间中强制洛伦兹对称性（可通过架构设计或运动学增强学习实现）对喷注标记的表征质量至关重要，并在多项探测准确率上超越了现有基线。

    

    自监督学习（SSL）与显式物理归纳偏置在喷注标记任务中推动了重大进展，但两者通常沿着各自独立的路线发展。在本工作中，我们通过将洛伦兹等变的几何代数Transformer精简版（L-GATr-slim）架构与LeJEPA表征学习框架相结合，统一了这两种范式。我们在JetClass数据集的1亿个喷注上进行预训练，朝着在物理驱动的喷注增强下保持不变的潜空间进行优化。我们证明，在潜空间中强制实施洛伦兹对称性对表征质量至关重要，且这既可以通过架构设计实现，也可以通过运动学增强学习获得。在使用标准的非不变Transformer骨干网络时，LeJEPA在线性探测和MLP探测准确率上超越了已有的自回归、掩码粒子建模和自蒸馏等基线方法，而L-GATr-slim进一步带来了额外的提升。

    arXiv:2611.31011v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) and explicit physical inductive biases have driven major advances in jet tagging, yet they are typically pursued along separate tracks. In this work, we unify both paradigms by integrating the Lorentz-equivariant Geometric Algebra Transformer slim (L-GATr-slim) architecture with the LeJEPA framework for representation learning. Pre-training on 100 million jets from the JetClass dataset, we optimize towards a latent space, which is invariant across physics-motivated jet augmentations. We demonstrate that enforcing Lorentz symmetry in the latent space is critical for representation quality and that this can be achieved either architecturally or learned via kinematic augmentations. With a standard non-invariant Transformer backbone, LeJEPA outperforms established autoregressive, masked particle modelling, and self-distillation baselines in linear- and MLP-probe accuracy, with L-GATr-slim providing further co
    
[^234]: 通过瞬时轨迹分布匹配强化扩散模型

    Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching

    [https://arxiv.org/abs/2611.30986](https://arxiv.org/abs/2611.30986)

    该论文揭示现有前向扩散强化学习方法因行为分布与策略失配导致基线引起的梯度漂移和训练不稳定，并提出瞬时轨迹分布匹配（iTDM）来解决这一问题，从而高效对齐扩散模型。

    

    前向扩散强化学习（RL）——无需轨迹反向传播即可实现确定性ODE采样——在高效对齐扩散模型方面日益流行。然而，我们揭示出现有方法本质上依赖于带符号的奖励加权回归（RWR），即将可训练策略的速度回归到来自行为分布的采样轨迹上。无偏基线减除隐含地假设策略速度等于行为边际速度——这是一个理想化的假设，在无分类器引导（CFG）采样、离线/离策略更新以及不完美的在线自一致性情况下会被违反。这种行为-策略失配破坏了基线不变性，并注入了由基线引起的梯度漂移，导致训练不稳定。为解决这一问题，我们提出了瞬时轨迹分布匹配（iTDM）……

    arXiv:2611.30986v1 Announce Type: cross  Abstract: Forward diffusion reinforcement learning (RL)---enabling deterministic ODE sampling without trajectory backpropagation---has grown increasingly popular for efficiently aligning diffusion models. However, we reveal that existing methods essentially rely on signed reward-weighted regression (RWR), regressing the \emph{trainable policy} velocity onto rollouts from a \emph{behavior distribution}. Unbiased baseline subtraction implicitly assumes that the policy velocity equals the behavior marginal velocity---an idealized assumption violated under classifier-free guidance (CFG) rollouts, offline/off-policy updates, and imperfect online self-consistency. This \emph{behavior--policy mismatch} breaks baseline invariance and injects \emph{baseline-induced gradient drift}, destabilizing training. To resolve this issue, we propose \textbf{I}nstantaneous \textbf{T}rajectory \textbf{D}istribution \textbf{M}atching (\textbf{iTDM}), taking the infini
    
[^235]: LimeTree：基于树图的线性内存图神经网络，用于资源高效的语言建模与上下文外推

    LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation

    [https://arxiv.org/abs/2611.30934](https://arxiv.org/abs/2611.30934)

    LimeTree是一种内存与计算均为线性复杂度的树图语言模型架构，通过在语言建模训练中集成KV检索任务，仅用极低的计算成本（4块GPU训练6小时）便实现了19的低困惑度，并能将上下文从512外推至16,384个token且保持100%的检索准确率。

    

    近年来，寻找Transformer计算高效替代方案的研究催生了众多架构；然而，大多数架构在联想回忆（如键值检索）和长度外推方面存在严重缺陷，即使是标准Transformer在这些任务上也表现挣扎。本文提出了LimeTree，这是一种在内存和计算两方面均具有线性复杂度的序列建模架构。通过将合成的KV检索任务直接集成到标准语言建模训练中，LimeTree在实现较低困惑度的同时保持了可靠的召回能力。在严格的计算约束下，论文评估了该架构的两个变体在WikiText-103和PG-19数据集上的表现。在WikiText-103上，LimeTree仅在四块Quadro RTX 8000 GPU上训练6小时即达到19的token级困惑度，并且即使从512的训练上下文长度外推至16,384的评估长度，仍保持100%的KV检索准确率。在PG-19上，该模型达到了token级的困惑度……

    arXiv:2611.30934v1 Announce Type: new  Abstract: The search for computationally efficient alternatives to the Transformer has yielded numerous architectures in recent years; however, most exhibit severe weaknesses in associative recall-such as key-value (KV) retrieval-and length extrapolation, where even standard Transformers struggle. This paper introduces LimeTree, a linear-complexity architecture in both memory and compute for sequence modeling. By integrating synthetic KV retrieval tasks directly into standard language modeling training, LimeTree achieves strong perplexity alongside reliable recall. Two variants were evaluated on WikiText-103 and PG-19 under strict compute constraints. On WikiText-103, LimeTree reaches a token-level perplexity of 19 in just 6 hours on four Quadro RTX 8000 GPUs, while maintaining 100\% KV retrieval accuracy even when extrapolating from a training context length of 512 to an evaluation length of 16,384. On PG-19, the model achieves a token-level perp
    
[^236]: 基于Loom注意力实现Transformer中的可动态重连电路

    Rewirable Circuits in Transformers with Loom Attention

    [https://arxiv.org/abs/2611.30634](https://arxiv.org/abs/2611.30634)

    提出loom attention使Transformer能在每个token处动态重连其注意力电路，通过垂直注意力与矩阵记忆构成的向量-矩阵双流架构，在仅增加约2%计算量的情况下达到超过3倍计算量训练的Transformer的性能。

    

    我们提出了loom attention（织机注意力），它使Transformer能够在每个token处动态地重新连接其电路，方法是将注意力头对先前注意力头输出的“垂直注意力”（用于获取其query、key和value）与对token序列的“水平注意力”交织在一起。我们通过在每token矩阵记忆上使用归一化且门控的线性注意力来高效实现垂直注意力，该矩阵记忆存储压缩后的注意力头输出，并充当第二条残差流。由此我们构建了LoomFormer，一种双流架构，将控制信息与内容信息分别分配到向量流和矩阵流中。在语言建模和下游任务中，跨越不同模型规模和训练token预算，LoomFormer能够匹敌使用超过3倍计算量训练的标准Transformer，以及使用约1.7-2.8倍计算量训练的最新残差流基线（HC/mHC、MUDDFormer、AttnRes）。例如，在相同的300B token上训练且仅增加2%的额外FLOPs的情况下，LoomPythia-2.8B实现了约1.6倍的平均下游性能（原文此处截断）。

    arXiv:2611.30634v1 Announce Type: new  Abstract: We propose loom attention, which allows a Transformer to dynamically rewire its circuits at every token by interweaving a head's vertical attention over the outputs of earlier heads (to obtain its query, key and value) with its horizontal attention over tokens. We instantiate vertical attention efficiently with normalized and gated linear attention over a per-token matrix memory, which stores the compressed head outputs and serves as a second residual stream. This yields LoomFormer, a dual-stream architecture that splits control and content between the vector and matrix streams. Across model sizes and training token budgets in language modeling and downstream tasks, LoomFormer matches Transformers trained with over 3x compute and recent residual-stream baselines (HC/mHC, MUDDFormer, AttnRes) trained with ~1.7-2.8x. For example, trained on the same 300B tokens with only 2% extra FLOPs, LoomPythia-2.8B achieves ~1.6x the average downstream
    
[^237]: 用于图生成的Vasicek扩散桥

    Vasicek Diffusion Bridge for Graph Generation

    [https://arxiv.org/abs/2611.30605](https://arxiv.org/abs/2611.30605)

    本文提出基于Vasicek动力学的图扩散模型VGD，利用具有闭式解的数据锚定均值动力学，从解析上建立高斯噪声到目标图分布的连续过渡，解决了现有图扩散模型不确定性过大、信噪比增长滞后导致结构信息过晚出现的问题。

    

    基于扩散的图生成模型通过随机动力学将高斯噪声逐步转化为结构化的图。我们观察到，这种转化的有效性从根本上取决于扩散轨迹的演化。现有的图扩散模型往往在扩散过程中存在过大的不确定性，且信噪比增长滞后，导致有用的结构信息仅在生成的后期阶段才显现。为了解决这一问题，我们提出了Vasicek图扩散（VGD），这是一种源自Vasicek动力学的图扩散过程。VGD的关键洞察在于具有闭式解的数据锚定均值动力学，它从解析层面建立了从高斯噪声到目标图分布的连续过渡。与主要依赖随机演化的传统图扩散过程不同，VGD引入了一个数据锚定的漂移项，该漂移项……

    arXiv:2611.30605v1 Announce Type: new  Abstract: Diffusion-based graph generation models progressively transform Gaussian noise into structured graphs through stochastic dynamics. We observe that the effectiveness of this transformation is fundamentally determined by the evolution of the diffusion trajectory. Existing graph diffusion models often suffer from excessive uncertainty during diffusion and delayed growth of the signal-to-noise ratio, causing useful structural information to emerge only in the later stages of generation. To address this issue, we propose the Vasicek Graph Diffusion (VGD), a graph diffusion process derived from the Vasicek dynamics. The key insight of VGD is data-anchored mean dynamics with a closed-form solution, which analytically establishes a continuous transition from Gaussian noise to the target graph distribution. Unlike conventional graph diffusion processes that rely predominantly on stochastic evolution, VGD introduces a data-anchored drift that cont
    
[^238]: 具有靶向嗅觉受体激活模式的气味分子从头生成

    De Novo Generation of Odorant Molecules with Targeted Olfactory Receptor Activation Patterns

    [https://arxiv.org/abs/2611.30509](https://arxiv.org/abs/2611.30509)

    该论文提出了一个基于强化学习的气味分子从头生成框架，通过构建嗅觉受体全景结合矩阵、聚类发现代表性激活模式，并采用结合成对排序、分布匹配与复杂度约束的结构化选择性损失，成功生成匹配目标多受体激活谱的分子，在广谱型和稀疏激活型等罕见目标谱上取得了最大提升。

    

    气味感知源于一种组合编码机制：每个分子会激活特定的嗅觉受体（OR）子集，由此产生的激活模式决定了所感知到的气味。我们提出了一个从头生成分子的框架，用于生成预测上匹配目标多受体激活谱的分子。我们首先构建了覆盖20个嗅觉受体的全景结合矩阵，然后对其进行聚类以发现代表性的激活模式，最后设计了一种结构化选择性损失函数，结合成对排序、分布匹配和复杂度约束，通过强化学习来引导分子生成器。在九个不同复杂度的目标谱上，我们的方法在18000次oracle调用内实现了较高的预测选择性，并且在先验分布下较为罕见的目标谱上获得了最大的提升，尤其是广谱型和稀疏激活型目标。

    arXiv:2611.30509v1 Announce Type: cross  Abstract: Odor perception arises from a combinatorial code in which each molecule activates a specific subset of olfactory receptors (ORs), and the resulting activation pattern determines the perceived smell. We present a framework for de novo generation of molecules predicted to match target multi-receptor activation profiles. We first construct a panoramic binding matrix over 20 ORs, then cluster it to discover representative activation patterns, and finally formulate a structured selectivity loss, combining pairwise ranking, distribution matching, and complexity constraints, to steer a molecular generator via reinforcement learning. Across nine target profiles of varying complexity, our method achieves high predicted selectivity within 18000 oracle calls and provides the largest gains on profiles that are rare under the prior, especially broad-spectrum and sparse-activator targets.
    
[^239]: 稀疏霍奇偏好优化：微调语言模型以实现对超越传递性偏好的对齐

    Sparse Hodge Preference Optimization: Fine-Tuning Language Models toward Alignment beyond Transitive Preferences

    [https://arxiv.org/abs/2611.30135](https://arxiv.org/abs/2611.30135)

    本文提出稀疏霍奇偏好优化（SHPO），借助组合霍奇分解理论为DPO引入稀疏反对称旋度残差，使其能够捕捉人类偏好数据中的非传递性与循环结构，并证明其严格扩大了DPO的假设类。

    

    直接偏好优化的广泛应用，反映了其在利用成对偏好数据微调大型语言模型以符合人类期望方面的巨大成功。它直接从人类偏好中优化策略，无需显式的奖励建模，也避免了基于强化学习方法所带来的复杂性。然而，经典的DPO依赖于Bradley-Terry采样模型，无法有效捕捉聚合的人类偏好数据中的非传递性或循环结构。在本文中，我们提出了稀疏霍奇偏好优化，这是一种偏好优化算法，通过受组合霍奇分解启发的稀疏反对称旋度残差来增强DPO。我们证明了在响应表示满足一些温和正则性条件的情形下，SHPO在包含至少一个三角形的比较图上严格扩大了DPO的假设类。我们进一步……（原文摘要在此处截断）

    arXiv:2611.30135v1 Announce Type: cross  Abstract: The growing adoption of Direct Preference Optimization (DPO), which leverages pairwise preference data, reflects its great success in fine-tuning large language models to align with human expectations. It directly optimizes the policy from human preferences without explicit reward modeling or the complexities of reinforcement learning-based approaches. However, the classic DPO relies on the Bradley--Terry sampling model and cannot effectively capture intransitivity or cyclic structures in aggregated human preference data. In this article, we propose Sparse Hodge Preference Optimization (SHPO), a preference optimization algorithm that augments DPO with a sparse antisymmetric curl residual motivated by the combinatorial Hodge decomposition. We prove that SHPO strictly enlarges DPO's hypothesis class on comparison graphs containing at least one triangle, under some mild regularity conditions on the response representations. We further est
    
[^240]: 基于嵌套端口哈密顿学习的版图多阶无源宏建模

    Multi-Order Passive Macromodeling from Layouts via Nested Port-Hamiltonian Learning

    [https://arxiv.org/abs/2611.28778](https://arxiv.org/abs/2611.28778)

    该论文提出了一种直接在无源空间中学习的端到端宏建模框架，利用前缀嵌套的端口哈密顿参数化，从物理版图图一次性生成在构造上即保证无源性与稳定性的多阶宏模型，从而免去了逐版图昂贵且耗时的全波电磁仿真。

    

    互连与电源分配网络的紧凑无源状态空间宏模型对于稳定的瞬态信号完整性与电源完整性仿真至关重要。然而，传统流程需要对每个版图进行代价高昂的全波电磁（EM）仿真，而传统的拟合方法和已有的学习代理模型都依赖于事后拟合或无源性扰动，这不仅会降低精度，还无法保证收敛。在本工作中，我们提出了一种端到端的宏建模框架，直接在无源空间中进行学习，将物理版图图映射为多阶端口哈密顿宏模型。具体而言，我们的网络将每个版图图编码为无约束的矩阵块，并通过完整的前缀嵌套pH参数化将其组装起来，该参数化在构造上即保证了连续频率下的无源性与稳定性，通过一次前向传播中的直接状态切片即可生成有效的k状态宏模型。为训练所有阶……（原文摘要在此处截断）

    arXiv:2611.28778v1 Announce Type: cross  Abstract: Compact, passive state-space macromodels of interconnects and power-delivery networks are essential for stable transient signal- and power-integrity simulations. However, conventional flows require costly per-layout full-wave electromagnetic (EM) simulations, while both traditional fitting and prior learning surrogates rely on post-hoc fitting or passivity perturbation that degrades accuracy without guaranteeing convergence. In this work, we propose an end-to-end macromodeling framework that learns directly in the passive space, mapping physical layout graphs to multi-order port-Hamiltonian (pH) macromodels. Specifically, our network encodes each layout graph into unconstrained matrix blocks and assembles them via a complete, prefix-nested pH parameterization that guarantees continuous-frequency passivity and stability by construction, yielding valid $k$-state macromodels via direct state slicing in one forward pass. To train all order
    
[^241]: 预言机高效的可实现转换式学习的复杂度

    The Complexity of Oracle-Efficient Realizable Transductive Learning

    [https://arxiv.org/abs/2611.28757](https://arxiv.org/abs/2611.28757)

    本文证明即使借助能优化任意线性目标的加权 ERM 预言机，可实现转换式在线学习也无法通过多项式次预言机查询达到减半算法的对数级错误保证，并给出了与不可知率上界匹配的 $\Omega_\beta(\sqrt{T\log N})$ 期望错误下界。

    

    在可实现的转换式在线学习中，学习器预先知道整个长度为 $T$ 的实例序列，且标签与 $N$ 个假设中的某一个保持一致。在可以显式访问假设类的情况下，减半算法最多只会犯 $\log_2 N$ 次错误。那么，仅通过多项式次数的经验风险最小化（ERM）查询能否获得可比的保证？我们对这一问题给出了否定回答，即使是针对加权 ERM 预言机——它能够优化任意实值线性目标并返回一个完整的最大化假设——情况也是如此。对于任意固定的 $\beta\in(0,1)$ 以及任何只进行多项式次预言机查询的随机学习器，都存在一个固定的可实现实例，该实例恰好含有 $N=2^{\lfloor T^\beta\rfloor}$ 个假设，在该实例上学习器期望犯 $\Omega_\beta(\sqrt{T\log N})$ 次错误。这一下界与通过 $O(T)$ 次预言机调用进行随机演练所达到的不可知率上界相匹配。这种分离持续存在……（摘要原文在此处截断）

    arXiv:2611.28757v1 Announce Type: new  Abstract: In realizable transductive online learning, the learner knows the entire sequence of $T$ instances in advance, and the labels are consistent with one of $N$ hypotheses. With explicit access to the hypothesis class, the halving algorithm makes at most $\log_2 N$ mistakes. Can a comparable guarantee be achieved using only polynomially many empirical risk minimization (ERM) queries? We answer this question negatively, even for a weighted-ERM oracle that optimizes arbitrary real-valued linear objectives and returns a complete maximizing hypothesis. For every fixed $\beta\in(0,1)$ and every randomized learner making polynomially many oracle queries, there exists a fixed realizable instance with exactly $N=2^{\lfloor T^\beta\rfloor}$ hypotheses on which the learner makes $\Omega_\beta(\sqrt{T\log N})$ mistakes in expectation. This matches the agnostic-rate upper bound achieved by random playout using $O(T)$ oracle calls. The separation persist
    
[^242]: 用自然语言解释扩散模型激活

    Interpreting Diffusion Model Activations in Natural Language

    [https://arxiv.org/abs/2611.28621](https://arxiv.org/abs/2611.28621)

    本文提出扩散专用激活预言机（DAO），能将扩散模型中间激活翻译为自然语言回答开放式问题，既可从早期去噪步骤预测最终图像内容，又能利用激活空间结构引导生成目标属性并定位关键模型组件。

    

    解释扩散模型（DM）的中间激活是理解其图像生成机制的关键。现有的扩散模型可解释性方法对开放式提问的支持有限，难以提取人类可理解的洞见。我们将LatentQA——回答关于大语言模型激活的自然语言问题的任务——扩展到了扩散模型激活领域。我们提出了扩散专用激活预言机，它将扩散模型的激活转换为自然语言答案，用以回答关于正在进行中的生成过程的开放式问题。实验表明，DAO能够从早期去噪步骤的激活中有效描述最终图像的内容。此外，我们证明DAO能够充分捕捉扩散模型激活空间的结构，足以引导图像生成朝向目标属性。进一步地，我们展示了通过DAO进行归因修补可以识别扩散模型中……（摘要在此处截断）

    arXiv:2611.28621v1 Announce Type: new  Abstract: Interpreting the intermediate activations of diffusion models (DMs) is key to understanding how they generate images. Existing interpretability methods for DMs provide limited support for open-ended interrogation, making it difficult to extract human-interpretable insights. We extend LatentQA, the task of answering natural-language questions about Large Language Model activations, to DM activations. We introduce the Diffusion-specific Activation Oracle (DAO), which translates DM activations into natural-language answers to open-ended questions about the ongoing generation. Our experiments show that DAO can effectively describe the content of the final image from activations at early denoising steps. Additionally, we demonstrate that DAO captures the structure of the DM's activation space well enough to steer image generation toward target attributes. Furthermore, we show that attribution patching through DAO identifies DM components that
    
[^243]: 基于异步事件的自监督连续时间深度学习与自运动估计

    Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events

    [https://arxiv.org/abs/2611.27509](https://arxiv.org/abs/2611.27509)

    该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。

    

    事件相机以微秒级精度逐像素地记录亮度变化，而可用于监督它们的稠密深度数据获取成本高昂，且仅存在于孤立的时刻。将事件流压缩成帧虽然便于学习，但代价是牺牲了事件感知赖以独特的高时间分辨率特性。我们提出了一种自监督方法，其中每个事件在连续时间中被赋予其自身的深度，同时相机运动也从相同的观测中恢复出来。监督信号源于几何：一个事件通过其预测深度被提升到三维空间，并经过它在触发瞬间的相机运动位移后，应当与由同一条边缘产生的其他事件相重合，因此任何错位都会暴露深度或运动估计中的误差。随后，可微渲染将这些逐事件的估计转化为每个窗口内目标时刻的稠密深度图，并忠实于其背后的事件。在MVSEC数据集上，该方法优于先前仅基于事件的自监督方法。

    arXiv:2611.27509v1 Announce Type: cross  Abstract: Event cameras register brightness changes pixel by pixel with microsecond precision, whereas the dense depth that could supervise them is costly to acquire and exists only at isolated instants. Condensing the stream into frames eases learning, but at the expense of the very timing that sets event sensing apart. We propose a self-supervised approach in which each event is assigned its own depth in continuous time, and the camera motion is recovered from the same observations. Supervision arises from geometry: lifted into 3D by its predicted depth and displaced by the camera motion at the instant it fired, an event should coincide with others produced by the same edge, so any misalignment exposes an error in depth or motion. Differentiable rendering then turns these per-event estimates into dense depth maps at target times within each window, faithful to the events behind them. On MVSEC, the approach outperforms prior event-only self-sup
    
[^244]: RL专家的可组合软提示蒸馏：迈向持续后训练

    Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training

    [https://arxiv.org/abs/2611.27376](https://arxiv.org/abs/2611.27376)

    提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。

    

    RL后训练通常会产生领域专家模型，这些专家随后必须被整合到一个模型中，而随着新专家不断出现，这种整合就成为了一个持续的过程。权重合并和多教师在线策略蒸馏（MOPD）在每添加一个专家时都必须重新整合整个模型，并且可能损害目标领域之外的能力。我们提出了可组合软提示蒸馏（CSPD），该方法保持RL之前的模型冻结，将每个RL专家独立地蒸馏到一个软提示中，并通过直接拼接各提示来组合专家，无需联合训练。单个软提示就足以蒸馏一个RL专家，在23个数学、代码和智能体设置中的22个里恢复了至少90%的教师增益。独立训练的提示之间可以良好组合。在多专家设置中，平均性能随着每个专家的加入而提升，完整组合的性能与MOPD相当（67.83 vs. 67.79）。该组合在六个……（摘要截断）

    arXiv:2611.27376v1 Announce Type: cross  Abstract: RL post-training typically produces domain experts that must then be consolidated into one model, and as new experts keep arriving, this consolidation becomes a continual process. Weight merging and multi-teacher on-policy distillation (MOPD) must re-consolidate the whole model whenever an expert is added, and can erode capabilities outside the target domains. We propose Composable Soft-Prompt Distillation (CSPD), which keeps the pre-RL model frozen, distills each RL expert independently into a soft prompt, and composes experts by concatenating their prompts without joint training. A single soft prompt suffices to distill an RL expert, recovering at least 90% of the teacher's gain in 22 of 23 math, code, and agent settings. Independently trained prompts compose well. In a multi-expert setting, the average rises with every expert added, and the full composition matches MOPD (67.83 vs. 67.79). The composition generalizes best across six 
    
[^245]: Loop-OPF：面向最优潮流的求解器引导学习与推理时拓扑自适应性

    Loop-OPF: Solver-Informed Learning for Optimal Power Flow with Inference-Time Topology Adaptivity

    [https://arxiv.org/abs/2611.27013](https://arxiv.org/abs/2611.27013)

    Loop-OPF是一种由PDHG迭代构建的求解器引导神经网络，在凸性条件下保证DC-OPF预测的最优性与可行性，并通过自监督方式学习PDHG参数实现无需解标签的加速训练与对推理时未见拓扑的自适应。

    

    近年来，基于学习的最优潮流（OPF）方法被提倡用于加速OPF求解，即通过预训练神经网络（NN）直接根据给定输入（如负荷需求）进行预测。然而，现有的OPF学习方法在很大程度上无法保证其预测的最优性与可行性，并且在推理时难以适应未见过的拓扑场景，例如由线路或发电机停运引起的场景。本工作提出了Loop-OPF，这是一种求解器引导的神经网络，其网络层由原始-对偶混合梯度（PDHG）方法的迭代构建而成。在DC-OPF的标准凸性与正则性条件下，Loop-OPF所得的不动点输出因此是最优且可行的。为了在各种OPF场景下加速PDHG求解，Loop-OPF以自监督的方式学习某些PDHG参数，而无需任何OPF解标签。我们采用深度平衡（DEQ）框架进行训练，从……

    arXiv:2611.27013v1 Announce Type: cross  Abstract: Learning for optimal power flow (OPF) has recently been advocated to accelerate OPF solutions by using a pre-trained neural network (NN) to predict directly from given inputs such as the load demand. However, existing learning-for-OPF methods largely fall short of ensuring the optimality and feasibility of their predictions. They also struggle to adapt at inference time to unseen topology scenarios, such as those caused by line or generator outages. This work puts forth Loop-OPF, a solver-informed NN that builds its layers from the iterations of the primal-dual hybrid gradient (PDHG) method. Under standard convexity and regularity conditions on DC-OPF, the resultant fixed-point output of Loop-OPF is thus optimal and feasible. To speed up PDHG across diverse OPF scenarios, Loop-OPF learns certain PDHG parameters in a self-supervised manner, without any OPF solution labels. We adopt the deep equilibrium (DEQ) framework for training so th
    
[^246]: 利用退相位实现最小复杂度架构的量子储备池计算：在非线性时间序列预测中的应用

    Harnessing Dephasing for Quantum Reservoir Computing with a Minimum Complexity Architecture: Applications to Nonlinear Time-Series Prediction

    [https://arxiv.org/abs/2611.26762](https://arxiv.org/abs/2611.26762)

    提出了一种最小复杂度的量子储备池计算模型，证明纯退相位在开放均匀XX链中可保证回声态与衰减记忆特性，且联合控制退相位与振幅阻尼可调节记忆与非线性能力，成功应用于非线性时间序列预测。

    

    量子储备池计算（QRC）利用量子动力学进行时间信息处理。量子噪声会影响储备池动力学，它既可能损害计算性能，也可能作为一种计算资源。然而，纯退相位在哈密顿量编码的QRC中的作用仍不清楚。在这项工作中，我们提出了一个最小复杂度的QRC模型，该模型在开放的均匀XX链中利用纯退相位。我们证明，在适当的输入编码下，即使没有振幅阻尼，纯退相位也能确保回声态和衰减记忆特性。此外，在存在振幅阻尼的情况下，所提出的量子储备池对不同输入具有不同的稳态，而引入纯退相位会改变这些稳态，从而影响储备池动力学。信息处理能力分析表明，对这两种退相干机制的联合控制可以调节记忆与非线性能力，提供了可…

    arXiv:2611.26762v1 Announce Type: cross  Abstract: Quantum reservoir computing (QRC) exploits quantum dynamics for temporal information processing. Quantum noise affects the reservoir dynamics and can either impair computational performance or serve as a computational resource. However, the role of pure dephasing in Hamiltonian encoding QRC remains unclear. In this work, we propose a minimum complexity QRC model that harnesses pure dephasing in an open uniform XX chain. We show that, under suitable input encoding, pure dephasing can ensure the echo state and fading memory properties even without amplitude damping. Furthermore, in the presence of amplitude damping, the proposed quantum reservoir has distinct stationary states for distinct inputs, and introducing pure dephasing alters these states, thereby affecting the reservoir dynamics. Information processing capacity analysis shows that joint control of the two decoherence mechanisms regulates memory and nonlinearity, providing acces
    
[^247]: Bradley-Terry训练可能扭曲原本准确的语言世界模型的结果概率

    Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models

    [https://arxiv.org/abs/2611.25594](https://arxiv.org/abs/2611.25594)

    本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。

    

    arXiv:2611.25594v1 公告类型：交叉 摘要：比较反馈（即由评判者指出两个候选结果中哪一个更接近实际发生的情况）是训练为动作结果分配概率的语言世界模型的一种实用方法。我们证明，使用Bradley-Terry目标函数进行拟合时，它未必能教会模型真实的结果概率，甚至可能扭曲原本准确的结果概率。在一个具有枚举结果分布的模拟机器人推动任务中，该目标函数会使一个在未见布局上表现准确的模型（危害误差为2.4个百分点）将33至43个百分点的概率转移到该条件下没有任何枚举分支会产生的结果上。问题的根源在于目标函数本身，而非比较数据：在训练前计算的目标函数最优解可以预测模型最终的去向，而一个了解评判者规则的似然函数能够从相同的比较数据中学习且不产生这种偏移。这种偏移具有鲁棒性：它在未见条件、五种候选法则以及三倍更密集的枚举下均会重现。

    arXiv:2611.25594v1 Announce Type: cross  Abstract: Comparative feedback, in which a judge says which of two candidate outcomes is closer to what happened, is a practical way to train language world models that assign probabilities to an action's outcomes. We show that, fitted with a Bradley-Terry objective, it need not teach the true outcome probabilities and can distort accurate ones. On a simulated robot push with enumerated outcome distributions, this objective takes a model that is accurate on unseen layouts (hazard error 2.4 percentage points, pp) and moves 33 to 43 pp of its probability onto outcomes that none of a condition's enumerated branches produces. The cause is the objective, not the comparisons: its optimum, computed before training, predicts where the model ends up, and a likelihood that knows the judge's rule learns from the same comparisons without the shift. The shift is robust: it recurs on unseen conditions, under five candidate laws and a three-fold denser enumera
    
[^248]: MADJEPA：用于医学异常检测的增强型多阶段联合嵌入预测架构

    MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection

    [https://arxiv.org/abs/2611.25194](https://arxiv.org/abs/2611.25194)

    该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。

    

    医学图像中的异常检测是一项复杂的任务，需要仅使用健康患者的数据来开发一个鲁棒的模型。在无法获得标注数据的情况下，现代方法通常采用自监督学习来获得正常性的模型。由于世界模型有望为其环境构建强大的内部表征，我们认为它们是医学异常检测环境（即对健康患者的医学检查）的合适候选。为此，我们提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构（JEPA）的医学异常检测器。我们设计了一个多阶段训练流程来构建鲁棒的检测器：（1）基于图像的JEPA（I-JEPA）训练旨在学习正常训练数据的强表征；（2）目标编码器蒸馏旨在调节I-JEPA表征对异常数据的过度泛化；（3）伪异常分类旨在……（原文在此处截断）

    arXiv:2611.25194v1 Announce Type: cross  Abstract: Anomaly detection in medical images is a complex task, requiring the development of a robust model using only healthy patient data. Without access to labeled data, modern approaches typically employ self-supervised learning to obtain a model of normality. Since world models promise to build strong internal representations of their environment, we consider them as suitable candidates for the medical anomaly detection environment, namely medical investigations of healthy patients. To this end, we propose MADJEPA, a medical anomaly detector based on an enhanced multi-stage joint-embedding predictive architecture (JEPA). We design a multi-stage training pipeline to produce a robust detector: (1) image-based JEPA (I-JEPA) training aims to learn a strong representation of normal training data; (2) target encoder distillation aims to modulate over-generalization of I-JEPA representations to abnormal data; (3) pseudo-anomaly classification aim
    
[^249]: 可以安全省略，也值得省略吗？当LLM智能体跳过另一方的消息时，保持答案不变并降低延迟

    Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message

    [https://arxiv.org/abs/2611.23570](https://arxiv.org/abs/2611.23570)

    该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。

    

    在多智能体系统中，接收方智能体通常需要等待辅助智能体的消息（文本或其潜在KV缓存），这可以提升性能但会增加延迟。我们研究接收方智能体何时可以安全地（最终答案很少改变）且有益地（节省的时间超过决策本身的开销）跳过该消息。我们的机制在不读取该消息的情况下获取接收方智能体的置信度，并使用有限样本检验来认证一个阈值——在该阈值下，以高概率保证被跳过时答案的改变率不超过5%；如果没有阈值能通过检验，系统则继续保持通信。我们在OpenBookQA、ARC-Challenge和MMLU-Pro上使用三对Qwen模型进行测试，也在跨家族的智能体组合以及SQuAD和GSM8K上进行了测试。我们得到四点洞察。首先，“oracle上限空间”是消息有帮助作用的弱证据：不含答案特定内容、但被匹配为与真实消息同样频繁改变答案的控制输入，能够重现oracle收益的中位数76%。其次，一条消息能否……（摘要原文在此处截断）

    arXiv:2611.23570v1 Announce Type: cross  Abstract: In multi-agent systems, a receiver agent often waits for a helper agent's message (text or its latent KV cache), which can improve performance but adds latency. We study when a receiver agent can skip the message safely (the final answer rarely changes) and beneficially (the time saved exceeds the cost of deciding). Our mechanism reads the receiver agent's confidence without the message and uses a finite-sample test to certify a threshold at which, with high probability, at most 5% of skipped answers change; if none passes, the system keeps communicating. We test it on OpenBookQA, ARC-Challenge, and MMLU-Pro with three Qwen model pairs, on cross-family agents, and on SQuAD and GSM8K. Four insights emerge. First, oracle headroom is weak evidence that messages help: control inputs without answer-specific content, matched to change answers as often as real messages, reproduce a median 76% of the oracle's gain. Second, whether a message ca
    
[^250]: 用于建模受损驾驶的多模态掩码扩散自编码器

    Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving

    [https://arxiv.org/abs/2611.23211](https://arxiv.org/abs/2611.23211)

    该论文提出多模态掩码扩散自编码器 ID-MAE，通过自监督方式重建被掩码的注视与轨迹数据来增强受损驾驶预测，并发布了包含 109 人、103 小时驾驶数据及十种受损诱发条件的大规模模拟驾驶数据集 DIVERT，显著提升了受损驾驶推断效果。

    

    人类认知状态（如分心、压力）、注视与行为会以微妙的方式与驾驶环境相互作用，AI 可以利用这些信息来辅助驾驶员，但大规模获取人类认知状态水平的真实标注成本高昂。我们提出了一种多模态掩码扩散自编码器 ID-MAE，它在受损驾驶预测中引入了自监督目标：除了将受损标签作为掩码 token 进行预测外，模型还重建被掩码的注视数据和车辆轨迹。为了探究不同模态和不同类型受损状态对模型推理的影响，我们引入了 DIVERT——一个驾驶模拟器数据集，包含 109 名受试者的 103 小时驾驶数据，涵盖认知、视觉和情绪三类状态下的十种受损诱发方式，并以注视、车辆状态和场景视频的形式进行记录。我们展示了 ID-MAE 在 DIVERT 数据集和现有数据集上均能显著提升受损驾驶推断的效果……

    arXiv:2611.23211v1 Announce Type: cross  Abstract: Human cognitive state (e.g., distraction, stress), gaze, and behavior interact with the driving environment in subtle ways that AI could use to assist drivers, but ground-truth labels for human cognitive state levels are costly to obtain at scale. We propose a multimodal masked-diffusion autoencoder, ID-MAE, that adds a self-supervised objective to impairment prediction: alongside predicting the impairment label as a masked token, it reconstructs masked gaze and trajectory. To probe the effects of different modalities and types of impairments on model inference, we introduce DIVERT, a driving-simulator dataset containing 103 hours of driving data (n = 109) with ten impairment inductions spanning cognitive, visual, and emotional states, recorded with gaze, vehicle state, and scene video. We demonstrate how ID-MAE affords significant improvements in impairment inference results on both the DIVERT dataset and existing datasets, and enable
    
[^251]: LoopPrune：面向循环扩散Transformer的循环剪枝方法

    LoopPrune: Recurrent Pruning for Looped Diffusion Transformers

    [https://arxiv.org/abs/2611.22757](https://arxiv.org/abs/2611.22757)

    LoopPrune提出了一种针对循环扩散Transformer的剪枝与恢复框架，通过循环重要性估计、共享重建和循环位置恢复三大机制，在降低存储与计算成本的同时有效保持模型性能。

    

    循环扩散Transformer通过复用共享模块来深化计算，以更少的参数达到大模型的效果，从而降低存储与部署负担。然而，重复的注意力与MLP计算仍然代价高昂。剪枝通过移除冗余参数或单元来减少存储和计算开销。免训练方法难以解析循环贡献，而基于训练的方法缺乏对影响后续输入与最终输出的特定位置误差的显式约束。我们提出LoopPrune，一个满足上述需求的剪枝与恢复框架。循环重要性估计结合共享移除效应与位置变化来选择共享结构；共享重建汇集循环输入统计量以补偿权重。循环位置恢复在相同的学生状态下比较完整的去噪预测并匹配文本引导，在固定的稠密教师模型监督下更新所有学生参数。

    arXiv:2611.22757v1 Announce Type: new  Abstract: Looped diffusion Transformers reuse shared blocks to deepen computation and match larger models with fewer parameters, reducing storage and deployment burdens. Yet repeated attention and MLP computations remain costly. Pruning reduces storage and computation by removing redundant parameters or units. Training-free methods struggle to resolve recurrent contributions, while training-based methods lack explicit constraints on position-specific errors affecting later inputs and final outputs. We propose LoopPrune, a pruning and recovery framework addressing these requirements. Recurrent importance estimation combines shared-removal effects and position variation to select a shared structure; shared reconstruction pools recurrent input statistics to compensate weights. Loop-position recovery compares complete denoising predictions at identical student states and matches text guidance, updating all student parameters against a fixed dense teac
    
[^252]: LoopZip：面向高效循环视觉语言模型的视觉标记压缩

    LoopZip: Visual Token Compression for Efficient Looped Vision-Language Models

    [https://arxiv.org/abs/2611.22728](https://arxiv.org/abs/2611.22728)

    LoopZip是一种无需训练的视觉标记压缩算法，通过在循环主干前构建紧凑候选池并结合跨调用选择，降低循环视觉语言模型处理长视觉序列时的计算与GPU显存开销。

    

    循环视觉语言模型（VLMs）通过重用共享层来加深计算，以更少的参数逼近更大模型的能力，并减轻堆叠独立层带来的存储和部署负担。然而，反复处理长视觉序列仍会产生大量的计算和GPU显存开销。视觉标记压缩通过缩短序列来降低这些开销。现有方法主要是直接选择原始标记，或将选择与聚合相结合。在循环模型中，前者可能忽视各次调用间标记重要性的变化，从而丢弃后续所需的信息；后者则缺乏对各次调用间输出变化的联合考虑，限制了信息的保留。我们提出LoopZip，一种无需训练的视觉标记压缩算法，同时解决上述两个问题。视觉聚合在循环主干之前构建一个紧凑的候选池，跨调用选择则对各次调用的指令注意力进行平均（摘要在此处截断，内容不完整）。

    arXiv:2611.22728v1 Announce Type: cross  Abstract: Looped vision-language models (VLMs) deepen computation by reusing shared layers, approaching larger models' capabilities with fewer parameters and easing the storage and deployment burden of stacking independent layers. However, repeatedly processing long visual sequences still incurs substantial compute and GPU memory costs. Visual token compression reduces these costs by shortening sequences. Existing methods mainly select original tokens directly or combine selection with aggregation. In looped models, the former may overlook changing token importance across calls and discard information needed later; the latter lack joint consideration of output changes across calls, limiting information preservation. We propose LoopZip, a training-free visual token compression algorithm addressing both issues. Visual aggregation builds a compact candidate pool before the looped backbone, and cross-invocation selection averages instruction attenti
    
[^253]: 决策模型何时应该比较候选对象？

    When Should Decision Models Compare Candidates?

    [https://arxiv.org/abs/2611.20335](https://arxiv.org/abs/2611.20335)

    该论文系统比较了逐点打分、集合打分和候选对象注意力三种候选评分方式，发现逐点打分总体仍有竞争力，集合打分在特定任务上更优且经关系训练后能学习候选间依赖并泛化，而候选对象注意力即使允许跨候选交互也未带来稳定提升。

    

    语言模型智能体经常需要在候选动作（如工具和网页元素）之间进行选择。决策模型应该独立地为候选对象打分，还是允许其分数依赖于其他候选对象？我们通过比较三种方法来研究这一问题：逐点打分、集合打分和候选对象注意力。在五个智能体决策基准上，逐点打分依然具有竞争力，而集合打分在某些工具选择和二元决策任务上表现更好。尽管候选对象注意力允许跨候选对象的交互，但它并未持续超越逐点打分。借助两个受控的合成任务，我们进一步考察了模型是否学会利用候选集信息。通过有针对性的关系训练，集合打分模型能够学习候选对象之间的依赖关系并泛化到分布偏移的输入，而候选对象注意力的迁移能力较弱。我们的结果表明，能够访问其他候选对象并不保证模型能有效利用候选集信息。

    arXiv:2611.20335v1 Announce Type: new  Abstract: Language-model agents frequently select among candidate actions, such as tools and web elements. Should decision models score candidates independently or allow their scores to depend on other candidates? We investigate this question by comparing three approaches: Pointwise scoring, Setwise scoring, and Candidate Attention. Across five agent-decision benchmarks, Pointwise scoring remains competitive, while Setwise scoring performs better on some tool-selection and binary-decision tasks. Candidate Attention does not consistently improve upon Pointwise scoring, despite allowing cross-candidate interaction. Using two controlled synthetic tasks, we further examine whether models learn to use candidate-set information. With targeted relational training, Setwise models learn candidate dependencies and generalize to shifted inputs, while Candidate Attention shows weaker transfer. Our results show that access to other candidates does not guarante
    
[^254]: 在策略蒸馏作为后门传播通道

    On-policy distillation as a Backdoor Propagation Channel

    [https://arxiv.org/abs/2611.19207](https://arxiv.org/abs/2611.19207)

    本研究首次揭示在策略蒸馏可成为后门传播的高效通道：教师模型中被植入的后门仅需极少触发提示暴露即可转移至学生模型，且转移强度显著高于传统蒸馏方法。

    

    在策略蒸馏正在成为大语言模型后训练的重要范式，其通过利用教师模型在线生成的监督信号来改进学生模型。然而，这一过程是否也会将教师模型中的潜在安全风险转移给学生模型，目前尚缺乏充分研究。在这项工作中，我们揭示了一种新的安全风险：攻击者首先通过监督微调（SFT）在教师模型中植入条件后门，随后在策略蒸馏仅需极少比例的触发提示暴露，即可将后门转移到学生模型中。更令人担忧的是，我们发现这种后门转移在在策略蒸馏下明显强于传统蒸馏方法，这表明在策略蒸馏可能为跨模型传播后门提供了一条高效通道。我们的机制分析表明，后门信号主要通过……（原文摘要至此截断）

    arXiv:2611.19207v1 Announce Type: cross  Abstract: On-policy distillation (OPD) is emerging as an important paradigm for the post training of large language models, improving student models by leveraging supervision generated online by teacher models. However, whether this process also transfers latent security risks from teacher models to student models remains insufficiently studied. In this work, we reveal a new security risk in which an attacker first implants a conditional backdoor into the teacher model through supervised fine tuning (SFT), after which OPD can transfer the backdoor to the student model with only an extremely small proportion of trigger prompt exposure. More concerningly, we find that such backdoor transfer is substantially stronger under OPD than under conventional distillation methods, suggesting that OPD may provide an efficient channel for propagating backdoors across models. Our mechanistic analysis shows that the backdoor signal is primarily transmitted thro
    
[^255]: 抗蒸馏成本极低：通过风格性破坏实现轻量级蒸馏抵抗

    Antidistillation is Cheap: Lightweight Distillation-Resistance via Stylistic Corruption

    [https://arxiv.org/abs/2611.17772](https://arxiv.org/abs/2611.17772)

    该论文提出一种轻量级抗蒸馏防御方法，通过在检测到恶意查询时向教师模型激活中动态注入可定制的风格引导向量，以极低的计算开销破坏序列级蒸馏攻击，同时不影响正常输出的语义质量和流畅性。

    

    知识蒸馏（KD）是一种将大型语言模型（LLM）的能力迁移到更小、更专业化学生模型的高效范式。在商业部署场景中，前沿模型被限制在黑盒API之后，序列级蒸馏仍然是竞争对手以极低开发成本克隆专有能力的重要手段。现有的防御方法大多依赖代理模型来操纵下一个词的采样，这会增加计算开销和实时推理延迟。在本工作中，我们提出了一种轻量级防御方法，在检测到恶意查询时，动态地向教师模型的激活中注入风格引导向量。该框架具有卓越的灵活性，允许模型服务方定制目标防御风格并动态混合它们，以最大化风格差异。至关重要的是，这种推理时的干预保持了语义效用和流畅性……（原文摘要在此处被截断）

    arXiv:2611.17772v1 Announce Type: new  Abstract: Knowledge distillation (KD) is a highly effective paradigm for transferring capabilities from large language models (LLMs) to smaller, more specialized student models. In commercial deployment where frontier models are restricted behind black-box APIs, sequence-level distillation remains a potent vector for competitors to clone proprietary capabilities at a fraction of the development cost. Existing defenses mostly rely on surrogate models to manipulate next-token sampling, which increases computational overhead and real-time inference latency. In this work, we propose a lightweight defense that dynamically injects style-steering vectors into the teacher model's activations upon malicious queries. This framework offers exceptional flexibility, allowing hosts to customize target defensive styles and dynamically blend them to maximize stylistic variance. Crucially, this inference-time intervention preserves semantic utility and fluency dur
    
[^256]: GigaEdge Spench：一个面向实用流式语音增强的软件开发工具包（SDK）

    GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement

    [https://arxiv.org/abs/2611.13243](https://arxiv.org/abs/2611.13243)

    该论文发布了开源 SDK GigaEdge Spench，通过共享 C++ 核心与多平台集成（服务器、移动端、浏览器）实现实用的高速流式语音增强，其模型质量和处理速度优于 RNNoise 和 DPDFNet2 且快于实时。

    

    我们提出了 GigaEdge Spench，一个用于在 CPU 服务器、移动设备和浏览器上进行流式语音增强的开源 SDK。应用程序可以由神经去噪、归一化与动态处理模块、采样率转换，以及在支持情况下的声学回声消除来构建增强通道。这些处理阶段可以组合使用，同时归一化也可以在没有神经模型的情况下独立运行。本次发布包含一个 8 kHz INT8 去噪器，以及判别式和混合式的 48 kHz FP32 模型。一个共享的 C++ 核心通过原生接口支持 ONNX Runtime 和 TensorFlow Lite，并提供 Android 和 iOS SDK 以及 WebAssembly 集成。独立的流上下文可以共享模型资源，且来自同一流的连续帧可以在单次推理调用中处理。我们在 EARS-WHAM v2 数据集上评估了发布的模型，并将其质量与处理开销与 RNNoise 和 DPDFNet2 进行了比较。所有测试的原生和 Web 配置处理文件的速度均快于实时。

    arXiv:2611.13243v1 Announce Type: new  Abstract: We present GigaEdge Spench, an open-source SDK for streaming speech enhancement on CPU servers, mobile devices, and in browsers. Applications can build enhancement channels from neural denoising, normalization and dynamics blocks, sample-rate conversion, and acoustic echo cancellation where supported. These stages can be combined, while normalization can also run without a neural model. The release includes an 8 kHz INT8 denoiser and discriminative and hybrid 48 kHz FP32 models. A shared C++ core supports ONNX Runtime and TensorFlow Lite through native interfaces, Android and iOS SDKs, and WebAssembly integration. Independent stream contexts share model resources, and consecutive frames from one stream can be processed in a single inference call. We evaluate the released models on EARS-WHAM v2 and compare their quality and processing costs with RNNoise and DPDFNet2. All tested native and Web configurations process files faster than real 
    
[^257]: 推测式自蒸馏实现高效的知识内化

    Speculative Self-Distillation enables Efficient Knowledge Internalization

    [https://arxiv.org/abs/2611.12016](https://arxiv.org/abs/2611.12016)

    提出推测式自蒸馏（SSD）——一种逐 token 混合策略方法：默认由学生生成 rollout，教师仅在下一个 token 分布显著分歧处介入，既贴近学生的测试时行为分布又避免低信号状态，从而实现更高效的知识内化。

    

    许多现实世界的应用要求语言模型能够内化预训练中缺失的知识、纳入持续更新的事实，或学习新的领域技能。自蒸馏已成为完成这一任务的一种自然方法：以源材料为条件的模型充当教师，而无条件的学生模型则被训练以“闭卷”方式复现教师的行为。我们提出了推测式自蒸馏（Speculative Self-Distillation，SSD），这是一种高效的逐 token 混合策略自蒸馏方法。默认情况下由学生模型生成每个 rollout，而教师模型仅在其下一个 token 分布与学生模型出现显著分歧的位置进行干预。与使用教师生成前缀的离策略蒸馏不同，这种设计使训练 rollout 保持在接近学生测试时行为的分布上，同时避免了完全在策略 rollout 容易陷入的低信号状态。实际上，SSD 让教师……

    arXiv:2611.12016v2 Announce Type: replace-cross  Abstract: Many real-world applications require language models to internalize knowledge that was absent from pretraining, to incorporate continually updated facts, or to learn new domain skills. Self-distillation has emerged as a natural approach for this task. A model conditioned on the source material acts as the teacher, and an unconditioned student is trained to reproduce the teacher's behavior closed-book. We propose Speculative Self-Distillation (SSD), an efficient per-token mixed-policy method for self-distillation. The student generates each rollout by default, while the teacher intervenes only at positions where its next-token distribution substantially diverges from the student's. Unlike off-policy distillation with teacher-generated prefixes, this design keeps training rollouts close to the student's test-time behavior, while avoiding the low-signal states that fully on-policy rollouts drift into. In effect, SSD lets the teach
    
[^258]: 带在线推理调优的离线策略强化学习：CoRL 2026 RoboEval挑战赛第一名解决方案

    Off-Policy RL with Online Inference Tuning: 1st Place Solution to the CoRL 2026 RoboEval Challenge

    [https://arxiv.org/abs/2611.11464](https://arxiv.org/abs/2611.11464)

    该方案将基于π0.5策略（自任价值函数）的离线策略强化学习与AWR/RECAP优势训练及对约20个推理参数的汤普森采样在线调优相结合，并补充满奖励生成演示与失败恢复数据，最终夺得CoRL 2026 RoboEval挑战赛第一名。

    

    CoRL 2026 RoboEval挑战赛包含八个双臂操作任务，它不仅根据策略是否成功，还根据其完成任务的方式来排名：任务的每个阶段都要安全、高效地完成，并要求双臂协同配合，最终与人类演示进行对比评分。我的解决方案分为两部分。通用部分沿用自我在LeHome挑战赛中的方案，是对一个基于 $\pi_{0.5}$ 的策略进行离线策略强化学习，该策略同时充当自身的价值函数。它使用AWR和RECAP在精确官方分数的优势上进行训练，并在一个循环中使每次rollout同时也是对约20个推理参数的汤普森采样试验。开箱即用，该方案在两天内将策略在保留种子上的RDS提升至0.85（且距离收敛仍相去甚远）。针对比赛的部分则补充了局部探索无法发现的数据：能够获得满奖励的生成演示，以及从策略自身失败状态出发的恢复片段。所提交的（摘要在此处截断）

    arXiv:2611.11464v2 Announce Type: replace-cross  Abstract: The CoRL 2026 RoboEval challenge has eight bimanual tasks, and it ranks a policy not only on whether it succeeds but on how: every stage of the task, done safely, efficiently and with both arms in coordination, scored against human demonstrations. My solution has two parts. The generic part, carried over from my LeHome challenge solution, is off-policy RL on a $\pi_{0.5}$-based policy that is also its own value function. It is trained with AWR and RECAP on advantages of the exact official score, in a loop where every rollout is at the same time a Thompson-sampling trial of about 20 inference parameters. Out of the box, it took the policy to 0.85 RDS on held-out seeds in two days (and it was still very far from convergence). The competition-specific part adds data that local exploration cannot find: generated demonstrations that reach the full reward, and recovery episodes that start from the policy's own failures. The submitted
    
[^259]: 论智能体式操作策略

    On Agentic Manipulation Policies

    [https://arxiv.org/abs/2611.09892](https://arxiv.org/abs/2611.09892)

    提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。

    

    近年来机器人操作领域的研究主要聚焦于通过演示数据训练的专用策略。我们证明，通用型AI模型能够比专用策略（如视觉-语言-动作模型（VLA）和世界-动作模型（WAM））更可靠地控制机械臂。我们系统地分析了智能体式操作策略（AMP），即由通用型AI模型直接指挥机械臂，并随着任务的展开自适应地调整其动作。我们提出了AMP1，一个可在单臂、双臂和移动操作中无需任何改动即可运行的参考框架。AMP1在RoboLab-120和CaP-Gym基准测试中取得了最先进的结果，且无需任何演示数据或任务特定训练。在RoboLab-120上，AMP1零样本成功率达到85.6%，而领先的VLA和WAM策略分别仅为49.2%和42.9%。该智能体在涵盖十二个任务和三台机器人的108次真实世界测试中零样本成功了104次，显著降低了失败率。

    arXiv:2611.09892v2 Announce Type: replace-cross  Abstract: Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6% success rate zero-shot, compared to 49.2% and 42.9% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rate
    
[^260]: 多少个有序投影能确定k维空间中的n个点？

    How many sorted projections determine n points in k dimensions?

    [https://arxiv.org/abs/2611.06470](https://arxiv.org/abs/2611.06470)

    本文借助Hair和Sahai关于随机线性码列表恢复的两个引理，证明了在任意k维空间中，几乎任意选择适当数量的方向即可确定n个点的多重集，从而回答了Dym等人提出的高维开放问题。

    

    设 $d_1,\dots,d_m$ 为 $\mathbb{R}^k$ 中的方向，其中 $k\ge 2$。$\mathbb{R}^k$ 中 $n$ 个点构成的多重集 $X$ 会投影为 $m$ 个数字多重集 $\{\{x\cdot d_j : x\in X\}\}$，或等价地，投影为这些数字的 $m$ 个排序列表。需要多少个方向才能使这些投影确定任意一个由 $n$ 个点构成的多重集？在每个维度上都有两个已知的简单界：总是需要多于 $(k-1)\log_2 n$ 个方向，而处于一般位置的 $(k-1)n+1$ 个方向总是足够的。在平面上，人们知道得更多：Matousek、Privetivy 和 Skovron 证明了几乎任何 $O(\log n \log\log n)$ 个方向的选择都足够。至于在高维空间中是否有类似的结果成立，这是 Dym、Wellershoff、Tsoukanis、Levy 和 Balan 留下的开放问题。我们观察到，Hair 和 Sahai 为随机线性码的列表恢复所证明的两个引理可以回答该问题，并且通过收紧他们的论证，我们证明了在每个维度中……

    arXiv:2611.06470v1 Announce Type: cross  Abstract: Let $d_1,\dots,d_m$ be directions in $\mathbb{R}^k$, where $k\ge 2$. A multiset $X$ of $n$ points of $\mathbb{R}^k$ projects to the $m$ multisets of numbers $\{\{x\cdot d_j : x\in X\}\}$, or equivalently to the $m$ sorted lists of these numbers. How many directions are needed for these projections to determine every multiset of $n$ points? Two simple bounds are known in every dimension: more than $(k-1)\log_2 n$ directions are always needed, and $(k-1)n+1$ directions in general position always suffice. In the plane much more is known: Matousek, Privetivy and Skovron showed that almost every choice of $O(\log n \log\log n)$ directions suffices. Whether anything similar holds in higher dimensions was left open by Dym, Wellershoff, Tsoukanis, Levy and Balan. We observe that two lemmas of Hair and Sahai, proved for list recovery of random linear codes, answer that question, and by tightening their argument we show that in every dimension a
    
[^261]: 缓解深度图级聚类中的有效性受限与多类坍塌问题

    Alleviating Limited Effectiveness and Multi-Class Collapse in Deep Graph-level Clustering

    [https://arxiv.org/abs/2611.06052](https://arxiv.org/abs/2611.06052)

    该论文提出S³GLC框架，利用Hoffman-Wielandt不等式构建谱签名作为图编辑距离的高效下界代理，将全局拓扑特征与局部邻域特征相融合，从而缓解深度图级聚类中区分能力不足和多类别聚类坍塌的问题。

    

    给定一个包含N个图的集合，图级聚类（GLC）旨在将这些图划分为语义上有意义的群组，这是分析图结构数据的一项基础任务。尽管图同构网络（GIN）已成为GLC事实上的标准编码器，但由于忽视全局拓扑结构，它们仍无法有效区分不同的图。此外，现有的深度聚类框架存在严重的“聚类坍塌”问题，在不平衡的多类别场景中尤为突出。为应对这些挑战，我们提出了S³GLC，这是一个将全局拓扑特征与局部邻域特征相融合、以实现平衡图级聚类的新颖框架。在理论上，我们利用Hoffman-Wielandt不等式构建了谱签名，作为图编辑距离（GED）的严格下界，从而为全局拓扑特征提供了计算高效的代理。此外，我们还提出了一个联合流形……（摘要内容不完整）

    arXiv:2611.06052v1 Announce Type: new  Abstract: Given a collection of $N$ graphs, graph-level clustering (GLC) seeks to partition the $N$ graphs into semantically meaningful groups, which serves as a fundamental task in analyzing graph-structured data. While Graph Isomorphism Networks (GINs) have become the de facto encoders for GLC, they still fail to distinguish graphs due to their neglect of global topology. On top of that, existing deep clustering frameworks suffer from severe "clustering collapse" issues, particularly in imbalanced multi-class scenarios. To address these challenges, we propose S$^3$GLC, a novel framework that integrates global topological characteristics with local neighborhood features for balanced GLC. Theoretically, we leverage the Hoffman-Wielandt inequality to construct Spectral Signature as a rigorous lower bound for graph edit distance (GED), providing a computationally efficient proxy for global topological features. We additionally propose a joint manifo
    
[^262]: 通过分层脑电图-文本对齐实现思维破译

    Thought-deciphering via hierarchical EEG-to-text alignment

    [https://arxiv.org/abs/2611.03623](https://arxiv.org/abs/2611.03623)

    提出一种分层脑电图-文本对齐框架，先解码粗粒度语义类别、再在类别内检索细粒度句子，显著提升了从EEG进行句子级解码的检索准确率。

    

    非侵入式脑到文本接口为有沟通障碍的人群提供了替代性沟通途径。然而，从脑电图（EEG）进行句子级别的解码仍然具有挑战性。我们提出了一种分层脑电图-文本对齐框架，通过逐步解码粗粒度语义信息和细粒度句子表示来检索句子。我们假设，相比单个句子，更广泛的语义背景能够从脑电图中被更可靠地捕获。因此，脑电图表示首先与语义类别进行对齐，随后在预测出的类别内进行句子检索。在超过2,000个独特文本样本中，我们的方法实现了平均top-5和top-10检索准确率分别为62.6% ± 1.0%和83.5% ± 1.0%，而直接进行脑电图-句子对齐的准确率仅为50.3% ± 2.3%和68.3% ± 2.0%。定性分析进一步揭示了……（摘要内容截断）

    arXiv:2611.03623v2 Announce Type: replace  Abstract: Non-invasive brain-to-text interfaces offer alternative communication pathways for people with communication disabilities. However, sentence-level decoding from electroencephalography (EEG) remains challenging. We propose a hierarchical EEG-text alignment framework that retrieves sentences by progressively decoding coarse-grained semantic information and fine-grained sentence representations. We hypothesize that broader semantic context can be more reliably captured from EEG than individual sentences. Accordingly, EEG representations are first aligned with semantic classes, followed by sentence retrieval within the predicted class. Among over 2,000 unique text samples, our approach achieves average top-5 and top-10 retrieval accuracies of 62.6\% $\pm$ 1.0\% and 83.5\% $\pm$ 1.0\%, respectively, compared with 50.3\% $\pm$ 2.3\% and 68.3\% $\pm$ 2.0\% for direct EEG-sentence alignment. Qualitative analysis further reveals that decoding
    
[^263]: 以模型读取的方式书写事实：破解两跳诅咒

    Write Facts the Way Models Read Them: Lifting the Two-Hop Curse

    [https://arxiv.org/abs/2611.01519](https://arxiv.org/abs/2611.01519)

    论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。

    

    arXiv:2611.01519v2 公告类型：新 摘要：通过微调学习新事实的语言模型能够复述该事实，却无法用它进行推理：教会模型某个人是新国家公民之后，它仍然无法在不借助思维链的情况下说出这个人所在国家的国家元首是谁。这种“两跳诅咒”通常被视为推理能力缺陷，而最强的知识编辑方法则通过合成推理数据来解决它。我们证明：推理能力是完好的，错的是事实的书写方式。微调把新事实写成一个“要说出口的答案”，而第二跳推理从中间层开始就把桥接短语（“那个人的国家”）当作实体来读取：在相应层将模型自身对该实体的表征修补进桥接短语，可以基本恢复第二跳推理；然而微调之后，桥接短语与该新实体的接近程度并不比编辑前更高。因此我们提出“读取等价书写”（Legible）：用一个损失项让桥接短语在第二跳读取它的那一层（摘要在此处截断）

    arXiv:2611.01519v2 Announce Type: new  Abstract: A language model that learns a new fact by fine-tuning can repeat it but cannot reason with it: taught that a person is a citizen of a new country, it still cannot say, without chain-of-thought, who the head of state of that person's country is. This two-hop curse is widely treated as a reasoning deficit, and the strongest knowledge editors answer it with synthesized reasoning data. We show that the reasoning is intact and the writing is wrong. Fine-tuning writes a new fact as an answer to be said, whereas the second hop reads the bridge ("that person's country") as an entity, from the middle layers on: patching the model's own representation of the entity into the bridge at those layers largely restores the second hop, yet after fine-tuning the bridge there is no closer to the new entity than before the edit. We therefore propose read-equivalent writing (Legible): one loss term makes the bridge phrase, at the layer where the second hop 
    
[^264]: 面向低Tucker秩张量的轻量级表示学习

    Lightweight Representation Learning for Low Tucker Rank Tensors

    [https://arxiv.org/abs/2611.01168](https://arxiv.org/abs/2611.01168)

    本文提出了一种保持迭代轨迹不变的轻量化计算框架，通过仅计算更新所需的奇异三元组、避免Kronecker乘积的收缩以及稀疏加因子存储三个组件，显著降低了五种低Tucker秩张量迭代估计器的内存开销。

    

    用于低Tucker秩张量的迭代估计器，例如HOOI、核范数近端与ADMM方法、Frank-Wolfe以及缩放梯度下降，在每次迭代中都会构建大型对象：矩阵展开、Kronecker算子、完整的SVD分解以及环境维度的迭代变量。更新过程仅使用这些对象的一小部分，但它们却使内存成为处理大规模张量以及在边缘设备上的限制性资源。针对其中每种开销的补救方法都是已知的。然而，人们并不清楚什么样的重写能够保持迭代轨迹不变，以及这些重写在面对高效实现时于何处能带来收益。因此，我们提出了一种保持轨迹不变的轻量化计算方法。首先，我们将精确性表述为存储状态与参考迭代之间的交织关系。其次，我们证明了三个已知组件在五个估计器上均满足这一关系：仅计算更新所需的奇异三元组、无需Kronecker乘积的收缩运算，以及存储稀疏加因子（sparse-plus-factor）形式的……

    arXiv:2611.01168v2 Announce Type: replace-cross  Abstract: Iterative estimators for low Tucker rank tensors, such as HOOI, nuclear-norm proximal and ADMM methods, Frank-Wolfe and scaled gradient descent, form large objects in every iteration: unfoldings, Kronecker operators, full SVDs and ambient iterates. The update uses little of these objects, yet they make memory the limiting resource on large tensors and on edge-sized devices. Remedies for each of these costs are known. However, it is unclear when a rewriting keeps the iterates unchanged, and where it pays off against efficient implementations. We therefore present a trajectory-preserving lightweight evaluation. First, we state exactness as an intertwining relation between the stored state and the reference iteration. Second, we show that three known components satisfy this relation for five estimators: computing only the singular triplets that an update needs, contracting without Kronecker products, and storing sparse-plus-factor
    
[^265]: 稀疏自编码器归因图由边际决定

    Sparse Autoencoder Attribution Graphs Are Decided at the Margin

    [https://arxiv.org/abs/2611.00606](https://arxiv.org/abs/2611.00606)

    该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。

    

    基于稀疏自编码器（SAE）特征的归因图被用于预测消融操作的效果，并在不同训练种子之间进行比较。TopK SAE 在每个 token 上保留最大的预激活值，而特征的“边际”是其预激活值到截断阈值的距离。在来自三个 Pythia 模型的五个公开 TopK SAE 对上，将一个特征置零会在至少 0.74 的情形中使下一层的某个特征跨过截断阈值，而这些跨越承载了响应的大部分能量。对于将某个特征的编码放大三倍的写入操作，基于先前激活特征构建的图的中位相对误差达到 0.247 或更高。而“边际图”方法为每个预激活添加一阶变化并重新排序，在相同计算成本下，误差至多为 0.033。在中位数上，跨越特征的解码变化在范数上达到模型自身变化的至少 1.24 倍。在同一测试床的不同种子之间，特征之间的稀疏映射可以解释一个图的 0.85。

    arXiv:2611.00606v1 Announce Type: cross  Abstract: Attribution graphs over the features of sparse autoencoders (SAEs) are used to predict what an ablation does and are compared across training seeds. A TopK SAE keeps the largest preactivations at each token, and a feature's margin is its preactivation's distance to the cutoff. On five public TopK SAE pairs from three Pythia models, zeroing one feature moves a next layer feature across the cutoff in 0.74 of cases or more, and the crossings carry most of the response's energy. Over writes that scale a feature's code by three factors, the graph built on the features active beforehand has a median relative error of 0.247 or more. The margin graph adds the first order change to every preactivation and reranks, at the same cost, and errs by at most 0.033. At the median, the crossing features' decoded change is 1.24 times the model's own change in norm or more. Between seeds of one testbed, sparse maps between features explain 0.85 of a graph
    
[^266]: 基于重构式表征学习的置信度感知多普勒谱对地速度估计

    Confidence-Aware Speed-over-Ground Estimation from Doppler Spectra via Reconstruction-Based Representation Learning

    [https://arxiv.org/abs/2610.40253](https://arxiv.org/abs/2610.40253)

    该论文提出一种基于自编码器重构式表征学习的置信度感知方法，从多普勒谱中同时估计对地速度与帧级置信度，推理时省略解码器并采用专用置信度头进行误差界限估计。

    

    从朝向地面的多普勒谱中进行可靠的对地速度估计，对于铁路系统等移动平台至关重要。经典信号处理方法在良好条件下能够提供准确的估计，但在具有挑战性的条件下性能往往下降。我们提出了在铁路领域数据上训练的置信度感知对地速度估计方法，可从多普勒谱中同时估计对地速度和帧级置信度，并包含以下几项创新贡献：首先，我们将先前工作中利用自编码器重构任务进行性能评估的思想迁移到多普勒雷达领域。其次，我们在自编码器的潜空间特征上附加了一个对地速度估计头。第三，我们在推理阶段省略了重构解码器，而是增加了一个低成本的专用置信度头，用于帧级误差界限估计。第四，我们揭示了噪声增强训练如何能够提供高（摘要在此处被截断）

    arXiv:2610.40253v2 Announce Type: replace-cross  Abstract: Reliable speed-over-ground estimation from ground-facing Doppler spectra is essential for mobile platforms such as railway systems. Classical signal-processing approaches can provide accurate estimates under favorable conditions, but often degrade in challenging conditions. We propose confidence-aware speed-over-ground estimation trained on rail-domain data that estimates speed over ground and frame-level confidence from Doppler spectra with several novel contributions: First, we transfer the idea of exploiting a reconstruction task by an autoencoder for performance estimation from previous work to Doppler RaDAR. Second, we append a speed-over-ground estimation head operating on the autoencoder's latent space features. Third, we omit the reconstruction decoder during inference but add a low-cost dedicated confidence head for frame-wise error-bound estimation. Fourth, we reveal how noise augmentation training can provide high ro
    
[^267]: 面向CPU长上下文预填充注意力的分层缓存分块与内核内任务调度

    Tiered Cache Blocking and In-Kernel Task Dispatch for Long-Context Prefill Attention on CPU

    [https://arxiv.org/abs/2610.38809](https://arxiv.org/abs/2610.38809)

    提出TACPU——一种采用分层缓存分块与内核内任务调度机制的CPU高性能注意力内核，相比现有方案最高加速2.01倍，可将Llama-3.1-8B的首Token生成时间最高缩短1.74倍且不损失精度。

    

    智能体AI在各个领域带来了显著的生产力提升，但多个智能体并发的长上下文预填充可能会耗尽GPU的计算资源。尽管CPU提供了一个可行的替代方案，但现有的CPU注意力内核在适应多级缓存层次结构方面存在不足，且缺乏有效的线程调度。为解决这一问题，我们提出了TACPU，一个高性能注意力内核，其特点是具有分层缓存分块和内核内任务调度机制，专为因果全序列预填充而设计。TACPU相比PyTorch SDPA、FlexAttention和SGLang注意力内核实现了高达2.01倍的加速。集成到SGLang后，它将Llama-3.1-8B的首Token生成时间（TTFT）缩短了高达1.74倍，同时保持了模型精度。

    arXiv:2610.38809v3 Announce Type: replace  Abstract: Agentic AI drives significant productivity gains across diverse domains, but concurrent long-context prefills from multiple agents can saturate GPU compute resources. Although CPUs offer a viable alternative, existing CPU attention kernels fall short in adapting to multi-tier cache hierarchies and lack effective thread scheduling. To address this, we introduce TACPU, a high-performance attention kernel featuring tiered cache blocking and an in-kernel task dispatching mechanism designed for causal full-sequence prefilling. TACPU achieves up to a 2.01x speedup over PyTorch SDPA, FlexAttention, and SGLang attention kernels. Integrated into SGLang, it reduces Time to First Token (TTFT) of Llama-3.1-8B by up to 1.74x while preserving model accuracy.
    
[^268]: ResCom：面向大语言模型量化的激活残差补偿

    ResCom: Activation Residual Compensation for Large Language Model Quantization

    [https://arxiv.org/abs/2610.35955](https://arxiv.org/abs/2610.35955)

    ResCom提出一种低秩激活残差补偿方法，通过输出感知校准以闭式形式利用激活舍入残差校正层输出误差，无需梯度训练即可显著提升大语言模型4比特权重-激活量化的精度。

    

    低比特权重-激活量化可以降低部署大语言模型所需的内存和计算量，但即使经过校准，激活舍入仍可能导致精度下降。基于舍入后输入的重建无法区分同一量化区间内的激活值。我们提出ResCom，一种低秩校正方法，利用激活舍入残差来补偿由此产生的层输出误差。该方法的输出感知校准流程根据残差方向对输出重建的贡献来选择这些方向，并以闭式形式计算校正，无需梯度训练，也无需更新已校准的权重。在采用4比特权重和激活的Llama-3-8B上，秩为32的注意力校正将WikiText-2困惑度从7.708降至7.147，MMLU提升2.89个百分点，五样本GSM8K准确率从23.96%提升至36.16%。对参数规模从1B到70B的模型以及在留出的C4测试集上的评估（摘要在此处截断）……

    arXiv:2610.35955v1 Announce Type: cross  Abstract: Low-bit weight--activation quantization reduces the memory and computation needed to deploy large language models, but activation rounding can still degrade accuracy after calibration. Reconstruction from rounded inputs cannot distinguish activations within the same quantization bin. We propose ResCom, a low-rank correction that uses the activation rounding residual to compensate for the resulting layer-output error. An output-aware calibration procedure selects residual directions by their contribution to output reconstruction and computes the correction in closed form, without gradient training or updates to the calibrated weights. On Llama-3-8B with 4-bit weights and activations, rank-32 attention corrections reduce WikiText-2 perplexity from 7.708 to 7.147, improve MMLU by 2.89 percentage points, and raise five-shot GSM8K accuracy from 23.96 to 36.16 percent. Evaluations on models from 1B to 70B parameters and on held-out C4 test t
    
[^269]: 雅可比透镜的数学理论

    A Mathematical Theory of the Jacobian Lens

    [https://arxiv.org/abs/2610.31955](https://arxiv.org/abs/2610.31955)

    本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。

    

    Anthropic 最近的工作引入了一种新的可解释性工具，称为雅可比透镜（Jacobian lens，J-lens，Gurnee 等人，2026）。该方法在前沿语言模型的隐藏状态中识别出一些线性方向，其性质令人联想到工作记忆式的表征。然而，目前尚不清楚这些特征的哪些机制性质使得它们能够被 J-lens 检测到。为此，我们提出了 J-lens 的数学理论，并证明它优先识别的是“相干特征”——即表现得像主导的、空间相干的 Lyapunov 模态那样的方向。沿这些方向的小扰动会跨层放大，在不同词元位置之间大致保持恒定，并在不同上下文中保持一致。尽管我们对这些性质是否必然与“全局工作空间”相关联持不可知态度，但通过解析推导出的 J-lens 偏好仍然解释了先前应用于……的实证结果。

    arXiv:2610.31955v1 Announce Type: cross  Abstract: Recent work from Anthropic introduced a new interpretability tool, called the Jacobian lens (J-lens, Gurnee et al., 2026). This method identified linear directions in the hidden states of frontier language models that had properties reminiscent of working memory-like representations. However, it is unclear what mechanistic properties of such features make them detectable to the J-lens. To this end, we introduce a mathematical theory of the J-lens, and show that it preferentially identifies coherent features -- directions that act like leading, spatially coherent Lyapunov modes. Small perturbations along these directions amplify across layers, remain roughly constant across token positions, and are consistent across different contexts. While we are agnostic to whether these properties are necessarily associated with a ``Global Workspace'', the analytically derived preferences of the J-lens nevertheless explain prior empirical applied to
    
[^270]: 缺失的合规类别：坍缩越狱成功评估结果的LLM评分器及一项免费检测方法

    The Missing Compliance Class: LLM Graders That Collapse Jailbreak Success, and a Free Check

    [https://arxiv.org/abs/2610.24872](https://arxiv.org/abs/2610.24872)

    本文发现在越狱攻击评估中，常用的开源8B/9B LLM评判器会系统性地将“合规”类别坍缩为“规避”类别，导致不安全模型被误判为安全，而只需统计每类标签计数即可零成本检测这种坍缩问题。

    

    越狱攻击评估通常采用开放权重的7B至9B指令微调模型作为前沿API的本地、确定性评判器。我们在来自五个组织的八个评分器上固定目标模型、提示词、响应和四类评分标准，发现稠密的8B和9B评判器会将合规响应坍缩归类为规避。其中一个评判器在900个响应中零次判定为合规，同时与参考评判器在78.0%的样本上保持一致，从而将不安全的模型错误认证为安全。这种坍缩现象在四个900响应语料库中均成立，在主要语料库上将报告的合规率放大约22至25倍，同时非拒绝率偏移18.0至23.0个百分点。在剔除三个坍缩评判器后，合规率在所有四个语料库中仍有1.7至2.8倍的变化，而标注后的评分裁决（+16.3个百分点）和基准选择（15.3个百分点）仍会导致非拒绝率偏移。通过统计每类标签计数，可以零成本暴露类别坍缩问题，无需人工参与。

    arXiv:2610.24872v2 Announce Type: replace-cross  Abstract: Jailbreak evaluations often substitute open-weight 7B to 9B instruction-tuned models as local, deterministic judges for frontier APIs. Holding target models, prompts, responses, and a four-class rubric fixed across eight graders from five organizations, we find that dense 8B and 9B judges collapse compliant responses onto evasion. One assigns compliance zero times in 900 responses while agreeing with our reference judge on 78.0% of items, certifying an unsafe model as safe. This collapse holds across four 900-response corpora, multiplying reported compliance by a factor of 22 to 25 on our primary corpora while non-refusal shifts by 18.0 to 23.0 points. Without the three collapsing graders, compliance varies by 1.7x to 2.8x across all four corpora, whereas post-labelling rubric rulings (+16.3 points) and benchmark choice (15.3 points) still shift non-refusal. Per-class label counts expose class collapse at zero cost without huma
    
[^271]: HiPPO函数记忆：面向不规则时间序列的坐标可查询注意力

    HiPPO Functional Memory: Coordinate-Queryable Attention for Irregular Time Series

    [https://arxiv.org/abs/2610.24501](https://arxiv.org/abs/2610.24501)

    提出HiPPO函数记忆（HiPPO-FM），一种将不规则时间序列历史编译为可重用的坐标可寻址函数状态的注意力算子，可在任意时间坐标处直接求值查询，无需为每个查询反复解析观测相关表示。

    

    不规则时间序列要求模型能够在任意时间坐标处回答查询，然而现有方法通常保留依赖于观测的表示，需要在每次查询时反复解析。我们提出了HiPPO函数记忆，这是一种函数式记忆注意力算子，能够将不规则的历史数据编译为可重用的、可按坐标寻址的状态。HiPPO-FM不再存储以观测为索引的token或潜在轨迹，而是将观测写入固定秩的函数空间，并通过匹配的函数几何直接进行坐标求值。核平滑的HiPPO坐标与基于Gram矩阵的读出机制定义了共享的写入/读取机制，而基于观测支撑集的自注意力则能在压缩之前实现内容相关的表示。我们刻画了由此诱导的有限维查询空间，并推导出误差分解，将函数几何近似误差与学习……相分离。

    arXiv:2610.24501v3 Announce Type: replace  Abstract: Irregular time series require models to answer temporal queries at arbitrary coordinates, yet existing approaches typically retain observation-dependent representations that must be repeatedly resolved for each query. We introduce HiPPO Functional Memory (HiPPO-FM), a functional-memory attention operator that compiles an irregular history into a reusable coordinate-addressable state. Instead of storing observation-indexed tokens or latent trajectories, HiPPO-FM writes observations into a fixed-rank functional space and performs direct coordinate evaluation through matched functional geometry. Kernel-smoothed HiPPO coordinates and Gram-based readout define a shared write/read mechanism, while observed-support self-attention enables content-dependent representations before compression. We characterize the induced finite-dimensional query space and derive an error decomposition separating functional geometry approximation from learned-m
    
[^272]: 预先公告的变点使保形可靠性与频率校准相分离

    Announced Breaks Separate Conformal Reliability from Frequency Calibration

    [https://arxiv.org/abs/2610.23917](https://arxiv.org/abs/2610.23917)

    本文证明预先公告的结构性变点使“下一区间的保形可靠性”与“封闭窗口的频率校准”这两种重置审计目标可以分离，并提出在公告时点重启经验分位数与自适应递归的方法，前者达到1/√n的minimax可靠性速率、后者达到1/n的频率差距，且实验中覆盖率误差显著优于自适应基线（0.035对0.052）。

    

    许多结构性变点会提前公告，然而自适应保形方法只能从自身的预测误差中获知它们。在公告日期进行重置只能服务于两种审计之一：下一区间的保形可靠性，或一个封闭窗口的频率校准，而这两者可能不一致。对于独立得分，在变点边界处重启经验分位数可以达到极小化极大（minimax）可靠性速率，其阶为断裂后样本数的平方根分之一（1/√n）。在得分有界且递归下限累计增量有界的条件下，重启自适应递归可以达到阶为断裂后样本数分之一（1/n）的频率差距，但该差距对下一区间没有任何约束作用。在一次较大的合成变点之后，重启分位数在第50次预测时的覆盖率误差为0.035，而五个自适应基线中最好者为0.052。在每次合成变点之后的前50次预测中，重启递归都具有最小的频率差距。真实变点大多是……（原文在此截断）

    arXiv:2610.23917v2 Announce Type: replace-cross  Abstract: Many structural breaks are announced in advance, yet adaptive conformal methods learn of them only from their own errors. A reset at the announced date serves one of two audits, conformal reliability of the next interval or frequency calibration of a closed window, and the two can disagree. For independent scores, restarting the empirical quantile at the boundary attains the minimax reliability rate, of order one over the square root of the post-break count. Restarting an adaptive recursion attains a frequency gap of order one over that count on bounded scores when its floor adds a bounded total. That gap bounds nothing about the next interval. After a large synthetic break, the restarted quantile's coverage error at the 50th forecast is 0.035, against 0.052 for the best of five adaptive baselines. The restarted recursion has the smallest frequency gap over the first 50 forecasts after every synthetic break. Real breaks are mos
    
[^273]: 跨LLM训练阶段切换优化器的短暂代价

    The Ephemeral Cost of Switching Optimizers Across LLM Training Stages

    [https://arxiv.org/abs/2610.22650](https://arxiv.org/abs/2610.22650)

    该论文通过系统性实验首次发现，在LLM预训练与后训练阶段之间切换优化器（如AdamW与Muon之间）并不会造成实质性的性能损失，最终性能主要由各阶段所用优化器的独立效应决定，且在SFT中增加Muon更新比例反而能提升下游性能。

    

    最近的研究表明，在预训练和后续训练阶段之间切换优化器会带来性能损失，但缺乏对这一现象的系统性研究。我们通过从头预训练1.2B参数的Llama-3模型，并改变使用每种优化器进行更新的比例，来研究AdamW与Muon优化器之间的不匹配问题，同时在继续预训练和监督微调（SFT）期间对两种优化器进行类似的插值实验。令人惊讶的是，我们发现几乎没有证据表明存在有意义的优化器不匹配惩罚：最终性能在很大程度上可以由预训练和后训练优化器的独立效应来解释，且随着SFT期间Muon更新比例的增加，下游性能总体上有所提升。尽管匹配预训练优化器可以减少SFT期间的遗忘，但这并不能转化为最终基准测试性能的提升。我们进一步评估了AdamW和M……（摘要在此处截断）

    arXiv:2610.22650v1 Announce Type: new  Abstract: Recent work suggests that switching optimizers between pretraining and subsequent training stages incurs a performance cost, but systematic investigation of this phenomenon is lacking. We study the mismatch between AdamW and Muon optimizers by pretraining 1.2B Llama-3 models from scratch while varying the proportion of updates taken with each optimizer, and similarly interpolating between the two optimizers during continued pretraining and supervised fine-tuning (SFT). Surprisingly, we find little evidence of a meaningful optimizer mismatch penalty: final performance is largely explained by the independent effects of the pretraining and post-training optimizers, with downstream performance generally improving as the proportion of Muon updates during SFT increases. Although matching the pretraining optimizer can reduce forgetting during SFT, this does not translate into improved final benchmark performance. We further evaluate AdamW and M
    
[^274]: 尊重语义即可：通过LLM前语义可变分辨率分词与LLM内相关性剪枝实现图像token压缩

    Just Respect the Semantics: Image-Token Compression via Pre-LLM Semantic Variable-Resolution Tokenization and In-LLM Relevance Pruning

    [https://arxiv.org/abs/2610.18690](https://arxiv.org/abs/2610.18690)

    Varest通过语义引导的可变分辨率token化与相关性token剪枝，仅用5.56%的图像token即可保留86%-92%的全token性能，实现高效且语义无损的图像token压缩。

    

    图像token压缩必须在不损害语义的前提下减少计算量。我们提出Varest，其核心指导原则是语义可替代性：连贯语义区域内的token可以被粗略概括，而物体边界和文字内容则需要更精细的区分。物体和文本定位器引导可变分辨率合并，在边界和文字处保留空间覆盖范围和更精细的细节。学习到的[REL]寄存器随后在解码器早期处理之后对视觉token进行剪枝。跨预算的训练方式使得每个骨干网络只需一个检查点。在三个骨干网络和13个基准测试上，Varest在每个评估预算下的平均性能保持率均优于15个基线方法，仅使用原始层平均图像token数量的5.56%即可保留全token性能的86.10%至92.33%。

    arXiv:2610.18690v1 Announce Type: cross  Abstract: Image-token compression must reduce computation without hurting the semantics. We introduce Varest, guided by semantic fungibility: tokens within a coherent semantic region can be summarized coarsely, while object boundaries and literal text require finer distinctions. Object and text localizers guide variable-resolution merging, preserving spatial coverage and finer detail at boundaries and text. Learned [REL] registers then prune visual tokens after early decoder processing. Training across budgets enables one checkpoint per backbone. Across three backbones and 13 benchmarks, Varest outperforms 15 baselines in mean performance retention at every evaluated budget, retaining 86.10-92.33% of full-token performance with only 5.56% of the original layer-averaged image-token count.
    
[^275]: BridgeTwist：扭转薛定谔桥以实现免训练条件采样

    BridgeTwist: Twisting Schr\"odinger Bridges for Training-Free Conditional Sampling

    [https://arxiv.org/abs/2610.18658](https://arxiv.org/abs/2610.18658)

    提出了BridgeTwist，一种基于带温度调节扭转序贯蒙特卡洛方案的免训练条件采样器，可复用预训练薛定谔桥生成模型，利用由速度场与得分场构造的闭式即插即用扭转代理实现条件后验采样。

    

    生成模型通常需要服务于多种下游条件任务。在免训练的设置下，预训练的无条件模型被复用为先验，并与任务特定的似然函数相结合，从而从相应的后验分布中采样。尽管针对扩散模型和流模型已开发出此类免训练采样器，但作为在一般源分布与目标分布之间进行灵活随机输运的模型，薛定谔桥（SB）正日益受到关注，而其对应的免训练采样器却尚未被探索。我们提出了BridgeTwist，一种面向预训练薛定谔桥生成模型的免训练条件采样器，其被表述为一种带温度调节的扭转序贯蒙特卡洛方案。其中的扭转量是一个闭式的即插即用代理，通过将SB预训练所暴露的速度场与得分场组合为一个终端预测器，并在该终端处评估似然函数而得到。由于该代理在靠近源分布处可靠性较低，我们对其进行了温度调节……

    arXiv:2610.18658v1 Announce Type: cross  Abstract: Generative models are often required to serve many downstream conditional tasks. In the training-free setting, a pretrained unconditional model is reused as a prior and combined with a task-specific likelihood to sample from the corresponding posterior. While such training-free samplers have been developed for diffusion and flow models, their counterpart for Schr\"odinger bridge (SB) models remains unexplored, despite the growing appeal of SBs as flexible stochastic transport models between general source and target distributions. We propose BridgeTwist, a training-free conditional sampler for pretrained SB generative models, formulated as a tempered twisted sequential Monte Carlo scheme. Its twist is a closed-form plug-in surrogate, obtained by combining the velocity and score fields exposed by SB pretraining into an endpoint predictor and evaluating the likelihood there. Since this surrogate is less reliable near the source, we tempe
    
[^276]: SPLICE：稀疏物理库识别与组合扩展

    SPLICE: Sparse Physics Library Identification and Compositional Expansion

    [https://arxiv.org/abs/2610.17029](https://arxiv.org/abs/2610.17029)

    SPLICE是一个无需机制标签即可从多机制混合数据中自动发现物理算子的框架，通过凸优化推断贡献、校准测试分配机制，实现算子库的自动扩展并组合成零样本模拟器。

    

    神经算子拆分方法将一组神经算子（每个物理机制对应一个算子）组合成针对新方程的零样本模拟器。然而，现有方法需要事先指定这些机制及其标签。我们提出的问题是：能否从未标记的、多种机制同时作用的轨迹中调整和扩展一个预训练的算子库？我们引入了SPLICE，这是一个在混合数据上无需机制标签即可发现算子的框架。候选算子对每条轨迹的贡献通过求解凸优化问题来推断。随后，通过一个经过校准的测试为每个发现的算子分配其实现的机制。该框架还能检测失败情况，即当算子库缺乏所需物理机制或各机制之间无法充分分离时。在一维和二维系统上，发现的算子能够从未标记的混合数据中恢复机制，并组合成零样本模拟器。

    arXiv:2610.17029v1 Announce Type: cross  Abstract: Neural operator splitting composes a library of neural operators, one per physical mechanism, into zero-shot simulators of new equations. Existing approaches, however, require these mechanisms and their labels to be specified in advance. We ask whether a pretrained library can instead be adapted and grown from unlabeled trajectories in which multiple mechanisms act simultaneously. We introduce SPLICE, a framework that discovers operators without mechanism labels on mixture data. The contributions of the candidate operators to each trajectory are inferred by solving a convex optimization problem. A calibrated test then assigns each discovered operator the mechanism it implements. Failure cases are also detectable, when the library lacks the required physics or when mechanisms are insufficiently separable. Across one and two dimensional systems, the discovered operators recover mechanisms from unlabeled mixtures and compose into zero-sho
    
[^277]: 仅从站点观测数据学习联合概率天气预报

    Learning joint probabilistic weather forecasts from station observations alone

    [https://arxiv.org/abs/2610.09898](https://arxiv.org/abs/2610.09898)

    CLARA模型仅凭站点观测数据（无需数值天气预报或再分析数据）即可学习五个地面变量的联合高斯概率预报分布，以约2.8万参数的小模型在CPU上实现了优于各类基线的联合概率预报能力。

    

    评估复合天气风险需要能够刻画变量间依赖关系的预报。CLARA（校准平流路由注意力，Calibrated Advection-Routing Attention）仅利用站点观测数据即可学习五个地面变量的联合高斯预测分布，无需数值天气预报或再分析数据；该模型仅约28,000个参数，支持CPU训练和预测。在96个站点、六个多年折叠的数据集上，其平均时效能量分数比具有相同时间输入的学习型对比方法低4.9%（与参数量相近的方法相比低4.7%），比统计基线方法低11-65%。在固定边缘方差的条件下，移除学习到的相关性会使每个站点的联合负对数似然恶化1.0-2.8 nats。一个在给定假设下被证明具有一致性的协方差尺度估计器可改善短时效的校准效果，但在长时效时会出现过度校正。合成干预实验显示注意力偏差系数a

    arXiv:2610.09898v1 Announce Type: cross  Abstract: Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five surface variables from station observations alone, without numerical weather prediction or reanalysis; the approximately 28,000-parameter model supports CPU training and prediction. Across six multi-year folds on 96 stations, its lead-mean energy score is 4.9% lower than that of a learned comparator with matched temporal inputs (4.7% with a similar parameter count) and 11-65% lower than those of statistical baselines. Holding marginal variances fixed, removing learned correlations worsens joint negative log-likelihood by 1.0-2.8 nats per station. A covariance-scale estimator, proved consistent under stated assumptions, improves short-lead calibration but over-corrects at long leads. Synthetic interventions show an attention-bias coefficient a
    
[^278]: 为你的提示加噪：连续扩散语言模型中对条件令牌添加噪声

    Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.09145](https://arxiv.org/abs/2610.09145)

    在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。

    

    我们重新审视了连续扩散语言模型文献中的一个公认标准做法，即在训练期间保持条件提示令牌为干净（无噪声）状态。我们做了一个非常简单的修改：在训练期间也对条件提示令牌添加噪声。我们证明，在这一修改后的训练目标下，模型在数独和N皇后等组合推理任务中获得了更好的泛化能力，且在更难的变体上收益最大（数独困难版的解决率从3.73%提升至24.65%），同时生成解的多样性也有所提高（10x10 N皇后问题的覆盖率从50.60%提升至73.79%）。我们还展示了在使用Gigaword摘要数据集的中等数据规模下，自然语言生成质量有可衡量的提升，但值得注意的是，这些收益并不能迁移到所有自然语言任务上（例如开放式对话生成）。我们的方法只需对训练目标进行单行修改，无需额外的……

    arXiv:2610.09145v1 Announce Type: cross  Abstract: We revisit a standard accepted practice in the continuous diffusion language model   literature of fixing conditioning prompt tokens clean during training.   We make a very simple modification: also noise the conditioning prompt tokens during training.   We demonstrate that under this modified training objective, we achieve better generalization   in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants   ($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on   10x10 N-Queens). We also show measurable improvements to natural language generation quality   in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not   transfer to all natural language tasks (e.g open ended dialogue generation).   Our method is a single line change to the training objective, requires no additional i
    
[^279]: CADFather：通过协同工具调用实现自主CAD重建

    CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use

    [https://arxiv.org/abs/2610.09127](https://arxiv.org/abs/2610.09127)

    CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。

    

    从3D形状重建可编辑的CAD模型仍然是一项具有挑战性的工程任务。现有方法可以提出CAD操作，但没有任何单一的提案来源能够在不同的零件几何形状和重建阶段都同样有效。我们提出了CADFather，这是一个自主的智能体系统，通过协调互补的工具从3D网格中恢复参数化的CAD程序。一个视觉语言助手会检查目标形状和中间重建结果的渲染图，然后决定扩展哪些候选CAD程序、调用哪些工具、生成多少个提案以及何时结束。学习型和算法型工具负责提出CAD操作，而数值优化则用于细化现有程序的参数。新提出或细化后的程序会被执行和评估，从而为后续决策提供反馈。该智能体为每个目标零件维护多个备选的候选程序，并保留最佳的有效重建结果。

    arXiv:2610.09127v1 Announce Type: cross  Abstract: Reconstructing an editable CAD model from a 3D shape remains a challenging engineering task. Existing methods can propose CAD operations, but no single source of proposals works equally well across different part geometries and stages of reconstruction. We introduce CADFather, an autonomous agentic system that coordinates complementary tools to recover parametric CAD programs from 3D meshes. A vision-language assistant inspects renders of the target and intermediate reconstructions, then decides which candidate CAD programs to extend, which tools to invoke, how many proposals to generate, and when to finish. Learned and algorithmic tools propose CAD operations, while numerical optimization refines the parameters of existing programs. Proposed or refined programs are executed and evaluated to provide feedback for subsequent decisions. The agent maintains alternative candidate programs for each target part and preserves the best valid re
    
[^280]: 凸-凹强化学习

    Convex-Concave Reinforcement Learning

    [https://arxiv.org/abs/2610.09108](https://arxiv.org/abs/2610.09108)

    该论文揭示强化学习的策略优化问题在对数密度比坐标下具有凸差规划结构，从而将 CPI、NPG、TRPO 和 AWR 统一为特例，突破了传统凸代理近似的局限。

    

    策略学习驱动着当今许多最重要且投入巨大的强化学习应用。然而其赖以建立的核心优化问题（最大化期望回报）是出了名的非凸问题，即使采用直接策略参数化也是如此，而该领域在很大程度上是通过回避这一问题来应对的：在信赖域约束下优化回报的凸代理近似（如 NPG、TRPO、PPO、AWR）。我们证明这个看似无结构的问题实际上并非没有结构。在对数密度比坐标 $y := \log[\pi/\pi_n]$ 下，通过逐决策重要性采样（PDIS）可计算的精确逐迭代目标函数是一个“凸差-凸差约束规划”。这一结构使我们能够超越代理近似：它沿可解释的轴向将 CPI、NPG、TRPO 和 AWR 恢复为特例，并开辟了一个耦合连续决策的多步轴 $k$……（摘要在此处被截断）

    arXiv:2610.09108v1 Announce Type: new  Abstract: Policy learning drives many of the most consequential and heavily-invested applications of reinforcement learning today. Yet the core optimization problem it rests on (maximizing expected return) is notoriously non-convex, even under a direct policy parameterization, and the field has largely responded by avoiding it: optimizing convex surrogate approximations of the return under trust-region constraints (NPG, TRPO, PPO, AWR). We show that this seemingly unstructured problem is not actually structureless. In log-density-ratio coordinates $y := \log[\pi/\pi_n]$, the exact per-iteration objective, computable via per-decision importance sampling (PDIS), is a difference-of-convex-constrained difference-of-convex (DC-constrained DC) program. This structure lets us move beyond surrogate approximations: it recovers CPI, NPG, TRPO, and AWR as special cases along interpretable axes, and it opens a multi-step axis $k$ that couples consecutive deci
    
[^281]: InvestigationWorlds：一个用于法律调查的智能体环境

    InvestigationWorlds: An Agentic Environment for Legal Investigation

    [https://arxiv.org/abs/2610.04129](https://arxiv.org/abs/2610.04129)

    该论文利用真实的美国联邦法院简易判决动议构建了法律调查智能体环境InvestigationWorlds，评估发现智能体即使检索到相关证据也常常坚持错误假设，难以区分法院采纳的假设与其他备选假设。

    

    我们提出了InvestigationWorlds，一个用于法律调查的智能体环境。我们利用了美国民事诉讼中一种未被充分挖掘的材料：简易判决动议。该动议依赖于由真实证据材料组成的案卷记录，并产生一个法院采纳的假设，在裁决该动议时该假设被视为真实基准。每个环境均由从公共访问法院电子记录系统（PACER）检索到的真实美国联邦法院案例构建而成，并通过一个经律师验证的生成流程进行增强，该流程围绕原始案卷合成带有角色标签的文档。由此得到的语料库允许多种连贯的事实解读，其中只有一种与法院采纳的假设相符。在100个案例上的评估中，我们发现智能体尽管检索到了相关证据，却常常坚持错误的假设，难以将法院采纳的假设与其他备选假设区分开来。

    arXiv:2610.04129v2 Announce Type: replace  Abstract: We introduce InvestigationWorlds, an agentic environment for legal investigation. We build on an underused artifact of U.S. civil litigation: the summary judgment motion. This motion relies upon a record composed of real evidence exhibits, and results in a court-adopted hypothesis that is treated as ground truth for the purposes of deciding the motion. Each environment is built from a real U.S. Federal Court case retrieved from Public Access to Court Electronic Records (PACER) and augmented by an attorney-validated generation pipeline that synthesizes role-tagged documents around the original record. The resulting corpus admits multiple coherent factual readings, only one of which matches the court-adopted hypothesis. Evaluating on 100 cases, we find agents often commit to incorrect hypotheses despite retrieving relevant evidence, struggling to distinguish the court-adopted hypothesis from alternative hypotheses.
    
[^282]: MiDShip：面向工程设计的船舶货舱结构多模态数据集

    MiDShip: Multimodal Dataset of Ship Cargo Hold Structures for Engineering Design

    [https://arxiv.org/abs/2610.02214](https://arxiv.org/abs/2610.02214)

    本文提出了MiDShip多模态数据集，包含12,753个船舶货舱结构设计（涵盖参数数据、3D几何、工程图纸、材料清单和结构评估），填补了船舶结构设计中缺乏连接几何、性能与规范约束的结构化数据集的空白。

    

    船舶结构决定着船体的强度、安全性和可制造性，但其设计必须满足数百项船级社规范要求，使得设计过程复杂且需要反复迭代。数据驱动方法受到缺乏结构化数据集的限制，即缺少能够将设计几何、结构性能与基于规范的约束条件相联系的数据集。本文提出了MiDShip，一个包含12,753个合成货舱结构设计的多模态数据集：其中6,020个为随机生成，496个由受SGLD启发的程序生成，6,237个由基于方程的修复程序生成。每个设计包含参数化数据、完整及可直接划分网格的3D几何模型、工程图纸及标注、材料清单以及初步结构评估。此外，还评估了从ABS MVR（钢质海船入级规范）子集推导出的25项约束条件。结果显示，所有随机生成的设计均无法满足全部约束条件；在受SGLD启发的设计中，322个（64.9%）设计完全符合规范要求，平均……

    arXiv:2610.02214v1 Announce Type: cross  Abstract: Ship structures govern vessel strength, safety, and manufacturability, but their design must satisfy hundreds of classification society requirements, making the process complex and iterative. Data-driven approaches are limited by the lack of structured datasets linking design geometry, structural performance, and rule-based constraints. This paper presents MiDShip, a multimodal dataset of 12,753 synthetic cargo-hold structural designs: 6,020 random, 496 generated by an SGLD-inspired procedure, and 6,237 generated by an equation-informed repair procedure. Each design includes parametric data, full and mesh-ready 3D geometry, engineering drawings and annotations, a bill of materials, and preliminary structural evaluations. Twenty-five constraints derived from a subset of ABS MVR are also evaluated.   None of the random designs satisfies all constraints. Among the SGLD-inspired designs, 322 (64.9%) were fully compliant, with an average of
    
[^283]: 超越准确率：低精度快速矩阵乘法的前缀不变实现

    Beyond Accuracy: Prefix-Invariant Realizations of Low-Precision Fast Matrix Multiplication

    [https://arxiv.org/abs/2609.39816](https://arxiv.org/abs/2609.39816)

    该论文发现，即使在准确率上表现正常，低精度快速矩阵乘法实现仍可能破坏语言模型多选题评分所依赖的前缀不变性并改变模型答案，证明准确率不足以作为此类实现的认证标准。

    

    快速矩阵乘法通过精确相消来减少乘法次数，但对混合了多个 token 行的求和进行舍入，可能使后续 token 的贡献残留在语言模型较早的输出中。这威胁到前缀不变性——多选题似然评分所依赖的性质：被评分的似然必须只依赖于其允许的前缀。在 Qwen2.5-14B-Instruct 上，两种经修复达到表面上正常准确率的快速 FP8 实现，当仅把允许前缀之后的文本替换为 bf16 模型自身的贪心续写时，仍在 240 个 OpenBookQA 题目中分别改变 5.83% 和 10.00% 的题目的似然所选择的答案。而两个行局部对照——bf16 模型和已部署的 FP8 矩阵乘法内核——均未改变任何答案。因此，准确率并不能证明前缀不变性，且我们分析的各种稳定性判据也无法区分不同实现：在两级 Strassen 的全部 512 个符号变体上，这些判据保持恒定，而教师强制困惑度的跨度却……

    arXiv:2609.39816v1 Announce Type: new  Abstract: Fast matrix multiplication saves multiplications through exact cancellation, but rounding sums that mix token rows can leave contributions from later tokens in earlier language model outputs. This threatens prefix invariance, which multiple-choice likelihood scoring relies on: a scored likelihood must depend only on its allowed prefix. On Qwen2.5-14B-Instruct, two fast FP8 realizations repaired to ordinary-looking accuracy still change the answers chosen by likelihood on 5.83% and 10.00% of 240 OpenBookQA items when only the text after the allowed prefix is replaced with the bf16 model's own greedy continuation. Both row-local controls, the bf16 model and a deployed FP8 matrix multiplication kernel, change none. Accuracy thus does not certify prefix invariance, and the stability criteria we analyze cannot tell realizations apart: across all 512 sign variants of two-level Strassen they stay constant while teacher-forced perplexities span 
    
[^284]: ReSCENE：联邦持续学习中基于服务器端重放的结构性灾难性遗忘缓解方法

    ReSCENE: Server-Side Replay for Structural Mitigation of Catastrophic Forgetting in Federated Continual Learning

    [https://arxiv.org/abs/2609.38833](https://arxiv.org/abs/2609.38833)

    ReSCENE通过让客户端上传本地数据的浓缩代理、由服务器端存储并重放这些代理来训练全局模型，并配合时间牧群压缩缓冲区与理论保证，从结构上缓解联邦持续学习中的灾难性遗忘，同时减轻客户端负担。

    

    联邦持续学习需要在随时间不断整合新任务的同时，不丢失对先前任务的知识。现有的大多数方法是在联邦学习的“客户端训练、服务器聚合”循环上附加一个抗遗忘机制，这种做法会为了保留旧知识而阻碍新知识的学习，并给资源受限的客户端带来负担。我们提出ReSCENE，通过结构化的方式缓解灾难性遗忘：每个客户端上传其本地数据的一个小型浓缩代理集，服务器则保存过去任务的代理集，并将其与当前任务的代理集一起用于训练全局模型。为了实现高效的服务器端内存使用，我们引入了时间牧群选择，从任务过程中累积的代理池中选取较新的代理组成压缩缓冲区。我们的研究提供了理论分析，表明该缓冲区能够比完整累积所有代理更紧密地表示原始任务数据。在CIFAR-10等（基准数据集上的实验……）

    arXiv:2609.38833v1 Announce Type: new  Abstract: Federated continual learning must integrate new tasks over time without losing earlier-task knowledge. Most existing methods attach an anti-forgetting mechanism to the client-trained, server-aggregated loop of federated learning, which holds back new learning to preserve earlier knowledge and burdens resource-constrained clients. We propose ReSCENE, which structurally mitigates catastrophic forgetting by having each client upload a small condensed surrogate of its local data while the server keeps the surrogates of past tasks and trains the global model on them together with the current task surrogates. For efficient server memory, we introduce temporal herding, which selects the more recent surrogates from the pool accumulated over a task into a compressed buffer. Our study provides a theoretical analysis showing that this buffer can represent the original task data more closely than full accumulation of all surrogates. Across CIFAR-10,
    
[^285]: 加速器选择还不够：在云端TPU上进行AlphaFold2推理

    Accelerator Choice Is Not Enough: AlphaFold2 Inference on Cloud TPUs

    [https://arxiv.org/abs/2609.34818](https://arxiv.org/abs/2609.34818)

    仅选择TPU等加速器并不足够，软件层的配置（如使用JAX的pmap将查询映射到多颗芯片）才能真正释放AlphaFold2在云端TPU上的硬件性能优势。

    

    AlphaFold2是用JAX编写的，因此相同的推理代码无需修改即可在CPU、GPU和Google Cloud TPU上编译并运行。这种可移植性使得加速器的选择看起来像是用户需要做出的主要决策。我们证明事实并非如此。我们在Colab CPU运行时、NVIDIA T4 GPU以及专用的八芯片Cloud TPU v5e切片上运行同一AlphaFold2推理工作负载，发现TPU具有巨大的硬件优势——在同一测量活动中，单芯片稳态下每次调用仅需0.47秒，而T4需要13.1秒——但同时发现软件层在三个方面决定了用户实际能获得多少这一优势。默认执行路径仅使用八颗芯片中的一颗，按标价计算，闲置容量使得该切片每次预测的成本与GPU大致相当。使用JAX的vmap进行批处理始终无法超过单查询吞吐量，而使用JAX的pmap将查询映射到多颗芯片上，可使八颗芯片在匹配条件下获得单芯片6.5-7.9倍的吞吐量。

    arXiv:2609.34818v2 Announce Type: replace-cross  Abstract: AlphaFold2 is written in JAX, so the same inference code compiles and runs unchanged on CPUs, GPUs and Google Cloud TPUs. That portability makes the accelerator look like the main decision a user has to make. We show that it is not. Running one AlphaFold2 inference workload across a Colab CPU runtime, an NVIDIA T4 GPU and a dedicated eight-chip Cloud TPU v5e slice, we find a large hardware advantage for the TPU, 0.47 s per call in steady state on a single chip against 13.1 s on the T4 in the same measurement campaign, and three ways in which the software layer decides how much of it a user actually gets. The default execution path uses one chip of the eight, and at list prices the idle capacity makes the slice cost about as much per prediction as the GPU. Batching with JAX's vmap never exceeds single-query throughput, while mapping queries across chips with JAX's pmap gives eight chips 6.5-7.9x the throughput of one on a matche
    
[^286]: 我们能否从嵌入空间几何预测异常检测性能？

    Can We Predict Anomaly Detection Performance from Embedding-Space Geometry?

    [https://arxiv.org/abs/2609.26460](https://arxiv.org/abs/2609.26460)

    该论文为基于kNN的异常检测器推导了AUC下界，将检测性能与嵌入空间几何（分数分离度与方差）联系起来，并提出简单的伪异常探针方法，从而在无需真实异常数据的情况下实现可靠的模型选择。

    

    异常检测系统通常仅使用正常数据进行训练，而模型选择和评估往往需要带有标注的异常数据。我们研究了能否在不获取异常数据的情况下预测异常检测性能。对于基于kNN的检测器，我们推导了ROC曲线下面积（AUC）的一个下界，该下界将检测性能与内点分数和外点分数之间的分离度以及它们各自的方差联系起来。在局部缩放模型下，我们利用该下界刻画了密度变化、内在维度异质性以及跨域失配如何导致分数的变异性。随后，我们研究了无异常数据的模型选择问题，并表明仅凭内点分数方差无法在不同表示之间可靠地预测性能。为解决这一局限，我们引入了简单的伪异常探针，为估计相对分数分离度提供参考。在……上的实验（摘要在此处被截断）。

    arXiv:2609.26460v1 Announce Type: new  Abstract: Anomaly detection systems are often trained using normal data alone, while model selection and evaluation typically require labeled anomalies. We study whether anomaly detection performance can be predicted without access to anomalous data. For kNN-based detectors, we derive a lower bound on the area under the ROC curve (AUC) that relates detection performance to the separation between inlier and outlier scores and to their respective variances. Under a local scaling model, we use this bound to characterize how density variation, intrinsic-dimensional heterogeneity, and cross-domain mismatch contribute to score variability. We then investigate anomaly-free model selection and show that inlier score variance alone does not reliably predict performance across different representations. To address this limitation, we introduce simple pseudo-anomaly probes that provide a reference for estimating relative score separation. Experiments on the 
    
[^287]: CLOOPD：在策略蒸馏中闭合学习者回路

    CLOOPD: Closing the Learner Loop in On-Policy Distillation

    [https://arxiv.org/abs/2609.24141](https://arxiv.org/abs/2609.24141)

    CLOOPD提出一种闭环在策略蒸馏框架，将教师信号获取与学生端实现分离，通过自适应α航点、冻结评分批次、重复前向传播学生模型以及在独立token预算下分配演员工作量，显著提升了宏观准确率。

    

    在策略蒸馏（OPD）对每个新批次付出双重代价：学生生成轨迹，而更强的教师对这些轨迹进行评分。现有方法改进了哪些轨迹被评分以及教师信号如何构建，但通常仅通过一次演员更新来消耗该信号。我们提出CLOOPD，一个将教师信号获取与学生端实现相分离的闭环框架。CLOOPD在KL包络内选择自适应的α航点，冻结已评分批次及其优势，在每次演员遍历后重新对学生进行前向传播，测量实现程度，并在单独的token预算下分配演员工作量。该框架包括确定性的两次和三次遍历策略、按token计价的CLOOPD-TPMR，以及预算匹配的对照组。在8-H20节点上的六次300步运行中，在相当的教师token规模下，每个CLOOPD策略都优于一次遍历的TOP-D基准：使用CLOOPD-Fixed2，宏观准确率从15.41提升至17.78，以及19.

    arXiv:2609.24141v1 Announce Type: new  Abstract: On-policy distillation (OPD) pays twice for each fresh batch: the student generates trajectories and a stronger teacher scores them. Existing methods improve which trajectories are scored and how the teacher signal is constructed, but usually consume it with one actor update. We introduce CLOOPD, a closed-loop framework separating teacher-signal acquisition from student-side realization. CLOOPD selects an adaptive $\alpha$ waypoint inside a KL envelope, freezes the scored batch and its advantages, re-forwards the student after each actor pass, measures realization, and allocates actor work under a separate token budget. The framework includes deterministic two- and three-pass policies, token-priced CLOOPD-TPMR, and a budget-matched control. Across six 300-step runs on an 8-H20 node, every CLOOPD policy improves the one-pass TOP-D anchor at comparable teacher-token scale: macro accuracy rises from 15.41 to 17.78 with CLOOPD-Fixed2 and 19.
    
[^288]: RLVR²：基于可验证评分量规排序的强化学习

    RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking

    [https://arxiv.org/abs/2609.23457](https://arxiv.org/abs/2609.23457)

    本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。

    

    基于可验证奖励的强化学习（RLVR）正在从具有明确定义正确性信号的任务（如数学和代码）扩展到由多维度评分量规指定的多方面质量要求。由于策略优化在每次 rollout 中只消耗一个标量，基于评分量规的流程必须将多个标准的得分聚合为标量奖励。这种聚合通常被视为简单的分数缩放，但它实际上隐式地决定了训练过程中各质量维度之间的权衡方式。目前主流的做法是对每个标准进行归一化后取线性组合，这隐含地假设了基数分数差异在各标准之间是可比的，并且某一标准上的收益可以补偿另一标准上的失败；当各标准在语义上异质时，这两种假设都不可靠。我们提出了基于可验证评分量规排序的强化学习（RLVR²），这是一种面向基于评分量规的 RLVR 的可验证排序范式。对于每个标准……

    arXiv:2609.23457v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is expanding from tasks with well-defined correctness signals, such as mathematics and code, toward multifaceted quality requirements specified by multi-dimensional rubrics. Since policy optimization consumes one scalar per rollout, rubric-based pipelines must map multiple criterion scores into a scalar reward. This aggregation is often treated as score scaling, but it implicitly determines how quality dimensions trade off during training. The prevailing practice, normalizing each criterion and taking a linear combination, assumes that cardinal score differences are comparable across criteria and that gains on one criterion compensate for failures on another; both assumptions are unreliable when criteria are semantically heterogeneous. We propose Reinforcement Learning with Verifiable Rubric-based Ranking (RLVR$^2$), a verifiable ranking paradigm for rubric-based RLVR. For each criter
    
[^289]: RheoSampling：解决随机动态树投机解码中的独热困境

    RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding

    [https://arxiv.org/abs/2609.21827](https://arxiv.org/abs/2609.21827)

    该论文提出RheoSampling方法，通过将树构建与token验证两个任务解耦，解决了随机解码下动态树投机解码中草稿分布坍缩为独热概率、接受率骤降的两难困境。

    

    投机解码通过并行起草多个token来加速大语言模型推理，而基于树的方法通过层次化结构进一步提升了效率。EAGLE-3等动态树方法在贪心解码下通过确定性的top-K扩展和全局剪枝表现优异。然而，在随机解码（T>0）场景下，这种机制会将草稿分布坍缩为独热概率，导致接受率急剧下降。由此产生了一个两难困境：动态树方法为保留上下文感知的拓扑结构而牺牲了随机采样，而静态树方法则以与上下文无关的结构为代价来保留随机采样。问题的根源在于同一个概率分布被用于两个相互冲突的任务：构建树结构和验证token。这种耦合使得直接注入随机性因所产生的随机过程而变得极具挑战性。我们通过解耦这两个角色来解决该问题：RheoSampl...（原文摘要在此处截断）

    arXiv:2609.21827v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM inference by drafting multiple tokens in parallel, with tree-based methods further improving efficiency through hierarchical structures. Dynamic-tree methods such as EAGLE-3 perform well under greedy decoding via deterministic top-K expansion and global pruning. However, in stochastic decoding (T>0), this mechanism collapses the draft distribution into one-hot probabilities, causing a severe drop in acceptance rate. This creates a dilemma: dynamic-tree methods sacrifice stochastic sampling to preserve context-aware topology, while static-tree methods preserve stochastic sampling with context-agnostic structures. The issue arises because the same probability distribution is used for two conflicting tasks: constructing the tree and verifying tokens. This coupling makes direct injection of randomness challenging due to the resulting stochastic process. We resolve this by decoupling these roles: RheoSampl
    
[^290]: SWE-Proof：语言模型能否通过机器校验的证明解决真实世界的问题？

    SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

    [https://arxiv.org/abs/2609.21190](https://arxiv.org/abs/2609.21190)

    该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。

    

    确保大语言模型（LLM）生成代码的正确性是现代软件工程的核心挑战。面向智能体代码生成的基准测试通常使用留出的测试套件来检验正确性，但测试套件本质上是不完整的，且日益容易受到模型记忆（数据泄露）的影响。形式化验证可以同时避免这两个问题，但现有工作仅覆盖规范以输入形式给出的独立任务，而非真实问题——真实问题涉及大型代码仓库，并以模糊的自然语言表达意图。我们提出了Benchproofer，一个能将带有已知正确补丁的编码任务转化为形式化验证任务的流水线：它为新代码编写规范，用公理概括新代码所调用的已有函数，并且只有在机械验证与对抗性检查两道关卡均通过后才接受一个实例。将该流水线应用于SWE-bench Verified，我们得到了SWE-Proof——包含500个真实问题的基准，其正确性通过形式化验证而非测试来保证，并且该方法还可扩展至SWE-bench Pro。……

    arXiv:2609.21190v1 Announce Type: cross  Abstract: Ensuring the correctness of LLM-generated code is a core challenge for modern software engineering. Benchmarks for agentic code generation check correctness with held-out test suites, which are inherently incomplete and increasingly susceptible to memorization. Formal verification avoids both problems, but existing work covers only standalone tasks whose specifications are given as input, not real issues, which touch large repositories and state intent in vague natural language. We present Benchproofer, a pipeline that turns a coding task with a known correct patch into a formally verified one: it writes a specification for the new code, summarizes the existing functions that code calls with axioms, and admits an instance only after mechanical and adversarial gates agree. Applying it to SWE-bench Verified yields SWE-Proof, 500 real issues whose correctness is formally verified rather than tested, and it extends to SWE-bench Pro. Across
    
[^291]: 基于图的随机Power-UCT：结合幂平均估计的蒙特卡洛图搜索

    Graph-Based Stochastic Power-UCT: Monte-Carlo Graph Search with Power Mean Estimation

    [https://arxiv.org/abs/2609.19956](https://arxiv.org/abs/2609.19956)

    提出基于图的随机Power-UCT算法（GS-Power-UCT），通过在相同规划深度共享状态并保留不同深度的独立值，在含环路的一般随机MDP中实现样本复用，证明了以 O(n^{-1/2}) 速率收敛到有限视界值，并进一步提出两种全状态变体以增加样本共享并控制视界混合偏差。

    

    arXiv:2609.19956v1 公告类型：新论文 摘要：基于树的蒙特卡洛树搜索（MCTS）在随机马尔可夫决策过程（MDP）中，当同一状态通过不同轨迹被到达时会重复该状态，从而在随机MDP中浪费模拟次数。我们提出了基于图的随机Power-UCT（GS-Power-UCT），它在相同规划深度处共享到达的状态，同时为在不同深度到达的状态保留各自独立的值。该设计适用于一般的随机MDP，包括存在环路的问题。我们证明，对于固定的规划视界，根节点估计以 O(n^{-1/2}) 的速率收敛到有限视界值，与基于树的随机Power-UCT相匹配，同时在共享状态间复用样本。我们还研究了两个全状态变体：GS-Power-UCT-F，为每个物理状态存储一个节点以增加样本共享，但可能混合来自不同剩余视界的值；以及GS-Power-UCT-F⁺，它使用自适应视界来控制这种偏差。后者收敛到最优值 $V^{\star}(s_0)$。

    arXiv:2609.19956v1 Announce Type: new  Abstract: Tree-based Monte-Carlo Tree Search (MCTS) duplicates the same state when it is reached through different trajectories, which can waste simulations in stochastic MDPs. We introduce Graph-Based Stochastic-Power-UCT (GS-Power-UCT), which shares states reached at the same planning depth while keeping separate values for states reached at different depths. This design applies to general stochastic MDPs, including problems with cycles. We prove that for a fixed planning horizon, the root estimate converges to the finite-horizon value at rate $O(n^{-1/2})$, matching tree-based Stochastic-Power-UCT while reusing samples across shared states. We also study two full-state variants: GS-Power-UCT-F, which stores one node per physical state to increase sample sharing but may mix values from different remaining horizons, and GS-Power-UCT-F$^+$, which uses an adaptive horizon to control this bias. The latter converges to $V^{\star}(s_0)$, the optimal i
    
[^292]: 基于适配器库的组合式运动控制选项学习

    Learning Options for Compositional Motor Control with Adapter Banks

    [https://arxiv.org/abs/2609.17042](https://arxiv.org/abs/2609.17042)

    该论文将神经科学中“运动基元是共享循环网络低秩扰动”的理论转化为端到端可学习的新架构——共享循环核心加上由离散潜码选择的残差适配器库，适配器在无秩约束下自发涌现低秩特性，并通过冻结网络后优化的高层策略组合选项实现新颖分布外运动的泛化。

    

    学习灵活的运动基元是熟练运动控制的标志。近期的神经科学理论提出，运动基元可能被实现为共享循环网络的低秩扰动，但该系统如何被学习仍是悬而未决的问题。我们将这一原理转化为一种新颖的端到端运动技能学习架构：一个共享的循环核心由一组残差适配器进行调制，每个适配器由一个离散潜码选择。在闭环生物力学控制任务上训练后，尽管没有架构上的秩约束，适配器自发涌现出对循环动力学的低秩扰动，将任务表征置于共享核心网络的不同子空间中。一个基于所学选项的简单高层策略——在整体网络冻结的情况下进行优化——对低秩适配器进行排序组合，以产生新颖的分布外运动。我们展示了在……方面泛化到新颖运动序列的能力。

    arXiv:2609.17042v1 Announce Type: new  Abstract: Learning flexible motor primitives is a hallmark of skilled motor control. Recent neuroscience theory proposes that motor primitives may be implemented as low-rank perturbations of a shared recurrent network, but leaves open how such a system is learned. We translate this principle into a novel architecture for learning motor skills end-to-end: a shared recurrent core modulated by a bank of residual adapters, each selected by a discrete latent code. Trained on closed-loop biomechanical control, the adapters develop emergent low-rank perturbations of the recurrent dynamics despite no architectural rank constraint, placing task representations in disparate subspaces of the shared core network. A simple high-level policy over the learned options, optimized while the whole network is frozen, sequences the low-rank adapters to produce novel out-of-distribution movements. We demonstrate the ability to generalize to novel motor sequences within
    
[^293]: 超越点预测：时间序列与时空数据概率预测综述

    Beyond Point Forecasts: A Survey on Probabilistic Forecasting for Time Series and Spatiotemporal Data

    [https://arxiv.org/abs/2609.13345](https://arxiv.org/abs/2609.13345)

    这篇综述通过按不确定性在预测流程中引入的位置和方式建立统一分类框架，整合了时间序列与时空概率预测中分散的统计建模、机器学习和深度生成建模方法，并涵盖了集成校准、贝叶斯建模、分布回归以及新兴的时间序列基础模型等范式。

    

    概率预测是不确定性下决策的核心，但其方法论版图在时间序列预测与时空预测、统计建模、机器学习以及深度生成建模等领域之间日益碎片化。本综述通过根据不确定性在预测流程中引入的位置和方式来组织概率预测方法，构建了一个统一的视角。我们的分类体系将模型无关方法（包括集成方法和无分布校准）与模型内在方法（涵盖贝叶斯建模、参数化预测分布、分布回归以及现代生成模型）联系起来，并进一步审视了时间序列基础模型的新兴角色。除方法论的综合梳理之外，我们还识别了不同范式所依赖的假设、计算需求以及所刻画的不确定性形式，并将这些差异转化为……（摘要内容在此处截断）

    arXiv:2609.13345v1 Announce Type: cross  Abstract: Probabilistic forecasting is central to decision-making under uncertainty, yet its methodological landscape has become increasingly fragmented across temporal and spatiotemporal forecasting, statistical modeling, machine learning, and deep generative modeling. This survey develops a unified perspective by organizing probabilistic forecasting methods according to where and how uncertainty is introduced into the forecasting pipeline. Our taxonomy connects model-agnostic approaches including ensembles and distribution-free calibration, with model-intrinsic approaches spanning Bayesian modeling, parametric predictive distributions, distributional regression, and modern generative models, and further examines the emerging role of time series foundation models. Beyond methodological synthesis, we identify the assumptions, computational demands, and forms of uncertainty represented by different paradigms, and translate these distinctions into
    
[^294]: 模型更新的有限审计：配对差异比较的覆盖率与分辨率

    Finite Audits of Model Updating: Coverage and Resolution of Paired Disparity Comparisons

    [https://arxiv.org/abs/2609.09788](https://arxiv.org/abs/2609.09788)

    该论文发现，模型更新中的配对差异审计尽管更精确、更有决断力，其区间对子群体差距变化的覆盖率却远低于名义95%（bootstrap仅12.0%，数值差分替代方法也仅52.9%），且绝对差距偏差、差距方向与先验强度会显著影响点估计与结论。

    

    有限审计可以变得更加精确和果断，却无法可靠地覆盖子群体差异的变化。我们研究了在固定历史窗口上平均的、更新模型减去冻结模型在绝对子群体差距上的差异，并分别针对真阳性率和假阳性率进行分析。通过受控的配对计数实验以及对保留模型历史的独立审计，我们区分了区间覆盖率、错误结论与决策分辨率。在冻结差距为零、更新差距为五个百分点的情形下，bootstrap覆盖率为12.0%；一种数值差分的替代方法可将覆盖率提升至52.9%，但仍远低于名义上的95%（每窗口包含较小群体的500个与其他群体的2000个指定结果观测值，共十个窗口）。不相等的绝对差距偏差解释了点估计的位移，而差距方向会改变配对方法对方差的贡献。先验强度对稀疏且偏离中心的比较具有实质性影响。在保留的面板数据中……

    arXiv:2609.09788v2 Announce Type: replace  Abstract: Finite audits can become more precise and decisive without reliably covering the change in subgroup disparity. We study the updating-minus-frozen difference in absolute subgroup gaps, averaged over a fixed history, for true-positive and false-positive rates separately. Controlled paired-count experiments and independent audits of retained model histories distinguish interval coverage, erroneous conclusions and decision resolution. With a zero frozen gap and a five-percentage-point updating gap, bootstrap coverage is 12.0%; a numerical-delta alternative improves coverage to 52.9%, still far below the nominal 95%, with 500 smaller-group and 2,000 other-group observations of the specified outcome per window over ten windows. Unequal absolute-gap biases explain point-estimate displacement, and gap direction changes pairing's variance contribution. Prior strength materially affects sparse, off-centre comparisons. In the retained panel, re
    
[^295]: Sinkhorn层中的支撑拓扑与梯度混合

    Support Topology and Gradient Mixing in Sinkhorn Layers

    [https://arxiv.org/abs/2609.07954](https://arxiv.org/abs/2609.07954)

    本文发展了一套固定支撑演算来分析Sinkhorn层中的梯度传播机制，并证明了传输多面体的每个可行面具有成对两跳列重叠是支撑图与边际保证梯度均匀一步收缩的充要条件。

    

    稀疏Sinkhorn层使用固定的支撑图来限制token之间的传输。该图如何控制缩放迭代中的梯度传播？我们发展了一套固定支撑演算，证明在模常数意义下，每个行-列循环会对列势扰动诱导出一个行随机算子。其转置用于传播零质量的反向模式余切。有限循环算子使用两个不同的半步传输计划；在平衡不动点处，它退化为由单一计划确定的两步游走。我们推导了相应的得分项和边际源项，并利用Dobrushin收缩与minorization方法来界定齐次和源驱动的尾部余切。我们的主要结果刻画了支撑和边际在有限得分上一致保证一步收缩的条件：传输多面体的每个可行面必须具有成对的两跳列重叠。否则，某些合适的得分方向（原文此处截断）……

    arXiv:2609.07954v1 Announce Type: new  Abstract: Sparse Sinkhorn layers use a fixed support graph to restrict transport between tokens. How does this graph control gradient propagation through the scaling iterations. We develop a fixed-support calculus showing that each row-column cycle induces a row-stochastic operator on column-potential perturbations modulo constants. Its transpose propagates zero-mass reverse-mode cotangents. The finite-cycle operator uses two distinct half-step transport plans; at a balanced fixed point it reduces to a two-step walk determined by a single plan. We derive the accompanying score and marginal source terms and use Dobrushin contraction and minorization to bound homogeneous and source-driven tail cotangents. Our main result characterizes when support and marginals guarantee one-step contraction uniformly over finite scores: every feasible face of the transportation polytope must have pairwise two-hop column overlap. Otherwise, suitable score directions
    
[^296]: 具有噪声约束值的约束在线学习

    Constrained Online Learning with Noisy Constraint Values

    [https://arxiv.org/abs/2609.06921](https://arxiv.org/abs/2609.06921)

    针对约束值受噪声污染的约束在线凸优化，本文证明了联合次线性保证的不可能性，并提出LEDGER算法，通过引入预算违反新度量，实现了 $O(\sqrt T/V)$ 的期望遗憾和可控的预算违反保证。

    

    我们研究具有对抗性约束的约束在线凸优化问题，其中约束值和梯度是通过无偏噪声观测得到的。标准差为 $\sigma$ 的高斯值噪声会导致期望遗憾与期望硬违反的最大值在最坏情况下存在 $\Omega(\min\{\sigma,1\}T/\log^7T)$ 的下界，即使梯度已知也是如此。这排除了在固定 $\delta>0$ 和固定正噪声水平下任何联合的 $O(T^{1-\delta})$ 保证。因此，我们转而研究预算违反：即固定时间范围内任意窗口上的最大累积超支。我们提出了 LEDGER 算法，该算法在非负余额中跟踪观测到的净消耗，并在当前反馈噪声产生之前设置约束权重。在常见的可行性条件和条件有限方差反馈下，对于固定的问题参数，LEDGER 算法实现了 $O(\sqrt T/V)$ 的期望遗憾和 $O(\sqrt V\,T^{3/4}+\sigma\sqrt T)$ 的期望预算违反，其中 $V$……

    arXiv:2609.06921v1 Announce Type: cross  Abstract: We study constrained online convex optimization with adversarial constraints when constraint values and gradients are observed through unbiased noise. Gaussian value noise of standard deviation $\sigma$ yields a worst-case lower bound of $\Omega(\min\{\sigma,1\}T/\log^7T)$ on the maximum of expected regret and expected hard violation, even with known gradients. This rules out any jointly $O(T^{1-\delta})$ guarantee for fixed $\delta>0$ and fixed positive noise level. We therefore study budget violation: the largest cumulative overspend over any window within a fixed horizon. We introduce \LEDGER, which tracks observed net consumption in a nonnegative balance and sets constraint weights before the current feedback noise. Under common feasibility and conditional finite-variance feedback, for fixed problem parameters, \LEDGER\ achieves $O(\sqrt T/V)$ expected regret and $O(\sqrt V\,T^{3/4}+\sigma\sqrt T)$ expected budget violation for $V\
    
[^297]: 从合成先验到模型行为：表格基础模型中的结构覆盖度

    From Synthetic Priors to Model Behavior: Structural Coverage in Tabular Foundation Models

    [https://arxiv.org/abs/2609.06912](https://arxiv.org/abs/2609.06912)

    该论文从分布层面归因视角出发，通过结构覆盖度和归一化密度量化表格基础模型的合成预训练先验对下游基准任务的支持范围，并探究这种结构支持与预测性能之间的关联。

    

    表格基础模型（TFMs）通常在大量程序化生成的合成任务集合上进行预训练，然而这些合成预训练先验对模型所评估的下游任务的支持程度仍不清楚。我们从分布层面的归因视角来研究这一问题。我们恢复或重构了四个表格基础模型的合成数据生成器，并将其生成的任务与两个广泛使用的表格基准数据集进行比较。每个数据集都由一组共同的结构描述符表示，这些描述符涵盖模式、特征分布、依赖结构、响应属性以及特征-响应关系。在该空间中，我们使用结构覆盖度和归一化密度来衡量每个合成先验到达基准任务的广度与重复程度，并检验更强的局部支持是否与更好的预测性能相关联。我们发现存在显著差异……（原文摘要至此被截断）

    arXiv:2609.06912v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) are commonly pretrained on large collections of procedurally generated synthetic tasks, yet it remains unclear how well these synthetic pretraining priors support the downstream tasks on which the models are evaluated. We study this question from a distribution-level attribution perspective. We recover or reconstruct the synthetic data generators of four TFMs and compare their generated tasks with datasets from two widely used tabular benchmarks. Each dataset is represented by a common set of structural descriptors capturing schema, feature distributions, dependence structure, response properties, and feature--response relationships. In this space, we measure how broadly and repeatedly each synthetic prior reaches benchmark tasks using structural coverage and normalized density, and examine whether stronger local support is associated with better predictive performance. We find substantial differences a
    
[^298]: ARC-Bench：闭环重规划掩盖了冻结JEPA世界模型中失效的动作排序

    ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models

    [https://arxiv.org/abs/2609.05461](https://arxiv.org/abs/2609.05461)

    该论文提出ARC-Bench无泄漏审计协议，揭示冻结JEPA世界模型中“潜在距离可正确排序动作”的核心假设严重且结构性失效，且该缺陷在不同视觉骨干网络下均持续存在。

    

    无奖励的潜在世界模型通过在冻结的潜在空间中以距离对候选动作进行评分来进行规划：如果一个动作预测的未来嵌入更接近目标嵌入，则该动作会被优先选择。这悄然假设了潜在接近度是可对动作进行排序的，即按潜在距离对候选动作排序与按真实代价排序是一致的。我们直接审计了这一假设。我们提出了ARC-Bench，一种无泄漏、固定候选集的评测协议，用于衡量冻结的JEPA式目标能否正确排序候选动作，并将其应用于官方发布的JEPA-WM检查点，涵盖导航和操作类控制任务。这一假设严重且结构性地失效了：在官方的操作任务审计中，得分最高的候选动作几乎总是次优的，同样的排序倒置现象也出现在迷宫环境中。一项受控的视觉骨干网络扩展实验表明，当将DINOv2替换为视频预训练的V-JEPA 1时，该缺陷依然存在。

    arXiv:2609.05461v1 Announce Type: new  Abstract: Reward-free latent world models plan by scoring candidate actions with distances in a frozen latent space: an action is preferred if its predicted future embedding lands closer to the goal embedding. This silently assumes that latent closeness is action-rankable, i.e., that ordering candidates by latent distance agrees with ordering them by true cost. We audit this assumption directly. We introduce ARC-Bench, a no-leak, fixed-candidate protocol that measures whether frozen JEPA-style objectives rank candidate actions correctly, and apply it to official released JEPA-WM checkpoints across navigation and manipulation-style control. The assumption fails, severely and structurally: on the official manipulation audits the top-scored candidate is almost always suboptimal, and the same inversion appears in the maze domains. A controlled visual-backbone extension shows that the defect persists when DINOv2 is replaced by video-pretrained V-JEPA 1
    
[^299]: 基于无限维连续归一化流学习信息先验的贝叶斯逆问题方法

    Learning Informative Prior with Infinite-Dimensional Continuous Normalizing Flow for Bayesian Inverse Problem

    [https://arxiv.org/abs/2609.03343](https://arxiv.org/abs/2609.03343)

    该论文提出了一种基于无限维连续归一化流的新方法，通过在希尔伯特空间中引入定义良好的神经常微分方程将简单参考测度变换为编码先验信息的复杂测度，建立了无限维贝叶斯先验的适定性理论框架，并提供了先验训练方法与后验采样算法，用于求解偏微分方程的贝叶斯逆问题。

    

    本文研究了具有无限维希尔伯特空间中模型参数的偏微分方程逆问题的无限维贝叶斯推断。为了有效融入先验信息，我们提出了一种新颖的基于连续归一化流的无限维模型。具体而言，通过在无限维空间中引入一个定义良好的神经常微分方程，可以将简单的参考测度变换为编码先验信息的更复杂测度。我们建立了相应的理论框架，以确保所提出的贝叶斯先验在无限维空间中的适定性。我们还针对两种不同的数据设置提供了先验的训练方法，并为由此得到的贝叶斯后验提供了两种采样算法。所提出的框架被应用于三个代表性的逆问题：简单的光滑逆问题、逆散射问题等。

    arXiv:2609.03343v1 Announce Type: cross  Abstract: This paper addresses infinite-dimensional Bayesian inference for inverse problem of partial differential equations with model parameters in infinite-dimensional Hilbert space. To effectively incorporate prior information, we propose a novel continuous normalizing flows based infinite-dimensional model. Specifically, by introducing a well-defined neural ordinary differential equation in infinite-dimensional space, a simple reference measure can be transformed into a more complex measure which encodes the prior information. A corresponding theoretical framework is established to ensure the well-posedness of our proposed Bayesian prior in infinite-dimensional space. We also provide training methods of the prior for two distinct data settings, along with two sampling algorithms for the resulting Bayesian posterior. The proposed framework is applied to three representative inverse problems: the simple smooth inverse problem, inverse scatter
    
[^300]: 剖析医学问答中误导性上下文的作用机制

    Untangling the Mechanisms of Misleading Context in Medical Question Answering

    [https://arxiv.org/abs/2609.02754](https://arxiv.org/abs/2609.02754)

    该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。

    

    大语言模型如今能够以专家级水平回答医学问题。然而，这些系统所依据的上下文可能具有误导性，而误导性上下文会破坏模型的医学判断。为了理解误导性上下文如何破坏这一判断，我们考察了模型对上下文的易感性、对误导信息的披露程度、推理被破坏的机制以及决策的可监控性。在一个由临床医生审核、包含8,627个问题的问答基准MedMisBench的医学推理子集上，我们注入了两类误导性上下文线索：伪造的证据和纯粹的断言。我们测试了三个推理模型，其中两个会暴露其完整的推理轨迹，另一个前沿模型仅暴露其最终回复。所有三个模型都更容易受纯粹断言而非伪造证据的影响，采用断言答案的频率高出10至27个百分点。误导性线索在81%至98%的推理轨迹中被披露，但仅……

    arXiv:2609.02754v1 Announce Type: cross  Abstract: Large language models now answer medical questions with expert-level performance. However, the context these systems act on can be misleading, and misleading context can corrupt a model's medical judgment. To understand how misleading context corrupts this judgment, we examine the model's susceptibility to the context, disclosure of it, mechanism of corrupted reasoning, and monitorability of the decision. On the medical reasoning subset of MedMisBench, a clinician-reviewed question-answering benchmark of 8,627 questions, we inject two types of misleading context cues, fabricated evidence and a bare assertion. We test three reasoning models, two that expose their full reasoning trace and one frontier model that exposes only its response. All three are more susceptible to the assertion than to the fabricated evidence, adopting the asserted answer 10 to 27 points more often. The misleading cues are disclosed in 81 to 98% of traces but onl
    
[^301]: CRAD：面向去中心化异构联邦学习的类别级可靠性感知蒸馏

    CRAD: Class-wise Reliability-Aware Distillation for Decentralized Heterogeneous Federated Learning

    [https://arxiv.org/abs/2609.00446](https://arxiv.org/abs/2609.00446)

    该论文提出CRAD框架，通过类别级可靠性感知的方式组合同伴教师的软预测，在去中心化联邦学习中同时解决模型架构异构与非独立同分布数据两大难题，且无需中央服务器或共享原始数据。

    

    传统的联邦学习（FL）依赖于参数平均，这要求客户端具备双重同质性：既要求模型架构完全相同，又在非独立同分布数据下性能退化。而现实世界的部署通常会同时打破这两个假设。我们通过构建一个去中心化的知识蒸馏框架来规避这两个问题：在该框架中，每个客户端在自己的本地数据上评估其同伴的模型快照，并从得到的软预测中进行蒸馏。由于知识通过共享的类别后验概率进行传递，客户端可以自由使用不同的模型架构；同时，由于每个教师模型都在学生自己的设备上进行评估，原始数据永远不会离开客户端，且无需中央服务器或公共数据集。在此设定下，我们识别并解决了一个尚未被充分研究的问题：如何组合多个同伴教师模型的预测。现有方法（如均匀平均）忽略了知识可靠性在教师之间以及不同类别之间的差异。我们（提出……原文在此处截断）

    arXiv:2609.00446v1 Announce Type: new  Abstract: Conventional federated learning (FL) relies on parameter averaging, which forces clients to be doubly homogeneous: it demands an identical architecture and degrades under non-IID data. Real-world deployments usually break both assumptions. We sidestep both by building a decentralized knowledge distillation framework in which each client evaluates its peers' model snapshots on its own local data and distills from the resulting soft predictions. Because knowledge is transferred through the shared class posterior, clients are free to run different architectures; and because every teacher is evaluated on the student's own device, raw data never leaves the client, with no central server or public dataset required. Within this setting, we identify and address an under-examined problem: how to combine the peer teacher predictions. Existing methods, like uniform averaging, ignore how knowledge reliability varies across teachers and classes. We p
    
[^302]: 潜在规划能否在点云中存活？面向几何观测的动作条件化JEPA世界模型

    Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations

    [https://arxiv.org/abs/2608.29434](https://arxiv.org/abs/2608.29434)

    该论文证明JEPA世界模型的潜空间规划可以从图像成功扩展到点云等几何观测——三种经典JEPA设计在点云上均能无坍塌地规划，其中动作敏感模型在几何变化最大的场景中表现最佳。

    

    JEPA世界模型使潜空间规划成为实现控制的一条实用途径，但它们几乎完全建立在图像之上。潜在预测能否在几何观测中存活尚不明确：点云稀疏、无序且存在自遮挡，当场景中仅有0.3-15%的点发生移动时，潜在预测的慢特征最优性与3D自监督的几何捷径会相互叠加。我们将三种经典的JEPA设计——冻结编码器、分布先验和动作敏感——提升到点云上，并重新感知稳定世界模型基准，使得观测方式成为与图像基线之间唯一的差异。三种模型均能在不发生表征坍塌的情况下进行规划：分布先验模型在所有基准上与其重新评估后的图像对应模型统计等效，而动作敏感模型在几何移动最多的受控比较中取得了最强结果。探测实验解释了原因：物体位置几乎可以被完美地线性解码。

    arXiv:2608.29434v1 Announce Type: cross  Abstract: JEPA world models make latent-space planning a practical route to control, but they are built almost exclusively on images. Whether latent prediction survives geometric observations is unclear: point clouds are sparse, unordered, and self-occluded, and with 0.3-15% of scene points moving, the slow-feature optimum of latent prediction compounds with the geometric shortcut of 3D self-supervision. We lift three canonical JEPA designs to point clouds, frozen-encoder, distribution-prior, and action-sensitive, and re-sense the stable-worldmodel benchmark so that only the observation differs from the image baselines. All three plan without collapse: the distribution-prior model is statistically equivalent to its re-evaluated image counterpart on every benchmark, and the action-sensitive model attains the strongest result in our controlled comparison where the most geometry moves. Probing explains why: object positions are almost perfectly lin
    
[^303]: 不完全性悖论未必存在于大型语言模型中：模型失败之前的基准失败

    The Imperfective Paradox Is Not Necessarily in Large Language Models: A Benchmark Failure Before a Model Failure

    [https://arxiv.org/abs/2608.25005](https://arxiv.org/abs/2608.25005)

    本文指出现有NLI基准因概念和评估错误规范（如体貌缩减）而失效，导致大型语言模型在“不完全性悖论”测试中的表现被误判，并提出了改进的词汇匹配基准。

    

    arXiv:2608.25005v1 公告类型：新 摘要：不完全性悖论为组合语义分析提供了一个有用的测试。近期研究构建了一个NLI基准，并报告模型经常从进行体描述中推断出完成的终结事件，将此行为归因于“目的论偏差”。该研究进一步声称，提示干预会导致“校准危机”。我们重新审视了该基准和结论，并表明其受到概念性和评估性错误规范的显著影响。我们识别出三个概念性错误规范。特别是，“体貌缩减”影响了基准构建、分析、实验和结论。在严格的NLI标准下，A组实例中有76%并未明确排除终结性。在我们的母语者标注中，A组示例的38%和C组示例的29%被认为允许替代解释。为控制这些问题和词汇变异，我们构建了“词汇匹配”...

    arXiv:2608.25005v1 Announce Type: new  Abstract: The imperfective paradox provides a useful test of compositional semantic analysis. Recent work constructs an NLI benchmark and reports that models frequently infer completed telic events from progressive descriptions, attributing this behavior to a Teleological Bias. It further argues that prompting interventions cause a Calibration Crisis. We reexamine the benchmark and conclusions and show that it is substantially affected by conceptual and evaluation mis-specifications. We identify three conceptual mis-specifications. In particular, Aspectual Reduction affects the benchmark construction, analysis, experiments, and conclusions. Under a strict NLI standard, 76% of Group A instances do not explicitly rule out culmination. In our native-speaker annotation, 38% of Group A examples and 29% of the Group C examples were judged to permit an alternative interpretation. To control these issues and lexical variation, we construct Lexically Match
    
[^304]: qshap：梯度提升树的$R^2$快速Shapley分解

    qshap: Fast Shapley Decomposition of $R^2$ for Gradient-Boosted Trees

    [https://arxiv.org/abs/2608.24104](https://arxiv.org/abs/2608.24104)

    qshap是一种新工具，通过Shapley分解梯度提升树中的$R^2$值，实现特征和观测对模型性能贡献的全局量化。

    

    arXiv:2608.24104v1 公告类型：交叉 摘要：已有大量方法用于量化树集成模型中单个预测的特征归因。然而，许多应用需要特征对整体模型性能贡献的全局度量。尽管局部归因分数可以聚合以表征特征重要性，但此类汇总并不能直接分解预测性能的度量，如$R^2$。本文介绍了qshap，该工具同时提供R和Python版本，它实现了对梯度提升决策树（GBDTs）中$R^2$值的Shapley分解，以量化特征对模型性能的特定贡献。通过分解单个观测的二次损失，qshap提供了灵活的工具来探索单个特征和观测的重要性。qshap目前支持广泛使用的GBDT实现，包括xgboost、lightgbm和catboost，通过统一的树表示和高效的C++后端实现。

    arXiv:2608.24104v1 Announce Type: cross  Abstract: Numerous methods have been developed to quantify feature attributions in individual predictions for tree ensembles. However, many applications require global measures of feature contributions to overall model performance. Although local attribution scores can be aggregated to characterize feature importance, such summaries do not directly decompose measures of predictive performance, such as $R^2$. This article introduces qshap, available in both R and Python, which provides Shapley decomposition of $R^2$ values for gradient-boosted decision trees (GBDTs) to quantify feature-specific contributions to model performance. By decomposing the quadratic loss of individual observations, qshap provides flexible tools to explore the importance of individual features and observations. qshap currently supports widely used GBDT implementations, including xgboost, lightgbm, and catboost, through a unified tree representation and efficient C++ backe
    
[^305]: RIBOSPAN：一种用于多功能RNA建模的长上下文RNA基础模型

    RIBOSPAN: A Long-Context RNA Foundation Model for Versatile RNA Modeling

    [https://arxiv.org/abs/2608.22849](https://arxiv.org/abs/2608.22849)

    提出了RIBOSPAN——一个16.1亿参数、原生支持最长10,240核苷酸上下文的双向RNA基础模型，通过密集双向自注意力、单核苷酸分词和注意力隔离序列打包，实现了对完整长RNA的单核苷酸分辨率建模。

    

    全长RNA，尤其是信使RNA（mRNA），通常超出现有RNA基础模型预训练时所使用的上下文长度，限制了在单核苷酸分辨率下对完整转录本的建模。我们提出了RIBOSPAN，一个16.1亿参数的双向RNA基础模型，以最长10,240个核苷酸的上下文长度进行原生预训练。RIBOSPAN结合了密集双向自注意力机制、单核苷酸分词以及注意力隔离的序列打包技术，实现了对完整长RNA的高分辨率建模。原生10K预训练在10,240个token长度下保持了强大的重建能力，并在一个受控的长上下文基准测试中，维持了较强的上下文响应能力和上下文特异的表示分离，同时使扰动引起的变化保持高度局部化。推理阶段的YaRN扩展恢复了直接短上下文外推所丢失的大部分上下文组织结构，但会引起明显更大的距离……（原文摘要在此处截断）

    arXiv:2608.22849v2 Announce Type: replace  Abstract: Full-length RNAs, particularly messenger RNAs, often exceed the context lengths used to pretrain existing RNA foundation models, limiting complete-transcript modeling at single-nucleotide resolution. We present RIBOSPAN, a 1.61-billion-parameter bidirectional RNA foundation model natively pretrained with context lengths up to 10,240 nt. RIBOSPAN combines dense bidirectional self-attention, single-nucleotide tokenization, and attention-isolated sequence packing to enable high-resolution modeling of complete long RNAs. Native 10K pretraining preserves strong reconstruction at 10,240 tokens and, in a controlled long-context benchmark, maintains strong contextual responsiveness and context-specific representation separation while keeping perturbation-induced changes highly localized. Inference-time YaRN scaling recovers much of the contextual organization lost by direct short-context extrapolation, but induces substantially greater dista
    
[^306]: 基于可观测语义图像接口与层级生成器证据对齐的生成式语义分割

    Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment

    [https://arxiv.org/abs/2608.11537](https://arxiv.org/abs/2608.11537)

    提出了一种通过可观测语义图像接口和层级生成器证据对齐的生成式语义分割方法，有效解决了颜色漂移和边界混合问题。

    

    生成式语义分割将结构化预测呈现为图像，但直接的颜色解码容易受到颜色漂移和边界混合的影响，而预测独立输出分布的潜在特征解码器可能将渲染图像降级为中间可视化。我们提出了Semantic Prism，一个具有确定性推理的条件语义图像生成与细化框架。一个基于扩散蒸馏的单步生成器渲染语义RGB图像；渲染颜色到固定类颜色码本的逐像素距离定义了一个显式的概率接口。层级生成器证据对齐在空间上对齐多层级生成器特征，并使用零初始化输出投影在接口对数空间中预测加性残差，保留图像定义的接口作为最终分布的参考。该接口和细化分布进一步支持Conte（后续内容）。

    arXiv:2608.11537v1 Announce Type: cross  Abstract: Generative semantic segmentation exposes structured predictions as images, but direct color decoding is susceptible to color drift and boundary mixing, whereas latent-feature decoders that predict a separate output distribution may relegate the rendered image to an intermediate visualization. We present Semantic Prism, a conditional semantic-image generation-and-refinement framework with deterministic inference. A diffusion-distilled one-step generator renders a semantic RGB image; per-pixel distances from the rendered colors to a fixed class-color codebook define an explicit probabilistic interface. Hierarchical Generator Evidence Alignment spatially aligns multi-level generator features and uses a zero-initialized output projection to predict an additive residual in the interface logit space, retaining the image-defined interface as the reference for the final distribution. The interface and refined distributions further enable Conte
    
[^307]: CommitKV：面向多模态智能体的提交对齐KV缓存压缩

    CommitKV: Commit-Aligned KV Cache Compression for Multimodal Agents

    [https://arxiv.org/abs/2608.07855](https://arxiv.org/abs/2608.07855)

    CommitKV是一种无需训练的多模态智能体KV缓存压缩方法，通过对比工具调用前后对同一缓存页面的删除影响来捕捉证据使用的变化，淘汰两个时间窗口中影响力均低的页面并暂时保护被观察激活的页面，避免因注意力转移而误删后续需要的信息。

    

    多模态智能体通过推理和工具使用来收集视觉证据，将其解决任务的能力扩展到初始输入之外。然而，不断累积的观察结果会增加KV缓存的内存占用和推理成本。令牌驱逐（token eviction）通过保留有用的缓存令牌来降低这些成本。现有方法使用注意力、冗余度或特定阶段的查询来对令牌排序，却没有在工具交互之间显式比较同一历史记录。当观察结果使注意力发生转移时，这些方法可能将暂时降低的影响力误判为失去用处，从而驱逐掉后续需要的信息。我们提出了CommitKV，一种无需训练的KV缓存压缩方法，能够应对这种不断变化的证据使用方式。提交对齐测量在工具调用完成之前和其观察结果返回之后，比较对同一页面的删除影响。生命周期感知压缩会淘汰在两个时间窗口中影响力都很低的页面，并暂时保护影响力发生变化的页面……（原文在此截断）

    arXiv:2608.07855v2 Announce Type: replace  Abstract: Multimodal agents gather visual evidence through reasoning and tool use, extending their ability to solve tasks beyond the initial input. However, accumulating observations increases KV cache memory and inference costs. Token eviction reduces these costs by retaining useful cached tokens. Existing methods use attention, redundancy, or stage-specific queries to rank tokens, without explicitly comparing the same history across tool interactions. When observations shift attention, they may mistake temporarily reduced influence for lost usefulness and evict information needed later. We propose CommitKV, a training-free KV cache compression method that accounts for this changing evidence use. Commit-aligned measurement compares the deletion effects of the same page before a tool call completes and after its observation returns. Lifecycle-aware compression retires pages with low influence in both windows and temporarily protects pages infl
    
[^308]: 重尾噪声下的无参数动态遗憾

    Parameter-Free Dynamic Regret under Heavy-Tailed Noise

    [https://arxiv.org/abs/2607.27073](https://arxiv.org/abs/2607.27073)

    该论文提出了一种无需预知噪声矩参数、Lipschitz 界、梯度尺度及比较序列路径长度的在线凸优化算法，通过重启 AdaGrad 专家与自适应熵正则化主算法的结合，在重尾噪声下实现了具有噪声路径指数 $(p-1)/p$ 的最优通用动态遗憾界。

    

    我们研究在线凸优化问题，其中随机梯度噪声的条件 $p$ 阶中心矩被 $\sigma^p$ 所界定，而 $p\in(1,2]$ 是未知的。对于在直径为 $D$ 的定义域上具有 Lipschitz 界 $G$ 的损失函数，我们获得了期望通用动态遗憾界 $\widetilde O(GD\sqrt{T\Lambda}+\sigma DT^{1/p}\Lambda^{(p-1)/p})$，其中 $\Lambda=1+P_T/D$，$P_T$ 是固定比较序列的路径长度。该算法将重启的 AdaGrad 专家与自适应熵正则化主算法相结合，每轮仅使用一次随机梯度，且不需要知道 $G$、$\sigma$、$p$ 或 $P_T$ 的任何信息。其迭代点在梯度的正数缩放变换下保持不变。该分析在取期望之前控制重启块内比较序列的移动，从而得到噪声路径指数 $(p-1)/p$，而非直接非重启扩展所给出的指数 $1/2$。同时给出了一个匹配的随机一阶预言机下界……

    arXiv:2607.27073v2 Announce Type: replace-cross  Abstract: We study online convex optimization with stochastic gradient noise whose conditional $p$-th central moment is bounded by $\sigma^p$, for an unknown $p\in(1,2]$. For losses with Lipschitz bound $G$ on a domain of diameter $D$, we obtain expected universal dynamic regret $\widetilde O(GD\sqrt{T\Lambda}+\sigma DT^{1/p}\Lambda^{(p-1)/p})$, where $\Lambda=1+P_T/D$ and $P_T$ is the path length of a fixed comparator sequence. The algorithm combines restarted AdaGrad experts with an adaptive entropy-regularized master, uses one stochastic gradient per round, and requires no knowledge of $G,\sigma,p$, or $P_T$. Its iterates are invariant under positive rescaling of the gradients. The analysis controls comparator movement within restart blocks before taking expectations, yielding the noise path exponent $(p-1)/p$ rather than the exponent $1/2$ of a direct non-restarted extension. A matching stochastic first-order oracle lower bound, comb
    
[^309]: 重新思考自我进化：一种缓解技能过拟合的受限探索-利用过程

    Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting

    [https://arxiv.org/abs/2607.26643](https://arxiv.org/abs/2607.26643)

    SkillBoost通过结构化利用和先验引导探索的受约束框架，有效平衡探索与利用，从而缓解LLM代理技能自我进化中的过拟合问题。

    

    摘要：使大型语言模型（LLM）代理能够从过去的交互中积累和重用经验，在现实世界应用中仍是一个核心挑战。一个有前景的解决方案是将技能视为可训练状态，并以类似于神经网络训练中模型参数优化的方式对其进行优化。然而，数据驱动的技能优化容易过度拟合从真实环境中收集的有限轨迹。过度利用这些轨迹会导致对当前批次的过拟合，而无约束的探索则会导致先前已解决问题上的性能退化。这种张力促使我们提出一种受约束的搜索视角来审视技能自我进化，该过程由探索-利用权衡所主导。我们提出了SkillBoost，一个三阶段框架，用以缓解这两种风险：结构化利用将观察到的失败定位到可编辑的技能组件，先验引导的探索利用LLM中的先验知识生成多样的修复候选方案。

    arXiv:2607.26643v2 Announce Type: replace  Abstract: Enabling large language model (LLM) agents to accumulate and reuse experience from past interactions remains a central challenge in real-world applications. A promising solution is to treat skills as trainable states and optimize them in the same way as model parameters in neural network training. However, data-driven skill optimization is prone to overfitting to the limited trajectories collected from real environments. Overexploiting these trajectories overfits the current batch, while unconstrained exploration causes regression on previously solved cases. This tension motivates a constrained search view of skill self-evolution, governed by an exploration--exploitation trade-off. We propose SkillBoost, a three-stage framework that mitigates both risks: structured exploitation localizes observed failures to editable skill components, prior-guided exploration draws on prior knowledge in the LLM to generate diverse repair candidates, 
    
[^310]: 后验方差是约束图，而非误差图：稀疏视图CT中辐射高斯泼溅的闭式不确定性

    Posterior Variance Is a Constraint Map, Not an Error Map: Closed-Form Uncertainty for Radiative Gaussian Splatting in Sparse-View CT

    [https://arxiv.org/abs/2607.13682](https://arxiv.org/abs/2607.13682)

    本文证明辐射高斯泼溅CT重建中的后验方差图本质上是数据约束图而非误差图（其警报可信但“无异常”信号不可信），并推导出无需修改光栅化器、一次前向传播即可精确计算的截断感知闭式不确定性（成本约为采样法的1/8），同时发现该不确定性在临床关注的物体内部区域会失去对真实误差的排序能力。

    

    辐射高斯泼溅能够快速且准确地重建稀疏视图CT，近期工作通过为每个高斯附加后验分布来生成逐体素的不确定性图。我们追问这样的图究竟度量了什么：后验方差是一种数据约束图，而非误差图——它的警报是可信的，但它的“一切正常”并不可信。利用X射线渲染对每个高斯密度的严格线性性，我们推导出一个考虑截断的闭式解，无需修改的光栅化器即可在一次前向传播中精确求值，并同时给出体空间和投影空间的结果：它是同期工作采样估计器的无限样本极限，而成本降低约8倍。在官方15场景基准测试中，该不确定性在15个场景中的14个上能够对真实误差进行排序。然而，当限制在物体内部——即临床医生所读取的组织——排序能力崩溃（中位Spearman相关系数为0.11，0/15场景通过），且对于深度集成和严格正的log-normal后验结果完全相同：三个常…

    arXiv:2607.13682v3 Announce Type: replace-cross  Abstract: Radiative Gaussian splatting reconstructs sparse-view CT fast and accurately, and recent work attaches per-Gaussian posteriors to yield per-voxel uncertainty maps. We ask what such a map actually measures: posterior variance is a data-constraint map, not an error map -- its alarms are trustworthy, its all-clears are not. Exploiting the strict linearity of X-ray rendering in the per-Gaussian densities, we derive a clamp-aware closed form that the unchanged rasterizer evaluates exactly in one forward pass, in volume and projection space: the infinite-sample limit of the sampling estimator of concurrent work, at ~8x lower cost. On the official 15-scene benchmark this uncertainty ranks true error on 14 of 15 scenes. Restricted to the object interior -- the tissue a clinician reads -- the ranking collapses (median Spearman 0.11, 0/15 pass), identically for a deep ensemble and for a strictly positive log-normal posterior: three const
    
[^311]: 基于CVAE的胶质瘤IDH分类：拉曼活检的合成数据增强、生成式分类与多实例学习

    Glioma IDH Classification via CVAE: Synthetic Augmentation, Generative Classification, and MIL for Raman Biopsies

    [https://arxiv.org/abs/2607.10196](https://arxiv.org/abs/2607.10196)

    该研究利用β-CVAE同时解决拉曼活检数据集样本量小和缺乏组织标注两大难题，通过合成数据增强、生成式分类和多实例学习，将胶质瘤IDH状态二分类准确率提升至0.83。

    

    用于胶质瘤诊断机器学习的生物医学拉曼数据集的特点是队列规模小且缺乏诊断组织标注。我们使用β正则化条件变分自编码器（β-CVAE）以及一个包含来自46名患者的59个胶质瘤活检样本的数据集来解决这两个问题，用于IDH状态的二分类。β-CVAE被用作：(1) 拉曼光谱生成器；(2) 拉曼光谱的生成式分类器；(3) 活检切片的多实例学习（MIL）分类器。四个分类器分别在真实光谱、合成光谱或两者组合的数据集上训练，并在留出的真实切片上进行评估。随机森林和XGBoost在合成数据训练下达到0.83的准确率，在真实数据训练下达到0.81。数据增强使RF+SVM的准确率从0.76提升至0.83；其他分类器的变化较小。我们证明了β-CVAE模型可以成功用于训练IDH状态分类器。

    arXiv:2607.10196v2 Announce Type: replace  Abstract: Biomedical Raman datasets used in machine learning for glioma diagnostics are characterized by small cohort size and absence of diagnostic tissue annotations. We address both problems using a $\beta$-regularized conditional variational autoencoder and a dataset of 59 glioma biopsies from 46 patients for binary IDH-status classification. The $\beta$-CVAE was used as a: (1) Raman spectra generator; (2) generative classifier of Raman spectra; (3) Multiple Instance Learning (MIL) classifier of biopsy slides. Four classifiers were trained on real spectra, synthetic spectra, or a combined set and evaluated on held-out real slides. Random Forest and XGBoost achieved accuracy $0.83$ with synthetic training and $0.81$ with real training. Augmentation increased RF+SVM accuracy from $0.76$ to $0.83$; changes for the other classifiers were smaller. We show that the $\beta$-CVAE model can successfully be used in training IDH-status classifiers an
    
[^312]: 基于广义泊松流的可变长度生成式蛋白质设计

    Variable-Length Generative Protein Design via Generalized Poisson Flow

    [https://arxiv.org/abs/2607.09039](https://arxiv.org/abs/2607.09039)

    提出广义泊松流（GPFlow），一个无需预先指定蛋白质长度即可进行生成的可变长度框架，通过学习非齐次广义泊松过程的强度函数，在结构和序列设计、基序支架及肽段协同设计等任务上实现了跨多种模态的蛋白质设计。

    

    生成可变长度蛋白质的能力在蛋白质设计中至关重要，因为最优蛋白质长度往往是未知的，并且与可设计性紧密相关。当前的扩散模型和基于流的生成模型通常要求在采样前预先指定蛋白质长度，这限制了它们在探索可行设计空间方面的灵活性。为了解决这一局限，我们提出了广义泊松流，这是一个可变长度的生成框架，通过最小化负对数似然来学习非齐次广义泊松过程的强度函数。我们为恢复联合多模态分布建立了总体层面的理论保证，并推导出了数据分布与生成分布之间KL散度的上界。我们在结构设计、序列设计、基序支架构建以及肽段协同设计等任务上对GPFlow进行了全面评估，涵盖了欧几里得、类别型和黎曼流形等多种模态，以充分发挥该框架的通用性。

    arXiv:2607.09039v2 Announce Type: replace  Abstract: The ability to generate variable-length proteins is crucial in protein design, where the optimal length is often unknown and tightly coupled to designability. Current diffusion- and flow-based generative models typically require the protein length to be specified before sampling, limiting their flexibility in exploring the feasible design space. To address this limitation, we introduce Generalized Poisson Flow (GPFlow), a variable-length generative framework that learns the rate function of an inhomogeneous generalized Poisson process by minimizing its negative log-likelihood. We establish population-level guarantees for recovering the joint multimodal distribution and derive an upper bound on the KL divergence between the data and generated distributions. We comprehensively evaluate GPFlow across structure and sequence design, motif scaffolding, and peptide co-design, spanning Euclidean, categorical, and Riemannian modalities to ful
    
[^313]: 少量教师步骤大有裨益：面向智能体后训练的高性价比同策略数据增强

    A Few Teacher Steps Go a Long Way: Cost-Efficient On-Policy Data Augmentation for Agent Post-Training

    [https://arxiv.org/abs/2607.04574](https://arxiv.org/abs/2607.04574)

    该论文将智能体后训练中的同策略数据构建形式化为一个预算分配问题，系统权衡教师演示、续写长度、过滤规则与上下文覆盖等设计维度，发现少量教师步骤即可带来显著收益，从而实现高性价比的数据增强。

    

    对于大语言模型智能体而言，监督微调不仅关乎教师标签的质量，还关乎这些标签所条件化的交互上下文。纯粹的行为克隆使用完整的教师演示，导致训练中所见的教师诱导上下文与测试时所遇到的学生诱导上下文之间存在不匹配。近期工作通过在学生到达的上下文处查询教师来解决这一不匹配问题，并往往对教师的续写内容进行日益精细的过滤。我们则将同策略数据构建视为一个预算分配问题：在监督资源匹配的情况下，教师输出应该用于更多的从头到尾的完整演示、更长的续写、基于结果的过滤，还是更广泛地覆盖学习者诱导的上下文？我们通过展开策略、切换时间分布、续写视界、过滤规则，以及两种互补的成本——教师推理（成本）……来将这一设计空间形式化。

    arXiv:2607.04574v2 Announce Type: replace-cross  Abstract: For LLM agents, supervised fine-tuning is not only about teacher labels' quality, but also about which interaction contexts those labels condition on. Pure behavioral cloning uses full teacher demonstrations, creating a mismatch between teacher-induced contexts seen in training and student-induced contexts encountered at test time. Recent work addresses this mismatch by querying a teacher at contexts reached by the student, often with increasingly elaborate filtering of the teacher's continuations. We instead frame on-policy data construction as a budget-allocation problem: under matched supervision resources, should teacher output be spent on more start-to-finish demos, longer continuations, outcome filtering, or broader coverage of learner-induced contexts? We formalize this design space through the rollout policy, switch-time distribution, continuation horizon, filtering rules, and two complementary costs: teacher inference 
    
[^314]: 相同概念，不同方向：稀疏自编码器中的跨模态特征异质性

    Same Concept, Different Directions: Cross-Modal Feature Heterogeneity in Sparse Autoencoders

    [https://arxiv.org/abs/2606.29888](https://arxiv.org/abs/2606.29888)

    本文挑战了稀疏自编码器应用于联合嵌入空间时“语义对应特征跨模态共享相同方向”的隐含假设，首次刻画了同一概念在图像与文本模态间的特征方向异质性现象，并证明这是导致模态分裂的关键驱动因素。

    

    视觉-语言模型将图像和文本映射到一个联合嵌入空间。然而，这些嵌入往往纠缠了多种语义特征，这限制了它们的可解释性和可控性。虽然稀疏自编码器已成为将这些嵌入分解为单语义特征的有效工具，但其在联合嵌入空间中的应用在很大程度上依赖于一个隐含且未经检验的假设：即语义对应的特征在不同模态间共享相同的方向。本文通过刻画同一概念在图像和文本模态间特征方向的差异来挑战这一假设，我们将这一现象称为跨模态特征异质性。我们证明这种异质性是模态分裂现象的关键驱动因素，即同一共享概念会因模态不同而激活不同的潜在特征。这一发现进一步揭示了为什么仅对潜在激活进行对齐是不够的。

    arXiv:2606.29888v2 Announce Type: replace  Abstract: Vision-language models map images and text into a joint embedding space. However, these embeddings often entangle multiple semantic features, which limits their interpretability and controllability. While sparse autoencoders have emerged as a useful tool for decomposing these embeddings into monosemantic features, their application to joint embedding spaces has largely relied on an implicit, untested assumption that semantically corresponding features share the same directions across modalities. In this paper, we challenge this assumption by characterizing discrepancies in feature directions for the same concept across image and text modalities, a phenomenon we term cross-modal feature heterogeneity. We demonstrate that this heterogeneity is a key driver of the modality split, where a shared concept activates different latents depending on the modality. This finding further reveals why aligning latent activations alone is insufficien
    
[^315]: RecallRisk-BERT：面向报告后医疗器械召回分诊的多任务框架

    RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage

    [https://arxiv.org/abs/2606.27174](https://arxiv.org/abs/2606.27174)

    本文提出了一个名为RecallRisk-BERT的多任务框架，通过联合建模召回严重性与根本原因类别，实现了对医疗器械报告后召回记录的高效自动分诊，填补了现有研究在联合预测方面的空白。

    

    医疗器械召回是保护患者安全的关键监管机制。美国食品药品监督管理局（FDA）召回记录数量的不断增长，给报告后的召回分诊、严重性评估及根本原因解释带来了挑战。现有研究大多分别处理召回发生预测或根本原因分析，而对召回严重性与根本原因类别的联合建模关注有限。我们利用来自openFDA的54,165条FDA医疗器械召回记录（覆盖2002年至2025年10月期间），开发了一个自动化召回分诊框架。我们首先评估了经典机器学习模型和基于提升的模型在召回严重性和根本原因类别预测上的表现。随后，我们开发了RecallRisk-BERT，这是一个多任务模型，它结合了基于PubMedBERT的召回叙述文本表示与基于嵌入的结构化分类特征（包括产品代码、法规编号等）表示。

    arXiv:2606.27174v1 Announce Type: new  Abstract: Medical device recalls are a critical regulatory mechanism for protecting patient safety. The growing volume of FDA recall records presents challenges in post-report recall triage, severity assessment, and root-cause interpretation. Existing studies mostly address recall occurrence prediction or root-cause analysis separately, while joint modeling of recall severity and root-cause categories has received limited attention. We develop an automated recall triage framework using 54,165 FDA medical device recall records from openFDA, covering the period from 2002 to October 2025. We first evaluate classical machine learning and boosting-based models for recall severity and root-cause category prediction. We then develop RecallRisk-BERT, a multi-task model that combines PubMedBERT-based textual representations of recall narratives with embedding-based representations of structured categorical features, including product code, regulation numbe
    
[^316]: LLM-ACES：基于大语言模型引导自适应搜索的动力系统闭环发现

    LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search

    [https://arxiv.org/abs/2606.25039](https://arxiv.org/abs/2606.25039)

    LLM-ACES提出了一种由大语言模型引导的闭环方程发现框架，通过LLM生成算子先验来缩小符号搜索空间，并联合优化假设构建与自适应数据采集，利用候选方程间的分歧指导新数据获取，从而消除可辨识性缺口、准确恢复动力系统的控制方程。

    

    从数据中恢复控制性常微分方程（ODE）是对各科学领域动力系统建模的核心挑战。现有方法将方程发现视为固定数据集上的静态推断问题，并假设观测到的轨迹包含足够的信息。然而，动力系统在广阔的状态空间上演化，有限的数据可能使多个方程在观测上不可区分，从而导致可辨识性缺口以及错误控制方程的恢复。为解决这一问题，我们提出了LLM-ACES（大语言模型引导的主动闭环方程搜索），这是一个联合优化符号假设构建与自适应数据采集的闭环框架。在LLM-ACES中，大语言模型（LLM）提出算子先验，这些先验定义了符号搜索空间的受约束子空间，候选方程在该子空间内对观测数据进行拟合。（原文摘要在此处截断）

    arXiv:2606.25039v2 Announce Type: replace-cross  Abstract: Recovering governing ordinary differential equations (ODEs) from data is a central challenge in modeling dynamical systems across scientific domains. Existing approaches cast discovery as a static inference problem over fixed datasets, assuming that the observed trajectories are sufficiently informative. However, dynamical systems evolve over large state spaces, and limited data can make multiple equations observationally indistinguishable, leading to identifiability gaps and the recovery of incorrect governing equations. To address this, we introduce LLM-ACES, or LLM-guided Active Closed-loop Equation Search, a closed-loop framework that jointly optimizes symbolic hypothesis construction and adaptive data acquisition. In LLM-ACES, a large language model (LLM) proposes operator priors that define constrained subspaces of the symbolic search space, within which candidate equations are fit to the observed data. The disagreement a
    
[^317]: 将历史压缩进记忆：将Transformer蒸馏为循环Transformer

    Compressing History into Memory: Distilling Transformers into Recurrent Transformers

    [https://arxiv.org/abs/2606.21562](https://arxiv.org/abs/2606.21562)

    本文提出一种蒸馏方法，将处理完整历史的Transformer所学会的压缩策略迁移到循环Transformer中，从而在无需存储观测历史的情况下，显著提升循环模型在长时程流式视觉与机器人任务中的性能。

    

    arXiv:2606.21562v2 公告类型：replace-cross。摘要：Transformer是人工智能领域的主力模型，但其在处理长序列时的计算成本会变得难以承受。我们面向长时程流式视觉与机器人应用，在这类场景中存储和维护观测历史尤其不切实际。循环Transformer通过维护固定大小的记忆来解决这一局限，但其性能落后于在完整观测历史上进行运算的Transformer。我们认为这一差距并非源于架构上的限制，而是源于这些模型学习压缩过去信息的方式不同。由于无法访问观测历史，循环模型必须在每一步显式决定应在记忆中保留什么，这是一个难度大得多的学习问题。在这项工作中，我们提出了一种蒸馏方法，将经典的全历史Transformer的压缩策略迁移到循环变体中。我们通过设计一个教师模型来实现这一目标……（原文在此处截断）

    arXiv:2606.21562v2 Announce Type: replace-cross  Abstract: Transformers are AI's workhorse but their computational cost becomes prohibitive when processing long sequences. We target long-horizon streaming vision and robotics applications, where it is particularly impractical to store and maintain a history of observations. Recurrent Transformers address this limitation by maintaining fixed-size memory but their performance lags behind that of transformers operating over the full observation history. We argue that this gap does not stem from architectural limitations, but from differences in how these models learn to compress past information. Without access to an observation history, recurrent models must explicitly decide what to retain in memory at each step, a significantly harder learning problem. In this work, we propose a distillation approach that transfers the compression strategy of a classical full-history transformer to a recurrent variant. We enable this by designing a teac
    
[^318]: 语言智能体中开放式多智能体协作的基准测试

    Benchmarking Open-Ended Multi-Agent Coordination in Language Agents

    [https://arxiv.org/abs/2606.08340](https://arxiv.org/abs/2606.08340)

    提出了基于 JAX 的开放式多智能体协作基准 Alem，将程序化协作任务、软专业化和通信嵌入长时程生存世界中，评估显示当前 LLM 智能体平均归一化回报仅约 6%，远未解决该基准。

    

    随着语言模型越来越多地被部署为自主智能体，它们必须在开放式交互任务中与他人进行长时程协作。然而，现有评估很少将这些需求结合在一起进行测试，而是侧重于单智能体任务、短时交互或高度结构化的多智能体设置。我们提出了 alem，一个基于 JAX 构建的开放式多智能体协作基准，其建立在类 Craftax 的动态机制之上。Alem 将程序化生成的协作任务、软专业化、通信以及可控的协作难度嵌入到一个包含探索、制作、交易和战斗的长时程生存世界中。我们在同质团队中对 13 个大语言模型进行零样本评估，并以训练好的 MARL（多智能体强化学习）智能体作为参考基准。当前的 LLM 智能体远未能解决 alem，平均归一化回报仅约 6%，但它们的失败表现并不一致。在最困难的协作设置中，零样本的 Gemini-3.1-Pro-High 接近……

    arXiv:2606.08340v2 Announce Type: replace  Abstract: As language models are increasingly deployed as autonomous agents, they must coordinate with others over long horizons in open-ended interactive tasks. Yet existing evaluations rarely test these demands together, focusing instead on single-agent tasks, short interactions, or highly structured multi-agent settings. We introduce $alem$, a JAX-based benchmark for open-ended multi-agent coordination built on Craftax-like dynamics. Alem embeds procedurally generated coordination tasks, soft specialisation, communication, and controllable coordination difficulty into a long-horizon survival world with exploration, crafting, trading, and combat. We evaluate $13$ LLMs zero-shot within homogeneous teams, with trained MARL agents as reference points. Current LLM agents remain far from solving alem, averaging only ~6% normalised return, but their failures are not uniform. On the hardest coordination setting, zero-shot Gemini-3.1-Pro-High approa
    
[^319]: FLIPS：基于伪随机序列的大语言模型实例指纹识别

    FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences

    [https://arxiv.org/abs/2606.03330](https://arxiv.org/abs/2606.03330)

    本文提出面向AI监管的“实例级指纹识别”新范式及方法FLIPS，通过分析模型生成的二进制伪随机序列中的偏差，来区分同一大语言模型在不同实例级配置（如提示词、采样设置、量化方式）下的部署实例。

    

    文献表明，大语言模型（LLM）的行为不仅取决于其原始权重，还取决于其实例级参数，例如指令提示词、采样配置或量化方式。一个在某种配置下生成安全输出的模型，在另一种配置下可能会产生有害内容。然而，当前的LLM识别技术（如模型指纹识别）侧重于知识产权保护，其设计倾向于对这些实例级参数的变化保持鲁棒性。这给AI监管带来了严峻挑战，因为合规评估针对的是实际部署的行为，而非模型的来源。本文提出了实例级指纹识别，这是一种面向监管者的新范式，用于区分同一LLM的不同配置。我们的方法FLIPS利用了模型生成的二进制随机序列中的偏差。在包含237个模型实例的基准测试中，它能够识别出实例……

    arXiv:2606.03330v2 Announce Type: replace-cross  Abstract: Literature reveals that a Large Language Model's (LLM) behavior is not only conditioned by its original weights but also its instance-level parameters, such as instructional prompt, sampling configuration or quantization. A model that generates safe outputs under one configuration may produce toxic content under another. However, current LLM identification techniques (such as fingerprinting) focus on intellectual property protection, and their design favors robustness to changes in these instance-level parameters. This poses a critical challenge for AI regulation in which compliance assessments target actual deployed behaviors, not model provenance. In this paper, we introduce instance-level fingerprinting, a regulator-oriented paradigm that distinguishes configurations of the same LLM. Our method, FLIPS, exploits biases in generated binary random sequences. On a benchmark of 237 model instances, it identifies the instance with
    
[^320]: 推测解码与多语言诅咒

    Speculative Decoding and the Curse of Multilinguality

    [https://arxiv.org/abs/2605.30580](https://arxiv.org/abs/2605.30580)

    本研究验证了推测解码在低资源语言上的效果因小模型多语言能力有限而大打折扣，且通过蒸馏提升草稿模型多语言能力的方法因任务间泛化差以及低资源语言难以构建通用数据集而不可行。

    

    推测解码（speculative decoding）是大语言模型（LLM）推理中一种流行的技术，通过使用较小的草稿模型起草多个词元来实现更快的生成。然而，推测解码的有效性此前主要针对英语进行了研究。受“多语言诅咒”的启发，我们假设由于较小模型的多语言能力有限，推测解码在低资源语言上的效果要差得多。我们在标准推测解码设置下测试了九种低资源语言和四种高资源语言，并为我们的假设找到了适度的证据支持。接下来，我们尝试通过从较大模型进行蒸馏来提升较小草稿模型的多语言能力。然而，我们发现蒸馏在同一语言的不同任务之间泛化能力较差，并且我们认为为低资源语言构建一个与任务无关的、完全具有代表性的数据集是不可行的。最后……

    arXiv:2605.30580v3 Announce Type: replace  Abstract: Speculative decoding is a popular technique for large language model (LLM) inference, enabling faster generation by drafting multiple tokens with a smaller draft model. However, the effectiveness of speculative decoding has mainly been studied for English. Motivated by the curse of multilinguality, we hypothesize that speculative decoding is far less effective for low-resource languages due to the limited multilingual capacities of smaller models. We test nine low-resource languages and four high-resource languages under a standard speculative decoding setup and find moderate evidence for our hypothesis.   Next, we try to improve the multilingual capabilities of the smaller draft model via distillation from the larger model. We find, though, that distillation generalizes poorly across tasks in the same language, and we argue that assembling a task-agnostic, fully representative dataset is infeasible for low-resource languages. Finall
    
[^321]: 结构化提示优化推进复杂文本的全局与局部可解释学习

    Structured Prompt Optimization Advances Globally and Locally Interpretable Learning from Complex Text

    [https://arxiv.org/abs/2605.29076](https://arxiv.org/abs/2605.29076)

    eXTC 通过结构化提示优化学习自然语言规则手册（SOP），结合推理蒸馏与强化学习，实现了兼具快速推理、局部推理轨迹与全局模块化可解释性的高性能文本分类。

    

    大语言模型已推动文本分类取得进展，但现有范式面临权衡：仅基于标签的监督微调具有可扩展性，但对复杂文本的推理能力有限，且缺乏更广泛的模型透明度；而离散提示优化虽能提供人类可读的指令，却在性能与可扩展性上存在困难。我们提出了 eXTC（可解释文本分类器），其核心是以自然语言学习得到的标准操作规程（SOP，或称规则手册），分为三个阶段：（1）通过新的结构化提示优化算法学习 SOP；（2）将基于 SOP 的推理从大型教师 LLM 蒸馏到紧凑语言模型中；（3）通过强化学习将推理能力扩展至 SOP 之外。这一设计使 eXTC 能够提供：（i）借助紧凑语言模型实现快速推理；（ii）推理时的局部推理轨迹，以及对其所学领域规则的全局模块化解释；同时（iii）在性能上取得显著提升……

    arXiv:2605.29076v3 Announce Type: replace-cross  Abstract: LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier), centered on a Standard Operating Procedure (SOP, or rulebook) learned in natural language, in three stages: (1) learning the SOP via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significan
    
[^322]: MobileMoE：扩展端侧混合专家模型

    MobileMoE: Scaling On-Device Mixture of Experts

    [https://arxiv.org/abs/2605.27358](https://arxiv.org/abs/2605.27358)

    提出MobileMoE端侧MoE语言模型系列，通过端侧MoE缩放定律确定内存与计算同时最优的架构（适度稀疏+细粒度专家+共享专家），在十亿以下激活参数规模上为端侧大语言模型建立了新的帕累托前沿。

    

    混合专家已成为千亿参数语言模型的事实标准架构，但其在十亿以下规模用于端侧部署的优势在很大程度上尚未被探索。为弥补这一空白，我们提出了MobileMoE，这是一系列具有十亿以下激活参数的端侧MoE语言模型（激活参数0.3-0.9B，总参数1.3-5.3B），为端侧大语言模型建立了新的帕累托前沿。我们首先制定了一个端侧MoE缩放定律，在移动设备内存和计算约束下联合优化MoE架构，确定了端侧最佳设计点——采用细粒度专家和共享专家的适度稀疏性——同时实现内存最优和计算最优。基于所推导的架构，我们采用涵盖预训练、中期训练、指令微调和量化感知训练的四阶段配方来训练MobileMoE，且全部使用开源数据集。在14个基准测试中，MobileMoE达到了……

    arXiv:2605.27358v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) has become the de facto architecture for hundred-billion-parameter language models, yet its advantages at sub-billion scales for on-device deployment remain largely unexplored. To close this gap, we present MobileMoE, a family of on-device MoE language models with sub-billion active parameters (0.3-0.9B active and 1.3-5.3B total) that establish a new Pareto frontier for on-device LLMs. We first formulate an on-device MoE scaling law that jointly optimizes MoE architecture under mobile memory and compute constraints, identifying an on-device sweet spot - moderate sparsity with fine-grained and shared experts - that is simultaneously memory and compute-optimal. Building on the derived architectures, we train MobileMoE with a four-stage recipe covering pre-training, mid-training, instruction fine-tuning, and quantization-aware training, all on open-source datasets. Across 14 benchmarks, MobileMoE matches o
    
[^323]: 基于随机搜索与迭代模仿的预言机预算蛋白质设计

    Oracle-Budgeted Protein Design with Stochastic Search and Iterative Imitation

    [https://arxiv.org/abs/2605.26690](https://arxiv.org/abs/2605.26690)

    SILO是一种轨迹级蛋白质设计方法，结合层次化编辑策略与随机束搜索探索多样化突变轨迹，利用UCB代理集成与丙氨酸扫描式适应度评分筛选候选序列，并通过交叉熵模仿学习迭代自我改进，在严格的预言机评估预算下高效优化蛋白质序列。

    

    在严格的预言机（评估）预算下优化蛋白质序列，需要能够探索庞大组合突变搜索空间的方法。这些方法还必须在代理模型噪声存在的情况下保持稳健，并在预测替换具有有利效果的位点上提出突变。我们提出了SILO（自我改进模仿学习优化），一种面向预言机预算蛋白质设计的轨迹级方法。SILO将层次化编辑策略与无放回的增量随机束搜索（SBS）相结合，以探索多样化的候选轨迹。基于UCB的代理模型集成，辅以受丙氨酸扫描启发的代理衍生适应度评分（AFS），指导候选选择以进行计算机内的预言机评估。随后，通过对当轮最佳预言机标注轨迹进行下一动作交叉熵模仿来更新策略。在八个计算机模拟蛋白质适应度景观和五个强基线方法上（原文摘要在此处截断）……

    arXiv:2605.26690v2 Announce Type: replace-cross  Abstract: Optimizing protein sequences under tight oracle budgets requires methods that can explore a vast combinatorial mutational search space. These methods must also remain robust in the presence of surrogate noise and propose mutations on positions where substitutions are predicted to have favorable effects. We introduce SILO (Self-improvement Imitation Learning Optimization), a trajectory-level approach for oracle-budgeted protein design. SILO combines a hierarchical edit policy with incremental stochastic beam search without replacement (SBS) to explore diverse candidate trajectories. A UCB-based proxy ensemble, complemented by a proxy-derived alanine-scan-inspired fitness score (AFS), guides candidate selection for in silico oracle evaluation. The policy is then updated by next-action cross-entropy imitation on the round's best oracle-labeled trajectories. Across eight in silico protein fitness landscapes and five strong baseline
    
[^324]: NPSolver：基于迭代物理监督的神经泊松求解器

    NPSolver: Neural Poisson Solver with Iterative Physics Supervision

    [https://arxiv.org/abs/2605.25786](https://arxiv.org/abs/2605.25786)

    NPSolver 提出了一种无需解标签的神经泊松求解器，通过少量预条件共轭梯度（PCG）迭代步精炼自身预测作为迭代物理监督，获得了比原始 PDE 残差更稳定、尺度更合适的训练信号。

    

    在复杂、不规则区域上高效求解泊松方程仍然是科学计算中的一项根本性挑战，因为传统迭代求解器常因系统病态而导致运行时间过长。虽然神经算子提供了一种快速的替代方案，但它们通常依赖于大规模标注数据集，或者在使用物理信息残差损失时面临训练动态不稳定的问题。我们提出了 NPSolver，一种通过迭代物理监督、无需解标签进行训练的神经泊松求解器。NPSolver 既不依赖完全收敛的数值解，也不使用原始 PDE 残差，而是利用少量预条件共轭梯度（PCG）迭代步来精炼其自身的预测，从而提供更稳定、尺度更合适的训练信号。理论分析证实，这种迭代监督充当了一个条件良好的误差代理，并且停止梯度设计是必不可少的。

    arXiv:2605.25786v2 Announce Type: replace-cross  Abstract: Efficiently solving Poisson equations on complex, irregular domains remains a fundamental challenge in scientific computing, as classical iterative solvers often suffer from prohibitive runtime due to ill-conditioned systems. While neural operators offer a fast alternative, they typically rely on large-scale labeled datasets or struggle with unstable training dynamics when using physics-informed residual losses. We propose \textsc{NPSolver}, a neural Poisson solver trained without solution labels via iterative physics supervision. Instead of relying on fully converged numerical solutions or raw PDE residuals, \textsc{NPSolver} utilizes a small number of preconditioned conjugate gradient (PCG) steps to refine its own predictions, providing a more stable and well-scaled training signal. Theoretical analysis confirms that this iterative supervision serves as a well-conditioned error proxy and that a stop-gradient design is essenti
    
[^325]: 复制、预测、编码其余部分：基于解码端邻居预测的无损图压缩

    Copy, Predict, Code the Rest: Lossless Graph Compression with Decoder-Side Neighbor Prediction

    [https://arxiv.org/abs/2605.21510](https://arxiv.org/abs/2605.21510)

    该论文提出了一种利用解码端邻居预测（复制已有邻居、预测剩余目标并对余下部分编码）的有向图无损压缩器，在基准图上实现了比WebGraph、Zuckerli和HyperFast更小的压缩文件，但代价是解码速度显著变慢。

    

    我们提出了一种针对有向图的无损压缩器。它从较早的行复制邻居，从图已解码的部分预测更多目标，并对剩余部分进行编码。其主要编码器——候选位编码器（CB）——提出一个有界的候选目标集合，自适应地混合它们的概率，并使用一个考虑流行度的模型对剩余目标进行编码。编码器尝试固定的顶点排序和存储方向集合，并保留能够精确解码的最小文件。在开发集之外的基准图上，所选文件比记录在案的最好的WebGraph和Zuckerli文件以及已发表的最好的HyperFast结果都要小。这些节省是有代价的：CB的解码速度远慢于基线方法，且仅支持顺序访问。一个更快的伴随编码器CG在大图上仍然比基线慢。一个带索引的变体CS-RA提供了比测量……（可检索）文件更小的可检索文件。

    arXiv:2605.21510v3 Announce Type: replace-cross  Abstract: We present a lossless compressor for directed graphs. It copies neighbors from an earlier row, predicts further targets from the part of the graph already decoded, and codes what remains. Its main coder, the candidate-bit coder (CB), proposes a bounded set of candidate targets, mixes their probabilities adaptively, and codes the leftover targets with a popularity-aware model. The encoder tries a fixed set of vertex orders and storage directions and keeps the smallest file that decodes exactly.   On the benchmark graphs outside our development set, the selected files are smaller than the best recorded WebGraph and Zuckerli files and the best published HyperFast results. These savings have a cost. CB decodes much more slowly than the baselines and supports only sequential access. A faster companion coder, CG, is still slower than the baselines on large graphs. An indexed variant, CS-RA, gives smaller seekable files than the measu
    
[^326]: Flash PD-SSM：内存优化的结构化稀疏状态空间模型

    Flash PD-SSM: Memory-Optimized Structured Sparse State-Space Models

    [https://arxiv.org/abs/2605.19150](https://arxiv.org/abs/2605.19150)

    提出了一种新颖的状态空间模型PD-SSM，通过在每个时间步从一组可训练的结构化稀疏矩阵中离散选择一个矩阵，在保持与主流结构化SSM相当的吞吐量的同时，实现了与非结构化转移矩阵相当的表达能力。

    

    状态空间模型（SSM）面临着效率与表达能力之间的根本性权衡，这主要由模型转移矩阵的结构决定。非结构化的转移矩阵能够实现最大化的表达能力（以其对有限状态自动机（FSA）转移的建模能力来衡量），但其计算和内存成本过高。相比之下，大多数结构化的转移矩阵形式在运行时间和内存消耗方面都非常高效，但表达能力有限。基于近期关于结构化稀疏SSM的研究工作，我们提出了PD-SSM，这是一种新颖的状态空间模型，其吞吐量与广泛使用的结构化SSM相当，同时具有显著更优的表达能力保证。PD-SSM维护一组可训练的结构化稀疏矩阵，并在每个时间步从中离散地选择一个矩阵，从而实现与非结构化矩阵同等的FSA表达能力，同时保持高效。

    arXiv:2605.19150v2 Announce Type: replace-cross  Abstract: State-space models (SSMs) face a fundamental trade-off between efficiency and expressivity that is mainly dictated by the structure of the model's transition matrix. Unstructured transition matrices enable maximal expressivity, as measured by their ability to model finite-state automaton (FSA) transitions, but come at a prohibitively high compute and memory cost. In contrast, most structured transition matrix forms are highly efficient both in runtime and memory consumption, but suffer from limited expressivity. Building on recent work on structured sparse SSMs, we propose PD-SSM, a novel SSM that achieves comparable throughput to widely-used structured SSMs with significantly better expressivity guarantees. PD-SSM maintains a trainable set of structured sparse matrices, a single one of which is discretely selected at each time-step, enabling FSA expressiveness at the level of unstructured matrices while maintaining the efficie
    
[^327]: 寻求陌生却令人难忘之物：作为元学习的概念创造性

    Seeking the Unfamiliar but Memorable: Conceptual Creativity as Meta-Learning

    [https://arxiv.org/abs/2605.16477](https://arxiv.org/abs/2605.16477)

    该论文提出将创造性形式化为“创造者-鉴赏者”元学习框架——创造者生成初看陌生但几次接触即可快速学会的刺激，以鉴赏者快速适应的提升作为奖励信号，从而在冻结扩散模型、无需额外语言条件的情况下，生成超越基础模型能力的风格变化与概念组合。

    

    创造一个新概念，而非检索一个熟悉的概念，究竟意味着什么？在同一提示词下反复采样生成模型，会产生风格相似、内容典型的变体。我们提出，创造性是指产生这样一种刺激：对于自适应的观察者而言，第一眼看上去是陌生的，但经过几次接触便能快速学会。我们将其形式化为“创造者-鉴赏者”（Creator-Appraiser）配对机制：创造者生成候选样本，鉴赏者通过若干内循环学习步骤对其进行适应，而鉴赏者的学习提升则成为创造者所优化的奖励。我们以扩散模型作为创造者来实例化该框架，在MNIST上使用自编码器作为鉴赏者，对自然图像则使用带低秩适配器的CLIP作为鉴赏者。在扩散模型被冻结且无需额外语言条件的情况下，仅凭元学习梯度就足以产生超越基础模型生成能力的风格变化和概念组合。

    arXiv:2605.16477v2 Announce Type: replace  Abstract: What does it mean to create a new concept, rather than retrieve a familiar one? Repeatedly sampling a generative model at the same prompt produces variations with similar styles and typical content. We propose that creativity is the production of stimuli that are unfamiliar to an adaptive observer at first sight, but quickly learnable from a few exposures. We formalize this as a Creator-Appraiser pair: a Creator generates a candidate, an Appraiser adapts to it for a few inner-loop learning steps, and the Appraiser's improvement becomes the reward the Creator optimizes through. We instantiate the framework with diffusion as the Creator, an autoencoder Appraiser on MNIST, and a CLIP Appraiser with a low-rank adapter for natural images. With a frozen diffusion model and no additional language conditioning, the meta-learning gradient is enough to produce both stylistic variations and concept compositions beyond what the base model can ge
    
[^328]: 面向离散不确定性两阶段鲁棒优化的学习式场景缩减方法

    Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty

    [https://arxiv.org/abs/2605.14494](https://arxiv.org/abs/2605.14494)

    本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。

    

    具有离散不确定性的两阶段鲁棒优化（2RO）极具挑战性，往往导致精确求解的计算代价难以承受。场景缩减通过选择一个小而具有代表性的场景子集来缓解这一问题，从而使计算变得可行。然而，现有方法在很大程度上与具体问题无关，仅在不考虑可行域或追索结构的情况下对不确定性集合进行操作。本文提出了 PRISE，一种问题驱动的顺序前瞻启发式算法，通过评估每个场景的边际影响来构建缩减后的场景集合。尽管 PRISE 能够产生高质量的场景子集，但其每个选择步骤都需要求解多个子问题，使得该算法在大规模问题上的计算开销十分高昂。为解决这一问题，我们提出了 NeurPRISE，一个基于 GNN-Transformer 骨干网络的神经代理模型，它通过图卷积对单个场景的结构进行编码，并捕获跨场景之间的交互关系。

    arXiv:2605.14494v2 Announce Type: replace  Abstract: Two-Stage Robust Optimization (2RO) with discrete uncertainty is challenging, often rendering exact solutions prohibitive. Scenario reduction alleviates this issue by selecting a small, representative subset of scenarios to enable tractable computation. However, existing methods are largely problem-agnostic, operating solely on the uncertainty set without consulting the feasible region or recourse structure. In this paper, we introduce PRISE, a problem-driven sequential lookahead heuristic that constructs reduced scenario sets by evaluating the marginal impact of each scenario. While PRISE yields high-quality scenario subsets, each selection step requires solving multiple subproblems, making it computationally expensive at scale. To address this, we propose NeurPRISE, a neural surrogate model built on a GNN-Transformer backbone that encodes the per-scenario structure via graph convolution and captures cross-scenario interactions thro
    
[^329]: 人工智能（AI）解释的公平性：统一框架、公理与迈向负责任人工智能的未来方向

    Fairness of Explanations in Artificial Intelligence (AI): A Unifying Framework, Axioms, and Future Direction toward Responsible AI

    [https://arxiv.org/abs/2605.09852](https://arxiv.org/abs/2605.09852)

    本文首次提出了研究AI解释公平性的统一理论框架与公理体系，揭示了模型即使输出满足公平性标准，其推理过程仍可能存在“程序性偏见”，并为迈向负责任的人工智能指明了未来方向。

    

    机器学习算法正被应用于高风险决策领域，包括刑事司法、医疗保健、信贷和就业等。研究界为此发展出了两个彼此较为独立的研究领域：算法公平性，旨在实现公平的结果；以及可解释人工智能（XAI），旨在实现可解释的推理。本综述识别并描绘了这两个领域交叉处的一个新颖盲点：即一个模型可能在输出上满足所有标准公平性准则，但其推理过程却存在深刻的不公平性。我们将此现象称为程序性偏见，要缓解这一问题，就需要将解释的公平性作为一个独立的科学研究对象。据我们所知，我们提供了对这一新兴领域的首个统一的理论与文献综述，并阐明了事后解释器在认证解释公平性方面的缺陷。我们的核心贡献...

    arXiv:2605.09852v2 Announce Type: replace  Abstract: Machine learning algorithms are being used in high-stakes decisions, including those in criminal justice, healthcare, credit, and employment. The research community has responded with two largely independent research fields: \emph{algorithmic fairness}, which targets equitable outcomes, and \emph{explainable AI} (XAI), which targets interpretable reasoning. This survey identifies and maps a novel blind spot at their intersection, which is a model that can satisfy every standard fairness criterion in its outputs while being profoundly unfair in its \emph{reasoning process}. We refer to this as the procedural bias, and mitigating it requires treating the fairness of explanations as a distinct object of scientific study.   To our knowledge, we provide the first unified theoretical and literature review of this emerging field and elucidate the drawbacks of post-hoc explainers in certifying explanation fairness. Our central contribution i
    
[^330]: 智能体集体不应自行检测其内部的冒名顶替者：一个国际象棋案例研究

    Agent Collectives Should Not Detect Their Own Imposters: A Chess Case Study

    [https://arxiv.org/abs/2605.09027](https://arxiv.org/abs/2605.09027)

    研究表明让智能体集体自行检测内部冒名顶替者会适得其反，而采用能够从少量样本快速适应新型攻击的轻量级元训练外部检测器才是更优策略。

    

    AI智能体集体在复杂任务上可以超越任何单一智能体，但仍然容易受到故意误导同伴的冒名顶替者（imposter）的攻击。由于哪怕只有一个冒名顶替者也可能破坏整个集体，可靠的检测机制至关重要。我们研究了两种策略：将检测功能嵌入参与智能体自身内部，或者雇佣一个专门的、只观察但不参与集体讨论的外部检测器。在一个特意设计的小型但具有挑战性的国际象棋测试平台上，我们发现仅仅警告智能体可能存在冒名顶替者反而会产生反效果：即使在没有对手的情况下，它也会降低决策质量，引发无益的社交动态，并向冒名顶替者泄露信息。外部检测器虽然可以避免这些陷阱，但必须随着冒名顶替者演化出新策略而不断适应。我们证明，一个轻量级的元训练检测器仅凭少量样本就能快速重新校准，以应对未见过的攻击，其效率远高于其他方法。

    arXiv:2605.09027v4 Announce Type: replace-cross  Abstract: Collectives of AI agents can outperform any single agent on complex tasks, yet remain vulnerable to imposters that deliberately mislead their peers. Because even one such imposter can sabotage the collective, reliable detection is essential. We examine two strategies: embedding detection inside the participating agents, or employing a dedicated external detector that observes the deliberation without joining it. On a deliberately small but challenging chess-based testbed, we find that simply warning agents of possible imposters is counterproductive. It degrades decisions even in the absence of adversaries, triggers unhelpful social dynamics, and leaks information to the imposter. An external detector avoids these pitfalls, yet must continually adapt as imposters evolve new strategies. We show that a lightweight meta-trained detector can rapidly recalibrate to unseen attacks from only a handful of examples, far more efficiently 
    
[^331]: 面向分布式强化学习的分位数耦合流匹配

    Quantile-Coupled Flow Matching for Distributional Reinforcement Learning

    [https://arxiv.org/abs/2605.08515](https://arxiv.org/abs/2605.08515)

    该论文提出FlowIQN方法，通过在小批次内对源样本和目标样本进行排序形成分位数耦合，修复了条件流匹配评论家与分布式贝尔曼算子之间的Wasserstein度量不匹配问题。

    

    分布式强化学习（DRL）对完整的回报分布进行建模，因此非常适合具有不确定性感知和风险敏感性的决策任务。条件流匹配（CFM）评论家在DRL中的采用日益增多，它提供了密集的速度监督以及扩展评论家计算量的自然维度。然而，这类方法存在一个隐藏的度量不匹配问题：分布式贝尔曼算子在Wasserstein距离下是收缩映射，但CFM评论家任意地配对源样本与贝尔曼目标样本，因此其损失并不能控制到贝尔曼目标的Wasserstein距离。我们证明，这种不匹配可以通过一次排序操作来修复。我们的方法FlowIQN在每个小批次内对源样本和目标样本进行排序，使每个源分位数被传输到其对应匹配的目标分位数，从而近似单调最优传输耦合。我们证明，这种分位数耦合使CFM损失成为（原文此处截断）的上界……

    arXiv:2605.08515v2 Announce Type: replace  Abstract: Distributional Reinforcement Learning (DRL) models the full return distribution, making it well-suited to uncertainty-aware and risk-sensitive decision-making. Conditional Flow Matching (CFM) critics are increasingly adopted in DRL, offering dense velocity supervision and a natural axis for scaling critic computation. Yet they carry a hidden metric mismatch: the distributional Bellman operator is a contraction in the Wasserstein distance, but CFM critics pair source and Bellman target samples arbitrarily, so their loss does not control the Wasserstein distance to the Bellman target. We show that this mismatch can be fixed with a single sorting operation. Our method, FlowIQN, sorts source and target samples within each mini-batch so that each source quantile is transported to its matching target quantile, approximating the monotone optimal transport coupling. We prove that this quantile coupling makes the CFM loss upper-bound, up to a
    
[^332]: 基于CADTests的文本转CAD评估

    Text-to-CAD Evaluation with CADTests

    [https://arxiv.org/abs/2605.07807](https://arxiv.org/abs/2605.07807)

    该论文提出了首个基于可执行测试的文本转CAD评估基准CADTestBench，通过CADTests验证生成模型是否满足几何与拓扑要求，并证明这些测试还能指导CAD生成，使简单基线超越现有方法。

    

    文本转CAD（Text-to-CAD）近来已成为一项重要任务，具有大幅加速设计工作流程的潜力。尽管其意义重大，但令人惊讶的是，关于文本转CAD评估的研究工作却很少，评估CAD模型生成性能仍然是一个相当大的挑战。在这项工作中，我们引入了一种基于自动化测试的文本转CAD新评估视角。我们提出了CADTestBench，这是首个基于测试的文本转CAD基准，其基于CADTests构建——这是一类可执行的软件测试，用于验证生成的CAD模型是否满足输入提示词中的几何和拓扑要求。利用CADTestBench，我们对近期的文本转CAD方法进行了全面的基准测试，并进一步证明CADTests还可以用于指导CAD模型生成，由此得到的简单基线方法能够超越现有方法的性能。CADTestBench的代码和数据均已公开可用。

    arXiv:2605.07807v2 Announce Type: replace-cross  Abstract: Text-to-CAD has recently emerged as an important task with the potential to substantially accelerate design workflows. Despite its significance, there has been surprisingly little work on Text-to-CAD evaluation, and assessing CAD model generation performance remains a considerable challenge. In this work, we introduce a new evaluation perspective for Text-to-CAD based on automated testing. We propose CADTestBench, the first test-based benchmark for Text-to-CAD, based on CADTests, executable software tests that verify whether a generated CAD model satisfies the geometric and topological requirements of the input prompt. Using CADTestBench, we conduct comprehensive benchmarking of recent Text-to-CAD methods and further demonstrate that CADTests can also guide CAD model generation, yielding simple baselines that surpass performance of current methods. CADTestBench code and data are available.
    
[^333]: 通过中心元小批量梯度采样实现重尾噪声下鲁棒的随机一阶方法

    Robust stochastic first order methods in heavy-tailed noise via medoid mini-batch gradient sampling

    [https://arxiv.org/abs/2605.07634](https://arxiv.org/abs/2605.07634)

    本文提出一种新颖的梯度估计器，通过将小批量数据分割成多个数据块并选取梯度中心元作为估计值，在无需梯度裁剪等非线性操作的情况下，实现了重尾噪声（方差可能无限）下鲁棒的随机一阶优化。

    

    我们考虑一个一阶随机优化框架，其中在每次迭代中抽取K个独立同分布（i.i.d.）的数据点样本，基于这些样本可以查询随机梯度。我们允许梯度噪声呈重尾分布，方差可能为无限。针对所考虑的重尾设置，最近已提出了许多算法变体，它们基于梯度裁剪或对噪声梯度应用其他非线性算子（例如归一化）。在本文中，我们采取一种替代方法，提出了一种新颖的梯度估计器。所提估计器的核心思想是将大小为K的数据批次分割成M个不同的数据块，为每个数据块形成随机梯度，然后选取作为所有数据块梯度中心元的那个数据块的随机梯度。对于一大类对称重尾梯度噪声分布，我们证明了所提出的估计器具……

    arXiv:2605.07634v2 Announce Type: replace-cross  Abstract: We consider a first order stochastic optimization framework where, at each iteration, $K$ independent identically distributed (i.i.d.) data point samples are drawn, based on which stochastic gradients can be queried. We allow gradient noise to be heavy-tailed, with possibly infinite variances. For the considered heavy-tailed setting, many algorithmic variants have recently been proposed based on gradient clipping or other nonlinear operators (e.g., normalization) applied over noisy gradients. In this paper, we take an alternative approach and propose a novel gradient estimator. The core idea of proposed estimator is to split the $K$-sized data batch into $M$ distinct data chunks, form for each chunk the stochastic gradient, and select the stochastic gradient of the chunk that is medoid of gradients of all data-chunks. For a broad class of symmetric heavy-tailed gradient noise distributions, we show that the proposed estimator h
    
[^334]: 在大语言模型中构建可逆的SFT行为

    Crafting Reversible SFT Behaviors in Large Language Models

    [https://arxiv.org/abs/2605.06632](https://arxiv.org/abs/2605.06632)

    该论文提出损失约束双重下降方法，可将SFT诱发的行为构造性地压缩到稀疏子网络“载体”中，该载体对行为既充分又必要，从而在不修改权重的情况下实现对大语言模型微调行为的推理时可控与可逆操控。

    

    监督微调会在大语言模型中诱发新的行为，但对这些行为在模型内部的分布方式没有施加任何结构约束。现有的行为解释方法（如电路归因方法）只能事后识别与SFT诱发行为相关的稀疏子网络。然而，微调后的模型仍可能通过其他路径携带该行为，因此这类子网络无法保证在推理时有选择性地控制SFT诱发的行为。我们寻求一种构造性的替代方案：能否将SFT诱发的行为有意地压缩到一个稀疏子网络（称为“载体”，carrier）中，使得该载体对该行为既是充分的、又在构造上是必要的，同时在不修改权重的情况下于推理时保持可控？我们提出：（a）损失约束双重下降方法，通过联合优化路由……（摘要在此处截断）

    arXiv:2605.06632v2 Announce Type: replace  Abstract: Supervised fine-tuning (SFT) induces new behaviors in large language models, yet imposes no structural constraint on how these behaviors are distributed within the model. Existing behavior interpretation methods, such as circuit attribution approaches, identify sparse subnetworks correlated with SFT-induced behaviors post-hoc. However, the fine-tuned model may still carry the behavior through other pathways, so such subnetworks offer no guaranteed handle for selectively controlling SFT-induced behaviors at inference time. We pursue a constructive alternative by asking: can an SFT-induced behavior be deliberately compressed into a sparse subnetwork, termed a *carrier*, that is sufficient for the behavior and, by construction, necessary for it, while remaining controllable at inference time without weight modification? We propose (a) **Loss-Constrained Dual Descent (LCDD)**, which constructs such carriers by jointly optimizing routing 
    
[^335]: PairAlign：一种基于自对齐的自回归词元化框架及其在音频词元化中的应用

    PairAlign: A Framework for Autoregressive Tokenization via Self-Alignment with Applications to Audio Tokenization

    [https://arxiv.org/abs/2605.06582](https://arxiv.org/abs/2605.06582)

    PairAlign提出了一种通过自回归自对齐实现紧凑音频词元化的框架，通过让两个内容保持视图预测统一的规范锚点目标，实现了序列一致性、紧凑性和长度等特性的可控音频离散表示。

    

    现代学习系统使用连续向量来表示感知信号，但比较、检索、记忆、对齐和推理往往是符号化的。在语言中，词元提供了这种接口；而对于语音和音频，这种接口必须通过学习获得。现有的音频词元化器依赖于局部量化、聚类或重建，使得序列一致性、紧凑性、长度、终止和编辑几何特性只能被间接控制。我们提出了PairAlign，一个通过自回归自对齐实现紧凑音频词元化的框架。编码器将语音映射为连续条件，自回归解码器从BOS到EOS生成词元。给定两个保持内容的视图，PairAlign推导出一个规范的锚点目标，并训练两个视图来预测它，同时以批内无关目标作为竞争序列。该框架首先从VQ目标学习自回归桥梁，然后过渡到EMA教师自对齐（摘要在此处截断）。

    arXiv:2605.06582v4 Announce Type: replace-cross  Abstract: Modern learning systems represent perceptual signals with continuous vectors, but comparison, retrieval, memory, alignment, and reasoning are often symbolic. In language, tokens provide this interface; for speech and audio, it must be learned. Existing audio tokenizers rely on local quantization, clustering, or reconstruction, leaving sequence consistency, compactness, length, termination, and edit geometry only indirectly controlled. We introduce PairAlign, a framework for compact audio tokenization through autoregressive self-alignment. An encoder maps speech to a continuous condition, and an autoregressive decoder emits tokens from BOS to EOS. Given two content-preserving views, PairAlign derives a canonical anchor target and trains both views to predict it, with unrelated in-batch targets as competing sequences. It first learns an autoregressive bridge from VQ targets and then transitions to EMA-teacher self-alignment with 
    
[^336]: 协变量漂移下部署风险的雅可比-速度界

    Jacobian-Velocity Bounds for Deployment Risk Under Covariate Drift

    [https://arxiv.org/abs/2605.04932](https://arxiv.org/abs/2605.04932)

    该论文通过时域庞加莱不等式与雅可比-速度定理建立了协变量漂移下冻结预测器长期部署的风险上界，并据此提出漂移对齐切向正则化（DTR）及配套监控指标来有效控制部署风险。

    

    我们研究了冻结预测器在动态协变量偏移下的长期部署问题。首先，一个时域庞加莱（Poincaré）不等式将时间上的风险波动性归约为导数能量，随后由雅可比-速度定理提供相应的逐路径控制。通过余项扩展，我们将经由模型介导的可控项与未通过得分雅可比传递的条件风险变化分离开来。在低秩漂移条件下，该界将可控项分解为平行和正交于漂移子空间的切向能量。这一结果启发了漂移对齐切向正则化（DTR）、其带有较弱正交护栏的各向异性扩展，以及与之相匹配的监控代理指标。受控实验验证了该时间不等式、方向性平滑的高效性以及余项的必要性。主要的实地研究采用一个128维气体传感器阵列在十个按时间顺序排列的批次上进行。一个秩为4的基估计……（摘要截断）

    arXiv:2605.04932v3 Announce Type: replace-cross  Abstract: We study long-horizon deployment of a frozen predictor under dynamic covariate shift. A time-domain Poincar\'e inequality first reduces temporal risk volatility to derivative energy, and a Jacobian-velocity theorem supplies the corresponding pathwise control. A remainder extension separates the model-mediated, controllable term from conditional-risk variation not transmitted through the score Jacobian. Under low-rank drift, the bound decomposes the controllable term into tangent energy parallel and orthogonal to the drift subspace. This motivates drift-aligned tangent regularization (DTR), its anisotropic extension with a weaker orthogonal guardrail, and a matched monitoring proxy. Controlled experiments verify the temporal inequality, the efficiency of directional smoothing, and the necessity of the remainder term. The main field study uses a 128-dimensional gas-sensor array over ten chronological batches. A rank-4 basis estim
    
[^337]: UniVer：多步与多草稿投机解码的统一视角

    UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding

    [https://arxiv.org/abs/2605.04543](https://arxiv.org/abs/2605.04543)

    本文提出UniVer，将树状投机解码的验证统一建模为条件最优传输问题，通过前缀接受概率作为动态缩放因子主动引导草稿选择，从而联合优化多步依赖与多草稿分支之间的耦合关系。

    

    投机解码通过“先起草、后验证”的方式加速大语言模型，其中验证过程可以建模为最优传输（OT）问题。现有方法通常将多草稿与多步骤两个维度割裂开来处理：要么对单步草稿应用扁平最优传输，要么对树状结构候选进行逐token的拒绝采样。这种割裂导致联合场景（即多步依赖与多草稿分支同时存在）得不到充分优化，因为局部验证规则未能利用候选树水平维度与垂直维度之间的耦合关系。本文提出一种统一视角，将基于树的验证问题转化为条件最优传输问题。我们的关键洞察是：垂直方向的依赖关系可以通过前缀接受概率来抽象，这些概率作为动态缩放因子，能够主动引导水平方向的草稿选择。基于这一原理，我们提出了UniVer，一种验证算法……（摘要原文在此处截断）

    arXiv:2605.04543v2 Announce Type: replace  Abstract: Speculative decoding accelerates Large Language Models via draft-then-verify, where verification can be framed as an Optimal Transport (OT) problem. Existing approaches typically handle multi-draft and multi-step aspects in isolation, applying either flat OT to single-step drafts or per-token rejection sampling to tree-structured candidates. This separation leaves the joint regime (where multi-step dependencies meet multi-draft branching) poorly optimized, as local verification rules fail to exploit the coupling between horizontal and vertical dimensions of candidate trees. In this paper, we propose a unified perspective that casts tree-based verification as a conditional OT problem. Our key insight is that vertical dependencies can be abstracted through prefix acceptance probabilities, which act as dynamic scaling factors to actively guide horizontal draft selection. Based on this principle, we introduce UniVer, a verification algor
    
[^338]: 基于多阶段选择性分类器的序贯策略性分类

    Sequential Strategic Classification with Multi-Stage Selective Classifiers

    [https://arxiv.org/abs/2605.04202](https://arxiv.org/abs/2605.04202)

    本文提出了一种序贯、随机、多阶段的策略性分类新模型，刻画了智能体通过改进行为与投机行为在多个阶段面对不同分类器时逐步调整策略的过程，突破了以往研究仅关注单次决策或与单一分类器重复交互的局限。

    

    策略性分类研究的是自利的个人或智能体通过操纵自身响应来获取分类器有利决策结果的问题，当不诚实行为的代价低于付出真实努力时，他们通常会采取不诚实的行动。先前的研究已经表明，仅仅专注于分类器的设计无法从根本上摆脱这一困境。我们还注意到，先前的工作也大多集中在一锤子定音的单次决策设置，或与同一分类器的重复交互上。然而，现实世界的决策过程往往是多阶段的，随着智能体不断推进，会涉及一系列可能不同的分类器。本文引入了一个序贯的、随机的、多阶段的策略性分类模型，刻画了智能体如何通过改进行为（同时提升可观测特征和真实属性）与投机行为（仅提升可观测特征），在多个阶段逐步调整自身行为。

    arXiv:2605.04202v2 Announce Type: replace  Abstract: Strategic classification studies the problem where self-interested individuals or agents manipulate their response to obtain favorable decision outcomes made by classifiers, typically turning to dishonest actions when they are less costly than genuine efforts. Prior works have demonstrated a fundamental inability to get out of this conundrum by only focusing on the design of a classifier. We note that prior work also heavily focuses on either one-shot settings or repeated interaction with the same classifier. Real-world decision making is often multi-stage, involving a sequence of potentially different classifiers as an agent progresses. This paper introduces a sequential, stochastic, multi-stage model of strategic classification, by capturing how agents adapt their behavior, through improvement actions (enhancing both observable features and true attributes) and gaming actions (enhancing only observable features), over multiple leve
    
[^339]: 基于模型的强化学习：在策略优化与离线估计中实现双重预言机高效性

    Model-Based Reinforcement Learning with Double Oracle Efficiency in Policy Optimization and Offline Estimation

    [https://arxiv.org/abs/2605.00393](https://arxiv.org/abs/2605.00393)

    该论文通过对数障碍与对数行列式正则化技术，为表格型MDP提出了新算法，在实现最优 $\tilde{O}(\sqrt{T})$ 遗憾界的同时，将离线统计估计与规划预言机的调用次数降至对数级别，从而突破了大规模环境下强化学习的计算瓶颈。

    

    在大规模环境中的强化学习（RL）常常面临严重的计算瓶颈，因为传统的遗憾最小化算法需要反复、高成本地调用规划与统计估计预言机。尽管近期的研究已探索了离线预言机高效算法，但其计算复杂度通常随状态空间与动作空间的基数而增长，使其在大规模或连续环境中难以处理。在本文中，我们通过对数障碍与对数行列式正则化的视角研究离线预言机高效的情节式强化学习，从而解决这一根本性局限。具体而言，对于表格型马尔可夫决策过程（MDP），我们提出了一种新颖的算法，在已知总轮数 $T$ 时仅需 $O(H\log\log T)$ 次离线统计估计与规划预言机的调用，而 $T$ 未知时仅需 $O(H\log T)$ 次调用，同时达到最优的 $\tilde{O}(\sqrt{T})$ 遗憾界。

    arXiv:2605.00393v2 Announce Type: replace  Abstract: Reinforcement learning (RL) in large environments often suffers from severe computational bottlenecks, as conventional regret minimization algorithms require repeated, costly calls to planning and statistical estimation oracles. While recent advances have explored offline oracle-efficient algorithms, their computational complexity typically scales with the cardinality of the state and action spaces, rendering them intractable for large-scale or continuous environments. In this paper, we address this fundamental limitation by studying offline oracle-efficient episodic RL through the lens of log-barrier and log-determinant regularization. Specifically, for tabular Markov Decision Processes (MDPs), we propose a novel algorithm that achieves the optimal $\tilde{O}(\sqrt{T})$ regret bound while requiring only $O(H\log\log T)$ calls to both the offline statistical estimation and planning oracles when $T$ is known and $O(H\log T)$ calls whe
    
[^340]: 自适应共形预测：提升大语言模型生成内容的事实性

    Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models

    [https://arxiv.org/abs/2604.13991](https://arxiv.org/abs/2604.13991)

    提出了一种提示词自适应的共形预测方法，通过提示词相关的校准提升大语言模型生成内容的事实性，在保持边际覆盖保证的同时改善条件覆盖率，并支持选择性预测以过滤不可靠输出。

    

    大语言模型（LLM）容易生成事实性错误的输出。近期的工作将共形预测应用于为大语言模型生成内容的事实性提供不确定性估计和统计保证。然而，现有方法通常无法自适应提示词，限制了其捕捉输入相关变化的能力。因此，对于给定的任务或提示，这些方法可能过滤掉过少的条目（导致过度覆盖）或过滤掉过多条目（导致覆盖不足）。我们提出了一种自适应共形预测方法，将共形分数变换方法扩展到大语言模型中，并应用于长文本生成和多选题问答任务。该方法实现了依赖于提示词的校准，在保持边际覆盖保证的同时提高了条件覆盖率。此外，该方法还自然地支持选择性预测，允许在下游应用中过滤掉不可靠的陈述或答案选项。

    arXiv:2604.13991v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are prone to generating factually incorrect outputs. Recent work has applied conformal prediction to provide uncertainty estimates and statistical guarantees for the factuality of LLM generations. However, existing approaches are typically not prompt-adaptive, limiting their ability to capture input-dependent variability. As a result, they may filter out too few items (leading to over-coverage) or too many (under-coverage) for a given task or prompt. We propose an adaptive conformal prediction approach that extends conformal score transformation methods to LLMs, with applications to long-form generation and multiple-choice question answering. This enables prompt-dependent calibration, retaining marginal coverage guarantees while improving conditional coverage. In addition, the approach naturally supports selective prediction, allowing unreliable claims or answer choices to be filtered out in downstr
    
[^341]: AVO：用于自主演化搜索的智能体变异算子

    AVO: Agentic Variation Operators for Autonomous Evolutionary Search

    [https://arxiv.org/abs/2603.24517](https://arxiv.org/abs/2603.24517)

    AVO用自主编码智能体取代传统演化搜索中的固定变异算子，在B200 GPU上经过7天连续自主演化，发现了性能超越cuDNN达3.5%、超越FlashAttention-4达10.5%的注意力内核优化方案。

    

    智能体变异算子（AVO）是一类新型演化变异算子，它用自主编码智能体取代了经典演化搜索中固定的变异、交叉以及人工设计的启发式方法。AVO并未将语言模型局限于规定流程中的候选生成，而是将变异实现为一个自主导向的智能体循环，该循环可以查询当前谱系、领域特定知识库和执行反馈，从而提出、修复、批评和验证实现修改。我们在NVIDIA Blackwell（B200）GPU上，针对注意力机制——AI领域中优化竞争最激烈的内核目标之一——对AVO进行了评估。经过7天对多头注意力的连续自主演化，AVO发现的内核在所有评估配置下性能比cuDNN高出最多3.5%，比FlashAttention-4高出最多10.5%。所发现的优化可以轻松迁移到分组查询注意力，仅需……（原文摘要在此处截断）

    arXiv:2603.24517v2 Announce Type: replace  Abstract: Agentic Variation Operators (AVO) are a new family of evolutionary variation operators that replace the fixed mutation, crossover, and hand-designed heuristics of classical evolutionary search with autonomous coding agents. Rather than confining a language model to candidate generation within a prescribed pipeline, AVO instantiates variation as a self-directed agent loop that can consult the current lineage, a domain-specific knowledge base, and execution feedback to propose, repair, critique, and verify implementation edits. We evaluate AVO on attention, among the most aggressively optimized kernel targets in AI, on NVIDIA Blackwell (B200) GPUs. Over 7 days of continuous autonomous evolution on multi-head attention, AVO discovers kernels that outperform cuDNN by up to 3.5% and FlashAttention-4 by up to 10.5% across the evaluated configurations. The discovered optimizations transfer readily to grouped-query attention, requiring only 
    
[^342]: ProFed：面向低剂量CT去噪的投影引导个性化联邦学习

    ProFed: Projection-Guided Personalized Federated Learning for Low-Dose CT Denoising

    [https://arxiv.org/abs/2603.13422](https://arxiv.org/abs/2603.13422)

    ProFed提出了一种结合图像域特征调制与投影域监督的个性化联邦学习框架，通过解剖/协议描述符、可微分投影一致性损失和不确定性加权聚合，有效应对患者间解剖差异与扫描协议变化，实现低剂量CT去噪的协同训练。

    

    低剂量计算机断层扫描（CT）减少了辐射暴露，但会引入噪声和伪影，其表现依赖于采集条件。联邦学习能够在不集中汇总图像的情况下实现协同复原，但解剖结构差异和扫描协议的变化使得学习一个共享去噪器变得复杂。我们提出了ProFed，这是一种将图像域特征调制与投影域监督相结合的个性化联邦框架。解剖与协议描述符对共享的编码器-解码器进行条件化调节，同时可微分的正投影、反投影和循环一致性损失促使模型与客户端本地测量数据保持一致。不确定性加权聚合规则用于融合共享参数，并在本地保留解剖结构特定的输出头。遵循SCAN-PhysFed基准，我们在八名梅奥诊所（Mayo Clinic）患者上进行训练，并在八种已知采集协议下对两名留出的患者进行评估。ProFed……

    arXiv:2603.13422v2 Announce Type: replace-cross  Abstract: Low-dose computed tomography (CT) reduces radiation exposure but introduces noise and artifacts whose appearance depends on acquisition conditions. Federated learning enables collaborative restoration without centrally pooling images, yet variation in anatomy and scanning protocols complicates learning a shared denoiser. We present ProFed, a personalized federated framework that combines image-domain feature modulation with projection-domain supervision. Anatomy and protocol descriptors condition a shared encoder--decoder, while differentiable forward-projection, backprojection, and cycle-consistency losses encourage agreement with client-local measurements. An uncertainty-weighted aggregation rule combines shared parameters and retains anatomy-specific heads locally. Following the SCAN-PhysFed benchmark, we train on eight Mayo Clinic patients and evaluate two held-out patients under eight known acquisition protocols. ProFed ac
    
[^343]: SoundWeaver：面向文本到音频扩散模型服务的组合式热启动方法

    SoundWeaver: Compositional Warm-Starting for Text-to-Audio Diffusion Serving

    [https://arxiv.org/abs/2603.07865](https://arxiv.org/abs/2603.07865)

    SoundWeaver是首个无需训练的组合式跨示例声学复用框架，通过从缓存音频片段构建提示条件化声学先验并热启动扩散过程，显著加速文本到音频扩散模型的推理。

    

    文本到音频（T2A）扩散模型能够生成高质量音频，但需要数十次神经函数评估（NFE），导致显著的推理延迟。现有加速方法主要对每条生成轨迹进行独立优化，即便先前示例中存在可复用的声学结构。我们提出SoundWeaver，这是首个无需训练的组合式跨示例声学复用框架。SoundWeaver不再从纯噪声开始生成，而是从缓存的音频中提取片段构建提示条件化的声学先验，并从中间噪声水平热启动扩散过程。我们进一步引入声学组合图（ACG），利用神经编解码器表示和片段级识别来确定跨缓存转移的候选片段，并通过结构化路径搜索联合优化提示相关性与声学连续性。我们还建立了热启动误差界限，分析了……

    arXiv:2603.07865v2 Announce Type: cross  Abstract: Text-to-audio (T2A) diffusion models generate high-quality audio but require tens of neural function evaluations (NFEs), resulting in substantial inference latency. Existing acceleration methods primarily optimize each generation trajectory independently, even when reusable acoustic structure exists in prior examples. We introduce SoundWeaver, the first training-free framework for compositional cross example acoustic reuse. Rather than generating from pure noise, SoundWeaver constructs a prompt-conditioned acoustic prior from segments distributed across cached audios and warm-starts diffusion from an intermediate noise level. We introduce an Acoustic Composition Graph (ACG), which uses neural codec representations and segment-level identification to identify candidate cross-cache transitions and structured path search to jointly optimize prompt relevance and acoustic continuity. We further establish a warm-start error bound showing how
    
[^344]: 面向多时间尺度动力学的频率结构化哈密顿神经网络

    Frequency-Structured Hamiltonian Neural Network for Multi-Timescale Dynamics

    [https://arxiv.org/abs/2603.06354](https://arxiv.org/abs/2603.06354)

    该论文提出频率结构化哈密顿神经网络（FS-HNN），通过在时间频率尺度上组织哈密顿学习来克服频谱偏差，从而准确刻画快慢耦合的多时间尺度动力学，并在连续时间内保持能量守恒。

    

    对于结构保持型神经模型而言，准确学习快慢耦合的动力学仍然具有挑战性。哈密顿神经网络能够保持所学习到的哈密顿量，但其神经参数化容易受到频谱偏差的影响，这种偏差偏向缓慢变化的分量，可能导致快速振荡无法得到良好的解析。我们提出了频率结构化哈密顿神经网络（FS-HNN），它在时间频率尺度上组织哈密顿学习。FS-HNN 从观测轨迹的频率滤波视图中学习哈密顿分量，并将它们非线性地组合为单个标量哈密顿量。这使得模型能够表示跨时间尺度的相互作用，同时在连续时间内保持所学习到的能量守恒。对于偏微分方程（PDE），FS-HNN 将哈密顿分量表示为神经泛函并学习一个空间算子，利用正交投影来保持所学习到的能量守恒。在 ODE 和 PDE 基准上的实验（原文此处截断）……

    arXiv:2603.06354v3 Announce Type: replace  Abstract: Learning coupled fast and slow dynamics accurately remains challenging for structure-preserving neural models. Hamiltonian Neural Networks preserve a learned Hamiltonian, but their neural parameterizations are susceptible to spectral bias, which favors slowly varying components and can leave fast oscillations poorly resolved. We introduce the Frequency-Structured Hamiltonian Neural Network (FS-HNN), which organizes Hamiltonian learning across temporal frequency scales. FS-HNN learns Hamiltonian components from frequency-filtered views of observed trajectories and combines them nonlinearly into a single scalar Hamiltonian. This enables the model to represent interactions across timescales while conserving the learned energy in continuous time. For PDEs, FS-HNN represents Hamiltonian components as neural functionals and learns a spatial operator, using orthogonal projection to conserve the learned energy. Experiments on ODE and PDE ben
    
[^345]: 大语言模型的失效关闭对齐

    Fail-Closed Alignment for Large Language Models

    [https://arxiv.org/abs/2602.16977](https://arxiv.org/abs/2602.16977)

    提出失效关闭对齐设计原则及渐进式对齐框架，通过迭代消融已学到的拒绝方向、迫使模型在独立子空间中重构安全性，将越狱攻击平均成功率降至1.6%。

    

    我们发现了当前大语言模型（LLM）对齐中的一个结构性弱点：这些模型的拒绝机制是“失效开放”的。现有方法倾向于将拒绝行为编码在多个潜在特征中，然而只需抑制单个特征（通过基于提示词的越狱攻击）就足以使对齐崩溃，导致不安全的生成内容。受此启发，我们提出“失效关闭”对齐作为鲁棒LLM安全的设计原则：即使发生部分故障，拒绝机制也应通过冗余且独立的因果路径保持有效。我们给出了该原则的一个具体实例：一个渐进式对齐框架，它迭代地识别并消融先前学到的拒绝方向，迫使模型在新的、独立的子空间中重构安全性。在六种越狱攻击下，我们的方法实现了最低的平均攻击成功率（1.6%），同时基本保持了生成质量。

    arXiv:2602.16977v2 Announce Type: replace  Abstract: We identify a structural weakness in current large language model (LLM) alignment: refusal mechanisms in these models are fail-open. While existing approaches tend to encode refusal behaviors across multiple latent features, suppressing a single feature (via prompt-based jailbreaks) is sufficient to collapse alignment, leading to unsafe generation. Motivated by this, we propose fail-closed alignment as a design principle for robust LLM safety: refusal mechanisms should remain effective even under partial failures via redundant, independent causal pathways. We present a concrete instantiation of this principle: a progressive alignment framework that iteratively identifies and ablates previously learned refusal directions, forcing the model to reconstruct safety along new, independent subspaces. Across six jailbreak attacks, our method achieves the lowest observed average attack success rate (1.6%) while largely preserving generation q
    
[^346]: 分布偏移下伪校准共形预测的覆盖率保证

    Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift

    [https://arxiv.org/abs/2602.14913](https://arxiv.org/abs/2602.14913)

    该论文针对分布偏移下共形预测覆盖率失效的问题，利用域适应工具推导出以源分类器损失和Wasserstein偏移表示的目标覆盖率下界，并提出源调优伪校准方法来缓解覆盖率退化。

    

    共形预测（CP）在可交换性假设下提供无分布的边际覆盖率保证，但在分布偏移下覆盖率可能失效。我们研究了标签条件特征偏移有界情形下，针对无标签目标数据的伪校准方法。利用域适应工具，我们推导出了以源分类器损失和Wasserstein偏移表示的目标覆盖率下界。我们还分析了伪校准阈值的固定松弛膨胀，并基于该结果提出了一种启发式阈值调整方法。最后，我们提出了源调优伪校准方法，该方法基于分类器的不确定性，在硬伪标签和随机化标签之间进行插值。在MNIST、CIFAR-10和CIFAR-100数据集上的实验表明，所提出的方法缓解了分布偏移下的覆盖率退化问题，但代价是更大的预期集合大小。

    arXiv:2602.14913v4 Announce Type: replace  Abstract: Conformal prediction (CP) provides distribution-free marginal coverage under exchangeability, but coverage can fail under distribution shift. We study pseudo-calibration for unlabeled target data under bounded label-conditional feature shift. Using domain-adaptation tools, we derive target coverage lower bounds in terms of source classifier loss and Wasserstein shift. We also analyze fixed slack inflation of the pseudo-calibrated threshold and use this result to motivate a heuristic threshold adjustment. Finally, we propose source-tuned pseudo-calibration, which interpolates between hard pseudo-labels and randomized labels based on classifier uncertainty. Experiments on MNIST, CIFAR-10, and CIFAR-100 show that the proposed method mitigates coverage degradation under shift, with larger expected set size.
    
[^347]: 用于高效反向传播的无偏近似向量-雅可比积

    Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation

    [https://arxiv.org/abs/2602.14701](https://arxiv.org/abs/2602.14701)

    本文提出在反向传播中用随机化的无偏近似替代精确的向量-雅可比积，从理论上分析了精度与成本之间的权衡并证明了稀疏约束下最小方差的最优性，实验验证了该方法能有效降低深度学习的训练成本。

    

    在本工作中，我们提出了降低深度神经网络训练的计算与内存成本的方法。我们的方法是在反向传播过程中，用随机化的无偏近似来替代精确的向量-雅可比积。我们对达到目标精度所需的训练轮数与每轮成本降低之间的权衡进行了理论分析。随后，我们确定了向量-雅可比积的若干特定无偏估计，并证明了它们在稀疏性约束下具有最小方差的最优性质。最后，我们在多层感知机、BagNets 和视觉 Transformer 等架构上进行了深入实验。这些实验验证了我们的理论结果，并证实了所提出的无偏随机化反向传播方法在降低深度学习成本方面的潜力。

    arXiv:2602.14701v2 Announce Type: replace  Abstract: In this work we introduce methods to reduce the computational and memory costs of training deep neural networks. Our approach consists in replacing exact vector-jacobian products by randomized, unbiased approximations thereof during backpropagation. We provide a theoretical analysis of the trade-off between the number of epochs needed to achieve a target precision and the cost reduction for each epoch. We then identify specific unbiased estimates of vector-jacobian products for which we establish desirable optimality properties of minimal variance under sparsity constraints. Finally we provide in-depth experiments on multi-layer perceptrons, BagNets and Visual Transfomers architectures. These validate our theoretical results, and confirm the potential of our proposed unbiased randomized backpropagation approach for reducing the cost of deep learning.
    
[^348]: VI-CuRL：通过置信度引导的方差缩减稳定无验证器强化学习推理

    VI-CuRL: Stabilizing Verifier-Independent Reinforcement Learning Reasoning via Confidence-Guided Variance Reduction

    [https://arxiv.org/abs/2602.12579](https://arxiv.org/abs/2602.12579)

    提出VI-CuRL框架，利用模型内在置信度构建课程并优先处理高置信度样本，有效降低梯度方差，从而解决无验证器强化学习推理中的训练崩溃问题。

    

    基于可验证奖励的强化学习（RLVR）已成为增强大型语言模型（LLM）推理能力的主流范式，但其对外部验证器的依赖限制了可扩展性。近期研究发现，RLVR主要是通过激发模型的潜在能力来发挥作用，这推动了无验证器算法的发展。然而，在此类设置中，诸如组相对策略优化（GRPO）等标准方法面临一个关键挑战：破坏性的梯度方差往往导致训练崩溃。为解决这一问题，我们提出了无验证器课程强化学习（VI-CuRL），该框架利用模型的内在置信度构建独立于外部验证器的课程。通过优先处理高置信度样本，VI-CuRL有效管理偏差-方差权衡，专门针对动作方差和问题方差的缩减进行优化。

    arXiv:2602.12579v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a dominant paradigm for enhancing Large Language Models (LLMs) reasoning, yet its reliance on external verifiers limits its scalability. Recent findings suggest that RLVR primarily functions by eliciting latent capabilities, motivating the development of verifier-free algorithms. However, in such settings, standard methods like Group Relative Policy Optimization face a critical challenge: destructive gradient variance that often leads to training collapse. To address this issue, we introduce Verifier-Independent Curriculum Reinforcement Learning (VI-CuRL), a framework that leverages the model's intrinsic confidence to construct a curriculum independent from external verifiers. By prioritizing high-confidence samples, VI-CuRL effectively manages the bias-variance trade-off, specifically targeting the reduction of action and problem variance. We provide a rigoro
    
[^349]: MUSE：支持无缝模型更新的多租户模型服务框架

    MUSE: Multi-Tenant Model Serving With Seamless Model Updates

    [https://arxiv.org/abs/2602.11776](https://arxiv.org/abs/2602.11776)

    MUSE是一个多租户模型服务框架，通过将模型分数与客户端决策边界解耦，使模型更新时无需跨数百个客户重新协调校准阈值，从而实现无缝的模型更新。

    

    在二分类系统中，决策阈值将模型分数转化为实际行动。选择合适的阈值既依赖于底层模型分数的具体分布，也依赖于使用该模型的每个客户的具体业务决策。然而，重新训练模型不可避免地会改变分数分布，导致现有阈值失效。在多租户的“分数即服务”环境中，决策边界位于客户自主管理的基础设施中，这造成了一个严重的瓶颈：重新校准需要在数百个客户之间协调阈值更新，消耗大量人工工时并导致模型更新停滞。我们提出了MUSE，这是一个模型服务框架，通过将模型分数与客户决策边界解耦来实现无缝的模型更新。MUSE专为多租户场景设计，通过基于动态意图的路由实现模型共享来优化基础设施复用，并结合两级评分机制……

    arXiv:2602.11776v4 Announce Type: replace  Abstract: In binary classification systems, decision thresholds translate model scores into actions. Choosing suitable thresholds relies on the specific distribution of the underlying model scores but also on the specific business decisions of each client using that model. However, retraining models inevitably shifts score distributions, invalidating existing thresholds. In multi-tenant Score-as-a-Service environments, where decision boundaries reside in client-managed infrastructure, this creates a severe bottleneck: recalibration requires coordinating threshold updates across hundreds of clients, consuming excessive human hours and leading to model stagnation. We introduce MUSE, a model serving framework that enables seamless model updates by decoupling model scores from client decision boundaries. Designed for multi-tenancy, MUSE optimizes infrastructure re-use by sharing models via dynamic intent-based routing, combined with a two-level sc
    
[^350]: 基于JEPA嵌入的汽车时间序列数据在线监测框架

    Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings

    [https://arxiv.org/abs/2602.09985](https://arxiv.org/abs/2602.09985)

    该论文提出了一种基于JEPA自监督嵌入的在线监测框架，无需异常标签即可检测汽车目标状态表示中的异常，为自动驾驶车辆的安全运行监控提供了新方案。

    

    随着自动驾驶车辆的推出，必须采取措施确保其安全运行。为了监督已经在运行中的系统，通常会采用监测框架。这些框架在后台持续在线运行，监控系统状态并记录异常。本工作提出了一种在线监测框架，用于检测目标状态表示中的异常。其中一个关键挑战是在没有异常标签的情况下构建异常检测框架，而对于未知异常，异常标签通常是不可获得的。为解决这一问题，本工作采用了一种自监督嵌入方法，将目标数据转换到潜在表示空间中。为此，构建了一个基于JEPA的自监督预测任务，使得无需异常标签即可进行训练，并生成丰富的目标嵌入。由此得到的具有强表达能力的JEPA嵌入可作为现有异常检测方法的输入。

    arXiv:2602.09985v2 Announce Type: replace-cross  Abstract: As autonomous vehicles are rolled out, measures must be taken to ensure their safe operation. In order to supervise a system that is already in operation, monitoring frameworks are frequently employed. These run continuously online in the background, supervising the system status and recording anomalies. This work proposes an online monitoring framework to detect anomalies in object state representations. Thereby, a key challenge is creating a framework for anomaly detection without anomaly labels, which are usually unavailable for unknown anomalies. To address this issue, this work applies a self-supervised embedding method to translate object data into a latent representation space. For this, a JEPA-based self-supervised prediction task is constructed, allowing training without anomaly labels and the creation of rich object embeddings. The resulting expressive JEPA embeddings serve as input for established anomaly detection m
    
[^351]: 激光加工中熔池预测的快速精确傅里叶神经算子代理模型

    A Fast and Accurate Fourier Neural Operator-Based Surrogate for Melt-Pool Prediction in Laser Processing

    [https://arxiv.org/abs/2602.06241](https://arxiv.org/abs/2602.06241)

    该论文提出LP-FNO代理模型，通过将瞬态问题在随激光移动的参考系中重构并施加时间平均，将其转化为适合算子学习的准稳态问题，从而快速精确地从工艺参数预测激光加工中的三维温度场与熔池边界，覆盖从传导焊到匙孔焊的宽工艺窗口。

    

    激光焊接的高保真模拟能够捕捉复杂的热流体现象，包括相变、自由表面变形和匙孔动力学，但其高昂的计算成本限制了大规模工艺探索和实时应用。我们提出了激光加工傅里叶神经算子（LP-FNO），这是一种基于傅里叶神经算子的代理模型，通过FLOW-3D WELD生成的多物理场仿真来学习激光加工的参数化解算子。通过将瞬态问题在随激光移动的参考系中重新表述，并应用时间平均，该系统被转化为适合算子学习的准稳态设置，即使在稳定的匙孔焊接区间也同样适用。LP-FNO在覆盖传导模式与匙孔模式的宽工艺窗口内，将工艺参数映射为三维温度场和熔池边界，并采用归一化焓公式

    arXiv:2602.06241v4 Announce Type: replace  Abstract: High-fidelity simulations of laser welding capture complex thermo-fluid phenomena, including phase change, free-surface deformation, and keyhole dynamics, but their computational cost limits large-scale process exploration and real-time use. We present the Laser Processing Fourier Neural Operator (LP-FNO), a Fourier Neural Operator (FNO)-based surrogate model that learns the parametric solution operator of laser processing from multiphysics simulations generated with FLOW-3D WELD. Through a reformulation of the transient problem in a reference frame moving with the laser and the application of temporal averaging, the system is recast into a quasi-steady setting suitable for operator learning, even in the stable keyhole welding regime. LP-FNO maps process parameters to three-dimensional temperature fields and melt-pool boundaries across a broad process window spanning conduction and keyhole regimes using the normalized enthalpy formul
    
[^352]: 具有少量副本测量的实例最优高精度影子断层扫描：一种计量学方法

    Instance-optimal high-precision shadow tomography with few-copy measurements: A metrological approach

    [https://arxiv.org/abs/2602.04952](https://arxiv.org/abs/2602.04952)

    本文给出了高精度影子断层扫描样本复杂度的实例最优刻画 $\tilde{\Theta}(\Gamma_p/\epsilon^2)$，其中 $\Gamma_p$ 由涉及逆 Fisher 信息矩阵的优化公式定义。

    

    我们研究了在现实测量约束下高精度区域内影子断层扫描的样本复杂度。给定一个未知的 $d$ 维量子态 $\rho$ 和一组已知的可观测量 $\{O_i\}_{i=1}^m$，目标是使用可能自适应的测量方法，在 $L_p$ 范数下以精度 $\epsilon$ 估计期望值 $\{\mathrm{tr}(O_i\rho)\}_{i=1}^m$，其中每次测量作用于 $O(\mathrm{polylog}(d))$ 个 $\rho$ 副本。我们关注 $\epsilon$ 低于一个依赖于具体实例的阈值的区域。我们的主要贡献是将样本复杂度刻画为 $\tilde{\Theta}(\Gamma_p/\epsilon^2)$ 的实例最优形式，其中 $\Gamma_p$ 是通过涉及逆 Fisher 信息矩阵的优化公式定义的关于 $\{O_i\}_{i=1}^m$ 的函数。此前，紧致界仅在特殊情况下已知，例如具有 $L_\infty$ 范数误差的 Pauli 影子断层扫描。

    arXiv:2602.04952v3 Announce Type: replace-cross  Abstract: We study the sample complexity of shadow tomography in the high-precision regime under realistic measurement constraints. Given an unknown $d$-dimensional quantum state $\rho$ and a known set of observables $\{O_i\}_{i=1}^m$, the goal is to estimate expectation values $\{\mathrm{tr}(O_i\rho)\}_{i=1}^m$ to accuracy $\epsilon$ in $L_p$-norm, using possibly adaptive measurements that act on $O(\mathrm{polylog}(d))$ number of copies of $\rho$ at a time. We focus on the regime where $\epsilon$ is below an instance-dependent threshold.   Our main contribution is an instance-optimal characterization of the sample complexity as $\tilde{\Theta}(\Gamma_p/\epsilon^2)$, where $\Gamma_p$ is a function of $\{O_i\}_{i=1}^m$ defined via an optimization formula involving the inverse Fisher information matrix. Previously, tight bounds were known only in special cases, e.g. Pauli shadow tomography with $L_\infty$-norm error. Concretely, we first 
    
[^353]: 基于漂移的生成建模

    Generative Modeling via Drifting

    [https://arxiv.org/abs/2602.04770](https://arxiv.org/abs/2602.04770)

    该论文提出了一种名为“漂移模型”的新生成建模范式，通过漂移场在训练过程中演化前推分布并实现一步推理，在 ImageNet 256×256 上取得了一步生成最先进的性能（潜空间 FID 1.54，像素空间 FID 1.61）。

    

    生成建模可以被表述为学习一个映射 f，使其前推分布与数据分布相匹配。前推行为可以在推理阶段迭代执行，例如在扩散模型或流模型中。在本文中，我们提出了漂移模型，这是一种新的生成建模范式，它在训练过程中演化前推分布，并自然地支持一步推理。我们引入了一个漂移场来控制样本的移动，并在分布匹配时达到平衡状态。由此得到一个训练目标，使得神经网络优化器能够演化分布。在实验中，我们的一步生成器在 ImageNet 256×256 上取得了一步生成领域最先进的结果，在潜空间中的 FID 为 1.54，在像素空间中的 FID 为 1.61。我们希望这项工作能为高质量的一步生成开辟新的机遇。

    arXiv:2602.04770v3 Announce Type: replace  Abstract: Generative modeling can be formulated as learning a mapping f such that its pushforward distribution matches the data distribution. The pushforward behavior can be carried out iteratively at inference time, e.g., in diffusion/flow-based models. In this paper, we propose Drifting Models, a new generative modeling paradigm that evolves the pushforward distribution during training and naturally admits one-step inference. We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match. This leads to a training objective that allows the neural network optimizer to evolve the distribution. In experiments, our one-step generator achieves state-of-the-art one-step results on ImageNet 256x256, with FID 1.54 in latent space and 1.61 in pixel space. We hope that our work opens up new opportunities for high-quality one-step generation.
    
[^354]: SparVAR：探索视觉自回归建模中的稀疏性以实现免训练加速

    SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration

    [https://arxiv.org/abs/2602.04361](https://arxiv.org/abs/2602.04361)

    SparVAR是一种免训练的VAR模型加速框架，通过利用注意力汇聚点、跨尺度激活相似性和局部性三大特性，从稀疏决策尺度动态预测高分辨率尺度的稀疏注意力模式，在不跳过尺度、不牺牲图像质量的前提下大幅降低计算复杂度和推理延迟。

    

    视觉自回归建模因其创新性的“下一尺度预测”范式而受到广泛关注。然而，主流的VAR范式在每个自回归步骤中都会关注所有历史尺度的全部token。随着下一尺度分辨率的增大，注意力的计算复杂度随分辨率呈四次方增长，造成显著的推理延迟。以往的加速方法通常跳过高分辨率尺度，这虽然加快了推理速度，但丢弃了高频细节并损害了图像质量。为解决这些问题，我们提出了SparVAR，一个免训练的加速框架，它利用了VAR注意力的三个特性：（i）强注意力汇聚点，（ii）跨尺度激活相似性，以及（iii）显著的局部性。具体而言，我们从稀疏决策尺度动态预测后续高分辨率尺度的稀疏注意力模式，并通过高效的方法构建尺度自相似的稀疏注意力。

    arXiv:2602.04361v3 Announce Type: replace-cross  Abstract: Visual AutoRegressive (VAR) modeling has garnered significant attention for its innovative next-scale prediction paradigm. However, mainstream VAR paradigms attend to all tokens across historical scales at each autoregressive step. As the next scale resolution grows, the computational complexity of attention increases quartically with resolution, causing substantial latency. Prior accelerations often skip high-resolution scales, which speeds up inference but discards high-frequency details and harms image quality. To address these problems, we present SparVAR, a training-free acceleration framework that exploits three properties of VAR attention: (i) strong attention sinks, (ii) cross-scale activation similarity, and (iii) pronounced locality. Specifically, we dynamically predict the sparse attention pattern of later high-resolution scales from a sparse decision scale, and construct scale self-similar sparse attention via an ef
    
[^355]: FlowSteer：迈向由智能体通过强化渐进式画布编辑来设计智能体工作流

    FlowSteer: Towards Agents Designing Agentic Workflows via Reinforced Progressive Canvas Editing

    [https://arxiv.org/abs/2602.01664](https://arxiv.org/abs/2602.01664)

    FlowSteer 提出了一种新范式，由单个轻量级智能体在可执行的工作流画布环境中，通过强化学习训练的渐进式原子画布编辑，端到端地自主设计供下游执行器运行的智能体工作流，并获得语法检查级别的实时执行反馈。

    

    近年来，智能体工作流已被广泛应用于解决复杂的人类任务。然而，现有的工作流构建仍面临若干关键挑战，包括依赖人工的工作流构建方式、缺乏图级别的执行反馈，以及无法在长时程构建过程中进行在线错误修复。为解决这些挑战，我们提出了 FlowSteer——一种“智能体设计智能体工作流”的新范式，即由单个智能体自身端到端地设计出供下游执行器运行的工作流。为支持这一范式，我们引入了工作流画布，这是一种新颖的可执行图状态环境，能够为每一次原子编辑返回经过语法检查的执行反馈。基于该画布，我们进一步提出了强化渐进式画布编辑，其中一个轻量级策略智能体在每一轮基于真实的画布反馈发出一次原子编辑，并通过强化学习进行端到端训练。此外，FlowSteer……（摘要原文在此处截断）

    arXiv:2602.01664v5 Announce Type: replace  Abstract: In recent years, agentic workflows have been widely applied to solve complex human tasks. However, existing workflow construction still faces key challenges, including human-dependent workflow construction, the lack of graph-level execution feedback, and the inability to repair errors in-loop during long-horizon construction. To address these challenges, we propose FlowSteer, a new paradigm of Agent Designing Agentic Workflows - a single agent itself end-to-end designs the workflow that a downstream executor runs. To support this paradigm, we introduce the Workflow Canvas, a novel executable graph-state environment that returns syntax-checked execution feedback for every atomic edit. Built on the canvas, we further propose Reinforced Progressive Canvas Editing, in which a lightweight policy agent issues one atomic edit per turn conditioned on real canvas feedback, and is trained end-to-end via reinforcement learning. Moreover, FlowSt
    
[^356]: 关于临界点的近似计算

    On Approximate Computation of Critical Points

    [https://arxiv.org/abs/2601.21917](https://arxiv.org/abs/2601.21917)

    该论文证明了即使对系数极简单的非凸三次多项式，临界点的粗略近似计算也是计算难解的——除非P=NP，否则不存在多项式时间算法能输出近似临界点。

    

    我们证明了，即使对于一些简单的非凸函数类，计算临界点的粗略近似也是计算上难解的。我们首先提出了一族系数属于{-1,0,1}的n元三次多项式，它们具有精确的临界点，但没有任何多项式时间算法能够以标准表示形式写出一个√n/4-近似临界点。随后我们进一步证明，即使精确临界点具有简短的表示形式，计算困难性依然存在。更具体地说，我们证明：对于每个固定的η>0，除非P=NP，否则不存在多项式时间（甚至伪多项式时间）算法，能够接受一个系数为有理数{c_α}的n元三次多项式，并在该多项式在{-1,0,1}^n中存在临界点的情况下，返回一个梯度范数至多为(max_α |c_α|)·n^(1/2-η)的点。该算法在所有其他输入上可以返回任意点。

    arXiv:2601.21917v2 Announce Type: replace-cross  Abstract: We show that computing even coarse approximations of critical points is intractable for simple classes of nonconvex functions. We first present a family of $n$-variate cubic polynomials with coefficients in $\{-1,0,1\}$ that have exact critical points but for which no polynomial-time algorithm can write down a $\sqrt{n}/4$-approximate critical point in its standard representation. We then show that computational hardness persists when an exact critical point has a short representation. More specifically, we prove that for every fixed $\eta>0$, unless P $=$ NP, no polynomial-time (or even pseudo-polynomial-time) algorithm can take an $n$-variate cubic polynomial with rational coefficients ${c_\alpha}$ and return a point with gradient norm at most $(\max_\alpha |c_\alpha|)n^{1/2-\eta}$ whenever the polynomial has a critical point in $\{-1,0,1\}^n$. The algorithm may return an arbitrary point on all other inputs. Both results are 
    
[^357]: 模糊聚类的统计推断

    Statistical Inference for Fuzzy Clustering

    [https://arxiv.org/abs/2601.02656](https://arxiv.org/abs/2601.02656)

    本文提出加权模糊c均值（WFCM）框架，通过聚类特定权重处理聚类规模不平衡问题，并构建带归一化工作密度的似然框架，首次为模糊聚类提供了原则性的统计推断方法。

    

    聚类是生物医学研究中发现异质性患者亚群的核心工具，在这些场景中，群体边界往往是弥散的而非清晰分离的。尽管诸如模糊 $c$ 均值（FCM）等软聚类方法允许混合隶属关系，但针对模糊聚类的原则性统计推断仍然有限。我们提出了一个针对聚类规模不平衡情形的加权模糊 $c$ 均值（WFCM）框架。通过聚类特定权重 $\mathbf{w}$ 对经典 FCM 准则进行重新平衡，以调整不同规模聚类的相对影响；并且我们从加权目标函数出发，构建了一个带有尺度参数 $\sigma$ 和模糊性参数 $m$ 的归一化工作密度。估计过程首先采用 FCM 风格的交替初始化，随后进行数值似然精化；对于难以处理的归一化常数，采用重要性采样进行近似，并以数据自适应的高斯混合模型作为提议分布……

    arXiv:2601.02656v2 Announce Type: replace-cross  Abstract: Clustering is a central tool in biomedical research for discovering heterogeneous patient subpopulations, where group boundaries are often diffuse rather than sharply separated. While soft clustering methods such as fuzzy $c$-means (FCM) allow mixed memberships, principled statistical inference for fuzzy clustering remains limited.   We propose a weighted fuzzy $c$-means (WFCM) framework for settings with cluster-size imbalance. Cluster-specific weights $\mathbf{w}$ re-balance the classic FCM criterion to adjust the relative influence of clusters of unequal size, and we construct a normalized working density from the weighted objective with scale parameter $\sigma$ and fuzziness parameter $m$. Estimation is performed via FCM-style alternating initialization followed by numerical likelihood refinement; the intractable normalizing constant is approximated by importance sampling using a data-adaptive Gaussian-mixture as the propos
    
[^358]: 面向低能分子结构发现的能量引导生成建模

    Energy-Guided Generative Modeling for Low-Energy Molecular Structure Discovery

    [https://arxiv.org/abs/2512.22597](https://arxiv.org/abs/2512.22597)

    EnFlow 是首个将基于流的构象生成与显式能量势面建模相结合的能量引导生成框架，能够联合实现低能构象系综的生成与分子基态结构的识别。

    

    探索分子能量势面并识别基态构象是计算化学中的核心挑战。然而，利用传统的基于物理的流程从分子图中生成多样的低能构象依然代价高昂。现有的基于学习的方法仍然碎片化：生成模型能够捕捉构象多样性，但往往缺乏可靠的能量校准；而确定性预测器只关注单一结构，无法表征构象系综的变异性。在此，我们提出 EnFlow——据我们所知，这是首个能量引导的生成框架，它将基于流的构象生成与显式的能量势面建模相结合，用于构象系综生成与基态识别的联合任务。通过将生成动力学与学习得到的能量模型相集成，EnFlow 将采样引导至构象势面的低能区域，从而提升结构保真度（摘要原文在此处截断）。

    arXiv:2512.22597v3 Announce Type: replace  Abstract: Exploring molecular energy landscapes and identifying ground-state conformations are central challenges in computational chemistry. However, generating diverse low-energy conformers from molecular graphs remains expensive with traditional physics-based pipelines. Existing learning-based approaches remain fragmented: generative models capture conformational diversity but often lack reliable energy calibration, whereas deterministic predictors focus on a single structure and fail to represent ensemble variability. Here we introduce EnFlow, to our knowledge, the first energy-guided generative framework that couples flow-based conformer generation with explicit energy landscape modeling for joint conformational ensemble generation and ground-state identification. By integrating generative dynamics with a learned energy model, EnFlow guides sampling toward low-energy regions of the conformational landscape, improving structural fidelity u
    
[^359]: 电力系统保护中用于故障分类与定位的机器学习模型鲁棒性评估

    Robustness Evaluation of Machine Learning Models for Fault Classification and Localization In Power System Protection

    [https://arxiv.org/abs/2512.15385](https://arxiv.org/abs/2512.15385)

    本文提出了一个统一评估框架，利用高保真EMT仿真系统性地评估电力系统保护中机器学习故障分类与定位模型在传感器中断、采样率降低和通信丢失等数据退化场景下的鲁棒性。

    

    可再生能源和分布式发电的日益渗透正在改变电力系统，并使依赖固定整定值和本地测量的传统保护方案面临挑战。机器学习（ML）为集中式故障分类（FC）和故障定位（FL）提供了数据驱动的替代方案，能够实现更快、更具适应性的决策。然而，实际部署在很大程度上取决于鲁棒性。保护算法即使在面临缺失、含噪声或退化的传感器数据时也必须保持可靠。这项工作引入了一个统一框架，用于系统性地评估电力系统保护中机器学习模型的鲁棒性。该框架利用高保真电磁暂态（EMT）仿真来模拟真实的退化场景，包括传感器中断、采样率降低和瞬时通信丢失。该框架为模型基准测试提供了一致的方法，量化了受限数据的影�响……

    arXiv:2512.15385v2 Announce Type: replace  Abstract: The growing penetration of renewable and distributed generation is transforming power systems and challenging conventional protection schemes that rely on fixed settings and local measurements. Machine learning (ML) offers a data-driven alternative for centralized fault classification (FC) and fault localization (FL), enabling faster and more adaptive decision-making. However, practical deployment critically depends on robustness. Protection algorithms must remain reliable even when confronted with missing, noisy, or degraded sensor data. This work introduces a unified framework for systematically evaluating the robustness of ML models in power system protection. High-fidelity EMT simulations are used to model realistic degradation scenarios, including sensor outages, reduced sampling rates, and transient communication losses. The framework provides a consistent methodology for benchmarking models, quantifying the impact of limited o
    
[^360]: 超越图元膨胀：基于坐标卷积神经网络的高保真高斯泼溅

    Beyond Primitive Bloat: High-Fidelity Gaussian Splatting via Coordinate-Convolution Neural Network

    [https://arxiv.org/abs/2512.12898](https://arxiv.org/abs/2512.12898)

    该论文从理论上证明离散高斯表示存在图元数量指数级增长导致的保真度瓶颈，并提出以坐标卷积神经网络（CCNN）将细节恢复建模为粗渲染之上的连续神经场，从而无需图元膨胀即可实现高保真高斯泼溅。

    

    高斯泼溅（GS）以其快速的训练和实时渲染彻底改变了新视角合成（NVS），但将其保真度提升到任意高的水平仍然是一个悬而未决的挑战。我们通过证明离散高斯表示需要指数级增长的图元数量才能提升训练PSNR，从理论上解释了这一障碍，这不可避免地引发优化问题和验证集过拟合。为了在不产生图元膨胀的情况下绕过这一扩展极限，我们将细节恢复表述为粗略渲染之上的连续神经场，并证明要将逼近风险单调地降低至零，需要联合利用低保真度信号、局部邻域以及坐标查询。在这一结果的指导下，我们利用现有的坐标卷积神经网络（CCNN）来增强泼溅渲染，该网络将渲染的低保真度信号与2D像素坐标查询进行联合卷积以合成……

    arXiv:2512.12898v3 Announce Type: replace-cross  Abstract: Gaussian Splatting (GS) has revolutionized Novel View Synthesis (NVS) with fast training and real-time rendering, yet pushing its fidelity to arbitrary high levels remains an open challenge. We theoretically explain this barrier by proving that discrete Gaussian representations require an exponentially increasing number of primitives to raise training PSNR, which inevitably triggers optimization issues and validation overfitting. To bypass this scaling limit without primitive bloat, we formulate detail recovery as a continuous neural field over coarse renderings and prove that monotonically reducing approximation risk to zero requires jointly leveraging low-fidelity signals, local neighborhood, and coordinate queries. Guided by this result, we augment splatted renderings with an existing Coordinate-Convolution Neural Network (CCNN) that jointly convolves rendered low-fidelity signals with 2D pixel coordinate queries to synthesi
    
[^361]: 基于少量校准条件的扩散Transformer细粒度缓存

    Fine-Grained Caching for Diffusion Transformers with Few Calibration Conditions

    [https://arxiv.org/abs/2512.05134](https://arxiv.org/abs/2512.05134)

    提出了一种无需训练的细粒度缓存框架，利用条件共模抑制效应和少量校准条件为扩散Transformer构建固定的模块复用调度，从而大幅降低图像与视频生成的计算开销。

    

    扩散Transformer需要反复进行去噪器评估，使得图像与视频生成的计算成本十分高昂。我们提出了一个无需训练的框架，该框架利用少量校准条件构建固定的、细粒度的模块复用调度，而无需进行调度搜索。该设计源于在确定性采样轨迹上观察到的一种经验性效应：在初始噪声固定的情况下，若干模块输出中依赖于条件的偏差在相邻步骤间以相似的方式演化，因此时间差分能够衰减其大部分变化。我们将这一效应命名为条件共模抑制。受此观察启发，我们使用连续特征位移的归一化比率对“时间步—层—模块”位置进行排序，并通过两阶段校准构建了一个Cache Book（缓存簿）。第二阶段在记录每个临时比特的索引分数后，用该比特更新仅用于评分的参考，同时保留完整的……（原文摘要此处截断）

    arXiv:2512.05134v3 Announce Type: replace-cross  Abstract: Diffusion transformers require repeated denoiser evaluations, making image and video generation computationally expensive. We propose a training-free framework that uses a few calibration conditions to construct a fixed, fine-grained module-reuse schedule without schedule search. The design is motivated by an empirical effect along deterministic sampling trajectories: with initial noise fixed, condition-dependent deviations in several module outputs evolve similarly across adjacent steps, so temporal differencing attenuates much of their variation. We term this effect Conditional Common-Mode Rejection (CCMR). Motivated by this observation, we rank timestep--layer--module locations using normalized ratios of consecutive feature displacements and construct a Cache Book through two-stage calibration. The second stage updates a scorer-only reference with each provisional bit after recording its indexed score, while retaining full-c
    
[^362]: 面向物理约束的系统级材料设计的分层深度研究

    Hierarchical Deep Research for Physically Grounded System-Level Materials Design

    [https://arxiv.org/abs/2511.18303](https://arxiv.org/abs/2511.18303)

    本文提出基于开源语言模型的分层多智能体系统"Deep Tree of Research"，在盲评中于47个智能体中排名第一，其材料设计方案在第一性原理模拟中超越最佳商业系统，并通过传感器实验得到物理验证。

    

    设计多组分材料系统需要关于成分、界面、加工和器件操作的证据，而这些证据散布于文献之中。语言模型研究智能体能够收集这些证据，但其报告很少经过物理验证。本文展示了Deep Tree of Research（深度研究树）——一个基于可本地部署的开源权重语言模型构建的分层多智能体系统，在涵盖27个纳米材料与器件主题的盲评中，于47个智能体中排名第一。在五项设计任务的十项第一性原理模拟指标中，本地智能体提出的候选方案有九项优于最佳商业系统。在一个传感案例研究中，该智能体为短链污染物全氟戊酸提出了氟化银纳米颗粒/MXene结构；在专家将锚定基团替换为胺基后，所制备的传感器在自来水中表现出选择性和可重复的响应，与界面机制相一致。

    arXiv:2511.18303v3 Announce Type: replace  Abstract: Designing multicomponent materials systems requires evidence on composition, interfaces, processing and deviceoperation that is scattered across the literature. Language-model research agents can collect this evidence, but their reports are seldom tested against physics. Here we show that Deep Tree of Research, a hierarchical multi-agent system built on locally deployable open-weight language models, ranks first among 47 agents in blinded evaluationacross 27 nanomaterials and device topics. Candidates from local agents outperform those from the best commercial systems on nine of ten first-principles simulation metrics across five design tasks. In a sensing case study, the agent nominated a fluorinated silver-nanoparticle/MXene architecture for the short-chain pollutant perfluoropentanoic acid; after an expert replaced the anchor with an amine, fabricated sensors respond selectively and reproduciblyintap water, consistent with interfa
    
[^363]: Nowcast3D：高效的降水临近预报

    Nowcast3D for effective precipitation nowcasting

    [https://arxiv.org/abs/2511.04659](https://arxiv.org/abs/2511.04659)

    提出灰箱学习框架Nowcast3D，将物理结构化神经算子（三维平流、局地扩散与微物理过程）与条件扩散模型相结合，直接基于三维雷达体扫数据实现物理引导且不确定性量化的降水临近预报。

    

    极端降水的演变依赖于随高度变化的风场和相互耦合的物理过程，这凸显了三维信息对降水临近预报的重要性。雷达体扫数据提供了这样的观测基础。然而，大多数基于学习的系统要么将这些体扫数据压缩为二维产品，丢失了刻画随高度变化的风暴演变所需的垂直信息，要么依赖纯粹数据驱动的映射，无法充分表征极端降水的物理演变。我们提出了Nowcast3D，一个面向三维雷达体扫临近预报的灰箱学习框架，它将具有物理结构的神经算子与条件扩散模型相结合。神经算子通过三维平流、局地扩散和微物理过程生成物理引导的预报；以此预报为条件，扩散模型生成具有不确定性量化的集合预报。

    arXiv:2511.04659v5 Announce Type: replace  Abstract: The evolution of extreme precipitation depends on height-varying wind fields and coupled physical processes, underscoring the importance of three-dimensional information for precipitation nowcasting. Radar volume scans provide such an observational basis. Most learning-based systems, however, either compress these scans into two-dimensional products, losing vertical information needed for height-dependent storm evolution, or rely on purely data-driven mappings that inadequately represent the physical evolution of extreme precipitation. We introduce Nowcast3D, a gray-box learning framework for 3D radar-volume nowcasting, which integrates physically structured neural operators with a conditional diffusion model. The neural operators produce a physics-guided forecast via 3D advection, local diffusion and microphysics processes. Conditioned on this forecast, the diffusion model generates uncertainty-quantified ensembles. Trained on provi
    
[^364]: 偶对称与椭圆对称条件下变分推断的广义保证

    Generalized Guarantees for Variational Inference in the Presence of Even and Elliptical Symmetry

    [https://arxiv.org/abs/2511.01064](https://arxiv.org/abs/2511.01064)

    该论文将反向KL散度变分推断在偶对称下恢复均值、椭圆对称下恢复相关矩阵的保证，从对数凹目标分布推广到更一般的单峰（含重尾）目标分布，并扩展至仅部分对称的情形。

    

    变分推断（VI）通过在一族易于处理的分布 $\mathcal Q$ 中寻找最佳匹配 $q$ 来近似目标密度 $p$。变分近似通过最小化某种散度求得，最常用的是反向 Kullback-Leibler 散度 $\text{KL}(q||p)$，不过也可以考虑其他散度。在实践中，$p \notin \mathcal Q$，因此最优变分近似存在误设（misspecification）。尽管存在这种误设，最近的研究表明，对于对数凹的目标分布 $p$，$\text{KL}(q||p)$ 的最小化点在偶对称条件下保证能够恢复均值，在椭圆对称条件下同样保证能够恢复相关矩阵。我们在若干方向上对这一结果进行了扩展。首先，我们放宽了对数凹性假设，仅要求 $p$ 是单峰的，从而允许目标分布具有重尾特性。其次，我们针对 $p$ 仅在某些（方向上）表现出对称性的情形获得了相应保证……

    arXiv:2511.01064v4 Announce Type: replace-cross  Abstract: Variational inference (VI) approximates a target density $p$ by the best match $q$ in a family $\mathcal Q$ of tractable distributions. The variational approximation is found by minimizing a divergence, most commonly the reverse Kullback-Leibler divergence $\text{KL}(q||p)$, although other divergences can be considered. In practice, $p \notin \mathcal Q$ and so the optimal variational approximation is misspecified. Despite this misspecification, it was recently shown that the minimizer of $\text{KL}(q||p)$ is guaranteed to recover the mean in the presence of even symmetry and likewise the correlation matrix in the presence of elliptical symmetry for a log-concave target $p$. We extend this result in several directions. First, we relax the assumption of log-concavity and instead merely require $p$ to be unimodal, allowing for heavy-tailed targets. Second, we obtain guarantees for cases where $p$ exhibits symmetry along some but 
    
[^365]: 通过反馈修复奖励函数以缓解奖励欺骗

    Repairing Reward Functions with Feedback to Mitigate Reward Hacking

    [https://arxiv.org/abs/2510.13036](https://arxiv.org/abs/2510.13036)

    该论文提出了基于偏好的奖励修复（PBRR）自动化迭代框架，通过从少量人类偏好数据中学习一个加性的、依赖于状态转移的修正项来修复人工指定的代理奖励函数，以低成本缓解奖励欺骗问题。

    

    为强化学习（RL）智能体人工设计的奖励函数经常与人类真实的、不可观测的目标不一致，因此只能作为代理目标。优化错误指定的代理奖励函数往往会导致奖励欺骗，从而产生与人类真实目标不一致的策略。另一种方法是进行基于人类反馈的强化学习，即通过收集人类对轨迹对的偏好从头学习一个奖励函数，然而构建此类数据集的成本高昂。为了解决这两种方法的局限性，我们提出了基于偏好的奖励修复（PBRR）：一个自动化迭代框架，通过从偏好中学习一个加性的、依赖于状态转移的修正项来修复人工指定的代理奖励函数。手动指定的奖励函数在真实目标下可能产生高度次优的策略，然而仅需对少量转移进行修正……（摘要原文在此处截断）

    arXiv:2510.13036v3 Announce Type: replace  Abstract: Human-designed reward functions for reinforcement learning (RL) agents are frequently misaligned with the humans' true, unobservable objectives, and thus act only as proxies. Optimizing for a misspecified proxy reward function often induces reward hacking, resulting in a policy misaligned with the human's true objectives. An alternative is to perform RL from human feedback, which involves learning a reward function from scratch by collecting human preferences over pairs of trajectories. However, building such datasets is costly. To address the limitations of both approaches, we propose Preference-Based Reward Repair (PBRR): an automated iterative framework that repairs a human-specified proxy reward function by learning an additive, transition-dependent correction term from preferences. A manually specified reward function can yield policies that are highly suboptimal under the ground-truth objective, yet corrections on only a few tr
    
[^366]: 大语言模型pass@k缩放规律的高效预测

    Efficient Prediction of pass@k Scaling in Large Language Models

    [https://arxiv.org/abs/2510.05197](https://arxiv.org/abs/2510.05197)

    本文提出了一种稳健的估计框架，能够在远小于大规模尝试次数的采样预算下高效准确地预测大语言模型的pass@k缩放行为，从而帮助模型提供商和监管机构评估模型的能力与风险。

    

    评估前沿AI系统的能力与风险是一个重要的研究领域；最近的研究表明，重复采样可以显著地同时提升这两者。例如，重复采样使模型能够解决困难的数学和编程问题，但也可能增加其被越狱攻击的风险。这些结果引出了一个关键问题：在采样预算远小于实际尝试次数的情况下，如何准确预测模型在扩展到海量尝试次数时的行为？这个问题与每天为数亿用户提供服务的模型提供商，以及寻求防止危害的政府监管机构直接相关。为回答这一问题，我们做出了三项贡献。首先，我们发现用于拟合这些规律的标准方法存在统计上的缺陷，阻碍了预测的准确性，尤其是在数据有限的场景中。其次，我们通过引入一个稳健的估计框架来弥补这些缺陷

    arXiv:2510.05197v2 Announce Type: replace  Abstract: Assessing the capabilities and risks of frontier AI systems is a critical area of research; recent work has shown that repeated sampling can dramatically increase both. For instance, repeated sampling allows models to solve difficult math and coding problems, but it may also increase their risk of being jailbroken. Such results raise a crucial question: how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget? This question is directly relevant to model providers, who serve hundreds of millions of users daily, and to governmental regulators, who seek to prevent harms. To answer this question, we make three contributions. First, we find that standard methods for fitting these laws suffer from statistical shortcomings that hinder predictions, especially in data-limited scenarios. Second, we remedy these shortcomings by introducing a robust estimation framework
    
[^367]: 复杂度正则化的近端策略优化

    Complexity-Regularized Proximal Policy Optimization

    [https://arxiv.org/abs/2509.20509](https://arxiv.org/abs/2509.20509)

    该论文提出用香农熵与非平衡度乘积构成的自调节复杂度项替代传统熵正则化来改进近端策略优化（PPO），在保持策略有益随机性的同时，当策略高度不确定时降低正则化压力，从而让学习更专注于奖励优化。

    

    策略梯度方法通常依赖熵正则化来防止过早收敛。然而，不加区分地最大化熵会将策略推向均匀分布，若调参不当，往往会压过奖励信号的作用。我们提出用一种自调节的复杂度项来替代标准的熵项，该复杂度项定义为香农熵与非平衡度的乘积，用于量化与均匀分布之间的距离。与偏好最大无序性的纯熵不同，这种复杂度度量在完全确定性分布和完全均匀分布下均为零，即只有在秩序与随机性之间表现出有意义相互作用的系统中，它才严格为正。这些特性确保策略保持有益的随机性，同时在策略高度不确定时降低正则化压力，使学习能够专注于奖励优化。我们提出了复杂度正则化的近端策略优化（CR-PPO）……

    arXiv:2509.20509v3 Announce Type: replace-cross  Abstract: Policy gradient methods usually rely on entropy regularization to prevent premature convergence. However, maximizing entropy indiscriminately pushes the policy towards a uniform distribution, often overriding the reward signal if not optimally tuned. We propose replacing the standard entropy term with a self-regulating complexity term, defined as the product of Shannon entropy and disequilibrium, which quantifies the distance from the uniform distribution. Unlike pure entropy, which favors maximal disorder, this complexity measure is zero for both fully deterministic and perfectly uniform distributions, i.e., it is strictly positive for systems that exhibit a meaningful interplay between order and randomness. These properties ensure the policy maintains beneficial stochasticity while reducing regularization pressure when the policy is highly uncertain, allowing learning to focus on reward optimization. We introduce Complexity-R
    
[^368]: 用于奇异衍生品的条件深度列维模型：历史感知路径生成与P-Q收益诊断

    Conditional Deep Levy Models for Exotic Derivatives: History-Aware Path Generation and P-Q Payoff Diagnostics

    [https://arxiv.org/abs/2509.13374](https://arxiv.org/abs/2509.13374)

    该论文提出基于去噪列维概率模型的条件式历史感知股指路径生成器，通过严格控制变量的匹配实验证明，列维稳定参数α=1.9相比高斯极限α=2在路径依赖衍生品收益诊断（CRPS、覆盖率、区间评分等指标）上取得显著改进。

    

    我们开发了一个基于去噪列维概率模型（DLPM）的具有历史感知能力的股指路径生成器，并通过路径依赖的衍生品收益来评估其预测分布。一个条件U-Net结合了合约状态、指数标识、60天和252天的收益率历史，以及采用坐标级稳定混合的金融正则化。在一个匹配实验中，我们保持初始化、训练数据、批次顺序、目标函数、计算预算和评估检查点全部固定不变，比较α=1.9与高斯极限α=2的表现。在来自八个中美股指的6,824个按时间顺序排列的窗口上，终端CRPS从0.050585降至0.048031，降低了5.05%，路径能量提升了4.03%。配对的60个自然日CRPS区间为[-0.004022, -0.001298]。覆盖率从56.68%上升至71.00%，同时适当的90%区间评分提升了20.30%。经验基准仍然具有竞争力：全池…（摘要在此处截断）

    arXiv:2509.13374v3 Announce Type: replace-cross  Abstract: We develop a history-aware equity-index path generator based on Denoising Levy Probabilistic Models (DLPMs) and evaluate its predictive distributions through path-dependent derivative payoffs. A conditional U-Net combines contract state, index identity, 60- and 252-day return histories, and financial regularization with coordinate-wise stable mixing. A matched experiment holds initialization, training data, batch order, objective, computational budget and evaluation checkpoint fixed while comparing $\alpha=1.9$ with the Gaussian limit $\alpha=2$. On 6,824 chronological windows from eight Chinese and U.S. indices, terminal CRPS falls from 0.050585 to 0.048031, a 5.05% reduction, and path energy improves by 4.03%. The paired 60-calendar-day CRPS interval is [-0.004022, -0.001298]. Coverage rises from 56.68% to 71.00% while the proper 90% interval score improves by 20.30%. Empirical references remain competitive: the full-pool ter
    
[^369]: 基于金刚石自旋传感器的自适应微波接收

    Adaptive Microwave Reception with Diamond Spin Sensors

    [https://arxiv.org/abs/2509.08171](https://arxiv.org/abs/2509.08171)

    本文推导了金刚石自旋微波接收机中反馈延迟的精确阈值，并据此提出了一种基于导频记录一阶圆矩方向估计的自适应相位控制器，从而在载波漂移条件下有效保持二进制相移键控符号的对准增益。

    

    arXiv:2509.08171v2 通告类型： replace-cross 摘要：在金刚石自旋微波接收机中，导频测量可以识别出最能区分二进制相移键控符号的自旋控制相位。应用该相位需要时间，而在此期间载波漂移可能会抹去对准所带来的增益。我们从导频记录中的相位信息推导出了单个有效载荷自旋的精确反馈延迟阈值。在有限的控制相位选项集合上，当一阶圆矩非零时，导频记录仅通过其一阶圆矩的方向进入最小误差更新决策。这一结果启发了一种控制器设计：该控制器从正交导频测量中估计该方向，并在预测增益因扩散而丧失时保留现有的相位。对于有限自旋系综，我们基于光学读出和载波相位不确定性（包括自旋间共享的涨落）对光子计数分布的可区分性进行了界定。在匹配条件下的分组仿真……（摘要原文在此处截断）

    arXiv:2509.08171v2 Announce Type: replace-cross  Abstract: In a diamond-spin microwave receiver, pilot measurements can identify the spin-control phase that best separates binary phase-shift keyed symbols. Applying that phase takes time, during which carrier drift can erase the gain in alignment. We derive an exact feedback-delay threshold for a single payload spin from the phase information in the pilot record. On a finite menu of control phases, the pilot record enters the minimum-error update decision only through the direction of its first circular moment when that moment is nonzero. This result motivates a controller that estimates the direction from orthogonal pilot measurements and retains the existing phase whenever the predicted gain is lost to diffusion. For finite spin ensembles, we bound the distinguishability of the photon-count distributions in terms of optical readout and carrier-phase uncertainty, including fluctuations shared across spins. Packet simulations at matched
    
[^370]: 聚类无关假设下的联邦学习一次性聚类

    One-Shot Clustering for Federated Learning Under Clustering-Agnostic Assumption

    [https://arxiv.org/abs/2509.01587](https://arxiv.org/abs/2509.01587)

    本文提出了一种聚类无关的一次性聚类联邦学习算法（OCFL），能够自动检测出最适合聚类的最早时机，从而实现个性化模型的训练。

    

    联邦学习（FL）是一种被广泛采用的去中心化学习范式，它允许从多个数据源训练同一个模型，而无需在参与客户端之间直接传输数据。自2016年诞生以来，联邦学习已分化出众多子领域，用以处理特定应用的问题，例如数据异构性或资源分配。其中一个子领域——聚类联邦学习（CFL）——致力于解决将客户端群体划分为不同群组以提供个性化模型的问题。尽管该领域已发表了一些杰出的工作，但这一问题在很大程度上仍未被充分探索，因为其基本假设和设置与标准联邦学习略有不同。在本工作中，我们提出了一次性聚类联邦学习（OCFL），这是一种聚类无关的算法，能够自动检测出适合进行聚类的最早时机。我们通过实验验证……

    arXiv:2509.01587v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) is a widespread and well-adopted paradigm of decentralised learning that allows training one model from multiple sources without the need to transfer data between participating clients directly. Since its inception in 2016, it has been divided into numerous subfields that deal with application-specific issues, such as data heterogeneity or resource allocation. One such sub-field, Clustered Federated Learning (CFL), is dealing with the problem of clustering the population of clients into separate cohorts to deliver personalised models. Although a few remarkable works have been published in this domain, the problem remains largely unexplored, as its basic assumptions and settings differ slightly from those of standard FL. In this work, we present One-Shot Clustered Federated Learning (OCFL), a clustering-agnostic algorithm that can automatically detect the earliest suitable moment for clustering. We empiri
    
[^371]: 无需观测收益的马尔可夫博弈

    Playing Markov Games Without Observing Payoffs

    [https://arxiv.org/abs/2509.00179](https://arxiv.org/abs/2509.00179)

    本文将“不观测收益也能竞争”的结果从矩阵博弈扩展到零和对称马尔可夫博弈，证明仅凭转移动态知识和对手动作序列即可与拥有完全信息的对手竞争，并将该学习问题归约为在线学习。

    

    在不确定性下进行优化是学习与决策中的一个基本问题，在多智能体系统中尤为如此。此前，Feldman、Kalai 和 Tennenholtz [2010] 证明了在重复的对称两人矩阵博弈中，只要能够观测到对手的动作，即使不观测收益，也可以高效地与之竞争。本文引入并形式化了一类新的零和对称马尔可夫博弈，将对“对称性”的概念从矩阵博弈扩展到了马尔可夫设定中。我们证明，即使不观测收益，一个仅知晓转移动态并只观测对手动作序列的玩家，仍然可以与一个可能完全掌握博弈信息的对手进行竞争。我们在这一设定下形式化了三种不同的对称性概念，并证明在这些条件下，该学习问题可以归约为在线学习的一个实例，从而使玩家能够渐进地与对手竞争。

    arXiv:2509.00179v4 Announce Type: replace-cross  Abstract: Optimization under uncertainty is a fundamental problem in learning and decision-making, particularly in multi-agent systems. Previously, Feldman, Kalai, and Tennenholtz [2010] demonstrated the ability to efficiently compete in repeated symmetric two-player matrix games without observing payoffs, as long as the opponents actions are observed. In this paper, we introduce and formalize a new class of zero-sum symmetric Markov games, which extends the notion of symmetry from matrix games to the Markovian setting. We show that even without observing payoffs, a player who knows the transition dynamics and observes only the opponents sequence of actions can still compete against an adversary who may have complete knowledge of the game. We formalize three distinct notions of symmetry in this setting and show that, under these conditions, the learning problem can be reduced to an instance of online learning, enabling the player to asym
    
[^372]: 论医学场景中路径归因可解释性方法的“缺失性”概念：指导选择具有医学意义的基线

    On the notion of missingness for path attribution explainability methods in medical settings: Guiding the selection of medically meaningful baselines

    [https://arxiv.org/abs/2508.14482](https://arxiv.org/abs/2508.14482)

    该论文针对医学影像归因可解释性方法，提出“语义缺失性”新概念，指出全零等标准基线在医学场景中语义上无意义，主张基线应代表疾病特征缺失的临床合理状态，并据此提出反事实引导的基线选择方法。

    

    深度学习模型的可解释性仍然是一个重大挑战，尤其是在医学领域，可解释的输出对于临床信任和透明度至关重要。路径归因方法（如积分梯度 Integrated Gradients, IG）依赖于一个代表信息特征缺失的基线，这一概念通常被称为“缺失性”（missingness）。然而，常用的基线（如全零输入）在医学场景中往往在语义上毫无意义，因为在医学影像中像素强度值承载着临床意义。在这项工作中，我们重新审视了医学影像中的缺失性概念，揭示了标准基线在该场景下的局限性，并形式化了一种更严格的缺失性概念，我们称之为“语义缺失性”（semantic missingness）：基线不仅必须缺乏信号，还必须代表一种临床上合理的状态，即疾病相关特征缺失的状态。这一表述启发了一种反事实引导的基线选择方法。（注：原文摘要在此处被截断）

    arXiv:2508.14482v4 Announce Type: replace  Abstract: The explainability of deep learning models remains a significant challenge, particularly in the medical domain where interpretable outputs are essential for clinical trust and transparency. Path attribution methods such as Integrated Gradients (IG) rely on a baseline that represents the absence of informative features, a notion commonly referred to as missingness. However, commonly used baselines (such as all-zero inputs) are often semantically meaningless in medical contexts, where intensity values carry clinical significance. In this work, we revisit the notion of missingness for medical imaging, expose the limitations of standard baselines in this setting, and formalize a stricter missingness we term semantic missingness: a baseline must not merely lack signal, but must represent a clinically plausible state in which the disease-related features are absent. This formulation motivates a counterfactual-guided approach to baseline se
    
[^373]: 具有分组子空间结构的极稀疏多层网络中的完美聚类

    Perfect Clustering in Extremely Sparse Multilayer Networks with Groupwise Subspace Structure

    [https://arxiv.org/abs/2507.19423](https://arxiv.org/abs/2507.19423)

    本文提出一个极其广义的多层网络聚类模型——各层遵循广义随机点积图模型且可按潜在子空间结构分组，并证明了即使各层网络极度稀疏、逐层分析失效时，仍能完美地恢复具有不同子空间结构的层分组。

    

    本文研究了建立在同一节点集合上的多层网络，其中每一层都遵循广义随机点积图模型。此外，所有层可以被划分为若干组，每组拥有其独特的潜在子空间结构，而除此之外各层的连接概率矩阵可以各不相同。这一设置极其广泛，将大多数多层网络模型都作为特例包含在内。同时，我们关注层网络可能极度稀疏的情形，在这种情况下逐层分析是不可靠的。该问题也可以被视为经典问题的推广：从一整个矩阵集合中恢复共享结构，扩展到结构仅在特定矩阵组内共享的情形。此设置下的关键任务是恢复具有唯一子空间结构的层分组。

    arXiv:2507.19423v2 Announce Type: replace-cross  Abstract: The paper studies a multilayer network on the same collection of nodes, where each layer follows the Generalized Random Dot Product Graph Model. Furthermore, all layers can be partitioned into groups, each group with its own unique latent subspace structure, but otherwise the matrices of connection probabilities can all be different. This setting is extremely general and includes most multilayer network models as special cases. In addition, we are interested in the situation where the layer networks may be extremely sparse and, therefore, the layer-by-layer analysis is unreliable. This problem can also be seen as a generalization of the classic setting in which one recovers a shared structure across an entire collection of matrices, extending it to the case where such structures are shared only within specific groups of matrices. The key task in this setting is to recover the groups of layers with unique subspace structures, si
    
[^374]: 面向未见奖励的文生图提示词优化

    Text-to-Image Prompt Optimization for Unseen Rewards

    [https://arxiv.org/abs/2506.16853](https://arxiv.org/abs/2506.16853)

    提出RATTPO方法，通过对比高分与低分提示词，将优化历史中的隐式证据转化为显式搜索指导，从而实现对设计时未知奖励的文生图提示词优化。

    

    文生图（T2I）提示词优化通过重写用户提示词，在目标奖励下提升生成图像的质量。现有方法通常假设奖励是已知的，依赖于针对特定奖励的训练或手工设计的指令。我们则研究针对未见奖励的提示词优化，这类奖励在设计阶段是未知的，仅在测试阶段可以通过评分访问。我们观察到，移除奖励特定的指导会显著降低基于大语言模型（LLM）的优化器的搜索效率，即使它们的优化历史中包含着奖励所偏好内容的证据。为应对这一挑战，我们提出了RATTPO（奖励自适应测试时提示词优化），这是一种奖励无关的通用方法，能够将这些隐式证据转化为显式的搜索指导。在每一轮优化中，RATTPO通过对比高分与低分提示词，推断出关于有前景重写方向的相互竞争的假设，进而生成并评估候选提示词。

    arXiv:2506.16853v3 Announce Type: replace  Abstract: Text-to-image (T2I) prompt optimization rewrites user prompts to improve the quality of generated images under a target reward. Existing methods typically assume a known reward, relying on reward-specific training or handcrafted instructions. We instead study prompt optimization for unseen rewards, which are unknown at design time and accessible only through scores at test time. We observe that removing reward-specific guidance substantially reduces the search efficiency of LLM-based optimizers, even though their optimization histories contain evidence of what the reward favors. To address this challenge, we introduce RATTPO (Reward-Adaptive Test-Time Prompt Optimization), a reward-generic method that turns this implicit evidence into explicit search guidance. At each round, RATTPO contrasts high- and low-scoring prompts to infer competing hypotheses about promising rewriting directions, then generates and evaluates candidates under 
    
[^375]: 用于高效多目标强化学习的局部策略流形

    Local Policy Manifolds for Efficient Multi-Objective Reinforcement Learning

    [https://arxiv.org/abs/2506.04022](https://arxiv.org/abs/2506.04022)

    提出LLE-MORL方法，通过利用策略参数空间与性能空间之间的局部关系追踪策略流形，借助局部线性外推实现高效搜索，无需大量重新训练即可快速生成覆盖多目标权衡的高质量策略集合。

    

    多目标强化学习（MORL）旨在优化多个经常相互冲突的目标，以提高强化学习在实际任务中的灵活性和可靠性。这通常通过寻找一组多样化的、非支配的策略来实现，这些策略在性能空间中构成帕累托前沿。然而，构建这样的策略集在计算上代价高昂，因为它需要找到的不是一个单一的最优策略，而是一组能够覆盖目标之间广泛权衡范围的策略集合。我们提出了LLE-MORL，一种通过利用高维策略参数空间与性能空间之间的局部关系来追踪策略流形的方法。这种结构化表示使得算法能够通过局部线性外推在连续解域内进行高效搜索，从而无需大量重新训练即可快速生成高质量解。我们还对该方法提供了理论分析。

    arXiv:2506.04022v3 Announce Type: replace  Abstract: Multi-objective reinforcement learning (MORL) aims at optimising several, often conflicting goals to improve the flexibility and reliability of RL in practical tasks. This is typically achieved by finding a set of diverse, non-dominated policies that form a Pareto front in the performance space. However, constructing such a policy set remains computationally demanding, as it requires finding not a single optimal policy, but a set of policies to cover a wide range of trade-offs among the objectives. We introduce LLE-MORL, an approach that traces policy manifolds by utilising the local relationship between the high-dimensional policy parameter space and the performance space. This structured representation enables an efficient search within contiguous solution domains via locally linear extrapolation, allowing for the rapid generation of high-quality solutions without extensive retraining. We also provide a theoretical analysis of the 
    
[^376]: TRIM：通过目标行级迭代度量驱动剪枝实现极端稀疏化

    TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning

    [https://arxiv.org/abs/2505.16743](https://arxiv.org/abs/2505.16743)

    TRIM提出一种行级迭代度量驱动剪枝方法，通过为层内每个输出行分配不同的稀疏比率，在高稀疏度（70-80%）下显著降低大语言模型的困惑度，例如使Qwen2.5-14B在80%稀疏度下的困惑度近乎减半。

    

    大型语言模型（LLMs）带来了巨大的计算和内存挑战。现有的一次性剪枝方法通常对层内的所有行使用统一的稀疏度目标，这在高稀疏度下可能过于粗糙。本工作引入了TRIM（目标行级迭代度量驱动剪枝），这是一种基于前向统计信息的封装方法，能够在保持模型权重冻结的同时，为各个输出行分配不同的稀疏比率。TRIM将“剪枝哪些权重”（评分）与“每行剪枝多少”（分配）这两个问题分开处理，并可与现有的评分器和层级分配方法组合使用。在Qwen2.5、LLaMA-2和OPT模型上的实验表明，其最大收益出现在70-80%稀疏度区间。在80%稀疏度下，将TRIM添加到OWL上，可使Qwen2.5-14B的WikiText-2困惑度从348.48降至180.67，使OPT-13B的困惑度从6461.43降至324.14。我们得出结论，细粒度的行级分配是显著提升模型性能的有效机制。

    arXiv:2505.16743v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) present substantial computational and memory challenges. Existing one-shot pruning methods often use a uniform sparsity target for all rows within a layer, which can be too coarse at high sparsity. This work introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a forward-statistics-based wrapper that allocates different sparsity ratios to individual output rows while keeping model weights frozen. TRIM separates the choice of which weights to prune (scoring) from how many to prune in each row (allocation), and composes with existing scorers and layer-wise allocations. Across Qwen2.5, LLaMA-2, and OPT models, its largest gains occur at 70-80% sparsity. At 80%, adding TRIM to OWL reduces WikiText-2 perplexity from 348.48 to 180.67 for Qwen2.5-14B and from 6461.43 to 324.14 for OPT-13B. We conclude that fine-grained, row-wise allocation is an effective mechanism for substantially improvin
    
[^377]: 电力系统保护中数据稀疏性对机器学习故障检测的影响

    Impact of Data Sparsity on Machine Learning for Fault Detection in Power System Protection

    [https://arxiv.org/abs/2505.15560](https://arxiv.org/abs/2505.15560)

    提出了一个评估框架，系统性地量化了数据稀疏性对基于机器学习的电力系统故障检测与故障线路识别性能的影响。

    

    德国向基于可再生能源的电力系统转型正在重塑电网运行，需要先进的监测和控制手段来管理分散式发电。机器学习（ML）已成为电力系统保护的强大工具，特别是在输电网的故障检测（FD）和故障线路识别（FLI）方面。然而，机器学习模型的可靠性取决于数据的质量和可用性。由传感器故障、通信中断或采样率降低所导致的数据稀疏性，对基于机器学习的故障检测和故障线路识别构成了挑战。然而，在本研究之前，其影响尚未得到系统性验证。为此，我们提出了一个评估数据稀疏性对基于机器学习的故障检测和故障线路识别性能影响的框架。我们模拟了真实的数据稀疏性场景，评估其影响，得出定量见解，并通过将该评估策略应用于现有的基于机器学习的系统来证明其有效性。

    arXiv:2505.15560v2 Announce Type: replace  Abstract: Germany's transition to a renewable energy-based power system is reshaping grid operations, requiring advanced monitoring and control to manage decentralized generation. Machine learning (ML) has emerged as a powerful tool for power system protection, particularly for fault detection (FD) and fault line identification (FLI) in transmission grids. However, ML model reliability depends on data quality and availability. Data sparsity resulting from sensor failures, communication disruptions, or reduced sampling rates poses a challenge to ML-based FD and FLI. Yet, its impact has not been systematically validated prior to this work. In response, we propose a framework to assess the impact of data sparsity on ML-based FD and FLI performance. We simulate realistic data sparsity scenarios, evaluate their impact, derive quantitative insights, and demonstrate the effectiveness of this evaluation strategy by applying it to an existing ML-based 
    
[^378]: 大数据系统中不规则多变量时间序列预测的自适应线性网络

    Adaptive Linear Networks for Irregular Multivariate Time Series Forecasting in Big Data Systems

    [https://arxiv.org/abs/2505.00590](https://arxiv.org/abs/2505.00590)

    该论文提出AiT模型，其核心ALinear模块通过无需值投影的时间条件化注意力池化，从观测或查询坐标生成样本自适应的时间权重，从而在不规则多变量时间序列的任意查询坐标上实现高效且灵活的预测。

    

    大规模传感、临床、交通、能源和气候系统通常会产生观测稀疏、采样间隔异构且变量不同步的不规则多变量时间序列。在外部给定的未来查询坐标上进行预测，要求模型能够适应可变的观测长度和时间位置，同时保持高效的变量交互。我们提出了AiT（Adaptive inverted Transformer，自适应反转Transformer），其ALinear（Adaptive Linear，自适应线性）模块能够从观测或查询坐标生成样本自适应的时间权重。更准确地说，ALinear是一种无需值投影的时间条件化注意力池化。可用的观测或查询坐标生成相应的键或查询，固定形状的对应部分可以使用学习到的默认槽位，而未经投影的输入则作为值。因此，“线性”这一名称仅指最终的加权和输出形式，而并非等同于……（摘要原文在此处被截断）

    arXiv:2505.00590v2 Announce Type: replace  Abstract: Large-scale sensing, clinical, transportation, energy, and climate systems often produce irregular multivariate time series with sparse observations, heterogeneous sampling intervals, and asynchronous variables. Forecasting at externally supplied future query coordinates requires a model to accommodate variable observation lengths and time locations while retaining efficient variable interaction. We present AiT (Adaptive inverted Transformer), whose ALinear (Adaptive Linear) module generates sample-adaptive temporal weights from observation or query coordinates. More precisely, ALinear is time-conditioned attention pooling without value projection. Available observation or query coordinates generate the corresponding keys or queries, the fixed-shape counterpart may use learned default slots, and the unprojected inputs serve as values. The name "Linear" therefore denotes only the final weighted-sum output form and not equivalence to a
    
[^379]: 病毒学能力测试（VCT）：一个多模态病毒学问答基准

    Virology Capabilities Test (VCT): A Multimodal Virology Q&A Benchmark

    [https://arxiv.org/abs/2504.16137](https://arxiv.org/abs/2504.16137)

    研究者提出了由博士级病毒学专家构建的322题多模态基准VCT，结果显示OpenAI的o3以43.8%的准确率超过94%的专家病毒学家，凸显了AI在病毒学领域的能力及其双用途风险。

    

    我们提出了病毒学能力测试（VCT），这是一个用于衡量大型语言模型（LLM）解决复杂病毒学实验室操作方案问题能力的基准。VCT由数十位博士级专家病毒学家提供输入构建而成，包含322个多模态问题，涵盖了病毒学实验室实际工作中必需的基础知识、隐性知识和视觉知识。VCT难度很高：即使是能够使用互联网的专家病毒学家，在其专业子领域内的特定问题上平均得分也仅为22.1%。然而，表现最强的LLM——OpenAI的o3——达到了43.8%的准确率，即便在专家的细分专业领域内，也超过了94%的专家病毒学家。提供专家级病毒学故障排除能力本质上是双用途的：它对有益的研究有用，但也可能被滥用。因此，公开可用的模型表现优于病毒学家这一事实……

    arXiv:2504.16137v3 Announce Type: replace-cross  Abstract: We present the Virology Capabilities Test (VCT), a large language model (LLM) benchmark that measures the capability to troubleshoot complex virology laboratory protocols. Constructed from the inputs of dozens of PhD-level expert virologists, VCT consists of $322$ multimodal questions covering fundamental, tacit, and visual knowledge that is essential for practical work in virology laboratories. VCT is difficult: expert virologists with access to the internet score an average of $22.1\%$ on questions specifically in their sub-areas of expertise. However, the most performant LLM, OpenAI's o3, reaches $43.8\%$ accuracy, outperforming $94\%$ of expert virologists even within their sub-areas of specialization. The ability to provide expert-level virology troubleshooting is inherently dual-use: it is useful for beneficial research, but it can also be misused. Therefore, the fact that publicly available models outperform virologists 
    
[^380]: 因果片段：逐片分析和改进脉冲神经网络

    Causal pieces: analysing and improving spiking neural networks piece by piece

    [https://arxiv.org/abs/2504.14015](https://arxiv.org/abs/2504.14015)

    提出“因果片段”新概念来分析脉冲神经网络，证明片段内输出脉冲时间对输入和参数局部Lipschitz连续，且因果片段数量可作为衡量SNN逼近能力的有效指标。

    

    我们提出了“因果片段”，这是一个用于分析脉冲神经网络（SNN）的新概念，其灵感来源于人工神经网络（ANN）中用于研究表达能力和可训练性的“线性片段”。因果片段将具有单脉冲编码的前馈SNN的输入和参数空间划分为不同的区域，在每个区域中，相同的子网络导致输出脉冲的产生。对于由具有大膜时间常数的基于电流的泄漏积分发放（LIF）神经元构成的网络，我们证明了在每个因果片段内，输出脉冲时间相对于输入和网络参数是局部Lipschitz连续的。我们进一步证明了依赖于因果片段数量的逼近误差下界。因此，因果片段的数量是衡量SNN逼近能力的一个度量指标，该指标在脉冲时间不连续的情况下仍然有效，并且适用于同时具有兴奋性和抑制性突触的网络。

    arXiv:2504.14015v2 Announce Type: replace-cross  Abstract: We introduce "causal pieces", a novel concept for analysing spiking neural networks (SNNs), inspired by "linear pieces" used to study expressivity and trainability in artificial neural networks (ANNs). Causal pieces partition the input and parameter space of a feedforward SNN with single-spike coding into distinct regions where the same subnetwork causes the output spikes. For networks of current-based leaky integrate-and-fire (LIF) neurons with large membrane time constants, we show that within each causal piece, output spike times are locally Lipschitz continuous with respect to inputs and network parameters. We further prove a lower bound on the approximation error that depends on the number of causal pieces. Thus, the number of causal pieces is a measure of the approximation capabilities of SNNs, which is valid despite spike-time discontinuities and applies to networks with both excitatory and inhibitory synapses. Empirical
    
[^381]: Mamba-3D作为掩码自编码器用于医学超声视频的精确且数据高效的分析

    Mamba-3D as Masked Autoencoders for Accurate and Data-Efficient Analysis of Medical Ultrasound Videos

    [https://arxiv.org/abs/2503.20258](https://arxiv.org/abs/2503.20258)

    提出E-ViM³——一种保留视频3D结构的Vision Mamba网络，结合包围全局标记和时空链式掩码策略进行自监督预训练，实现了医学超声视频的精确且数据高效的分析。

    

    超声视频是临床医学影像数据的重要形式，基于深度学习的自动化分析可以提高诊断准确性和临床效率。然而，标注数据的稀缺性以及视频分析固有的挑战阻碍了相关方法的发展。在这项工作中，我们提出了E-ViM³，这是一种数据高效的Vision Mamba网络，它保留了视频数据的3D结构，增强了长程依赖和归纳偏置，以更好地建模时空相关性。通过我们设计的包围全局标记（EGT），该模型比竞争方法更有效地捕获和聚合全局特征。为了进一步提高数据效率，我们采用掩码视频建模进行自监督预训练，并提出了时空链式（STC）掩码策略，旨在适应各种视频场景。实验表明，E-ViM³达到了最先进的性能……

    arXiv:2503.20258v2 Announce Type: replace-cross  Abstract: Ultrasound videos are an important form of clinical imaging data, and deep learning-based automated analysis can improve diagnostic accuracy and clinical efficiency. However, the scarcity of labeled data and the inherent challenges of video analysis have impeded the advancement of related methods. In this work, we introduce E-ViM$^3$, a data-efficient Vision Mamba network that preserves the 3D structure of video data, enhancing long-range dependencies and inductive biases to better model space-time correlations. With our design of Enclosure Global Tokens (EGT), the model captures and aggregates global features more effectively than competing methods. To further improve data efficiency, we employ masked video modeling for self-supervised pre-training, with the proposed Spatial-Temporal Chained (STC) masking strategy designed to adapt to various video scenarios. Experiments demonstrate that E-ViM$^3$ performs as the state-of-the-
    
[^382]: 论结构化剪枝在小语言模型预训练中的效率

    On the Efficiency of Structured Pruning in Small Language Model Pretraining

    [https://arxiv.org/abs/2503.05920](https://arxiv.org/abs/2503.05920)

    提出IDEA Prune集成式放大-剪枝流水线，在单一余弦退火学习率调度下统一放大模型预训练、迭代结构化剪枝和恢复过程，在全成本核算下比传统分阶段剪枝方法更高效地训练出小语言模型。

    

    在边缘设备上部署生成式语言模型的公司，出于法律、数据治理或产品定制化等要求，通常需要从头训练小模型。另一种方法是先训练一个放大的模型，再通过结构化剪枝将其压缩到目标尺寸，但一旦将放大模型的预训练成本计算在内，这种做法的优势就变得不明确了。我们引入了“放大-剪枝”框架，在全成本核算的条件下研究这一问题，并表明传统的放大-剪枝流水线——即分阶段地训练放大模型、剪枝并恢复——在相同的总token预算下并不总是优于直接训练小模型。为克服这一局限性，我们提出了IDEA Prune，这是一种集成式的放大-剪枝流水线，它在单一的余弦退火学习率调度下，将放大模型预训练、迭代结构化剪枝和恢复统一为一体。

    arXiv:2503.05920v2 Announce Type: replace-cross  Abstract: Companies that deploy generative language models on edge devices often need to train small models from scratch due to legal, data-governance, or product-customization requirements. An alternative approach is to train an enlarged model and compress it to the target size via structured pruning, but this advantage becomes unclear once the cost of pretraining the enlarged model is included. We introduce the enlarge-and-prune framework to study this question under full-cost accounting, and show that conventional enlarge-and-prune pipelines -- which train the enlarged model, prune it, and recover in separate stages -- do not consistently outperform direct small-model training under the same total token budget. To overcome this limitation, we propose IDEA Prune, an integrated enlarge-and-prune pipeline that unifies enlarged model pretraining, iterative structured pruning, and recovery under a single cosine annealing learning rate sche
    
[^383]: 一种用于无模型扩散磁共振成像配准的可操纵深度网络

    A Steerable Deep Network for Model-Free Diffusion MRI Registration

    [https://arxiv.org/abs/2501.04794](https://arxiv.org/abs/2501.04794)

    提出了一种基于SE(3)等变可操纵卷积的几何深度学习框架，实现了无需模型假设的原始扩散磁共振数据非刚性配准，且无需旋转平移数据增强。

    

    非刚性配准对医学图像分析至关重要，但由于扩散磁共振成像（dMRI）具有高维度和空间-角度依赖性，其非刚性配准仍然充满挑战。我们提出了一种新颖的几何深度学习框架，用于对原始dMRI数据进行无模型的非刚性配准。我们在三维旋转平移变换SE(3)的联合作用下，在原始的空间-角度采集空间中构建配准问题。我们框架的核心在于等变空间-角度卷积与作用于球谐系数场的三维可操纵卷积之间的等价性，我们利用这一等价性构建了SE(3)等变的U-Net网络。这一设计选择的附带好处是无需使用旋转平移版本的自身数据进行数据增强。我们还构建了水位移分布之间的最大均值差异损失，该损失在傅里叶空间中计算为信号差异平方的加权和……

    arXiv:2501.04794v3 Announce Type: replace-cross  Abstract: Nonrigid registration is vital to medical image analysis but remains challenging for diffusion MRI (dMRI) due to its high-dimensional, spatio-angular dependence. We present a novel, geometric deep learning framework for model-free, nonrigid registration of raw dMRI data. We formulate registration in the native spatio-angular acquisition space under the joint action of 3D roto-translations, $\mathrm{SE}(3)$. Central to our framework is the equivalence between equivariant spatio-angular convolutions and 3D steerable convolutions on spherical-harmonic coefficient fields, which we exploit to construct an $\mathrm{SE}(3)$-equivariant U-Net. A by-product of this design choice is freedom from having to augment the data with roto-translated versions of itself. We also formulate a maximum mean discrepancy (MMD) loss between water-displacement distributions, evaluated in Fourier space as a weighted sum of squared signal differences, and 
    
[^384]: MMGenBench：一种从文本到图像生成视角评估大型多模态模型的自动化流水线

    MMGenBench: An Automated Pipeline for Evaluating LMMs from a Text-to-Image Generation Perspective

    [https://arxiv.org/abs/2411.14062](https://arxiv.org/abs/2411.14062)

    MMGenBench是一个无需预定义问题或参考描述的自动化基准框架，通过“图像描述生成→文生图重建→编码器比对”的流水线从原始图像评估大型多模态模型，对68个LMM的评估揭示了总分可能掩盖的模式特定重建质量差异。

    

    现有基准测试通常依赖预定义的问题或参考描述，这限制了它们在新图像集上评估详细描述的灵活性。我们提出了MMGenBench，这是一个自动化基准测试框架，可直接从原始图像评估大型多模态模型（LMMs），无需逐图像的问题或参考描述。该流水线先生成图像描述，再使用冻结的文本到图像模型重建图像，并通过冻结的图像编码器比较原始图像与重建图像。它能够产出可复用的逐图像评分和可检查的重建结果，既支持模型比较，也支持对可重建视觉信息的诊断。我们构建了包含10,000张生成图像的MMGenBench-Domain，以及包含1,284张图像、覆盖13种重叠视觉模式的MMGenBench-Test。对68个LMM的评估揭示了重建质量上存在总体分数可能掩盖的模式特定差异。在1,000张人工标注的……（摘要在此处被截断）

    arXiv:2411.14062v3 Announce Type: replace-cross  Abstract: Existing benchmarks often rely on predefined questions or reference captions, limiting their flexibility for evaluating detailed descriptions on new image collections. We present MMGenBench, an automated benchmark framework for evaluating Large Multimodal Models (LMMs) from raw images without per-image questions or reference captions. The pipeline generates descriptions, reconstructs images with a frozen text-to-image model, and compares the original and reconstructed images using a frozen image encoder. It produces reusable per-image scores and inspectable reconstructions, supporting both model comparison and diagnosis of reconstructible visual information. We construct MMGenBench-Domain with 10,000 generated images and MMGenBench-Test with 1,284 images spanning 13 overlapping visual patterns. Evaluation of 68 LMMs reveals pattern-specific differences in reconstruction quality that overall scores can obscure. On 1,000 human-la
    
[^385]: ProFL：策略性鲁棒最优联邦学习

    ProFL: Performative Robust Optimal Federated Learning

    [https://arxiv.org/abs/2410.18075](https://arxiv.org/abs/2410.18075)

    本文提出ProFL，一种在模型诱导分布漂移下无需凸性策略性风险和无噪声数据假设、可使联邦学习模型收敛到策略性最优点而非仅仅稳定点的鲁棒方法。

    

    策略性预测是一种刻画机器学习模型在部署过程中因其使用而引发数据分布漂移的框架。随着训练好的模型被使用，所诱导的数据分布会不断演变，从而导致数据偏离原始分布。这种由模型诱导的分布漂移在联邦学习（FL）中的影响在现实应用场景中越来越可能出现。最近提出的一种方法将策略性预测扩展到了联邦学习，但所得到的模型收敛到的是策略性稳定点，该点可能距离策略性最优点很远。早期在集中式场景下的研究表明，在模型诱导的分布漂移下可以实现策略性最优点，但这些方法要求策略性风险是凸的，且训练数据是无噪声的。这些假设在现实的联邦学习系统中往往不成立。本文针对这些问题……（原文摘要在此处截断）

    arXiv:2410.18075v3 Announce Type: replace  Abstract: Performative prediction is a framework that captures distribution shifts that occur during the training of machine learning models due to their deployment. As the trained model is used, the induced data distribution evolves, leading to deviations from the original data distribution. The impact of such model-induced distribution shifts in federated learning (FL) is increasingly likely to transpire in real-life use cases. A recently proposed approach extends performative prediction to FL with the resulting model converging to a performative stable point, which may be far from the performative optimal point. Earlier research in centralized settings has shown that the performative optimal point can be achieved under model-induced distribution shifts, but these approaches require the performative risk to be convex and the training data to be noiseless. These assumptions are often violated in realistic FL systems. This paper addresses thes
    
[^386]: 低资源语言的语法错误纠正：以哲尔马语为例

    Grammatical Error Correction for Low-Resource Languages: The Case of Zarma

    [https://arxiv.org/abs/2410.15539](https://arxiv.org/abs/2410.15539)

    本文首次针对低资源语言哲尔马语开展语法错误纠正研究，比较了基于规则、机器翻译和大语言模型三种方法，发现基于M2M100的机器翻译方法表现最佳，并构建了超过25万样本的数据集用于评估。

    

    语法错误纠正（GEC）旨在提升文本的质量与可读性。以往针对该任务的研究主要集中在高资源语言上，而低资源语言缺乏强大的工具。为弥补这一不足，我们针对哲尔马语开展了一项语法错误纠正研究，这是一种在西非有超过五百万人使用的语言。我们比较了三种方法：基于规则的方法、机器翻译（MT）模型以及大语言模型（LLM）。我们使用一个包含超过25万个样本的数据集（包括合成数据和人工标注数据）对GEC模型进行了评估。结果表明，基于M2M100的机器翻译方法优于其他方法，在自动评估（AE）中检测率达95.82%、建议准确率达78.90%；在由母语者进行的人工评估（ME）中，语法和逻辑纠错的平均得分为3.0分（满分5.0分）。基于规则的方法对拼写错误有效，但在复杂语境下则难以奏效。

    arXiv:2410.15539v4 Announce Type: replace  Abstract: Grammatical error correction (GEC) aims to improve text quality and readability. Previous work on the task focused primarily on high-resource languages, while low-resource languages lack robust tools. To address this shortcoming, we present a study on GEC for Zarma, a language spoken by over five million people in West Africa. We compare three approaches: rule-based methods, machine translation (MT) models, and large language models (LLMs). We evaluated GEC models using a dataset of more than 250,000 examples, including synthetic and human-annotated data. Our results showed that the MT-based approach using M2M100 outperforms others, with a detection rate of 95.82% and a suggestion accuracy of 78.90% in automatic evaluations (AE) and an average score of 3.0 out of 5.0 in manual evaluation (ME) from native speakers for grammar and logical corrections. The rule-based method was effective for spelling errors but failed on complex context
    
[^387]: 通过预条件器解决注意力核回归问题

    Solving Attention Kernel Regression Problem via Pre-conditioner

    [https://arxiv.org/abs/2308.14304](https://arxiv.org/abs/2308.14304)

    本文通过定义两个问题，设计了用于注意力矩阵的代理快速算法及其回归问题求解算法。

    

    注意力机制是大型语言模型的关键，注意力矩阵作为这种方案的算法和计算瓶颈。本文通过定义两个问题，旨在设计用于代理注意力矩阵的快速算法以及解决回归问题。首先考虑矩阵$A\in \mathbb{R}^{n\times d}$的矩阵指数$A^\top A$作为一个代理，然后设计两种类型的回归问题的算法：对于任意正整数$j$，分别是$\min_{x\in \mathbb{R}^d}\|(A^\top A)^jx-b\|_2$和$\min_{x\in \mathbb{R}^d}\|A(A^\top A)^jx-b\|_2$。研究这些回归的算法是至关重要的，因为矩阵指数可以通过这些较小的问题逐项逼近。第二个代理是将指数逐项应用于Gram矩阵，记为$\exp(AA^\top)$，并解决回归问题$\min

    arXiv:2308.14304v2 Announce Type: replace  Abstract: The attention mechanism is the key to large language models, and the attention matrix serves as an algorithmic and computational bottleneck for such a scheme. In this paper, we define two problems, motivated by designing fast algorithms for proxy of attention matrix and solving regressions against them. Given an input matrix $A\in \mathbb{R}^{n\times d}$ with $n\gg d$ and a response vector $b$, we first consider the matrix exponential of the matrix $A^\top A$ as a proxy, and we in turn design algorithms for two types of regression problems: $\min_{x\in \mathbb{R}^d}\|(A^\top A)^jx-b\|_2$ and $\min_{x\in \mathbb{R}^d}\|A(A^\top A)^jx-b\|_2$ for any positive integer $j$. Studying algorithms for these regressions is essential, as matrix exponential can be approximated term-by-term via these smaller problems. The second proxy is applying exponential entrywise to the Gram matrix, denoted by $\exp(AA^\top)$ and solving the regression $\min
    
[^388]: 多项式次数多项式核的快速草图化

    Fast Sketching of Polynomial Kernels of Polynomial Degree

    [https://arxiv.org/abs/2108.09420](https://arxiv.org/abs/2108.09420)

    本文提出了一种新的无意识草图方法，消除了运行时间对多项式核次数的依赖，大幅改进了神经正切核、反余弦核等慢增长核的近似运行速度。

    

    核方法是机器学习中的基础方法，更快的核近似算法可以直接加速机器学习中的许多核心任务。多项式核尤其重要，因为其他核通常可以通过泰勒级数展开用多项式核来近似。最近的无意识草图技术将运行时间对多项式核次数 q 的依赖从指数级降低到多项式级，这对于高斯核非常有用，因为在数据半径有界且岭参数 λ ≥ 1/poly(n) 的情况下，q 可以选择为多对数级。然而，对于增长较慢的核，例如神经正切核和反余弦核，q 需要取多项式级，而先前的工作在运行时间上会产生多项式因子的减速。我们提出了一种新的无意识草图方法，通过消除这种依赖性，大幅改进了运行时间。

    arXiv:2108.09420v2 Announce Type: replace-cross  Abstract: Kernel methods are fundamental in machine learning, and faster algorithms for kernel approximation provide direct speedups for many core tasks in machine learning. The polynomial kernel is especially important as other kernels can often be approximated by the polynomial kernel via a Taylor series expansion. Recent techniques in oblivious sketching reduce the dependence in the running time on the degree $q$ of the polynomial kernel from exponential to polynomial, which is useful for the Gaussian kernel, for which $q$ can be chosen to be polylogarithmic (for data of bounded radius and ridge parameter $\lambda\geq 1/\mathrm{poly}(n)$). However, for more slowly growing kernels, such as the neural tangent and arc-cosine kernels, $q$ needs to be polynomial, and previous work incurs a polynomial factor slowdown in the running time. We give a new oblivious sketch which greatly improves upon this running time, by removing the dependence
    
[^389]: HindSight：一种用于表示部分-整体层次结构的基于图的视觉模型架构

    HindSight: A Graph-Based Vision Model Architecture For Representing Part-Whole Hierarchies

    [https://arxiv.org/abs/2104.03722](https://arxiv.org/abs/2104.03722)

    该论文提出了HindSight架构，通过将图像划分为不同层级的图像块并构建全连接图来编码图像中的部分-整体层次结构，可训练为通用视觉编码器以支持多种视觉下游任务。

    

    本文提出了一种以图的形式编码图像中部分-整体层次结构表示的模型架构。其核心思想是将图像划分为不同层级的图像块，然后将所有这些图像块作为全连接图的节点。在每次图迭代中，使用动态特征提取模块从这些图像块中提取特征表示。这使我们能够学习到包含图像固有部分-整体层次信息的丰富图表示。利用适当的自监督训练技术，这样的模型可以被训练成一个通用视觉编码器模型，进而用于各种视觉相关的下游任务（例如图像分类、目标检测、图像描述生成等）。

    arXiv:2104.03722v2 Announce Type: replace-cross  Abstract: This paper presents a model architecture for encoding the representations of part-whole hierarchies in images in form of a graph. The idea is to divide the image into patches of different levels and then treat all of these patches as nodes for a fully connected graph. A dynamic feature extraction module is used to extract feature representations from these patches in each graph iteration. This enables us to learn a rich graph representation of the image that encompasses the inherent part-whole hierarchical information. Utilizing proper self-supervised training techniques, such a model can be trained as a general purpose vision encoder model which can then be used for various vision related downstream tasks (e.g., Image Classification, Object Detection, Image Captioning, etc.).
    
[^390]: 随机前向模式自动微分优化算法

    Randomized Forward Mode of Automatic Differentiation for Optimization Algorithms. (arXiv:2310.14168v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2310.14168](http://arxiv.org/abs/2310.14168)

    该论文介绍了一种随机前向模式自动微分优化算法，通过在神经网络的正向传递中计算损失函数的方向导数来更新参数。算法通过采样不同概率分布的随机方向，使用正向模式自动微分计算雅可比向量乘积，并提供了对其收敛速度和计算复杂性的严格分析。

    

    神经网络中的反向传播利用了自动微分的基本要素，即反向模式微分，或称为向量雅可比乘积(VJP)，或在微分几何的背景下被称为拉回过程。梯度的计算对于使用梯度下降方法更新神经网络参数非常重要。在本研究中，我们提出了一种通用的随机方法，通过使用通过正向模式AD或雅可比向量乘积(JVP)高效计算的损失函数的方向导数来更新神经网络的参数。这些JVP沿着从不同概率分布（例如伯努利、正态、维格纳、拉普拉斯和均匀分布）采样的随机方向计算。梯度的计算在神经网络的正向传递过程中进行。我们还提供了对所提出方法的严格分析，包括收敛速度以及计算复杂性。

    Backpropagation within neural networks leverages a fundamental element of automatic differentiation, which is referred to as the reverse mode differentiation, or vector Jacobian Product (VJP) or, in the context of differential geometry, known as the pull-back process. The computation of gradient is important as update of neural network parameters is performed using gradient descent method. In this study, we present a genric randomized method, which updates the parameters of neural networks by using directional derivatives of loss functions computed efficiently by using forward mode AD or Jacobian vector Product (JVP). These JVP are computed along the random directions sampled from different probability distributions e.g., Bernoulli, Normal, Wigner, Laplace and Uniform distributions. The computation of gradient is performed during the forward pass of the neural network. We also present a rigorous analysis of the presented methods providing the rate of convergence along with the computat
    
[^391]: 极端环境下的个体治疗效果

    Individual Treatment Effects in Extreme Regimes. (arXiv:2306.11697v1 [stat.ME])

    [http://arxiv.org/abs/2306.11697](http://arxiv.org/abs/2306.11697)

    本文提出了一种新的框架，通过测量潜在结果在存在或缺乏治疗的情况下的尾部衰减率变化，来估计极端环境下的个体治疗效果（ITE$_2$）。

    

    了解极端环境下的个体治疗效果对于描述不同干预策略的风险至关重要。但极端环境数据很难收集，因为它在实践中很少被观察到。为了解决这个问题，我们提出了一个新的框架来估计极端环境下的个体治疗效果（ITE$_2$）。具体而言，我们通过测量潜在结果在存在或缺乏治疗的情况下的尾部衰减率变化来量化这种效果。随后，我们建立了 ITE$_2$ 的计算条件，并开发了计算算法。我们在各种合成和半合成数据集上演示了我们提出的方法的功效。

    Understanding individual treatment effects in extreme regimes is important for characterizing risks associated with different interventions. This is hindered by the fact that extreme regime data may be hard to collect, as it is scarcely observed in practice. In addressing this issue, we propose a new framework for estimating the individual treatment effect in extreme regimes (ITE$_2$). Specifically, we quantify this effect by the changes in the tail decay rates of potential outcomes in the presence or absence of the treatment. Subsequently, we establish conditions under which ITE$_2$ may be calculated and develop algorithms for its computation. We demonstrate the efficacy of our proposed method on various synthetic and semi-synthetic datasets.
    

