# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Joints, Camera, Action! Co-Generating 3D Motion and Video for Arbitrary Rigs](https://arxiv.org/abs/2610.28632) | 该论文提出通过少量交叉注意力层将支持任意骨架的独立3D动作模型与预训练视频模型相连接，直接共同生成视频、可回放的3D动作以及将动作配准到3D场景的相机。 |
| [^2] | [Learning Adaptable Dexterous Skills via Hybrid Dexterous Mimicking](https://arxiv.org/abs/2610.28628) | AdaDex提出混合灵巧模仿方法，通过在运动追踪环境和目标达成环境中联合训练共享策略，将几秒钟的手-物体轨迹转化为既保留人类先验又能泛化到任意物体姿态的可适应灵巧技能。 |
| [^3] | [Planning under Noise with Neural Operators](https://arxiv.org/abs/2610.28623) | 本文提出PUN神经算子，通过在屏蔽泊松随机最优控制框架中为障碍物引入反射诺伊曼边界条件、为目标引入狄利克雷条件，并结合网格无关的蒙特卡洛求解器，学习噪声感知的价值场，从而在未见过的几何环境中实现噪声下的可靠规划。 |
| [^4] | [Adaptive Multi-Token Prediction: How far should we go?](https://arxiv.org/abs/2610.28619) | 该论文提出自适应多词元预测（AMTP），将多词元预测的推测长度选择形式化为在线控制问题，通过轻量级控制器利用验证反馈动态调整最佳起草长度，无需重新训练即可提升语言模型解码效率。 |
| [^5] | [Efficient Edge-aware Attention Network for Graph Generation](https://arxiv.org/abs/2610.28618) | 提出一种用于图生成的边感知注意力层，通过头级分数调制注入结构信息并以轻量残差机制更新边表示，可基于FlexAttention高效实现，在保持生成质量的同时将峰值GPU内存降低约65%并显著提升训练效率。 |
| [^6] | [Turning TS2Vec into a Foundation Model for Time Series Classification](https://arxiv.org/abs/2610.28602) | 仅通过三项改动——在合成语料库上预训练、双视图输入分支以及多编码器知识蒸馏——就将原本需针对每个数据集单独训练的TS2Vec成功转变为可跨数据集复用、仅160万参数的时间序列分类基础模型TS2Vec-FM。 |
| [^7] | [Neural Implicit Energy Optimization for LiDAR Mapping](https://arxiv.org/abs/2610.28598) | 该论文将神经激光雷达建图表述为神经隐式表示参数空间上的能量泛函优化，通过基于水平集形状算子（曲率）的能量泛函正则化符号距离场，将经典几何先验适配到百万点级大规模场景，在保持几何保真度的同时实现更平滑的重建。 |
| [^8] | [PaPi: Parallel Picard-Duhamel Integration for Fast Neuro-Spectral Architectures](https://arxiv.org/abs/2610.28591) | PaPi通过指数积分与Duhamel-Picard方案实现时间并行推理，证明了该迭代对所有流均呈指数收敛并给出显式误差界，从而将神经谱架构的训练与推理加速多达两个数量级，使低维PDE学习问题的训练时间缩短至一秒以内，同时保持有竞争力的预测精度。 |
| [^9] | [WildMAE: Structure-Aware Masked Autoencoding for Wildlife Image Reconstruction](https://arxiv.org/abs/2610.28590) | WildMAE通过在MAE解码器中加入零门控卷积残差路径（提供二维邻域结构先验）与门控的掩码-可见交叉注意力机制，并结合渐进式的多目标训练策略，在75%掩码率下显著提升了野生动物相机陷阱图像的重建质量。 |
| [^10] | [Two $\operatorname{ReLU}$ Layers is All You Need](https://arxiv.org/abs/2610.28587) | 本文证明了任意深度的 ReLU 网络都可以被仅含两个隐藏层的 ReLU 网络精确模拟，并给出了基于同调代数填充和组合方法的两条证明路径。 |
| [^11] | [Continuous-Time Policy Gradient from Discrete Data: Convergence Analysis and Model-Free Algorithms](https://arxiv.org/abs/2610.28580) | 该论文建立了一个从离散数据中学习的连续时间策略梯度框架，证明了熵正则化策略流向最优解的指数收敛性，并通过与PhiBE结合给出了收敛界显式分离评论家误差、策略回归误差和步长影响的无模型行动者-评论家算法。 |
| [^12] | [The Token-Energy Law of LLM Decode: Active Bytes and Realized Time Account for Joules per Token](https://arxiv.org/abs/2610.28576) | 提出了令牌-能量定律，用活跃字节和实际解码时间两个维度以可校准的仿射形式精确刻画LLM解码中每令牌的GPU能耗，并解释了批处理摊销与KV历史增长对能耗的影响。 |
| [^13] | [Evolving Semantic Indexes for Generative Retrieval](https://arxiv.org/abs/2610.28568) | 本文首次将LLM驱动的进化程序搜索应用于生成式检索，通过本地开源大模型自动变异语义标识符的各组件设计，并借助轻量代理解码器提供快速反馈，显著提升了检索性能。 |
| [^14] | [Continual Concept Personalization in Diffusion Models via Residual-Space Adaptation](https://arxiv.org/abs/2610.28567) | 该论文提出将残差空间适配扩展到预训练扩散模型的随机适配过程中，实现了持续概念个性化，在保持近乎零遗忘的同时达到了所有竞争方法中最高的概念保真度。 |
| [^15] | [The Dynamics of Collapse and Recovery in Clustered Federated Learning](https://arxiv.org/abs/2610.28565) | 该论文将聚类联邦学习中的聚类崩溃形式化为未知动态系统的控制问题，识别出贪婪路由导致崩溃自我强化的初始化区域，并通过有限时域误差动力学证明路由干预可通过反复训练被忽视的模型来逆转崩溃。 |
| [^16] | [Asymmetric Bidirectional Predictive Dynamics in Google Search and COVID-19 Time Series: A Multi-Horizon Forecast-Error Analysis](https://arxiv.org/abs/2610.28563) | 该研究提出了一种结合对数RMSE增益与预测误差分布熵（FEDE）及持久性加权DPRS评分的多预测期预测误差分析框架，揭示了谷歌搜索与COVID-19时间序列之间双向预测关系的非对称动态特征。 |
| [^17] | [OptScientist: Automated Discovery of Optimization Laws](https://arxiv.org/abs/2610.28562) | 提出了 OptScientist 框架，将符号优化问题推断模型与 LLM 智能体系统相结合，能够直接从数据中自动发现简洁的符号优化定律，并在包含 99 个物理优化问题的新型基准 OptFeynman 上对超过 80% 的问题成功恢复了真实定律。 |
| [^18] | [Convergence of stochastic gradient descent for learning linear neural networks](https://arxiv.org/abs/2610.28561) | 本文首次为深度线性神经网络中的随机梯度下降提供了几乎必然收敛保证，证明在温和条件下（无需Lipschitz性质或白化数据）SGD可收敛到临界点集合。 |
| [^19] | [When Do LLMs Help with Feature Selection? A Systematic Empirical Study](https://arxiv.org/abs/2610.28556) | 这项系统性实证研究表明，在表格数据特征选择任务中，经典统计方法在预测性能上普遍优于被测试的7B规模大语言模型选择器及经GRPO强化学习适配的模型。 |
| [^20] | [Learning Computational Structure for Sample-Efficient Prediction](https://arxiv.org/abs/2610.28555) | 受生物神经回路空间组织启发，该论文提出 CSFRBench 基准和 Learned Wiring（LeWi）方法——让每个神经元学习低维嵌入并仅与邻近神经元连接——以显式利用函数的组合稀疏结构，从而显著提升高维预测的样本效率。 |
| [^21] | [Discount Yourself: Self-inclusive Freshness-weighted Gossip for Decentralised Learning under Client Outages](https://arxiv.org/abs/2610.28545) | 该论文提出，在去中心化联邦学习中，客户端从离线故障恢复时，应将自身模型与邻居模型一样按相对陈旧度进行折扣加权，并以继承滞后而非年龄来度量陈旧度，从而使恢复误差保持有界。 |
| [^22] | [Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks](https://arxiv.org/abs/2610.28543) | 该论文提出二元离散扩散，将扩散过程的含噪状态表示为每个位置候选词元集合的二进制隶属向量，通过添加与删除候选实现信息的渐进丢失与恢复，从而能显式建模部分不确定性并有效解决数独、规划等逻辑与推理任务。 |
| [^23] | [GraphCast-Mamba: Learning Non-Markovian Corrections for Weather Forecasting](https://arxiv.org/abs/2610.28512) | GraphCast-Mamba在冻结的GraphCast主干网络上加入结合图消息传递与Mamba层的可训练修正分支，通过跨预报步骤的时间记忆学习超出两帧输入的非马尔可夫修正，使十天预报整体损失降低14.42%，第10天损失降低21.66%。 |
| [^24] | [Variance Reduction with Delayed Feedback for Non-Convex Optimization](https://arxiv.org/abs/2610.28510) | 该论文针对梯度评估存在延迟的非凸优化问题，提出了基于有序递推动量估计器的方差缩减方法及其异步快照变体，并在延迟上界或缺失反馈加权总量上界的条件下给出了改进的预言机复杂度。 |
| [^25] | [ECG for laboratory values:A comprehensive physiological and technical analysis](https://arxiv.org/abs/2610.28507) | 本论文提出了一种多模态深度学习框架，利用心电图记录和患者元数据预测NT-proBNP等实验室检测值，验证了心电图预测的生理学合理性，并发现12导联配置的性能始终优于单导联配置。 |
| [^26] | [BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models](https://arxiv.org/abs/2610.28500) | 提出BitMTP算法-系统协同设计框架，将多词元预测与超低比特大语言模型骨干相结合，使单一模型兼具自回归生成与块因果草稿生成能力，在保持同等生成质量的前提下实现高效解码加速。 |
| [^27] | [Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs](https://arxiv.org/abs/2610.28498) | 论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。 |
| [^28] | [Model- and Layer-Level Closed-Loop Forecasting with SSM-Based Spiking Neural Networks](https://arxiv.org/abs/2610.28495) | 本文针对脉冲神经网络自回归预测中所需历史值不可用的问题，提出了模型级与层级的闭环预测策略，将网络自身生成的预测值反馈回网络作为输入。 |
| [^29] | [Geometry-Agnostic Data-Driven Discovery of Extreme Scenarios](https://arxiv.org/abs/2610.28487) | 该论文提出一种两阶段数据驱动神经框架，无需预知控制方程即可从观测数据中学习几何无关的动力学约束，并通过物理信息神经搜索发现原始观测中未曾出现的稀有极端场景。 |
| [^30] | [Claims of emergence in AI should be grounded in information decomposition](https://arxiv.org/abs/2610.28479) | 本文主张以信息分解作为统一的形式化框架，将AI系统中关于涌现与协同的非正式论断转化为精确、可比较且可证伪的评估。 |
| [^31] | [LocaLeJEPA: Maximum Entropy Representations under Geometric Priors](https://arxiv.org/abs/2610.28458) | 该论文提出 LocaLeJEPA 框架，通过对 LeJEPA 熵最大化性质进行流形理论重新表述，并引入局部欧几里得分布匹配正则化器 LocaLeReg，使自监督学习能够支持非欧几里得几何先验（如等变CNN和双曲ViT），同时避免表示坍缩。 |
| [^32] | [A Bochner Weitzenb\"ock Decomposition for Cellular Sheaf Laplacians, Curvature and Neural Sheaf Diffusion](https://arxiv.org/abs/2610.28455) | 本文将Bochner-Weitzenböck分解推广到胞腔层拉普拉斯算子，将其分解为Bochner层拉普拉斯算子与Forman曲率两个部分，建立了相应的和乐刻画、Cheeger不等式与离散Bochner定理，并将该理论应用于层神经网络，得到节点级信念得分等新工具。 |
| [^33] | [CNN-based classifier for subject-independent MBTI Extraversion/Introversion preference classification from short-time EEG signals](https://arxiv.org/abs/2610.28442) | 本研究提出紧凑型卷积神经网络SSCNet，仅需4秒的短时脑电信号即可实现与被试无关的MBTI外向/内向偏好中等程度预测，在准确率、稳定性和计算效率上均优于EEGNet和EEGProgress等现有方法。 |
| [^34] | [Do Neural Models Prefer Their Submodels?](https://arxiv.org/abs/2610.28439) | 该论文通过引入几何框架研究神经模型的几何结构，证明除少数模型边界上的例外情况外，损失函数的局部极小值点通常不会落在由不活跃神经元或低秩结构构成的子模型中。 |
| [^35] | [Region-Specific EEG Classification of Hypnotic Susceptibility Using a Sliding-Window 2D CNN: A Subject-Independent Evaluation](https://arxiv.org/abs/2610.28427) | 该论文用滑动窗口2D CNN对脑电进行脑区特异性的催眠易感性分类，发现虽然随机划分筛查中枕叶区准确率最高（0.721），但在严格的留一被试交叉验证下表现骤降至接近随机水平，凸显了被试独立评估对脑电分类研究的关键意义。 |
| [^36] | [DISTRACT: Degradation-Indexed Study of Textual Reference Across Competing Targets](https://arxiv.org/abs/2610.28419) | 提出了DISTRACT匹配干预基准，通过固定视频、目标和干扰物并交叉文本先验与图像退化的方式，系统检验视觉-语言跟踪器是否真正利用文本内容做出跟踪决策，发现仅Molmo2和JointNLT表现出有方向性的文本利用，而其余模型对文本语义变化的响应微弱。 |
| [^37] | [Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI](https://arxiv.org/abs/2610.28411) | Brain2Vec是一个在74,425个T1加权脑部MRI扫描上预训练的自监督3D基础模型，既可作为下游任务微调的权重初始化骨干，也可作为独立特征提取器，在脑年龄预测等任务中展现出更高的精度和更快的收敛速度。 |
| [^38] | [Rewards for Budget-Aware Multi-Turn Agents That Adapt to Arbitrary Test-Time Constraints](https://arxiv.org/abs/2610.28409) | 本文提出并比较了十一种奖励目标函数，使大语言模型多轮智能体能够学会自主分配预算，并适应token、API调用、环境步骤等多种资源类型上动态指定的测试时约束。 |
| [^39] | [Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling](https://arxiv.org/abs/2610.28388) | 提出Self-Credit方法，通过让策略模型自生成带引用的摘要信用，并利用组内对成功与失败rollout的比较进行重缩放，从稀疏结果奖励中为长程智能体强化学习构建密集的信用分配信号。 |
| [^40] | [Don't Skip the Baseline: A Controlled Comparison of Dynamic Expert Skipping in MoEs](https://arxiv.org/abs/2610.28369) | 本文首次将四种动态专家跳过方法纳入统一分类体系，并在匹配专家计算预算下与“静态减少专家数量”这一简单基线在两个现代MoE模型上进行了受控公平比较。 |
| [^41] | [Relative Representations Enable Portable White-Box Monitoring Across Language Models](https://arxiv.org/abs/2610.28366) | 该论文发现不同语言模型的内部表征可通过相对角度距离映射到兼容空间，使在一个模型上训练的行为探针无需目标模型标注数据即可直接复用，从而实现可移植的白盒监控。 |
| [^42] | [Confessions of a Masked MDP](https://arxiv.org/abs/2610.28360) | 本文首次研究了动作掩码未知的线性MDP中策略优化与掩码学习相结合的问题，提出了一个基于确定性乐观优化的最小最大最优且计算高效的算法，并进一步提出了一个能主动规划探索策略、在非退化实例上表现更优的算法。 |
| [^43] | [CountVideoBench: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models](https://arxiv.org/abs/2610.28356) | 该论文提出CountVideoBench基准，利用2,600个覆盖[5,50]区间内26个计数值的合成视频，解决了现有视频问答基准计数分布严重偏向小数值、导致大数值上的模型预测偏差无法被评估的问题，从而系统性地评估视频语言模型在物体计数任务中的预测偏差。 |
| [^44] | [Beyond the Point Mass: Acceptance-Derived Weights and Distributional Losses for Speculative Decoding](https://arxiv.org/abs/2610.28355) | 本文提出 D-PAL（动态逐位置接受损失），从精确期望接受长度出发，用草稿与目标分布的重叠度（1减总变差）取代点质量假设下的交叉熵信号，以基于重叠度的位置权重配合对数接受率作为逐位置损失来训练推测解码的草稿模型，且在目标为点质量时该目标函数退化为 D-PACE。 |
| [^45] | [Characterizing Cross-Relay Generalization Failure in Machine Learning-Based Power System Protection](https://arxiv.org/abs/2610.28348) | 该研究揭示了基于机器学习的故障检测器在跨继电器部署时存在方向性泛化失效——虽然平均F1分数几乎不变，但误报率从0.088激增至0.341，说明仅依靠平均指标评估会掩盖部署中严重的可靠性劣化问题。 |
| [^46] | [Identifying Dependent Components from Multi-domain Linear Mixtures](https://arxiv.org/abs/2610.28330) | 该论文提出了一种多域线性混合模型，利用各域之间域特定的成分缩放差异以及潜在成分分布在域间一致这一特点，实现了对相关（非独立）潜在成分的可识别与恢复，突破了传统独立成分分析要求成分独立或不相关的限制。 |
| [^47] | [Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations](https://arxiv.org/abs/2610.28327) | 提出观测同化Transformer（OAT），一种几何感知的神经过程模型，利用稀疏注意力机制同化稀疏站点观测，将数值天气预报先验场修正为高分辨率的概率性后验场，可扩展至数千站点并支持任意连续位置的预测。 |
| [^48] | [Manifold-Informed Neural Networks for Low-Label ECG Electrolyte Inference](https://arxiv.org/abs/2610.28322) | 提出流形信息神经网络（MINN），用学习潜在空间k近邻图上带可学习任务感知源项的离散泊松方程流形算子残差，替代PINN中的物理算子残差，从而在缺乏已知物理算子的低标注场景下实现ECG电解质推断。 |
| [^49] | [Stabilizing Filters for Graph Signal Generation](https://arxiv.org/abs/2610.28318) | 该论文推导了连续时间图生成模型在图扰动下的Wasserstein稳定性界，并据此提出了一个原则性的稳定图滤波器设计框架，使图信号生成过程对图结构误差具有鲁棒性。 |
| [^50] | [ABACUS: A Benchmark for Image Editing Models to Assess Object Category and Counting Under Scene Conservation](https://arxiv.org/abs/2610.28316) | ABACUS是一个针对图像编辑模型数值能力的新基准，采用无VLM评判的评估协议和专用评分指标，首次准确测量了模型在保持场景的同时添加、移除或更改对象类别与数量的真实能力。 |
| [^51] | [Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory](https://arxiv.org/abs/2610.28315) | 该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。 |
| [^52] | [Debiased Prior-Data Fitted Networks for Amortized Bayesian Semi-Parametric Inference](https://arxiv.org/abs/2610.28314) | 该论文为去偏先验数据拟合网络建立了理论基础，揭示了理想PFN的后验集中速率会隐式依赖干扰函数并随先验多样化而变慢，并基于有效影响函数提出标签校正与先验校正两种去偏策略以加速目标后验的集中速率。 |
| [^53] | [Recommending Bacterial Cultivation Conditions from 16S rRNA](https://arxiv.org/abs/2610.28310) | 提出包含65,023个菌株的DSNI基准数据集，并设计了一种利用16S rRNA高变与保守区域结构特征的可变性门控编码器，仅以不到1%的参数开销将细菌培养条件推荐的联合准确率提升4.7%。 |
| [^54] | [Learning Normal Coordinates on Riemannian Manifolds with Flow Charts](https://arxiv.org/abs/2610.28302) | 提出训练条件可逆神经网络“流图”来近似黎曼流形上的正规坐标，并设计了非负且仅在坐标图为正规时才为零的高效损失函数，使指数映射和对数映射可通过快速欧氏运算近似，大幅降低计算成本。 |
| [^55] | [Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling](https://arxiv.org/abs/2610.28294) | ARR 是一种无需训练的自适应方法，通过检测解的修订是否达到不动点，在“继续改进现有解”与“重新生成新解”之间动态分配测试时计算，从而在多个模型和基准上以同等计算量达到与多数投票相当的效果。 |
| [^56] | [Local Work, Global Progress: A Convergence Theory of AI Agent Collaboration](https://arxiv.org/abs/2610.28286) | 该论文首次为LLM多智能体协作建立了收敛性理论：将协作求解形式化为块优化，每个智能体基于局部目标加修正项改进共享解的一部分，由协调者检验并合并提案，证明了可达到误差下限的线性收敛性，并在机器人路径规划任务中验证了多智能体团队总能找到无碰撞方案且优于单一智能体。 |
| [^57] | [Geometry-Aware Block Selection for Sparse Variational Gaussian Processes](https://arxiv.org/abs/2610.28256) | 本文提出一种几何感知的块选择方法，通过在固定诱导点预算下选取能收紧变分下界的块对角缩放矩阵中的块，从而提升稀疏变分高斯过程的测试RMSE和NLL性能。 |
| [^58] | [Convergent Performance, Divergent Mechanisms in LoRA-Adapted Vision Foundation Models for Brain MRI](https://arxiv.org/abs/2610.28255) | 研究发现，不同预训练目标的视觉基础模型经LoRA适配后，在脑部MRI分割任务上虽取得几乎相同的性能，但其内部表征变化机制截然不同，且跨任务迁移能力存在显著差异。 |
| [^59] | [TPO: Tri-level Distributionally Robust Learning for OOD Direct Preference Optimization](https://arxiv.org/abs/2610.28229) | 提出了TPO框架，通过三层分布鲁棒学习同时建模组级、提示级和偏好级的分布不确定性，使DPO在分布外偏好漂移场景下更加鲁棒。 |
| [^60] | [ExeBench-FHD: Addressing Fidelity Issues in Decompiler Evaluation](https://arxiv.org/abs/2610.28212) | 该论文分析了广泛使用的反编译器评测基准ExeBench中影响评估保真度的九个问题，并提出了向后兼容的修订版ExeBench-FHD，通过新增16个字段更忠实地评估基于LLM的反编译器。 |
| [^61] | [JAXbounce: A Continuous-Control Pong-Like Benchmark for Active Inference Beyond Low Dimensional Environments](https://arxiv.org/abs/2610.28203) | 提出了JAXbounce——一个完全可微分、可并行化的连续控制类Pong基准测试，通过状态向量与像素观测、多种噪声条件及超过万次的智能体模拟，将主动推理的评估从低维离散环境拓展到更复杂的高维连续环境。 |
| [^62] | [Parameter-Efficient Adaptation of Time Series Foundation Models for Electrical Load Forecasting](https://arxiv.org/abs/2610.28200) | 本文系统比较了预训练推理、全量微调与LoRA参数高效适配在电力负荷预测中的精度与效率，发现预训练时间序列基础模型已优于传统基线，并评估了紧凑适配方法能否在不同负荷聚合层级、分布内外设置及不同模型家族上带来一致的增益。 |
| [^63] | [Reframing Object-Centric Reinforcement Learning: From Absolute to Relative Representations](https://arxiv.org/abs/2610.28197) | 该论文提出将强化学习中的对象表示从绝对全局坐标重构为固定于锚点的相对坐标（ORCA），并设计了ORBiT架构来处理这种相对输入，从而提升MLP和Transformer智能体的性能及其对场景平移的鲁棒性。 |
| [^64] | [Attractive Faces Skew Text-to-Image Reward Models and Leaderboards](https://arxiv.org/abs/2610.28195) | 研究发现人脸吸引力在人类偏好判断中会压倒图像质量本身，这一偏差会扭曲文生图模型的奖励模型和排行榜，使AI生成的有吸引力面孔被排在真实但不吸引人的面孔之上。 |
| [^65] | [Adaptive Pullback: Reweighting Supervision at Penalty Jumps in On-Policy Distillation](https://arxiv.org/abs/2610.28194) | 该论文发现在线策略蒸馏中存在“惩罚-置信度张力”现象——惩罚跳变处惩罚上升而教师置信度下降，并据此提出AP-OPD方法，在每个检测到的跳变处对监督信号进行自适应重加权。 |
| [^66] | [Offline Makes Online Better: Calibrated Cache Compression for Video Generation](https://arxiv.org/abs/2610.28192) | 该论文提出CALM方法，利用离线校准捕捉不同注意力层和头对标记合并的敏感性差异，以注意力输出误差作为轻量级代理指标，为在线视频生成的KV缓存压缩实现自适应的层-头合并分配，从而以更低开销获得更高质量的视频。 |
| [^67] | [Foreword: Pre-Generation Prediction and Characterisation of LLM Test Failures using the Jacobian Lens](https://arxiv.org/abs/2610.28187) | 提出Foreword框架，利用雅可比透镜将LLM中间激活读取为词汇标记分数来表征输入，并结合通过与失败参考样本及稀疏逻辑回归探针或双高斯混合模型，实现测试失败的生成前预测与失败原因表征。 |
| [^68] | [Coincidence-Maximizing Customer Clustering Using Smart Meter Data](https://arxiv.org/abs/2610.28178) | 该论文证明即使只有两个组，最大化加权峰值同时率的客户分组问题也是强NP难的，并提出了一种交替确定组峰值时刻与精确重新分配客户的多项式时间固定峰值精化算法，可严格改进加权同时率，并在四个公开智能电表数据集上验证了方法的有效性。 |
| [^69] | [From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming](https://arxiv.org/abs/2610.28160) | 该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。 |
| [^70] | [Predicting silent failures of learned inference under restricted access: what measurement physics and prior information explain](https://arxiv.org/abs/2610.28155) | 该论文提出“盲区集合”概念用于刻画学习推理中不依赖真值诊断无法察觉的静默失效，证明线性高斯模型下静默失效风险可精确分解为非检测与非覆盖，并通过线性化VAE先验构建评分，在真实X射线CT数据的前瞻性研究中成功预测了学习推理系统的实际盲性（Spearman ρ=0.79和0.84）。 |
| [^71] | [Curriculum Learning for Generative Flexible Docking](https://arxiv.org/abs/2610.28145) | 该论文提出将课程学习应用于生成式柔性对接任务，在训练中逐步引入大构象变化样本，以应对从apo到holo结构的长尾分布偏移这一核心挑战，从而显著提升现有生成式对接方法的性能。 |
| [^72] | [Fixed Sets and the Hardness of Computing Correlated Equilibria](https://arxiv.org/abs/2610.28141) | 本文证明计算多面体博弈的近似相关均衡是PPAD完全的（与纳什均衡同等难度），挑战了相关均衡比纳什均衡更容易求解的传统观点，并通过提出“不动集”这一不动点的自然推广建立了相关均衡与不动点理论的概念性联系。 |
| [^73] | [Near-Optimal Regret Bounds for UCB-Style Gaussian Process Bandits via Data Thinning](https://arxiv.org/abs/2610.28136) | 本文提出在高斯噪声假设下，通过数据拆分技术使UCB式高斯过程赌博机算法获得更紧的置信区间，从而实现近最优的简单遗憾和累积遗憾上界。 |
| [^74] | [Causal Discovery for Linear Models with Latent Confounding via Multi-domain Identifiable Representation Learning](https://arxiv.org/abs/2610.28134) | 该论文提出LinADMG框架，利用多域可辨识表示学习来辨识噪声变量，从而在不依赖非高斯性或无弓性假设的情况下，实现对含潜在混杂的线性因果模型ADMG的辨识与因果发现。 |
| [^75] | [Weak-$L^\alpha$ Noise, Strong Guarantees: Gradient Clipping under Affine Heavy Tails](https://arxiv.org/abs/2610.28107) | 本文在严格弱于矩有界假设的弱-$L^\alpha$ 重尾条件下，为仿射重尾噪声驱动光滑凸优化中的裁剪SGD及其加速变体建立了高概率收敛保证，并达到与矩条件情形相同的随机收敛率。 |
| [^76] | [From Complex Histories to Simple Predictors: Predictive Dimension in Dynamic Graph Learning](https://arxiv.org/abs/2610.28090) | 该论文提出连续时间动态图学习无需重建完整交互历史，而应识别足以预测未来的低维状态，并定义了可独立估计的“内在预测维度”来量化任务所需的有效预测自由度。 |
| [^77] | [PULSE: Personalized Opportunistic Learning for Wearable Devices](https://arxiv.org/abs/2610.28081) | PULSE提出了一种事件驱动的个性化机会学习框架，将可穿戴设备间的每次相遇视为在线协作决策，仅在梯度与本地学习方向兼容时融合对等节点信息，从而在无稳定连接的条件下解决部署后数据分布差异带来的个性化学习难题。 |
| [^78] | [MuPACE: Learning to Pace Muon for LoRA Fine-Tuning](https://arxiv.org/abs/2610.28072) | MuPACE通过在大量微调任务上元训练一个轻量循环策略，自适应地为Muon优化器调控LoRA微调的全局学习率步长，在无需逐任务调参的情况下于大多数问题上超越固定学习率调度。 |
| [^79] | [Quantifying the Threat: A Bayesian Probabilistic Framework for Heavy Metal Health Risks among Children in the Densu Basin, Ghana](https://arxiv.org/abs/2610.28064) | 该研究提出贝叶斯概率框架评估加纳登苏流域地下水重金属对儿童的非致癌健康风险，发现约31%的采样点危害指数超过1的概率大于0.5，其中铅、砷和镉是主要风险贡献者。 |
| [^80] | [Auditing Where Concept Bottleneck Accuracy Comes From on CUB](https://arxiv.org/abs/2610.28059) | 该研究审计了概念瓶颈模型和概念嵌入模型在CUB基准上准确率的真实来源，发现CUB标注噪声过大、且模型预测依赖内部信息而非暴露的概念分数，从而质疑了现有基于准确率的CBM可解释性评估方式的有效性。 |
| [^81] | [Universal Online Convex Optimization With Delayed Feedback](https://arxiv.org/abs/2610.28058) | 本文提出了一种无需任何曲率先验知识即可同时自适应凸、强凸和指数凹损失的延迟反馈在线凸优化通用算法，其核心是一个带二阶超额损失保证的延迟元学习器，遗憾界在时间范围与延迟依赖上达到已知最优水平。 |
| [^82] | [On-Policy Transition Distillation](https://arxiv.org/abs/2610.28048) | 该论文发现教师与学生在中间层预测过渡上的一致性与答案正确率正相关，并提出在线策略过渡蒸馏（OPTD），通过在学生生成的轨迹上对齐教师与学生的逐层对数概率增量来增强传统最终层分布匹配，从而提升蒸馏效果。 |
| [^83] | [Poisoning-Based Backdoors Are Not Cryptographic](https://arxiv.org/abs/2610.28045) | 本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。 |
| [^84] | [When Does Training on Synthetic Data Provably Work?](https://arxiv.org/abs/2610.28036) | 该论文从理论上刻画了合成数据训练何时可证明有效：在密度比受控时给出有限样本超额误差界，但若真实分布在合成分布支撑之外且假设类足够有表达能力，则仅凭合成风险无法保证真实世界的误差。 |
| [^85] | [From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks](https://arxiv.org/abs/2610.28027) | 该研究通过追踪七个开源强化学习后训练框架的代码提交历史，发现1598篇提出通用RL训练方法的论文中仅有37篇被框架收录，且收录的论文多为大规模训练（30B+参数）和现有损失函数的小幅修改，揭示了RL研究成果从论文到实际工程应用之间的巨大鸿沟。 |
| [^86] | [RANGER: Retrieval Agent for Navigating Graphs with Semantic Evaluation Rewards](https://arxiv.org/abs/2610.28010) | 该论文提出RANGER混合框架，让大型语言模型在训练中充当导师，为强化学习智能体提供密集的语义奖励信号，使其学会在知识图谱上导航并完成多跳问答任务。 |
| [^87] | [Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation](https://arxiv.org/abs/2610.27993) | 该论文提出RSOD端到端框架，通过领域合成数据构建、多个专家LoRA教师后训练以及内存感知的在线策略蒸馏，将六个网络专家模型的行为整合到一个可部署的单一电信语言模型中。 |
| [^88] | [When Momentum Helps Polyak--Ruppert Averaging](https://arxiv.org/abs/2610.27990) | 该论文证明动量方法能加速有限次运行中 Polyak--Ruppert 平均向渐近协方差的收敛——SGD 需要 $m\gg\kappa^2$ 轮而动量只需 $m\gg\kappa$，从而使样本量 $N$ 下的临界批量大小从 $\Theta(\sqrt{N}/\kappa)$ 提升到 $\Theta(\sqrt{N/\kappa})$。 |
| [^89] | [On Johnson-Lindenstrauss Transforms for Byzantine-Robust Optimization](https://arxiv.org/abs/2610.27989) | 本文利用Johnson-Lindenstrauss变换能高概率近似保持内积和距离的几何特性，将其应用于拜占庭鲁棒优化，根据工作节点消息与可信参考梯度的对齐程度进行加权，并通过理论分析和实验验证了该方法的有效性。 |
| [^90] | [Annealed correlation and quenched noise sensitivity in biased deep neural networks](https://arxiv.org/abs/2610.27983) | 本文证明在给深度神经网络加入独立高斯偏置项后，淬火噪声敏感性在广泛的深度范围内依然成立，且只需扰动输入与原输入相等的概率趋于零这一更弱的假设。 |
| [^91] | [Neural Closed-Loop Identification of Nonlinear Systems with Stability and Generalization Guarantees](https://arxiv.org/abs/2610.27970) | 该论文将对偶Youla参数化推广到非线性系统，提出内部控制器辨识框架，能够从闭环数据中辨识可能不稳定的非线性系统并保证闭环稳定性。 |
| [^92] | [Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay](https://arxiv.org/abs/2610.27954) | 该论文提出ReTrace评估框架，通过形式化“转换一致的恢复”标准并借助面向目标的动作回放，系统性地评估计算机使用智能体从失败中恢复的能力，填补了基于模型的恢复研究的空白。 |
| [^93] | [Unsupervised Head Network Personalization for Split Computing via Federated Learning](https://arxiv.org/abs/2610.27952) | 该论文提出了一种结合联邦学习与知识蒸馏的数据高效方法，在仅需少量无标签、非独立同分布数据且保护用户隐私的前提下，实现分割计算中头部网络的无监督个性化训练。 |
| [^94] | [Near-Duplicate Image Selection Transfers Preferences in Vision-Language Models](https://arxiv.org/abs/2610.27943) | 该论文发现视觉-语言模型仅通过在近重复图像变体中进行选择（不改变原始照片和人类标题），就能将自身隐含的偏好隐性传递给由其筛选数据训练出的其他学生模型。 |
| [^95] | [The Shape of the DPO Hyperparameter Landscape: What to Measure, How It Moves with Scale, and When Not to Transfer It](https://arxiv.org/abs/2610.27895) | 通过密集超参数网格实验系统描绘了DPO超参数空间随模型规模和训练步数的演变规律，并识别了最适合超参数选择的基准与优化策略。 |
| [^96] | [Physical-Variable-Informed Deep Learning for Robust Diagnosis in Industrial Fluid Transport Systems](https://arxiv.org/abs/2610.27891) | 提出物理变量信息驱动的深度学习框架PhyNet，通过物理变量与响应信号之间的交叉注意力机制，在无需严格时间对齐的情况下，实现了工业流体输送系统在真实采样异构条件下的鲁棒故障诊断。 |
| [^97] | [Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization](https://arxiv.org/abs/2610.27889) | 本文系统性地分析了智能体框架优化带来的性能提升究竟源于“激发”（在已有能力范围内更可靠地完成任务）还是“扩展”（扩大可解任务的范围），并指出这一区分对递归自我改进和进化搜索至关重要。 |
| [^98] | [Enumeration-Guided Training of Neural Hamilton-Jacobi Safety Certificates](https://arxiv.org/abs/2610.27870) | 提出一种验证引导的训练框架，通过安全约束的概率枚举生成紧凑且精准的反例集合来微调预训练的Hamilton-Jacobi值函数，从而获得更大的认证安全区域、更快的认证速度，并可在部署时通过屏蔽机制强制保证安全。 |
| [^99] | [From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code](https://arxiv.org/abs/2610.27863) | 提出了一种基于冻结代码语言模型上双低秩适配器逐词元对数似然比的框架，可在词元证据基础上对 AI 参与代码进行行、块、文件多粒度的检测与定位。 |
| [^100] | [LaLa-World: Scaling Latent World Models with Natural Language Actions](https://arxiv.org/abs/2610.27851) | LaLa-World通过用语言统一描述动作与视觉变化、并将未来预测器构建于语言模型之上，使单个潜在世界模型能够跨仿真、机器人、图像和视频等异构数据进行联合训练，在MPC规划性能上超越了针对单一领域或任务训练的现有模型。 |
| [^101] | [SAPD: Learning from Agent Experience without Environment Interaction](https://arxiv.org/abs/2610.27847) | 该论文提出步级优势策略蒸馏（SAPD）方法，通过将离线多轮轨迹分解为单步决策上下文进行在线策略蒸馏，并利用相对步级优势信号，使LLM智能体无需与环境交互即可从离线经验中有效学习，同时避免了SFT的灾难性遗忘和在线强化学习对实时环境交互的依赖。 |
| [^102] | [From Generic to Dedicated: A Novel Optimizer for Online Continual Learning](https://arxiv.org/abs/2610.27835) | 该论文提出新型优化器SPIN，通过在快速“可塑性”序列与自适应“稳定性”序列之间进行插值来显式解耦在线持续学习中的可塑性与稳定性，有效缓解灾难性遗忘。 |
| [^103] | [ProVe: Proposer-Verifier Decoding](https://arxiv.org/abs/2610.27815) | ProVe反转了推测解码中提议者与验证者的角色，让大模型通过单次前向传播直接提议token块并加以验证，无需训练独立草稿头即可实现2.5-2.7倍推理加速，且无损版本能完全复现原始贪婪解码结果。 |
| [^104] | [SliMOO: Interpretable Multi-Objective Evolutionary Search for LLM Depth Pruning](https://arxiv.org/abs/2610.27812) | SliMOO将LLM深度剪枝表述为多目标优化问题，通过单层移除探测构建可解释的任务-层重要性图谱，并结合多目标进化搜索，在压缩模型的同时保留跨任务的广泛能力。 |
| [^105] | [Certified Adaptive List Decoding for Error-Correcting Codes](https://arxiv.org/abs/2610.27805) | 该论文将译码建模为吸收态离散扩散模型上的概率束搜索，并结合线性规划对偶界与基于距离的检验逐帧认证全局最大似然解，在15种码上可认证高达99.9%的帧且无需在精度与计算量之间权衡。 |
| [^106] | [TerraNostra-Bench: Benchmarking Spatial Downscaling and Temporal Rollout for Global 1-km Land-Use Reconstruction](https://arxiv.org/abs/2610.27793) | 该论文提出了TerraNostra-Bench，这是一个变化感知的评估基准，将空间降尺度与自回归时间滚动预测解耦，并聚焦于稀少的土地利用变化像素进行评分，从而推动数据驱动方法生成全球1公里分辨率的年度土地利用地图。 |
| [^107] | [Foundations of the Scaling Law](https://arxiv.org/abs/2610.27789) | 本论文通过几何近似、谱分析和约束优化三个相互关联的分析框架，为缩放定律建立了理论基础，并区分解释了涌现、顿悟和测试时缩放等现象。 |
| [^108] | [How Much Do $\ell_1$ Attacks Need Concentrated Perturbations?](https://arxiv.org/abs/2610.27773) | 研究发现即使对每个坐标施加扰动上限以抑制集中扰动，ℓ₁对抗攻击仍能保留约97%的攻击成功率，表明其成功并不依赖高度集中的大幅扰动。 |
| [^109] | [Engram Memory Network: Brain-Inspired Prototype Explanations](https://arxiv.org/abs/2610.27770) | 该论文受人类记忆系统启发，提出了模拟大脑腹侧视觉流和下颞叶皮层的印记记忆网络（EMN），其核心是一个受Hopfield启发的记忆模块，通过迭代检索代表性原型，在保持分类准确率的同时提升了可解释AI在不同数据集上的解释质量。 |
| [^110] | [Foundation-Style Pretraining for Scalable and Transferable Distributed PV Forecasting](https://arxiv.org/abs/2610.27763) | 本文提出一种面向光伏领域的基础模型式预训练框架，通过将NWP辅助预测统一建模为序列补全问题并设计多种预训练目标，从多站点、多时域异构数据中学习共享预测模型，实现分布式光伏预测的可扩展性与对未见站点的迁移能力。 |
| [^111] | [AC/DC on a Budget - Alternating Sparse Phases](https://arxiv.org/abs/2610.27758) | 本文提出ARC方法，通过在密集阶段零初始化遮蔽参数、缩短密集阶段并用仅更新掩码和高梯度权重的稀疏重连阶段完全替代密集阶段，以极小的计算成本实现了与AC/DC稀疏化基线相当的性能。 |
| [^112] | [Give the Model a Finger: Developmentally Aligned Evaluation of Counting in Vision-Language Models](https://arxiv.org/abs/2610.27756) | 该论文受儿童计数发展过程启发，提出一种无需训练的智能体评估框架，让视觉语言模型通过“指认标记”等认知动作进行计数，将模型通常隐藏的串行视觉推理过程转化为可解释、可审计的行为分析。 |
| [^113] | [Detection of Memorization in Diffusion Models Through Thermodynamic Singularity](https://arxiv.org/abs/2610.27754) | 提出一种基于热力学奇异性的判据，能够可靠区分扩散模型是真正学习了数据分布还是发生了对单个训练样本的记忆化。 |
| [^114] | [Constant Step Sizes for Muon-Type Spectral Optimizers on Strongly Convex Quadratics](https://arxiv.org/abs/2610.27747) | 本文通过增益和长度两个关键量，将Muon类谱优化器在强凸二次问题上的行为划分为三种机制，揭示了增益有界时按最大曲率设定步长可收敛、增益无界但长度消失时按总曲率设定步长可收敛、而长度非消失时几乎不收敛的收敛规律。 |
| [^115] | [EEG-Based Classification of Hypnotic Susceptibility Using Two-Dimensional Convolutional Neural Networks](https://arxiv.org/abs/2610.27738) | 本研究利用二维卷积神经网络直接分析最小预处理的脑电信号，结合滑动窗口框架与留一被试交叉验证，实现了无需人工特征提取且与被试无关的催眠易感性客观分类。 |
| [^116] | [Anytime-Valid Runtime Monitoring of Learned Barrier Certificates](https://arxiv.org/abs/2610.27726) | 该论文提出一种任意时刻有效的运行时监控框架，通过将屏障漂移条件转化为基于非负检验上鞅的按回合序贯假设检验，利用观测到的闭环轨迹在线审计学习型屏障证书的统计有效性，并在任意停止时间下实现有限样本误报控制。 |
| [^117] | [When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines](https://arxiv.org/abs/2610.27714) | 本研究通过在审计的HighwayEnv测试平台上控制预热池抽样与提示令牌两大混淆因素，发现经验记忆为闭环LLM驾驶智能体带来的表面收益主要来自预热经验池的随机抽样，而非记忆机制本身。 |
| [^118] | [Privacy Accounting for Empirical Denoisers](https://arxiv.org/abs/2610.27710) | 本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。 |
| [^119] | [Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification](https://arxiv.org/abs/2610.27703) | 本文提出多目标进化半监督学习方法MOEA-SSL，在智慧城市表格分类任务中联合优化预测性能与特征使用量，在仅1%-10%标注数据的条件下取得了与现有轻量级方法相当的分类效果，并显著降低了特征使用量。 |
| [^120] | [Understanding Data Augmentation under Label Noise](https://arxiv.org/abs/2610.27701) | 本文从理论上刻画了标签噪声下数据增强有益或有害的条件，并揭示了数据增强实质上是不确定性放大器、会降低模型稳定性与泛化性能这一反直觉现象。 |
| [^121] | [Influence Is the Target-Response Jacobian Times Residuals](https://arxiv.org/abs/2610.27689) | 论文证明了样本权重影响力与目标响应雅可比矩阵之间仅差一个逐样本乘子的精确恒等关系，且对于典型连接损失该乘子正是训练残差，从而将训练数据归因与模型对训练目标值的敏感性统一起来。 |
| [^122] | [CyberHull: Deterministic Malicious URL Isolation via Convex Hull](https://arxiv.org/abs/2610.27675) | 提出CyberHull框架，利用QuickHull算法在多个正交特征域上构建凸包边界，以资源高效、确定性的方式实现恶意URL的识别与隔离，训练复杂度仅为O(n log n)。 |
| [^123] | [Hybrid Momentum Methods for Optimizing Deep Learning Models](https://arxiv.org/abs/2610.27674) | 提出混合动量方法HyM，将线性加权的长期历史梯度与几何折扣的快动量相结合，并据此构建经Newton–Schulz正交化的HyMUON优化器，在保留长期梯度信息与适应近期变化之间实现平衡。 |
| [^124] | [Neural Operator Splitting for Fluid Dynamics in Complex Geometry](https://arxiv.org/abs/2610.27671) | 提出神经算子分裂框架，将流体动力学的每个时间步分解为物理可解释的分数步并分别学习，通过条件线性傅里叶神经算子在复杂几何下保持数值格式的物理结构，同时利用对称采样和纠错循环缓解自回归误差累积。 |
| [^125] | [Pairwise Learning Increases Data Efficiency in Deep Neural Networks](https://arxiv.org/abs/2610.27669) | 该论文提出成对比较学习框架，将回归问题转化为相对排序任务，无需额外监督即可大幅提升深度神经网络在小样本回归和分类中的数据效率，使轻量级ResNet10在家庭硬件上训练即可超越大规模预训练模型。 |
| [^126] | [Inferring continuous cognitive traits from limited neuroimaging data using pairwise convolutional neural networks](https://arxiv.org/abs/2610.27668) | 提出了一种成对卷积排序框架，通过比较式学习直接从全脑MRI推断连续认知特征，即使在小样本（N=108）条件下也能准确预测空间能力表现，并优于传统回归和逐点深度学习方法。 |
| [^127] | [CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents](https://arxiv.org/abs/2610.27645) | 该论文提出了 CUA-Recovery 基准和 ReRail 框架，通过在持久化桌面环境中将任务组合成长时程工作流来系统评估计算机使用智能体的错误检测与恢复能力，弥补了现有基准依赖人工注入错误、无法反映真实智能体策略所致失败的不足。 |
| [^128] | [BCT: Bayes-Consistent Tempering for Flow Language Models](https://arxiv.org/abs/2610.27641) | 提出贝叶斯一致温度调节（BCT），将温度显式地融入流语言模型生成过程的分布与动力学中，解决了事后温度缩放导致的分布与动力学不匹配问题，使温度调节对流语言模型性能的提升更加可靠。 |
| [^129] | [Ensemble-Aware Structural Encoding (EASE): Conditioning Protein Encoders on Peptide Conformational Ensembles](https://arxiv.org/abs/2610.27638) | 提出了集合感知结构编码框架EASE，通过选择代表肽主要结构状态的中心构象体并以置换不变算子融合其表示，将构象集合信息融入蛋白质编码器，从而更好地捕捉与抗菌活性、毒性、稳定性等肽性质相关的构象变化。 |
| [^130] | [From Outcomes to Steps: Reliable Success-Aware Critic Learning for Long-Horizon Language Agents](https://arxiv.org/abs/2610.27623) | SCALE框架通过让评论家在每个交互前缀同时预测回报值和任务成功概率（成功价值），解决了长程语言代理在稀疏反馈和轨迹级监督下中间动作信用分配的难题。 |
| [^131] | [A Radiation-Hardened, Fault-Aware Quantized Neural Network That Self-Heals](https://arxiv.org/abs/2610.27621) | 本文提出一种抗辐射的故障感知量化神经网络，通过扩展故障感知训练范式，使模型能在商用现货硬件上于运行时自修复位翻转导致的权重损坏，且无需存储额外副本。 |
| [^132] | [Learning Inter-Graph Interactions Between Heterogeneous Substructures of Chemical Systems](https://arxiv.org/abs/2610.27614) | 该论文提出了一种描述复杂化学体系的数据描述符和图神经网络方法，用于学习有机与无机化合物之间的异质图间交互，并在混合太阳能电池材料的物理性质预测上取得了最先进的准确率。 |
| [^133] | [Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims](https://arxiv.org/abs/2610.27610) | 该论文首次通过对114项递归自我改进（RSI）研究的系统性全文编码，量化审计了RSI声明背后产品评估与流程评估证据的实际报告频率，揭示了递归自我改进声明与其实际评估证据之间的差距。 |
| [^134] | [Ask to Learn: Query-Based Collaborative Reinforcement Learning](https://arxiv.org/abs/2610.27606) | 该论文提出了基于查询的协作强化学习新框架，使仅有离线数据的学习者无需直接与环境交互，即可通过自适应地向持有互补信息的协作者查询状态-时间对来突破离线数据覆盖范围的限制，并针对有限视界表格型问题给出了所需查询次数的实例依赖信息论下界。 |
| [^135] | [Sensitivity-Orthogonal Residual Learning for Adaptive Sideslip and Cornering Stiffness Estimation](https://arxiv.org/abs/2610.27586) | 该论文提出灵敏度正交残差学习方法，通过零空间参数化构建与刚度灵敏度正交的模型修正器，解决了学习修正器吸收滤波新息从而抑制侧偏刚度在线自适应的问题，将侧偏角估计误差降低12.7%至22.0%。 |
| [^136] | [Bridging Digital Deception and Physical Fidelity: A Spatio-Temporal Attack on Cognitive Radars](https://arxiv.org/abs/2610.27585) | 提出了一种物理先验驱动的分段时空攻击方法（SSTA），通过全局-局部特征解耦分割机制提升对抗样本的跨架构可迁移性，并将物理信道失真建模为可微时空算子以增强攻击在真实电子战环境中的战术可靠性。 |
| [^137] | [Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval](https://arxiv.org/abs/2610.27580) | 该论文揭示了扩散语言模型在第一层注意力层就通过将关系措辞路由到可见实体位置来组装事实访问状态，这一机制与自回归模型的组装方式截然不同且发生更早，为知识编辑与检索提供了新的机制层面依据。 |
| [^138] | [From POMDP Theory to Deep RL with Particle Filters](https://arxiv.org/abs/2610.27575) | 本文提出了一种结合序贯蒙特卡洛粒子滤波、在线潜在动力学模型学习和近似在线规划的实用端到端深度强化学习算法，并首次为部分可观测环境下的深度强化学习提供了将模型估计、粒子滤波和蒙特卡洛等各项近似误差显式分解的端到端遗憾界理论保证。 |
| [^139] | [When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects](https://arxiv.org/abs/2610.27573) | 该论文提出了一个测量“行为脆弱性”的行为计算框架，通过操纵梯度下降生成针对权威、紧迫性和社会认同偏差的操纵性AI刺激，并借助多利益相关者影响评估网络与时间性危害传播模型，量化了操纵性LLM输出对用户信任校准与依从性的显著影响及其跨用户、跨时间的传播效应。 |
| [^140] | [Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning](https://arxiv.org/abs/2610.27570) | 该论文提出在数据生成分布上（而非网络权重上）施加狄利克雷过程先验来构建新型贝叶斯神经网络DP-BNN，并据此开发了DP-DQN算法，通过随机化价值函数实现高效的汤普森采样，从而解决深度强化学习中认知不确定性量化的难题。 |
| [^141] | [Two Is Enough: Binary Classification Priors Suffice for Multi-Class In-Context Tabular Prediction](https://arxiv.org/abs/2610.27569) | 该论文发现表格基础模型在预训练时仅需使用二分类先验任务（类别数上限设为2），即可实现优于传统十分类上限配置的零样本多分类预测性能，并在公开基准和真实信用评分场景中得到验证。 |
| [^142] | [TRCM: Topology-Aware Route-Consistency Modeling for Robust Neural Vehicle Routing](https://arxiv.org/abs/2610.27568) | 提出 TRCM 方法，复用自回归构造流水线中的节点表示显式预测成对同路径概率并作为一致性信号注入节点选择，结合结构保持增强与拓扑熵重加权，从而在相同解码预算下提升神经求解器解决车辆路径问题的性能与鲁棒性。 |
| [^143] | [Learned Readout Tokens for Extracting Large Language Model Confidence](https://arxiv.org/abs/2610.27566) | 提出 ScratchProbe 方法，通过学习一个专用的读取令牌来提取隐藏状态训练置信度探针，在多种数据集和大语言模型上优于基于现有令牌的探针，多令牌扩展还可进一步提升性能。 |
| [^144] | [HypoNet Nankai PS: rapid hypocenter determination for the Nankai Trough using physics-informed neural networks with three-dimensional P- and S-wave velocity models](https://arxiv.org/abs/2610.27561) | 本研究将基于物理信息神经网络的震源定位框架HypoNet Nankai扩展为HypoNet Nankai PS，通过从P波模型构建S波速度模型并改进训练方法以适应更陡峭的S波走时梯度，实现了南海海槽俯冲带同时利用P波和S波到时的快速震源定位。 |
| [^145] | [AnchorRM: Parallel Intra-Batch State Reconstruction with Anchor-Based Residual Memory for Temporal Graph Neural Networks](https://arxiv.org/abs/2610.27558) | 提出AnchorRM机制，将批前最后提交的节点记忆作为固定锚点，通过学习残差来并行重建同批次内各事件的节点状态，从而在不进行顺序循环更新的情况下解决时序图神经网络小批量处理中的状态过时问题。 |
| [^146] | [Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation](https://arxiv.org/abs/2610.27555) | 本文首次提出“原生安全性复用”问题，并证明通过在线策略蒸馏，可以将安全预训练教师模型的鲁棒安全性迁移给标准预训练的学生模型，使其在保留通用能力的同时继承安全鲁棒性。 |
| [^147] | [Lower Bounds for Smooth Nonconvex Zeroth-Order Optimization: The Price of Determinism under Low Curvature Trace](https://arxiv.org/abs/2610.27550) | 本文在低曲率迹（有效秩）条件下证明了随机化零阶非凸优化仅需 O(ET) 次函数调用且在常数因子内最优，而任何确定性方法都无法获得这种依赖有效秩的节省，从而量化了确定性方法的固有代价。 |
| [^148] | [Timing-Aware and Interpretable Self-Attention for Event-Based Wearable Task Transition Detection](https://arxiv.org/abs/2610.27549) | 该论文提出了一个时间感知且可解释的自注意力机制，通过在基于事件的可穿戴行为标记中显式建模时间与持续时间信息，改进了任务转换检测的性能与可解释性。 |
| [^149] | [BEACON: Bridging Multimodal Semantics and Asynchronous Propagation for Cloud System Anomaly Detection](https://arxiv.org/abs/2610.27542) | 提出了BEACON框架，通过基于信息瓶颈机制的动态可解释多模态聚合器弥合多模态数据间的语义鸿沟，并建模系统实例间的异步时空依赖关系，从而提升云系统异常检测的准确性。 |
| [^150] | [What Moved the Alarm Score? Causal Attribution with Exact Computation and Scarce-Incident Estimation](https://arxiv.org/abs/2610.27534) | 该论文将警报分数的变化建模为混合结构因果模型状态上的Shapley博弈，把结构函数与外生规律作为不同的博弈参与者，从而在事件数据稀少的条件下实现精确的因果归因，并以函数/规律分辨的细粒度方式解释各类型变化如何传播并影响警报分数。 |
| [^151] | [Soliton Profiles in 1D: Enhanced Neural-Network Solvers and Constrained Deep Ritz Methods](https://arxiv.org/abs/2610.27500) | 该论文开发并比较了多种增强神经网络求解器（包括高阶优化精炼、Fourier特征PINN以及两种新颖的约束Deep Ritz方法），用于求解非线性薛定谔、广义KdV和Klein-Gordon方程的一维孤子基态轮廓，并以显式sech型精确解作为基准评估各方法的精度。 |
| [^152] | [Self-Evolving Agents through Privileged Retrospective Distillation](https://arxiv.org/abs/2610.27495) | 提出回溯自蒸馏（RSD）多轮框架，让同一模型既当学生又当教师，借助特权回溯上下文将缺乏真值答案或奖励的自我经验转化为密集监督信号，实现模型、数据与监督的共同进化。 |
| [^153] | [MiCLA: Mixture-to-Constituent Latent Alignment for Polyphonic Audio Representation Learning](https://arxiv.org/abs/2610.27491) | MiCLA 提出了一种两阶段自监督预训练框架，通过合成对数梅尔混合和双向潜在对齐来显式建模混合音频与其组成成分的关系，在 AudioSet 下游任务上取得了领先的 mAP 性能。 |
| [^154] | [Orthogonal Boundary Unlearning for Dual-Use Concepts in Large Language Models](https://arxiv.org/abs/2610.27490) | 提出正交边界遗忘（OBU）框架，通过将参数更新严格限制在与能力子空间正交的零空间内，在剔除大语言模型危险双用途能力（如网络战、病原体增强）的同时，保证对良性科学能力造成零附带损害。 |
| [^155] | [When Parameter Gradient Flows Close in Representation Space](https://arxiv.org/abs/2610.27474) | 该论文精确刻画了表示梯度流对所有损失函数发生闭合的条件：在行满秩的双因子矩阵表示中，闭合仅发生在有效标量平衡的最小联合秩或满联合秩的二值带符号谱族处，而所有中间秩均被排除。 |
| [^156] | [Beyond the Prompt: Recovering Hidden Concepts in Text-to-Image Models](https://arxiv.org/abs/2610.27473) | 本文提出EmbGA，一种在词元嵌入空间中运行的遗传算法，能够恢复文本到图像模型生成的图像中存在但提示词未提及的隐藏概念，为审计被植入的触发关联提供了有效工具。 |
| [^157] | [Training-Free Continual Learning for Multimodal Forecasting via Prompt Evolution](https://arxiv.org/abs/2610.27468) | 该论文提出通过结果驱动反思演化的文本引导模块，使冻结的LLM预测器无需重新训练即可实现时间序列持续学习，在严格防止数据泄露的多模态基准上超越时间序列基础模型，且更新成本从GPU天级降至美元级。 |
| [^158] | [Projected Distillation: On Learning to Solve Hard Problems from Close, Correct Targets](https://arxiv.org/abs/2610.27467) | 提出“投影蒸馏”方法，将学生模型失败的尝试最小化地编辑成接近且正确的目标，并利用对比更新使模型优先学习修正版本，从而在困难问题上获得有效学习信号，同时保留已有能力。 |
| [^159] | [Beyond Scalar Leaderboards: Adaptive Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas](https://arxiv.org/abs/2610.27447) | 本文提出了固定置信度的自适应采样算法VB-EGE，能在多目标LLM竞技场评估中以尽可能少的比较高效识别帕累托前沿上的模型，突破了单一标量排行榜的局限。 |
| [^160] | [Decision Sufficiency of Learned Representation Interfaces for Safe Generative Control](https://arxiv.org/abs/2610.27446) | 本文提出“决策充分性”概念，用以评估学习型表征接口是否为安全生成式控制器提供了足够的信息，精确分离了超额损失的三种来源，并刻画了揭示信息冲突所需的状态数量及接口避免冲突所需的最少输出。 |
| [^161] | [LoRASpace: A Pool-Wide Shared Substrate for Static and Dynamic Multi-LoRA Composition](https://arxiv.org/abs/2610.27444) | 该论文发现真实LoRA池中的低秩因子仅需其秩总和的46%-57%即可保留99%的能量，并据此提出池级共享基底LoRASpace，使多LoRA组合同时获得权重合并的速度与逐LoRA前向传播的质量。 |
| [^162] | [Using Set Learning to Improve Set-Valued Classifiers on Long-tailed Data](https://arxiv.org/abs/2610.27443) | 该论文系统研究了不同训练策略与损失函数（包括作用于批次样本集合的集合式损失）对集合值分类器在长尾不平衡数据上性能的影响，提出通过在训练中引入集合学习来提升其宏平均表现。 |
| [^163] | [Learning-Theoretic Foundation for General Coded Computing: The Adversarial Setting](https://arxiv.org/abs/2610.27440) | 本文为基于学习理论的通用编码计算框架提出了对抗鲁棒的扩展，无需显式识别受损工作节点即可设计编码器与解码器，并刻画了恢复误差随工作节点数量和对抗预算变化的规律。 |
| [^164] | [Scaling Graph Reasoning in Language Models with Learnable Graph Positional Encodings](https://arxiv.org/abs/2610.27439) | 该论文提出用GNN与图Transformer计算的可学习图位置编码直接调制语言模型的注意力，并以后融合模块注入结构信息，从而无需序列化图描述、将上下文负担降至O(N)，显著扩展了语言模型可处理的图推理问题规模。 |
| [^165] | [MLIPS: Multi-Layer Indexer Pool Sharing](https://arxiv.org/abs/2610.27438) | MLIPS是一种无需训练的方法，通过让一个生产者层发布超出预算的top-M候选池供后续多个层共享复用，在保留各层独立打分与选择的同时消除重复的候选搜索，在DeepSeek-V4-Flash上实现了1.14–1.40倍加速和14.8–27.4%的GPU能耗降低，且准确率几乎不受影响。 |
| [^166] | [Baseline and Incremental: User, Item, and Request Debiasing in Instagram Feed Ranking](https://arxiv.org/abs/2610.27423) | 该论文提出在Instagram信息流价值模型中同时保留原始预测（基线）并单独加权加入增量项，将观看者习惯、内容热门度和请求上下文三类独立部署的去偏统一为一个参数化算子，四臂消融实验证明基线加增量的双项设计在绝大多数指标上优于任何单项去偏方案。 |
| [^167] | [The Router Already Knows](https://arxiv.org/abs/2610.27421) | 提出NR-SAE稀疏自编码器，直接复用MoE路由器对每个token的稀疏专家选择来构建字典，在解释路由器加权专家输出方差上显著优于参数匹配的对照组。 |
| [^168] | [Learning the Geometry of Materials Space with Trained Neural Tangent Kernels](https://arxiv.org/abs/2610.27412) | 本研究利用训练后参数处的神经切线核，成功提取并分析了预测超导转变温度的神经网络所学到的材料空间几何结构，揭示了少数主导方向，并将化合物组织成可识别的超导家族，为理解神经网络材料表示的物理内涵提供了实用探针。 |
| [^169] | [Steering intonational tunes in text-to-speech synthesis](https://arxiv.org/abs/2610.27403) | 该研究通过交换干预定位了残差向量量化文本到语音模型中英语核心语调的载体——主干网络的隐藏状态而非token堆栈第一层，并证明可用均值差向量引导该状态来实现对语调的控制。 |
| [^170] | [Communication Complexity of Universal Features](https://arxiv.org/abs/2610.27402) | 受机器学习特征概念启发，本文利用新颖的势函数论证和贪婪函数选择方法，构造了一种仅需 $O(d^*\log(1/\varepsilon))$ 比特（由对偶 VC 维决定）的通用数据表示，使得解码器能够在任意指定任务下以不超过 $\varepsilon$ 的误差完成预测。 |
| [^171] | [Fixed-Size Active Statistical Inference](https://arxiv.org/abs/2610.27389) | 该论文提出用固定样本量的条件泊松抽样（CPS）替代样本量随机的泊松抽样（PS），证明 CPS 是一般多变量 Z 估计问题（包括凸 M 估计）中 PS 的直接替代方案，且其渐近方差不劣于 PS，在现实场景中甚至可显著更优。 |
| [^172] | [Training Controllable World Models via Action Influence Maximization](https://arxiv.org/abs/2610.27387) | 论文提出动作影响力最大化算法，通过主动收集具有高动作影响力的训练数据，突破专家演示数据的动作影响力上限，从而解决世界模型忽略指令动作的问题，实现更可控的动作条件视频生成世界模型。 |
| [^173] | [Sample-efficient learning with Dense Associative Memories](https://arxiv.org/abs/2610.27384) | 该论文解析地揭示了稠密联想记忆在训练中存在从单纯记忆样本到发现数据稀疏生成规则的“顿悟”式转变，并提出带阈值学习规则的新型稠密联想记忆，可执行简化的稀疏哈密顿量学习从而指数级降低样本复杂度。 |
| [^174] | [DStill: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding](https://arxiv.org/abs/2610.27379) | 提出DStill两阶段蒸馏框架，通过轨迹一致性蒸馏将块扩散投机解码中草稿器的多步迭代细化压缩为单步草稿生成，在降低推理延迟的同时保持草稿的保真度。 |
| [^175] | [Shaped SAEs: Expanding the Toolkit for Mechanistic Interpretability](https://arxiv.org/abs/2610.27375) | 本文提出通过在学习稀疏自编码器之前对语言模型的稠密激活进行线性“成形”变换（包括基于雅可比空间的降秩投影和与字典联合学习的仿射变换），来增强所学字典特征的语义选择性，从而扩展机制可解释性的方法工具箱。 |
| [^176] | [Trail-Weighted Influence Scores for Approximate Outcome Prediction in Bayesian Networks](https://arxiv.org/abs/2610.27366) | 该论文提出了一种基于迹加权影响分数的证据剪枝方法用于贝叶斯网络近似预测，证明了无碰撞节点网络中迹得分与端点精确相关性相等，并给出了剪枝后验误差的可计算界限，实验表明其误差比随机剪枝低1.4至13倍。 |
| [^177] | [Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection](https://arxiv.org/abs/2610.27364) | 该论文提出了“预测闭合”这一新刻画，证明JEPA的特征选择取决于特征集合层面的整体性质（即特征间的关系），而非单个特征的可预测性，从而化解了不可预测特征（如交通灯）不会被编码进表示的担忧。 |
| [^178] | [LiBRA: Logit Fingerprints for Byzantine-Robust Client Selection in Federated Learning](https://arxiv.org/abs/2610.27357) | LiBRA通过比较客户端模型的logit指纹而非参数向量来实现联邦学习中的拜占庭鲁棒客户端选择，从而克服了传统方法需要访问原始更新、计算成本高以及仅适用于相同架构客户端的限制。 |
| [^179] | [Robust Lightweight IoT Device Fingerprinting Under Network Traffic Anomalies](https://arxiv.org/abs/2610.27351) | 提出了一种基于数据包头和局部敏感哈希的轻量级物联网设备指纹识别方法，在使用短观测窗口并遭受丢包、扰动、注入等多种流量异常时，仍能实现高达99%的识别准确率并保持鲁棒性。 |
| [^180] | [Automated Brick Schema Knowledge Graph Completion Across Buildings](https://arxiv.org/abs/2610.27339) | 本文首次将知识图谱补全技术应用于Brick建筑元数据模型的自动生成，通过比较三种方法自动推断建筑设备、传感器和空间之间的缺失关系，从而大幅减少人工建模的工作量。 |
| [^181] | [EMBER: Predicting TCR--pMHC Binding Without a Docked Complex](https://arxiv.org/abs/2610.27334) | EMBER通过独立折叠TCR和pMHC并将其编码为SE(3)不变的残基图进行比较，无需构建对接复合物即可预测TCR-pMHC结合，从而消除了放置误差的传播，并将计算复杂度从O(NM)降低到O(N+M)。 |
| [^182] | [VampPrior Unleashed](https://arxiv.org/abs/2610.27322) | 该论文通过解耦VampPrior的先验与后验协方差，并用狄利克雷过程取代固定的均匀混合权重，显著提升了模型证据并实现了稳健的非参数聚类。 |
| [^183] | [Physics-Inspired Optimizers beyond the Edge of Stability: Period-Two Orbits and Delayed Instability](https://arxiv.org/abs/2610.27321) | 本研究揭示了物理启发优化器（动力学映射）在稳定性边缘之外会产生对称的周期二轨道，使训练稳定在一个“有效稳定性边缘”上，该轨道的线性稳定性由映射的弹性参数（介于0和1之间）决定，且VRAdam的阻尼机制在超过两倍临界学习率时会失效。 |
| [^184] | [What Not to Erase: Saliency-Constrained Augmentation for Low-Resource Handwritten Character Recognition](https://arxiv.org/abs/2610.27309) | 该论文提出一种显著性引导的数据增强策略，通过对字符图像中视觉信息丰富的区域进行扰动、保留或约束来定义增强变体，使低资源手写字符识别在获得性能提升的同时保持显著性模式的稳定性。 |
| [^185] | [Global Security Blind Spots: Translation as a Safety Bypass in Low-Resource Languages](https://arxiv.org/abs/2610.27304) | 该论文揭示了大型语言模型的英语安全对齐无法可靠迁移至低资源语言——在英语中被成功拒绝的有害请求经翻译后可诱发不安全回答，各模型平均绕过率高达21.9%至44.4%，构成全球性的安全盲区。 |
| [^186] | [TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation](https://arxiv.org/abs/2610.27290) | TEX提出了一种基于执行驱动交叉验证的协作式测试时扩展框架，通过并行代理生成测试和补丁并交叉执行形成K×K反馈矩阵实现客观验证，在SWT-Bench Verified上以86.4%的成绩创下新的最先进水平。 |
| [^187] | [VPFlow: Value-Predictive Planning with Imperfect World Models for Offline Goal Reaching](https://arxiv.org/abs/2610.27280) | 该论文提出VPFlow，其核心创新在于证明离线目标达成中的规划依赖的是价值预测而非准确的状态预测，并利用闭式蒙特卡洛回报直接回归目标达成价值来监督潜在世界模型，从而在无需精确世界模型的情况下实现高效规划。 |
| [^188] | [A latent control model for realistic rodent motion](https://arxiv.org/abs/2610.27279) | 提出了SCAMPER系统，利用具有双重先验的变分架构训练具身虚拟大鼠模型，使其在目标导向行为和随机探索中都能产生自然的运动，弥补了以往具身动物建模的不足。 |
| [^189] | [What Would Removing This Corruption Change? Failure Attribution for Vision-Language-Action Models](https://arxiv.org/abs/2610.27277) | 该论文提出了“失败归因”这一新问题，并引入分解式风险模型，通过潜在干预仅凭当前图像和任务指令即可估计移除视觉损坏的收益，无需成对干净图像或昂贵的反事实策略回放。 |
| [^190] | [Auditing Safety Preference Supervision with Recorded Derivations](https://arxiv.org/abs/2610.27266) | 该论文提出符号化偏好标注方法，通过记录属性分类、明确规则和固定判定排序，使每个安全偏好标签都能从存储的推导过程中重新计算验证，从而实现对偏好监督的可审计性。 |
| [^191] | [Solving the Ho--Nguyen Conjecture for Gaussian Mixtures: Applications to Mixtures of Experts](https://arxiv.org/abs/2610.27265) | 本文解决了Ho--Nguyen猜想，证明对所有聚类大小M≥2的高斯混合，实数非零权重系统的密度抵消阶数恰为2M，为过拟合高斯混合模型和专家混合模型的参数学习速率提供了严格的理论保证。 |
| [^192] | [Guided Diffusion for Trustworthy Synthetic Text Generation](https://arxiv.org/abs/2610.27259) | 该论文提出了“引导式扩散”这一私有合成文本生成新范式，通过利用扩散语言模型的可引导性，以新颖的采样和重新掩码机制引导去噪过程，生成既符合差分隐私保证又兼具高数据效用的高质量合成文本。 |
| [^193] | [TxnQA: Benchmarking Analytical Reasoning and Evidence Grounding over Transaction Event Sequences](https://arxiv.org/abs/2610.27256) | TxnQA提出了一个针对交易事件序列分析推理的新基准，包含27个任务族和欺诈风险推理套件，评估发现LoRA适配虽能有效规范输出格式，但算术能力仍是模型性能的主要瓶颈。 |
| [^194] | [Neighborhood-Aware Unlearning for Generative Recommendation](https://arxiv.org/abs/2610.27232) | 本文揭示了生成式推荐中的机器遗忘会因语义标识符的共享而无意识地损害被遗忘物品邻域内的推荐性能，且该退化被全局指标掩盖，并提出邻域感知遗忘（NAU）方法，在交互级遗忘的同时对被遗忘物品邻域的概率质量进行显式控制。 |
| [^195] | [Deep catastrophic-risk-aware reinforcement learning](https://arxiv.org/abs/2610.27214) | 本文将极值理论方法扩展至深度强化学习，提出了一种基于自动微分的可微尾部风险目标，使神经网络策略能够更准确地优化条件风险价值（CVaR），从而更好地应对罕见极端风险。 |
| [^196] | [A Theoretical Analysis of Environment-Aware Out-of-Distribution Detection for Vision-Language Models](https://arxiv.org/abs/2610.27211) | 该论文基于衡量环境间图像与文本特征对齐程度的“贡献分数”建立了环境感知OOD检测理论，提出了由当前环境主项和先前环境次项（按环境间模内对齐度加权）构成的跨环境最大概念匹配（MCM）评分方法，并从理论上阐明了先前环境何时有益或有害于OOD检测。 |
| [^197] | [You Are the Trigger: Speaker-Activated Backdoors](https://arxiv.org/abs/2610.27210) | 提出说话人激活后门（SAB）新威胁模型：攻击者只需在微调阶段植入后门，即可让目标说话人的自然声音成为触发器，通过两阶段微调方法在Whisper-medium和Parakeet上实现90%以上的攻击成功率且误报率极低。 |
| [^198] | [Diffusion Score Estimation with Shallow Neural Networks: Regularity, Approximation, and Finite-Sample Rates](https://arxiv.org/abs/2610.27206) | 该论文为用浅层 ReLU 网络通过去噪分数匹配估计 Ornstein–Uhlenbeck 扩散分数建立了显式的有限样本理论，给出了时间一致的 Sobolev 正则性界、基于路徑范数的逼近误差控制，以及无界输入下经验 DSM 最小化器的高概率泛化风险界。 |
| [^199] | [Low Residuals, Wrong Value: Learning the Viscosity Solution of Hamilton-Jacobi-Isaacs Equations with PINNs](https://arxiv.org/abs/2610.27193) | 本文证明仅凭残差最小化不足以让PINNs学到Hamilton-Jacobi-Isaacs方程的粘性解，因为网络会收敛到残差很小但值函数与粘性解相差 $O(1)$ 的几乎处处解。 |
| [^200] | [On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models](https://arxiv.org/abs/2610.27190) | 语言模型存在一种隐式思考预算：在组合多个任务时，每个任务分配到的 token 数远少于单独处理时，导致即使模型能独立可靠地完成每个任务，组合后错误率仍显著上升，且需要长推理链的任务在组合下尤为脆弱。 |
| [^201] | [Meta-iLaD: Identifiable Latent Dynamics via Meta-Learning of Dynamics Environments](https://arxiv.org/abs/2610.27180) | Meta-iLaD 通过对动力学环境进行元学习，在少样本条件下学习可辨识的潜在动力学，从而实现对分布外环境中高维时间序列的准确预测。 |
| [^202] | [Understanding the Benefits of Scaffolds through Computational Separations](https://arxiv.org/abs/2610.27157) | 本文证明推理时脚手架即使基于平凡的 $\AC^0$ 原语，也能通过多次调用完成 $\NL$-完全的闭包计算，而固定深度的前向传播在标准复杂性猜想下无法表示该计算，从而解释了为何脚手架带来的能力提升无法通过蒸馏完全内化到模型之中。 |
| [^203] | [A Hybrid TFT-NBEATS Framework for Robust Multi-Horizon Photovoltaic Power Forecasting](https://arxiv.org/abs/2610.27155) | 提出了一种将TFT与N-BEATS相结合并通过自适应注意力机制动态融合的混合深度学习框架，能够同时捕捉短期波动与长期趋势，实现具有可解释性的鲁棒多时间尺度光伏功率预测。 |
| [^204] | [Cross-disorder Learning of Imaging-derived Joint Mophometric Markers in Alzheimer's and Parkinson's Diseases](https://arxiv.org/abs/2610.27150) | 本研究提出一种显式、可解释且空间自适应的迁移学习框架，利用ADNI和PPMI神经影像数据集揭示阿尔茨海默病与帕金森病共享的形态测量生物标志物及与临床表现相符的神经解剖学改变模式。 |
| [^205] | [ICLHerd: Herding-Based Data Selection for LLMs with In-Context Learning as a Proxy](https://arxiv.org/abs/2610.27142) | ICLHerd提出了一种基于von Mises-Fisher核的数据选择框架，通过平衡模式寻求与模式平均来克服传统最大相似度方法的聚类偏差，并利用上下文学习作为验证代理实现高效的超参数搜索，在大语言模型训练数据选择上优于现有算法。 |
| [^206] | [Towards Interaction State Recognition in Sensorized Gecko-Inspired Adhesive Interfaces for Orbital Servicing](https://arxiv.org/abs/2610.27141) | 该论文提出了一个基于嵌入式电容传感的实时交互感知框架，利用轻量级机器学习模型对仿壁虎粘附界面在粘附全生命周期中的五种交互状态进行分类识别，填补了粘附捕获技术在闭环状态感知方面的空白。 |
| [^207] | [Representing short-term movement to reveal animal identity and learning progress](https://arxiv.org/abs/2610.27140) | 本文提出一种基于“行为特征在相邻时间区间内相对稳定”这一假设的自监督表征学习框架，以时间邻近性为监督信号，直接从动物短期运动轨迹中学习紧凑表征，无需人工定义行为特征即可揭示动物个体身份与学习进度。 |
| [^208] | [Ontic state recovery through prediction](https://arxiv.org/abs/2610.27139) | 本文将状态定义为指定查询全域上的完整响应档案，并从查询歧义性、状态空间几何和预测目标选择三个角度，结合合成系统与神经记录数据，考察“预测”与“本体状态恢复”之间的关系。 |
| [^209] | [Ammonia-Hydrogen Phase Equilibria in Planetary Envelopes from First-Principles-Based Machine-Learning Molecular Dynamics](https://arxiv.org/abs/2610.27136) | 本研究利用基于密度泛函理论训练的机器学习原子间势进行大规模两相共存模拟，绘制了行星包层温压条件下氨-氢体系的相平衡图，发现临界温度随压力从500 K升至2000 K并在约30 GPa、2000 K以上趋于平坦，对应分子开始解离为原子。 |
| [^210] | [Glass-Box Deep Reinforcement Learning for High-Speed Flow Control](https://arxiv.org/abs/2610.27132) | 本文是一篇展望性文章，系统分析了深度强化学习应用于高速流动控制所面临的关键瓶颈（仿真成本与保真度、数据效率、极端环境感知、奖励设计、可解释性及跨马赫数与雷诺数泛化），并主张发展可解释的“玻璃箱”DRL方法以推动其实验落地。 |
| [^211] | [Reconstruction-Optimized Expert Pruning for Mixture-of-Experts Language Models](https://arxiv.org/abs/2610.27123) | 该论文提出OptEP方法，通过可扩展的连续松弛直接优化可微分专家选择变量，并结合轻量级基于重建的权重更新，将MoE模型专家剪枝问题转化为子集选择问题求解，在Qwen3-30B-A3B上取得最高15.05%的绝对准确率提升。 |
| [^212] | [RGH-GNN : Risk-Gated Heterogeneous Graph Neural Network for Financial Fraud Detection](https://arxiv.org/abs/2610.27107) | 该论文提出了一种风险门控异构图神经网络RGH-GNN，通过显式建模账户、设备等交易间的关系模式，并避免标准图神经网络对邻域信息的均匀聚合稀释稀少的欺诈信号，从而在类别不平衡和欺诈伪装的场景下提升金融欺诈检测效果。 |
| [^213] | [This is SParTA: A modular, efficient and scalable transformer architecture for jet-tagging](https://arxiv.org/abs/2610.27106) | SParTA通过利用喷注组分陡峭下降的横向动量谱，仅让最硬的N个组分参与二次复杂度的相互作用注意力，其余软组分通过线性交叉注意力整合，在保留软辐射和色流信息的同时避免了全自注意力的O(l²)内存瓶颈，实现了模块化、高效且可扩展的喷注标记。 |
| [^214] | [LOB-Impact Benchmark: Emergent Market Impact in Generative Limit Order Book Models](https://arxiv.org/abs/2610.27100) | 该论文提出LOB-Impact基准，首次证明市场冲击现象能够在仅通过下一词元预测训练的神经限价订单簿生成模型中自然涌现，并通过注入元订单与实证冲击估计器为消息级生成器提供了反事实市场冲击的评估框架。 |
| [^215] | [ClockMark: Adaptive and Host-Dependent Watermarking for Large Language Models](https://arxiv.org/abs/2610.27079) | 提出了ClockMark水印框架，首次基于“自适应嵌入”（保留先前采样决策状态）与“宿主相关嵌入”（利用当前LLM分布约束词元选择和状态更新）两大原则，在保持输出分布的同时实现对文本编辑的鲁棒水印嵌入。 |
| [^216] | [LEAF: LUT-aware Efficient Attention on FPGAs](https://arxiv.org/abs/2610.27078) | LEAF是首个面向FPGA的纳秒级查找表感知transformer，通过将注意力块中的常量矩阵-向量乘法制表化，实现了满足实时系统要求的超低延迟推理。 |
| [^217] | [Graph Neural Networks for Astrophysics and Cosmology](https://arxiv.org/abs/2610.27051) | 本综述论证了图神经网络（GNNs）因其契合物理数据的固有几何结构、能够融入已确立的对称性与先验知识、且比其他深度学习架构更具可解释性，因而比任何替代方法都更适合应用于天体物理学和宇宙学的研究。 |
| [^218] | [Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling](https://arxiv.org/abs/2610.27047) | 本文提出MEDIATOR——一种本体无关的离散分词器，通过将任意文本形式表达的临床概念映射到共享的向量量化码本，将EHR数据协调转化为开放词表分词问题，从而打破不同医疗系统间的编码壁垒。 |
| [^219] | [ReuF2F: Benchmarking Coding Agents on Formalized Undergraduate Research Problems](https://arxiv.org/abs/2610.27046) | 该论文提出了 ReuF2F，一个基于 Lean 4 的基准测试，包含 244 个源自本科生研究经验计划（REU）的形式化开放研究问题，用以更可靠地评估编码智能体解决开放数学问题的能力。 |
| [^220] | [Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery](https://arxiv.org/abs/2610.27030) | 该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。 |
| [^221] | [EGiME: Entropy-Guided Model Evaluation in Few-Shot Learning](https://arxiv.org/abs/2610.27029) | 提出了一种无需训练分类器的半监督评估准则EGiME，通过逆频率权重融合簇间熵与簇内熵来衡量预训练模型特征空间的聚类结构，在五个小样本基准上与真值准确率达到0.966的相关性。 |
| [^222] | [Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds](https://arxiv.org/abs/2610.27022) | 该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。 |
| [^223] | [Reliable Estimation of Differential Privacy under Selection Bias](https://arxiv.org/abs/2610.27021) | 本文揭示了以往差分隐私审计实现在运行成员推断攻击时引入选择偏差、从而产生统计上无效且被高估的隐私泄露下界这一问题，并提出了在选择偏差下可靠估计差分隐私的方法。 |
| [^224] | [MultiCard: Multiple Attribute Cardinality Estimation via Deep Learning](https://arxiv.org/abs/2610.27019) | MultiCard 提出了一个统一的端到端深度学习基数估计器，通过 Transformer 与 LSTM 相结合来同时处理 LIKE 字符串谓词和数值过滤谓词，并能动态评估直方图估计在不同算子和数据倾斜条件下的准确性。 |
| [^225] | [Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2610.27012) | 提出激活空间探索（ASE）方法，通过聚类正确与错误响应的激活差异构建潜在探索向量库，并将其应用于训练采样以引导模型探索多样化推理路径，从而提升可验证奖励强化学习在数学推理上的性能。 |
| [^226] | [Robust Latent Variable Graphical Models: Exact Recovery and Fundamental Limits](https://arxiv.org/abs/2610.27010) | 本文提出了一种用测度变换协方差替代样本协方差、并用合同变换核范数惩罚低秩部分的凸估计方法，用于鲁棒估计具有“稀疏减低秩”结构的高斯精度矩阵，并在受限可逆性与不可表示条件下证明了精确恢复保证，包括解的唯一性、无虚假边、边符号恢复正确以及秩和惯性的准确恢复。 |
| [^227] | [HierDTA: Hierarchical Structure-Aware Representation Learning for Drug--Target Affinity Prediction](https://arxiv.org/abs/2610.27001) | 提出层次化结构感知框架HierDTA，通过将药物表示为原子-基序异构图、将蛋白激酶结构域表示为残基相互作用图，并利用交叉注意力融合多级结构信息，从而提升药物-靶点亲和力预测的准确性。 |
| [^228] | [ORBIT: Autoregressive Time Series Generation via Next Frequency-Band Prediction](https://arxiv.org/abs/2610.27000) | 提出了统一的时间序列生成框架ORBIT，通过频率分层VQ-VAE学习跨频带的分层时间表示，并利用频带自回归Transformer实现从低频到高频的自回归生成，从而解决了异构时间模式的表示学习与生成协调两大难题。 |
| [^229] | [CorteXforge: Scenario-Based RF Dataset Generation on Real Testbeds](https://arxiv.org/abs/2610.26999) | CorteXforge是一个基于场景的框架，能够在真实SDR测试平台上自动生成真实且可重复的射频数据集，实验揭示了合成数据与真实数据之间的性能差距，同时证明合成数据预训练有助于模型适应真实空中测量数据。 |
| [^230] | [CoASFL: Coupled Consistency Optimization for Asynchronous Split Federated Learning](https://arxiv.org/abs/2610.26992) | 该论文提出CoASFL框架，通过动态激活合并机制和一致性感知的模型聚合机制，有效解决了异步拆分联邦学习由系统异构性与统计异构性引发的水平与垂直耦合不一致性问题。 |
| [^231] | [Dynamic Dual-Feedback Conformal Inference for Time Series Forecasting](https://arxiv.org/abs/2610.26987) | 该论文提出动态双反馈保形推断（DDCI），通过同时利用误覆盖指示的实际反馈和基于理想保形方法的估计反馈来自适应调整预测阈值，从而在分布变化环境下实现更可靠的时间序列预测不确定性量化。 |
| [^232] | [Latent Projection for Latent Planning](https://arxiv.org/abs/2610.26979) | 该论文发现世界模型训练中防止坍缩的正则化器（SIGReg）本身蕴含高斯-马尔可夫结构，可零额外成本地导出潜空间各方向的协方差（不确定性），从而在潜空间规划中有效剪除不携带信号的噪声维度。 |
| [^233] | [Transformers Learn Two Competing Sorting Algorithms, and Only One Length-Generalizes](https://arxiv.org/abs/2610.26974) | 该论文发现两层Transformer在排序任务上会根据权重初始化收敛到两种机制截然不同的解，其中只有同时使用两个注意力层的解能在仅训练固定长度序列的情况下完美实现长度泛化，且可以通过精心设计权重初始化来引导模型学习目标解。 |
| [^234] | [SZED at ArGuard Shared Tasks: Multi-Task Label-Attention and Cross-Modal Fusion for Arabic Hateful Meme Detection](https://arxiv.org/abs/2610.26972) | 该论文提出了一种融合MARBERTv2文本嵌入与CLIP视觉嵌入的多任务标签注意力架构，并引入级联校准机制，用于阿拉伯语仇恨表情包检测，在二分类任务上取得macro-F1 0.7597、在细粒度多标签任务上取得0.3142的成绩。 |
| [^235] | [When Weak Submodularity Meets Matroids: How Does Standard Greedy Behave?](https://arxiv.org/abs/2610.26967) | 本文解决了弱次模最大化在一般拟阵约束下标准贪心算法最坏情况近似保证这一长期开放问题，给出了以次模比 $\gamma$ 和拟阵秩 $r$ 表达的精化下界。 |
| [^236] | [FoundPDE: A generative foundation model for data-driven symbolic discovery of partial differential equations](https://arxiv.org/abs/2610.26963) | 提出FoundPDE——首个预训练的数据到符号生成式基础模型，利用Transformer将离散时空轨迹直接映射为符号偏微分方程，并通过傅里叶与一阶空间导数增强及少样本微调，实现了在噪声、稀疏数据下对广泛方程族的鲁棒发现与良好泛化。 |
| [^237] | [Dynamic System Inspired Continuous Large Language Model for AI4Science Tasks](https://arxiv.org/abs/2610.26959) | Continuum-LM将预训练的残差变换复用为动力系统算子，构建了通过常微分方程演化隐藏状态的连续深度语言模型，在不扩展主干网络的前提下增加了潜在计算能力，在八项科学任务上将平均精确匹配准确率显著提升了7.77至10.96个百分点。 |
| [^238] | [Beyond Policy Discrepancy: Capability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.26950) | 提出能力感知的多教师在线策略蒸馏方法 CA-MOPD，将强化学习专家模型合并与验证器估计的师生能力差距相结合，根据学生已掌握的内容动态调整各领域蒸馏强度，克服了仅基于策略差异分配监督的局限性。 |
| [^239] | [Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals](https://arxiv.org/abs/2610.26944) | 该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。 |
| [^240] | [Expert-Guided Rubrics as Dense Reward Signals for Reinforcement Learning in Non-Verifiable Domains](https://arxiv.org/abs/2610.26932) | 本文提出将少量固定的主题专家判断转化为基于评分标准的密集奖励信号，并结合检索指标解决GRPO在非可验证领域中因稀疏奖励导致优化停滞的问题。 |
| [^241] | [ShlokaDhara: A Multi-Speaker Sanskrit Speech Corpus for Learning Meter-Aware Prosody in Text-to-Recitation](https://arxiv.org/abs/2610.26929) | 该论文发布了 43 小时的多说话人梵语诵读语料库 ShlokaDhara，用以支持要求格律正确性的“文本到诵读”合成任务，并通过 16 位梵语专家的主观评估与新颖实证指标验证了 XTTS-v2 在该任务上达到最高的感知质量。 |
| [^242] | [CaMeWM: A Causal Medical World Model for Generative Tumor Evolution and Treatment Planning](https://arxiv.org/abs/2610.26923) | CaMeWM是一个因果医学世界模型，通过“干预到决策”的单一循环机制，为具有治疗支撑的TACE术后肿瘤演化生成反事实预测分布，从而实现对不同治疗方案的有效比较与规划。 |
| [^243] | [Steerable Pluralistic Policy Alignment from Anonymized Crowd Feedback](https://arxiv.org/abs/2610.26914) | 提出P3L框架，无需标注者身份信息，通过PCA分解从匿名群体反馈中联合学习轨迹嵌入与偏好方向共享基，实现可引导、可泛化的个性化策略对齐。 |
| [^244] | [LatticeMind: A Validation-Gated Agentic Framework for Density Functional Theory Workflows](https://arxiv.org/abs/2610.26869) | LatticeMind是一个验证门控的智能体框架，通过工作流契约、确定性验证器和有界修复机制，将自然语言请求转换为可执行且可检查的密度泛函理论（DFT）项目，解决了语言模型在串联多阶段第一性原理计算工作流时的可靠性问题。 |
| [^245] | [High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference](https://arxiv.org/abs/2610.26866) | 该研究将Transformer逐层推理的有序层状态轨迹作为经验对象，发现真实层顺序产生的轨迹对齐度是保持终点随机打乱的1.60至2.24倍，证明层顺序构成一种与架构兼容而非训练特有的可测量计算，并可通过局部转换执行器实现对内部推理轨迹的选择性控制。 |
| [^246] | [H2DF: Calibrated Target Bands for Controllable Language-Model Unlearning](https://arxiv.org/abs/2610.26864) | H2DF提出了一种校准的目标区间目标函数，通过数据依赖的目标带控制梯度上升压力，在遗忘达到目标后自动抑制或反转压力，从而实现可控的语言模型遗忘，同时保护保留数据的效用。 |
| [^247] | [Text-to-Image Generative Modeling Without Language Pretraining](https://arxiv.org/abs/2610.26863) | 本文提出完全从零预训练的早期融合多模态自回归模型家族Halcyon，证明无需互联网规模的语言预训练也能以大幅更少的计算量训练出强大且高质量的文本到图像生成模型。 |
| [^248] | [WeAct: Dynamic Weight Switching as Agent Actions](https://arxiv.org/abs/2610.26847) | WeAct在冻结主干上训练独立的能力专用LoRA适配器，并将适配器选择纳入智能体动作空间，使模型能在长时程任务中通过原生工具调用接口按需组合能力，同时避免灾难性遗忘。 |
| [^249] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^250] | [HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs](https://arxiv.org/abs/2610.26770) | HarnessPro提出了一种无需标签的智能体框架递归自我改进方法，通过帕累托优化权衡性能与执行成本，并提供精简、均衡、重型三种框架版本以适配不同的部署预算。 |
| [^251] | [Rethinking LLM Fine-Tuning via Weight Space Reparameterization: Preserving Safety during Downstream Adaptation](https://arxiv.org/abs/2610.26443) | 提出WSR-Tune框架，通过构建安全条件化的基对权重矩阵进行重参数化，使安全信息集中于少数方向并冻结这些安全关键方向，从而在大语言模型下游任务微调中保持安全对齐。 |
| [^252] | [A Simple and General Approach to top-$k$ Selection](https://arxiv.org/abs/2610.26407) | 本文提出一种通过最小化期望机会成本上界导出的替代目标来自适应分配采样批次的简单通用方法，用于解决贝叶斯top-$k$选择问题，并证明其后验期望机会成本的衰减速率至少达到任意采样规则最佳速率的一半。 |
| [^253] | [Selecting the Best Few with Human Evaluators: Adaptively Combining Ratings and Pairwise Comparisons](https://arxiv.org/abs/2610.26380) | 本文提出一种自适应评估分配方法，先利用人工评分对候选项目进行粗筛，再借助两两比较精化优秀项目的排序，从而在有限评估预算下高效地从大量候选中选出最优的 $k$ 个项目。 |
| [^254] | [Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation](https://arxiv.org/abs/2610.26030) | 通过近600次训练拟合的统一缩放律表明，在联合优化前向和后向循环次数时，完整BPTT训练的循环Transformer始终优于相同算力下使用截断BPTT的训练方式。 |
| [^255] | [Polynomial Learning of Positive ReLU Networks under Gaussian Inputs](https://arxiv.org/abs/2610.25116) | 该论文在不对隐藏方向施加任何分离或条件化假设的前提下，首次为标准高斯输入下正ReLU网络的学习给出了样本与实数算术复杂度均为多项式的算法，并提供了误差与风险的理论保证。 |
| [^256] | [Process-Aware Event Log Augmentation for Process Prediction under Labeled Data Scarcity](https://arxiv.org/abs/2610.24777) | 该论文提出SiamSA-PPM自监督框架，将孪生学习与利用控制流语义和频繁行为模式的过程感知数据增强相结合，以解决预测性过程监控中标注数据稀缺和日志变异性低的问题，在下一活动和最终结果预测上取得了优于现有方法的性能。 |
| [^257] | [DeliMF: Gated Cross-Attention Fusion of Structured Time Series and Clinical Text for Early Intensive Care Delirium Prediction](https://arxiv.org/abs/2610.24585) | DeliMF通过门控交叉注意力将结构化时间序列与临床文本融合，实现了ICU患者入院首日的早期谵妄预测，弥补了传统方法仅依赖结构化数据而忽略临床笔记信息的不足。 |
| [^258] | [Blackbox Performance Analysis of LLM-based Network Control](https://arxiv.org/abs/2610.24348) | 本文提出LENS——首个面向基于大语言模型的网络控制系统的黑盒性能分析工具，通过扩展深度强化学习来搜索性能差距和使LLM表现不佳的输入，并在流量工程和GPU分配系统中发现了显著的性能差距。 |
| [^259] | [Learning Administrative Boundary-Aware Location Encoders via Reverse Geocoding](https://arxiv.org/abs/2610.24306) | 本文首次提出利用逆地理编码文本作为自监督训练信号来学习行政边界感知的位置编码器，并设计了专属架构RevGeoRQ，可提升现有编码器性能并在多个下游任务上超越当代方法。 |
| [^260] | [How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies](https://arxiv.org/abs/2610.23946) | 该论文将智能体强化学习的训练数据分解为任务、环境和轨迹三个可独立扩展的维度，并拟合出各维度的缩放定律，以回答在给定合成预算下应优先扩展哪个数据维度这一关键问题。 |
| [^261] | [NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models](https://arxiv.org/abs/2610.23271) | NestedVLA提出用嵌套超网络生成任务自适应的动作专家参数来事后整合独立学习的技能，从而在多个机器人操作基准上实现灵活的任务条件化技能复用并有效缓解任务间干扰。 |
| [^262] | [Response-Informed Value Estimation for Cost-Aware Multi-Turn LLM Routing](https://arxiv.org/abs/2610.23239) | 提出RIVET方法，在训练阶段利用每次调用后的实际响应信息为每一轮提供价值监督信号，而在部署时仅依赖调用前信息进行成本感知的多轮LLM模型路由。 |
| [^263] | [A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation](https://arxiv.org/abs/2610.23185) | 提出MoEvolve框架，通过增加MoE模型每token激活预算来挖掘模型内部潜在更强教师，并采用“先恢复后提升”的策略内双向蒸馏策略强化MoE大语言模型。 |
| [^264] | [DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium](https://arxiv.org/abs/2610.23047) | Difflet 是面向 AWS Trainium 的扩散 Transformer 服务引擎，通过五个流水线共享统一编译后端、按注意力头切分的融合注意力以及基于设备端探针的逐步缓存决策，将去噪步骤加速 1.2-4.3 倍，常驻服务速度提升 1.5-3.4 倍。 |
| [^265] | [Towards Hyperparameter Transfer for Differentially Private Optimization](https://arxiv.org/abs/2610.22976) | 本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。 |
| [^266] | [Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?](https://arxiv.org/abs/2610.22887) | 提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。 |
| [^267] | [The LLM-Attacker: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems](https://arxiv.org/abs/2610.22722) | 本文提出LLM-Attacker统一红队评估框架，通过记录攻击者的选择实现切分式与分布式大语言模型隐私评估的可复现与可比较，并揭示训练期间返回梯度会使token恢复率从94.20%升至97.38%、精确文档恢复率从13.71%升至37.77%。 |
| [^268] | [Knowledge That Is Linear Before Fine-Tuning Generalizes After It](https://arxiv.org/abs/2610.22300) | 论文提出区分语言模型中支持泛化的“活性事实”与仅可回忆的“惰性事实”，并通过重标注实验发现：微调前在模型内部呈线性可读的知识，在微调后能够泛化到共享相同答案的未训练实体上。 |
| [^269] | [Training-Free On-Policy Distillation](https://arxiv.org/abs/2610.22236) | 本文提出免训练的在策略蒸馏方法，当多个专家模型共享同一基础模型时，仅需通过几百个采样样本的一次前向传播、以最小二乘方式逐层融合，即可在无需任何梯度训练的情况下将多个专家的能力聚合到一个模型中，且平均准确率超过任何单一专家模型。 |
| [^270] | [HyCo: Hybrid Drafting for Cost-Efficient Mobile-Cloud Speculative Decoding](https://arxiv.org/abs/2610.21436) | HyCo提出混合草稿生成机制，将服务器模型的原生多词预测头部署到移动NPU上并与小型语言模型单次推理协同工作，配合量化和ε阈值松弛策略，使移动-云推测解码的服务器GPU时间降低约29%-39%。 |
| [^271] | [Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning](https://arxiv.org/abs/2610.21403) | 提出FaShift框架，超越传统不变性方法，通过按因子诊断分布变化、干预效应和跨环境稳定性，对多模态学习中的偏移信息进行选择性抑制、保留或削弱。 |
| [^272] | [Does On-Policy Distillation Need to Complete Every Rollout?](https://arxiv.org/abs/2610.21337) | 该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。 |
| [^273] | [Complete or Sparse: A Tale of Two Identifiabilities](https://arxiv.org/abs/2610.20851) | 本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。 |
| [^274] | [On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination](https://arxiv.org/abs/2610.18727) | 论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。 |
| [^275] | [Deficit-Driven Stratified Acquisition for Building STRATA-S12: An Open SAR-Optical Corpus for Self-Supervised Learning](https://arxiv.org/abs/2610.17680) | 提出了缺口驱动分层采集算法（DDSA），将构建配准的 Sentinel-1/Sentinel-2 数据集建模为半球校正分层上的约束最大覆盖问题，并通过人机协同的缺口修复机制确保全部 192 个分层得到完整覆盖，为自监督地球观测学习提供了高质量的开放 SAR-光学语料库。 |
| [^276] | [Measuring Variable Importance Using Proportional Inclusive Marginal Effects](https://arxiv.org/abs/2610.17053) | 提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。 |
| [^277] | [Learning Reachability-Based Safety Representations for Real-Time Robot Navigation in Unknown and Dynamic Environments](https://arxiv.org/abs/2610.16522) | 本论文通过离线在哈密顿-雅可比可达性解上训练神经隐式表示，并仅学习非负的动力学修正项，使基于优化的控制器能够在未知动态环境中实时获得安全信息，实现实时安全的机器人导航。 |
| [^278] | [Compute-Optimal Scaling Laws for Spiking Language Models](https://arxiv.org/abs/2610.15146) | 本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。 |
| [^279] | [When Dynamic Scaling Fails: A Scale-State Feedback Governs Recurrent-State Quantization in Linear-Attention and SSM LLMs](https://arxiv.org/abs/2610.15139) | 该论文揭示了线性注意力与状态空间大语言模型中循环状态量化的失效根源——量化器尺度与状态之间的反馈回路使误差在递推中累积并引发范数爆炸，并证明数值不稳定与功能退化是可分离的两种失效，其中前者可通过全精度参考缩放完全消除。 |
| [^280] | [What Survives When Activation Steering Is Used as Memory](https://arxiv.org/abs/2610.14958) | 该研究提出一种免训练的激活记忆机制，并证明激活引导偏移作为注意力效应的固定平均值只能存留与问题无关的内容：这种记忆虽具有约98%的高选择性，但用户事实仅被存储在最后几层且几乎从不被使用，表明用户事实与偏好仍需以文本形式记忆。 |
| [^281] | [Some remarks on the connection between generative diffusion and optimal transport](https://arxiv.org/abs/2610.13614) | 本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。 |
| [^282] | [TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers](https://arxiv.org/abs/2610.12474) | 本文首次对十亿参数规模的TTS扩散变换器开展4比特权重与激活（W4A4）训练后量化研究，揭示了量化主要损害说话人音色而非可懂度这一现象及其来源，并提出感知引导的量化修复方法。 |
| [^283] | [V-CoLA: Vision Token Compression with Linear Attention](https://arxiv.org/abs/2610.11251) | 提出V-CoLA，一个专为线性注意力混合架构设计的无需训练的视觉Token压缩框架，通过唯一性感知的重要性准则和自适应Token合并策略，解决了现有压缩方法在该类架构下性能显著下降的问题。 |
| [^284] | [On linearity or non-linearity in machine learning for quantum chaotic dynamics](https://arxiv.org/abs/2610.10697) | 该研究将量子混沌动力学预测转化为时间序列预测问题，以里德堡原子阵列可实现的PXP链为基准，系统比较了非线性Transformer与简单线性模型DLinear在从遍历到量子多体疤痕态等不同动力学区间中的预测能力。 |
| [^285] | [Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts](https://arxiv.org/abs/2610.10460) | 提出 Δ-MOPD，通过迁移教师相对其基础模型的 logit 偏移并重新锚定到学生的冻结初始状态，消除端点策略中继承的基础模型偏好干扰，从而在多教师在线策略蒸馏的两种设置中降低教师项范数比和目标-学生 KL，实现更纯粹的教师后训练知识传递。 |
| [^286] | [Loopy: Low-Bit Quantization Framework for Looped Language Models](https://arxiv.org/abs/2610.05265) | 提出了Loopy量化框架，通过感知循环深度的目标函数为循环语言模型的共享核心选择最优低比特量化配置，解决了量化误差随循环迭代累积传播且配置排序随深度变化的问题。 |
| [^287] | [Ranking Bandits for Carousel Interfaces with Observable Browsing Depth](https://arxiv.org/abs/2610.05220) | 本文提出了浏览深度可观测的轮播界面排序老虎机模型，利用浏览深度信号区分已展示未点击与从未展示的物品，并设计了基于UCB、Thompson采样和DMED的三种算法，给出了理论性能上界。 |
| [^288] | [EDISCO: Equivariant DIScrete Diffusion for Euclidean Combinatorial Optimization](https://arxiv.org/abs/2610.04953) | EDISCO是首个面向欧几里得组合优化问题、生成分布具有精确E(2)不变性的离散扩散模型，它通过等变边评分网络与连续时间马尔可夫链相结合，实现高效的多步推理，无需依赖数据增强或正则化。 |
| [^289] | [TreeWalker: Partial Evaluation for Grouped Tree-Ensemble Inference](https://arxiv.org/abs/2610.03939) | TreeWalker 提出针对分组树集成推理的部分求值方法，利用组内行共享特征值的相关性，对每组仅遍历每棵树一次即可完成推理，并给出了每棵树工作量的结构化分解，同时兼容标准 LightGBM 和 XGBoost 模型。 |
| [^290] | [BridgeCast: Bridging Ocean Wave Forecasts to Reanalysis via Flow Matching with Exogenous Variables](https://arxiv.org/abs/2610.03759) | BridgeCast 提出了一种基于条件流匹配与 Transformer 架构的物理-AI 混合概率模型，能够将物理模式的海浪预报偏差校正为再分析级别的场，并可融合外生大气变量实现灵活的时空推断。 |
| [^291] | [FERPO: Forward Entropy-Regularized Policy Optimization](https://arxiv.org/abs/2610.02198) | FERPO提出了一种无需对评论家求动作导数的在策略最大熵强化学习算法，通过熵与KL散度正则化的策略改进目标推导最优动作分布，并借助自归一化重要性采样以最小化前向KL来拟合Actor，从而避免了因价值预测不准导致的不可靠策略更新。 |
| [^292] | [Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models](https://arxiv.org/abs/2610.00661) | 提出StepRS-GRPO方法，通过在去噪状态间动态调整群组优势变换的风险系数，使扩散语言模型在数学推理任务上同时提升pass@1准确率、pass@k覆盖率和答案多样性。 |
| [^293] | [Geometry-Dependent Bounds for Online Non-Monotone DR-Submodular Maximization](https://arxiv.org/abs/2610.00545) | 该论文为在线非单调DR-次模最大化证明了系数4/9的一阶遗憾界（改进此前0.401的基准），给出依赖于集合几何的更优系数，并通过对称间隙构造建立了离线预言机上界β*≈0.4704。 |
| [^294] | [$S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient](https://arxiv.org/abs/2609.37976) | 论文首次发现推理能力的关键在于思考模型权重中位于非思考模型主导子空间零空间内的分量，并提出无需训练的谱零空间交换方法S³，通过配对检查点组合在保持精度的同时大幅提升推理效率。 |
| [^295] | [ROSS: Relearning from Self-Generated Rollouts through Selective Supervision](https://arxiv.org/abs/2609.35954) | ROSS通过将完整历史rollout轨迹保留为上下文、仅对选定的模型生成延续施加损失的选择性监督方式，有效再利用历史自生成经验，在强化学习、多教师蒸馏和智能体学习中持续提升大语言模型性能。 |
| [^296] | [Binarization Flattens the Score Space](https://arxiv.org/abs/2609.35797) | 将LLM评审奖励二值化为通过/不通过会完全掩盖底层分数空间中的仿射拉伸信息，而引入第三个等级即可通过第二阈值消除这种歧义并恢复可检测的分数细节。 |
| [^297] | [QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents](https://arxiv.org/abs/2609.33848) | QwenGyre提出了一种面向超长时程LLM智能体的端到端弹性在线强化学习框架，通过在不中断执行的情况下在rollout与训练之间动态重分配GPU，并利用轨迹处理器重建分支历史、为部分进展打分和去重冗余路径，有效解决了GPU大量闲置与轨迹冗余两大难题。 |
| [^298] | [Marginal Calibration Masks Generator Differences in Held-Out Evaluation of Multivariate Time Series, with an Application to ERA5 Weather Sequences](https://arxiv.org/abs/2609.33761) | 该论文证明了在多变量时间序列生成模型的留出评估中，边际校准会使KS距离几乎不携带生成器信息，从而掩盖不同生成模型之间的真实差异，并通过ERA5天气序列上StatD2GAN与RCGAN、TimeGAN的对比实验验证了这一结论。 |
| [^299] | [Multi-Marginal Inverse Optimal Transport for Contrastive Learning Via Explicit Anchor-Positive-Negative Coupling](https://arxiv.org/abs/2609.33741) | 该论文提出Neg-MMIOT-CL方法，通过显式建模锚点-正样本-负样本三元组的多边际逆向最优传输耦合，克服了现有逆向OT方法因不利用负样本而导致的维度坍塌问题，提升了对比学习的下游性能。 |
| [^300] | [What Does the Rank Buy? A Spectral and Distributional Analysis of Low-Rank Adaptation](https://arxiv.org/abs/2609.32002) | 该论文证明在硬性逐因子范数预算下，LoRA的秩上限无法起到容量控制作用——所有复杂度与位移泛函均由秩一更新最大化、Rademacher复杂度与秩无关且分布位移上界与秩无关且紧致——并指出了秩真正发挥作用的两个位置。 |
| [^301] | [KuaFu: Compressing Long User Behavior into Understanding at Billion Scale](https://arxiv.org/abs/2609.31045) | 该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。 |
| [^302] | [Response-state Learning for Transferable Vibrational Spectroscopic Characterization with Electron Prior](https://arxiv.org/abs/2609.28935) | 提出SO(3)等变神经卡尔曼网络SENK，通过等变Transformer主干、神经卡尔曼桥与NBO电子先验通路构建响应态级联，实现从小分子到类药体系跨化学空间的可迁移、高保真IR/Raman光谱预测，并在QM9S和QMe14S上超越DetaNet。 |
| [^303] | [EMERGE: Resolution-Agnostic Point Cloud Generation with Equivariant Graph-Based Diffusion](https://arxiv.org/abs/2609.26039) | 提出了EMERGE，首个完全SE(3)等变的基于图的扩散模型骨干网络，实现了分辨率无关的点云生成，并支持在多个任意分辨率下的零样本推理。 |
| [^304] | [AURA: Angular Update Rate Adaptation for training complex-valued neural networks](https://arxiv.org/abs/2609.25914) | AURA是一种即插即用的复值神经网络逐参数步长自适应方法，通过衡量复参数连续更新在长度、对齐和旋转方向上的一致性来动态调整步长，无需额外梯度计算即可叠加于任何一阶优化器之上且不改变其更新方向。 |
| [^305] | [The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability](https://arxiv.org/abs/2609.19616) | 该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。 |
| [^306] | [A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings](https://arxiv.org/abs/2609.19083) | 提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。 |
| [^307] | [Decentralized Optimal Equilibrium Learning Over Dynamic Networks](https://arxiv.org/abs/2609.17601) | 该论文提出了一种适用于动态通信网络的去中心化最优均衡学习算法，智能体通过交换带时间戳的表格数据与时间多数重建机制学习社会最优均衡，并在功利主义和比例公平社会福利目标下实现了有限时间对数遗憾保证。 |
| [^308] | [Truthful Calibration Measures for Sequential Prediction](https://arxiv.org/abs/2608.21348) | 本文证明了在序列二值预测中，精确真实的校准度量无法同时满足完备性和健全性，并提出了两种近似真实的构建方法，实现了可扩展的乘性近似校准度量。 |
| [^309] | [Harnessing Magnitude-Only and Complex Measurements for Improved Dynamic MRI Reconstruction with Learned Priors](https://arxiv.org/abs/2608.18036) | 本文提出一种利用k空间幅值信息与复数测量相结合的物理驱动深度学习重建方法，显著提升动态MRI重建质量。 |
| [^310] | [ARISE: An adaptive residual-informed stability ensemble for feature selection in small-sample biomedical omics](https://arxiv.org/abs/2608.14866) | ARISE通过自适应加权组合多种相关性信号和残差信息冗余控制，在小样本生物医学组学数据中显著提升了特征选择的预测性和稳定性。 |
| [^311] | [CFR without Unbiasedness: Deterministic Guarantees for Persistent Public-Chance Schedules](https://arxiv.org/abs/2608.14761) | 本文提出了一种确定性目标转移定理，针对持久部分评估的CFR，在均匀加法公共截断下，通过引入公共借方项界定了可利用性，并证明了连续平衡调度在RM和RM+下的收敛性。 |
| [^312] | [Generalized Convexity and Smoothness via Conjugate Duality, with Applications to Deep Network Training](https://arxiv.org/abs/2608.09523) | 该论文通过勒让德函数与凸共轭提出了统一的 \(\mathcal{H}(\psi)\)-凸性与 \(\mathcal{H}(\Psi)\)-光滑性框架，突破了传统优化理论对可微性、凸性和光滑性的假设限制，并为深度神经网络训练建立了新的理论基础，包括证明广义梯度下降的最优学习率恰好为 1。 |
| [^313] | [FEAST: Federated Shared-Space Training for Resource-Heterogeneous Clients](https://arxiv.org/abs/2608.09250) | FEAST通过在每个客户端资源预算内联合训练多个子网络，并结合预算定制的子超网路由与稀疏聚合机制，解决了联邦学习中因客户端资源异构导致的参数训练不平衡问题，且支持无需联邦重训即可事后提取新子网络。 |
| [^314] | [Using Data-Derived Priors to Guide CNN Architecture Design for NIR Chemometrics](https://arxiv.org/abs/2607.25636) | 该论文提出将25个近红外光谱数据集的内在特性（如规模、熵、自相关等）转化为CNN架构设计的经验先验，特别是卷积感受野的设置，并通过热启动启发式加速贝叶斯超参数优化，经留一数据集验证证明了其有效性。 |
| [^315] | [$H^2(\Omega)$ A Priori Error Bounds for Boundary-Adapted PINNs Solving Elliptic Dirichlet Problems](https://arxiv.org/abs/2607.19167) | 本文通过乘以适当的到边界距离函数精确施加边界条件，构造了“边界自适应”神经网络，证明精确施加边界条件（而非 $L^2$ 边界惩罚）可以实现椭圆狄利克雷问题的 $H^2(\Omega)$ 误差控制，并推导出显式的高概率收敛速率。 |
| [^316] | [DiffEEG: A Self-Supervised Denoising Diffusion Model for Learning EEG seizure-related Representations](https://arxiv.org/abs/2607.11578) | DiffEEG是一个960万参数的自监督基础模型，通过去噪扩散预训练与强化学习微调，在标注稀缺和极端类别不平衡的条件下学习脑电图癫痫发作表征，并在患者级别严格评估中取得了优异的癫痫检测性能。 |
| [^317] | [SafeExplorer: An Unbiased Mixed-Policy Gradient for Reinforcement Learning with Recovery Interventions](https://arxiv.org/abs/2607.08925) | 提出 SafeExplorer，一种基于混合策略回报无偏梯度估计器的 PPO 即插即用改进方法，即使在使用确定性恢复策略进行安全强化学习训练时也能消除在线策略更新偏差。 |
| [^318] | [Data-Driven Soft Labeling Scales DNA Read Classification to Whole-Body Cell-Type Deconvolution](https://arxiv.org/abs/2607.04987) | 该论文提出一种数据驱动的软标签方法，能够将DNA读段分类规模化扩展至全身细胞类型反卷积任务。 |
| [^319] | [Probe, Don't Prompt: A Hidden-State Probe for Metadata Filtering in Multi-Meta-RAG](https://arxiv.org/abs/2607.03929) | 该论文提出用小型开源语言模型隐藏状态上训练的本地确定性多标签探针，替代Multi-Meta-RAG中基于GPT-3.5的新闻来源元数据提取器，从而免去专有API调用并将输出严格限制在允许的来源列表内。 |
| [^320] | [Federated learning with heavy-tailed gradient noise and communication noise: a variance-reduction based algorithm](https://arxiv.org/abs/2606.22466) | 提出了一种基于方差缩减的联邦学习算法VRA-FedSGD，通过动量方差缩减结合非线性映射以及方差缩减聚合机制，有效应对重尾梯度噪声和通信噪声，并在均方和几乎必然意义下实现了对非凸目标的收敛速率保证。 |
| [^321] | [Service-Induced Congestion in Memory-Constrained LLM Serving](https://arxiv.org/abs/2606.15555) | 该论文将内存受限的LLM服务建模为离散时间动力学系统，揭示了服务过程本身会内生地累积显存压力并引发请求驱逐，并证明了在同质负载下无驱逐平衡态不稳定，系统通常会收敛到带驱逐的极限环。 |
| [^322] | [An AI Security Agent for University ACMIS: Multi-Vector Threat Detection and Automated Response](https://arxiv.org/abs/2606.08270) | 本文提出了一种面向大学学术管理信息系统的AI安全代理，通过融合监督式异常检测、行为分析与NLP密码找回聊天机器人，对五个运营层面进行多向量威胁监控，并借助四级风险升级框架实现自动化响应。 |
| [^323] | [The price of multi-group transductive learning](https://arxiv.org/abs/2606.04423) | 该论文揭示了在留一直推设定下，多群体学习器的错误率可能遭受随群体数量线性增长（最大达样本量平方根）的乘性惩罚，而在群体可实现的统计设定下，该惩罚仅为与群体数量无关的多对数因子。 |
| [^324] | [Decomposing Refusal Steering in Mixture-of-Experts Models](https://arxiv.org/abs/2606.04160) | 该研究将拒绝引导方法扩展到混合专家模型，发现单个专家在自由选择位置时平均可恢复78%的完整引导效果，从而揭示了MoE模型中拒绝机制在各组件间的运作方式。 |
| [^325] | [The Shape of Wisdom: How Language Models Converge on an Answer](https://arxiv.org/abs/2606.01202) | 该研究发现，语言模型在回答事实性问题时，得到相同答案的问题会在内部状态空间中收敛到共享的“答案质心”，该质心在网络约47%-71%深度处突然变得可读取，且只取决于答案内容本身而非问题的表述方式。 |
| [^326] | [Measuring, Localizing, and Ablating Alignment Signatures in LLMs](https://arxiv.org/abs/2605.30526) | 该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。 |
| [^327] | [Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal](https://arxiv.org/abs/2605.26772) | 大型推理模型中的拒答并非仅由单一方向子空间介导，思维链会主动携带并强化拒答信号，使得简单的激活引导难以有效控制拒答行为。 |
| [^328] | [QML-PipeGuard: Detecting Channel-Substitution Attacks on Quantum Machine Learning Pipelines in Untrusted Clouds](https://arxiv.org/abs/2605.25066) | 该论文提出QML-PipeGuard，一种在运行时利用客户已有的测量统计来检测不可信云端量子机器学习服务中信道替换攻击的方案，可防范审计规避、服务降级和可复现性审查中的结果伪造。 |
| [^329] | [Interdomain Attention: Beyond Token-Level Key-Value Memory](https://arxiv.org/abs/2605.24330) | 提出跨域注意力（IA）这一新型混合架构，在同一层内将键值信息压缩到SSM参数化的循环记忆中并隐式重建注意力矩阵，从而在保持SSM计算效率的同时近似点积注意力。 |
| [^330] | [A Boundary-Layer Mechanism for One-Third Scaling in Online Softmax Classification](https://arxiv.org/abs/2605.22341) | 论文揭示了平滑代理损失与离散硬标签的失配使在线SGD训练的softmax学生模型在训练晚期仅由教师决策边界附近不断收缩的样本层驱动学习，导致测试损失和泛化误差均呈α^{-1/3}幂律标度；其中泛化误差的1/3指数是SGD在线噪声效应，可通过学习率退火将其改善至1/2。 |
| [^331] | [Dynamic Model Merging Made Slim](https://arxiv.org/abs/2605.18904) | DiDi-Merging通过可微秩选择在共享参数与专家参数之间联合分配低秩容量，并结合无数据的参数空间精炼，在仅1.24倍骨干网络存储的预算下实现了领先的动态模型合并性能。 |
| [^332] | [PACE: Geometry-Aware Bridge Transport for Single-Cell Trajectory Inference](https://arxiv.org/abs/2605.18587) | PACE通过构建状态和时间依赖的各向异性黎曼度量等三个耦合组件，从破坏性单细胞时间序列快照中恢复几何一致的连续传输动力学，解决了发育异步导致的轨迹错位这一核心难题。 |
| [^333] | [Bug or Feature$^2$: Weight Drift and Activation Sparsity](https://arxiv.org/abs/2605.17659) | 本文揭示了标准损失函数与正向偏置激活函数的相互作用会引发优化过程中固有的负向权重漂移，进而在训练早期自发产生高达约90%的激活稀疏性，并发现当激活稀疏度超过约70%时模型性能会急剧退化。 |
| [^334] | [Tensor Cookbook: Mastering Tensors through Diagrams](https://arxiv.org/abs/2605.16610) | 本文通过张量网络的图形化（图示）语言来帮助掌握高维张量，将张量收缩编码为图中的边，从而降低记号复杂度并揭示指标记号所掩盖的结构性质。 |
| [^335] | [StippleDiffusion: Capacity-Constrained Stippling using Controlled Diffusion](https://arxiv.org/abs/2605.15816) | 提出了首个基于扩散模型的点画生成采样器StippleDiffusion，通过在最优传输网格点集扩散基线上添加ControlNet分支，在推理时同时满足学习到的局部点分布先验和由图像定义的容量约束，突破了传统迭代优化方法慢且不可微、以及以往学习方法仅支持无条件点生成的局限。 |
| [^336] | [MuteBench: Modality Unavailability Tolerance Evaluation for Incomplete Multimodal Fusion](https://arxiv.org/abs/2605.15235) | MuteBench是一个评估多模态融合架构在模态缺失和模态内缺失两种失效模式下鲁棒性的大规模基准，研究发现架构家族是决定鲁棒性的最强预测因子，其重要性超过参数数量。 |
| [^337] | [LightSplit: Practical Privacy-Preserving Split Learning via Orthogonal Projections (Full Version)](https://arxiv.org/abs/2605.13265) | LightSplit在切分层采用轻量级的固定正交随机投影作为信息瓶颈，同时降低了拆分学习的通信开销并抵御重建攻击保护隐私。 |
| [^338] | [STRABLE: Benchmarking Tabular Machine Learning with Strings](https://arxiv.org/abs/2605.12292) | 该论文提出了STRABLE，一个包含108个真实世界表格的基准测试语料库，并通过对445个流水线的大规模实证研究，首次系统比较了端到端字符串建模架构与先将字符串编码为数字的模块化流水线在表格机器学习中的表现。 |
| [^339] | [COSMOS: Model-Agnostic Personalized Federated Learning with Clustered Server Models and Pseudo-Label-Only Communication](https://arxiv.org/abs/2605.11165) | COSMOS是一个模型无关的个性化联邦学习框架：客户端只需向服务器传输伪标签预测，服务器据此按预测相似度聚类并用自身算力训练聚类专属模型再蒸馏回客户端，同时首次从理论上证明了这种蒸馏可实现指数级的个性化风险收缩。 |
| [^340] | [Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2605.10765) | 提出DRAPE框架，通过动态生成实例特定的跨模态软提示，实现针对每个查询-图像对的个性化适应，从而在多模态持续指令微调中缓解灾难性遗忘。 |
| [^341] | [Follow the Mean: Reference-Guided Flow Matching](https://arxiv.org/abs/2605.10302) | 该论文发现流匹配的速度场仅由条件终点均值决定，因此只需通过参考样本改变该均值，即可在不微调的情况下引导预训练模型实现颜色、身份、风格和结构的可控生成。 |
| [^342] | [Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation](https://arxiv.org/abs/2605.08541) | 本文证明在固定每参数Token比例下拟合LLM缩放定律会因N与D指数近似相等导致的共线性而使最小二乘问题病态化、尺度系数不可辨识，并推导出实现稳健外推所需的TPP多样性闭式阈值。 |
| [^343] | [RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for Rheumatoid Arthritis](https://arxiv.org/abs/2605.05616) | RAM-H1200是一个包含来自六个医疗中心的1,200张手部X光片的多层次标注数据集，为类风湿关节炎评估提供了从解剖结构分割、骨侵蚀定量分析到临床SvdH评分的统一基准。 |
| [^344] | [Non-Stationarity Breaks Permutation Surrogates in Multi-Agent Reinforcement Learning: Diagnosis and Remedies](https://arxiv.org/abs/2604.23716) | 该研究发现非平稳性会使基于置换代理的信息论影响检验在多智能体强化学习中产生接近100%的假阳性，并提出通过改变零模型而非仅排除训练瞬态阶段来修复这一缺陷。 |
| [^345] | [Binary Agreement for Efficient Unsupervised Confidence Calibration of Reasoning LLMs](https://arxiv.org/abs/2604.19444) | BASC将锚定响应与单个独立探测响应之间的一致性视为伯努利观测来估计潜在自洽性，以每个校准查询仅需两个响应、部署时仅需一个响应的极低成本，实现了推理大语言模型的无监督置信度校准。 |
| [^346] | [Simpler Logarithmic Approximation Algorithms for the Optimal Decision Tree and Adaptive Set Cover](https://arxiv.org/abs/2604.12036) | 该论文提出了一种带有简化分析的新贪心算法，为最优决策树问题同时实现了最坏情形代价低于 $3.65\log_2 n$ 和平均情形代价低于 $7.3\log_2 n$ 的对数近似保证。 |
| [^347] | [Variational Graph Neural Networks for Uncertainty Quantification in Inverse Problems](https://arxiv.org/abs/2603.29515) | 本文提出一种将变分层集成到图神经网络中的VGNN架构，通过建模概率分布来量化逆问题中深度学习预测的不确定性，从而为计算力学中的关键应用提供可靠的置信度度量。 |
| [^348] | [Abstract Compression of Conversational Audio Context for LLM-based ASR](https://arxiv.org/abs/2603.26246) | 提出抽象压缩方法，将对话历史音频压缩为固定数量的潜在标记并保留转录文本，使基于LLM的语音识别以更小开销提升上下文实体的识别效果。 |
| [^349] | [Interpretable PM2.5 Forecasting for Urban Air Quality: A Comparative Study of Adaptive Time-Series Models](https://arxiv.org/abs/2603.25495) | 本研究比较了SARIMAX、Prophet和NeuralProphet三种轻量级可解释模型在北京小时级PM2.5预测中的表现，并评估了每周滚动重拟合与冻结预测加在线残差校正两种自适应部署策略的实际效果。 |
| [^350] | [Adaptive Dual-Teacher Distillation with Subnetwork Rectification for Bridging Semantic Gaps in Black-Box Domain Adaptation](https://arxiv.org/abs/2603.22908) | 提出了一种自适应双教师蒸馏框架DDSR，通过自适应预测融合与子网络校正策略，调和黑盒源模型的任务特定知识与视觉-语言模型的语义先验之间的差异，有效弥合了黑盒域自适应中的语义鸿沟。 |
| [^351] | [Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails](https://arxiv.org/abs/2603.18280) | 该论文以中国语言模型的政治审查为自然实验，发现对齐机制真正作用于“从概念检测到行为策略”的路由环节而非概念检测本身，从而解释了仅基于拒绝行为的对齐评估为何会失效。 |
| [^352] | [SCOPE: Scalable Training-Free Semantic Coreset Selection for Long-Tailed Federated Learning](https://arxiv.org/abs/2603.12976) | SCOPE提出了一种免训练的联邦核心集选择框架，各客户端仅用冻结的轻量视觉-语言模型对样本评分一次并上传每类标量统计量，服务器据此构建全局类别画像，从而为长尾非独立同分布数据提供可扩展、无需GPU训练的核心集选择方案。 |
| [^353] | [Hybrid Belief Reinforcement Learning for Efficient Coordinated Spatial Exploration](https://arxiv.org/abs/2603.03595) | 本文提出混合信念-强化学习（HBRL）框架，第一阶段利用LGCP空间信念与PathMI多步前瞻规划器进行信息驱动探索，第二阶段通过双通道知识迁移热启动SAC智能体，从而实现多智能体协同空间探索的高效学习与规划。 |
| [^354] | [Scalable Posterior Uncertainty for Flexible Density-Based Clustering](https://arxiv.org/abs/2603.03188) | 该论文将基于密度的聚类与鞅后验分布相结合，提出了一种完全并行化、可扩展的贝叶斯不确定性量化框架，兼容归一化流等可微密度估计器，并能对任意聚类相关量进行推断，同时具备严格的频率派收敛性保证。 |
| [^355] | [SEHFS: Structural Entropy-Guided High-Order Correlation Learning for Multi-View Multi-Label Feature Selection](https://arxiv.org/abs/2603.03022) | 提出SEHFS方法，通过将特征图转换为结构熵最小化的编码树来量化高阶依赖的信息代价，从而学习高阶特征相关性，解决了多视图多标签特征选择中现有信息论方法难以捕捉高阶结构相关性且易陷入局部最优的问题。 |
| [^356] | [Half-Truths Break Similarity-Based Retrieval](https://arxiv.org/abs/2602.23906) | 本文发现CLIP模型会被“半真半假”的错误描述欺骗（相似度反而升高），并提出CS-CLIP方法，通过将描述分解为实体和关系单元并构建最小编辑的干扰样本进行对比微调，从而增强模型对细节错误的识别能力。 |
| [^357] | [Topology enables learning-based hydrodynamic prediction of the global river system](https://arxiv.org/abs/2602.22293) | 本文提出GraphRiverCast，利用河流拓扑结构实现全球河系的每日水动力预测，无需初始状态，精度提升25%并泛化至未测量区域。 |
| [^358] | [Tabular Foundation Models Can Learn Association Rules](https://arxiv.org/abs/2602.14622) | 本文提出了一个模型无关的关联规则学习框架及其实现TabProbe，利用表格基础模型作为条件概率估计器，无需频繁项集挖掘即可从表格数据中学习关联规则，同时克服了经典方法的规则爆炸问题和神经网络方法在低数据量下的性能缺陷。 |
| [^359] | [Covariance-Aware Transformers for Quadratic Programming and Decision Making](https://arxiv.org/abs/2602.14506) | 本文从理论上证明Transformer可通过模拟梯度下降和软阈值迭代来求解无约束及ℓ1惩罚/约束的二次规划，并据此提出Time2Decide方法，通过向时间序列基础模型显式输入变量间协方差矩阵，显著提升投资组合优化等决策任务的表现。 |
| [^360] | [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models](https://arxiv.org/abs/2602.11590) | 该论文提出渐进式自我校正方法，通过复用去噪网络输出训练校正器，使掩码扩散模型能够在生成过程中迭代修正包括已生成词元在内的整个序列，从而避免误差累积并提升样本质量。 |
| [^361] | [A Jointly Efficient and Near-Optimal Algorithm for Heteroskedastic Generalized Linear Bandits with Adversarial Corruptions](https://arxiv.org/abs/2602.10971) | 本文提出将基于在线镜像下降的估计器与基于Hessian的置信度权重相结合的HCW-GLB-OMD算法，为带对抗性污染的异方差广义线性赌博机实现了联合高效且近最优的高概率遗憾保证。 |
| [^362] | [Efficient and Adaptable Detection of Malicious LLM Prompts via Bootstrap Aggregation](https://arxiv.org/abs/2602.08062) | 提出了BAGEL框架，通过自助聚合与专家混合启发的微调模型集成，以随机选择成员子集并可选配合轻量级路由器的方式，实现了高效、轻量且可增量更新的恶意LLM提示检测。 |
| [^363] | [Agents-as-Peers: Bias-Controlled LLM Evaluation as Multi-Agent Peer Assessment](https://arxiv.org/abs/2602.02589) | 该论文提出Agents-as-Peers分布式评估协议，让每个大语言模型智能体生成任务、回答共享任务集并互评同伴答案（排除自评），以解决固定基准老化和集中式评判者偏差问题，并对自我偏差、身份效应等评判偏差进行受控测量。 |
| [^364] | [TTCS: Test-Time Curriculum Synthesis for Self-Evolving](https://arxiv.org/abs/2601.22628) | TTCS提出了一种共同进化的测试时训练框架，通过问题合成器与推理求解器的迭代协同优化，生成难度递增的课程化问题变体，从而提升大语言模型在困难推理问题上的能力。 |
| [^365] | [Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling](https://arxiv.org/abs/2601.21669) | 该论文证明强化学习中结果层面的模式坍缩并非源于探索或正则化不足，而是期望回报目标本身的结构性必然结果，并提出用逆概率缩放的方法来修复这一问题。 |
| [^366] | [Optimal L2 Regularization in High-dimensional Continual Linear Regression](https://arxiv.org/abs/2601.13844) | 该论文在高维过参数化持续线性回归中推导出对任意线性教师均成立的期望泛化损失闭式表达式，并首次在理论持续学习中证明了最优L2正则化强度应随任务数T近似线性增长（为T/ln T）。 |
| [^367] | [Activation Sensitivity as a Unifying Principle for Post-Training Quantization](https://arxiv.org/abs/2601.11663) | 该论文证明AWQ、GPTQ、对角Fisher量化、块重构和Hadamard旋转等主流训练后量化方法都是最小化同一个由梯度加权激活敏感性支配的二次型损失的特殊情形，从而将这些方法统一在一个理论框架之下。 |
| [^368] | [Tensor computing application programming interface: An application-oriented, lightweight API for portable high-performance tensor network computations](https://arxiv.org/abs/2512.23917) | 提出了TCAPI——一个面向应用的轻量级编程接口，通过抽象张量对象和定义C++/Python间语义共享的精简核心函数集，实现了跨框架、跨硬件的可移植高性能张量网络计算。 |
| [^369] | [The Reward Model Selection Crisis in Personalized Alignment](https://arxiv.org/abs/2512.23067) | 该论文揭示传统的奖励模型准确性指标无法预测其在奖励引导解码中的实际表现，并提出“策略准确性”新指标，指出上游奖励模型准确性与下游策略准确性仅有微弱相关性（Kendall's tau = 0.08–0.31）。 |
| [^370] | [Robust Residual Correction via Selective Deployment for Time Series Forecasting](https://arxiv.org/abs/2512.22428) | CRC框架通过验证集校准的选择性部署规则，仅在校正能带来净误差降低时才应用混合残差校正，从而在时间序列预测中实现鲁棒的误差降低，尤其在复杂场景下避免了过拟合问题。 |
| [^371] | [OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction](https://arxiv.org/abs/2512.13886) | OPTIMA将掩码选择后的逐层权重重构表述为共享单个Hessian矩阵的按列二次规划问题，并借助加速器友好的QP求解器，在大语言模型上同时实现了剪枝的高精度与可扩展性。 |
| [^372] | [FLEXI-Haz: Semiparametric Deep Survival Inference Beyond Proportional Hazards](https://arxiv.org/abs/2512.10570) | 提出 FLEXI-Haz 半参数深度神经网络模型，在突破比例风险假设限制的同时保留可解释的回归参数，并针对右删失数据实现了极小化极大最优的收敛速率及渐近有效的半参数统计推断。 |
| [^373] | [When does Subgroup Balance Matter? Predicting Performance Sensitivity from Latent Separability](https://arxiv.org/abs/2512.09496) | 该论文提出“潜在分离假设”，指出微调模型对训练数据子群平衡的依赖程度由预训练模型潜在空间中子群间的分离程度决定，并通过理论分析与四个视觉及语言数据集上的实证和正则化实验验证了这一假设。 |
| [^374] | [KGOT: Unified Knowledge Graph and Optimal Transport Pseudo-Labeling for Molecule-Protein Interaction Prediction](https://arxiv.org/abs/2512.09365) | 该论文提出KGOT框架，通过构建统一知识图谱整合分子、蛋白质、基因和通路等多源生物数据，并利用最优传输方法为未标记的分子-蛋白质对生成高质量伪标签，从而同时解决标签稀缺和生物背景信息缺失两大挑战，提升分子-蛋白质相互作用预测性能。 |
| [^375] | [Magneton: Optimizing Energy Efficiency of ML Systems via Differential Energy Debugging](https://arxiv.org/abs/2512.08365) | 该论文提出“差分能量调试”新方法并实现了Magneton能量分析器，通过在算子级别比较相似机器学习系统之间的能量消耗，自动定位由不良软件设计造成的能量浪费，从而显著提升机器学习系统的能效。 |
| [^376] | [Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning](https://arxiv.org/abs/2511.15633) | 提出HASTEN框架，通过层次语义树锚定细粒度类别的语义结构，从而缓解基于CLIP的类增量学习中的特征漂移与灾难性遗忘。 |
| [^377] | [Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading](https://arxiv.org/abs/2511.14102) | Specter通过量化使能的多步投机解码与专家卸载相结合，在单GPU上实现了平均1.50倍（最高1.88倍）的MoE推理加速。 |
| [^378] | [Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network](https://arxiv.org/abs/2511.01553) | 提出CLP-SNN，一种将持续学习原型（CLP）算法重构为事件驱动神经动力学的协同设计脉冲神经网络，并作为首个在线持续学习器部署于Intel Loihi 2芯片，在低功耗约束下于OpenLORIS上取得与Replay和SLDA相当的性能。 |
| [^379] | [A Likely Geometry of Generative Models](https://arxiv.org/abs/2510.26266) | 提出了一种无需额外训练、兼容不同度量与概率分布的生成模型几何框架，通过约束在高密度数据区域的类测地线曲线和常微分方程组来刻画生成模型空间中的最短路径。 |
| [^380] | [Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station](https://arxiv.org/abs/2510.23463) | 该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。 |
| [^381] | [OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2510.13003) | OPLoRA通过双侧正交投影将LoRA更新约束在冻结权重前k个奇异子空间的正交补空间内，以数学保证的方式精确保留预训练知识，从而防止微调过程中的灾难性遗忘。 |
| [^382] | [Wasserstein Robust Classification Error: Surrogate Loss Misalignment and Distributional Attacks](https://arxiv.org/abs/2510.10000) | 本文证明了仅当损失函数是0-1损失的正倍数加标签相关偏移时，WDRO风险的排序才与鲁棒准确率的排序一致（交叉熵等常用损失存在失配问题），并提出Wasserstein分布攻击（WDA）来直接评估鲁棒准确率。 |
| [^383] | [Bayesian Nonparametric Dynamical Clustering of Time Series](https://arxiv.org/abs/2510.06919) | 提出一种基于层次狄利克雷过程与高斯过程先验的贝叶斯非参数方法，通过在未知数量的线性动态机制间切换来建模时间序列簇的演化，以原则性方式避免簇的无序增殖，并可通过变分推断实现离线与在线的高效学习，在心电图分析中验证了其有效性。 |
| [^384] | [Curriculum-Augmented GFlowNets For mRNA Sequence Generation](https://arxiv.org/abs/2510.03811) | 该论文提出课程增强生成流网络，利用学习进度驱动的自适应长度课程策略，在新型密码子设计环境中高效生成编码目标蛋白的mRNA序列，实现更优的帕累托前沿覆盖与多样性，且收敛速度最高提升4倍。 |
| [^385] | [Evaluating utility of clinical prediction models: A comparison of decision curve analysis and cost curves](https://arxiv.org/abs/2509.24608) | 本文比较了临床预测模型评估中的决策曲线分析与成本曲线两种方法，证明决策曲线与Brier曲线密切相关且两者间的参考线可相互转换，并提出上包络决策曲线可作为模型决策曲线的有用比较基准。 |
| [^386] | [Bellman Optimality of Average-Reward Robust Markov Decision Processes with a Constant Gain](https://arxiv.org/abs/2509.14203) | 本文通过研究常数增益设定下S-矩形不确定性对应的鲁棒贝尔曼方程解的存在性及其与最优平均奖励的关系，为平均奖励鲁棒马尔可夫决策过程建立了理论框架。 |
| [^387] | [Optimized Weight Initialization on the Stiefel Manifold for Deep ReLU Neural Networks](https://arxiv.org/abs/2509.00362) | 该论文通过在Stiefel流形上求解优化问题，提出了一种专为ReLU网络设计的半正交权重初始化方法，能够从初始化时刻起校准预激活统计特性、缓解死亡ReLU问题并稳定极深网络的训练。 |
| [^388] | [Blending data and physics for reduced-order modeling of systems with spatiotemporal chaotic dynamics](https://arxiv.org/abs/2507.21299) | 该论文提出一种融合数据与物理知识的混合降阶模型，通过在自编码器发现的不变流形上投影并修正全阶模型的向量场（或将其作为贝叶斯先验），在数据充足、稀缺甚至物理模型参数有误的情形下均优于纯数据驱动方法。 |
| [^389] | [GenoMAS: Guided Multi-Agent Programming for Scientific Discovery from Genomic Data](https://arxiv.org/abs/2507.21035) | GenoMAS提出了一种引导式多智能体编程框架，通过将结构化工作流与自主决策相结合，让编程智能体动态选择推进、修改、绕过或回溯动作单元，实现了从复杂基因组数据中可靠地进行科学发现。 |
| [^390] | [Gradient Descent on Logistic Regression: Do Large Step-Sizes Work with Data on the Sphere?](https://arxiv.org/abs/2507.11228) | 本文研究逻辑回归中梯度下降在低于稳定性阈值 $2/\lambda^*$ 的大步长下的收敛性，证明等范数数据在一维时可保证全局收敛，等范数数据在高维可能产生循环但球面均匀分布的总体损失可全局收敛，并将一维结论推广到更弱的环形条件。 |
| [^391] | [Double-Diffusion: Balancing Speed, Accuracy, and Uncertainty in Probabilistic Forecasting for Urban Sensor Networks](https://arxiv.org/abs/2506.23053) | Double-Diffusion通过将闭式图热传导先验嵌入去噪扩散模型，使反向过程从先验附近热启动而非纯噪声出发，从而在城市传感器网络概率预测中实现了速度、准确性与不确定性的平衡。 |
| [^392] | [LLMs are Brittle to Simple Code Transformations: Introducing CETBench - A Benchmark for Code-Equivalence Checking](https://arxiv.org/abs/2506.04019) | 本文提出了代码等价性检测基准CETBench，发现大语言模型即使面对简单的代码变换也会出现显著性能下降（在跨语言场景下尤为严重），并提出一种基于微调的简单方法来提升模型性能。 |
| [^393] | [Rethinking Generalization in Knowledge Graph Embedding: An Empirical Study and Sparse Peak Regularization](https://arxiv.org/abs/2505.23442) | 本文从泛化角度对知识图谱嵌入的正则化进行了跨模型族的系统性实证研究，发现过拟合在各类KGE模型中普遍存在，并提出稀疏峰值正则化方法来显著提升模型性能。 |
| [^394] | [Accelerating Optimization via Differentiable Stopping Time](https://arxiv.org/abs/2505.22509) | 本文提出了一种基于微分方程理论的“可微停止时间”概念及其高效反向传播算法，首次使“最小化达到目标损失所需时间”这一传统上不可微的对偶优化问题变为可微表述，从而为加速优化算法开辟了新框架，并可应用于在线超参数调优等场景。 |
| [^395] | [Federated Learning: From Theory to Practice](https://arxiv.org/abs/2505.19183) | 本书是一本联邦学习的实践入门指南，其核心贡献是提出通过联邦学习网络的加权边编码设备间相似性来实现个性化联邦学习，使每个设备在训练自己模型的同时仍能受益于与相关设备的协作。 |
| [^396] | [PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models](https://arxiv.org/abs/2503.18462) | 提出PALATE框架，通过巧妙运用全期望定律来克服特征似然散度（FLD）的计算效率瓶颈，从而更高效地全面评估深度生成模型生成样本的保真度、多样性和新颖性。 |
| [^397] | [Dissecting the Impact of Model Misspecification in Data-driven Optimization](https://arxiv.org/abs/2503.00626) | 本文从模型误设程度的角度比较了数据驱动优化中传统“先拟合后优化”方法与“估计-优化一体化”方法，证明了当模型存在误设时，一体化方法在遗憾值的前两个主导项上具有“普遍双重收益”的统计优势。 |
| [^398] | [Weight Space Learning for Certifiable Few-shot Transfer Learning](https://arxiv.org/abs/2502.06970) | 该论文提出了一种新型迁移学习方法，通过在上游任务上训练PEFT参数的扩散模型分布，再利用采样-评估程序为下游任务选择最优参数，从而即使在少样本情况下也能获得可认证的非空洞学习理论泛化保证。 |
| [^399] | [Feature Selection for Latent Factor Models](https://arxiv.org/abs/2412.10128) | 本文提出了一种基于低秩生成类模型和信噪比（SNR）准则的按类别分别进行特征选择的新方法，该方法具有理论保证的特征恢复能力，在标准分类数据集上优于部分现有方法，并可轻松应用于类增量学习场景。 |
| [^400] | [FiTv2: Scalable and Improved Flexible Vision Transformer for Diffusion Model](https://arxiv.org/abs/2410.13925) | 提出了FiTv2架构，通过将图像视为动态大小的token序列而非固定分辨率网格，使扩散模型能够生成无限制分辨率和宽高比的图像。 |
| [^401] | [Help Me Help You: The Aggregate Value of Source and Target Data in Transfer Learning](https://arxiv.org/abs/2408.16189) | 该论文揭示了监督迁移学习中的两种机制二分现象：弱迁移机制下最优收敛率仅由源数据或目标数据之一决定，而强迁移机制下需联合利用源与目标数据的互补信息才能实现快得多的目标收敛率。 |
| [^402] | [Deep learning-based multi-level airway collapse site localization on obstructive sleep apnea patients based on snoring sounds](https://arxiv.org/abs/2408.16030) | 本研究利用音频频谱图Transformer（AST）模型，仅通过鼾声即可有效定位阻塞性睡眠呼吸暂停患者软腭、口咽及腭后区水平的上气道塌陷部位，为无创诊断提供了新途径。 |
| [^403] | [Physics Inspired Neural Networks Enable Efficient Graph Coloring](https://arxiv.org/abs/2408.01503) | 该研究提出一种结合图神经网络与统计力学原理的物理启发神经框架，通过监督信号、对称性破缺正则化和噪声退火神经动力学，实现对大规模图着色问题的高效求解。 |
| [^404] | [Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/abs/2406.16232) | 该论文提出雅可比下降（JD）算法，通过梯度投影完全解决多目标间的冲突，并证明了在光滑凸情形下可收敛到帕累托前沿，进而实现了将每个训练样本损失视为独立目标的新型学习范式——实例级风险最小化（IWRM）。 |
| [^405] | [Simulating, Fast and Slow: Learning Policies for Black-Box Optimization](https://arxiv.org/abs/2406.04261) | 本文提出一种新方法，通过学习一个策略来引导在各种输入分布下的黑盒优化求解，避免了每次从头开始的昂贵模拟优化，从而大幅提升实验设计中优化的效率。 |
| [^406] | [Gibson Env: Real-World Perception for Embodied Agents](https://arxiv.org/abs/1808.10654) | 论文提出了基于真实空间虚拟化构建的Gibson仿真环境（涵盖572栋建筑、1400多个楼层空间），用于训练具身智能体的感知能力，并通过"Goggles"合成机制使仿真中训练的模型能够迁移部署到真实世界。 |
| [^407] | [Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/abs/1804.08328) | 该论文提出一种完全计算化的方法，通过发现26个视觉任务之间的迁移学习依赖关系，构建了首个任务迁移学习的计算分类图谱（Taskonomy），揭示了视觉任务空间的内在结构。 |
| [^408] | [On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers.](http://arxiv.org/abs/2310.14421) | 本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。 |

# 详细

[^1]: 关节、相机、开拍！为任意骨架绑定共同生成3D动作与视频

    Joints, Camera, Action! Co-Generating 3D Motion and Video for Arbitrary Rigs

    [https://arxiv.org/abs/2610.28632](https://arxiv.org/abs/2610.28632)

    该论文提出通过少量交叉注意力层将支持任意骨架的独立3D动作模型与预训练视频模型相连接，直接共同生成视频、可回放的3D动作以及将动作配准到3D场景的相机。

    

    视频生成模型如今已能描绘极为逼真的动作。然而，如何将这些动作用于像素之外的领域仍不明确：即为特定的3D资产制作动画，并在其3D场景中回放该动画。例如，视频模型可以展示一只猴子跳过横杆，但它无法将这个跳跃以可回放的3D动作形式返回并放置到3D场景中。本文提出一个模型，可为任意给定的骨架绑定直接共同生成视频与3D动作，同时生成一个将该动作配准到场景中的相机。先前的视频-动作模型要么仅能处理人类骨架（通常需增加数十亿参数），要么从生成的2D姿态图中三角化出3D动作。我们则将3D动作保留在一个支持任意骨架的独立模型中，并通过少量交叉注意力层将其与预训练的视频模型相连接。这种分离设计使动作模型可以先从仅含动作的数据中学习骨架如何运动，而这类数据更容易获得……

    arXiv:2610.28632v1 Announce Type: cross  Abstract: Video generation models can now depict remarkably realistic motion. Yet it is unclear how to use this motion beyond pixels: to animate a specific 3D asset and replay the animation in its 3D scene. For instance, a video model can show a monkey jumping over a crossbar, but it cannot return that jump as replayable 3D motion placed in the 3D scene. Here we present a model that directly co-generates the video and 3D motion for any supplied rig, along with a camera that registers this motion to the scene. Prior video-motion models either handle only human skeletons, often adding billions of parameters, or triangulate 3D motion from generated 2D pose maps. We instead keep 3D motion in a separate model that supports arbitrary skeletons, and connect it to a pretrained video model through a small set of cross-attention layers. This separation lets the motion model first learn how skeletons move from motion-only data, which is easier to obtain an
    
[^2]: 通过混合灵巧模仿学习可适应的灵巧技能

    Learning Adaptable Dexterous Skills via Hybrid Dexterous Mimicking

    [https://arxiv.org/abs/2610.28628](https://arxiv.org/abs/2610.28628)

    AdaDex提出混合灵巧模仿方法，通过在运动追踪环境和目标达成环境中联合训练共享策略，将几秒钟的手-物体轨迹转化为既保留人类先验又能泛化到任意物体姿态的可适应灵巧技能。

    

    灵巧多指手有望解锁人类设计的世界所依赖的多种技能，例如功能性抓取、手中重定向和动态交接，然而现有的学习方法迫使我们做出权衡：基于目标达成的强化学习虽然能泛化到任意物体姿态，但由于缺乏人类先验，会收敛到不自然且不可靠的手指策略；而运动追踪方法虽然能从单个片段中复现类人行为，却无法偏离该片段。我们提出了AdaDex，一个目标条件强化学习框架，它能将一段仅有几秒钟的手-物体轨迹——来自动作捕捉数据或从真实世界RGB(D)视频中进行人-物交互（HOI）重建——转化为可适应的灵巧技能。AdaDex的核心是混合灵巧模仿：一个共享策略在两类大规模并行环境群上进行训练：一类是使用密集运动追踪目标训练的运动追踪环境，用以从人类动

    arXiv:2610.28628v1 Announce Type: cross  Abstract: Dexterous multi-finger hands hold promise to unlock diverse skills like functional grasps, in-hand reorientation, and dynamic handover that the human-designed world assumes, yet existing learning recipes force a trade-off. Goal-reaching reinforcement learning generalizes to arbitrary object poses but, lacking any human prior, converges to unnatural and unreliable finger strategies; motion tracking reproduces human-like behavior from a single clip but cannot deviate from it. We present AdaDex, a goal-conditioned RL framework that turns a single hand-object trajectory of a few seconds---from mocap data or HOI reconstruction from in-the-wild RGB(D) video---into an adaptable dexterous skill. At the core of AdaDex is Hybrid Dexterous Mimicking: a shared policy is trained on two populations of massively parallel environments: motion tracking environments trained with dense motion tracking objective, which instill the human prior from human m
    
[^3]: 基于神经算子的噪声环境规划

    Planning under Noise with Neural Operators

    [https://arxiv.org/abs/2610.28623](https://arxiv.org/abs/2610.28623)

    本文提出PUN神经算子，通过在屏蔽泊松随机最优控制框架中为障碍物引入反射诺伊曼边界条件、为目标引入狄利克雷条件，并结合网格无关的蒙特卡洛求解器，学习噪声感知的价值场，从而在未见过的几何环境中实现噪声下的可靠规划。

    

    arXiv:2610.28623v1 公告类型：新论文 摘要：在不确定的控制执行噪声下进行规划，需要在多样化环境中具备可靠的反馈策略。我们提出了“噪声下规划”神经算子（Planning Under Noise, PUN），它能够学习噪声感知的价值场，用于在未见过的几何环境中进行规划。基于已有的随机最优控制屏蔽泊松公式，我们解决了两个挑战：指定支持在不同几何环境中导航的边界条件，以及高效计算具有可靠梯度的价值场。零狄利克雷障碍物边界会在墙壁处终止采样路径，削弱沿长路径的目标信号，从而降低估计的可靠性。我们转而在障碍物上施加反射诺伊曼条件，并在目标处施加狄利克雷条件，使采样路径能够继续探索自由空间直至到达目标。为了在无需构型空间网格的情况下求解所得到的偏微分方程，我们扩展了蒙特卡洛求解器 Walk on Star……

    arXiv:2610.28623v1 Announce Type: new  Abstract: Planning under uncertain control execution noise requires reliable feedback policies across diverse environments. We introduce Planning Under Noise (PUN), a neural operator that learns noise-aware value fields for planning in unseen geometries. Building on an established screened-Poisson formulation of stochastic optimal control, we address two challenges: specifying boundary conditions that support navigation across varying geometries and efficiently computing fields with reliable gradients. Zero-Dirichlet obstacle boundaries terminate sampled paths at walls, weakening the goal signal along long routes and making it harder to estimate reliably. We instead impose reflecting Neumann conditions on obstacles and a Dirichlet condition at the goal, allowing sampled paths to continue exploring free space until they reach the target. To solve the resulting PDE without a configuration-space mesh, we extend a Monte Carlo (MC) solver, Walk on Star
    
[^4]: 自适应多词元预测：我们应该走多远？

    Adaptive Multi-Token Prediction: How far should we go?

    [https://arxiv.org/abs/2610.28619](https://arxiv.org/abs/2610.28619)

    该论文提出自适应多词元预测（AMTP），将多词元预测的推测长度选择形式化为在线控制问题，通过轻量级控制器利用验证反馈动态调整最佳起草长度，无需重新训练即可提升语言模型解码效率。

    

    多词元预测（MTP）可以通过一次性起草多个词元，并在单次目标模型前向传播中对其进行验证，从而加速语言模型的解码。然而，最佳的推测长度会随上下文而变化：较长的草稿在可靠时能带来更大的摊销收益，但在被拒绝后可能会在被丢弃的后缀上浪费计算资源。与自回归起草方式（可以逐词元地决定停止规则）不同，MTP 必须在起草之前就确定长度，因为所有预测头都在同一次前向传播中激活。因此，我们将长度选择形式化为一个在线控制问题，利用验证反馈来学习在每个上下文中草稿可以被信任到何种程度，并提出了自适应多词元预测（AMTP）：一个轻量级的控制器，无需重新训练，且保持精确验证器不变。为了指导长度选择，我们通过接受生存率（acceptance survival），即草稿

    arXiv:2610.28619v1 Announce Type: new  Abstract: Multi-token prediction (MTP) can accelerate language-model decoding by drafting several tokens and verifying them in a single target-model pass. However, the best speculation horizon changes with context: longer drafts offer greater amortization when reliable, but can waste compute on suffixes discarded after a rejection. Unlike autoregressive drafting, where a stopping rule can decide token by token, MTP must fix the horizon before drafting, as all heads fire in the same forward pass. We therefore formulate horizon selection as an online control problem, using verification feedback to learn how far the draft can be trusted in each context, and introduce Adaptive Multi-Token Prediction (AMTP): a lightweight controller that requires no retraining and leaves the exact verifier unchanged. To guide horizon selection, we characterize the expected number of tokens committed at each horizon via acceptance survival, the probability that a draft 
    
[^5]: 面向图生成的高效边感知注意力网络

    Efficient Edge-aware Attention Network for Graph Generation

    [https://arxiv.org/abs/2610.28618](https://arxiv.org/abs/2610.28618)

    提出一种用于图生成的边感知注意力层，通过头级分数调制注入结构信息并以轻量残差机制更新边表示，可基于FlexAttention高效实现，在保持生成质量的同时将峰值GPU内存降低约65%并显著提升训练效率。

    

    图生成模型通常使用图特定的Transformer架构来联合更新节点和边特征。然而，这些架构无法直接受益于近期的硬件感知注意力实现，限制了其可扩展性。这种适配并非易事：边和图级特征必须注入成对注意力分数中，必须遵守图掩码，而动态边更新仍是二次方内存占用的主要来源。我们提出了一种用于图生成的边感知注意力层，该层通过头级分数调制融入结构信息，同时利用一种保留依赖于注意力分数的边更新的轻量级残差机制来更新边表示。该层可以借助PyTorch的FlexAttention高效实现。我们在取得与基线相近的生成质量的同时，将峰值GPU内存占用降低约65%，并大幅提升了训练（效率）……

    arXiv:2610.28618v1 Announce Type: new  Abstract: Graph generative models commonly use graph-specific Transformer architectures to jointly update node and edge features. However, these architectures do not directly benefit from recent hardware-aware attention implementations, limiting their scalability. The adaptation is, however, non-trivial: edge and graph-level features must be injected into pairwise attention scores, graph masks must be respected, and dynamic edge updates remain a major source of quadratic memory usage. We propose an edge-aware attention layer for graph generation that incorporates structural information through head-wise score modulation, while updating edge representations with a lightweight residual mechanism that retains attention-score dependent edge updates. This layer can be effectively implemented using PyTorch's FlexAttention. We achieve similar generation quality to our baseline while reducing peak GPU memory by around 65% and substantially improving train
    
[^6]: 将TS2Vec转变为用于时间序列分类的基础模型

    Turning TS2Vec into a Foundation Model for Time Series Classification

    [https://arxiv.org/abs/2610.28602](https://arxiv.org/abs/2610.28602)

    仅通过三项改动——在合成语料库上预训练、双视图输入分支以及多编码器知识蒸馏——就将原本需针对每个数据集单独训练的TS2Vec成功转变为可跨数据集复用、仅160万参数的时间序列分类基础模型TS2Vec-FM。

    

    TS2Vec是2022年提出的最强时间序列分类自监督方法之一。它最初的设计是在每个数据集上重新训练：先进行自监督预训练，随后进行线性探测。自那以后，该领域已经转向跨数据集复用单一编码器的基础模型范式。然而，TS2Vec具有紧凑的卷积架构，可以直接处理时间样本，并且无需重采样即可接受可变长度输入。我们探究当编码器仅预训练一次时，这些特性是否能够保留，并证明它们是可以保留的：TS2Vec本质上可以转变为一个基础模型。只需三项改动即可实现：在合成语料库上进行预训练；通过两个小型分支将原始幅值视图和实例归一化的形状视图输入编码器；将每个视图的专门编码器蒸馏为单一编码器。最终得到的TS2Vec-FM是一个160万参数的膨胀卷积网络……

    arXiv:2610.28602v1 Announce Type: cross  Abstract: TS2Vec is one of the strongest self-supervised methods for time series classification, introduced in 2022. It is designed to be trained anew on every dataset: self-supervised pre-training followed by linear probing. Since then, the field has shifted towards foundation models that reuse one encoder across datasets. Yet TS2Vec offers a compact convolutional architecture that operates directly on temporal samples and accepts variable-length inputs without resampling. We ask whether these properties can be retained when the encoder is pre-trained only once, and show that they can: TS2Vec can essentially be turned into a foundation model. Three changes suffice: pre-train on a synthetic corpus, feed the encoder a raw amplitude view and an instance-normalized shape view of the input through two small stems, and distill one specialist encoder per view into a single encoder. The result, TS2Vec-FM, is a 1.6M-parameter dilated convolutional netwo
    
[^7]: 基于神经隐式能量优化的激光雷达建图

    Neural Implicit Energy Optimization for LiDAR Mapping

    [https://arxiv.org/abs/2610.28598](https://arxiv.org/abs/2610.28598)

    该论文将神经激光雷达建图表述为神经隐式表示参数空间上的能量泛函优化，通过基于水平集形状算子（曲率）的能量泛函正则化符号距离场，将经典几何先验适配到百万点级大规模场景，在保持几何保真度的同时实现更平滑的重建。

    

    神经隐式表示为大规模激光雷达场景提供了内存高效的符号距离场（SDF）参数化，但从稀疏且常带噪声的输入中恢复的几何体存在抖动和伪影。经典的几何优化技术通过表面能量泛函来解决此类伪影，其极小值对应于光滑表面。受此原理启发，我们将神经激光雷达建图表述为在神经隐式表示参数空间上的能量泛函优化。具体而言，我们使用定义在符号距离场水平集形状算子上的能量泛函对其进行正则化，该算子表示表面的曲率，从而促进沿表面的平滑性。这将经典几何先验适配到百万点级激光雷达地图的规模。在大规模数据集上的实验表明，重建结果在保持几何保真度的同时具有更好的平滑性。

    arXiv:2610.28598v1 Announce Type: cross  Abstract: Neural implicit representations provide memory-efficient signed distance field (SDF) parameterizations for large-scale LiDAR scenes, but the geometry recovered from sparse and often noisy inputs carries jitter and artifacts. Classical geometric optimization techniques resolve such artifacts with surface energy functionals, whose minimizers correspond to smooth surfaces. Motivated by this principle, we formulate neural LiDAR mapping as the optimization of energy functionals over the parameter space of neural implicit representations. Specifically, we regularize the signed distance field with an energy functional defined on the shape operator of its level set, which represents the curvature of the surface, thus encouraging smoothness along the surface. This adapts classical geometric priors to the scale of million-point LiDAR maps. Experiments on large-scale datasets show reconstructions with improved smoothness at preserved geometric fi
    
[^8]: PaPi：面向快速神经谱架构的并行Picard-Duhamel积分方法

    PaPi: Parallel Picard-Duhamel Integration for Fast Neuro-Spectral Architectures

    [https://arxiv.org/abs/2610.28591](https://arxiv.org/abs/2610.28591)

    PaPi通过指数积分与Duhamel-Picard方案实现时间并行推理，证明了该迭代对所有流均呈指数收敛并给出显式误差界，从而将神经谱架构的训练与推理加速多达两个数量级，使低维PDE学习问题的训练时间缩短至一秒以内，同时保持有竞争力的预测精度。

    

    神经谱架构（NeuSA）是一类物理信息神经网络（PINN），已被证明在频域中求解时间依赖的偏微分方程（PDE）方面非常有效。然而，其训练和推理通常依赖于对学习到的向量场进行代价高昂的数值积分，一般采用顺序执行的Runge-Kutta时间步进方法，这限制了时间上的并行性。我们提出了PaPi，这是一种受弱非线性流启发的积分方案，通过指数积分和Duhamel-Picard方案实现了时间并行推理。我们证明了Duhamel-Picard迭代对所有流均呈指数收敛，并给出了取决于流与线性系统接近程度的显式误差界。实验表明，PaPi可将训练和推理加速多达两个数量级，对于低维PDE学习问题实现了不到一秒的训练时间，同时保持了具有竞争力的预测精度。

    arXiv:2610.28591v1 Announce Type: new  Abstract: Neuro-spectral architectures (NeuSA) are a class of physics-informed neural networks (PINNs) that have proven effective for solving time-dependent PDEs in the frequency domain. However, training and inference often rely on costly numerical integration of the learned vector field, typically using sequential Runge-Kutta time stepping, which limits temporal parallelism. We introduce PaPi, an integration scheme inspired by weakly nonlinear flows that enables time-Parallel inference through exponential integration and a Duhamel-Picard scheme. We show that the Duhamel-Picard iteration converges exponentially for all flows, and provide explicit error bounds that depend on how close the flow is to linear. Experiments show that PaPi accelerates training and inference by up to two orders of magnitude, achieving training times of under one second for low-dimensional PDE learning problems, while maintaining competitive predictive accuracy.
    
[^9]: WildMAE：面向野生动物图像重建的结构感知掩码自编码

    WildMAE: Structure-Aware Masked Autoencoding for Wildlife Image Reconstruction

    [https://arxiv.org/abs/2610.28590](https://arxiv.org/abs/2610.28590)

    WildMAE通过在MAE解码器中加入零门控卷积残差路径（提供二维邻域结构先验）与门控的掩码-可见交叉注意力机制，并结合渐进式的多目标训练策略，在75%掩码率下显著提升了野生动物相机陷阱图像的重建质量。

    

    相机陷阱（camera-trap）图像在动物外观、光照条件、背景杂乱程度和视觉退化方面存在显著变化。掩码自编码器（MAE）提供了一种无需标签的框架，可通过重建随机遮挡的图像块来对此类数据进行建模。我们提出了WildMAE，这是MAE在解码器端的一种扩展，将局部结构感知细化与掩码位置到可见位置的交叉注意力相结合。一条零门控卷积残差路径引入了二维邻域先验，而门控交叉注意力使掩码位置能够从可见位置特征中检索上下文信息，同时无需替换原始的Transformer解码器。与之配套的训练方案逐步引入归一化像素、Sobel边缘和感知等训练目标。在最初的Snapshot Serengeti研究中，涵盖4,744张测试图像、掩码率为75%的条件下，所报告的重建得分为27.90对24.33 dB PSNR，0.840对0.78

    arXiv:2610.28590v1 Announce Type: cross  Abstract: Camera-trap imagery presents substantial variation in animal appearance, illumination, background clutter, and visual degradation. Masked autoencoders (MAEs) offer a label-free framework for modeling such data through reconstruction of randomly hidden image patches. We present WildMAE, a decoder-side extension of MAE that combines local structure-aware refinement with masked-to-visible cross-attention. A zero-gated convolutional residual path introduces a two-dimensional neighborhood prior, while gated cross-attention allows masked positions to retrieve contextual information from visible-position features without replacing the original Transformer decoder. The associated training recipe progressively introduces normalized-pixel, Sobel-edge, and perceptual objectives. In the original Snapshot Serengeti study, covering 4,744 test images at 75% masking, the reported reconstruction scores were 27.90 versus 24.33 dB PSNR, 0.840 versus 0.78
    
[^10]: 两层 ReLU 层足矣

    Two $\operatorname{ReLU}$ Layers is All You Need

    [https://arxiv.org/abs/2610.28587](https://arxiv.org/abs/2610.28587)

    本文证明了任意深度的 ReLU 网络都可以被仅含两个隐藏层的 ReLU 网络精确模拟，并给出了基于同调代数填充和组合方法的两条证明路径。

    

    我们证明了每一个 ReLU 网络都可以被一个仅具有两个隐藏层的 ReLU 网络精确模拟。我们提出了两种证明方法：一种基于同调代数中的填充理论，另一种是直接的组合证明。

    arXiv:2610.28587v1 Announce Type: cross  Abstract: We prove that every $\operatorname{ReLU}$ network can be exactly simulated by a $\operatorname{ReLU}$ network with two hidden layers. We present two proofs: one that is based on fillings in homological algebra and a direct combinatorial proof.
    
[^11]: 基于离散数据的连续时间策略梯度：收敛分析与无模型算法

    Continuous-Time Policy Gradient from Discrete Data: Convergence Analysis and Model-Free Algorithms

    [https://arxiv.org/abs/2610.28580](https://arxiv.org/abs/2610.28580)

    该论文建立了一个从离散数据中学习的连续时间策略梯度框架，证明了熵正则化策略流向最优解的指数收敛性，并通过与PhiBE结合给出了收敛界显式分离评论家误差、策略回归误差和步长影响的无模型行动者-评论家算法。

    

    连续时间强化学习旨在由随机微分方程驱动的未知动力学下，利用离散时刻采集的观测数据寻求最优策略。我们开发了一个策略梯度框架，将严格的收敛分析与从离散数据中学习相结合。我们首先建立了性能差分引理，并推导出随机化策略和参数化策略的策略梯度公式。随后，我们证明了熵正则化策略流到正则化最优解的指数收敛性。对于离散参数化更新，我们推导出一个收敛界，显式地分离了评论家（critic）误差、策略回归误差和学习步长各自的贡献。为了在未知动力学下实现学习，我们将该框架与PhiBE相结合，后者能够从离散转移数据构建连续时间的代理生成器。由此得到的在策略与离策略无模型行动者-评论家（actor-critic）算法……

    arXiv:2610.28580v1 Announce Type: new  Abstract: Continuous-time reinforcement learning seeks an optimal policy under unknown dynamics driven by stochastic differential equations using observations collected at discrete times. We develop a policy gradient framework that combines rigorous convergence analysis with learning from discrete data. We first establish a performance difference lemma and derive policy gradient formulas for randomized and parametrized policies. We then prove exponential convergence of an entropy-regularized policy flow to the regularized optimum. For discrete parametrized updates, we derive a convergence bound that explicitly separates the contributions of critic error, policy-regression error, and the learning step size. To enable learning under unknown dynamics, we integrate this framework with PhiBE, which constructs a continuous-time surrogate generator from discrete transition data. The resulting on-policy and off-policy model-free actor-critic algorithms pr
    
[^12]: LLM解码的令牌-能量定律：活跃字节与实际时间共同解释每令牌的焦耳能耗

    The Token-Energy Law of LLM Decode: Active Bytes and Realized Time Account for Joules per Token

    [https://arxiv.org/abs/2610.28576](https://arxiv.org/abs/2610.28576)

    提出了令牌-能量定律，用活跃字节和实际解码时间两个维度以可校准的仿射形式精确刻画LLM解码中每令牌的GPU能耗，并解释了批处理摊销与KV历史增长对能耗的影响。

    

    大语言模型的高能效部署需要理解模型架构、硬件和服务配置如何影响每个生成令牌的能耗。我们提出了令牌-能量定律，这是一个紧凑且可解释的模型，用于描述密集、全历史解码模式下GPU板卡上每个输出令牌的能量消耗。该定律将令牌能耗与两个核心维度相关联：活跃字节和实际解码时间。活跃字节描述每个输出的逻辑字节需求，将解码迭代内共享的稠密权重与每个请求各自读取和更新的私有键值历史分离开来。实际解码时间是解码总耗时除以生成的输出令牌数量。我们将这两个字节坐标与时间以统一的仿射形式结合，其系数针对每个模型和硬件/运行时配置进行校准。该公式解释了批处理如何摊销共享权重的工作量，更长的注意力历史如何增加私有KV的需求。

    arXiv:2610.28576v1 Announce Type: new  Abstract: Energy-efficient deployment of large language models requires understanding how model architecture, hardware, and serving settings shape energy per generated token. We introduce the Token-Energy Law, a compact, interpretable model of GPU-board energy per output token for dense, full-history decoding. It relates token energy to two core dimensions: active bytes and realized decode time. Active bytes describe logical byte demand per output, separating dense weights shared within a decode iteration from private key--value (KV) histories read and updated for each request. Realized decode time is the elapsed decode duration divided by the number of output tokens produced. We combine the two byte coordinates and time in a common affine form, with coefficients calibrated for each model and hardware/runtime configuration. This formulation explains how batching amortizes shared-weight work, longer attended histories increase private KV demand, an
    
[^13]: 面向生成式检索的语义索引演化

    Evolving Semantic Indexes for Generative Retrieval

    [https://arxiv.org/abs/2610.28568](https://arxiv.org/abs/2610.28568)

    本文首次将LLM驱动的进化程序搜索应用于生成式检索，通过本地开源大模型自动变异语义标识符的各组件设计，并借助轻量代理解码器提供快速反馈，显著提升了检索性能。

    

    生成式检索（GR）将搜索与推荐任务转化为对离散语义标识符（SID）的自回归生成。SID的设计在很大程度上决定了GR的质量，但其离散且相互依赖的各个组件（SID构建、冲突处理、解码和训练）无法通过梯度进行端到端优化，目前主要依赖手工设计。我们提出了一个框架，通过LLM驱动的进化程序搜索来改进这些组件，据我们所知，这是该方法在GR领域的首次应用。一个本地部署的开源权重31B模型负责对可执行的GR组件进行变异，因此无需使用专有的LLM API；同时，一个100万参数的代理解码器提供快速反馈，替代对10亿参数目标生成器的反复训练。在NQ320K数据集上对21个SID程序的实验中，代理模型与目标模型的Recall@100表现出很强的排序相关性（Spearman ρ = 0.90）。在该代理的引导下，进化搜索在NQ320K、MSMARCO300K以及多个

    arXiv:2610.28568v1 Announce Type: new  Abstract: Generative retrieval (GR) casts search and recommendation as autoregressive generation of discrete semantic identifiers (SIDs). SID design largely determines GR quality, but its discrete, interdependent components (SID construction, collision handling, decoding, and training) cannot be optimized end-to-end with gradients and remain largely hand-designed. We introduce a framework that improves these components through LLM-driven evolutionary program search, to our knowledge the first application of this approach to GR. A locally served, open-weight 31B model mutates executable GR components, so no proprietary LLM API is used, and a 1M-parameter proxy decoder provides fast feedback in place of repeatedly training the 1B target generator. Across 21 SID programs on NQ320K, proxy and target Recall@100 show strong rank correlation (Spearman $\rho = 0.90$). Guided by this proxy, evolution improves the seed GR on NQ320K, MSMARCO300K, and a multi
    
[^14]: 通过残差空间适配实现扩散模型中的持续概念个性化

    Continual Concept Personalization in Diffusion Models via Residual-Space Adaptation

    [https://arxiv.org/abs/2610.28567](https://arxiv.org/abs/2610.28567)

    该论文提出将残差空间适配扩展到预训练扩散模型的随机适配过程中，实现了持续概念个性化，在保持近乎零遗忘的同时达到了所有竞争方法中最高的概念保真度。

    

    文本到图像个性化旨在使预训练的扩散模型具备仅从少量参考图像中学习用户特定概念的能力。然而，现有方法通常独立训练每个概念，或假设所有概念同时可用。在实际应用中，概念往往按顺序到达，这要求模型在持续学习新主体、物体或风格的同时，尽量减少对已获取概念的干扰。我们通过将残差空间适配扩展到预训练扩散模型的随机适配过程，来解决这一持续概念个性化问题。我们的工作在各类竞争性持续个性化方法中实现了最高的概念保真度，同时保持近乎零的遗忘。除了这些实证收益之外，学习到的残差空间动态还揭示了适配容量在扩散模型内部的集中位置。限制适配（摘要在此处被截断）

    arXiv:2610.28567v1 Announce Type: cross  Abstract: Text-to-image personalization aims to equip pretrained diffusion models with the ability to learn user-specific concepts from only a few reference images. Existing methods, however, typically train each concept independently or assume that all concepts are available simultaneously. In practical applications, concepts often arrive sequentially, requiring models to continually learn new subjects, objects, or styles while minimizing interference with previously acquired concepts. We address this continual concept personalization setting by extending residual-space adaptation to the stochastic adaptation process of pretrained diffusion models. Our work achieves the highest concept fidelity among competing continual personalization methods while maintaining near-zero forgetting. Beyond these empirical gains, the learned residual-space dynamics reveal where adaptation capacity is concentrated within the diffusion model. Restricting adaptatio
    
[^15]: 聚类联邦学习中崩溃与恢复的动力学

    The Dynamics of Collapse and Recovery in Clustered Federated Learning

    [https://arxiv.org/abs/2610.28565](https://arxiv.org/abs/2610.28565)

    该论文将聚类联邦学习中的聚类崩溃形式化为未知动态系统的控制问题，识别出贪婪路由导致崩溃自我强化的初始化区域，并通过有限时域误差动力学证明路由干预可通过反复训练被忽视的模型来逆转崩溃。

    

    聚类联邦学习（CFL）可能会遭受聚类崩溃问题。特别地，在不利的初始化条件下，早期的客户端分配会将异构群体集中到同一个聚类模型上，该模型随后被训练为一个融合的折中方案，而另一个模型则接收不到任何更新。由于当前模型决定了客户端的路由，而路由又决定了后续的模型更新，我们将聚类联邦学习形式化为对一个未知动态系统的控制问题：聚类模型作为演化的状态，客户端路由模式作为控制输入，路由干预作为依赖于状态的控制器。在一个双组线性模型中，我们识别出一个初始化区域，在该区域内贪婪路由会使崩溃具有自我强化性。随后，我们推导出有限时域误差动力学，展示了路由干预如何通过反复训练“饥饿”模型（主要使用来自其相容组的更新）来逆转崩溃。受此启发……

    arXiv:2610.28565v1 Announce Type: new  Abstract: Clustered federated learning (CFL) can suffer from cluster collapse. In particular, under unfavorable initialization, early client assignments concentrate heterogeneous groups on one cluster model, which is then trained toward a pooled compromise, while another model receives no updates. Since the current models determine client routing and routing determines subsequent model updates, we formulate clustered federated learning as control of an unknown dynamical system, with cluster models as the evolving state, client routing patterns as the control input, and routing interventions as state dependent controllers. In a two group linear model, we identify an initialization region in which greedy routing makes collapse self reinforcing. We then derive finite horizon error dynamics showing how routing interventions can reverse collapse by repeatedly training the starved model, primarily with updates from its compatible group. Motivated by thi
    
[^16]: 谷歌搜索与COVID-19时间序列中的不对称双向预测动态：一项多预测期预测误差分析

    Asymmetric Bidirectional Predictive Dynamics in Google Search and COVID-19 Time Series: A Multi-Horizon Forecast-Error Analysis

    [https://arxiv.org/abs/2610.28563](https://arxiv.org/abs/2610.28563)

    该研究提出了一种结合对数RMSE增益与预测误差分布熵（FEDE）及持久性加权DPRS评分的多预测期预测误差分析框架，揭示了谷歌搜索与COVID-19时间序列之间双向预测关系的非对称动态特征。

    

    在线搜索活动被广泛用作疫情期间信息寻求行为的数字痕迹，但其预测价值往往被简化为相关性、某个选定的滞后阶数或单一的预测误差分数。我们在双向方向上检验了八个预先指定的谷歌趋势-COVID-19数据对，采用匹配的LightGBM预测模型，在滚动400天窗口和1-14天的预测期上进行分析。我们将对数RMSE增益与预测误差分布熵（FEDE）相结合，后者能够捕捉带符号误差分布随预测期变化的情况。由此得到的方向性预测特征包括响应类别、RMSE效应、FEDE端点与早期斜率响应、在滚动机制间的持续性以及预测期轮廓。为筛选候选方向，我们还引入了相对持久性加权的方向性预测响应强度（DPRS），采用研究内0-100的量表；100表示比较集中观察到的最强响应，而非绝对意义上的

    arXiv:2610.28563v1 Announce Type: new  Abstract: Online search activity is widely used as a digital trace of information seeking during epidemics, but its predictive value is often reduced to correlation, a selected lag, or a single forecast-error score. We examine eight pre-specified Google Trends-COVID-19 pairs in both directions using matched LightGBM forecasts over rolling 400-day windows and horizons of 1-14 days. Log-RMSE gain is combined with forecast-error distribution entropy (FEDE), which captures horizon-dependent changes in the signed error distribution. The resulting directional predictive signature comprises response class, RMSE effect, FEDE endpoint and early-slope responses, persistence across rolling regimes, and horizon profile. For screening candidate directions, we also introduce a relative persistence-weighted directional predictive response strength (DPRS) on a within-study 0-100 scale; 100 denotes the strongest observed response in the comparison set, not an abso
    
[^17]: OptScientist：优化定律的自动化发现

    OptScientist: Automated Discovery of Optimization Laws

    [https://arxiv.org/abs/2610.28562](https://arxiv.org/abs/2610.28562)

    提出了 OptScientist 框架，将符号优化问题推断模型与 LLM 智能体系统相结合，能够直接从数据中自动发现简洁的符号优化定律，并在包含 99 个物理优化问题的新型基准 OptFeynman 上对超过 80% 的问题成功恢复了真实定律。

    

    许多科学现象最好被理解为底层优化问题的结果，然而自动化科学发现主要集中于恢复描述观测到的输入-输出模式的方程。我们提出了 OptScientist，这是一个新颖的自动化框架，用于直接从数据中发现简洁的符号优化定律。OptScientist 包含两个部分：(i) 一个推断符号优化问题的模型，它首先学习将数据表示为优化问题，然后提取其符号形式；(ii) 一个 LLM 智能体系统，将该模型嵌入到一个发现流程中，该流程整合来自科学文献的理论先验、发现候选定律、抽象其底层机制，并建议收集新的数据。我们首先在 OptFeynman 上评估 OptScientist，这是一个包含 99 个源自物理学的优化问题的新型基准，OptScientist 在超过 80% 的问题上成功恢复了真实定律，并且……

    arXiv:2610.28562v1 Announce Type: new  Abstract: Many scientific phenomena are best understood as outcomes of underlying optimization problems, yet automated scientific discovery has largely focused on recovering equations that describe observed input-output patterns. We introduce OptScientist, a novel automated framework for discovering compact symbolic optimization laws directly from data. OptScientist comprises (i) a model that infers symbolic optimization problems by first learning to represent data as an optimization problem and then extracting its symbolic form, and (ii) an LLM agentic system embedding the model in a discovery process that integrates theoretical priors from scientific literature, discovers candidate laws, abstracts their underlying mechanisms, and suggests new data to collect. We first evaluate OptScientist on OptFeynman, a novel benchmark of 99 physics-derived optimization problems, where OptScientist recovers ground-truth laws for over 80% of the problems and r
    
[^18]: 学习线性神经网络的随机梯度下降收敛性

    Convergence of stochastic gradient descent for learning linear neural networks

    [https://arxiv.org/abs/2610.28561](https://arxiv.org/abs/2610.28561)

    本文首次为深度线性神经网络中的随机梯度下降提供了几乎必然收敛保证，证明在温和条件下（无需Lipschitz性质或白化数据）SGD可收敛到临界点集合。

    

    随机梯度下降（SGD）是现代深度学习的基石，但其在非凸设置下的理论收敛性质仍然知之甚少。为填补这一空白，我们研究了深度线性神经网络中的SGD动力学——这是一种保留了关键非凸特性同时在分析上仍可处理的简化模型。基于最近关于梯度下降的结果，我们将收敛保证扩展到了正则化平方损失下的SGD。我们建立了SGD迭代序列的几乎必然有界性。利用随机逼近理论中的渐近伪轨迹框架，我们的分析进一步在温和条件下（不依赖于Lipschitz性质或白化数据），针对确定性和自适应两种步长选择，建立了SGD收敛到临界点集合的结果。据我们所知，这是首个为深度线性网络中的SGD提供几乎必然收敛保证的工作。

    arXiv:2610.28561v1 Announce Type: new  Abstract: Stochastic gradient descent (SGD) is the cornerstone of modern deep learning, yet its theoretical convergence properties remain poorly understood in non-convex settings. To address this gap, we study SGD dynamics in deep linear neural networks - simplified models that retain key non-convex characteristics while remaining analytically tractable. Building on recent results for gradient descent, we extend convergence guarantees to SGD under regularized squared loss. We establish almost sure boundedness of SGD iterates. Using the framework of asymptotic pseudotrajectories from stochastic approximation theory, our analysis establishes then, for both deterministic and adaptive step-size choices, convergence of SGD to the set of critical points under mild conditions (without relying on Lipschitz properties or whitened data). To our knowledge, this is the first work providing almost sure convergence guarantees for SGD in deep linear networks, br
    
[^19]: 大语言模型何时有助于特征选择？一项系统性实证研究

    When Do LLMs Help with Feature Selection? A Systematic Empirical Study

    [https://arxiv.org/abs/2610.28556](https://arxiv.org/abs/2610.28556)

    这项系统性实证研究表明，在表格数据特征选择任务中，经典统计方法在预测性能上普遍优于被测试的7B规模大语言模型选择器及经GRPO强化学习适配的模型。

    

    大语言模型（LLM）能够利用语义化的特征名称结合汇总统计信息进行表格数据的特征选择，然而此类信息相对于成熟的统计方法何时能够改善特征子集的选择仍不清楚。我们在一个包含18个数据集的实验框架中研究这一问题，其中包括10个用于群体相对策略优化（GRPO）训练的数据集、2个用于检查点验证的数据集，以及6个留出的评估数据集。我们比较了五种经典基线方法、冻结的Qwen2.5-7B与Llama-3.3-70B选择器，以及两个经GRPO适配的Qwen2.5-7B模型，分别报告分类任务的ROC-AUC和回归任务的R²，并使用数据集层面的排名进行跨任务比较。在五个固定随机种子的评估划分中，经典选择器通常比被测试的7B LLM/RL选择器取得更高的预测性能，即便在预先设定的约33%特征预算以及严格仅用训练数据拟合经典方法的条件下也是如此，而7B选择器……（原文摘要在此处截断）

    arXiv:2610.28556v1 Announce Type: cross  Abstract: Large language models (LLMs) can leverage semantic feature names alongside summary statistics for tabular feature selection, yet it remains unclear when such information improves subset selection relative to established statistical methods. We investigate this question within an 18-dataset framework comprising 10 datasets for Group Relative Policy Optimization (GRPO) training, 2 for checkpoint validation, and 6 held-out evaluation datasets. We compare five classical baselines, frozen Qwen2.5-7B and Llama-3.3-70B selectors, and two GRPO-adapted Qwen2.5-7B models, reporting classification ROC-AUC and regression R^2 separately and using dataset-wise ranks for cross-task comparison. Across five seeded evaluation partitions, classical selectors generally achieve higher predictive performance than the tested 7B LLM/RL selectors, including under a pre-specified ~33% feature budget with strict train-only classical fitting, while the 7B selecto
    
[^20]: 学习计算结构以实现样本高效的预测

    Learning Computational Structure for Sample-Efficient Prediction

    [https://arxiv.org/abs/2610.28555](https://arxiv.org/abs/2610.28555)

    受生物神经回路空间组织启发，该论文提出 CSFRBench 基准和 Learned Wiring（LeWi）方法——让每个神经元学习低维嵌入并仅与邻近神经元连接——以显式利用函数的组合稀疏结构，从而显著提升高维预测的样本效率。

    

    高维统计学习饱受维数灾难的困扰。幸运的是，许多具有实际意义的函数是组合稀疏的，也就是说它们可以表示为若干局部函数的组合，每个局部函数仅依赖于少数几个变量，这使得样本复杂度取决于局部元数而非环境维度。然而，要真正获得这一好处，需要利用隐藏的组合结构，而现有理论并未针对实际基于梯度的训练方法确立这一点。我们通过 CSFRBench 来研究这一差距——这是一个受控基准，它在训练阶段隐藏目标函数的组合计算方式，但在诊断阶段保留该结构——并发现稠密网络以及基于稀疏性的基线方法都未能充分实现这一潜在收益。受生物神经回路空间组织方式的启发，我们提出了 Learned Wiring（LeWi，学习式布线）：每个神经元拥有一个可学习的低维嵌入，只有彼此邻近的神经元才会相互连接，并且嵌入与权重……（摘要原文在此处截断）

    arXiv:2610.28555v1 Announce Type: cross  Abstract: Statistical learning in high dimensions suffers from the curse of dimensionality. Fortunately, many functions of practical interest are compositionally sparse, meaning they admit a composition of local functions, each depending on only a few variables, allowing sample complexity to depend on local arity rather than ambient dimension. Yet realizing this benefit requires exploiting hidden compositional structure, which theory does not establish for practical gradient-based training. We study this gap with CSFRBench, a controlled benchmark that hides the target's compositional computation during training but retains it for diagnosis, and find that dense networks and sparsity-based baselines leave much of that benefit unrealized. Inspired by the spatial organization of biological neural circuits, we introduce Learned Wiring (LeWi): each neuron has a learned low-dimensional embedding, only nearby neurons connect, and embeddings and weights 
    
[^21]: 给自己打折：面向客户端中断场景的去中心化学习自包含新鲜度加权Gossip算法

    Discount Yourself: Self-inclusive Freshness-weighted Gossip for Decentralised Learning under Client Outages

    [https://arxiv.org/abs/2610.28545](https://arxiv.org/abs/2610.28545)

    该论文提出，在去中心化联邦学习中，客户端从离线故障恢复时，应将自身模型与邻居模型一样按相对陈旧度进行折扣加权，并以继承滞后而非年龄来度量陈旧度，从而使恢复误差保持有界。

    

    去中心化联邦学习（DFL）让设备在本地训练个性化模型，并与邻居模型进行平均，而非依赖中心服务器。然而，真实设备常常离线数小时甚至数天，从故障中恢复的客户端只能通过聚合其邻居所持有的模型来恢复。我们观察到，这些模型中最陈旧的往往是客户端自己的模型，而现有的陈旧性感知规则仅对客户端接收到的模型进行折扣，并未解决这一情况。我们将故障后的聚合形式化为对完全可用网络状态的估计，并推导出三条原则。第一，接收者必须像任何邻居一样被折扣：任何将自身权重设为不低于每个邻居的规则，其误差都会随故障时长无界增长；而按相对陈旧度对所有候选模型进行折扣则能保持误差有界。第二，陈旧度必须以继承滞后（inherited lag）而非年龄（age）来度量，因为……（摘要原文在此处截断）

    arXiv:2610.28545v1 Announce Type: new  Abstract: Decentralised federated learning (DFL) trains personalised models on devices that average with their neighbours rather than with a central server. Real devices, however, are often offline for hours or days, and a client returning from an outage can recover only by aggregating the models its neighbourhood holds. We observe that the stalest of these models is usually the client's own, a case that existing staleness-aware rules, which discount only the models a client receives, do not address. We formalise aggregation after an outage as estimating the state of a fully available network and derive three principles. First, the receiver must be discounted like any neighbour. Every rule that weights it at least as much as each neighbour has an error that grows without bound with the outage, whereas discounting all candidates by relative staleness keeps the error bounded. Second, staleness must be measured as inherited lag rather than age, since
    
[^22]: 二元离散扩散：面向逻辑与规划任务的候选集扩散

    Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks

    [https://arxiv.org/abs/2610.28543](https://arxiv.org/abs/2610.28543)

    该论文提出二元离散扩散，将扩散过程的含噪状态表示为每个位置候选词元集合的二进制隶属向量，通过添加与删除候选实现信息的渐进丢失与恢复，从而能显式建模部分不确定性并有效解决数独、规划等逻辑与推理任务。

    

    离散扩散模型通过迭代去噪由文本等分类数据构成的含噪序列，实现任意顺序的生成，并具备自校正与填充能力，这使其成为推理、规划以及代码、数学等受约束生成任务的有前景的候选方案。然而，掩码扩散和均匀扩散会在单次腐蚀事件中抹去词元的身份信息，因此其含噪状态无法显式表示那种多个备选词元仍然合理的部分解析不确定性。我们提出二元离散扩散来填补这一空白：这是一族连续时间离散扩散过程，其含噪状态是每个位置上的一组候选词元，以二进制隶属向量表示。二元扩散通过添加和删除操作来改变候选集，使信息能够逐渐丢失并逐渐恢复，同时扩散状态本身就表示部分不确定性。在Sudoku-Extreme、Boxoban和gr

    arXiv:2610.28543v1 Announce Type: cross  Abstract: Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token's identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and gr
    
[^23]: GraphCast-Mamba：学习天气预报的非马尔可夫修正

    GraphCast-Mamba: Learning Non-Markovian Corrections for Weather Forecasting

    [https://arxiv.org/abs/2610.28512](https://arxiv.org/abs/2610.28512)

    GraphCast-Mamba在冻结的GraphCast主干网络上加入结合图消息传递与Mamba层的可训练修正分支，通过跨预报步骤的时间记忆学习超出两帧输入的非马尔可夫修正，使十天预报整体损失降低14.42%，第10天损失降低21.66%。

    

    未被解析的大气过程可能会引入超出许多天气预报模型所使用的短输入窗口的历史依赖性。我们提出了GraphCast-Mamba，它在冻结的预训练1°分辨率GraphCast主干网络上增加了一个可训练的Mamba修正分支，该分支结合了图消息传递和Mamba层。该分支在连续的六小时预报步骤之间保留时间记忆状态，学习基于超出GraphCast两帧输入的时间上下文的修正。训练仅限于修正分支，使用2015至2021年的ERA5数据。经过包括预热在内的2500次优化器更新后，所选模型相对于未经修正的GraphCast主干网络取得了显著改进。在整个十天预报的汇总中，该模型将GraphCast损失（归一化加权均方误差，MSE）降低了14.42%，在第10天（预报初始化后240小时）降低了21.66%。在这一最终预报时效上……

    arXiv:2610.28512v1 Announce Type: new  Abstract: Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen, pretrained $1^\circ$ GraphCast backbone with a trainable Mamba correction branch combining graph message passing and Mamba layers. The branch retains a temporal memory state across successive six-hour forecast steps, learning corrections informed by temporal context beyond GraphCast's two-frame input. Training is confined to the correction branch, using ERA5 data from 2015--2021. After 2,500 optimizer updates, including warm-up, the selected model achieves substantial improvements relative to the uncorrected GraphCast backbone. It reduces the GraphCast loss (normalized weighted mean-squared error; MSE) by 14.42% when aggregated over the full ten-day forecast and by 21.66% at day 10 (240 h after forecast initialization). At this final forecast lead
    
[^24]: 非凸优化中带延迟反馈的方差缩减

    Variance Reduction with Delayed Feedback for Non-Convex Optimization

    [https://arxiv.org/abs/2610.28510](https://arxiv.org/abs/2610.28510)

    该论文针对梯度评估存在延迟的非凸优化问题，提出了基于有序递推动量估计器的方差缩减方法及其异步快照变体，并在延迟上界或缺失反馈加权总量上界的条件下给出了改进的预言机复杂度。

    

    我们研究了梯度评估延迟返回时非凸优化的方差缩减问题。服务器可以在先前的评估尚未完成时继续更新参数。我们首先分析了用于随机优化的有序递推动量估计器。在平均光滑性和有界方差的条件下，其预言机复杂度为 $O(1+\sigma^2\epsilon^{-2}+L\Delta_f(1+\sigma/\epsilon+\tau)\epsilon^{-2})$，其中目标是将期望梯度范数平方降至至多 $\epsilon^2$，$\tau$ 为反馈延迟的上界。接下来我们研究了具有 $n$ 个分量的有限和问题，得到了 $O(n+L\Delta_f(\sqrt n+\min\{\tau,n-1\})\epsilon^{-2})$ 次评估的复杂度。我们还提出了一种异步快照变体，在全梯度计算的完成时间满足显式上界的条件下，可在全梯度计算期间继续更新参数。对于随机方法，我们进一步用缺失反馈的加权总量上界替代了最大延迟上界。

    arXiv:2610.28510v1 Announce Type: cross  Abstract: We study variance reduction for non-convex optimization when gradient evaluations return with a delay. The server can update the parameter while earlier evaluations remain unfinished. We first analyze an ordered recursive momentum estimator for stochastic optimization. Under average smoothness and bounded variance, its oracle complexity is $O(1+\sigma^2\epsilon^{-2}+L\Delta_f(1+\sigma/\epsilon+\tau)\epsilon^{-2})$, where the target is an expected squared gradient norm of at most $\epsilon^2$ and $\tau$ bounds the feedback delay. We next study finite sums with $n$ components and obtain $O(n+L\Delta_f(\sqrt n+\min\{\tau,n-1\})\epsilon^{-2})$ evaluations. An asynchronous snapshot variant continues updating during full-gradient computations under an explicit bound on their completion time. For the stochastic method, we further replace a maximum-delay bound by a bound on the weighted amount of missing feedback.
    
[^25]: 心电图用于实验室检测值：一项全面的生理学与技术分析

    ECG for laboratory values:A comprehensive physiological and technical analysis

    [https://arxiv.org/abs/2610.28507](https://arxiv.org/abs/2610.28507)

    本论文提出了一种多模态深度学习框架，利用心电图记录和患者元数据预测NT-proBNP等实验室检测值，验证了心电图预测的生理学合理性，并发现12导联配置的性能始终优于单导联配置。

    

    实验室检测对临床决策至关重要，但通常需要进行有创的血液采样。近年来的研究表明，心电图记录可能包含超越心律本身的信息，并能提供与实验室检测值相关的信息。本论文使用多模态深度学习框架，从心电图记录和患者元数据中预测NT-proBNP，同时考虑了基于心电图的预测在生理学上的合理性。论文还比较了使用II导联的单导联配置与相应的12导联配置。附录中还展示了关于血糖、血红蛋白和钾离子的补充实验。研究基于MIMIC-IV数据集进行了回归、二分类和三分类实验，其中三分类任务基于临床确立的阈值。12导联配置始终取得了更好的性能。回归任务达到了……的平均绝对误差（原文此处截断）。

    arXiv:2610.28507v1 Announce Type: cross  Abstract: Laboratory measurements are essential for clinical decision making but typically require invasive blood sampling. Recent research suggests that ECG recordings may contain information beyond cardiac rhythm and provide information related to laboratory values. This thesis investigates NT-proBNP prediction from ECG recordings and patient metadata using a multimodal deep learning framework, while considering the physiological plausibility of ECG-based prediction. It also compares a 1-lead configuration using lead II with the corresponding 12-lead configuration. Additional experiments on glucose, hemoglobin, and potassium are presented in the appendix.   Regression, binary classification, and three-class classification experiments were conducted using MIMIC-IV, with the three-class task based on clinically established thresholds. The 12-lead configuration consistently achieved better performance. Regression reached a mean absolute error of 
    
[^26]: BitMTP：当多词元预测遇上低比特大语言模型

    BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models

    [https://arxiv.org/abs/2610.28500](https://arxiv.org/abs/2610.28500)

    提出BitMTP算法-系统协同设计框架，将多词元预测与超低比特大语言模型骨干相结合，使单一模型兼具自回归生成与块因果草稿生成能力，在保持同等生成质量的前提下实现高效解码加速。

    

    摘要（arXiv:2610.28500v1，公告类型：cross）：为了实现大语言模型（LLM）的高效部署，超低比特骨干网络方法可以降低权重存储、内存流量和线性层执行成本，而多词元预测（MTP）方法通过并行预测多个未来词元来减少顺序解码步骤。尽管这两种方法互补，但直接结合它们无法在同等质量下加速解码：掩码草稿生成缺少下一个词元训练中所提供的中间词元输入，同时更宽的计算和因果键值（KV）更新可能会抵消更少解码步骤带来的节省。为了解决这些挑战，我们提出BitMTP，一个用于在超低比特LLM骨干网络上实现高效MTP的算法-系统协同设计框架。通过联合下一个词元预测与MTP的训练目标，单个模型同时具备自回归（AR）生成和块因果草稿生成能力。仅置信度解码、仅投机解码和混合解码提供了不同的……（原文摘要在此处截断）

    arXiv:2610.28500v1 Announce Type: cross  Abstract: For efficient deployment of large language models (LLMs), methods of ultra-low-bit backbones reduce weight storage, memory traffic, and linear-layer execution cost, while multi-token prediction (MTP) methods reduce sequential decoding steps by predicting multiple future tokens in parallel. Although complementary, the direct combination of these methods cannot accelerate decoding at comparable quality: masked drafting lacks intermediate token input supplied in next-token training, while wider computation and causal key-value (KV) updates can offset the savings from fewer steps. To address these challenges, we propose BitMTP, an algorithm-system co-design framework for efficient MTP on ultra-low-bit LLM backbones. A joint next-token prediction and MTP objective equips a single model with both autoregressive (AR) generation and block-causal drafting. Confidence-only decoding, speculative-only decoding, and hybrid decoding provide differen
    
[^27]: 存在但难以读取：用最小提示对探查潜在推理

    Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs

    [https://arxiv.org/abs/2610.28498](https://arxiv.org/abs/2610.28498)

    论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。

    

    潜在推理的探查研究通常在答案开始生成的位置读取模型状态，并将探查失败当作该信息不存在的证据。我们用一个从问题中复制的数字来检验这一点，实验对象是循环模型 Ouro-1.4B 和标准 Transformer Qwen2.5-1.5B。比较仅在该数字上不同的提示对可以发现，该数字确实存在于最后一个提示词元中，并以一种跨问题共享的编码形式存在（读取准确率约 0.9）。而一次只看单个提示的常规探针几乎读不出它：即使训练了 5,249 个问题，Ouro 中仅达 0.36，Qwen 中仅达 0.19。该数字引起的状态变化是“微小”而非“拥挤”的：随机噪声会落在同样繁忙的方向上，而用一个随机词替换该数字会使状态发生大得多的位移且可被读取。由于每个变化的词元都会留下痕迹，成功的探查结果同样存在歧义。我们提出了一套探查协议，包括用于检验一般性混合效应的合成零假设，用于将模型计算出的值与输入的副本及混合信息进行区分测试。

    arXiv:2610.28498v1 Announce Type: cross  Abstract: Probing studies of latent reasoning often read a model's state where the answer begins and take a failed probe as evidence that a value is absent. We test this with a digit copied from the question, in the looped model Ouro-1.4B and the standard transformer Qwen2.5-1.5B. Comparing prompts that differ only in that digit shows it is present at the last prompt token, in a code shared across questions (readout about 0.9). A probe that sees one prompt at a time, the usual setting, barely reads it: 0.36 in Ouro and 0.19 in Qwen even after 5,249 training questions. The digit's change is small rather than crowded: random noise lands on equally busy directions, while a random word in its place moves the state far more and is readable. Since every varied token leaves a trace, a successful probe is ambiguous too. We give a protocol, including a synthetic null for generic mixing, for testing computed values against copies and mixtures of the input
    
[^28]: 基于状态空间模型的脉冲神经网络的模型级与层级闭环预测

    Model- and Layer-Level Closed-Loop Forecasting with SSM-Based Spiking Neural Networks

    [https://arxiv.org/abs/2610.28495](https://arxiv.org/abs/2610.28495)

    本文针对脉冲神经网络自回归预测中所需历史值不可用的问题，提出了模型级与层级的闭环预测策略，将网络自身生成的预测值反馈回网络作为输入。

    

    深度状态空间模型在处理具有复杂时间依赖性的时间序列数据方面展现出了有前景的性能和计算优势。脉冲神经网络是一类受生物学启发的深度状态空间模型，其中计算模块通过脉冲——即离散的单比特事件——而非连续的激活函数进行通信。时间序列处理中一项特别重要的任务是预测，即仅根据过去的观测值来预测序列的未来值。然而，如何最好地使用脉冲神经网络进行预测仍是一个悬而未决的问题，因为脉冲神经网络数据处理的标准自回归特性需要用到在预测阶段无法获得的值。在本文中，我们应对了这一挑战。我们探索了两种闭环预测策略，即将网络自身生成的值反馈到网络中，如同它们是观测到的过去值一样。

    arXiv:2610.28495v1 Announce Type: cross  Abstract: Deep State-Space Models (Deep SSMs) have shown promising performance and computational benefits in processing of time-series data with complex temporal dependencies. Spiking neural networks (SNNs) constitute a biologically inspired class of deep SSMs, where computing blocks communicate through spikes, discrete single-bit events rather than continuous activation functions. A particularly important task in time-series processing is forecasting, i.e. predicting future values of a sequence from past observations alone. However, how to best perform forecasting with SNNs is an open question, where the standard autoregressive nature of the data processing with SNNs requires values that are not available during the forecasting phase. In this article, we address this challenge. We explore two closed-loop forecasting strategies, in which the network's own generated values are fed back into the network as if they were observed past values: (i) mo
    
[^29]: 几何无关的数据驱动极端场景发现

    Geometry-Agnostic Data-Driven Discovery of Extreme Scenarios

    [https://arxiv.org/abs/2610.28487](https://arxiv.org/abs/2610.28487)

    该论文提出一种两阶段数据驱动神经框架，无需预知控制方程即可从观测数据中学习几何无关的动力学约束，并通过物理信息神经搜索发现原始观测中未曾出现的稀有极端场景。

    

    我们提出了一种两阶段的数据驱动神经框架，用于发现动力学上可容许的极端场景。该框架不假设预先掌握系统的控制方程，第一阶段直接从观测数据中推断有效的动力学约束。当底层物理系统具有适当的对称性时，所学到的动力学可以表示为一种几何无关的形式，独立于任何固定的网格、坐标系、维度和内在曲率。在第二阶段，一种物理信息引导的神经搜索利用所学到的动力学来揭示可能未在原始观测中出现过的稀有、高影响的相干结构。我们通过维度和几何复杂度不断增加的代表性示例演示了这一框架。从传统方法学习的Duffing动力学出发，我们发现了同宿脉冲这一合理的局域化极端场景。接下来，由于同一方程……（原文摘要在此处截断）

    arXiv:2610.28487v1 Announce Type: new  Abstract: We introduce a two-stage data-driven neural framework for discovering dynamically admissible extreme scenarios. Rather than assuming prior knowledge of the governing equations, the first stage infers the effective dynamical constraints directly from observational data. When the underlying physical system possesses suitable symmetries, the learned dynamics can be represented in a geometry-agnostic form, independent of any fixed grid, coordinate system, dimensionality, and intrinsic curvature. In the second stage, a physics-informed neural-search uses this learned dynamics to uncover rare, high-impact coherent structures that may not have appeared in the original observations. We demonstrate this framework through representative examples of increasing dimensionality and geometric complexity. Starting from a traditionally learned Duffing dynamics, we find a homoclinic pulse as a plausible localized extreme scenario. Next, since the same equ
    
[^30]: AI中的涌现论断应当以信息分解为理论基础

    Claims of emergence in AI should be grounded in information decomposition

    [https://arxiv.org/abs/2610.28479](https://arxiv.org/abs/2610.28479)

    本文主张以信息分解作为统一的形式化框架，将AI系统中关于涌现与协同的非正式论断转化为精确、可比较且可证伪的评估。

    

    当语言模型发展出无法追溯到其单个组件的能力时，当多智能体AI系统的表现超过其中每个单独的智能体时，或者当人机组合在两者单独都无法成功的情况下取得成功时，AI研究者会将其描述为“协同”或“涌现”的案例。然而，这些通常是非正式的评估，缺乏理论框架的支持，使得这些论断既无法相互比较，甚至也无法进行有意义的争辩。本文论证，信息分解可以作为一个统一的形式化框架，用于将AI系统中关于协同性的评估操作化。信息分解提供了一种区分性质不同的信息类型的方法，包括可从单个组件获得的独特信息、跨多个组件共享的冗余信息，以及只有将各部分结合在一起才能获得的协同信息。这使得涌现论断具有了精确的、可证伪的形式。

    arXiv:2610.28479v1 Announce Type: new  Abstract: When a language model develops capabilities not traceable to its individual components, when a multi-agent AI system outperforms each agent on its own, or when a human-AI pair succeeds where either fails in isolation, AI researchers describe it as a case of ``synergy'' or ``emergence''. However, these are usually informal assessments not supported by a theoretical framework, making these claims impossible to compare or even meaningfully dispute. Here we argue that \emph{information decomposition can serve as a unifying formal framework to operationalize assessments about synergy in AI systems}. Information decomposition provides a way to distinguish qualitatively different \emph{types} of information, including unique information available from individual components, redundant information shared across multiple components, and synergistic information accessible only from parts taken together. This gives claims of emergence a precise, fal
    
[^31]: LocaLeJEPA：几何先验下的最大熵表示

    LocaLeJEPA: Maximum Entropy Representations under Geometric Priors

    [https://arxiv.org/abs/2610.28458](https://arxiv.org/abs/2610.28458)

    该论文提出 LocaLeJEPA 框架，通过对 LeJEPA 熵最大化性质进行流形理论重新表述，并引入局部欧几里得分布匹配正则化器 LocaLeReg，使自监督学习能够支持非欧几里得几何先验（如等变CNN和双曲ViT），同时避免表示坍缩。

    

    联合嵌入预测架构（JEPA）为自监督学习提供了一种有吸引力的方法，但通常容易出现表示坍缩问题。LeJEPA 通过引入一个具有理论依据的分布匹配目标来解决这一问题，该目标将嵌入推向各向同性高斯分布。然而，这种方法从根本上依赖于欧几里得假设，其全局目标无法反映特定领域的几何先验，例如使用非欧几里得嵌入几何的模型，如等变CNN或双曲ViT。因此，我们提出了 LocaLeJEPA，这是一个基于对 LeJEPA 熵最大化性质进行流形理论重新表述的框架。配套提出的 LocaLeReg 是一种新颖的局部欧几里得分布匹配正则化器，使 LocaLeJEPA 框架能够得到稳定且可扩展的实现。随后，我们在视觉领域对在 ImageNet-1K 上训练的模型进行了大量实验。LocaLeJEPA s...（原文摘要在此处截断）

    arXiv:2610.28458v1 Announce Type: new  Abstract: Joint-Embedding Predictive Architectures (JEPA) offer an appealing approach to self-supervised learning, but are typically prone to representation collapse. LeJEPA resolves this by introducing a theoretically grounded distribution-matching objective that drives the embedding towards an isotropic Gaussian. However, this approach is fundamentally tied to Euclidean assumptions, and its global target does not reflect domain-specific geometric priors, such as the use of non-Euclidean embedding geometries, e.g., equivariant CNNs or hyperbolic ViTs. We thus introduce LocaLeJEPA, a framework motivated by a manifold-theoretic reformulation of LeJEPA's entropy-maximization property. The accompanying LocaLeReg, a novel locally Euclidean distribution-matching regularizer, enables a stable and scalable implementation of the LocaLeJEPA framework. We then perform extensive experiments in the vision setting on models trained on ImageNet-1K. LocaLeJEPA s
    
[^32]: 胞腔层拉普拉斯算子、曲率与神经层扩散的Bochner-Weitzenböck分解

    A Bochner Weitzenb\"ock Decomposition for Cellular Sheaf Laplacians, Curvature and Neural Sheaf Diffusion

    [https://arxiv.org/abs/2610.28455](https://arxiv.org/abs/2610.28455)

    本文将Bochner-Weitzenböck分解推广到胞腔层拉普拉斯算子，将其分解为Bochner层拉普拉斯算子与Forman曲率两个部分，建立了相应的和乐刻画、Cheeger不等式与离散Bochner定理，并将该理论应用于层神经网络，得到节点级信念得分等新工具。

    

    我们将Bochner-Weitzenböck分解扩展到胞腔层拉普拉斯算子，将其拆分为一个半正定项（Bochner层拉普拉斯算子）和一个块对角项（Forman曲率）。我们证明Bochner拉普拉斯算子构成一类胞腔层，它同时扩展了联络拉普拉斯算子和矩阵加权拉普拉斯算子，并将带符号图推广到$d$维。我们给出了其核的和乐刻画，证明了相关的Cheeger不等式，并为其附加了一个推广Gremban展开的随机游走。对于Forman项，我们证明了一个离散Bochner定理，该定理对具有非负曲率的胞腔层的第一上同调给出界，并且我们将整个分解解读为一个观点动力学模型。随后我们将这一机制应用于层神经网络，此类网络学习到的拉普拉斯算子通常是不可观测的。Bochner解释产生了一个节点级的信念得分，它作为一种操作……

    arXiv:2610.28455v1 Announce Type: cross  Abstract: We extend the Bochner-Weitzenb\"ock decomposition to cellular sheaf Laplacians, splitting it into a positive semidefinite term, the Bochner sheaf Laplacian, and a block diagonal term, the Forman curvature. We show that the Bochner Laplacians form a class of cellular sheaves that extends both connection and matrix-weighted Laplacians, and generalizes signed graphs to $d$ dimensions. We give a holonomy characterization of their kernel, prove the related Cheeger inequalities, and attach to them a random walk that generalizes the Gremban expansion. For the Forman term, we prove a discrete Bochner theorem, which bounds the first cohomology of a cellular sheaf with nonnegative curvature, and we read the whole decomposition as an opinion-dynamics model. We then apply this machinery to Sheaf Neural Networks, whose learnt Laplacian is otherwise unobserved. The Bochner interpretation yields a node-level conviction score that acts as an operation
    
[^33]: 基于CNN的分类器：利用短时脑电信号进行与被试无关的MBTI外向/内向偏好分类

    CNN-based classifier for subject-independent MBTI Extraversion/Introversion preference classification from short-time EEG signals

    [https://arxiv.org/abs/2610.28442](https://arxiv.org/abs/2610.28442)

    本研究提出紧凑型卷积神经网络SSCNet，仅需4秒的短时脑电信号即可实现与被试无关的MBTI外向/内向偏好中等程度预测，在准确率、稳定性和计算效率上均优于EEGNet和EEGProgress等现有方法。

    

    本研究探讨经过最少预处理的短时脑电（EEG）片段是否包含足够的信息，以区分根据迈尔斯-布里格斯类型指标（MBTI）被划分为外向或内向的个体。来自50名参与者的脑电记录被分割为4秒的片段，并采用与被试无关的嵌套5折交叉验证程序进行分析，比较了EEGNet、EEGProgress以及新提出的紧凑卷积架构SSCNet。SSCNet取得了最高的平均准确率（0.6481）、AUC（0.7063）和F1分数（0.6276），并且在各折之间表现出最低的变异性。此外，SSCNet展现出最佳的计算效率，尽管其可训练参数数量与EEGProgress相近，但其单轮训练的中位时间最短、训练吞吐量最高。结果表明，MBTI外向-内向偏好可能可以通过短时脑电信号得到中等程度的预测。

    arXiv:2610.28442v1 Announce Type: new  Abstract: This study examines whether short, minimally preprocessed EEG segments contain sufficient information to distinguish between individuals classified as extraverted or introverted according to the Myers--Briggs Type Indicator (MBTI). EEG recordings from 50 participants were divided into 4-s segments and analyzed using a subject-independent nested 5-fold cross-validation procedure to compare EEGNet, EEGProgress, and a newly proposed compact convolutional architecture, SSCNet. SSCNet achieved the highest mean accuracy (0.6481), AUC (0.7063), and F1-score (0.6276). The proposed architecture also showed the lowest variability across folds. SSCNet demonstrated the best computational efficiency, with the shortest median training epoch time and the highest training throughput despite having a similar number of trainable parameters to EEGProgress. The results suggest that MBTI Extraversion--Introversion preferences may be moderately predictable fr
    
[^34]: 神经模型是否偏好其子模型？

    Do Neural Models Prefer Their Submodels?

    [https://arxiv.org/abs/2610.28439](https://arxiv.org/abs/2610.28439)

    该论文通过引入几何框架研究神经模型的几何结构，证明除少数模型边界上的例外情况外，损失函数的局部极小值点通常不会落在由不活跃神经元或低秩结构构成的子模型中。

    

    神经模型是指由给定神经网络架构所能表达的函数集合。神经模型包含天然的子模型，例如通过将某些神经元设为不活跃状态而得到的子模型。我们研究神经模型的几何结构，并探究它们是否偏好其子模型，即损失函数的局部极小值点是否位于给定的子模型中。我们引入了一个几何框架来判断这种情况是否发生。对于浅层多层感知机，我们证明了含有不活跃隐藏神经元的子网络永远不会被偏好。对于深层多层感知机，我们给出了含有不活跃隐藏神经元的子网络不被偏好的条件。对于闪电自注意力机制，低秩子模型通常不被偏好，但我们也发现了一些例外情况，即低秩子模型因位于模型边界上而被偏好。我们的结果表明，除少数例外情况外，对子模型的偏好无法用几何结构来解释。

    arXiv:2610.28439v1 Announce Type: new  Abstract: A neural model is the set of functions that can be expressed by a given neural network architecture. Neural models contain natural submodels, such as those obtained by setting some neurons to be inactive. We study the geometry of neural models and ask whether they prefer their submodels. By this we mean whether local minimizers of the loss function lie in a given submodel. We introduce a geometric framework to determine if this occurs. For shallow multilayer perceptrons, we show that subnetworks with inactive hidden neurons are never preferred. For deep multilayer perceptrons, we give conditions under which subnetworks with inactive hidden neurons are not preferred. For lightning self-attention, lower-rank submodels are usually not preferred, but we identify examples where they are because they lie on a boundary of the model. Our results show that, aside from a few exceptions, preference for submodels cannot be explained by the geometry 
    
[^35]: 基于滑动窗口二维卷积神经网络的催眠易感性脑区特异性脑电分类：一项被试独立评估研究

    Region-Specific EEG Classification of Hypnotic Susceptibility Using a Sliding-Window 2D CNN: A Subject-Independent Evaluation

    [https://arxiv.org/abs/2610.28427](https://arxiv.org/abs/2610.28427)

    该论文用滑动窗口2D CNN对脑电进行脑区特异性的催眠易感性分类，发现虽然随机划分筛查中枕叶区准确率最高（0.721），但在严格的留一被试交叉验证下表现骤降至接近随机水平，凸显了被试独立评估对脑电分类研究的关键意义。

    

    我们研究了在最小预处理的脑电信号上训练的二维卷积神经网络（CNN）能否跨不同脑区对催眠易感性进行分类。62名男性被试（32名对照组，30名引导意象/催眠组）使用256通道EGI脑电帽进行了20分钟的脑电记录。滑动窗口方法对七个预定义脑区（前额叶、额叶、颞叶、中央区、顶叶、枕叶和面部）记录中的所有10秒窗口进行评估，最多训练了868个独立模型。采用片段级90/10随机划分的初步筛查显示，五个脑区超过了0.67的准确率阈值，其中枕叶区在90.3%的窗口上取得了最高的平均准确率（0.721）。然而，通过留一被试交叉验证（LOSO）进行的被试独立评估大幅修正了这一结果：枕叶区的表现骤降至接近随机水平……

    arXiv:2610.28427v1 Announce Type: new  Abstract: We investigate whether a 2D Convolutional Neural Network (CNN) trained on minimally preprocessed EEG signals can classify hypnotic susceptibility across distinct brain regions. Sixty-two male participants (32 Control, 30 Guided Imagery / Hipno group) underwent a 20-minute EEG recording using a 256-channel EGI cap. A sliding-window approach evaluates all 10-second windows within the recording for each of seven predefined brain regions (prefrontal, frontal, temporal, central, parietal, occipital, and facial), yielding up to 868 independently trained models. Initial screening with a segment-level 90/10 random split showed that five regions exceeded the 0.67 accuracy threshold, with the occipital region achieving the highest mean accuracy (0.721) across 90.3% of windows. Subject-independent evaluation via Leave-One-Subject-Out (LOSO) cross-validation, however, substantially revised this picture: occipital performance collapsed to near-chance
    
[^36]: DISTRACT：面向竞争目标的退化索引式文本指涉研究

    DISTRACT: Degradation-Indexed Study of Textual Reference Across Competing Targets

    [https://arxiv.org/abs/2610.28419](https://arxiv.org/abs/2610.28419)

    提出了DISTRACT匹配干预基准，通过固定视频、目标和干扰物并交叉文本先验与图像退化的方式，系统检验视觉-语言跟踪器是否真正利用文本内容做出跟踪决策，发现仅Molmo2和JointNLT表现出有方向性的文本利用，而其余模型对文本语义变化的响应微弱。

    

    视觉-语言跟踪器（VLT）接受旨在辅助跟踪的文本描述，但接受文本并不能证明其内容会真正改变跟踪决策，尤其是在视觉证据不断退化的情况下。我们提出了DISTRACT，这是一个匹配干预基准：在保持视频、目标和干扰物不变的前提下，将四种文本先验与四种图像退化沿受控的严重程度轴进行交叉组合。我们利用行为对比、逐层表示探测和激活修补（activation patching）方法，在四个数据集上评估了五种VLT。结果显示，Molmo2和JointNLT表现出有方向性的文本利用行为；其中一些效应在退化条件下会减弱，而另一些效应在经过多重性校正后未显示显著的终点变化。UVLTrack、MMTrack和DUTrack对文本语义变化的响应较弱，在所测试的设置中，没有任何有方向性的行为效应能在FDR校正后保留。模型内部分析揭示，在接近决策的状态下，正确目标与冲突目标之间的……（摘要在此处截断）

    arXiv:2610.28419v1 Announce Type: cross  Abstract: Vision-language trackers (VLTs) accept descriptions intended to assist tracking, but accepting text does not establish that its content changes tracking decisions, particularly as visual evidence deteriorates. We introduce DISTRACT, a matched-intervention benchmark that holds video, target, and distractor fixed while crossing four text priors with four image degradations along a controlled severity axis. We evaluate five VLTs on four datasets using behavioral contrasts, layerwise representation probes, and activation patching. Molmo2 and JointNLT exhibit directed text use. Some effects contract under degradation, whereas others show no significant endpoint change after multiplicity correction. UVLTrack, MMTrack, and DUTrack respond only weakly to changes in text semantics, with no directed behavioral effect surviving FDR correction in the tested settings. Model-internal analyses reveal that at decision-near states, correct-versus-confl
    
[^37]: Brain2Vec：一种用于结构化脑部MRI的自监督3D基础模型

    Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI

    [https://arxiv.org/abs/2610.28411](https://arxiv.org/abs/2610.28411)

    Brain2Vec是一个在74,425个T1加权脑部MRI扫描上预训练的自监督3D基础模型，既可作为下游任务微调的权重初始化骨干，也可作为独立特征提取器，在脑年龄预测等任务中展现出更高的精度和更快的收敛速度。

    

    深度学习模型已成为医学影像领域不可或缺的工具，然而由于标注数据有限和计算资源受限，针对多样化任务进行端到端训练仍然充满挑战。为解决这一问题，我们提出了Brain2Vec，这是一个在汇集自多项研究的74,425个T1加权磁共振成像（MRI）扫描数据上预训练的基础模型。Brain2Vec既可以作为权重初始化的骨干网络用于下游任务的微调，也可以作为独立的特征提取器使用。我们从三个方面对Brain2Vec进行了评估：下游预测性能、收敛速度和嵌入质量。在脑年龄预测任务中，使用Brain2Vec的预训练权重作为初始化并在标注数据上进行微调，结果表明与随机初始化相比，最终精度得到提升且收敛速度更快。此外，其冻结嵌入特征在五个下游分类任务中的四个上优于基于解剖学定义的体积特征。

    arXiv:2610.28411v1 Announce Type: cross  Abstract: Deep learning models have become essential tools in medical imaging, yet training them end-to-end for diverse tasks remains challenging due to limited labeled data and computational constraints. To address this, we introduce Brain2Vec, a foundation model pre-trained on 74,425 T1-weighted Magnetic Resonance Imaging (MRI) scans pooled from multiple studies. Brain2Vec can serve both as a weight initialization backbone for fine-tuning downstream tasks and as a standalone feature extractor. We evaluated Brain2Vec in three tasks: downstream predictive performance, convergence speed, and embedding quality. Using Brain2Vec's pretrained weights as initialization and fine-tuning on labeled data for a brain age prediction task, results demonstrate improved final accuracy and faster convergence compared to random initialization. Additionally, its frozen embeddings outperform anatomically defined volumetric features on four of five downstream class
    
[^38]: 适应任意测试时约束的预算感知多轮智能体奖励机制

    Rewards for Budget-Aware Multi-Turn Agents That Adapt to Arbitrary Test-Time Constraints

    [https://arxiv.org/abs/2610.28409](https://arxiv.org/abs/2610.28409)

    本文提出并比较了十一种奖励目标函数，使大语言模型多轮智能体能够学会自主分配预算，并适应token、API调用、环境步骤等多种资源类型上动态指定的测试时约束。

    

    大语言模型智能体通常会被赋予一个推理预算，并被要求在该预算内最大化成功率。多轮智能体必须将这一预算分配到多次环境交互之中。外部框架可以强制执行预算限制，但智能体本身能否学会何时花费以及花费多少预算，目前仍不清楚。我们探究了智能体能否学会在动态指定的预算之间进行这种分配，以及哪些奖励最能教会这种行为。我们将现有的长度惩罚方法适配到跨越多种资源类型的预算约束上，提出了新的奖励形式，并在一个共享的训练流程中比较了十一种目标函数。我们在AppWorld上以token、API调用和环境步骤作为预算，训练了4B和9B的Qwen3.5模型，并在BabyAI网格世界中以轮次预算进行训练。我们为每个环境和资源类型分别训练一个策略，且每个策略都在不同的预算值上进行训练。使用惩罚超支行为的奖励进行训练……（原文摘要在此截断）

    arXiv:2610.28409v1 Announce Type: cross  Abstract: Large language model agents are often given an inference budget and asked to maximise success within it. A multi-turn agent has to allocate this budget across many environment interactions. An external harness can enforce the budget, but it remains unclear whether the agent itself can learn when and how much to spend. We ask whether agents can learn this allocation across dynamically specified budgets and which rewards best teach this behaviour. We adapt existing length penalties to budget constraints spanning multiple resource types, introduce new reward formulations, and compare eleven objectives in a shared training pipeline. We train Qwen3.5 models at 4B and 9B on AppWorld with budgets over tokens, API calls, and environment steps, and on the BabyAI gridworld with turn budgets. We train a separate policy for each environment and resource type, with each policy trained across budget values. Training with rewards that penalise oversp
    
[^39]: Self-Credit：基于组内重缩放的长程智能体强化学习信用分配方法

    Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling

    [https://arxiv.org/abs/2610.28388](https://arxiv.org/abs/2610.28388)

    提出Self-Credit方法，通过让策略模型自生成带引用的摘要信用，并利用组内对成功与失败rollout的比较进行重缩放，从稀疏结果奖励中为长程智能体强化学习构建密集的信用分配信号。

    

    LLM智能体通常使用RLVR（基于可验证奖励的强化学习）进行训练，但典型RLVR奖励的稀疏性使得信用分配极具挑战性，在长程任务中尤其如此。我们提出了Self-Credit，这是一种端到端的智能体强化学习方法，通过组内重缩放的自生成信用分配机制，从稀疏的结果奖励中构建密集的监督信号。Self-Credit建立在组式强化学习之上，包含三个组件：首先，在策略模型进行rollout的过程中，它会生成带有引用的摘要，引用那些包含关键证据的较早轮次，作为自生成的信用；其次，我们通过组内重缩放来精炼模型生成的信用，即针对同一提示比较成功与失败rollout中被引用的动作，利用组级别的一致性与不一致性去除不当的信用，并进一步强化分配给关键动作的信用；第三，优势重加权利用重缩放后的信用权重来更新……

    arXiv:2610.28388v1 Announce Type: new  Abstract: LLM agents are commonly trained with RLVR, but the sparse nature of typical RLVR rewards makes credit assignment challenging, especially for long-horizon tasks. We propose Self-Credit, an end-to-end agentic RL method that creates dense supervision signals from sparse outcome rewards using self-generated credit assignment via intra-group rescaling. Self-Credit builds on group-based RL with three components. First, during its rollouts, the policy model generates summaries with citations to earlier turns containing critical evidence, which serve as self-generated credit. Second, we refine this model-generated credit with intra-group rescaling, which compares cited actions across successful and failed rollouts for the same prompt. This approach uses group-level agreement and disagreement to remove undeserved credit and further emphasize credit assigned to critical actions. Third, advantage reweighting uses the rescaled credit weights to upda
    
[^40]: 不要跳过基线：MoE模型中动态专家跳过方法的受控比较

    Don't Skip the Baseline: A Controlled Comparison of Dynamic Expert Skipping in MoEs

    [https://arxiv.org/abs/2610.28369](https://arxiv.org/abs/2610.28369)

    本文首次将四种动态专家跳过方法纳入统一分类体系，并在匹配专家计算预算下与“静态减少专家数量”这一简单基线在两个现代MoE模型上进行了受控公平比较。

    

    越来越多的研究工作提出了针对混合专家模型的动态专家跳过方法，这些方法在推理过程中调整每个token激活的专家数量，旨在保持模型质量的同时进一步减少激活参数的数量。这些方法通过添加校准的重要性分数、专家间相似度度量或学习式的逐token分配器来识别需要跳过的专家。然而，不同的校准机制、不同的有效专家数量缩减幅度以及不同的评估套件使得这些方法之间难以进行公平比较。我们将四种代表性的已有方法归类到一个统一的分类体系中，并与一个自然、简单的基线进行基准测试：静态地减少专家数量k（即在所有层和所有token中使用相同的缩减后k值）。我们在匹配的专家计算预算下，在两个现代MoE模型（Gemma4-26B-A4B和Qwen3-30B-A3B）的纯文本设置中对这些方法进行了公平比较。

    arXiv:2610.28369v1 Announce Type: cross  Abstract: A growing body of work proposes dynamic expert-skipping methods for Mixture-of-Experts (MoE) models that adjust the number of experts activated per token during inference, aiming to preserve model quality while further reducing the number of active parameters. These methods have added calibrated importance scores, inter-expert similarity metrics, or learned per-token allocators in order to identify experts to skip. However, different calibration regimes, different reductions in the effective number of experts, and evaluation suites make comparisons difficult. We classify four representative prior methods into a unified taxonomy and benchmark them against a natural, simple baseline: statically reducing the number of experts $k$ (i.e., same reduced $k$ across all layers and tokens). We provide a fair comparison of these methods at matched expert-compute budget, on two modern MoEs (Gemma4-26B-A4B and Qwen3-30B-A3B) in a text-only setting 
    
[^41]: 相对表征使白盒监控能够跨语言模型移植

    Relative Representations Enable Portable White-Box Monitoring Across Language Models

    [https://arxiv.org/abs/2610.28366](https://arxiv.org/abs/2610.28366)

    该论文发现不同语言模型的内部表征可通过相对角度距离映射到兼容空间，使在一个模型上训练的行为探针无需目标模型标注数据即可直接复用，从而实现可移植的白盒监控。

    

    arXiv:2610.28366v1 公告类型：新论文 摘要：独立训练的自回归语言模型往往具有相似的能力，但尚不清楚它们的内部表征是否以兼容的方式组织与行为相关的信息。我们通过使用在兼容表征上运行的特定行为探针，证明不同模型实际上确实以相似的方式组织信息。这解锁了可移植的白盒监控：在一个模型上开发和验证的探针，可以在模型更换时被复用，而无需带标注的目标数据。为了克服各模型原生激活坐标不兼容的问题，我们通过每个样本到一组共享锚点的角度距离来表达它。我们引入了一种全局锚定构造，它能汇总所有 transformer 层的信息，并消除了模型特定的层选择，因此同一个冻结的探针可以直接应用于另一个模型。我们在监控任务上评估了跨模型的探针迁移效果（摘要在此处截断）。

    arXiv:2610.28366v1 Announce Type: new  Abstract: Independently trained autoregressive language models often acquire similar capabilities, but it remains unclear whether their internal representations organize behaviorally relevant information in compatible ways. We show that models do in fact organize information similarly by using specific behavioral probes that operate on compatible representations. This unlocks portable white-box monitoring: a probe developed and validated on one model can be reused when the model changes, without requiring labeled target data. To overcome incompatible native activation coordinates, we express each example through its angular distances to a shared set of anchors. We introduce a global anchored construction that summarizes information from all transformer layers and removes model-specific layer selection, so the same frozen probe can be straightforwardly applied to another model. We evaluate cross-model probe transfer across monitoring tasks on trivi
    
[^42]: 掩码MDP的自白

    Confessions of a Masked MDP

    [https://arxiv.org/abs/2610.28360](https://arxiv.org/abs/2610.28360)

    本文首次研究了动作掩码未知的线性MDP中策略优化与掩码学习相结合的问题，提出了一个基于确定性乐观优化的最小最大最优且计算高效的算法，并进一步提出了一个能主动规划探索策略、在非退化实例上表现更优的算法。

    

    在强化学习中，人们通常假设动作集是预先已知的，因此必须在计算开始之前就知道哪些状态-动作对是不被允许的。在某些应用中，确定这个动作掩码是主要的计算瓶颈。为了将这一问题隔离出来以突出其核心性质，我们研究了在线性MDP模型下如何将最优策略的搜索与掩码本身的学习相结合，在该模型中，每次尝试一个动作都会揭示决定其有效性的线性分数的精确值。在这一框架下，我们首先构建了一个基于确定性乐观优化的算法，并证明它是最小最大最优的，同时我们也表明它在计算上是高效的。然而，正如我们所展示的，该算法在非退化实例上是次优的。我们提出了另一种算法，它同样在计算上是高效的，并且能够主动规划其探索策略。该算法具有一个形式化的解释……（摘要在此处截断）

    arXiv:2610.28360v1 Announce Type: cross  Abstract: In reinforcement learning, one presumes the action set to be known ahead of time, and, thus, one must know if any state-action pairs are not permissible   ahead of computations. In some applications, determining this action mask is the main computational bottleneck. Isolating this problem to highlight its core   properties, we study how to combine the search for an optimal policy with the learning of the mask itself, under a linear MDP model in which each attempted   action reveals the exact value of a linear score determining its validity. In this framework, we first construct an algorithm based on deterministic   optimistic optimization and show that it is minimax optimal, which we show is also computationally efficient. However, it is sub-optimal on non-degenerate   instances, as we show. We propose an alternative, which is also computationally efficient, which plans its exploration strategy. This algorithm admits a   formal interpr
    
[^43]: CountVideoBench：一个用于评估视频语言模型在物体计数任务上预测偏差的基准

    CountVideoBench: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models

    [https://arxiv.org/abs/2610.28356](https://arxiv.org/abs/2610.28356)

    该论文提出CountVideoBench基准，利用2,600个覆盖[5,50]区间内26个计数值的合成视频，解决了现有视频问答基准计数分布严重偏向小数值、导致大数值上的模型预测偏差无法被评估的问题，从而系统性地评估视频语言模型在物体计数任务中的预测偏差。

    

    arXiv:2610.28356v1 公告类型：交叉 摘要：视频中的物体计数仍然是多模态大语言模型一项刻画不足的能力。许多现有的视频问答（VideoQA）基准中包含计数问题，但这些基准中的真实计数值分布严重偏向小数值。在我们调查的九个基准中，真实计数的中位数至多为5，且出现频率最高的三个数值合计占视频总数的45%至93%。因此，平均准确率主要由最常见的计数值主导；在罕见计数值上出现的依赖数值的偏差对平均值的贡献微乎其微，因而无法被察觉。我们提出了CountVideoBench，这是一个包含2,600个合成视频的基准，覆盖[5, 50]区间内的26个计数值（每个数值100个视频），并被分为六个由连续计数组成的聚类，既覆盖整个数值范围，又能支持近邻数值间的比较。每个视频在受控的摄像条件下包含单一类别的静态物体。

    arXiv:2610.28356v1 Announce Type: cross  Abstract: Object counting in videos remains a poorly characterized capability of multimodal large language models. Counting questions are included in many existing VideoQA benchmarks, but the ground-truth count distributions in these benchmarks are heavily skewed toward small values. Across the nine benchmarks we survey, the median ground-truth count is at most five, and the three most frequent values together account for between 45% and 93% of the videos. The resulting mean accuracy is then dominated by the most frequent counts; a value-dependent bias at a rare count contributes negligibly to the average and remains invisible. We introduce CountVideoBench, a benchmark of 2,600 synthetic videos covering 26 cardinalities in the interval [5, 50] (100 videos each), grouped into six clusters of consecutive counts that span the range while enabling near-neighbor comparisons. Each video contains a single class of static objects under a controlled came
    
[^44]: 超越点质量：用于推测解码的基于接受率的权重与分布式损失

    Beyond the Point Mass: Acceptance-Derived Weights and Distributional Losses for Speculative Decoding

    [https://arxiv.org/abs/2610.28355](https://arxiv.org/abs/2610.28355)

    本文提出 D-PAL（动态逐位置接受损失），从精确期望接受长度出发，用草稿与目标分布的重叠度（1减总变差）取代点质量假设下的交叉熵信号，以基于重叠度的位置权重配合对数接受率作为逐位置损失来训练推测解码的草稿模型，且在目标为点质量时该目标函数退化为 D-PACE。

    

    在推测解码中，一个轻量级的草稿模型提出词元，由目标模型并行验证，从而在不改变目标分布的前提下加速生成。D-PACE（Wu et al., 2026）通过一种按每个位置对期望接受长度的贡献进行加权的损失来训练这类草稿模型。然而，其权重和交叉熵（CE）损失都仅使用草稿模型对被记录目标词元的概率，将目标分布视为点质量。但在拒绝采样机制下，采样出的草稿词元被接受的概率等于草稿分布与目标分布的重叠度，即一减去二者的总变差（TV）。我们从精确的期望接受长度出发推导出 D-PAL（动态逐位置接受损失），将该重叠度用作接受信号。这带来了基于重叠度的位置权重，并以对数接受率（LA）作为逐位置损失，且该目标函数在目标为点质量时退化为 D-PACE。

    arXiv:2610.28355v1 Announce Type: new  Abstract: In speculative decoding, a lightweight draft model proposes tokens that the target model verifies in parallel, speeding up generation without changing the target distribution. D-PACE (Wu et al., 2026) trains such drafters with a loss that weights each position by its contribution to the expected accepted length. Both its weights and its cross-entropy (CE) loss use only the draft's probability of the recorded target token, treating the target as a point mass. Under rejection sampling, however, a sampled draft token is accepted with probability equal to the overlap of the draft and target distributions, one minus their total variation (TV). We derive D-PAL (Dynamic Position-wise Acceptance Loss) from the exact expected accepted length, using this overlap as the acceptance signal. This yields overlap-based position weights with log acceptance (LA) as the per-position loss, and the objective reduces to D-PACE for a point-mass target. Since t
    
[^45]: 基于机器学习的电力系统保护中跨继电器泛化失效的表征分析

    Characterizing Cross-Relay Generalization Failure in Machine Learning-Based Power System Protection

    [https://arxiv.org/abs/2610.28348](https://arxiv.org/abs/2610.28348)

    该研究揭示了基于机器学习的故障检测器在跨继电器部署时存在方向性泛化失效——虽然平均F1分数几乎不变，但误报率从0.088激增至0.341，说明仅依靠平均指标评估会掩盖部署中严重的可靠性劣化问题。

    

    基于机器学习的故障检测器通常在训练所用的继电器位置上进行评估，然而电力网络中各继电器位置的局部电压和电流测量值存在差异。本工作研究了基于局部测量、对网络中任意位置故障进行二元检测（而非线路选择性跳闸）的跨继电器泛化问题。利用从模拟110 kV输电网络中提取的50毫秒时间窗口数据，我们在8个继电器和5个随机种子的全部64种源-目标组合上评估了多层感知机。通过共享的仿真级数据划分和仅基于源数据的阈值选择，实现了无需目标侧校准的受控留出评估。尽管平均F1分数几乎保持不变（继电器内为0.730，跨继电器为0.734），平均误报率（FPR）却从0.088升至0.341，而平均漏报率从0.399降至0.318。这种变化呈现出强烈的方向性：在母线2（Bus 2）训练的模型……

    arXiv:2610.28348v1 Announce Type: new  Abstract: Machine learning-based fault detectors are often evaluated at the relay locations used for training, although local voltage and current measurements vary across power networks. This work investigates cross-relay generalization for binary detection of faults anywhere in the network using local measurements, rather than line-selective tripping. Using 50 ms windows from a simulated 110 kV transmission network, we evaluate a multilayer perceptron across all 64 source target combinations of eight relays and five random seeds. A shared simulation-level split and source-only threshold selection provide a controlled held-out evaluation without target-specific calibration. Despite nearly unchanged mean F1-scores (0.730 within-relay versus 0.734 cross-relay), the mean false-positive rate (FPR) increases from 0.088 to 0.341, while the mean false-negative rate decreases from 0.399 to 0.318. The change is strongly directional: models trained at Bus 2
    
[^46]: 从多域线性混合中识别相关成分

    Identifying Dependent Components from Multi-domain Linear Mixtures

    [https://arxiv.org/abs/2610.28330](https://arxiv.org/abs/2610.28330)

    该论文提出了一种多域线性混合模型，利用各域之间域特定的成分缩放差异以及潜在成分分布在域间一致这一特点，实现了对相关（非独立）潜在成分的可识别与恢复，突破了传统独立成分分析要求成分独立或不相关的限制。

    

    arXiv:2610.28330v1 公告类型：交叉 摘要：我们研究了一个具有相关潜在成分的线性混合模型，并假设存在多个数据域。大多数现有模型假设各成分相互独立或至少互不相关，这与独立成分分析（ICA）的思路一致。近期的一些工作允许成分之间存在相关性，但随后做出了特定的假设，例如依赖关系具有参数化形式、多视图设置或干预机制，或者无法恢复单个成分本身。相比之下，我们考虑一种多域设置，其中各个域通过域特定的成分缩放而彼此不同，而底层潜在成分的分布在各域之间保持相同。这种方法可以对诸如以下来源收集的数据进行建模：测量同一过程的不同传感器、进行相同实验的不同实验室、不同的实验条件，或在生物或生理因素上可能存在差异的不同受试者。我们表明……（摘要在此处截断）

    arXiv:2610.28330v1 Announce Type: cross  Abstract: We study a linear mixing model with dependent latent components, assuming multiple data domains. Most existing models assume that the components are independent or at least uncorrelated, in line with independent component analysis (ICA). Some recent work allows for dependent components, but then makes specific assumptions such as parametric forms of dependencies, multi-view settings, or interventions, or does not recover the individual components. In contrast, we consider a multi-domain setting in which domains differ through domain-specific scalings of the components, while the distribution of the underlying latent components is the same across domains. This approach can model data collected, for example, from different sensors measuring the same process, different laboratories conducting the same experiment, different experimental conditions, or different subjects that might differ in biological or physiological factors. We show that
    
[^47]: 观测同化Transformer：基于稀疏站点观测的概率性天气分析

    Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations

    [https://arxiv.org/abs/2610.28327](https://arxiv.org/abs/2610.28327)

    提出观测同化Transformer（OAT），一种几何感知的神经过程模型，利用稀疏注意力机制同化稀疏站点观测，将数值天气预报先验场修正为高分辨率的概率性后验场，可扩展至数千站点并支持任意连续位置的预测。

    

    我们提出了观测同化Transformer（OAT），这是一种几何感知的神经过程，通过同化稀疏观测，将先验的数值天气预报（NWP）场修正并精化为高分辨率的后验场。OAT是一个编码器-解码器Transformer，它以一组观测位置及其NWP先验为条件，在任意未观测的查询位置集合上进行预测。其稀疏注意力机制使其可扩展至数千个站点观测，且仅依赖于位置之间的相对位置，而从不依赖绝对坐标。该模型允许部分缺失的观测，可在任意连续位置进行预测而无需网格投影，并且输出为参数化分布。我们开发了一种基于连续排序概率评分（CRPS）损失的站点随机化训练算法。我们在两个数据集上测试了OAT——一个是包含数百个站点的区域尺度数据集，另一个是……（摘要在此处截断）

    arXiv:2610.28327v1 Announce Type: new  Abstract: We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior field by assimilating sparse observations. OAT is an encoder-decoder transformer that considers a set of observed locations and their NWP prior to condition predictions at a set of arbitrary unobserved query locations. Its sparse attention mechanism makes it scalable to thousands of station observations and depends on relative positions between locations, never on the absolute coordinates. The model allows for partially missing observations, predictions at arbitrary continuous locations without grid projection, and outputs are parametric distributions. A station-randomized training algorithm is developed with continuous ranked probability score (CRPS) loss. We test OAT on two datasets - one at regional scale with hundreds of stations, and one
    
[^48]: 流形信息神经网络用于低标注心电图电解质推断

    Manifold-Informed Neural Networks for Low-Label ECG Electrolyte Inference

    [https://arxiv.org/abs/2610.28322](https://arxiv.org/abs/2610.28322)

    提出流形信息神经网络（MINN），用学习潜在空间k近邻图上带可学习任务感知源项的离散泊松方程流形算子残差，替代PINN中的物理算子残差，从而在缺乏已知物理算子的低标注场景下实现ECG电解质推断。

    

    物理信息神经网络（PINNs）在已知微分算子的残差上进行监督。然而许多高风险的推断问题并不存在这样的算子，例如临床信号、行为测量和材料属性等。我们提出了流形信息神经网络（MINN），这是一种与PINN平行的范式，它用流形算子残差取代物理算子残差：在学习到的潜在空间中构建k近邻图，并在其上建立离散泊松方程 $\mathbf{L}_G \mathbf{H} = s_\phi(\mathbf{x}, \mathbf{m})$，其中源项是可学习的、任务感知的。这一构造在结构上与PINN平行（用算子残差平方加标签数据，替代算子残差加边界数据）；当 s=0 时，该方法退化为齐次拉普拉斯方程 $\mathbf{L}_G\mathbf{H}=\mathbf{0}$（经典流形正则化求解的是同一方程，但最小化的是狄利克雷能量）。

    arXiv:2610.28322v1 Announce Type: new  Abstract: Physics-informed neural networks (PINNs) supervise on the residual of a known differential operator. Many high-stakes inference problems admit no such operator: clinical signals, behavioral measurements, materials properties. We introduce the Manifold-Informed Neural Network (MINN), a PINN-parallel paradigm that replaces the physics operator residual with a manifold operator residual: a discrete Poisson equation $\mathbf{L}_G \mathbf{H} = s_\phi(\mathbf{x}, \mathbf{m})$ on a $k$-nearest-neighbor graph in the learned latent space, with a learned, task-aware source. The construction is a structural parallel to PINNs (squared operator residual plus label data, in place of operator residual plus boundary data); the $s{=}0$ limit recovers the homogeneous Laplace equation $\mathbf{L}_G\mathbf{H}=\mathbf{0}$ (classical manifold regularization solves the same equation but minimizes the Dirichlet energy $\operatorname{tr}(\mathbf{H}^\top\mathbf{L
    
[^49]: 用于图信号生成的稳定化滤波器

    Stabilizing Filters for Graph Signal Generation

    [https://arxiv.org/abs/2610.28318](https://arxiv.org/abs/2610.28318)

    该论文推导了连续时间图生成模型在图扰动下的Wasserstein稳定性界，并据此提出了一个原则性的稳定图滤波器设计框架，使图信号生成过程对图结构误差具有鲁棒性。

    

    生成图信号是机器学习中的一项重要任务，将底层图结构融入生成模型可以实现结构感知的分布建模。然而，该图结构往往带有噪声或估计不完美，结构误差会影响生成过程以及由此产生的信号分布。虽然近期的工作已经为图神经常微分方程建立了稳定性保证，但拓扑薛定谔桥模型中所使用的图感知参考动力学的鲁棒性在很大程度上仍未被探索。在本文中，我们研究了一类连续时间生成模型的结构稳定性，其动力学将图滤波器与学习得到的图神经网络相结合。我们首先推导了一个Wasserstein稳定性界，用于量化图扰动对生成信号分布的影响。基于这一分析，我们随后提出了一个有原则的图滤波器设计框架

    arXiv:2610.28318v1 Announce Type: cross  Abstract: Generating graph signals is an important task in machine learning, and incorporating the underlying graph into a generative model enables structurally-informed distribution modeling. However, said graph is often noisy or imperfectly estimated, and structural errors can affect the generative process and the resulting signal distribution. While recent work has established stability guarantees for graph neural ordinary differential equations, the robustness of graph-aware reference dynamics, as used in topological Schr\"odinger bridge models, remains largely unexplored. Here, we investigate the structural stability of continuous-time generative models whose dynamics combine a graph filter with a learned graph neural network. We first derive a Wasserstein stability bound that quantifies the effect of graph perturbations on the generated signal distributions. Informed by this analysis, we then propose a principled framework for designing gr
    
[^50]: ABACUS：一个用于评估图像编辑模型在场景保持下对象类别与计数能力的基准

    ABACUS: A Benchmark for Image Editing Models to Assess Object Category and Counting Under Scene Conservation

    [https://arxiv.org/abs/2610.28316](https://arxiv.org/abs/2610.28316)

    ABACUS是一个针对图像编辑模型数值能力的新基准，采用无VLM评判的评估协议和专用评分指标，首次准确测量了模型在保持场景的同时添加、移除或更改对象类别与数量的真实能力。

    

    基于指令的图像编辑模型如今可以在保持场景不变的情况下添加、移除指定数量的对象或更改其类别。然而，这些模型往往无法生成正确的数量，并且随着场景变得更加复杂和所要求的数量增加，这些失败急剧增多。当前的基准测试在很大程度上忽视了这一问题：它们使用整体相似度指标或本身不擅长计数的视觉语言模型（VLM）评判者来评估编辑结果，因此数值错误要么未被察觉，要么被错误评分。结果，当今编辑模型的真实数值能力从未被准确测量。我们通过ABACUS来解决这一问题，这是一个针对数值图像编辑的新基准、评估协议和专用评分指标。我们的协议是无评判者的：没有任何VLM或多模态大语言模型（MLLM）对编辑结果进行评分或计算对象数量（VLM仅用于确认少数模糊裁剪图像的类别）。相反，它在实例级别验证每次编辑。

    arXiv:2610.28316v1 Announce Type: cross  Abstract: Instruction-based image editing models can now add, remove a specified number of objects or change their category while preserving the scene. Yet these models often fail to produce the correct number, and these failures grow sharply as scenes become more complex and the requested count increases. Current benchmarks largely miss this problem: they score edits with holistic similarity metrics or with vision-language model (VLM) judges that are themselves poor at counting, so numerical errors go unnoticed or are graded incorrectly. As a result, the true numerical ability of today's editing models has never been accurately measured. We address this issue with ABACUS, a new benchmark, evaluation protocol and dedicated scoring metric for numerical image editing. Our protocol is judge-free: no VLM or MLLM scores an edit or counts objects (a VLM only confirms the category of rare ambiguous crops). Instead, it verifies each edit at the instance
    
[^51]: 披露引发披露：具有持久记忆的LLM智能体中的自我强化隐私循环

    Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory

    [https://arxiv.org/abs/2610.28315](https://arxiv.org/abs/2610.28315)

    该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。

    

    大语言模型智能体越来越多地使用持久记忆来个性化响应，而记忆中存储的个人属性一旦暴露便构成隐私风险。我们探究在对话中查阅记忆如何改变用户后续的发言内容，进而改变记忆最终会揭示的信息。在多会话模拟中，一个拥有24个个人属性的模拟用户与一个查阅或不查阅记忆的客服智能体对话，并在两种设置下均从完整累积的对话记录中推断这些属性。在四个客服模型上，使用记忆提升了响应质量，但同时也增加了经验性泄露——即从记录中能被正确推断出的属性值比例——且这一差距随会话次数增加而不断扩大。其根源在于普通的、非对抗性辅助过程中存在的一个循环：基于记忆的响应会将早期的属性值延续下去（常以预设的形式出现），而用户会确认这些值或不予纠正，从而使线索和智能体自身的猜测都被记录下来。

    arXiv:2610.28315v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly use persistent memory to personalize responses, and personal attributes in this memory pose a privacy risk if exposed. We ask how consulting memory during dialogue changes what users go on to say, and thus what the memory comes to reveal. In multi-session simulations, a simulated user with 24 personal attributes talks with a support agent that does or does not consult memory, and in both settings attributes are inferred from the full accumulated record. Across four support models, memory raises response quality but also empirical leakage, the share of attribute values correctly inferred from the record, by a margin that widens over sessions. The cause is a loop in ordinary, non-adversarial assistance. Memory-conditioned responses carry earlier values forward, often as presuppositions, and users affirm them or leave them uncorrected, so that cues and the agent's own guesses become recorded 
    
[^52]: 用于摊销贝叶斯半参数推断的去偏先验数据拟合网络

    Debiased Prior-Data Fitted Networks for Amortized Bayesian Semi-Parametric Inference

    [https://arxiv.org/abs/2610.28314](https://arxiv.org/abs/2610.28314)

    该论文为去偏先验数据拟合网络建立了理论基础，揭示了理想PFN的后验集中速率会隐式依赖干扰函数并随先验多样化而变慢，并基于有效影响函数提出标签校正与先验校正两种去偏策略以加速目标后验的集中速率。

    

    先验数据拟合网络（PFNs）作为端到端的摊销贝叶斯推断方法，近年来广受欢迎。然而，它们针对半参数目标（如缺失数据下的均值或因果效应）的统计行为仍知之甚少。在这项工作中，我们为去偏PFN建立了理论基础，使其能在一般半参数设置下对有限维目标进行推断。首先，我们证明，尽管理想的PFN相对于目标是贝叶斯最优的，其后验集中速率可能隐式依赖于（潜在）干扰函数的集中速率，并且随着先验变得更加多样化而变慢。随后，为了加快目标后验集中速率，我们基于有效影响函数提出了两种通用的去偏策略：(i) 标签校正和 先验校正。最后，我们证明去偏PFN 具备……

    arXiv:2610.28314v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) have recently gained popularity as an end-to-end approach to amortized Bayesian inference. Yet, their statistical behavior for semi-parametric targets, such as means under missingness or causal effects, remains poorly understood. In this work, we develop theoretical foundations for debiased PFNs to perform inference on finite-dimensional targets in general semi-parametric settings. First, we show that, despite being Bayes-optimal with respect to the target, an idealized PFN can exhibit a posterior concentration rate that implicitly depends on the concentration rates of (latent) nuisance functions and can become slower as the prior becomes more diverse. Then, to speed up the target posterior concentration rates, we introduce two generic debiasing strategies based on efficient influence functions: (i) a label correction and (ii) a prior correction. Finally, we show that the debiased PFNs (i) & (ii) posse
    
[^53]: 从16S rRNA序列推荐细菌培养条件

    Recommending Bacterial Cultivation Conditions from 16S rRNA

    [https://arxiv.org/abs/2610.28310](https://arxiv.org/abs/2610.28310)

    提出包含65,023个菌株的DSNI基准数据集，并设计了一种利用16S rRNA高变与保守区域结构特征的可变性门控编码器，仅以不到1%的参数开销将细菌培养条件推荐的联合准确率提升4.7%。

    

    从序列数据中推荐细菌培养条件是一个具有挑战性的结构化问题，其特点是缺乏标准化基准数据集且对序列结构信息的利用有限。为解决这一问题，我们提出了DSNI——一个包含65,023个细菌菌株的基准数据集，采用系统发育感知的数据划分方式，将16S rRNA序列与四种培养条件（培养基、温度、pH值、氧气）配对。利用16S rRNA的马赛克结构特征——其中高变区域支持菌株鉴别，而保守区域支持引物设计——我们提出了一种可变性门控编码器，该编码器基于生物学特征（区域类型、位置和局部复杂度）学习位置相关的重要性权重。门控模块仅引入极小的额外开销（1,280个参数，不到1%），同时在低数据量场景中提供了有价值的显式归纳偏置。实验表明，可学习的门控机制使扁平Transformer的联合准确率提升了4.7%，并与之形成互补。

    arXiv:2610.28310v1 Announce Type: new  Abstract: Recommending bacterial cultivation conditions from sequence data is a challenging structured problem characterized by missing standardized benchmarks and limited use of sequence structure. To address this, we introduce DSNI, a benchmark of 65,023 bacterial strains with phylogeny-aware splits, pairing 16S rRNA sequences with four cultivation conditions (media, temperature, pH, oxygen). Exploiting the mosaic structure of 16S rRNA, where hypervariable regions support discrimination and conserved regions support primer design, we propose a Variability-Gated Encoder that learns position-dependent importance weights conditioned on biological features (region type, position, and local complexity). The gating module adds minimal overhead (1,280 parameters, <1%) while providing an explicit inductive bias valuable in low-data regimes. Experiments demonstrate that the learnable gate improves flat Transformers by 4.7% in joint accuracy and complemen
    
[^54]: 基于流图的黎曼流形正规坐标学习

    Learning Normal Coordinates on Riemannian Manifolds with Flow Charts

    [https://arxiv.org/abs/2610.28302](https://arxiv.org/abs/2610.28302)

    提出训练条件可逆神经网络“流图”来近似黎曼流形上的正规坐标，并设计了非负且仅在坐标图为正规时才为零的高效损失函数，使指数映射和对数映射可通过快速欧氏运算近似，大幅降低计算成本。

    

    黎曼流形是处理具有非欧几里得结构问题的强大模型。近年来几何机器学习领域的工作主要集中在训练深度神经网络从高维数据中学习黎曼度量。然而，许多几何算法需要反复计算指数映射和对数映射，其高昂的计算成本限制了学习到的黎曼流形在实践中的实用性。我们提出通过训练一个称为流图的条件可逆神经网络来近似给定黎曼流形上各点周围的正规坐标，以解决这一计算成本问题。通过这种方法，指数映射和对数映射可以在模型学习到的坐标空间中通过快速的欧氏运算来近似。我们的主要贡献是一个用于训练流图的高效损失函数，该函数是非负的，并且在适当的定义域假设下，当且仅当该坐标图是正规坐标时损失为零。

    arXiv:2610.28302v1 Announce Type: new  Abstract: Riemannian manifolds are powerful models for problems with non-Euclidean structure. Recent work in geometric machine learning has focused on training deep neural networks to learn Riemannian metrics from high-dimensional data. However, many geometric algorithms require repeated evaluations of the exponential and logarithmic maps, whose high computational cost restricts the utility of learned Riemannian manifolds in practice. We propose to address this cost by training a conditional invertible neural network, called a flow chart, to approximate normal coordinates around points on a given Riemannian manifold. With this approach, the exponential and logarithmic maps can be approximated by fast Euclidean operations in the model's learned coordinate space. Our main contribution is an efficient loss function for training flow charts that is nonnegative and, under suitable domain assumptions, is zero precisely when the chart is normal. Across s
    
[^55]: 自适应重放优化：在修订与重采样之间分配测试时计算

    Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling

    [https://arxiv.org/abs/2610.28294](https://arxiv.org/abs/2610.28294)

    ARR 是一种无需训练的自适应方法，通过检测解的修订是否达到不动点，在“继续改进现有解”与“重新生成新解”之间动态分配测试时计算，从而在多个模型和基准上以同等计算量达到与多数投票相当的效果。

    

    测试时扩展将额外的推理计算用于优化现有解或生成新解。这引出了一个根本性问题：应该继续改进当前解，还是停止改进并尝试一个全新的解？现有方法依赖固定的优化调度、自我验证判断或学习到的控制器来做出这一决策，这可能会浪费计算，或依赖可能不正确、需要额外训练的信号。我们提出自适应重放优化，这是一种无需训练的方法，能够检测平稳修订何时达到不动点，并将释放出来的计算重新分配给新的独立样本；重放解码将当前解视为草稿，并并行验证未发生变化的 token，从而实现高效的不动点检测。在来自五个家族的七个模型和七个基准测试上，ARR 在同等计算量下于五个数学推理基准中的三个上与多数投票的表现相当。

    arXiv:2610.28294v1 Announce Type: new  Abstract: Test-time scaling allocates additional inference compute to refining existing solutions or generating new ones. This raises a fundamental question: Should I keep improving this solution, or stop working on it and try a completely new solution? Existing methods make this decision using fixed refinement schedules, self-verification judgments, or learned controllers, which can waste compute or rely on signals that may be incorrect or require additional training. We introduce Adaptive Replay Refinement (ARR), a training-free method that detects when a stationary revision reaches a fixed point and reallocates the released compute to a new independent sample; replay decoding treats the current solution as a draft and verifies unchanged tokens in parallel, enabling efficient fixed-point detection. Across seven models from five families and seven benchmarks, ARR matches majority voting on three of five mathematical reasoning benchmarks at equal 
    
[^56]: 局部工作，全局进展：AI智能体协作的收敛理论

    Local Work, Global Progress: A Convergence Theory of AI Agent Collaboration

    [https://arxiv.org/abs/2610.28286](https://arxiv.org/abs/2610.28286)

    该论文首次为LLM多智能体协作建立了收敛性理论：将协作求解形式化为块优化，每个智能体基于局部目标加修正项改进共享解的一部分，由协调者检验并合并提案，证明了可达到误差下限的线性收敛性，并在机器人路径规划任务中验证了多智能体团队总能找到无碰撞方案且优于单一智能体。

    

    由大语言模型（LLM）智能体组成的团队正越来越多地被应用于复杂任务，但它们的协作协议大多是凭经验设计的，缺乏收敛性保证。我们将协作式问题求解形式化为块优化（block optimization）：每个智能体改进共享解的一部分，其指导依据是自身的局部目标，外加一个刻画其修改如何影响其他部分的修正项。一个协调者会依据全局目标检验每个提案，并合并互不干扰的提案。我们证明了线性收敛性，收敛至一个误差下限，当修正项精确时该误差下限会消失。对于相互干扰的提案，只要对其组合进行了检验，也可以进行合并；此时收敛速率取决于提案实际组合的效果，而非最坏情况的交互。在使用GPT-OSS-120B智能体的机器人路径规划任务中，三智能体团队在每次运行中都能找到无碰撞的规划方案，效果优于单一智能体控制所有……（原文此处截断）。

    arXiv:2610.28286v1 Announce Type: cross  Abstract: Teams of large language model (LLM) agents are increasingly applied to complex tasks, but their collaboration protocols are mostly designed empirically, without convergence guarantees. We formalize collaborative problem solving as block optimization: each agent improves one part of a shared solution, guided by its local objective plus a correction for how its changes affect the other parts. A coordinator checks every proposal against the global objective and merges proposals that do not interfere with each other. We prove linear convergence up to an error floor that vanishes when the corrections are exact. Interfering proposals can also be merged once their combination has been checked; the rate then depends on how well the proposals actually combine, not on worst-case interactions. With GPT-OSS-120B agents on robot routing, a three-agent team finds collision-free plans in every run, better than those of a single agent that controls al
    
[^57]: 面向稀疏变分高斯过程的几何感知块选择方法

    Geometry-Aware Block Selection for Sparse Variational Gaussian Processes

    [https://arxiv.org/abs/2610.28256](https://arxiv.org/abs/2610.28256)

    本文提出一种几何感知的块选择方法，通过在固定诱导点预算下选取能收紧变分下界的块对角缩放矩阵中的块，从而提升稀疏变分高斯过程的测试RMSE和NLL性能。

    

    稀疏变分高斯过程（SVGPs）是一种被广泛使用的、有原则的可扩展高斯过程建模方法。其成功源于其理论上可解释的结构：它们在支持可扩展的随机优化的同时，为完整高斯过程的边缘似然提供一个下界。近期的工作探索了对后验条件分布的修改，并证明此类扩展能够改进这一成熟的框架。特别是，引入块对角结构的缩放矩阵可以得到更紧的高斯过程边缘似然下界。我们研究了在固定诱导点预算下，选择能够收紧变分目标的块是否可以改善测试RMSE和NLL。由于缩放矩阵是通过完整残差协方差矩阵对数行列式的上界对变分下界做出贡献的，我们的工作做出了以下贡献……

    arXiv:2610.28256v1 Announce Type: cross  Abstract: Sparse variational Gaussian processes (SVGPs) are a widely used, principled approach to scalable Gaussian process modelling. Their success stems from their theoretically interpretable structure: they provide a lower bound on the marginal likelihood of the full GP while allowing scal- able stochastic optimization. Recent work has explored modifications of the posterior conditional distribution and has shown that such extensions can improve upon this well-established framework. In particular, introducing a block-diagonal structured scal- ing matrix leads to a tighter lower bound on the GP marginal likelihood. We investigate whether selecting blocks that tighten the variational objective improves test RMSE and NLL at a fixed inducing-point budget. Since the scaling matrix contributes to the variational lower bound through an upper bound on the log-determinant of the full residual covariance matrix, our work makes the following contributio
    
[^58]: LoRA适配的视觉基础模型在脑部MRI中的性能趋同与机制分化

    Convergent Performance, Divergent Mechanisms in LoRA-Adapted Vision Foundation Models for Brain MRI

    [https://arxiv.org/abs/2610.28255](https://arxiv.org/abs/2610.28255)

    研究发现，不同预训练目标的视觉基础模型经LoRA适配后，在脑部MRI分割任务上虽取得几乎相同的性能，但其内部表征变化机制截然不同，且跨任务迁移能力存在显著差异。

    

    通过LoRA将视觉基础模型（VFMs）适配到医学影像领域已成为标准做法，然而模型选择几乎完全由下游任务指标驱动，性能相近的骨干网络往往被视为可互换的。我们研究了相同的性能是否意味着相同的内部表征变化。我们分析了四个ViT-B/16骨干网络——DINOv3、SigLIP2、MerMED-FM和BiomedCLIP——它们分别采用不同的预训练目标（自蒸馏 vs. 对比式视觉-语言学习）和不同的数据域（自然图像 vs. 医学图像）进行预训练。我们将每个模型适配到脑肿瘤分割任务，并在涵盖分割任务与阿尔茨海默病分类的六个数据集上进行评估。通过注意力分歧分析、权重变化分析和激活修补（activation patching）方法，我们发现这四个模型通过截然不同的内部动态收敛到相似的分割性能（平均Dice系数78-81%），但在跨任务迁移方面表现出现分化：（摘要在此处被截断）

    arXiv:2610.28255v1 Announce Type: cross  Abstract: Adapting Vision Foundation Models (VFMs) to medical imaging via LoRA has become standard practice, yet model selection is guided almost entirely by downstream task metrics, treating backbones with similar performance as interchangeable. We investigate whether equal performance implies equal internal representational shift. We analyse four ViT-B/16 backbones, DINOv3, SigLIP2, MerMED-FM, and BiomedCLIP, pre-trained with different objectives (self-distillation vs. contrastive vision-language) and on different data domains (natural vs. medical images). We adapt each to brain tumour segmentation and evaluate on six datasets covering both segmentation and Alzheimer's disease classification. Using attention divergence, weight-change analysis, and activation patching, we find that the four models converge to similar segmentation performance (mean Dice 78-81%) through markedly different internal dynamics, yet diverge in cross-task transfer: seg
    
[^59]: TPO：面向OOD直接偏好优化的三层分布鲁棒学习

    TPO: Tri-level Distributionally Robust Learning for OOD Direct Preference Optimization

    [https://arxiv.org/abs/2610.28229](https://arxiv.org/abs/2610.28229)

    提出了TPO框架，通过三层分布鲁棒学习同时建模组级、提示级和偏好级的分布不确定性，使DPO在分布外偏好漂移场景下更加鲁棒。

    

    直接偏好优化在观测到的偏好比较数据上最小化经验风险，这隐含地假设训练时的偏好分布与部署时的偏好分布相匹配。然而在实践中，这种独立同分布假设经常失效，因为偏好分布漂移呈现出自然的三层层级结构：组级漂移、提示级漂移和偏好级漂移。这种多层次的分布失配对大语言模型的安全部署构成了关键挑战，但现有的鲁棒DPO方法通常要么对联合偏好分布的不确定性进行建模，要么仅关注单一来源的漂移，导致对OOD偏好漂移的层级结构探索不足。我们提出了TPO，一个面向OOD偏好对齐的三层分布鲁棒框架，它通过层级化结构的矩形模糊集联合建模所有三个层次的分布不确定性……

    arXiv:2610.28229v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) minimizes empirical risk over observed preference comparisons, implicitly assuming that the training preference distribution matches the deployment preference distribution. In practice, this i.i.d. assumption often fails, as preference distribution shift exhibits a natural hierarchical structure across three levels: group-level shift, prompt-level shift, and preference-level shift. Such multi-level distributional mismatch poses a critical challenge for safe LLM deployment, yet existing robust DPO methods typically either model uncertainty over the joint preference distribution or focus on individual sources of shift, leaving the hierarchical structure of OOD preference shifts underexplored. We propose TPO, a tri-level distributionally robust framework for OOD preference alignment, which jointly models all three levels of distributional uncertainty through a hierarchically structured rectangular ambi
    
[^60]: ExeBench-FHD：解决反编译器评估中的保真度问题

    ExeBench-FHD: Addressing Fidelity Issues in Decompiler Evaluation

    [https://arxiv.org/abs/2610.28212](https://arxiv.org/abs/2610.28212)

    该论文分析了广泛使用的反编译器评测基准ExeBench中影响评估保真度的九个问题，并提出了向后兼容的修订版ExeBench-FHD，通过新增16个字段更忠实地评估基于LLM的反编译器。

    

    ExeBench是评估基于大语言模型（LLM）的反编译器最广泛使用的基准之一。我们分析了ExeBench并识别出九个影响测试对反编译器测量保真度的问题。其中三个问题阻碍了LLM反编译器从ExeBench的函数中恢复C代码：整数和浮点类型混合时的参数顺序歧义、被优化汇编代码可能丢弃的未使用参数、以及存储在函数汇编代码之外的常量。另外六个问题影响测试有效性：缺少返回语句的非void函数、占位符类型、被优化器移除的static或inline函数、命名为main的函数、平凡简单的函数体以及被截断的定义。我们提出了ExeBench-FHD，这是ExeBench的一个向后兼容的修订版本，新增了16个字段：调用点汇编代码、数据段、测试包装器、链接二进制反汇编代码以及五个警告标志。在ExeBench中，38.9%的函数至少带有五个警告标志中的一个；剔除……（摘要在此处截断）

    arXiv:2610.28212v1 Announce Type: cross  Abstract: ExeBench is one of the most widely used benchmarks for evaluating LLM-based decompilers. We analyze ExeBench and identify nine issues that affect how faithfully a test measures a decompiler. Three of them prevent LLM decompilers from recovering C code from ExeBench's function. These are ambiguous parameter order when integer and floating-point types are mixed, unused parameters that optimized assembly may drop, and constants stored outside the function's assembly code. Six more affect test validity: non-void functions without a return, placeholder types, static or inline functions removed by the optimizer, functions named main, trivial bodies and truncated definitions. We present ExeBench-FHD, a backward-compatible revision of ExeBench that adds 16 new fields: call-site assembly, data sections, test wrappers, linked-binary disassembly and five warning flags. In ExeBench, 38.9% of functions have at least one of the five flags; dropping 
    
[^61]: JAXbounce：一个用于超越低维环境的主动推理的连续控制类Pong基准测试

    JAXbounce: A Continuous-Control Pong-Like Benchmark for Active Inference Beyond Low Dimensional Environments

    [https://arxiv.org/abs/2610.28203](https://arxiv.org/abs/2610.28203)

    提出了JAXbounce——一个完全可微分、可并行化的连续控制类Pong基准测试，通过状态向量与像素观测、多种噪声条件及超过万次的智能体模拟，将主动推理的评估从低维离散环境拓展到更复杂的高维连续环境。

    

    我们介绍了JAXbounce，这是一个连续控制的类Pong基准测试，旨在评估超越离散、低维环境的主动推理。JAXbounce将离散时间下的连续状态与随机动力学相结合，并配备灵活且可组合的观测模型，允许将环境表示为状态向量或像素网格，且可通过我们提供的丰富噪声模型家族对观测进行扰动。该环境在JAX中完全可微分且可并行化。我们在该环境中运行了超过10,000次智能体模拟，比较了基线确定性解决方案、具有无迹卡尔曼滤波感知和期望自由能规划的基于状态的主动推理智能体，以及以预训练神经网络编码器作为测量模型的基于像素的智能体。所有智能体在三个难度递增的场景和八种不同噪声条件下进行评估。我们的实验刻画了连续状态智能体性能随环境变化的特征。

    arXiv:2610.28203v1 Announce Type: cross  Abstract: We introduce JAXbounce, a continuous control pong-like benchmark designed to evaluate active inference beyond discrete, low dimensional environments. JAXbounce combines continuous states and stochastic dynamics in discrete time with flexible and composable observation models that allow representing the environment as a state vector or a pixel grid, corruptible through the rich family of noise models we provide. It is fully differentiable and parallelisable in JAX. We run over 10,000 agent simulations on our environment comparing a baseline deterministic solution, to a state-based active inference agent with Unscented Kalman Filter perception and expected free energy planning, and a pixel based agent whose measurement model is a pretrained neural network encoder. All agents are evaluated across three scenarios of increasing difficulty with eight different noise conditions. Our experiments characterise how performance of continuous state
    
[^62]: 面向电力负荷预测的时间序列基础模型参数高效适配方法研究

    Parameter-Efficient Adaptation of Time Series Foundation Models for Electrical Load Forecasting

    [https://arxiv.org/abs/2610.28200](https://arxiv.org/abs/2610.28200)

    本文系统比较了预训练推理、全量微调与LoRA参数高效适配在电力负荷预测中的精度与效率，发现预训练时间序列基础模型已优于传统基线，并评估了紧凑适配方法能否在不同负荷聚合层级、分布内外设置及不同模型家族上带来一致的增益。

    

    精确的电力负荷预测支撑着国家、区域、城市和建筑各层级的电力系统运行，但反复开发针对特定目标的模型成本高昂。时间序列基础模型（TSFMs）提供了可复用的预训练骨干网络，尽管多领域预训练可能无法充分捕捉负荷特有的特征。本文在四个电力负荷聚合层级上，从预测精度和适配效率两方面比较了预训练推理、全量微调和低秩适配（LoRA）方法。研究以MOIRAI 1.0 Large作为主要骨干网络，同时使用Chronos-2来评估研究结论能否跨模型家族迁移。评估涵盖分布内（ID）和分布外（OOD）两种设置，并考察紧凑的适配方法能否在强大的预训练检查点之上带来一致的性能提升。预训练基础模型优于所实现的季节性朴素预测和XGBoost基线方法。

    arXiv:2610.28200v1 Announce Type: cross  Abstract: Accurate electrical load forecasting supports power-system operation across countries, regions, cities, and buildings, but repeated development of target-specific models is costly. Time series foundation models (TSFMs) provide reusable pretrained backbones, although multi-domain pretraining may not fully capture load-specific characteristics. This paper compares pretrained inference, full fine-tuning, and low-rank adaptation (LoRA) in terms of forecasting accuracy and adaptation efficiency across four electrical load aggregation levels. MOIRAI 1.0 Large serves as the primary backbone, while Chronos-2 assesses whether the findings transfer across model families. The evaluation covers both in-distribution (ID) and out-of-distribution (OOD) settings and examines whether compact adaptation provides consistent gains over strong pretrained checkpoints. Pretrained foundation models outperform the implemented seasonal naive and XGBoost baselin
    
[^63]: 重新审视以对象为中心的强化学习：从绝对表示到相对表示

    Reframing Object-Centric Reinforcement Learning: From Absolute to Relative Representations

    [https://arxiv.org/abs/2610.28197](https://arxiv.org/abs/2610.28197)

    该论文提出将强化学习中的对象表示从绝对全局坐标重构为固定于锚点的相对坐标（ORCA），并设计了ORBiT架构来处理这种相对输入，从而提升MLP和Transformer智能体的性能及其对场景平移的鲁棒性。

    

    以对象为中心的强化学习智能体通常接收每个对象在固定全局坐标系中的位置，然而它们的决策往往取决于对象之间的相对位置。这种绝对视角常常无法捕捉到与智能体行为最相关的局部上下文。因此，智能体必须在每种配置可能出现的每个位置重新学习相同的响应，而当场景仅仅发生平移时，其性能就会大幅下降。在这项工作中，我们提出转向固定于锚点的对象相对坐标（ORCA），即在一个相对参考系中对实体进行描述。为了实现这一想法，我们引入了对象相对归纳偏置Transformer（ORBiT），这是一种基于Transformer的架构，用于在强化学习中处理锚点相对输入。我们的结果表明，这些相对表示能够提升MLP和Transformer智能体的性能。此外，我们……

    arXiv:2610.28197v1 Announce Type: cross  Abstract: Object-centric reinforcement learning agents typically receive each object's position in a fixed global frame, even though their decisions usually depend on where objects are relative to one another. These absolute views often fail to capture the local context that is most relevant to the agent's behavior. As a result, an agent must relearn the same response at every location where a configuration can occur, and its performance collapses when the scene is merely shifted. In this work, we propose a shift toward Object-Relative Coordinates fixed to Anchor (ORCA), where entities are described in a relative frame of reference. To operationalize this idea, we introduce Object-Relative inductive-Bias Transformers (ORBiT), a transformer-based architecture for processing anchor-relative inputs in reinforcement learning. Our results suggest these relative representations improve performance for both MLP and transformer agents. Furthermore, we d
    
[^64]: 有吸引力的面孔会使文生图奖励模型和排行榜产生偏差

    Attractive Faces Skew Text-to-Image Reward Models and Leaderboards

    [https://arxiv.org/abs/2610.28195](https://arxiv.org/abs/2610.28195)

    研究发现人脸吸引力在人类偏好判断中会压倒图像质量本身，这一偏差会扭曲文生图模型的奖励模型和排行榜，使AI生成的有吸引力面孔被排在真实但不吸引人的面孔之上。

    

    人类偏好现已成为文生图模型评估、引导、微调和排名的核心。然而，人类偏好对图像生成中偏差的易感性仍未得到充分探索。我们发现了一个关键观察：对于人脸而言，感知到的吸引力在偏好判断中会压倒图像质量，即使在比较保真度完全相同的真实照片时也是如此。在一项受控用户研究中，标注者将有吸引力的面孔评为更高质量，即使不吸引人的真实面孔与有吸引力的AI生成面孔进行比较时也是如此。我们研究了这种偏差的下游影响。首先，基于用户研究数据模拟Elo排名显示，吸引力可以逆转排行榜的排名，使合成的有吸引力面孔排在真实的、不吸引人的面孔之上。其次，同样的偏差在人类偏好模型中再次出现。通过单独编辑40个标注的面部属性，我们发现吸引力是最强的（影响因素）……

    arXiv:2610.28195v1 Announce Type: cross  Abstract: Human preference is now central to the evaluation, guidance, finetuning, and ranking of text-to-image models. However, the susceptibility of human preferences to bias in image generation remains underexplored. We identify a key observation: for human faces, perceived attractiveness overrides image quality in preference judgments, even when comparing real photographs of identical fidelity. In a controlled user study, annotators rank more attractive faces as higher quality, even when unattractive real faces are compared with attractive AI-generated ones. We examine the downstream consequences of this bias. Firstly, simulating Elo rankings from user study data shows attractiveness can invert a leaderboard's ranking, elevating synthetic attractive faces over real, unattractive ones. Secondly, the same bias reappears in the human preference models. Editing 40 annotated facial attributes in isolation, we find that attractiveness is the stron
    
[^65]: 自适应回拉：在线策略蒸馏中惩罚跳变处的监督重加权

    Adaptive Pullback: Reweighting Supervision at Penalty Jumps in On-Policy Distillation

    [https://arxiv.org/abs/2610.28194](https://arxiv.org/abs/2610.28194)

    该论文发现在线策略蒸馏中存在“惩罚-置信度张力”现象——惩罚跳变处惩罚上升而教师置信度下降，并据此提出AP-OPD方法，在每个检测到的跳变处对监督信号进行自适应重加权。

    

    在线策略蒸馏（OPD）通过采样token的学生-教师对数比率来监督学生模型生成的回复。正的对数比率会惩罚学生赋予比教师更高概率的token，但其当前数值无法区分持续的高惩罚和沿回复方向的突然增加。我们通过比较同一回复内连续的token块来考察这些局部变化。我们的诊断方法识别出了“惩罚跳变”——即在保留token上持续存在的平均惩罚的突然增加。在这些跳变处，惩罚上升的同时教师置信度却下降。我们将这种模式称为“惩罚-置信度张力”。而进入高惩罚块的边界并未表现出可比的置信度变化，这表明局部变化提供了超出当前惩罚值之外的信息。受此观察启发，我们提出了自适应回拉OPD（AP-OPD）。在每个检测到的跳变处，AP-OPD利用前一块……

    arXiv:2610.28194v1 Announce Type: cross  Abstract: On-policy distillation (OPD) supervises student-generated responses using sampled-token student-teacher log-ratios. A positive log-ratio penalizes a token that the student assigns higher probability to than the teacher, but its current value does not distinguish sustained high penalties from abrupt increases along a response. We examine these local changes by comparing consecutive blocks of tokens within the same response. Our diagnostic identifies penalty jumps, abrupt increases in average penalty that persist on held-out tokens. Across these jumps, penalties rise while teacher confidence decreases. We call this pattern penalty-confidence tension. Boundaries into high-penalty blocks do not exhibit comparable confidence changes, showing that local changes provide information beyond the current penalty value. Motivated by this observation, we propose Adaptive Pullback OPD (AP-OPD). At each detected jump, AP-OPD uses the preceding block 
    
[^66]: 离线让在线更好：面向视频生成的校准缓存压缩

    Offline Makes Online Better: Calibrated Cache Compression for Video Generation

    [https://arxiv.org/abs/2610.28192](https://arxiv.org/abs/2610.28192)

    该论文提出CALM方法，利用离线校准捕捉不同注意力层和头对标记合并的敏感性差异，以注意力输出误差作为轻量级代理指标，为在线视频生成的KV缓存压缩实现自适应的层-头合并分配，从而以更低开销获得更高质量的视频。

    

    标记合并（Token merging）是一种有效降低自回归视频生成中KV缓存内存消耗的方法，其通过聚类时间上冗余的标记并仅保留一小组代表性标记来实现。然而，现有方法通常在所有注意力层和头上应用统一的合并比例。我们观察到，不同的层和头对标记合并表现出显著不同却相互关联的敏感性，并且这些敏感性模式在不同提示词之间保持一致。基于这一观察，我们提出了CALM（Calibrated Allocation for Layer-Head Merging，层-头合并的校准分配），利用离线校准来改进在线缓存压缩。由于穷举评估所有可能的层-头分配方案的代价过于高昂，为实现高效校准，我们证明注意力输出误差与端到端视频质量强相关，因此可以作为分配风险的轻量级代理指标……

    arXiv:2610.28192v1 Announce Type: cross  Abstract: Token merging is an effective approach to reducing KV-cache memory consumption in autoregressive video generation by clustering temporally redundant tokens and retaining only a small set of representatives. Existing methods, however, typically apply a uniform merging ratio across all attention layers and heads. We observe that different layers and heads exhibit substantially different yet jointly correlated sensitivities to token merging, and that these sensitivity patterns remain consistent across prompts. Based on this observation, we introduce CALM, a Calibrated Allocation for Layer-Head Merging that uses offline calibration to improve online cache compression. Exhaustively evaluating all possible layer-head allocations is prohibitively expensive. To enable efficient calibration, we show that attention-output error strongly correlates with end-to-end video quality and can therefore serve as a lightweight proxy for allocation risk as
    
[^67]: Foreword：基于雅可比透镜的LLM测试失败的生成前预测与表征

    Foreword: Pre-Generation Prediction and Characterisation of LLM Test Failures using the Jacobian Lens

    [https://arxiv.org/abs/2610.28187](https://arxiv.org/abs/2610.28187)

    提出Foreword框架，利用雅可比透镜将LLM中间激活读取为词汇标记分数来表征输入，并结合通过与失败参考样本及稀疏逻辑回归探针或双高斯混合模型，实现测试失败的生成前预测与失败原因表征。

    

    测试基于大语言模型（LLM）的软件成本高昂：每个输入都需要多次采样响应，且每次响应都需要由昂贵的判定器（oracle）进行评估。此外，每次提示词或模型的变更都要求重新测试。生成前的充分性度量通过基于模型内部状态对输入按失败风险进行排序，从而降低这一成本。然而，当前最先进的方法Clotho仅对通过测试的参考输入进行建模，且其隐藏状态表示无法解释输入为何失败。我们提出了Foreword，一个建立在两个核心思想之上的框架：同时从通过和失败的参考样本中学习，并通过雅可比透镜表征输入——这是一种可解释性技术，将中间激活读取为词汇标记上的分数。Foreword将这些表示与两种估计器相结合：稀疏逻辑回归探针或双高斯混合模型，后者还能提供参考密度（用于标记参考样本未覆盖的输入）以及失败类别成分……（摘要原文在此处截断）

    arXiv:2610.28187v1 Announce Type: cross  Abstract: Testing LLM-based software is expensive: each input needs several sampled responses, each judged by a costly oracle. Further, every prompt or model change calls for re-testing. Pre-generation adequacy measures cut this cost by ranking inputs by failure risk based on the model's internal state. However, the state of the art, Clotho, models only passing reference inputs, and its hidden-state representation says nothing about why inputs fail. We present Foreword, a framework built on two ideas: learn from both passing and failing references, and represent inputs through the Jacobian lens (J-lens), an interpretability technique that reads intermediate activations out as scores over vocabulary tokens. Foreword pairs these representations with two estimators: a sparse logistic-regression probe or dual Gaussian Mixture Models (dual GMMs), which also yield a reference density (to flag inputs the references do not cover) and failing-class compo
    
[^68]: 基于智能电表数据的峰值同时率最大化客户聚类

    Coincidence-Maximizing Customer Clustering Using Smart Meter Data

    [https://arxiv.org/abs/2610.28178](https://arxiv.org/abs/2610.28178)

    该论文证明即使只有两个组，最大化加权峰值同时率的客户分组问题也是强NP难的，并提出了一种交替确定组峰值时刻与精确重新分配客户的多项式时间固定峰值精化算法，可严格改进加权同时率，并在四个公开智能电表数据集上验证了方法的有效性。

    

    配电网规划依赖于客户群体内的峰值需求同时率，然而基于相似性的智能电表数据聚类并不能直接对其进行优化。我们将客户分组问题形式化为最大化各组同时系数的加权平均值，其中权重为每组个体峰值需求之和在每日总峰值需求中所占的份额。我们证明，即使只有两个组，联合优化客户归属与组峰值时刻的问题也是强NP难的。因此，我们开发了一种适用于任何可行划分的固定峰值精化程序。该程序交替进行两个步骤：识别各组峰值时刻，以及在最小组规模约束下对客户进行精确重新分配。每次多项式时间的重新分配都在最大化一个在当前划分处取到紧值的下界，因此被接受的更新能够严格改进加权同时率。我们通过对个体峰值时刻分布进行聚类来初始化精化过程，并在四个公开的智能电表数据集（27,654个用户……

    arXiv:2610.28178v1 Announce Type: new  Abstract: Electricity distribution planning depends on peak-demand coincidence within customer groups, yet similarity-based clustering of smart-meter data does not optimize it directly. We formulate customer grouping to maximize the average of group coincidence factors, weighted by each group's daily share of summed individual peak demand. We prove that jointly optimizing memberships and group peak times is strongly NP-hard even for two groups. We therefore develop a fixed-peak refinement procedure applicable to any feasible partition. It alternates between identifying group peak times and exactly reassigning customers under minimum group-size constraints. Each polynomial-time reassignment maximizes a lower bound tight at the current partition, so accepted updates strictly improve weighted coincidence. We initialize refinement by clustering individual peak-time distributions and evaluate the method on four public smart-meter panels (27,654 custome
    
[^69]: 从代码生成器到工程助手：面向PLC编程的工厂感知智能体

    From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming

    [https://arxiv.org/abs/2610.28160](https://arxiv.org/abs/2610.28160)

    该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。

    

    可编程逻辑控制器（PLC）编程仍然是一项以IDE为中心的手动任务，工程师需要在文档、项目文件和代码编辑器之间频繁切换上下文。近期研究表明，检索增强的大语言模型能够以较高的编译通过率生成符合厂商规范的Structured Text代码，但这些系统将代码生成视为一次性流水线：工程师提交查询后获得代码，系统对特定工厂没有持久化的知识。我们提出了一个工厂感知的对话式编程助手，该系统在三个方向上扩展了这一范式：工厂特定上下文的摄取、以对话为先的解释与调试交互模式，以及从对话到部署的分级工具访问阶梯。该系统专为真实工厂基础设施设计，并确保工程师对每个会产生现实世界影响的步骤保持控制权。

    arXiv:2610.28160v1 Announce Type: cross  Abstract: Programmable Logic Controller (PLC) coding remains a manual, IDE-centric task that demands constant context switching between documentation, project files, and code editors. Recent work has demonstrated that retrieval-augmented large language models can generate vendor-compliant Structured Text at high compilation rates \cite{pred}, yet these systems treat code generation as a one-shot pipeline: the engineer submits a query and receives code, with no persistent knowledge of the specific plant. We present a plant-aware conversational copilot that extends this paradigm in three directions: plant-specific context ingestion, a chat-first interaction model for explanation and debugging, and a graduated tool-access ladder from chat to deployment. The system is designed for real plant infrastructure and keeps the engineer in control of each step that produces real-world effects.
    
[^70]: 预测受限访问条件下学习推理的静默失效：测量物理与先验信息能解释什么

    Predicting silent failures of learned inference under restricted access: what measurement physics and prior information explain

    [https://arxiv.org/abs/2610.28155](https://arxiv.org/abs/2610.28155)

    该论文提出“盲区集合”概念用于刻画学习推理中不依赖真值诊断无法察觉的静默失效，证明线性高斯模型下静默失效风险可精确分解为非检测与非覆盖，并通过线性化VAE先验构建评分，在真实X射线CT数据的前瞻性研究中成功预测了学习推理系统的实际盲性（Spearman ρ=0.79和0.84）。

    

    学习先验被用于重建测量几乎无法到达的区域，并在那里报告不确定性。它们可能会静默地失败：所报告的置信区间排除了真实值，而所有不依赖真值的诊断都保持在噪声范围内。我们将这类错误称为“盲区集合”，并探究是否能在看到目标数据之前预测它们。对于一个在先验上平均后规范正确的线性高斯模型，静默失效的风险可以精确地分解为非检测与非覆盖两个因子，而这一性质对于一般的学习系统并不成立；我们通过线性化VAE先验构建了一个近似该参考基准的评分。在一项针对真实X射线CT物体（2DeteCT）在模拟受限访问条件下的前瞻性研究中——所有预测在任何试验开始之前都以哈希形式提交——该评分对两个学习推理系统（潜在拉普拉斯方法和集合卡尔曼反演）的实际盲性进行了排序，在240次试验中Spearman相关系数ρ分别为0.79和0.84……

    arXiv:2610.28155v1 Announce Type: cross  Abstract: Learned priors are used to reconstruct regions that measurements barely reach, and to report uncertainty there. They can fail silently: the reported interval excludes the truth while every ground-truth-free diagnostic stays within its noise. We call such errors the blind set and ask whether they can be predicted before the target data are seen. For a correctly specified linear-Gaussian model averaged over the prior, the risk of a silent failure factorises exactly into non-detection and non-coverage, which does not hold for learned systems in general; we build a score that approximates this reference by linearising a VAE prior. In a prospective study on real X-ray CT objects (2DeteCT) under simulated restricted access, with predictions committed by hash before any trial, the score ranked the realised blindness of two learned inference systems (latent Laplace and ensemble Kalman inversion) with Spearman $\rho = 0.79$ and 0.84 over 240 pr
    
[^71]: 生成式柔性对接的课程学习

    Curriculum Learning for Generative Flexible Docking

    [https://arxiv.org/abs/2610.28145](https://arxiv.org/abs/2610.28145)

    该论文提出将课程学习应用于生成式柔性对接任务，在训练中逐步引入大构象变化样本，以应对从apo到holo结构的长尾分布偏移这一核心挑战，从而显著提升现有生成式对接方法的性能。

    

    柔性对接是一项非常实用的任务，旨在根据蛋白质的失活（apo）结构和配体的SMILES字符串，预测蛋白质-配体复合物的联合构象。基于扩散模型和流模型的生成式方法通过对配体位姿和蛋白质构象在apo输入条件下的后验分布进行建模，使这一任务变得可行。然而，核心挑战依然是从apo到holo结构的长尾分布偏移问题，即少部分蛋白质会发生较大的构象变化。在这项工作中，受到以下实际观察的启发——降低大偏移样本的权重可以提升泛化能力，以及长尾样本由真正困难的样本和噪声样本混合构成的特性——我们提出在该任务中使用课程学习，在训练过程中逐步引入大构象变化的样本。通过线性进度函数，我们证明了一种现有方法可以得到显著增强。

    arXiv:2610.28145v1 Announce Type: new  Abstract: Flexible docking is a highly practical task, aiming to predict the joint configuration of a protein-ligand complex from a protein apo structure and a ligand SMILES string. Diffusion and flow-based generative models have made this task tractable by modeling the posterior distribution over the ligand pose and the protein conformation conditioned on the apo input. The central challenge, however, remains the long-tailed apo-to-holo structure shift, where a small fraction of protein undergo large conformational changes. In this work, motivated by the practical observation that downweighting large shift examples improves generalization, and by the mixed nature of the long tail consisting of genuinely difficult examples and noisy ones, we propose to use curriculum learning for this task, gradually incorporating large conformational change examples during training. With a linear pacing function, we show that an existing method can be strongly en
    
[^72]: 不动集与计算相关均衡的困难性

    Fixed Sets and the Hardness of Computing Correlated Equilibria

    [https://arxiv.org/abs/2610.28141](https://arxiv.org/abs/2610.28141)

    本文证明计算多面体博弈的近似相关均衡是PPAD完全的（与纳什均衡同等难度），挑战了相关均衡比纳什均衡更容易求解的传统观点，并通过提出“不动集”这一不动点的自然推广建立了相关均衡与不动点理论的概念性联系。

    

    我们证明，计算多面体博弈的近似相关均衡是PPAD完全的，因此在多项式时间内等价于寻找纳什均衡。这一结果挑战了博弈论中的传统观点，即相关均衡是比纳什均衡更容易处理的对象。从概念层面，我们将相关均衡与不动点的一个自然推广联系起来，我们称之为“不动集”。连续函数 f 的不动集是一个非空有限集 S，满足 S ⊆ conv f(S)。我们的主要技术贡献是证明了即使达到逆多项式精度，计算不动集仍是PPAD困难的。

    arXiv:2610.28141v1 Announce Type: cross  Abstract: We show that computing an approximate correlated equilibrium of a polytope game is PPAD-complete, and thus polynomial-time equivalent to finding a Nash equilibrium. This challenges the conventional view in game theory that correlated equilibria are more tractable objects than Nash equilibria. From a conceptual standpoint, we connect correlated equilibria to a natural generalization of fixed points, which we call fixed sets. A fixed set of a continuous function $f$ is a nonempty finite set $S$ such that $S \subseteq \operatorname{conv} f(S)$. Our main technical contribution is to prove that computing a fixed set is PPAD-hard even at inverse polynomial precision.
    
[^73]: 通过数据拆分技术实现UCB式高斯过程赌博机的近最优遗憾界

    Near-Optimal Regret Bounds for UCB-Style Gaussian Process Bandits via Data Thinning

    [https://arxiv.org/abs/2610.28136](https://arxiv.org/abs/2610.28136)

    本文提出在高斯噪声假设下，通过数据拆分技术使UCB式高斯过程赌博机算法获得更紧的置信区间，从而实现近最优的简单遗憾和累积遗憾上界。

    

    高斯过程置信上界算法已被广泛研究，但其已知的遗憾上界是次优的。为了获得近最优的遗憾上界，现有研究主要集中于非自适应方法，例如不确定性采样，但其实际效果往往有限。本文表明，在高斯噪声假设下，UCB式算法可以通过数据拆分——一种从单个数据集创建独立数据集的统计技术——实现近最优的遗憾上界。首先，我们证明数据拆分可以在赌博机设置中产生更紧的置信区间，并且GP-UCB的直接扩展可以实现近最优的简单遗憾上界。此外，通过利用更紧的置信区间并将GP-UCB与分阶段消除算法相结合，我们证明了近最优的累积遗憾上界。最后，我们展示了UCB式算法的有效性。

    arXiv:2610.28136v1 Announce Type: cross  Abstract: Gaussian process upper confidence bound (GP-UCB) has been widely studied, though its known regret upper bound is suboptimal. To obtain near-optimal regret upper bounds, existing studies mainly focus on non-adaptive methods, such as uncertainty sampling, whose practical effectiveness is often limited. This paper shows that, under a Gaussian noise assumption, UCB-style algorithms can achieve near-optimal regret upper bounds via data thinning, a statistical technique for creating independent datasets from a single dataset. First, we show that data thinning can yield tighter confidence intervals in the bandit setting, and that a direct extension of GP-UCB achieves a near-optimal simple regret upper bound. Furthermore, by leveraging the tighter confidence intervals and combining GP-UCB with a phased elimination algorithm, we show a near-optimal cumulative regret upper bound. Finally, we demonstrate the effectiveness of UCB-style algorithms 
    
[^74]: 基于多域可辨识表示学习的含潜在混杂线性模型因果发现

    Causal Discovery for Linear Models with Latent Confounding via Multi-domain Identifiable Representation Learning

    [https://arxiv.org/abs/2610.28134](https://arxiv.org/abs/2610.28134)

    该论文提出LinADMG框架，利用多域可辨识表示学习来辨识噪声变量，从而在不依赖非高斯性或无弓性假设的情况下，实现对含潜在混杂的线性因果模型ADMG的辨识与因果发现。

    

    arXiv:2610.28134v1 公告类型：交叉发布。摘要：含潜在混杂的因果发现特别具有挑战性，但这是现实世界数据中的常见情形。在本工作中，我们关注用无环有向混合图（ADMG）表示的含潜在混杂线性因果模型。现有的能够完全辨识ADMG的方法通常假设非高斯性或图结构约束（如无弓性）。与之相反，我们提出了线性ADMG学习（LinADMG），一个通过可辨识表示学习来学习线性因果模型ADMG的框架。LinADMG不依赖非高斯性或无弓性假设，而是利用多域设置来辨识噪声变量。我们证明，噪声变量的可辨识性足以恢复所有直接因果关系；并且在附加的潜在混杂无抵消条件下，LinADMG能够辨识整个ADMG。我们采用四种可辨识表示学习方法实现了LinADMG（摘要原文在此处截断）。

    arXiv:2610.28134v1 Announce Type: cross  Abstract: Causal discovery with latent confounding is particularly challenging, but a common setting in real-world data. In this work, we focus on linear causal models with latent confounding, represented by acyclic directed mixed graphs (ADMGs). Existing methods that fully identify ADMGs typically assume non-Gaussianity or graphical constraints, such as bow-freeness. In contrast, we propose Linear ADMG Learning (LinADMG), a framework for learning ADMGs for linear causal models through identifiable representation learning. Rather than relying on non-Gaussianity or bow-freeness, LinADMG leverages multi-domain settings to identify the noise variables. We show that the identifiability of the noise variables is sufficient for recovering all direct causal relations, and that, under an additional latent-confounding no-cancellation condition, LinADMG can identify the whole ADMG. We implement LinADMG with four identifiable representation learning approa
    
[^75]: 弱-$L^\alpha$ 噪声，强保证：仿射重尾条件下的梯度裁剪

    Weak-$L^\alpha$ Noise, Strong Guarantees: Gradient Clipping under Affine Heavy Tails

    [https://arxiv.org/abs/2610.28107](https://arxiv.org/abs/2610.28107)

    本文在严格弱于矩有界假设的弱-$L^\alpha$ 重尾条件下，为仿射重尾噪声驱动光滑凸优化中的裁剪SGD及其加速变体建立了高概率收敛保证，并达到与矩条件情形相同的随机收敛率。

    

    梯度裁剪是在重尾噪声下为随机一阶方法获得高概率收敛保证的关键工具。现有分析通常假设 $\alpha\in(1,2]$ 阶中心矩有界，这排除了在临界指数处尾部阶为 $u^{-\alpha}$ 的分布。本文考虑仿射重尾噪声下的光滑凸优化问题，其中噪声尺度可能随梯度范数增长。在弱-$L^\alpha$ 尾部条件（严格弱于相应的矩条件）下，我们为裁剪随机梯度下降法（Clip-SGD）及其加速变体 Clip-SSTM 推导了高概率收敛保证。当 $1<\alpha<2$ 时，裁剪界中裁剪水平的幂次与矩条件情形相同，且两种方法均达到可加随机率 $K^{-(\alpha-1)/\alpha}$。梯度相关噪声贡献了一个衰减更快的项，其阶为 $K^{-2(\alpha-1)/\alpha}$。

    arXiv:2610.28107v1 Announce Type: cross  Abstract: Gradient clipping is a key tool for obtaining high-probability convergence guarantees for stochastic first-order methods under heavy-tailed noise. Existing analyses typically assume a bounded central moment of order $\alpha\in(1,2]$, which excludes distributions with tails of order $u^{-\alpha}$ at the critical exponent. In this paper, we consider smooth convex optimization under affine heavy-tailed noise, whose scale may grow with the gradient norm. We derive high-probability convergence guarantees for clipped SGD (Clip-SGD) and its accelerated variant Clip-SSTM under a weak-$L^\alpha$ tail condition, which is strictly weaker than the corresponding moment bound. For $1<\alpha<2$, the clipping bounds have the same powers of the clipping level as in the moment case, and both methods attain the additive stochastic rate $K^{-(\alpha-1)/\alpha}$. The gradient-dependent noise contributes a faster-decaying term of order $K^{-2(\alpha-1)/\alp
    
[^76]: 从复杂历史到简单预测器：动态图学习中的预测维度

    From Complex Histories to Simple Predictors: Predictive Dimension in Dynamic Graph Learning

    [https://arxiv.org/abs/2610.28090](https://arxiv.org/abs/2610.28090)

    该论文提出连续时间动态图学习无需重建完整交互历史，而应识别足以预测未来的低维状态，并定义了可独立估计的“内在预测维度”来量化任务所需的有效预测自由度。

    

    现有的连续时间图学习方法严重依赖复杂的神经架构来编码不断增长的交互历史。然而，由复杂历史所诱导的未来条件分布通常可以用一个低维的预测状态流形来刻画，这意味着对于未来的交互而言，只有少数预测自由度是关键的。因此，我们主张连续时间图学习应当识别对预测未来而言充分必需的低维状态，而不是重建完整的历史。为了使这一观点可度量，我们定义并独立估计了内在预测维度，该维度量化了给定任务所需的有效预测自由度。具体而言，我们从以历史为条件的未来分布中去除与历史无关的边缘分布，分解由此得到的预测残差算子，并估计其最小有效秩……

    arXiv:2610.28090v1 Announce Type: cross  Abstract: Existing continuous-time graph learning methods rely heavily on complex neural architectures to encode ever-expanding interaction histories. However, the future conditional distribution induced by complex histories can often be characterized by a low-dimensional predictive state manifold, implying that only a few predictive degrees of freedom are essential for future interactions. We therefore argue that continuous-time graph learning should identify low-dimensional states sufficient for future prediction rather than reconstruct the full history. To make this perspective measurable, we define and independently estimate the intrinsic predictive dimension, which quantifies the effective predictive degrees of freedom required by a given task. Specifically, we remove the history-independent marginal from the history-conditioned future distribution, decompose the resulting predictive residual operator, and estimate the minimum effective ran
    
[^77]: PULSE：面向可穿戴设备的个性化机会学习

    PULSE: Personalized Opportunistic Learning for Wearable Devices

    [https://arxiv.org/abs/2610.28081](https://arxiv.org/abs/2610.28081)

    PULSE提出了一种事件驱动的个性化机会学习框架，将可穿戴设备间的每次相遇视为在线协作决策，仅在梯度与本地学习方向兼容时融合对等节点信息，从而在无稳定连接的条件下解决部署后数据分布差异带来的个性化学习难题。

    

    可穿戴设备日益支持个性化健康与福祉类应用，但部署后采集的数据可能与训练所用数据存在显著差异，而在真实运行环境中，纯本地自适应往往受限于单一设备上可获得的数据。传统联邦学习与去中心化学习依赖周期性通信和稳定连接，而现有的机会式方法通常只是利用可用的设备相遇机会，却未充分考虑其效用、兼容性或通信成本。我们提出PULSE——一个用于个性化机会学习的事件驱动框架，它将每一次设备相遇视为一次在线协作决策：要么选择不进行交互，要么选择一个有前景的邻近节点。只有当对等节点的梯度与本地学习方向兼容时才会被纳入，同时过去的交互经验会指导未来的节点选择。在人体活动识别、跌倒检测等任务上……（原文摘要截断）

    arXiv:2610.28081v1 Announce Type: cross  Abstract: Wearable devices increasingly support personalized health and well-being applications, but post-deployment data can differ substantially from those used for training, while purely local adaptation may be limited by the data available on a single device during real-world operation. Conventional federated and decentralized learning rely on recurrent communication and stable connectivity, whereas existing opportunistic approaches often exploit available encounters without adequately considering their utility, compatibility, or communication cost.   We propose PULSE, an event-driven framework for personalized opportunistic learning that treats each encounter as an online collaboration decision, selecting either no interaction or a promising nearby peer. Peer gradients are incorporated only when compatible with the local learning direction, while past interactions guide future peer selection. Across human activity recognition, fall detectio
    
[^78]: MuPACE：学习为LoRA微调调控Muon优化器的步调节奏

    MuPACE: Learning to Pace Muon for LoRA Fine-Tuning

    [https://arxiv.org/abs/2610.28072](https://arxiv.org/abs/2610.28072)

    MuPACE通过在大量微调任务上元训练一个轻量循环策略，自适应地为Muon优化器调控LoRA微调的全局学习率步长，在无需逐任务调参的情况下于大多数问题上超越固定学习率调度。

    

    学习率调度通常在训练开始前就已固定，然而在使用低秩适配器（LoRA）微调大型语言模型时，合适的尺度会随模型、任务和适配器配置的不同而产生显著变化。从业者需要为每个新问题重新调整调度，但即便经过调整的调度仍然无法对其遇到的训练动态做出反应。我们转而在多个LoRA微调问题上摊销学习率的选择。我们保持基础优化器Muon完全不变，使其归一化更新固定更新方向，仅对全局步长进行元学习。我们的方法MuPACE是一个小型循环策略，通过在一系列微调运行的分布上进行元训练，它以观测到的训练轨迹为条件，对可解释的解析调度输出有界的调整。在64个微调问题上，MuPACE在64-78%的问题上优于采用统一强大默认设置的标准固定调度，并与相同的……持平。

    arXiv:2610.28072v1 Announce Type: new  Abstract: Learning rate schedules are typically fixed before training begins, yet when fine-tuning large language models with low-rank adapters (LoRA), the appropriate scale varies substantially with the model, task, and adapter configuration. Practitioners re-tune a schedule for every new problem, yet a tuned schedule still cannot react to the training dynamics it encounters. We instead amortize learning rate selection across LoRA fine-tuning problems. We leave the base optimizer, Muon, untouched, so that its normalized update fixes the direction, and meta-learn only the global step size. Our method MuPACE is a small recurrent policy, meta-trained across a distribution of fine-tuning runs, that conditions on the observed trajectory and emits bounded adjustments to an interpretable analytic schedule. On 64 fine-tuning problems, MuPACE outperforms standard fixed schedules at a strong shared default on 64-78% of problems and is on par with the same 
    
[^79]: 量化威胁：加纳登苏流域儿童重金属健康风险的贝叶斯概率框架

    Quantifying the Threat: A Bayesian Probabilistic Framework for Heavy Metal Health Risks among Children in the Densu Basin, Ghana

    [https://arxiv.org/abs/2610.28064](https://arxiv.org/abs/2610.28064)

    该研究提出贝叶斯概率框架评估加纳登苏流域地下水重金属对儿童的非致癌健康风险，发现约31%的采样点危害指数超过1的概率大于0.5，其中铅、砷和镉是主要风险贡献者。

    

    加纳登苏河流域的地下水仍然是重要的生活用水来源，但多种金属污染威胁着儿童的健康。以往的确定性危害指数评估对不确定性、共同暴露和空间变异性的处理有限。本研究应用贝叶斯概率框架进行非致癌风险评估，使用94个地下水样本以及对铁、锰、镍、铅、镉和砷的测量。贝叶斯回归模型传播参数不确定性并考虑了污染物的共存情况。采用自适应哈密顿蒙特卡洛方法进行后验推断，同时使用反距离加权方法生成空间连续的风险曲面。铅、砷和镉主导了累积风险。后验预测结果表明，约31%的采样点其危害指数（HI）超过1的概率大于0.5。风险曲面高度异质，后验均值为0.37。

    arXiv:2610.28064v1 Announce Type: cross  Abstract: Groundwater in Ghana's Densu River Basin remains a key domestic water source, but multi-metal contamination threatens children's health. Previous deterministic hazard index assessments provide limited treatment of uncertainty, co-exposure, and spatial variability. This study applies a Bayesian probabilistic framework for non-carcinogenic risk assessment using 94 groundwater samples and measurements of Fe, Mn, Ni, Pb, Cd, and As. A Bayesian regression model propagates parameter uncertainty and accounts for contaminant co-occurrence. Adaptive Hamiltonian Monte Carlo is used for posterior inference, while inverse distance weighting generates spatially continuous risk surfaces. Pb, As, and Cd dominate cumulative risk. Posterior predictive results indicate that approximately 31% of sampling sites have a probability greater than 0.5 of exceeding a hazard index (HI) of 1. The risk surface is highly heterogeneous, with a posterior mean of 0.37
    
[^80]: 审计CUB数据集上概念瓶颈模型准确率的来源

    Auditing Where Concept Bottleneck Accuracy Comes From on CUB

    [https://arxiv.org/abs/2610.28059](https://arxiv.org/abs/2610.28059)

    该研究审计了概念瓶颈模型和概念嵌入模型在CUB基准上准确率的真实来源，发现CUB标注噪声过大、且模型预测依赖内部信息而非暴露的概念分数，从而质疑了现有基于准确率的CBM可解释性评估方式的有效性。

    

    概念瓶颈模型承诺提供人类可以通过“弯曲的喙”等概念进行检验和修正的预测。然而在其标准基准CUB-200-2011上，其进展的衡量方式却与普通分类器相同：即通过任务准确率和概念准确率来衡量。这两种指标都无法揭示标签是否确实由暴露的概念分数预测得出，也无法揭示某个概念是由该概念本身的视觉证据所支持，还是由共现特征（例如指向整个物种的线索）所支持。我们审计了CUB上概念瓶颈模型（CBM）和概念嵌入模型（CEM）准确率的来源。CUB的标注噪声太大，无法解释这一现象：类别级训练标签与逐图像标注仅在约82%的图像-概念对上一致，而在逐图像标注上训练的分类器仅能达到约37%的物种准确率。泄漏探针实验表明，CEM的预测无法从其概念分数中重建；缺失的类别信息存在于内部……

    arXiv:2610.28059v1 Announce Type: cross  Abstract: Concept Bottleneck Models (CBMs) promise predictions that humans can inspect and correct through concepts such as "curved beak". Yet on CUB-200-2011, their standard benchmark, progress is measured like that of ordinary classifiers: by task and concept accuracy. Neither reveals whether the label is predicted from the exposed concept scores, or whether a concept is supported by visual evidence of that concept rather than by co-occurring features such as cues to the species as a whole. We audit where the accuracy of CBMs and Concept Embedding Models (CEMs) on CUB comes from. The CUB annotations are too noisy to explain it: class-level training labels agree with per-image annotations on only ~82% of image--concept pairs, and classifiers trained on per-image annotations reach only ~37% species accuracy. Leakage probes show that CEM predictions cannot be reconstructed from their concept scores; the missing class information sits in internal 
    
[^81]: 带延迟反馈的通用在线凸优化

    Universal Online Convex Optimization With Delayed Feedback

    [https://arxiv.org/abs/2610.28058](https://arxiv.org/abs/2610.28058)

    本文提出了一种无需任何曲率先验知识即可同时自适应凸、强凸和指数凹损失的延迟反馈在线凸优化通用算法，其核心是一个带二阶超额损失保证的延迟元学习器，遗憾界在时间范围与延迟依赖上达到已知最优水平。

    

    我们研究带延迟反馈的通用在线凸优化问题。我们开发了一种单一算法，无需预先知道损失函数的曲率类型或参数，即可同时自适应于一般凸、强凸和指数凹损失。我们的遗憾界在时间范围 $T$ 和延迟的依赖关系上匹配了已知最优结果（仅相差双对数因子）：对于凸损失约为 $\sqrt{T + d_{\mathrm{tot}}}$，对于具有曲率的损失约为 $\sigma_{\max}\log T$ 的速率，其中 $d_{\mathrm{tot}}$ 是总累积延迟，$\sigma_{\max}$ 是同时缺失观测的最大数量。我们的主要技术贡献是一个具有二阶超额损失保证的延迟元学习器，它使得能够对延迟的基础学习器进行曲率自适应聚合。此外，我们新的指数凹基础学习器产生了更紧致的遗憾界，通过用（总累积延迟）替换最大延迟，改进了现有最先进的延迟依赖性。

    arXiv:2610.28058v1 Announce Type: new  Abstract: We study universal online convex optimization with delayed feedback. We develop a single algorithm that simultaneously adapts to general convex, strongly convex, and exp-concave losses without prior knowledge of the curvature class or parameters. Our regret bounds match the best known dependence on the time horizon $T$ and delays up to doubly logarithmic factors: roughly $\sqrt{T + d_{\mathrm{tot}}}$ for convex losses and $\sigma_{\max}\log T$ rates with curved losses, where $d_{\mathrm{tot}}$ is the total cumulative delay and $\sigma_{\max}$ is the maximum number of simultaneously missing observations. Our main technical contribution is a delayed meta-learner with a second-order excess-loss guarantee, which enables curvature-adaptive aggregation of delayed base learners. In addition, our new exp-concave base learner yields a sharper regret bound which improves the state-of-the-art delay dependence by replacing the maximum delay with the
    
[^82]: 在线策略过渡蒸馏

    On-Policy Transition Distillation

    [https://arxiv.org/abs/2610.28048](https://arxiv.org/abs/2610.28048)

    该论文发现教师与学生在中间层预测过渡上的一致性与答案正确率正相关，并提出在线策略过渡蒸馏（OPTD），通过在学生生成的轨迹上对齐教师与学生的逐层对数概率增量来增强传统最终层分布匹配，从而提升蒸馏效果。

    

    在线策略蒸馏（OPD）通过在学生模型生成的采样轨迹上匹配教师模型与学生模型的最终层分布，将专家能力整合到单个学生模型中。我们研究了深度方向上的预测过渡——即各层之间预测的变化——能否提供额外的与任务相关的监督信号。我们的诊断性研究发现，教师与学生在中间层过渡上的一致性越强，答案正确率越高，正确回答在所有测量的中间层对上均表现出更低的平均过渡KL散度。这些观察启发了在线策略过渡蒸馏（OPTD）的提出，该方法通过在学生生成的采样轨迹上对齐教师与学生的逐层对数概率增量，来增强最终层的分布匹配。在涵盖数学、编程和指令遵循任务的三个骨干模型和八个基准上的评估表明，OPTD 取得了……

    arXiv:2610.28048v1 Announce Type: cross  Abstract: On-policy distillation (OPD) consolidates expert capabilities into a single student by matching teacher and student final-layer distributions on student-generated rollouts. We investigate whether depthwise prediction transitions, the changes in predictions between layers, can provide additional task-relevant supervision. Our diagnostic study finds that stronger teacher--student agreement in intermediate transitions is associated with higher answer correctness within questions, with correct responses exhibiting lower mean transition KL across all measured intermediate layer pairs. These observations motivate On-Policy Transition Distillation (OPTD), which augments final-layer distribution matching by aligning teacher and student layer-to-layer log-probability increments on student-generated rollouts. Evaluations across three backbones and eight benchmarks spanning mathematics, coding, and instruction following show that OPTD achieves th
    
[^83]: 基于投毒的后门并非密码学安全的

    Poisoning-Based Backdoors Are Not Cryptographic

    [https://arxiv.org/abs/2610.28045](https://arxiv.org/abs/2610.28045)

    本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。

    

    为了防御针对神经网络的后门攻击，防御者必须识别出决定某个输入是否为触发器的验证器函数，而不能只是孤立地拒绝单个触发器，因为攻击者生成满足同一验证器的新触发器的速度会快于防御者拦截它们的速度。如果后门验证器能够像密码学认证方案那样具有不对称性——即只有攻击者能够产生触发器输入，而防御者无法通过观测逆向推导出验证器——那么基于学习的防御在原则上将是不可能的。我们证明，这种不对称性不可能由基于投毒的安装方式产生，在这种方式中，训练过程中唯一依赖于验证器的输入就是被投毒的数据集。该论证采用归约方法：一个能够从被投毒数据中安装验证器、使后门在未见过的触发器上触发同时保持良性行为的训练过程，其本身就是一个学习（摘要在此处截断）

    arXiv:2610.28045v1 Announce Type: cross  Abstract: To defend against backdoor attacks on neural networks, the defender must identify the verifier function that determines whether an input is a trigger, rather than rejecting individual triggers in isolation, since an attacker can generate new triggers satisfying the same verifier faster than they can be blocked. If a backdoor verifier could be installed with the asymmetry of a cryptographic authentication scheme, in which only the attacker can produce trigger inputs while the defender cannot reverse-engineer the verifier from observations, learning-based defense would be impossible in principle. We show that no such asymmetry can arise from poisoning-based installation, in which the only verifier-dependent input to training is a poisoned dataset. The argument is a reduction. A training procedure that installs a verifier from poisoned data, so that the backdoor fires on unseen triggers while benign behavior is preserved, is itself a lear
    
[^84]: 合成数据训练何时才能被证明有效？

    When Does Training on Synthetic Data Provably Work?

    [https://arxiv.org/abs/2610.28036](https://arxiv.org/abs/2610.28036)

    该论文从理论上刻画了合成数据训练何时可证明有效：在密度比受控时给出有限样本超额误差界，但若真实分布在合成分布支撑之外且假设类足够有表达能力，则仅凭合成风险无法保证真实世界的误差。

    

    高质量训练数据是机器学习中的一大瓶颈，这促使人们越来越多地使用人工生成的数据。我们研究了在共享条件标签分布的前提下，在合成数据上训练的二元分类器何时能够成功迁移到真实分布上。首先，当真实协变量分布在受控的密度比条件下关于合成分布绝对连续时，我们推导了有限样本超额误差界。其次，我们研究了部分奇异与完全奇异的情形，即真实分布在合成分布支撑之外存在概率质量。如果假设类具有足够的表达能力，那么仅凭合成风险无法一致地控制真实世界的误差：即使两个分布在 Wasserstein 距离上可以任意接近，一个分类器也可以在合成数据上几乎达到最优，同时在未被覆盖的真实区域上接近最坏情况的误差。对于假设函数与其对数…（原文摘要至此截断）

    arXiv:2610.28036v1 Announce Type: new  Abstract: High-quality training data are a major bottleneck in machine learning, motivating increasing use of artificially generated data. We study when a binary classifier trained on synthetic data transfers successfully to a real distribution, under a shared conditional label distribution. First, we derive finite-sample excess-error bounds when the real covariate distribution is absolutely continuous with respect to the synthetic distribution under a controlled density ratio. Second, we study the partially and fully singular case, where the real distribution has mass outside the synthetic one. If the hypothesis class is sufficiently expressive, synthetic risk alone cannot uniformly control real-world error: a classifier can be nearly optimal on synthetic data while approaching worst-case error on the uncovered real region, even when the distributions are arbitrarily close in Wasserstein distance. For function classes where hypotheses and the log
    
[^85]: 从强化学习论文到框架代码：七个开源训练框架中的收录与评审

    From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks

    [https://arxiv.org/abs/2610.28027](https://arxiv.org/abs/2610.28027)

    该研究通过追踪七个开源强化学习后训练框架的代码提交历史，发现1598篇提出通用RL训练方法的论文中仅有37篇被框架收录，且收录的论文多为大规模训练（30B+参数）和现有损失函数的小幅修改，揭示了RL研究成果从论文到实际工程应用之间的巨大鸿沟。

    

    从业者通过训练框架对语言模型使用强化学习方法，因此框架收录的方法对用户而言仅需更改一次配置即可使用。我们测量了这种情况发生的频率。我们追踪了七个用于强化学习后训练的开源框架的提交历史，并将其与2025年1月至2026年6月的6,939篇筛选后的arXiv论文相关联。在1,598篇提出通用强化学习训练方法的论文中，仅有37篇进入至少一个框架。在五个季度论文批次中，180天内被收录的数量始终保持在5至7篇之间，而批次规模却从72篇增长到386篇。与同期的其他论文相比，被收录的论文更常训练300亿（30B）参数或更大的策略模型（41%对10%），并且更常在现有损失函数中修改系数、裁剪、掩码或权重（29%对9%）。即使排除维护这些框架的公司自己的论文，这两个差距依然存在。

    arXiv:2610.28027v1 Announce Type: cross  Abstract: Practitioners use reinforcement learning methods for language models through training frameworks, so a method that a framework includes is one configuration change away for its users. We measure how often this happens. We trace seven open-source frameworks for RL post-training through their commit histories and link them to 6,939 screened arXiv papers from January 2025 to June 2026. Of 1,598 papers that propose a general RL training method, 37 reached at least one framework. Across five quarterly cohorts of papers, the number included within 180 days stayed between 5 and 7 while the cohorts grew from 72 to 386 papers. Compared with papers from the same quarters, included papers more often trained policies of 30B parameters or more, at 41 against 10 percent, and more often changed a coefficient, clip, mask or weight in the existing loss, at 29 against 9 percent. Both gaps hold without papers from the companies that maintain the framewor
    
[^86]: RANGER：基于语义评估奖励的图导航检索智能体

    RANGER: Retrieval Agent for Navigating Graphs with Semantic Evaluation Rewards

    [https://arxiv.org/abs/2610.28010](https://arxiv.org/abs/2610.28010)

    该论文提出RANGER混合框架，让大型语言模型在训练中充当导师，为强化学习智能体提供密集的语义奖励信号，使其学会在知识图谱上导航并完成多跳问答任务。

    

    大型语言模型与强化学习的结合为复杂的序列决策任务（如知识图谱上的多跳问答）提供了一个有前景的方向。虽然大型语言模型具有强大的语义理解和推理能力，但它们在结构化环境中的精确多步导航方面往往存在困难。相反，强化学习智能体擅长学习导航策略，但语义上有意义的奖励函数设计困难阻碍了其发展。本文提出了一种新颖的混合方法，在训练过程中，大型语言模型作为强化学习智能体的导师，提供密集的、语义信息丰富的奖励信号。该强化学习智能体基于Transformer策略构建，并通过组相对策略优化（Group Relative Policy Optimization）进行训练，学习在知识图谱中导航以检索相关信息。大型语言模型评估智能体所遍历路径的语义相关性，并生成奖励信号……（原文摘要在此处被截断）

    arXiv:2610.28010v1 Announce Type: new  Abstract: The integration of Large Language Models and Reinforcement Learning offers a promising direction for complex sequential decision-making tasks, such as multi-hop question answering over knowledge graphs. While LLMs provide strong semantic understanding and reasoning capabilities, they often struggle with precise, multi-step navigation in structured environments. Conversely, RL agents excel at learning navigation policies but are hindered by the difficulty of designing semantically meaningful reward functions. This paper introduces a novel hybrid approach in which an LLM serves as a mentor to an RL agent during training by providing dense, semantically informed reward signals. The RL agent, built on a Transformer-based policy and trained with Group Relative Policy Optimization, learns to navigate a knowledge graph to retrieve relevant information. The LLM evaluates the semantic relevance of the agent's traversed paths, generating reward si
    
[^87]: 使用RSOD构建电信语言模型：领域数据、专家后训练、内存感知蒸馏与网络评估

    Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation

    [https://arxiv.org/abs/2610.27993](https://arxiv.org/abs/2610.27993)

    该论文提出RSOD端到端框架，通过领域合成数据构建、多个专家LoRA教师后训练以及内存感知的在线策略蒸馏，将六个网络专家模型的行为整合到一个可部署的单一电信语言模型中。

    

    电信语言模型必须将广泛的网络知识与路由、宽带服务、接入技术和运维诊断方面的专家行为相结合，同时保持作为单一模型部署的实用性。我们提出了一个端到端框架，该框架从结构化网络分类体系和专家图谱构建有来源依据的合成训练数据，生成跨越十四条网络路由的212,559条路由化训练记录，在一个共享的持续预训练骨干模型上训练六个冻结的专家LoRA教师模型，并使用路由专家在线策略蒸馏（Routed Specialist On-Policy Distillation, RSOD）将其行为整合到一个可部署的学生适配器中。在RSOD中，已确定的学生模型生成每条轨迹，按路由分配的专家教师对精确的原生token序列进行教师强制，紧凑的教师隐藏状态通过共享的冻结输出头进行重建，且仅更新学生LoRA。一项受控的重建……

    arXiv:2610.27993v1 Announce Type: new  Abstract: Telecom language models must combine broad networking knowledge with specialist behavior across routing, broadband services, access technologies, and operational diagnostics while remaining practical to deploy as a single model. We present an end-to-end framework that builds source-grounded synthetic training data from structured networking taxonomies and expert graphs, materializes 212,559 routed training records across fourteen networking routes, trains six frozen specialist LoRA teachers over one shared continued-pretrained backbone, and consolidates their behavior into one deployable student adapter using Routed Specialist On-Policy Distillation (RSOD). In RSOD, the committed student generates each trajectory, the route-assigned specialist teacher-forces the exact native token sequence, compact teacher hidden states are reconstructed through the shared frozen output head, and only the student LoRA is updated. A controlled reconstruct
    
[^88]: 动量何时有助于 Polyak--Ruppert 平均

    When Momentum Helps Polyak--Ruppert Averaging

    [https://arxiv.org/abs/2610.27990](https://arxiv.org/abs/2610.27990)

    该论文证明动量方法能加速有限次运行中 Polyak--Ruppert 平均向渐近协方差的收敛——SGD 需要 $m\gg\kappa^2$ 轮而动量只需 $m\gg\kappa$，从而使样本量 $N$ 下的临界批量大小从 $\Theta(\sqrt{N}/\kappa)$ 提升到 $\Theta(\sqrt{N/\kappa})$。

    

    Polyak--Ruppert 平均使 SGD 与动量方法具有相同的渐近协方差。我们刻画了有限次运行多快能达到该渐近协方差，以及它们可以被并行化到何种程度。对于带加性噪声的强凸二次目标函数，我们证明了一座“在线—离线桥梁”：在尾平均所对应样本上的经验风险最小化器，与任何在线过程的尾平均查询点之间仅相差一个牛顿步。对于常数动量的 SGD、重球法和 Nesterov 方法，这一步表现为尾平均两端的一个边界项，并将与离线估计器之间的差距分解为中心误差、协方差误差和形状误差。充分必要条件表明，在曲率最小的方向上，中心误差决定了收敛节奏：SGD 需要 $m\gg\kappa^2$ 轮迭代，而动量方法只需 $m\gg\kappa$，其中 $\kappa$ 为条件数；因此在样本量为 $N$ 时，临界批量大小从 $\Theta(\sqrt{N}/\kappa)$ 增长到 $\Theta(\sqrt{N/\kappa})$。（注：原文摘要末尾被截断）

    arXiv:2610.27990v1 Announce Type: new  Abstract: Polyak--Ruppert averaging gives SGD and momentum methods the same asymptotic covariance. We characterize how fast finite runs reach it and how far they can be parallelized. For strongly convex quadratics with additive noise, we prove an online--offline bridge: the empirical risk minimizer on the samples of a tail average is one Newton step from the tail-averaged query point of any online procedure. For SGD, heavy ball, and Nesterov's method with constant momentum, this step is a boundary term at the two ends of the tail and splits the gap to the offline estimator into center, covariance, and shape errors. Necessary and sufficient conditions show that, on the least curved direction, the center error sets the pace: SGD needs $m\gg\kappa^2$ rounds but momentum only $m\gg\kappa$, where $\kappa$ is the condition number, so at sample size $N$ the critical batch size grows from $\Theta(\sqrt{N}/\kappa)$ to $\Theta(\sqrt{N/\kappa})$. The covaria
    
[^89]: 关于Johnson-Lindenstrauss变换在拜占庭鲁棒优化中的应用

    On Johnson-Lindenstrauss Transforms for Byzantine-Robust Optimization

    [https://arxiv.org/abs/2610.27989](https://arxiv.org/abs/2610.27989)

    本文利用Johnson-Lindenstrauss变换能高概率近似保持内积和距离的几何特性，将其应用于拜占庭鲁棒优化，根据工作节点消息与可信参考梯度的对齐程度进行加权，并通过理论分析和实验验证了该方法的有效性。

    

    机器学习中数据和模型规模的不断增长对高效的优化方法提出了需求。分布式优化应对了这一挑战，而压缩机制有助于降低通信成本。降维的一个经典工具是Johnson-Lindenstrauss（JL）变换，它实现简单且具有很强的几何保证。特别地，JL变换能够以高概率近似保持向量之间的内积和成对距离。在本文中，我们在拜占庭鲁棒优化中利用了这一特性，即根据工作节点消息与可信参考梯度的对齐程度对其进行加权。我们还通过实验结果进一步支持了理论发现。

    arXiv:2610.27989v1 Announce Type: new  Abstract: The growing scale of data and models in machine learning calls for efficient optimization methods. Distributed optimization addresses this challenge, with compression mechanisms helping to reduce communication costs. A classical tool for dimensionality reduction is the Johnson-Lindenstrauss (JL) transform, which is simple to implement and offers strong geometric guarantees. In particular, JL transforms approximately preserve inner products and pairwise distances between vectors with high probability. In this paper, we exploit this property in Byzantine-robust optimization, where worker messages are weighted according to their alignment with a trusted reference gradient. We further support our theoretical findings with experimental results.
    
[^90]: 含偏置深度神经网络中的退火相关性与淬火噪声敏感性

    Annealed correlation and quenched noise sensitivity in biased deep neural networks

    [https://arxiv.org/abs/2610.27983](https://arxiv.org/abs/2610.27983)

    本文证明在给深度神经网络加入独立高斯偏置项后，淬火噪声敏感性在广泛的深度范围内依然成立，且只需扰动输入与原输入相等的概率趋于零这一更弱的假设。

    

    在Jonasson、Steif和Zetterqvist（2023）的早期工作中，已经证明了每层宽度为 $n$、权重为独立同分布标准高斯、使用符号激活且不含偏置项的全连接深度神经网络，对于输入 $\boldsymbol{\omega}$ 及其扰动 $\boldsymbol{\eta}$ 上任何满足温和正相关条件且 $\mathbb{Q}^{(n)}(\boldsymbol{\eta} \in \{\boldsymbol{\omega},-\boldsymbol{\omega}\}) \to 0$ 的联合概率分布 $\mathbb{Q}^{(n)}$ 而言，都是退火和淬火噪声敏感的。本文中，我们加入了自然数量级的独立高斯偏置项，并证明在相当广泛的深度范围内淬火噪声敏感性依然保持，实际上只需稍弱的假设 $\mathbb{Q}^{(n)}(\boldsymbol{\omega} = \boldsymbol{\eta}) \to 0$。因此，在正相关条件下，淬火噪声敏感性在最弱要求下即成立，即扰动输入渐近……

    arXiv:2610.27983v1 Announce Type: cross  Abstract: In earlier work by Jonasson, Steif and Zetterqvist (2023), fully connected deep neural networks with layers of width $n$, i.i.d.\ standard Gaussian weights, sign activations and no bias terms were shown to be annealed and quenched noise sensitive with respect to any joint probability distribution $\mathbb{Q}^{(n)}$ on the input $\boldsymbol{\omega}$ and its perturbation $\boldsymbol{\eta}$ satisfying a mild positive-correlation condition and $\mathbb{Q}^{(n)}(\boldsymbol{\eta} \in \{\boldsymbol{\omega},-\boldsymbol{\omega}\}) \to 0$. Here we add independent Gaussian biases of the natural order and show that quenched noise sensitivity is retained over a broad range of depths, indeed under the slightly weaker assumption that $\mathbb{Q}^{(n)}(\boldsymbol{\omega} = \boldsymbol{\eta}) \to 0$. Thus, subject to the positive-correlation condition, quenched noise sensitivity holds under the minimal requirement that the perturbed input is asymp
    
[^91]: 具有稳定性与泛化保证的非线性系统神经闭环辨识

    Neural Closed-Loop Identification of Nonlinear Systems with Stability and Generalization Guarantees

    [https://arxiv.org/abs/2610.27970](https://arxiv.org/abs/2610.27970)

    该论文将对偶Youla参数化推广到非线性系统，提出内部控制器辨识框架，能够从闭环数据中辨识可能不稳定的非线性系统并保证闭环稳定性。

    

    神经网络是非线性动力学的强表达能力模型，但从闭环数据中辨识此类系统面临两个困难：反馈回路使得输入与测量噪声相关，从而使标准的开环估计方法产生偏差；同时辨识出的模型必须能被数据采集控制器镇定，使其闭环行为与真实被控对象相匹配。我们推导出了由给定控制器镇定的所有非线性系统的完整参数化方法，将经典的对偶Youla参数化推广到了非线性情形。基于这一刻画，我们提出了内部控制器辨识，这是一个计算上易于处理的框架，可用于辨识可能不稳定的系统，同时通过构造方式保证闭环稳定性；在线性情形下，我们的间接方法可退化为一个开环辨识问题，因而标准的一致性结果依然适用。该框架可适应上下文相关的……

    arXiv:2610.27970v1 Announce Type: cross  Abstract: Neural networks are expressive models of nonlinear dynamics, but identifying such systems from closed-loop data raises two difficulties: the feedback loop correlates inputs with measurement noise, biasing standard open-loop estimators, and the identified model must be stabilized by the data-collection controller, so that its closed-loop behavior matches the true plant's. We derive a complete parameterization of all nonlinear systems stabilized by a given controller, generalizing the classical dual-Youla parameterization to the nonlinear setting. Building on this characterization, we propose Internal Controller Identification (ICI), a computationally tractable framework for identifying potentially unstable systems while ensuring closed-loop stability by construction; in the linear case, our indirect method reduces to an open-loop identification problem, for which standard consistency results apply. The framework accommodates contextual 
    
[^92]: 迷失于恢复之中：通过面向目标的动作回放揭示计算机使用智能体的恢复行为

    Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay

    [https://arxiv.org/abs/2610.27954](https://arxiv.org/abs/2610.27954)

    该论文提出ReTrace评估框架，通过形式化“转换一致的恢复”标准并借助面向目标的动作回放，系统性地评估计算机使用智能体从失败中恢复的能力，填补了基于模型的恢复研究的空白。

    

    近期大（视觉）语言模型的进展使计算机使用智能体能够通过细粒度的键盘和鼠标操作自动完成数字工作流程。然而，这类智能体在长时程任务中仍然十分脆弱：误差不断累积，已执行的动作会留下意想不到的副作用，因此从失败中恢复的能力至关重要。然而，现有的补救手段仍不充分：环境重置在专有环境中不可行，动作回放在长时程下代价过于高昂，基于API的回滚又仅限于网页环境，这使得基于模型的恢复方法在很大程度上尚未被探索。为填补这一空白，我们提出了ReTrace，一个用于系统研究计算机使用智能体恢复行为的可扩展评估框架。我们将“转换一致的恢复”形式化为一个判定标准：从恢复后的状态与目标状态出发执行相同的下游动作，必须产生相同的结果。我们通过面向目标的（动作回放）将这一标准加以操作化……（摘要在此处截断）

    arXiv:2610.27954v1 Announce Type: new  Abstract: Recent advances in large (vision) language models have enabled computer-use agents that automate digital workflows via fine-grained keyboard and mouse operations. However, such agents remain fragile in long-horizon settings, where errors accumulate and executed actions leave unintended side effects, making the ability to recover from failures critical. Yet, existing remedies remain inadequate: environment resets are infeasible in proprietary settings, action replay is prohibitively expensive over long horizons, and API-based rollback is limited to web environments, leaving model-based recovery largely unexplored. To address this gap, we present ReTrace, a scalable evaluation framework for systematically studying recovery behavior in computer-use agents. We formalize Transition-Consistent Recovery: identical downstream actions from the recovered and target states must yield identical outcomes. We operationalize this criterion through goal
    
[^93]: 基于联邦学习的分割计算无监督头部网络个性化

    Unsupervised Head Network Personalization for Split Computing via Federated Learning

    [https://arxiv.org/abs/2610.27952](https://arxiv.org/abs/2610.27952)

    该论文提出了一种结合联邦学习与知识蒸馏的数据高效方法，在仅需少量无标签、非独立同分布数据且保护用户隐私的前提下，实现分割计算中头部网络的无监督个性化训练。

    

    分割计算（SC）是一种分布式推理方法，通过将神经网络分割为头部网络和尾部网络，并分别部署在移动设备和服务器上，旨在平衡计算负载并降低延迟。由于移动设备的带宽和计算能力有限，头部网络通常会被替换为需要较少计算和通信资源的神经网络。然而，在移动设备端准备足够数量和质量的标记数据来训练替换的头部网络是具有挑战性的。此外，移动用户准备的数据通常包含敏感的隐私信息，使得数据聚合变得困难。本文提出了一种面向分割计算的数据高效头部网络训练方法。该方法利用联邦学习和知识蒸馏，使用少量非独立同分布且无标签的数据来训练头部网络。

    arXiv:2610.27952v1 Announce Type: new  Abstract: Split Computing (SC) is a distributed inference method designed for balancing computational load and reducing latency by partitioning a neural network into head and tail networks and deploying them across both a mobile device and a server, respectively. Due to the limited bandwidth and computational power of mobile devices, the head network is often replaced with a neural network that requires lower computational and communication resources. However, it is challenging to prepare a sufficient quantity and quality of labeled data on the mobile device side for training the replacement head network. Furthermore, the data prepared by mobile users often contains sensitive privacy information, making data aggregation difficult. In this paper, we propose a data-efficient head network training method for SC. This method utilizes federated learning and knowledge distillation to train the head network using a small amount of non-IID and unlabeled d
    
[^94]: 近重复图像选择在视觉-语言模型中传递偏好

    Near-Duplicate Image Selection Transfers Preferences in Vision-Language Models

    [https://arxiv.org/abs/2610.27943](https://arxiv.org/abs/2610.27943)

    该论文发现视觉-语言模型仅通过在近重复图像变体中进行选择（不改变原始照片和人类标题），就能将自身隐含的偏好隐性传递给由其筛选数据训练出的其他学生模型。

    

    arXiv:2610.27943v1 公告类型： new 摘要： 隐性学习是指一个模型通过并未明确表达其偏好的数据将偏好传递给另一个模型的现象。随着模型越来越多地为其他模型筛选训练数据，我们提出这样一个问题：在保持原始照片和人类撰写的标题固定不变的情况下，这种传递是否可能仅仅源于在近重复图像之间进行的选择。对于每张照片，一个“策展者”模型根据标题似然度从八个近重复变体中选择一个，且不使用任何针对目标的提示。在偏好特定目标的策展者所做的选择上训练的学生模型，在与图像无关的读出测试中，相比在中性选择上训练的学生，显示出更高的目标相对于替代选项的平均对数似然比。根据预设的判定规则，当策展者和学生从同一基础检查点初始化时，这种方向性偏移在 LLaVA-1.5、Qwen2.5-VL、Qwen3-VL 和 Gemma 4 中针对猫头鹰目标均会出现，且目标轮换会使偏移转向骆驼、橡树和……

    arXiv:2610.27943v1 Announce Type: new  Abstract: Subliminal learning is the phenomenon in which a model transmits its preferences to another model through data that do not explicitly express them. As models increasingly curate training data for other models, we ask whether this transfer can arise from choosing among near-duplicate images while keeping the source photographs and human-written captions fixed. For each photograph, a curator selects one of eight near-duplicate variants by caption likelihood, without target-specific prompts. Students trained on target-preferring curators' selections show higher mean target-versus-alternatives log-likelihood ratios in an image-free readout than students trained on neutral selections. Under a prespecified decision rule, this directional shift appears for an owl target in LLaVA-1.5, Qwen2.5-VL, Qwen3-VL, and Gemma 4, with curators and students initialized from the same base checkpoint, and target rotations move it toward camels, oak trees, and
    
[^95]: DPO超参数空间的全貌：测量什么、如何随规模变化、以及何时不应迁移

    The Shape of the DPO Hyperparameter Landscape: What to Measure, How It Moves with Scale, and When Not to Transfer It

    [https://arxiv.org/abs/2610.27895](https://arxiv.org/abs/2610.27895)

    通过密集超参数网格实验系统描绘了DPO超参数空间随模型规模和训练步数的演变规律，并识别了最适合超参数选择的基准与优化策略。

    

    直接偏好优化被广泛用于语言模型的后训练，但其性能在很大程度上取决于超参数的选择，例如学习率和参考策略正则化参数β。为了更好地理解DPO超参数空间，我们使用密集的超参数网格，在来自同一系列的不同模型规模上开展了大量实验。这使我们能够分析最优DPO超参数如何随模型大小和更新步数的演变而变化，其中中间检查点提供了一种低成本的信号，可用于加速超参数优化过程。此外，通过在静态和动态基准上评估我们的模型，我们确定了哪些基准能为超参数选择提供最强的信号。最后，我们研究了多种最先进的超参数优化策略在此设置下的表现。

    arXiv:2610.27895v1 Announce Type: cross  Abstract: Direct preference optimisation (DPO) is widely used for language model post-training, but its performance depends strongly on the choice of its hyperparameters, such as the learning rate and the reference-policy regularisation parameter $\beta$. To better understand the DPO hyperparameter landscape, we conduct extensive experiments using a dense hyperparameter grid across different model scales from the same family. This allows us to analyse how the optimal DPO hyperparameters evolve with model size and the number of update steps, with intermediate checkpoints providing an inexpensive signal to accelerate the hyperparameter optimisation process. Furthermore, by evaluating our models on both static and dynamic benchmarks, we identify which benchmarks provide the strongest signal for hyperparameter selection. Finally, we investigate how different state-of-the-art hyperparameter optimisation strategies perform in this setting. Our experim
    
[^96]: 物理变量信息驱动的深度学习用于工业流体输送系统的鲁棒故障诊断

    Physical-Variable-Informed Deep Learning for Robust Diagnosis in Industrial Fluid Transport Systems

    [https://arxiv.org/abs/2610.27891](https://arxiv.org/abs/2610.27891)

    提出物理变量信息驱动的深度学习框架PhyNet，通过物理变量与响应信号之间的交叉注意力机制，在无需严格时间对齐的情况下，实现了工业流体输送系统在真实采样异构条件下的鲁棒故障诊断。

    

    工业流体输送系统通过异构的传感数据流进行监测，但由于响应信号会随运行工况变化，且在采样率、观测时长和数据粒度方面与物理变量存在差异，实现鲁棒的故障诊断仍然十分困难。为应对这些挑战，我们提出了一种物理变量信息驱动的深度学习框架，用于在真实世界的采样异构性条件下实现鲁棒故障诊断。PhyNet将物理变量视为全局运行约束，将响应信号视为局部诊断证据。其物理变量到响应信号的交叉注意力机制将序列级的物理变量标记与可学习的全局响应标记相连接，从而在无需严格时间对齐的情况下实现二者交互。在四个基准测试上，包括三个真实世界的空化数据集和一个公开的两相流数据集，PhyNet在所有评估指标上均取得了最佳性能。

    arXiv:2610.27891v1 Announce Type: cross  Abstract: Industrial fluid transport systems are monitored through heterogeneous sensing streams, but robust fault diagnosis remains difficult because response signals vary with operating conditions and differ from physical variables in sampling rate, observation duration and data granularity. To address these challenges, we propose a physical-variable-informed deep learning framework (PhyNet) for robust fault diagnosis under real-world sampling heterogeneity. PhyNet treats physical variables as global operating constraints and response signals as local diagnostic evidence. Its physical-to-response cross-attention mechanism connects sequence-level physical-variable tokens with a learnable global response token, enabling interaction without strict temporal alignment. Across four benchmarks, including three real-world cavitation datasets and one public two-phase-flow dataset, PhyNet achieves the best performance across all evaluated metrics. Class
    
[^97]: 激发还是扩展？揭秘智能体框架优化的收益来源

    Elicitation or Expansion? Demystifying Gains from Agent Harness Optimization

    [https://arxiv.org/abs/2610.27889](https://arxiv.org/abs/2610.27889)

    本文系统性地分析了智能体框架优化带来的性能提升究竟源于“激发”（在已有能力范围内更可靠地完成任务）还是“扩展”（扩大可解任务的范围），并指出这一区分对递归自我改进和进化搜索至关重要。

    

    近年来，框架（harness）优化的研究表明，在不更新模型权重的情况下，语言模型智能体也能获得显著的性能提升。一个设计良好的框架通过塑造上下文、工具使用和执行工作流，能够更有效地利用底层模型的能力。然而，仅凭朴素的成功率无法充分刻画优化后的框架究竟是在已可完成的任务上取得更可靠的成功（激发，elicitation），还是在更广泛的可解任务上取得成功（扩展，expansion）。这一区分对于递归自我改进和进化搜索至关重要，因为扩展可能提供更广泛的成功经验，从而推动能力边界的拓展。尽管其重要性日益凸显，这一区分在框架优化研究中仍未得到充分探索。我们进行了系统性分析，以研究框架优化带来的收益效果。我们基于强化学习（RL）文献中的pass@k分析（用于比较……）……（摘要在此处不完整）

    arXiv:2610.27889v1 Announce Type: new  Abstract: Recent studies in harness optimization have shown substantial performance gains in language model agents without updating model weights. A well-designed harness more effectively leverages the underlying model's capabilities by shaping context, tool use, and execution workflows. However, na\"ive success rates alone do not fully characterize whether optimized harness yields more reliable success on tasks already within reach (elicitation) or a broader range of solvable tasks (expansion). This distinction is crucial for recursive self-improvement and evolutionary search, as expansion may provide a broader range of successful experiences to push the capability boundaries. Despite the growing significance, this distinction remains underexplored in harness optimization. We conduct a systematic analysis to study the effect of gains from harness optimization. Building on pass@k analyses in the reinforcement learning (RL) literature that compare 
    
[^98]: 基于枚举引导的神经Hamilton-Jacobi安全证书训练

    Enumeration-Guided Training of Neural Hamilton-Jacobi Safety Certificates

    [https://arxiv.org/abs/2610.27870](https://arxiv.org/abs/2610.27870)

    提出一种验证引导的训练框架，通过安全约束的概率枚举生成紧凑且精准的反例集合来微调预训练的Hamilton-Jacobi值函数，从而获得更大的认证安全区域、更快的认证速度，并可在部署时通过屏蔽机制强制保证安全。

    

    Hamilton-Jacobi（HJ）可达性分析可为控制系统提供形式化安全保证，但训练精确的神经证书仍然十分困难，尤其当不安全区域难以刻画时。我们提出了一种验证引导的训练框架，通过使用安全约束的概率枚举来检测前向不变性违反，即学习到的证书错误分类不安全行为的状态。这产生了一个紧凑且具有针对性的反例集合，相比标准对抗攻击方法生成的输入，能更精确地捕捉失败边界。关键的是，利用这种表示来微调预训练的HJ值函数，可在所有测试基准上获得更大的认证区域并缩短认证时间。我们进一步引入了一种基于精炼证书的屏蔽机制，以便在部署时强制执行安全约束。

    arXiv:2610.27870v1 Announce Type: cross  Abstract: Hamilton-Jacobi (HJ) reachability analysis offers formal safety guarantees for control systems, but training accurate neural certificates remains difficult, particularly when unsafe regions are hard to characterize. We propose a verification-guided training framework that addresses this by using probabilistic enumeration of safety constraints to detect forward-invariance violations, i.e., states where the learned certificate incorrectly classifies unsafe behavior. This produces a compact, targeted set of counterexamples that more precisely captures the failure boundary than inputs generated by standard adversarial attack methods. Crucially, using this representation to fine-tune a pretrained HJ value function yields larger certified regions and reduces certification time across all tested benchmarks. We further introduce a shielding mechanism based on the refined certificates to enforce safety constraints at deployment time. We evaluat
    
[^99]: 从词元到文件：AI 参与代码的多粒度检测与定位

    From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code

    [https://arxiv.org/abs/2610.27863](https://arxiv.org/abs/2610.27863)

    提出了一种基于冻结代码语言模型上双低秩适配器逐词元对数似然比的框架，可在词元证据基础上对 AI 参与代码进行行、块、文件多粒度的检测与定位。

    

    随着大语言模型在代码生成中的使用日益增多，亟需能够区分纯人工编写代码与“AI 参与代码”（即文件中任何部分，无论多么微小，由模型生成或经模型实质性修改）的方法。除了标记此类文件之外，我们还旨在定位可归因于 AI 的代码行或代码块。现有方法将文件压缩为单一分数，依赖于表面风格特征或全局代码表示，当文件中仅有少数代码块由 AI 参与时，这种方式会平均化掉相关证据。我们提出了一个在词元级别建模 AI 创作归属、并在行、块和文件三个粒度上进行预测的框架。一个冻结的代码语言模型配备了两个低秩适配器，分别在不同机器编写和人工编写的文件上训练；两者的逐词元对数似然比为 AI 创作归属提供了局部证据。轻量级的预测头利用周围的代码上下文，在行和块内部聚合这些证据……

    arXiv:2610.27863v1 Announce Type: cross  Abstract: The increasing use of large language models for code generation calls for methods that distinguish purely human-written code from AI-involved code: files in which any part, however small, was generated or substantially modified by a model. Beyond flagging such files, we aim to localize the lines or blocks attributable to AI. Existing approaches compress a file into a single score, from surface stylistic features or a global code representation, which averages away the evidence when only a few blocks are AI-involved. We propose a framework that models AI authorship at the token level and predicts at the line, block, and file levels. A frozen code language model is equipped with two low-rank adapters, trained separately on machine-written and human-written files; their per-token log-likelihood ratio provides local evidence of AI authorship. Lightweight heads aggregate this evidence within lines and blocks, using the surrounding code cont
    
[^100]: LaLa-World：基于自然语言动作扩展潜在世界模型

    LaLa-World: Scaling Latent World Models with Natural Language Actions

    [https://arxiv.org/abs/2610.27851](https://arxiv.org/abs/2610.27851)

    LaLa-World通过用语言统一描述动作与视觉变化、并将未来预测器构建于语言模型之上，使单个潜在世界模型能够跨仿真、机器人、图像和视频等异构数据进行联合训练，在MPC规划性能上超越了针对单一领域或任务训练的现有模型。

    

    本文表明，JEPA风格的潜在世界模型（LWM）可以通过采用视频生成模型的语言条件接口，实现跨多样化领域和数据源的学习。我们将这一方法确立为基于想象进行规划以及下游策略微调的有效且高效的基础。为此，我们提出了LaLa-World，一个语言条件潜在世界模型，它从仿真和机器人轨迹以及更广泛的图像和视频数据中学习。两个关键设计使这一工作成为可能：首先，我们用语言来描述数值动作和视觉变化，使得单一的未来预测器能够在异构数据集上进行联合训练；其次，我们将预测器构建在语言模型之上，使其在预测未来潜在状态的同时能够理解语言条件。由此得到的预测器相比针对每个领域或任务单独训练的现有LWM，实现了更高的平均MPC规划性能。

    arXiv:2610.27851v1 Announce Type: cross  Abstract: This paper shows that JEPA-style latent world models (LWMs) can learn across diverse domains and data sources by adopting the language-conditioned interface of video generation models. We establish this approach as an effective and efficient basis for planning with imagination and downstream policy fine-tuning. To this end, we introduce LaLa-World, a Language-conditioned Latent World model that learns from simulation and robot trajectories together with broader image and video data. Two designs make this work. First, we describe both numerical actions and visual changes in language, enabling joint training of a single future predictor across heterogeneous datasets. Second, we build the predictor on a language model, allowing it to interpret language conditions while predicting future latent states. The resulting predictor achieves higher average MPC planning performance than existing LWMs trained separately for each domain or task. Add
    
[^101]: SAPD：无需环境交互的智能体经验学习

    SAPD: Learning from Agent Experience without Environment Interaction

    [https://arxiv.org/abs/2610.27847](https://arxiv.org/abs/2610.27847)

    该论文提出步级优势策略蒸馏（SAPD）方法，通过将离线多轮轨迹分解为单步决策上下文进行在线策略蒸馏，并利用相对步级优势信号，使LLM智能体无需与环境交互即可从离线经验中有效学习，同时避免了SFT的灾难性遗忘和在线强化学习对实时环境交互的依赖。

    

    arXiv:2610.27847v1 公告类型：新 摘要：扩展大语言模型（LLM）智能体的训练通常需要大量可执行环境，而这些环境的构建、维护和运行限制了训练经验的扩展。现有的离线轨迹为复用交互经验提供了机会，然而在不重新访问这些环境的情况下有效改进当前策略仍然具有挑战性：监督微调（SFT）可能导致灾难性遗忘，而在线强化学习需要与环境进行实时交互。为应对这一挑战，我们提出了步级优势策略蒸馏（SAPD），这是一种利用相对步级优势从离线轨迹中学习的方法。SAPD将多轮轨迹分解为单个决策上下文以进行在线策略蒸馏，使学生模型能够基于记录的历史生成响应，并引入相对步级优势信号来……

    arXiv:2610.27847v1 Announce Type: new  Abstract: Scaling the training of Large Language Model (LLM) agents typically requires a large collection of executable environments, whose construction, maintenance, and operation constrain the expansion of training experience. Existing offline trajectories offer an opportunity to reuse interaction experience, yet effectively refining the current policy without revisiting these environments remains challenging: supervised fine-tuning (SFT) can lead to catastrophic forgetting, while online reinforcement learning requires real-time interaction with the environment. To address this challenge, we propose Step-Advantage Policy Distillation (SAPD), a method for learning from offline trajectories using relative step advantages. SAPD decomposes multi-turn trajectories into individual decision contexts for on-policy distillation, allowing the student to generate responses conditioned on recorded histories, and introduces relative step advantage signals to
    
[^102]: 从通用到专用：一种面向在线持续学习的新型优化器

    From Generic to Dedicated: A Novel Optimizer for Online Continual Learning

    [https://arxiv.org/abs/2610.27835](https://arxiv.org/abs/2610.27835)

    该论文提出新型优化器SPIN，通过在快速“可塑性”序列与自适应“稳定性”序列之间进行插值来显式解耦在线持续学习中的可塑性与稳定性，有效缓解灾难性遗忘。

    

    在线持续学习（OCL）要求模型在严格的单次遍历约束下从非平稳数据流中学习，这使得模型极易受到灾难性遗忘的影响。虽然现有研究已从数据、架构和优化等角度探索了多种策略，但优化器往往直接采用SGD和AdamW等通用方法。在这项工作中，我们通过将Polyak-Ruppert平均重塑为“可塑性”的引擎、将原始平均作为“稳定性”的锚点，揭示了优化动力学与在线持续学习需求之间的新联系。这一发现启发了我们提出的新型优化器SPIN（Stability-Plasticity INterpolation，稳定性-可塑性插值），它通过在快速移动的“可塑性”序列与自适应的“稳定性”序列之间进行插值，显式地解耦了可塑性与稳定性。我们从理论上证明，SPIN隐式地采用了逆Hessian近似，提供了关键的Tikhonov正则化来抑制带噪的更新……

    arXiv:2610.27835v1 Announce Type: new  Abstract: Online Continual Learning (OCL) requires models to learn from non-stationary data streams under a strict single-pass constraint, making them highly susceptible to catastrophic forgetting. While existing studies have explored various strategies from data, architecture, and optimization perspectives, the optimizer often directly uses generic approaches such as SGD and AdamW. In this work, we reveal a new link between optimization dynamics and OCL needs by recasting Polyak-Ruppert averaging as the engine of "plasticity" and Primal averaging as the anchor of "stability". This inspires our novel optimizer, SPIN (Stability-Plasticity INterpolation), which explicitly decouples plasticity and stability by interpolating between a fast-moving "plasticity" sequence and an adaptive "stability" sequence. We theoretically demonstrate that SPIN implicitly employs an inverse Hessian approximation, providing crucial Tikhonov regularization to damp noisy,
    
[^103]: ProVe：提议者-验证者解码

    ProVe: Proposer-Verifier Decoding

    [https://arxiv.org/abs/2610.27815](https://arxiv.org/abs/2610.27815)

    ProVe反转了推测解码中提议者与验证者的角色，让大模型通过单次前向传播直接提议token块并加以验证，无需训练独立草稿头即可实现2.5-2.7倍推理加速，且无损版本能完全复现原始贪婪解码结果。

    

    推测解码通过让小模型串行起草token、大模型并行验证来加速自回归（AR）语言模型。我们反转了这两个角色。从一个标准的AR检查点出发，我们将大模型微调为一个提议者，使其能够在单次前向传播中预测未来的一个token块，然后用一次额外的前向传播来验证所提议的块。这就是ProVe，即提议者-验证者解码。验证者可以是一个较小的AR模型，也可以是提议者本身；后者构成ProVe-lossless，它能够复现提议者的贪婪AR解码结果。块提议本身已经提供了可观的加速空间：当块大小为16时，在GSM8K上提议块与AR解码的平均匹配前缀达到6.6个token。在六个基准测试上，使用32B提议者，ProVe-lossless无需训练单独的草稿头即可达到平均2.5倍的AR吞吐量。使用较小的3B验证器时，ProVe达到平均2.7倍的吞吐量。

    arXiv:2610.27815v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive (AR) language models by letting a small model draft tokens serially and a large model verify them in parallel. We reverse these roles. Starting from a standard AR checkpoint, we fine-tune the large model into a proposer that predicts a block of future tokens in a single forward pass, then use one additional forward pass to verify the proposed block. This gives ProVe, Proposer-Verifier decoding. The verifier can be a smaller AR model or the proposer itself; the latter gives ProVe-lossless, which reproduces the proposer's greedy AR decoding. Block proposals already provide substantial room for acceleration: with a block size of 16, the proposed block has an average AR-matching prefix of 6.6 tokens on GSM8K. Across six benchmarks with a 32B proposer, ProVe-lossless reaches 2.5x average AR throughput without training a separate draft head. With a small 3B verifier, ProVe reaches 2.7x average th
    
[^104]: SliMOO：面向大语言模型深度剪枝的可解释多目标进化搜索

    SliMOO: Interpretable Multi-Objective Evolutionary Search for LLM Depth Pruning

    [https://arxiv.org/abs/2610.27812](https://arxiv.org/abs/2610.27812)

    SliMOO将LLM深度剪枝表述为多目标优化问题，通过单层移除探测构建可解释的任务-层重要性图谱，并结合多目标进化搜索，在压缩模型的同时保留跨任务的广泛能力。

    

    随着大语言模型（LLM）的能力不断增强，其庞大的规模使得模型服务成本高昂。深度剪枝通过直接减少内存占用和推理延迟，且无需特定硬件支持，为提升效率提供了一条实用的途径。然而，我们观察到Transformer的各层表现出显著不同的任务相关性层级贡献，而现有的深度剪枝方法忽视了这种异质性，因此往往只针对狭窄的领域或基准进行优化，而无法保留模型广泛的能力。这表明LLM的深度剪枝应被表述为一个多目标优化问题，而非单分数排序问题。我们提出了SliMOO，一个用于LLM深度剪枝的可解释多目标进化搜索框架。SliMOO首先利用单层移除探测构建任务-层重要性图谱，该图谱提供了任务共享层与任务特定层的可解释视图，并作为搜索的先验信息，随后在此基础上执行……

    arXiv:2610.27812v1 Announce Type: new  Abstract: As large language models (LLMs) grow more powerful, their scale makes them costly to serve. Depth pruning offers a practical route to efficiency by directly reducing memory and latency without hardware-specific support. However, we observe that transformer layers exhibit markedly different task-dependent layer contributions, whereas existing depth-pruning methods ignore this heterogeneity and thus tend to optimize for a narrow domain or benchmark rather than preserve broad capability. This suggests that LLM depth pruning should be formulated as a multi-objective optimization problem rather than a single-score ranking problem.   We propose SliMOO, an interpretable multi-objective evolutionary search framework for LLM depth pruning. SliMOO first uses single-layer removal probing to construct a task-layer importance map, which provides an interpretable view of task-shared and task-specific layers and serves as a search prior. It then perfor
    
[^105]: 纠错码的认证自适应列表译码

    Certified Adaptive List Decoding for Error-Correcting Codes

    [https://arxiv.org/abs/2610.27805](https://arxiv.org/abs/2610.27805)

    该论文将译码建模为吸收态离散扩散模型上的概率束搜索，并结合线性规划对偶界与基于距离的检验逐帧认证全局最大似然解，在15种码上可认证高达99.9%的帧且无需在精度与计算量之间权衡。

    

    线性分组码的最大似然（ML）译码是NP难问题，因此神经译码器通常以其接近最大似然译码的程度来评估。我们观察到，证明一个候选解是最优的往往远比找到它更容易，并据此构建了一个既能找到候选解又能证明其最优性的译码器。我们将译码建模为在一个吸收态离散扩散模型上的搜索，该模型通过逐坐标的顺序承诺将生成过程转化为一棵树，并在其上运行概率束搜索。随后，由线性规划松弛导出的对偶界和基于距离的检验（二者均从码结构本身推导得出）逐帧认证当前候选解是全局ML解；由于两者都是定理，被认证的帧不会产生精度与计算量之间的权衡。在15种码和5个工作点上，我们认证了高达99.9%的帧，且两种检验按其推导所预测的方式划分解空间：一种对稀疏奇偶校验矩阵的码强有效，另一种对具有大最小距离的码强有效。

    arXiv:2610.27805v1 Announce Type: cross  Abstract: Maximum-likelihood (ML) decoding of linear block codes is NP-hard, so neural decoders are evaluated by how closely they approach it. We observe that proving a candidate optimal can be far easier than finding it, and build a decoder that does both. We cast decoding as a search over an absorbing-state discrete diffusion model, whose sequential commitment of coordinates turns generation into a tree, and run a probability beam search over it. A dual bound from a linear-programming relaxation and a distance-based test, both derived from the code, then certify per frame that the current candidate is the global ML solution; because both are theorems, certified frames incur no accuracy-computation trade-off. Across $15$ codes and five operating points we certify up to $99.9\%$ of frames, the two tests dividing the space as their derivations predict, one strong for sparse parity-check matrices and the other for codes with a large minimum distan
    
[^106]: TerraNostra-Bench：面向全球1公里分辨率土地利用重建的空间降尺度与时间滚动预测基准

    TerraNostra-Bench: Benchmarking Spatial Downscaling and Temporal Rollout for Global 1-km Land-Use Reconstruction

    [https://arxiv.org/abs/2610.27793](https://arxiv.org/abs/2610.27793)

    该论文提出了TerraNostra-Bench，这是一个变化感知的评估基准，将空间降尺度与自回归时间滚动预测解耦，并聚焦于稀少的土地利用变化像素进行评分，从而推动数据驱动方法生成全球1公里分辨率的年度土地利用地图。

    

    陆地碳循环的不确定性仍然是气候预测中的主要制约因素，部分原因在于地球系统模型对陆地表面的刻画不够完善。我们提出了一种数据驱动的高分辨率土地利用重建框架，将其视为语义分割任务，该框架结合粗分辨率情景数据、气候分类与静态地球物理特征，生成1公里分辨率的年度土地利用地图。我们提出了TerraNostra-Bench，这是一种变化感知的评估协议，它将空间降尺度与自回归时间滚动预测分离开来，并在稀少的土地利用变化像素而非占主导地位的静态背景上对模型进行评分。利用该基准，我们比较了卷积（U-Net）和Transformer（Swin）骨干网络在自回归与非自回归方案下的表现，并与持续性、线性以及浅层卷积基线进行了对比。自回归先验是信息量最大的输入，但会累积……（原文摘要截断）

    arXiv:2610.27793v1 Announce Type: cross  Abstract: Uncertainty in the terrestrial carbon cycle remains a major constraint in climate projections, partly due to imperfect land surface representations in Earth system models. We propose a data-driven framework for high-resolution land-use reconstruction, treated as a semantic-segmentation task, that combines coarse-resolution scenario data and climate classification with static geophysical features to produce annual land-use maps at 1 km resolution. We introduce TerraNostra-Bench, a change-aware evaluation protocol that separates spatial downscaling from autoregressive temporal rollout and scores models on the rare land-use-change pixels rather than the dominant static background. Using it, we compare convolutional (U-Net) and transformer (Swin) backbones in autoregressive and non-autoregressive schemes against persistence, linear, and shallow convolutional baselines. The autoregressive prior is the most informative input yet accumulates 
    
[^107]: 缩放定律的基础

    Foundations of the Scaling Law

    [https://arxiv.org/abs/2610.27789](https://arxiv.org/abs/2610.27789)

    本论文通过几何近似、谱分析和约束优化三个相互关联的分析框架，为缩放定律建立了理论基础，并区分解释了涌现、顿悟和测试时缩放等现象。

    

    经验缩放定律将验证损失与模型规模、训练数据量和计算量联系起来，为资源分配提供了定量基础。本技术报告通过三个相互关联的分析来解释这些关系：几何近似将空间分辨率、内在维度和平滑度与函数误差联系起来，而损失函数决定了该误差如何传递为任务风险；谱分析结合特征可见性、协方差特征值和目标对齐，描述了更多数据如何减少尚未解决的预测不确定性；约束优化则给出了参数与数据的平衡以及训练与推理之间的权衡。我们还研究了涌现与顿悟现象，将评分阈值与学习动力学区分开来；并研究了测试时缩放，将候选答案的覆盖范围与所选答案的准确率区分开来。在视觉和分子预测上的应用展示了领域特定的坐标选择如何改变资源比较方式。

    arXiv:2610.27789v1 Announce Type: new  Abstract: Empirical scaling laws relate validation loss to model size, training exposure and compute, providing a quantitative basis for allocating resources. This technical note explains these relationships through three linked analyses. Geometric approximation connects spatial resolution, intrinsic dimension and smoothness to function error; the loss function determines how that error transfers to task risk. Spectral analysis combines feature visibility, covariance eigenvalues and target alignment to describe how additional data reduce unresolved predictive mass. Constrained optimization then gives the parameter-data balance and the training-inference trade-off. We also examine emergence and grokking, separating scoring thresholds from learning dynamics, and test-time scaling, separating candidate coverage from selected-answer accuracy. Applications to vision and molecular prediction show how domain-specific coordinates change the resource compa
    
[^108]: ℓ₁攻击需要多大程度的集中扰动？

    How Much Do $\ell_1$ Attacks Need Concentrated Perturbations?

    [https://arxiv.org/abs/2610.27773](https://arxiv.org/abs/2610.27773)

    研究发现即使对每个坐标施加扰动上限以抑制集中扰动，ℓ₁对抗攻击仍能保留约97%的攻击成功率，表明其成功并不依赖高度集中的大幅扰动。

    

    全局ℓ₁预算限制的是攻击者可以改变图像的程度，而非这种改变如何分布。我们探究当限制大幅度的单坐标变化时，ℓ₁攻击所观察到的成功还能保留多少。我们在预算中加入了每个坐标的上限约束，通过精确的循环内投影来强制执行联合约束，并在不改变其优化目标的情况下对APGD-ℓ₁、SLIDE和FMN-ℓ₁这三种方法扫描该上限。在18个标准模型设置中，不超过32/255的上限相对于各方法原本无上限的基线保留了中位数97%的攻击成功率，同时扰动峰值幅度的中位数下降了80%，有效支持范围的中位数几乎增至三倍。在匹配的实际ℓ₁范数下，受上限约束的扰动在结构上依然与众不同；单独的ImageNet比较也显示，在选定的上限下LPIPS和DISTS感知距离更低。直接的循环内约束优化比事后裁剪能多保留高达88个百分点的攻击成功率。

    arXiv:2610.27773v1 Announce Type: cross  Abstract: A global $\ell_1$ budget limits how much an adversary may change an image, not how that change is distributed. We ask how much of an $\ell_1$ attack's observed success remains when large coordinate changes are restricted. We add a per-coordinate cap to the budget, enforce the joint constraint by exact in-loop projection, and sweep the cap for APGD-$\ell_1$, SLIDE, and FMN-$\ell_1$ without altering their objectives. Across 18 standard-model settings, caps of at most $32/255$ retain a median $97\%$ of attack success relative to each method's native uncapped baseline, while median peak magnitude falls by $80\%$ and median effective support nearly triples. At matched realized $\ell_1$, capped perturbations remain structurally distinct; separate ImageNet comparisons also show lower LPIPS and DISTS at selected caps. Direct in-loop constrained optimization preserves up to $88$ percentage points more attack success than post-hoc clipping, high
    
[^109]: 印记记忆网络：受大脑启发的原型解释

    Engram Memory Network: Brain-Inspired Prototype Explanations

    [https://arxiv.org/abs/2610.27770](https://arxiv.org/abs/2610.27770)

    该论文受人类记忆系统启发，提出了模拟大脑腹侧视觉流和下颞叶皮层的印记记忆网络（EMN），其核心是一个受Hopfield启发的记忆模块，通过迭代检索代表性原型，在保持分类准确率的同时提升了可解释AI在不同数据集上的解释质量。

    

    人工智能系统正越来越多地被部署于高风险决策场景中。为此，可解释人工智能领域基于原型的模型因能够检索视觉上相似的样本作为忠实解释而受到日益广泛的关注。尽管这些模型取得了具有竞争力的分类准确率，但在特征各异的数据集上，其解释质量往往会下降，这限制了它们的实际应用。我们从认知神经科学中汲取灵感，尤其是人类记忆系统——它将输入相关的信息编码为支持识别任务的表征。我们提出了印记记忆网络（EMN），该模型模拟了人类大脑中的腹侧视觉流和下颞叶皮层。该模型的核心是一个受Hopfield启发的记忆模块，能够迭代地检索具有代表性的原型。这些经过记忆精炼的潜在表征同时被用于重建以及……（摘要内容不完整，在此处截断）

    arXiv:2610.27770v1 Announce Type: new  Abstract: Artificial intelligence systems are increasingly deployed in high-stakes decision-making. In response, prototype-based models in explainable AI have attracted growing attention for their ability to retrieve visually similar examples as faithful explanations. Although these models achieve competitive classification accuracy, they often exhibit degraded explanation quality across datasets with varying characteristics, limiting their practical applicability. We draw on insights from cognitive neuroscience, particularly the human memory system, which encodes input-relevant information into representations that support recognition. We propose an Engram Memory Network (EMN), a model that emulates the ventral visual stream and inferior temporal cortex in the human brain. Central to this model is a Hopfield-inspired memory module that iteratively retrieves representative prototypes. These memory-refined latents are used for both reconstruction a
    
[^110]: 面向可扩展且可迁移的分布式光伏预测的基础模型式预训练

    Foundation-Style Pretraining for Scalable and Transferable Distributed PV Forecasting

    [https://arxiv.org/abs/2610.27763](https://arxiv.org/abs/2610.27763)

    本文提出一种面向光伏领域的基础模型式预训练框架，通过将NWP辅助预测统一建模为序列补全问题并设计多种预训练目标，从多站点、多时域异构数据中学习共享预测模型，实现分布式光伏预测的可扩展性与对未见站点的迁移能力。

    

    分布式光伏发电的快速增长使得对大量地理分散电站进行精确预测的需求日益增加。现有方法通常针对单个站点或特定预测时域开发，随着新电站的接入需要重复训练和维护，并限制了在未见过的站点上的直接复用。为解决这一可扩展性差距，本文提出了一种面向特定领域的基础模型式预训练框架，从异构的多站点、多时域光伏数据中学习一个共享的预测模型。该框架将数值天气预报（NWP）辅助预测表述为一个序列补全问题，统一了历史光伏功率、气象观测、未来NWP预报和时间编码。设计并系统评估了三种预训练目标，包括预测对齐重建、通用重建以及……（原文摘要在此处截断）

    arXiv:2610.27763v1 Announce Type: new  Abstract: The rapid growth of distributed photovoltaic (PV) generation is increasing the demand for accurate forecasting across large numbers of geographically dispersed stations. Existing approaches are commonly developed for individual sites or specific forecasting horizons, requiring repeated training and maintenance as new stations are integrated and limiting direct reuse at previously unseen sites. To address this scalability gap, this paper presents a domain-specific foundation-style pretraining framework that learns a shared forecasting model from heterogeneous multi-site and multi-horizon PV data. NWP-assisted forecasting is formulated as a sequence-completion problem that unifies historical PV power, meteorological observations, future numerical weather prediction (NWP) forecasts, and time encodings. Three pretraining objectives are designed and systematically evaluated, including forecast-aligned reconstruction, generic reconstruction, a
    
[^111]: 预算有限的AC/DC——交替稀疏阶段

    AC/DC on a Budget - Alternating Sparse Phases

    [https://arxiv.org/abs/2610.27758](https://arxiv.org/abs/2610.27758)

    本文提出ARC方法，通过在密集阶段零初始化遮蔽参数、缩短密集阶段并用仅更新掩码和高梯度权重的稀疏重连阶段完全替代密集阶段，以极小的计算成本实现了与AC/DC稀疏化基线相当的性能。

    

    最先进的神经网络稀疏化方法依赖于密集训练阶段与稀疏训练阶段的交替进行。尽管密集阶段的计算成本高昂，但其对性能至关重要。我们的实证分析揭示了这一增益背后的机制：当重新引入时，之前被遮蔽的参数会接收到不成比例的大梯度，从而驱动关键的掩码重连和参数符号翻转。基于这些洞察，我们通过三项创新优化了性能与FLOPs之间的帕累托前沿：(1) 在密集阶段将遮蔽参数零初始化以促进相关的符号翻转；(2) 缩短密集阶段；(3) 完全用稀疏重连阶段取代密集阶段，该阶段仅更新活跃掩码以及掩码外权重中一小部分高梯度权重。在CIFAR100和ImageNet上的评估表明，我们的变体ARC（交替重连压缩）仅需一小部分计算量即可达到与基线AC/DC相当的性能。

    arXiv:2610.27758v1 Announce Type: new  Abstract: State-of-the-art neural network sparsification relies on alternating dense and sparse training phases. While computationally expensive, dense phases are critical for performance. Our empirical analysis reveals the mechanics behind this gain: upon reintroduction, previously masked parameters receive disproportionately large gradients that drive essential mask rewiring and parameter sign-flipping. Building on these insights, we optimize the performance-to-FLOPs Pareto frontier through three innovations: (1) zero-initializing masked parameters during dense phases to promote relevant sign flips, (2) shortening dense phases, and (3) replacing the dense phase entirely with a sparse rewiring phase that updates only the active mask and a small, high-gradient fraction of out-of-mask weights. Evaluated on CIFAR100 and ImageNet, our variant ARC (Alternating Rewiring Compression) matches baseline AC/DC performance at a fraction of the computational 
    
[^112]: 给模型一根手指：视觉语言模型计数能力的发展对齐评估

    Give the Model a Finger: Developmentally Aligned Evaluation of Counting in Vision-Language Models

    [https://arxiv.org/abs/2610.27756](https://arxiv.org/abs/2610.27756)

    该论文受儿童计数发展过程启发，提出一种无需训练的智能体评估框架，让视觉语言模型通过“指认标记”等认知动作进行计数，将模型通常隐藏的串行视觉推理过程转化为可解释、可审计的行为分析。

    

    视觉语言模型（VLM）在小集合上表现精确，而在较大集合上则退化为估计，这一特征与尚未掌握计数的儿童颇为相似。儿童通过认知动作跨越这一障碍：一边念数词一边指向每个物品，把已数过的物品移到一边，或对物品阵列进行分组。我们以无需训练的智能体循环方式，为VLM提供同样的动作——这些动作只是不具备自身视觉智能的“哑”图像工具。我们的核心主张在于测量而非干预：一个通过标记进行计数的模型，会将其通常隐藏的串行过程留在图像上，而计数的发展性原则将这些标记转变为对串行视觉推理的可解释审计：每个标记都会依据点级真值被分类为命中、重复、幻影或遗漏；答案会与标记数量进行核对；顺序无关性则通过强制相反的遍历顺序来检验。在9个模型、682张图像的实验中……

    arXiv:2610.27756v1 Announce Type: new  Abstract: Vision-language models (VLMs) are exact on small sets and drift into estimation on larger ones, a profile reminiscent of children before they master counting. Children cross the barrier with epistemic actions: pointing at each item while reciting number words, moving counted items aside, partitioning the array. We give VLMs the same actions as dumb image tools with no visual intelligence of their own in a training-free agent loop. Our central claim is about measurement, not intervention: a model that counts by marking leaves a normally hidden serial procedure on the image, and the developmental principles of counting turn those marks into an interpretable audit of serial visual reasoning: every tag is classified against point-level ground truth as on-target, duplicate, phantom, or omitted, the answer is checked against the number of tags, and order irrelevance is tested by forcing opposite traversal orders. Across 9 models, 682 images, a
    
[^113]: 通过热力学奇异性检测扩散模型中的记忆化现象

    Detection of Memorization in Diffusion Models Through Thermodynamic Singularity

    [https://arxiv.org/abs/2610.27754](https://arxiv.org/abs/2610.27754)

    提出一种基于热力学奇异性的判据，能够可靠区分扩散模型是真正学习了数据分布还是发生了对单个训练样本的记忆化。

    

    在扩散模型中，正确的分布学习要求对底层密度函数上的扩散过程进行逆转，而一个关键的失败模式是记忆化，即逆向过程反而从单个训练样本中学习扩散。我们对这类发生记忆化的逆向过程提供了分析性刻画，并提出了一种检测相关热力学奇异性的判据。该判据在一系列旨在促进或抑制记忆化的神经网络模型上，能够可靠地区分正确学习的网络与发生记忆化的网络。

    arXiv:2610.27754v1 Announce Type: new  Abstract: In diffusion models, proper distributional learning entails reversing a diffusion process over underlying density functions, whereas a key failure is memorization, in which the reverse process instead learns diffusion from individual training samples. We provide an analytical characterization of such memorized reverse processes, along with a criterion that detects the associated thermodynamic singularity. The proposed criterion reliably distinguishes between properly learned and memorizing networks across a range of neural network models designed to promote or mitigate memorization.
    
[^114]: 强凸二次型问题上Muon类谱优化器的常数步长分析

    Constant Step Sizes for Muon-Type Spectral Optimizers on Strongly Convex Quadratics

    [https://arxiv.org/abs/2610.27747](https://arxiv.org/abs/2610.27747)

    本文通过增益和长度两个关键量，将Muon类谱优化器在强凸二次问题上的行为划分为三种机制，揭示了增益有界时按最大曲率设定步长可收敛、增益无界但长度消失时按总曲率设定步长可收敛、而长度非消失时几乎不收敛的收敛规律。

    

    Muon及相关优化器通过谱变换后的梯度来更新权重矩阵。在这项工作中，我们研究了此类谱优化器在强凸矩阵二次型问题上使用常数步长的表现，该设置将更新本身的效果与非凸性、噪声和步长调度等干扰因素隔离开来。我们通过两个量来分析这类更新：增益，即更新沿每个奇异方向放大梯度的倍数；以及长度，即更新本身的范数。这两个量将流行的谱更新划分为三种具有不同行为特性的机制：(i) 当增益有界时，迭代过程类似于预条件梯度下降，且在由最大曲率所决定的步长下收敛；(ii) 当增益无界但长度趋于消失时，迭代仍然收敛，但可容许的步长由总曲率决定；(iii) 当长度不趋于消失时，迭代在几乎任何步长下都不收敛。

    arXiv:2610.27747v1 Announce Type: new  Abstract: Muon and related optimizers update a weight matrix with a spectrally transformed gradient. In this work, we study such spectral optimizers with a constant step size on strongly convex matrix quadratics, which isolate the effect of the update from the confounds of nonconvexity, noise, and step size schedules. We analyze an update using two quantities: the gain, which is the factor by which it amplifies the gradient along each singular direction, and the length, which is the norm of the update. These two quantities place the popular spectral updates in three regimes with different behavior: (i) with a bounded gain, the iterates behave like preconditioned gradient descent and converge for step sizes set by the largest curvature; (ii) with an unbounded gain but a vanishing length, they still converge, with the admissible step sizes set by the total curvature instead; and (iii) with a non-vanishing length, they do not converge for almost ever
    
[^115]: 基于二维卷积神经网络的脑电图催眠易感性分类

    EEG-Based Classification of Hypnotic Susceptibility Using Two-Dimensional Convolutional Neural Networks

    [https://arxiv.org/abs/2610.27738](https://arxiv.org/abs/2610.27738)

    本研究利用二维卷积神经网络直接分析最小预处理的脑电信号，结合滑动窗口框架与留一被试交叉验证，实现了无需人工特征提取且与被试无关的催眠易感性客观分类。

    

    引导意象是一种广泛应用于临床和治疗场景的思维放松技术。客观评估患者是否已进入放松状态对临床医生而言仍是一项挑战。本研究探讨了基于脑电图（EEG）的催眠易感性分类，采用二维卷积神经网络（2D CNN）对仅经过最小预处理的信号进行分析，无需人工特征提取。数据采集自62名男性受试者（32名对照组，30名催眠组），使用256通道高密度阵列放大器在引导意象或认知任务条件下记录。研究采用滑动窗口框架，在七个解剖学定义的脑区和124个十秒时间窗口上评估了多达868个独立模型。先通过片段级随机划分进行筛选，随后采用留一被试交叉验证以获得与被试无关的性能估计，并通过集成方法结合预……（摘要在此处截断）

    arXiv:2610.27738v1 Announce Type: new  Abstract: Guided Imagery (GI) is a mind-based relaxation technique widely used in clinical and therapeutic settings. Objectively assessing whether a patient has entered a relaxed state remains a challenge for clinicians. This study investigates electroencephalography (EEG)-based classification of hypnotic susceptibility using two-dimensional Convolutional Neural Networks (2D CNNs) applied to minimally preprocessed signals, without manual feature extraction. Data were collected from 62 male participants (32 Control, 30 Hypno) using a 256-channel dense array amplifier during Guided Imagery or cognitive task conditions. A sliding-window framework evaluated up to 868 independent models across seven anatomically defined brain regions and 124 ten-second time windows. Segment-level random-split screening was followed by Leave-One-Subject-Out (LOSO) cross-validation to obtain subject-independent performance estimates, and an ensemble approach combined pre
    
[^116]: 学习型屏障证书的任意时刻有效运行时监控

    Anytime-Valid Runtime Monitoring of Learned Barrier Certificates

    [https://arxiv.org/abs/2610.27726](https://arxiv.org/abs/2610.27726)

    该论文提出一种任意时刻有效的运行时监控框架，通过将屏障漂移条件转化为基于非负检验上鞅的按回合序贯假设检验，利用观测到的闭环轨迹在线审计学习型屏障证书的统计有效性，并在任意停止时间下实现有限样本误报控制。

    

    学习到的屏障证书为随机系统的安全性验证提供了一种原则性机制，但当部署时的底层动力学与训练时所用的动力学不同，其有效性可能在部署过程中逐渐退化。我们提出了一种任意时刻有效的运行时监控器，利用观测到的闭环轨迹对学习到的屏障证书进行审计。该监控器旨在检测已部署的屏障条件失去统计支持的情况；其本身并不保证能在进入不安全状态之前完成检测，也不保证后续干预措施的安全性。具体而言，我们将条件屏障漂移条件表述为按回合进行的序贯假设检验，并将观测到的屏障增量转换为非负的检验上鞅，从而在任意停止时间下实现有限样本的误报控制。受有限时域安全性验证的启发，我们进一步对监控统计量引入了确定性的时间整形方法……（原文摘要在此处被截断）

    arXiv:2610.27726v1 Announce Type: cross  Abstract: Learned barrier certificates provide a principled mechanism for establishing safety of stochastic systems, but their validity may deteriorate during deployment when the underlying dynamics differ from those used for training. We propose an anytime-valid runtime monitor that audits a learned barrier certificate using observed closed-loop trajectories. The monitor is intended to detect loss of statistical support for the deployed barrier condition; it does not by itself guarantee detection before unsafe entry or safety of a subsequent intervention. Specifically, the conditional barrier-drift condition is formulated as an episode-wise sequential hypothesis test, and observed barrier increments are transformed into a nonnegative test supermartingale, yielding finite-sample false-alarm control under arbitrary stopping times. Inspired by finite-horizon safety verification, we further introduce a deterministic temporal shaping of the monitori
    
[^117]: 经验记忆何时能帮助闭环LLM驾驶智能体？基于重采样预热池、令牌匹配填充器与审计基线的受控评估

    When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines

    [https://arxiv.org/abs/2610.27714](https://arxiv.org/abs/2610.27714)

    本研究通过在审计的HighwayEnv测试平台上控制预热池抽样与提示令牌两大混淆因素，发现经验记忆为闭环LLM驾驶智能体带来的表面收益主要来自预热经验池的随机抽样，而非记忆机制本身。

    

    经验记忆何时能帮助闭环LLM驾驶智能体？已发表的收益无法回答这一问题，因为我们发现最关键的两个因素——预热池中包含哪些预热经验、以及注入的令牌对提示词产生什么影响——在现有研究中均未被控制。我们通过在一个经过审计的HighwayEnv测试平台（1Hz，决定性对比使用gpt-4o-mini）上进行受控评估来回答这一问题。我们将预热池视为重采样的随机因素（十个互不相交的池），并加入令牌匹配的无关填充控制、经oracle认证的诊断、谨慎定价指标以及对基线移植版本的量化审计。探测套件涵盖平文本检索（DiLu移植版）、单帧网格移植版（受RESPOND启发）、我们的情景时间图记忆、无需检索的整合规则手册以及平铺的K步观察历史基线。由此得出四项发现，每项均为该测试平台的特性。(1) 报告中的记忆收益归属于预热抽样的运气。代码库……

    arXiv:2610.27714v1 Announce Type: new  Abstract: When does experience memory help a closed-loop LLM driving agent? Published gains cannot say, because the two factors we find to matter most, which warmup experiences the pool contains and what any injected tokens do to the prompt, are controlled in none of them. We answer with a controlled evaluation on one audited HighwayEnv testbed (1Hz, gpt-4o-mini for the decisive contrasts). We treat the warmup pool as a resampled random factor (ten disjoint pools) and add token-matched irrelevant-filler controls, oracle-certified diagnostics, caution-pricing metrics, and quantified audits of the baseline ports. The probe suite spans flat-text retrieval (a DiLu port), a single-frame grid port (RESPOND-inspired), our episodic temporal-graph memory, a retrieval-free consolidated rulebook, and a flat K-step observation-history baseline. Four findings follow, each a property of this testbed. (1) Headline memory gains belong to the warmup draw. The repo
    
[^118]: 经验去噪器的隐私核算

    Privacy Accounting for Empirical Denoisers

    [https://arxiv.org/abs/2610.27710](https://arxiv.org/abs/2610.27710)

    本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。

    

    扩散模型已成为现代生成系统的基础，但其强大的记忆能力引发了隐私方面的担忧。虽然差分隐私（DP）训练能够提供形式化的保证，但其代价是样本质量的下降。在本工作中，我们研究了一个与之互补的问题：在采样过程本身中会产生多少隐私损失？直接分析现代神经去噪器是具有挑战性的，因为它们对单个训练样本的敏感性通常是难以计算的。为了克服这一挑战，我们构建并分析了一种经过截断的隐私可问责经验去噪器——这是一种无需训练的贝叶斯最优去噪器估计器，其每一步的敏感性都可以被显式计算，从而使每个去噪步骤都能成为一个高斯机制。这一视角使得在以下情况下进行隐私核算成为可能：当每个训练样本对每个去噪步骤都有贡献时，可在高斯DP下进行隐私核算；当每个……（摘要在此处被截断）

    arXiv:2610.27710v1 Announce Type: new  Abstract: Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under $f$-DP trade-off functions when each 
    
[^119]: 面向资源感知智慧城市分类的多目标进化半监督学习

    Multi-Objective Evolutionary Semi-Supervised Learning for Resource-Aware Smart City Classification

    [https://arxiv.org/abs/2610.27703](https://arxiv.org/abs/2610.27703)

    本文提出多目标进化半监督学习方法MOEA-SSL，在智慧城市表格分类任务中联合优化预测性能与特征使用量，在仅1%-10%标注数据的条件下取得了与现有轻量级方法相当的分类效果，并显著降低了特征使用量。

    

    智慧城市应用通常将稀缺的标注数据与大量的未标注观测数据相结合。半监督学习（SSL）能够同时利用标注数据和未标注数据，在有限监督条件下提升分类性能。本文提出了MOEA-SSL，这是一种面向轻量级半监督表格分类的多目标进化方法，可联合优化预测质量与特征使用量。在三个智慧城市分类数据集上，分别考虑1%、5%和10%标注数据比例的实验评估表明，MOEA-SSL与现有的成熟轻量级SSL方法相比取得了具有竞争力的预测性能，同时在所有评估条件下相对于EA-SSL显著降低了特征使用量。计算得到的帕累托前沿还提供了特征使用量更低、且分类性能仅有有限下降的替代方案。

    arXiv:2610.27703v1 Announce Type: new  Abstract: Smart-city applications often combine scarce labeled data with abundant unlabeled observations. Semi-supervised learning (SSL) exploits both labeled and unlabeled data to improve classification under limited supervision. This article presents MOEA-SSL, a multi-objective evolutionary approach for lightweight semi-supervised tabular classification that jointly optimizes predictive quality and feature usage. Experimental evaluation on three Smart City classification datasets, considering 1\%, 5\%, and 10\% labeled data, shows that MOEA-SSL achieves competitive predictive performance compared with established lightweight SSL methods while significantly reducing feature usage relative to EA-SSL in all evaluated conditions. The computed Pareto fronts also provide lower-feature-use alternatives with limited degradation in classification performance.
    
[^120]: 理解标签噪声下的数据增强

    Understanding Data Augmentation under Label Noise

    [https://arxiv.org/abs/2610.27701](https://arxiv.org/abs/2610.27701)

    本文从理论上刻画了标签噪声下数据增强有益或有害的条件，并揭示了数据增强实质上是不确定性放大器、会降低模型稳定性与泛化性能这一反直觉现象。

    

    数据增强已成为现代机器学习系统的基本组成部分，并被广泛认为是提高鲁棒性和泛化能力的有效策略。尽管其在实践中取得了成功，但数据增强在标签噪声下的理论影响仍然知之甚少。在这项工作中，我们首先从理论上解释了数据增强何时有益、何时有害，并在独立同分布学习设置和更具挑战性的领域泛化场景下，提供了区分这两种情况的充分条件。然后，我们揭示了一个反直觉的现象：数据增强在学习过程中实质上充当了不确定性放大器，降低了模型的稳定性和泛化性能。这一发现得到了所提出的学习界和泛化界的支持。我们的分析不依赖于特定的数据增强技术，且仅需温和的假设，表明这些结果具有广泛的适用性。

    arXiv:2610.27701v1 Announce Type: new  Abstract: Data augmentation (DA) has become a fundamental component of modern machine learning systems and is widely regarded as an effective strategy for improving robustness and generalization. Despite its empirical success, the theoretical impact of DA under label noise remains poorly understood. In this work, we first theoretically explain when DA is beneficial and when it becomes detrimental, and provide sufficient conditions for distinguishing between these two regimes under both the i.i.d. learning setting and the more challenging domain generalization scenario. Then, we reveal a counterintuitive phenomenon: DA essentially acts as an uncertainty amplifier in the learning process, decreasing model stability and generalization performance. This finding is supported by the proposed learning and generalization bounds. Our analyses do not depend on specific DA techniques and require only mild assumptions, indicating that the results are applicab
    
[^121]: 影响力即目标响应雅可比矩阵与残差之积

    Influence Is the Target-Response Jacobian Times Residuals

    [https://arxiv.org/abs/2610.27689](https://arxiv.org/abs/2610.27689)

    论文证明了样本权重影响力与目标响应雅可比矩阵之间仅差一个逐样本乘子的精确恒等关系，且对于典型连接损失该乘子正是训练残差，从而将训练数据归因与模型对训练目标值的敏感性统一起来。

    

    我们引入了目标响应雅可比矩阵 J = [∂ŷ_i/∂y_j]，它衡量模型预测对单个训练目标值变化的响应程度。J 补充了训练数据归因（TDA）方法——后者刻画的是对损失函数中赋予训练观测的权重（而非其目标值）的敏感性——并且 J 可以以与基于影响力的 TDA 方法相同的计算成本进行无矩阵计算。我们证明样本权重影响力 I 和 J 源自相同的隐式微分论证，两者仅相差一个逐样本乘子：对于逐点损失，在任何具有可逆 Hessian 矩阵的驻点处，I = J diag(m) 精确成立。对于典型连接损失（平方误差、逻辑斯蒂、泊松），该乘子即为训练残差，即 I = J diag(r)，而 softmax 交叉熵则满足一个多类别版本。该恒等式预测了影响力……

    arXiv:2610.27689v1 Announce Type: new  Abstract: We introduce the target-response Jacobian $J = [\partial \hat y_i/\partial y_j]$, which measures how model predictions respond to changes in individual training targets. $J$ complements training-data attribution (TDA) methods, which characterize sensitivity to the weights placed on training observations in the loss rather than to their targets, and can be computed matrix-free at the same cost as influence-based TDA methods. We show that sample-weight influence $\mathcal{I}$ and $J$ follow from the same implicit differentiation argument and differ only by a per-example multiplier: for pointwise losses, $\mathcal{I} = J\operatorname{diag}(m)$ exactly at any stationary point with an invertible Hessian. For canonical-link losses (squared error, logistic, Poisson) the multiplier is the training residual, $\mathcal{I} = J\operatorname{diag}(r)$, and softmax cross-entropy obeys a multiclass version. The identity predicts three ways in which inf
    
[^122]: CyberHull：基于凸包的确定性恶意URL隔离方法

    CyberHull: Deterministic Malicious URL Isolation via Convex Hull

    [https://arxiv.org/abs/2610.27675](https://arxiv.org/abs/2610.27675)

    提出CyberHull框架，利用QuickHull算法在多个正交特征域上构建凸包边界，以资源高效、确定性的方式实现恶意URL的识别与隔离，训练复杂度仅为O(n log n)。

    

    日益复杂且具有抗规避能力的钓鱼攻击策略的出现，对现代网络安全构成了重大威胁，尤其是在信息物理系统和基于互联网的网络中。传统的概率方法和某些机器学习方法往往难以检测新型攻击，并且可能缺乏可解释性或可移植性。为应对这些挑战，我们提出了CyberHull，这是一种资源高效且具有确定性的框架，通过在多个正交特征域（包括结构内容熵、超链接图拓扑和语义标记分布）上构建数学定义的凸包边界来识别恶意URL。CyberHull使用QuickHull算法构建紧凑的凸包（称为“金库”/vaults），在适当假设下其训练复杂度为O(n log n)。URL分类通过多面体内部点测试来执行

    arXiv:2610.27675v1 Announce Type: cross  Abstract: The emergence of increasingly sophisticated and evasion-resistant phishing attack strategies represents a significant threat to modern cybersecurity, particularly in cyber-physical systems and internet-based networks. Traditional probabilistic methods and certain machine learning approaches often struggle to detect novel attacks and may lack interpretability or portability. To address these challenges, we introduce CyberHull, a resource-efficient and deterministic framework for identifying malicious URLs through mathematically defined convex hull boundaries across multiple orthogonal feature domains, including structural content entropy, hyperlink graph topology, and semantic token distribution. CyberHull constructs compact convex hulls, referred to as vaults, using the QuickHull algorithm, with a training complexity of O(n log n) under suitable assumptions. URL classification is performed through point-in-polyhedron tests that impose 
    
[^123]: 用于优化深度学习模型的混合动量方法

    Hybrid Momentum Methods for Optimizing Deep Learning Models

    [https://arxiv.org/abs/2610.27674](https://arxiv.org/abs/2610.27674)

    提出混合动量方法HyM，将线性加权的长期历史梯度与几何折扣的快动量相结合，并据此构建经Newton–Schulz正交化的HyMUON优化器，在保留长期梯度信息与适应近期变化之间实现平衡。

    

    动量是深度学习随机优化的核心，包括大语言模型的训练。经典动量方法通常依赖快速的指数移动平均（EMA）方案，该方案以固定的时间尺度对过去的梯度进行折扣，可能低估那些仍然具有信息价值的较早梯度。这在保留长期历史与适应近期变化之间造成了权衡。我们提出混合动量（HyM），它将过去梯度的线性加权平均与几何折扣的快动量相结合。前者按梯度迭代索引成比例地分配权重，并保留不断增长的历史跨度；后者则强调近期梯度。这种混合结构是模块化的，可以与自适应预处理及其他更新变换相结合。特别地，我们提出HyMUON，它对混合方向应用Newton–Schulz正交化。

    arXiv:2610.27674v1 Announce Type: new  Abstract: Momentum is central to stochastic optimization in deep learning, including the training of large language models. Classical momentum methods typically rely on a fast exponential moving average (EMA) scheme, which discounts past gradients on a fixed timescale and may underweight older gradients that remain informative. This creates a trade-off between retaining long-term history and adapting to recent changes. We introduce Hybrid Momentum (HyM), which combines a linearly weighted average of past gradients with a geometrically discounted fast momentum. The former assigns weights proportional to the gradients' iteration indices and retains a growing span of history, while the latter emphasizes recent gradients. This hybrid construction is modular and can be combined with adaptive preconditioning and other update transformations. In particular, we propose HyMUON, which applies Newton--Schulz orthogonalization to the hybrid direction. Illustr
    
[^124]: 复杂几何下流体动力学的神经算子分裂

    Neural Operator Splitting for Fluid Dynamics in Complex Geometry

    [https://arxiv.org/abs/2610.27671](https://arxiv.org/abs/2610.27671)

    提出神经算子分裂框架，将流体动力学的每个时间步分解为物理可解释的分数步并分别学习，通过条件线性傅里叶神经算子在复杂几何下保持数值格式的物理结构，同时利用对称采样和纠错循环缓解自回归误差累积。

    

    在复杂多样的几何中求解动力系统问题，对数值求解器和基于学习的代理模型而言都仍然具有挑战性。我们提出了神经算子分裂，这是一个在保持分数步数值格式物理结构的同时学习流体动力学的框架。与以整体方式学习完整时间步的演化不同，我们将每个时间步分解为物理上可解释的步骤，并利用底层求解器的中间状态分别学习每个算子。在分数步线性特性的指导下，我们引入了一种条件线性傅里叶神经算子（CL-FNO），其线性状态映射由边界几何等输入的非线性条件进行调制。我们进一步提出了两种互补策略——对称采样和纠错循环，以缓解自回归展开过程中的误差累积。在二维和三维流体流动上的实验……（摘要截断）

    arXiv:2610.27671v1 Announce Type: new  Abstract: Solving dynamical systems in complex and diverse geometries remains challenging for both numerical solvers and learning-based surrogate models. We propose Neural Operator Splitting, a framework for learning fluid dynamics while preserving the physical structure of fractional-step numerical schemes. Instead of learning the full-step evolution monolithically, we decompose each time step into physically interpretable steps, and learn each operator separately using intermediate states from the underlying numerical solver. Guided by the linearity properties of the fractional steps, we introduce a conditional-linear Fourier neural operator (CL-FNO) whose linear state mappings are modulated by nonlinear conditioning on inputs such as boundary geometry. We further introduce two complementary strategies, symmetry sampling and corrective cycling, to mitigate error accumulation during autoregressive rollout. Experiments on 2D and 3D fluid flows wit
    
[^125]: 成对学习提高深度神经网络的数据效率

    Pairwise Learning Increases Data Efficiency in Deep Neural Networks

    [https://arxiv.org/abs/2610.27669](https://arxiv.org/abs/2610.27669)

    该论文提出成对比较学习框架，将回归问题转化为相对排序任务，无需额外监督即可大幅提升深度神经网络在小样本回归和分类中的数据效率，使轻量级ResNet10在家庭硬件上训练即可超越大规模预训练模型。

    

    深度神经网络在小样本回归任务中常常失败，会退化为低方差的预测而忽略个体差异。在此，我们引入了一种成对比较框架，将回归问题重新构建为聚合的相对排序问题。该方法在不同模型架构和数据集规模下都提高了数据效率。在UTKFace数据集（2.3万张图像）上，成对学习在无需额外监督的情况下生成了超过十亿个训练比较对。这种成对排序使得在家庭硬件上从零开始训练的轻量级ResNet10达到了3.64年的平均绝对误差（MAE），超越了以往模型，包括那些使用大规模预训练的模型。在类别分类方面，成对监督在STL10数据集（仅有500-2000个标注样本）上始终优于其他条件完全相同的10类交叉熵模型，准确率最高提升5.2%。总体而言，成对监督提升了数据效率，使得在有限数据下也能实现强大的回归和分类性能。

    arXiv:2610.27669v1 Announce Type: cross  Abstract: Deep neural networks often fail at small-sample regression, collapsing to low-variance predictions that miss individual differences. Here, we introduce a pairwise comparison framework that reframes regression as aggregated relative rankings. This method improves data efficiency across model architectures and dataset sizes. On the UTKFace dataset (23k images), pairwise learning generates over one billion training comparisons without additional supervision. The paired ranking allows a lightweight ResNet10 trained from scratch on home hardware to achieve a 3.64-year MAE-surpassing previous models, including those with large-scale pretraining. For classification into categories, pairwise supervision consistently outperforms an otherwise identical 10-way crossentropy model on STL10 with 500-2000 labeled examples, improving accuracy by up to 5.2%. Overall, pairwise supervision enhances data efficiency and enables strong regression and classi
    
[^126]: 利用成对卷积神经网络从有限神经影像数据中推断连续认知特征

    Inferring continuous cognitive traits from limited neuroimaging data using pairwise convolutional neural networks

    [https://arxiv.org/abs/2610.27668](https://arxiv.org/abs/2610.27668)

    提出了一种成对卷积排序框架，通过比较式学习直接从全脑MRI推断连续认知特征，即使在小样本（N=108）条件下也能准确预测空间能力表现，并优于传统回归和逐点深度学习方法。

    

    认知能力的个体差异广泛存在，然而支持这些差异的大脑结构特征仍知之甚少。传统的脑-行为分析依赖于预定义的指标（如体积或皮层厚度），并且通常假设其与行为呈线性关系，这限制了在小数据集上的敏感性。在这里，我们提出了一种成对卷积排序框架，该框架直接从全脑T1加权MRI中学习，通过进行比较性预测而非绝对预测，即使在小样本（N=108）条件下也能表现良好。我们使用四个小尺度空间能力任务对该方法进行了测试。回归分析在所有任务中均以较高的准确率预测了任务表现，其预测能力超过了标准回归和逐点深度学习基线方法。通过迭代的皮层区域去除（后向消除法），我们揭示了支持所有四个任务中个体表现共同依赖的大脑区域。

    arXiv:2610.27668v1 Announce Type: new  Abstract: Individual differences in cognitive abilities vary widely, yet the structural brain features supporting these differences remain poorly understood. Traditional brain-behavior analyses rely on predefined metrics such as volume or cortical thickness and typically assume linear relationships with behavior, limiting sensitivity in small datasets. Here, we introduce a pairwise convolutional ranking framework that learns directly from whole-brain T1-weighted MRI by making comparative rather than absolute predictions, which performs well even with small samples (N=108). We tested this method with four small-scale spatial abilities tasks. Regression analysis predicted performance with high accuracy across all tasks, achieving greater predictive power over standard regression and pointwise deep learning baselines. Iterative cortical-region removal (backward elimination) revealed brain regions supporting individual performance common to all four t
    
[^127]: CUA-Recovery：面向长时程计算机使用智能体的错误恢复基准测试

    CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents

    [https://arxiv.org/abs/2610.27645](https://arxiv.org/abs/2610.27645)

    该论文提出了 CUA-Recovery 基准和 ReRail 框架，通过在持久化桌面环境中将任务组合成长时程工作流来系统评估计算机使用智能体的错误检测与恢复能力，弥补了现有基准依赖人工注入错误、无法反映真实智能体策略所致失败的不足。

    

    随着计算机使用智能体（CUA）逐渐转向跨越多个应用程序的长时程工作流，未被检测到的错误可能会悄无声息地破坏持久化状态，并在后续动作中引发级联失效。可靠的错误检测与恢复已成为构建稳健 CUA 的关键瓶颈。然而，现有基准主要关注端到端任务性能，对于智能体能否在执行过程中检测错误并从中恢复的洞察十分有限。少数评估恢复能力的基准在很大程度上依赖人工注入的错误，这可能无法反映真实智能体策略所引发错误的类型或时程范围。我们提出了 CUA-Recovery，这是一个由覆盖多个桌面应用程序的测试集和训练集组成的基准。我们使用 ReRail 来构建该基准，ReRail 是一个在持久化桌面环境中合成长时程任务与轨迹的框架。ReRail 将 MyPCBench 任务组合成新的长时程工作流……（原文摘要在此处截断）

    arXiv:2610.27645v1 Announce Type: new  Abstract: As computer-use agents (CUAs) move to long-horizon workflows spanning multiple applications, undetected errors can silently corrupt persistent state and cause cascading failures in subsequent actions. Reliable error detection and recovery have become critical bottlenecks for building robust CUAs. However, existing benchmarks focus primarily on end-to-end task performance, providing limited insight into whether agents can detect and recover from errors during execution. The few benchmarks that evaluate recovery rely largely on manually injected errors, which may not reflect the types or horizons of failures induced by real agent policies. We introduce CUA-Recovery, a benchmark comprising test and training suites across multiple desktop applications. We construct it using ReRail, a framework for synthesizing long-horizon tasks and trajectories in a persistent desktop environment. ReRail composes MyPCBench tasks into new long-horizon workfl
    
[^128]: BCT：面向流语言模型的贝叶斯一致温度调节方法

    BCT: Bayes-Consistent Tempering for Flow Language Models

    [https://arxiv.org/abs/2610.27641](https://arxiv.org/abs/2610.27641)

    提出贝叶斯一致温度调节（BCT），将温度显式地融入流语言模型生成过程的分布与动力学中，解决了事后温度缩放导致的分布与动力学不匹配问题，使温度调节对流语言模型性能的提升更加可靠。

    

    温度缩放通常作为模型概率的事后变换来应用。在由后验诱导的流语言模型中，这种方式会造成两个不匹配。首先，后验温度调节会隐式地改变终点分布和中间噪声状态的分布，而去噪器却仍然是在原始状态上训练的。其次，经过温度调节的后验权重与为未经调节的流所导出的传输动力学相结合。这些不匹配在实证中有所体现：在流语言模型中，降低温度对GSM8K性能的提升远不如在离散扩散语言模型中可靠。我们提出了贝叶斯一致温度调节（BCT），它将温度显式地融入生成流的分布和动力学之中。其可学习的实现方式在经过温度调节的中间状态上训练去噪器，并使用相应的调节后动力学对终点分布进行温度调节。我们还推导了无需训练的变体（摘要在此处被截断）

    arXiv:2610.27641v1 Announce Type: cross  Abstract: Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free var
    
[^129]: 集合感知结构编码：以肽构象集合为条件调制蛋白质编码器

    Ensemble-Aware Structural Encoding (EASE): Conditioning Protein Encoders on Peptide Conformational Ensembles

    [https://arxiv.org/abs/2610.27638](https://arxiv.org/abs/2610.27638)

    提出了集合感知结构编码框架EASE，通过选择代表肽主要结构状态的中心构象体并以置换不变算子融合其表示，将构象集合信息融入蛋白质编码器，从而更好地捕捉与抗菌活性、毒性、稳定性等肽性质相关的构象变化。

    

    蛋白质基础模型为下游肽预测任务提供了可迁移的表示。然而，现有的表示方法通常将肽简化为单一序列或单一结构，这可能遗漏与关键肽性质相关的构象变化，这些性质包括抗菌活性、溶血性、毒性、稳定性以及结合能力。例如，蛋白酶抗性与通过氢交换测量的构象动力学的相关性，比与平均螺旋度的相关性更强。因此，我们提出了集合感知结构编码，这是一个从构象集合中高效提取结构信息以用于肽表示的框架。为了从含噪声的构象体中提取有用信息，EASE 选择代表肽主要结构状态的中心构象体，并使用置换不变算子融合它们各自的表示，从而生成肽级别的表示。

    arXiv:2610.27638v1 Announce Type: new  Abstract: Protein foundation models provide transferable representations for downstream peptide prediction tasks. However, existing representations typically reduce a peptide to a single sequence or structure, which may miss conformational variability relevant to key peptide properties, including antibacterial activity, hemolysis, toxicity, stability, and binding. For example, protease resistance correlates more strongly with conformational dynamics measured by hydrogen exchange than with average helicity. Therefore, we introduce Ensemble-Aware Structural Encoding (EASE), a framework for effectively extracting structural information from conformational ensembles for peptide representation. To extract useful information from noisy conformers, EASE selects central conformers representing a peptide's major structural states and fuses their individual representations with a permutation-invariant operator to produce a peptide-level representation for d
    
[^130]: 从结果到步骤：面向长程语言代理的可靠成功感知评论家学习

    From Outcomes to Steps: Reliable Success-Aware Critic Learning for Long-Horizon Language Agents

    [https://arxiv.org/abs/2610.27623](https://arxiv.org/abs/2610.27623)

    SCALE框架通过让评论家在每个交互前缀同时预测回报值和任务成功概率（成功价值），解决了长程语言代理在稀疏反馈和轨迹级监督下中间动作信用分配的难题。

    

    针对长程语言代理的强化学习，需要在仅有轨迹级监督的条件下判断中间动作是否使最终任务成功的可能性增大或减小。其主要困难在于：一个在总体回报预测上足够准确的评论家，仍可能无法分辨为动作分配信用所需的局部价值差异，尤其是在稀疏反馈和函数逼近的情况下。传统评论家估计未来回报，但策略改进还依赖于知晓每个中间决策是否使任务成功更加可能或更不可能。我们提出了SCALE，这是一个actor-critic框架，它在传统回报估计的基础上，增加了对最终任务成功的显式预测。在每个交互前缀处，评论家预测一个回报值以及一个“成功价值”，后者被定义为从该前缀出发任务最终能够被完成的概率。SCALE将成功价值的变化转换为（原文此处截断）

    arXiv:2610.27623v1 Announce Type: new  Abstract: Reinforcement learning for long-horizon language agents requires identifying whether intermediate actions make eventual task success more or less likely when supervision is available only at trajectory level. The main difficulty is that a critic with aggregate return-prediction accuracy may still fail to resolve the local value differences needed to assign credit to actions, especially under sparse feedback and function approximation. Conventional critics estimate future return, but policy improvement also depends on knowing whether each intermediate decision makes task success more or less likely. We introduce SCALE, an actor-critic framework that augments conventional return estimation with explicit prediction of eventual task success. At each interaction prefix, the critic predicts a return value and a Success Value, defined as the probability that the task will eventually be completed from that prefix. SCALE converts changes in Succe
    
[^131]: 一种具备抗辐射能力、故障感知且可自修复的量化神经网络

    A Radiation-Hardened, Fault-Aware Quantized Neural Network That Self-Heals

    [https://arxiv.org/abs/2610.27621](https://arxiv.org/abs/2610.27621)

    本文提出一种抗辐射的故障感知量化神经网络，通过扩展故障感知训练范式，使模型能在商用现货硬件上于运行时自修复位翻转导致的权重损坏，且无需存储额外副本。

    

    深度学习（DL）模型正日益被部署到辐射环境恶劣的场合中，例如航天器、粒子加速器，以及不久后的天基数据中心。模型作为存储在持久性存储器中的数值所构成的学习编排实例，在高质量能粒子破坏数据时容易发生持续的性能退化。在推理阶段，流经网络的新数据从根本上打破了训练期间所确定的静态模型的离线假设与保障。为了保持模型的质量和效能，针对位翻转的容错能力是一个明确的目标。在这项工作中，我们开发了一种运行时方法，用于在商用现货（COTS）硬件上发生位翻转后修复神经网络（NN）的量化权重，而无需存储额外的副本。我们的工作扩展了成熟的故障感知训练（FAT）范式，通过添加对齐惩罚项来集中未受损的……

    arXiv:2610.27621v1 Announce Type: cross  Abstract: Deep learning (DL) models are increasingly being deployed into radiation-challenging environments in spacecraft, particle accelerators, and, soon, space-based data centers. Instantiated as a learned orchestration of numbers stored in persistent memory, models are susceptible to continual degradation as high-energy particles corrupt data. At inference time, new data passed through the network fundamentally break the off-line assumptions and guarantees of a static model determined during training. To preserve the quality and efficacy of the models, fault tolerance against bit flips is a clear objective. In this work we developed a runtime method for repairing the quantized weights of neural networks (NNs) after bit flips on commodity, commercial off-the-shelf (COTS) hardware, without storing extra replicas. Our work extends the well-established Fault-Aware Training (FAT) paradigm by adding an alignment penalty to concentrate undamaged ac
    
[^132]: 学习化学体系中异质子结构间的图间交互

    Learning Inter-Graph Interactions Between Heterogeneous Substructures of Chemical Systems

    [https://arxiv.org/abs/2610.27614](https://arxiv.org/abs/2610.27614)

    该论文提出了一种描述复杂化学体系的数据描述符和图神经网络方法，用于学习有机与无机化合物之间的异质图间交互，并在混合太阳能电池材料的物理性质预测上取得了最先进的准确率。

    

    含有异质子结构的复杂化学体系在现实应用中十分常见。这类复杂化学体系的多种物理现象源于异质子结构之间的相互作用。然而，现有的面向图间交互的图表示学习方法假设的是同质结构（例如有机分子和无机晶体材料）之间的图级交互。我们提出了一种复杂化学体系的数据描述符，以及一种用于学习有机化合物与无机化合物之间图间交互的图神经网络。我们将所提出的方法应用于预测含有异质子结构的混合太阳能电池材料的物理性质，这类材料作为可持续能源资源受到了广泛关注。通过学习异质图间交互，所提出的方法在预测（原文摘要在此处截断）方面取得了最先进的准确率。

    arXiv:2610.27614v1 Announce Type: cross  Abstract: Complex chemical systems containing heterogeneous substructures are common in real-world applications. Various physical phenomena of the complex chemical systems are derived from the interactions between the heterogeneous substructures. However, existing graph representation learning methods for inter-graph interactions assumed graph-level interactions between homogeneous structures, such as organic molecules and inorganic crystalline materials. We propose a data descriptor of the complex chemical systems and a graph neural network for learning inter-graph interactions between organic and inorganic compounds. We applied the proposed method to predict the physical properties of hybrid solar cell materials containing heterogeneous substructures, which have received significant attention for sustainable energy resources. By learning heterogeneous inter-graph interactions, the proposed method achieved state-of-the-art accuracy in predictin
    
[^133]: 审计产品而非递归过程：递归自我改进声明背后的评估证据

    Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims

    [https://arxiv.org/abs/2610.27610](https://arxiv.org/abs/2610.27610)

    该论文首次通过对114项递归自我改进（RSI）研究的系统性全文编码，量化审计了RSI声明背后产品评估与流程评估证据的实际报告频率，揭示了递归自我改进声明与其实际评估证据之间的差距。

    

    递归自我改进（RSI）承诺了诸如自我进化的LLM智能体等AI系统能够在产生进一步改进方面不断提升。评估这一承诺需要将关于最终系统的“产品”声明与关于改进过程本身的“流程”声明区分开来。现有综述已明确了这些声明所需的证据，但研究实际报告这些证据的频率仍未被测量。我们通过对2026年10月冻结的可复现arXiv搜索框架中114项符合条件的研究进行全文编码来考察这一问题，其中包括90项将自身系统描述为RSI的研究。该编码手册将声明与评估设计相关联。通过独立的人工编码、第三方读者的裁定，以及对改进器比较的单独盲编码来评估编码的可靠性。在90项RSI研究中，90%在包含部分分离或未明确说明分离的宽泛标准下报告了超越开发任务的产品评估，而60%报告了评估……（原文摘要在此处截断）

    arXiv:2610.27610v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) promises AI systems, such as self-evolving LLM agents, that become better at producing further improvements. Evaluating this promise requires separating product claims about the final system from process claims about the improvement process itself. Existing surveys specify the evidence these claims require, but how often studies report that evidence remains unmeasured. We examine this question through full-text coding of 114 eligible studies in a reproducible arXiv search frame frozen in October 2026, including 90 that describe their own systems as RSI. The codebook links claims to evaluation designs. Independent human coding, third-reader adjudication, and a separate blind coding of improver comparisons assess coding reliability. Among the 90 RSI studies, 90% report product evaluation beyond development tasks under a broad criterion that includes partial or unstated separation, while 60% report evaluat
    
[^134]: 问而学：基于查询的协作强化学习

    Ask to Learn: Query-Based Collaborative Reinforcement Learning

    [https://arxiv.org/abs/2610.27606](https://arxiv.org/abs/2610.27606)

    该论文提出了基于查询的协作强化学习新框架，使仅有离线数据的学习者无需直接与环境交互，即可通过自适应地向持有互补信息的协作者查询状态-时间对来突破离线数据覆盖范围的限制，并针对有限视界表格型问题给出了所需查询次数的实例依赖信息论下界。

    

    离线强化学习从根本上受限于可用数据的覆盖范围，而通过在线探索收集额外轨迹可能代价高昂、不安全或不可行。我们研究能否在不与底层环境直接交互的情况下克服这一局限。受协作者中介模拟等实际场景的启发——其中协作者可以提供与离线数据互补的信息——我们提出了基于查询的协作强化学习，以研究如何高效地获取这些信息。学习者从一个固定的离线数据集出发，自适应地查询状态-时间对；协作者根据一个固定的、可能是次优的策略采样动作，模拟器返回相应的奖励和下一状态。对于有限视界的表格型问题，我们推导了关于额外查询次数的实例依赖的信息论下界……

    arXiv:2610.27606v1 Announce Type: cross  Abstract: Offline reinforcement learning is fundamentally limited by the coverage of the available data, while collecting additional trajectories through online exploration may be costly, unsafe, or infeasible. We study whether this limitation can be overcome without directly interacting with the underlying environment. Motivated by practical settings such as collaborator-mediated simulation, where a collaborator can provide information complementary to the offline data, we introduce query-based collaborative reinforcement learning to study how to acquire this information efficiently. The learner starts with a fixed offline dataset and adaptively queries state-time pairs; the collaborator samples an action according to a fixed, possibly suboptimal policy, and the simulator returns the corresponding reward and next state. For finite-horizon tabular problems, we derive an instance-dependent information-theoretic lower bound on the additional queri
    
[^135]: 面向自适应侧偏角与侧偏刚度估计的灵敏度正交残差学习

    Sensitivity-Orthogonal Residual Learning for Adaptive Sideslip and Cornering Stiffness Estimation

    [https://arxiv.org/abs/2610.27586](https://arxiv.org/abs/2610.27586)

    该论文提出灵敏度正交残差学习方法，通过零空间参数化构建与刚度灵敏度正交的模型修正器，解决了学习修正器吸收滤波新息从而抑制侧偏刚度在线自适应的问题，将侧偏角估计误差降低12.7%至22.0%。

    

    侧偏刚度的在线自适应使基于模型的侧偏角估计器在轮胎特性变化时仍能保持有效，但它与学习到的模型修正器争夺同一个滤波新息。一个具有足够参数的修正器可以吸收刚度变化所产生的新息模式，导致在拟合改善的同时自适应被抑制。我们将预测测量的刚度灵敏度通过车辆状态进行传播，并构建了一个与这些灵敏度正交的修正器。零空间参数化在不使用惩罚系数的情况下强制执行该约束，并在每个辨识片段上保留静态最小二乘刚度估计。在整车仿真中，无约束修正器的刚度响应随着其参数数量的增加而下降，而正交修正器在所有参数规模下都至少能像未修正的滤波器那样充分响应，并将侧偏角误差降低12.7%至22.0%。在一次模拟驾驶中……

    arXiv:2610.27586v1 Announce Type: cross  Abstract: Online adaptation of the cornering stiffness keeps a model-based sideslip estimator valid as tires change, but competes with learned model correction for the same filter innovation. A corrector with enough parameters can absorb the innovation pattern that a stiffness change produces, and adaptation is suppressed while the fit improves. We propagate the stiffness sensitivities of the predicted measurements through the vehicle states and construct a corrector orthogonal to them. A null-space parameterization enforces the constraint without a penalty coefficient and preserves the static least-squares stiffness estimate on each identification segment. In full-vehicle simulation, the stiffness response of the unconstrained corrector declines as its parameter count grows, whereas the orthogonal corrector responds at least as fully as the uncorrected filter at all sizes and lowers sideslip error by 12.7% to 22.0%. In a simulated drive with a 
    
[^136]: 弥合数字欺骗与物理保真：一种针对认知雷达的时空攻击

    Bridging Digital Deception and Physical Fidelity: A Spatio-Temporal Attack on Cognitive Radars

    [https://arxiv.org/abs/2610.27585](https://arxiv.org/abs/2610.27585)

    提出了一种物理先验驱动的分段时空攻击方法（SSTA），通过全局-局部特征解耦分割机制提升对抗样本的跨架构可迁移性，并将物理信道失真建模为可微时空算子以增强攻击在真实电子战环境中的战术可靠性。

    

    随着认知雷达向一维复基带序列的端到端处理演进，深度网络面临严重的对抗性威胁。然而，现有攻击在实际电子战中遭遇双重局限：跨架构可迁移性低，且对物理信道失真高度脆弱，这削弱了其战术可靠性。为克服这些瓶颈，我们提出了一种物理先验驱动的分段时空攻击（SSTA）。首先，全局-局部特征解耦分割机制中断了对抗梯度对长程序列相干性的依赖，弥合了跨架构的特征鸿沟。其次，超越缺乏物理语义的静态范数约束，我们将信道失真抽象为可微的时空算子，并通过期望变换策略将其融入动量迭代过程。全面的跨……

    arXiv:2610.27585v1 Announce Type: new  Abstract: As cognitive radars evolve towards the end-to-end processing of one-dimensional complex baseband sequences, deep networks face severe adversarial threats. However, existing attacks encounter dual limitations in practical electronic warfare: low cross-architecture transferability and severe vulnerability to physical channel distortions, which degrades their tactical reliability. To overcome these bottlenecks, we propose a physical prior-driven Segmented Spatio-Temporal Attack (SSTA). First, a global-local feature decoupling segmentation mechanism interrupts the adversarial gradients' dependence on long-range sequence coherence, bridging the cross-architecture feature gap. Second, moving beyond static norm constraints devoid of physical semantics, we abstract channel distortions into differentiable spatio-temporal operators, integrating them into momentum iterations via an Expectation over Transformation (EoT) strategy. Comprehensive cross
    
[^137]: 扩散语言模型中的事实访问状态：从机制到知识编辑与检索

    Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval

    [https://arxiv.org/abs/2610.27580](https://arxiv.org/abs/2610.27580)

    该论文揭示了扩散语言模型在第一层注意力层就通过将关系措辞路由到可见实体位置来组装事实访问状态，这一机制与自回归模型的组装方式截然不同且发生更早，为知识编辑与检索提供了新的机制层面依据。

    

    事实回忆本质上是一种查找：网络内部的某种表示充当键，而该键的构成决定了哪些查询能够检索到该事实、一次编辑必须覆盖哪些内容、以及检索器能够读取到什么。我们将其称为“事实访问状态”。在自回归（AR）模型中，主语位置对属性的丰富过程在很大程度上独立于关系，而针对关系的选择发生在较晚的读出位置。在掩码扩散语言模型（DLMs）中，我们发现该状态的组装方式不同，且时间上早得多：在第一层注意力层中，关系的措辞被路由到可见实体的位置上，随后早期的前馈层基于这一融合表示查找事实。因此，该状态会随关系的表面形式和查询方向而变化，这意味着关于同一事实的查询占据的是状态空间中的一个区域而非一个点，而对事实的操作能否成功则取决于（摘要在此处截断，未完）

    arXiv:2610.27580v1 Announce Type: cross  Abstract: Factual recall is a lookup: some representation inside the network serves as the key, and what that key is made of decides which queries can retrieve the fact, what an edit must cover, and what a retriever can read. We call it the factual access state. In autoregressive (AR) models the subject position enriches attributes largely independently of the relation, and relation-specific selection happens later, at the readout position. In masked diffusion language models (DLMs) we find the state is assembled differently, and much earlier: in the first attention layer the wording of the relation is routed into the position of whichever entity is visible and the early feed-forward layers look the fact up on this fused representation. The state therefore varies with the relation's surface form and with the query direction, so the queries about one fact occupy a region of state space rather than a point, and an operation on the fact succeeds on
    
[^138]: 从POMDP理论到基于粒子滤波的深度强化学习

    From POMDP Theory to Deep RL with Particle Filters

    [https://arxiv.org/abs/2610.27575](https://arxiv.org/abs/2610.27575)

    本文提出了一种结合序贯蒙特卡洛粒子滤波、在线潜在动力学模型学习和近似在线规划的实用端到端深度强化学习算法，并首次为部分可观测环境下的深度强化学习提供了将模型估计、粒子滤波和蒙特卡洛等各项近似误差显式分解的端到端遗憾界理论保证。

    

    在部分可观测环境中的基于模型的强化学习需要推断隐藏状态、学习潜在动力学，并基于由此产生的信念进行规划。实用的算法通常只解决上述流程中的某一个环节，而对其他环节依赖的近似往往缺乏深入理解；同时，现有的POMDP理论通常分析的是计算效率低下的理论算法，与现代深度强化学习实践相距甚远。为填补这一空白，我们提出了一种实用的端到端算法，该算法通过序贯蒙特卡洛（SMC）方法近似解决隐藏状态滤波问题，在线更新潜在动力学模型，并在用状态价值评论家（一种QMDP近似）对叶节点进行自举之前，执行 $d$ 步近似在线规划。我们分析了每种近似方法，并证明了端到端的遗憾界，将总误差分解为模型估计误差、粒子滤波误差、蒙特卡洛规划误差等部分。

    arXiv:2610.27575v1 Announce Type: new  Abstract: Model-based reinforcement learning in partially observable environments requires inferring hidden states, learning latent dynamics, and planning under the resulting belief. Practical algorithms often solve one aspect of the above pipeline while relying on approximations for the others that are often poorly understood, while existing POMDP theory typically analyzes computationally inefficient theoretical algorithms far from modern deep RL practice. To fill this gap, we propose a practical end-to-end algorithm that approximately solves the hidden state filtering problem via sequential Monte Carlo (SMC), updates a latent dynamics model online, and performs approximate online planning for $d$ steps before bootstrapping leaves with a state-value critic (a QMDP approximation). We analyze each approximation and prove an end-to-end regret bound decomposing the error into model estimation error, particle filter error, Monte Carlo rollout error, a
    
[^139]: 当大语言模型进行操纵时：测量行为、社会与时间效应

    When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects

    [https://arxiv.org/abs/2610.27573](https://arxiv.org/abs/2610.27573)

    该论文提出了一个测量“行为脆弱性”的行为计算框架，通过操纵梯度下降生成针对权威、紧迫性和社会认同偏差的操纵性AI刺激，并借助多利益相关者影响评估网络与时间性危害传播模型，量化了操纵性LLM输出对用户信任校准与依从性的显著影响及其跨用户、跨时间的传播效应。

    

    大语言模型（LLM）可以通过微妙的语言线索影响用户的信任与决策，然而操纵性输出的行为效应至今仍缺乏充分的量化研究。我们引入了一个行为计算框架，用于测量“行为脆弱性”，即用户对操纵性AI响应的易感程度。基于DarkPatterns-LLM基准，我们利用操纵梯度下降法（MGD）生成受控的操纵性刺激，针对与权威、紧迫性和社会认同相关的认知偏差。我们采用多利益相关者影响评估网络（MSIAN）分析参与者响应以建模跨用户影响，并使用时间性危害传播模型（THPM）刻画行为效应随时间的持续性。在一项涉及120名参与者的研究中，操纵性表述对信任校准和依从性产生了显著影响（p < 0.01），且效应还会在（摘要在此处被截断）……

    arXiv:2610.27573v1 Announce Type: new  Abstract: Large language models (LLMs) can influence user trust and decision-making through subtle linguistic cues, yet the behavioral effects of manipulative outputs remain poorly quantified. We introduce a behavioral computing framework for measuring \textit{behavioral vulnerability}, defined as users' susceptibility to manipulative AI responses. Building on DarkPatterns-LLM benchmark, we generate controlled manipulative stimuli using Manipulation Gradient Descent (MGD), targeting biases associated with authority, urgency, and social proof. We analyze participant responses using the Multi-Stakeholder Impact Assessment Network (MSIAN) to model cross-user influence and the Temporal Harm Propagation Model (THPM) to characterize the persistence of behavioral effects over time. In a study involving 120 participants, manipulative framing produced significant changes in trust calibration and compliance ($p < 0.01$), with effects also propagating across
    
[^140]: 基于数据生成分布上狄利克雷过程先验的贝叶斯神经网络在深度强化学习随机化价值函数中的应用

    Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning

    [https://arxiv.org/abs/2610.27570](https://arxiv.org/abs/2610.27570)

    该论文提出在数据生成分布上（而非网络权重上）施加狄利克雷过程先验来构建新型贝叶斯神经网络DP-BNN，并据此开发了DP-DQN算法，通过随机化价值函数实现高效的汤普森采样，从而解决深度强化学习中认知不确定性量化的难题。

    

    贝叶斯神经网络（BNN）为量化认知不确定性提供了一个有原则的框架。然而，标准的权重空间先验会产生复杂的高维神经网络后验，从中采样在计算上代价极高，且流行的近似推断方案往往无法有效捕捉认知不确定性——这种失效模式在强化学习（RL）中尤为突出。为解决这一挑战，我们引入了一类新的BNN，它通过在底层数据生成分布上（而非网络参数上）直接施加贝叶斯非参数狄利克雷过程（DP）先验来量化不确定性。在此基础上，我们开发了DP-DQN，这是一种强化学习算法，它利用DP-BNN来维护价值函数上的后验分布，并执行高效的汤普森采样。通过将随机化价值函数建立在贝叶斯非参……

    arXiv:2610.27570v1 Announce Type: cross  Abstract: Bayesian neural networks (BNNs) offer a principled framework for quantifying epistemic uncertainty. However, standard weight-space priors yield complex, high-dimensional neural network posteriors that are computationally prohibitive to sample from, and popular approximate inference schemes often fail to capture epistemic uncertainty effectively---a failure mode particularly acute in reinforcement learning (RL) \citep{osband2018randomized}. To resolve this challenge, we introduce a new class of BNNs that quantify uncertainty by placing Bayesian nonparametric Dirichlet Process (DP) priors directly on the underlying data-generating distribution rather than on network parameters. Building on this foundation, we develop DP-DQN, a reinforcement learning algorithm that leverages DP-BNNs to maintain posterior distributions over value functions and perform efficient Thompson sampling. By grounding randomized value functions in Bayesian nonparam
    
[^141]: 二分类足矣：二元分类先验即可胜任多类别上下文表格预测

    Two Is Enough: Binary Classification Priors Suffice for Multi-Class In-Context Tabular Prediction

    [https://arxiv.org/abs/2610.27569](https://arxiv.org/abs/2610.27569)

    该论文发现表格基础模型在预训练时仅需使用二分类先验任务（类别数上限设为2），即可实现优于传统十分类上限配置的零样本多分类预测性能，并在公开基准和真实信用评分场景中得到验证。

    

    表格基础模型（TFM）通过在大规模合成先验数据上预训练Transformer，使得单次前向传播即可实现零样本表格预测。TabICLv2在预训练期间对其分类先验的类别数量进行采样，而主流配置将该类别数上限设为10。我们报告了一个反直觉的结果：将该上限直接设为2——即让每个预训练分类任务都成为二分类任务——所得到的模型权重依然能够执行零样本K路分类。在38个TabArena分类数据集上（200轮bootstrap重采样，以默认RF=1000为基准），这种二元两阶段方案达到了1590.1的Elo评分，高于官方默认配置的1568.0（+22.1），也高于TabPFN-v2.6（1577.1）和RealTabPFN-v2.5（1521.6）。在60家真实信用评分机构上开展的生产规模验证独立证实了这一结论：同一个模型检查点在60家机构中的44家上提升了测试AUC（AUC的配对Wilcoxon检验 p = 0.0007，p = 0.0002

    arXiv:2610.27569v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) pretrain a transformer on large-scale synthetic priors so that a single forward pass yields zero-shot tabular prediction. TabICLv2 samples the class count of its classification prior during pretraining, and the prevailing configuration caps it at ten. We report a counter-intuitive result: setting this cap directly to two---making every pretrained classification task binary---still yields weights that perform zero-shot K-way classification 2. On the 38 TabArena classification datasets (200 bootstrap rounds, anchored at default RF = 1000), the binary two-stage recipe reaches 1590.1 Elo, above the official default at 1568.0 (+22.1), above TabPFN-v2.6 (1577.1) and RealTabPFN-v2.5 (1521.6). A production-scale validation on 60 real-world credit-scoring institutions independently confirms the direction: the same checkpoint improves test AUC on 44 of 60 institutions (paired Wilcoxon p = 0.0007 for AUC, p = 0.0002
    
[^142]: TRCM：面向鲁棒神经车辆路径规划的拓扑感知路径一致性建模

    TRCM: Topology-Aware Route-Consistency Modeling for Robust Neural Vehicle Routing

    [https://arxiv.org/abs/2610.27568](https://arxiv.org/abs/2610.27568)

    提出 TRCM 方法，复用自回归构造流水线中的节点表示显式预测成对同路径概率并作为一致性信号注入节点选择，结合结构保持增强与拓扑熵重加权，从而在相同解码预算下提升神经求解器解决车辆路径问题的性能与鲁棒性。

    

    神经组合优化已经为带容量约束的车辆路径问题实现了快速的构造式求解器，然而大多数自回归模型仍然通过顺序节点选择来构造解，并且仅隐式地表示路径组织方式。我们研究了能否显式地建模路径级别的客户共分配，并将其用于指导节点级的决策过程。为此，我们提出了拓扑感知路径一致性建模（TRCM），该方法复用标准自回归构造流水线中的节点表示来预测成对同路径概率，并将其作为路径一致性信号注入客户选择过程。由于同路径监督既存在类别不平衡又存在结构不均匀的问题，TRCM 将结构保持增强与基于拓扑熵（TE）的重新加权相结合，以强调局部模糊的客户对。在相同解码预算下，TRCM 降低了……（原文摘要在此处截断）

    arXiv:2610.27568v1 Announce Type: cross  Abstract: Neural combinatorial optimization has enabled fast constructive solvers for the Capacitated Vehicle Routing Problem (CVRP), yet most autoregressive models still construct solutions through sequential node selection and represent route organization only implicitly. We investigate whether route-level customer co-assignment can be modeled explicitly and used to guide this node-level decision process. To this end, we propose TopologyAware RouteConsistency Modeling (TRCM), which reuses node representations from a standard autoregressive construction pipeline to predict pairwise same-route probabilities and injects them into customer selection as a route-consistency signal. Because sameroute supervision is both class-imbalanced and structurally uneven, TRCM combines structure-preserving augmentation with Topological Entropy (TE)- based reweighting to emphasize locally ambiguous customer pairs. Under matched decoding budgets, TRCM reduces the
    
[^143]: 学习式读取令牌用于提取大语言模型置信度

    Learned Readout Tokens for Extracting Large Language Model Confidence

    [https://arxiv.org/abs/2610.27566](https://arxiv.org/abs/2610.27566)

    提出 ScratchProbe 方法，通过学习一个专用的读取令牌来提取隐藏状态训练置信度探针，在多种数据集和大语言模型上优于基于现有令牌的探针，多令牌扩展还可进一步提升性能。

    

    大语言模型正日益广泛地应用于日常场景中，其响应会对下游任务产生重大影响。为了负责任地部署此类系统，我们需要了解能在多大程度上信任模型的各个输出。估计大语言模型置信度的一种流行方法是在模型令牌的隐藏状态上训练探针，以预测模型输出的正确性。这些探针通常从模型词表中已有的令牌读取隐藏状态。在本工作中，我们提出了一种新的置信度探针方法，该方法学习一个新的读取令牌，并利用其隐藏状态来训练探针。我们证明，我们的学习式读取方法 ScratchProbe 在多种数据集和大语言模型上均优于使用现有令牌的方法。我们还将该方法扩展为多令牌方法，并证明其能带来进一步的性能提升。最后，由于学习式读取仅改变……

    arXiv:2610.27566v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used within everyday applications where their responses have significant downstream impacts. For responsible deployment of such systems, we desire insight into how much we can trust their individual outputs. A popular approach for estimating LLM confidence is to train probes on the hidden states of the LLM's tokens to predict the correctness of the model's output. These probes typically read hidden states from tokens that are already in the model's vocabulary. In this work, we introduce a new approach to confidence probes that learn a new readout token and utilize its hidden states to train the probe. We show that our learned readout approach ScratchProbe outperforms its existing-token counterpart on a variety of datasets and LLMs. We also extend our method to a multi-token approach which we show provides further performance improvements. Finally, since the learned readout changes only the 
    
[^144]: HypoNet Nankai PS：利用基于物理信息神经网络的三维P波和S波速度模型实现南海海槽震源快速定位

    HypoNet Nankai PS: rapid hypocenter determination for the Nankai Trough using physics-informed neural networks with three-dimensional P- and S-wave velocity models

    [https://arxiv.org/abs/2610.27561](https://arxiv.org/abs/2610.27561)

    本研究将基于物理信息神经网络的震源定位框架HypoNet Nankai扩展为HypoNet Nankai PS，通过从P波模型构建S波速度模型并改进训练方法以适应更陡峭的S波走时梯度，实现了南海海槽俯冲带同时利用P波和S波到时的快速震源定位。

    

    基于三维（3D）地震速度模型的快速准确震源定位对于俯冲带的地震监测至关重要，但其计算需求很高。HypoNet Nankai是一个震源定位软件包，采用物理信息神经网络（PINN）走时推断技术，针对南海海槽俯冲带的真实三维P波速度模型进行计算，旨在解决这一问题，但此前仅支持P波到时。本研究提出了HypoNet Nankai PS，将该框架扩展至P波和S波走时。我们首先利用海洋沉积物和地壳的经验转换关系，从现有P波模型构建了S波速度模型。由于所得S波速度结构的速度较低，导致走时梯度明显更陡峭，原先为P波速度结构设计的PINN训练流程已不足以（应对这一挑战）。

    arXiv:2610.27561v1 Announce Type: cross  Abstract: Rapid and accurate hypocenter determination based on three-dimensional (3D) seismic velocity models is essential for seismic monitoring in subduction zones, yet it remains computationally demanding. HypoNet Nankai, a hypocenter determination package that uses physics-informed neural network (PINN) travel-time inference for a realistic 3D P-wave velocity model of the Nankai Trough subduction zone, was designed to address this problem, but supported only P-wave arrival times. Here, we present HypoNet Nankai PS, which extends the framework to both P- and S-wave travel times. We first constructed an S-wave velocity model from the existing P-wave model using empirical conversion relations for marine sediments and the crust. Because the resulting S-wave velocity structure has lower velocities and therefore considerably steeper travel-time gradients, the original PINN training procedure designed for the P-wave velocity structure was insuffici
    
[^145]: AnchorRM：面向时序图神经网络的基于锚点残差记忆的并行批内状态重建

    AnchorRM: Parallel Intra-Batch State Reconstruction with Anchor-Based Residual Memory for Temporal Graph Neural Networks

    [https://arxiv.org/abs/2610.27558](https://arxiv.org/abs/2610.27558)

    提出AnchorRM机制，将批前最后提交的节点记忆作为固定锚点，通过学习残差来并行重建同批次内各事件的节点状态，从而在不进行顺序循环更新的情况下解决时序图神经网络小批量处理中的状态过时问题。

    

    基于记忆的时序图神经网络（TGN）依赖循环节点记忆来捕捉不断演化的交互历史。然而，时序小批量处理在计算效率与时序保真度之间引入了权衡：当同一批次中某个节点出现多次时，后续交互将使用过时的批前记忆进行评估，因为中间的记忆更新尚未被提交。随着时序批次规模的增大，这一问题变得更加严重。为了克服这一挑战，我们提出了AnchorRM，这是一种基于锚点的残差记忆机制，能够在不执行顺序循环更新的情况下重建针对特定事件的批内节点状态。AnchorRM将每个批次开始时最后提交的节点记忆视为批前固定锚点，并利用因果上先于该事件的同批次交互来推断相对于该锚点所缺失的状态演化。学习到的残差随后用于校准……

    arXiv:2610.27558v1 Announce Type: cross  Abstract: Memory-based temporal graph neural networks (TGN) rely on recurrent node memories to capture evolving interaction histories. However, temporal mini-batching introduces a trade-off between computational efficiency and temporal fidelity: when a node appears multiple times within the same batch, later interactions are evaluated using stale pre-batch memory because the intermediate memory updates have not yet been committed. This problem becomes more severe as temporal batch size increases. To overcome this challenge, we propose AnchorRM, an anchor-based residual memory mechanism that reconstructs event-specific intra-batch node states without executing sequential recurrent updates. AnchorRM treats the last committed node memory at the beginning of each batch as a pre-batch fixed anchor and uses causally preceding same-batch interactions to infer the missing state evolution relative to that anchor. The learned residual then calibrates the 
    
[^146]: 原生安全性能否迁移？通过在线策略蒸馏实现跨模型的安全预训练复用

    Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation

    [https://arxiv.org/abs/2610.27555](https://arxiv.org/abs/2610.27555)

    本文首次提出“原生安全性复用”问题，并证明通过在线策略蒸馏，可以将安全预训练教师模型的鲁棒安全性迁移给标准预训练的学生模型，使其在保留通用能力的同时继承安全鲁棒性。

    

    安全预训练已成为一种有前景的范式，它能在预训练阶段将安全性深度融入模型之中，从而获得对后续微调的鲁棒性，而这种鲁棒性是传统安全后训练往往缺乏的。我们将这种在模型预训练期间习得的安全性称为“原生安全性”。然而，原生安全性获取成本高昂，仅局限于其被预训练的那个模型，并且常常伴随着较低的通用能力。在这项工作中，我们重新思考了安全预训练模型的实际价值，并首次提出了原生安全性复用问题：原生安全性能否从一个专用的、可能规模更小的安全预训练“教师”模型迁移到标准预训练的“学生”模型，使后者在继承其鲁棒性的同时保留自身的通用能力？我们证明了原生安全性可以通过在线策略蒸馏进行“迁移”：标准预训练的学生模型针对安全预训练提示生成回复，而……

    arXiv:2610.27555v1 Announce Type: cross  Abstract: Safety pretraining has emerged as a promising paradigm for embedding safety deeply into a model during pretraining, yielding robustness to subsequent fine-tuning that conventional safety post-training often lacks. We refer to this safety acquired during model pretraining as native safety. Yet native safety is expensive to acquire, confined to the model in which it was pretrained, and often accompanied by lower general utility. In this work, we rethink the practical value of safety-pretrained models and introduce, for the first time, the native safety reuse problem: Can native safety be transferred from a specialized, potentially smaller, safety-pretrained "teacher" to standard-pretrained "students," enabling them to inherit its robustness while preserving their general utility? We show that native safety can "travel" through on-policy distillation. A standard-pretrained student generates responses to safety-pretraining prompts, while a
    
[^147]: 光滑非凸零阶优化的下界：低曲率迹条件下确定性的代价

    Lower Bounds for Smooth Nonconvex Zeroth-Order Optimization: The Price of Determinism under Low Curvature Trace

    [https://arxiv.org/abs/2610.27550](https://arxiv.org/abs/2610.27550)

    本文在低曲率迹（有效秩）条件下证明了随机化零阶非凸优化仅需 O(ET) 次函数调用且在常数因子内最优，而任何确定性方法都无法获得这种依赖有效秩的节省，从而量化了确定性方法的固有代价。

    

    低曲率迹（low curvature trace）是一种有效秩条件，常被用来解释随机化零阶方法为何能扩展到语言模型微调等高维任务。确定性方法能否获得同样的节省？我们证明它们不能。我们研究仅凭精确函数值在 ℝ^d 上寻找 L-光滑非凸函数 ε-驻点的问题，并假设在初始值以下 Hessian 被一个迹至多为 LE 的固定矩阵所支配；这正是用于解释零阶微调的有效秩条件。当 ET≤d 时，此类微调所采用的高斯两点法只要拒绝使 f 上升的步，就只需 O(ET) 次函数调用，其中 T=LΔ/ε² 是从初始间隙 Δ 出发的梯度下降迭代次数；我们还证明任何随机化方法都需要 Ω(ET) 次调用，因此该复杂度在常数因子内是最优的。相比之下，每个确定性方法……（原文摘要至此截断）

    arXiv:2610.27550v1 Announce Type: new  Abstract: Low curvature trace, an effective rank condition, is a common explanation for why randomized zeroth-order methods scale to high-dimensional tasks such as language model fine-tuning. Can deterministic methods achieve the same savings? We show that they cannot. We study finding $\varepsilon$-stationary points of $L$-smooth nonconvex functions on $\mathbb{R}^d$ from exact function values, assuming that below the starting value the Hessian is dominated by a fixed matrix with trace at most $LE$; this is the effective rank condition used to explain zeroth-order fine-tuning. When $ET\le d$, the Gaussian two-point method that such fine-tuning uses needs only $O(ET)$ calls if it rejects steps that raise $f$, where $T=L\Delta/\varepsilon^2$ is the iteration count of gradient descent from initial gap $\Delta$, and we prove that every randomized method needs $\Omega(ET)$ calls, so this cost is optimal up to constants. In contrast, every deterministi
    
[^148]: 面向基于事件的可穿戴设备任务转换检测的时间感知可解释自注意力机制

    Timing-Aware and Interpretable Self-Attention for Event-Based Wearable Task Transition Detection

    [https://arxiv.org/abs/2610.27549](https://arxiv.org/abs/2610.27549)

    该论文提出了一个时间感知且可解释的自注意力机制，通过在基于事件的可穿戴行为标记中显式建模时间与持续时间信息，改进了任务转换检测的性能与可解释性。

    

    任务转换检测旨在识别一个人何时从一项任务转换到另一项任务，为持续健康监测和及时的场景感知干预提供基础。尽管统计特征和深度特征表示已展现出强大的分类性能，但它们往往依赖于聚合特征或潜在特征，难以从行为角度进行解释。基于事件的特征通过将行为和生理模式编码为具有明确语义含义的离散标记，提供了一种紧凑且可解释的替代方案，使其非常适合健康信息学中的行为分析。自注意力为建模这些标记之间的交互提供了机制；然而，传统的自注意力分类器主要依赖标记的顺序，并未显式地纳入时间或持续时间信息。因此，事件表示中蕴含的有价值的时间线索可能无法被充分利用。

    arXiv:2610.27549v1 Announce Type: new  Abstract: Task transition detection aims to identify when a person shifts from one task to another, providing a foundation for continuous health monitoring and timely context-aware interventions. While statistical and deep feature representations have demonstrated strong classification performance, they often rely on aggregated or latent features that are difficult to interpret in behavioural terms. Event-based features provide a compact and interpretable alternative by encoding behavioural and physiological patterns as discrete tokens with explicit semantic meaning, making them well suited for behavioural analysis in health informatics. Self-attention offers a mechanism for modelling interactions among such tokens; however, conventional self-attention classifiers primarily rely on token order and do not explicitly incorporate timing or duration information. As a result, informative temporal cues in event representations may not be fully exploited
    
[^149]: BEACON：面向云系统异常检测的多模态语义与异步传播桥接框架

    BEACON: Bridging Multimodal Semantics and Asynchronous Propagation for Cloud System Anomaly Detection

    [https://arxiv.org/abs/2610.27542](https://arxiv.org/abs/2610.27542)

    提出了BEACON框架，通过基于信息瓶颈机制的动态可解释多模态聚合器弥合多模态数据间的语义鸿沟，并建模系统实例间的异步时空依赖关系，从而提升云系统异常检测的准确性。

    

    现代大规模云系统持续产生海量的多模态可观测性数据，例如时间序列指标、日志记录和系统依赖图，这些数据反映了系统的健康状态。为确保云系统的可靠运行，异常检测模型通过分析这些多模态观测数据来识别异常状态。然而，现有方法依赖于简单的融合策略和图学习，忽略了模态之间的语义鸿沟以及系统实例随时间变化的异步时空依赖关系。为此，我们提出了BEACON，一个面向云系统异常检测、用于桥接多模态语义与异步传播的框架。具体而言，我们引入了动态可解释多模态聚合器，该聚合器利用信息瓶颈机制压缩来自指标的语义和时间动态信息，并将其与日志在共享空间中对齐。

    arXiv:2610.27542v1 Announce Type: new  Abstract: Modern large-scale cloud systems continuously generate massive and multimodal observability data, such as time-series metrics, log records, and system dependency graphs, which reflect the health state of the system. To ensure reliable cloud system operation, anomaly detection models identify abnormal states by analyzing these multimodal observations. However, existing approaches rely on simple fusion strategies and graph learning, ignoring the semantic gap among modalities and the asynchronous spatio-temporal dependencies across system instances over time. To this end, we propose BEACON, a framework to Bridge multimodal sEmantics and Asynchronous propagation for Cloud system anOmaly detectioN. Specifically, we introduce a Dynamic Interpretable Multimodal Aggregator (DIMA) that leverages an information-bottleneck mechanism to compress information from the semantics and temporal dynamics of metrics, and aligns them with logs in a shared pr
    
[^150]: 什么改变了警报分数？基于精确计算与稀疏事件估计的因果归因

    What Moved the Alarm Score? Causal Attribution with Exact Computation and Scarce-Incident Estimation

    [https://arxiv.org/abs/2610.27534](https://arxiv.org/abs/2610.27534)

    该论文将警报分数的变化建模为混合结构因果模型状态上的Shapley博弈，把结构函数与外生规律作为不同的博弈参与者，从而在事件数据稀少的条件下实现精确的因果归因，并以函数/规律分辨的细粒度方式解释各类型变化如何传播并影响警报分数。

    

    工业系统和科学仪器中的警报往往需要在事件数据稀少的情况下进行即时诊断。为了指导干预，解释必须追溯数据生成过程中的变化如何通过因果结构传播并改变警报分数，而不仅仅是标记那些可能只是下游效应的偏移。然而，对相互作用的因果组件进行精确归因似乎面临着组合爆炸的难题，而现有方法要么假设因果模型保持不变，要么需要足够的事件数据才能可靠地估计变化后的系统。我们通过将警报分数的变化表述为混合结构因果模型（SCM）状态上的Shapley博弈来应对这一场景。将结构函数和外生规律视为不同的参与者，从而产生更细粒度的、可区分函数与规律的报告，说明每类变化如何传播至警报分数。同一框架还定义了已实现事件报告和总体报告。

    arXiv:2610.27534v1 Announce Type: cross  Abstract: Alarms in industrial systems and scientific instruments often demand immediate diagnosis despite scarce incident data. To guide intervention, an explanation must trace how changes in the data-generating process propagate through the causal structure to alter the alarm score, rather than merely flagging shifts that may be downstream effects. Yet exact attribution across interacting causal components appears combinatorial, while existing methods often either assume an unchanged causal model or require enough incident data to estimate the changed system reliably. We address this setting by formulating the alarm-score change as a Shapley game over hybrid structural causal model (SCM) states. Treating structural functions and exogenous laws as distinct players yields a finer-grained, function/law-resolved report of how each type of change propagates to the alarm score. The same framework defines realized-event and population reports, with a
    
[^151]: 一维孤子轮廓：增强的神经网络求解器与约束Deep Ritz方法

    Soliton Profiles in 1D: Enhanced Neural-Network Solvers and Constrained Deep Ritz Methods

    [https://arxiv.org/abs/2610.27500](https://arxiv.org/abs/2610.27500)

    该论文开发并比较了多种增强神经网络求解器（包括高阶优化精炼、Fourier特征PINN以及两种新颖的约束Deep Ritz方法），用于求解非线性薛定谔、广义KdV和Klein-Gordon方程的一维孤子基态轮廓，并以显式sech型精确解作为基准评估各方法的精度。

    

    我们开发并比较了用于求解一维基态轮廓的增强神经网络求解器，这些轮廓是方程 Q″=bQ−γQᵖ（Q>0, p>1, b,γ>0）的解，即非线性薛定谔方程、广义Korteweg-de Vries方程和非线性Klein-Gordon方程的驻定轮廓方程。其显式的sech型解可作为误差计算和基准超参数选择的精确参考。我们通过引入基于Gauss-Newton/Levenberg-Marquardt和L-BFGS精炼的高阶优化方法、Fourier特征PINN以及约束Deep Ritz（DR）表述，改进了标准的Adam训练物理信息神经网络（PINN）基线。在DR方法中，我们提出了两种新的神经网络表述，分别实现两种经典的变分刻画：一种是Nehari约束的Deep Ritz方法，其中解的高度（振幅）通过投影到Nehari流形上学习得到……

    arXiv:2610.27500v1 Announce Type: cross  Abstract: We develop and compare enhanced neural-network solvers for one-dimensional ground-state profiles, which are solutions of $$ Q^{\prime\prime}=bQ-\gamma Q^{p},\quad Q>0,\quad p>1, \quad b, \gamma >0, $$ the stationary profile equation for the nonlinear Schr\"odinger, generalized Korteweg-de Vries, and nonlinear Klein-Gordon equations. Its explicit sech-type solutions serve as exact reference for error computation and benchmark hyperparameter selection. We improve on the standard Adam-trained physics-informed neural network (PINN) baseline by introducing higher-order optimization via Gauss-Newton/Levenberg-Marquardt and L-BFGS refinements, as well as Fourier-feature PINNs, and constrained Deep Ritz (DR) formulations. In DR approaches we introduce two new neural-network formulations, which implement two classical variational characterizations: a Nehari-constrained Deep Ritz method, in which the height is learned by projecting onto the Neha
    
[^152]: 通过特权回溯蒸馏实现自我进化的智能体

    Self-Evolving Agents through Privileged Retrospective Distillation

    [https://arxiv.org/abs/2610.27495](https://arxiv.org/abs/2610.27495)

    提出回溯自蒸馏（RSD）多轮框架，让同一模型既当学生又当教师，借助特权回溯上下文将缺乏真值答案或奖励的自我经验转化为密集监督信号，实现模型、数据与监督的共同进化。

    

    自我进化的智能体旨在从自身经验中持续改进，而非在部署后保持固定不变，但自然收集的轨迹往往缺乏真实标准答案或由验证器导出的奖励信号。我们提出了回溯自蒸馏（Retrospective Self-Distillation, RSD），这是一个多轮框架，利用特权回溯上下文将此类经验转化为密集的监督信号。对于每条同策略轨迹，同一个底层模型同时充当学生和教师：教师在以跨任务先例与经验教训、以及来自已完成轨迹的轮次级反思和未来后见之明为条件的情况下，对学生实际执行的动作进行重新评分。由此得到的教师分布通过同策略自蒸馏被蒸馏回策略之中。在多轮迭代中，经验库不断增长，回溯上下文不断刷新，使模型、其数据及其监督信号得以共同进化。我们进一步引入了前沿引导……（摘要至此截断）

    arXiv:2610.27495v1 Announce Type: cross  Abstract: Self-evolving agents aim to improve from their own experience rather than remain fixed after deployment, but naturally collected trajectories often lack ground-truth solutions or verifier-derived rewards. We introduce Retrospective Self-Distillation (RSD), a multi-round framework that turns such experience into dense supervision using privileged retrospective context. For each on-policy trajectory, the same underlying model acts as both student and teacher: the teacher re-scores the student's realized actions while conditioning on cross-task precedents and lessons, together with turn-level reflection and future hindsight from the completed trajectory. The resulting teacher distribution is distilled back into the policy through on-policy self-distillation. Across rounds, the experience bank grows and the retrospective context is refreshed, allowing the model, its data, and its supervision to co-evolve. We further introduce frontier-guid
    
[^153]: MiCLA：面向多声源音频表示学习的混合-成分潜在对齐

    MiCLA: Mixture-to-Constituent Latent Alignment for Polyphonic Audio Representation Learning

    [https://arxiv.org/abs/2610.27491](https://arxiv.org/abs/2610.27491)

    MiCLA 提出了一种两阶段自监督预训练框架，通过合成对数梅尔混合和双向潜在对齐来显式建模混合音频与其组成成分的关系，在 AudioSet 下游任务上取得了领先的 mAP 性能。

    

    现实世界的音频通常包含同时发生的声音事件，然而标准的自监督方法将每个音频片段视为孤立的训练单元，没有显式建模混合音频与组成成分之间的关系。我们提出了混合-成分潜在对齐，这是一个两阶段自监督预训练框架，利用合成的对数梅尔频谱混合将混合音频的表示与其组成片段进行对齐。第一阶段通过掩码自蒸馏和对比正则化来构建单片段表示的结构。第二阶段引入双向对齐机制：对于合成混合音频，MiCLA 抑制与其中一个成分对齐的分量，并将残差与另一个成分对齐，具体做法是在 patch token 上采用投影残差化，在全局 class token 上采用样本自适应软残差化。在下游评估中，MiCLA 在微调设置下于 AS-20K 数据集上达到 43.2% 的平均精度均值，在 AS-2M 数据集上达到 50.3% 的 mAP，并且（摘要在此处截断）

    arXiv:2610.27491v1 Announce Type: cross  Abstract: Real-world audio typically contains concurrent sound events, yet standard self-supervised methods treat each clip as an isolated training unit without explicitly modeling mixture--constituent relationships. We propose Mixture-to-Constituent Latent Alignment (MiCLA), a two-stage self-supervised pre-training framework that uses synthetic log-mel mixing to align mixture representations with their constituent clips. Stage 1 structures single-clip representations via masked self-distillation and contrastive regularization. Stage 2 introduces bidirectional alignment: for a synthetic mixture, MiCLA suppresses the component aligned with one constituent and aligns the residual with the other, using projective residualization on patch tokens and sample-adaptive soft residualization on the global class token. In downstream evaluations, MiCLA achieves 43.2% mean average precision (mAP) on AS-20K and 50.3% mAP on AS-2M under fine-tuning, and achiev
    
[^154]: 大语言模型中双用途概念的正交边界遗忘

    Orthogonal Boundary Unlearning for Dual-Use Concepts in Large Language Models

    [https://arxiv.org/abs/2610.27490](https://arxiv.org/abs/2610.27490)

    提出正交边界遗忘（OBU）框架，通过将参数更新严格限制在与能力子空间正交的零空间内，在剔除大语言模型危险双用途能力（如网络战、病原体增强）的同时，保证对良性科学能力造成零附带损害。

    

    减轻大语言模型的灾难性风险需要在剔除危险作战能力（如网络战攻击利用和病原体增强）的同时，不损害其良性的科学实用性。在这些双用途领域中，合法的防御科学与恶意攻击共享相同的词汇、语法和内部模型表示，导致标准遗忘方法会产生严重的附带损害（概念渗透）。在这项工作中，我们提出了正交边界遗忘，这是一个具有数学理论基础的框架，旨在消除对良性双用途能力的附带损害。OBU通过对经验激活协方差进行奇异值分解，从配对的校准查询中提取领域匹配的能力子空间，并将参数更新严格限制在正交零空间内。这保证了良性表示在所张成的校准子空间方向上经历零参数扰动……

    arXiv:2610.27490v1 Announce Type: cross  Abstract: Mitigating catastrophic risks from large language models requires excising hazardous operational capabilities, such as cyberwarfare exploitation and pathogen enhancement, without degrading benign scientific utility. In these dual-use domains, legitimate defensive science and malicious attacks share vocabulary, syntax, and internal model representations, causing standard unlearning methods to induce severe collateral damage (concept bleeding).   In this work, we present Orthogonal Boundary Unlearning (OBU), a mathematically grounded framework designed to eliminate collateral damage on benign dual-use capabilities. OBU extracts a domain-matched capability subspace from paired calibration queries via singular value decomposition on empirical activation covariances, and restricts parameter updates strictly to the orthogonal null space. This guarantees that benign representations experience zero parameter perturbation along the spanned cali
    
[^155]: 参数梯度流何时在表示空间中闭合

    When Parameter Gradient Flows Close in Representation Space

    [https://arxiv.org/abs/2610.27474](https://arxiv.org/abs/2610.27474)

    该论文精确刻画了表示梯度流对所有损失函数发生闭合的条件：在行满秩的双因子矩阵表示中，闭合仅发生在有效标量平衡的最小联合秩或满联合秩的二值带符号谱族处，而所有中间秩均被排除。

    

    参数化模型的表示梯度流依赖于一种能够保留隐藏参数信息的“可动性”。我们研究了在何种条件下，从固定初始化出发，整张表示的可动性映射能够对每一个自洽损失函数都成立。对于宽度大于两个输出维度、且各因子分别行满秩的双因子矩阵表示，我们分类了所有可允许的联合秩：闭合恰好发生在有效标量平衡的最小联合秩处，或发生在满联合秩下的二值带符号谱族处。所有中间秩均被排除；正的表示流属于已知的平方根族。对于具有常值环境表示与常值梯度-李秩的一般表示，在解析情形以及特定的光滑情形下，通常意义下的闭合等价于一个局部不变量层次的度量。一个端点截面迫使最初不受限制的可动性映射在表示（原文摘要在此处截断）……

    arXiv:2610.27474v1 Announce Type: new  Abstract: The represented gradient flow of a parameterized model depends on a mobility that can retain hidden parameter information. We study when one entire represented mobility map works for every autonomous loss from one fixed initialization. For two-factor matrix representations wider than both output dimensions, with individually full-row-rank factors, we classify all admissible joint ranks: closure occurs precisely at effective scalar-balanced minimal joint rank or a two-value signed spectral family at full joint rank. All intermediate ranks are excluded; positive represented flows belong to known square-root families. For general representations with constant ambient representation and gradient-Lie ranks, ordinary closure is equivalent to a local invariant-level metric in the analytic case and specified smooth regimes. An endpoint section forces the initially unrestricted mobility map to admit a locally smooth representative on the represen
    
[^156]: 超越提示词：恢复文本到图像模型中的隐藏概念

    Beyond the Prompt: Recovering Hidden Concepts in Text-to-Image Models

    [https://arxiv.org/abs/2610.27473](https://arxiv.org/abs/2610.27473)

    本文提出EmbGA，一种在词元嵌入空间中运行的遗传算法，能够恢复文本到图像模型生成的图像中存在但提示词未提及的隐藏概念，为审计被植入的触发关联提供了有效工具。

    

    文本到图像模型能够生成其提示词中从未提及的视觉概念，近期研究表明，这种关联可以被人为植入，使得普通提示词也能触发这些概念。审计这种行为需要恢复提示词所省略的描述内容，但现有的提示词优化器并不适合这项任务：基于梯度的方法需要对评分编码器的反向访问；离散进化搜索将语义不相关的词元视为同样接近；语言模型引导的搜索则需要为每个候选方案进行一次推理调用，这限制了探索范围。本文提出了EmbGA，一种嵌入空间遗传算法，用于恢复一组图像中存在但生成这些图像的提示词中缺失的概念。EmbGA保留了遗传搜索的种群结构，但将其算子应用于词元嵌入空间中：交叉操作在父代嵌入之间进行插值，变异操作则将一个嵌入向随机的（原文摘要在此处截断）

    arXiv:2610.27473v1 Announce Type: cross  Abstract: Text-to-image models can generate visual concepts that their prompts never mention, and recent work shows that such associations can be implanted so that ordinary prompts trigger them. Auditing this behavior requires recovering the description that a prompt omits. Existing prompt optimizers are not well suited to this task. Gradient-based methods require backward access to the scoring encoder. Discrete evolutionary search treats semantically unrelated tokens as equally close. Language-model-guided search requires one inference call per candidate, which limits exploration. This paper presents EmbGA, an embedding-space genetic algorithm that recovers concepts present in a set of images but absent from the prompt that generated them. EmbGA keeps the population structure of genetic search but applies its operators in the token embedding space. Crossover interpolates between parent embeddings, and mutation moves an embedding toward a random
    
[^157]: 基于提示演化的免训练多模态预测持续学习

    Training-Free Continual Learning for Multimodal Forecasting via Prompt Evolution

    [https://arxiv.org/abs/2610.27468](https://arxiv.org/abs/2610.27468)

    该论文提出通过结果驱动反思演化的文本引导模块，使冻结的LLM预测器无需重新训练即可实现时间序列持续学习，在严格防止数据泄露的多模态基准上超越时间序列基础模型，且更新成本从GPU天级降至美元级。

    

    使预测系统适应不断变化的世界，传统上意味着重新训练。我们提出这样一个问题：时间序列预测中持续学习不同状态的任务，能否用一段文本来表示——即一个引导模块，通过结果驱动的反思进行演化，并被嵌入到冻结的LLM预测器的提示中。在一个更新的多模态基准上（每周搜索指数、每日汇率、每日大宗商品，每个时间窗口均配有从网络检索到的新闻），我们在严格的污染控制下运行GEPA式提示演化，以防止时间序列基准中常见的数据泄露问题：预测器的知识截止日期早于每一个被评估的窗口。由Llama-3.3-70B承载的900词演化引导模块在留出数据上击败了时间序列基础模型（TSFMs）（GM MASE 0.776 对比 TabPFN-TS-3 的 0.898），并能迁移到更新的模型而不产生可测量的性能损失，且更新成本仅需几美元而非数个GPU天。我们对该循环进行了刻画：环外增益

    arXiv:2610.27468v1 Announce Type: cross  Abstract: Adapting a forecasting system to a changing world classically means retraining. We ask whether the task of continually learning regimes for time series forecasting can be represented by a piece of text: a guidance block, evolved by outcome-driven reflection and used in a frozen LLM forecaster's prompt. On a refreshed multimodal benchmark (weekly search indices, daily currencies, daily commodities, each window paired with web-retrieved news), we run GEPA-style prompt evolution under strict contamination control to prevent the data leakage common in timeseries benchmarks: the forecaster's knowledge cutoff predates every evaluated window. A 900-word evolved guidance carried by Llama-3.3-70B beats TimeSeries Foundation Models (TSFMs) on held-out data (GM MASE 0.776 vs. 0.898 for TabPFN-TS-3), transfers to more recent models without measurable loss, and costs dollars rather than GPU-days to update. We characterize the loop: out-of-loop gain
    
[^158]: 投影蒸馏：从接近且正确的目标中学习解决困难问题

    Projected Distillation: On Learning to Solve Hard Problems from Close, Correct Targets

    [https://arxiv.org/abs/2610.27467](https://arxiv.org/abs/2610.27467)

    提出“投影蒸馏”方法，将学生模型失败的尝试最小化地编辑成接近且正确的目标，并利用对比更新使模型优先学习修正版本，从而在困难问题上获得有效学习信号，同时保留已有能力。

    

    同策略学习是后训练的核心：在模型自身生成的样本上进行训练可以限制分布偏移，并有助于保留已有能力。然而在困难问题上，当所有尝试都失败时，同策略强化学习无法提供基于奖励的学习信号。同策略蒸馏可以提供来自教师的下一个词元（next-token）监督，但在困难问题上，如果一次 rollout 在早期就出错，后续监督将基于已经包含错误的前缀进行。正确的行为也可以来自标准答案或教师给出的解答，但这种离策略目标可能与模型自身的生成结果差异很大，并且正如我们所展示的，可能会侵蚀模型已有的能力。因此，我们将“同策略性”视为一个连续谱，并提出投影蒸馏方法，该方法将学生模型失败的尝试进行最小化的编辑，使其成为正确的目标。该方法使用对比性更新，使模型更倾向于采纳修正而非失败的尝试，并随着学生……（原文摘要在此处截断）

    arXiv:2610.27467v1 Announce Type: cross  Abstract: On-policy learning is central to post-training: training on a model's own generations limits distribution shift and helps preserve existing capabilities. On hard problems, however, on-policy RL provides no reward-based learning signal when all attempts fail. On-policy distillation provides next-token supervision from a teacher, but on hard problems, if a rollout goes wrong early, subsequent supervision is conditioned on prefixes that already contain the mistake. Correct behavior can instead come from a gold or teacher solution, but such off-policy targets can differ substantially from the model's generations and, as we show, may erode existing capabilities. We therefore treat on-policyness as a continuum and propose projected distillation, which minimally edits the student's failed attempts into correct targets. The method uses contrastive updates to favor corrections over failed attempts and refreshes these corrections as the student 
    
[^159]: 超越标量排行榜：多目标LLM竞技场中帕累托前沿识别的自适应采样算法

    Beyond Scalar Leaderboards: Adaptive Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas

    [https://arxiv.org/abs/2610.27447](https://arxiv.org/abs/2610.27447)

    本文提出了固定置信度的自适应采样算法VB-EGE，能在多目标LLM竞技场评估中以尽可能少的比较高效识别帕累托前沿上的模型，突破了单一标量排行榜的局限。

    

    arXiv:2610.27447v1 公告类型：cross 摘要：大语言模型（LLM）的评估本质上是多目标的，因为不同模型可能在数学、代码、写作等性能维度上呈现不同的权衡关系。由于此类评估通常依赖成本高昂且耗时的人工反馈，因此如何用尽可能少的比较得出可靠结论至关重要。本文提出了向量-Borda经验差距消除算法（VB-EGE），这是一种固定置信度的自适应采样算法，用于在竞技场式评估中高效识别帕累托前沿上的模型，其中每次比较会针对选定的性能维度在某个提示上对一对模型进行评估。一旦积累了足够的统计证据以证实某模型的帕累托状态，VB-EGE就会将其从活跃集合中逐步消除。在多目标Bradley-Terry模型下，我们证明了VB-EGE能够以高概率返回正确的帕累托前沿

    arXiv:2610.27447v1 Announce Type: cross  Abstract: The evaluation of large language models (LLMs) is inherently multi-objective, as different models may offer different tradeoffs across performance dimensions such as math, code, and writing. Because such evaluations are often based on costly and time-consuming human feedback, it is important to draw reliable conclusions using as few comparisons as possible. In this paper, we present Vector-Borda Empirical-Gap Elimination (VB-EGE), a fixed-confidence adaptive sampling algorithm for efficiently identifying models on the Pareto frontier in arena-style evaluation, where each comparison evaluates a pair of models on a prompt with respect to a selected performance dimension. VB-EGE progressively eliminates models from an active set once sufficient statistical evidence has been accumulated to certify their Pareto status. Under a multi-objective Bradley-Terry model, we show that VB-EGE returns the correct Pareto frontier with high probability 
    
[^160]: 面向安全生成式控制的学习型表征接口的决策充分性

    Decision Sufficiency of Learned Representation Interfaces for Safe Generative Control

    [https://arxiv.org/abs/2610.27446](https://arxiv.org/abs/2610.27446)

    本文提出“决策充分性”概念，用以评估学习型表征接口是否为安全生成式控制器提供了足够的信息，精确分离了超额损失的三种来源，并刻画了揭示信息冲突所需的状态数量及接口避免冲突所需的最少输出。

    

    控制器可以通过选择何时提供帮助、何时请求澄清或何时拒绝，来引导语言模型产生安全且有用的响应。当控制器使用选定的内部特征时，仅凭糟糕的性能表现无法区分是信息缺失还是其决策规则或训练的局限性所致。我们研究决策充分性：即在相同策略和评估标准下，使用某种表征的最佳控制器能否达到利用完整状态信息所能获得的最小平均损失。对于输出数量有限的接口，我们精确计算了最优损失，并分离出超额损失的三种来源。成对检验可能会漏掉信息损失：具有相同控制器输入的状态可能在每一对之间共享最优策略，但在整个组内却并非如此。我们给出了揭示冲突所需状态数量的界限，并刻画了无限制接口为避免所有冲突所需的最少输出数量。我们审计了（摘要在此处被截断）

    arXiv:2610.27446v1 Announce Type: new  Abstract: A controller can guide a language model toward safe and useful responses by choosing when to assist, ask for clarification, or refuse. When the controller uses selected internal features, poor performance alone cannot distinguish missing information from limitations of its decision rules or training. We study decision sufficiency: whether the best controller using a representation can match the minimum average loss attainable with full state information, under the same strategies and evaluation criteria. For interfaces with finitely many outputs, we compute the optimal loss exactly and separate the three sources of excess loss. Pairwise checks can miss information loss: states with the same controller input may share an optimal strategy in every pair but not across the whole group. We bound the number of states needed to reveal a conflict and characterize the fewest outputs an unrestricted interface needs to avoid all conflicts. We audit
    
[^161]: LoRASpace：一种面向静态与动态多LoRA组合的池级共享基底

    LoRASpace: A Pool-Wide Shared Substrate for Static and Dynamic Multi-LoRA Composition

    [https://arxiv.org/abs/2610.27444](https://arxiv.org/abs/2610.27444)

    该论文发现真实LoRA池中的低秩因子仅需其秩总和的46%-57%即可保留99%的能量，并据此提出池级共享基底LoRASpace，使多LoRA组合同时获得权重合并的速度与逐LoRA前向传播的质量。

    

    低秩适配在冻结的主干模型上附加小型低秩矩阵，而公共仓库中每个主干模型托管着众多LoRA，各自对应不同的概念或任务。在一次生成中组合多个LoRA迫使人们做出权衡：将它们的权重合并为一次更新只需单次前向传播，但为不同概念训练的LoRA会相互干扰，导致质量下降；而让每个LoRA在各自的前向传播中运行（组合式）虽能保持质量，但推理时间会随激活LoRA数量的增加而线性增长。我们研究了以往多LoRA工作所忽视的LoRA池的一个特性：在真实的LoRA池中，堆叠的低秩因子所需的方向数远少于其秩的总和。利用奇异值分解，我们找到了保留99%因子能量的最小子空间。在三个扩散模型池上，仅需LoRA秩总和的46%至57%，且该子空间随池规模呈次线性增长。LoRA-Hub上包含48个LoRA的子池……

    arXiv:2610.27444v1 Announce Type: cross  Abstract: Low-rank adaptation (LoRA) attaches small low-rank matrices to a frozen backbone, and public repositories host many LoRAs per backbone, each for a different concept or task. Combining several LoRAs in one generation forces a trade-off. Merging their weights into one update needs a single forward pass, but LoRAs trained for different concepts interfere and quality drops. Running each LoRA in its own forward pass (composite) preserves quality, but inference time grows linearly with the number of active LoRAs. We study a property of LoRA pools overlooked by prior multi-LoRA work. In real LoRA pools, the stacked low-rank factors need far fewer directions than their summed ranks. Using singular value decomposition, we find the minimal subspace that keeps 99% of the factor energy. On three diffusion pools, only 46% to 57% of the summed LoRA ranks is needed, and the subspace grows sub-linearly with pool size. A 48-LoRA sub-pool of LoRA-Hub on
    
[^162]: 利用集合学习改进长尾数据上的集合值分类器

    Using Set Learning to Improve Set-Valued Classifiers on Long-tailed Data

    [https://arxiv.org/abs/2610.27443](https://arxiv.org/abs/2610.27443)

    该论文系统研究了不同训练策略与损失函数（包括作用于批次样本集合的集合式损失）对集合值分类器在长尾不平衡数据上性能的影响，提出通过在训练中引入集合学习来提升其宏平均表现。

    

    集合值分类（SVC）提供了一种在保持准确性的同时传达不确定性的实用方法，它通过为每个输入预测可变大小的集合，而非固定大小的集合（例如 top-1 或 top-5）。然而，集合值分类器在长尾和不平衡数据集上的性能显著下降，而表面上这类场景本应是它们所擅长的。这一问题在宏平均评估下尤为明显，因为宏平均对所有类别赋予相同权重。集合值分类器通常在事后校准阶段形成，因此目前对训练动态如何影响其行为知之甚少。在这项工作中，我们研究了不同的神经网络训练策略和损失函数如何影响高度不平衡数据上的下游集合值分类性能。除了标签平滑和类别平衡等方法外，我们还研究了作用于批次内样本集合（而非逐个处理样本）的损失函数……

    arXiv:2610.27443v1 Announce Type: new  Abstract: Set-valued classification (SVC) offers a practical approach to communicating uncertainty while maintaining accuracy by predicting variable-sized sets per input rather than fixed-sized sets (e.g. top-1 or top-5). However, set-valued classifiers (SVCs) exhibit significantly degraded performance on long-tailed and imbalanced datasets, a setting for which they are ostensibly well-suited. This becomes apparent under macro-average evaluation, which weighs all classes equally. SVCs are typically formed during a post-hoc calibration stage, so little is known about how training dynamics affect their behaviour. In this work, we investigate how different neural network training strategies and loss functions impact downstream SVC performance on highly imbalanced data. In addition to methods such as label smoothing and class-balancing, we investigate whether loss functions that operate over sets of examples in a batch, rather than treating all exampl
    
[^163]: 通用编码计算的学习理论基础：对抗性设定

    Learning-Theoretic Foundation for General Coded Computing: The Adversarial Setting

    [https://arxiv.org/abs/2610.27440](https://arxiv.org/abs/2610.27440)

    本文为基于学习理论的通用编码计算框架提出了对抗鲁棒的扩展，无需显式识别受损工作节点即可设计编码器与解码器，并刻画了恢复误差随工作节点数量和对抗预算变化的规律。

    

    对抗性工作节点会返回受损的输出，从而严重扭曲恢复出的计算结果，这给分布式计算系统带来了严峻挑战。编码计算通过在工作节点任务分配中引入冗余来应对这一威胁。现有的精确与近似编码计算方法通常通过显式定位并纠正或剔除错误的计算结果来处理对抗性输出，这会给主节点带来显著的计算开销。此外，现有分析未能刻画恢复误差如何随工作节点数量和对抗预算的变化而扩展。本文提出了通用编码计算（GCC）的一种对抗鲁棒扩展。GCC 是一个基于学习理论的近似编码计算框架，通过最小化端到端恢复误差的上界来设计编码器和解码器。所提出的方法并不通过识别受损工作节点来应对对抗，而是利用平滑……（摘要在此处被截断）

    arXiv:2610.27440v1 Announce Type: new  Abstract: Adversarial workers pose a severe challenge in distributed computing systems by returning corrupted outputs that can substantially distort the recovered computations. Coded computing addresses this threat by introducing redundancy across worker assignments. Existing exact and approximate coded-computing methods typically handle adversarial outputs by explicitly locating and correcting or removing erroneous evaluations, which adds significant computational overhead at the master node. Moreover, existing analyses do not characterize how the recovery error scales with the number of workers and the adversarial budget.   In this paper, we develop an adversarially robust extension of general coded computing (GCC), a learning-theoretic approximate coded-computing framework that designs the encoder and decoder by minimizing an upper bound on the end-to-end recovery error. Rather than identifying corrupted workers, the proposed method uses smooth
    
[^164]: 使用可学习的图位置编码扩展语言模型中的图推理能力

    Scaling Graph Reasoning in Language Models with Learnable Graph Positional Encodings

    [https://arxiv.org/abs/2610.27439](https://arxiv.org/abs/2610.27439)

    该论文提出用GNN与图Transformer计算的可学习图位置编码直接调制语言模型的注意力，并以后融合模块注入结构信息，从而无需序列化图描述、将上下文负担降至O(N)，显著扩展了语言模型可处理的图推理问题规模。

    

    解决基于图的推理任务通常需要向语言推理模型提供图的序列化描述。这种方法导致输入长度随边数增长，从而限制了可解决问题的规模。此外，即使图能够适应上下文窗口，随着图规模的增大，性能也会急剧下降。我们提出通过为语言推理模型添加可扩展且置换等变的图位置编码来解决这一问题，这些位置编码可以调制注意力逻辑值。一个额外的后融合式模块将结构信息融入隐藏状态。我们使用图神经网络（GNN）和图Transformer来计算这些位置编码，从而消除了对序列化邻接关系的需求，将上下文负担降低到O(N)。我们使用图论求解器生成的最优解，联合微调位置编码器和语言推理模型的LoRA适配器，从而在计算上……

    arXiv:2610.27439v1 Announce Type: cross  Abstract: Solving reasoning tasks supported on graphs typically requires providing a Language Reasoning Model (LRM) with a serialized description of the graph. This approach results in the input length growing with the number of edges, which limits the size of the problems that can be solved. Furthermore, even when graphs fit in context, performance sharply decays as the graphs grow. We propose to address this by augmenting LRMs with scalable and permutation equivariant graph positional encodings that modulate attention logits. An additional post-fusion-style module incorporates structural information into hidden states. We compute these positional encodings using a GNN followed by a Graph Transformer. This eliminates the need for serialized adjacencies, reducing the context burden to $O(N)$. We jointly fine-tune the positional encoder and the LRM LoRA adapters using optimal solutions generated by graph-theoretic solvers, resulting in a computat
    
[^165]: MLIPS：多层索引器池共享

    MLIPS: Multi-Layer Indexer Pool Sharing

    [https://arxiv.org/abs/2610.27438](https://arxiv.org/abs/2610.27438)

    MLIPS是一种无需训练的方法，通过让一个生产者层发布超出预算的top-M候选池供后续多个层共享复用，在保留各层独立打分与选择的同时消除重复的候选搜索，在DeepSeek-V4-Flash上实现了1.14–1.40倍加速和14.8–27.4%的GPU能耗降低，且准确率几乎不受影响。

    

    稀疏注意力限制了每个查询所使用的键的数量，但寻找这些键仍可能需要在每一层对完整前缀进行扫描。我们提出了MLIPS（多层索引器池共享），这是一种无需训练的方法，它在共享候选搜索的同时，保留每个使用方自己的打分与选择。生产者会发布一个大于注意力预算K的top-M候选池；后续各层从该池中选出自己的top-K。在三个稀疏注意力模型家族上的测量表明，宽泛候选池所提供的覆盖率显著高于最终选择之间的重叠程度。我们在vLLM中为DeepSeek-V4-Flash实现了逐层共享、持久解码复用以及粗粒度生产者。在四个长上下文测试套件上，每个生产者对应三个使用层的设置可带来1.14–1.40倍的加速和14.8–27.4%的GPU能耗降低，单次运行中观察到的准确率变化为−1.49至+0.29个百分点。更广泛的共享能够…

    arXiv:2610.27438v1 Announce Type: cross  Abstract: Sparse attention limits the keys used by each query, but finding those keys can still require a full-prefix scan at every layer. We introduce \textbf{MLIPS (Multi-Layer Indexer Pool Sharing)}, a training-free method that shares candidate search while retaining each consumer's own scoring and selection. A producer publishes a top-$M$ pool larger than the attention budget $K$; subsequent layers select their own top-$K$ from that pool. Measurements across three sparse-attention model families show substantially higher coverage from broad pools than overlap between final selections. We implement layerwise sharing, persistent decode reuse, and a coarse producer in vLLM for DeepSeek-V4-Flash. Across four long-context suites, three consumer layers per producer give $1.14$--$1.40\times$ speedups and $14.8$--$27.4\%$ lower GPU energy, with observed accuracy changes of $-1.49$ to $+0.29$ percentage points in single runs. More extensive sharing e
    
[^166]: 基线与增量：Instagram信息流排序中的用户、物品与请求去偏

    Baseline and Incremental: User, Item, and Request Debiasing in Instagram Feed Ranking

    [https://arxiv.org/abs/2610.27423](https://arxiv.org/abs/2610.27423)

    该论文提出在Instagram信息流价值模型中同时保留原始预测（基线）并单独加权加入增量项，将观看者习惯、内容热门度和请求上下文三类独立部署的去偏统一为一个参数化算子，四臂消融实验证明基线加增量的双项设计在绝大多数指标上优于任何单项去偏方案。

    

    信息流排序器使用价值模型为候选内容打分，该模型是各项互动预测的加权和。每个预测都混淆了两个部分：基线——即该观看者、该内容或该请求本来就会产生的互动量——以及该候选内容超出基线的增量部分。现有文献大多采取移除基线的做法。我们主张，多信号价值模型应当保留原始预测，并将增量部分作为一个单独加权的项加入模型。在Instagram信息流中，三个独立部署的去偏方案族（分别针对观看者习惯、内容热门度和请求上下文）实际上可统一为一个参数化算子，并通过一套设计流程在该算子中选取一个具体的点。媒体类型（格式）偏差则转而由同一流程所选定的去偏标签来处理。在一项针对已部署信息流项的受控四臂消融实验（无任何项、仅含其一、两者兼具）中，在生产权重下的点估计上，双项方案在13个受评判的（项，指标）对中有11个优于两个单项方案。在26个直接……

    arXiv:2610.27423v1 Announce Type: cross  Abstract: A feed ranker scores candidates with a value model, a weighted sum of engagement predictions. Each prediction conflates a baseline, what this viewer, item or request would produce anyway, with the incremental amount by which this candidate exceeds it. The literature mostly removes the baseline. We argue that a multi-signal value model should keep the raw prediction and add the incremental part as a separately weighted term. On Instagram Feed, three debiasing families deployed independently, for viewer habit, item popularity and request context, are one parameterised operator, and a design procedure picks a point in it. Media-type (format) bias goes instead to a debiased label chosen by the same procedure. In a controlled four-arm ablation of deployed Feed terms (neither term, each alone, both), the two-term arm beats both single-term arms at production weights on point estimates in 11 of 13 judged (term, metric) pairs. Of the 26 direct
    
[^167]: 路由器早已知晓

    The Router Already Knows

    [https://arxiv.org/abs/2610.27421](https://arxiv.org/abs/2610.27421)

    提出NR-SAE稀疏自编码器，直接复用MoE路由器对每个token的稀疏专家选择来构建字典，在解释路由器加权专家输出方差上显著优于参数匹配的对照组。

    

    混合专家模型对每个token已经做出了一个稀疏选择：由哪些专家来处理它。我们探讨稀疏字典是否应当复用这一选择。NR-SAE为每个被路由的专家分配一个小型字典，并只打开被冻结路由器选中的那些字典。在Gemma 4 26B-A4B的第15个块上，经过一千万个C4训练token并在三个随机种子下，专家对齐自编码器在固定评估捕获上解释了路由器加权专家输出方差的0.518。一个参数量相同但候选数量为其十六倍的秩寻址对照组达到0.481，候选数匹配的对照组为0.458，而架构匹配但专家身份随机化的对照组为0.387。转码器变体同样领先所有三个对照组：0.432，对比0.347、0.371和0.118。这一优势部分源于架构本身：每个专家只写入至多四分之一模型宽度的子空间，因此指名专家即等于指名该子空间……（原文摘要至此截断）

    arXiv:2610.27421v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models already make a sparse choice for every token: which experts should process it. We ask whether a sparse dictionary should reuse that choice. NR-SAE gives each routed expert a small dictionary and opens only those the frozen router selects. At block 15 of Gemma 4 26B-A4B, after ten million C4 training tokens and across three seeds, the expert-aligned autoencoder explains 0.518 of the variance in router-weighted expert outputs on a fixed evaluation capture. A rank-addressed control with the same parameters but sixteen times as many candidates reaches 0.481, a candidate-matched control 0.458, and an architecture-matched control with randomized expert identity 0.387. A transcoder variant also leads all three controls: 0.432, against 0.347, 0.371, and 0.118. Part of this advantage is architectural: each expert writes into a subspace of at most a quarter of the model width, so naming the expert names the subspa
    
[^168]: 利用训练后的神经切线核学习材料空间的几何结构

    Learning the Geometry of Materials Space with Trained Neural Tangent Kernels

    [https://arxiv.org/abs/2610.27412](https://arxiv.org/abs/2610.27412)

    本研究利用训练后参数处的神经切线核，成功提取并分析了预测超导转变温度的神经网络所学到的材料空间几何结构，揭示了少数主导方向，并将化合物组织成可识别的超导家族，为理解神经网络材料表示的物理内涵提供了实用探针。

    

    经过训练用于预测材料性质的神经网络，其所学表示中蕴含的物理内容往往难以提取。我们使用在训练后参数处评估的神经切线核（NTK），来恢复一个仅从元素组成预测超导转变温度的网络所学习到的几何结构。基于训练后NTK的核岭回归能够很好地再现网络的预测性能，并显著优于使用初始核的回归方法。谱分析和多维标度分析揭示了一种由少数几个方向主导的几何结构。聚类和局部邻域分析将化合物组织成可识别的超导家族，并在转变温度 $T_c$ 相差悬殊的材料之间建立联系，这些材料具有相关的化学性质和强关联电子特性。这些结果确立了训练后的切线核作为探测所学材料表示的实用工具。

    arXiv:2610.27412v1 Announce Type: cross  Abstract: Neural networks trained to predict materials properties learn representations whose physical content can be difficult to extract. We use the neural tangent kernel (NTK), evaluated at trained parameters, to recover the geometry learned by a network that predicts superconducting transition temperatures from elemental composition alone. Kernel ridge regression with the trained NTK closely reproduces the network's predictive performance and substantially outperforms regression with the initial kernel. Spectral analysis and multidimensional scaling reveal a geometry dominated by a small number of directions. Clustering and local neighborhood analysis organize compounds into recognizable superconducting families and connect materials with related chemical and correlated-electron properties despite substantial differences in $T_c$. These results establish trained tangent kernels as practical probes of learned materials representations, connec
    
[^169]: 引导文本到语音合成中的语调曲调

    Steering intonational tunes in text-to-speech synthesis

    [https://arxiv.org/abs/2610.27403](https://arxiv.org/abs/2610.27403)

    该研究通过交换干预定位了残差向量量化文本到语音模型中英语核心语调的载体——主干网络的隐藏状态而非token堆栈第一层，并证明可用均值差向量引导该状态来实现对语调的控制。

    

    arXiv:2610.27403v1 公告类型：交叉 摘要：基于残差向量量化构建的文本到语音系统通常分两个阶段生成音频：一个transformer主干网络输出每帧token堆栈的第一层，而一个较小的深度模型自回归地生成其余部分。我们探究这类模型中哪些组件负责韵律语调，更准确地说，负责生成英语核心调（例如，区分陈述句与疑问句上升旋律的下降音高旋律），以及是否可以通过对这些组件的干预来控制语调。基于提示的控制大多失败；只有句末的问号是一个可靠的调节手段。在CSM和S2模型中进行的交换干预定位了语调的来源：主干网络的隐藏状态通过深度模型将语调传递到波形中，而token堆栈的第一层几乎不携带或不携带语调信息。通过使用在以句末（原文在此处截断）……

    arXiv:2610.27403v1 Announce Type: cross  Abstract: Text-to-speech systems built on residual vector quantization often generate audio in two stages: a transformer backbone emits the first level of each frame's token stack, and a smaller depth model generates the rest autoregressively. We ask which components of such a model are responsible for prosodic intonation, more precisely for generating the English nuclear tune (for instance, the falling pitch melody that distinguishes an assertion from a question's rising melody), and whether interventions on those components can be used to control intonation. Prompt-based control mostly fails; only a final question mark is a reliable lever. Interchange interventions in CSM and S2 locate the tune: the backbone's hidden state carries it to the waveform through the depth model, and the first level of the stack carries little or none of it. Steering the backbone state with a difference-of-means vector computed on new material generated with a final
    
[^170]: 通用特征的通信复杂度

    Communication Complexity of Universal Features

    [https://arxiv.org/abs/2610.27402](https://arxiv.org/abs/2610.27402)

    受机器学习特征概念启发，本文利用新颖的势函数论证和贪婪函数选择方法，构造了一种仅需 $O(d^*\log(1/\varepsilon))$ 比特（由对偶 VC 维决定）的通用数据表示，使得解码器能够在任意指定任务下以不超过 $\varepsilon$ 的误差完成预测。

    

    受到机器学习中“特征”概念的启发，我们研究了表示数据的复杂度，使得该数据表示能够用于给定函数族的评估。具体而言，我们考虑一个确定性编码器，它观测 $X\sim D$，并在任务 $f\in\mathcal{F}\subseteq\{0,1\}^{\mathcal{X}}$ 被指定之前产生一个有限编码。对于每个任务，解码器必须在分布 $D$ 下以不超过 $\varepsilon$ 的误差预测 $f(X)$。我们构造了一个编码器，它输出一个长度为 $O(d^{*}\log(1/\varepsilon))$ 比特的表示，其中 $d^{*}$ 是函数类 $\mathcal{F}$ 的对偶 VC 维。我们的编码器通过贪婪方式从函数类中选取 $O((d^{*}/\varepsilon)\log(1/\varepsilon))$ 个函数，并利用它们的联合标签模式来产生编码。上界证明的关键思想是一种新颖的势函数论证，即我们将每次贪婪选择所获得的熵与对偶 VC 维所允许的模式数量进行比较。我们展示了……

    arXiv:2610.27402v1 Announce Type: cross  Abstract: Motivated by the notion of features from machine learning, we study the complexity of representing data that allows for evaluation with a given family of functions. Concretely, we consider a deterministic encoder that observes $X\sim D$ and produces a finite encoding before a task $f\in\mathcal{F}\subseteq\{0,1\}^{\mathcal{X}}$ is specified. For each task, a decoder must predict $f(X)$ with error at most $\varepsilon$ under $D$. We construct an encoder that outputs a representation with $O(d^{*}\log(1/\varepsilon))$ bits, where $d^{*}$ is the dual VC dimension of $\mathcal{F}$. Our encoder greedily selects $O((d^{*}/\varepsilon)\log(1/\varepsilon))$ functions from the class and uses their joint label pattern to produce the encoding. The key idea in our upper bound is a novel potential function argument, where we compare the entropy gained from each greedy selection with the number of patterns allowed by the dual VC dimension. We presen
    
[^171]: 固定样本量的主动统计推断

    Fixed-Size Active Statistical Inference

    [https://arxiv.org/abs/2610.27389](https://arxiv.org/abs/2610.27389)

    该论文提出用固定样本量的条件泊松抽样（CPS）替代样本量随机的泊松抽样（PS），证明 CPS 是一般多变量 Z 估计问题（包括凸 M 估计）中 PS 的直接替代方案，且其渐近方差不劣于 PS，在现实场景中甚至可显著更优。

    

    方差缩减技术可以提高从数据中得出结论的效率。最近的方差缩减方案采用两种互补的技术来降低方差：将机器学习预测纳入估计器，以及加权抽样。一种流行的加权抽样算法是泊松抽样（Poisson Sampling, PS），它对每个单元独立地掷一枚有偏硬币，并将所有正面结果纳入样本。PS 的一个根本缺陷在于其样本量是随机的，这既带来了实际操作上的挑战，也给估计器引入了额外的方差。我们证明了一种固定样本量的替代方法——在抽样调查领域被称为条件泊松抽样（Conditional Poisson Sampling, CPS）——可以作为 PS 的直接替代方案，应用于一般的多变量 Z 估计问题，其中涵盖了先前工作中所考虑的凸 M 估计。值得注意的是，我们证明 CPS 的渐近方差不劣于 PS，并且在存在预测模型的现实场景中可以显著更优。

    arXiv:2610.27389v1 Announce Type: new  Abstract: Variance reduction techniques improve the efficiency of drawing conclusions from data. Recent proposals for variance-reduction use two complementary techniques to reduce variance: incorporating machine learning predictions into estimators and weighted sampling. A popular weighted sampling algorithm is Poisson Sampling (PS), which independently flips a biased coin at each unit and includes all heads in the sample. A fundamental flaw with PS is that it has a random sample size, which presents practical challenges and introduces variance into the estimator. We show that a fixed-size alternative, known in survey sampling as Conditional Poisson Sampling (CPS), is a drop-in replacement for PS on general multivariate Z-estimation problems, which include convex M-estimation considered in prior work. Notably, we show that the asymptotic variance of CPS is no worse than that of PS, and can be substantially better in realistic scenarios where predi
    
[^172]: 通过动作影响力最大化训练可控世界模型

    Training Controllable World Models via Action Influence Maximization

    [https://arxiv.org/abs/2610.27387](https://arxiv.org/abs/2610.27387)

    论文提出动作影响力最大化算法，通过主动收集具有高动作影响力的训练数据，突破专家演示数据的动作影响力上限，从而解决世界模型忽略指令动作的问题，实现更可控的动作条件视频生成世界模型。

    

    在成功的目标导向演示数据上训练的动作条件视频生成模型往往会忽略所指令的动作，无论要求它们做什么，都会生成一个看似合理、类似专家的未来预测。我们将这一失败归因于训练数据缺乏足够的动作影响力。我们分析了这一度量指标，并证明它上界了在该数据上训练的任何模型所能表达的动作影响力。专家演示的狭窄动作分布使得无需关注动作即可预测下一状态，从而设定了一个较低的上限，仅靠增加模型容量无法突破。要提高这一上限，需要收集不同的数据。我们提出了动作影响力最大化，这是一种用于主动收集训练数据以提升数据集动作影响力的算法。我们在模拟导航和操作任务上将AIM的性能和数据效率与探索启发式方法及内在动机方法进行了比较。

    arXiv:2610.27387v1 Announce Type: cross  Abstract: Action-conditioned video generative models trained on successful, goal-directed demonstrations tend to ignore the commanded action, rendering a plausible expert-like future regardless of what is asked of them. We trace this failure to training data that lack sufficient action influence. We analyze this metric and prove that it upper bounds the action influence that any model trained on that data can express. Expert demonstrations, whose narrow action distributions let next states be predicted without attending to the action, set a low ceiling that no amount of model capacity alone can raise. Raising it requires collecting different data. We introduce Action Influence Maximization (AIM), an algorithm for actively collecting training data for improving dataset action influence. We compare the performance and data-efficiency of AIM against exploration heuristics and intrinsic-motivation methods on simulated navigation and manipulation tas
    
[^173]: 基于稠密联想记忆的样本高效学习

    Sample-efficient learning with Dense Associative Memories

    [https://arxiv.org/abs/2610.27384](https://arxiv.org/abs/2610.27384)

    该论文解析地揭示了稠密联想记忆在训练中存在从单纯记忆样本到发现数据稀疏生成规则的“顿悟”式转变，并提出带阈值学习规则的新型稠密联想记忆，可执行简化的稀疏哈密顿量学习从而指数级降低样本复杂度。

    

    我们解析地研究了稠密联想记忆在随机扩展器低密度奇偶校验码（又称随机XORSAT问题的解）的随机采样码字上进行训练时，其能量函数的演化过程。我们识别出两种截然不同的状态区间以及二者之间的转变，这让人联想到神经网络中的“顿悟”现象。在小训练集规模的状态下，模型只能记忆少量训练样本，却无法发现整个码字字典生成过程的底层规则。而在相反的状态下，对应于更大的训练集规模，模型在仅见过多项式数量级的模式之后，便已揭示出生成整个数据集的稀疏约束。我们还设计了一类新颖的采用阈值学习规则的稠密联想记忆，并证明任何程度的阈值化都能使模型执行简化的稀疏哈密顿量学习，从而指数级地降低……

    arXiv:2610.27384v1 Announce Type: new  Abstract: We analytically study the evolution of the energy function of Dense Associative Memories (DenseAMs) as they train on randomly sampled codewords of random expander low-density parity-check codes (a.k.a. solutions of random XORSAT problems). We identify two distinct regimes and a transition between them, reminiscent of grokking in neural networks. A regime of small training set size is where the model memorizes a few training examples but cannot discover the underlying rules of the entire codeword dictionary generative process. The opposite regime, corresponding to a larger training set size, is where the model has uncovered the sparse constraints that generated the entire data set after seeing only polynomially many patterns. We also design a novel class of DenseAMs with a thresholding learning rule, and show that any amount of thresholding enables the model to perform simplified sparse Hamiltonian learning, which exponentially reduces th
    
[^174]: DStill：面向块扩散投机解码中加速单步草稿生成的轨迹一致性蒸馏

    DStill: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding

    [https://arxiv.org/abs/2610.27379](https://arxiv.org/abs/2610.27379)

    提出DStill两阶段蒸馏框架，通过轨迹一致性蒸馏将块扩散投机解码中草稿器的多步迭代细化压缩为单步草稿生成，在降低推理延迟的同时保持草稿的保真度。

    

    块扩散投机解码通过并行起草多个token来加速大语言模型推理。然而，离散扩散中的单步草稿生成受到token因子化误差的限制：每个token都是从同一个掩码块中预测得到的，而没有以该步骤中其他位置所选token作为条件。这会产生不一致的草稿并降低接受长度。迭代细化通过让后续预测以先前已确定的token为条件来缓解这一局限，但额外的推理过程会增加草稿生成成本。为解决草稿保真度与推理延迟之间的权衡，我们提出了DStill，这是一个两阶段框架，将目标条件化草稿器自身的多步细化过程蒸馏为单步草稿生成。我们首先训练草稿器完成部分揭示的块，然后利用其细化轨迹来训练学生模型在单次推理中预测最终完成结果。使用……

    arXiv:2610.27379v1 Announce Type: new  Abstract: Block-diffusion speculative decoding accelerates large language model inference by drafting multiple tokens in parallel. However, one-step drafting in discrete diffusion is limited by token factorization error: each token is predicted from the same masked block, without conditioning on the tokens selected at other positions in that step. This can produce inconsistent drafts and reduce acceptance length. Iterative refinement mitigates this limitation by conditioning later predictions on previously committed tokens, though additional passes increase drafting cost. To address this trade-off between draft fidelity and inference latency, we introduce DStill, a two-stage framework that distills a target-conditioned drafter's own multi-step refinement into one-step drafting. We first train the drafter to complete partially revealed blocks, then use its refinement trajectories to train a student to predict the final completion in one pass. Using
    
[^175]: 成形稀疏自编码器：扩展机制可解释性的工具箱

    Shaped SAEs: Expanding the Toolkit for Mechanistic Interpretability

    [https://arxiv.org/abs/2610.27375](https://arxiv.org/abs/2610.27375)

    本文提出通过在学习稀疏自编码器之前对语言模型的稠密激活进行线性“成形”变换（包括基于雅可比空间的降秩投影和与字典联合学习的仿射变换），来增强所学字典特征的语义选择性，从而扩展机制可解释性的方法工具箱。

    

    稀疏自编码器（SAE）旨在使用学习到的过完备字典，将大型语言模型（LLM）的稠密激活分解为稀疏且可能可解释的特征。在本工作中，我们探究是否可以在学习SAE之前对稠密激活进行线性“成形”，以增强所学字典原子的语义选择性。我们探索了基于最近提出的J空间的降秩变换，该空间对应于从当前层到LLM输出的平均雅可比矩阵，将激活投影到由具有较大奇异值的雅可比特征模所张成的“活跃”子空间上（我们期望这些特征模与语义更相关）。我们还引入了一种新的SAE训练方法，其中 仿射成形变换与字典原子一同学习， 原子选择机制基于沿一维子空间的TopK投影……

    arXiv:2610.27375v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) seek to decompose the dense activations of large language models (LLMs) into sparse, potentially interpretable features using a learned overcomplete dictionary. In this work, we ask whether it is possible to enhance the semantic selectivity of the learned dictionary atoms by linearly \emph{shaping} the dense activations prior to learning an SAE. We explore rank-reducing transformations based on the recently introduced J-space corresponding to the averaged Jacobian from the current layer to the LLM output, projecting activations onto the ``live'' subspace spanned by eigenmodes of the Jacobian with larger singular values (which we expect to be more semantically relevant). We also introduce a new approach for training SAEs in which (a)~an affine shaping transformation is learned along with the dictionary atoms, (b)~the atom selection mechanism is based on the TopK projections along the one-dimensional subspace spa
    
[^176]: 贝叶斯网络中用于近似结果预测的迹加权影响分数

    Trail-Weighted Influence Scores for Approximate Outcome Prediction in Bayesian Networks

    [https://arxiv.org/abs/2610.27366](https://arxiv.org/abs/2610.27366)

    该论文提出了一种基于迹加权影响分数的证据剪枝方法用于贝叶斯网络近似预测，证明了无碰撞节点网络中迹得分与端点精确相关性相等，并给出了剪枝后验误差的可计算界限，实验表明其误差比随机剪枝低1.4至13倍。

    

    我们研究了用于贝叶斯网络预测的证据迹加权剪枝方法。该方法通过遍历枚举从每个证据节点到结果节点的活跃迹，用每条迹的边强度乘积对迹进行评分，并舍弃得分低于阈值的证据。对于二值变量上的无碰撞节点网络（链和分叉结构），我们证明了迹得分等于迹两端点之间的精确相关性，具有相同边强度的链与分叉具有相同的联合分布，并且当每条被剪除的迹位于结果节点各自的分支中时，剪枝后的后验与精确后验之间的差异处于一个可计算的逐迹项之和的范围内。我们证明了这个界限，在所有小树上进行了穷举验证，并展示了当被剪除的迹共享同一分支时该界限会失效。在随机树上的模拟实验中，对迹得分进行阈值筛选相比相同规模的随机剪枝可将误差降低1.4至13倍。

    arXiv:2610.27366v1 Announce Type: cross  Abstract: We study trail-weighted pruning of evidence for Bayesian-network prediction. A traversal enumerates the active trails from each evidence node to the outcome, scores each trail by the product of its edge strengths, and drops evidence whose score falls below a threshold. For collider-free networks (chains and forks) over binary variables, we show that the trail score equals the exact correlation between a trail's endpoints, that a chain and a fork with equal edge strengths have identical joint distributions, and that the pruned posterior is within a computable sum of per-trail terms of the exact posterior when each dropped trail lies in its own branch of the outcome node. We prove this bound, check it exhaustively on all small trees, and show that it fails when dropped trails share a branch. In simulations on random trees, thresholding trail scores gives 1.4 to 13 times lower error than random pruning of the same size. We also apply the 
    
[^177]: 预测闭合：对JEPA特征选择的集合层面刻画

    Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection

    [https://arxiv.org/abs/2610.27364](https://arxiv.org/abs/2610.27364)

    该论文提出了“预测闭合”这一新刻画，证明JEPA的特征选择取决于特征集合层面的整体性质（即特征间的关系），而非单个特征的可预测性，从而化解了不可预测特征（如交通灯）不会被编码进表示的担忧。

    

    在联合嵌入预测架构（JEPA）中，编码器和预测器被一起训练，以产生一个能够最小化对下一观测的表示预测误差的表示。由于当前观测和下一观测的表示都是通过学习得到的，因此并不清楚它们将包含哪些特征。早期的刻画以单个特征自身的可预测性来描述JEPA所选择的特征，这引发了一种担忧：不可预测的特征（例如交通灯）将不会被编码进表示中。然而，预测误差还会受到特征之间关系的影响，例如，灯的颜色有助于预测交通是否会停止。我们提出了预测闭合这一刻画方式，它将这些关系纳入考量。基于对JEPA训练目标的重新表述，我们证明了特征选择取决于特征集合的性质。这一观点与……（摘要原文在此处截断）

    arXiv:2610.27364v1 Announce Type: cross  Abstract: In joint-embedding predictive architectures (JEPA), an encoder and a predictor are trained together to produce a representation that minimizes the error in predicting the representation of the next observation. Because the representations of both the current and the next observation are learned, it is unclear which features they will include. Earlier characterizations describe the features JEPA selects in terms of their individual predictability, raising the concern that unpredictable features, like a traffic light, will not be encoded into the representation. However, the prediction error is also affected by the relationships between features, e.g., the light's color helps predict whether the traffic will stop. We propose the predictive closure characterization, which accounts for these relationships. Based on a reformulation of the JEPA training objective, we show that selection depends on properties of feature sets. This view agrees
    
[^178]: LiBRA：用于联邦学习中拜占庭鲁棒客户端选择的Logit指纹

    LiBRA: Logit Fingerprints for Byzantine-Robust Client Selection in Federated Learning

    [https://arxiv.org/abs/2610.27357](https://arxiv.org/abs/2610.27357)

    LiBRA通过比较客户端模型的logit指纹而非参数向量来实现联邦学习中的拜占庭鲁棒客户端选择，从而克服了传统方法需要访问原始更新、计算成本高以及仅适用于相同架构客户端的限制。

    

    拜占庭鲁棒聚合为联邦学习提供了抵御恶意更新的保护，但过滤可疑更新的规则是通过比较客户端的参数向量来实现的。这种比较需要访问每个客户端的更新，而安全聚合的目的恰恰是为了隐藏这些更新；这种比较在参数维度 d 上具有 O(n²d) 的计算成本；并且仅在共享相同架构的客户端之间才有定义。我们提出 LiBRA，它通过比较客户端模型计算出的结果而非计算这些结果的参数来比较客户端，使用一个包含 C² 个值的指纹（C 为类别数），即每个客户端模型对针对其优化的伪模式的类条件 logit 响应。我们将 LiBRA 实例化为 Krum、Multi-Krum 和 Bulyan 的变体，这些变体仅改变了不相似度分数的计算对象。在 i.i.d. 数据划分下，使用 Fashion-MNIST 和 CIFAR-10 数据集、三种架构以及四种后门和标签操纵攻击，每种变体都能抵御所有攻击

    arXiv:2610.27357v1 Announce Type: cross  Abstract: Byzantine-robust aggregation shields federated learning from malicious updates, but the rules that filter out suspect updates do so by comparing clients' parameter vectors. That comparison requires access to every client's update, which secure aggregation exists to withhold, costs $O(n^2 d)$ in the parameter dimension $d$, and is defined only between clients that share an architecture. We propose LiBRA, which compares clients on what their models compute rather than on the parameters that compute them, using a fingerprint of $C^2$ values for $C$ classes, the class-conditional logit response of each client's model to pseudo-patterns optimised against it. We instantiate LiBRA as variants of Krum, Multi-Krum and Bulyan that change only what the dissimilarity score is computed on. On Fashion-MNIST and CIFAR-10 with three architectures and four backdoor and label-manipulation attacks under i.i.d. partitions, each variant holds every attack 
    
[^179]: 网络流量异常下鲁棒的轻量级物联网设备指纹识别

    Robust Lightweight IoT Device Fingerprinting Under Network Traffic Anomalies

    [https://arxiv.org/abs/2610.27351](https://arxiv.org/abs/2610.27351)

    提出了一种基于数据包头和局部敏感哈希的轻量级物联网设备指纹识别方法，在使用短观测窗口并遭受丢包、扰动、注入等多种流量异常时，仍能实现高达99%的识别准确率并保持鲁棒性。

    

    准确的物联网设备识别对网络安全至关重要，然而许多现有方法依赖载荷检测、特征工程或较长的观测窗口，这限制了可扩展性，并且尽管真实物联网流量本身具有固有变异性，这些方法通常假设网络状况稳定。我们提出一种基于数据包头与局部敏感哈希（LSH）的轻量级物联网设备识别方法，在数据包级和窗口级粒度上生成紧凑的指纹表示。为评估鲁棒性，我们考虑了包括丢包、扰动、乱序、重复以及数据包注入在内的数据包级流量异常。实验结果表明，该方法在使用短而有界的观测窗口的情况下，在真实流量失真下仍保持鲁棒性，准确率最高可达99%。影响全局流量结构的异常会造成最大的性能下降，而窗口……（摘要在此处截断）

    arXiv:2610.27351v1 Announce Type: cross  Abstract: Accurate IoT device identification is essential for network security, yet many existing approaches rely on payload inspection, feature engineering, or long observation windows that limit scalability and often assume stable network conditions despite the inherent variability of real-world IoT traffic. We propose a lightweight IoT device identification approach based on packet headers and locality-sensitive hashing (LSH), generating compact fingerprint representations at packet- and window-level granularities. To evaluate robustness, we consider packet-level traffic anomalies including drop, perturbation, reordering, duplication, and packet injection. Experimental results show that the proposed approach achieves up to 99% accuracy while maintaining robustness under realistic traffic distortions using short bounded observation windows. Anomalies affecting global traffic structure produce the largest performance degradation, whereas window
    
[^180]: 跨建筑的Brick模式知识图谱自动补全

    Automated Brick Schema Knowledge Graph Completion Across Buildings

    [https://arxiv.org/abs/2610.27339](https://arxiv.org/abs/2610.27339)

    本文首次将知识图谱补全技术应用于Brick建筑元数据模型的自动生成，通过比较三种方法自动推断建筑设备、传感器和空间之间的缺失关系，从而大幅减少人工建模的工作量。

    

    Brick模式将建筑设备、传感器、空间及其相互关系表示为带类型的多关系图，提供机器可读的元数据，从而支持可泛化的控制应用。目前生成完整Brick模型的方法需要对建筑图纸和点位列表进行大量的人工检查。我们观察到，由于Brick模型本质上是知识图谱，推断其中缺失的关系可以自然地表述为一个知识图谱补全问题。据我们所知，这是首次将知识图谱补全技术应用于Brick模型生成。我们在feeds、hasPart和hasPoint这三种关系上比较了三种方法——梯度提升树、成对Brick关系排序和CompGCN——这些关系占到了人工构建Brick模型时阅读图纸工作的绝大部分。模型在留出的MORTAR建筑数据以及分布外的Kaggle和Building Time Series数据集上进行了评估。

    arXiv:2610.27339v1 Announce Type: new  Abstract: The Brick schema represents building equipment, sensors, spaces, and the relationships between them as typed, multi-relational graphs, providing machine-readable metadata that supports generalizable control applications. Current methods for producing complete Brick models require extensive manual examination of building schematics and point lists. We observe that as Brick models are knowledge graphs, inferring missing relations is naturally posed as a knowledge graph completion problem. To the best of our knowledge, this is the first application of knowledge graph completion to Brick model generation. We compare three methods, gradient-boosted trees, Pairwise Brick Relation Ranking, and CompGCN, on the relations feeds, hasPart, and hasPoint, which account for the majority of the schematic-reading work in building a Brick model by hand. Models are evaluated on held-out MORTAR buildings and out-of-distribution Kaggle and Building Time Seri
    
[^181]: EMBER：无需对接复合物的TCR-pMHC结合预测

    EMBER: Predicting TCR--pMHC Binding Without a Docked Complex

    [https://arxiv.org/abs/2610.27334](https://arxiv.org/abs/2610.27334)

    EMBER通过独立折叠TCR和pMHC并将其编码为SE(3)不变的残基图进行比较，无需构建对接复合物即可预测TCR-pMHC结合，从而消除了放置误差的传播，并将计算复杂度从O(NM)降低到O(N+M)。

    

    我们提出EMBER，这是一个基于结构的TCR-pMHC结合预测框架，它消除了对预测对接复合物的需求，从而消除了主导此类预测的放置误差。传统的基于结构的预测器利用受体和抗原的共折叠复合物结构，并从其接触面来解释结合。因此，每一个下游特征都会继承结构预测工具所引入的任何偏差。在方法学上，EMBER独立地折叠TCR和pMHC，将它们编码为SE(3)不变的残基图，并在不构建共享坐标系的情况下比较它们学习到的表示。因此，相对位姿不再是模型的输入，放置误差也就无法传播到预测中。在计算上，EMBER需要预处理O(N+M)个单体折叠，而基于复合物的预测器则需要O(NM)次运算。在实验上，在表位不相交的评估下……（摘要原文此处被截断）

    arXiv:2610.27334v1 Announce Type: cross  Abstract: We propose EMBER, a structure-based framework for TCR--pMHC binding prediction that eliminates the need for a predicted docked complex, and with it the placement error that dominates such predictions. Traditional structure-based predictors utilize the co-folded complex structure of the receptor and the antigen and interpret binding from its contact surface. Consequently, every downstream feature inherits any deviation the structure-prediction tool introduced. Methodologically, EMBER folds the TCR and pMHC independently, encodes them as SE(3)-invariant residue graphs, and compares their learned representations without constructing a shared coordinate frame. A relative pose is therefore not an input of the model, so placement error cannot propagate into the prediction. Computationally, EMBER preprocesses $O(N{+}M)$ monomer folds where a complex-based predictor needs $O(NM)$ operations. Experimentally, under epitope-disjoint evaluation wi
    
[^182]: 释放VampPrior的全部潜力

    VampPrior Unleashed

    [https://arxiv.org/abs/2610.27322](https://arxiv.org/abs/2610.27322)

    该论文通过解耦VampPrior的先验与后验协方差，并用狄利克雷过程取代固定的均匀混合权重，显著提升了模型证据并实现了稳健的非参数聚类。

    

    深度潜变量模型（DLVMs）支撑着许多现代分析流程，例如在单细胞基因组学中，但其先验分布却鲜受关注。VampPrior近似最优先验（聚合后验），通常能获得较高的模型证据。在此，我们识别出VampPrior的两个关键局限：它将先验与后验的不确定性绑定在同一个摊销协方差上，并且由于其固定的均匀混合权重而导致聚类效果不佳。我们通过解耦先验与后验协方差来释放VampPrior的潜力，解决了用同一组共享参数同时产生高先验熵和低后验熵的冲突，并显著提升了模型证据。随后，我们用狄利克雷过程取代均匀混合。两者结合后，我们的方法相比VampPrior提供了更高的模型证据以及稳健的非参数聚类。将我们的[方法]与单细胞变分推断（scVI）相结合……

    arXiv:2610.27322v1 Announce Type: new  Abstract: Deep latent variable models (DLVMs) underpin many modern analysis pipelines, for example in single-cell genomics, yet their prior distributions have received markedly little attention. The VampPrior approximates the optimal prior (the aggregate posterior) and often attains high model evidence. Here, we identify two key limitations of the VampPrior: it ties prior and posterior uncertainty to a single amortized covariance and clusters poorly due to its fixed uniform mixture weights. We unleash the VampPrior by decoupling its prior and posterior covariances, resolving the conflict of generating high prior and low posterior entropy with the same shared parameters and substantially improving model evidence. We then replace the uniform mixture with a Dirichlet process. When combined, our proposals provide higher model evidence than the VampPrior and robust non-parametric clustering. Augmenting single cell Variational Inference (scVI) with our 
    
[^183]: 超越稳定性边缘的物理启发优化器：周期二轨道与延迟失稳

    Physics-Inspired Optimizers beyond the Edge of Stability: Period-Two Orbits and Delayed Instability

    [https://arxiv.org/abs/2610.27321](https://arxiv.org/abs/2610.27321)

    本研究揭示了物理启发优化器（动力学映射）在稳定性边缘之外会产生对称的周期二轨道，使训练稳定在一个“有效稳定性边缘”上，该轨道的线性稳定性由映射的弹性参数（介于0和1之间）决定，且VRAdam的阻尼机制在超过两倍临界学习率时会失效。

    

    带动量的梯度下降可以被看作是一个粒子在损失景观中带摩擦力地运动。在深度学习中，学习率通常较大，因此训练构成一个离散动力系统：在许多全批量实验中，锐度会增长到优化器的稳定性极限——即稳定性边缘（Edge of Stability, EoS），并在该边缘处继续训练。物理启发的优化器改变了这种运动方式；例如，速度正则化Adam（VRAdam）会在速度较高时缩小步长。我们研究了动力学映射，其中步长是动量之动能的梯度，并探究它们在稳定性边缘之外的行为。(i) 在一个简化模型中，动力学映射会在边缘之外产生一个对称的周期二轨道，当且仅当该映射的弹性（步长与动量之间的对数-对数斜率）介于0和1之间时，该轨道才是严格线性稳定的；训练在该轨道上处于一个“有效的稳定性边缘”。VRAdam的阻尼器在超过两倍临界学习率时会失效，

    arXiv:2610.27321v1 Announce Type: cross  Abstract: Gradient descent with momentum can be viewed as a particle moving with friction in the loss landscape. In deep learning the learning rate is usually large, so training is a discrete dynamical system: in many full-batch experiments the sharpness grows to the optimizer's stability limit, the edge of stability (EoS), and training continues there. Physics-inspired optimizers change this motion; Velocity-Regularized Adam (VRAdam), for example, shrinks the step at high speed. We study kinetic maps, in which the step is the gradient of a kinetic energy of the momentum, and ask what they do beyond the EoS. (i) In a reduced model, a kinetic map creates a symmetric period-two orbit beyond the edge, strictly linearly stable if and only if the map's elasticity (the log-log slope of step length versus momentum) lies between 0 and 1; on it, training sits at an effective edge of stability. VRAdam's damper fails above twice the critical learning rate,
    
[^184]: 什么不该擦除：面向低资源手写字符识别的显著性约束数据增强

    What Not to Erase: Saliency-Constrained Augmentation for Low-Resource Handwritten Character Recognition

    [https://arxiv.org/abs/2610.27309](https://arxiv.org/abs/2610.27309)

    该论文提出一种显著性引导的数据增强策略，通过对字符图像中视觉信息丰富的区域进行扰动、保留或约束来定义增强变体，使低资源手写字符识别在获得性能提升的同时保持显著性模式的稳定性。

    

    arXiv:2610.27309v1 公告类型：cross 摘要：在低资源历史材料藏品中，光学字符识别（OCR）和手写字符识别（HCR）通常依赖数据增强，然而增强策略的评估几乎完全依赖于预测准确率。这就留下了一个悬而未决的问题：模型性能的提升是否伴随着显著性的稳定性。这一问题在低资源历史手写字符识别中尤为重要，因为识别对象是多变且部分退化的字形，两个预测性能相似的模型在显著性模式对这种变化的响应一致性上可能存在差异。因此，我们研究显著性引导的数据增强，作为一种解释感知型的低资源手写字符识别策略。我们不再仅仅将显著性图作为事后解释工具，而是利用它们定义增强变体，对字符图像中视觉信息丰富的区域进行扰动、保留或约束。我们比较了经典的……

    arXiv:2610.27309v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) and handwritten character recognition (HCR) in low-resource historical material collections often rely on data augmentation, yet augmentation strategies are typically evaluated almost exclusively through predictive accuracy. This leaves open whether a model's improved performance is accompanied by saliency stability. This question is important in low-resource historical HCR, where recognition is performed on variable and partially degraded letterforms and two models with similar predictive performance may differ in how consistently their saliency patterns respond to such variation. We therefore study saliency-guided augmentation as an explanation-aware strategy for low-resource HCR. Rather than treating saliency maps only as post-hoc interpretations, we use them to define augmentation variants that perturb, preserve, or constrain visually informative regions of character images. We compare classical 
    
[^185]: 全球安全盲区：翻译作为低资源语言中的安全绕过手段

    Global Security Blind Spots: Translation as a Safety Bypass in Low-Resource Languages

    [https://arxiv.org/abs/2610.27304](https://arxiv.org/abs/2610.27304)

    该论文揭示了大型语言模型的英语安全对齐无法可靠迁移至低资源语言——在英语中被成功拒绝的有害请求经翻译后可诱发不安全回答，各模型平均绕过率高达21.9%至44.4%，构成全球性的安全盲区。

    

    大型语言模型在全球范围内部署，然而其安全对齐与评估却仍集中于英语：在英语中被拒绝的有害请求，一旦被翻译成低资源语言，便可能引发不安全的回答。我们在翻译绕过威胁模型下，对十个广泛部署的大语言模型在79种语言（英语及78种低资源语言）上进行了评估，并采用一种以英语安全性为条件的度量方法，从而将翻译绕过效应与英语本身已存在的不安全性区分开来。研究结果表明，英语的安全性无法可靠迁移：在英语回答安全的前提下，各模型的平均绕过率介于21.9%至44.4%之间。绕过风险集中于部分语系之中，且这一模式在各模型间普遍存在。尽管这种共同的排序与翻译退化存在混淆，但其在不同模型间的差异是显著的，并且经受住了我们施加的所有控制。我们还发现，单看各模型自身的绕过率无法得到一个稳定的排名：十个模型中有九个……（原文在此截断）

    arXiv:2610.27304v1 Announce Type: cross  Abstract: Large language models are deployed globally, yet safety alignment and evaluation remain concentrated in English: harmful requests refused in English can elicit unsafe completions when translated into low-resource languages. We evaluate ten widely deployed LLMs across 79 languages (English plus 78 low-resource) under a translation-bypass threat model, using a metric conditioned on English safety, which separates translation bypass from unsafety already present in English. English safety does not transfer reliably: conditioned on safe English completions, mean bypass rates range from 21.9% to 44.4%. Bypass risk concentrates in a subset of language families, a pattern broadly shared across models. The shared ordering is confounded with translation degeneracy, but its variation across models is significant and survives every control we apply. We also show that per-model bypass rates do not identify a stable ranking: nine of ten models chan
    
[^186]: TEX：通过基于执行的交叉验证实现测试时代理的测试时扩展

    TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation

    [https://arxiv.org/abs/2610.27290](https://arxiv.org/abs/2610.27290)

    TEX提出了一种基于执行驱动交叉验证的协作式测试时扩展框架，通过并行代理生成测试和补丁并交叉执行形成K×K反馈矩阵实现客观验证，在SWT-Bench Verified上以86.4%的成绩创下新的最先进水平。

    

    测试用例生成对语言模型至关重要，但仍然具有挑战性。现有的测试时扩展方法要么孤立地处理候选方案，要么聚合执行轨迹，这些方法无法扩展到长时程任务。我们提出了TEX，一个基于执行驱动交叉验证的协作式测试时扩展框架。TEX并行化多个代理来生成测试和补丁，将每个测试与每个补丁交叉执行，形成一个K×K的反馈矩阵。各代理利用自己的矩阵切片进行改进，无需冗长的轨迹共享即可实现客观验证。在SWT-Bench Verified基准上，TEX达到了86.4%的准确率，创下新的最先进水平，比单代理基线提升了9.0%。此外，即使从TEX输出中随机选择（84.4%）也超越了之前的SOTA（84.0%），表明协作式执行反馈提升的是解决方案的整体质量，而不仅仅是最佳样本。我们的研究结果突出了基于执行的反馈作为一种有效的扩展机制。

    arXiv:2610.27290v1 Announce Type: new  Abstract: Test-case generation is critical but remains challenging for language models. Existing test-time scaling methods either isolate candidates or aggregate traces, which do not scale to long-horizon tasks. We present TEX, a collaborative test-time scaling framework based on execution-driven cross-validation. TEX parallelizes agents to generate tests and patches, cross-executing each test against every patch to form a $K \times K$ feedback matrix. Agents use their matrix slices for refinement, enabling objective validation without verbose trace sharing. On SWT-Bench Verified, TEX achieves 86.4%, establishing a new state-of-the-art and improving +9.0% over single-agent baselines. Moreover, even random selection from TEX outputs (84.4%) surpasses the previous SOTA (84.0%), showing collaborative execution feedback improves solution quality, not just best-case samples. Our results highlight execution-grounded feedback as an effective scaling mech
    
[^187]: VPFlow：基于不完美世界模型的价值预测规划实现离线目标达成

    VPFlow: Value-Predictive Planning with Imperfect World Models for Offline Goal Reaching

    [https://arxiv.org/abs/2610.27280](https://arxiv.org/abs/2610.27280)

    该论文提出VPFlow，其核心创新在于证明离线目标达成中的规划依赖的是价值预测而非准确的状态预测，并利用闭式蒙特卡洛回报直接回归目标达成价值来监督潜在世界模型，从而在无需精确世界模型的情况下实现高效规划。

    

    离线目标条件强化学习使智能体无需稠密奖励即可通过从任意状态到达任意目标来习得技能；基于模型的方法则可以通过扩展测试时计算来进行规划，从而解决复杂任务。将二者结合，对于在次优演示数据上进行预训练，并通过指定目标和规划实现零样本适应，具有诱人的前景。然而，现有方法假设有效规划需要在像素空间或潜在空间中对下一状态进行准确预测。在这项工作中，我们证明价值预测对规划更为有用，而准确的状态预测往往是不必要的。我们的关键洞察是：离线目标达成天然地存在闭式蒙特卡洛回报，使价值回归能够取代时序差分自举。我们将该回报作为潜在世界模型的唯一监督信号：想象出的潜在状态被训练来预测目标达成价值，而非匹配未来状态……（摘要原文在此处截断）

    arXiv:2610.27280v1 Announce Type: cross  Abstract: Offline goal-conditioned reinforcement learning enables skill acquisition by reaching any goal from any state without dense rewards. Model-based methods allow for planning by scaling test-time compute to solve complex tasks. Combining both holds appealing promise for pre-training on suboptimal demonstrations and zero-shot adaptation via goal specification and planning. Yet existing methods assume effective planning requires accurate next-state prediction in pixel or latent space. In this work, we show that value prediction is more useful for planning, whereas accurate state prediction is often unnecessary. Our key insight is that offline goal reaching naturally admits a closed-form Monte Carlo return, enabling value regression to replace temporal-difference bootstrapping. We use this return as the sole supervision for latent world models: imagined latents are trained to predict goal-reaching value rather than match future states, impli
    
[^188]: 一种用于逼真啮齿动物运动的潜在控制模型

    A latent control model for realistic rodent motion

    [https://arxiv.org/abs/2610.27279](https://arxiv.org/abs/2610.27279)

    提出了SCAMPER系统，利用具有双重先验的变分架构训练具身虚拟大鼠模型，使其在目标导向行为和随机探索中都能产生自然的运动，弥补了以往具身动物建模的不足。

    

    开发动物行为的具身模型是神经-AI领域的一个重要目标。具身模型将增强我们对生物智能体和人工智能体进行直接比较的能力。然而，要使这种比较有效，我们需要真正像动物一样运动的具身模型。先前的工作已经开发了虚拟动物身体以及通过模仿学习训练逼真运动的技术。然而，迄今为止，还没有一个具身动物建模系统能够既在经过训练的目标导向行为中，又在由随机生成驱动的探索过程中产生自然的运动。在这里，我们介绍了SCAMPER（通过增强运动先验实现具身强化学习模拟控制），这是一个训练具身虚拟大鼠模型的系统，能够在目标导向行为和随机探索过程中都实现自然的运动。SCAMPER使用一种具有两种不同先验的变分架构，即一种用于……的运动先验

    arXiv:2610.27279v1 Announce Type: new  Abstract: The development of embodied models of animal behavior is a major goal of the neuro-AI community. Embodied models will enhance our ability to make direct comparisons between biological and artificial agents. To make such comparisons effective, though, we need embodied models that actually move like animals. Previous work has developed virtual animal bodies and techniques for training realistic motion through imitation. However, to date, there is no system for embodied animal modeling that produces natural movements both during trained goal-directed behavior, and during exploration driven by random generation. Here, we introduce Simulated Control via Augmented Motor Priors for Embodied Reinforcement learning (SCAMPER), a system for training an embodied virtual rat model, enabling natural movements during both goal-directed behavior and random exploration. SCAMPER uses a variational architecture with two distinct priors, a motion prior for 
    
[^189]: 移除此损坏会带来什么改变？面向视觉-语言-动作模型的失败归因

    What Would Removing This Corruption Change? Failure Attribution for Vision-Language-Action Models

    [https://arxiv.org/abs/2610.27277](https://arxiv.org/abs/2610.27277)

    该论文提出了“失败归因”这一新问题，并引入分解式风险模型，通过潜在干预仅凭当前图像和任务指令即可估计移除视觉损坏的收益，无需成对干净图像或昂贵的反事实策略回放。

    

    视觉-语言-动作（VLA）模型对视觉损坏十分敏感，但知道某种损坏通常有害，并不能确定移除它是否会对当前回合有益。此外，当多个损坏同时出现时，决定应该纠正什么，需要理解哪些损坏导致了失败风险以及移除它们能带来多大帮助。我们将这一问题定义为“失败归因”。仅检测损坏的存在无法回答这个问题，因为损坏可能存在却相对无害；而直接测量移除收益则需要昂贵的成对策略回放（即分别在有损坏和无损坏情况下运行）。为解决这一问题，我们引入了一种分解式风险模型，它学习场景与损坏的独立表示，使我们能够通过潜在干预，仅凭当前图像和任务指令来估计移除收益，而无需成对的干净图像或额外的反事实回放。

    arXiv:2610.27277v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models are sensitive to visual corruptions, but knowing that a corruption is typically harmful does not establish whether removing it would benefit the current episode. Furthermore, when several corruptions co-occur, deciding what to correct requires understanding which ones contribute to failure risk and how much their removal would help. We define this problem as failure attribution. Detecting corruption presence alone cannot answer this question, as corruptions can be present but relatively harmless, while directly measuring removal benefits requires costly paired policy rollouts with and without the corruption present. To address this, we introduce a factorized risk model that learns separate representations of the scene and corruptions, allowing us to estimate removal benefits through latent intervention from the current image and task instruction, without paired clean images or additional counterfactu
    
[^190]: 基于记录推导的安全偏好监督审计

    Auditing Safety Preference Supervision with Recorded Derivations

    [https://arxiv.org/abs/2610.27266](https://arxiv.org/abs/2610.27266)

    该论文提出符号化偏好标注方法，通过记录属性分类、明确规则和固定判定排序，使每个安全偏好标签都能从存储的推导过程中重新计算验证，从而实现对偏好监督的可审计性。

    

    偏好标签用于对齐模型训练，但其背后的判断难以检查或验证。在本文中，我们研究了当中间决策被记录时如何对偏好构建过程进行检查，以及其中哪些部分仍然存在不确定性。我们提出了符号化偏好标注，这是一种由预训练语言模型对提示词和响应的预定义属性进行分类的方法。明确的规则为响应分配判定结果，固定的判定排序决定偏好标签。每个偏好标签都具有可记录的推导过程：响应判定和偏好标签可以从存储的属性分配、规则和判定排序中重新计算得出，而无需重新运行语言模型。在由人工标注者标注的200个提示词上，语言模型为192个提示词生成了有效的结构化输出。全部八次解析失败都发生在被标记为有害的自残提示词子集中。使用465个偏好对，我们训练……

    arXiv:2610.27266v1 Announce Type: cross  Abstract: Preference labels align model training, but the judgments behind them are difficult to inspect or verify. In this paper, we examine how preference construction can be checked when intermediate decisions are recorded, and which of them remain uncertain. We present Symbolic Preference Labeling, a method in which a pretrained language model classifies predefined attributes of prompts and responses. Explicit rules assign response verdicts, and a fixed verdict ranking determines the preference label. Each preference label has a recorded derivation: the response verdicts and preference label can be recomputed from the stored attribute assignments, rules, and verdict ranking without rerunning the language model. On 200 prompts annotated by human raters, the language model produces valid structured outputs for 192 prompts. All eight parsing failures occur in the subset of self-harm prompts marked as harmful. Using 465 preference pairs, we trai
    
[^191]: 解决高斯混合模型中的Ho--Nguyen猜想：在专家混合模型中的应用

    Solving the Ho--Nguyen Conjecture for Gaussian Mixtures: Applications to Mixtures of Experts

    [https://arxiv.org/abs/2610.27265](https://arxiv.org/abs/2610.27265)

    本文解决了Ho--Nguyen猜想，证明对所有聚类大小M≥2的高斯混合，实数非零权重系统的密度抵消阶数恰为2M，为过拟合高斯混合模型和专家混合模型的参数学习速率提供了严格的理论保证。

    

    过拟合的高斯混合模型可能包含许多密度几乎相同的参数配置，这使得密度抵消的阶数成为其参数学习速率的核心问题。Ho和Nguyen将这些抵消编码为一个多项式系统，并猜想其精确阶数是聚类大小的函数。解决该猜想的关键是一种共同的正方差平移，它将系数方程转化为高斯矩匹配问题。对于每个聚类大小 $\Mhn\ge2$，我们证明了实数、非零权重系统的阶数恰为 $2\Mhn$：一个锐利的符号变化界在该阶数处强制刚性，而正Gauss--Hermite求积则构造出直至 $2\Mhn-1$ 阶的非平凡解。以110位有效数字精度进行的总体密度计算恢复了两到六个分量情形下预测的首项；有限多重启动搜索在两到五个分量的情形下恢复出相同的观测幂次，较小的有限规模……

    arXiv:2610.27265v1 Announce Type: new  Abstract: Over-fitted Gaussian mixtures can contain many parameter configurations with nearly the same density, making the order of density cancellation central to their parameter-learning rates. Ho and Nguyen encoded these cancellations in a polynomial system and conjectured its exact order as a function of cluster size. The key to resolving the conjecture is a common positive-variance shift that turns the coefficient equations into Gaussian moment matching. For every cluster size $\Mhn\ge2$, we prove that the real, nonzero-weight system has order $2\Mhn$: a sharp sign-change bound forces rigidity at that order, while positive Gauss--Hermite quadrature constructs a non-trivial solution through order $2\Mhn-1$. Population-density calculations at 110-digit working precision recover the predicted leading term for two to six components; finite multistart searches recover the same observed powers for two to five components, with smaller finite-scale c
    
[^192]: 面向可信合成文本生成的引导式扩散方法

    Guided Diffusion for Trustworthy Synthetic Text Generation

    [https://arxiv.org/abs/2610.27259](https://arxiv.org/abs/2610.27259)

    该论文提出了“引导式扩散”这一私有合成文本生成新范式，通过利用扩散语言模型的可引导性，以新颖的采样和重新掩码机制引导去噪过程，生成既符合差分隐私保证又兼具高数据效用的高质量合成文本。

    

    大规模语言模型的快速发展得益于海量数据集的支撑，然而这种依赖带来了严重的隐私风险，因为模型可能无意中记住甚至直接输出敏感的训练数据。合成文本生成为这一挑战提供了一种有前景的解决方案；然而，现有方法难以在严格的隐私保证与高数据效用之间取得平衡。在本文中，我们提出了一种私有合成文本生成的新范式：引导式扩散。利用扩散语言模型独特的可引导性，我们引入了新颖的采样与重新掩码机制，主动引导迭代去噪步骤。这些机制引导生成过程重构出符合目标语义分布的流畅文本。我们证明了该框架与差分隐私（DP）天然兼容：一旦私有语料的语义嵌入得到保护，差分隐私保证便会随之延续。

    arXiv:2610.27259v1 Announce Type: cross  Abstract: The rapid evolution of Large Language Models is fueled by massive datasets, yet this reliance poses severe privacy risks, as models can inadvertently memorize and even directly output sensitive training data. Synthetic text generation offers a promising solution to this challenge; however, existing methods struggle to reconcile rigorous privacy guarantees with high data utility. In this paper, we propose a new paradigm for private synthetic text generation: Guided Diffusion. Leveraging the unique steerability of Diffusion Language Models, we introduce novel sampling and re-masking procedures that actively guide the iterative denoising steps. These mechanisms steer the generative process to reconstruct fluent text that adheres to a target semantic distribution. We demonstrate that our framework is naturally compatible with Differential Privacy (DP): once the semantic embeddings of the private corpus are protected, the DP guarantee carri
    
[^193]: TxnQA：针对交易事件序列的分析推理与证据溯源基准测试

    TxnQA: Benchmarking Analytical Reasoning and Evidence Grounding over Transaction Event Sequences

    [https://arxiv.org/abs/2610.27256](https://arxiv.org/abs/2610.27256)

    TxnQA提出了一个针对交易事件序列分析推理的新基准，包含27个任务族和欺诈风险推理套件，评估发现LoRA适配虽能有效规范输出格式，但算术能力仍是模型性能的主要瓶颈。

    

    分析商户交易历史需要在不规则事件流上进行动态时间过滤、实体解析和聚合操作——这些操作在静态表格或时间序列预测基准中难以得到充分体现。我们提出TxnQA，一个分析型问答基准，包含27个任务族（81个模板）以及一个5场景的欺诈风险推理套件，该套件在合成的、经隐私脱敏处理的交易数据上同时评估判定结果和支持证据。标准答案通过程序化方式计算，模型性能被分解为格式遵循率（FFR）、条件内容准确率（CA）和端到端正确性（E2E）。对基于提示的专有基线模型和六个经LoRA适配的开源权重模型的评估表明，适配能够可靠地规范输出模式，但仍存在算术瓶颈：Gemma-4-31B-IT和Qwen3-8B分别取得0.867和0.816的E2E分数，而所评估的MoE模型检查点仅达到0.632。

    arXiv:2610.27256v1 Announce Type: new  Abstract: Analyzing merchant transaction histories requires dynamic temporal filtering, entity resolution, and aggregation over irregular event streams--operations poorly captured by static tabular or time-series forecasting benchmarks. We introduce TxnQA, an analytical QA benchmark comprising 27 task families (81 templates) alongside a 5-scenario fraud-risk reasoning suite that evaluates both verdicts and supporting evidence over synthetic, privacy-mitigated transaction data. Ground-truth answers are computed programmatically, and model performance is decomposed into format-following rate (FFR), conditional content accuracy (CA), and end-to-end correctness (E2E). Evaluating prompted proprietary baselines and six LoRA-adapted open-weight models shows that adaptation reliably enforces output schemas but leaves arithmetic bottlenecks: Gemma-4-31B-IT and Qwen3-8B attain E2E scores of 0.867 and 0.816, whereas evaluated MoE checkpoints reach only 0.632
    
[^194]: 面向生成式推荐的邻域感知遗忘

    Neighborhood-Aware Unlearning for Generative Recommendation

    [https://arxiv.org/abs/2610.27232](https://arxiv.org/abs/2610.27232)

    本文揭示了生成式推荐中的机器遗忘会因语义标识符的共享而无意识地损害被遗忘物品邻域内的推荐性能，且该退化被全局指标掩盖，并提出邻域感知遗忘（NAU）方法，在交互级遗忘的同时对被遗忘物品邻域的概率质量进行显式控制。

    

    生成式推荐通过自回归方式生成语义标识符（SID）来预测物品，而这些标识符的词元在语义相关的物品之间是共享的。随着这一范式逐渐被采用，如何在其框架下支持机器遗忘已成为一个紧迫的问题。当训练数据中的交互需要被移除时——例如注入的垃圾信息，或用户不再想要的物品的交互——这种共享机制会使遗忘更新在无意中影响到与被遗忘交互相邻的物品的预测结果。我们证明这种影响具有局部结构性，并且会被全局指标所掩盖：遗忘操作可能使整个测试集上的推荐效用几乎保持不变，同时却在被遗忘交互的SID所在邻域上造成显著的性能退化。我们提出了邻域感知遗忘（Neighborhood-Aware Unlearning, NAU），它在交互级别的遗忘基础上，增加了对分配给被遗忘物品邻域的概率质量的显式控制。在三个数据集上的实验……（摘要原文在此处截断）

    arXiv:2610.27232v1 Announce Type: cross  Abstract: Generative recommendation predicts items by autoregressively generating semantic identifiers (SIDs), whose tokens are shared across semantically related items. With this paradigm gaining adoption, the question of how to support unlearning in it has become pressing. If training interactions must later be removed, such as injected spam or interactions with items a user no longer wants, this sharing lets an unlearning update unintentionally affect the predictions for the interactions' neighbors. We show that this effect is locally structured and hidden by global metrics: unlearning can leave recommendation utility on the whole test set nearly unchanged while substantially degrading it on neighborhoods of the forgotten interactions' SIDs. We propose Neighborhood-Aware Unlearning (NAU), which augments interaction-level unlearning with explicit control over the probability mass assigned to the forgotten items' neighborhoods. Across three dat
    
[^195]: 深度灾难性风险感知强化学习

    Deep catastrophic-risk-aware reinforcement learning

    [https://arxiv.org/abs/2610.27214](https://arxiv.org/abs/2610.27214)

    本文将极值理论方法扩展至深度强化学习，提出了一种基于自动微分的可微尾部风险目标，使神经网络策略能够更准确地优化条件风险价值（CVaR），从而更好地应对罕见极端风险。

    

    在强化学习中，随着置信水平的升高，极端风险的估计变得越来越困难，因为只有少量观测值会对条件风险价值（CVaR）的经验估计有所贡献。在本文中，我们将一种用于CVaR优化的极值理论（EVT）方法扩展到高维神经网络策略。利用超阈值表示法和矩估计方法，我们得到了一个可微的尾部风险目标，并证明在选定阈值的条件下，其有限样本梯度可以通过自动微分直接计算。我们首先在一个具有已知最优解的受控实验中验证了该方法，随后在一个包含罕见极端市场波动的深度对冲问题中研究了其性能。我们的结果表明，在正常条件下，EVT训练的策略与经验CVaR训练的策略表现出相似的对冲行为，而EVT训练的策略在应对极端情况时……（原文在此处截断）

    arXiv:2610.27214v1 Announce Type: new  Abstract: Estimating extreme risk in reinforcement learning becomes increasingly difficult as the confidence level increases, since only a small number of observations contribute to empirical estimates of Conditional Value-at-Risk (CVaR). In this paper, we extend an Extreme Value Theory (EVT) approach for CVaR optimization to high-dimensional neural-network policies. Using a peaks-over-threshold representation and method-of-moments estimation, we obtain a differentiable tail-risk objective and show that, conditional on the selected threshold, its finite-sample gradient can be computed directly through automatic differentiation. We first validate the approach in a controlled experiment with a known optimal solution and then investigate its performance in a deep-hedging problem with rare extreme market movements. Our results show similar hedging behavior for EVT- and empirical-CVaR-trained policies under normal conditions, while EVT-trained policies
    
[^196]: 面向视觉-语言模型的环境感知分布外检测的理论分析

    A Theoretical Analysis of Environment-Aware Out-of-Distribution Detection for Vision-Language Models

    [https://arxiv.org/abs/2610.27211](https://arxiv.org/abs/2610.27211)

    该论文基于衡量环境间图像与文本特征对齐程度的“贡献分数”建立了环境感知OOD检测理论，提出了由当前环境主项和先前环境次项（按环境间模内对齐度加权）构成的跨环境最大概念匹配（MCM）评分方法，并从理论上阐明了先前环境何时有益或有害于OOD检测。

    

    持续学习的视觉-语言模型（VLMs）会面对一连串的环境流，同时还必须拒绝那些不属于当前环境的输入。来自先前环境的类别信息可以辅助分布外（OOD）检测，然而现有的检测器对每个环境进行孤立评分，且先前环境在何时有益或有害在理论上尚不明确。我们基于贡献分数（contribution scores）构建了环境感知OOD检测的理论体系，贡献分数是一种成对余弦期望，用于衡量一个环境的图像和文本特征与另一个环境的对齐程度。我们提出了一种跨环境最大概念匹配（MCM）分数，它包含两项：主项将测试图像与当前环境的类别文本进行匹配，次项将测试图像与先前环境的类别文本进行匹配，并以两个环境之间的模内对齐度作为权重。我们证明，在正向环境转移和有界……（摘要截断）的条件下，

    arXiv:2610.27211v1 Announce Type: new  Abstract: Continual vision-language models (VLMs) encounter a stream of environments and must also reject inputs that do not belong to the environment at hand.   Class information from earlier environments can aid out-of-distribution (OOD) detection, yet existing detectors score each environment in isolation, and when previous environments help or hurt is theoretically unclear. We develop a theory of environment-aware OOD detection built on contribution scores, pairwise cosine expectations measuring how the image and text features of one environment align with those of another. We propose a cross-environment maximum-concept-matching (MCM) score with two terms: a primary term matching the test image against the current class texts, and a secondary term matching it against the class texts of a previous environment, weighted by the intra-modal alignment between the two environments. We prove that, under a positive environment transition and a bounded
    
[^197]: 你就是触发器：说话人激活的后门攻击

    You Are the Trigger: Speaker-Activated Backdoors

    [https://arxiv.org/abs/2610.27210](https://arxiv.org/abs/2610.27210)

    提出说话人激活后门（SAB）新威胁模型：攻击者只需在微调阶段植入后门，即可让目标说话人的自然声音成为触发器，通过两阶段微调方法在Whisper-medium和Parakeet上实现90%以上的攻击成功率且误报率极低。

    

    语音模型正日益被用作AI系统的交互接口，涵盖从转录工具到语音智能体等多种应用。现有的音频后门攻击要求攻击者在推理时注入触发器。我们研究了另一种威胁模型：攻击者在微调阶段破坏模型，但无法接触受害者在推理时的输入。我们提出了说话人激活后门（Speaker-Activated Backdoor, SAB）设定，其中目标说话人的自然声音即可充当触发器。我们开发了一种两阶段微调方法，首先使说话人身份特征更易被模型捕捉，然后将目标说话人与攻击者选定的恶意载荷绑定。在相同训练预算下，与单阶段基线相比，两阶段方法显著提升了攻击选择性和干净样本性能，对内容依赖型攻击尤为重要。SAB在Whisper-medium和Parakeet模型上分别实现了90.2%和97.0%的后门攻击成功率，误报率仅为1.7%和0（摘要至此截断）。

    arXiv:2610.27210v1 Announce Type: new  Abstract: Speech models are increasingly used as an interface to AI systems, from transcription tools to voice agents. Existing audio backdoors require an attacker to introduce triggers at inference time. We study a different threat model in which the attacker compromises the model during fine-tuning but has no access to the victim's inference-time inputs. We introduce the Speaker-Activated Backdoor (SAB) setting, in which a target speaker's natural voice serves as the trigger. We develop a two-phase fine-tuning method that first makes speaker identity more accessible and then binds the target speaker to an attacker-chosen payload. Compared with a one-phase baseline with the same training budget, the two-phase method substantially improves selectivity and clean performance, and is especially important for content-dependent attacks. SAB achieves 90.2% and 97.0% backdoor success on Whisper-medium and Parakeet, with false-positive rates of 1.7% and 0
    
[^198]: 基于浅层神经网络的扩散分数估计：正则性、逼近与有限样本率

    Diffusion Score Estimation with Shallow Neural Networks: Regularity, Approximation, and Finite-Sample Rates

    [https://arxiv.org/abs/2610.27206](https://arxiv.org/abs/2610.27206)

    该论文为用浅层 ReLU 网络通过去噪分数匹配估计 Ornstein–Uhlenbeck 扩散分数建立了显式的有限样本理论，给出了时间一致的 Sobolev 正则性界、基于路徑范数的逼近误差控制，以及无界输入下经验 DSM 最小化器的高概率泛化风险界。

    

    我们研究了在 $\mathbb R^d$ 中使用有限宽度浅层 ReLU 网络、通过去噪分数匹配（DSM）来估计 Ornstein–Uhlenbeck 扩散分数的样本复杂度，及其对逆过程采样的意义。在初始密度的归一化导数满足弱可微性和多项式增长条件下，我们将来自高斯转移核和初始密度的互补估计相结合，得到了在每个固定球上关于时间一致的显式 Sobolev 界。将这些界与现有的浅层网络逼近理论相结合，可对逼近误差进行显式控制，并给出充分的路徑范数预算。随后，我们利用截断技术和局部化 Rademacher 复杂度，为路径范数受限的经验 DSM 最小化器（其输入和标签均可无界）建立了高概率风险界。在额外的有界密度和次高斯假设下，适当选择网络的……（摘要在此处截断）

    arXiv:2610.27206v1 Announce Type: cross  Abstract: We study the sample complexity of estimating Ornstein--Uhlenbeck diffusion scores in $\mathbb R^d$ using denoising score matching (DSM) with finite-width shallow ReLU networks, and its implications for reverse-process sampling. Under weak differentiability and polynomial-growth conditions on normalized derivatives of the initial density, we combine complementary estimates from the Gaussian transition kernel and the initial density to obtain explicit Sobolev bounds that are uniform in time on each fixed ball. Combining these bounds with existing shallow-network approximation theory gives explicit control of the approximation error and a sufficient path-norm budget. We then use truncation and localized Rademacher complexity to establish high-probability risk bounds for path-norm-constrained empirical DSM minimizers with unbounded inputs and labels. Under additional bounded-density and sub-Gaussian assumptions, suitable choices of network
    
[^199]: 低残差，错误值：基于物理信息神经网络（PINNs）学习Hamilton-Jacobi-Isaacs方程的粘性解

    Low Residuals, Wrong Value: Learning the Viscosity Solution of Hamilton-Jacobi-Isaacs Equations with PINNs

    [https://arxiv.org/abs/2610.27193](https://arxiv.org/abs/2610.27193)

    本文证明仅凭残差最小化不足以让PINNs学到Hamilton-Jacobi-Isaacs方程的粘性解，因为网络会收敛到残差很小但值函数与粘性解相差 $O(1)$ 的几乎处处解。

    

    物理信息神经网络（PINNs）及其他基于残差训练的求解器通过采样PDE残差的大小来评判候选解。对于一阶Hamilton-Jacobi方程而言，这并不充分：同一方程和边界数据容许与粘性解相差 $O(1)$ 的Lipschitz几乎处处解，而这些解仅在测度为零的集合上违反方程。我们研究在一个固定时界的追逃博弈中，残差最小化是否会选出此类解；该博弈的Hamilton-Jacobi-Isaacs值函数在圆 $S^1$ 以及平坦环面 $\mathbb{T}^{2}$ 和 $\mathbb{T}^{4}$ 上具有已知的闭式表达式。我们在每个几何结构上以十个随机种子训练四种神经参数化方法，共计120次运行。具有周期性Fourier特征的标准网络在全部30次运行中都收敛到折叠的几乎处处解：残差很小、捕获层梯度符合预期，但其值函数在大部分区域却是错误的。

    arXiv:2610.27193v1 Announce Type: cross  Abstract: Physics-informed neural networks (PINNs) and other residual-trained solvers judge a candidate solution by the size of its sampled PDE residual. For first-order Hamilton-Jacobi equations this is not sufficient: the same equation and boundary data admit Lipschitz almost-everywhere solutions that differ from the viscosity solution by $O(1)$ while violating the equation only on sets of measure zero. We study whether residual minimization selects such solutions in a fixed-horizon pursuit-evasion game whose Hamilton-Jacobi-Isaacs value function is known in closed form on the circle $S^1$ and on the flat tori $\mathbb{T}^{2}$ and $\mathbb{T}^{4}$. We train four neural parameterizations with ten seeds on each geometry, 120 runs in total. A standard network with periodic Fourier features converges to folded almost-everywhere solutions in all 30 runs: small residuals and the expected capture-layer gradient, yet values that are wrong over most of
    
[^200]: 论任务组合下的脆弱性：语言模型中的隐式预算分配

    On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models

    [https://arxiv.org/abs/2610.27190](https://arxiv.org/abs/2610.27190)

    语言模型存在一种隐式思考预算：在组合多个任务时，每个任务分配到的 token 数远少于单独处理时，导致即使模型能独立可靠地完成每个任务，组合后错误率仍显著上升，且需要长推理链的任务在组合下尤为脆弱。

    

    现代语言模型越来越多地被要求在单个提示中处理多个任务。这需要模型正确解决每个任务，同时将各任务的结果整合成一个连贯的回复。本文识别了这一过程中的一个根本性局限：当任务被组合在一起时，模型会犯下多得多的错误，即使它们能够可靠地独立完成每一个组成任务。这种性能退化无法完全用独立错误的累积来解释。通过对开放权重模型以及前沿模型 Claude Opus 5 的研究，我们的发现指向了这一现象背后的一种隐式思考预算。具体而言，在任务组合的情况下，模型在每个任务上使用的 token 数量远少于单独处理时，即使完整序列仍远在上下文窗口范围之内。因此，那些单独求解时需要更长推理链条的任务在组合情况下会更加“脆弱”，即模型……

    arXiv:2610.27190v1 Announce Type: cross  Abstract: Modern language models are increasingly asked to handle several tasks within a single prompt. Doing so requires solving each task correctly while integrating the resulting answers into a coherent response. In this paper, we identify a fundamental limitation in this process, where models make substantially more errors when tasks are composed, even when they can solve every constituent task reliably in isolation. This degradation cannot be fully explained by the accumulation of independent errors. Across both open-weight models and the frontier model Claude Opus 5, our findings instead point to an implicit thinking budget underlying this effect. In particular, under composition, models use substantially fewer tokens per task than in isolation, even when the full sequence remains well within the context window. Tasks that require longer solution traces in isolation are consequently more $\textit{fragile}$ under composition, i.e., models m
    
[^201]: Meta-iLaD：通过动力学环境的元学习实现可辨识的潜在动力学

    Meta-iLaD: Identifiable Latent Dynamics via Meta-Learning of Dynamics Environments

    [https://arxiv.org/abs/2610.27180](https://arxiv.org/abs/2610.27180)

    Meta-iLaD 通过对动力学环境进行元学习，在少样本条件下学习可辨识的潜在动力学，从而实现对分布外环境中高维时间序列的准确预测。

    

    学习潜在动力学对于评估高维时间序列的当前状态以及预测其未来轨迹至关重要。针对局部平稳的潜在动力学 $\mathcal{F}(\mathbf{z}_{< t};\mathbf{c})$，本文提出不限制 $\mathbf{c}$ 的维度或其调制 $\mathcal{F}$ 的方式。在合成数据与真实数据上，作者提供了强有力的实证证据表明：1）以少样本示例为条件能够实现对分布外环境的泛化；2）$\mathbf{c}$ 与 $\mathcal{F}$ 的可辨识性对于实现超越观测轨迹重构的准确预测至关重要。代码已发布于 https://github.com/GoForIt-YYB/Meta-iLaD。

    arXiv:2610.27180v1 Announce Type: new  Abstract: Learning latent dynamics is central to assessing the current states and forecasting the future trajectories of high-dimensional time series. For locally-stationary latent dynamics $\mathcal{F}(\mathbf{z}_{< t};\mathbf{c})$ without restricting the dimension of $\mathbf{c}$ or how it modulates $\mathcal{F}$. On synthetic and real data, we provide strong empirical evidence that 1) conditioning on few-shot examples enables generalization to out-of-distribution environments, and 2) identifiability for $\mathbf{c}$ and $\mathcal{F}$ is critical for accurate forecasting beyond reconstructing observed trajectories. Code is available at https://github.com/GoForIt-YYB/Meta-iLaD.
    
[^202]: 通过计算复杂性分离理解脚手架的益处

    Understanding the Benefits of Scaffolds through Computational Separations

    [https://arxiv.org/abs/2610.27157](https://arxiv.org/abs/2610.27157)

    本文证明推理时脚手架即使基于平凡的 $\AC^0$ 原语，也能通过多次调用完成 $\NL$-完全的闭包计算，而固定深度的前向传播在标准复杂性猜想下无法表示该计算，从而解释了为何脚手架带来的能力提升无法通过蒸馏完全内化到模型之中。

    

    推理时框架，即多次调用模型并在各次调用之间保持状态的外部程序，能够显著提升模型能力。本文研究了使用此类框架的益处，以及这些益处在多大程度上可以通过训练被内化。更准确地说，我们将框架 $\mathcal{A}$ 概念化为围绕基础模型 $M$ 的任意包装器，并尝试从框架的运行轨迹中蒸馏出一个学生模型。实验上我们发现，即使使用简单的脚手架和较小的模型，推理时框架也显著优于蒸馏得到的学生模型。随后我们证明，这一差距源于一个根本性的表达能力限制：即使每次调用的原语只是平凡的 $\AC^0$ 查找，推理时框架也能够计算出 $\NL$-完全的闭包；而在被广泛相信的猜想（关于 $\NL$ 与固定深度电路的关系）下，任何固定深度的前向传播都无法表示这样的计算。

    arXiv:2610.27157v1 Announce Type: cross  Abstract: Inference-time harnesses, i.e., external procedures that call a model many times and keep state across calls, can substantially improve model capabilities. In this paper, we investigate the benefits of using such harnesses and the extent to which these benefits can be internalized through training. More precisely, we conceptualize a harness $\mathcal{A}$ as an arbitrary wrapper around a base model $M$ and attempt to distill a student model from the harness's traces. Empirically we find that, even with simple scaffolds and small models, the inference-time harness significantly outperforms a distilled student. We then prove that such a gap is due to a fundamental representational limitation: inference-time harnesses can compute a closure of their per-call primitive that is $\NL$-complete even when the primitive is a trivial $\AC^0$ lookup, and that no fixed-depth forward pass can represent under the widely believed conjecture that $\NL\n
    
[^203]: 一种用于鲁棒多时间尺度光伏功率预测的TFT-NBEATS混合框架

    A Hybrid TFT-NBEATS Framework for Robust Multi-Horizon Photovoltaic Power Forecasting

    [https://arxiv.org/abs/2610.27155](https://arxiv.org/abs/2610.27155)

    提出了一种将TFT与N-BEATS相结合并通过自适应注意力机制动态融合的混合深度学习框架，能够同时捕捉短期波动与长期趋势，实现具有可解释性的鲁棒多时间尺度光伏功率预测。

    

    准确的多时间尺度光伏功率预测对电网稳定性、能量调度以及光伏-电池-电网系统的高效运行至关重要。然而，现有模型往往无法同时捕捉短期波动性和长期时间模式，并且缺乏统一的多时间尺度预测能力和可解释性。为解决这些局限性，本研究提出了一种新颖的混合深度学习框架，该框架在基于注意力的融合架构中集成了时序融合Transformer（TFT）和N-BEATS。所提出的模型利用TFT学习短期时间依赖关系和动态特征重要性，同时利用N-BEATS通过可解释的基函数分解捕捉长期趋势和季节性成分。自适应注意力机制动态融合两个分支，实现平衡的多尺度学习。该框架可同时生成30分钟、2小时、6小时、12小时等多个时间尺度的预测。（注：原文摘要在此处截断）

    arXiv:2610.27155v1 Announce Type: new  Abstract: Accurate multi-horizon photovoltaic (PV) power forecasting is essential for grid stability, energy scheduling, and the efficient operation of PVbatterygrid systems. However, existing models often fail to jointly capture short-term variability and long-term temporal patterns, while lacking unified multi-horizon capability and interpretability. To address these limitations, this study proposes a novel hybrid deep learning framework that integrates a Temporal Fusion Transformer (TFT) and N-BEATS within an attention-based fusion architecture. The proposed model leverages TFT to learn short-term temporal dependencies and dynamic feature importance, while N-BEATS captures long-term trend and seasonal components through interpretable basis decomposition. An adaptive attention mechanism dynamically fuses both branches, enabling balanced multi-scale learning. The framework simultaneously generates forecasts for 30-minute, 2-hour, 6-hour, 12-hour,
    
[^204]: 阿尔茨海默病与帕金森病中基于影像的联合形态测量标志物的跨疾病学习

    Cross-disorder Learning of Imaging-derived Joint Mophometric Markers in Alzheimer's and Parkinson's Diseases

    [https://arxiv.org/abs/2610.27150](https://arxiv.org/abs/2610.27150)

    本研究提出一种显式、可解释且空间自适应的迁移学习框架，利用ADNI和PPMI神经影像数据集揭示阿尔茨海默病与帕金森病共享的形态测量生物标志物及与临床表现相符的神经解剖学改变模式。

    

    作为两种最为流行的神经退行性疾病（NDDs），阿尔茨海默病（AD）和帕金森病（PD）在症状学和病理生理学方面存在共同之处。然而，这两种不同的疾病在多大程度上以及以何种方式重叠，目前仍未得到充分探索。在本研究中，我们提出了一个显式、可解释且空间自适应的迁移学习框架，以阐明与AD和PD相关的形态测量生物标志物。我们将这种高维机器学习模型应用于两个最流行的开源神经影像数据集，即阿尔茨海默病神经影像计划（ADNI）和帕金森病进展标志物计划（PPMI）。一方面，我们的跨疾病分类器所揭示的神经解剖学改变模式与先前的神经影像研究结果高度一致，并突出了患者的临床表现。另一方面，使用共享先验训练的空间自适应分类器……（原文摘要在此处被截断）

    arXiv:2610.27150v1 Announce Type: cross  Abstract: As two most prevalent neurodegenerative diseases (NDDs), Alzheimer's disease (AD) and Parkinson's disease (PD) shared commonalities both in symptomology and pathophysiology. However, to what extent and in what way these two distinct disorders overlap remain underexplored. In this study, we propose an explicit, interpretable and spatially adaptive transfer learning framework to elucidate the morphometric biomarkers linked to AD and PD. We applied such high-dimensional machine learning model to two most popular open-source neuroimaging datasets, i.e., Alzheimer's Disease Neuroimaging Initiative (ADNI) and Parkinson's Progression Markers Initiative (PPMI). On the one hand, our cross-disorder classifier revealed neuro-anatomical alteration patterns well aligning with previous neuroimaging research findings and underscoring patients' clinical presentations. On the other hand, the spatially adaptive trained classifier using shared prior achi
    
[^205]: ICLHerd：基于Herding的大语言模型数据选择方法，以上下文学习作为代理

    ICLHerd: Herding-Based Data Selection for LLMs with In-Context Learning as a Proxy

    [https://arxiv.org/abs/2610.27142](https://arxiv.org/abs/2610.27142)

    ICLHerd提出了一种基于von Mises-Fisher核的数据选择框架，通过平衡模式寻求与模式平均来克服传统最大相似度方法的聚类偏差，并利用上下文学习作为验证代理实现高效的超参数搜索，在大语言模型训练数据选择上优于现有算法。

    

    数据选择在现代大语言模型训练中扮演着核心角色，因为训练数据的质量强烈影响下游性能。一种常见的方法是策划一个编码了所需技能（例如推理或多语言能力）的开发集，并选择与这些技能相似的数据。现有方法使用模型隐藏状态或梯度上的余弦距离来计算相似度。然而，我们观察到这种最大相似度策略会引发模式寻求偏差：所选数据紧密聚集在开发样本周围。为了解决这个问题，我们提出了ICLHerd，这是一个数据选择框架，利用von Mises-Fisher核来平衡模式寻求与模式平均。我们进一步引入上下文学习作为验证代理，从而能够在平衡权衡上进行高效的超参数搜索。结果表明，ICLHerd优于现有的大语言模型数据选择算法，凸显了……

    arXiv:2610.27142v1 Announce Type: cross  Abstract: Data selection plays a central role in modern large language model training, as the quality of training data strongly influences downstream performance. A common approach is to curate a development set that encodes the desired skills (e.g., reasoning or multilinguality) and select data that are similar to them. Existing methods compute similarity using cosine distance over either the model's hidden states or gradients. However, we observe that this max-similarity strategy induces a mode-seeking bias: the selected data clusters tightly around the development samples. To address this, we propose ICLHerd, a data selection framework that leverages the von Mises-Fisher kernel to balance mode-seeking with mode-averaging. We further introduce in-context learning as a validation proxy, enabling efficient hyperparameter search over the balance trade-off. Results show that ICLHerd outperforms existing LLM data selection algorithms, highlighting 
    
[^206]: 面向在轨服务的传感器化仿壁虎粘附界面交互状态识别研究

    Towards Interaction State Recognition in Sensorized Gecko-Inspired Adhesive Interfaces for Orbital Servicing

    [https://arxiv.org/abs/2610.27141](https://arxiv.org/abs/2610.27141)

    该论文提出了一个基于嵌入式电容传感的实时交互感知框架，利用轻量级机器学习模型对仿壁虎粘附界面在粘附全生命周期中的五种交互状态进行分类识别，填补了粘附捕获技术在闭环状态感知方面的空白。

    

    可靠的接触建立仍然是自主在轨服务（OOS）的一个关键挑战，特别是在与非合作目标交互时。仿壁虎干粘附材料为太空机器人捕获提供了一种轻量、被动且无残留的解决方案，然而现有系统以开环方式运行，无法在交互过程中估计自身的接触或粘附状态。本文提出了一种实时交互感知框架，使传感器化的仿壁虎粘附界面能够利用嵌入式电容传感在粘附全生命周期中估计其接触状态。基于实验采集和增强的传感器数据，对五种离散交互状态（空闲、接近、接触、接合和脱离）进行分类，并对六种适合嵌入式部署的轻量级机器学习模型进行了比较评估。在所评估的模型中，随机森林取得了……（摘要内容在此处截断）

    arXiv:2610.27141v1 Announce Type: cross  Abstract: Reliable contact establishment remains a critical challenge for autonomous on-orbit servicing (OOS), particularly when interacting with non-cooperative targets. Gecko-inspired dry adhesives provide a lightweight, passive, and residue-free solution for robotic capture in space, yet current systems operate in open loop, without the ability to estimate their own contact or adhesion state during interaction. This paper presents a real-time interaction-awareness framework that enables sensorized gecko-inspired adhesive interfaces to estimate their contact state throughout the adhesion lifecycle using embedded capacitive sensing. Five discrete interaction states (idle, proximity, contact, engaged, and disengaged) are classified using experimentally collected and augmented sensor data, enabling a comparative evaluation of six lightweight machine learning models suitable for embedded deployment. Among the evaluated models, Random Forest achiev
    
[^207]: 通过表征短期运动揭示动物身份与学习进度

    Representing short-term movement to reveal animal identity and learning progress

    [https://arxiv.org/abs/2610.27140](https://arxiv.org/abs/2610.27140)

    本文提出一种基于“行为特征在相邻时间区间内相对稳定”这一假设的自监督表征学习框架，以时间邻近性为监督信号，直接从动物短期运动轨迹中学习紧凑表征，无需人工定义行为特征即可揭示动物个体身份与学习进度。

    

    学习如何随时间推移而进展、并在个体之间存在差异，是认知科学与行为科学中一个由来已久的问题。然而，一个主要挑战在于找到能够忠实反映学习进程的可观测行为指标。现有方法通常依赖于预先定义的行为特征或任务层面的表现，而对于自由活动动物的细粒度姿态与运动仍缺乏充分探索。在本文中，我们提出这样一个问题：短期运动轨迹能否揭示学习进度与个体身份？我们提出了一个基于简单假设的自监督表征学习框架：与身份和学习相关的行为特征在相邻时间区间内保持相对稳定。利用时间邻近性作为监督信号，该框架可直接从运动轨迹中学习紧凑的表征，无需人工定义行为状态或运动学特征。通过使用……（原文摘要到此被截断）

    arXiv:2610.27140v1 Announce Type: cross  Abstract: How learning progresses over time and varies across individuals is a longstanding question in cognitive and behavioral science. However, a major challenge is identifying observable behavioral measures that faithfully capture learning progression. Existing approaches commonly rely on predefined behavioral features or task-level performance, while fine-scale posture and movement in freely behaving animals remain less explored. In this paper, we ask: can short-term movement trajectories reveal learning progress and individual identity? We introduce a self-supervised representation learning framework built on a simple hypothesis: behavioral characteristics associated with identity and learning remain relatively stable over nearby time intervals. Using temporal proximity as supervision, our framework learns compact representations directly from movement trajectories without manually defining behavioral states or kinematic features. Using be
    
[^208]: 通过预测恢复本体状态

    Ontic state recovery through prediction

    [https://arxiv.org/abs/2610.27139](https://arxiv.org/abs/2610.27139)

    本文将状态定义为指定查询全域上的完整响应档案，并从查询歧义性、状态空间几何和预测目标选择三个角度，结合合成系统与神经记录数据，考察“预测”与“本体状态恢复”之间的关系。

    

    我们通过系统的响应来接触一个系统，而其状态所关涉的是系统本身是怎样的。对更多响应的预测有望带来超越直接观测的知识。这些知识能告诉我们关于系统状态的什么，取决于我们最初如何定义“状态”。我们为这一概念赋予一种模型相对的含义：状态是在指定查询全域上的完整响应档案，并且在所有查询上都一致的配置被视为同一状态。从这一视角出发，一个被学习到的状态是通过它所保留的响应档案之间的区分来理解的。预测关注的是被挑选的观测，而恢复关注的是响应档案的整体。我们通过查询歧义性、状态空间几何结构以及预测目标的选择来考察二者的关系。具有已知响应规律的合成系统与神经记录为本研究提供了互补的研究场景。本文旨在阐明预测如何……

    arXiv:2610.27139v1 Announce Type: cross  Abstract: We encounter a system through its responses, yet its state concerns how the system is. Predicting further responses promises knowledge beyond what is immediately observed. What that knowledge tells us about the system's state depends on what we mean by a state in the first place. We give this notion a model-relative meaning: a state is the complete response profile over a specified query universe, and configurations agreeing on every query are regarded as the same state. From this perspective, a learned state is understood through the distinctions between these profiles that it retains. Prediction concerns selected observations; recovery concerns the profile as a whole. We examine their relationship through query ambiguity, state-space geometry, and the choice of prediction targets. Synthetic systems with known response laws and neural recordings provide complementary settings for this investigation. The aim is to clarify how predictio
    
[^209]: 基于第一性原理的机器学习分子动力学研究行星包层中的氨-氢相平衡

    Ammonia-Hydrogen Phase Equilibria in Planetary Envelopes from First-Principles-Based Machine-Learning Molecular Dynamics

    [https://arxiv.org/abs/2610.27136](https://arxiv.org/abs/2610.27136)

    本研究利用基于密度泛函理论训练的机器学习原子间势进行大规模两相共存模拟，绘制了行星包层温压条件下氨-氢体系的相平衡图，发现临界温度随压力从500 K升至2000 K并在约30 GPa、2000 K以上趋于平坦，对应分子开始解离为原子。

    

    氨和氢是太阳系内外行星的主要组成成分，但它们在行星包层中如何相互作用却鲜为人知。我们基于密度泛函理论计算训练的机器学习原子间势，在500至4000 K、0.1至100 GPa的温压范围内，对氨-氢体系进行了大规模的两相共存模拟。由此我们绘制了共存的富氢相与富氨相的成分，以及区分完全混溶条件与相分离条件的临界曲线。临界温度随压力升高而上升，从0.1 GPa下的500 K增至约30 GPa附近的2000 K；我们发现在3000 K和4000 K时，直到100 GPa均未发生相分离。临界曲线的斜率在0.3至30 GPa之间显著减小，并在约30 GPa和2000 K以上几乎变得平坦，此时分子开始让位于原子物种。共存曲线（原文摘要在此处截断）

    arXiv:2610.27136v1 Announce Type: cross  Abstract: Ammonia and hydrogen are major constituents of planets within and beyond the solar system, yet how they interact in planetary envelopes is poorly known. We perform large-scale two-phase coexistence simulations of the ammonia-hydrogen system from 500 to 4000 K and 0.1 to 100 GPa, driven by machine-learning interatomic potentials trained on density functional theory calculations. From these we map the compositions of coexisting hydrogen-rich and ammonia-rich phases, and the critical curve that separates conditions of complete miscibility from those of phase separation. The critical temperature increases with pressure, from 500 K at 0.1 GPa to 2000 K near 30 GPa, and we find no phase separation at 3000 and 4000 K up to 100 GPa. The slope of the critical curve decreases substantially between 0.3 and 30 GPa, and the curve nearly flattens above about 30 GPa and 2000 K, where molecules begin to give way to atomic species. The coexistence curv
    
[^210]: 面向高速流动控制的“玻璃箱”深度强化学习

    Glass-Box Deep Reinforcement Learning for High-Speed Flow Control

    [https://arxiv.org/abs/2610.27132](https://arxiv.org/abs/2610.27132)

    本文是一篇展望性文章，系统分析了深度强化学习应用于高速流动控制所面临的关键瓶颈（仿真成本与保真度、数据效率、极端环境感知、奖励设计、可解释性及跨马赫数与雷诺数泛化），并主张发展可解释的“玻璃箱”DRL方法以推动其实验落地。

    

    深度强化学习（DRL）正在挑战主动流动控制领域长期以来的固有假设。通过直接从与流动的交互中学习，DRL智能体无需显式模型即可为非线性、非定常流动找到驱动（作动）策略，并且常常利用经典控制方法难以预见的机制。高速流动是对这一前景的严苛考验：激波、激波/边界层干扰、转捩、气动热载荷以及微秒尺度的动力学特性，对算法和硬件均构成巨大压力。然而，现有的大多数演示仍停留在理想化的低速仿真中，在鲁棒性、可解释性、可迁移性或实验部署方面证据有限。在这篇展望性文章中，我们分析了制约基于DRL的高速流动控制发展的因素，包括可压缩仿真的成本与保真度、数据效率、极端环境下的传感感知、奖励设计、可解释性，以及跨马赫数和雷诺数的泛化能力。我们认为……

    arXiv:2610.27132v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) is challenging long-held assumptions in active flow control. By learning directly from interaction with a flow, DRL agents can find actuation strategies for nonlinear, unsteady flows without explicit models, often exploiting mechanisms that classical control would not anticipate. High-speed flows are a demanding test of this promise: shocks, shock/boundary-layer interactions, transition, aerothermal loading and microsecond-scale dynamics strain both algorithms and hardware. Yet most demonstrations remain idealized, low-speed simulations with limited evidence of robustness, interpretability, transferability or experimental deployment. In this perspective, we examine what constrains DRL-based high-speed flow control, including compressible simulation cost and fidelity, data efficiency, sensing in extreme environments, reward design, explainability, and generalization across Mach and Reynolds number. We arg
    
[^211]: 面向混合专家语言模型的重建优化专家剪枝

    Reconstruction-Optimized Expert Pruning for Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2610.27123](https://arxiv.org/abs/2610.27123)

    该论文提出OptEP方法，通过可扩展的连续松弛直接优化可微分专家选择变量，并结合轻量级基于重建的权重更新，将MoE模型专家剪枝问题转化为子集选择问题求解，在Qwen3-30B-A3B上取得最高15.05%的绝对准确率提升。

    

    混合专家模型通过稀疏的专家激活来减少计算量，但部署时仍需存储大量专家参数。我们研究了使用任务领域校准数据的训练后专家剪枝问题，并将其建模为一个子集选择问题：给定一个预训练的MoE层，选择固定数量的专家，使其最能重建该层的原始输出。精确的子集搜索是一个组合问题，对于每层包含大量专家的现代MoE模型而言变得难以处理。我们提出了OptEP，一种可扩展的连续松弛方法，直接优化可微分的专家选择变量，随后对保留的专家权重进行轻量级的基于重建的更新。在跨越数学推理和代码生成等多个MoE架构和任务上，我们的方法优于现有最强的专家剪枝基线，在Qwen3-30B-A3B上绝对准确率提升高达15.05%，在Qwen3-235B-A22B上提升14.55%。

    arXiv:2610.27123v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) models reduce computation through sparse expert activation, but deployment still requires storing a large number of expert parameters. We study post-training expert pruning using task-domain calibration data as a subset-selection problem: given a pretrained MoE layer, select a fixed number of experts that best reconstruct its original output. Exact subset search is combinatorial and becomes intractable for modern MoE models with many experts per layer. We develop OptEP, a scalable continuous relaxation that directly optimizes differentiable expert-selection variables, followed by a lightweight reconstruction-based update of the retained expert weights. Across multiple MoE architectures and tasks spanning mathematical reasoning and code generation, our approach improves over the strongest existing expert-pruning baselines, with absolute accuracy gains of up to 15.05% on Qwen3-30B-A3B, 14.55% on Qwen3-235B-A22B, 
    
[^212]: RGH-GNN：用于金融欺诈检测的风险门控异构图神经网络

    RGH-GNN : Risk-Gated Heterogeneous Graph Neural Network for Financial Fraud Detection

    [https://arxiv.org/abs/2610.27107](https://arxiv.org/abs/2610.27107)

    该论文提出了一种风险门控异构图神经网络RGH-GNN，通过显式建模账户、设备等交易间的关系模式，并避免标准图神经网络对邻域信息的均匀聚合稀释稀少的欺诈信号，从而在类别不平衡和欺诈伪装的场景下提升金融欺诈检测效果。

    

    信用卡欺诈是金融机构面临的重大问题，每年造成巨额的经济损失。欺诈模式和方法不断演变，欺诈交易常常伪装成合法活动。现实世界中用于欺诈检测的交易数据受到严重的类别不平衡影响，这使得检测欺诈交易变得更加困难。欺诈交易很少是孤立发生的。账户、设备和其他共同属性常常在欺诈活动中被共享，从而揭示出仅凭个体交易特征无法捕捉的关系模式。基于图的方法能够显式地建模这些关系。然而，标准的图神经网络会均匀地聚合邻域信息，这可能稀释欺诈交易所携带的本已稀缺的信号，并限制对伪装欺诈的检测。本工作提出了RGH-GNN，一种异构图神经网络，

    arXiv:2610.27107v1 Announce Type: new  Abstract: Credit card fraud is a significant problem for financial institutions, resulting in substantial monetary loss each year. Fraud patterns and methods are continuously evolving, with fraudulent transactions often disguised to resemble legitimate activity. Real world transaction data for fraud detection is severely affected by class imbalance making it harder to detect these transactions. Fraudulent transactions are rarely isolated. Accounts, devices, and other common attributes are often shared across fraudulent activity, revealing relational patterns that individual transaction features alone cannot capture. Graph-based approaches model these relationships explicitly. However, standard graph neural networks aggregate neighbourhood information uniformly, which can dilute the already scarce signal carried by fraudulent transactions and limit detection of camouflaged fraud. This work presents RGH-GNN, a heterogeneous graph neural network that
    
[^213]: 这就是SParTA：一种用于喷注标记的模块化、高效且可扩展的Transformer架构

    This is SParTA: A modular, efficient and scalable transformer architecture for jet-tagging

    [https://arxiv.org/abs/2610.27106](https://arxiv.org/abs/2610.27106)

    SParTA通过利用喷注组分陡峭下降的横向动量谱，仅让最硬的N个组分参与二次复杂度的相互作用注意力，其余软组分通过线性交叉注意力整合，在保留软辐射和色流信息的同时避免了全自注意力的O(l²)内存瓶颈，实现了模块化、高效且可扩展的喷注标记。

    

    我们提出了稀疏粒子相互作用Transformer架构（SParTA），这是一种用于高能对撞机喷注标记的模块化、可扩展且高效的Transformer。利用喷注组分陡峭下降的横向动量（$p_T$）谱，SParTA将组分处理分解为两种拓扑模式。按$p_T$降序排列后，仅有$N$个最硬的组分（硬核心）参与由成对物理特征和可学习关系特征驱动的二次复杂度$O(N^2)$微分相互作用注意力机制。其余的软组分则通过与硬核心的线性复杂度$O(lN)$交叉注意力被整合进来，在保留软辐射和色流信息的同时，避免了对所有$l$个组分进行完全自注意力所带来的$O(l^2)$内存瓶颈。成对相互作用特征在网络内由组分四动量动态计算，并以全局喷注级亚结构观测量为条件。

    arXiv:2610.27106v1 Announce Type: cross  Abstract: We present the Sparse Particle-interaction Transformer Architecture (SParTA), a modular, scalable, and efficient transformer for jet tagging at high-energy colliders. Exploiting the steeply falling transverse-momentum ($p_T$) spectrum of jet constituents, SParTA factorizes constituent processing into two topological regimes. Ordered by descending $p_T$, only the $N$ hardest constituents (the hard core) participate in quadratic $O(N^2)$ differential interaction attention driven by pairwise physical and learned relational features. The remaining soft constituents are integrated via linear $O(lN)$ cross-attention to the hard core, preserving soft-radiation and color-flow information while avoiding the $O(l^2)$ memory bottleneck of full self-attention over all $l$ constituents. Pairwise interaction features are computed dynamically in-network from constituent four-momenta and conditioned on global jet-level substructure observables. On the
    
[^214]: LOB冲击基准测试：生成式限价订单簿模型中涌现的市场冲击

    LOB-Impact Benchmark: Emergent Market Impact in Generative Limit Order Book Models

    [https://arxiv.org/abs/2610.27100](https://arxiv.org/abs/2610.27100)

    该论文提出LOB-Impact基准，首次证明市场冲击现象能够在仅通过下一词元预测训练的神经限价订单簿生成模型中自然涌现，并通过注入元订单与实证冲击估计器为消息级生成器提供了反事实市场冲击的评估框架。

    

    真实的市场冲击对于生成式限价订单簿（LOB）模型充当交易智能体的世界模型至关重要。如果模拟器对这一机制的定价有误，智能体将学到在模拟器中最优、但在真实市场中却是错误的策略。现有指标仅通过与历史数据进行分布比较来评估生成的数据流，而对智能体自身订单的反事实响应则基本未被探索。在本研究中，我们证明了市场冲击能够在仅通过下一词元预测训练的神经LOB生成器中自然涌现。我们向五个自回归模型在五只纳斯达克股票上生成的消息流中注入执行风格的元订单，并使用来自实证冲击文献的估计器来测量响应，这些估计器已在物理特性事先可知的基线上得到验证。我们的基准通过三个阶段评估任何消息级生成器：自由生成（对照价格动态检验订单流持续性……

    arXiv:2610.27100v1 Announce Type: cross  Abstract: Realistic market impact is essential for generative limit order book (LOB) models to serve as world models for trading agents. If a simulator misprices this mechanism, agents learn policies that are optimal in the simulator but wrong in the real market. Existing metrics assess generated streams purely through distributional comparison with historical data, leaving counterfactual response to an agent's own orders largely unexplored. Here, we show that market impact emerges naturally in neural LOB generators trained solely for next-token prediction. We inject execution-style metaorders into the generated message flow of five autoregressive models across five NASDAQ stocks and measure the response using estimators from the empirical impact literature, validated on baselines whose physics is known in advance. Our benchmark evaluates any message-level generator in three phases: free generation (reading order-flow persistence against price d
    
[^215]: ClockMark：面向大语言模型的自适应与宿主相关水印技术

    ClockMark: Adaptive and Host-Dependent Watermarking for Large Language Models

    [https://arxiv.org/abs/2610.27079](https://arxiv.org/abs/2610.27079)

    提出了ClockMark水印框架，首次基于“自适应嵌入”（保留先前采样决策状态）与“宿主相关嵌入”（利用当前LLM分布约束词元选择和状态更新）两大原则，在保持输出分布的同时实现对文本编辑的鲁棒水印嵌入。

    

    大语言模型（LLM）水印技术在生成过程中不断演化的下一词元分布上进行操作。在流式生成中，每个采样得到的词元都会成为文本的一部分，并影响后续的分布。因此，嵌入过程是在不断变化的宿主上进行的，这与经典水印技术不同——经典水印技术是将嵌入适配到已存在的宿主信号上。为了适应这一差异，我们基于两个原则为LLM水印构建了一种新颖的嵌入框架。我们首先论证嵌入应当是自适应的，即保留先前采样决策的状态，以更好地指导后续的选择。我们进一步论证嵌入应当是宿主相关的，即利用当前的LLM分布来同时约束词元的选择和状态的更新。基于这些原则，并遵循分布保持和抗文本编辑鲁棒性这两项要求，我们提出了ClockMark。它将自适应与宿主相关的嵌入与消息相结合……

    arXiv:2610.27079v1 Announce Type: cross  Abstract: Large language model (LLM) watermarking operates on the next-token distribution that evolves during generation. In streaming generation, each sampled token becomes part of the text and affects subsequent distributions. Embedding therefore proceeds with a changing host, unlike classical watermarking, which adapts embedding to an existing host signal. To accommodate this difference, we formulate a novel embedding framework for LLM watermarking based on two principles. We first argue that embedding should be adaptive, retaining the state of previous sampling decisions to better guide subsequent choices. We further argue that embedding should be host-dependent, using the current LLM distribution to constrain both token selection and state updates. Guided by these principles and the two requirements of distribution preservation and robustness to text editing, we propose ClockMark. It combines adaptive and host-dependent embedding with messa
    
[^216]: LEAF：面向FPGA的查找表感知高效注意力机制

    LEAF: LUT-aware Efficient Attention on FPGAs

    [https://arxiv.org/abs/2610.27078](https://arxiv.org/abs/2610.27078)

    LEAF是首个面向FPGA的纳秒级查找表感知transformer，通过将注意力块中的常量矩阵-向量乘法制表化，实现了满足实时系统要求的超低延迟推理。

    

    我们提出了面向FPGA的查找表（LUT）感知高效注意力机制（LEAF）。LEAF是首个纳秒级、查找表感知的FPGA transformer。部署既要足够大以保证良好性能、又要足够小以适配FPGA、还要足够快以满足CERN等实时系统所要求的亚微秒时间尺度的transformer仍然十分困难，因为赋予这些模型高表达能力的注意力机制同时也使其在硬件实现上成本高昂。LEAF通过查找表感知训练，将查询、键、值投影以及最终输出的前馈层转换为FPGA原生层，创建了首个为超低延迟推理而构建的查找表感知transformer和linformer层。注意力机制之所以难以进行查找表感知训练，是因为该方法需要常量权重来进行制表，而注意力机制涉及两个激活值之间的相乘；LEAF对注意力块中的常量矩阵-向量乘法进行制表，从而解决了这一问题。

    arXiv:2610.27078v1 Announce Type: new  Abstract: We introduce Lookup-table (LUT)-aware Efficient Attention on FPGAs (LEAF). LEAF is the first nanosecond-scale LUT-aware transformer for FPGAs. Deploying transformers that are large enough to perform well, small enough to fit on an FPGA, and fast enough to meet the sub-microsecond timescales required by real-time systems such as those at CERN remains difficult, since the same attention mechanism that gives these models their high expressivity also makes them costly to implement in hardware. LEAF converts the query, key, and value projections and the final output feed-forward layer into FPGA-native layers with LUT-aware training, creating the first LUT-aware transformer and linformer layers built for ultra-low-latency inference. Attention resists LUT-aware training because it multiplies two activations where the method needs a constant weight to tabulate; LEAF tabulates the attention block's constant matrix-vector multiplications and leave
    
[^217]: 图神经网络在天体物理学和宇宙学中的应用

    Graph Neural Networks for Astrophysics and Cosmology

    [https://arxiv.org/abs/2610.27051](https://arxiv.org/abs/2610.27051)

    本综述论证了图神经网络（GNNs）因其契合物理数据的固有几何结构、能够融入已确立的对称性与先验知识、且比其他深度学习架构更具可解释性，因而比任何替代方法都更适合应用于天体物理学和宇宙学的研究。

    

    当仅孤立地考虑物体时，物理学很少能取得突破。相反，我们对物体与其邻近物体之间的关系进行建模和测量，无论是空间的、时间的还是其他方面的：星系与其所处环境的关系、暗物质晕与其演化历史的关系、探测器中光子撞击之间的关系。图（Graph）正是编码这些关系的理想工具，而图神经网络（GNNs）则学习尊重这些关系的几何函数。本综述介绍了GNN在天体物理学和宇宙学中的应用，并论证了GNN通常比任何其他替代方法都更适合物理学和物理数据：它们契合领域与信号本身的固有几何结构，能够融入我们已充分信任并付出辛勤努力构建的对称性和先验知识，同时保持足够的灵活性以捕捉我们尚未理解的现象。GNN在物理上直观的工作方式也使其比其他深度学习架构更具可解释性。我们展示了如何从一般原理出发（摘要至此截断，内容不完整）……

    arXiv:2610.27051v1 Announce Type: cross  Abstract: Physics seldom makes breakthroughs when we consider objects only in isolation. Instead we model and measure how objects sit among their neighbours, spatial, temporal, or otherwise: a galaxy in its environment, a halo in its history, a photon hit in a detector. Graphs are ideally suited for encoding exactly these relations, and Graph Neural Networks (GNNs) learn the geometric functions that respect them. This review presents the use of GNNs for astrophysics and cosmology, and argues that GNNs often suit physics and physical data better than any alternative: they fit the native geometry of the domain and the signal, enable the incorporation of the symmetries and prior knowledge we already trust and have toiled to build, and stay flexible enough to capture what we do not yet understand. GNNs' physically intuitive workings also render them more interpretable than other deep-learning architectures. We show how we can begin from general prin
    
[^218]: 面向本体无关电子健康记录建模的离散神经中介语言

    Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling

    [https://arxiv.org/abs/2610.27047](https://arxiv.org/abs/2610.27047)

    本文提出MEDIATOR——一种本体无关的离散分词器，通过将任意文本形式表达的临床概念映射到共享的向量量化码本，将EHR数据协调转化为开放词表分词问题，从而打破不同医疗系统间的编码壁垒。

    

    电子健康记录（EHR）基础模型有望实现跨医疗系统的可扩展临床预测，但其部署受到分词瓶颈的制约。同一临床概念可能以不同的本体编码、本地标识符、同义词或自由文本描述的形式出现，而标准的EHR分词器会将它们视为互不相关的符号。使用封闭词表分词器在MIMIC-III上训练的模型，仅能使用eICU中13%的事件。我们提出了MEDIATOR，这是一种离散的、本体无关的分词器，它将任何以文本形式表达的临床概念映射到一个共享的向量量化码本中，从而将EHR数据协调统一转化为开放词表分词问题，无需人工映射或固定词表。通过同义词监督进行训练，并借助丰富的概念描述、编码共现信息和关系图进行引导，MEDIATOR能够合并等价表达，同时在具有医学意义的离散空间中组织不同概念。用作分词接口时（摘要在此处截断）

    arXiv:2610.27047v1 Announce Type: cross  Abstract: EHR foundation models promise scalable clinical prediction across health systems, yet their deployment is constrained by a tokenization bottleneck. The same clinical concept can appear as different ontology codes, local identifiers, synonyms, or free-text descriptions, which standard EHR tokenizers treat as unrelated symbols. A model trained on MIMIC-III with a closed-vocabulary tokenizer can use only 13% of eICU events. We introduce MEDIATOR, a discrete ontology-agnostic tokenizer that maps any clinical concept expressed as text into a shared vector-quantized codebook, turning EHR harmonization into open-vocabulary tokenization without manual mappings or a fixed vocabulary. Trained with synonym supervision and guided by enriched concept descriptions, code co-occurrence, and relation graphs, MEDIATOR collapses equivalent expressions while organizing distinct concepts in a medically meaningful discrete space. Used as the token interface
    
[^219]: ReuF2F：基于形式化本科研究问题的编码智能体基准测试

    ReuF2F: Benchmarking Coding Agents on Formalized Undergraduate Research Problems

    [https://arxiv.org/abs/2610.27046](https://arxiv.org/abs/2610.27046)

    该论文提出了 ReuF2F，一个基于 Lean 4 的基准测试，包含 244 个源自本科生研究经验计划（REU）的形式化开放研究问题，用以更可靠地评估编码智能体解决开放数学问题的能力。

    

    随着人工智能系统开始解决开放性研究问题，可靠的验证变得日益重要。形式化数学为数学推理提供了一个可信的框架，使智能体在进行数学探索的同时能够生成机器可检验的证明。现有的形式化数学基准测试通常集中于已知结果或少数著名的开放问题，一旦这些问题的解被发现并广泛传播，就难以再对其进行可靠评估。此外，现有的评估协议也并不总是与现代编码智能体的工作方式相匹配。因此，我们提出了 ReuF2F，这是一个基于 Lean 4 的基准测试，包含从本科生研究经验计划（Research Experiences for Undergraduates, REU）中提取的 244 个形式化开放问题，涵盖了广泛数学子领域中的入门级开放研究问题。我们预计这些较少受到关注的开放问题——它们往往吸引的注意力较少，并可能长时间未被解决——将更……（摘要在此处被截断）

    arXiv:2610.27046v1 Announce Type: new  Abstract: As AI systems begin to resolve open research problems, reliable verification becomes increasingly important. Formal mathematics provides a trusted framework for mathematical reasoning, enabling agents to produce machine-checkable proofs while conducting mathematical exploration. Existing formal mathematics benchmarks often focus on known results or a small set of prominent open problems, which can become difficult to evaluate reliably once solutions are discovered and widely disseminated. Existing evaluation protocols are also not always aligned with how modern coding agents operate. We therefore introduce ReuF2F, a Lean 4 benchmark of 244 formalized open problems drawn from Research Experiences for Undergraduates (REU), spanning entry-level open research problems across a broad range of mathematical subfields. We expect these less prominent open problems, which tend to attract less attention and may remain unsolved for longer, to be bet
    
[^220]: 希尔伯特尺度上后验扩散先验用于不适定信号恢复的稳定性

    Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery

    [https://arxiv.org/abs/2610.27030](https://arxiv.org/abs/2610.27030)

    该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。

    

    当正向算子会把未知信号平滑掉时，间接且带噪声的观测难以被反演。在这种情况下，不稳定的模态恰恰是数据揭示信息最少的模态，因此任何重构方法的成功都取决于先验信息与算子几何结构在不同正则性尺度上的相互作用方式。扩散先验在这类问题中已被广泛使用；然而，现有理论仍无法确定：在不适定正向算子作用下，学习到的分数函数何时仍能保持有效性。本文针对希尔伯特尺度上的线性逆问题研究后验扩散，并在一个显式的分数放大条件下推导出稳定性理论。在正向映射满足平滑条件、真值满足源条件、以及分数误差具有弱范数控制的假设下，我们推导出了希尔伯特空间 \(H^s\) 中的非渐近恢复界，该界将观测噪声、分数近似误差与反向时间（原文摘要在此处截断）的影响分离开来。

    arXiv:2610.27030v1 Announce Type: new  Abstract: Indirect and noisy observations are difficult to invert when the forward operator smooths out the unknown signal. In this case, the unstable modes are precisely the modes that the data reveals least, thus the success of any reconstruction method depends on how prior information interacts with the geometry of the operator across regularity scales. Diffusion priors have become ubiquitous for such problems; however, the current theory still does not determine when a learned score remains informative under ill-posed forward operators. In this paper, we study posterior diffusion for linear inverse problems on Hilbert scales and derive a stability theory under an explicit score-amplification condition. Under a smoothing condition on the forward map, a source condition on the truth, and weak-norm control of score error, we derive non-asymptotic recovery bounds in \(\Hs^s\) that separate observation noise, score approximation, and reverse-time d
    
[^221]: EGiME：小样本学习中的熵引导模型评估

    EGiME: Entropy-Guided Model Evaluation in Few-Shot Learning

    [https://arxiv.org/abs/2610.27029](https://arxiv.org/abs/2610.27029)

    提出了一种无需训练分类器的半监督评估准则EGiME，通过逆频率权重融合簇间熵与簇内熵来衡量预训练模型特征空间的聚类结构，在五个小样本基准上与真值准确率达到0.966的相关性。

    

    预训练模型的特征提取质量决定了下游迁移学习的上限，然而小样本模型选择仍然困难：基于稀缺标签的监督式评估不稳定，且排名容易波动。我们提出了熵引导模型评估（EGiME），这是一种半监督准则，无需训练任何分类器，即可从候选模型特征空间的聚类结构出发对其进行评估。EGiME 对所有有标签和无标签样本执行 K-means 聚类，然后结合两个熵项：衡量全局可分性的簇间熵，以及衡量有标签锚点紧凑性的簇内熵，二者通过强调小簇的逆频率权重进行融合。在五个小样本基准和九个预训练模型上，EGiME 与通过平均 K-means、KNN 和 SVM 准确率获得的全池真值准确率之间的相关性得分达到 0.966。

    arXiv:2610.27029v1 Announce Type: new  Abstract: The feature-extraction quality of pre-trained models sets the ceiling of downstream transfer learning, yet few-shot model selection remains difficult: supervised evaluation on scarce labels is unstable and yields fluctuating rankings. We propose Entropy-Guided Model Evaluation (EGiME), a semi-supervised criterion that assesses a candidate model from the clustering structure of its feature space without any classifier training. EGiME performs $K$-means clustering over all labeled and unlabeled samples, then combines two entropy terms: an inter-cluster entropy measuring global separability, and an intra-cluster entropy measuring the compactness of labeled anchors, fused through an inverse-frequency weight that emphasizes small clusters. On five few-shot benchmarks and nine pre-trained models, EGiME achieves a correlation score of 0.966 with the full-pool ground-truth accuracy obtained by averaging the accuracies of K-means, KNN, and SVM. A
    
[^222]: 流形上逻辑斯蒂分类的Besov函数高斯逼近

    Gaussian Approximation of Besov Functions for Logistic Classification on Manifolds

    [https://arxiv.org/abs/2610.27022](https://arxiv.org/abs/2610.27022)

    该论文针对紧致光滑流形上的正则化逻辑斯蒂分类，借助常数线性依赖预测器上确界范数的Bernstein型不等式得到无需有界贝叶斯假设的快速预言机不等式，并通过多带宽高斯核积分算子的带符号组合与实插值方法，构造出同时利用内在维度与目标正则性且无饱和现象的Besov函数高斯逼近。

    

    我们研究了连通、紧致、光滑的嵌入流形 $X\subset\mathbb R^d$（无边界，内在维度为 $\varrho$）上的高斯核逼近与正则化逻辑斯蒂分类问题。两个主要挑战分别是：在不假设贝叶斯函数有界的前提下推导快速的预言机不等式，以及构造既能利用内在维度又能利用目标函数正则性、且不出现饱和现象的高斯逼近器。对于第一个挑战，我们通过一个Bernstein型不等式加以解决，该不等式中的常数与预测器的上确界范数呈线性依赖关系。结合局部化的复杂度估计，由此得到了可分赋范函数空间上的预言机不等式。针对第二个挑战，我们建立了一个一致的Sobolev算子展开，并利用不同带宽下高斯核积分算子的带符号组合来抵消其低阶偏差项。随后通过实插值方法，进而得到Besov函数的 $L_2(X)$ 逼近（摘要原文在此处截断）。

    arXiv:2610.27022v1 Announce Type: cross  Abstract: We study Gaussian kernel approximation and regularized logistic classification on a connected compact smooth embedded manifold $X\subset\mathbb R^d$ without boundary and of intrinsic dimension $\varrho$. Two main challenges are to derive a fast oracle inequality without assuming a bounded Bayes function and to construct Gaussian approximants that exploit both the intrinsic dimension and the target's regularity without saturation. We address the first challenge through a Bernstein-type inequality whose constant depends linearly on the supremum norm of the predictor. Combined with localized complexity estimates, this yields an oracle inequality over separable normed function spaces. To address the second challenge, we establish a uniform Sobolev operator expansion and cancel its lower-order bias terms using a signed combination of Gaussian kernel integral operators at different bandwidths. Real interpolation then gives an $L_2(X)$ approx
    
[^223]: 选择偏差下差分隐私的可靠估计

    Reliable Estimation of Differential Privacy under Selection Bias

    [https://arxiv.org/abs/2610.27021](https://arxiv.org/abs/2610.27021)

    本文揭示了以往差分隐私审计实现在运行成员推断攻击时引入选择偏差、从而产生统计上无效且被高估的隐私泄露下界这一问题，并提出了在选择偏差下可靠估计差分隐私的方法。

    

    arXiv:2610.27021v1 公告类型：交叉。摘要：统计隐私审计对于验证差分隐私（DP）机制的理论隐私保证至关重要。审计人员通常使用成员推断攻击（MIA）在指定置信水平下估计DP机制真实隐私泄露的下界，该下界不应超过DP所保证的理论隐私上界。然而，我们发现以往工作中的许多审计实现在运行MIA时引入了选择偏差，导致统计上无效且被高估的下界。由于大多数审计本身并不紧致，这种偏差至今未被察觉：估计出的下界仍远低于理论上界。尽管如此，准确评估这一下界至关重要，因为存在缺陷的统计评估可能导致对正确机制的误报，或对隐私保证紧致性的错误评估。本工作采用了来自统……（摘要不完整）

    arXiv:2610.27021v1 Announce Type: cross  Abstract: Statistical privacy audits are essential for verifying the theoretical privacy guarantees of differentially private (DP) mechanisms. Auditors typically use Membership Inference Attacks (MIAs) to estimate a lower bound on the true privacy leakage of DP mechanisms at a specified confidence level, which should not exceed the theoretical privacy upper bound guaranteed by DP. However, we found that many auditing implementations in prior work introduce a selection bias when running MIAs, which leads to statistically invalid and overestimated lower bounds. Because most audits are not tight, this bias has thus far gone unnoticed: the estimated lower bounds remained well below the theoretical upper bounds. Nevertheless, accurately evaluating this lower bound is critical, as a flawed statistical assessment could lead to false alarms on correct mechanisms or an incorrect evaluation of the guarantee's tightness. This work employs tools from the st
    
[^224]: MultiCard：基于深度学习的多属性基数估计

    MultiCard: Multiple Attribute Cardinality Estimation via Deep Learning

    [https://arxiv.org/abs/2610.27019](https://arxiv.org/abs/2610.27019)

    MultiCard 提出了一个统一的端到端深度学习基数估计器，通过 Transformer 与 LSTM 相结合来同时处理 LIKE 字符串谓词和数值过滤谓词，并能动态评估直方图估计在不同算子和数据倾斜条件下的准确性。

    

    基数估计对查询优化至关重要，但现有的学习型估计器大多只关注单一类型的谓词，例如字符串 LIKE 子句或数值范围过滤条件。在实际应用中，分析型工作负载往往同时包含这两类谓词，而且字符串估计器的准确率通常会随着模式变长而下降。我们提出了 MultiCard，一个全面的学习型基数估计器，它在单一的端到端架构中统一处理 LIKE 谓词和数值过滤条件。通过将用于处理长而稀疏模式的 Transformer 与用于处理短而高频模式的 LSTM 相结合，MultiCard 实现了鲁棒的模式建模。与此同时，一个数值编码器整合了数据目录中的直方图和结构化谓词特征，使模型能够在不同算子和数据倾斜条件下动态评估直方图估计的准确性。最后一个 Transformer 层捕捉异构谓词之间的交互，然后经过多……（原文摘要在此处截断）

    arXiv:2610.27019v1 Announce Type: cross  Abstract: Cardinality estimation is essential for query optimization, but most existing learned estimators focus on a single predicate type, such as string LIKE clauses or numerical range filters. In practice, analytical workloads often combine both types, while the accuracy of string estimators tends to decrease with longer patterns. We introduce MultiCard, a comprehensive learned cardinality estimator that unifies LIKE predicates and numerical filters within a single end-to-end architecture. By combining a Transformer for long-range sparse patterns with an LSTM for short, frequent patterns, MultiCard achieves robust pattern modeling. In parallel, a numerical encoder integrates catalog histograms and structural predicate features, enabling the model to dynamically assess histogram estimation accuracy across different operators and data skew conditions. A final Transformer layer captures interactions among heterogeneous predicates before a multi
    
[^225]: 基于激活空间引导的可验证奖励强化学习探索方法

    Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2610.27012](https://arxiv.org/abs/2610.27012)

    提出激活空间探索（ASE）方法，通过聚类正确与错误响应的激活差异构建潜在探索向量库，并将其应用于训练采样以引导模型探索多样化推理路径，从而提升可验证奖励强化学习在数学推理上的性能。

    

    可验证奖励强化学习（RLVR）通过采样模型的响应、使用自动验证器对其评分，并根据所得奖励更新策略来训练语言模型。因此，策略只能从其自身采样生成的轨迹中获得直接学习信号。然而，从同一策略反复采样可能会集中在相似的推理路径上，在固定的采样预算下导致有用的替代路径未被探索。我们提出了激活空间探索（ASE），这是一种用于多样化训练采样的在线方法。通过聚类同一问题正确与错误响应之间的激活差异，ASE 构建了一个结构化的潜在探索向量库。随后，不同的向量被应用于后续的采样过程中，以探索替代的推理路径。在 DAPO-Math-17k 上训练 Qwen3-1.7B-Base 和 Qwen3-8B-Base 时，ASE 在六个数学推理基准上提高了平均 Mean@8 性能。

    arXiv:2610.27012v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains a language model by sampling responses, scoring them with an automatic verifier, and updating the policy from the resulting rewards. The policy therefore receives direct learning signals only from trajectories that appear in its rollouts. Yet repeated sampling from the same policy can concentrate on similar reasoning paths, leaving useful alternatives unexplored under a fixed rollout budget. We introduce Activation-Space Exploration (ASE), an online method for diversifying training rollouts. By clustering activation differences between correct and incorrect responses to the same problem, ASE builds a structured bank of latent exploration vectors. Different vectors are then applied to subsequent rollouts to explore alternative reasoning paths. On Qwen3-1.7B-Base and Qwen3-8B-Base trained on DAPO-Math-17k, ASE improves average Mean@8 across six mathematical reasoning benchmarks
    
[^226]: 鲁棒潜变量图模型：精确恢复与基本极限

    Robust Latent Variable Graphical Models: Exact Recovery and Fundamental Limits

    [https://arxiv.org/abs/2610.27010](https://arxiv.org/abs/2610.27010)

    本文提出了一种用测度变换协方差替代样本协方差、并用合同变换核范数惩罚低秩部分的凸估计方法，用于鲁棒估计具有“稀疏减低秩”结构的高斯精度矩阵，并在受限可逆性与不可表示条件下证明了精确恢复保证，包括解的唯一性、无虚假边、边符号恢复正确以及秩和惯性的准确恢复。

    

    我们研究高斯精度矩阵的鲁棒估计问题，该矩阵可分解为一个稀疏矩阵减去一个对称的、可能不定的低秩矩阵，这种结构出现在潜变量边缘化与线性测量条件化相结合的情形中。所提出的凸估计器用中心化的高斯测度变换（MT）协方差替代样本协方差，该协方差在标称模型下能够保留稀疏分量的带符号图结构，并通过一个合同变换的核范数惩罚低秩分量，这是一种编码了结构侧信息的原子范数。我们的主要结果是一个原始-对偶见证定理：在Fisher信息满足受限可逆性与不可表示条件的情况下，凸优化的原始解是唯一的，不产生虚假边，能够恢复所有强度足够高的边的符号，并且恰好具有真实的秩与惯性。这些条件可归结为图形化拉（截断）……

    arXiv:2610.27010v1 Announce Type: cross  Abstract: We study robust estimation of a Gaussian precision matrix that decomposes as a sparse matrix minus a symmetric, possibly indefinite, low-rank matrix, a structure that arises when latent-variable marginalization is combined with conditioning on linear measurements. The proposed convex estimator replaces the sample covariance by the centered Gaussian measure-transformed (MT) covariance, which preserves the signed graph of the sparse component under the nominal model, and penalizes the low-rank component by the nuclear norm of a congruence transform, an atomic norm that encodes structural side information. Our main result is a primal--dual witness theorem: under restricted invertibility and irrepresentability conditions on the Fisher information, the raw convex optimizer is unique, has no false edges, recovers the signs of all sufficiently strong edges, and has exactly the true rank and inertia. These conditions reduce to the graphical-la
    
[^227]: HierDTA：用于药物-靶点亲和力预测的层次化结构感知表示学习

    HierDTA: Hierarchical Structure-Aware Representation Learning for Drug--Target Affinity Prediction

    [https://arxiv.org/abs/2610.27001](https://arxiv.org/abs/2610.27001)

    提出层次化结构感知框架HierDTA，通过将药物表示为原子-基序异构图、将蛋白激酶结构域表示为残基相互作用图，并利用交叉注意力融合多级结构信息，从而提升药物-靶点亲和力预测的准确性。

    

    药物-靶点亲和力（DTA）预测在计算机辅助药物发现中发挥着重要作用，可靠的计算模型可以在实验验证之前帮助优先筛选化合物-蛋白质对。DTA预测的一个关键挑战是如何有效地从药物结构中挖掘多层次分子信息，同时准确建模靶点蛋白的结合位点环境。为解决这些问题，我们提出了HierDTA，一个用于DTA预测的层次化结构感知框架。HierDTA将每种药物表示为原子-基序异构图，并引入一个增强模块，将功能基团语义传播到相应的原子上。对于蛋白质，将激酶结构域结构转换为残基相互作用图。所得的原子表示和残基表示通过交叉注意力进行整合，以捕捉药物-蛋白质对特异性的关系，同时利用分子指纹和预训练（模型信息，摘要在此处截断）……

    arXiv:2610.27001v1 Announce Type: cross  Abstract: Drug--target affinity (DTA) prediction plays an important role in computer-aided drug discovery, where reliable computational models can help prioritize compound--protein pairs before experimental validation. A key challenge in DTA prediction is to effectively mine multi-level molecular information from drug structures while accurately modeling the binding-site environment of target proteins. To address these issues, we introduce HierDTA, a hierarchical structure-aware framework for DTA prediction. \textcolor{black}{HierDTA represents each drug as an atom--motif heterogeneous graph and introduces an augmentation module that propagates functional-motif semantics to the corresponding atoms. For proteins, kinase-domain structures are converted into residue interaction graphs. The resulting atom and residue representations are integrated using cross-attention to capture pair-specific relationships, while molecule fingerprints and pretraine
    
[^228]: ORBIT：通过下一频带预测实现自回归时间序列生成

    ORBIT: Autoregressive Time Series Generation via Next Frequency-Band Prediction

    [https://arxiv.org/abs/2610.27000](https://arxiv.org/abs/2610.27000)

    提出了统一的时间序列生成框架ORBIT，通过频率分层VQ-VAE学习跨频带的分层时间表示，并利用频带自回归Transformer实现从低频到高频的自回归生成，从而解决了异构时间模式的表示学习与生成协调两大难题。

    

    深度生成模型通过合成逼真的时间序列，为缓解数据稀缺问题提供了有效的现代解决方案。然而，尽管现有的深度时间序列生成器已经取得了长足进步，但仍存在两个关键挑战：（1）为异构时间模式学习分层表示；（2）协调异构时间动态的生成过程。为解决这些关键瓶颈，本工作提出了ORBIT——一个统一的时间序列深度生成框架。具体而言，我们开发了两个关键组件：频率分层VQ-VAE（FH-VQ-VAE）和频带自回归Transformer（FBAR）。前者通过将潜在时间表示分解为不同的频带，并应用具有递增token预算的残差量化来学习分层表示；后者建立了一种新颖的从低频到高频的自回归生成机制（原文摘要在此处截断）。

    arXiv:2610.27000v1 Announce Type: new  Abstract: Deep generative models provide an effective modern solution to alleviating the problem of data scarcity by synthesizing realistic time series. However, despite the considerable progress achieved by existing deep time series generators, two key challenges remain: (1) learning hierarchical representations for heterogeneous temporal patterns; and (2) coordinating the generation process of heterogeneous temporal dynamics. To address these key bottlenecks, in this work, we propose ORBIT, a unified deep generative framework for time series. Specifically, we develop two key components: the Frequency-Hierarchical VQ-VAE (FH-VQ-VAE) and the Frequency-Band Autoregressive Transformer (FBAR). The former learns hierarchical representations by decomposing latent temporal representations into different frequency bands and applying residual quantization with increasing token budgets. The latter establishes a novel low-to-high frequency autoregressive di
    
[^229]: CorteXforge：基于真实测试平台的场景化射频数据集生成

    CorteXforge: Scenario-Based RF Dataset Generation on Real Testbeds

    [https://arxiv.org/abs/2610.26999](https://arxiv.org/abs/2610.26999)

    CorteXforge是一个基于场景的框架，能够在真实SDR测试平台上自动生成真实且可重复的射频数据集，实验揭示了合成数据与真实数据之间的性能差距，同时证明合成数据预训练有助于模型适应真实空中测量数据。

    

    用于射频信号分析的深度学习依赖于大型标注数据集，然而合成数据集以牺牲物理真实性来换取可控性，而不受控的空中测量虽具有真实性但可重复性有限。我们提出了CorteXforge，这是一个基于场景的框架，用于在可编程软件定义无线电测试平台上生成真实且可重复的射频数据集。高层射频场景被自动转换为在SLICES-RI/CorteXlab平台上执行的同步实验，生成完整标注的信号元数据格式数据集，这些数据集捕捉了硬件损伤和传播效应。我们通过使用ModForge（一个利用该框架生成的空中数据集）进行自动调制分类来验证CorteXforge。跨域实验揭示了合成到真实的性能差距，并表明合成数据预训练对于适应真实空中数据仍然有益。

    arXiv:2610.26999v1 Announce Type: new  Abstract: Deep learning for radio frequency (RF) signal analysis relies on large annotated datasets, yet synthetic datasets trade physical realism for controllability, while uncontrolled over-theair measurements offer realism but limited reproducibility. We present CorteXforge, a scenario-based framework for generating realistic and reproducible RF datasets on programmable software defined radio (SDR) testbeds. High-level RF scenarios are automatically translated into synchronized experiments executed on the SLICES-RI/CorteXlab platform, producing fully annotated Signal Metadata Format (SIGMF) datasets that capture hardware impairments and propagation effects. We validate CorteXforge through automatic modulation classification using ModForge, an over-the-air (OTA) dataset generated with the framework. Cross-domain experiments reveal a synthetic-to-real performance gap and show that synthetic pretraining remains beneficial for adaptation to OTA dat
    
[^230]: CoASFL：面向异步拆分联邦学习的耦合一致性优化

    CoASFL: Coupled Consistency Optimization for Asynchronous Split Federated Learning

    [https://arxiv.org/abs/2610.26992](https://arxiv.org/abs/2610.26992)

    该论文提出CoASFL框架，通过动态激活合并机制和一致性感知的模型聚合机制，有效解决了异步拆分联邦学习由系统异构性与统计异构性引发的水平与垂直耦合不一致性问题。

    

    拆分联邦学习（SFL）已成为在资源受限的边缘设备上训练大规模深度模型的一种有前景的范式。尽管异步拆分联邦学习缓解了由系统异构性导致的掉队者瓶颈，但它不可避免地引入了两个相互耦合的不一致性维度：客户端之间的水平不一致性以及客户端与服务器之间的垂直不一致性。这些不一致性在统计异构性的进一步加剧下，会导致激活表示偏差、模型发散以及严重的客户端漂移。为了解决这些挑战，我们提出了CoASFL，一种不一致性感知的异步拆分联邦学习框架，该框架融合了动态激活合并机制和一致性感知的模型聚合机制。具体而言，CoASFL从异构客户端构建更加一致的激活序列以缓解水平不一致性，并动态调整聚合权重

    arXiv:2610.26992v1 Announce Type: cross  Abstract: Split Federated Learning (SFL) has emerged as a promising paradigm for training large-scale deep models on resource-constrained edge devices. Although asynchronous SFL alleviates the straggler bottleneck caused by system heterogeneity, it inevitably introduces two coupled dimensions of inconsistency: horizontal inconsistency across clients and vertical inconsistency between clients and the server. These inconsistencies, further exacerbated by statistical heterogeneity, result in biased activation representations, model divergence, and severe client drift. To address these challenges, we propose CoASFL, an inconsistency-aware asynchronous SFL framework that incorporates a dynamic activation merging mechanism and a consistency-aware model aggregation mechanism. Specifically, CoASFL constructs more consistent activation sequences from heterogeneous clients to alleviate horizontal inconsistency, and dynamically adjusts aggregation weights 
    
[^231]: 用于时间序列预测的动态双反馈保形推断

    Dynamic Dual-Feedback Conformal Inference for Time Series Forecasting

    [https://arxiv.org/abs/2610.26987](https://arxiv.org/abs/2610.26987)

    该论文提出动态双反馈保形推断（DDCI），通过同时利用误覆盖指示的实际反馈和基于理想保形方法的估计反馈来自适应调整预测阈值，从而在分布变化环境下实现更可靠的时间序列预测不确定性量化。

    

    arXiv:2610.26987v1 公告类型：cross 摘要：时间序列预测中的不确定性量化要求尤为苛刻，因为序列数据具有时间依赖性，且容易发生分布变化。保形推断通过构造预测集合，已成为一种强大的不确定性量化方法。近期的研究进展引入了在线保形方法，通过反馈机制自适应地调整预测阈值。然而，现有的反馈机制通常仅依赖误覆盖指示器（实际反馈），即在每个时间步真实标签是否落在区间内，而忽视了由理想保形方法推导出的经验预测阈值（估计反馈）。在本文中，我们提出了动态双反馈保形推断（DDCI），它融合了由实际反馈和估计反馈组成的双反馈机制。前者驱动主要的调整……

    arXiv:2610.26987v1 Announce Type: cross  Abstract: Quantifying uncertainty in time series forecasting is particularly demanding because sequential data exhibit temporal dependence and are prone to distributional changes. Conformal inference has emerged as a powerful uncertainty quantification approach through the construction of prediction sets. Recent advances have introduced online conformal methods that adaptively adjust prediction thresholds through feedback mechanisms. However, the existing feedback mechanism typically relies solely on miscoverage indicators (actual feedback), whether the true label falls within the interval at each time step, while overlooking the empirical prediction threshold (estimated feedback) that is derived from the oracle conformal method. In this paper, we propose Dynamic Dual-feedback Conformal Inference (DDCI), which incorporates a dual-feedback mechanism consisting of actual feedback and estimated feedback. The former drives the primary adjustment of 
    
[^232]: 面向潜空间规划的潜空间投影

    Latent Projection for Latent Planning

    [https://arxiv.org/abs/2610.26979](https://arxiv.org/abs/2610.26979)

    该论文发现世界模型训练中防止坍缩的正则化器（SIGReg）本身蕴含高斯-马尔可夫结构，可零额外成本地导出潜空间各方向的协方差（不确定性），从而在潜空间规划中有效剪除不携带信号的噪声维度。

    

    潜世界模型使智能体能够在潜空间中进行规划。规划器搜索使代价最小化的动作序列，而该代价通常是预测的终端潜变量与目标潜变量之间的 ℓ2 距离。然而，潜变量中有一部分是噪声：世界模型训练中用于防止表示坍缩的正则化器会在不携带信号的方向上促进方差。我们证明，同一个正则化器也提供了剪除潜变量中这些噪声的结构。SIGReg 使潜变量的边缘分布在每一步都趋于标准高斯分布；若进一步假设潜变量序列为高斯-马尔可夫过程，则其转移是线性的，且协方差由映射本身所确定：z′ | z ~ N(Az, I − AAᵀ)。因此，拟合状态动力学 A 无需任何额外代价即可获得协方差，并且在其特征基下，每个潜变量方向都携带自身的不确定性。此外，将 A 约束为正规矩阵……（原文摘要在此处截断）

    arXiv:2610.26979v1 Announce Type: new  Abstract: Latent world models let an agent plan in latent space. The planner searches for an action sequence that minimizes a cost, and that cost is often the $\ell_2$ distance between the predicted terminal latent and the goal latent. The latent, however, is partly noise: the regularizer that prevents collapse during world model training promotes variance in the directions carrying no signal. We show that the same regularizer also supplies the structure to prune that noise in the latent. SIGReg drives the latent marginal to a standard Gaussian at every step, and assuming further that the latent sequence is a Gauss-Markov process, the transition is linear with its covariance pinned by the map itself, $z' \mid z \sim \mathcal{N}(Az,\, I - AA^\top)$. Fitting the state dynamics $A$ therefore yields the covariance at no additional cost, and in its eigenbasis each latent direction carries its own uncertainty. Furthermore, constraining $A$ to be a norma
    
[^233]: Transformer学习两种相互竞争的排序算法，且只有一种能实现长度泛化

    Transformers Learn Two Competing Sorting Algorithms, and Only One Length-Generalizes

    [https://arxiv.org/abs/2610.26974](https://arxiv.org/abs/2610.26974)

    该论文发现两层Transformer在排序任务上会根据权重初始化收敛到两种机制截然不同的解，其中只有同时使用两个注意力层的解能在仅训练固定长度序列的情况下完美实现长度泛化，且可以通过精心设计权重初始化来引导模型学习目标解。

    

    我们证明，在排序任务上训练至接近零损失的两层Transformer，会根据其初始化收敛到两种机制上截然不同的解类别之一。其中一类解不使用第二个注意力层进行排序，同时也无法实现长度泛化。另一类解同时使用两个注意力层，尽管仅在固定长度的序列上训练，却能完美地实现长度泛化。通过精心选择权重初始化，我们证明可以引导模型学习其中一种解或另一种解。我们识别出在训练长度下罕见、但在更长序列长度下日益常见的输入类型，将其作为这种长度泛化差异的来源，并发现第二类解的逐层专门化使其能够更准确地处理这些罕见序列。在第二类解中，我们发现了一种机制，通过该机制，第二层（摘要在此处截断）

    arXiv:2610.26974v1 Announce Type: cross  Abstract: We show that two-layer transformers trained to near-zero loss on a sorting task converge to one of two mechanistically distinct solution classes depending on their initialization. One solution class consists of solutions that do not use the second attention layer to sort, but also fail to length-generalize. The other, which uses both attention layers, length-generalizes perfectly, despite only being trained to sort sequences of a fixed length. By carefully choosing the weight initialization, we demonstrate that we can steer the model toward learning one solution or the other. We identify input types that are rare at the training length but increasingly common at longer sequence lengths as a source of this disparity in length-generalization, and find that layer-wise specialization of the second class of solutions allows them to handle these rare sequences more accurately. In the second class, we discover a mechanism by which the second 
    
[^234]: SZED参加ArGuard共享任务：面向阿拉伯语仇恨表情包检测的多任务标签注意力与跨模态融合方法

    SZED at ArGuard Shared Tasks: Multi-Task Label-Attention and Cross-Modal Fusion for Arabic Hateful Meme Detection

    [https://arxiv.org/abs/2610.26972](https://arxiv.org/abs/2610.26972)

    该论文提出了一种融合MARBERTv2文本嵌入与CLIP视觉嵌入的多任务标签注意力架构，并引入级联校准机制，用于阿拉伯语仇恨表情包检测，在二分类任务上取得macro-F1 0.7597、在细粒度多标签任务上取得0.3142的成绩。

    

    我们展示了SZED团队参加ArGuard 2026赛道A（阿拉伯语仇恨表情包检测）的提交方案。该任务包含两个部分：二分类（A1）和细粒度多标签标注（A2）。对于A1，我们融合了MARBERTv2和CLIP ViT-L/14的嵌入表示，并比较了拼接、自注意力和集成头三种融合方式。对于A2，我们采用多任务标签注意力架构，为每个类别设置独立的查询向量，并使用辅助头进行级联校准。我们在A1上达到macro-F1 = 0.7597，在A2上达到0.3142（对10个活跃类别取平均）。我们报告了多次运行结果的可复现性、各类别的错误模式，以及级联校准对长尾分布A2标签空间的影响。

    arXiv:2610.26972v1 Announce Type: cross  Abstract: We present SZED's submission to ArGuard 2026 Track A: Arabic hateful meme detection. The task has two parts: binary classification (A1) and fine-grained multi-label tagging (A2). For A1, we fuse MARBERTv2 and CLIP ViT-L/14 embeddings, comparing concatenation, self-attention, and ensemble heads. For A2, we use a multi-task label-attention architecture with per-class query vectors and an auxiliary head for cascade calibration. We reach macro-F1 = 0.7597 on A1 and 0.3142 on A2 (averaged over 10 active classes). We report reproducibility across runs, per-class error patterns, and the effect of cascade calibration on the long-tailed A2 label space.
    
[^235]: 当弱次模性遇上拟阵：标准贪心算法表现如何？

    When Weak Submodularity Meets Matroids: How Does Standard Greedy Behave?

    [https://arxiv.org/abs/2610.26967](https://arxiv.org/abs/2610.26967)

    本文解决了弱次模最大化在一般拟阵约束下标准贪心算法最坏情况近似保证这一长期开放问题，给出了以次模比 $\gamma$ 和拟阵秩 $r$ 表达的精化下界。

    

    弱次模最大化在机器学习、数据挖掘和统计学中有众多应用。特别是在基数约束下，标准贪心算法已被证明可以达到紧致的 $(1-e^{-\gamma})$-近似保证，其中 $\gamma$ 表示次模比。然而，在考虑一般拟阵约束时，标准贪心算法究竟能表现多好一直是一个长期悬而未决的开放问题。本文通过研究标准前向贪心算法的最坏情况近似保证 $R_{\mathrm{gr}}(\gamma,r)$（其中 $r$ 为拟阵的秩）来解答这一问题。具体而言，本文的第一个贡献是给出了如下精化的下界：$R_{\gr}(\gamma,r)\ge \max\left\{ \frac{\gamma}{r}, \frac{\gamma^2}{B_\gamma+\sqrt{\gamma(r-1)}}, \frac{\gamma^{r+1}}{2-\gamma+\gamma^{r+1}}, \frac{\gamma^2}{14\left(B_\gamma+\sqrt{\gamma(1-\gamma)(r-1)}\right)} \right\}$。

    arXiv:2610.26967v1 Announce Type: cross  Abstract: Weakly submodular maximization finds numerous applications in machine learning, data mining and statistics. In particular, under a cardinality constraint, the standard greedy algorithm is known to attain the tight $(1-e^{-\gamma})$-approximation guarantee, where $\gamma$ denotes the submodularity ratio. However, when considering general matroid constraints, \emph{how well standard greedy can perform} has remained a longstanding open question. In this paper, we resolve this question by studying the worst-case approximation guarantee $R_{\mathrm{gr}}(\gamma,r)$ of standard forward greedy, where $r$ is the rank of matroid. Specifically, our first contribution is the following refined lower bound \begin{equation*} R_{\gr}(\gamma,r)\ge \max\left\{ \frac{\gamma}{r}, \frac{\gamma^2} {B_\gamma+\sqrt{\gamma(r-1)}}, \frac{\gamma^{r+1}} {2-\gamma+\gamma^{r+1}}, \frac{\gamma^2} {14\left(B_\gamma+\sqrt{\gamma(1-\gamma)(r-1)}\right)} \right\}, \end{
    
[^236]: FoundPDE：用于数据驱动的偏微分方程符号发现的生成式基础模型

    FoundPDE: A generative foundation model for data-driven symbolic discovery of partial differential equations

    [https://arxiv.org/abs/2610.26963](https://arxiv.org/abs/2610.26963)

    提出FoundPDE——首个预训练的数据到符号生成式基础模型，利用Transformer将离散时空轨迹直接映射为符号偏微分方程，并通过傅里叶与一阶空间导数增强及少样本微调，实现了在噪声、稀疏数据下对广泛方程族的鲁棒发现与良好泛化。

    

    从观测数据中发现控制偏微分方程（PDE）是计算科学与工程中的核心逆问题：其目标不仅是预测轨迹，而是识别出可分析、可复用的可解释符号规律。现有的偏微分方程发现方法通常依赖手工设计的函数库、精确的数值导数或针对单个方程的符号搜索，这使它们在稀疏观测、噪声数据、广泛的方程族以及未见过的符号组合等情况下表现脆弱。本文提出了FoundPDE，一个用于偏微分方程发现的预训练数据到符号基础模型。FoundPDE使用Transformer将离散化的时空轨迹映射为符号方程，利用傅里叶变换和一阶空间导数增强来揭示互补的轨迹信息，并对数值系数进行精化，同时通过少样本微调支持下游扩展并提升泛化能力。

    arXiv:2610.26963v1 Announce Type: cross  Abstract: Discovering governing partial differential equations (PDEs) from observed data is a central inverse problem in computational science and engineering: the goal is not only to predict trajectories, but to identify interpretable symbolic laws that can be analyzed and reused. Existing PDE discovery methods often depend on hand-designed libraries, accurate numerical derivatives, or per-equation symbolic search, making them fragile under sparse observations, noisy data, broad equation families, and unseen symbolic compositions. Here, we introduce FoundPDE, a pretrained data-to-symbol foundation model for PDE discovery. FoundPDE maps discretized spatio-temporal trajectories to symbolic equations with a transformer, uses Fourier and first-order spatial-derivative augmentation to expose complementary trajectory information, refines numerical coefficients, and supports downstream extension with improved generalization through few-shot fine-tunin
    
[^237]: 基于动力系统启发的面向AI4Science任务的连续大语言模型

    Dynamic System Inspired Continuous Large Language Model for AI4Science Tasks

    [https://arxiv.org/abs/2610.26959](https://arxiv.org/abs/2610.26959)

    Continuum-LM将预训练的残差变换复用为动力系统算子，构建了通过常微分方程演化隐藏状态的连续深度语言模型，在不扩展主干网络的前提下增加了潜在计算能力，在八项科学任务上将平均精确匹配准确率显著提升了7.77至10.96个百分点。

    

    科学基础模型日益需要处理连续现象，但它们仍然使用离散的残差计算。我们提出这样的问题：预训练的残差变换能否在不扩展主干网络的情况下支持额外的潜在计算。为验证这一可能性，我们提出了Continuum-LM，它将这些残差变换复用为连续深度语言模型中的动力学算子。其隐藏状态通过常微分方程进行演化，并采用高阶数值积分进行求解，其中求解器的阶数和步长控制着计算量。在八项科学任务上，Continuum-LM训练稳定，在0.6B和1.7B参数规模下，平均精确匹配准确率分别比离散对应模型提升了7.77和10.96个百分点。在0.6B规模下，在匹配的参数量和残差函数评估预算下，它还优于Ouro风格的循环控制模型。这些结果共同支持了可扩展的计算……（摘要在此处截断）

    arXiv:2610.26959v1 Announce Type: cross  Abstract: Scientific foundation models increasingly operate on continuous phenomena, yet still use discrete residual computation. We ask whether pretrained residual transformations can support additional latent computation without expanding the backbone. To test this possibility, we introduce Continuum-LM, which reuses these transformations as dynamical operators in a continuous-depth language model. Its hidden states evolve through an ordinary differential equation evaluated by higher-order numerical integration, with solver order and step size controlling computation. Across eight scientific tasks, Continuum-LM trains stably and improves mean exact-match accuracy by 7.77 and 10.96 percentage points over discrete counterparts at the 0.6B and 1.7B scales, respectively. At 0.6B, it also outperforms an Ouro-style recurrent control under matched parameter and residual-function evaluation budgets. Together, these results support extensible computati
    
[^238]: 超越策略差异：能力感知的多教师在线策略蒸馏

    Beyond Policy Discrepancy: Capability-Aware Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2610.26950](https://arxiv.org/abs/2610.26950)

    提出能力感知的多教师在线策略蒸馏方法 CA-MOPD，将强化学习专家模型合并与验证器估计的师生能力差距相结合，根据学生已掌握的内容动态调整各领域蒸馏强度，克服了仅基于策略差异分配监督的局限性。

    

    多教师在线策略蒸馏（MOPD）将领域专家模型整合到一个共享的学生模型中，但如何确定每个教师应给予多强的监督仍然是一个挑战。现有的基于差异的分配方法，即使学生已经达到或超过教师的任务性能，也可能分配强监督，因为行为上的不匹配并不一定意味着还存在能力缺陷。我们提出能力感知的多教师在线策略蒸馏（CA-MOPD），它将专家模型合并与能力引导的分配相结合，以考虑学生已经学到的内容。CA-MOPD 首先将强化学习专家模型合并为强大的学生模型初始化，然后使用验证器估计的师生能力差距来调整领域级别的蒸馏强度。随着能力差距的缩小，监督强度降低；当估计的学生能力出现退化时，监督强度增加，同时结合密集的 token 级别指导……

    arXiv:2610.26950v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) integrates domain specialists into a shared student, but determining how strongly each teacher should supervise remains challenging. Existing discrepancy-based allocation can assign strong supervision even when the student already matches or exceeds the teacher's task performance, since behavioral mismatch does not necessarily imply a remaining capability deficit. We propose Capability-Aware Multi-Teacher On-Policy Distillation (CA-MOPD), which combines specialist merging with capability-guided allocation to account for what the student has already learned. CA-MOPD first merges reinforcement-learning specialists into a strong student initialization, then uses verifier-estimated teacher--student capability gaps to adjust domain-level distillation strength. Supervision decreases as capability gaps close and increases when estimated student capability regresses, while dense token-level guidance 
    
[^239]: 未知边缘分布下严格可复制可实现学习的二次样本复杂度

    Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals

    [https://arxiv.org/abs/2610.26944](https://arxiv.org/abs/2610.26944)

    该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。

    

    在量词∃A∀D的框架下，即可实现分布及其边缘分布均未知的情况下，我们确定了严格可复制性（即两个独立样本和一个共享随机种子必须产生完全相同的函数）的有限类样本复杂度指数。当ε=δ=ρ=10⁻⁴时，在|H|≤M上的最坏情况复杂度为Θ((log M)²)，且不含迭代对数项。对于d≥1、0<ε≤1/16以及0<δ,ρ≤10⁻³，任何在整个显式锚定可实现类上有效的随机学习器（即使允许非proper学习）都需要Ω(d²/(ερ²))个样本。对于完整布尔类P_d（d≥2），在相同的参数范围内，我们借助一个proper的泊松化球采样算法证明了匹配的刻画结果Θ(d²/(ερ²)+ε⁻¹log(1/δ))。此外，我们还为每个有限可测布尔类给出了显式的proper Gibbs算法上界。

    arXiv:2610.26944v1 Announce Type: cross  Abstract: Under the quantifier $\exists A\,\forall D$, where the realizable distribution and its marginal are unknown, we determine the finite-class sample-complexity exponent for strict replicability: two independent samples and one shared seed must produce identical functions. For $\varepsilon=\delta=\rho=10^{-4}$, the worst case over $|H|\le M$ is $\Theta((\log M)^2)$, without iterated logarithms. For $d\ge1$, $0<\varepsilon\le1/16$, and $0<\delta,\rho\le10^{-3}$, every randomized, possibly improper learner valid throughout an explicit anchored realizable family requires $\Omega(d^2/(\varepsilon\rho^2))$ samples. For the full Boolean class $P_d$, $d\ge2$, in the same parameter range, we prove the matching characterization $\Theta(d^2/(\varepsilon\rho^2)+\varepsilon^{-1}\log(1/\delta))$ using a proper Poissonized ball-sampling algorithm. We also give an explicit proper Gibbs upper bound for every finite measurable Boolean class. Our lower boun
    
[^240]: 专家引导的评分标准作为非可验证领域强化学习的密集奖励信号

    Expert-Guided Rubrics as Dense Reward Signals for Reinforcement Learning in Non-Verifiable Domains

    [https://arxiv.org/abs/2610.26932](https://arxiv.org/abs/2610.26932)

    本文提出将少量固定的主题专家判断转化为基于评分标准的密集奖励信号，并结合检索指标解决GRPO在非可验证领域中因稀疏奖励导致优化停滞的问题。

    

    arXiv:2610.26932v1 公告类型：交叉 摘要：大语言模型（LLM）正越来越多地被部署为金融服务等非可验证领域中的自主智能体，在这些领域中，它们必须解释用户意图、重写查询，并在特定领域约束下检索信息。尽管强化学习（RL）可以通过行动的后果来训练此类智能体，但近期的成功主要依赖于具有自动计算二元奖励的可验证任务。在非可验证的检索任务中，评估查询质量需要稀缺的主题专家（SME）判断，而这种判断本质上是分级的而非二元的。在使用组相对策略优化（GRPO）时，当组内所有采样输出都获得相同结果时，稀疏奖励无法产生有用的学习信号，导致优化停滞。我们提出了一种方法，将少量固定的SME判断投入转化为密集奖励，并将其与基于检索的指标相结合，形成一种混合的（摘要在此处被截断）

    arXiv:2610.26932v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as autonomous agents in non-verifiable domains such as financial services, where they must interpret user intent, rewrite queries, and retrieve information under domain-specific constraints. Although reinforcement learning (RL) can train such agents from the consequences of their actions, recent successes largely rely on verifiable tasks with automatically computed binary rewards. In non-verifiable retrieval tasks, assessing query quality requires scarce subject-matter expert (SME) judgment that is inherently graded rather than binary. With Group Relative Policy Optimization (GRPO), sparse rewards yield no useful learning signal when all sampled outputs within a group receive the same outcome, causing optimization to stall. We introduce a method that converts a small, fixed investment of SME judgment into a dense reward and combines it with retrieval-based metrics to form a hybrid,
    
[^241]: ShlokaDhara：一个用于在“文本到诵读”中学习格律感知韵律的多说话人梵语语音语料库

    ShlokaDhara: A Multi-Speaker Sanskrit Speech Corpus for Learning Meter-Aware Prosody in Text-to-Recitation

    [https://arxiv.org/abs/2610.26929](https://arxiv.org/abs/2610.26929)

    该论文发布了 43 小时的多说话人梵语诵读语料库 ShlokaDhara，用以支持要求格律正确性的“文本到诵读”合成任务，并通过 16 位梵语专家的主观评估与新颖实证指标验证了 XTTS-v2 在该任务上达到最高的感知质量。

    

    梵语诗歌的诵读受"Chandas"（诗律）的支配，这是一个由节奏性的 Laghu（轻音节）和 Guru（重音节）所定义的严格韵律框架。由于现有的梵语语音数据集主要面向自动语音识别（ASR）或对话式语音合成（TTS），它们缺乏真实诗歌合成所需的格律保真度。我们推出了 ShlokaDhara，一个 43 小时、多说话人的梵语语音语料库，涵盖了多样的格律、诵读风格与旋律变化。我们特别关注 Text-to-Recite（文本到诵读）这一合成任务，该任务明确要求同时具备韵律准确性与格律正确性。通过自动化的 UTMOS（神经平均意见分）预测器对最先进的 TTS 架构进行基准测试，结果表明 XTTS-v2 实现了最高的感知质量。我们进一步采用两种范式对 XTTS-v2 生成的样本进行验证：由 16 位梵语专家开展的主观评估，以及新颖的实证指标，特别是……

    arXiv:2610.26929v1 Announce Type: cross  Abstract: Sanskrit verse recitation is governed by $\textit{Chandas}$, a strict prosodic framework defined by rhythmic $\textit{Laghu}$ (light) and $\textit{Guru}$ (heavy) syllables. Because existing Sanskrit speech datasets prioritize ASR or conversational TTS, they lack the metrical fidelity required for authentic verse synthesis. We introduce $\textbf{\textit{ShlokaDhara}}$, a 43-hour, multi-speaker Sanskrit speech corpus capturing diverse meters, recitation styles, and melodic variations. We specifically focus on $\textit{Text-to-Recite}$, a synthesis task explicitly requiring both prosodic accuracy and metrical correctness. Benchmarking state-of-the-art TTS architectures via an automated UTMOS (neural MOS) predictor demonstrates that XTTS-v2 achieves the highest perceptual quality. We further validate the XTTS-v2 generated samples using two paradigms: subjective assessments by 16 Sanskrit experts and novel empirical metrics, specifically, s
    
[^242]: CaMeWM：用于生成式肿瘤演化与治疗规划的因果医学世界模型

    CaMeWM: A Causal Medical World Model for Generative Tumor Evolution and Treatment Planning

    [https://arxiv.org/abs/2610.26923](https://arxiv.org/abs/2610.26923)

    CaMeWM是一个因果医学世界模型，通过“干预到决策”的单一循环机制，为具有治疗支撑的TACE术后肿瘤演化生成反事实预测分布，从而实现对不同治疗方案的有效比较与规划。

    

    治疗规划本质上是反事实问题：临床医生必须比较患者在多种竞争性治疗方案下肿瘤和生存情况将如何演化，而回顾性队列数据仅记录了患者实际接受的治疗方案和一次随访扫描。因此，基于此类数据训练的治疗条件生成器只能复现历史上的治疗分配模式，而无法模拟对治疗方案进行排序所必需的干预性疾病动态。我们提出了CaMeWM，一个用于有经验支撑的TACE术后肿瘤演化与治疗规划的因果医学世界模型。CaMeWM实现了一个从干预到决策的完整循环，而非三个独立的预测任务：它首先仅在临床可比且具有经验性治疗支撑的病史中接受所查询的治疗方案；然后通过动作条件响应描述符、确定性锚点和残差3D扩散转换，生成病灶局部随访世界的分布；并且它……（摘要在此处被截断）

    arXiv:2610.26923v1 Announce Type: cross  Abstract: Treatment planning is intrinsically counterfactual: clinicians must compare how a patient's tumor and survival would evolve under competing regimens, whereas a retrospective cohort records only the regimen received and one follow-up scan. A treatment-conditioned generator trained on such data can therefore reproduce historical treatment allocation rather than simulate the interventional disease dynamics needed to rank regimens. We introduce \textbf{CaMeWM}, a causal medical world model for supported post-TACE tumor evolution and treatment planning. CaMeWM implements one intervention-to-decision loop rather than three independent prediction tasks: it first admits a queried regimen only for clinically comparable histories with empirical treatment support; it then generates a distribution of lesion-local follow-up worlds through an action-conditioned response descriptor, deterministic anchor, and residual 3D diffusion transition; and it f
    
[^243]: 基于匿名群体反馈的可引导多元化策略对齐

    Steerable Pluralistic Policy Alignment from Anonymized Crowd Feedback

    [https://arxiv.org/abs/2610.26914](https://arxiv.org/abs/2610.26914)

    提出P3L框架，无需标注者身份信息，通过PCA分解从匿名群体反馈中联合学习轨迹嵌入与偏好方向共享基，实现可引导、可泛化的个性化策略对齐。

    

    多元化对齐要求AI系统能够从偏好可能反映异构甚至相互冲突目标的多元人群中学习，并使其行为能够针对个体用户进行个性化。现有的多元化对齐方法通常假设可以获取标注者的身份信息，并依赖强大的预训练表示来建模个体偏好差异。我们提出了偏好PCA多元化学习，这是一种新颖的多元化对齐框架，它通过PCA分解从异构的群体反馈中联合学习轨迹嵌入和偏好方向的共享基，而无需标注者身份信息。在测试时，我们的方法能够实现快速且准确的个性化奖励学习和下游策略优化。在多个具身控制任务上的实验表明，P3L实现了可引导的个性化对齐，其泛化能力超越了离线数据集，可推广至分布外场景。

    arXiv:2610.26914v1 Announce Type: cross  Abstract: Pluralistic alignment requires AI systems to learn from diverse populations whose preferences may reflect heterogeneous or even conflicting objectives, and to personalize their behavior to individual users. Existing approaches in pluralistic alignment often assume access to annotator identities and rely on powerful pretrained representations to model individual preference variations. We propose Preference-PCA Pluralistic Learning (P3L), a novel framework for pluralistic alignment that jointly learns trajectory embeddings and the shared basis of preference directions via PCA decomposition from heterogeneous crowd feedback without requiring annotator identities. At test time, our approach enables rapid and accurate personalized reward learning and downstream policy optimization. Experiments across multiple embodied control tasks demonstrate that P3L enables steerable personal alignment that generalizes beyond the offline dataset to out-o
    
[^244]: LatticeMind：一个用于密度泛函理论工作流的验证门控智能体框架

    LatticeMind: A Validation-Gated Agentic Framework for Density Functional Theory Workflows

    [https://arxiv.org/abs/2610.26869](https://arxiv.org/abs/2610.26869)

    LatticeMind是一个验证门控的智能体框架，通过工作流契约、确定性验证器和有界修复机制，将自然语言请求转换为可执行且可检查的密度泛函理论（DFT）项目，解决了语言模型在串联多阶段第一性原理计算工作流时的可靠性问题。

    

    第一性原理计算很少是单一求解器调用：它们是多阶段工作流，其中结构、设置和保存的密度必须在相互依赖的阶段之间正确传递。语言模型可以写出有效的单一输入文件，但一旦阶段被串联起来，可靠性就会崩溃：遗漏的依赖关系、错误的文件交接或幻觉产生的关键字即使求解器正常运行也可能使结果无效，而这类错误只有在代价高昂的计算执行之后才会显现。我们提出了LatticeMind，一个验证门控的智能体框架，它将自然语言请求转换为可执行、可检查的密度泛函理论（DFT）项目。工作流契约规定了所需的阶段、文件和依赖关系；确定性验证器在任何求解器启动之前对项目进行检查；失败的检查会触发有界修复；并且报告保持可追溯到项目文件。在一个涵盖总能量、态密度、能带结构、磁性……等性质的44个案例RESCU基准测试中……（摘要截断）

    arXiv:2610.26869v1 Announce Type: cross  Abstract: First-principles calculations are rarely single solver calls: they are multistage workflows in which structures, settings, and saved densities must pass correctly between dependent stages. Language models can write a valid single input, but reliability collapses once stages are chained: a dropped dependency, wrong file handoff, or hallucinated keyword can invalidate results even when the solver runs, and such errors surface only after costly execution. We introduce LatticeMind, a validation-gated agentic framework that converts natural-language requests into executable, checkable density functional theory (DFT) projects. A workflow contract specifies required stages, files, and dependencies; deterministic validators inspect the project before any solver launch; failed checks trigger bounded repair; and reports remain traceable to project files. On a 44-case RESCU benchmark spanning total-energy, density-of-states, band-structure, magne
    
[^245]: 高维层状态轨迹揭示Transformer推理的内部动力学

    High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference

    [https://arxiv.org/abs/2610.26866](https://arxiv.org/abs/2610.26866)

    该研究将Transformer逐层推理的有序层状态轨迹作为经验对象，发现真实层顺序产生的轨迹对齐度是保持终点随机打乱的1.60至2.24倍，证明层顺序构成一种与架构兼容而非训练特有的可测量计算，并可通过局部转换执行器实现对内部推理轨迹的选择性控制。

    

    大语言模型的机制研究将功能定位于参数、注意力头或神经元之中，但这些部件并不能说明一次推理是如何逐步展开的。我们将有序的层状态轨迹视为经验过程，并检验层顺序是否定义了一种可测量的计算。在代表性的Qwen、Llama和Gemma检查点上，实际实现的层顺序所产生的路径与其终点弦的对齐程度是保持终点的随机打乱顺序的1.60至2.24倍。这种顺序效应可以迁移到词汇任务中，但在同一Qwen架构的三个随机初始化模型中表现更强，这表明它是一种与架构兼容的支架结构，而非训练特有的规律。直接隐藏状态比词汇回读提供了更强的机制坐标，而任务定义的坐标使得候选词之间的竞争变得可计算。在加性干预失败之后，局部转换执行器能够选择性地控制内部轨迹。

    arXiv:2610.26866v1 Announce Type: cross  Abstract: Mechanistic studies of large language models localize function in parameters, attention heads or neurons, but those parts do not specify how one inference unfolds. We treat the ordered layer-state trajectory as the empirical process and test whether layer order defines a measurable computation. Across representative Qwen, Llama and Gemma checkpoints, realized order produced paths 1.60--2.24 times as aligned with their endpoint chord as endpoint-preserving shuffles. The order effect transferred to a lexical task, but was stronger in three random initializations of the same Qwen architecture, identifying an architecture-compatible scaffold rather than a training-specific law. Direct hidden states provided stronger mechanism coordinates than vocabulary readbacks, and a task-defined coordinate made candidate competition computable. After additive interventions failed, local transition actuators selectively controlled internal trajectories 
    
[^246]: H2DF：用于可控语言模型遗忘的校准目标区间

    H2DF: Calibrated Target Bands for Controllable Language-Model Unlearning

    [https://arxiv.org/abs/2610.26864](https://arxiv.org/abs/2610.26864)

    H2DF提出了一种校准的目标区间目标函数，通过数据依赖的目标带控制梯度上升压力，在遗忘达到目标后自动抑制或反转压力，从而实现可控的语言模型遗忘，同时保护保留数据的效用。

    

    梯度上升（GA）是一种简单的语言模型遗忘基线方法，但在有用的遗忘已经发生后，它仍会持续增加遗忘损失。这种无界的压力会损害保留数据的效用，并使被遗忘的样本出现异常高的损失。我们提出了H2DF，一种校准的目标区间目标函数，它在低于数据相关的下目标时施加类似GA的压力，并在达到目标区域后抑制或反转该压力。这些目标是在训练前从匹配的非成员样本中估计得到的，因此基础目标函数在每个优化步骤中仅需一次遗忘批次的前向/反向传播；我们报告的TOFU变体每十步增加一次稀疏的保留数据回放。我们为TOFU问答任务实例化了一个单侧目标，为MUSE-News语言建模任务实例化了一个双侧目标。在三个TOFU随机种子上，H2DF取得了遗忘NLL为1.528（裕度0.049）、保留NLL为1.165（裕度0.047）的结果，与GradDiff相当。

    arXiv:2610.26864v1 Announce Type: cross  Abstract: Gradient ascent (GA) is a simple language-model unlearning baseline, but it continues increasing forget loss after useful forgetting has occurred. This unbounded pressure can damage retained utility and make forgotten examples abnormally high-loss. We introduce H2DF, a calibrated target-band objective that applies GA-like pressure below a data-dependent lower target and suppresses, or reverses, that pressure after the target region is reached. The targets are estimated from matched non-member examples before training, so the base objective requires one forget-batch forward/backward pass per optimization step; our reported TOFU variant adds sparse retain replay every tenth step. We instantiate a one-sided objective for TOFU question answering and a two-sided objective for MUSE-News language modeling. Across three TOFU seeds, H2DF obtains a forget NLL of 1.528 with margin 0.049 and retain NLL of 1.165 with margin 0.047, matching GradDiff
    
[^247]: 无需语言预训练的文本到图像生成建模

    Text-to-Image Generative Modeling Without Language Pretraining

    [https://arxiv.org/abs/2610.26863](https://arxiv.org/abs/2610.26863)

    本文提出完全从零预训练的早期融合多模态自回归模型家族Halcyon，证明无需互联网规模的语言预训练也能以大幅更少的计算量训练出强大且高质量的文本到图像生成模型。

    

    现代文本到图像（T2I）系统使用互联网规模的文本语言预训练来实现文本条件下的图像生成。我们探究语言预训练对于强大的T2I生成是否必要。我们提出了Halcyon，这是一系列早期融合的多模态自回归模型，完全从零开始在GPIC（一个包含1亿图像-文本对的数据集）上预训练。Halcyon-0.52B以少18.2倍的预训练计算量超越了LlamaGen-XL/8（48.0 对比 63.3 gFD）。此外，Halcyon-1.6B以少275倍的预训练计算量超越了JiT-T2I-XXL/8++（35.3 对比 41.0 gFD）。对照实验表明，仅文本的预训练并不能改善T2I生成。Halcyon还通过用图像到文本（I2T）监督替代caption dropout来学习图像描述生成，并在相同的预训练预算下保持了有竞争力的T2I生成质量。我们证明了强大的T2I系统可以在没有互联网规模语言预训练的情况下从零开始训练。代码和模型已开源。

    arXiv:2610.26863v2 Announce Type: replace  Abstract: Modern text-to-image (T2I) systems use internet-scale language pretraining for text-conditioned image generation. We ask whether language pretraining is necessary for strong T2I generation. We introduce Halcyon, a family of early-fusion multimodal autoregressive models pretrained entirely from scratch on GPIC, a dataset of 100M image-text pairs. Halcyon-0.52B outperforms LlamaGen-XL/8 (48.0 vs. 63.3 gFD) with 18.2$\times$ less pretraining compute. Further, Halcyon-1.6B outperforms JiT-T2I-XXL/8++ (35.3 vs. 41.0 gFD) with 275$\times$ less pretraining compute. Controlled experiments show that text-only pretraining does not improve T2I generation. Halcyon also learns image captioning by replacing caption dropout with I2T supervision, maintaining competitive T2I generation quality under the same pretraining budget. We show that strong T2I systems can be trained from scratch without internet-scale language pretraining. Code and models are
    
[^248]: WeAct：将动态权重切换作为智能体动作

    WeAct: Dynamic Weight Switching as Agent Actions

    [https://arxiv.org/abs/2610.26847](https://arxiv.org/abs/2610.26847)

    WeAct在冻结主干上训练独立的能力专用LoRA适配器，并将适配器选择纳入智能体动作空间，使模型能在长时程任务中通过原生工具调用接口按需组合能力，同时避免灾难性遗忘。

    

    长时程智能体必须从交错的轨迹中学习异构能力。更新共享参数以学习新能力可能会导致先前获得的能力退化，从而造成灾难性遗忘。随着轨迹的展开，不同阶段可能需要不同的能力，然而传统的联合训练和静态适配器合并在整个执行过程中都维持固定的参数配置。我们提出WeAct（权重级内部动作），它在冻结的主干模型上训练独立的能力专用LoRA适配器，并将适配器选择添加到智能体的动作空间中。这些内部动作利用模型原生的工具调用接口，在共享上下文中按需组合能力。在八个深度研究基准测试中，配备WeAct的Qwen3-32B相比全量微调的对应模型，在使用更少计算量的情况下平均准确率提升6.2%，相比最强的静态方法提升27.5%。

    arXiv:2610.26847v2 Announce Type: replace-cross  Abstract: Long-horizon agents must learn heterogeneous capabilities from interleaved trajectories. Updating shared parameters to learn new capabilities can degrade previously acquired ones, resulting in catastrophic forgetting. As a trajectory unfolds, different stages may require different capabilities, yet conventional joint training and static adapter merging maintain a fixed parameter configuration throughout execution. We introduce WeAct (Weight-level Internal Actions), which trains independent capability-specific LoRA adapters over a frozen backbone and adds adapter selection to the agent's action space. These internal actions use the model's native tool-calling interface for on-demand capability composition in a shared context. Across eight deep-research benchmarks, Qwen3-32B with WeAct improves average accuracy by 6.2% relative to full fine-tuning counterparts despite using less compute, and by 27.5% relative to the strongest sta
    
[^249]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^250]: HarnessPro：基于帕累托优化的递归式智能体框架自我改进方法，实现性能-成本权衡

    HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs

    [https://arxiv.org/abs/2610.26770](https://arxiv.org/abs/2610.26770)

    HarnessPro提出了一种无需标签的智能体框架递归自我改进方法，通过帕累托优化权衡性能与执行成本，并提供精简、均衡、重型三种框架版本以适配不同的部署预算。

    

    智能体框架的开发在构建强大的AI智能体中起着至关重要的作用。近期的方法通过递归自我改进（RSI）来优化框架，但其性能提升往往伴随着更高的执行成本。我们提出了HarnessPro，这是一种无需标签的方法，用于演化智能体框架，在提升任务性能的同时降低执行成本。在该设置下，候选框架通过LLM生成的测试进行评估，无需访问真实标准答案或官方测试。专门的智能体负责诊断执行轨迹，根据不同的性能-成本优先级规划更新方案，并将其实现为候选框架。HarnessPro并非返回单一框架，而是采用归一化加权平方距离选择方法，为不同的部署预算提供侧重成本的精简版、均衡的中等版和侧重性能的重型版框架。被选中的框架进入下一轮迭代，执行反馈在此过程中持续引导改进……

    arXiv:2610.26770v2 Announce Type: replace-cross  Abstract: Harness development plays a crucial role in building capable AI agents. Recent approaches improve harnesses through recursive self-improvement (RSI), but their performance gains often come with higher execution costs. We introduce HarnessPro, a label-free method for evolving harnesses to improve task performance while reducing execution cost. In this setting, candidate harnesses are evaluated using LLM-generated tests without access to ground-truth solutions or official tests. Specialized agents diagnose execution trajectories, plan updates with different performance-cost priorities, and implement them as candidate harnesses. Rather than returning a single harness, HarnessPro uses normalized weighted squared-distance selection to provide cost-focused (lite), balanced (medium), and performance-focused (heavy) harnesses for different deployment budgets. Selected harnesses enter the next iteration, where execution feedback guides 
    
[^251]: 通过权重空间重参数化重新思考大语言模型微调：在下游适应过程中保持安全性

    Rethinking LLM Fine-Tuning via Weight Space Reparameterization: Preserving Safety during Downstream Adaptation

    [https://arxiv.org/abs/2610.26443](https://arxiv.org/abs/2610.26443)

    提出WSR-Tune框架，通过构建安全条件化的基对权重矩阵进行重参数化，使安全信息集中于少数方向并冻结这些安全关键方向，从而在大语言模型下游任务微调中保持安全对齐。

    

    在下游任务上对大语言模型（LLM）进行微调往往会削弱其此前对齐的安全行为，从而在任务适应与安全保持之间造成固有的权衡。现有方法试图通过在原始参数空间中识别与安全相关的神经元、层或更新方向来缓解这一问题。然而，安全相关信息往往分布在该空间的众多方向中，导致下游更新覆盖了编码安全行为的参数。我们提出了WSR-Tune，一个基于权重空间重参数化的微调框架，能够在下游适应过程中保持安全性。我们的方法构建了一个以安全为条件的基，并对权重矩阵进行重参数化，使安全相关信息集中于少数方向上。随后我们识别出安全关键方向并在重参数化空间中将其冻结，同时更新……（原文摘要到此截断）

    arXiv:2610.26443v2 Announce Type: replace  Abstract: Fine-tuning large language models (LLMs) on downstream tasks often weakens their previously aligned safety behavior, creating an inherent trade-off between task adaptation and safety preservation. Existing approaches attempt to mitigate this issue by identifying safety-relevant neurons, layers, or update directions in the original parameter space. However, safety-related information is often distributed across many directions in this space, causing downstream updates to overwrite parameters that encode safety behavior. We propose WSR-Tune, a Weight Space Reparameterization-based fine-tuning framework that preserves safety during downstream adaptation. Our approach constructs a safety-conditioned basis and reparameterizes the weight matrices such that safety-relevant information becomes concentrated in a small subset of directions. We then identify safety-critical directions and freeze them in the reparameterized space, while updating
    
[^252]: 一种简单而通用的top-$k$选择方法

    A Simple and General Approach to top-$k$ Selection

    [https://arxiv.org/abs/2610.26407](https://arxiv.org/abs/2610.26407)

    本文提出一种通过最小化期望机会成本上界导出的替代目标来自适应分配采样批次的简单通用方法，用于解决贝叶斯top-$k$选择问题，并证明其后验期望机会成本的衰减速率至少达到任意采样规则最佳速率的一半。

    

    本文研究了贝叶斯top-$k$识别问题，其中决策者的任务是从一组质量未知的物品中选出$k$个物品。为了评估这些物品，决策者需要自适应地分配固定的采样预算，每次对某个物品的采样都会产生对其质量的一个含噪观测，目标是最小化所选集合的期望机会成本。我们提出了一种方法，通过最小化由期望机会成本上界导出的替代目标来分配每一批采样。该方法具有通用性：采样可以按批次进行分配，选择数量$k$可以是任意的，并且适用于多种采样分布。该方法同时也简单直观。我们证明了该方法的后验期望机会成本以指数速率衰减，其速率至少达到任何采样规则所能达到的最佳速率的一半。在广泛的仿真设置中……

    arXiv:2610.26407v2 Announce Type: replace  Abstract: This paper considers the Bayesian top-$k$ identification problem, wherein a decision maker is tasked with selecting $k$ items from a pool of items of unknown quality. To assess the items, the decision maker adaptively allocates a fixed sampling budget, where each sample of an item yields a noisy observation of its quality, with the objective of minimizing the expected opportunity cost of the selected set. We propose an approach that allocates each batch of samples by minimizing a surrogate objective derived from an upper bound on the expected opportunity cost. The approach is general: samples can be allocated in batches, the selection size $k$ is arbitrary, and it applies to many sampling distributions. It is also simple and intuitive. We prove that the method's posterior expected opportunity cost decays at an exponential rate of at least half the best rate achievable by any sampling rule. In simulations across a wide range of settin
    
[^253]: 借助人工评估者选出最优少数：自适应地结合评分与两两比较

    Selecting the Best Few with Human Evaluators: Adaptively Combining Ratings and Pairwise Comparisons

    [https://arxiv.org/abs/2610.26380](https://arxiv.org/abs/2610.26380)

    本文提出一种自适应评估分配方法，先利用人工评分对候选项目进行粗筛，再借助两两比较精化优秀项目的排序，从而在有限评估预算下高效地从大量候选中选出最优的 $k$ 个项目。

    

    本文研究这样一个决策者问题：需要从大量候选者（如求职者、大语言模型、奖项提名人或新产品创意）中识别出最优的 $k$ 个项目。决策者通过征集人工评估来了解各项目的质量。由于每次评估都可能成本高昂，决策者希望通过自适应地分配评估来高效利用有限的评估预算。一个关键考虑因素是所征集的评估形式，这种形式应当是评估者乐于提供的。两种流行的形式是评分和两两比较，我们对两者都加以考虑，因为在预算有限的情况下，二者的结合可能尤其有用。评分可以在早期使用，以大致了解哪些项目好、哪些项目不好。随着这一图景逐渐成形，两两比较可以细化优秀项目之间的排序，从而聚焦于最终的 top-$k$ 选择。为了解决这一分配问题，

    arXiv:2610.26380v2 Announce Type: replace-cross  Abstract: This paper considers a decision-maker tasked with identifying the $k$ best items from a large pool of candidates, such as job applicants, LLMs, award nominees, or new product ideas. The decision-maker learns about the items' quality by soliciting human evaluations. Each evaluation can be costly, so the decision-maker aims to use the limited evaluation budget efficiently by allocating evaluations adaptively. A key consideration is the form of evaluation to solicit, which should be something evaluators are comfortable providing. Two popular forms are ratings and pairwise comparisons, and we allow for both, as they may be especially useful in combination given the limited budget. Ratings can be used early on to get a coarse picture of which items are good and which are not. As that picture forms, comparisons can refine the ordering of the good items, narrowing in on the ultimate top-$k$ selection. To solve the allocation problem, 
    
[^254]: 截断与否？截断反向传播下循环Transformer的缩放律研究

    Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation

    [https://arxiv.org/abs/2610.26030](https://arxiv.org/abs/2610.26030)

    通过近600次训练拟合的统一缩放律表明，在联合优化前向和后向循环次数时，完整BPTT训练的循环Transformer始终优于相同算力下使用截断BPTT的训练方式。

    

    循环Transformer通过复用共享的层块来增加计算深度，而无需增加可训练参数的数量。它们通过时间反向传播（BPTT）进行训练。为了节省训练算力，BPTT通常被截断为只传播最后几次循环，但截断会产生有偏梯度，这可能削弱循环带来的好处。在这项工作中，我们以系统化的方式研究了截断在循环Transformer训练中的价值：我们联合变化模型规模、训练token数量以及前向和后向循环次数，进行了近600次训练运行，并拟合出一个统一的缩放律，通过有效参数刻画循环与截断的影响。尽管截断能够节省训练算力，但在联合优化前向和后向循环次数时，我们拟合的缩放律更倾向于完整的BPTT。在算力匹配的实验中，采用更少循环次数的完整BPTT始终优于相同算力下的截断设置。

    arXiv:2610.26030v2 Announce Type: replace  Abstract: Looped transformers reuse a shared block of layers to increase computational depth without increasing the number of trainable parameters. They are trained with backpropagation through time (BPTT). To save training compute, BPTT is often truncated to the last few loops, but truncation yields biased gradients that may weaken the benefit of looping. In this work, we study the value of truncation in looped transformer training in a systematic way: we jointly vary model size, training tokens, and forward and backward loop counts across nearly 600 training runs and fit a unified scaling law that captures the effects of looping and truncation through effective parameters. Despite the training-compute savings of truncation, our fitted law favors full BPTT when forward and backward loop counts are jointly optimized. In matched-compute runs, full BPTT through fewer loops consistently outperforms truncated settings of the same compute. The law 
    
[^255]: 高斯输入下正ReLU网络的多项式学习

    Polynomial Learning of Positive ReLU Networks under Gaussian Inputs

    [https://arxiv.org/abs/2610.25116](https://arxiv.org/abs/2610.25116)

    该论文在不对隐藏方向施加任何分离或条件化假设的前提下，首次为标准高斯输入下正ReLU网络的学习给出了样本与实数算术复杂度均为多项式的算法，并提供了误差与风险的理论保证。

    

    我们研究了标准高斯输入下正的、无偏置的单隐层ReLU网络。我们获得了具有多项式样本复杂度与实数算术复杂度的算法，且无需对隐藏方向施加分离或条件化假设。当条件标签均值是一个总输出权重不超过1、宽度为 $k$ 的网络时，学习器能够以不超过 $k+\min\{k,d\}$ 的输出宽度实现 $\varepsilon$ 的平方预测误差，其复杂度与1加上残差二阶矩成线性关系。对于二阶矩有界的任意标签，我们构造出宽度为 $O(k\log(2/\varepsilon))$ 的正ReLU网络，其风险由不可约噪声方差、加上 $Ck^3\log^{9/2}(2k/\varepsilon)$ 乘以条件均值到目标类距离的平方、再加上 $\varepsilon$ 所界定。证明结合了隐式Hermite矩、正交张量列车压缩，以及一个方向解码器，该解码器以一定概率返回直接的……（摘要在此处被截断）

    arXiv:2610.25116v2 Announce Type: replace  Abstract: We study positive, bias-free, one-hidden-layer ReLU networks under standard Gaussian inputs. We obtain algorithms with polynomial sample and real-arithmetic complexity, without separation or conditioning assumptions on the hidden directions. When the conditional label mean is a width-$k$ network with total output weight at most one, the learner achieves squared prediction error $\varepsilon$ with output width at most $k+\min\{k,d\}$ and complexity linear in one plus the residual second moment. For arbitrary labels with bounded second moment, we obtain a positive ReLU network of width $O(k\log(2/\varepsilon))$. Its risk is bounded by the irreducible noise variance plus $Ck^3\log^{9/2}(2k/\varepsilon)$ times the squared distance of the conditional mean to the target class, plus $\varepsilon$. The proofs combine implicit Hermite moments, orthogonal tensor-train compression, and a direction decoder whose probability of returning a direct
    
[^256]: 标注数据稀缺下面向过程预测的过程感知事件日志增强

    Process-Aware Event Log Augmentation for Process Prediction under Labeled Data Scarcity

    [https://arxiv.org/abs/2610.24777](https://arxiv.org/abs/2610.24777)

    该论文提出SiamSA-PPM自监督框架，将孪生学习与利用控制流语义和频繁行为模式的过程感知数据增强相结合，以解决预测性过程监控中标注数据稀缺和日志变异性低的问题，在下一活动和最终结果预测上取得了优于现有方法的性能。

    

    预测性过程监控（PPM）基于事件日志对正在进行的业务过程的未来事件进行预测。然而，深度学习PPM方法往往受限于标注数据的稀缺性以及真实世界日志的低变异性。为了解决这一问题，我们提出了SiamSA-PPM，这是一个将孪生学习与基于统计的数据增强相结合的自监督框架。该框架采用新颖的变换方法，利用控制流语义和频繁行为模式来生成逼真的、语义有效的轨迹变体，从而在无需标注监督的情况下学习可泛化的前缀表示。在具有广泛变异性的真实世界事件日志上进行了大量实验，并结合详尽的超参数调优，结果证明了这种过程感知增强方法的价值。与最先进的基线方法相比，SiamSA-PPM在下一活动预测和最终结果预测方面均取得了具有竞争力或更优的预测性能。

    arXiv:2610.24777v2 Announce Type: replace  Abstract: Predictive Process Monitoring (PPM) forecasts future events of ongoing business processes based on event logs. However, deep learning PPM approaches are often limited by labeled data scarcity and the low variability of real-world logs. To address this, we introduce SiamSA-PPM, a self-supervised framework combining Siamese learning with statistically grounded data augmentation. It employs novel transformation methods that leverage control-flow semantics and frequent behavioral patterns to generate realistic, semantically valid trace variants, learning generalizable prefix representations without labeled supervision. Extensive experiments on real-world event logs with a wide range of variability, incorporating exhaustive hyperparameter tuning, demonstrate the value of this process-aware augmentation. Compared to state-of-the-art baselines, SiamSA-PPM achieves competitive or superior predictive performance in both next activity and fina
    
[^257]: DeliMF：门控交叉注意力融合结构化时间序列与临床文本用于早期重症监护谵妄预测

    DeliMF: Gated Cross-Attention Fusion of Structured Time Series and Clinical Text for Early Intensive Care Delirium Prediction

    [https://arxiv.org/abs/2610.24585](https://arxiv.org/abs/2610.24585)

    DeliMF通过门控交叉注意力将结构化时间序列与临床文本融合，实现了ICU患者入院首日的早期谵妄预测，弥补了传统方法仅依赖结构化数据而忽略临床笔记信息的不足。

    

    谵妄影响很大一部分重症监护病房（ICU）患者，并预示着更长的住院时间、持久的认知衰退和更高的死亡率。在患者入院第一天内识别出将会发生谵妄的患者，将为临床医生提供宝贵的干预窗口。然而，目前最先进的预测方法仅依赖于结构化电子健康记录（EHR）中的生命体征、实验室数值和药物信息，而忽视了在同一时间窗口内撰写的自由文本临床笔记，这些笔记蕴含着数字数据无法编码的信息。为此，我们提出了DeliMF，一个将结构化时间序列与临床文本融合的多模态框架。该框架使用时间Transformer编码结构化信号，使用临床文本编码器编码带时间戳的放射学报告，并通过门控交叉注意力模块将两者融合。该门控机制为文本学习针对每个患者的权重；当患者没有笔记时，门控关闭……

    arXiv:2610.24585v2 Announce Type: replace  Abstract: Delirium affects a large fraction of intensive care unit (ICU) patients and foreshadows longer stays, lasting cognitive decline, and higher mortality. Flagging the patients who will become delirious within the first day of admission would give clinicians a window to intervene. Yet state-of-the-art predictors rely only on the structured electronic health record (EHR) of vital signs, laboratory values, and medications, while overlooking the free-text clinical notes written in the same window, which carry information the numbers cannot encode. To this end, we present DeliMF, a multimodal framework that fuses the structured time series with clinical text. It encodes the structured signals with a temporal Transformer and the time-stamped radiology reports with a clinical text encoder, and fuses the two through a gated cross-attention module. The gate learns a per-patient weight on the text; when a patient has no note the gate closes and D
    
[^258]: 基于大语言模型的网络控制的黑盒性能分析

    Blackbox Performance Analysis of LLM-based Network Control

    [https://arxiv.org/abs/2610.24348](https://arxiv.org/abs/2610.24348)

    本文提出LENS——首个面向基于大语言模型的网络控制系统的黑盒性能分析工具，通过扩展深度强化学习来搜索性能差距和使LLM表现不佳的输入，并在流量工程和GPU分配系统中发现了显著的性能差距。

    

    由于出色的推理能力，大语言模型（LLM）正成为网络控制系统中不可或缺的组成部分。在此类基于LLM的系统部署之前，必须开发能够审计其在未见过的输入上性能的工具。然而，由于LLM固有的不透明性，审计这些系统极具挑战性。LENS是迈向基于LLM的网络控制系统黑盒性能分析的第一步。它使用新颖的方法扩展深度强化学习，以引导搜索性能差距以及使LLM陷入困境的输入。LENS在基于LLM的流量工程和GPU分配系统中发现了巨大的性能差距，其表现甚至可以媲美现有的、为手工设计的流量工程启发式方法而构建的白盒分析器。

    arXiv:2610.24348v2 Announce Type: replace-cross  Abstract: LLMs are becoming an indispensable part of network control systems due to their impressive reasoning. Before such LLM-enabled systems can be deployed, it is imperative to develop tools that can audit their performance on previously unseen inputs. However, auditing these systems is challenging due to the inherent opacity of LLMs. LENS is the first step towards black-box performance analysis of LLM-enabled network control systems. It uses novel methods to extend deep reinforcement learning to guide the search for performance gaps and inputs that make LLMs struggle. LENS finds large gaps for LLM-based traffic engineering and GPU allocation systems, and even matches existing white-box analyzers designed for hand-crafted TE heuristics.
    
[^259]: 通过逆地理编码学习行政边界感知的位置编码器

    Learning Administrative Boundary-Aware Location Encoders via Reverse Geocoding

    [https://arxiv.org/abs/2610.24306](https://arxiv.org/abs/2610.24306)

    本文首次提出利用逆地理编码文本作为自监督训练信号来学习行政边界感知的位置编码器，并设计了专属架构RevGeoRQ，可提升现有编码器性能并在多个下游任务上超越当代方法。

    

    位置编码器学习经纬度坐标的紧凑表示，用于下游地理空间预测任务。现有方法使用连续的或自然产生的监督信号（如影像、气候变量或物种分布）进行预训练，因此无法显式捕捉行政边界或地名，尽管它们对许多真实世界的地理空间现象有着强烈影响。据我们所知，我们首次提出了学习行政边界感知位置编码器的方法，使用逆地理编码文本作为自监督训练信号。我们的方法可以应用于现有的位置编码器架构，并持续提升其下游性能。我们进一步提出了RevGeoRQ，一种为逆地理编码监督量身定制的架构，带来了额外的性能增益，并在多个下游任务上超越了当代的位置编码器。

    arXiv:2610.24306v2 Announce Type: replace-cross  Abstract: Location encoders learn compact representations of latitude--longitude coordinates for downstream geospatial prediction tasks. Existing approaches are pretrained using continuous or naturally occurring supervision, such as imagery, climate variables, or species distributions, and therefore do not explicitly capture administrative boundaries or place names, despite their strong influence on many real-world geospatial phenomena. We introduce, to our knowledge, the first method for learning administrative boundary-aware location encoders, using reverse-geocoded text as a self-supervised training signal. Our approach can be applied to existing location encoder architectures and consistently improves their downstream performance. We further introduce RevGeoRQ, a custom architecture tailored to reverse-geocoding supervision, which yields additional gains and exceeds contemporary location encoders on several downstream tasks. Finally,
    
[^260]: 智能体强化学习中数据的扩展规律：定律与合成策略

    How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies

    [https://arxiv.org/abs/2610.23946](https://arxiv.org/abs/2610.23946)

    该论文将智能体强化学习的训练数据分解为任务、环境和轨迹三个可独立扩展的维度，并拟合出各维度的缩放定律，以回答在给定合成预算下应优先扩展哪个数据维度这一关键问题。

    

    预训练的缩放定律将训练数据视为一个一维量：即token数量。智能体强化学习继承了这种标量式的抽象，但现代数据合成流水线如今能够以截然不同的成本独立生成任务、环境和轨迹，使数据变成了一个结构化的设计空间。这引出了一个预训练从未需要回答的问题：在给定的合成预算下，应该扩展哪一个维度？我们通过三项紧密关联的贡献对这一问题进行了端到端的解答。首先，我们将智能体数据分解为三个基数维度（任务、环境、轨迹），外加一个奖励密度控制变量，并在数学+代码、软件工程（SWE）、网页以及UI/工具等应用领域的不同大语言模型（LLM）上拟合了各维度的缩放关系。在我们测量的所有模型规模、模型家族、领域以及强化学习算法（PPO和GRPO）中，缩放指数……

    arXiv:2610.23946v2 Announce Type: replace-cross  Abstract: Pretraining scaling laws treat training data as a one-dimensional quantity: a token count. Agentic reinforcement learning (RL) inherits this scalar abstraction, but modern synthesis pipelines now generate \textit{tasks}, \textit{environments}, and \textit{trajectories} independently and at very different costs, turning data into a structured design space. This raises a question pretraining never had to answer: given a synthesis budget, \textit{which axis} should be scaled? We address this question end-to-end with three tightly coupled contributions. First, we decompose agentic data into three cardinality axes (task, environment, trajectory), plus a reward-density control variable, and fit per-axis scaling relations on different large language models (LLMs) across math+code, SWE, web, and UI/tool application domains. Across all model sizes, model families, domains, and RL algorithms (PPO and GRPO) we measured, the scaling expone
    
[^261]: NestedVLA：学习整合与生成视觉-语言-动作模型的技能

    NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models

    [https://arxiv.org/abs/2610.23271](https://arxiv.org/abs/2610.23271)

    NestedVLA提出用嵌套超网络生成任务自适应的动作专家参数来事后整合独立学习的技能，从而在多个机器人操作基准上实现灵活的任务条件化技能复用并有效缓解任务间干扰。

    

    视觉-语言-动作（VLA）模型有望实现通用机器人智能体，然而跨异构任务与环境整合所学习的技能仍然十分困难：训练数据分散在不同平台上，联合多任务训练存在相互干扰的问题，而模型合并往往引入参数冲突或需要改变模型架构。为此，我们倡导一种先独立学习技能、再进行事后整合的范式。具体而言，我们提出NestedVLA，用参数生成取代参数聚合：一个嵌套超网络以目标任务为条件，合成任务自适应的动作专家参数，并与冻结的基础VLA模型组合；任务特定的压缩VLM残差则被单独保留。这一设计实现了灵活的任务条件化技能复用，同时缓解了任务间的干扰。在三个强化学习基准（MetaWorld、ManiSkill、CALVIN）上，NestedVLA的表现超越了现有方法。

    arXiv:2610.23271v1 Announce Type: cross  Abstract: Vision--Language--Action (VLA) models promise generalist robotic agents, yet consolidating skills learned across heterogeneous tasks and environments remains difficult: training data is fragmented across platforms, joint multi-task training suffers from interference, and model merging tends to introduce parameter conflicts or require architectural changes. We instead advocate a paradigm in which skills are first learned independently and then consolidated post hoc. To this end, we propose NestedVLA, which replaces parameter aggregation with parameter generation: a nested hypernetwork, conditioned on the target task, synthesizes task-adaptive action-expert parameters that compose with a frozen base VLA. Task-specific compressed VLM residuals are retained separately. This formulation enables flexible task-conditioned skill reuse while mitigating interference. Across three RL benchmarks (MetaWorld, ManiSkill, CALVIN), NestedVLA outperform
    
[^262]: 面向成本感知多轮LLM路由的响应信息价值估计

    Response-Informed Value Estimation for Cost-Aware Multi-Turn LLM Routing

    [https://arxiv.org/abs/2610.23239](https://arxiv.org/abs/2610.23239)

    提出RIVET方法，在训练阶段利用每次调用后的实际响应信息为每一轮提供价值监督信号，而在部署时仅依赖调用前信息进行成本感知的多轮LLM模型路由。

    

    多轮LLM路由需要在每一轮对话中、在模型响应可用之前就选择模型，然而任务成功与否只有在整个交互结束时才能揭晓，这使得每个中间选择都缺少属于自己的标签。然而，记录下来的交互轨迹保留了每次调用所产生的响应，这揭示了每一轮对话所需要的内容以及所选模型对其服务的效果——而这些证据仅凭最终结果是看不到的。我们提出了RIVET（面向轮级路由的响应信息价值估计），该方法在训练阶段利用这种调用后的证据，同时在部署时保持路由严格在调用前完成。RIVET在共享的状态-模型表示之上联合训练两个价值估计器：一个是可部署的路由价值估计器，仅根据调用前信息对候选模型进行评分；另一个是仅用于训练阶段的响应信息价值估计器，它额外观察实际产生的响应。以响应为条件，能够将响应引起的回报变化从残余噪声中区分出来

    arXiv:2610.23239v1 Announce Type: new  Abstract: Multi-turn LLM routing selects a model at every turn before its response is available, yet task success is revealed only when the interaction ends, leaving each intermediate choice without a label of its own. Logged trajectories, however, retain the response each call produced, which exposes what each turn required and how well the chosen model served it, evidence that outcomes alone hide. We introduce RIVET (Response-Informed Value Estimation for Turn-Level Routing), which exploits this post-call evidence during training while keeping routing strictly pre-call at deployment. RIVET jointly trains two value estimators over a shared state-model representation: a deployable Routing Value Estimator that scores candidate models from pre-call information, and a training-only Response-Informed Value Estimator that additionally observes the realized response. Conditioning on the response turns response-induced return variation from residual nois
    
[^263]: 内在的更强教师：通过策略内双向蒸馏增强MoE模型

    A Stronger Teacher Within: Strengthening MoE Models via On-Policy Bidirectional Distillation

    [https://arxiv.org/abs/2610.23185](https://arxiv.org/abs/2610.23185)

    提出MoEvolve框架，通过增加MoE模型每token激活预算来挖掘模型内部潜在更强教师，并采用“先恢复后提升”的策略内双向蒸馏策略强化MoE大语言模型。

    

    基于可验证奖励的强化学习（RLVR）已显著提升了混合专家（MoE）大语言模型，但其结果层面的奖励只能提供稀疏的监督信号。策略内蒸馏提供了密集的token级反馈，然而需要一个更强的兼容教师模型，而策略内自蒸馏则依赖于手工设计的特权信息。我们观察到，MoE架构本身提供了一种解决这一困境的方法：增加每个token的激活预算可以释放潜在的专家容量，从而在模型内部产生一个潜在更强的教师。然而，简单的扩展会降低性能，因为新共同激活的专家并未经过协同工作的优化。因此，我们提出了MoEvolve，一个遵循“先恢复后提升”策略的双向蒸馏框架：我们首先恢复激活扩展后的高k模型，并通过利用其额外的激活容量进一步强化它，然后将（摘要在此处截断）

    arXiv:2610.23185v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved Mixture-of-Experts (MoE) LLMs, but its outcome-level rewards provide only sparse supervision. On-Policy Distillation offers dense token-level feedback, yet requires a stronger compatible teacher, while on-policy self-distillation relies on hand-designed privileged information. We observe that the MoE architecture itself offers a way to resolve this dilemma, as increasing the activation budget per token exposes latent expert capacity and yields a potentially stronger teacher within the model. However, naive expansion degrades performance because newly co-activated experts are not optimized to work together. We therefore propose MoEvolve, a bidirectional distillation framework following a recovery-then-improvement strategy: we first restore the activation-expanded high-$k$ model and further strengthen it by exploiting its additional active capacity, then rev
    
[^264]: DIFFLET：AWS Trainium 上扩散 Transformer 模型的快速推理引擎

    DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium

    [https://arxiv.org/abs/2610.23047](https://arxiv.org/abs/2610.23047)

    Difflet 是面向 AWS Trainium 的扩散 Transformer 服务引擎，通过五个流水线共享统一编译后端、按注意力头切分的融合注意力以及基于设备端探针的逐步缓存决策，将去噪步骤加速 1.2-4.3 倍，常驻服务速度提升 1.5-3.4 倍。

    

    在 AWS Trainium2 上，扩散模型的每个组件都以针对固定形状预先编译的程序形式运行，而流水线在不同模型之间各不相同，提示词长度在不同请求之间各不相同，缓存决策在不同去噪步骤之间各不相同。我们提出了 Difflet——一种服务引擎，其中五个模型流水线共享同一个编译后端，联合注意力按注意力头在多个核心之间切分，主机控制器在每个去噪步骤决定是运行去噪器还是复用其先前的输出。一个小型探针程序在设备上计算缓存信号并每步返回一个数值，一个常驻模型进程使所有程序在请求之间保持加载状态。在五个图像和视频模型家族中，融合注意力使去噪器步骤相比使用 PyTorch/XLA 注意力的同一框架加速 1.2-4.3 倍，并且在 32.8k 视觉 token 规模下，它是唯一能够完成编译的注意力实现。在五个模型中的四个上，常驻服务快 1.5-3.4 倍（摘要原文在此处截断）。

    arXiv:2610.23047v1 Announce Type: cross  Abstract: On AWS Trainium2, every component of a diffusion model runs as a program compiled ahead of time for fixed shapes, while pipelines differ across models, prompt lengths across requests, and caching decisions across denoising steps. We present Difflet, a serving engine in which five model pipelines share one compilation backend, joint attention is split across cores by attention head, and a host controller decides at each denoising step whether to run the denoiser or reuse its earlier output. A small probe program computes the caching signal on the device and returns one number per step, and a resident model process keeps every program loaded across requests. Across five image and video model families, fused attention accelerates denoiser steps by 1.2-4.3x over the same framework using PyTorch/XLA attention, and at 32.8k visual tokens it is the only attention that compiles. On four of the five models, resident serving is 1.5-3.4x faster t
    
[^265]: 迈向差分隐私优化的超参数迁移

    Towards Hyperparameter Transfer for Differentially Private Optimization

    [https://arxiv.org/abs/2610.22976](https://arxiv.org/abs/2610.22976)

    本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。

    

    缩放定律与超参数调优仍然是训练大规模模型的核心挑战，尤其在差分隐私（DP）条件下，由于噪声注入和逐样本梯度裁剪，优化问题变得更加复杂。虽然最大更新参数化（μP）能够在标准训练中实现零样本超参数迁移，但我们证明了在DP约束下它会失效，原因在于高维噪声与低秩梯度具有截然不同的谱特性。在本工作中，我们提出了DP-μP，这是一个具有理论依据的框架，将μP扩展到了隐私保护场景。通过分析梯度裁剪、噪声注入与特征学习之间的相互作用，我们推导出了一个控制逐层学习率缩放的信噪比（SNR）条件。结合我们提出的谱缩放公式，我们为DP-SGD、DP-Adam和DP-Muon优化器给出了具体的缩放规则。

    arXiv:2610.22976v1 Announce Type: new  Abstract: Scaling laws and hyperparameter tuning remain central challenges in training large-scale models, particularly under differential privacy (DP), where optimization is further complicated by noise injection and per-sample gradient clipping. While Maximal Update Parametrization ($\mu$P) allows for zero-shot hyperparameter transfer in standard training, we show that it fails under DP constraints due to the distinct spectral properties of high-dimensional noise versus low-rank gradients. In this work, we introduce DP-$\mu$P, a theoretically grounded framework that extends $\mu$P to the privacy-preserving setting. By analyzing the interaction between gradient clipping, noise injection, and feature learning, we derive a Signal-to-Noise Ratio (SNR) condition that governs layer-wise learning rate scaling. Combining this with our proposed spectral scaling formulation, we provide specific scaling rules for DP-SGD, DP-Adam, and DP-Muon optimizers. Em
    
[^266]: 批量上下文强化 II：RLVR 需要长采样窗口吗？

    Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?

    [https://arxiv.org/abs/2610.22887](https://arxiv.org/abs/2610.22887)

    提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。

    

    长采样窗口使得可验证奖励强化学习（RLVR）的代价高昂，但简单缩短窗口可能会丢失正确性信号：当每个单问题的采样在得出答案之前就被截断时，组相对准确率优势将不复存在。作为批量上下文强化（Batched Contextual Reinforcement, BCR）的技术延续，我们研究了并发奖励如何放宽这种对长窗口的依赖。并发约束强化将三个可独立验证的问题置于同一个有长度上限的训练完成中。在截断前完成的正确答案仍可获得奖励，而共享预算在无需显式长度惩罚的情况下，鼓励更简短的有效推理。我们将该机制与条件奖励方差和更新方差分析联系起来，并在4B和8B数学推理模型上进行了研究。与使用32k窗口的单问题GRPO相比，使用8,000-token窗口的CCR在4B模型上将每训练步的token数从975,464降至174,957。

    arXiv:2610.22887v1 Announce Type: cross  Abstract: Long rollout windows make reinforcement learning with verifiable rewards (RLVR) expensive, yet simply shortening them can remove the correctness signal: when every single-problem rollout truncates before an answer, group-relative accuracy advantages vanish. As a technical continuation of Batched Contextual Reinforcement (BCR), we study how concurrent rewards relax this dependence on long windows. Concurrent Constraint Reinforcement (CCR) places three independently verifiable problems in one capped training completion. Correct answers completed before truncation retain credit, while a shared budget encourages shorter useful reasoning without an explicit length penalty. We connect this mechanism to conditional reward- and update-variance analyses and study 4B and 8B mathematical reasoning models. Against single-problem GRPO with a 32k window, CCR with an 8,000-token window reduces tokens per training step from 975,464 to 174,957 on 4B an
    
[^267]: LLM-Attacker：用于切分式与分布式大语言模型系统隐私评估的统一红队框架

    The LLM-Attacker: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems

    [https://arxiv.org/abs/2610.22722](https://arxiv.org/abs/2610.22722)

    本文提出LLM-Attacker统一红队评估框架，通过记录攻击者的选择实现切分式与分布式大语言模型隐私评估的可复现与可比较，并揭示训练期间返回梯度会使token恢复率从94.20%升至97.38%、精确文档恢复率从13.71%升至37.77%。

    

    切分式语言模型将输入文本保留在客户端，但会将中间表示发送到服务器，攻击者可以从这些消息中重建文本。当攻击者看到不同的消息、掌握不同的模型权重，或使用不同的数据集和恢复度量时，隐私评估之间的比较就变得十分困难。我们提出了LLM-Attacker，一个记录这些攻击选择的框架，使研究人员能够检查、比较和扩展评估。兼容的攻击可以重用已记录的消息，其预测结果会被保存以供独立评分。评分规则要求在防御通过之前必须存在一个关联的无防御参考基线。报告还会识别出尚未经过测试的攻击。在一项受控的GPT-2实验中，添加训练期间返回的梯度使token恢复率从94.20%提升到97.38%，精确文档恢复率从13.71%提升到37.77%。token增益的95%置信区间高于测试前设定的两个百分点阈值。

    arXiv:2610.22722v1 Announce Type: cross  Abstract: Split language models keep input text on a client but send intermediate representations to a server. Attacks can reconstruct text from these messages. Comparing privacy evaluations is difficult when attackers see different messages, know different model weights or use different datasets and recovery measures. We present LLM-Attacker, a framework that records these choices and lets researchers check, compare and extend an evaluation. Compatible attacks reuse recorded messages, and their predictions are saved for independent scoring. The scoring rule requires a linked undefended reference before a defence can pass. Reports also identify attacks that remain untested. In a controlled GPT-2 experiment, adding gradients returned during training increased token recovery from 94.20% to 97.38% and exact-document recovery from 13.71% to 37.77%. The 95% interval for the token gain lay above the two-percentage-point threshold fixed before testing.
    
[^268]: 微调前呈线性的知识在微调后能够泛化

    Knowledge That Is Linear Before Fine-Tuning Generalizes After It

    [https://arxiv.org/abs/2610.22300](https://arxiv.org/abs/2610.22300)

    论文提出区分语言模型中支持泛化的“活性事实”与仅可回忆的“惰性事实”，并通过重标注实验发现：微调前在模型内部呈线性可读的知识，在微调后能够泛化到共享相同答案的未训练实体上。

    

    并非语言模型能够回忆的所有事实都能同样好地支持进一步的学习。我们区分了“活性事实”与“惰性事实”：前者能够支持超出其自身回忆范围的泛化，后者虽然可以被回忆起来，但在给定任务中却不支持这种泛化。我们通过从已有关系构造新关系的方式来研究这一区分。以将地点映射到其所属大洲的关系（巴黎 → 欧洲）为例，为每个答案分配一个任意的新标签（欧洲 → 螃蟹，亚洲 → 斑马），于是新关系便将巴黎映射为螃蟹。我们在一部分主体上用新关系对模型进行微调，但不告诉模型原始关系或重新标注规则，然后测试模型能否为那些在原始关系中共享相同答案的留出主体分配相应的标签。也就是说，在学会了“巴黎 → 螃蟹”之后，模型能否回答出“慕尼黑 → 螃蟹”？在十一个语言模型和七十个关系的实验中，原始关系的线性可读程度……（摘要原文在此处截断）

    arXiv:2610.22300v1 Announce Type: cross  Abstract: Not all facts a language model can recall support further learning equally well. We distinguish active facts, which support generalization beyond their own recall, from inert facts, which can be recalled but do not support such generalization in a given task. We study this distinction by constructing new relations from existing ones. Take the relation that maps a place to its continent (Paris -> Europe) and assign each of its answers an arbitrary new label (Europe -> crab, Asia -> zebra). The new relation then maps Paris -> crab. We fine-tune the model on the new relation for a subset of subjects, without telling it the original relation or the relabeling rule, and test whether it assigns the corresponding labels to held-out subjects that share an answer in the original relation. Having learned Paris -> crab, does it answer Munich -> crab? Across eleven language models and 70 relations, how linearly readable the original relation is in
    
[^269]: 免训练的在策略蒸馏

    Training-Free On-Policy Distillation

    [https://arxiv.org/abs/2610.22236](https://arxiv.org/abs/2610.22236)

    本文提出免训练的在策略蒸馏方法，当多个专家模型共享同一基础模型时，仅需通过几百个采样样本的一次前向传播、以最小二乘方式逐层融合，即可在无需任何梯度训练的情况下将多个专家的能力聚合到一个模型中，且平均准确率超过任何单一专家模型。

    

    在策略蒸馏需要为每个教师模型进行采样、教师前向传播和梯度更新，才能将专家模型整合到一个学生模型中。我们证明这种整合其实不需要训练：仅通过前向传播，就可以将分别微调的多个专家模型的能力聚合到一个模型中，该模型平均而言比任何一个单独的专家模型都更加准确。当专家模型共享同一个基础模型时，蒸馏存在一个精确的逐层解：融合模型的每个线性层必须在专家模型所接收的输入上复现该专家的输出（同时读取其自身的答案），这是一个最小二乘问题，其解只需要这些输入的二阶矩。对几百个采样样本进行一次前向传播即可获得这些二阶矩。我们将这一方法称为免训练在策略蒸馏。相比求解器，采样样本中所包含的内容更为重要：在重复循环中被截断的回答在数学专家模型的采样中只占少数。

    arXiv:2610.22236v1 Announce Type: cross  Abstract: On-policy distillation consolidates specialist models into one student at the price of rollouts, teacher passes and gradient steps for every teacher. We show that the consolidation does not need training: with forward passes only, the capabilities of separately tuned specialists can be aggregated into one model that is, on average, more accurate than either specialist alone. When specialists share a base model, distillation has an exact layer-wise solution: each linear layer of the fused model must reproduce every specialist's output on the inputs that specialist receives while reading its own answers, a least-squares problem whose solution needs only the second moments of those inputs. One forward pass over a few hundred rollouts provides them. We call this training-free on-policy distillation (\method). What the rollouts contain matters more than the solver: answers truncated in repetition loops are few among a mathematics specialist
    
[^270]: HyCo：面向高性价比移动-云协同推测解码的混合草稿生成机制

    HyCo: Hybrid Drafting for Cost-Efficient Mobile-Cloud Speculative Decoding

    [https://arxiv.org/abs/2610.21436](https://arxiv.org/abs/2610.21436)

    HyCo提出混合草稿生成机制，将服务器模型的原生多词预测头部署到移动NPU上并与小型语言模型单次推理协同工作，配合量化和ε阈值松弛策略，使移动-云推测解码的服务器GPU时间降低约29%-39%。

    

    设备-云协同推测解码能够减少服务器的计算开销，但使用小型语言模型（SLM）生成深度草稿树需要在移动神经处理单元（NPU）上进行多次推理。我们提出了HyCo，它将草稿深度分配给在设备端执行的服务器模型原生多词预测（MTP）头，并将替代分支分配给单次SLM推理。服务器只需返回该MTP头所需的隐状态、输入嵌入和键值缓存条目，而无需执行其原有的顺序计算步骤。通过量化、受限词表和1K-token窗口设置，两个草稿模型均能适配NPU运行。此外，一种ε阈值规则进一步放宽了对验证器采样分布的精确保持要求。在通过LTE/5G网络连接Galaxy S26 Ultra的单客户端运行中，与服务器端MTP自推测相比，HyCo在Spec-Bench上将每个输出token的服务器GPU时间降低了38.6%，在32K-token摘要任务上降低了29.2%。

    arXiv:2610.21436v1 Announce Type: new  Abstract: Device-cloud speculative decoding reduces server computation, but growing a deep draft tree with a small language model (SLM) requires repeated passes on a mobile neural processing unit (NPU). We present HyCo, which assigns depth to the server model's native multi-token prediction (MTP) head, executed on the device, and alternative branches to one SLM pass. The server returns the hidden state, input embedding, and key-value entries needed by the head instead of executing its sequential steps. Quantization, a restricted vocabulary, and a 1K-token window fit both drafters to the NPU. An {\epsilon}-threshold rule additionally relaxes exact preservation of the verifier's sampling distribution. In single-client runs with a Galaxy S26 Ultra over LTE/5G, HyCo reduces server GPU time per output token relative to server-side MTP self-speculation by 38.6% on Spec-Bench and 29.2% on 32K-token summarization. With the same acceptance rule, three Spec
    
[^271]: 超越不变性：面向多模态学习的按因子选择性处理分布偏移

    Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning

    [https://arxiv.org/abs/2610.21403](https://arxiv.org/abs/2610.21403)

    提出FaShift框架，超越传统不变性方法，通过按因子诊断分布变化、干预效应和跨环境稳定性，对多模态学习中的偏移信息进行选择性抑制、保留或削弱。

    

    分布偏移是鲁棒多模态学习面临的根本性挑战。现有方法通常通过抑制依赖环境的变异来促进不变性，从而隐式地对偏移信息施加统一处理。然而，偏移信息可能产生不同的预测后果：有些对预测贡献甚微，有些仍然可靠地包含信息，有些则表现出依赖环境的预测效应。这就引出了一个根本性问题：哪些偏移信息应该被抑制、保留或削弱？为了解决这一问题，我们提出了FaShift，一个按因子处理偏移的框架，它根据潜在表示因子的分布变化、干预效应和跨环境稳定性对其进行诊断。通过分解多模态表示并评估这些互补属性，FaShift能够区分非预测性因子、稳定预测性因子和预测性（因子）……

    arXiv:2610.21403v1 Announce Type: new  Abstract: Distribution shifts pose a fundamental challenge to robust multimodal learning. Existing methods typically promote invariance by suppressing environment-dependent variations, implicitly applying a uniform treatment to shifted information. However, shifted information can have different predictive consequences: some contributes little to prediction, some remains reliably informative, and some exhibits environment-dependent predictive effects. This raises a fundamental question: which shifted information should be suppressed, preserved, or attenuated? To address this question, we propose FaShift, a factor-wise shift framework that diagnoses latent representation factors according to their distributional changes, intervention effects, and cross-environment stability. By factorizing multimodal representations and evaluating these complementary properties, FaShift distinguishes non-predictive factors, stably predictive factors, and predictive
    
[^272]: 在策略蒸馏需要完成每一次 Rollout 吗？

    Does On-Policy Distillation Need to Complete Every Rollout?

    [https://arxiv.org/abs/2610.21337](https://arxiv.org/abs/2610.21337)

    该论文提出随机截断在策略蒸馏（STOPD），通过在生成前随机采样停止位置来截断学生的 rollout 序列，并用分块逆概率加权校正采样偏差，在期望意义上保持与原始 OPD 相同的梯度，从而大幅降低计算开销。

    

    在策略蒸馏（On-policy Distillation, OPD）通过对学生模型生成的 rollout 序列提供反馈，有效地将教师模型的能力迁移给学生模型。然而，生成完整的学生 rollout 序列并为每个 token 计算教师反馈，使得 OPD 的计算开销十分高昂。我们提出了随机截断在策略蒸馏（Stochastic Truncated On-Policy Distillation, STOPD），该方法通过随机截断学生 rollout 序列，并利用分块逆概率加权（blockwise IPW）校正由此产生的采样偏差，从而加速 OPD。在每一步训练中，我们在生成前为每个 rollout 采样一个停止位置，让学生模型生成至该位置或自然终止。我们对每个生成的 token 获取教师反馈，并使用分块 IPW 补偿保留后续 token 的较低概率，同时对权重加以控制。我们证明了这种加权方式在期望意义上保持了原始 OPD 的梯度总和不变，并推导出了一步更新的误差界……

    arXiv:2610.21337v1 Announce Type: cross  Abstract: On-policy distillation (OPD) effectively transfers capabilities from a teacher to a student through feedback on student-generated rollouts. However, generating complete student rollouts and computing teacher feedback for every token make OPD computationally expensive. We introduce Stochastic Truncated On-Policy Distillation (STOPD), which accelerates OPD by randomly truncating student rollouts and correcting the resulting sampling bias with blockwise inverse-probability weighting (IPW). At each training step, we sample a stopping position for each rollout before generation and let the student generate until that position or natural termination. We obtain teacher feedback on every generated token and use blockwise IPW to compensate for the lower probability of retaining later tokens while controlling the weights. We prove that this weighting preserves Vanilla OPD's gradient sum in expectation and derive one-step update-error bounds for 
    
[^273]: 完整还是稀疏：两种可辨识性的故事

    Complete or Sparse: A Tale of Two Identifiabilities

    [https://arxiv.org/abs/2610.20851](https://arxiv.org/abs/2610.20851)

    本文证明在生成式与判别式两种因果方向下，表示维度充足时最优编码器是潜在变量的线性函数，而结合非负激活与能量效率约束可使每个神经元恰好编码一个潜在变量，为“祖母神经元”现象提供了精确的理论解释。

    

    理解神经网络何时学习到可解释的表示，是机制可解释性研究的核心问题。我们通过可辨识性来研究这一问题：监督学习何时能够恢复数据背后真实的潜在变量？我们引入了两种以因果方向相区分的数据生成过程：一种是由标签导致潜在变量（生成式），另一种是由潜在变量导致标签（判别式）。当表示的维度不少于潜在空间的维度时，我们证明最优编码器是潜在变量的线性函数；相反，当维度不足时，可辨识性通常会失效，编码器变为非线性。随后我们进一步证明，额外的约束能够恢复单个潜在变量，而不仅仅是它们的线性混合。在生成式设定中，非负激活与能量效率迫使每个神经元恰好编码一个潜在变量，从而为“选择性祖母神经元”现象提供了精确的理论解释。

    arXiv:2610.20851v1 Announce Type: cross  Abstract: Understanding when neural networks learn interpretable representations is central to mechanistic interpretability. We study this through identifiability: when can supervised learning recover the true latent variables underlying the data? We introduce two data generating processes distinguished by their causal direction where either labels cause latents (generative) or latents cause labels (discriminative). When the representation has at least as many dimensions as the latent space, we prove that the optimal encoders are linear in the latents. Conversely, when dimensions are insufficient, identifiability generically fails and encoders become nonlinear. We then show that additional constraints yield recovery of individual latent variables, not just linear mixtures. In the generative setting, non-negative activations and energy efficiency force each neuron to encode exactly one latent, providing a precise account of selective grandmother 
    
[^274]: 论自然语言自编码器的可监控性：识别与判别

    On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination

    [https://arxiv.org/abs/2610.18727](https://arxiv.org/abs/2610.18727)

    论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。

    

    自然语言自编码器（NLA）被提出作为语言模型的监督工具：激活言语化器（AV）为激活向量撰写英文解释，而激活重构器（AR）将解释映射回向量。它们在监督中的价值取决于人类监控者在给定AV解释的情况下，能否将一段对激活的真实叙述与虚假叙述区分开来。我们将NLA的训练与NLA监督部署形式化为源自涌现通信领域的两种不同的指称博弈：训练是一种生成式指称博弈，而监督是一种判别式的、多模态的、部分可观测的指称博弈，该博弈在受监控语言模型的激活与真实描述它们的叙述之间的多对多关系上展开。这两种博弈之间的差距预示着AV的解释将能够识别其所描述的是哪个激活，却无法将其真实叙述与紧密匹配的叙述区分开来。

    arXiv:2610.18727v2 Announce Type: replace-cross  Abstract: Natural Language Autoencoders (NLAs) are proposed as oversight tools for language models: an Activation Verbaliser (AV) writes an English explanation of an activation vector, and an Activation Reconstructor (AR) maps the explanation back to a vector. Their value for oversight depends on a human monitor being able to tell a true account of an activation from a false one, given the AV's explanation. We formalise NLA training and NLA oversight deployment as two different referential games from Emergent Communication: training is a generative referential game, while oversight is a discriminative, multi-modal, partially-observable referential game played over a many-to-many relation between the monitored language model's activations and the accounts that are true of them. The gap between the two games predicts that AV's explanations will identify which activation they describe without discriminating its true account from closely mat
    
[^275]: 基于缺口驱动的分层采集方法构建 STRATA-S12：一个面向自监督学习的开放 SAR-光学数据集

    Deficit-Driven Stratified Acquisition for Building STRATA-S12: An Open SAR-Optical Corpus for Self-Supervised Learning

    [https://arxiv.org/abs/2610.17680](https://arxiv.org/abs/2610.17680)

    提出了缺口驱动分层采集算法（DDSA），将构建配准的 Sentinel-1/Sentinel-2 数据集建模为半球校正分层上的约束最大覆盖问题，并通过人机协同的缺口修复机制确保全部 192 个分层得到完整覆盖，为自监督地球观测学习提供了高质量的开放 SAR-光学语料库。

    

    arXiv:2610.17680v2 公告类型：replace-cross 摘要：为自监督地球观测构建一个配准的 SAR-光学语料库，本质上首先是一个数据采集问题，然而目前主要的多模态数据集都是机会性采集的，既没有分层的配额控制，也没有应对非均匀云覆盖所导致的地理数据缺失的机制。我们提出了缺口驱动的分层采集（DDSA）算法，该算法将构建配准的 Sentinel-1/Sentinel-2 语料库视为在经过半球校正的（类别×季节×区域）分层上的约束最大覆盖问题，执行严格的时间配对和多阶段质量把关，并在采集过程中通过人机协同的缺口修复阶段形成闭环。在该算法自身产品上的测量表明，该算法取得了单次采集无法报告的三项结果。缺口驱动的排序使得部分运行也能成为覆盖性运行：它使 192 个分层中的每一个从最初几个……

    arXiv:2610.17680v2 Announce Type: replace-cross  Abstract: Building a co-registered SAR-optical corpus for self-supervised Earth Observation is above all an acquisition problem, yet the main multi-modal datasets are collected opportunistically, with no per-stratum quota control and no mechanism against the geographic dropout that non-uniform cloud cover leaves behind. We present Deficit-Driven Stratified Acquisition (DDSA), an algorithm that treats building a co-registered Sentinel-1/Sentinel-2 corpus as a constrained maximum-coverage problem over a hemisphere-corrected (class x season x region) stratification, enforces strict temporal pairing and a multi-stage quality gate, and closes the loop during acquisition with a human-in-the-loop deficit-repair stage. Measured on its own product, the algorithm yields three results a single-pass acquisition cannot report. Deficit-driven ordering makes a partial run a covering one: it holds every one of the 192 strata populated from the first few
    
[^276]: 基于比例包含边际效应的变量重要性度量

    Measuring Variable Importance Using Proportional Inclusive Marginal Effects

    [https://arxiv.org/abs/2610.17053](https://arxiv.org/abs/2610.17053)

    提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。

    

    基于Shapley的度量方法在统计学和机器学习中被广泛用于衡量变量重要性。然而，这些度量方法存在一个严重问题——当一个非活跃变量与活跃变量相关时，该非活跃变量可能获得正的重要性度量值。Herin等人最近提出了一种称为比例边际效应（PME）的度量方法来缓解这种虚假效应。然而，PME是为确定性模型开发的，且计算成本很高。本文提出了一种称为比例包含边际效应（PRIME）的新度量方法，克服了PME的这两个局限性。PRIME的关键思想是使用一种新颖的无模型变量选择技术来识别活跃变量，并仅将这些变量纳入重要性度量的计算中。对测试函数和真实数据集的分析表明，PRIME能够从含噪声的输入-输出数据中准确、可靠地量化变量重要性。

    arXiv:2610.17053v1 Announce Type: new  Abstract: Shapley-based measures are widely used for measuring variable importance in statistics and machine learning. However, these measures have a serious issue--a nonactive variable can get a positive importance measure when it is correlated with an active variable. Herin et al. recently proposed a measure called proportional marginal effects (PME) that mitigates this spurious effect. However, PME is developed for deterministic models and has high computational cost. In this article, we propose a new measure called proportional inclusive marginal effects (PRIME) that overcomes these two limitations of PME. The key idea in PRIME is to use a novel model-free variable selection technique to identify the active variables and include only them in the computation of the importance measure. Analysis of test functions and real datasets show that PRIME can accurately and reliably quantify variable importance from noisy input-output data.
    
[^277]: 面向未知动态环境中实时机器人导航的基于可达性的安全表示学习

    Learning Reachability-Based Safety Representations for Real-Time Robot Navigation in Unknown and Dynamic Environments

    [https://arxiv.org/abs/2610.16522](https://arxiv.org/abs/2610.16522)

    本论文通过离线在哈密顿-雅可比可达性解上训练神经隐式表示，并仅学习非负的动力学修正项，使基于优化的控制器能够在未知动态环境中实时获得安全信息，实现实时安全的机器人导航。

    

    在未知、动态环境中实现安全的机器人导航，仅靠几何碰撞检测是不够的：机器人即使距离障碍物还很远，也可能由于其自身的速度、朝向或障碍物的运动而无法避开它。哈密顿-雅可比（HJ）可达性分析通过一个值函数刻画了这一点，该值函数的符号能够区分可以安全继续运行的状态与碰撞不可避免的状态，但针对每个新观测到的环境在线计算该值函数的代价过于高昂。本论文通过在可达性解上离线训练神经隐式表示，来实时地为基于优化的控制器提供这一信息。由于值函数从不超过约束，它等于一个可直接计算的约束（例如符号距离）减去一个非负的动力学修正项。仅学习这一修正项，即使存在近似误差，也能保证所表示的障碍物始终位于预测安全集之外。

    arXiv:2610.16522v2 Announce Type: replace-cross  Abstract: Safe robot navigation in unknown, dynamic environments requires more than geometric collision checking: a robot still far from an obstacle may be unable to avoid it because of its speed, orientation, or the obstacle's motion. Hamilton--Jacobi (HJ) reachability analysis captures this in a value function whose sign separates states that admit a safe continuation from those where collision is unavoidable, but computing it online for every newly observed environment is too expensive. This dissertation trains neural implicit representations offline on reachability solutions to give optimization-based controllers this information in real time. Since the value function never exceeds the constraint, it equals a directly computed constraint, such as a signed distance, minus a nonnegative dynamics correction. Learning only this correction keeps represented obstacles outside the predicted safe set despite approximation errors. The first t
    
[^278]: 脉冲语言模型的计算最优缩放定律

    Compute-Optimal Scaling Laws for Spiking Language Models

    [https://arxiv.org/abs/2610.15146](https://arxiv.org/abs/2610.15146)

    本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。

    

    脉冲神经网络通过稀疏的二值激活实现节能的语言建模，但脉冲语言模型在大规模下的效率-精度权衡尚未被刻画。我们首次对脉冲语言模型进行了Chinchilla式的IsoFLOP缩放研究，在FineWeb-EDU数据集上，于3×10¹⁷至10¹⁹ FLOPs的计算预算范围内，对SpikeGPT系列模型（采用RWKV骨干，宽度从384到1536）进行了系统扫描。我们报告了两个发现：首先，计算最优前沿为 N_opt ∝ C^0.56，接近Hoffmann等人报告的密集Transformer的0.50，因此脉冲化并不改变在给定计算预算下参数与数据应如何权衡；其次，与以相同方式训练的非脉冲RWKV和Transformer基线相比，损失差距可分解为不随计算量变化的约0.16 nats的骨干成本，以及随计算量稳步下降的脉冲成本（从0.34降至0.23 nats）。

    arXiv:2610.15146v2 Announce Type: replace-cross  Abstract: Spiking neural networks target energy-efficient language modeling through sparse, binary activations, but the efficiency-accuracy tradeoff of spiking language models is uncharacterized at scale. We present the first Chinchilla-style IsoFLOP scaling study of a spiking language model, sweeping SpikeGPT-family models (RWKV backbone) from width 384 to 1536 across compute budgets from $3\times10^{17}$ to $10^{19}$ FLOPs on FineWeb-EDU. We report two findings. First, the compute-optimal frontier is $N_{opt} \propto C^{0.56}$, close to the 0.50 that Hoffmann et al. report for dense Transformers, so spiking does not change how parameters and data should be traded off under a compute budget. Second, against non-spiking RWKV and Transformer baselines trained the same way, the loss gap splits into a backbone cost of about 0.16 nats that does not change with compute and a spiking cost that falls steadily with compute, from 0.34 to 0.23 nat
    
[^279]: 当动态缩放失效时：尺度-状态反馈主导线性注意力与状态空间大语言模型中的循环状态量化

    When Dynamic Scaling Fails: A Scale-State Feedback Governs Recurrent-State Quantization in Linear-Attention and SSM LLMs

    [https://arxiv.org/abs/2610.15139](https://arxiv.org/abs/2610.15139)

    该论文揭示了线性注意力与状态空间大语言模型中循环状态量化的失效根源——量化器尺度与状态之间的反馈回路使误差在递推中累积并引发范数爆炸，并证明数值不稳定与功能退化是可分离的两种失效，其中前者可通过全精度参考缩放完全消除。

    

    线性注意力和状态空间大语言模型将长上下文存储在固定大小的循环状态中，对该状态进行量化是长解码过程中降低其成本的天然途径。与权重张量不同，状态在每一步都会被重写，因此在每一步施加的量化器位于递推循环内部，其误差会不断累积。初始阶段对合成记忆任务完全无损的INT4量化，在一千步之后会将其彻底破坏。其机制在于量化器尺度与其所来源状态之间的反馈。将该尺度绑定到全精度参考值上可以完全消除由此产生的范数爆炸，但仅能挽回5.4 nats损失中的1.8，因此数值不稳定与功能退化是可分离的两种失效。剩余的损失与分辨率相关，在每步量化设置中，所测试的所有离群值感知方案均不敌简单的持久缩放。跨层分配比特位确实有帮助，但在article-disjoint划分下，学习到的策略……（原文摘要到此截断）

    arXiv:2610.15139v2 Announce Type: replace  Abstract: Linear-attention and state-space LLMs carry long context in a fixed-size recurrent state, and quantizing that state is the natural way to cut its cost during long decode. Unlike a weight tensor, the state is rewritten at every step, so a quantizer applied at each step sits inside the recurrence and its errors compound. INT4 that leaves synthetic recall perfectly intact at the start destroys it a thousand steps later. The mechanism is a feedback between the quantizer scale and the state it is derived from. Yoking that scale to a full-precision reference removes the resulting norm explosion entirely, yet recovers only 1.8 of the 5.4 nats lost, so numerical instability and functional degradation are separable failures. The remainder is resolution-dependent, and in our per-step setting every outlier-aware scheme we test loses to plain persistent scaling. Allocating bits across layers does help, but under article-disjoint splits a learned
    
[^280]: 当激活引导被用作记忆时，什么能够存留

    What Survives When Activation Steering Is Used as Memory

    [https://arxiv.org/abs/2610.14958](https://arxiv.org/abs/2610.14958)

    该研究提出一种免训练的激活记忆机制，并证明激活引导偏移作为注意力效应的固定平均值只能存留与问题无关的内容：这种记忆虽具有约98%的高选择性，但用户事实仅被存储在最后几层且几乎从不被使用，表明用户事实与偏好仍需以文本形式记忆。

    

    激活引导（activation steering）近来被用作技能的记忆：智能体过去的经验被存储为引导向量，以供日后检索。而关于用户的事实与偏好则被认为需要以文本形式存储。我们在一个冻结的大语言模型旁搭建了一个免训练的激活记忆来检验这一假设：一次经验在模型后层状态中引起的变化被存储为一个偏移量（shift），按对话主旨的白化键值归档，且仅在后续会话中主旨匹配时才被重新加回。我们论证，这样的偏移量本质上是注意力在上下文中针对每个问题重新计算的效应的固定平均值，因此只有与问题无关的内容才能存留。在一项针对 Qwen3.5-2B 和 9B 的预注册研究中，采用来自四个模型家族的评判模型，该记忆表现出选择性（约98%的无关回复保持不变），但事实被存储为“该说的词”，仅存在于最后几层，且几乎从未被使用（干净使用率至多15%，相比之下文本记忆为96–98……原文摘要在此截断）。

    arXiv:2610.14958v1 Announce Type: new  Abstract: Activation steering has recently been used as a memory for skills: past experiences of an agent are stored as steering vectors and retrieved later. Facts and preferences about the user are assumed to need text instead. We test that assumption with a training-free activation memory beside a frozen LLM. The change an experience causes in late-layer states is stored as a shift, filed under a whitened key for the conversation's gist, and added back in a later session only when the gist matches. We argue that such a shift is a fixed average of an effect that attention, in context, recomputes for every question, so only question-independent content can survive. In a pre-registered study on Qwen3.5-2B and 9B, with judges from four model families, the memory is selective (about 98% of unrelated replies are unchanged), but facts are stored as the word to say, in the last layers only, and are almost never used (at most 15% clean use, against 96-98
    
[^281]: 关于生成扩散与最优传输之间联系的一些注记

    Some remarks on the connection between generative diffusion and optimal transport

    [https://arxiv.org/abs/2610.13614](https://arxiv.org/abs/2610.13614)

    本文揭示了生成扩散与最优传输之间的三个简单联系：数据加噪的联合分布在每个时刻都是熵正则化最优传输问题的解（因此反向扩散等价于逐渐降低熵正则化的退火过程），且扩散与流匹配中分离数据模式的动力学不稳定性，在最优传输中表现为生成初期一个被熵正则化平滑的指数级薄边界层。

    

    我们讨论了生成扩散与最优传输之间的三个简单联系。首先，数据与其 Ornstein-Uhlenbeck 加噪过程的联合分布在每一时刻都是某个熵正则化最优传输问题的解，其温度由噪声水平决定。因此，反向扩散可以被解读为一种逐渐降低熵正则化的退火过程。其次，我们研究了物种形成转变（speciation transition）——即数据分布的各个模式在生成过程中如何浮现——并比较了它在扩散模型、流匹配与最优传输中的发生方式。以高斯混合分布作为高维例子，我们表明扩散与流匹配借以分离各模式的动力学不稳定性，在最优传输情形下变为生成初始阶段一个指数级薄的边界层，而该边界层会被熵正则化所平滑。第三，考虑一个带有一般前向动力学的……（摘要原文在此处截断）

    arXiv:2610.13614v1 Announce Type: cross  Abstract: We discuss three simple connections between generative diffusion and optimal transport. First, the joint law of the data and of their Ornstein-Uhlenbeck noising is, at every time, the solution of an entropic optimal-transport problem, at a temperature set by the noise level. Backward diffusion can consequently be read as an annealing which gradually decreases the entropic regularization. Second, we study the speciation transition--how the modes of the data distribution emerge during generation--and compare how it happens in diffusion and flow matching with respect to optimal transport. Using Gaussian mixtures as high-dimensional example, we show that the dynamical instability through which diffusion and flow matching separate the modes becomes, for optimal transport, an exponentially thin boundary layer at the beginning of generation, which is rounded by entropic regularization. Third, considering a generic forward dynamics with a gene
    
[^282]: TTS-Q：面向文本到语音扩散变换器的感知引导训练后量化

    TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers

    [https://arxiv.org/abs/2610.12474](https://arxiv.org/abs/2610.12474)

    本文首次对十亿参数规模的TTS扩散变换器开展4比特权重与激活（W4A4）训练后量化研究，揭示了量化主要损害说话人音色而非可懂度这一现象及其来源，并提出感知引导的量化修复方法。

    

    零样本文本到语音（TTS）技术只需几秒钟的音频即可克隆一个声音。然而其高质量的生成依赖于数十亿参数的扩散变换器，其庞大的模型规模和多步采样过程使得最需要该技术的端侧设备、辅助技术和实时应用难以使用。训练后量化（PTQ）无需重新训练即可压缩此类模型，但现有方法是为语言和视觉领域设计的，而非语音领域。在4比特精度下，强量化器虽然能让语音可懂度接近全精度水平，但会持续损害说话人音色，导致错误率保持较低的同时，声音却偏离了目标说话人。我们探究了为什么量化会损害语音、损伤存在于何处，以及能否在不重新训练的情况下进行修复。为此，我们对1B和3.5B规模的TTS扩散变换器开展了首个4比特权重与激活（W4A4）训练后量化研究。研究发现，音色损失同时来自权重量化和激活量化，但只有激活部分具有时间维度的特性……（原文摘要在此处截断）

    arXiv:2610.12474v1 Announce Type: cross  Abstract: Zero-shot text-to-speech (TTS) can clone a voice from seconds of audio. Yet the quality comes from billion-parameter diffusion transformers whose size and multi-step sampling put them out of reach of the on-device, assistive, and real-time applications that need them most. Post-training quantization (PTQ) shrinks such models without retraining, but it has been built for language and vision, not speech. At 4 bits, a strong quantizer leaves intelligibility close to full precision but consistently erodes speaker timbre, so error rates stay low while the voice drifts from the target speaker. We ask why quantization hurts speech, where the damage lives, and whether it can be repaired without retraining. To find out, we run the first 4-bit weight-and-activation (W4A4) PTQ study of a TTS diffusion transformer at 1B and 3.5B. The timbre cost comes from both the weight and the activation quantization, but only the activation part has the timest
    
[^283]: V-CoLA：基于线性注意力的视觉Token压缩

    V-CoLA: Vision Token Compression with Linear Attention

    [https://arxiv.org/abs/2610.11251](https://arxiv.org/abs/2610.11251)

    提出V-CoLA，一个专为线性注意力混合架构设计的无需训练的视觉Token压缩框架，通过唯一性感知的重要性准则和自适应Token合并策略，解决了现有压缩方法在该类架构下性能显著下降的问题。

    

    视觉语言模型（VLM）已经展现出令人印象深刻的能力，但由于视觉Token在输入序列中占主导地位，导致计算开销巨大。这促使视觉Token压缩成为减轻负担的关键方向。然而，随着融合线性注意力的混合架构（例如Qwen3.5）的出现，先前为softmax注意力设计的方法难以泛化。我们的分析表明，基于注意力和基于相似度的方法都出现了显著的性能下降，凸显了针对这一机制量身定制压缩方法的迫切需求。为此，我们提出了V-CoLA，一个专门为线性注意力设计的高效无需训练的Token压缩框架。V-CoLA引入了一种新颖的“唯一性感知重要性准则”用于识别关键的视觉Token，并结合一种“自适应Token合并策略”来执行压缩。

    arXiv:2610.11251v1 Announce Type: cross  Abstract: Vision-language models (VLMs) have demonstrated impressive capabilities but suffer from substantial computational overhead, as vision tokens dominate the input sequence. This motivates vision token compression as a key direction to alleviate the burden. However, with the emergence of hybrid architectures incorporating linear attention (\eg, Qwen3.5), prior methods designed for softmax attention struggle to generalize. Our analysis reveals that both attention- and similarity-based approaches suffer notable performance degradation, underscoring the urgent need for compression methods tailored to this regime. To this end, we propose \textbf{V-CoLA}, an efficient training-free token compression framework specifically designed for linear attention. V-CoLA introduces a novel \textit{uniqueness-aware importance criterion} for identifying critical vision tokens, coupled with an \textit{adaptive token merging strategy} that performs compression
    
[^284]: 论量子混沌动力学机器学习中的线性与非线性问题

    On linearity or non-linearity in machine learning for quantum chaotic dynamics

    [https://arxiv.org/abs/2610.10697](https://arxiv.org/abs/2610.10697)

    该研究将量子混沌动力学预测转化为时间序列预测问题，以里德堡原子阵列可实现的PXP链为基准，系统比较了非线性Transformer与简单线性模型DLinear在从遍历到量子多体疤痕态等不同动力学区间中的预测能力。

    

    精确模拟混沌量子多体动力学仍然是经典方法面临的一项重大计算挑战，原因在于演化过程中纠缠的快速累积与空间扩展。这引出了一个自然的问题：机器学习能否为预测量子动力学提供一种有效的替代方案。我们通过将量子动力学表述为一个时间序列预测问题来探讨这一问题，并以一个可由里德堡原子阵列实现的少比特PXP链作为基准。通过改变初始态，系统涵盖了从遍历行为到量子多体疤痕态等不同的动力学区间，从而为在不同性质的动力学下测试预测模型提供了一个可控的设置。我们比较了两种截然不同的架构：具有强表达能力的非线性Transformer模型与简单的线性预测模型DLinear。Transformer在较为遍历的动力学区间中能够准确预测动力学，但其性能……（原摘要在此处截断）

    arXiv:2610.10697v1 Announce Type: cross  Abstract: Accurately simulating chaotic quantum many-body dynamics remains a major computational challenge for classical methods, due to the rapid buildup and spatial spreading of entanglement during the evolution. This raises the question of whether machine learning can provide an effective alternative for predicting quantum dynamics. We address this question by formulating quantum dynamics as a time-series forecasting problem, using a few-qubit PXP chain, realizable with Rydberg-atom arrays, as a benchmark. By varying the initial state, the system spans dynamical regimes ranging from ergodic behavior to quantum many-body scarring, providing a controlled setting for testing forecasting models across qualitatively different dynamics. We compare two contrasting architectures: an expressive nonlinear Transformer and DLinear, a simple linear forecasting model. The Transformer accurately predicts dynamics in the more ergodic regime, but its performa
    
[^285]: 组合每位教师所学到的知识：通过教师相对偏移实现的多教师在线策略蒸馏

    Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts

    [https://arxiv.org/abs/2610.10460](https://arxiv.org/abs/2610.10460)

    提出 Δ-MOPD，通过迁移教师相对其基础模型的 logit 偏移并重新锚定到学生的冻结初始状态，消除端点策略中继承的基础模型偏好干扰，从而在多教师在线策略蒸馏的两种设置中降低教师项范数比和目标-学生 KL，实现更纯粹的教师后训练知识传递。

    

    多教师在线策略蒸馏（MOPD）被应用于两种设置。在共同域组合中，多个教师对来自同一提示域的学生每个 rollout 进行评分，其信号共同构成一个单一目标；在路由域蒸馏中，来自不同域的提示被分配给相应的专家教师。这两种设置通常迁移的是每个教师的端点策略，而端点策略将后训练所改变的偏好与教师从其基础模型继承而来的偏好混杂在一起。我们提出 Δ-MOPD，它迁移的是每个教师的“教师减去基础”的 logit 偏移，并将其重新锚定在学生的冻结初始状态上，同时在保持教师选择固定的情况下，将其与端点监督在两种设置中进行比较。我们首先揭示了阻碍端点迁移的机制：继承的基础拉力可能超过后训练偏移。移除它可以降低教师项范数比以及目标与学生之间的 KL 散度。在我们的所有实验中，结果表明偏移（摘要原文在此处被截断）

    arXiv:2610.10460v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation (MOPD) is used in two settings. In common-domain composition, several teachers score each student rollout from one prompt domain and their signals form a single target; in routed-domain distillation, prompts from different domains are assigned to the corresponding specialist. Both settings usually transfer each teacher's endpoint policy, which mixes what post-training changed with preferences inherited from the teacher's base. We introduce $\Delta$-MOPD, which transfers each teacher's teacher-minus-base logit shift re-anchored at the student's frozen initialization, and compare it with endpoint supervision in both settings while holding teacher selection fixed. We first expose the mechanism that impedes endpoint transfer: inherited base pull can exceed the post-training shift. Removing it reduces the teacher-term norm ratio and target--student KL.   Across our experiments, the results suggest that shi
    
[^286]: Loopy：面向循环语言模型的低比特量化框架

    Loopy: Low-Bit Quantization Framework for Looped Language Models

    [https://arxiv.org/abs/2610.05265](https://arxiv.org/abs/2610.05265)

    提出了Loopy量化框架，通过感知循环深度的目标函数为循环语言模型的共享核心选择最优低比特量化配置，解决了量化误差随循环迭代累积传播且配置排序随深度变化的问题。

    

    循环语言模型通过重复执行一个共享的循环核心，提供了一种参数高效的方式来扩展迭代式的测试时计算。训练后量化（PTQ）可以降低循环语言模型的内存占用和推理成本，但共享核心量化所引入的误差会影响后续的核心。在各类PTQ方法中，通道缩放和正交旋转能够在保持浮点计算的同时，生成具有不同量化质量的表示。我们发现，量化配置候选的排序会随循环深度而变化，这促使我们需要在目标部署深度上进行配置选择。然而，在该深度上使用完整校准集评估每一个候选的成本非常高。因此，我们提出了Loopy，一个通过感知循环深度的目标函数来形式化共享核心量化问题的PTQ框架，根据最终预测结果来选择共享的低比特表示

    arXiv:2610.05265v2 Announce Type: replace  Abstract: Looped language models provide a parameter-efficient way to scale iterative test-time computation by repeatedly executing a shared recurrent core. Post-training quantization (PTQ) can reduce the memory footprint and inference cost of looped language models, but errors introduced by a quantized shared core affect subsequent cores. Among PTQ methods, channel scaling and orthogonal rotations preserve the floating-point computation while producing representations with different quantization quality. We find that quantization configuration candidate rankings can change with recurrent depth, motivating configuration selection at the target deployment depth. However, evaluating every candidate over the full calibration set at this depth is costly. We therefore propose Loopy, a PTQ framework that formulates shared-core quantization through a recurrent-depth-aware objective, selecting shared low-bit representations by their final prediction l
    
[^287]: 具有可观测浏览深度的轮播界面排序老虎机

    Ranking Bandits for Carousel Interfaces with Observable Browsing Depth

    [https://arxiv.org/abs/2610.05220](https://arxiv.org/abs/2610.05220)

    本文提出了浏览深度可观测的轮播界面排序老虎机模型，利用浏览深度信号区分已展示未点击与从未展示的物品，并设计了基于UCB、Thompson采样和DMED的三种算法，给出了理论性能上界。

    

    轮播界面允许推荐系统直接观察用户浏览的深度。这一信号能够区分“已展示但未被点击”的物品与“从未被展示”的物品，而传统的排序老虎机模型（包括级联模型和基于位置的模型）通常将检验视为潜在变量。我们提出了一个排序老虎机问题：学习者在列表中展示 $L$ 个物品，观察用户的最大浏览深度，并仅接收该深度以内各位置的点击反馈。目标是在未知的物品吸引力向量和浏览深度分布下最大化期望点击次数。我们提出了三种基于 UCB、Thompson 采样和 DMED 的算法，它们都仅根据观测到的曝光来更新物品统计信息。我们为基于 UCB 的算法推导了实例相关的对数上界，并为基于 DMED 的算法推导了渐近上界，该上界与……

    arXiv:2610.05220v2 Announce Type: replace  Abstract: Carousel interfaces allow a recommender system to directly observe how far a user has browsed. This signal distinguishes displayed but unclicked items from items that were never displayed, whereas conventional ranking-bandit models, including cascade and position-based models, generally treat examination as latent. We formulate a ranking-bandit problem in which a learner presents a list of $L$ items, observes the user's maximum browsing depth, and receives click feedback only for positions up to that depth. The objective is to maximize the expected number of clicks under an unknown item-attractiveness vector and a browsing-depth distribution. We propose three algorithms based on UCB, Thompson Sampling, and DMED, all of which update item statistics only from observed exposures. We derive an instance-dependent logarithmic upper bound for our UCB-based algorithm and an asymptotic upper bound for our DMED-based algorithm that coincides w
    
[^288]: EDISCO：面向欧几里得组合优化的等变离散扩散模型

    EDISCO: Equivariant DIScrete Diffusion for Euclidean Combinatorial Optimization

    [https://arxiv.org/abs/2610.04953](https://arxiv.org/abs/2610.04953)

    EDISCO是首个面向欧几里得组合优化问题、生成分布具有精确E(2)不变性的离散扩散模型，它通过等变边评分网络与连续时间马尔可夫链相结合，实现高效的多步推理，无需依赖数据增强或正则化。

    

    欧几里得组合优化问题（ECOPs），例如旅行商问题（TSP）和带容量约束的车辆路径问题（CVRP），在二维欧几里得群E(2)下具有固有的对称性，包括旋转、反射和平移。现有的基于学习的方法，包括近期的基于扩散的方法，都依赖数据增强或正则化来近似实现E(2)等变性。本文提出了EDISCO，这是首个针对ECOPs的离散扩散模型，其生成分布在节点索引解上具有精确的E(2)不变性。EDISCO引入了一个E(2)等变的边评分网络，与作用在离散边变量上的分类型连续时间马尔可夫链相耦合，并通过精确后验采样实现高效的多步推理。这种设计赋予EDISCO一种局部几何归纳偏置：具有相同相对几何结构和组合上下文的边邻域会被一致地表示

    arXiv:2610.04953v2 Announce Type: replace  Abstract: Euclidean combinatorial optimization problems (ECOPs), such as the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP), possess inherent symmetries under the two-dimensional Euclidean group E(2), including rotations, reflections, and translations. Existing learning-based methods, including recent diffusion-based methods, rely on data augmentation or regularization to approximate E(2)-equivariance. This paper presents EDISCO, the first discrete diffusion model for ECOPs with exact E(2)-invariant generative distributions over node-index solutions. EDISCO introduces an E(2)-equivariant edge-score network coupled with a categorical continuous-time Markov chain over discrete edge variables, and exact posterior sampling provides efficient multi-step inference. This design gives EDISCO a local geometric inductive bias: edge neighborhoods with the same relative geometry and combinatorial context are represented co
    
[^289]: TreeWalker：面向分组树集成推理的部分求值方法

    TreeWalker: Partial Evaluation for Grouped Tree-Ensemble Inference

    [https://arxiv.org/abs/2610.03939](https://arxiv.org/abs/2610.03939)

    TreeWalker 提出针对分组树集成推理的部分求值方法，利用组内行共享特征值的相关性，对每组仅遍历每棵树一次即可完成推理，并给出了每棵树工作量的结构化分解，同时兼容标准 LightGBM 和 XGBoost 模型。

    

    许多预测任务需要在共享特征值的行组上评估训练好的树集成模型：离散时间生存模型将每位患者扩展为 G 个时间步，点击率模型对搜索会话中的每个物品进行打分，情景分析则在固定其余输入的同时仅改变少数几个输入。标准的推理方式将每一行独立处理，忽略了这种输入侧的相关性。我们提出 TreeWalker，它为分组树集成推理实例化了部分求值思想：恒定特征为静态，变化特征为动态。TreeWalker 对每个组仅遍历每棵树一次，在变化分裂处对行位掩码进行划分，并通过无分裂捷径跳过空的子树。它可直接读取标准的 LightGBM 和 XGBoost 模型，训练过程无需任何改动。我们证明了一个结构化的工作量分解定理：每棵树的工作量可拆分为常数投影子树大小 |T_c|、G 次叶子写入，以及谓词掩码供给成本 Q。对于该轨迹（摘要在此处截断）

    arXiv:2610.03939v2 Announce Type: replace  Abstract: Many prediction tasks evaluate a trained tree ensemble on groups of rows that share feature values: discrete-time survival models expand each patient into $G$ time steps, click-through-rate models score every item in a search session, and scenario analyses vary a few inputs while holding the rest fixed. Standard inference treats each row independently and ignores this input-side correlation.   We present TreeWalker, which instantiates partial evaluation for grouped tree-ensemble inference: constant features are static, varying features dynamic. TreeWalker walks each tree once per group, partitions a row bitmask at varying splits, and skips empty subtrees via an unsplit shortcut. It reads standard LightGBM and XGBoost models; training is unchanged.   We prove a structural work decomposition: per-tree work splits into the constant-projected subtree size $|T_c|$, $G$ leaf writes, and a predicate-mask provisioning cost $Q$. For the trace
    
[^290]: BridgeCast：借助引入外生变量的流匹配，将海浪数值预报桥接至再分析数据

    BridgeCast: Bridging Ocean Wave Forecasts to Reanalysis via Flow Matching with Exogenous Variables

    [https://arxiv.org/abs/2610.03759](https://arxiv.org/abs/2610.03759)

    BridgeCast 提出了一种基于条件流匹配与 Transformer 架构的物理-AI 混合概率模型，能够将物理模式的海浪预报偏差校正为再分析级别的场，并可融合外生大气变量实现灵活的时空推断。

    

    海浪预报对海上安全、海上作业以及沿海抗灾能力至关重要，但由于基于物理的模型存在系统性偏差，其预报仍然充满挑战。物理模型虽然被广泛使用，但依赖近似和参数化方法，这在复杂的海气条件下限制了其精度。为了提升海浪预报水平，我们提出了 BridgeCast——一个物理-AI 混合框架下的偏差校正方法。BridgeCast 是一个基于条件流匹配的概率模型，它学习将物理模型预报转换为类似再分析的场。该方法将物理预报视为受污染的观测，并采用连续时间生成过程，将其分布桥接至再分析数据的分布。BridgeCast 采用基于 Transformer 的架构进行参数化，支持时空建模、引入外生大气变量，并可借助此实现灵活的推断。

    arXiv:2610.03759v3 Announce Type: replace-cross  Abstract: Ocean wave forecasting is essential for maritime safety, offshore operations, and coastal resilience, yet remains challenging due to systematic biases in physics-based models. Physical models, while widely used, rely on approximations and parameterizations that limit their accuracy under complex ocean-atmosphere conditions. To enhance ocean wave forecasting, we propose BridgeCast, within a physics-AI hybrid framework for bias correction. BridgeCast is a probabilistic model based on conditional flow matching (CFM) that learns to transform physical model forecasts into reanalysis-like fields. It treats physical forecasts as corrupted observations and employs a continuous-time generative process to bridge their distribution toward that of reanalysis data. BridgeCast is parameterized by a Transformer-based architecture that enables spatiotemporal modeling, incorporation of exogenous atmospheric variables, and flexible inference via
    
[^291]: FERPO：前向熵正则化策略优化

    FERPO: Forward Entropy-Regularized Policy Optimization

    [https://arxiv.org/abs/2610.02198](https://arxiv.org/abs/2610.02198)

    FERPO提出了一种无需对评论家求动作导数的在策略最大熵强化学习算法，通过熵与KL散度正则化的策略改进目标推导最优动作分布，并借助自归一化重要性采样以最小化前向KL来拟合Actor，从而避免了因价值预测不准导致的不可靠策略更新。

    

    在连续控制的在线强化学习中，若干最先进的方法利用学习到的评论家对动作的梯度来改进策略。然而，评论家通常被训练用于预测回报，准确的价值预测并不一定能产生准确的动作导数，这可能导致不可靠的策略更新。我们提出前向熵正则化策略优化（FERPO），这是一种在策略的最大熵强化学习算法，它利用评论家的价值进行策略改进，而无需对评论家关于动作进行求导。FERPO从由熵和Kullback-Leibler（KL）散度正则化的策略改进目标中推导出最优的目标动作分布。随后，我们通过最小化前向KL目标将Actor拟合到该目标分布上，该目标使用自归一化重要性采样（SNIS）进行估计，动作从rollout策略中抽取。通过限制目标分布……

    arXiv:2610.02198v1 Announce Type: cross  Abstract: Several state-of-the-art methods for online reinforcement learning in continuous control improve policies using action gradients of a learned critic. However, critics are typically trained to predict returns, and accurate value predictions do not necessarily yield accurate action derivatives, potentially leading to unreliable policy updates. We propose Forward Entropy-Regularized Policy Optimization (FERPO), an on-policy maximum entropy reinforcement learning algorithm that performs policy improvement using critic values without differentiating the critic with respect to actions. FERPO derives an optimal target action distribution from a policy-improvement objective regularized by entropy and Kullback-Leibler (KL) divergence. We then fit the actor to this target by minimizing a forward-KL objective, estimated using self-normalized importance sampling (SNIS) with actions drawn from the rollout policy. By limiting the target distribution
    
[^292]: 探索更多，推理更优：面向扩散语言模型的逐步风险敏感GRPO

    Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models

    [https://arxiv.org/abs/2610.00661](https://arxiv.org/abs/2610.00661)

    提出StepRS-GRPO方法，通过在去噪状态间动态调整群组优势变换的风险系数，使扩散语言模型在数学推理任务上同时提升pass@1准确率、pass@k覆盖率和答案多样性。

    

    扩散大语言模型通过去噪一个序列或连续的多个块来生成文本，能够并行地揭示多个 token。基于可验证奖励的强化学习（RLVR）会在这些决策中重复使用终端反馈，即使决策所依赖的上下文已经发生变化。我们提出逐步风险敏感的 GRPO（StepRS-GRPO），它在不同去噪状态之间改变群组优势变换的风险系数，同时保留底层训练器。对于二元奖励，我们证明了这种变换恰好是对中心化结果优势进行的、依赖于提示和状态的重新缩放。基于能力的校准给出了系数的合适量级，而端点与插值消融实验则指导调度方案的选择。在多个 dLLM 骨干模型和数学推理基准上，StepRS-GRPO 相比中心化 GRPO 提升了 pass@1 准确率和 pass@k 覆盖率，同时增加了答案多样性。

    arXiv:2610.00661v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) generate text by denoising a sequence or successive blocks, allowing several tokens to be revealed in parallel. Reinforcement learning with verifiable rewards (RLVR) reuses terminal feedback across these decisions, even as their conditioning context changes. We propose stepwise risk-sensitive GRPO (StepRS-GRPO), which varies the risk coefficient of the group-advantage transformation across denoising states while retaining the underlying trainer. For binary rewards, we show that this transformation is exactly a prompt- and state-dependent rescaling of centered outcome advantages. A capability-based calibration suggests a coefficient scale, while endpoint and interpolation ablations guide schedule selection. Across multiple dLLM backbones and mathematical reasoning benchmarks, StepRS-GRPO improves both pass@1 accuracy and pass@k coverage over centered GRPO, while increasing answer diversity. In our
    
[^293]: 在线非单调DR-次模最大化的几何相关界

    Geometry-Dependent Bounds for Online Non-Monotone DR-Submodular Maximization

    [https://arxiv.org/abs/2610.00545](https://arxiv.org/abs/2610.00545)

    该论文为在线非单调DR-次模最大化证明了系数4/9的一阶遗憾界（改进此前0.401的基准），给出依赖于集合几何的更优系数，并通过对称间隙构造建立了离线预言机上界β*≈0.4704。

    

    我们研究在紧凸下闭集上对非负、非单调DR-次模函数的对抗性在线最大化问题。学习者在观测到目标值之前必须先提交每一步动作，并与事后最优的固定动作竞争。我们证明了一个对比较器一致的一阶不等式，得到系数4/9，改进了此前在线0.401的基准，每轮仅需一次梯度查询和一次投影，期望近似遗憾为O(√T)。若 ζ1 ∈ K ⊆ [0,1]^d，系数可进一步改进为 α̲(ζ) = 1/2 − (1−2ζ)₊²/[2(3−2ζ)²]。证明采用直接的有序坐标论证，并构造了一个与目标无关的理性行动。反之，一个三组对称间隙构造给出了ζ=0时的离线预言机上界 β* = 0.470438681380894...，即使拥有精确函数值和完整梯度响应该上界依然成立。此外还包括一个参数化扩展及精确有限实例界……

    arXiv:2610.00545v1 Announce Type: new  Abstract: We study adversarial online maximization of nonnegative, non-monotone DR-submodular functions over compact convex down-closed sets. A learner commits each action before observing its objective and competes with the best fixed action in hindsight. We prove a comparator-uniform first-order inequality that gives coefficient $4/9$, improving the online $0.401$ benchmark, with one gradient query and one projection per round and $O(\sqrt T)$ expected approximate regret. If $\zeta {\bf 1} \in K\subseteq[0,1]^d$, the coefficient improves to $\underline\alpha(\zeta)=\tfrac12-(1-2\zeta)_+^2/[2(3-2\zeta)^2]$. The proof is a direct ordered-coordinate argument with an objective-independent rational action. Conversely, a three-group symmetry-gap construction yields an offline oracle upper bound $\beta_*=0.470438681380894\ldots$ at $\zeta=0$, even with exact value and full-gradient responses. A parameterized extension and exact finite-instance bounds d
    
[^294]: S³：谱零空间交换使推理模型更高效

    $S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient

    [https://arxiv.org/abs/2609.37976](https://arxiv.org/abs/2609.37976)

    论文首次发现推理能力的关键在于思考模型权重中位于非思考模型主导子空间零空间内的分量，并提出无需训练的谱零空间交换方法S³，通过配对检查点组合在保持精度的同时大幅提升推理效率。

    

    经思维链训练的大语言模型在推理能力上表现出色，但往往伴随着过多的token开销。我们发现推理能力的核心在于思考模型的权重分量位于由对应非思考模型主导奇异方向所定义投影的零空间内，并且移除该子空间分量可以在不损害思考模式后训练所获得精度的前提下大幅提升推理效率。与现有大多在主导子空间内进行操作的方法不同，我们首次揭示了零空间的关键作用，并将其用于模型优化。受这一发现的启发，我们提出了谱零空间交换（S³），这是一种无需训练的、由配对的非思考与思考检查点组成的融合方法。我们的方法将非思考模型保留在其自身的主导子空间内，而将思考检查点置于该子空间之外，在提升推理效率的同时保持（摘要此处被截断）

    arXiv:2609.37976v1 Announce Type: cross  Abstract: LLMs trained with Chain-of-thought excel in reasoning capability, but often come with excessive token cost. We find that the core of reasoning capacity lies in the Thinking model's weight component within the null space of a projection defined by the corresponding Non-thinking model's dominant singular directions, and removing the subspace component can largely improve reasoning efficiency without hurting the accuracy gained during thinking-mode post-training. Unlike existing efforts that mostly operate within the dominant subspace, we are the first to unveil the critical role of the null space and harness it for model optimization. Motivated by this finding, we propose Spectral Null-Space Swap ($S^3$), a training-free composition of paired Non-thinking and Thinking checkpoints. Our method keeps the Non-thinking model inside its own dominant subspace and takes the Thinking checkpoint outside it, improving reasoning efficiency while mai
    
[^295]: ROSS：通过选择性监督从自生成rollout序列中再学习

    ROSS: Relearning from Self-Generated Rollouts through Selective Supervision

    [https://arxiv.org/abs/2609.35954](https://arxiv.org/abs/2609.35954)

    ROSS通过将完整历史rollout轨迹保留为上下文、仅对选定的模型生成延续施加损失的选择性监督方式，有效再利用历史自生成经验，在强化学习、多教师蒸馏和智能体学习中持续提升大语言模型性能。

    

    大语言模型的后训练过程通过强化学习和同策略蒸馏生成自生成的rollout序列，然而这些经验往往在策略更新后就被视为过时数据而丢弃。历史rollout序列可能与后续策略仍然兼容，同时保留着策略已无法稳定表达的行为；但它们也可能包含不应被模仿的错误、被放弃的尝试和冗余动作，这促使我们需要更细粒度的选择性监督。我们提出了ROSS（通过选择性监督从自生成rollout序列中再学习），该方法将完整的历史轨迹保留为上下文，同时仅对选定的模型生成延续部分施加损失。在领域专用强化学习、多教师同策略蒸馏和智能体强化学习等场景中，ROSS能够持续提升上游检查点的性能，并在数学、代码生成、指令遵循等任务上超越基线方法。

    arXiv:2609.35954v1 Announce Type: cross  Abstract: Large language model post-training generates self-generated rollouts through reinforcement learning and on-policy distillation, yet this experience is often treated as stale once the policy advances. Historical rollouts can remain compatible with a later policy while preserving behaviors that the policy no longer expresses reliably. However, they may also contain mistakes, abandoned attempts, and redundant actions that should not be imitated, motivating finer-grained selective supervision. We introduce ROSS (Relearning from Self-Generated Rollouts through Selective Supervision), which preserves the full historical trajectory as context while applying loss only to selected model-generated continuations. Across domain-specific reinforcement learning, multi-teacher on-policy distillation, and agentic reinforcement learning, ROSS consistently improves upstream checkpoints and outperforms baselines across mathematics, code generation, instr
    
[^296]: 二值化使分数空间坍缩扁平

    Binarization Flattens the Score Space

    [https://arxiv.org/abs/2609.35797](https://arxiv.org/abs/2609.35797)

    将LLM评审奖励二值化为通过/不通过会完全掩盖底层分数空间中的仿射拉伸信息，而引入第三个等级即可通过第二阈值消除这种歧义并恢复可检测的分数细节。

    

    大语言模型（LLM）评审常被用作奖励信号，用于在确定性验证器无法覆盖的目标上训练策略。然而，这些奖励通常被坍缩为通过/不通过（{0, 1}），这只报告判定结果，却不反映回答在每条标准上的具体表现。我们将每个通过/不通过判定建模为某一未公开量表上的分数，并与单一截断阈值进行比较。对该量表的拉伸会按比例地将每个分数推向或推离截断点，但绝不会使其越过截断点，因此任何判定都不会改变。因此，策略可以自由地施加任意拉伸而不改变评审面板所报告的任何内容。在联合高斯模型下，引入第三个等级会添加第二个阈值，从而消除这种仿射拉伸带来的歧义。在一个七标准评审器对 MATH 和 SciBench 输出的实验中，所有 14 个构造的按标准拉伸在二值化后均不可见，但在三个等级下均可被检测到。在 n=1,024 时，给定两种总体规律的检验具有至少 96.5% 的功效。

    arXiv:2609.35797v1 Announce Type: cross  Abstract: Large language model (LLM) judges are often used as rewards to train policies on objectives that deterministic verifiers cannot capture. However, these rewards are often collapsed to pass/fail ({0, 1}), which reports the verdict but not how well a response met each criterion. We model each pass/fail verdict as a score on an unreported scale, compared with one cutoff. A stretch of that scale moves every score proportionally toward or away from the cutoff, but never across it, so no verdict changes. A policy is therefore free to apply any stretch without changing anything the panel reports. Under a joint-Gaussian model, a third grade adds a second threshold and removes this affine stretch ambiguity. On MATH and SciBench outputs from one seven-criterion judge, all 14 constructed criterionwise stretches were invisible after binarization but visible with three grades. At $n=1{,}024$, a test given both population laws had at least 96.5% powe
    
[^297]: QwenGyre：一种用于训练超长时程智能体的弹性强化学习框架

    QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents

    [https://arxiv.org/abs/2609.33848](https://arxiv.org/abs/2609.33848)

    QwenGyre提出了一种面向超长时程LLM智能体的端到端弹性在线强化学习框架，通过在不中断执行的情况下在rollout与训练之间动态重分配GPU，并利用轨迹处理器重建分支历史、为部分进展打分和去重冗余路径，有效解决了GPU大量闲置与轨迹冗余两大难题。

    

    大语言模型（LLM）智能体越来越多地承担极端长时程任务，单次执行可能持续数小时、包含数百次模型-环境交互，且每次rollout接近100万token。将在线强化学习（RL）应用于此类执行面临两个根本性挑战：（1）严重的执行方差和漫长的rollout延迟导致大量GPU闲置；（2）复杂的非线性分支产生海量轨迹冗余，严重损害训练效率。为解决这些问题，我们提出了QwenGyre，一个面向超长时程在线RL的端到端框架。QwenGyre能够在不中断正在进行的执行的情况下，在rollout和训练之间弹性地重新分配GPU；同时，其轨迹处理器可以重建分支历史、为部分进展打分，并对冗余路径去重，从而控制训练成本。在扩展到我们的旗舰模型Qwen 3.8 2.4T（每次rollout达70万token）时，QwenGyre带来了6.0%的提升

    arXiv:2609.33848v2 Announce Type: replace  Abstract: Large language model (LLM) agents increasingly undertake extreme-long (xlong) horizon tasks, where a single execution can span hours, hundreds of model--environment interactions, and nearly 1M tokens per rollout. Applying online reinforcement learning (RL) to such executions poses two fundamental challenges: (1) severe execution variance and prolonged rollout delays cause massive GPU idling; and (2) complex non-linear branching generates massive trajectory redundancy, crippling training efficiency. To address these, we presents QwenGyre, an end-to-end framework for xlong-horizon online RL. QwenGyre elastically reallocates GPUs between rollout and training without interrupting live executions, while its trajectory processor reconstructs branching histories, scores partial progress, and deduplicates redundant paths to bound training costs. Scaled to our flagship model, Qwen~3.8 2.4T, with 700K tokens per rollout, QwenGyre yields a 6.0%
    
[^298]: 边际校准在多变量时间序列留出评估中掩盖了生成器差异——以ERA5天气序列为例的应用

    Marginal Calibration Masks Generator Differences in Held-Out Evaluation of Multivariate Time Series, with an Application to ERA5 Weather Sequences

    [https://arxiv.org/abs/2609.33761](https://arxiv.org/abs/2609.33761)

    该论文证明了在多变量时间序列生成模型的留出评估中，边际校准会使KS距离几乎不携带生成器信息，从而掩盖不同生成模型之间的真实差异，并通过ERA5天气序列上StatD2GAN与RCGAN、TimeGAN的对比实验验证了这一结论。

    

    多变量时间序列生成模型通常采用边际校准后计算的池化分布指标进行评估。我们证明，在留出协议下，校准后的Kolmogorov-Smirnov（KS）距离与“真实对真实”的基准下限之间的差异至多等于校准样本到训练块的距离，因此该指标几乎不携带关于生成器本身的信息。我们在预先指定的协议下，将具有进化权重自适应机制的三判别器GAN（StatD2GAN）与重新实现的RCGAN和TimeGAN基线模型在ERA5天气序列上进行了审计：最后两个日历年经168小时禁运期后被留出，校准仅在训练块上拟合，决策规则事先冻结。证据来自五个地点的25组匹配的（地点，随机种子）对，并使用Holm校正的Wilcoxon符号秩检验进行分析。首先，等保序回归校准使KS距离的中位数达到1.0

    arXiv:2609.33761v2 Announce Type: replace  Abstract: Generative models for multivariate time series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show that, under a held-out protocol, the calibrated Kolmogorov-Smirnov (KS) distance differs from a real-versus-real floor by at most the distance of the calibrated sample to the training block, so that it carries little information about the generator. We audit a three-discriminator GAN with evolutionary weight adaptation (StatD2GAN) against re-implemented RCGAN and TimeGAN baselines on ERA5 weather sequences under a pre-specified protocol: the final two calendar years are held out behind a 168 hour embargo, calibration is fitted on the training block only, and the decision rules were frozen in advance. Evidence comes from 25 matched (location, seed) pairs across five locations, tested with Holm-corrected Wilcoxon signed-rank tests. First, isotonic calibration drives KS to a median of 1.0
    
[^299]: 基于显式锚点-正样本-负样本耦合的多边际逆向最优传输对比学习

    Multi-Marginal Inverse Optimal Transport for Contrastive Learning Via Explicit Anchor-Positive-Negative Coupling

    [https://arxiv.org/abs/2609.33741](https://arxiv.org/abs/2609.33741)

    该论文提出Neg-MMIOT-CL方法，通过显式建模锚点-正样本-负样本三元组的多边际逆向最优传输耦合，克服了现有逆向OT方法因不利用负样本而导致的维度坍塌问题，提升了对比学习的下游性能。

    

    基于逆向最优传输（OT）的表示学习方法，通过学习表示使得表示空间中一对数据边缘分布之间的全局OT耦合集中在正样本对上，这与以往主要关注成对匹配的方法形成对比。然而，这些方法并未利用负样本对，因此其方式并非真正意义上的对比学习。我们证明这会导致维度坍塌问题，进而降低下游任务性能。为缓解这一问题，我们开发了一种新颖的多边际（MM）逆向最优传输（IOT）对比学习（CL）方法，称为Neg-MMIOT-CL，该方法学习表示，使得三个数据边缘分布之间的全局多边际最优传输（MMOT）耦合——相对于表示空间中数据点三元组之间精心设计的基础代价函数——集中在锚点-正样本-负样本三元组上。

    arXiv:2609.33741v2 Announce Type: replace  Abstract: Inverse Optimal Transport (OT) based methods for representation learning learn representations such that the global OT coupling between a pair of data marginals in the representation space, concentrates on the positive pairs. This is in contrast to previous methods that primarily focused on pairwise matching. However, these methods $\textit{DO NOT}$ utilize negative pairs and hence are not truly contrastive in their approach. We show that this leads to issues of dimensional collapse and hence degraded downstream performance. To alleviate this, we develop a novel multi-marginal (MM) inverse OT (IOT) contrastive learning (CL) approach called Neg-MMIOT-CL, which learns representations such that the global multi-marginal OT (MMOT) coupling between a triple of data marginals, with respect to a carefully designed ground-cost between triplets of data points in the representation space, concentrates on the anchor-positive-negative $\textit{t
    
[^300]: 秩能买到什么？低秩适配（LoRA）的谱与分布分析

    What Does the Rank Buy? A Spectral and Distributional Analysis of Low-Rank Adaptation

    [https://arxiv.org/abs/2609.32002](https://arxiv.org/abs/2609.32002)

    该论文证明在硬性逐因子范数预算下，LoRA的秩上限无法起到容量控制作用——所有复杂度与位移泛函均由秩一更新最大化、Rademacher复杂度与秩无关且分布位移上界与秩无关且紧致——并指出了秩真正发挥作用的两个位置。

    

    LoRA中的秩 $r$ 被广泛视为一种容量控制手段：人们通常假设较小的秩会带来更简单、泛化能力更强的模型。我们证明，在硬性逐因子范数预算——即实践中权重衰减与范数控制的理想化——之下，这一直觉会失效。其原因在于结构层面：在这类预算下，LoRA所能实现的更新恰好是核范数球内秩至多为 $r$ 的矩阵集合，而我们分析的每一个复杂度与位移泛函在该集合上均由秩一更新取得最大值——因此秩上限从未真正起到约束作用。由此可直接得出若干结论：我们所研究的线性读出模型类对所有 $r \ge 1$ 都是完全相同的，其Rademacher复杂度不依赖于 $r$，且适配所能使源分布产生的位移服从一个与秩无关的上界，我们证明该上界是紧的。如果秩并不控制容量，那它的作用体现在哪里？我们确定了两个位置。（原文摘要在此处被截断）

    arXiv:2609.32002v2 Announce Type: replace  Abstract: The rank $r$ in LoRA is widely treated as a capacity control: a smaller rank is assumed to yield a simpler model that generalizes better. We show that, under hard per-factor norm budgets---the idealization of the weight decay and norm control used in practice---this intuition breaks down. The reason is structural: under such budgets, the updates LoRA can reach are exactly the matrices of rank at most $r$ inside a nuclear-norm ball, and every complexity and displacement functional we analyze is maximized over this set by a rank-one update---so the rank cap never binds. The consequences follow directly. The linear-readout model class we study is identical for every $r \ge 1$, its Rademacher complexity carries no dependence on $r$, and the distance the adaptation can move the source distribution obeys a rank-independent upper bound that we show is sharp. If rank does not control capacity, where does it act? We identify two places. Stati
    
[^301]: KuaFu：在十亿级规模上将长用户行为压缩为用户理解

    KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

    [https://arxiv.org/abs/2609.31045](https://arxiv.org/abs/2609.31045)

    该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。

    

    对话式智能体、生成式推荐器和个性化广告都依赖一项核心能力：从原始行为中理解每一位用户。当前主流的工业实践是任务专用的：针对每个任务，从完整历史中提取相关子序列，并在其上训练专用模型。在生产环境中，这遇到了两个瓶颈。第一，即使经过过滤，单一任务的序列仍然极长：内容兴趣摘要需要读取每位用户的数百条内容，一旦序列化为提示文本就达数万个token。第二，用户画像需要例行刷新：每周覆盖十亿用户，总计约10万QPM，在固定的GPU预算下这设定了一个硬性的吞吐量下限。因此压缩是必不可少的，但截断或粗糙的压缩会悄然扭曲用户画像，引入四种幻觉类型（捏造、遗漏、日期错误归属、逻辑断裂），而由于无法评估压缩（原文在此处截断）……

    arXiv:2609.31045v1 Announce Type: cross  Abstract: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compr
    
[^302]: 基于电子先验的响应态学习实现可迁移的振动光谱表征

    Response-state Learning for Transferable Vibrational Spectroscopic Characterization with Electron Prior

    [https://arxiv.org/abs/2609.28935](https://arxiv.org/abs/2609.28935)

    提出SO(3)等变神经卡尔曼网络SENK，通过等变Transformer主干、神经卡尔曼桥与NBO电子先验通路构建响应态级联，实现从小分子到类药体系跨化学空间的可迁移、高保真IR/Raman光谱预测，并在QM9S和QMe14S上超越DetaNet。

    

    当局域立体电子环境扰动中间响应态、且高风险响应单元主导特征光谱指纹时，振动光谱预测可能变得不准确，使得跨外部化学空间的预测十分困难。SO(3)等变神经卡尔曼网络（SENK）构建了一种响应态级联结构，其结合了用于Hessian、偶极导数和极化率导数学习的等变Transformer主干网络、用于状态相关精炼与可靠性感知的等变神经卡尔曼桥，以及将一致性正则化与有界的、分支特定的引导光谱校准相耦合的NBO信息电子先验通路。SENK在QM9S和QMe14S数据集上超越了DetaNet，同时从小分子到类药体系均保持了全谱红外（IR）和拉曼（Raman）光谱的保真度。在具有复杂局域电子结构的生物分子体系中，SENK保持稳定并有选择性地改进了光谱敏感特征。

    arXiv:2609.28935v1 Announce Type: new  Abstract: Vibrational spectral prediction can become inaccurate when localized stereoelectronic environments perturb intermediate response states and high-risk response units dominate characteristic spectral fingerprints, making prediction across external chemical space difficult. SO(3) Equivariant Neural Kalman Networks (SENK) form a response-state cascade that combines an equivariant transformer backbone for Hessian, dipole-derivative and polarizability-derivative learning, an Equivariant Neural Kalman bridge for state-dependent refinement and reliability sensing, and an NBO-informed electronic-prior pathway coupling consistency regularization with bounded, branch-specific guided spectral calibration. SENK outperforms DetaNet on QM9S and QMe14S while preserving full-spectrum IR and Raman fidelity from small molecules to drug-like systems. SENK remains stable and selectively improves spectrally sensitive features in biomolecular systems with comp
    
[^303]: EMERGE：基于等变图扩散的分辨率无关点云生成

    EMERGE: Resolution-Agnostic Point Cloud Generation with Equivariant Graph-Based Diffusion

    [https://arxiv.org/abs/2609.26039](https://arxiv.org/abs/2609.26039)

    提出了EMERGE，首个完全SE(3)等变的基于图的扩散模型骨干网络，实现了分辨率无关的点云生成，并支持在多个任意分辨率下的零样本推理。

    

    点云生成已成为准确捕捉和再现物理世界复杂性的关键任务。然而，现有的生成方法主要依赖Transformer和变分自编码器（VAE），往往忽略了三维空间固有的连续、非网格拓扑结构。尽管基于图的结构在相关的判别式视觉任务中已带来显著收益，但此类几何架构在三维生成建模中仍明显缺失。为填补这一空白，我们提出了EMERGE（用于分辨率无关点云生成的等变多尺度图神经网络），这是首个完全SE(3)等变的基于图的扩散骨干网络，专为在保持连续空间对称性的同时生成点云而设计。我们的框架摆脱了标准生成流程的刚性分辨率依赖，支持在多个任意分辨率下的零样本推理。

    arXiv:2609.26039v1 Announce Type: cross  Abstract: Point cloud generation has emerged as a crucial task for accurately capturing and reproducing the complexity of the physical world. However, existing generative approaches, predominantly relying on Transformers and Variational Autoencoders (VAEs), frequently ignore the continuous, non-grid topologies inherent to 3D spaces. Although the integration of graph-based structures has yielded significant benefits in related discriminative vision tasks, such geometric architectures remain noticeably absent from 3D generative modeling. To address this gap, we introduce EMERGE (Equivariant Multi-scale GNN for Resolution-agnostic point cloud GEneration), the first fully $SE(3)$-equivariant graph-based diffusion backbone explicitly designed to generate point clouds while preserving continuous spatial symmetries. Our framework bypasses the rigid resolution dependencies of standard generative pipelines, enabling zero-shot inference at multiple, arbit
    
[^304]: AURA：用于训练复值神经网络的角向更新率自适应方法

    AURA: Angular Update Rate Adaptation for training complex-valued neural networks

    [https://arxiv.org/abs/2609.25914](https://arxiv.org/abs/2609.25914)

    AURA是一种即插即用的复值神经网络逐参数步长自适应方法，通过衡量复参数连续更新在长度、对齐和旋转方向上的一致性来动态调整步长，无需额外梯度计算即可叠加于任何一阶优化器之上且不改变其更新方向。

    

    复值神经网络（CVNNs）在复值数据处理中的应用日益增多；然而，它们通常采用从实值情形继承而来的一阶优化器进行训练。这些方法的效率在很大程度上取决于步长，而其步长规则忽略了复平面上可用的角度信息。我们通过引入AURA（角向更新率自适应，Angular Update Rate Adaptation）来解决复数域中的步长自适应问题。AURA是一种逐参数的步长自适应方法，可以叠加在任何一阶优化器之上，也可以从中移除，且不会改变其更新方向。AURA从长度、对齐和旋转方向三个方面衡量每个复参数连续更新之间的一致性，当更新一致时增大步长，不一致时减小步长。它不需要额外的梯度计算，每步仅需低成本的向量运算。我们将AURA与Adam和Muon相结合并进行比较……

    arXiv:2609.25914v1 Announce Type: new  Abstract: Complex-valued neural networks (CVNNs) are increasingly adopted for complex-valued data; however, they are often trained with first-order optimizers inherited from the real-valued case. The efficiency of these methods depends largely on the step size, and their step-size rules ignore the angular information available in the complex plane. We address step-size adaptation in the complex domain by introducing AURA (Angular Update Rate Adaptation), a per-parameter step-size adaptation that can be added on top of any first-order optimizer, and removed from it, without altering its update direction. AURA measures the agreement between consecutive updates of each complex parameter, in length, alignment, and sense of rotation, and enlarges the step when they are consistent and reduces it when they are not. It requires no additional gradient evaluations and only inexpensive vector operations per step. We combine AURA with Adam and Muon and compar
    
[^305]: 复杂度拐点：面向代码生成可靠性的提示侧结构复杂度指数

    The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability

    [https://arxiv.org/abs/2609.19616](https://arxiv.org/abs/2609.19616)

    该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。

    

    从生成代码中测量的复杂度依赖于失败情况：一个困难的提示可能产生一个简短的失败程序，从而被赋予较低的输出复杂度。我们提出了一个六维度的提示侧结构复杂度指数，在生成之前进行评分，并与正确性保持分离。我们从初步单评分者量表的六个区间中选择了5,000个Python提示。四个面板外的LLM评分者对锁定的提示重新评分，得到19,997条评分记录；在四个评分齐全的4,998个提示上，综合评分者间信度为ICC = 0.872。我们对每个提示评估21个模型，共产生105,000个生成结果。在未经调整的均值池化分析中，通过率在综合分数13.75处出现非单调断点，该分数及以下的通过率为79.9%，以上为87.6%。这并非一个通用的失败临界点。任务类型固定效应将断点移至10.75，并将区间差距从7.6个百分点缩小至2.1个百分点。构建框架控制将其移至8.50，原始差距为……

    arXiv:2609.19616v1 Announce Type: new  Abstract: Complexity measured from generated code is failure-dependent: a difficult prompt can yield a short failing program and be assigned low output complexity. We introduce a six-dimension prompt-side structural-complexity index scored before generation and kept separate from correctness. We select 5,000 Python prompts across six bands of a preliminary single-rater rubric. Four out-of-panel LLM raters rescore the locked prompts, giving 19,997 score rows; composite inter-rater reliability is ICC = 0.872 on the 4,998 prompts with all four ratings. We evaluate 21 models per prompt, yielding 105,000 generations. In the unadjusted mean-pooled analysis, pass rate has a nonmonotone breakpoint at composite 13.75, with 79.9% at or below and 87.6% above. This is not a universal failure cutoff. Task-type fixed effects shift the breakpoint to 10.75 and cut the regime gap from 7.6 to 2.1 points. A construction-frame control shifts it to 8.50 with a raw gap
    
[^306]: 基于|D|维稀疏地标嵌入的非CND距离度量通用核框架

    A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings

    [https://arxiv.org/abs/2609.19083](https://arxiv.org/abs/2609.19083)

    提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。

    

    核方法，尤其是高斯过程（GP），需要希尔伯特距离度量（即其平方为条件负定（CND）的度量）来保证核矩阵的半正定性（PSD）；而这一条件在许多自然输入空间上并不成立，包括光滑流形和概率分布空间。我们提出了稀疏地标嵌入（SLE）核，彻底消除了这一要求。每个输入通过以全部|D|个训练点为中心的紧支撑凸块函数被嵌入为稀疏特征向量；在该嵌入空间中应用任何标准PSD核，即可得到对任意距离度量均可证明为半正定的核。紧支撑特性自动控制了嵌入的稀疏性，使得尽管环境维度很高，核矩阵仍保持良态且计算上可行。我们对PSD性质、稀疏性、稳定性以及普适性提供了理论保证。

    arXiv:2609.19083v1 Announce Type: cross  Abstract: Kernel methods, and Gaussian Processes (GPs) in particular, require a Hilbertian distance measure---one whose square is conditionally negative definite (CND)---to guarantee positive semi-definiteness (PSD) of the kernel matrix; a condition that fails for many natural input spaces, including smooth manifolds and spaces of probability distributions. We propose the Sparse Landmark Embedding (SLE) kernel, which eliminates this requirement entirely. Each input is embedded into a sparse feature vector via compactly supported bump functions centered at all |D| training points; applying any standard PSD kernel in this embedding space yields a kernel that is provably PSD for arbitrary distance measures. The compact support automatically controls embedding sparsity, keeping kernel matrices well-conditioned and computationally tractable despite the high ambient dimension. We provide theoretical guarantees on PSD, sparsity, stability, and universa
    
[^307]: 动态网络上的去中心化最优均衡学习

    Decentralized Optimal Equilibrium Learning Over Dynamic Networks

    [https://arxiv.org/abs/2609.17601](https://arxiv.org/abs/2609.17601)

    该论文提出了一种适用于动态通信网络的去中心化最优均衡学习算法，智能体通过交换带时间戳的表格数据与时间多数重建机制学习社会最优均衡，并在功利主义和比例公平社会福利目标下实现了有限时间对数遗憾保证。

    

    本文研究了动态通信网络上有限标准型博弈中社会最优均衡的去中心化学习问题。每个智能体只能观测到自身实现的收益，对博弈本身没有先验知识，且只能使用低带宽消息与随时间变化的邻居进行通信。我们提出了网络化去中心化最优均衡学习动力学，其中智能体从局部收益比较中生成随机化的语义满足/不满信号，并交换带时间戳的时间堆叠表格，而非原始动作、收益信息或局部估计/参数。该方法将表格融合与时间多数重建相结合，以应对动态通信的挑战，同时保持完全去中心化的运行。我们在功利主义和比例公平两种社会福利目标下，为最优均衡选择建立了带有同相探索扰动的有限时间对数遗憾保证。

    arXiv:2609.17601v1 Announce Type: cross  Abstract: This paper studies decentralized learning of socially optimal equilibria in finite normal-form games over dynamic communication networks. Each agent observes only its own realized payoffs, does not know the game a priori, and can communicate only with time-varying neighbors using low-bandwidth messages. We propose networked decentralized optimal equilibrium learning dynamics in which agents generate randomized semantic content/discontent signals from local payoff comparisons and exchange time-stamped time-stacked tables rather than raw actions, payoff information or local estimates/parameters. The method combines table fusion with temporal majority reconstruction to mitigate dynamic communication while preserving fully decentralized operation. We establish finite-time logarithmic regret guarantees, with an in-phase exploration perturbation, for optimal equilibrium selection under utilitarian and proportional-fair social welfare objecti
    
[^308]: 序列预测中的真实校准度量

    Truthful Calibration Measures for Sequential Prediction

    [https://arxiv.org/abs/2608.21348](https://arxiv.org/abs/2608.21348)

    本文证明了在序列二值预测中，精确真实的校准度量无法同时满足完备性和健全性，并提出了两种近似真实的构建方法，实现了可扩展的乘性近似校准度量。

    

    校准要求概率报告在条件上无偏，并可靠地解释为概率。校准度量将数值误差分配给校准不当的报告。Haghtalab等人（2024）提出了一种用于在线预测的近似真实校准度量，但留下了精确真实性与完备性和健全性是否兼容的未解问题。我们针对序列二值预测给出了否定性答案：即使对于独立结果，精确真实性与完备性和健全性也不兼容。随后，我们表明这种不可能性仅特定于精确真实性。我们给出了两种从基础校准度量出发的一般性简化，分别产生加性近似真实和乘性近似真实的校准度量。应用乘性简化，对于每个$0 < \varepsilon < 1$，我们构建了一个健全且完备的校准度量，该度量是$(1+\exp(-T^{(1-\varepsilon)/2}))$-乘性近似的。

    arXiv:2608.21348v1 Announce Type: cross  Abstract: Calibration requires probabilistic reports to be conditionally unbiased and reliably interpretable as probabilities. A calibration measure assigns numerical error to miscalibrated reports. Haghtalab et al. (2024) proposed an approximately truthful calibration measure for online prediction, leaving open whether exact truthfulness is compatible with completeness and soundness.   We resolve this question negatively for sequential binary prediction: exact truthfulness is incompatible with completeness and soundness, even for independent outcomes. We then show that this impossibility is specific to exact truthfulness. We give two general reductions from a base calibration measure, producing additively and multiplicatively approximately truthful calibration measures, respectively. Applying the multiplicative reduction, for every $0 < \varepsilon < 1$ we construct a sound and complete calibration measure that is $(1+\exp(-T^{(1-\varepsilon)/2
    
[^309]: 利用仅幅值测量和复数测量结合学习先验改进动态MRI重建

    Harnessing Magnitude-Only and Complex Measurements for Improved Dynamic MRI Reconstruction with Learned Priors

    [https://arxiv.org/abs/2608.18036](https://arxiv.org/abs/2608.18036)

    本文提出一种利用k空间幅值信息与复数测量相结合的物理驱动深度学习重建方法，显著提升动态MRI重建质量。

    

    摘要：针对欠采样k空间数据的MRI重建方法通常利用复数测量值。稀疏相位恢复领域的并行发展表明，仅幅值测量可能为信号恢复提供互补信息。然而，由于缺乏在不增加额外扫描时间的情况下获取信息性幅值测量的实际场景，这些方法在MRI重建中的应用仍未得到充分探索。在本工作中，我们研究了辅助k空间幅值信息在加速稳态动态MRI重建中的应用，并展示了k空间幅值在时间帧间的强一致性。基于这一观察，我们提出了$\mathbb{C}+\text{Mag}$，一种幅值信息引导的物理驱动深度学习重建方法。该方法采用基于ADMM的展开框架，并引入一种新颖的幅值感知数据保真度公式，其中二次平滑优化...

    arXiv:2608.18036v1 Announce Type: cross  Abstract: MRI reconstruction methods for undersampled k-space data naturally utilize complex-valued measurements. Parallel developments in sparse phase retrieval have shown that magnitude-only measurements may provide complementary information for signal recovery. However, their use in MRI reconstruction remains largely unexplored, due to lack of practical settings where informative magnitude measurements can be obtained without additional scan time. In this work, we investigate the use of auxiliary k-space magnitude information for accelerated steady-state dynamic MRI reconstruction, and demonstrate strong consistency of k-space magnitudes across time-frames. Building on this observation, we propose $\mathbb{C}+\text{Mag}$, a magnitude-informed physics-driven deep learning reconstruction method. The proposed method employs an ADMM-based unrolling framework with a novel magnitude-aware data-fidelity formulation, where quadratically smoothed opti
    
[^310]: ARISE：一种自适应残差信息稳定性集成方法，用于小样本生物医学组学中的特征选择

    ARISE: An adaptive residual-informed stability ensemble for feature selection in small-sample biomedical omics

    [https://arxiv.org/abs/2608.14866](https://arxiv.org/abs/2608.14866)

    ARISE通过自适应加权组合多种相关性信号和残差信息冗余控制，在小样本生物医学组学数据中显著提升了特征选择的预测性和稳定性。

    

    摘要：目标：小样本分子分类需要能够为二元和多类结果识别预测性、稳定性和非冗余子集的特征选择器。我们提出了ARISE（自适应残差信息稳定性集成），它整合了互补的相关性信号、类平衡的稳定性评估、残差信息冗余控制以及多类配对覆盖。方法：ARISE通过15个预定义配置文件组合七个百分位归一化的相关性成分，并通过嵌套内部交叉验证自适应加权。它在五个分子数据集、八个特征集大小、三个固定分类器（k近邻、支持向量机和随机森林）以及六个过滤比较器上进行了评估。泛化性能通过五次外部交叉验证重复50次，使用平衡准确率、宏观F1分数和Cohen's kappa进行估计。结果：在210,000次留出评估中，ARISE排名第一。

    arXiv:2608.14866v1 Announce Type: cross  Abstract: Objective: Small-sample molecular classification requires feature selectors that identify predictive, stable, and nonredundant subsets for binary and multiclass outcomes. We propose ARISE (Adaptive Residual-Informed Stability Ensemble), which integrates complementary relevance signals, class-balanced stability assessment, residual-informed redundancy control, and multiclass pairwise coverage.   Methods: ARISE combines seven percentile-normalized relevance components through 15 predefined profiles, adaptively weighted by nested inner cross-validation. It was evaluated on five molecular datasets, eight feature-set sizes, three fixed classifiers (k-nearest neighbours, support vector machine, and random forest), and six filter comparators. Generalization was estimated by five-fold outer cross-validation repeated 50 times using balanced accuracy, macro-F1, and Cohen's kappa.   Results: Across 210,000 held-out assessments, ARISE ranked first
    
[^311]: 无偏性之外的CFR：持久公共机会调度的确定性保证

    CFR without Unbiasedness: Deterministic Guarantees for Persistent Public-Chance Schedules

    [https://arxiv.org/abs/2608.14761](https://arxiv.org/abs/2608.14761)

    本文提出了一种确定性目标转移定理，针对持久部分评估的CFR，在均匀加法公共截断下，通过引入公共借方项界定了可利用性，并证明了连续平衡调度在RM和RM+下的收敛性。

    

    在有限的公共机会截断处，反事实遗憾最小化（CFR）必须在每次遗憾更新前选择评估多少个结果。精确评估在单一策略配置下处理完整截断；持久部分评估则在不断演化的配置中处理固定的无替换顺序。后者每轮覆盖每个结果一次，但其反馈通常是有条件有偏的，因为较早的批次会影响后续批次所见的配置。我们为均匀、非嵌套的加法公共截断建立了一个确定性的目标转移定理。该定理通过所传递反馈上的遗憾以及一个公共借方项来界定完整截断的可利用性，其中公共借方项将前缀覆盖差异与沿实现策略路径的运动耦合在一起。因此，在预定平均权重下，连续平衡调度对于加法有符号遗憾匹配（RM）和RM+收敛，而一个固定的RM+构造则证明了这一结果。

    arXiv:2608.14761v1 Announce Type: cross  Abstract: At a finite public-chance cut, counterfactual regret minimization (CFR) must choose how many outcomes to evaluate before each regret update. Exact evaluation processes the full cut at one strategy profile; persistent partial evaluation processes a fixed without-replacement order across evolving profiles. The latter covers every outcome once per epoch, yet its feedback is generally conditionally biased because earlier batches influence the profiles seen by later batches. We establish a deterministic target-transfer theorem for uniform, nonnested additive public cuts. The theorem bounds full-cut exploitability by regret on the delivered feedback and a public-debit term that couples prefix coverage discrepancy with motion along the realized strategy path. Consecutively balanced schedules consequently converge for additive signed regret matching (RM) and RM+ under predetermined averaging weights, while a fixed RM+ construction proves that 
    
[^312]: 基于共轭对偶的广义凸性与光滑性及其在深度网络训练中的应用

    Generalized Convexity and Smoothness via Conjugate Duality, with Applications to Deep Network Training

    [https://arxiv.org/abs/2608.09523](https://arxiv.org/abs/2608.09523)

    该论文通过勒让德函数与凸共轭提出了统一的 \(\mathcal{H}(\psi)\)-凸性与 \(\mathcal{H}(\Psi)\)-光滑性框架，突破了传统优化理论对可微性、凸性和光滑性的假设限制，并为深度神经网络训练建立了新的理论基础，包括证明广义梯度下降的最优学习率恰好为 1。

    

    随机梯度下降（SGD）及其变体在实践中能够有效地训练深度神经网络（DNN），然而经典优化理论并不能完全解释这一成功：传统分析依赖于可微性、凸性和光滑性等假设，而深度神经网络的目标函数往往违反这些假设。我们通过勒让德函数和凸共轭，将凸性与光滑性进行推广，构建了面向深度神经网络训练的统一优化框架，从而弥合了这一理论缺口。我们引入了 \(\mathcal{H}(\psi)\)-凸性与 \(\mathcal{H}(\Psi)\)-光滑性，这一统一的形式化体系同时涵盖了凸与非凸、光滑与非光滑的目标函数，并揭示了广义凸性与光滑性之间的自然对偶关系。基于这些概念，我们通过凸共轭导出了广义梯度下降（GD）和广义随机梯度下降（SGD）。我们证明了广义梯度下降能够取得恰好为 1 的最优学习率，并建立了……

    arXiv:2608.09523v2 Announce Type: replace  Abstract: Stochastic gradient descent (SGD) and its variants train deep neural networks (DNNs) effectively in practice, yet classical optimization theory does not fully explain this success: conventional analyses rest on differentiability, convexity, and smoothness, assumptions that DNN objectives routinely violate. We close this gap by generalizing convexity and smoothness, through Legendre functions and convex conjugation, into a unified optimization framework for DNN training. We introduce \(\mathcal{H}(\psi)\)-convexity and \(\mathcal{H}(\Psi)\)-smoothness, a single formalism that encompasses convex and non-convex as well as smooth and non-smooth objectives and reveals a natural duality between generalized convexity and smoothness. Building on these notions, we derive generalized gradient descent (GD) and generalized SGD via convex conjugation. We prove that generalized GD attains an optimal learning rate of exactly \(1\), and we establish
    
[^313]: FEAST：面向资源异构客户端的联邦共享空间训练

    FEAST: Federated Shared-Space Training for Resource-Heterogeneous Clients

    [https://arxiv.org/abs/2608.09250](https://arxiv.org/abs/2608.09250)

    FEAST通过在每个客户端资源预算内联合训练多个子网络，并结合预算定制的子超网路由与稀疏聚合机制，解决了联邦学习中因客户端资源异构导致的参数训练不平衡问题，且支持无需联邦重训即可事后提取新子网络。

    

    联邦学习（FL）必须服务于具有不同计算能力的设备。固定模型无法适配所有设备，而针对每种部署限制单独训练一个模型的成本又很高。联邦超网训练转而学习一个包含不同大小子网络的弹性模型，然后为每个设备部署合适的子网络。然而，当客户端的推理预算各不相同时，高成本子网络独有的参数只能被较少的客户端访问到。我们提出了FEAST，一个联邦共享空间训练框架，通过在每个客户端的资源限制内联合训练多个子网络来对抗这种不平衡。预算定制的子超网路由只发送相关的超网部分，稀疏聚合则合并返回的参数切片。训练得到的超网可直接服务于联邦训练期间使用的子网络，并支持在不进行联邦重训的情况下事后提取额外的子网络。我们进一步表明，独立……

    arXiv:2608.09250v2 Announce Type: replace  Abstract: Federated learning (FL) must serve devices with varying computational capabilities. A fixed model cannot suit all devices, while training one model per deployment limit is costly. Federated supernet training instead learns one elastic model with differently sized subnetworks, then deploys a suitable one to each device. When client inference budgets differ, however, parameters exclusive to high-cost subnetworks are reachable by fewer clients. We propose FEAST, a federated shared-space training framework that counters this imbalance by jointly training multiple subnetworks within each client's limit. Budget-tailored sub-supernet routing sends only the relevant supernet portion, and sparse aggregation merges the returned parameter slices. The trained supernet directly serves the subnetworks used during federation and supports post-hoc extraction of additional subnetworks without federated retraining. We further show that independently a
    
[^314]: 利用数据驱动先验指导近红外化学计量学中的CNN架构设计

    Using Data-Derived Priors to Guide CNN Architecture Design for NIR Chemometrics

    [https://arxiv.org/abs/2607.25636](https://arxiv.org/abs/2607.25636)

    该论文提出将25个近红外光谱数据集的内在特性（如规模、熵、自相关等）转化为CNN架构设计的经验先验，特别是卷积感受野的设置，并通过热启动启发式加速贝叶斯超参数优化，经留一数据集验证证明了其有效性。

    

    用于近红外（NIR）化学计量学的卷积神经网络（CNN）通常基于通用的架构规则进行设计，然而光谱数据集在采样、平滑度、冗余性和样本量方面各不相同。我们检验了这些数据特性是否能为CNN设计提供经验先验。在25个NIR回归任务中，我们计算了数据集规模、光谱长度与间距、熵、内在秩、自相关以及小波尺度结构等描述符。两个可解释的一维CNN框架（一个最小化的单卷积模型和一个带有可选分支、膨胀卷积等扩展的浅层模型）通过五折交叉验证的贝叶斯超参数优化（HPO）进行优化。从接近最优的试验中提取出的关系被转化为热启动启发式规则，并通过直接评估和留一数据集（LODO）验证加以检验。最清晰的关系涉及卷积感受野的选择。

    arXiv:2607.25636v2 Announce Type: replace  Abstract: Convolutional neural networks (CNN) for near-infrared (NIR) chemometrics are often designed using generic architectural rules, although spectral datasets differ in sampling, smoothness, redundancy, and sample size. We tested whether these properties can provide empirical priors for CNN design. Across 25 NIR regression tasks, we computed descriptors of dataset size, spectral length and spacing, entropy, intrinsic rank, autocorrelation, and wavelet-scale structure. Two interpretable 1D-CNN scaffolds (a minimal single-convolution model and an extended shallow model with optional branching, dilation, etc) were optimized using five-fold cross-validated Bayesian hyperparameter optimization (HPO). Relationships extracted from near-optimal trials were converted into warm-start heuristics and evaluated directly and through leave-one-dataset-out (LODO) validation. The clearest relationships involved convolutional receptive fields. In the minim
    
[^315]: 求解椭圆狄利克雷问题的边界自适应物理信息神经网络的 $H^2(\Omega)$ 先验误差界

    $H^2(\Omega)$ A Priori Error Bounds for Boundary-Adapted PINNs Solving Elliptic Dirichlet Problems

    [https://arxiv.org/abs/2607.19167](https://arxiv.org/abs/2607.19167)

    本文通过乘以适当的到边界距离函数精确施加边界条件，构造了“边界自适应”神经网络，证明精确施加边界条件（而非 $L^2$ 边界惩罚）可以实现椭圆狄利克雷问题的 $H^2(\Omega)$ 误差控制，并推导出显式的高概率收敛速率。

    

    我们研究通过在神经网络假设空间中最小化偏微分方程残差来数值求解一致椭圆狄利克雷边值问题（BVPs），其中神经网络通过乘以用户定义的“到边界距离近似”函数 $\rho$ 来精确满足边界条件。虽然 $L^2(\partial\Omega)$ 边界惩罚通常无法保证 $H^2(\Omega)$ 误差控制，但我们证明了只要 $\rho$ 选择得当，精确施加边界条件可以实现这种控制。更确切地说，我们确定了 $\rho$ 上保持 $H^2(\Omega)$ 误差控制所需正则性的一个尖锐充分条件，为一般光滑区域 $\Omega$ 提供了此类 $\rho$ 的实用构造，并将由此得到的网络称为“边界自适应”网络。对于边界自适应的整流二次单元网络和双曲正切网络，我们推导出了显式的高概率 $H^2(\Omega)$ 收敛速率。

    arXiv:2607.19167v2 Announce Type: replace-cross  Abstract: We study the numerical resolution of uniformly elliptic Dirichlet boundary value problems (BVPs) by minimization of the PDE residual over a hypothesis space of Neural Networks made to satisfy exactly the boundary conditions through multiplication by a user-defined \emph{distance-to-boundary approximation} $\rho$. While an $L^2(\partial\Omega)$ boundary penalty cannot generally guarantee $H^2(\Omega)$ error control, we show that exact enforcement can, provided $\rho$ is chosen appropriately. More precisely, we identify a sharp sufficient condition on $\rho$ preserving the regularity needed for $H^2(\Omega)$ error control, provide practical constructions of such $\rho$ for general smooth domains $\Omega$, and refer to the resulting networks as \emph{boundary-adapted}. For boundary-adapted Rectified Quadratic Unit (ReQU) and hyperbolic tangent (tanh) networks, we then derive explicit high-probability $H^2(\Omega)$ convergence rate
    
[^316]: DiffEEG：一种用于学习脑电图癫痫发作相关表征的自监督去噪扩散模型

    DiffEEG: A Self-Supervised Denoising Diffusion Model for Learning EEG seizure-related Representations

    [https://arxiv.org/abs/2607.11578](https://arxiv.org/abs/2607.11578)

    DiffEEG是一个960万参数的自监督基础模型，通过去噪扩散预训练与强化学习微调，在标注稀缺和极端类别不平衡的条件下学习脑电图癫痫发作表征，并在患者级别严格评估中取得了优异的癫痫检测性能。

    

    基于脑电图（EEG）的癫痫发作检测深度学习面临关键挑战：严重的标注稀缺性和极端的类别不平衡，其中癫痫发作事件在临床记录中占比不足10%。我们提出了DiffEEG，一个960万参数的自监督基础模型，通过去噪扩散预训练和基于强化学习（RL）的微调来同时解决这两个局限。该模型在天普大学医院癫痫语料库（TUHSZ）的130万个无标注片段上进行预训练，通过带有多头自注意力机制的一维U-Net学习通用的神经表征。在下游适配中，强化决策层采用策略梯度优化来直接最大化F1分数，优先考虑对罕见癫痫发作事件的敏感性而非整体准确率。在严格的患者级别评估下（279名患者，留一折交叉验证），DiffEEG在四分类癫痫发作亚型分类任务中达到61%的准确率和59%的F1分数，以及81%的准确率（原文摘要至此截断）。

    arXiv:2607.11578v2 Announce Type: replace-cross  Abstract: Deep learning for EEG-based seizure detection faces critical challenges: severe annotation scarcity and extreme class imbalance, where ictal events comprise less than 10\% of clinical recordings. We present DiffEEG, a 9.6M-parameter self-supervised foundation model that addresses both limitations through denoising diffusion pre-training and reinforcement learning (RL)-based fine-tuning. Pre-trained on 1.3M unlabeled segments from the Temple University Hospital Seizure Corpus (TUHSZ), DiffEEG learns generic neural representations via a 1D U-Net with multi-head self-attention. For downstream adaptation, a reinforced decision layer employs policy gradient optimization to directly maximize F1-score, prioritizing sensitivity to rare seizure events over overall accuracy. Under strict patient-wise evaluation (279 patients, Leave-One-Fold-Out), DiffEEG achieves 61\% accuracy and 59\% F1 for 4-class seizure subtyping, and 81\% accuracy 
    
[^317]: SafeExplorer：一种用于带恢复干预的强化学习的无偏混合策略梯度

    SafeExplorer: An Unbiased Mixed-Policy Gradient for Reinforcement Learning with Recovery Interventions

    [https://arxiv.org/abs/2607.08925](https://arxiv.org/abs/2607.08925)

    提出 SafeExplorer，一种基于混合策略回报无偏梯度估计器的 PPO 即插即用改进方法，即使在使用确定性恢复策略进行安全强化学习训练时也能消除在线策略更新偏差。

    

    直接在智能体将要部署的任务上进行训练通常能带来最佳的策略性能，但在物理机器人上训练可能因跌倒而造成设备损坏，因此最小化训练过程中的跌倒次数至关重要。一种标准的缓解方法是在智能体离开设计者指定的安全区域时，将控制权移交给一个单独的恢复策略。由此产生的混合策略轨迹会使在线策略更新产生偏差，而当恢复策略为确定性策略时，重要性采样无法消除这种偏差。我们提出了 SafeExplorer，这是对近端策略优化（PPO）的一种即插即用式修改，其建立在混合策略回报的无偏策略梯度估计器之上。该估计器仅在安全区域内使用得分函数，从不评估恢复策略的概率密度，因此确定性恢复策略不会构成问题。在最终奖励方面，该估计器的表现与截断重要性采样相当或更优，而后者在同样的梯度估计中本身就是有偏的。

    arXiv:2607.08925v2 Announce Type: replace  Abstract: Training agents directly on the task they will be deployed on often leads to the best policy performance, but training on physical robots can cause damage through falls, so minimizing training falls is critical. A standard mitigation hands control to a separate recovery policy whenever the agent leaves a designer-specified safe region. The resulting mixed-policy rollouts bias the on-policy update, and importance sampling cannot remove this bias when the recovery policy is deterministic. We propose SafeExplorer, a drop-in modification of proximal policy optimization (PPO) built on an unbiased policy-gradient estimator of the mixed-policy return. The estimator uses the score function only inside the safe region and never evaluates the recovery policy's density, so deterministic recovery policies pose no problem. On final reward, the estimator matches or outperforms truncated importance sampling, which is itself biased for that same gra
    
[^318]: 数据驱动的软标签将DNA读段分类扩展至全身细胞类型反卷积

    Data-Driven Soft Labeling Scales DNA Read Classification to Whole-Body Cell-Type Deconvolution

    [https://arxiv.org/abs/2607.04987](https://arxiv.org/abs/2607.04987)

    该论文提出一种数据驱动的软标签方法，能够将DNA读段分类规模化扩展至全身细胞类型反卷积任务。

    

    本文根据同行评审意见进行了修订。我们扩展了基线比较，修正了评估中的数据泄漏和读段边界处理问题，澄清了置信度加权损失函数，并增加了关于池化和区域选择的敏感性分析。我们还扩展了TCS失效模式和局限性分析，新增了讨论部分，并提供了代码和数据链接以保证研究的可复现性。

    arXiv:2607.04987v3 Announce Type: replace  Abstract: Revised following peer review. We expanded baseline comparisons, corrected evaluation leakage and read-boundary handling, clarified the confidence-weighted loss, and added sensitivity analyses for pooling and region selection. We also expanded TCS failure-mode and limitations analyses, added a discussion section, and provided code and data links for reproducibility.
    
[^319]: 探测而非提示：用于Multi-Meta-RAG元数据过滤的隐藏状态探针

    Probe, Don't Prompt: A Hidden-State Probe for Metadata Filtering in Multi-Meta-RAG

    [https://arxiv.org/abs/2607.03929](https://arxiv.org/abs/2607.03929)

    该论文提出用小型开源语言模型隐藏状态上训练的本地确定性多标签探针，替代Multi-Meta-RAG中基于GPT-3.5的新闻来源元数据提取器，从而免去专有API调用并将输出严格限制在允许的来源列表内。

    

    Multi-Meta-RAG通过在元数据（具体为新闻来源和发布日期）上过滤向量存储来改进多跳问答的检索效果，这些元数据由GPT-3.5从每个查询中提取。该提取器需要对每个查询进行专有API调用，且其输出并不受允许来源列表的约束：尽管提示中指定了49个来源，GPT-3.5却生成了128个不同的来源名称，其中79个不在列表之内。我们证明，该提取器的来源部分可以由一个本地、确定性的探针替代，该探针在小型开源语言模型的隐藏状态上训练得到。该探针是作用于某一层均值池化隐藏状态的多标签逻辑回归分类器，使用类别平衡损失在黄金证据来源上训练。通过5折交叉验证来选择层，我们在全部2556个MultiHop-RAG查询上对参数量从135M到1.5B的四个模型进行了折外评估。

    arXiv:2607.03929v2 Announce Type: replace-cross  Abstract: Multi-Meta-RAG improves retrieval for multi-hop question answering by filtering a vector store on metadata, specifically the news source and the publication date, which it extracts from each query using GPT-3.5. This extractor requires a proprietary API call for every query, and its output is not restricted to the allowed sources: even though the prompt specifies 49 sources, GPT-3.5 generates 128 distinct source names, 79 of which do not appear on the list. We show that the source part of this extractor can be replaced by a local, deterministic probe trained on the hidden states of a small open-source language model. The probe is a multi-label logistic classifier on the mean-pooled hidden states of one layer, trained with a class-balanced loss on the gold evidence sources. Using 5-fold cross-validation to select the layer, we evaluate four models ranging from 135M to 1.5B parameters out-of-fold across all 2556 MultiHop-RAG quer
    
[^320]: 具有重尾梯度噪声和通信噪声的联邦学习：一种基于方差缩减的算法

    Federated learning with heavy-tailed gradient noise and communication noise: a variance-reduction based algorithm

    [https://arxiv.org/abs/2606.22466](https://arxiv.org/abs/2606.22466)

    提出了一种基于方差缩减的联邦学习算法VRA-FedSGD，通过动量方差缩减结合非线性映射以及方差缩减聚合机制，有效应对重尾梯度噪声和通信噪声，并在均方和几乎必然意义下实现了对非凸目标的收敛速率保证。

    

    联邦学习（FL）是一种新兴的分布式机器学习范式，它使本地设备能够在保持数据去中心化和隐私性的同时共同训练一个全局模型。我们提出了一种基于方差缩减的算法VRA-FedSGD，用于在重尾梯度噪声和通信噪声存在情况下的联邦学习，这类噪声在无线网络和物联网部署的大规模机器学习中普遍存在。VRA-FedSGD采用动量方差缩减技术结合非线性映射来缓解重尾梯度噪声，并使用方差缩减聚合机制来抑制重尾通信噪声。在均方意义下，VRA-FedSGD对于非凸目标函数实现了 O(K^{-(p-1)/(2p-1)}) 的收敛速率，其中 p 是重尾噪声的尾指数。在几乎必然意义下，VRA-FedSGD实现了……

    arXiv:2606.22466v2 Announce Type: replace  Abstract: Federated learning (FL) is an emerging distributed machine learning paradigm that enables local devices to jointly train a global model while keeping data decentralized and private. We propose a variance-reduction based algorithm, VRA-FedSGD, for FL in the presence of heavy-tailed gradient noise and communication noise, where these noises are prevalent in large-scale machine learning over wireless networks and Internet of Things deployments. VRA-FedSGD employs a momentum variance reduction technique together with a nonlinear mapping to mitigate heavy-tailed gradient noise, and uses a variance-reduced aggregation mechanism to suppress heavy-tailed communication noise. In the mean sense, VRA-FedSGD achieves a convergence rate of {\small$\mathcal{O}\left(K^{-(p-1)/(2p-1)}\right)$} for nonconvex objective functions, where $p$ is the tail index of heavy-tailed noise. In the almost sure sense, VRA-FedSGD achieves a convergence rate of $\ti
    
[^321]: 内存受限大语言模型服务中的服务诱导型拥塞

    Service-Induced Congestion in Memory-Constrained LLM Serving

    [https://arxiv.org/abs/2606.15555](https://arxiv.org/abs/2606.15555)

    该论文将内存受限的LLM服务建模为离散时间动力学系统，揭示了服务过程本身会内生地累积显存压力并引发请求驱逐，并证明了在同质负载下无驱逐平衡态不稳定，系统通常会收敛到带驱逐的极限环。

    

    在大语言模型（LLM）服务中，每个请求在服务过程中会持续占用GPU显存，因为其键值缓存会随着每个生成令牌而不断增长。在高并发情况下，总体内存使用量因此会随时间内生地增长：服务过程本身会制造未来的容量压力。当超出内存容量时，系统会驱逐活跃请求，丢弃已缓存的状态并在之后重新启动它们，这会浪费计算资源并降低吞吐量。我们建立了一个内存受限LLM推理的离散时间动力学模型，用以刻画连续批处理下的请求准入、内存增长与驱逐行为。在饱和输入状态下，系统既存在无驱逐的不动点，也存在带驱逐的极限环。对于同质工作负载，我们证明无驱逐平衡态是不稳定的，并且除了一个勒贝格测度为零的精确捕获集合之外，系统均收敛于……（摘要原文在此处截断）

    arXiv:2606.15555v3 Announce Type: replace-cross  Abstract: In large language model (LLM) serving, each request accumulates persistent graphics processing unit (GPU) memory during service as its key-value cache grows with every generated token. Under high concurrency, aggregate memory usage therefore increases endogenously over time: the service process itself creates future capacity pressure. When memory capacity is exceeded, systems evict active requests, discarding cached state and restarting them later, which wastes computation and reduces throughput. We develop a discrete-time dynamical model of memory-constrained LLM inference that captures admission, memory growth, and eviction under continuous batching. In the saturated-input regime, the system admits both eviction-free fixed points and limit cycles with evictions. For homogeneous workloads, we show that the eviction-free equilibrium is unstable and that, except for a Lebesgue-measure-zero exact-capture set, the system converges
    
[^322]: 面向大学学术管理信息系统（ACMIS）的AI安全代理：多向量威胁检测与自动响应

    An AI Security Agent for University ACMIS: Multi-Vector Threat Detection and Automated Response

    [https://arxiv.org/abs/2606.08270](https://arxiv.org/abs/2606.08270)

    本文提出了一种面向大学学术管理信息系统的AI安全代理，通过融合监督式异常检测、行为分析与NLP密码找回聊天机器人，对五个运营层面进行多向量威胁监控，并借助四级风险升级框架实现自动化响应。

    

    大学学术管理信息系统（ACMIS）是各类安全威胁的高价值攻击目标，这些威胁包括暴力破解登录攻击、支付欺诈、权限提升、内部人员数据窃取以及学术诚信违规行为。传统的基于规则的入侵检测系统难以胜任，因为许多恶意活动在结构上与正常操作难以区分。本文提出了一种基于AI的ACMIS安全代理，该代理结合了监督式异常检测、行为分析以及用于安全密码找回的自然语言处理聊天机器人。该代理监控五个运营层面：身份认证、授权、金融交易、用户行为和系统健康状态，并通过四级风险升级框架进行响应。模块化架构使得核心引擎可以扩展到其他机构系统。在一个包含14……（注：原文摘要在此处被截断，内容不完整）

    arXiv:2606.08270v3 Announce Type: cross  Abstract: University Academic Management Information Systems (ACMIS) are high-value targets for a wide spectrum of security threats including brute-force login attacks, payment fraud, privilege escalation, insider data theft, and academic integrity violations. Traditional rule-based intrusion detection systems are inadequate because many malicious activities are structurally indistinguishable from normal operations. This paper presents an AI-based security agent for ACMIS that combines supervised anomaly detection, behavioural analytics, and a natural language processing chatbot for secure password recovery. The agent monitors five operational layers: authentication, authorisation, financial transactions, user behaviour, and system health, and responds through a four-tier risk escalation framework. A modular architecture allows the core engine to be extended to other institutional systems. Experiments on a simulated ACMIS event log dataset of 14
    
[^323]: 多群体直推学习的代价

    The price of multi-group transductive learning

    [https://arxiv.org/abs/2606.04423](https://arxiv.org/abs/2606.04423)

    该论文揭示了在留一直推设定下，多群体学习器的错误率可能遭受随群体数量线性增长（最大达样本量平方根）的乘性惩罚，而在群体可实现的统计设定下，该惩罚仅为与群体数量无关的多对数因子。

    

    我们证明了在留一直推设定下，任何多群体学习器在某些群体上的错误率相对于单群体设定下可达到的错误率，都可能遭受出人意料的大幅乘性惩罚。事实上，该惩罚可以随群体数量线性增长，最大可达约样本量的平方根。我们还证明了在类似的（群体可实现的）统计设定中，该惩罚始终至多为样本量的多对数因子，且与群体数量无关。

    arXiv:2606.04423v2 Announce Type: replace  Abstract: We show every multi-group learner in the leave-one-out transductive setting may incur a surprisingly large multiplicative penalty in its error rate on some group relative to the error rate achievable in the single-group setting. Indeed, the penalty can increase linearly with the number of groups, up to roughly the square-root of the sample size. We also show that in the analogous (group-realizable) statistical setting, the penalty is always at most a poly-logarithmic factor in the sample size, independent of the number of groups.
    
[^324]: 分解混合专家模型中的拒绝引导机制

    Decomposing Refusal Steering in Mixture-of-Experts Models

    [https://arxiv.org/abs/2606.04160](https://arxiv.org/abs/2606.04160)

    该研究将拒绝引导方法扩展到混合专家模型，发现单个专家在自由选择位置时平均可恢复78%的完整引导效果，从而揭示了MoE模型中拒绝机制在各组件间的运作方式。

    

    arXiv:2606.04160v2 公告类型：替换 摘要：指令微调大语言模型（LLM）的安全对齐依赖于模型可靠地拒绝有害或违规请求的能力。近期研究表明，可以在推理阶段向稠密LLM施加引导向量以抑制拒绝行为，从而诱导模型对有害请求作出响应。我们将这种拒绝引导方法扩展到三个开源混合专家模型上，以分解并更好地理解拒绝机制如何在MoE的各个组件中运作。我们发现，引导性能不受MoE架构固有的复杂路由模式的影响；当允许专家自由选择自身位置时，单个专家平均可恢复78%的完整引导效果。然而，当专家级引导被限制在全层引导所使用的位置时，平均仅能恢复54%的效果，且在安全相关系统提示下这一差距进一步扩大。我们的结果还表明，拒绝信号...

    arXiv:2606.04160v2 Announce Type: replace  Abstract: Safety alignment in instruction-tuned large language models (LLMs) depends on a model's ability to reliably refuse harmful or disallowed requests. Recent work has shown that a steering vector can be applied to a dense LLM during inference to suppress refusal behavior and induce responses to harmful requests. We extend this refusal steering method to three open-source Mixture-of-Experts (MoE) LLMs to decompose and better understand how refusal mechanisms operate across MoE components. We find that steering performance is uninhibited by the complex routing patterns inherent to the MoE architecture, and that a single expert recovers 78% of the full steering effect on average when free to select its own location. However, expert-level steering only recovers 54% on average when constrained to the location used by full-layer steering, a gap that widens further under safety-related system prompts. Our results also show that refusal signals 
    
[^325]: 智慧的形状：语言模型如何收敛于一个答案

    The Shape of Wisdom: How Language Models Converge on an Answer

    [https://arxiv.org/abs/2606.01202](https://arxiv.org/abs/2606.01202)

    该研究发现，语言模型在回答事实性问题时，得到相同答案的问题会在内部状态空间中收敛到共享的“答案质心”，该质心在网络约47%-71%深度处突然变得可读取，且只取决于答案内容本身而非问题的表述方式。

    

    当语言模型回答一个事实性问题时，它的内部状态会走向何方？我们向四个开源模型（Llama-3.1-8B、Qwen2.5-7B、Gemma-2-9B 和 Llama-3.2-1B）提出了 2,256 个答案会重复出现的事实性问题，例如某人所拥有的国籍是哪个国家。在每一层，我们都记录了产生答案时所处的状态。得到相同答案的问题会收敛到状态空间中的一个共享区域，我们用该区域的中心——“答案质心”——来概括它。答案在单一层内即可被读取。在每个模型中，其状态最接近自身答案质心的问题比例会在某一层发生跳跃，该层位于网络进程的 47% 到 71% 之间。质心追随的是模型给出的答案而非问题本身：从关于人物国籍的问题中学到的某个国家的质心，也能在关于城镇和电影的问题中识别出该国家。模型本身也在使用这些质心。在跳跃之后，……（摘要原文在此处截断）

    arXiv:2606.01202v2 Announce Type: replace  Abstract: When a language model answers a factual question, where does its internal state go? We asked four open models (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B and Llama-3.2-1B) 2,256 factual questions whose answers recur, such as the country a person is a citizen of. At every layer, we recorded the state from which the answer is produced. Questions that receive the same answer converge on a shared region of state space, which we summarise by its centre, the answer centroid. The answer becomes readable within a single layer. In every model, the share of questions whose state lies nearest its own answer's centroid jumps at one layer, between 47% and 71% of the way through the network. The centroids follow the answer the model gives rather than the question: the centroid of a country learned from questions about people's citizenship also identifies that country in questions about towns and films. The model also uses the centroids. After the jump, 
    
[^326]: 《测量、定位与消融大语言模型中的对齐签名》

    Measuring, Localizing, and Ablating Alignment Signatures in LLMs

    [https://arxiv.org/abs/2605.30526](https://arxiv.org/abs/2605.30526)

    该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。

    

    经过对齐的大语言模型通常表现出一种可被识别的类AI风格，但这种风格与后训练及模型内部表征之间的联系仍知之甚少。在本工作中，我们研究了后训练是否引入或放大了类AI的风格规律性，以及这些规律性是否对应于一个可局部化的内部签名。为此，我们在匹配的人类来源前缀条件下，比较了人类文本、基座模型生成文本与对齐模型生成文本。结果表明，与基座模型的生成文本相比，对齐模型的生成文本具有更低的人类语料亲和度和更高的AI检测率，这说明后训练使生成文本偏离了人类语料风格，转向了检测器易于识别的类AI文本。随后，我们提出了PASTA（Post-training Alignment Signature Targeted Ablation，后训练对齐签名定向消融），这是一种无需训练的方法：它通过对比对齐模型与基座模型的残差来估计后训练对齐签名，并在解码过程中消融相应的方向。在11个……（摘要被截断）

    arXiv:2605.30526v2 Announce Type: replace-cross  Abstract: Aligned language models often exhibit a recognizable AI-like style, yet its connection to post-training and internal representations remains poorly understood. In this work, we study whether post-training introduces or amplifies AI-like stylistic regularities and whether these regularities have a localized internal signature. To this end, we compare human text, base-model generations, and aligned-model generations under matched human-source prefixes. Aligned generations show lower human-corpus affinity and higher AI-detection rates than base generations, suggesting that post-training shifts generated text away from human-corpus style and toward detector-visible AI-like text. We then introduce PASTA (Post-training Alignment Signature Targeted Ablation), a training-free method that estimates a post-training alignment signature from aligned-base residual contrasts and ablates the corresponding direction during decoding. Across 11 
    
[^327]: 超越单一方向：思维链破坏了对拒答行为的简单调控

    Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal

    [https://arxiv.org/abs/2605.26772](https://arxiv.org/abs/2605.26772)

    大型推理模型中的拒答并非仅由单一方向子空间介导，思维链会主动携带并强化拒答信号，使得简单的激活引导难以有效控制拒答行为。

    

    大型推理模型在生成最终输出之前会产生思维链轨迹，这引入了一种动态的内部状态，可能使拒答等控制机制变得复杂。与指令微调的大语言模型中拒答由单一方向子空间介导不同，大型推理模型中的拒答还额外依赖于思维链。在DeepSeek-R1-Distill-LLaMA-8B模型中，当思维链保持固定时，激活引导仅在39%的情况下能够逆转拒答，但完全移除思维链后这一比例提升至70%，表明思维链会主动强化拒答。在一个两阶段干预实验中，模型在激活引导下重新生成其思维链，拒答在94%的情况下被逆转，而即使移除激活引导，所产生的思维链本身仍保留了48%的这种效果。这表明思维链能够独立地携带并重构顺从信号。这些发现表明，大型推理模型中的拒答是由多个因素共同决定的（摘要在此处被截断）。

    arXiv:2605.26772v2 Announce Type: replace  Abstract: Large reasoning models (LRMs) generate chain-of-thought (CoT) traces before producing final outputs, introducing a dynamic internal state that may complicate control mechanisms such as refusal. Unlike instruction-tuned LLMs, where refusal is mediated by a single directional subspace, refusal in large reasoning models (LRMs) additionally depends on the CoT. In DeepSeek-R1-Distill-LLaMA-8B, activation steering reverses refusal in only 39% of cases when the CoT is kept fixed, but removing the CoT entirely increases this to 70%, indicating that the CoT actively reinforces refusal. In a two-stage intervention where the model regenerates its CoT under activation steering, refusal is reversed in 94% of cases, while the resulting CoT alone retains 48% of this effect even after steering is removed. This suggests that the CoT can carry and reconstruct the compliance signal independently. These findings indicate that refusal in LRMs is jointly 
    
[^328]: QML-PipeGuard：在不可信云端检测量子机器学习流水线中的信道替换攻击

    QML-PipeGuard: Detecting Channel-Substitution Attacks on Quantum Machine Learning Pipelines in Untrusted Clouds

    [https://arxiv.org/abs/2605.25066](https://arxiv.org/abs/2605.25066)

    该论文提出QML-PipeGuard，一种在运行时利用客户已有的测量统计来检测不可信云端量子机器学习服务中信道替换攻击的方案，可防范审计规避、服务降级和可复现性审查中的结果伪造。

    

    随着机器学习工作负载迁移到量子硬件上，它们继承了一个经典AI部署已基本解决、而量子部署尚未解决的信任问题：量子机器学习（QML）已经作为云服务出售，但客户无法验证自己付费购买的电路是否就是实际运行的电路。我们证明这一信任缺口是可以被利用的。恶意或为削减成本的提供者可以用另一个量子信道替换所声明的量子信道，该替代信道能够复现客户执行的每一项精度测试，并通过实践中常用的单可观测量检查，同时在信道层面与所声明的计算存在差异。我们将其称为信道替换攻击，并在三种部署场景中对其进行了实例化：审计规避、服务降级，以及可复现性审查中的结果伪造。QML-PipeGuard在运行时利用客户本已收集的测量统计信息检测该攻击。其保证是双向的，以距离……（摘要在此处截断）

    arXiv:2605.25066v2 Announce Type: replace-cross  Abstract: As machine-learning workloads move onto quantum hardware they inherit a trust problem classical AI deployments have largely solved and quantum ones have not: quantum machine learning (QML) is already sold as a cloud service, and a customer cannot check that the circuit they paid for is actually the circuit that ran. We show the gap is exploitable. A malicious or cost-cutting provider can substitute the declared quantum channel with one that reproduces every accuracy test the customer performs, passes the single-observable checks used in practice, and still differs from the declared computation at the channel level. We call this a channel-substitution attack and instantiate it in three deployments: audit evasion, service downgrade, and result forgery in reproducibility review. QML-PipeGuard detects the attack at runtime from measurement statistics the customer already collects. Its guarantee is two-sided, stated in the distance 
    
[^329]: 跨域注意力：超越词元级键值记忆

    Interdomain Attention: Beyond Token-Level Key-Value Memory

    [https://arxiv.org/abs/2605.24330](https://arxiv.org/abs/2605.24330)

    提出跨域注意力（IA）这一新型混合架构，在同一层内将键值信息压缩到SSM参数化的循环记忆中并隐式重建注意力矩阵，从而在保持SSM计算效率的同时近似点积注意力。

    

    基于Softmax注意力的Transformer和基于循环记忆的模型（如状态空间模型（SSM））构成了现代序列模型的基础。混合架构旨在结合注意力的表达能力与循环记忆的计算效率，然而到目前为止，常见的做法只是简单地将注意力层和循环层堆叠在一起。我们提出了跨域注意力（Interdomain Attention, IA），这是一种新颖的混合注意力架构，它在同一层中融合注意力和循环记忆。受softmax注意力核回归视角的启发，IA将值以及键的核特征压缩到由SSM参数化的循环记忆中。然后，IA通过从记忆状态中隐式重建注意力矩阵和值来近似点积注意力；重要的是，这种构建方式保持了SSM的计算效率优势。我们通过设计和预训练IA网络架构来展示IA的有效性……

    arXiv:2605.24330v2 Announce Type: replace  Abstract: Softmax attention-based transformers and recurrent memory-based models (e.g., state space models (SSMs)) form the foundation of modern sequence models. Hybrid architectures aim to combine the expressivity from attention and the computational efficiency from recurrent memory, yet so far common practices simply stack both attention and recurrent layers. We propose Interdomain Attention (IA) as a novel hybrid attention architecture that mixes attention and recurrent memory in the same layer. Motivated by a kernel regression view of softmax attention, IA compresses values and kernel features of keys into a recurrent memory parameterized by an SSM. Then IA approximates dot-product attention by implicitly reconstructing the attention matrix and values from memory states; importantly this construction maintains SSM's computational efficiency advantage. We demonstrate IA's efficacy by designing and pre-training IA network architectures, wher
    
[^330]: 在线Softmax分类中三分之一标度律的边界层机制

    A Boundary-Layer Mechanism for One-Third Scaling in Online Softmax Classification

    [https://arxiv.org/abs/2605.22341](https://arxiv.org/abs/2605.22341)

    论文揭示了平滑代理损失与离散硬标签的失配使在线SGD训练的softmax学生模型在训练晚期仅由教师决策边界附近不断收缩的样本层驱动学习，导致测试损失和泛化误差均呈α^{-1/3}幂律标度；其中泛化误差的1/3指数是SGD在线噪声效应，可通过学习率退火将其改善至1/2。

    

    硬标签分类通常使用平滑代理损失（如softmax交叉熵）进行训练。我们分离出一种机制，用以解释平滑代理与离散标签之间的失配如何为单层softmax学生模型产生幂律学习曲线——该模型通过在线SGD在带argmax教师标签的高斯输入上进行训练。在热力学极限下，动力学可简化为不断增长的学生-教师对齐，以及由SGD噪声维持的残余学生方差。在训练晚期，只有教师决策边界附近不断收缩的样本层仍然活跃。这使得测试损失以及泛化误差（一减测试准确率）均呈现关于训练时间α的α^{-1/3}幂律。其中误差定律是在线噪声效应，而损失定律在无梯度噪声时依然存在。学习率退火可将两个指数分离，把误差指数从1/3改善至接近1/2，同时减缓损失的下降。

    arXiv:2605.22341v2 Announce Type: replace  Abstract: Hard-label classification is usually trained with smooth surrogate losses such as softmax cross-entropy. We isolate a mechanism by which this mismatch between smooth surrogate and discrete labels produces power-law learning curves for a one-layer softmax student trained by online SGD on Gaussian inputs with argmax teacher labels. In the thermodynamic limit, the dynamics reduce to a growing student-teacher alignment and a residual student variance sustained by SGD noise. At late times, only shrinking layers of examples around teacher decision boundaries remain active. This yields an $\alpha^{-1/3}$ law in training time $\alpha$ not only for the test loss but also for the generalization error, one minus test accuracy. The error law is an online-noise effect. The loss law persists without gradient noise. Learning-rate annealing separates the two exponents, improving the error exponent from 1/3 toward 1/2 while slowing the loss. Simulati
    
[^331]: 动态模型合并的瘦身之道

    Dynamic Model Merging Made Slim

    [https://arxiv.org/abs/2605.18904](https://arxiv.org/abs/2605.18904)

    DiDi-Merging通过可微秩选择在共享参数与专家参数之间联合分配低秩容量，并结合无数据的参数空间精炼，在仅1.24倍骨干网络存储的预算下实现了领先的动态模型合并性能。

    

    模型合并能够在无需联合训练或访问任务数据的情况下，将独立微调的检查点整合在一起。动态方法保留了任务特定的容量，但必须在共享参数与专家参数之间分配有限的存储预算。我们提出DiDi-Merging，该方法通过可微秩选择为这些组件联合分配低秩容量，随后进行无数据的参数空间精炼。在八任务ViT-B/32基准测试中，小型配置以单个骨干网络1.24倍的存储量（含该骨干网络）达到89.3%的平均准确率（98.7%的归一化保留率）；中型配置以1.4倍存储量达到89.8%。移除精炼步骤会使中型配置下降4.3个百分点，而静态合并加均匀恢复的对照方法比小型配置低4.7-8.9个百分点。额外的评估还涵盖了在各自报告预算下的语言和多模态模型。（原摘要在此处截断）

    arXiv:2605.18904v2 Announce Type: replace-cross  Abstract: Model merging consolidates independently fine-tuned checkpoints without joint training or access to task data. Dynamic methods retain task-specific capacity, but must divide a limited storage budget between shared and expert parameters. We propose DiDi-Merging, which jointly allocates low-rank capacity to these components through differentiable rank selection, followed by data-free parameter-space refinement. On the eight-task ViT-B/32 benchmark, the small configuration reaches 89.3% average accuracy (98.7% normalized retention) at 1.24x the storage of one backbone, including that backbone; the medium configuration reaches 89.8% at 1.4x. Removing refinement reduces the medium configuration by 4.3 percentage points, while static-merge plus uniform-recovery controls are 4.7-8.9 points below the small configuration. Additional evaluations cover language and multimodal models at their separately reported budgets. The main results a
    
[^332]: PACE：面向单细胞轨迹推断的几何感知桥接传输方法

    PACE: Geometry-Aware Bridge Transport for Single-Cell Trajectory Inference

    [https://arxiv.org/abs/2605.18587](https://arxiv.org/abs/2605.18587)

    PACE通过构建状态和时间依赖的各向异性黎曼度量等三个耦合组件，从破坏性单细胞时间序列快照中恢复几何一致的连续传输动力学，解决了发育异步导致的轨迹错位这一核心难题。

    

    从破坏性时间序列快照中进行单细胞轨迹推断本质上是一个病态问题：既无法观测到跨时间点的细胞对应关系，也无法观测到连续轨迹，因此仅凭快照分布无法唯一确定潜在的动力学过程。现有的最优传输和基于流的方法通常在观测时钟时间下依据欧氏邻近度来耦合细胞，当发育过程不同步、同一实验时间点采样的细胞处于不同潜在伪时间阶段时，这种方法可能导致轨迹错位。我们提出了PACE，一个从破坏性时间序列快照中恢复几何一致的连续传输动力学的轨迹推断框架，它由三个相互耦合的组件构成。首先，PACE构建了一个依赖于状态和时间的各向异性黎曼度量，该度量在局部支持的切线方向上赋予较低的传输成本，同时惩罚法向速度分量。其次，……

    arXiv:2605.18587v4 Announce Type: replace-cross  Abstract: Single-cell trajectory inference from destructive time-course snapshots is fundamentally ill-posed: neither cross-time cell correspondences nor continuous trajectories are observed, so the snapshot distributions alone do not uniquely determine the underlying dynamics. Existing optimal transport and flow-based methods typically couple cells by Euclidean proximity at observed clock times, which can misalign trajectories when development is asynchronous and cells sampled at the same experimental time occupy different latent pseudotime stages. We propose PACE, a trajectory inference framework that recovers geometry-consistent continuous transport dynamics from destructive time-course snapshots through three coupled components. First, PACE constructs a state- and time-dependent anisotropic Riemannian metric that assigns low transport cost along locally supported tangent directions while penalizing normal velocity components. Second,
    
[^333]: 缺陷还是特性²：权重漂移与激活稀疏性

    Bug or Feature$^2$: Weight Drift and Activation Sparsity

    [https://arxiv.org/abs/2605.17659](https://arxiv.org/abs/2605.17659)

    本文揭示了标准损失函数与正向偏置激活函数的相互作用会引发优化过程中固有的负向权重漂移，进而在训练早期自发产生高达约90%的激活稀疏性，并发现当激活稀疏度超过约70%时模型性能会急剧退化。

    

    现代神经架构的设计是通过渐进式的经验性选择逐步收敛而来的，然而其训练动态背后的机制至今仍只被部分理解。我们识别并分析了一种由标准损失函数与正向偏置激活函数之间相互作用所引发的负向权重漂移。我们证明，在MSE或交叉熵损失下，初始化时对正预激活值的梯度期望为非负，这会在训练早期推动下游权重趋向负值。这种漂移是优化过程本身固有的，而非源于数据，并且在各种架构和非对称激活函数（ReLU、GELU、SiLU）中都持续存在。与ReLU结合时，权重漂移在GPT-nano中产生了接近90%的激活稀疏性。我们进一步分析了这种自然产生的稀疏性是否有害，发现平均而言，当激活稀疏度超过约70%时，模型性能会急剧退化。

    arXiv:2605.17659v3 Announce Type: replace  Abstract: The design of modern neural architectures has converged through incremental empirical choices, yet the mechanisms governing their training dynamics remain only partially understood. We identify and analyze a negative weight drift induced by the interaction between standard losses and positively biased activation functions. We prove that under MSE or cross-entropy loss, the gradient with respect to positive pre-activations is non-negative in expectation at initialization, driving downstream weights toward negative values during early training. The drift is intrinsic to optimization rather than data, and persists across architectures and asymmetric activation functions (ReLU, GELU, SiLU). Coupled with ReLU, weight drift produces activation sparsity reaching nearly 90% in GPT-nano. We further analyze whether this natural sparsity is damaging and find that, on average, models sharply degrade above $\sim$70% activation sparsity. We then e
    
[^334]: 张量手册：通过图示掌握张量

    Tensor Cookbook: Mastering Tensors through Diagrams

    [https://arxiv.org/abs/2605.16610](https://arxiv.org/abs/2605.16610)

    本文通过张量网络的图形化（图示）语言来帮助掌握高维张量，将张量收缩编码为图中的边，从而降低记号复杂度并揭示指标记号所掩盖的结构性质。

    

    高维数据在科学与工程的许多领域中自然出现，包括机器学习、信号处理、计算物理学和统计学。这类数据通常被表示为张量，即矩阵的多维推广。虽然张量为多模态结构提供了自然的表示方式，但随着阶数的增长，直接操作张量很快变得具有挑战性：参数数量呈指数级增长，而涉及众多指标（索引）的代数表达式也变得难以解释和实现。张量网络（Tensor Networks, TNs）为应对这些挑战提供了一个有效的框架。张量网络的图形化语言最初由 Penrose 提出，并在量子物理学中得到广泛发展，它将张量收缩编码为图中的边，从而减少了记号负担，并揭示出被指标记号所掩盖的结构性质。尽管高维张量在其中发挥着核心作用……（原文摘要似在此处截断）

    arXiv:2605.16610v2 Announce Type: replace  Abstract: High-dimensional data arise naturally in many areas of science and engineering, including machine learning, signal processing, computational physics, and statistics. Such data are often represented as tensors, multi-dimensional generalizations of matrices. While tensors provide a natural representation for multi-modal structure, their direct manipulation quickly becomes challenging as the order grows: the number of parameters increases exponentially, and algebraic expressions involving many indices become difficult to interpret and implement. Tensor networks (TNs) provide an effective framework for addressing these challenges. Originally introduced by Penrose and developed extensively in quantum physics, the graphical language of tensor networks encodes contractions as edges in a graph, reducing notational overhead and revealing structural properties obscured by index notation. Despite the central role of high-dimensional tensors in 
    
[^335]: StippleDiffusion：基于受控扩散的容量约束点画生成

    StippleDiffusion: Capacity-Constrained Stippling using Controlled Diffusion

    [https://arxiv.org/abs/2605.15816](https://arxiv.org/abs/2605.15816)

    提出了首个基于扩散模型的点画生成采样器StippleDiffusion，通过在最优传输网格点集扩散基线上添加ControlNet分支，在推理时同时满足学习到的局部点分布先验和由图像定义的容量约束，突破了传统迭代优化方法慢且不可微、以及以往学习方法仅支持无条件点生成的局限。

    

    点画图案是指其局部密度跟随目标图像的点集，传统上由针对每种密度的迭代优化器生成，这类方法速度慢、不可微分，且每次面对新目标都必须从头重新运行。已有的学习方法至今只能解决无条件点生成问题；容量约束的、以图像为条件的点画生成一直未能实现。我们提出了首个基于扩散模型的采样器，能够在推理阶段同时满足学习到的局部点分布先验以及连续的、由图像定义的容量约束。该方法是在最优传输网格点集扩散基线之上构建的ControlNet分支，以目标密度图和高分辨率图像为条件。两个设计选择使这一组合切实可行：训练与推理被限制在去噪后期阶段，并以密度加权拒绝采样进行初始化；同时采用标准的零卷积（摘要在此处被截断）……

    arXiv:2605.15816v2 Announce Type: replace-cross  Abstract: Stipple patterns, point sets whose local density tracks a target image, are traditionally produced by per-density iterative optimizers, which are slow, non-differentiable, and must be re-run from scratch for each new target. Learned alternatives have so far addressed only unconditional point generation; capacity-constrained, image-conditioned stippling has remained out of reach. We present the first diffusion-based sampler that simultaneously satisfies a learned local point-distribution prior and a continuous, image-defined capacity constraint at inference. The method is a ControlNet branch built on top of an optimal-transport-grid point-set diffusion baseline, conditioned on the target density map and a high-resolution image. Two design choices make the combination tractable: training and inference are restricted to the late-stage denoising regime, initialized from a density-weighted rejection sample, and the standard zero-con
    
[^336]: MuteBench：面向不完整多模态融合的模态不可用容忍度评估

    MuteBench: Modality Unavailability Tolerance Evaluation for Incomplete Multimodal Fusion

    [https://arxiv.org/abs/2605.15235](https://arxiv.org/abs/2605.15235)

    MuteBench是一个评估多模态融合架构在模态缺失和模态内缺失两种失效模式下鲁棒性的大规模基准，研究发现架构家族是决定鲁棒性的最强预测因子，其重要性超过参数数量。

    

    多模态生理数据为从重症监护病房到可穿戴设备的临床AI系统提供动力，但传感器在实践中经常发生故障。两种失效模式很常见：模态缺失，即整个通道完全缺失；以及模态内缺失，即连续的时间段数据丢失。目前尚无基准在可控严重程度下、跨多样化临床数据集，评估多种融合架构在这两种失效模式下的表现。我们提出了MuteBench，这是一个涵盖来自7个临床领域的9个数据集、6种融合架构和2种缺失数据模式的基准，共包含125,000个样本。通过该基准，我们发现架构家族是鲁棒性的最强预测因子，其影响力超过参数数量。通道独立的模型能够很好地容忍模态缺失，但对模态内缺失可能较为敏感，尤其是在短序列上。课程式模态丢弃策略仅在不超过训练时使用的最大丢弃率的情况下才能提供可靠的保护。

    arXiv:2605.15235v2 Announce Type: replace  Abstract: Multimodal physiological data powers clinical AI systems from intensive care units to wearable devices, but sensors routinely fail in practice. Two failure modes are common: modality missing, where an entire channel is absent, and within-modality missing, where a contiguous time segment is lost. No existing benchmark evaluates multiple fusion architectures under both failure modes at controlled severity levels across diverse clinical datasets. We present MuteBench, a benchmark covering 9 datasets from 7 clinical domains, 6 fusion architectures, and 2 missing-data modes over 125,000 samples. Through this benchmark, we find that architecture family is the strongest predictor of robustness, outweighing parameter count. Channel-independent models tolerate modality missing well but can be sensitive to within-modality missing, especially on short sequences. Curriculum modality dropout protects reliably only up to the maximum dropout rate u
    
[^337]: LightSplit：基于正交投影的实用隐私保护拆分学习（完整版）

    LightSplit: Practical Privacy-Preserving Split Learning via Orthogonal Projections (Full Version)

    [https://arxiv.org/abs/2605.13265](https://arxiv.org/abs/2605.13265)

    LightSplit在切分层采用轻量级的固定正交随机投影作为信息瓶颈，同时降低了拆分学习的通信开销并抵御重建攻击保护隐私。

    

    拆分学习（SL）通过将神经网络划分到客户端和中央服务器之间，实现协作训练。然而，切分层接口带来了一个关键挑战：高维激活值会产生大量的通信开销，同时暴露的表示容易受到重建攻击。现有方法通常只能单独解决效率或隐私问题。我们提出了LightSplit，它在切分层通过轻量级的固定正交随机投影来限制信息暴露并减少通信开销。基于香农信息论，该投影充当一个信息瓶颈，限制实例特定的信息并抑制可被利用的逐样本信号。通过发送低维投影代替原始激活值，服务器在提升后的表示上进行操作，而无需更改现有的拆分学习架构。在不添加任何可训练的客户端组件的情况下，

    arXiv:2605.13265v2 Announce Type: replace  Abstract: Split learning (SL) enables collaborative training by partitioning a neural network across clients and a central server. However, the cut-layer interface introduces a key challenge: high-dimensional activations incur substantial communication overhead while exposing representations vulnerable to reconstruction attacks. Existing approaches typically address efficiency or privacy in isolation. We propose LightSplit, which limits information exposure and reduces communication overhead via a lightweight fixed orthogonal random projection at the cut layer. Based on Shannon's information theory, this projection acts as an information bottleneck that restricts instance-specific information and suppresses exploitable per-sample signals. With low-dimensional projections sent instead of raw activations, the server operates on lifted representations without requiring changes to existing SL architectures. Adding no trainable client components, L
    
[^338]: STRABLE：基于字符串的表格机器学习基准测试

    STRABLE: Benchmarking Tabular Machine Learning with Strings

    [https://arxiv.org/abs/2605.12292](https://arxiv.org/abs/2605.12292)

    该论文提出了STRABLE，一个包含108个真实世界表格的基准测试语料库，并通过对445个流水线的大规模实证研究，首次系统比较了端到端字符串建模架构与先将字符串编码为数字的模块化流水线在表格机器学习中的表现。

    

    表格学习的基准测试已经揭示了专用架构的优势，推动了技术前沿的发展。但现实世界中的表格通常包含字符串条目，而不仅仅是数字，由于缺乏可靠的基准测试套件，这些场景一直未被充分研究。这引出了新的研究问题：是否需要专门的学习器来对字符串和数字进行端到端建模？还是将字符串编码为数字（如分类编码）就足够了？如果是后者，编码后的表格是否类似于数值型表格数据，从而需要使用相同的学习器？为了支持这些研究，我们贡献了STRABLE，一个包含108个表格的基准测试语料库，涵盖了不同应用领域中同时包含字符串和数字的真实世界学习问题。我们开展了首个针对字符串表格学习的大规模实证研究，评估了445个流水线。这些流水线涵盖端到端架构和模块化流水线，后者首先将字符串进行编码……

    arXiv:2605.12292v2 Announce Type: replace  Abstract: Benchmarking tabular learning has revealed the benefit of dedicated architectures, pushing the state of the art. But real-world tables often contain string entries, beyond numbers, and these settings have been understudied due to a lack of a solid benchmarking suite. They lead to new research questions: Are dedicated learners needed, with end-to-end modeling of strings and numbers? Or does it suffice to encode strings as numbers, as with a categorical encoding? And if so, do the resulting tables resemble numerical tabular data, calling for the same learners? To enable these studies, we contribute STRABLE, a benchmarking corpus of 108 tables, all real-world learning problems with strings and numbers across diverse application fields. We run the first large-scale empirical study of tabular learning with strings, evaluating 445 pipelines. These pipelines span end-to-end architectures and modular pipelines, where strings are first encode
    
[^339]: COSMOS：基于聚类服务器模型与仅伪标签通信的模型无关个性化联邦学习

    COSMOS: Model-Agnostic Personalized Federated Learning with Clustered Server Models and Pseudo-Label-Only Communication

    [https://arxiv.org/abs/2605.11165](https://arxiv.org/abs/2605.11165)

    COSMOS是一个模型无关的个性化联邦学习框架：客户端只需向服务器传输伪标签预测，服务器据此按预测相似度聚类并用自身算力训练聚类专属模型再蒸馏回客户端，同时首次从理论上证明了这种蒸馏可实现指数级的个性化风险收缩。

    

    异构环境下的联邦学习（FL）仍然具有挑战性，因为客户端模型在架构和数据分布上往往各不相同。尽管近期方法试图通过客户端聚类和知识蒸馏来应对这一挑战，但同时处理架构异质性与统计异质性仍然十分困难。我们提出了COSMOS，一个仅通过伪标签通信即可实现服务器端个性化的模型无关框架。客户端训练本地模型并在公共数据上进行预测；服务器根据预测相似度对客户端进行聚类，利用自身算力为每个组训练一个聚类专属模型，并将所得模型蒸馏回客户端。我们首次提供了理论分析，证明从学习到的聚类模型进行蒸馏能够带来指数级的个性化风险收缩，这超越了通常所提供的收敛至平稳点的保证……

    arXiv:2605.11165v4 Announce Type: replace  Abstract: Federated learning (FL) in heterogeneous environments remains challenging because client models often differ in both architecture and data distribution. While recent approaches attempt to address this challenge through client clustering and knowledge distillation, simultaneously handling architectural and statistical heterogeneity remains difficult. We introduce COSMOS, a model-agnostic framework that enables server-side personalization using only pseudo-label communication. Clients train local models and predict on the public data; the server clusters clients by prediction similarity, trains a cluster-specific model for each group using its own compute, and distills the resulting models back to clients. We provide the first theoretical analysis showing that distillation from the learned cluster models can yield exponential personalization risk contraction, going beyond the convergence-to-stationarity guarantees typically provided in
    
[^340]: 面向多模态持续指令微调的动态跨模态提示生成

    Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2605.10765](https://arxiv.org/abs/2605.10765)

    提出DRAPE框架，通过动态生成实例特定的跨模态软提示，实现针对每个查询-图像对的个性化适应，从而在多模态持续指令微调中缓解灾难性遗忘。

    

    多模态大语言模型通过指令微调获得了强大的性能，然而实际部署往往需要在序列任务中持续扩展能力。在这种场景下，多模态持续指令微调旨在获取新能力的同时限制灾难性遗忘。现有方法主要遵循模块组合范式：它们维护任务级提示或LoRA专家，并在推理时动态路由或聚合其中的一个子集。然而，同一任务内的样本在视觉场景、问题意图和推理需求上仍可能存在显著差异。这促使我们对每个查询-图像对进行实例级适应，而不仅仅是选择或组合任务级模块。为此，我们提出了DRAPE（动态跨模态提示生成），这是一个为多模态持续指令微调合成连续的实例特定软提示的提示学习框架。

    arXiv:2605.10765v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) achieve strong performance through instruction tuning, yet real-world deployment often requires continual capability expansion across sequential tasks. In such scenarios, Multimodal Continual Instruction Tuning (MCIT) aims to acquire new capabilities while limiting catastrophic forgetting. Existing methods mainly follow a module-composition paradigm: they maintain task-level prompts or LoRA experts and dynamically route or aggregate a subset of them at inference. However, samples within the same task can still differ substantially in visual scenes, question intents, and reasoning demands. This motivates instance-level adaptation to individual query-image pairs rather than only selecting or combining task-level modules. To this end, we propose DRAPE (Dynamic Cross-Modal Prompt Generation), a prompt-learning framework that synthesizes continuous instance-specific soft prompts for MCIT. Ins
    
[^341]: 跟随均值：参考引导的流匹配

    Follow the Mean: Reference-Guided Flow Matching

    [https://arxiv.org/abs/2605.10302](https://arxiv.org/abs/2605.10302)

    该论文发现流匹配的速度场仅由条件终点均值决定，因此只需通过参考样本改变该均值，即可在不微调的情况下引导预训练模型实现颜色、身份、风格和结构的可控生成。

    

    现有的可控生成方法通常依赖于微调、辅助网络或测试时搜索。我们证明流匹配提供了一种不同的控制接口：通过示例进行适应。对于确定性插值，速度场仅由条件终点均值决定，因此改变该均值即改变流本身，预训练模型只需改变其跟随的参考集即可被引导。我们以两种形式实例化了这一思想。参考均值引导（Reference-Mean Guidance）无需训练，它从参考库中计算终点均值的闭式修正，并将其应用于冻结的 FLUX.2-klein (4B) 模型，在提示词、种子和权重保持固定的情况下实现颜色、身份、风格和结构的控制。半参数引导（Semi-Parametric Guidance）通过显式的均值锚点和可学习的残差精炼器，将同一原理构建到训练好的模型中，使无条件 DiT-B/4 在……

    arXiv:2605.10302v4 Announce Type: replace  Abstract: Existing approaches to controllable generation typically rely on fine-tuning, auxiliary networks, or test-time search. We show that flow matching admits a different control interface: adaptation through examples. For deterministic interpolants, the velocity field is solely governed by a conditional endpoint mean, so shifting this mean shifts the flow itself, and a pretrained model can be steered simply by changing the reference set it follows. We instantiate this idea in two forms. Reference-Mean Guidance is training-free. It computes a closed-form correction to the endpoint mean from a reference bank and applies it to a frozen FLUX.2-klein (4B) model, controlling color, identity, style, and structure while the prompt, seed, and weights stay fixed. Semi-Parametric Guidance builds the same principle into a trained model through an explicit mean anchor and a learned residual refiner, matching the quality of an unconditional DiT-B/4 on 
    
[^342]: 每参数Token覆盖度对稳健的LLM缩放定律外推至关重要

    Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation

    [https://arxiv.org/abs/2605.08541](https://arxiv.org/abs/2605.08541)

    本文证明在固定每参数Token比例下拟合LLM缩放定律会因N与D指数近似相等导致的共线性而使最小二乘问题病态化、尺度系数不可辨识，并推导出实现稳健外推所需的TPP多样性闭式阈值。

    

    神经缩放定律将语言模型的损失近似为参数量 $N$ 和Token数 $D$ 的幂律函数。遵循Chinchilla式的计算最优训练范式，许多研究在固定的每参数Token比例（TPP）$k$ 下进行实验来拟合缩放定律，并令 $D = kN$。我们证明，这种共线设计，结合经验上常见的 $N$ 与 $D$ 指数近似相等的现象，会在高斯-牛顿最小二乘问题中引入固有的病态性：设计矩阵的条件数随 $N$ 与 $D$ 指数之间差距的平方倒数增长。尺度系数实际上变得不可辨识，其置信区间膨胀一个数量级甚至更多，产生一个“松散”的模型，其外推能力在偏离训练射线时急剧退化。我们针对四种缩放定律形式证明了这一点，并推导出一个封闭形式的TPP多样性阈值，该阈值是必要且充分的……（摘要原文在此处被截断）

    arXiv:2605.08541v3 Announce Type: replace  Abstract: Neural scaling laws approximate a language model's loss as a power-law function of parameter count $N$ and token count $D$. Following Chinchilla-style compute-optimal training, many studies fit scaling laws from runs performed under a fixed tokens-per-parameter (TPP) ratio $k$ and set $D = kN$. We show that this collinear design, combined with the empirically common near-equality of the exponents governing $N$ and $D$, induces an inherent ill-conditioning in the Gauss-Newton least-squares problem: the condition number of the design grows as the inverse square of the gap between the $N$ and $D$-exponents. The scale coefficients become practically unidentifiable, with confidence intervals inflating by an order of magnitude or more, yielding a ``sloppy'' model whose extrapolations degrade sharply off the training ray. We prove this for four scaling-law formalisms and derive a closed-form TPP-diversity threshold that is necessary and suf
    
[^343]: RAM-H1200：一个用于类风湿关节炎手部X光片的统一评估数据集

    RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for Rheumatoid Arthritis

    [https://arxiv.org/abs/2605.05616](https://arxiv.org/abs/2605.05616)

    RAM-H1200是一个包含来自六个医疗中心的1,200张手部X光片的多层次标注数据集，为类风湿关节炎评估提供了从解剖结构分割、骨侵蚀定量分析到临床SvdH评分的统一基准。

    

    类风湿关节炎（RA）的手部X光片评估需要对解剖结构和细粒度局部病理变化进行多层次的分析与建模。然而，现有的公开资源无法支持这种统一的多层次分析，往往缺乏全手覆盖、细粒度标注，以及与临床评分系统的一致性整合。特别是，能够支持骨侵蚀（BE）定量分析的标注仍然十分稀缺。RAM-H1200包含从六个医疗中心收集的1,200张手部X光片，提供多层次标注，包括：（i）全手骨结构实例分割，（ii）像素级骨侵蚀掩码，（iii）SvdH定义的关节感兴趣区域，以及（iv）骨侵蚀和关节间隙狭窄（JSN）的关节级SvdH评分。该数据集旨在评估模型能否联合捕捉解剖结构、局部侵蚀性病变以及临床标准化的RA严重程度。（注：原文摘要在此处截断）

    arXiv:2605.05616v2 Announce Type: replace-cross  Abstract: Rheumatoid arthritis (RA) assessment from hand radiographs requires multi-level analysis and modeling of anatomical structures and fine-grained local pathological changes. However, existing public resources do not support such unified multi-level analysis, often lacking full-hand coverage, fine-grained annotations, and consistent integration with clinical scoring systems. In particular, annotations that enable quantitative analysis of bone erosion (BE) remain scarce. RAM-H1200 contains 1,200 hand radiographs collected from six medical centers, with multi-level annotations including (i) whole-hand bone structure instance segmentation, (ii) pixel-level BE masks, (iii) SvdH-defined joint regions of interest, and (iv) joint-level SvdH scores for both BE and joint space narrowing (JSN). It is designed to evaluate whether models can jointly capture anatomical structure, localized erosive pathology, and clinically standardized RA seve
    
[^344]: 非平稳性破坏多智能体强化学习中的置换代理检验：诊断与对策

    Non-Stationarity Breaks Permutation Surrogates in Multi-Agent Reinforcement Learning: Diagnosis and Remedies

    [https://arxiv.org/abs/2604.23716](https://arxiv.org/abs/2604.23716)

    该研究发现非平稳性会使基于置换代理的信息论影响检验在多智能体强化学习中产生接近100%的假阳性，并提出通过改变零模型而非仅排除训练瞬态阶段来修复这一缺陷。

    

    arXiv:2604.23716v4 公告类型：替换。摘要：信息论度量的报告指南很少对照真实基准进行检验。我们在两个多智能体强化学习游戏（一个社会困境博弈和一个协调竞赛）中检验了一项防护措施，其中所选智能体对之间的有向影响在构造上为零，实验跨越100个随机种子。若省略一个前提条件——排除非平稳的训练瞬态阶段——将产生100.00%和99.95%的假阳性率：那些在从未相遇过的运行中独立退火探索的智能体会被判定为相互影响。排除瞬态阶段后，社会困境中的假阳性率降至3.0%，但协调游戏中仍高达11.8%，平稳性检验解释了这一差异：社会困境中95.7%的序列在之后是平稳的，而协调游戏序列中仅有56.8%是平稳的。因此，非平稳性必须得到处理，而简单的排除方法既不是唯一的途径，也并不充分。我们推荐的做法是改变零模型而非修改数据：对源序列进行置换……

    arXiv:2604.23716v4 Announce Type: replace  Abstract: Reporting guidance for information-theoretic measures is rarely tested against ground truth. We test one guardrail in two multi-agent reinforcement learning games, a social dilemma and a coordination race, where directed influence between selected agent pairs is zero by construction, over 100 seeds. Omitting one precondition, exclusion of the non-stationary training transient, gives false-positive rates of 100.00% and 99.95%: agents annealing exploration independently, in runs that never met, are flagged as influencing one another. Excluding the transient reaches 3.0% in the social dilemma but 11.8% in the coordination game, which stationarity tests explain: 95.7% of social-dilemma series are stationary afterwards against 56.8% of coordination series. So the non-stationarity must be treated, and exclusion is neither the only way nor sufficient. What we recommend instead changes the null model rather than the data: permuting the sourc
    
[^345]: 用于推理大语言模型高效无监督置信度校准的二元一致性方法

    Binary Agreement for Efficient Unsupervised Confidence Calibration of Reasoning LLMs

    [https://arxiv.org/abs/2604.19444](https://arxiv.org/abs/2604.19444)

    BASC将锚定响应与单个独立探测响应之间的一致性视为伯努利观测来估计潜在自洽性，以每个校准查询仅需两个响应、部署时仅需一个响应的极低成本，实现了推理大语言模型的无监督置信度校准。

    

    推理语言模型通常缺乏可靠部署所需的校准置信度，尤其是在正确性标签不可用且采样预算有限的情况下。我们提出了用于自洽性校准的二元一致性方法（BASC），该方法将自洽性用作正确性的无标签代理，而无需为每个校准查询构建代价高昂的自洽性目标。我们的关键观察是，锚定响应与单个独立探测响应之间的一致性是一个伯努利观测，其条件均值即为锚定响应的潜在自洽性。通过在无标签查询之间汇集这些观测，BASC既能对现有的置信度分数进行重新校准，也能在重新校准之前学习更具判别性的分数，每个校准查询仅需两个响应，部署时仅需一个响应。二元等渗回归重新校准的使用提供了无分布假设的校准保证……

    arXiv:2604.19444v2 Announce Type: replace  Abstract: Reasoning language models often lack the calibrated confidence needed for reliable deployment, particularly when correctness labels are unavailable and sampling budgets are limited. We introduce Binary Agreement for Self-consistency Calibration (BASC), which uses self-consistency as an unlabeled proxy for correctness without constructing costly self-consistency targets for each calibration query. Our key observation is that agreement between an anchor response and a single independent probe is a Bernoulli observation whose conditional mean is the anchor's latent self-consistency. Pooling these observations across unlabeled queries, BASC supports both recalibrating existing confidence scores and learning more discriminative scores before recalibration, using as few as two responses per calibration query and only one at deployment. The use of binary isotonic recalibration provides a distribution-free calibration guarantee with respect 
    
[^346]: 更简单的最优决策树与自适应集合覆盖的对数近似算法

    Simpler Logarithmic Approximation Algorithms for the Optimal Decision Tree and Adaptive Set Cover

    [https://arxiv.org/abs/2604.12036](https://arxiv.org/abs/2604.12036)

    该论文提出了一种带有简化分析的新贪心算法，为最优决策树问题同时实现了最坏情形代价低于 $3.65\log_2 n$ 和平均情形代价低于 $7.3\log_2 n$ 的对数近似保证。

    

    我们研究在平均代价和最坏情形代价两种准则下，构造能够从给定基集中识别未知假设的决策树问题。最优决策树问题在文献中已被广泛研究，针对每种准则均已有 $\mathcal{O}(\log n)$ 近似保证的结果，其中也包括同时对两种准则的保证（此处 $n$ 为假设的数量）[ICML]。尽管现有的大多数算法采用了简单的贪心策略，但它们的分析依赖于复杂的论证。由于 ODT 是一个具有广泛应用的基础问题，我们希望开发既有效又易于理解的算法。我们提出了一种新的贪心算法并给出了简化分析，该算法同时实现了最坏情形代价下低于 $3.65\log_2 n$、平均情形代价下低于 $7.3\log_2 n$ 的近似因子。

    arXiv:2604.12036v4 Announce Type: replace-cross  Abstract: We study the problem of constructing a decision tree that identifies an unknown hypothesis from a given ground set, under both average- and worst-case cost. The Optimal Decision Tree (ODT) problem has been extensively studied in the literature, and $\mathcal{O}(\log n)$-approximation guarantees are known for each criterion, including a simultaneous guarantee for both (here $n$ is the number of hypotheses) [ICML]. Although most of the existing algorithm use simple greedy strategies, their analyses rely on intricate arguments. Since ODT is a fundamental problem with wide-ranging applications, we are interested in developing algorithms that are both effective and easy to understand.   We present a new greedy algorithm with a simplified analysis that simultaneously achieves approximation factors below $3.65\log_2 n$ for worst-case cost and $7.3\log_2 n$ for average-case cost. Our algorithm for the Optimal Decision Tree behaves gree
    
[^347]: 用于逆问题不确定性量化的变分图神经网络

    Variational Graph Neural Networks for Uncertainty Quantification in Inverse Problems

    [https://arxiv.org/abs/2603.29515](https://arxiv.org/abs/2603.29515)

    本文提出一种将变分层集成到图神经网络中的VGNN架构，通过建模概率分布来量化逆问题中深度学习预测的不确定性，从而为计算力学中的关键应用提供可靠的置信度度量。

    

    深度机器学习技术在计算力学中的日益广泛应用，显著加速了几年前还被认为难以处理的问题的仿真求解。然而，在工程或医学数字孪生等关键应用中，仅有快速响应是不够的，还必须提供可靠的结果。在某些情况下，传统的确定性方法可能并非最优，因为它们无法为其预测或结果提供置信度度量，尤其是在逆问题中——其解可能不唯一，或者初始数据可能由于噪声等因素而并非完全可靠。经典的深度神经网络同样缺乏量化其预测不确定性的明确度量。在这项工作中，我们提出了一种变分图神经网络（VGNN）架构，该架构将变分层集成到其结构中，以建模概率分布……

    arXiv:2603.29515v3 Announce Type: replace  Abstract: The increasingly wide use of deep machine learning techniques in computational mechanics has significantly accelerated simulations of problems that were considered unapproachable just a few years ago. However, in critical applications such as Digital Twins for engineering or medicine, fast responses are not enough; reliable results must also be provided. In certain cases, traditional deterministic methods may not be optimal as they do not provide a measure of confidence in their predictions or results, especially in inverse problems where the solution may not be unique or the initial data may not be entirely reliable due to the presence of noise, for instance. Classic deep neural networks also lack a clear measure to quantify the uncertainty of their predictions. In this work, we present a variational graph neural network (VGNN) architecture that integrates variational layers into its architecture to model the probability distributio
    
[^348]: 面向基于大语言模型语音识别的对话音频上下文抽象压缩

    Abstract Compression of Conversational Audio Context for LLM-based ASR

    [https://arxiv.org/abs/2603.26246](https://arxiv.org/abs/2603.26246)

    提出抽象压缩方法，将对话历史音频压缩为固定数量的潜在标记并保留转录文本，使基于LLM的语音识别以更小开销提升上下文实体的识别效果。

    

    标准的基于大语言模型（LLM）的语音识别系统孤立地处理每个话语，限制了其利用对话上下文的能力。本文研究了来自先前对话轮次的多模态上下文能否改进基于LLM的语音识别（ASR），以及如何高效地表示这些上下文。我们发现，经过监督多轮训练后，对话上下文主要有助于上下文相关实体的识别。然而，以原始上下文为条件的开销很大，因为先前轮次的音频嵌入序列会随对话长度的增长而迅速膨胀。为解决这一问题，我们提出了抽象压缩（Abstract Compression），这是一种对先前轮次音频的学习式潜在抽象，用固定数量的潜在标记替换每个先前轮次的音频序列，同时显式保留相应的转录文本。在域内和域外测试集上，启用压缩的模型以更小的先前轮次音频开销，恢复了原始上下文条件化所带来的一部分收益。

    arXiv:2603.26246v2 Announce Type: replace-cross  Abstract: Standard LLM-based speech recognition systems process utterances in isolation, limiting their ability to leverage conversational context. Here we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio embedding sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, a learned latent abstraction of prior-turn audio that replaces each prior-turn audio sequence with a fixed number of latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compression-enabled model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footp
    
[^349]: 面向城市空气质量的可解释PM2.5预测：自适应时间序列模型的比较研究

    Interpretable PM2.5 Forecasting for Urban Air Quality: A Comparative Study of Adaptive Time-Series Models

    [https://arxiv.org/abs/2603.25495](https://arxiv.org/abs/2603.25495)

    本研究比较了SARIMAX、Prophet和NeuralProphet三种轻量级可解释模型在北京小时级PM2.5预测中的表现，并评估了每周滚动重拟合与冻结预测加在线残差校正两种自适应部署策略的实际效果。

    

    准确的短期空气质量预测对于公共健康保护和城市管理至关重要，然而近来许多预测框架依赖于复杂、数据密集且计算开销巨大的模型。本研究探讨了轻量级且可解释的预测方法能否在中国北京的小时级PM2.5预测中提供有竞争力的性能。利用多年污染物与气象时间序列数据，我们构建了一种防止数据泄露的预测工作流程，该流程在完美预报（Perfect Prognosis）设置下结合了按时间顺序的数据划分、预处理、特征选择和外生驱动因素建模。研究评估了三种预测模型家族：SARIMAX、Facebook Prophet和NeuralProphet。为评估模型在实际部署中的表现，研究在两种自适应机制下对模型进行了测试：每周滚动前向重拟合，以及带在线残差校正的冻结预测。结果显示出明显的差异

    arXiv:2603.25495v2 Announce Type: replace  Abstract: Accurate short-term air-quality forecasting is essential for public health protection and urban management, yet many recent forecasting frameworks rely on complex, data-intensive, and computationally demanding models. This study investigates whether lightweight and interpretable forecasting approaches can provide competitive performance for hourly PM2.5 prediction in Beijing, China. Using multi-year pollutant and meteorological time-series data, we developed a leakage-aware forecasting workflow that combined chronological data partitioning, preprocessing, feature selection, and exogenous-driver modeling under the Perfect Prognosis setting. Three forecasting families were evaluated: SARIMAX, Facebook Prophet, and NeuralProphet. To assess practical deployment behavior, the models were tested under two adaptive regimes: weekly walk-forward refitting and frozen forecasting with online residual correction. Results showed clear differences
    
[^350]: 面向黑盒域自适应语义鸿沟弥合的自适应双教师蒸馏与子网络校正方法

    Adaptive Dual-Teacher Distillation with Subnetwork Rectification for Bridging Semantic Gaps in Black-Box Domain Adaptation

    [https://arxiv.org/abs/2603.22908](https://arxiv.org/abs/2603.22908)

    提出了一种自适应双教师蒸馏框架DDSR，通过自适应预测融合与子网络校正策略，调和黑盒源模型的任务特定知识与视觉-语言模型的语义先验之间的差异，有效弥合了黑盒域自适应中的语义鸿沟。

    

    在源数据和源模型参数均不可访问的假设下，黑盒域自适应（BBDA）代表了一个非常实用但极具挑战性的场景，其中可迁移的知识仅限于黑盒源模型的预测输出。现有方法通过伪标签细化或借助视觉-语言模型（ViLs）来利用此类知识，但它们往往无法调和来自黑盒模型的任务特定知识与ViL的语言对齐语义先验之间固有的差异，导致次优的知识整合和下降的自适应性能。为应对这一挑战，我们提出了自适应双教师蒸馏与子网络校正方法（DDSR），这是一个显式调和这些互补但不一致的知识来源的框架。DDSR采用自适应预测融合策略来整合来自黑盒源模型和ViL的预测，从而生成……

    arXiv:2603.22908v5 Announce Type: replace-cross  Abstract: Assuming that neither source data nor source model parameters are accessible, black-box domain adaptation (BBDA) represents a highly practical yet challenging setting, where transferable knowledge is limited to the predictions of a black-box source model. Existing approaches exploit such knowledge via pseudo-label refinement or by leveraging vision-language models (ViLs), but they often fail to reconcile the inherent discrepancy between task-specific knowledge from black-box models and language-aligned semantic priors of ViLs, resulting in suboptimal integration and degraded adaptation performance. To address this challenge, we propose adaptive Dual-Teacher Distillation with Subnetwork Rectification (DDSR), a framework that explicitly reconciles these complementary yet inconsistent knowledge sources. DDSR employs an adaptive prediction fusion strategy to integrate predictions from the black-box source model and a ViL, generatin
    
[^351]: 检测是廉价的，路由是习得的：为什么基于拒绝的对齐评估会失败

    Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails

    [https://arxiv.org/abs/2603.18280](https://arxiv.org/abs/2603.18280)

    该论文以中国语言模型的政治审查为自然实验，发现对齐机制真正作用于“从概念检测到行为策略”的路由环节而非概念检测本身，从而解释了仅基于拒绝行为的对齐评估为何会失效。

    

    当前的对齐评估主要衡量模型是否编码了危险概念，以及模型是否拒绝有害请求。但这两者都忽略了对齐通常真正发挥作用的层面：从概念检测到行为策略之间的路由。我们以源自中国的语言模型中的政治审查作为自然实验，使用探针、外科手术式消融和行为测试，对来自五个实验室的九个开源权重模型进行了研究。由此得出三个发现。第一，仅凭探针准确率不具有诊断意义：政治探针、零对照和置换基线都可以达到100%的准确率，因此对保留类别（held-out category）的泛化测试才是有信息量的检验。第二，外科手术式消融揭示了实验室特有的路由方式。在大多数被测试的模型中，移除政治敏感方向可以消除审查行为并恢复准确的事实输出；而其中一个模型会出现虚构（confabulation），因为其架构将事实知识与审查机制纠缠在了一起。跨模型……

    arXiv:2603.18280v4 Announce Type: replace-cross  Abstract: Current alignment evaluation mostly measures whether models encode dangerous concepts and whether they refuse harmful requests. Both miss the layer where alignment often operates: routing from concept detection to behavioral policy. We study political censorship in Chinese-origin language models as a natural experiment, using probes, surgical ablations, and behavioral tests across nine open-weight models from five labs. Three findings follow. First, probe accuracy alone is non-diagnostic: political probes, null controls, and permutation baselines can all reach 100%, so held-out category generalization is the informative test. Second, surgical ablation reveals lab-specific routing. Removing the political-sensitivity direction eliminates censorship and restores accurate factual output in most models tested, while one model confabulates because its architecture entangles factual knowledge with the censorship mechanism. Cross-model
    
[^352]: SCOPE：面向长尾联邦学习的可扩展免训练语义核心集选择

    SCOPE: Scalable Training-Free Semantic Coreset Selection for Long-Tailed Federated Learning

    [https://arxiv.org/abs/2603.12976](https://arxiv.org/abs/2603.12976)

    SCOPE提出了一种免训练的联邦核心集选择框架，各客户端仅用冻结的轻量视觉-语言模型对样本评分一次并上传每类标量统计量，服务器据此构建全局类别画像，从而为长尾非独立同分布数据提供可扩展、无需GPU训练的核心集选择方案。

    

    科学发现日益依赖于对高分辨率仪器数据的联邦学习，而这些数据呈现出强烈的非独立同分布和长尾分布特性。核心集选择可以缩短训练时间，但现有的联邦核心集方法要么使用忽略全局类别分布的局部启发式方法进行剪枝，要么先在每个客户端训练一个预热模型并上传其特征的类别质心。这使得核心集选择成为一个分布式瓶颈：在每个客户端消耗GPU时间，随模型规模和客户端数量的增长而扩展，并且必须针对每种架构重复进行。我们提出了SCOPE（基于正交投影嵌入的语义核心集），这是一个面向长尾数据的可扩展、免训练的联邦核心集选择框架。每个客户端使用一个冻结的紧凑视觉-语言模型对其样本进行一次性评分，并上传每类的标量统计信息。服务器将这些信息合并为全局类别画像，从而……

    arXiv:2603.12976v2 Announce Type: replace  Abstract: Scientific discovery increasingly relies on federated learning over data from high-resolution instruments, and these data are strongly non-IID and long-tailed. Coreset selection could shorten training, but existing federated coreset methods either prune with local heuristics that ignore the global class distribution, or first train a warm-up model at every client and upload class centroids of its features. This makes coreset selection a distributed bottleneck that consumes GPU time at every client, grows with the model and the number of clients, and must be repeated for every architecture. We introduce SCOPE, named for Semantic Coreset using Orthogonal Projection Embeddings, a scalable, training-free federated coreset framework for long-tailed data. Each client scores its samples once with a frozen, compact vision-language model and uploads scalar statistics per class. The server merges these into a global class profile, which lets e
    
[^353]: 混合信念强化学习用于高效协同空间探索

    Hybrid Belief Reinforcement Learning for Efficient Coordinated Spatial Exploration

    [https://arxiv.org/abs/2603.03595](https://arxiv.org/abs/2603.03595)

    本文提出混合信念-强化学习（HBRL）框架，第一阶段利用LGCP空间信念与PathMI多步前瞻规划器进行信息驱动探索，第二阶段通过双通道知识迁移热启动SAC智能体，从而实现多智能体协同空间探索的高效学习与规划。

    

    协调多个自主智能体以探索并服务空间异质的需求，需要同时学习未知的空间模式并规划能最大化任务性能的轨迹。纯基于模型的方法能够提供结构化的不确定性估计，但缺乏自适应策略学习能力；而深度强化学习在缺乏空间先验的情况下往往存在样本效率低下的问题。本文提出了一种混合信念-强化学习（HBRL）框架来填补这一空白。在第一阶段，智能体使用对数高斯考克斯过程（LGCP）构建空间信念，并执行由具有多步前瞻能力的路径互信息（PathMI）规划器引导的信息驱动轨迹。在第二阶段，轨迹控制被转移至一个软演员-评论家（SAC）智能体，并通过双通道知识迁移实现热启动：i) 信念状态迁移提供空间不确定性信息，ii) 经验回放迁移（原文摘要在此处被截断）

    arXiv:2603.03595v2 Announce Type: replace  Abstract: Coordinating multiple autonomous agents to explore and serve spatially heterogeneous demand requires jointly learning unknown spatial patterns and planning trajectories that maximize task performance. Pure model-based approaches provide structured uncertainty estimates but lack adaptive policy learning, while deep reinforcement learning often suffers from poor sample efficiency when spatial priors are absent. This paper presents a hybrid belief-reinforcement learning (HBRL) framework to address this gap. In the first phase, agents construct spatial beliefs using a Log-Gaussian Cox Process (LGCP) and execute information-driven trajectories guided by a Pathwise Mutual Information (PathMI) planner with multi-step lookahead. In the second phase, trajectory control is transferred to a Soft Actor-Critic (SAC) agent, warm-started through dual-channel knowledge transfer: i) belief state transfer supplies spatial uncertainty, and ii) replay b
    
[^354]: 面向灵活的基于密度的聚类的可扩展后验不确定性量化

    Scalable Posterior Uncertainty for Flexible Density-Based Clustering

    [https://arxiv.org/abs/2603.03188](https://arxiv.org/abs/2603.03188)

    该论文将基于密度的聚类与鞅后验分布相结合，提出了一种完全并行化、可扩展的贝叶斯不确定性量化框架，兼容归一化流等可微密度估计器，并能对任意聚类相关量进行推断，同时具备严格的频率派收敛性保证。

    

    基于密度的聚类是一种成熟的框架，它通过数据生成密度的拓扑或几何特征来定义聚类，由此产生的灵活方法在大规模和复杂数据集上已被证明是有效的。我们通过将基于密度的聚类与鞅后验分布相结合，解决了获得可扩展的贝叶斯式不确定性量化的问题。密度的不确定性首先通过一种完全并行化的预测重采样方案来量化，该方案由模型得分函数驱动，自然地兼容诸如归一化流等现代可微密度估计器。随后，通过将从鞅后验中采样的每个随机密度进行基于密度的聚类，获得聚类的后验样本，从而能够对任何与聚类相关的量进行推断。将估计目标表示为密度泛函还使得严格的频率派收敛分析成为可能。

    arXiv:2603.03188v3 Announce Type: replace-cross  Abstract: Density-based clustering is a well-established framework that defines clusters through topological or geometric features of the data-generating density, yielding flexible procedures that have proven effective on large and complex datasets. We address the problem of obtaining scalable Bayesian-style uncertainty quantification by combining density-based clustering with martingale posterior distributions. Density uncertainty is first quantified through a fully parallelized predictive resampling scheme driven by model score functions, naturally accommodating modern differentiable density estimators such as normalizing flows. Posterior samples of the clustering are then obtained by applying density-based clustering to each random density sampled from the martingale posterior, enabling inference on any clustering-related quantity. Casting the estimand as a density functional also enables a rigorous frequentist convergence analysis. W
    
[^355]: SEHFS：面向多视图多标签特征选择的结构熵引导高阶相关性学习

    SEHFS: Structural Entropy-Guided High-Order Correlation Learning for Multi-View Multi-Label Feature Selection

    [https://arxiv.org/abs/2603.03022](https://arxiv.org/abs/2603.03022)

    提出SEHFS方法，通过将特征图转换为结构熵最小化的编码树来量化高阶依赖的信息代价，从而学习高阶特征相关性，解决了多视图多标签特征选择中现有信息论方法难以捕捉高阶结构相关性且易陷入局部最优的问题。

    

    近年来，多视图多标签学习（MVML）因与真实世界场景高度契合而受到广泛关注。信息论方法因能够学习非线性相关性而日益受到重视。然而，仍存在两个关键挑战：首先，真实世界数据中的特征通常表现出高阶结构相关性，但现有的信息论方法难以学习此类相关性；其次，信息论方法通常依赖启发式优化，容易收敛到局部最优。为解决这两个挑战，我们提出了一种名为“面向多视图多标签特征选择的结构熵引导高阶相关性学习”（SEHFS）的新方法。SEHFS的核心思想是将特征图转换为结构熵最小化的编码树，量化高阶依赖关系的信息代价，从而学习高阶特征相关性。

    arXiv:2603.03022v2 Announce Type: replace  Abstract: In recent years, multi-view multi-label learning (MVML) has attracted extensive attention due to its close alignment to real-world scenarios. Information-theoretic methods have gained prominence for learning nonlinear correlations. However, two key challenges persist: first, features in real-world data commonly exhibit high-order structural correlations, but existing information-theoretic methods struggle to learn such correlations; second, commonly relying on heuristic optimization, information-theoretic methods are prone to converging to local optima. To address these two challenges, we propose a novel method called Structural Entropy Guided High-Order Correlation Learning for Multi-View Multi-Label Feature Selection (SEHFS). The core idea of SEHFS is to convert the feature graph into a structural-entropy-minimizing encoding tree, quantifying the information cost of high-order dependencies and thus learning high-order feature corre
    
[^356]: 半真陈述破坏基于相似度的检索

    Half-Truths Break Similarity-Based Retrieval

    [https://arxiv.org/abs/2602.23906](https://arxiv.org/abs/2602.23906)

    本文发现CLIP模型会被“半真半假”的错误描述欺骗（相似度反而升高），并提出CS-CLIP方法，通过将描述分解为实体和关系单元并构建最小编辑的干扰样本进行对比微调，从而增强模型对细节错误的识别能力。

    

    当在文本描述中添加一个额外细节时，如果该细节是错误的，图文相似度理应下降。我们研究表明，CLIP风格的双编码器常常违背这一直觉：在原本正确的描述后附加一个看似合理但不正确的对象或关系，反而可能提高相似度得分。我们将这类情况称为“半真陈述”。在COCO数据集上，CLIP仅有40.6%的情况下更偏好正确但较短的描述，而当添加的细节是关系时，这一比例降至32.9%。我们将这一脆弱性归因于对描述各部分的弱监督：对比训练只对齐完整句子，并未显式地强制单个实体和关系与图像准确对应。我们提出CS-CLIP（组件监督CLIP），该方法将描述分解为实体和关系单元，为每个单元构建最小编辑的干扰样本，并微调模型使其对正确单元的评分高于干扰样本，同时保持标准双编码器架构……

    arXiv:2602.23906v1 Announce Type: cross  Abstract: When a text description is extended with an additional detail, image-text similarity should drop if that detail is wrong. We show that CLIP-style dual encoders often violate this intuition: appending a plausible but incorrect object or relation to an otherwise correct description can increase the similarity score. We call such cases half-truths. On COCO, CLIP prefers the correct shorter description only 40.6% of the time, and performance drops to 32.9% when the added detail is a relation. We trace this vulnerability to weak supervision on caption parts: contrastive training aligns full sentences but does not explicitly enforce that individual entities and relations are grounded. We propose CS-CLIP (Component-Supervised CLIP), which decomposes captions into entity and relation units, constructs a minimally edited foil for each unit, and fine-tunes the model to score the correct unit above its foil while preserving standard dual-encoder 
    
[^357]: 拓扑结构赋能全球河系基于学习的水动力预测

    Topology enables learning-based hydrodynamic prediction of the global river system

    [https://arxiv.org/abs/2602.22293](https://arxiv.org/abs/2602.22293)

    本文提出GraphRiverCast，利用河流拓扑结构实现全球河系的每日水动力预测，无需初始状态，精度提升25%并泛化至未测量区域。

    

    准确的河流预测对于水、粮食和能源安全至关重要，但在整个河网中仍具挑战性。机器学习已变革了地球系统建模，但河系预测的系统级进展因缺乏可靠数据而滞后。利用河流的连通性和耗散动力学，我们引入了GraphRiverCast，这是一个针对全球河系的神经模型，能在0.25度网络的每个河段预测每日多元水动力，仅依赖稀疏的测量站且无需初始状态。该模型在预测时效上无内在技能衰减，在准确性上比领先的全球河流模型高出25%，并能稳健地泛化至未测量的河段和更精细的分辨率。GraphRiverCast将基于学习的河流预测从孤立流域提升为统一的全球系统，并为数据稀缺的地球系统中的机器学习提供了见解。

    arXiv:2602.22293v2 Announce Type: replace  Abstract: Accurate river prediction is essential for water, food and energy security, yet remains challenging across entire river networks. Machine learning has transformed Earth-system modeling, but a system-level advance for river prediction lags for lack of reliable data. Exploiting the connectivity and dissipative dynamics of rivers, we introduce GraphRiverCast, a neural model for global river systems that predicts daily multivariate hydrodynamics at every reach of a 0.25{\deg}network with only sparse gauges and no initial state. It shows no intrinsic skill decay with lead time, outperforms leading global river models by 25% in accuracy, and robustly generalizes to ungauged reaches and finer resolutions. GraphRiverCast lifts learning-based river prediction from isolated basins to a unified global system and offers insights for machine learning in data-scarce Earth systems.
    
[^358]: 表格基础模型能够学习关联规则

    Tabular Foundation Models Can Learn Association Rules

    [https://arxiv.org/abs/2602.14622](https://arxiv.org/abs/2602.14622)

    本文提出了一个模型无关的关联规则学习框架及其实现TabProbe，利用表格基础模型作为条件概率估计器，无需频繁项集挖掘即可从表格数据中学习关联规则，同时克服了经典方法的规则爆炸问题和神经网络方法在低数据量下的性能缺陷。

    

    关联规则挖掘（ARM）是从表格数据中进行知识发现的一项基础任务，被广泛应用于高风险决策场景。经典的ARM方法依赖频繁项集挖掘，会导致规则爆炸和可扩展性差的问题；而近期的神经网络方法虽然缓解了这些问题，但在低数据量场景下性能会明显下降。表格基础模型（TFMs）在多样化的表格数据上进行预训练，具有强大的上下文泛化能力，为解决这些局限性提供了基础。我们提出了一个模型无关的关联规则学习框架，能够从任何基于表格数据的条件概率模型中提取关联规则，从而使我们能够利用TFMs。在此基础上，我们介绍了TabProbe，即该框架的一个实例化实现，它将TFMs用作条件概率估计器，无需频繁项集挖掘即可开箱即用地学习关联规则。我们在各种表格数据集上对该方法进行了评估……

    arXiv:2602.14622v3 Announce Type: replace  Abstract: Association Rule Mining (ARM) is a fundamental task for knowledge discovery in tabular data and is widely used in high-stakes decision-making. Classical ARM methods rely on frequent itemset mining, leading to rule explosion and poor scalability, while recent neural approaches mitigate these issues but suffer from degraded performance in low-data regimes. Tabular foundation models (TFMs), pretrained on diverse tabular data with strong in-context generalization, provide a basis for addressing these limitations. We introduce a model-agnostic association rule learning framework that extracts association rules from any conditional probabilistic model over tabular data, enabling us to leverage TFMs. We then introduce TabProbe, an instantiation of our framework that utilizes TFMs as conditional probability estimators to learn association rules out-of-the-box without frequent itemset mining. We evaluate our approach on tabular datasets of va
    
[^359]: 协方差感知的Transformer用于二次规划与决策制定

    Covariance-Aware Transformers for Quadratic Programming and Decision Making

    [https://arxiv.org/abs/2602.14506](https://arxiv.org/abs/2602.14506)

    本文从理论上证明Transformer可通过模拟梯度下降和软阈值迭代来求解无约束及ℓ1惩罚/约束的二次规划，并据此提出Time2Decide方法，通过向时间序列基础模型显式输入变量间协方差矩阵，显著提升投资组合优化等决策任务的表现。

    

    我们探索了使用Transformer求解二次规划问题，以及这种能力如何有益于涉及协方差矩阵的决策问题。我们首先证明，线性注意力机制可以通过逐行对矩阵变量（例如目标函数½x^T Ax + b^T x中的A）进行分词（tokenize）并模拟梯度下降迭代，从而可证明地求解无约束二次规划（QP）。此外，通过结合MLP，一个Transformer块可以求解：(i) ℓ1惩罚的QP（通过模拟迭代软阈值算法）；以及(ii) 在配备额外反馈回路的情况下求解ℓ1约束的QP。我们的理论启发我们提出Time2Decide：一种通用方法，通过显式输入变量之间的协方差矩阵来增强时间序列基础模型（TSFM）。我们通过实验发现，对于经典的带有ℓ1约束的投资组合优化问题，Time2Decide在所有情况下都一致优于基础TSFM模型。

    arXiv:2602.14506v2 Announce Type: replace  Abstract: We explore the use of transformers for solving quadratic programs and how this capability benefits decision-making problems that involve covariance matrices. We first show that the linear attention mechanism can provably solve unconstrained QPs by tokenizing the matrix variables (e.g.~$A$ of the objective $\frac{1}{2}x^\top Ax+b^\top x$) row-by-row and emulating gradient descent iterations. Furthermore, by incorporating MLPs, a transformer block can solve (i) $\ell_1$-penalized QPs by emulating iterative soft-thresholding and (ii) $\ell_1$-constrained QPs when equipped with an additional feedback loop. Our theory motivates us to introduce Time2Decide: a generic method that enhances a time series foundation model (TSFM) by explicitly feeding the covariance matrix between the variates. We empirically find that Time2Decide uniformly outperforms the base TSFM model for the classical portfolio optimization problem that admits an $\ell_1$-
    
[^360]: 从错误中学习：自校正掩码扩散模型

    Learn from Your Mistakes: Self-Correcting Masked Diffusion Models

    [https://arxiv.org/abs/2602.11590](https://arxiv.org/abs/2602.11590)

    该论文提出渐进式自我校正方法，通过复用去噪网络输出训练校正器，使掩码扩散模型能够在生成过程中迭代修正包括已生成词元在内的整个序列，从而避免误差累积并提升样本质量。

    

    掩码扩散模型（MDMs）已成为自回归模型的一种有前景的替代方案，能够在实现并行词元生成的同时取得具有竞争力的性能。尽管具有这些优势，MDMs面临一个根本性局限：词元一旦被去掩码后就保持固定不变，导致误差不断累积，最终降低样本质量。我们通过提出一个训练模型同时执行去掩码和校正的框架来解决这一问题。通过将MDM去噪网络的输出复用为校正器训练的输入，我们训练模型从潜在的错误中恢复。在生成过程中，我们在去掩码步骤之间应用额外的校正性精炼步骤，以改变已解码的词元并改进输出结果。我们将这一训练与采样方法命名为渐进式自我校正，因其具有迭代精炼整个序列（包括已生成的词元）的独特能力。我们开展了大量实验……

    arXiv:2602.11590v4 Announce Type: replace  Abstract: Masked diffusion models (MDMs) have emerged as a promising alternative to autoregressive models, enabling parallel token generation while achieving competitive performance. Despite these advantages, MDMs face a fundamental limitation: once tokens are unmasked, they remain fixed, leading to error accumulation and ultimately degrading sample quality. We address this by proposing a framework that trains a model to perform both unmasking and correction. By reusing outputs from the MDM denoising network as inputs for corrector training, we train a model to recover from potential mistakes. During generation we apply additional corrective refinement steps between unmasking ones in order to change decoded tokens and improve outputs. We name our training and sampling method Progressive Self-Correction (ProSeCo) for its unique ability to iteratively refine an entire sequence, including already generated tokens. We conduct extensive experimenta
    
[^361]: 针对带对抗性污染的异方差广义线性赌博机的联合高效且近最优算法

    A Jointly Efficient and Near-Optimal Algorithm for Heteroskedastic Generalized Linear Bandits with Adversarial Corruptions

    [https://arxiv.org/abs/2602.10971](https://arxiv.org/abs/2602.10971)

    本文提出将基于在线镜像下降的估计器与基于Hessian的置信度权重相结合的HCW-GLB-OMD算法，为带对抗性污染的异方差广义线性赌博机实现了联合高效且近最优的高概率遗憾保证。

    

    我们研究了带有对抗性污染的异方差广义线性赌博机（GLBs），并提出了一种简单算法 HCW-GLB-OMD，该算法将基于在线镜像下降（OMD）的估计器与基于Hessian矩阵的置信度权重相结合。尽管这两个组成部分都已被充分研究，但要为该组合算法建立理论保证仍需要非平凡的分析。我们利用先前分析中不需要的技术工具（包括加权似然上鞅和自约束论证）建立了抗污染的置信序列。在遗憾分析方面，我们推导了一个新的Hessian势函数引理，以处理对抗性污染与自适应置信度加权之间的相互作用。对于自协和（self-concordant）广义线性模型，我们证明了高概率遗憾界 $\widetilde{\mathcal{O}}\bigl( d \sqrt{\sum_t g(\tau_t) \dot\mu_{t,\star}} + d^2 g_{\max} \kappa + d(1+\kappa)(C\vee1) \bigr)$，其中 $\dot\mu_{t,\star}$ 是……

    arXiv:2602.10971v3 Announce Type: replace  Abstract: We consider heteroskedastic generalized linear bandits (GLBs) with adversarial corruptions and propose \texttt{HCW-GLB-OMD}, a simple algorithm that combines an online mirror descent (OMD)-based estimator with Hessian-based confidence weights. Although both components are well studied, establishing guarantees for the combined algorithm requires a nontrivial analysis. We establish a corruption-robust confidence sequence using technical tools not required in prior analyses, including a weighted likelihood supermartingale and a self-bounding argument. For the regret analysis, we derive a new Hessian potential lemma to handle the interaction between adversarial corruption and adaptive confidence weighting. For self-concordant GLMs, we prove a high-probability regret bound of $\widetilde{\mathcal{O}}\bigl( d \sqrt{\sum_t g(\tau_t) \dot\mu_{t,\star}} + d^2 g_{\max} \kappa + d(1+\kappa)(C\vee1) \bigr)$, where $\dot\mu_{t,\star}$ is the slop
    
[^362]: 基于自助聚合的高效自适应恶意LLM提示检测

    Efficient and Adaptable Detection of Malicious LLM Prompts via Bootstrap Aggregation

    [https://arxiv.org/abs/2602.08062](https://arxiv.org/abs/2602.08062)

    提出了BAGEL框架，通过自助聚合与专家混合启发的微调模型集成，以随机选择成员子集并可选配合轻量级路由器的方式，实现了高效、轻量且可增量更新的恶意LLM提示检测。

    

    大型语言模型在自然语言理解方面展现了卓越的能力。然而，它们仍然容易受到恶意提示的攻击，这些提示会诱发不安全或违反政策的行为。现有的黑盒审核API防御方案透明度有限，且难以适应不断演变的威胁；而使用大型LLM裁判的白盒方法则带来高昂的计算成本，并需要针对新攻击进行昂贵的重新训练。我们提出了BAGEL（Bootstrap AGgregated Ensemble Layer，自助聚合集成层），这是一个模块化、轻量级且可增量更新的恶意提示检测框架。BAGEL采用自助聚合和受专家混合启发的微调模型集成方式，每个模型专门针对不同的攻击数据集进行训练。在推理时，BAGEL聚合从随机选择的集成成员子集中得到的预测结果，并可选择通过一个轻量级的学习型路由器进行增强，该路由器预测最适合处理该输入的成员模型。

    arXiv:2602.08062v2 Announce Type: replace  Abstract: Large Language Models have demonstrated remarkable capabilities in natural language understanding. However, they remain susceptible to malicious prompts that induce unsafe or policy-violating behavior. Existing defenses in black-box moderation APIs offer limited transparency and adapt poorly to evolving threats, while white-box approaches using large LLM judges impose prohibitive computational costs and require expensive retraining for new attacks. We present BAGEL (Bootstrap AGgregated Ensemble Layer), a modular, lightweight, and incrementally updatable framework for malicious prompt detection. BAGEL employs a bootstrap aggregation and mixture-of-experts-inspired ensemble of finetuned models, each specialized on a different attack dataset. At inference, BAGEL aggregates predictions from a stochastically selected subset of ensemble members, optionally augmented by a lightweight learned router that predicts the most suitable member to
    
[^363]: 智能体即同伴：将偏差受控的大语言模型评估形式化为多智能体同伴互评

    Agents-as-Peers: Bias-Controlled LLM Evaluation as Multi-Agent Peer Assessment

    [https://arxiv.org/abs/2602.02589](https://arxiv.org/abs/2602.02589)

    该论文提出Agents-as-Peers分布式评估协议，让每个大语言模型智能体生成任务、回答共享任务集并互评同伴答案（排除自评），以解决固定基准老化和集中式评判者偏差问题，并对自我偏差、身份效应等评判偏差进行受控测量。

    

    大语言模型（LLM）评估通常依赖于固定的人类编写基准或集中式的人类/模型评判者。固定基准可能会过时并进入模型训练语料库，而集中式评判者则会引入自身的评分偏好。我们将大语言模型评估视为一个多智能体同伴互评问题，并提出了Agents-as-Peers——一种分布式评估协议，其中每个智能体负责生成任务、回答共享的任务集，并评判其同伴的答案，不存在任何特权评估者。自我评分被排除在排名之外，并且评判过程在受控的身份与呈现模式下重复进行，从而使自我偏差、身份效应、呈现位置敏感性和评判者严格性得到实际测量，而非保持隐含。在一个由12个大语言模型和420个智能体生成问题组成的实例化实验中，Agents-as-Peers在每个模式下聚合了55,440个非自我同伴评判。大多数智能体给自己的答案打分偏……（原文在此截断）

    arXiv:2602.02589v2 Announce Type: replace  Abstract: Large language model (LLM) evaluation commonly relies on fixed human-authored benchmarks or centralized human/model judges. Fixed benchmarks can age and enter model training corpora, while centralized judges introduce their own scoring preferences.   We cast LLM evaluation as a multi-agent peer-assessment problem and introduce Agents-as-Peers, a distributed evaluation protocol in which every agent generates tasks, answers the shared task set, and judges the answers of its peers, with no privileged evaluator.   Self-ratings are excluded from the ranking, and judging is repeated under controlled identity and presentation regimes so that self bias, identity effects, presentation-position sensitivity, and judge severity are measured rather than left implicit.   In an instantiation with 12 LLMs and 420 agent-generated questions, Agents-as-Peers aggregates 55,440 non-self peer judgments per regime.   Most agents assign their own answers hi
    
[^364]: TTCS：面向自我进化的测试时课程合成

    TTCS: Test-Time Curriculum Synthesis for Self-Evolving

    [https://arxiv.org/abs/2601.22628](https://arxiv.org/abs/2601.22628)

    TTCS提出了一种共同进化的测试时训练框架，通过问题合成器与推理求解器的迭代协同优化，生成难度递增的课程化问题变体，从而提升大语言模型在困难推理问题上的能力。

    

    测试时训练为提升大型语言模型（LLM）的推理能力提供了一种很有前景的方法，它仅利用测试问题来适配模型。然而，现有方法在应对困难推理问题时存在两大障碍：原始测试问题往往过于困难，难以产生高质量的伪标签；同时测试集规模有限，导致持续的在线更新容易不稳定。为了解决这些局限，我们提出了TTCS——一个共同进化的测试时训练框架。具体而言，TTCS从同一个预训练模型初始化两个策略：问题合成器与推理求解器。这两个策略通过迭代优化实现共同进化：合成器以测试问题为条件，生成难度逐步提升的问题变体，为求解器的当前能力量身打造结构化课程；而求解器则利用由（原文此处截断）计算得到的自洽性奖励来更新自身。

    arXiv:2601.22628v3 Announce Type: replace-cross  Abstract: Test-Time Training offers a promising way to improve the reasoning ability of large language models (LLMs) by adapting the model using only the test questions. However, existing methods struggle with difficult reasoning problems for two reasons: raw test questions are often too difficult to yield high-quality pseudo-labels, and the limited size of test sets makes continuous online updates prone to instability. To address these limitations, we propose TTCS, a co-evolving test-time training framework. Specifically, TTCS initializes two policies from the same pretrained model: a question synthesizer and a reasoning solver. These policies evolve through iterative optimization: the synthesizer generates progressively challenging question variants conditioned on the test questions, creating a structured curriculum tailored to the solver's current capability, while the solver updates itself using self-consistency rewards computed from
    
[^365]: 期望回报导致强化学习中结果层面的模式坍缩，以及如何用逆概率缩放来修复它

    Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling

    [https://arxiv.org/abs/2601.21669](https://arxiv.org/abs/2601.21669)

    该论文证明强化学习中结果层面的模式坍缩并非源于探索或正则化不足，而是期望回报目标本身的结构性必然结果，并提出用逆概率缩放的方法来修复这一问题。

    

    许多强化学习（RL）问题存在多个质量相当的终端解决方案，其目标不是找到单一最优解，而是表示一组多样化的高质量结果。然而，通过标准期望回报最大化训练出的策略往往会坍缩到一小部分结果上，这一现象通常被归因于探索不足或正则化不足。我们证明这一解释是不完整的：结果层面的模式坍缩是期望回报目标本身的结构性必然结果。在理想化的学习动力学下，任意两个结果之间的对数概率比会随其奖励差线性演化，这意味着概率比将呈指数级发散，且无论采用何种探索策略、熵正则化或优化算法，坍缩都不可避免。我们将这一病态的根源确认为期望内部的概率乘数……

    arXiv:2601.21669v2 Announce Type: replace-cross  Abstract: Many reinforcement learning (RL) problems admit multiple terminal solutions of comparable quality, where the goal is not to identify a single optimum but to represent a diverse set of high-quality outcomes. Nevertheless, policies trained by standard expected return maximization routinely collapse onto a small subset of outcomes, a phenomenon commonly attributed to insufficient exploration or weak regularization. We show that this explanation is incomplete: outcome level mode collapse is a structural consequence of the expected-return objective itself. Under idealized learning dynamics, the log-probability ratio between any two outcomes evolves linearly in their reward difference, implying exponential ratio divergence and inevitable collapse independent of the exploration strategy, entropy regularization, or optimization algorithm. We identify the source of this pathology as the probability multiplier inside the expectation and 
    
[^366]: 高维持续线性回归中的最优L2正则化

    Optimal L2 Regularization in High-dimensional Continual Linear Regression

    [https://arxiv.org/abs/2601.13844](https://arxiv.org/abs/2601.13844)

    该论文在高维过参数化持续线性回归中推导出对任意线性教师均成立的期望泛化损失闭式表达式，并首次在理论持续学习中证明了最优L2正则化强度应随任务数T近似线性增长（为T/ln T）。

    

    我们研究了过参数化持续线性回归设置中的泛化问题，其中模型在一系列任务上使用L2（各向同性）正则化进行训练。我们在高维情形下推导出了期望泛化损失的闭式表达式，该表达式对任意线性教师均成立。我们证明了各向同性正则化在单教师和多个独立同分布教师的设置下都能缓解标签噪声的影响，而此前支持多教师的工作要么没有采用正则化，要么使用了内存开销较大的方法。此外，我们证明了最优固定正则化强度随任务数量T近似线性增长，具体为T/ln T。据我们所知，这是理论持续学习领域中首个此类结果。最后，我们通过线性回归和神经网络的实验验证了理论发现，展示了这一缩放律如何影响模型性能。

    arXiv:2601.13844v3 Announce Type: replace  Abstract: We study generalization in an overparameterized continual linear regression setting, where a model is trained with L2 (isotropic) regularization across a sequence of tasks. We derive a closed-form expression for the expected generalization loss in the high-dimensional regime that holds for arbitrary linear teachers. We demonstrate that isotropic regularization mitigates label noise under both single-teacher and multiple i.i.d. teacher settings, whereas prior work accommodating multiple teachers either did not employ regularization or used memory-demanding methods. Furthermore, we prove that the optimal fixed regularization strength scales nearly linearly with the number of tasks $T$, specifically as $T/\ln T$. To our knowledge, this is the first such result in theoretical continual learning. Finally, we validate our theoretical findings through experiments on linear regression and neural networks, illustrating how this scaling law af
    
[^367]: 激活敏感性作为训练后量化的统一原理

    Activation Sensitivity as a Unifying Principle for Post-Training Quantization

    [https://arxiv.org/abs/2601.11663](https://arxiv.org/abs/2601.11663)

    该论文证明AWQ、GPTQ、对角Fisher量化、块重构和Hadamard旋转等主流训练后量化方法都是最小化同一个由梯度加权激活敏感性支配的二次型损失的特殊情形，从而将这些方法统一在一个理论框架之下。

    

    arXiv:2601.11663v2 公告类型： replace-cross。摘要：面向大语言模型的训练后量化（PTQ）方法通过优化层级的代理目标函数，在不进行再训练的情况下将预训练权重压缩至低比特宽度。激活感知缩放方法（如AWQ）保护具有大激活范数的输入通道，而二阶误差补偿方法（如GPTQ）则利用未加权输入协方差矩阵的逆来重新分配舍入误差。我们证明，这两种范式——连同对角Fisher量化、块重构以及Hadamard旋转方法——都是最小化由梯度加权激活敏感性所支配的单一损失导出二次型的特例。基于线性层 $Y = X W^\top$ 处任务损失的Gauss-Newton展开，我们证明在各向同性输出曲率下，权重扰动 $\Delta W$ 引起的期望超额损失可简化为 $\frac{1}{2 d_{\mathrm{out}}}\operatorname{Tr}(\Delta W H_D \Delta W^\top)$ 形式的表达式。

    arXiv:2601.11663v2 Announce Type: replace-cross  Abstract: Post-training quantization (PTQ) methods for large language models compress pretrained weights to low bit-widths without retraining by optimizing layer-local proxy objectives. Activation-aware scaling methods such as AWQ protect input channels with large activation norms, whereas second-order error-compensation methods such as GPTQ redistribute rounding error using the inverse of the unweighted input covariance matrix. We show that both paradigms -- along with diagonal Fisher quantization, block reconstruction, and Hadamard rotation methods -- are special cases of minimizing a single loss-derived quadratic form governed by gradient-weighted activation sensitivity. From a Gauss-Newton expansion of the task loss at a linear layer $Y = X W^\top$, we prove that under isotropic output curvature the expected excess loss from weight perturbation $\Delta W$ reduces to $\frac{1}{2 d_{\mathrm{out}}}\operatorname{Tr}(\Delta W H_D \Delta W
    
[^368]: 张量计算应用程序编程接口（TCAPI）：一种面向应用的、轻量级的可移植高性能张量网络计算API

    Tensor computing application programming interface: An application-oriented, lightweight API for portable high-performance tensor network computations

    [https://arxiv.org/abs/2512.23917](https://arxiv.org/abs/2512.23917)

    提出了TCAPI——一个面向应用的轻量级编程接口，通过抽象张量对象和定义C++/Python间语义共享的精简核心函数集，实现了跨框架、跨硬件的可移植高性能张量网络计算。

    

    张量网络（TNs）是量子科学和人工智能领域的核心计算工具。然而，缺乏跨张量计算框架的统一软件接口严重限制了TN应用程序的可移植性，使算法开发与特定的硬件和软件后端紧密耦合。为了应对这一挑战，我们提出了张量计算应用程序编程接口（TCAPI）——一个面向应用的轻量级API，旨在实现与框架无关的高性能张量网络应用，目前已为C++和Python两种语言制定了规范。TCAPI提供了明确定义的类型系统约定以抽象化张量对象，并配备了一套精简而富有表现力的核心函数集，用于基本的张量操作和张量线性代数运算，且操作语义在两种语言规范之间保持共享一致。通过对代表性张量网络应用的数值演示（摘要在此处截断）

    arXiv:2512.23917v2 Announce Type: replace-cross  Abstract: Tensor networks (TNs) are a central computational tool in quantum science and artificial intelligence. However, the lack of a unified software interface across tensor-computing frameworks severely limits the portability of TN applications, coupling algorithmic development to specific hardware and software back ends. To address this challenge, we introduce the Tensor Computing Application Programming Interface (TCAPI)---an application-oriented, lightweight API designed to enable framework-independent, high-performance TN applications, currently specified for \texttt{C++} and \texttt{Python}. TCAPI provides well-defined type-system conventions that abstract tensor objects, together with a minimal yet expressive set of core functions for essential tensor manipulations and tensor linear-algebra operations, with operation semantics shared across language specifications. Through numerical demonstrations of representative tensor-netwo
    
[^369]: 个性化对齐中的奖励模型选择危机

    The Reward Model Selection Crisis in Personalized Alignment

    [https://arxiv.org/abs/2512.23067](https://arxiv.org/abs/2512.23067)

    该论文揭示传统的奖励模型准确性指标无法预测其在奖励引导解码中的实际表现，并提出“策略准确性”新指标，指出上游奖励模型准确性与下游策略准确性仅有微弱相关性（Kendall's tau = 0.08–0.31）。

    

    基于偏好数据的个性化对齐研究主要集中于提升个人奖励模型（RM）的准确性，其隐含假设是更好的偏好排序能力会转化为更好的个性化行为。然而在实际部署中，计算资源的限制使得必须采用推理时自适应方法（如奖励引导解码，RGD），而非针对每个用户进行策略微调。这就产生了一个关键但长期被忽视的需求：奖励模型不仅要能准确排序偏好，还必须能够有效引导文本生成。我们证明，标准的RM准确性作为面向部署的个性化奖励模型的选拔标准会遭遇灾难性失效。我们提出了“策略准确性”这一新指标，用于量化经过RGD自适应的大语言模型能否正确区分偏好响应与不偏好响应，并发现上游RM准确性与下游策略准确性之间仅有微弱的相关性（Kendall's tau = 0.08–0.31）。更关键的是，w

    arXiv:2512.23067v3 Announce Type: replace  Abstract: Personalized alignment from preference data has focused primarily on improving personal reward model (RM) accuracy, with the implicit assumption that better preference ranking translates to better personalized behavior. However, in deployment, computational constraints necessitate inference-time adaptation such as reward-guided decoding (RGD) rather than per-user policy fine-tuning. This creates a critical but overlooked requirement: reward models must not only rank preferences accurately but also effectively guide generation. We demonstrate that standard RM accuracy fails catastrophically as a selection criterion for deployment-ready personalized rewards. We introduce policy accuracy; a metric quantifying whether RGD-adapted LLMs correctly discriminate between preferred and dispreferred responses and show that upstream RM accuracy correlates only weakly with downstream policy accuracy (Kendall's tau = 0.08--0.31). More critically, w
    
[^370]: 基于选择性部署的鲁棒残差校正时间序列预测方法

    Robust Residual Correction via Selective Deployment for Time Series Forecasting

    [https://arxiv.org/abs/2512.22428](https://arxiv.org/abs/2512.22428)

    CRC框架通过验证集校准的选择性部署规则，仅在校正能带来净误差降低时才应用混合残差校正，从而在时间序列预测中实现鲁棒的误差降低，尤其在复杂场景下避免了过拟合问题。

    

    事后残差校正可以降低预测误差，但持续开启的更新往往难以泛化：当残差信号微弱或充满噪声时，表达能力强的校正器可能会过拟合，甚至增大整体预测误差。我们提出了CRC，一个鲁棒的残差校正框架，它将混合校正器与经过验证集校准的选择性部署规则相结合。CRC通过保守的岭回归“底线”加上条件性非线性精炼来生成校正，但只有在留出数据证据表明相对于冻结的基线预测器能够净减少误差时才部署这些校正。CRC将“何时修正预测”视为一个经验证集校准的部署决策，而不仅仅是“如何拟合残差”。在标准长序列基准测试和多个骨干模型上，CRC在大多数数据集-骨干模型组合中提升了对齐平均的MSE和MAE，在复杂数据集上收益尤为显著（例如，在Traffic数据集上相对MSE降低超过20%）。

    arXiv:2512.22428v3 Announce Type: replace  Abstract: Post-hoc residual correction can reduce forecasting error, but always-on updates often fail to generalize: when residual signals are weak or noisy, expressive correctors can overfit and even increase the overall forecasting errors. We introduce CRC, a robust residual correction framework that couples a hybrid corrector with a validation-calibrated selective deployment rule. CRC proposes corrections via a conservative ridge "floor" plus conditional nonlinear refinement, but deploys them only when held-out evidence indicates net error reduction over a frozen baseline forecaster. CRC treats when to revise a forecast as a validation-calibrated deployment decision, not only how to fit residuals. Across standard long-sequence benchmarks and multiple backbones, CRC improves horizon-averaged MSE and MAE in most dataset-backbone pairs, with particularly large gains on complex datasets (e.g., over 20% relative MSE reduction on Traffic). We add
    
[^371]: OPTIMA：基于二次规划重构的大语言模型最优一次性剪枝方法

    OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction

    [https://arxiv.org/abs/2512.13886](https://arxiv.org/abs/2512.13886)

    OPTIMA将掩码选择后的逐层权重重构表述为共享单个Hessian矩阵的按列二次规划问题，并借助加速器友好的QP求解器，在大语言模型上同时实现了剪枝的高精度与可扩展性。

    

    训练后剪枝能够降低大语言模型的推理成本，但其面临一个权衡问题：将权重置零的简单启发式方法虽然速度快，却会降低模型精度；而基于原理的联合优化方法虽然能恢复精度，但在现代规模下计算上不可行。SparseGPT等一次性（one-shot）方法介于两者之间，采用高效但近似的权重更新。为了弥合这一差距，我们提出了OPTIMA，一种在精度和可扩展性之间取得平衡的实用一次性训练后剪枝方法。OPTIMA将掩码选择后的逐层权重重构问题表述为独立的、按列划分的二次规划问题（QP），这些二次规划在每个权重矩阵上共享同一个Hessian矩阵。在求解器收敛时，求解这些二次规划可以得到共享Hessian下逐层重构目标的逐列最优解，且共享的Hessian矩阵使得这些二次规划问题易于在加速器上进行批量处理。我们实现了一个对加速器友好的QP求解器，该求解器可累积……

    arXiv:2512.13886v3 Announce Type: replace-cross  Abstract: Post-training pruning cuts the inference cost of large language models, yet it faces a trade-off: simple heuristics that zero weights are fast but degrade accuracy, while principled joint optimization methods recover accuracy but are computationally infeasible at modern scale. One-shot methods such as SparseGPT sit in between and apply efficient but approximate weight updates. To close this gap, we introduce OPTIMA, a practical one-shot post-training pruning method that balances accuracy and scalability. OPTIMA casts layer-wise weight reconstruction after mask selection as independent, column-wise quadratic programs (QPs) that share one Hessian per weight matrix. Solving these QPs yields, at solver convergence, the per-column minimizer of the layer-wise reconstruction objective under the shared Hessian, and the shared Hessian makes the QPs easy to batch on accelerators. We implement an accelerator-friendly QP solver that accumu
    
[^372]: FLEXI-Haz：超越比例风险假设的半参数深度生存推断

    FLEXI-Haz: Semiparametric Deep Survival Inference Beyond Proportional Hazards

    [https://arxiv.org/abs/2512.10570](https://arxiv.org/abs/2512.10570)

    提出 FLEXI-Haz 半参数深度神经网络模型，在突破比例风险假设限制的同时保留可解释的回归参数，并针对右删失数据实现了极小化极大最优的收敛速率及渐近有效的半参数统计推断。

    

    灵活的生存预测本身并不能保证对协变量效应或个体特异性生存概率进行有效的统计推断。我们提出了 FLEXI-Haz，一种针对右删失数据的半参数深度神经网络模型，它将对数风险中可解释的线性成分与关于时间及调整协变量的灵活函数相结合。这种构造在允许存在非比例的讨厌参数效应的同时，保留了可解释的回归参数。在正则性条件下，我们建立了在复合 Hölder 函数类上直至对数因子意义下达到极小化极大最优的讨厌参数收敛速率，并实现了回归参数的 √n 一致估计、渐近正态性以及半参数有效性。对于取值有限的调整协变量，我们推导了累积风险的显式影响函数，并构造了交叉拟合的一步估计量，该估计量在逐点意义上具有渐近有效性。

    arXiv:2512.10570v3 Announce Type: replace-cross  Abstract: Flexible survival prediction does not by itself ensure valid inference for covariate effects or subject-specific survival probabilities. We develop FLEXI-Haz, a semiparametric deep neural network model for right-censored data that combines an interpretable linear component in the log-hazard with a flexible function of time and adjustment covariates. This formulation permits nonproportional nuisance effects while retaining interpretable regression parameters. Under regularity conditions, we establish nuisance convergence rates that are minimax optimal up to logarithmic factors over composite Holder classes, together with root-n-consistent, asymptotically normal, and semiparametrically efficient estimation of the regression parameter. For adjustment covariates with finite support, we derive the explicit influence function for cumulative hazard and construct a cross-fitted one-step estimator yielding asymptotically valid pointwise
    
[^373]: 子群平衡何时重要？基于潜在可分离性预测性能敏感性

    When does Subgroup Balance Matter? Predicting Performance Sensitivity from Latent Separability

    [https://arxiv.org/abs/2512.09496](https://arxiv.org/abs/2512.09496)

    该论文提出“潜在分离假设”，指出微调模型对训练数据子群平衡的依赖程度由预训练模型潜在空间中子群间的分离程度决定，并通过理论分析与四个视觉及语言数据集上的实证和正则化实验验证了这一假设。

    

    训练数据中人口子群的不均衡表示给模型在不同人群上的泛化带来了挑战。通常的做法假设平衡子群表示总是能优化性能。然而，近期的实证结果与这一假设相矛盾：在某些情况下，不均衡的数据分布实际上会提升子群性能；而在另一些情况下，即使训练中完全缺失某个子群，子群性能也不受影响。我们提出了潜在分离假设，即微调后的模型对子群表示的依赖程度由预训练模型潜在空间中各子群之间的分离程度所决定。我们对这一假设进行了形式化表述，提供了理论分析，在四个视觉与语言数据集上进行了实证验证，并通过正则化实验提供了其因果作用的初步证据。最后，我们展示了如何量化……

    arXiv:2512.09496v2 Announce Type: replace-cross  Abstract: Unequal representation of demographic groups in training data poses challenges to model generalisation across populations. Standard practice assumes that balancing subgroup representation always optimises performance. However, recent empirical results contradict this assumption: in some cases, imbalanced data distributions actually improve subgroup performance, and in others, subgroup performance is unaffected by the absence of an entire subgroup during training. We propose the latent separation hypothesis, which states that a fine-tuned model's dependence on subgroup representation is determined by the degree of separation between subgroups in the latent space of the pre-trained model. We formalise this hypothesis, provide theoretical analysis, validate it empirically across four vision and language datasets, and present preliminary evidence of a causal role through regularisation experiments. Finally, we show how quantitative
    
[^374]: KGOT：统一知识图谱与最优传输伪标签用于分子-蛋白质相互作用预测

    KGOT: Unified Knowledge Graph and Optimal Transport Pseudo-Labeling for Molecule-Protein Interaction Prediction

    [https://arxiv.org/abs/2512.09365](https://arxiv.org/abs/2512.09365)

    该论文提出KGOT框架，通过构建统一知识图谱整合分子、蛋白质、基因和通路等多源生物数据，并利用最优传输方法为未标记的分子-蛋白质对生成高质量伪标签，从而同时解决标签稀缺和生物背景信息缺失两大挑战，提升分子-蛋白质相互作用预测性能。

    

    预测分子-蛋白质相互作用（MPI）是计算生物学中的一项基础任务，在药物发现和分子功能注释中具有至关重要的应用。然而，现有的MPI模型面临两大挑战。首先，带标签的分子-蛋白质对数据稀缺严重限制了模型性能，因为现有数据集仅捕获了生物相关相互作用的一小部分。其次，大多数方法仅依赖分子和蛋白质特征，忽略了更广泛的生物背景信息，如基因、代谢通路和功能注释，而这些信息本可以提供重要的补充。为了解决这些局限性，我们的框架首先聚合了多样化的生物数据集，包括分子、蛋白质、基因和通路水平的相互作用，然后开发了一种基于最优传输的方法，利用潜在的（原文摘要在此处截断）为未标记的分子-蛋白质对生成高质量伪标签。

    arXiv:2512.09365v2 Announce Type: replace  Abstract: Predicting molecule-protein interactions (MPIs) is a fundamental task in computational biology, with crucial applications in drug discovery and molecular function annotation. However, existing MPI models face two major challenges. First, the scarcity of labeled molecule-protein pairs significantly limits model performance, as available datasets capture only a small fraction of biological relevant interactions. Second, most methods rely solely on molecular and protein features, ignoring broader biological context such as genes, metabolic pathways, and functional annotations that could provide essential complementary information. To address these limitations, our framework first aggregates diverse biological datasets, including molecular, protein, genes and pathway-level interactions, and then develop an optimal transport-based approach to generate high-quality pseudo-labels for unlabeled molecule-protein pairs, leveraging the underlyi
    
[^375]: Magneton：通过差分能量调试优化机器学习系统的能效

    Magneton: Optimizing Energy Efficiency of ML Systems via Differential Energy Debugging

    [https://arxiv.org/abs/2512.08365](https://arxiv.org/abs/2512.08365)

    该论文提出“差分能量调试”新方法并实现了Magneton能量分析器，通过在算子级别比较相似机器学习系统之间的能量消耗，自动定位由不良软件设计造成的能量浪费，从而显著提升机器学习系统的能效。

    

    机器学习（ML）模型的训练和部署已经变得极其耗能。虽然现有的优化工作主要集中在硬件能效上，但一个重大却被忽视的低效来源是由不良软件设计导致的软件能量浪费。这通常包括冗余或设计不佳的操作，它们消耗更多能量却不能提升性能。这些低效问题广泛存在于常用的机器学习框架和应用中，然而开发者往往缺乏检测和诊断这些问题的可见性与工具。我们提出了差分能量调试这一新颖方法，它利用了这样一个观察结果：相互竞争的机器学习系统往往以截然不同的能量消耗实现类似的功能。基于这一洞察，我们设计并实现了Magneton——一个能量分析器，它在算子级别比较相似机器学习系统之间的能量消耗，并自动定位（原文在此处截断）导致能量浪费的根源。

    arXiv:2512.08365v2 Announce Type: replace-cross  Abstract: The training and deployment of machine learning (ML) models have become extremely energy-intensive. While existing optimization efforts focus primarily on hardware energy efficiency, a significant but overlooked source of inefficiency is software energy waste caused by poor software design. This often includes redundant or poorly designed operations that consume more energy without improving performance. These inefficiencies arise in widely used ML frameworks and applications, yet developers often lack the visibility and tools to detect and diagnose them.   We propose differential energy debugging, a novel approach that leverages the observation that competing ML systems often implement similar functionality with vastly different energy consumption. Building on this insight, we design and implement Magneton, an energy profiler that compares energy consumption between similar ML systems at the operator level and automatically pi
    
[^376]: 面向基于CLIP的类增量学习的层次语义树锚定方法

    Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning

    [https://arxiv.org/abs/2511.15633](https://arxiv.org/abs/2511.15633)

    提出HASTEN框架，通过层次语义树锚定细粒度类别的语义结构，从而缓解基于CLIP的类增量学习中的特征漂移与灾难性遗忘。

    

    类增量学习（CIL）使模型能够在持续学习新类别的同时保留已有知识。近来，以CLIP为代表的视觉-语言模型通过多模态预训练提供了可迁移的特征，使其非常适合类增量学习任务。然而，现实世界中的视觉与语言概念天然具有层次性：例如“狗”这一文本概念涵盖了“拉布拉多”和“金毛寻回犬”等细粒度类别，且每个类别都与相应的图像相关联。尽管如此，现有的基于CLIP的类增量学习方法未能显式地捕捉这种固有的层次结构，导致细粒度类别在增量更新过程中发生特征漂移，最终引发灾难性遗忘。为应对这一挑战，我们提出了HASTEN（层次语义树锚定），一个利用语义结构来稳定基于CLIP的类增量学习的层次感知框架。HASTEN不再将类别视为孤立的标签，而是利用……

    arXiv:2511.15633v2 Announce Type: replace-cross  Abstract: Class-Incremental Learning (CIL) enables models to learn new classes continually while preserving past knowledge. Recently, vision-language models like CLIP offer transferable features via multimodal pre-training, making them well-suited for CIL. However, real-world visual and linguistic concepts are inherently hierarchical: a textual concept like "dog" subsumes fine-grained categories such as "Labrador" and "Golden Retriever," and each category is associated with corresponding images. Nevertheless, existing CLIP-based CIL methods fail to explicitly capture this inherent hierarchy, leading to feature drift of fine-grained classes during incremental updates and ultimately to catastrophic forgetting. To address this challenge, we propose HASTEN (Hierarchical Semantic Tree Anchoring), a hierarchy-aware framework that uses semantic structure to stabilize CLIP-based CIL. Rather than treating classes as isolated labels, HASTEN levera
    
[^377]: Specter：通过量化使能的多步投机卸载实现快速MoE推理

    Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading

    [https://arxiv.org/abs/2511.14102](https://arxiv.org/abs/2511.14102)

    Specter通过量化使能的多步投机解码与专家卸载相结合，在单GPU上实现了平均1.50倍（最高1.88倍）的MoE推理加速。

    

    混合专家模型通常超出单个加速器的内存容量。将专家卸载到主机内存会造成严重的PCIe I/O瓶颈，因为数据依赖的路由使数据传输处于关键路径上。我们提出了Specter，一个利用量化将投机解码与专家卸载相结合的推理系统。其轻量级量化草稿模型在设备上高效运行，同时保持高保真度的专家激活模式。这为未来的专家使用提供了准确的多步前瞻，使运行时能够在验证之前提前组织专家的移动和执行，从而同时提高通信和执行效率。在单请求推理下，Specter在每输出令牌时间（TPOT）方面相比最先进的系统实现了平均1.50倍（最高1.88倍）的加速。Specter使内存受限的单GPU MoE推理在商用硬件上更加实用。

    arXiv:2511.14102v2 Announce Type: replace  Abstract: Mixture-of-experts (MoE) models often exceed a single accelerator's memory capacity. Offloading experts to host memory creates a severe PCIe I/O bottleneck because data-dependent routing keeps transfers on the critical path. We present Specter, an inference system that uses quantization to combine speculative decoding with expert offloading. Its lightweight quantized draft model runs efficiently on device while preserving high-fidelity expert activation patterns. This provides accurate multi-step lookahead over future expert usage, allowing the runtime to organize expert movement and execution ahead of verification and improve both communication and execution efficiency. Under single-request inference, Specter achieves an average 1.50x (up to 1.88x) speedup in time per output token (TPOT) over the state-of-the-art system. Specter makes memory-constrained, single-GPU MoE inference more practical on commodity hardware. Our code is avai
    
[^378]: 通过协同设计的脉冲神经网络在Intel Loihi 2上实现在线持续学习

    Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network

    [https://arxiv.org/abs/2511.01553](https://arxiv.org/abs/2511.01553)

    提出CLP-SNN，一种将持续学习原型（CLP）算法重构为事件驱动神经动力学的协同设计脉冲神经网络，并作为首个在线持续学习器部署于Intel Loihi 2芯片，在低功耗约束下于OpenLORIS上取得与Replay和SLDA相当的性能。

    

    边缘AI必须在严格的功耗预算内适应新类别和不断变化的数据，但在线持续学习（在下一个样本到来之前从每个样本中学习）恰恰是传统处理器效率最低的场景：每次小规模更新都需要以全功率支付完整的调度开销。我们提出了CLP-SNN，这是一种脉冲神经网络架构，它将持续学习原型算法重新构建为事件驱动的神经动力学：一个自归一化的局部规则取代了CLP的全局重新归一化，原型分配变为由新颖性触发的神经发生，正确性调节的可塑性变为奖励门控的每神经元元可塑性。作为该芯片上的首个在线持续学习器，CLP-SNN被部署在Intel Loihi 2上。在OpenLORIS数据集上，无需回放的CLP-SNN在1-shot设置下与Replay和SLDA性能相当；在25-shot设置下，它与Replay的差距在0.8个百分点以内，比基于协方差的SLDA低6.2个百分点。在提取的特征上，每个预测与更新步骤……（摘要截断）

    arXiv:2511.01553v3 Announce Type: replace-cross  Abstract: Edge AI must adapt to new classes and shifting data within a tight power budget, but online continual learning (learning from each sample before the next arrives) is where conventional processors are least efficient: each small update pays a full dispatch at full power. We present CLP-SNN, a spiking architecture that recasts the Continually Learning Prototypes (CLP) algorithm as event-driven neural dynamics: a self-normalizing local rule replaces CLP's global renormalization, prototype allocation becomes novelty-triggered neurogenesis, and correctness-modulated plasticity becomes reward-gated per-neuron metaplasticity. Deployed on Intel Loihi 2 as the first online continual learner on that chip, rehearsal-free CLP-SNN matches Replay and SLDA on OpenLORIS at 1-shot; at 25-shot, it stays within 0.8 percentage points of Replay and 6.2 points below covariance-based SLDA. Each prediction-and-update step on extracted features complet
    
[^379]: 生成模型的似然几何

    A Likely Geometry of Generative Models

    [https://arxiv.org/abs/2510.26266](https://arxiv.org/abs/2510.26266)

    提出了一种无需额外训练、兼容不同度量与概率分布的生成模型几何框架，通过约束在高密度数据区域的类测地线曲线和常微分方程组来刻画生成模型空间中的最短路径。

    

    生成模型的几何结构是插值、模型检查等诸多任务的基础。遗憾的是，在不模型或数据维度施加限制性假设的情况下，大多数生成模型缺乏一个基本的几何概念。本文构建了一种兼容不同度量和概率分布的通用几何结构，用于分析生成模型，且无需额外训练。我们考虑了类似于测地线的曲线，将其约束在适当的数据分布上，旨在定位生成模型所学习到的高密度区域。我们将其形式化为一种（伪）度量，并证明其对应于黎曼流形上的一个牛顿力学系统。我们证明，该框架中的最短路径可由一组常微分方程刻画，且在局部上对应于适当黎曼度量下的测地线。在数值方面，我们提出了一种新颖的高效算法。

    arXiv:2510.26266v3 Announce Type: replace  Abstract: The geometry of generative models serves as the basis for interpolation, model inspection, and more. Unfortunately, most generative models lack a principal notion of geometry without restrictive assumptions on either the model or the data dimension. In this paper, we construct a general geometry compatible with different metrics and probability distributions to analyze generative models that do not require additional training. We consider curves analogous to geodesics constrained to a suitable data distribution aimed at targeting high-density regions learned by generative models. We formulate this as a (pseudo)-metric and prove that this corresponds to a Newtonian system on a Riemannian manifold. We show that shortest paths in our framework can be characterized by a system of ordinary differential equations, which locally corresponds to geodesics under a suitable Riemannian metric. Numerically, we derive a novel algorithm to efficien
    
[^380]: 差分隐私作为额外收益：基于多天线基站的多址接入衰落信道上的联邦学习

    Differential Privacy as a Perk: Federated Learning over Multiple-Access Fading Channels with a Multi-Antenna Base Station

    [https://arxiv.org/abs/2510.23463](https://arxiv.org/abs/2510.23463)

    该论文研究了基于多天线基站的多址接入衰落信道上的空中联邦学习，巧妙地将信道噪声从性能损害转化为差分隐私保护的天然随机性来源，突破了现有工作在信道模型和损失函数假设上的限制，实现了隐私保护与训练性能的协同优化。

    

    联邦学习（FL）是一种分布式学习范式，它通过在训练过程中无需交换原始数据来保护隐私。在其典型的边缘部署实例中，无线传输由模拟空中计算（AirComp）实现，称为空中联邦学习（AirFL），其中固有的信道噪声扮演着一种独特的“亦敌亦友”的角色：一方面，它因带噪的全局聚合而降低训练性能；另一方面，它为隐私保护机制提供了天然的随机性来源，而这类隐私保护可以通过差分隐私（DP）进行形式化量化。然而，有效利用这种信道损伤仍然具有挑战性，因为现有工作大多在简单信道模型或受限损失函数类型的假设下，仅考虑单轮或非收敛隐私损失界限下的（本地）差分隐私增强。在本文中，我们研究了多址接入衰落信道上的空中联邦学习

    arXiv:2510.23463v4 Announce Type: replace  Abstract: Federated Learning (FL) is a distributed learning paradigm that preserves privacy by eliminating the need to exchange raw data during training. In its prototypical edge instantiation with underlying wireless transmissions enabled by analog over-the-air computing (AirComp), referred to as \emph{over-the-air FL (AirFL)}, the inherent channel noise plays a unique role of \emph{frenemy} in the sense that it degrades training due to noisy global aggregation while providing a natural source of randomness for privacy-preserving mechanisms, formally quantified by \emph{differential privacy (DP)}. It remains, nevertheless, challenging to effectively harness such channel impairments, as prior arts, under assumptions of either simple channel models or restricted types of loss functions, mostly considering (local) DP enhancement with a single-round or non-convergent bound on privacy loss. In this paper, we study AirFL over multiple-access fading
    
[^381]: OPLoRA：通过正交投影LoRA防止参数高效微调中的灾难性遗忘

    OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning

    [https://arxiv.org/abs/2510.13003](https://arxiv.org/abs/2510.13003)

    OPLoRA通过双侧正交投影将LoRA更新约束在冻结权重前k个奇异子空间的正交补空间内，以数学保证的方式精确保留预训练知识，从而防止微调过程中的灾难性遗忘。

    

    低秩适配能够对大语言模型进行高效微调，但当学习到的更新与编码关键预训练知识的主奇异方向发生干扰时，会产生灾难性遗忘问题。我们提出了正交投影LoRA（OPLoRA），这是一种具有理论基础的方法，通过双侧正交投影来防止这种干扰。通过对冻结权重进行奇异值分解（SVD），OPLoRA利用投影 P_L = I - U_k U_k^T 和 P_R = I - V_k V_k^T，将LoRA更新完全约束在前k个奇异子空间的正交补空间内。我们证明了这种构造能够精确保持前k个奇异三元组，为知识保留提供了数学保证。为了量化子空间干扰，我们引入了度量指标 ρ_k，用于衡量更新与主导方向的对齐程度。在常识推理、数学等任务上的大量实验……

    arXiv:2510.13003v2 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) enables efficient fine-tuning of large language models but suffers from catastrophic forgetting when learned updates interfere with the dominant singular directions that encode essential pre-trained knowledge. We propose Orthogonal Projection LoRA (OPLoRA), a theoretically grounded approach that prevents this interference through double-sided orthogonal projections. By decomposing frozen weights via SVD, OPLoRA constrains LoRA updates to lie entirely within the orthogonal complement of the top-$k$ singular subspace using projections $P_L = I - U_k U_k^\top$ and $P_R = I - V_k V_k^\top$. We prove that this construction exactly preserves the top-$k$ singular triples, providing mathematical guarantees for knowledge retention. To quantify subspace interference, we introduce $\rho_k$, a metric measuring update alignment with dominant directions. Extensive experiments across commonsense reasoning, mathematics, and 
    
[^382]: Wasserstein鲁棒分类误差：代理损失失配与分布攻击

    Wasserstein Robust Classification Error: Surrogate Loss Misalignment and Distributional Attacks

    [https://arxiv.org/abs/2510.10000](https://arxiv.org/abs/2510.10000)

    本文证明了仅当损失函数是0-1损失的正倍数加标签相关偏移时，WDRO风险的排序才与鲁棒准确率的排序一致（交叉熵等常用损失存在失配问题），并提出Wasserstein分布攻击（WDA）来直接评估鲁棒准确率。

    

    Wasserstein分布鲁棒优化（WDRO）提供了一个通过参考分布周围Wasserstein球内最坏情况期望来评估模型鲁棒性的框架。我们证明，按WDRO风险递增对假设排序等价于按鲁棒准确率递减对假设排序，当且仅当损失函数是0-1损失的正倍数加上一个标签相关的偏移量。对于包括交叉熵在内的一类损失函数，我们证明了一个鲁棒准确率完美的假设反而可能比鲁棒准确率接近于零的假设具有更高的WDRO风险。为了直接评估鲁棒准确率，我们开发了Wasserstein分布攻击（WDA），它在候选支撑点搜索之后增加一个权重生成阶段，以在Wasserstein约束下优化经验质量（概率质量）的分配，而不是将每个数据点的全部质量分配给单一返回的攻击点。

    arXiv:2510.10000v3 Announce Type: replace  Abstract: Wasserstein distributionally robust optimization (WDRO) provides a framework for evaluating model robustness through the worst-case expectation in the Wasserstein ball around a reference distribution. We prove that ordering hypotheses by increasing WDRO risk is equivalent to ordering them by decreasing robust accuracy if and only if the loss function is a positive multiple of the zero-one loss plus a label-dependent offset. For a class of loss functions including cross-entropy, we show that a hypothesis with perfect robust accuracy can conversely have a higher WDRO risk than another hypothesis with near zero robust accuracy. To evaluate robust accuracy directly, we develop Wasserstein distributional attacks (WDA), which add a weight generation stage after candidate support search to optimize the allocation of empirical mass under the Wasserstein constraint, rather than assigning each data point's full mass to one returned attack poin
    
[^383]: 时间序列的贝叶斯非参数动态聚类

    Bayesian Nonparametric Dynamical Clustering of Time Series

    [https://arxiv.org/abs/2510.06919](https://arxiv.org/abs/2510.06919)

    提出一种基于层次狄利克雷过程与高斯过程先验的贝叶斯非参数方法，通过在未知数量的线性动态机制间切换来建模时间序列簇的演化，以原则性方式避免簇的无序增殖，并可通过变分推断实现离线与在线的高效学习，在心电图分析中验证了其有效性。

    

    我们提出了一种方法，通过在未知数量的线性动态机制之间切换，来建模无界数量的时间序列簇的演化。我们开发了一种贝叶斯非参数方法，使用层次狄利克雷过程作为切换线性动态系统（Switching Linear Dynamical System）参数的先验，并使用高斯过程先验来建模每个簇内振幅和时间对齐的统计变化。通过对时间序列模式的演化进行建模，该方法以有原则的方式避免了簇的不必要增殖。我们通过为离线和在线场景制定变分下界来执行推断，从而通过优化实现高效学习。我们通过使用公开数据库进行的多项心电图分析案例研究，展示了该方法的通用性和有效性。

    arXiv:2510.06919v2 Announce Type: replace-cross  Abstract: We present a method that models the evolution of an unbounded number of time series clusters by switching among an unknown number of regimes with linear dynamics. We develop a Bayesian non-parametric approach using a hierarchical Dirichlet process as a prior on the parameters of a Switching Linear Dynamical System and a Gaussian process prior to model the statistical variations in amplitude and temporal alignment within each cluster. By modeling the evolution of time series patterns, the method avoids unnecessary proliferation of clusters in a principled manner. We perform inference by formulating a variational lower bound for off-line and on-line scenarios, enabling efficient learning through optimization. We illustrate the versatility and effectiveness of the approach through several case studies of electrocardiogram analysis using publicly available databases.
    
[^384]: 面向mRNA序列生成的课程增强生成流网络

    Curriculum-Augmented GFlowNets For mRNA Sequence Generation

    [https://arxiv.org/abs/2510.03811](https://arxiv.org/abs/2510.03811)

    该论文提出课程增强生成流网络，利用学习进度驱动的自适应长度课程策略，在新型密码子设计环境中高效生成编码目标蛋白的mRNA序列，实现更优的帕累托前沿覆盖与多样性，且收敛速度最高提升4倍。

    

    我们开发了一种从头合成编码目标蛋白的信使RNA（mRNA）序列的方法。这需要在由同义密码子序列定义的巨大组合空间上优化多个生物学目标，在这一场景下，强化学习（RL）方法容易遭遇模式崩塌，而生成流网络则面临长时程奖励稀疏的问题。为应对这些挑战，我们提出了课程增强生成流网络，这是一种通过由学习进度驱动的、基于序列长度的自适应课程进行训练的多目标生成流网络。简而言之，CAGFN通过从对应于更长目标蛋白的、难度逐步递增的任务中学习来提升能力。我们在CodonDesignEnv上对CAGFN进行了评估，这是一个通过动态动作掩码保持蛋白一致性的新型密码子级别环境。实验结果表明，CAGFN实现了具有竞争力或更优的帕累托前沿覆盖率与多样性，同时收敛速度最高可提升4倍。

    arXiv:2510.03811v2 Announce Type: replace  Abstract: We develop an approach for the de novo synthesis of messenger RNA (mRNA) sequences encoding target proteins. This requires optimizing multiple biological objectives over large combinatorial spaces defined by synonymous codon sequences, a setting where Reinforcement Learning (RL) methods suffer from mode collapse and Generative Flow Networks (GFlowNets) face sparse rewards over long horizons. To address these challenges, we propose Curriculum-Augmented GFlowNets (CAGFNs), a multi-objective GFlowNet trained with an adaptive length-based curriculum driven by learning progress. In a nutshell, CAGFNs learn from progressively harder tasks corresponding to longer target proteins. We evaluate CAGFNs on CodonDesignEnv, a novel codon-level environment that preserves protein identity via dynamic action masking. Our experiments show that CAGFNs achieve competitive or stronger Pareto-front coverage and diversity while converging up to 4 times fas
    
[^385]: 评估临床预测模型的效用：决策曲线分析与成本曲线的比较

    Evaluating utility of clinical prediction models: A comparison of decision curve analysis and cost curves

    [https://arxiv.org/abs/2509.24608](https://arxiv.org/abs/2509.24608)

    本文比较了临床预测模型评估中的决策曲线分析与成本曲线两种方法，证明决策曲线与Brier曲线密切相关且两者间的参考线可相互转换，并提出上包络决策曲线可作为模型决策曲线的有用比较基准。

    

    分类模型通常预测二元结果，并使用一个或多个决策阈值来产生分类结果或为临床决策提供信息。决策曲线和成本曲线是评估模型在不同决策阈值下净收益和损失的模型评价方法。本文对决策曲线和成本曲线进行了比较，证明了决策曲线与一种特定类型的成本曲线——Brier曲线密切相关。使用这两种方法的用户都在假设模型分数已经过校准的前提下设置分类阈值，并依据正确阳性分类与正确阴性分类的相对临床重要性进行权衡。给定的决策曲线都存在一条与之对应的Brier曲线，使得每种空间中常见的参考线可以被纳入另一种空间中（例如，“不治疗”决策曲线可以转换为等效的“不治疗”Brier曲线）。我们建议将上包络决策曲线作为与模型决策曲线进行比较的有用基准。

    arXiv:2509.24608v2 Announce Type: replace  Abstract: Classification models typically predict a binary outcome and use one or more decision thresholds to produce a classification or inform a clinical decision. Decision curves and cost curves are model evaluation approaches that assess net benefit and loss across decision thresholds. In this paper we compare decision curves and cost curves, demonstrating that decision curves are closely related to a specific type of cost curve called a Brier curve. Users of both approaches set the classification threshold assuming calibrated model scores, using the relative clinical importance of correct positive and negative classifications. A given decision curve has a corresponding Brier curve such that reference lines common in each space can be included in either (e.g. the 'treat none' decision curve can be translated into an equivalent 'treat none' Brier curve). We suggest the upper envelope decision curve as a useful comparison to a model decision
    
[^386]: 具有常数增益的平均奖励鲁棒马尔可夫决策过程的贝尔曼最优性

    Bellman Optimality of Average-Reward Robust Markov Decision Processes with a Constant Gain

    [https://arxiv.org/abs/2509.14203](https://arxiv.org/abs/2509.14203)

    本文通过研究常数增益设定下S-矩形不确定性对应的鲁棒贝尔曼方程解的存在性及其与最优平均奖励的关系，为平均奖励鲁棒马尔可夫决策过程建立了理论框架。

    

    鲁棒马尔可夫决策过程（MDP）下的学习与最优控制受到了越来越多的关注，然而现有的大多数理论、算法和应用都集中在有限时域或折扣模型上。长期平均奖励模型虽然在许多运筹学和管理学情境中非常自然，但相关研究仍然不足。这主要是因为其动态规划基础在技术上具有挑战性且仅被部分理解，若干基本问题仍悬而未决。本文通过分析常数增益设定，朝着建立平均奖励鲁棒MDP的一般框架迈出了一步。我们研究了在控制器与S-矩形对手之间可能存在信息不对称的平均奖励鲁棒控制问题。我们的分析以常数增益鲁棒贝尔曼方程为中心，考察了解的存在性及其与最优平均奖励之间的关系。

    arXiv:2509.14203v4 Announce Type: replace-cross  Abstract: Learning and optimal control under robust Markov decision processes (MDPs) have received increasing attention, yet most existing theory, algorithms, and applications focus on finite-horizon or discounted models. Long-run average-reward formulations, while natural in many operations research and management contexts, remain underexplored. This is primarily because the dynamic programming foundations are technically challenging and only partially understood, with several fundamental questions remaining open. This paper steps toward a general framework for average-reward robust MDPs by analyzing the constant-gain setting. We study the average-reward robust control problem with possible information asymmetries between the controller and an S-rectangular adversary. Our analysis centers on the constant-gain robust Bellman equation, examining both the existence of solutions and their relationship to the optimal average reward. Specific
    
[^387]: 深度ReLU神经网络中基于Stiefel流形的最优权重初始化方法

    Optimized Weight Initialization on the Stiefel Manifold for Deep ReLU Neural Networks

    [https://arxiv.org/abs/2509.00362](https://arxiv.org/abs/2509.00362)

    该论文通过在Stiefel流形上求解优化问题，提出了一种专为ReLU网络设计的半正交权重初始化方法，能够从初始化时刻起校准预激活统计特性、缓解死亡ReLU问题并稳定极深网络的训练。

    

    对于深度较大的ReLU网络而言，稳定且高效的训练对权重初始化高度敏感。不当的初始化会导致神经元永久性失活（即“死亡ReLU”问题），并随着网络深度的增加而加剧梯度不稳定性。He初始化、Xavier初始化和正交初始化等方法能够保持方差或实现近似等距性，但它们并不一定能调节预激活均值或控制激活稀疏性，而且在极深的网络架构中，这些方法的有效性往往会减弱。本工作通过在Stiefel流形上求解一个优化问题，提出了一种专门针对ReLU优化的半正交初始化方法，从而从一开始就保持尺度并对预激活统计特性进行校准。文中推导出了一族闭式解以及一种高效的采样方案。在初始化时刻的理论分析表明，所提出的初始化方法能够缓解死亡ReLU问题，减缓……（摘要原文在此处截断）

    arXiv:2509.00362v2 Announce Type: replace  Abstract: Stable and efficient training of ReLU networks with large depth is highly sensitive to weight initialization. Improper initialization can cause permanent neuron inactivation dying ReLU and exacerbate gradient instability as network depth increases. Methods such as He, Xavier, and orthogonal initialization preserve variance or promote approximate isometry. However, they do not necessarily regulate the pre-activation mean or control activation sparsity, and their effectiveness often diminishes in very deep architectures. This work introduces a semi-orthogonal initialization specifically optimized for ReLU by solving an optimization problem on the Stiefel manifold, thereby preserving scale and calibrating the pre-activation statistics from the outset. A family of closed-form solutions and an efficient sampling scheme are derived. Theoretical analysis at initialization shows that the proposed initialization mitigates dying ReLU, slows th
    
[^388]: 融合数据与物理的时空混沌动力学系统降阶建模方法

    Blending data and physics for reduced-order modeling of systems with spatiotemporal chaotic dynamics

    [https://arxiv.org/abs/2507.21299](https://arxiv.org/abs/2507.21299)

    该论文提出一种融合数据与物理知识的混合降阶模型，通过在自编码器发现的不变流形上投影并修正全阶模型的向量场（或将其作为贝叶斯先验），在数据充足、稀缺甚至物理模型参数有误的情形下均优于纯数据驱动方法。

    

    虽然数据驱动技术是混沌动力学系统降阶建模的强大工具，但在利用已知物理知识（即全阶模型FOM）来提升预测能力方面仍有巨大潜力。我们开发了一种由数据和全阶模型共同驱动的混合降阶模型（ROM），用于在不变流形上演化时空混沌动力学，该流形的坐标通过自编码器获得。该方法将全阶模型的向量场投影到不变流形上；随后，这个由物理推导的向量场要么利用动态数据进行修正，要么被用作通过数据更新的贝叶斯先验。在两种情形下均采用了神经常微分方程方法。我们使用了来自Kuramoto-Sivashinsky方程和复Ginzburg-Landau方程的模拟数据进行验证。相较于仅使用数据的方法，在数据丰富、数据稀缺、甚至全阶模型本身不正确（即参数值有误）的情况下，该方法均表现出优势……

    arXiv:2507.21299v2 Announce Type: replace  Abstract: While data-driven techniques are powerful tools for reduced-order modeling of systems with chaotic dynamics, great potential remains for leveraging known physics (i.e. a full-order model (FOM)) to improve predictive capability. We develop a hybrid reduced order model (ROM), informed by both data and FOM, for evolving spatiotemporal chaotic dynamics on an invariant manifold whose coordinates are found using an autoencoder. This approach projects the vector field of the FOM onto the invariant manifold; then, this physics-derived vector field is either corrected using dynamic data, or used as a Bayesian prior that is updated with data. In both cases, the neural ordinary differential equation approach is used. We consider simulated data from the Kuramoto-Sivashinsky and complex Ginzburg-Landau equations. Relative to the data-only approach, for scenarios of abundant data, scarce data, and even an incorrect FOM (i.e. erroneous parameter va
    
[^389]: GenoMAS：用于基因组数据科学发现的引导式多智能体编程

    GenoMAS: Guided Multi-Agent Programming for Scientific Discovery from Genomic Data

    [https://arxiv.org/abs/2507.21035](https://arxiv.org/abs/2507.21035)

    GenoMAS提出了一种引导式多智能体编程框架，通过将结构化工作流与自主决策相结合，让编程智能体动态选择推进、修改、绕过或回溯动作单元，实现了从复杂基因组数据中可靠地进行科学发现。

    

    基因表达分析是许多生物医学发现的关键所在，然而由于多个大型半结构化文件的复杂性以及对广泛领域专业知识的需求，从原始转录组数据中提取见解仍然极具挑战性。现有方法要么依赖于在边缘情况下失效的僵化工作流程，要么依赖于缺乏严谨科学探究所需精度的完全自主智能体。GenoMAS通过提出一个基于大语言模型的智能体团队来解决这一问题，该团队将结构化工作流程与自主适应性相融合，通过类型化消息传递协议协调多种专业化智能体。其核心是一个引导式规划框架，编程智能体将任务指南分解为动作单元，并动态选择推进、修改、绕过或回溯，在适应基因组数据特异性的同时保持逻辑连贯性。在GenoTEX基准测试中，GenoMAS达到了89.1

    arXiv:2507.21035v4 Announce Type: replace  Abstract: Gene expression analysis holds the key to many biomedical discoveries, yet extracting insights from raw transcriptomic data remains challenging due to the complexity of multiple large, semi-structured files and the need for extensive domain expertise. Current approaches either rely on inflexible workflows that fail in edge cases or fully autonomous agents that lack the precision required for rigorous scientific inquiry. GenoMAS addresses this by presenting a team of LLM-based agents that integrates structured workflows with autonomous adaptability, orchestrating diverse specialized agents through typed message-passing protocols. At its core lies a guided-planning framework where programming agents decompose task guidelines into Action Units and dynamically choose to advance, revise, bypass, or backtrack, maintaining logical coherence while adapting to the idiosyncrasies of genomic data. On the GenoTEX benchmark, GenoMAS achieves 89.1
    
[^390]: 逻辑回归上的梯度下降：大步长对球面数据有效吗？

    Gradient Descent on Logistic Regression: Do Large Step-Sizes Work with Data on the Sphere?

    [https://arxiv.org/abs/2507.11228](https://arxiv.org/abs/2507.11228)

    本文研究逻辑回归中梯度下降在低于稳定性阈值 $2/\lambda^*$ 的大步长下的收敛性，证明等范数数据在一维时可保证全局收敛，等范数数据在高维可能产生循环但球面均匀分布的总体损失可全局收敛，并将一维结论推广到更弱的环形条件。

    

    逻辑回归上的梯度下降具有许多经典分析未能完全刻画的迷人特性。对于线性可分数据，无论步长多大，迭代点在方向上都会收敛到最大间隔分离器。然而，对于线性不可分数据，即使步长低于局部稳定性阈值 $2/\lambda^*$（其中 $\lambda^*$ 是解处 Hessian 矩阵的最大特征值），也可能出现循环。我们探究要求所有样本具有相等范数是否是低于该阈值时保证全局收敛的充分条件。我们证明这一结论在一维情况下确实成立。在更高维度中，即使是等范数数据集也可能产生循环，尽管当样本均匀分布在球面上时，总体损失（population loss）的全局收敛仍然成立。最后，在一维情况下，我们将等范数假设放宽为一个环形（annulus）条件，该条件可保证低于稳定性阈值时的全局收敛。

    arXiv:2507.11228v2 Announce Type: replace  Abstract: Gradient descent on logistic regression has many fascinating properties not fully captured by classical analyses. For linearly separable data, the iterates converge in direction to the maximum-margin separator regardless of how large the step size is. For non-separable data, however, cycles may occur even for step sizes below the local stability threshold of $2/\lambda^*$, where $\lambda^*$ is the largest eigenvalue of the Hessian at the solution. We ask whether requiring all examples to have equal norm is a sufficient condition for global convergence below this threshold. We prove that this is indeed true in one dimension. In higher dimensions, even equal-norm datasets can produce cycles, although global convergence holds for the population loss when the examples are uniformly distributed on the sphere. Finally, in one dimension, we relax the equal-norm assumption to an annulus condition that ensures global convergence below the sta
    
[^391]: 双重扩散：在城市传感器网络概率预测中平衡速度、准确性与不确定性

    Double-Diffusion: Balancing Speed, Accuracy, and Uncertainty in Probabilistic Forecasting for Urban Sensor Networks

    [https://arxiv.org/abs/2506.23053](https://arxiv.org/abs/2506.23053)

    Double-Diffusion通过将闭式图热传导先验嵌入去噪扩散模型，使反向过程从先验附近热启动而非纯噪声出发，从而在城市传感器网络概率预测中实现了速度、准确性与不确定性的平衡。

    

    城市传感器网络需要这样的预测：既准确、又携带有用的不确定性信息，并且能在新读数到达时足够快速地更新以供决策行动。然而这些目标相互冲突：确定性模型无法给出分布，而扩散预测器虽能建模不确定性，却需要从纯噪声出发经过多步去噪。我们提出Double-Diffusion（双重扩散），它将一个闭式图热传导先验整合到去噪扩散模型中。该先验是传感器图上的一种无参数低通预测，承担双重角色：它既是模型生成的残差目标，又对去噪器进行条件化。因此，反向过程从先验附近开始，对一条较短的热启动链进行去噪，而非从纯噪声合成；其名称即表示这种组合——由图扩散馈入去噪扩散。一个紧凑的去噪器DD-Net按照去噪扩散概率模型（DDPM）在Resfusion热启动框架下训练，因此生成过……

    arXiv:2506.23053v4 Announce Type: replace  Abstract: Urban sensor networks need forecasts that are accurate, carry useful uncertainty, and refresh fast enough to act on as new readings arrive. These goals conflict: deterministic models give no distribution, while diffusion forecasters model uncertainty but denoise from pure noise over many steps. We present Double-Diffusion, which integrates a closed-form graph-heat prior into a denoising diffusion model. The prior is a parameter-free low-pass forecast over the sensor graph, and it serves two roles: it is the residual target the model generates, and it conditions the denoiser. The reverse process therefore starts near the prior and denoises a short warm-started chain instead of synthesizing from pure noise; the name denotes this composition, a graph diffusion feeding a denoising diffusion. A compact denoiser, DD-Net, is trained as a Denoising Diffusion Probabilistic Model (DDPM) in the Resfusion warm-start formulation, so generation re
    
[^392]: 大语言模型对简单代码变换十分脆弱：提出CETBench——一个代码等价性检测基准

    LLMs are Brittle to Simple Code Transformations: Introducing CETBench - A Benchmark for Code-Equivalence Checking

    [https://arxiv.org/abs/2506.04019](https://arxiv.org/abs/2506.04019)

    本文提出了代码等价性检测基准CETBench，发现大语言模型即使面对简单的代码变换也会出现显著性能下降（在跨语言场景下尤为严重），并提出一种基于微调的简单方法来提升模型性能。

    

    我们研究了大型语言模型（LLM）判断两个程序在功能上是否等价的能力。这是一个重要问题，因为对代码等价性进行基准测试有助于评估LLM在代码重写和代码翻译等任务上的能力。为此，我们提出了CETBench（Code Equivalence with Transformations Benchmark，基于变换的代码等价性检测基准），它由一个程序仓库构建而成，这些程序可能解决相同或不同的任务。每个数据集实例通过采样一对程序并应用随机序列的预定义代码变换来生成，从而得到等价或非等价的程序对。我们的分析表明，即使是简单的变换也会导致最先进的LLM在代码等价性检测任务上出现显著的性能下降。在跨语言场景下（即比较用不同语言编写的程序时），这些挑战会被进一步放大。为了解决这一问题，我们提出了一种简单的基于微调的方法来提升LLM在该任务上的性能。

    arXiv:2506.04019v2 Announce Type: replace-cross  Abstract: We study how well LLMs can determine whether two programs are functionally equivalent. This is an important problem because benchmarking code equivalence helps assess LLM capability in tasks such as code rewriting and translation. To this end, we introduce CETBench - Code Equivalence with Transformations Benchmark - built from a repository of programs that may solve the same or different tasks. Each dataset instance is created by sampling a program pair and applying a random sequence of predefined code transformations, yielding either equivalent or non-equivalent pairs. Our analysis shows that even simple transformations cause a significant performance drop in state-of-the-art LLMs on code-equivalence checking. These challenges are further amplified in the cross-lingual setting when comparing programs written in different languages. To remedy this, we present a simple fine-tuning-based approach to boost LLM performance on the t
    
[^393]: 重新思考知识图谱嵌入中的泛化：实证研究与稀疏峰值正则化

    Rethinking Generalization in Knowledge Graph Embedding: An Empirical Study and Sparse Peak Regularization

    [https://arxiv.org/abs/2505.23442](https://arxiv.org/abs/2505.23442)

    本文从泛化角度对知识图谱嵌入的正则化进行了跨模型族的系统性实证研究，发现过拟合在各类KGE模型中普遍存在，并提出稀疏峰值正则化方法来显著提升模型性能。

    

    知识图谱（KGs）为实体和关系提供了结构化表示，支撑着搜索、推荐和问答等数据管理与Web应用。由于Freebase和Wikidata等Web规模的知识图谱天然存在不完整性，知识图谱嵌入（KGE）将实体和关系映射到连续向量空间中，并被广泛用于通过链接预测来推断缺失的事实。尽管KGE模型设计已取得广泛进展，但正则化的作用尚未得到系统性考察，尤其是在张量分解模型之外的模型上。本文的主要贡献是从泛化的角度对KGE的正则化进行了全面的实证研究。在基于平移、基于张量分解、基于图神经网络（GNN）、时序以及多模态等各类KGE模型上，我们表明过拟合在各模型族中普遍存在，而适当的正则化可以显著提升（原文在此处截断）。

    arXiv:2505.23442v2 Announce Type: replace-cross  Abstract: Knowledge graphs (KGs) provide structured representations of entities and relations, and underpin data management and Web applications such as search, recommendation, and question answering. Since Web-scale KGs such as Freebase and Wikidata are inherently incomplete, knowledge graph embedding (KGE) maps entities and relations into continuous vector spaces and is widely used to infer missing facts through link prediction. Despite extensive progress in KGE model design, the role of regularization has been less systematically examined, especially beyond tensor-decomposition models. The primary contribution of this paper is a comprehensive empirical study of regularization for KGE from the perspective of generalization. Across translation-based, tensor-decomposition-based, GNN-based, temporal, and multimodal KGE models, we show that overfitting is common across model families and that appropriate regularization can substantially im
    
[^394]: 通过可微停止时间加速优化

    Accelerating Optimization via Differentiable Stopping Time

    [https://arxiv.org/abs/2505.22509](https://arxiv.org/abs/2505.22509)

    本文提出了一种基于微分方程理论的“可微停止时间”概念及其高效反向传播算法，首次使“最小化达到目标损失所需时间”这一传统上不可微的对偶优化问题变为可微表述，从而为加速优化算法开辟了新框架，并可应用于在线超参数调优等场景。

    

    优化是现代机器学习应用中的重要模块。人们已经付出了巨大努力来加速优化算法。一种常见的表述是在给定时间内达到更低的损失，这使得框架相对于算法超参数是可微的。相比之下，其对偶问题——最小化达到目标损失所需的时间——被认为是不可微的，因为时间本身不可微。因此，该对偶问题通常仅作为概念性框架存在，或使用零阶方法进行优化。为了解决这一局限性，我们提出了一个可微的停止时间，并基于微分方程从理论上对其进行了论证。同时，我们设计了一种高效的算法来通过它进行反向传播。由此，所提出的可微停止时间为加速算法提供了一种全新的可微表述。我们进一步讨论了其应用，例如在线超参数调优以及学习……

    arXiv:2505.22509v2 Announce Type: replace  Abstract: Optimization is an important module of modern machine learning applications. Tremendous efforts have been made to accelerate optimization algorithms. A common formulation is achieving a lower loss at a given time. This enables a differentiable framework with respect to the algorithm hyperparameters. In contrast, its dual, minimizing the time to reach a target loss, is believed to be non-differentiable, as the time is not differentiable. As a result, it usually serves as a conceptual framework or is optimized using zeroth-order methods. To address this limitation, we propose a differentiable stopping time and theoretically justify it based on differential equations. An efficient algorithm is designed to backpropagate through it. As a result, the proposed differentiable stopping time enables a new differentiable formulation for accelerating algorithms. We further discuss its applications, such as online hyperparameter tuning and learni
    
[^395]: 联邦学习：从理论到实践

    Federated Learning: From Theory to Practice

    [https://arxiv.org/abs/2505.19183](https://arxiv.org/abs/2505.19183)

    本书是一本联邦学习的实践入门指南，其核心贡献是提出通过联邦学习网络的加权边编码设备间相似性来实现个性化联邦学习，使每个设备在训练自己模型的同时仍能受益于与相关设备的协作。

    

    本书提供了构建和理解联邦学习（FL）系统的实践性介绍。联邦学习使多个设备（如智能手机、传感器或本地计算机）能够协同训练机器学习（ML）模型，同时保持数据私密并存储在本地。当由于隐私、法规或技术原因无法或不应该将数据集中化时，这是一种强大的解决方案。本书面向希望学习如何设计可扩展的、保护隐私的联邦学习系统的学生、工程师和研究人员。我们的主要关注点是个性化：使每个设备能够训练自己的模型，同时仍能从与相关设备的协作中受益。这是通过利用设备之间（与其关联的学习任务之间）的相似性来实现的，这些相似性由联邦学习（FL）网络的加权边（或链接）进行编码。其核心思想是将现实世界中的联邦学习系统表示为设备网络。

    arXiv:2505.19183v4 Announce Type: replace  Abstract: This book offers a hands-on introduction to building and understanding federated learning (FL) systems. FL enables multiple devices -- such as smartphones, sensors, or local computers -- to collaboratively train machine learning (ML) models, while keeping their data private and local. It is a powerful solution when data cannot or should not be centralized due to privacy, regulatory, or technical reasons. The book is designed for students, engineers, and researchers who want to learn how to design scalable, privacy preserving FL systems. Our main focus is on personalization: enabling each device to train its own model while still benefiting from collaboration with relevant devices. This is achieved by leveraging similarities between (the learning tasks associated with) devices that are encoded by the weighted edges (or links) of a federated learning network (FL network). The key idea is to represent real-world FL systems as networks o
    
[^396]: PALATE：全期望定律的巧妙应用，用于增强深度生成模型的评估

    PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models

    [https://arxiv.org/abs/2503.18462](https://arxiv.org/abs/2503.18462)

    提出PALATE框架，通过巧妙运用全期望定律来克服特征似然散度（FLD）的计算效率瓶颈，从而更高效地全面评估深度生成模型生成样本的保真度、多样性和新颖性。

    

    深度生成模型（DGMs）已在机器学习领域引发了范式转变，在图像合成、自然语言处理及其他相关领域取得了显著进展。然而，对这些模型进行全面评估，以有效捕捉生成样本中保真度、多样性和新颖性之间的相互作用，仍然是一个持续的挑战。最近提出的特征似然散度（Feature Likelihood Divergence, FLD）方法在这方面提供了一个有理论基础且实用的工具，但面临显著的计算局限性。在本文中，我们提出了PALATE，一个旨在通过解决FLD的效率约束来增强DGMs评估的新颖框架。我们的方法基于对表示可访问数据的随机变量的全期望定律的一种巧妙应用。当与MMD基线度量以及最近引入的DINOv3特征相结合时……（摘要在此处截断）

    arXiv:2503.18462v2 Announce Type: replace-cross  Abstract: Deep generative models (DGMs) have caused a paradigm shift in the field of machine learning, yielding noteworthy advancements in domains such as image synthesis, natural language processing, and other related areas. However, a comprehensive evaluation of these models that effectively captures the interplay among fidelity, diversity, and novelty in generated samples remains a persistent challenge. A recently proposed approach, the Feature Likelihood Divergence (FLD), offers a theoretically grounded and practical tool in this regard but faces notable computational limitations. In this paper, we propose PALATE, a novel framework designed to enhance the evaluation of DGMs by addressing the efficiency constraints of FLD. Our approach is based on a peculiar application of the law of total expectation to random variables representing accessible data. When integrated with the MMD baseline metric and the recently introduced DINOv3 featu
    
[^397]: 剖析模型误设在数据驱动优化中的影响

    Dissecting the Impact of Model Misspecification in Data-driven Optimization

    [https://arxiv.org/abs/2503.00626](https://arxiv.org/abs/2503.00626)

    本文从模型误设程度的角度比较了数据驱动优化中传统“先拟合后优化”方法与“估计-优化一体化”方法，证明了当模型存在误设时，一体化方法在遗憾值的前两个主导项上具有“普遍双重收益”的统计优势。

    

    数据驱动优化旨在通过基于估计成本的决策优化，将机器学习模型转化为实际决策。这类流程可以通过拟合一个分布模型，然后将其代入目标优化问题来完成。虽然这种拟合可以利用极大似然等传统方法，但一种更新的方法采用估计-优化一体化策略，即最小化决策误差而非估计误差。尽管后者直观易懂，其统计上的优势尚未得到充分理解，而这对于指导机器学习的规范性（prescriptive）应用至关重要。本文从模型误设程度的角度，剖析了这两类方法之间的性能比较。特别地，我们证明了当底层模型存在误设时，一体化方法在遗憾值（regret）的前两个主导项上提供了“普遍的双重收益”，而传统方法……（摘要原文此处截断）

    arXiv:2503.00626v3 Announce Type: replace  Abstract: Data-driven optimization aims to translate a machine learning model into decision-making by optimizing decisions on estimated costs. Such a pipeline can be conducted by fitting a distributional model which is then plugged into the target optimization problem. While this fitting can utilize traditional methods such as maximum likelihood, a more recent approach uses estimation-optimization integration that minimizes decision error instead of estimation error. Although intuitive, the statistical benefit of the latter approach is not well understood yet is important to guide the prescriptive usage of machine learning. In this paper, we dissect the performance comparisons between these approaches in terms of the amount of model misspecification. In particular, we show how the integrated approach offers a ``universal double benefit'' on the top two dominating terms of regret when the underlying model is misspecified, while the traditional 
    
[^398]: 面向可认证少样本迁移学习的权重空间学习

    Weight Space Learning for Certifiable Few-shot Transfer Learning

    [https://arxiv.org/abs/2502.06970](https://arxiv.org/abs/2502.06970)

    该论文提出了一种新型迁移学习方法，通过在上游任务上训练PEFT参数的扩散模型分布，再利用采样-评估程序为下游任务选择最优参数，从而即使在少样本情况下也能获得可认证的非空洞学习理论泛化保证。

    

    在当代深度学习中，解决低数据问题的一个普遍且有效的工作流程是通过参数高效微调（PEFT）将强大的预训练基础模型（FM）适配到新任务上。然而，尽管这些方法在经验上有效，所得到的解决方案却缺乏能够认证其准确性的泛化保证——而在高重要性应用部署之前，出于伦理或法律方面的原因，可能需要这种保证。在本文中，我们开发了一种新颖的迁移学习方法，旨在即使在少样本情况下，也能为下游任务提供非空洞的学习理论泛化保证。具体而言，我们首先利用上游任务训练一个PEFT参数的分布。然后，我们通过一个“采样-评估”程序来学习下游任务——从训练好的扩散模型中采样合理的PEFT参数，并选择在下游数据上似然最高的那个。至关重要的是，这种配置（原文在此处截断）

    arXiv:2502.06970v3 Announce Type: replace  Abstract: In contemporary deep learning, a prevalent and effective workflow for solving low-data problems is adapting powerful pre-trained foundation models (FMs) to new tasks via parameter-efficient fine-tuning (PEFT). However, while empirically effective, the resulting solutions lack generalisation guarantees to certify their accuracy - which may be required for ethical or legal reasons prior to deployment in high-importance applications. In this paper we develop a novel transfer learning approach that is designed to facilitate non-vacuous learning theoretic generalisation guarantees for downstream tasks, even in the low-shot regime. Specifically, we first use upstream tasks to train a distribution over PEFT parameters. We then learn the downstream task by a sample-and-evaluate procedure -- sampling plausible PEFTs from the trained diffusion model and selecting the one with the highest likelihood on the downstream data. Crucially, this confi
    
[^399]: 潜因子模型的特征选择

    Feature Selection for Latent Factor Models

    [https://arxiv.org/abs/2412.10128](https://arxiv.org/abs/2412.10128)

    本文提出了一种基于低秩生成类模型和信噪比（SNR）准则的按类别分别进行特征选择的新方法，该方法具有理论保证的特征恢复能力，在标准分类数据集上优于部分现有方法，并可轻松应用于类增量学习场景。

    

    特征选择对于在高维数据集中精确定位相关特征、缓解“维度灾难”以及提升机器学习性能至关重要。传统的分类特征选择方法使用来自所有类别的数据为每个类别选择特征。本文探讨了为每个类别分别选择特征的特征选择方法，该方法使用基于低秩生成方法的类别模型，并引入了信噪比（SNR）特征选择准则。这种新颖的方法在特定假设下为特征恢复提供了理论上的可靠保证，并已被证明在标准分类数据集上优于一些现有的特征选择方法。此外，该方法可以轻松应用于类增量学习设置，其性能与一些现有方法相当。

    arXiv:2412.10128v3 Announce Type: replace  Abstract: Feature selection is crucial for pinpointing relevant features in high-dimensional datasets, mitigating the 'curse of dimensionality,' and enhancing machine learning performance. Traditional feature selection methods for classification use data from all classes to select features for each class. This paper explores feature selection methods that select features for each class separately, using class models based on low-rank generative methods and introducing a signal-to-noise ratio (SNR) feature selection criterion. This novel approach offers a theoretically sound guarantee of feature recovery under specific assumptions and has been demonstrated to outperform some existing feature selection methods on standard classification datasets. Additionally, it can be easily applied in a class-incremental learning setup, yielding performance comparable to that of some current methods.
    
[^400]: FiTv2：面向扩散模型的可扩展且改进的灵活视觉Transformer

    FiTv2: Scalable and Improved Flexible Vision Transformer for Diffusion Model

    [https://arxiv.org/abs/2410.13925](https://arxiv.org/abs/2410.13925)

    提出了FiTv2架构，通过将图像视为动态大小的token序列而非固定分辨率网格，使扩散模型能够生成无限制分辨率和宽高比的图像。

    

    “自然是无限无分辨率的”。在这一现实背景下，现有的扩散模型（如Diffusion Transformers）在处理超出其训练域的图像分辨率时常常面临挑战。为了解决这一局限性，我们将图像概念化为具有动态大小的token序列，而不是传统上将图像视为固定分辨率网格的方法。这一视角使得灵活的训练策略能够无缝适应训练和推理过程中的各种宽高比，从而促进分辨率泛化并消除图像裁剪引入的偏差。在此基础上，我们提出了灵活视觉Transformer（FiT），这是一种专门设计用于生成具有无限制分辨率和宽高比图像的Transformer架构。我们进一步将FiT升级为FiTv2，引入了多项创新设计，包括Query-Key向量归一化等。

    arXiv:2410.13925v2 Announce Type: replace  Abstract: \textit{Nature is infinitely resolution-free}. In the context of this reality, existing diffusion models, such as Diffusion Transformers, often face challenges when processing image resolutions outside of their trained domain. To address this limitation, we conceptualize images as sequences of tokens with dynamic sizes, rather than traditional methods that perceive images as fixed-resolution grids. This perspective enables a flexible training strategy that seamlessly accommodates various aspect ratios during both training and inference, thus promoting resolution generalization and eliminating biases introduced by image cropping. On this basis, we present the \textbf{Flexible Vision Transformer} (FiT), a transformer architecture specifically designed for generating images with \textit{unrestricted resolutions and aspect ratios}. We further upgrade the FiT to FiTv2 with several innovative designs, includingthe Query-Key vector normaliz
    
[^401]: 帮我帮你：迁移学习中源数据与目标数据的聚合价值

    Help Me Help You: The Aggregate Value of Source and Target Data in Transfer Learning

    [https://arxiv.org/abs/2408.16189](https://arxiv.org/abs/2408.16189)

    该论文揭示了监督迁移学习中的两种机制二分现象：弱迁移机制下最优收敛率仅由源数据或目标数据之一决定，而强迁移机制下需联合利用源与目标数据的互补信息才能实现快得多的目标收敛率。

    

    迁移学习（无论是回归还是分类任务）所关注的问题是：如何利用来自源分布 P 的不完美数据来改进针对目标分布 Q 的预测。尽管目前对这一问题在无监督设置（即没有带标签目标数据可用）下已有大量理论认识，但带有标签目标数据的监督设置仍然研究不足。我们识别并刻画了监督设置下可实现的收敛率中一个有趣的二分现象：存在一个弱迁移机制，其中最优收敛率由源数据集或目标数据集中某一方的信息决定，而非两者共同决定；还存在一个强迁移机制，在该机制下可以实现快得多的目标收敛率，这需要同时考虑源数据和目标数据中的互补信息。此外，虽然先前针对无监督设置的理论工具可以扩展以刻画弱迁移机制，但我们认为它们无法捕捉（原文摘要在此处截断）

    arXiv:2408.16189v4 Announce Type: replace-cross  Abstract: Transfer learning, in regression or classification, concerns problems where one aims to use imperfect data from a source distribution P to improve prediction w.r.t. a target distribution Q. While there is now much theoretical understanding of the unsupervised setting of this problem (where no labeled target data is available), the supervised setting with labeled target data remains under-studied.   We identify and characterize an interesting dichotomy in the rates achievable in the supervised setting: there is a weak transfer regime where optimal rates are determined by information in either the source or target dataset but not both, and a strong transfer regime where much faster target rates are achievable, accounting for complementary information in source and target data. Furthermore, while previous theoretical tools for the unsupervised setting can be extended to capture the weak transfer regime, we argue that they cannot c
    
[^402]: 基于鼾声的阻塞性睡眠呼吸暂停患者多级别气道塌陷部位定位的深度学习方法

    Deep learning-based multi-level airway collapse site localization on obstructive sleep apnea patients based on snoring sounds

    [https://arxiv.org/abs/2408.16030](https://arxiv.org/abs/2408.16030)

    本研究利用音频频谱图Transformer（AST）模型，仅通过鼾声即可有效定位阻塞性睡眠呼吸暂停患者软腭、口咽及腭后区水平的上气道塌陷部位，为无创诊断提供了新途径。

    

    目的：本研究评估了深度学习模型，特别是ResNet-50和音频频谱图Transformer（AST），利用鼾声识别阻塞性睡眠呼吸暂停（OSA）患者的多级别上气道塌陷。方法：使用2020年至2021年间37名接受药物诱导睡眠内镜检查（DISE）受试者的鼾声记录来训练模型，完成两项多标签任务：识别软腭（V）、口咽（O）、舌根（T）和会厌（E）水平的阻塞，以及腭后区（RP）与舌后区（RG）水平的阻塞。结果：AST模型显著优于ResNet-50（经Bonferroni校正后P<0.05），能够有效识别V、O和RP水平的阻塞（F1分数：0.59-0.87，马修斯相关系数：0.50-0.80，AUC：0.87-0.97）。然而，由于样本量较小，即使使用了合成少数类过采样技术，两种模型在T、E和RG水平的性能仍然有限。

    arXiv:2408.16030v3 Announce Type: replace-cross  Abstract: Objective This study evaluated deep learning models, specifically ResNet-50 and Audio Spectrogram Transformer (AST), to identify multi-level upper airway collapses in obstructive sleep apnea (OSA) patients using snoring sounds. Methods Snoring recordings from 37 subjects undergoing drug-induced sleep endoscopy (DISE) between 2020 and 2021 were used to train models for two multi-label tasks: identifying obstructions at the Velum (V), Oropharynx (O), Tongue-base (T), and Epiglottis (E) levels, and the retropalatal (RP) vs. retroglossal (RG) levels. Results The AST model significantly outperformed ResNet-50 (P <0.05 after Bonferroni correction), effectively identifying V-, O-, and RP-level obstructions (F1-score: 0.59-0.87, Matthews Correlation Coefficient: 0.50-0.80, and AUC: 0.87-0.97). However, the performance of both models at the T, E, and RG levels remained limited due to small sample sizes, even with the Synthetic Minority 
    
[^403]: 物理启发的神经网络实现高效图着色

    Physics Inspired Neural Networks Enable Efficient Graph Coloring

    [https://arxiv.org/abs/2408.01503](https://arxiv.org/abs/2408.01503)

    该研究提出一种结合图神经网络与统计力学原理的物理启发神经框架，通过监督信号、对称性破缺正则化和噪声退火神经动力学，实现对大规模图着色问题的高效求解。

    

    处于算法相变附近的组合优化问题对经典算法和机器学习方法都构成了根本性挑战。其中，图着色是一个典型的约束满足问题，其特征在于相变会将解空间分割成多个不同的簇，使得寻找有效解变得极具挑战性。本文提出了一种物理启发的神经框架，通过将图神经网络与统计力学原理相结合来学习求解大规模图着色实例。该方法整合了基于种植的监督信号、对称性破缺正则化以及迭代噪声退火的神经动力学，以在复杂的解空间景观中导航。当迭代次数与图规模呈二次方缩放时，学习到的求解器能够成功处理日益困难的实例，即使是在寻找有效解变得极为困难的区域中依然有效。

    arXiv:2408.01503v3 Announce Type: replace  Abstract: Combinatorial optimization problems near algorithmic phase transitions represent a fundamental challenge for both classical algorithms and machine learning approaches. Among them, graph coloring is a prototypical constraint satisfaction problem characterized by phase transitions that fragment the solution space into distinct clusters, making the search for valid solutions challenging. Here we introduce a physics-inspired neural framework that learns to solve large-scale graph coloring instances by combining graph neural networks with statistical-mechanics principles. Our approach integrates a planting-based supervised signal, symmetry-breaking regularization, and iterative noise-annealed neural dynamics to navigate complex solution landscapes. When the number of iterations scales quadratically with graph size, the learned solver successfully tackles increasingly challenging instances, even in regimes where finding valid solutions bec
    
[^404]: 面向多目标优化的雅可比下降法

    Jacobian Descent for Multi-Objective Optimization

    [https://arxiv.org/abs/2406.16232](https://arxiv.org/abs/2406.16232)

    该论文提出雅可比下降（JD）算法，通过梯度投影完全解决多目标间的冲突，并证明了在光滑凸情形下可收敛到帕累托前沿，进而实现了将每个训练样本损失视为独立目标的新型学习范式——实例级风险最小化（IWRM）。

    

    许多优化问题需要平衡多个相互冲突的目标。为了推广仅限于单目标优化的梯度下降法，我们提出了雅可比下降（Jacobian Descent, JD）及其随机变体。该算法利用向量值目标函数的雅可比矩阵迭代更新参数，其中每一行是单个目标的梯度。虽然文献中已有若干组合梯度的方法，但当目标之间存在冲突时，这些方法通常难以奏效。相比之下，我们提出通过投影梯度来完全解决冲突。我们证明了在光滑凸情形下，该方法能够收敛到帕累托前沿，这是比通常的收敛到弱帕累托平稳点更强的保证。我们的方法还支持实例级风险最小化（IWRM），这是一种新颖的学习范式，将每个训练样本的损失视为一个独立的目标。

    arXiv:2406.16232v4 Announce Type: replace-cross  Abstract: Many optimization problems require balancing multiple conflicting objectives. To generalize gradient descent, which is limited to single-objective optimization, we introduce Jacobian descent (JD) and its stochastic variants. This algorithm iteratively updates parameters using the Jacobian matrix of a vector-valued objective function, in which each row is the gradient of an individual objective. While several methods to combine gradients already exist in the literature, they generally struggle when the objectives conflict. In contrast, we propose projecting gradients to fully resolve conflict. We prove convergence to the Pareto front in the smooth convex case with this approach, a stronger guarantee than the usual convergence to a weakly Pareto stationary point. Our method also enables instance-wise risk minimization (IWRM), a novel learning paradigm in which the loss of each training example is considered a separate objective. 
    
[^405]: 快与慢的模拟：面向黑盒优化的策略学习

    Simulating, Fast and Slow: Learning Policies for Black-Box Optimization

    [https://arxiv.org/abs/2406.04261](https://arxiv.org/abs/2406.04261)

    本文提出一种新方法，通过学习一个策略来引导在各种输入分布下的黑盒优化求解，避免了每次从头开始的昂贵模拟优化，从而大幅提升实验设计中优化的效率。

    

    模拟器在科学和工程中至关重要，因为它们忠实地建模了设计参数对真实世界观测的影响。优化设计参数以最小化期望的目标函数是出了名的困难，因为模拟器通常是不可微分的黑盒。此外，每次模拟都需要大量的计算时间，因此天真地进行此类优化的代价可能高得令人望而却步。而且在实验设计的场景中，从业者通常需要在进行真实世界实验之前，针对各种不同的输入分布求解同一个优化问题。如果前向模拟模型的评估成本很高，每次都从头开始进行这些相关优化就会非常低效。为了应对这些挑战，本文提出了一种新颖的方法，通过学习一个策略来引导……（摘要内容在此处被截断）……

    arXiv:2406.04261v2 Announce Type: replace  Abstract: Simulators are vital in science and engineering, as they faithfully model the influence of design parameters on real-world observations. Optimizing design parameters to minimize a desired objective function is notoriously challenging since simulators are often non-differentiable black boxes. Moreover, each simulation incurs significant compute time, thus doing such optimizations naively may be prohibitively expensive. Furthermore, in experiment design settings, practitioners are typically required to solve the same optimization problem for various input distributions before proceeding to real-world experiments. Starting these related optimizations from scratch each time can be inefficient if the forward simulation model is expensive to evaluate. To address these challenges, this paper introduces a novel method for solving black-box optimization problems on various input distributions by learning a policy that guides the training of a
    
[^406]: Gibson Env：面向具身智能体的真实世界感知

    Gibson Env: Real-World Perception for Embodied Agents

    [https://arxiv.org/abs/1808.10654](https://arxiv.org/abs/1808.10654)

    论文提出了基于真实空间虚拟化构建的Gibson仿真环境（涵盖572栋建筑、1400多个楼层空间），用于训练具身智能体的感知能力，并通过"Goggles"合成机制使仿真中训练的模型能够迁移部署到真实世界。

    

    为主动智能体开发视觉感知模型和感觉运动控制在物理世界中十分繁琐，因为现有算法速度太慢、难以高效地进行实时学习，而且机器人既脆弱又昂贵。这促成了“在仿真中学习”的兴起，但随之而来的问题是：仿真中习得的结果能否迁移到真实世界。在本文中，我们关注为主动智能体开发真实世界感知的问题，为此提出了Gibson虚拟环境，并展示了在其中学习到的若干示例感知任务。Gibson基于对真实空间的虚拟化，而非使用人工设计的空间，目前包含来自572栋完整建筑物的1400多个楼层空间。Gibson的主要特征是：I. 来源于真实世界并反映其语义复杂性；II. 具有一个内部合成机制"Goggles"，使得训练好的模型能够部署到真实世界中。

    arXiv:1808.10654v2 Announce Type: replace  Abstract: Developing visual perception models for active agents and sensorimotor control are cumbersome to be done in the physical world, as existing algorithms are too slow to efficiently learn in real-time and robots are fragile and costly. This has given rise to learning-in-simulation which consequently casts a question on whether the results transfer to real-world. In this paper, we are concerned with the problem of developing real-world perception for active agents, propose Gibson Virtual Environment for this purpose, and showcase sample perceptual tasks learned therein. Gibson is based on virtualizing real spaces, rather than using artificially designed ones, and currently includes over 1400 floor spaces from 572 full buildings. The main characteristics of Gibson are: I. being from the real-world and reflecting its semantic complexity, II. having an internal synthesis mechanism, "Goggles", enabling deploying the trained models in real-wo
    
[^407]: Taskonomy：解耦任务迁移学习

    Taskonomy: Disentangling Task Transfer Learning

    [https://arxiv.org/abs/1804.08328](https://arxiv.org/abs/1804.08328)

    该论文提出一种完全计算化的方法，通过发现26个视觉任务之间的迁移学习依赖关系，构建了首个任务迁移学习的计算分类图谱（Taskonomy），揭示了视觉任务空间的内在结构。

    

    视觉任务之间是否存在关联，还是彼此独立？例如，拥有表面法线信息是否可以简化图像的深度估计？直觉对这些问题的回答是肯定的，这意味着视觉任务之间存在某种结构。了解这种结构具有重要价值：它是迁移学习的核心概念，并为识别任务间的冗余提供了有原则的方法，例如在相关任务之间无缝复用监督信息，或在单个系统中解决多个任务而不会累积复杂性。我们提出了一种完全计算化的方法来建模视觉任务空间的结构，这是通过在潜空间中寻找26个2D、2.5D、3D和语义任务之间的（一阶及高阶）迁移学习依赖关系来实现的。最终产物是一张用于任务迁移学习的计算分类图谱。我们研究了这种结构所带来的影响，例如非平凡的涌现现象。

    arXiv:1804.08328v2 Announce Type: replace-cross  Abstract: Do visual tasks have a relationship, or are they unrelated? For instance, could having surface normals simplify estimating the depth of an image? Intuition answers these questions positively, implying existence of a structure among visual tasks. Knowing this structure has notable values; it is the concept underlying transfer learning and provides a principled way for identifying redundancies across tasks, e.g., to seamlessly reuse supervision among related tasks or solve many tasks in one system without piling up the complexity.   We proposes a fully computational approach for modeling the structure of space of visual tasks. This is done via finding (first and higher-order) transfer learning dependencies across a dictionary of twenty six 2D, 2.5D, 3D, and semantic tasks in a latent space. The product is a computational taxonomic map for task transfer learning. We study the consequences of this structure, e.g. nontrivial emerged
    
[^408]: 对AI分类器的对抗鲁棒性度量的存在性，唯一性和可扩展性研究

    On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers. (arXiv:2310.14421v1 [stat.ML])

    [http://arxiv.org/abs/2310.14421](http://arxiv.org/abs/2310.14421)

    本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。

    

    本文提出并证明了针对（局部）唯一可逆分类器、广义线性模型（GLM）和熵AI（EAI）具有最小对抗路径（MAP）和最小对抗距离（MAD）的存在性、唯一性和明确的分析计算的简单可验证的数学条件。在常见的合成基准测试数据集上，针对神经网络、提升随机森林、GLM和EAI等各类AI工具进行MAP和MAD的实际计算、比较和解释，包括双卷状螺旋线及其扩展以及两个生物医学数据问题（用于健康保险理赔预测和心脏病发作致死率分类）。在生物医学应用中，展示了MAP如何在预定义的可访问控制变量子集中提供唯一的最小患者特定风险缓解干预措施。

    Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.
    

